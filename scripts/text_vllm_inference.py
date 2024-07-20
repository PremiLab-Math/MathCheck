import torch
import sys
import json
import tqdm
import os
import argparse
sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/")
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from scripts.utils.extract_ans import get_checklist
from scripts.utils.prompt_template import task2prompt, read_task_prompt, task2prompt_base, task2prompt_chat
from vllm import LLM, SamplingParams

task_type_list = ["solving","outcome_judging","process_judging","answerable_judging"]
question_type_list = ["seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"]

def get_model(args):

    model_name_or_path = args.model_name_or_path

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    model = LLM(model_name_or_path, tokenizer=model_name_or_path, tensor_parallel_size=args.tp_size, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    
    return model, tokenizer

def vllm_inference(model: LLM, batched_queries, args):
    
    responses = model.generate(
        batched_queries,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        )
    )
    return responses
    
    
def construct_query(system_prompt, user_prompt, is_base_model):
    
    if not is_base_model:
        return system_prompt + ' ' + user_prompt
    else:
        return user_prompt


def inference(args, input_file: str="", output_file: str="", model: str = "", model_name_or_path: str = "", task_prompt: str = "fewshot", check_task: str = "all", check_question: str = "all"):
    
    if output_file == "":
        data = input_file.split("/")[-1].split(".")[0]
        output_file = f"results/{data}_{model}_task_{check_task}_question_{check_question}_{task_prompt}_prediction.json"
    # output_file_jsonl = output_file.replace(".json",".jsonl")
    

    task_prompt = read_task_prompt(task_prompt)
    questions, solutions, answers, task_types, question_types, _  = get_checklist(input_file)
    print(f"data size: {len(questions)}")
    output_data = []
    
    # jsonl_file = open(output_file_jsonl, 'a', encoding='utf-8')
    
    model, tokenizer = get_model(args)

    all_inputs = []
    
    print(f"Is chat model: {args.is_chat_model}")
    
    for idx, (question, solution, answer, task_type, question_type) in enumerate(tqdm.tqdm(zip(questions, solutions, answers, task_types, question_types))):
        if idx < len(output_data):
            print("CONTINUE")
            continue
        item = {"question": question, "solution": solution, "answer": answer, "task_type": task_type, "question_type": question_type}
        if task_type == check_task or task_type in task_type_list:
            
            if args.is_chat_model:
                system_prompt, user_prompt, _, nshot = task2prompt_chat(task_type, question=question, solution=solution, task_prompt=task_prompt, solving_fewshot=args.solving_fewshot)
            else:
                system_prompt, user_prompt, _ = task2prompt_base(task_type, question=question, solution=solution, task_prompt=task_prompt, solving_fewshot=args.solving_fewshot)
                
            query = construct_query(system_prompt, user_prompt, is_base_model=not args.is_chat_model)
            
            if args.is_chat_model:
                
                messages = []
                
                if nshot is not None:
                    for shot in nshot:
                        messages.extend(
                            [
                                {"role": "user", "content": shot["question"]},
                                {"role": "assistant", "content": shot["answer"]},
                             ]
                            )
                messages.append(
                    {"role": "user", "content": user_prompt},
                )
                prompt = tokenizer.apply_chat_template(messages, tokenize = False)
            else:
                # prompt = tokenizer.bos_token + query.replace("<eos>", tokenizer.eos_token)
                prompt = tokenizer.bos_token + query.replace("<eos>", "")
                
            # breakpoint()
            item["prompt"] = prompt
            
            all_inputs.append(item)
            
    for i in range(0, len(all_inputs), args.batch_size):
        
        batched_queries = [item["prompt"] for item in all_inputs[i:i+args.batch_size]]
        batched_responses = vllm_inference(model, batched_queries, args)
        
        item = all_inputs[i:i+args.batch_size]
           
        assert len(batched_responses) == len(item)
        
        for j in range(len(batched_responses)):
            item[j]["model_prediction"] = batched_responses[j].outputs[0].text
            # jsonl_file.write(json.dumps(item[j], ensure_ascii=False) + "\n")
        
        # jsonl_file.flush()
        output_data.extend(item)
    
    print("len(output_data)",len(output_data))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #model 
    parser.add_argument("--model", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    parser.add_argument("--model_name_or_path", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    
    # input output
    parser.add_argument("--input_file", default="", type=str, required=False, help="")
    parser.add_argument("--output_file", default="", type=str, required=False, help="")
    parser.add_argument("--attack_type", default=None, type=list, required=False, help="")
    parser.add_argument("--check_task", default="all", type=str, required=False, choices=["all","solving","outcome_judging","process_judging","answerable_judging"])
    parser.add_argument("--check_question", default="all", type=str, required=False, choices=["all","seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"])
    parser.add_argument("--task_prompt", default="zeroshot", type=str, required=False)
    
    # vllm params
    parser.add_argument("--batch_size", default=64, type=int, required=False, help="")
    parser.add_argument("--tp_size", default=1, type=int, required=False, help="")
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float, required=False, help="")
    parser.add_argument("--temperature", default=0.0, type=float, required=False, help="")
    parser.add_argument("--max_tokens", default=2048, type=int, required=False, help="")
    parser.add_argument("--top_p", default=1.0, type=float, required=False, help="")
    
    parser.add_argument("--is_chat_model", action = "store_true", help="is chat model or base model")
    
    parser.add_argument("--solving_fewshot", action = "store_true", help="Whether use fewshots on solving tasks")


    args = parser.parse_args()

    inference(
              args=args,
              model=args.model,
              model_name_or_path=args.model_name_or_path,
              input_file=args.input_file,
              output_file=args.output_file,
              task_prompt=args.task_prompt,
              check_task=args.check_task,
              check_question=args.check_question)
    
# General Model: fewshot
# Deepseek Model: fewshot_deepseek