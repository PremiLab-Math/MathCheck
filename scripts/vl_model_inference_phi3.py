# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/") # Set all the path as "MathCheck"
from scripts.utils.extract_ans import invoke_openai, get_checklist
from scripts.utils.prompt_template import task2prompt, read_task_prompt
import json
import argparse
import tqdm
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from PIL import Image 
import torch

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

task_type_list = ["solving","outcome_judging","process_judging","answerable_judging"]
question_type_list = ["seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"]

vl_models  = ['internvl-1.5', 'phi-3', 'cogvlm-2']


phi3_path = '!!! Change to your model path !!!'
if phi3_path == '!!! Change to your model path !!!':
    raise ValueError("Please change the path of 'phi3_path' to your local model weights forler.")


def inference_vllm(input_file="", output_file="", model="null", check_task="all", check_question="all", task_prompt="zeroshot"):
    if model not in vl_models:
        raise ValueError("model should be in vl_models: ", vl_models)

        model_id = phi3_path
        mllm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        print(f'template_type: Phi-3')

    seed_everything(42)

    # Do inference
    count = 0 # NOTE test

    if output_file == "":
        data = input_file.split("/")[-1].split(".")[0]
        output_file = f"results/{data}_{model}_task_{check_task}_question_{check_question}_{task_prompt}_prediction.json"
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        output_data = []
    task_prompt = read_task_prompt(task_prompt)
    questions, solutions, answers, task_types, question_types, img_paths  = get_checklist(input_file)
    print(f"data size: {len(questions)}")

    try:
        for idx, (question, solution, answer, task_type, question_type, img_path) in enumerate(tqdm.tqdm(zip(questions, solutions, answers, task_types, question_types, img_paths))):
            if idx < len(output_data):
                print("CONTINUE")
                continue
            item = {"question": question, "solution": solution, "answer": answer, "task_type": task_type, "question_type": question_type, 'image': img_path}



            if task_type == check_task or task_type in task_type_list:
                system_prompt, user_prompt, base64_image = task2prompt(task_type,question=question,task_prompt=task_prompt,solution=solution,img_path=img_path,encode_img_base64=False)
    
                image = Image.open(base64_image)
                query = user_prompt
                messages = [{"role": "user", "content": "<|image_1|>\n"+query}]
                prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 
                generation_args = { "max_new_tokens": 1024, "temperature": 0.0, "do_sample": False,} 
                generate_ids = mllm_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                prediction = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
                # print(response)

                print("idx,len(output_data),task_type,messages,prediction: ",idx,len(output_data),task_type,query,prediction)
                item["model_prediction"] = prediction
                # item["check_task"] = check_task
                output_data.append(item)

            # count += 1 # NOTE test
            # if count > 50:
            #     raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("KeyboardInterrupt, save the current output data to: ", output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("len(output_data)",len(output_data))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="phi-3", type=str, required=True, help="")
    parser.add_argument("--input_file", default="", type=str, required=False, help="")
    parser.add_argument("--output_file", default="", type=str, required=False, help="")
    parser.add_argument("--attack_type", default=None, type=list, required=False, help="")
    parser.add_argument("--check_task", default="all", type=str, required=False, choices=["all","solving","outcome_judging","process_judging","answerable_judging"])
    parser.add_argument("--check_question", default="all", type=str, required=False, choices=["all","seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"])
    parser.add_argument("--task_prompt", default="zeroshot", type=str, required=False)

    args = parser.parse_args()

    inference_vllm(model=args.model_name,
              input_file=args.input_file,
              output_file=args.output_file,
              check_task=args.check_task,
              check_question=args.check_question,
              task_prompt=args.task_prompt)


# python scripts/vl_model_inference_phi3.py --input_file geo_checklist.json --model_name phi-3  --check_task all