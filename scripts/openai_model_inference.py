# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/") # Set all the path as "MathCheck"
from scripts.utils.extract_ans import invoke_openai, get_checklist, batch_inference_api
from scripts.utils.prompt_template import task2prompt, read_task_prompt
import json
import argparse
import tqdm

task_type_list = ["solving","outcome_judging","process_judging","answerable_judging"]
question_type_list = ["seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"]
separator = "\n" + "_" * 60 + "\n"

def inference(input_file="", output_file="", model="", check_task="all", check_question="all", task_prompt="fewshot"):
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
    all_messages = []
    all_data = []
    try:
        for idx, (question, solution, answer, task_type, question_type, img_path) in enumerate(tqdm.tqdm(zip(questions, solutions, answers, task_types, question_types, img_paths))):
            if idx < len(output_data):
                print("CONTINUE")
                continue
            # prediction = ""
            # while prediction == "":
            item = {"question": question, "solution": solution, "answer": answer, "task_type": task_type, "question_type": question_type, 'image': img_path}
            if task_type == check_task or task_type in task_type_list:
                system_prompt, user_prompt, base64_image = task2prompt(task_type,question=question,task_prompt=task_prompt,solution=solution,img_path=img_path,encode_img_base64=True)
                
                if base64_image is None:    # text-only, single modal
                    messages = [
                        # {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                else:                       # text + image, multi-modal
                    messages=[
                        {"role": "system", "content": [{'type': 'text', 'text': system_prompt}]},
                        {"role": "user", "content": [{'type': 'text', 'text': user_prompt}, {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}]},
                    ]

                all_messages.append(messages)
                all_data.append(item)

                # prediction = invoke_openai(messages=messages, model=model)
                # print(task_type,idx,messages,separator,prediction,separator)
                # # if prediction == "":
                # #     raise Exception("Empty prediction encountered, triggering data save and exit.")
                # if prediction == "":
                #     print("Empty prediction encountered, retrying...")
                # else:
                #     item["model_prediction"] = prediction
                #     output_data.append(item)

        all_predictions = batch_inference_api(all_messages, model)
        for prediction, data_item in zip(all_predictions, all_data):
            if prediction:
                data_item["model_prediction"] = prediction
                output_data.append(data_item)
            else:
                print("data_item: ",data_item)

        
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt, save the current output data to: ", output_file)
    except Exception as e:
        print(str(e))
    finally:
        print("Len of output_data: ",len(output_data))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    parser.add_argument("--input_file", default="", type=str, required=False, help="")
    parser.add_argument("--output_file", default="", type=str, required=False, help="")
    parser.add_argument("--check_task", default="all", type=str, required=False, choices=["all","solving","outcome_judging","process_judging","answerable_judging"])
    parser.add_argument("--check_question", default="all", type=str, required=False, choices=["all","seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"])
    parser.add_argument("--task_prompt", default="zeroshot", type=str, required=False)

    args = parser.parse_args()

    inference(model=args.model_name,
              input_file=args.input_file,
              output_file=args.output_file,
              check_task=args.check_task,
              check_question=args.check_question,
              task_prompt=args.task_prompt)

# Usage:
# python3 scripts/openai_model_inference.py --input_file Geo_checklist.json --model_name gpt-4o-2024-05-13 --check_task all --check_question all --task_prompt zeroshot > log/gsm_checklist_gpt-3.5-turbo_all_task_question_zeroshot.out