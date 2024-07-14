# -*- coding: utf-8 -*-
import sys
import os
import json
import base64
sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/") # Set all the path as "MathCheck"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def read_task_prompt(prompt_type):
    with open('scripts/utils/task_prompt_'+prompt_type+'.json', 'r', encoding='utf-8') as prompt_file:
        task_prompt = json.load(prompt_file)
    return task_prompt

with open('scripts/utils/task_prompt_fewshot.json', 'r', encoding='utf-8') as file:
    task_prompt = json.load(file)

with open('scripts/utils/task_prompt_zeroshot.json', 'r', encoding='utf-8') as file:
    task_prompt_zeroshot_geometry = json.load(file)

def task2prompt(task, question, task_prompt, solution = None, judgement = None, few_shot = True, img_path=None, encode_img_base64=False):
    if "nshot" not in task_prompt[task].keys():
        few_shot = False
    if img_path is not None and encode_img_base64 == True:
        base64_image = encode_image(img_path) # encode img for gpt inference
    elif img_path is not None and encode_img_base64 == False:
        base64_image = img_path
    else:
        base64_image = None
 
    if img_path is None: # text-only, single modal, original task_prompt
        if task == "solving":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nAnswer: " + shot["Answer"] + "\n\n"
                user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nSolution: " + shot["Answer"]  + "\nJudgement: " + shot["Judgement"] + "\n\n"
                user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nJudgement: " + shot["Judgement"] + "\n\n"
                user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nJudgement: "


    else:               # text + image, multi-modal, use task_prompt_zeroshot json file
        if task == "solving":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nJudgement: "

    return system_prompt,user_prompt,base64_image


def task2prompt_base(task, question, task_prompt, solution = None, judgement = None, few_shot = True, img_path=None, encode_img_base64=False, solving_fewshot = True):
    if "nshot" not in task_prompt[task].keys():
        few_shot = False
    if img_path is not None and encode_img_base64 == True:
        base64_image = encode_image(img_path) # encode img for gpt inference
    elif img_path is not None and encode_img_base64 == False:
        base64_image = img_path
    else:
        base64_image = None
 
    if img_path is None: # text-only, single modal, original task_prompt
        if task == "solving":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True and solving_fewshot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nAnswer: " + shot["Answer"] + "\n\n"
                # user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
                user_prompt = nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
                nshot = ""
                
            user_prompt += "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nSolution: " + shot["Answer"]  + "\nJudgement: " + shot["Judgement"] + "\n\n"
                # user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
                user_prompt = nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                nshot = ""
                for shot in task_prompt[task]["nshot"]:
                    nshot += "Question: " + shot["Question"] + "\nJudgement: " + shot["Judgement"] + "\n\n"
                # user_prompt = task_prompt[task]["user-prompt"] +"\n\n" +nshot
                user_prompt = nshot
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "Question: " + question + "\nJudgement: "


    else:               # text + image, multi-modal, use task_prompt_zeroshot json file
        if task == "solving":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "Question: " + question + "\nJudgement: "

    return system_prompt,user_prompt,base64_image

def task2prompt_chat(task, question, task_prompt, solution = None, judgement = None, few_shot = True, img_path=None, encode_img_base64=False, solving_fewshot = True):
    
    if "nshot" not in task_prompt[task].keys():
        few_shot = False
    if img_path is not None and encode_img_base64 == True:
        base64_image = encode_image(img_path) # encode img for gpt inference
    elif img_path is not None and encode_img_base64 == False:
        base64_image = img_path
    else:
        base64_image = None
    
    if few_shot == True:
        nshot = []
    else:
        nshot = None
    
    if img_path is None: # text-only, single modal, original task_prompt
        if task == "solving":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True and solving_fewshot == True:
                for shot in task_prompt[task]["nshot"]:
                    # nshot += "Question: " + shot["Question"] + "\nAnswer: " + shot["Answer"] + "\n\n"
                    nshot.append({"question": "Question: " + shot["Question"] + "Answer: ", "answer": shot["Answer"]})
                user_prompt = task_prompt[task]["user-prompt"]
            else:
                user_prompt = task_prompt[task]["user-prompt"]
                nshot = None
            user_prompt += "\n\n" + "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                for shot in task_prompt[task]["nshot"]:
                    # nshot += "Question: " + shot["Question"] + "\nSolution: " + shot["Answer"]  + "\nJudgement: " + shot["Judgement"] + "\n\n"
                    nshot.append({"question": "Question: " + shot["Question"] + "\nSolution: " + shot["Answer"] + "\nJudgement: ", "answer": shot["Judgement"]})
                user_prompt = task_prompt[task]["user-prompt"]
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt[task]["system-prompt"]
            if few_shot == True:
                for shot in task_prompt[task]["nshot"]:
                    # nshot += "Question: " + shot["Question"] + "\nJudgement: " + shot["Judgement"] + "\n\n"
                    nshot.append({"question": "Question: " + shot["Question"] + "\nJudgement: ", "answer": shot["Judgement"]})
                user_prompt = task_prompt[task]["user-prompt"]
            else:
                user_prompt = task_prompt[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nJudgement: "


    else:               # text + image, multi-modal, use task_prompt_zeroshot json file
        if task == "solving":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nAnswer: "

        if task == "outcome_judging" or  task == "process_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nSolution: " + solution + "\nJudgement: "

        if task == "answerable_judging":
            system_prompt = task_prompt_zeroshot_geometry[task]["system-prompt"]
            user_prompt = task_prompt_zeroshot_geometry[task]["user-prompt"]
            user_prompt += "\n\n" + "Question: " + question + "\nJudgement: "

    return system_prompt, user_prompt, base64_image, nshot