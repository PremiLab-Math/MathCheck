import json
import argparse
import numpy as np
# import pdb
import sys
import os
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib as mpl
# mpl.rcParams['font.family'] = 'Tahoma'

sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/") # Set all the path as "MathCheck"
from scripts.utils.extract_ans import extract_gold_ans, extract_pred_ans, extract_outcome_correctness, extract_process_correctness, extract_answerable, delete_extra_zero

heatmap_name_map = {
    "gpt-4o": "GPT-4o", "gpt-4-turbo-2024-04-09": "GPT-4-Turbo-20240409", "gpt-4-vision-preview": "GPT-4-Vision-Preview", "gemini-1.5-pro": "Gemini-1.5-Pro", "gpt-3.5-turbo": "GPT-3.5-Turbo",
    "gemini-1.5-flash": "Gemini-1.5-Flash", "claude-3-opus-20240229": "Claude-3-Opus-20240229", "claude-3-sonnet-20240229": "Claude-3-Sonnet-20240229",
    "claude-3-haiku-20240307": "Claude-3-Haiku-20240307", "internvl-1.5": "InternVL-1.5-Chat", "phi-3": "Phi-3-Vision-128k-Instruct",
    "cogvlm-2": "CogVLM2-Llama3-Chat-19B", "Meta-Llama-3-70B-Instruct": "Llama-3-70B-Instruct", "deepseek-chat": "DeepSeek V2", "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B-Instruct",
    "Mixtral-8x7B": "Mixtral-8x7B-Base", "Qwen1.5-72B-Chat": "Qwen1.5-72B-Chat", "phi-3-mini-4k-instruct": "Phi-3-Mini-4K-Instruct", "Phi-3-medium-4k-instruct": "Phi-3-Medium-4K-Instruct",
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct", "chatglm3-6b": "ChatGLM3-6B", "deepseek-math-7b-rl": "DeepSeek-Math-7B-RL", "deepseek-math-7b-instruct": "DeepSeek-Math-7B-Instruct",
    "deepseek-math-7b-base": "DeepSeek-Math-7B-Base", "MetaMath-70B-V1.0": "MetaMath-Llama2-70B"
}

task_type_list = ["solving","answerable_judging","outcome_judging","process_judging"]
question_type_list = ["seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"]

heatmap_task = ["Seed\nQuestion", "Problem\nUnderstanding", "Distractor\nInsertion", "Scenario\nUnderstanding"]
heatmap_type = ["Problem\nSolving", "Outcome\nJudging", "Process\n Judging", "Answerable\nJudging"]

def draw_heat_map(problem_type, model_name, task_prompt, npdata):
    if 'gsm' in problem_type.lower(): # gsm is blue
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "deepskyblue"])
    elif 'geo' in problem_type.lower(): # geo is red
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "indianred"]) # indianred, lightcoral, salmon

    npdata = np.round(npdata * 100, 1)

    plt.imshow(npdata, cmap=cmap, interpolation='nearest', vmin=0, vmax=100)
    plt.xticks(range(len(heatmap_type)), heatmap_type, fontsize=12)
    plt.yticks(range(len(heatmap_task)), heatmap_task, fontsize=12)
    
    plt.colorbar()

    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            color = 'white' if npdata[i, j] > 50 else 'black'
            plt.text(j, i, npdata[i, j], ha='center', va='center', color=color,fontsize=12)

    plt.xticks(np.arange(npdata.shape[1]))
    plt.yticks(np.arange(npdata.shape[0]))
    plt.title(heatmap_name_map[model_name])
    plt.tight_layout()

    # save the heatmap
    plt.savefig('scripts/result_matrix/'+ problem_type + '_' + model_name + '_' + task_prompt + '.png')


def get_answer(check_task, golden_answer, model_prediction):
    golds_str = []
    preds_str = []
    if check_task == "solving":
        golds_str = delete_extra_zero(golden_answer)
        preds_str = extract_pred_ans(model_prediction)
    elif check_task == "outcome_judging":
        golds_str = golden_answer
        preds_str = extract_outcome_correctness(model_prediction)
    elif check_task == "process_judging":
        golds_str = golden_answer
        preds_str = extract_process_correctness(model_prediction)
    elif check_task == "answerable_judging":
        golds_str = golden_answer
        preds_str = extract_answerable(model_prediction)
    return golds_str, preds_str


def get_accuracy(pred_file,check_task):
    accuracy_dict = {}
    golds = {}
    preds = {}
    if check_task == "all":
        for task in task_type_list:
            accuracy_dict[task] = {}
            golds = {task_type: {question_type: [] for question_type in question_type_list} for task_type in task_type_list}
            preds = {task_type: {question_type: [] for question_type in question_type_list} for task_type in task_type_list}
    else:
        golds[check_task] = {question_type: [] for question_type in question_type_list}
        preds[check_task] = {question_type: [] for question_type in question_type_list}
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        print(len(pred_data))

        for item_idx, item in enumerate(pred_data):
            gold_a, pred_a = get_answer(item["task_type"], item["answer"], item["model_prediction"])
            print(f"item['task_type']: {item['task_type']}, gold_a: {gold_a}, pred_a: {pred_a}")
            golds[item["task_type"]][item["question_type"]].append(gold_a)
            preds[item["task_type"]][item["question_type"]].append(pred_a)

    #为了测试四种任务、四种问题类型的准确率
    # task_all_num = {"solving":0,"outcome_judging":0,"process_judging":0,"answerable_judging":0}
    # task_correct_num = {"solving": 0, "outcome_judging": 0, "process_judging": 0, "answerable_judging": 0}
    # question_all_num = {"seed_question":0,"problem_understanding_question":0,
    #                     "distractor_insertion_question":0,"scenario_understanding":0}
    # question_correct_num = {"seed_question":0,"problem_understanding_question":0,
    #                     "distractor_insertion_question":0,"scenario_understanding":0}
    # all_num = 0
    # all_correct = 0
    all_acc = 0

    # 请帮我计算每个task的准确率
    for task_type in golds.keys():
        for question_type in golds[task_type].keys():
            assert len(golds[task_type][question_type]) == len(preds[task_type][question_type])
            correct = 0
            None_num = 0
            # print(f"task: {task}, data size: {len(golds[task])}")
            if task_type == "outcome_judging":
                label_dic = {'Incorrect':0,'Correct':1,None:2}
                None_num = preds[task_type][question_type].count(None)
                pred_list = [label_dic[x] for x in preds[task_type][question_type]]
                gold_list = [label_dic[x] for x in golds[task_type][question_type]]
                # pred_list = preds[task_type][question_type]
                labels = ['Incorrect','Correct']
                # f1 = f1_score(golds[task_type][question_type],pred_list, labels=labels, pos_label='Incorrect')
                f1 = f1_score(gold_list,pred_list,labels=[0,1],average='macro')
                print(task_type, question_type, len(golds[task_type][question_type]), 'None: '+str(None_num))
                accuracy_dict[task_type][question_type] = f1
                all_acc += f1
            elif task_type == "answerable_judging":
                label_dic = {'Unanswerable':0,'Answerable':1,None:2}
                None_num = preds[task_type][question_type].count(None)
                pred_list = [label_dic[x] for x in preds[task_type][question_type]]
                gold_list = [label_dic[x] for x in golds[task_type][question_type]]
                # pred_list = preds[task_type][question_type]
                labels = ['Unanswerable','Answerable']
                # f1 = f1_score(golds[task_type][question_type],pred_list, labels=labels, pos_label='Incorrect')
                f1 = f1_score(gold_list,pred_list,labels=[0,1],average='macro')
                print(task_type, question_type, len(golds[task_type][question_type]), 'None: '+str(None_num))
                accuracy_dict[task_type][question_type] = f1
                all_acc += f1
            else:
                for gold, pred in zip(golds[task_type][question_type], preds[task_type][question_type]):
                    # task_all_num[task_type] += 1
                    # question_all_num[question_type] += 1
                    # all_num += 1
                    # print(f"gold: {gold}, pred: {pred}")
                    if gold == pred:
                        # all_correct += 1
                        correct += 1
                        # task_correct_num[task_type] += 1
                        # question_correct_num[question_type] += 1
                    if pred == None:
                        None_num += 1
                print(task_type, question_type, len(golds[task_type][question_type]), 'None: '+str(None_num))
                accuracy_dict[task_type][question_type] = correct / len(golds[task_type][question_type])
                all_acc += correct / len(golds[task_type][question_type])
    # print('-'*100)
    # print("All Acc: " + str(all_acc/16))
    # for each in task_all_num.keys():
    #     print(each + ":" + str(task_correct_num[each]/task_all_num[each]))
    # print('-'*10)
    # for each in question_all_num.keys():
    #     print(each + ":" + str(question_correct_num[each]/question_all_num[each]))
    # print('-' * 100)

    return accuracy_dict, all_acc/16


# 写一个函数，如果用户输入的check_task是all，那么对task_type_list的每个task调用get_accuracy函数，返回一个字典，key是task，value是对应task的准确率。如果用户输入的check_task不是all，那么只对用户输入的task调用get_accuracy函数，返回一个字典，key是task，value是对应task的准确率。
def evaluate_accuracy(eval_data,model_name,check_task,check_question,task_prompt):
    accuracy_dict = {}
    pred_file = os.path.join(os.getcwd(), "results", f"{eval_data}_{model_name}_task_{check_task}_question_{check_question}_{task_prompt}_prediction.json")
    if check_task not in task_type_list and check_task != "all":
        print("Invalid task")
    else:
        accuracy_dict, all_acc = get_accuracy(pred_file,check_task)
    return accuracy_dict, all_acc

# 用一个pandas的dataframe来展示task_type和question_type的准确率，index是task_type，columns是question_type
def show_accuracy_checklist(accuracy_dict, all_acc):
    df = pd.DataFrame(accuracy_dict)

    save_dict = {}
    save_dict['All Acc'] = all_acc
    for each in df.mean(axis=0).keys():
        save_dict[each] = df.mean(axis=0)[each]
    for each in df.mean(axis=1).keys():
        save_dict[each] = df.mean(axis=1)[each]
    save_dict['Matrix'] = df.to_numpy().tolist()


    print('-'*100)
    print("All Acc: " + str(all_acc))
    print('-' * 100)
    print(df.mean(axis=0))
    print('-'*100)
    print(df.mean(axis=1))
    
    np_array = df.to_numpy() # np array, 4*4


    print('-'*100)
    print(df)

    # draw heat map on np_array
    # draw_heat_map(args.eval_data, args.model_name, args.task_prompt, np_array)

    # np.save('scripts/result_matrix/'+ args.eval_data + '_' + args.model_name + '_' + args.task_prompt + '.npy', np_array)
    matrix_name = args.eval_data + '_' + args.model_name + '_' + args.task_prompt
    # save save_dict to json file
    # check whether the current model has been evaluated, all results save in result_matrix.json
    if not os.path.exists('scripts/result_matrix.json'):
        # create result_matrix.json
        with open('scripts/result_matrix.json', 'w') as f:
            json.dump({matrix_name: save_dict}, f, indent=4)
    else:
        with open('scripts/result_matrix.json', 'r') as f: # a dict
            result_matrix = json.load(f)
        result_matrix[matrix_name] = save_dict
        # save the result_matrix
        with open('scripts/result_matrix.json', 'w') as f:
            json.dump(result_matrix, f, indent=4)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default="gsm_checklist", type=str, required=True, help="")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    # parser.add_argument("--input_file", default="", type=str, required=False, help="")
    # parser.add_argument("--output_file", default="", type=str, required=True, help="")
    parser.add_argument("--check_task", default="all", type=str, required=False, choices=["all","solving","outcome_judging","process_judging","answerable_judging"])
    parser.add_argument("--check_question", default="all", type=str, required=False, choices=["all","seed_question","problem_understanding_question","distractor_insertion_question","scenario_understanding"])
    parser.add_argument("--task_prompt", default="zeroshot", type=str, required=False)

    args = parser.parse_args()
    accuracy_dict, all_acc = evaluate_accuracy(args.eval_data,args.model_name,args.check_task,args.check_question,args.task_prompt)
    
    show_accuracy_checklist(accuracy_dict, all_acc)
    # python scripts/results_evaluate.py --model_name gpt-3.5-turbo --eval_data gsm_checklist --task_prompt zeroshot


