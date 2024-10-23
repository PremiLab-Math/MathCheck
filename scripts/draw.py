import json
import argparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib as mpl
from tqdm import tqdm

heatmap_name_map = {
    "gpt-4o": "GPT-4o", "gpt-4-turbo-2024-04-09": "GPT-4-Turbo-20240409", "gpt-4-vision-preview": "GPT-4-Vision-Preview", "gemini-1.5-pro": "Gemini-1.5-Pro", "gpt-3.5-turbo": "GPT-3.5-Turbo",
    "gemini-1.5-flash": "Gemini-1.5-Flash", "claude-3-opus-20240229": "Claude-3-Opus-20240229", "claude-3-sonnet-20240229": "Claude-3-Sonnet-20240229",
    "claude-3-haiku-20240307": "Claude-3-Haiku-20240307", "internvl-1.5": "InternVL-1.5-Chat", "phi-3": "Phi-3-Vision-128k-Instruct",
    "cogvlm-2": "CogVLM2-Llama3-Chat-19B", "Meta-Llama-3-70B-Instruct": "Llama-3-70B-Instruct", "deepseek-chat": "DeepSeek V2", "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B-Instruct",
    "Mixtral-8x7B": "Mixtral-8x7B-Base", "Qwen1.5-72B-Chat": "Qwen1.5-72B-Chat", "phi-3-mini-4k-instruct": "Phi-3-Mini-4K-Instruct", "Phi-3-medium-4k-instruct": "Phi-3-Medium-4K-Instruct",
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct", "chatglm3-6b": "ChatGLM3-6B", "deepseek-math-7b-rl": "DeepSeek-Math-7B-RL", "deepseek-math-7b-instruct": "DeepSeek-Math-7B-Instruct",
    "deepseek-math-7b-base": "DeepSeek-Math-7B-Base", "MetaMath-70B-V1.0": "MetaMath-Llama2-70B", "gemini-1.5-pro-latest": "Gemini-1.5-Pro",'Llama-2-70b-hf':"Llama-2-70b-base"
}

heatmap_task = ["Original\nProblem", "Problem\nUnderstanding", "Irrelevant\nDisturbance", "Scenario\nUnderstanding"]
heatmap_type = ["Problem\nSolving", "Answerable\nJudging", "Outcome\nJudging", "Process\n Judging"]

def draw_heat_map(problem_type, model_name, task_prompt, npdata):
    if 'gsm' in problem_type.lower(): # gsm is blue
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "deepskyblue" , "#1d4e89"])
    elif 'geo' in problem_type.lower(): # geo is red
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "#DD4045", "darkred"]) # indianred, lightcoral, salmon

    npdata = np.round(npdata * 100, 1)

    plt.imshow(npdata, cmap=cmap, interpolation='nearest', vmin=0, vmax=100)
    plt.xticks(range(len(heatmap_type)), heatmap_type, fontsize=12)
    plt.yticks(range(len(heatmap_task)), heatmap_task, fontsize=12)
    
    plt.colorbar()

    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            color = 'white' if npdata[i, j] > 50 else 'black'
            plt.text(j, i, npdata[i, j], ha='center', va='center', color=color,fontsize=18)

    plt.xticks(np.arange(npdata.shape[1]))
    plt.yticks(np.arange(npdata.shape[0]))
    plt.title(heatmap_name_map[model_name], fontsize=16)
    plt.tight_layout()

    # save the heatmap
    plt.savefig('./result_heatmap/'+ problem_type + '_' + model_name + '_' + task_prompt + '.png')

    # clear the plot
    plt.clf()

with open('./result_matrix.json') as f:
    all_data = json.load(f)

for matrix_name in tqdm(all_data): # 
    data = all_data[matrix_name]
    matrix_info = matrix_name.split('_')
    data_type = matrix_info[0] + '_' + matrix_info[1]
    model_type = matrix_info[2]
    prompt_type = matrix_info[3]
    matrix_data = np.array(all_data[matrix_name]['Matrix'])

    draw_heat_map(data_type, model_type, prompt_type, matrix_data)
    