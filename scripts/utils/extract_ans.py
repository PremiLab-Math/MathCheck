import json
import re
import pdb
from sympy.parsing.latex import parse_latex # parse the latex equation
import numpy as np
import openai
import random
import time
import copy
import func_timeout
from tqdm import tqdm
from collections import defaultdict, Counter
from fraction import Fraction
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.getcwd().split("MathCheck")[0] + "MathCheck/") # Set all the path as "MathCheck"


# openai.api_base = "https://giegie.green/v1"  # 换成代理，一定要加v1
# OPENAI_KEY = "sk-iMRrZc77AOExOQYG3cBd98A04d53479cB9D5A644B3705aA0"

openai.api_base = "https://api2.aigcbest.top/v1"  # 换成代理，一定要加v1
OPENAI_KEY = "sk-ERVBBySgrXabbxrx237d62346e904035921d65E5B5Da4630"

openai.api_key = OPENAI_KEY


def invoke_openai(messages, model="o1-preview", mode="", max_num_tokens=512, temperature=0.0, max_retries=10, delay=7, stop=None):
    top_p = 1
    if mode == 'pot':
        stop = ['\n\n']
    max_try = 1
    prediction = ""
    # print(messages)
    print(model, end=" ")
    while max_try<max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                # max_tokens=max_num_tokens,
                # top_p=top_p,
                # temperature=temperature,
                # stop=stop # NOTE: raise: none is not an allowed value (type=type_error.none.not_allowed), temporarily remove it
            )
            prediction = response['choices'][0]['message']['content']
        except Exception as e:
            print("Exception: ",e)
            time.sleep(delay)
            max_try +=1
        else:
            break
    return prediction

def batch_inference_api(messages, seed=0, max_retries=10, delay=7, model="o1-preview"):
    responses = Parallel(n_jobs=20, prefer="threads")(
        delayed(invoke_openai)(message, model, max_retries=max_retries, delay=delay)
        for message in tqdm(messages, desc="Processing prompts")  
    )
    return responses


# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]


def extract_outcome_correctness(text):
    answer_pattern = r'The answer is:\s*(\w+)'
    answer_match = re.search(answer_pattern, text, flags=re.IGNORECASE)
    if answer_match:
        first_answer = answer_match.group(1).lower()
        if first_answer in ["correct", "incorrect"]:
            return "Correct" if first_answer == "correct" else "Incorrect"

    matches = re.findall(r'\b(correct|incorrect)\b', text, flags=re.IGNORECASE)
    if matches:
        last_match = matches[-1].lower()
        return "Correct" if last_match == "correct" else "Incorrect"

    return None

def extract_process_correctness(text):
    if re.search(r'all steps are correct', text, flags=re.IGNORECASE):
        return "All-Correct"
    
    answer_match = re.search(r'The answer is(.*?)(Step \d+)', text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(2).replace("step","Step")  # "Step X"
    
    judgement_match = re.search(r'Judgement:\s*(\w+)', text, flags=re.IGNORECASE)
    if judgement_match:
        judgement_text = judgement_match.group(1)
        if judgement_text.lower().startswith('step'):
            return judgement_text.replace("step","Step")
    
    all_steps = re.findall(r'Step \d+', text, flags=re.IGNORECASE)
    unique_steps = set(map(str.lower, all_steps))  
    if len(unique_steps) == 1:
        return list(unique_steps)[0].capitalize()

    begins_match = re.search(r'(begin(s)?|began|start(s)?) (at|with) (Step \d+)', text, flags=re.IGNORECASE)
    if begins_match:
        return begins_match.group(5).capitalize() 


    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 1:
        last_sentence = sentences[-2].strip()  
        first_sentence = sentences[0].strip()
    else:
        first_sentence = sentences[0]
        last_sentence = sentences[0]  

    if " correct" in last_sentence.lower():
        return "All-Correct"
    match_step = re.search(r'Step \d+', last_sentence, re.IGNORECASE)
    if match_step:
        return match_step.group(0)
    

    if " correct" in first_sentence.lower():
        return "All-Correct"
    match_step = re.search(r'Step \d+', first_sentence, re.IGNORECASE)
    if match_step:
        return match_step.group(0)
    
    return None

def extract_answerable(text):
    matches = re.findall(r'(Answerable|Unanswerable)', text, re.IGNORECASE)

    if matches:
        last_match = matches[-1]
        if "unanswerable" in last_match.lower():
            return "Unanswerable"
        elif "answerable" in last_match.lower():
            return "Answerable"
    
    return None

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass  
    try:
        import unicodedata
        unicodedata.numeric(s) 
        return True
    except (TypeError, ValueError):
        pass
    return False


def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  
        n=str(n)
        return n

def synthesize_program_my(result: str, prefix: str) -> str:
    program = prefix
    program_cand = prefix

    count = 0
    for i, line in enumerate(result.split('\n')):
        if line.strip(' ') in prefix:
            continue
        if count == 0:
            program += line + '\n'
        else:
            if line.startswith('    '):
                program += line + '\n'
            else:
                break
            if line.startswith('    return '):
                break
        count += 1

    count = 0
    for i, line in enumerate(result.split('\n')):
        if line.strip(' ') in prefix:
            continue

        if line.startswith('    '):
            program_cand += line + '\n'
        else:
            # break
            line = '    ' + line
            program_cand += line + '\n'
        if line.startswith('    return '):
            break
        count += 1

    program += 'ans = solver()'
    program = program.replace(' [/INST]', '    # [/INST]')
    program_cand += 'ans = solver()'
    program_cand = program_cand.replace(' [/INST]', '    # [/INST]')

    return program, program_cand


def synthesize_program_sego(result: str, prefix: str) -> str:
    result += '\nans = solve()'
    return result


def safe_execute_sego(code_string: str, keys=None):
    GLOBAL_DICT = {}
    _global_vars = copy.copy(GLOBAL_DICT)
    def execute(x):
        try:
            exec(x, globals()) in globals(), globals()
            locals_ = locals()
            globals_ = globals()

            if keys is None:
                res = locals_.get('ans', None) or globals_.get('ans', None)
                if res is None:
                    globals_.get('ans', None)
                    exec(x)
                    locals_ = locals()
                    res = locals_.get('ans', None)
                return res
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(10, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None
    return ans



def safe_execute(code_string: str, keys=None):
    GLOBAL_DICT = {}
    _global_vars = copy.copy(GLOBAL_DICT)
    def execute(x):
        try:
            exec(x, globals()) in globals(), globals()
            locals_ = locals()
            globals_ = globals()
            if keys is None:
                res = locals_.get('ans', None) or globals_.get('ans', None)
                if res is None:
                    globals_.get('ans', None)
                    exec(x)
                    locals_ = locals()
                    res = locals_.get('ans', None)
                return res
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(10, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def simplify_ans(ans, convert_to_str: bool = True):
    if 'relational' in str(type(ans)):
        return str(ans)
    elif 'numpy' in str(type(ans)):
        if ans.shape == ():
            # scalar value
            ans = round(float(ans), 2)
        else:
            # array value
            ans = round(float(ans[0]), 2)
        if convert_to_str:
            return str(ans)
        else:
            return ans
    elif not ans:
        return None
    else:
        if type(ans) in [list, tuple]:
            if 'sympy' in str(type(ans[0])):
                try:
                    ans = [round(float(x), 2) for x in ans]
                except Exception:
                    ans = [str(x) for x in ans]
            if len(ans) == 1:
                ans = ans[0]
        else:
            if 'sympy' in str(type(ans)):
                try:
                    ans = round(float(ans), 2)
                except Exception:
                    ans = str(ans)
        if convert_to_str:
            return str(ans)
        else:
            return ans


def extract_gold_ans(answer_str):
    answer_str = answer_str.strip("\n").strip(" ").rstrip(".").replace(",", "")
    pattern = "####(.*)"
    if len(re.findall(pattern, answer_str)) >= 1:
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    else:
        pattern = "boxed{(.*)}"
        if len(re.findall(pattern, answer_str)) < 1:
            pdb.set_trace()
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    if target != "None":
        if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', target)) < 1:
            print(answer_str)
            pdb.set_trace()
        temp_ans = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', target)[0]
        temp_ans = delete_extra_zero(temp_ans)
    else:
        temp_ans = "None"
    return temp_ans


def extract_pred_ans(pred_str, prompt_type="cot", match_pattern="", input_str=""):
    if "code" not in prompt_type:
        pred_str = pred_str.rstrip(".").replace(",", "")

    pattern = "####(.*)"
    if "Question" in pred_str:
        pred_str = pred_str.split("Question")[0]
    preds = re.findall(pattern, pred_str)
    pred = delete_extra_zero(preds[-1].strip(" ")) if len(preds) >= 1 and bool(re.search(r"\d", preds[-1])) else ""
    if pred == "":
        pred = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)
        if len(pred) >= 1:
            pred = delete_extra_zero(pred[-1].replace(",", "").strip(".").strip(" "))
        else:
            pred = ""
    else:
        pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred.replace(",", ""))[0].strip(".").strip(" "))
    if "</s>" in pred:
        pred = pred[:-4]

    if "code" not in prompt_type:
        pred = pred.rstrip(".").strip(" ")
    return pred


def extract_pred_ans_none(pred_str, prompt_type="", match_pattern="####"):
    pred_str = pred_str.rstrip(".").replace(",", "").lower()
    pred = ""
    patterns = ["does not provide enough information", "does not specify", "does not provide", "can't provide", "can not provide", "don't know", "do not know", "doesn't specify", "not specify", "not mention", "doesn't mention", "don't have enough information", "do not have enough", "not provide", "doesn't provide", "cannot calculate", "can't calculate", "can't determine", "cannot determine", "missing necessary information", "none"]
    for p in patterns:
        if p in pred_str:
            pred = "None"
    if "cot" in prompt_type:
        match_pattern = "####"
    elif prompt_type == "ltm":
        match_pattern = "the answer is:"
    elif prompt_type == "complex":
        match_pattern = "the answer is "
    elif prompt_type == "codellama":
        result_counter = Counter()
        ans = safe_execute(pred_str)
        ans = floatify_ans(ans)
        if ans is not None:
            result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = "None"
        pred = delete_extra_zero(prediction)
        if pred == "":
            pred = "None"
    else:
        match_pattern = match_pattern.lower()
    if pred != "None" and match_pattern not in pred_str:
        pred = "None"
    return pred


def extract_answer_number_mistral(completion):
    text = completion.split('####')
    if len(text) > 1:
        extract_ans = text[-1].strip().strip("\n")
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def write_results_to_pred_file(original_pred_data, preds, golds, results, pred_file, neglect_ncr=False, mv=1, type="json"):
    new_pred_data = []
    for i, item in enumerate(original_pred_data):
        if neglect_ncr and item["type"] == "critical thinking":
            new_pred_data.append(item)
            continue
        if mv == 1:
            item['pred'] = preds[i]
        else:
            item['pred'] = preds[i][0]
            item["pred_list"] = preds[i][1]
        item['gold'] = golds[i]
        item['result'] = results[i]
        new_pred_data.append(item)

    # Save the updated list of dictionaries back to the jsonl file
    assert len(new_pred_data) == len(original_pred_data)
    if type == "json":
        with open(pred_file, 'w') as file:
            json.dump(new_pred_data, file, indent=4)
    else:
        with open(pred_file, 'w') as file:
            for item in new_pred_data:
                file.write(json.dumps(item) + "\n")


def check_acc(input_file, fine_grained=False, new_file=False):
    input_data = json.load(open(input_file))
    results = {}
    new_data = []
    for item in input_data:
        gold = item["gold"]
        if "pred" not in item:
            item["pred"] = extract_pred_ans(item["model_prediction"])
        pred = item["pred"]
        if fine_grained:
            type = item["type"]
        else:
            type = "default"
        if type not in results:
            results[type] = []
        results[type].append(gold == pred)
        if new_file:
            item["result"] = (gold==pred)
            new_data.append(item)
    for key in results:
        acc = np.array(results[key]).sum()/len(results[key])
        print(f"{key}: {acc}")
    if new_file:
        json.dump(new_data, open(input_file, "w"), indent=4)
    return results


def split_sentences(text):
    sentences = re.split(r'(?<=[.,])\s+', text)
    return sentences


def is_question_sentence(sentence):
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'calculate', 'determine']

    # Check if the sentence ends with a question mark
    if sentence.endswith('?'):
        return True

    # Check if any question words are present in the sentence
    for word in question_words:
        if word in sentence.lower():
            return True

    return False


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  
    matches = re.findall(pattern, text)
    return bool(matches)


def get_fine_grained_perf_change(acc_table, gsm8k_value=0.0, only_delta=False):
    color_list = [
        "\cellcolor{DarkGreen!75}", "\cellcolor{DarkGreen!65}", "\cellcolor{DarkGreen!55}",
        "\cellcolor{DarkGreen!45}",
        "\cellcolor{DarkGreen!35}", "\cellcolor{DarkGreen!25}", "\cellcolor{DarkGreen!18}",
        "\cellcolor{DarkGreen!13}"
    ]
    acc_table_color = {}
    all_acc_table_color = {}
    all_decay_table_color = {}
    prev_number = 0
    color_idx = -1
    for key, acc in acc_table:
        this_acc = str(round(acc * 100, 2))
        all_acc_table_color[key] = float(this_acc)
        if this_acc != prev_number:
            color_idx += 1
        if float(this_acc) < gsm8k_value:
            if only_delta:
                acc_table_color[key] = color_list[color_idx] + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))
            else:
                acc_table_color[key] = color_list[color_idx] + "$\\text{" + this_acc + "}_{\\text{" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 1)) + "}}$"
        else:
            if only_delta:
                acc_table_color[key] = "\cellcolor{lightred}" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))
            else:
                acc_table_color[key] = "\cellcolor{lightred}$\\text{" + this_acc + "}_{\\text{" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 1)) + "}}$"
        all_decay_table_color[key] = str(round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))

    color_str = ""
    all_acc_str = ""
    all_decay_str = ""

    for key in ["numerical substitution", "digit expansion", "integer-decimal-fraction conversion",
                "problem understanding", "adding operation", "distractor insertion", "reversing operation"]:
        print(key + ": " + acc_table_color[key])
        color_str += (acc_table_color[key] + " & ")
        all_acc_str += str(all_acc_table_color[key]) + ", "
        all_decay_str += all_decay_table_color[key] + ", "
    color_str = color_str.strip(" ").rstrip("&") + " \\"
    print(color_str)
    print("all_acc: ", all_acc_str)
    print("all_decay: ", all_decay_str)

def remove_numbered_prefixes(string):
    pattern = r'^#\d+\.\s'  # Regular expression pattern to match numbered prefixes
    result = re.sub(pattern, '', string)
    return result


def get_checklist(input_file=""):
    load_img = False
    if input_file=="" or input_file is None:
        from datasets import load_dataset
        dataset = load_dataset("zihaozhou/GSM-Checklist")['test']
    else:
        dataset = json.load(open(input_file))
        if input_file[:3].lower() == 'geo': # geometry dataset, load image, the file name should start with: 'geo' -> geo_checklist.json
            load_img = True

    questions = []
    answers = []
    solutions = []
    task_types = []
    question_types = []
    img_paths = []

    for i, item in enumerate(dataset):
        questions.append(item["question"]) if "question" in item.keys() else questions.append(None)
        solutions.append(item["solution"]) if "solution" in item.keys() else solutions.append(None)
        answers.append(item["answer"]) if "answer" in item.keys() else answers.append(None)
        task_types.append(item["task_type"])
        question_types.append(item["question_type"])
        if load_img:
            img_paths.append(item["image"])
        else:
            img_paths.append(None)

    return questions, solutions, answers, task_types, question_types, img_paths


def get_gsm8k(input_path):
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            item = json.loads(line)
            data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
    return data