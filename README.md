# <img src="https://github.com/MathCheck/MathCheck.github.io/blob/main/static/images/icon.png" width="70" /> MathCheck

**Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist**

[**🌐 Homepage**](https://mathcheck.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/PremiLab-Math/MathCheck/tree/main) | [**📖 Paper**](https://arxiv.org/abs/2407.08733) | [**💻 Results**](https://github.com/PremiLab-Math/MathCheck/tree/main/results) 


## Intro
Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively
define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, which presents a substantial risk of model overfitting and fails to accurately represent genuine
mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly and readily applied across a diverse array
of tasks. Motivated by this, we introduce MATHCHECK, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool
to generate checklists efficiently. MATHCHECK includes multiple mathematical reasoning tasks and robustness test types to facilitate a comprehensive evaluation of
both mathematical reasoning ability and behavior testing. Utilizing MATHCHECK, we develop MATHCHECK-GSM and MATHCHECK-GEO to assess mathematical textual reasoning and multi-modal reasoning capabilities, respectively, servingas upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K.


![image](https://github.com/MathCheck/MathCheck.github.io/blob/main/static/images/Overview.png)


### Commands for Prediction
```
# Call GPT model for GSM-checklist
python scripts/openai_model_inference.py --input_file gsm_checklist.json --model_name gpt-4o  --check_task all --check_question all --task_prompt zeroshot > log/gsm_checklist_gpt-3.5-turbo_all_task_question_zeroshot.out

# Call GPT model for GEO-checklist
python scripts/openai_model_inference.py --input_file geo_checklist.json --model_name gpt-4o  --check_task all --check_question all --task_prompt zeroshot > log/gsm_checklist_gpt-3.5-turbo_all_task_question_zeroshot.out

# [MODEL_NAME] can be: gpt-3.5-turbo, gpt-4o, gpt-4-turbo-2024-04-09, etc.
# [TASK_PROMPT] can be: fewshot, zeroshot
```

### Commands for Score Output
```
python scripts/results_evaluate.py --model_name gpt-4o --eval_data gsm_checklist --task_prompt zeroshot
# [MODEL_NAME] can be: gpt-3.5-turbo, gpt-4o, gpt-4-turbo-2024-04-09, etc.
# [TASK_PROMPT] can be: fewshot, zeroshot
```


## Contact

* Zihao Zhou: zihao.zhou@liverpool.ac.uk
* Shudong Liu: nlp2ct.shudong@gmail.com

## Citation
```
@article{zhou2024modelreallygoodmath,
    title={Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist}, 
    author={Zihao Zhou and Shudong Liu and Maizhen Ning and Wei Liu and Jindong Wang and Derek F. Wong and Xiaowei Huang and Qiufeng Wang and Kaizhu Huang},
    year={2024},
    eprint={2407.08733}
}
```

