# export PYTHONPATH="/home/LeiFeng/weiliu/MathCheck:$PYTHONPATH"

modelname="Meta-Llama-3-8B-Instruct"
modelpath="meta-llama/Meta-Llama-3-8B-Instruct"

taskprompt="zeroshot"

CUDA_VISIBLE_DEVICES="0" python scripts/text_vllm_inference.py \
    --model $modelname \
    --model_name_or_path $modelpath \
    --input_file "gsm_checklist.json" \
    --output_file "results/gsm_checklist_${modelname}_task_all_question_all_${taskprompt}_prediction.json" \
    --check_task all \
    --batch_size 128 \
    --tp_size 1 \
    --task_prompt ${taskprompt} \
    --is_chat_model \

# python scripts/results_evaluate.py --model_name $modelname --eval_data "gsm_checklist" --task_prompt ${taskprompt}
