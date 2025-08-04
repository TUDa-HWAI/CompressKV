#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mask_tops=(10 20 30)
score_paths=(
    "head_score/Llama-3.1-8B-Instruct_SRH.json"
    "head_score/Llama-3.1-8B-Instruct.json"
)
model_path="meta-llama/Llama-3.1-8B-Instruct"
context_length_arg="--context_lengths 8000 16000 24000 32000 40000 48000 56000 64000 72000 80000 88000 96000 104000 112000 120000 128000"
s_len=1000
e_len=130000

# without masking performance
python -m needle_in_haystack_with_mask \
    --mask_topk 0 \
    --model_path "$model_path" \
    --score_path "${score_paths[0]}" \
    --s_len $s_len \
    --e_len $e_len \
    $context_length_arg

for score_path in "${score_paths[@]}"
do
    for mask_top in "${mask_tops[@]}"
    do
        echo "Running mask_top=$mask_top score_path=$score_path"
        python -m needle_in_haystack_with_mask \
            --mask_topk $mask_top \
            --model_path "$model_path" \
            --score_path "$score_path" \
            --s_len $s_len \
            --e_len $e_len \
            $context_length_arg
    done
done
