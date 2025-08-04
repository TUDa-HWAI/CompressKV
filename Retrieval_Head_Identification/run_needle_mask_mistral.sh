
export CUDA_VISIBLE_DEVICES=0

mask_tops=(10 20 30)
score_paths=(
    "head_score/Mistral-7B-Instruct-v0.3_SRH.json"
    "head_score/Mistral-7B-Instruct-v0.3.json"
)
model_path="mistralai/Mistral-7B-Instruct-v0.3"
s_len=1000
e_len=35000
context_length_arg="--context_lengths 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000"

without masking performance
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
