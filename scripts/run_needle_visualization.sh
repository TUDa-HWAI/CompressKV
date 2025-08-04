#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

save_dir="YOUR_SAVE_DIR"
eval_paths=(
    "PATH_TO_EVAL1"
    "PATH_TO_EVAL2"
    # "PATH_TO_EVAL3"
)

models=(
    "MODEL_1"
    "MODEL_2"
    # "MODEL_3"
)

for model in "${models[@]}"; do
    for path in "${eval_paths[@]}"; do
        echo "Running model ${model} on eval path ${path} ..."
        python -m needle.show_image_paper \
        --save_dir "$save_dir" \
        --model "$model" \
        --eval_path "$path"
    done
done