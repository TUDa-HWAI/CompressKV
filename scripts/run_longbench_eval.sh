#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


eval_paths=(
    "PATH_TO_EVAL1"
    "PATH_TO_EVAL2"
    # "PATH_TO_EVAL3"
)


for path in "${eval_paths[@]}"; do
    echo "Processing ${path} ..."
    python -m longbench.eval \
        --path "${path}" \
        --eval_avg
done
