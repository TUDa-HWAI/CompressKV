#!/bin/bash



export CUDA_VISIBLE_DEVICES=0

models=(
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
)
max_capacity_prompts=(
    128
    256
    512
    1024
    2048
)
methods=(
    "fullkv"
    "compresskv"
    "cakekv"
    "snapkv"
    "pyramidkv"
    "streamingllm"
)

for model in "${models[@]}"
do
    head_idx_path=""
    layer_utility_path=""

    for method in "${methods[@]}"
    do
        if [[ "$method" == "fullkv" ]]; then
            echo "Running model=$model method=$method"
            python -m longbench.pred_compresskv \
                --model $model \
                --method $method 
        else
            for budget in "${max_capacity_prompts[@]}"
            do
                echo "Running model=$model method=$method budget=$budget "
                if [[ "$method" == "compresskv" ]]; then
                    if [[ "$model" == *"Llama-3.1-8B-Instruct"* ]]; then
                        head_idx_path="scores/Llama-3.1-8B-Instruct_head_idx.json"
                        layer_utility_path="scores/Llama-3.1-8B-Instruct_layer_score.jsonl"
                    elif [[ "$model" == *"Mistral-7B-Instruct-v"* ]]; then
                        head_idx_path="scores/Mistral-7B-Instruct-v0.3_head_idx.json"
                        layer_utility_path="scores/Mistral-7B-Instruct-v0.3_layer_score.jsonl"
                    else
                        echo "Unknown model: $model"
                        exit 1
                    fi
                    python -m longbench.pred_compresskv \
                        --model $model \
                        --max_capacity_prompt $budget \
                        --method $method \
                        --layer_importance_score_path "$layer_utility_path" \
                        --importance_head_path "$head_idx_path" 
                elif [[ "$method" == "cakekv" ]]; then
                    tau1=""
                    tau2=""
                    if [[ "$budget" == "512" || "$budget" == "1024" || "$budget" == "2048" ]]; then
                        if [[ "$model" == *"Llama-3.1-8B-Instruct"* ]]; then
                            tau1="--tau1 1.6"
                            tau2="--tau2 0.4"
                        elif [[ "$model" == *"Mistral-7B-Instruct-v"* ]]; then
                            tau1="--tau1 0.8"
                            tau2="--tau2 0.5"
                        fi
                    elif [[ "$budget" == "128" || "$budget" == "256" ]]; then
                        if [[ "$model" == *"Llama-3.1-8B-Instruct"* ]]; then
                            tau1="--tau1 1.6"
                            tau2="--tau2 0.6"
                        elif [[ "$model" == *"Mistral-7B-Instruct-v"* ]]; then
                            tau1="--tau1 1.0"
                            tau2="--tau2 0.8"
                        fi
                    fi
                    python -m longbench.pred_compresskv \
                        --model $model \
                        --max_capacity_prompt $budget \
                        --method $method \
                        $tau1 $tau2
                else
                    python -m longbench.pred_compresskv \
                        --model $model \
                        --max_capacity_prompt $budget \
                        --method $method \
                        
                fi
            done
        fi
    done
done

