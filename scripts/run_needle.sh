
export CUDA_VISIBLE_DEVICES=1

models=(
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
)
max_capacity_prompts=(
    2048
    1024
    512
    256
    128
)
methods=(
    "fullkv"
    "compresskv"
    "cakekv"
    "snapkv"
    "pyramidkv"
    "streamingllm"
    
)
save_dir="outputs/needle_result"
for model in "${models[@]}"
do
    head_idx_path=""
    layer_utility_path=""

    if [[ "$model" == *"Llama-3.1-8B-Instruct"* ]]; then
        head_idx_path="scores/Llama-3.1-8B-Instruct_head_idx.json"
        layer_utility_path="scores/Llama-3.1-8B-Instruct_layer_score.jsonl"
        context_length_arg="--context_length 8000 16000 24000 32000 40000 48000 56000 64000 72000 80000 88000 96000 104000 112000 120000 128000"
        save_dir="outputs/needle_result/needle_result_llama3_128k"
    elif [[ "$model" == *"Mistral-7B-Instruct-v"* ]]; then
        head_idx_path="scores/Mistral-7B-Instruct-v0.3_head_idx.json"
        layer_utility_path="scores/Mistral-7B-Instruct-v0.3_layer_score.jsonl"
        context_length_arg="--context_length 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000"
        save_dir="outputs/needle_result/needle_result_mistral_32k"
    else
        echo "Unknown model: $model"
        exit 1
    fi

    for method in "${methods[@]}"
    do
        if [[ "$method" == "fullkv" ]]; then
            echo "Running model=$model method=$method"
            python -m needle.run_needle_in_haystack \
                --model_path "$model" \
                --method "$method" \
                --save_dir "$save_dir" \
                $context_length_arg
        else
            for budget in "${max_capacity_prompts[@]}"
            do
                echo "Running model=$model method=$method budget=$budget save_dir=$save_dir"
                tau1=""
                tau2=""
                
                if [[ "$method" == "compresskv" ]]; then
                    python -m needle.run_needle_in_haystack \
                        --model_path $model \
                        --max_capacity_prompts $budget \
                        --method $method \
                        --layer_importance_score_path "$layer_utility_path" \
                        --importance_head_path "$head_idx_path" \
                        --save_dir "$save_dir" \
                        $context_length_arg
                elif [[ "$method" == "cakekv" ]]; then
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
                    echo "cakekv $model $budget $tau1 $tau2"
                    python -m needle.run_needle_in_haystack \
                        --model_path "$model" \
                        --max_capacity_prompt "$budget" \
                        --method "$method" \
                        --save_dir "$save_dir" \
                        $tau1 $tau2  \
                        $context_length_arg
                else
                    python -m needle.run_needle_in_haystack \
                        --model_path "$model" \
                        --max_capacity_prompt "$budget" \
                        --method "$method" \
                        --save_dir "$save_dir" \
                        $context_length_arg
                        $save_dir
                fi
            done
        fi
    done
done

