
export CUDA_VISIBLE_DEVICES=0

models=(
    meta-llama/Llama-3.1-8B-Instruct 
    mistralai/Mistral-7B-Instruct-v0.3
)

budget=32
datasets=(narrativeqa qasper multifieldqa_en hotpotqa musique 2wikimqa passage_count passage_retrieval_en trec triviaqa samsum lcc repobench-p qmsum multi_news gov_report)



for model in "${models[@]}"
do
    if [[ $model == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
        importance_head_path="scores/Llama-3.1-8B-Instruct_head_idx.json"
    elif [[ $model == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
        importance_head_path="scores/Mistral-7B-Instruct-v0.3_head_idx.json"
    else
        echo "Unknown model: $model"
        exit 1
    fi
    for dataset in "${datasets[@]}"
    do
        echo "Running $model $dataset"
        python -m longbench.pred_get_importance_score \
            --model "$model" \
            --importance_head_path "$importance_head_path" \
            --max_capacity_prompt "$budget" \
            --dataset "$dataset"
    done
done


# fast version -> special the sampling number of each dataset like

# sampling_number=10


# for model in "${models[@]}"
# do
#     if [[ $model == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
#         importance_head_path="scores/Llama-3.1-8B-Instruct_head_idx.json"
#     elif [[ $model == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
#         importance_head_path="scores/Mistral-7B-Instruct-v0.3_head_idx.json"
#     else
#         echo "Unknown model: $model"
#         exit 1
#     fi
#     for dataset in "${datasets[@]}"
#     do
#         echo "Running $model $dataset"
#         python -m longbench.pred_get_importance_score \
#             --model "$model" \
#             --importance_head_path "$importance_head_path" \
#             --max_capacity_prompt "$budget" \
#             --dataset "$dataset" \
#             --sampling_number "$sampling_number"
#     done
# done

