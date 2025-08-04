# import ipdb;
import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig,LlamaForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from pathlib import Path
from methods.cakekv.cake_cache import CakeprefillKVCache 

def error_aware_layer_budget_allocation(layer_ratio, total_budget, min_tokens=32, max_tokens=1536):
    importance = np.array(layer_ratio)
    base = np.full_like(importance, min_tokens, dtype=float)
    remaining_budget = total_budget - base.sum()
    extra = np.round(importance * remaining_budget)
    budget_per_layer = base + extra
    budget_per_layer = np.clip(budget_per_layer, min_tokens, max_tokens)
    diff = int(total_budget - budget_per_layer.sum())

    while diff != 0:
        if diff > 0:
            candidates = np.where(budget_per_layer < max_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(-importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] < max_tokens:
                    budget_per_layer[i] += 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            candidates = np.where(budget_per_layer > min_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] > min_tokens:
                    budget_per_layer[i] -= 1
                    diff += 1
                    if diff == 0:
                        break
    return budget_per_layer.astype(int).tolist()

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    # KV Compression
    parser.add_argument("--method", type=str, default="fastkv", choices=["fullkv", "pyramidkv", "snapkv", "cakekv","compresskv","adakv", "headkv","streamingllm"])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--pooling", type=str, default="avgpool")
    
    #pyramidkv
    parser.add_argument('--pyram_beta', default=20,type=int)
    #cakekv
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--tau1', default=1.0,type=float)
    parser.add_argument('--tau2', default=1.0,type=float)
    #compresskv
    parser.add_argument('--layer_importance_score_path',type=str,default=None,help="path to load the layer importance score for budget allocation")
    parser.add_argument('--importance_head_path',type=str,default=None,help="path to load the importance head for selection")
    parser.add_argument('--first_k',type=int,default=4,help="path to load the importance head for selection")


    return parser.parse_args(args)

# # This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):

    if 'Llama-2-7b-chat-hf' in model_name:

        prompt = f"[INST]{prompt}[/INST]"
    elif 'Llama-3.1-8B-Instruct' in model_name:
        # print("======== llama build chat ========")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path,args_all):
    


    if args_all.method == 'snapkv':
        from methods.snapkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args_all.method == 'pyramidkv':
        from methods.pyramidkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args_all.method == 'cakekv':
        from methods.cakekv.monkeypatch import replace_flashllama_attn_with_cakeattn, replace_flashmistral_attn_with_cakeattn
        replace_flashllama_attn_with_cakeattn()
        replace_flashmistral_attn_with_cakeattn()
    elif args_all.method == 'compresskv':
        from methods.compresskv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args_all.method == 'streamingllm':
        from methods.streamingllm.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args_all.method == "fullkv":
            logger.info(f"using full cache")
    else:
        raise ValueError(f"We does not support {args_all.method} mode") 
    
    device = torch.device(f'cuda:{rank}')
    tokenizer = AutoTokenizer.from_pretrained(args_all.model)
    model = AutoModelForCausalLM.from_pretrained(args_all.model,attn_implementation='flash_attention_2', torch_dtype=torch.float16,use_cache=True).to(device)
    model.eval()
    

    if args_all.method != "fullkv": 
        if args_all.method != "cakekv":
            model.model.config.window_size = args_all.window_size
            model.model.config.kernel_size = args_all.kernel_size
            model.model.config.max_capacity_prompt = args_all.max_capacity_prompt
            model.model.config.pooling = args_all.pooling
            if args_all.method == "pyramidkv":
                model.model.config.pyram_beta = args_all.pyram_beta
            elif args_all.method == "compresskv":
                layers = model.model.config.num_hidden_layers
                if args_all.layer_importance_score_path is not None:
                    layer_score = json.load(open(args_all.layer_importance_score_path, "r"))["avg_score"]
                    max_capacity_prompt_layer_adaptive = error_aware_layer_budget_allocation(layer_score,args_all.max_capacity_prompt*layers,32,args_all.max_capacity_prompt*3)
                else:
                    max_capacity_prompt_layer_adaptive = [args_all.max_capacity_prompt] * layers
                with open(args_all.importance_head_path, 'r') as f:
                    important_head = json.load(f)  
                important_head = [important_head[str(i)] for i in range(len(important_head))]
                model.model.config.important_heads = important_head
                model.model.config.first_k = args_all.first_k
                model.model.config.max_capacity_prompt_layer_adaptive = max_capacity_prompt_layer_adaptive
        elif args_all.method == "cakekv":
            layers = model.model.config.num_hidden_layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.key_size = [args_all.max_capacity_prompt - args_all.window_size]*layers
                model.model.layers[i].self_attn.config.window_size = [args_all.window_size]*layers
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
                model.model.layers[i].self_attn.config.tau1 = args_all.tau1
                model.model.layers[i].self_attn.config.tau2 = args_all.tau2
                model.model.layers[i].self_attn.config.gamma = args_all.gamma
                model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                    cache_size=args_all.max_capacity_prompt,
                    window_size=args_all.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=True
                )]*layers


    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)

            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    # pad_token_id=tokenizer.eos_token_id,
                )[0]
        if args_all.method == "cakekv":
            layers = len(model.model.layers)
            for i in range(layers):
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    
    seed_everything(42)
    args_all= parse_args()
    world_size = torch.cuda.device_count()
    logger.info(f"number of gpu {world_size}")
    


    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args_all.model

    max_length = model2maxlen[model_name]
    logger.info(f"Model: {model_name}")
    logger.info(f"Max length: {max_length}")

    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
    


    if args_all.method != "fullkv":
        save_path = f"outputs/longbench/{args_all.model}/{args_all.method}/{args_all.max_capacity_prompt}/"
    else:
        save_path = f"outputs/longbench/{args_all.model}/{args_all.method}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for dataset in datasets:

        logger.info("Evaluating on dataset: {}".format(dataset))
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        
        out_path = os.path.join(save_path, f"{dataset}.jsonl")

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(0,world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, out_path,args_all))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

