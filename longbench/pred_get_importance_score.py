
import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from longbench.model.modeling_llama_new import LlamaForCausalLM
from longbench.model.modeling_mistral_new import MistralForCausalLM
from tqdm import trange


from longbench.eval import dataset2metric    


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    parser.add_argument('--sampling_number',type=int, default=None, help="number of sampling for evaluating shapely value")
    
    parser.add_argument('--max_capacity_prompt',type=int,default=None,choices=[16,32,64,128,256,512,1024,2048], help="Max Budget")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--importance_head_path',type=str,default=None,help="path to load the importance head for selection")
    #first_topk
    parser.add_argument('--first_topk', type=int, default=4, help="number of first k heads to select")

    return parser.parse_args(args)

# # This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    
    if 'llama2' in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif 'llama-3' in model_name.lower():
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name,queue_fnorm,args_all):    

    dtype = torch.float16
    config = AutoConfig.from_pretrained(args_all.model,  trust_remote_code=True)
    config.kernel_size = 5
    config.pooling=  'avgpool'
    config.window_size = 8
    config.max_capacity_prompt = args_all.max_capacity_prompt
    config.first_topk = args_all.first_topk
    with open(args_all.importance_head_path, 'r') as f:
        important_head = json.load(f)  


    important_head = [important_head[str(i)] for i in range(len(important_head))]
    config.importance_head_idx = important_head
    device = torch.device(f'cuda:{rank}')
    tokenizer = AutoTokenizer.from_pretrained(args_all.model,  trust_remote_code=True)

    
    if "llama" in args_all.model.lower():
        logger.info(f"Loading llama model {args_all.model} with flash attention 2")
        model = LlamaForCausalLM.from_pretrained(args_all.model, config = config, torch_dtype=dtype,  attn_implementation="flash_attention_2").to(device)
    elif "mistral" in args_all.model.lower():
        logger.info(f"Loading mistral model {args_all.model} with flash attention 2")
        model = MistralForCausalLM.from_pretrained(args_all.model, config = config, torch_dtype=dtype,  attn_implementation="flash_attention_2").to(device)
    else:
        raise ValueError(f"Model {args_all.model} not supported")
        
    model.generation_config.top_p=None

    num_layers = config.num_hidden_layers

    layer_error_fnorm = [0 for i in range(num_layers)]
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

        output = model.generate(
                    **input,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    min_length=context_length+1,
                    max_new_tokens=max_gen,
                    pad_token_id=tokenizer.eos_token_id,
                    )[0]

        for i in range(num_layers):
            layer_error_fnorm[i] += model.model.layers[i].self_attn.layer_error_fnorm
    if queue_fnorm is not None:
        queue_fnorm.put(layer_error_fnorm)
    else:
        return np.array(layer_error_fnorm)

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
    args_all = parse_args()
    world_size = torch.cuda.device_count()
    logger.info(f"number of gpu {world_size}")
    
    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args_all.model.split("/")[-1]
    max_length = model2maxlen[args_all.model]
    logger.info(f"Model: {args_all.model}")
    logger.info(f"Max length: {max_length}")
    
    
    budget = args_all.max_capacity_prompt
    dataset = args_all.dataset
    
    
    dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
    
    save_path = f"importance_score/{model_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_layer_utility={}
    dataset_layer_utility_fnorm = {}
    num_layers = AutoConfig.from_pretrained(args_all.model).num_hidden_layers

    logger.info("Evaluating on dataset: {}".format(dataset))

    data = load_dataset('THUDM/LongBench', dataset, split='test')


    layer_score_norm_path = f"{save_path}/{model_name}_layer_score.jsonl"
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]
    #fast version
    if args_all.sampling_number is not None:
        data_all = data_all[:args_all.sampling_number]

    data_subsets = [data_all[i::world_size] for i in range(world_size)]

    
    
    layer_utility = np.zeros(num_layers, dtype=np.float64)
    layer_utility_fnorm = np.zeros(num_layers, dtype=np.float64)

    mp.set_start_method('spawn', force=True)
    processes = []
    queue_fnorm = mp.Queue()
    for rank in range(0,world_size):
        p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                    max_gen, prompt_format, dataset, device, model_name,queue_fnorm,args_all))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    while not queue_fnorm.empty():
        layer_utility_fnorm += np.array(queue_fnorm.get())
        
    layer_utility_fnorm = layer_utility_fnorm/layer_utility_fnorm.sum()
    layer_utility_fnorm = layer_utility_fnorm.tolist()

    print(f"layer_utility_fnorm: {layer_utility_fnorm}")
    print(f"saving layer_utility of dataset {dataset }")
    with open(layer_score_norm_path, "a", encoding="utf-8") as f:
        json.dump({f"{dataset}":layer_utility_fnorm}, f, ensure_ascii=False)
        f.write('\n')
