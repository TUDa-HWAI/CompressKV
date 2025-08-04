"""
This script is adapted from 
https://github.com/FYYFU/HeadKV.git
"""
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from loguru import logger

# def compress(model, args):
#     layers = len(model.model.layers)
#     model.config.window_size = args.window_size
#     model.config.max_capacity_prompt = args.max_capacity_prompt
#     model.config.kernel_size = args.kernel_size
#     model.config.pooling = args.pooling

# Copied from transformers.models.llama.modeling_llama.repeat_kv

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class PyradmidKVCluster():
    def __init__(self, num_hidden_layers=32,window_size = 8, max_capacity_prompt = 256, kernel_size = 5, pooling = 'avgpool', pyram_beta = 20):
        
        
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.pyram_init = False

        self.pyram_beta = pyram_beta

        

    def reset(self,num_hidden_layers=32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling


    def update_kv(self, key_states, query_states, value_states, num_key_value_groups,layer_idx):
        # check if prefix phase
        # print(f"Pyram mode adaptive capacity, layer: {layer_idx},layer_budget: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_key_value_heads = key_states.shape[1]        
        # TODO
        if self.pyram_beta and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - layer_idx * steps + self.window_size
            self.pyram_init = True
            # print(f"Pyram mode adaptive capacity, layer: {layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            key_states_tmp = repeat_kv(key_states,num_key_value_groups)
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_tmp.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            attn_cache = attn_cache.reshape(attn_cache.shape[0],num_key_value_groups,num_key_value_heads,-1)
            attn_cache = attn_cache.mean(dim=1)
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
        

def init_pyradmidkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 8
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 1024
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'pyram_beta'):
            self.config.pyram_beta = 20
        if not hasattr(self.config, 'num_hidden_layers'):
            self.config.num_hidden_layers = 32
    self.kv_cluster = PyradmidKVCluster(
                                    num_hidden_layers =self.config.num_hidden_layers,
                                    window_size = self.config.window_size,
                                    max_capacity_prompt = self.config.max_capacity_prompt,
                                    kernel_size = self.config.kernel_size,
                                    pooling = self.config.pooling,
                                    pyram_beta = self.config.pyram_beta)