"""
This script is adapted from 
https://github.com/Zefan-Cai/KVCache-Factory.git
"""
# import ipdb;
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

class StreamingLLMKVCluster():
    def __init__(self, window_size = 8, max_capacity_prompt = 256 , kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 8, max_capacity_prompt = 256 , kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        

    def update_kv(self, key_states, query_states, value_states, num_key_value_groups,layer_idx):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_key_value_heads = key_states.shape[1]
        # print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_key_value_heads, 1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
def init_streamingllm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 8
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 256
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'

    self.kv_cluster = StreamingLLMKVCluster(window_size = self.config.window_size, 
                                    max_capacity_prompt = self.config.max_capacity_prompt, 
                                    kernel_size = self.config.kernel_size,
                                    pooling = self.config.pooling)