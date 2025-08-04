import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from loguru import logger


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

def get_important_head_kv(hidden_states,num_key_value_groups,important_head_cl):
    group_ids = important_head_cl // num_key_value_groups
    return hidden_states[:,group_ids]

class compresskvCluster():
    def __init__(self, window_size = 8, max_capacity_prompt_layer_adaptive = 256, kernel_size = 5, pooling = 'avgpool',important_heads=[],first_k=4):
        
        self.window_size = window_size
        self.max_capacity_prompt_layer_adaptive = max_capacity_prompt_layer_adaptive
        assert self.max_capacity_prompt_layer_adaptive[-1] - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.first_k = first_k
        self.important_heads = torch.tensor(important_heads)[:,:self.first_k]
        logger.info(f"budget {max_capacity_prompt_layer_adaptive}")
    def reset(self, window_size = 8, max_capacity_prompt_layer_adaptive = 256, kernel_size = 5, pooling = 'avgpool',important_heads=[],first_k=4):
        assert isinstance(max_capacity_prompt_layer_adaptive, list), "max_capacity_prompt must be a list type"
        self.window_size = window_size
        self.max_capacity_prompt_layer_adaptive = max_capacity_prompt_layer_adaptive
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.first_k = first_k
        self.important_heads = torch.tensor(important_heads)[:,:self.first_k]
        
    @torch.no_grad()
    def update_kv(self, key_states, query_states, value_states, num_key_value_groups,layer_idx):

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_key_value_heads = key_states.shape[1]

        if q_len < self.max_capacity_prompt_layer_adaptive[layer_idx]:
            return key_states, value_states
        else:
            important_head_cl = self.important_heads[layer_idx]
            key_states_tmp = get_important_head_kv(key_states,num_key_value_groups,important_head_cl)
            attn_weights = torch.matmul(query_states[:,important_head_cl,-self.window_size:,:], key_states_tmp.transpose(2, 3)) / math.sqrt(head_dim)
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
            
            attn_cache = attn_cache.mean(dim=1)
            indices= attn_cache.topk(self.max_capacity_prompt_layer_adaptive[layer_idx] - self.window_size, dim=-1).indices

            indices = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_key_value_heads, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        



def init_compresskv(self):
    if not hasattr(self, "kv_cluster"):
        # import ipdb;ipdb.set_trace()
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 8
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 1024
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not  hasattr(self.config, 'important_heads'):
            raise AttributeError("self.config must have both 'layer_score' and 'important_head' attributes")
        if not hasattr(self.config, 'first_k'):
            self.config.first_k = 4
    self.kv_cluster = compresskvCluster(window_size = self.config.window_size, 
                                    max_capacity_prompt_layer_adaptive = self.config.max_capacity_prompt_layer_adaptive, 
                                    kernel_size = self.config.kernel_size,
                                    pooling = self.config.pooling,
                                    first_k = self.config.first_k,
                                    important_heads = self.config.important_heads 
                                    )