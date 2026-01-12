"""
This script is adapted from 
https://github.com/FYYFU/HeadKV.git
"""

import warnings
import os

import torch
import time
import numpy as np
import json
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache

class DynamicCacheSplitHeadFlatten(Cache):
    """
    Flattened version of DynamicCacheSplitHead
    """
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # NOTE: k, v = [head_num](bs, 1, seqlen, dim)
        # each layer is a flatten layout like:
        # [head_0_len + head_1_len + ..., dim]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []):
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            # NOTE: phase 2. we got [bs, head, seqlen, dim] as k, v input
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # TODO: wrap as a python interface
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)


            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx or (len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []):
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


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

class ReasonCompressKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None, head_choice=None, beta=None, temp=None, layer_idx = None, num_hidden_layers = None, num_attention_heads=None, model=None,gqa_support=False,num_key_value_heads=8,
                 head_score_path=None, first_k=None, important_heads=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.beta = beta
        self.temp = temp
        self.gqa_support = gqa_support
        self.head_score_path = head_score_path
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        
        self.first_k = first_k
        self.important_heads = torch.tensor(important_heads)[:,:self.first_k]
        
        with open(self.head_score_path, 'r') as file:
            head_list = json.loads(file.readline())
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        head_score_list = torch.pow(head_score_list, self.temp)
        head_score_list = head_score_list / torch.sum(head_score_list)
        self.total_attention = head_score_list.reshape(self.num_hidden_layers, self.num_attention_heads)
        total_pool_capacity = (self.base_capacity // self.beta) * self.num_hidden_layers * self.num_attention_heads
        if self.gqa_support:
            # NOTE: GQA support
            new_total_attention = torch.zeros((self.num_hidden_layers, self.num_key_value_heads), device=self.total_attention.device)
            for i in range(self.num_key_value_groups):
                new_total_attention += self.total_attention[:, i * self.num_key_value_heads: (i + 1) * self.num_key_value_heads]
            self.total_attention = new_total_attention
            
            total_pool_capacity = (self.base_capacity // self.beta) * self.num_hidden_layers * self.num_key_value_heads
        min_num = (self.base_capacity - self.base_capacity // self.beta)
        self.head_capacity = torch.round(self.total_attention * total_pool_capacity + min_num).int()
            
    def calcul_attn_score(self, key_states, query_states,layer_idx):
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        important_head_cl = self.important_heads[layer_idx]
        key_states_tmp = get_important_head_kv(key_states,self.num_key_value_groups,important_head_cl)
        
        attn_weights = torch.matmul(query_states[:,important_head_cl,-self.window_size:,:], key_states_tmp.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        attn_cache = attn_cache.mean(dim=1)
        return attn_cache


    def update_kv(self, key_states, query_states, value_states,layer_idx):
            
   
        return self.update_kv_gqa(key_states, query_states, value_states,layer_idx)

    def update_kv_gqa(self,  key_states, query_states, value_states,layer_idx):
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        attn_score= self.calcul_attn_score(key_states,query_states,layer_idx)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)
        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            if self.gqa_support:
                self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

            else:
                self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)
        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)
        _,indices = attn_score.sort(dim=-1,descending=True)
        
        # indices = indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = indices[...,:self.head_capacity[self.layer_idx][head_idx]].unsqueeze(0)
            
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)
        
        return heads_key_states,heads_value_states
        



def init_reason_compresskv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config, 'head_choice'), "head_choice not set"
    assert hasattr(self.config, 'beta'), "beta not set"
    assert hasattr(self.config, 'temp'), 'temp not set'
    assert hasattr(self.config, "num_attention_heads"), "num_attention_heads not set"
    assert hasattr(self.config, "num_key_value_heads"), "num_key_value_heads not set"
    assert hasattr(self.config, "gqa_support"), "gqa_support not set"

    # init only once
    if not hasattr(self, "kv_cluster"):
        
        self.kv_cluster = ReasonCompressKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            head_choice=self.config.head_choice,
            beta=self.config.beta,
            temp=self.config.temp,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            model=self.config._name_or_path,
            gqa_support=self.config.gqa_support,
            head_score_path=self.config.head_score_path,
            first_k = self.config.first_k,
            important_heads = self.config.important_heads,
            )





