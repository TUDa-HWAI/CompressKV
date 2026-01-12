"""
This script is adapted from 
https://github.com/FYYFU/HeadKV.git
"""
import warnings

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache
from flash_attn import flash_attn_func
# perform qk calculation and get indices
# this version will not update in inference mode



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

class AdaptiveCompressKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None, 
                num_hidden_layers = None, pyram_mode = False, pyram_beta = 20, gqa_support = False, num_key_value_groups = 1,
                important_heads=[],first_k=4,max_capacity_prompt_pyramid=None
                ):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        # gqa_support
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups

        #compresskv
        self.max_capacity_prompt_pyramid = max_capacity_prompt_pyramid
        self.first_k = first_k
        self.important_heads = torch.tensor(important_heads)[:,:self.first_k]
        
    def calcul_attn_sore(self, key_states, query_states,layer_idx):
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        important_head_cl = self.important_heads[layer_idx]
        attn_weights_importance_head_sum = attn_weights[:, important_head_cl, -self.window_size:, : -self.window_size].sum(dim = -2)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            attn_cache_importance_head = F.avg_pool1d(attn_weights_importance_head_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            attn_cache_importance_head = F.max_pool1d(attn_weights_importance_head_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        if self.gqa_support:

            attn_cache = attn_cache.reshape(attn_cache.shape[0],self.num_key_value_groups,num_heads//self.num_key_value_groups,-1)
            attn_cache = attn_cache.mean(dim=1)
        attn_cache_importance_head = attn_cache_importance_head.mean(dim=1)
        
        return attn_cache,attn_cache_importance_head

    def update_kv(self, key_states, query_states, value_states,layer_idx):
        return self.update_kv_gqa(key_states, query_states, value_states,layer_idx)
    
    # update kv with gqa_support
    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states,layer_idx):

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = origin_key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_score,attn_score_importance_head = self.calcul_attn_sore(origin_key_states,query_states,layer_idx)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}")

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
            
            
            # gqa support
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)
            
        adaptive_layer_capacity = self.max_capacity_prompt_pyramid[layer_idx] - self.window_size

        if adaptive_layer_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)


        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        _,sorted_attn_score_importance_head_indices = attn_score_importance_head.sort(dim=-1,descending=True)
        if layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:adaptive_layer_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight

            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)

            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*adaptive_layer_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length
            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*adaptive_layer_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
            

        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * adaptive_attn_score
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

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
            
            
            # cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]
            cache_index = sorted_attn_score_importance_head_indices[...,:head_adaptive_capacity[0][head_idx]].unsqueeze(0)
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

        return heads_key_states, heads_value_states


def init_adaptive_compresskv(self):
  
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config,"floor_alpha"),"floor_alpha not set"
    assert self.config.floor_alpha is not None
    

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaptiveCompressKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_alpha,
            skip = self.config.skip,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = self.config.pyram_beta,
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            first_k = self.config.first_k,
            important_heads = self.config.important_heads,
            max_capacity_prompt_pyramid = self.config.max_capacity_prompt_pyramid
        )