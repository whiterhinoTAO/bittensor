import torch
from torch import nn, einsum
from torch.nn import functional as F

import math
from functools import partial, wraps

import bittensor as bt
from bittensor._neuron.text.core_server.nucleus_impl import server
# from bittensor.utils.flash_attention import attention_ref

from typing import Optional, Dict, Tuple

from einops import rearrange

import argparse
import pdb

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm_cpu(t):
    eps = 1e-12 if t.dtype == torch.float32 else 1e-3
    norm = t.norm(dim = 0)
    # norm_clamped = torch.where(norm > eps, norm, eps)
    # o = t / norm_clamped[..., None]
    return norm

def l2norm(t):
    if t.data.is_cuda:
        return F.normalize(t, dim = -1)

    return l2norm_cpu(t)

def grouped_l2norm(t, groups = 1):
    shape = t.shape
    dim = shape[-1]
    t = t.reshape(*shape[:-1], groups, dim // groups)
    t = l2norm(t)
    return t.reshape(shape)

def l2norm_tensors(*tensors, groups = 1):
    assert len(tensors) > 0
    dtype = tensors[0].dtype

    fn = partial(grouped_l2norm, groups = groups)

    tensors = tuple(map(fn, tensors))
    tensors = tuple(map(lambda t: t.type(dtype), tensors))
    return tensors

def plain_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 8,
    groups = 1,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False

):
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    is_merged_batch_heads_query = q.ndim == 3
    single_head_kv = k.ndim == 3

    if is_merged_batch_heads_query:
        assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

        attn_bias_batch_dim = True
        q = q[:, None, ...]

    if l2norm_qk:
        q, k = l2norm_tensors(q, k, groups = groups)

    kv_einsum_eq = 'b j d' if single_head_kv else 'b h j d'
    sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
    sim = sim * scale

    if exists(attn_bias):
        attn_bias = attn_bias.unsqueeze(1 if attn_bias_batch_dim else 0)
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

    attn = sim.softmax(dim = -1)
    out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

    if is_merged_batch_heads_query:
        out = out.squeeze(1)

    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8,
        l2norm_groups = 1,
        pre_norm = False,
        use_cuda_kernel = False,
        non_cosine_sim_attn = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

        self.scale = scale
        self.heads = heads

        self.l2norm_groups = l2norm_groups

        self.attn_fn = plain_cosine_sim_attention

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self, 
        x,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        ):
        h, scale, l2norm_groups = self.heads, self.scale, self.l2norm_groups

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self.attn_fn(q, k, v, causal = True, scale = scale, groups = l2norm_groups)

        o = rearrange(o, 'b h n d -> b n (h d)')
        return (self.to_out(o), None)

config = server.config()

pre_model = server(config=config).pre_model
hidden_states_model = pre_model.transformer

dim = pre_model.config.n_embd
dim_head = pre_model.config.n_inner if not None else dim 

for block in hidden_states_model.h:
    attn = block.attn
    print("old:", attn)
    block.attn = Attention(dim=dim, heads=pre_model.config.n_head, causal=True, dropout=pre_model.config.attn_pdrop, use_triton=False) # TODO: need to add dropout and maybe projection
    print("new:", block.attn)

tokenizer = bt.tokenizer()
input_ids = tokenizer.encode('hello my name is', return_tensors='pt')
output = pre_model(input_ids)
# model = FlashNucleus(config)
pdb.set_trace()
