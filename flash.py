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

# def FeedForward(
#     dim, 
#     mult = 4, 
#     pre_norm = False,
#     layer_past: Optional[Tuple[torch.Tensor]] = None,
#     attention_mask: Optional[torch.FloatTensor] = None,
#     head_mask: Optional[torch.FloatTensor] = None,
#     encoder_hidden_states: Optional[torch.Tensor] = None,
#     encoder_attention_mask: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = False,
#     output_attentions: Optional[bool] = False,
#     ):
#     dim_hidden = int(dim * mult)
#     return nn.Sequential(
#         nn.LayerNorm(dim) if pre_norm else nn.Identity(),
#         nn.Linear(dim, dim_hidden, bias = False),
#         nn.GELU(),
#         nn.Linear(dim_hidden, dim, bias = False)
#     )
    

class Sequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

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
        for layer in self.layers:
            x = layer(x)
        return (x, None)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, pre_norm = False):
        super().__init__()
        dim_hidden = int(dim * mult)
        self.net = nn.Sequential(
            LayerNorm(dim) if pre_norm else nn.Identity(),
            nn.Linear(dim, dim_hidden, bias = False),
            nn.GELU(),
            nn.Linear(dim_hidden, dim, bias = False)
        )
    
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
        return self.net(x)
    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

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
        # pdb.set_trace()
        # x = x[0]
        y = F.layer_norm(x, x.shape[-1:], self.g, self.b, self.eps)
        # y = (y, None)
        # pdb.set_trace()
        return y

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
        old_attn=None,
        dim_head = 64, # TODO: find out what the dim is from HF config
        heads = 8,
        scale = 8,
        l2norm_groups = 1,
        singleton = True,
        pre_norm = False,
        use_cuda_kernel = False,
        non_cosine_sim_attn = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads

        # self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.norm = LayerNorm(dim)

        self.singleton = singleton
        self.scale = scale
        self.heads = heads

        self.l2norm_groups = l2norm_groups

        self.c_attn = old_attn.c_attn
        self.c_proj = old_attn.c_proj
        self.split_size = old_attn.split_size
        self.num_heads = old_attn.num_heads
        self.head_dim = old_attn.head_dim
        self.attn_fn = plain_cosine_sim_attention


        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

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

        query, key, value = self.c_attn(x).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self.attn_fn(query, key, value, causal = True, scale = scale, groups = l2norm_groups)

        o = rearrange(o, 'b h n d -> b n (h d)')
        o = self.c_proj(o)

        if self.singleton:
            return (self.to_out(o), None)
        else:
            return self.to_out(o)

        
config = server.config()

unchanged_model = server(config=config)
model = server(config=config)
pre_model = model.pre_model
hidden_states_model = pre_model.transformer


if 'gpt-neo' in config.neuron.model_name:
    dim = pre_model.config.hidden_size
    dim_head = dim
    heads = pre_model.config.num_heads
    attn_dropout = pre_model.config.summary_first_dropout
elif "gpt2" in config.neuron.model_name:
    dim = pre_model.config.n_embd
    dim_head = pre_model.config.n_inner if not None else dim 
    heads = pre_model.config.n_head
    attn_dropout = pre_model.config.attn_pdrop
else:
    raise ValueError("Model name not supported")

for idx, block in enumerate(hidden_states_model.h):
    print(f"{idx=}")
    #layer = Sequential(
        #Attention(dim=dim, heads=heads, causal=True, dropout=attn_dropout, use_triton=False),
        # nn.LayerNorm(dim),
        # FeedForward(dim=dim, pre_norm=False),
        # nn.LayerNorm(dim)
    #)
    attn = block.attn

    print("old:", attn)
    # block.attn = layer
    new_attn = Attention(dim=dim, old_attn=attn,
                         heads=heads, causal=True,
                         dropout=attn_dropout, use_triton=False, ) # TODO: need to add dropout and maybe projection
    block.attn = new_attn
    print("new:", block.attn)

tokenizer = bt.tokenizer()
input_ids = tokenizer.encode('In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.', return_tensors='pt')
reg_output = unchanged_model(input_ids)
attn_output = model(input_ids)
attn_decoded_inputs = tokenizer.decode(attn_output[1].argmax(-1)[0])
reg_decoded_inputs = tokenizer.decode(reg_output[1].argmax(-1)[0])

print("reg: ", reg_output)
print("attn: ", attn_output)
# model = FlashNucleus(config)
pdb.set_trace()
