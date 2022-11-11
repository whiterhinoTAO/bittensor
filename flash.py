from time import time
import torch
from torch import nn, einsum
from torch.nn import functional as F
from tqdm import tqdm

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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        max_positions=1024,
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
        # self.norm = LayerNorm(dim)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.singleton = singleton
        self.scale = scale
        self.heads = heads

        self.l2norm_groups = l2norm_groups

        self.c_attn = old_attn.c_attn
        self.c_proj = old_attn.c_proj
        self.split_size = old_attn.split_size
        self.num_heads = old_attn.num_heads
        self.head_dim = old_attn.head_dim


        # self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # self.to_k = nn.Linear(dim, inner_dim, bias = False)
        # self.to_v = nn.Linear(dim, inner_dim, bias = False)
        # self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.attn_dropout = nn.Dropout(0.1) # TODO: change this from 0.1 to config.attn_pdrop that works with any kodel
        self.resid_dropout = nn.Dropout(0.1) #TODO: change this from 0.1 to config.resid_pdrop that works with any kodel

    def _attn(
        self,
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
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        if is_merged_batch_heads_query:
            out = out.squeeze(1)

        return out

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

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

        # x = self.norm(x)

        # q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        query, key, value = self.c_attn(x).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self._attn(query, key, value, causal = True, scale = scale, groups = l2norm_groups)

        o = rearrange(o, 'b h n d -> b n (h d)')

        # self._merge_heads(o, self.num_heads, self.head_dim)
        o = self.c_proj(o)
        o = self.resid_dropout(o)
        # pdb.set_trace()

        if self.singleton:
            o = (o, None)
        return o

        # dropout layer for attention
        # o = self.attn_drop(o)


        # if self.singleton:
        #     return (self.to_out(o), None)
        # else:
        #     return self.to_out(o)
            

        
config = server.config()

unchanged_model = server(config=config)
model = server(config=config)
pre_model = model.pre_model
hidden_states_model = pre_model.transformer


if 'gpt-neo' in config.neuron.model_name:
    dim = pre_model.config.hidden_size
    dim_head = dim
    heads = pre_model.config.num_heads
    max_positions = pre_model.config.max_position_embeddings
    attn_dropout = pre_model.config.summary_first_dropout
elif "gpt2" in config.neuron.model_name:
    dim = pre_model.config.n_embd
    dim_head = pre_model.config.n_inner if not None else dim 
    heads = pre_model.config.n_head
    attn_dropout = pre_model.config.attn_pdrop
    max_positions = pre_model.config.n_positions
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
                         dropout=attn_dropout, use_triton=False,
                         max_positions=max_positions ) # TODO: need to add dropout and maybe projection
    old_state_dict = attn.state_dict()
    # pdb.set_trace()
    new_attn.load_state_dict(old_state_dict, strict=False)
    block_attn = new_attn
    print("new:", block.attn)


runs = 100
tokenizer = bt.tokenizer()
input_ids = tokenizer.encode('In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.', return_tensors='pt')

t = time()
for _ in tqdm(range(runs), desc="reg"):
    reg_output = unchanged_model(input_ids)
t_reg = time() - t

t = time()
for _ in tqdm(range(runs), desc="flash"):
    attn_output = model(input_ids)
t_attn = time() - t

print(f"Reg time: {t_reg/runs:3f}")
print(f"Attn time: {t_attn/runs:3f}")

attn_decoded_inputs = tokenizer.decode(attn_output[1].argmax(-1)[0])
reg_decoded_inputs = tokenizer.decode(reg_output[1].argmax(-1)[0])

print("reg: ", reg_output)
print("attn: ", attn_output)
# model = FlashNucleus(config)
pdb.set_trace()
