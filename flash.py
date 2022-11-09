import torch
import torch.nn as nn

import math

import bittensor as bt
from bittensor._neuron.text.core_server.nucleus_impl import server
# from bittensor.utils.flash_attention import attention_ref

from einops import rearrange

import argparse
import pdb

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def attention_ref(q, k, v):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    attention = torch.softmax(scores, dim=-1)

    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads
        d_head = self.dim_head
        BATCH = x.shape[0]
        N_CTX = x.shape[1]
        H = h
        D_HEAD = d_head
        # dtype = x.dtype
        in_dtype = torch.float16
        out_dtype = torch.float32

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))


        # BATCH = x.shape[0]
        # H is self.heads
        # SEQ_LEN is x.shape[1]
        # DIM_HEAD is x.shape[2]

        # reshape q, k, v to (BATCH, H, N_CTX, D_HEAD)
        query = q.reshape(x.shape[0], h, x.shape[1], d_head)
        k = k.reshape(x.shape[0], h, x.shape[1], d_head)
        v = v.reshape(x.shape[0], h, x.shape[1], d_head)

        # # cast to float16
        query = query.to(in_dtype)
        k = k.to(in_dtype)
        v = v.to(in_dtype)

        # # einsum transform q, k, v to (BATCH, H, N_CTX, N_CTX)

        out = attention_ref(query, k, v, self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')
  
        # # out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # # cast to float32
        out = out.to(out_dtype)
        # # pdb.set_trace()

        out = self.to_out(out)

        return out

# class FlashNucleus(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.pre_model = server(config=config).pre_model
#         self.hidden_states_model = self.pre_model.transformer

#         for block in self.hidden_states_model.h:
#             attn = block.attn
#             print(attn)

#     def forward(self, inputs, targets):
#         hidden_states = self.hidden_states_model(inputs)
#         attention = self.attention(hidden_states)
#         outputs = self.post_model(attention)
#         return outputs


config = server.config()

pre_model = server(config=config).pre_model
hidden_states_model = pre_model.transformer

for block in hidden_states_model.h:
    attn = block.attn
    print("old:", attn)
    block.attn = Attention(dim=pre_model.dim, dim_head=pre_model.dim_head, heads=pre_model.heads, causal=True, dropout=pre_model.dropout, use_triton=False)
    print("new:", block.attn)

# model = FlashNucleus(config)
pdb.set_trace()