# The encoder-decoder Transformer of lessons 2 and 5, with only the design the published
# model uses: pre-norm with RMSNorm, SwiGLU, RoPE. The parameter names are the lessons', so
# lesson 6's weights load as they are.

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_angles(max_len, d_k, base=10_000):
    theta = base ** (-torch.arange(0, d_k, 2) / d_k)
    positions = torch.arange(max_len)
    return positions[:, None] * theta[None, :]


def apply_rope(x, angles):
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x32 = x.float()
        mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(mean_square + self.eps)
        return (normed * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        width = 8 * round(2 * d_ff / 3 / 8)  # three matrices with the weights of two
        self.w_gate = nn.Linear(d_model, width, bias=False)
        self.w_up = nn.Linear(d_model, width, bias=False)
        self.w_down = nn.Linear(width, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

    def keys_values(self, x, angles=None):
        k, v = self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        if angles is not None:  # RoPE in self-attention; cross-attention has no positions
            k = apply_rope(k, angles)
        return k, v

    def forward(self, x_query, x_key_value, mask, angles=None, cache=None):
        q = self.split_heads(self.w_q(x_query))
        if angles is not None:
            q = apply_rope(q, angles)
        if cache is not None and "fixed" in cache:  # the source's keys and values, once
            k, v = cache["k"], cache["v"]
        else:
            k, v = self.keys_values(x_key_value, angles)
            if cache is not None:  # the keys and values of earlier steps, then the new ones
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        batch, _, length, _ = out.shape
        return self.w_o(out.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.self_attention = Attention(config.d_model, config.n_heads)
        self.cross_attention = Attention(config.d_model, config.n_heads) if cross else None
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        if cross:
            self.norm3 = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask, angles, memory=None, memory_mask=None, cache=None):
        own = cache["self"] if cache is not None else None
        h = self.norm1(x)
        x = x + self.dropout(self.self_attention(h, h, mask, angles, own))
        if self.cross_attention is None:
            return x + self.dropout(self.feed_forward(self.norm2(x)))
        other = cache["cross"] if cache is not None else None
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attention(h, memory, memory_mask, cache=other))
        return x + self.dropout(self.feed_forward(self.norm3(x)))


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        d_k = config.d_model // config.n_heads
        self.register_buffer("angles", rope_angles(config.max_len, d_k))
        self.encoder = nn.ModuleList([Block(config, False) for _ in range(config.n_layers)])
        self.decoder = nn.ModuleList([Block(config, True) for _ in range(config.n_layers)])
        self.encoder_norm = RMSNorm(config.d_model)
        self.decoder_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def embed(self, ids):
        return self.dropout(self.embedding(ids) * math.sqrt(self.config.d_model))

    def encode(self, src):
        mask = (src != self.config.pad_id)[:, None, None, :]
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, mask, self.angles[: src.shape[1]])
        return self.encoder_norm(x)

    def decode(self, tgt, memory, src):
        length = tgt.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
        mask = causal[None, None] & (tgt != self.config.pad_id)[:, None, None, :]
        memory_mask = (src != self.config.pad_id)[:, None, None, :]
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, mask, self.angles[:length], memory, memory_mask)
        return self.decoder_norm(x) @ self.embedding.weight.T

    def new_caches(self, memory):
        # one per decoder layer: an empty self-attention cache, and the cross-attention's
        # keys and values, which depend only on the source
        caches = []
        for layer in self.decoder:
            k, v = layer.cross_attention.keys_values(memory)
            caches.append({"self": {}, "cross": {"fixed": True, "k": k, "v": v}})
        return caches

    def decode_step(self, token, memory_mask, caches, position):
        # scores for the next token from the newest token alone: the rest is in the caches
        x = self.embed(token)
        angles = self.angles[position : position + 1]
        for layer, cache in zip(self.decoder, caches, strict=True):
            x = layer(x, None, angles, None, memory_mask, cache)
        return (self.decoder_norm(x) @ self.embedding.weight.T)[:, -1]

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)
