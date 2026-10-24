"""The encoder–decoder Transformer of lesson 5: lesson 2's model with Llama's changes behind switches.

`setup` chooses the design: "norm" is "post" (2017), "pre" (pre-norm with LayerNorm) or "rms" (pre-norm with
RMSNorm); "ffn" is "relu" or "swiglu"; "pos" is "sinusoidal" or "rope". The code and the parameter names are the
lesson's, so lesson 5 can check that this module gives exactly its outputs, and the 2017 setting loads lesson 3's
weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import positional_encoding


def attention(q, k, v, mask=None):
    """Lesson 2's attention with a boolean mask: True where a position may be seen."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


def residual(x, sublayer, norm, dropout, pre_norm):
    if pre_norm:
        return x + dropout(sublayer(norm(x)))
    return norm(x + dropout(sublayer(x)))


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
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def swiglu_width(d_ff, multiple=8):
    """Two thirds of the ReLU layer's width, so that three matrices hold as many weights as two."""
    return multiple * round(2 * d_ff / 3 / multiple)


def rope_angles(max_len, d_k, base=10_000):
    """The angle m·θ_i for every position m and pair i: (max_len, d_k / 2)."""
    theta = base ** (-torch.arange(0, d_k, 2) / d_k)
    positions = torch.arange(max_len)
    return positions[:, None] * theta[None, :]


def apply_rope(x, angles):
    """Rotate the pairs (0, 1), (2, 3), ... of every query or key, x: (B, H, T, d_k), by their angles."""
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, fused=True):
        super().__init__()
        self.n_heads, self.d_k, self.fused = n_heads, d_model // n_heads, fused
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

    def keys_values(self, x, angles=None):
        k, v = self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        if angles is not None:
            k = apply_rope(k, angles)
        return k, v

    def forward(self, x_query, x_key_value, mask, angles=None, cache=None):
        q = self.split_heads(self.w_q(x_query))
        if angles is not None:
            q = apply_rope(q, angles)
        if cache is not None and cache.get("fixed"):  # cross-attention: computed once
            k, v = cache["k"], cache["v"]
        else:
            k, v = self.keys_values(x_key_value, angles)
            if cache is not None:  # self-attention: the past keys and values, plus the new ones
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        if self.fused:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = attention(q, k, v, mask)
        batch, _, length, _ = out.shape
        return self.w_o(out.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, cross, setup):
        super().__init__()
        self.pre_norm = setup["norm"] != "post"
        Norm = RMSNorm if setup["norm"] == "rms" else nn.LayerNorm
        fused = setup.get("fused", True)
        self.self_attention = Attention(d_model, n_heads, fused)
        self.cross_attention = Attention(d_model, n_heads, fused) if cross else None
        if setup["ffn"] == "swiglu":
            self.feed_forward = SwiGLU(d_model, swiglu_width(d_ff))
        else:
            self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = Norm(d_model), Norm(d_model)
        if cross:
            self.norm3 = Norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, angles, memory=None, memory_mask=None, cache=None):
        def sub(x, sublayer, norm):
            return residual(x, sublayer, norm, self.dropout, self.pre_norm)

        own = cache["self"] if cache else None
        x = sub(x, lambda h: self.self_attention(h, h, mask, angles, own), self.norm1)
        if self.cross_attention is None:
            return sub(x, self.feed_forward, self.norm2)
        other = cache["cross"] if cache else None
        x = sub(x, lambda h: self.cross_attention(h, memory, memory_mask, cache=other), self.norm2)
        return sub(x, self.feed_forward, self.norm3)


class Seq2Seq(nn.Module):
    def __init__(
        self, setup, vocab_size, d_model=256, n_heads=4, d_ff=1024, n_layers=3, dropout=0.1, pad_id=0, max_len=256
    ):
        super().__init__()
        self.setup, self.d_model, self.pad_id = setup, d_model, pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        if setup["pos"] == "sinusoidal":
            self.register_buffer("pe", positional_encoding(max_len, d_model))
        else:
            self.register_buffer("angles", rope_angles(max_len, d_model // n_heads))
        args = (d_model, n_heads, d_ff, dropout)
        self.encoder = nn.ModuleList([Block(*args, False, setup) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([Block(*args, True, setup) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        if setup["norm"] != "post":
            Norm = RMSNorm if setup["norm"] == "rms" else nn.LayerNorm
            self.encoder_norm, self.decoder_norm = Norm(d_model), Norm(d_model)

    def embed(self, ids, start=0):
        x = self.embedding(ids) * math.sqrt(self.d_model)
        if self.setup["pos"] == "sinusoidal":
            x = x + self.pe[start : start + ids.shape[1]]
        return self.dropout(x)

    def rope(self, length, start=0):
        if self.setup["pos"] != "rope":
            return None
        return self.angles[start : start + length]

    def encode(self, src):
        mask = (src != self.pad_id)[:, None, None, :]
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, mask, self.rope(src.shape[1]))
        return self.encoder_norm(x) if self.setup["norm"] != "post" else x

    def decode(self, tgt, memory, src):
        length = tgt.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
        mask = causal[None, None] & (tgt != self.pad_id)[:, None, None, :]
        memory_mask = (src != self.pad_id)[:, None, None, :]
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, mask, self.rope(length), memory, memory_mask)
        if self.setup["norm"] != "post":
            x = self.decoder_norm(x)
        return x @ self.embedding.weight.T

    def decode_step(self, token, memory, src, caches, position):
        """Scores for the next token of every sentence, (B, |V|), from the newest token only and the caches."""
        memory_mask = (src != self.pad_id)[:, None, None, :]
        x = self.embed(token, start=position)
        angles = self.rope(1, start=position)
        for layer, cache in zip(self.decoder, caches, strict=True):
            x = layer(x, None, angles, memory, memory_mask, cache)
        if self.setup["norm"] != "post":
            x = self.decoder_norm(x)
        return (x @ self.embedding.weight.T)[:, -1]

    def new_caches(self, memory):
        """Empty self-attention caches and the cross-attention keys and values, one per decoder layer."""
        caches = []
        for layer in self.decoder:
            k, v = layer.cross_attention.keys_values(memory)
            caches.append({"self": {}, "cross": {"fixed": True, "k": k, "v": v}})
        return caches

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)
