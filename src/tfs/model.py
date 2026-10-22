"""The encoder–decoder Transformer of lesson 2, so that later lessons can import it.

The code is the lesson's, class for class and parameter name for parameter name; lesson 2 checks that loading its
weights into this module gives the same outputs. Lesson 5 changes parts of it and measures each change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    """softmax(q kᵀ / √d_k + mask) v, and the weights. Shapes (..., T_q, d_k), (..., T_k, d_k), (..., T_k, d_v)."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def padding_mask(ids, pad_id):
    """0 for real tokens and -inf for padding, shaped (B, 1, 1, T) to broadcast over heads and queries."""
    mask = torch.zeros(ids.shape, device=ids.device)
    mask[ids == pad_id] = float("-inf")
    return mask[:, None, None, :]


def causal_mask(length, device=None):
    """-inf above the diagonal: position t may look at positions up to t. Shape (1, 1, T, T)."""
    mask = torch.full((length, length), float("-inf"), device=device).triu(diagonal=1)
    return mask[None, None, :, :]


def positional_encoding(max_len, d_model):
    """The sinusoids of the 2017 paper, shape (max_len, d_model)."""
    positions = torch.arange(max_len).unsqueeze(1)
    frequencies = 10_000 ** (-torch.arange(0, d_model, 2) / d_model)
    angles = positions * frequencies
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.weights = None  # the last attention weights, kept for plotting

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x_query, x_key_value, mask=None):
        q = self.split_heads(self.w_q(x_query))
        k = self.split_heads(self.w_k(x_key_value))
        v = self.split_heads(self.w_v(x_key_value))
        out, self.weights = attention(q, k, v, mask)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, pad_id, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        self.register_buffer("pe", positional_encoding(max_len, d_model))
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.d_model) + self.pe[: ids.shape[1]]
        return self.dropout(x)

    def encode(self, src):
        src_mask = padding_mask(src, self.pad_id)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src):
        src_mask = padding_mask(src, self.pad_id)
        tgt_mask = padding_mask(tgt, self.pad_id) + causal_mask(tgt.shape[1], tgt.device)
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x @ self.embedding.weight.T  # the same matrix E turns vectors into scores for every token

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)
