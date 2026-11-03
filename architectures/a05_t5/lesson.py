# %% [markdown]
# # Lesson 5. T5 and BART: the encoder–decoder, and a model that writes SQL
#
# @@HOOK@@
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a05/hero-light.png" width="760" alt="A question and the database's schema go into T5's encoder, which reads all of it at once. The decoder writes an SQL query one token at a time, looking back at the encoder's output through cross-attention. The query is then run on the SQLite database, and its result is compared with the result of the correct query">
#
# **What you'll do**
#
# 1. See how T5 turns every task into text in, text out, and make the smallest T5 translate, summarise and judge
#    grammar from a prefix.
# 2. Take the encoder–decoder apart: an encoder that reads the whole input, a decoder that writes, and the
#    cross-attention that connects them.
# 3. Write T5 in PyTorch, with its unusual parts: a norm without a mean, attention without scaling and positions
#    as 32 learned biases. Load Google's weights into it and match Hugging Face's outputs.
# 4. Look at what the position buckets learned, and watch the decoder find the source words while it translates.
# 5. Corrupt text the way T5 and BART were pre-trained, and let both models repair it.
# 6. Teach T5 to write SQL for 200 databases it has never seen, and check every query by running it.
#
# **What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type →
# T4 GPU*. The notebook downloads t5-small and t5-base (1.1 GB), BART (560 MB) and the Spider databases (200 MB),
# and fine-tunes T5 for @@MINUTES@@ minutes; on a T4 it runs in about @@TOTAL@@ minutes.

# %% tags=["include:setup"]

# %%
import json
import math
import random
import sqlite3
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib.colors import LinearSegmentedColormap
from safetensors.torch import load_file
from tfs.viz import heatmap
from transformers import AutoTokenizer
# the reference implementations, for the checks
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration

SMOKE = env.smoke()  # CI runs the lesson on a small slice; you run it in full
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" or SMOKE, "This lesson fine-tunes a model for 20 minutes: use a GPU"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Every task as text
#
# @@T5_STORY@@
#
# The smallest T5 still knows its prefixes. Four of them, four tasks, one model with no task-specific layer:

# %%
T5_SMALL = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(T5_SMALL)
reference = T5ForConditionalGeneration.from_pretrained(T5_SMALL).to(device).eval()

prompts = [
    "translate English to German: The guards arrive at the museum at seven in the morning.",
    "translate English to French: The night train to Vienna leaves at eleven.",
    "cola sentence: The guards arrives at the museum at seven.",
    "stsb sentence1: The night train leaves at eleven. "
    "sentence2: At eleven the night train departs.",
    "stsb sentence1: The night train leaves at eleven. "
    "sentence2: The museum opens at nine.",
]
batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    answers = reference.generate(**batch, max_new_tokens=40, num_beams=1, do_sample=False)
answers = tokenizer.batch_decode(answers, skip_special_tokens=True)
table(pd.DataFrame({"prompt": prompts, "answer": answers}))

# %% [markdown]
# @@PREFIX_COMMENT@@

# %% [markdown]
# ## 2. Two stacks and a bridge
#
# BERT, in lesson 1, was an encoder: it read a whole text at once, every token looking at every other, and turned it
# into vectors. GPT, in lesson 4, was a decoder: it wrote text one token at a time, each token looking only at the
# ones before it. The original Transformer of 2017 was both, because it was built for translation, where the
# input and the output are different texts. T5 and BART went back to that design.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a05/encdec-light.png" width="760" alt="The encoder–decoder. On the left the encoder: a stack of layers with self-attention that sees the whole input in both directions and a feed-forward network; it runs once and turns the input into a sequence of vectors. On the right the decoder: in every layer causal self-attention over what it has written so far, then cross-attention, whose queries come from the decoder and whose keys and values come from the encoder's output, then a feed-forward network. Only the decoder runs once per generated token">
#
# The encoder turns the $S$ input tokens into a matrix $H \in \mathbb{R}^{S \times d}$, once. The decoder has, in
# every layer, one part that neither BERT nor GPT had: *cross-attention*. It is ordinary attention with the
# queries taken from the decoder's own stream $Y \in \mathbb{R}^{T \times d}$ and the keys and values taken from
# the encoder's output:
#
# $$\text{CrossAttention}(Y, H) = \text{softmax}\!\left(\frac{(Y W_Q)(H W_K)^\top}{\sqrt{d_k}}\right) H W_V.$$
#
# The matrix of scores is $T \times S$: each of the $T$ positions being written looks at all $S$ positions of the
# input, and no mask is needed, since the input is complete before the first word is written. Self-attention in the
# decoder stays causal, as in GPT.
#
# Two consequences follow, and both matter for the rest of this lesson. The input is read with full context in
# both directions, as BERT reads, and it is read once: generating $T$ tokens runs the encoder once and the decoder
# $T$ times. And the parameters are split between two stacks: t5-small has 6 encoder layers and 6 decoder
# layers, so a token of output passes through only half of the model.

# %% [markdown]
# ## 3. T5 in PyTorch
#
# T5's layer has the familiar shape, pre-norm with a residual around each part, as in GPT-2. Four details differ,
# and each is a decision the T5 paper made on purpose.
#
# **A norm without a mean.** LayerNorm subtracts the mean of a vector, divides by the standard deviation, scales and
# shifts. T5's norm only rescales:
#
# $$\text{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}}.$$
#
# Here $g$ is a learned vector of scales and $\varepsilon = 10^{-6}$ guards against division by zero. No mean is
# subtracted and no bias is added. The idea has its own name and paper, RMSNorm
# ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)), which argues that the rescaling, not the
# centring, is what makes LayerNorm work, and it is the norm of Llama, Qwen and nearly every model since. PyTorch has
# it built in, which makes for an easy check:

# %%
class Norm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


x = torch.randn(2, 5, 8)
mine, builtin = Norm(8, 1e-6), nn.RMSNorm(8, eps=1e-6)
print("largest difference from nn.RMSNorm:", (mine(x) - builtin(x)).abs().max().item())

# %% [markdown]
# **Attention without $\sqrt{d_k}$.** T5 drops the division by $\sqrt{d_k}$ in the scores. Nothing is lost: the
# scale is folded into the initialisation of the query matrix, @@SCALE_SOURCE@@ and a transplant has to leave the
# division out, or every score comes out eight times too small.
#
# **Positions as a bias.** T5 has no position embeddings at all. Instead, each attention head adds a learned number
# to each score, which depends only on how far the key is from the query:
#
# $$s_{ij} = q_i \cdot k_j + b_h\big(\text{bucket}(j - i)\big).$$
#
# The distances are sorted into 32 buckets. In the encoder, which looks both ways, half of the buckets are for keys
# to the left and half for keys to the right. Within each half, the first 8 distances get a bucket each, and the rest
# share the remaining 8 buckets on a logarithmic scale up to a distance of 128; anything farther falls into the last
# bucket. Near words get precise positions, far ones only a rough sense of "far". The decoder looks only back, so it
# uses all 32 buckets for distances to the left.

# %%
def position_bucket(relative, bidirectional, buckets=32, max_distance=128):
    bucket = torch.zeros_like(relative)
    if bidirectional:
        buckets = buckets // 2
        # keys to the right of the query get the upper half of the buckets
        bucket = bucket + (relative > 0).long() * buckets
        distance = relative.abs()
    else:
        distance = (-relative).clamp(min=0)  # the decoder only looks back
    exact = buckets // 2
    log_distance = torch.log(distance.clamp(min=1).float() / exact)
    scaled = log_distance / math.log(max_distance / exact)
    far = exact + (scaled * (buckets - exact)).long()
    far = far.clamp(max=buckets - 1)
    return bucket + torch.where(distance < exact, distance, far)


relative = torch.tensor([-300, -128, -64, -20, -9, -8, -7, -1,
                         0, 1, 7, 8, 20, 64, 128, 300])
table(pd.DataFrame({"key minus query": relative.tolist(),
                    "encoder bucket": position_bucket(relative, True).tolist(),
                    "decoder bucket": position_bucket(relative, False).tolist()}))

# %% [markdown]
# A distance of 1 and a distance of 7 get different buckets; 64 and 100 share one. Every key to the right of a
# decoder query lands in bucket 0 with the query itself, but the causal mask hides those keys anyway. The whole map,
# for distances up to 150 either way:

# %%
distances = torch.arange(-150, 151)
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.step(distances, position_bucket(distances, True), where="mid", color=COLORS["enc"],
        label="encoder: both directions")
ax.step(distances, position_bucket(distances, False), where="mid", color=COLORS["dec"],
        label="decoder: only back")
ax.axvspan(-7.5, 7.5, color=COLORS["grid"], zorder=0)
ax.text(0, 33.5, "one bucket per distance", ha="center", color=COLORS["muted"], fontsize=9)
ax.set_xlabel("key position minus query position")
ax.set_ylabel("bucket")
ax.set_ylim(-1, 36)
ax.legend(loc="lower right", frameon=False)
plt.show()

# %% [markdown]
# The table of biases, 32 buckets by the number of heads, lives in the first layer of the encoder and the first layer
# of the decoder. Every other layer reuses the same biases, so a model of six layers learns positions with
# $2 \times 32 \times 8 = 512$ numbers. GPT-2 spent 786,432 on its position embeddings.
#
# **The output layer.** T5's decoder scores the vocabulary with the input embedding matrix, tied as in GPT-2, after
# multiplying the hidden state by $d^{-1/2}$. @@TIE_SOURCE@@
#
# The attention layer below takes an optional `source`: without it, it is self-attention; with it, the keys and
# values come from the encoder and it is cross-attention. The optional `cache` is the KV cache of lesson 4, with one
# addition: in cross-attention the keys and values come from the encoder, which doesn't change while the decoder
# writes, so they are computed once for the whole answer. For the pictures of section 5 it can also return the
# attention weights.

# %%
class Attention(nn.Module):
    def __init__(self, config, relative=False, bidirectional=True):
        super().__init__()
        d = config["d_model"]
        self.heads, self.head_dim = config["num_heads"], config["d_kv"]
        inner = self.heads * self.head_dim
        self.q = nn.Linear(d, inner, bias=False)
        self.k = nn.Linear(d, inner, bias=False)
        self.v = nn.Linear(d, inner, bias=False)
        self.o = nn.Linear(inner, d, bias=False)
        self.buckets = config["relative_attention_num_buckets"]
        self.bidirectional = bidirectional
        if relative:  # only the first layer has the table; the others reuse its biases
            self.relative_bias = nn.Embedding(self.buckets, self.heads)

    def position_bias(self, queries, keys):
        relative = keys[None, :] - queries[:, None]
        buckets = position_bucket(relative, self.bidirectional, self.buckets)
        return self.relative_bias(buckets).permute(2, 0, 1).unsqueeze(0)  # 1, heads, q, k

    def split(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, source=None, bias=None, cache=None, keep_weights=False):
        q = self.split(self.q(x))
        if source is not None and cache is not None and "k" in cache:
            # cross-attention: the encoder's keys and values never change
            k, v = cache["k"], cache["v"]
        else:
            k = self.split(self.k(x if source is None else source))
            v = self.split(self.v(x if source is None else source))
            if source is None and cache is not None and "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        weights = None
        if keep_weights:
            scores = q @ k.transpose(-2, -1) + bias  # no division by the square root of d_k
            weights = scores.float().softmax(dim=-1).to(q.dtype)
            y = weights @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.o(y), weights


# %% [markdown]
# The feed-forward network of the original T5 is two matrices with a ReLU between them and no biases; there are no
# biases anywhere in T5. Dropout only matters for training, in section 8.

# %%
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config["d_model"], config["d_ff"], bias=False)
        self.wo = nn.Linear(config["d_ff"], config["d_model"], bias=False)
        self.drop = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        return self.wo(self.drop(F.relu(self.wi(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        d, eps = config["d_model"], config["layer_norm_epsilon"]
        self.norm1, self.attention = Norm(d, eps), Attention(config, relative=first)
        self.norm2, self.ff = Norm(d, eps), FeedForward(config)
        self.drop = nn.Dropout(config["dropout_rate"])

    def forward(self, x, bias):
        x = x + self.drop(self.attention(self.norm1(x), bias=bias)[0])
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        d, eps = config["d_model"], config["layer_norm_epsilon"]
        self.norm1 = Norm(d, eps)
        self.attention = Attention(config, relative=first, bidirectional=False)
        self.norm2, self.cross = Norm(d, eps), Attention(config)
        self.norm3, self.ff = Norm(d, eps), FeedForward(config)
        self.drop = nn.Dropout(config["dropout_rate"])

    def forward(self, y, memory, self_bias, cross_bias, cache=None, keep_weights=False):
        own = cache["self"] if cache is not None else None
        other = cache["cross"] if cache is not None else None
        y = y + self.drop(self.attention(self.norm1(y), bias=self_bias, cache=own)[0])
        out, weights = self.cross(self.norm2(y), source=memory, bias=cross_bias,
                                  cache=other, keep_weights=keep_weights)
        y = y + self.drop(out)
        return y + self.drop(self.ff(self.norm3(y))), weights


# %% [markdown]
# The model puts the stacks together. The encoder's padding is hidden by a bias of $-\infty$, in practice the
# smallest float, on the padded keys, both in the encoder and in cross-attention; the decoder adds the causal mask
# to its position biases. `decode` takes `past`, the number of tokens already written, so that a single new token
# gets the right position biases when the earlier ones are in the cache.

# %%
def padding_bias(mask):
    blocked = torch.finfo(torch.float32).min
    return (1.0 - mask[:, None, None, :].float()) * blocked  # batch, 1, 1, keys


class T5(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, eps = config["d_model"], config["layer_norm_epsilon"]
        self.d = d
        self.embed = nn.Embedding(config["vocab_size"], d)
        layers = config["num_layers"]
        self.encoder = nn.ModuleList(EncoderLayer(config, i == 0) for i in range(layers))
        self.encoder_norm = Norm(d, eps)
        layers = config.get("num_decoder_layers", config["num_layers"])
        self.decoder = nn.ModuleList(DecoderLayer(config, i == 0) for i in range(layers))
        self.decoder_norm = Norm(d, eps)
        self.drop = nn.Dropout(config["dropout_rate"])

    def encode(self, ids, mask):
        positions = torch.arange(ids.shape[1], device=ids.device)
        bias = self.encoder[0].attention.position_bias(positions, positions)
        bias = bias + padding_bias(mask)
        x = self.drop(self.embed(ids))
        for layer in self.encoder:
            x = layer(x, bias)
        return self.drop(self.encoder_norm(x))

    def decode(self, targets, memory, mask, caches=None, past=0, keep_weights=False):
        length = targets.shape[1]
        queries = torch.arange(past, past + length, device=targets.device)
        keys = torch.arange(past + length, device=targets.device)
        self_bias = self.decoder[0].attention.position_bias(queries, keys)
        future = keys[None, :] > queries[:, None]
        self_bias = self_bias.masked_fill(future, torch.finfo(torch.float32).min)
        cross_bias = padding_bias(mask)
        y = self.drop(self.embed(targets))
        cross = []
        for i, layer in enumerate(self.decoder):
            cache = caches[i] if caches is not None else None
            y, weights = layer(y, memory, self_bias, cross_bias, cache, keep_weights)
            cross.append(weights)
        y = self.drop(self.decoder_norm(y))
        logits = (y * self.d ** -0.5) @ self.embed.weight.T  # tied, and rescaled first
        return (logits, cross) if keep_weights else logits


# %% [markdown]
# ## 4. Google's weights in our code
#
# The checkpoint's names follow the structure of Google's code, `block.{i}.layer.{j}`, where the parts of a layer
# are numbered: in the decoder, 0 is self-attention, 1 cross-attention and 2 the feed-forward network. We write down
# which goes where; `load_state_dict` refuses anything missing or left over.

# %%
def transplant(model, state):
    mine = {"embed.weight": state["shared.weight"],
            "encoder_norm.weight": state["encoder.final_layer_norm.weight"],
            "decoder_norm.weight": state["decoder.final_layer_norm.weight"]}
    for i in range(len(model.encoder)):
        old, new = f"encoder.block.{i}.layer.", f"encoder.{i}."
        for name in "qkvo":
            weight = state[old + f"0.SelfAttention.{name}.weight"]
            mine[new + f"attention.{name}.weight"] = weight
        mine[new + "norm1.weight"] = state[old + "0.layer_norm.weight"]
        mine[new + "ff.wi.weight"] = state[old + "1.DenseReluDense.wi.weight"]
        mine[new + "ff.wo.weight"] = state[old + "1.DenseReluDense.wo.weight"]
        mine[new + "norm2.weight"] = state[old + "1.layer_norm.weight"]
    for i in range(len(model.decoder)):
        old, new = f"decoder.block.{i}.layer.", f"decoder.{i}."
        for name in "qkvo":
            weight = state[old + f"0.SelfAttention.{name}.weight"]
            mine[new + f"attention.{name}.weight"] = weight
            weight = state[old + f"1.EncDecAttention.{name}.weight"]
            mine[new + f"cross.{name}.weight"] = weight
        for part in range(3):
            mine[new + f"norm{part + 1}.weight"] = state[old + f"{part}.layer_norm.weight"]
        mine[new + "ff.wi.weight"] = state[old + "2.DenseReluDense.wi.weight"]
        mine[new + "ff.wo.weight"] = state[old + "2.DenseReluDense.wo.weight"]
    for side in ["encoder", "decoder"]:
        key = f"{side}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        mine[f"{side}.0.attention.relative_bias.weight"] = state[key]
    model.load_state_dict(mine)
    return model


def load_t5(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        config = json.load(f)
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    return transplant(T5(config), state), config


t5, t5_config = load_t5(T5_SMALL)
t5 = t5.to(device).eval()
print(f"t5-small: {sum(p.numel() for p in t5.parameters()) / 1e6:.1f} million parameters")

# %% [markdown]
# The check: the same input and the same decoder input through both models, teacher-forced, so every position is
# compared at once. The decoder's first input is the padding token, T5's start-of-output.

# %%
batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
written = tokenizer(answers, return_tensors="pt", padding=True).input_ids.to(device)
# the padding token is also T5's start token
start = torch.zeros(len(prompts), 1, dtype=torch.long, device=device)
decoder_input = torch.cat([start, written[:, :-1]], dim=1)
with torch.no_grad():
    ours = t5.decode(decoder_input, t5.encode(batch.input_ids, batch.attention_mask),
                     batch.attention_mask)
    theirs = reference(**batch, decoder_input_ids=decoder_input).logits
print("largest difference in logits:", (ours - theirs).abs().max().item())

# %% [markdown]
# Generation is lesson 4's loop with an encoder in front: encode the input once, start the decoder from the padding
# token, and keep feeding back the most probable token until every row has written the end-of-text token `</s>`.

# %%
@torch.no_grad()
def generate(model, ids, mask, max_new_tokens=64, eos=1, pad=0):
    memory = model.encode(ids, mask)
    caches = [{"self": {}, "cross": {}} for _ in model.decoder]
    token = torch.full((ids.shape[0], 1), pad, device=ids.device)
    written = []
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for step in range(max_new_tokens):
        logits = model.decode(token, memory, mask, caches, past=step)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        token = token.masked_fill(done[:, None], pad)  # finished rows write padding
        written.append(token)
        done = done | (token[:, 0] == eos)
        if done.all():
            break
    return torch.cat(written, dim=1)


ours = tokenizer.batch_decode(generate(t5, batch.input_ids, batch.attention_mask),
                              skip_special_tokens=True)
print("our generation matches Hugging Face's:", ours == answers)
for text in ours[:2]:
    print(" ", text)

# %% [markdown]
# ## 5. What the positions learned, and where the decoder looks
#
# The 32 biases of each head are all T5 knows about word order, so they are worth a look. Each row below is one of
# the eight heads of t5-small's encoder; each column a distance from the query, from 20 tokens to the left to 20 to
# the right; the color is the bias the head adds to the score of a key at that distance.

# %%
def bias_by_distance(attention, distances):
    zero = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        bias = attention.position_bias(zero, distances.to(device))
    return bias[0, :, 0].cpu()  # heads, distances


distances = torch.arange(-20, 21)
encoder_bias = bias_by_distance(t5.encoder[0].attention, distances)
limit = encoder_bias.abs().max().item()
# negative biases in teal, positive in rose, zero white
ends = [COLORS["enc"], COLORS["card"], COLORS["dec"]]
diverging = LinearSegmentedColormap.from_list("bias", ends)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
image = ax.imshow(encoder_bias, cmap=diverging, vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(0, 41, 5), [str(d) for d in range(-20, 21, 5)])
ax.set_yticks(range(8), [f"head {h}" for h in range(8)])
ax.set_xlabel("key position minus query position")
ax.grid(False)
fig.colorbar(image, ax=ax, label="bias added to the score")
plt.show()

# %% [markdown]
# @@BIAS_COMMENT@@
#
# Cross-attention is where the encoder–decoder earns its name, and it can be watched. We translate the sentence of
# section 1 again with our model, keep the cross-attention weights of every decoder layer, and average the eight
# heads of one layer: each row is a German token being written, each column an English token of the input.

# %%
source = tokenizer([prompts[0]], return_tensors="pt").to(device)
german = generate(t5, source.input_ids, source.attention_mask)
start = torch.zeros(1, 1, dtype=torch.long, device=device)
decoder_input = torch.cat([start, german[:, :-1]], dim=1)
with torch.no_grad():
    memory = t5.encode(source.input_ids, source.attention_mask)
    _, cross = t5.decode(decoder_input, memory, source.attention_mask, keep_weights=True)
# SentencePiece marks a word's start with "▁"; we show it as a space
def readable(ids):
    return [t.replace("▁", " ") for t in tokenizer.convert_ids_to_tokens(ids)]


source_tokens, german_tokens = readable(source.input_ids[0]), readable(german[0])
LAYER = 4
weights = cross[LAYER][0].mean(dim=0).cpu()  # rows: German tokens, columns: English tokens
heatmap(weights, rows=german_tokens, cols=source_tokens, cell=0.3)
plt.show()

# %% [markdown]
# @@CROSS_COMMENT@@

# %% [markdown]
# ## 6. Pre-training by repair: T5 and BART
#
# An encoder–decoder needs a task with an input and an output, and raw text has neither. Both T5 and BART make one
# up the same way: damage the text, give the damaged version to the encoder, and train the decoder to undo the
# damage. They damage it differently.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a05/objectives-light.png" width="760" alt="Two ways to pre-train an encoder–decoder on raw text. T5's span corruption replaces spans of the sentence with sentinel tokens, extra_id_0 and extra_id_1, and the target lists only the dropped spans, each after its sentinel, so the target is short. BART's text infilling replaces each span with a single mask token, which can stand for any number of words, and the target is the whole original sentence">
#
# @@OBJECTIVES_STORY@@
#
# T5's recipe takes three numbers: drop @@DENSITY@@ of the tokens, in spans of @@SPAN@@ tokens on average, at random
# places. Google's code first decides how many tokens to drop and how many spans to cut them into, then splits both
# the dropped and the kept tokens into that many pieces of random length, and interleaves them. Here it is on the
# example sentence from the T5 paper:

# %%
def random_split(total, parts, rng):
    # a random way to write `total` as a sum of `parts` positive whole numbers
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    edges = [0] + cuts + [total]
    return [edges[i + 1] - edges[i] for i in range(parts)]


def span_corruption(tokens, rng, density=0.15, mean_span=3):
    n = len(tokens)
    dropped = min(max(round(n * density), 1), n - 1)
    spans = max(round(dropped / mean_span), 1)
    dropped_lengths = random_split(dropped, spans, rng)
    kept_lengths = random_split(n - dropped, spans, rng)
    source, target, i = [], [], 0
    for k in range(spans):
        source += tokens[i:i + kept_lengths[k]]
        i += kept_lengths[k]
        sentinel = f"<extra_id_{k}>"
        source.append(sentinel)
        target.append(sentinel)
        target += tokens[i:i + dropped_lengths[k]]
        i += dropped_lengths[k]
    target.append(f"<extra_id_{spans}>")  # the last sentinel says: nothing more
    return source, target


paper_sentence = tokenizer.tokenize("Thank you for inviting me to your party last week.")
paper_sentence = [t.replace("▁", " ") for t in paper_sentence]
source, target = span_corruption(paper_sentence, random.Random(3), density=0.3, mean_span=2)
token_chips([source, target], labels=["encoder reads", "decoder writes"],
            strong=lambda row, col, token: token.startswith("<extra"))
plt.show()

# %% [markdown]
# @@CORRUPTION_COMMENT@@
#
# So does a pre-trained T5 repair text? Ask it to fill the spans the T5 paper drops. We use t5-base here, which we
# also need in section 8; the smallest model fills them less reliably.

# %%
T5_BASE = "google-t5/t5-base"
base_reference = T5ForConditionalGeneration.from_pretrained(T5_BASE).to(device).eval()
damaged = ["Thank you <extra_id_0> me to your party <extra_id_1> week.",
           "The guards <extra_id_0> at the museum at <extra_id_1> in the morning.",
           "The night train to <extra_id_0> leaves at eleven and arrives in <extra_id_1> "
           "the next morning."]
batch = tokenizer(damaged, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    repaired = base_reference.generate(**batch, max_new_tokens=16, num_beams=1,
                                       do_sample=False)
rows = []
for text, ids in zip(damaged, repaired):
    spans = tokenizer.decode(ids, skip_special_tokens=False).replace("<pad>", "")
    rows.append({"damaged": text, "t5-base writes": spans[:60]})
table(pd.DataFrame(rows))

# %% [markdown]
# @@REPAIR_COMMENT@@
#
# BART's version is closer to BERT's masked language model, and to what the decoder of lesson 4 does. Each span
# becomes one `<mask>`, whatever its length, and the decoder writes the whole text back, so it also has to decide how
# many words each mask hid:

# %%
BART = "facebook/bart-base"
bart_tokenizer = AutoTokenizer.from_pretrained(BART)
bart = BartForConditionalGeneration.from_pretrained(BART).to(device).eval()
masked = ["Thank you <mask> me to your party <mask> week.",
          "The guards <mask> at the museum at <mask> in the morning.",
          "The night train to <mask> leaves at eleven and arrives in <mask> "
          "the next morning."]
batch = bart_tokenizer(masked, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    repaired = bart.generate(**batch, max_new_tokens=32, num_beams=1, do_sample=False)
written = bart_tokenizer.batch_decode(repaired, skip_special_tokens=True)
table(pd.DataFrame({"masked": masked, "BART writes": written}))

# %% [markdown]
# @@BART_COMMENT@@

# %% [markdown]
# ## 7. Spider: 200 databases and a question for each table
#
# @@SPIDER_STORY@@
#
# The Spider authors distribute the databases as SQLite files; the Hugging Face copy we use is a re-host of the
# official archive, pinned to one revision, under the same CC BY-SA 4.0 licence.

# %%
SPIDER = "HAL-9001/spider-databases"
SPIDER_REVISION = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
archive = hf_hub_download(SPIDER, "spider_data.zip", repo_type="dataset",
                          revision=SPIDER_REVISION)
with zipfile.ZipFile(archive) as z:
    z.extractall(WORK_DIR)
SPIDER_DIR = WORK_DIR / "spider_data"

with open(SPIDER_DIR / "train_spider.json") as f:
    train_set = json.load(f)
with open(SPIDER_DIR / "dev.json") as f:
    dev_set = json.load(f)
with open(SPIDER_DIR / "tables.json") as f:
    schemas = {db["db_id"]: db for db in json.load(f)}
train_dbs = {e["db_id"] for e in train_set}
dev_dbs = {e["db_id"] for e in dev_set}
print(f"training: {len(train_set):,} questions on {len(train_dbs)} databases")
print(f"development: {len(dev_set):,} questions on {len(dev_dbs)} databases")
print("databases in both:", len(train_dbs & dev_dbs))

# %% [markdown]
# The development databases never appear in training: to answer a question about `concert_singer`, a model has to
# read a schema it has never seen. Here is one example, and the table it asks about:

# %%
example = dev_set[0]
print("question:", example["question"])
print("query:   ", example["query"])
database = SPIDER_DIR / "database" / example["db_id"] / (example["db_id"] + ".sqlite")
with sqlite3.connect(database) as connection:
    singers = pd.read_sql_query("SELECT * FROM singer", connection)
table(singers)

# %% [markdown]
# The model gets the question and the schema as one line of text: the database's name, then every table with its
# columns, the names exactly as SQL has to write them. Nothing else: no rows, no types, no keys. That keeps the input
# short, and it means the model has to guess, from names alone, that `Singer_ID` in `singer_in_concert` points to
# `singer`.

# %%
def schema_text(db_id):
    db = schemas[db_id]
    columns = {}
    for table_index, column in db["column_names_original"][1:]:  # the first is "*"
        columns.setdefault(table_index, []).append(column)
    parts = []
    for i, name in enumerate(db["table_names_original"]):
        parts.append(name + " : " + " , ".join(columns.get(i, [])))
    return db_id + " | " + " | ".join(parts)


def model_input(example):
    return example["question"] + " | " + schema_text(example["db_id"])


def model_target(example):
    # Spider's queries have stray spaces and line breaks
    return " ".join(example["query"].split())


print(model_input(example))
print(model_target(example))
lengths = [len(tokenizer(model_input(e)).input_ids) for e in train_set]
print("input tokens: median", int(np.median(lengths)), "longest", max(lengths))
print("training inputs longer than 512 tokens:", sum(n > 512 for n in lengths))
dev_lengths = [len(tokenizer(model_input(e)).input_ids) for e in dev_set]
print("development inputs longer than 512 tokens:", sum(n > 512 for n in dev_lengths))

# %% [markdown]
# ## 8. A T5 that writes SQL
#
# Before training anything, one problem. Ask the tokenizer to read a query with a comparison:

# %%
query = "SELECT count(*) FROM singer WHERE Age < 30"
ids = tokenizer(query).input_ids
print(tokenizer.convert_ids_to_tokens(ids))
print("back to text:", tokenizer.decode(ids, skip_special_tokens=True))
with_less = sum("<" in model_target(e) for e in train_set)
print(f"training queries with '<': {with_less} of {len(train_set):,}")

# %% [markdown]
# @@LESS_THAN_STORY@@

# %%
print("vocabulary rows in the model:", t5.embed.weight.shape[0])
print("tokens the tokenizer knows:  ", len(tokenizer))
sql_tokenizer = AutoTokenizer.from_pretrained(T5_SMALL)
sql_tokenizer.add_tokens(["<"])
print("id of the new token '<':     ", sql_tokenizer.convert_tokens_to_ids("<"))
ids = sql_tokenizer(query).input_ids
print(sql_tokenizer.convert_ids_to_tokens(ids))

# %% [markdown]
# @@NEW_TOKEN_COMMENT@@
#
# Now the training. We fine-tune t5-base, the same architecture as t5-small with 12 layers on each side and a width
# of 768, loaded into our own `T5` class; the transplant needs no change. Each step takes a random batch of
# questions, runs the encoder on the inputs and the decoder on the queries shifted right by the start token, and
# minimises the cross-entropy of every query token, ignoring padding, exactly as in lesson 4.

# %%
del base_reference, bart  # the reference models have done their job
if device == "cuda":
    torch.cuda.empty_cache()
model, model_config = load_t5(T5_BASE)
model = model.to(device)
print(f"t5-base: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} million parameters")


def make_batch(examples):
    inputs = sql_tokenizer([model_input(e) for e in examples], max_length=512,
                           truncation=True, padding=True, return_tensors="pt")
    targets = sql_tokenizer([model_target(e) for e in examples], max_length=256,
                            truncation=True, padding=True, return_tensors="pt").input_ids
    start = torch.zeros(len(examples), 1, dtype=torch.long)  # the start token
    decoder_input = torch.cat([start, targets[:, :-1]], dim=1)
    labels = targets.masked_fill(targets == sql_tokenizer.pad_token_id, -100)
    return inputs.input_ids, inputs.attention_mask, decoder_input, labels


def loss_on(model, batch):
    ids, mask, decoder_input, labels = [t.to(device) for t in batch]
    logits = model.decode(decoder_input, model.encode(ids, mask), mask)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)


held_out = random.Random(SEED).sample(dev_set, 256 if not SMOKE else 16)
with torch.no_grad():
    model.eval()
    before = loss_on(model, make_batch(held_out[:32])).item()
print(f"loss on development queries before training: {before:.2f}")

# %% [markdown]
# @@TRAINING_SETUP@@

# %%
TRAIN_MINUTES = 0.5 if SMOKE else 20
BATCH = 16
LR = 3e-4
WARMUP = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)


TOKEN_BUDGET = 4096  # input tokens per batch: 16 questions of 256 tokens, or 8 of 512
input_tokens = {}
for e, n in zip(train_set, lengths):
    input_tokens[e["question"] + e["query"]] = min(n, 512)


def batches_by_length(examples, size, rng):
    shuffled = examples[:]
    rng.shuffle(shuffled)
    batches = []
    for i in range(0, len(shuffled), size * 50):
        pool = sorted(shuffled[i:i + size * 50],
                      key=lambda e: input_tokens[e["question"] + e["query"]])
        batch = []
        for e in pool:
            longest = input_tokens[e["question"] + e["query"]]  # the pool is sorted
            if len(batch) == size or (len(batch) + 1) * longest > TOKEN_BUDGET:
                batches.append(batch)
                batch = []
            batch.append(e)
        batches.append(batch)
    rng.shuffle(batches)  # similar lengths within a batch, random order of batches
    return batches


def dev_loss(model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(held_out), 32):
            chunk = held_out[i:i + 32]
            total += loss_on(model, make_batch(chunk)).item() * len(chunk)
    model.train()
    return total / len(held_out)


rng = random.Random(SEED)
history = []
model.train()
started, step = time.time(), 0
while time.time() - started < TRAIN_MINUTES * 60:
    for examples in batches_by_length(train_set, BATCH, rng):
        if time.time() - started >= TRAIN_MINUTES * 60:
            break
        for group in optimizer.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        loss = loss_on(model, make_batch(examples))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 200 == 0:
            history.append({"step": step, "minutes": (time.time() - started) / 60,
                            "train loss": loss.item(), "dev loss": dev_loss(model)})
        step += 1
history.append({"step": step, "minutes": (time.time() - started) / 60,
                "train loss": loss.item(), "dev loss": dev_loss(model)})
model.eval()
history = pd.DataFrame(history)
passes = step * BATCH / len(train_set)
print(f"{step:,} steps, {passes:.1f} passes over the training questions")

# %%
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(history["step"], history["train loss"], color=COLORS["enc"], label="training batch")
ax.plot(history["step"], history["dev loss"], color=COLORS["dec"],
        label="development questions")
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy per token")
ax.set_ylim(0, None)
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# @@LOSS_COMMENT@@
#
# ## 9. Checking by running
#
# A loss says how probable the model finds the correct query. It doesn't say whether the query the model writes is
# right, and for SQL there is a better judge than any metric: run it. Spider's databases are real SQLite files, so we
# run the model's query and the correct one on the same database and compare what comes back. This is *execution
# accuracy*. @@EXEC_SOURCE@@
#
# Two results are the same if they hold the same rows. Order matters only if the correct query asks for one with
# `ORDER BY`; otherwise we compare the rows as sorted lists. A query that runs longer than five seconds is stopped
# and counts as wrong, and so does one that fails.

# %%
def run_query(db_id, query, seconds=5.0):
    path = SPIDER_DIR / "database" / db_id / (db_id + ".sqlite")
    connection = sqlite3.connect(path)
    # a few databases hold text that isn't valid UTF-8
    connection.text_factory = lambda raw: raw.decode(errors="replace")
    deadline = time.time() + seconds
    # SQLite calls the handler every 10,000 steps; returning True stops the query
    connection.set_progress_handler(lambda: time.time() > deadline, 10_000)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def same_result(db_id, predicted, gold):
    expected = run_query(db_id, gold)
    try:
        got = run_query(db_id, predicted)
    except Exception as error:
        return False, type(error).__name__
    if "order by" in gold.lower():
        return got == expected, "ran"
    return sorted(map(repr, got)) == sorted(map(repr, expected)), "ran"


print(same_result("concert_singer", "SELECT count(*) FROM singer WHERE Age < 30",
                  "SELECT count(*) FROM singer WHERE age < 30"))
print(same_result("concert_singer", "SELECT count(*) FROM singer WHERE Age <= 30",
                  "SELECT count(*) FROM singer WHERE age < 30"))
print(same_result("concert_singer", "SELECT count(*) FROM singers",
                  "SELECT count(*) FROM singer WHERE age < 30"))

# %% [markdown]
# Now every development question, 1,034 of them about 20 databases the model has never seen. Generation uses our
# own `generate` with the KV cache, in batches.

# %%
evaluated = dev_set if not SMOKE else dev_set[:24]
predictions = []
started = time.time()
for i in range(0, len(evaluated), 64):
    chunk = evaluated[i:i + 64]
    batch = sql_tokenizer([model_input(e) for e in chunk], max_length=512, truncation=True,
                          padding=True, return_tensors="pt").to(device)
    written = generate(model, batch.input_ids, batch.attention_mask, max_new_tokens=192)
    predictions += sql_tokenizer.batch_decode(written, skip_special_tokens=True)
print(f"{len(predictions):,} queries written in {time.time() - started:.0f} s")

results = []
for e, predicted in zip(evaluated, predictions):
    correct, status = same_result(e["db_id"], predicted, model_target(e))
    exact = " ".join(predicted.lower().split()) == model_target(e).lower()
    results.append({"db_id": e["db_id"], "question": e["question"], "gold": model_target(e),
                    "predicted": predicted, "correct": correct, "status": status,
                    "exact": exact})
results = pd.DataFrame(results)
print(f"execution accuracy: {results['correct'].mean():.1%}")
print(f"exact string match: {results['exact'].mean():.1%}")
print("queries that failed to run:", (results["status"] != "ran").sum())

# %% [markdown]
# @@ACCURACY_COMMENT@@
#
# Where does it go wrong? Spider's own scripts sort queries into four levels of hardness by counting their parts; a
# rougher but transparent way is to look at what a correct query contains: a join, a nested `SELECT`, a `GROUP BY`,
# an `ORDER BY`, or none of these.

# %%
def query_parts(query):
    q = " " + query.lower() + " "
    return {"join": " join " in q, "nested select": q.count("select ") > 1,
            "group by": " group by " in q, "order by": " order by " in q}


rows = []
for name in ["join", "nested select", "group by", "order by", "none of these"]:
    if name == "none of these":
        chosen = [not any(query_parts(g).values()) for g in results["gold"]]
    else:
        chosen = [query_parts(g)[name] for g in results["gold"]]
    part = results[chosen]
    rows.append({"correct query has": name, "questions": len(part),
                 "execution accuracy": part["correct"].mean()})
parts = pd.DataFrame(rows)
table(parts)

# %%
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh(parts["correct query has"], parts["execution accuracy"], color=COLORS["enc"])
for k in range(len(parts)):
    accuracy, count = parts["execution accuracy"][k], parts["questions"][k]
    ax.text(accuracy + 0.01, k, f"{accuracy:.0%} of {count}", va="center",
            color=COLORS["muted"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("execution accuracy")
plt.show()

# %% [markdown]
# @@PARTS_COMMENT@@
#
# Three kinds of answers are worth looking at one by one: queries that are right although they differ from the
# reference, queries that don't run, and queries that run and return the wrong thing.

# %%
right_but_different = results[results["correct"] & ~results["exact"]]
failed = results[results["status"] != "ran"]
wrong = results[(results["status"] == "ran") & ~results["correct"]]
for title, frame in [("right, but not the reference string", right_but_different),
                     ("does not run", failed), ("runs, wrong result", wrong)]:
    print(f"== {title}: {len(frame)}")
    for row in frame.head(3).itertuples():
        print("  question: ", row.question)
        print("  reference:", row.gold)
        note = "" if row.status == "ran" else f"({row.status})"
        print("  model:    ", row.predicted, note)
        print()

# %% [markdown]
# @@EXAMPLES_COMMENT@@
#
# ## 10. Why decoder-only models won anyway
#
# @@WHY_DECODERS@@
#
# ## 11. Where the encoder–decoder is the right tool today
#
# @@WHERE_TODAY@@
#
# T5's passport, and the family tree with this lesson's branch:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a05/passport-light.png" width="760" alt="T5's passport. Published October 2019 by Google. Type: encoder–decoder. Pre-training: span corruption on C4, and in version 1.0 a mixture of supervised tasks. Attention: bidirectional in the encoder, causal self-attention and cross-attention in the decoder. Positions: no position embeddings, a learned bias per head over 32 log-spaced buckets of distance. Normalisation: pre-norm with scale only, no mean and no bias. Activation: ReLU in version 1.0, gated GELU from version 1.1. Vocabulary: SentencePiece, 32,128 rows, without the characters less-than and curly braces. Sizes: 60 million, 220 million, 770 million, 3 and 11 billion parameters. Good at: input-to-output tasks such as translation, extraction, SQL, and as a text encoder. Not for: open-ended chat, long free generation, in-context learning">
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a05/family-light.png" width="760" alt="The family of the Transformer as this course covers it, with lesson 5, T5 and BART, highlighted and lessons 1 to 4 done">

# %% [markdown]
# ## References
#
# @@REFERENCES@@
