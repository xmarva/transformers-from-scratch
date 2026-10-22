# %% [markdown]
# # Lesson 2. The Transformer, built by hand
#
# GPT-4o, Llama 3 and DeepSeek-V3 come from different companies, learned from different data and differ in size by
# orders of magnitude. Open any of them and you find the same thing: one block, stacked. Here are four real models
# drawn to scale, next to the one you will build in this lesson:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/hero-light.png" width="760" alt="Five Transformers drawn to scale, width proportional to d_model and height to the number of layers: ours is tiny next to opus-mt-en-ru, GPT-2, Llama 3.1 8B and DeepSeek-V3">
#
# The block comes from a 2017 paper with a famous title, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
# (Vaswani et al.). It wasn't about chatbots. It was about translating English into German and French, and its
# model, the **Transformer**, beat every translation system published before it. On English–German even the smaller
# of its two versions did, after twelve hours of training on eight GPUs.
#
# Translation also shows best why the block is built the way it is. Take the course's sentence:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/agreement-light.png" width="540" alt="To translate 'The researcher said she was exhausted.' into Russian, three words take feminine forms because of 'she': Исследовательница, сказала, вымоталась">
#
# English doesn't mark gender on *researcher*, *said* or *exhausted*; Russian marks it on all three. To write the
# very first word, *Исследовательница* (a woman researcher), a translator needs *she*, which comes later in the
# English sentence, and the same *she* decides the endings of *сказала* and *вымоталась*. The mechanism that lets
# every word look directly at any other word, however far away, is **attention**. In this lesson you build it, and
# the Transformer around it, line by line.
#
# **What you'll do**
#
# 1. Read the blueprints of real models: GPT-2, Llama 3, DeepSeek-V3, and opus-mt-en-ru, the translator we will
#    compete with in lesson 4.
# 2. Build attention from its formula and check it against PyTorch's own.
# 3. Add what turns attention into a Transformer: masks, several heads, word order, feed-forward layers, residual
#    connections and normalization.
# 4. Assemble the encoder–decoder model of the 2017 paper and test it: its loss before training matches a formula
#    (once you know what it is doing), and it can learn a batch of sentences by heart.
#
# **What you need**: the tokenizer and the data of lesson 1, which the notebook downloads. Some PyTorch helps
# (tensors, `nn.Module`), but every step is explained. A CPU is enough; the notebook runs in a few minutes.

# %% tags=["include:setup"]

# %%
import collections
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from matplotlib.colors import ListedColormap
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% [markdown]
# Lesson 1 published its tokenizer and its splits of the data on the Hugging Face Hub. We need the tokenizer and,
# for a few experiments, the validation sentences:

# %%
tokenizer_path = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

valid_path = hf_hub_download(
    "xmarva/tfs-tatoeba-en-ru", "data/validation.parquet", repo_type="dataset"
)
valid = pd.read_parquet(valid_path)

VOCAB_SIZE = tokenizer.get_vocab_size()
PAD = tokenizer.token_to_id("<pad>")
BOS = tokenizer.token_to_id("<bos>")
EOS = tokenizer.token_to_id("<eos>")
print("vocabulary:", VOCAB_SIZE, "validation pairs:", len(valid))

# %% [markdown]
# ## 1. The blueprints of real models
#
# Every model on the Hub comes with a `config.json`: a few hundred bytes that describe its shape, so that a library
# can build the right network before loading the weights into it. Download four of them:

# %%
BLUEPRINTS = {
    "GPT-2 (2019)": "openai-community/gpt2",
    "opus-mt-en-ru (2020)": "Helsinki-NLP/opus-mt-en-ru",
    "Llama 3.1 8B (2024)": "unsloth/Meta-Llama-3.1-8B",
    "DeepSeek-V3 (2024)": "deepseek-ai/DeepSeek-V3",
}

configs = {}
for name, repo in BLUEPRINTS.items():
    with open(hf_hub_download(repo, "config.json")) as f:
        configs[name] = json.load(f)

gpt2_config = configs["GPT-2 (2019)"]
for key in ["n_layer", "n_embd", "n_head", "vocab_size", "n_positions"]:
    print(key, "=", gpt2_config[key])

# %% [markdown]
# Every lab names things its own way: GPT-2's `n_embd` is Llama's `hidden_size` and Marian's `d_model`. (Marian is
# the translation toolkit behind opus-mt-en-ru.) A small function reads all of them into the course's notation:


# %%
def first_key(config, keys):
    for key in keys:
        if config.get(key) is not None:
            return config[key]
    return None


def blueprint(name, config):
    d_model = first_key(config, ["hidden_size", "d_model", "n_embd"])
    if "encoder_layers" in config:
        layers = f'{config["encoder_layers"]} + {config["decoder_layers"]}'
    else:
        layers = str(first_key(config, ["num_hidden_layers", "n_layer"]))

    d_ff = first_key(config, ["intermediate_size", "encoder_ffn_dim", "n_inner"])
    if d_ff is None:
        d_ff = 4 * d_model  # GPT-2 leaves it out; its default is four times d_model

    heads = first_key(config, ["num_attention_heads", "encoder_attention_heads", "n_head"])

    note = ""
    if "n_routed_experts" in config:
        n_experts = config["n_routed_experts"]
        note = f'{n_experts} experts, {config["num_experts_per_tok"]} used per token'

    return {
        "model": name,
        "layers": layers,
        "d_model": d_model,
        "heads": heads,
        "d_ff": d_ff,
        "vocabulary": config["vocab_size"],
        "note": note,
    }


# %% [markdown]
# Our model gets its own row. Its settings live here, next to the table that compares them; `D_MODEL` is the width
# that lesson 1 used to price the embedding table. The first row is the base model of the 2017 paper, copied from
# its Table 3.

# %%
D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 3  # in the encoder, and as many in the decoder
DROPOUT = 0.1

rows = []
rows.append(
    {
        "model": "Transformer, base (2017)",
        "layers": "6 + 6",
        "d_model": 512,
        "heads": 8,
        "d_ff": 2048,
        "vocabulary": 37000,
        "note": "from the paper",
    }
)
for name, config in configs.items():
    rows.append(blueprint(name, config))
rows.append(
    {
        "model": "Ours (this lesson)",
        "layers": f"{N_LAYERS} + {N_LAYERS}",
        "d_model": D_MODEL,
        "heads": N_HEADS,
        "d_ff": D_FF,
        "vocabulary": VOCAB_SIZE,
        "note": "",
    }
)
table(pd.DataFrame(rows))

# %% [markdown]
# Read the table column by column.
#
# - **layers**: how many blocks are stacked. The translators (the 2017 model, opus-mt-en-ru and ours) have two
#   stacks: an **encoder** that reads the source sentence and a **decoder** that writes the translation. GPT-2,
#   Llama and DeepSeek keep only the decoder: they continue a text rather than translate one.
# - **d_model**: the length of the vector that represents every token on its way through the model.
# - **heads** and **d_ff**: the number of attention heads and the width of the feed-forward layer, both explained
#   below.
# - **note**: DeepSeek-V3 replaces most feed-forward layers with a *mixture of experts*, many feed-forward layers
#   of which every token uses only a few. Changes of this kind are the subject of lesson 5; the skeleton stays the
#   same.
#
# Compare two rows. opus-mt-en-ru, the model we will try to beat, has exactly the shape of the 2017 base model:
# three years of progress in translation changed the data and the training, not the blueprint. Ours is that
# blueprint at half the size in every dimension, small enough to train on a free GPU in lesson 3.
#
# ## 2. The plan
#
# Here is the whole model. Don't try to read every box yet; the rest of the lesson builds them one at a time, from
# the bottom up.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/architecture-light.png" width="760" alt="The encoder–decoder Transformer at our size: an encoder of self-attention and feed-forward layers reads the English tokens, a decoder of masked self-attention, cross-attention and feed-forward layers writes the Russian tokens, and the shared embedding matrix turns the decoder's output into scores for every token">
#
# The **encoder** (teal) reads the English sentence. Every token becomes a vector, receives a signal about its
# position, and goes through $N$ layers of two parts: *self-attention*, where the tokens look at each other, and a
# *feed-forward* network, which processes every token on its own. The **decoder** (rose) writes the Russian
# sentence one token at a time. Its layers have one more part, *cross-attention*, through which it reads the
# encoder's output, called the **memory**. At the top, the same embedding matrix that turned tokens into vectors
# turns vectors back into a score for every token of the vocabulary.

# %% [markdown]
# ## 3. From tokens to vectors
#
# A model can't do arithmetic with token ids: id 5 000 isn't "bigger" than id 50 in any meaningful way. So every
# token gets a learned vector of `D_MODEL` numbers, a row of the **embedding matrix**
# $E \in \mathbb{R}^{\lvert\mathcal{V}\rvert \times d_{\text{model}}}$. Here is our sentence as ids:

# %%
EN_EXAMPLE = "The researcher said she was exhausted."
RU_EXAMPLE = "Исследовательница сказала, что вымоталась."

encoding = tokenizer.encode(EN_EXAMPLE)
src_ids = torch.tensor([encoding.ids])  # a batch of one sentence: shape (1, T)
token_chips([encoding.tokens], ids=[encoding.ids])
plt.show()

# %% [markdown]
# Looking up a row is the same as multiplying a one-hot vector by $E$, which is why an embedding layer can be
# trained like any other layer. Rather than believe it, check it:

# %%
embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
nn.init.normal_(embedding.weight, mean=0.0, std=D_MODEL**-0.5)

vectors = embedding(src_ids)
print("vectors:", tuple(vectors.shape))  # (B, T, d_model)

one_hot = F.one_hot(src_ids, VOCAB_SIZE).float()  # (1, T, |V|)
assert torch.allclose(one_hot @ embedding.weight, vectors)

# %% [markdown]
# The initialization isn't PyTorch's default, and the reason is at the top of the architecture. The paper uses the
# same matrix $E$ twice: as the embedding at the bottom and as the output layer at the top, where the score of
# token $j$ is the dot product of the decoder's final vector $\mathbf{h}$ with row $\mathbf{e}_j$. That vector has
# $d_{\text{model}}$ entries of size about 1, so with entries of $E$ drawn with standard deviation
# $d_{\text{model}}^{-1/2}$ the scores start out of size about 1: a model that knows nothing should be unsure of
# everything.
#
# On the input side the same rows are small, while the position signals added to them in section 8 have entries of
# size 1. So the paper multiplies the embeddings by $\sqrt{d_{\text{model}}}$:

# %%
print("embedding entries, std:", round(vectors.std().item(), 4))
print("times sqrt(d_model), std:", round((vectors * math.sqrt(D_MODEL)).std().item(), 4))

# %% [markdown]
# ## 4. Attention
#
# Attention is a soft lookup. Every position has three vectors: a **query**, what it is looking for; a **key**,
# what it offers to other positions' queries; and a **value**, what it hands over to those who find it. A position
# compares its query with every key, and the better the match, the more of that key's value it takes.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/attention-light.png" width="760" alt="Scaled dot-product attention: Q times transposed K gives the scores, which are divided by the square root of d_k, masked and turned into weights by a softmax over each row; the weights times V give the output">
#
# In the notation of the course, with every query, key and value as a row of a matrix:
#
# $$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
#
# - $Q$, shape $(T_q, d_k)$: one query per position that asks.
# - $K$, shape $(T_k, d_k)$, and $V$, shape $(T_k, d_v)$: one key and one value per position that can be found.
# - $Q K^\top$, shape $(T_q, T_k)$: the dot product of every query with every key. A large dot product means the
#   two vectors point the same way: a good match.
# - $\sqrt{d_k}$ keeps the scores in a range where softmax still works well; section 5 shows why.
# - $M$, the **mask**: $0$ where looking is allowed and $-\infty$ where it isn't (section 6).
# - $\operatorname{softmax}$, applied to every row, turns scores into positive weights that sum to 1.
# - The result, shape $(T_q, d_v)$: for every query, a weighted average of the values.
#
# Start with numbers you can check by hand: one query, three keys, and values that are one-hot, so the output shows
# the weights themselves.

# %%
q = torch.tensor([[1.0, 2.0]])  # one query, d_k = 2
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # three keys
v = torch.eye(3)  # three one-hot values

scores = q @ k.T / math.sqrt(2)
weights = scores.softmax(dim=-1)
print("scores: ", scores)
print("weights:", weights)
print("output: ", weights @ v)

# %% [markdown]
# The dot products are $1$, $2$ and $3$; divided by $\sqrt 2$ they become the scores, and softmax turns those into
# weights that sum to 1. The third key matches the query best and gets the largest share, but not all of it:
# attention mixes, it doesn't pick. That softness is what makes it trainable, because every weight changes a little
# when a score changes a little.
#
# The general function works on whole batches at once. Its inputs can have any number of leading dimensions (batch,
# heads); the last two are positions and vector entries.


# %%
def attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T_q, T_k)
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


# %% [markdown]
# PyTorch has its own implementation, `F.scaled_dot_product_attention`, which is faster and saves memory (lesson 5
# switches to it). It computes the same thing. Compare the two on random tensors of a realistic shape, with a mask
# that hides some positions:

# %%
q = torch.randn(2, 4, 7, 64)  # (batch, heads, positions, d_k)
k = torch.randn(2, 4, 7, 64)
v = torch.randn(2, 4, 7, 64)
mask = torch.zeros(7, 7)
mask[:, 5:] = float("-inf")  # nobody may look at the last two positions

ours, _ = attention(q, k, v, mask)
theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
assert torch.allclose(ours, theirs, atol=1e-5)
print("same output as torch's scaled_dot_product_attention")

# %% [markdown]
# ## 5. Why divide by $\sqrt{d_k}$
#
# Suppose the entries of a query $\mathbf{q}$ and a key $\mathbf{k}$ are independent, with mean 0 and variance 1.
# (A real model's are only roughly like that, but the argument is about how things grow with $d_k$.) Their dot
# product is a sum of $d_k$ independent terms, and the variances of independent terms add up:
#
# $$\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i)
# = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_k.$$
#
# So the scores grow like $\sqrt{d_k}$. Large scores make softmax extreme: it hands almost all the weight to one
# key. That stalls learning, because the derivative of the softmax output $p_i$ with respect to a score $s_j$ is
#
# $$\frac{\partial p_i}{\partial s_j} = p_i\,(\delta_{ij} - p_j),$$
#
# and when one $p$ is close to 1 and the rest close to 0, every one of these terms is close to 0: the scores stop
# receiving gradient. Dividing by $\sqrt{d_k}$ brings the variance back to 1, whatever the width of a head.
#
# Measure both claims: the variance of the dot product, and the average weight of the top key among 16, with and
# without the division.

# %%
rows = []
for d_k in [4, 16, 64, 256, 1024]:
    q = torch.randn(2_000, 1, d_k)  # 2 000 random queries
    k = torch.randn(2_000, 16, d_k)  # 16 random keys for each
    scores = (q @ k.transpose(-2, -1)).squeeze(1)  # (2 000, 16)

    top_unscaled = scores.softmax(dim=-1).max(dim=-1).values.mean()
    top_scaled = (scores / math.sqrt(d_k)).softmax(dim=-1).max(dim=-1).values.mean()
    rows.append(
        {
            "d_k": d_k,
            "variance of q·k": scores.var().item(),
            "top weight, raw": top_unscaled.item(),
            "top weight, ÷ √d_k": top_scaled.item(),
        }
    )
table(pd.DataFrame(rows))

# %% [markdown]
# The variance follows $d_k$, as derived. Without the division the top key takes more and more of the weight as
# heads get wider, until attention becomes a hard choice; with it, the picture is the same at every width. Our
# heads have $d_k = 64$, the same as in the 2017 paper.

# %% [markdown]
# ## 6. Masks
#
# Attention lets every position look at every other. Two kinds of looking must be forbidden.
#
# **Padding.** Sentences in a batch have different lengths, and a tensor must be rectangular, so shorter sentences
# are filled up with the `<pad>` token. Nobody should attend to padding; it isn't part of the text. Make a batch of
# three sentences of different lengths:


# %%
def pad_batch(sequences):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), PAD)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids)
    return batch


word_counts = valid.en.str.split().str.len()
short_sentence = valid.en[word_counts == 2].iloc[0]
long_sentence = valid.en[word_counts == 10].iloc[0]
sentences = [EN_EXAMPLE, short_sentence, long_sentence]

src = pad_batch([tokenizer.encode(sentence).ids for sentence in sentences])
for row in src.tolist():
    print(row)  # the ids the model receives; 0 is <pad>

# %% [markdown]
# **The future.** The decoder is trained on whole Russian sentences at once (section 10 shows how), but when it
# translates, it writes one token at a time and can't see the tokens it hasn't written yet. So during training,
# position $t$ may look only at positions up to $t$; otherwise the model would learn to copy the next word instead
# of predicting it.
#
# Both rules become additive masks of $0$ and $-\infty$. After softmax, $e^{-\infty} = 0$: a masked position gets
# exactly zero weight. The shapes are chosen so that one mask applies to every head and every query.


# %%
def padding_mask(ids, pad_id):
    mask = torch.zeros(ids.shape, device=ids.device)
    mask[ids == pad_id] = float("-inf")
    return mask[:, None, None, :]  # (B, 1, 1, T): the same for every head and every query


def causal_mask(length, device=None):
    mask = torch.full((length, length), float("-inf"), device=device).triu(diagonal=1)
    return mask[None, None, :, :]  # (1, 1, T, T): row t may look at columns 0..t


# %% [markdown]
# The padding mask of our batch: every query may look at the violet keys, and none at the pale `<pad>` ones.

# %%
batch_tokens = []
for ids in src.tolist():
    batch_tokens.append([tokenizer.id_to_token(token_id) for token_id in ids])

allowed = padding_mask(src, PAD)[:, 0, 0, :] == 0  # (3, T): True where looking is allowed
token_chips(
    batch_tokens,
    labels=["sentence 1", "sentence 2", "sentence 3"],
    strong=lambda row, col, token: bool(allowed[row, col]),
    role="att",
    size=10,
)
plt.show()

# %% [markdown]
# The causal mask of the Russian sentence, one row per query: row $t$ may look at the violet columns, the
# positions up to and including $t$.

# %%
ru_tokens = tokenizer.encode(RU_EXAMPLE).tokens
future_mask = causal_mask(len(ru_tokens))[0, 0]  # (T, T)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
mask_colors = ListedColormap([COLORS["grid"], COLORS["att"]])  # masked, allowed
ax.imshow(future_mask == 0, cmap=mask_colors, vmin=0, vmax=1)
ax.set_xticks(range(len(ru_tokens)), ru_tokens, rotation=90, fontsize=8)
ax.set_yticks(range(len(ru_tokens)), ru_tokens, fontsize=8)
ax.set_xlabel("keys")
ax.set_ylabel("queries")
ax.set_title("Causal mask: no looking ahead")
ax.grid(False)
plt.show()

# %% [markdown]
# Two checks that the masks do their job, on attention without heads yet: queries, keys and values are all the
# scaled embeddings themselves, and the masks lose their heads dimension. With the padding mask, the short sentence
# padded inside the batch gets exactly the output it gets alone, without any padding:

# %%
x = embedding(src) * math.sqrt(D_MODEL)  # (3, T, d_model)
no_heads_mask = padding_mask(src, PAD)[:, 0]  # (3, 1, T)
out_batch, _ = attention(x, x, x, no_heads_mask)

alone_ids = torch.tensor([tokenizer.encode(short_sentence).ids])
x_alone = embedding(alone_ids) * math.sqrt(D_MODEL)
out_alone, _ = attention(x_alone, x_alone, x_alone)

n = alone_ids.shape[1]
assert torch.allclose(out_batch[1, :n], out_alone[0], atol=1e-5)
print("padding changes nothing for the real tokens")

# %% [markdown]
# With the causal mask, changing the last token of the Russian sentence changes nothing before it:

# %%
ru_ids = torch.tensor([tokenizer.encode(RU_EXAMPLE).ids])
changed_ids = ru_ids.clone()
changed_ids[0, -1] = tokenizer.token_to_id("!")  # replace the final "." with "!"

outputs = []
for ids in [ru_ids, changed_ids]:
    x = embedding(ids) * math.sqrt(D_MODEL)
    out, _ = attention(x, x, x, causal_mask(ids.shape[1])[0])  # (1, T, T) without the heads
    outputs.append(out)

assert torch.allclose(outputs[0][0, :-1], outputs[1][0, :-1])
assert not torch.allclose(outputs[0][0, -1], outputs[1][0, -1])
print("earlier positions can't see the change; the last one can")

# %% [markdown]
# ## 7. Several heads
#
# One attention gives every position one weighted average. A word often needs several kinds of information at
# once: *сказала* needs the subject's gender, the tense of *said*, and the word it translates. The paper's answer is
# **multi-head attention**: $h$ attentions run side by side, each in its own smaller space of
# $d_k = d_{\text{model}} / h$ dimensions, with its own learned projections:
#
# $$\operatorname{head}_i = \operatorname{Attention}(X W_i^Q,\ X W_i^K,\ X W_i^V), \qquad
# \operatorname{MultiHead}(X) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h)\, W^O.$$
#
# - $X$, shape $(T, d_{\text{model}})$: the vectors of the positions.
# - $W_i^Q, W_i^K, W_i^V$, each $(d_{\text{model}}, d_k)$: what head $i$ asks for, offers and hands over.
# - $W^O$, $(d_{\text{model}}, d_{\text{model}})$: mixes the heads' results back into one vector.
#
# Because every head works on $d_k = d_{\text{model}} / h$ dimensions, all heads together cost about as much as one
# full-width attention. In code, the projections of all heads are one big matrix, and splitting into heads is just
# a reshape:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/heads-light.png" width="760" alt="Multi-head attention: the 256 numbers of every position are split into 4 heads of 64; each head runs attention on its own slice; the results are concatenated and mixed by W^O">


# %%
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
        x = x.view(batch, length, self.n_heads, self.d_k)  # (B, T, h, d_k)
        return x.transpose(1, 2)  # (B, h, T, d_k): heads become a batch dimension

    def forward(self, x_query, x_key_value, mask=None):
        q = self.split_heads(self.w_q(x_query))
        k = self.split_heads(self.w_k(x_key_value))
        v = self.split_heads(self.w_v(x_key_value))
        out, self.weights = attention(q, k, v, mask)  # (B, h, T, d_k)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, -1)  # concatenate the heads
        return self.w_o(out)


# %% [markdown]
# The queries come from `x_query` and the keys and values from `x_key_value`. In self-attention both are the same
# sequence; in the decoder's cross-attention the queries come from the Russian side and the keys and values from
# the encoder's memory.
#
# PyTorch has a ready-made `nn.MultiheadAttention`. Copy our weights into it and compare the outputs:

# %%
ours = MultiHeadAttention(D_MODEL, N_HEADS)
theirs = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)

with torch.no_grad():  # PyTorch keeps the three input projections in one matrix
    in_weights = [ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]
    in_biases = [ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]
    theirs.in_proj_weight.copy_(torch.cat(in_weights))
    theirs.in_proj_bias.copy_(torch.cat(in_biases))
    theirs.out_proj.weight.copy_(ours.w_o.weight)
    theirs.out_proj.bias.copy_(ours.w_o.bias)

x = torch.randn(2, 9, D_MODEL)
mask = causal_mask(9)
out_ours = ours(x, x, mask)
out_theirs, _ = theirs(x, x, x, attn_mask=mask[0, 0])
assert torch.allclose(out_ours, out_theirs, atol=1e-5)

n_params = sum(p.numel() for p in ours.parameters())
print("same output as nn.MultiheadAttention;", n_params, "parameters = 4 × (d² + d)")

# %% [markdown]
# ## 8. Word order
#
# Attention has a blind spot: it doesn't know where anything is. Each output is a weighted average over the other
# positions, and an average doesn't depend on the order of its terms. Shuffle the words of the sentence, and the
# outputs are shuffled the same way, with nothing else changed:

# %%
x = embedding(src_ids) * math.sqrt(D_MODEL)  # (1, T, d_model)
shuffle = torch.randperm(x.shape[1])

out = ours(x, x)
out_shuffled = ours(x[:, shuffle], x[:, shuffle])
assert torch.allclose(out[:, shuffle], out_shuffled, atol=1e-5)

shuffled_tokens = [encoding.tokens[i] for i in shuffle.tolist()]
token_chips([encoding.tokens, shuffled_tokens], labels=["sentence", "shuffled"])
plt.show()

# %% [markdown]
# To attention alone, these two rows are the same sentence. So the position has to be part of the vector. The paper
# adds a **positional encoding**: for position $pos$ and the pair of entries $2i$ and $2i+1$,
#
# $$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
# PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$
#
# - $pos$ is the position, $0, 1, 2, \dots$
# - Each pair of entries is a sine and a cosine of the same frequency. The frequencies fall geometrically from 1 to
#   $1/10000$: the first pairs change at every step, like the second hand of a clock, and the last ones barely move
#   over a whole sentence, like the hour hand.
# - The values lie between $-1$ and $1$, the same size as the scaled embeddings they are added to.


# %%
def positional_encoding(max_len, d_model):
    positions = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
    frequencies = 10_000 ** (-torch.arange(0, d_model, 2) / d_model)  # (d_model / 2,)
    angles = positions * frequencies  # (max_len, d_model / 2)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


pe = positional_encoding(256, D_MODEL)

# %% [markdown]
# Why sines and cosines? Because of the identity $\sin a \sin b + \cos a \cos b = \cos(a - b)$: the dot product of
# the encodings of positions $p$ and $p + k$ is $\sum_i \cos(k\,\omega_i)$, where $\omega_i$ are the frequencies.
# It depends only on the distance $k$, not on where in the sentence the pair is. Attention compares positions
# through dot products, so it can learn about relative distances that hold anywhere. Check it from three starting
# positions:

# %%
offsets = range(64)
curves = {}
for start in [0, 20, 100]:
    dots = [(pe[start] @ pe[start + offset]).item() for offset in offsets]
    curves[start] = dots

assert np.allclose(curves[0], curves[20], atol=1e-3)
assert np.allclose(curves[0], curves[100], atol=1e-3)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
axes[0].imshow(pe[:64].T, cmap="PuOr", aspect="auto", vmin=-1, vmax=1)
axes[0].set_xlabel("position")
axes[0].set_ylabel("entry of the vector")
axes[0].set_title("Positional encoding: fast hands at the top")
axes[0].grid(False)
styles = {0: "-", 20: "--", 100: ":"}  # the curves overlap; line styles tell them apart
for start, dots in curves.items():
    axes[1].plot(offsets, dots, linestyle=styles[start], label=f"from position {start}")
axes[1].set_xlabel("distance k")
axes[1].set_ylabel("PE(p) · PE(p + k)")
axes[1].set_title("Similarity depends only on distance")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The three curves lie on top of each other. Llama and DeepSeek use a descendant of this idea, rotary position
# embeddings (RoPE), which apply the rotation inside attention itself; lesson 5 builds them.

# %% [markdown]
# ## 9. The rest of a layer
#
# Three smaller pieces complete a layer.
#
# **Feed-forward network.** Attention moves information between positions; the feed-forward network then
# processes what each position has gathered, every position separately and with the same weights:
#
# $$\operatorname{FFN}(\mathbf{x}) = \max(0,\ \mathbf{x} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2,$$
#
# with $W_1$ of shape $(d_{\text{model}}, d_{\text{ff}})$ and $W_2$ of shape $(d_{\text{ff}}, d_{\text{model}})$:
# the vector is widened four times, passed through ReLU, $\max(0, \cdot)$, and narrowed back.


# %%
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


feed_forward = FeedForward(D_MODEL, D_FF)
print("feed-forward:", sum(p.numel() for p in feed_forward.parameters()), "parameters")
print("attention:   ", sum(p.numel() for p in ours.parameters()), "parameters")

# %% [markdown]
# Two thirds of an encoder layer's parameters sit in the feed-forward network, not in attention.
#
# **Residual connection.** Every sub-layer's output is *added* to its input,
# $\mathbf{x} + \operatorname{Sublayer}(\mathbf{x})$ ([He et al., 2016](https://arxiv.org/abs/1512.03385)). A
# sub-layer then only has to learn a correction to what is already there, and during training the gradient flows
# back through the "+" undiminished, however many layers are stacked. It also means the input embedding travels all
# the way to the top, which matters more than you'd think in section 11.
#
# **Layer normalization** ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) rescales every vector to mean 0 and
# variance 1, then lets the model choose its own scale and shift:
#
# $$\operatorname{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 +
# \epsilon}} + \boldsymbol{\beta}, \qquad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i -
# \mu)^2.$$
#
# $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are learned vectors of length $d_{\text{model}}$, starting at 1 and
# 0; $\epsilon$ prevents division by zero. Written out and compared with PyTorch's:


# %%
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(variance + eps) + beta


norm = nn.LayerNorm(D_MODEL)
x = torch.randn(2, 9, D_MODEL) * 3 + 1
assert torch.allclose(layer_norm(x, norm.weight, norm.bias), norm(x), atol=1e-5)
print("same as nn.LayerNorm; from here on we use PyTorch's")

# %% [markdown]
# The 2017 paper puts the normalization after the addition,
# $\operatorname{LayerNorm}(\mathbf{x} + \operatorname{Sublayer}(\mathbf{x}))$: the "add & norm" boxes of the
# diagram. Most models since put it before the sub-layer instead; lesson 5 measures the difference. The paper also
# applies **dropout** with probability $0.1$ to every sub-layer's output and to the embeddings: during training,
# random entries are set to zero, so the model can't rely on any single one.

# %% [markdown]
# ## 10. The whole model
#
# An encoder layer is self-attention and a feed-forward network, each wrapped in dropout, a residual connection
# and normalization:


# %%
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


# %% [markdown]
# A decoder layer has one more sub-layer between the two: cross-attention, whose queries come from the Russian side
# and whose keys and values come from the encoder's memory. Its self-attention gets the causal mask.


# %%
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


# %% [markdown]
# The decoder is trained with **teacher forcing**: it reads the correct Russian sentence, shifted one position to
# the right behind a `<bos>` token, and at every position predicts the next token. Thanks to the causal mask, one
# pass over the sentence trains all positions at once, and none of them can peek at its answer.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/02/shift-right-light.png" width="760" alt="Teacher forcing: the decoder reads <bos> followed by the Russian tokens and is trained to predict, at every position, the next token, ending with <eos>">
#
# The full model puts it together. One matrix $E$ serves three times: as the source embedding, as the target
# embedding (lesson 1 built one shared vocabulary for both languages) and as the output layer. Besides saving
# parameters, tying the output layer to the embedding tends to improve language models
# ([Press and Wolf, 2017](https://arxiv.org/abs/1608.05859)).


# %%
class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, pad_id, max_len=256
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        # the positional encoding is fixed, not trained: a buffer, not a parameter
        self.register_buffer("pe", positional_encoding(max_len, d_model))
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.d_model) + self.pe[: ids.shape[1]]
        return self.dropout(x)

    def encode(self, src):
        src_mask = padding_mask(src, self.pad_id)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x  # the memory: (B, T_src, d_model)

    def decode(self, tgt, memory, src):
        src_mask = padding_mask(src, self.pad_id)
        tgt_mask = padding_mask(tgt, self.pad_id) + causal_mask(tgt.shape[1], tgt.device)
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x @ self.embedding.weight.T  # scores for every token: (B, T_tgt, |V|)

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)


model = Transformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, DROPOUT, PAD).to(device)
print(sum(p.numel() for p in model.parameters()), "parameters")

# %% [markdown]
# Where do they live? Group the parameters by the part of the model they belong to:

# %%
parts = collections.Counter()
for name, parameter in model.named_parameters():
    if name.startswith("embedding"):
        part = "embedding E, shared"
    elif "norm" in name:
        part = "layer norms"
    else:
        stack = name.split(".")[0]  # "encoder" or "decoder"
        sub_layer = name.split(".")[2]  # self_attention, cross_attention or feed_forward
        part = stack + ": " + sub_layer.replace("_", "-")
    parts[part] += parameter.numel()

colors = []
for part in parts:
    if part.startswith("encoder"):
        colors.append(COLORS["enc"])
    elif part.startswith("decoder"):
        colors.append(COLORS["dec"])
    elif part.startswith("embedding"):
        colors.append(COLORS["data"])
    else:
        colors.append(COLORS["muted"])

fig, ax = plt.subplots(figsize=(6.4, 3.2))
bars = ax.barh(list(parts), list(parts.values()), color=colors)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9, color=COLORS["muted"])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, max(parts.values()) * 1.3)
ax.set_xticks([])  # the bars carry their own numbers
ax.set_title("Where the parameters live")
plt.show()

# %% [markdown]
# The shared embedding is the largest single part, which is why lesson 1 weighed every extra token against its
# row. Every count follows from the shapes, so the total is a formula, with $d = d_{\text{model}}$:
#
# $$\underbrace{\lvert\mathcal{V}\rvert\, d}_{\text{embedding}}
# + N\big(\underbrace{4(d^2 + d)}_{\text{attention}} + \underbrace{2 d d_{\text{ff}} + d_{\text{ff}} + d}_{\text{
# feed-forward}} + \underbrace{2 \cdot 2d}_{\text{norms}}\big)
# + N\big(2 \cdot 4(d^2 + d) + 2 d d_{\text{ff}} + d_{\text{ff}} + d + 3 \cdot 2d\big):$$
#
# the embedding, $N$ encoder layers, and $N$ decoder layers with their two attentions and three norms.


# %%
def count_parameters(vocab_size, d, d_ff, n_layers):
    attention_params = 4 * (d * d + d)
    feed_forward_params = 2 * d * d_ff + d_ff + d
    encoder_layer = attention_params + feed_forward_params + 2 * 2 * d
    decoder_layer = 2 * attention_params + feed_forward_params + 3 * 2 * d
    return vocab_size * d + n_layers * encoder_layer + n_layers * decoder_layer


by_formula = count_parameters(VOCAB_SIZE, D_MODEL, D_FF, N_LAYERS)
assert by_formula == sum(p.numel() for p in model.parameters())
print("the formula matches the model:", by_formula)

# %% [markdown]
# ## 11. A first run, and a parrot
#
# Make a batch of real sentence pairs and run the model. The source is the English ids; the decoder reads `<bos>`
# and the Russian ids, and the target is the Russian ids followed by `<eos>`.


# %%
def make_batch(pairs):
    src = pad_batch([tokenizer.encode(en).ids for en, _ in pairs])
    tgt_in = pad_batch([[BOS] + tokenizer.encode(ru).ids for _, ru in pairs])
    tgt_out = pad_batch([tokenizer.encode(ru).ids + [EOS] for _, ru in pairs])
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


pairs = list(zip(valid.en[:256], valid.ru[:256]))
src, tgt_in, tgt_out = make_batch(pairs)

model.eval()  # no dropout while we measure
with torch.no_grad():
    logits = model(src, tgt_in)
print("source:", tuple(src.shape))
print("decoder input:", tuple(tgt_in.shape))
print("scores:", tuple(logits.shape))

# %% [markdown]
# The training objective is the course's loss, the average cross-entropy of the correct next token:
#
# $$\mathcal{L}(\theta) = -\frac{1}{m} \sum_{t=1}^{m} \log p_\theta(y_t \mid y_{<t}, \mathbf{x}).$$
#
# Before training, we can predict its value. The scores $z_j$ of an untrained model are roughly independent, with
# mean 0 and some variance $\sigma^2$ (section 3 set it up to be about 1). The loss at one position is
# $\log \sum_j e^{z_j} - z_{\text{target}}$. The target's score is 0 on average, and a sum over sixteen thousand
# tokens is close to its expected value, $\lvert\mathcal{V}\rvert\,\mathbb{E}[e^{z}] = \lvert\mathcal{V}\rvert\,
# e^{\sigma^2/2}$ (the mean of $e^z$ for a normal $z$). So
#
# $$\mathcal{L}_{\text{init}} \approx \log\lvert\mathcal{V}\rvert + \frac{\sigma^2}{2}.$$
#
# The $\log\lvert\mathcal{V}\rvert$ part is the loss of a model that gives every token the same probability.
# Measure it:

# %%
real = tgt_out != PAD  # positions that hold real target tokens
loss = F.cross_entropy(logits[real], tgt_out[real]).item()
sigma2 = logits[real].var().item()

print("measured loss:     ", round(loss, 3))
print("log|V| + sigma²/2: ", round(math.log(VOCAB_SIZE) + sigma2 / 2, 3))

# %% [markdown]
# More than a whole unit too high. A gap like this usually means a bug, so look at what the model predicts: its top
# token at every position, under the token it has just read.

# %%
predicted = logits.argmax(dim=-1)
same_as_input = (predicted[real] == tgt_in[real]).float().mean().item()
print(f"top prediction = the token the decoder just read: {same_as_input:.1%} of positions")

n = int(real[0].sum())  # the first sentence of the batch
read = [tokenizer.id_to_token(i) for i in tgt_in[0, :n].tolist()]
guessed = [tokenizer.id_to_token(i) for i in predicted[0, :n].tolist()]
token_chips([read, guessed], labels=["reads", "predicts"])
plt.show()

# %% [markdown]
# The untrained model is a parrot: at every position it predicts the token it has just read. That's not a bug, and
# the explanation joins two sections of this lesson. The residual connections carry the input embedding
# $\sqrt{d_{\text{model}}}\,\mathbf{e}_t$ up through every layer, so the final vector still points partly along
# $\mathbf{e}_t$. And the output layer is the same matrix $E$, so the score of token $t$ is the dot product of that
# vector with $\mathbf{e}_t$ itself. Compare that one score with all the others:

# %%
own_scores = logits.gather(-1, tgt_in.unsqueeze(-1)).squeeze(-1)
print("score of the token just read, on average:", round(own_scores[real].mean().item(), 2))
print("spread of all scores, std:", round(logits[real].std().item(), 2))

# %% [markdown]
# One score in sixteen thousand, far above the spread of all the others, is exactly what pushes
# $\log\sum_j e^{z_j}$ up. Take it out at every position, as if that token weren't in the vocabulary, and check
# the two parts of the loss separately: the log of the sum, which the formula predicts as
# $\log(\lvert\mathcal{V}\rvert - 1) + \sigma^2/2$ now, and the score of the right token, which it assumes to be 0
# on average.

# %%
without_own = logits.clone()
# drop the score of the token just read
without_own.scatter_(-1, tgt_in.unsqueeze(-1), float("-inf"))
# and skip the rare positions where the answer is that same token
keep = real & (tgt_in != tgt_out)

others = without_own[keep]  # (positions, |V|)
log_sum = torch.logsumexp(others, dim=-1).mean().item()
right_score = others.gather(-1, tgt_out[keep].unsqueeze(-1)).mean().item()
sigma2_others = others[torch.isfinite(others)].var().item()
expected = math.log(VOCAB_SIZE - 1) + sigma2_others / 2

print("log of the sum, measured:   ", round(log_sum, 4))
print("log(|V| - 1) + sigma²/2:    ", round(expected, 4))
print("score of the right token:   ", round(right_score, 3))
assert abs(log_sum - expected) < 0.02

# %% [markdown]
# Now the formula holds to the second decimal. The score of the right token is near 0 but not exactly: a different
# random initialization moves it a little above or below, and the loss with it. The parrot is the big effect, and
# the first steps of training break the habit, because it is wrong almost everywhere: a translation rarely repeats
# a token twice in a row.
#
# ## 12. Can it learn?
#
# Before spending an hour of GPU time on training (lesson 3), make sure the model can learn at all. The cheapest
# test is to teach it a handful of sentence pairs by heart: a model that can't memorize sixteen pairs has a bug
# somewhere. [Karpathy (2019)](https://karpathy.github.io/2019/04/25/recipe/) calls this step "overfit one batch".
#
# One detail: lesson 1 kept groups of linked pairs together, so the validation set holds several translations of
# some English sentences. Nobody can memorize two answers to one question, so we take one pair per English
# sentence, plus our own sentence.

# %%
distinct = valid.drop_duplicates("en")
lesson_pairs = [(EN_EXAMPLE, RU_EXAMPLE)] + list(zip(distinct.en[:15], distinct.ru[:15]))
src, tgt_in, tgt_out = make_batch(lesson_pairs)

model.train()  # dropout on, as in real training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
start = time.perf_counter()
for _ in range(200):
    logits = model(src, tgt_in)
    flat_logits = logits.reshape(-1, VOCAB_SIZE)  # every position of every sentence
    loss = F.cross_entropy(flat_logits, tgt_out.reshape(-1), ignore_index=PAD)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

seconds = round(time.perf_counter() - start, 1)
print(len(losses), "steps in", seconds, "s")
print("loss:", round(losses[0], 2), "->", round(losses[-1], 4))

# %%
fig, ax = plt.subplots()
ax.plot(losses, color=COLORS["enc"])
ax.axhline(math.log(VOCAB_SIZE), color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)
ax.text(len(losses) - 1, math.log(VOCAB_SIZE) + 0.3, "log|V|", ha="right")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("Sixteen pairs, learned by heart")
plt.show()

# %% [markdown]
# To translate, the model writes one token at a time: encode the English sentence once, then repeatedly feed the
# decoder everything written so far and append its top prediction, until it predicts `<eos>`. This is **greedy
# decoding**; lesson 3 shows why it isn't the best way to decode, and does better.


# %%
def translate_greedy(model, sentence, max_tokens=60):
    model.eval()
    src = torch.tensor([tokenizer.encode(sentence).ids], device=device)
    written = [BOS]
    with torch.no_grad():
        memory = model.encode(src)
        for _ in range(max_tokens):
            tgt = torch.tensor([written], device=device)
            scores = model.decode(tgt, memory, src)[0, -1]  # scores for the next token
            next_id = scores.argmax().item()
            if next_id == EOS:
                break
            written.append(next_id)
    return tokenizer.decode(written[1:])


rows = []
for en, ru in lesson_pairs:
    output = translate_greedy(model, en)
    rows.append({"English": en, "model's Russian": output, "exact": output == ru})
memorized = pd.DataFrame(rows)
print(memorized.exact.sum(), "of", len(memorized), "translations reproduced exactly")
assert memorized.exact.all()
table(memorized.head(6))

# %% [markdown]
# Every pair comes back exactly, our sentence with its three feminine forms included. The model memorized; it
# hasn't learned to translate anything it hasn't seen. That's lesson 3's job, with half a million pairs. But the
# whole path works: tokens in, attention within and across both sentences, gradients back through every layer,
# tokens out.

# %% [markdown]
# ## 13. Our code, at the size of a real model
#
# Back to the table of section 1. opus-mt-en-ru has the shape of the 2017 base model, and our class implements that
# architecture. Build it with opus-mt-en-ru's numbers instead of ours, and count:

# %%
opus = configs["opus-mt-en-ru (2020)"]
opus_sized = Transformer(
    vocab_size=opus["vocab_size"],
    d_model=opus["d_model"],
    n_heads=opus["encoder_attention_heads"],
    d_ff=opus["encoder_ffn_dim"],
    n_layers=opus["encoder_layers"],
    dropout=0.1,
    pad_id=opus["pad_token_id"],
)
opus_sized_params = sum(p.numel() for p in opus_sized.parameters())
print("our class at opus-mt-en-ru's size:", opus_sized_params, "parameters")
print(round(opus_sized_params / by_formula, 1), "times the size of our model")

# %% [markdown]
# The real model's weights are a file of about 300 MB. Without downloading it, the Hub tells us its size, and at
# four bytes per number we can account for nearly every byte. Two differences from our model are known in advance:
# Marian stores its sinusoidal position tables in the file (one for the encoder and one for the decoder) instead of
# computing them, and it adds a bias to the output scores.

# %%
info = HfApi().model_info("Helsinki-NLP/opus-mt-en-ru", files_metadata=True)
file_size = [f.size for f in info.siblings if f.rfilename == "pytorch_model.bin"][0]

position_tables = 2 * opus["max_position_embeddings"] * opus["d_model"]
output_bias = opus["vocab_size"]
accounted = 4 * (opus_sized_params + position_tables + output_bias)

print("the real file:             ", file_size, "bytes")
print("our parameters and extras: ", accounted, "bytes")
left_over = file_size - accounted
print(f"left over: {left_over} bytes, {left_over / file_size:.3%} of the file")

# %% [markdown]
# What is left over is the file format itself: the tensors' names and headers. The translator we will compete with
# is, number for number, the model you just wrote, at several times our size. Lesson 3 trains ours on half a million
# sentence pairs, and then we'll look at where its attention goes when it writes *сказала*.
#
# The model's code is also in the course package, `tfs.model`, so that later lessons can import it instead of
# repeating it. Check that it is the same model, weight for weight:

# %%
from tfs.model import Transformer as PackagedTransformer

packaged = PackagedTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, DROPOUT, PAD)
packaged = packaged.to(device)
packaged.load_state_dict(model.state_dict())

src, tgt_in, _ = make_batch(lesson_pairs[:4])
model.eval()
packaged.eval()
with torch.no_grad():
    assert torch.allclose(packaged(src, tgt_in), model(src, tgt_in), atol=1e-5)
print("tfs.model.Transformer is the model of this lesson")

# %% [markdown]
# ## References
#
# - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
# - Bahdanau, D., Cho, K., Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). ICLR. *Attention itself was invented for translation, three years before the Transformer.*
# - He, K., Zhang, X., Ren, S., Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.
# - Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450).
# - Press, O., Wolf, L. (2017). [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859). EACL.
# - Rush, A. (2018). [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). *The paper, line by line, in PyTorch.*
# - Junczys-Dowmunt, M. et al. (2018). [Marian: Fast Neural Machine Translation in C++](https://arxiv.org/abs/1804.00344). ACL.
# - Tiedemann, J., Thottingal, S. (2020). [OPUS-MT: Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/). EAMT.
# - Karpathy, A. (2019). [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).
# - Model configurations: [GPT-2](https://huggingface.co/openai-community/gpt2), [opus-mt-en-ru](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru), [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3).
