# %% [markdown]
# # Lesson 3. From parrot to translator: training and decoding
#
# Ask a chatbot the same question twice and you may get two different answers. Ask a translation model twice and
# you get the same translation both times. The networks inside are close relatives, so the difference isn't in the
# model. It is in the step after it: every model of this kind produces probabilities for the next token, and a
# separate procedure, **decoding**, turns those probabilities into text. Here is how four real models are shipped
# to decode, read from their own configuration files:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/03/hero-light.png" width="760" alt="Decoding settings shipped with real models: the translators opus-mt-en-ru and m2m100 search with 4 and 5 beams, the chat models Llama 3.1 Instruct and Qwen 2.5 Instruct sample with temperatures 0.6 and 0.7">
#
# The translators *search* for the most probable sentence; the chat models *sample*, drawing each next token at
# random from its probabilities. By the end of this lesson you'll have written both, and seen why each side chose
# its way.
#
# First the model has to have probabilities worth decoding. In lesson 2 our Transformer could only parrot the
# token it had just read, or recite sixteen sentences it had learned by heart. In this lesson it trains on half a
# million sentence pairs, on a free GPU, in about half an hour, with the recipe of the 2017 paper: batches built
# from tokens, a smoothed loss, Adam with a warm-up, and half-precision arithmetic. Every piece is explained and
# tested before it goes into the loop.
#
# **What you'll do**
#
# 1. Turn half a million pairs into batches that waste little on padding.
# 2. Build the training step: label smoothing, Adam with the paper's learning-rate schedule, gradient clipping,
#    mixed precision. See what warm-up is for by training without it.
# 3. Train the lesson 2 Transformer and watch it go from parrot to translator, epoch by epoch.
# 4. Decode three ways: greedy, beam search and sampling.
# 5. Look inside: does *сказала* now look at *she*? And put our model next to opus-mt-en-ru.
#
# **What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type →
# T4 GPU*. The notebook downloads lesson 1's data and tokenizer and imports lesson 2's model. It runs in under an
# hour.

# %% tags=["include:setup"]

# %%
import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from tfs.model import Transformer
from tfs.viz import heatmap
from tokenizers import Tokenizer

SMOKE = env.smoke()  # CI runs the lesson on a tiny slice; you train in full
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" or SMOKE, "This lesson trains a model: switch to a GPU (see above)."
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# The data and the tokenizer of lesson 1, from the Hugging Face Hub:

# %%
tokenizer_path = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

splits = {}
for name in ["train", "validation", "test"]:
    path = hf_hub_download(
        "xmarva/tfs-tatoeba-en-ru", f"data/{name}.parquet", repo_type="dataset"
    )
    splits[name] = pd.read_parquet(path)
train, valid, test = splits["train"], splits["validation"], splits["test"]

VOCAB_SIZE = tokenizer.get_vocab_size()
PAD = tokenizer.token_to_id("<pad>")
BOS = tokenizer.token_to_id("<bos>")
EOS = tokenizer.token_to_id("<eos>")
print("train:", len(train), "validation:", len(valid), "test:", len(test))

# %% [markdown]
# ## 1. How real models turn probabilities into text
#
# A model on the Hugging Face Hub can come with a `generation_config.json`: the decoding settings its authors
# recommend. Read four of them, two translators and two chat models:

# %%
GENERATION_CONFIGS = {
    "opus-mt-en-ru (translator)": "Helsinki-NLP/opus-mt-en-ru",
    "m2m100 418M (translator)": "facebook/m2m100_418M",
    "Llama 3.1 8B Instruct (chat)": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Qwen 2.5 7B Instruct (chat)": "Qwen/Qwen2.5-7B-Instruct",
}

rows = []
for name, repo in GENERATION_CONFIGS.items():
    with open(hf_hub_download(repo, "generation_config.json")) as f:
        config = json.load(f)
    rows.append(
        {
            "model": name,
            "beams": config.get("num_beams", 1),
            "samples": config.get("do_sample", False),
            "temperature": config.get("temperature", "–"),
            "top-p": config.get("top_p", "–"),
        }
    )
table(pd.DataFrame(rows))

# %% [markdown]
# The two translators keep 4 and 5 **beams**: candidate translations followed in parallel, so that the model can
# choose the most probable whole sentence rather than the most probable next word. The chat models instead
# **sample** with a **temperature** below 1 and a **top-p** cut-off, settings that control how adventurous the
# random choice is. Sections 8 and 9 build all of this. Behind the difference is a different idea of a good output:
# a translation has to be faithful, an answer in a conversation has to be varied and alive.
#
# ## 2. The plan
#
# Training repeats one step thousands of times. Each piece of this loop is a section of the lesson:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/03/step-light.png" width="760" alt="One training step: a batch of sentence pairs goes through the Transformer, the loss compares the scores with the targets, backward computes the gradients, they are clipped, and Adam updates the weights with a learning rate from the schedule">
#
# In words: take a batch of sentence pairs, let the model score every possible next token at every position,
# measure how far the scores are from the right answers (the **loss**), compute how each weight should change to
# reduce it (the **gradients**, by backpropagation), and move every weight a small step in that direction. The size
# of the step is the **learning rate**, and how it changes over training is the **schedule**.

# %% [markdown]
# ## 3. Batches
#
# The GPU reads the data in batches, and the first decision is what a batch is. Start by turning every sentence
# into token ids once, so that the training loop never touches text. `encode_batch` runs the Rust tokenizer on many
# sentences in parallel:

# %%
if SMOKE:
    train = train.sample(3_000, random_state=SEED)
    valid = valid.head(300)


def encode_all(sentences):
    return [encoding.ids for encoding in tokenizer.encode_batch(list(sentences))]


start = time.perf_counter()
train_src, train_tgt = encode_all(train.en), encode_all(train.ru)
valid_src, valid_tgt = encode_all(valid.en), encode_all(valid.ru)
seconds = round(time.perf_counter() - start, 1)
print("tokenized", len(train_src) + len(valid_src), "pairs in", seconds, "s")

# %% [markdown]
# How long are the sentences, in tokens?

# %%
en_lengths = [len(ids) for ids in train_src]
ru_lengths = [len(ids) for ids in train_tgt]

fig, ax = plt.subplots(figsize=(6.4, 3))
bins = np.arange(0, 40)
ax.hist(en_lengths, bins=bins, color=COLORS["enc"], alpha=0.75, label="English")
ax.hist(ru_lengths, bins=bins, color=COLORS["dec"], alpha=0.6, label="Russian")
ax.set_xlabel("tokens per sentence")
ax.set_ylabel("sentences")
ax.set_title("Tatoeba sentences are short")
ax.legend()
plt.show()

print("longest English:", max(en_lengths), "tokens; longest Russian:", max(ru_lengths))

# %% [markdown]
# Most sentences take a handful of tokens, and a long thin tail reaches far to the right: the paragraphs that
# lesson 1 let through. Positions beyond a sentence's length are padding, and a padded position costs the GPU as
# much as a real one, so one paragraph in a batch would turn every other sentence of that batch into mostly
# padding. We leave out the few pairs longer than 64 tokens on either side:

# %%
MAX_LEN = 64

keep = []
for i in range(len(train_src)):
    if len(train_src[i]) <= MAX_LEN and len(train_tgt[i]) + 1 <= MAX_LEN:  # +1: <bos>/<eos>
        keep.append(i)
print("training pairs kept:", len(keep), "of", len(train_src))

# %% [markdown]
# Now the batches. A fixed number of sentences per batch would make the batches of long sentences much more
# expensive than the others, so the size of a batch is measured in **tokens**: a batch takes sentences until it
# would exceed `MAX_TOKENS` positions, padding included. And because padding is wasted work, it matters *which*
# sentences share a batch. Compare a random batch with a batch of sentences of similar length; violet cells are
# real tokens, grey ones padding:


# %%
def pad_batch(sequences):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), PAD)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids)
    return batch


by_length = sorted(keep, key=lambda i: len(train_tgt[i]))  # shortest Russian first
examples = {
    "random": random.sample(keep, 12),
    "similar lengths": by_length[len(by_length) // 2 :][:12],  # 12 pairs from the middle
}

for name, rows_of_batch in examples.items():
    batch = pad_batch([train_tgt[i] for i in rows_of_batch])
    padding = (batch == PAD).float().mean().item()
    title = f"A {name} batch: {padding:.0%} padding"
    heatmap(batch != PAD, low="grid", cell=0.18, title=title)
plt.show()

# %% [markdown]
# The standard fix is **bucketing**: sort the pairs by length, cut the sorted list into batches, and shuffle the
# order of the batches, so that the model still sees a random mix of lengths over an epoch while every single batch
# is nearly free of padding. `make_batches` does exactly that and builds the three tensors a training step needs,
# as in lesson 2: the English ids, the Russian ids behind `<bos>` that the decoder reads, and the Russian ids
# followed by `<eos>` that it must predict.


# %%
def make_batches(src, tgt, rows, max_tokens, sort=True):
    if sort:
        # similar lengths side by side
        rows = sorted(rows, key=lambda i: (len(tgt[i]), len(src[i])))
    batches, current, longest = [], [], 0
    for i in rows:
        new_longest = max(longest, len(src[i]), len(tgt[i]) + 1)
        if current and new_longest * (len(current) + 1) > max_tokens:  # the batch is full
            batches.append(current)
            current, new_longest = [], max(len(src[i]), len(tgt[i]) + 1)
        current.append(i)
        longest = new_longest
    batches.append(current)

    tensors = []
    for batch in batches:
        src_ids = pad_batch([src[i] for i in batch])
        tgt_in = pad_batch([[BOS] + tgt[i] for i in batch])
        tgt_out = pad_batch([tgt[i] + [EOS] for i in batch])
        tensors.append((src_ids, tgt_in, tgt_out))
    return tensors


def padding_share(batches):
    real, total = 0, 0
    for src_ids, _, tgt_out in batches:
        real += int((src_ids != PAD).sum() + (tgt_out != PAD).sum())
        total += src_ids.numel() + tgt_out.numel()
    return 1 - real / total


MAX_TOKENS = 6_000  # positions per batch and side, padding included

train_batches = make_batches(train_src, train_tgt, keep, MAX_TOKENS)
valid_batches = make_batches(valid_src, valid_tgt, range(len(valid_src)), MAX_TOKENS)
print("training batches:", len(train_batches), "validation batches:", len(valid_batches))
print(f"padding in the bucketed batches: {padding_share(train_batches):.1%}")

# %% [markdown]
# Little padding left. For comparison, the same token budget with the pairs in random order:

# %%
random_rows = random.sample(keep, len(keep))
unsorted = make_batches(train_src, train_tgt, random_rows, MAX_TOKENS, sort=False)
print(f"padding in random batches: {padding_share(unsorted):.1%}")
print("random batches:", len(unsorted), "bucketed:", len(train_batches))
del unsorted  # only needed for the comparison

# %% [markdown]
# Random batches waste a large share of every step on padding, and need more steps to cover the same data. One
# trade-off of bucketing: a batch now holds sentences of one length, so its gradient is a little less
# representative of the whole data than a random batch's. Shuffling the batch order every epoch keeps that
# harmless in practice, and the paper's authors batched the same way.
#
# ## 4. The loss, smoothed
#
# The loss is the cross-entropy of lesson 2: for every position, $-\log$ of the probability the model gives the
# right token. Written as a sum over the vocabulary, it compares the model's distribution $p$ with a target
# distribution $q$ that puts all its mass on the right token $y$:
#
# $$\mathcal{L} = -\sum_{j=1}^{\lvert\mathcal{V}\rvert} q_j \log p_j, \qquad q_j = \begin{cases} 1 & j = y \\ 0 &
# \text{otherwise.} \end{cases}$$
#
# That target is a demand for certainty, and the model can only approach it by pushing the right token's score
# ever higher above the others. On data like ours, where *He's stupid.* has twenty-five correct translations, total
# certainty is wrong anyway. **Label smoothing** ([Szegedy et al., 2016](https://arxiv.org/abs/1512.00567)), used
# in the 2017 paper with $\varepsilon = 0.1$, softens the target: it keeps $1 - \varepsilon$ of the mass on the
# right token and spreads $\varepsilon$ evenly over the whole vocabulary,
#
# $$q_j = (1 - \varepsilon)\,[j = y] + \frac{\varepsilon}{\lvert\mathcal{V}\rvert}.$$
#
# The model is then rewarded for being *mostly* sure. The paper reports that this hurts perplexity, the model's
# confidence in the right tokens, but improves the translations. Check the formula against PyTorch on a vocabulary
# of five tokens, small enough to follow:

# %%
EPSILON = 0.1

scores = torch.tensor([[2.0, 0.5, -1.0, 0.0, 1.0]])  # one position, five tokens
right = torch.tensor([0])  # the right token is token 0
p = scores.softmax(dim=-1)

q = torch.full((1, 5), EPSILON / 5)  # spread epsilon evenly...
q[0, 0] += 1 - EPSILON  # ...and keep the rest on the right token
ours = -(q * p.log()).sum()

theirs = F.cross_entropy(scores, right, label_smoothing=EPSILON)
assert torch.allclose(ours, theirs)
print("smoothed target q:", q)
print("loss:", round(ours.item(), 4), "= PyTorch's", round(theirs.item(), 4))

# %% [markdown]
# A consequence to remember when reading the training curves: with a smoothed target the loss can't reach zero. Its
# minimum is reached when $p = q$, and then it equals the entropy of $q$ itself:

# %%
q_right = 1 - EPSILON + EPSILON / VOCAB_SIZE
q_other = EPSILON / VOCAB_SIZE
floor = -(q_right * math.log(q_right) + (VOCAB_SIZE - 1) * q_other * math.log(q_other))
print("lowest possible smoothed loss:", round(floor, 3))

# %% [markdown]
# That's why we will watch two different numbers: the smoothed loss the model trains on, which levels off above this
# floor, and the plain cross-entropy on the validation set, which is what we actually want to be low.
#
# ## 5. The optimizer and the learning rate
#
# The optimizer is **Adam** ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)). Plain gradient descent moves
# every weight by the learning rate times its gradient. Adam keeps two running averages per weight, of the gradient
# $g$ and of its square, and moves each weight by the first divided by the square root of the second:
#
# $$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2, \qquad
# \theta_t = \theta_{t-1} - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$
#
# The average $m$ smooths out the noise of individual batches (it acts as momentum); dividing by $\sqrt{v}$ gives
# every weight a step of a similar size, whether its gradients are large or tiny. The hats mark a correction for the
# averages starting at zero. The paper uses $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$.
#
# The learning rate $\eta_t$ follows a schedule from the paper:
#
# $$\eta_t = d_{\text{model}}^{-0.5} \cdot \min\!\big(t^{-0.5},\ t \cdot t_{\text{warm}}^{-1.5}\big).$$
#
# For the first $t_{\text{warm}}$ steps the second term is the smaller one, and the rate grows linearly from zero:
# the **warm-up**. After that the first term takes over and the rate decays like $1/\sqrt{t}$. The factor
# $d_{\text{model}}^{-0.5}$ makes wider models take smaller steps.


# %%
def learning_rate(step, d_model=256, warmup=4_000):
    step = max(step, 1)  # step 0 would divide by zero
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


WARMUP = 4_000
steps = np.arange(1, 30_000)
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.plot(steps, [learning_rate(s, warmup=WARMUP) for s in steps], color=COLORS["enc"])
ax.axvline(WARMUP, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.set_title("Warm up, then decay")
plt.show()

peak = learning_rate(WARMUP, warmup=WARMUP)
print("peak learning rate:", f"{peak:.2e}", "at step", WARMUP)

# %% [markdown]
# Why start slowly? At the beginning, Adam's running average $v$ has seen only a few gradients, so its estimate of
# their size is poor and some steps come out far too large ([Liu et al., 2020](https://arxiv.org/abs/1908.03265)).
# And a Transformer with normalization after the residual addition, like ours, has especially large gradients near
# its output at initialization ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). Large early steps can
# throw the weights somewhere training never recovers from.
#
# Rather than take this on trust, train two copies of the model for a few hundred steps with the same peak learning
# rate, one with a warm-up and one without:


# %%
def new_model():
    return Transformer(VOCAB_SIZE, 256, 4, 1024, 3, 0.1, PAD).to(device)


def adam(model):
    # the learning rate is set before every step, from the schedule
    return torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


def train_steps(model, optimizer, schedule, n_steps, first_step=1, use_amp=True, clip=1.0):
    """Train for n_steps on shuffled batches; returns the smoothed loss of every step."""
    use_amp = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    model.train()
    losses = []
    while len(losses) < n_steps:
        order = random.sample(range(len(train_batches)), len(train_batches))  # every pass
        for index in order:
            src_ids, tgt_in, tgt_out = (t.to(device) for t in train_batches[index])
            for group in optimizer.param_groups:
                group["lr"] = schedule(first_step + len(losses))

            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                logits = model(src_ids, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE).float(),
                    tgt_out.reshape(-1),
                    ignore_index=PAD,
                    label_smoothing=EPSILON,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # gradients of the scaled loss (section 6)
            scaler.unscale_(optimizer)  # back to their true size before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            if len(losses) == n_steps:
                break
    return losses


# %% [markdown]
# `train_steps` is the whole loop of the picture in section 2; the rest of this section and the next explain its
# last few lines.
#
# One reference point helps to read the experiment: the loss of a model that ignores the sentence altogether and
# always predicts the same distribution, how often each token occurs in the training targets. A model that has
# learned nothing more than that sits at this loss:

# %%
counts = torch.zeros(VOCAB_SIZE)
for _, _, tgt_out in train_batches:
    counts += torch.bincount(tgt_out[tgt_out != PAD], minlength=VOCAB_SIZE)
frequencies = (counts + 1) / (counts + 1).sum()  # +1 to every count: no probability is 0
log_f = frequencies.log()

# the smoothed loss of always predicting the frequencies:
# 1 - eps of the target on the right token, eps spread evenly over all of them
right_part = -(counts * log_f).sum() / counts.sum()
spread_part = -log_f.mean()
frequency_only = (1 - EPSILON) * right_part + EPSILON * spread_part
frequency_only = frequency_only.item()
print("smoothed loss of predicting only the token frequencies:", round(frequency_only, 2))

# %% [markdown]
# For the experiment, a high peak rate makes the effect easy to see:

# %%
PEAK = 3e-3
EXPERIMENT_STEPS = 1_500 if not SMOKE else 10
experiments = {
    "no warm-up": lambda step: PEAK,
    "warm-up over 1 000 steps": lambda step: PEAK * min(1.0, step / 1_000),
}

curves = {}
for name, schedule in experiments.items():
    torch.manual_seed(SEED)  # the same starting weights for both
    model = new_model()
    curves[name] = train_steps(model, adam(model), schedule, EXPERIMENT_STEPS)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for (name, losses), color in zip(curves.items(), [COLORS["dec"], COLORS["enc"]]):
    smooth = pd.Series(losses).rolling(25, min_periods=1).mean()  # average out the noise
    ax.plot(smooth, color=color, label=name)
ax.axhline(frequency_only, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)
label = "knows only the token frequencies"
ax.text(EXPERIMENT_STEPS, frequency_only + 0.15, label, ha="right", fontsize=9)
ax.set_xlabel("step")
ax.set_ylabel("training loss (smoothed)")
ax.set_title("The same peak learning rate, with and without warm-up")
ax.legend()
plt.show()

# %% [markdown]
# Same model, same data, same peak learning rate; only the first thousand steps differ. With the warm-up, the loss
# falls steadily. Without it, the first steps at full speed throw the weights somewhere the model doesn't recover
# from: the loss sticks near the dashed line, where a model knows little more than which tokens are frequent. In
# the real run below the warm-up lasts longer and the peak is lower, but the reason for it is the one you just saw.
#
# The last protection is **gradient clipping**, the `clip_grad_norm_` line: if the length of the whole gradient
# vector, over all the model's weights, exceeds a limit $c$, the vector is scaled down to that length,
#
# $$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right).$$
#
# The direction stays, only an occasional huge step is cut short ([Pascanu et al., 2013](https://arxiv.org/abs/1211.5063)).
# The 2017 paper doesn't mention it; nearly every recipe since uses it, with $c = 1$.
#
# ## 6. Half precision
#
# A GPU computes much faster with 16-bit numbers than with the usual 32-bit ones: half the memory traffic, and on
# the T4 special units (tensor cores) built for 16-bit matrix products. The price is range and precision:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/03/formats-light.png" width="760" alt="Bit layouts of float32, bfloat16 and float16, with their largest values and smallest normal values; float16 has only 5 exponent bits and covers a much smaller range">
#
# float16 keeps 10 bits of precision, plenty for activations and weights in a matrix product, but only 5 bits of
# exponent: nothing above 65 504, nothing below about $6 \cdot 10^{-8}$. Gradients are often smaller than that, and
# a gradient rounded to zero teaches nothing:

# %%
tiny_gradient = torch.tensor(1e-8)
print("in float16:", tiny_gradient.half().item())
print("times 65 536, in float16:", (tiny_gradient * 65_536).half().item())

# %% [markdown]
# Hence **mixed precision** ([Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)): matrix products run in
# float16 under `torch.autocast`, while the weights and the sensitive operations (softmax, the loss) stay in
# float32. And **loss scaling**: multiply the loss by a large factor before backpropagation, so every gradient is
# multiplied by it too and stays representable, then divide the gradients back before the optimizer uses them.
# `GradScaler` does both and picks the factor itself: it raises it while all goes well and lowers it, skipping that
# step, whenever a gradient overflows. In `train_steps` these are the lines with `autocast` and `scaler`.
#
# What does it buy? Time a few steps each way:

# %%
TIMING_STEPS = 40 if not SMOKE else 3
timings = {}
for name, use_amp in [("float32", False), ("mixed precision", True)]:
    torch.manual_seed(SEED)
    model = new_model()
    optimizer = adam(model)
    train_steps(model, optimizer, lambda step: 5e-4, 5, use_amp=use_amp)  # warm the GPU up
    if device == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish before reading the clock
    start = time.perf_counter()
    train_steps(model, optimizer, lambda step: 5e-4, TIMING_STEPS, use_amp=use_amp)
    if device == "cuda":
        torch.cuda.synchronize()
    timings[name] = (time.perf_counter() - start) / TIMING_STEPS

speedup = timings["float32"] / timings["mixed precision"]
for name, seconds in timings.items():
    print(f"{name}: {seconds * 1000:.0f} ms per step")
print(f"mixed precision is {speedup:.1f} times faster")

# %% [markdown]
# ## 7. Training
#
# Everything is in place. The training run uses the paper's schedule with a warm-up of `WARMUP` steps, label
# smoothing and mixed precision, and after every epoch, one pass over all the batches, it does three things:
# measures the plain cross-entropy on the validation set, keeps the weights if they are the best so far, and
# translates a few probe sentences, so that we can watch the model learn.


# %%
@torch.no_grad()
def validation_loss(model):
    model.eval()
    total, count = 0.0, 0
    for src_ids, tgt_in, tgt_out in valid_batches:
        src_ids, tgt_in, tgt_out = src_ids.to(device), tgt_in.to(device), tgt_out.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(src_ids, tgt_in)
        flat = logits.reshape(-1, VOCAB_SIZE).float()
        loss = F.cross_entropy(flat, tgt_out.reshape(-1), ignore_index=PAD, reduction="sum")
        total += loss.item()
        count += int((tgt_out != PAD).sum())
    return total / count  # plain cross-entropy per target token


@torch.no_grad()
def translate_greedy(model, sentence, max_tokens=60):
    model.eval()
    src_ids = torch.tensor([tokenizer.encode(sentence).ids], device=device)
    memory = model.encode(src_ids)
    written = [BOS]
    for _ in range(max_tokens):
        tgt = torch.tensor([written], device=device)
        next_id = model.decode(tgt, memory, src_ids)[0, -1].argmax().item()
        if next_id == EOS:
            break
        written.append(next_id)
    return tokenizer.decode(written[1:])


EN_EXAMPLE = "The researcher said she was exhausted."
PROBES = [EN_EXAMPLE, "Tom forgot to lock the door.", "I can't believe you did that."]

# %% [markdown]
# The run: `EPOCHS` passes over the data with the paper's schedule. After every epoch the loop measures the
# validation loss, saves the weights if they are the best so far, and translates the three probe sentences. It
# prints one line per epoch, with the minutes since the start; this is the long cell of the lesson.

# %%
EPOCHS = 20 if not SMOKE else 1
CHECKPOINT = WORK_DIR / "transformer.pt"

torch.manual_seed(SEED)
random.seed(SEED)
model = new_model()
optimizer = adam(model)

def schedule(step):
    return learning_rate(step, warmup=WARMUP)


history = []
best = float("inf")
steps_done = 0
start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    n_steps = len(train_batches)  # one epoch
    losses = train_steps(model, optimizer, schedule, n_steps, first_step=steps_done + 1)
    steps_done += len(losses)

    record = {
        "epoch": epoch,
        "step": steps_done,
        "training loss": float(np.mean(losses)),
        "validation loss": validation_loss(model),
        "minutes": (time.perf_counter() - start) / 60,
    }
    for probe in PROBES:
        record[probe] = translate_greedy(model, probe)
    history.append(record)

    if record["validation loss"] < best:
        best = record["validation loss"]
        torch.save(model.state_dict(), CHECKPOINT)
    print(
        f"epoch {epoch:2d}  training {record['training loss']:.3f}  "
        f"validation {record['validation loss']:.3f}  {record['minutes']:.1f} min"
    )

# %% [markdown]
# The validation loss is a cross-entropy, and its exponent has a readable meaning, the **perplexity**: the model is
# as unsure about the next token, on average, as if it were choosing uniformly among that many.

# %%
print("best validation loss:", round(best, 3))
print("perplexity:", round(math.exp(best), 2))

# %%
history_table = pd.DataFrame(history)

epochs = history_table.epoch
fig, ax = plt.subplots(figsize=(6.4, 3.4))
training, validation = history_table["training loss"], history_table["validation loss"]
ax.plot(epochs, training, marker="o", color=COLORS["enc"], label="training, smoothed")
ax.plot(epochs, validation, marker="o", color=COLORS["dec"], label="validation, plain")
ax.axhline(floor, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)
ax.text(epochs.max(), floor + 0.05, "floor of the smoothed loss", ha="right", fontsize=9)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Training")
ax.legend()
plt.show()

# %% [markdown]
# The validation loss falls fast for a few epochs and then flattens: most of what this model can learn from
# Tatoeba, it learns early. The training curve lies *above* the validation one, which looks backwards and isn't:
# it is measured with label smoothing, which can't go below the dashed floor, and with dropout switching off a
# tenth of the model at every step. Only the validation curve is plain cross-entropy.
#
# > The loss you train on and the number you judge by are different numbers. Compare like with like.
#
# The probe sentences at a few points of the run:

# %%
shown_epochs = [1, 2, 3, 5, 10, EPOCHS]
progress = history_table[history_table.epoch.isin(shown_epochs)]
table(progress[["epoch", *PROBES]])

# %% [markdown]
# In the first epoch the model strings frequent words together. A few epochs later the everyday sentences come out
# right and stay right; where two translations are equally good (*Не могу поверить*, *Поверить не могу*), the model
# may switch between them from one epoch to the next. Our own sentence never comes out right. Section 10 finds out
# why.

# %% [markdown]
# ## 8. Decoding by search
#
# The trained model gives, for any English sentence and any Russian prefix, a probability for every possible next
# token. A translation needs a whole sentence, and the step from one to the other is less innocent than it looks.
# In training the decoder always read the correct Russian prefix; when it translates, it reads its own output, and
# an early mistake becomes the history every later word is written from. Here is a real case, from the translator
# we will compete with:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/03/exposure-light.png" width="760" alt="In training the decoder reads the reference translation; when translating it reads its own words. opus-mt-en-ru writes Исследователь сказал, что она устала: a masculine first word, and the verb agrees with the mistake">
#
# The first word came out masculine, and the verb, written after it, agrees with it. The sentence is good Russian,
# and it says something else: a man said that some woman was tired. Two people where the English had one.
#
# Start from the best model of the training run, and from greedy decoding, the simplest procedure: at every step
# take the single most probable token. Next to each translation, print its **score**, the log-probability the model
# gives the whole sentence, $\log P(\mathbf{y} \mid \mathbf{x}) = \sum_t \log p(y_t \mid y_{<t}, \mathbf{x})$: the
# sum of the log-probabilities of its tokens, closer to 0 for more probable sentences.

# %%
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


@torch.no_grad()
def sentence_score(model, sentence, translation):
    """log P(translation | sentence): the sum over its tokens, <eos> included."""
    src_ids = torch.tensor([tokenizer.encode(sentence).ids], device=device)
    ids = tokenizer.encode(translation).ids
    tgt_in = torch.tensor([[BOS] + ids], device=device)
    tgt_out = torch.tensor([ids + [EOS]], device=device)
    log_probs = model(src_ids, tgt_in).float().log_softmax(dim=-1)  # (1, T, |V|)
    return log_probs[0, torch.arange(len(ids) + 1), tgt_out[0]].sum().item()


MINIMAL_PAIRS = [
    "The researcher said she was exhausted.",
    "The researcher said he was exhausted.",
    "The teacher said she was tired.",
    "The teacher said he was tired.",
    "My neighbour said she was busy.",
    "My neighbour said he was busy.",
]

rows = []
for sentence in MINIMAL_PAIRS:
    translation = translate_greedy(model, sentence)
    score = sentence_score(model, sentence, translation)
    rows.append({"English": sentence, "greedy": translation, "score": score})
table(pd.DataFrame(rows))

# %% [markdown]
# The pronoun is translated right every time, *она* for *she* and *он* for *he*, with a verb that agrees with it.
# The subject is not: whatever the pronoun says, the model writes the masculine form of the profession and of the
# verb after it. Hold that thought until section 10.
#
# Greedy decoding commits to every token for good. The most probable next token, though, isn't always the start of
# the most probable sentence:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/03/beam-toy-light.png" width="760" alt="A toy model: greedy decoding takes the most likely first token A, then its best continuation x, with probability 0.20; beam search keeps the two best first tokens and finds B then z, with probability 0.36">
#
# **Beam search** keeps the $k$ most probable partial translations (the beam) instead of one. At every step it
# extends each of them by every possible token, scores all $k \cdot \lvert\mathcal{V}\rvert$ candidates by their
# total log-probability, and keeps the best $k$. A candidate that ends with `<eos>` is finished and set aside. With
# $k = 1$ it is greedy decoding; with $k$ as large as the number of possible sentences it would find the most
# probable translation exactly, at an impossible cost. In between, a handful of beams already catches most of the
# cases where greedy goes wrong.
#
# One correction is needed. Every token multiplies the probability by a number below 1, so longer sentences always
# have lower scores, and a plain beam search prefers translations that stop early. The usual fix divides the score
# by a **length penalty** ([Wu et al., 2016](https://arxiv.org/abs/1609.08144)),
#
# $$\operatorname{lp}(\mathbf{y}) = \left(\frac{5 + \lvert\mathbf{y}\rvert}{6}\right)^{\alpha},$$
#
# which grows with the length $\lvert\mathbf{y}\rvert$; $\alpha = 0$ switches it off, and the 2017 paper used
# $\alpha = 0.6$ with $k = 4$.


# %%
def length_penalty(length, alpha=0.6):
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search(model, sentence, beam_size=4, alpha=0.6, max_tokens=60, trace=None):
    model.eval()
    src_ids = torch.tensor([tokenizer.encode(sentence).ids], device=device)
    memory = model.encode(src_ids)
    beams = [([BOS], 0.0)]  # (tokens so far, their total log-probability)
    finished = []

    for _ in range(max_tokens):
        k = len(beams)
        tgt = torch.tensor([tokens for tokens, _ in beams], device=device)  # one batch
        scores = model.decode(tgt, memory.expand(k, -1, -1), src_ids.expand(k, -1))[:, -1]
        log_probs = scores.float().log_softmax(dim=-1)  # (k, |V|)

        candidates = []
        for (tokens, total), row in zip(beams, log_probs):
            top = row.topk(beam_size)  # more than beam_size per beam can never survive
            for log_p, token_id in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((tokens + [token_id], total + log_p))
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)  # best first

        beams = []
        for tokens, total in candidates:
            if tokens[-1] == EOS:
                finished.append((tokens, total))
            else:
                beams.append((tokens, total))
            if len(beams) == beam_size:
                break
        if trace is not None:
            trace.append([(tokenizer.decode(tokens[1:]), total) for tokens, total in beams])
        if len(finished) >= beam_size:
            break

    def normalized(candidate):
        tokens, total = candidate
        return total / length_penalty(len(tokens) - 1, alpha)  # length without <bos>

    best_tokens, _ = max(finished or beams, key=normalized)
    return tokenizer.decode(best_tokens[1:])  # decode drops <eos>


# %% [markdown]
# Watch the four beams grow on our sentence, step by step, each with its total log-probability:

# %%
trace = []
beam_search(model, EN_EXAMPLE, trace=trace)
rows = []
for step, beams in enumerate(trace):
    for rank, (text, total) in enumerate(beams):
        row = {"step": step + 1, "beam": rank + 1, "partial translation": text}
        row["log-probability"] = total
        rows.append(row)
    if step >= 5:
        break
table(pd.DataFrame(rows))

# %% [markdown]
# At every step four partial translations survive, and the ranking changes as they grow: a prefix that leads at one
# step can fall behind at the next. Greedy decoding would have committed to the first leader for good.
#
# Compare the two decoders on validation sentences: how often beam search changes the translation, and whether the
# translations it picks are more probable under the model, which is what it searches for.

# %%
n_sample = 150 if not SMOKE else 5
sample_pairs = valid.drop_duplicates("en").sample(n_sample, random_state=SEED)

rows = []
start = time.perf_counter()
for en, ru in zip(sample_pairs.en, sample_pairs.ru):
    greedy = translate_greedy(model, en)
    beam = beam_search(model, en)
    rows.append(
        {
            "English": en,
            "reference": ru,
            "greedy": greedy,
            "beam": beam,
            "greedy score": sentence_score(model, en, greedy),
            "beam score": sentence_score(model, en, beam),
        }
    )
decoded = pd.DataFrame(rows)
seconds = round(time.perf_counter() - start)
print("decoded", len(decoded), "sentences twice in", seconds, "s")

changed = decoded[decoded.greedy != decoded.beam]
print(f"beam search changed {len(changed)} of {len(decoded)} translations")
print("average score, greedy:", round(decoded["greedy score"].mean(), 3))
print("average score, beam:  ", round(decoded["beam score"].mean(), 3))
table(changed[["English", "reference", "greedy", "beam"]].head(6))

# %% [markdown]
# Beam search changes some of the translations, and the ones it picks score higher under the model on average, which
# is what it searches for. Whether they are also better *translations* is a separate question, sentence by
# sentence; lesson 4 answers it with metrics that compare outputs with the references.
#
# ## 9. Decoding by sampling
#
# Search looks for the single most probable output. For a translation that's what we want. For a chat model, or any
# open-ended text, it is not: the most probable continuation of a conversation is bland, and beam search on long
# open text famously falls into repeating itself ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).
# Those models **sample** instead: at every step they draw the next token at random, with the probabilities the
# model gives.
#
# Two knobs shape the draw. The **temperature** $\tau$ divides the scores before softmax,
#
# $$p_j = \frac{e^{z_j / \tau}}{\sum_k e^{z_k / \tau}}:$$
#
# $\tau < 1$ sharpens the distribution towards the most probable tokens, $\tau > 1$ flattens it, and as $\tau \to 0$
# sampling becomes greedy decoding. Here is the distribution of the next token after *Исследовательница сказала,
# что* at three temperatures:

# %%
prefix = "Исследовательница сказала, что"
src_ids = torch.tensor([tokenizer.encode(EN_EXAMPLE).ids], device=device)
tgt = torch.tensor([[BOS] + tokenizer.encode(prefix).ids], device=device)
with torch.no_grad():
    next_scores = model(src_ids, tgt)[0, -1].float().cpu()

top_ids = next_scores.topk(8).indices.tolist()
top_tokens = [tokenizer.id_to_token(i) for i in top_ids]
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, temperature in zip(axes, [0.5, 1.0, 1.5]):
    probs = (next_scores / temperature).softmax(dim=-1)
    ax.bar(range(8), probs[top_ids], color=COLORS["data"])
    ax.set_xticks(range(8), top_tokens, rotation=60, fontsize=9)
    ax.set_title(f"temperature {temperature}", fontsize=10)
axes[0].set_ylabel("probability")
plt.tight_layout()
plt.show()

# %% [markdown]
# A low temperature hands almost everything to the most probable tokens; a high one lets the alternatives catch up.
#
# The second knob, **top-p** or nucleus sampling, cuts off the long tail: sort the tokens by probability, keep the
# smallest set whose probabilities add up to at least $p$, and sample only from that set, renormalized. The tail of
# thousands of individually unlikely tokens holds a surprising share of the probability in total, and a single draw
# from it can derail a sentence.


# %%
@torch.no_grad()
def translate_sampling(model, sentence, temperature=1.0, top_p=1.0, max_tokens=60):
    model.eval()
    src_ids = torch.tensor([tokenizer.encode(sentence).ids], device=device)
    memory = model.encode(src_ids)
    written = [BOS]
    for _ in range(max_tokens):
        tgt = torch.tensor([written], device=device)
        scores = model.decode(tgt, memory, src_ids)[0, -1].float()
        probs = (scores / temperature).softmax(dim=-1)

        if top_p < 1.0:
            sorted_probs, order = probs.sort(descending=True)
            before = sorted_probs.cumsum(dim=0) - sorted_probs  # mass of the tokens ahead
            in_nucleus = before < top_p  # keep tokens until the running total reaches top_p
            probs = torch.zeros_like(probs)
            probs[order[in_nucleus]] = sorted_probs[in_nucleus]
            probs = probs / probs.sum()

        next_id = torch.multinomial(probs, 1).item()  # one random draw
        if next_id == EOS:
            break
        written.append(next_id)
    return tokenizer.decode(written[1:])


torch.manual_seed(SEED)
rows = []
for name, settings in [
    ("temperature 1", dict(temperature=1.0)),
    ("temperature 0.6, top-p 0.9", dict(temperature=0.6, top_p=0.9)),
    ("temperature 1.5", dict(temperature=1.5)),
]:
    for _ in range(3):
        translation = translate_sampling(model, EN_EXAMPLE, **settings)
        rows.append({"settings": name, "translation": translation})
table(pd.DataFrame(rows))

# %% [markdown]
# At temperature 1 the samples wander: a draw from the long tail of sixteen thousand tokens now and then, and the
# tail includes English tokens, because lesson 1 built one vocabulary for both languages. At 1.5 the output falls
# apart. A low temperature with a top-p cut-off, the setting of the chat models in section 1, stays fluent and still
# varies from draw to draw. For a translation that variety is no virtue: we want the one best reading, and that's
# why translators search.
#
# ## 10. Inside the trained model
#
# Lesson 2 ended with a flat map: after memorizing sixteen pairs, the cross-attention of the last decoder layer
# spread evenly over the English sentence. Here are the same maps for the trained model, one per decoder layer,
# averaged over the four heads. The model reads the reference translation of our sentence, and each row is the step
# at which it writes one Russian token: the row labelled *▁сказала* is where the model decides on *сказала*, and its
# cells show which English tokens it looks at while doing so.

# %%
RU_EXAMPLE = "Исследовательница сказала, что вымоталась."
ru_ids = tokenizer.encode(RU_EXAMPLE).ids
src_ids = torch.tensor([tokenizer.encode(EN_EXAMPLE).ids], device=device)
tgt_in = torch.tensor([[BOS] + ru_ids], device=device)
with torch.no_grad():
    logits = model(src_ids, tgt_in)  # the attention modules keep their weights

en_tokens = tokenizer.encode(EN_EXAMPLE).tokens
written = tokenizer.encode(RU_EXAMPLE).tokens + ["<eos>"]  # row t writes token t

for number, layer in enumerate(model.decoder, start=1):
    weights = layer.cross_attention.weights[0].mean(dim=0).float().cpu()  # (T_ru, T_en)
    heatmap(
        weights,
        rows=written,
        cols=en_tokens,
        vmax=1,
        title=f"Cross-attention, decoder layer {number}",
        xlabel="English (keys)",
        ylabel="the Russian token being written",
    )
plt.show()

# %% [markdown]
# ATTENTION MAPS PLACEHOLDER
#
# So the attention works. Why, then, does the model never write *Исследовательница*? Ask it directly: give it the
# reference translation and read off the probability it assigns to each token at the moment of writing it. Tokens it
# finds unlikely, below 0.1, are filled:

# %%
probs = logits[0].float().softmax(dim=-1)  # (T, |V|)
targets = ru_ids + [EOS]
token_probs = [probs[t, token_id].item() for t, token_id in enumerate(targets)]

token_chips(
    [written],
    ids=[[f"{p:.2f}" for p in token_probs]],  # the probability under each token
    strong=lambda row, col, token: token_probs[col] < 0.1,
)
plt.show()

# %% [markdown]
# PROBABILITIES PLACEHOLDER
#
# And while it writes that suffix, does it look at *she* at all? Every head of every layer, at that one step:

# %%
step = written.index("ница")
rows, labels = [], []
for number, layer in enumerate(model.decoder, start=1):
    for head in range(layer.cross_attention.n_heads):
        rows.append(layer.cross_attention.weights[0, head, step].float().cpu())
        labels.append(f"layer {number}, head {head + 1}")
at_suffix = torch.stack(rows)  # (layers × heads, T_en)

title = "Where each head looks while writing «ница»"
heatmap(at_suffix, rows=labels, cols=en_tokens, vmax=1, title=title)
plt.show()

she = en_tokens.index("▁she")
print("the most any head gives to ▁she here:", round(at_suffix[:, she].max().item(), 2))

# %% [markdown]
# HEADS PLACEHOLDER
#
# The rest of the answer is in the data. Count the sentences of the training set that contain the masculine and the
# feminine forms of a few nouns of people:

# %%
NOUN_PAIRS = [
    ("учитель", "учительница"),
    ("студент", "студентка"),
    ("сосед", "соседка"),
    ("исследователь", "исследовательница"),
]
rows = []
for masculine, feminine in NOUN_PAIRS:
    rows.append(
        {
            "masculine": masculine,
            "sentences": train.ru.str.contains(rf"\b{masculine}\b", case=False).sum(),
            "feminine": feminine,
            "sentences ": train.ru.str.contains(rf"\b{feminine}\b", case=False).sum(),
        }
    )
table(pd.DataFrame(rows))

# %% [markdown]
# DATA PLACEHOLDER

# %% [markdown]
# ## 11. Next to a real translator
#
# opus-mt-en-ru, the model we will compete with in lesson 4, is on the Hub with its weights, and the `transformers`
# library runs it in a few lines with its own default decoding, four beams. Put the two models side by side on the
# minimal pairs, and on a few sentences from the test set that neither has trained on:

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

opus_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
opus = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device).eval()


@torch.no_grad()
def translate_opus(sentences):
    batch = opus_tokenizer(list(sentences), return_tensors="pt", padding=True).to(device)
    output = opus.generate(**batch)  # its generation_config: 4 beams
    return opus_tokenizer.batch_decode(output, skip_special_tokens=True)


test_sentences = list(test.en.sample(6, random_state=SEED))
sentences = MINIMAL_PAIRS + test_sentences
rows = []
for sentence, theirs in zip(sentences, translate_opus(sentences)):
    ours = beam_search(model, sentence)
    rows.append({"English": sentence, "ours, beam of 4": ours, "opus-mt-en-ru": theirs})
table(pd.DataFrame(rows))

# %% [markdown]
# COMPARISON PLACEHOLDER
#
# ## 12. Save and publish
#
# Lesson 4 evaluates this model on the whole test set and needs its weights. We save them in the `safetensors`
# format, which stores tensors and nothing else, so loading a file can't run code, together with a `config.json`
# that records how to rebuild the model.

# %%
from safetensors.torch import save_file

MODEL_DIR = WORK_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 1024,
    "n_layers": 3,
    "dropout": 0.1,
    "pad_id": PAD,
}
save_file(model.state_dict(), MODEL_DIR / "model.safetensors")
(MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

reloaded = Transformer(**CONFIG).to(device)
from safetensors.torch import load_file

reloaded.load_state_dict(load_file(MODEL_DIR / "model.safetensors", device=device))
assert beam_search(reloaded, EN_EXAMPLE) == beam_search(model, EN_EXAMPLE)
print("saved and reloaded:", sorted(path.name for path in MODEL_DIR.iterdir()))

# %%
MODEL_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
tags: [translation, transformer]
datasets: [xmarva/tfs-tatoeba-en-ru]
---
# A small English→Russian Transformer, trained from scratch

Built in lesson 3 of the course "Transformers: from scratch to production":
https://github.com/xmarva/transformers-from-scratch

- the encoder-decoder Transformer of Vaswani et al. (2017) at half its base size:
  {CONFIG["n_layers"]} + {CONFIG["n_layers"]} layers, d_model {CONFIG["d_model"]},
  {CONFIG["n_heads"]} heads
- trained on xmarva/tfs-tatoeba-en-ru with the tokenizer xmarva/tfs-bpe-en-ru-16k
- best validation cross-entropy: {best:.3f}

The model class is `tfs.model.Transformer` from the course repository.
"""

HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")

if HUB_USER and hf_token and not SMOKE:
    api = HfApi(token=hf_token)
    model_repo = f"{HUB_USER}/tfs-transformer-en-ru"
    api.create_repo(model_repo, exist_ok=True)
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=model_repo)
    card = MODEL_CARD.encode()
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=model_repo)
    print("published", model_repo)
else:
    print("not publishing; lesson 4 downloads the course's copy:")
    print("xmarva/tfs-transformer-en-ru")

# %% [markdown]
# ## References
#
# - Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS. *The training recipe and the decoding settings.*
# - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (2016). [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). CVPR. *Label smoothing.*
# - Müller, R., Kornblith, S., Hinton, G. (2019). [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629). NeurIPS.
# - Kingma, D. P., Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR.
# - Liu, L. et al. (2020). [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265). ICLR. *Why Adam needs a warm-up.*
# - Xiong, R. et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745). ICML. *Why post-norm Transformers do.*
# - Pascanu, R., Mikolov, T., Bengio, Y. (2013). [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063). ICML. *Gradient clipping.*
# - Micikevicius, P. et al. (2018). [Mixed Precision Training](https://arxiv.org/abs/1710.03740). ICLR.
# - Wu, Y. et al. (2016). [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144). *The length penalty.*
# - Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y. (2020). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR. *Top-p sampling.*
# - Stanovsky, G., Smith, N. A., Zettlemoyer, L. (2019). [Evaluating Gender Bias in Machine Translation](https://arxiv.org/abs/1906.00591). ACL.
# - Tiedemann, J., Thottingal, S. (2020). [OPUS-MT: Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/). EAMT.
