# %% [markdown]
# # Lesson 8. Serving a Transformer with FastAPI and Gradio: latency under load
#
# In November 2006 Greg Linden, who had built Amazon's recommendations, gave a talk at Stanford. One slide said
# "+100 ms → −1% sales @ Amazon" and, next to it, "+500 ms → −20% traffic @ Google"
# ([Linden, 2006](https://glinden.blogspot.com/2006/12/slides-from-my-talk-at-stanford.html)). The two lines have
# been quoted ever since, mostly without the footnotes: the Google experiment also showed thirty results instead
# of ten, so it measured a heavier page, not just a slower one. When Google ran a clean experiment three years
# later, adding 100 to 400 ms to search cost 0.2–0.6% of searches
# ([Brutlag, 2009](https://services.google.com/fh/files/blogs/google_delayexp.pdf)). Smaller than the legend, and
# still enough for Google to measure it in fractions of a percent.
#
# The other half of the story is what happens when many people ask at once. A request that is fast alone can wait
# behind others, and the waiting, not the computing, is what users feel. Dean and Barroso's paper on Google's
# serving put the consequence bluntly: if a request touches a hundred servers and each is slow one time in a
# hundred, 63% of requests are slow
# ([Dean & Barroso, 2013](https://doi.org/10.1145/2408776.2408794)).
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/hero-light.png" width="760" alt="The share of slow requests against the number of servers a request waits for, when each server is slow one time in a hundred: 1% for one server, about 10% for ten, 63% for a hundred. The slowest part of a system decides how often its users wait">
#
# This lesson puts perevod behind an HTTP API, and then measures it the way users will meet it: under load. On
# the way it makes the model several times faster without changing a single translation, tries a famous speed-up
# that doesn't pay off here, and finds out why a queue, not the model, decides most of the latency.
#
# **What you'll do**
#
# 1. Time a real translator and ours, and find where a translation spends its time.
# 2. Bring lesson 5's KV cache into perevod's beam search, and check it changes no translation.
# 3. Try int8 quantization and measure why it doesn't help this model.
# 4. Build a FastAPI service, load it the way users do, and compare the curves with queueing theory.
# 5. Add dynamic batching, a Gradio page, and release perevod 0.2.0.
#
# **What you need**: a CPU, as in lesson 7; Kaggle's free CPU sessions will do. The lesson downloads opus-mt
# (300 MB) for a comparison and lesson 7's model (40 MB), and runs in about three quarters of an hour, a third of
# it in load tests that run in real time.

# %% tags=["include:setup"]

# %%
import asyncio
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

for package in ["sacrebleu", "httpx"]:  # Kaggle has them, Colab may not
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
import httpx
import sacrebleu

print("CPU cores Python sees:", os.cpu_count())
print("threads PyTorch uses:", torch.get_num_threads())

# %% [markdown]
# ## 1. How long a translation takes
#
# Start with a real model on this machine. opus-mt-en-ru, the baseline of the course, is a Marian Transformer with
# six encoder and six decoder layers, trained on millions of sentence pairs. Here it translates sentences one at a
# time, as a service would when requests arrive one by one, with the beam search it ships with:

# %%
from transformers import MarianMTModel, MarianTokenizer

OPUS = "Helsinki-NLP/opus-mt-en-ru"
opus_tokenizer = MarianTokenizer.from_pretrained(OPUS)
opus = MarianMTModel.from_pretrained(OPUS).eval()

DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
test_path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet",
                            repo_type="dataset", revision=DATA_REVISION)
test = pd.read_parquet(test_path)
probe = list(test.sample(30, random_state=SEED).en)


@torch.inference_mode()
def opus_translate(sentence):
    batch = opus_tokenizer([sentence], return_tensors="pt")
    output = opus.generate(**batch, num_beams=4, max_new_tokens=128)
    return opus_tokenizer.decode(output[0], skip_special_tokens=True)


def latencies(translate, sentences):
    times = []
    for sentence in sentences:
        start = time.perf_counter()
        translate(sentence)
        times.append(time.perf_counter() - start)
    return np.array(times)


opus_times = latencies(opus_translate, probe)
n_opus = sum(parameter.numel() for parameter in opus.parameters())
print(f"opus-mt: {n_opus / 1e6:.0f}M parameters, "
      f"median {np.median(opus_times) * 1000:.0f} ms per sentence")

# %% [markdown]
# And perevod, the package of lesson 7. Its source lives in the course repository; we take the release we made,
# by its tag, and only the `perevod/` folder of it, so the clone stays small. That folder is where this lesson
# works:

# %%
REPO = WORK_DIR / "course"
shutil.rmtree(REPO, ignore_errors=True)
URL = "https://github.com/xmarva/transformers-from-scratch"
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "perevod-v0.1.0",
                "--filter=blob:none", "--sparse", URL, str(REPO)], check=True)
subprocess.run(["git", "sparse-checkout", "set", "perevod"], cwd=REPO, check=True)
os.chdir(REPO)


def write(path, text):
    Path(path).write_text(text.lstrip("\n"), encoding="utf-8")


# !{sys.executable} -m pip install -q ./perevod
importlib.invalidate_caches()

import perevod
from perevod import Translator

translator = Translator.from_pretrained(device="cpu")
n_ours = sum(parameter.numel() for parameter in translator.model.parameters())
ours_times = latencies(translator, probe)
print(f"perevod {perevod.__version__}: {n_ours / 1e6:.1f}M parameters, "
      f"median {np.median(ours_times) * 1000:.0f} ms")

# %% [markdown]
# Ours is several times smaller and, on the same sentences, faster. A sentence in a few dozen milliseconds sounds
# like the end of the story. It isn't, for two reasons. The time per sentence depends on its length, and the next
# section shows how badly. And a service doesn't translate one sentence: it translates whatever arrives, whenever
# it arrives, and section 8 shows what that does.
#
# ## 2. Where a translation spends its time
#
# A translation is two very different computations. The encoder reads the English sentence once, all positions in
# parallel. The decoder writes the Russian one token at a time, and every token is another pass through the
# decoder. Time both, on one long sentence, token by token:

# %%
from perevod import decoding

model, tokenizer = translator.model, translator.tokenizer
LONG = ("The European Central Bank raised interest rates by a quarter of a percentage "
        "point on Thursday, citing persistent inflation in the services sector.")
src = torch.tensor([decoding.source_ids(tokenizer, LONG, model.config.max_len)])
pad_id, bos, eos = decoding.special_ids(tokenizer)

with torch.inference_mode():
    start = time.perf_counter()
    memory = model.encode(src)
    encode_ms = (time.perf_counter() - start) * 1000
    written = torch.full((1, 1), bos)
    plain_steps = []
    for _ in range(100):
        start = time.perf_counter()
        next_id = model.decode(written, memory, src)[:, -1].argmax(dim=-1)
        plain_steps.append((time.perf_counter() - start) * 1000)
        written = torch.cat([written, next_id[:, None]], dim=1)
print(f"encoder: {encode_ms:.1f} ms for {src.shape[1]} tokens")
print(f"decoder step 1: {plain_steps[1]:.1f} ms, step 100: {plain_steps[99]:.1f} ms")

# %% [markdown]
# The decoder is called with everything written so far, so every step recomputes all the earlier positions to
# get the scores for the next one. Step $t$ processes $t$ positions, and a translation of $T$ tokens processes
#
# $$1 + 2 + \dots + T = \frac{T(T+1)}{2}$$
#
# positions in all: twice as long a sentence costs four times as much. Lesson 5 met this and fixed it for greedy
# decoding with a **KV cache**. perevod 0.1.0 ships the uncached code, because lesson 7 moved the simplest decoding
# into the package. The package's default, beam search, never had a cache at all.
#
# ## 3. The KV cache, in the package
#
# A refresher from lesson 5, in one sentence: at step $t$ the new token's query needs the keys and values of all
# earlier positions, and those never change once computed, so we keep them per layer and compute only the new
# token's. The cross-attention's keys and values come from the source alone and are computed once per sentence.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/kv-step-light.png" width="760" alt="Decoding step 5 without and with a cache. Without one, the decoder runs on all five tokens and recomputes the keys and values of the first four. With one, it runs on the fifth token only, appends its key and value to the cache, and attends to all five">
#
# Beam search adds one complication lesson 5 didn't have. After every step, the beams are re-chosen: a hypothesis
# can produce two survivors, and another none. Each surviving beam continues the history of its parent, so its row
# in the cache must be its parent's row. Before the next step, the cache is reordered by the parents:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/beam-reorder-light.png" width="760" alt="Beam search with a cache, beam size three. After a step, the three best continuations come from rows 0, 0 and 2 of the cache: row 1's hypothesis died and row 0's had two children. The cache is gathered with the index [0, 0, 2] so that each new beam's row holds its parent's keys and values">
#
# The new `decoding.py` of perevod. The helpers at the top are lesson 7's:

# %%
# %%writefile perevod/src/perevod/decoding.py
# Greedy decoding and beam search of lesson 3, batched as in lesson 4, with lesson 5's KV
# cache: every step feeds the decoder only the newest token. A source longer than the
# model's positions is cut to fit, instead of failing inside the model.

import torch
import torch.nn.functional as F

from .model import apply_rope


def special_ids(tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    return pad_id, tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")


def source_ids(tokenizer, sentence, limit):
    return tokenizer.encode(sentence).ids[:limit]


def pad(sequences, pad_id, device):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), pad_id, device=device)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids, device=device)
    return batch


def batches(sources, batch_size):
    # sentences in batches of similar length, so that little of a batch is padding
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def length_penalty(length, alpha=0.6):
    # Wu et al. (2016): ((5 + |y|) / 6) ** alpha
    return ((5 + length) / 6) ** alpha

# %% [markdown]
# `attend` is one attention sublayer for the newest token. It works on the model's own `Attention` modules, so the
# weights are the model's; only the order of operations is new. For self-attention it rotates the new query and
# key with RoPE and appends the key and value to the cache; for cross-attention the cache is fixed:

# %%
# %%writefile -a perevod/src/perevod/decoding.py


def attend(attention, x, cache, mask, angles):
    # one attention sublayer for the newest token; the cache holds everything before it
    q = attention.split_heads(attention.w_q(x))
    if angles is not None:  # RoPE: self-attention rotates queries and keys by position
        q = apply_rope(q, angles)
    if "fixed" not in cache:  # self-attention: add the new token's key and value
        k = attention.split_heads(attention.w_k(x))
        v = attention.split_heads(attention.w_v(x))
        if angles is not None:
            k = apply_rope(k, angles)
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
    out = F.scaled_dot_product_attention(q, cache["k"], cache["v"], attn_mask=mask)
    batch, _, length, _ = out.shape
    return attention.w_o(out.transpose(1, 2).reshape(batch, length, -1))

# %% [markdown]
# `decode_step` runs the decoder of `model.py` for one token, in the same order: pre-norm, self-attention,
# cross-attention, feed-forward. It repeats the model's arithmetic outside the model, which is a debt: if
# `model.py` changes, this has to change with it. The golden test of lesson 7 is what collects that debt.

# %%
# %%writefile -a perevod/src/perevod/decoding.py


def new_caches(model, memory):
    # per decoder layer: an empty self-attention cache, and the cross-attention's keys and
    # values, which depend only on the source and are computed once
    caches = []
    for layer in model.decoder:
        attention = layer.cross_attention
        k = attention.split_heads(attention.w_k(memory))
        v = attention.split_heads(attention.w_v(memory))
        caches.append({"self": {}, "cross": {"fixed": True, "k": k, "v": v}})
    return caches


def decode_step(model, token, memory_mask, caches, position):
    # the decoder of model.py, in the same order, for one new token; dropout is off anyway
    x = model.embed(token)
    angles = model.angles[position : position + 1]
    for layer, cache in zip(model.decoder, caches, strict=True):
        x = x + attend(layer.self_attention, layer.norm1(x), cache["self"], None, angles)
        x = x + attend(layer.cross_attention, layer.norm2(x), cache["cross"], memory_mask,
                       None)
        x = x + layer.feed_forward(layer.norm3(x))
    return (model.decoder_norm(x) @ model.embedding.weight.T)[:, -1]

# %%
# %%writefile -a perevod/src/perevod/decoding.py


def reorder(caches, rows):
    # the cached keys and values follow their hypotheses: row r of the new caches is row
    # rows[r] of the old ones
    for cache in caches:
        for part in cache.values():
            for key in ("k", "v"):
                if key in part:
                    part[key] = part[key][rows]

# %% [markdown]
# Greedy decoding with the cache is lesson 5's `greedy_cached`, batched:

# %%
# %%writefile -a perevod/src/perevod/decoding.py


@torch.inference_mode()
def greedy(model, tokenizer, sentences, max_tokens=128, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = new_caches(model, model.encode(src))
        token = torch.full((len(indices), 1), bos, device=device)
        written = []
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for position in range(max_tokens):
            scores = decode_step(model, token, memory_mask, caches, position)
            next_ids = scores.argmax(dim=-1)
            next_ids[done] = pad_id  # a finished sentence only collects padding
            written.append(next_ids)
            done |= next_ids == eos
            token = next_ids[:, None]
            if done.all():
                break
        written = torch.stack(written, dim=1)
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row].tolist():
                if token_id in (eos, pad_id):
                    break
                ids.append(token_id)
            translations[i] = tokenizer.decode(ids)
    return translations

# %% [markdown]
# Beam search is lesson 3's, step for step, with two changes: a hypothesis remembers its row in the cache, and the
# cache is reordered before each step.

# %%
# %%writefile -a perevod/src/perevod/decoding.py


@torch.inference_mode()
def beam_search(model, tokenizer, sentences, beam_size=4, alpha=0.6, max_tokens=128,
                batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = new_caches(model, model.encode(src))
        # per sentence: (tokens so far, total log-probability, its row in the caches)
        beams = []
        for k in range(len(indices)):
            beams.append([([bos], 0.0, k)])
        finished = [[] for _ in indices]
        active = [True for _ in indices]

# %%
# %%writefile -a perevod/src/perevod/decoding.py

        for position in range(max_tokens):
            rows, owners, last = [], [], []
            for k in range(len(indices)):
                if active[k]:
                    for tokens, _, row in beams[k]:
                        rows.append(row)
                        owners.append(k)
                        last.append(tokens[-1])
            if not rows:
                break
            reorder(caches, torch.tensor(rows, device=device))
            token = torch.tensor(last, device=device)[:, None]
            owner = torch.tensor(owners, device=device)
            scores = decode_step(model, token, memory_mask[owner], caches, position)
            top = scores.float().log_softmax(dim=-1).topk(beam_size)
            values, ids = top.values.tolist(), top.indices.tolist()

            row = 0
            for k in range(len(indices)):
                if not active[k]:
                    continue
                candidates = []
                for tokens, total, _ in beams[k]:
                    for log_p, token_id in zip(values[row], ids[row], strict=True):
                        candidates.append((tokens + [token_id], total + log_p, row))
                    row += 1
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams[k] = []
                for tokens, total, parent in candidates:
                    if tokens[-1] == eos:
                        finished[k].append((tokens, total))
                    else:
                        beams[k].append((tokens, total, parent))
                    if len(beams[k]) == beam_size:
                        break
                if len(finished[k]) >= beam_size:
                    active[k] = False

# %%
# %%writefile -a perevod/src/perevod/decoding.py

        for k, i in enumerate(indices):

            def normalized(candidate):
                tokens, total = candidate[0], candidate[1]
                return total / length_penalty(len(tokens) - 1, alpha)  # without <bos>

            best = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best[0][1:])  # decode drops <eos>
    return translations

# %% [markdown]
# The file now holds the new code, and the installed package still has the old one. That's useful for a moment:
# the old module is already imported as `decoding`, and the new file can be loaded next to it under another name.

# %%
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = decoding  # perevod 0.1.0, as installed
new = load_module("perevod.decoding_new", "perevod/src/perevod/decoding.py")
print("old:", old.__file__)
print("new:", new.__file__)

# %% [markdown]
# The cache must not change a single translation. On 500 sentences of the test set and 30 of the news test set of
# lesson 7, with beam search, the old code against the new:

# %%
from sacrebleu.utils import get_reference_files, get_source_file

tatoeba = test.sample(500, random_state=SEED)
news_file = Path(get_source_file("wmt20", "en-ru"))
news = random.Random(SEED).sample(news_file.read_text(encoding="utf-8").splitlines(), 30)

rows = []
for name, sentences in [("Tatoeba", list(tatoeba.en)), ("WMT20 news", news)]:
    start = time.perf_counter()
    before = old.beam_search(model, tokenizer, sentences)
    middle = time.perf_counter()
    after = new.beam_search(model, tokenizer, sentences)
    end = time.perf_counter()
    assert after == before, name
    rows.append({"test set": name, "sentences": len(sentences),
                 "without the cache, s": middle - start, "with it, s": end - middle,
                 "faster": (middle - start) / (end - middle)})
speed = pd.DataFrame(rows)
display(table(speed))

# %% [markdown]
# The same translations, word for word, and faster, by more on the news. That's the $T^2$ at work: Tatoeba's
# sentences are short, and a few steps of recomputation are cheap; a news sentence runs for dozens of steps, and the
# recomputation dominates. The step times of section 2 show it directly:

# %%
with torch.inference_mode():
    memory_mask = (src != pad_id)[:, None, None, :]
    caches = new.new_caches(model, model.encode(src))
    token = torch.full((1, 1), bos)
    cached_steps = []
    for position in range(100):
        start = time.perf_counter()
        scores = new.decode_step(model, token, memory_mask, caches, position)
        next_id = scores.argmax(dim=-1)
        cached_steps.append((time.perf_counter() - start) * 1000)
        token = next_id[:, None]

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(1, 101), plain_steps, color=COLORS["dec"], label="recompute everything")
ax.plot(range(1, 101), cached_steps, color=COLORS["enc"], label="with the KV cache")
ax.set_xlabel("decoding step")
ax.set_ylabel("ms per step")
ax.set_title("One decoding step, on a CPU")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# Without the cache a step gets slower as the translation grows; with it, every step costs about the same.
#
# ## 4. int8 quantization, measured
#
# The next speed-up everyone reaches for is **quantization**: store weights as 8-bit integers instead of 32-bit
# floats, and multiply in integers. PyTorch's dynamic quantization does it in one call: the weights of every
# `nn.Linear` become int8, and the activations are quantized on the fly, at every call
# ([PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)).

# %%
from torch.ao.quantization import quantize_dynamic

engines = torch.backends.quantized.supported_engines
if torch.backends.quantized.engine == "none" and "qnnpack" in engines:  # ARM, e.g. a Mac
    torch.backends.quantized.engine = "qnnpack"
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
sentences = list(tatoeba.en)
start = time.perf_counter()
fp32 = new.beam_search(model, tokenizer, sentences)
fp32_seconds = time.perf_counter() - start
start = time.perf_counter()
int8 = new.beam_search(quantized, tokenizer, sentences)
int8_seconds = time.perf_counter() - start
changed = sum(a != b for a, b in zip(fp32, int8))
fp32_chrf = sacrebleu.corpus_chrf(fp32, [list(tatoeba.ru)]).score
int8_chrf = sacrebleu.corpus_chrf(int8, [list(tatoeba.ru)]).score
print(f"float32: {fp32_seconds:.1f} s, chrF {fp32_chrf:.1f}")
print(f"int8:    {int8_seconds:.1f} s, chrF {int8_chrf:.1f}, {changed} of {len(sentences)} "
      "translations changed")

# %% [markdown]
# Not faster, and worse. The speed-up that int8 promises comes from large matrix multiplications, where moving
# 8-bit numbers instead of 32-bit ones saves memory traffic. Count where the work of one decoding step goes in our
# model, layer by layer:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/step-flops-light.png" width="760" alt="Multiply-adds of one decoding step of the published model: the output projection onto 16,000 tokens is the largest single part, larger than all attention projections together; dynamic quantization converts only the nn.Linear layers and leaves the output projection, which reuses the embedding matrix, in float32">
#
# The largest single matrix is the output projection onto the 16,000 tokens of the vocabulary, and it isn't an
# `nn.Linear`: the model reuses the embedding matrix there, so quantization never touches it. What it does touch
# are 256-wide layers, too small for int8 arithmetic to beat the cost of quantizing the activations at every call.
# For a model this size, on this CPU, the trade is all cost. perevod 0.2.0 doesn't ship it; a measurement is a fine
# reason to drop a feature. (PyTorch is moving quantization to its `torchao` library, and the old API is on its way
# out; the arithmetic above won't change with the API.)
#
# ## 5. Batches and threads
#
# A CPU does more work per second when it multiplies bigger matrices: one call on a batch of sixteen sentences costs
# much less than sixteen calls. Here is perevod's throughput with batches of different sizes, on 256 sentences:

# %%
sentences = list(test.sample(256, random_state=SEED + 1).en)
rows = []
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    start = time.perf_counter()
    new.beam_search(model, tokenizer, sentences, batch_size=batch_size)
    seconds = time.perf_counter() - start
    rows.append({"batch": batch_size, "sentences per second": len(sentences) / seconds,
                 "ms per batch": 1000 * seconds / math.ceil(len(sentences) / batch_size)})
by_batch = pd.DataFrame(rows)
display(table(by_batch, digits=1))

# %% [markdown]
# The throughput grows several times over, and the time per batch grows much slower than the batch. That's the
# fact the second half of this lesson is built on: one translation of sixteen sentences costs little more than one
# translation of one.
#
# One more knob: threads. PyTorch splits a matrix multiplication between CPU threads, and more isn't always better:

# %%
threads_before = torch.get_num_threads()
rows = []
for threads in [1, 2, 4]:
    torch.set_num_threads(threads)
    start = time.perf_counter()
    new.beam_search(model, tokenizer, sentences[:128], batch_size=16)
    rows.append({"threads": threads, "seconds": time.perf_counter() - start})
torch.set_num_threads(threads_before)
display(table(pd.DataFrame(rows)))

# %% [markdown]
# On Kaggle the free CPU session shows four cores to Python but has two physical ones, each running two hardware
# threads; two threads that share a core mostly wait for each other. PyTorch picks its default from what it can
# see, so check it on the machine you deploy to.
#
# ## 6. A FastAPI service
#
# A command line loads the model for every call, and lesson 7 measured that the loading costs more than translating
# a sentence. A **service** loads it once and answers requests over HTTP for as long as it runs.
# [FastAPI](https://fastapi.tiangolo.com) is the common choice in Python: you describe the request and the response
# as typed classes, and it validates the input, writes the errors, and generates documentation.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/service-light.png" width="760" alt="How a request travels through perevod's service: an HTTP request reaches uvicorn and FastAPI, which validates it and puts it into a queue; a batcher takes the requests that arrived within a few milliseconds, translates them as one batch in a worker thread, and answers each request with its own translation">
#
# The request and the response are pydantic models; a text longer than 5,000 characters is refused before it
# reaches the model:

# %%
# %%writefile perevod/src/perevod/server.py
import argparse
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .translator import MODEL, Translator


class TranslateRequest(BaseModel):
    text: str = Field(max_length=5000, examples=["I think, therefore I am."])


class TranslateResponse(BaseModel):
    translation: str

# %% [markdown]
# Between the HTTP handlers and the model sits a queue. Every request puts its text and a future into it and waits
# for the future. A **batcher** takes requests out of the queue: the first one, then whatever else arrives within
# `max_wait` seconds, up to `max_batch` requests. With `max_batch=1` it is a plain queue, one request at a time.

# %%
# %%writefile -a perevod/src/perevod/server.py


class Batcher:
    """Translates requests that arrive close together as one batch.

    A batch closes when it holds max_batch requests or max_wait seconds after its first one.
    """

    def __init__(self, translator, max_batch=16, max_wait=0.005):
        self.translator = translator
        self.max_batch, self.max_wait = max_batch, max_wait
        self.queue = asyncio.Queue()
        self.batch_sizes = []

    async def translate(self, text):
        answer = asyncio.get_running_loop().create_future()
        await self.queue.put((text, answer))
        return await answer

# %%
# %%writefile -a perevod/src/perevod/server.py

    async def collect(self):
        batch = [await self.queue.get()]
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch

# %% [markdown]
# The batcher translates in a worker thread, so that the server keeps accepting requests while the model works,
# and gives every request its own translation back. If the translation fails, every request of the batch gets the
# error instead of waiting forever.

# %%
# %%writefile -a perevod/src/perevod/server.py

    async def run(self):
        while True:
            batch = await self.collect()
            self.batch_sizes.append(len(batch))
            texts = [text for text, _ in batch]
            # in a thread, so that the server keeps accepting requests meanwhile
            job = asyncio.ensure_future(asyncio.to_thread(self.translator.translate, texts))
            await asyncio.wait([job])
            if job.exception() is not None:  # every request of the batch gets the error
                for _, answer in batch:
                    answer.set_exception(job.exception())
                continue
            for (_, answer), translation in zip(batch, job.result(), strict=True):
                answer.set_result(translation)

# %% [markdown]
# The app itself. The `lifespan` function runs when the server starts and stops: it loads the model once and
# starts the batcher. The two handlers are three lines each. The `demo` switch mounts a Gradio page, section 10's.

# %%
# %%writefile -a perevod/src/perevod/server.py


def create_app(model=MODEL, revision=None, max_batch=16, max_wait=0.005, demo=False):
    @asynccontextmanager
    async def lifespan(app):
        translator = Translator.from_pretrained(model, revision)
        app.state.batcher = Batcher(translator, max_batch, max_wait)
        worker = asyncio.create_task(app.state.batcher.run())
        yield
        worker.cancel()

    app = FastAPI(title="perevod", version=__version__, lifespan=lifespan)

    @app.post("/translate")
    async def translate(request: TranslateRequest) -> TranslateResponse:
        translation = await app.state.batcher.translate(request.text)
        return TranslateResponse(translation=translation)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    if demo:
        import gradio as gr

        from .demo import build_demo

        app = gr.mount_gradio_app(app, build_demo(app), path="/")
    return app

# %% [markdown]
# And a command, `perevod-server`, with the batcher's settings as options:

# %%
# %%writefile -a perevod/src/perevod/server.py


def main(argv=None):
    parser = argparse.ArgumentParser(prog="perevod-server",
                                     description="Serve perevod over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=16, help="1 turns batching off")
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    parser.add_argument("--demo", action="store_true", help="serve a Gradio page at /")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(max_batch=args.max_batch, max_wait=args.max_wait_ms / 1000,
                     demo=args.demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

# %% [markdown]
# The package grows a new optional dependency, the server's: `pip install "perevod[serve]"` installs FastAPI and
# uvicorn, and a plain `pip install perevod` stays as light as it was. The version becomes 0.2.0, and the server
# gets a command of its own:

# %%
write("perevod/pyproject.toml", """
[project]
name = "perevod"
version = "0.2.0"
description = "English-to-Russian translation with a small Transformer built from scratch"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "tokenizers>=0.15",
    "huggingface_hub>=0.20",
    "safetensors>=0.4",
]

[project.optional-dependencies]
serve = ["fastapi>=0.110", "uvicorn>=0.29"]
demo = ["fastapi>=0.110", "uvicorn>=0.29", "gradio>=4.40"]
test = ["pytest>=8", "ruff>=0.6", "sacrebleu>=2.4", "pandas>=2.0", "pyarrow>=14",
        "fastapi>=0.110", "httpx>=0.27"]

[project.scripts]
perevod = "perevod.cli:main"
perevod-server = "perevod.server:main"

[project.urls]
Source = "https://github.com/xmarva/transformers-from-scratch/tree/main/perevod"
Model = "https://huggingface.co/xmarva/perevod-en-ru"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["slow: downloads the published model and translates real sentences"]

[tool.ruff]
line-length = 92  # what a notebook cell shows: lesson 7 writes these files in its cells
""")

# %%
# %%writefile perevod/src/perevod/__init__.py
__version__ = "0.2.0"


def __getattr__(name):
    # `import perevod` stays light, so that --help and --version answer at once;
    # PyTorch is loaded only when someone asks for the translator
    if name == "Translator":
        from .translator import Translator

        return Translator
    raise AttributeError(f"module 'perevod' has no attribute {name!r}")

# %% [markdown]
# From here on we change the package as we go, so it's installed in **editable** mode: pip puts a pointer to
# `perevod/src` into site-packages instead of a copy, and every process that imports perevod gets the current
# files. That's the right install for working on a package; lesson 7's CI job that installs the built wheel is
# still what checks the copy users get.

# %%
# !{sys.executable} -m pip install -q -e "./perevod[serve]"

# %% [markdown]
# A service runs in its own process. Start one, wait until `/health` answers, and send it a sentence, first with
# batching turned off:

# %%
def start_server(port, max_batch, max_wait_ms=5, demo=False):
    command = [sys.executable, "-m", "perevod.server", "--port", str(port),
               "--max-batch", str(max_batch), "--max-wait-ms", str(max_wait_ms)]
    if demo:
        command.append("--demo")
    quiet = subprocess.DEVNULL
    process = subprocess.Popen(command, stdout=quiet, stderr=quiet)
    for _ in range(180):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                return process
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("the server did not start")


server = start_server(8001, max_batch=1)
URL = "http://127.0.0.1:8001/translate"
answer = httpx.post(URL, json={"text": "Nobody understands me."})
print(answer.status_code, answer.json())
refused = httpx.post(URL, json={"text": "word " * 2000})
print(refused.status_code, refused.json()["detail"][0]["msg"])

# %% [markdown]
# A translation, and a 422 with a readable reason for the text that is too long, which our code never saw:
# FastAPI refused it from the declaration of `TranslateRequest`. The same declarations describe the API at
# `/docs`, an interactive page anyone can try requests on, and at `/openapi.json`, a machine-readable description
# from which clients in other languages can be generated:

# %%
spec = httpx.get("http://127.0.0.1:8001/openapi.json").json()
for path, operations in spec["paths"].items():
    for method, operation in operations.items():
        print(method.upper(), path, "-", operation["summary"])

# %% [markdown]
# ## 7. Load testing without fooling yourself
#
# Now the question this lesson is about: how fast does the service answer when many people use it? The obvious
# way to find out is a loop that sends a request, waits for the answer, records the time, and sends the next one,
# with a few such loops in parallel. Most load testing tools work that way, and Gil Tene named what's wrong with it:
# **coordinated omission** ([Tene, 2013](https://groups.google.com/g/mechanical-sympathy/c/icNZJejUHfE/m/BfDekfBEs_sJ)).
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/omission-light.png" width="760" alt="A server stalls for one second. Users arriving at their own pace keep sending during the stall, and a quarter of their requests are slow. A closed-loop tester waits for its answer and sends nothing during the stall, so it records one slow request in thirty-two and reports a service that is almost never slow">
#
# When the server stalls, a tester that waits for each answer stops sending, so it records one slow request where
# real users, who don't coordinate with the server, would have produced dozens. The fix is an **open-loop** test:
# decide in advance when every request is sent, independently of the answers, and measure each latency from the
# moment the request was *supposed* to go. Users arriving independently at an average rate $\lambda$ are well
# described by a **Poisson process**, whose gaps between arrivals are exponentially distributed with mean
# $1/\lambda$:

# %%
async def load_test(url, rate, seconds, sentences, seed=0):
    rng = random.Random(seed)
    send_at = []
    moment = rng.expovariate(rate)
    while moment < seconds:
        send_at.append(moment)
        moment += rng.expovariate(rate)  # exponential gaps: a Poisson process
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        start = time.perf_counter()

        async def one(k, at):
            await asyncio.sleep(max(0.0, start + at - time.perf_counter()))
            response = await client.post(url, json={"text": sentences[k % len(sentences)]})
            response.raise_for_status()
            return time.perf_counter() - (start + at)  # from when it was due, not sent

        tasks = []
        for k, at in enumerate(send_at):
            tasks.append(one(k, at))
        return np.array(await asyncio.gather(*tasks))

# %% [markdown]
# Each request is a coroutine that sleeps until its moment, then sends. If the server is slow, the coroutines don't
# wait for each other, just like users. First, the **service time**: how long the server takes per request when
# nobody else is asking. One request every two seconds keeps it idle between them:

# %%
traffic = list(test.sample(2_000, random_state=SEED + 2).en)
idle = await load_test(URL, rate=0.5, seconds=40, sentences=traffic, seed=1)
service = idle.mean()
mu = 1 / service
print(f"service time over {len(idle)} requests: mean {service * 1000:.0f} ms, "
      f"median {np.median(idle) * 1000:.0f} ms, max {idle.max() * 1000:.0f} ms")
print(f"one request at a time, at most {mu:.1f} requests per second")

# %% [markdown]
# A service that answers one request at a time has a capacity of $\mu = 1/S$ requests per second, where $S$ is the
# mean service time. The load test runs at fractions of it, from a fifth to almost all, fifteen seconds each:

# %%
LOADS = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
one_at_a_time = {}
for load in LOADS:
    rate = load * mu
    one_at_a_time[rate] = await load_test(URL, rate, 15, traffic, seed=int(100 * load))
    p50, p99 = np.percentile(one_at_a_time[rate], [50, 99]) * 1000
    print(f"{rate:5.1f} requests/s, {load:.0%} of capacity: "
          f"p50 {p50:6.0f} ms, p99 {p99:6.0f} ms")

# %% [markdown]
# At a fifth of its capacity the server answers in about its service time. At nine tenths, the median is several
# times that and the 99th percentile is much worse, though the model works exactly as fast as before. The time went
# into the queue.
#
# Queueing theory predicts this curve, and the prediction is worth having before you buy servers. The simplest
# model is the **M/M/1 queue**: Poisson arrivals at rate $\lambda$, one server, exponentially distributed service
# times with mean $S = 1/\mu$. The **utilization** $\rho = \lambda / \mu$ is the fraction of time the server is
# busy; it has to stay below 1, or the queue grows without end. The time a request spends in the system, waiting
# plus being served, is then exponentially distributed with rate $\mu - \lambda$, so its mean and its quantiles
# are
#
# $$\mathbb{E}[T] = \frac{1}{\mu - \lambda} = \frac{S}{1 - \rho}, \qquad
# T_q = \frac{-\ln(1 - q)}{\mu - \lambda} = \frac{-\ln(1 - q)\, S}{1 - \rho}.$$
#
# The factor $1/(1 - \rho)$ is the whole story: at $\rho = 0.5$ a request takes twice its service time on average,
# at $0.9$ ten times, at $0.99$ a hundred. The last 10% of capacity costs more latency than the first 90%.
#
# Our service times aren't exponential: a sentence takes about as long as its length, and the lengths are spread
# less than an exponential distribution would spread them. Kingman's approximation keeps the factor and corrects
# for the spread ([Kingman, 1961](https://doi.org/10.1017/S0305004100036094)):
#
# $$\mathbb{E}[T] \approx S + \frac{\rho}{1 - \rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot S,$$
#
# where $c_a$ and $c_s$ are the coefficients of variation (standard deviation over mean) of the gaps between
# arrivals and of the service times. Poisson arrivals have $c_a = 1$; exponential service has $c_s = 1$, and then
# the formula is exactly M/M/1's. Ours, measured at low load:

# %%
c_s = idle.std() / idle.mean()
print(f"coefficient of variation of the service time: {c_s:.2f}")
rows = []
for rate, times in one_at_a_time.items():
    rho = rate / mu
    kingman = service + rho / (1 - rho) * (1 + c_s**2) / 2 * service
    rows.append({"utilization": rho, "measured mean, ms": 1000 * times.mean(),
                 "M/M/1, ms": 1000 * service / (1 - rho), "Kingman, ms": 1000 * kingman})
theory = pd.DataFrame(rows)
display(table(theory, digits=0))

# %%
rhos = np.linspace(0.05, 0.97, 100)
fig, ax = plt.subplots(figsize=(8, 3.8))
measured = [1000 * times.mean() for times in one_at_a_time.values()]
ax.plot(rhos, 1000 * (service + rhos / (1 - rhos) * (1 + c_s**2) / 2 * service),
        color=COLORS["att"], label="Kingman's approximation")
ax.plot(rhos, 1000 * service / (1 - rhos), color=COLORS["muted"], linestyle=(0, (3, 3)),
        label="M/M/1")
ax.plot(theory.utilization, measured, "o", color=COLORS["dec"], label="measured mean")
ax.set_xlabel("utilization ρ = λ / μ")
ax.set_ylabel("mean latency, ms")
ax.set_ylim(0, 1.15 * max(measured))
ax.set_title("Latency against load, one request at a time")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# @@QUEUE_RESULT@@
#
# ## 8. Dynamic batching
#
# Section 5 showed that one batch of sixteen costs little more than one sentence. A queue full of waiting requests
# is a batch nobody formed. The batcher of section 6 forms it: it takes the first request, waits up to a few
# milliseconds for more, and translates whatever it has together.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/batching-light.png" width="760" alt="Requests arriving over time, one at a time and with dynamic batching. One at a time, each request waits for all those before it. With dynamic batching, the requests that arrive while a batch is being translated go into the next batch together, so the queue empties in a few batches instead of many single translations">
#
# The cost is a wait of at most `max_wait` for the first request of a batch when the server is idle; the gain is
# that under load the server translates many requests at the price of a few. The same server with batches of up
# to sixteen, at the same loads and at loads it couldn't take before:

# %%
server.terminate()
server.wait()
server = start_server(8002, max_batch=16, max_wait_ms=5)
URL = "http://127.0.0.1:8002/translate"
batched = {}
for rate in list(one_at_a_time) + [1.5 * mu, 2 * mu, 3 * mu]:
    batched[rate] = await load_test(URL, rate, 15, traffic, seed=int(10 * rate))
    p50, p99 = np.percentile(batched[rate], [50, 99]) * 1000
    print(f"{rate:5.1f} requests/s: p50 {p50:6.0f} ms, p99 {p99:6.0f} ms")

# %%
fig, ax = plt.subplots(figsize=(8, 3.8))
for runs, colour, name in [(one_at_a_time, COLORS["dec"], "one at a time"),
                           (batched, COLORS["enc"], "dynamic batching, up to 16")]:
    rates = list(runs)
    p50 = [1000 * np.percentile(runs[rate], 50) for rate in rates]
    p99 = [1000 * np.percentile(runs[rate], 99) for rate in rates]
    ax.plot(rates, p50, "o-", color=colour, label=f"{name}: median")
    ax.plot(rates, p99, "o--", color=colour, alpha=0.6, label=f"{name}: 99th percentile")
ax.axvline(mu, color=COLORS["muted"], linewidth=1, linestyle=(0, (3, 3)))
ax.text(mu, ax.get_ylim()[1] * 0.92, " capacity one at a time", color=COLORS["muted"],
        fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("requests per second")
ax.set_ylabel("latency, ms (log scale)")
ax.set_title("The same server, without and with dynamic batching")
ax.legend(frameon=False, fontsize=8)
plt.show()

# %% [markdown]
# @@BATCH_RESULT@@
#
# Large language model servers take the same idea further. A request to an LLM generates hundreds of tokens, and
# waiting for a whole batch to finish would hold short answers behind long ones, so vLLM and its predecessors
# batch at the level of single decoding steps: a finished sequence leaves the batch after its last token and a
# new one joins at the next step ([Yu et al., 2022](https://www.usenix.org/conference/osdi22/presentation/yu);
# [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)). Our sentences are short enough that batching whole
# requests is the right trade.
#
# ## 9. A Gradio page
#
# An API is for programs. For people, and for the next lesson's demo, a page with a text box helps more. Gradio,
# part of Hugging Face since 2021 ([Abid, 2021](https://huggingface.co/blog/gradio-joins-hf)), builds one from a
# few lines, and it can live inside the same FastAPI app, so the page and the API share one model and one queue:

# %%
# %%writefile perevod/src/perevod/demo.py
import gradio as gr

EXAMPLES = ["I think, therefore I am.", "Do you dream in color?", "Nobody understands me.",
            "Memory is the diary we all carry about with us."]
ABOUT = """# perevod
English to Russian with a small Transformer built from scratch in the course
[Transformers: from scratch to production][course]. It was trained on short everyday
sentences; its [model card][card] says where it fails.

[course]: https://github.com/xmarva/transformers-from-scratch
[card]: https://huggingface.co/xmarva/perevod-en-ru"""


def build_demo(app):
    async def translate(text):
        return await app.state.batcher.translate(text)  # through the same queue as the API

    with gr.Blocks(title="perevod") as demo:
        gr.Markdown(ABOUT)
        source = gr.Textbox(label="English", lines=3)
        target = gr.Textbox(label="Russian", lines=3)
        button = gr.Button("Translate", variant="primary")
        button.click(translate, source, target)
        gr.Examples(EXAMPLES, source)
    return demo

# %% [markdown]
# The page translates through `app.state.batcher`, so a person typing and a program calling the API wait in the
# same queue and fill the same batches. Gradio is a heavy dependency, so it is an extra of its own, `demo`, and the
# page appears only with `--demo`. Here is what it looks like:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/08/gradio.png" width="700" alt="The Gradio page of perevod: a short description with links to the course and the model card, an English text box, a Russian one, a Translate button and four example sentences">

# %%
server.terminate()
server.wait()
# !{sys.executable} -m pip install -q -e "./perevod[demo]"
server = start_server(8003, max_batch=16, demo=True)
page = httpx.get("http://127.0.0.1:8003/")
answer = httpx.post("http://127.0.0.1:8003/translate", json={"text": "Do you dream?"})
print("the page:", page.status_code, "| the API:", answer.json())
server.terminate()
server.wait()

# %% [markdown]
# ## 10. Tests and the release of 0.2.0
#
# The server gets tests of its own, on lesson 7's tiny model. pytest runs the app in-process: FastAPI's
# `TestClient` calls it without a network, and runs its lifespan, so the batcher is real. The last test is the
# one that matters most: requests that end up in one batch must each get their own translation back, the bug
# lesson 7's mutation testing taught us to fear.

# %%
# %%writefile perevod/tests/test_server.py
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from perevod.server import create_app


@pytest.fixture
def client(tiny_dir):
    app = create_app(str(tiny_dir), max_wait=0.05)
    with TestClient(app) as client:  # entering it runs the lifespan: model and batcher
        yield client


def test_the_health_check(client):
    assert client.get("/health").json()["status"] == "ok"


def test_the_api_translates_like_the_package(client, translator, pairs):
    english = next(iter(pairs))
    response = client.post("/translate", json={"text": english})
    assert response.json() == {"translation": translator.translate(english)}


def test_a_text_too_long_is_refused(client):
    assert client.post("/translate", json={"text": "tom " * 2000}).status_code == 422


def test_requests_in_one_batch_get_their_own_translations(tiny_dir, pairs):
    app = create_app(str(tiny_dir), max_wait=0.2)

    async def ask_all():
        async with app.router.lifespan_context(app):  # the model and the batcher
            transport = httpx.ASGITransport(app=app)
            base_url = "http://perevod"
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                requests = []
                for english in pairs:
                    requests.append(client.post("/translate", json={"text": english}))
                return await asyncio.gather(*requests)

    responses = asyncio.run(ask_all())
    assert app.state.batcher.batch_sizes == [len(pairs)]  # one batch for all of them
    for response, russian in zip(responses, pairs.values(), strict=True):
        assert response.json()["translation"] == russian

# %%
# !{sys.executable} -m pip install -q -e "./perevod[test,serve]"
# !cd perevod && {sys.executable} -m pytest -q -p no:cacheprovider --color=no

# %% [markdown]
# All of them, the slow ones included. The golden test of lesson 7 is the statement this release rests on: the
# cached beam search, the new decoding step and the reordered caches produce, on the twelve golden sentences and
# on the chrF sample, exactly what 0.1.0 produced. That makes 0.2.0 a **minor** release in lesson 7's terms: new
# features (the server, the page, the speed), nothing that changes what users get.
#
# The release is a commit and a tag, `perevod-v0.2.0`, in the course repository; lesson 9 builds its Docker image
# from that tag:
#
# ```bash
# pip install "perevod[serve] @ git+https://github.com/xmarva/transformers-from-scratch@perevod-v0.2.0#subdirectory=perevod"
# perevod-server --port 8000 --demo
# ```
#
# ## References
#
# - Linden, G. (2006). [Make Data Useful](https://glinden.blogspot.com/2006/12/slides-from-my-talk-at-stanford.html), a talk at Stanford. *The slide with Amazon's 100 ms and Google's 500 ms.*
# - Brutlag, J. (2009). [Speed Matters for Google Web Search](https://services.google.com/fh/files/blogs/google_delayexp.pdf). Google.
# - Dean, J., Barroso, L. A. (2013). [The Tail at Scale](https://doi.org/10.1145/2408776.2408794). Communications of the ACM 56(2).
# - Tene, G. (2013). [Coordinated Omission](https://groups.google.com/g/mechanical-sympathy/c/icNZJejUHfE/m/BfDekfBEs_sJ), on the mechanical-sympathy list; and the talk [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (2015).
# - Little, J. D. C. (1961). [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383). Operations Research 9(3).
# - Kingman, J. F. C. (1961). [The single server queue in heavy traffic](https://doi.org/10.1017/S0305004100036094). Proceedings of the Cambridge Philosophical Society 57(4).
# - Harchol-Balter, M. (2013). *Performance Modeling and Design of Computer Systems: Queueing Theory in Action*. Cambridge University Press. *The M/M/1 queue.*
# - Yu, G.-I. et al. (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI. *Iteration-level scheduling.*
# - Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP. *vLLM.*
# - [NVIDIA Triton: dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).
# - PyTorch: [dynamic quantization](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html), [CPU threading](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).
# - [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org), [Gradio](https://www.gradio.app/docs), [HTTPX](https://www.python-httpx.org).
# - Abid, A. (2021). [Gradio is joining Hugging Face!](https://huggingface.co/blog/gradio-joins-hf)
