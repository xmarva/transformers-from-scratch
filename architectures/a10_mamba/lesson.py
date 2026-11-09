# %% [markdown]
# # Lesson 10. Beyond attention with Mamba, state space models and hybrids
#
# Everything this course has built so far pays the same bill. Attention keeps every token it has read, in every
# layer, for as long as the conversation lasts (lesson 8), and compares every new token with all of them (lesson 9).
# The bill grows with the text, and all the tricks of the last three lessons only make it grow more slowly.
#
# A recurrent network doesn't pay it. It reads a token, updates a state of fixed size, and forgets the token; the
# millionth token costs what the first one did. The trouble is that recurrent networks lost to Transformers for good
# reasons: they train one token at a time, and their state forgets what matters as easily as what doesn't.
#
# In December 2023 Albert Gu and Tri Dao, the same Tri Dao who wrote FlashAttention, published *Mamba*, a recurrent
# model that fixes both. It trains in parallel, like a Transformer, and its state decides, token by token, what to
# keep and what to let go. The idea comes from an unexpected place, the *state space models* of control theory, the
# language of the Kalman filter that helped navigate Apollo to the Moon. This lesson takes Mamba apart on the real
# weights of mamba-130m: the equation it starts from, how a continuous equation becomes a layer, what the model
# chooses to remember in a real sentence, the scan that makes a recurrence parallel, and generation with a state that
# never grows. Then we look at what a fixed state costs, where Mamba ends up inside today's hybrid models, and train
# one on DNA.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a10/hero-light.png" width="760" alt="The same eight tokens read by attention and by Mamba. Attention keeps a key and a value for every token, so its cache grows by one slot per token and never shrinks. Mamba writes every token into one state of fixed size, 1,536 channels of 16 numbers per layer in mamba-130m, and the state is the same size after eight tokens or eight million">
#
# **What you'll do**
#
# 1. Write a state space model from its differential equation and turn it into a layer with exact and approximate
#    discretisation.
# 2. Make it selective, as Mamba does, and write the whole Mamba block from scratch; load mamba-130m into it and match
#    Hugging Face.
# 3. Watch what the trained model writes into its state and how long each of its channels remembers.
# 4. Write the scan twice, as a loop and as a parallel scan, and generate text with a state of fixed size, measured
#    against a Transformer's growing cache.

# %% tags=["include:setup"]

# %%
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

SMOKE = env.smoke()  # CI runs the lesson on a small slice; you run it in full
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. A state that doesn't grow
#
# A *state space model* describes a system by a vector of numbers, its state $$h(t) \in \mathbb{R}^N$$, that
# changes over time under an input $$x(t)$$ and produces an output $$y(t)$$:
#
# $$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t).$$
#
# The first equation says how the state moves: $$A$$ ($$N \times N$$) is how it would drift on its own, $$B$$
# ($$N \times 1$$) is how the input pushes it. The second says what we read out of it. Rudolf Kálmán made this form
# the language of control theory in 1960, and it has run in autopilots, phone receivers and spacecraft since.
#
# For a sequence model the useful case is a diagonal $$A$$ with negative entries $$a_1, \dots, a_N$$. The equations
# then split into $$N$$ independent ones, and each is a *leaky integrator*:
#
# $$h_n'(t) = a_n\,h_n(t) + b_n\,x(t).$$
#
# Picture a bucket with a hole. The input pours water in; the hole lets a fraction $$\lvert a_n \rvert$$ of whatever
# is inside leak out per unit of time. Without input, the level falls as $$e^{a_n t}$$: a bucket with a small hole,
# $$a_n$$ near zero, keeps its water for a long time; one with a big hole forgets in an instant. A state of $$N$$
# buckets with holes of different sizes keeps a record of the input at $$N$$ different time scales, recent and
# distant, and $$C$$ mixes the levels into an output.
#
# Here are four such buckets, fed one splash of input at $$t = 1$$ and a steady stream between $$t = 4$$ and $$6$$;
# the curves are the exact solution of each equation.

# %%
t = np.linspace(0, 10, 1001)
splash = (np.abs(t - 1) < 0.05) * 10.0  # a short burst whose total is 1
stream = ((t > 4) & (t < 6)) * 1.0
dt = t[1] - t[0]

fig, ax = plt.subplots(figsize=(7.4, 3.4))
colors = [COLORS["enc"], COLORS["att"], COLORS["data"], COLORS["dec"]]
for a, color in zip([-0.125, -0.25, -0.5, -1.0], colors):  # four holes, small to large
    level = np.zeros_like(t)
    for i in range(1, len(t)):  # exact step of h' = a h + x for constant x over dt
        x = splash[i - 1] + stream[i - 1]
        level[i] = level[i - 1] * math.exp(a * dt) + (math.exp(a * dt) - 1) / a * x
    ax.plot(t, level, color=color, linewidth=2, label=f"a = {a:g}")
ax.fill_between(t, 0, stream * 0.15, color=COLORS["line"], alpha=0.4, label="input")
ax.set_xlabel("time")
ax.set_ylabel("state")
ax.set_title("Four leaky buckets: the same input, four memories")
ax.legend(frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# The bucket with the smallest hole still remembers the splash when the stream arrives; the one with the largest has
# forgotten it within a unit of time and tracks only the present. That spread of memories is what a state space
# layer is made of.

# %% [markdown]
# ## 2. From an equation to a layer
#
# Text doesn't arrive in continuous time; it arrives a token at a time. To turn the equation into a layer, assume
# the input stays constant for a short time $$\Delta$$, the *step*, and solve the equation exactly over that time.
# This is the *zero-order hold* of signal processing, and for a diagonal $$A$$ it gives, per channel,
#
# $$h_t = \bar a\, h_{t-1} + \bar b\, x_t, \qquad \bar a = e^{\Delta a}, \qquad
# \bar b = \frac{e^{\Delta a} - 1}{a}\, b.$$
#
# $$\bar a$$ is the fraction of the state that survives one step: between 0 and 1, because $$a < 0$$. The state space
# model has become a linear recurrence, an RNN without a nonlinearity inside the loop. That missing nonlinearity is
# what will make it parallel in section 6.
#
# Mamba's code takes a shortcut for $$\bar b$$. For small $$\Delta a$$, $$e^{\Delta a} - 1 \approx \Delta a$$, so
# $$\bar b \approx \Delta\, b$$, the Euler step. Is that a big approximation? The ratio of the two is
# $$(e^{\Delta a} - 1)/(\Delta a)$$, which depends only on the product $$\Delta a$$:

# %%
for product in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    exact_over_euler = (1 - math.exp(-product)) / product
    print(f"|Δa| = {product:<5}  exact b̄ / Euler b̄ = {exact_over_euler:.3f}")

# %% [markdown]
# For a step that is small against the channel's time scale, the two agree; for a large one the Euler step
# overestimates the input's weight. We'll measure in section 5 which regime the trained model lives in.
#
# A model with fixed $$A$$, $$B$$, $$C$$ and $$\Delta$$ is *linear and time-invariant*: whatever the token, it is
# treated the same way. S4, the 2021 model by Gu, Goel and Ré that started this line of work, was of this kind, and
# it could exploit that: a time-invariant recurrence is a convolution with a fixed kernel, and long convolutions are
# fast with the FFT. S4 set records on the Long Range Arena, a benchmark of sequences up to 16k long, where the
# Transformers of its time did badly. On language it lagged behind. The reason is visible in the bucket picture: a
# fixed hole leaks the important word and the filler word at the same rate.

# %% [markdown]
# ## 3. Selectivity
#
# Mamba's change is one sentence long: make $$\Delta$$, $$B$$ and $$C$$ functions of the current token. Each token
# then decides, through its own $$\Delta_t$$, how much of the state to keep and how much of itself to write:
#
# $$h_t = e^{\Delta_t a}\, h_{t-1} + \Delta_t\, b_t\, x_t, \qquad y_t = c_t^{\top} h_t.$$
#
# The step $$\Delta_t$$ works as a gate. A large $$\Delta_t$$ drives $$e^{\Delta_t a}$$ toward 0 and multiplies the
# token's input: the state is reset and the token is written in. A $$\Delta_t$$ near zero keeps the state as it was
# and lets the token pass without trace. Gu and Dao call this *selectivity*, and they show its point on a toy task,
# *selective copying*: remember the coloured tokens scattered among noise tokens and reproduce them at the end. A
# time-invariant model can't do it, because it can't tell a coloured token from noise; a selective one can, because
# noise tokens get a $$\Delta$$ near zero.
#
# Selectivity has a price: the recurrence is no longer a fixed convolution, so the FFT trick of S4 is gone. Mamba
# pays it with a scan, section 6, and with a GPU kernel written for it.

# %% [markdown]
# ## 4. The Mamba block
#
# The selective state space model is the core of a block that also contains a short convolution and a gate. One
# block, for a sequence of $$L$$ vectors of width $$D$$:
#
# 1. `in_proj` widens each vector to $$2E$$, with $$E = 2D$$, and splits it into a stream $$u$$ and a gate $$z$$.
# 2. `conv1d` mixes each channel of $$u$$ with its three predecessors, a causal convolution of width 4, then SiLU.
#    It gives the state space model a little local context to compute its $$\Delta$$, $$B$$, $$C$$ from.
# 3. `x_proj` maps each $$u_t$$ to a low-rank seed of $$\Delta_t$$ (rank $$R$$), and to $$b_t$$ and $$c_t$$, $$N$$
#    numbers each, shared by all $$E$$ channels.
# 4. `dt_proj` expands the seed to one step per channel, and a softplus keeps the steps positive.
# 5. The selective scan runs $$E$$ independent channels, each with a state of $$N$$ numbers, and adds a skip
#    $$D \odot u$$.
# 6. The output is multiplied by $$\mathrm{SiLU}(z)$$, the gate, and `out_proj` narrows it back to $$D$$.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a10/block-light.png" width="760" alt="The Mamba block of mamba-130m with the shape of every tensor for L tokens. The input, L × 768, is widened by in_proj to L × 3,072 and split into a stream u and a gate z, L × 1,536 each. The stream goes through a causal convolution of width 4 and SiLU. x_proj turns it into a step seed of 48 numbers and B and C, 16 numbers each, per token; dt_proj and softplus give a step Δ for each of the 1,536 channels. The selective scan runs 1,536 channels with a state of 16 each, using A (1,536 × 16) and the skip D. Its output is multiplied by SiLU of the gate and out_proj narrows it back to L × 768">
#
# The matrix $$A$$ is stored as `A_log`, $$a = -e^{A_{\log}}$$, which keeps every entry negative whatever the
# training does. Mamba-130m has $$D = 768$$, $$E = 1{,}536$$, $$N = 16$$, $$R = 48$$ and 24 such blocks with no
# attention and no MLP between them: the block is the whole layer.

# %%
REPO = "state-spaces/mamba-130m-hf"
mamba_config = json.load(open(hf_hub_download(REPO, "config.json")))
for key in [
    "hidden_size",
    "intermediate_size",
    "state_size",
    "time_step_rank",
    "conv_kernel",
    "num_hidden_layers",
    "vocab_size",
]:
    print(f"{key:18} {mamba_config[key]}")


# %% [markdown]
# Two scans will be written in section 6; `Mixer` takes one as an argument. The loop is the definition: run the
# recurrence one step at a time, for all channels and all $$N$$ state numbers at once.


# %%
def scan_loop(decay, write, c):
    # decay, write: (batch, channels, L, N); c: (batch, L, N) -> y: (batch, channels, L)
    h = torch.zeros(decay.shape[0], decay.shape[1], decay.shape[3], device=decay.device)
    ys = []
    for t in range(decay.shape[2]):
        h = decay[:, :, t] * h + write[:, :, t]
        ys.append((h * c[:, None, t]).sum(-1))
    return torch.stack(ys, dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


# %%
class Mixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, e = c["hidden_size"], c["intermediate_size"]
        self.n, self.rank = c["state_size"], c["time_step_rank"]
        k = c["conv_kernel"]
        self.in_proj = nn.Linear(d, 2 * e, bias=c["use_bias"])
        self.conv1d = nn.Conv1d(e, e, k, groups=e, padding=k - 1, bias=c["use_conv_bias"])
        self.x_proj = nn.Linear(e, self.rank + 2 * self.n, bias=False)
        self.dt_proj = nn.Linear(self.rank, e, bias=True)
        self.A_log = nn.Parameter(torch.zeros(e, self.n))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d, bias=c["use_bias"])
        self.scan = scan_loop
        self.keep = False  # section 5 reads the steps
        self.steps = None

    def forward(self, x):
        L = x.shape[1]
        u, gate = self.in_proj(x).transpose(1, 2).chunk(2, dim=1)  # (batch, E, L) each
        u = F.silu(self.conv1d(u)[..., :L])
        seed, b, c = self.x_proj(u.transpose(1, 2)).split(
            [self.rank, self.n, self.n], dim=-1
        )
        step = F.softplus(self.dt_proj(seed)).transpose(1, 2)  # (batch, E, L)
        if self.keep:
            self.steps = step.detach()
        a = -torch.exp(self.A_log.float())  # (E, N), every entry negative
        decay = torch.exp(step[..., None] * a[:, None, :])  # (batch, E, L, N)
        write = step[..., None] * b[:, None].float() * u[..., None].float()
        y = self.scan(decay, write, c.float()).to(u.dtype)
        y = y + u * self.D[None, :, None]
        y = y * F.silu(gate)
        return self.out_proj(y.transpose(1, 2))


# %%
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = RMSNorm(c["hidden_size"], c["layer_norm_epsilon"])
        self.mixer = Mixer(c)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Block(c) for _ in range(c["num_hidden_layers"])])
        self.norm_f = RMSNorm(c["hidden_size"], c["layer_norm_epsilon"])
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)

    def forward(self, ids):
        x = self.embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm_f(x)).float()


# %% [markdown]
# Where do the parameters of a block go? Counted from the class:

# %%
block = Block(mamba_config)
rows = []
for name, p in block.mixer.named_parameters():
    rows.append(
        {
            "part": name,
            "shape": " × ".join(str(s) for s in p.shape),
            "parameters": p.numel(),
        }
    )
parts = pd.DataFrame(rows)
parts["share"] = parts.parameters / parts.parameters.sum()
table(parts, digits=3)

# %% [markdown]
# The two projections, in and out, hold almost all of it, as the two matrices of an MLP do in a Transformer. The state
# space model itself, $$A$$, $$D$$, the convolution and the step's projections, is a small fraction: Mamba's work is
# done by a few thousand numbers per channel acting over time.
#
# Now the weights. The library's model runs in the same process here, because at 130 million parameters two copies
# fit anywhere; the smoke run uses a small random model of the same class.

# %%
from transformers import MambaConfig, MambaForCausalLM

if SMOKE:
    small = MambaConfig(
        vocab_size=mamba_config["vocab_size"],
        hidden_size=64,
        intermediate_size=128,
        state_size=8,
        time_step_rank=4,
        num_hidden_layers=2,
    )
    torch.manual_seed(0)
    reference = MambaForCausalLM(small).eval()
    mamba_config = small.to_dict()
    state = {k.removeprefix("backbone."): v for k, v in reference.state_dict().items()}
else:
    reference = MambaForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
    state = load_file(hf_hub_download(REPO, "model.safetensors"))
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
state.setdefault("lm_head.weight", state["embeddings.weight"])  # tied to the embeddings
mamba = Mamba(mamba_config)
mamba.load_state_dict(state, strict=True)
mamba.eval()

tok = AutoTokenizer.from_pretrained(REPO)


def tokens_of(ids):
    return [tok.decode([i]) for i in ids[0]]


sentence = (
    "In December 1911 Roald Amundsen reached the South Pole with four men and fifty-two "
    "dogs; Robert Falcon Scott arrived thirty-four days later and found a Norwegian tent."
)
ids = tok(sentence, return_tensors="pt").input_ids
with torch.no_grad():
    theirs = reference(ids).logits
    ours = mamba(ids)
print("tokens:", ids.shape[1])
print("largest logit:", f"{theirs.abs().max().item():.1f}")
print("max difference from Hugging Face:", f"{(ours - theirs).abs().max().item():.1e}")

# %% [markdown]
# Our loop and the library's agree to the rounding of 32-bit floats accumulated over 24 layers. Every Mamba block
# is now ours to open.

# %% [markdown]
# ### One sentence through one block
#
# The class hides the shapes, so here is the sentence going through the first block by hand, one stage at a time,
# with the shape of every tensor and a few of its numbers. Follow it with the block diagram above.

# %%
m = mamba.layers[0].mixer
with torch.no_grad():
    x = mamba.layers[0].norm(mamba.embeddings(ids))
    print("tokens in, normalised:          ", tuple(x.shape), "  (batch, L, D)")
    both = m.in_proj(x).transpose(1, 2)
    print("in_proj, channels first:        ", tuple(both.shape), "  (batch, 2E, L)")
    u, gate = both.chunk(2, dim=1)
    print("stream u and gate z:            ", tuple(u.shape), "each")
    L = ids.shape[1]
    u = F.silu(m.conv1d(u)[..., :L])
    print("after the convolution and SiLU: ", tuple(u.shape))
    seed, b, c = m.x_proj(u.transpose(1, 2)).split([m.rank, m.n, m.n], dim=-1)
    print(
        "x_proj → step seed, B, C:       ",
        tuple(seed.shape),
        tuple(b.shape),
        tuple(c.shape),
    )
    step = F.softplus(m.dt_proj(seed)).transpose(1, 2)
    print(
        "dt_proj, softplus → Δ:          ",
        tuple(step.shape),
        "  one step per channel and token",
    )
    a = -torch.exp(m.A_log.float())
    print("A, all negative:                ", tuple(a.shape))
    decay = torch.exp(step[..., None] * a[:, None, :])
    write = step[..., None] * b[:, None] * u[..., None]
    print("decay e^(Δa) and write ΔBu:     ", tuple(decay.shape), "  (batch, E, L, N)")
    y = scan_loop(decay, write, c)
    print("the scan, read out by C:        ", tuple(y.shape))
    y = (y + u * m.D[None, :, None]) * F.silu(gate)
    out = m.out_proj(y.transpose(1, 2))
    print(
        "skip D, gate, out_proj:         ",
        tuple(out.shape),
        "  back to the width of the model",
    )
    print("same as the block's own forward:", f"{(out - m(x)).abs().max():.1e}")

print()
word = 4
print(f"channel 0 at token {word} ({tokens_of(ids)[word]!r}):")
print("  step Δ:", f"{step[0, 0, word]:.4f}", "  share of the state kept:", end=" ")
print(" ".join(f"{v:.2f}" for v in decay[0, 0, word, :6].tolist()), "…")

# %% [markdown]
# ## 5. What the model writes down
#
# For each token and each of the 1,536 channels of a layer the model chooses a step $$\Delta_t$$. A useful single
# number per token is how much of the state it *replaces*: $$1 - e^{\Delta_t a}$$, averaged over channels and state
# numbers. Zero means the token passed through without touching the state; one means the state was wiped and the
# token written instead.

# %%
for layer in mamba.layers:
    layer.mixer.keep = True
with torch.no_grad():
    mamba(ids)
tokens = tokens_of(ids)
replaced = []
for layer in mamba.layers:
    m = layer.mixer
    a = -torch.exp(m.A_log.detach().float())  # (E, N)
    kept = torch.exp(m.steps[0][:, :, None] * a[:, None, :])  # (E, L, N)
    replaced.append((1 - kept).mean(dim=(0, 2)))
    m.keep = False
replaced = torch.stack(replaced)  # (layers, L)
relative = (
    replaced / replaced.mean(dim=1, keepdim=True)
).numpy()  # 1 = the layer's average

show = list(range(0, len(mamba.layers), max(1, len(mamba.layers) // 6)))
fig, ax = plt.subplots(figsize=(9.6, 3.0))
ax.imshow(relative[show], cmap="Purples", aspect="auto", vmin=0.4, vmax=1.6)
ax.set_xticks(
    range(len(tokens)), [s.strip() or "␣" for s in tokens], rotation=70, fontsize=8
)
ax.set_yticks(range(len(show)), [f"layer {i}" for i in show])
ax.set_title("How much of the state each token replaces, against the layer's average")
plt.show()

middle = relative[len(relative) // 12 : len(relative) // 2 + 2].mean(axis=0)
order = np.argsort(-middle)
print("replace most: ", " · ".join(tokens[i].strip() for i in order[:8]))
print("replace least:", " · ".join(tokens[i].strip() for i in order[-8:]))

# %% [markdown]
# Each layer is divided by its own average, because the early layers rewrite much more of their state than the late
# ones, and that difference would drown everything else. What remains is a pattern you can read. The tokens that
# *start* something replace the most: the first word of the sentence, the first piece of a name, "Robert", the month.
# The tokens that *finish* something replace the least: the remaining pieces of "Amundsen", which the tokenizer cut
# into four, and the second half of "fifty-two" and "thirty-four". Once the state holds "Ro" and "ald", the rest of
# the name is predictable and there is little new to write. Nobody told the model any of this; the steps were learned
# from predicting the next word.
#
# The same steps give each channel a *memory horizon*: the number of tokens after which a channel keeps less than a
# third, $$e^{-1}$$, of what it held. For a channel with entry $$a$$ and a typical step $$\Delta$$ that is
# $$1 / (\Delta \lvert a\rvert)$$ tokens. Across all channels and state numbers of the whole model, over a page of
# text:

# %%
page = tok(
    "The expedition left the Bay of Whales on 20 October with four sledges. The weather "
    "was clear, the dogs were strong, and by the end of the first week they had covered "
    "more ground than Scott's party would in a month. Amundsen had planned the depots "
    "with care: every one was marked with flags across the line of march, so that a "
    "party returning in poor light could not miss it.",
    return_tensors="pt",
).input_ids
for layer in mamba.layers:
    layer.mixer.keep = True
with torch.no_grad():
    mamba(page)
horizons = []
for layer in mamba.layers:
    m = layer.mixer
    a = torch.exp(m.A_log.detach().float())  # |a|, (E, N)
    typical_step = m.steps[0].median(dim=-1).values  # (E,)
    horizons.append((1 / (typical_step[:, None] * a)).flatten())
    m.keep = False
horizons = torch.cat(horizons).clamp(max=1e6).numpy()

fig, ax = plt.subplots(figsize=(7.2, 3.2))
bins = np.logspace(-2, 6, 50)
ax.hist(horizons, bins=bins, color=COLORS["att"])
ax.set_xscale("log")
ax.set_xlabel("memory horizon, tokens")
ax.set_ylabel("channel × state pairs")
ax.set_title("How long the channels of mamba-130m remember")
plt.show()
for q in [0.1, 0.5, 0.9, 0.99]:
    print(f"{q:.0%} of horizons are under {np.quantile(horizons, q):,.2g} tokens")

# %% [markdown]
# Most channels forget within a few tokens: they handle the local business of syntax. A long tail keeps its content
# for hundreds or thousands of tokens, and those are the ones that carry a topic or a name across a page. The section
# 2 question has an answer too: for the short-memory channels $$\Delta \lvert a \rvert$$ is large and the Euler step
# overstates the input, but those channels forget it in a token or two anyway; for the long-memory ones the product is
# small and the shortcut is exact for practical purposes.

# %% [markdown]
# ## 6. The scan
#
# The loop of section 4 is correct and slow: $$L$$ steps, one after the other. A Transformer processes all positions
# of a sequence at once, and that is half of why it won. Mamba recovers the parallelism through a property of the
# recurrence $$h_t = \bar a_t h_{t-1} + w_t$$: it is linear, so two consecutive steps compose into one step of the
# same form,
#
# $$(\bar a_2, w_2) \circ (\bar a_1, w_1) = (\bar a_2 \bar a_1,\; \bar a_2 w_1 + w_2),$$
#
# and this composition is associative. Associative operations can be accumulated over a sequence in $$\log_2 L$$
# rounds instead of $$L$$ steps: in round $$k$$, every position combines itself with the position $$2^k$$ to its left.
# After the last round every position holds the composition of everything before it. This is the Hillis–Steele scan
# from the parallel algorithms of the 1980s, and it needs only elementwise multiplications and additions.


# %%
def scan_parallel(decay, write, c):
    a, h = decay, write
    shift = 1
    while shift < a.shape[2]:
        a_left = F.pad(
            a[:, :, :-shift], (0, 0, shift, 0), value=1.0
        )  # identity beyond the start
        h_left = F.pad(h[:, :, :-shift], (0, 0, shift, 0), value=0.0)
        h = a * h_left + h
        a = a * a_left
        shift *= 2
    return (h * c[:, None]).sum(-1)


torch.manual_seed(SEED)
decay = torch.rand(1, 3, 37, 4)
write = torch.randn(1, 3, 37, 4)
c = torch.randn(1, 37, 4)
gap = (scan_parallel(decay, write, c) - scan_loop(decay, write, c)).abs().max()
print("parallel against loop:", f"{gap:.1e}")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a10/scan-light.png" width="760" alt="The Hillis–Steele scan on eight positions. In the first round every position combines with its left neighbour, in the second with the position two to its left, in the third with the position four to its left; after three rounds, the logarithm of eight, every position holds the combined effect of all the steps before it, and each round is one elementwise operation over the whole sequence">
#
# The parallel scan does more work than the loop, $$L \log_2 L$$ combinations instead of $$L$$, and keeps a tensor of
# shape (batch, channels, $$L$$, $$N$$) in memory, $$N$$ times the size of the layer's activations. On a GPU it still
# wins by a wide margin, because every round is one big elementwise operation instead of $$L$$ small ones.

# %%
sizes = [256, 512] if SMOKE else [256, 1024, 4096]
timing = []
for L in sizes:
    torch.manual_seed(SEED)
    decay = torch.rand(1, 1536, L, 16, device=device)
    write = torch.randn(1, 1536, L, 16, device=device)
    c = torch.randn(1, L, 16, device=device)
    for name, scan in [("loop", scan_loop), ("parallel", scan_parallel)]:
        scan(decay, write, c)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        scan(decay, write, c)
        if device == "cuda":
            torch.cuda.synchronize()
        timing.append({"tokens": L, "scan": name, "ms": (time.time() - start) * 1000})
table(pd.DataFrame(timing).pivot(index="tokens", columns="scan", values="ms"), digits=1)

# %% [markdown]
# Mamba's own kernel goes further, and its design is FlashAttention's lesson applied to a recurrence. The tensor of
# all states, (batch, channels, $$L$$, $$N$$), is never written to the GPU's main memory: the kernel loads $$\Delta$$,
# $$A$$, $$B$$, $$C$$ and $$x$$ into the fast on-chip memory, runs the scan there and writes back only the outputs
# $$y$$. For training it recomputes the states in the backward pass instead of storing them. With that kernel, the
# paper reports, the scan runs 20 to 40 times faster than a standard scan in PyTorch.

# %% [markdown]
# ## 7. Generation with a state of fixed size
#
# To generate, a Transformer keeps its KV cache; Mamba keeps, per layer, the state $$h$$, $$E \times N$$ numbers, and
# the last three inputs of the convolution. One step of generation reads the new token, updates both, and forgets
# the token. The step is the recurrence of section 3 for one position:


# %%
def mixer_step(m, x, state):
    window, h = (
        state  # the convolution's last inputs (batch, E, k), the state (batch, E, N)
    )
    u, gate = m.in_proj(x).chunk(2, dim=-1)  # (batch, E) each
    window = torch.cat([window[:, :, 1:], u[:, :, None]], dim=-1)
    u = F.silu((window * m.conv1d.weight[:, 0]).sum(-1) + m.conv1d.bias)
    seed, b, c = m.x_proj(u).split([m.rank, m.n, m.n], dim=-1)
    step = F.softplus(m.dt_proj(seed))  # (batch, E)
    a = -torch.exp(m.A_log.float())
    h = torch.exp(step[..., None] * a) * h + step[..., None] * b[:, None] * u[..., None]
    y = (h * c[:, None]).sum(-1) + u * m.D
    return m.out_proj(y * F.silu(gate)), (window, h)


def model_step(model, token, states):
    x = model.embeddings(token)
    new_states = []
    for layer, state in zip(model.layers, states):
        out, state = mixer_step(layer.mixer, layer.norm(x), state)
        x = x + out
        new_states.append(state)
    return model.lm_head(model.norm_f(x)).float(), new_states


def empty_states(model, batch=1):
    states = []
    for layer in model.layers:
        m = layer.mixer
        e, k = m.D.shape[0], m.conv1d.weight.shape[-1]
        dev = m.D.device
        states.append(
            (torch.zeros(batch, e, k, device=dev), torch.zeros(batch, e, m.n, device=dev))
        )
    return states


with torch.no_grad():
    full = mamba(ids)
    states = empty_states(mamba)
    stepped = []
    for t in range(ids.shape[1]):
        logits, states = model_step(mamba, ids[:, t], states)
        stepped.append(logits)
stepped = torch.stack(stepped, dim=1)
state_bytes = sum(w.numel() + h.numel() for w, h in states) * 2  # in 16-bit floats
print("step by step against the whole sequence:", f"{(stepped - full).abs().max():.1e}")
print(f"the whole state of the model: {state_bytes / 1e6:.2f} MB, at any length")

# %% [markdown]
# Lesson 8 measured what a token costs in a Transformer's cache. The same count for Pythia-160m, the Transformer that
# EleutherAI trained on the same 300 billion tokens of the Pile as mamba-130m, with the same tokenizer:

# %%
pythia_config = json.load(open(hf_hub_download("EleutherAI/pythia-160m", "config.json")))
per_token = (
    2 * pythia_config["num_hidden_layers"] * pythia_config["hidden_size"] * 2
)  # K and V, fp16
memory = []
for n in [2048, 32768, 131072, 1_000_000]:
    memory.append(
        {
            "tokens read": n,
            "Pythia-160m cache, MB": per_token * n / 1e6,
            "mamba-130m state, MB": state_bytes / 1e6,
        }
    )
table(pd.DataFrame(memory), digits=2)

# %% [markdown]
# The state is about as big as Pythia's cache after a few dozen tokens, and it stays that size. This is the whole
# promise of the architecture, and it comes with a question that the next sections answer: what doesn't fit into a
# state that never grows?

# %% [markdown]
# ## 8. Mamba-2: the same model as attention
#
# Six months after Mamba, Dao and Gu published Mamba-2 with a title that sounds like a provocation: *Transformers are
# SSMs*. The change to the model is small. $$A$$ becomes one number per head instead of one per channel and state
# number, the channels are grouped into heads of 64 like attention heads, $$B$$ and $$C$$ are computed in parallel
# with the input instead of after it, and the state grows from 16 numbers to 128. The consequence is large. Unroll
# the recurrence with a scalar $$a$$ per head and the output at position $$t$$ is a weighted sum over all earlier
# positions:
#
# $$y_t = \sum_{s \le t} \underbrace{c_t^{\top} b_s}_{\text{like } q_t^{\top} k_s}\;
# \underbrace{e^{\,a(\Delta_{s+1} + \dots + \Delta_t)}}_{\text{the mask}}\; \Delta_s\, x_s.$$
#
# That is attention. $$c_t$$ plays the query, $$b_s$$ the key, $$\Delta_s x_s$$ the value, and instead of a softmax
# the scores are multiplied by a mask that decays with distance at a rate the tokens choose. The paper calls the
# equivalence *state space duality*: the same layer can be computed as a recurrence, cheap per token, or as a masked
# attention matrix, which is all matrix multiplication and therefore fast on a GPU. The practical algorithm mixes the
# two: attention inside chunks of 64 to 256 tokens, the recurrence between chunks. Four steps, all of them `einsum`.


# %%
def segment_sums(a):
    # a (..., T) -> (..., T, T): out[t, s] = a[s+1] + ... + a[t] for s <= t,
    # and -inf above the diagonal
    T = a.shape[-1]
    below = torch.ones(T, T, dtype=torch.bool, device=a.device).tril(-1)
    x = a[..., None].expand(*a.shape, T).masked_fill(~below, 0)
    x = torch.cumsum(x, dim=-2)
    on_or_below = torch.ones(T, T, dtype=torch.bool, device=a.device).tril()
    return x.masked_fill(~on_or_below, float("-inf"))


def ssd(x, a, b, c, chunk=64):
    # x (batch, L, heads, P), already times Δ; a (batch, L, heads) = Δ·A;
    # b and c (batch, L, heads, N)
    batch, L, heads, p = x.shape
    pad = (chunk - L % chunk) % chunk
    x, b, c = [F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, b, c)]
    a = F.pad(a, (0, 0, 0, pad))
    n = x.shape[1] // chunk
    x, b, c = [t.reshape(batch, n, chunk, heads, -1) for t in (x, b, c)]
    a = a.reshape(batch, n, chunk, heads).permute(
        0, 3, 1, 2
    )  # (batch, heads, chunks, chunk)
    a_cum = torch.cumsum(a, dim=-1)
    # 1. inside each chunk: attention with the decay as its mask
    mask = torch.exp(segment_sums(a))
    y_inside = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, mask, x)
    # 2. the state each chunk leaves behind
    to_end = torch.exp(a_cum[..., -1:] - a_cum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, to_end, x)
    # 3. the recurrence between chunks
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    between = torch.exp(segment_sums(F.pad(a_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", between, states)[:, :-1]
    # 4. what each position reads from the state that entered its chunk
    y_entering = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, torch.exp(a_cum))
    return (y_inside + y_entering).reshape(batch, n * chunk, heads, p)[:, :L]


def ssd_loop(x, a, b, c):
    h = torch.zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1], device=x.device)
    ys = []
    for t in range(x.shape[1]):
        h = (
            torch.exp(a[:, t])[..., None, None] * h
            + x[:, t, :, :, None] * b[:, t, :, None, :]
        )
        ys.append((h * c[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, dim=1)


torch.manual_seed(SEED)
x = torch.randn(1, 150, 3, 8)
a = -torch.rand(1, 150, 3) * 0.3
b = torch.randn(1, 150, 3, 16)
c = torch.randn(1, 150, 3, 16)
print(
    "chunks of 32 against the step-by-step loop:",
    f"{(ssd(x, a, b, c, 32) - ssd_loop(x, a, b, c)).abs().max():.1e}",
)

# %% [markdown]
# The Mamba-2 block keeps the shape of section 4 with the changes above: one projection makes the gate, the stream,
# $$B$$, $$C$$ and the steps at once; the convolution runs over the stream together with $$B$$ and $$C$$; and the
# gate is applied inside an RMSNorm at the end. The checkpoint is the 130-million-parameter Mamba-2 released by the
# authors, in the Hugging Face format of a community conversion, `AntonV/mamba2-130m-hf`.


# %%
class GatedRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        x32 = x.float() * F.silu(gate.float())
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class Mixer2(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.heads, self.p, self.n = c["num_heads"], c["head_dim"], c["state_size"]
        self.groups = c["n_groups"]
        self.e = self.heads * self.p
        conv = self.e + 2 * self.groups * self.n
        k = c["conv_kernel"]
        self.in_proj = nn.Linear(
            d, self.e + conv + self.heads, bias=c.get("use_bias", False)
        )
        self.conv1d = nn.Conv1d(conv, conv, k, groups=conv, padding=k - 1, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.heads))
        self.A_log = nn.Parameter(torch.zeros(self.heads))
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = GatedRMSNorm(self.e, c["layer_norm_epsilon"])
        self.out_proj = nn.Linear(self.e, d, bias=c.get("use_bias", False))
        self.chunk = c.get("chunk_size", 64)
        self.keep = False  # the next cell reads the matrix of one head
        self.parts = None

    def forward(self, x):
        batch, L, _ = x.shape
        sizes = [self.e, self.e + 2 * self.groups * self.n, self.heads]
        gate, xbc, dt = self.in_proj(x).split(sizes, dim=-1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :L].transpose(1, 2))
        u, b, c = xbc.split([self.e, self.groups * self.n, self.groups * self.n], dim=-1)
        step = F.softplus((dt + self.dt_bias).float())  # (batch, L, heads), in 32 bits
        a = -torch.exp(self.A_log.float())  # one number per head
        u = u.reshape(batch, L, self.heads, self.p)
        share = self.heads // self.groups
        b = b.reshape(batch, L, self.groups, self.n).repeat_interleave(share, dim=2)
        c = c.reshape(batch, L, self.groups, self.n).repeat_interleave(share, dim=2)
        if self.keep:
            self.parts = (u.detach(), step.detach(), a.detach(), b.detach(), c.detach())
        y = ssd(u.float() * step[..., None], step * a, b.float(), c.float(), self.chunk)
        y = y + u * self.D[:, None]
        y = self.norm(y.reshape(batch, L, self.e).to(x.dtype), gate)
        return self.out_proj(y)


class Mamba2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList()
        for _ in range(c["num_hidden_layers"]):
            block = nn.Module()
            block.norm = RMSNorm(c["hidden_size"], c["layer_norm_epsilon"])
            block.mixer = Mixer2(c)
            self.layers.append(block)
        self.norm_f = RMSNorm(c["hidden_size"], c["layer_norm_epsilon"])
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)

    def hidden(self, ids):
        x = self.embeddings(ids)
        for block in self.layers:
            x = x + block.mixer(block.norm(x))
        return self.norm_f(x)

    def forward(self, ids):
        return self.lm_head(self.hidden(ids)).float()


# %%
from transformers import Mamba2Config, Mamba2ForCausalLM

REPO2 = "AntonV/mamba2-130m-hf"
if SMOKE:
    small2 = Mamba2Config(
        vocab_size=50288,
        hidden_size=64,
        num_heads=4,
        head_dim=32,
        state_size=16,
        n_groups=1,
        num_hidden_layers=2,
        chunk_size=16,
    )
    torch.manual_seed(0)
    reference2 = Mamba2ForCausalLM(small2).eval()
    mamba2_config = small2.to_dict()
    state = {k.removeprefix("backbone."): v for k, v in reference2.state_dict().items()}
else:
    reference2 = Mamba2ForCausalLM.from_pretrained(REPO2, dtype=torch.float32).eval()
    mamba2_config = json.load(open(hf_hub_download(REPO2, "config.json")))
    state = load_file(hf_hub_download(REPO2, "model.safetensors"))
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
state.setdefault("lm_head.weight", state["embeddings.weight"])
mamba2 = Mamba2(mamba2_config)
mamba2.load_state_dict(state, strict=True)
mamba2.eval()
with torch.no_grad():
    gap = (mamba2(ids) - reference2(ids).logits).abs().max()
print("Mamba-2, max difference from Hugging Face:", f"{gap:.1e}")
for key in ["num_heads", "head_dim", "state_size", "chunk_size"]:
    print(f"{key:12} {mamba2_config[key]}")
del reference2

# %% [markdown]
# Now the attention matrix that Mamba-2 never writes down. For one head of a middle layer, on the Amundsen sentence,
# the matrix $$M_{ts} = c_t^{\top} b_s\, e^{a(\Delta_{s+1} + \dots + \Delta_t)}\, \Delta_s$$; next to it, the
# attention of a head of Pythia-160m on the same tokens. The check under the picture multiplies the matrix by the
# head's values and compares with the scan.

# %%
layer2 = len(mamba2.layers) // 2
mixer2 = mamba2.layers[layer2].mixer
mixer2.keep = True
with torch.no_grad():
    mamba2(ids)
mixer2.keep = False
u, step, a, b, c = mixer2.parts
head = 0
L = ids.shape[1]
decay = torch.exp(segment_sums(step[0, :, head] * a[head]))  # (L, L)
M = (c[0, :, head] @ b[0, :, head].T) * decay * step[0, None, :, head]
y_matrix = M @ u[0, :, head]
y_scan = ssd(u * step[..., None], step * a, b, c, mixer2.chunk)[0, :, head]
print("matrix form against the chunked scan:", f"{(y_matrix - y_scan).abs().max():.1e}")

# %%
from transformers import AutoModelForCausalLM

PYTHIA = (
    "EleutherAI/pythia-14m" if SMOKE else "EleutherAI/pythia-160m"
)  # 14m: same tokenizer, tiny
pythia = AutoModelForCausalLM.from_pretrained(
    PYTHIA, dtype=torch.float32, attn_implementation="eager"
).eval()
with torch.no_grad():
    attention = pythia(
        ids, output_attentions=True
    ).attentions  # one (1, heads, L, L) per layer
pythia_head = attention[len(attention) // 2][0, 0].numpy()

labels = [s.strip() or "␣" for s in tokens_of(ids)]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
weights = M.abs().numpy()
weights = weights / weights.sum(
    axis=1, keepdims=True
)  # each row as a share, like attention
for ax, matrix, title in [
    (axes[0], weights, f"Mamba-2, layer {layer2}, head {head}: |M|, rows as shares"),
    (axes[1], pythia_head, f"Pythia-160m, layer {len(attention) // 2}, head 0: softmax"),
]:
    ax.imshow(np.sqrt(matrix), cmap="Purples", vmin=0)
    ax.set_xticks(range(L), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(L), labels, fontsize=6)
    ax.set_title(title, fontsize=9)
plt.show()

# %% [markdown]
# Both matrices are lower-triangular: neither model looks ahead. The softmax head spends its attention on a few
# chosen tokens, often the first one, the attention sink of lesson 6. Mamba-2's matrix is a different texture: every
# row is a smooth decay into the past, interrupted where the steps $$\Delta$$ are large, because a large step erases
# what came before. One is a lookup, the other a fading memory, and the next section measures what that difference
# means in practice.

# %% [markdown]
# ## 9. Same data, different machinery
#
# Pythia-160m and mamba-130m are as close to a controlled experiment as two public models get: the same 300 billion
# tokens of the Pile, the same tokenizer, the same training context of 2,048 tokens, and a similar number of
# parameters outside the embeddings. Only the layer differs. Two tests.
#
# **Reading past the training length.** Lesson 9 watched a model with rotary positions fall off a cliff past its
# training length. Pythia rotates a quarter of each head's dimensions; Mamba has no positions at all, only the order
# in which the recurrence runs. Loss by position on the same PG-19 books as lesson 9:

# %%
import pandas as pd

pg19 = pd.read_parquet(
    "hf://datasets/emozilla/pg19-test/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
)
LENGTH = 512 if SMOKE else 8192
N_BOOKS = 2 if SMOKE else 6
book_ids = []
for text in pg19.text:
    t = tok(text[:120_000], return_tensors="pt").input_ids
    if t.shape[1] > LENGTH:
        book_ids.append(t[:, : LENGTH + 1])
    if len(book_ids) == N_BOOKS:
        break

for layer in mamba.layers:
    layer.mixer.scan = scan_parallel
mamba.to(device)
pythia.to(device)


@torch.no_grad()
def position_losses(model_logits, ids):
    logits = model_logits(ids[:, :-1].to(device))
    return F.cross_entropy(logits[0], ids[0, 1:].to(device), reduction="none").cpu()


curves = {"mamba-130m": torch.zeros(LENGTH), "Pythia-160m": torch.zeros(LENGTH)}
for ids_book in book_ids:
    curves["mamba-130m"] += position_losses(mamba, ids_book) / len(book_ids)
    curves["Pythia-160m"] += position_losses(
        lambda x: pythia(x).logits.float(), ids_book
    ) / len(book_ids)

fig, ax = plt.subplots(figsize=(7.4, 3.4))
width = max(4, LENGTH // 64)
for name, color in [("Pythia-160m", COLORS["muted"]), ("mamba-130m", COLORS["att"])]:
    smooth = np.convolve(curves[name].numpy(), np.ones(width) / width, mode="valid")
    ax.plot(smooth, color=color, linewidth=2, label=name)
ax.axvline(2048, color=COLORS["line"], linestyle="--", linewidth=1)
ax.set_xlabel("position in the book")
ax.set_ylabel("loss, nats per token")
ax.set_title(f"{len(book_ids)} PG-19 books, both trained on 2,048 tokens")
ax.legend(frameon=False)
plt.show()
for name in curves:
    inside = curves[name][1024:2048].mean().item()
    beyond = curves[name][-2048:].mean().item()
    print(f"{name:12} loss at 1–2k: {inside:.2f}, at the last 2k: {beyond:.2f}")

# %% [markdown]
# **Copying.** The second test asks for exactly what a fixed state can't promise: repeat a list of random words. A
# Transformer can look back at every word it was shown; Mamba has to have squeezed all of them into its state. Jelassi
# and colleagues found in 2024 that even a 410-million-parameter Pythia beats a 2.8-billion-parameter Mamba at
# looking up a number in a long phone book, and the same gap shows on this simplest of tasks. Each prompt has one
# worked example and then a list of a given length; an answer counts when every word comes back in order.

# %%
import random

rng = random.Random(SEED)
words = sorted(
    {w.lower() for w in pg19.text.iloc[0].split() if w.isalpha() and 3 <= len(w) <= 7}
)


def copy_prompt(n_words):
    example = " ".join(rng.sample(words, 6))
    target = " ".join(rng.sample(words, n_words))
    text = f"Repeat the words exactly.\nWords: {example}\nRepeat: {example}\n\n"
    text += f"Words: {target}\nRepeat:"
    return text, target


@torch.no_grad()
def mamba_continue(model, ids, steps):
    states = empty_states(model)
    for t in range(ids.shape[1]):
        logits, states = model_step(model, ids[:, t].to(device), states)
    out = []
    for _ in range(steps):
        token = logits.argmax(-1)
        out.append(token.item())
        logits, states = model_step(model, token, states)
    return out


@torch.no_grad()
def pythia_continue(ids, steps):
    out = pythia.generate(
        ids.to(device), max_new_tokens=steps, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return out[0, ids.shape[1] :].tolist()


copy_lengths = [5, 10] if SMOKE else [10, 25, 50, 100, 200]
COPY_TRIALS = 2 if SMOKE else 10
copied = {"mamba-130m": [], "Pythia-160m": []}
for n in copy_lengths:
    hits = {"mamba-130m": 0, "Pythia-160m": 0}
    for _ in range(COPY_TRIALS):
        text, target = copy_prompt(n)
        prompt = tok(text, return_tensors="pt").input_ids
        steps = len(tok(" " + target).input_ids) + 2
        said = {
            "mamba-130m": tok.decode(mamba_continue(mamba, prompt, steps)),
            "Pythia-160m": tok.decode(pythia_continue(prompt, steps)),
        }
        for name in hits:
            hits[name] += said[name].strip().startswith(target)
    for name in hits:
        copied[name].append(hits[name] / COPY_TRIALS)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
for name, color in [("Pythia-160m", COLORS["muted"]), ("mamba-130m", COLORS["att"])]:
    ax.plot(copy_lengths, copied[name], color=color, marker="o", linewidth=2, label=name)
ax.set_xscale("log")
ax.set_xticks(copy_lengths, [str(n) for n in copy_lengths])
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("words to repeat")
ax.set_ylabel("lists repeated exactly")
ax.set_title(f"Repeating a list of random words, {COPY_TRIALS} lists per length")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# The shorter lists are easy for both. As the list grows, the Transformer keeps copying, because every word it needs
# is still in its cache, one lookup away; Mamba's state has to hold the whole list at once in a fixed number of
# numbers, and the longer the list, the less of each word survives. This is the trade at the heart of the
# architecture: a state that doesn't grow is a state that can't hold everything. For most of language that is fine,
# because most of what we read doesn't have to be recalled word for word. For the parts that do, a little attention
# goes a long way, which is the idea of the next section.

# %% [markdown]
# ## 10. Hybrids
#
# The answer the industry settled on is not "Mamba or attention" but a mixture: mostly recurrent layers for cheap long
# reading, and a few attention layers that can look anything up. The Mamba-2 paper measured the ratio itself, on a
# 350-million-parameter model: replacing about one layer in ten with attention gave the best perplexity, better than
# either pure design. Here are the layouts of seven recent models, read from their configs. Each config describes its
# layout in its own way, so each needs a line of its own.


# %%
def layout(repo):
    c = read_hybrid_config(repo)
    n = c.get("num_hidden_layers") or c.get("n_layer")
    if "hybrid_override_pattern" in c:  # Nemotron-H: M Mamba, * attention, - an MLP layer
        names = {"M": "ssm", "*": "attention", "-": "mlp"}
        return [names[ch] for ch in c["hybrid_override_pattern"]]
    if "layer_types" in c:  # Granite 4.0
        return ["attention" if "attention" in t else "ssm" for t in c["layer_types"]]
    if "layers_block_type" in c:  # Zamba2: "hybrid" layers add a shared attention block
        return ["attention" if t == "hybrid" else "ssm" for t in c["layers_block_type"]]
    if "attn_layer_period" in c:  # Jamba
        period, offset = c["attn_layer_period"], c["attn_layer_offset"]
        return ["attention" if i % period == offset else "ssm" for i in range(n)]
    if "full_attention_interval" in c:  # Qwen3-Next: Gated DeltaNet, then attention
        k = c["full_attention_interval"]
        return ["attention" if (i + 1) % k == 0 else "linear" for i in range(n)]
    if "linear_attn_config" in c:  # Kimi Linear: its attention layers are counted from 1
        full = c["linear_attn_config"]["full_attn_layers"]
        return ["attention" if i + 1 in full else "linear" for i in range(n)]
    if "attn_layer_idx" in c:  # Mamba-2 with attention, from the Mamba-2 paper
        return ["attention" if i in c["attn_layer_idx"] else "ssm" for i in range(n)]
    return ["ssm"] * n


def read_hybrid_config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)


HYBRIDS = [
    ("state-spaces/mamba2attn-2.7b", "Mamba-2 + attention, 2024"),
    ("ai21labs/Jamba-v0.1", "Jamba, AI21, 2024"),
    ("Zyphra/Zamba2-1.2B", "Zamba2, Zyphra, 2024"),
    ("nvidia/Nemotron-H-8B-Base-8K", "Nemotron-H, NVIDIA, 2025"),
    ("ibm-granite/granite-4.0-h-tiny", "Granite 4.0 H, IBM, 2025"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3-Next, 2025"),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "Kimi Linear, 2025"),
]
layouts = {name: layout(repo) for repo, name in HYBRIDS}
colors = {
    "attention": COLORS["dec"],
    "ssm": COLORS["att"],
    "linear": COLORS["enc"],
    "mlp": COLORS["line"],
}
fig, ax = plt.subplots(figsize=(10.4, 3.6))
for row, kinds in enumerate(layouts.values()):
    for col, kind in enumerate(kinds):
        ax.add_patch(plt.Rectangle((col, -row - 0.4), 0.9, 0.8, color=colors[kind]))
ax.set_xlim(0, max(len(k) for k in layouts.values()))
ax.set_ylim(-len(layouts) + 0.4, 0.6)
ax.set_yticks([-i for i in range(len(layouts))], list(layouts))
ax.set_xlabel("layer")
for kind, label in [
    ("attention", "attention"),
    ("ssm", "Mamba"),
    ("linear", "linear attention"),
    ("mlp", "MLP only"),
]:
    ax.plot([], [], color=colors[kind], linewidth=8, label=label)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.show()

rows = []
for name, kinds in layouts.items():
    mixing = [k for k in kinds if k != "mlp"]
    rows.append(
        {
            "model": name,
            "layers that mix tokens": len(mixing),
            "attention": mixing.count("attention"),
            "share of attention": mixing.count("attention") / len(mixing),
        }
    )
table(pd.DataFrame(rows), digits=2)

# %% [markdown]
# The shares cluster where the Mamba-2 paper found its optimum: one attention layer in four to one in ten, spread
# evenly through the depth. Two other designs exist. Falcon-H1 from TII runs attention and Mamba-2 heads *in
# parallel* inside every layer instead of alternating them, and Qwen3-Next and Kimi Linear replace Mamba with a
# different recurrence, the gated delta rule, which writes into its state more selectively; that family is the
# subject of [a separate post on hybrid attention](https://xmarva.github.io/blog/2026/hybrid-attention/). Not
# everyone stayed: MiniMax used linear attention in most layers of MiniMax-01 and went back to full attention in
# every layer for MiniMax-M2.

# %% [markdown]
# ## 11. Whose DNA is this?
#
# Genomes are where long context is not a luxury. A gene's switches can sit tens of thousands of bases away from the
# gene, and the alphabet is four letters, so a "token" carries two bits and a meaningful stretch is long. This is the
# kind of data on which Mamba's linear cost matters, and HyenaDNA and Mamba itself were tested on it.
#
# The task: given a stretch of DNA, say which of seven species it comes from: human, chimpanzee, orangutan,
# marmoset, mouse, cattle or sheep. BGI released the benchmark in December 2025 under the MIT license, with training
# and test stretches taken from *different chromosomes*, so a model can't memorise positions. Mouse and cattle are
# easy to tell from a primate. Human and chimpanzee share about 99% of their DNA; at a thousand bases there may be
# no difference at all to see, and a longer window has more chances to contain one. So the question is how accuracy
# grows with context, and what each architecture can afford to read.
#
# The records are 32,768 bases long; we read the first few thousand of the training file, which is already shuffled
# across species, without downloading all of it.

# %%
from huggingface_hub import HfFileSystem

DNA = "datasets/BGI-HangzhouAI/Benchmark_Dataset-Primate_mammal_species_classification"
FOLDER = "Primate_mammal_species_classification_" + ("1024" if SMOKE else "32768")
N_TRAIN, N_TEST = (64, 32) if SMOKE else (7000, 1400)


def read_records(name, n):
    records = []
    with HfFileSystem().open(f"{DNA}/{FOLDER}/{name}", "r") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == n:
                break
    return records


train_records = read_records("train.jsonl", N_TRAIN)
test_records = read_records("test.jsonl", N_TEST)
species = sorted({r["species_name"] for r in train_records})
print("species:", ", ".join(species))
print(
    "train:", pd.Series([r["species_name"] for r in train_records]).value_counts().to_dict()
)
first = train_records[0]
print(
    f"one record: {first['species_name']}, chromosome {first['chr']},",
    f"{len(first['seq']):,} bases",
)
print(first["seq"][:120], "…")

# %% [markdown]
# Characters are tokens: A, C, G, T and N, which marks a base the sequencers couldn't read. Each record is cut to the
# context we want to test, its last $$L$$ bases, so that every length sees the same stretches of the same genomes
# and only the window changes.

# %%
LETTERS = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}  # anything else, N included, becomes 5; 0 is unused


def encode_dna(records, L):
    x = torch.full((len(records), L), 5, dtype=torch.long)
    for i, r in enumerate(records):
        seq = r["seq"][-L:]
        x[i, : len(seq)] = torch.tensor([LETTERS.get(ch, 5) for ch in seq])
    y = torch.tensor([species.index(r["species_name"]) for r in records])
    return x, y


# %% [markdown]
# Three learners, the same budget of bases read in training for each length. A *k-mer* baseline first, the way
# bioinformatics classified sequences for decades: count every word of six letters in the stretch, 4,096 counts, and
# fit a logistic regression, one linear layer on the counts. Then a small Mamba-2 built from this lesson's class, and
# a Transformer encoder of the same width and depth and about as many parameters. Both read the whole stretch, average
# their outputs over positions and classify the average.

# %%
def kmer_counts(x, k=6):
    # x: the encoded bases (n, L), 1 to 4 for A, C, G, T and 5 for N
    bases = (x - 1).numpy()
    windows = np.lib.stride_tricks.sliding_window_view(bases, k, axis=1)  # (n, L-k+1, k)
    codes = (windows * 4 ** np.arange(k - 1, -1, -1)).sum(
        -1
    )  # a word of k bases as a number
    valid = (windows < 4).all(-1)  # words that contain an N are skipped
    rows = np.zeros((len(bases), 4**k), dtype=np.float32)
    for i in range(len(bases)):
        rows[i] = np.bincount(codes[i][valid[i]], minlength=4**k)
    return rows / np.maximum(rows.sum(axis=1, keepdims=True), 1)


def fit_linear(features, labels, steps=500):
    # logistic regression: one linear layer and the cross-entropy, fitted with L-BFGS
    x = torch.tensor(features)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    layer = nn.Linear(x.shape[1], len(species))
    opt = torch.optim.LBFGS(layer.parameters(), max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(layer(x), labels) + 1e-3 * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    mean, std = torch.tensor(features).mean(dim=0), torch.tensor(features).std(dim=0) + 1e-6
    return lambda f: layer((torch.tensor(f) - mean) / std).argmax(-1)


# %%
class DNAMamba(nn.Module):
    def __init__(self, width=128, layers=4):
        super().__init__()
        c = dict(
            vocab_size=6,
            hidden_size=width,
            num_heads=2 * width // 64,
            head_dim=64,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            num_hidden_layers=layers,
            layer_norm_epsilon=1e-5,
            chunk_size=128,
        )
        self.body = Mamba2(c)
        self.head = nn.Linear(width, len(species))

    def forward(self, x):
        return self.head(self.body.hidden(x).mean(dim=1))


class DNATransformer(nn.Module):
    def __init__(self, width=128, layers=4, longest=32768):
        super().__init__()
        self.embed = nn.Embedding(6, width)
        self.position = nn.Embedding(longest, width)
        layer = nn.TransformerEncoderLayer(
            width,
            4,
            2 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, len(species))

    def forward(self, x):
        h = self.embed(x) + self.position(torch.arange(x.shape[1], device=x.device))
        return self.head(self.body(h).mean(dim=1))


for name, model in [("Mamba-2", DNAMamba()), ("Transformer", DNATransformer())]:
    count = sum(p.numel() for n, p in model.named_parameters() if "position" not in n)
    print(f"{name}: {count / 1e6:.2f} M parameters (without position embeddings)")

# %% [markdown]
# Every learner sees the same number of bases in training at every length: with longer stretches, fewer of them per
# step. Mixed precision on the GPU, as in the earlier lessons.

# %%
BASES_PER_RUN = 40_000 if SMOKE else 60_000_000
BASES_PER_STEP = 4096 if SMOKE else 65536


def train_dna(model, x, y):
    L = x.shape[1]
    batch = max(2, BASES_PER_STEP // L)
    steps = max(1, BASES_PER_RUN // (batch * L))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    start = time.time()
    for _ in range(steps):
        pick = torch.randint(0, len(x), (batch,))
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = F.cross_entropy(model(x[pick].to(device)), y[pick].to(device))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        schedule.step()
    return time.time() - start


@torch.no_grad()
def predict(model, x, batch=8):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x[i : i + batch].to(device)).argmax(-1).cpu())
    return torch.cat(out)


dna_lengths = [128, 512] if SMOKE else [1024, 4096, 16384]
dna = []
best_predictions = None
for L in dna_lengths:
    x_train, y_train = encode_dna(train_records, L)
    x_test, y_test = encode_dna(test_records, L)
    with torch.enable_grad():
        probe = fit_linear(kmer_counts(x_train), y_train)
    with torch.no_grad():
        accuracy = (probe(kmer_counts(x_test)) == y_test).float().mean().item()
    dna.append(
        {"bases": L, "learner": "6-mer counts", "accuracy": accuracy, "minutes": 0.0}
    )
    for name, make in [("Mamba-2", DNAMamba), ("Transformer", DNATransformer)]:
        torch.manual_seed(SEED)
        model = make()
        seconds = train_dna(model, x_train, y_train)
        guess = predict(model, x_test)
        accuracy = (guess == y_test).float().mean().item()
        dna.append(
            {"bases": L, "learner": name, "accuracy": accuracy, "minutes": seconds / 60}
        )
        if name == "Mamba-2" and L == dna_lengths[-1]:
            best_predictions = guess
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"{L:,} bases done")
dna = pd.DataFrame(dna)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
for name, color in [
    ("6-mer counts", COLORS["line"]),
    ("Transformer", COLORS["muted"]),
    ("Mamba-2", COLORS["att"]),
]:
    part = dna[dna.learner == name]
    ax.plot(part.bases, part.accuracy, color=color, marker="o", linewidth=2, label=name)
ax.axhline(1 / len(species), color=COLORS["line"], linestyle=":", linewidth=1)
ax.set_xscale("log", base=2)
ax.set_xticks(dna_lengths, [f"{n:,}" for n in dna_lengths])
ax.set_xlabel("bases read")
ax.set_ylabel("accuracy on held-out chromosomes")
ax.set_title(f"Which of {len(species)} species, by length of the stretch")
ax.legend(frameon=False)
plt.show()
table(dna.pivot(index="bases", columns="learner", values="accuracy"), digits=3)
table(dna.pivot(index="bases", columns="learner", values="minutes"), digits=1)

# %% [markdown]
# Where the best model still errs, at the longest stretch:

# %%
x_test, y_test = encode_dna(test_records, dna_lengths[-1])
confusion = np.zeros((len(species), len(species)))
for truth, guess in zip(y_test.tolist(), best_predictions.tolist()):
    confusion[truth, guess] += 1
confusion = confusion / confusion.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.imshow(confusion, cmap="Purples", vmin=0, vmax=1)
ax.set_xticks(range(len(species)), species, rotation=60, fontsize=8)
ax.set_yticks(range(len(species)), species, fontsize=8)
for i in range(len(species)):
    for j in range(len(species)):
        if confusion[i, j] >= 0.05:
            ink = "white" if confusion[i, j] > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{confusion[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=ink,
            )
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Mamba-2 on {dna_lengths[-1]:,} bases")
plt.show()

# %% [markdown]
# ## 12. Where state space models are today
#
# **Pure Mamba at scale.** Mistral released Codestral Mamba 7B in July 2024 and TII released Falcon Mamba 7B a month
# later, the first pure state space models of that size to compete with Transformers on standard benchmarks. Gu and
# Dao's group went on to Mamba-3 in 2026, with a state that updates more expressively and decodes faster.
#
# **Hybrids as the default.** Where long context and cheap serving matter, the shipped answer is a mixture: Jamba,
# Zamba2, Nemotron-H, Granite 4.0 H and Falcon-H1 keep a few attention layers for recall and use Mamba for the rest.
# AI21 tried Mamba-2 for Jamba 1.5 and kept Mamba-1, which worked better next to attention; the newest version is not
# always the best partner.
#
# **Beyond text.** Long sequences with simple alphabets are the natural home: DNA, as section 11 showed, audio, and
# time series.
#
# Mamba's passport, the model that made state space models a language technology, and the family tree:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a10/passport-light.png" width="760" alt="Mamba's passport. Published December 2023 by Albert Gu of Carnegie Mellon University and Tri Dao of Princeton University, code and weights under Apache 2.0. Type: a selective state space model, with no attention and no MLP blocks. Sizes: 130 million to 2.8 billion parameters. Training: 300 billion tokens of the Pile, context 2,048. State: per layer, 1,536 channels of 16 numbers in mamba-130m, plus a window of three inputs. Positions: none; order comes from the recurrence. Generation: the same memory and time for every token. Good at: long sequences at linear cost, fast generation, DNA and audio. Not for: recalling long spans exactly; pair it with attention">
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a10/family-light.png" width="760" alt="The family of the Transformer as this course covers it, with lesson 10, Mamba and hybrids, highlighted and lessons 1 to 9 done">
#
# ## References
#
# - Kalman, R. E. (1960). [A new approach to linear filtering and prediction problems](https://doi.org/10.1115/1.3662552). *Journal of Basic Engineering*, 82(1), 35–45.
# - Gu, A., Goel, K., & Ré, C. (2022). [Efficiently modeling long sequences with structured state spaces](https://arxiv.org/abs/2111.00396). *ICLR 2022*.
# - Gu, A., & Dao, T. (2023). [Mamba: linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752). arXiv:2312.00752.
# - Dao, T., & Gu, A. (2024). [Transformers are SSMs: generalized models and efficient algorithms through structured state space duality](https://arxiv.org/abs/2405.21060). *ICML 2024*.
# - Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). [Pythia: a suite for analyzing large language models across training and scaling](https://arxiv.org/abs/2304.01373). *ICML 2023*.
# - Jelassi, S., Brandfonbrener, D., Kakade, S. M., & Malach, E. (2024). [Repeat after me: Transformers are better than state space models at copying](https://arxiv.org/abs/2402.01032). arXiv:2402.01032.
# - Arora, S., Eyuboglu, S., Timalsina, A., et al. (2023). [Zoology: measuring and improving recall in efficient language models](https://arxiv.org/abs/2312.04927). arXiv:2312.04927.
# - Lieber, O., Lenz, B., Bata, H., et al. (2024). [Jamba: a hybrid Transformer–Mamba language model](https://arxiv.org/abs/2403.19887). arXiv:2403.19887.
# - Hillis, W. D., & Steele, G. L. (1986). [Data parallel algorithms](https://doi.org/10.1145/7902.7903). *Communications of the ACM*, 29(12), 1170–1183.
# - Nguyen, E., Poli, M., Faizi, M., et al. (2023). [HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution](https://arxiv.org/abs/2306.15794). *NeurIPS 2023*.
# - BGI-HangzhouAI (2025). [Primate and mammal species classification benchmark](https://huggingface.co/datasets/BGI-HangzhouAI/Benchmark_Dataset-Primate_mammal_species_classification), MIT license.
