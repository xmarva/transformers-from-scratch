# %% [markdown]
# # Lesson 6. Hyperparameter tuning on a free GPU with W&B Sweeps
#
# HOOK
#
# **What you'll do**
#
# 1. Track every training run with Weights & Biases, with or without an account.
# 2. Find the range of learning rates that work, for the 2017 design and for lesson 5's, in a minute each.
# 3. Write random search and successive halving from scratch and spend a fixed GPU budget on them.
# 4. Run the same kind of search as a W&B Sweep, with Bayesian optimization and early stopping.
# 5. Train the winner to the end, check it against lesson 5's model with the bootstrap, and publish it.
#
# **What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type →
# T4 GPU*. A free [Weights & Biases](https://wandb.ai) account is optional: add your API key as the secret
# `WANDB_API_KEY` (Kaggle: *Add-ons → Secrets*; Colab: the key icon on the left) to see the runs online and run
# the sweep of section 7. Without it everything else runs the same, and the runs are kept on disk. The notebook
# runs in about RUNTIME.

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
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from tfs import decoding, training
from tfs.comet import Comet
from tfs.modern import Seq2Seq
from tokenizers import Tokenizer

for package in ["wandb", "sacrebleu"]:  # Kaggle has wandb, Colab may not
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
import sacrebleu
import wandb

SMOKE = env.smoke()  # CI runs the lesson on a small slice; you tune in full
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" or SMOKE, "This lesson trains two dozen models: switch to a GPU."
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

WANDB_KEY = env.secret("WANDB_API_KEY")
if WANDB_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_KEY
WANDB_MODE = "online" if WANDB_KEY and not SMOKE else "offline"
os.environ["WANDB_SILENT"] = "true"  # wandb's own messages stay out of the outputs
PROJECT = "tfs-lesson-6"
print("W&B:", WANDB_MODE)

# %% [markdown]
# ## 1. What we tune, and what we can spend
#
# BUDGET_TEXT
#
# The data, the tokenizer and the batches, as in lesson 5. Since the batch size is one of the things we tune,
# the training pairs are cut into batches of three sizes, once:

# %%
tokenizer_path = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
splits = {}
for name in ["train", "validation", "test"]:
    path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", f"data/{name}.parquet",
                           repo_type="dataset")
    splits[name] = pd.read_parquet(path)
train, valid, test = splits["train"], splits["validation"], splits["test"]
if SMOKE:
    train = train.sample(3_000, random_state=SEED)
    valid = valid.head(300)
    test = test.sample(300, random_state=SEED)

VOCAB_SIZE = tokenizer.get_vocab_size()
PAD = tokenizer.token_to_id("<pad>")
BOS = tokenizer.token_to_id("<bos>")
EOS = tokenizer.token_to_id("<eos>")


def encode_all(sentences):
    return [encoding.ids for encoding in tokenizer.encode_batch(list(sentences))]


train_src, train_tgt = encode_all(train.en), encode_all(train.ru)
valid_src, valid_tgt = encode_all(valid.en), encode_all(valid.ru)
keep = []
for i in range(len(train_src)):
    if len(train_src[i]) <= 64 and len(train_tgt[i]) + 1 <= 64:
        keep.append(i)

special = {"pad_id": PAD, "bos_id": BOS, "eos_id": EOS}
BATCH_TOKENS = [3_000, 6_000, 12_000]  # lesson 3 used 6,000
batches = {}
for tokens in BATCH_TOKENS:
    batches[tokens] = training.make_batches(train_src, train_tgt, keep, tokens, **special)
valid_rows = range(len(valid_src))
valid_batches = training.make_batches(valid_src, valid_tgt, valid_rows, 6_000, **special)
for tokens in BATCH_TOKENS:
    print(f"batches of {tokens:,} tokens: {len(batches[tokens])} per epoch")

# %% [markdown]
# The model is lesson 5's, from `tfs.modern`: pre-norm with RMSNorm, SwiGLU and RoPE. For comparison, the 2017
# design is one switch away.

# %%
LLAMA_STYLE = {"norm": "rms", "ffn": "swiglu", "pos": "rope"}
DESIGN_2017 = {"norm": "post", "ffn": "relu", "pos": "sinusoidal"}


def new_model(setup=LLAMA_STYLE, dropout=0.1, seed=SEED):
    torch.manual_seed(seed)
    return Seq2Seq(setup, VOCAB_SIZE, dropout=dropout, pad_id=PAD).to(device)


# %% [markdown]
# ## 2. Tracking runs with Weights & Biases
#
# TRACKING_TEXT

# %%
run = wandb.init(project=PROJECT, mode=WANDB_MODE, name="lesson-5 recipe, one epoch",
                 config={"peak_lr": 1e-3, "warmup": 1_000, "dropout": 0.1,
                         "label_smoothing": 0.1, "batch_tokens": 6_000})
config = run.config  # the hyperparameters, stored with the run

random.seed(SEED)
model = new_model(dropout=config.dropout)
optimizer = training.adam(model)


def schedule(step):
    return config.peak_lr * min(step / config.warmup, math.sqrt(config.warmup / step))


half_epoch = len(batches[config.batch_tokens]) // 2
steps = 0
for half in range(1, 3):  # two halves of an epoch, measured after each
    losses = training.train_steps(model, optimizer, batches[config.batch_tokens], schedule,
                                  half_epoch, pad_id=PAD, first_step=steps + 1,
                                  label_smoothing=config.label_smoothing, device=device)
    steps += len(losses)
    loss = training.validation_loss(model, valid_batches, pad_id=PAD, device=device)
    run.log({"epochs": half / 2, "train loss": np.mean(losses), "validation loss": loss})

# %% [markdown]
# A table of translations goes with the run too, so that the numbers come with something to read:

# %%
probe = test.sample(8, random_state=SEED)
written = decoding.greedy(model, tokenizer, list(probe.en))
translations = wandb.Table(columns=["English", "reference", "after one epoch"],
                           data=[list(row) for row in zip(probe.en, probe.ru, written)])
run.log({"translations": translations})
run.summary["validation loss"] = loss
print("validation loss after one epoch:", round(loss, 3))
print("stored in:", run.url if WANDB_MODE == "online" else run.dir)
run.finish()

# %% [markdown]
# RUN_PAGE_TEXT
#
# ## 3. The learning rate first
#
# LR_INTRO
#
# A **learning-rate range test** ([Smith, 2017](https://arxiv.org/abs/1506.01186)) finds that range in one short
# run: start with a tiny learning rate and multiply it by the same factor at every step, so that it sweeps several
# orders of magnitude, and watch the loss. While the rate is too small, nothing happens; in the useful range the
# loss falls fast; beyond it, training breaks down.

# %%
def range_test(setup, start=1e-6, end=1.0, n_steps=600):
    random.seed(SEED)
    model = new_model(setup)
    optimizer = training.adam(model)
    factor = (end / start) ** (1 / n_steps)  # the same multiplication at every step

    def sweep_rate(step):
        return start * factor ** (step - 1)

    losses = training.train_steps(model, optimizer, batches[6_000], sweep_rate, n_steps,
                                  pad_id=PAD, device=device)
    rates = [sweep_rate(step) for step in range(1, len(losses) + 1)]
    return rates, losses


RANGE_STEPS = 600 if not SMOKE else 20
ranges = {}
for name, setup in [("2017 design", DESIGN_2017), ("lesson 5's design", LLAMA_STYLE)]:
    ranges[name] = range_test(setup, n_steps=RANGE_STEPS)

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for (name, (rates, losses)), color in zip(ranges.items(), [COLORS["muted"], COLORS["att"]]):
    smooth = pd.Series(losses).rolling(15, min_periods=1).mean()
    ax.plot(rates, smooth, color=color, label=name)
ax.set_xscale("log")
ax.set_ylim(top=12)
ax.set_xlabel("learning rate (rising every step)")
ax.set_ylabel("training loss, smoothed")
ax.set_title("A learning-rate range test")
ax.legend()
plt.show()

for name, (rates, losses) in ranges.items():
    smooth = pd.Series(losses).rolling(15, min_periods=1).mean().to_numpy()
    best_rate = rates[int(np.nanargmin(smooth))]
    print(f"{name}: lowest loss at a learning rate of {best_rate:.1e}")

# %% [markdown]
# RANGE_TEXT
#
# ## 4. Random search
#
# RANDOM_TEXT
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/06/grid-random-light.png" width="760" alt="Nine trials of grid search and nine of random search on a toy problem where only the horizontal hyperparameter matters. The grid tries only three distinct values of it; random search tries nine different values and usually lands closer to the minimum">
#
# SPACE_TEXT

# %%
search_rng = random.Random(SEED)


def log_uniform(low, high):
    return 10 ** search_rng.uniform(math.log10(low), math.log10(high))


def sample_config():
    return {
        "peak_lr": log_uniform(2e-4, 1e-2),
        "warmup": int(log_uniform(200, 4_000)),
        "dropout": search_rng.uniform(0.0, 0.3),
        "label_smoothing": search_rng.uniform(0.0, 0.2),
        "batch_tokens": search_rng.choice(BATCH_TOKENS),
    }


N_CONFIGS = 18 if not SMOKE else 3
configs = [sample_config() for _ in range(N_CONFIGS)]
table(pd.DataFrame(configs).head(6), digits=4)

# %% [markdown]
# ## 5. Successive halving
#
# HALVING_TEXT
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/06/halving-light.png" width="760" alt="Successive halving with 18 configurations: all train for half an epoch, the best third continues to 1.5 epochs, the best third of those to 4.5, and the winner to 20. The search costs 36 epochs instead of 360 for training every configuration to the end">
#
# A trial is one configuration with its own model, optimizer and W&B run, able to train for a given number of
# half-epochs and to pick up where it stopped:

# %%
class Trial:
    def __init__(self, number, config, run=None):
        self.number, self.config = number, config
        self.model = new_model(dropout=config["dropout"], seed=SEED + number)
        self.optimizer = training.adam(self.model)
        self.steps, self.epochs, self.history = 0, 0.0, []
        if run is None:  # every trial is a W&B run of its own, all of them open at once
            run = wandb.init(project=PROJECT, mode=WANDB_MODE, group="successive halving",
                             name=f"trial {number}", config=config, reinit="create_new")
        self.run = run

    def schedule(self, step):
        warmup, peak = self.config["warmup"], self.config["peak_lr"]
        return peak * min(step / warmup, math.sqrt(warmup / step))

    def train(self, half_epochs):
        own_batches = batches[self.config["batch_tokens"]]
        n_steps = max(1, int(len(own_batches) / 2 * half_epochs))
        smoothing = self.config["label_smoothing"]
        training.train_steps(self.model, self.optimizer, own_batches, self.schedule,
                             n_steps, pad_id=PAD, first_step=self.steps + 1,
                             label_smoothing=smoothing, device=device)
        self.steps += n_steps
        self.epochs += half_epochs / 2
        loss = training.validation_loss(self.model, valid_batches, pad_id=PAD,
                                        device=device)
        self.history.append({"epochs": self.epochs, "validation loss": loss})
        self.run.log({"epochs": self.epochs, "validation loss": loss})
        return loss


# %% [markdown]
# And the search: every rung trains the survivors up to its budget, ranks them by validation loss and keeps the
# best third. This is the long cell of the lesson.

# %%
RUNGS = [1, 3, 9]  # cumulative budget of each rung, in half-epochs
KEEP = 3  # keep the best 1 / KEEP

trials = [Trial(number, config) for number, config in enumerate(configs)]
survivors = list(trials)
done = 0
start = time.perf_counter()
for rung, budget in enumerate(RUNGS):
    for trial in survivors:
        trial.train(budget - done)
    done = budget
    survivors.sort(key=lambda trial: trial.history[-1]["validation loss"])
    minutes = (time.perf_counter() - start) / 60
    print("rung", rung, "|", len(survivors), "trials at", budget / 2, "epochs |",
          round(minutes), "min")
    if rung < len(RUNGS) - 1:
        survivors = survivors[: max(1, len(survivors) // KEEP)]
winner = survivors[0]

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for trial in trials:
    epochs = [record["epochs"] for record in trial.history]
    losses = [record["validation loss"] for record in trial.history]
    is_winner = trial is winner
    color = COLORS["att"] if is_winner else COLORS["line"]
    width = 2 if is_winner else 1
    ax.plot(epochs, losses, marker="o", markersize=3, color=color, linewidth=width)
ax.set_xlabel("epochs")
ax.set_ylabel("validation loss")
ax.set_ylim(top=min(6, ax.get_ylim()[1]))
ax.set_title("Eighteen starts, one finish")
plt.show()

rows = []
def furthest_first(trial):  # the longest-trained first, then by their last loss
    return (-trial.epochs, trial.history[-1]["validation loss"])


for trial in sorted(trials, key=furthest_first)[:6]:
    rows.append({"trial": trial.number, "epochs": trial.epochs,
                 "validation loss": trial.history[-1]["validation loss"], **trial.config})
table(pd.DataFrame(rows), digits=4)

# %% [markdown]
# HALVING_RESULT
#
# ## 6. What mattered
#
# MATTERED_INTRO

# %%
first_rung = pd.DataFrame([{**trial.config, "loss": trial.history[0]["validation loss"]}
                           for trial in trials])
names = ["peak_lr", "warmup", "dropout", "label_smoothing", "batch_tokens"]
fig, axes = plt.subplots(1, len(names), figsize=(12, 2.8), sharey=True)
for ax, name in zip(axes, names):
    ax.scatter(first_rung[name], first_rung["loss"], color=COLORS["data"], s=18)
    if name in ["peak_lr", "warmup"]:
        ax.set_xscale("log")
    ax.set_xlabel(name)
axes[0].set_ylabel("validation loss after half an epoch")
plt.tight_layout()
plt.show()

# rank correlation: replace every value by its rank, then the usual correlation
correlations = first_rung[names].rank().corrwith(first_rung["loss"].rank())
table(pd.DataFrame({"hyperparameter": names,
                   "rank correlation with the loss": correlations.values}), digits=2)

# %% [markdown]
# MATTERED_TEXT
#
# ## 7. The same search as a W&B Sweep
#
# SWEEP_INTRO
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/06/sweep-light.png" width="760" alt="How a W&B sweep runs: the sweep controller on W&B's servers holds the search space and the method; agents on one or more machines ask it for a configuration, train with it, and log the metric back; the controller uses the results to choose the next configuration and can stop runs that fall behind">
#
# BAYES_TEXT
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/06/bayes-light.png" width="760" alt="Bayesian optimization in one dimension: after four trials, a Gaussian process gives a mean estimate of the loss and a band of uncertainty; expected improvement is highest where the estimate is low or the uncertainty large, and the next trial goes there">
#
# The search space in W&B's terms is a dictionary, the same space as section 4:

# %%
SWEEP = {
    "method": "bayes",
    "metric": {"name": "validation loss", "goal": "minimize"},
    "parameters": {
        "peak_lr": {"distribution": "log_uniform_values", "min": 2e-4, "max": 1e-2},
        "warmup": {"distribution": "q_log_uniform_values", "min": 200, "max": 4_000,
                   "q": 1},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
        "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 0.2},
        "batch_tokens": {"values": BATCH_TOKENS},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 1, "eta": 3},
}

# %% [markdown]
# Each run of the sweep is a function the agent calls: it reads its configuration from W&B, trains for up to three
# half-epochs, and logs the validation loss after each, which is what the early stopping watches.

# %%
def sweep_run():
    with wandb.init() as run:  # the agent filled run.config with the controller's choice
        trial = Trial(0, dict(run.config), run=run)
        for _ in range(3):
            trial.train(1)


SWEEP_RUNS = 6
if WANDB_MODE == "online":
    sweep_id = wandb.sweep(SWEEP, project=PROJECT)
    wandb.agent(sweep_id, function=sweep_run, count=SWEEP_RUNS)
    swept = wandb.Api().sweep(f"{wandb.Api().default_entity}/{PROJECT}/{sweep_id}")
    rows = []
    for swept_run in swept.runs:
        rows.append({"run": swept_run.name, "state": swept_run.state,
                     "validation loss": swept_run.summary.get("validation loss"),
                     **{k: swept_run.config.get(k) for k in SWEEP["parameters"]}})
    table(pd.DataFrame(rows).sort_values("validation loss"), digits=4)
    print("the sweep:", swept.url)
else:
    print("no W&B key: the sweep needs W&B's servers, so this cell does nothing offline")

# %% [markdown]
# SWEEP_TEXT
#
# ## 8. The winner, trained to the end
#
# WINNER_INTRO

# %%
FINAL_EPOCHS = 20 if not SMOKE else 1
CHECKPOINT = WORK_DIR / "tuned.safetensors"
best = winner.history[-1]["validation loss"]  # where the search left it
save_file(winner.model.state_dict(), CHECKPOINT)
while winner.epochs < FINAL_EPOCHS:
    loss = winner.train(2)  # one more epoch
    if loss < best:
        best = loss
        save_file(winner.model.state_dict(), CHECKPOINT)
for trial in trials:
    trial.run.finish()
tuned = winner.model
tuned.load_state_dict(load_file(CHECKPOINT, device=device))
tuned.eval()
print("best validation loss:", round(best, 4))
table(pd.DataFrame([winner.config]), digits=4)

# %% [markdown]
# Lesson 5's model, trained with the recipe of 2017, from the Hub, on the same validation set and the whole test
# set, with lesson 4's metrics and the paired bootstrap:

# %%
LESSON5_REPO = "xmarva/tfs-transformer-en-ru-llama"
with open(hf_hub_download(LESSON5_REPO, "config.json")) as f:
    lesson5 = Seq2Seq(**json.load(f)).to(device).eval()
weights = hf_hub_download(LESSON5_REPO, "model.safetensors")
lesson5.load_state_dict(load_file(weights, device=device))

comet = Comet(device)
sentences = list(dict.fromkeys(test.en))
rows = []
for name, model in [("lesson 5", lesson5), ("tuned", tuned)]:
    translations = dict(zip(sentences, decoding.beam_search(model, tokenizer, sentences)))
    test[name] = test.en.map(translations)
    test[f"COMET {name}"] = comet.score(test.en, test[name], test.ru)
    loss = training.validation_loss(model, valid_batches, pad_id=PAD, device=device)
    rows.append({"model": name, "validation loss": loss,
                 "BLEU": sacrebleu.corpus_bleu(list(test[name]), [list(test.ru)]).score,
                 "chrF": sacrebleu.corpus_chrf(list(test[name]), [list(test.ru)]).score,
                 "COMET": test[f"COMET {name}"].mean()})
table(pd.DataFrame(rows), digits=4)

# %%
rng = np.random.default_rng(SEED)
a, b = test["COMET tuned"].to_numpy(), test["COMET lesson 5"].to_numpy()
differences = []
for _ in range(1_000):
    drawn = rng.integers(0, len(test), size=len(test))
    differences.append(a[drawn].mean() - b[drawn].mean())
low, high = np.percentile(differences, [2.5, 97.5])
print(f"COMET, tuned − lesson 5: {np.mean(differences):+.4f}")
print(f"95% of the draws between {low:+.4f} and {high:+.4f}")

# %% [markdown]
# FINAL_TEXT
#
# ## 9. Save and publish

# %%
MODEL_DIR = WORK_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
save_file(tuned.state_dict(), MODEL_DIR / "model.safetensors")
TUNED_CONFIG = {"setup": LLAMA_STYLE, "vocab_size": VOCAB_SIZE, "d_model": 256,
                "n_heads": 4, "d_ff": 1024, "n_layers": 3,
                "dropout": winner.config["dropout"], "pad_id": PAD}
(MODEL_DIR / "config.json").write_text(json.dumps(TUNED_CONFIG, indent=2))
(MODEL_DIR / "training.json").write_text(json.dumps(winner.config, indent=2))
print("saved:", sorted(path.name for path in MODEL_DIR.iterdir()))

# %%
MODEL_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
tags: [translation, transformer]
datasets: [xmarva/tfs-tatoeba-en-ru]
---
# A small English→Russian Transformer, tuned

Built in lesson 6 of the course "Transformers: from scratch to production":
https://github.com/xmarva/transformers-from-scratch

- lesson 5's model (pre-norm, RMSNorm, SwiGLU, RoPE), hyperparameters from a search
- the training hyperparameters: training.json
- best validation cross-entropy: {best:.3f}
"""

HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")
if HUB_USER and hf_token and not SMOKE:
    api = HfApi(token=hf_token)
    model_repo = f"{HUB_USER}/tfs-transformer-en-ru-tuned"
    api.create_repo(model_repo, exist_ok=True)
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=model_repo)
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=model_repo)
    print("published", model_repo)
else:
    print("not publishing; later lessons download the course's copy:")
    print("xmarva/tfs-transformer-en-ru-tuned")

# %% [markdown]
# ## References
#
# REFERENCES
