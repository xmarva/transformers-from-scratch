"""Train the models of the post on the corpus of prepare.py and record what the figures need.

Every model has 6 layers of width 256 with 4 attention heads and the same tokenizer; they differ only in the
feed-forward layers:

- Dense: a SwiGLU network of width 768 in every layer;
- MoE, 8 experts: 8 SwiGLU experts of width 384, 2 per token, so a token passes through the same 768 hidden units
  as in the dense model while the layer holds four times as many; three versions differ in load balancing;
- MoE, 32 experts: 32 experts of width 96, 8 per token: the same compute and the same parameters, cut finer;
- Dense, 4x wider: width 3,072, as many parameters as the mixtures and all of them used for every token.

Each run trains for STEPS steps of 32 sequences of 256 tokens drawn at random from the three kinds of text, with
AdamW, warmup and a cosine schedule. Every 50 steps it records the training loss and the share of tokens each
expert got in every layer; every 250 steps the validation loss on each kind of text. The record goes to
OUT/runs/<name>.json and the weights to OUT/checkpoints/<name>.pt; routing.py analyses the checkpoints.

    uv run python articles/moe/train.py                                    # all runs, one after another
    MODELS="Dense;MoE, 8 experts" SEEDS=1 uv run python articles/moe/train.py   # some of them, another seed
    SMOKE=1 uv run python articles/moe/train.py                            # a few steps, to check it runs
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from moe import Model

SMOKE = os.environ.get("SMOKE") == "1"
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-moe"))
OUT = Path(os.environ.get("OUT", DATA_DIR / "out"))
DOMAINS = ["en", "ru", "code"]
D_MODEL, LAYERS, HEADS, LENGTH, BATCH = 256, 6, 4, 256, 32
STEPS = 30 if SMOKE else int(os.environ.get("STEPS", 3000))
WARMUP, RATE, MIN_RATE = 200, 2e-3, 2e-4
AUX_WEIGHT = 0.01  # the coefficient of the Switch Transformer's balancing loss
BIAS_RATE = 0.001  # the bias update speed of DeepSeek-V3
MODELS = {
    "Dense": {"hidden": 768},
    "MoE, 8 experts": {"hidden": 384, "experts": 8, "top_k": 2, "balance": "loss"},
    "MoE, 8 experts, no balancing": {"hidden": 384, "experts": 8, "top_k": 2, "balance": "none"},
    "MoE, 8 experts, bias balancing": {"hidden": 384, "experts": 8, "top_k": 2, "balance": "bias"},
    "Dense, 4x wider": {"hidden": 3072},
    "MoE, 32 experts": {"hidden": 96, "experts": 32, "top_k": 8, "balance": "loss"},
}
if os.environ.get("MODELS"):
    MODELS = {name: MODELS[name] for name in os.environ["MODELS"].split(";")}
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]


def device_name():
    if os.environ.get("DEVICE") == "cpu":  # e.g. to analyse checkpoints while the GPU trains
        return "cpu", "cpu"
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name()
    if torch.backends.mps.is_available():
        return "mps", "Apple GPU (MPS)"
    return "cpu", "cpu"


def load(split):
    return {name: np.fromfile(DATA_DIR / f"{name}_{split}.bin", dtype=np.uint16) for name in DOMAINS}


def batch(texts, generator, device, size=BATCH):
    rows, domains = [], []
    for _ in range(size):
        name = DOMAINS[generator.integers(len(DOMAINS))]
        start = generator.integers(len(texts[name]) - LENGTH - 1)
        rows.append(texts[name][start:start + LENGTH + 1].astype(np.int64))
        domains.append(name)
    ids = torch.from_numpy(np.stack(rows)).to(device)
    return ids[:, :-1], ids[:, 1:], domains


def validation_batches(texts, per_domain, device):
    """Fixed, non-overlapping sequences from the start of each validation text."""
    out = {}
    for name in DOMAINS:
        rows = [texts[name][i * (LENGTH + 1):(i + 1) * (LENGTH + 1)].astype(np.int64) for i in range(per_domain)]
        ids = torch.from_numpy(np.stack(rows)).to(device)
        out[name] = (ids[:, :-1], ids[:, 1:])
    return out


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    losses = {}
    for name, (x, y) in batches.items():
        total = 0.0
        for i in range(0, len(x), 16):
            logits = model(x[i:i + 16])
            total += F.cross_entropy(logits.flatten(0, 1), y[i:i + 16].flatten(), reduction="sum").item()
        losses[name] = total / y.numel()
    model.train()
    return losses


def rate_at(step):
    if step < WARMUP:
        return RATE * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_RATE + 0.5 * (RATE - MIN_RATE) * (1 + math.cos(math.pi * progress))


def train(name, spec, seed, device, train_texts, val_batches):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = Model(8192, D_MODEL, LAYERS, HEADS, spec).to(device)
    total, active = model.parameters_count()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                                  lr=RATE, betas=(0.9, 0.95))
    moe = model.moe_layers()
    record = {"model": name, "spec": spec, "seed": seed, "parameters": total, "active_parameters": active,
              "steps": STEPS, "curve": [], "validation": [], "load": []}
    window_loss, window_load = [], [torch.zeros(layer.experts_count) for layer in moe]
    start = time.time()
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = rate_at(step)
        x, y, _ = batch(train_texts, generator, device)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        aux = sum(layer.aux_loss for layer in moe if layer.balance == "loss") if moe else 0.0
        optimizer.zero_grad(set_to_none=True)
        (loss + AUX_WEIGHT * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for i, layer in enumerate(moe):
            layer.update_bias(BIAS_RATE)
            window_load[i] += layer.load.cpu()
        window_loss.append(loss.item())
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            record["curve"].append({"step": step + 1, "loss": float(np.mean(window_loss))})
            if moe:
                record["load"].append({"step": step + 1,
                                       "layers": [(w / len(window_loss)).tolist() for w in window_load]})
            window_loss, window_load = [], [torch.zeros(layer.experts_count) for layer in moe]
        if (step + 1) % 250 == 0 or step + 1 == STEPS:
            losses = evaluate(model, val_batches)
            record["validation"].append({"step": step + 1, **losses})
            print(f"{name} seed {seed} step {step + 1}: train {record['curve'][-1]['loss']:.3f}, "
                  + ", ".join(f"{k} {v:.3f}" for k, v in losses.items()), flush=True)
    record["seconds"] = time.time() - start
    record["seconds_per_step"] = record["seconds"] / STEPS
    return model, record


def main():
    device, name = device_name()
    train_texts, val_texts = load("train"), load("val")
    val_batches = validation_batches(val_texts, 8 if SMOKE else 96, device)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for model_name, spec in MODELS.items():
            model, record = train(model_name, spec, seed, device, train_texts, val_batches)
            record["device"] = name
            slug = f"{model_name.lower().replace(', ', '-').replace(' ', '-')}-seed{seed}"
            (OUT / "runs" / f"{slug}.json").write_text(json.dumps(record, indent=1))
            torch.save({"spec": spec, "state": model.state_dict()}, OUT / "checkpoints" / f"{slug}.pt")
            del model
            if device == "mps":
                torch.mps.empty_cache()
            print(f"{model_name} seed {seed}: {record['parameters'] / 1e6:.1f}M parameters, "
                  f"{record['active_parameters'] / 1e6:.1f}M active, {record['seconds_per_step']:.2f} s/step",
                  flush=True)


if __name__ == "__main__":
    main()
