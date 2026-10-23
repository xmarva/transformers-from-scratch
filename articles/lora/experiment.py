"""Measurements for the post "Algebraic Foundations of Low-Rank Adaptation" on xmarva.github.io, run on a Kaggle T4.

bert-base-uncased is fine-tuned on SST-2 (sentiment of movie-review sentences) with LoRA on the query and value
projections of attention, in the setting of the adapters post: 10,000 training sentences, three epochs, AdamW with a
learning rate of 5e-4, batches of 32, three seeds. What changes from run to run is the algebra of the update:

    the rank, from 1 to 256, with the scale alpha / r of the LoRA paper (alpha = 16);
    the same ranks with rsLoRA's scale alpha' / sqrt(r), alpha' chosen so that the two scales agree at r = 8;
    LoRA+: the learning rate of B set 4 and 16 times that of A;
    one factor trained and the other frozen: only B (A stays the random matrix it started as), or only A;
    and full fine-tuning, for reference.

Every ten steps a run logs its training loss and the size of its update, ||W - W_0||_F, averaged over the 24 adapted
matrices. After training the script measures what Hu et al. measured in section 7 of the LoRA paper: how much the
subspaces learned at r = 8 and r = 64, and by two seeds, overlap; and how strongly the update amplifies directions
that W already has. It also keeps the singular values of every update. (Runs with the same seed draw A from the same
random stream, so in layer 1's query projection, the first one adapted, the rank-8 and rank-64 runs start with the
same first 8 rows of A; in the other 23 matrices their starts are unrelated.)

Results go to lora.json.
"""

import json
import math
import os
import subprocess
import sys
import time

if os.environ.get("SMOKE") != "1":  # on Kaggle: the version these classes were written against
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.2"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

transformers.logging.set_verbosity_error()
MODEL = "bert-base-uncased"
SMOKE = os.environ.get("SMOKE") == "1"
TRAIN_SIZE, EPOCHS = (256, 1) if SMOKE else (10_000, 3)
SEEDS = [0, 1] if SMOKE else [0, 1, 2]
RANKS = [1, 8, 64] if SMOKE else [1, 4, 8, 16, 64, 256]
BATCH, MAX_LEN = 32, 64
LOG_EVERY = 2 if SMOKE else 10
LR, LR_FULL, ALPHA = 5e-4, 2e-5, 16
ALPHA_RS = ALPHA / math.sqrt(8)  # alpha' / sqrt(r) = alpha / r at r = 8, so rsLoRA's r = 8 run is LoRA's
OUT = "lora-smoke.json" if SMOKE else "/kaggle/working/lora.json"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- LoRA, with the knobs the runs turn

class LoRALinear(nn.Module):
    """y = W x + scale * B A x, with W frozen. `train` names the factors that learn; the other keeps its start."""

    def __init__(self, base, r, scale, train="AB"):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features), requires_grad="A" in train)
        self.B = nn.Parameter(torch.zeros(base.out_features, r), requires_grad="B" in train)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # U(-1/sqrt(d_in), 1/sqrt(d_in)), as in peft
        if train == "A":  # B random and frozen, A starts at zero: the update still starts at zero
            bound = 1 / math.sqrt(base.in_features)
            nn.init.uniform_(self.B, -bound, bound)
            nn.init.zeros_(self.A)
        self.register_buffer("A0", self.A.detach().clone(), persistent=False)
        self.scale = scale

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta(self):
        return self.scale * self.B @ self.A


def projections(model):
    """The 24 matrices every run adapts: query and value of the 12 layers, by name."""
    for i, layer in enumerate(model.bert.encoder.layer):
        for name in ("query", "value"):
            yield f"{i}.{name}", layer.attention.self, name


def add_lora(model, r, scale, train="AB"):
    for _, attention, name in projections(model):
        setattr(attention, name, LoRALinear(getattr(attention, name), r, scale, train))


def runs():
    """Seed by seed, so that if the session ends early the first seed is complete."""
    for seed in SEEDS:
        yield {"group": "full fine-tuning", "seed": seed}
        for r in RANKS:
            yield {"group": "LoRA", "r": r, "scale": ALPHA / r, "seed": seed}
        for r in RANKS:
            if r != 8:
                yield {"group": "rsLoRA", "r": r, "scale": ALPHA_RS / math.sqrt(r), "seed": seed}
        for ratio in (4, 16):
            yield {"group": "LoRA+", "r": 8, "scale": ALPHA / 8, "ratio": ratio, "seed": seed}
        for train in ("B", "A"):
            yield {"group": f"only {train}", "r": 8, "scale": ALPHA / 8, "train": train, "seed": seed}


# ---------------------------------------------------------------- data and training

tokenizer = AutoTokenizer.from_pretrained(MODEL)
sst2 = load_dataset("nyu-mll/glue", "sst2")
train_rows = sst2["train"].shuffle(seed=0).select(range(TRAIN_SIZE))
valid_rows = sst2["validation"].select(range(64)) if SMOKE else sst2["validation"]


def encode(rows):
    enc = tokenizer(list(rows["sentence"]), truncation=True, max_length=MAX_LEN)
    return [(ids, label) for ids, label in zip(enc["input_ids"], list(rows["label"]), strict=True)]


train_data, valid_data = encode(train_rows), encode(valid_rows)


def batches(data, shuffle, seed=0):
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed)).tolist() if shuffle \
        else list(range(len(data)))
    for start in range(0, len(order), BATCH):
        picked = [data[i] for i in order[start:start + BATCH]]
        padded = tokenizer.pad({"input_ids": [ids for ids, _ in picked]}, return_tensors="pt")
        labels = torch.tensor([label for _, label in picked])
        yield padded["input_ids"].to(device), padded["attention_mask"].to(device), labels.to(device)


def build(run):
    torch.manual_seed(run["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, attn_implementation="sdpa")
    if run["group"] != "full fine-tuning":
        for p in model.parameters():
            p.requires_grad = False
        for p in [*model.bert.pooler.parameters(), *model.classifier.parameters()]:
            p.requires_grad = True  # every run trains the task head
        add_lora(model, run["r"], run["scale"], run.get("train", "AB"))
    return model.to(device)


def param_groups(model, run):
    if run["group"] == "full fine-tuning":
        return [{"params": list(model.parameters()), "lr": LR_FULL, "weight_decay": 0.01}]
    b = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".B")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".B")]
    # LoRA+ (Hayou et al. 2024): B learns `ratio` times faster than A
    groups = [{"params": rest, "lr": LR}, {"params": b, "lr": LR * run.get("ratio", 1)}]
    return [g for g in groups if g["params"]]


class Updates:
    """W - W_0 for the 24 adapted matrices, whatever the method."""

    def __init__(self, model, full):
        self.full = full
        self.modules = {key: getattr(attention, name) for key, attention, name in projections(model)}
        if full:
            self.start = {key: m.weight.detach().clone() for key, m in self.modules.items()}

    @torch.no_grad()
    def get(self):
        if self.full:
            return {key: (m.weight - self.start[key]).float() for key, m in self.modules.items()}
        return {key: m.delta().float() for key, m in self.modules.items()}

    @torch.no_grad()
    def mean_norm(self):
        return float(np.mean([torch.linalg.matrix_norm(d).item() for d in self.get().values()]))

    @torch.no_grad()
    def drift(self):
        """How far A has moved from its random start, relative to that start: ||A - A_0|| / ||A_0||."""
        if self.full or not self.modules["0.query"].A0.any():
            return None
        return float(np.mean([(torch.linalg.matrix_norm(m.A - m.A0) / torch.linalg.matrix_norm(m.A0)).item()
                              for m in self.modules.values()]))


@torch.no_grad()
def accuracy(model):
    model.eval()
    right = 0
    for ids, mask, labels in batches(valid_data, shuffle=False):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        right += (logits.argmax(-1) == labels).sum().item()
    model.train()
    return right / len(valid_rows)


def train(run):
    model = build(run)
    groups = param_groups(model, run)
    params = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = EPOCHS * math.ceil(TRAIN_SIZE / BATCH)
    warmup = int(0.06 * steps)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(warmup, 1), max(0.0, (steps - s) / max(steps - warmup, 1)))
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    updates = Updates(model, run["group"] == "full fine-tuning")
    model.train()
    started, per_epoch, step, losses = time.perf_counter(), [], 0, []
    log = {"step": [], "loss": [], "update": [], "drift": [], "grad": []}
    for epoch in range(EPOCHS):
        for ids, mask, labels in batches(train_data, shuffle=True, seed=run["seed"] * 100 + epoch):
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            losses.append(loss.item())
            if step % LOG_EVERY == 0:
                log["step"].append(step)
                log["loss"].append(float(np.mean(losses[-LOG_EVERY:])))
                log["update"].append(updates.mean_norm())
                log["drift"].append(updates.drift())
                log["grad"].append(grad)
        per_epoch.append(accuracy(model))
    trainable = sum(p.numel() for p in params)
    record = {**run, "accuracy": per_epoch, "seconds": time.perf_counter() - started, "trainable": trainable,
              "log": log}
    return model, updates, record


# ---------------------------------------------------------------- what Hu et al. measured in section 7

def right_singular(m):
    """The input directions of a matrix, strongest first: the columns of V in m = U S V^T."""
    return torch.linalg.svd(m.double(), full_matrices=False)[2].T


def similarity(u1, u2):
    """Hu et al., eq. 4: phi(i, j) = ||U1[:, :i]^T U2[:, :j]||_F^2 / min(i, j), 1 if one subspace holds the other."""
    overlap = np.square((u1.T @ u2).cpu().numpy())  # (i, j): squared cosines between single directions
    cumulative = overlap.cumsum(0).cumsum(1)
    i, j = np.indices(cumulative.shape) + 1
    return cumulative / np.minimum(i, j)


def amplification(delta, w, r, generator):
    """Hu et al., section 7.3: project W onto the top-r subspace of the update, ||U^T W V||_F, and compare."""
    u, s, vt = torch.linalg.svd(delta.double(), full_matrices=False)
    u, s, v = u[:, :r], s[:r], vt[:r].T
    w = w.double()
    random_u = torch.linalg.qr(torch.randn(w.shape[0], r, generator=generator, dtype=torch.float64))[0]
    random_v = torch.linalg.qr(torch.randn(w.shape[1], r, generator=generator, dtype=torch.float64))[0]
    w_s = torch.linalg.svdvals(w)
    return {
        "update": torch.linalg.vector_norm(s).item(),  # ||Delta W_r||_F
        "W_in_update": torch.linalg.matrix_norm(u.T @ w @ v).item(),  # ||U^T W V||_F
        "W_top": torch.linalg.vector_norm(w_s[:r]).item(),  # the same for W's own top-r directions
        "W_random": torch.linalg.matrix_norm(random_u.T @ w @ random_v).item(),
        "W": torch.linalg.matrix_norm(w).item(),
    }


def rounded(values, digits=6):
    return [float(f"{v:.{digits}g}") for v in values]


# ---------------------------------------------------------------- the runs

results = {"model": MODEL, "train_size": TRAIN_SIZE, "epochs": EPOCHS, "batch": BATCH, "max_len": MAX_LEN,
           "lr": LR, "lr_full": LR_FULL, "alpha": ALPHA, "alpha_rs": ALPHA_RS, "log_every": LOG_EVERY,
           "valid_size": len(valid_rows), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "runs": []}
kept = {}  # the updates the analyses below need
for run in runs():
    model, updates, record = train(run)
    results["runs"].append(record)
    print(f"{record['group']:16s} r {record.get('r', '-')!s:>3} seed {record['seed']}: accuracy {record['accuracy']}, "
          f"update {record['log']['update'][-1]:.3f}, {record['seconds']:.0f} s", flush=True)
    key = (record["group"], record.get("r"), record["seed"])
    if key in {("full fine-tuning", None, 0), ("LoRA", 8, 0), ("LoRA", 64, 0), ("LoRA", 64, 1)}:
        kept[key] = {name: d.cpu() for name, d in updates.get().items()}
        if record["group"] == "LoRA":
            kept[key, "A"] = {name: m.A.detach().float().cpu() for name, m in updates.modules.items()}
            kept[key, "B"] = {name: (m.scale * m.B).detach().float().cpu() for name, m in updates.modules.items()}
    del model, updates
    if device == "cuda":
        torch.cuda.empty_cache()

pretrained = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
W = {key: getattr(attention, name).weight.detach() for key, attention, name in projections(pretrained)}
generator = torch.Generator().manual_seed(0)
full, r8, r64, r64b = (kept[k] for k in [("full fine-tuning", None, 0), ("LoRA", 8, 0), ("LoRA", 64, 0),
                                         ("LoRA", 64, 1)])

analysis = {"similarity": {}, "amplification": {}, "spectra": {}, "full_in_lora": {}}
for key, w in W.items():
    a8, a64, a64b = (kept[k, "A"][key] for k in [("LoRA", 8, 0), ("LoRA", 64, 0), ("LoRA", 64, 1)])
    analysis["similarity"][key] = {
        # Hu et al., figure 3: the right-singular vectors of A, rank 8 against rank 64 (8 x 64)
        "A_r8_r64": similarity(right_singular(a8), right_singular(a64)).round(5).tolist(),
        # the same for the updates themselves, ordered by how much each direction is amplified
        "dW_r8_r64": similarity(right_singular(r8[key])[:, :8], right_singular(r64[key])[:, :64]).round(5).tolist(),
        # figure 4: rank 64, two seeds; the diagonal phi(i, i). Against random directions phi(i, i) = i / 768 on
        # average, by symmetry, so no noise matrix is needed (one drawn here would share the seeds' random stream)
        "A_seeds_r64": np.diag(similarity(right_singular(a64), right_singular(a64b))).round(5).tolist(),
    }
    analysis["amplification"][key] = {
        "LoRA r = 8": amplification(r8[key], w, 8, generator),
        "LoRA r = 64": amplification(r64[key], w, 64, generator),
        "full, top 8": amplification(full[key], w, 8, generator),
        "full, top 64": amplification(full[key], w, 64, generator),
    }
    analysis["spectra"][key] = {
        "W": rounded(torch.linalg.svdvals(w.double()).tolist()),
        "full": rounded(torch.linalg.svdvals(full[key].double()).tolist()),
        "LoRA r = 8": rounded(torch.linalg.svdvals(r8[key].double()).tolist()),
        "LoRA r = 64": rounded(torch.linalg.svdvals(r64[key].double()).tolist()),
    }
    # how much of what full fine-tuning changed lies in the subspaces LoRA (r = 8) chose
    u, _, vt = torch.linalg.svd(r8[key].double(), full_matrices=False)
    inside = torch.linalg.matrix_norm(u[:, :8].T @ full[key].double() @ vt[:8].T)
    analysis["full_in_lora"][key] = (inside / torch.linalg.matrix_norm(full[key].double())).item()
results["analysis"] = analysis

with open(OUT, "w") as f:
    json.dump(results, f)
arrays = {f"full|{name}": d for name, d in full.items()}  # for a closer look later; not in the post
for k, factors in kept.items():
    if k[-1] in ("A", "B"):
        group, r, seed = k[0]
        arrays.update({f"{group} r={r} seed={seed} {k[-1]}|{name}": m for name, m in factors.items()})
np.savez_compressed(OUT.replace(".json", "-updates.npz"), **{k: v.half().numpy() for k, v in arrays.items()})
print("saved", OUT)
