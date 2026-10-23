"""Measurements for the post "Adapters and PEFT" on xmarva.github.io, run on a Kaggle T4.

bert-base-uncased is fine-tuned on SST-2 (sentiment of movie-review sentences) eleven ways: the whole model, the task
head alone, and nine parameter-efficient methods, each written out below in plain PyTorch. Every method trains the
same task head (pooler and classifier) plus its own parameters, on the same 10,000 training sentences, for three
epochs, with three seeds; accuracy is measured on the 872 validation sentences.

Also measured: trainable parameters, peak GPU memory and time of training; how far the merged LoRA model is from the
unmerged one; and the singular values of the weight updates that full fine-tuning makes, to see how low their rank is.

Results go to adapters.json.
"""

import json
import math
import os
import subprocess
import sys
import time

if os.environ.get("SMOKE") != "1":  # on Kaggle: the version these classes were written against
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.2"], check=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AttentionInterface, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask  # noqa: E402

transformers.logging.set_verbosity_error()
MODEL = "bert-base-uncased"
SMOKE = os.environ.get("SMOKE") == "1"
TRAIN_SIZE, EPOCHS = (256, 1) if SMOKE else (10_000, 3)
SEEDS = [0] if SMOKE else [0, 1, 2]
BATCH, MAX_LEN = 32, 64
OUT = "adapters-smoke.json" if SMOKE else "/kaggle/working/adapters.json"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- the methods

class Adapter(nn.Module):
    """Houlsby et al. 2019: a bottleneck MLP with a skip connection, starting as the identity."""

    def __init__(self, d, bottleneck):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.zeros_(self.up.weight)  # the adapter adds nothing until training moves it
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


def add_adapters(model, bottleneck=64):
    for layer in model.bert.encoder.layer:
        # after attention and after the feed-forward block, before the residual and the layer norm
        for sub in (layer.attention.output, layer.output):
            sub.dense = nn.Sequential(sub.dense, Adapter(sub.dense.out_features, bottleneck))


class ParallelFeedForward(nn.Module):
    """He et al. 2022: an adapter beside the feed-forward block; it reads the block's input, adds to its output."""

    def __init__(self, output, bottleneck, scale=4.0):
        super().__init__()
        self.output = output  # BERT's: dense, dropout, LayerNorm(h + x)
        d = output.dense.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = scale

    def forward(self, intermediate, x):
        h = self.output.dropout(self.output.dense(intermediate))
        h = h + self.scale * self.up(F.gelu(self.down(x)))
        return self.output.LayerNorm(h + x)


def add_parallel_adapters(model, bottleneck=64):
    for layer in model.bert.encoder.layer:
        layer.output = ParallelFeedForward(layer.output, bottleneck)


class LoRALinear(nn.Module):
    """Hu et al. 2021: y = W x + (alpha / r) B A x, with W frozen and B starting at zero."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:
            y = y + (x @ self.A.T @ self.B.T) * self.scale
        return y

    @torch.no_grad()
    def merge(self):
        """Fold the update into W: after this the layer costs exactly what it cost before LoRA."""
        self.base.weight += self.scale * self.B @ self.A
        self.merged = True


def add_lora(model, r=8, alpha=16, targets=("query", "value")):
    for layer in model.bert.encoder.layer:
        attention = layer.attention.self
        for name in targets:
            setattr(attention, name, LoRALinear(getattr(attention, name), r, alpha))


def prefix_attention(module, query, key, value, attention_mask, **kwargs):
    """Li & Liang 2021: every layer attends to `length` extra keys and values that are free parameters."""
    batch = key.shape[0]
    # (batch, heads, length, head_dim), in the precision of the keys
    prefix_k = module.prefix_key.expand(batch, -1, -1, -1).to(key.dtype)
    prefix_v = module.prefix_value.expand(batch, -1, -1, -1).to(value.dtype)
    key = torch.cat([prefix_k, key], dim=2)
    value = torch.cat([prefix_v, value], dim=2)
    # the mask is boolean, True where attention is allowed: the prefix is always visible
    if attention_mask is not None:
        visible = attention_mask.new_ones(*attention_mask.shape[:-1], prefix_k.shape[2])
        attention_mask = torch.cat([visible, attention_mask], dim=-1)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("prefix", prefix_attention)
AttentionMaskInterface.register("prefix", sdpa_mask)


def add_prefix(model, length=20):
    model.set_attn_implementation("prefix")
    for layer in model.bert.encoder.layer:
        attention = layer.attention.self
        shape = (1, attention.num_attention_heads, length, attention.attention_head_size)
        attention.prefix_key = nn.Parameter(torch.randn(shape) * 0.02)
        attention.prefix_value = nn.Parameter(torch.randn(shape) * 0.02)


class SoftPrompt(nn.Module):
    """Lester et al. 2021: `length` trainable embeddings inserted after [CLS], at the input only."""

    def __init__(self, model, length=20):
        super().__init__()
        self.model = model
        vocabulary = model.get_input_embeddings().weight
        # begin from the embeddings of real words
        start = torch.randint(1000, vocabulary.shape[0], (length,))
        self.prompt = nn.Parameter(vocabulary[start].detach().clone())

    def forward(self, input_ids, attention_mask, labels=None):
        embeds = self.model.get_input_embeddings()(input_ids)
        prompt = self.prompt.expand(input_ids.shape[0], -1, -1)
        embeds = torch.cat([embeds[:, :1], prompt, embeds[:, 1:]], dim=1)
        mask = torch.cat([attention_mask[:, :1].expand(-1, prompt.shape[1]), attention_mask], dim=1)
        return self.model(inputs_embeds=embeds, attention_mask=mask, labels=labels)


class Rescale(nn.Module):
    """Liu et al. 2022, (IA)^3: multiply a module's output by a learned vector that starts at ones."""

    def __init__(self, module, size):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.ones(size))

    def forward(self, x):
        return self.module(x) * self.scale


def add_ia3(model):
    for layer in model.bert.encoder.layer:
        attention = layer.attention.self
        attention.key = Rescale(attention.key, attention.key.out_features)  # l_k
        attention.value = Rescale(attention.value, attention.value.out_features)  # l_v
        hidden = layer.intermediate.dense.out_features
        layer.intermediate = Rescale(layer.intermediate, hidden)  # l_ff, after GELU


def bitfit(model):
    """Ben Zaken et al. 2022: train the bias terms and nothing else (the head is trained for every method)."""
    for name, p in model.bert.named_parameters():
        if name.endswith("bias"):
            p.requires_grad = True


class PHMLinear(nn.Module):
    """Karimi Mahabadi et al. 2021, Compacter: W = sum_i A_i ⊗ B_i, where the small A_i are shared by every layer of
    the model and each B_i = s_i t_i^T has rank one."""

    def __init__(self, shared, d_in, d_out, zero=False):
        super().__init__()
        n = shared.shape[0]
        self.shared = shared  # (n, n, n): A_1 ... A_n
        if zero:
            self.s = nn.Parameter(torch.zeros(n, d_out // n, 1))
        else:
            self.s = nn.Parameter(torch.randn(n, d_out // n, 1) * 0.1)
        self.t = nn.Parameter(torch.randn(n, 1, d_in // n) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def weight(self):
        B = self.s @ self.t  # (n, d_out / n, d_in / n), each of rank one
        return sum(torch.kron(self.shared[i], B[i]) for i in range(len(B)))

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class CompacterAdapter(nn.Module):
    def __init__(self, shared, d, bottleneck):
        super().__init__()
        self.down = PHMLinear(shared, d, bottleneck)
        self.up = PHMLinear(shared, bottleneck, d, zero=True)  # starts as the identity, like the adapter

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


def add_compacter(model, bottleneck=48, n=4):
    model.compacter_shared = nn.Parameter(torch.randn(n, n, n) * 0.5)
    for layer in model.bert.encoder.layer:
        for sub in (layer.attention.output, layer.output):
            adapter = CompacterAdapter(model.compacter_shared, sub.dense.out_features, bottleneck)
            sub.dense = nn.Sequential(sub.dense, adapter)


def nothing(model):
    pass


METHODS = {  # name: (what to add, learning rate)
    "full fine-tuning": (None, 2e-5),
    "task head only": (nothing, 1e-3),
    "BitFit": (bitfit, 1e-3),
    "(IA)³": (add_ia3, 3e-3),
    "prompt tuning": (SoftPrompt, 3e-2),
    "prefix tuning": (add_prefix, 5e-3),
    "LoRA, q and v, r = 8": (add_lora, 5e-4),
    "Compacter": (add_compacter, 3e-3),
    "adapters, sequential": (add_adapters, 5e-4),
    "adapters, parallel": (add_parallel_adapters, 5e-4),
}


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


def build(method, seed):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, attn_implementation="sdpa")
    add, lr = METHODS[method]
    if add is not None:
        for p in model.parameters():
            p.requires_grad = False
        for p in [*model.bert.pooler.parameters(), *model.classifier.parameters()]:
            p.requires_grad = True  # every method trains the task head
        made = add(model)
        if isinstance(made, nn.Module):
            model = made
    return model.to(device), lr


def count(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = sum(p.numel() for n, p in trainable if ".pooler." in f".{n}" or "classifier" in n)
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for _, p in trainable), head, total


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


def train(method, seed):
    model, lr = build(method, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if method != "full fine-tuning" else 0.01)
    steps = EPOCHS * math.ceil(TRAIN_SIZE / BATCH)
    warmup = int(0.06 * steps)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(warmup, 1), max(0.0, (steps - s) / max(steps - warmup, 1)))
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    started, per_epoch = time.perf_counter(), []
    for epoch in range(EPOCHS):
        for ids, mask, labels in batches(train_data, shuffle=True, seed=seed * 100 + epoch):
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
        per_epoch.append(accuracy(model))
    seconds = time.perf_counter() - started
    trainable, head, total = count(model)
    memory = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return model, {"method": method, "seed": seed, "accuracy": per_epoch, "seconds": seconds, "trainable": trainable,
                   "head": head, "total": total, "peak_memory": memory, "lr": lr}


# ---------------------------------------------------------------- the runs

results = {"model": MODEL, "train_size": TRAIN_SIZE, "epochs": EPOCHS, "batch": BATCH, "max_len": MAX_LEN,
           "valid_size": len(valid_rows), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "runs": []}
for seed in SEEDS:
    for method in METHODS:
        model, run = train(method, seed)
        results["runs"].append(run)
        print(f"{method:24s} seed {seed}: accuracy {run['accuracy']}, trainable {run['trainable']:,}, "
              f"{run['seconds']:.0f} s, {run['peak_memory'] / 2**30:.2f} GiB", flush=True)

        if seed == 0 and method == "LoRA, q and v, r = 8":  # merging must not change what the model says
            ids, mask, _ = next(batches(valid_data, shuffle=False))
            model.eval()
            with torch.no_grad():
                before = model(input_ids=ids, attention_mask=mask).logits.float()
                for module in model.modules():
                    if isinstance(module, LoRALinear):
                        module.merge()
                after = model(input_ids=ids, attention_mask=mask).logits.float()
            results["lora_merge_max_diff"] = (before - after).abs().max().item()

        if seed == 0 and method == "full fine-tuning":  # how low is the rank of a real fine-tuning update?
            pretrained = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
            spectra = {}
            for i, (tuned, base) in enumerate(zip(model.bert.encoder.layer, pretrained.bert.encoder.layer,
                                                  strict=True)):
                for name in ("query", "value"):
                    delta = (getattr(tuned.attention.self, name).weight - getattr(base.attention.self, name).weight)
                    spectra[f"{i}.{name}"] = torch.linalg.svdvals(delta.float()).tolist()
            torch.manual_seed(0)
            noise = torch.randn(768, 768, device=device)
            spectra["random"] = torch.linalg.svdvals(noise).tolist()
            results["delta_spectra"] = spectra
            del pretrained
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("saved", OUT)
