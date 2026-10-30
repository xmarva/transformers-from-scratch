"""Multi-query associative recall (Arora et al., 2023), the test the post uses to compare memories.

A sequence lists N key-value pairs, then asks for the values of all N keys in a shuffled order:

    k1 v1 k2 v2 ... kN vN   q1 a1 q2 a2 ... qN aN

and the model must put the right value a_i after each query q_i. Six models of the same width and depth, four
layers each, differ only in how their layers mix tokens. Each is trained on fresh random sequences and scored on
held-out ones.

The whole grid, N from 4 to 128 and two learning rates, takes about five hours on a T4. The published data/mqar.json
comes from two runs that together cover less, and every run in it records its device. The first, on a Kaggle T4, ran
the whole grid for the Transformer, linear attention and DeltaNet. The second, on an Apple M2 through PyTorch's MPS
backend, ran the models with Gated DeltaNet layers up to 64 pairs (at 128 not even the Transformer learned the task in
4,000 steps), at the rate that was better for every recurrent model in the first run, and at both rates for the
hybrid with attention on top, the one that struggled. MODELS (separated by semicolons), PAIRS and RATES pick a part of
the grid:

    uv run --group articles python articles/hybrid_attention/mqar.py            # the whole grid
    MODELS="Gated DeltaNet;Hybrid (attention 2nd)" PAIRS=4,8,16,32,64 RATES=3e-3 \
        uv run --group articles python articles/hybrid_attention/mqar.py        # part of the second run
    SMOKE=1 uv run --group articles python articles/hybrid_attention/mqar.py    # a few steps, to check it runs
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mixers import Model

SMOKE = os.environ.get("SMOKE") == "1"
KEYS, VALUES = 512, 512  # token ids: 0 unused, 1..512 keys, 513..1024 values
VOCAB = 1 + KEYS + VALUES
MODELS = {
    "Transformer": ["softmax"] * 4,
    "Linear attention": ["linear"] * 4,
    "DeltaNet": ["delta"] * 4,
    "Gated DeltaNet": ["gated delta"] * 4,
    # two hybrids with the same 3:1 ratio, the attention layer in different places
    "Hybrid (attention 2nd)": ["gated delta", "softmax", "gated delta", "gated delta"],
    "Hybrid (attention on top)": ["gated delta"] * 3 + ["softmax"],
}
PAIRS = [4, 8] if SMOKE else [4, 8, 16, 32, 64, 128]
RATES = [1e-3] if SMOKE else [1e-3, 3e-3]
if os.environ.get("MODELS"):
    MODELS = {name: MODELS[name] for name in os.environ["MODELS"].split(";")}
if os.environ.get("PAIRS"):
    PAIRS = [int(n) for n in os.environ["PAIRS"].split(",")]
if os.environ.get("RATES"):
    RATES = [float(r) for r in os.environ["RATES"].split(",")]
STEPS = 20 if SMOKE else 4000
BATCH = 64
D_MODEL, HEADS = 64, 2


def batch_of(pairs, batch, generator, device):
    keys = torch.rand(batch, KEYS, generator=generator).argsort(-1)[:, :pairs] + 1  # distinct keys
    values = torch.randint(KEYS + 1, VOCAB, (batch, pairs), generator=generator)
    order = torch.rand(batch, pairs, generator=generator).argsort(-1)
    context = torch.stack([keys, values], dim=-1).reshape(batch, 2 * pairs)
    asked, answers = keys.gather(1, order), values.gather(1, order)
    questions = torch.stack([asked, answers], dim=-1).reshape(batch, 2 * pairs)
    ids = torch.cat([context, questions], dim=1)
    targets = torch.full_like(ids, -100)  # no loss except where a value must be recalled
    targets[:, 2 * pairs::2] = answers
    return ids.to(device), targets.to(device)


def train(kinds, pairs, rate, device, seed=0):
    torch.manual_seed(seed)
    model = Model(kinds, VOCAB, D_MODEL, HEADS).to(device)
    # the forget gate's own parameters stay out of weight decay, as in Mamba-2 and Gated DeltaNet: decay would pull
    # the gate back towards a short memory
    gate = [p for n, p in model.named_parameters() if n.endswith(("A_log", "dt_bias"))]
    rest = [p for n, p in model.named_parameters() if not n.endswith(("A_log", "dt_bias"))]
    optimizer = torch.optim.AdamW([{"params": rest}, {"params": gate, "weight_decay": 0.0}], lr=rate,
                                  weight_decay=0.1)
    warmup = STEPS // 20

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (STEPS - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        ids, targets = batch_of(pairs, BATCH, generator, device)
        loss = F.cross_entropy(model(ids).flatten(0, 1), targets.flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    correct = total = 0
    held_out = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for _ in range(8):
            ids, targets = batch_of(pairs, BATCH, held_out, device)
            predicted = model(ids).argmax(-1)
            mask = targets != -100
            correct += (predicted[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
    return correct / total


def main():
    if torch.cuda.is_available():
        device, device_name = "cuda", torch.cuda.get_device_name()
    elif torch.backends.mps.is_available():
        device, device_name = "mps", "Apple GPU (MPS)"
    else:
        device, device_name = "cpu", "cpu"
    out = Path(os.environ.get("OUT", Path(__file__).parent / "data"))
    out.mkdir(parents=True, exist_ok=True)
    results = {"device": device_name, "steps": STEPS,
               "batch": BATCH, "d_model": D_MODEL, "heads": HEADS, "runs": []}
    for name, kinds in MODELS.items():
        for pairs in PAIRS:
            for rate in RATES:
                start = time.time()
                accuracy = train(kinds, pairs, rate, device)
                results["runs"].append({"model": name, "layers": kinds, "pairs": pairs, "rate": rate,
                                        "accuracy": accuracy, "seconds": time.time() - start})
                print(f"{name:18} N={pairs:4d} lr={rate:g}: {accuracy:.3f} ({time.time() - start:.0f} s)",
                      flush=True)
                (out / "mqar.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
