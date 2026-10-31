"""What the routers of the trained mixtures do, read from the checkpoints of train.py, into data/routing.json.

For every mixture-of-experts model:

- the share of the token-expert assignments each expert gets in every layer (f in the balancing loss) and the mean
  probability the router gives it (P);
- the share of each kind of text (English, Russian, Python) that every expert of every layer receives;
- the router's probabilities for a few tokens of held-out text, for the worked example of the post;
- the experts each token of three short held-out snippets goes to, layer by layer;
- what a fixed capacity per expert does: the share of token-expert assignments it drops and the validation loss
  with those tokens skipping the layer, for capacity factors from 1 to 2;
- how many different experts a decoding step needs when it serves B sequences at once: B tokens drawn at random
  from held-out text, the experts their routers chose, counted per layer.

It also collects the training records of train.py into data/train.json.

    uv run python articles/moe/routing.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from moe import Model
from tokenizers import Tokenizer
from train import D_MODEL, DOMAINS, HEADS, LAYERS, LENGTH, MODELS, device_name

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-moe"))
OUT = Path(os.environ.get("OUT", DATA_DIR / "out"))
SEQUENCES = 48  # held-out sequences of each kind of text
FACTORS = [1.0, 1.25, 1.5, 2.0]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SNIPPETS = {  # held-out text from the validation part of each kind
    "en": "At the beginning of the 21st century, the population structure of Andalusia shows a clear inversion",
    "ru": "3 декабря — президент Либерии Чарльз Д. Б. Кинг ушёл в отставку.",
    "code": "tid = threading.get_ident()\ntry:\n    signal.pthread_kill(tid, signum)",
}


def slug(name, seed=0):
    return f"{name.lower().replace(', ', '-').replace(' ', '-')}-seed{seed}"


def load_model(name, device):
    checkpoint = torch.load(OUT / "checkpoints" / f"{slug(name)}.pt", map_location=device)
    model = Model(8192, D_MODEL, LAYERS, HEADS, checkpoint["spec"]).to(device)
    model.load_state_dict(checkpoint["state"])
    return model.eval()


def held_out(device, batch=24):
    """SEQUENCES sequences of each kind, interleaved so that every batch of 24 mixes the three equally."""
    rows, kinds = [], []
    texts = {name: np.fromfile(DATA_DIR / f"{name}_val.bin", dtype=np.uint16) for name in DOMAINS}
    for i in range(SEQUENCES):
        for name in DOMAINS:
            rows.append(texts[name][i * (LENGTH + 1):(i + 1) * (LENGTH + 1)].astype(np.int64))
            kinds.append(name)
    ids = torch.from_numpy(np.stack(rows)).to(device)
    batches = [(ids[i:i + batch, :-1], ids[i:i + batch, 1:]) for i in range(0, len(ids), batch)]
    return batches, kinds


@torch.no_grad()
def decisions(model, batches, kinds):
    """Run the held-out text; return the loss and, per layer, the experts chosen for every token, the kind of text
    of every token and the share of assignments each layer dropped."""
    layers = model.moe_layers()
    chosen = [[] for _ in layers]
    dropped = [[] for _ in layers]
    probs = [0.0 for _ in layers]
    total, count = 0.0, 0
    for x, y in batches:
        logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        count += y.numel()
        for j, layer in enumerate(layers):
            chosen[j].append(layer.chosen.cpu())
            dropped[j].append(layer.dropped)
            probs[j] = probs[j] + layer.probs.sum(dim=0).cpu()
    token_kinds = np.repeat(np.array(kinds), LENGTH)
    mean_probs = [(p / count).tolist() for p in probs]
    return total / count, [torch.cat(c).numpy() for c in chosen], token_kinds, dropped, mean_probs


def domain_shares(chosen, token_kinds, experts):
    out = []
    for layer in chosen:
        shares = {}
        for name in DOMAINS:
            picked = layer[token_kinds == name].flatten()
            shares[name] = (np.bincount(picked, minlength=experts) / len(picked)).tolist()
        out.append(shares)
    return out


def distinct_experts(chosen, experts, top_k, generator, draws=400):
    out = []
    for batch in BATCHES:
        per_layer = []
        for layer in chosen:
            counts = [len(np.unique(layer[generator.choice(len(layer), batch, replace=False)])) for _ in range(draws)]
            per_layer.append(float(np.mean(counts)))
        uniform = experts * (1 - (1 - top_k / experts) ** batch)
        out.append({"batch": batch, "layers": per_layer, "uniform": uniform})
    return out


@torch.no_grad()
def snippets(model, tokenizer, device):
    out = {}
    for name, text in SNIPPETS.items():
        ids = tokenizer.encode(text).ids
        model(torch.tensor([ids], device=device))
        out[name] = {
            "tokens": [tokenizer.decode([i]) for i in ids],
            "chosen": [layer.chosen.cpu().tolist() for layer in model.moe_layers()],  # best first
            "probs": [layer.probs.cpu().tolist() for layer in model.moe_layers()],
        }
    return out


def main():
    device, device_label = device_name()
    tokenizer = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
    batches, kinds = held_out(device)
    result = {"device": device_label, "sequences_per_kind": SEQUENCES, "length": LENGTH, "models": {}}
    for name, spec in MODELS.items():
        if "experts" not in spec or not (OUT / "checkpoints" / f"{slug(name)}.pt").exists():
            continue
        model = load_model(name, device)
        experts, top_k = spec["experts"], spec["top_k"]
        loss, chosen, token_kinds, _, mean_probs = decisions(model, batches, kinds)
        load = [(np.bincount(c.flatten(), minlength=experts) / c.size).tolist() for c in chosen]
        entry = {"spec": spec, "loss": loss, "shares": domain_shares(chosen, token_kinds, experts),
                 "load": load, "mean_probs": mean_probs,
                 "distinct": distinct_experts(chosen, experts, top_k, np.random.default_rng(0)),
                 "snippets": snippets(model, tokenizer, device), "capacity": []}
        tokens_per_batch = 24 * LENGTH
        for factor in FACTORS:
            capacity = math.ceil(factor * tokens_per_batch * top_k / experts)
            for layer in model.moe_layers():
                layer.capacity = capacity
            capped_loss, _, _, dropped, _ = decisions(model, batches, kinds)
            entry["capacity"].append({"factor": factor, "capacity": capacity, "loss": capped_loss,
                                      "dropped": [float(np.mean(d)) for d in dropped]})
            print(f"{name}: capacity factor {factor}: loss {capped_loss:.4f} (dropless {loss:.4f}), dropped "
                  + ", ".join(f"{np.mean(d):.1%}" for d in dropped), flush=True)
        for layer in model.moe_layers():
            layer.capacity = None
        result["models"][name] = entry
    data = Path(__file__).parent / "data"
    data.mkdir(exist_ok=True)
    (data / "routing.json").write_text(json.dumps(result) + "\n")
    # the training records of all runs, for the loss and balance charts
    runs = [json.loads(path.read_text()) for path in sorted((OUT / "runs").glob("*.json"))]
    (data / "train.json").write_text(json.dumps({"runs": runs}) + "\n")


if __name__ == "__main__":
    main()
