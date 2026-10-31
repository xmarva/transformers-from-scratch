"""How long a training step takes: the dense model against the mixtures, with two ways of running the experts.

The models are those of train.py. A mixture runs its experts either in a loop, one small product per expert on the
tokens that chose it (the code shown in the post), or as moe.py does it, sorted by expert and padded into one batched
product. Both compute the same thing; the time per step of forward, backward and optimizer is the median of 20 steps
after 5 warm-up steps, on random tokens, written to data/speed.json.

    uv run python articles/moe/speed.py   # a few minutes on the GPU it finds
"""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from moe import Model, MoE
from train import BATCH, D_MODEL, HEADS, LAYERS, LENGTH, MODELS, device_name


def loop_forward(self, x):
    """MoE.forward written as a loop over experts, as in the post."""
    tokens = x.reshape(-1, x.shape[-1])
    chosen, weights, probs = self.route(tokens)
    out = torch.zeros_like(tokens)
    for e in range(self.experts_count):
        rows, slot = torch.nonzero(chosen == e, as_tuple=True)  # the tokens that chose expert e
        if rows.numel():
            out.index_add_(0, rows, weights[rows, slot, None].to(tokens.dtype) * self.expert(e, tokens[rows]))
    self.load = (torch.bincount(chosen.flatten(), minlength=self.experts_count) / chosen.numel()).detach()
    self.aux_loss = self.experts_count * (self.load * probs.mean(dim=0)).sum()
    return out.reshape(x.shape)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def step_time(spec, device, steps=20, warmup=5):
    torch.manual_seed(0)
    model = Model(8192, D_MODEL, LAYERS, HEADS, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 8192, (BATCH, LENGTH + 1), device=device)
    times = []
    for i in range(warmup + steps):
        sync(device)
        start = time.perf_counter()
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
        if model.moe_layers():
            loss = loss + 0.01 * sum(layer.aux_loss for layer in model.moe_layers())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    device, name = device_name()
    result = {"device": name, "batch": BATCH, "length": LENGTH, "steps": []}
    for model_name in ("Dense", "Dense, 4x wider", "MoE, 8 experts", "MoE, 32 experts"):
        spec = MODELS[model_name]
        variants = ["batched", "loop"] if "experts" in spec else ["dense"]
        for variant in variants:
            original = MoE.forward
            if variant == "loop":
                MoE.forward = loop_forward
            try:
                seconds = step_time(spec, device)
            finally:
                MoE.forward = original
            result["steps"].append({"model": model_name, "dispatch": variant, "seconds": seconds})
            print(f"{model_name} ({variant}): {seconds * 1e3:.0f} ms per step", flush=True)
            if device == "mps":
                torch.mps.empty_cache()
    (Path(__file__).parent / "data" / "speed.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
