"""Measurements for the post "LLM inference: where the time and the memory go", run on a Kaggle T4.

The model is Qwen2.5-1.5B-Instruct in float16. A decoding step does two kinds of work, and we time them apart:

    weights    every weight matrix of the model applied to one new token per sequence (q, k, v, o, the MLP, the head)
    attention  the new token of every sequence attending to its KV cache, layer by layer

Both run as CUDA graphs: the GPU replays the recorded kernels with no Python in between, so the time is the GPU's
work alone. A serving engine does the same for its decoding steps; transformers run eagerly does not, and we time
that too, to see what the difference costs.

    1. weights: one step for batches of 1 to 2048 sequences
    2. attention: one step for 16 sequences, with a cache of 512 to 16384 tokens each, fused and not
    3. what nvidia-smi reports while the GPU replays the batch-1 and the batch-1024 step
    4. transformers, eagerly: one decoding step at batch 1 and 8, a context of 512
    5. prefill: the time to read a prompt of 128 to 8192 tokens
    6. generating 1024 tokens with and without the KV cache, the time of every step

Results go to t4.json.
"""

import json
import math
import os
import subprocess
import threading
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE = os.environ.get("SMOKE") == "1"  # a quick run on the CPU, to check the script before sending it to Kaggle
OUT = "t4-smoke.json" if SMOKE else "/kaggle/working/t4.json"
device = "cpu" if SMOKE else "cuda"
dtype = torch.float32 if SMOKE else torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa").to(device, dtype).eval()
config = model.config
linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
weight_params = sum(m.weight.numel() for m in linears)
torch.manual_seed(0)
results = {
    "model": MODEL,
    "gpu": "cpu" if SMOKE else torch.cuda.get_device_name(0),
    "params": sum(p.numel() for p in model.parameters()),
    "matmul_params": weight_params,  # the head shares its matrix with the embedding; counted once, as it is read once
    "matmul_bytes": weight_params * dtype.itemsize,
    "layers": config.num_hidden_layers,
    "heads": config.num_attention_heads,
    "kv_heads": config.num_key_value_heads,
    "head_dim": config.hidden_size // config.num_attention_heads,
}


def sync():
    if not SMOKE:
        torch.cuda.synchronize()


def release():
    if not SMOKE:
        torch.cuda.empty_cache()


class Step:
    """A piece of GPU work recorded once as a CUDA graph and then replayed; on the CPU, simply called."""

    def __init__(self, work):
        self.work = work
        if SMOKE:
            return
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # the first calls pick kernels and allocate; they must not be recorded
            for _ in range(3):
                work()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            work()

    def run(self):
        if SMOKE:
            self.work()
        else:
            self.graph.replay()

    def seconds(self, repeats=20):
        self.run()
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            self.run()
        sync()
        return (time.perf_counter() - start) / repeats


@torch.no_grad()
def weights_step(batch):
    inputs = {n: torch.randn(batch, n, device=device, dtype=dtype) for n in {m.in_features for m in linears}}
    return Step(lambda: [m(inputs[m.in_features]) for m in linears])


@torch.no_grad()
def attention_step(batch, context, fused=True):
    """One query per sequence against a cache of `context` keys and values in every layer, as grouped-query attention
    computes it: each of the kv_heads keys serves heads // kv_heads query heads. The query heads of a group go in
    as `group` queries of one head, so that every key and value is read once, the way decoding kernels do it.
    fused: one SDPA kernel per layer; otherwise two matrix products and a softmax."""
    L, kv, d = results["layers"], results["kv_heads"], results["head_dim"]
    group = results["heads"] // kv
    keys = torch.randn(L, batch, kv, context, d, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    query = torch.randn(batch, kv, group, d, device=device, dtype=dtype)

    def work():
        for layer in range(L):
            if fused:
                F.scaled_dot_product_attention(query, keys[layer], values[layer])
            else:
                scores = query @ keys[layer].transpose(-1, -2) / math.sqrt(d)
                scores.softmax(-1) @ values[layer]

    return Step(work), 2 * keys.numel() * dtype.itemsize


# 1. the weights: as many sequences as we like for the price of reading the matrices once, up to a point
results["weights"] = []
for batch in [1, 2, 4] if SMOKE else [2**k for k in range(12)]:
    seconds = weights_step(batch).seconds()
    release()
    results["weights"].append({"batch": batch, "seconds": seconds})
    print(f"weights, batch {batch:5d}: {seconds * 1e3:7.2f} ms, "
          f"{results['matmul_bytes'] / seconds / 1e9:6.0f} GB/s, "
          f"{2 * batch * weight_params / seconds / 1e12:6.1f} TFLOP/s", flush=True)

# 2. the KV cache: the longer the context, the more there is to read for every new token
results["attention"] = []
batch = 16
for context in [64, 128] if SMOKE else [512, 1024, 2048, 4096, 8192, 16384]:
    for fused in [True, False]:
        step, cache_bytes = attention_step(batch, context, fused)
        seconds = step.seconds()
        del step
        release()
        results["attention"].append(
            {"batch": batch, "context": context, "fused": fused, "seconds": seconds, "cache_bytes": cache_bytes}
        )
        print(f"attention, {'fused' if fused else 'matmul'}, batch {batch}, context {context:5d}: "
              f"{seconds * 1e3:7.2f} ms, cache {cache_bytes / 2**30:5.2f} GiB, "
              f"{cache_bytes / seconds / 1e9:6.0f} GB/s", flush=True)


# 3. GPU utilization, as nvidia-smi reports it
class Monitor:
    """Samples nvidia-smi every 100 ms while active."""

    FIELDS = ["utilization.gpu", "utilization.memory", "power.draw", "clocks.sm"]

    def __init__(self):
        self.samples, self.running = [], False

    def _loop(self):
        while self.running:
            line = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True,
            ).stdout.strip()
            try:
                self.samples.append(dict(zip(self.FIELDS, map(float, line.split(", ")), strict=True)))
            except ValueError:
                pass
            time.sleep(0.1)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()


if not SMOKE:
    results["monitor"] = []
    for batch in [1, 1024]:
        step = weights_step(batch)
        seconds = step.seconds()
        repeats = max(1, int(5 / seconds))  # five seconds of work
        with Monitor() as monitor:
            for _ in range(repeats):
                step.run()
            sync()
        del step
        release()
        samples = monitor.samples[3:]  # the counters lag behind the start of the work
        mean = {f: sum(s[f] for s in samples) / len(samples) for f in Monitor.FIELDS}
        results["monitor"].append({"batch": batch, "seconds": seconds, "samples": samples, "mean": mean})
        print(f"nvidia-smi, batch {batch}:", {f: round(v) for f, v in mean.items()},
              f"achieved {2 * batch * weight_params / seconds / 1e12:.2f} TFLOP/s", flush=True)


def random_ids(batch, length):
    return torch.randint(100, config.vocab_size - 100, (batch, length), device=device)


# 4. the same decoding step, run by transformers eagerly: Python launches the kernels one by one
@torch.no_grad()
def eager_decode_seconds(batch, context, steps=20):
    out = model(random_ids(batch, context), use_cache=True, logits_to_keep=1)
    cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    times = []
    for _ in range(steps):
        sync()
        start = time.perf_counter()
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
        sync()
        times.append(time.perf_counter() - start)
        cache = out.past_key_values
    return sorted(times)[len(times) // 2]


results["eager"] = []
context = 32 if SMOKE else 512
for batch in [1, 8]:
    seconds = eager_decode_seconds(batch, context, 3 if SMOKE else 20)
    results["eager"].append({"batch": batch, "context": context, "seconds": seconds})
    print(f"transformers, eager, batch {batch}: {seconds * 1e3:7.1f} ms per step", flush=True)
release()


# 5. prefill: the whole prompt in one pass
@torch.no_grad()
def prefill_seconds(length, repeats=5):
    ids = random_ids(1, length)
    model(ids, use_cache=True, logits_to_keep=1)
    times = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        model(ids, use_cache=True, logits_to_keep=1)
        sync()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


results["prefill"] = []
for length in [16, 32] if SMOKE else [128, 256, 512, 1024, 2048, 4096, 8192]:
    seconds = prefill_seconds(length)
    results["prefill"].append({"tokens": length, "seconds": seconds})
    print(f"prefill, {length:5d} tokens: {seconds * 1e3:7.1f} ms, {length / seconds:8.0f} tokens/s", flush=True)
release()


# 6. generating with the KV cache, and without it: then every step reads the whole text again
@torch.no_grad()
def generation_steps(n_tokens, prompt=32, use_cache=True):
    ids, cache, times = random_ids(1, prompt), None, []
    for _ in range(n_tokens):
        sync()
        start = time.perf_counter()
        if use_cache:
            out = model(ids if cache is None else ids[:, -1:], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        else:
            out = model(ids, use_cache=False, logits_to_keep=1)
        ids = torch.cat([ids, out.logits[:, -1:].argmax(-1)], dim=1)
        sync()
        times.append(time.perf_counter() - start)
    return times


n_tokens = 8 if SMOKE else 1024
results["generation"] = {
    "with_cache": generation_steps(n_tokens),
    "without_cache": generation_steps(n_tokens, use_cache=False),
}
print(f"generating {n_tokens} tokens: {sum(results['generation']['with_cache']):.1f} s with the cache, "
      f"{sum(results['generation']['without_cache']):.1f} s without", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("saved", OUT)
