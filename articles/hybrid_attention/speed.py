"""What each mixer costs as the context grows, on one GPU: one layer with 16 heads of 128 dimensions.

- prefill: the whole sequence at once, the way a prompt is read: softmax attention through PyTorch's fused kernel
  (FlashAttention-style, quadratic arithmetic, no T x T matrix in memory), the Gated DeltaNet chunkwise in plain
  PyTorch (linear arithmetic);
- decoding: one new token, the way an answer is written: softmax attention reads the keys and values of every
  earlier token; the Gated DeltaNet updates a fixed 128 x 128 state per head.

Production models run both with dedicated kernels (FlashAttention, flash-linear-attention); these are plain PyTorch
on purpose, to show the shape of the curves, not the best absolute numbers.

    uv run --group articles python articles/hybrid_attention/speed.py   # a Kaggle GPU, a few minutes
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from mixers import delta_chunked

HEADS, DIM = 16, 128
SMOKE = os.environ.get("SMOKE") == "1"
LENGTHS = [512, 1024, 2048] if SMOKE else [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def timed(fn, repeat=5):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


def measure(length, device):
    """Reading a prompt of `length` tokens, and writing one token after `length` tokens, with each mixer."""
    q, k, v = (torch.randn(1, HEADS, length, DIM, device=device, dtype=torch.float16) for _ in range(3))
    row = {"length": length}
    row["softmax"] = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    qf, kf, vf = (F.normalize(x.float(), dim=-1) for x in (q, k, v))
    beta = torch.rand(1, HEADS, length, device=device)
    log_alpha = -0.1 * torch.rand(1, HEADS, length, device=device)
    try:
        row["gated_delta"] = timed(lambda: delta_chunked(qf, kf, vf, beta, log_alpha), repeat=3)
    except torch.cuda.OutOfMemoryError:
        row["gated_delta"] = None
    # decoding one token with `length` tokens before it
    q1 = torch.randn(1, HEADS, 1, DIM, device=device, dtype=torch.float16)
    step = {"length": length}
    step["softmax"] = timed(lambda: F.scaled_dot_product_attention(q1, k, v), repeat=20)
    state = torch.zeros(1, HEADS, DIM, DIM, device=device)
    k1, v1, q1f = (F.normalize(torch.randn(1, HEADS, DIM, device=device), dim=-1) for _ in range(3))

    def recurrent_step():
        s = state * 0.99
        recalled = torch.einsum("bhk,bhkv->bhv", k1, s)
        s = s + k1[..., :, None] * (0.5 * (v1 - recalled))[..., None, :]
        return torch.einsum("bhk,bhkv->bhv", q1f, s)

    step["gated_delta"] = timed(recurrent_step, repeat=20)
    step["kv_bytes"] = 2 * HEADS * length * DIM * 2  # keys and values in fp16
    step["state_bytes"] = HEADS * DIM * DIM * 4  # the state in fp32
    return row, step


def main():
    assert torch.cuda.is_available(), "speed.py measures a GPU"
    out = Path(os.environ.get("OUT", Path(__file__).parent / "data"))
    out.mkdir(parents=True, exist_ok=True)
    result = {"device": torch.cuda.get_device_name(), "heads": HEADS, "dim": DIM, "prefill": [], "decode": []}
    for length in LENGTHS:
        row, step = measure(length, "cuda")
        result["prefill"].append(row)
        result["decode"].append(step)
        print(row, step, flush=True)
        torch.cuda.empty_cache()  # the tensors of this length went with measure()
    (out / "speed.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
