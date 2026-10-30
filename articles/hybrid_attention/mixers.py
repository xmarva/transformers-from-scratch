"""Token mixers of the post "Hybrid attention", written out in PyTorch.

Every mixer maps queries, keys and values of shape (batch, heads, length, dim) to outputs of the same shape, causally:
position t sees positions 1..t only.

- softmax attention: the ordinary kind, through torch's scaled_dot_product_attention;
- linear attention (Katharopoulos et al., 2020): a running sum of outer products k v^T, read with q;
- DeltaNet (Schlag et al., 2021): the same memory, written with the delta rule;
- Gated DeltaNet (Yang et al., 2025): the delta rule with a forget gate.

The linear ones come in two forms that compute the same thing: a recurrence over positions, which is what generating
text looks like, and a chunkwise form, which processes 64 positions at a time with matrix products and is what
training uses (Yang et al., 2024). check() compares the two.
"""

import math

import torch
import torch.nn.functional as F


def softmax_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def feature(x):
    # Katharopoulos et al.: a positive feature map in place of the exponential
    return F.elu(x) + 1


def linear_recurrent(q, k, v):
    q, k = feature(q), feature(k)
    batch, heads, length, d_k = q.shape
    state = q.new_zeros(batch, heads, d_k, v.shape[-1])  # S = sum of k v^T
    norm = q.new_zeros(batch, heads, d_k)  # z = sum of k
    out = []
    for t in range(length):
        state = state + k[:, :, t, :, None] * v[:, :, t, None, :]
        norm = norm + k[:, :, t]
        numerator = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
        denominator = torch.einsum("bhk,bhk->bh", q[:, :, t], norm)[..., None]
        out.append(numerator / denominator)
    return torch.stack(out, dim=2)


def linear_attention(q, k, v):
    # the same as linear_recurrent, all positions at once: (phi(Q) phi(K)^T, masked) V, row-normalized
    q, k = feature(q), feature(k)
    length = q.shape[2]
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    scores = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0)
    return (scores @ v) / scores.sum(-1, keepdim=True)


def delta_recurrent(q, k, v, beta, log_alpha=None):
    """S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T, o_t = S_t^T q_t; alpha = 1 is DeltaNet."""
    batch, heads, length, d_k = q.shape
    state = q.new_zeros(batch, heads, d_k, v.shape[-1])
    out = []
    for t in range(length):
        k_t, v_t, b_t = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        if log_alpha is not None:
            state = state * log_alpha[:, :, t, None, None].exp()
        recalled = torch.einsum("bhk,bhkv->bhv", k_t, state)  # what the memory says for k_t now
        state = state + k_t[..., :, None] * (b_t * (v_t - recalled))[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], state))
    return torch.stack(out, dim=2)


def delta_chunked(q, k, v, beta, log_alpha=None, chunk=64):
    """The delta rule chunk by chunk (Yang et al., 2024), with the forget gate of Gated DeltaNet if log_alpha is given.

    Inside a chunk, the writes depend on each other only through a lower-triangular system, solved at once; between
    chunks, the state carries over. The result equals delta_recurrent up to rounding.
    """
    batch, heads, length, d_k = q.shape
    d_v = v.shape[-1]
    pad = (-length) % chunk
    if log_alpha is None:
        log_alpha = q.new_zeros(batch, heads, length)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, log_alpha = F.pad(beta, (0, pad)), F.pad(log_alpha, (0, pad))
    n = q.shape[2] // chunk

    def split(x):
        return x.reshape(batch, heads, n, chunk, *x.shape[3:])

    q, k, v, beta, log_alpha = split(q), split(k), split(v), split(beta), split(log_alpha)
    decay = log_alpha.cumsum(-1)  # log of the product of gates from the chunk's start, inclusive
    gap = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()  # alpha_{j+1} ... alpha_i for j <= i
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    strictly_lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    # T = (I + A)^-1 diag(beta), with A_ij = beta_i (k_i . k_j) gap_ij for j < i
    a = ((k @ k.transpose(-1, -2)) * gap * beta[..., None]).masked_fill(~strictly_lower, 0)
    t = torch.linalg.solve_triangular(eye + a, torch.diag_embed(beta), upper=False)
    w = t @ (k * decay[..., None].exp())  # what each write takes away from the state at the chunk's start
    u = t @ v  # and what it adds
    inside = ((q @ k.transpose(-1, -2)) * gap).masked_fill(~(strictly_lower | eye.bool()), 0)
    state = q.new_zeros(batch, heads, d_k, d_v)
    out = []
    for i in range(n):
        new = u[:, :, i] - w[:, :, i] @ state  # the corrections each position writes, given the state
        out.append((q[:, :, i] * decay[:, :, i, :, None].exp()) @ state + inside[:, :, i] @ new)
        last = decay[:, :, i, -1]
        to_end = (last[..., None] - decay[:, :, i]).exp()  # gates from each position to the chunk's end
        state = state * last[..., None, None].exp() + (k[:, :, i] * to_end[..., None]).transpose(-1, -2) @ new
    out = torch.stack(out, dim=2).reshape(batch, heads, n * chunk, d_v)
    return out[:, :, :length]


def check(seed=0, length=150):
    """The two forms of each mixer agree; returns the largest differences."""
    torch.manual_seed(seed)
    shape = (2, 3, length, 16)
    q, k, v = (torch.randn(shape, dtype=torch.float64) for _ in range(3))
    k_unit, q_unit = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
    beta = torch.rand(shape[:3], dtype=torch.float64)
    log_alpha = -torch.rand(shape[:3], dtype=torch.float64) * 0.3
    return {
        "linear": (linear_recurrent(q, k, v) - linear_attention(q, k, v)).abs().max().item(),
        "delta": (delta_recurrent(q_unit, k_unit, v, beta) - delta_chunked(q_unit, k_unit, v, beta, chunk=32))
        .abs().max().item(),
        "gated delta": (delta_recurrent(q_unit, k_unit, v, beta, log_alpha)
                        - delta_chunked(q_unit, k_unit, v, beta, log_alpha, chunk=32)).abs().max().item(),
    }


class Mixer(torch.nn.Module):
    """One token-mixing layer: projections, a short causal convolution, the mixer, an output projection."""

    def __init__(self, kind, d_model, heads):
        super().__init__()
        self.kind, self.heads, self.head_dim = kind, heads, d_model // heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        # a short causal convolution lets a position see its neighbours, as in Mamba, DeltaNet and their kin
        self.conv = torch.nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=4, groups=3 * d_model, padding=3)
        if kind in ("delta", "gated delta"):
            self.beta = torch.nn.Linear(d_model, heads)
        if kind == "gated delta":
            # the gate as in Mamba-2 and the Gated DeltaNet code: log alpha = -A softplus(W x + dt_bias), with A and
            # the step dt drawn so that alpha starts close to 1 and the memory starts long; a gate that starts near
            # 0.5 halves the memory at every token, and the gradient through 30 tokens is too small to fix it
            self.alpha = torch.nn.Linear(d_model, heads)
            self.A_log = torch.nn.Parameter(torch.empty(heads).uniform_(0, 16).log())
            dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            self.dt_bias = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus(dt_bias) = dt
        self.out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, length, d_model = x.shape
        qkv = self.conv(self.qkv(x).transpose(1, 2))[..., :length].transpose(1, 2)
        qkv = F.silu(qkv).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.kind == "softmax":
            y = softmax_attention(q, k, v)
        elif self.kind == "linear":
            y = linear_attention(q, k, v)
        else:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            beta = torch.sigmoid(self.beta(x)).transpose(1, 2)
            log_alpha = None
            if self.kind == "gated delta":
                log_alpha = (-self.A_log.exp() * F.softplus(self.alpha(x) + self.dt_bias)).transpose(1, 2)
            y = delta_chunked(q, k, v, beta, log_alpha)
        return self.out(y.transpose(1, 2).reshape(batch, length, d_model))


class Model(torch.nn.Module):
    """A small language model whose layers mix tokens with the mixers listed in `kinds`, one per layer."""

    def __init__(self, kinds, vocab, d_model=64, heads=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.mixers = torch.nn.ModuleList(Mixer(kind, d_model, heads) for kind in kinds)
        self.mlps = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(d_model, 4 * d_model), torch.nn.GELU(),
                                torch.nn.Linear(4 * d_model, d_model)) for _ in kinds)
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(d_model) for _ in range(2 * len(kinds) + 1))
        self.head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for i, (mixer, mlp) in enumerate(zip(self.mixers, self.mlps, strict=True)):
            x = x + mixer(self.norms[2 * i](x))
            x = x + mlp(self.norms[2 * i + 1](x))
        return self.head(self.norms[-1](x))


if __name__ == "__main__":
    print(check())
