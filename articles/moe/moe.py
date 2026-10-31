"""A small decoder-only Transformer whose feed-forward layers are either dense or a mixture of experts.

The rest is the usual modern block (RMSNorm, attention with rotary positions, SwiGLU feed-forward layers), the same
in every model, so that the experiments differ only in the feed-forward part.

A mixture-of-experts layer has E feed-forward networks, the experts, and a router: a linear map from a token's
vector to E scores. A softmax turns the scores into probabilities, the k largest pick the experts the token goes
to, and the token's output is the sum of those experts' outputs, weighted by their probabilities renormalised to
sum to 1. Three ways of keeping the experts evenly loaded can be switched on:

- "none": nothing, the router is trained by the language modelling loss alone;
- "loss": the auxiliary loss of the Switch Transformer, E * sum_i f_i P_i, where f_i is the share of the token-expert
  assignments that went to expert i and P_i the mean probability the router gave it;
- "bias": the auxiliary-loss-free method of DeepSeek-V3: a bias per expert is added to the scores for choosing the
  experts only, and after every step it is raised for experts that got fewer tokens than average and lowered for
  the others.

    uv run python articles/moe/moe.py   # checks the expert dispatch against a dense reference
"""

import torch
import torch.nn.functional as F


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, hidden, bias=False)
        self.up = torch.nn.Linear(d_model, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(torch.nn.Module):
    def __init__(self, d_model, hidden, experts, top_k, balance="loss", shared=0):
        super().__init__()
        self.experts_count, self.top_k, self.balance = experts, top_k, balance
        self.router = torch.nn.Linear(d_model, experts, bias=False)
        # the experts' weights side by side, so that all of them run in one batched product: expert e is a SwiGLU
        # network with the matrices gate[e], up[e] and down[e]
        self.gate = torch.nn.Parameter(torch.randn(experts, d_model, hidden) * 0.02)
        self.up = torch.nn.Parameter(torch.randn(experts, d_model, hidden) * 0.02)
        self.down = torch.nn.Parameter(torch.randn(experts, hidden, d_model) * 0.02)
        # an expert every token goes through, as in DeepSeekMoE; its width is `shared` times an expert's
        self.shared = SwiGLU(d_model, hidden * shared) if shared else None
        self.register_buffer("bias", torch.zeros(experts))  # used only when balance == "bias"
        self.aux_loss = torch.zeros(())
        self.load = torch.zeros(experts)
        self.capacity = None  # assignments per expert beyond which tokens are dropped; None: never drop
        self.dropped = 0.0

    def expert(self, e, x):
        return (F.silu(x @ self.gate[e]) * (x @ self.up[e])) @ self.down[e]

    def route(self, tokens):
        """For every token: the indices of its k experts and their weights; also the router's probabilities."""
        probs = self.router(tokens).float().softmax(dim=-1)
        scores = probs + self.bias if self.balance == "bias" else probs
        chosen = scores.topk(self.top_k, dim=-1).indices
        weights = probs.gather(-1, chosen)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return chosen, weights, probs

    def forward(self, x):
        capacity = self.capacity
        tokens = x.reshape(-1, x.shape[-1])
        chosen, weights, probs = self.route(tokens)
        # one row per (token, expert) assignment, sorted by expert; within an expert, tokens keep their order
        expert = chosen.flatten()
        row = torch.arange(tokens.shape[0], device=x.device).repeat_interleave(self.top_k)
        order = expert.argsort(stable=True)
        expert, row, weight = expert[order], row[order], weights.flatten()[order]
        counts = torch.bincount(expert, minlength=self.experts_count)
        place = torch.arange(expert.numel(), device=x.device) - (counts.cumsum(0) - counts)[expert]
        # every expert gets a queue of the same length: the longest one, rounded up so that the shapes repeat from
        # step to step (a GPU backend compiles its kernels for every new shape); with a capacity, assignments beyond
        # it are dropped
        if capacity is None:
            length = -(-int(counts.max()) // 256) * 256
        else:
            length = capacity
            keep = place < length
            self.dropped = 1.0 - keep.float().mean().item()
            expert, row, weight, place = expert[keep], row[keep], weight[keep], place[keep]
        queue = tokens.new_zeros(self.experts_count, length, tokens.shape[-1])
        queue[expert, place] = tokens[row]
        result = (F.silu(queue @ self.gate) * (queue @ self.up)) @ self.down  # all experts at once
        out = torch.zeros_like(tokens)
        out.index_add_(0, row, result[expert, place] * weight[:, None].to(tokens.dtype))
        if self.shared is not None:
            out = out + self.shared(tokens)
        self.chosen, self.probs = chosen.detach(), probs.detach()  # the last decision, for routing.py
        # the share of assignments each expert got (f), and the mean probability the router gave it (P)
        self.load = (torch.bincount(chosen.flatten(), minlength=self.experts_count) / chosen.numel()).detach()
        self.aux_loss = self.experts_count * (self.load * probs.mean(dim=0)).sum()
        return out.reshape(x.shape)

    @torch.no_grad()
    def update_bias(self, rate):
        # DeepSeek-V3: raise the bias of experts below the mean load, lower it for those above
        if self.balance == "bias":
            self.bias += rate * torch.sign(self.load.mean() - self.load).to(self.bias.device)


def rotary(x, base=10_000.0):
    """Rotate pairs of dimensions by angles that grow with the position (RoPE); x: (batch, heads, length, dim)."""
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * freqs
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(torch.nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, length, d = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotary(q), rotary(k), v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(batch, length, d))


class Block(torch.nn.Module):
    def __init__(self, d_model, heads, ffn):
        super().__init__()
        self.norm1, self.norm2 = torch.nn.RMSNorm(d_model), torch.nn.RMSNorm(d_model)
        self.attention = Attention(d_model, heads)
        self.ffn = ffn

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def feed_forward(d_model, spec):
    """spec: {"hidden": h} for a dense layer, or {"hidden": h, "experts": E, "top_k": k, ...} for a mixture."""
    if "experts" not in spec:
        return SwiGLU(d_model, spec["hidden"])
    return MoE(d_model, spec["hidden"], spec["experts"], spec["top_k"], spec.get("balance", "loss"),
               spec.get("shared", 0))


class Model(torch.nn.Module):
    def __init__(self, vocab, d_model, layers, heads, ffn):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.blocks = torch.nn.ModuleList(Block(d_model, heads, feed_forward(d_model, ffn)) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied, as in small models
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def moe_layers(self):
        return [block.ffn for block in self.blocks if isinstance(block.ffn, MoE)]

    def parameters_count(self):
        """All parameters, and those one token passes through (k of E experts in every mixture layer)."""
        total = sum(p.numel() for p in self.parameters())
        idle = 0
        for layer in self.moe_layers():
            per_expert = (layer.gate.numel() + layer.up.numel() + layer.down.numel()) // layer.experts_count
            idle += per_expert * (layer.experts_count - layer.top_k)
        return total, total - idle


def check(seed=0):
    """The sorted, batched dispatch must give what a dense computation gives: every expert on every token, then
    the outputs of the k chosen ones mixed with the renormalised probabilities and the rest multiplied by zero."""
    torch.manual_seed(seed)
    layer = MoE(d_model=16, hidden=24, experts=6, top_k=2).double()
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    tokens = x.reshape(-1, 16)
    chosen, weights, _ = layer.route(tokens)
    mix = torch.zeros(tokens.shape[0], 6, dtype=torch.float64).scatter(1, chosen, weights.double())
    every = torch.stack([layer.expert(e, tokens) for e in range(6)], dim=1)  # (tokens, experts, d)
    reference = (mix[..., None] * every).sum(dim=1).reshape(x.shape)
    return (layer(x) - reference).abs().max().item()


if __name__ == "__main__":
    print("largest difference from the dense reference:", check())
