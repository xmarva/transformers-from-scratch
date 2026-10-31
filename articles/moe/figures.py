"""Figures, charts and the cover of the post "Mixture of experts" on xmarva.github.io.

The experiments are those of train.py and routing.py (data/train.json, data/routing.json), run with three seeds on
two T4 GPUs on Kaggle by kaggle.py; speed.py's timings, on the same GPU, are in data/speed.json. Facts about real
models come from their config.json files on the Hugging Face Hub, kept in data/configs/ as they were when the post
was written.

    uv run python tools/render.py --article moe --out <site>/assets/posts/mixture-of-experts
"""

import json
import math
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"

# the models of train.py, repeated here so that the figures don't import torch
D_MODEL, DENSE_HIDDEN, EXPERTS, EXPERT_HIDDEN, TOP_K = 256, 768, 8, 384, 2
FINE_EXPERTS, FINE_HIDDEN, FINE_TOP_K = 32, 96, 8


# ---------------------------------------------------------------- a dense layer and a mixture


def moe_layer() -> Figure:
    fig = Figure(
        760,
        330,
        alt=f"A dense feed-forward layer next to a mixture of experts. In the dense layer every token passes through "
        f"all {DENSE_HIDDEN} hidden units. In the mixture, a router scores {EXPERTS} experts of {EXPERT_HIDDEN} "
        f"hidden units each for the token, the token goes to the {TOP_K} best, and its output is their outputs "
        f"added with the router's weights. The token passes through {TOP_K * EXPERT_HIDDEN} hidden units in both, "
        f"but the mixture holds {EXPERTS * EXPERT_HIDDEN}",
    )
    fig.text(14, 28, "A dense feed-forward layer and a mixture of experts", kind="title")
    fig.text(14, 50, "the same work per token, four times as many weights to choose from", kind="note")
    # dense: x -> one wide network -> y
    fig.box(14, 70, 214, 246, role="plain", r=10)
    fig.text(28, 94, "dense", kind="subtitle")
    fig.chip(104, 116, "x", role="data")
    fig.arrow((121, 150), (121, 176), role="muted")
    fig.box(40, 180, 162, 52, role="enc", label=f"FFN, {DENSE_HIDDEN} units", strong=True)
    fig.arrow((121, 236), (121, 262), role="muted")
    fig.chip(104, 266, "y", role="data")
    # mixture: x -> router -> 8 experts, 2 lit -> weighted sum -> y
    x0 = 242
    fig.box(x0, 70, 504, 246, role="plain", r=10)
    fig.text(x0 + 14, 94, "mixture of experts", kind="subtitle")
    fig.chip(x0 + 236, 104, "x", role="data")
    fig.arrow((x0 + 253, 138), (x0 + 253, 150), role="muted")
    fig.box(x0 + 168, 152, 170, 32, role="dec", label="router: 8 scores")
    lit = {2: "g₃", 5: "g₆"}
    width, gap = 52, 8
    left = x0 + (504 - (EXPERTS * width + (EXPERTS - 1) * gap)) / 2
    for e in range(EXPERTS):
        ex = left + e * (width + gap)
        fig.box(ex, 214, width, 36, role="enc", label=f"E{e + 1}", strong=e in lit)
        if e in lit:
            fig.arrow((x0 + 253, 186), (ex + width / 2, 212), role="dec", width=1.4)
            fig.arrow((ex + width / 2, 252), (x0 + 253, 276), role="enc", width=1.4)
            fig.text(ex + width / 2 + (-30 if e == 2 else 30), 270, lit[e], kind="mono_dec", anchor="middle")
    fig.chip(x0 + 236, 278, "y", role="data")
    fig.text(x0 + 300, 298, "y = g₃·E3(x) + g₆·E6(x)", kind="mono")
    fig.text(x0 + 14, 298, f"{EXPERT_HIDDEN} units each", kind="small")
    return fig


# ---------------------------------------------------------------- the router, with real numbers

ROUTER_MODEL, ROUTER_LAYER = "MoE, 8 experts", 2  # the model and layer (0-based) the worked examples come from
ROUTER_SNIPPET, ROUTER_TOKEN = "ru", 3  # the token " президент" of the Russian snippet


@cache
def routing():
    return json.loads((DATA / "routing.json").read_text())


def router_example() -> Figure:
    snippet = routing()["models"][ROUTER_MODEL]["snippets"][ROUTER_SNIPPET]
    probs = np.array(snippet["probs"][ROUTER_LAYER][ROUTER_TOKEN])
    chosen = snippet["chosen"][ROUTER_LAYER][ROUTER_TOKEN]
    weights = probs[chosen] / probs[chosen].sum()
    token = snippet["tokens"][ROUTER_TOKEN]
    context = snippet["tokens"][:ROUTER_TOKEN + 1]
    fig = Figure(
        760,
        380,
        alt=f"The router of layer {ROUTER_LAYER + 1} of the trained model with {EXPERTS} experts, deciding where "
        f"the token '{token.strip()}' of a held-out Russian sentence goes. Its probabilities for the eight experts are "
        + ", ".join(f"E{e + 1} {p:.2f}" for e, p in enumerate(probs))
        + f". The two largest, E{chosen[0] + 1} and E{chosen[1] + 1}, are kept and divided by their sum, "
        f"giving weights {weights[0]:.2f} and {weights[1]:.2f}; the token's output is the two experts' outputs "
        "added with these weights",
    )
    fig.text(14, 28, "One token through the router", kind="title")
    fig.text(14, 50, f"layer {ROUTER_LAYER + 1} of the trained model with {EXPERTS} experts, on a held-out Russian "
             "sentence", kind="note")
    fig.row([t.strip() or "·" for t in context], 14, 70, role="data", strong=lambda i, t: i == ROUTER_TOKEN, size=13,
            h=28)
    # the probabilities as bars
    base, top, bar, pitch, x0 = 316, 150, 34, 50, 40
    scale = (base - top) / probs.max()
    fig.line((x0 - 10, base), (x0 + EXPERTS * pitch, base), role="muted", width=1)
    for e, p in enumerate(probs):
        x = x0 + e * pitch
        height = max(p * scale, 1.5)
        fig.rect(x, base - height, bar, height, role="enc", tint=e not in chosen, r=2)
        fig.text(x + bar / 2, base - height - 8, f"{p:.2f}", kind="small", anchor="middle")
        fig.text(x + bar / 2, base + 20, f"E{e + 1}", kind="label", anchor="middle")
    fig.text(x0 - 10, 124, "1. softmax over 8 scores: probabilities", kind="small")
    fig.text(x0 - 10, 362, f"2. keep the {TOP_K} largest", kind="small")
    # renormalisation and the output
    x1 = 470
    fig.text(x1, 150, "3. divide by their sum", kind="small")
    total = probs[chosen].sum()
    for i, e in enumerate(chosen):
        fig.text(x1, 182 + 28 * i, f"g{e + 1} = {probs[e]:.2f} / {total:.2f} = {weights[i]:.2f}", kind="mono")
    fig.text(x1, 262, "4. the output", kind="small")
    fig.text(x1, 292, f"y = {weights[0]:.2f}·E{chosen[0] + 1}(x)", kind="mono_dec")
    fig.text(x1, 316, f"  + {weights[1]:.2f}·E{chosen[1] + 1}(x)", kind="mono_dec")
    return fig


# ---------------------------------------------------------------- sorting tokens into queues

DISPATCH_TOKENS = 8  # the first tokens of the English snippet


def dispatch() -> Figure:
    snippet = routing()["models"][ROUTER_MODEL]["snippets"]["en"]
    tokens = [t.strip() for t in snippet["tokens"][:DISPATCH_TOKENS]]
    chosen = snippet["chosen"][ROUTER_LAYER][:DISPATCH_TOKENS]
    queues = {e: [i for i, pair in enumerate(chosen) if e in pair] for e in range(EXPERTS)}
    fig = Figure(
        760,
        96 + 36 * EXPERTS + 52,
        alt=f"How a mixture-of-experts layer processes a batch. The first {DISPATCH_TOKENS} tokens of a held-out "
        f"English sentence each go to two of the eight experts of layer {ROUTER_LAYER + 1}: "
        + "; ".join(f"'{t}' to E{a + 1} and E{b + 1}" for t, (a, b) in zip(tokens, chosen, strict=True))
        + ". Sorted by expert, they form a queue for each expert: "
        + "; ".join(f"E{e + 1}: {len(q)}" for e, q in queues.items())
        + ". The queues are padded to the same length and every expert processes its queue in one batched product",
    )
    fig.text(14, 28, "Sorting the tokens into the experts' queues", kind="title")
    fig.text(14, 50, f"the first {DISPATCH_TOKENS} tokens of an English sentence, layer {ROUTER_LAYER + 1}; each "
             f"token goes to {TOP_K} experts", kind="note")
    y0 = 80
    for i, (token, pair) in enumerate(zip(tokens, chosen, strict=True)):
        y = y0 + i * 36
        width = fig.chip(14, y, token, role="data", h=28, size=12)
        fig.text(14 + width + 10, y + 19, f"→ E{pair[0] + 1}, E{pair[1] + 1}", kind="label")
    x0 = 300
    longest = max(len(q) for q in queues.values())
    for e, queue in queues.items():
        y = y0 + e * 36
        fig.box(x0, y, 44, 28, role="enc", label=f"E{e + 1}", strong=bool(queue))
        x = x0 + 54
        for i in queue:
            x += fig.chip(x, y, tokens[i], role="data", h=28, size=12) + 5
        fig.text(746, y + 19, f"{len(queue)}", kind="count", anchor="end")
    fig.text(746, y0 - 8, "queue", kind="small", anchor="end")
    fig.text(14, 96 + 36 * EXPERTS + 14, f"the queues are padded to {longest} rows and multiplied by the experts' "
             "weights in one batched product;", kind="small")
    fig.text(14, 96 + 36 * EXPERTS + 34, "the outputs return to their tokens, weighted by the router", kind="small")
    return fig


# ---------------------------------------------------------------- which expert each token goes to

TOKEN_LAYERS = (0, 5)  # the first and the last layer
KIND_ROLES = {"en": "data", "ru": "dec", "code": "enc"}  # as in the specialization chart
LEANING = 0.6  # an expert leans to a kind of text if at least this share of its tokens is of that kind


def expert_kinds(model, layer):
    """For every expert of a layer, the kind of text it leans to, or None."""
    shares = routing()["models"][model]["shares"][layer]
    traffic = np.array([shares[kind] for kind in KIND_ROLES])
    composition = traffic / traffic.sum(axis=0)
    return [list(KIND_ROLES)[composition[:, e].argmax()] if composition[:, e].max() >= LEANING else None
            for e in range(composition.shape[1])]


def shown(token):
    """A token as it is drawn: without its leading space, with newlines and indentation visible."""
    if token.strip():
        return token.strip()
    return token.replace("\n", "\\n").replace(" ", "·")


def token_rows(width, size=12, gap=4):
    """The leading tokens of each snippet that fit in a row of the given width."""
    rows = {}
    for kind, snippet in routing()["models"][ROUTER_MODEL]["snippets"].items():
        used, count = 0.0, 0
        for token in snippet["tokens"]:
            w = Figure.chip_width(shown(token), size) + gap
            if used + w > width:
                break
            used, count = used + w, count + 1
        rows[kind] = count
    return rows


def tokens() -> Figure:
    snippets = routing()["models"][ROUTER_MODEL]["snippets"]
    kinds = {layer: expert_kinds(ROUTER_MODEL, layer) for layer in TOKEN_LAYERS}
    x0, size = 92, 12
    counts = token_rows(746 - x0, size)
    pitch = 50
    height = 96 + len(snippets) * (24 + len(TOKEN_LAYERS) * pitch) + 30

    def own(kind, layer):
        chosen = snippets[kind]["chosen"][layer][:counts[kind]]
        return sum(kinds[layer][pair[0]] == kind for pair in chosen)

    fig = Figure(
        760,
        height,
        alt=f"The first-choice expert of every token of three held-out snippets, English, Russian and Python, in the "
        f"first and the last layer of the trained model with {EXPERTS} experts. A token is coloured by the kind of "
        f"text its expert leans to, if at least {LEANING:.0%} of that expert's tokens are of one kind. Tokens that go "
        "to an expert of their own kind: "
        + "; ".join(f"layer {layer + 1}: " + ", ".join(f"{own(kind, layer)} of {counts[kind]} {KINDS[kind]}"
                                                       for kind in snippets) for layer in TOKEN_LAYERS)
        + ". In the first layer the Russian expert takes mostly the Cyrillic pieces of names, the English one short "
        "words, and a third expert Python's symbols and newlines; in the last, almost every token goes to an expert of "
        "its language",
    )
    fig.text(14, 28, "The first expert of every token, in the first layer and the last", kind="title")
    fig.text(14, 50, f"colour: the kind of text that makes up ≥{LEANING:.0%} of the expert's tokens; grey: none does",
             kind="note")
    y = 80
    for kind, snippet in snippets.items():
        fig.text(14, y + 10, KINDS[kind], kind="subtitle")
        y += 22
        for layer in TOKEN_LAYERS:
            fig.text(14, y + 18, f"layer {layer + 1}", kind="label")
            x = x0
            for i in range(counts[kind]):
                expert = snippet["chosen"][layer][i][0]
                lean = kinds[layer][expert]
                width = fig.chip(x, y, shown(snippet["tokens"][i]), role=KIND_ROLES[lean] if lean else "neutral",
                                 h=26, size=size)
                fig.text(x + width / 2, y + 40, f"E{expert + 1}", kind="small", anchor="middle")
                x += width + 4
            y += pitch
        y += 2
    fig.text(14, height - 14, "under each token: its first-choice expert in that layer; expert numbers are not shared "
             "between layers", kind="small")
    return fig


# ---------------------------------------------------------------- finer experts, and one for everybody


def fine_grained() -> Figure:
    rows = [
        (f"{EXPERTS} experts, {TOP_K} per token", EXPERTS, TOP_K, 0, math.comb(EXPERTS, TOP_K)),
        (f"{FINE_EXPERTS} experts, {FINE_TOP_K} per token", FINE_EXPERTS, FINE_TOP_K, 0,
         math.comb(FINE_EXPERTS, FINE_TOP_K)),
        ("DeepSeek-V3: 1 shared + 256 routed, 8 per token", 256, 8, 1, math.comb(256, 8)),
    ]
    fig = Figure(
        760,
        330,
        alt="Three ways to cut the same budget of experts. Eight experts with two per token can be combined in "
        f"{rows[0][4]} ways; 32 experts a quarter of the size with eight per token, the same compute and the "
        f"same parameters, in {rows[1][4]:,} ways. DeepSeek-V3 has one shared expert every token passes through and "
        f"256 routed experts with eight per token: about {rows[2][4]:.1e} combinations",
    )
    fig.text(14, 28, "Finer experts and a shared one", kind="title")
    fig.text(14, 50, "the same parameters and the same compute, cut into more pieces: many more combinations",
             kind="note")
    span = 520
    for r, (name, count, chosen, shared, combinations) in enumerate(rows):
        y = 78 + r * 84
        fig.text(14, y + 4, name, kind="subtitle")
        generator = np.random.default_rng(3 + r)
        lit = set(generator.choice(count, chosen, replace=False).tolist())
        x = 14
        if shared:
            fig.box(x, y + 18, 44, 34, role="neutral", label="S", strong=True)
            x += 54
        pitch = (span - (x - 14)) / count
        gap = 6 if count <= 8 else (3 if count <= 32 else 0.8)
        for e in range(count):
            if count <= 32:
                fig.box(x + e * pitch, y + 18, pitch - gap, 34, role="enc", strong=e in lit, r=4 if count > 8 else 7)
            else:
                fig.rect(x + e * pitch, y + 18, max(pitch - gap, 1.2), 34, role="enc", tint=e not in lit, r=0.5)
        if combinations < 1e9:
            label = f"{combinations:,}"
        else:
            exponent = int(math.log10(combinations))
            label = f"{combinations / 10 ** exponent:.1f} × 10" + str(exponent).translate(str.maketrans("0123456789",
                                                                                                         "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        fig.text(560, y + 32, label, kind="mono")
        fig.text(560, y + 52, "ways to pick the experts", kind="small")
    fig.text(14, 316, "lit: the experts one token goes to; S: the shared expert, used by every token", kind="small")
    return fig


# ---------------------------------------------------------------- experts on several GPUs


def expert_parallel() -> Figure:
    # where GPU 0's two tokens go: each to the GPUs that hold its two experts
    routes = {"t1": [2, 7], "t2": [1, 4]}  # expert numbers, 1..8; expert e lives on GPU (e - 1) // 2
    fig = Figure(
        760,
        420,
        alt="Expert parallelism on four GPUs, each holding two of the eight experts and a quarter of the tokens. "
        "The tokens of GPU 0 are followed: after the router, each is sent to the GPUs that hold its two experts, in "
        "an all-to-all exchange in which every GPU sends to every other; the experts process what they received; a "
        "second all-to-all brings the results back to GPU 0, where each token's two results are added",
    )
    fig.text(14, 28, "Expert parallelism: two all-to-all exchanges per layer", kind="title")
    fig.text(14, 50, "four GPUs, two experts on each; the arrows follow the two tokens of GPU 0", kind="note")
    width = 154
    xs = [104 + g * (width + 12) for g in range(4)]
    rows = {"tokens": 92, "experts": 210, "results": 328}
    for g, x in enumerate(xs):
        fig.text(x + width / 2, 82, f"GPU {g}", kind="subtitle", anchor="middle")
        for y in rows.values():
            fig.box(x, y, width, 50, role="plain", r=10)
        fig.row([f"t{2 * g + 1}", f"t{2 * g + 2}"], x + 12, rows["tokens"] + 12, role="data",
                strong=lambda i, t, g=g: g == 0, size=12, h=26)
        for i in range(2):
            fig.box(x + 12 + i * 68, rows["experts"] + 11, 62, 28, role="enc", label=f"E{2 * g + i + 1}",
                    strong=any(2 * g + i + 1 in experts for experts in routes.values()))
        fig.row([f"y{2 * g + 1}", f"y{2 * g + 2}"], x + 12, rows["results"] + 12, role="data",
                strong=lambda i, t, g=g: g == 0, size=12, h=26)
    for name, y in rows.items():
        fig.text(14, y + 30, name, kind="label")
    chip = Figure.chip_width("t1", 12) + 5
    for i, experts in enumerate(routes.values()):
        x_token = xs[0] + 12 + i * chip + chip / 2 - 2.5
        for e in experts:
            g, slot = divmod(e - 1, 2)
            x_expert = xs[g] + 12 + slot * 68 + 31
            fig.arrow((x_token, rows["tokens"] + 40), (x_expert, rows["experts"] + 9), role="dec", width=1.4)
            fig.arrow((x_expert, rows["experts"] + 41), (x_token, rows["results"] + 10), role="enc", width=1.4)
    fig.text(14, 406, "rose: the first all-to-all, tokens to their experts; teal: the second, results back to the "
             "tokens' GPU", kind="small")
    return fig


# ---------------------------------------------------------------- where the parameters are


def config(repo):
    c = json.loads((DATA / "configs" / f"{repo.replace('/', '__')}.json").read_text())
    return {**c, **c.get("text_config", {})}  # multimodal models keep the language model's fields in text_config


def first(c, *keys, default=None):
    return next((c[k] for k in keys if c.get(k) is not None), default)


def moe_fields(repo):
    """Routed experts per layer, routed experts per token and shared experts, from a config; the field names differ
    from one family to the next."""
    c = config(repo)
    experts = first(c, "n_routed_experts", "num_local_experts", "num_experts")
    per_token = first(c, "num_experts_per_tok", "num_experts_per_token")
    shared = first(c, "n_shared_experts", "num_shared_experts", default=0)
    if c.get("shared_expert_intermediate_size"):  # Qwen3-Next and Qwen3.5: one shared expert of this width
        shared = 1
    if c.get("model_type", "").startswith("llama4"):  # Llama 4's shared expert is in the code, not the config
        shared = 1
    return experts, per_token, shared


def qwen3_parameters(repo):
    """Parameters of a Qwen3 model by part, in billions, from its config: attention, the feed-forward networks a
    token passes through, those it doesn't (the idle experts), the routers, and the embeddings."""
    c = config(repo)
    d, layers, heads, kv, head = (c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                                  c["num_key_value_heads"], c["head_dim"])
    attention = layers * (d * heads * head + 2 * d * kv * head + heads * head * d + 2 * head)  # q, k, v, o, q/k norms
    embeddings = c["vocab_size"] * d * (1 if c["tie_word_embeddings"] else 2)
    norms = layers * 2 * d + d
    if "num_experts" in c:  # every layer is a mixture (decoder_sparse_step 1, no mlp_only_layers)
        expert = 3 * d * c["moe_intermediate_size"]
        used = layers * c["num_experts_per_tok"] * expert
        idle = layers * (c["num_experts"] - c["num_experts_per_tok"]) * expert
        routers = layers * d * c["num_experts"]
    else:
        used, idle, routers = layers * 3 * d * c["intermediate_size"], 0, 0
    parts = {"attention": attention, "used": used, "idle": idle, "routers": routers, "embeddings": embeddings,
             "norms": norms}
    return {name: value / 1e9 for name, value in parts.items()}


PARAMETER_MODELS = [("Qwen3-32B", "Qwen/Qwen3-32B"), ("Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B")]


def chart_parameters(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.66, bottom=0.2)
    segments = [("attention", "attention", c["att"]), ("used", "feed-forward, used per token", c["enc"]),
                ("idle", "experts not used by the token", c["enc_tint"]), ("embeddings", "embeddings", c["data"])]
    for row, (_, repo) in enumerate(PARAMETER_MODELS):
        parts = qwen3_parameters(repo)
        left = 0.0
        for key, label, color in segments:
            ax.barh(row, parts[key], left=left, color=color, edgecolor=c["enc"] if key == "idle" else color,
                    linewidth=0.8, height=0.55, label=label if row == 1 else None)
            left += parts[key]
        active = parts["attention"] + parts["used"] + parts["routers"] + parts["embeddings"] + parts["norms"]
        ax.text(left + 3, row, f"{active:.1f}B of {sum(parts.values()):.1f}B per token", va="center", fontsize=9.5,
                color=c["ink"])
    ax.set_yticks(range(len(PARAMETER_MODELS)), [name for name, _ in PARAMETER_MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 345)
    ax.set_xticks(range(0, 251, 50))
    ax.set_xlabel("billions of parameters")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.36), ncol=2, fontsize=9, frameon=False)
    _caption(fig, "Where the parameters are", "counted from the models' config.json", c)
    return fig


# ---------------------------------------------------------------- mixtures in real models

# name, repository, first release, total and active parameters in billions as the model card or paper states them
REAL_MODELS = [
    ("Mixtral 8x7B", "mistralai/Mixtral-8x7B-v0.1", "Dec 2023", 46.7, 12.9),
    ("DeepSeek-V2", "deepseek-ai/DeepSeek-V2", "May 2024", 236, 21),
    ("OLMoE-1B-7B", "allenai/OLMoE-1B-7B-0924", "Sep 2024", 6.9, 1.3),
    ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3", "Dec 2024", 671, 37),
    ("Llama 4 Maverick", "meta-llama/Llama-4-Maverick-17B-128E", "Apr 2025", 400, 17),
    ("Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B", "Apr 2025", 235, 22),
    ("Kimi K2", "moonshotai/Kimi-K2-Instruct", "Jul 2025", 1000, 32),
    ("GLM-4.5", "zai-org/GLM-4.5", "Jul 2025", 355, 32),
    ("gpt-oss-120b", "openai/gpt-oss-120b", "Aug 2025", 116.8, 5.1),
    ("Qwen3-Next-80B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct", "Sep 2025", 80, 3),
    ("MiniMax-M2", "MiniMaxAI/MiniMax-M2", "Oct 2025", 230, 10),
    ("Qwen3.5-397B-A17B", "Qwen/Qwen3.5-397B-A17B", "Feb 2026", 397, 17),
    ("DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro", "Apr 2026", 1600, 49),
    ("Kimi K3", "moonshotai/Kimi-K3", "Jun 2026", 2800, 104),
]


def chart_models(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 5.6), sharey=True, gridspec_kw={"width_ratios": [2.1, 1]})
    fig.subplots_adjust(left=0.29, right=0.97, top=0.85, bottom=0.09, wspace=0.05)
    rows = range(len(REAL_MODELS))
    for row, (_, repo, _, total, active) in zip(rows, REAL_MODELS, strict=True):
        experts, per_token, shared = moe_fields(repo)
        left.plot([per_token, experts], [row, row], color=c["grid"], linewidth=3, solid_capstyle="round")
        left.plot([experts], [row], "o", color=c["enc"], markersize=7)
        left.plot([per_token], [row], "o", color=c["dec"], markersize=7)
        note = f"{per_token} of {experts}" + (f" + {shared} shared" if shared else "")
        left.text(experts * 1.3, row, note, va="center", fontsize=8.5, color=c["muted"])
        right.barh(row, 100 * active / total, color=c["enc"], height=0.6)
        right.text(100 * active / total + 0.8, row, f"{active:g}B of {total:,g}B", va="center", fontsize=8,
                   color=c["muted"])
    left.set_xscale("log", base=2)
    left.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    left.set_xlim(0.7, 2**15)
    left.minorticks_off()
    left.set_yticks(list(rows), [f"{name}, {date}" for name, _, date, _, _ in REAL_MODELS], fontsize=9)
    left.invert_yaxis()
    left.set_xlabel("routed experts: per token (rose), per layer (teal)")
    right.set_xlim(0, 50)
    right.set_xticks([0, 10, 20, 30])
    right.set_xlabel("active, % of all")
    _caption(fig, "Mixtures of experts in open models", "experts from each model's config.json; parameters as "
             "the model card or paper states them", c)
    return fig


# ---------------------------------------------------------------- charts from the experiments


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, 0.895, note, ha="left", va="top", fontsize=10, color=c["muted"])


@cache
def runs():
    return json.loads((DATA / "train.json").read_text())["runs"]


def run(name, seed=0):
    return next(r for r in runs() if r["model"] == name and r["seed"] == seed)


def mean_validation(entry):
    return float(np.mean([entry[name] for name in ("en", "ru", "code")]))


LOSS_STYLE = {  # model -> (palette key, line style): dense models grey, mixtures teal (the experts' colour)
    "Dense": ("muted", "-"),
    "Dense, 4x wider": ("ink", "-"),
    "MoE, 8 experts": ("enc", "-"),
    "MoE, 32 experts": ("enc", "--"),
}
BALANCE_STYLE = {  # the three versions of the 8-expert mixture
    "MoE, 8 experts, no balancing": ("no balancing", "muted"),
    "MoE, 8 experts": ("auxiliary loss", "enc"),
    "MoE, 8 experts, bias balancing": ("bias, no loss", "data"),
}
KINDS = {"en": "English", "ru": "Russian", "code": "Python"}


FINAL_ROWS = [  # model, label, palette key: every model of train.py, worst to best
    ("Dense", "Dense", "muted"),
    ("MoE, 8 experts, no balancing", "8 experts, no balancing", "enc"),
    ("MoE, 8 experts", "8 experts, balancing loss", "enc"),
    ("MoE, 8 experts, bias balancing", "8 experts, bias", "enc"),
    ("MoE, 32 experts", "32 experts", "enc"),
    ("Dense, 4x wider", "Dense, 4x wider", "ink"),
]


def seeds(name):
    return sorted(r["seed"] for r in runs() if r["model"] == name)


def final_losses(name):
    """The final validation loss of every seed of a model, mean over the three kinds of text."""
    return [mean_validation(run(name, seed)["validation"][-1]) for seed in seeds(name)]


def chart_loss(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, (ax, right) = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.14, wspace=0.62)
    for name, (key, style) in LOSS_STYLE.items():
        records = [run(name, seed)["validation"] for seed in seeds(name)]
        steps = [v["step"] for v in records[0]]
        curve = np.mean([[mean_validation(v) for v in r] for r in records], axis=0)
        ax.plot(steps, curve, color=c[key], linestyle=style, linewidth=2.0, label=name)
    ax.set_xlim(900, 3050)
    ax.set_xticks([1000, 2000, 3000])
    ax.set_ylim(2.9, 3.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.legend(loc="upper right", fontsize=8.5)
    for row, (name, _, key) in enumerate(FINAL_ROWS):
        values = final_losses(name)
        right.plot(values, [row] * len(values), "o", color=c[key], markersize=5, alpha=0.75)
        right.plot([np.mean(values)] * 2, [row - 0.3, row + 0.3], color=c[key], linewidth=2.2)
    right.set_yticks(range(len(FINAL_ROWS)), [label for _, label, _ in FINAL_ROWS], fontsize=8.5)
    right.invert_yaxis()
    right.set_xlabel("final validation loss")
    _caption(fig, "Same compute per token, more parameters",
             f"mean over English, Russian and Python; left: mean of {len(seeds('Dense'))} seeds; right: each seed "
             "(dots) and the mean", c)
    return fig


def busiest(r):
    """The load of the busiest expert as a multiple of an even share, averaged over the layers, per record."""
    out = []
    for entry in r["load"]:
        layers = [max(layer) * len(layer) for layer in entry["layers"]]
        out.append((entry["step"], float(np.mean(layers))))
    return out


def chart_balance(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.15)
    for name, (label, key) in BALANCE_STYLE.items():
        points = busiest(run(name))
        ax.plot([s for s, _ in points], [v for _, v in points], color=c[key], linewidth=2.0, label=label)
    ax.axhline(1.0, color=c["muted"], linewidth=1, linestyle=":")
    ax.set_ylim(0.9, EXPERTS / TOP_K + 0.1)  # with k of E, an expert can get at most 1/k of the assignments
    ax.set_xlabel("training step")
    ax.set_ylabel("busiest expert / even share")
    ax.legend(loc="upper right", fontsize=9.5)
    _caption(fig, "How evenly the router spreads the tokens",
             f"{EXPERTS} experts, {TOP_K} per token, first seed, mean of 6 layers; 1 is perfect balance, "
             f"{EXPERTS // TOP_K} the most possible", c)
    return fig


AUX_LAYER = 2


def chart_aux(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.12, wspace=0.08)
    for ax, name in zip(axes, ("MoE, 8 experts, no balancing", "MoE, 8 experts"), strict=True):
        model = routing()["models"][name]
        f, prob = np.array(model["load"][AUX_LAYER]), np.array(model["mean_probs"][AUX_LAYER])
        order = np.argsort(-f)
        xs = np.arange(EXPERTS)
        ax.bar(xs - 0.2, f[order], width=0.38, color=c["enc"], label="f: share of the assignments")
        ax.bar(xs + 0.2, prob[order], width=0.38, color=c["dec"], label="P: mean probability")
        ax.axhline(1 / EXPERTS, color=c["muted"], linewidth=1, linestyle=":")
        value = EXPERTS * float((f * prob).sum())
        ax.set_title(f"{BALANCE_STYLE[name][0]}: E·Σ f·P = {value:.2f}", fontsize=10.5, color=c["ink"], loc="left")
        ax.set_xticks(xs, [f"E{e + 1}" for e in order], fontsize=8.5)
    axes[0].set_ylabel("share")
    axes[0].legend(loc="upper right", fontsize=8.5)
    _caption(fig, "The balancing loss, on two trained routers",
             f"layer {AUX_LAYER + 1}, held-out text, experts busiest first; dotted: an even share, 1/{EXPERTS}; "
             "the loss is 1 when both are even", c)
    return fig


def overall_load(model, layer):
    shares = routing()["models"][model]["shares"][layer]
    return np.mean([shares[kind] for kind in KINDS], axis=0)  # the three kinds of text are equally represented


def chart_capacity(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.15, wspace=0.3)
    layer = CAPACITY_LAYER
    for g, (name, (_, key)) in enumerate(BALANCE_STYLE.items()):
        load = np.sort(overall_load(name, layer))[::-1] * EXPERTS  # a share of the assignments; 1 = an even share
        xs = g * (EXPERTS + 2) + np.arange(EXPERTS)
        left.bar(xs, load, color=c[key], width=0.8)
    end = 3 * (EXPERTS + 2) - 2  # the right edge of the last group of bars
    for factor in (1.0, 1.25):
        left.plot([-1, end], [factor, factor], color=c["dec"], linewidth=1.2, linestyle="-" if factor == 1.0 else "--")
        left.text(end + 0.4, factor, f"c = {factor:g}", va="center", fontsize=8.5, color=c["dec"])
    left.set_xlim(-1.5, end + 5)
    left.set_xticks([g * (EXPERTS + 2) + (EXPERTS - 1) / 2 for g in range(len(BALANCE_STYLE))],
                    [label for label, _ in BALANCE_STYLE.values()], fontsize=9)
    left.tick_params(axis="x", length=0)
    left.set_ylabel("tokens per expert / even share")
    for name, (label, key) in BALANCE_STYLE.items():
        entries = routing()["models"][name]["capacity"]
        right.plot([e["factor"] for e in entries], [100 * float(np.mean(e["dropped"])) for e in entries],
                   color=c[key], marker="o", linewidth=2.0, label=label)
    right.set_xlabel("capacity factor")
    right.set_ylabel("assignments dropped, %")
    right.legend(loc="upper right", fontsize=9)
    _caption(fig, "A fixed budget per expert drops the overflow",
             f"left: the experts of layer {layer + 1}, busiest first; right: share dropped, mean of 6 layers, on "
             "held-out text", c)
    return fig


CAPACITY_LAYER = 2


def chart_specialization(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    model = routing()["models"][ROUTER_MODEL]
    colors = {"en": c["data"], "ru": c["dec"], "code": c["enc"]}
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.16)
    for layer, shares in enumerate(model["shares"]):
        traffic = np.array([shares[kind] for kind in KINDS])  # (kinds, experts): share of each kind's assignments
        composition = traffic / traffic.sum(axis=0)  # what each expert's tokens are made of
        xs = layer * (EXPERTS + 2) + np.arange(EXPERTS)
        bottom = np.zeros(EXPERTS)
        for k, kind in enumerate(KINDS):
            ax.bar(xs, composition[k], bottom=bottom, color=colors[kind], width=0.85,
                   label=KINDS[kind] if layer == 0 else None)
            bottom += composition[k]
        ax.text(xs.mean(), -0.06, f"layer {layer + 1}", ha="center", va="top", fontsize=9, color=c["muted"])
    ax.axhline(1 / 3, color=c["card"], linewidth=0.8, linestyle=":")
    ax.axhline(2 / 3, color=c["card"], linewidth=0.8, linestyle=":")
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of an expert's tokens")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=9.5, frameon=False)
    _caption(fig, "What each expert receives", f"the model with {EXPERTS} experts on held-out text, equal amounts of "
             "each kind; a bar split in thirds means no preference", c)
    return fig


def chart_decode(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.09, right=0.7, top=0.8, bottom=0.15)
    batches = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    for (name, experts, top_k), key in zip(DECODE_MODELS, ("muted", "data", "dec", "att"), strict=True):
        label = f"{name}: {top_k} of {experts}"
        if top_k / experts == FINE_TOP_K / FINE_EXPERTS:
            label += f",\nour model: {FINE_TOP_K} of {FINE_EXPERTS}"
        ax.plot(batches, 1 - (1 - top_k / experts) ** batches, color=c[key], linewidth=2.0, label=label)
    measured = routing()["models"]["MoE, 32 experts"]["distinct"]
    ax.plot([m["batch"] for m in measured], [np.mean(m["layers"]) / FINE_EXPERTS for m in measured], color=c["enc"],
            linestyle="none", marker="o", markersize=5, label="our model, measured")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.minorticks_off()
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("sequences decoded at once")
    ax.set_ylabel("share of experts needed")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    _caption(fig, "One decoding step reads more experts as the batch grows",
             "lines: if the router spread tokens evenly; dots: our trained router, on held-out text", c)
    return fig


# the share of experts one token uses, k/E, is all the curve depends on: gpt-oss-120b's 4 of 128 would lie on
# DeepSeek-V3's 8 of 256, and our model's 8 of 32 lies on Mixtral's 2 of 8
DECODE_MODELS = [("Mixtral 8x7B", 8, 2), ("Qwen3-235B-A22B", 128, 8), ("DeepSeek-V3", 256, 8), ("Kimi K2", 384, 8)]


@cache
def speed():
    return json.loads((DATA / "speed.json").read_text())


SPEED_LABELS = {"dense": "", "batched": ", sorted and batched", "loop": ", loop over experts"}


def chart_speed(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = speed()
    steps = s["steps"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.37, right=0.9, top=0.78, bottom=0.15)
    for row, entry in enumerate(steps):
        key = "enc" if entry["dispatch"] == "batched" else ("dec" if entry["dispatch"] == "loop" else "muted")
        ax.barh(row, 1e3 * entry["seconds"], color=c[key], height=0.6)
        ax.text(1e3 * entry["seconds"] * 1.02 + 5, row, f"{1e3 * entry['seconds']:.0f} ms", va="center", fontsize=9,
                color=c["muted"])
    ax.set_yticks(range(len(steps)), [e["model"] + SPEED_LABELS[e["dispatch"]] for e in steps], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1e3 * max(e["seconds"] for e in steps) * 1.18)
    ax.set_xlabel("one training step: forward, backward and optimizer, ms")
    _caption(fig, "Running the experts", f"{s['batch']} sequences of {s['length']} tokens, median of 20 steps, on the "
             f"{s['device']}", c)
    return fig


CHARTS = {
    "parameters": chart_parameters,
    "models": chart_models,
    "loss": chart_loss,
    "balance": chart_balance,
    "aux": chart_aux,
    "capacity": chart_capacity,
    "specialization": chart_specialization,
    "decode": chart_decode,
    "speed": chart_speed,
}


def chart_alts():
    alts = {}
    dense, moe = (qwen3_parameters(repo) for _, repo in PARAMETER_MODELS)
    alts["parameters"] = (
        f"Parameters of two Qwen3 models by part, from their config files. Qwen3-32B, dense: attention "
        f"{dense['attention']:.1f} billion, feed-forward networks {dense['used']:.1f} billion, embeddings "
        f"{dense['embeddings']:.1f} billion, all used for every token. Qwen3-235B-A22B, a mixture of experts: "
        f"attention {moe['attention']:.1f} billion, the experts a token uses {moe['used']:.1f} billion, the "
        f"experts it doesn't use {moe['idle']:.1f} billion, embeddings {moe['embeddings']:.1f} billion"
    )
    rows = []
    for name, repo, date, total, active in REAL_MODELS:
        experts, per_token, shared = moe_fields(repo)
        rows.append(f"{name} ({date}): {per_token} of {experts} experts" + (f" plus {shared} shared" if shared else "")
                    + f", {active:g}B of {total:,g}B parameters active")
    alts["models"] = "Routed experts per token and per layer, and the active share of parameters, in open " \
        "mixture-of-experts models in the order of release: " + "; ".join(rows)
    finals = {name: run(name)["validation"][-1] for name in LOSS_STYLE}
    alts["loss"] = (
        f"Validation loss during {runs()[0]['steps']:,} training steps, mean over English, Russian and Python, and "
        "the final loss on each: "
        + "; ".join(f"{name} {mean_validation(v):.3f} (" + ", ".join(f"{KINDS[k]} {v[k]:.2f}" for k in KINDS) + ")"
                    for name, v in finals.items())
    )
    alts["balance"] = (
        f"The busiest expert's share of the tokens as a multiple of an even share, mean of 6 layers, during training "
        f"of the model with {EXPERTS} experts: "
        + "; ".join(f"{label} {busiest(run(name))[0][1]:.2f} at the start and {busiest(run(name))[-1][1]:.2f} at the "
                    "end" for name, (label, _) in BALANCE_STYLE.items())
    )
    parts = []
    for name in ("MoE, 8 experts, no balancing", "MoE, 8 experts"):
        model = routing()["models"][name]
        f, prob = np.array(model["load"][AUX_LAYER]), np.array(model["mean_probs"][AUX_LAYER])
        parts.append(f"{BALANCE_STYLE[name][0]}: assignments from {f.min():.0%} to {f.max():.0%} per expert, mean "
                     f"probabilities from {prob.min():.0%} to {prob.max():.0%}, E times the sum of f·P "
                     f"{EXPERTS * float((f * prob).sum()):.2f}")
    alts["aux"] = ("Share of the assignments f and mean router probability P for each expert of layer "
                   f"{AUX_LAYER + 1}, on held-out text, for two trained routers. " + "; ".join(parts))
    parts = []
    for name, (label, _) in BALANCE_STYLE.items():
        load = overall_load(name, CAPACITY_LAYER) * EXPERTS
        dropped = ", ".join(f"{100 * float(np.mean(e['dropped'])):.1f}% at {e['factor']:g}"
                            for e in routing()["models"][name]["capacity"])
        parts.append(f"{label}: the busiest expert gets {load.max():.2f} times an even share; dropped {dropped}")
    alts["capacity"] = ("Tokens per expert against the capacity, and the share of assignments dropped for capacity "
                        f"factors from 1 to 2, for three routers of the model with {EXPERTS} experts. "
                        + "; ".join(parts))
    parts = []
    for layer in range(len(routing()["models"][ROUTER_MODEL]["shares"])):
        kinds = expert_kinds(ROUTER_MODEL, layer)
        leaning = [kind for kind in kinds if kind]
        parts.append(f"layer {layer + 1}: " + (", ".join(f"{leaning.count(k)} {KINDS[k]}" for k in KINDS
                                                           if k in leaning) if leaning else "none"))
    alts["specialization"] = (f"What each expert of the model with {EXPERTS} experts receives on held-out text with "
                              "equal amounts of English, Russian and Python. The middle layers are mixed; the first "
                              "and especially the last layer have experts that lean to one kind of text. Experts "
                              f"that get at least {LEANING:.0%} of their tokens from one kind, per layer: "
                              + "; ".join(parts))
    measured = routing()["models"]["MoE, 32 experts"]["distinct"]
    alts["decode"] = (
        "Expected share of the experts of a layer that one decoding step has to read, for batches of 1 to 256 "
        "sequences, if the router spreads tokens evenly: "
        + "; ".join(f"{name} ({top_k} of {experts}) {top_k / experts:.0%} at 1, "
                    f"{1 - (1 - top_k / experts) ** 16:.0%} at 16, {1 - (1 - top_k / experts) ** 256:.0%} at 256"
                    for name, experts, top_k in DECODE_MODELS)
        + f". Our trained router with {FINE_TOP_K} of {FINE_EXPERTS} experts, measured: "
        + ", ".join(f"{np.mean(m['layers']) / FINE_EXPERTS:.0%} at {m['batch']}" for m in measured
                    if m["batch"] in (1, 4, 16, 64, 256))
    )
    s = speed()
    alts["speed"] = (f"Time of one training step on the {s['device']}: "
                     + "; ".join(f"{e['model']}{SPEED_LABELS[e['dispatch']]} {1e3 * e['seconds']:.0f} ms"
                                 for e in s["steps"]))
    return alts


# ---------------------------------------------------------------- cover


def cover():
    experts, chosen = 128, {5, 19, 40, 58, 77, 90, 101, 122}  # Qwen3-235B-A22B: 8 of 128 experts per token
    out = [cv.label(cv.MOTIF_X0, 128, "128 EXPERTS, 8 PER TOKEN")]
    x0, y0, size, gap = cv.MOTIF_X0 + 4, 158, 22, 6
    for i in range(experts):
        row, column = divmod(i, 16)
        x, y = x0 + column * (size + gap), y0 + row * (size + gap)
        if i in chosen:
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="5" fill="{cv.RED}"/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="5" fill="{cv.CHIP_FILL}" '
                       f'stroke="{cv.LINE}" stroke-width="1.2"/>')
    cv.fits(x0 + 16 * (size + gap) - gap)
    legend_y = y0 + 8 * (size + gap) + 30
    out.append(f'<rect x="{x0}" y="{legend_y - 14}" width="16" height="16" rx="4" fill="{cv.RED}"/>')
    out.append(cv.text(x0 + 26, legend_y, "the experts one token uses", 16, cv.MUTED, cv.MONO))
    return cv.frame("blog / machine learning", ["Mixture", "of experts"], ["How large models use", "a small part of",
                                                                         "their weights per token"], "".join(out),
                    footer="Routers, load balancing and serving")


COVER = cover

FIGURES = {
    "moe-layer": moe_layer,
    "router": router_example,
    "dispatch": dispatch,
    "tokens": tokens,
    "fine-grained": fine_grained,
    "expert-parallel": expert_parallel,
}
