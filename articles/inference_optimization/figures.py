"""Figures, charts and the cover of the post "LLM inference optimization" on xmarva.github.io.

Facts about real models are read at render time: layer and head counts from each model's config.json on the
Hugging Face Hub (the Llama repositories are ungated mirrors with Meta's configs), parameter counts from the
Hub's safetensors metadata. Measurements come from data/: t4.json (experiment.py, a Kaggle T4) and
speculative.json (speculative.py, greedy speculative decoding on a CPU). GPU specifications are NVIDIA's
datasheet numbers, linked where they are defined. Toy schedules (batching, paging) are defined here and every
count they show is computed from them.

    uv run python tools/render.py --article inference_optimization --out <site>/assets/posts/inference-optimization
"""

import json
from functools import cache
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import Box, Figure, Rect

DATA = Path(__file__).parent / "data"
KiB, GiB = 2**10, 2**30


@cache
def read_config(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        return json.load(f)


@cache
def parameter_count(repo):
    from huggingface_hub import model_info

    return model_info(repo).safetensors.total


def kv_bytes_per_token(config, bytes_per_value=2):
    layers = config["num_hidden_layers"]
    if "kv_lora_rank" in config:  # MLA: one compressed vector, plus the part of the key that carries positions
        per_layer = config["kv_lora_rank"] + config["qk_rope_head_dim"]
    else:
        heads = config["num_attention_heads"]
        kv_heads = config.get("num_key_value_heads", heads)
        head_dim = config.get("head_dim") or config["hidden_size"] // heads
        per_layer = 2 * kv_heads * head_dim
    return layers * per_layer * bytes_per_value


def full_kv_bytes_per_token(config, bytes_per_value=2):
    """What an MLA model would cache with ordinary attention: a full key and value for every head."""
    key = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    return config["num_hidden_layers"] * config["num_attention_heads"] * (key + config["v_head_dim"]) * bytes_per_value


def attention_kind(config):
    if "kv_lora_rank" in config:
        return f"MLA, rank {config['kv_lora_rank']}"
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    if kv_heads == heads:
        return f"MHA, {heads} heads"
    return f"GQA, {kv_heads} of {heads}"


def size(n_bytes):
    if n_bytes >= GiB:
        return f"{n_bytes / GiB:.1f} GiB"
    if n_bytes >= 10 * KiB:
        return f"{n_bytes / KiB:,.0f} KiB"
    return f"{n_bytes / KiB:.1f} KiB"


# ---------------------------------------------------------------- the KV cache of real models

KV_MODELS = [
    ("Llama 2 7B", "NousResearch/Llama-2-7b-hf"),
    ("Mistral 7B v0.3", "mistralai/Mistral-7B-v0.3"),
    ("Llama 3.1 8B", "unsloth/Meta-Llama-3.1-8B"),
    ("Qwen2.5 7B", "Qwen/Qwen2.5-7B-Instruct"),
    ("Llama 3.1 70B", "unsloth/Meta-Llama-3.1-70B"),
    ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
]
CONVERSATION = 32_768


def kv_per_token() -> Figure:
    rows = []
    for name, repo in KV_MODELS:
        config = read_config(repo)
        rows.append((name, attention_kind(config), kv_bytes_per_token(config), False))
        if "kv_lora_rank" in config:
            rows.append(("  without MLA", "MHA, 128 heads", full_kv_bytes_per_token(config), True))
    fig = Figure(
        760,
        96 + 40 * len(rows),
        alt="KV cache per token in bfloat16, from each model's config: "
        + "; ".join(f"{name.strip()} {size(b)}" for name, _, b, _ in rows),
    )
    fig.text(14, 28, "How much one token costs to remember", kind="title")
    fig.text(14, 50, "KV cache per token, all layers, bfloat16; computed from each model's config.json", kind="note")
    x0, scale_max = 300, 512 * KiB
    bar_w = 260
    fig.text(x0, 80, "per token", kind="label")
    fig.text(746, 80, f"{CONVERSATION // 1024}K tokens", kind="label", anchor="end")
    for k, (name, kind, n_bytes, hypothetical) in enumerate(rows):
        y = 96 + 40 * k
        fig.text(14, y + 19, name.strip(), kind="note" if hypothetical else "body")
        fig.text(150, y + 19, kind, kind="label")
        w = bar_w * min(n_bytes, scale_max) / scale_max
        if hypothetical:  # far off the scale: draw it to the edge and break it
            fig.rect(x0, y + 6, bar_w, 18, role="att", tint=True, r=3)
            fig.line((x0 + bar_w - 22, y + 2), (x0 + bar_w - 30, y + 28), role="muted", width=1.4)
            fig.line((x0 + bar_w - 14, y + 2), (x0 + bar_w - 22, y + 28), role="muted", width=1.4)
            fig.text(x0 + bar_w + 8, y + 19, size(n_bytes), kind="mono")
        else:
            fig.rect(x0, y + 6, max(w, 2), 18, role="att", r=3)
            fig.text(x0 + max(w, 2) + 8, y + 19, size(n_bytes), kind="mono")
        fig.text(746, y + 19, size(n_bytes * CONVERSATION), kind="mono", anchor="end")
    return fig


# ---------------------------------------------------------------- MHA, GQA, MQA, MLA

def attention_variants() -> Figure:
    fig = Figure(
        760,
        354,
        alt="Four ways to give attention heads their keys and values. Multi-head attention: every query head has "
        "its own key and value head. Grouped-query attention: a group of query heads shares one. Multi-query "
        "attention: all heads share a single one. Multi-head latent attention: the cache holds one small latent "
        "vector per token, and every head's keys and values are computed from it",
    )
    fig.text(14, 28, "What a layer keeps for every token", kind="title")
    fig.text(14, 50, "query heads on top, keys and values below: solid ones are cached, light ones computed",
             kind="note")
    panels = [
        ("multi-head (MHA)", 8, "2·h·d", "Llama 2 7B"),
        ("grouped-query (GQA)", 2, "2·g·d, g groups", "Llama 3, Qwen2.5"),
        ("multi-query (MQA)", 1, "2·d", "PaLM, Falcon 7B"),
        ("latent (MLA)", 0, "d_c + d_r", "DeepSeek-V2, V3"),
    ]
    pw, cell, gap = 176, 14, 5
    for p, (name, kv, cached, models) in enumerate(panels):
        x0 = 14 + p * (pw + 9)
        fig.box(x0, 70, pw, 270, role="plain", r=10)
        fig.text(x0 + 12, 94, name, kind="subtitle")
        span = 8 * cell + 7 * gap
        qx = x0 + (pw - span) / 2
        queries = [qx + i * (cell + gap) for i in range(8)]
        for x in queries:
            fig.rect(x, 116, cell, cell, role="att", tint=True, r=3)
        fig.text(qx - 6, 127, "q", kind="small", anchor="end")
        if kv:
            kv_xs = []
            for j in range(kv):
                members = queries[j * 8 // kv:(j + 1) * 8 // kv]
                center = (members[0] + members[-1] + cell) / 2
                kv_xs.append(center)
                for x in members:
                    fig.line((x + cell / 2, 131), (center, 186), role="att", width=1)
                fig.rect(center - cell / 2, 188, cell, cell, role="att", r=3)
                fig.rect(center - cell / 2, 206, cell, cell, role="att", r=3)
            fig.text(qx - 6, 199, "k", kind="small", anchor="end")
            fig.text(qx - 6, 217, "v", kind="small", anchor="end")
        else:  # keys and values are rebuilt from one cached latent vector; only the vector is stored
            latent_w = 3 * cell
            lx = x0 + (pw - latent_w) / 2
            for x in queries:
                fig.line((x + cell / 2, 131), (x + cell / 2, 186), role="att", width=1)
                fig.rect(x, 188, cell, cell, role="att", tint=True, r=3)
                fig.rect(x, 206, cell, cell, role="att", tint=True, r=3)
                fig.line((x + cell / 2, 221), (lx + latent_w / 2, 236), role="muted", width=1, dash=True)
            fig.rect(lx, 238, latent_w, cell, role="att", r=3)
            fig.text(qx - 6, 199, "k", kind="small", anchor="end")
            fig.text(qx - 6, 217, "v", kind="small", anchor="end")
            fig.text(lx - 6, 249, "c", kind="small", anchor="end")
        fig.text(x0 + 12, 280, "cached per layer", kind="small")
        fig.text(x0 + 12, 300, cached, kind="mono_att")
        fig.text(x0 + 12, 324, models, kind="small")
    return fig


# ---------------------------------------------------------------- paging the KV cache

BLOCK = 4  # tokens per block in the drawing; vLLM's default is 16
MAX_LEN = 16
PAGED = {"A": (10, [5, 2, 7]), "B": (7, [1, 4])}  # request: (tokens so far, physical blocks of its block table)
PHYSICAL_BLOCKS = 8


def paged() -> Figure:
    reserved_waste = sum(MAX_LEN - n for n, _ in PAGED.values())
    paged_waste = sum(len(blocks) * BLOCK - n for n, blocks in PAGED.values())
    for n, blocks in PAGED.values():
        assert len(blocks) == -(-n // BLOCK), "a block table holds exactly the blocks its tokens need"
    fig = Figure(
        760,
        444,
        alt=f"Two requests, A with {PAGED['A'][0]} tokens and B with {PAGED['B'][0]}. Reserving the maximum "
        f"length of {MAX_LEN} for each leaves {reserved_waste} slots unused. With blocks of {BLOCK} tokens, each "
        f"request's block table points to blocks anywhere in memory, and only {paged_waste} slots are unused, in "
        "the last block of each request",
    )
    fig.text(14, 28, "Reserving the maximum vs allocating pages", kind="title")
    fig.text(14, 50, f"each cell is one token's keys and values in every layer; toy blocks of {BLOCK} tokens",
             kind="note")
    c = 24

    def cells(x, y, request, n_used, n_total, start=0):
        for i in range(n_total):
            used = i < n_used
            fig.els.append(_cell(x + i * c, y, c - 3, request, start + i if used else None))

    legend = [("A", 0, "token of A"), ("B", 0, "token of B"), ("A", None, "reserved, empty"), (None, None, "free")]
    x = 300
    for request, position, name in legend:
        cell = _cell(x, 410, 16, request, position)
        if isinstance(cell, Box):
            cell.label = None
        fig.els.append(cell)
        fig.text(x + 22, 423, name, kind="small")
        x += 22 + len(name) * 6.6 + 18
    fig.text(14, 92, f"contiguous, {MAX_LEN} slots reserved per request", kind="subtitle")
    for k, (request, (n, _)) in enumerate(PAGED.items()):
        y = 108 + k * 34
        fig.text(14, y + 16, request, kind="mono")
        cells(40, y, request, n, MAX_LEN)
        fig.text(40 + MAX_LEN * c + 8, y + 16, f"{MAX_LEN - n} reserved, unused", kind="label")
    fig.text(14, 196, f"{reserved_waste} of {MAX_LEN * len(PAGED)} slots hold nothing", kind="count_hot")

    fig.text(14, 240, f"paged, blocks of {BLOCK} tokens", kind="subtitle")
    fig.text(14, 266, "block tables", kind="label")
    for k, (request, (_, blocks)) in enumerate(PAGED.items()):
        y = 280 + k * 40
        fig.text(14, y + 17, request, kind="mono")
        for j, b in enumerate(blocks):
            fig.box(40 + j * 34, y, 30, 26, role="plain", label=str(b), r=5)
    fig.text(14, 380, f"{paged_waste} slots unused, all in last blocks", kind="count_hot")
    owner = {}
    for request, (_, blocks) in PAGED.items():
        for j, b in enumerate(blocks):
            owner[b] = (request, j)
    fig.text(300, 266, "GPU memory, physical blocks", kind="label")
    per_row = PHYSICAL_BLOCKS // 2
    for b in range(PHYSICAL_BLOCKS):
        x = 300 + (b % per_row) * (BLOCK * c + 18)
        y = 290 + (b // per_row) * 62
        fig.text(x, y - 6, f"#{b}", kind="small")
        if b in owner:
            request, j = owner[b]
            n = PAGED[request][0]
            used = min(BLOCK, n - j * BLOCK)
            for i in range(BLOCK):
                fig.els.append(_cell(x + i * c, y, c - 3, request, j * BLOCK + i if i < used else None))
        else:
            for i in range(BLOCK):
                fig.els.append(_cell(x + i * c, y, c - 3, None, None))
    return fig


def _cell(x, y, w, request, position):
    """A token slot: the request's token number; an outline if reserved for it but empty; faint if free."""
    if request is None:
        return Rect(x, y, w, w, "neutral", tint=True, r=4)
    if position is None:
        return Box(x, y, w, w, "plain", None, r=4)
    return Box(x, y, w, w, "data", str(position), mono=True, strong=request == "A", r=4, size=11)


# ---------------------------------------------------------------- static and continuous batching

OUTPUT_LENGTHS = [7, 3, 9, 4, 6, 2, 8, 5]  # tokens each request generates, in order of arrival
SLOTS = 4


def static_schedule():
    """Requests go in groups of SLOTS; a group leaves the GPU when its longest request is done."""
    rows, t = [[] for _ in range(SLOTS)], 0
    for start in range(0, len(OUTPUT_LENGTHS), SLOTS):
        group = list(range(start, min(start + SLOTS, len(OUTPUT_LENGTHS))))
        for slot, r in enumerate(group):
            rows[slot].append((r, t, OUTPUT_LENGTHS[r]))
        t += max(OUTPUT_LENGTHS[r] for r in group)
    return rows, t


def continuous_schedule():
    """After every step, a slot whose request has finished takes the next one from the queue."""
    rows, free_at = [[] for _ in range(SLOTS)], [0] * SLOTS
    for r, n in enumerate(OUTPUT_LENGTHS):
        slot = free_at.index(min(free_at))
        rows[slot].append((r, free_at[slot], n))
        free_at[slot] += n
    return rows, max(free_at)


def batching() -> Figure:
    static_rows, static_steps = static_schedule()
    cont_rows, cont_steps = continuous_schedule()
    work = sum(OUTPUT_LENGTHS)
    fig = Figure(
        760,
        360,
        alt=f"Eight requests of different lengths on four slots. Static batching takes {static_steps} steps, "
        f"because each group waits for its longest request; continuous batching refills a slot as soon as a "
        f"request finishes and takes {cont_steps} steps for the same {work} tokens",
    )
    fig.text(14, 28, "Static and continuous batching", kind="title")
    fig.text(14, 50, f"{len(OUTPUT_LENGTHS)} requests, {SLOTS} slots; a cell is one decoding step of one request",
             kind="note")
    c = 32
    x0 = 92
    letters = "ABCDEFGHIJKLMNOP"

    def draw(rows, steps, y0, title):
        fig.text(14, y0 - 12, title, kind="subtitle")
        idle = SLOTS * steps - work
        fig.text(746, y0 - 12, f"{steps} steps, {idle} slot-steps idle", kind="count_hot", anchor="end")
        for slot, row in enumerate(rows):
            y = y0 + slot * 26
            fig.text(14, y + 16, f"slot {slot + 1}", kind="label")
            fig.rect(x0, y, steps * c - 3, 22, role="neutral", tint=True, r=4)
            for r, start, n in row:
                fig.box(x0 + start * c, y, n * c - 3, 22, role="data", r=4)
                fig.els.append(Box(x0 + start * c, y, c - 3, 22, "data", letters[r], mono=True, strong=True, r=4))

    draw(static_rows, static_steps, 92, "static: a group waits for its longest request")
    draw(cont_rows, cont_steps, 236, "continuous: a free slot takes the next request")
    assert x0 + static_steps * c <= 746
    return fig


# ---------------------------------------------------------------- one request, to scale

def phases() -> Figure:
    d = t4()
    prompt, answer = 1024, 64
    prefill = [p for p in d["prefill"] if p["tokens"] == prompt][0]["seconds"]
    steps = sorted(d["generation"]["with_cache"])
    step = steps[len(steps) // 2]
    total = prefill + (answer - 1) * step
    fig = Figure(
        760,
        262,
        alt=f"A request on a T4, drawn to scale: reading the {prompt}-token prompt takes {prefill * 1e3:.0f} ms and "
        f"yields the first token; each of the other {answer - 1} tokens takes one decoding step of "
        f"{step * 1e3:.0f} ms, {total:.1f} s in all",
    )
    fig.text(14, 28, "One request, to scale", kind="title")
    fig.text(14, 50, f"Qwen2.5-1.5B on a T4, transformers run eagerly: a {prompt:,}-token prompt, "
             f"a {answer}-token answer", kind="note")
    x0, x1 = 14, 746
    scale = (x1 - x0) / total
    y = 104
    fig.box(x0, y, prefill * scale - 2, 34, role="data", r=5, strong=True)
    x = x0 + prefill * scale
    for _ in range(answer - 1):
        fig.rect(x, y, step * scale - 2, 34, role="data", tint=True, r=2)
        x += step * scale
    fig.text(x0, y - 12, "prefill", kind="label", role="data")
    fig.text(x0 + prefill * scale + 4, y - 12, "decode: one token per step", kind="label")
    fig.line((x0, 150), (x0, 160), (x0 + prefill * scale - 2, 160), (x0 + prefill * scale - 2, 150), width=1.2)
    fig.text(x0, 180, f"time to first token {prefill * 1e3:.0f} ms", kind="mono")
    k = x0 + prefill * scale + 30 * step * scale
    fig.line((k, 150), (k, 160), (k + step * scale - 2, 160), (k + step * scale - 2, 150), width=1.2)
    fig.text(k, 180, f"time per output token {step * 1e3:.0f} ms", kind="mono")
    for t in range(int(total) + 1):
        tx = x0 + t * scale
        fig.line((tx, 204), (tx, 210), width=1)
        fig.text(tx, 226, f"{t} s", kind="small", anchor="start" if t == 0 else "middle")
    fig.line((x0, 204), (x1, 204), width=1)
    fig.text(14, 252, f"the prompt is read at {prompt / prefill:,.0f} tokens/s, the answer written at "
             f"{1 / step:.0f} tokens/s", kind="note")
    return fig


# ---------------------------------------------------------------- what fits next to the weights

H100_BYTES, H100_BANDWIDTH = 80 * GiB, 3.35e12
ENGINE_SHARE = 0.9  # vLLM's gpu_memory_utilization default
BUDGETS = [  # (model, repository, bytes per parameter, GPUs)
    ("Llama 3.1 8B", "unsloth/Meta-Llama-3.1-8B", 2, 1),
    ("Llama 3.1 70B", "unsloth/Meta-Llama-3.1-70B", 2, 2),
    ("Llama 3.1 70B", "unsloth/Meta-Llama-3.1-70B", 1, 2),
    ("Llama 3.1 70B", "unsloth/Meta-Llama-3.1-70B", 0.5, 1),
]
CONVERSATION_8K = 8192


def memory_budget() -> Figure:
    rows = []
    for name, repo, bytes_per_param, gpus in BUDGETS:
        weights = parameter_count(repo) * bytes_per_param
        budget = ENGINE_SHARE * gpus * H100_BYTES
        room = budget - weights
        tokens = room / kv_bytes_per_token(read_config(repo))
        ceiling = gpus * H100_BANDWIDTH / weights
        rows.append((name, bytes_per_param, gpus, weights, room, tokens, ceiling))
    fig = Figure(
        760,
        124 + 66 * len(rows),
        alt="Memory of H100 GPUs given to the engine: "
        + "; ".join(f"{n}, {int(8 * b)}-bit weights on {g} GPU{'s' if g > 1 else ''}: weights {w / GiB:.0f} GiB, "
                    f"room for {t / 1000:.0f} thousand tokens of KV cache, at most {c:.0f} tokens/s for one sequence"
                    for n, b, g, w, _, t, c in rows),
    )
    fig.text(14, 28, "What fits next to the weights", kind="title")
    fig.text(14, 50, f"H100 80GB, {ENGINE_SHARE:.0%} of it for the engine; bf16 KV cache; activations and "
             "quantization scales left out", kind="note")
    x0, gpu_w, gap = 14, 194, 8
    px = gpu_w / H100_BYTES
    fig.text(446, 84, "KV cache room", kind="label")
    fig.text(746, 84, "one sequence, at most", kind="label", anchor="end")
    for k, (name, bytes_per_param, gpus, weights, room, tokens, ceiling) in enumerate(rows):
        y = 100 + 66 * k
        bits = int(8 * bytes_per_param)
        fig.text(x0, y + 14, f"{name} · {bits}-bit weights · {gpus} GPU{'s' if gpus > 1 else ''}", kind="body")
        for g in range(gpus):  # tensor parallelism: every GPU holds its share of the weights and of the cache
            x = x0 + g * (gpu_w + gap)
            fig.rect(x, y + 24, gpu_w, 20, role="neutral", tint=True, r=3)
            fig.rect(x, y + 24, weights / gpus * px, 20, role="neutral", r=3)
            fig.rect(x + weights / gpus * px + 1, y + 24, room / gpus * px - 1, 20, role="att", r=3)
        fig.text(446, y + 40, f"{tokens / 1000:,.0f}K tokens", kind="mono_att")
        fig.text(556, y + 40, f"{tokens / CONVERSATION_8K:.0f} × 8K", kind="small")
        fig.text(746, y + 40, f"{ceiling:.0f} tokens/s", kind="mono", anchor="end")
    legend = [("neutral", False, "weights"), ("att", False, "KV cache"), ("neutral", True, "kept free")]
    x = 14
    y = 100 + 66 * len(rows) + 8
    for role, tint, label in legend:
        fig.rect(x, y - 10, 14, 12, role=role, tint=tint, r=2)
        fig.text(x + 20, y, label, kind="small")
        x += 20 + len(label) * 6.6 + 18
    return fig


# ---------------------------------------------------------------- one round of speculative decoding

def speculative_round() -> Figure:
    s = speculative()
    t = s["traced"]
    accepted = t["accepted"]
    fig = Figure(
        760,
        300,
        alt=f"One round of speculative decoding with real models: the draft proposes {len(t['proposed'])} tokens, "
        f"the target agrees with the first {accepted} and replaces the next one with its own choice, "
        f"{t['target'][accepted].strip()!r}; the rest are discarded. {accepted + 1} tokens from one pass of the "
        "target",
    )
    fig.text(14, 28, "One round of speculative decoding", kind="title")
    fig.text(14, 50, f"greedy; {s['draft'].split('/')[1]} drafts, {s['target'].split('/')[1]} checks all "
             f"{s['gamma']} proposals in one pass", kind="note")
    x_label, x0 = 14, 160
    context = [w for w in t["context"] if w.strip()][-3:]
    fig.text(x_label, 98, "the text so far", kind="label")
    spans = fig.row([w.strip() for w in context], x0, 78, role="neutral")
    start = spans[-1][0] + spans[-1][1] + 10
    fig.text(x_label, 148, "draft proposes", kind="label")
    fig.text(x_label, 198, "target's choice", kind="label")
    x, discarded = start, []
    for i, word in enumerate(t["proposed"]):
        if i < accepted:
            role = "enc"
        elif i == accepted:
            role = "dec"
        else:
            role = "plain"
        w = fig.chip(x, 128, word.strip(), role=role)
        if i <= accepted:
            mine = t["target"][i].strip()
            fig.chip(x, 178, mine, role="enc", strong=i == accepted)
        else:
            discarded.append((x, w))
        x += max(w, fig.chip_width(t["target"][i].strip()) if i <= accepted else 0) + 5
    first, last = discarded[0], discarded[-1]
    fig.text((first[0] + last[0] + last[1]) / 2, 198, "discarded", kind="small", anchor="middle")
    fig.text(14, 250, f"the target agrees on {accepted} and corrects the next: {accepted + 1} tokens from one pass",
             kind="count_hot")
    fig.text(14, 274, f"over a {s['tokens']}-token answer: {s['tokens'] / len(s['rounds']):.2f} tokens per pass, "
             "and the very text the large model writes alone", kind="note")
    return fig


# ---------------------------------------------------------------- tensor parallelism

def tensor_parallel() -> Figure:
    config = read_config("unsloth/Meta-Llama-3.1-70B")
    layers, hidden = config["num_hidden_layers"], config["hidden_size"]
    fig = Figure(
        760,
        330,
        alt="A transformer layer on two GPUs, as Megatron-LM splits it: each GPU holds half of the attention heads and "
        "half of the MLP's hidden units, computes a partial result, and the two partial results are summed by an "
        f"all-reduce, twice per layer. Llama 3.1 70B has {layers} layers: {2 * layers} all-reduces per generated "
        "token",
    )
    fig.text(14, 28, "Tensor parallelism: one layer on two GPUs", kind="title")
    fig.text(14, 50, "Megatron-LM's split: every GPU holds half of every matrix; partial sums meet in an all-reduce",
             kind="note")
    lanes = [(84, "GPU 0", "heads 1 … h/2", "units 1 … f/2"), (176, "GPU 1", "heads h/2+1 … h", "units f/2+1 … f")]
    for y, gpu, heads, units in lanes:
        fig.box(14, y - 10, 732, 74, role="plain", r=10)
        fig.text(28, y + 22, gpu, kind="mono")
        fig.box(92, y + 4, 40, 34, role="data", label="x")
        fig.arrow((134, y + 21), (156, y + 21))
        fig.box(158, y + 4, 150, 34, role="att", label="attention")
        fig.text(233, y + 56, heads, kind="small", anchor="middle")
        fig.arrow((310, y + 21), (360, y + 21))
        fig.box(470, y + 4, 140, 34, role="neutral", label="MLP")
        fig.text(540, y + 56, units, kind="small", anchor="middle")
        fig.arrow((612, y + 21), (648, y + 21))
    for x in (362, 650):
        fig.box(x, 84, 82, 156, role="neutral", label="all-reduce", r=8)
    for y, *_ in lanes:
        fig.arrow((446, y + 21), (466, y + 21))
    fig.text(403, 262, "sum of both halves", kind="small", anchor="middle")
    fig.text(691, 262, "to the next layer", kind="small", anchor="middle")
    fig.text(14, 298, f"Llama 3.1 70B: {2 * layers} all-reduces per token, {hidden:,} values each",
             kind="count_hot")
    fig.text(14, 320, "the GPUs wait for each other at every one of them: inside one server, over NVLink", kind="note")
    return fig


# ---------------------------------------------------------------- chunked prefill

def chunked_prefill() -> Figure:
    fig = Figure(
        760,
        300,
        alt="Three requests are decoding when a long prompt arrives. Run whole, its prefill is one long step, and "
        "the three wait for it without producing a token. Split into chunks, each step carries one token of each "
        "running request and a chunk of the prompt, and they keep producing tokens",
    )
    fig.text(14, 28, "A long prompt arrives: one long step, or chunks", kind="title")
    fig.text(14, 50, "schematic; each column is one step of the engine, its width the step's time", kind="note")

    def timeline(y0, title, steps):
        fig.text(14, y0 - 12, title, kind="subtitle")
        x = 110
        for kind, w in steps:
            for r, name in enumerate(("A", "B", "C")):
                y = y0 + r * 26
                if kind == "prefill":
                    continue
                fig.box(x, y, 26, 22, role="data", label=name, r=4)
            if kind in ("prefill", "chunk"):
                fig.box(x + (30 if kind == "chunk" else 0), y0, w - (30 if kind == "chunk" else 0) - 4, 74,
                        role="att", label="prompt" if kind == "prefill" else "¼", r=6)
            x += w
        for r, name in enumerate(("A", "B", "C")):
            fig.text(14, y0 + r * 26 + 16, f"request {name}", kind="label")
        return x

    timeline(92, "whole: A, B and C produce nothing while the prompt is read",
                   [("decode", 32), ("decode", 32), ("prefill", 300), ("decode", 32), ("decode", 32)])
    timeline(206, "chunked: every step has a token for A, B and C",
             [("decode", 32), ("chunk", 92), ("chunk", 92), ("chunk", 92), ("chunk", 92), ("decode", 32)])
    return fig


# ---------------------------------------------------------------- the serving path

def serving() -> Figure:
    fig = Figure(
        760,
        330,
        alt="The path of a request: the client, a reverse proxy that must not buffer the stream, the API server "
        "that authenticates, rate-limits and tokenizes, a queue with admission control, and the engine whose "
        "scheduler runs the GPU step by step over a pool of KV cache blocks. Tokens flow back as server-sent "
        "events. Under each stage, the numbers to watch",
    )
    fig.text(14, 28, "The path of one request", kind="title")
    fig.text(14, 50, "request to the right, tokens back to the left, one server-sent event at a time", kind="note")
    stages = [
        ("client", ["time to first token", "as the user sees it"]),
        ("proxy", ["no buffering", "long read timeout"]),
        ("API server", ["auth, rate limits", "tokenizer, 429s"]),
        ("queue", ["requests waiting", "time in the queue"]),
        ("engine", ["KV cache in use", "preemptions", "time per token"]),
    ]
    w, gap = 128, 21
    for k, (name, notes) in enumerate(stages):
        x = 14 + k * (w + gap)
        fig.box(x, 96, w, 76, role="att" if name == "engine" else "plain", label=name, r=10)
        for j, note in enumerate(notes):
            fig.text(x + 4, 214 + 20 * j, note, kind="small")
        if k:
            fig.arrow((x - gap + 2, 118), (x - 4, 118), role="data")  # the request
            fig.arrow((x - 2, 152), (x - gap + 4, 152), role="enc")
    fig.text(14, 194, "what to watch, what to set", kind="label")
    fig.text(14, 300, "request", kind="label", role="data")
    fig.text(90, 300, "tokens, streamed", kind="label", role="enc")
    fig.line((70, 296), (84, 296), role="data", width=2)
    fig.line((210, 296), (224, 296), role="enc", width=2)
    return fig


# ---------------------------------------------------------------- charts of the measurements

# NVIDIA datasheets: dense 16-bit tensor throughput and memory bandwidth. The A100 and H100 sheets print the
# throughput with sparsity, twice the dense figure; the T4 has no sparsity mode.
GPUS = {  # name: (FLOP/s, bytes/s, datasheet)
    "T4": (65e12, 320e9, "https://www.nvidia.com/en-us/data-center/tesla-t4/"),
    "A100 SXM": (312e12, 2039e9, "https://www.nvidia.com/en-us/data-center/a100/"),
    "H100 SXM": (1979e12 / 2, 3.35e12, "https://www.nvidia.com/en-us/data-center/h100/"),
}


@cache
def t4():
    return json.loads((DATA / "t4.json").read_text(encoding="utf-8"))


@cache
def speculative():
    return json.loads((DATA / "speculative.json").read_text(encoding="utf-8"))


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, 0.895, note, ha="left", va="top", fontsize=10, color=c["muted"])


def _log2_batches(ax, batches):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.minorticks_off()


def chart_utilization(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = t4()
    peak = GPUS["T4"][0]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.2, right=0.92, top=0.76, bottom=0.11)
    rows = []
    for m in d["monitor"]:
        achieved = 2 * m["batch"] * d["matmul_params"] / m["seconds"]
        sequences = "1 sequence" if m["batch"] == 1 else f"{m['batch']:,} sequences"
        rows.append((sequences, "nvidia-smi, busy", m["mean"]["utilization.gpu"], c["line"]))
        rows.append((sequences, "math done", 100 * achieved / peak, c["enc"]))
    positions = [3.2, 2.6, 1.0, 0.4]
    for y, (_, _, value, color) in zip(positions, rows, strict=True):
        ax.barh(y, value, height=0.46, color=color)
        label = f"{value:.0f}%" if value >= 10 else f"{value:.1f}%"
        ax.text(value + 1.2, y, label, va="center", fontsize=10.5, color=c["ink"], fontweight="medium")
    ax.set_yticks(positions, [name for _, name, _, _ in rows])
    for y, sequences in ((3.62, rows[0][0]), (1.42, rows[2][0])):
        ax.text(0, y, f"decoding {sequences} at once", ha="left", va="bottom", fontsize=10.5,
                fontweight="semibold", color=c["ink"])
    ax.set_ylim(0.05, 4.05)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _caption(fig, "Busy is not the same as working",
             "Qwen2.5-1.5B decoding on a Kaggle T4. Math done: FLOP/s achieved, as a share of the T4's 65 TFLOPS",
             c)
    return fig


def chart_decode_batch(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = t4()
    flops, bandwidth, _ = GPUS["T4"]
    batches = [w["batch"] for w in d["weights"]]
    ms = [1e3 * w["seconds"] for w in d["weights"]]
    limit = [1e3 * max(d["matmul_bytes"] / bandwidth, 2 * b * d["matmul_params"] / flops) for b in batches]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.16, wspace=0.28)
    left.plot(batches, limit, color=c["muted"], linestyle="--", linewidth=1.4)
    left.plot(batches, ms, color=c["enc"], marker="o", markersize=5)
    left.set_yscale("log")
    left.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300 ms"])
    left.minorticks_off()
    left.text(1.4, limit[0] * 0.9, "datasheet limit", fontsize=9.5, color=c["muted"], va="top")
    left.set_ylim(6, 400)
    left.set_title("time of one step", fontsize=11)
    eager = [e for e in d["eager"] if e["batch"] == 1][0]
    tokens = [b / (t / 1e3) for b, t in zip(batches, ms, strict=True)]
    right.plot(batches, tokens, color=c["enc"], marker="o", markersize=5)
    right.plot([1], [1 / eager["seconds"]], color=c["dec"], marker="o", markersize=7, linestyle="none")
    right.text(1.5, 1 / eager["seconds"], "the same step, run eagerly", fontsize=9.5, color=c["dec_ink"],
               va="center")
    right.set_yscale("log")
    right.set_yticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
    right.minorticks_off()
    right.set_ylim(10, 20000)
    right.set_title("tokens per second", fontsize=11)
    for ax in (left, right):
        _log2_batches(ax, [1, 4, 16, 64, 256, 1024])
        ax.set_xlabel("sequences in the batch")
    _caption(fig, "A batch of 64 costs a little more than a batch of 1",
             "one decoding step of Qwen2.5-1.5B on a T4: every weight matrix applied to one new token per sequence",
             c)
    return fig


def chart_roofline(theme):
    import matplotlib.pyplot as plt
    import numpy as np

    c = _style(theme)
    d = t4()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)
    x = np.logspace(-0.3, 3.6, 200)
    colors = {"T4": c["enc"], "A100 SXM": c["muted"], "H100 SXM": c["ink"]}
    for name, (flops, bandwidth, _) in GPUS.items():
        ax.plot(x, np.minimum(flops, bandwidth * x) / 1e12, color=colors[name], linewidth=1.8)
        ridge = flops / bandwidth
        ax.plot([ridge], [flops / 1e12], marker="o", markersize=5, color=colors[name])
        ax.text(x[-1], flops / 1e12 * 1.12, name, ha="right", va="bottom", fontsize=10, color=c["ink"])
        ax.text(ridge * 1.08, flops / 1e12 * 0.8, f"{ridge:.0f} FLOP/byte", ha="left", va="top", fontsize=9,
                color=c["muted"])
    batches = [w["batch"] for w in d["weights"]]
    achieved = [2 * b * d["matmul_params"] / w["seconds"] / 1e12 for b, w in zip(batches, d["weights"], strict=True)]
    intensity = [2 * b * d["matmul_params"] / d["matmul_bytes"] for b in batches]
    ax.plot(intensity, achieved, linestyle="none", marker="o", markersize=6, color=c["enc"],
            markeredgecolor=c["card"], markeredgewidth=1.2)
    ax.annotate("measured on the T4,\nbatch 1 to 2,048", (intensity[8], achieved[8]), xytext=(320, 2.2),
                fontsize=9.5, color=c["muted"], arrowprops={"arrowstyle": "-", "color": c["muted"], "lw": 0.8})
    ax.axvline(1, color=c["dec"], linewidth=1.2, linestyle="--")
    ax.text(1.08, 1400, "decoding one sequence:\n1 FLOP per byte", fontsize=9.5, color=c["dec_ink"], va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.1, 2500)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("FLOPs per byte of weights read (16-bit weights: the batch size)")
    ax.set_ylabel("TFLOP/s")
    ax.grid(axis="x")
    _caption(fig, "The roofline: memory on the slope, compute on the flat",
             "attainable throughput = min(peak FLOP/s, bandwidth × FLOPs per byte); datasheet numbers, dense 16-bit",
             c)
    return fig


def chart_cache(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    g = t4()["generation"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    fig.subplots_adjust(left=0.09, right=0.8, top=0.74, bottom=0.15)
    steps = range(1, len(g["with_cache"]) + 1)
    for key, color, name in (("without_cache", c["dec"], "without the cache"), ("with_cache", c["enc"], "with it")):
        ms = [1e3 * t for t in g[key]]
        ax.plot(steps, ms, color=color, linewidth=1.4)
        tail = sorted(ms[-20:])[10]
        ax.text(len(ms) + 18, tail, f"{name}\n{sum(g[key]):.0f} s in total", va="center", fontsize=10,
                color=c["ink"])
    ax.set_ylim(0, None)
    ax.set_xlim(0, len(g["with_cache"]))
    ax.set_xlabel("token being generated")
    ax.set_ylabel("ms per token")
    _caption(fig, "Without the KV cache, every token rereads the whole text",
             f"Qwen2.5-1.5B on a T4, transformers run eagerly, a {len(g['with_cache']):,}-token answer", c)
    return fig


def chart_context(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = t4()
    batch = 16
    weights = [w for w in d["weights"] if w["batch"] == batch][0]["seconds"] * 1e3
    rows = [a for a in d["attention"] if a["fused"]]
    labels = [f"{a['context'] // 1024}K" if a["context"] >= 1024 else str(a["context"]) for a in rows]
    attention = [1e3 * a["seconds"] for a in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.72, bottom=0.15)
    xs = range(len(rows))
    ax.bar(xs, [weights] * len(rows), width=0.6, color=c["line"], label="weights")
    ax.bar(xs, attention, width=0.6, bottom=[weights] * len(rows), color=c["att"], label="KV cache")
    for x, a in zip(xs, attention, strict=True):
        ax.text(x, weights + a + 1.2, f"{weights + a:.0f} ms", ha="center", va="bottom", fontsize=10,
                color=c["ink"])
    ax.set_xticks(list(xs), labels)
    ax.set_xlabel("tokens in each sequence's context")
    ax.set_ylabel("ms per step")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, (weights + attention[-1]) * 1.15)
    _caption(fig, "Long contexts: the cache outweighs the model",
             f"one decoding step for {batch} sequences on a T4: reading the weights, then each sequence's KV cache", c)
    return fig


def chart_speculative(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb

    c = _style(theme)
    s = speculative()
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.09, right=0.84, top=0.74, bottom=0.15)
    gammas = list(range(1, 11))
    card, att = to_rgb(c["card"]), to_rgb(c["att"])
    for alpha, mix in ((0.6, 0.5), (round(s["alpha"], 2), 1.0), (0.9, 0.5)):
        color = to_hex([k + (a - k) * mix for k, a in zip(card, att, strict=True)])
        expected = [(1 - alpha ** (g + 1)) / (1 - alpha) for g in gammas]
        ax.plot(gammas, expected, color=color, marker="o", markersize=4)
        ax.text(gammas[-1] + 0.3, expected[-1], f"α = {alpha:.2f}", va="center", fontsize=10, color=c["ink"])
    measured = s["tokens"] / len(s["rounds"])
    ax.plot([s["gamma"]], [measured], marker="D", markersize=8, color=c["dec"], linestyle="none")
    ax.text(s["gamma"] + 0.35, measured - 0.32, f"measured: {measured:.2f} tokens per pass", fontsize=9.5,
            color=c["dec_ink"], va="center")
    ax.set_xticks(gammas)
    ax.set_xlabel("γ, tokens the draft proposes per round")
    ax.set_ylabel("tokens per pass of the large model")
    ax.set_ylim(0, None)
    _caption(fig, "How much a draft model buys",
             "expected tokens per target pass, (1 − α^(γ+1)) / (1 − α); the point: Qwen2.5-0.5B drafting for 1.5B", c)
    return fig


# ---------------------------------------------------------------- cover

def cover():
    import math

    d = t4()
    out = [cv.label(cv.MOTIF_X0, 128, "ONE DECODING STEP ON A T4, 1 TO 2,048 SEQUENCES")]
    x0, x1, y0, y1 = cv.MOTIF_X0 + 4, cv.MOTIF_X1 - 8, 160, 320
    batches = [w["batch"] for w in d["weights"]]
    ms = [1e3 * w["seconds"] for w in d["weights"]]
    low, high = math.log(min(ms) * 0.8), math.log(max(ms))
    points = []
    for b, t in zip(batches, ms, strict=True):
        x = x0 + (x1 - x0) * math.log2(b) / math.log2(batches[-1])
        y = y1 - (y1 - y0) * (math.log(t) - low) / (high - low)
        points.append((x, y))
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{cv.LINE}" stroke-width="1"/>')
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out.append(f'<polyline points="{path}" fill="none" stroke="{cv.RED}" stroke-width="3" stroke-linejoin="round"/>')
    marked = {1: "1 sequence", 64: "64 sequences", 2048: "2,048 sequences"}
    for (x, y), b in zip(points, batches, strict=True):
        if b in marked:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{cv.RED if b == 64 else cv.BG}" '
                       f'stroke="{cv.RED}" stroke-width="2"/>')
    out.append(cv.label(cv.MOTIF_X0, 380, "THE PRICE OF A STEP"))
    for row, b in enumerate(marked):
        y = 400 + row * 52
        t = ms[batches.index(b)]
        out.append(cv.text(cv.MOTIF_X0, y + 26, marked[b], 15, cv.FAINT, cv.MONO))
        chip, w = cv.chip(cv.MOTIF_X0 + 180, y, f"{t:.0f} ms", size=17, kind="hot" if b == 64 else "plain")
        cv.fits(cv.MOTIF_X0 + 180 + w)
        out.append(chip)
    return cv.frame("blog / system design", ["Inference", "optimization"], ["Where the time and", "the memory go"],
                    "".join(out), footer="LLM serving, measured")


def chart_alts():
    """Alt text of every chart, from the same data the chart is drawn from."""
    d, s = t4(), speculative()
    flops = GPUS["T4"][0]
    monitor = {m["batch"]: m for m in d["monitor"]}
    share = {b: 100 * 2 * b * d["matmul_params"] / m["seconds"] / flops for b, m in monitor.items()}
    weights = {w["batch"]: 1e3 * w["seconds"] for w in d["weights"]}
    eager = [e for e in d["eager"] if e["batch"] == 1][0]["seconds"] * 1e3
    g = d["generation"]
    fused = [a for a in d["attention"] if a["fused"]]
    measured = s["tokens"] / len(s["rounds"])
    return {
        "utilization": f"nvidia-smi reports the T4 100% busy both when decoding one sequence and 1,024 at once; the "
        f"arithmetic actually done is {share[1]:.1f}% of the T4's 65 TFLOPS for one sequence and "
        f"{share[1024]:.0f}% for 1,024",
        "decode-batch": f"Time of one decoding step of Qwen2.5-1.5B on a T4: {weights[1]:.0f} ms for one sequence, "
        f"{weights[64]:.0f} ms for 64, then growing in proportion to the batch up to {weights[2048]:.0f} ms for "
        f"2,048; tokens per second rise from {1e3 / weights[1]:.0f} to {2048e3 / weights[2048]:,.0f}. The same "
        f"step run eagerly by transformers takes {eager:.0f} ms",
        "roofline": "Roofline of the T4, A100 and H100: attainable TFLOP/s rise with FLOPs per byte up to the ridge "
        "points of 203, 153 and 295, then stay flat. Decoding one sequence sits at 1 FLOP per byte. The T4 "
        "measurements follow the slope below the datasheet line and level off near 22 TFLOP/s",
        "cache": f"Time per token while generating {len(g['with_cache']):,} tokens on a T4: flat with the KV cache, "
        f"{sum(g['with_cache']):.0f} s in total; growing steadily without it, {sum(g['without_cache']):.0f} s",
        "context": f"One decoding step for 16 sequences on a T4: reading the weights takes {weights[16]:.0f} ms at "
        f"every context length, reading the KV cache from {1e3 * fused[0]['seconds']:.1f} ms at 512 tokens to "
        f"{1e3 * fused[-1]['seconds']:.0f} ms at 16K",
        "speculative-gain": f"Expected tokens per pass of the target model against the number of drafted tokens, for "
        f"acceptance rates 0.6, {s['alpha']:.2f} and 0.9; the measured point, {measured:.2f} tokens per pass at "
        f"five drafted tokens, lies near the curve for {s['alpha']:.2f}",
    }


CHARTS = {
    "utilization": chart_utilization,
    "decode-batch": chart_decode_batch,
    "roofline": chart_roofline,
    "cache": chart_cache,
    "context": chart_context,
    "speculative-gain": chart_speculative,
}

COVER = cover

FIGURES = {
    "phases": phases,
    "memory-budget": memory_budget,
    "speculative": speculative_round,
    "tensor-parallel": tensor_parallel,
    "chunked-prefill": chunked_prefill,
    "serving": serving,
    "kv-per-token": kv_per_token,
    "attention-variants": attention_variants,
    "paged": paged,
    "batching": batching,
}
