"""Static figures and covers of lesson 8.

The numbers in them are computed here: the fan-out curve from its formula, the multiply-adds of a decoding step
from the published model's config, the coordinated-omission toy and the batching timeline from fixed seeds.
"""

import json
import math
import random

from tools.figures import cover as cv
from tools.figures.lib import Figure

MODEL = "xmarva/perevod-en-ru"
REVISION = "4abc0d63e862ca99d2bd99cd1007e4a9fe47c358"


# ---------------------------------------------------------------- hero: the tail at scale


def hero() -> Figure:
    slow = 0.01  # each server is slow one time in a hundred
    fig = Figure(
        760,
        330,
        alt="The share of slow requests against the number of servers a request waits for, when each server is "
        "slow one time in a hundred: 1% for one server, about 10% for ten, 63% for a hundred. The slowest part of a "
        "system decides how often its users wait",
    )
    fig.text(14, 28, "One server in a hundred is slow; how often is a request?", kind="title")
    fig.text(14, 50, "share of requests that wait for at least one slow server: 1 − 0.99ⁿ", kind="note")
    x0, x1, y0, y1 = 70, 720, 280, 80

    def px(n):
        return x0 + (x1 - x0) * math.log10(n) / math.log10(200)

    def py(share):
        return y0 + (y1 - y0) * share

    fig.line((x0, y0), (x1, y0), role="muted", width=1)
    fig.line((x0, y0), (x0, y1), role="muted", width=1)
    for share in [0.25, 0.5, 0.75, 1.0]:
        fig.line((x0, py(share)), (x1, py(share)), role="muted", width=0.5, dash=True)
        fig.text(x0 - 8, py(share) + 4, f"{share:.0%}", kind="small", anchor="end")
    for n in [1, 10, 100]:
        fig.text(px(n), y0 + 18, str(n), kind="small", anchor="middle")
    fig.text(x1, y0 + 18, "servers", kind="small", anchor="end")
    points = []
    for k in range(0, 101):
        n = 10 ** (k / 100 * math.log10(200))
        points.append((px(n), py(1 - (1 - slow) ** n)))
    for a, b in zip(points, points[1:], strict=False):
        fig.line(a, b, role="dec", width=2.2)
    for n, dx in [(1, 10), (10, 10), (100, -16)]:
        share = 1 - (1 - slow) ** n
        fig.rect(px(n) - 4, py(share) - 4, 8, 8, role="dec", r=4)
        anchor = "start" if dx > 0 else "end"
        fig.text(px(n) + dx, py(share) - 8, f"{share:.0%}", kind="count_hot", anchor=anchor)
    fig.text(14, 316, "Dean & Barroso (2013): with 100 servers, 63% of requests take as long as the slowest 1%.",
             kind="note")
    return fig


# ---------------------------------------------------------------- one decoding step, without and with a cache


def kv_step() -> Figure:
    fig = Figure(
        760,
        300,
        alt="Decoding step 5 without and with a cache. Without one, the decoder runs on all five tokens and "
        "recomputes the keys and values of the first four. With one, it runs on the fifth token only, appends its "
        "key and value to the cache, and attends to all five",
    )
    fig.text(14, 28, "Decoding step 5: what the decoder computes", kind="title")
    for k, (name, recompute) in enumerate([("without a cache", True), ("with a KV cache", False)]):
        y = 76 + k * 112
        fig.text(14, y + 24, name, kind="subtitle")
        for t in range(5):
            x = 200 + t * 104
            new = t == 4
            if recompute or new:
                fig.box(x, y, 92, 40, role="dec" if new else "data", label=f"K,V of {t + 1}")
            else:
                fig.box(x, y, 92, 40, role="plain", label=f"cached {t + 1}")
        cost = "5 positions, again" if recompute else "1 position, the rest kept"
        fig.text(200, y + 64, cost, kind="label")
    fig.text(14, 290, "Step t costs t positions without the cache, one with it: T²/2 against T for a whole sentence.",
             kind="note")
    return fig


# ---------------------------------------------------------------- beam search reorders the cache


def beam_reorder() -> Figure:
    fig = Figure(
        760,
        330,
        alt="Beam search with a cache, beam size three. After a step, the three best continuations come from rows "
        "0, 0 and 2 of the cache: row 1's hypothesis died and row 0's had two children. The cache is gathered with "
        "the index [0, 0, 2] so that each new beam's row holds its parent's keys and values",
    )
    fig.text(14, 28, "After each step, the cache follows the surviving hypotheses", kind="title")
    old = ["Я думаю", "Я думал", "Мне кажется"]
    new = [("Я думаю ,", 0), ("Я думаю ,", 0), ("Мне кажется ,", 2)]
    fig.text(14, 62, "cache rows before", kind="label")
    fig.text(470, 62, "beams after the step", kind="label")
    for row, text in enumerate(old):
        y = 78 + row * 62
        fig.box(14, y, 250, 42, role="enc" if row != 1 else "plain", label=f"row {row}: {text}")
    labels = ["Я думаю, что", "Я думаю, это", "Мне кажется, что"]
    for k, (_, parent) in enumerate(new):
        y = 78 + k * 62
        fig.box(470, y, 276, 42, role="enc", label=labels[k])
        fig.arrow((268, 99 + parent * 62), (466, y + 21), role="att")
    fig.text(14, 286, "row 1's hypothesis fell out of the beam; row 0 had two children", kind="note")
    fig.text(14, 312, "cache = cache[[0, 0, 2]]: one gather per step, before the next decoder call", kind="mono")
    return fig


# ---------------------------------------------------------------- multiply-adds of one cached decoding step


def _config():
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(MODEL, "config.json", revision=REVISION), encoding="utf-8") as f:
        return json.load(f)


def step_flops() -> Figure:
    config = _config()
    d, layers, vocab = config["d_model"], config["n_layers"], config["vocab_size"]
    width = 8 * round(2 * config["d_ff"] / 3 / 8)  # SwiGLU's width, as in model.py
    prefix, source = 20, 15  # a step in the middle of an everyday sentence
    parts = [
        ("output projection onto the vocabulary", d * vocab, False),
        ("feed-forward, 3 matrices per layer", layers * 3 * d * width, True),
        ("self-attention q, k, v, o", layers * 4 * d * d, True),
        ("cross-attention q, o", layers * 2 * d * d, True),
        ("attention scores and sums", layers * 2 * d * (prefix + source), False),
    ]
    total = sum(value for _, value, _ in parts)
    fig = Figure(
        760,
        96 + 44 * len(parts),
        alt="Multiply-adds of one decoding step of the published model: the output projection onto 16,000 tokens "
        "is the largest single part, larger than all attention projections together; dynamic quantization converts "
        "only the nn.Linear layers and leaves the output projection, which reuses the embedding matrix, in float32",
    )
    fig.text(14, 28, f"One decoding step: {total / 1e6:.1f} million multiply-adds, by where they go", kind="title")
    fig.text(14, 50, f"d_model {d}, {layers} decoder layers, {vocab:,} tokens; the 20th token of a "
             f"translation of {source} source tokens", kind="note")
    top = max(value for _, value, _ in parts)
    for k, (name, value, linear) in enumerate(parts):
        y = 74 + 44 * k
        fig.text(14, y + 20, name, kind="body")
        w = 300 * value / top
        fig.rect(330, y + 6, w, 22, role="data" if linear else "dec", tint=False, r=3)
        fig.text(330 + w + 8, y + 22, f"{value / total:.0%}", kind="count")
    fig.rect(14, 96 + 44 * len(parts) - 26, 14, 14, role="data", r=2)
    fig.text(34, 96 + 44 * len(parts) - 14, "nn.Linear: int8 after quantize_dynamic", kind="label")
    fig.rect(380, 96 + 44 * len(parts) - 26, 14, 14, role="dec", r=2)
    fig.text(400, 96 + 44 * len(parts) - 14, "not nn.Linear: stays float32", kind="label")
    return fig


# ---------------------------------------------------------------- the service


def service() -> Figure:
    fig = Figure(
        760,
        250,
        alt="How a request travels through perevod's service: an HTTP request reaches uvicorn and FastAPI, which "
        "validates it and puts it into a queue; a batcher takes the requests that arrived within a few milliseconds, "
        "translates them as one batch in a worker thread, and answers each request with its own translation",
    )
    fig.text(14, 28, "A request's way through perevod-server", kind="title")
    steps = [("POST /translate", "plain"), ("FastAPI checks", "data"), ("queue", "att"),
             ("batcher: ≤16, 5 ms", "att"), ("model, a thread", "enc")]
    w, gap = 132, 18
    for k, (name, role) in enumerate(steps):
        x = 14 + k * (w + gap)
        fig.box(x, 80, w, 44, role=role, label=name)
        if k < len(steps) - 1:
            fig.arrow((x + w + 2, 102), (x + w + gap - 2, 102), role="muted")
    last = 14 + (len(steps) - 1) * (w + gap) + w / 2
    fig.arrow((last, 128), (last, 168), (80, 168), (80, 128), role="enc")
    fig.text(150, 192, "each request gets its own translation back, from its own future", kind="note")
    fig.text(14, 232, "The model loads once, when the server starts; requests share it through the queue.",
             kind="note")
    return fig


# ---------------------------------------------------------------- coordinated omission, a toy


def omission() -> Figure:
    period, stall_start, stall = 0.1, 1.0, 1.0  # a user every 100 ms; the server freezes for a second
    service_time = 0.01
    # open loop: everyone arrives on schedule; those arriving in the stall wait until it ends
    open_loop = []
    for k in range(40):
        at = k * period
        if stall_start <= at < stall_start + stall:
            open_loop.append(stall_start + stall - at + service_time)
        else:
            open_loop.append(service_time)
    # closed loop: one tester that waits for each answer sends nothing during the stall
    closed_loop, t = [], 0.0
    while t < 4.0:
        if stall_start <= t < stall_start + stall:
            closed_loop.append(stall_start + stall - t + service_time)
            t = stall_start + stall + service_time
        else:
            closed_loop.append(service_time)
            t += period
    fig = Figure(
        760,
        320,
        alt="A server stalls for one second. Users arriving at their own pace keep sending during the stall, and a "
        "quarter of their requests are slow. A closed-loop tester waits for its answer and sends nothing during the "
        "stall, so it records one slow request in thirty-two and reports a service that is almost never slow",
    )
    fig.text(14, 28, "A one-second stall, seen by two testers", kind="title")
    x0, scale = 220, 130
    for k, (name, values, role) in enumerate([("users on their own schedule", open_loop, "dec"),
                                              ("a tester that waits for answers", closed_loop, "data")]):
        y = 80 + k * 100
        fig.text(14, y + 6, name, kind="body")
        slow = sum(1 for v in values if v > 0.05)
        fig.text(14, y + 26, f"{slow} of {len(values)} requests slow, {slow / len(values):.0%}", kind="label")
        fig.line((x0, y + 40), (x0 + 4 * scale, y + 40), role="muted", width=1)
        fig.rect(x0 + stall_start * scale, y, stall * scale, 40, role="plain", tint=True, r=2)
        times, t = [], 0.0
        if values is open_loop:
            times = [k2 * period for k2 in range(40)]
        else:
            t = 0.0
            for value in values:
                times.append(t)
                t = t + value if value > 0.05 else t + period
        for at, value in zip(times, values, strict=False):
            if at < 4.0:
                h = 8 if value <= 0.05 else 30
                fig.rect(x0 + at * scale - 1.5, y + 40 - h, 3, h, role=role, r=1)
    fig.text(x0 + stall_start * scale, 64, "stall", kind="small")
    fig.text(14, 300, "The waiting tester sends nothing while the server is stuck, so the stall barely shows.",
             kind="note")
    return fig


# ---------------------------------------------------------------- dynamic batching, a timeline


def batching() -> Figure:
    rng = random.Random(3)
    arrivals, t = [], 0.0
    while t < 1.0:
        t += rng.expovariate(22)
        if t < 1.0:
            arrivals.append(t)
    single, batch_size_cost = 0.08, lambda n: 0.08 + 0.012 * (n - 1)
    # one at a time
    one, free = [], 0.0
    for at in arrivals:
        start = max(at, free)
        free = start + single
        one.append((start, free))
    # batched: take everything waiting when the server frees up
    batches, free, k = [], 0.0, 0
    while k < len(arrivals):
        start = max(arrivals[k], free)
        members = [i for i in range(k, len(arrivals)) if arrivals[i] <= start]
        end = start + batch_size_cost(len(members))
        batches.append((start, end, len(members)))
        free, k = end, members[-1] + 1
    fig = Figure(
        760,
        300,
        alt="Requests arriving over time, one at a time and with dynamic batching. One at a time, each request "
        "waits for all those before it. With dynamic batching, the requests that arrive while a batch is being "
        "translated go into the next batch together, so the queue empties in a few batches instead of many single "
        "translations",
    )
    fig.text(14, 28, f"{len(arrivals)} requests in one second, one at a time and in batches", kind="title")
    x0, horizon = 160, 2.0
    scale = (746 - x0) / horizon
    fig.text(14, 82, "arrivals", kind="label")
    for at in arrivals:
        fig.rect(x0 + at * scale - 1.5, 70, 3, 16, role="data", r=1)
    fig.text(14, 140, "one at a time", kind="label")
    for start, end in one:
        if start < horizon:
            width = max(2, (min(end, horizon) - start) * scale - 2)
            fig.rect(x0 + start * scale, 124, width, 24, role="dec", tint=True, r=2)
    fig.text(14, 200, "batched", kind="label")
    for start, end, n in batches:
        if start >= horizon:
            continue
        end = min(end, horizon)
        fig.rect(x0 + start * scale, 184, (end - start) * scale - 2, 24, role="enc", tint=True, r=2)
        if (end - start) * scale > 22:
            fig.text(x0 + (start + end) / 2 * scale, 201, str(n), kind="small", anchor="middle")
    last_one = one[-1][1]
    last_batch = batches[-1][1]
    fig.text(14, 260, f"the last request is answered at {last_one:.2f} s one at a time, at {last_batch:.2f} s "
             "in batches", kind="note")
    fig.text(14, 284, "toy numbers: 80 ms per translation, 12 ms more per extra sentence in a batch; "
             "the axis runs to 2 s", kind="small")
    return fig


# ---------------------------------------------------------------- cover: the hockey stick


def _cover(lang):
    english = lang == "en"
    out = [cv.label(cv.MOTIF_X0, 128, "LATENCY AGAINST LOAD" if english else "ЗАДЕРЖКА ПРОТИВ НАГРУЗКИ")]
    x0, x1, y0, y1 = cv.MOTIF_X0 + 10, cv.MOTIF_X1 - 10, 500, 170
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{cv.LINE}" stroke-width="1.4"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{cv.LINE}" stroke-width="1.4"/>')
    top_rate = 2.8  # batching measured about 2.5 times the capacity of one request at a time
    for capacity, colour, width in [(1.0, cv.MUTED, 2.5), (2.5, cv.RED, 4)]:
        points = []
        for k in range(400):
            rate = top_rate * k / 400
            latency = 1 / (1 - rate / capacity) if rate < capacity else float("inf")
            if latency > 12:  # the curve leaves the top of the plot
                break
            points.append((x0 + (x1 - x0) * rate / top_rate, y0 - (y0 - y1) * latency / 12))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="{width}"/>')
    wall = x0 + (x1 - x0) / top_rate  # where one request at a time runs out of capacity
    out.append(cv.text(wall + 14, y1 + 40, "one at a time" if english else "по одному", 18, cv.MUTED, cv.MONO))
    out.append(cv.text(x1 - 4, y1 + 90, "batched" if english else "батчами", 18, cv.RED, cv.MONO, anchor="end"))
    out.append(cv.text(x1, y0 + 30, "requests / s" if english else "запросов / с", 16, cv.FAINT, cv.MONO,
                       anchor="end"))
    title = ["Serving under", "load"] if english else ["Сервис под", "нагрузкой"]
    hook = ["Where do the", "milliseconds go?"] if english else ["Куда уходят", "миллисекунды?"]
    return cv.frame("08_serving.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "kv-step": kv_step,
    "beam-reorder": beam_reorder,
    "step-flops": step_flops,
    "service": service,
    "omission": omission,
    "batching": batching,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
