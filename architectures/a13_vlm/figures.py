"""Static figures and cover of the second course's lesson 13, vision-language models from LLaVA to Qwen-VL."""

import json
import math

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

OV = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
QV = "Qwen/Qwen2-VL-2B-Instruct"


def config(repo, name="config.json"):
    with open(hf_hub_download(repo, name)) as f:
        return json.load(f)


def photo(fig, x, y, cell=9, gap=2, n=4, role="enc"):
    """A small square of tinted cells: a photo, as the course draws images."""
    for r in range(n):
        for col in range(n):
            fig.rect(x + col * (cell + gap), y + r * (cell + gap), cell, cell, role=role, tint=(r + col) % 3 != 0)


def best_resolution(size, pinpoints):
    """LLaVA's choice of tile grid: the candidate that keeps the most of the image, then the least padding."""
    h0, w0 = size
    best, best_eff, best_waste = None, 0, float("inf")
    for h, w in pinpoints:
        scale = min(w / w0, h / h0)
        eff = min(int(w0 * scale) * int(h0 * scale), w0 * h0)
        waste = w * h - eff
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (h, w), eff, waste
    return best


def smart_resize(h, w, max_pixels, factor=28, min_pixels=56 * 56):
    """Qwen2-VL's resizing: sides rounded to multiples of 28, the area kept between the two limits."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


# ---------------------------------------------------------------- hero: an image becomes tokens

HERO_ALT = ("A vision-language model in one picture. A photo goes through a vision encoder, a Vision Transformer, and "
            "becomes a grid of patch vectors; a small projector turns each into a vector of the language model's width; "
            "these visual tokens take the place of an image placeholder in the prompt, between the text tokens, and the "
            "language model reads the whole sequence and answers in words")


def hero() -> Figure:
    c = config(OV)
    v, t = c["vision_config"], c["text_config"]
    g = v["image_size"] // v["patch_size"]
    fig = Figure(760, 330, alt=HERO_ALT)
    fig.text(14, 28, "An image becomes tokens that a language model reads", kind="title")
    fig.text(14, 50, f"LLaVA's recipe: encoder, projector, LLM; a {v['image_size']}-px tile gives {g}×{g} = {g * g} "
                     "visual tokens", kind="note")
    photo(fig, 20, 86)
    fig.arrow((66, 106), (88, 106))
    fig.box(92, 88, 132, 36, role="enc", label="vision encoder")
    fig.text(158, 142, f"width {v['hidden_size']}", kind="small", anchor="middle")
    fig.arrow((225, 106), (247, 106))
    for k in range(6):
        fig.rect(251 + k * 13, 99, 9, 14, role="enc", tint=k % 2 == 1)
    fig.text(290, 142, "patch vectors", kind="small", anchor="middle")
    fig.arrow((330, 106), (352, 106))
    fig.box(356, 88, 112, 36, role="att", label="projector")
    fig.text(476, 110, f"→ width {t['hidden_size']}", kind="small")
    fig.arrow((412, 125), (412, 166), (330, 166), (330, 188))
    y = 192
    spans = fig.row(["<|im_start|>user"], 20, y, role="data", size=12, h=28)
    x = spans[-1][0] + spans[-1][1] + 6
    for k in range(10):
        fig.rect(x + k * 13, y + 7, 9, 14, role="enc", tint=k % 2 == 1)
    x += 10 * 13 + 4
    fig.text(x - 69, y + 46, "visual tokens", kind="small", anchor="middle")
    fig.row(["What", "is", "the", "total", "?"], x + 4, y, role="data", size=12, h=28)
    fig.arrow((380, 226), (380, 248))
    fig.box(250, 252, 260, 36, role="dec", label="language model")
    fig.arrow((511, 270), (533, 270))
    fig.chip(537, 256, "60,000", role="dec", size=12, h=28)
    fig.text(14, 316, "the projector is the only part trained from scratch; encoder and LLM come pretrained", kind="note")
    return fig


# ---------------------------------------------------------------- three ways to connect an image to an LLM

DESIGNS_ALT = ("Three ways to connect a vision encoder to a language model. Cross-attention, as in Flamingo and Llama 3.2 "
               "Vision: the image never enters the token sequence; extra layers inside the language model attend to the "
               "image features. A resampler, as in BLIP-2 and the first Qwen-VL: a fixed set of learned queries reads the "
               "image and hands the language model a fixed number of tokens, 32 or 256. A projector, as in LLaVA and most "
               "models since: every patch vector becomes a token of the sequence")


def designs() -> Figure:
    fig = Figure(760, 344, alt=DESIGNS_ALT)
    fig.text(14, 28, "Three ways to give a language model an image", kind="title")
    fig.text(14, 50, "the field converged on the simplest one: put the image into the sequence", kind="note")
    cols = [(20, "cross-attention", "Flamingo, Llama 3.2 Vision"), (270, "resampler", "BLIP-2, Qwen-VL"),
            (520, "projector", "LLaVA and most since")]
    for x, name, who in cols:
        fig.text(x, 84, name, kind="subtitle")
        fig.text(x, 102, who, kind="small")
    # cross-attention: text tokens through LLM layers, the image beside them
    x = 20
    photo(fig, x, 124)
    fig.rect(x + 54, 124, 26, 40, role="enc", tint=True)
    fig.text(x + 67, 118, "features", kind="small", anchor="middle")
    for k in range(3):
        y = 124 + k * 46
        fig.box(x + 104, y, 116, 34, role="dec", label="LLM layer")
        fig.arrow((x + 81, 144), (x + 103, y + 17))
    fig.row(["a", "dog"], x + 110, 270, role="data", size=11, h=24)
    fig.text(x, 318, "image tokens: 0", kind="count")
    # resampler: learned queries compress the image
    x = 270
    photo(fig, x, 124)
    fig.arrow((x + 46, 144), (x + 66, 144))
    fig.box(x + 70, 124, 110, 40, role="att", label="queries")
    fig.text(x + 136, 218, "fixed count", kind="small")
    for k in range(4):
        fig.rect(x + 74 + k * 13, 206, 9, 14, role="enc", tint=k % 2 == 1)
    fig.arrow((x + 125, 165), (x + 125, 200))
    fig.box(x + 20, 236, 190, 34, role="dec", label="LLM")
    fig.text(x, 318, "image tokens: 32 or 256", kind="count")
    # projector: every patch becomes a token
    x = 520
    photo(fig, x, 124)
    fig.arrow((x + 46, 144), (x + 66, 144))
    fig.box(x + 70, 124, 120, 40, role="att", label="MLP")
    for k in range(12):
        fig.rect(x + 4 + k * 13, 206, 9, 14, role="enc", tint=k % 2 == 1)
    fig.arrow((x + 130, 165), (x + 130, 200))
    fig.box(x + 20, 236, 190, 34, role="dec", label="LLM")
    fig.text(x, 318, "image tokens: one per patch", kind="count")
    return fig


# ---------------------------------------------------------------- AnyRes tiling in LLaVA-OneVision

ANYRES_EXAMPLE = (1024, 768)  # height, width of the example image, a portrait photo of a receipt


def anyres() -> Figure:
    c = config(OV)
    side = c["vision_config"]["image_size"]
    g = side // c["vision_config"]["patch_size"]
    h0, w0 = ANYRES_EXAMPLE
    bh, bw = best_resolution((h0, w0), c["image_grid_pinpoints"])
    nh, nw = bh // side, bw // side
    scale = min(bw / w0, bh / h0)
    rh, rw = int(h0 * scale), int(w0 * scale)  # the resized image inside the padded canvas
    feat_h, feat_w = nh * g, nw * g
    if w0 / h0 > feat_w / feat_h:
        keep_h, keep_w = int(round(h0 * feat_w / w0, 7)), feat_w
        keep_h = feat_h - 2 * ((feat_h - keep_h) // 2)
    else:
        keep_h, keep_w = feat_h, int(round(w0 * feat_h / h0, 7))
        keep_w = feat_w - 2 * ((feat_w - keep_w) // 2)
    ratio = math.sqrt(keep_h * keep_w / (9 * g * g))
    if ratio > 1.1:
        keep_h, keep_w = int(keep_h // ratio), int(keep_w // ratio)
    total = g * g + keep_h * (keep_w + 1)
    alt = (f"How LLaVA-OneVision cuts an image into tiles. A {w0} by {h0} pixel image is fitted into the best of its "
           f"candidate grids, {nw} tiles wide and {nh} high at {side} pixels each, with a thumbnail of the whole image "
           f"beside them. Each tile gives {g} by {g} feature vectors. The tiles' features are joined into one map, the "
           f"rows and columns that only cover padding are cut away, the map is shrunk if it holds more than nine tiles' "
           f"worth, and a learned newline vector closes every row. The image costs {total} tokens: {g * g} for the "
           f"thumbnail and {keep_h} rows of {keep_w} plus a newline")
    fig = Figure(760, 362, alt=alt)
    fig.text(14, 28, "AnyRes: a high-resolution image as tiles, a thumbnail and newlines", kind="title")
    fig.text(14, 50, f"a {w0}×{h0} photo in LLaVA-OneVision: {nw}×{nh} tiles of {side} px plus a thumbnail, "
                     f"{total:,} visual tokens", kind="note")
    # the image fitted into the grid of tiles
    x0, y0, s = 20, 80, 0.16
    fig.box(x0, y0, bw * s, bh * s, role="plain")
    ox, oy = x0 + (bw - rw) / 2 * s, y0 + (bh - rh) / 2 * s
    fig.rect(ox, oy, rw * s, rh * s, role="enc", tint=True)
    for i in range(1, nw):
        fig.line((x0 + i * side * s, y0), (x0 + i * side * s, y0 + bh * s), dash=True)
    for j in range(1, nh):
        fig.line((x0, y0 + j * side * s), (x0 + bw * s, y0 + j * side * s), dash=True)
    fig.text(x0 + bw * s / 2, y0 + bh * s + 18, f"{nw}×{nh} tiles of {side} px", kind="small", anchor="middle")
    # the thumbnail
    tx = x0 + bw * s + 24
    fig.box(tx, y0, side * s, side * s, role="plain")
    photo(fig, tx + 8, y0 + 8, cell=10, gap=2, n=4)
    fig.text(tx + side * s / 2, y0 + side * s + 18, "thumbnail", kind="small", anchor="middle")
    # per tile: g x g vectors
    ax = tx + side * s + 22
    fig.arrow((ax, y0 + 40), (ax + 30, y0 + 40))
    fig.box(ax + 34, y0 + 22, 118, 36, role="enc", label="SigLIP + MLP")
    fig.text(ax + 93, y0 + 76, f"{g}×{g} vectors per tile", kind="small", anchor="middle")
    # the joined map with newlines
    mx, my = 470, 200
    cell = 12
    rows, colsn = 6, 7
    for r in range(rows):
        for q in range(colsn):
            fig.rect(mx + q * (cell + 2), my + r * (cell + 2), cell, cell, role="enc", tint=(r + q) % 2 == 0)
        fig.rect(mx + colsn * (cell + 2) + 4, my + r * (cell + 2), cell, cell, role="att")
    fig.text(mx, my - 12, "joined, unpadded map", kind="small")
    fig.text(mx + colsn * (cell + 2) + 22, my + 30, "newline", kind="small")
    fig.text(mx + colsn * (cell + 2) + 22, my + 46, "per row", kind="small")
    fig.arrow((ax + 93, y0 + 84), (ax + 93, my + 40), (mx - 8, my + 40))
    fig.text(14, 324, f"sequence = {g * g} thumbnail tokens + {keep_h} rows × ({keep_w} + 1 newline) = {total:,}",
             kind="count")
    fig.text(14, 346, "cutting the padding and capping at nine tiles' worth keep the count from exploding", kind="note")
    return fig


# ---------------------------------------------------------------- Qwen2-VL: dynamic resolution, 2x2 merging

def dynres() -> Figure:
    c = config(QV)
    p = c["vision_config"]["patch_size"]
    m = c["vision_config"]["spatial_merge_size"]
    px = p * m
    h0, w0 = 1296, 864
    budgets = [256, 1024]
    sizes = [smart_resize(h0, w0, b * px * px) for b in budgets]
    alt = (f"How Qwen2-VL sizes an image. The image keeps its proportions: its sides are rounded to multiples of {px} "
           f"pixels and the area is kept under a budget. The encoder cuts it into {p}-pixel patches, and a merger joins "
           f"every {m} by {m} block of patches into one token, so each token covers {px} by {px} pixels. A {w0} by "
           f"{h0} receipt becomes " + " or ".join(f"{w} by {h} pixels, {(h // px) * (w // px)} tokens" for h, w in sizes)
           + " under budgets of " + " and ".join(str(b) for b in budgets) + " tokens")
    fig = Figure(760, 330, alt=alt)
    fig.text(14, 28, f"Qwen2-VL: the image keeps its shape, and one token covers {px}×{px} pixels", kind="title")
    fig.text(14, 50, f"{p}-px patches, merged {m}×{m}; the budget sets how many tokens the image may cost", kind="note")
    # one token from four patches
    x0, y0 = 20, 84
    for r in range(2):
        for q in range(2):
            fig.rect(x0 + q * 26, y0 + r * 26, 22, 22, role="enc", tint=True)
    fig.text(x0, y0 + 72, f"{m}×{m} patches of {p} px", kind="small")
    fig.arrow((x0 + 56, y0 + 24), (x0 + 82, y0 + 24))
    fig.box(x0 + 86, y0 + 6, 96, 36, role="att", label="merger")
    fig.arrow((x0 + 183, y0 + 24), (x0 + 209, y0 + 24))
    fig.rect(x0 + 213, y0 + 14, 20, 20, role="enc")
    fig.text(x0 + 223, y0 + 72, "1 token", kind="small", anchor="middle")
    # the same receipt under two budgets
    bx = 320
    for k, ((h, w), b) in enumerate(zip(sizes, budgets)):
        gx, gy = bx + k * 220, 84
        cols, rows = w // px, h // px
        cell = min(150 / rows, 110 / cols)
        fig.box(gx, gy, cols * cell, rows * cell, role="plain")
        for r in range(rows):
            for q in range(cols):
                fig.rect(gx + q * cell + 0.5, gy + r * cell + 0.5, cell - 1, cell - 1, role="enc", tint=(r + q) % 2 == 1)
        fig.text(gx, gy + rows * cell + 18, f"budget {b}: {w}×{h} px", kind="small")
        fig.text(gx, gy + rows * cell + 34, f"{cols}×{rows} = {cols * rows} tokens", kind="count")
    fig.text(14, 314, f"a {w0}×{h0} receipt, the same picture: the budget trades reading detail for tokens", kind="note")
    return fig


# ---------------------------------------------------------------- M-RoPE positions

def mrope() -> Figure:
    start, gh, gw, after = 4, 3, 4, 3
    alt = (f"Qwen2-VL's positions in three numbers. Each token has a time, a height and a width position. The {start} "
           f"text tokens before the image count 0 to {start - 1} in all three. The image's {gh} by {gw} tokens keep the "
           f"time position {start} and take height positions {start} to {start + gh - 1} and width positions {start} to "
           f"{start + gw - 1}, so neighbours in the picture are neighbours in position. The text after the image resumes "
           f"at {start + max(gh, gw)}, the image's start plus its larger side, not plus its {gh * gw} tokens")
    fig = Figure(760, 300, alt=alt)
    fig.text(14, 28, "M-RoPE: every token has a time, a row and a column position", kind="title")
    fig.text(14, 50, "text counts up in all three; image tokens count along the rows and columns of the picture",
             kind="note")
    labels = ["t", "h", "w"]
    x, y0 = 70, 96
    cols = []
    for k in range(start):
        cols.append(("data", (k, k, k)))
    for r in range(gh):
        for q in range(gw):
            cols.append(("enc", (start, start + r, start + q)))
    nxt = start + max(gh, gw)
    for k in range(after):
        cols.append(("data", (nxt + k, nxt + k, nxt + k)))
    for j, lab in enumerate(labels):
        fig.text(40, y0 + 22 + j * 40, lab, kind="mono", anchor="end")
    w = 30
    for i, (role, (t, h, ww)) in enumerate(cols):
        xi = x + i * (w + 3)
        for j, v in enumerate((t, h, ww)):
            fig.box(xi, y0 + j * 40, w, 30, role=role, label=None)
            fig.text(xi + w / 2, y0 + 20 + j * 40, str(v), kind="mono", anchor="middle")
    fig.text(x + (start + gh * gw / 2) * (w + 3), y0 + 142, f"a {gh}×{gw} image", kind="small", anchor="middle")
    fig.text(x + (start + gh * gw + 1) * (w + 3), y0 + 142, f"resumes at {nxt}", kind="small", anchor="middle")
    fig.text(14, 282, f"after the image the count jumps by {max(gh, gw)}, its larger side, not by its {gh * gw} tokens",
             kind="note")
    return fig


# ---------------------------------------------------------------- FastV-style pruning

def pruning() -> Figure:
    alt = ("Pruning visual tokens inside the language model. All tokens pass through the first two layers. At layer "
           "two, the attention that the last prompt token pays to each image token ranks them; only the most-attended "
           "image tokens continue into the later layers, together with all the text tokens. The later layers do less "
           "work, and the question is how much the answer suffers")
    fig = Figure(760, 336, alt=alt)
    fig.text(14, 28, "Drop the image tokens the model ignores, after two layers", kind="title")
    fig.text(14, 50, "FastV's idea: rank image tokens by the attention they receive, keep the top share", kind="note")
    n_img, n_txt = 10, 4
    keep = {0, 2, 5, 6, 9}
    x0 = 120
    for layer in range(5):
        x = x0 + layer * 120
        fig.text(x + 40, 84, f"layer {layer + 1}" if layer < 4 else "…", kind="small", anchor="middle")
        for k in range(n_img + n_txt):
            y = 96 + k * 13
            img = k < n_img
            alive = (not img) or layer < 2 or k in keep
            if alive:
                fig.rect(x, y, 80, 9, role="enc" if img else "data", tint=img and k % 2 == 1)
    fig.text(20, 96 + n_img * 13 / 2, "image", kind="small")
    fig.text(20, 96 + (n_img + n_txt / 2) * 13, "text", kind="small")
    fig.line((x0 + 2 * 120 - 20, 90), (x0 + 2 * 120 - 20, 96 + (n_img + n_txt) * 13), dash=True)
    fig.text(x0 + 2 * 120 - 20, 96 + (n_img + n_txt) * 13 + 20, "rank and drop", kind="small", anchor="middle")
    fig.text(14, 322, "a control drops the same number of image tokens at random", kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(OV)
    v, t = c["vision_config"], c["text_config"]
    g = v["image_size"] // v["patch_size"]
    rows = [
        ("published", "August 2024, LLaVA team; Apache 2.0 (the Hugging Face port)"),
        ("type", "vision encoder + 2-layer MLP projector + decoder LLM, ~0.9B parameters"),
        ("vision", f"SigLIP so400m, {v['image_size']} px tiles, {g * g} tokens each, {v['num_hidden_layers']} blocks"),
        ("language", f"Qwen2-0.5B, {t['num_hidden_layers']} blocks of {t['hidden_size']}, "
                     f"{t['num_key_value_heads']} KV heads"),
        ("images", f"AnyRes: thumbnail + up to {len(c['image_grid_pinpoints'])} grids of tiles, newline per row"),
        ("training", "projector alignment, then single-image, then one-vision mixtures"),
        ("good at", "describing photos, short answers, small-device demos"),
        ("not for", "dense documents, counting, facts it cannot see"),
    ]
    alt = ("LLaVA-OneVision 0.5B's passport. Published August 2024 by the LLaVA team; the Hugging Face port is under "
           "Apache 2.0. A vision encoder, a two-layer MLP projector and a decoder language model, about 0.9 billion "
           f"parameters. Vision: SigLIP so400m, {v['image_size']}-pixel tiles of {g * g} tokens each. Language: Qwen2 "
           f"0.5B with {t['num_hidden_layers']} blocks. Images: AnyRes, a thumbnail plus a grid of tiles with a newline "
           "per row. Training in stages: projector alignment, single images, then a one-vision mixture. Good at "
           "describing photos and short answers on small devices. Not for dense documents, counting or facts it "
           "cannot see")
    return arch.passport("LLaVA-OneVision 0.5B", rows, alt=alt)


# ---------------------------------------------------------------- cover

def _cover():
    c = config(OV)
    g = c["vision_config"]["image_size"] // c["vision_config"]["patch_size"]
    out = [cv.label(cv.MOTIF_X0, 150, "A PHOTO, CUT INTO PATCHES")]
    cell, gap = 26, 6
    x0, y0 = cv.MOTIF_X0, 176
    for r in range(5):
        for q in range(5):
            fill = cv.RED if (r, q) in {(1, 1), (2, 3), (3, 2)} else "none"
            out.append(f'<rect x="{x0 + q * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" height="{cell}" '
                       f'rx="4" fill="{fill}" stroke="{cv.LINE}" stroke-width="1.4"/>')
    ry = y0 + 5 * (cell + gap) + 54
    out.append(cv.label(cv.MOTIF_X0, ry - 16, f"…AND READ AS {g * g} TOKENS AMONG THE WORDS"))
    svg, spans, right = cv.chip_row(["What", "is"], cv.MOTIF_X0, ry, size=18)
    out.append(svg)
    x = right + 10
    for k in range(5):
        out.append(f'<rect x="{x + k * 18}" y="{ry + 10}" width="12" height="20" rx="3" fill="{cv.RED}"/>')
    x += 5 * 18 + 8
    svg, spans, right = cv.chip_row(["?"], x, ry, size=18)
    out.append(svg)
    cv.fits(right)
    return cv.frame("a13_vlm.ipynb", ["Vision-", "language", "models"], ["Images as tokens", "for a language model"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "designs": designs,
    "anyres": anyres,
    "dynres": dynres,
    "mrope": mrope,
    "pruning": pruning,
    "passport": passport,
    "family": lambda: arch.family(13),
}
COVERS = {"en": _cover}
