"""Static figures and cover of the second course's lesson 11, the Vision Transformer."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

VIT = "google/vit-base-patch16-224"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: from a photo to a class

HERO_ALT = ("The Vision Transformer in one line. A 224 by 224 photo is cut into a 14 by 14 grid of 16-pixel squares; "
            "each square is flattened into 768 numbers and projected to a vector of 768; a learned class token is put "
            "in front and a learned position vector is added to each of the 197 vectors; the sequence goes through "
            "twelve encoder blocks, and the class token's output vector goes to a linear classifier that names the "
            "picture")


def hero() -> Figure:
    c = config(VIT)
    p, size, d = c["patch_size"], c["image_size"], c["hidden_size"]
    grid = size // p
    fig = Figure(760, 330, alt=HERO_ALT)
    fig.text(14, 28, "From a photo to a class, as a ViT does it", kind="title")
    fig.text(14, 50, f"{size}×{size} pixels, {p}-pixel patches, vectors of {d}", kind="note")
    x0, y0, cell = 20, 86, 12
    for r in range(6):
        for col in range(6):
            fig.rect(x0 + col * cell, y0 + r * cell, cell - 2, cell - 2, role="data", tint=(r + col) % 3 != 0)
    fig.text(x0 + 36, y0 + 92, f"{grid} × {grid} patches", kind="small", anchor="middle")
    fig.arrow((96, 122), (124, 122))
    for k in range(5):
        fig.rect(130 + k * 16, 98 + k * 2, 12, 46 - 4 * k, role="data", tint=True)
    fig.text(166, 176, f"{grid * grid} × {3 * p * p}", kind="small", anchor="middle")
    fig.arrow((214, 122), (242, 122))
    fig.box(244, 102, 96, 40, role="neutral", label="E·p + b")
    fig.text(292, 162, f"{grid * grid} × {d}", kind="small", anchor="middle")
    fig.arrow((341, 122), (369, 122))
    fig.chip(372, 108, "CLS", role="dec", strong=True)
    for k in range(4):
        fig.chip(418 + k * 34, 108, "▦", role="data")
    fig.text(560, 128, "…", kind="body")
    fig.text(372, 162, "+ a learned position for each", kind="small")
    fig.text(372, 178, f"{grid * grid + 1} × {d}", kind="small")
    fig.arrow((470, 196), (470, 222))
    fig.box(330, 224, 280, 40, role="att", label=f"{c['num_hidden_layers']} encoder blocks, as in BERT",
            strong=True)
    fig.arrow((611, 244), (641, 244))
    fig.box(642, 224, 104, 40, role="enc", label="class")
    fig.text(14, 300, "only the class token's output is read; the other 196 outputs are what other tasks use",
             kind="note")
    fig.text(14, 318, "the encoder can't tell these vectors from word embeddings: the input is a sequence either way",
             kind="note")
    return fig


# ---------------------------------------------------------------- a token's embedding against a patch's

EMBED_ALT = ("How text and images become vectors, side by side. On the left, a token id picks one row of an embedding "
             "table, which is the same as multiplying the table by a one-hot vector. On the right, a patch of 16 by 16 "
             "pixels in three channels is flattened into 768 numbers and multiplied by a learned matrix, plus a bias, "
             "giving a vector of the same width as a word's embedding. Both produce one vector per token; after that "
             "the Transformer can't tell where the vectors came from")


def embedding() -> Figure:
    c = config(VIT)
    p, d = c["patch_size"], c["hidden_size"]
    fig = Figure(760, 330, alt=EMBED_ALT)
    fig.text(14, 28, "A word and a patch become vectors the same way", kind="title")
    fig.text(14, 50, "a lookup is a matrix times a one-hot vector; a patch is a matrix times its pixels", kind="note")
    fig.text(20, 86, "text", kind="subtitle")
    fig.chip(20, 100, "token 1,723", role="data")
    fig.arrow((140, 115), (168, 115))
    for r in range(7):
        fig.rect(176, 92 + r * 12, 150, 9, role="enc" if r == 3 else "neutral", tint=r != 3)
    fig.text(251, 186, "embedding table, vocabulary × D", kind="small", anchor="middle")
    fig.arrow((330, 131), (358, 131))
    fig.rect(362, 126, 150, 10, role="enc")
    fig.text(437, 152, "row 1,723 = table · one-hot", kind="small", anchor="middle")
    fig.text(20, 222, "image", kind="subtitle")
    for r in range(4):
        for col in range(4):
            fig.rect(20 + col * 11, 236 + r * 11, 9, 9, role="data", tint=(r * col) % 2 == 0)
    fig.text(42, 296, f"{p}×{p}×3", kind="small", anchor="middle")
    fig.arrow((70, 258), (98, 258))
    for k in range(3):
        role = ["dec", "enc", "att"][k]
        fig.rect(104 + k * 50, 252, 48, 10, role=role, tint=True)
    fig.text(178, 280, f"{3 * p * p} pixels: R, G, B", kind="small", anchor="middle")
    fig.arrow((258, 257), (286, 257))
    fig.box(290, 238, 120, 38, role="neutral", label="E · p + b")
    fig.text(350, 296, f"E is {d} × {3 * p * p}", kind="small", anchor="middle")
    fig.arrow((411, 257), (439, 257))
    fig.rect(443, 252, 150, 10, role="enc")
    fig.text(518, 280, f"a vector of {d}", kind="small", anchor="middle")
    fig.line((610, 131), (640, 131), (640, 257), (610, 257))
    fig.arrow((640, 194), (664, 194))
    fig.box(668, 176, 80, 36, role="att", label="encoder")
    fig.text(708, 228, "same input", kind="small", anchor="middle")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("ViT-B/16's passport. Published October 2020 by Google Research, Brain Team, weights under Apache 2.0. "
                "Type: encoder-only Transformer for images, 86 million parameters. Input: 224 by 224 pixels cut into 196 "
                "patches of 16 by 16, plus a class token. Encoder: 12 blocks of width 768 with 12 heads and an MLP of "
                "3,072, layer norm before each part. Positions: 197 learned vectors. Training: ImageNet-21k, 14 million "
                "images, then ImageNet-1k. Good at: classifying photos and serving as a backbone for other tasks. Not "
                "for: small datasets trained from scratch")


def passport() -> Figure:
    c = config(VIT)
    p, size = c["patch_size"], c["image_size"]
    grid = size // p
    rows = [
        ("published", "October 2020, Google Research, Brain Team; Apache 2.0"),
        ("type", "encoder-only Transformer for images, 86M parameters"),
        ("input", f"{size}×{size} pixels, {grid * grid} patches of {p}×{p}, plus a class token"),
        ("encoder", f"{c['num_hidden_layers']} blocks, width {c['hidden_size']}, {c['num_attention_heads']} heads, "
                    f"MLP {c['intermediate_size']:,}"),
        ("positions", f"{grid * grid + 1} learned vectors"),
        ("training", "ImageNet-21k (14M images), then ImageNet-1k"),
        ("good at", "classifying photos; a backbone for other tasks"),
        ("not for", "small datasets trained from scratch"),
    ]
    return arch.passport("ViT-B/16", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "AN IMAGE, CUT INTO PATCHES, READ AS A SEQUENCE")]
    cell, gap = 34, 6
    for r in range(4):
        for col in range(4):
            x = cv.MOTIF_X0 + col * (cell + gap)
            y = 180 + r * (cell + gap)
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="none" '
                       f'stroke="{cv.LINE}" stroke-width="1.4"/>')
    seq_y = 380
    out.append(f'<rect x="{cv.MOTIF_X0}" y="{seq_y}" width="{cell}" height="{cell}" rx="5" fill="{cv.RED}"/>')
    for k in range(1, 12):
        x = cv.MOTIF_X0 + k * (cell + gap)
        out.append(f'<rect x="{x}" y="{seq_y}" width="{cell}" height="{cell}" rx="5" fill="none" '
                   f'stroke="{cv.LINE}" stroke-width="1.4"/>')
    cv.fits(cv.MOTIF_X0 + 12 * (cell + gap))
    out.append(cv.label(cv.MOTIF_X0, 450, "RED: THE CLASS TOKEN"))
    return cv.frame("a11_vit.ipynb", ["Vision", "Transformer"], ["Images as sequences", "of patches"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "embedding": embedding,
    "passport": passport,
    "family": lambda: arch.family(11),
}
COVERS = {"en": _cover}
