"""Static figures and cover of the second course's lesson 12, CLIP and SigLIP."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

CLIP = "openai/clip-vit-base-patch32"
SIGLIP = "google/siglip-base-patch16-224"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


def photo(fig, x, y, cell=9, gap=2, n=4):
    """A small square of tinted cells: a photo, as the course draws images."""
    for r in range(n):
        for col in range(n):
            fig.rect(x + col * (cell + gap), y + r * (cell + gap), cell, cell, role="enc", tint=(r + col) % 3 != 0)


# ---------------------------------------------------------------- hero: two towers and the matrix of pairs

HERO_ALT = ("CLIP in one picture. A photo goes through an image tower, a Vision Transformer, and becomes a vector; its "
            "caption goes through a text tower, a Transformer, and becomes a vector of the same length. A batch of four "
            "photos and four captions gives a four by four table of cosine similarities; the four cells on the "
            "diagonal are the pairs that belong together, and training raises them and lowers the other twelve")


def hero() -> Figure:
    c = config(CLIP)
    d = c["projection_dim"]
    fig = Figure(760, 374, alt=HERO_ALT)
    fig.text(14, 28, "Two encoders and one space for images and text", kind="title")
    fig.text(14, 50, f"each photo and each caption becomes a vector of {d} numbers; a batch is scored pair by pair",
             kind="note")
    photo(fig, 20, 92)
    fig.arrow((66, 112), (92, 112))
    fig.box(96, 94, 140, 36, role="enc", label="image tower")
    fig.arrow((237, 112), (263, 112))
    fig.rect(267, 107, 96, 10, role="enc")
    fig.text(315, 140, f"image vector, {d}", kind="small", anchor="middle")
    fig.chip(20, 186, "a fox in snow", role="data", size=11, h=26)
    fig.arrow((132, 199), (158, 199))
    fig.box(162, 181, 120, 36, role="enc", label="text tower")
    fig.arrow((283, 199), (309, 199))
    fig.rect(313, 194, 50, 10, role="data")
    fig.text(338, 230, f"text, {d}", kind="small", anchor="middle")
    x0, y0, cell, gap = 520, 112, 40, 4
    for j in range(4):
        fig.rect(x0 + j * (cell + gap), y0 - 22, cell, 8, role="data")
    for i in range(4):
        fig.rect(x0 - 22, y0 + i * (cell + gap), 8, cell, role="enc")
        for j in range(4):
            if i == j:
                fig.box(x0 + j * (cell + gap), y0 + i * (cell + gap), cell, cell, role="dec", strong=True, r=4)
            else:
                fig.box(x0 + j * (cell + gap), y0 + i * (cell + gap), cell, cell, role="neutral", r=4)
    fig.text(x0 + 86, 76, "4 captions", kind="small", anchor="middle")
    fig.text(x0 - 22, y0 + 196, "4 photos", kind="small", anchor="start")
    fig.line((380, 112), (470, 150))
    fig.line((380, 199), (470, 160))
    fig.text(424, 100, "cosine", kind="small", anchor="middle")
    fig.text(424, 228, "of every pair", kind="small", anchor="middle")
    fig.text(14, 334, "the diagonal holds the pairs that belong together; training raises it and lowers the rest",
             kind="note")
    fig.text(14, 356, "after training, a class name is just another caption, and a search query is too", kind="note")
    return fig


# ---------------------------------------------------------------- a caption through the text tower

TEXT_ALT = ("A caption through CLIP's text tower, step by step. The caption is lower-cased and cut into byte-pair tokens "
            "between a start token and an end token; each token gets its embedding plus a learned position; twelve "
            "Transformer blocks with a causal mask let each token see only the tokens before it, so only the end token "
            "has seen the whole caption; its output vector, after a final layer norm and a linear projection, is the "
            "caption's vector, divided by its length")


def text_tower() -> Figure:
    c = config(CLIP)
    t = c["text_config"]
    d, layers, heads, pos, proj = (t["hidden_size"], t["num_hidden_layers"], t["num_attention_heads"],
                                   t["max_position_embeddings"], c["projection_dim"])
    fig = Figure(760, 372, alt=TEXT_ALT)
    fig.text(14, 28, "A caption through CLIP's text tower", kind="title")
    fig.text(14, 50, f"width {d}, {layers} blocks of {heads} heads, at most {pos} tokens", kind="note")
    tokens = ["<start>", "a", "fox", "in", "the", "snow", "<end>"]
    spans = fig.row(tokens, 20, 74, role="data", strong=lambda i, s: s == "<end>", size=12, h=28)
    fig.text(20, 124, "1  lower-cased byte-pair tokens between two markers", kind="small")
    for x, w in spans:
        fig.rect(x + 4, 136, w - 8, 9, role="data", tint=True)
    fig.text(20, 166, "2  each token's embedding plus a learned position", kind="small")
    fig.box(20, 180, 380, 40, role="enc", label=f"{layers} blocks, causal attention")
    n = 7
    for r in range(n):
        for col in range(r + 1):
            fig.rect(440 + col * 13, 82 + r * 13, 11, 11, role="att", tint=col != r)
    fig.text(440, 190, "3  the causal mask: each token", kind="small")
    fig.text(440, 208, "sees itself and what came before", kind="small")
    end_x = spans[-1][0] + spans[-1][1] / 2
    fig.arrow((end_x, 221), (end_x, 262))
    fig.box(end_x - 70, 266, 140, 32, role="dec", label="<end>'s output", strong=True)
    fig.arrow((end_x + 71, 282), (end_x + 101, 282))
    fig.box(end_x + 105, 266, 150, 32, role="neutral", label=f"norm, {d} → {proj}")
    fig.arrow((end_x + 256, 282), (end_x + 284, 282))
    fig.rect(end_x + 288, 277, 70, 10, role="data")
    fig.text(20, 326, "4  only the end token has read the whole caption: its vector is the caption's vector", kind="small")
    fig.text(20, 346, "5  divided by its length, it lands on a sphere, where similarity is a dot product", kind="small")
    return fig


# ---------------------------------------------------------------- softmax against sigmoid

LOSS_ALT = ("Two losses on the same table of pairs. On the left, CLIP's: each row of the table is a softmax over all "
            "the captions, asking which one caption belongs to this photo, and each column is a softmax over all the "
            "photos; every row needs every column. On the right, SigLIP's: each cell on its own is a yes-or-no "
            "question, the diagonal cells should say yes and the others no, through a sigmoid of a scaled similarity "
            "plus a bias; cells don't depend on each other, so the table can be computed in pieces")


def losses() -> Figure:
    fig = Figure(760, 352, alt=LOSS_ALT)
    fig.text(14, 28, "The same table of pairs, two questions about it", kind="title")
    fig.text(14, 50, "CLIP asks each row and each column for one answer; SigLIP asks every cell yes or no", kind="note")
    cell, gap = 40, 4
    for side, x0 in enumerate([40, 396]):
        y0 = 104
        for i in range(4):
            for j in range(4):
                role = "dec" if i == j else "neutral"
                fig.box(x0 + j * (cell + gap), y0 + i * (cell + gap), cell, cell, role=role, strong=i == j, r=4)
                if side == 1 and i != j:
                    fig.text(x0 + j * (cell + gap) + cell / 2, y0 + i * (cell + gap) + 25, "+1" if i == j else "−1",
                             kind="small", anchor="middle")
    fig.text(40, 88, "CLIP, softmax", kind="subtitle")
    fig.rect(36, 102 + (cell + gap), 4 * (cell + gap) + 4, 2, role="att")
    fig.rect(36, 104 + 2 * (cell + gap) - 2, 4 * (cell + gap) + 4, 2, role="att")
    fig.text(230, 150, "one row: which caption?", kind="small")
    fig.text(230, 168, "a softmax over 4", kind="small")
    fig.rect(40 + 2 * (cell + gap) - 2, 100, 2, 4 * (cell + gap) + 4, role="att")
    fig.rect(40 + 3 * (cell + gap) - 4, 100, 2, 4 * (cell + gap) + 4, role="att")
    fig.text(230, 226, "one column: which photo?", kind="small")
    fig.text(230, 244, "another softmax", kind="small")
    fig.text(396, 88, "SigLIP, sigmoid", kind="subtitle")
    fig.text(588, 150, "each cell on its own:", kind="small")
    fig.text(588, 168, "σ(t · cos + b)", kind="small")
    fig.text(588, 226, "+1 on the diagonal,", kind="small")
    fig.text(588, 244, "−1 everywhere else", kind="small")
    fig.text(14, 310, "a softmax row needs every caption in the batch before it can be normalised", kind="note")
    fig.text(14, 330, "a sigmoid cell needs one photo and one caption, so the table can be filled in blocks", kind="note")
    return fig


# ---------------------------------------------------------------- a classifier made of sentences

ZERO_ALT = ("Zero-shot classification with CLIP. Each class name is put into a sentence, a photo i took in Iceland, in "
            "Japan, in Italy, in Brazil, and the text tower turns each sentence into a vector; together the vectors "
            "form the weight matrix of a classifier. A photo goes through the image tower; its vector's dot product "
            "with each row is the score of that class, and the highest score is the answer. No photo of any of these "
            "countries was labelled for this")


def zero_shot() -> Figure:
    fig = Figure(760, 330, alt=ZERO_ALT)
    fig.text(14, 28, "A classifier made of sentences", kind="title")
    fig.text(14, 50, "the rows of the weight matrix are the text tower's vectors for one sentence per class",
             kind="note")
    names = ["Iceland", "Japan", "Italy", "Brazil"]
    for k, name in enumerate(names):
        y = 84 + k * 44
        fig.chip(20, y, f"a photo i took in {name}.", role="data", size=11, h=28)
        fig.arrow((226, y + 14), (252, y + 14))
        fig.rect(256, y + 9, 120, 10, role="data")
    fig.box(256, 266, 120, 28, role="neutral", label="text tower")
    fig.text(316, 312, "one row per class", kind="small", anchor="middle")
    photo(fig, 446, 112, cell=11, gap=2)
    fig.arrow((500, 136), (526, 136))
    fig.box(530, 118, 110, 36, role="enc", label="image tower")
    fig.arrow((641, 136), (667, 136))
    fig.rect(671, 131, 70, 10, role="enc")
    scores = [0.62, 0.21, 0.30, 0.12]
    for k, (name, s) in enumerate(zip(names, scores)):
        y = 196 + k * 24
        fig.text(530, y + 8, name, kind="small")
        fig.rect(600, y, 140 * s, 12, role="dec" if k == 0 else "neutral")
    fig.text(446, 312, "score of a class = image vector · its row", kind="small")
    return fig


# ---------------------------------------------------------------- the sigmoid loss in blocks

CHUNK_ALT = ("SigLIP's loss computed in blocks across three devices. Each device keeps its own photos and starts with its "
             "own captions, filling the diagonal block of the table; then the captions move one device along the "
             "ring and each device fills the next block, twice more, until every block is filled. No device ever holds "
             "the whole table, and no row has to be normalised over all the captions")


def chunks() -> Figure:
    fig = Figure(760, 372, alt=CHUNK_ALT)
    fig.text(14, 28, "The sigmoid loss, filled block by block on three devices", kind="title")
    fig.text(14, 50, "each device keeps its photos; the captions travel around the ring", kind="note")
    x0, y0, size, gap = 190, 90, 76, 8
    for dev in range(3):
        fig.text(x0 - 16, y0 + dev * (size + gap) + 36, f"device {dev + 1}", kind="small", anchor="end")
        fig.text(x0 + dev * (size + gap) + size / 2, y0 - 12, f"captions {dev + 1}", kind="small", anchor="middle")
        for col in range(3):
            step = (col - dev) % 3 + 1
            role = "dec" if step == 1 else ("att" if step == 2 else "enc")
            fig.box(x0 + col * (size + gap), y0 + dev * (size + gap), size, size, role=role, strong=step == 1,
                    label=f"step {step}")
    fig.text(476, 120, "step 1: own captions, the diagonal", kind="small")
    fig.text(476, 150, "step 2: the neighbour's captions", kind="small")
    fig.text(476, 180, "step 3: the last ones", kind="small")
    fig.text(476, 230, "each block's loss is added up at once", kind="small")
    fig.text(14, 350, "a softmax row would need the captions of all three devices before its loss exists", kind="note")
    return fig


# ---------------------------------------------------------------- where the image vector comes from

POOL_ALT = ("Two ways to get one vector out of a Vision Transformer. CLIP's image tower, like lesson 11's ViT, puts a "
            "class token in front of the patches and takes its output. SigLIP has no class token: after the blocks a "
            "single learned query, the probe, attends to all the patch outputs, and its result, through a layer norm "
            "and an MLP with a residual connection, is the image's vector")


def pooling() -> Figure:
    patch = config(SIGLIP)["vision_config"]["patch_size"]
    with open(hf_hub_download(SIGLIP, "preprocessor_config.json")) as f:
        size = json.load(f)["size"]["height"]
    grid = size // patch
    fig = Figure(760, 300, alt=POOL_ALT)
    fig.text(14, 28, "Where the image's vector comes from", kind="title")
    fig.text(14, 50, "CLIP reads a class token; SigLIP lets one learned query attend to every patch", kind="note")
    v = config(CLIP)["vision_config"]
    clip_grid = v["image_size"] // v["patch_size"]
    fig.text(20, 88, "CLIP", kind="subtitle")
    fig.chip(20, 104, "CLS", role="dec", strong=True, size=11, h=26)
    for k in range(5):
        fig.rect(70 + k * 26, 108, 20, 18, role="enc", tint=True)
    fig.text(206, 122, f"{clip_grid * clip_grid} patches", kind="small")
    fig.box(20, 148, 196, 32, role="neutral", label="12 blocks")
    fig.arrow((40, 181), (40, 210))
    fig.box(20, 214, 140, 30, role="dec", label="CLS output", strong=True)
    fig.text(20, 270, "the class token's output, normalised and projected", kind="small")
    fig.text(420, 88, "SigLIP", kind="subtitle")
    for k in range(8):
        fig.rect(420 + k * 26, 108, 20, 18, role="enc", tint=True)
    fig.text(640, 122, f"{grid * grid} patches", kind="small")
    fig.box(420, 148, 200, 32, role="neutral", label="12 blocks")
    fig.box(470, 214, 110, 30, role="dec", label="probe", strong=True)
    for k in range(8):
        fig.line((525, 214), (430 + k * 26, 181), role="att", width=1)
    fig.arrow((581, 229), (607, 229))
    fig.box(611, 214, 130, 30, role="neutral", label="norm, MLP, +")
    fig.text(420, 270, "one query: a weighted average of the patches", kind="small")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("CLIP ViT-B/32's passport. Published January 2021 by OpenAI, weights under the MIT licence. Type: two "
                "encoders, an image tower and a text tower, about 151 million parameters, trained with a contrastive "
                "loss. Image tower: a Vision Transformer, 224 by 224 pixels in 49 patches of 32, twelve blocks of width "
                "768. Text tower: twelve blocks of width 512 with eight heads, causal attention, at most 77 tokens, the "
                "end token's output. Space: 512 numbers, cosine similarity scaled by a learned temperature. Training: "
                "400 million image and text pairs collected from the internet, batches of 32,768. Good at: zero-shot "
                "classification, search, the image or text encoder of other models. Not for: counting, reading small "
                "text reliably, telling apart scenes that differ only in word order")


def passport() -> Figure:
    c = config(CLIP)
    t, v = c["text_config"], c["vision_config"]
    grid = v["image_size"] // v["patch_size"]
    rows = [
        ("published", "January 2021, OpenAI; MIT licence"),
        ("type", "an image tower and a text tower, ~151M parameters, contrastive loss"),
        ("image tower", f"ViT, {v['image_size']}×{v['image_size']}, {grid * grid} patches of {v['patch_size']}, "
                        f"{v['num_hidden_layers']} blocks of {v['hidden_size']}"),
        ("text tower", f"{t['num_hidden_layers']} causal blocks of {t['hidden_size']}, {t['num_attention_heads']} heads, "
                       f"{t['max_position_embeddings']} tokens"),
        ("space", f"{c['projection_dim']} numbers, cosine × a learned temperature"),
        ("training", "400M image–text pairs from the web, batches of 32,768"),
        ("good at", "zero-shot classification, search, encoder for other models"),
        ("not for", "counting, small text, scenes that differ in word order"),
    ]
    return arch.passport("CLIP ViT-B/32", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "A BATCH OF PHOTOS AND CAPTIONS")]
    cell, gap = 46, 8
    x0, y0 = cv.MOTIF_X0 + 40, 196
    for j in range(5):
        x = x0 + j * (cell + gap)
        out.append(f'<rect x="{x}" y="{y0 - 22}" width="{cell}" height="8" rx="2" fill="{cv.LINE}"/>')
    for i in range(5):
        y = y0 + i * (cell + gap)
        out.append(f'<rect x="{cv.MOTIF_X0}" y="{y}" width="24" height="{cell}" rx="4" fill="none" '
                   f'stroke="{cv.LINE}" stroke-width="1.4"/>')
        for j in range(5):
            x = x0 + j * (cell + gap)
            if i == j:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="{cv.RED}"/>')
            else:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="none" '
                           f'stroke="{cv.LINE}" stroke-width="1.4"/>')
    cv.fits(x0 + 5 * (cell + gap))
    out.append(cv.label(cv.MOTIF_X0, y0 + 5 * (cell + gap) + 26, "RED: THE PAIRS THAT BELONG TOGETHER"))
    return cv.frame("a12_clip.ipynb", ["CLIP and", "SigLIP"], ["Images and text", "in one space"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "text-tower": text_tower,
    "losses": losses,
    "zero-shot": zero_shot,
    "chunks": chunks,
    "pooling": pooling,
    "passport": passport,
    "family": lambda: arch.family(12),
}
COVERS = {"en": _cover}
