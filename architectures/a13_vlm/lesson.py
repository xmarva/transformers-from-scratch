# %% [markdown]
# # Lesson 13. How vision-language models read images, from LLaVA to Qwen2-VL
#
# Photograph a receipt, ask a chat assistant how much you spent, and it answers with the number at the bottom. The
# language model that answers learned everything it knows from text. It never saw a pixel during its training, and it
# still reads the receipt. Something turned the photo into a form it could read, and that something is the subject of
# this lesson.
#
# Five years ago, connecting images to a language model meant building a new architecture around both. In April 2023 a
# paper from Wisconsin and Microsoft showed that it takes much less: a vision encoder that already exists, a language
# model that already exists, and between them a small network trained in a few hours that turns image features into
# vectors the language model accepts as if they were words ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)). The
# model was LLaVA, and most open vision-language models since are variations on its recipe, Qwen2-VL, InternVL, Gemma 3,
# SmolVLM among them.
#
# The variations are almost all about one number. A picture becomes hundreds or thousands of tokens, and every token
# costs the language model the same compute and memory as a word. A receipt read at full detail can cost more tokens
# than this whole introduction. Everything you will see today, tiles, merged patches, positions in three dimensions,
# tokens dropped inside the model, is a way of spending that budget well.
#
# <aside class="tldr" markdown="1">
# <p class="tldr-label">In short</p>
#
# A vision-language model turns an image into vectors that the language model reads as tokens. LLaVA's recipe needs only
# a projector between a pretrained encoder and a pretrained LLM; the rest of the engineering decides how many tokens an
# image may cost and which of them to keep.
#
# <div class="stats">
# <div class="stat"><span class="stat-value">576</span><span class="stat-label">tokens per image in LLaVA-1.5, whatever its size</span></div>
# <div class="stat"><span class="stat-value">729</span><span class="stat-label">tokens per 384-pixel tile in LLaVA-OneVision, up to ten tiles' worth</span></div>
# <div class="stat"><span class="stat-value">28 px</span><span class="stat-label">the side of the square one token covers in Qwen2-VL</span></div>
# </div>
# </aside>
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/hero-light.png" width="760" alt="A vision-language model in one picture. A photo goes through a vision encoder, a Vision Transformer, and becomes a grid of patch vectors; a small projector turns each into a vector of the language model's width; these visual tokens take the place of an image placeholder in the prompt, between the text tokens, and the language model reads the whole sequence and answers in words">
#
# **What you'll do**
#
# 1. Take 95 real receipts from the CORD dataset and 100 bar charts drawn here with known values.
# 2. Follow one receipt through LLaVA-OneVision step by step: the choice of tiles, the encoder, the projector, the
#    newline tokens and the splice into the prompt, written from scratch and matched against the model's own logits.
# 3. Look at the image tokens from the language model's side: what they resemble, and what they turn into layer by
#    layer.
# 4. Price a picture: how many tokens six models spend on the same image at three sizes, and what that costs.
# 5. Rebuild Qwen2-VL's two ideas, an image that keeps its shape and positions in three dimensions, and check them
#    against the model.
# 6. Read receipts and charts with Qwen2-VL under budgets from 64 to 2,048 tokens.
# 7. Drop image tokens inside the model, as FastV does, and test whether the ones it keeps are better than a random
#    choice.
#
# **What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type → T4
# GPU*. The notebook downloads LLaVA-OneVision 0.5B (1.8 GB), Qwen2-VL 2B (4.4 GB) and the CORD test set (234 MB).

# %% tags=["include:setup"]

# %%
import io
import json
import math
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from transformers import AutoProcessor, DynamicCache
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

SMOKE = env.smoke()  # CI runs the lesson on a small slice; you run it in full
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# images teal, text amber, the language model rose, the projector violet, as in the figures
heat = LinearSegmentedColormap.from_list("heat", ["#FFFFFF", COLORS["enc"]])

# %% [markdown]
# ## 1. Receipts and charts
#
# Two kinds of image test whether a model can *read*, which is harder than recognising what a photo shows. The first is
# real: [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2), a set of about a thousand receipts photographed
# in Indonesian shops and restaurants, released by Naver Clova with every word boxed and labelled ([Park et al.,
# 2019](https://openreview.net/forum?id=SJl3z659UH)). Its test split has 100 receipts, 95 of them with a labelled total,
# and it is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which lets us show them here.
#
# The receipts are the honest test: crumpled paper, small print, photos taken at an angle, sizes from a phone's
# thumbnail to four thousand pixels tall. The question is always the same, the total to pay, and the answer is a number
# we can check exactly.

# %%
CORD = hf_hub_download(
    "naver-clova-ix/cord-v2",
    "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
    repo_type="dataset",
)
receipts = []
for index, row in pd.read_parquet(CORD).iterrows():
    truth = json.loads(row["ground_truth"])
    total = (truth["gt_parse"].get("total") or {}).get("total_price")
    if isinstance(total, list):
        total = total[0]
    if not total:
        continue  # five receipts have no labelled total
    words = [w["text"] for line in truth["valid_line"] for w in line["words"]]
    image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    receipts.append(dict(id=index, image=image, total=total, words=words))
if SMOKE:
    receipts = receipts[:6]
sizes = [r["image"].size for r in receipts]
print(len(receipts), "receipts with a total")
print("width from", min(w for w, h in sizes), "to", max(w for w, h in sizes), "pixels")
print("height from", min(h for w, h in sizes), "to", max(h for w, h in sizes), "pixels")

# %% [markdown]
# The second kind we make ourselves: bar charts with between five and ten bars, each bar's value written above it in a
# small font, and a question about one of them. Drawing the charts here means we know every answer, we can draw as many
# as we like, and no model has seen them in training.

# %%
CITIES = [
    "Lisbon",
    "Oslo",
    "Quito",
    "Hanoi",
    "Accra",
    "Lima",
    "Tbilisi",
    "Bergen",
    "Porto",
    "Kyoto",
    "Cusco",
    "Dakar",
]


def make_chart(seed):
    rng = random.Random(seed)
    labels = rng.sample(CITIES, rng.randint(5, 10))
    values = [rng.randint(12, 980) for _ in labels]
    with plt.style.context("default"):  # an ordinary chart, not the course's style
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        bars = ax.bar(labels, values, color="#4c72b0")
        for bar, value in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(
                x,
                value,
                str(value),
                ha="center",
                va="bottom",
                fontsize=rng.choice([7, 8, 9, 10]),
            )
        ax.set_title(f"Visitors per day, thousands (sample {seed})", fontsize=11)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
    target = rng.randrange(len(labels))
    image = Image.open(buffer).convert("RGB")
    return dict(image=image, label=labels[target], value=str(values[target]))


charts = [make_chart(seed) for seed in range(6 if SMOKE else 100)]
print(len(charts), "charts of", charts[0]["image"].size, "pixels")

# %%
fig, axes = plt.subplots(
    1, 4, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 2]}
)
for ax, rec in zip(axes[:3], receipts[:3]):
    ax.imshow(rec["image"])
    ax.set_title(f"total {rec['total']}", fontsize=10)
    ax.axis("off")
axes[3].imshow(charts[0]["image"])
axes[3].set_title(
    f"the value for {charts[0]['label']}: {charts[0]['value']}", fontsize=10
)
axes[3].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# *Receipts: CORD (Naver Clova), CC BY 4.0. Chart: drawn in this notebook.*

# %% [markdown]
# ## 2. Three ways to give a language model an image
#
# Any vision-language model has a vision encoder and a language model, and the question is where they meet. Three
# answers were tried, in this order.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/designs-light.png" width="760" alt="Three ways to connect a vision encoder to a language model. Cross-attention, as in Flamingo and Llama 3.2 Vision: the image never enters the token sequence; extra layers inside the language model attend to the image features. A resampler, as in BLIP-2 and the first Qwen-VL: a fixed set of learned queries reads the image and hands the language model a fixed number of tokens, 32 or 256. A projector, as in LLaVA and most models since: every patch vector becomes a token of the sequence">
#
# **Cross-attention.** DeepMind's Flamingo kept a 70-billion-parameter language model frozen and inserted new layers
# between its blocks that attend to the image features, the way a translation model's decoder attends to its encoder
# ([Alayrac et al., 2022](https://arxiv.org/abs/2204.14198)). The image never enters the sequence, so it costs no tokens,
# but every new layer is new weights, trained on a large mixture of web pages and image–text pairs. Meta's Llama 3.2 Vision
# still works this way ([Llama Team, 2024](https://arxiv.org/abs/2407.21783)).
#
# **A resampler.** Salesforce's BLIP-2 put a small Transformer, the Q-Former, between a frozen encoder and a frozen
# language model: 32 learned query vectors attend to the image and come out as 32 tokens for the language model ([Li et
# al., 2023](https://arxiv.org/abs/2301.12597)). The first Qwen-VL used one cross-attention layer with 256 queries in
# the same role ([Bai et al., 2023](https://arxiv.org/abs/2308.12966)). The price of a picture is fixed and small, and so
# is the detail that survives the squeeze.
#
# **A projector.** LLaVA did the simplest thing: every patch vector from the encoder goes through a linear map to the
# language model's width and enters the sequence as a token ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)).
# Training happened in two short stages. First only the projector learned, on 595 thousand captioned images filtered
# from the CC3M dataset, to put image features where the language model expects words. Then projector and language
# model learned together on 158 thousand conversations about COCO photos that GPT-4 had written from their captions and
# object boxes, without seeing the images
# ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)).
#
# Six months later LLaVA-1.5 replaced the linear map with a two-layer MLP, fed the encoder images of 336 pixels, and
# trained on 558 thousand captioned images ([LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain))
# and then a 665-thousand-example mix of conversations and academic tasks; the whole training took about a day on one
# machine with eight A100s ([Liu et al., 2024](https://arxiv.org/abs/2310.03744)). It matched or beat models trained on
# far more data, and the projector won.
#
# > **Key point.** The projector is cheap to train because the hard parts already exist. The encoder knows what is in
# > the image, the language model knows how to talk about it, and the projector only has to translate between two
# > spaces that were each trained without the other.
#
# The rest of the lesson follows the projector design through its two main lines: LLaVA's, which cuts a large image into
# tiles of the size its encoder was trained on, and Qwen's, which changed the encoder so that it takes images of any
# shape.

# %% [markdown]
# ## 3. A receipt through LLaVA-OneVision, step by step
#
# LLaVA-OneVision is the LLaVA line's model of August 2024 ([Li et al., 2024](https://arxiv.org/abs/2408.03326)). Its
# smallest version joins SigLIP's So400m encoder, the larger sibling of the model in lesson 12, to Qwen2's 0.5-billion
# language model with a two-layer projector, about 0.9 billion parameters in all. We'll follow one receipt through it
# and write each step ourselves.

# %%
OV = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
ov_processor = AutoProcessor.from_pretrained(OV)
ov = LlavaOnevisionForConditionalGeneration.from_pretrained(OV, dtype=torch.float32)
ov = ov.to(device).eval()
ov_config = ov.config
vision_config, text_config = ov_config.vision_config, ov_config.text_config
side, patch = vision_config.image_size, vision_config.patch_size
grid = side // patch
print(
    "vision encoder:",
    vision_config.num_hidden_layers,
    "blocks of",
    vision_config.hidden_size,
)
print(
    f"one tile: {side}×{side} pixels in {grid}×{grid} = {grid * grid} patches of {patch}"
)
print(
    "language model:",
    text_config.num_hidden_layers,
    "blocks of",
    text_config.hidden_size,
)
print("candidate grids of tiles:", len(ov_config.image_grid_pinpoints))

rec = receipts[0]
question = (
    "What is the total amount to pay on this receipt? Answer with the number only."
)
conversation = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
]
ov_prompt = ov_processor.apply_chat_template(conversation, add_generation_prompt=True)
print("the receipt:", rec["image"].size, "pixels, total", rec["total"])

# %% [markdown]
# **Step 1: the grid of tiles.** SigLIP was trained on squares of 384 pixels, and a receipt is a tall rectangle many
# times that size. LLaVA's answer since LLaVA-NeXT is *AnyRes*: pick, from a list of grids made of 384-pixel tiles, the
# one that keeps the most of the image after resizing it to fit, and among equals the one that wastes the least space
# on padding ([Liu et al., 2024](https://llava-vl.github.io/blog/2024-01-30-llava-next/)). The choice is a few lines.


# %%
def best_grid(height, width, pinpoints):
    best, best_kept, best_waste = None, 0, float("inf")
    for h, w in pinpoints:
        scale = min(w / width, h / height)
        kept = min(int(width * scale) * int(height * scale), width * height)
        waste = w * h - kept
        if kept > best_kept or (kept == best_kept and waste < best_waste):
            best, best_kept, best_waste = (h, w), kept, waste
    return best


best = best_grid(
    rec["image"].height, rec["image"].width, ov_config.image_grid_pinpoints
)
rows, cols = best[0] // side, best[1] // side
print(f"chosen canvas: {best[1]}×{best[0]} pixels, {cols} tiles wide and {rows} high")

ov_inputs = ov_processor(images=rec["image"], text=ov_prompt, return_tensors="pt").to(
    device
)
print(
    "pixel_values:",
    tuple(ov_inputs["pixel_values"].shape),
    "(image, tiles, channels, height, width)",
)
print("tiles: 1 thumbnail of the whole receipt +", rows * cols, "pieces of the grid")

# %% [markdown]
# The processor made one tile more than the grid has: the first is a thumbnail, the whole receipt squeezed into one
# 384-pixel square, so that the model also sees the layout at a glance. Here is what the model gets.

# %%
tiles = ov_inputs["pixel_values"][0][: 1 + rows * cols].float().cpu()
tiles = (tiles * 0.5 + 0.5).clamp(0, 1)  # undo the normalisation to (x - 0.5) / 0.5
fig = plt.figure(figsize=(1.6 * (rows + cols) + 0.4, 1.6 * rows + 0.5))
layout = fig.add_gridspec(rows, rows + cols, wspace=0.05, hspace=0.05)
ax = fig.add_subplot(layout[:, :rows])
ax.imshow(tiles[0].permute(1, 2, 0))
ax.set_title("thumbnail", fontsize=10)
ax.axis("off")
for k in range(rows * cols):
    r, c = divmod(k, cols)
    ax = fig.add_subplot(layout[r, rows + c])
    ax.imshow(tiles[1 + k].permute(1, 2, 0))
    ax.axis("off")
fig.suptitle(
    f"what SigLIP sees: a thumbnail and {rows * cols} tiles of {side} px", fontsize=11
)
plt.show()

# %% [markdown]
# **Step 2: the encoder.** Each tile goes through SigLIP on its own, as in lesson 12, and comes out as a grid of patch
# vectors. LLaVA-OneVision takes the output of the last block, all of it: SigLIP has no class token to drop.

# %%
with torch.no_grad():
    hidden = ov.model.vision_tower(
        ov_inputs["pixel_values"][0][: 1 + rows * cols], output_hidden_states=True
    )
    features = hidden.hidden_states[ov_config.vision_feature_layer]
print("encoder output:", tuple(features.shape), "(tiles, patches, width)")

# %% [markdown]
# **Step 3: the projector.** Two linear layers with a GELU between them take each patch vector from SigLIP's width to
# the language model's. This is all the "translation" there is.

# %%
projector = ov.model.multi_modal_projector
with torch.no_grad():
    projected = F.linear(
        F.gelu(F.linear(features, projector.linear_1.weight, projector.linear_1.bias)),
        projector.linear_2.weight,
        projector.linear_2.bias,
    )
    reference = projector(features)
print("projected:", tuple(projected.shape), "(tiles, patches, the LLM's width)")
print(
    "largest difference from the model's projector:",
    (projected - reference).abs().max().item(),
)

# %% [markdown]
# **Step 4: one map, without the padding, with a newline per row.** The grid tiles are laid back side by side into one
# map of patch vectors. Then three things happen that the model learned to expect. The rows or columns that only cover
# the padding are cut away. If the map still holds more than nine tiles' worth of vectors, it is shrunk by bilinear
# interpolation; this is what `anyres_max_9` in the config means. And a learned vector, the *newline*, is appended at
# the end of every row, so that the language model, which reads a flat sequence, can tell where one row of the image
# ends and the next begins.

# %%
H, W = rec["image"].height, rec["image"].width
with torch.no_grad():
    thumbnail, pieces = projected[0], projected[1:]
    mapped = pieces.view(rows, cols, grid, grid, -1).permute(4, 0, 2, 1, 3)
    mapped = mapped.flatten(1, 2).flatten(2, 3)  # (width, rows * grid, cols * grid)
    print("joined map:", tuple(mapped.shape[1:]), "patch positions")
    mh, mw = mapped.shape[1:]
    if W / H > mw / mh:  # the receipt is wider than the canvas: padding above and below
        pad = (mh - int(round(H * mw / W, 7))) // 2
        mapped = mapped[:, pad : mh - pad, :]
    else:  # taller: padding left and right
        pad = (mw - int(round(W * mh / H, 7))) // 2
        mapped = mapped[:, :, pad : mw - pad]
    print("without the padding:", tuple(mapped.shape[1:]))
    ratio = math.sqrt(mapped.shape[1] * mapped.shape[2] / (9 * grid * grid))
    if ratio > 1.1:
        size = [int(mapped.shape[1] // ratio), int(mapped.shape[2] // ratio)]
        mapped = F.interpolate(mapped[None], size, mode="bilinear")[0]
        print("shrunk to nine tiles' worth:", tuple(mapped.shape[1:]))
    newline = ov.model.image_newline[:, None, None].expand(
        mapped.shape[0], mapped.shape[1], 1
    )
    mapped = torch.cat([mapped, newline], dim=-1)
    image_tokens = torch.cat([thumbnail, mapped.flatten(1, 2).T], dim=0)
print(
    "image tokens:",
    image_tokens.shape[0],
    "=",
    grid * grid,
    "thumbnail +",
    mapped.shape[1],
    "rows ×",
    mapped.shape[2],
    "(with the newline)",
)

# %% [markdown]
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/anyres-light.png" width="760" alt="How LLaVA-OneVision cuts an image into tiles. A 768 by 1024 pixel image is fitted into the best of its candidate grids, 2 tiles wide and 3 high at 384 pixels each, with a thumbnail of the whole image beside them. Each tile gives 27 by 27 feature vectors. The tiles' features are joined into one map, the rows and columns that only cover padding are cut away, the map is shrunk if it holds more than nine tiles' worth, and a learned newline vector closes every row. The image costs 4744 tokens: 729 for the thumbnail and 73 rows of 54 plus a newline">
#
# **Step 5: the splice.** The processor already wrote a placeholder token in the prompt for every image token the model
# will need, and the count must match ours exactly. The prompt's tokens are embedded as usual, the placeholders are
# replaced by our image vectors, and the language model reads the result. If every step above is right, its logits are
# the model's own.

# %%
ids = ov_inputs["input_ids"]
image_token = getattr(ov_config, "image_token_id", None) or ov_config.image_token_index
print(
    "placeholders in the prompt:",
    int((ids == image_token).sum()),
    "| our image tokens:",
    image_tokens.shape[0],
)
print("text tokens around them:", int((ids != image_token).sum()))
with torch.no_grad():
    embedded = ov.get_input_embeddings()(ids).clone()
    embedded[ids == image_token] = image_tokens.to(embedded.dtype)
    out = ov.model.language_model(
        inputs_embeds=embedded, attention_mask=ov_inputs["attention_mask"]
    )
    ours = ov.lm_head(out.last_hidden_state)
    theirs = ov(**ov_inputs).logits
print(
    "largest difference from LLaVA-OneVision's logits:",
    (ours - theirs).abs().max().item(),
)
with torch.no_grad():
    generated = ov.generate(**ov_inputs, max_new_tokens=20, do_sample=False)
print(
    "answer:",
    ov_processor.decode(generated[0, ids.shape[1] :], skip_special_tokens=True),
)
print("truth: ", rec["total"])

# %% [markdown]
# @@ (the logits match exactly or to float rounding; the answer and the token count: most of the sequence is the image,
# a dozen or so tokens are the question)
#
# → @@ (one sentence with the share of the sequence that is image)

# %% [markdown]
# ## 4. What an image token looks like to the language model
#
# The projector was trained to put image features where the language model can use them. A natural guess is that it
# turned them into something like words: that a patch showing the word *TOTAL* lands next to the embedding of "total".
# We can check. The language model's input embeddings are a table of 152 thousand vectors, one per token of its
# vocabulary; compare the image tokens with them.

# %%
with torch.no_grad():
    vocab = ov.get_input_embeddings().weight.float()
    text_ids = ids[0][ids[0] != image_token]
    words_in_prompt = vocab[text_ids]
    vocab_unit = F.normalize(vocab, dim=-1)
    image_nearest = (
        (F.normalize(image_tokens.float(), dim=-1) @ vocab_unit.T).max(-1).values
    )
    text_nearest = (
        (F.normalize(words_in_prompt, dim=-1) @ vocab_unit.T)
        .topk(2, dim=-1)
        .values[:, 1]
    )
rows_space = [
    {
        "vectors": "image tokens",
        "average length": image_tokens.float().norm(dim=-1).mean().item(),
        "cosine with the nearest vocabulary token": image_nearest.mean().item(),
    },
    {
        "vectors": "the prompt's word tokens",
        "average length": words_in_prompt.norm(dim=-1).mean().item(),
        "cosine with the nearest vocabulary token": text_nearest.mean().item(),
    },
]
table(pd.DataFrame(rows_space), digits=2)

# %% [markdown]
# @@ (image tokens are many times longer than word embeddings, and their nearest word has a small cosine, while a word's
# nearest other word is much closer; image tokens do not live among words. Link to lesson 12's modality gap.)
#
# That doesn't mean the language model can't read them; it means it doesn't read them as *input* words. A sharper
# question is what each image token becomes as it rises through the language model. The *logit lens* answers it: take
# the hidden state at any layer, push it through the final norm and the output head as if it were the last layer, and
# see which word it would predict ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).
# Neo and colleagues did this for LLaVA and found object names appearing at image positions in the later layers ([Neo et
# al., 2024](https://arxiv.org/abs/2410.07149)). On receipts, the question becomes: do the words printed on the receipt
# show up at its image tokens?
#
# We count, layer by layer, the share of image tokens whose logit-lens word is one of the receipt's own words of three
# letters or more (CORD gives every word). The chance level is the share of the whole vocabulary made of those words.

# %%
ov = ov.half()  # long sequences of hidden states: half precision is plenty for a lens
vocab_text = [
    ov_processor.tokenizer.decode([t]).strip().lower()
    for t in range(len(ov_processor.tokenizer))
]
lens_rows = []
for rec_lens in receipts[: 2 if SMOKE else 10]:
    inputs = ov_processor(
        images=rec_lens["image"], text=ov_prompt, return_tensors="pt"
    ).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    words = {w.lower().strip(".,:;()") for w in rec_lens["words"]}
    words = {
        w
        for w in words
        if len(w) >= 3 and not w.replace(".", "").replace(",", "").isdigit()
    }
    chance = sum(1 for v in vocab_text if len(v) >= 3 and v in words) / len(vocab_text)
    with torch.no_grad():
        states = ov(**inputs, output_hidden_states=True).hidden_states
        where = (inputs["input_ids"][0] == image_token).nonzero().flatten()
        for layer, state in enumerate(states):
            top = (
                ov.lm_head(ov.model.language_model.norm(state[0, where]))
                .argmax(-1)
                .tolist()
            )
            found = [
                vocab_text[t]
                for t in top
                if t < len(vocab_text) and vocab_text[t] in words
            ]
            lens_rows.append(
                dict(
                    receipt=rec_lens["id"],
                    layer=layer,
                    share=len(found) / len(top),
                    chance=chance,
                    found=sorted(set(found)),
                )
            )
    del states
    torch.cuda.empty_cache()
lens = pd.DataFrame(lens_rows)
by_layer = lens.groupby("layer")[["share", "chance"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(
    by_layer["layer"],
    100 * by_layer["share"],
    color=COLORS["enc"],
    marker="o",
    ms=3,
    label="image tokens whose lens word is on the receipt",
)
ax.plot(
    by_layer["layer"],
    100 * by_layer["chance"],
    color=COLORS["muted"],
    ls="--",
    label="chance",
)
ax.set_xlabel("layer of the language model (0: the input)")
ax.set_ylabel("% of image tokens")
ax.set_title("Words from the receipt appear at its image tokens in the later layers")
ax.legend(frameon=False, fontsize=9)
plt.show()
peak = by_layer.loc[by_layer["share"].idxmax()]
print(
    f"peak at layer {int(peak['layer'])}: {100 * peak['share']:.2f}% of image tokens, "
    f"{peak['share'] / max(peak['chance'], 1e-9):.0f} times the chance level"
)
print(
    "words found at the peak, first receipt:",
    lens[lens.layer == int(peak["layer"])]["found"].iloc[0][:15],
)

# %% [markdown]
# @@ (interpret: near chance at the input and the first layers, rising in the middle-to-late layers, falling at the very
# end where the model prepares the next token of the answer; the words found are the receipt's own: item names, TOTAL,
# CASH. The image tokens become the text they show, inside the model, not at its input.)
#
# > **Key point.** @@ (one sentence: the projector doesn't translate pixels into words; it hands the language model
# > vectors that its own layers turn into words)

# %% [markdown]
# ## 5. The price of a picture
#
# Every image token costs what a word costs: a row in every attention matrix, a slot in every layer's KV cache, its share
# of the compute in every block. So the first question to ask of any vision-language model is how many tokens it spends
# on an image. The processors know; we only need to ask them, with blank images of three sizes, from a small photo to a
# scanned page.

# %%
PROCESSORS = {
    "LLaVA-1.5": "llava-hf/llava-1.5-7b-hf",
    "LLaVA-NeXT": "llava-hf/llava-v1.6-mistral-7b-hf",
    "LLaVA-OneVision": OV,
    "Qwen2-VL": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2.5-VL": "Qwen/Qwen2.5-VL-3B-Instruct",
    "SmolVLM": "HuggingFaceTB/SmolVLM-256M-Instruct",
}
SIZES = [(448, 448), (768, 1024), (1536, 2048)]
counts = []
for name, repo in PROCESSORS.items():
    processor = ov_processor if repo == OV else AutoProcessor.from_pretrained(repo)
    talk = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}
    ]
    prompt = processor.apply_chat_template(talk, add_generation_prompt=True)
    token = getattr(processor, "image_token_id", None)
    if token is None:
        token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    row = {"model": name}
    for h, w in SIZES:
        blank = Image.new("RGB", (w, h), (200, 200, 200))
        row[f"{w}×{h}"] = int(
            (
                processor(images=blank, text=prompt, return_tensors="pt")["input_ids"]
                == token
            ).sum()
        )
    counts.append(row)
counts = pd.DataFrame(counts)
display(table(counts, digits=0))

fig, ax = plt.subplots(figsize=(8.5, 3.8))
x = np.arange(len(counts))
for k, (h, w) in enumerate(SIZES):
    shade = [COLORS["enc_tint"], "#8CCFC4", COLORS["enc"]][k]
    ax.bar(
        x + (k - 1) * 0.27,
        counts[f"{w}×{h}"],
        width=0.25,
        color=shade,
        label=f"{w}×{h} px",
    )
ax.set_xticks(x, counts["model"], fontsize=9)
ax.set_ylabel("image tokens")
ax.set_title("The same image costs a different number of tokens in each model")
ax.legend(frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# @@ (read the table: LLaVA-1.5 fixed; LLaVA-NeXT and OneVision tile, OneVision the most expensive at every size; Qwen
# grows with the area, cheap for small images and the priciest-but-one for a page; SmolVLM's count even falls for larger
# images because it resizes the long side and compresses with pixel shuffle.)
#
# What does a token cost? In memory, the KV cache of lesson 8: every token keeps a key and a value in every layer. For
# Qwen2-VL 2B's language model that is:

# %%
QV = "Qwen/Qwen2-VL-2B-Instruct"
qv_config = json.load(open(hf_hub_download(QV, "config.json")))
q_text = qv_config.get("text_config", qv_config)
head_dim = q_text["hidden_size"] // q_text["num_attention_heads"]
per_token = (
    2 * q_text["num_hidden_layers"] * q_text["num_key_value_heads"] * head_dim * 2
)  # K and V, fp16
layers, kv_heads = q_text["num_hidden_layers"], q_text["num_key_value_heads"]
print(
    f"KV cache per token: 2 × {layers} layers × {kv_heads} KV heads × {head_dim} × 2 bytes"
    f" = {per_token / 1024:.0f} KB"
)
for h, w in SIZES:
    n = int(counts.loc[counts.model == "Qwen2-VL", f"{w}×{h}"].iloc[0])
    print(f"  a {w}×{h} image: {n} tokens, {n * per_token / 2**20:.1f} MB of cache")

# %% [markdown]
# @@ (small for one image; a conversation with ten photos or a 30-page PDF changes the picture; and compute grows faster
# than linearly in attention. The design question is never "more tokens or fewer" but which tokens.)

# %% [markdown]
# ## 6. Qwen2-VL: an image that keeps its shape
#
# Qwen2-VL took the other road ([Wang et al., 2024](https://arxiv.org/abs/2409.12191)). Instead of cutting an image
# into tiles for an encoder that only knows squares, it trained its own Vision Transformer, 675 million parameters, to
# take an image of any shape directly. Two ideas make that work, and we'll rebuild both.
#
# **The image keeps its shape.** The processor rounds each side to a multiple of 28 pixels, then scales the whole image
# so that its area stays between two limits, `min_pixels` and `max_pixels`. The encoder cuts it into patches of 14
# pixels, and after its last block a *merger* joins every 2×2 block of neighbouring patches into one vector, so that one
# token covers 28×28 pixels. The number of tokens is the area divided by 28², and the area is something we choose.

# %%
qv_processor = AutoProcessor.from_pretrained(QV)
qv = Qwen2VLForConditionalGeneration.from_pretrained(
    QV, dtype=torch.float16, attn_implementation="sdpa"
)
qv = qv.to(device).eval()
eos = qv.generation_config.eos_token_id
stop_ids = set(eos if isinstance(eos, list) else [eos])
stop_ids.add(qv_processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))


def smart_resize(height, width, max_pixels, min_pixels=56 * 56, factor=28):
    h, w = (
        max(factor, round(height / factor) * factor),
        max(factor, round(width / factor) * factor),
    )
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = (
            math.floor(height / beta / factor) * factor,
            math.floor(width / beta / factor) * factor,
        )
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = (
            math.ceil(height * beta / factor) * factor,
            math.ceil(width * beta / factor) * factor,
        )
    return h, w


def qwen_inputs(image, text, budget):
    """Inputs for one image and a question, at most `budget` image tokens."""
    images = qv_processor.image_processor
    images.max_pixels, images.min_pixels = budget * 28 * 28, min(budget, 16) * 28 * 28
    if isinstance(getattr(images, "size", None), dict):
        images.size = {
            "shortest_edge": images.min_pixels,
            "longest_edge": images.max_pixels,
        }
    talk = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    ]
    prompt = qv_processor.apply_chat_template(talk, add_generation_prompt=True)
    return qv_processor(images=image, text=prompt, return_tensors="pt").to(device)


budget = 1024
q_inputs = qwen_inputs(rec["image"], question, budget)
t, gh, gw = q_inputs["image_grid_thw"][0].tolist()
ours_h, ours_w = smart_resize(
    rec["image"].height, rec["image"].width, budget * 28 * 28, min(budget, 16) * 28 * 28
)
print(
    "the receipt:",
    rec["image"].size,
    "→ ours:",
    (ours_w, ours_h),
    "| processor:",
    (gw * 14, gh * 14),
)
print(
    "patches of 14:",
    gh,
    "×",
    gw,
    "| after the 2×2 merger:",
    gh // 2,
    "×",
    gw // 2,
    "=",
    gh * gw // 4,
    "tokens",
)
print(
    "pixel_values:",
    tuple(q_inputs["pixel_values"].shape),
    "(patches, 3 channels × 2 frames × 14 × 14)",
)

# %% [markdown]
# The patches arrive as a flat list, each flattened to 1,176 numbers: three channels, 14×14 pixels and two frames,
# because the same encoder also reads video, where two consecutive frames share a patch, and a still image is simply
# repeated.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/dynres-light.png" width="760" alt="How Qwen2-VL sizes an image. The image keeps its proportions: its sides are rounded to multiples of 28 pixels and the area is kept under a budget. The encoder cuts it into 14-pixel patches, and a merger joins every 2 by 2 block of patches into one token, so each token covers 28 by 28 pixels. A 864 by 1296 receipt becomes 364 by 532 pixels, 247 tokens or 728 by 1092 pixels, 1014 tokens under budgets of 256 and 1024 tokens">
#
# **Positions in three dimensions.** A Transformer with rotary positions (lesson 6) knows where a token is from a
# single number. For an image that is a poor fit: two tokens that touch, one above the other, would be a whole row of
# tokens apart. Qwen2-VL's *M-RoPE* gives every token three positions, a time, a row and a column, and splits the
# rotary dimensions of each attention head into three groups that rotate with one of them each. Text tokens get the same
# number in all three, so for text nothing changes. The rule is short enough to write.


# %%
def mrope_positions(input_ids, image_token, grid_h, grid_w):
    """Three positions per token, (time, row, column), for one image of grid_h × grid_w."""
    n = len(input_ids)
    start = int((input_ids == image_token).nonzero()[0])
    positions = torch.zeros(3, n, dtype=torch.long)
    positions[:, :start] = torch.arange(
        start
    )  # text before the image: the same number three times
    image = torch.arange(grid_h * grid_w)
    positions[0, start : start + grid_h * grid_w] = start  # one frame: one time step
    positions[1, start : start + grid_h * grid_w] = start + image // grid_w  # rows
    positions[2, start : start + grid_h * grid_w] = start + image % grid_w  # columns
    resume = start + max(grid_h, grid_w)
    after = n - (start + grid_h * grid_w)
    positions[:, start + grid_h * grid_w :] = resume + torch.arange(after)
    return positions


q_ids = q_inputs["input_ids"][0].cpu()
ours_pos = mrope_positions(q_ids, qv.config.image_token_id, gh // 2, gw // 2)


def rope_index(inputs):
    """Qwen2-VL's own positions, through either of its two signatures across versions."""
    extra = {k: inputs[k] for k in ("mm_token_type_ids",) if k in inputs}
    try:
        return qv.model.get_rope_index(
            inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
            **extra,
        )[0]
    except TypeError:
        return qv.model.get_rope_index(
            inputs["input_ids"],
            inputs.get("mm_token_type_ids"),
            inputs["image_grid_thw"],
            None,
            inputs["attention_mask"],
        )[0]


theirs_pos = rope_index(q_inputs)
print("positions equal to Qwen2-VL's:", torch.equal(ours_pos, theirs_pos[:, 0].cpu()))
print("tokens in the sequence:", len(q_ids), "| largest position:", int(ours_pos.max()))

# %% [markdown]
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/mrope-light.png" width="760" alt="Qwen2-VL's positions in three numbers. Each token has a time, a height and a width position. The 4 text tokens before the image count 0 to 3 in all three. The image's 3 by 4 tokens keep the time position 4 and take height positions 4 to 6 and width positions 4 to 7, so neighbours in the picture are neighbours in position. The text after the image resumes at 8, the image's start plus its larger side, not plus its 12 tokens">
#
# @@ (the largest position is far smaller than the length: after the image, text resumes at the image's start plus its
# larger side, so a thousand image tokens advance the position by a few dozen. For long documents this matters: the
# position budget the language model was trained on is not eaten by pictures.)

# %% [markdown]
# ## 7. Reading under a budget
#
# Now the experiment the lesson has been building towards. With Qwen2-VL we choose how many tokens an image may cost, so
# we can ask the receipts and the charts the same questions at budgets from 64 tokens to 2,048 and watch what each
# extra token buys. A receipt at 64 tokens is a thumbnail of a few hundred pixels; at 2,048 the small print is legible.
#
# The answers are compared digit by digit with the truth: "60,000", "60.000" and "Rp 60.000" all count as 60000.


# %%
def digits(text):
    return re.sub(r"\D", "", text or "")


def answer(inputs, max_new_tokens=14):
    with torch.no_grad():
        out = qv.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return qv_processor.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def chart_question(chart):
    return f"What is the value shown for {chart['label']}? Answer with the number only."


BUDGETS = [64, 256] if SMOKE else [64, 128, 256, 512, 1024, 2048]
budget_rows, examples = [], []
for budget in BUDGETS:
    for kind, items in (("receipts", receipts), ("charts", charts)):
        right, tokens, seconds = 0, [], []
        for item in items:
            text = question if kind == "receipts" else chart_question(item)
            truth = item["total"] if kind == "receipts" else item["value"]
            inputs = qwen_inputs(item["image"], text, budget)
            start = time.time()
            said = answer(inputs)
            seconds.append(time.time() - start)
            right += digits(said) == digits(truth)
            tokens.append(int((inputs["input_ids"] == qv.config.image_token_id).sum()))
            if item is receipts[0] and kind == "receipts":
                examples.append((budget, said))
        budget_rows.append(
            dict(
                kind=kind,
                budget=budget,
                tokens=np.mean(tokens),
                accuracy=right / len(items),
                seconds=np.mean(seconds),
            )
        )
budgets = pd.DataFrame(budget_rows)
display(table(budgets, digits=2))

fig, ax = plt.subplots(figsize=(8, 3.8))
for kind, color in (("receipts", COLORS["enc"]), ("charts", COLORS["data"])):
    part = budgets[budgets.kind == kind]
    ax.plot(part["tokens"], 100 * part["accuracy"], marker="o", color=color, label=kind)
ax.set_xscale("log", base=2)
ax.set_xlabel("image tokens actually used (log scale)")
ax.set_ylabel("% answered exactly")
ax.set_title("Qwen2-VL 2B reading receipts and charts under a token budget")
ax.legend(frameon=False)
plt.show()
print(
    "the first receipt, answer by budget:", examples, "| truth:", receipts[0]["total"]
)

# %% [markdown]
# @@ (interpret: the curves; where receipts saturate; charts saturate earlier because their font is fixed relative to a
# small image; the cost in seconds; how many tokens the images actually used vs the budget, small receipts don't fill a
# large budget.)
#
# → @@ (takeaway with the budget beyond which more tokens stop helping)

# %%
crop_rec = receipts[0]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
for ax, b in zip(axes, [64, 256, 1024]):
    h, w = smart_resize(
        crop_rec["image"].height,
        crop_rec["image"].width,
        b * 28 * 28,
        min(b, 16) * 28 * 28,
    )
    small = crop_rec["image"].resize((w, h), Image.BICUBIC)
    ax.imshow(small.resize(crop_rec["image"].size, Image.NEAREST))
    ax.set_title(f"budget {b}: {w}×{h} px, {(h // 28) * (w // 28)} tokens", fontsize=9)
    ax.axis("off")
plt.suptitle("What the encoder sees of the same receipt", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# *Receipt: CORD (Naver Clova), CC BY 4.0, shown at the resolution each budget allows.*
#
# @@ (one paragraph on the images: at 64 tokens no digit is legible and the model still answers, a guess shaped like a
# total; reading is a resolution problem before it is a language problem.)

# %% [markdown]
# ## 8. Dropping image tokens inside the model
#
# If an image costs a thousand tokens, perhaps the model doesn't need all of them once it has looked. Chen and
# colleagues measured where LLaVA-1.5's attention goes and found that after the second layer image tokens receive very
# little of it; their method, FastV, ranks the image tokens at layer 2 by the attention they receive, drops the bottom
# half, and runs the remaining layers on what is left ([Chen et al., 2024](https://arxiv.org/abs/2403.06764)). On
# LLaVA-1.5 13B they report 45% fewer FLOPs without a loss on their benchmarks.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/pruning-light.png" width="760" alt="Pruning visual tokens inside the language model. All tokens pass through the first two layers. At layer two, the attention that the last prompt token pays to each image token ranks them; only the most-attended image tokens continue into the later layers, together with all the text tokens. The later layers do less work, and the question is how much the answer suffers">
#
# Their benchmarks were mostly about what a photo shows. Reading is different: the one patch with the total on it may
# not look important at layer 2. To test it we need our own loop over Qwen2-VL's layers, because the pruning happens in
# the middle of the forward pass. It runs the first layers on the whole sequence, ranks and drops image tokens at layer
# 2, runs the remaining layers on the shorter sequence, and then generates the answer token by token with a cache in
# which the early layers remember every token and the later ones only the kept ones.
#
# We compare three rules for which tokens to keep: FastV's own, the attention each image token receives on average from
# the tokens that can see it; a simpler variant, the attention from the prompt's last token only; and a random choice of
# the same size, as the control any such method must beat.

# %%
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb


def attention_at(layer, hidden, positions, rotary):
    """The attention weights of one layer, averaged over heads: (queries, keys)."""
    attn = layer.self_attn
    x = layer.input_layernorm(hidden)
    n = x.shape[1]
    cos, sin = rotary(x, positions)
    q = attn.q_proj(x).view(1, n, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(1, n, -1, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(
        q.shape[1] // k.shape[1], dim=1
    )  # grouped-query attention, lesson 6
    scores = (q @ k.transpose(-1, -2)).float() * attn.scaling
    scores = scores + torch.full((n, n), float("-inf"), device=scores.device).triu(1)
    return torch.softmax(scores, -1)[0].mean(0)


def generate_pruned(
    inputs, keep=1.0, rule="fastv", layer_k=2, max_new_tokens=14, seed=0
):
    lm = qv.model.language_model
    ids = inputs["input_ids"]
    image_token = qv.config.image_token_id
    with torch.no_grad():
        hidden = qv.get_input_embeddings()(ids).clone()
        image = qv.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        ).pooler_output
        hidden[ids == image_token] = torch.cat(list(image), 0).to(hidden.dtype)
        positions = rope_index(inputs)
        is_image = ids[0] == image_token
        cache = (
            DynamicCache(config=lm.config)
            if "config" in DynamicCache.__init__.__code__.co_varnames
            else DynamicCache()
        )
        for index, layer in enumerate(lm.layers):
            if index == layer_k and keep < 1.0:
                where = is_image.nonzero().flatten()
                n_keep = max(1, round(len(where) * keep))
                if rule == "random":
                    order = torch.randperm(
                        len(where), generator=torch.Generator().manual_seed(seed)
                    )
                    chosen = where[order[:n_keep].to(where.device)]
                else:
                    att = attention_at(layer, hidden, positions, lm.rotary_emb)
                    if (
                        rule == "fastv"
                    ):  # received from all the tokens that can see it, on average
                        received = att.sum(0) / torch.arange(
                            att.shape[0], 0, -1, device=att.device
                        )
                    else:  # "last": received from the prompt's last token only
                        received = att[-1]
                    chosen = where[received[where].topk(n_keep).indices]
                kept = ~is_image.clone()
                kept[chosen] = True
                hidden, positions = hidden[:, kept], positions[:, :, kept]
            cos, sin = lm.rotary_emb(hidden, positions)
            hidden = layer(
                hidden,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=(cos, sin),
            )
        logits = qv.lm_head(lm.norm(hidden[:, -1:]))
        out, nxt = [], int(positions.max()) + 1
        for step in range(max_new_tokens):
            token = int(logits[0, -1].argmax())
            if token in stop_ids:
                break
            out.append(token)
            x = qv.get_input_embeddings()(torch.tensor([[token]], device=device))
            p = torch.full((3, 1, 1), nxt + step, device=device, dtype=torch.long)
            cos, sin = lm.rotary_emb(x, p)
            for layer in lm.layers:
                x = layer(
                    x,
                    position_ids=p,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=(cos, sin),
                )
            logits = qv.lm_head(lm.norm(x))
    return qv_processor.tokenizer.decode(out, skip_special_tokens=True)


check = qwen_inputs(rec["image"], question, 1024)
print("our loop:          ", generate_pruned(check))
print("Qwen2-VL's generate:", answer(check))

# %% [markdown]
# With nothing pruned, our loop and the library's `generate` give the same answer, so the loop is right. Now the
# experiment, at a budget of 1,024 tokens, keeping a half, a quarter and a tenth of the image tokens.

# %%
KEEPS = [1.0, 0.25] if SMOKE else [1.0, 0.5, 0.25, 0.1]
prune_rows = []
for kind, items in (("receipts", receipts), ("charts", charts)):
    for rule in ("fastv", "last", "random"):
        for keep in KEEPS:
            if keep == 1.0 and rule != "fastv":
                continue  # nothing dropped: the same for every rule
            right, seconds = 0, []
            for j, item in enumerate(items):
                text = question if kind == "receipts" else chart_question(item)
                truth = item["total"] if kind == "receipts" else item["value"]
                start = time.time()
                said = generate_pruned(
                    qwen_inputs(item["image"], text, 1024), keep=keep, rule=rule, seed=j
                )
                seconds.append(time.time() - start)
                right += digits(said) == digits(truth)
            prune_rows.append(
                dict(
                    kind=kind,
                    rule="all kept" if keep == 1.0 else rule,
                    keep=keep,
                    accuracy=right / len(items),
                    seconds=np.mean(seconds),
                )
            )
pruned = pd.DataFrame(prune_rows)
display(table(pruned, digits=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, kind in zip(axes, ("receipts", "charts")):
    part = pruned[pruned.kind == kind]
    full = part[part.keep == 1.0]["accuracy"].iloc[0]
    for rule, color, label in (
        ("fastv", COLORS["att"], "FastV (average received)"),
        ("last", COLORS["dec"], "attention from the last token"),
        ("random", COLORS["muted"], "random"),
    ):
        r = part[part.rule == rule].sort_values("keep")
        ax.plot(
            [*r["keep"], 1.0],
            [*(100 * r["accuracy"]), 100 * full],
            marker="o",
            color=color,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("share of image tokens kept after layer 2")
    ax.set_title(kind)
axes[0].set_ylabel("% answered exactly")
axes[1].legend(frameon=False, fontsize=9)
plt.suptitle("Which image tokens can a reading model lose?", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# @@ (interpret against FastV's claim and against random: which rule holds up, which collapses, and why the last
# token's attention at layer 2 is a poor guide for reading: it goes to a few tokens that collect attention for every
# query, not to the digits the answer needs. The time saved.)
#
# > **Trap.** @@ (a token-pruning method that does not beat a random choice of the same size has not shown that it
# > knows which tokens matter)

# %% [markdown]
# ## 9. Where vision-language models are today
#
# **Qwen's line.** Qwen2.5-VL added window attention to the encoder, so that large images cost it less, and positions
# tied to absolute time for video ([Bai et al., 2025](https://arxiv.org/abs/2502.13923)); Qwen3-VL followed in 2025
# ([Qwen Team, 2025](https://arxiv.org/abs/2511.21631)). Their technical reports are the best public descriptions of
# how a strong reading model is trained.
#
# **Other open families.** InternVL tiles like LLaVA but compresses each tile four times by *pixel shuffle*, folding
# 2×2 neighbouring vectors into one ([Chen et al., 2024](https://arxiv.org/abs/2404.16821)); SmolVLM pushes the same
# idea to models of 256 million parameters ([Marafioti et al., 2025](https://arxiv.org/abs/2504.05299)); Gemma 3 gives
# every image a fixed 256 tokens from a SigLIP encoder ([Gemma Team, 2025](https://arxiv.org/abs/2503.19786)). Allen
# AI's Molmo was trained on PixMo, image descriptions collected from people speaking about the images rather than
# generated by a proprietary model, and released with the data ([Deitke et al.,
# 2024](https://arxiv.org/abs/2409.17146)).
#
# **What they are measured on.** Reading benchmarks such as [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
# ([Masry et al., 2022](https://arxiv.org/abs/2203.10244)) and [DocVQA](https://www.docvqa.org/) ([Mathew et al.,
# 2021](https://arxiv.org/abs/2007.00398)) ask exactly the kind of question this lesson asked; POPE asks a model whether
# an object is in a photo and found that early models said yes to objects that were not there, the *hallucination*
# problem of vision-language models ([Li et al., 2023](https://arxiv.org/abs/2305.10355)).
#
# LLaVA-OneVision 0.5B's passport, and the family tree with this lesson's branch:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/passport-light.png" width="760" alt="LLaVA-OneVision 0.5B's passport. Published August 2024 by the LLaVA team; the Hugging Face port is under Apache 2.0. A vision encoder, a two-layer MLP projector and a decoder language model, about 0.9 billion parameters. Vision: SigLIP so400m, 384-pixel tiles of 729 tokens each. Language: Qwen2 0.5B with 24 blocks. Images: AnyRes, a thumbnail plus a grid of tiles with a newline per row. Training in stages: projector alignment, single images, then a one-vision mixture. Good at describing photos and short answers on small devices. Not for dense documents, counting or facts it cannot see">
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/a13/family-light.png" width="760" alt="The family of the Transformer as this course covers it, with lesson 13, vision-language models, highlighted and lessons 1 to 12 done">
#
# ## References
#
# - Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). [Visual instruction tuning](https://arxiv.org/abs/2304.08485). *NeurIPS 2023*.
# - Liu, H., Li, C., Li, Y., & Lee, Y. J. (2024). [Improved baselines with visual instruction tuning](https://arxiv.org/abs/2310.03744). *CVPR 2024*.
# - Liu, H., Li, C., Li, Y., et al. (2024). [LLaVA-NeXT: improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/). Blog post.
# - Li, B., Zhang, Y., Guo, D., et al. (2024). [LLaVA-OneVision: easy visual task transfer](https://arxiv.org/abs/2408.03326). *arXiv*.
# - Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022). [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198). *NeurIPS 2022*.
# - Li, J., Li, D., Savarese, S., & Hoi, S. (2023). [BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models](https://arxiv.org/abs/2301.12597). *ICML 2023*.
# - Bai, J., Bai, S., Yang, S., et al. (2023). [Qwen-VL: a versatile vision-language model for understanding, localization, text reading, and beyond](https://arxiv.org/abs/2308.12966). *arXiv*.
# - Wang, P., Bai, S., Tan, S., et al. (2024). [Qwen2-VL: enhancing vision-language model's perception of the world at any resolution](https://arxiv.org/abs/2409.12191). *arXiv*.
# - Bai, S., Chen, K., Liu, X., et al. (2025). [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923). *arXiv*.
# - Qwen Team (2025). [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631). *arXiv*.
# - Llama Team, AI @ Meta (2024). [The Llama 3 herd of models](https://arxiv.org/abs/2407.21783). *arXiv*.
# - Chen, L., Zhao, H., Liu, T., et al. (2024). [An image is worth 1/2 tokens after layer 2: plug-and-play inference acceleration for large vision-language models](https://arxiv.org/abs/2403.06764). *ECCV 2024*.
# - Neo, C., Ong, L., Torr, P., Geva, M., Krueger, D., & Barez, F. (2024). [Towards interpreting visual information processing in vision-language models](https://arxiv.org/abs/2410.07149). *arXiv*.
# - nostalgebraist (2020). [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). *LessWrong*.
# - Chen, Z., Wang, W., Tian, H., et al. (2024). [How far are we to GPT-4V? Closing the gap to commercial multimodal models with open-source suites](https://arxiv.org/abs/2404.16821). *arXiv*.
# - Marafioti, A., Zohar, O., Farré, M., et al. (2025). [SmolVLM: redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299). *arXiv*.
# - Gemma Team (2025). [Gemma 3 technical report](https://arxiv.org/abs/2503.19786). *arXiv*.
# - Deitke, M., Clark, C., Lee, S., et al. (2024). [Molmo and PixMo: open weights and open data for state-of-the-art vision-language models](https://arxiv.org/abs/2409.17146). *arXiv*.
# - Masry, A., Long, D. X., Tan, J. Q., Joty, S., & Hoque, E. (2022). [ChartQA: a benchmark for question answering about charts with visual and logical reasoning](https://arxiv.org/abs/2203.10244). *Findings of ACL 2022*.
# - Mathew, M., Karatzas, D., & Jawahar, C. V. (2021). [DocVQA: a dataset for VQA on document images](https://arxiv.org/abs/2007.00398). *WACV 2021*.
# - Li, Y., Du, Y., Zhou, K., et al. (2023). [Evaluating object hallucination in large vision-language models](https://arxiv.org/abs/2305.10355). *EMNLP 2023*.
# - Park, S., Shin, S., Lee, B., et al. (2019). [CORD: a consolidated receipt dataset for post-OCR parsing](https://openreview.net/forum?id=SJl3z659UH). *Document Intelligence Workshop at NeurIPS 2019*. Data: [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), CC BY 4.0.
# - Datasets of the LLaVA line: [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).
