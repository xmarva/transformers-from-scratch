"""Static figures and cover of the second course's lesson 2, from RoBERTa to ModernBERT.

ModernBERT's layer pattern and sizes are read at render time from its config.json on the Hugging Face Hub.
"""

import json

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

MODERNBERT = "answerdotai/ModernBERT-base"


def config(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: four encoders, four changes

def hero() -> Figure:
    rows = [
        ("BERT", "2018", "Google", "the start: masked LM, post-norm, learned positions", "plain"),
        ("RoBERTa", "2019", "Facebook AI", "the recipe: more data, big batches, no NSP game", "data"),
        ("ELECTRA", "2020", "Stanford, Google", "the task: spot replaced tokens, learn from all", "enc"),
        ("DeBERTa", "2020", "Microsoft", "the attention: content and position kept apart", "att"),
        ("ModernBERT", "2024", "Answer.AI, LightOn", "the block: RoPE, GeGLU, local and global layers", "dec"),
    ]
    fig = Figure(
        760,
        96 + len(rows) * 44 + 10,
        alt="Four encoders after BERT, each with the one thing it changed: RoBERTa in 2019 the training recipe, "
        "ELECTRA in 2020 the training task, DeBERTa in 2020 the attention, ModernBERT in 2024 the whole block",
    )
    fig.text(14, 28, "Seven years of encoders, one change at a time", kind="title")
    fig.text(14, 50, "each model kept most of what came before and changed one thing", kind="note")
    for k, (name, year, who, change, role) in enumerate(rows):
        y = 70 + k * 44
        fig.box(14, y, 130, 34, role=role, label=name, strong=k == len(rows) - 1)
        fig.text(158, y + 21, year, kind="mono")
        fig.text(200, y + 21, who, kind="label")
        fig.text(344, y + 21, change, kind="body")
        if k < len(rows) - 1:
            fig.arrow((79, y + 35), (79, y + 43))
    return fig


# ---------------------------------------------------------------- ELECTRA

def electra() -> Figure:
    words = ["the", "build", "fails", "on", "arm64"]
    masked = ["the", "[MASK]", "fails", "on", "[MASK]"]
    guessed = ["the", "test", "fails", "on", "arm64"]
    verdict = ["original", "replaced", "original", "original", "original"]
    fig = Figure(
        760,
        330,
        alt="Replaced token detection. A sentence has some tokens masked; a small generator fills each mask with a "
        "plausible word, sometimes the right one; the discriminator reads the result and labels every token as "
        "original or replaced, so it learns from all of them",
    )
    fig.text(14, 28, "ELECTRA: a small model fills the gaps, the big one spots the fakes", kind="title")
    fig.text(14, 50, "the generator guessed arm64 right, so that token counts as original", kind="note")
    rows = [("text", words, "neutral"), ("masked", masked, "data"), ("generator's guess", guessed, "data"),
            ("discriminator says", verdict, "enc")]
    for k, (label, tokens, role) in enumerate(rows):
        y = 74 + k * 62
        fig.text(14, y + 20, label, kind="label")
        x = 170
        for i, token in enumerate(tokens):
            highlight = (k == 1 and token == "[MASK]") or (k == 2 and i in (1, 4)) or (k == 3 and token == "replaced")
            this_role = "dec" if (k == 3 and token == "replaced") else role
            w = max(fig.chip_width(token), 96)
            fig.box(x, y, w, 30, role=this_role, label=token, strong=highlight and k != 3)
            x += w + 8
        if k < len(rows) - 1:
            fig.arrow((150, y + 32), (150, y + 58))
    fig.text(14, 316, "loss: the generator's masked LM + 50 × the discriminator's yes/no, at every token", kind="note")
    return fig


# ---------------------------------------------------------------- DeBERTa's three terms

def deberta() -> Figure:
    fig = Figure(
        760,
        300,
        alt="DeBERTa's attention score between two tokens as the sum of three terms: content to content, as in "
        "BERT; content to position, the query of the first token against the embedding of their relative distance; "
        "and position to content, the key of the second token against the distance seen from the other side",
    )
    fig.text(14, 28, "DeBERTa: an attention score in three parts", kind="title")
    fig.text(14, 50, "token i looks at token j; δ is their distance, put into one of 256 buckets", kind="note")
    terms = [
        ("content → content", "q of word i · k of word j", "what i looks for, what j is", "plain"),
        ("content → position", "q of word i · k of distance i→j", "what i looks for at that distance", "att"),
        ("position → content", "k of word j · q of distance j→i", "what j offers at that distance", "att"),
    ]
    for k, (name, formula, meaning, role) in enumerate(terms):
        y = 76 + k * 58
        fig.box(14, y, 200, 40, role=role, label=name)
        fig.text(232, y + 18, formula, kind="mono")
        fig.text(232, y + 36, meaning, kind="note")
        if k < 2:
            fig.text(110, y + 54, "+", kind="subtitle", anchor="middle")
    fig.box(14, 254, 732, 32, role="neutral", label="sum ÷ √(3 · head size), then softmax over j, as usual")
    return fig


# ---------------------------------------------------------------- ModernBERT's layers

def modernbert() -> Figure:
    c = config(MODERNBERT)
    n, every = c["num_hidden_layers"], c["global_attn_every_n_layers"]
    fig = Figure(
        760,
        300,
        alt=f"ModernBERT-base's {n} layers: every {every}rd layer, starting with the first, attends globally with "
        f"rotary positions of base {c['global_rope_theta']:,.0f}; the others attend to a window of "
        f"{c['local_attention']} tokens with rotary base {c['local_rope_theta']:,.0f}. Each layer is pre-norm, without "
        "biases, with a GeGLU feed-forward network",
    )
    fig.text(14, 28, f"ModernBERT-base: {n} layers, every {every}rd one global", kind="title")
    fig.text(14, 50, f"read from its config: hidden {c['hidden_size']}, GeGLU {c['intermediate_size']} wide, "
             f"{c['max_position_embeddings']:,} tokens", kind="note")
    width = 31
    for i in range(n):
        x = 14 + i * (width + 2)
        is_global = i % every == 0
        fig.box(x, 76, width, 60, role="att" if is_global else "enc", label="G" if is_global else "L",
                strong=is_global)
        if i % 3 == 0:
            fig.text(x + width / 2, 152, str(i + 1), kind="small", anchor="middle")
    fig.box(14, 178, 360, 34, role="att", label=f"G: global, RoPE base {c['global_rope_theta']:,.0f}")
    fig.box(386, 178, 360, 34, role="enc",
            label=f"L: window of {c['local_attention']} tokens, RoPE base {c['local_rope_theta']:,.0f}")
    fig.text(14, 240, "every layer: x + attention(LayerNorm(x)), then x + GeGLU(LayerNorm(x)); no biases",
             kind="body")
    fig.text(14, 264, "the first layer skips its attention norm: the embeddings were just normalised", kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(MODERNBERT)
    rows = [
        ("published", "December 2024, Answer.AI and LightOn (Warner, Chaffin, Clavié and others)"),
        ("type", "encoder only, a drop-in replacement for BERT"),
        ("pre-training", "masked language model, 30% of tokens, 2 trillion tokens of text and code"),
        ("attention", f"{c['num_attention_heads']} heads; every {c['global_attn_every_n_layers']}rd layer global, "
                      f"the others a {c['local_attention']}-token window"),
        ("positions", f"rotary (RoPE), up to {c['max_position_embeddings']:,} tokens"),
        ("normalisation", "LayerNorm before each sublayer (pre-norm); no biases but the decoder's"),
        ("activation", f"GeGLU: GELU of one {c['intermediate_size']:,}-wide half times the other"),
        ("vocabulary", f"BPE, {c['vocab_size']:,} tokens, code-aware"),
        ("sizes", f"base: {c['num_hidden_layers']} layers, {c['hidden_size']} wide, 149M; large: 28 layers, 395M"),
        ("good at", "classification, retrieval, long documents, code; fast with FlashAttention"),
        ("not for", "generating text"),
    ]
    return arch.passport(
        "ModernBERT", rows,
        alt="ModernBERT's passport: published December 2024 by Answer.AI and LightOn; encoder; masked language model "
        "with 30% of tokens masked, on 2 trillion tokens of text and code; attention alternating global and local "
        "windows of 128 tokens; rotary positions, 8,192 tokens; pre-norm LayerNorm, no biases except the vocabulary "
        "decoder; GeGLU; base 149 million parameters and 22 layers, large 395 million and 28 layers",
    )


# ---------------------------------------------------------------- cover

def _cover():
    names = ["BERT", "RoBERTa", "ELECTRA", "DeBERTa", "ModernBERT"]
    changes = ["2018", "recipe", "task", "attention", "the block"]
    out = [cv.label(cv.MOTIF_X0, 150, "SEVEN YEARS OF ENCODERS")]
    for k, (name, change) in enumerate(zip(names, changes, strict=True)):
        y = 182 + k * 62
        chip, w = cv.chip(cv.MOTIF_X0, y, name, size=19, kind="hot" if k == len(names) - 1 else "plain")
        out.append(chip)
        out.append(cv.text(cv.MOTIF_X0 + 230, y + 27, change, 19, cv.MUTED, cv.MONO))
        cv.fits(cv.MOTIF_X0 + 230 + cv.chip_w(change, 19))
    return cv.frame("a02_modernbert.ipynb", ["RoBERTa to", "ModernBERT"],
                    ["What seven years", "did to BERT"], "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "electra": electra,
    "deberta": deberta,
    "modernbert": modernbert,
    "passport": passport,
    "family": lambda: arch.family(2),
}
COVERS = {"en": _cover}
