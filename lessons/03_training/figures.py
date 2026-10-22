"""Static figures and covers of lesson 3.

Facts about real systems are read at render time: the decoding settings from the models' generation_config.json
files on the Hugging Face Hub, the number formats from torch.finfo. The beam search figure is a toy model whose
probabilities are defined here, and every product it shows is computed from them.
"""

import json

from tools.figures import cover as cv
from tools.figures.lib import Figure

# ---------------------------------------------------------------- hero: how real models decode

DECODERS = [  # (group, name, repository)
    ("search", "opus-mt-en-ru", "Helsinki-NLP/opus-mt-en-ru"),
    ("search", "m2m100 418M", "facebook/m2m100_418M"),
    ("sample", "Llama 3.1 8B Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("sample", "Qwen 2.5 7B Instruct", "Qwen/Qwen2.5-7B-Instruct"),
]
SETTINGS = {"num_beams": "beams", "temperature": "temperature", "top_p": "top-p", "top_k": "top-k"}


def generation_settings(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "generation_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return {key: config[key] for key in SETTINGS if key in config}


def _tree_icon(fig, x, y):
    """A tiny beam: one root, two kept branches, each with two children."""
    fig.line((x, y + 20), (x + 26, y + 6), role="data", width=2)
    fig.line((x, y + 20), (x + 26, y + 34), role="data", width=2)
    for by in (y + 6, y + 34):
        fig.line((x + 26, by), (x + 50, by - 6), role="muted", width=1.2)
        fig.line((x + 26, by), (x + 50, by + 6), role="muted", width=1.2)
    fig.rect(x - 3, y + 17, 6, 6, role="data", r=3)


def _dice_icon(fig, x, y):
    """A tiny distribution with one bar drawn at random."""
    heights = [30, 22, 16, 10, 6]
    for k, h in enumerate(heights):
        fig.rect(x + k * 10, y + 40 - h, 7, h, role="data", tint=k != 1, r=1)


def hero() -> Figure:
    groups = {"search": [], "sample": []}
    for group, name, repo in DECODERS:
        groups[group].append((name, generation_settings(repo)))
    fig = Figure(
        760,
        300,
        alt="Decoding settings shipped with real models: the translators opus-mt-en-ru and m2m100 search with "
        "beams, the chat models Llama 3.1 Instruct and Qwen 2.5 Instruct sample with a temperature",
    )
    fig.text(14, 28, "Translators search, chat models sample", kind="title")
    fig.text(14, 50, "the decoding settings each model ships with, from its generation_config.json", kind="note")
    columns = [(14, "search", "translators: beam search", _tree_icon), (394, "sample", "chat models: sampling", _dice_icon)]
    for x0, group, title, icon in columns:
        fig.text(x0, 88, title, kind="subtitle")
        icon(fig, x0 + 290, 62)
        for k, (name, settings) in enumerate(groups[group]):
            y = 112 + k * 86
            fig.box(x0, y, 352, 74, role="plain", r=10)
            fig.text(x0 + 16, y + 26, name, kind="body")
            x = x0 + 16
            for key, value in settings.items():
                w = fig.chip(x, y + 36, f"{SETTINGS[key]} {value}", role="data", h=26, size=12)
                x += w + 6
            assert x <= x0 + 352, f"{name}: settings don't fit"
    return fig


# ---------------------------------------------------------------- one training step

def step() -> Figure:
    fig = Figure(
        760,
        300,
        alt="One training step: a batch of sentence pairs goes through the Transformer, the scores are compared "
        "with the targets by cross-entropy with label smoothing, the loss is backpropagated into gradients, the "
        "gradient norm is clipped, and Adam updates the weights with the learning rate from the schedule; then "
        "the next batch",
    )
    fig.text(14, 28, "One training step, repeated thousands of times", kind="title")
    w, h = 170, 40
    top, bottom = 76, 196
    xs = [44, 300, 556]
    fig.box(xs[0], top, w, h, role="data", label="batch of pairs")
    fig.box(xs[1], top, w, h, role="att", label="Transformer")
    fig.box(xs[2], top, w, h, role="neutral", label="loss")
    fig.box(xs[2], bottom, w, h, role="neutral", label="backward: gradients")
    fig.box(xs[1], bottom, w, h, role="neutral", label="clip the gradients")
    fig.box(xs[0], bottom, w, h, role="neutral", label="Adam step")
    mid = h / 2
    fig.arrow((xs[0] + w + 2, top + mid), (xs[1] - 4, top + mid))
    fig.arrow((xs[1] + w + 2, top + mid), (xs[2] - 4, top + mid))
    fig.arrow((xs[2] + w / 2, top + h + 2), (xs[2] + w / 2, bottom - 4))
    fig.arrow((xs[2] - 2, bottom + mid), (xs[1] + w + 4, bottom + mid))
    fig.arrow((xs[1] - 2, bottom + mid), (xs[0] + w + 4, bottom + mid))
    fig.arrow((xs[0] + w / 2 + 40, bottom - 2), (xs[1] + 30, top + h + 4), role="data")
    fig.text(xs[1] + w / 2, top - 10, "scores for every token", kind="label", anchor="middle")
    fig.text(xs[2] + w / 2, top - 10, "cross-entropy, smoothed", kind="label", anchor="middle")
    fig.text(xs[0] + w / 2, bottom + h + 20, "learning rate from the schedule", kind="label", anchor="middle")
    fig.text(xs[1] + w / 2, bottom + h + 20, "no single step too large", kind="label", anchor="middle")
    fig.text(xs[2] + w / 2, bottom + h + 20, "in float16, scaled up", kind="label", anchor="middle")
    fig.text(xs[0] + w / 2 + 66, 160, "new weights", kind="label", role="data")
    fig.text(14, 286, "Sections 3 to 6 build the parts; section 7 runs the loop.", kind="note")
    return fig


# ---------------------------------------------------------------- number formats

FORMATS = [("float32", 8, 23), ("bfloat16", 8, 7), ("float16", 5, 10)]


def formats() -> Figure:
    import torch

    names = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    fig = Figure(
        760,
        300,
        alt="Bit layouts: float32 has 1 sign bit, 8 exponent bits and 23 mantissa bits; bfloat16 1, 8 and 7; "
        "float16 1, 5 and 10. float16 covers a much smaller range, so tiny gradients underflow to zero unless the "
        "loss is scaled up first",
    )
    fig.text(14, 28, "Three ways to store a number", kind="title")
    fig.text(14, 50, "sign · exponent (range) · mantissa (precision)", kind="note")
    cell = 13
    for r, (name, exponent, mantissa) in enumerate(FORMATS):
        info = torch.finfo(names[name])
        y = 80 + r * 62
        fig.text(14, y + 16, name, kind="mono")
        x = 110
        fig.rect(x, y, cell - 2, 22, role="dec", r=2)
        x += cell
        for _ in range(exponent):
            fig.rect(x, y, cell - 2, 22, role="att", r=2)
            x += cell
        for _ in range(mantissa):
            fig.rect(x, y, cell - 2, 22, role="data", tint=True, r=2)
            x += cell
        fig.text(110, y + 40, f"largest {info.max:.3g}, smallest normal {info.smallest_normal:.2g}", kind="small")
    fig.text(14, 280, "Free T4 and P100 GPUs are fast in float16, not in bfloat16: we take float16 and scale the loss.",
             kind="note")
    return fig


# ---------------------------------------------------------------- greedy versus beam search, on a toy model

TOY = {  # probabilities of the next token after each prefix; a three-token toy model
    (): {"A": 0.5, "B": 0.4, "C": 0.1},
    ("A",): {"x": 0.4, "y": 0.35, "z": 0.25},
    ("B",): {"x": 0.05, "y": 0.05, "z": 0.9},
}


def beam_toy() -> Figure:
    first = TOY[()]
    greedy_first = max(first, key=first.get)
    greedy_second = max(TOY[(greedy_first,)], key=TOY[(greedy_first,)].get)
    greedy_p = first[greedy_first] * TOY[(greedy_first,)][greedy_second]
    kept = sorted(first, key=first.get, reverse=True)[:2]  # beam of two
    finals = []
    for a in kept:
        for b, p in TOY[(a,)].items():
            finals.append((first[a] * p, a, b))
    best_p, best_a, best_b = max(finals)
    assert best_p > greedy_p, "the toy must show greedy losing"
    fig = Figure(
        760,
        330,
        alt=f"A toy model. Greedy decoding takes the most likely first token, {greedy_first}, then its best "
        f"continuation, {greedy_second}: probability {greedy_p:.2f}. Beam search keeps the two best first tokens and "
        f"finds {best_a} then {best_b}, probability {best_p:.2f}",
    )
    fig.text(14, 28, "Greedy takes the best next token; beam search keeps several", kind="title")
    root = (40, 170)
    level1 = {"A": 92, "B": 172, "C": 252}
    fig.box(root[0] - 18, root[1] - 16, 36, 32, role="neutral", label="<bos>", r=7)
    fig.els[-1].size = 10
    for token, y in level1.items():
        on_beam = token in kept
        fig.line((root[0] + 20, root[1]), (200, y), role="data" if on_beam else "muted", width=2 if on_beam else 1)
        fig.chip(202, y - 15, token, role="data" if on_beam else "plain", strong=token == best_a)
        fig.text(216, y - 22, f"{first[token]:.2f}", kind="count", anchor="middle")  # above the chip
    leaf_y = {}
    for token in kept:
        for k, (nxt, p) in enumerate(TOY[(token,)].items()):
            y = level1[token] - 26 + k * 26 if token == "A" else level1[token] - 4 + k * 26
            leaf_y[(token, nxt)] = y
            best = (token, nxt) == (best_a, best_b)
            greedy = (token, nxt) == (greedy_first, greedy_second)
            fig.line((244, level1[token]), (400, y), role="data" if best else "muted", width=2 if best else 1)
            fig.chip(402, y - 13, nxt, role="data" if best or greedy else "plain", strong=best, h=26)
            product = f"{first[token]:.2f} × {p:.2f} = {first[token] * p:.2f}"
            fig.text(450, y + 5, product, kind="count_hot" if best else "count")
            if greedy:
                fig.text(620, y + 5, "← greedy", kind="label")
            if best:
                fig.text(620, y + 5, "← beam of 2", kind="label", role="data")
    fig.text(202, 290, f"greedy: {greedy_first}, then {greedy_second}, probability {greedy_p:.2f}", kind="note")
    fig.text(202, 312, f"beam search: {best_a}, then {best_b}, probability {best_p:.2f}", kind="note")
    return fig


# ---------------------------------------------------------------- training reads the answer, translating reads itself

EN_EXAMPLE = "The researcher said she was exhausted."
REFERENCE = "Исследовательница сказала, что вымоталась."


def opus_translation(sentence):
    """What opus-mt-en-ru writes for a sentence, with its own default decoding (4 beams)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    with torch.no_grad():
        output = model.generate(**tokenizer([sentence], return_tensors="pt"))
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _words(text):
    """Words with their punctuation split off: "сказала," -> "сказала", ","."""
    words = []
    for word in text.split():
        if word[-1] in ",.":
            words += [word[:-1], word[-1]]
        else:
            words.append(word)
    return words


def exposure() -> Figure:
    produced = opus_translation(EN_EXAMPLE)
    reference, written = _words(REFERENCE), _words(produced)
    fig = Figure(
        760,
        300,
        alt=f"In training the decoder reads the reference translation, {REFERENCE} When translating, it reads its "
        f"own previous words. opus-mt-en-ru writes: {produced} Its first word is masculine, and the next word "
        "agrees with that mistake",
    )
    fig.text(14, 28, "Training reads the answer; translating reads its own guesses", kind="title")
    fig.text(14, 50, f"English: {EN_EXAMPLE}", kind="note")

    def row(y, label, words, wrong):
        fig.text(14, y + 20, label, kind="label")
        x = 190
        spans = []
        for word in ["<bos>", *words]:
            w = Figure.chip_width(word, 13)
            hot = word in wrong
            fig.chip(x, y, word, role="dec" if hot else "data", strong=hot)
            spans.append((x, w))
            x += w + 6
        assert x <= 750, f"row reaches {x:.0f}"
        return spans

    row(78, "training reads", reference, set())
    fig.text(190, 132, "the reference, word by word: every next prediction starts from the right history",
             kind="small")
    # a wrong form, not a paraphrase: a word that is a shorter form of a reference word (Исследователь|ница)
    wrong = {w for w in written if any(r.startswith(w) and r != w for r in reference)}
    spans = row(176, "translating reads", written, wrong)
    first_wrong = next(i for i, w in enumerate(written) if w in wrong) + 1  # +1 for <bos>
    x, w = spans[first_wrong]
    fig.text(190, 232, "its own output: a masculine first word, and the verb agrees with the mistake", kind="small")
    fig.text(14, 282, f"Real output of opus-mt-en-ru, our lesson 4 baseline: {produced}", kind="note")
    return fig


FIGURES = {"hero": hero, "step": step, "formats": formats, "beam-toy": beam_toy, "exposure": exposure}
