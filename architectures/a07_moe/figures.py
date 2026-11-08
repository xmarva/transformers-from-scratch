"""Static figures and cover of the second course's lesson 7, mixtures of experts."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

OLMOE = "allenai/OLMoE-1B-7B-0924"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: one layer of experts

HERO_ALT = ("A mixture-of-experts layer. A token's vector goes to the router, a linear layer that scores all 64 "
            "experts; the softmax turns the scores into probabilities, and the eight largest pick the experts that "
            "run. Each chosen expert is a small SwiGLU network; their outputs are added, each weighted by its "
            "probability, and the result joins the residual stream. The other 56 experts do nothing for this token")


def hero() -> Figure:
    fig = Figure(760, 352, alt=HERO_ALT)
    fig.text(14, 28, "One layer of OLMoE, for one token", kind="title")
    fig.text(14, 50, "64 experts wait in every layer; the router wakes eight of them", kind="note")
    fig.box(14, 92, 110, 44, role="data", label="token vector")
    fig.arrow((125, 114), (147, 114))
    fig.box(148, 92, 120, 44, role="att", label="router  W·x", strong=True)
    fig.arrow((269, 114), (291, 114))
    fig.box(292, 92, 136, 44, role="att", label="softmax, top 8")
    fig.text(148, 160, "64 scores, one per expert", kind="small")
    chosen = {3, 9, 17, 22, 30, 41, 50, 58}
    x0, y0, cell, gap = 462, 66, 30, 6
    for e in range(64):
        row, col = divmod(e, 8)
        x, y = x0 + col * (cell + gap), y0 + row * 16
        fig.box(x, y, cell, 12, role="enc" if e in chosen else "neutral", label=None, strong=e in chosen)
    fig.text(462, 206, "64 experts; the eight chosen are dark", kind="small")
    fig.arrow((429, 114), (455, 114))
    fig.box(292, 232, 136, 44, role="enc", label="8 SwiGLU experts")
    fig.arrow((360, 137), (360, 231))
    fig.text(256, 190, "only these run", kind="small")
    fig.arrow((429, 254), (471, 254))
    fig.box(472, 232, 150, 44, role="neutral", label="Σ  weight · output")
    fig.arrow((623, 254), (651, 254))
    fig.box(652, 232, 94, 44, role="data", label="+ residual")
    fig.text(14, 312, "the weights are the eight probabilities, not renormalised: an unsure router sends a smaller "
                      "update", kind="note")
    fig.text(14, 334, "the other 56 experts cost memory but no compute for this token", kind="note")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("OLMoE's passport. Published September 2024 by the Allen Institute for AI, fully open: weights, "
                "data, code and logs, Apache 2.0. Type: decoder-only mixture of experts, 6.9 billion parameters, "
                "1.3 billion active per token. Training: 5.1 trillion tokens, mostly English. Attention: 16 heads "
                "of 128, multi-head, QK-norm over the whole query and key projections. Positions: rotary, base "
                "10,000. Experts: 64 SwiGLU experts of width 1,024 in each of 16 layers, 8 per token, no shared "
                "expert. Router: softmax over all experts, top 8, not renormalised; trained with a load-balancing "
                "loss and a z-loss. Vocabulary: 50,304 tokens, untied input and output. Good at: studying routing, "
                "English text at the cost of a 1B model. Not for: languages it barely saw, long contexts of more "
                "than 4,096 tokens")


def passport() -> Figure:
    c = config(OLMOE)
    rows = [
        ("published", "September 2024, Allen Institute for AI; all open, Apache 2.0"),
        ("type", "decoder-only mixture; 6.9B parameters, 1.3B per token"),
        ("training", "5.1 trillion tokens, mostly English"),
        ("attention", f"{c['num_attention_heads']} heads of {c['hidden_size'] // c['num_attention_heads']}; "
                      "QK-norm on whole projections"),
        ("positions", f"rotary, base {int(c.get('rope_theta', 10000)):,}"),
        ("experts", f"{c['num_experts']} of width {c['intermediate_size']:,} per layer, "
                    f"{c['num_experts_per_tok']} per token, none shared"),
        ("router", "softmax, top 8, not renormalised"),
        ("vocabulary", f"{c['vocab_size']:,} tokens; input and output not tied"),
        ("good at", "studying routing; English at a 1B cost"),
        ("not for", f"rare languages; contexts over {c['max_position_embeddings']:,} tokens"),
    ]
    return arch.passport("OLMoE", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "64 EXPERTS IN A LAYER, 8 PER TOKEN")]
    chosen = {3, 9, 17, 22, 30, 41, 50, 58}
    size, gap = 24, 8
    for e in range(64):
        row, col = divmod(e, 16)
        x = cv.MOTIF_X0 + col * (size + gap)
        y = 176 + row * (size + gap)
        fill, stroke = (cv.RED, cv.RED) if e in chosen else ("none", cv.LINE)
        out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="5" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1.4"/>')
    cv.fits(cv.MOTIF_X0 + 16 * (size + gap))
    out.append(cv.label(cv.MOTIF_X0, 340, "WHICH ONES DEPENDS ON THE TEXT"))
    svg, spans, right = cv.chip_row(["English", "Python", "maths", "Russian"], cv.MOTIF_X0, 362, size=17,
                                    kinds={1: "hot"})
    cv.fits(right)
    out.append(svg)
    return cv.frame("a07_moe.ipynb", ["Mixture", "of experts"], ["Routing, pruning and caching", "in a real MoE"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "passport": passport,
    "family": lambda: arch.family(7),
}
COVERS = {"en": _cover}
