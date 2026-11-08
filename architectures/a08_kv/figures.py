"""Static figures and cover of the second course's lesson 8, the KV cache and latent attention."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

DEEPSEEK_V2 = "deepseek-ai/DeepSeek-V2"

# numbers one token keeps per layer, for 32 heads of 128 (and DeepSeek's latent of 512 + 64)
DESIGNS = [
    ("multi-head", "a key and a value per head", 2 * 32 * 128, "dec"),
    ("grouped-query, 8 groups", "a key and a value per group", 2 * 8 * 128, "data"),
    ("latent", "a latent of 512, a rotary key of 64", 512 + 64, "att"),
    ("multi-query", "one key and one value", 2 * 128, "enc"),
]


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: one token in the cache

HERO_ALT = ("What one token keeps in the cache under four designs, drawn to scale for 32 heads of 128. Multi-head "
            "attention keeps a key and a value for every head, 8,192 numbers per layer. Grouped-query attention "
            "with 8 groups keeps 2,048. Latent attention keeps one compressed vector of 512 and a rotary key of 64, "
            "576 numbers, from which every head's key and value are rebuilt. Multi-query attention keeps one key and "
            "one value, 256 numbers")


def hero() -> Figure:
    fig = Figure(760, 344, alt=HERO_ALT)
    fig.text(14, 28, "What one token leaves in the cache of one layer", kind="title")
    fig.text(14, 50, "32 query heads of 128 numbers; the bars are to scale", kind="note")
    widest = max(n for _, _, n, _ in DESIGNS)
    for k, (name, what, numbers, role) in enumerate(DESIGNS):
        y = 78 + k * 58
        fig.text(14, y + 16, name, kind="subtitle")
        fig.text(14, y + 36, what, kind="small")
        width = max(6, 370 * numbers / widest)
        fig.box(300, y, width, 34, role=role, label=None, strong=True)
        fig.text(300 + width + 10, y + 22, f"{numbers:,}", kind="body")
    fig.text(14, 330, "latent attention rebuilds every head's key and value from its 576 numbers when it needs them",
             kind="note")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("DeepSeek-V2's passport. Published May 2024 by DeepSeek-AI with open weights. Type: decoder-only "
                "mixture of experts, 236 billion parameters, 21 billion per token, 160 routed and 2 shared experts, "
                "6 routed per token. Training: 8.1 trillion tokens. Attention: multi-head latent attention, 128 "
                "heads, a latent of 512 numbers and a rotary key of 64 per token and layer; queries compressed to "
                "1,536. Positions: decoupled rotary dimensions, 64 per head. Cache: 576 numbers per token per layer, "
                "as much as grouped-query attention with 2.25 groups. Vocabulary: 102,400 tokens. Good at: long "
                "conversations served to many users at once. Not for: a single GPU")


def passport() -> Figure:
    c = config(DEEPSEEK_V2)
    rows = [
        ("published", "May 2024, DeepSeek-AI; open weights"),
        ("type", "mixture of experts; 236B parameters, 21B per token"),
        ("training", "8.1 trillion tokens"),
        ("attention", f"latent, {c['num_attention_heads']} heads, latent {c['kv_lora_rank']}, "
                      f"rotary key {c['qk_rope_head_dim']}"),
        ("queries", f"compressed to {c['q_lora_rank']:,}, then {c['qk_nope_head_dim']} + "
                    f"{c['qk_rope_head_dim']} per head"),
        ("cache", f"{c['kv_lora_rank'] + c['qk_rope_head_dim']} numbers per token per layer"),
        ("experts", f"{c['n_routed_experts']} routed and {c['n_shared_experts']} shared, "
                    f"{c['num_experts_per_tok']} per token"),
        ("vocabulary", f"{c['vocab_size']:,} tokens"),
        ("good at", "long chats served to many users at once"),
        ("not for", "a single GPU"),
    ]
    return arch.passport("DeepSeek-V2", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "NUMBERS ONE TOKEN KEEPS PER LAYER")]
    widest = max(n for _, _, n, _ in DESIGNS)
    names = {"multi-head": "MHA", "grouped-query, 8 groups": "GQA", "latent": "MLA", "multi-query": "MQA"}
    for k, (name, _, numbers, _) in enumerate(DESIGNS):
        y = 176 + k * 52
        out.append(cv.text(cv.MOTIF_X0, y + 24, names[name], 18, cv.MUTED, cv.MONO))
        width = max(8, 330 * numbers / widest)
        color = cv.RED if name == "latent" else cv.LINE
        out.append(f'<rect x="{cv.MOTIF_X0 + 70}" y="{y}" width="{width:.0f}" height="34" rx="6" '
                   f'fill="{color}"/>')
        out.append(cv.text(cv.MOTIF_X0 + 80 + width, y + 24, f"{numbers:,}", 17, cv.MUTED, cv.MONO))
    cv.fits(cv.MOTIF_X0 + 80 + 330 + 70)
    return cv.frame("a08_kv.ipynb", ["The KV cache", "budget"], ["MQA, GQA and DeepSeek's", "latent attention"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "passport": passport,
    "family": lambda: arch.family(8),
}
COVERS = {"en": _cover}
