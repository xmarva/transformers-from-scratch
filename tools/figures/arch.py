"""Figures shared by the lessons of the second course, Transformer architectures: the family tree that grows by one
branch per lesson, and the architecture's passport, the same card at the end of every lesson."""

from tools.figures.lib import Figure

COURSE = "Transformer architectures: from BERT to diffusion"

# the course's lessons by family, in the order of the course within each family
FAMILY = {
    "encoders": [(1, "BERT"), (2, "RoBERTa→ModernBERT"), (3, "embeddings & search")],
    "decoders": [(4, "GPT"), (6, "modern LLMs"), (7, "mixture of experts"), (8, "KV cache & MLA"),
                 (9, "long context"), (10, "Mamba & hybrids")],
    "encoder–decoders": [(5, "T5 & BART"), (14, "Whisper")],
    "beyond text": [(11, "ViT"), (12, "CLIP & SigLIP"), (13, "vision–language"), (15, "diffusion transformers")],
}


def family(current: int) -> Figure:
    """The family tree with lesson `current` highlighted and the earlier lessons marked as done."""
    names = {n: name for members in FAMILY.values() for n, name in members}
    fig = Figure(
        760,
        424,
        alt=f"The Transformer's family as this course covers it, lesson {current}, {names[current]}, highlighted. "
        "From the 2017 Transformer: encoders (BERT, RoBERTa to ModernBERT, embeddings and search), decoders (GPT, "
        "modern LLMs, mixture of experts, the KV cache and latent attention, long context, Mamba and hybrids), "
        "encoder–decoders (T5 and BART, Whisper) and models beyond text (ViT, CLIP and SigLIP, vision–language "
        "models, diffusion transformers)",
    )
    fig.text(14, 28, "The family of the Transformer, as this course meets it", kind="title")
    fig.text(14, 50, f"lesson {current} highlighted; the number is the lesson", kind="note")
    fig.box(290, 70, 180, 34, role="plain", label="Transformer, 2017")
    width, gap, top = 176, 12, 150
    for k, (title, members) in enumerate(FAMILY.items()):
        x = 14 + k * (width + gap)
        fig.line((380, 104), (380, 118), (x + width / 2, 118), (x + width / 2, top - 22))
        fig.text(x + width / 2, top - 8, title, kind="subtitle", anchor="middle")
        for i, (number, name) in enumerate(members):
            y = top + 6 + i * 40
            role = "att" if number == current else ("enc" if number < current else "neutral")
            fig.box(x, y, width, 32, role=role, label=f"{number:02d}  {name}", strong=number == current)
    return fig


def passport(name: str, rows: list[tuple[str, str]], alt: str) -> Figure:
    """The architecture's passport: the same fields in every lesson, so that models compare side by side."""
    fig = Figure(760, 84 + len(rows) * 34 + 14, alt=alt)
    fig.text(14, 28, f"{name}: the passport", kind="title")
    fig.text(14, 50, "the same card ends every lesson of the course", kind="note")
    for k, (field, value) in enumerate(rows):
        y = 70 + k * 34
        fig.text(14, y + 19, field, kind="label")
        fig.box(190, y, 556, 28, role="att" if k == 0 else "plain", label=None)
        fig.text(200, y + 19, value, kind="body")
    return fig
