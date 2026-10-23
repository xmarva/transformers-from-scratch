"""Figures and charts of the portfolio page "Medical Assistant" on xmarva.github.io.

The data is public and saved in data/ by fetch.py: PubMed counts from NCBI E-utilities, a piece of Hetionet v1.0
(Himmelstein et al., eLife 2017) around type 2 diabetes, and recent trials on metformin. Identifiers in the
normalisation figure were checked against the MeSH, RxNorm, WHO ICD-10 and SNOMED CT services. The drawing of the
assistant's screen is a reconstruction: the project's own screenshots were not kept.

    uv run python tools/render.py --article medical_assistant \\
        --out <site>/assets/img/projects/1_medical_assistant/figures
"""

import json
from functools import cache
from html import escape
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"


@cache
def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975, gap=0.075):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - gap, note, ha="left", va="top", fontsize=10, color=c["muted"])


# ---------------------------------------------------------------- the literature

def chart_growth(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    data = load("pubmed_counts")["per_year"]
    years = np.array(sorted(int(y) for y in data["all"]))
    total = np.array([data["all"][str(y)] for y in years])
    trials = np.array([data["randomized controlled trials"][str(y)] for y in years])
    meta = np.array([data["meta-analyses"][str(y)] for y in years])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.14, wspace=0.28)
    left.bar(years, total / 1e6, color=c["att"], width=0.75)
    left.bar(years[-2:], total[-2:] / 1e6, color=c["dec"], width=0.75)
    left.annotate(f"{total[-1] / 1e6:.2f} million\nin {years[-1]}", (years[-1], total[-1] / 1e6),
                  xytext=(-70, 4), textcoords="offset points", fontsize=9, color=c["ink"])
    left.set_title("articles per year", loc="left", fontsize=10)
    left.set_ylabel("millions")
    right.plot(years, trials / 1e3, color=c["enc"], linewidth=2.2)
    right.plot(years, meta / 1e3, color=c["data"], linewidth=2.2)
    right.text(2002, trials[12] / 1e3 + 2.0, "randomized trials", ha="right", fontsize=9, color=c["enc_ink"])
    right.text(2009, meta[19] / 1e3 + 1.8, "meta-analyses", ha="right", fontsize=9, color=c["data_ink"])
    right.set_title("of them, the strongest evidence", loc="left", fontsize=10)
    right.set_ylabel("thousands")
    right.set_ylim(0, None)
    for ax in (left, right):
        ax.set_xticks([1990, 2000, 2010, 2018])
    growth = meta[-1] / meta[0]
    _caption(fig, "PubMed by year of publication, 1990 to 2018",
             f"rose: the project's years; meta-analyses grew {growth:.0f}-fold, all articles "
             f"{total[-1] / total[0]:.1f}-fold (counts as indexed today)", c)
    return fig


# ---------------------------------------------------------------- entities

SENTENCE = ("Treatment with linagliptin resulted in noninferior glycemic control and in significantly lower risk of "
            "hypoglycemia compared to insulin glargine in long-term care and skilled nursing facility residents "
            "with type 2 diabetes .").split()
SPANS = {(2, 3): "DRUG", (14, 15): "DISEASE", (17, 19): "DRUG", (28, 31): "DISEASE"}
CODES = {(2, 3): "RxNorm 1100699", (14, 15): "MeSH D007003", (17, 19): "RxNorm 274783",
         (28, 31): "MeSH D003924 · ICD-10 E11"}
PMID = "29289540"


def bio_tags():
    tags = ["O"] * len(SENTENCE)
    for (a, b), kind in SPANS.items():
        tags[a] = f"B-{kind}"
        for k in range(a + 1, b):
            tags[k] = f"I-{kind}"
    return tags


def ner() -> Figure:
    tags = bio_tags()
    fig = Figure(
        760,
        368,
        alt="A sentence from a 2018 trial abstract split into tokens, each with a BIO tag below it. Linagliptin is "
        "B-DRUG; insulin glargine is B-DRUG I-DRUG; hypoglycemia is B-DIS, for disease; type 2 diabetes is B-DIS "
        "I-DIS I-DIS; every other token is O. Under each entity is the identifier it is normalised to: "
        "RxNorm 1100699 for linagliptin, RxNorm 274783 for insulin glargine, MeSH D007003 for hypoglycemia, and "
        "MeSH D003924 and ICD-10 E11 for type 2 diabetes",
    )
    fig.text(14, 28, "Entities in one sentence, as BIO tags", kind="title")
    fig.text(14, 50, f"a sentence from a 2018 randomized trial (PubMed {PMID}); violet: drugs, rose: diseases (DIS)",
             kind="note")
    roles = []
    for t in tags:
        roles.append("att" if t.endswith("DRUG") else "dec" if t.endswith("DISEASE") else "neutral")
    x, y, line_h = 14, 76, 72
    starts = {}
    span_end = {a: b for a, b in SPANS}
    for k, (token, tag, role) in enumerate(zip(SENTENCE, tags, roles, strict=True)):
        w = Figure.chip_width(token, 13)
        whole = sum(Figure.chip_width(t, 13) + 6 for t in SENTENCE[k:span_end.get(k, k + 1)]) - 6
        if x + whole > 746:  # an entity is never split across lines
            x, y = 14, y + line_h
        fig.chip(x, y, token, role=role, strong=role != "neutral", h=28)
        fig.text(x + w / 2, y + 44, tag.replace("DISEASE", "DIS"), kind="small", anchor="middle")
        starts[k] = (x, y, w)
        x += w + 6
    for (a, b), code in CODES.items():
        xa, ya, _ = starts[a]
        xb, _, wb = starts[b - 1]
        middle = max((xa + xb + wb) / 2, 14 + len(code) * 13 * 0.6 / 2)
        fig.text(middle, ya + 60, code, kind="mono_att" if "RxNorm" in code else "mono_dec", anchor="middle")
    return fig


NORMAL = [  # surface forms, concept, (vocabulary, code, note)
    (["heart attack", "MI", "myocardial infarction", "acute MI"], "myocardial infarction",
     [("SNOMED CT", "22298006", ""), ("MeSH", "D009203", ""), ("UMLS", "C0027051", ""),
      ("ICD-10", "I21", "acute only")]),
    (["Glucophage", "metformin HCl", "metformin 500 mg", "metformin"], "metformin",
     [("RxNorm", "6809", "ingredient"), ("DrugBank", "DB00331", ""), ("ATC", "A10BA02", ""),
      ("MeSH", "D008687", "")]),
]


def normalize() -> Figure:
    fig = Figure(
        760,
        360,
        alt="Two examples of normalisation. The phrases heart attack, MI, myocardial infarction and acute MI all "
        "map to one concept, myocardial infarction, which has SNOMED CT 22298006, MeSH D009203, UMLS C0027051 and, "
        "for the acute case only, ICD-10 I21. The phrases Glucophage, metformin HCl, metformin 500 mg and "
        "metformin map to the ingredient metformin: RxNorm 6809, DrugBank DB00331, ATC A10BA02, MeSH D008687",
    )
    fig.text(14, 28, "Many ways to write one thing", kind="title")
    fig.text(14, 50, "mentions found in text, the concept they name, and its codes in four vocabularies",
             kind="note")
    for row, (forms, concept, codes) in enumerate(NORMAL):
        top = 80 + row * 140
        role = "dec" if row == 0 else "att"
        for k, form in enumerate(forms):
            fig.chip(14, top + k * 30, form, role="neutral", h=24, size=12)
            fig.arrow((14 + Figure.chip_width(form, 12) + 4, top + k * 30 + 12), (236, top + 56), width=1.0)
        fig.box(240, top + 36, 180, 40, role=role, r=8)
        fig.text(330, top + 61, concept, kind="body", role=role, anchor="middle")
        for k, (vocab, code, note) in enumerate(codes):
            y = top + k * 30
            fig.arrow((424, top + 56), (480, y + 12), width=1.0)
            fig.text(486, y + 17, vocab, kind="label")
            fig.text(576, y + 17, code, kind="mono_dec" if role == "dec" else "mono_att")
            if note:
                fig.text(650, y + 17, note, kind="small")
    return fig


# ---------------------------------------------------------------- the graph

def chart_graph(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    h = load("hetionet_t2d")
    disease = h["disease"]
    groups = {"CtD": [], "DpS": [], "DrD": []}
    for a, kind, b in h["edges"]:
        other = a if b == disease else b
        groups[kind].append(h["nodes"][other]["name"])
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    arcs = {"CtD": (98, 218, c["att"], "treats"), "DpS": (-42, 82, c["dec"], "presents"),
            "DrD": (-128, -62, c["enc"], "resembles")}
    for kind, (a0, a1, colour, verb) in arcs.items():
        names = sorted(groups[kind], key=str.lower)
        angles = np.radians(np.linspace(a0, a1, len(names)))
        for name, t in zip(names, angles, strict=True):
            x, y = np.cos(t), np.sin(t)
            ax.plot([0, x * 0.93], [0, y * 0.93], color=colour, linewidth=0.7, alpha=0.5, zorder=1)
            ax.plot([x], [y], "o", color=colour, markersize=7, markeredgecolor=c["card"], zorder=3)
            label = name if kind == "DrD" else name.lower()
            ax.text(x * 1.06, y * 1.06, label, fontsize=8, color=c["ink"], rotation=np.degrees(t) if
                    abs(np.degrees(t)) < 90 else np.degrees(t) + 180, rotation_mode="anchor",
                    ha="left" if abs(np.degrees(t)) < 90 else "right", va="center")
        mid = np.radians((a0 + a1) / 2)
        ax.text(np.cos(mid) * 0.55, np.sin(mid) * 0.55, f"{verb}\n{len(names)}", ha="center", va="center",
                fontsize=9, color=colour, fontweight="semibold")
    ax.plot([0], [0], "o", color=c["ink"], markersize=16, zorder=4)
    ax.text(0, -0.12, "type 2 diabetes", ha="center", va="top", fontsize=10, color=c["ink"], fontweight="semibold",
            zorder=5)
    ax.set_xlim(-1.75, 1.85)
    ax.set_ylim(-1.95, 1.58)
    ax.set_aspect("equal")
    ax.axis("off")
    _caption(fig, "Type 2 diabetes in Hetionet, a 2017 graph of 29 public resources",
             "violet: compounds that treat it; rose: symptoms that co-occur with it in MEDLINE; teal: diseases "
             "it resembles", c, gap=0.04)
    return fig


# ---------------------------------------------------------------- evidence

LEVELS = [  # publication type, label, Oxford CEBM 2011 level for questions of treatment, colour role
    ("Meta-Analysis", "meta-analyses", "1", "enc"),
    ("Randomized Controlled Trial", "randomized trials", "2", "enc"),
    ("Clinical Trial", "clinical trials", "2-3", "att"),
    ("Observational Study", "observational studies", "3", "att"),
    ("Comparative Study", "comparative studies", "3", "att"),
    ("Case Reports", "case reports", "4", "data"),
    ("Review", "reviews", "-", "data"),
]


def chart_evidence(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    data = load("pubmed_counts")
    ev = data["evidence"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.33, right=0.93, top=0.78, bottom=0.15)
    ys = np.arange(len(LEVELS))[::-1]
    for y, (pt, _label, level, role) in zip(ys, LEVELS, strict=True):
        n = ev[pt]
        ax.barh(y, n, color=c[role], height=0.62)
        ax.text(n + 25, y, f"{n:,}", va="center", fontsize=9, color=c["ink"])
        ax.text(-60, y, f"level {level}" if level != "-" else "not graded", va="center", ha="right", fontsize=8.5,
                color=c["muted"])
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _, _ in LEVELS], fontsize=9.5)
    ax.tick_params(axis="y", pad=62)
    ax.set_xlim(0, max(ev[pt] for pt, *_ in LEVELS) * 1.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c["grid"])
    ax.set_xlabel("PubMed articles")
    _caption(fig, "The evidence on one question: metformin in type 2 diabetes, up to 2018",
             f"{ev['any type']:,} articles; an article can carry several types; levels for questions of treatment, "
             f"Oxford CEBM 2011", c)
    return fig


# ---------------------------------------------------------------- the pipeline

STEPS = [
    ("sources", "PubMed and more", "data"),
    ("text", "sentences, tokens", "enc"),
    ("entities", "BiLSTM-CRF tagger", "enc"),
    ("concepts", "UMLS, SNOMED CT", "att"),
    ("relations", "who treats what", "att"),
]
STEPS_2 = [
    ("graph", "Neo4j, Cypher", "att"),
    ("evidence", "study type, quality", "enc"),
    ("ranking", "per question", "enc"),
    ("API", "Django, REST", "data"),
    ("the screen", "for a doctor", "data"),
]


def pipeline() -> Figure:
    fig = Figure(
        760,
        268,
        alt="The pipeline in two rows. Sources (PubMed, trial registries, drug databases) are split into sentences "
        "and tokens; a BiLSTM-CRF tagger finds entities; they are mapped to concepts in UMLS, SNOMED CT and RxNorm; "
        "relations between them are extracted. The result goes into a Neo4j graph queried with Cypher; each study "
        "is graded by its type and quality, answers are ranked per question and served through a Django REST API "
        "to the doctor's screen",
    )
    fig.text(14, 28, "From articles to an answer on a screen", kind="title")
    fig.text(14, 50, "top row: reading the literature; bottom row: storing it and answering questions", kind="note")
    w, gap, h = 132, 18, 52
    for row, steps in enumerate((STEPS, STEPS_2)):
        top = 86 + row * 110
        for k, (first, second, role) in enumerate(steps):
            x = 14 + k * (w + gap)
            fig.box(x, top, w, h, role=role, r=8)
            fig.text(x + w / 2, top + 22, first, kind="body", role=role, anchor="middle")
            fig.text(x + w / 2, top + 40, second, kind="small", anchor="middle")
            if k < len(steps) - 1:
                fig.arrow((x + w + 3, top + h / 2), (x + w + gap - 3, top + h / 2), width=1.3)
    fig.arrow((14 + 4 * (w + gap) + w / 2, 86 + h + 3), (14 + w / 2 + 30, 196 - 4), width=1.3)
    return fig


# ---------------------------------------------------------------- the screen (a reconstruction)

LIGHT_UI = {"bg": "#EEF1F5", "panel": "#FFFFFF", "ink": "#1D2430", "muted": "#5B6575", "faint": "#8A93A3",
            "line": "#DDE2E9", "accent": "#2F6FDB", "accent_tint": "#E6EEFC", "good": "#0F8C7E",
            "good_tint": "#E1F3F0", "warn": "#C27A0E", "warn_tint": "#FCF0DA", "rose": "#D23F57",
            "rose_tint": "#FBE6EA", "side": "#1F2A3A", "side_ink": "#C9D3E3", "side_active": "#2F6FDB"}
DARK_UI = {"bg": "#161618", "panel": "#1E1F23", "ink": "#E7EAF0", "muted": "#9AA3B2", "faint": "#6E7380",
           "line": "#2E3036", "accent": "#FF3636", "accent_tint": "#3A1A1C", "good": "#3CC2B0",
           "good_tint": "#12302C", "warn": "#F0B04A", "warn_tint": "#362A14", "rose": "#F2667C",
           "rose_tint": "#391D24", "side": "#121214", "side_ink": "#9DA3AE", "side_active": "#FF3636"}
CLASSES = {"metformin": "biguanide", "sitagliptin": "DPP-4 inhibitor", "linagliptin": "DPP-4 inhibitor",
           "pioglitazone": "thiazolidinedione", "glimepiride": "sulfonylurea", "gliclazide": "sulfonylurea",
           "acarbose": "alpha-glucosidase inhibitor"}
SANS, MONO = cv.SANS, cv.MONO


def _t(x, y, s, size, colour, weight=400, anchor="start", family=SANS):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{colour}" text-anchor="{anchor}">{escape(s)}</text>')


def _r(x, y, w, h, fill, stroke="none", rx=6, width=1):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>')


def _pill(x, y, s, fill, colour, size=8.5, family=SANS):
    w = len(s) * size * (0.6 if family == MONO else 0.55) + 14
    return _r(x, y, w, 17, fill, rx=8.5) + _t(x + w / 2, y + 12, s, size, colour, 500, "middle", family), w


def screen(ui=LIGHT_UI, width=760, height=480):
    """The assistant's main screen, drawn again for this page with public data of 2018."""
    a = load("assistant")
    h = load("hetionet_t2d")
    disease = h["disease"]
    symptoms = sorted({h["nodes"][x]["name"] for e in h["edges"] if e[1] == "DpS" for x in (e[0], e[2])
                       if x != disease})
    similar = sorted({h["nodes"][x]["name"] for e in h["edges"] if e[1] == "DrD" for x in (e[0], e[2])
                      if x != disease})
    ev = load("pubmed_counts")["evidence"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
         f'height="{height}">', _r(0, 0, width, height, ui["bg"], rx=0)]
    # sidebar
    o.append(_r(0, 0, 138, height, ui["side"], rx=0))
    o.append(f'<circle cx="24" cy="26" r="9" fill="{ui["side_active"]}"/>')
    o.append(_t(24, 30, "+", 13, "#FFFFFF", 700, "middle"))
    o.append(_t(40, 30, "Evidence", 12.5, "#FFFFFF", 600))
    for k, item in enumerate(["Overview", "Treatments", "Symptoms", "Studies", "Graph", "Sources"]):
        y = 74 + k * 30
        if k == 1:
            o.append(_r(8, y - 15, 122, 24, ui["side_active"], rx=6))
        o.append(_t(22, y + 1, item, 10.5, "#FFFFFF" if k == 1 else ui["side_ink"], 500 if k == 1 else 400))
    o.append(_t(16, height - 44, "Sources updated nightly:", 8, ui["side_ink"]))
    o.append(_t(16, height - 30, "PubMed, ClinicalTrials.gov,", 8, ui["side_ink"]))
    o.append(_t(16, height - 18, "DrugBank, SNOMED CT", 8, ui["side_ink"]))
    # top bar with the search
    o.append(_r(138, 0, width - 138, 46, ui["panel"], rx=0))
    o.append(f'<line x1="138" y1="46" x2="{width}" y2="46" stroke="{ui["line"]}"/>')
    o.append(_r(156, 11, 330, 25, ui["bg"], ui["line"], rx=12))
    o.append(f'<circle cx="172" cy="23.5" r="5" fill="none" stroke="{ui["faint"]}" stroke-width="1.5"/>'
             f'<line x1="175.5" y1="27" x2="179" y2="30.5" stroke="{ui["faint"]}" stroke-width="1.5"/>')
    o.append(_t(186, 27.5, "type 2 diabetes mellitus", 10.5, ui["ink"]))
    o.append(_t(width - 54, 27.5, "clinician view", 9.5, ui["muted"], anchor="end"))
    o.append(f'<circle cx="{width - 32}" cy="23" r="11" fill="{ui["accent_tint"]}"/>')
    o.append(_t(width - 32, 26.5, "MD", 8, ui["accent"], 600, "middle"))
    # the disease
    x0, y0 = 152, 58
    main_w = 410
    o.append(_r(x0, y0, main_w, 70, ui["panel"], ui["line"], rx=8))
    o.append(_t(x0 + 14, y0 + 24, "Type 2 diabetes mellitus", 15, ui["ink"], 600))
    px = x0 + 14
    for code in ("ICD-10 E11", "SNOMED CT 44054006", "MeSH D003924"):
        pill, w = _pill(px, y0 + 34, code, ui["accent_tint"], ui["accent"], 8, MONO)
        o.append(pill)
        px += w + 6
    o.append(_t(x0 + 14, y0 + 63, f"{ev['any type']:,} studies on metformin alone, {ev['Meta-Analysis']} of them "
             f"meta-analyses", 9, ui["muted"]))
    # treatments table
    ty = y0 + 82
    rows = sorted(a["drugs"].items(), key=lambda kv: -(kv[1]["meta"] * 5 + kv[1]["trials"]))
    row_h = 27
    o.append(_r(x0, ty, main_w, 56 + row_h * len(rows), ui["panel"], ui["line"], rx=8))
    o.append(_t(x0 + 14, ty + 21, "Treatments, ranked by evidence", 11.5, ui["ink"], 600))
    head_y = ty + 40
    cols = [(x0 + 14, "drug"), (x0 + 108, "class"), (x0 + 236, "randomized trials"), (x0 + 346, "meta-an.")]
    for cx, label in cols:
        o.append(_t(cx, head_y, label.upper(), 7, ui["faint"], 600, family=MONO))
    top_trials = max(d["trials"] for _, d in rows)
    for k, (drug, d) in enumerate(rows):
        y = head_y + 10 + k * row_h
        if k:
            o.append(f'<line x1="{x0 + 10}" y1="{y - 6}" x2="{x0 + main_w - 10}" y2="{y - 6}" '
                     f'stroke="{ui["line"]}"/>')
        o.append(_t(x0 + 14, y + 12, drug, 10.5, ui["ink"], 500))
        o.append(_t(x0 + 108, y + 12, CLASSES[drug].replace("alpha-glucosidase", "α-glucosidase"), 8.5,
                    ui["muted"]))
        bar = 60 * d["trials"] / top_trials
        o.append(_r(x0 + 236, y + 4, 60, 8, ui["bg"], rx=4) + _r(x0 + 236, y + 4, max(bar, 3), 8, ui["good"], rx=4))
        o.append(_t(x0 + 302, y + 12, f"{d['trials']:,}", 8.5, ui["muted"], family=MONO))
        pill, _ = _pill(x0 + 346, y, f"{d['meta']}", ui["good_tint"], ui["good"], 8, MONO)
        o.append(pill)
    # the levels of evidence the ranking uses
    ly = ty + 56 + row_h * len(rows) + 12
    o.append(_r(x0, ly, main_w, height - ly - 14, ui["panel"], ui["line"], rx=8))
    o.append(_t(x0 + 14, ly + 21, "Levels of evidence, Oxford CEBM 2011 (treatment benefits)", 10.5, ui["ink"], 600))
    px = x0 + 14
    for level, what in (("1", "reviews of RCTs"), ("2", "RCTs"), ("3", "cohorts"), ("4", "case series"),
                        ("5", "mechanism")):
        pill, w = _pill(px, ly + 32, level, ui["accent_tint"], ui["accent"], 8, MONO)
        o.append(pill)
        o.append(_t(px + w + 4, ly + 44, what, 8.5, ui["muted"]))
        px += w + 14 + len(what) * 8.5 * 0.52
    # the right column
    rx0, rw = x0 + main_w + 12, width - (x0 + main_w + 12) - 12
    side = []
    side.append(_t(rx0 + 12, y0 + 21, "Presents with", 11, ui["ink"], 600))
    px, py = rx0 + 12, y0 + 32
    for s in [x for x in symptoms if len(x) < 18][:7]:
        pill, w = _pill(px, py, s.lower(), ui["rose_tint"], ui["rose"], 8)
        if px + w > rx0 + rw - 8:
            px, py = rx0 + 12, py + 22
            pill, w = _pill(px, py, s.lower(), ui["rose_tint"], ui["rose"], 8)
        side.append(pill)
        px += w + 5
    side.append(_t(rx0 + 12, py + 40, "Resembles", 11, ui["ink"], 600))
    px, py = rx0 + 12, py + 50
    for s in [x for x in similar if len(x) < 20][:5]:
        pill, w = _pill(px, py, s, ui["good_tint"], ui["good"], 8)
        if px + w > rx0 + rw - 8:
            px, py = rx0 + 12, py + 22
            pill, w = _pill(px, py, s, ui["good_tint"], ui["good"], 8)
        side.append(pill)
        px += w + 5
    o.append(_r(rx0, y0, rw, py + 28 - y0, ui["panel"], ui["line"], rx=8))
    o.extend(side)
    ny = py + 40
    o.append(_r(rx0, ny, rw, height - ny - 14, ui["panel"], ui["line"], rx=8))
    o.append(_t(rx0 + 12, ny + 21, "Newest studies", 11, ui["ink"], 600))
    for k, paper in enumerate(a["papers"][:2]):
        y = ny + 40 + k * 74
        kind = "RCT" if "Randomized Controlled Trial" in paper["types"] else "meta-analysis"
        pill, _ = _pill(rx0 + 12, y - 4, kind, ui["accent_tint"], ui["accent"], 7.5, MONO)
        o.append(pill)
        o.append(_t(rx0 + 52, y + 8, paper["journal"], 8, ui["faint"]))
        words, line, lines = paper["title"].rstrip(".").split(), "", []
        for word in words:
            if len(line) + len(word) + 1 > 36:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        for j, text in enumerate(lines[:3]):
            o.append(_t(rx0 + 12, y + 26 + j * 12, text + ("…" if j == 2 and len(lines) > 3 else ""), 8.8,
                        ui["ink"]))
    o.append("</svg>")
    return "".join(o)


def interface():
    return screen(LIGHT_UI)


# ---------------------------------------------------------------- cover and card

@cache
def _screen_uri(width=760, height=480):
    """The screen in the site's dark colours, as a PNG to embed in the cover."""
    import base64
    import tempfile

    from tools.render import png

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "screen.png"
        png(screen(DARK_UI), out, width, height)
        return "data:image/png;base64," + base64.b64encode(out.read_bytes()).decode()


def cover():
    w = cv.MOTIF_X1 - cv.MOTIF_X0 + 40
    h = w * 480 / 760
    x, y = cv.MOTIF_X0 - 10, 150
    motif = (f'<clipPath id="screen"><rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="10"/></clipPath>'
             f'<image href="{_screen_uri()}" x="{x}" y="{y}" width="{w}" height="{h:.1f}" '
             f'clip-path="url(#screen)"/>'
             f'<rect x="{x}" y="{y}" width="{w}" height="{h:.1f}" rx="10" fill="none" stroke="{cv.LINE}" '
             f'stroke-width="1.5"/>')
    labels = cv.label(x, y - 16, "THE ASSISTANT'S MAIN SCREEN")
    chips, _, right = cv.chip_row(["NER", "UMLS", "Neo4j"], x, y + h + 26, size=15)
    cv.fits(right)
    return cv.frame("portfolio / nlp", ["Medical", "Assistant"], ["The literature, read", "for a doctor"],
                    motif + labels + chips, footer="Entities, a graph, and the strength of evidence")


def card(width=640, height=440):
    w = width - 60
    h = w * 480 / 760
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern>'
            f'<clipPath id="s"><rect x="30" y="{(height - h) / 2:.1f}" width="{w}" height="{h:.1f}" rx="10"/>'
            f'</clipPath></defs>'
            f'<rect width="{width}" height="{height}" fill="{cv.BG}"/>'
            f'<rect width="{width}" height="{height}" fill="url(#dots)"/>'
            f'<image href="{_screen_uri()}" x="30" y="{(height - h) / 2:.1f}" width="{w}" height="{h:.1f}" '
            f'clip-path="url(#s)"/></svg>')


COVER = cover
IMAGES = {"card": (card, 640, 440), "interface": (interface, 760, 480)}

CHARTS = {
    "growth": chart_growth,
    "graph": chart_graph,
    "evidence": chart_evidence,
}

FIGURES = {
    "ner": ner,
    "normalize": normalize,
    "pipeline": pipeline,
}


def chart_alts():
    data = load("pubmed_counts")
    per = data["per_year"]
    ev = data["evidence"]
    h = load("hetionet_t2d")
    return {
        "growth": f"Two panels. Left: bars of PubMed articles per year of publication from 1990 to 2018, growing "
        f"from {per['all']['1990'] / 1e6:.2f} million to {per['all']['2018'] / 1e6:.2f} million. Right: lines of "
        f"randomized trials, from {per['randomized controlled trials']['1990']:,} to "
        f"{per['randomized controlled trials']['2018']:,} a year, and meta-analyses, from "
        f"{per['meta-analyses']['1990']:,} to {per['meta-analyses']['2018']:,}",
        "graph": f"A radial graph around type 2 diabetes from Hetionet: "
        f"{sum(1 for e in h['edges'] if e[1] == 'CtD')} compounds that treat it, including metformin, sitagliptin, "
        f"pioglitazone and glimepiride; {sum(1 for e in h['edges'] if e[1] == 'DpS')} symptoms that co-occur with "
        f"it in MEDLINE, such as obesity, albuminuria and gastroparesis; and "
        f"{sum(1 for e in h['edges'] if e[1] == 'DrD')} diseases it resembles, such as type 1 diabetes, metabolic "
        "syndrome and polycystic ovary syndrome",
        "evidence": "Horizontal bars of PubMed articles on metformin in type 2 diabetes up to 2018, by publication "
        "type, with the Oxford levels of evidence: " + ", ".join(f"{label} {ev[pt]:,}" for pt, label, _, _ in
                                                                    LEVELS),
    }
