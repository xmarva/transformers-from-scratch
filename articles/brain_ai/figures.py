"""Figures, charts and the cover of the post on brains and AI models on blog.korolevalab.com.

Brains are fsaverage6 cortex coloured by published atlases (tools/figures/brain.py). The dopamine chart is computed
from a temporal-difference model, the one Schultz, Dayan & Montague (1997) fitted to recordings of dopamine neurons.

    uv run --group articles python tools/render.py --article brain_ai --out <site>/assets/posts/<slug>
"""

import textwrap
from pathlib import Path

import numpy as np

from tools.figures import brain
from tools.figures import cover as cv
from tools.figures.lib import Figure


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


DATA = Path(__file__).parent / "data"


def _greys(theme):
    return ("#E9ECF1", "#BCC2CC") if theme == "light" else ("#4A5160", "#2E343E")


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


# ---------------------------------------------------------------- dopamine as a prediction error

STEPS, CUE, REWARD = 30, 6, 21  # time steps in a trial; the cue (a light, a tone) comes on, the juice arrives


def td_trials(trials=300, alpha=0.15, gamma=0.98, omit_last=True):
    """Temporal-difference learning with one feature per time step since the cue (a "complete serial compound").

    Returns the prediction error delta(t) of every trial; the last trial has its reward omitted."""
    weights = np.zeros(STEPS)
    errors = []
    for trial in range(trials):
        reward = np.zeros(STEPS)
        if not (omit_last and trial == trials - 1):
            reward[REWARD] = 1.0
        value = np.array([weights[t - CUE] if t >= CUE else 0.0 for t in range(STEPS + 1)])
        delta = np.zeros(STEPS)
        for t in range(STEPS):
            delta[t] = reward[t] + gamma * value[t + 1] - value[t]
            if t >= CUE:
                weights[t - CUE] += alpha * delta[t]
        errors.append(delta)
    return np.array(errors)


def chart_dopamine(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    errors = td_trials()
    panels = [
        (errors[0], "before learning", "the juice is a surprise"),
        (errors[-2], "after learning", "the cue predicts the juice"),
        (errors[-1], "after learning, juice withheld", "a dip when it fails to come"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    fig.subplots_adjust(top=0.66, bottom=0.2, left=0.08, right=0.99, wspace=0.12)
    t = np.arange(STEPS)
    for ax, (delta, title, note) in zip(axes, panels, strict=True):
        colors = [c["enc"] if d >= 0 else c["dec"] for d in delta]
        ax.bar(t, delta, width=0.8, color=colors)
        ax.axhline(0, color=c["line"], linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, -0.24, note, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=c["muted"])
        ax.set_xticks([CUE, REWARD], ["cue", "juice"])
        ax.set_ylim(-1.15, 1.15)
        ax.grid(False)
    axes[0].set_ylabel("prediction error δ")
    _caption(fig, "Dopamine responses predicted by a temporal-difference model",
             "computed here with TD learning, one feature per moment after the cue; compare Schultz et al. (1997)",
             c)
    return fig


# ---------------------------------------------------------------- the language network

def lana(hemi):
    """LanA (Lipkin et al., 2022): for each fsaverage vertex, the share of 804 people for whom it is among the 10% most
    language-selective (sentences > nonwords); fsaverage6, our mesh, is the first 40,962 vertices of fsaverage."""
    import nibabel as nib

    name = {"left": "LH", "right": "RH"}[hemi]
    values = np.asarray(nib.load(DATA / f"{name}_LanA_n804.nii.gz").get_fdata()).reshape(-1)
    return values[: len(brain.mesh(hemi)[0])]


def chart_language(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = _style(theme)
    fig = plt.figure(figsize=(7.6, 3.6))
    panels = [("left", "lateral", (0.0, 0.17, 0.5, 0.64)), ("right", "lateral", (0.5, 0.17, 0.5, 0.64))]
    for hemi, view, box in panels:
        ax = fig.add_axes(box)
        empty = np.array([""] * len(brain.mesh(hemi)[0]))
        brain.draw(ax, hemi, view, empty, {}, base=_greys(theme), overlays=[(lana(hemi), c["att"], 0.1, 0.7)])
        ax.text(0.5, -0.02, f"{hemi} hemisphere", transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                color=c["muted"])
    bar = fig.add_axes((0.36, 0.055, 0.28, 0.035))
    ramp = LinearSegmentedColormap.from_list("ramp", [_greys(theme)[0], c["att"]])
    bar.imshow(np.linspace(0, 1, 200)[None, :], aspect="auto", cmap=ramp, extent=(0.1, 0.7, 0, 1))
    bar.set_yticks([])
    bar.set_xticks([0.1, 0.4, 0.7], ["10%", "40%", "70% of people"], fontsize=8.5, color=c["muted"])
    bar.tick_params(length=0)
    for side in bar.spines.values():
        side.set_visible(False)
    _caption(fig, "Where the language network lies in the cortex",
             "how often each spot is among a person's most language-selective, LanA atlas of 804 people", c)
    return fig


def _mix(color, toward, amount):
    from matplotlib.colors import to_hex, to_rgb

    a, b = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return to_hex(a + (b - a) * amount)


# ---------------------------------------------------------------- Tuckute et al. 2024: driving and suppressing

def chart_drive(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    # Tuckute et al. (2024), event-related test in new participants: response relative to baseline sentences,
    # on z-scored BOLD responses of the language network
    rows = [
        ("chosen to drive it", 85.7, c["att"],
         ["“People on Insta Be Like, “Gross!””", "“Turin loves me not, nor will.”"]),
        ("ordinary sentences", 0.0, c["line"], ["the 1,000 sentences the model was trained on"]),
        ("chosen to silence it", -97.5, c["dec"],
         ["“We were sitting on the couch.”", "“Inside was a tiny silver sculpture.”"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.04)
    for k, (name, value, color, examples) in enumerate(rows):
        ax.barh(k, value, color=color, height=0.45)
        label = f"{value:+.1f}%" if value else "baseline"
        side = 1 if value >= 0 else -1
        ax.text(value + side * 4, k, label, va="center", ha="left" if side > 0 else "right", fontsize=10,
                color=c["ink"], fontweight="semibold")
        ax.text(-330, k - 0.22, name, va="center", fontsize=10, color=c["ink"])
        for j, example in enumerate(examples):
            ax.text(-330, k + 0.06 + j * 0.24, example, va="center", fontsize=8.5, color=c["muted"],
                    style="italic")
    ax.axvline(0, color=c["line"], linewidth=1)
    ax.set_xlim(-335, 130)
    ax.set_ylim(2.75, -0.6)
    ax.axis("off")
    _caption(fig, "Sentences a GPT-based model chose to steer the language network",
             "response in new people relative to ordinary sentences, standardised; Tuckute et al. (2024)", c)
    return fig


# ---------------------------------------------------------------- the encoding model

def encoding() -> Figure:
    fig = Figure(
        760, 318,
        alt="How a brain is compared with a model. The same sentence goes to a language model and to a person in an "
        "fMRI scanner. The model's internal activity, a vector of numbers, is mapped to each voxel's response by a "
        "linear regression fitted on training sentences. On new sentences the predicted response is compared with "
        "the measured one, and the correlation is divided by the noise ceiling: how well the brain predicts itself "
        "across repeated measurements or across people",
    )
    fig.text(14, 28, "How an encoding model predicts the brain from a model", kind="title")
    fig.text(14, 50, "fitted on some sentences, tested on sentences it has never seen", kind="note")
    fig.box(14, 132, 150, 50, role="data", label="“The tower fell”")
    fig.arrow((165, 150), (199, 106))
    fig.arrow((165, 164), (199, 226))
    fig.box(200, 84, 160, 44, role="dec", label="language model")
    fig.arrow((361, 106), (389, 106))
    fig.box(390, 84, 120, 44, role="dec", label="activity x")
    fig.arrow((511, 106), (539, 106))
    fig.box(540, 84, 206, 44, role="att", label="linear map: ŷ = W x", strong=True)
    fig.box(200, 204, 160, 44, role="enc", label="person in a scanner")
    fig.arrow((361, 226), (539, 226))
    fig.box(540, 204, 206, 44, role="enc", label="measured response y")
    fig.arrow((643, 129), (643, 203), role="att")
    fig.text(634, 172, "compare: r(ŷ, y)", kind="label", role="att", anchor="end")
    fig.text(14, 280, "score = r(ŷ, y) / noise ceiling, where the ceiling is how well the brain predicts itself",
             kind="note")
    fig.text(14, 302, "the map W is fitted on training sentences; the score is computed on held-out ones",
             kind="note")
    return fig


# ---------------------------------------------------------------- the ventral stream

STREAM = [  # stage, Destrieux labels, how far along the stream (0 early, 1 late)
    ("early: edges", ("Pole_occipital", "S_calcarine", "G_cuneus", "G_oc-temp_med-Lingual"), 0.15),
    ("middle: shapes, textures", ("G_occipital_middle", "G_occipital_sup", "G_and_S_occipital_inf",
                                  "S_oc_middle_and_Lunatus", "S_occipital_ant", "S_oc-temp_lat"), 0.55),
    ("late: objects, faces", ("G_oc-temp_lat-fusifor", "G_temporal_inf", "S_temporal_inf"), 1.0),
]


def chart_ventral(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = _style(theme)
    labels = brain.destrieux("left")
    light = _mix(c["enc"], c["card"], 0.75)
    colors = {}
    shades = {}
    for stage, names, along in STREAM:
        shade = _mix(light, c["enc"], along)
        shades[stage] = shade
        for name in names:
            colors[name] = shade
    fig = plt.figure(figsize=(7.6, 3.3))
    names = {"lateral": "from the side", "medial": "from the middle", "ventral": "from below"}
    for k, view in enumerate(["lateral", "medial", "ventral"]):
        ax = fig.add_axes((0.005 + 0.333 * k, 0.22, 0.32, 0.56))
        brain.draw(ax, "left", view, labels, colors, base=_greys(theme))
        ax.text(0.5, -0.03, names[view], transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                color=c["muted"])
    handles = [Patch(facecolor=shades[stage], label=stage) for stage, _, _ in STREAM]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9.5,
               handlelength=1.2, columnspacing=1.6)
    _caption(fig, "The ventral stream runs forward along the underside of the brain",
             "left hemisphere, approximate stages with Destrieux's gyri; "
             "a vision network's layers line up the same way",
             c)
    return fig


# ---------------------------------------------------------------- vision: the payoff ran out

def vision() -> Figure:
    rows = [
        ("Yamins et al., 2014", "monkey IT, 168 sites", "trained only to categorise: 48.5% of explainable variance",
         "enc"),
        ("Schrimpf et al., 2018", "Brain-Score, monkeys", "the link with ImageNet accuracy weakens above ~70%", "data"),
        ("Linsley et al., 2023", "135 networks, monkey IT", "more accurate networks became worse models of IT",
         "dec"),
        ("Gokce & Schrimpf, 2025", "600+ models", "behaviour keeps improving with scale, neural fit saturates",
         "dec"),
        ("Conwell et al., 2024", "224 models, human fMRI", "accuracy vs brain fit: rank correlation −0.006 and 0.17",
         "dec"),
    ]
    fig = Figure(
        760, 84 + len(rows) * 44 + 34,
        alt="Five studies in order. Yamins and colleagues in 2014 found that a network trained only to categorise "
        "objects predicted 48.5% of the explainable variance of monkey inferior temporal cortex. Brain-Score in 2018 "
        "found the link between ImageNet accuracy and brain similarity weakening above about 70% accuracy. Linsley "
        "and colleagues in 2023 found that across 135 networks more accurate ones were worse models of IT. Gokce and "
        "Schrimpf in 2025 found, over more than 600 models, that agreement with behaviour kept improving with scale "
        "while agreement with neurons saturated. Conwell and colleagues in 2024 found ImageNet accuracy unrelated to "
        "how well 224 models predicted human fMRI",
    )
    fig.text(14, 28, "How the brain similarity of vision networks changed with their accuracy", kind="title")
    fig.text(14, 50, "five studies against monkey IT and the human visual cortex", kind="note")
    for k, (who, data, finding, role) in enumerate(rows):
        y = 70 + k * 44
        fig.box(14, y, 186, 34, role=role, label=who, strong=k == 0)
        fig.text(214, y + 15, data, kind="small")
        fig.text(214, y + 30, finding, kind="body")
    fig.text(14, 84 + len(rows) * 44 + 8, "and the same models explain ~31% or ~79.5% of the explainable variance, "
             "depending on the metric", kind="note")
    return fig


# ---------------------------------------------------------------- space: place cells, grid cells, hippocampus

def chart_space(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = _style(theme)
    n = 200
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))  # a 1 m square box
    place = np.exp(-((x - 0.62) ** 2 + (y - 0.38) ** 2) / (2 * 0.09 ** 2))
    spacing = 0.3  # grid spacing in metres
    k = 4 * np.pi / (np.sqrt(3) * spacing)
    grid = np.zeros_like(x)
    for angle in [0, np.pi / 3, 2 * np.pi / 3]:  # three plane waves 60 degrees apart make a triangular lattice
        grid += np.cos(k * (np.cos(angle) * (x - 0.2) + np.sin(angle) * (y - 0.1)))
    grid = np.clip((grid + 1.5) / 4.5, 0, 1) ** 2
    ramp = LinearSegmentedColormap.from_list("ramp", [c["card"], c["att"]])
    fig = plt.figure(figsize=(7.6, 3.3))
    left = fig.add_axes((0.0, 0.1, 0.36, 0.66))
    tissue = ("#2E3542", "#F1F3F6") if theme == "light" else ("#1E232B", "#C9CED7")
    brain.coronal(left, -22, {"Left Hippocampus": c["att"], "Right Hippocampus": c["att"]}, tissue=tissue,
                  background=c["card"])
    left.text(0.5, -0.04, "hippocampus, coronal slice y = −22 mm", transform=left.transAxes, ha="center", va="top",
              fontsize=9, color=c["muted"])
    for k, (values, title) in enumerate([(place, "a place cell"), (grid, "a grid cell")]):
        ax = fig.add_axes((0.42 + k * 0.29, 0.12, 0.25, 0.62))
        ax.imshow(values, origin="lower", cmap=ramp, extent=(0, 1, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(c["line"])
        ax.set_title(title, fontsize=10, pad=4)
        ax.text(0.5, -0.06, "firing across a 1 m box", transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color=c["muted"])
    _caption(fig, "Place cells in the hippocampus and grid cells in the entorhinal cortex",
             "idealised firing maps computed here, one Gaussian field and three waves 60° apart for the lattice", c)
    return fig


# ---------------------------------------------------------------- memory systems and a model's stores

def memory() -> Figure:
    rows = [
        ("working memory", "about 4 chunks, kept active", "context window", "10⁵+ tokens, passive, middle used worst"),
        ("episodic memory", "one-shot, your own experiences", "retrieval (RAG)", "documents, not the model's past"),
        ("semantic memory", "facts, slowly consolidated", "weights", "absorbed slowly in training"),
        ("consolidation", "replay during sleep", "nothing", "a conversation never reaches the weights"),
    ]
    fig = Figure(
        760, 84 + len(rows) * 52 + 18,
        alt="Human memory systems paired with a language model's stores. Working memory, about four chunks kept "
        "active, pairs with the context window, a hundred thousand tokens or more held passively and used worst in "
        "the middle. Episodic memory, one-shot memories of one's own experiences, pairs with retrieval of documents, "
        "which are not the model's own past. Semantic memory, slowly consolidated facts, pairs with the weights, "
        "absorbed slowly in training. Consolidation by replay during sleep pairs with nothing: what happens in a "
        "conversation never reaches the weights",
    )
    fig.text(14, 28, "Human memory and the memory of a language model", kind="title")
    fig.text(14, 50, "for each memory system, the nearest store in a deployed model", kind="note")
    fig.text(14, 76, "the brain", kind="subtitle")
    fig.text(420, 76, "a language model", kind="subtitle")
    for k, (human, note, model, model_note) in enumerate(rows):
        y = 86 + k * 52
        fig.box(14, y, 160, 38, role="enc", label=human)
        fig.text(184, y + 24, note, kind="note")
        role = "dec" if model != "nothing" else "neutral"
        fig.box(420, y, 150, 38, role=role, label=model, strong=model == "nothing")
        for j, part in enumerate(textwrap.wrap(model_note, 26)):
            fig.text(580, y + 16 + j * 15, part, kind="small")
        fig.line((390, y + 19), (412, y + 19), role="muted", dash=True)
    return fig


# ---------------------------------------------------------------- intuitive physics

def chart_physics(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    # Bordes et al. (2025), IntPhys 2, Table 2, overall accuracy (chance 50%)
    scores = [("people (majority of three)", 96.44), ("V-JEPA 2", 57.51), ("Gemini 2.5 Flash", 55.63),
              ("GPT-4o", 53.75), ("V-JEPA + RoPE", 53.75), ("Qwen-VL 2.5", 52.27), ("Cosmos-4B", 49.41)]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.78, bottom=0.14)
    names = [s[0] for s in scores]
    values = [s[1] for s in scores]
    colors = [c["enc"]] + [c["dec"]] * (len(scores) - 1)
    ax.barh(names, values, color=colors, height=0.6)
    for k, v in enumerate(values):
        ax.text(v + 0.8, k, f"{v:.1f}%", va="center", fontsize=9, color=c["muted"])
    ax.axvline(50, color=c["line"], linestyle="--", linewidth=1)
    ax.text(50.5, len(scores) - 0.45, "chance", fontsize=8.5, color=c["muted"])
    ax.set_xlim(40, 103)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("possible or impossible video: accuracy")
    _caption(fig, "People and the best AI models on intuitive physics in complex scenes",
             "IntPhys 2 (Bordes et al., 2025) on permanence, immutability, continuity and solidity", c)
    return fig


# ---------------------------------------------------------------- the module map

MODULES = [  # region, Destrieux labels (or LanA), status, AI counterpart
    ("vision", ("Pole_occipital", "S_calcarine", "G_cuneus", "G_oc-temp_med-Lingual", "G_occipital_middle",
                "G_occipital_sup", "G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "S_occipital_ant",
                "G_oc-temp_lat-fusifor", "G_temporal_inf", "S_oc-temp_lat", "S_temporal_inf"), "strong"),
    ("hearing and speech", ("G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo"), "strong"),
    ("language", "LanA", "strong"),
    ("movement", ("G_precentral", "S_central", "S_precentral-inf-part", "S_precentral-sup-part",
                  "G_and_S_paracentral"), "partial"),
    ("planning, control", ("G_front_middle", "G_front_sup", "S_front_middle", "S_front_sup"), "partial"),
    ("space, memory", ("G_oc-temp_med-Parahip",), "partial"),
    ("body, emotion", ("G_insular_short", "G_Ins_lg_and_S_cent_ins", "S_circular_insula_ant",
                       "S_circular_insula_inf", "S_circular_insula_sup", "G_and_S_cingul-Ant", "G_subcallosal",
                       "G_rectus"), "absent"),
]


def chart_modules(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = _style(theme)
    status_color = {"strong": c["enc"], "partial": c["data"], "absent": c["dec"]}
    fig = plt.figure(figsize=(7.6, 3.7))
    for k, view in enumerate(["lateral", "medial"]):
        labels = brain.destrieux("left")
        colors = {}
        overlays = []
        for _name, regions, status in MODULES:
            if regions == "LanA":
                overlays.append((lana("left"), status_color[status], 0.25, 0.45))
            else:
                for region in regions:
                    colors[region] = status_color[status]
        ax = fig.add_axes((0.01 + 0.5 * k, 0.2, 0.47, 0.6))
        brain.draw(ax, "left", view, labels, colors, base=_greys(theme), overlays=overlays)
        ax.text(0.5, -0.02, f"left hemisphere, {view}", transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=c["muted"])
    handles = [Patch(facecolor=status_color[s], label=label) for s, label in
               [("strong", "a close AI counterpart"), ("partial", "a partial one"), ("absent", "none yet")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9.5,
               handlelength=1.2, columnspacing=1.6)
    _caption(fig, "Which brain systems have an AI counterpart",
             "my reading of the evidence in this post; regions approximate (Destrieux gyri, LanA for language)", c)
    return fig


def modtable() -> Figure:
    rows = [
        ("vision", "vision networks", "predict the ventral stream; gains stopped", "strong"),
        ("hearing, speech", "Whisper and kin", "predict the auditory-to-language hierarchy", "strong"),
        ("language network", "language models", "predict it, mostly word meaning", "strong"),
        ("dopamine teaching signal", "TD learning", "predicted new properties of neurons", "strong"),
        ("hippocampus, space", "path integration, retrieval", "grids only by design; no consolidation", "partial"),
        ("motor cortex", "robot policies, VLAs", "web knowledge yes, new motions no", "partial"),
        ("prefrontal control", "reasoning models", "reasoning without a brain-level match", "partial"),
        ("insula, interoception", "—", "no body to sense", "absent"),
        ("sleep, neuromodulators", "—", "no counterpart", "absent"),
    ]
    role = {"strong": "enc", "partial": "data", "absent": "dec"}
    fig = Figure(
        760, 84 + len(rows) * 34 + 10,
        alt="A table of brain systems, their nearest AI counterparts and the state of the match. Strong: vision and "
        "vision networks, which predict the ventral stream though gains have stopped; hearing and speech and models "
        "like Whisper, which predict the auditory-to-language hierarchy; the language network and language models, "
        "which predict it mostly through word meaning; the dopamine teaching signal and temporal-difference "
        "learning, which predicted new properties of neurons. Partial: the hippocampus and space, where grid-like "
        "units appear only by design and there is no consolidation; motor cortex and robot policies, which gained "
        "web knowledge but no new motions; prefrontal control and reasoning models, without a brain-level match. "
        "Absent: the insula and interoception, with no body to sense, and sleep and neuromodulators, with no "
        "counterpart",
    )
    fig.text(14, 28, "Brain systems and their nearest AI counterparts", kind="title")
    fig.text(14, 50, "strong means activity predicted or signature reproduced, partial some of the job, absent none",
             kind="note")
    for k, (system, ai, note, status) in enumerate(rows):
        y = 66 + k * 34
        fig.box(14, y, 196, 28, role=role[status], label=system)
        fig.text(222, y + 19, ai, kind="label")
        fig.text(450, y + 19, note, kind="note")
    return fig


# ---------------------------------------------------------------- energy

def chart_energy(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    training = [  # MWh
        ("a brain, 20 years at 20 W", 20 * 20 * 8766 / 1e6, c["enc"]),
        ("GPT-3, training", 1287, c["dec"]),
        ("Llama 3.1 405B, GPUs only", 30.84e6 * 700 / 1e6, c["dec"]),
        ("Grok 4, estimate", 310e3, c["dec"]),
    ]
    answers = [  # Wh
        ("a brain, one minute", 20 / 60, c["enc"]),
        ("median Gemini prompt", 0.24, c["dec"]),
        ("typical GPT-4o query", 0.3, c["dec"]),
        ("GPT-4o, 10k-token input", 2.5, c["dec"]),
        ("GPT-4o, 100k-token input", 40, c["dec"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.76, bottom=0.18, wspace=0.95)
    for ax, rows, unit, title in [(axes[0], training, "MWh", "to train"), (axes[1], answers, "Wh", "to answer")]:
        names = [r[0] for r in rows]
        values = [r[1] for r in rows]
        ax.barh(names, values, color=[r[2] for r in rows], height=0.6)
        ax.set_xscale("log")
        for k, v in enumerate(values):
            label = f"{v:,.0f}" if v >= 10 else f"{v:.2g}"
            ax.text(v * 1.3, k, f"{label} {unit}", va="center", fontsize=8.5, color=c["muted"])
        ax.set_xlim(min(values) / 3, max(values) * 40)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
    _caption(fig, "Energy used by the brain and by AI models",
             "Levy & Calvert 2021; Patterson et al. 2021; Llama 3.1 card at 700 W; Epoch AI 2025; Google 2025", c)
    return fig


# ---------------------------------------------------------------- data

def chart_data(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rows = [  # name, low, high, colour
        ("words heard by age 5", 6e6, 6e7, c["enc"]),
        ("words heard by age 13", 1e7, 1e8, c["enc"]),
        ("words by age 20, with reading", 3e7, 4e8, c["enc"]),
        ("BabyLM budget", 1e8, 1e8, c["data"]),
        ("GPT-3, tokens", 5e11, 5e11, c["dec"]),
        ("Llama 3.1, tokens", 1.56e13, 1.56e13, c["dec"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.76, bottom=0.17)
    for k, (_name, low, high, color) in enumerate(rows):
        if high > low:
            ax.plot([low, high], [k, k], color=color, linewidth=9, solid_capstyle="round")
        else:
            ax.plot([low], [k], "o", color=color, markersize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1e6, 1e14)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("words or tokens (log scale)")
    _caption(fig, "Words a child hears and tokens a language model is trained on",
             "Frank 2023 (child estimates); BabyLM; Patterson et al. 2021; Llama 3 (Meta, 2024)", c)
    return fig


# ---------------------------------------------------------------- cover

def cover():
    out = [cv.label(cv.MOTIF_X0, 140, "IN THE BRAIN  ↔  IN A MODEL")]
    pairs = [("language network", "language model"), ("ventral stream", "vision network"),
             ("dopamine burst", "TD error")]
    y = 168
    for left, right in pairs:
        svg, w = cv.chip(cv.MOTIF_X0, y, left, size=16)
        out.append(svg)
        out.append(cv.text(cv.MOTIF_X0 + w + 20, y + 27, "↔", 22, cv.RED, cv.SANS))
        svg, w2 = cv.chip(cv.MOTIF_X0 + w + 52, y, right, size=16, kind="match")
        out.append(svg)
        cv.fits(cv.MOTIF_X0 + w + 52 + w2)
        y += 58
    out.append(cv.label(cv.MOTIF_X0, y + 34, "NO AI COUNTERPART YET"))
    svg, spans, right = cv.chip_row(["sleep", "a body", "lifelong learning"], cv.MOTIF_X0, y + 56, size=16,
                                    kinds={0: "warm", 1: "warm", 2: "warm"})
    cv.fits(right)
    out.append(svg)
    return cv.frame("blog / cognitive science", ["The human brain", "and AI models"],
                    ["Language, vision, memory,", "dopamine and energy"], "".join(out),
                    footer="Cognitive science")


def chart_alts():
    return {
        "drive": "Bars of the language network's response in new people. Sentences a GPT-based encoding model chose "
        "to drive it, such as People on Insta Be Like, Gross, and Turin loves me not, nor will, raised it by 85.7% "
        "over ordinary sentences. Sentences chosen to silence it, such as We were sitting on the couch, lowered it by "
        "97.5%, on a standardised scale",
        "ventral": "The left hemisphere seen from the side, from the middle and from below, with the ventral visual "
        "stream in three shades: early visual cortex at the back of the head and along the calcarine sulcus, middle "
        "stages on the lateral occipital surface, and late stages in the fusiform and inferior temporal gyri on the "
        "underside of the temporal lobe",
        "space": "A coronal slice of the brain with both hippocampi highlighted, and two idealised firing maps over a "
        "one-metre box: a place cell fires in one spot, a grid cell at every vertex of a triangular lattice",
        "physics": "Accuracy at telling possible from impossible videos on the IntPhys 2 benchmark. "
        "People, by majority of three annotators, 96.4%. The best models are near chance at 50%: "
        "V-JEPA 2 57.5%, Gemini 2.5 Flash 55.6%, "
        "GPT-4o 53.8%, V-JEPA with RoPE 53.8%, Qwen-VL 2.5 52.3%, Cosmos-4B 49.4%",
        "modules": "The left hemisphere from the side and from the middle, coloured by how close an AI counterpart is. "
        "Visual cortex, auditory cortex and the language network: a close counterpart. Motor cortex, dorsolateral "
        "prefrontal cortex and the parahippocampal region: a partial one. The insula, the anterior cingulate and the "
        "ventromedial prefrontal cortex, which handle the body's state and emotion: none yet",
        "energy": "Two bar charts on log scales. To train: a brain over 20 years at 20 watts uses 3.5 megawatt-hours, "
        "GPT-3's training 1,287, Llama 3.1 405B about 21,600 for its GPUs alone, Grok 4 an estimated 310,000. To "
        "answer: a brain in one minute uses 0.33 watt-hours, a median Gemini prompt 0.24, a typical GPT-4o query "
        "0.3, a GPT-4o query with 10,000 tokens of input 2.5 and with 100,000 tokens about 40",
        "data": "Words and tokens on a log scale. A child hears between 6 and 60 million words by age 5, up to about "
        "100 million by age 13, and 30 to 400 million words by age 20 including reading. The BabyLM challenge allows "
        "100 million words. GPT-3 was trained on 500 billion tokens and Llama 3.1 on 15.6 trillion",
        "language": "The language network on an inflated cortex, both hemispheres seen from the side. Violet marks "
        "where, across 804 people, a spot was among the most responsive to sentences compared with lists of "
        "nonwords. The left hemisphere has large, strong patches in the inferior and middle frontal gyri and along "
        "the temporal lobe from front to back; the right hemisphere has weaker, smaller versions of the same "
        "patches",
        "dopamine": "Three panels of the prediction error of a temporal-difference model across one trial. Before "
        "learning, the error is a positive spike when the juice arrives. After learning, the spike has moved to the "
        "cue that predicts the juice, and there is nothing at the juice itself. When the juice is withheld after "
        "learning, there is a spike at the cue and a negative dip at the moment the juice was expected. These are "
        "the three patterns Schultz and colleagues recorded in dopamine neurons of monkeys",
    }


CHARTS = {"drive": chart_drive, "language": chart_language, "ventral": chart_ventral, "space": chart_space,
          "dopamine": chart_dopamine, "physics": chart_physics, "modules": chart_modules, "energy": chart_energy,
          "data": chart_data}
FIGURES = {"encoding": encoding, "vision": vision, "memory": memory, "modtable": modtable}
COVER = cover
