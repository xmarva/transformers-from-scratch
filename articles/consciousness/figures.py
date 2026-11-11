"""Figures, charts and the cover of the post on consciousness for physicalists on blog.korolevalab.com.

The survey chart is from the 2020 PhilPapers Survey (Bourget & Chalmers, 2023), the patients chart from Bodien et al.
(2024); brains are fsaverage6 cortex with Destrieux's gyri (tools/figures/brain.py).

    uv run --group articles python tools/render.py --article consciousness --out <site>/assets/posts/<slug>
"""

import numpy as np

from tools.figures import brain
from tools.figures import cover as cv
from tools.figures.lib import Figure


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _greys(theme):
    return ("#E9ECF1", "#BCC2CC") if theme == "light" else ("#4A5160", "#2E343E")


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


# ---------------------------------------------------------------- easy problems and the hard one

def problems() -> Figure:
    fig = Figure(
        760, 372,
        alt="The easy problems of consciousness are to explain functions: discriminating and reacting to stimuli, "
        "integrating information, reporting mental states, focusing attention, deliberate control of behaviour, and "
        "the difference between waking and sleep. Each is explained by finding a mechanism that performs it. The "
        "hard problem is why the performance of these functions is accompanied by experience. Below, Levine's "
        "explanatory gap: heat equals molecular motion explains why heat melts ice, while pain equals C-fibre firing "
        "explains wincing but not why pain feels the way it does",
    )
    fig.text(14, 28, "The easy problems and the hard problem of consciousness", kind="title")
    fig.text(14, 50, "Chalmers (1995); Levine (1983)", kind="note")
    fig.text(14, 84, "the easy problems", kind="subtitle")
    easy = ["discriminate and react", "integrate information", "report a mental state", "focus attention",
            "control behaviour", "wake and sleep"]
    for k, name in enumerate(easy):
        x, y = 14 + (k % 2) * 186, 96 + (k // 2) * 40
        fig.box(x, y, 178, 32, role="enc", label=name)
    fig.text(14, 232, "solved by a mechanism that does the job", kind="note")
    fig.text(420, 84, "the hard problem", kind="subtitle")
    fig.box(420, 96, 326, 112, role="dec", label=None, strong=False)
    fig.text(436, 126, "\"Why is the performance of", kind="body")
    fig.text(436, 148, "these functions accompanied", kind="body")
    fig.text(436, 170, "by experience?\"", kind="body")
    fig.text(436, 194, "a mechanism answers a different question", kind="small")
    fig.text(14, 274, "the explanatory gap", kind="subtitle")
    fig.box(14, 286, 250, 30, role="neutral", label="heat = molecular motion")
    fig.arrow((265, 301), (289, 301))
    fig.text(296, 306, "explains why it melts ice and warms your hand", kind="body")
    fig.box(14, 326, 250, 30, role="neutral", label="pain = C-fibre firing")
    fig.arrow((265, 341), (289, 341))
    fig.text(296, 346, "explains the wince; not why it feels like this", kind="body", role="dec")
    return fig


# ---------------------------------------------------------------- the PhilPapers survey

SURVEY = [  # question, [(answer, share accepting or leaning)], N
    ("Mind", [("physicalism", 51.93), ("non-physicalism", 32.08)], 1733),
    ("A hard problem?", [("yes", 62.42), ("no", 29.76)], 998),
    ("Zombies", [("conceivable, not possible", 36.52), ("possible", 24.41), ("inconceivable", 16.40)], 1610),
    ("Consciousness", [("functionalism", 33.04), ("dualism", 21.96), ("identity theory", 13.33),
                       ("panpsychism", 7.55), ("eliminativism", 4.51)], 1020),
]


def chart_survey(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [2, 2, 3, 5]})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.72, bottom=0.26, wspace=0.35)
    colors = [c["att"], c["enc"], c["data"], c["dec"], c["line"]]
    for ax, (question, answers, n) in zip(axes, SURVEY, strict=True):
        x = np.arange(len(answers))
        values = [a[1] for a in answers]
        ax.bar(x, values, color=colors[: len(answers)], width=0.7)
        for k, v in enumerate(values):
            ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=c["muted"])
        ax.set_xticks(x, [a[0] for a in answers], rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 72)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{question}\nN = {n:,}", fontsize=9, loc="left")
    _caption(fig, "What philosophers think about consciousness in the 2020 PhilPapers Survey",
             "accept or lean towards; several answers allowed, so a question needn't add up to 100%", c)
    return fig


# ---------------------------------------------------------------- Mary's room

def mary() -> Figure:
    fig = Figure(
        760, 360,
        alt="Mary's room. Mary has lived all her life in a black-and-white room and knows every physical fact about "
        "colour vision. She walks out and sees a red tomato. Does she learn something? Jackson in 1982 said yes, a "
        "new fact, so physicalism is false. The physicalist replies: she gains new abilities to imagine and "
        "recognise red (Lewis, Nemirow); she gains acquaintance (Conee); she learns an old fact under a new concept "
        "(Loar, Papineau). Jackson himself gave up the argument by 1995",
    )
    fig.text(14, 28, "Mary's room asks whether knowing every physical fact is knowing everything", kind="title")
    fig.text(14, 50, "Jackson (1982, 1986)", kind="note")
    fig.box(14, 78, 250, 132, role="neutral", label=None)
    fig.text(30, 104, "a black-and-white room", kind="subtitle")
    fig.text(30, 128, "Mary knows all of physics,", kind="note")
    fig.text(30, 148, "every neuron and wavelength", kind="note")
    fig.text(30, 168, "of colour vision", kind="note")
    fig.text(30, 192, "she has never seen red", kind="label")
    fig.arrow((265, 144), (309, 144), role="dec")
    fig.text(287, 134, "out", kind="small", anchor="middle")
    fig.box(310, 112, 120, 64, role="dec", label="a red tomato", strong=True)
    fig.arrow((431, 144), (459, 144))
    fig.text(466, 120, "does she learn", kind="subtitle")
    fig.text(466, 142, "something new?", kind="subtitle")
    fig.text(466, 166, "\"It seems just obvious that", kind="note")
    fig.text(466, 186, "she will learn something\"", kind="note")
    replies = [
        ("a new fact", "so physicalism is false (Jackson, 1982)", "dec"),
        ("new abilities", "to imagine and recognise red (Lewis; Nemirow)", "enc"),
        ("acquaintance", "with what she already knew about (Conee)", "enc"),
        ("a new concept", "of an old physical fact (Loar; Papineau)", "enc"),
    ]
    for k, (what, why, role) in enumerate(replies):
        y = 232 + k * 26
        fig.box(14, y, 150, 22, role=role, label=what)
        fig.text(176, y + 16, why, kind="body")
    fig.text(560, 270, "by 1995 Jackson", kind="note", role="dec")
    fig.text(560, 290, "himself had given", kind="note", role="dec")
    fig.text(560, 310, "up the argument", kind="note", role="dec")
    return fig


# ---------------------------------------------------------------- zombies

def zombie() -> Figure:
    rows = [("atoms", "the same", "the same"), ("behaviour", "the same", "the same"),
            ("says \"I'm conscious\"", "yes", "yes"), ("experience", "yes", "none")]
    fig = Figure(
        760, 346,
        alt="The zombie argument. You and your zombie twin have the same atoms and the same behaviour, and both say "
        "they are conscious; you have experience and the twin has none. Premise one: such a twin is conceivable. "
        "Premise two: what is conceivable is possible. Conclusion: the physical facts do not fix experience, so "
        "physicalism is false. In 2020, 16% of philosophers said zombies are inconceivable, 37% conceivable but not "
        "possible, 24% possible",
    )
    fig.text(14, 28, "A philosophical zombie is an exact physical copy of you without experience", kind="title")
    fig.text(14, 50, "not a robot that acts conscious but an exact physical duplicate (Kirk; Chalmers, 1996)",
             kind="note")
    fig.text(214, 84, "you", kind="subtitle", anchor="middle")
    fig.text(330, 84, "your zombie twin", kind="subtitle", anchor="middle")
    for k, (what, you, twin) in enumerate(rows):
        y = 96 + k * 36
        fig.text(14, y + 20, what, kind="label")
        last = k == len(rows) - 1
        fig.box(164, y, 100, 28, role="enc", label=you)
        fig.box(280, y, 100, 28, role="dec" if last else "enc", label=twin, strong=last)
    fig.text(420, 96, "the argument", kind="subtitle")
    steps = ["1  such a twin is conceivable", "2  what is conceivable is possible",
             "3  so physics doesn't fix experience"]
    for k, s in enumerate(steps):
        fig.text(420, 122 + k * 22, s, kind="body")
    fig.text(420, 210, "where physicalists get off (2020 survey)", kind="subtitle")
    answers = [("inconceivable", "A", 16), ("conceivable, not possible", "B", 37), ("possible", "", 24)]
    for k, (a, kind, share) in enumerate(answers):
        y = 224 + k * 34
        fig.rect(420, y, share * 2.0, 22, role="att" if k < 2 else "dec")
        fig.text(420 + share * 2.0 + 8, y + 16, f"{share}% {a}" + (f" ({kind})" if kind else ""), kind="small")
    fig.text(14, 274, "Kripke: to make heat, God makes molecules move;", kind="note")
    fig.text(14, 296, "to make pain, he \"must do something in addition\"", kind="note")
    fig.text(14, 318, "to the C-fibre stimulation", kind="note")
    return fig


# ---------------------------------------------------------------- fading qualia

def fading() -> Figure:
    fig = Figure(
        760, 330,
        alt="Fading qualia. Your neurons are replaced one at a time by silicon chips that do the same local job: "
        "none, a quarter, half, three quarters, all. Behaviour and speech stay exactly the same at every step. If "
        "the all-silicon brain had no experience, a middle person would see pale pink while sincerely describing "
        "bright red, systematically wrong about their own experience. Chalmers concludes experience does not fade: a "
        "fine-grained functional copy is conscious",
    )
    fig.text(14, 28, "Replacing neurons with chips one at a time", kind="title")
    fig.text(14, 50, "Chalmers (1995), a dualist's argument that a silicon copy of a brain would be conscious",
             kind="note")
    stages = [0, 0.25, 0.5, 0.75, 1.0]
    for s, share in enumerate(stages):
        x = 14 + s * 148
        fig.text(x, 84, f"{int(share * 100)}% silicon", kind="subtitle")
        for i in range(4):
            for j in range(4):
                silicon = i * 4 + j < share * 16
                fig.rect(x + j * 26, 96 + i * 26, 22, 22, role="data" if silicon else "dec", tint=not silicon)
    fig.text(14, 222, "behaviour, speech, reports of seeing bright red:", kind="label")
    fig.box(14, 232, 732, 26, role="enc", label="exactly the same at every step")
    fig.text(14, 286, "if experience faded, the person in the middle would see pale pink while describing bright red,",
             kind="note")
    fig.text(14, 308, "\"systematically wrong about everything that he is experiencing\": so experience doesn't fade",
             kind="note", role="att")
    return fig


# ---------------------------------------------------------------- the positions

def positions() -> Figure:
    fig = Figure(
        760, 476,
        alt="A decision tree of positions on consciousness. First question: is there a hard problem beyond "
        "explaining the functions? No: type-A physicalism, illusionism and eliminativism (Dennett, Frankish). Yes: "
        "second question, is experience physical? Yes: type-B physicalism, an identity known only a posteriori "
        "(Loar, Papineau); biological naturalism (Searle); mysterianism, true but beyond us (McGinn). No: third "
        "question, then what is it? A separate thing that acts on the body: substance dualism (Descartes). "
        "Fundamental properties linked to the physical by laws: property dualism (Chalmers). Caused by the brain but "
        "causing nothing: epiphenomenalism. The inner nature of matter itself: Russellian monism and panpsychism "
        "(Russell, Strawson, Goff). Everything is mental: idealism (Kastrup). Functionalism, which systems are "
        "conscious, cuts across the tree",
    )
    fig.text(14, 28, "Positions on consciousness as answers to three questions", kind="title")
    fig.text(14, 50, "Chalmers's types A, B, D, E and F in brackets", kind="note")
    fig.box(14, 70, 360, 34, role="att", label="Is there a hard problem beyond the functions?", strong=True)
    fig.line((60, 105), (60, 127))
    fig.text(68, 121, "no", kind="small")
    fig.box(14, 128, 262, 46, role="enc", label=None)
    fig.text(26, 147, "illusionism, eliminativism (A)", kind="body")
    fig.text(26, 166, "Dennett, Frankish", kind="small")
    fig.line((330, 105), (330, 141))
    fig.text(338, 125, "yes", kind="small")
    fig.box(300, 142, 300, 34, role="att", label="Is experience physical?", strong=True)
    fig.line((320, 177), (320, 194), (40, 194), (40, 207))
    fig.text(180, 189, "yes", kind="small")
    physical = [("type-B physicalism (B)", "an identity known a posteriori: Loar, Papineau"),
                ("biological naturalism", "caused by the brain's biology: Searle"),
                ("mysterianism", "physical, but beyond our concepts: McGinn")]
    for k, (name, who) in enumerate(physical):
        y = 208 + k * 46
        fig.box(14, y, 340, 40, role="enc", label=None)
        fig.text(26, y + 17, name, kind="body")
        fig.text(26, y + 33, who, kind="small")
    fig.line((580, 177), (580, 194), (620, 194), (620, 203))
    fig.text(588, 189, "no", kind="small")
    fig.text(392, 220, "then what is it?", kind="subtitle")
    rest = [("substance dualism (D)", "a thing of its own: Descartes"),
            ("property dualism", "fundamental, bound by laws: Chalmers"),
            ("epiphenomenalism (E)", "caused, but causes nothing"),
            ("Russellian monism (F)", "matter's inner nature: Strawson, Goff"),
            ("idealism", "everything is mental: Kastrup")]
    for k, (name, who) in enumerate(rest):
        y = 230 + k * 36
        fig.box(392, y, 354, 32, role="dec", label=None)
        fig.text(404, y + 14, name, kind="body")
        fig.text(404, y + 28, who, kind="small")
    fig.text(14, 430, "functionalism answers a different question, which systems are conscious, and cuts across",
             kind="note")
    fig.text(14, 452, "the tree: type-A, type-B and Chalmers's property dualism can all be functionalist", kind="note")
    return fig


# ---------------------------------------------------------------- the theories on the brain

THEORIES = [  # name, where, Destrieux labels, palette role
    ("global workspace", "prefrontal and parietal hubs",
     ("G_front_middle", "G_front_sup", "S_front_middle", "S_front_sup", "G_front_inf-Triangul",
      "G_front_inf-Opercular", "G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Supramar"), "att"),
    ("integrated information", "the posterior hot zone",
     ("Pole_occipital", "G_occipital_middle", "G_occipital_sup", "G_and_S_occipital_inf", "S_occipital_ant",
      "G_oc-temp_lat-fusifor", "G_temporal_inf", "G_temporal_middle", "G_pariet_inf-Angular", "G_parietal_sup",
      "S_oc_middle_and_Lunatus", "S_temporal_sup"), "enc"),
    ("higher-order", "dorsolateral prefrontal",
     ("G_front_middle", "S_front_middle", "S_front_inf"), "dec"),
    ("recurrent processing", "visual cortex, by itself",
     ("Pole_occipital", "G_occipital_middle", "G_occipital_sup", "G_and_S_occipital_inf", "S_oc_middle_and_Lunatus",
      "S_occipital_ant"), "data"),
]


def chart_theories(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    labels = brain.destrieux("left")
    fig = plt.figure(figsize=(7.6, 5.4))
    for k, (name, where, regions, role) in enumerate(THEORIES):
        col, row = k % 2, k // 2
        ax = fig.add_axes((0.04 + col * 0.5, 0.47 - row * 0.43, 0.42, 0.3))
        brain.draw(ax, "left", "lateral", labels, {r: c[role] for r in regions}, base=_greys(theme))
        ax.text(0.5, 1.07, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
                fontweight="semibold", color=c["ink"])
        ax.text(0.5, -0.04, where, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=c["muted"])
    _caption(fig, "Where four theories look for consciousness",
             "left hemisphere from the side; regions approximate, drawn with Destrieux's gyri", c)
    return fig


# ---------------------------------------------------------------- the adversarial test

def cogitate() -> Figure:
    rows = [
        ("where content lives", "posterior cortex is enough", "prefrontal cortex needed",
         "content decodable at the back; prefrontal added nothing", "enc", "data"),
        ("how it lasts", "sustained posterior activity", "ignition at onset and offset",
         "duration tracked, sparsely; 0 of 655 prefrontal sites at offset", "enc", "dec"),
        ("connectivity", "sustained posterior synchrony", "brief long-range synchrony",
         "no sustained synchrony; the preregistered metric backed neither", "dec", "data"),
    ]
    fig = Figure(
        760, 84 + len(rows) * 76 + 36,
        alt="The adversarial test of integrated information theory and global neuronal workspace theory. First "
        "prediction, where conscious content lives: IIT predicted posterior cortex is enough, GNWT that prefrontal "
        "cortex is needed; content was decodable at the back and adding prefrontal cortex added nothing, a pass for "
        "IIT and mixed for GNWT. Second, how it lasts: IIT predicted sustained posterior activity, GNWT ignition at "
        "onset and offset; duration was tracked, sparsely, a pass for IIT, while none of 655 prefrontal electrodes "
        "showed offset ignition, a failure for GNWT. Third, connectivity: IIT predicted sustained posterior "
        "synchrony, which was absent, a failure; GNWT brief long-range synchrony, not supported by the preregistered "
        "metric but consistent in an exploratory analysis, mixed",
    )
    fig.text(14, 28, "How the Cogitate experiment tested IIT against global workspace theory", kind="title")
    fig.text(14, 50, "Cogitate Consortium (2025), 256 people, fMRI, MEG and intracranial EEG, preregistered",
             kind="note")
    fig.text(180, 80, "IIT predicted", kind="label", role="enc")
    fig.text(400, 80, "GNWT predicted", kind="label", role="att")
    for k, (what, iit, gnwt, found, iit_role, gnwt_role) in enumerate(rows):
        y = 92 + k * 76
        fig.text(14, y + 20, what, kind="subtitle")
        fig.box(180, y, 204, 30, role=iit_role, label=iit)
        fig.box(400, y, 204, 30, role=gnwt_role, label=gnwt)
        fig.text(180, y + 52, "found: " + found, kind="note")
    verdict = {"enc": "passed", "data": "mixed", "dec": "failed"}
    fig.text(14, 84 + len(rows) * 76 + 22, "box colour: teal passed, amber mixed, rose failed; the consortium gave no "
             "verdict", kind="small")
    del verdict
    return fig


# ---------------------------------------------------------------- patients

def chart_patients(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    # Bodien et al. (2024): 60 of 241 without an observable response to commands performed the tasks; 43 of 112
    # with an observable response did
    groups = [("no response at the bedside", 241, 60), ("responding at the bedside", 112, 43)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [241, 150]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.72, bottom=0.08, wspace=0.12)
    for ax, (name, total, did) in zip(axes, groups, strict=True):
        cols = 20
        for i in range(total):
            r, col = divmod(i, cols)
            color = c["att"] if i < did else c["grid"]
            ax.add_patch(plt.Rectangle((col, -r), 0.82, 0.82, color=color, linewidth=0))
        rows = (total + cols - 1) // cols
        ax.set_xlim(-0.3, cols)
        ax.set_ylim(-rows - 0.3, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}: {did} of {total}", fontsize=9.5, loc="left")
    _caption(fig, "Who performed mental tasks in the scanner, by bedside behaviour",
             "in violet those who imagined playing tennis, moving a hand and so on, on fMRI or EEG; "
             "Bodien et al. (2024)", c)
    return fig


def chart_alts():
    return {
        "survey": "Four bar charts from the 2020 PhilPapers Survey. Mind: physicalism 52%, non-physicalism 32%. Is "
        "there a hard problem: yes 62%, no 30%. Zombies: conceivable but not metaphysically possible 37%, "
        "metaphysically possible 24%, inconceivable 16%. Consciousness: functionalism 33%, dualism 22%, identity "
        "theory 13%, panpsychism 8%, eliminativism 5%",
        "theories": "Four views of the left hemisphere. Global workspace theory: prefrontal and parietal hubs. "
        "Integrated information theory: the posterior hot zone of occipital, temporal and parietal cortex. "
        "Higher-order theory: dorsolateral prefrontal cortex. Recurrent processing theory: visual cortex by itself",
        "patients": "Two grids of squares. Of 241 patients with no response to commands at the bedside, 60 performed "
        "the mental imagery tasks on fMRI or EEG. Of 112 patients who did follow commands at the bedside, only 43 "
        "showed it on the scanner tests",
    }


def cover():
    out = [cv.label(cv.MOTIF_X0, 150, "THE EASY PROBLEMS")]
    svg, spans, right = cv.chip_row(["discriminate", "report", "attend"], cv.MOTIF_X0, 172, size=17)
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 262, "THE HARD PROBLEM"))
    svg, spans, right = cv.chip_row(["why is any of it felt?"], cv.MOTIF_X0, 284, size=17, kinds={0: "hot"})
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 374, "THOUGHT EXPERIMENTS"))
    svg, spans, right = cv.chip_row(["Mary", "zombies", "a silicon brain"], cv.MOTIF_X0, 396, size=17,
                                    kinds={0: "warm", 1: "warm", 2: "warm"})
    cv.fits(right)
    out.append(svg)
    return cv.frame("blog / philosophy", ["Theories of", "consciousness"],
                    ["The hard problem for", "a physicalist reader"], "".join(out),
                    footer="Philosophy of mind")


CHARTS = {"survey": chart_survey, "theories": chart_theories, "patients": chart_patients}
FIGURES = {"problems": problems, "mary": mary, "zombie": zombie, "fading": fading, "positions": positions,
           "cogitate": cogitate}
COVER = cover
