"""Static figures and covers of lesson 7.

Facts about real things are computed when the figures are rendered: the hero counts the lines of the package in
perevod/, the wheel figure opens the wheel of sacreBLEU downloaded from PyPI, the test figure counts the tests in
perevod/tests/.
"""

import ast
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import Figure

PACKAGE = Path(__file__).resolve().parents[2] / "perevod"
SACREBLEU = "2.5.1"

# what the lines of the package do; colours: teal the model, amber the interfaces people use, violet the checks,
# rose packaging and release
PARTS = [
    ("the model", "enc", ["src/perevod/model.py"]),
    ("decoding", "enc", ["src/perevod/decoding.py"]),
    ("API, config, CLI", "data", ["src/perevod/translator.py", "src/perevod/config.py", "src/perevod/cli.py",
                                  "src/perevod/__init__.py"]),
    ("tests", "att", ["tests/*.py"]),
    ("packaging and CI", "dec", ["pyproject.toml", "README.md", "../.github/workflows/perevod.yml"]),
]


def _lines(pattern):
    total = 0
    for path in sorted(PACKAGE.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += 1
    return total


def _test_count():
    count = 0
    for path in sorted((PACKAGE / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                count += 1
    return count


# ---------------------------------------------------------------- hero: what the package is made of


def hero() -> Figure:
    parts = []
    for name, role, patterns in PARTS:
        lines = 0
        for pattern in patterns:
            lines += _lines(pattern)
        parts.append((name, role, lines))
    total = sum(lines for _, _, lines in parts)
    model = parts[0][2] + parts[1][2]
    fig = Figure(
        760,
        300,
        alt=f"The package perevod by lines of code: the model and its decoding are {model} of {total} non-empty "
        f"lines, {model / total:.0%}; the rest is the interface people use, the tests, and the packaging and CI "
        "around them",
    )
    fig.text(14, 28, "What a translator package is made of", kind="title")
    fig.text(14, 50, f"non-empty lines of perevod, {total} in all; the model and its decoding are "
             f"{model / total:.0%} of them", kind="note")
    x0, width, y = 14, 732, 92
    x = x0
    for k, (name, role, lines) in enumerate(parts):
        w = width * lines / total
        fig.rect(x, y, w - 3, 64, role=role, tint=role != "enc", r=4)
        label_y = 186 + (k % 2) * 44
        middle = x + w / 2
        fig.line((middle, y + 66), (middle, label_y - 16), role="muted", width=1)
        anchor, at = "middle", middle
        if k == len(parts) - 1:  # the last label ends at the right edge of the bar
            anchor, at = "end", x0 + width
        fig.text(at, label_y, name, kind="body", anchor=anchor)
        fig.text(at, label_y + 18, f"{lines} lines", kind="label", anchor=anchor)
        x += w
    fig.text(14, 286, "Sculley et al. (2015) drew this for whole ML systems: the model is the small box "
             "in the middle.", kind="note")
    return fig


# ---------------------------------------------------------------- a wheel, opened


def _sacrebleu_wheel():
    with urllib.request.urlopen(f"https://pypi.org/pypi/sacrebleu/{SACREBLEU}/json", timeout=60) as response:
        release = json.load(response)
    for item in release["urls"]:
        if item["filename"].endswith(".whl"):
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                return item["filename"], zipfile.ZipFile(io.BytesIO(response.read()))
    raise RuntimeError("no wheel for sacrebleu")


def wheel() -> Figure:
    filename, archive = _sacrebleu_wheel()
    names = archive.namelist()
    code = [n for n in names if n.startswith("sacrebleu/") and n.endswith(".py")]
    dist_info = next(n for n in names if ".dist-info/" in n).split("/")[0]
    info = sorted(n.split("/", 1)[1] for n in names if n.startswith(dist_info + "/"))
    entry_points = archive.read(f"{dist_info}/entry_points.txt").decode().splitlines()
    script = next(line.strip() for line in entry_points if "=" in line)
    fig = Figure(
        760,
        360,
        alt=f"Inside the wheel {filename}: a zip archive with the package's {len(code)} Python files and a "
        f"dist-info folder ({', '.join(info)}). pip copies the package into site-packages, installs the "
        "dependencies listed in METADATA, and writes a sacrebleu command from entry_points.txt",
    )
    fig.text(14, 28, "A wheel is a zip file with code and a few files about the code", kind="title")
    fig.text(14, 50, filename, kind="label")
    fig.box(14, 70, 404, 270, role="neutral", label="")
    fig.text(30, 96, "sacrebleu/", kind="mono", role="enc")
    fig.text(30, 118, f"{len(code)} Python files", kind="label")
    fig.text(30, 156, dist_info + "/", kind="mono", role="dec")
    meaning = {"METADATA": "name, version, Requires-Dist", "entry_points.txt": "commands to create",
               "RECORD": "every file and its hash", "WHEEL": "the tags: py3-none-any",
               "LICENSE.txt": "the license", "top_level.txt": "top-level packages"}
    y = 182
    for name in info:
        short = name.split("/")[-1]
        fig.text(46, y, short, kind="mono")
        fig.text(196, y, meaning.get(short, ""), kind="small")
        y += 24
    fig.arrow((422, 120), (470, 120), role="enc")
    fig.box(474, 98, 272, 44, role="enc", label="site-packages/sacrebleu/")
    fig.arrow((422, 200), (470, 200), role="data")
    fig.box(474, 178, 272, 44, role="data", label="pip installs Requires-Dist")
    fig.arrow((422, 262), (470, 262), role="dec")
    fig.box(474, 240, 272, 44, role="dec", label="bin/sacrebleu, a new command")
    fig.text(474, 312, script, kind="small")
    return fig


# ---------------------------------------------------------------- the plan: notebook cells to package files


def plan() -> Figure:
    rows = [
        ("the model of lessons 2 and 5", "model.py", "enc"),
        ("beam search of lessons 3 and 4", "decoding.py", "enc"),
        ("config dicts in cells", "config.py", "data"),
        ("download, load_state_dict, translate", "translator.py", "data"),
        ("“run this cell to translate”", "cli.py: the perevod command", "data"),
        ("asserts scattered in lessons", "tests/", "att"),
        ("Runtime → Run all", ".github/workflows/perevod.yml", "dec"),
        ("three lines of README", "a model card", "dec"),
    ]
    fig = Figure(
        760,
        74 + 38 * len(rows),
        alt="Where the parts of the course's notebooks go: the model and the decoding become model.py and "
        "decoding.py; config dictionaries become config.py; downloading and loading become translator.py; running "
        "a cell becomes the perevod command; asserts become tests; running the notebook becomes a CI workflow; and "
        "a three-line README becomes a model card",
    )
    fig.text(14, 28, "Each part of the notebooks gets a place in the package", kind="title")
    fig.text(14, 56, "in the notebooks", kind="label")
    fig.text(420, 56, "in the package perevod", kind="label")
    for k, (old, new, role) in enumerate(rows):
        y = 70 + 38 * k
        fig.box(14, y, 330, 30, role="plain", label=old)
        fig.arrow((348, y + 15), (414, y + 15), role="muted")
        fig.box(418, y, 328, 30, role=role, label=new)
    return fig


# ---------------------------------------------------------------- which copy does `import perevod` find?


def import_path() -> Figure:
    fig = Figure(
        760,
        330,
        alt="How Python finds `import perevod` when pytest runs in the project folder. The current folder comes "
        "first in the search path. In a flat layout the folder holds perevod/ itself, so the tests import the "
        "source and never the installed copy. In the src layout the source sits in src/, not on the path, so the "
        "tests import what pip installed, the same thing users get",
    )
    fig.text(14, 28, "Which perevod do the tests import?", kind="title")
    fig.text(14, 50, "Python tries the folders of sys.path in order; the current folder comes first",
             kind="note")
    for k, (name, folder, found, role) in enumerate([
        ("flat layout", "project/perevod/", "the source folder", "dec"),
        ("src layout", "project/src/perevod/", "the installed copy", "enc"),
    ]):
        x = 14 + k * 376
        fig.text(x, 86, name, kind="subtitle")
        fig.box(x, 100, 356, 40, role="plain", label="import perevod")
        fig.text(x + 8, 172, "1. current folder", kind="mono")
        fig.text(x + 8, 192, folder, kind="label")
        fig.text(x + 8, 232, "2. site-packages", kind="mono")
        fig.text(x + 8, 252, "perevod, as pip installed it", kind="label")
        hit_y = 160 if k == 0 else 220
        fig.box(x + 220, hit_y, 136, 40, role=role, label="found here")
        fig.text(x, 300, f"the tests check {found}", kind="body", role=role)
    return fig


# ---------------------------------------------------------------- four kinds of tests


def test_kinds() -> Figure:
    kinds = [
        ("config and glue", "no model", "milliseconds", "a typo in a key, an empty input, a lost line", "data"),
        ("invariants", "a tiny random model", "milliseconds", "a leaking mask, padding that matters",
         "att"),
        ("golden translations", "the published model", "seconds", "any change in what the model writes", "enc"),
        ("a quality floor", "the model and a test set", "a minute", "a model that got worse on average", "dec"),
    ]
    fig = Figure(
        760,
        96 + 64 * len(kinds),
        alt="Four kinds of tests for a model package. Tests of the config and glue need no model and run in "
        "milliseconds. Invariant tests need only a tiny random model. Golden tests run the published model on a "
        "dozen sentences and catch any change in its output. A quality floor scores the model on a test set and "
        "catches a model that got worse on average",
    )
    fig.text(14, 28, f"Four kinds of tests, {_test_count()} tests in perevod", kind="title")
    fig.text(214, 62, "needs", kind="label")
    fig.text(410, 62, "catches", kind="label")
    for k, (name, needs, speed, catches, role) in enumerate(kinds):
        y = 76 + 64 * k
        fig.box(14, y, 186, 50, role=role, label=name)
        fig.text(214, y + 22, needs, kind="body")
        fig.text(214, y + 42, speed, kind="label")
        fig.text(410, y + 30, catches, kind="note")
    fig.text(14, 96 + 64 * len(kinds) - 14, "Down the list: slower, more expensive, closer to what users see.",
             kind="note")
    return fig


# ---------------------------------------------------------------- the CI workflow


def ci() -> Figure:
    fig = Figure(
        760,
        360,
        alt="The CI workflow of perevod. A push to main, a pull request or the Monday schedule starts it. The test "
        "job runs on Python 3.10 and 3.13: install, ruff, the fast tests. The wheel job builds the wheel, installs "
        "it and runs the command outside the source folder. The quality job, on main, on Mondays and on pull "
        "requests labelled model, downloads the published model and runs the slow tests",
    )
    fig.text(14, 28, "Every push runs the same checks on a clean machine", kind="title")
    for k, trigger in enumerate(["push to main", "pull request", "Monday 06:00"]):
        fig.box(14, 90 + 76 * k, 150, 40, role="plain", label=trigger)
        fig.line((164, 110 + 76 * k), (196, 110 + 76 * k), role="muted")
    fig.line((196, 110), (196, 290), role="muted")  # any trigger starts every job
    jobs = [
        ("test × Python 3.10, 3.13", ["install", "ruff", "pytest (fast)"], "data"),
        ("wheel", ["build", "install", "run in /tmp"], "dec"),
        ("quality: main, Mondays", ["cache", "download model", "pytest (slow)"], "enc"),
    ]
    for k, (name, steps, role) in enumerate(jobs):
        y = 62 + 94 * k
        fig.arrow((196, y + 40), (228, y + 40), role="muted")
        fig.box(232, y, 514, 80, role="neutral", label="")
        fig.text(246, y + 24, name, kind="block", role=role)
        x = 246
        for step in steps:
            w = Figure.chip_width(step) + 10
            fig.box(x, y + 38, w, 30, role=role, label=step)
            x += w + 14
    fig.text(14, 346, "A failing check marks the pull request red; nothing reaches main without the fast tests.",
             kind="note")
    return fig


# ---------------------------------------------------------------- a model card, taken apart


def card() -> Figure:
    fig = Figure(
        760,
        380,
        alt="A model card on the Hugging Face Hub has two parts. A YAML header that machines read: language, "
        "license, datasets, tags, and model-index with the metrics, which the Hub shows as an evaluation table. "
        "And a text for people: what the model is, what it is for and not for, the data, the evaluation and the "
        "limitations, following Mitchell et al. (2019)",
    )
    fig.text(14, 28, "A model card: a header for machines, a text for people", kind="title")
    fig.box(14, 56, 340, 140, role="neutral", label="")
    fig.text(28, 80, "---", kind="mono")
    for k, key in enumerate(["language, license, datasets", "tags: translation", "model-index: chrF, BLEU"]):
        fig.text(28, 104 + 22 * k, key, kind="mono", role="dec" if k == 2 else None)
    fig.text(28, 182, "---", kind="mono")
    fig.box(14, 210, 340, 156, role="plain", label="")
    sections = ["Model details", "Intended use and out of scope", "Training data", "Evaluation",
                "Limitations and biases"]
    for k, section in enumerate(sections):
        fig.text(28, 236 + 26 * k, f"## {section}", kind="mono", role="enc" if k >= 3 else None)
    fig.arrow((358, 128), (426, 128), role="dec")
    fig.box(430, 100, 316, 56, role="dec", label="the Hub: search filters, a table")
    fig.arrow((358, 290), (426, 290), role="enc")
    fig.box(430, 262, 316, 56, role="enc", label="measured in this notebook")
    fig.text(430, 342, "chrF in and out of domain, by length, gender", kind="small")
    return fig


# ---------------------------------------------------------------- two version numbers


def versions() -> Figure:
    rows = [
        ("a bug fix in the code", "patch  0.1.0 → 0.1.1", "same", "plain"),
        ("a new option, e.g. --max-tokens", "minor  0.1.1 → 0.2.0", "same", "data"),
        ("a new default model", "major  0.2.0 → 1.0.0", "a new pinned commit", "dec"),
        ("a corrected model card", "no release", "a commit nobody pins", "plain"),
    ]
    fig = Figure(
        760,
        96 + 50 * len(rows),
        alt="Two version numbers move separately. The package version follows semantic versioning: a bug fix is a "
        "patch, a new option a minor release, and a new default model, which changes the translations users get, a "
        "major one. The model is pinned by the commit of its repository, so a corrected model card on the Hub "
        "changes no one's translations",
    )
    fig.text(14, 28, "Two versions: the package's number and the model's commit", kind="title")
    fig.text(14, 62, "the change", kind="label")
    fig.text(330, 62, "perevod", kind="label")
    fig.text(520, 62, "model revision", kind="label")
    for k, (change, package, model, role) in enumerate(rows):
        y = 76 + 50 * k
        fig.box(14, y, 300, 36, role=role, label=change)
        fig.text(330, y + 23, package, kind="mono")
        fig.text(520, y + 23, model, kind="label")
    return fig


# ---------------------------------------------------------------- cover: a terminal session


def _cover(lang):
    english = lang == "en"
    tests = _test_count()
    lines = [
        ("$ pip install ./perevod", cv.INK),
        ("$ perevod --version", cv.INK),
        ("perevod 0.1.0", cv.MUTED),
        ("$ pytest -q", cv.INK),
        (f"{tests} passed", cv.MUTED),
        ("$ git push", cv.INK),
        ("CI: all checks have passed" if english else "CI: все проверки пройдены", cv.RED),
    ]
    out = [cv.label(cv.MOTIF_X0, 128, "THE TRANSLATOR AS A PACKAGE" if english else "ПЕРЕВОДЧИК КАК ПАКЕТ")]
    out.append(f'<rect x="{cv.MOTIF_X0}" y="150" width="{cv.MOTIF_X1 - cv.MOTIF_X0}" height="360" rx="14" '
               f'fill="{cv.CHIP_FILL}" stroke="{cv.LINE}" stroke-width="1.4"/>')
    y = 200
    for text, colour in lines:
        out.append(cv.text(cv.MOTIF_X0 + 28, y, text, 24, colour, cv.MONO))
        y += 44
    title = ["Notebook code", "as a package"] if english else ["Код из ноутбука", "как пакет"]
    hook = ["Which bug would", "your tests miss?"] if english else ["Какой баг пропустят", "ваши тесты?"]
    return cv.frame("07_package.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "wheel": wheel,
    "plan": plan,
    "import-path": import_path,
    "test-kinds": test_kinds,
    "ci": ci,
    "card": card,
    "versions": versions,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
