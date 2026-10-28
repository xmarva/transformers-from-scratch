"""Static figures and covers of lesson 9.

The numbers in them are real: layer sizes and digests come from the published image's manifests and configs,
saved in image.json (`python lessons/09_docker/figures.py` fetches them again from ghcr.io); the number of
packages from perevod's lock file; the .pyc headers from py_compile run on a small file; the sizes of the ONNX
files from the Space.
"""

import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import Figure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "0.3.1"
SPACE = "xmarva/perevod"
BASE_LAYERS = 4  # python:3.12-slim is four layers: Debian and three steps that add Python


def image():
    return json.loads((HERE / "image.json").read_text(encoding="utf-8"))


def short(digest, n=12):
    return digest.removeprefix("sha256:")[:n]


def megabytes(size):
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1e3:.0f} kB"
    return f"{size / 1e6:.0f} MB"


def steps(platform):
    out = []
    for step in platform["config"]["history"]:
        if not step.get("empty_layer"):
            command = " ".join(step["created_by"].split())
            command = command.split(" /bin/sh -c ")[-1].replace(" # buildkit", "")
            out.append(command)
    return out


def linux_packages():
    lock = (ROOT / "perevod" / "requirements.lock").read_text(encoding="utf-8")
    count = 0
    for line in lock.splitlines():
        if "==" in line and not line.startswith((" ", "#")):
            if "sys_platform == 'darwin'" in line or "sys_platform == 'win32'" in line:
                continue
            count += 1
    return count


# ---------------------------------------------------------------- hero: what runs when perevod translates


def hero() -> Figure:
    data = image()["platforms"]["amd64"]
    sizes = [layer["size"] for layer in data["manifest"]["layers"]]
    base = sum(sizes[:BASE_LAYERS])
    packages = 0
    for size, command in zip(sizes, steps(data), strict=True):
        if "-r requirements.lock" in command:
            packages = size
    model = sizes[-1]  # the last step downloads the model
    python_version = ""
    for item in data["config"]["config"]["Env"]:
        if item.startswith("PYTHON_VERSION="):
            python_version = item.split("=")[1]
    rows = [
        ("the weights", megabytes(model), "the model's commit on the Hub, 4abc0d6", "dec"),
        ("perevod's code", megabytes(sizes[-2]), f"a git tag, perevod-v{VERSION}", "enc"),
        (f"{linux_packages()} Python packages", megabytes(packages), "a lock file: versions and sha256 hashes",
         "data"),
        (f"Python {python_version}, Debian", megabytes(base), "the base image, by its digest", "att"),
        ("the build", "", "a pinned BuildKit and the commit's time", "plain"),
    ]
    fig = Figure(
        760,
        392,
        alt="What runs when perevod translates, as a stack: the weights, perevod's code, the Python packages, "
        "Python and Debian's libraries, the build itself, and the processor. Next to each, what pins it in perevod "
        "0.3.1: the model's commit, a git tag, a lock file with hashes, a base image pinned by digest, a pinned "
        "builder and a fixed time; the processor is the one part no image can pin",
    )
    fig.text(14, 28, "What runs when perevod translates, and what pins each part", kind="title")
    fig.text(14, 50, "sizes are the compressed layers of the image for x86-64", kind="note")
    fig.text(430, 80, "pinned by", kind="label")
    y = 92
    for name, size, pin, role in rows:
        fig.box(14, y, 300, 40, role=role)
        fig.text(30, y + 25, name, kind="body")
        if size:
            fig.text(300, y + 25, size, kind="count", anchor="end")
        fig.arrow((322, y + 20), (420, y + 20), role="muted", width=1.2)
        fig.text(430, y + 25, pin, kind="body")
        y += 50
    fig.line((14, y), (314, y), (314, y + 40), (14, y + 40), (14, y), role="muted", width=1.2, dash=True)
    fig.text(30, y + 25, "the processor", kind="body")
    fig.text(430, y + 25, "nothing: whatever processor runs it", kind="note")
    return fig


# ---------------------------------------------------------------- anatomy of an image


def anatomy() -> Figure:
    data = image()
    amd64 = data["platforms"]["amd64"]
    layers = amd64["manifest"]["layers"]
    commands = steps(amd64)
    fig = Figure(
        760,
        540,
        alt="The structure of an image, top to bottom. A tag points to an image index. The index lists one "
        "manifest per platform by digest. A manifest lists a config and the layers by digest. The config holds the "
        "environment, the command and the hash of each unpacked layer; each layer is a gzipped tar archive of the "
        "files one Dockerfile step changed. Each level names the next by its sha256, so the digest at the top "
        "fixes every byte below it",
    )
    fig.text(14, 28, f"ghcr.io/xmarva/perevod:{VERSION}, by hand", kind="title")
    fig.text(14, 50, "every arrow is a sha256: each level names the next by the hash of its bytes", kind="note")
    fig.box(14, 70, 220, 36, role="plain", label=f"tag {VERSION}")
    fig.arrow((240, 88), (300, 88), role="muted")
    fig.box(306, 70, 440, 36, role="att", label=f"index  sha256:{short(data['index_digest'], 16)}…")
    fig.arrow((420, 108), (420, 128), role="att")
    fig.arrow((640, 108), (640, 128), role="att")
    arm = data["platforms"]["arm64"]["manifest_digest"]
    fig.box(306, 132, 228, 34, role="att", label=f"amd64  {short(amd64['manifest_digest'], 10)}…")
    fig.box(540, 132, 206, 34, role="plain", label=f"arm64  {short(arm, 10)}…")
    fig.arrow((330, 168), (330, 188), (190, 188), (190, 198), role="att")
    fig.arrow((470, 168), (470, 198), role="att")
    config = amd64["manifest"]["config"]
    fig.box(14, 202, 300, 70, role="enc")
    fig.text(28, 226, f"config  {short(config['digest'], 10)}…", kind="mono")
    fig.text(28, 248, "environment, user, command,", kind="note")
    fig.text(28, 264, "the hash of each unpacked layer", kind="note")
    settings = amd64["config"]["config"]
    lines = [f"User  {settings['User']}", f"Cmd   {' '.join(settings['Cmd'][:1])} …"]
    for item in settings["Env"]:
        if item.startswith(("HF_HUB_OFFLINE", "HF_HUB_CACHE", "PYTHON_VERSION")):
            lines.append(f"Env   {item}")
    fig.text(28, 300, "from this image's config:", kind="label")
    for k, line in enumerate(lines):
        fig.text(28, 324 + k * 22, line, kind="small")
    fig.text(330, 216, "layers: gzipped tar archives, one per step", kind="label")
    top = 226
    biggest = max(layer["size"] for layer in layers)
    for k, (layer, command) in enumerate(zip(layers, commands, strict=True)):
        y = top + k * 23
        width = max(3, 110 * layer["size"] / biggest)
        fig.rect(330, y + 4, width, 14, role="data", tint=k < BASE_LAYERS, r=2)
        fig.text(500, y + 15, megabytes(layer["size"]), kind="small", anchor="end")
        fig.text(510, y + 15, command[:33], kind="small")
    fig.text(330, top + len(layers) * 23 + 16, "the four pale ones are python:3.12-slim", kind="note")
    return fig


# ---------------------------------------------------------------- .pyc headers


def _pyc_headers():
    source_text = "GREETING = 'Tom is afraid of the dark.'\n"
    out = {}
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "module.py"
        source.write_text(source_text)
        for mode in ["timestamp", "hash"]:
            headers = []
            for minute in [0, 1]:
                moment = 1793212520 + 60 * minute
                os.utime(source, (moment, moment))
                target = Path(folder) / f"{mode}{minute}.pyc"
                if mode == "hash":
                    os.environ["SOURCE_DATE_EPOCH"] = "1793212520"
                try:
                    py_compile.compile(str(source), cfile=str(target))
                finally:
                    os.environ.pop("SOURCE_DATE_EPOCH", None)
                headers.append(target.read_bytes()[:16])
            out[mode] = headers
    return out


def pyc() -> Figure:
    headers = _pyc_headers()
    fig = Figure(
        760,
        380,
        alt="The 16-byte header of a .pyc file, compiled twice from the same source saved a minute apart. Timestamp "
        "mode: magic number, flags 0, the source's modification time and size; the time differs, so the files "
        "differ. Checked-hash mode, with SOURCE_DATE_EPOCH set: magic number, flags 3, an 8-byte hash of the "
        "source; the two files are identical",
    )
    fig.text(14, 28, "The first 16 bytes of a .pyc, compiled from one file saved a minute apart", kind="title")
    fields = {
        "timestamp": [(0, 4, "magic"), (4, 8, "flags"), (8, 12, "source time"), (12, 16, "source size")],
        "hash": [(0, 4, "magic"), (4, 8, "flags"), (8, 16, "hash of the source")],
    }
    titles = {"timestamp": "by default", "hash": "with SOURCE_DATE_EPOCH set"}
    y = 58
    for mode in ["timestamp", "hash"]:
        first, second = headers[mode]
        fig.text(14, y + 14, titles[mode], kind="subtitle")
        for k, header in enumerate([first, second]):
            row_y = y + 46 + k * 40
            fig.text(14, row_y + 20, f"saved at +{k} min", kind="label")
            for start, end, name in fields[mode]:
                x = 150 + start * 37
                differs = header[start:end] != (second if k == 0 else first)[start:end]
                role = "dec" if differs else ("data" if name == "magic" else "plain")
                fig.box(x, row_y, (end - start) * 37 - 6, 30, role=role, label=header[start:end].hex(" "))
                if k == 0:
                    fig.text(x + ((end - start) * 37 - 6) / 2, y + 38, name, kind="small", anchor="middle")
        same = "the files differ" if first != second else "identical bytes"
        fig.text(746, y + 14, same, kind="label", anchor="end")
        y += 150
    fig.text(14, 368, "flags 0: check the time; flags 3: check the hash (PEP 552)", kind="note")
    return fig


# ---------------------------------------------------------------- the CI job


def ci() -> Figure:
    fig = Figure(
        760,
        300,
        alt="The image job in GitHub Actions. A pull request or a release tag starts it. It builds the image for "
        "x86-64 and ARM, then builds it again from scratch, and compares the two digests; if they differ, the job "
        "fails. On a release tag, the first build is pushed to ghcr.io/xmarva/perevod with the package's version as "
        "its tag",
    )
    fig.text(14, 28, "The image job: build, build again from scratch, compare, publish", kind="title")
    fig.box(14, 70, 150, 40, role="plain", label="pull request")
    fig.box(14, 130, 150, 40, role="plain", label="tag perevod-v*")
    fig.arrow((168, 90), (210, 90), (210, 118), (232, 118), role="muted")
    fig.arrow((168, 150), (210, 150), (210, 122), (232, 122), role="muted")
    fig.box(236, 70, 200, 40, role="enc", label="build: amd64 + arm64")
    fig.box(236, 130, 200, 40, role="enc", label="build again, no cache")
    fig.arrow((336, 112), (336, 126), role="enc")
    fig.arrow((440, 120), (486, 120), role="enc")
    fig.box(490, 98, 120, 44, role="att", label="same digest?")
    fig.arrow((614, 108), (652, 84), role="att")
    fig.box(656, 64, 90, 36, role="data", label="push", strong=True)
    fig.arrow((614, 132), (652, 156), role="dec")
    fig.box(656, 140, 90, 36, role="dec", label="fail")
    fig.text(656, 120, "on a tag only", kind="small")
    fig.text(14, 214, "SOURCE_DATE_EPOCH = the commit's time · BuildKit v0.34.1 · no attestations",
             kind="mono")
    fig.text(14, 240, "QEMU emulates ARM on the x86 runner, so one job builds both architectures.", kind="note")
    fig.text(14, 262, f"The pushed image: ghcr.io/xmarva/perevod:{VERSION}, public, pulled without a login.",
             kind="note")
    return fig


# ---------------------------------------------------------------- the model in a browser


def _space_sizes():
    from huggingface_hub import HfApi

    sizes = {}
    for item in HfApi().list_repo_tree(SPACE, repo_type="space", recursive=True):
        if item.path.startswith("model/"):
            sizes[item.path.removeprefix("model/")] = item.size
    return sizes


def browser() -> Figure:
    sizes = _space_sizes()
    fig = Figure(
        760,
        350,
        alt="The in-browser demo. A static Hugging Face Space serves the page, the tokenizer as JSON and two ONNX "
        "files: the encoder and one decoder step. In the visitor's browser, a JavaScript tokenizer turns the text "
        "into ids, ONNX Runtime Web runs the encoder once and the decoder step once per beam search step, and the "
        "beam search in JavaScript picks the translation",
    )
    fig.text(14, 28, "The model in the visitor's browser: a static Space and ONNX Runtime Web", kind="title")
    fig.text(14, 62, "huggingface.co/spaces/xmarva/perevod", kind="label")
    files = [
        ("index.html, app.js", None, "plain"),
        ("tokenizer.json", sizes["tokenizer.json"], "data"),
        ("encoder.onnx", sizes["encoder.onnx"], "enc"),
        ("decoder.onnx", sizes["decoder.onnx"], "dec"),
    ]
    y = 76
    for name, size, role in files:
        fig.box(14, y, 220, 34, role=role)
        fig.text(28, y + 22, name, kind="body")
        if size:
            fig.text(222, y + 22, megabytes(size), kind="count", anchor="end")
        y += 44
    fig.text(14, y + 10, "downloaded once, then kept by the browser", kind="note")
    fig.arrow((240, 170), (300, 170), role="muted")
    fig.line((306, 62), (746, 62), (746, 330), (306, 330), (306, 62), role="muted", width=1.2, dash=True)
    fig.text(322, 84, "the visitor's device", kind="label")
    fig.box(322, 98, 180, 36, role="data", label="tokenizer.js")
    fig.text(512, 121, "English text → ids", kind="note")
    fig.box(322, 152, 180, 36, role="enc", label="encoder, once")
    fig.arrow((412, 136), (412, 148), role="data")
    fig.box(322, 206, 180, 36, role="dec", label="decoder step")
    fig.arrow((412, 190), (412, 202), role="enc")
    fig.box(540, 206, 190, 36, role="att", label="beam search, JS")
    fig.arrow((506, 216), (536, 216), role="dec")
    fig.arrow((634, 244), (634, 262), (412, 262), (412, 246), role="att")
    fig.text(340, 286, "four hypotheses, one token more each step", kind="note")
    fig.text(322, 316, "ONNX Runtime Web, in WebAssembly: no server", kind="label")
    return fig


# ---------------------------------------------------------------- cover: two builds, one digest


def _cover(lang):
    english = lang == "en"
    data = image()
    layers = data["platforms"]["amd64"]["manifest"]["layers"]
    out = [cv.label(cv.MOTIF_X0, 128, "TWO BUILDS, ONE DIGEST" if english else "ДВЕ СБОРКИ, ОДИН ХЕШ")]
    biggest = max(layer["size"] for layer in layers)
    for column, x0 in enumerate([cv.MOTIF_X0 + 6, cv.MOTIF_X0 + 270]):
        out.append(cv.text(x0, 168, ("build 1" if english else "сборка 1") if column == 0 else
                           ("build 2, no cache" if english else "сборка 2, без кеша"), 16, cv.FAINT, cv.MONO))
        for k, layer in enumerate(reversed(layers)):
            y = 186 + k * 21
            width = max(14, 220 * (layer["size"] / biggest) ** 0.5)
            out.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{width:.1f}" height="15" rx="3" '
                       f'fill="{cv.CHIP_FILL}" stroke="{cv.LINE}" stroke-width="1.2"/>')
    digest = "sha256:" + short(data["index_digest"], 12) + "…"
    y = 186 + len(layers) * 21 + 22
    for x0 in [cv.MOTIF_X0 + 6, cv.MOTIF_X0 + 270]:
        chip, width = cv.chip(x0, y, digest, size=16, kind="hot", h=36)
        out.append(chip)
        cv.fits(x0 + width)
    out.append(cv.text(cv.MOTIF_X0 + 6, y + 70, f"ghcr.io/xmarva/perevod:{VERSION}", 16, cv.MUTED, cv.MONO))
    title = ["A reproducible", "Docker image"] if english else ["Воспроизводимый", "Docker-образ"]
    hook = ["Same commit,", "same bytes"] if english else ["Тот же коммит,", "те же байты"]
    return cv.frame("09_docker.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "anatomy": anatomy,
    "pyc": pyc,
    "ci": ci,
    "browser": browser,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}


def fetch(version=VERSION):
    # the published image's index, and each platform's manifest and config, as figures need them
    import hashlib

    import requests

    repository = "xmarva/perevod"
    scope = f"repository:{repository}:pull"
    token = requests.get("https://ghcr.io/token", params={"scope": scope}).json()["token"]
    accept = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json"
    headers = {"Authorization": f"Bearer {token}", "Accept": accept}
    response = requests.get(f"https://ghcr.io/v2/{repository}/manifests/{version}", headers=headers)
    response.raise_for_status()
    out = {"version": version, "index_digest": "sha256:" + hashlib.sha256(response.content).hexdigest(),
           "platforms": {}}
    for entry in response.json()["manifests"]:
        url = f"https://ghcr.io/v2/{repository}/manifests/{entry['digest']}"
        manifest = requests.get(url, headers=headers).json()
        url = f"https://ghcr.io/v2/{repository}/blobs/{manifest['config']['digest']}"
        config = requests.get(url, headers=headers).json()
        out["platforms"][entry["platform"]["architecture"]] = {
            "manifest_digest": entry["digest"], "manifest": manifest, "config": config}
    (HERE / "image.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("saved", out["index_digest"])


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else VERSION)
