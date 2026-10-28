# %% [markdown]
# # Lesson 9. It works on my laptop: a reproducible Docker image of a model
#
# On 16 June 2024 NumPy 2.0.0 came out, and code that nobody had touched started to fail. A model served with
# PyTorch 2.3.1 printed this on import, and stopped
# ([PyTorch issue #129310](https://github.com/pytorch/pytorch/issues/129310#issuecomment-2185605356)):
#
# ```
# A module that was compiled using NumPy 1.x cannot be run in
# NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
# versions of NumPy, modules must be compiled with NumPy 2.0.
# Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
# ```
#
# Nothing in those projects had changed. Their requirements said `numpy`, or `numpy>=1.24`, and on that Sunday the
# newest NumPy that matched became a different program. NumPy had warned about it in its release notes: packages
# "built against a NumPy 1.xx release will not work with NumPy 2.0"
# ([NumPy 2.0.0 release notes](https://numpy.org/doc/stable/release/2.0.0-notes.html)). PyTorch caught up five
# weeks later, in 2.4.0. Everyone who reinstalled in between learned what their model was really made of.
#
# It is made of more than weights. When perevod translates a sentence, its 38 MB of weights are one part of what
# runs: around them are the package's code, three dozen Python packages, Python itself, the system libraries
# that Python and PyTorch load, and a processor. Change any of them, and you have a different program,
# which may or may not give the same translations, or start at all.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/hero-light.png" width="760" alt="What runs when perevod translates, as a stack: the weights, perevod's code, the Python packages, Python and Debian's libraries, the build itself, and the processor. Next to each, what pins it in perevod 0.3.1: the model's commit, a git tag, a lock file with hashes, a base image pinned by digest, a pinned builder and a fixed time; the processor is the one part no image can pin">
#
# This lesson pins every layer of that stack except the last, and puts the result in one Docker image that gives
# the same bytes each time it's built, so that anyone can check it was built from the code they can read. Then it
# takes the image apart without Docker, runs the model from it, and finally ships the same model a second way:
# to the browser, where a free page translates on the visitor's own device.
#
# **What you'll do**
#
# 1. Watch the same requirements turn into three different programs on three different days.
# 2. Freeze them in a lock file with hashes, and see pip refuse a file that doesn't match.
# 3. Take a container image apart by hand: a registry, manifests, layers and digests.
# 4. Find out why two builds of one Dockerfile differ, and remove each cause.
# 5. Write perevod's Dockerfile, and a GitHub Actions job that builds the image twice and publishes it.
# 6. Pull the published image without Docker, run the model from it, and check its weights.
# 7. Export the model to ONNX and run it in a browser, on a free Hugging Face Space.
#
# **What you need**: a CPU, as in lessons 7 and 8. Not Docker: Kaggle and Colab don't have it, so the notebook
# does by hand what Docker does, which turns out to be the best way to see what an image is. It downloads the
# published image (about 365 MB) and runs in under ten minutes. On your own machine with Docker, the lesson also
# shows the commands that build and run the image.

# %% tags=["include:setup"]

# %%
import gzip
import hashlib
import io
import json
import math
import os
import py_compile
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download

print("Python", sys.version.split()[0], "on", os.uname().machine)

# %% [markdown]
# ## 1. One requirements list, three programs
#
# Start from perevod as lesson 8 released it, version 0.2.0: the release's tag, and only the `perevod/` folder
# of the course repository.

# %%
REPO = WORK_DIR / "course"
shutil.rmtree(REPO, ignore_errors=True)
URL = "https://github.com/xmarva/transformers-from-scratch"
QUIET = ["-c", "advice.detachedHead=false"]
subprocess.run(["git", *QUIET, "clone", "-q", "--depth", "1", "--branch", "perevod-v0.2.0",
                "--filter=blob:none", "--sparse", URL, str(REPO)], check=True)
subprocess.run(["git", "sparse-checkout", "set", "perevod"], cwd=REPO, check=True)
os.chdir(REPO)


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text.lstrip("\n"), encoding="utf-8")


def replace(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


print(Path("perevod/pyproject.toml").read_text().split("[project.scripts]")[0])

# %% [markdown]
# Its dependencies are ranges: `torch>=2.1` means "any PyTorch from 2.1 on". That's right for a package other
# people install. perevod has to live next to whatever else they have, and the Python Packaging User Guide calls
# pinning exact versions there "overly-restrictive"
# ([install_requires vs requirements files](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/)).
# But a range is a question, and pip answers it on the day you install.
#
# You can ask it on other days. [uv](https://docs.astral.sh/uv/), a fast replacement for pip written in Rust, can
# resolve requirements as if it were an earlier date: `--exclude-newer` makes it ignore every file uploaded after
# it. Here is what perevod's requirements, with the server extra, resolved to a year before this lesson, half a
# year before, and on the day perevod's lock file was made:

# %%
# !{sys.executable} -m pip install -q uv==0.12.24
UV = [sys.executable, "-m", "uv"]


def resolve(day):
    command = [*UV, "pip", "compile", "perevod/pyproject.toml", "--extra", "serve",
               "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28",
               "--torch-backend", "cpu", "--exclude-newer", day,
               "--no-header", "--no-annotate", "--quiet"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    versions = {}
    for line in result.stdout.splitlines():
        name, version = line.split("==")
        versions[name] = version
    return versions


DAYS = ["2025-10-27T00:00:00Z", "2026-04-27T00:00:00Z", "2026-10-27T00:00:00Z"]
resolved = {}
for day in DAYS:
    resolved[day[:10]] = resolve(day)

DIRECT = ["torch", "tokenizers", "huggingface-hub", "safetensors", "fastapi", "uvicorn"]
rows = []
for name in DIRECT:
    row = {"package": name}
    for day, versions in resolved.items():
        row[day] = versions[name]
    rows.append(row)
display(table(pd.DataFrame(rows)))

first, last = resolved["2025-10-27"], resolved["2026-10-27"]
changed = []
for name in first:
    if name in last and first[name] != last[name]:
        changed.append(name)
print(len(first), "packages a year ago,", len(last), "on the release day")
print(len(changed), "packages installed on both days, at different versions")
print("gone:", sorted(set(first) - set(last)))
print("new: ", sorted(set(last) - set(first)))

# %% [markdown]
# Same `pyproject.toml`, three programs. PyTorch moves by several releases, and each of them changes kernels,
# defaults and warnings. huggingface-hub, which downloads the model, jumps two major versions. And these are
# only the packages perevod names: behind them, nearly every package it pulls in changed version too, and some
# were replaced outright, an HTTP library by another. None of this is a bug. It's what a range asks for.
#
# For a library, that's the deal. For a service, it's a gamble: the server you test today and the one that
# restarts next month install different code. Donald Stufft, who maintained pip and PyPI for years, drew the line
# in 2013: "While a library tends to want to have wide open ended version specifiers an application wants very
# specific dependencies" ([setup.py vs requirements.txt](https://caremad.io/posts/2013/07/setup-vs-requirement/)).
# perevod is both. It keeps the ranges in `pyproject.toml` for people who install it as a library, and the
# service is deployed from a second file that pins every package: a **lock file**.
#
# ## 2. A lock file, with hashes
#
# First, a change that the image revealed. On a laptop, some other package has always installed NumPy; in an
# image that holds only what perevod asks for, it's missing, and PyTorch, which tries to import NumPy when it
# loads, prints `Failed to initialize NumPy: No module named 'numpy'`. perevod never calls NumPy, so nothing
# breaks, but a server that starts with a warning trains people to ignore warnings. perevod now asks for it:

# %%
replace("perevod/pyproject.toml", '    "safetensors>=0.4",\n',
        '    "safetensors>=0.4",\n'
        '    "numpy>=1.24",  # perevod doesn\'t use it, '
        'but PyTorch warns on import without it\n')

# %% [markdown]
# Now the lock. `uv pip compile` resolves the requirements once and writes the answer down, one exact version per
# package. Three options matter here:
#
# - `--universal`: one file for every platform. Where the answer differs, a line carries a marker, such as
#   `; sys_platform == 'win32'`, and pip skips the lines that don't apply.
# - `--torch-backend cpu`: PyTorch's CPU-only build from PyTorch's own index. The default Linux wheel on PyPI
#   is built with CUDA and pulls in NVIDIA's libraries: for torch 2.14.1 that's about 3 GB of downloads, against
#   196 MB for the CPU build. A translation server without a GPU needs none of it.
# - `--generate-hashes`: next to each version, the sha256 of every file that version may be installed from.
#
# And one more for this lesson: `--exclude-newer` with the day perevod 0.3's lock was made, so that the notebook
# resolves the same answer even if you run it a year from now.

# %%
LOCKED_ON = "2026-10-27T12:00:00Z"  # the day perevod 0.3's lock was made
COMPILE = [*UV, "pip", "compile", "--universal", "--python-version", "3.12",
           "--generate-hashes", "--exclude-newer", LOCKED_ON, "--quiet"]
subprocess.run([*COMPILE, "pyproject.toml", "--extra", "serve", "--torch-backend", "cpu",
                "-o", "requirements.lock"], cwd="perevod", check=True)

lock = Path("perevod/requirements.lock").read_text()
pins = []
for line in lock.splitlines():
    if "==" in line and not line.startswith((" ", "#")):
        pins.append(line.rstrip(" \\"))
print(len(pins), "pinned lines,", lock.count("--hash=sha256:"), "hashes")
for pin in pins:
    if pin.startswith(("torch==", "numpy==", "huggingface-hub==")):
        print(pin)

# %% [markdown]
# PyTorch gets two lines. On macOS, PyPI's wheel is already CPU-only and the PyTorch index has no `+cpu` build,
# so the lock takes the plain one there and the `+cpu` build everywhere else.
#
# The hashes are what make the lock more than a list of versions. A version is a name: whoever controls the
# index, a mirror, or a proxy between you and it, decides which bytes stand behind it. A hash is the bytes
# themselves. With a hash on every line, pip switches to hash-checking mode, in which "Hashes are required for
# *all* requirements" and every downloaded file must match one of them
# ([pip: secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/)).
#
# Try it on one small package from the lock, h11, an HTTP/1.1 library the server uses. First as locked:

# %%
lines = lock.splitlines()
start = 0
for k, line in enumerate(lines):
    if line.startswith("h11=="):
        start = k
entry = [lines[start]]
for line in lines[start + 1:]:
    if not line.startswith("    --hash"):
        break
    entry.append(line)
good = "\n".join(entry).rstrip(" \\") + "\n"
print(good)

PIP_DOWNLOAD = [sys.executable, "-m", "pip", "download", "--no-deps", "--require-hashes",
                "--no-color", "--disable-pip-version-check"]
shutil.rmtree(WORK_DIR / "downloads", ignore_errors=True)
Path(WORK_DIR / "good.txt").write_text(good)
subprocess.run([*PIP_DOWNLOAD, "-q", "-d", str(WORK_DIR / "downloads"),
                "-r", str(WORK_DIR / "good.txt")], check=True)
for path in sorted((WORK_DIR / "downloads").glob("h11-*")):
    print(path.name, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())

# %% [markdown]
# The file pip downloaded has one of the locked hashes. Now change the last hex digit of each hash, which is what
# pip would see if someone swapped the file behind the version:

# %%
bad_lines = []
for line in good.splitlines():
    if "--hash=sha256:" in line:
        digest = line.split("sha256:")[1][:64]
        last = "1" if digest[-1] == "0" else "0"
        line = line.replace(digest, digest[:-1] + last)
    bad_lines.append(line)
Path(WORK_DIR / "bad.txt").write_text("\n".join(bad_lines) + "\n")
shutil.rmtree(WORK_DIR / "tampered", ignore_errors=True)
result = subprocess.run([*PIP_DOWNLOAD, "-d", str(WORK_DIR / "tampered"),
                         "-r", str(WORK_DIR / "bad.txt")], capture_output=True, text=True)
print("pip exit code:", result.returncode)
print(result.stderr.strip().replace(str(WORK_DIR) + "/", ""))
print("files downloaded:", list((WORK_DIR / "tampered").glob("*")))

# %% [markdown]
# pip refuses, names the package, and shows both hashes, the one the lock expects and the one it got. Nothing
# is installed. The lock now pins the bytes of every file, not just a version number.
#
# One more program runs during the install, and nobody asked for it by name: the **build backend**. To install
# perevod from source, pip first builds a wheel with hatchling, the backend `pyproject.toml` names, and by default
# it builds in a fresh environment with whatever hatchling is newest that day. A newer hatchling writes slightly
# different metadata into the wheel, so it gets a lock of its own, and the Dockerfile will build without that
# fresh environment:

# %%
write("perevod/build.in", "hatchling\n")
subprocess.run([*COMPILE, "build.in", "-o", "build.lock"], cwd="perevod", check=True)
for line in Path("perevod/build.lock").read_text().splitlines():
    if "==" in line and not line.startswith((" ", "#")):
        print(line.rstrip(" \\"))

# %% [markdown]
# ## 3. What an image is
#
# A Docker image sounds like a disk image, one big file. It isn't: it's a few small JSON documents and a stack of
# tar archives, stored in a registry under names that are their own sha256 hashes. All of it comes over plain
# HTTP, and fetching it once by hand explains most of the rest of this lesson.
#
# Registries speak one protocol, the OCI distribution spec: Docker Hub, GitHub's registry and the others. To
# pull, a client asks `/v2/<name>/manifests/<tag>` for a **manifest**, then downloads what the manifest lists from
# `/v2/<name>/blobs/<digest>`
# ([OCI distribution spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md#pulling-manifests)).
# Public images need no account, only a short-lived anonymous token. Here are the four requests, as functions:

# %%
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"


def sha256(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def pull_token(registry, repository):
    scope = f"repository:{repository}:pull"
    if registry == "registry-1.docker.io":
        params = {"service": "registry.docker.io", "scope": scope}
        response = requests.get("https://auth.docker.io/token", params=params)
    else:
        response = requests.get(f"https://{registry}/token", params={"scope": scope})
    return response.json()["token"]


def get_manifest(registry, repository, reference, token):
    accept = ", ".join([INDEX, MANIFEST, DOCKER_LIST, DOCKER_MANIFEST])
    headers = {"Authorization": f"Bearer {token}", "Accept": accept}
    url = f"https://{registry}/v2/{repository}/manifests/{reference}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response


def get_blob(registry, repository, digest, token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://{registry}/v2/{repository}/blobs/{digest}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    assert sha256(response.content) == digest  # the name of a blob is the hash of its bytes
    return response.content


# %% [markdown]
# perevod's image is built on `python:3.12-slim`, the official Python image on Debian. Ask Docker Hub for it:

# %%
HUB = "registry-1.docker.io"
hub_token = pull_token(HUB, "library/python")
response = get_manifest(HUB, "library/python", "3.12-slim", hub_token)
print("Content-Type:         ", response.headers["Content-Type"])
print("Docker-Content-Digest:", response.headers["Docker-Content-Digest"])
print("sha256 of the body:   ", sha256(response.content))

index = response.json()
rows = []
attestations = 0
for entry in index["manifests"]:
    platform = entry["platform"]
    if platform["architecture"] == "unknown":
        attestations += 1
        continue
    name = platform["os"] + "/" + platform["architecture"]
    if "variant" in platform:
        name += "/" + platform["variant"]
    rows.append({"platform": name, "manifest": entry["digest"][:19] + "…",
                 "bytes": entry["size"]})
display(table(pd.DataFrame(rows)))
print("and", attestations, "attestation manifests, one per image")

# %% [markdown]
# The digest the registry reports is the sha256 of the bytes it sent, and the OCI image spec makes that the
# content's name: a digest "acts as a content identifier, enabling content addressability"
# ([OCI image spec: descriptors](https://github.com/opencontainers/image-spec/blob/main/descriptor.md)). Change
# one byte, and it's a different object with a different name.
#
# What came back is an **image index**: one image per processor architecture, from x86-64 to IBM's mainframes,
# plus as many attestations, records of how each image was built. `docker pull` takes the entry for your
# machine. Follow the one for x86-64 down to its **config** and its **layers**:

# %%
for entry in index["manifests"]:
    if entry["platform"]["architecture"] == "amd64":
        amd64 = entry
manifest = get_manifest(HUB, "library/python", amd64["digest"], hub_token).json()
config_digest = manifest["config"]["digest"]
config = json.loads(get_blob(HUB, "library/python", config_digest, hub_token))
print("built:", config["created"])
for item in config["config"]["Env"]:
    if item.startswith("PYTHON_VERSION"):
        print(item)

steps = []
for step in config["history"]:
    if not step.get("empty_layer"):
        steps.append(step["created_by"])
rows = []
for layer, step in zip(manifest["layers"], steps, strict=True):
    command = " ".join(step.split())
    rows.append({"layer": layer["digest"][:19] + "…", "MB": layer["size"] / 1e6,
                 "made by": command[:58]})
display(table(pd.DataFrame(rows), digits=1))

# %% [markdown]
# That's the whole structure:
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/anatomy-light.png" width="760" alt="The structure of an image, top to bottom. A tag points to an image index. The index lists one manifest per platform by digest. A manifest lists a config and the layers by digest. The config holds the environment, the command and the hash of each unpacked layer; each layer is a gzipped tar archive of the files one Dockerfile step changed. Each level names the next by its sha256, so the digest at the top fixes every byte below it">
#
# A **layer** is a gzipped tar archive of the files one step of a Dockerfile added or changed. The **config** is a
# JSON document with what Docker needs to start the image: environment variables, the user, the command, and
# the hash of each unpacked layer. The **manifest** lists the config and the layers by digest, and the **index**
# lists the manifests. Every level names the next one by its hash, so the digest at the top fixes every byte
# below it. `FROM python:3.12-slim@sha256:…` pins the whole base image down to the last file.
#
# `FROM python:3.12-slim` pins nothing. A tag is a pointer, and Docker says so plainly: "Image tags are mutable,
# meaning a publisher can update a tag to point to a new image"
# ([Docker: build best practices](https://docs.docker.com/build/building/best-practices/#pin-base-image-versions)).
# The official images are rebuilt whenever Debian or Python ships a fix; the maintainers "strive to publish
# updated images at least monthly for Debian" ([docker-library FAQ](https://github.com/docker-library/faq)).
# Compare today's tag with the digest perevod's Dockerfile pins, which was the tag's target the day the
# Dockerfile was written:

# %%
PINNED = "sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f"
today = response.headers["Docker-Content-Digest"]
if today == PINNED:
    print("python:3.12-slim still points to the image perevod pins")
else:
    print("python:3.12-slim has moved: today it points to", today[:19] + "…")
pinned = get_manifest(HUB, "library/python", PINNED, hub_token)
print("the pinned image is still served by its digest:", sha256(pinned.content) == PINNED)

# %% [markdown]
# ## 4. Why two builds of one Dockerfile differ
#
# The definition the reproducible-builds project uses is strict: "A build is reproducible if given the same
# source code, build environment and build instructions, any party can recreate bit-by-bit identical copies of
# all specified artifacts" ([reproducible-builds.org](https://reproducible-builds.org/docs/definition/)). For an
# image, bit by bit means the same digest at the top. It's worth the trouble for one reason: then anyone can
# check that the image was built from the source they can read, by building it and comparing one hash.
#
# Docker doesn't give you that by default. Build one Dockerfile twice without the cache, and you get two
# digests. This lesson's image did exactly that on its first two builds, and the differences were in places that
# have nothing to do with the program. Two of them you can reproduce right here.
#
# The first is time. A layer is a tar archive, and tar stores each file's modification time. Pack perevod's
# source as a layer, then "save" the same files a minute later, as a fresh checkout in CI would:

# %%
commit_time = subprocess.run(["git", "log", "-1", "--format=%ct"], capture_output=True,
                             text=True, check=True)
EPOCH = int(commit_time.stdout)  # the time of the commit the notebook started from


def layer_digest(folder, normalize):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(Path(folder).rglob("*")):
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(folder)))
            if normalize:
                info.mtime = min(info.mtime, EPOCH)  # no time later than the commit
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
            if path.is_file():
                with open(path, "rb") as file:
                    tar.addfile(info, file)
            else:
                tar.addfile(info)
    return sha256(gzip.compress(buffer.getvalue(), mtime=0))  # gzip stores a time too


def touch(path):
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 60))


SOURCE = Path("perevod/src")
as_is = layer_digest(SOURCE, normalize=False)
normalized = layer_digest(SOURCE, normalize=True)
for path in SOURCE.rglob("*.py"):
    touch(path)
print("as on disk:", as_is[:23], "->", layer_digest(SOURCE, normalize=False)[:23])
print("normalized:", normalized[:23], "->", layer_digest(SOURCE, normalize=True)[:23])

# %% [markdown]
# Same bytes in every file, a new digest. Clamping every time to the commit's time, and dropping the owner's
# name, fixes it. BuildKit, the engine behind `docker build`, does exactly this when asked: it takes the time
# from the `SOURCE_DATE_EPOCH` variable, a convention of the reproducible-builds project, and with the exporter
# option `rewrite-timestamp=true` it clamps the time of every file in every layer to it
# ([BuildKit: reproducible builds](https://github.com/moby/buildkit/blob/master/docs/build-repro.md)).
#
# The second cause hides inside the files. pip compiles every `.py` file it installs to bytecode, a `.pyc`, and a
# `.pyc` starts with a 16-byte header that tells Python whether the cache is still valid. By default the header
# holds the source file's modification time:

# %%
module = Path("perevod/src/perevod/config.py")
target = WORK_DIR / "config.pyc"


def compiled():
    py_compile.compile(str(module), cfile=str(target))
    return target.read_bytes()


def header(pyc):
    flags = int.from_bytes(pyc[4:8], "little")
    if flags == 0:
        mtime = int.from_bytes(pyc[8:12], "little")
        size = int.from_bytes(pyc[12:16], "little")
        return f"flags {flags}, source modified at {mtime}, source size {size}"
    return f"flags {flags}, source hash {pyc[8:16].hex()}"


first = compiled()
touch(module)
second = compiled()
print(header(first))
print(header(second))
print("same bytes:", first == second)

# %% [markdown]
# Even if the archive's times are clamped, the time inside the file survives, and pip installs thousands of
# `.pyc` files. PEP 552 added a second kind of header, with a hash of the source instead of its time, and
# `py_compile`, which pip uses through `compileall`, switches to it by itself when `SOURCE_DATE_EPOCH` is set
# ([py_compile](https://docs.python.org/3/library/py_compile.html)):

# %%
os.environ["SOURCE_DATE_EPOCH"] = str(EPOCH)
first = compiled()
touch(module)
second = compiled()
del os.environ["SOURCE_DATE_EPOCH"]
print(header(first))
print(header(second))
print("same bytes:", first == second)

# %% [markdown]
# Flags 3 mean "checked hash": Python compares the stored hash with the source's when it imports the module. The
# bytes no longer depend on when the file was saved. There's a catch in the Dockerfile, though: BuildKit hands
# `SOURCE_DATE_EPOCH` to the build, but a `RUN` step only sees a build argument that the Dockerfile declares with
# `ARG`. Without that one line, pip in the image never sees the variable.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/pyc-light.png" width="760" alt="The 16-byte header of a .pyc file, compiled twice from the same source saved a minute apart. Timestamp mode: magic number, flags 0, the source's modification time and size; the time differs, so the files differ. Checked-hash mode, with SOURCE_DATE_EPOCH set: magic number, flags 3, an 8-byte hash of the source; the two files are identical">
#
# The rest of the list, from the first naive builds to two identical ones:
#
# | What differed | Why | The fix |
# |---|---|---|
# | file times in every layer | each file gets the time it was written | `SOURCE_DATE_EPOCH` and `rewrite-timestamp=true` |
# | the dates in the image's config | the clock at build time | `SOURCE_DATE_EPOCH` |
# | every `.pyc` that pip wrote | the header holds the source's time | `ARG SOURCE_DATE_EPOCH`, so pip sees it |
# | a log file in the model's layer | hf-xet names its log by the time | delete it in the same `RUN` |
# | the base image, next month | the tag moves | `FROM python:3.12-slim@sha256:…` |
# | the packages, next month | new releases | the lock file, with hashes |
# | the wheel's metadata | pip fetches the newest hatchling | `build.lock` and `--no-build-isolation` |
# | the archives, and the parser | each BuildKit version writes and parses its own way | BuildKit's version pinned in CI |
# | attestations | they record when and where a build ran | `--provenance=false --sbom=false` |
#
# The order of the entries in a tar archive can differ too, in general; here BuildKit's builds agreed on it. And
# one cause this table missed: perevod 0.3.0 shipped with it, and section 7 shows it.
#
# ## 5. perevod's Dockerfile
#
# Before the image, two small changes in the package. An image runs somewhere you can't see, so it should be able
# to say what it is: `/health` now reports the model's commit along with the package's version. The translator
# remembers where its weights came from; a snapshot downloaded from the Hub lives in a folder named by its
# commit:

# %%
TRANSLATOR = "perevod/src/perevod/translator.py"
replace(TRANSLATOR, "        self.generation = generation\n",
        "        self.generation = generation\n"
        "        self.source = None  "
        "# where the weights came from; from_pretrained fills it in\n")
replace(TRANSLATOR,
        "        if not path.is_dir():\n"
        "            path = Path(snapshot_download(model, revision=revision, "
        "allow_patterns=FILES))\n",
        "        if path.is_dir():\n"
        "            source = str(path)\n"
        "        else:\n"
        "            path = Path(snapshot_download(model, revision=revision, "
        "allow_patterns=FILES))\n"
        '            source = f"{model}@{path.name}"  '
        "# a snapshot's folder is named by its commit\n")
replace(TRANSLATOR,
        "        return cls(network.to(device), tokenizer, "
        "GenerationConfig.from_dict(settings))\n",
        "        generation = GenerationConfig.from_dict(settings)\n"
        "        translator = cls(network.to(device), tokenizer, generation)\n"
        "        translator.source = source\n"
        "        return translator\n")

# %% [markdown]
# The server returns it, and its test checks it:

# %%
replace("perevod/src/perevod/server.py",
        "    async def health():\n"
        '        return {"status": "ok", "version": __version__}\n',
        "    async def health():  "
        "# what is running: the package's version and the model's commit\n"
        "        model = app.state.batcher.translator.source\n"
        '        return {"status": "ok", "version": __version__, "model": model}\n')
replace("perevod/tests/test_server.py",
        "def test_the_health_check(client):\n"
        '    assert client.get("/health").json()["status"] == "ok"\n',
        "def test_the_health_check_names_the_model(client, tiny_dir):\n"
        '    health = client.get("/health").json()\n'
        '    assert health["status"] == "ok"\n'
        '    assert health["model"] == str(tiny_dir)\n')
replace("perevod/src/perevod/__init__.py", '__version__ = "0.2.0"', '__version__ = "0.3.1"')

# %% [markdown]
# Now the Dockerfile, top to bottom.
#
# `FROM` takes the Python image by digest, as section 3 showed. The Dockerfile's parser is part of BuildKit, so
# pinning BuildKit's version in CI pins it too. (A `# syntax=` line on top can name a newer parser, as an image
# of its own; then pin that by digest as well.)
#
# `ARG SOURCE_DATE_EPOCH` is the line from section 4 that lets pip see the build's time. A user with uid 1000
# instead of root: a process that has no reason to write to the system shouldn't be able to. The dependencies are
# installed before the code, and exactly as locked: `--require-hashes` turns on the checks, `--no-deps` makes pip
# install only what the lock lists, and the lock already lists everything. The order matters for speed, too.
# Docker caches each step and reruns a step only when something it depends on changes, so the slow step that
# installs PyTorch, which changes least often, comes first, and editing perevod's code rebuilds only the cheap
# steps after it.
#
# The package is installed with `--no-build-isolation`, so that pip builds it with the locked hatchling from
# the previous step. Then the model, in a folder of its own, at the commit perevod pins. The download leaves more
# than the model behind in `~/.cache`, and all of that goes in the same step, because a file deleted in a later
# step stays in the earlier layer forever. Finally `HF_HUB_OFFLINE=1`: in the running container, "no HTTP calls
# will be made to the Hugging Face Hub"
# ([huggingface_hub: environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)),
# so the image runs where there is no network, and never quietly downloads anything else. The labels say where
# the image came from; GitHub uses `org.opencontainers.image.source` to link the image to its repository.

# %%
write("perevod/Dockerfile", """
# perevod's server with its model inside: one commit builds one image, bit for bit.
# The base image by digest; the Dockerfile's parser comes with BuildKit, pinned in CI.
FROM \\
    python:3.12-slim@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

# a fixed time instead of the clock: with it pip writes .pyc files that don't record when
ARG SOURCE_DATE_EPOCH
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN useradd --uid 1000 --create-home perevod
WORKDIR /home/perevod

# dependencies first, exactly as locked: they change least often, so this layer is cached
COPY requirements.lock build.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock -r build.lock \\
        --extra-index-url https://download.pytorch.org/whl/cpu

# the package itself, built with the locked build backend
COPY pyproject.toml README.md ./src/
COPY src ./src/src
RUN pip install --no-deps --no-build-isolation ./src && rm -rf ./src

# the model, at the commit this version of perevod was tested with, in a folder of its own;
# the rest of what the download leaves in ~/.cache goes: hf-xet's log is named by the time,
# and huggingface_hub caches a list of AI agents that the Hub can change at any moment
ENV HF_HUB_CACHE=/home/perevod/models
USER perevod
RUN python -c "from perevod import Translator; Translator.from_pretrained()" \\
    && rm -rf /home/perevod/.cache
ENV HF_HUB_OFFLINE=1

ARG VERSION=dev REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/xmarva/transformers-from-scratch \\
      org.opencontainers.image.version=$VERSION \\
      org.opencontainers.image.revision=$REVISION
EXPOSE 8000
CMD ["perevod-server", "--host", "0.0.0.0", "--port", "8000"]
""")
assert PINNED in Path("perevod/Dockerfile").read_text()

# %% [markdown]
# The build context, the folder Docker sends to the builder, is `perevod/`. A `.dockerignore` keeps out what the
# image doesn't need, and what would change its bytes without changing the program, like Python's caches:

# %%
write("perevod/.dockerignore", """
# only what the image needs: the code, its metadata and the lock files
tests
**/__pycache__
.pytest_cache
.ruff_cache
dist
build
*.egg-info
""")

# %% [markdown]
# On a machine with Docker, this builds the image and runs it, with the same options CI uses below (Docker
# Desktop and OrbStack ship with BuildKit):
#
# ```bash
# cd perevod
# export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
# docker buildx build . --build-arg SOURCE_DATE_EPOCH --provenance=false --sbom=false \
#     --output type=docker,name=perevod:dev,rewrite-timestamp=true
# docker run --rm -p 8000:8000 perevod:dev
# ```
#
# And without a network at all, from the command line of lesson 7:
#
# ```bash
# docker run --rm --network none perevod:dev perevod "Tom is afraid of the dark."
# ```
#
# One trap when you limit the container's CPUs. `docker run --cpus 2` is a quota on CPU time, not a set of
# processors, and PyTorch doesn't read quotas: it starts as many threads as the host has physical cores, and they
# fight over the two CPUs' worth of time. Tell it explicitly, with `-e OMP_NUM_THREADS=2`.
#
# ## 6. Build twice, publish once
#
# Reproducibility that nobody checks decays: one day a step starts writing a timestamp, and nobody notices until
# the day it matters. So the CI job that publishes the image also checks it. On every pull request that touches
# the package it builds the image twice, the second time without any cache, and compares the two digests. On a
# release tag it does the same and pushes the first build to the GitHub Container Registry, `ghcr.io`, as
# `ghcr.io/xmarva/perevod:<version>`.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/ci-light.png" width="760" alt="The image job in GitHub Actions. A pull request or a release tag starts it. It builds the image for x86-64 and ARM, then builds it again from scratch, and compares the two digests; if they differ, the job fails. On a release tag, the first build is pushed to ghcr.io/xmarva/perevod with the package's version as its tag">
#
# Some details of the workflow below. `SOURCE_DATE_EPOCH` is the time of the commit, so a rebuild of the same
# commit next year uses the same time. QEMU, an emulator, lets the x86 runner build the ARM image too, for
# people on Apple silicon and ARM servers. The builder is pinned, `moby/buildkit:v0.34.1`, because another
# version of BuildKit may pack the same files into different bytes. Attestations are switched off: they record
# when and where a build ran, so no two builds have the same ones. The job logs in to the registry with the
# token GitHub gives every workflow run, and needs the `packages: write` permission for it
# ([GitHub: publishing Docker images](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)).

# %%
write(".github/workflows/image.yml", """
name: image

# perevod's Docker image. On pull requests that touch the package: build it twice and check
# that the two builds are the same, bit for bit. On a release tag: the same, and publish it.
on:
  push:
    tags: ["perevod-v*"]
  pull_request:
    paths: ["perevod/**", ".github/workflows/image.yml"]

env:
  IMAGE: ghcr.io/xmarva/perevod
  PLATFORMS: linux/amd64,linux/arm64
  # OCI media types in both builds, and no attestations: those record when and where a build
  # ran, so no two builds have the same ones
  OUTPUT: oci-mediatypes=true,rewrite-timestamp=true
  PUSH: ${{ github.ref_type == 'tag' }}  # only a release tag publishes

jobs:
  image:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - uses: docker/setup-qemu-action@v4  # runs the arm64 build on an x86 machine
      - uses: docker/setup-buildx-action@v4
        with:
          driver-opts: image=moby/buildkit:v0.34.1  # a pinned builder writes the same bytes
      - name: Settings from the commit
        run: |
          echo "SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)" >> "$GITHUB_ENV"
          init=perevod/src/perevod/__init__.py
          version=$(sed -n 's/^__version__ = "\\(.*\\)"$/\\1/p' "$init")
          echo "VERSION=$version" >> "$GITHUB_ENV"
          echo "REVISION=$GITHUB_SHA" >> "$GITHUB_ENV"
      - name: The tag matches the package's version
        if: github.ref_type == 'tag'
        run: test "$GITHUB_REF_NAME" = "perevod-v$VERSION"
      - uses: docker/login-action@v4
        if: github.ref_type == 'tag'
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build, and push on a release tag
        run: |
          docker buildx build perevod --platform "$PLATFORMS" \\
            --provenance=false --sbom=false \\
            --build-arg SOURCE_DATE_EPOCH --build-arg VERSION --build-arg REVISION \\
            --output "type=image,name=$IMAGE:$VERSION,push=$PUSH,$OUTPUT" \\
            --metadata-file first.json
      - name: Build again from scratch, and compare
        run: |
          docker buildx build perevod --platform "$PLATFORMS" \\
            --provenance=false --sbom=false --no-cache \\
            --build-arg SOURCE_DATE_EPOCH --build-arg VERSION --build-arg REVISION \\
            --output "type=oci,dest=second.tar,$OUTPUT" --metadata-file second.json
          first=$(jq -r '."containerimage.digest"' first.json)
          second=$(jq -r '."containerimage.digest"' second.json)
          echo "first build:  $first"
          echo "second build: $second"
          test "$first" = "$second"
""")

# %% [markdown]
# The README gets the commands to run the image, and the server's, which 0.2.0's README never mentioned:

# %%
replace("perevod/README.md", 'perevod "I think, therefore I am."',
        'perevod "Tom is afraid of the dark."')
replace("perevod/README.md", "The model was trained on short everyday sentences", """\
As a service, with an HTTP API at `/translate`, its documentation at `/docs`, and with
`--demo` a page at `/`:

```bash
pip install "perevod[demo] @ git+$REPO#subdirectory=perevod"
perevod-server --port 8000 --demo
```

Or from the image of a release, with the model inside, which needs no network to run:

```bash
docker run --rm -p 8000:8000 ghcr.io/xmarva/perevod:0.3.1
curl -X POST localhost:8000/translate -H "Content-Type: application/json" \\
    -d '{"text": "Hello!"}'
```

The model was trained on short everyday sentences""")

# %% [markdown]
# A release tag starts the workflow ([its runs](https://github.com/xmarva/transformers-from-scratch/actions/workflows/image.yml)),
# and its log ends with two equal digests, or the job fails. The image it pushed is public, so the notebook can
# check the result itself. The registry is GitHub's this time, with the same four requests:

# %%
VERSION = "0.3.1"
GHCR = "ghcr.io"
IMAGE = "xmarva/perevod"
ghcr_token = pull_token(GHCR, IMAGE)
response = get_manifest(GHCR, IMAGE, VERSION, ghcr_token)
print(f"{GHCR}/{IMAGE}:{VERSION}", response.headers["Docker-Content-Digest"])

manifests = {}
configs = {}
for entry in response.json()["manifests"]:
    arch = entry["platform"]["architecture"]
    manifests[arch] = get_manifest(GHCR, IMAGE, entry["digest"], ghcr_token).json()
    config_digest = manifests[arch]["config"]["digest"]
    configs[arch] = json.loads(get_blob(GHCR, IMAGE, config_digest, ghcr_token))
for name, value in configs["amd64"]["config"]["Labels"].items():
    print(name, "=", value)

# %% [markdown]
# The labels say which commit the image was built from. And the two architectures can be compared layer by
# layer:

# %%
def short(command):
    command = " ".join(command.split())
    command = command.split(" /bin/sh -c ")[-1].replace(" # buildkit", "")
    return command[:56]


steps = []
for step in configs["amd64"]["history"]:
    if not step.get("empty_layer"):
        steps.append(step["created_by"])
rows = []
for k, step in enumerate(steps):
    x86 = manifests["amd64"]["layers"][k]
    arm = manifests["arm64"]["layers"][k]
    same = "yes" if x86["digest"] == arm["digest"] else "no"
    rows.append({"step": short(step), "MB (x86)": x86["size"] / 1e6, "same on ARM": same})
display(table(pd.DataFrame(rows), digits=1))

# %% [markdown]
# Most layers differ, and should: Debian, Python and PyTorch are compiled for each processor. The interesting
# rows are the others. The copied files are the same bytes on both, and so is the model's layer, which was made
# by running Python on each architecture, the ARM one inside an emulator, and keeping what the download wrote.
# Two different processors, one emulated, produced the same bytes, because nothing in that layer depends on when
# or where it was made.
#
# The layer where pip installs perevod differs for a reason that took some digging. Running pip makes Python
# import a few modules of its own standard library that the base image hadn't cached yet, and Python writes
# their `.pyc` files into this layer, along with the folders they live in. Python's import system writes
# timestamp headers no matter what `SOURCE_DATE_EPOCH` says, which only changes the default of `py_compile` and
# `compileall`. The times are those of the standard library's files and folders, which differ between the two
# base images; inside one architecture they're pinned along with the base, so two builds still agree.
#
# There's a stronger test than CI's. CI builds twice on the same machine, minutes apart. The real promise of a
# reproducible build is that anyone, anywhere, later, gets the same bytes. A day after the release, a rebuild
# of the same commit on a laptop, an ARM Mac, with the same command and the same pinned BuildKit, gave the
# published digest, on a fresh builder. On a builder that had built earlier versions, one directory in one
# layer came out with the time of an old build, and the digest differed: rebuild on a fresh builder, as CI always
# does.
#
# ## 7. The image, without Docker
#
# The notebook can't run Docker, but it can do what Docker does when it pulls an image: download the layers,
# check each against its digest, and unpack them on top of each other into one folder. That folder is the
# image's filesystem. One detail matters while unpacking. A layer can't change an earlier layer, so when a step
# deletes a file, its layer carries an empty marker named `.wh.<name>`, a **whiteout**, and the file must be
# removed from what's been unpacked so far. And a tar archive from the internet deserves suspicion: Python's
# `data` filter refuses entries that would land outside the folder, including links to absolute paths, which
# would point into this machine's own system instead of the image's
# ([tarfile: extraction filters](https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter)).

# %%
MACHINES = {"x86_64": ("amd64", "x86_64-linux-gnu", "ld-linux-x86-64.so.2"),
            "aarch64": ("arm64", "aarch64-linux-gnu", "ld-linux-aarch64.so.1")}
ARCH, TRIPLET, LOADER = MACHINES[os.uname().machine]
ROOT = WORK_DIR / "rootfs"
shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir()
refused = []


def unpack(member, path):
    name = Path(member.name)
    if name.name.startswith(".wh."):  # a whiteout: a later layer deleted this file
        gone = Path(path) / name.parent / name.name.removeprefix(".wh.")
        if gone.is_dir() and not gone.is_symlink():
            shutil.rmtree(gone)
        elif gone.exists() or gone.is_symlink():
            gone.unlink()
        return None
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        refused.append(member.name)
        return None


start = time.time()
downloaded = 0
for layer in manifests[ARCH]["layers"]:
    data = get_blob(GHCR, IMAGE, layer["digest"], ghcr_token)  # checked against its digest
    downloaded += len(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(ROOT, filter=unpack)
seconds = time.time() - start
print(f"{downloaded / 1e6:.0f} MB downloaded, checked and unpacked in {seconds:.0f} s")
print(len(refused), "links refused, for example", refused[:3])

# %% [markdown]
# The refused entries are Debian's links such as `/etc/alternatives/awk`, which point to absolute paths. They
# only make sense inside the container, and the program doesn't need them. Look around:

# %%
print(sorted(path.name for path in ROOT.iterdir()))
site = ROOT / "usr/local/lib/python3.12/site-packages"
print(len(list(site.glob("*.dist-info"))), "installed packages")
models = ROOT / "home/perevod/models"
for path in sorted(models.rglob("*")):
    if path.is_file() and "snapshots" in path.parts:
        print(path.relative_to(models), f"{path.stat().st_size:,} bytes")
leftovers = []
if (ROOT / "home/perevod/.cache").exists():
    leftovers = list((ROOT / "home/perevod/.cache").rglob("*"))
print("anything in ~/.cache:", leftovers or "no")

# %% [markdown]
# A Linux system, Python's packages, and the model at the commit perevod pins. Is it the model that's on the
# Hub? The Hub stores the sha256 of every large file, so ask it, and hash the file from the image:

# %%
from huggingface_hub import HfApi

MODEL = "xmarva/perevod-en-ru"
REVISION = "4abc0d63e862ca99d2bd99cd1007e4a9fe47c358"
info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
for file in info.siblings:
    if file.rfilename == "model.safetensors":
        hub_sha = file.lfs.sha256
snapshot = models / "models--xmarva--perevod-en-ru" / "snapshots" / REVISION
weights = snapshot / "model.safetensors"
image_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
print("on the Hub:   ", hub_sha)
print("in the image: ", image_sha)
print("the same:", hub_sha == image_sha)

# %% [markdown]
# Now the cause that the table in section 4 missed. perevod 0.3.0 passed CI's check, two identical builds, and its
# model layer still held a file that had no business there. Open that layer of the 0.3.0 image:

# %%
old_index = get_manifest(GHCR, IMAGE, "0.3.0", ghcr_token).json()
for entry in old_index["manifests"]:
    if entry["platform"]["architecture"] == ARCH:
        old_manifest = get_manifest(GHCR, IMAGE, entry["digest"], ghcr_token).json()
old_model_layer = get_blob(GHCR, IMAGE, old_manifest["layers"][-1]["digest"], ghcr_token)
with tarfile.open(fileobj=io.BytesIO(old_model_layer), mode="r:gz") as tar:
    for member in tar.getmembers():
        if member.isfile() and "/hub/" not in member.name:
            print(member.name, member.size, "bytes")
    found = tar.extractfile("home/perevod/.cache/huggingface/.agent_harnesses.json")
    agents = json.loads(found.read())
names = list(agents["harnesses"])
print(len(names), "AI coding agents, starting with", names[:4])

# %% [markdown]
# Next to the model sat a list of AI coding agents. huggingface_hub 2.x labels its requests with the agent that
# made them, if any, and to recognise agents it downloads the list from the Hub, "at most once a day", and caches
# it in `~/.cache/huggingface`. The list lives on the Hub's server and changes when the Hub adds an agent. So
# the two builds of the release agreed, but a rebuild of the same commit after the next change to that list
# would not: the model layer would hold a different file and get a different digest. Nothing pins a file the
# build downloads from a server, except the build itself. perevod 0.3.1 keeps the model in a folder of its own
# and deletes everything else under `~/.cache` in the same step, which is the line you saw in the Dockerfile.
#
# The lesson in it is general: a step that runs a program records everything that program writes, not only what
# you asked for. Look inside a layer at least once.
#
# Now run the image. A container is, at heart, a process that sees the image's folder as its whole filesystem.
# Docker arranges that with the kernel's namespaces, which a notebook can't use, but there's a simpler trick
# that's enough here: run the image's own Python through the image's own dynamic loader, the program that
# starts every Linux binary and loads its libraries. Pointed at the image's libraries, it runs the image's Python
# with the image's C library and the image's packages, whatever this machine has installed. The environment
# comes from the image's config, with its paths moved into the folder:

# %%
config = configs[ARCH]["config"]
env = {}
for item in config["Env"]:
    name, value = item.split("=", 1)
    if value.startswith("/home/"):
        value = str(ROOT) + value  # a path inside the image
    env[name] = value
env["PATH"] = os.environ["PATH"]
env["HOME"] = str(ROOT / "home/perevod")
env["PYTHONHOME"] = str(ROOT / "usr/local")
env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
libraries = [ROOT / "usr/local/lib", ROOT / "usr/lib" / TRIPLET]
PYTHON = [str(ROOT / "usr/lib" / TRIPLET / LOADER), "--library-path",
          ":".join(str(path) for path in libraries), str(ROOT / "usr/local/bin/python3.12")]
for name in ["HF_HUB_OFFLINE", "HF_HUB_CACHE", "PYTHON_VERSION"]:
    print(name, "=", env[name])

probe = """
import sys, torch, perevod
from perevod import Translator
translator = Translator.from_pretrained(device="cpu")
print("Python", sys.version.split()[0], "| torch", torch.__version__,
      "| perevod", perevod.__version__)
print("model:", translator.source)
"""
result = subprocess.run([*PYTHON, "-c", probe], env=env, capture_output=True, text=True)
print(result.stdout.strip() or result.stderr[-2000:])

# %% [markdown]
# Python 3.12 from the image, its PyTorch, its perevod, and the model loaded from inside it with the Hub switched
# off. This notebook's own Python is a different version, with a different PyTorch, which makes a good
# experiment: install the package from the source the notebook has been editing, translate the same sentences
# both ways, and compare.

# %%
# !{sys.executable} -m pip install -q "./perevod[test,serve]"
importlib.invalidate_caches()

import perevod
from perevod import Translator, decoding

DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
test_path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet",
                            repo_type="dataset", revision=DATA_REVISION)
sentences = pd.read_parquet(test_path)["en"].sample(200, random_state=SEED).tolist()

translator = Translator.from_pretrained(device="cpu")
here = translator.translate(sentences)
print("this notebook: Python", sys.version.split()[0], "| torch", torch.__version__,
      "| perevod", perevod.__version__)

job = """
import json, sys
from perevod import Translator
sentences = json.load(sys.stdin)
translations = Translator.from_pretrained(device="cpu").translate(sentences)
print(json.dumps(translations, ensure_ascii=False))
"""
result = subprocess.run([*PYTHON, "-c", job], env=env, input=json.dumps(sentences),
                        capture_output=True, text=True, check=True)
there = json.loads(result.stdout)
same = 0
for a, b in zip(here, there, strict=True):
    if a == b:
        same += 1
print(f"identical translations: {same} of {len(sentences)}")

# %% [markdown]
# Finally the image's own command, the server, started the same way, and asked the question that 0.3 added:
#
# (On Kaggle and Colab nothing else is listening on port 8009; on your machine, pick a free port.)

# %%
start_server = "from perevod.server import main; main(['--port', '8009'])"
server = subprocess.Popen([*PYTHON, "-c", start_server], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(120):  # up to a minute for the server to start
    try:
        health = requests.get("http://127.0.0.1:8009/health", timeout=1).json()
        break
    except requests.ConnectionError:
        time.sleep(0.5)
print(health)
question = {"text": "Tom is afraid of the dark."}
answer = requests.post("http://127.0.0.1:8009/translate", json=question)
print(answer.json())
server.terminate()
server.wait()

# %% [markdown]
# That's the image working, minus what Docker adds on top: its own process tree, network and hostname
# (namespaces), limits on CPU and memory (cgroups), and a root filesystem the process can't see past. Our trick
# shares all of those with the notebook, which is fine for a check and no way to run a service.
#
# ## 8. The same model in a browser
#
# An image needs a machine to run on, and someone to pay for the machine. For a demo there's a cheaper place to
# run a model this small: the visitor's own device.
#
# The obvious host for a free demo used to be a Hugging Face Space with Gradio on a free CPU. That changed in
# July 2026: now "Gradio and Docker Spaces run on compute and require a paid plan to create", while "Static
# Spaces are free for everyone" ([Spaces documentation](https://huggingface.co/docs/hub/spaces-overview)). A
# static Space serves files, HTML and JavaScript, and nothing stops one of those files from being a model.
# [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) runs ONNX models in the browser with
# WebAssembly, and perevod's weights take 38 MB. So the page downloads the model once, and every translation runs
# on the visitor's processor: no server, and nothing typed leaves the device.
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/browser-light.png" width="760" alt="The in-browser demo. A static Hugging Face Space serves the page, the tokenizer as JSON and two ONNX files: the encoder and one decoder step. In the visitor's browser, a JavaScript tokenizer turns the text into ids, ONNX Runtime Web runs the encoder once and the decoder step once per beam search step, and the beam search in JavaScript picks the translation">
#
# ONNX is a file format for a computation graph: the operations of a network and its weights, readable by many
# runtimes. The model goes in as two graphs. The **encoder** runs once per sentence. The **decoder step** takes
# the hypotheses so far and returns the log-probabilities of the next token for each, so the beam search of
# lesson 3 stays outside the graph, written in JavaScript, where it's a loop of forty lines. The step has no KV
# cache, so each step recomputes the whole prefix; for short sentences that costs less than it sounds, and keeps
# the graph simple.
#
# The export is a script in the course repository, next to the page's files:

# %%
write("space/export_onnx.py", '''
# Export perevod's model to the two ONNX graphs of the in-browser demo:
#   encoder.onnx: source ids -> memory
#   decoder.onnx: hypotheses, memory, source ids -> log-probabilities of the next token
# and the tokenizer, which the page loads as it is.
# Usage: python export_onnx.py --out model
import argparse
import logging
from pathlib import Path

import torch
from torch.export import Dim

from perevod import Translator, decoding


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        return self.model.encode(src)


class DecoderStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tgt, memory, src):
        return self.model.decode(tgt, memory, src)[:, -1].log_softmax(-1)


def main():
    parser = argparse.ArgumentParser(description="Export perevod's model to ONNX")
    parser.add_argument("--out", default="model")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)  # the exporter is chatty
    translator = Translator.from_pretrained(device="cpu")
    model = translator.model
    _, bos, _ = decoding.special_ids(translator.tokenizer)
    ids = decoding.source_ids(translator.tokenizer, "Tom is afraid of the dark.", 256)
    src = torch.tensor([ids, ids])  # two rows, three target tokens: any sizes above one do
    tgt = torch.tensor([[bos, bos, bos], [bos, bos, bos]])
    batch = Dim("batch")
    source = Dim("source", max=256)  # the model's positions
    target = Dim("target", max=256)
    # export leaves a module in the mode it found it in, and a new module starts in training
    # mode: without eval() the wrappers would switch the model's dropout back on afterwards
    encoder = Encoder(model).eval()
    step = DecoderStep(model).eval()
    with torch.no_grad():
        memory = model.encode(src)
        torch.onnx.export(encoder, (src,), out / "encoder.onnx", input_names=["src"],
                          output_names=["memory"], external_data=False, verbose=False,
                          dynamic_shapes={"src": {0: batch, 1: source}})
        torch.onnx.export(step, (tgt, memory, src), out / "decoder.onnx",
                          input_names=["tgt", "memory", "src"], output_names=["logp"],
                          external_data=False, verbose=False,
                          dynamic_shapes={"tgt": {0: batch, 1: target},
                                          "memory": {0: batch, 1: source},
                                          "src": {0: batch, 1: source}})
    translator.tokenizer.save(str(out / "tokenizer.json"))  # the page needs it too
    for path in sorted(out.glob("*.onnx")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
''')

# %%
# !{sys.executable} -m pip install -q onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.31.1
result = subprocess.run([sys.executable, "space/export_onnx.py", "--out", "space/model"],
                        capture_output=True, text=True, check=True)
print(result.stdout.strip())

# %% [markdown]
# The two graphs share the embeddings in PyTorch, so together they're larger than the weights: each file carries
# its own copy. `Dim` tells the exporter which sizes may change between calls: the number of hypotheses, and the
# lengths of the source and of the translation so far. Before anything goes near a browser, check that the graphs
# compute what PyTorch computes. ONNX Runtime has a Python package with the same engine:

# %%
import onnxruntime as ort

encoder = ort.InferenceSession("space/model/encoder.onnx")
decoder = ort.InferenceSession("space/model/decoder.onnx")
pad_id, bos_id, eos_id = decoding.special_ids(translator.tokenizer)

src = torch.tensor([decoding.source_ids(translator.tokenizer, sentences[0], 256)])
tgt = torch.tensor([[bos_id]])
with torch.no_grad():
    memory = translator.model.encode(src)
    expected = translator.model.decode(tgt, memory, src)[:, -1].log_softmax(-1).numpy()
onnx_memory = encoder.run(None, {"src": src.numpy()})[0]
got = decoder.run(None, {"tgt": tgt.numpy(), "memory": onnx_memory, "src": src.numpy()})[0]
print("largest difference in log-probabilities:", float(np.abs(got - expected).max()))
print("the same first token:", int(got.argmax()) == int(expected.argmax()))

# %% [markdown]
# The differences are in the fifth or sixth decimal place: the same arithmetic, done in a different order. What
# matters is the translations. Here is lesson 3's beam search once more, written against the two graphs the way
# the page's JavaScript does it, step for step: every hypothesis is fed again whole, and the four best
# continuations survive.

# %%
def length_penalty(length, alpha=0.6):
    return ((5 + length) / 6) ** alpha


def onnx_translate(sentence, beam_size=4, max_tokens=128):
    ids = decoding.source_ids(translator.tokenizer, sentence, 256)
    src = np.array([ids], dtype=np.int64)
    memory = encoder.run(None, {"src": src})[0]
    beams = [([bos_id], 0.0)]
    finished = []
    for _ in range(max_tokens):
        n = len(beams)
        tgt = np.array([tokens for tokens, score in beams], dtype=np.int64)
        feeds = {"tgt": tgt, "memory": np.repeat(memory, n, axis=0),
                 "src": np.repeat(src, n, axis=0)}
        logp = decoder.run(None, feeds)[0]
        candidates = []
        for (tokens, score), row in zip(beams, logp):
            for token in np.argsort(-row, kind="stable")[:beam_size]:
                candidates.append((tokens + [int(token)], score + float(row[token])))
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = []
        for tokens, score in candidates:
            if tokens[-1] == eos_id:
                finished.append((tokens, score))
            else:
                beams.append((tokens, score))
            if len(beams) == beam_size:
                break
        if len(finished) >= beam_size or not beams:
            break
    pool = finished or beams
    best, best_score = pool[0][0], -math.inf
    for tokens, score in pool:
        normalized = score / length_penalty(len(tokens) - 1)
        if normalized > best_score:
            best, best_score = tokens, normalized
    return translator.tokenizer.decode([token for token in best[1:] if token != eos_id])


start = time.time()
through_onnx = []
for sentence in sentences:
    through_onnx.append(onnx_translate(sentence))
same = 0
for a, b in zip(here, through_onnx, strict=True):
    if a == b:
        same += 1
seconds = time.time() - start
print(f"identical to the package: {same} of {len(sentences)}, in {seconds:.0f} s")
for sentence, translation in list(zip(sentences, through_onnx))[:3]:
    print(sentence, "->", translation)

# %% [markdown]
# Word for word. The page itself is three small files in the course repository's
# [`space/`](https://github.com/xmarva/transformers-from-scratch/tree/main/space) folder, and only one of them
# is about the page. `translate.js` is the function above in JavaScript. `tokenizer.js` rewrites the tokenizer of
# lesson 1, whose `tokenizer.json` the page loads as it is, and it had one trap worth knowing. The tokenizer
# splits text into words with the regular expression `\w+`, and in the Rust library behind Hugging Face's
# `tokenizers`, `\w` matches letters of every alphabet. In JavaScript it matches only `[A-Za-z0-9_]`, even with the
# Unicode flag, so a word like "naïve" or "Москва" would come apart. The JavaScript version spells the classes
# out, `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]`, and a script in the same folder checks it against the
# original on every English sentence of the test set, id for id.
#
# The rest is a static Space: a `README.md` whose header says `sdk: static`, uploaded with the two graphs:
#
# ```bash
# python space/export_onnx.py --out space/model
# hf upload xmarva/perevod space . --repo-type space
# ```
#
# Here it is, [xmarva/perevod](https://huggingface.co/spaces/xmarva/perevod):
#
# <img src="https://raw.githubusercontent.com/xmarva/transformers-from-scratch/main/assets/figures/09/space.png" width="700" alt="The perevod page on Hugging Face Spaces: an English text box with 'My brother lives in a small town by the sea.', the Russian translation 'Мой брат живёт в маленьком городке у моря.' below it, the time the translation took on this device, a note that the model runs on the visitor's device and nothing typed leaves it, and four example sentences">
#
# The first visit downloads the two graphs once, and the browser keeps them. After that the page works without
# a network, and the translation takes about as long as in this notebook, on whatever device it opened on. The
# model can't do more there than here, of course; it's the same model, with the same limits its card lists.
#
# Two ways to ship one model, then, for two jobs. The image is for running it as a service: the same bytes on
# every server, checkable by anyone, with nothing left to chance at startup. The page is for showing it to
# people: free, private, and limited by the visitor's device. Both are built from the same weights at the same
# commit, and both were checked against the package, translation for translation.
#
# ## 9. Release 0.3.1
#
# Before a release, the fast tests, which now include the health check that names the model:

# %%
# !cd perevod && {sys.executable} -m pytest -q -p no:cacheprovider -m "not slow" --color=no

# %% [markdown]
# The release is a commit and a tag, `perevod-v0.3.1`, and the image is whatever CI builds from them. That gives
# the notebook one last check: are the files it wrote and edited the same as the release's? Only the files that
# go into the image count, the build context and the workflow. The lock files are compared without their first
# lines, where uv records the command, which here includes `--exclude-newer`:

# %%
RELEASE = "perevod-v0.3.1"
subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", "tag", RELEASE], check=True)
listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", RELEASE, "perevod"],
                        capture_output=True, text=True, check=True).stdout.split()
IMAGE_FILES = {"Dockerfile", ".dockerignore", "requirements.lock", "build.lock",
               "pyproject.toml", "README.md"}
context = [".github/workflows/image.yml"]
for name in listed:
    if name.startswith("perevod/src/") or name.removeprefix("perevod/") in IMAGE_FILES:
        context.append(name)


def without_command(text):
    lines = []
    for line in text.splitlines():
        if not line.startswith("#    uv pip compile"):
            lines.append(line)
    return lines


different = []
for name in context:
    released = subprocess.run(["git", "show", f"{RELEASE}:{name}"], capture_output=True,
                              text=True, check=True).stdout
    if without_command(Path(name).read_text()) != without_command(released):
        different.append(name)
print(len(context), "files compared, different:", different or "none")

# %% [markdown]
# The notebook rebuilt the release's inputs file for file, and CI turned them into
# `ghcr.io/xmarva/perevod:0.3.1`, the image section 7 took apart. On your machine:
#
# ```bash
# docker run --rm -p 8000:8000 ghcr.io/xmarva/perevod:0.3.1
# curl localhost:8000/health
# ```
#
# 0.3.1 is a **patch** release in lesson 7's terms: the package's code is the same as in 0.3.0, and only the way
# the image is built changed. Pinned by its digest, the image will translate the same way in a year, on any
# machine that runs Docker, with or without a network. That was the point of the lesson.
#
# ## References
#
# - [NumPy 2.0.0 release notes](https://numpy.org/doc/stable/release/2.0.0-notes.html), and the error in practice: [PyTorch issue #129310](https://github.com/pytorch/pytorch/issues/129310#issuecomment-2185605356).
# - Stufft, D. (2013). [setup.py vs requirements.txt](https://caremad.io/posts/2013/07/setup-vs-requirement/). *Abstract and concrete dependencies.*
# - Python Packaging User Guide: [install_requires vs requirements files](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/).
# - pip: [secure installs and hash-checking mode](https://pip.pypa.io/en/stable/topics/secure-installs/); uv: [`uv pip compile`](https://docs.astral.sh/uv/reference/cli/#uv-pip-compile) and [using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
# - Open Container Initiative: [image specification](https://github.com/opencontainers/image-spec) (index, manifest, config, layers, descriptors) and [distribution specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md).
# - [docker-library FAQ](https://github.com/docker-library/faq) and Docker's [build best practices](https://docs.docker.com/build/building/best-practices/), on tags and digests.
# - [reproducible-builds.org](https://reproducible-builds.org/docs/definition/): the definition, and [`SOURCE_DATE_EPOCH`](https://reproducible-builds.org/docs/source-date-epoch/).
# - BuildKit: [reproducible builds](https://github.com/moby/buildkit/blob/master/docs/build-repro.md); Docker: [image exporter options](https://docs.docker.com/build/exporters/image-registry/).
# - [PEP 552: Deterministic pycs](https://peps.python.org/pep-0552/); [py_compile](https://docs.python.org/3/library/py_compile.html).
# - GitHub: [publishing Docker images](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images) and [the Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
# - Python: [tarfile extraction filters](https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter); the dynamic loader: [ld.so(8)](https://man7.org/linux/man-pages/man8/ld.so.8.html).
# - [huggingface_hub environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) (`HF_HUB_CACHE`, `HF_HUB_OFFLINE`).
# - PyTorch: [exporting to ONNX](https://docs.pytorch.org/docs/stable/onnx_export.html); [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
# - Hugging Face: [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and [static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static).
