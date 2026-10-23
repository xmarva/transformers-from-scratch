"""Three body shapes for the 3D figures of the portfolio page "AI Fashion App".

MakeHuman's base mesh and its body-shape targets, released under CC0 (makehumancommunity/makehuman): a young
female body (the mean of the three ethnic targets, as MakeHuman's default does), and the same body with its
minimum-weight and maximum-weight targets applied in full. Only the faces of the body group are kept.

Writes data/bodies.npz: faces (quads) and the vertices of the three builds, in centimetres, y up.

    uv run python articles/fashion_tryon/bodies.py
"""

import io
import urllib.request
from pathlib import Path

import numpy as np

REPO = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/"
TARGETS = REPO + "targets/macrodetails/"
OUT = Path(__file__).parent / "data" / "bodies.npz"


def fetch(url):
    return urllib.request.urlopen(url).read().decode("utf-8")


vertices, faces, group = [], [], None
for line in io.StringIO(fetch(REPO + "3dobjs/base.obj")):
    if line.startswith("v "):
        vertices.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("g "):
        group = line.split()[1]
    elif line.startswith("f ") and group == "body":
        faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
base = np.array(vertices)
faces = np.array(faces)


def target(name):
    offsets = np.zeros_like(base)
    for line in io.StringIO(fetch(TARGETS + name + ".target")):
        if line.startswith("#") or not line.strip():
            continue
        index, x, y, z = line.split()
        offsets[int(index)] = [float(x), float(y), float(z)]
    return offsets


female = base + sum(target(f"{race}-female-young") for race in ("african", "asian", "caucasian")) / 3
builds = {
    "thin": female + target("universal-female-young-averagemuscle-minweight"),
    "average": female,
    "full": female + target("universal-female-young-averagemuscle-maxweight"),
}

used = np.unique(faces)
remap = np.full(len(base), -1)
remap[used] = np.arange(len(used))
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, faces=remap[faces].astype(np.int32),
                    **{name: (v[used] * 10).astype(np.float32) for name, v in builds.items()})  # decimetres to cm
heights = {name: round(float(np.ptp(v[used][:, 1]) * 10), 1) for name, v in builds.items()}
print(heights, "cm tall;", len(faces), "faces")
