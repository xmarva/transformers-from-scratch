"""Figures and charts of the portfolio page "AI Fashion App" on xmarva.github.io.

The warp is computed here: a thin-plate spline (Bookstein, 1989) carries fourteen points of a flat product photo
of a dress onto the same points of a person, the way warping-based try-on models of 2018-2021 placed a garment
before a generator repainted it. The images are the project's own example (data/try_on_example.png: the person,
the product photo and the trained model's result).

    uv run python tools/render.py --article fashion_tryon --out <site>/assets/img/projects/4-ai-fashion-app/figures
"""

from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"

# the same fourteen points on the flat dress (middle panel) and on the person (left panel), in the example's pixels
GARMENT = np.array([
    (421, 127), (423, 174), (612, 128), (610, 177),  # the sleeves: top and bottom of each
    (463, 125), (566, 125), (509, 138),  # neckline: left, right, centre
    (468, 169), (561, 169), (479, 223), (553, 223),  # bust and waist
    (404, 353), (631, 353), (512, 397),  # hem: left, right, centre
], dtype=float)
PERSON = np.array([
    (110, 154), (115, 202), (219, 154), (218, 205),
    (134, 146), (200, 146), (166, 154),
    (125, 179), (207, 179), (132, 218), (195, 218),
    (108, 307), (220, 307), (164, 315),
], dtype=float)
PANEL = 342  # the example is three panels side by side, each about 342 pixels wide
PRODUCT = (346, 676)  # the product photo's columns, without the seams between the panels


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


@cache
def example():
    from PIL import Image

    return np.asarray(Image.open(DATA / "try_on_example.png").convert("RGB")).astype(float) / 255


@cache
def warp():
    """The flat dress, carried onto the person by a thin-plate spline through the fourteen pairs of points."""
    from scipy.interpolate import RBFInterpolator
    from scipy.ndimage import map_coordinates

    image = example()
    height = image.shape[0]
    person = image[:, :PANEL]
    # backward mapping: for every pixel of the person, where on the product photo to read the colour
    to_garment = RBFInterpolator(PERSON, GARMENT, kernel="thin_plate_spline")
    ys, xs = np.mgrid[0:height, 0:PANEL]
    source = to_garment(np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float))
    sx, sy = source[:, 0].reshape(height, PANEL), source[:, 1].reshape(height, PANEL)
    colour = np.stack([map_coordinates(image[..., k], [sy, sx], order=1, cval=1.0) for k in range(3)], axis=-1)
    # the dress is whatever is not white on the product photo
    inside = (sx > PRODUCT[0]) & (sx < PRODUCT[1]) & (sy >= 0) & (sy < height)
    garment = (colour.min(axis=-1) < 0.93) & inside
    out = person.copy()
    out[garment] = colour[garment]
    return out, garment, to_garment


def chart_warp(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    image = example()
    warped, _, to_garment = warp()
    flat = image[:, PRODUCT[0]:PRODUCT[1]]
    result = image[:, 2 * PANEL - 2:]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.3))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.76, bottom=0.08, wspace=0.04)
    panels = [(flat, "the product photo"), (image[:, :PANEL], "the person"),
              (warped, "thin-plate spline warp"), (result, "the trained model")]
    for ax, (picture, title) in zip(axes, panels, strict=True):
        ax.imshow(picture)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="center", fontsize=10, pad=4)
    # the points, and the grid of the product photo carried along by the spline
    axes[0].plot(GARMENT[:, 0] - PRODUCT[0], GARMENT[:, 1], "o", color=c["dec"], markersize=3.5)
    axes[1].plot(PERSON[:, 0], PERSON[:, 1], "o", color=c["dec"], markersize=3.5)
    lines = np.linspace(110, 400, 9)
    for v in lines:
        axes[0].plot([395 - PRODUCT[0], 640 - PRODUCT[0]], [v, v], color=c["att"], linewidth=0.6, alpha=0.7)
    from_garment_x = np.linspace(395, 640, 9)
    for v in from_garment_x:
        axes[0].plot([v - PRODUCT[0], v - PRODUCT[0]], [110, 400], color=c["att"], linewidth=0.6, alpha=0.7)
    from scipy.interpolate import RBFInterpolator

    to_person = RBFInterpolator(GARMENT, PERSON, kernel="thin_plate_spline")
    t = np.linspace(0, 1, 60)
    for v in lines:
        pts = to_person(np.stack([395 + 245 * t, np.full_like(t, v)], axis=1))
        axes[2].plot(pts[:, 0], pts[:, 1], color=c["att"], linewidth=0.6, alpha=0.8)
    for v in from_garment_x:
        pts = to_person(np.stack([np.full_like(t, v), 110 + 290 * t], axis=1))
        axes[2].plot(pts[:, 0], pts[:, 1], color=c["att"], linewidth=0.6, alpha=0.8)
    axes[2].set_xlim(0, PANEL)
    axes[2].set_ylim(image.shape[0], 0)
    _caption(fig, "Placing a garment by warping it",
             f"{len(PERSON)} matched points (rose) define a thin-plate spline; violet: the photo's grid as it bends",
             c)
    return fig


# ---------------------------------------------------------------- the two routes

ROUTES = [  # role, title, steps (two lines each), what the route is good and bad at
    ("enc", "the image route", [
        ("a person's photo", "+ product photo"), ("read the body", "pose and body parts"),
        ("warp the garment", "onto the body"), ("repaint", "GAN, then diffusion"), ("new photo", ""),
    ], ["+ works from any product photo, looks like a photo", "− guesses what it cannot see; no real sizes"]),
    ("att", "the 3D route", [
        ("a person's photo", "or measurements"), ("fit a body model", "SMPL-X: shape, pose"),
        ("3D garment", "from its pattern"), ("simulate cloth", "drape on that body"),
        ("render", "into the photo"),
    ], ["+ real sizes and fit, any pose and view", "− needs a 3D file of every garment"]),
]


def routes() -> Figure:
    fig = Figure(
        760,
        378,
        alt="Two ways to put a garment on a photo. The image route reads the body from a person's photo, warps "
        "the product photo onto it and repaints the result with a generator, at first a GAN and later a diffusion "
        "model: it works from any product photo but guesses what it cannot see and knows nothing of sizes. The 3D "
        "route fits a body model, SMPL-X, to the photo or to measurements, drapes a 3D garment made from its "
        "sewing pattern on that body with a cloth simulation and renders it into the photo: it knows sizes and fit "
        "but needs a 3D file for every garment",
    )
    fig.text(14, 28, "Two ways to put a garment on a photo", kind="title")
    fig.text(14, 50, "the image route edits pixels; the 3D route rebuilds the body and simulates the fabric",
             kind="note")
    w, gap, h = 132, 18, 52
    for row, (role, title, steps, verdict) in enumerate(ROUTES):
        top = 92 + row * 158
        fig.text(14, top - 10, title, kind="subtitle")
        for k, (first, second) in enumerate(steps):
            x = 14 + k * (w + gap)
            last = k == len(steps) - 1
            fig.box(x, top, w, h, role="data" if k in (0, len(steps) - 1) else role, r=8)
            if second:
                fig.text(x + w / 2, top + 22, first, kind="body", role=role, anchor="middle")
                fig.text(x + w / 2, top + 40, second, kind="small", anchor="middle")
            else:
                fig.text(x + w / 2, top + 31, first, kind="body", role=role, anchor="middle")
            if not last:
                fig.arrow((x + w + 2, top + h / 2), (x + w + gap - 2, top + h / 2), width=1.3)
        for j, line in enumerate(verdict):
            fig.text(14, top + h + 26 + j * 19, line, kind="note")
    return fig


# ---------------------------------------------------------------- how the image route is trained

PAIRS = [  # title, steps (two lines each, the role of the box), a note under the row
    ("training: the photo checks the answer", [
        (("a photo", "person in shirt A"), "data"), (("hide the shirt", "keep face and pose"), "enc"),
        (("+ shop photo", "of shirt A"), "data"), (("generator", "paints it back"), "enc"),
        (("compare", "with the photo"), "data"),
    ], "the answer is known: it is the photo we started from"),
    ("use: nothing to check against", [
        (("a photo", "person in shirt A"), "data"), (("hide the shirt", "keep face and pose"), "enc"),
        (("+ shop photo", "of shirt B"), "dec"), (("generator", "paints B in"), "enc"),
        (("new photo", "person in shirt B"), "data"),
    ], "if the mask lets A's outline through, B comes out shaped like A"),
]


def pairs() -> Figure:
    fig = Figure(
        760,
        326,
        alt="How an image try-on model is trained and how it is used. In training, a photo of a person in shirt A "
        "has the shirt hidden, keeping the face, hair and pose; the generator gets the shop photo of the same shirt "
        "A, paints it back, and the result is compared with the original photo. In use, the same masked photo gets "
        "the shop photo of a different shirt B, and there is no photo to compare with. If the mask lets the "
        "outline of shirt A through, shirt B comes out in the shape of A",
    )
    fig.text(14, 28, "One photo, two jobs", kind="title")
    fig.text(14, 50, "nobody has the same person in the same pose in two shirts, so models learn to repair a photo",
             kind="note")
    w, gap, h = 132, 18, 52
    for row, (title, steps, note) in enumerate(PAIRS):
        top = 92 + row * 132
        fig.text(14, top - 10, title, kind="subtitle")
        for k, ((first, second), role) in enumerate(steps):
            x = 14 + k * (w + gap)
            fig.box(x, top, w, h, role=role, r=8)
            fig.text(x + w / 2, top + 22, first, kind="body", role=role, anchor="middle")
            fig.text(x + w / 2, top + 40, second, kind="small", anchor="middle")
            if k < len(steps) - 1:
                fig.arrow((x + w + 2, top + h / 2), (x + w + gap - 2, top + h / 2), width=1.3)
        fig.text(14, top + h + 26, note, kind="note")
    return fig


# ---------------------------------------------------------------- the 3D route: bodies and measurements

BUILDS = ["thin", "average", "full"]
# where to look for each girth (cm above the floor) and how to choose the height inside that band:
# the bust at the most forward point of the chest, as tailors measure it; the waist at the narrowest girth;
# the hips at the widest
BANDS = {"bust": (100, 125, "forward"), "waist": (95, 112, "narrowest"), "hips": (68, 92, "widest")}


@cache
def bodies():
    data = np.load(DATA / "bodies.npz")
    faces = data["faces"]
    return faces, {name: data[name].astype(float) for name in BUILDS}


@cache
def triangles():
    faces, _ = bodies()
    return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])


def contours(v, y):
    """Where the plane at height y cuts the surface: each crossed triangle gives a segment between two of its
    edges, and segments that share an edge join into closed loops."""
    tris = triangles()
    d = v[:, 1] - y
    above = d[tris] > 0
    links = {}
    for tri in tris[above.any(1) & ~above.all(1)]:
        cut = [tuple(sorted((tri[a], tri[b]))) for a, b in ((0, 1), (1, 2), (2, 0))
               if (d[tri[a]] > 0) != (d[tri[b]] > 0)]
        if len(cut) == 2:
            links.setdefault(cut[0], []).append(cut[1])
            links.setdefault(cut[1], []).append(cut[0])
    seen, loops = set(), []
    for start in links:
        if start in seen:
            continue
        loop, edge, previous = [], start, None
        while edge not in seen:
            seen.add(edge)
            loop.append(edge)
            following = [e for e in links[edge] if e != previous]
            if not following:
                break
            previous, edge = edge, following[0]
        i, j = np.array(loop).T
        t = d[i] / (d[i] - d[j])
        loops.append(v[i] + t[:, None] * (v[j] - v[i]))
    return loops


def _area(p):
    x, z = p[:, 0], p[:, 2]
    return 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))


def torso(v, height):
    """The contour around the body's axis at a height above the floor: arms and legs are separate loops."""
    around = [p for p in contours(v, v[:, 1].min() + height) if p[:, 0].min() < 0 < p[:, 0].max()]
    return max(around, key=_area) if around else None


def tape(loop):
    """What a tape measure reads: the length of the convex hull, the tape bridging every hollow."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(loop[:, [0, 2]])
    return hull.area, loop[hull.vertices]


@cache
def measurements(name):
    """Bust, waist and hips, searched every half centimetre within each band."""
    _, builds = bodies()
    v = builds[name]
    out = {}
    for part, (low, high, rule) in BANDS.items():
        rings = {h: torso(v, h) for h in np.arange(low, high, 0.5)}
        rings = {h: ring for h, ring in rings.items() if ring is not None}
        if rule == "forward":
            height = max(rings, key=lambda h: rings[h][:, 2].max())
        else:
            girths = {h: tape(ring)[0] for h, ring in rings.items()}
            height = (min if rule == "narrowest" else max)(girths, key=girths.get)
        length, ring = tape(rings[height])
        out[part] = (float(height), float(length), ring)
    return out


def _view(points, yaw=-28, pitch=9):
    a, b = np.radians(yaw), np.radians(pitch)
    rot_y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return points @ (rot_x @ rot_y).T  # x right, y up, z towards the viewer


def draw_mesh(ax, vertices, faces, base_rgb, offset=0.0, light=(-0.4, 0.5, 0.75)):
    """A tiny renderer: orthographic view, faces sorted back to front, each shaded by its normal."""
    from matplotlib.collections import PolyCollection

    v = _view(vertices)
    quads = v[faces]
    normals = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array(light) / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normals @ light, 0, 1)
    order = np.argsort(quads[..., 2].mean(1))
    colours = np.clip(np.array(base_rgb)[None, :] * shade[order, None], 0, 1)
    polygons = quads[order][..., :2] + np.array([offset, 0.0])
    ax.add_collection(PolyCollection(polygons, facecolors=colours, edgecolors=colours, linewidths=0.25))
    return v


def chart_bodies(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    c = _style(theme)
    faces, builds = bodies()
    skin = to_rgb("#D9DDE3") if theme == "light" else to_rgb("#AEB6C2")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.84, bottom=0.0)
    spacing = 120
    for k, name in enumerate(BUILDS):
        offset = k * spacing
        draw_mesh(ax, builds[name], faces, skin, offset)
        floor = builds[name][:, 1].min()
        for part, (height, length, ring) in measurements(name).items():
            ring = np.vstack([ring, ring[:1]])
            seen = _view(ring)
            centre = seen[:, 2].mean()
            front = seen[:, 2] >= centre
            for mask, style in ((~front, ":"), (front, "-")):
                pts = np.where(mask[:, None], seen[:, :2], np.nan)
                ax.plot(pts[:, 0] + offset, pts[:, 1], style, color=c["dec"], linewidth=1.6)
            label_y = _view(np.array([[0, floor + height, 0]]))[0, 1]
            ax.text(offset + 34, label_y, f"{part} {length:.0f}", fontsize=8.5, color=c["ink"], va="center",
                    fontfamily=["IBM Plex Mono", "DejaVu Sans Mono"])
        ax.text(offset, _view(np.array([[0, floor, 0]]))[0, 1] - 6, name, ha="center", va="top", fontsize=10,
                color=c["muted"])
    ax.set_aspect("equal")
    ax.set_xlim(-62, 2 * spacing + 80)
    ax.set_ylim(-100, 85)
    ax.axis("off")
    _caption(fig, "Measuring a body that only exists as a mesh",
             "MakeHuman's open body at three builds; rose: horizontal cuts, read like a tape measure (cm)", c)
    return fig


# ---------------------------------------------------------------- the 3D route: cloth

FABRICS = {"silk": 12.0, "wool": 0.0}  # compliance of the bending constraints: silk bends freely, wool resists
DRAPES = [("thin", "silk"), ("thin", "wool"), ("full", "silk"), ("full", "wool")]


def vertex_normals(v, faces):
    n = np.cross(v[faces[:, 2]] - v[faces[:, 0]], v[faces[:, 3]] - v[faces[:, 1]])
    out = np.zeros_like(v)
    for k in range(4):
        np.add.at(out, faces[:, k], n)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


@cache
def drape(build, fabric, around=60, rings=27, length=55.0, flare=22.0, seconds=1.5, substeps=24,
          thickness=1.2, friction=0.5):
    """An A-line skirt pinned along this body's waist and left to fall, simulated with XPBD (Macklin et al. 2016).

    Particles carry the cloth; distance constraints hold it together: along and across the weave (stiff), on the
    diagonals (shear, a little compliant) and between particles two apart (bending, compliance set by the fabric).
    Each substep moves the particles under gravity, relaxes the constraints once, pushes particles out of the body
    (the nearest vertex and its normal give a signed distance) and damps their sliding along it (friction)."""
    from scipy.spatial import cKDTree

    faces, builds = bodies()
    body = builds[build]
    normals = vertex_normals(body, faces)
    tree = cKDTree(body)
    height, _, _ = measurements(build)["waist"]
    waist = torso(body, height)
    cx, cz = waist[:, 0].mean(), waist[:, 2].mean()
    angle = np.arctan2(waist[:, 2] - cz, waist[:, 0] - cx)
    order = np.argsort(angle)
    theta = np.linspace(-np.pi, np.pi, around, endpoint=False)
    radius0 = np.interp(theta, angle[order], np.hypot(waist[:, 0] - cx, waist[:, 2] - cz)[order], period=2 * np.pi)
    t = np.linspace(0, 1, rings)[:, None]
    radius = radius0[None, :] + thickness + t * flare  # the cut: a cone from the waist down
    p = np.stack([cx + radius * np.cos(theta), np.broadcast_to(body[:, 1].min() + height - t * length, radius.shape),
                  cz + radius * np.sin(theta)], axis=-1).reshape(-1, 3)
    idx = np.arange(rings * around).reshape(rings, around)
    right, right2 = np.roll(idx, -1, axis=1), np.roll(idx, -2, axis=1)
    groups = [((idx, right), 0.0), ((idx[:-1], idx[1:]), 0.0),  # along and across
              ((idx[:-1], right[1:]), 1e-3), ((idx[1:], right[:-1]), 1e-3),  # shear
              ((idx, right2), FABRICS[fabric]), ((idx[:-2], idx[2:]), FABRICS[fabric])]  # bending
    pairs = np.concatenate([np.stack([a.ravel(), b.ravel()], 1) for (a, b), _ in groups])
    compliance = np.concatenate([np.full(a.size, value) for (a, _), value in groups])
    rest = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
    inverse_mass = np.ones(len(p))
    inverse_mass[idx[0]] = 0.0  # the waistband is pinned
    shared = np.maximum(np.bincount(pairs.ravel(), minlength=len(p)), 1)[:, None]
    velocity = np.zeros_like(p)
    gravity = np.array([0.0, -981.0, 0.0])  # cm/s^2
    h = 1 / 60 / substeps
    for _ in range(int(seconds * 60) * substeps):
        previous = p.copy()
        velocity = velocity + h * gravity * inverse_mass[:, None]
        p = p + h * velocity
        d = p[pairs[:, 0]] - p[pairs[:, 1]]
        length_now = np.linalg.norm(d, axis=1) + 1e-9
        weights = inverse_mass[pairs[:, 0]] + inverse_mass[pairs[:, 1]]
        step = -(length_now - rest) / (weights + compliance / h ** 2 + 1e-12)
        correction = (step / length_now)[:, None] * d
        delta = np.zeros_like(p)
        np.add.at(delta, pairs[:, 0], correction * inverse_mass[pairs[:, 0], None])
        np.add.at(delta, pairs[:, 1], -correction * inverse_mass[pairs[:, 1], None])
        p = p + 1.5 * delta / shared  # averaged (Jacobi) corrections, over-relaxed
        _, nearest = tree.query(p)
        n = normals[nearest]
        depth = np.einsum("ij,ij->i", p - body[nearest], n)
        hit = (depth < thickness) & (inverse_mass > 0)
        p[hit] += (thickness - depth[hit])[:, None] * n[hit]
        slide = p[hit] - previous[hit]
        slide -= np.einsum("ij,ij->i", slide, n[hit])[:, None] * n[hit]
        p[hit] -= friction * slide
        velocity = (p - previous) / h
    quads = np.stack([idx[:-1], idx[1:], right[1:], right[:-1]], axis=-1).reshape(-1, 4)
    return p, quads


def _shaded(quads, rgb, light=(-0.4, 0.5, 0.75)):
    normals = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array(light) / np.linalg.norm(light)
    return np.clip(np.array(rgb)[None, :] * (0.35 + 0.65 * np.abs(normals @ light))[:, None], 0, 1)


def chart_drape(theme):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    c = _style(theme)
    faces, builds = bodies()
    skin = to_rgb("#D9DDE3") if theme == "light" else to_rgb("#AEB6C2")
    colour = {"silk": to_rgb(c["dec"]), "wool": to_rgb(c["att"])}
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.82, bottom=0.02, wspace=0.0)
    for ax, (build, fabric) in zip(axes, DRAPES, strict=True):
        cloth, cloth_quads = drape(build, fabric)
        quads = np.concatenate([_view(builds[build])[faces], _view(cloth)[cloth_quads]])
        colours = np.concatenate([_shaded(_view(builds[build])[faces], skin),
                                  _shaded(_view(cloth)[cloth_quads], colour[fabric])])
        depth = quads[..., 2].mean(1)
        depth[len(faces):] += 0.6  # where cloth and skin nearly touch, draw the cloth on top
        order = np.argsort(depth)
        ax.add_collection(PolyCollection(quads[order][..., :2], facecolors=colours[order], edgecolors=colours[order],
                                         linewidths=0.2))
        ax.set_xlim(-52, 52)
        ax.set_ylim(-84, 84)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{build} body, {fabric}", loc="center", fontsize=10, pad=2)
    _caption(fig, "One skirt, two bodies, two fabrics",
             "an A-line skirt pinned at each body's waist and dropped; XPBD cloth, 1.5 s of simulated time", c)
    return fig


def chart_shape(theme):
    """How far each point of the body moves between the thin and the full build: one 'shape direction'."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    c = _style(theme)
    faces, builds = bodies()
    moved = np.linalg.norm(builds["full"] - builds["thin"], axis=1)
    cmap = LinearSegmentedColormap.from_list("moved", [c["card_alt"], c["data_tint"], c["data"], c["dec"]])
    norm = Normalize(0, float(np.percentile(moved, 99.5)))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.0, right=0.86, top=0.82, bottom=0.02, wspace=0.0)
    for ax, (yaw, title) in zip(axes, [(-20, "front"), (-90, "side")], strict=True):
        v = _view(builds["average"], yaw=yaw, pitch=6)
        quads = v[faces]
        shade = _shaded(quads, (1.0, 1.0, 1.0))[:, 0]
        colours = cmap(norm(moved[faces].mean(1)))[:, :3] * (0.55 + 0.45 * shade[:, None])
        order = np.argsort(quads[..., 2].mean(1))
        ax.add_collection(PolyCollection(quads[order][..., :2], facecolors=colours[order], edgecolors=colours[order],
                                         linewidths=0.2))
        ax.set_xlim(-55, 55)
        ax.set_ylim(-84, 82)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="center", fontsize=10, pad=2)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("moves by, cm", color=c["muted"])
    bar.outline.set_visible(False)
    _caption(fig, "Where a body changes when it changes build",
             "distance each vertex moves from the thin to the full body: one direction in the space of body shapes",
             c)
    return fig


# ---------------------------------------------------------------- the field around the project

# (lane, colour role, events); a date is arXiv's first version or the day of the announcement
LANES = [
    ("try-on models", "dec", [
        ("2017-11-22", "VITON"), ("2018-07-20", "CP-VTON"), ("2020-03-12", "ACGPN"), ("2021-03-31", "VITON-HD"),
        ("2022-06-28", "HR-VITON"), ("2023-06-14", "TryOnDiffusion"),
    ]),
    ("image generators", "att", [
        ("2019-12-03", "StyleGAN2"), ("2020-06-19", "DDPM"), ("2021-01-05", "DALL·E"),
        ("2021-12-20", "Latent Diffusion"), ("2022-04-06", "DALL·E 2"), ("2022-08-22", "Stable Diffusion"),
        ("2023-02-10", "ControlNet"),
    ]),
    ("bodies in 3D", "enc", [
        ("2018-02-01", "DensePose"), ("2019-04-11", "SMPL-X"), ("2020-04-01", "PIFuHD"),
        ("2021-06-17", "SMPL-X in Unity"), ("2021-12-16", "ICON"), ("2022-06-14", "SHAPY"), ("2022-12-14", "ECON"),
    ]),
    ("digital fashion", "data", [
        ("2019-05-27", "a digital dress, $9,500"), ("2020-08-01", "DressX"), ("2021-05-25", "Gucci bag on Roblox"),
        ("2021-10-28", "Facebook → Meta"), ("2022-03-24", "Metaverse Fashion Week"),
        ("2023-06-14", "try-on in Google Search"),
    ]),
]
PROJECT = ("2021-01-01", "2023-06-30")


def _year(date):
    from datetime import date as day

    d = day.fromisoformat(date)
    return d.year + (d.timetuple().tm_yday - 1) / (366 if d.year % 4 == 0 else 365)


def chart_timeline(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.07)
    start, end = 2017.72, 2023.95
    ax.set_xlim(start, end)
    ax.set_ylim(-0.75, len(LANES) - 0.05)
    ax.axvspan(_year(PROJECT[0]), _year(PROJECT[1]), color=c["grid"], alpha=0.55 if theme == "light" else 0.8,
               linewidth=0, zorder=0)
    ax.text((_year(PROJECT[0]) + _year(PROJECT[1])) / 2, len(LANES) - 0.22, "the project", ha="center",
            va="center", fontsize=9.5, color=c["muted"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for row, (lane, role, events) in enumerate(LANES):
        y = len(LANES) - 1 - row
        ax.plot([start, end], [y, y], color=c["line"], linewidth=0.8, zorder=1)
        title = ax.text(start + 0.03, y + 0.4, lane, ha="left", va="center", fontsize=9, color=c[f"{role}_ink"],
                        fontweight="semibold")
        placed.append(title.get_window_extent(renderer).expanded(1.04, 1.1))
        for date, name in events:
            x = _year(date)
            ax.plot([x], [y], "o", markersize=6, color=c[role], markeredgecolor=c["card"], markeredgewidth=1.5,
                    zorder=3)
            for level in (-0.2, 0.2, -0.4, 0.4):  # below the line first, then above, then further out
                label = ax.text(x, y + level, name, ha="center", va="center", fontsize=8.5, color=c["ink"])
                box = label.get_window_extent(renderer).expanded(1.06, 1.15)
                if not any(box.overlaps(other) for other in placed):
                    placed.append(box)
                    if abs(level) > 0.3:  # a label two rows out gets a thin line to its dot
                        sign = 1 if level > 0 else -1
                        ax.plot([x, x], [y + sign * 0.07, y + level - sign * 0.09], color=c["line"],
                                linewidth=0.7, zorder=2)
                    break
                label.remove()
            else:
                raise ValueError(f"no room for the label {name!r}")
    ax.set_yticks([])
    ax.set_xticks(range(2018, 2024))
    ax.grid(axis="x", color=c["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _caption(fig, "The field around the project",
             "dates of arXiv first versions and announcements; shaded: January 2021 to June 2023", c, top=0.975)
    return fig


# ---------------------------------------------------------------- cover and thumbnail

CROPS = {  # (left, top, right, bottom) in the example: the person before, the product photo, the person after
    "before": (30, 40, 330, 512),
    "product": (380, 92, 660, 415),
    "after": (712, 40, 1012, 512),
}


@cache
def crop_uri(name):
    import base64
    import io

    from PIL import Image

    image = Image.open(DATA / "try_on_example.png").convert("RGB").crop(CROPS[name])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _photo(name, x, y, width):
    left, top, right, bottom = CROPS[name]
    height = width * (bottom - top) / (right - left)
    clip = f"clip-{name}-{int(x)}"
    svg = (f'<clipPath id="{clip}"><rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10"/>'
           f'</clipPath><image href="{crop_uri(name)}" x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
           f'height="{height:.1f}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip})"/>'
           f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" fill="none" '
           f'stroke="{cv.LINE}" stroke-width="1.5"/>')
    return svg, height


def _equation(left, middle_y, widths, gap, labels=True):
    """your photo + a product -> the try-on, centred on a horizontal line; returns the SVG and the right edge."""
    names = ("before", "product", "after")
    heights = [_photo(name, 0, 0, width)[1] for name, width in zip(names, widths, strict=True)]
    label_y = middle_y - max(heights) / 2 - 14
    out, x = [], left
    for k, (name, width, height) in enumerate(zip(names, widths, heights, strict=True)):
        svg, _ = _photo(name, x, middle_y - height / 2, width)
        out.append(svg)
        if labels:
            out.append(cv.label(x, label_y, ("YOUR PHOTO", "A PRODUCT", "THE TRY-ON")[k]))
        x += width
        if k == 0:
            out.append(cv.text(x + gap / 2, middle_y + 10, "+", 30, cv.MUTED, cv.SANS, 400, "middle"))
        elif k == 1:
            a, b, y = x + gap * 0.22, x + gap * 0.78, middle_y
            out.append(f'<path d="M{a:.1f} {y:.1f} H{b:.1f} M{b - 8:.1f} {y - 7:.1f} L{b:.1f} {y:.1f} '
                       f'L{b - 8:.1f} {y + 7:.1f}" stroke="{cv.RED}" stroke-width="3" fill="none" '
                       f'stroke-linecap="round" stroke-linejoin="round"/>')
        x += gap
    return "".join(out), x - gap


def cover():
    motif, right = _equation(cv.MOTIF_X0, 330, (112, 112, 184), 52)
    cv.fits(right)
    chips, _, chips_right = cv.chip_row(["DensePose", "Stable Diffusion", "SMPL-X"], cv.MOTIF_X0, 498, size=15)
    cv.fits(chips_right)
    return cv.frame("portfolio / computer vision", ["AI Fashion", "App"],
                    ["Digital clothes on real", "people, 2021 to 2023"], motif + chips,
                    footer="Virtual try-on, in 2D and in 3D")


def card(width=640, height=440):
    """The cover's picture alone, in the proportions of a portfolio card."""
    widths, gap = (138, 138, 222), 44
    motif, _ = _equation((width - sum(widths) - 2 * gap) / 2, height / 2, widths, gap, labels=False)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern></defs>'
            f'<rect width="{width}" height="{height}" fill="{cv.BG}"/>'
            f'<rect width="{width}" height="{height}" fill="url(#dots)"/>{motif}</svg>')


COVER = cover
IMAGES = {"card": (card, 640, 440)}


CHARTS = {
    "warp": chart_warp,
    "bodies": chart_bodies,
    "drape": chart_drape,
    "shape": chart_shape,
    "timeline": chart_timeline,
}

FIGURES = {
    "routes": routes,
    "pairs": pairs,
}


def most_moved():
    _, builds = bodies()
    return float(np.percentile(np.linalg.norm(builds["full"] - builds["thin"], axis=1), 99.5))


def chart_alts():
    girths = {name: {part: round(length) for part, (_, length, _) in measurements(name).items()} for name in BUILDS}
    return {
        "bodies": "Three versions of the same open 3D body, thin, average and full, with tape-measure loops drawn "
        "around the bust, waist and hips: " + "; ".join(
            f"{name}: bust {g['bust']} cm, waist {g['waist']} cm, hips {g['hips']} cm" for name, g in girths.items()),
        "shape": "The average body seen from the front and the side, coloured by how far each point moves between "
        f"the thin and the full build, up to about {most_moved():.1f} cm: "
        "most around the belly, hips, buttocks and thighs, a little on the arms, almost nothing on the head, hands "
        "and feet",
        "drape": "The same A-line skirt simulated on the thin and the full body in two fabrics. Silk, which bends "
        "freely, falls in narrow folds close to the legs; wool, which resists bending, keeps a wide A-line flare. "
        "On the full body the hips push the skirt outward",
        "warp": f"Four panels. The product photo of a floral dress with {len(PERSON)} points marked on its sleeves, "
        "neckline, waist and hem; the person in a white dress with the same points marked on her body; the floral "
        "dress bent onto her by a thin-plate spline, with the old white sleeves and skirt still showing around it "
        "and the fabric as flat as on the product photo; and the project's trained model, which replaces the old "
        "dress, shapes the puffed sleeves to the arms and matches the light of the scene",
        "timeline": "A timeline from late 2017 to mid-2023 in four lanes, with the project, January 2021 to June "
        "2023, shaded. " + " ".join(
            f"{lane[0].upper() + lane[1:]}: "
            + ", ".join(f"{_spoken(name)} ({_month(date)})" for date, name in events) + "."
            for lane, _, events in LANES),
    }


def _spoken(name):
    """A label as alt text: '$9,500' as '9,500 dollars', since a dollar sign in a post starts math."""
    import re

    return re.sub(r"\$([\d,]+)", r"\1 dollars", name)


def _month(date):
    from datetime import date as day

    return day.fromisoformat(date).strftime("%b %Y")
