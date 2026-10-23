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
    _caption(fig, "A thin-plate spline warp of a product photo",
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
    ], ["+ real sizes and fit; garments that exist only as 3D models", "− needs a 3D model of every garment"]),
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
    fig.text(14, 28, "Image-based and 3D try-on", kind="title")
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


# ---------------------------------------------------------------- reading the pose: BlazePose keypoints (pose.py)

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (15, 17), (15, 19), (17, 19),
         (16, 18), (16, 20), (18, 20), (9, 10)]
# the joint each limb point hangs from; a bone can point towards the camera or away from it and look the same
PARENT = {13: 11, 15: 13, 17: 15, 19: 15, 21: 15, 14: 12, 16: 14, 18: 16, 20: 16, 22: 16,
          25: 23, 27: 25, 29: 27, 31: 27, 26: 24, 28: 26, 30: 28, 32: 28}
HIDDEN = 0.8  # below this visibility BlazePose doubts the point is in sight
BONE_NAMES = {13: "left upper arm", 14: "right upper arm", 15: "left forearm", 16: "right forearm",
              25: "left thigh", 26: "right thigh", 27: "left shin", 28: "right shin"}


@cache
def keypoints():
    import json

    data = json.loads((DATA / "keypoints.json").read_text())
    return {name: {k: np.array(v, dtype=float) for k, v in item.items()} for name, item in data.items()}


@cache
def crouch():
    from PIL import Image

    return np.asarray(Image.open(DATA / "crouch.png").convert("RGB")).astype(float) / 255


MAIN = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # nose, shoulders, elbows, wrists, hips, knees, ankles


def _skeleton(ax, points, c, width=1.6, style="-", colours=None, size=3.2, hollow=(), bones=BONES, marks=MAIN,
              hidden_fill=None):
    """Bones of the left side (odd BlazePose numbers) teal, of the right side rose, the rest ink."""
    for a, b in bones:
        side = "enc" if a % 2 and b % 2 else "dec" if not a % 2 and not b % 2 else "ink"
        colour = (colours or {}).get(side, c[side])
        ax.plot(points[[a, b], 0], points[[a, b], 1], style, color=colour, linewidth=width, solid_capstyle="round")
    if not size:
        return
    shown = [k for k in marks if k not in hollow]
    hidden = [k for k in marks if k in hollow]
    ax.plot(points[shown, 0], points[shown, 1], "o", color=c["card"], markeredgecolor=c["ink"],
            markersize=size, markeredgewidth=0.8, linestyle="none")
    if hidden and hidden_fill:  # points the model is unsure about, filled with a colour of their own
        ax.plot(points[hidden, 0], points[hidden, 1], "o", color=hidden_fill, markeredgecolor=c["card"],
                markersize=size + 2.5, markeredgewidth=1.0, linestyle="none")
    elif hidden:
        ax.plot(points[hidden, 0], points[hidden, 1], "o", color="none", markeredgecolor=c["muted"],
                markersize=size + 1.5, markeredgewidth=1.0, linestyle="none")


def _subtree(joint):
    out = [joint]
    for child, parent in PARENT.items():
        if parent == joint:
            out += _subtree(child)
    return out


def flipped(world, joint):
    """The same keypoints with one bone turned to the other side of the camera's line of sight: the bone ending at
    `joint` keeps its length and its picture, and everything hanging from it moves along."""
    out = world.copy()
    out[_subtree(joint), 2] += -2 * (world[joint, 2] - world[PARENT[joint], 2])
    return out


def flips(world, count=3):
    """The bones whose flip changes the pose the most."""
    moved = {j: np.abs(flipped(world, j)[:, 2] - world[:, 2]).sum() for j in (13, 14, 15, 16, 25, 26, 27, 28)}
    return sorted(moved, key=moved.get, reverse=True)[:count]


def chart_pose(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    points = keypoints()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [0.67, 0.78, 1.0]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.03, wspace=0.08)
    standing, crouching = points["standing"], points["crouching"]
    for ax, picture, item, title in ((axes[0], example()[:, :PANEL], standing, "standing: 33 keypoints"),
                                     (axes[1], crouch(), crouching, "crouching: amber = likely hidden")):
        ax.imshow(picture)
        hidden = [k for k in range(33) if item["image"][k, 2] < HIDDEN]
        _skeleton(ax, item["image"][:, :2], c, width=1.4, hollow=hidden, hidden_fill=c["data"])
        ax.set_title(title, loc="center", fontsize=10, pad=4)
        ax.axis("off")
    world = crouching["world"].copy()
    limbs = [bone for bone in BONES if bone[0] in MAIN and bone[1] in MAIN] + [(0, 11), (0, 12)]
    ax = axes[2]
    turned = flips(world)
    for joint in turned:
        _skeleton(ax, flipped(world, joint)[:, [2, 1]], c, width=1.2, style="--", size=0, bones=limbs,
                  colours={"enc": c["att"], "dec": c["att"], "ink": c["att"]})
    _skeleton(ax, world[:, [2, 1]], c, width=2.0, size=3.4, bones=limbs)
    ax.text(0.5, -0.02, "turned: " + ", ".join(BONE_NAMES[j] for j in turned), transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=c["att_ink"])
    ax.set_xlim(-0.62, 0.48)
    ax.set_ylim(0.45, -0.62)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("camera", xy=(-0.6, -0.56), xytext=(-0.36, -0.56), fontsize=8.5, color=c["muted"], va="center",
                arrowprops={"arrowstyle": "->", "color": c["muted"], "linewidth": 0.8})
    ax.set_title("the crouch in 3D, from the side", loc="center", fontsize=10, pad=4)
    for each in axes:
        each.set_anchor("N")
    _caption(fig, "Keypoints in the photo and the 3D pose behind them",
             "BlazePose on two photos from the project; violet: the same photo with one bone turned the other way",
             c)
    return fig


# ---------------------------------------------------------------- photos that do not work: crops

FRAMES = [  # name, (left, top, right, bottom) in the person's panel
    ("full length", (0, 0, 342, 512)),
    ("cut at the thighs", (0, 0, 342, 330)),
    ("waist up", (40, 40, 300, 250)),
    ("a selfie", (95, 70, 245, 190)),
]


def in_frame(name):
    left, top, right, bottom = dict(FRAMES)[name]
    points = keypoints()["standing"]["image"]
    return (points[:, 0] >= left) & (points[:, 0] < right) & (points[:, 1] >= top) & (points[:, 1] < bottom)


def chart_crops(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    c = _style(theme)
    person = example()[:, :PANEL]
    points = keypoints()["standing"]["image"][:, :2]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.1, wspace=0.05)
    for ax, (name, (left, top, right, bottom)) in zip(axes, FRAMES, strict=True):
        dimmed = person * 0.3 + (0.7 if theme == "light" else 0.08)
        dimmed[top:bottom, left:right] = person[top:bottom, left:right]
        ax.imshow(dimmed)
        ax.add_patch(Rectangle((left, top), right - left - 1, bottom - top - 1, fill=False, edgecolor=c["dec"],
                               linewidth=1.6))
        inside = in_frame(name)
        _skeleton(ax, points, c, width=1.0, size=2.4, hollow=[k for k in range(33) if not inside[k]])
        ax.set_title(name, loc="center", fontsize=10, pad=4)
        ax.text(0.5, -0.04, f"{inside.sum()} of 33 keypoints in frame", transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color=c["muted"])
        ax.axis("off")
    _caption(fig, "How much of the body a crop leaves",
             "BlazePose keypoints of the example; hollow: outside the frame, so a model has to guess them", c)
    return fig


# ---------------------------------------------------------------- inpainting: masks and the latent grid

@cache
def dress_masks():
    """The old dress (white pixels near the outline of the fourteen points) and a clothing-agnostic mask: the
    outline grown by 18 px, plus both arms from shoulder to wrist, as VITON-HD masks them."""
    from matplotlib.path import Path as Outline
    from scipy.ndimage import binary_closing, binary_dilation, binary_opening

    person = example()[:, :PANEL]
    height, width = person.shape[:2]
    order = [0, 4, 6, 5, 2, 3, 10, 12, 13, 11, 9, 1]  # around the dress through the fourteen points
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    inside = Outline(PERSON[order]).contains_points(pixels).reshape(height, width)
    white = person.min(axis=-1) > 0.72
    tight = binary_opening(binary_closing(binary_dilation(inside, iterations=10) & white, iterations=2))
    agnostic = binary_dilation(inside, iterations=18)
    points = keypoints()["standing"]["image"][:, :2]
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        p, q = points[a], points[b]
        t = np.clip(((pixels - p) @ (q - p)) / ((q - p) @ (q - p)), 0, 1)
        agnostic |= (np.linalg.norm(pixels - (p + t[:, None] * (q - p)), axis=1) < 15).reshape(height, width)
    return tight, agnostic


def chart_masks(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    c = _style(theme)
    person = example()[:, :PANEL]
    tight, agnostic = dress_masks()
    rose, grey = np.array(to_rgb(c["dec"])), np.array(to_rgb(c["line"]))
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.03, wspace=0.05)
    panels = []
    for mask, title in ((tight, "mask of the old dress"), (agnostic, "clothing-agnostic mask")):
        shown = person.copy()
        shown[mask] = 0.45 * shown[mask] + 0.55 * rose
        panels.append((shown, title))
    shown = person.copy()
    shown[agnostic] = grey
    panels.append((shown, "what the generator keeps"))
    for ax, (picture, title) in zip(axes[:3], panels, strict=True):
        ax.imshow(picture)
        ax.set_title(title, loc="center", fontsize=10, pad=4)
        ax.axis("off")
    _skeleton(axes[2], keypoints()["standing"]["image"][:, :2], c, width=1.0, size=2.2)
    # the new dress at the scale a 512-pixel Stable Diffusion sees it: a latent cell covers 8 by 8 pixels
    result = example()[:, 2 * PANEL - 2:]
    left, top, size = 120, 170, 80
    ax = axes[3]
    ax.imshow(result[top:top + size, left:left + size], extent=(0, size, size, 0), interpolation="nearest")
    for v in range(0, size + 1, 8):
        ax.axhline(v, color=c["card"], linewidth=0.6, alpha=0.9)
        ax.axvline(v, color=c["card"], linewidth=0.6, alpha=0.9)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_title("the print, 8×8 px per latent", loc="center", fontsize=10, pad=4)
    ax.axis("off")
    _caption(fig, "What an inpainting model is given",
             "rose: the region it repaints; right: the new dress under the grid of the autoencoder's latents",
             c)
    return fig


# ---------------------------------------------------------------- combining the routes: what 3D hands to 2D

def chart_maps(theme):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    c = _style(theme)
    faces, builds = bodies()
    body = builds["full"]
    cloth, cloth_quads = drape("full", "silk")
    quads = np.concatenate([_view(body)[faces], _view(cloth)[cloth_quads]])
    is_cloth = np.r_[np.zeros(len(faces), bool), np.ones(len(cloth_quads), bool)]
    depth = quads[..., 2].mean(1)
    depth[is_cloth] += 0.6
    order = np.argsort(depth)
    normals = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals *= np.sign(normals[:, 2:3] + 1e-9)  # every face turned towards the viewer
    skin = to_rgb("#D9DDE3") if theme == "light" else to_rgb("#AEB6C2")
    shaded = np.where(is_cloth[:, None], _shaded(quads, to_rgb(c["dec"])), _shaded(quads, skin))
    near, far = np.percentile(depth, [2, 98])
    grey = (1 - np.clip((depth - near) / (far - near), 0, 1))[:, None] * np.ones(3)
    maps = [  # what each map shows, and the colour behind it
        (shaded, "rendered garment", None),
        (np.where(is_cloth[:, None], 1.0, 0.0) * np.ones(3), "garment mask", "black"),
        ((normals + 1) / 2, "normal map", "#8080FF"),
        (grey, "depth map", "black"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.8, bottom=0.02, wspace=0.02)
    for ax, (colours, title, background) in zip(axes, maps, strict=True):
        if background:
            ax.add_patch(plt.Rectangle((-60, -90), 120, 180, color=background, zorder=0))
        ax.add_collection(PolyCollection(quads[order][..., :2], facecolors=colours[order],
                                         edgecolors=colours[order], linewidths=0.2))
        ax.set_xlim(-48, 48)
        ax.set_ylim(-84, 84)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="center", fontsize=10, pad=2)
    _caption(fig, "What a 3D render can pass to an image generator",
             "the silk skirt on the full body, drawn four ways; ControlNet had models for depth and normal maps", c)
    return fig


HYBRID = [  # row title, colour role, steps
    ("3D: geometry and fit", "att", [
        ("a person's photo", "+ the garment in 3D"), ("fit a body", "SMPL-X: shape, pose"),
        ("drape the garment", "cloth simulation"), ("render maps", "mask, depth, normals"),
    ]),
    ("2D: the photograph", "enc", [
        ("a person's photo", ""), ("mask the clothes", "keep face, hands"), ("inpaint", "conditioned on maps"),
        ("new photo", ""),
    ]),
]


def hybrid() -> Figure:
    fig = Figure(
        760,
        312,
        alt="An experimental pipeline that combines the two routes. In the 3D part, a body model, SMPL-X, is fitted "
        "to the person's photo, the 3D garment is draped on it by a cloth simulation, and the result is rendered "
        "as a garment mask, a depth map and a normal map. In the 2D part, the clothes in the photo are masked, "
        "keeping the face and hands, and a diffusion model inpaints the region conditioned on those maps, giving "
        "the new photo",
    )
    fig.text(14, 28, "Combining 3D and image generation", kind="title")
    fig.text(14, 50, "the 3D part gives the garment's shape and fit, the image model the light and texture of a photo",
             kind="note")
    w, gap, h = 160, 32, 52
    tops = [96, 236]
    for row, (title, role, steps) in enumerate(HYBRID):
        top = tops[row]
        fig.text(14, top - 10, title, kind="subtitle")
        for k, (first, second) in enumerate(steps):
            x = 14 + k * (w + gap)
            ends = (row == 0 and k == 0) or (row == 1 and k in (0, 3))
            fig.box(x, top, w, h, role="data" if ends else role, r=8)
            if second:
                fig.text(x + w / 2, top + 22, first, kind="body", role=role, anchor="middle")
                fig.text(x + w / 2, top + 40, second, kind="small", anchor="middle")
            else:
                fig.text(x + w / 2, top + 31, first, kind="body", role=role, anchor="middle")
            if k < len(steps) - 1:
                fig.arrow((x + w + 3, top + h / 2), (x + w + gap - 3, top + h / 2), width=1.3)
    # the render's maps go down into the inpainting step
    x_maps = 14 + 3 * (w + gap) + w / 2
    x_inpaint = 14 + 2 * (w + gap) + w
    fig.arrow((x_maps, tops[0] + h + 3), (x_inpaint - 12, tops[1] - 4), width=1.3)
    return fig


# ---------------------------------------------------------------- the system around the models

SYSTEM_LANES = ["on the phone", "on the server", "on GPU workers"]
SYSTEM_BOXES = {  # (lane, column): (title, detail, colour role)
    (0, 0): ("take a photo", "or pick one", "data"),
    (0, 1): ("check it", "a small pose model", "enc"),
    (0, 2): ("upload", "a resized photo", "data"),
    (0, 4): ("show the result", "push or poll", "data"),
    (1, 2): ("API", "accounts, limits", "enc"),
    (1, 3): ("job queue", "one job per pair", "enc"),
    (1, 4): ("results", "storage and CDN", "enc"),
    (2, 2): ("body analysis", "once per photo", "att"),
    (2, 3): ("generation", "once per pair", "att"),
    (2, 4): ("garment prep", "once per product", "att"),
}


def system() -> Figure:
    fig = Figure(
        760,
        420,
        alt="The system around the models, in three lanes. On the phone: take or pick a photo, check it with a small "
        "pose model, upload it resized and compressed, and later show the result by push notification or polling. "
        "On the server: an API with accounts and limits, a job queue with one job per pair of photo and garment, "
        "and results in storage behind a CDN. On GPU workers: body analysis once per photo, garment preparation "
        "once per product, and generation once per pair, which uses both and writes to the results",
    )
    fig.text(14, 28, "The system around the models", kind="title")
    fig.text(14, 50, "work is cached by what it depends on: the photo, the garment, or the pair", kind="note")
    w, gap, h = 132, 18, 52
    lane_top = {0: 96, 1: 214, 2: 332}

    def at(lane, col):
        return 14 + col * (w + gap), lane_top[lane]

    for lane, title in enumerate(SYSTEM_LANES):
        fig.text(14, lane_top[lane] - 10, title, kind="subtitle")
    for (lane, col), (first, second, role) in SYSTEM_BOXES.items():
        x, y = at(lane, col)
        fig.box(x, y, w, h, role=role, r=8)
        fig.text(x + w / 2, y + 22, first, kind="body", role=role, anchor="middle")
        fig.text(x + w / 2, y + 40, second, kind="small", anchor="middle")

    def across(lane, a, b):  # from column a to column b in one lane
        (xa, y), (xb, _) = at(lane, a), at(lane, b)
        if b > a:
            fig.arrow((xa + w + 3, y + h / 2), (xb - 3, y + h / 2), width=1.3)
        else:
            fig.arrow((xa - 3, y + h / 2), (xb + w + 3, y + h / 2), width=1.3)

    def between(col, a, b):  # from lane a to lane b in one column
        x, ya = at(a, col)
        _, yb = at(b, col)
        if b > a:
            fig.arrow((x + w / 2, ya + h + 3), (x + w / 2, yb - 4), width=1.3)
        else:
            fig.arrow((x + w / 2, ya - 3), (x + w / 2, yb + h + 4), width=1.3)

    across(0, 0, 1)
    across(0, 1, 2)
    between(2, 0, 1)
    across(1, 2, 3)
    between(2, 1, 2)
    between(3, 1, 2)
    across(2, 2, 3)
    across(2, 4, 3)
    x, y = at(2, 3)
    xr, yr = at(1, 4)
    fig.arrow((x + w - 10, y - 3), (xr + 10, yr + h + 4), width=1.3)
    between(4, 1, 0)
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
    fig.text(14, 28, "How image try-on models are trained and used", kind="title")
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
    _caption(fig, "Bust, waist and hips measured on a mesh",
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
    _caption(fig, "An A-line skirt simulated on two bodies in two fabrics",
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
    _caption(fig, "How far each point moves between a thin and a full body",
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
    _caption(fig, "Try-on, image generation, 3D bodies and digital fashion, 2017–2023",
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
                    ["Digital clothes", "on real people"], motif + chips,
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
    "pose": chart_pose,
    "crops": chart_crops,
    "masks": chart_masks,
    "maps": chart_maps,
}

FIGURES = {
    "routes": routes,
    "pairs": pairs,
    "hybrid": hybrid,
    "system": system,
}


def most_moved():
    _, builds = bodies()
    return float(np.percentile(np.linalg.norm(builds["full"] - builds["thin"], axis=1), 99.5))


JOINT_NAMES = {0: "nose", 11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
               15: "left wrist", 16: "right wrist", 23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee",
               27: "left ankle", 28: "right ankle"}


def _either(names):
    return ", the ".join(names[:-1]) + " or the " + names[-1] if len(names) > 1 else names[0]


def chart_alts():
    crouching = keypoints()["crouching"]
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
        "pose": "Three panels. The standing person of the try-on example with BlazePose's 33 keypoints joined into a "
        "skeleton; a woman in a pink tracksuit crouching, with her skeleton and the "
        + " and the ".join(JOINT_NAMES[k] for k in MAIN if crouching["image"][k, 2] < HIDDEN)
        + " marked as likely hidden; and the crouch's 3D keypoints seen from the side, with dashed alternatives in "
        "which the " + _either([BONE_NAMES[j] for j in flips(crouching["world"])])
        + " is turned the other way along the camera's line of sight: each of them projects onto the same photo",
        "crops": "Four crops of the standing example with its keypoints, those outside the frame drawn hollow: "
        + "; ".join(f"{name}, {int(in_frame(name).sum())} of 33 keypoints in frame" for name, _ in FRAMES),
        "masks": "Four panels. The person with her white dress masked tightly, so the mask keeps the dress's outline; "
        "a larger clothing-agnostic mask that also covers both arms; the photo with that region grey and the "
        "skeleton on top, which is what the generator keeps; and a close-up of the generated floral dress under a "
        "grid of 8 by 8 pixel cells, in which a flower of the print spans one or two cells",
        "maps": "The silk skirt on the full body drawn four ways: shaded, as a white garment mask on black, as a "
        "normal map in which colour encodes the direction of the surface, and as a depth map in which nearer is "
        "lighter",
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
