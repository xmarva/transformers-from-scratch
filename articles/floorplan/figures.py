"""Figures and charts of the portfolio page "3D Floorplan Reconstruction" on xmarva.github.io.

Everything is computed from one real plan (data/plan.png, the project's own example) by plan.py: walls, rooms,
doors, the scale, the 3D model and a furniture layout.

    uv run python tools/render.py --article floorplan \\
        --out <site>/assets/img/projects/3-vr-floorplan-reconstruction/figures
"""

from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure


@cache
def fp():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("floorplan_plan", Path(__file__).parent / "plan.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975, gap=0.075):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - gap, note, ha="left", va="top", fontsize=10, color=c["muted"])


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


ROOM_ROLES = {"living room and hall": "data", "bathroom": "enc", "balcony": "att", "wardrobe": "dec",
              "closet": "dec"}


def room_colours(c, alpha=1.0):
    from matplotlib.colors import to_rgba

    names = fp().name_rooms()
    return {k: to_rgba(c[f"{ROOM_ROLES[n]}_tint"], alpha) for k, n in names.items()}


def _paper(c):
    """The plan as it would print in this theme: ink on the card colour."""
    from matplotlib.colors import to_rgb

    a = fp().image().mean(-1, keepdims=True)
    ink, paper = np.array(to_rgb(c["ink"])), np.array(to_rgb(c["card"]))
    return paper * a + ink * (1 - a)


# ---------------------------------------------------------------- reading the plan

def chart_reading(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle

    c = _style(theme)
    p = fp()
    ink, walls, _ = p.masks()
    labels, sizes = p.rooms()
    names = p.name_rooms()
    est, per = p.scale()
    areas = p.areas()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 4.3))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.8, bottom=0.02, wspace=0.05)
    axes[0].imshow(_paper(c))
    axes[0].set_title("the plan\nas drawn", loc="center", fontsize=9.5, pad=4)
    shown = np.ones(walls.shape + (3,)) * np.array(to_rgb(c["card"]))
    shown[ink & ~walls] = np.array(to_rgb(c["line"]))
    shown[walls] = np.array(to_rgb(c["ink"]))
    axes[1].imshow(shown)
    axes[1].set_title("walls: strokes that survive\nan opening", loc="center", fontsize=9.5, pad=4)
    colours = room_colours(c)
    rooms_rgb = np.ones(walls.shape + (3,)) * np.array(to_rgb(c["card"]))
    for k, colour in colours.items():
        rooms_rgb[labels == k] = colour[:3]
    rooms_rgb[walls] = np.array(to_rgb(c["ink"]))
    axes[2].imshow(rooms_rgb)
    for k, name in names.items():
        rows, cols = np.nonzero(labels == k)
        short = {"living room and hall": "living\nroom", "wardrobe": "ward-\nrobe"}.get(name, name)
        size = 8.5 if k == 1 else 7
        axes[2].text(cols.mean(), rows.mean(), f"{short}\n{areas[name]:.1f} m²", ha="center", va="center",
                     fontsize=size, color=c["ink"], linespacing=1.1)
    axes[2].set_title("rooms: floor between\nall drawn lines", loc="center", fontsize=9.5, pad=4)
    axes[3].imshow(_paper(c) * 0.35 + np.array(to_rgb(c["card"])) * 0.65)
    for d in p.doors():
        x, y, w, h = d["box"]
        axes[3].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c["dec"], linewidth=1.6))
        overlay = np.zeros(walls.shape + (4,))
        overlay[d["mask"]] = (*to_rgb(c["dec"]), 1.0)
        axes[3].imshow(overlay)
        axes[3].text(x + w + 4 if x < 250 else x - 4, y + h / 2, f"{d['width']} px", fontsize=7.5,
                     color=c["dec_ink"], va="center", ha="left" if x < 250 else "right", fontweight="semibold")
    axes[3].set_title("doors: lines that touch\ntwo rooms", loc="center", fontsize=9.5, pad=4)
    for ax in axes:
        _clean(ax)
    _caption(fig, "Reading a clean plan, step by step",
             f"areas from the scale the doors give: {per * 100:.2f} cm per pixel", c)
    return fig


def chart_scale(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    p = fp()
    _, sizes = p.rooms()
    est, _ = p.scale()
    interior = np.mean([w for kind, w, _ in est if kind == "interior"])
    widths = np.linspace(0.55, 0.95, 81)
    total_px = sum(sizes)
    area = total_px * (widths / interior) ** 2
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.16)
    ax.plot(widths * 100, area, color=c["att"], linewidth=2.2)
    for w in (0.6, 0.7, 0.8, 0.9):
        a = total_px * (w / interior) ** 2
        ax.plot([w * 100], [a], "o", color=c["att"], markersize=6, markeredgecolor=c["card"])
        ax.annotate(f"{a:.0f} m²", (w * 100, a), xytext=(-6, 8), textcoords="offset points", fontsize=9,
                    color=c["ink"], ha="right")
    ax.set_xlabel("width assumed for an interior door, cm")
    ax.set_ylabel("floor area of the flat, m²")
    ax.grid(axis="x", color=c["grid"])
    _caption(fig, "One assumption sets every size in the model",
             f"the interior doors are {interior:.0f} px wide; the area grows with the square of the assumed width",
             c)
    return fig


def chart_scan(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    c = _style(theme)
    p = fp()
    scan = p.scanned()
    found = p.scanned_walls()
    clean = p.walls_in_scan()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.02, wspace=0.05)
    axes[0].imshow(scan)
    axes[0].set_title("a phone photo of a printout", loc="center", fontsize=9.5, pad=4)
    shown = np.ones(found.shape + (3,)) * np.array(to_rgb(c["card"]))
    shown[found] = np.array(to_rgb(c["ink"]))
    axes[1].imshow(shown)
    axes[1].set_title("the same threshold and opening", loc="center", fontsize=9.5, pad=4)
    diff = np.ones(found.shape + (3,)) * np.array(to_rgb(c["card"]))
    diff[found & ~clean] = np.array(to_rgb(c["dec"]))
    diff[clean & ~found] = np.array(to_rgb(c["att"]))
    diff[clean & found] = np.array(to_rgb(c["line"]))
    axes[2].imshow(diff)
    axes[2].set_title("against the true walls", loc="center", fontsize=9.5, pad=4)
    for ax in axes:
        _clean(ax)
    agree = (found & clean).sum() / (found | clean).sum()
    _caption(fig, "The same rules on a photographed plan",
             f"rose: found but not a wall; violet: a wall that was missed; overlap with the true walls "
             f"{agree:.0%} (intersection over union)", c, gap=0.06)
    return fig


# ---------------------------------------------------------------- the 3D model

AZIMUTH, ELEVATION = np.radians(32), np.radians(52)


def _project(x, y, z):
    a, e = AZIMUTH, ELEVATION
    return (x * np.cos(a) - y * np.sin(a), (x * np.sin(a) + y * np.cos(a)) * np.sin(e) + z * np.cos(e))


def _depth(x, y):
    return x * np.sin(AZIMUTH) + y * np.cos(AZIMUTH)


def _floor_rgba(c, tint=True):
    from matplotlib.colors import to_rgb

    p = fp()
    ink, walls, footprint = p.masks()
    labels, _ = p.rooms()
    img = np.zeros(walls.shape + (4,))
    img[footprint] = (*to_rgb(c["card_alt"]), 1.0)
    if tint:
        for k, colour in room_colours(c).items():
            img[labels == k] = colour
    lines = ink & ~walls
    img[lines] = (*to_rgb(c["muted"]), 0.9)
    return img


def draw_model(ax, c, furniture=None, height=None, tint=True, cut=0.35):
    """The plan's walls extruded, seen from above and to the side, with its drawing on the floor; the outer walls
    nearest the viewer are cut to `cut` metres so the rooms can be seen."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    from matplotlib.transforms import Affine2D

    p = fp()
    _, per = p.scale()
    height = height or p.CEILING
    H, W = p.image().shape[:2]
    a, e = AZIMUTH, ELEVATION
    s = per
    matrix = np.array([[s * np.cos(a), s * np.sin(a), -s * np.sin(a) * H],
                       [s * np.sin(a) * np.sin(e), -s * np.cos(a) * np.sin(e), s * np.cos(a) * np.sin(e) * H],
                       [0, 0, 1]])
    ax.imshow(_floor_rgba(c, tint), extent=(0, W, H, 0), transform=Affine2D(matrix) + ax.transData,
              interpolation="bilinear", zorder=0)

    boxes = []  # (x0, y0, x1, y1 in metres, height, top colour, side colours)
    wall_top, wall_left, wall_front = to_rgb(c["ink"]), to_rgb(c["line"]), to_rgb(c["grid"])
    _, _, footprint = p.masks()
    for x0, y0, x1, y1 in p.wall_rectangles():
        # an outer wall on the near side (its left or front face looks outside) is cut low, dollhouse style
        left_out = x0 - 3 < 0 or not footprint[(y0 + y1) // 2, max(x0 - 3, 0)]
        front_out = y1 + 3 >= H or not footprint[min(y1 + 3, H - 1), (x0 + x1) // 2]
        h = cut if (left_out or front_out) else height
        X0, X1 = x0 * s, x1 * s
        Y0, Y1 = (H - y1) * s, (H - y0) * s
        # long walls are cut into short pieces so that drawing back to front stays right
        nx, ny = max(1, int(np.ceil((X1 - X0) / 0.4))), max(1, int(np.ceil((Y1 - Y0) / 0.4)))
        for i in range(nx):
            for j in range(ny):
                boxes.append((X0 + (X1 - X0) * i / nx, Y0 + (Y1 - Y0) * j / ny, X0 + (X1 - X0) * (i + 1) / nx,
                              Y0 + (Y1 - Y0) * (j + 1) / ny, h, wall_top, wall_left, wall_front))
    furniture_colours = {"bed": c["att"], "desk": c["data"], "table": c["data"], "armchair": c["enc"]}
    for item in furniture or []:
        x0, y0, x1, y1 = item["px"]
        colour = np.array(to_rgb(furniture_colours[item["name"]]))
        boxes.append((x0 * s, (H - y1) * s, x1 * s, (H - y0) * s, item["height"], tuple(colour),
                      tuple(colour * 0.78), tuple(colour * 0.9)))
    boxes.sort(key=lambda b: -_depth((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    polys, colours = [], []
    for x0, y0, x1, y1, h, top, left, front in boxes:
        def q(points):
            return np.array([_project(x, y, z) for x, y, z in points])

        polys.append(q([(x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h)]))
        colours.append(left)
        polys.append(q([(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)]))
        colours.append(front)
        polys.append(q([(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]))
        colours.append(top)
    ax.add_collection(PolyCollection(polys, facecolors=colours, edgecolors=colours, linewidths=0.4, zorder=2))
    corners = np.array([_project(x, y, z) for x in (0, W * s) for y in (0, H * s) for z in (0, height)])
    pad = 0.3
    ax.set_xlim(corners[:, 0].min() - pad, corners[:, 0].max() + pad)
    ax.set_ylim(corners[:, 1].min() - pad, corners[:, 1].max() + pad)
    ax.set_aspect("equal")
    _clean(ax)


def chart_model(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    c = _style(theme)
    p = fp()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 5.0), gridspec_kw={"width_ratios": [0.62, 1.38]})
    fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.02, wspace=0.03)
    shown = _paper(c) * 0.45 + np.array(plt.matplotlib.colors.to_rgb(c["card"])) * 0.55
    left.imshow(shown)
    for x0, y0, x1, y1 in p.wall_rectangles():
        left.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c["att_tint"], edgecolor=c["att"],
                                 linewidth=0.6))
    _clean(left)
    left.set_title(f"walls as {len(p.wall_rectangles())} rectangles", loc="center", fontsize=10, pad=4)
    draw_model(right, c)
    right.set_title(f"extruded to {p.CEILING} m, near walls cut", loc="center", fontsize=10, pad=4)
    _caption(fig, "From rectangles to a model",
             "the walls cut into rectangles become boxes; the plan's own drawing stays on the floor", c, gap=0.06)
    return fig


def chart_furnish(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle
    from scipy import ndimage as ndi

    c = _style(theme)
    p = fp()
    placed, _, passed = p.furnish()
    floor, wall, blocked, targets, _, mpc = p.furnishing_grid()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 5.0), gridspec_kw={"width_ratios": [0.62, 1.38]})
    fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.02, wspace=0.03)
    shown = _paper(c) * 0.4 + np.array(to_rgb(c["card"])) * 0.6
    left.imshow(shown)
    # where a person can walk with the furniture in place: at least PASSAGE from anything
    occupied = np.zeros_like(floor)
    for item in placed:
        x0, y0, x1, y1 = (v // p.CELL for v in item["px"])
        occupied[y0:y1, x0:x1] = True
    clear = ndi.distance_transform_edt(floor & ~occupied) * mpc >= p.PASSAGE
    overlay = np.zeros(clear.shape + (4,))
    overlay[clear] = (*to_rgb(c["enc"]), 0.28)
    H, W = p.image().shape[:2]
    left.imshow(overlay, extent=(0, clear.shape[1] * p.CELL, clear.shape[0] * p.CELL, 0))
    left.imshow(np.zeros((H, W, 4)), extent=(0, W, H, 0))
    for item in placed:
        x0, y0, x1, y1 = item["px"]
        fx0, fy0, fx1, fy1 = item["front"]
        left.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, facecolor="none", edgecolor=c["muted"],
                                 linewidth=0.8, linestyle=(0, (2, 2))))
        left.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c["card"], edgecolor=c["ink"], linewidth=1.2))
        left.text((x0 + x1) / 2, (y0 + y1) / 2, item["name"].replace("armchair", "arm-\nchair"), ha="center",
                  va="center", fontsize=7.5, linespacing=1.0,
                  color=c["ink"], rotation=90 if (y1 - y0) > 1.3 * (x1 - x0) and item["name"] != "bed" else 0)
    for t in targets:
        left.plot([t[1] * p.CELL], [t[0] * p.CELL], "o", color=c["enc"], markersize=4.5, markeredgecolor=c["card"])
    _clean(left)
    left.set_title("clearances and free floor", loc="center", fontsize=10, pad=4)
    draw_model(right, c, furniture=placed)
    right.set_title("the chosen layout", loc="center", fontsize=10, pad=4)
    _caption(fig, "Furniture placed by search",
             f"dashed: room each item needs; teal: floor at least {2 * p.PASSAGE:.1f} m wide joining every door; "
             f"best of {passed} valid layouts", c, gap=0.06)
    return fig


# ---------------------------------------------------------------- the pipeline

STEPS = [  # title, detail, role
    ("plan image", "scan, photo, PDF", "data"),
    ("segmentation", "walls, doors, rooms", "enc"),
    ("vectors", "rectangles, polygons", "att"),
    ("scale", "doors, dimensions", "att"),
    ("3D model", "walls and floors", "enc"),
]
STEPS_2 = [
    ("room types", "from fixtures, sizes", "att"),
    ("furniture", "search with rules", "enc"),
    ("materials", "per room style", "enc"),
    ("engine scene", "Unreal Engine 4", "data"),
    ("VR or web", "walk through", "data"),
]


def pipeline() -> Figure:
    fig = Figure(
        760,
        268,
        alt="The pipeline in two rows. A plan image, scanned, photographed or exported as a PDF, goes through "
        "segmentation of walls, doors and rooms, then becomes vectors, rectangles and polygons; the scale comes "
        "from doors and dimension text; the result is a 3D model of walls, floors and openings. Then rooms get "
        "their types from fixtures and sizes, furniture is placed by a search with rules, materials are chosen "
        "per room style, and the scene goes to Unreal Engine 4 for a walk-through in VR or in a browser",
    )
    fig.text(14, 28, "From a drawing to a walk-through", kind="title")
    fig.text(14, 50, "top row: the geometry of the flat; bottom row: what makes it look lived in", kind="note")
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


# ---------------------------------------------------------------- cover and card

def _png(draw, width_px, height_px, theme="dark"):
    import base64
    import io

    import matplotlib.pyplot as plt

    c = dict(_style(theme))
    c.update({"card": "#161618", "card_alt": "#1E1F23", "ink": "#C9CCD3", "line": "#4A4D55", "grid": "#33363D",
              "muted": "#6E7380"})
    for role in ("data", "enc", "att", "dec"):
        c[f"{role}_tint"] = "#26272C"
    c["data_tint"] = "#3A1A1C"
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    draw(ax, c)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _plan_on(ax, c):
    from matplotlib.colors import to_rgb

    a = fp().image().mean(-1)
    img = np.zeros(a.shape + (4,))
    img[..., :3] = to_rgb(c["ink"])
    img[..., 3] = np.clip((1 - a) * 1.4, 0, 1)
    ax.imshow(img)
    _clean(ax)


def cover():
    p = fp()
    H, W = p.image().shape[:2]
    plan_h = 330
    plan_w = plan_h * W / H
    model_w, model_h = 330, 330
    plan = _png(_plan_on, plan_w, plan_h)
    model = _png(lambda ax, c: draw_model(ax, c, furniture=p.furnish()[0]), model_w, model_h)
    x0, y0 = cv.MOTIF_X0, 170
    arrow_x = x0 + plan_w + 18
    out = (f'<image href="{plan}" x="{x0}" y="{y0}" width="{plan_w:.1f}" height="{plan_h}"/>'
           f'<path d="M{arrow_x:.1f} {y0 + plan_h / 2:.1f} h34 m-8 -7 l8 7 l-8 7" stroke="{cv.RED}" '
           f'stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
           f'<image href="{model}" x="{cv.MOTIF_X1 - model_w + 10:.1f}" y="{y0 - 8}" width="{model_w}" '
           f'height="{model_h}"/>')
    labels = cv.label(x0, y0 - 18, "THE DRAWING") + cv.label(cv.MOTIF_X1 - model_w + 40, y0 - 18, "THE MODEL")
    return cv.frame("portfolio / computer vision", ["3D Floorplan", "Reconstruction"],
                    ["From a drawing", "to a walk-through"], out + labels,
                    footer="Plans read by machine, built in 3D")


def card(width=640, height=440):
    p = fp()
    model = _png(lambda ax, c: draw_model(ax, c, furniture=p.furnish()[0]), 420, 420)
    H, W = p.image().shape[:2]
    plan_h = 330
    plan_w = plan_h * W / H
    plan = _png(_plan_on, plan_w, plan_h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern></defs>'
            f'<rect width="{width}" height="{height}" fill="{cv.BG}"/>'
            f'<rect width="{width}" height="{height}" fill="url(#dots)"/>'
            f'<image href="{plan}" x="40" y="{(height - plan_h) / 2:.1f}" width="{plan_w:.1f}" height="{plan_h}"/>'
            f'<path d="M{40 + plan_w + 14:.1f} {height / 2} h30 m-8 -7 l8 7 l-8 7" stroke="{cv.RED}" '
            f'stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<image href="{model}" x="{width - 430}" y="{(height - 420) / 2 + 6:.1f}" width="420" height="420"/>'
            f'</svg>')


COVER = cover
IMAGES = {"card": (card, 640, 440)}

CHARTS = {
    "reading": chart_reading,
    "scale": chart_scale,
    "scan": chart_scan,
    "model": chart_model,
    "furnish": chart_furnish,
}

FIGURES = {
    "pipeline": pipeline,
}


def chart_alts():
    p = fp()
    areas = p.areas()
    est, per = p.scale()
    placed, _, passed = p.furnish()
    rooms = "; ".join(f"{name} {area:.1f} m²" for name, area in areas.items())
    doors = ", ".join(f"{d['width']} px" for d in p.doors())
    return {
        "reading": f"Four panels of the same studio plan: the drawing; the walls left after a morphological "
        f"opening, with thin lines removed; the rooms found as connected floor, coloured and labelled ({rooms}); "
        f"and the doors, found as thin drawings that touch two rooms, with their widths ({doors}). Scale "
        f"{per * 100:.2f} cm per pixel",
        "scale": "A line chart of the flat's floor area against the width assumed for an interior door, from 55 "
        "to 95 cm; the area grows with the square of the assumed width, marked at 60, 70, 80 and 90 cm",
        "scan": "Three panels: the plan as if photographed from a printout, tilted, with a shadow in one corner, "
        "noise and handwritten scribbles; the walls the same rules find on it, with the shadowed corner turned "
        "solid; and a comparison with the true walls, where false walls are coloured and almost nothing is missed",
        "model": f"Left: the plan with its walls covered by {len(p.wall_rectangles())} rectangles. Right: those "
        "rectangles extruded to full height in an axonometric view, with the plan's own drawing of doors and "
        "fixtures on the floors and each room tinted",
        "furnish": "Left: the plan with a bed, a desk, a table and an armchair placed in the living room, dashed "
        "zones in front of each, and the floor where a person can still walk shaded, joining every door. Right: "
        f"the same layout in 3D. It is the best of {passed} layouts that passed the rules",
    }
