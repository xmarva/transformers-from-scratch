"""Figures and charts of the portfolio page "Cashierless Store" on xmarva.github.io.

All of them come from the simulation in store.py: a store with real camera geometry and shoppers walking in it,
detected, put on the floor plan and tracked the way the project's system did, while its data stays under NDA.

    uv run python tools/render.py --article cashierless_store \
        --out <site>/assets/img/projects/2_cachierless_store/figures
"""

from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

MOMENT = 36.0  # s: the moment shown in the camera views, the fusion figure and the cover
SHOWN_CAMERAS = [0, 6, 7, 2, 4, 9]  # the six of twelve views in the figure


@cache
def sim():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("cashierless_store_sim", Path(__file__).parent / "store.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
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


def shopper_colour(i, c):
    return [c["dec"], c["enc"], c["data"], c["att"], "#5B8FD6"][i]


# ---------------------------------------------------------------- the map

def draw_map(ax, c, cameras=True, labels=True):
    from matplotlib.patches import Rectangle

    s = sim()
    ax.add_patch(Rectangle((0, 0), s.WIDTH, s.DEPTH, facecolor=c["card_alt"], edgecolor="none", zorder=0))
    walls = [((0, s.DEPTH), (s.WIDTH, s.DEPTH)), ((0, 0), (0, s.DEPTH)), ((s.WIDTH, 0), (s.WIDTH, s.DEPTH)),
             ((0, 0), (s.ENTRANCE[0], 0)), ((s.ENTRANCE[1], 0), (s.WIDTH, 0))]
    for (x0, y0), (x1, y1) in walls:
        ax.plot([x0, x1], [y0, y1], color=c["ink"], linewidth=2.2, solid_capstyle="butt", zorder=3)
    fills = {"wall": c["line"], "gondola": c["line"], "fridge": "#9CC3D5", "table": "#C9A27A"}
    for x0, x1, y0, y1, _, kind in s.SHELVES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fills[kind], edgecolor="none", alpha=0.85,
                               zorder=1))
    if labels:
        ax.text(sum(s.ENTRANCE) / 2, -0.35, "door", ha="center", va="top", fontsize=8.5, color=c["muted"])
        ax.text(11.6, 4.5, "fridges", ha="center", va="center", rotation=90, fontsize=8, color=c["ink"], zorder=2)
        ax.text(9.7, 4.0, "fruit", ha="center", va="center", rotation=90, fontsize=8, color=c["ink"], zorder=2)
    if cameras:
        for cam in s.cameras():
            x, y, _ = cam.position
            tx, ty = cam.target
            d = np.array([tx - x, ty - y])
            d = d / np.linalg.norm(d) * 0.45
            ax.annotate("", xy=(x + d[0], y + d[1]), xytext=(x, y),
                        arrowprops={"arrowstyle": "-|>", "color": c["ink"], "linewidth": 1.0, "mutation_scale": 7},
                        zorder=6)
            ax.plot([x], [y], "s", color=c["ink"], markersize=4.5, zorder=6)
    ax.set_xlim(-0.4, s.WIDTH + 0.4)
    ax.set_ylim(-0.9, s.DEPTH + 0.35)
    ax.set_aspect("equal")
    _clean(ax)


@cache
def coverage(step=0.4):
    """How many cameras would detect a person of average height standing at each point of the floor."""
    s = sim()
    xs = np.arange(step / 2, s.WIDTH, step)
    ys = np.arange(step / 2, s.DEPTH, step)
    counts = np.zeros((len(ys), len(xs)))
    blocked = np.zeros_like(counts, dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1, _, _ in s.SHELVES):
                blocked[j, i] = True
                continue
            person = [(0, x, y, s.HEIGHT_GUESS, "#000000")]
            for cam in s.cameras():
                items = s.view(cam, person)
                ids = s.id_buffer(items, 0.25)
                alone = s.id_buffer([it for it in items if it[3] == 0], 0.25)
                full, seen = (alone == 0).sum(), (ids == 0).sum()
                rows = np.nonzero(ids == 0)[0]
                if full >= 10 and seen >= s.MIN_VISIBLE * full and rows.min() > 0:
                    counts[j, i] += 1
    return xs, ys, counts, blocked


def chart_layout(theme):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    c = _style(theme)
    s = sim()
    xs, ys, counts, blocked = coverage()
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    fig.subplots_adjust(left=0.01, right=0.9, top=0.86, bottom=0.02)
    draw_map(ax, c, cameras=False)
    top = int(counts.max())
    cmap = LinearSegmentedColormap.from_list("cams", [c["card"], c["att_tint"], c["att"], c["att_ink"]])
    norm = BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N)
    shown = np.ma.masked_where(blocked, counts)
    step = xs[1] - xs[0]
    ax.imshow(shown, extent=(xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2),
              origin="lower", cmap=cmap, norm=norm, alpha=0.9, zorder=0.5, interpolation="nearest")
    draw_map(ax, c, cameras=True)
    ax.imshow(shown, extent=(xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2),
              origin="lower", cmap=cmap, norm=norm, alpha=0.9, zorder=0.5, interpolation="nearest")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.01,
                       ticks=range(0, top + 1, 2))
    bar.set_label("cameras that see a person here", color=c["muted"])
    bar.outline.set_visible(False)
    _caption(fig, f"{len(s.CAMERAS)} ceiling cameras over a 12 by 8 m store",
             "squares: cameras, arrows: where they look; colour: how many of them see a 1.72 m person standing there",
             c, gap=0.06)
    return fig


# ---------------------------------------------------------------- the camera views

def draw_view(ax, cam_index, t, c, boxes=True):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    s = sim()
    cam = s.cameras()[cam_index]
    items = s.view(cam, s.people_at(t))
    ax.add_patch(Rectangle((0, 0), *s.IMAGE, facecolor="#B9C1CB", edgecolor="none", zorder=0))
    ax.add_collection(PolyCollection([it[1] for it in items], facecolors=[it[2] for it in items],
                                     edgecolors=[it[2] for it in items], linewidths=0.3, zorder=1))
    if boxes:
        for d in s.detections()[(round(t, 3), cam_index)]:
            x0, y0, x1, y1 = d["box"]
            colour = shopper_colour(d["id"], c)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=1.6,
                                   linestyle="-" if d["whole"] else (0, (3, 2)), zorder=3))
            if d["whole"]:
                ax.plot([(x0 + x1) / 2], [y1], "o", color="white", markeredgecolor=colour, markersize=4.5, zorder=4)
            else:
                ax.plot([(x0 + x1) / 2], [y0], "v", color="white", markeredgecolor=colour, markersize=5, zorder=4)
    ax.set_xlim(0, s.IMAGE[0])
    ax.set_ylim(s.IMAGE[1], 0)
    ax.set_aspect("equal")
    _clean(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(c["line"])
        spine.set_linewidth(0.8)


def chart_views(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.01, wspace=0.03, hspace=0.12)
    for ax, k in zip(axes.ravel(), SHOWN_CAMERAS, strict=True):
        draw_view(ax, k, MOMENT, c)
        ax.set_title(f"camera {k + 1}", loc="left", fontsize=9, pad=3, color=c["muted"])
    _caption(fig, f"Six of the twelve cameras at the same moment, {MOMENT:.0f} s into the simulation",
             "a box per person in the shopper's colour; dot: the feet are in the box; dashed, with a triangle: "
             "hidden feet, the head is used", c, gap=0.065)
    return fig


# ---------------------------------------------------------------- homography

HOMOGRAPHY_CAMERA = 2  # the camera over the third aisle, looking down it
GRID_POINTS = [(5.6, 2.0), (7.2, 2.0), (7.2, 6.0), (5.6, 6.0)]


def chart_homography(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = sim()
    cam = s.cameras()[HOMOGRAPHY_CAMERA]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.33, 1.0]})
    fig.subplots_adjust(left=0.005, right=0.995, top=0.8, bottom=0.02, wspace=0.04)
    draw_view(left, HOMOGRAPHY_CAMERA, MOMENT, c, boxes=False)
    draw_map(right, c, cameras=False, labels=False)
    for z, colour, name in ((0.0, c["att"], "floor"), (s.HEIGHT_GUESS, c["enc"], "head height")):
        for v in np.arange(1.0, 8.0, 1.0):  # lines of the plane, seen by the camera and on the map
            line = np.array([[x, v, z] for x in np.linspace(5.6, 7.2, 20)])
            px = cam.project(line)
            left.plot(px[:, 0], px[:, 1], color=colour, linewidth=0.8, alpha=0.9)
        for u in (5.6, 6.4, 7.2):
            line = np.array([[u, y, z] for y in np.linspace(1.0, 7.0, 40)])
            px = cam.project(line)
            left.plot(px[:, 0], px[:, 1], color=colour, linewidth=0.8, alpha=0.9)
        left.text(0.02, 0.04 if z == 0 else 0.11, f"grid on the {name}", transform=left.transAxes, fontsize=8.5,
                  color=colour, fontweight="semibold")
    for v in np.arange(1.0, 8.0, 1.0):
        right.plot([5.6, 7.2], [v, v], color=c["att"], linewidth=0.8)
    for u in (5.6, 6.4, 7.2):
        right.plot([u, u], [1.0, 7.0], color=c["att"], linewidth=0.8)
    for k, (x, y) in enumerate(GRID_POINTS):
        px = cam.project(np.array([[x, y, 0.0]]))[0]
        for ax, (a, b) in ((left, px), (right, (x, y))):
            ax.plot([a], [b], "o", color=c["dec"], markersize=6, markeredgecolor=c["card"], zorder=5)
            ax.text(a + (12 if ax is left else 0.25), b, str(k + 1), fontsize=9, color=c["dec_ink"],
                    fontweight="semibold", va="center", zorder=5)
    cx, cy, _ = cam.position
    right.plot([cx], [cy], "s", color=c["ink"], markersize=5, zorder=6)
    right.text(cx + 0.3, cy - 0.15, "camera", fontsize=8.5, color=c["ink"], va="top")
    left.set_title("what the camera sees", loc="center", fontsize=10, pad=4)
    right.set_title("the floor plan", loc="center", fontsize=10, pad=4)
    _caption(fig, "Pixels to floor coordinates",
             "four marked floor points fix the floor's homography; the same algebra gives one for head height", c)
    return fig


# ---------------------------------------------------------------- fusion

def chart_fusion(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = sim()
    t = round(MOMENT, 3)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02)
    draw_map(ax, c, cameras=True)
    truth = s.positions(t)
    for cam_index, cam in enumerate(s.cameras()):
        for d in s.detections()[(t, cam_index)]:
            colour = shopper_colour(d["id"], c)
            p = d["floor"]
            ax.plot([cam.position[0], p[0]], [cam.position[1], p[1]], color=colour, linewidth=0.5, alpha=0.35,
                    zorder=2)
            ax.plot([p[0]], [p[1]], "o" if d["whole"] else "v", color=colour, markersize=4.5,
                    markeredgecolor=c["card"], markeredgewidth=0.6, alpha=0.95, zorder=4)
    for i, xy in truth.items():
        ax.plot([xy[0]], [xy[1]], "o", markersize=15, markerfacecolor="none",
                markeredgecolor=shopper_colour(i, c), markeredgewidth=1.6, zorder=5)
        points = [d["floor"] for k in range(len(s.cameras())) for d in s.detections()[(t, k)] if d["id"] == i]
        if points:
            m = np.median(np.array(points), axis=0)
            ax.plot([m[0]], [m[1]], "X", color=c["ink"], markersize=7, markeredgecolor=c["card"], zorder=6)
    _caption(fig, "Every camera's guess of where each shopper stands",
             "circles: the true positions; dots: from the feet, triangles: from the head; lines: back to the "
             "camera; ×: the median", c, gap=0.06)
    return fig


def chart_errors(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = sim()
    found = [d for ds in s.detections().values() for d in ds]
    groups = [
        ("from the feet, whole person in the box", [d["from_feet"] for d in found if d["whole"]],
         [d["truth"] for d in found if d["whole"]], c["att"]),
        ("from the head, assumed height 1.72 m", [d["from_head"] for d in found],
         [d["truth"] for d in found], c["enc"]),
        ("from the feet, box cut short", [d["from_feet"] for d in found if not d["whole"]],
         [d["truth"] for d in found if not d["whole"]], c["dec"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.15)
    for name, est, truth, colour in groups:
        err = np.sort(np.linalg.norm(np.array(est) - np.array(truth), axis=1)) * 100
        share = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, share, color=colour, linewidth=2.2, label=f"{name}: median {np.median(err):.0f} cm")
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=c["ink"])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("distance from the true position, cm")
    ax.set_ylabel("share of detections")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=c["grid"])
    _caption(fig, "How far a single camera's floor point lands from the person",
             f"{len(found):,} detections in the simulation; each line: the share of them within a given distance",
             c)
    return fig


# ---------------------------------------------------------------- tracks

def _track_owner(track):
    """The shopper a track followed most of the time (for colour only)."""
    s = sim()
    votes = {}
    for t, x, y in track.history:
        truth = s.positions(round(t, 3))
        if truth:
            i = min(truth, key=lambda k: np.linalg.norm(truth[k] - [x, y]))
            if np.linalg.norm(truth[i] - [x, y]) < 1.0:
                votes[i] = votes.get(i, 0) + 1
    return max(votes, key=votes.get) if votes else None


def chart_tracks(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = sim()
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02)
    draw_map(ax, c, cameras=False)
    for i, walk in enumerate(s.walks()):
        ax.plot(walk[:, 1], walk[:, 2], color=shopper_colour(i, c), linewidth=0.8, linestyle=(0, (2, 2)),
                alpha=0.7, zorder=2)
        for _, _, _, _, waypoints in [s.SHOPPERS[i]]:
            route, stops = s._route(waypoints, s.SHOPPERS[i][3])
            for k, seconds in stops.items():
                ax.plot([route[k][0]], [route[k][1]], "o", markersize=4 + 2 * seconds, markerfacecolor="none",
                        markeredgecolor=shopper_colour(i, c), markeredgewidth=0.9, alpha=0.8, zorder=2)
    for track in s.tracks():
        owner = _track_owner(track)
        h = np.array([(x, y) for _, x, y in track.history])
        colour = shopper_colour(owner, c) if owner is not None else c["muted"]
        ax.plot(h[:, 0], h[:, 1], color=colour, linewidth=1.5, alpha=0.95, zorder=3)
    sc = s.scores()
    _caption(fig, "Five shoppers' paths, as the tracker followed them",
             f"solid: tracks, dashed: true paths, rings: stops. Mean error {sc['mean'] * 100:.0f} cm, "
             f"tracked {sc['tracked']:.1%} of the time, {sc['switches']} identity switches", c, gap=0.06)
    return fig


def chart_handoff(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = _style(theme)
    s = sim()
    times = s.frames()
    n = len(s.SHOPPERS)
    grid = np.full((n, len(times)), np.nan)
    for k, t in enumerate(times):
        t = round(t, 3)
        for i in s.positions(t):
            grid[i, k] = sum(1 for cam in range(len(s.cameras())) for d in s.detections()[(t, cam)] if d["id"] == i)
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.72, bottom=0.18)
    cmap = LinearSegmentedColormap.from_list("cams", [c["card_alt"], c["att_tint"], c["att"], c["att_ink"]])
    image = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(grid),
                      extent=(times[0], times[-1] + 1 / s.FPS, n - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"shopper {i + 1}" for i in range(n)], fontsize=9)
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(shopper_colour(i, c))
    ax.set_xlabel("seconds")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("cameras", color=c["muted"])
    bar.outline.set_visible(False)
    counts = grid[~np.isnan(grid)]
    _caption(fig, "How many cameras saw each shopper, frame by frame",
             f"{s.FPS} frames a second; on average {counts.mean():.1f} cameras at once, never fewer than "
             f"{int(counts.min())}", c, gap=0.1)
    return fig


def chart_visits(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = sim()
    owner = {tr.id: _track_owner(tr) for tr in s.tracks()}
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.15)
    n = len(s.SHOPPERS)
    for i, start, end, _, _shelf in s.stops_true():
        y = n - 1 - i
        ax.barh(y + 0.17, end - start, left=start, height=0.3, color=shopper_colour(i, c), alpha=0.3)
    for tid, start, end, _, shelf in s.stops_found():
        i = owner[tid]
        y = n - 1 - i
        ax.barh(y - 0.17, end - start, left=start, height=0.3, color=shopper_colour(i, c))
        ax.text((start + end) / 2, y - 0.42, shelf, ha="center", va="top", fontsize=7.5, color=c["ink"])
    for i, walk in enumerate(s.walks()):
        y = n - 1 - i
        ax.plot([walk[0, 0], walk[-1, 0]], [y, y], color=c["line"], linewidth=0.8, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"shopper {n - k}" for k in range(n)], fontsize=9)
    for k, label in enumerate(ax.get_yticklabels()):
        label.set_color(shopper_colour(n - 1 - k, c))
    ax.set_ylim(-0.75, n - 0.45)
    ax.set_xlabel("seconds")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c["grid"])
    found, true = len(s.stops_found()), len(s.stops_true())
    _caption(fig, "Where the shoppers stopped, read from the tracks",
             f"pale: planned; solid: found from the track's speed, {found} of {true}, with the nearest shelf", c)
    return fig


# ---------------------------------------------------------------- the pipeline

PIPELINE = [  # (title, detail, role)
    ("12 cameras", "5 frames a second", "data"),
    ("people detector", "a box per person", "enc"),
    ("floor point", "feet, or head", "enc"),
    ("fuse cameras", "median per person", "att"),
    ("track", "Kalman + Hungarian", "att"),
]
PIPELINE_2 = [
    ("shelf visits", "a track stops", "att"),
    ("take or put back", "hand at the shelf", "enc"),
    ("which product", "per-shelf model", "enc"),
    ("virtual cart", "per track", "data"),
    ("receipt", "at the door", "data"),
]


def pipeline() -> Figure:
    fig = Figure(
        760,
        268,
        alt="The pipeline. Twelve cameras at five frames a second feed a people detector that puts a box around "
        "each person; each box gives a floor point, from the feet or, when they are hidden, from the head; the "
        "points of all cameras are fused per person by their median and tracked with a Kalman filter and the "
        "Hungarian method. Where a track stops at a shelf, shelf cameras look at the hands to decide whether a "
        "product was taken or put back, a classifier per shelf says which product, and it goes to the track's "
        "virtual cart, which becomes a receipt at the door",
    )
    fig.text(14, 28, "From camera frames to a receipt", kind="title")
    fig.text(14, 50, "top row: where every person is; bottom row: what each of them took", kind="note")
    w, gap, h = 132, 18, 52
    for row, steps in enumerate((PIPELINE, PIPELINE_2)):
        top = 86 + row * 110
        for k, (first, second, role) in enumerate(steps):
            x = 14 + k * (w + gap)
            fig.box(x, top, w, h, role=role, r=8)
            fig.text(x + w / 2, top + 22, first, kind="body", role=role, anchor="middle")
            fig.text(x + w / 2, top + 40, second, kind="small", anchor="middle")
            if k < len(steps) - 1:
                fig.arrow((x + w + 3, top + h / 2), (x + w + gap - 3, top + h / 2), width=1.3)
    # the track feeds the shelf visits
    x_track = 14 + 4 * (w + gap) + w / 2
    fig.arrow((x_track, 86 + h + 3), (14 + w / 2 + 30, 196 - 4), width=1.3)
    return fig


# ---------------------------------------------------------------- cover and card

def _map_png(width_px, height_px, dark=True):
    """The store map with the tracks, drawn for the cover in the site's colours."""
    import base64
    import io

    import matplotlib.pyplot as plt

    s = sim()
    c = dict(_style("dark"))
    c.update({"card_alt": "#1E1F23", "ink": "#C9CCD3", "line": "#3A3D44", "muted": "#6E7380"})
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    draw_map(ax, c, cameras=True, labels=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colours = [cv.RED, "#C9CCD3", "#8E939E", "#C9CCD3", "#8E939E"]
    for track in s.tracks():
        owner = _track_owner(track)
        h = np.array([(x, y) for _, x, y in track.history])
        colour = colours[owner] if owner is not None else "#6E7380"
        ax.plot(h[:, 0], h[:, 1], color=colour, linewidth=2.6 if owner == 0 else 1.4,
                alpha=1.0 if owner == 0 else 0.8, zorder=4 if owner == 0 else 3)
    t = round(MOMENT, 3)
    for i, xy in s.positions(t).items():
        ax.plot([xy[0]], [xy[1]], "o", markersize=8, color=colours[i], markeredgecolor="#161618", zorder=7)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _view_png(cam_index, width_px):
    import base64
    import io

    import matplotlib.pyplot as plt

    s = sim()
    c = _style("dark")
    fig = plt.figure(figsize=(width_px / 100, width_px * s.IMAGE[1] / s.IMAGE[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_view(ax, cam_index, MOMENT, c)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _motif(x, y, width):
    s = sim()
    height = width * (s.DEPTH + 1.25) / (s.WIDTH + 0.8)
    return (f'<image href="{_map_png(width, height)}" x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}"/>'), height


def cover():
    motif, height = _motif(cv.MOTIF_X0 - 6, 150, 440)
    view_w = 200
    view_h = view_w * 3 / 4
    vx, vy = cv.MOTIF_X1 - view_w + 8, 150 + height - view_h * 0.45
    view = (f'<clipPath id="view"><rect x="{vx}" y="{vy}" width="{view_w}" height="{view_h}" rx="8"/></clipPath>'
            f'<image href="{_view_png(SHOWN_CAMERAS[1], view_w)}" x="{vx}" y="{vy}" width="{view_w}" '
            f'height="{view_h}" clip-path="url(#view)"/>'
            f'<rect x="{vx}" y="{vy}" width="{view_w}" height="{view_h}" rx="8" fill="none" stroke="{cv.LINE}" '
            f'stroke-width="1.5"/>')
    labels = cv.label(cv.MOTIF_X0, 132, "THE STORE, FROM ABOVE") + cv.label(vx, vy + view_h + 22, "CAMERA 7")
    return cv.frame("portfolio / computer vision", ["Cashierless", "Store"],
                    ["Following every shopper", "across a dozen cameras"], motif + view + labels,
                    footer="Multi-camera tracking on a floor plan")


def card(width=640, height=440):
    s = sim()
    map_w = width - 60
    map_h = map_w * (s.DEPTH + 1.25) / (s.WIDTH + 0.8)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern></defs>'
            f'<rect width="{width}" height="{height}" fill="{cv.BG}"/>'
            f'<rect width="{width}" height="{height}" fill="url(#dots)"/>'
            f'<image href="{_map_png(map_w, map_h)}" x="30" y="{(height - map_h) / 2:.1f}" width="{map_w}" '
            f'height="{map_h:.1f}"/></svg>')


COVER = cover
IMAGES = {"card": (card, 640, 440)}

CHARTS = {
    "layout": chart_layout,
    "views": chart_views,
    "homography": chart_homography,
    "fusion": chart_fusion,
    "errors": chart_errors,
    "tracks": chart_tracks,
    "handoff": chart_handoff,
    "visits": chart_visits,
}

FIGURES = {
    "pipeline": pipeline,
}


def _track_owner_by_id(tid):
    return _track_owner(next(tr for tr in sim().tracks() if tr.id == tid))


def chart_alts():
    s = sim()
    sc = s.scores()
    xs, ys, counts, blocked = coverage()
    free = counts[~blocked]
    found = [d for ds in s.detections().values() for d in ds]

    def median_cm(key, keep):
        e = [np.linalg.norm(d[key] - d["truth"]) for d in found if keep(d)]
        return round(float(np.median(e)) * 100)

    return {
        "layout": f"A floor plan of a 12 by 8 m store with wall shelves, three gondolas, a fruit table and a fridge "
        f"wall, and {len(s.CAMERAS)} ceiling cameras: one at each end of the five aisles, looking down them, and two "
        f"over the door and the front corridor. The floor is coloured by how many cameras see a person standing "
        f"there: {int(free.min())} to {int(free.max())}, {np.mean(free >= 2):.0%} of the free floor is seen by at "
        f"least two",
        "views": f"Six camera pictures of the simulated store at {MOMENT:.0f} seconds: shelves with coloured rows "
        "of products, a tiled floor and shoppers drawn as simple figures, each with a coloured box. Where a shelf "
        "or the picture's edge hides the feet, the box is dashed and marked at the top",
        "homography": "Left: the picture of one aisle camera with two grids drawn into it, one on the floor and one "
        "at head height, and four numbered floor points. Right: the floor plan with the same grid and the same "
        "four points, which is enough to fix the homography between them",
        "fusion": f"The floor plan at {MOMENT:.0f} seconds. Each shopper's true position is a ring; every camera's "
        "floor point for that shopper is a dot or a triangle in the shopper's colour, with a faint line back to "
        "the camera; the median of a shopper's points is a cross, close to the ring",
        "errors": "Cumulative distributions of a single camera's floor-point error. From the feet with the whole "
        f"person in the box the median is {median_cm('from_feet', lambda d: d['whole'])} cm; from the head with an "
        f"assumed height, {median_cm('from_head', lambda d: True)} cm; from the bottom of a box cut short by a "
        f"shelf or the image edge, {median_cm('from_feet', lambda d: not d['whole'])} cm",
        "tracks": f"The floor plan with five shoppers' true paths dashed and the tracker's paths solid, in the same "
        f"colours, from the door through the aisles and back, with rings where they stopped. Mean error "
        f"{sc['mean'] * 100:.0f} cm, {sc['tracked']:.1%} of the time tracked, {sc['switches']} identity switches",
        "visits": "A timeline per shopper with the stops the script planned and the stops found from the tracks, "
        "each labelled with the nearest shelf: " + "; ".join(
            f"shopper {_track_owner_by_id(tid) + 1} at the {shelf} from {start:.0f} to {end:.0f} s"
            for tid, start, end, _, shelf in s.stops_found()),
        "handoff": "A heat strip per shopper over the minute of the simulation, showing how many cameras detected "
        "them in each frame, changing as they walk from aisle to aisle",
    }
