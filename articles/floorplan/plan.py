"""Reading one real floor plan, the way a 2018 pipeline read clean drawings, and building it in 3D.

The plan is the project's own example (data/plan.png, a studio with a balcony, a bathroom, a closet and built-in
wardrobes). Every step is classical image processing, so each one can be shown: walls are the thick dark strokes
that survive a morphological opening; rooms are the connected pieces of floor once every drawn line, door leaves
and swing arcs included, counts as a barrier; doors are the thin drawings that touch two rooms; the scale comes from
standard door widths. The walls are then cut into rectangles and extruded, and furniture is placed by a search that
keeps walking paths open.

Coordinates: pixels of plan.png (column, row), or metres with x along the columns and y up the page.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
DARK, INK = 0.45, 0.8  # brightness below which a pixel is a wall stroke, or any drawn line
MIN_WALL = 400  # px: smaller dark pieces are fixtures (a stove's burners), not walls
MIN_ROOM = 1500  # px
ENTRANCE_DOOR_M, INTERIOR_DOOR_M = 0.9, 0.7  # standard leaf widths assumed for the scale
CEILING = 2.7  # m


@cache
def image():
    from PIL import Image

    return np.asarray(Image.open(DATA / "plan.png").convert("RGB")).astype(float) / 255


@cache
def masks():
    """ink: every drawn line; walls: thick strokes; footprint: the apartment with its openings closed."""
    from scipy import ndimage as ndi
    from skimage.morphology import disk

    a = image()
    ink = a.min(-1) < INK
    walls = ndi.binary_opening(a.min(-1) < DARK, structure=np.ones((5, 5)))
    lab, n = ndi.label(walls)
    sizes = ndi.sum(walls, lab, range(1, n + 1))
    walls = np.isin(lab, 1 + np.nonzero(sizes > MIN_WALL)[0])
    r = 60
    footprint = ndi.binary_fill_holes(ndi.binary_closing(np.pad(walls, r), structure=disk(r)))[r:-r, r:-r]
    return ink, walls, footprint


@cache
def rooms():
    """Room labels (0: not a room) and their pixel counts, largest first."""
    from scipy import ndimage as ndi

    ink, _, footprint = masks()
    free = footprint & ~ndi.binary_dilation(ink, iterations=2)
    lab, n = ndi.label(free)
    sizes = ndi.sum(free, lab, range(1, n + 1))
    keep = sorted([k + 1 for k in range(n) if sizes[k] > MIN_ROOM], key=lambda k: -sizes[k - 1])
    out = np.zeros_like(lab)
    for new, old in enumerate(keep, 1):
        out[lab == old] = new
    # give back the two pixels the barrier took from each side, so areas are not shrunk
    grown = ndi.grey_dilation(out, size=(5, 5))
    out = np.where((out == 0) & footprint & ~masks()[1] & (grown > 0), grown, out)
    return out, [int((out == k).sum()) for k in range(1, len(keep) + 1)]


@cache
def doors():
    """Thin drawings (leaves, arcs) that touch two rooms, or a room and the outside: (rooms, box, width px)."""
    from scipy import ndimage as ndi

    ink, walls, footprint = masks()
    labels, sizes = rooms()
    thin = ink & ~ndi.binary_dilation(walls, iterations=2)
    lab, _ = ndi.label(ndi.binary_dilation(thin, iterations=2))
    grown = {k: ndi.binary_dilation(labels == k, iterations=6) for k in range(1, len(sizes) + 1)}
    outside = ndi.binary_dilation(~footprint, iterations=6)
    found = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab == k
        touch = [r for r, g in grown.items() if (comp & g).any()]
        if (comp & outside).any():
            touch.append(0)
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if len(touch) >= 2 and max(w, h) >= 20:  # shorter bits are ends of the same doors' lines
            found.append({"rooms": tuple(sorted(touch)), "box": (sl[1].start, sl[0].start, w, h),
                          "width": max(w, h), "mask": comp})
    return found


ROOM_NAMES = {}  # filled by name_rooms


@cache
def name_rooms():
    """Names from simple facts: the biggest room is the living room; the room behind the double door on the short
    side is the balcony; the room with the entrance next to it... is decided below from doors and sizes."""
    labels, sizes = rooms()
    found = doors()
    living = 1
    names = {living: "living room and hall"}
    for d in found:
        others = [r for r in d["rooms"] if r not in (0, living)]
        if living in d["rooms"] and others and d["width"] > 120:
            r = others[0]
            box = d["box"]
            names[r] = "balcony" if box[2] > box[3] else "wardrobe"
    small = [r for r in range(1, len(sizes) + 1) if r not in names]
    by_size = sorted(small, key=lambda r: -sizes[r - 1])
    if by_size:
        names[by_size[0]] = "bathroom"
    for r in by_size[1:]:
        names[r] = "closet"
    return names


def scale():
    """Metres per pixel from door widths: the entrance door is assumed ENTRANCE_DOOR_M wide, the single interior
    doors INTERIOR_DOOR_M. Returns the estimate from each door and their mean."""
    estimates = []
    for d in doors():
        if 0 in d["rooms"]:
            estimates.append(("entrance", d["width"], ENTRANCE_DOOR_M / d["width"]))
        elif d["width"] < 100:
            estimates.append(("interior", d["width"], INTERIOR_DOOR_M / d["width"]))
    return estimates, float(np.mean([e[2] for e in estimates]))


def areas():
    _, per = scale()
    labels, sizes = rooms()
    return {name_rooms()[k]: sizes[k - 1] * per ** 2 for k in range(1, len(sizes) + 1)}


# ---------------------------------------------------------------- walls as boxes

@cache
def wall_rectangles(cell=3):
    """The wall mask cut into axis-aligned rectangles (pixels: x0, y0, x1, y1): runs along each row of a coarse
    grid, merged downwards while they keep the same extent."""
    _, walls, _ = masks()
    h, w = walls.shape
    gh, gw = h // cell, w // cell
    grid = walls[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean((1, 3)) > 0.5
    open_runs, rects = {}, []
    for r in range(gh + 1):
        runs = set()
        if r < gh:
            c = 0
            while c < gw:
                if grid[r, c]:
                    start = c
                    while c < gw and grid[r, c]:
                        c += 1
                    runs.add((start, c))
                else:
                    c += 1
        for run in list(open_runs):
            if run not in runs:
                rects.append((run[0] * cell, open_runs.pop(run) * cell, run[1] * cell, r * cell))
        for run in runs:
            open_runs.setdefault(run, r)
    return rects


# ---------------------------------------------------------------- furniture

# name, width, depth (m), height, which side must touch a wall ("back" is the depth side), clearance in front (m)
FURNITURE = [
    ("bed", 1.6, 2.0, 0.5, "head", 0.6),
    ("desk", 1.2, 0.6, 0.75, "back", 0.7),
    ("table", 1.0, 0.8, 0.75, None, 0.6),
    ("armchair", 0.8, 0.8, 0.8, None, 0.4),
]
PASSAGE = 0.35  # m: half the width a walking path needs


def _fixture_mask():
    """Kitchen fixtures, sinks, the shower: thin closed drawings inside a room, grown by a working zone."""
    from scipy import ndimage as ndi

    ink, walls, _ = masks()
    thin = ink & ~ndi.binary_dilation(walls, iterations=2)
    lab, _ = ndi.label(ndi.binary_dilation(thin, iterations=2))
    door_px = np.zeros_like(thin)
    for d in doors():
        door_px |= d["mask"]
    fixtures = np.zeros_like(thin)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        piece = lab[sl] == k
        if min(h, w) >= 14 and not (piece & door_px[sl]).any():  # a fixture is a shape; slivers line the walls
            fixtures[sl] |= piece
    return fixtures


CELL = 4  # px per furnishing grid cell, about 5 cm


@cache
def furnishing_grid():
    """The living room on a coarse grid: floor, walls (or outside), cells blocked by door swings and fixtures'
    working zones, and the cells a walking path must connect (the entrance first)."""
    from scipy import ndimage as ndi

    _, per = scale()
    labels, _ = rooms()
    _, walls, footprint = masks()
    room = labels == 1
    h, w = room.shape
    gh, gw = h // CELL, w // CELL

    def coarse(m, how=np.any):
        return how(m[:gh * CELL, :gw * CELL].reshape(gh, CELL, gw, CELL), axis=(1, 3))

    floor = coarse(room, np.all)
    wall = coarse(walls | ~footprint)
    zone = ndi.binary_dilation(_fixture_mask(), iterations=int(0.6 / per))
    blocked = coarse(zone)
    for d in doors():
        x, y, bw, bh = d["box"]
        swing = np.zeros_like(room)
        swing[max(y - 4, 0):y + bh + 4, max(x - 4, 0):x + bw + 4] = True
        blocked |= coarse(swing)
    entrance = next(d for d in doors() if 0 in d["rooms"])
    ordered = [entrance] + [d for d in doors() if d is not entrance]
    targets = []
    for d in ordered:  # the floor cell of this room nearest to each door's middle
        x, y, bw, bh = d["box"]
        middle = np.array([(y + bh / 2) / CELL, (x + bw / 2) / CELL])
        rows, cols = np.nonzero(floor & ~blocked)
        k = np.argmin((rows - middle[0]) ** 2 + (cols - middle[1]) ** 2)
        targets.append((int(rows[k]), int(cols[k])))
    rows, cols = np.nonzero(coarse(_fixture_mask()) & floor | coarse(_fixture_mask() & (labels == 1)))
    kitchen = np.array([rows.mean(), cols.mean()]) if len(rows) else None
    return floor, wall, blocked, targets, kitchen, per * CELL


def _cells(metres, m_per_cell):
    return max(1, int(round(metres / m_per_cell)))


def _item(name):
    return next(f for f in FURNITURE if f[0] == name)


def item_boxes(name, r, c, rot, m_per_cell):
    """Grid boxes (r0, c0, r1, c1) of an item's body and of the clearance zone in front of it."""
    _, wd, dp, _, _, front = _item(name)
    bw, bd = (wd, dp) if rot % 2 == 0 else (dp, wd)
    r0, c0, r1, c1 = r, c, r + _cells(bd, m_per_cell), c + _cells(bw, m_per_cell)
    f = _cells(front, m_per_cell)
    zone = {0: (r1, c0, r1 + f, c1), 1: (r0, c0 - f, r1, c0), 2: (r0 - f, c0, r0, c1), 3: (r0, c1, r1, c1 + f)}[rot]
    return (r0, c0, r1, c1), tuple(max(v, 0) for v in zone)


@cache
def candidates(name):
    """(row, col, rotation) where the item's body fits on free floor and, if it needs one, its back touches a wall.
    Rotation 0 has the back to the top of the page, then clockwise."""
    from scipy import ndimage as ndi

    floor, wall, blocked, _, _, mpc = furnishing_grid()
    usable = floor & ~blocked
    gh, gw = usable.shape
    _, wd, dp, _, against, _ = _item(name)
    out = []
    for rot in range(4):
        (r0, c0, r1, c1), _ = item_boxes(name, 0, 0, rot, mpc)
        bh, bw = r1, c1
        # fits[r, c]: the box with its top-left corner at (r, c) lies on usable floor
        ok = ndi.minimum_filter(usable.astype(np.uint8), size=(bh, bw), mode="constant", cval=0)
        fits = np.zeros_like(usable)
        fits[: gh - bh + 1, : gw - bw + 1] = ok[bh // 2: gh - bh + 1 + bh // 2, bw // 2: gw - bw + 1 + bw // 2] > 0
        for r, c in zip(*np.nonzero(fits), strict=True):
            if against is None:
                out.append((int(r), int(c), rot))
                continue
            back = {0: wall[max(r - 2, 0):r, c:c + bw], 1: wall[r:r + bh, c + bw:c + bw + 2],
                    2: wall[r + bh:r + bh + 2, c:c + bw], 3: wall[r:r + bh, max(c - 2, 0):c]}[rot]
            if back.size and back.any(axis=0 if rot % 2 == 0 else 1).mean() > 0.8:
                out.append((int(r), int(c), rot))
    return out


def path_ok(occupied):
    """Whether, with these cells taken, a path 2 * PASSAGE wide joins the entrance to every other door."""
    from scipy import ndimage as ndi

    floor, wall, _, targets, _, mpc = furnishing_grid()
    free = floor & ~occupied
    clear = ndi.distance_transform_edt(free) * mpc >= PASSAGE
    lab, _ = ndi.label(clear)
    near = [lab[max(t[0] - 8, 0):t[0] + 9, max(t[1] - 8, 0):t[1] + 9] for t in targets]
    reach = [set(np.unique(n[n > 0])) for n in near]
    return bool(reach[0]) and all(reach[0] & other for other in reach[1:])


@cache
def furnish(trials=3000, seed=7):
    """Randomized greedy search over layouts of FURNITURE in the living room. Each trial places the items one by
    one, each at a random position among those where it fits, touches its wall and clashes with nothing placed so
    far (bodies and clearance zones may not overlap other bodies); a layout counts if the walking paths stay open
    (path_ok). Among those, the best keeps the bed far from the entrance, the table near the kitchen and the desk
    near the balcony door, for the light."""
    floor, _, _, targets, kitchen, mpc = furnishing_grid()
    rng = np.random.default_rng(seed)
    options = {name: candidates(name) for name, *_ in FURNITURE}
    entrance = np.array(targets[0]) * mpc
    balcony = np.array(targets[1]) * mpc  # the doors after the entrance come in plan order: balcony first
    best, best_cost, passed = None, np.inf, 0
    for _ in range(trials):
        layout, bodies = {}, {}
        for name in options:
            order = rng.permutation(len(options[name]))
            for k in order[:400]:
                where = options[name][k]
                body, zone = item_boxes(name, *where, mpc)
                if all(not _overlap(body, b) and not _overlap(body, z) and not _overlap(zone, b)
                       for b, z in bodies.values()):
                    layout[name], bodies[name] = where, (body, zone)
                    break
            else:
                break
        if len(layout) < len(options):
            continue
        occupied = np.zeros_like(floor)
        for (r0, c0, r1, c1), _ in bodies.values():
            occupied[r0:r1, c0:c1] = True
        if not path_ok(occupied):
            continue
        passed += 1

        def centre(name, bodies=bodies):
            r0, c0, r1, c1 = bodies[name][0]
            return np.array([(r0 + r1) / 2, (c0 + c1) / 2]) * mpc

        cost = (-np.linalg.norm(centre("bed") - entrance) + 0.8 * np.linalg.norm(centre("table") - kitchen * mpc)
                + 0.5 * np.linalg.norm(centre("desk") - balcony))
        if cost < best_cost:
            best, best_cost = layout, cost
    if best is None:
        raise RuntimeError("no layout passed")
    placed = []
    for name, (r, c, rot) in best.items():
        (r0, c0, r1, c1), (z0, z1, z2, z3) = item_boxes(name, r, c, rot, mpc)
        placed.append({"name": name, "px": (c0 * CELL, r0 * CELL, c1 * CELL, r1 * CELL),
                       "front": (z1 * CELL, z0 * CELL, z3 * CELL, z2 * CELL), "rot": rot,
                       "height": _item(name)[3]})
    return placed, best_cost, passed


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- a scanned copy

SCAN_SHIFT = np.array([[18, 10], [-6, 22], [-20, -8], [8, -16]])  # where the photo moved each corner, px


def _scan_transform():
    from skimage.transform import ProjectiveTransform

    h, w = image().shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    return ProjectiveTransform.from_estimate(src + SCAN_SHIFT, src)


@cache
def scanned(seed=4):
    """The plan as a phone photo of a printout: tilted, slightly in perspective, with the phone's soft shadow on
    one corner, blurred, with sensor noise and handwritten notes. Returns an RGB array the same size."""
    from scipy import ndimage as ndi
    from skimage.transform import warp

    rng = np.random.default_rng(seed)
    a = image().mean(-1)
    h, w = a.shape
    g = warp(a, _scan_transform(), output_shape=(h, w), cval=1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.86 + 0.14 * np.exp(-(((xx - 0.3 * w) / (0.9 * w)) ** 2 + ((yy - 0.25 * h) / (0.9 * h)) ** 2))
    shadow = 1 - 0.55 * np.clip(1.6 - np.hypot((xx - 0.95 * w) / (0.55 * w), (yy - 0.95 * h) / (0.32 * h)), 0, 1)
    g = g * light * shadow
    # handwriting: wobbly strokes of numbers and arrows, drawn as random walks in pen colour
    pen = np.zeros_like(g)
    for _ in range(9):
        y, x = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
        angle = rng.uniform(0, 2 * np.pi)
        for _step in range(int(rng.integers(40, 120))):
            angle += rng.normal(0, 0.35)
            y, x = y + 1.2 * np.sin(angle), x + 1.2 * np.cos(angle)
            if 0 <= int(y) < h and 0 <= int(x) < w:
                pen[int(y), int(x)] = 1
    pen = ndi.binary_dilation(pen, iterations=1)
    g = np.where(pen, 0.35, g)
    g = ndi.gaussian_filter(g, 1.1)
    g = g + rng.normal(0, 0.035, g.shape)
    return np.clip(np.stack([g * 0.98, g * 0.96, g * 0.9], -1), 0, 1)


def walls_in_scan():
    """The clean plan's walls carried into the photo's geometry, to compare like with like."""
    from skimage.transform import warp

    _, walls, _ = masks()
    return warp(walls.astype(float), _scan_transform(), output_shape=walls.shape, order=0, cval=0) > 0.5


def scanned_walls():
    """The same rules as masks() on the photographed plan: threshold, opening, small pieces dropped."""
    from scipy import ndimage as ndi

    walls = ndi.binary_opening(scanned().min(-1) < DARK, structure=np.ones((5, 5)))
    lab, n = ndi.label(walls)
    sizes = ndi.sum(walls, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sizes > MIN_WALL)[0])
