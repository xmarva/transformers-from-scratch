"""A small store simulated with real camera geometry, standing in for the project's data, which stays under NDA.

The store is 12 by 8 m with wall shelves, three gondolas, a fruit table and a fridge wall. Six cameras hang 3 m up
and look down at the floor through pinhole optics. Five shoppers walk between the shelves, stop in front of some of
them and leave. For every camera and every frame the scene is drawn into an ID buffer, so a person hidden behind a
shelf or another person is hidden here too; the detector puts a box around the visible part of each person, with
pixel noise. The bottom middle of the box, where the feet should be, is carried onto the floor plan by the camera's
ground-plane homography, and a Kalman filter with Hungarian matching joins the floor points of all cameras into
tracks.

Units are metres; the floor is z = 0, x runs along the long wall and y from the entrance to the back.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

WIDTH, DEPTH, CEILING = 12.0, 8.0, 3.2
ENTRANCE = (0.8, 2.4)  # the gap in the front wall, along x
FPS = 5  # frames per second the cameras are processed at
IMAGE = (640, 480)

# x0, x1, y0, y1, height, kind
SHELVES = [
    (0.1, 0.6, 2.4, 7.4, 1.9, "wall"),
    (0.1, 11.3, 7.4, 7.9, 1.9, "wall"),
    (11.3, 11.9, 1.2, 7.9, 2.0, "fridge"),
    (2.4, 3.2, 2.2, 6.2, 1.5, "gondola"),
    (4.8, 5.6, 2.2, 6.2, 1.5, "gondola"),
    (7.2, 8.0, 2.2, 6.2, 1.5, "gondola"),
    (9.2, 10.2, 3.0, 5.0, 0.9, "table"),
    (4.0, 10.5, 0.1, 0.6, 1.2, "wall"),
]
WALL_HEIGHT = CEILING
SHELF_NAMES = ["left wall shelf", "back wall shelf", "fridges", "gondola 1", "gondola 2", "gondola 3", "fruit table",
               "front shelf"]

AISLES = [1.5, 4.0, 6.4, 8.6, 10.75]  # x of the aisles' middles
# position, the floor point it looks at: a camera at each end of every aisle, looking down it, and two over the
# entrance and the front corridor
CAMERAS = [((x, 1.0, CEILING), (x, 4.2)) for x in AISLES] + [((x, 7.0, CEILING), (x, 3.8)) for x in AISLES] + [
    ((1.6, 2.4, CEILING), (1.6, 0.0)),
    ((6.0, 2.0, CEILING), (9.5, 1.2)),
]
FOCAL = 300.0  # pixels, for a 640-pixel-wide image: about 94 degrees across
HEIGHT_GUESS = 1.72  # m: the height assumed for a person when only the head can be trusted

# each shopper: start time (s), walking speed (m/s), height (m), a sideways offset that keeps people apart in the
# aisles, and the waypoints; a third number at a waypoint is how long the person stands there (s)
SHOPPERS = [
    (0.0, 1.0, 1.78, (-0.25, 0.15), [(1.6, -0.6), (1.6, 1.4), (4.0, 1.4), (4.0, 4.2, 4), (4.0, 6.8), (6.4, 6.8),
                                      (6.4, 3.4, 3), (6.4, 1.4), (1.6, 1.4), (1.6, -0.6)]),
    (3.0, 1.15, 1.70, (0.25, -0.2), [(1.6, -0.6), (1.6, 1.4), (10.7, 1.4), (10.7, 5.0, 5), (10.7, 6.8), (8.6, 6.8),
                                      (8.6, 4.0, 3), (8.6, 1.4), (1.6, 1.4), (1.6, -0.6)]),
    (7.0, 0.95, 1.65, (0.1, 0.3), [(1.6, -0.6), (1.5, 1.4), (1.5, 5.4, 4), (1.5, 6.8), (4.0, 6.8), (4.0, 2.8, 3),
                                    (4.0, 1.4), (1.6, 1.4), (1.6, -0.6)]),
    (12.0, 1.05, 1.84, (-0.2, -0.25), [(1.6, -0.6), (1.6, 1.4), (6.4, 1.4), (6.4, 6.8), (10.6, 6.8, 3),
                                        (10.6, 2.6, 3), (10.6, 1.4), (1.6, 1.4), (1.6, -0.6)]),
    (20.0, 1.1, 1.62, (0.3, 0.1), [(1.6, -0.6), (1.6, 1.4), (8.6, 1.4), (8.6, 2.9, 4), (8.6, 1.4), (1.6, 1.4),
                                    (1.6, -0.6)]),
]


# ---------------------------------------------------------------- cameras

@dataclass(frozen=True)
class Camera:
    position: tuple
    target: tuple

    @property
    def R(self):
        """Rows: the camera's right, down and forward axes in world coordinates."""
        pos = np.array(self.position, float)
        forward = np.array([*self.target, 0.0]) - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return np.stack([right, np.cross(forward, right), forward])

    @property
    def K(self):
        w, h = IMAGE
        return np.array([[FOCAL, 0, w / 2], [0, FOCAL, h / 2], [0, 0, 1.0]])

    def to_camera(self, points):
        return (np.asarray(points, float) - np.array(self.position)) @ self.R.T

    def project(self, points):
        """World points to pixels; points behind the camera come out as NaN."""
        c = self.to_camera(points)
        z = np.where(c[..., 2] > 1e-6, c[..., 2], np.nan)
        return np.stack([FOCAL * c[..., 0] / z + IMAGE[0] / 2, FOCAL * c[..., 1] / z + IMAGE[1] / 2], -1)

    def H(self, z=0.0):
        """The homography of the horizontal plane at height z: (x, y, 1) on that plane to pixels (u, v, 1), up to
        scale. z = 0 is the floor."""
        R, t = self.R, -self.R @ np.array(self.position)
        return self.K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + t])

    def to_floor(self, pixels, z=0.0):
        """Pixels back to (x, y) on the plane at height z."""
        q = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self.H(z)).T
        return q[:, :2] / q[:, 2:3]


@cache
def cameras():
    return [Camera(tuple(p), tuple(t)) for p, t in CAMERAS]


# ---------------------------------------------------------------- shoppers

def _route(waypoints, offset):
    points, stops = [], {}
    for k, w in enumerate(waypoints):
        x, y = w[0], w[1]
        if 0 < k < len(waypoints) - 1:  # keep the door on the door; shift the rest so people don't walk in step
            x, y = x + offset[0], y + offset[1]
        points.append((x, y))
        if len(w) > 2:
            stops[k] = w[2]
    return np.array(points), stops


def _smooth(points, rounds=3):
    """Chaikin corner cutting: corners become curves, the ends stay put."""
    for _ in range(rounds):
        out = [points[0]]
        for a, b in zip(points[:-1], points[1:], strict=True):
            out += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        out.append(points[-1])
        points = np.array(out)
    return points


@cache
def walks():
    """For each shopper an array of (t, x, y) at the camera frame rate, while they are in or at the store."""
    out = []
    for start, speed, _, offset, waypoints in SHOPPERS:
        points, stops = _route(waypoints, offset)
        # walk leg by leg so that a stop happens exactly at its waypoint
        samples, t = [], start
        for k in range(len(points) - 1):
            leg = _smooth(np.array([points[max(k - 1, 0)] * 0.15 + points[k] * 0.85, points[k], points[k + 1],
                                    points[k + 1] * 0.85 + points[min(k + 2, len(points) - 1)] * 0.15]), 3)
            leg = leg[1:-1] if 0 < k < len(points) - 2 else leg
            seg = np.linalg.norm(np.diff(leg, axis=0), axis=1)
            s = np.r_[0, np.cumsum(seg)]
            if k in stops:
                for _ in range(int(stops[k] * FPS)):
                    samples.append((t, *points[k]))
                    t += 1 / FPS
            for d in np.arange(0, s[-1], speed / FPS):
                samples.append((t, np.interp(d, s, leg[:, 0]), np.interp(d, s, leg[:, 1])))
                t += 1 / FPS
        samples.append((t, *points[-1]))
        walk = np.array(samples)
        walk[:, 0] = np.round(walk[:, 0] * FPS) / FPS
        out.append(walk)
    return out


def frames():
    end = max(w[-1, 0] for w in walks())
    return np.round(np.arange(0, end + 1e-9, 1 / FPS) * FPS) / FPS


def positions(t):
    """{shopper: (x, y)} at time t, for those present."""
    out = {}
    for i, walk in enumerate(walks()):
        k = np.searchsorted(walk[:, 0], t - 1e-6)
        if k < len(walk) and abs(walk[k, 0] - t) < 1e-6:
            out[i] = walk[k, 1:]
    return out


# ---------------------------------------------------------------- the scene as polygons

PRODUCT_COLOURS = ["#C9573E", "#E2A93B", "#5C8F5A", "#3F6FA6", "#B9586F", "#D9C7A3", "#7A5C99", "#E07B3C",
                   "#4E9C9A", "#A3473D"]
SHIRTS = ["#D23F57", "#0F8C7E", "#C27A0E", "#6A55D8", "#5B8FD6"]  # the shoppers' colours in the figures


def _box_faces(x0, x1, y0, y1, h):
    c = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, h], [x1, y0, h], [x1, y1, h],
                  [x0, y1, h]], float)
    faces = {"bottom": [0, 3, 2, 1], "top": [4, 5, 6, 7], "front": [0, 1, 5, 4], "right": [1, 2, 6, 5],
             "back": [2, 3, 7, 6], "left": [3, 0, 4, 7]}
    return {name: c[idx] for name, idx in faces.items()}


@cache
def static_scene():
    """(polygon in world coordinates, colour, kind) for walls, shelves with their product rows, and floor tiles."""
    rng = np.random.default_rng(3)
    polys = []
    for i in range(int(WIDTH)):
        for j in range(int(DEPTH)):
            tile = np.array([[i, j, 0], [i + 1, j, 0], [i + 1, j + 1, 0], [i, j + 1, 0]], float)
            polys.append((tile, "#E7E3DC" if (i + j) % 2 else "#D9D4CB", "floor"))
    t = 0.12
    walls = [(0, WIDTH, DEPTH, DEPTH + t), (-t, 0, 0, DEPTH), (WIDTH, WIDTH + t, 0, DEPTH),
             (0, ENTRANCE[0], -t, 0), (ENTRANCE[1], WIDTH, -t, 0)]
    for x0, x1, y0, y1 in walls:
        for name, face in _box_faces(x0, x1, y0, y1, WALL_HEIGHT).items():
            if name != "bottom":
                polys.append((face, "#C9D0D8", "wall"))
    for x0, x1, y0, y1, h, kind in SHELVES:
        body = {"fridge": "#CFDCE3", "table": "#B98C5E"}.get(kind, "#8C8F96")
        for name, face in _box_faces(x0, x1, y0, y1, h).items():
            if name == "bottom":
                continue
            polys.append((face, body if name != "top" else "#A6A9AF", "shelf"))
            if name == "top" or kind == "table":
                continue
            # rows of products on the long faces, as coloured strips a little in front of the face
            a, b = face[0], face[1]
            length = np.linalg.norm(b[:2] - a[:2])
            if length < 0.9:
                continue
            centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0])
            out = (a + b) / 2 - centre
            out[2] = 0
            out = out / np.linalg.norm(out) * 0.01
            levels = np.linspace(0.15, h - 0.12, 5)
            for lo, hi in zip(levels[:-1], levels[1:], strict=True):
                steps = np.arange(0, length, 0.45)
                for s0 in steps:
                    s1 = min(s0 + 0.42, length)
                    p0 = a + (b - a) * (s0 / length)
                    p1 = a + (b - a) * (s1 / length)
                    strip = np.array([[*p0[:2], lo], [*p1[:2], lo], [*p1[:2], hi - 0.06], [*p0[:2], hi - 0.06]])
                    polys.append((strip + out, PRODUCT_COLOURS[rng.integers(len(PRODUCT_COLOURS))], "product"))
    return polys


def person_parts(x, y, height, shirt):
    """A shopper as legs, torso and head: rings of points whose silhouettes are drawn as convex hulls."""
    angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], 1)

    def cylinder(r, z0, z1):
        return np.concatenate([ring * r + [x, y, z0], ring * r + [x, y, z1]])

    head_z = height - 0.12
    head = np.concatenate([ring * 0.11 + [x, y, head_z + dz] for dz in (-0.1, -0.05, 0, 0.05, 0.1)])
    return [(cylinder(0.14, 0.0, 0.86), "#3A3F4B"), (cylinder(0.21, 0.82, height - 0.24), shirt),
            (head, "#5A4636")]


# ---------------------------------------------------------------- drawing one camera's view

def _clip_near(points, near=0.05):
    """Sutherland-Hodgman against the plane z = near in camera coordinates."""
    out = []
    n = len(points)
    for k in range(n):
        a, b = points[k], points[(k + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


def _clip_rect(poly, margin=0.5):
    """Sutherland-Hodgman against the picture grown by a margin on each side, so that a polygon passing right
    under the camera stays a sane shape."""
    w, h = IMAGE
    edges = [(0, -margin * w, 1), (0, (1 + margin) * w, -1), (1, -margin * h, 1), (1, (1 + margin) * h, -1)]
    for axis, limit, sign in edges:
        if len(poly) == 0:
            break
        out = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            ina, inb = sign * (a[axis] - limit) >= 0, sign * (b[axis] - limit) >= 0
            if ina:
                out.append(a)
            if ina != inb:
                t = (limit - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        poly = np.array(out)
    return poly


def _to_pixels(cam_points):
    return np.stack([FOCAL * cam_points[:, 0] / cam_points[:, 2] + IMAGE[0] / 2,
                     FOCAL * cam_points[:, 1] / cam_points[:, 2] + IMAGE[1] / 2], 1)


def _hull(points):
    from scipy.spatial import ConvexHull

    return points[ConvexHull(points).vertices]


@cache
def static_view(cam: Camera):
    """The store as this camera sees it: (depth, pixel polygon, colour, kind), with faces turned away dropped."""
    items = []
    pos = np.array(cam.position)
    for poly, colour, kind in static_scene():
        if kind != "floor":
            normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
            if np.dot(normal, poly.mean(0) - pos) >= 0 and kind != "product":
                continue  # a face turned away
            if kind == "product":
                normal = np.cross(poly[1] - poly[0], poly[3] - poly[0])
                if np.dot(normal, poly.mean(0) - pos) >= 0:
                    continue
        c = _clip_near(cam.to_camera(poly))
        if len(c) < 3:
            continue
        depth = 1e9 if kind == "floor" else c[:, 2].mean() - (0.02 if kind == "product" else 0)
        poly = _clip_rect(_to_pixels(c))
        if len(poly) >= 3:
            items.append((depth, poly, colour, kind))
    return items


def view(cam: Camera, people):
    """The camera's picture as a list of (depth, pixel polygon, colour, owner), farthest first. people: list of
    (id, x, y, height, shirt). owner is the shopper's id, or -1 for the store; kind is what the polygon is."""
    items = [(d, poly, colour, -1, kind) for d, poly, colour, kind in static_view(cam)]
    for pid, x, y, height, shirt in people:
        centre = cam.to_camera([[x, y, height / 2]])[0]
        if centre[2] < 0.3:
            continue
        for k, (pts, colour) in enumerate(person_parts(x, y, height, shirt)):
            c = cam.to_camera(pts)
            c = c[c[:, 2] > 0.05]
            if len(c) < 4:
                continue
            # the parts of one person are drawn together, legs, torso, head
            poly = _clip_rect(_hull(_to_pixels(c)))
            if len(poly) >= 3:
                items.append((centre[2] + 0.003 * (2 - k), poly, colour, pid, "person"))
    items.sort(key=lambda item: -item[0])
    return items


def id_buffer(items, scale=0.5):
    """Who is seen at each pixel, drawn farthest first: shopper id, or -1."""
    from PIL import Image, ImageDraw

    w, h = int(IMAGE[0] * scale), int(IMAGE[1] * scale)
    image = Image.new("I", (w, h), -1)
    draw = ImageDraw.Draw(image)
    for _, poly, _, owner, kind in items:
        if kind in ("floor", "product"):
            continue  # neither hides anyone
        p = poly * scale
        draw.polygon([tuple(q) for q in p], fill=int(owner))
    return np.array(image)


# ---------------------------------------------------------------- detection, the floor, tracking

PIXEL_NOISE = 2.5
MIN_VISIBLE = 0.3


def people_at(t):
    return [(i, *xy, SHOPPERS[i][2], SHIRTS[i]) for i, xy in positions(t).items()]


LOOK_DIM, LOOK_NOISE = 32, 0.1  # a stand-in for a re-identification network's embedding of the person's crop


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _feet_in_view(cam, box, floor_from_head):
    """Whether the box reaches as low as a whole person standing where the head says would: if not, the bottom of
    the box is not the feet (a shelf, a cart or the image's edge cut it off)."""
    top = cam.project(np.array([[*floor_from_head, HEIGHT_GUESS]]))[0, 1]
    bottom = cam.project(np.array([[*floor_from_head, 0.0]]))[0, 1]
    return box[3] >= bottom - 0.15 * (bottom - top) and box[3] < IMAGE[1] - 4


@cache
def detections():
    """Per frame and camera: list of dicts with the shopper id (for scoring only), the box, the floor point from the
    bottom of the box (feet), the floor point from its top (head, assuming a height of HEIGHT_GUESS), the point the
    tracker uses, and whether the box looked like a whole person."""
    rng = np.random.default_rng(11)
    looks = _unit(np.random.default_rng(5).normal(size=(len(SHOPPERS), LOOK_DIM)))
    out = {}
    scale = 0.5
    for t in frames():
        people = people_at(t)
        for c, cam in enumerate(cameras()):
            items = view(cam, people)
            ids = id_buffer(items, scale)
            found = []
            for pid, x, y, height, _shirt in people:
                alone = id_buffer([item for item in items if item[3] == pid], scale)
                full = (alone == pid).sum()
                seen = ids == pid
                if full < 40 or seen.sum() < MIN_VISIBLE * full:
                    continue
                rows, cols = np.nonzero(seen)
                box = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], float) / scale
                if box[1] <= 2:
                    continue  # the head is out of the picture: nothing reliable to place
                box += rng.normal(0, PIXEL_NOISE, 4)
                centre = (box[0] + box[2]) / 2
                from_feet = cam.to_floor(np.array([[centre, box[3]]]))[0]
                from_head = cam.to_floor(np.array([[centre, box[1]]]), HEIGHT_GUESS)[0]
                whole = _feet_in_view(cam, box, from_head)
                look = _unit(looks[pid] + rng.normal(0, LOOK_NOISE, LOOK_DIM))
                found.append({"id": pid, "box": box, "from_feet": from_feet, "from_head": from_head,
                              "floor": from_feet if whole else from_head, "whole": bool(whole), "look": look,
                              "visible": seen.sum() / full, "truth": np.array([x, y]), "height": height})
            out[(round(t, 3), c)] = found
    return out


class Track:
    def __init__(self, tid, xy, t, look):
        self.id, self.x, self.P = tid, np.array([*xy, 0.0, 0.0]), np.diag([0.3, 0.3, 1.0, 1.0])
        self.hits, self.missed, self.history, self.confirmed = 1, 0, [(t, *xy)], False
        self.look, self.together = look, 0

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = 0.8  # how much a shopper's velocity may change, (m/s^2)^2
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q

    def update(self, z, r=0.09):
        Hm = np.eye(2, 4)
        S = Hm @ self.P @ Hm.T + np.eye(2) * r
        gain = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + gain @ (z - Hm @ self.x)
        self.P = (np.eye(4) - gain @ Hm) @ self.P


GATE = 1.2  # metres between a predicted track and a floor point that may belong to it
DOOR = np.array([sum(ENTRANCE) / 2, 0.0])
DOOR_RADIUS = 1.6  # people appear and disappear only this close to the door
PATIENCE = 12  # seconds a track is kept without detections away from the door: the person is hidden, not gone


@cache
def tracks(appearance=True):
    """Kalman filter (constant velocity) per track; every camera's floor points are matched to the predicted tracks
    with the Hungarian method, and a track is updated with the median of the points it got, so that one camera that
    sees only the upper body cannot drag it away. New tracks start, and finished ones end, only at the door. With
    appearance, the matching cost adds how unlike the track's running appearance a detection looks, as DeepSORT does,
    and two tracks that walk together for a second are merged."""
    from scipy.optimize import linear_sum_assignment

    found = detections()
    live, done, next_id = [], [], 0
    dt = 1 / FPS
    for t in frames():
        t = round(t, 3)
        for track in live:
            track.predict(dt)
            if track.missed:  # a person who is not seen is more likely standing than walking on
                track.x[2:] *= 0.6
        got = {track.id: [] for track in live}
        loose = []
        looks = {track.id: [] for track in live}
        # how far a track may be from a detection: a person can walk about a metre a second while unseen
        gates = np.array([GATE + (track.missed / FPS if appearance else 0) for track in live])
        for c in range(len(cameras())):
            dets = found[(t, c)]
            if not dets:
                continue
            if live:
                dist = np.array([[np.linalg.norm(d["floor"] - track.x[:2]) for track in live] for d in dets])
                cost = dist / gates
                if appearance:
                    cost = cost + 2 * np.array([[1 - d["look"] @ track.look for track in live] for d in dets])
                rows, cols = linear_sum_assignment(cost)
                taken = set()
                for r, k in zip(rows, cols, strict=True):
                    similar = dets[r]["look"] @ live[k].look
                    if dist[r, k] < gates[k] and (not appearance or similar > (0.8 if live[k].missed else 0.5)):
                        got[live[k].id].append(dets[r]["floor"])
                        looks[live[k].id].append(dets[r]["look"])
                        taken.add(r)
                loose += [(d["floor"], d["look"], c) for r, d in enumerate(dets) if r not in taken]
            else:
                loose += [(d["floor"], d["look"], c) for d in dets]
        for track in live:
            if got[track.id]:
                track.update(np.median(np.array(got[track.id]), axis=0))
                track.look = _unit(0.9 * track.look + 0.1 * np.mean(looks[track.id], axis=0))
                track.hits += 1
                track.missed = 0
                track.confirmed = track.confirmed or track.hits >= 3
            else:
                track.missed += 1
            track.history.append((t, *track.x[:2]))
        # new tracks from floor points near the door that are not near any track: group those within half a metre
        # a new person: points near the door, not near any track, that at least two cameras agree on
        loose = [item for item in loose if np.linalg.norm(item[0] - DOOR) < DOOR_RADIUS]
        while loose:
            seed, look, cam = loose.pop(0)
            group = [(seed, cam)] + [(p, c) for p, _, c in loose if np.linalg.norm(p - seed) < 0.6]
            loose = [item for item in loose if np.linalg.norm(item[0] - seed) >= 0.6]
            where = np.median(np.array([p for p, _ in group]), 0)
            if len({c for _, c in group}) < 2 or any(np.linalg.norm(where - tr.x[:2]) < GATE for tr in live):
                continue
            live.append(Track(next_id, where, t, look))
            next_id += 1
        if appearance:  # two tracks that walk together and look alike are one person
            for a in live:
                for b in live:
                    if a.id < b.id and a.confirmed and b.confirmed:
                        close = np.linalg.norm(a.x[:2] - b.x[:2]) < 0.6 and a.look @ b.look > 0.6
                        b.together = b.together + 1 if close else 0
                        if b.together > FPS:
                            b.missed = 10 ** 6
        keep = []
        for track in live:
            at_door = np.linalg.norm(track.x[:2] - DOOR) < DOOR_RADIUS
            if (not track.confirmed and track.missed > 2) or (track.missed > FPS and at_door) \
                    or track.missed > PATIENCE * FPS:
                done.append(track)
            else:
                keep.append(track)
        live = keep
    done += live
    return [tr for tr in done if tr.confirmed]


@cache
def scores(appearance=True):
    """CLEAR MOT style (Bernardin and Stiefelhagen, 2008): each frame, a shopper keeps last frame's track if it is
    still within 1 m, the rest are matched with the Hungarian method. Returns the mean and 95th percentile position
    error, the share of shopper-frames inside the store that had a track, and the number of identity switches."""
    from scipy.optimize import linear_sum_assignment

    by_time = {}
    for tr in tracks(appearance):
        for t, x, y in tr.history:
            by_time.setdefault(round(t, 3), {})[tr.id] = np.array([x, y])
    errors, present, matched, last, switches = [], 0, 0, {}, 0
    for t in frames():
        t = round(t, 3)
        truth = {i: xy for i, xy in positions(t).items() if xy[1] > 0.2}  # inside the store
        present += len(truth)
        cands = dict(by_time.get(t, {}))
        pairs = {}
        for i, xy in truth.items():  # keep last frame's pairs that still hold
            tid = last.get(i)
            if tid in cands and np.linalg.norm(cands[tid] - xy) < 1.0 and tid not in pairs.values():
                pairs[i] = tid
        rest = [i for i in truth if i not in pairs]
        free = [tid for tid in cands if tid not in pairs.values()]
        if rest and free:
            cost = np.array([[np.linalg.norm(truth[i] - cands[tid]) for tid in free] for i in rest])
            rows, cols = linear_sum_assignment(cost)
            for r, k in zip(rows, cols, strict=True):
                if cost[r, k] < 1.0:
                    pairs[rest[r]] = free[k]
        for i, tid in pairs.items():
            matched += 1
            errors.append(np.linalg.norm(truth[i] - cands[tid]))
            if i in last and last[i] != tid:
                switches += 1
            last[i] = tid
    errors = np.array(errors)
    return {"mean": float(errors.mean()), "p95": float(np.percentile(errors, 95)), "tracked": matched / present,
            "switches": switches, "tracks": len(tracks(appearance))}


# ---------------------------------------------------------------- where people stopped

STOP_SPEED, STOP_TIME = 0.3, 1.5  # m/s and s: slower than this for longer than this is a stop


def nearest_shelf(xy):
    def gap(shelf):
        x0, x1, y0, y1 = shelf[:4]
        return np.hypot(max(x0 - xy[0], 0, xy[0] - x1), max(y0 - xy[1], 0, xy[1] - y1))

    return SHELF_NAMES[int(np.argmin([gap(sh) for sh in SHELVES]))]


def _runs(times, slow):
    out, k = [], 0
    while k < len(slow):
        if slow[k]:
            j = k
            while j < len(slow) and slow[j]:
                j += 1
            if times[j - 1] - times[k] >= STOP_TIME - 1e-6:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


@cache
def stops_found():
    """Stops read from each track: its speed, smoothed over a second, stays below STOP_SPEED for STOP_TIME.
    Returns (track id, start, end, (x, y), nearest shelf)."""
    from scipy.ndimage import uniform_filter1d

    out = []
    for tr in tracks():
        h = np.array(tr.history)
        if len(h) < FPS * 2:
            continue
        xy = uniform_filter1d(h[:, 1:], size=FPS, axis=0)
        speed = np.linalg.norm(np.gradient(xy, axis=0), axis=1) * FPS
        for a, b in _runs(h[:, 0], speed < STOP_SPEED):
            where = xy[a:b].mean(0)
            out.append((tr.id, float(h[a, 0]), float(h[b - 1, 0]), where, nearest_shelf(where)))
    return out


@cache
def stops_true():
    """The stops in the script: (shopper, start, end, (x, y), nearest shelf)."""
    out = []
    for i, walk in enumerate(walks()):
        still = np.r_[False, np.all(np.diff(walk[:, 1:], axis=0) == 0, axis=1)]
        for a, b in _runs(walk[:, 0], still):
            where = walk[a, 1:]
            out.append((i, float(walk[max(a - 1, 0), 0]), float(walk[b - 1, 0]), where, nearest_shelf(where)))
    return out
