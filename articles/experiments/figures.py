"""16:9 covers for the experiments on the portfolio page of xmarva.github.io.

Each card takes one real picture from the project (a detection result, a structure render, a segmentation, a camera
frame), cuts the object out or recolours it into the site's palette, and sets it on the dark background with the dot
grid used by the site's covers. The palette is the one the posts' figures use on dark (amber, teal, rose, violet) plus
the brand red, and where the source picture uses colour to tell things apart, each of its colours maps to one of these.
Sources are in data/source/: the projects' own images, and for the language-model experiment a public-domain title page
from 1902 (Wikimedia Commons, "Cover Lenin Zadachy.jpg").

    uv run --group articles python tools/render.py --article experiments \
        --out ../../websites/xmarva.github.io/assets/img/projects/personal
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from tfs.viz.palette import DARK
from tools.figures import cover as cv

SOURCE = Path(__file__).parent / "data" / "source"
W, H = 800, 450
INK, MUTED, RED = "#C9CCD3", "#8E939E", cv.RED
AMBER, TEAL, ROSE, VIOLET = DARK["data"], DARK["enc"], DARK["dec"], DARK["att"]


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def _load(name):
    return np.asarray(Image.open(SOURCE / name).convert("RGB"), dtype=float) / 255


def _blur(a, radius):
    """Gaussian blur of a float image or mask in [0, 1]."""
    image = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=float) / 255


def _uri(rgba: np.ndarray) -> str:
    image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _card(objects: str, glow: str = RED) -> str:
    """The site's dark cover background with its dot grid, a faint glow, and a soft shadow filter for objects
    that are 'things' on the table."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs>'
            f'<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern>'
            f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            f'<feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#000" flood-opacity="0.55"/></filter>'
            f'<radialGradient id="glow" cx="50%" cy="50%" r="55%"><stop offset="0" stop-color="{glow}" '
            f'stop-opacity="0.10"/><stop offset="1" stop-color="{glow}" stop-opacity="0"/></radialGradient></defs>'
            f'<rect width="{W}" height="{H}" fill="{cv.BG}"/><rect width="{W}" height="{H}" fill="url(#dots)"/>'
            f'<rect width="{W}" height="{H}" fill="url(#glow)"/>{objects}</svg>')


def _place(rgba, x, y, width, filter_=None, rotate=0.0):
    h, w = rgba.shape[:2]
    height = width * h / w
    f = f' filter="url(#{filter_})"' if filter_ else ""
    t = f' transform="rotate({rotate} {x + width / 2:.1f} {y + height / 2:.1f})"' if rotate else ""
    return f'<image href="{_uri(rgba)}" x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}"{f}{t}/>'


def _framed(rgb, radius=10):
    """An opaque picture with rounded corners (alpha mask)."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dx = np.maximum(np.maximum(radius - xx, xx - (w - 1 - radius)), 0)
    dy = np.maximum(np.maximum(radius - yy, yy - (h - 1 - radius)), 0)
    alpha = np.clip(radius + 0.5 - np.hypot(dx, dy), 0, 1)
    return np.dstack([rgb, alpha])


def _ramp(t, stops):
    """Piecewise-linear colour ramp through `stops` (hex colours spaced evenly over t in [0, 1])."""
    stops = np.array([_rgb(s) for s in stops])
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


# ---------------------------------------------------------------- floor plans: the model's detections, on dark

def card_floorplan():
    from scipy import ndimage

    img = _load("floorplan-detections.png")
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    grey = img.mean(axis=2)
    green = (g - np.maximum(r, b) > 0.08) & (g > 0.25)  # the detection boxes
    out = np.zeros(img.shape[:2] + (4,))
    out[..., :3] = _rgb(INK)
    out[..., 3] = np.clip((0.92 - grey) / 0.7, 0, 1) * 0.9  # plan lines: the darker, the more opaque
    out[green, :3] = _rgb(TEAL)
    out[green, 3] = 0.8
    # the score labels: grey (51, 51, 50) rectangles with light text become coloured tags with dark text, amber for
    # rooms (the large labels) and rose for walls
    fill = ndimage.binary_opening(np.abs(grey - 51 / 255) < 5 / 255, structure=np.ones((2, 2)))
    labels, _ = ndimage.label(ndimage.binary_closing(fill, structure=np.ones((11, 25))))  # bridge the letters
    for sl in ndimage.find_objects(labels):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 5 <= h <= 40 and w > h:
            tag = _rgb(AMBER if w >= 100 else ROSE)
            text = np.clip((grey[sl] - 0.25) / 0.35, 0, 1)[..., None]
            out[sl[0], sl[1], :3] = tag * (1 - text) + _rgb(cv.BG) * text
            out[sl[0], sl[1], 3] = 1.0
    height = 420
    width = height * img.shape[1] / img.shape[0]
    return _card(_place(out, (W - width) / 2, (H - height) / 2, width), glow=TEAL)


# ---------------------------------------------------------------- proteins: ribbon renders, rainbow to the palette

PANELS = {  # crops of the project's proteins.png (x0, y0, x1, y1): the renders without their titles
    "beta barrel": (700, 150, 1200, 600),
    "alpha-beta barrel": (740, 780, 1180, 1220),
    "alpha-beta roll": (180, 820, 660, 1260),
}


def _ribbons(crop):
    from matplotlib.colors import rgb_to_hsv

    hsv = rgb_to_hsv(crop)
    s, v = hsv[..., 1], hsv[..., 2]
    alpha = np.clip((s - 0.12) / 0.25, 0, 1)
    alpha = np.maximum(alpha, np.clip((0.75 - v) / 0.3, 0, 1))  # dark edges of the ribbons
    t = np.clip(1 - ((hsv[..., 0] + 0.05) % 1.0) / 0.80, 0, 1)  # the render's rainbow: violet (N-terminus) ... red (C)
    colour = _ramp(t, [VIOLET, TEAL, AMBER, ROSE])
    shade = 0.35 + 0.75 * v[..., None]
    return np.dstack([np.clip(colour * shade, 0, 1), alpha])


def card_proteins():
    img = _load("proteins.png")
    parts = []
    width = 250
    for (x0, y0, x1, y1), x in zip(PANELS.values(), [20, 275, 530], strict=True):
        height = width * (y1 - y0) / (x1 - x0)
        parts.append(_place(_ribbons(img[y0:y1, x0:x1]), x, (H - height) / 2, width))
    return _card("".join(parts), glow=VIOLET)


# ---------------------------------------------------------------- computer vision tasks: SAM masks on curved surfaces

def card_cv():
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage

    img = _load("cv-sam-masks.png")[33:443, 10:630]  # the picture without its title
    bg = np.all(np.abs(img - 63 / 255) < 6 / 255, axis=2)
    labels, _ = ndimage.label(bg)
    edge = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    background = np.isin(labels, edge[edge > 0])
    hsv = rgb_to_hsv(img)
    mask = hsv[..., 1] > 0.3
    # the three kinds of curved surface, told apart by the median hue of each SAM mask: violet corner pockets
    # (0.76), pale magenta top cuts (0.81) and magenta quarter-round steps (0.88)
    masks, n = ndimage.label(ndimage.binary_closing(mask, structure=np.ones((5, 5))))
    hue = ndimage.median(hsv[..., 0], masks, index=np.arange(1, n + 1))
    out = np.dstack([img * 1.05, np.ones(img.shape[:2])])
    for i, h in enumerate(hue, start=1):
        kind = (masks == i) & mask
        colour = AMBER if h < 0.785 else TEAL if h < 0.845 else ROSE
        out[kind, :3] = _rgb(colour) * (0.35 + 0.8 * hsv[kind, 2][:, None])
    out[background, 3] = 0
    return _card(_place(out, 70, 18, 660, "shadow"), glow=TEAL)


# ---------------------------------------------------------------- geology: the layered section with the well path

def card_geology():
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage

    img = _load("geology-section.png")[262:880, 102:1298]  # the cross-section without axes or the log label
    hsv = rgb_to_hsv(img)
    hue, sat = hsv[..., 0], hsv[..., 1]
    lum = img @ np.array([0.299, 0.587, 0.114])
    # the layers in dark greys with a cool cast, the deepest the darkest
    t = np.clip((lum - 0.4) / 0.55, 0, 1)[..., None]
    rgb = _rgb("#16181D") * (1 - t) + _rgb("#3D434E") * t
    well = lum < 0.18  # the black well path and the section's outlines
    dashed = (sat < 0.15) & (lum > 0.25) & (lum < 0.55)  # the grey dashed line
    horizons = (sat > 0.45) & ~well
    for kind, colour in [((hue < 0.03) | (hue > 0.95), ROSE), ((hue > 0.3) & (hue < 0.5), TEAL),
                         ((hue >= 0.5) & (hue < 0.75), VIOLET)]:  # the red, green and blue horizons
        rgb[ndimage.binary_dilation(horizons & kind)] = _rgb(colour)
    rgb[dashed] = _rgb(AMBER)
    rgb[ndimage.binary_dilation(well)] = _rgb("#F2F3F5")
    framed = _framed(rgb, radius=14)
    return _card(_place(framed, 60, (H - 680 * framed.shape[0] / framed.shape[1]) / 2, 680, "shadow"), glow=VIOLET)


# ---------------------------------------------------------------- pandemic vision: a camera frame with the alerts

def card_pandemic():
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage

    img = _load("pandemic-dashboard.png")[318:618, 626:1176]  # the "distance exceeded" camera frame, no overlay text
    hsv = rgb_to_hsv(img)
    hue = hsv[..., 0]
    vivid = (hsv[..., 1] > 0.6) & (hsv[..., 2] > 0.5)
    alert = ndimage.binary_closing(vivid & ((hue < 0.03) | (hue > 0.97)), structure=np.ones((7, 7)))  # red boxes
    fine = ndimage.binary_closing(vivid & (hue > 0.25) & (hue < 0.42), structure=np.ones((7, 7)))  # green boxes
    boxes = alert | fine
    # blur the head in each detection box, so the cover shows the alerts, not who was in the shop
    labels, _ = ndimage.label(ndimage.binary_dilation(boxes, iterations=2))
    heads = np.zeros(img.shape[:2])
    for sl in ndimage.find_objects(labels):  # the top third of each person's box, feathered
        if (sl[0].stop - sl[0].start) > 60 and (sl[1].stop - sl[1].start) > 30:
            top = sl[0].start + (sl[0].stop - sl[0].start) // 3
            heads[sl[0].start:top, sl[1]] = 1
    heads = _blur(heads, 6)[..., None]
    base = heads * _blur(img, 9) + (1 - heads) * img
    lum = base @ np.array([0.299, 0.587, 0.114])
    rgb = _rgb("#121214") + (_rgb("#9DA3AE") - _rgb("#121214")) * np.clip(lum, 0, 1)[..., None]
    rgb[ndimage.binary_dilation(fine)] = _rgb(TEAL)
    rgb[ndimage.binary_dilation(alert)] = _rgb(RED)
    width = 580
    return _card(_place(_framed(rgb, radius=12), 110, (H - width * rgb.shape[0] / rgb.shape[1]) / 2, width, "shadow"))


# ---------------------------------------------------------------- red russian: a 1902 title page, tokens marked

LINES = [  # rows of the scan (y0, y1) and the marker colour: the page as a language model reads it, in tokens
    ((100, 117), VIOLET),  # Н. ЛЕНИНЪ.
    ((157, 183), TEAL),  # ЗАДАЧИ
    ((210, 251), AMBER),  # РУССКИХЪ
    ((281, 365), ROSE),  # СОЦIАЛЬДЕМОКРАТОВЪ
]


def card_book():
    scan = np.asarray(Image.open(SOURCE / "lenin-zadachy-1902.jpg").convert("L"), dtype=float) / 255
    ink = np.clip((0.85 - scan) / 0.6, 0, 1)[..., None]
    paper = np.broadcast_to(_rgb("#F2F3F5"), ink.shape[:2] + (3,)).copy()
    inner = ink[:, 40:-40, 0]
    for (y0, y1), colour in LINES:  # a highlighter over the text of the line, from its first to its last letter
        xs = np.flatnonzero(inner[y0:y1].mean(axis=0) > 0.05) + 40
        paper[y0 - 4:y1 + 4, xs.min() - 6:xs.max() + 6] = _rgb("#F2F3F5") * 0.55 + _rgb(colour) * 0.45
    ink_colour = np.broadcast_to(_rgb("#1A1F26"), paper.shape).copy()
    ink_colour[:int(scan.shape[0] * 0.03)] = _rgb(RED)  # the slogan line in red, as such title pages were printed
    rgb = paper * (1 - ink) + ink_colour * ink
    page = np.dstack([rgb, np.ones(scan.shape)])
    width = 245
    height = width * scan.shape[0] / scan.shape[1]
    return _card(_place(page, (W - width) / 2, (H - height) / 2, width, "shadow", rotate=-4))


IMAGES = {
    "floorplan-cover": (card_floorplan, W, H),
    "proteins-cover": (card_proteins, W, H),
    "cv-tasks-cover": (card_cv, W, H),
    "geology-cover": (card_geology, W, H),
    "pandemic-vision-cover": (card_pandemic, W, H),
    "red-russian-cover": (card_book, W, H),
}
