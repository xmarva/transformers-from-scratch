"""16:9 covers for the experiments on the portfolio page of xmarva.github.io.

Each card takes one real picture from the project (a detection result, a structure render, a segmentation, a camera
frame), cuts the object out or recolours it into the site's palette, and sets it on the dark background with the dot
grid used by the site's covers. Sources are in data/source/: the projects' own images, and for the language-model
experiment a public-domain title page from 1902 (Wikimedia Commons, "Cover Lenin Zadachy.jpg").

    uv run --group articles python tools/render.py --article experiments \
        --out ../../websites/xmarva.github.io/assets/img/projects/personal
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from tools.figures import cover as cv

SOURCE = Path(__file__).parent / "data" / "source"
W, H = 800, 450
PAPER, INK, MUTED, RED = "#ECE6D8", "#C9CCD3", "#8E939E", cv.RED


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _load(name):
    return np.asarray(Image.open(SOURCE / name).convert("RGB"), dtype=float) / 255


def _uri(rgba: np.ndarray) -> str:
    image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _card(objects: str) -> str:
    """The site's dark cover background with its dot grid, a faint red glow, and a soft shadow filter for objects
    that are 'things' on the table."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs>'
            f'<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<circle cx="2" cy="2" r="1.2" fill="{cv.GRID}"/></pattern>'
            f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            f'<feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#000" flood-opacity="0.55"/></filter>'
            f'<radialGradient id="glow" cx="50%" cy="50%" r="55%"><stop offset="0" stop-color="{RED}" '
            f'stop-opacity="0.10"/><stop offset="1" stop-color="{RED}" stop-opacity="0"/></radialGradient></defs>'
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


# ---------------------------------------------------------------- floor plans: the model's detections, on dark

def card_floorplan():
    from scipy import ndimage

    img = _load("floorplan-detections.png")
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    grey = img.mean(axis=2)
    green = (g - np.maximum(r, b) > 0.08) & (g > 0.25)  # the detection boxes
    red, bg = _rgb(RED) / 255, _rgb(cv.BG) / 255
    out = np.zeros(img.shape[:2] + (4,))
    out[..., :3] = _rgb(INK) / 255
    out[..., 3] = np.clip((0.92 - grey) / 0.7, 0, 1) * 0.9  # plan lines: the darker, the more opaque
    out[green, :3] = red
    out[green, 3] = 0.75
    # the score labels: grey (51, 51, 50) rectangles with light text become red tags with dark text
    fill = ndimage.binary_opening(np.abs(grey - 51 / 255) < 5 / 255, structure=np.ones((2, 2)))
    labels, _ = ndimage.label(ndimage.binary_closing(fill, structure=np.ones((11, 25))))  # bridge the letters
    for sl in ndimage.find_objects(labels):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 5 <= h <= 40 and w > h:
            text = np.clip((grey[sl] - 0.25) / 0.35, 0, 1)[..., None]
            out[sl[0], sl[1], :3] = red * (1 - text) + bg * text
            out[sl[0], sl[1], 3] = 1.0
    height = 420
    width = height * img.shape[1] / img.shape[0]
    return _card(_place(out, (W - width) / 2, (H - height) / 2, width))


# ---------------------------------------------------------------- proteins: ribbon renders, rainbow to the site's red

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
    hue = hsv[..., 0]
    t = np.clip(1 - ((hue + 0.05) % 1.0) / 0.80, 0, 1)  # violet (N-terminus) ... red (C-terminus)
    light, red = _rgb("#E7EAF0") / 255, _rgb(RED) / 255
    colour = light[None, None] * (1 - t[..., None]) + red[None, None] * t[..., None]
    shade = 0.35 + 0.75 * v[..., None]
    return np.dstack([np.clip(colour * shade, 0, 1), alpha])


def card_proteins():
    img = _load("proteins.png")
    parts = []
    width = 250
    for (x0, y0, x1, y1), x in zip(PANELS.values(), [20, 275, 530], strict=True):
        height = width * (y1 - y0) / (x1 - x0)
        parts.append(_place(_ribbons(img[y0:y1, x0:x1]), x, (H - height) / 2, width))
    return _card("".join(parts))


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
    mask = hsv[..., 1] > 0.3  # the magenta and violet SAM masks
    out = np.dstack([img, np.ones(img.shape[:2])])
    red = _rgb(RED) / 255
    out[mask, :3] = red * (0.35 + 0.8 * hsv[mask, 2][:, None])
    out[~mask, :3] = img[~mask] * 1.05
    out[background, 3] = 0
    return _card(_place(out, 70, 18, 660, "shadow"))


# ---------------------------------------------------------------- geology: the layered section with the well path

def card_geology():
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage

    img = _load("geology-section.png")[262:880, 102:1298]  # the cross-section without axes or the log label
    hsv = rgb_to_hsv(img)
    lum = img @ np.array([0.299, 0.587, 0.114])
    # the layers in dark greys with a warm cast, the deepest the darkest
    deep, pale = _rgb("#1B1719") / 255, _rgb("#4E4246") / 255
    t = np.clip((lum - 0.4) / 0.55, 0, 1)[..., None]
    rgb = deep * (1 - t) + pale * t
    well = lum < 0.18  # the black well path and the section's outlines
    dashed = (hsv[..., 1] < 0.15) & (lum > 0.25) & (lum < 0.55)  # the grey dashed line
    horizons = (hsv[..., 1] > 0.45) & ((hsv[..., 0] < 0.03) | (hsv[..., 0] > 0.3)) & ~well  # red, green, blue lines
    rgb[ndimage.binary_dilation(horizons)] = _rgb(RED) / 255
    rgb[dashed] = _rgb(MUTED) / 255
    rgb[ndimage.binary_dilation(well)] = _rgb("#F2F3F5") / 255
    framed = _framed(rgb, radius=14)
    # the white area below the last surface has no data: cut it out, so the section keeps its shape
    empty = (lum > 0.93) & (hsv[..., 1] < 0.08)
    labels, _ = ndimage.label(empty)
    big = np.bincount(labels.ravel())
    big[0] = 0
    framed[..., 3] *= ~np.isin(labels, np.flatnonzero(big > 2000))
    return _card(_place(framed, 60, (H - 680 * framed.shape[0] / framed.shape[1]) / 2, 680, "shadow"))


# ---------------------------------------------------------------- pandemic vision: a camera frame with the alerts

def card_pandemic():
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage

    img = _load("pandemic-dashboard.png")[318:618, 626:1176]  # the "distance exceeded" camera frame, no overlay text
    hsv = rgb_to_hsv(img)
    hue = hsv[..., 0]
    boxes = (hsv[..., 1] > 0.6) & (hsv[..., 2] > 0.5) & ((hue < 0.03) | (hue > 0.97) | ((hue > 0.25) & (hue < 0.42)))
    # blur the people inside each detection box, so the cover shows the alerts, not who was in the shop
    boxes = ndimage.binary_closing(boxes, structure=np.ones((7, 7)))  # fill the score text in the labels
    labels, _ = ndimage.label(ndimage.binary_dilation(boxes, iterations=2))
    blurred = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)),
                         dtype=float) / 255
    heads = np.zeros(img.shape[:2])
    for sl in ndimage.find_objects(labels):  # the top third of each person's box, feathered
        if (sl[0].stop - sl[0].start) > 60 and (sl[1].stop - sl[1].start) > 30:
            top = sl[0].start + (sl[0].stop - sl[0].start) // 3
            heads[sl[0].start:top, sl[1]] = 1
    heads = np.asarray(Image.fromarray((heads * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                       dtype=float)[..., None] / 255
    base = heads * blurred + (1 - heads) * img
    lum = base @ np.array([0.299, 0.587, 0.114])
    dark, light = _rgb("#121214") / 255, _rgb("#9DA3AE") / 255
    rgb = dark + (light - dark) * np.clip(lum, 0, 1)[..., None]
    rgb[ndimage.binary_dilation(boxes)] = _rgb(RED) / 255
    return _card(_place(_framed(rgb, radius=12), 110, (H - 580 * rgb.shape[0] / rgb.shape[1]) / 2, 580, "shadow"))


# ---------------------------------------------------------------- red russian: a 1902 title page as a paper sheet

def card_book():
    scan = np.asarray(Image.open(SOURCE / "lenin-zadachy-1902.jpg").convert("L"), dtype=float) / 255
    ink = np.clip((0.85 - scan) / 0.6, 0, 1)
    paper, black, red = _rgb(PAPER) / 255, _rgb("#1D1A18") / 255, _rgb("#B5251E") / 255
    rgb = paper * (1 - ink[..., None]) + black * ink[..., None]
    top = slice(0, int(scan.shape[0] * 0.05))  # the slogan line in red, as such title pages were often printed
    rgb[top] = paper * (1 - ink[top][..., None]) + red * ink[top][..., None]
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
