"""Course palette. A color always means the same concept, in every lesson and figure:

    data  (amber)   tokens, datasets, model inputs and outputs
    enc   (teal)    encoder blocks; training curves
    dec   (rose)    decoder blocks; validation curves
    att   (violet)  every kind of attention

Each concept has a stroke color, a light tint for fills and an ink color for text on that tint.
"""

LIGHT = {
    "page": "#F3F4F6",
    "card": "#FFFFFF",
    "card_alt": "#F6F7F9",
    "ink": "#1D2430",
    "muted": "#5B6575",
    "grid": "#E6E9EE",
    "line": "#B9C0CB",
    "data": "#C27A0E",
    "data_tint": "#FCF0DA",
    "data_ink": "#8A5508",
    "enc": "#0F8C7E",
    "enc_tint": "#E1F3F0",
    "enc_ink": "#0B6B60",
    "dec": "#D23F57",
    "dec_tint": "#FBE6EA",
    "dec_ink": "#A82D42",
    "att": "#6A55D8",
    "att_tint": "#ECE8FC",
    "att_ink": "#4B39B5",
}

DARK = {
    "page": "#111419",
    "card": "#1A1F26",
    "card_alt": "#20262E",
    "ink": "#E7EAF0",
    "muted": "#9AA3B2",
    "grid": "#2C333D",
    "line": "#4A5361",
    "data": "#F0B04A",
    "data_tint": "#362A14",
    "data_ink": "#F6C878",
    "enc": "#3CC2B0",
    "enc_tint": "#12302C",
    "enc_ink": "#7FDCCF",
    "dec": "#F2667C",
    "dec_tint": "#391D24",
    "dec_ink": "#F7A1AE",
    "att": "#9C8CF5",
    "att_tint": "#272144",
    "att_ink": "#C4B9FF",
}

THEMES = {"light": LIGHT, "dark": DARK}

# Covers use the site's brand red on a near-black background.
BRAND = {"bg": "#161618", "red": "#FF3636", "red_tint": "#3A1A1C", "ink": "#F2F3F5", "muted": "#9DA3AE"}


def palette(theme: str = "light") -> dict[str, str]:
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {sorted(THEMES)}, got {theme!r}")
    return THEMES[theme]
