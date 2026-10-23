"""What a text-only language model knows about colour, for the post "Minds as computers".

Qwen2.5-1.5B-Instruct has read colour words but never seen a colour. For 50 colour names we take its hidden state
at the name (layer 12 of 28, in the sentence "The color of the paint was <name>") and compare the distances between
those states with the distances between the colours themselves, in CIELAB, a space built so that equal distances look
equally different to people. The colours of the names come from the XKCD colour survey (Munroe 2010, CC0), where
each name maps to the average RGB that survey participants gave it.

Writes data/color.json: per word its hex, CIELAB and 2D coordinates of both geometries for the figure; Spearman's
correlation between the two sets of pairwise distances, at every layer; and a permutation test for layer 12.

    uv run --group articles python articles/minds_as_computers/color_geometry.py
"""

import json
import urllib.request
from pathlib import Path

import colour
import numpy as np
import torch
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SURVEY = "https://xkcd.com/color/rgb.txt"
LAYER = 12
WORDS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey", "teal",
    "maroon", "navy", "olive", "lavender", "beige", "turquoise", "magenta", "cyan", "gold", "tan", "salmon",
    "violet", "lime", "indigo", "crimson", "peach", "mint", "mauve", "khaki", "coral", "plum", "ochre", "rust",
    "lilac", "aqua", "burgundy", "mustard", "cream", "amber", "emerald", "scarlet", "azure", "charcoal", "sand",
    "chocolate", "rose", "sky blue", "forest green",
]
OUT = Path(__file__).parent / "data" / "color.json"

survey = {}
for line in urllib.request.urlopen(SURVEY).read().decode().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    name, hex_code = line.split("\t")[:2]
    survey[name] = hex_code


def cielab(hex_code):
    rgb = np.array([int(hex_code[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))


lab = np.array([cielab(survey[w]) for w in WORDS])
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

states = []
for word in WORDS:
    ids = tokenizer(f"The color of the paint was {word}", return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states
    states.append(torch.stack([h[0, -1] for h in hidden]).numpy())  # (layers + 1, d) at the last token
states = np.array(states).transpose(1, 0, 2)  # (layers + 1, words, d)

pairs = np.triu_indices(len(WORDS), 1)
seen = np.linalg.norm(lab[:, None] - lab[None], axis=-1)


def model_distances(layer):
    x = states[layer] - states[layer].mean(axis=0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return 1 - x @ x.T  # cosine distance


by_layer = [spearmanr(seen[pairs], model_distances(k)[pairs]).correlation for k in range(len(states))]

# the permutation test: shuffle which word is which in the model's distances
read = model_distances(LAYER)
rho = spearmanr(seen[pairs], read[pairs]).correlation
rng = np.random.default_rng(0)
null = []
for _ in range(2000):
    p = rng.permutation(len(WORDS))
    null.append(spearmanr(seen[pairs], read[p][:, p][pairs]).correlation)
p_value = (np.sum(np.array(null) >= rho) + 1) / (len(null) + 1)

# for the picture: the model's distances laid out in 3D by MDS, then rotated and scaled (nothing else) onto CIELAB
layout = MDS(n_components=3, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(read)
lab_std, fitted, disparity = procrustes(lab, layout)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({
    "model": MODEL,
    "layer": LAYER,
    "words": WORDS,
    "hex": [survey[w] for w in WORDS],
    "lab": lab.round(3).tolist(),
    "lab_ab": lab_std[:, 1:].round(5).tolist(),
    "model_ab": fitted[:, 1:].round(5).tolist(),
    "rho": rho,
    "p_value": p_value,
    "null_99": float(np.percentile(null, 99)),
    "rho_by_layer": by_layer,
    "procrustes_disparity": disparity,
}, indent=1), encoding="utf-8")
print(f"layer {LAYER}: rho = {rho:.3f}, p = {p_value:.4f}; by layer:", [round(r, 2) for r in by_layer])
