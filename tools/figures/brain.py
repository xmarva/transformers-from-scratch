"""Brains for the articles: the cortex of fsaverage6, coloured by atlas labels, drawn with matplotlib.

The surface is FreeSurfer's fsaverage6 (41k vertices per hemisphere) as nilearn provides it; the folds are shown by
the sulcal depth map, light on the gyri and darker in the sulci. Regions come from published atlases:

    networks    Schaefer et al. 2018, 400 parcels in Yeo et al. 2011's seven networks, defined on fsaverage6
                (FreeSurfer annotations from the ThomasYeoLab/CBIG repository)
    anatomy     Destrieux et al. 2010, FreeSurfer's gyri and sulci, defined on fsaverage5 (nilearn); fsaverage meshes
                are nested, so each fsaverage6 vertex takes the label of the nearest fsaverage5 vertex on the sphere

`draw(ax, labels, colors, view)` paints one hemisphere seen from one side; every face takes the colour of its
vertices' label, shaded by a light from the viewer's upper left so that the shape reads without a 3D engine.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

SCHAEFER = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage6/label/{hemi}.Schaefer2018_400Parcels_7Networks_order.annot"
)
MESH = "fsaverage6"
CACHE = Path.home() / ".cache" / "tfs-articles"
# the camera sits at -x and looks toward +x, z up, so the page's right is -y (posterior for a left lateral view);
# a view turns the hemisphere about z first: (azimuth, elevation) in degrees
VIEWS = {
    ("left", "lateral"): (0, 0),
    ("left", "medial"): (180, 0),
    ("right", "lateral"): (180, 0),
    ("right", "medial"): (0, 0),
    ("left", "ventral"): (0, -90),
    ("right", "ventral"): (0, -90),
}


@cache
def mesh(hemi: str, kind: str = "inflated"):
    """Vertices (n, 3) and faces (m, 3) of the surface."""
    from nilearn import datasets

    surface = datasets.load_fsaverage(MESH)[kind]
    part = surface.parts[hemi]
    return np.asarray(part.coordinates, dtype=float), np.asarray(part.faces, dtype=int)


@cache
def sulcal_depth(hemi: str):
    from nilearn import datasets

    data = datasets.load_fsaverage_data(mesh=MESH, data_type="sulcal")
    return np.asarray(data.data.parts[hemi], dtype=float).ravel()


@cache
def schaefer_networks(hemi: str):
    """For every vertex, the name of its Yeo network ('Default', 'Cont', 'SalVentAttn', ...) or '' (medial wall)."""
    import urllib.request

    import nibabel as nib

    short = {"left": "lh", "right": "rh"}[hemi]
    path = CACHE / f"{short}.{MESH}.Schaefer2018_400Parcels_7Networks_order.annot"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SCHAEFER.format(hemi=short), path)
    labels, _, names = nib.freesurfer.read_annot(str(path))
    parcel = [n.decode() for n in names]
    network = ["" if p.startswith("Background") else p.split("_")[2] for p in parcel]
    return np.array([network[k] if k >= 0 else "" for k in labels])


@cache
def destrieux(hemi: str):
    """For every vertex, its Destrieux label ('G_rectus', 'G_insular_short', ...); 'Medial_wall' off the cortex."""
    from nilearn import datasets
    from scipy.spatial import cKDTree

    atlas = datasets.fetch_atlas_surf_destrieux()
    names = [n.decode() if isinstance(n, bytes) else n for n in atlas["labels"]]
    coarse = np.asarray(atlas[f"map_{hemi}"], dtype=int)
    sphere5 = np.asarray(datasets.load_fsaverage("fsaverage5")["sphere"].parts[hemi].coordinates)
    sphere = np.asarray(datasets.load_fsaverage(MESH)["sphere"].parts[hemi].coordinates)
    _, nearest = cKDTree(sphere5).query(sphere)
    return np.array([names[k] for k in coarse[nearest]])


def _rotation(azimuth: float, elevation: float) -> np.ndarray:
    a, e = np.radians(azimuth), np.radians(elevation)
    turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tilt = np.array([[np.cos(e), 0, -np.sin(e)], [0, 1, 0], [np.sin(e), 0, np.cos(e)]])
    return tilt @ turn


def draw(ax, hemi: str, view: str, labels, colors: dict[str, str], *, base=("#E9ECF1", "#C9CED7"), kind="inflated"):
    """Paint one hemisphere on `ax`, seen from `view`, faces coloured by `colors[label]` or the folds' greys.

    After the rotation the camera looks along +x from -x: the page shows (-y, z)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    vertices, faces = mesh(hemi, kind)
    vertices = vertices - vertices.mean(axis=0)
    seen = vertices @ _rotation(*VIEWS[(hemi, view)]).T
    sulc = sulcal_depth(hemi)
    gyrus, sulcus = np.array(to_rgb(base[0])), np.array(to_rgb(base[1]))
    depth = np.clip(0.5 + sulc / (2 * np.percentile(np.abs(sulc), 90)), 0, 1)[:, None]  # 0 on the gyri, 1 deep
    vertex_rgb = gyrus + (sulcus - gyrus) * depth
    for name, color in colors.items():
        vertex_rgb[labels == name] = to_rgb(color)
    face_rgb = vertex_rgb[faces].mean(axis=1)

    corners = seen[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    toward_viewer = normal[:, 0] < 0  # the camera looks along -x: faces whose normal points to -x face it
    light = np.array([-0.75, -0.35, 0.55])
    light /= np.linalg.norm(light)
    shade = 0.62 + 0.38 * np.clip(normal @ light, 0, 1)
    order = np.argsort(corners[:, :, 0].mean(axis=1))[::-1]  # far faces first
    order = order[toward_viewer[order]]
    polygons = np.stack([-corners[order][:, :, 1], corners[order][:, :, 2]], axis=-1)
    ax.add_collection(PolyCollection(polygons, facecolors=np.clip(face_rgb[order] * shade[order, None], 0, 1),
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(-seen[:, 1].max() - 2, -seen[:, 1].min() + 2)
    ax.set_ylim(seen[:, 2].min() - 2, seen[:, 2].max() + 2)
    ax.set_aspect("equal")
    ax.axis("off")


@cache
def _mni_and_atlases():
    """The MNI152 template at 1 mm and the Harvard-Oxford cortical and subcortical atlases resampled onto it."""
    from nilearn import datasets, image

    template = datasets.load_mni152_template(resolution=1)
    cortical = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
    subcortical = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    maps = {}
    for atlas in (cortical, subcortical):
        resampled = image.resample_to_img(atlas.maps, template, interpolation="nearest", copy_header=True,
                                          force_resample=True)
        maps[id(atlas)] = (np.asarray(resampled.get_fdata(), dtype=int), list(atlas.labels))
    return template, list(maps.values())


def coronal(ax, y_mm: float, colors: dict[str, str], *, tissue=("#1D2430", "#F3F4F6"), background="#FFFFFF"):
    """A coronal slice of the MNI152 template at y = y_mm, Harvard-Oxford regions in `colors` painted over it.

    Region names are the atlases' labels, e.g. 'Left Amygdala' (subcortical) or 'Insular Cortex' (cortical)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    template, atlases = _mni_and_atlases()
    inverse = np.linalg.inv(template.affine)
    j = int(round((inverse @ np.array([0, y_mm, 0, 1]))[1]))
    anatomy = np.asarray(template.get_fdata())[:, j, :].T  # rows: z from bottom, columns: x
    rgb = np.zeros(anatomy.shape + (3,))
    cmap = LinearSegmentedColormap.from_list("tissue", [tissue[0], tissue[1]])
    level = anatomy / np.percentile(anatomy[anatomy > 0], 99)
    rgb[:] = np.asarray(cmap(np.clip(level, 0, 1)))[..., :3]
    rgb[anatomy <= 0.05 * anatomy.max()] = to_rgb(background)
    for labels, names in atlases:
        plane = labels[:, j, :].T
        for name, color in colors.items():
            if name in names:
                mask = plane == names.index(name)
                rgb[mask] = 0.25 * rgb[mask] + 0.75 * np.array(to_rgb(color))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.axis("off")
    rows, cols = np.nonzero(anatomy > 0.05 * anatomy.max())
    ax.set_xlim(cols.min() - 3, cols.max() + 3)
    ax.set_ylim(rows.min() - 3, rows.max() + 3)
    return j
