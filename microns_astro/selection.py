"""Selection files: the hand-off between the 3D viewer and the downloader.

The viewer exports JSON like
    {"format": "microns_astro.selection", "version": 1,
     "astro_id": "864691...", "datastack_version": 1822, "mip": 0,
     "resolution_nm": [4, 4, 40], "tile_size": [1024, 1024, 128],
     "bbox": {"min": [x, y, z], "max": [x, y, z]},     # voxels at `mip`, max exclusive
     "tiles": [{"start": [...], "end": [...], "status": "clean"}, ...]}
Pure numpy/json: no network or heavy dependencies.
"""

import json

import numpy as np

SELECTION_FORMAT = "microns_astro.selection"


def load_selection(path):
    """Read and validate a selection file exported by the viewer."""
    with open(path) as f:
        sel = json.load(f)
    if sel.get("format") != SELECTION_FORMAT:
        raise ValueError(f"{path} is not an {SELECTION_FORMAT} file.")
    sel["astro_id"] = int(sel["astro_id"])  # stored as a string (JS precision)
    for key in ("min", "max"):
        sel["bbox"][key] = [int(v) for v in sel["bbox"][key]]
    if any(hi <= lo for lo, hi in zip(sel["bbox"]["min"], sel["bbox"]["max"])):
        raise ValueError("Selection has an empty bbox (nothing selected?).")
    return sel


def selection_bbox(sel):
    return tuple(sel["bbox"]["min"]), tuple(sel["bbox"]["max"])


def selection_tile_boxes(sel):
    return [(tuple(t["start"]), tuple(t["end"])) for t in sel["tiles"]]


def pad_bbox(mins, maxs, pad_vox):
    """Grow a bbox by `pad_vox` on every side (clip to the volume afterwards)."""
    return (tuple(int(m) - pad_vox for m in mins),
            tuple(int(m) + pad_vox for m in maxs))


def rescale_bbox(mins, maxs, res_from, res_to):
    """Convert a voxel bbox between mips, rounding outward so it still covers
    the same physical region. res_* are nm per voxel (x, y, z)."""
    scale = np.asarray(res_from, float) / np.asarray(res_to, float)
    lo = np.floor(np.asarray(mins, float) * scale + 1e-9).astype(int)
    hi = np.ceil(np.asarray(maxs, float) * scale - 1e-9).astype(int)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)
