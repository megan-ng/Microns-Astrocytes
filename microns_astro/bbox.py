"""Download an arbitrary box (or an astrocyte's entire bounding box), block by
block, into uncompressed (Z, Y, X) TIFFs written through memory maps.

Unlike pipeline.py, nothing is rejected here: the whole box is kept, and any
unproofread astrocytes found inside it are reported instead.
"""

import json
import os
from collections import Counter

import fastremap
import numpy as np
import tifffile
from cloudvolume import Bbox
from tqdm import tqdm

from .meshes import fetch_mesh_bbox_fast
from .tile_io import tile_pair_paths
from .tiling import iterate_tiles


def clip_bbox(seg_vol, cfg, mins, maxs):
    """Clip a voxel bbox (at cfg.mip) to the volume bounds."""
    bounds = seg_vol.meta.bounds(cfg.mip)
    mins = tuple(max(int(m), int(b)) for m, b in zip(mins, bounds.minpt))
    maxs = tuple(min(int(m), int(b)) for m, b in zip(maxs, bounds.maxpt))
    if any(hi <= lo for lo, hi in zip(mins, maxs)):
        raise ValueError(f"Bbox {mins}->{maxs} is empty after clipping to the volume.")
    return mins, maxs


def astro_bbox(seg_vol, cfg, astro_id):
    """Padded mesh bbox of the astrocyte, clipped to the volume, at cfg.mip."""
    mins, maxs, _ = fetch_mesh_bbox_fast(seg_vol, cfg, astro_id)
    return clip_bbox(seg_vol, cfg, mins, maxs)


def estimate_gb(mins, maxs, bytes_per_voxel=2):
    """Uncompressed size of image (1 byte) + mask (1 byte) in GB."""
    n_vox = int(np.prod(np.subtract(maxs, mins).astype(np.int64)))
    return n_vox * bytes_per_voxel / 1e9


def _zyx(arr):
    """(X, Y, Z[, 1]) CloudVolume array -> (Z, Y, X)."""
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.transpose(2, 1, 0)


def download_bbox(seg_vol, img_vol, cfg, astro_id, keep_ids, all_astro_ids,
                  timestamp, out_dir, mask_mode="target",
                  block_size=(1024, 1024, 128), max_gb=50.0, bbox=None):
    """Download EM + mask for a box.

    bbox: (mins, maxs) in voxels at cfg.mip (max exclusive). If None, the
          astrocyte's whole mesh bounding box is used.
    mask_mode: "target"    -> only `astro_id` is labeled
               "proofread" -> every proofread astrocyte in the box is labeled
    Writes <base>_image.tif, <base>_mask.tif and <base>_meta.json.
    Returns a dict: base path, bbox, and voxel counts of unproofread
    astrocytes seen (those are NOT in the mask).
    """
    if bbox is None:
        mins, maxs = astro_bbox(seg_vol, cfg, astro_id)
    else:
        mins, maxs = clip_bbox(seg_vol, cfg, *bbox)
    gb = estimate_gb(mins, maxs)
    shape_xyz = tuple(int(s) for s in np.subtract(maxs, mins))
    print(f"Bbox {mins} -> {maxs}  shape(xyz)={shape_xyz}  ~{gb:.1f} GB "
          f"uncompressed at mip {cfg.mip}")
    if gb > max_gb:
        raise ValueError(
            f"Bbox is ~{gb:.1f} GB (> max_gb={max_gb}). Use a coarser mip "
            "or raise max_gb.")

    label_ids = [int(astro_id)] if mask_mode == "target" else keep_ids
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(
        out_dir, f"{astro_id}_bbox_mip{cfg.mip}_{mins[0]}_{mins[1]}_{mins[2]}_"
                 f"{shape_xyz[0]}x{shape_xyz[1]}x{shape_xyz[2]}")
    img_path, mask_path = tile_pair_paths(base)
    shape_zyx = shape_xyz[::-1]

    img_mm = tifffile.memmap(img_path + ".tmp", shape=shape_zyx,
                             dtype=img_vol.dtype, bigtiff=True)
    mask_mm = tifffile.memmap(mask_path + ".tmp", shape=shape_zyx,
                              dtype=np.uint8, bigtiff=True)

    unproofread_vox, labeled_ids = Counter(), set()
    blocks = list(iterate_tiles(mins, maxs, block_size))  # includes edge blocks
    for (x0, y0, z0), (x1, y1, z1) in tqdm(blocks, desc="Blocks"):
        seg = seg_vol.download(Bbox((x0, y0, z0), (x1, y1, z1)), mip=cfg.mip,
                               agglomerate=True, timestamp=timestamp)
        ids, counts = fastremap.unique(seg, return_counts=True)

        unp = np.isin(ids, all_astro_ids) & ~np.isin(ids, keep_ids)
        for i, c in zip(ids[unp], counts[unp]):
            unproofread_vox[int(i)] += int(c)

        present = ids[np.isin(ids, label_ids)].tolist()
        labeled_ids.update(int(i) for i in present)
        mask = (fastremap.mask_except(seg, present, in_place=True)
                if present else np.zeros_like(seg))

        sl = (slice(z0 - mins[2], z1 - mins[2]),
              slice(y0 - mins[1], y1 - mins[1]),
              slice(x0 - mins[0], x1 - mins[0]))
        img_mm[sl] = _zyx(img_vol[x0:x1, y0:y1, z0:z1])
        mask_mm[sl] = (_zyx(mask) != 0)

    img_mm.flush()
    mask_mm.flush()
    del img_mm, mask_mm  # release the maps before renaming (needed on Windows)
    os.replace(img_path + ".tmp", img_path)
    os.replace(mask_path + ".tmp", mask_path)

    meta = dict(
        astro_id=str(astro_id), datastack_version=getattr(cfg, "datastack_version", None),
        mip=cfg.mip, bbox_min=list(mins), bbox_max=list(maxs),
        mask_mode=mask_mode, labeled_ids=[str(i) for i in sorted(labeled_ids)],
        unproofread_voxels={str(k): v for k, v in unproofread_vox.items()})
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if unproofread_vox:
        print(f"WARNING: {len(unproofread_vox)} unproofread astrocyte(s) inside "
              "the box; they are unlabeled in the mask:")
        for i, c in unproofread_vox.most_common(10):
            print(f"  {i}: {c} voxels")
    return dict(base=base, mins=mins, maxs=maxs,
                unproofread_voxels=dict(unproofread_vox))
