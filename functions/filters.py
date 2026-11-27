"""Tile prefilters: neighbor meshes, coarse-mip check, reject cache."""

import os

import fastremap
import numpy as np
from cloudvolume import Bbox
from scipy.spatial import cKDTree
from tqdm import tqdm

from .meshes import get_neighbor_verts
from .tiling import occupied_tile_indices, tile_index


def drop_tiles_near_unproofread(tiles, info, cfg, unp_ids, unp_pos_nm, seg_vol):
    """Remove tiles touched by nearby unproofread meshes; order the rest."""
    tile_size = cfg.subvolume_size
    res = np.array(seg_vol.meta.resolution(cfg.mip), dtype=float)
    mins, maxs = np.array(info["mins"]), np.array(info["maxs"])
    pos_vox = unp_pos_nm / res
    bounds = seg_vol.meta.bounds(cfg.mip)

    inside = ((pos_vox >= np.array(bounds.minpt))
              & (pos_vox <= np.array(bounds.maxpt))).all(axis=1).mean()
    if inside < 0.9:
        print(f"WARNING: only {inside:.0%} of unproofread soma positions fall inside "
              "the volume bounds. Neighbor prefilter may be unreliable.")

    margin = cfg.neighbor_margin_nm / res
    near = ((pos_vox >= mins - margin) & (pos_vox <= maxs + margin)).all(axis=1)
    neighbors = unp_ids[near]
    if len(neighbors) == 0:
        print("Neighbor prefilter: no unproofread astrocytes near this bbox.")
        return tiles

    results = [get_neighbor_verts(seg_vol, cfg, n)
               for n in tqdm(neighbors, desc="Neighbor meshes")]
    verts_list = [v for v in results if v is not None]

    bad = set()
    for verts in verts_list:
        bad |= occupied_tile_indices(verts, info["mins"], tile_size)
    kept = [(s, e) for s, e in tiles if tile_index(s, mins, tile_size) not in bad]

    print(f"Neighbor prefilter: {len(neighbors)} nearby unproofread astrocytes "
          f"({len(results) - len(verts_list)} meshes failed); "
          f"kept {len(kept)} of {len(tiles)} tiles")

    # Order remaining tiles from farthest to nearest neighbor mesh.
    if kept and verts_list:
        all_verts = np.concatenate(verts_list)[::5] * res
        centers = np.array([(np.array(s) + np.array(e)) / 2 for s, e in kept]) * res
        dist, _ = cKDTree(all_verts).query(centers)
        kept = [kept[i] for i in np.argsort(-dist)]
    return kept


def coarse_has_unproofread(start, end, coarse_vol, cfg, check_mip, keep_ids,
                           all_astro_ids, timestamp):
    """True if an unproofread astrocyte is visible at a coarse mip."""
    r0 = np.array(coarse_vol.meta.resolution(cfg.mip), dtype=float)
    r1 = np.array(coarse_vol.meta.resolution(check_mip), dtype=float)
    lo = np.floor(np.array(start) * r0 / r1).astype(int)
    hi = np.ceil(np.array(end) * r0 / r1).astype(int)

    bounds = coarse_vol.meta.bounds(check_mip)
    lo, hi = np.maximum(lo, bounds.minpt), np.minimum(hi, bounds.maxpt)
    if (hi <= lo).any():
        return False

    seg = coarse_vol.download(Bbox(lo.tolist(), hi.tolist()), mip=check_mip,
                              agglomerate=True, timestamp=timestamp)
    ids = fastremap.unique(seg)
    return bool((np.isin(ids, all_astro_ids) & ~np.isin(ids, keep_ids)).any())


def load_rejected(path, signature):
    """Load rejected tiles if the grid/version signature matches."""
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
        if lines and lines[0] == signature:
            return {tuple(map(int, l.split())) for l in lines[1:] if l.strip()}
    with open(path, "w") as f:
        f.write(signature + "\n")
    return set()
