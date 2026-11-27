"""Mesh fetching and caching (target astrocyte and neighbor meshes)."""

import os

import numpy as np
from tqdm import tqdm

from .cache import cached_save, remove_quietly


def mesh_vertices_vox(seg_vol, segid, lod, mip):
    """Fetch a mesh (optionally at a given LOD) and return voxel vertices."""
    mesh = seg_vol.mesh.get(segid)[segid] if lod is None \
        else seg_vol.mesh.get(segid, lod=lod)[segid]
    res = np.array(seg_vol.meta.resolution(mip), dtype=float)
    return np.floor(mesh.vertices / res).astype(np.int64)


def fetch_mesh_bbox_fast(seg_vol, cfg, segid):
    """Target mesh vertices + padded bbox, using a versioned local cache.

    Returns (mins, maxs, verts_vox) in voxels at cfg.mip.
    """
    segid = int(segid)
    lod_tag = "full" if cfg.mesh_lod is None else f"lod{cfg.mesh_lod}"
    path = os.path.join(
        cfg.target_mesh_dir,
        f"{segid}_{lod_tag}_mip{cfg.mip}_v{cfg.datastack_version}.npz")

    if cfg.cache_target_mesh and os.path.exists(path):
        try:
            data = np.load(path)
            print(f"Target mesh cache hit: {path}")
            return (tuple(data["mins_vox"].tolist()),
                    tuple(data["maxs_vox"].tolist()), data["verts_vox"])
        except Exception as e:
            print(f"Warning: could not read target mesh cache: {e}")
            remove_quietly(path)

    print(f"Fetching target mesh: {segid}")
    verts = mesh_vertices_vox(seg_vol, segid, cfg.mesh_lod, cfg.mip)
    mins = verts.min(axis=0) - cfg.pad_vox
    maxs = verts.max(axis=0) + 1 + cfg.pad_vox

    if cfg.cache_target_mesh:
        cached_save(path, lambda f: np.savez(
            f, verts_vox=verts, mins_vox=mins.astype(np.int64),
            maxs_vox=maxs.astype(np.int64)))
    return tuple(int(x) for x in mins), tuple(int(x) for x in maxs), verts


def get_neighbor_verts(seg_vol, cfg, segid):
    """Coarse neighbor mesh vertices, using a versioned local cache."""
    segid = int(segid)
    path = os.path.join(
        cfg.neighbor_mesh_dir,
        f"{segid}_lod{cfg.neighbor_mesh_lod}_mip{cfg.mip}"
        f"_v{cfg.datastack_version}.npy")

    if cfg.cache_neighbor_meshes and os.path.exists(path):
        try:
            return np.load(path)
        except Exception as e:
            tqdm.write(f"Corrupt mesh cache for {segid}; deleting: {e}")
            remove_quietly(path)

    try:
        try:
            verts = mesh_vertices_vox(seg_vol, segid, cfg.neighbor_mesh_lod, cfg.mip)
        except Exception:  # fall back to the full-resolution mesh
            verts = mesh_vertices_vox(seg_vol, segid, None, cfg.mip)
        if cfg.cache_neighbor_meshes:
            cached_save(path, lambda f: np.save(f, verts))
        return verts
    except Exception as e:
        tqdm.write(f"Neighbor mesh failed for {segid}: {type(e).__name__}: {e}")
        return None
