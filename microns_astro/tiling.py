"""Tile grid generation (pure geometry; no network access except compute_tiles)."""

import numpy as np

from .meshes import fetch_mesh_bbox_fast


def tile_index(start, origin, tile_size):
    """Grid index of the tile starting at `start`."""
    return tuple((np.array(start) - np.array(origin)) // np.array(tile_size))


def occupied_tile_indices(verts_vox, origin, tile_size):
    """Tile grid indices containing at least one mesh vertex."""
    idx = (verts_vox - np.array(origin)) // np.array(tile_size)
    idx = idx[(idx >= 0).all(axis=1)]
    return set(map(tuple, np.unique(idx, axis=0)))


def iterate_tiles(mins, maxs, tile_size):
    """Yield (start, end) bounding boxes covering [mins, maxs)."""
    ranges = [range(lo, hi, s) for lo, hi, s in zip(mins, maxs, tile_size)]
    for x0 in ranges[0]:
        for y0 in ranges[1]:
            for z0 in ranges[2]:
                start = (x0, y0, z0)
                yield start, tuple(min(a + s, hi)
                                   for a, s, hi in zip(start, tile_size, maxs))


def align_origin(mins, bounds_min, chunk):
    """Snap minimum coordinates down to the storage chunk grid."""
    return tuple(int(b + ((m - b) // c) * c)
                 for m, b, c in zip(mins, bounds_min, chunk))


def compute_tiles(seg_vol, img_vol, cfg, astro_id):
    """Generate full-size candidate tiles from the target astrocyte mesh."""
    tile_size = cfg.subvolume_size
    mins_raw, maxs, verts = fetch_mesh_bbox_fast(seg_vol, cfg, astro_id)
    chunk = np.lcm(np.array(seg_vol.meta.chunk_size(cfg.mip)),
                   np.array(img_vol.meta.chunk_size(cfg.mip)))

    mins = mins_raw
    if cfg.align_to_chunks:
        bmin = np.array(seg_vol.meta.bounds(cfg.mip).minpt, dtype=int)
        mins = align_origin(mins_raw, bmin, chunk)

    all_tiles = list(iterate_tiles(mins, maxs, tile_size))
    full = [(s, e) for s, e in all_tiles
            if tuple(np.subtract(e, s)) == tuple(tile_size)]
    kept = full
    if cfg.use_mesh_prefilter:
        occupied = occupied_tile_indices(verts, mins, tile_size)
        kept = [(s, e) for s, e in full if tile_index(s, mins, tile_size) in occupied]

    return dict(mins_raw=mins_raw, mins=mins, maxs=maxs, verts_vox=verts,
                chunk=chunk, all_tiles=all_tiles, full_tiles=full, kept_tiles=kept)


def drop_outermost_tiles(tiles, mins, tile_size, distance=1):
    """Remove tiles along the outer edges of the candidate grid."""
    idxs = [np.array(tile_index(s, mins, tile_size)) for s, _ in tiles]
    lo, hi = np.min(idxs, axis=0), np.max(idxs, axis=0)
    inner = [t for t, i in zip(tiles, idxs)
             if (i >= lo + distance).all() and (i <= hi - distance).all()]
    print(f"Outermost filter: dropped {len(tiles) - len(inner)} edge tiles, "
          f"kept {len(inner)}")
    return inner
