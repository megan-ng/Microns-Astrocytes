"""Gather what the 3D screening viewer needs for one astrocyte.

  1. candidate tiles from the target mesh (tiling.compute_tiles)
  2. which astrocytes each tile contains, from a cheap coarse-mip scan
  3. a status per tile (clean / unproofread / empty)
  4. display meshes: the target plus every other astrocyte seen in a tile

The result is a plain dict consumed by viewer3d.write_html. Heavy imports
(caveclient, cloudvolume) are deferred into build_screening_data so the pure
helpers here can be imported and tested without them.
"""

import json
import os

import numpy as np
from tqdm import tqdm


def scan_tile_contents(tiles, coarse_vol, cfg, scan_mip, watch_ids, timestamp,
                       cache_path=None, signature=""):
    """Astrocyte voxel counts per tile, read at a coarse mip.

    watch_ids: IDs worth recording (all astrocytes + the target).
    Returns a list parallel to `tiles` of
        {"total_vox": int, "astros": {"<root id>": voxel_count}}.
    Results are cached per tile in a JSON file keyed by `signature`, so
    interrupted scans resume and reruns are instant.
    """
    from .filters import coarse_tile_ids

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                blob = json.load(f)
            if blob.get("signature") == signature:
                cache = blob["tiles"]
                print(f"Scan cache hit: {len(cache)} tiles in {cache_path}")
        except Exception as e:
            print(f"Warning: could not read scan cache: {e}")

    out = []
    try:
        for start, end in tqdm(tiles, desc=f"Scanning tiles (mip {scan_mip})"):
            key = "%d_%d_%d" % tuple(start)
            if key not in cache:
                ids, counts = coarse_tile_ids(start, end, coarse_vol, cfg,
                                              scan_mip, timestamp)
                hit = np.isin(ids, watch_ids)
                cache[key] = {
                    "total_vox": int(counts.sum()),
                    "astros": {str(int(i)): int(c)
                               for i, c in zip(ids[hit], counts[hit])}}
            out.append(cache[key])
    finally:  # keep partial progress if interrupted
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"signature": signature, "tiles": cache}, f)
            os.replace(tmp, cache_path)
    return out


def classify_tiles(tiles, contents, astro_id, keep_ids):
    """Turn raw scan results into one row per tile with a status.

    status: "unproofread" - an unproofread astrocyte (other than the target)
                            is present at the scanned mip
            "empty"       - no target and no other proofread astrocyte
            "clean"       - proofread astrocyte(s) present, no unproofread ones
    """
    astro_id = int(astro_id)
    keep = {int(k) for k in keep_ids}
    rows = []
    for i, ((start, end), c) in enumerate(zip(tiles, contents)):
        total = max(c["total_vox"], 1)
        target_vox = c["astros"].get(str(astro_id), 0)
        proofread, unproofread = [], []
        for k in c["astros"]:
            k = int(k)
            if k == astro_id:
                continue
            (proofread if k in keep else unproofread).append(k)
        if unproofread:
            status = "unproofread"
        elif target_vox == 0 and not proofread:
            status = "empty"
        else:
            status = "clean"
        rows.append(dict(
            id=i, start=[int(v) for v in start], end=[int(v) for v in end],
            status=status, target_frac=target_vox / total,
            other_proofread=sorted(proofread), unproofread=sorted(unproofread)))
    return rows


def build_screening_data(cfg, astro_id, max_tiles=1500, include_neighbors=True):
    """Fetch everything for the viewer. Network + CPU heavy on the first run;
    tile scans and meshes are cached so reruns are fast."""
    from collections import Counter

    from .cave import load_vol, setup_caveclient
    from .ids import get_cell_type_ids, get_proofread_astrocyte_root_ids
    from .meshes import get_view_mesh
    from .tiling import compute_tiles

    astro_id = int(astro_id)
    client = setup_caveclient(cfg.datastack_name, cfg.datastack_version)
    timestamp = client.materialize.get_timestamp(cfg.datastack_version)
    seg_path = client.info.segmentation_source()

    keep_ids = get_proofread_astrocyte_root_ids(client, cfg)
    _, all_astro_ids = get_cell_type_ids(client, cfg)
    if astro_id not in set(keep_ids.tolist()):
        print(f"WARNING: {astro_id} is not in the proofread list. Tiles are still "
              "built around it, but it counts as the target, not as proofread.")

    seg_vol = load_vol(seg_path, cfg.mip)
    img_vol = load_vol(client.info.image_source(), cfg.mip)

    # Candidate tiles (mesh-touched, full-size).
    info = compute_tiles(seg_vol, img_vol, cfg, astro_id)
    tiles = info["kept_tiles"]
    print(f"{len(tiles)} candidate tiles of size {cfg.subvolume_size}")
    if not tiles:
        raise ValueError("No candidate tiles for this astrocyte.")
    if len(tiles) > max_tiles:
        raise ValueError(
            f"{len(tiles)} tiles exceeds max_tiles={max_tiles}. Use a larger "
            "--tile-size (fewer, bigger tiles) or raise --max-tiles.")

    # Coarse scan of what each tile contains.
    scan_mip = cfg.check_mip if cfg.check_mip is not None else 3
    available = list(seg_vol.available_mips)
    if scan_mip not in available:
        scan_mip = max(available)
        print(f"WARNING: requested scan mip unavailable; using mip {scan_mip}.")
    coarse_vol = load_vol(seg_path, scan_mip)
    watch = np.union1d(all_astro_ids, np.array([astro_id], dtype=np.uint64))
    signature = (f"v{cfg.datastack_version}|{tuple(cfg.subvolume_size)}|"
                 f"{tuple(info['mins'])}|mip{cfg.mip}|scan{scan_mip}")
    cache_path = os.path.join(cfg.out_dir(astro_id),
                              f"screen_scan_mip{cfg.mip}_scan{scan_mip}.json")
    contents = scan_tile_contents(tiles, coarse_vol, cfg, scan_mip, watch,
                                  timestamp, cache_path, signature)
    rows = classify_tiles(tiles, contents, astro_id, keep_ids)
    print("Tile status:", dict(Counter(r["status"] for r in rows)))

    # Display meshes.
    target = get_view_mesh(seg_vol, cfg, astro_id)
    if target is None:
        raise RuntimeError(f"Could not fetch a mesh for {astro_id}.")
    others = {}
    if include_neighbors:
        seen = {}
        for r in rows:
            for k in r["other_proofread"]:
                seen[k] = "proofread"
            for k in r["unproofread"]:
                seen[k] = "unproofread"
        for k, kind in tqdm(seen.items(), desc="Neighbor display meshes"):
            m = get_view_mesh(seg_vol, cfg, k)
            if m is not None:
                others[k] = dict(verts_nm=m[0], faces=m[1], kind=kind)

    return dict(
        astro_id=astro_id, datastack_version=cfg.datastack_version,
        mip=cfg.mip, scan_mip=scan_mip,
        resolution_nm=[float(r) for r in seg_vol.meta.resolution(cfg.mip)],
        tile_size=list(cfg.subvolume_size),
        mins=[int(v) for v in info["mins"]], maxs=[int(v) for v in info["maxs"]],
        tiles=rows,
        target_mesh=dict(verts_nm=target[0], faces=target[1]),
        other_meshes=others)
