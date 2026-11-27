"""Per-astrocyte driver and top-level run()."""

import os
import random
import time
from collections import Counter

from tqdm import tqdm

from .cave import load_vol, setup_caveclient
from .config import Config
from .filters import drop_tiles_near_unproofread, load_rejected
from .ids import (get_cell_type_ids, get_proofread_astrocyte_root_ids,
                  get_unproofread_astro_table)
from .process import phase, process_tile
from .report import print_report, summarize
from .tiling import compute_tiles, drop_outermost_tiles

REJECT_CACHE_STATUSES = {"empty", "unproofread", "unproofread_coarse"}


def process_single_astro(astro_id, cfg, keep_ids, all_astro_ids, known_ids,
                         unp_ids, unp_pos_nm, img_path, seg_path, out_dir,
                         timestamp):
    """Generate and process candidate tiles for one astrocyte."""
    print(f"\nProcessing astro_id={astro_id}, out_dir={out_dir}")
    phases = {}

    with phase(phases, "volume setup"):  # load volumes once and reuse them
        img_vol = load_vol(img_path, cfg.mip)
        seg_vol = load_vol(seg_path, cfg.mip)
        check_mip = cfg.check_mip
        if check_mip is not None and check_mip not in list(seg_vol.available_mips):
            print(f"WARNING: check_mip={check_mip} is not available. "
                  "Disabling coarse check.")
            check_mip = None
        coarse_vol = load_vol(seg_path, check_mip) if check_mip is not None else None

    # A. Candidate tiles from the target astrocyte mesh.
    try:
        with phase(phases, "target mesh + tiling"):
            info = compute_tiles(seg_vol, img_vol, cfg, astro_id)
    except Exception as e:
        print(f"Could not compute tiles for {astro_id}: {e}\n(If the mesh is "
              f"missing, this root may have been edited after "
              f"v{cfg.datastack_version}.)")
        return

    tiles = info["kept_tiles"]
    print(f"Voxel bbox: {info['mins']} -> {info['maxs']}")
    print(f"Funnel: {len(tiles)} candidate tiles (full-size, touched by the mesh)")
    if not tiles:
        print("No tiles to process for this astro.")
        return

    # B. Remove tiles already known to be rejected.
    rej_file = None
    if cfg.use_reject_cache:
        signature = (f"v{cfg.datastack_version}|{tuple(cfg.subvolume_size)}|"
                     f"{tuple(info['mins'])}|mip{cfg.mip}")
        rej_path = os.path.join(out_dir, "rejected_tiles.txt")
        rejected = load_rejected(rej_path, signature)
        n_before = len(tiles)
        tiles = [t for t in tiles if t[0] not in rejected]
        print(f"Funnel: {len(tiles)} after reject cache (-{n_before - len(tiles)})")
        rej_file = open(rej_path, "a")

    # C. Remove tiles touched by nearby unproofread astrocyte meshes.
    if cfg.use_neighbor_prefilter and tiles:
        with phase(phases, "neighbor prefilter"):
            tiles = drop_tiles_near_unproofread(tiles, info, cfg, unp_ids,
                                                unp_pos_nm, seg_vol)
        print(f"Funnel: {len(tiles)} after neighbor prefilter")

    # D. Optionally remove outermost tiles.
    if cfg.drop_outermost_tiles and tiles:
        tiles = drop_outermost_tiles(tiles, info["mins"], cfg.subvolume_size,
                                     cfg.outermost_tile_distance)
        print(f"Funnel: {len(tiles)} after outermost filter")

    if cfg.sample_n and cfg.sample_n < len(tiles):
        tiles = random.Random(cfg.seed).sample(tiles, cfg.sample_n)
        print(f"Sampling {len(tiles)} tiles")

    if not tiles:
        print("No tiles left to process.")
        if rej_file:
            rej_file.close()
        return

    print(f"Processing {len(tiles)} tiles sequentially (coarse check mip={check_mip})"
          + (f", stopping after {cfg.target_written} written"
             if cfg.target_written else ""))

    status_counts, unproofread_counts = Counter(), Counter()
    results, mask_fracs, unknown_fracs = [], [], []
    t_wall = time.perf_counter()

    # E. Process tiles one at a time.
    for start, end in tqdm(tiles, desc="Tiles"):
        try:
            result = process_tile(start, end, astro_id, cfg, keep_ids,
                                  all_astro_ids, known_ids, img_vol, seg_vol,
                                  coarse_vol, check_mip, out_dir, timestamp)
        except Exception as e:
            status_counts["failed"] += 1
            tqdm.write(f"Tile {start} FAILED: {type(e).__name__}: {e}")
            continue

        status = result["status"]
        status_counts[status] += 1
        results.append(result)
        if "unknown_frac" in result:
            unknown_fracs.append(result["unknown_frac"])
        if status == "written":
            mask_fracs.append(result["mask_frac"])
        if status == "unproofread":
            unproofread_counts[result["n_unproofread"]] += 1

        if rej_file and status in REJECT_CACHE_STATUSES:  # remember for future runs
            rej_file.write("%d %d %d\n" % tuple(start))
            rej_file.flush()

        if cfg.log_every_tile:
            tqdm.write(f"{start} {status:<18s} {result['total']:6.1f}s  "
                       + " ".join(f"{k}={v:.1f}" for k, v in result["timings"].items()))

        # F. Stop once enough tiles have been written (or already existed).
        if cfg.target_written and (
                status_counts["written"] + status_counts["existing"]
                >= cfg.target_written):
            break

    wall = time.perf_counter() - t_wall
    if rej_file:
        rej_file.close()

    print(f"\nResults for astro {astro_id} ({len(tiles)} candidate tiles): "
          + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    if unproofread_counts:
        print("  unproofread astrocytes per skipped tile: "
              + ", ".join(f"{n}: {c} tiles"
                          for n, c in sorted(unproofread_counts.items())))
    summarize("mask fraction (written tiles)", mask_fracs)
    summarize("unknown-segment fraction (tiles passing the astrocyte checks)",
              unknown_fracs)
    print(f"Output -> {out_dir}")
    print_report(astro_id, results, wall, phases)


def run(cfg: Config = Config()):
    """Process the astrocyte(s) selected by `cfg`."""
    setup = {}
    client = setup_caveclient(cfg.datastack_name, cfg.datastack_version)
    img_path = client.info.image_source()
    seg_path = client.info.segmentation_source()
    print(img_path)
    print(seg_path)

    # Pin voxel IDs to the same snapshot as the ID tables.
    timestamp = client.materialize.get_timestamp(cfg.datastack_version)
    print(f"Segmentation timestamp (v{cfg.datastack_version}): {timestamp}")

    with phase(setup, "ID tables"):
        keep_ids = get_proofread_astrocyte_root_ids(client, cfg)
        print("Proofread astrocytes:", len(keep_ids))
        if len(keep_ids) == 0:
            print("No proofread astrocytes found. Exiting.")
            return
        known_ids, all_astro_ids = get_cell_type_ids(client, cfg)
        unp_ids, unp_pos_nm = get_unproofread_astro_table(client, cfg, keep_ids)

    print(f"Cell-type table: {len(known_ids)} classified cells, "
          f"{len(all_astro_ids)} astrocytes ({len(unp_ids)} unproofread) "
          f"[{setup['ID tables']:.1f}s]")

    # Select astrocytes to process.
    if cfg.astro_id is not None:
        astro_ids = [int(cfg.astro_id)]
    elif cfg.astro_index is None:
        astro_ids = [int(x) for x in keep_ids]
    elif 0 <= cfg.astro_index < len(keep_ids):
        astro_ids = [int(keep_ids[cfg.astro_index])]
    else:
        raise IndexError("astro_index out of range of proofread list.")

    keep_set = set(keep_ids.tolist())
    for astro_id in astro_ids:
        if astro_id not in keep_set:
            print(f"WARNING: {astro_id} is not in the proofread list. Its tiles will "
                  "still be masked with the proofread IDs present, but the cell "
                  "itself will not be labeled.")
        out_dir = cfg.out_dir(astro_id)
        os.makedirs(out_dir, exist_ok=True)
        process_single_astro(astro_id, cfg, keep_ids, all_astro_ids, known_ids,
                             unp_ids, unp_pos_nm, img_path, seg_path, out_dir,
                             timestamp)
