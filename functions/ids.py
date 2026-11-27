"""Proofread / cell-type ID tables, with local caching."""

import os

import numpy as np

from .cache import cached_save


def get_proofread_astrocyte_root_ids(client, cfg):
    """Proofread astrocyte root IDs, using a local cache when available."""
    path = os.path.join(cfg.id_cache_dir, "proofread_astrocytes.npy")
    if cfg.cache_id_tables and os.path.exists(path):
        try:
            print(f"Proofread astrocyte ID cache hit: {path}")
            return np.load(path).astype(np.uint64, copy=False)
        except Exception as e:
            print(f"Warning: could not read proofread ID cache: {e}")

    print("Fetching proofread astrocyte IDs...")
    df = client.materialize.tables.vortex_astrocyte_proofreading_status(
        status="clean").query(split_positions=True)
    ids = np.unique(df["pt_root_id"].to_numpy().astype(np.uint64))
    if cfg.cache_id_tables:
        cached_save(path, lambda f: np.save(f, ids))
    return ids


def get_cell_type_ids(client, cfg):
    """Known classified IDs and astrocyte IDs, using local caches."""
    known_path = os.path.join(cfg.id_cache_dir, "known_ids.npy")
    astro_path = os.path.join(cfg.id_cache_dir, "astro_ids.npy")
    if cfg.cache_id_tables and os.path.exists(known_path) and os.path.exists(astro_path):
        try:
            print(f"Cell-type ID cache hit: {cfg.id_cache_dir}")
            return (np.load(known_path).astype(np.uint64, copy=False),
                    np.load(astro_path).astype(np.uint64, copy=False))
        except Exception as e:
            print(f"Warning: could not read cell-type ID cache: {e}")

    print("Fetching cell-type table...")
    df = client.materialize.query_table(cfg.celltype_table)
    ids = df["pt_root_id"].to_numpy().astype(np.uint64)
    nonzero = ids != 0
    known_ids = np.unique(ids[nonzero])
    astro_ids = np.unique(ids[nonzero & (df["cell_type"] == "astrocyte").to_numpy()])
    if cfg.cache_id_tables:
        cached_save(known_path, lambda f: np.save(f, known_ids))
        cached_save(astro_path, lambda f: np.save(f, astro_ids))
    return known_ids, astro_ids


def get_unproofread_astro_table(client, cfg, keep_ids):
    """Unproofread astrocyte IDs and soma positions (nm)."""
    df = client.materialize.query_table(
        cfg.celltype_table, filter_equal_dict={"cell_type": "astrocyte"})
    df = df[df["pt_root_id"] != 0].drop_duplicates("pt_root_id")
    ids = df["pt_root_id"].to_numpy().astype(np.uint64)
    pos_nm = np.stack(df["pt_position"].to_numpy()).astype(float)
    unp = ~np.isin(ids, keep_ids)
    return ids[unp], pos_nm[unp]
