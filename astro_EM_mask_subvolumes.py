#!/usr/bin/env python3
"""
generate_astro_subvols_parallel.py   (instrumented)

Generate (EM image, mask) TIFF pairs around proofread astrocytes, and measure where the time goes.

Intended guarantee (as far as the cell-type table allows):
  * the mask contains EVERY proofread astrocyte present in the tile, and
  * NO astrocyte in the tile is unproofread.
Evaluated against materialization version DATASTACK_VERSION: the segmentation is read with
agglomerate=True at that version's timestamp, so voxel root IDs and the ID tables match.

Funnel for one astrocyte (each step is cheaper than the one after it):
  A. candidate tiles from the astrocyte's mesh bbox (chunk-aligned, full-size, mesh prefilter)
  B. drop tiles already known to be rejected (rejected_tiles.txt cache)        [USE_REJECT_CACHE]
  C. drop tiles touched by meshes of nearby UNPROOFREAD astrocytes, order the
     rest cleanest-first (no voxel downloads)                                   [USE_NEIGHBOR_PREFILTER]
  D. per tile, in a thread pool (one CloudVolume pair per thread):
       0. optional coarse-mip check: reject if an unproofread astrocyte is visible [CHECK_MIP]
       1. full-res segmentation download (agglomerated, timestamp-pinned)
       2. skip if no proofread astrocyte present
       3. skip if any unproofread astrocyte present            <- exact check, backstop for B/C/0
       4. optional: skip if too many voxels are in unknown (no-nucleus) segments
       5. optional: skip if the mask is too sparse
       6. mask = all present proofread astrocytes; download EM; write TIFFs
  E. optional early stop after TARGET_WRITTEN tiles                              [TARGET_WRITTEN]

Every run ends with a bottleneck report: per-stage time, time wasted on rejected tiles, worker
utilization, CPU / memory / network, and suggested fixes.

Requirements:
  pip install cloud-volume caveclient numpy scipy tqdm tifffile fastremap psutil
  (psutil is optional; without it the CPU / memory / network section is skipped)
"""

import os
import random
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import fastremap
from scipy.spatial import cKDTree
from tqdm import tqdm

import cloudvolume
from cloudvolume import Bbox

from caveclient import CAVEclient

try:
    import psutil
except ImportError:
    psutil = None

import intervaltree

# --------------------------- CONFIG ---------------------------------------

DATASTACK_NAME = "minnie65_public"
DATASTACK_VERSION = 1822

# Pick astro by index in the proofread list (None = all), or set ASTRO_ID explicitly.
ASTRO_INDEX = None
ASTRO_ID = 864691136010788524

# Tile size (voxels, x/y/z) and bbox padding. Multiples of the chunk size align best.
SUBVOLUME_SIZE = (1024, 1024, 128)
PAD_VOX = 0

MESH_LOD = None          # mesh LOD for the target astrocyte's bbox (None = full resolution)
OUT_DIR_BASE = "./astro_subvols"
SAMPLE_N = None            # e.g. 6 for a profiling run (random tiles), None = all tiles
SEED = 0
MIP = 0

CELLTYPE_TABLE = "aibs_metamodel_celltypes_v661"

CV_PARALLEL = 1           # CloudVolume's internal parallelism; keep 1 with many workers
MAX_WORKERS = 1        # tiles in flight. Each needs several GB of RAM

USE_MESH_PREFILTER = True  # drop tiles with no mesh vertex of the target astrocyte
ALIGN_TO_CHUNKS = True

# --- speedups (toggle one at a time and compare the report) ---
CACHE_ROOT = "./debug_cache"  # local cache for expensive metadata/mesh fetches
USE_REJECT_CACHE = True        # remember empty / unproofread tiles across runs
USE_NEIGHBOR_PREFILTER = True  # drop tiles touched by nearby unproofread astrocyte meshes
NEIGHBOR_MESH_LOD = 3          # coarse is plenty for ~10 um tiles (lower = finer, safer)
NEIGHBOR_MARGIN_NM = 20000     # unproofread somas within this distance of the bbox are checked
NEIGHBOR_THREADS = 3
MESH_CACHE_DIR = os.path.join(CACHE_ROOT, "neighbor_meshes")
CHECK_MIP = 3               # e.g. 2: coarse pre-check before the full-res read (None = off)
TARGET_WRITTEN = 1          # stop after this many written tiles (None = all)

# --- output ---
IMG_COMPRESSION = "zlib"       # None = no compression (fastest writes, biggest files)
IMG_COMPRESSION_LEVEL = 1      # lower = faster; ignored if tifffile doesn't support it
MASK_COMPRESSION = "zlib"      # masks compress extremely well

# --- strictness knobs (None = off; both are always LOGGED so you can tune them) ---
UNKNOWN_FRAC_MAX = None    # skip if > this fraction of voxels are in IDs absent from the cell-type table
MIN_MASK_FRACTION = None   # skip if the mask covers < this fraction of voxels
DROP_OUTERMOST_TILES = False        # Enable/disable edge filtering
OUTERMOST_TILE_DISTANCE = 1        # How many tiles inset from edges (1=remove 1-tile border)

# --- diagnostics ---
LOG_EVERY_TILE = True     # print one timing line per tile
MONITOR_INTERVAL_S = 2.0   # resource sampling interval

# ---------------------------------------------------------------------------

REJECT_CACHE_STATUSES = {"empty", "unproofread", "unproofread_coarse"}
STAGE_ORDER = ["coarse_check", "seg_download", "unique", "checks", "mask",
               "img_download", "write_prep", "write_disk"]

# --- local debug cache -------------------------------------------------------
# Cache expensive, repeatedly fetched metadata/mesh data locally while debugging.
# Cache keys include the datastack version, MIP, and mesh LOD where relevant.
CACHE_ROOT = "./cache"

CACHE_TARGET_MESH = True
CACHE_NEIGHBOR_MESHES = True
CACHE_ID_TABLES = True

# Full-resolution tile caching is intentionally OFF by default.
# A 1280 x 1280 x 256 uint64 segmentation tile is ~3.36 GB uncompressed.
CACHE_TILE_SEG = False
CACHE_TILE_IMAGE = False

# Tile-level caching is intentionally not wired into process_tile yet;
# keep these flags as explicit placeholders until needed for a controlled
# multi-GB debug cache.

ID_CACHE_DIR = os.path.join(
    CACHE_ROOT, f"id_tables_v{DATASTACK_VERSION}"
)
MESH_CACHE_DIR = os.path.join(
    CACHE_ROOT, "neighbor_meshes"
)

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(ID_CACHE_DIR, exist_ok=True)
os.makedirs(MESH_CACHE_DIR, exist_ok=True)

# --- timing utilities ---------------------------------------------------------
# Function to fix Windows perf_counter zero-length interval bug in cloudfiles/intervaltree
def patch_intervaltree():
    """Fix Windows perf_counter zero-length interval bug in cloudfiles."""
    if not getattr(intervaltree.IntervalTree.addi, '_is_safe_patch', False):
        _original_addi = intervaltree.IntervalTree.addi

        def _safe_addi(self, begin, end, data=None):
            if end <= begin:
                end = begin + 1
            return _original_addi(self, begin, end, data)

        _safe_addi._is_safe_patch = True
        intervaltree.IntervalTree.addi = _safe_addi


# Function to set up the CAVEclient and return the client instance
def setup_caveclient(datastack_name, version):
    """
    Set up the CAVEclient for a given datastack and version.

    Parameters:
    - datastack_name: Name of the datastack (e.g., 'minnie65_public').
    - version: Materialization version to use.

    Returns:
    - An instance of CAVEclient configured for the specified datastack and version.
    """
    print(f"Setting up CAVEclient for datastack '{datastack_name}' with version {version}...")
    # desired_resolution=[1,1,1] makes query_table() return position columns
    # already converted to nanometers, matching the units CloudVolume mesh
    # vertices use. Without this, positions come back in native voxel
    # resolution (e.g. 4x4x40nm for minnie65), which silently breaks any
    # spatial comparison against mesh coordinates (bounding-box queries,
    # point-to-mesh distances, etc. will look like they find nothing nearby).
    client = CAVEclient(datastack_name, desired_resolution=[1, 1, 1])
    client.version = version
    patch_intervaltree()
    print(f"CAVEclient setup complete for datastack '{datastack_name}' with version {version}.")

    return client

# --- timing utilities ---------------------------------------------------------

class Lap:
    """Accumulates elapsed time per named stage within one tile."""
    def __init__(self):
        self.t = {}
        self._t0 = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + (now - self._t0)
        self._t0 = now


@contextmanager
def phase(store, name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = store.get(name, 0.0) + time.perf_counter() - t0


class ResourceMonitor:
    """Background sampler: CPU %, process RSS, system free memory, network bytes received."""
    def __init__(self, interval=2.0):
        self.interval = interval
        self.stop_evt = threading.Event()
        self.cpu, self.peak_rss, self.min_avail, self.total_mem = [], 0, None, None
        self.net_bytes, self.wall = 0, 0.0
        self._net0, self._t0, self._th = None, None, None

    def _net(self):
        try:
            return psutil.net_io_counters().bytes_recv
        except Exception:
            return None

    def _run(self):
        proc = psutil.Process()
        psutil.cpu_percent(None)
        while not self.stop_evt.wait(self.interval):
            self.cpu.append(psutil.cpu_percent(None))
            self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
            vm = psutil.virtual_memory()
            self.total_mem = vm.total
            self.min_avail = vm.available if self.min_avail is None else min(self.min_avail, vm.available)

    def start(self):
        if psutil is None:
            return
        self._t0, self._net0 = time.perf_counter(), self._net()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        if psutil is None:
            return
        self.stop_evt.set()
        self._th.join()
        self.wall = time.perf_counter() - self._t0
        n1 = self._net()
        self.net_bytes = (n1 - self._net0) if (n1 is not None and self._net0 is not None) else None


# --- ID tables --------------------------------------------------------------

def get_proofread_astrocyte_root_ids(client) -> np.ndarray:
    """Proofread astrocyte root IDs, using a versioned local cache when available."""
    cache_path = os.path.join(
        ID_CACHE_DIR, "proofread_astrocytes.npy"
    )

    if CACHE_ID_TABLES and os.path.exists(cache_path):
        try:
            ids = np.load(cache_path)
            print(f"Proofread astrocyte ID cache hit: {cache_path}")
            return ids.astype(np.uint64, copy=False)
        except Exception as e:
            print(f"Warning: could not read proofread ID cache; refetching: {e}")

    print("Fetching proofread astrocyte IDs...")
    df = client.materialize.tables.vortex_astrocyte_proofreading_status(
        status="clean"
    ).query(split_positions=True)

    ids = np.unique(
        df["pt_root_id"].to_numpy().astype(np.uint64)
    )

    if CACHE_ID_TABLES:
        tmp_path = cache_path[:-4] + ".tmp.npy"
        np.save(tmp_path, ids)
        os.replace(tmp_path, cache_path)
        print(f"Saved proofread astrocyte ID cache: {cache_path}")

    return ids


def get_cell_type_ids(client) -> Tuple[np.ndarray, np.ndarray]:
    """Known classified IDs and astrocyte IDs, using a local cache when available."""
    known_path = os.path.join(ID_CACHE_DIR, "known_ids.npy")
    astro_path = os.path.join(ID_CACHE_DIR, "astro_ids.npy")

    if CACHE_ID_TABLES and os.path.exists(known_path) and os.path.exists(astro_path):
        try:
            known_ids = np.load(known_path).astype(np.uint64, copy=False)
            astro_ids = np.load(astro_path).astype(np.uint64, copy=False)
            print(f"Cell-type ID cache hit: {ID_CACHE_DIR}")
            return known_ids, astro_ids
        except Exception as e:
            print(f"Warning: could not read cell-type ID cache; refetching: {e}")

    print("Fetching cell-type table...")
    df = client.materialize.query_table(CELLTYPE_TABLE)
    ids = df["pt_root_id"].to_numpy().astype(np.uint64)
    is_astro = (df["cell_type"] == "astrocyte").to_numpy()
    nz = ids != 0

    known_ids = np.unique(ids[nz])
    astro_ids = np.unique(ids[nz & is_astro])

    if CACHE_ID_TABLES:
        known_tmp = known_path + ".tmp"
        astro_tmp = astro_path + ".tmp"

        with open(known_tmp, "wb") as f:
            np.save(f, known_ids)

        with open(astro_tmp, "wb") as f:
            np.save(f, astro_ids)

        os.replace(known_tmp, known_path)
        os.replace(astro_tmp, astro_path)

        print(f"Saved cell-type ID cache: {ID_CACHE_DIR}")

    return known_ids, astro_ids


def get_unproofread_astro_table(client, keep_ids) -> Tuple[np.ndarray, np.ndarray]:
    """IDs and soma positions (nm: setup_caveclient uses desired_resolution=[1,1,1]) of unproofread astrocytes."""
    df = client.materialize.query_table(CELLTYPE_TABLE, filter_equal_dict={"cell_type": "astrocyte"})
    df = df[df["pt_root_id"] != 0].drop_duplicates("pt_root_id")
    ids = df["pt_root_id"].to_numpy().astype(np.uint64)
    pos_nm = np.stack(df["pt_position"].to_numpy()).astype(float)
    m = ~np.isin(ids, keep_ids)
    return ids[m], pos_nm[m]


# --- volumes ----------------------------------------------------------------

def load_vol(cloudpath, mip=0, progress=False, use_https=True, parallel=1):
    return cloudvolume.CloudVolume(
        cloudpath, mip=mip, progress=progress,
        use_https=use_https, parallel=parallel, agglomerate=True,
    )


_tls = threading.local()

def get_thread_vols(img_path, seg_path):
    """One (img_vol, seg_vol) pair per thread; CloudVolume objects aren't guaranteed thread-safe."""
    if not hasattr(_tls, "vols"):
        _tls.vols = (load_vol(img_path, mip=MIP, parallel=CV_PARALLEL),
                     load_vol(seg_path, mip=MIP, parallel=CV_PARALLEL))
    return _tls.vols


def get_thread_seg_vol(seg_path):
    if not hasattr(_tls, "seg"):
        _tls.seg = load_vol(seg_path, mip=MIP, parallel=1)
    return _tls.seg


def get_thread_coarse_vol(seg_path, check_mip):
    if not hasattr(_tls, "coarse"):
        _tls.coarse = load_vol(seg_path, mip=check_mip, parallel=1)
    return _tls.coarse


# --- tiling -------------------------------------------------------------------

def fetch_mesh_bbox_fast(seg_vol, segid, lod=None, mip=0, base_pad_vox=8):
    """Get target mesh vertices, using a versioned local cache when available."""
    segid = int(segid)

    lod_tag = "full" if lod is None else f"lod{lod}"
    cache_dir = os.path.join(CACHE_ROOT, "target_meshes")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir,
        f"{segid}_{lod_tag}_mip{mip}_v{DATASTACK_VERSION}.npz"
    )

    if CACHE_TARGET_MESH and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            verts_vox = data["verts_vox"]
            mins_vox = tuple(data["mins_vox"].tolist())
            maxs_vox = tuple(data["maxs_vox"].tolist())
            print(f"Target mesh cache hit: {cache_path}")
            return mins_vox, maxs_vox, verts_vox
        except Exception as e:
            print(
                f"Warning: could not read target mesh cache; refetching: {e}"
            )
            try:
                os.remove(cache_path)
            except OSError:
                pass

    print(f"Fetching target mesh: {segid}")

    if lod is None:
        mesh = seg_vol.mesh.get(segid)[segid]
    else:
        mesh = seg_vol.mesh.get(segid, lod=lod)[segid]

    resolution = np.array(
        seg_vol.meta.resolution(mip), dtype=float
    )
    verts_vox = np.floor(
        mesh.vertices / resolution
    ).astype(np.int64)

    mins_vox = verts_vox.min(axis=0) - base_pad_vox
    maxs_vox = verts_vox.max(axis=0) + 1 + base_pad_vox

    if CACHE_TARGET_MESH:
        tmp_path = cache_path[:-4] + ".tmp.npz"
        np.savez(
            tmp_path,
            verts_vox=verts_vox,
            mins_vox=np.asarray(mins_vox, dtype=np.int64),
            maxs_vox=np.asarray(maxs_vox, dtype=np.int64),
        )
        os.replace(tmp_path, cache_path)
        print(f"Saved target mesh cache: {cache_path}")

    return (
        tuple(int(x) for x in mins_vox),
        tuple(int(x) for x in maxs_vox),
        verts_vox,
    )


def align_origin(mins, bounds_min, chunk):
    """Snap mins down to the storage chunk grid."""
    return tuple(int(b + ((m - b) // c) * c) for m, b, c in zip(mins, bounds_min, chunk))


def occupied_tile_indices(verts_vox, origin, tile_size):
    """Tile grid indices containing at least one mesh vertex."""
    idx = (verts_vox - np.array(origin)) // np.array(tile_size)
    idx = idx[(idx >= 0).all(axis=1)]
    return set(map(tuple, np.unique(idx, axis=0)))


def iterate_tiles(mins, maxs, tile_size: Tuple[int, int, int]) -> Iterable:
    sx, sy, sz = tile_size
    minx, miny, minz = mins
    maxx, maxy, maxz = maxs
    for x0 in range(minx, maxx, sx):
        x1 = min(x0 + sx, maxx)
        for y0 in range(miny, maxy, sy):
            y1 = min(y0 + sy, maxy)
            for z0 in range(minz, maxz, sz):
                z1 = min(z0 + sz, maxz)
                yield (x0, y0, z0), (x1, y1, z1)


def compute_tiles(seg_vol, img_vol, astro_id, subvolume_size, pad_vox, mip, mesh_lod,
                  align=True, use_prefilter=True):
    mins_raw, maxs, verts_vox = fetch_mesh_bbox_fast(
        seg_vol, astro_id, lod=mesh_lod, mip=mip, base_pad_vox=pad_vox)

    chunk = np.lcm(np.array(seg_vol.meta.chunk_size(mip)),
                   np.array(img_vol.meta.chunk_size(mip)))
    mins = mins_raw
    if align:
        bounds_min = np.array(seg_vol.meta.bounds(mip).minpt, dtype=int)
        mins = align_origin(mins_raw, bounds_min, chunk)

    all_tiles = list(iterate_tiles(mins, maxs, subvolume_size))
    full_tiles = [(s, e) for s, e in all_tiles
                  if tuple(np.subtract(e, s)) == tuple(subvolume_size)]

    kept_tiles = full_tiles
    if use_prefilter:
        occupied = occupied_tile_indices(verts_vox, mins, subvolume_size)
        kept_tiles = [(s, e) for s, e in full_tiles
                      if tuple((np.array(s) - np.array(mins)) // np.array(subvolume_size)) in occupied]

    return dict(mins_raw=mins_raw, mins=mins, maxs=maxs, verts_vox=verts_vox, chunk=chunk,
                all_tiles=all_tiles, full_tiles=full_tiles, kept_tiles=kept_tiles)


# --- tile prefilters ------------------------------------------------------------

def get_neighbor_verts(segid, seg_path):
    """Get coarse neighbor mesh vertices, using a versioned local cache."""
    segid = int(segid)

    cache_path = os.path.join(
        MESH_CACHE_DIR,
        f"{segid}_lod{NEIGHBOR_MESH_LOD}_mip{MIP}_v{DATASTACK_VERSION}.npy"
    )

    if CACHE_NEIGHBOR_MESHES and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception as e:
            tqdm.write(
                f"  corrupt mesh cache for {segid}; deleting: {e}"
            )
            try:
                os.remove(cache_path)
            except OSError:
                pass

    try:
        vol = get_thread_seg_vol(seg_path)

        try:
            mesh = vol.mesh.get(
                segid, lod=NEIGHBOR_MESH_LOD
            )[segid]
        except Exception:
            mesh = vol.mesh.get(segid)[segid]

        res = np.array(
            vol.meta.resolution(MIP), dtype=float
        )

        verts = np.floor(
            mesh.vertices / res
        ).astype(np.int64)

        if CACHE_NEIGHBOR_MESHES:
            tmp_path = cache_path[:-4] + ".tmp.npy"
            np.save(tmp_path, verts)
            os.replace(tmp_path, cache_path)

        return verts

    except Exception as e:
        tqdm.write(
            f"  neighbor mesh failed for {segid}: "
            f"{type(e).__name__}: {e}"
        )
        return None


def drop_tiles_near_unproofread(tiles, info, unp_ids, unp_pos_nm, seg_vol, seg_path):
    """Remove tiles touched by nearby unproofread astrocyte meshes; order the rest cleanest-first."""
    res = np.array(seg_vol.meta.resolution(MIP), dtype=float)
    mins, maxs = np.array(info["mins"]), np.array(info["maxs"])
    pos_vox = unp_pos_nm / res

    bounds = seg_vol.meta.bounds(MIP)
    inside = ((pos_vox >= np.array(bounds.minpt)) & (pos_vox <= np.array(bounds.maxpt))).all(axis=1).mean()
    if inside < 0.9:
        print(f"WARNING: only {inside:.0%} of unproofread soma positions fall inside the volume "
              f"bounds; positions may not be in nm. Neighbor prefilter will be unreliable.")

    margin = np.array(NEIGHBOR_MARGIN_NM) / res
    near = ((pos_vox >= mins - margin) & (pos_vox <= maxs + margin)).all(axis=1)
    neighbors = unp_ids[near]
    if len(neighbors) == 0:
        print("Neighbor prefilter: no unproofread astrocytes near this bbox.")
        return tiles

    with ThreadPoolExecutor(max_workers=NEIGHBOR_THREADS) as ex:
        results = list(tqdm(ex.map(lambda n: get_neighbor_verts(n, seg_path), neighbors),
                            total=len(neighbors), desc="Neighbor meshes"))
    verts_list = [v for v in results if v is not None]
    n_failed = len(results) - len(verts_list)

    bad = set()
    for v in verts_list:
        bad |= occupied_tile_indices(v, info["mins"], SUBVOLUME_SIZE)

    tile_sz = np.array(SUBVOLUME_SIZE)
    kept = [(s, e) for s, e in tiles if tuple((np.array(s) - mins) // tile_sz) not in bad]
    print(f"Neighbor prefilter: {len(neighbors)} nearby unproofread astrocytes "
          f"({n_failed} meshes failed); kept {len(kept)} of {len(tiles)} tiles")

    if kept and verts_list:                                           # cleanest tiles first
        allv = np.concatenate(verts_list)[::5] * res
        centers = np.array([(np.array(s) + np.array(e)) / 2 for s, e in kept]) * res
        d, _ = cKDTree(allv).query(centers)
        kept = [kept[i] for i in np.argsort(-d)]
    return kept

def drop_outermost_tiles(tiles, info, distance_vox=1):
    """
    Remove tiles that are on the outer edges of the bounding box.

    Args:
        tiles: List of ((x0, y0, z0), (x1, y1, z1)) tile tuples
        info: Dict from compute_tiles() containing 'mins', 'maxs', 'verts_vox'
        distance_vox: How many tile-widths to inset from the edges (default 1)

    Returns:
        Filtered list of inner tiles
    """
    mins = np.array(info["mins"])
    maxs = np.array(info["maxs"])
    tile_sz = np.array(SUBVOLUME_SIZE)

    # Compute tile grid indices
    tile_indices = [
        tuple((np.array(s) - mins) // tile_sz)
        for s, e in tiles
    ]

    # Find the grid bounds
    tile_indices_arr = np.array(tile_indices)
    grid_min = tile_indices_arr.min(axis=0)
    grid_max = tile_indices_arr.max(axis=0)

    # Keep only tiles that are inset by distance_vox from edges
    inner = [
        (s, e) for (s, e), idx in zip(tiles, tile_indices)
        if (idx >= grid_min + distance_vox).all() and
           (idx <= grid_max - distance_vox).all()
    ]

    n_dropped = len(tiles) - len(inner)
    print(f"Outermost filter: dropped {n_dropped} edge tiles, kept {len(inner)}")
    return inner

def coarse_has_unproofread(start, end, seg_path, check_mip, keep_ids, all_astro_ids, timestamp):
    """True if an unproofread astrocyte is visible in the tile at a coarse mip (positive sighting only)."""
    vol = get_thread_coarse_vol(seg_path, check_mip)
    r0 = np.array(vol.meta.resolution(MIP), dtype=float)
    r1 = np.array(vol.meta.resolution(check_mip), dtype=float)       # per axis, so z scaling is handled
    lo = np.floor(np.array(start) * r0 / r1).astype(int)
    hi = np.ceil(np.array(end) * r0 / r1).astype(int)
    b = vol.meta.bounds(check_mip)
    lo, hi = np.maximum(lo, np.array(b.minpt)), np.minimum(hi, np.array(b.maxpt))
    if (hi <= lo).any():
        return False
    seg = vol.download(Bbox(lo.tolist(), hi.tolist()), mip=check_mip, agglomerate=True, timestamp=timestamp)
    ids = fastremap.unique(seg)
    return bool((np.isin(ids, all_astro_ids) & ~np.isin(ids, keep_ids)).any())


# --- reject cache -----------------------------------------------------------------

def load_rejected(path, signature):
    """Tiles known to be unusable. Ignored (and reset) if the grid/version signature changed."""
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
        if lines and lines[0] == signature:
            return {tuple(map(int, ln.split())) for ln in lines[1:] if ln.strip()}
    with open(path, "w") as f:
        f.write(signature + "\n")
    return set()


# --- writing ------------------------------------------------------------------

def write_tiff(path, arr, compression, level):
    kw = dict(bigtiff=True)
    if compression:
        kw["compression"] = compression
        if level is not None:
            kw["compressionargs"] = {"level": level}
    try:
        tifffile.imwrite(path, arr, **kw)
    except (TypeError, ValueError):
        kw.pop("compressionargs", None)                  # older tifffile: no compressionargs
        tifffile.imwrite(path, arr, **kw)


def save_tile_pair(base_path: str, image: np.ndarray, mask: np.ndarray):
    """
    Write {base_path}_image.tif / _mask.tif as (Z, Y, X) stacks via temp file + rename.
    Returns (prep_seconds, disk_seconds).
    """
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    if image.ndim == 4:
        image = image[..., 0]
    if mask.ndim == 4:
        mask = mask[..., 0]
    image = np.ascontiguousarray(image.transpose(2, 1, 0))
    mask = np.ascontiguousarray((mask.transpose(2, 1, 0) != 0).astype(np.uint8))
    t1 = time.perf_counter()

    img_final, mask_final = base_path + "_image.tif", base_path + "_mask.tif"
    img_tmp, mask_tmp = img_final + ".tmp", mask_final + ".tmp"
    write_tiff(img_tmp, image, IMG_COMPRESSION, IMG_COMPRESSION_LEVEL)
    write_tiff(mask_tmp, mask, MASK_COMPRESSION, None)
    os.replace(img_tmp, img_final)      # only complete files get the final name,
    os.replace(mask_tmp, mask_final)    # so the "already exists" resume check is reliable
    return t1 - t0, time.perf_counter() - t1


# --- per-tile worker ----------------------------------------------------------

def process_tile(start, end, astro_id, keep_ids, all_astro_ids, known_ids,
                 img_path, seg_path, out_dir, timestamp, check_mip):
    """Returns {'status', 'start', 'timings', 'total', ...}. Timings are seconds per stage."""
    lap = Lap()

    def done(status, **kw):
        return dict(status=status, start=start, timings=lap.t, total=sum(lap.t.values()), **kw)

    x0, y0, z0 = start
    x1, y1, z1 = end
    base_path = os.path.join(out_dir, f"{astro_id}_{x0}_{y0}_{z0}")
    if os.path.exists(base_path + "_image.tif") and os.path.exists(base_path + "_mask.tif"):
        return done("existing")

    img_vol, seg_vol = get_thread_vols(img_path, seg_path)

    # 0. coarse pre-check (rejects only on a positive sighting of an unproofread astrocyte)
    if check_mip is not None:
        lap.lap("_start")
        if coarse_has_unproofread(start, end, seg_path, check_mip, keep_ids, all_astro_ids, timestamp):
            lap.lap("coarse_check")
            return done("unproofread_coarse")
        lap.lap("coarse_check")
        lap.t.pop("_start", None)

    # 1. full-res segmentation at the version's timestamp
    lap._t0 = time.perf_counter()
    seg = seg_vol.download(Bbox(start, end), mip=MIP, agglomerate=True, timestamp=timestamp)
    lap.lap("seg_download")

    ids, counts = fastremap.unique(seg, return_counts=True)
    lap.lap("unique")

    nonzero = ids != 0
    is_keep = np.isin(ids, keep_ids)

    # 2. at least one proofread astrocyte present
    if not is_keep.any():
        lap.lap("checks")
        return done("empty")

    # 3. no unproofread astrocyte present
    n_unproofread = int((np.isin(ids, all_astro_ids) & ~is_keep).sum())
    if n_unproofread > 0:
        lap.lap("checks")
        return done("unproofread", n_unproofread=n_unproofread)

    # 4. unknown-segment fraction (no nucleus in the table)
    unknown = nonzero & ~np.isin(ids, known_ids)
    unknown_frac = float(counts[unknown].sum()) / seg.size
    if UNKNOWN_FRAC_MAX is not None and unknown_frac > UNKNOWN_FRAC_MAX:
        lap.lap("checks")
        return done("unverified", unknown_frac=unknown_frac)

    # 5. mask coverage
    mask_frac = float(counts[is_keep].sum()) / seg.size
    if MIN_MASK_FRACTION is not None and mask_frac < MIN_MASK_FRACTION:
        lap.lap("checks")
        return done("sparse", mask_frac=mask_frac, unknown_frac=unknown_frac)
    lap.lap("checks")

    # 6. mask = every proofread astrocyte present; then EM and write
    mask = fastremap.mask_except(seg, ids[is_keep].tolist(), in_place=True)
    lap.lap("mask")

    img = img_vol[x0:x1, y0:y1, z0:z1]
    lap.lap("img_download")

    t_prep, t_disk = save_tile_pair(base_path, img, mask)
    lap.t["write_prep"], lap.t["write_disk"] = t_prep, t_disk
    lap._t0 = time.perf_counter()

    return done("written", mask_frac=mask_frac, unknown_frac=unknown_frac)


# --- reporting ----------------------------------------------------------------

def summarize(name, values):
    if values:
        p = np.percentile(values, [0, 25, 50, 75, 100])
        print(f"  {name}: min {p[0]:.2%} | 25% {p[1]:.2%} | median {p[2]:.2%} | "
              f"75% {p[3]:.2%} | max {p[4]:.2%}")


def print_report(astro_id, results, wall, phases, mon, n_workers):
    print("\n" + "=" * 78)
    print(f"BOTTLENECK REPORT  astro {astro_id}")
    print("=" * 78)

    if phases:
        print("Setup phases (main thread):")
        for k, v in phases.items():
            print(f"  {k:<28s} {v:7.1f}s")

    real = [r for r in results if r["status"] != "existing"]
    if not real:
        print("No tiles were processed (all existing / cancelled).")
        return

    stage_tot, stage_n = defaultdict(float), Counter()
    by_status = defaultdict(lambda: [0, 0.0])
    for r in real:
        for k, v in r["timings"].items():
            stage_tot[k] += v
            stage_n[k] += 1
        by_status[r["status"]][0] += 1
        by_status[r["status"]][1] += r["total"]
    tile_sec = sum(stage_tot.values()) or 1e-9

    print(f"\nTiles processed: {len(real)} in {wall/60:.1f} min wall "
          f"({len(real) / max(wall, 1e-9) * 60:.2f} tiles/min)")
    print("\nTime by stage (summed over tiles / threads):")
    print(f"  {'stage':<14s}{'total s':>10s}{'share':>8s}{'avg s':>9s}{'n':>6s}")
    for k in STAGE_ORDER:
        if k in stage_tot:
            print(f"  {k:<14s}{stage_tot[k]:>10.1f}{stage_tot[k] / tile_sec:>8.0%}"
                  f"{stage_tot[k] / stage_n[k]:>9.1f}{stage_n[k]:>6d}")

    print("\nTime by outcome:")
    print(f"  {'status':<20s}{'tiles':>6s}{'total s':>10s}{'avg s':>9s}{'share':>8s}")
    for k, (n, t) in sorted(by_status.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:<20s}{n:>6d}{t:>10.1f}{t / n:>9.1f}{t / tile_sec:>8.0%}")

    written_t = by_status["written"][1] if "written" in by_status else 0.0
    waste = 1 - written_t / tile_sec
    util = tile_sec / (max(wall, 1e-9) * n_workers)
    print(f"\nTime spent on tiles that were NOT written: {waste:.0%}")
    print(f"Worker utilization (busy time / (wall x workers)): {util:.0%}")

    hints = []

    if mon is not None and psutil is not None and mon.cpu:
        cpu = float(np.mean(mon.cpu))
        print("\nResources:")
        print(f"  mean CPU (all cores): {cpu:.0f}%")
        print(f"  peak process memory:  {mon.peak_rss / 1e9:.1f} GB"
              + (f" of {mon.total_mem / 1e9:.0f} GB total" if mon.total_mem else ""))
        if mon.min_avail is not None:
            print(f"  lowest free memory:   {mon.min_avail / 1e9:.1f} GB")
        net_mbps = None
        if mon.net_bytes is not None and mon.wall > 0:
            net_mbps = mon.net_bytes / mon.wall / 1e6
            print(f"  avg network in:       {net_mbps:.1f} MB/s")
        mem_tight = (mon.total_mem and mon.min_avail is not None and mon.min_avail < 0.1 * mon.total_mem)
    else:
        cpu, mem_tight = None, False

    # ---- heuristics (suggestions, not proof) ----
    seg_share = stage_tot.get("seg_download", 0) / tile_sec
    write_share = (stage_tot.get("write_prep", 0) + stage_tot.get("write_disk", 0)) / tile_sec
    img_share = stage_tot.get("img_download", 0) / tile_sec
    coarse_n = by_status["unproofread_coarse"][0] if "unproofread_coarse" in by_status else 0

    if waste > 0.6:
        hints.append("Most time goes to tiles that get rejected. Enable / widen the neighbor prefilter "
                     "(USE_NEIGHBOR_PREFILTER, NEIGHBOR_MARGIN_NM, lower NEIGHBOR_MESH_LOD), try CHECK_MIP=2, "
                     "keep USE_REJECT_CACHE on, set TARGET_WRITTEN, or choose a better-surrounded astrocyte.")
    if seg_share > 0.5:
        if mem_tight:
            hints.append("Segmentation download dominates AND memory is nearly exhausted: lower MAX_WORKERS "
                         "or use smaller tiles; swapping would slow everything.")
        elif cpu is not None and cpu < 50 and util > 0.7:
            hints.append(f"Segmentation download dominates while CPU is mostly idle ({cpu:.0f}%): it's "
                         f"latency/network bound. Raise MAX_WORKERS (memory permitting).")
        elif cpu is not None and cpu >= 70:
            hints.append(f"Segmentation download dominates and CPU is busy ({cpu:.0f}%): decoding / agglomeration "
                         f"is the limit, so more workers won't help. Reduce how many tiles reach this step "
                         f"(prefilters, CHECK_MIP) or shrink tiles.")
        else:
            hints.append("Segmentation download dominates. Compare mean CPU and network above: low CPU -> "
                         "more workers; high CPU -> fewer tiles reaching this step.")
    if write_share > 0.2:
        hints.append("Writing is a large share: set IMG_COMPRESSION_LEVEL=1 (or IMG_COMPRESSION=None), "
                     "and check that the output disk isn't slow or network-mounted.")
    if img_share > 0.25:
        hints.append("EM download is a large share of written-tile time; it only runs for accepted tiles, so "
                     "this mostly means few tiles are being rejected. Try more workers if memory allows.")
    if util < 0.7 and len(real) > MAX_WORKERS * 2:
        hints.append(f"Workers are idle {1 - util:.0%} of the time. The tail of the queue or main-thread work "
                     f"(result handling, neighbor setup) may be limiting; raise tile count per worker or check "
                     f"for one very slow tile.")
    if mem_tight:
        hints.append("Free memory fell below 10% of total. Lower MAX_WORKERS or use smaller tiles.")
    if coarse_n:
        avg_c = stage_tot.get("coarse_check", 0) / max(stage_n.get("coarse_check", 1), 1)
        avg_f = stage_tot.get("seg_download", 0) / max(stage_n.get("seg_download", 1), 1)
        hints.append(f"Coarse check rejected {coarse_n} tiles at ~{avg_c:.1f}s each (a full read averages "
                     f"~{avg_f:.1f}s). If the coarse check isn't rejecting many tiles, turn it off.")

    print("\nSuggestions:" if hints else "\nNo obvious single bottleneck from these numbers.")
    for h in hints:
        print(f"  - {h}")
    print("=" * 78)


# --- per-astrocyte driver -------------------------------------------------------

def process_single_astro(astro_id, keep_ids, all_astro_ids, known_ids, unp_ids, unp_pos_nm,
                         img_path, seg_path, out_dir, timestamp):
    print(f"\nProcessing astro_id={astro_id}, out_dir={out_dir}")
    phases = {}

    img_vol = load_vol(img_path, mip=MIP, parallel=CV_PARALLEL)
    seg_vol = load_vol(seg_path, mip=MIP, parallel=CV_PARALLEL)

    check_mip = CHECK_MIP
    if check_mip is not None and check_mip not in list(seg_vol.available_mips):
        print(f"WARNING: CHECK_MIP={check_mip} not available in this volume; disabling coarse check.")
        check_mip = None

    # A. candidate tiles from the target astrocyte's mesh
    try:
        with phase(phases, "target mesh + tiling"):
            info = compute_tiles(seg_vol, img_vol, astro_id, SUBVOLUME_SIZE, PAD_VOX, MIP, MESH_LOD,
                                 align=ALIGN_TO_CHUNKS, use_prefilter=USE_MESH_PREFILTER)
    except Exception as e:
        print(f"Could not compute tiles for {astro_id}: {e}\n"
              f"(If the mesh is missing, this root may have been edited after v{DATASTACK_VERSION}.)")
        return

    tiles = info["kept_tiles"]
    print(f"Voxel bbox: {info['mins']} -> {info['maxs']}")
    print(f"Funnel: {len(tiles)} candidate tiles (full-size, touched by the astrocyte mesh)")
    if not tiles:
        print("No tiles to process for this astro.")
        return

    # B. drop tiles already known to be rejected
    rej_file = None
    if USE_REJECT_CACHE:
        sig = f"v{DATASTACK_VERSION}|{tuple(SUBVOLUME_SIZE)}|{tuple(info['mins'])}|mip{MIP}"
        rej_path = os.path.join(out_dir, "rejected_tiles.txt")
        rejected = load_rejected(rej_path, sig)
        n0 = len(tiles)
        tiles = [t for t in tiles if t[0] not in rejected]
        print(f"Funnel: {len(tiles)} after reject cache (-{n0 - len(tiles)})")
        rej_file = open(rej_path, "a")

    # C. drop tiles touched by unproofread neighbors, order cleanest first
    if USE_NEIGHBOR_PREFILTER and tiles:
        with phase(phases, "neighbor prefilter"):
            tiles = drop_tiles_near_unproofread(tiles, info, unp_ids, unp_pos_nm, seg_vol, seg_path)
        print(f"Funnel: {len(tiles)} after neighbor prefilter")

    # D. Drop outermost tiles
    if DROP_OUTERMOST_TILES and tiles:  # Only if we have tiles left
        tiles = drop_outermost_tiles(tiles, info, distance_vox=OUTERMOST_TILE_DISTANCE)
        print(f"Funnel: {len(tiles)} after outermost filter")

    if SAMPLE_N and SAMPLE_N < len(tiles):
        tiles = random.Random(SEED).sample(tiles, SAMPLE_N)
        print(f"Sampling {len(tiles)} tiles")

    if not tiles:
        print("No tiles left to process.")
        if rej_file:
            rej_file.close()
        return

    print(f"Processing {len(tiles)} tiles with {MAX_WORKERS} workers "
          f"(CloudVolume parallel={CV_PARALLEL}, coarse check mip={check_mip})"
          + (f", stopping after {TARGET_WRITTEN} written" if TARGET_WRITTEN else ""))

    status_counts = Counter()
    results, mask_fracs, unknown_fracs, unproofread_counts = [], [], [], Counter()
    stopped_early = False

    mon = ResourceMonitor(MONITOR_INTERVAL_S)
    mon.start()
    t_wall = time.perf_counter()

    # D + E. exact per-tile processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_tile, s, e, astro_id, keep_ids, all_astro_ids, known_ids,
                      img_path, seg_path, out_dir, timestamp, check_mip): s
            for s, e in tiles
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Tiles")
        for fut in pbar:
            if fut.cancelled():
                status_counts["cancelled"] += 1
                continue
            start = futures[fut]
            try:
                r = fut.result()
            except Exception as e:                    # one bad tile shouldn't kill the run;
                status_counts["failed"] += 1          # it has no files, so a rerun retries it
                tqdm.write(f"Tile {start} FAILED: {type(e).__name__}: {e}")
                pbar.set_postfix(dict(status_counts))
                continue

            status_counts[r["status"]] += 1
            results.append(r)
            if "unknown_frac" in r:
                unknown_fracs.append(r["unknown_frac"])
            if r["status"] == "written":
                mask_fracs.append(r["mask_frac"])
            if r["status"] == "unproofread":
                unproofread_counts[r["n_unproofread"]] += 1
            if rej_file and r["status"] in REJECT_CACHE_STATUSES:
                rej_file.write("%d %d %d\n" % tuple(start))
                rej_file.flush()

            if LOG_EVERY_TILE:
                tqdm.write(f"{start} {r['status']:<18s} {r['total']:6.1f}s  " +
                           " ".join(f"{k}={v:.1f}" for k, v in r["timings"].items()))
            pbar.set_postfix(dict(status_counts))

            if (TARGET_WRITTEN and not stopped_early
                    and status_counts["written"] + status_counts["existing"] >= TARGET_WRITTEN):
                stopped_early = True
                for f in futures:
                    f.cancel()                        # queued tiles are dropped; running ones finish

    wall = time.perf_counter() - t_wall
    mon.stop()
    if rej_file:
        rej_file.close()

    print(f"\nResults for astro {astro_id} ({len(tiles)} tiles): " +
          ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    if unproofread_counts:
        print("  unproofread astrocytes per skipped tile: " +
              ", ".join(f"{n}: {c} tiles" for n, c in sorted(unproofread_counts.items())))
        print("  (these passed the prefilters; many of them means widen NEIGHBOR_MARGIN_NM "
              "or lower NEIGHBOR_MESH_LOD)")
    summarize("mask fraction (written tiles)", mask_fracs)
    summarize("unknown-segment fraction (tiles passing the astrocyte checks)", unknown_fracs)
    print(f"Output -> {out_dir}")

    print_report(astro_id, results, wall, phases, mon, MAX_WORKERS)


# --- main -------------------------------------------------------------------------

def main():
    setup = {}
    client = setup_caveclient(DATASTACK_NAME, DATASTACK_VERSION)
    img_path = client.info.image_source()
    seg_path = client.info.segmentation_source()
    print(img_path)
    print(seg_path)

    # Pin voxel IDs to the same snapshot as the ID tables
    timestamp = client.materialize.get_timestamp(DATASTACK_VERSION)
    print(f"Segmentation timestamp (v{DATASTACK_VERSION}): {timestamp}")

    with phase(setup, "ID tables"):
        keep_ids = get_proofread_astrocyte_root_ids(client)
        print("Proofread astrocytes:", len(keep_ids))
        if len(keep_ids) == 0:
            print("No proofread astrocytes found. Exiting.")
            return
        known_ids, all_astro_ids = get_cell_type_ids(client)
        unp_ids, unp_pos_nm = get_unproofread_astro_table(client, keep_ids)
    print(f"Cell-type table: {len(known_ids)} classified cells, {len(all_astro_ids)} astrocytes "
          f"({len(unp_ids)} unproofread)  [{setup['ID tables']:.1f}s]")

    if ASTRO_ID is not None:
        astro_ids = [int(ASTRO_ID)]
    elif ASTRO_INDEX is None:
        astro_ids = [int(x) for x in keep_ids]
    else:
        if not 0 <= ASTRO_INDEX < len(keep_ids):
            raise IndexError("ASTRO_INDEX out of range of proofread list.")
        astro_ids = [int(keep_ids[ASTRO_INDEX])]

    keep_set = set(keep_ids.tolist())
    for aid in astro_ids:
        if aid not in keep_set:
            print(f"WARNING: {aid} is not in the proofread list; its tiles will still be masked "
                  f"with the proofread IDs present, but the cell itself will not be labeled.")
        out_dir = os.path.join(OUT_DIR_BASE, str(aid))
        os.makedirs(out_dir, exist_ok=True)
        process_single_astro(aid, keep_ids, all_astro_ids, known_ids, unp_ids, unp_pos_nm,
                             img_path, seg_path, out_dir, timestamp)


if __name__ == "__main__":
    print("Starting generate_astro_subvols_parallel.py")
    main()