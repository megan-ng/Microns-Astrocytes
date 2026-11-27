"""Per-tile processing and timing helpers."""

import time
from contextlib import contextmanager

import fastremap
import numpy as np
from cloudvolume import Bbox

from .filters import coarse_has_unproofread
from .tile_io import save_tile_pair, tile_base_path, tile_pair_exists


class Lap:
    """Accumulate elapsed time per named stage within one tile."""

    def __init__(self):
        self.t, self._t0 = {}, time.perf_counter()

    def reset(self):
        self._t0 = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + now - self._t0
        self._t0 = now


@contextmanager
def phase(store, name):
    """Record elapsed time for a setup phase into `store`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = store.get(name, 0.0) + time.perf_counter() - t0


def process_tile(start, end, astro_id, cfg, keep_ids, all_astro_ids, known_ids,
                 img_vol, seg_vol, coarse_vol, check_mip, out_dir, timestamp):
    """Process one tile; return a dict with status, stage timings, measurements."""
    lap = Lap()

    def done(status, **kw):
        return dict(status=status, start=start, timings=lap.t,
                    total=sum(lap.t.values()), **kw)

    (x0, y0, z0), (x1, y1, z1) = start, end
    base_path = tile_base_path(out_dir, astro_id, start)

    if tile_pair_exists(base_path):
        return done("existing")

    # 0. Optional coarse-mip check.
    lap.reset()
    if coarse_vol is not None:
        rejected = coarse_has_unproofread(start, end, coarse_vol, cfg, check_mip,
                                          keep_ids, all_astro_ids, timestamp)
        lap.lap("coarse_check")
        if rejected:
            return done("unproofread_coarse")

    # 1-2. Full-resolution segmentation and its unique IDs / voxel counts.
    seg = seg_vol.download(Bbox(start, end), mip=cfg.mip, agglomerate=True,
                           timestamp=timestamp)
    lap.lap("seg_download")
    ids, counts = fastremap.unique(seg, return_counts=True)
    lap.lap("unique")

    is_keep = np.isin(ids, keep_ids)

    # 3. At least one proofread astrocyte must be present.
    if not is_keep.any():
        lap.lap("checks")
        return done("empty")

    # 4. No unproofread astrocyte may be present.
    n_unproofread = int((np.isin(ids, all_astro_ids) & ~is_keep).sum())
    if n_unproofread > 0:
        lap.lap("checks")
        return done("unproofread", n_unproofread=n_unproofread)

    # 5. Unknown-segment fraction.
    unknown = (ids != 0) & ~np.isin(ids, known_ids)
    unknown_frac = float(counts[unknown].sum()) / seg.size
    if cfg.unknown_frac_max is not None and unknown_frac > cfg.unknown_frac_max:
        lap.lap("checks")
        return done("unverified", unknown_frac=unknown_frac)

    # 6. Mask coverage.
    mask_frac = float(counts[is_keep].sum()) / seg.size
    if cfg.min_mask_fraction is not None and mask_frac < cfg.min_mask_fraction:
        lap.lap("checks")
        return done("sparse", mask_frac=mask_frac, unknown_frac=unknown_frac)
    lap.lap("checks")

    # 7. Mask contains every proofread astrocyte present in the tile.
    mask = fastremap.mask_except(seg, ids[is_keep].tolist(), in_place=True)
    lap.lap("mask")

    # 8. Download EM, then 9. write TIFFs.
    img = img_vol[x0:x1, y0:y1, z0:z1]
    lap.lap("img_download")
    lap.t["write_prep"], lap.t["write_disk"] = save_tile_pair(
        base_path, img, mask, cfg.img_compression, cfg.img_compression_level,
        cfg.mask_compression)

    return done("written", mask_frac=mask_frac, unknown_frac=unknown_frac)
