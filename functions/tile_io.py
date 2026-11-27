"""Reading and writing (image, mask) TIFF pairs.

Depends only on numpy + tifffile, so analysis/viewer scripts can import it
without pulling in caveclient or cloudvolume.
"""

import os
import time

import numpy as np
import tifffile


def tile_base_path(out_dir, astro_id, start):
    """Single source of truth for the output filename convention."""
    x0, y0, z0 = start
    return os.path.join(out_dir, f"{astro_id}_{x0}_{y0}_{z0}")


def tile_pair_paths(base_path):
    return base_path + "_image.tif", base_path + "_mask.tif"


def tile_pair_exists(base_path):
    return all(os.path.exists(p) for p in tile_pair_paths(base_path))


def load_tile_pair(base_path):
    """Return (image, mask) as (Z, Y, X) arrays."""
    img_path, mask_path = tile_pair_paths(base_path)
    return tifffile.imread(img_path), tifffile.imread(mask_path)


def list_tile_bases(out_dir):
    """Base paths of every complete (image, mask) pair in a directory."""
    bases = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith("_image.tif"):
            base = os.path.join(out_dir, name[:-len("_image.tif")])
            if tile_pair_exists(base):
                bases.append(base)
    return bases


def write_tiff(path, arr, compression, level):
    """Write a TIFF, falling back for older tifffile versions."""
    kwargs = dict(bigtiff=True)
    if compression:
        kwargs["compression"] = compression
        if level is not None:
            kwargs["compressionargs"] = {"level": level}
    try:
        tifffile.imwrite(path, arr, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("compressionargs", None)
        tifffile.imwrite(path, arr, **kwargs)


def save_tile_pair(base_path, image, mask, img_compression="zlib",
                   img_level=1, mask_compression="zlib"):
    """
    Write image and mask TIFFs as (Z, Y, X) stacks. Temp files are renamed
    after writing so incomplete files are never mistaken for finished ones.
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
    img_final, mask_final = tile_pair_paths(base_path)
    write_tiff(img_final + ".tmp", image, img_compression, img_level)
    write_tiff(mask_final + ".tmp", mask, mask_compression, None)
    os.replace(img_final + ".tmp", img_final)
    os.replace(mask_final + ".tmp", mask_final)
    return t1 - t0, time.perf_counter() - t1
