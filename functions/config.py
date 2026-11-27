"""All tunable settings in one place. No heavy imports, no side effects."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- dataset ---
    datastack_name: str = "minnie65_public"
    # datastack_version: int = 1822
    datastack_version: int = 1926
    celltype_table: str = "aibs_metamodel_celltypes_v661"

    # Pick astrocyte by explicit ID, or by index in the proofread list.
    # If both are None, every proofread astrocyte is processed.
    # astro_id: int | None = 864691136010788524
    astro_id: int | None = 864691136364261346
    astro_index: int | None = None

    # --- tiling ---
    subvolume_size: tuple = (1024, 1024, 128)  # voxels (x, y, z)
    pad_vox: int = 0
    mesh_lod: int | None = None
    mip: int = 0
    use_mesh_prefilter: bool = True
    align_to_chunks: bool = True
    sample_n: int | None = None
    seed: int = 0

    # --- prefilters / caching ---
    cache_root: str = "./cache"
    use_reject_cache: bool = True
    use_neighbor_prefilter: bool = True
    neighbor_mesh_lod: int = 3
    neighbor_margin_nm: float = 20000
    check_mip: int | None = 3
    target_written: int | None = 1
    cache_target_mesh: bool = True
    cache_neighbor_meshes: bool = True
    cache_id_tables: bool = True

    # --- output ---
    out_dir_base: str = "./output"
    img_compression: str | None = "zlib"
    img_compression_level: int | None = 1
    mask_compression: str | None = "zlib"

    # --- strictness knobs (None = off) ---
    unknown_frac_max: float | None = None
    min_mask_fraction: float | None = None
    drop_outermost_tiles: bool = False
    outermost_tile_distance: int = 1

    log_every_tile: bool = True

    # --- derived paths (computed, never created here) ---
    @property
    def id_cache_dir(self) -> str:
        return os.path.join(self.cache_root, f"id_tables_v{self.datastack_version}")

    @property
    def neighbor_mesh_dir(self) -> str:
        return os.path.join(self.cache_root, "neighbor_meshes")

    @property
    def target_mesh_dir(self) -> str:
        return os.path.join(self.cache_root, "target_meshes")

    def out_dir(self, astro_id) -> str:
        return os.path.join(self.out_dir_base, str(astro_id))
