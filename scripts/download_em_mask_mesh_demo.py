#!/usr/bin/env python3
"""
Demo script for generating a single subvolume from a proofread astrocyte, and visualizing the whole mesh.
"""

# --- imports
import numpy as np
import fastremap
import pyvista as pv
from cloudvolume import Bbox
from functions.config import Config
from functions.cave import load_vol, setup_caveclient
from functions.meshes import fetch_mesh_bbox_fast
from functions.ids import get_proofread_astrocyte_root_ids
from functions.tile_io import save_tile_pair, tile_base_path

# --- setup configuration and CAVEclient
cfg = Config()
client =  setup_caveclient(cfg.datastack_name, cfg.datastack_version)
seg_path = client.info.segmentation_source()
img_path = client.info.image_source()
seg_vol = load_vol(seg_path, cfg.mip)
img_vol = load_vol(img_path, cfg.mip)

# --- fetch the bounding box of the proofread astrocyte mesh
mins, maxs, _ = fetch_mesh_bbox_fast(seg_vol, cfg, cfg.astro_id)
astro_bbox = (mins, maxs)
print(f"Astrocyte {cfg.astro_id} bounding box: {astro_bbox}")

# --- define subvolume size and position within the astrocyte bounding box
subvol_size = np.array(cfg.subvolume_size)
subvol_min = np.array([90385, 85860, 21890])  # example starting point within the astrocyte bounding box])
subvol_max = subvol_min + subvol_size
subvol_bbox = (tuple(subvol_min), tuple(subvol_max))

# --- extract the subvolume from the image and segmentation volumes
img = img_vol[subvol_min[0]:subvol_max[0], subvol_min[1]:subvol_max[1], subvol_min[2]:subvol_max[2]]
seg = seg_vol.download(Bbox(subvol_min, subvol_max), mip=cfg.mip, agglomerate=True)
keep_ids = get_proofread_astrocyte_root_ids(client, cfg)
ids, counts = fastremap.unique(seg, return_counts=True)
is_keep = np.isin(ids, keep_ids)
mask = fastremap.mask_except(seg, ids[is_keep].tolist(), in_place=True)

# --- save the subvolumes to disk
base_path = tile_base_path(cfg.out_dir_base, cfg.astro_id, subvol_min)
save_tile_pair(base_path, img, mask, cfg.img_compression, cfg.img_compression_level, cfg.mask_compression)
print(f"Saved subvolumes to {base_path}")

# --- download and visualize astrocyte mesh
# fetch mesh
mesh = seg_vol.mesh.get(cfg.astro_id)[cfg.astro_id]

# convert mesh to PyVista format
vertices = mesh.vertices
faces = mesh.faces
padded_faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)
mesh_poly = pv.PolyData(vertices, faces=padded_faces)

# plot mesh
plotter = pv.Plotter()
plotter.add_mesh(
    mesh_poly,
    color="lightblue",
    show_edges=False
)
plotter.show()
