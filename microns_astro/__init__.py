"""Tools for extracting EM + astrocyte-mask subvolumes from the MICrONS dataset.

Kept intentionally empty so that `import microns_astro` stays cheap. Import the
submodule you need, for example:

    from microns_astro.tile_io import load_tile_pair   # numpy + tifffile only
    from microns_astro.config import Config            # settings, no heavy imports
    from microns_astro.pipeline import run              # needs caveclient + cloudvolume

See README.md for a table of which function does what.
"""
