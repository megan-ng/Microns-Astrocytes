#!/usr/bin/env python3
"""Scroll through an (EM, mask) tile pair with a mask overlay.

    python scripts/view_overlay.py astro_subvols_data/<astro_id>/<astro_id>_X_Y_Z

Pass the base path (no "_image.tif"/"_mask.tif" suffix), or a directory to
view the first pair found in it. Scroll wheel / arrow keys change z; the
slider works too.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from astro_subvols.tile_io import list_tile_bases, load_tile_pair


def view_overlay(image, mask, alpha=0.4, title=""):
    """Interactive viewer for (Z, Y, X) image and mask arrays."""
    n_z = image.shape[0]
    lo, hi = np.percentile(image, [1, 99])

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.1)
    z0 = n_z // 2
    im = ax.imshow(image[z0], cmap="gray", vmin=lo, vmax=hi)
    ov = ax.imshow(np.ma.masked_equal(mask[z0], 0), cmap="autumn",
                   alpha=alpha, vmin=0, vmax=1, interpolation="none")
    ax.axis("off")

    slider = Slider(fig.add_axes([0.2, 0.03, 0.6, 0.03]), "z", 0, n_z - 1,
                    valinit=z0, valstep=1)

    def show(z):
        z = int(np.clip(z, 0, n_z - 1))
        im.set_data(image[z])
        ov.set_data(np.ma.masked_equal(mask[z], 0))
        ax.set_title(f"{title}  z={z}/{n_z - 1}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        slider.set_val(np.clip(slider.val + (1 if event.step > 0 else -1), 0, n_z - 1))

    def on_key(event):
        if event.key in ("up", "right"):
            slider.set_val(min(slider.val + 1, n_z - 1))
        elif event.key in ("down", "left"):
            slider.set_val(max(slider.val - 1, 0))

    slider.on_changed(show)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show(z0)
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="tile base path, or a directory of tiles")
    p.add_argument("--alpha", type=float, default=0.4)
    args = p.parse_args()

    base = args.path
    if os.path.isdir(base):
        bases = list_tile_bases(base)
        if not bases:
            raise SystemExit(f"No complete tile pairs found in {base}")
        base = bases[0]

    image, mask = load_tile_pair(base)
    print(f"{os.path.basename(base)}: image {image.shape} {image.dtype}, "
          f"mask {mask.shape}, mask fraction {np.mean(mask > 0):.2%}")
    view_overlay(image, mask, alpha=args.alpha, title=os.path.basename(base))


if __name__ == "__main__":
    main()
