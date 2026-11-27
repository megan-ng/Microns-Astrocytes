"""Timing / outcome reporting."""

from collections import Counter, defaultdict

import numpy as np

STAGE_ORDER = ["coarse_check", "seg_download", "unique", "checks", "mask",
               "img_download", "write_prep", "write_disk"]


def summarize(name, values):
    """Print descriptive statistics for a list of fractions."""
    if values:
        p = np.percentile(values, [0, 25, 50, 75, 100])
        print(f"  {name}: min {p[0]:.2%} | 25% {p[1]:.2%} | median {p[2]:.2%} | "
              f"75% {p[3]:.2%} | max {p[4]:.2%}")


def print_report(astro_id, results, wall, phases):
    """Print processing timings and basic bottleneck suggestions."""
    bar = "=" * 78
    print(f"\n{bar}\nPROCESSING REPORT  astro {astro_id}\n{bar}")
    if phases:
        print("Setup phases:")
        for name, elapsed in phases.items():
            print(f"  {name:<28s} {elapsed:7.1f}s")

    real = [r for r in results if r["status"] != "existing"]
    if not real:
        print(f"No new tiles were processed (all tiles already exist).\n{bar}")
        return

    stage_tot, stage_n = defaultdict(float), Counter()
    by_status = defaultdict(lambda: [0, 0.0])
    for r in real:
        for name, elapsed in r["timings"].items():
            stage_tot[name] += elapsed
            stage_n[name] += 1
        by_status[r["status"]][0] += 1
        by_status[r["status"]][1] += r["total"]
    tile_sec = sum(stage_tot.values()) or 1e-9

    print(f"\nTiles processed: {len(real)}")
    print(f"Total wall time: {wall / 60:.1f} min")
    print(f"Processing speed: {len(real) / max(wall, 1e-9) * 60:.2f} tiles/min")

    print("\nTime by stage:")
    print(f"  {'Stage':<16s}{'Total (s)':>12s}{'Share':>10s}{'Avg (s)':>10s}{'Tiles':>8s}")
    for name in STAGE_ORDER:
        if name in stage_tot:
            print(f"  {name:<16s}{stage_tot[name]:>12.1f}"
                  f"{stage_tot[name] / tile_sec:>10.1%}"
                  f"{stage_tot[name] / stage_n[name]:>10.1f}{stage_n[name]:>8d}")

    print("\nTime by outcome:")
    print(f"  {'Status':<22s}{'Tiles':>8s}{'Total (s)':>12s}{'Avg (s)':>10s}")
    for name, (n, elapsed) in sorted(by_status.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<22s}{n:>8d}{elapsed:>12.1f}{elapsed / n:>10.1f}")

    waste = 1 - by_status["written"][1] / tile_sec
    print(f"\nTime spent on tiles that were NOT written: {waste:.1%}")

    share = lambda *names: sum(stage_tot.get(n, 0) for n in names) / tile_sec
    hints = []
    if waste > 0.6:
        hints.append("Most processing time was spent on rejected tiles. "
                     "Consider improving the tile prefilters.")
    if share("seg_download") > 0.5:
        hints.append("Segmentation download dominates processing time. Consider "
                     "reducing the number of tiles reaching this stage or using "
                     "smaller tiles.")
    if share("write_prep", "write_disk") > 0.2:
        hints.append("Writing TIFF files takes a substantial amount of time. "
                     "Consider reducing image compression or checking disk performance.")
    if share("img_download") > 0.25:
        hints.append("EM download takes a substantial amount of time.")

    print("\nSuggestions:" if hints else "\nNo obvious single bottleneck.")
    for hint in hints:
        print(f"  - {hint}")
    print(bar)
