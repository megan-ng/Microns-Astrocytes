#!/usr/bin/env python3
"""Generate (EM, mask) tile pairs around proofread astrocytes.

Edit the Config(...) call below, or pass overrides on the command line:
    python scripts/generate_subvols.py --astro-id 864691136010788524 --target-written 5
"""

import argparse

from functions.config import Config
from functions.pipeline import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--astro-id", type=int)
    p.add_argument("--astro-index", type=int)
    p.add_argument("--target-written", type=int)
    p.add_argument("--out-dir-base")
    args = p.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    if "astro_index" in overrides:  # index selection must not be shadowed by the default ID
        overrides.setdefault("astro_id", None)

    run(Config(**overrides))


if __name__ == "__main__":
    main()
