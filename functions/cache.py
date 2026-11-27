"""Small file helpers shared by the caching code."""

import os


def cached_save(path, saver):
    """Write via a temp file then rename, so partial files are never seen.

    Creates the parent directory lazily.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        saver(f)
    os.replace(tmp, path)


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass
