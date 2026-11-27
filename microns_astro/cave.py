"""CAVEclient / CloudVolume setup."""

import cloudvolume
import intervaltree
from caveclient import CAVEclient


def patch_intervaltree():
    """Fix Windows perf_counter zero-length interval bug in cloudfiles."""
    if getattr(intervaltree.IntervalTree.addi, "_is_safe_patch", False):
        return
    original = intervaltree.IntervalTree.addi

    def safe_addi(self, begin, end, data=None):
        return original(self, begin, max(end, begin + 1), data)

    safe_addi._is_safe_patch = True
    intervaltree.IntervalTree.addi = safe_addi


def setup_caveclient(datastack_name, version):
    print(f"Setting up CAVEclient for '{datastack_name}' version {version}...")

    client = CAVEclient(
        datastack_name,
        server_address="https://global.daf-apis.com",
        desired_resolution=[1, 1, 1],
        version=version,
    )

    patch_intervaltree()
    return client


def load_vol(cloudpath, mip=0):
    return cloudvolume.CloudVolume(cloudpath, mip=mip, progress=False,
                                   use_https=True, agglomerate=True)
