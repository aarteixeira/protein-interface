"""Download test PDB fixtures from RCSB if they are not present."""
from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

_FIXTURES = {
    "1fyt.pdb": "https://files.rcsb.org/download/1FYT.pdb",
    # 1ZVH: anti-lysozyme nanobody (chain A) vs lysozyme (chain L)
    "nb_ag_test.pdb": "https://files.rcsb.org/download/1ZVH.pdb",
    # 1ITB: IL-1beta (chain A) vs type-1 IL-1 receptor ectodomain (chain B)
    "1itb.pdb": "https://files.rcsb.org/download/1ITB.pdb",
}


def pytest_configure(config):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in _FIXTURES.items():
        dest = DATA_DIR / filename
        if not dest.exists():
            print(f"\nDownloading {filename} from {url} …")
            urllib.request.urlretrieve(url, dest)
