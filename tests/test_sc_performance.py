from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

from protein_interface import compute_sc
from protein_interface.io import _extract_atom_arrays


pytestmark = pytest.mark.skipif(
    os.environ.get("PROTEIN_INTERFACE_PERF") != "1",
    reason="set PROTEIN_INTERFACE_PERF=1 to run SC performance proof",
)


def _timed(fn, reps: int = 8) -> tuple[float, object]:
    fn()
    times = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times[2:]), result


def test_spatial_index_matches_legacy_and_is_at_least_2x_faster():
    model = list(
        PDBParser(QUIET=True)
        .get_structure("1fyt", str(Path("tests/data/1fyt.pdb")))
        .get_models()
    )[0]
    coords_a, names_a, res_a = _extract_atom_arrays(model, ["D", "E"], False, False)
    coords_b, names_b, res_b = _extract_atom_arrays(model, ["A", "B", "C"], False, False)

    legacy_ms, legacy = _timed(
        lambda: compute_sc(
            coords_a, names_a, res_a,
            coords_b, names_b, res_b,
            True,
            False,
        )
    )
    spatial_ms, spatial = _timed(
        lambda: compute_sc(
            coords_a, names_a, res_a,
            coords_b, names_b, res_b,
            True,
            True,
        )
    )

    assert legacy.atoms_a == spatial.atoms_a
    assert legacy.atoms_b == spatial.atoms_b
    assert abs(legacy.sc - spatial.sc) < 1e-9
    assert abs(legacy.median_distance - spatial.median_distance) < 1e-9
    assert abs(legacy.trimmed_area - spatial.trimmed_area) < 1e-9
    assert legacy_ms / spatial_ms >= 2.0, (
        f"expected >=2x speedup, got {legacy_ms / spatial_ms:.2f}x "
        f"(legacy={legacy_ms * 1000:.1f} ms, spatial={spatial_ms * 1000:.1f} ms)"
    )
