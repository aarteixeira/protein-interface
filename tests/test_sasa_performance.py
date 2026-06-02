from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PROTEIN_INTERFACE_PERF") != "1",
    reason="set PROTEIN_INTERFACE_PERF=1 to run performance tests",
)


def _bench(fn, reps: int = 5) -> float:
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def test_sasa_parallel_batch_matches_serial_and_is_faster():
    from protein_interface import AtomArrays, compute_sasa_batch, load_atoms

    data = Path(__file__).parent / "data"
    inputs = []
    for path, chains_a, chains_b in [
        (data / "nb_ag_test.pdb", ["A"], ["L"]),
        (data / "1fyt.pdb", ["D"], ["A"]),
        (data / "1fyt.pdb", ["D", "E"], ["A", "B", "C"]),
    ]:
        a = load_atoms(path, chains_a)
        b = load_atoms(path, chains_b)
        combined = AtomArrays(
            a.coords + b.coords,
            a.atom_names + b.atom_names,
            a.residue_names + b.residue_names,
            a.residue_ids + b.residue_ids,
        )
        inputs.extend([
            (a.coords, a.atom_names, a.residue_names),
            (b.coords, b.atom_names, b.residue_names),
            (combined.coords, combined.atom_names, combined.residue_names),
        ])

    serial = compute_sasa_batch(inputs, 1.4, 92, False)
    parallel = compute_sasa_batch(inputs, 1.4, 92, True)
    assert parallel == serial

    serial_time = _bench(lambda: compute_sasa_batch(inputs, 1.4, 92, False))
    parallel_time = _bench(lambda: compute_sasa_batch(inputs, 1.4, 92, True))
    assert serial_time / parallel_time >= 2.0
