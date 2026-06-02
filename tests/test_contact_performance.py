from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest


def _bench(fn, reps: int = 7) -> float:
    for _ in range(2):
        fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _dense_contact_summary(a, b):
    from protein_interface.interface import _ic_bins_from_contacts

    A = np.asarray(a.coords)
    B = np.asarray(b.coords)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    within_5 = d2 <= 25.0
    within_55 = d2 <= 30.25
    bins = _ic_bins_from_contacts(a, b, within_55)
    bins["total"] = sum(bins.values())
    return (
        int(within_5.sum()),
        frozenset(a.residue_ids[i] for i in np.flatnonzero(within_5.any(axis=1))),
        frozenset(b.residue_ids[j] for j in np.flatnonzero(within_5.any(axis=0))),
        bins,
    )


def _spatial_contact_summary(a, b):
    from protein_interface import find_contact_pairs
    from protein_interface.interface import _ic_bins_from_contact_pairs

    pairs = find_contact_pairs(a.coords, b.coords, 5.5)
    pairs_5 = [(ai, bi) for ai, bi, d2 in pairs if d2 <= 25.0]
    pairs_55 = [(ai, bi) for ai, bi, d2 in pairs if d2 <= 30.25]
    bins = _ic_bins_from_contact_pairs(a, b, pairs_55)
    bins["total"] = sum(bins.values())
    return (
        len(pairs_5),
        frozenset(a.residue_ids[i] for i, _ in pairs_5),
        frozenset(b.residue_ids[j] for _, j in pairs_5),
        bins,
    )


def _python_salt_bridges(a, b):
    from protein_interface.interface import ANION_ATOMS, CATION_ATOMS

    pairs = set()
    cutoff2 = 16.0
    for i, ca in enumerate(a.coords):
        a_anion = (a.residue_names[i], a.atom_names[i]) in ANION_ATOMS
        a_cation = (a.residue_names[i], a.atom_names[i]) in CATION_ATOMS
        if not a_anion and not a_cation:
            continue
        x = np.asarray(ca)
        for j, cb in enumerate(b.coords):
            b_anion = (b.residue_names[j], b.atom_names[j]) in ANION_ATOMS
            b_cation = (b.residue_names[j], b.atom_names[j]) in CATION_ATOMS
            if not ((a_anion and b_cation) or (a_cation and b_anion)):
                continue
            y = np.asarray(cb)
            if float((x - y) @ (x - y)) <= cutoff2:
                pairs.add((a.residue_ids[i], b.residue_ids[j]))
    return len(pairs)


@pytest.mark.skipif(
    os.environ.get("PROTEIN_INTERFACE_PERF") != "1",
    reason="set PROTEIN_INTERFACE_PERF=1 to run performance checks",
)
def test_contact_and_salt_paths_match_reference_and_are_faster():
    from protein_interface import load_atoms, salt_bridges

    root = Path(__file__).parent
    a = load_atoms(root / "data" / "1fyt.pdb", ["D", "E"])
    b = load_atoms(root / "data" / "1fyt.pdb", ["A", "B", "C"])

    assert _spatial_contact_summary(a, b) == _dense_contact_summary(a, b)
    assert salt_bridges(a, b) == _python_salt_bridges(a, b)

    dense_contact_ms = _bench(lambda: _dense_contact_summary(a, b))
    spatial_contact_ms = _bench(lambda: _spatial_contact_summary(a, b))
    python_salt_ms = _bench(lambda: _python_salt_bridges(a, b))
    rust_salt_ms = _bench(lambda: salt_bridges(a, b))

    assert dense_contact_ms / spatial_contact_ms >= 2.0
    assert python_salt_ms / rust_salt_ms >= 2.0
