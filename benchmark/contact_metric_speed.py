"""Benchmark contact-family metrics against dense/Python reference paths.

The reference paths mirror the previous implementation style: dense numpy
atom-pair matrices for contact summaries and a Python loop for salt-bridge
residue-pair counting.

Run:
    python benchmark/contact_metric_speed.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from protein_interface import analyze, find_contact_pairs, load_atoms, salt_bridges
from protein_interface.interface import (
    ANION_ATOMS,
    CATION_ATOMS,
    _ic_bins_from_contact_pairs,
    _ic_bins_from_contacts,
)


ROOT = Path(__file__).resolve().parents[1]


def bench(fn, reps: int) -> float:
    for _ in range(2):
        fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def dense_contact_summary(a, b):
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


def spatial_contact_summary(a, b):
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


def python_salt_bridges(a, b):
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


def main() -> None:
    cases = [
        ("nb_ag_test A:L", "tests/data/nb_ag_test.pdb", ["A"], ["L"]),
        ("1fyt D:A", "tests/data/1fyt.pdb", ["D"], ["A"]),
        ("1fyt D:E vs A:B:C", "tests/data/1fyt.pdb", ["D", "E"], ["A", "B", "C"]),
    ]
    print(
        "case,atoms,dense_contact_ms,spatial_contact_ms,contact_speedup,"
        "python_salt_ms,rust_salt_ms,salt_speedup,full_analyze_ms"
    )
    for name, rel, chains_a, chains_b in cases:
        a = load_atoms(ROOT / rel, chains_a)
        b = load_atoms(ROOT / rel, chains_b)
        dense = dense_contact_summary(a, b)
        spatial = spatial_contact_summary(a, b)
        py_salt = python_salt_bridges(a, b)
        rs_salt = salt_bridges(a, b)
        if dense != spatial:
            raise AssertionError(f"contact mismatch for {name}")
        if py_salt != rs_salt:
            raise AssertionError(f"salt mismatch for {name}")
        dense_ms = bench(lambda: dense_contact_summary(a, b), 7)
        spatial_ms = bench(lambda: spatial_contact_summary(a, b), 7)
        py_salt_ms = bench(lambda: python_salt_bridges(a, b), 7)
        rs_salt_ms = bench(lambda: salt_bridges(a, b), 7)
        full_ms = bench(lambda: analyze(a, b), 5)
        print(
            f"{name},{len(a.coords) + len(b.coords)},"
            f"{dense_ms:.3f},{spatial_ms:.3f},{dense_ms / spatial_ms:.2f},"
            f"{py_salt_ms:.3f},{rs_salt_ms:.3f},{py_salt_ms / rs_salt_ms:.2f},"
            f"{full_ms:.3f}"
        )


if __name__ == "__main__":
    main()
