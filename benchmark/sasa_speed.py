"""Benchmark optimized SASA against the serial compatibility path.

The comparison uses the same atom arrays for both paths and requires exact
per-atom SASA equality before reporting timings.

Usage:
    python benchmark/sasa_speed.py
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from protein_interface import AtomArrays, compute_sasa, compute_sasa_batch, load_atoms


DATA = Path(__file__).parent.parent / "tests" / "data"


CASES = [
    ("nb_ag_test A:L", DATA / "nb_ag_test.pdb", ["A"], ["L"]),
    ("1FYT D:A", DATA / "1fyt.pdb", ["D"], ["A"]),
    ("1FYT D:E vs A:B:C", DATA / "1fyt.pdb", ["D", "E"], ["A", "B", "C"]),
]


def combine(a: AtomArrays, b: AtomArrays) -> AtomArrays:
    return AtomArrays(
        a.coords + b.coords,
        a.atom_names + b.atom_names,
        a.residue_names + b.residue_names,
        a.residue_ids + b.residue_ids,
    )


def bench(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def load_cases() -> list[tuple[str, AtomArrays, AtomArrays, AtomArrays]]:
    loaded = []
    for label, path, chains_a, chains_b in CASES:
        a = load_atoms(path, chains_a)
        b = load_atoms(path, chains_b)
        loaded.append((label, a, b, combine(a, b)))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=92)
    parser.add_argument("--reps", type=int, default=9)
    args = parser.parse_args()

    cases = load_cases()

    print(f"SASA benchmark, n_points={args.n_points}, median of {args.reps} timed runs")
    print()
    print("| Case | Atoms | Serial | Optimized | Speedup | Max per-atom delta |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, _a, _b, combined in cases:
        serial = compute_sasa(
            combined.coords,
            combined.atom_names,
            combined.residue_names,
            1.4,
            args.n_points,
            False,
        )
        optimized = compute_sasa(
            combined.coords,
            combined.atom_names,
            combined.residue_names,
            1.4,
            args.n_points,
            True,
        )
        max_delta = max((abs(a - b) for a, b in zip(serial, optimized)), default=0.0)
        if max_delta != 0.0:
            raise AssertionError(f"{label}: max per-atom delta {max_delta}")
        serial_time = bench(
            lambda c=combined: compute_sasa(
                c.coords, c.atom_names, c.residue_names, 1.4, args.n_points, False
            ),
            args.reps,
        )
        optimized_time = bench(
            lambda c=combined: compute_sasa(
                c.coords, c.atom_names, c.residue_names, 1.4, args.n_points, True
            ),
            args.reps,
        )
        print(
            f"| {label} | {len(combined.coords)} | {serial_time * 1000:.1f} ms | "
            f"{optimized_time * 1000:.1f} ms | {serial_time / optimized_time:.2f}x | "
            f"{max_delta:.1e} |"
        )

    print()
    print("| Batch | Structures | Serial | Optimized | Speedup | Max per-atom delta |")
    print("|---|---:|---:|---:|---:|---:|")
    batch_inputs = []
    for _label, a, b, combined in cases:
        batch_inputs.extend([
            (a.coords, a.atom_names, a.residue_names),
            (b.coords, b.atom_names, b.residue_names),
            (combined.coords, combined.atom_names, combined.residue_names),
        ])

    serial_batch = compute_sasa_batch(batch_inputs, 1.4, args.n_points, False)
    optimized_batch = compute_sasa_batch(batch_inputs, 1.4, args.n_points, True)
    max_delta = max(
        (
            abs(a - b)
            for serial_values, optimized_values in zip(serial_batch, optimized_batch)
            for a, b in zip(serial_values, optimized_values)
        ),
        default=0.0,
    )
    if max_delta != 0.0:
        raise AssertionError(f"mixed batch: max per-atom delta {max_delta}")
    serial_time = bench(
        lambda: compute_sasa_batch(batch_inputs, 1.4, args.n_points, False), args.reps
    )
    optimized_time = bench(
        lambda: compute_sasa_batch(batch_inputs, 1.4, args.n_points, True), args.reps
    )
    print(
        f"| mixed x1 | {len(batch_inputs)} | {serial_time * 1000:.1f} ms | "
        f"{optimized_time * 1000:.1f} ms | {serial_time / optimized_time:.2f}x | "
        f"{max_delta:.1e} |"
    )


if __name__ == "__main__":
    main()
