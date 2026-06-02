"""Benchmark protein_interface SASA against FreeSASA.

The timing compares calculation only on prebuilt atom structures. Values are
not expected to be identical because protein_interface uses the SC/MS-style
radii table, while FreeSASA uses its default classifier/radii.

Usage:
    python benchmark/sasa_vs_freesasa.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import time

import freesasa
import numpy as np

from protein_interface import AtomArrays, compute_sasa, load_atoms


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


def make_freesasa_structure(atoms: AtomArrays) -> freesasa.Structure:
    structure = freesasa.Structure()
    for coord, atom_name, residue_name, residue_id in zip(
        atoms.coords, atoms.atom_names, atoms.residue_names, atoms.residue_ids
    ):
        chain_id, resseq, _icode = residue_id
        structure.addAtom(
            atom_name,
            residue_name,
            str(resseq),
            chain_id,
            float(coord[0]),
            float(coord[1]),
            float(coord[2]),
        )
    return structure


def bench(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def correlation(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=960)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    freesasa.setVerbosity(freesasa.silent)
    params = freesasa.Parameters({
        "algorithm": freesasa.ShrakeRupley,
        "probe-radius": 1.4,
        "n-points": args.n_points,
    })

    print(
        f"SASA vs FreeSASA, n_points={args.n_points}, "
        f"median of {args.reps} timed runs"
    )
    print()
    print(
        "| Case | Atoms | Ours | FreeSASA | Speedup vs FreeSASA | "
        "Total ratio | Atom r | Residue r | Residue MAD |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for label, path, chains_a, chains_b in CASES:
        a = load_atoms(path, chains_a)
        b = load_atoms(path, chains_b)
        atoms = combine(a, b)
        fs_structure = make_freesasa_structure(atoms)

        ours = compute_sasa(
            atoms.coords,
            atoms.atom_names,
            atoms.residue_names,
            1.4,
            args.n_points,
            True,
        )
        fs_result = freesasa.calc(fs_structure, params)
        them = [fs_result.atomArea(i) for i in range(fs_structure.nAtoms())]

        ours_res: dict[tuple[str, int, str], float] = defaultdict(float)
        them_res: dict[tuple[str, int, str], float] = defaultdict(float)
        for i, residue_id in enumerate(atoms.residue_ids):
            key = (residue_id[0], residue_id[1], residue_id[2])
            ours_res[key] += ours[i]
            them_res[key] += them[i]

        residue_keys = sorted(ours_res)
        ours_res_values = [ours_res[k] for k in residue_keys]
        them_res_values = [them_res[k] for k in residue_keys]
        residue_mad = float(
            np.abs(np.asarray(ours_res_values) - np.asarray(them_res_values)).mean()
        )

        ours_time = bench(
            lambda atoms=atoms: compute_sasa(
                atoms.coords,
                atoms.atom_names,
                atoms.residue_names,
                1.4,
                args.n_points,
                True,
            ),
            args.reps,
        )
        fs_time = bench(
            lambda fs_structure=fs_structure: freesasa.calc(fs_structure, params),
            args.reps,
        )

        print(
            f"| {label} | {len(atoms.coords)} | {ours_time * 1000:.1f} ms | "
            f"{fs_time * 1000:.1f} ms | {fs_time / ours_time:.2f}x | "
            f"{sum(ours) / fs_result.totalArea():.3f} | "
            f"{correlation(ours, them):.4f} | "
            f"{correlation(ours_res_values, them_res_values):.4f} | "
            f"{residue_mad:.2f} Å² |"
        )


if __name__ == "__main__":
    main()
