"""Head-to-head SC benchmark against the upstream sc-rs CLI.

The upstream CLI accepts one PDB file and one chain ID per side. To compare the
same atom sets, this script writes normalized temporary PDB files with side A as
chain X and side B as chain Y. The optimized implementation is timed through
the Python `compute_sc_batch()` binding on the same arrays.

Usage:
    python benchmark/sc_cli_head_to_head.py --original-cli /path/to/sc

The optimized implementation and the cached upstream CLI are not bit-for-bit
identical on every case; the default tolerance is an absolute 1e-4 across SC,
median distance, and trimmed area.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser

from protein_interface import compute_sc_batch
from protein_interface.io import _extract_atom_arrays


DATA = Path(__file__).parent.parent / "tests" / "data"


@dataclass(frozen=True)
class Case:
    name: str
    pdb_path: Path
    chains_a: tuple[str, ...]
    chains_b: tuple[str, ...]


@dataclass
class PreparedCase:
    case: Case
    pdb_path: Path
    sc_input: tuple[
        list[list[float]],
        list[str],
        list[str],
        list[list[float]],
        list[str],
        list[str],
    ]


CASES = [
    Case("nb_ag_test A:L", DATA / "nb_ag_test.pdb", ("A",), ("L",)),
    Case("1FYT D:A", DATA / "1fyt.pdb", ("D",), ("A",)),
    Case("1FYT D:E vs A:B:C", DATA / "1fyt.pdb", ("D", "E"), ("A", "B", "C")),
]


BATCHES = [
    ("mixed x1", [0, 1, 2]),
    ("mixed x4", [0, 1, 2] * 4),
    ("mixed x8", [0, 1, 2] * 8),
]


def _element(atom_name: str) -> str:
    for ch in atom_name:
        if ch.isalpha():
            return ch.upper()
    return ""


def _pdb_atom_name(atom_name: str) -> str:
    name = atom_name[:4]
    return f"{name:>4s}"


def _write_normalized_pdb(
    path: Path,
    coords_a: list[list[float]],
    names_a: list[str],
    res_a: list[str],
    coords_b: list[list[float]],
    names_b: list[str],
    res_b: list[str],
) -> None:
    serial = 1
    with path.open("w", encoding="utf-8") as handle:
        for chain, coords, names, residues in (
            ("X", coords_a, names_a, res_a),
            ("Y", coords_b, names_b, res_b),
        ):
            for idx, (coord, atom_name, res_name) in enumerate(zip(coords, names, residues), start=1):
                x, y, z = coord
                handle.write(
                    f"ATOM  {serial:5d} {_pdb_atom_name(atom_name)} "
                    f"{res_name[:3]:>3s} {chain}{idx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                    f"{_element(atom_name):>2s}\n"
                )
                serial += 1
        handle.write("END\n")


def _prepare_cases(workdir: Path) -> list[PreparedCase]:
    parser = PDBParser(QUIET=True)
    prepared = []
    for case in CASES:
        model = list(parser.get_structure(case.name, str(case.pdb_path)).get_models())[0]
        coords_a, names_a, res_a = _extract_atom_arrays(model, list(case.chains_a), False, False)
        coords_b, names_b, res_b = _extract_atom_arrays(model, list(case.chains_b), False, False)
        out = workdir / f"{case.name.replace(' ', '_').replace(':', '_')}.pdb"
        _write_normalized_pdb(out, coords_a, names_a, res_a, coords_b, names_b, res_b)
        prepared.append(PreparedCase(case, out, (coords_a, names_a, res_a, coords_b, names_b, res_b)))
    return prepared


def _run_cli(original_cli: Path, pdb_path: Path) -> tuple[float, dict]:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(original_cli), str(pdb_path), "X", "Y", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - start
    return wall, json.loads(completed.stdout)


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _compare_values(prepared: list[PreparedCase], original_cli: Path, tolerance: float) -> list[dict]:
    cli_results = []
    for item in prepared:
        _, result = _run_cli(original_cli, item.pdb_path)
        cli_results.append(result)
    optimized = compute_sc_batch([item.sc_input for item in prepared], parallel=True)

    rows = []
    for item, cli, opt in zip(prepared, cli_results, optimized):
        deltas = {
            "sc": abs(float(cli["sc"]) - float(opt.sc)),
            "median_distance": abs(float(cli["median_distance"]) - float(opt.median_distance)),
            "trimmed_area": abs(float(cli["trimmed_area"]) - float(opt.trimmed_area)),
        }
        if (
            cli["atoms_mol1"] != opt.atoms_a
            or cli["atoms_mol2"] != opt.atoms_b
            or any(delta > tolerance for delta in deltas.values())
        ):
            raise AssertionError(
                f"{item.case.name}: original CLI and optimized batch values differ: "
                f"cli={cli}, opt={{'sc': {opt.sc}, 'median_distance': {opt.median_distance}, "
                f"'trimmed_area': {opt.trimmed_area}, 'atoms_a': {opt.atoms_a}, 'atoms_b': {opt.atoms_b}}}, "
                f"deltas={deltas}"
            )
        rows.append({
            "case": item.case.name,
            "atoms": f"{opt.atoms_a}+{opt.atoms_b}",
            "sc": opt.sc,
            "median_distance": opt.median_distance,
            "trimmed_area": opt.trimmed_area,
            **{f"delta_{k}": v for k, v in deltas.items()},
        })
    return rows


def _benchmark_batch(
    prepared: list[PreparedCase],
    original_cli: Path,
    batch_indices: list[int],
    reps: int,
) -> dict:
    batch = [prepared[i] for i in batch_indices]

    def run_cli_batch() -> tuple[float, float]:
        wall_total = 0.0
        calc_total = 0.0
        for item in batch:
            wall, result = _run_cli(original_cli, item.pdb_path)
            wall_total += wall
            calc_total += float(result["elapsed_ms"]) / 1000.0
        return wall_total, calc_total

    def run_optimized_batch() -> float:
        start = time.perf_counter()
        compute_sc_batch([item.sc_input for item in batch], parallel=True)
        return time.perf_counter() - start

    run_cli_batch()
    run_optimized_batch()

    cli_wall = []
    cli_calc = []
    optimized_wall = []
    for _ in range(reps):
        wall, calc = run_cli_batch()
        cli_wall.append(wall)
        cli_calc.append(calc)
        optimized_wall.append(run_optimized_batch())

    cli_wall_median = _median(cli_wall)
    cli_calc_median = _median(cli_calc)
    optimized_median = _median(optimized_wall)
    return {
        "batch": "unset",
        "n": len(batch),
        "original_cli_wall_s": cli_wall_median,
        "original_cli_calc_s": cli_calc_median,
        "optimized_batch_s": optimized_median,
        "speedup_vs_cli_wall": cli_wall_median / optimized_median,
        "speedup_vs_cli_calc": cli_calc_median / optimized_median,
    }


def _print_accuracy(rows: list[dict]) -> None:
    print("\nAccuracy: original sc-rs CLI vs optimized compute_sc_batch")
    print("| Case | Atoms | SC | Median distance | Trimmed area | max delta |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        max_delta = max(row["delta_sc"], row["delta_median_distance"], row["delta_trimmed_area"])
        print(
            f"| {row['case']} | {row['atoms']} | {row['sc']:.12f} | "
            f"{row['median_distance']:.12f} | {row['trimmed_area']:.6f} | {max_delta:.1e} |"
        )


def _print_performance(rows: list[dict]) -> None:
    print("\nPerformance")
    print(
        "| Batch | Complexes | Original CLI wall | Original CLI calc | "
        "Optimized Python batch | Speedup vs CLI wall | Speedup vs CLI calc |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['batch']} | {row['n']} | {row['original_cli_wall_s']:.3f} s | "
            f"{row['original_cli_calc_s']:.3f} s | {row['optimized_batch_s']:.3f} s | "
            f"{row['speedup_vs_cli_wall']:.2f}x | {row['speedup_vs_cli_calc']:.2f}x |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cli", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()

    if not args.original_cli.exists():
        raise SystemExit(f"original CLI not found: {args.original_cli}")

    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="sc-head-to-head-") as tmp:
            _run(Path(tmp), args.original_cli, args.reps, args.tolerance)
    else:
        args.workdir.mkdir(parents=True, exist_ok=True)
        _run(args.workdir, args.original_cli, args.reps, args.tolerance)


def _run(workdir: Path, original_cli: Path, reps: int, tolerance: float) -> None:
    prepared = _prepare_cases(workdir)
    accuracy_rows = _compare_values(prepared, original_cli, tolerance)
    performance_rows = []
    for name, indices in BATCHES:
        row = _benchmark_batch(prepared, original_cli, indices, reps)
        row["batch"] = name
        performance_rows.append(row)
    _print_accuracy(accuracy_rows)
    _print_performance(performance_rows)


if __name__ == "__main__":
    main()
