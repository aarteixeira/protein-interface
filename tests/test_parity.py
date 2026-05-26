"""Parity tests: pysc.from_pdb vs the sc-rs CLI binary.

These tests verify that pysc.io parses the same atoms as sc-rs parses —
by confirming that atom counts match and results are numerically close.

Why not 1e-6 exact tolerance?
  Both pysc and the sc CLI compile sc-rs independently into separate LLVM code
  units (cdylib vs binary). LLVM's FMA folding and inlining decisions can
  produce floating-point differences of ~1e-6 to ~1e-5 even from the same
  source code. We use 1e-3 as the tolerance: ~100× larger than observed
  differences, still trivially smaller than any physically meaningful SC
  precision (values are typically reported to 3 decimal places).

  The key correctness check is atom_count: if pysc and the CLI accept the
  same atoms, the algorithm is being applied identically.

Skipped automatically when the `sc` binary is not in PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from protein_interface import from_pdb

DATA = Path(__file__).parent / "data"
SC_BINARY = shutil.which("sc")

requires_sc = pytest.mark.skipif(
    SC_BINARY is None,
    reason=(
        "`sc` binary not found in PATH. "
        "Build sc-rs with `cargo install --git https://github.com/cytokineking/sc-rs "
        "--tag v1.0.0` and ensure it is on PATH to run parity tests."
    ),
)

CASES = [
    # (pdb_file, chain_a, chain_b)
    # Single chains only — sc-rs CLI accepts exactly one chain per side.
    # 1FYT: TCR alpha (D) vs MHC alpha (A)
    (DATA / "1fyt.pdb", "D", "A"),
    # 1ZVH: anti-lysozyme nanobody (A) vs lysozyme (L)
    (DATA / "nb_ag_test.pdb", "A", "L"),
]

# Cross-compilation FP tolerance: same Rust source compiled into a cdylib
# (pysc) vs a standalone binary (sc CLI) can give differences up to ~1e-5
# due to LLVM FMA folding and inlining differences. 1e-3 is safe and still
# physically meaningful (SC is typically reported to 2-3 decimal places).
_TOL = 1e-3


def _run_cli(pdb: Path, chain1: str, chain2: str) -> dict:
    """Run `sc <pdb> <chain1> <chain2> --json --no-parallel` and return parsed JSON."""
    result = subprocess.run(
        [SC_BINARY, str(pdb), chain1, chain2, "--json", "--no-parallel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@requires_sc
@pytest.mark.parametrize("pdb,chain_a,chain_b", CASES)
def test_parity_atom_counts(pdb, chain_a, chain_b):
    """Atom counts must match exactly — this confirms identical PDB parsing."""
    cli = _run_cli(pdb, chain_a, chain_b)
    py = from_pdb(pdb, [chain_a], [chain_b], parallel=False)
    assert py.atoms_a == cli["atoms_mol1"], (
        f"atoms_a mismatch: pysc={py.atoms_a}, cli={cli['atoms_mol1']}"
    )
    assert py.atoms_b == cli["atoms_mol2"], (
        f"atoms_b mismatch: pysc={py.atoms_b}, cli={cli['atoms_mol2']}"
    )


@requires_sc
@pytest.mark.parametrize("pdb,chain_a,chain_b", CASES)
def test_parity_sc(pdb, chain_a, chain_b):
    """SC value matches CLI within cross-compilation FP tolerance."""
    cli = _run_cli(pdb, chain_a, chain_b)
    py = from_pdb(pdb, [chain_a], [chain_b], parallel=False)
    assert abs(py.sc - cli["sc"]) < _TOL, (
        f"SC mismatch: pysc={py.sc:.8f}, cli={cli['sc']:.8f}, "
        f"diff={abs(py.sc - cli['sc']):.2e} (tol={_TOL:.0e})"
    )


@requires_sc
@pytest.mark.parametrize("pdb,chain_a,chain_b", CASES)
def test_parity_median_distance(pdb, chain_a, chain_b):
    """Median distance matches CLI within cross-compilation FP tolerance."""
    cli = _run_cli(pdb, chain_a, chain_b)
    py = from_pdb(pdb, [chain_a], [chain_b], parallel=False)
    assert abs(py.median_distance - cli["median_distance"]) < _TOL, (
        f"median_distance mismatch: pysc={py.median_distance:.8f}, "
        f"cli={cli['median_distance']:.8f}, "
        f"diff={abs(py.median_distance - cli['median_distance']):.2e}"
    )


@requires_sc
@pytest.mark.parametrize("pdb,chain_a,chain_b", CASES)
def test_parity_trimmed_area(pdb, chain_a, chain_b):
    """Trimmed area matches CLI within cross-compilation FP tolerance."""
    cli = _run_cli(pdb, chain_a, chain_b)
    py = from_pdb(pdb, [chain_a], [chain_b], parallel=False)
    assert abs(py.trimmed_area - cli["trimmed_area"]) < _TOL, (
        f"trimmed_area mismatch: pysc={py.trimmed_area:.8f}, "
        f"cli={cli['trimmed_area']:.8f}, "
        f"diff={abs(py.trimmed_area - cli['trimmed_area']):.2e}"
    )
