"""Validation of our SASA implementation against FreeSASA on 1ZVH.

We don't expect bit-identical output — sc-rs uses Lee-Richards-style MS radii
(from CCP4's sc Fortran source), FreeSASA defaults to ProtOr radii — so total
SASA values differ by a few percent. Algorithmic correctness shows up as
**high per-atom and per-residue correlation** and a per-residue mean absolute
difference that stays in the few-Å² range.

If these tests start failing it almost certainly means the SASA kernel changed,
not the radii table — radii differences move all atoms together; algorithm
bugs introduce outliers and lower correlation.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from protein_interface import compute_sasa, load_atoms

freesasa = pytest.importorskip("freesasa")

DATA = Path(__file__).parent / "data"
PDB_1ZVH = DATA / "nb_ag_test.pdb"


@pytest.fixture(scope="module")
def comparison():
    """Compute SASA both ways on 1ZVH and join the per-atom arrays by atom key."""
    freesasa.setVerbosity(freesasa.silent)
    params = freesasa.Parameters({
        "algorithm": freesasa.ShrakeRupley,
        "probe-radius": 1.4,
        "n-points": 960,
    })
    fs_struct = freesasa.Structure(str(PDB_1ZVH))
    fs_result = freesasa.calc(fs_struct, params)

    # Build (chain, resnum, atom_name) → FreeSASA SASA map.
    fs_by_key: dict[tuple[str, int, str], float] = {}
    for i in range(fs_struct.nAtoms()):
        ch = fs_struct.chainLabel(i).strip()
        rn = int(fs_struct.residueNumber(i).strip())
        an = fs_struct.atomName(i).strip()
        fs_by_key[(ch, rn, an)] = fs_result.atomArea(i)

    a = load_atoms(PDB_1ZVH, ["A"])
    l = load_atoms(PDB_1ZVH, ["L"])
    coords = a.coords + l.coords
    names = a.atom_names + l.atom_names
    res = a.residue_names + l.residue_names
    rids = a.residue_ids + l.residue_ids
    ours = compute_sasa(coords, names, res, 1.4, 960)

    us_atom: list[float] = []
    them_atom: list[float] = []
    us_res: dict[tuple[str, int], float] = defaultdict(float)
    them_res: dict[tuple[str, int], float] = defaultdict(float)
    for i, rid in enumerate(rids):
        key = (rid[0], rid[1], names[i])
        if key in fs_by_key:
            us_atom.append(ours[i])
            them_atom.append(fs_by_key[key])
            us_res[rid] += ours[i]
            them_res[rid] += fs_by_key[key]
    keys = sorted(us_res.keys())
    return {
        "n_atoms": len(us_atom),
        "us_atom": np.asarray(us_atom),
        "them_atom": np.asarray(them_atom),
        "us_res": np.asarray([us_res[k] for k in keys]),
        "them_res": np.asarray([them_res[k] for k in keys]),
        "fs_total": fs_result.totalArea(),
        "us_total": sum(ours),
    }


def test_freesasa_atom_match_count(comparison):
    """We must successfully match at least 98 % of FreeSASA's atoms by (chain, resnum, name)."""
    assert comparison["n_atoms"] >= 1800  # 1ZVH has ~1870 heavy atoms in chains A+L


def test_freesasa_total_sasa_within_5pct(comparison):
    """Total SASA differs only because of radii. Expect within 5 %."""
    ratio = comparison["us_total"] / comparison["fs_total"]
    assert 0.95 < ratio < 1.05, f"ours/freesasa = {ratio:.3f}"


def test_freesasa_atom_correlation(comparison):
    """Per-atom Pearson r should be ≥ 0.98 — algorithm-quality signal."""
    r = float(np.corrcoef(comparison["us_atom"], comparison["them_atom"])[0, 1])
    assert r >= 0.98, f"atom-level r = {r:.4f}"


def test_freesasa_residue_correlation(comparison):
    """Per-residue Pearson r should be ≥ 0.99 — radii-difference noise averages out."""
    r = float(np.corrcoef(comparison["us_res"], comparison["them_res"])[0, 1])
    assert r >= 0.99, f"residue-level r = {r:.4f}"


def test_freesasa_residue_mean_abs_diff(comparison):
    """Per-residue mean absolute difference should be < 3 Å² (typical residue SASA ~50–200)."""
    mad = float(np.abs(comparison["us_res"] - comparison["them_res"]).mean())
    assert mad < 3.0, f"per-residue mean abs diff = {mad:.2f} Å²"
