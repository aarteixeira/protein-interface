"""Tests for the PRODIGY ΔG_bind predictor.

Unit tests cover the IC binning + classification tables. Integration tests on
1ZVH (nanobody:lysozyme) verify the value is in the expected range for a real
binder. An optional validation test (`pytest.importorskip("prodigy_prot")`)
compares against the upstream reference implementation.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from protein_interface import (
    AtomArrays,
    PRODIGY_IC_CLASS,
    PRODIGY_NIS_CLASS,
    REFERENCE_SASA,
    analyze,
    load_atoms,
    prodigy,
    prodigy_ics,
    prodigy_nis,
)

DATA = Path(__file__).parent / "data"
PDB_1ZVH = DATA / "nb_ag_test.pdb"


# ── Classification table integrity ───────────────────────────────────────────

def test_prodigy_ic_classes_cover_20_standard_aa():
    assert set(PRODIGY_IC_CLASS) == {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }


def test_prodigy_ic_vs_nis_aromatic_difference():
    """TYR / TRP / CYS / HIS are deliberately classified differently across the two tables."""
    assert PRODIGY_IC_CLASS["TYR"] == "A" and PRODIGY_NIS_CLASS["TYR"] == "P"
    assert PRODIGY_IC_CLASS["TRP"] == "A" and PRODIGY_NIS_CLASS["TRP"] == "P"
    assert PRODIGY_IC_CLASS["CYS"] == "A" and PRODIGY_NIS_CLASS["CYS"] == "P"
    assert PRODIGY_IC_CLASS["HIS"] == "C" and PRODIGY_NIS_CLASS["HIS"] == "P"


def test_reference_sasa_covers_20_standard_aa():
    assert set(REFERENCE_SASA) == set(PRODIGY_IC_CLASS)


# ── IC binning unit tests ────────────────────────────────────────────────────

def test_prodigy_ics_single_charged_charged_pair():
    a = AtomArrays([[0.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    bins = prodigy_ics(a, b)
    assert bins["CC"] == 1
    assert bins["total"] == 1


def test_prodigy_ics_residue_counted_once():
    """Two atoms of the same residue contacting one B atom → still 1 contact."""
    a = AtomArrays(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ["OD1", "OD2"], ["ASP", "ASP"],
        [("A", 1), ("A", 1)],
    )
    b = AtomArrays([[3.0, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    bins = prodigy_ics(a, b)
    assert bins["CC"] == 1


def test_prodigy_ics_beyond_cutoff():
    a = AtomArrays([[0.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("A", 1)])
    b = AtomArrays([[10.0, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    bins = prodigy_ics(a, b)
    assert bins["total"] == 0


# ── 1ZVH integration ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nb_ag():
    a = load_atoms(PDB_1ZVH, ["A"])
    b = load_atoms(PDB_1ZVH, ["L"])
    return a, b


def test_prodigy_1zvh_dg_in_binder_range(nb_ag):
    """1ZVH is a real high-affinity Nb-lysozyme complex; predicted ΔG should be < -5 kcal/mol."""
    a, b = nb_ag
    dg = prodigy(a, b)["dg"]
    assert -20.0 < dg < -5.0, f"1ZVH ΔG = {dg:+.2f}"


def test_prodigy_dg_matches_analyze_field(nb_ag):
    a, b = nb_ag
    r = analyze(a, b)
    standalone = prodigy(a, b)["dg"]
    # Both routes use the same coefficients and the same atom data, so they
    # should be bit-identical (modulo floating-point reordering across the
    # combined-vs-side-by-side SASA pass — which is the same in both paths).
    assert r.prodigy_dg == pytest.approx(standalone, abs=1e-6)


def test_prodigy_nis_fractions_sum_to_100(nb_ag):
    a, b = nb_ag
    nis = prodigy_nis(a, b)
    assert nis["apolar"] + nis["polar"] + nis["charged"] == pytest.approx(100.0, abs=0.01)


def test_prodigy_dg_independent_of_chain_order(nb_ag):
    """ΔG is a symmetric prediction — swapping A and B must give the same number."""
    a, b = nb_ag
    forward = prodigy(a, b)["dg"]
    reverse = prodigy(b, a)["dg"]
    assert forward == pytest.approx(reverse, abs=1e-6)


# ── Validation against upstream prodigy-prot ────────────────────────────────

prodigy_prot = pytest.importorskip("prodigy_prot")


@pytest.fixture(scope="module")
def upstream_prodigy():
    from prodigy_prot.modules.prodigy import Prodigy
    from prodigy_prot.modules.parsers import parse_structure
    models, _, _ = parse_structure(str(PDB_1ZVH))
    p = Prodigy(models[0], selection=["A", "L"])
    p.predict()
    return {
        "dg": float(p.ba_val),
        "bins": {k: int(v) for k, v in p.bins.items()},
        "nis_apolar": float(p.nis_a),
        "nis_charged": float(p.nis_c),
    }


def test_prodigy_dg_matches_upstream(nb_ag, upstream_prodigy):
    """Our ΔG should agree with upstream prodigy-prot within 2 kcal/mol.

    PRODIGY's own published RMSE is ~1.9 kcal/mol, so 2 kcal/mol is a tight
    bound dominated by altloc / HETATM handling differences in the parsers.
    """
    a, b = nb_ag
    ours = prodigy(a, b)["dg"]
    diff = abs(ours - upstream_prodigy["dg"])
    assert diff < 2.0, f"|Δ| = {diff:.2f} kcal/mol (ours {ours:+.2f}, upstream {upstream_prodigy['dg']:+.2f})"


def test_prodigy_ic_bins_match_upstream(nb_ag, upstream_prodigy):
    """IC bins should match upstream within ±3 per bin (small differences from atom selection)."""
    a, b = nb_ag
    ours = prodigy_ics(a, b)
    for key in ("CC", "AC", "AP", "AA", "PP", "CP"):
        diff = abs(ours[key] - upstream_prodigy["bins"][key])
        assert diff <= 3, f"{key}: ours={ours[key]}, upstream={upstream_prodigy['bins'][key]}"


def test_prodigy_nis_matches_upstream(nb_ag, upstream_prodigy):
    """NIS percentages should agree within 5 percentage points."""
    a, b = nb_ag
    ours = prodigy_nis(a, b)
    assert abs(ours["apolar"] - upstream_prodigy["nis_apolar"]) < 5.0
    assert abs(ours["charged"] - upstream_prodigy["nis_charged"]) < 5.0
