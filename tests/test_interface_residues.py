"""Per-residue interface classification on the 1ITB IL-1beta / IL-1R complex.

Tests use a low n_points for speed; the classification logic and the invariants
asserted here do not depend on SASA precision.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from protein_interface import CATEGORIES, classify_residues
from protein_interface.interface import BACKBONE_ATOMS
from protein_interface.residue_classifier import ResidueClassification

DATA = Path(__file__).parent / "data"
PDB = DATA / "1itb.pdb"

pytest.importorskip("scipy")

N_POINTS = 200  # fast; invariants are independent of SASA precision


@pytest.fixture(scope="module")
def result() -> ResidueClassification:
    return classify_residues(PDB, n_points=N_POINTS)  # default: strict mode, near off


@pytest.fixture(scope="module")
def result_near() -> ResidueClassification:
    return classify_residues(PDB, n_points=N_POINTS, near_interface=True)


def test_two_chains_and_residue_count(result):
    assert set(result.params["groups"]) == {"A", "B"}
    # IL-1beta (~153 res) + receptor ectodomain (~310 res)
    assert 450 <= len(result.records) <= 470


def test_every_residue_has_exactly_one_valid_category(result):
    cats = [r.category for r in result.records]
    assert set(cats) <= set(CATEGORIES)
    counts = result.counts()["ALL"]
    assert sum(counts[c] for c in CATEGORIES) == len(result.records)


def test_interface_matches_dsasa_or_contact_definition(result):
    assert result.params["combine"] == "or"  # default is the union
    thr = result.params["dsasa_threshold"]
    cc = result.params["contact_cutoff"]
    recompute = {
        (r.chain, r.resseq, r.icode)
        for r in result.records
        if r.dsasa > thr or r.min_interchain_dist <= cc
    }
    labelled = {
        (r.chain, r.resseq, r.icode) for r in result.records if r.category == "interface"
    }
    assert recompute == labelled
    assert len(labelled) > 0


def test_combine_and_is_intersection_and_subset_of_or(result):
    thr = result.params["dsasa_threshold"]
    cc = result.params["contact_cutoff"]
    both = classify_residues(PDB, n_points=N_POINTS, combine="and")
    and_set = {
        (r.chain, r.resseq, r.icode) for r in both.records if r.category == "interface"
    }
    recompute = {
        (r.chain, r.resseq, r.icode)
        for r in both.records
        if r.dsasa > thr and r.min_interchain_dist <= cc
    }
    or_set = {
        (r.chain, r.resseq, r.icode) for r in result.records if r.category == "interface"
    }
    assert and_set == recompute
    assert and_set < or_set  # strict subset: PRO131-style rim residues drop out


def test_invalid_combine_raises():
    with pytest.raises(ValueError, match="combine must be"):
        classify_residues(PDB, n_points=N_POINTS, combine="xor")


def test_near_interface_are_noninterface(result_near):
    near = [r for r in result_near.records if r.category == "near_interface"]
    assert near, "expected some near-interface residues on 1ITB"
    assert all(r.category != "interface" for r in near)


def test_core_are_buried_and_not_interface(result):
    core = [r for r in result.records if r.category == "core"]
    assert core
    cut = result.params["core_rsasa"]
    for r in core:
        assert r.monomer_rsasa == r.monomer_rsasa  # not NaN
        assert r.monomer_rsasa < cut


def test_near_interface_within_euclidean_bound(result_near):
    """Graph geodesic >= straight-line distance, so a near-interface residue
    (geodesic <= near_cutoff) must have a side-chain heavy atom within
    near_cutoff Euclidean of an interface residue's side chain in the same chain.
    This independently validates the geodesic engine's output."""
    near_cut = result_near.params["near_cutoff"]
    atoms = _load_sidechain_atoms(PDB)
    iface_by_chain: dict[str, list] = {}
    for r in result_near.records:
        if r.category == "interface":
            iface_by_chain.setdefault(r.chain, []).extend(atoms[(r.chain, r.resseq, r.icode)])

    for r in result_near.records:
        if r.category != "near_interface":
            continue
        my = np.asarray(atoms[(r.chain, r.resseq, r.icode)])
        others = np.asarray(iface_by_chain[r.chain])
        d = np.linalg.norm(my[:, None, :] - others[None, :, :], axis=-1).min()
        assert d <= near_cut + 1e-6, f"{r.chain}{r.resseq} euclidean {d:.2f} > {near_cut}"


def test_geodesic_column_semantics(result_near):
    near_cut = result_near.params["near_cutoff"]
    for r in result_near.records:
        if r.category == "interface":
            assert r.geodesic_to_interface == 0.0  # interface residues are the sources
        elif r.category == "near_interface":
            assert 0.0 < r.geodesic_to_interface <= near_cut + 1e-6
        # core / non_interface: geodesic > near_cut or NaN (chain has no interface)


def test_default_is_strict_mode_near_off(result):
    assert result.params["mode"] == "strict"
    assert result.params["near_interface"] is False
    assert (result.params["dsasa_threshold"], result.params["contact_cutoff"]) == (3.0, 5.0)
    assert all(r.category != "near_interface" for r in result.records)


def test_no_near_interface_disables_category():
    res = classify_residues(PDB, n_points=N_POINTS, near_interface=False)
    assert res.params["near_interface"] is False
    assert all(r.category != "near_interface" for r in res.records)
    assert all(math.isnan(r.geodesic_to_interface) for r in res.records)


def test_mode_presets_resolve_thresholds():
    s = classify_residues(PDB, n_points=N_POINTS, mode="strict")
    le = classify_residues(PDB, n_points=N_POINTS, mode="lenient")
    assert (s.params["dsasa_threshold"], s.params["contact_cutoff"], s.params["combine"]) == (3.0, 5.0, "or")
    assert (le.params["dsasa_threshold"], le.params["contact_cutoff"], le.params["combine"]) == (0.0, 7.0, "or")
    n_strict = sum(r.category == "interface" for r in s.records)
    n_lenient = sum(r.category == "interface" for r in le.records)
    assert n_lenient > n_strict  # looser thresholds capture more


def test_explicit_args_override_mode():
    r = classify_residues(PDB, n_points=N_POINTS, mode="lenient", dsasa_threshold=3.0, contact_cutoff=5.0)
    assert r.params["dsasa_threshold"] == 3.0
    assert r.params["contact_cutoff"] == 5.0
    assert r.params["combine"] == "or"  # not overridden -> from lenient preset


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        classify_residues(PDB, n_points=N_POINTS, mode="medium")


def test_known_interface_residues(result):
    """Sanity vs biology: top buried IL-1beta residues are interface (strict mode)."""
    cat = {(r.chain, r.resseq): r.category for r in result.records}
    # Arg4 and Gln32 of IL-1beta bury >100 A^2 against the receptor.
    assert cat[("A", 4)] == "interface"
    assert cat[("A", 32)] == "interface"
    # Pro131 buries ~20 A^2 (> 3) so it is interface via the dSASA arm of the OR.
    assert cat[("A", 131)] == "interface"


def test_higher_dsasa_threshold_shrinks_interface(result):
    strict = classify_residues(PDB, n_points=N_POINTS, dsasa_threshold=40.0)
    n_default = sum(1 for r in result.records if r.category == "interface")
    n_strict = sum(1 for r in strict.records if r.category == "interface")
    assert n_strict < n_default


def test_groups_explicit_equals_default(result):
    grouped = classify_residues(PDB, n_points=N_POINTS, groups=[["A"], ["B"]])
    a = {(r.chain, r.resseq, r.icode): r.category for r in result.records}
    b = {(r.chain, r.resseq, r.icode): r.category for r in grouped.records}
    assert a == b


def test_unassigned_chain_raises():
    with pytest.raises(ValueError, match="not assigned to any group"):
        classify_residues(PDB, n_points=N_POINTS, groups=[["A"]])


def test_duplicate_chain_in_groups_raises():
    with pytest.raises(ValueError, match="more than one group"):
        classify_residues(PDB, n_points=N_POINTS, groups=[["A"], ["A", "B"]])


def test_chains_subset_restricts_loading():
    res = classify_residues(PDB, n_points=N_POINTS, chains=["A"])
    assert {r.chain for r in res.records} == {"A"}
    assert res.params["chains"] == ["A"]
    # one chain loaded -> no partner -> nothing can be interface
    assert all(r.category != "interface" for r in res.records)


def test_invalid_chain_raises():
    with pytest.raises(ValueError, match="not in structure"):
        classify_residues(PDB, n_points=N_POINTS, chains=["Z"])


def test_excel_output(result, tmp_path):
    pd = pytest.importorskip("pandas")
    out = result.to_excel(tmp_path / "out.xlsx")
    assert out.exists()
    df = pd.read_excel(out, sheet_name="residues")
    assert list(df.columns) == [
        "group", "chain", "resseq", "icode", "resname", "category", "dsasa",
        "min_interchain_dist", "geodesic_to_interface", "monomer_rsasa", "complex_rsasa",
    ]
    assert len(df) == len(result.records)
    summary = pd.read_excel(out, sheet_name="summary")
    assert "interface" in summary.columns


def test_html_output(result, tmp_path):
    out = result.to_html(tmp_path / "out.html")
    html = out.read_text()
    assert "3Dmol" in html
    assert "ATOM" in html  # embedded structure
    for color in ("d62728", "ff7f0e", "1f77b4"):  # interface / near / core
        assert color in html


# ── helper ────────────────────────────────────────────────────────────────────

def _load_sidechain_atoms(pdb: Path) -> dict[tuple, list[list[float]]]:
    """Side-chain heavy-atom coords per residue id, using the exact same atoms
    and side-chain definition as the classifier (CA fallback for Gly)."""
    from protein_interface import load_atoms
    from protein_interface.io import _load_structure

    model = list(_load_structure(pdb).get_models())[0]
    chains = [ch.id for ch in model.get_chains()]
    arr = load_atoms(pdb, chains=chains)

    out: dict[tuple, list[list[float]]] = {}
    ca: dict[tuple, list[float]] = {}
    for i, rid in enumerate(arr.residue_ids):
        name = arr.atom_names[i]
        if name == "CA":
            ca[rid] = arr.coords[i]
        if name not in BACKBONE_ATOMS:
            out.setdefault(rid, []).append(arr.coords[i])
    for rid, c in ca.items():
        out.setdefault(rid, [c])  # Gly / backbone-only: fall back to CA
    return out
