"""Tests for SASA, H-bond counting, and the interface analysis wrapper.

Tolerances are loose: these geometric metrics are sensitive to radii choices
and atom-set conventions. The goal is to verify the kernels work and produce
values in physically reasonable ranges for a known nanobody-antigen complex
(1ZVH, chains A=nanobody, L=lysozyme).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from protein_interface import (
    analyze,
    analyze_batch,
    compute_sasa,
    compute_sasa_batch,
    count_hbonds,
    count_salt_bridge_atom_pairs,
    delta_sasa,
    hbonds,
    interface_residues,
    load_atoms,
    n_interface_residues,
    sasa,
)

DATA = Path(__file__).parent / "data"
PDB_1ZVH = DATA / "nb_ag_test.pdb"


def test_compute_sasa_isolated_atom_matches_sphere_area():
    """A single isolated carbon-beta with no neighbours has SASA = 4π(r+probe)²."""
    # CB of ALA has radius 1.95 Å in sc-rs's table.
    r = 1.95
    probe = 1.4
    expected = 4.0 * math.pi * (r + probe) ** 2
    s = compute_sasa([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], probe, 960)
    assert len(s) == 1
    assert s[0] == pytest.approx(expected, rel=0.01)


def test_compute_sasa_unknown_atom_returns_zero():
    """Atoms with no radius entry should get SASA = 0.0 (not crash)."""
    s = compute_sasa([[0.0, 0.0, 0.0]], ["XX"], ["ZZZ"], 1.4, 92)
    assert s == [0.0]


def test_compute_sasa_validates_array_lengths():
    with pytest.raises(ValueError):
        compute_sasa([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["CB"], ["ALA"], 1.4, 92)


@pytest.mark.parametrize("probe_radius", [-1.0, math.nan])
def test_compute_sasa_rejects_invalid_probe_radius(probe_radius):
    with pytest.raises(ValueError, match="probe_radius"):
        compute_sasa([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], probe_radius, 92)


@pytest.mark.parametrize("probe_radius", [-1.0, math.nan])
def test_compute_sasa_batch_rejects_invalid_probe_radius(probe_radius):
    with pytest.raises(ValueError, match="probe_radius"):
        compute_sasa_batch([([[0.0, 0.0, 0.0]], ["CB"], ["ALA"])], probe_radius, 92, True)


def test_compute_sasa_two_touching_atoms_have_less_than_isolated():
    """Two overlapping atoms must each expose less surface than in isolation."""
    r = 1.95
    probe = 1.4
    isolated = 4.0 * math.pi * (r + probe) ** 2
    s = compute_sasa(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ["CB", "CB"],
        ["ALA", "ALA"],
        probe,
        960,
    )
    assert s[0] < isolated
    assert s[1] < isolated
    assert s[0] > 0
    assert s[1] > 0


def test_count_hbonds_simple_pair():
    """Backbone N (donor) and backbone O (acceptor) within 3.0 Å → 1 H-bond."""
    n = count_hbonds(
        [[0.0, 0.0, 0.0]], ["N"], ["ALA"],
        [[2.9, 0.0, 0.0]], ["O"], ["ALA"],
        3.5,
    )
    assert n == 1


def test_count_hbonds_pro_n_is_not_donor():
    """Proline backbone N has no H — should not be counted as donor."""
    n = count_hbonds(
        [[0.0, 0.0, 0.0]], ["N"], ["PRO"],
        [[2.9, 0.0, 0.0]], ["O"], ["ALA"],
        3.5,
    )
    assert n == 0


def test_count_hbonds_beyond_cutoff():
    n = count_hbonds(
        [[0.0, 0.0, 0.0]], ["N"], ["ALA"],
        [[5.0, 0.0, 0.0]], ["O"], ["ALA"],
        3.5,
    )
    assert n == 0


@pytest.mark.parametrize("cutoff", [-1.0, math.nan])
def test_count_hbonds_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        count_hbonds(
            [[0.0, 0.0, 0.0]], ["N"], ["ALA"],
            [[2.9, 0.0, 0.0]], ["O"], ["ALA"],
            cutoff,
        )


@pytest.mark.parametrize("cutoff", [-1.0, math.nan])
def test_count_salt_bridge_atom_pairs_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        count_salt_bridge_atom_pairs(
            [[0.0, 0.0, 0.0]], ["OD1"], ["ASP"],
            [[3.0, 0.0, 0.0]], ["NZ"], ["LYS"],
            cutoff,
        )


# ── Integration tests on 1ZVH (nanobody-lysozyme) ────────────────────────────

@pytest.fixture(scope="module")
def nb_ag():
    a = load_atoms(PDB_1ZVH, chains=["A"])
    b = load_atoms(PDB_1ZVH, chains=["L"])
    return a, b


def test_load_atoms_returns_parallel_arrays(nb_ag):
    a, _ = nb_ag
    n = len(a.coords)
    assert n > 500  # nanobody has ~900 heavy atoms
    assert len(a.atom_names) == n
    assert len(a.residue_names) == n
    assert len(a.residue_ids) == n
    # All atoms from chain A should have chain_id 'A'
    assert all(rid[0] == "A" for rid in a.residue_ids)


def test_delta_sasa_in_protein_interface_range(nb_ag):
    """nb-antigen interfaces typically bury 1200-2500 Å² total surface."""
    a, b = nb_ag
    d = delta_sasa(a, b, n_points=92)  # 92 points is fast and good enough for sanity
    assert 800 < d < 3500, f"dSASA = {d:.1f} outside expected range"


def test_n_interface_residues_reasonable(nb_ag):
    a, b = nb_ag
    n_a, n_b = n_interface_residues(a, b, cutoff=5.0)
    # Nanobody paratopes typically have 10-25 interface residues per side.
    assert 8 <= n_a <= 30, f"interface residues on A = {n_a}"
    assert 8 <= n_b <= 30, f"interface residues on B = {n_b}"


def test_interface_residues_are_subset_of_input(nb_ag):
    a, b = nb_ag
    int_a, int_b = interface_residues(a, b, cutoff=5.0)
    ids_a = set(a.residue_ids)
    ids_b = set(b.residue_ids)
    assert int_a.issubset(ids_a)
    assert int_b.issubset(ids_b)


def test_hbonds_count_reasonable(nb_ag):
    a, b = nb_ag
    n = hbonds(a, b)
    # Nb-antigen interfaces typically have 5-20 H-bonds.
    assert 2 <= n <= 30, f"H-bond count = {n}"


def test_analyze_combines_metrics(nb_ag):
    a, b = nb_ag
    res = analyze(a, b, n_points=92)
    assert res.dsasa > 800
    assert res.n_interface_a > 0
    assert res.n_interface_b > 0
    assert 0.0 <= res.aromatic_dsasa_fraction <= 1.0
    assert res.hbonds >= 0


def test_analyze_skip_metrics_returns_none_and_avoids_metric(monkeypatch, nb_ag):
    import protein_interface.interface as iface

    def boom(*args, **kwargs):
        raise AssertionError("hbonds should not run")

    monkeypatch.setattr(iface, "hbonds", boom)
    a, b = nb_ag
    res = analyze(a, b, n_points=92, skip_metrics={"hbonds", "hbond_density"})
    assert res.hbonds is None
    assert res.hbond_density is None
    assert res.dsasa > 800


def test_analyze_metrics_subset_skips_sasa_validation_and_compute(monkeypatch):
    from protein_interface import AtomArrays
    import protein_interface.interface as iface

    def boom(*args, **kwargs):
        raise AssertionError("SASA should not run")

    monkeypatch.setattr(iface, "compute_sasa", boom)
    a = AtomArrays([[0.0, 0.0, 0.0]], ["XX"], ["ZZZ"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    res = analyze(a, b, metrics={"hbonds"}, strict=True)
    assert res.hbonds == 0
    assert res.dsasa is None
    assert res.sc is None


def test_analyze_rejects_unknown_metric(nb_ag):
    a, b = nb_ag
    with pytest.raises(ValueError, match="unknown analyze metric"):
        analyze(a, b, skip_metrics={"not_a_metric"})


def test_analyze_rejects_empty_atom_group():
    from protein_interface import AtomArrays
    empty = AtomArrays([], [], [], [])
    one = AtomArrays([[0.0, 0.0, 0.0]], ["CA"], ["ALA"], [("A", 1)])
    with pytest.raises(ValueError, match="at least one atom"):
        analyze(empty, one, include_sc=False)


def test_analyze_strict_rejects_unknown_sasa_radius():
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["XX"], ["ZZZ"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="no SASA radius"):
        analyze(a, b, include_sc=False)


def test_analyze_permissive_allows_unknown_sasa_radius():
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["XX"], ["ZZZ"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    res = analyze(a, b, include_sc=False, strict=False)
    assert res.dsasa >= 0.0


def test_analyze_strict_propagates_sc_failure(monkeypatch):
    from protein_interface import AtomArrays
    import protein_interface.interface as iface

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic SC failure")

    monkeypatch.setattr(iface, "compute_sc", boom)
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CA"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="synthetic SC failure"):
        analyze(a, b)


def test_analyze_permissive_sc_failure_returns_nan(monkeypatch):
    from protein_interface import AtomArrays
    import protein_interface.interface as iface

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic SC failure")

    monkeypatch.setattr(iface, "compute_sc", boom)
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CA"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    res = analyze(a, b, strict=False)
    assert math.isnan(res.sc)


def test_atomarrays_rejects_side_qualified_public_residue_ids():
    from protein_interface import AtomArrays
    with pytest.raises(ValueError, match="internal-only"):
        AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("a", "A", 1, "")])


@pytest.mark.parametrize("probe_radius", [-1.0, math.nan])
def test_analyze_rejects_invalid_probe_radius(probe_radius):
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="probe_radius"):
        analyze(a, b, probe_radius=probe_radius, include_sc=False)


def test_analyze_batch_rejects_invalid_probe_radius():
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="probe_radius"):
        analyze_batch([(a, b)], probe_radius=math.nan, include_sc=False)


@pytest.mark.parametrize("cutoff", [-5.0, math.nan])
def test_interface_residues_rejects_invalid_cutoff(cutoff):
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CA"], ["ALA"], [("A", 1)])
    b = AtomArrays([[4.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="cutoff"):
        interface_residues(a, b, cutoff=cutoff)


@pytest.mark.parametrize("metric_name", [
    "hbonds",
    "salt_bridges",
    "atomic_contacts",
    "disulfide_bridges",
    "pi_pi_contacts",
    "cation_pi_contacts",
])
@pytest.mark.parametrize("cutoff", [-1.0, math.nan])
def test_contact_metrics_reject_invalid_cutoffs(metric_name, cutoff):
    import protein_interface as pi
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["SG"], ["CYS"], [("A", 1)])
    b = AtomArrays([[2.0, 0.0, 0.0]], ["SG"], ["CYS"], [("B", 1)])
    metric = getattr(pi, metric_name)
    kwargs = {"distance_cutoff": cutoff} if metric_name in {"pi_pi_contacts", "cation_pi_contacts"} else {"cutoff": cutoff}
    with pytest.raises(ValueError):
        metric(a, b, **kwargs)


def test_analyze_rejects_invalid_threshold_parameters():
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    with pytest.raises(ValueError, match="n_points"):
        analyze(a, b, n_points=3, include_sc=False)
    with pytest.raises(ValueError, match="hbond_cutoff"):
        analyze(a, b, hbond_cutoff=math.nan, include_sc=False)
    with pytest.raises(ValueError, match="min_atom_dsasa_for_shape"):
        analyze(a, b, min_atom_dsasa_for_shape=-0.1, include_sc=False)


@pytest.mark.parametrize("metric_name,kwargs", [
    ("interface_depth", {"min_atom_dsasa": -0.1}),
    ("confidence_at_interface", {"min_atom_dsasa": -0.1}),
    ("buried_unsat_polar", {"sasa_cutoff": math.nan}),
    ("buried_unsat_polar", {"hbond_cutoff": -1.0}),
    ("interface_shape", {"min_atom_dsasa": math.nan}),
    ("gly_pro_fraction", {"cutoff": -1.0}),
    ("charge_complementarity", {"cutoff": math.nan}),
    ("prodigy_ics", {"cutoff": -1.0}),
    ("prodigy_nis", {"rsasa_cutoff": math.nan}),
    ("prodigy", {"cutoff": math.nan}),
])
def test_public_metric_helpers_reject_invalid_parameters(metric_name, kwargs):
    import protein_interface as pi
    from protein_interface import AtomArrays
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)], bfactors=[80.0])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)], bfactors=[70.0])
    with pytest.raises(ValueError):
        getattr(pi, metric_name)(a, b, **kwargs)


# ── New-metric unit tests ────────────────────────────────────────────────────

def test_salt_bridges_simple_pair():
    from protein_interface import AtomArrays, salt_bridges
    a = AtomArrays([[0.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("A", 1)])
    b = AtomArrays([[3.5, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    assert salt_bridges(a, b) == 1


def test_salt_bridges_deduplicates_residue_pair():
    from protein_interface import AtomArrays, salt_bridges
    a = AtomArrays(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["OD1", "OD2"], ["ASP", "ASP"],
        [("A", 1), ("A", 1)],
    )
    b = AtomArrays([[0.5, 0.5, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    assert salt_bridges(a, b) == 1


def test_salt_bridges_counts_distinct_residue_pairs():
    from protein_interface import AtomArrays, salt_bridges
    a = AtomArrays(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ["OD1", "OE1"], ["ASP", "GLU"],
        [("A", 1), ("A", 2)],
    )
    b = AtomArrays(
        [[3.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
        ["NZ", "NH1"], ["LYS", "ARG"],
        [("B", 1), ("B", 2)],
    )
    assert salt_bridges(a, b) == 2


def test_salt_bridges_excludes_non_charged():
    from protein_interface import AtomArrays, salt_bridges
    a = AtomArrays([[0.0, 0.0, 0.0]], ["O"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.5, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    assert salt_bridges(a, b) == 0


def test_bsa_breakdown_sums_to_dsasa(nb_ag):
    from protein_interface import bsa_breakdown
    a, b = nb_ag
    r = bsa_breakdown(a, b, n_points=92)
    # bhsa + bpsa should equal dsasa to within rounding (other atom letters are rare)
    assert abs((r["bhsa"] + r["bpsa"]) - r["dsasa"]) < 1.0
    assert r["bcsa"] <= r["bpsa"] + 1e-6
    assert 0.0 <= r["hydrophobic_fraction"] <= 1.0


def test_per_residue_dsasa_consistent_with_total(nb_ag):
    from protein_interface import delta_sasa, per_residue_dsasa
    a, b = nb_ag
    per_res = per_residue_dsasa(a, b, n_points=92)
    d_total = delta_sasa(a, b, n_points=92)
    assert abs(sum(per_res.values()) - d_total) < 1.0


def test_per_residue_dsasa_side_qualified_keys_do_not_collide():
    from protein_interface import AtomArrays, per_residue_dsasa
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)])
    b = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["GLY"], [("A", 1)])
    per_res = per_residue_dsasa(a, b, n_points=92)
    assert set(per_res) == {("a", "A", 1, ""), ("b", "A", 1, "")}


def test_residue_ids_preserve_insertion_codes():
    from protein_interface import AtomArrays, per_residue_dsasa
    a = AtomArrays(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ["CB", "CB"], ["ALA", "ALA"],
        [("A", 10, "A"), ("A", 10, "B")],
    )
    b = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    per_res = per_residue_dsasa(a, b, n_points=92)
    assert ("a", "A", 10, "A") in per_res
    assert ("a", "A", 10, "B") in per_res


def test_hotspot_residues_threshold():
    from protein_interface import hotspot_residues
    per = {("A", 1): 50.0, ("A", 2): 10.0, ("B", 5): 100.0}
    hs = hotspot_residues(per, threshold=30.0)
    assert hs == [("B", 5), ("A", 1)]


def test_hbond_density_basic():
    from protein_interface import hbond_density
    assert hbond_density(5, 1000.0) == 0.5
    assert hbond_density(0, 0.0) == 0.0


def test_pi_pi_synthetic_pair():
    """Two coplanar PHE rings 5 Å apart should give one π-π contact."""
    from protein_interface import AtomArrays, pi_pi_contacts
    # Six points of a regular hexagon (r=1.4 Å), centred at origin.
    ring_xy = [
        (1.4, 0.0), (0.7, 1.21), (-0.7, 1.21),
        (-1.4, 0.0), (-0.7, -1.21), (0.7, -1.21),
    ]
    names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    coords_a = [[x, y, 0.0] for x, y in ring_xy]
    coords_b = [[x, y, 5.0] for x, y in ring_xy]
    a = AtomArrays(coords_a, names, ["PHE"] * 6, [("A", 1)] * 6)
    b = AtomArrays(coords_b, names, ["PHE"] * 6, [("B", 1)] * 6)
    assert pi_pi_contacts(a, b, distance_cutoff=7.0) == 1
    # Beyond cutoff: no contact.
    coords_b2 = [[x, y, 10.0] for x, y in ring_xy]
    b2 = AtomArrays(coords_b2, names, ["PHE"] * 6, [("B", 1)] * 6)
    assert pi_pi_contacts(a, b2, distance_cutoff=7.0) == 0


def test_cation_pi_synthetic():
    """A LYS NZ 5 Å from a PHE ring centroid → one cation-π contact."""
    from protein_interface import AtomArrays, cation_pi_contacts
    ring_xy = [
        (1.4, 0.0), (0.7, 1.21), (-0.7, 1.21),
        (-1.4, 0.0), (-0.7, -1.21), (0.7, -1.21),
    ]
    ring_names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    ring = AtomArrays(
        [[x, y, 0.0] for x, y in ring_xy],
        ring_names, ["PHE"] * 6, [("A", 1)] * 6,
    )
    cation = AtomArrays([[0.0, 0.0, 5.0]], ["NZ"], ["LYS"], [("B", 2)])
    assert cation_pi_contacts(ring, cation, distance_cutoff=6.0) == 1
    far = AtomArrays([[0.0, 0.0, 10.0]], ["NZ"], ["LYS"], [("B", 2)])
    assert cation_pi_contacts(ring, far, distance_cutoff=6.0) == 0


def test_interface_shape_planar_patch():
    """A flat grid of CB atoms across an interface should return ~0 planarity RMSD."""
    from protein_interface import AtomArrays, interface_shape
    # 5×5 grid of CB atoms at z=0 (side A) and z=4 (side B): when scored, only
    # atoms with measurable dSASA contribute; the patches are close enough that
    # interior atoms get buried.
    coords_a = [[x * 2.0, y * 2.0, 0.0] for x in range(5) for y in range(5)]
    coords_b = [[x * 2.0, y * 2.0, 4.0] for x in range(5) for y in range(5)]
    a = AtomArrays(coords_a, ["CB"] * 25, ["ALA"] * 25, [("A", i) for i in range(25)])
    b = AtomArrays(coords_b, ["CB"] * 25, ["ALA"] * 25, [("B", i) for i in range(25)])
    s = interface_shape(a, b, n_points=92)
    # Two parallel planes 4 Å apart → small but nonzero planarity_rmsd (~2 Å).
    assert s["planarity_rmsd"] < 5.0
    assert s["elongation"] >= 1.0


def _tight_cage(atom_name: str = "CB", residue: str = "ALA", radius: float = 2.2):
    """Return cage coords/names/res arrays that fully bury an atom at origin.

    Six face directions + eight corner directions, both at `radius` Å. With
    radius 2.2 Å and CB MS radius 1.95 + probe 1.4 = inflated 3.35 Å, every
    sphere point on a probed N at origin lies inside at least one cage atom's
    inflated shell, giving sasa_complex(N) ≈ 0.
    """
    coords: list[list[float]] = []
    for ax in range(3):
        for sign in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0]
            v[ax] = sign * radius
            coords.append(v)
    r_corner = radius / math.sqrt(3.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                coords.append([sx * r_corner, sy * r_corner, sz * r_corner])
    n = len(coords)
    return coords, [atom_name] * n, [residue] * n


def test_buried_unsat_polar_isolated_buried_donor():
    """A buried backbone N with no acceptor nearby on the other side is unsatisfied."""
    from protein_interface import AtomArrays, buried_unsat_polar
    cage_c, cage_n, cage_r = _tight_cage()
    a = AtomArrays([[0.0, 0.0, 0.0]], ["N"], ["ALA"], [("A", 1)])
    b = AtomArrays(cage_c, cage_n, cage_r, [("B", i) for i in range(len(cage_c))])
    assert buried_unsat_polar(a, b, n_points=92) == 1


def test_buried_unsat_polar_satisfied_when_partner_present():
    """The same buried N becomes satisfied if a backbone O sits within 3.5 Å on the other side."""
    from protein_interface import AtomArrays, buried_unsat_polar
    cage_c, cage_n, cage_r = _tight_cage()
    # Add a backbone O acceptor inside cutoff (the cage already provides burial).
    cage_c = cage_c + [[2.9, 0.0, 0.0]]
    cage_n = cage_n + ["O"]
    cage_r = cage_r + ["ALA"]
    a = AtomArrays([[0.0, 0.0, 0.0]], ["N"], ["ALA"], [("A", 1)])
    b = AtomArrays(cage_c, cage_n, cage_r, [("B", i) for i in range(len(cage_c))])
    assert buried_unsat_polar(a, b, n_points=92) == 0


def test_analyze_returns_all_new_fields(nb_ag):
    a, b = nb_ag
    res = analyze(a, b, n_points=92)
    assert res.salt_bridges >= 0
    assert res.bhsa >= 0
    assert res.bpsa >= 0
    assert res.bcsa >= 0
    assert 0.0 <= res.hydrophobic_fraction <= 1.0
    assert res.hbond_density >= 0
    assert res.pi_pi >= 0
    assert res.cation_pi >= 0
    assert res.buried_unsat_polar >= 0
    # PCA values should be finite for a real interface.
    assert res.planarity_rmsd == res.planarity_rmsd  # not NaN
    assert res.elongation >= 1.0
    assert isinstance(res.hotspots_a, list)
    assert isinstance(res.hotspots_b, list)


# ── Atomic contacts, asymmetry, depth, disulfides, Gly/Pro ───────────────────

def test_atomic_contacts_pair_within_cutoff():
    from protein_interface import AtomArrays, atomic_contacts
    a = AtomArrays([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], ["CA"] * 2, ["ALA"] * 2,
                   [("A", 1), ("A", 2)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CA"], ["ALA"], [("B", 1)])
    # Only the first A atom is within 5 Å of B; the second is 7 Å away.
    assert atomic_contacts(a, b, cutoff=5.0) == 1


def test_atomic_contacts_empty():
    from protein_interface import AtomArrays, atomic_contacts
    empty = AtomArrays([], [], [], [])
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CA"], ["ALA"], [("A", 1)])
    assert atomic_contacts(a, empty) == 0


def test_disulfide_bridges_pair():
    from protein_interface import AtomArrays, disulfide_bridges
    a = AtomArrays([[0.0, 0.0, 0.0]], ["SG"], ["CYS"], [("A", 1)])
    b = AtomArrays([[2.05, 0.0, 0.0]], ["SG"], ["CYS"], [("B", 1)])
    assert disulfide_bridges(a, b) == 1


def test_disulfide_bridges_ignores_non_cys_sg():
    from protein_interface import AtomArrays, disulfide_bridges
    a = AtomArrays([[0.0, 0.0, 0.0]], ["SG"], ["CYS"], [("A", 1)])
    b = AtomArrays([[2.05, 0.0, 0.0]], ["SG"], ["MET"], [("B", 1)])
    assert disulfide_bridges(a, b) == 0


def test_asymmetry_symmetric_pair(nb_ag):
    from protein_interface import asymmetry
    a, b = nb_ag
    d_a, d_b, asym = asymmetry(a, b, n_points=92)
    assert d_a > 0 and d_b > 0
    # Real interfaces are rarely perfectly symmetric, but should still be < 0.5
    # (one side buries < 1.5× the other).
    assert 0.0 <= asym < 0.5


def test_interface_depth_planar_offset():
    """Two parallel CB grids 4 Å apart → depth ≈ 4 Å."""
    from protein_interface import AtomArrays, interface_depth
    coords_a = [[x * 2.0, y * 2.0, 0.0] for x in range(5) for y in range(5)]
    coords_b = [[x * 2.0, y * 2.0, 4.0] for x in range(5) for y in range(5)]
    a = AtomArrays(coords_a, ["CB"] * 25, ["ALA"] * 25, [("A", i) for i in range(25)])
    b = AtomArrays(coords_b, ["CB"] * 25, ["ALA"] * 25, [("B", i) for i in range(25)])
    depth = interface_depth(a, b, n_points=92)
    assert 3.0 < depth < 5.0, f"depth = {depth}"


def test_gly_pro_fraction_synthetic():
    from protein_interface import AtomArrays, gly_pro_fraction
    # A side: 1 Gly + 1 Ala in contact. B side: 1 Pro + 1 Ala in contact.
    a = AtomArrays(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ["CA", "CA"], ["GLY", "ALA"],
        [("A", 1), ("A", 2)],
    )
    b = AtomArrays(
        [[2.5, 0.0, 0.0], [5.5, 0.0, 0.0]],
        ["CA", "CA"], ["PRO", "ALA"],
        [("B", 1), ("B", 2)],
    )
    # All four residues end up at the interface (5 Å cutoff). 2 of 4 are Gly/Pro.
    assert gly_pro_fraction(a, b) == 0.5


def test_analyze_returns_remaining_fields(nb_ag):
    a, b = nb_ag
    res = analyze(a, b, n_points=92)
    assert res.dsasa_a > 0
    assert res.dsasa_b > 0
    assert 0.0 <= res.asymmetry < 1.0
    assert res.atomic_contacts > 0
    assert res.interface_depth == res.interface_depth  # not NaN
    assert res.disulfides >= 0
    assert 0.0 <= res.gly_pro_fraction <= 1.0
    # dsasa_a + dsasa_b should match total dsasa.
    assert abs((res.dsasa_a + res.dsasa_b) - res.dsasa) < 0.5


# ── Confidence (B-factor), backbone/sidechain, charge complementarity ────────

def test_load_atoms_populates_bfactors(nb_ag):
    a, _ = nb_ag
    assert a.bfactors is not None
    assert len(a.bfactors) == len(a.coords)
    # 1ZVH is an X-ray structure: typical B-factors 5–80 Å².
    assert all(0.0 <= bf <= 200.0 for bf in a.bfactors)


def test_confidence_at_interface_synthetic():
    """A single A-side atom buried by a cage: mean and min equal its B-factor."""
    from protein_interface import AtomArrays, confidence_at_interface
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)], bfactors=[80.0])
    cage_c, cage_n, cage_r = _tight_cage()
    b = AtomArrays(
        cage_c, cage_n, cage_r,
        [("B", i) for i in range(len(cage_c))],
        bfactors=[50.0] * len(cage_c),
    )
    r = confidence_at_interface(a, b, n_points=92)
    # Only A's CB has dSASA above threshold here (cage atoms collectively bury it).
    # mean and min should report its bfactor.
    assert r["mean"] == pytest.approx(80.0, abs=0.001)
    assert r["min"] == pytest.approx(80.0, abs=0.001)


def test_confidence_at_interface_returns_nan_without_bfactors(nb_ag):
    from protein_interface import AtomArrays, confidence_at_interface
    a, b = nb_ag
    a_no_bf = AtomArrays(a.coords, a.atom_names, a.residue_names, a.residue_ids, bfactors=None)
    r = confidence_at_interface(a_no_bf, b, n_points=92)
    assert math.isnan(r["mean"]) and math.isnan(r["min"])


def test_bb_sc_split_synthetic():
    from protein_interface import AtomArrays, bb_sc_dsasa_split
    # One backbone N on A, one side-chain CB on B; both buried by tight cages.
    a = AtomArrays([[0.0, 0.0, 0.0]], ["N"], ["ALA"], [("A", 1)])
    b = AtomArrays([[20.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    r = bb_sc_dsasa_split(a, b, n_points=92)
    # Both atoms are isolated (no opposing chain nearby) so dSASA is ~0.
    # Just verify split is structurally correct: backbone and side-chain sums add.
    assert r["bb_dsasa"] + r["sc_dsasa"] == pytest.approx(0.0, abs=1.0)


def test_bb_sc_split_sums_to_dsasa(nb_ag):
    from protein_interface import bb_sc_dsasa_split, delta_sasa
    a, b = nb_ag
    r = bb_sc_dsasa_split(a, b, n_points=92)
    total = delta_sasa(a, b, n_points=92)
    assert abs((r["bb_dsasa"] + r["sc_dsasa"]) - total) < 1.0
    assert 0.0 <= r["sidechain_fraction"] <= 1.0


def test_charge_complementarity_opposite():
    """ASP next to LYS across the interface → positive complementarity."""
    from protein_interface import AtomArrays, charge_complementarity
    a = AtomArrays([[0.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["NZ"], ["LYS"], [("B", 1)])
    r = charge_complementarity(a, b)
    assert r["charge_a"] == -1
    assert r["charge_b"] == +1
    assert r["complementarity"] == 1.0


def test_charge_complementarity_same_sign():
    """Two ASPs facing each other → negative (repulsive)."""
    from protein_interface import AtomArrays, charge_complementarity
    a = AtomArrays([[0.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["OD1"], ["ASP"], [("B", 1)])
    r = charge_complementarity(a, b)
    assert r["complementarity"] == -1.0


def test_charge_complementarity_neutral():
    """All ALA interface → 0."""
    from protein_interface import AtomArrays, charge_complementarity
    a = AtomArrays([[0.0, 0.0, 0.0]], ["CB"], ["ALA"], [("A", 1)])
    b = AtomArrays([[3.0, 0.0, 0.0]], ["CB"], ["ALA"], [("B", 1)])
    r = charge_complementarity(a, b)
    assert r["charge_a"] == 0 and r["charge_b"] == 0
    assert r["complementarity"] == 0.0


def test_compute_sasa_batch_matches_compute_sasa(nb_ag):
    """Batched SASA must produce bit-identical per-atom arrays vs serial calls."""
    from protein_interface import compute_sasa, compute_sasa_batch
    a, b = nb_ag
    serial_a = compute_sasa(a.coords, a.atom_names, a.residue_names, 1.4, 92)
    serial_b = compute_sasa(b.coords, b.atom_names, b.residue_names, 1.4, 92)
    batch = compute_sasa_batch(
        [(a.coords, a.atom_names, a.residue_names),
         (b.coords, b.atom_names, b.residue_names)],
        1.4, 92, True,
    )
    assert batch[0] == serial_a
    assert batch[1] == serial_b


def test_compute_sasa_batch_empty():
    from protein_interface import compute_sasa_batch
    assert compute_sasa_batch([], 1.4, 92, True) == []


def test_analyze_batch_matches_analyze(nb_ag):
    """analyze_batch must match per-complex analyze() across every numeric field."""
    from dataclasses import fields
    from protein_interface import analyze, analyze_batch
    a, b = nb_ag
    serial = [analyze(a, b), analyze(b, a)]  # two complexes, second swapped
    batched = analyze_batch([(a, b), (b, a)])
    assert len(batched) == 2
    for s, ba in zip(serial, batched):
        for f in fields(s):
            v_s = getattr(s, f.name)
            v_b = getattr(ba, f.name)
            if isinstance(v_s, float):
                if math.isnan(v_s):
                    assert math.isnan(v_b), f"{f.name}: serial NaN, batch {v_b}"
                else:
                    assert v_s == pytest.approx(v_b, rel=1e-9, abs=1e-9), f"{f.name}: {v_s} vs {v_b}"
            else:
                assert v_s == v_b, f"{f.name}: {v_s} vs {v_b}"


def test_analyze_batch_empty():
    from protein_interface import analyze_batch
    assert analyze_batch([]) == []


def test_analyze_returns_confidence_bbsc_charge(nb_ag):
    a, b = nb_ag
    res = analyze(a, b, n_points=92)
    # B-factor block — 1ZVH is X-ray, so values are real thermal Bs.
    assert not math.isnan(res.mean_bfactor_interface)
    assert res.min_bfactor_interface <= res.mean_bfactor_interface
    # Backbone + side-chain dSASA should sum to total.
    assert abs((res.bb_dsasa + res.sc_dsasa) - res.dsasa) < 0.5
    assert 0.0 <= res.sidechain_fraction <= 1.0
    # Charges are integers; complementarity has the right sign convention.
    assert isinstance(res.charge_a, int)
    assert isinstance(res.charge_b, int)
    assert res.charge_complementarity == -res.charge_a * res.charge_b
