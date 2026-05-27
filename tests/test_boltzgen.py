"""Tests for BoltzGen integration (from_biotite, from_boltzgen_structure, from_boltzgen_refold).

from_biotite and from_boltzgen_refold require biotite; tests are skipped when not installed.
from_boltzgen_structure is tested via duck-typed numpy mocks — no boltzgen import needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from protein_interface import from_biotite, from_boltzgen_structure
from protein_interface._core import ScResult
from protein_interface.io import _is_hydrogen


# ── helpers to build minimal mock Structure ──────────────────────────────────

def _make_structure(chain_specs: list[tuple[str, list[tuple[str, list[tuple[str, tuple[float, float, float]]]]]]]
                    ):
    """Build a duck-typed mock matching BoltzGen's Structure numpy layout.

    chain_specs: list of (chain_name, residues)
    residues:    list of (res_name, atoms)
    atoms:       list of (atom_name, (x, y, z))

    Returns an object with .atoms, .residues, .chains structured arrays.
    """
    Atom_dt = np.dtype([
        ("name", "<U4"),
        ("coords", "3f4"),
        ("is_present", "?"),
        ("bfactor", "f4"),
        ("plddt", "f4"),
    ])
    Residue_dt = np.dtype([
        ("name", "<U5"),
        ("res_type", "i1"),
        ("res_idx", "i4"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
        ("atom_center", "i4"),
        ("atom_disto", "i4"),
        ("is_standard", "?"),
        ("is_present", "?"),
    ])
    Chain_dt = np.dtype([
        ("name", "<U5"),
        ("mol_type", "i1"),
        ("entity_id", "i4"),
        ("sym_id", "i4"),
        ("asym_id", "i4"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
        ("res_idx", "i4"),
        ("res_num", "i4"),
        ("cyclic_period", "i4"),
        ("symmetric_group", "i4"),
    ])

    atom_rows = []
    residue_rows = []
    chain_rows = []
    global_atom_idx = 0
    global_res_idx = 0

    for chain_name, residues in chain_specs:
        chain_atom_start = global_atom_idx
        chain_res_start = global_res_idx
        chain_atom_count = 0

        for res_name, atoms in residues:
            res_atom_start = global_atom_idx
            for atom_name, (x, y, z) in atoms:
                atom_rows.append((atom_name, (x, y, z), True, 0.0, 0.0))
                global_atom_idx += 1
                chain_atom_count += 1

            residue_rows.append((
                res_name, 0, global_res_idx, res_atom_start, len(atoms), 0, 0, True, True
            ))
            global_res_idx += 1

        chain_rows.append((
            chain_name, 1, 0, 0, 0,
            chain_atom_start, chain_atom_count,
            chain_res_start, len(residues),
            0, 0,
        ))

    class MockStructure:
        atoms = np.array(atom_rows, dtype=Atom_dt)
        residues = np.array(residue_rows, dtype=Residue_dt)
        chains = np.array(chain_rows, dtype=Chain_dt)

    return MockStructure()


def _ala_chain(chain_name: str, n: int, offset: tuple[float, float, float]) -> tuple:
    """n ALA residues with N, CA, C, O atoms, chain offset applied to all coords."""
    dx, dy, dz = offset
    residues = []
    for i in range(n):
        x = i * 3.8 + dx
        atoms = [
            ("N",  (x - 1.2, dy, dz)),
            ("CA", (x,       dy, dz)),
            ("C",  (x + 1.1, dy, dz)),
            ("O",  (x + 1.6, dy + 1.0, dz)),
            ("CB", (x,       dy - 1.5, dz)),
        ]
        residues.append(("ALA", atoms))
    return (chain_name, residues)


# ── Layout validator tests ───────────────────────────────────────────────────

def test_from_boltzgen_structure_rejects_missing_attribute():
    """A plain object without .chains / .residues / .atoms must fail cleanly."""
    class NotAStructure:
        pass
    with pytest.raises(TypeError, match=r"missing \.chains"):
        from_boltzgen_structure(NotAStructure(), chains_a=["A"], chains_b=["B"])


def test_from_boltzgen_structure_rejects_non_structured_array():
    """.chains being a plain ndarray (no dtype.names) should fail loudly."""
    class WrongShape:
        chains = np.arange(10)        # plain array, no fields
        residues = np.arange(10)
        atoms = np.arange(10)
    with pytest.raises(TypeError, match="not a numpy structured array"):
        from_boltzgen_structure(WrongShape(), chains_a=["A"], chains_b=["B"])


def test_from_boltzgen_structure_rejects_missing_field():
    """A structured array missing a required field should name the missing field."""
    bad_chain_dt = np.dtype([("name", "<U4")])  # missing res_idx, res_num
    Atom_dt = np.dtype([("name", "<U4"), ("coords", "3f4"), ("is_present", "?")])
    Residue_dt = np.dtype([("name", "<U5"), ("atom_idx", "i4"), ("atom_num", "i4")])
    class Drift:
        chains = np.array([("A",)], dtype=bad_chain_dt)
        residues = np.zeros(0, dtype=Residue_dt)
        atoms = np.zeros(0, dtype=Atom_dt)
    with pytest.raises(TypeError, match=r"\['res_idx', 'res_num'\]"):
        from_boltzgen_structure(Drift(), chains_a=["A"], chains_b=["B"])


# ── from_boltzgen_structure tests ────────────────────────────────────────────

def test_from_boltzgen_structure_smoke():
    struct = _make_structure([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    result = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"])
    assert isinstance(result, ScResult)
    assert -1.0 <= result.sc <= 1.0
    assert result.atoms_a > 0
    assert result.atoms_b > 0


def test_from_boltzgen_structure_chains_b_none():
    struct = _make_structure([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    r_explicit = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"])
    r_implicit = from_boltzgen_structure(struct, chains_a=["A"], chains_b=None)
    assert r_explicit.sc == pytest.approx(r_implicit.sc, abs=1e-9)
    assert r_explicit.atoms_b == r_implicit.atoms_b


def test_from_boltzgen_structure_missing_chain():
    struct = _make_structure([
        _ala_chain("A", 5, (0.0, 0.0, 0.0)),
        _ala_chain("B", 5, (0.0, 5.0, 0.0)),
    ])
    with pytest.raises(ValueError, match="not found"):
        from_boltzgen_structure(struct, chains_a=["Z"], chains_b=["A"])


def test_from_boltzgen_structure_skips_absent_atoms():
    """Atoms with is_present=False must be excluded."""
    Atom_dt = np.dtype([
        ("name", "<U4"), ("coords", "3f4"), ("is_present", "?"),
        ("bfactor", "f4"), ("plddt", "f4"),
    ])
    Residue_dt = np.dtype([
        ("name", "<U5"), ("res_type", "i1"), ("res_idx", "i4"),
        ("atom_idx", "i4"), ("atom_num", "i4"), ("atom_center", "i4"),
        ("atom_disto", "i4"), ("is_standard", "?"), ("is_present", "?"),
    ])
    Chain_dt = np.dtype([
        ("name", "<U5"), ("mol_type", "i1"), ("entity_id", "i4"),
        ("sym_id", "i4"), ("asym_id", "i4"), ("atom_idx", "i4"),
        ("atom_num", "i4"), ("res_idx", "i4"), ("res_num", "i4"),
        ("cyclic_period", "i4"), ("symmetric_group", "i4"),
    ])

    # Chain A: 1 residue, 2 atoms (CA present, CB absent)
    # Chain B: 10 ALA for a valid group B
    struct_a = _make_structure([_ala_chain("B", 10, (0.0, 5.0, 0.0))])

    atoms = np.array([
        ("CA", (0.0, 0.0, 0.0), True,  0.0, 0.0),
        ("CB", (0.0, 0.0, 0.0), False, 0.0, 0.0),  # absent
    ], dtype=Atom_dt)
    residues = np.array([("ALA", 0, 0, 0, 2, 0, 0, True, True)], dtype=Residue_dt)
    chains_a = np.array([("A", 1, 0, 0, 0, 0, 2, 0, 1, 0, 0)], dtype=Chain_dt)
    chains_b = struct_a.chains

    class MockStruct:
        pass

    s = MockStruct()
    s.atoms = np.concatenate([atoms, struct_a.atoms])
    s.residues = np.concatenate([residues, struct_a.residues])
    # Fix chain B atom/res offsets
    chains_b_fixed = chains_b.copy()
    chains_b_fixed["atom_idx"] += 2
    chains_b_fixed["res_idx"] += 1
    s.chains = np.concatenate([chains_a, chains_b_fixed])

    result = from_boltzgen_structure(s, chains_a=["A"], chains_b=["B"])
    assert result.atoms_a == 1  # only the present CA atom


def test_from_boltzgen_structure_hydrogen_excluded():
    """Hydrogens in atom names are excluded by default (atoms_a < include_hydrogens case)."""
    h_residues = [
        ("ALA", [
            ("N",  (i * 3.8 - 1.2, 0.0, 0.0)),
            ("CA", (i * 3.8,       0.0, 0.0)),
            ("C",  (i * 3.8 + 1.1, 0.0, 0.0)),
            ("O",  (i * 3.8 + 1.6, 1.0, 0.0)),
            ("CB", (i * 3.8,      -1.5, 0.0)),
            ("H",  (i * 3.8 - 1.5, 0.5, 0.0)),
        ])
        for i in range(10)
    ]
    struct = _make_structure([
        ("A", h_residues),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    result = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"])
    # 5 heavy atoms per residue × 10 residues = 50; H excluded
    assert result.atoms_a == 50


def test_from_boltzgen_structure_hydrogen_included():
    """include_hydrogens=True makes atoms_a larger than with the default (False)."""
    # Mix of heavy and H atoms in each ALA residue
    h_residues = [
        ("ALA", [
            ("N",  (i * 3.8 - 1.2, 0.0, 0.0)),
            ("CA", (i * 3.8,       0.0, 0.0)),
            ("C",  (i * 3.8 + 1.1, 0.0, 0.0)),
            ("O",  (i * 3.8 + 1.6, 1.0, 0.0)),
            ("CB", (i * 3.8,      -1.5, 0.0)),
            ("H",  (i * 3.8 - 1.5, 0.5, 0.0)),  # backbone H
        ])
        for i in range(10)
    ]
    struct = _make_structure([
        ("A", h_residues),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    r_no_h = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"],
                                     include_hydrogens=False)
    r_with_h = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"],
                                       include_hydrogens=True)
    assert r_with_h.atoms_a > r_no_h.atoms_a


def test_from_boltzgen_structure_determinism():
    struct = _make_structure([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    r1 = from_boltzgen_structure(struct, ["A"], ["B"])
    r2 = from_boltzgen_structure(struct, ["A"], ["B"])
    assert r1.sc == r2.sc
    assert r1.median_distance == r2.median_distance


# ── from_biotite tests (skipped if biotite not installed) ────────────────────

try:
    import biotite.structure as _biotite_struc
    _HAS_BIOTITE = True
    _BIOTITE_SKIP_REASON = ""
except Exception as exc:
    _HAS_BIOTITE = False
    _BIOTITE_SKIP_REASON = f"biotite unavailable or incompatible: {exc}"

requires_biotite = pytest.mark.skipif(
    not _HAS_BIOTITE,
    reason=_BIOTITE_SKIP_REASON or "biotite not installed; install with `pip install biotite`",
)


def _make_biotite_array(chain_specs):
    """Build a minimal biotite AtomArray from the same chain_specs format."""
    import biotite.structure as struc

    atoms_list = []
    for chain_name, residues in chain_specs:
        res_id = 1
        for res_name, atom_defs in residues:
            for atom_name, (x, y, z) in atom_defs:
                a = struc.Atom(
                    coord=[x, y, z],
                    chain_id=chain_name,
                    res_id=res_id,
                    res_name=res_name,
                    atom_name=atom_name,
                    element=atom_name[0] if not atom_name[0].isdigit() else atom_name[1],
                    hetero=False,
                )
                atoms_list.append(a)
            res_id += 1

    return struc.array(atoms_list)


@requires_biotite
def test_from_biotite_smoke():
    arr = _make_biotite_array([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    result = from_biotite(arr, chains_a=["A"], chains_b=["B"])
    assert isinstance(result, ScResult)
    assert -1.0 <= result.sc <= 1.0
    assert result.atoms_a > 0
    assert result.atoms_b > 0


@requires_biotite
def test_from_biotite_matches_boltzgen_structure():
    """from_biotite and from_boltzgen_structure on identical atoms must agree."""
    chain_specs = [
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ]
    bio_arr = _make_biotite_array(chain_specs)
    boltz_struct = _make_structure(chain_specs)

    r_bio = from_biotite(bio_arr, ["A"], ["B"])
    r_boltz = from_boltzgen_structure(boltz_struct, ["A"], ["B"])

    # Same atoms → same result
    assert r_bio.atoms_a == r_boltz.atoms_a
    assert r_bio.atoms_b == r_boltz.atoms_b
    assert r_bio.sc == pytest.approx(r_boltz.sc, abs=1e-6)


@requires_biotite
def test_from_biotite_chains_b_none():
    arr = _make_biotite_array([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    r_explicit = from_biotite(arr, ["A"], ["B"])
    r_implicit = from_biotite(arr, ["A"], None)
    assert r_explicit.sc == pytest.approx(r_implicit.sc, abs=1e-9)


@requires_biotite
def test_from_biotite_hetatm_excluded():
    import biotite.structure as struc

    arr = _make_biotite_array([
        _ala_chain("A", 10, (0.0, 0.0, 0.0)),
        _ala_chain("B", 10, (0.0, 5.0, 0.0)),
    ])
    # Append a HETATM atom far away
    hetatm = struc.Atom(
        coord=[99.0, 99.0, 99.0],
        chain_id="A",
        res_id=999,
        res_name="SO4",
        atom_name="S",
        element="S",
        hetero=True,
    )
    arr_with_hetatm = arr + struc.array([hetatm])
    r_orig = from_biotite(arr, ["A"], ["B"])
    r_hetatm = from_biotite(arr_with_hetatm, ["A"], ["B"])
    assert r_orig.sc == pytest.approx(r_hetatm.sc, abs=1e-9)
