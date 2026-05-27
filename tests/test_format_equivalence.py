"""Cross-format equivalence tests.

Verify that loading the same structure through every supported intake path
(biopython-PDB, biopython-CIF, biotite-CIF, BoltzGen-Structure) yields the
same atom selection and the same downstream metrics.

Empirically all four paths agree to bit-identical SC values on 1FYT — these
tests lock that property in so a future parser change can't silently shift
the numbers.
"""
from __future__ import annotations

import math
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from protein_interface import (
    analyze,
    from_biotite,
    from_pdb,
    from_structure,
    load_atoms,
)
# from_boltzgen_structure is un-exported pending real-Structure end-to-end
# test coverage; reach into the module to exercise it.
from protein_interface.io import _is_hydrogen, _select_real_atom, from_boltzgen_structure

DATA = Path(__file__).parent / "data"
PDB_1FYT = DATA / "1fyt.pdb"
CIF_1FYT = DATA / "1fyt.cif"
CHAINS_A = ["A"]
CHAINS_B = ["B"]


# Both files must be present for this suite to be meaningful.
if not PDB_1FYT.exists() or not CIF_1FYT.exists():
    pytest.skip("1FYT PDB and CIF fixtures required", allow_module_level=True)


# ── biopython PDB ↔ CIF via load_atoms ───────────────────────────────────────

def test_load_atoms_pdb_cif_atom_counts_match():
    a_pdb = load_atoms(PDB_1FYT, CHAINS_A)
    a_cif = load_atoms(CIF_1FYT, CHAINS_A)
    assert len(a_pdb.coords) == len(a_cif.coords)
    assert a_pdb.atom_names == a_cif.atom_names
    assert a_pdb.residue_names == a_cif.residue_names


def test_load_atoms_pdb_cif_coords_match():
    """Per-atom coordinates should agree to within 0.001 Å — both parsers read
    the same numeric fields from the same RCSB-provided files."""
    a_pdb = load_atoms(PDB_1FYT, CHAINS_A)
    a_cif = load_atoms(CIF_1FYT, CHAINS_A)
    for i, (c1, c2) in enumerate(zip(a_pdb.coords, a_cif.coords)):
        for k in range(3):
            assert math.isclose(c1[k], c2[k], abs_tol=1e-3), (
                f"atom {i} coord {k}: pdb={c1[k]}, cif={c2[k]}"
            )


def test_load_atoms_pdb_cif_residue_ids_match():
    a_pdb = load_atoms(PDB_1FYT, CHAINS_A)
    a_cif = load_atoms(CIF_1FYT, CHAINS_A)
    assert a_pdb.residue_ids == a_cif.residue_ids


def test_analyze_pdb_cif_all_fields_match():
    """Every InterfaceResult field must agree (bit-identical for ints, tight
    tolerance for floats) between PDB and CIF inputs."""
    a_pdb = load_atoms(PDB_1FYT, CHAINS_A)
    b_pdb = load_atoms(PDB_1FYT, CHAINS_B)
    a_cif = load_atoms(CIF_1FYT, CHAINS_A)
    b_cif = load_atoms(CIF_1FYT, CHAINS_B)
    r_pdb = analyze(a_pdb, b_pdb)
    r_cif = analyze(a_cif, b_cif)
    for f in fields(r_pdb):
        v_pdb = getattr(r_pdb, f.name)
        v_cif = getattr(r_cif, f.name)
        if isinstance(v_pdb, float):
            if math.isnan(v_pdb):
                assert math.isnan(v_cif), f"{f.name}: pdb=NaN, cif={v_cif}"
            else:
                assert v_pdb == pytest.approx(v_cif, abs=1e-6, rel=1e-6), (
                    f"{f.name}: pdb={v_pdb}, cif={v_cif}"
                )
        else:
            assert v_pdb == v_cif, f"{f.name}: pdb={v_pdb}, cif={v_cif}"


# ── SC pipeline (from_pdb) ↔ analyze pipeline (load_atoms + analyze) ─────────

def test_from_pdb_sc_matches_analyze_sc():
    """SC from the SC-only entry point must match the SC inside InterfaceResult."""
    sc_only = from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B).sc
    a = load_atoms(PDB_1FYT, CHAINS_A)
    b = load_atoms(PDB_1FYT, CHAINS_B)
    sc_full = analyze(a, b).sc
    assert sc_only == pytest.approx(sc_full, abs=1e-6)


def test_from_pdb_cif_sc_matches_pdb_sc():
    """from_pdb() reads both formats; SC must agree."""
    r_pdb = from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B)
    r_cif = from_pdb(CIF_1FYT, CHAINS_A, CHAINS_B)
    assert r_pdb.sc == pytest.approx(r_cif.sc, abs=1e-6)
    assert r_pdb.atoms_a == r_cif.atoms_a
    assert r_pdb.atoms_b == r_cif.atoms_b


def test_from_structure_matches_from_pdb():
    """Passing a pre-parsed biopython Structure must give the same SC as reading the file."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("1fyt", str(PDB_1FYT))
    r_struct = from_structure(structure, CHAINS_A, CHAINS_B)
    r_pdb = from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B)
    assert r_struct.sc == pytest.approx(r_pdb.sc, abs=1e-6)
    assert r_struct.atoms_a == r_pdb.atoms_a
    assert r_struct.atoms_b == r_pdb.atoms_b


# ── biotite-CIF ↔ biopython-PDB ──────────────────────────────────────────────

biotite_pdbx = pytest.importorskip("biotite.structure.io.pdbx")


def test_biotite_cif_matches_biopython_pdb():
    """Same structure read through biotite (CIF) and biopython (PDB) should
    give the same SC and the same atom count."""
    cif = biotite_pdbx.CIFFile.read(str(CIF_1FYT))
    atoms = biotite_pdbx.get_structure(cif, model=1, use_author_fields=True)
    r_biotite = from_biotite(atoms, CHAINS_A, CHAINS_B)
    r_pdb = from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B)
    assert r_biotite.sc == pytest.approx(r_pdb.sc, abs=1e-3), (
        f"biotite={r_biotite.sc:.4f} vs biopython={r_pdb.sc:.4f}"
    )
    # Atom selection conventions are tight but not necessarily identical;
    # accept ±5 atoms / side.
    assert abs(r_biotite.atoms_a - r_pdb.atoms_a) <= 5
    assert abs(r_biotite.atoms_b - r_pdb.atoms_b) <= 5


# ── from_boltzgen_structure on a realistic input built from 1FYT ─────────────

def _chain_specs_from_biopython(structure, chains):
    """Convert a biopython Structure into the (chain, residues, atoms) nested-list
    format consumed by the _make_structure helper in test_boltzgen.py."""
    specs = []
    model = structure[0]
    for chain in model.get_chains():
        if chain.id not in chains:
            continue
        residues = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":  # skip HETATM
                continue
            atoms = []
            for da in residue.get_atoms():
                real = _select_real_atom(da)
                if real is None:
                    continue
                name = real.name.strip()
                elem = (real.element or "").strip()
                if _is_hydrogen(name, elem):
                    continue
                c = real.coord
                atoms.append((name, (float(c[0]), float(c[1]), float(c[2]))))
            residues.append((residue.resname.strip(), atoms))
        specs.append((chain.id, residues))
    return specs


def test_from_boltzgen_structure_on_1fyt_matches_from_pdb():
    """Build a BoltzGen-shaped Structure from 1FYT and round-trip via
    from_boltzgen_structure. Since both paths consume the same biopython
    atom selection, SC and atom counts must match exactly."""
    # Cross-test import of the mock-builder helper from test_boltzgen.py.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_boltzgen import _make_structure  # noqa: PLC0415

    from Bio.PDB import PDBParser  # noqa: PLC0415
    structure = PDBParser(QUIET=True).get_structure("1fyt", str(PDB_1FYT))
    specs = _chain_specs_from_biopython(structure, CHAINS_A + CHAINS_B)
    boltz_struct = _make_structure(specs)

    r_boltz = from_boltzgen_structure(boltz_struct, chains_a=CHAINS_A, chains_b=CHAINS_B)
    r_pdb = from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B)
    assert r_boltz.sc == pytest.approx(r_pdb.sc, abs=1e-6)
    assert r_boltz.atoms_a == r_pdb.atoms_a
    assert r_boltz.atoms_b == r_pdb.atoms_b
