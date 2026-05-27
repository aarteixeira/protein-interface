"""Validation against the real upstream BoltzGen Structure.

``from_boltzgen_structure`` is duck-typed against the layout of
``boltzgen.data.data.Structure``. This test file pins our assumptions
against the **actual** upstream module — when boltzgen is installed in the
test environment, we instantiate a real ``Structure`` (using their real
``Atom`` / ``Residue`` / ``Chain`` dtype constants) and run our function
against it. If upstream renames a field we depend on, the test fails
immediately, with no manual constant refresh needed.

Boltzgen pulls heavy dependencies (mashumaro, rdkit, torch) and is GPU-
centric, so it's not a hard test dep — the suite uses ``pytest.importorskip``
and skips cleanly on environments without it. To enable locally:

    pip install boltzgen --no-deps
    pip install mashumaro rdkit torch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from protein_interface import from_boltzgen_structure, from_pdb
from protein_interface.io import (
    _BOLTZGEN_REQUIRED_FIELDS,
    _is_hydrogen,
    _select_real_atom,
    _validate_boltzgen_layout,
)

bd = pytest.importorskip("boltzgen.data.data")

DATA = Path(__file__).parent / "data"
PDB_1FYT = DATA / "1fyt.pdb"


def _build_real_structure_from_pdb(pdb_path: Path, chains: list[str]):
    """Construct a real boltzgen.data.data.Structure from a PDB file, using
    upstream's dtype constants directly. Populates only the fields downstream
    consumers (us included) actually read; the rest are zero-filled."""
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    atom_rows: list[tuple] = []
    residue_rows: list[tuple] = []
    chain_rows: list[tuple] = []
    global_atom_idx = 0
    global_res_idx = 0

    def _row(dtype, **values):
        """Return a tuple in dtype.names order, defaulting missing fields to 0/False/''."""
        row = []
        for name in dtype.names:
            if name in values:
                row.append(values[name])
            else:
                kind = np.dtype(dtype.fields[name][0]).kind
                row.append("" if kind == "U" else False if kind == "b" else 0)
        return tuple(row)

    atom_dt = np.dtype(bd.Atom)
    residue_dt = np.dtype(bd.Residue)
    chain_dt = np.dtype(bd.Chain)

    for chain in structure[0].get_chains():
        if chain.id not in chains:
            continue
        chain_atom_start = global_atom_idx
        chain_res_start = global_res_idx
        chain_atom_count = 0
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            res_atom_start = global_atom_idx
            res_atom_count = 0
            for da in residue.get_atoms():
                real = _select_real_atom(da)
                if real is None:
                    continue
                name = real.name.strip()
                elem = (real.element or "").strip()
                if _is_hydrogen(name, elem):
                    continue
                c = real.coord
                atom_rows.append(_row(
                    atom_dt,
                    name=name,
                    coords=(float(c[0]), float(c[1]), float(c[2])),
                    is_present=True,
                ))
                global_atom_idx += 1
                chain_atom_count += 1
                res_atom_count += 1
            residue_rows.append(_row(
                residue_dt,
                name=residue.resname.strip(),
                res_idx=global_res_idx,
                atom_idx=res_atom_start,
                atom_num=res_atom_count,
                is_standard=True,
                is_present=True,
            ))
            global_res_idx += 1
        chain_rows.append(_row(
            chain_dt,
            name=chain.id,
            atom_idx=chain_atom_start,
            atom_num=chain_atom_count,
            res_idx=chain_res_start,
            res_num=len(residue_rows) - chain_res_start,
        ))

    atoms = np.array(atom_rows, dtype=atom_dt)
    residues = np.array(residue_rows, dtype=residue_dt)
    chains_arr = np.array(chain_rows, dtype=chain_dt)
    n_atoms = len(atoms)

    return bd.Structure(
        atoms=atoms,
        bonds=np.zeros(0, dtype=bd.Bond),
        residues=residues,
        chains=chains_arr,
        interfaces=np.zeros(0, dtype=bd.Interface),
        mask=np.ones(n_atoms, dtype=bool),
        coords=atoms["coords"].reshape(1, n_atoms, 3).astype(np.float32),
        ensemble=np.zeros(1, dtype=bd.Ensemble),
    )


# ── Tests ───────────────────────────────────────────────────────────────────

def test_upstream_structure_passes_validator():
    """A real boltzgen.data.data.Structure must satisfy our runtime layout check."""
    struct = _build_real_structure_from_pdb(PDB_1FYT, ["A", "B"])
    _validate_boltzgen_layout(struct)   # raises on any drift


def test_from_boltzgen_structure_on_real_structure_matches_from_pdb():
    """Run our function on a real upstream Structure and check SC matches the
    biopython pipeline on the same source atoms."""
    struct = _build_real_structure_from_pdb(PDB_1FYT, ["A", "B"])
    r_boltz = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"])
    r_pdb = from_pdb(PDB_1FYT, chains_a=["A"], chains_b=["B"])
    assert r_boltz.sc == pytest.approx(r_pdb.sc, abs=1e-6)
    assert r_boltz.atoms_a == r_pdb.atoms_a
    assert r_boltz.atoms_b == r_pdb.atoms_b


def test_our_required_fields_are_in_upstream_dtypes():
    """Cross-reference: every field name we read is present in the upstream
    dtype constants right now. If BoltzGen renames a field, this fires
    immediately."""
    upstream = {
        "chains":   set(np.dtype(bd.Chain).names),
        "residues": set(np.dtype(bd.Residue).names),
        "atoms":    set(np.dtype(bd.Atom).names),
    }
    for attr, required in _BOLTZGEN_REQUIRED_FIELDS.items():
        missing = set(required) - upstream[attr]
        assert not missing, (
            f"Drift detected: .{attr} required fields {missing} are no longer "
            f"in upstream boltzgen.data.data ({attr.title()}). Present upstream "
            f"fields: {sorted(upstream[attr])}."
        )
