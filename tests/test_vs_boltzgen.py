"""Validation against the canonical BoltzGen Structure layout.

`from_boltzgen_structure` is duck-typed against the numpy structured-array
layout of ``boltzgen.data.data.Structure``. This file pins our assumptions
about that layout against the upstream source by:

1. Copying the upstream Atom / Residue / Chain dtype lists verbatim (see
   `UPSTREAM_*` constants below — match the definitions at
   https://github.com/jwohlwend/boltzgen / boltzgen/data/data.py).
2. Building a Structure using those *full* upstream dtypes (every field,
   including ones our function never reads — bfactor, plddt, mol_type, etc.).
3. Running `from_boltzgen_structure` on it and comparing against `from_pdb`
   for the same underlying atoms.

If BoltzGen ever renames or drops a field we depend on, the upstream constants
below will fall out of sync with their source and the test should be updated
(intentionally) at that point. Until then, this test guards against silent
drift.

We don't `import boltzgen` here: the upstream package pulls torch + CUDA-only
dependencies that don't install cleanly on macOS or CI runners without GPUs.
The duck-typed design is what lets us validate without taking the dep.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from protein_interface import from_boltzgen_structure, from_pdb
from protein_interface.io import _is_hydrogen, _select_real_atom

DATA = Path(__file__).parent / "data"
PDB_1FYT = DATA / "1fyt.pdb"


# ── Upstream dtype layout (pinned to boltzgen 0.3.2 / data.py) ──────────────
# Source: boltzgen/data/data.py, "STRUCTURE" section. Keep this in sync with
# upstream when BoltzGen releases a schema change.

UPSTREAM_ATOM = [
    ("name", np.dtype("<U4")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("bfactor", np.dtype("f4")),
    ("plddt", np.dtype("f4")),
]

UPSTREAM_RESIDUE = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

UPSTREAM_CHAIN = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
    ("symmetric_group", np.dtype("i4")),
]


def _build_upstream_shaped_structure_from_pdb(pdb_path: Path, chains: list[str]):
    """Build a numpy-array bundle using the **full** upstream BoltzGen dtypes,
    populated from a biopython parse of a PDB file. Returns an object with
    .atoms / .residues / .chains."""
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    atom_rows, residue_rows, chain_rows = [], [], []
    global_atom_idx = 0
    global_res_idx = 0

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
                # bfactor and plddt set to placeholder values; not read by our function
                atom_rows.append((name, (float(c[0]), float(c[1]), float(c[2])),
                                  True, 0.0, 0.0))
                global_atom_idx += 1
                chain_atom_count += 1
                res_atom_count += 1
            residue_rows.append((
                residue.resname.strip(),
                0,                  # res_type (unused)
                global_res_idx,     # res_idx
                res_atom_start,     # atom_idx
                res_atom_count,     # atom_num
                0, 0, True, True,   # atom_center, atom_disto, is_standard, is_present
            ))
            global_res_idx += 1
        chain_rows.append((
            chain.id,
            1, 0, 0, 0,                                # mol_type, entity_id, sym_id, asym_id
            chain_atom_start, chain_atom_count,        # atom_idx, atom_num
            chain_res_start, len(residue_rows) - chain_res_start,  # res_idx, res_num
            0, 0,                                      # cyclic_period, symmetric_group
        ))

    class UpstreamShaped:
        atoms = np.array(atom_rows, dtype=UPSTREAM_ATOM)
        residues = np.array(residue_rows, dtype=UPSTREAM_RESIDUE)
        chains = np.array(chain_rows, dtype=UPSTREAM_CHAIN)

    return UpstreamShaped()


# ── Tests ───────────────────────────────────────────────────────────────────

def test_upstream_layout_passes_validator():
    """A Structure built with every upstream field must satisfy our runtime check."""
    from protein_interface.io import _validate_boltzgen_layout
    struct = _build_upstream_shaped_structure_from_pdb(PDB_1FYT, ["A", "B"])
    _validate_boltzgen_layout(struct)   # raises if anything is missing


def test_from_boltzgen_structure_on_upstream_layout_matches_from_pdb():
    """Run our function on a Structure with the canonical upstream dtypes
    (every field present) and verify it produces the same SC and atom counts
    as the biopython-PDB pipeline on the same source."""
    struct = _build_upstream_shaped_structure_from_pdb(PDB_1FYT, ["A", "B"])
    r_boltz = from_boltzgen_structure(struct, chains_a=["A"], chains_b=["B"])
    r_pdb = from_pdb(PDB_1FYT, chains_a=["A"], chains_b=["B"])
    assert r_boltz.sc == pytest.approx(r_pdb.sc, abs=1e-6)
    assert r_boltz.atoms_a == r_pdb.atoms_a
    assert r_boltz.atoms_b == r_pdb.atoms_b


def test_our_required_fields_are_subset_of_upstream():
    """Sanity guard: every field name we *require* is exposed by the upstream
    dtype lists. If upstream renames a field we use, this test catches it
    immediately (assuming someone has refreshed UPSTREAM_*)."""
    from protein_interface.io import _BOLTZGEN_REQUIRED_FIELDS
    upstream_fields = {
        "chains":   {n for n, _ in UPSTREAM_CHAIN},
        "residues": {n for n, _ in UPSTREAM_RESIDUE},
        "atoms":    {n for n, _ in UPSTREAM_ATOM},
    }
    for attr, required in _BOLTZGEN_REQUIRED_FIELDS.items():
        missing = set(required) - upstream_fields[attr]
        assert not missing, (
            f".{attr}: we depend on {missing}, but upstream pinned-layout does not "
            f"expose them. Either upstream renamed the field (update UPSTREAM_*) "
            f"or our duck-typing is broken."
        )
