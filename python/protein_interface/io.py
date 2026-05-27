"""PDB/CIF parsing for pysc.

Parsing logic mirrors src/bin/sc.rs in sc-rs exactly so that parity tests pass:
- ATOM records only (HETATM skipped unless include_hetatm=True)
- Alternate locations: keep ' ' and 'A', skip all others
- Hydrogens: excluded by default using the same heuristic as the CLI

BoltzGen integration (from_biotite, from_boltzgen_structure) is appended at the
bottom. BoltzGen `refold_cif/*.cif` files are ordinary mmCIF — load them with
`from_pdb()` (biopython) or `load_atoms()` for the analyze pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import math

import numpy as np

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure as BioStructure

from protein_interface._core import ScResult, compute_sc, unknown_sasa_radius_atoms


# ── Hydrogen detection (mirrors sc-rs bin/sc.rs) ────────────────────────────

def _is_hydrogen(atom_name: str, element: str) -> bool:
    """Mirror the hydrogen-detection logic in sc-rs bin/sc.rs."""
    elem = element.strip().upper()
    name = atom_name.strip()
    if elem == "H":
        return True
    if name.startswith("H"):
        return True
    if name.endswith("H"):
        return True
    # Catch names like "1H", "2HB" (digit-prefixed hydrogen names)
    if "H" in name and name[:1].isdigit():
        return True
    return False


# ── biopython helpers ────────────────────────────────────────────────────────

def _select_real_atom(atom):
    """Return a concrete (non-disordered) Atom for altloc ' ' or 'A'.

    Returns None if no acceptable altloc exists.
    """
    if atom.is_disordered():
        child_dict = atom.child_dict
        for altloc in ("A", " "):
            if altloc in child_dict:
                return child_dict[altloc]
        return None
    altloc = atom.altloc
    if altloc not in (" ", "A"):
        return None
    return atom


def _extract_atom_arrays(
    model,
    chains: list[str],
    include_hetatm: bool,
    include_hydrogens: bool,
) -> tuple[list[list[float]], list[str], list[str]]:
    coords: list[list[float]] = []
    atom_names: list[str] = []
    res_names: list[str] = []

    for chain in model.get_chains():
        if chain.id not in chains:
            continue
        for residue in chain.get_residues():
            # residue.id[0] == ' ' for standard ATOM records
            het = residue.id[0]
            if not include_hetatm and het != " ":
                continue
            for disordered_or_atom in residue.get_atoms():
                real = _select_real_atom(disordered_or_atom)
                if real is None:
                    continue
                atom_name = real.name.strip()
                element = (real.element or "").strip()
                if not include_hydrogens and _is_hydrogen(atom_name, element):
                    continue
                c = real.coord
                coords.append([float(c[0]), float(c[1]), float(c[2])])
                atom_names.append(atom_name)
                res_names.append(residue.resname.strip())

    return coords, atom_names, res_names


def _load_structure(path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def _validate_sc_arrays(coords, atom_names, residue_names, label: str, strict: bool) -> None:
    n = len(coords)
    if strict and n == 0:
        raise ValueError(f"{label} must contain at least one atom")
    if len(atom_names) != n or len(residue_names) != n:
        raise ValueError(f"{label} coords, atom_names, and residue_names must have the same length")
    for i, coord in enumerate(coords):
        if len(coord) != 3:
            raise ValueError(f"{label} coord {i} must have exactly 3 values")
        if not all(math.isfinite(float(x)) for x in coord):
            raise ValueError(f"{label} coord {i} contains a non-finite value")
    if strict:
        unknown = unknown_sasa_radius_atoms(atom_names, residue_names)
        if unknown:
            examples = ", ".join(f"{res}:{atom}@{idx}" for idx, res, atom in unknown[:5])
            extra = "" if len(unknown) <= 5 else f", ... ({len(unknown)} total)"
            raise ValueError(f"{label} contains atoms with no SC/SASA radius: {examples}{extra}")


# ── Public biopython-based API ───────────────────────────────────────────────

def from_structure(
    structure: "BioStructure",
    chains_a: list[str],
    chains_b: list[str] | None = None,
    model: int = 0,
    include_hetatm: bool = False,
    include_hydrogens: bool = False,
    parallel: bool = True,
    strict: bool = True,
) -> ScResult:
    """Compute SC from a biopython Structure object.

    Args:
        structure:          biopython Structure (any source)
        chains_a:           chain IDs for molecule A
        chains_b:           chain IDs for molecule B; None = all chains not in chains_a
        model:              model index (0-based)
        include_hetatm:     include HETATM residues (default False, matches sc-rs)
        include_hydrogens:  include hydrogen atoms (default False, matches sc-rs)
        parallel:           enable Rayon parallelism inside sc-rs
    """
    models = list(structure.get_models())
    if model >= len(models):
        raise ValueError(
            f"model index {model} out of range (structure has {len(models)} model(s))"
        )
    m = models[model]

    if chains_b is None:
        all_chain_ids = {ch.id for ch in m.get_chains()}
        chains_b = sorted(all_chain_ids - set(chains_a))

    coords_a, names_a, res_a = _extract_atom_arrays(m, chains_a, include_hetatm, include_hydrogens)
    coords_b, names_b, res_b = _extract_atom_arrays(m, chains_b, include_hetatm, include_hydrogens)

    _validate_sc_arrays(coords_a, names_a, res_a, "chains_a", strict)
    _validate_sc_arrays(coords_b, names_b, res_b, "chains_b", strict)
    return compute_sc(coords_a, names_a, res_a, coords_b, names_b, res_b, parallel)


def from_pdb(
    pdb_path: str | Path,
    chains_a: list[str],
    chains_b: list[str] | None = None,
    model: int = 0,
    include_hetatm: bool = False,
    include_hydrogens: bool = False,
    parallel: bool = True,
    strict: bool = True,
) -> ScResult:
    """Compute SC from a PDB or mmCIF file.

    Args:
        pdb_path:           path to .pdb, .ent, or .cif file
        chains_a:           chain IDs for molecule A
        chains_b:           chain IDs for molecule B; None = all chains not in chains_a
        model:              model index (0-based, default first model)
        include_hetatm:     include HETATM residues (default False)
        include_hydrogens:  include hydrogen atoms (default False)
        parallel:           enable Rayon parallelism inside sc-rs
    """
    structure = _load_structure(pdb_path)
    return from_structure(
        structure,
        chains_a,
        chains_b,
        model=model,
        include_hetatm=include_hetatm,
        include_hydrogens=include_hydrogens,
        parallel=parallel,
        strict=strict,
    )


# ── BoltzGen integration ─────────────────────────────────────────────────────
#
# BoltzGen pipeline output layout (relative to output_dir):
#
#   intermediate_designs/<id>.npz            – Structure NPZ, backbone only
#                                              (sidechains are [0,0,0])
#   intermediate_designs_inverse_folded/
#     <id>.npz                               – Structure NPZ, full all-atom
#     fold_out_npz/<id>.npz                  – raw fold tensors (coords + confidences)
#     refold_cif/<id>.cif                    – ← USE THIS for SC scoring
#     refold_design_cif/<id>.cif             – binder-only refold
#
# The refold_cif files are full-atom mmCIF with pLDDT in B-factors. They are
# the Boltz-validated structures and the right input for SC. Load them with
# from_pdb() (biopython) or load_atoms() — both already handle mmCIF.
#
# Chain naming in BoltzGen output: the binder chain is typically the last chain
# (e.g. "B" when the target is "A"), but verify from the CIF or the Record JSON
# rather than assuming.


def _extract_boltzgen_chains(
    structure,
    target_chains: list[str],
    include_hydrogens: bool,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Extract atom arrays from a BoltzGen Structure object for the given chains.

    Iterates chains → residues → atoms using the absolute-index layout of
    Structure.chains / .residues / .atoms (all indices are into the global arrays).
    Skips atoms where is_present is False.
    """
    all_chain_names = [str(n) for n in structure.chains["name"]]
    target_set = set(target_chains)

    coords: list[list[float]] = []
    atom_names: list[str] = []
    res_names: list[str] = []

    for ci, chain in enumerate(structure.chains):
        if str(chain["name"]) not in target_set:
            continue

        res_start = int(chain["res_idx"])
        res_count = int(chain["res_num"])

        for ri in range(res_start, res_start + res_count):
            res = structure.residues[ri]
            res_name = str(res["name"])
            a_start = int(res["atom_idx"])
            a_count = int(res["atom_num"])

            for ai in range(a_start, a_start + a_count):
                atom = structure.atoms[ai]
                if not bool(atom["is_present"]):
                    continue
                atom_name = str(atom["name"])
                if not include_hydrogens and _is_hydrogen(atom_name, ""):
                    continue
                c = atom["coords"]
                coords.append([float(c[0]), float(c[1]), float(c[2])])
                atom_names.append(atom_name)
                res_names.append(res_name)

    missing = target_set - set(all_chain_names)
    if missing:
        raise ValueError(
            f"Chain(s) {sorted(missing)} not found in Structure. "
            f"Available: {list(dict.fromkeys(all_chain_names))}"
        )

    return coords, atom_names, res_names


def from_boltzgen_structure(
    structure,
    chains_a: list[str],
    chains_b: list[str] | None = None,
    include_hydrogens: bool = False,
    parallel: bool = True,
    strict: bool = True,
) -> ScResult:
    """Compute SC from an in-memory BoltzGen Structure object.

    Accepts any object with .atoms / .residues / .chains numpy structured arrays
    matching the BoltzGen dtype layout (boltzgen.data.data.Structure).

    Args:
        structure:          BoltzGen Structure (or duck-typed equivalent)
        chains_a:           chain IDs for molecule A (e.g. ["B"] for binder)
        chains_b:           chain IDs for molecule B; None = all chains not in chains_a
        include_hydrogens:  include hydrogen atoms (default False)
        parallel:           enable Rayon parallelism inside sc-rs

    Important: use post-refold structures for meaningful SC scores. Structures
    from intermediate_designs/ have zeroed sidechain coordinates and will give
    unreliable results. For refold_cif/*.cif files on disk, use from_pdb() or
    load_atoms() directly (both handle mmCIF via biopython).
    """
    all_chain_names = [str(n) for n in structure.chains["name"]]

    if chains_b is None:
        chains_a_set = set(chains_a)
        chains_b = list(dict.fromkeys(
            n for n in all_chain_names if n not in chains_a_set
        ))

    coords_a, names_a, res_a = _extract_boltzgen_chains(structure, chains_a, include_hydrogens)
    coords_b, names_b, res_b = _extract_boltzgen_chains(structure, chains_b, include_hydrogens)

    _validate_sc_arrays(coords_a, names_a, res_a, "chains_a", strict)
    _validate_sc_arrays(coords_b, names_b, res_b, "chains_b", strict)
    return compute_sc(coords_a, names_a, res_a, coords_b, names_b, res_b, parallel)


def from_biotite(
    atom_array,
    chains_a: list[str],
    chains_b: list[str] | None = None,
    include_hetatm: bool = False,
    include_hydrogens: bool = False,
    parallel: bool = True,
    strict: bool = True,
) -> ScResult:
    """Compute SC from a biotite AtomArray or AtomArrayStack (first model used).

    BoltzGen's analysis stack (analyze_utils.py) works with biotite AtomArrays.
    This function is the natural bridge when you already have an AtomArray loaded.

    Example — scoring a BoltzGen refold CIF with biotite directly:

        import biotite.structure.io.pdbx as pdbx
        cif = pdbx.CIFFile.read("refold_cif/design_0.cif")
        atoms = pdbx.get_structure(cif, model=1, use_author_fields=False)
        result = pysc.from_biotite(atoms, chains_a=["B"], chains_b=["A"])

    Args:
        atom_array:         biotite AtomArray (or first-model slice of AtomArrayStack)
        chains_a:           chain IDs for molecule A
        chains_b:           chain IDs for molecule B; None = all chains not in chains_a
        include_hetatm:     include hetero atoms (default False)
        include_hydrogens:  include hydrogen atoms (default False)
        parallel:           enable Rayon parallelism inside sc-rs
    """
    # Support AtomArrayStack by taking the first model
    try:
        import biotite.structure as struc
        if isinstance(atom_array, struc.AtomArrayStack):
            atom_array = atom_array[0]
    except ImportError:
        pass  # duck-typing fallback: assume it already behaves like an AtomArray

    all_chains = list(dict.fromkeys(str(c) for c in atom_array.chain_id))
    if chains_b is None:
        chains_a_set = set(chains_a)
        chains_b = [c for c in all_chains if c not in chains_a_set]

    def _extract(chain_ids: list[str]) -> tuple[list, list, list]:
        mask = np.zeros(len(atom_array), dtype=bool)
        for ch in chain_ids:
            mask |= atom_array.chain_id == ch
        if not include_hetatm:
            mask &= ~atom_array.hetero
        sub = atom_array[mask]

        coords: list[list[float]] = []
        anames: list[str] = []
        rnames: list[str] = []
        for i in range(len(sub)):
            name = str(sub.atom_name[i])
            elem = str(sub.element[i]) if hasattr(sub, "element") else ""
            if not include_hydrogens and _is_hydrogen(name, elem):
                continue
            c = sub.coord[i]
            coords.append([float(c[0]), float(c[1]), float(c[2])])
            anames.append(name)
            rnames.append(str(sub.res_name[i]))
        return coords, anames, rnames

    coords_a, names_a, res_a = _extract(chains_a)
    coords_b, names_b, res_b = _extract(chains_b)
    _validate_sc_arrays(coords_a, names_a, res_a, "chains_a", strict)
    _validate_sc_arrays(coords_b, names_b, res_b, "chains_b", strict)
    return compute_sc(coords_a, names_a, res_a, coords_b, names_b, res_b, parallel)


