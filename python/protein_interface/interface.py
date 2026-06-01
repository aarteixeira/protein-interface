"""Interface analysis metrics built on top of the Rust SASA / H-bond / salt-bridge kernels.

All functions operate on flat atom arrays (parallel lists). Use load_atoms() to
extract arrays from a PDB/CIF file via biopython, or supply your own arrays.

Individual metrics:
    sasa(...)                    — per-atom SASA (Rust)
    delta_sasa(...)              — buried surface area on binding (Å²)
    interface_residues(...)      — distance-based residue mask per side
    n_interface_residues(...)    — counts per side
    aromatic_dsasa_fraction(...) — fraction of dSASA from F/Y/W/H
    hbonds(...)                  — cross-interface H-bond count (Rust)
    salt_bridges(...)            — cross-interface salt-bridge count (Rust)
    disulfide_bridges(...)       — cross-interface Cys SG–SG pairs
    atomic_contacts(...)         — heavy-atom pairs across the interface
    asymmetry(...)               — |dSASA_A − dSASA_B| / max(dSASA_A, dSASA_B)
    interface_depth(...)         — distance between A-side and B-side interface centroids
    gly_pro_fraction(...)        — Gly + Pro fraction of interface residues
    confidence_at_interface(...) — mean / min B-factor (= pLDDT for AF/Boltz) at interface atoms
    bb_sc_dsasa_split(...)       — backbone vs side-chain dSASA decomposition
    charge_complementarity(...)  — formal-charge match across the interface
    bsa_breakdown(...)           — BHSA / BPSA / BCSA split of dSASA
    per_residue_dsasa(...)       — buried area per residue
    hotspot_residues(...)        — residues with dSASA above threshold
    hbond_density(...)           — H-bonds normalised per 100 Å² of dSASA
    pi_pi_contacts(...)          — aromatic ring stacking pairs
    cation_pi_contacts(...)      — cation-π pairs
    buried_unsat_polar(...)      — buried polar atoms with no cross-interface partner
    interface_shape(...)         — planarity RMSD + PCA eigenvalue ratios

Combined:
    analyze(...)                 — runs everything; returns InterfaceResult
    analyze_batch(...)           — same, but processes many complexes via one
                                   batched Rust SASA call (Rayon-parallel)
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from protein_interface._core import (
    compute_sasa,
    compute_sasa_batch,
    compute_sc,
    compute_sc_batch,
    count_hbonds,
    unknown_sasa_radius_atoms,
)
from protein_interface.io import _is_hydrogen, _load_structure, _select_real_atom

AROMATIC = frozenset({"PHE", "TYR", "TRP", "HIS"})

# Heavy-atom backbone names. Everything else (CB onward, including CYS SG) is
# treated as side-chain. Hydrogens are excluded by default upstream.
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

# Formal charges at physiological pH for the standard residues. His is treated
# as neutral (mixed protonation); termini are not assigned (we operate on
# interface residues only, not chain ends specifically).
RESIDUE_CHARGE: dict[str, int] = {
    "ASP": -1, "GLU": -1,
    "LYS": +1, "ARG": +1,
}

# PRODIGY residue classification (Vangone & Bonvin 2015). Mirrors the two
# tables in prodigy_prot.modules.aa_properties exactly — they are different
# for IC counting vs NIS counting. CYS, TYR, TRP move between "apolar" (IC
# table) and "polar" (NIS table); HIS moves between "charged" (IC) and
# "polar" (NIS).
PRODIGY_IC_CLASS: dict[str, str] = {
    "ALA": "A", "CYS": "A", "GLU": "C", "ASP": "C", "GLY": "A",
    "PHE": "A", "ILE": "A", "HIS": "C", "LYS": "C", "MET": "A",
    "LEU": "A", "ASN": "P", "GLN": "P", "PRO": "A", "SER": "P",
    "ARG": "C", "THR": "P", "TRP": "A", "VAL": "A", "TYR": "A",
}
PRODIGY_NIS_CLASS: dict[str, str] = {
    "ALA": "A", "CYS": "P", "GLU": "C", "ASP": "C", "GLY": "A",
    "PHE": "A", "ILE": "A", "HIS": "P", "LYS": "C", "MET": "A",
    "LEU": "A", "ASN": "P", "GLN": "P", "PRO": "A", "SER": "P",
    "ARG": "C", "THR": "P", "TRP": "P", "VAL": "A", "TYR": "P",
}

# Reference SASA (Å²) per residue type in a Gly-X-Gly tripeptide — used for
# relative-SASA (rSASA) computation. Values from Tien et al. 2013 ("empirical"
# column), which is the modern alternative to Miller 1987 / NACCESS defaults
# and is what PRODIGY ships in recent versions.
REFERENCE_SASA: dict[str, float] = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}

# Trained coefficients of the PRODIGY ΔG_bind predictor (Vangone & Bonvin 2015,
# eLife 4:e07454). Counts are residue-residue intermolecular contacts within
# 5.5 Å (any heavy atom). NIS percentages are over surface residues (rel-SASA
# > 5 %) outside the interface in the bound complex.
PRODIGY_COEFFS = {
    "CC":          -0.09459,   # charged-charged ICs
    "AC":          -0.10007,   # apolar-charged ICs   (called "CA" upstream)
    "PP":           0.19577,   # polar-polar ICs
    "AP":          -0.22671,   # apolar-polar ICs     (called "PA" upstream)
    "NIS_apolar":   0.18681,   # % apolar in NIS
    "NIS_charged":  0.13810,   # % charged in NIS
    "intercept":  -15.9433,
}
PRODIGY_IC_CUTOFF = 5.5         # Å
PRODIGY_NIS_RSASA_CUTOFF = 5.0  # % rel-SASA threshold for "surface"


def _prodigy_ic_class(rname: str) -> str | None:
    """IC-table polarity: 'A' / 'P' / 'C', or None for non-standard residues."""
    return PRODIGY_IC_CLASS.get(rname)


def _prodigy_nis_class(rname: str) -> str | None:
    """NIS-table polarity: 'A' / 'P' / 'C', or None for non-standard residues."""
    return PRODIGY_NIS_CLASS.get(rname)

# Six-membered ring atoms used for the centroid + normal of each aromatic residue.
# TRP uses the six-membered benzene ring of the indole; HIS uses its five-ring.
RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
}

ANION_ATOMS: frozenset[tuple[str, str]] = frozenset({
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
})
CATION_ATOMS: frozenset[tuple[str, str]] = frozenset({
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
})

# Side-chain atoms carrying a formal charge at physiological pH.
CHARGED_ATOMS: frozenset[tuple[str, str]] = ANION_ATOMS | CATION_ATOMS

# Donor and acceptor classification mirrors src/hbonds.rs so per-atom satisfaction
# checks stay consistent with the cross-interface H-bond counter.
DONOR_SET: frozenset[tuple[str, str]] = frozenset({
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
    ("ASN", "ND2"), ("GLN", "NE2"),
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("TRP", "NE1"),
})
ACCEPTOR_SET: frozenset[tuple[str, str]] = frozenset({
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
    ("HIS", "ND1"), ("HIS", "NE2"),
})


def _is_donor(residue: str, atom: str) -> bool:
    if atom == "N" and residue != "PRO":
        return True
    return (residue, atom) in DONOR_SET


def _is_acceptor(residue: str, atom: str) -> bool:
    if atom in ("O", "OXT"):
        return True
    return (residue, atom) in ACCEPTOR_SET


@dataclass
class AtomArrays:
    coords: list[list[float]]
    atom_names: list[str]
    residue_names: list[str]
    residue_ids: list[tuple]  # (chain_id, resseq, icode) per atom; legacy 2-tuples accepted
    bfactors: list[float] | None = None  # per-atom B-factor (= pLDDT for AF2/3, BoltzGen)
    _allow_side_qualified: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.residue_ids = [
            _normalize_residue_id(rid, allow_side_qualified=self._allow_side_qualified)
            for rid in self.residue_ids
        ]


def _normalize_residue_id(rid: tuple, *, allow_side_qualified: bool = False) -> tuple:
    """Normalize public residue IDs; side-qualified keys are internal only."""
    if len(rid) == 2:
        chain, resseq = rid
        return (str(chain), int(resseq), "")
    if len(rid) == 3:
        chain, resseq, icode = rid
        return (str(chain), int(resseq), str(icode).strip())
    if len(rid) == 4 and rid[0] in ("a", "b"):
        if not allow_side_qualified:
            raise ValueError(
                "side-qualified residue IDs are internal-only; use "
                "(chain, resseq) or (chain, resseq, icode)"
            )
        side, chain, resseq, icode = rid
        return (side, str(chain), int(resseq), str(icode).strip())
    raise ValueError(
        "residue IDs must be (chain, resseq) or (chain, resseq, icode)"
    )


def _side_residue_id(side: str, rid: tuple) -> tuple[str, str, int, str]:
    chain, resseq, icode = _normalize_residue_id(rid)
    return (side, chain, resseq, icode)


def _strip_side_residue_id(rid: tuple) -> tuple[str, int, str]:
    if len(rid) == 4 and rid[0] in ("a", "b"):
        _, chain, resseq, icode = rid
        return (chain, resseq, icode)
    chain, resseq, icode = _normalize_residue_id(rid)
    return (chain, resseq, icode)


def _validate_atom_arrays(atoms: AtomArrays, label: str, *, require_nonempty: bool) -> None:
    n = len(atoms.coords)
    if require_nonempty and n == 0:
        raise ValueError(f"{label} must contain at least one atom")
    if len(atoms.atom_names) != n or len(atoms.residue_names) != n or len(atoms.residue_ids) != n:
        raise ValueError(
            f"{label} coords, atom_names, residue_names, and residue_ids must have the same length"
        )
    if atoms.bfactors is not None and len(atoms.bfactors) != n:
        raise ValueError(f"{label} bfactors must have the same length as coords")
    for i, coord in enumerate(atoms.coords):
        if len(coord) != 3:
            raise ValueError(f"{label} coord {i} must have exactly 3 values")
        if not all(math.isfinite(float(x)) for x in coord):
            raise ValueError(f"{label} coord {i} contains a non-finite value")


def _validate_known_sasa_radii(atoms: AtomArrays, label: str) -> None:
    unknown = unknown_sasa_radius_atoms(atoms.atom_names, atoms.residue_names)
    if unknown:
        examples = ", ".join(f"{res}:{atom}@{idx}" for idx, res, atom in unknown[:5])
        extra = "" if len(unknown) <= 5 else f", ... ({len(unknown)} total)"
        raise ValueError(f"{label} contains atoms with no SASA radius: {examples}{extra}")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_n_points(n_points: int) -> None:
    if isinstance(n_points, bool) or not isinstance(n_points, int):
        raise ValueError("n_points must be an integer >= 4")
    if n_points < 4:
        raise ValueError("n_points must be an integer >= 4")


def _validate_sasa_params(probe_radius: float, n_points: int) -> None:
    _validate_nonnegative_finite("probe_radius", probe_radius)
    _validate_n_points(n_points)


def _validate_for_sasa(a: AtomArrays, b: AtomArrays | None = None, *, strict: bool) -> None:
    _validate_atom_arrays(a, "atoms_a", require_nonempty=strict)
    if strict:
        _validate_known_sasa_radii(a, "atoms_a")
    if b is not None:
        _validate_atom_arrays(b, "atoms_b", require_nonempty=strict)
        if strict:
            _validate_known_sasa_radii(b, "atoms_b")


@dataclass
class InterfaceResult:
    sc: float | None                         # Lawrence-Colman shape complementarity (-1 to 1); NaN if failed permissively
    dsasa: float | None
    n_interface_a: int | None
    n_interface_b: int | None
    aromatic_dsasa_fraction: float | None
    hbonds: int | None
    salt_bridges: int | None
    bhsa: float | None                       # buried hydrophobic surface area (Å²)
    bpsa: float | None                       # buried polar surface area (Å²)
    bcsa: float | None                       # buried charged surface area (Å²)
    hydrophobic_fraction: float | None       # bhsa / dsasa
    hbond_density: float | None              # hbonds per 100 Å² of dsasa
    pi_pi: int | None
    cation_pi: int | None
    buried_unsat_polar: int | None
    planarity_rmsd: float | None             # Å, RMS distance to best-fit plane
    elongation: float | None                 # σ1 / σ2 (≥ 1; large = elongated)
    planarity_ratio: float | None            # σ3 / σ2 (≤ 1; small = flat)
    dsasa_a: float | None                    # buried area on side A only (Å²)
    dsasa_b: float | None                    # buried area on side B only (Å²)
    asymmetry: float | None                  # |dsasa_a − dsasa_b| / max(dsasa_a, dsasa_b); 0 = symmetric
    atomic_contacts: int | None              # cross-interface heavy-atom pairs within 5 Å
    interface_depth: float | None            # Å, between A-side and B-side interface centroids
    disulfides: int | None                   # cross-interface Cys SG–SG pairs
    gly_pro_fraction: float | None           # (Gly + Pro) / total interface residues
    bb_dsasa: float | None                   # backbone buried surface area (Å²)
    sc_dsasa: float | None                   # side-chain buried surface area (Å²)
    sidechain_fraction: float | None         # sc_dsasa / dsasa
    charge_a: int | None                     # net formal charge over interface residues, side A
    charge_b: int | None                     # net formal charge over interface residues, side B
    charge_complementarity: float | None     # −charge_a × charge_b; positive = opposing signs
    mean_bfactor_interface: float | None     # mean B-factor (= pLDDT) over interface atoms; NaN if absent
    min_bfactor_interface: float | None      # min B-factor over interface atoms; NaN if absent
    prodigy_dg: float | None                 # predicted binding ΔG (kcal/mol), Vangone & Bonvin 2015
    hotspots_a: list[tuple[str, int, str]] | None = field(default_factory=list)
    hotspots_b: list[tuple[str, int, str]] | None = field(default_factory=list)


ANALYZE_METRICS = frozenset(InterfaceResult.__dataclass_fields__)
_SASA_ANALYZE_METRICS = frozenset({
    "dsasa", "dsasa_a", "dsasa_b", "asymmetry",
    "aromatic_dsasa_fraction", "bhsa", "bpsa", "bcsa", "hydrophobic_fraction",
    "bb_dsasa", "sc_dsasa", "sidechain_fraction",
    "hbond_density", "buried_unsat_polar",
    "planarity_rmsd", "elongation", "planarity_ratio", "interface_depth",
    "mean_bfactor_interface", "min_bfactor_interface",
    "hotspots_a", "hotspots_b", "prodigy_dg",
})


def _resolve_analyze_metrics(
    metrics: Iterable[str] | None,
    skip_metrics: Iterable[str] | None,
) -> frozenset[str]:
    selected = set(ANALYZE_METRICS if metrics is None else metrics)
    skipped = set(() if skip_metrics is None else skip_metrics)
    unknown = (selected | skipped) - ANALYZE_METRICS
    if unknown:
        names = ", ".join(sorted(unknown))
        valid = ", ".join(sorted(ANALYZE_METRICS))
        raise ValueError(f"unknown analyze metric(s): {names}; valid metrics are: {valid}")
    return frozenset(selected - skipped)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_atoms(
    pdb_path: str | Path,
    chains: list[str],
    model: int = 0,
    include_hetatm: bool = False,
    include_hydrogens: bool = False,
) -> AtomArrays:
    """Extract atom arrays (including per-atom residue IDs) from a PDB/CIF file."""
    structure = _load_structure(pdb_path)
    models = list(structure.get_models())
    if model >= len(models):
        raise ValueError(f"model index {model} out of range ({len(models)} model(s))")
    m = models[model]

    coords: list[list[float]] = []
    atom_names: list[str] = []
    res_names: list[str] = []
    res_ids: list[tuple[str, int, str]] = []
    bfactors: list[float] = []
    chains_set = set(chains)
    for chain in m.get_chains():
        if chain.id not in chains_set:
            continue
        for residue in chain.get_residues():
            if not include_hetatm and residue.id[0] != " ":
                continue
            rid = (chain.id, residue.id[1], residue.id[2].strip())
            for disordered_or_atom in residue.get_atoms():
                real = _select_real_atom(disordered_or_atom)
                if real is None:
                    continue
                name = real.name.strip()
                elem = (real.element or "").strip()
                if not include_hydrogens and _is_hydrogen(name, elem):
                    continue
                c = real.coord
                coords.append([float(c[0]), float(c[1]), float(c[2])])
                atom_names.append(name)
                res_names.append(residue.resname.strip())
                res_ids.append(rid)
                bfactors.append(float(real.bfactor) if real.bfactor is not None else 0.0)
    return AtomArrays(coords, atom_names, res_names, res_ids, bfactors)


# ── Individual metrics ───────────────────────────────────────────────────────

def sasa(
    atoms: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> list[float]:
    """Per-atom SASA in Å². Thin wrapper around the Rust kernel."""
    _validate_sasa_params(probe_radius, n_points)
    _validate_for_sasa(atoms, strict=strict)
    return compute_sasa(atoms.coords, atoms.atom_names, atoms.residue_names, probe_radius, n_points)


def _combined(a: AtomArrays, b: AtomArrays) -> AtomArrays:
    return AtomArrays(
        a.coords + b.coords,
        a.atom_names + b.atom_names,
        a.residue_names + b.residue_names,
        [_side_residue_id("a", rid) for rid in a.residue_ids]
        + [_side_residue_id("b", rid) for rid in b.residue_ids],
        _allow_side_qualified=True,
    )


def delta_sasa(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> float:
    """Buried surface area on binding: SASA(A alone) + SASA(B alone) − SASA(A+B)."""
    _validate_sasa_params(probe_radius, n_points)
    _validate_for_sasa(a, b, strict=strict)
    sasa_a = sasa(a, probe_radius, n_points, strict=strict)
    sasa_b = sasa(b, probe_radius, n_points, strict=strict)
    sasa_ab = compute_sasa(
        a.coords + b.coords,
        a.atom_names + b.atom_names,
        a.residue_names + b.residue_names,
        probe_radius,
        n_points,
    )
    return sum(sasa_a) + sum(sasa_b) - sum(sasa_ab)


def interface_residues(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = 5.0,
) -> tuple[set[tuple[str, int, str]], set[tuple[str, int, str]]]:
    """Distance-based interface residue identification.

    A residue is at the interface if any of its heavy atoms is within `cutoff` Å
    of any heavy atom on the other side. Uses a numpy distance matrix
    (memory ≈ 8·N_a·N_b bytes; ~18 MB for a 1500×1500 complex).
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    if not a.coords or not b.coords:
        return set(), set()
    A = np.asarray(a.coords)
    B = np.asarray(b.coords)
    cutoff2 = cutoff * cutoff
    # Squared distance via ||a||² + ||b||² − 2 a·bᵀ — avoids a large 3-D
    # intermediate and uses BLAS for the matmul.
    sq_a = (A * A).sum(axis=1)
    sq_b = (B * B).sum(axis=1)
    d2 = sq_a[:, None] + sq_b[None, :] - 2.0 * (A @ B.T)
    within = d2 <= cutoff2
    mask_a = within.any(axis=1)
    mask_b = within.any(axis=0)
    int_a = {a.residue_ids[i] for i in np.flatnonzero(mask_a)}
    int_b = {b.residue_ids[j] for j in np.flatnonzero(mask_b)}
    return int_a, int_b


def n_interface_residues(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = 5.0,
) -> tuple[int, int]:
    int_a, int_b = interface_residues(a, b, cutoff)
    return len(int_a), len(int_b)


def aromatic_dsasa_fraction(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> float:
    """Fraction of buried surface area contributed by aromatic residues (F/Y/W/H)."""
    _validate_sasa_params(probe_radius, n_points)
    _validate_for_sasa(a, b, strict=strict)
    sasa_a = sasa(a, probe_radius, n_points, strict=strict)
    sasa_b = sasa(b, probe_radius, n_points, strict=strict)
    combined = _combined(a, b)
    sasa_ab = compute_sasa(combined.coords, combined.atom_names, combined.residue_names, probe_radius, n_points)

    sasa_sep = sasa_a + sasa_b
    res_all = combined.residue_names

    total = 0.0
    arom = 0.0
    for i in range(len(res_all)):
        d = sasa_sep[i] - sasa_ab[i]
        total += d
        if res_all[i] in AROMATIC:
            arom += d
    return arom / total if total > 0 else 0.0


def hbonds(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = 3.5,
) -> int:
    """Cross-interface H-bond count (distance-only criterion). Thin wrapper around the Rust kernel."""
    _validate_nonnegative_finite("cutoff", cutoff)
    return count_hbonds(
        a.coords, a.atom_names, a.residue_names,
        b.coords, b.atom_names, b.residue_names,
        cutoff,
    )


# ── New individual metrics ───────────────────────────────────────────────────

def salt_bridges(a: AtomArrays, b: AtomArrays, cutoff: float = 4.0) -> int:
    """Cross-interface salt-bridge count as acidic/basic residue pairs.

    A residue pair is counted once when any acidic side-chain oxygen
    (Asp/Glu) and any basic side-chain nitrogen (Lys/Arg/His) across the
    interface lie within `cutoff` Å.
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    pairs: set[tuple[tuple, tuple]] = set()
    cutoff2 = cutoff * cutoff
    for i, ca in enumerate(a.coords):
        a_anion = (a.residue_names[i], a.atom_names[i]) in ANION_ATOMS
        a_cation = (a.residue_names[i], a.atom_names[i]) in CATION_ATOMS
        if not a_anion and not a_cation:
            continue
        x = np.asarray(ca)
        for j, cb in enumerate(b.coords):
            b_anion = (b.residue_names[j], b.atom_names[j]) in ANION_ATOMS
            b_cation = (b.residue_names[j], b.atom_names[j]) in CATION_ATOMS
            if not ((a_anion and b_cation) or (a_cation and b_anion)):
                continue
            y = np.asarray(cb)
            if float((x - y) @ (x - y)) <= cutoff2:
                pairs.add((a.residue_ids[i], b.residue_ids[j]))
    return len(pairs)


def atomic_contacts(a: AtomArrays, b: AtomArrays, cutoff: float = 5.0) -> int:
    """Cross-interface heavy-atom pairs within `cutoff` Å.

    A density proxy for the interface — independent of SASA and chemistry.
    Uses a numpy broadcast (O(N·M)); fine for typical complex sizes.
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    A = np.asarray(a.coords)
    B = np.asarray(b.coords)
    if A.size == 0 or B.size == 0:
        return 0
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return int((d2 <= cutoff * cutoff).sum())


def disulfide_bridges(a: AtomArrays, b: AtomArrays, cutoff: float = 2.5) -> int:
    """Cross-interface Cys SG–SG pairs within `cutoff` Å.

    Standard disulfide S–S bond length is ~2.05 Å; the 2.5 Å cutoff accepts
    a bit of distortion. Disulfide bridges across protein-protein interfaces
    are rare outside engineered systems, but useful to count when present.
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    sg_a = [np.asarray(a.coords[i]) for i in range(len(a.coords))
            if a.residue_names[i] == "CYS" and a.atom_names[i] == "SG"]
    sg_b = [np.asarray(b.coords[i]) for i in range(len(b.coords))
            if b.residue_names[i] == "CYS" and b.atom_names[i] == "SG"]
    cutoff2 = cutoff * cutoff
    return sum(1 for x in sg_a for y in sg_b if float((x - y) @ (x - y)) <= cutoff2)


def asymmetry(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
) -> tuple[float, float, float]:
    """Per-side buried surface area and a symmetric asymmetry index.

    Returns (dsasa_a, dsasa_b, asymmetry) where:
        asymmetry = |dsasa_a − dsasa_b| / max(dsasa_a, dsasa_b)
        0 = perfectly symmetric, 1 = totally one-sided.
    """
    _validate_sasa_params(probe_radius, n_points)
    sasa_a = sasa(a, probe_radius, n_points)
    sasa_b = sasa(b, probe_radius, n_points)
    combined = _combined(a, b)
    sasa_ab = compute_sasa(
        combined.coords, combined.atom_names, combined.residue_names, probe_radius, n_points
    )
    na = len(a.coords)
    d_a = sum(sasa_a) - sum(sasa_ab[:na])
    d_b = sum(sasa_b) - sum(sasa_ab[na:])
    m = max(d_a, d_b)
    asym = abs(d_a - d_b) / m if m > 0 else 0.0
    return d_a, d_b, asym


def interface_depth(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    min_atom_dsasa: float = 0.5,
    strict: bool = True,
) -> float:
    """Distance between A-side and B-side interface centroids (Å).

    The centroid on each side averages the coordinates of atoms whose per-atom
    dSASA ≥ `min_atom_dsasa`. Roughly captures how 'deep' one chain protrudes
    into the other; small values for shallow planar interfaces, larger values
    for concave/cradled binding modes.

    Returns NaN if either side has no qualifying atom.
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("min_atom_dsasa", min_atom_dsasa)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    na = len(a.coords)
    mask_a = d[:na] >= min_atom_dsasa
    mask_b = d[na:] >= min_atom_dsasa
    if not mask_a.any() or not mask_b.any():
        return float("nan")
    coords_a = np.asarray(a.coords)[mask_a].mean(axis=0)
    coords_b = np.asarray(b.coords)[mask_b].mean(axis=0)
    return float(np.linalg.norm(coords_a - coords_b))


def gly_pro_fraction(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = 5.0,
) -> float:
    """Fraction of interface residues that are Gly or Pro.

    Interface residues defined by the 5 Å heavy-atom contact rule (matches
    `interface_residues`). High values indicate a flexible/kinked interface
    (Gly removes a side chain; Pro restricts backbone φ).
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    int_a, int_b = interface_residues(a, b, cutoff)
    res_name_a = {a.residue_ids[i]: a.residue_names[i] for i in range(len(a.coords))}
    res_name_b = {b.residue_ids[i]: b.residue_names[i] for i in range(len(b.coords))}
    total = len(int_a) + len(int_b)
    if total == 0:
        return 0.0
    gp = sum(1 for r in int_a if res_name_a.get(r) in ("GLY", "PRO"))
    gp += sum(1 for r in int_b if res_name_b.get(r) in ("GLY", "PRO"))
    return gp / total


def confidence_at_interface(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    min_atom_dsasa: float = 0.5,
    strict: bool = True,
) -> dict[str, float]:
    """Mean and minimum B-factor over atoms that buried surface on binding.

    For AlphaFold / Boltz / BoltzGen outputs the B-factor column stores pLDDT
    (0–100; higher = more confident). For X-ray structures it stores thermal
    B-factors in Å². The interpretation is the caller's responsibility.

    Returns a dict with keys 'mean' and 'min'. Both are NaN if either atom
    array lacks `bfactors` or no atom is sufficiently buried.
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("min_atom_dsasa", min_atom_dsasa)
    nan = {"mean": float("nan"), "min": float("nan")}
    if a.bfactors is None or b.bfactors is None:
        return nan
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    na = len(a.coords)
    mask_a = d[:na] >= min_atom_dsasa
    mask_b = d[na:] >= min_atom_dsasa
    bf = np.concatenate([
        np.asarray(a.bfactors)[mask_a],
        np.asarray(b.bfactors)[mask_b],
    ])
    if bf.size == 0:
        return nan
    return {"mean": float(bf.mean()), "min": float(bf.min())}


def bb_sc_dsasa_split(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> dict[str, float]:
    """Split dSASA into backbone and side-chain contributions.

    Backbone atoms: N, CA, C, O, OXT. Returns a dict with keys 'bb_dsasa',
    'sc_dsasa', 'sidechain_fraction'.
    """
    _validate_sasa_params(probe_radius, n_points)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    bb = 0.0
    sc = 0.0
    for i, name in enumerate(combined.atom_names):
        if name in BACKBONE_ATOMS:
            bb += float(d[i])
        else:
            sc += float(d[i])
    total = bb + sc
    return {
        "bb_dsasa": bb,
        "sc_dsasa": sc,
        "sidechain_fraction": (sc / total) if total > 0 else 0.0,
    }


def charge_complementarity(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = 5.0,
) -> dict[str, float]:
    """Net formal charge over interface residues and complementarity score.

    Interface residues are identified by the 5 Å heavy-atom contact rule.
    Formal charges (physiological pH): Asp/Glu = −1, Lys/Arg = +1; His is
    treated as neutral. Returns:

        charge_a, charge_b   net charge on each side
        complementarity      −(charge_a × charge_b); positive = opposing signs,
                             negative = same sign (electrostatically clashing),
                             zero = at least one side is uncharged.
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    int_a, int_b = interface_residues(a, b, cutoff)
    res_name_a = {a.residue_ids[i]: a.residue_names[i] for i in range(len(a.coords))}
    res_name_b = {b.residue_ids[i]: b.residue_names[i] for i in range(len(b.coords))}
    qa = sum(RESIDUE_CHARGE.get(res_name_a.get(r, ""), 0) for r in int_a)
    qb = sum(RESIDUE_CHARGE.get(res_name_b.get(r, ""), 0) for r in int_b)
    return {
        "charge_a": qa,
        "charge_b": qb,
        "complementarity": float(-qa * qb),
    }


# ── PRODIGY ──────────────────────────────────────────────────────────────────

def _ic_bins_from_contacts(
    a: AtomArrays,
    b: AtomArrays,
    within: np.ndarray,
) -> dict[str, int]:
    """Tally cross-interface residue-residue contacts by polarity-pair class.

    `within` is the (N_a, N_b) bool atom-pair mask (typically `d² ≤ 5.5²`).
    A residue pair counts once even if many of its atoms contact.
    """
    pairs = np.argwhere(within)
    if pairs.size == 0:
        return {"CC": 0, "AC": 0, "AP": 0, "AA": 0, "PP": 0, "CP": 0}

    res_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for ai, bi in pairs:
        res_pairs.add((a.residue_ids[ai], b.residue_ids[bi]))

    res_name_a = {a.residue_ids[i]: a.residue_names[i] for i in range(len(a.coords))}
    res_name_b = {b.residue_ids[i]: b.residue_names[i] for i in range(len(b.coords))}

    bins = {"CC": 0, "AC": 0, "AP": 0, "AA": 0, "PP": 0, "CP": 0}
    for ra, rb in res_pairs:
        pa = _prodigy_ic_class(res_name_a[ra])
        pb = _prodigy_ic_class(res_name_b[rb])
        if pa is None or pb is None:
            continue
        key = "".join(sorted([pa, pb]))  # AA, AC, AP, CC, CP, PP
        if key in bins:
            bins[key] += 1
    return bins


def prodigy_ics(
    a: AtomArrays,
    b: AtomArrays,
    cutoff: float = PRODIGY_IC_CUTOFF,
) -> dict[str, int]:
    """Cross-interface residue-residue contacts by polarity class.

    A residue pair is counted as a contact when any heavy atom of A is within
    `cutoff` Å of any heavy atom of B (default 5.5 Å, the PRODIGY convention).
    Returns a dict with keys CC, AC, AP, AA, PP, CP and `total`.
    Non-standard residues are skipped.
    """
    _validate_nonnegative_finite("cutoff", cutoff)
    if not a.coords or not b.coords:
        return {"CC": 0, "AC": 0, "AP": 0, "AA": 0, "PP": 0, "CP": 0, "total": 0}
    A = np.asarray(a.coords)
    B = np.asarray(b.coords)
    sq_a = (A * A).sum(axis=1)
    sq_b = (B * B).sum(axis=1)
    d2 = sq_a[:, None] + sq_b[None, :] - 2.0 * (A @ B.T)
    within = d2 <= cutoff * cutoff
    bins = _ic_bins_from_contacts(a, b, within)
    bins["total"] = sum(bins.values())
    return bins


def _nis_from_arrays(
    combined: AtomArrays,
    sasa_complex: np.ndarray,
    interface_rids: set[tuple],
    rsasa_cutoff: float = PRODIGY_NIS_RSASA_CUTOFF,
) -> dict[str, float]:
    """Compute % apolar / polar / charged in the non-interacting surface.

    NIS is defined (per PRODIGY) as residues with relative SASA > 5 % in the
    bound complex AND not at the interface. Residue rSASA = sum of per-atom
    complex SASA divided by the Tien et al. 2013 reference for that residue.
    """
    _validate_nonnegative_finite("rsasa_cutoff", rsasa_cutoff)
    per_res: dict[tuple, float] = {}
    per_res_name: dict[tuple, str] = {}
    for i, rid in enumerate(combined.residue_ids):
        per_res[rid] = per_res.get(rid, 0.0) + float(sasa_complex[i])
        per_res_name[rid] = combined.residue_names[i]

    n_apolar = n_polar = n_charged = 0
    for rid, s in per_res.items():
        if rid in interface_rids:
            continue
        rname = per_res_name[rid]
        ref = REFERENCE_SASA.get(rname)
        if ref is None:
            continue
        if 100.0 * s / ref <= rsasa_cutoff:
            continue
        cls = _prodigy_nis_class(rname)
        if cls == "A":
            n_apolar += 1
        elif cls == "P":
            n_polar += 1
        elif cls == "C":
            n_charged += 1
    total = n_apolar + n_polar + n_charged
    if total == 0:
        return {"apolar": 0.0, "polar": 0.0, "charged": 0.0, "n_residues": 0}
    return {
        "apolar":     100.0 * n_apolar / total,
        "polar":      100.0 * n_polar / total,
        "charged":    100.0 * n_charged / total,
        "n_residues": total,
    }


def prodigy_nis(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 92,
    cutoff: float = PRODIGY_IC_CUTOFF,
    rsasa_cutoff: float = PRODIGY_NIS_RSASA_CUTOFF,
    strict: bool = True,
) -> dict[str, float]:
    """% apolar / polar / charged composition of the non-interacting surface.

    Surface = relative SASA > `rsasa_cutoff` % in the bound complex (per-atom
    SASA summed by residue, divided by the Tien et al. 2013 reference for
    that residue type).
    Interface = residues with any heavy atom within `cutoff` Å of the opposite
    chain (5.5 Å is the PRODIGY convention).
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("cutoff", cutoff)
    _validate_nonnegative_finite("rsasa_cutoff", rsasa_cutoff)
    combined, sasa_c, _ = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    int_a, int_b = interface_residues(a, b, cutoff)
    interface_rids = (
        {_side_residue_id("a", rid) for rid in int_a}
        | {_side_residue_id("b", rid) for rid in int_b}
    )
    return _nis_from_arrays(combined, sasa_c, interface_rids, rsasa_cutoff)


def _prodigy_dg_from_parts(ics: dict[str, int], nis: dict[str, float]) -> float:
    return (
        PRODIGY_COEFFS["CC"]          * ics["CC"]
        + PRODIGY_COEFFS["AC"]        * ics["AC"]
        + PRODIGY_COEFFS["PP"]        * ics["PP"]
        + PRODIGY_COEFFS["AP"]        * ics["AP"]
        + PRODIGY_COEFFS["NIS_apolar"]  * nis["apolar"]
        + PRODIGY_COEFFS["NIS_charged"] * nis["charged"]
        + PRODIGY_COEFFS["intercept"]
    )


def prodigy(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 92,
    cutoff: float = PRODIGY_IC_CUTOFF,
    rsasa_cutoff: float = PRODIGY_NIS_RSASA_CUTOFF,
    strict: bool = True,
) -> dict:
    """Predicted binding free energy via the PRODIGY model (Vangone & Bonvin 2015).

    Returns a dict with the intermediate counts (`ics`), NIS percentages
    (`nis`), and the predicted `dg` in kcal/mol. More negative = tighter
    predicted binding.
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("cutoff", cutoff)
    _validate_nonnegative_finite("rsasa_cutoff", rsasa_cutoff)
    ics = prodigy_ics(a, b, cutoff)
    nis = prodigy_nis(a, b, probe_radius, n_points, cutoff, rsasa_cutoff, strict=strict)
    return {"ics": ics, "nis": nis, "dg": _prodigy_dg_from_parts(ics, nis)}


# ── SASA helpers ─────────────────────────────────────────────────────────────

def _per_atom_dsasa(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float,
    n_points: int,
    *,
    strict: bool = True,
):
    """Return (combined, sasa_complex, sasa_separated) — used by metrics that
    re-slice dSASA by atom class. Computes SASA three times."""
    _validate_sasa_params(probe_radius, n_points)
    _validate_for_sasa(a, b, strict=strict)
    sasa_a = sasa(a, probe_radius, n_points, strict=strict)
    sasa_b = sasa(b, probe_radius, n_points, strict=strict)
    combined = _combined(a, b)
    sasa_ab = compute_sasa(
        combined.coords, combined.atom_names, combined.residue_names, probe_radius, n_points
    )
    return combined, np.asarray(sasa_ab), np.asarray(sasa_a + sasa_b)


def bsa_breakdown(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> dict[str, float]:
    """Split dSASA into BHSA / BPSA / BCSA.

    BHSA — buried hydrophobic surface area (atom-name starts with C or S)
    BPSA — buried polar surface area (atom-name starts with N or O)
    BCSA — buried charged surface area (subset of BPSA: ASP/GLU/LYS/ARG/HIS terminal atoms)

    Returns a dict with keys 'dsasa', 'bhsa', 'bpsa', 'bcsa', 'hydrophobic_fraction'.
    """
    _validate_sasa_params(probe_radius, n_points)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    bhsa = 0.0
    bpsa = 0.0
    bcsa = 0.0
    for i, name in enumerate(combined.atom_names):
        first = name[:1]
        di = float(d[i])
        if first in ("C", "S"):
            bhsa += di
        elif first in ("N", "O"):
            bpsa += di
            if (combined.residue_names[i], name) in CHARGED_ATOMS:
                bcsa += di
    total = float(d.sum())
    return {
        "dsasa": total,
        "bhsa": bhsa,
        "bpsa": bpsa,
        "bcsa": bcsa,
        "hydrophobic_fraction": (bhsa / total) if total > 0 else 0.0,
    }


def per_residue_dsasa(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> dict[tuple[str, str, int, str], float]:
    """Per-residue buried surface area keyed by (side, chain_id, resseq, icode)."""
    _validate_sasa_params(probe_radius, n_points)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    out: dict[tuple[str, str, int, str], float] = {}
    d = sasa_s - sasa_c
    for i, rid in enumerate(combined.residue_ids):
        out[rid] = out.get(rid, 0.0) + float(d[i])
    return out


def hotspot_residues(
    per_res: dict[tuple, float],
    threshold: float = 30.0,
) -> list[tuple]:
    """Residues whose buried surface area exceeds `threshold` Å² (default 30).

    Sorted by descending dSASA. 30 Å² is a common heuristic for 'significant'
    interface contribution (Bogan & Thorn 1998 use a similar magnitude for
    alanine-scanning hot spots).
    """
    return [rid for rid, _ in sorted(per_res.items(), key=lambda kv: -kv[1]) if _ >= threshold]


def hbond_density(n_hbonds: int, dsasa_value: float) -> float:
    """H-bonds per 100 Å² of buried surface area. 0.0 if dSASA ≤ 0."""
    return (100.0 * n_hbonds / dsasa_value) if dsasa_value > 0 else 0.0


# ── Aromatic geometry ────────────────────────────────────────────────────────

def _aromatic_rings(arr: AtomArrays) -> list[tuple[tuple, str, np.ndarray, np.ndarray]]:
    """Group ring atoms by residue and compute (centroid, normal) for each ring.

    Returns a list of (residue_id, residue_name, centroid, normal). Skips
    residues missing required ring atoms.
    """
    by_res: dict[tuple, dict[str, np.ndarray]] = {}
    res_names: dict[tuple, str] = {}
    for i, rid in enumerate(arr.residue_ids):
        rname = arr.residue_names[i]
        if rname not in RING_ATOMS:
            continue
        if arr.atom_names[i] in RING_ATOMS[rname]:
            by_res.setdefault(rid, {})[arr.atom_names[i]] = np.asarray(arr.coords[i])
            res_names[rid] = rname

    out = []
    for rid, atoms in by_res.items():
        rname = res_names[rid]
        needed = RING_ATOMS[rname]
        if not all(k in atoms for k in needed):
            continue
        pts = np.stack([atoms[k] for k in needed])
        centroid = pts.mean(axis=0)
        # Two non-parallel in-plane vectors for the normal.
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm == 0:
            continue
        out.append((rid, rname, centroid, n / norm))
    return out


def pi_pi_contacts(
    a: AtomArrays,
    b: AtomArrays,
    distance_cutoff: float = 7.0,
    angle_cutoff_deg: float = 90.0,
) -> int:
    """Count aromatic ring pairs across the interface.

    A pair is counted when centroid-centroid distance ≤ `distance_cutoff` Å and
    the absolute angle between ring normals is ≤ `angle_cutoff_deg` (default
    90° accepts both face-to-face and T-shaped geometries). Reference geometric
    range follows McGaughey et al. 1998.
    """
    _validate_nonnegative_finite("distance_cutoff", distance_cutoff)
    _validate_nonnegative_finite("angle_cutoff_deg", angle_cutoff_deg)
    rings_a = _aromatic_rings(a)
    rings_b = _aromatic_rings(b)
    cos_min = math.cos(math.radians(angle_cutoff_deg))
    count = 0
    for _, _, ca, na in rings_a:
        for _, _, cb, nb in rings_b:
            if np.linalg.norm(ca - cb) > distance_cutoff:
                continue
            # angle_cutoff_deg = 90 → cos_min = 0, always satisfied. Allow either
            # parallel or anti-parallel by comparing the absolute cosine.
            if abs(float(np.dot(na, nb))) >= cos_min:
                count += 1
    return count


def cation_pi_contacts(
    a: AtomArrays,
    b: AtomArrays,
    distance_cutoff: float = 6.0,
) -> int:
    """Count cation-π contacts across the interface (distance-only).

    Cation atoms: LYS NZ and ARG CZ (geometric centre of the guanidinium).
    Aromatic centres: ring centroids of PHE/TYR/TRP/HIS. A contact is counted
    when the cation lies within `distance_cutoff` Å of a centroid on the other
    side. Reference range follows Gallivan & Dougherty 1999 (~3.4–6.0 Å).
    """
    _validate_nonnegative_finite("distance_cutoff", distance_cutoff)
    def _cations(arr: AtomArrays):
        for i, name in enumerate(arr.atom_names):
            r = arr.residue_names[i]
            if (r == "LYS" and name == "NZ") or (r == "ARG" and name == "CZ"):
                yield np.asarray(arr.coords[i])

    cats_a = list(_cations(a))
    cats_b = list(_cations(b))
    rings_a = _aromatic_rings(a)
    rings_b = _aromatic_rings(b)

    count = 0
    for c in cats_a:
        for _, _, centroid, _ in rings_b:
            if np.linalg.norm(c - centroid) <= distance_cutoff:
                count += 1
    for c in cats_b:
        for _, _, centroid, _ in rings_a:
            if np.linalg.norm(c - centroid) <= distance_cutoff:
                count += 1
    return count


# ── Buried unsatisfied polars ────────────────────────────────────────────────

def buried_unsat_polar(
    a: AtomArrays,
    b: AtomArrays,
    sasa_cutoff: float = 1.0,
    hbond_cutoff: float = 3.5,
    probe_radius: float = 1.4,
    n_points: int = 960,
    strict: bool = True,
) -> int:
    """Count polar atoms buried at the interface with no cross-interface partner.

    A polar atom (donor or acceptor) is counted as unsatisfied if:
        • its SASA in the complex is below `sasa_cutoff` (Å²), AND
        • it gains burial on binding (dSASA > sasa_cutoff), AND
        • no complementary polar partner exists within `hbond_cutoff` Å on the
          opposite chain.

    This is a geometric count of buried unsatisfied polars. Intra-chain H-bond
    partners are not considered — if an atom were satisfied intramolecularly in
    the unbound state, it would not normally meet the burial-gain criterion.
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("sasa_cutoff", sasa_cutoff)
    _validate_nonnegative_finite("hbond_cutoff", hbond_cutoff)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    return _buried_unsat_from_arrays(
        combined, sasa_c, d, len(a.coords), sasa_cutoff, hbond_cutoff
    )


# ── Shape ────────────────────────────────────────────────────────────────────

def interface_shape(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 960,
    min_atom_dsasa: float = 0.5,
    strict: bool = True,
) -> dict[str, float]:
    """Planarity and elongation of the interface patch via PCA.

    Interface atoms are those with per-atom dSASA ≥ `min_atom_dsasa` (Å²). The
    covariance matrix of their coordinates gives singular values σ1 ≥ σ2 ≥ σ3.

    Returns:
        planarity_rmsd  — RMS perpendicular distance to the best-fit plane (Å)
        elongation      — σ1 / σ2 (≥ 1; large = elongated)
        planarity_ratio — σ3 / σ2 (≤ 1; small = flat)

    Returns NaN values when fewer than 3 interface atoms are present.
    """
    _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("min_atom_dsasa", min_atom_dsasa)
    combined, sasa_c, sasa_s = _per_atom_dsasa(a, b, probe_radius, n_points, strict=strict)
    d = sasa_s - sasa_c
    mask = d >= min_atom_dsasa
    pts = np.asarray(combined.coords)[mask]
    nan = {"planarity_rmsd": float("nan"), "elongation": float("nan"), "planarity_ratio": float("nan")}
    if pts.shape[0] < 3:
        return nan
    centered = pts - pts.mean(axis=0)
    # SVD of centered coords gives singular values proportional to PCA axes.
    s = np.linalg.svd(centered, compute_uv=False)
    n = pts.shape[0]
    s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])
    return {
        "planarity_rmsd": s3 / math.sqrt(n),
        "elongation": (s1 / s2) if s2 > 0 else float("nan"),
        "planarity_ratio": (s3 / s2) if s2 > 0 else float("nan"),
    }


# ── Combined ─────────────────────────────────────────────────────────────────

def _compute_sc_value(a: AtomArrays, b: AtomArrays, parallel: bool, *, strict: bool) -> float:
    """Run compute_sc, raising in strict mode or returning NaN only by explicit opt-out."""
    try:
        r = compute_sc(
            a.coords, a.atom_names, a.residue_names,
            b.coords, b.atom_names, b.residue_names,
            parallel,
        )
        return float(r.sc)
    except Exception as exc:
        if strict:
            raise ValueError(f"shape complementarity failed: {exc}") from exc
        return float("nan")


def analyze(
    a: AtomArrays,
    b: AtomArrays,
    probe_radius: float = 1.4,
    n_points: int = 92,
    interface_cutoff: float = 5.0,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
    pi_pi_distance: float = 7.0,
    cation_pi_distance: float = 6.0,
    hotspot_threshold: float = 30.0,
    unsat_sasa_cutoff: float = 1.0,
    min_atom_dsasa_for_shape: float = 0.5,
    include_sc: bool = True,
    sc_parallel: bool = True,
    strict: bool = True,
    metrics: Iterable[str] | None = None,
    skip_metrics: Iterable[str] | None = None,
) -> InterfaceResult:
    """Compute all interface metrics in one pass.

    SASA is computed three times (A alone, B alone, A+B) and the per-atom arrays
    are reused for dSASA, BSA breakdown, per-residue aggregation, buried-unsat
    detection, and shape PCA. The Rust counters (H-bonds, salt bridges) and the
    aromatic geometry pass over independent atom data. Lawrence-Colman shape
    complementarity is computed via the ``sc-rs`` kernel when ``include_sc=True``
    (default). With ``strict=True`` (default), invalid atom arrays, unknown
    SASA radii, and SC failures raise clear errors; pass ``strict=False`` for
    permissive screening where SC failures return ``NaN``.

    The default ``n_points=92`` is tuned for batch screening: total SASA noise
    on a typical protein-protein interface is ≈ 1 % of dSASA, well below the
    spread of any practical filtering threshold. For one-off measurement
    against published numbers, pass ``n_points=960``. For sweeping many
    complexes at once see :func:`analyze_batch`, which batches all SASA work
    into a single parallel Rust call.
    """
    enabled_metrics = _resolve_analyze_metrics(metrics, skip_metrics)
    needs_sasa = bool(enabled_metrics & _SASA_ANALYZE_METRICS)
    if needs_sasa:
        _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("interface_cutoff", interface_cutoff)
    _validate_nonnegative_finite("hbond_cutoff", hbond_cutoff)
    _validate_nonnegative_finite("salt_bridge_cutoff", salt_bridge_cutoff)
    _validate_nonnegative_finite("pi_pi_distance", pi_pi_distance)
    _validate_nonnegative_finite("cation_pi_distance", cation_pi_distance)
    _validate_nonnegative_finite("hotspot_threshold", hotspot_threshold)
    _validate_nonnegative_finite("unsat_sasa_cutoff", unsat_sasa_cutoff)
    _validate_nonnegative_finite("min_atom_dsasa_for_shape", min_atom_dsasa_for_shape)
    if needs_sasa:
        _validate_for_sasa(a, b, strict=strict)
    else:
        _validate_atom_arrays(a, "atoms_a", require_nonempty=strict)
        _validate_atom_arrays(b, "atoms_b", require_nonempty=strict)
    combined = _combined(a, b)
    if needs_sasa:
        sasa_a = sasa(a, probe_radius, n_points, strict=strict)
        sasa_b = sasa(b, probe_radius, n_points, strict=strict)
        sasa_ab = compute_sasa(
            combined.coords, combined.atom_names, combined.residue_names, probe_radius, n_points
        )
    else:
        sasa_a = sasa_b = sasa_ab = None
    sc_value = (
        _compute_sc_value(a, b, sc_parallel, strict=strict)
        if include_sc and "sc" in enabled_metrics
        else (float("nan") if "sc" in enabled_metrics else None)
    )
    return _analyze_from_sasa(
        a, b, combined, sasa_a, sasa_b, sasa_ab,
        sc_value=sc_value,
        enabled_metrics=enabled_metrics,
        interface_cutoff=interface_cutoff,
        hbond_cutoff=hbond_cutoff,
        salt_bridge_cutoff=salt_bridge_cutoff,
        pi_pi_distance=pi_pi_distance,
        cation_pi_distance=cation_pi_distance,
        hotspot_threshold=hotspot_threshold,
        unsat_sasa_cutoff=unsat_sasa_cutoff,
        min_atom_dsasa_for_shape=min_atom_dsasa_for_shape,
    )


def analyze_batch(
    complexes: list[tuple[AtomArrays, AtomArrays]],
    probe_radius: float = 1.4,
    n_points: int = 92,
    parallel: bool = True,
    interface_cutoff: float = 5.0,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
    pi_pi_distance: float = 7.0,
    cation_pi_distance: float = 6.0,
    hotspot_threshold: float = 30.0,
    unsat_sasa_cutoff: float = 1.0,
    min_atom_dsasa_for_shape: float = 0.5,
    include_sc: bool = True,
    strict: bool = True,
    metrics: Iterable[str] | None = None,
    skip_metrics: Iterable[str] | None = None,
) -> list[InterfaceResult]:
    """Analyse many complexes with a single batched SASA call.

    All 3·N SASA computations (each complex needs sasa(A), sasa(B), sasa(A∪B))
    are pushed into one Rust call that releases the GIL and parallelises across
    Rayon threads. When SC is enabled, SC is also computed through one batched
    Rust call before the per-complex Python orchestration runs serially.

    With ``parallel=True`` (default), set the environment variable
    ``RAYON_NUM_THREADS`` to control thread count if needed. When dispatching
    from a ``ProcessPoolExecutor`` pass ``parallel=False`` to avoid CPU
    oversubscription — the same convention as :func:`compute_sc` and
    :func:`compute_sasa_batch`.

    Args mirror :func:`analyze`. Returns one :class:`InterfaceResult` per input
    complex, in order.
    """
    enabled_metrics = _resolve_analyze_metrics(metrics, skip_metrics)
    needs_sasa = bool(enabled_metrics & _SASA_ANALYZE_METRICS)
    if needs_sasa:
        _validate_sasa_params(probe_radius, n_points)
    _validate_nonnegative_finite("interface_cutoff", interface_cutoff)
    _validate_nonnegative_finite("hbond_cutoff", hbond_cutoff)
    _validate_nonnegative_finite("salt_bridge_cutoff", salt_bridge_cutoff)
    _validate_nonnegative_finite("pi_pi_distance", pi_pi_distance)
    _validate_nonnegative_finite("cation_pi_distance", cation_pi_distance)
    _validate_nonnegative_finite("hotspot_threshold", hotspot_threshold)
    _validate_nonnegative_finite("unsat_sasa_cutoff", unsat_sasa_cutoff)
    _validate_nonnegative_finite("min_atom_dsasa_for_shape", min_atom_dsasa_for_shape)
    if not complexes:
        return []
    for i, (a, b) in enumerate(complexes):
        try:
            if needs_sasa:
                _validate_for_sasa(a, b, strict=strict)
            else:
                _validate_atom_arrays(a, "atoms_a", require_nonempty=strict)
                _validate_atom_arrays(b, "atoms_b", require_nonempty=strict)
        except ValueError as exc:
            raise ValueError(f"complex {i}: {exc}") from exc

    combineds = [_combined(a, b) for a, b in complexes]
    if needs_sasa:
        inputs: list[tuple[list[list[float]], list[str], list[str]]] = []
        for (a, b), combined in zip(complexes, combineds):
            inputs.append((a.coords, a.atom_names, a.residue_names))
            inputs.append((b.coords, b.atom_names, b.residue_names))
            inputs.append((combined.coords, combined.atom_names, combined.residue_names))
        all_sasa = compute_sasa_batch(inputs, probe_radius, n_points, parallel)
    else:
        all_sasa = []

    if include_sc and "sc" in enabled_metrics:
        sc_inputs = [
            (a.coords, a.atom_names, a.residue_names, b.coords, b.atom_names, b.residue_names)
            for a, b in complexes
        ]
        try:
            sc_results = compute_sc_batch(sc_inputs, parallel)
        except Exception as exc:
            if strict:
                raise ValueError(f"shape complementarity failed: {exc}") from exc
            sc_values = [float("nan")] * len(complexes)
        else:
            sc_values = [float(r.sc) for r in sc_results]
    else:
        sc_values = [
            float("nan") if "sc" in enabled_metrics else None
            for _ in complexes
        ]

    results: list[InterfaceResult] = []
    for i, (a, b) in enumerate(complexes):
        sasa_a = all_sasa[3 * i] if needs_sasa else None
        sasa_b = all_sasa[3 * i + 1] if needs_sasa else None
        sasa_ab = all_sasa[3 * i + 2] if needs_sasa else None
        results.append(_analyze_from_sasa(
            a, b, combineds[i], sasa_a, sasa_b, sasa_ab,
            sc_value=sc_values[i],
            enabled_metrics=enabled_metrics,
            interface_cutoff=interface_cutoff,
            hbond_cutoff=hbond_cutoff,
            salt_bridge_cutoff=salt_bridge_cutoff,
            pi_pi_distance=pi_pi_distance,
            cation_pi_distance=cation_pi_distance,
            hotspot_threshold=hotspot_threshold,
            unsat_sasa_cutoff=unsat_sasa_cutoff,
            min_atom_dsasa_for_shape=min_atom_dsasa_for_shape,
        ))
    return results


def _analyze_from_sasa(
    a: AtomArrays,
    b: AtomArrays,
    combined: AtomArrays,
    sasa_a: list[float] | None,
    sasa_b: list[float] | None,
    sasa_ab: list[float] | None,
    *,
    sc_value: float | None,
    enabled_metrics: frozenset[str],
    interface_cutoff: float,
    hbond_cutoff: float,
    salt_bridge_cutoff: float,
    pi_pi_distance: float,
    cation_pi_distance: float,
    hotspot_threshold: float,
    unsat_sasa_cutoff: float,
    min_atom_dsasa_for_shape: float,
) -> InterfaceResult:
    """Internal orchestrator: assemble an InterfaceResult from pre-computed SASA arrays."""
    def enabled(name: str) -> bool:
        return name in enabled_metrics

    needs_sasa = bool(enabled_metrics & _SASA_ANALYZE_METRICS)
    if needs_sasa:
        if sasa_a is None or sasa_b is None or sasa_ab is None:
            raise ValueError("SASA arrays are required for the selected analyze metrics")
        sasa_c = np.asarray(sasa_ab)
        sasa_s = np.asarray(sasa_a + sasa_b)
        d = sasa_s - sasa_c
    else:
        sasa_c = np.asarray([])
        d = np.asarray([])

    # dSASA decompositions in a single sweep.
    na_atoms = len(a.coords)
    dsasa_total = arom = bhsa = bpsa = bcsa = bb_d = sc_d = None
    d_a = d_b = asym = None
    hs_a = hs_b = None
    if needs_sasa:
        dsasa_total = float(d.sum())
        arom = bhsa = bpsa = bcsa = bb_d = sc_d = 0.0
        per_res: dict[tuple, float] = {}
        for i, rname in enumerate(combined.residue_names):
            di = float(d[i])
            rid = combined.residue_ids[i]
            per_res[rid] = per_res.get(rid, 0.0) + di
            if rname in AROMATIC:
                arom += di
            aname = combined.atom_names[i]
            if aname in BACKBONE_ATOMS:
                bb_d += di
            else:
                sc_d += di
            first = aname[:1]
            if first in ("C", "S"):
                bhsa += di
            elif first in ("N", "O"):
                bpsa += di
                if (rname, aname) in CHARGED_ATOMS:
                    bcsa += di

        hotspots = [rid for rid, v in sorted(per_res.items(), key=lambda kv: -kv[1]) if v >= hotspot_threshold]
        hs_a = [_strip_side_residue_id(rid) for rid in hotspots if rid[0] == "a"]
        hs_b = [_strip_side_residue_id(rid) for rid in hotspots if rid[0] == "b"]
        d_a = float(d[:na_atoms].sum())
        d_b = float(d[na_atoms:].sum())
        m_ab = max(d_a, d_b)
        asym = abs(d_a - d_b) / m_ab if m_ab > 0 else 0.0

    needs_interface_residues = any(enabled(m) for m in (
        "n_interface_a", "n_interface_b", "gly_pro_fraction",
        "charge_a", "charge_b", "charge_complementarity",
    ))
    if needs_interface_residues:
        int_a_set, int_b_set = interface_residues(a, b, interface_cutoff)
        n_a, n_b = len(int_a_set), len(int_b_set)
    else:
        int_a_set, int_b_set = set(), set()
        n_a = n_b = None

    n_hb = hbonds(a, b, hbond_cutoff) if (enabled("hbonds") or enabled("hbond_density")) else None
    n_sb = salt_bridges(a, b, salt_bridge_cutoff) if enabled("salt_bridges") else None

    # Gly+Pro fraction over the same interface-residue set.
    res_name_a = {a.residue_ids[i]: a.residue_names[i] for i in range(na_atoms)}
    res_name_b = {b.residue_ids[i]: b.residue_names[i] for i in range(len(b.coords))}
    if enabled("gly_pro_fraction"):
        int_total = len(int_a_set) + len(int_b_set)
        gp_count = sum(1 for r in int_a_set if res_name_a.get(r) in ("GLY", "PRO"))
        gp_count += sum(1 for r in int_b_set if res_name_b.get(r) in ("GLY", "PRO"))
        gp_frac = (gp_count / int_total) if int_total > 0 else 0.0
    else:
        gp_frac = None

    # Formal-charge sums over interface residues.
    if enabled("charge_a") or enabled("charge_b") or enabled("charge_complementarity"):
        qa_value = sum(RESIDUE_CHARGE.get(res_name_a.get(r, ""), 0) for r in int_a_set)
        qb_value = sum(RESIDUE_CHARGE.get(res_name_b.get(r, ""), 0) for r in int_b_set)
        charge_compl = float(-qa_value * qb_value)
        qa = qa_value if enabled("charge_a") or enabled("charge_complementarity") else None
        qb = qb_value if enabled("charge_b") or enabled("charge_complementarity") else None
    else:
        qa = qb = charge_compl = None

    # Disulfides — Cys SG–SG pairs across the interface.
    if enabled("disulfides"):
        sg_a = [np.asarray(a.coords[i]) for i in range(na_atoms)
                if a.residue_names[i] == "CYS" and a.atom_names[i] == "SG"]
        sg_b = [np.asarray(b.coords[i]) for i in range(len(b.coords))
                if b.residue_names[i] == "CYS" and b.atom_names[i] == "SG"]
        ssbond = sum(1 for x in sg_a for y in sg_b if float((x - y) @ (x - y)) <= 2.5 * 2.5)
    else:
        ssbond = None

    # Atomic contact count via numpy broadcast.
    needs_xyz = needs_sasa or enabled("atomic_contacts") or enabled("prodigy_dg")
    A_xyz = np.asarray(a.coords, dtype=float).reshape((len(a.coords), 3)) if needs_xyz else None
    B_xyz = np.asarray(b.coords, dtype=float).reshape((len(b.coords), 3)) if needs_xyz else None
    if enabled("atomic_contacts") or enabled("prodigy_dg"):
        if len(a.coords) == 0 or len(b.coords) == 0:
            contact_d2 = np.zeros((len(a.coords), len(b.coords)), dtype=float)
        else:
            contact_d2 = ((A_xyz[:, None, :] - B_xyz[None, :, :]) ** 2).sum(-1)
    else:
        contact_d2 = None
    n_contacts = (
        int((contact_d2 <= interface_cutoff * interface_cutoff).sum())
        if enabled("atomic_contacts")
        else None
    )

    # Aromatic geometry uses only ring atoms — independent of SASA.
    n_pi = pi_pi_contacts(a, b, distance_cutoff=pi_pi_distance) if enabled("pi_pi") else None
    n_cpi = cation_pi_contacts(a, b, distance_cutoff=cation_pi_distance) if enabled("cation_pi") else None

    # Buried-unsat reuses sasa_c / d arrays.
    unsat = (
        _buried_unsat_from_arrays(combined, sasa_c, d, na_atoms, unsat_sasa_cutoff, hbond_cutoff)
        if enabled("buried_unsat_polar")
        else None
    )

    # Shape PCA reuses sasa_c / d arrays.
    needs_mask = any(enabled(m) for m in (
        "planarity_rmsd", "elongation", "planarity_ratio",
        "interface_depth", "mean_bfactor_interface", "min_bfactor_interface",
    ))
    if needs_mask:
        mask = d >= min_atom_dsasa_for_shape
        pts = np.asarray(combined.coords)[mask]
        if pts.shape[0] >= 3:
            centered = pts - pts.mean(axis=0)
            s = np.linalg.svd(centered, compute_uv=False)
            n = pts.shape[0]
            s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])
            plan_rmsd = s3 / math.sqrt(n)
            elong = (s1 / s2) if s2 > 0 else float("nan")
            plan_ratio = (s3 / s2) if s2 > 0 else float("nan")
        else:
            plan_rmsd = elong = plan_ratio = float("nan")
    else:
        mask = np.asarray([], dtype=bool)
        plan_rmsd = elong = plan_ratio = None

    # Interface depth — centroid-to-centroid distance using the same mask.
    if needs_mask:
        mask_a = mask[:na_atoms]
        mask_b = mask[na_atoms:]
    else:
        mask_a = mask_b = np.asarray([], dtype=bool)
    if enabled("interface_depth"):
        if mask_a.any() and mask_b.any():
            c_a = A_xyz[mask_a].mean(axis=0)
            c_b = B_xyz[mask_b].mean(axis=0)
            depth = float(np.linalg.norm(c_a - c_b))
        else:
            depth = float("nan")
    else:
        depth = None

    # B-factor (pLDDT) at interface — uses the same mask as the shape PCA.
    if enabled("mean_bfactor_interface") or enabled("min_bfactor_interface"):
        if a.bfactors is not None and b.bfactors is not None and (mask_a.any() or mask_b.any()):
            bf = np.concatenate([
                np.asarray(a.bfactors)[mask_a],
                np.asarray(b.bfactors)[mask_b],
            ])
            mean_bf = float(bf.mean())
            min_bf = float(bf.min())
        else:
            mean_bf = float("nan")
            min_bf = float("nan")
    else:
        mean_bf = min_bf = None

    # PRODIGY ΔG — reuse the contact_d² matrix (at PRODIGY's 5.5 Å cutoff) and
    # the per-atom complex SASA. Adds ~3 ms/call.
    if enabled("prodigy_dg"):
        prodigy_within = contact_d2 <= PRODIGY_IC_CUTOFF * PRODIGY_IC_CUTOFF
        prodigy_ic_bins = _ic_bins_from_contacts(a, b, prodigy_within)
        prodigy_interface = (
            {_side_residue_id("a", a.residue_ids[ai]) for ai, _ in np.argwhere(prodigy_within)}
            | {_side_residue_id("b", b.residue_ids[bi]) for _, bi in np.argwhere(prodigy_within)}
        )
        prodigy_nis_parts = _nis_from_arrays(combined, sasa_c, prodigy_interface)
        prodigy_dg_value = _prodigy_dg_from_parts(prodigy_ic_bins, prodigy_nis_parts)
    else:
        prodigy_dg_value = None

    return InterfaceResult(
        sc=sc_value if enabled("sc") else None,
        dsasa=dsasa_total if enabled("dsasa") else None,
        n_interface_a=n_a if enabled("n_interface_a") else None,
        n_interface_b=n_b if enabled("n_interface_b") else None,
        aromatic_dsasa_fraction=((arom / dsasa_total) if dsasa_total > 0 else 0.0) if enabled("aromatic_dsasa_fraction") else None,
        hbonds=n_hb if enabled("hbonds") else None,
        salt_bridges=n_sb if enabled("salt_bridges") else None,
        bhsa=bhsa if enabled("bhsa") else None,
        bpsa=bpsa if enabled("bpsa") else None,
        bcsa=bcsa if enabled("bcsa") else None,
        hydrophobic_fraction=((bhsa / dsasa_total) if dsasa_total > 0 else 0.0) if enabled("hydrophobic_fraction") else None,
        hbond_density=((100.0 * n_hb / dsasa_total) if dsasa_total > 0 else 0.0) if enabled("hbond_density") else None,
        pi_pi=n_pi if enabled("pi_pi") else None,
        cation_pi=n_cpi if enabled("cation_pi") else None,
        buried_unsat_polar=unsat,
        planarity_rmsd=plan_rmsd if enabled("planarity_rmsd") else None,
        elongation=elong if enabled("elongation") else None,
        planarity_ratio=plan_ratio if enabled("planarity_ratio") else None,
        dsasa_a=d_a if enabled("dsasa_a") else None,
        dsasa_b=d_b if enabled("dsasa_b") else None,
        asymmetry=asym if enabled("asymmetry") else None,
        atomic_contacts=n_contacts,
        interface_depth=depth,
        disulfides=ssbond,
        gly_pro_fraction=gp_frac,
        bb_dsasa=bb_d if enabled("bb_dsasa") else None,
        sc_dsasa=sc_d if enabled("sc_dsasa") else None,
        sidechain_fraction=((sc_d / dsasa_total) if dsasa_total > 0 else 0.0) if enabled("sidechain_fraction") else None,
        charge_a=qa if enabled("charge_a") else None,
        charge_b=qb if enabled("charge_b") else None,
        charge_complementarity=charge_compl,
        mean_bfactor_interface=mean_bf if enabled("mean_bfactor_interface") else None,
        min_bfactor_interface=min_bf if enabled("min_bfactor_interface") else None,
        prodigy_dg=prodigy_dg_value,
        hotspots_a=hs_a if enabled("hotspots_a") else None,
        hotspots_b=hs_b if enabled("hotspots_b") else None,
    )


def _buried_unsat_from_arrays(
    combined: AtomArrays,
    sasa_c: np.ndarray,
    d: np.ndarray,
    na: int,
    sasa_cutoff: float,
    hbond_cutoff: float,
) -> int:
    """Vectorised count of buried unsatisfied polar atoms.

    Only polar atoms participate in the distance matrix, so memory cost is
    8·N_polar_a·N_polar_b bytes (a few MB at most for realistic complexes).
    """
    coords_all = np.asarray(combined.coords)
    n_total = len(coords_all)

    # Classify polarity once per atom via a list comprehension (faster than
    # an indexed for-loop, since Python attribute lookups dominate at this scale).
    rn = combined.residue_names
    an = combined.atom_names
    donor = np.fromiter((_is_donor(rn[i], an[i]) for i in range(n_total)), dtype=bool, count=n_total)
    acceptor = np.fromiter((_is_acceptor(rn[i], an[i]) for i in range(n_total)), dtype=bool, count=n_total)
    polar = donor | acceptor

    side_a = np.zeros(n_total, dtype=bool)
    side_a[:na] = True

    pa_idx = np.flatnonzero(polar & side_a)
    pb_idx = np.flatnonzero(polar & ~side_a)

    # Partner masks: only computed when both sides contribute polars. When
    # one side has none, every buried polar on the other side is unsatisfied
    # by construction.
    if pa_idx.size > 0 and pb_idx.size > 0:
        pa = coords_all[pa_idx]
        pb = coords_all[pb_idx]
        d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
        within = d2 <= hbond_cutoff * hbond_cutoff
        da = donor[pa_idx]
        aa = acceptor[pa_idx]
        db = donor[pb_idx]
        ab = acceptor[pb_idx]
        pair = (da[:, None] & ab[None, :]) | (aa[:, None] & db[None, :])
        partnered = within & pair
        has_partner_a = partnered.any(axis=1)
        has_partner_b = partnered.any(axis=0)
    else:
        has_partner_a = np.zeros(pa_idx.size, dtype=bool)
        has_partner_b = np.zeros(pb_idx.size, dtype=bool)

    unsat = 0
    if pa_idx.size > 0:
        buried_a = (sasa_c[pa_idx] < sasa_cutoff) & (d[pa_idx] > sasa_cutoff)
        unsat += int((buried_a & ~has_partner_a).sum())
    if pb_idx.size > 0:
        buried_b = (sasa_c[pb_idx] < sasa_cutoff) & (d[pb_idx] > sasa_cutoff)
        unsat += int((buried_b & ~has_partner_b).sum())
    return unsat
