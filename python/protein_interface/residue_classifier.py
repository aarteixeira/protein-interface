"""Per-residue interface classification for a protein complex.

Given one complex structure, every residue of every chain is labelled as one of:

    interface       buries surface on binding OR in heavy-atom contact with
                    another group (``combine="and"`` to require both)
    near_interface  not interface, but a side-chain heavy atom lies within
                    ``near_cutoff`` Å (atom-graph geodesic, same chain) of an
                    interface residue's side-chain heavy atom — opt-in via
                    ``near_interface=True``; off by default
    core            not interface/near, and buried in the isolated chain
                    (monomer relative SASA < ``core_rsasa``)
    non_interface   everything else (exposed surface)

Definitions
-----------
A *group* is a set of chains treated as one binding partner (the "monomer"). By
default each chain is its own group, so dSASA is chain-vs-rest and contact is
chain-vs-any-other-chain. Pass ``groups`` to keep a multi-chain partner together,
e.g. ``groups=[["H", "L"], ["A"]]`` for an antibody Fab (H+L) vs antigen (A).

Presets (``mode``): "strict" = dSASA > 3 Å² OR ≤ 5 Å contact (default);
"lenient" = dSASA > 0 Å² OR ≤ 7 Å contact. Both use the OR rule. Override any of
``dsasa_threshold`` / ``contact_cutoff`` / ``combine`` for custom criteria.

interface (``combine`` selects how the two criteria are merged; default "or"):
    * per-residue ``dSASA > dsasa_threshold`` Å², where
      ``dSASA = SASA(group in isolation) − SASA(group in the full complex)``
      summed over the residue's heavy atoms; and/or
    * any heavy atom within ``contact_cutoff`` Å of a heavy atom in another group.

    ``combine="or"`` (default) labels a residue interface if either holds — a
    residue can bury surface without a heavy atom inside the contact shell, or
    contact the partner without losing much accessibility. ``combine="and"``
    requires both.

near_interface (geodesic):
    Shortest-path distance over a graph of the chain's heavy atoms (edges between
    atoms within ``edge_cutoff`` Å, weighted by Euclidean distance), via scipy
    Dijkstra. The path routes through the molecule rather than across solvent, so
    residues close in space but separated by a cleft are not counted. At the
    default ``near_cutoff == edge_cutoff`` (4 Å) this closely matches a plain 4 Å
    Euclidean rule; lowering ``edge_cutoff`` or raising ``near_cutoff`` makes the
    geodesic behaviour more pronounced. Glycine has no side-chain heavy atom, so
    its CA is used as the side-chain proxy.

core:
    Relative SASA of the residue in the *isolated* chain (sum of per-atom SASA
    divided by the Tien et al. 2013 reference for the residue type) below
    ``core_rsasa``. Captures intrinsic burial independent of the partner.
    Non-standard residues (no reference) are never labelled core.

Outputs: a tidy table (one row per residue) writable to Excel — with two residue
numberings, ``resseq`` (author/PDB number, as in the structure) and ``seq_index``
(per-chain sequential index from 1 that counts structure gaps) — plus an optional
3Dmol.js HTML viewer (structure embedded, library from CDN) coloured by category.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from protein_interface._core import compute_sasa
from protein_interface.interface import (
    BACKBONE_ATOMS,
    REFERENCE_SASA,
    AtomArrays,
    _validate_nonnegative_finite,
    load_atoms,
)
from protein_interface.io import _load_structure

CATEGORIES = ("interface", "near_interface", "core", "non_interface")

# Named interface presets. The dSASA test is strictly greater-than, so "strict"
# means dSASA > 3 Å² (≈ ≥ 3 for continuous areas) and "lenient" means dSASA > 0.
# Both use the OR (union) rule. Pass dsasa_threshold / contact_cutoff / combine
# to classify_residues to override a preset value.
MODES: dict[str, dict] = {
    "strict": {"dsasa_threshold": 3.0, "contact_cutoff": 5.0, "combine": "or"},
    "lenient": {"dsasa_threshold": 0.0, "contact_cutoff": 7.0, "combine": "or"},
}

# Colour-blind-safe defaults (Okabe-Ito-ish); override via classify args / to_html.
DEFAULT_COLORS: dict[str, str] = {
    "interface": "#d62728",      # red
    "near_interface": "#ff7f0e",  # orange
    "core": "#1f77b4",           # blue
    "non_interface": "#d9d9d9",  # light grey
}

ResidueId = tuple[str, int, str]  # (chain, resseq, icode)


@dataclass
class ResidueRecord:
    group: str                  # group label this residue's chain belongs to
    chain: str
    resseq: int                 # author/PDB residue number (as in the structure)
    icode: str
    seq_index: int              # per-chain sequential index from 1, counting structure gaps
    resname: str
    category: str               # one of CATEGORIES
    dsasa: float                # Å², monomer(group) − complex, summed over residue
    min_interchain_dist: float  # Å, nearest heavy atom in any other group
    geodesic_to_interface: float  # Å, graph-geodesic dist (same chain) to nearest interface
                                  # side chain; ~0 for interface residues, NaN if none in chain
    monomer_rsasa: float        # fraction 0–1 (isolated-chain SASA / reference); NaN if non-standard
    complex_rsasa: float        # fraction 0–1 (complex SASA / reference); NaN if non-standard


@dataclass
class ResidueClassification:
    """Result of :func:`classify_residues`. ``records`` is one per residue."""

    records: list[ResidueRecord]
    structure_path: str
    params: dict = field(default_factory=dict)

    # ── tabular ──────────────────────────────────────────────────────────────
    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "group": r.group,
                    "chain": r.chain,
                    "resseq": r.resseq,
                    "icode": r.icode,
                    "seq_index": r.seq_index,
                    "resname": r.resname,
                    "category": r.category,
                    "dsasa": r.dsasa,
                    "min_interchain_dist": r.min_interchain_dist,
                    "geodesic_to_interface": r.geodesic_to_interface,
                    "monomer_rsasa": r.monomer_rsasa,
                    "complex_rsasa": r.complex_rsasa,
                }
                for r in self.records
            ]
        )

    def counts(self) -> dict[str, dict[str, int]]:
        """Per-chain category counts: ``{chain: {category: n}}`` plus ``"ALL"``."""
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            out.setdefault(r.chain, {c: 0 for c in CATEGORIES})[r.category] += 1
        total = {c: 0 for c in CATEGORIES}
        for chain_counts in out.values():
            for c in CATEGORIES:
                total[c] += chain_counts[c]
        out["ALL"] = total
        return out

    def to_excel(self, path: str | Path, with_summary: bool = True) -> Path:
        import pandas as pd

        path = Path(path)
        df = self.to_dataframe().sort_values(["chain", "resseq", "icode"]).reset_index(drop=True)
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="residues", index=False)
            if with_summary:
                counts = self.counts()
                summary = pd.DataFrame(
                    [{"chain": ch, **counts[ch]} for ch in counts]
                )[["chain", *CATEGORIES]]
                summary.to_excel(xw, sheet_name="summary", index=False)
        return path

    def to_html(
        self,
        path: str | Path,
        colors: dict[str, str] | None = None,
        title: str | None = None,
    ) -> Path:
        path = Path(path)
        html = _build_html(
            Path(self.structure_path),
            self.records,
            colors or DEFAULT_COLORS,
            title or Path(self.structure_path).stem,
        )
        path.write_text(html, encoding="utf-8")
        return path


# ── core computation ──────────────────────────────────────────────────────────

def _sidechain_local_indices(
    res_atoms: dict[ResidueId, list[int]],
    atom_names: list[str],
    global_to_local: dict[int, int],
) -> dict[ResidueId, list[int]]:
    """Side-chain heavy-atom local indices per residue (CA fallback for Gly)."""
    out: dict[ResidueId, list[int]] = {}
    for rid, idxs in res_atoms.items():
        sc = [i for i in idxs if atom_names[i] not in BACKBONE_ATOMS]
        if not sc:  # glycine (or backbone-only fragment): use CA as proxy
            sc = [i for i in idxs if atom_names[i] == "CA"]
        out[rid] = [global_to_local[i] for i in sc if i in global_to_local]
    return out


def _geodesic_min_distances(
    sub_coords: np.ndarray,
    sources: list[int],
    edge_cutoff: float,
) -> np.ndarray:
    """Per-node shortest-path distance to the nearest source over a graph of the
    chain's heavy atoms (edges between atoms within ``edge_cutoff`` Å, weighted by
    Euclidean distance). Unreachable nodes are ``inf``."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    tree = cKDTree(sub_coords)
    graph = tree.sparse_distance_matrix(tree, edge_cutoff, output_type="coo_matrix")
    return dijkstra(graph, directed=False, indices=sources, min_only=True)


def classify_residues(
    structure_path: str | Path,
    groups: list[list[str]] | None = None,
    *,
    chains: list[str] | None = None,
    mode: str = "strict",
    dsasa_threshold: float | None = None,
    contact_cutoff: float | None = None,
    combine: str | None = None,
    near_cutoff: float = 4.0,
    near_interface: bool = False,
    core_rsasa: float = 0.25,
    edge_cutoff: float | None = None,
    probe_radius: float = 1.4,
    n_points: int = 960,
    include_hetatm: bool = False,
    model: int = 0,
    strict: bool = True,
) -> ResidueClassification:
    """Classify every residue of a complex as interface / near_interface / core /
    non_interface.

    Args:
        structure_path:   path to a .pdb / .ent / .cif / .mmcif complex.
        groups:           chain groupings treated as binding partners (monomers).
                          None ⇒ each chain is its own group. Every loaded chain
                          must appear in exactly one group.
        chains:           restrict analysis to this subset of chains (e.g.
                          ["B", "C"]); None ⇒ all chains in the structure. dSASA
                          and contact are computed only among the loaded chains.
        mode:             interface preset (default "strict"): "strict" =
                          dSASA > 3 Å² OR ≤ 5 Å contact; "lenient" =
                          dSASA > 0 Å² OR ≤ 7 Å contact. See MODES.
        dsasa_threshold:  override the mode's dSASA threshold (Å²); None ⇒ use mode.
        contact_cutoff:   override the mode's contact cutoff (Å); None ⇒ use mode.
        combine:          override how the two criteria merge — "or" (union) or
                          "and" (both); None ⇒ use mode (both modes use "or").
        near_cutoff:      max geodesic distance (Å) to an interface side chain.
        near_interface:   enable the geodesic near-interface category (default
                          False). When False the geodesic step is skipped, so no
                          residue is labelled near_interface and
                          geodesic_to_interface is NaN.
        core_rsasa:       monomer relative-SASA fraction below which a buried,
                          non-interface residue is labelled core (0.25 = 25 %).
        edge_cutoff:      contact distance (Å) for atom-graph geodesic edges;
                          defaults to ``near_cutoff``.
        probe_radius, n_points: SASA parameters (960 points for dSASA accuracy).
        include_hetatm:   include HETATM residues (waters/ligands) — default False.
        model:            model index (0-based).
        strict:           validate SASA radii / inputs (see interface.analyze).

    Returns:
        ResidueClassification with one ResidueRecord per residue.
    """
    # Resolve the mode preset; explicit threshold/contact/combine args override it.
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    preset = MODES[mode]
    if dsasa_threshold is None:
        dsasa_threshold = preset["dsasa_threshold"]
    if contact_cutoff is None:
        contact_cutoff = preset["contact_cutoff"]
    if combine is None:
        combine = preset["combine"]

    for name, val in (
        ("dsasa_threshold", dsasa_threshold),
        ("contact_cutoff", contact_cutoff),
        ("near_cutoff", near_cutoff),
        ("core_rsasa", core_rsasa),
    ):
        _validate_nonnegative_finite(name, val)
    if edge_cutoff is None:
        edge_cutoff = near_cutoff
    _validate_nonnegative_finite("edge_cutoff", edge_cutoff)
    if combine not in ("or", "and"):
        raise ValueError(f"combine must be 'or' or 'and', got {combine!r}")

    structure_path = Path(structure_path)

    # All chains that carry ≥1 heavy ATOM record.
    structure = _load_structure(structure_path)
    models = list(structure.get_models())
    if model >= len(models):
        raise ValueError(f"model index {model} out of range ({len(models)} model(s))")
    all_struct_chains = [ch.id for ch in models[model].get_chains()]
    if chains is None:
        load_chains = all_struct_chains
    else:
        load_chains = [str(c) for c in chains]
        unknown = [c for c in load_chains if c not in all_struct_chains]
        if unknown:
            raise ValueError(
                f"requested chains not in structure: {unknown}. "
                f"Available: {all_struct_chains}"
            )
    atoms: AtomArrays = load_atoms(
        structure_path, chains=load_chains, model=model, include_hetatm=include_hetatm
    )
    if not atoms.coords:
        raise ValueError("no heavy atoms loaded from structure")
    present_chains = list(dict.fromkeys(rid[0] for rid in atoms.residue_ids))

    # Resolve groups.
    if groups is None:
        groups = [[c] for c in present_chains]
    groups = [[str(c) for c in g] for g in groups]
    chain_to_group: dict[str, int] = {}
    for gi, g in enumerate(groups):
        for c in g:
            if c in chain_to_group:
                raise ValueError(f"chain {c!r} appears in more than one group")
            chain_to_group[c] = gi
    missing = [c for c in present_chains if c not in chain_to_group]
    if missing:
        raise ValueError(
            f"chains present in structure but not assigned to any group: {missing}. "
            f"Present chains: {present_chains}"
        )
    group_label = ["+".join(g) for g in groups]

    coords = np.asarray(atoms.coords, dtype=np.float64)
    n_atoms = len(coords)
    atom_group = np.array([chain_to_group[rid[0]] for rid in atoms.residue_ids], dtype=np.int64)

    # SASA in the full complex (one call) and per group in isolation.
    sasa_complex = np.asarray(
        compute_sasa(atoms.coords, atoms.atom_names, atoms.residue_names, probe_radius, n_points),
        dtype=np.float64,
    )
    sasa_iso = np.empty(n_atoms, dtype=np.float64)
    for gi in range(len(groups)):
        mask = atom_group == gi
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        s = compute_sasa(
            [atoms.coords[i] for i in idx],
            [atoms.atom_names[i] for i in idx],
            [atoms.residue_names[i] for i in idx],
            probe_radius,
            n_points,
        )
        sasa_iso[idx] = np.asarray(s, dtype=np.float64)
    dsasa_atom = sasa_iso - sasa_complex

    # Min heavy-atom distance from each atom to atoms in *other* groups.
    from scipy.spatial import cKDTree

    min_other = np.full(n_atoms, np.inf, dtype=np.float64)
    for gi in range(len(groups)):
        mask = atom_group == gi
        other = ~mask
        if not mask.any() or not other.any():
            continue
        tree = cKDTree(coords[other])
        d, _ = tree.query(coords[mask], k=1)
        min_other[mask] = d

    # Group atoms by residue (preserve first-seen order).
    res_atoms: dict[ResidueId, list[int]] = {}
    res_name: dict[ResidueId, str] = {}
    res_order: list[ResidueId] = []
    for i, rid in enumerate(atoms.residue_ids):
        if rid not in res_atoms:
            res_atoms[rid] = []
            res_name[rid] = atoms.residue_names[i]
            res_order.append(rid)
        res_atoms[rid].append(i)

    # Per-chain sequential index from 1, counting structure gaps. Walk residues
    # in (resseq, icode) order: a jump in the author numbering (e.g. 50 -> 53)
    # adds the missing residues to the count; an insertion code (same resseq,
    # next icode) advances by one. The first observed residue of each chain is 1.
    res_seq_index: dict[ResidueId, int] = {}
    for chain in present_chains:
        chain_rids = sorted(
            (rid for rid in res_order if rid[0] == chain), key=lambda r: (r[1], r[2])
        )
        prev: ResidueId | None = None
        n = 0
        for rid in chain_rids:
            if prev is None:
                n = 1
            elif rid[1] == prev[1]:
                n += 1  # insertion code (same author number)
            else:
                n += max(1, rid[1] - prev[1])  # gap-aware; >=1 guards non-increasing numbering
            res_seq_index[rid] = n
            prev = rid

    # Per-residue aggregates.
    res_dsasa: dict[ResidueId, float] = {}
    res_min_other: dict[ResidueId, float] = {}
    res_mono_rsasa: dict[ResidueId, float] = {}
    res_cplx_rsasa: dict[ResidueId, float] = {}
    for rid in res_order:
        idxs = res_atoms[rid]
        res_dsasa[rid] = float(dsasa_atom[idxs].sum())
        res_min_other[rid] = float(min_other[idxs].min())
        ref = REFERENCE_SASA.get(res_name[rid])
        if ref:
            res_mono_rsasa[rid] = float(sasa_iso[idxs].sum()) / ref
            res_cplx_rsasa[rid] = float(sasa_complex[idxs].sum()) / ref
        else:
            res_mono_rsasa[rid] = float("nan")
            res_cplx_rsasa[rid] = float("nan")

    # interface = buries surface on binding and/or is in contact with another
    # group; `combine` selects the union ("or", default) or intersection ("and").
    def _is_interface(rid: ResidueId) -> bool:
        buried = res_dsasa[rid] > dsasa_threshold
        contact = res_min_other[rid] <= contact_cutoff
        return (buried and contact) if combine == "and" else (buried or contact)

    interface_rids: set[ResidueId] = {rid for rid in res_order if _is_interface(rid)}

    # near_interface — per chain, atom-graph geodesic from interface side chains.
    # res_geodesic[rid]: shortest-path distance (same chain) from the residue's
    # side chain to the nearest interface residue's side chain. ~0 for interface
    # residues (their atoms are the sources); NaN when the chain has no interface
    # residue or the residue is unreachable. Skipped entirely if near_interface
    # is False.
    res_geodesic: dict[ResidueId, float] = {}
    near_rids: set[ResidueId] = set()
    if near_interface:
        for chain in present_chains:
            chain_atom_idx = [i for i in range(n_atoms) if atoms.residue_ids[i][0] == chain]
            if not chain_atom_idx:
                continue
            global_to_local = {g: l for l, g in enumerate(chain_atom_idx)}
            sub_coords = coords[chain_atom_idx]
            chain_rids = [rid for rid in res_order if rid[0] == chain]
            sidechain_local = _sidechain_local_indices(
                {rid: res_atoms[rid] for rid in chain_rids}, atoms.atom_names, global_to_local
            )
            sources: list[int] = []
            for rid in chain_rids:
                if rid in interface_rids:
                    sources.extend(sidechain_local.get(rid, []))
            if not sources:
                for rid in chain_rids:
                    res_geodesic[rid] = float("nan")
                continue
            dist = _geodesic_min_distances(sub_coords, sources, edge_cutoff)
            for rid in chain_rids:
                locs = sidechain_local.get(rid, [])
                g = float(np.min(dist[locs])) if locs else float("inf")
                res_geodesic[rid] = g if math.isfinite(g) else float("nan")

        near_rids = {
            rid
            for rid in res_order
            if rid not in interface_rids
            and not math.isnan(res_geodesic.get(rid, float("nan")))
            and res_geodesic[rid] <= near_cutoff
        }
    else:
        res_geodesic = {rid: float("nan") for rid in res_order}

    # Assign categories with precedence interface > near > core > non_interface.
    records: list[ResidueRecord] = []
    for rid in res_order:
        chain, resseq, icode = rid
        if rid in interface_rids:
            cat = "interface"
        elif rid in near_rids:
            cat = "near_interface"
        elif res_mono_rsasa[rid] == res_mono_rsasa[rid] and res_mono_rsasa[rid] < core_rsasa:
            cat = "core"
        else:
            cat = "non_interface"
        records.append(
            ResidueRecord(
                group=group_label[chain_to_group[chain]],
                chain=chain,
                resseq=resseq,
                icode=icode,
                seq_index=res_seq_index[rid],
                resname=res_name[rid],
                category=cat,
                dsasa=round(res_dsasa[rid], 3),
                min_interchain_dist=round(res_min_other[rid], 3),
                geodesic_to_interface=round(res_geodesic.get(rid, float("nan")), 3),
                monomer_rsasa=round(res_mono_rsasa[rid], 4),
                complex_rsasa=round(res_cplx_rsasa[rid], 4),
            )
        )

    params = {
        "mode": mode,
        "dsasa_threshold": dsasa_threshold,
        "contact_cutoff": contact_cutoff,
        "combine": combine,
        "near_cutoff": near_cutoff,
        "near_interface": near_interface,
        "core_rsasa": core_rsasa,
        "edge_cutoff": edge_cutoff,
        "probe_radius": probe_radius,
        "n_points": n_points,
        "chains": present_chains,
        "groups": group_label,
        "include_hetatm": include_hetatm,
    }
    return ResidueClassification(records, str(structure_path), params)


# ── 3Dmol.js HTML viewer ──────────────────────────────────────────────────────

_THREEDMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"


def _fmt_for_suffix(suffix: str) -> str:
    return "mmcif" if suffix.lower() in (".cif", ".mmcif") else "pdb"


def _build_html(
    structure_path: Path,
    records: list[ResidueRecord],
    colors: dict[str, str],
    title: str,
) -> str:
    import json

    text = structure_path.read_text(encoding="utf-8", errors="replace")
    fmt = _fmt_for_suffix(structure_path.suffix)

    # category -> chain -> [resi]
    sel: dict[str, dict[str, list[int]]] = {c: {} for c in CATEGORIES}
    for r in records:
        sel[r.category].setdefault(r.chain, []).append(r.resseq)

    counts = {c: sum(len(v) for v in sel[c].values()) for c in CATEGORIES}
    colors_js = {c: "0x" + colors[c].lstrip("#") for c in CATEGORIES}

    legend = "".join(
        f'<div class="row"><span class="sw" style="background:{colors[c]}"></span>'
        f"{c.replace('_', ' ')} ({counts[c]})</div>"
        for c in CATEGORIES
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — interface residues</title>
<script src="{_THREEDMOL_CDN}"></script>
<style>
  html,body{{margin:0;height:100%;font-family:system-ui,sans-serif}}
  #viewer{{width:100vw;height:100vh;position:relative}}
  #legend{{position:absolute;top:10px;left:10px;z-index:10;background:rgba(255,255,255,.92);
           padding:10px 12px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.2);font-size:13px}}
  #legend h1{{font-size:13px;margin:0 0 6px}}
  #legend .row{{display:flex;align-items:center;margin:2px 0}}
  #legend .sw{{width:14px;height:14px;border-radius:3px;margin-right:7px;border:1px solid #0003}}
</style></head>
<body>
<div id="legend"><h1>{title}</h1>{legend}</div>
<div id="viewer"></div>
<script>
const STRUCT = {json.dumps(text)};
const SEL = {json.dumps(sel)};
const COLORS = {json.dumps(colors_js)};
const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
viewer.addModel(STRUCT, {json.dumps(fmt)});
viewer.setStyle({{}}, {{cartoon: {{color: COLORS["non_interface"]}}}});
for (const cat of Object.keys(SEL)) {{
  for (const chain of Object.keys(SEL[cat])) {{
    viewer.setStyle(
      {{chain: chain, resi: SEL[cat][chain]}},
      {{cartoon: {{color: parseInt(COLORS[cat])}}}}
    );
  }}
}}
viewer.zoomTo();
viewer.render();
</script>
</body></html>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="protein-interface-residues",
        description="Classify every residue of a complex as interface / near_interface / core / non_interface.",
    )
    p.add_argument("structure", help="path to .pdb/.ent/.cif/.mmcif complex")
    p.add_argument("-o", "--excel", help="output .xlsx path (default: <structure>.residues.xlsx)")
    p.add_argument("--html", nargs="?", const="", help="write 3Dmol HTML viewer (optional path)")
    p.add_argument(
        "-g", "--group", action="append", default=None,
        help="chain group as comma-separated chains, repeatable, e.g. -g H,L -g A. "
             "Default: each chain is its own group.",
    )
    p.add_argument("--chains", help="restrict to a comma-separated subset of chains, e.g. B,C (default: all)")
    p.add_argument("--mode", choices=["strict", "lenient"], default="strict",
                   help="interface preset: strict=dSASA>3 or <=5A, lenient=dSASA>0 or <=7A (default: strict)")
    p.add_argument("--dsasa-threshold", type=float, default=None, help="override mode's dSASA threshold (A^2)")
    p.add_argument("--contact-cutoff", type=float, default=None, help="override mode's contact cutoff (A)")
    p.add_argument("--combine", choices=["or", "and"], default=None,
                   help="override how dSASA and contact criteria merge (default: from mode)")
    p.add_argument("--near-cutoff", type=float, default=4.0)
    p.add_argument("--near-interface", action="store_true",
                   help="enable the geodesic near-interface category (off by default)")
    p.add_argument("--core-rsasa", type=float, default=0.25)
    p.add_argument("--edge-cutoff", type=float, default=None)
    p.add_argument("--n-points", type=int, default=960)
    p.add_argument("--include-hetatm", action="store_true")
    args = p.parse_args(argv)

    groups = [g.split(",") for g in args.group] if args.group else None
    chains = args.chains.split(",") if args.chains else None
    result = classify_residues(
        args.structure,
        groups=groups,
        chains=chains,
        mode=args.mode,
        dsasa_threshold=args.dsasa_threshold,
        contact_cutoff=args.contact_cutoff,
        combine=args.combine,
        near_cutoff=args.near_cutoff,
        near_interface=args.near_interface,
        core_rsasa=args.core_rsasa,
        edge_cutoff=args.edge_cutoff,
        n_points=args.n_points,
        include_hetatm=args.include_hetatm,
    )

    stem = Path(args.structure)
    excel = Path(args.excel) if args.excel else stem.with_suffix(".residues.xlsx")
    result.to_excel(excel)
    print(f"wrote {excel}")
    if args.html is not None:
        html = Path(args.html) if args.html else stem.with_suffix(".residues.html")
        result.to_html(html)
        print(f"wrote {html}")

    counts = result.counts()["ALL"]
    print("counts: " + ", ".join(f"{c}={counts[c]}" for c in CATEGORIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
