# protein_interface [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, geometry-based interface analysis for protein–protein complexes. Computes
SASA, buried surface area, H-bonds, salt bridges, π-π and cation-π contacts,
buried unsatisfied polars, interface shape (planarity, elongation, depth),
backbone/side-chain decomposition, charge complementarity, B-factor / pLDDT at
the interface, Lawrence-Colman shape complementarity, and more — about 30
metrics in total — in a single `analyze()` call.

Built for high-throughput design filtering: SASA runs in Rust with a spatial-
hash neighbour grid, the inner loops of `interface_residues` and
`buried_unsat_polar` are vectorised with numpy, and `analyze_batch()` pushes
all SASA work for N complexes into one Rayon-parallel Rust call (~4 ms per
structure at `n_points=92`).

Shape complementarity is delegated to the upstream [sc-rs](https://github.com/cytokineking/sc-rs)
crate via PyO3 — see [Shape complementarity](#shape-complementarity-sc) below
for details specific to that metric.

---

## Install

**Development (editable, re-run after Rust changes):**
```bash
pip install maturin
maturin develop --release
```

**Release / production:**
```bash
pip install protein-interface
```

---

## Quick start

```python
from protein_interface import load_atoms, analyze

a = load_atoms("complex.pdb", chains=["H"])
b = load_atoms("complex.pdb", chains=["A"])
r = analyze(a, b)

print(f"sc                   = {r.sc:.3f}")
print(f"dsasa                = {r.dsasa:.1f} Å²")
print(f"hbonds               = {r.hbonds}")
print(f"salt_bridges         = {r.salt_bridges}")
print(f"buried_unsat_polar   = {r.buried_unsat_polar}")
print(f"hotspots_a           = {r.hotspots_a}")
```

For a screen across many complexes use `analyze_batch()`:

```python
from protein_interface import analyze_batch

complexes = [(load_atoms(p, ["H"]), load_atoms(p, ["A"])) for p in pdbs]
results = analyze_batch(complexes)  # one InterfaceResult per complex
```

See [Interface analysis](#interface-analysis) for the full metric reference,
[Validation against FreeSASA](#validation-against-freesasa) for accuracy,
and [Performance](#performance) for timing on 200K-structure pipelines.

---

## Loading structures

Every metric in `analyze()` operates on `AtomArrays` — a dataclass holding
parallel lists of coordinates, atom names, residue names, residue IDs
(`(chain_id, resseq)`), and optional B-factors (= pLDDT for AlphaFold / Boltz /
BoltzGen outputs).

### From a PDB or mmCIF file

```python
from protein_interface import load_atoms

a = load_atoms("complex.pdb", chains=["H"])
b = load_atoms("complex.pdb", chains=["A"])
```

Accepts `.pdb`, `.ent`, `.cif`, and `.mmcif`. ATOM records only by default —
pass `include_hetatm=True` to include HETATM, `include_hydrogens=True` to keep
hydrogens, `model=N` to pick a model from a multi-model file. Altlocs are
selected as ' ' or 'A' (matches the `sc-rs` CLI).

### From in-memory coordinates

If your coordinates come from a simulation, generative model, or a parser this
package doesn't wrap, build the `AtomArrays` directly:

```python
from protein_interface import AtomArrays

a = AtomArrays(
    coords        = [[x1, y1, z1], [x2, y2, z2], ...],
    atom_names    = ["N", "CA", "C", ...],
    residue_names = ["ALA", "ALA", "ALA", ...],
    residue_ids   = [("A", 1), ("A", 1), ("A", 1), ...],   # (chain, resseq) per atom
    bfactors      = [80.5, 79.2, 78.6, ...],               # optional; pLDDT or B
)
```

### SC-only entry points (skip the AtomArrays step)

For users who only need Lawrence-Colman shape complementarity, file-loading
wrappers that go straight to `compute_sc()` are available. They return a
`ScResult` (`sc`, `median_distance`, `trimmed_area`, `atoms_a`, `atoms_b`):

| Function | Use when |
|---|---|
| `from_pdb(path, chains_a, chains_b)` | `.pdb` or `.cif` on disk |
| `from_structure(struct, chains_a, chains_b)` | biopython `Structure` already in memory |
| `from_biotite(atom_array, chains_a, chains_b)` | biotite `AtomArray` (e.g. from BoltzGen's analysis stack) |
| `from_boltzgen_structure(struct, chains_a, chains_b)` | BoltzGen `Structure` in memory |
| `from_boltzgen_refold(refold_cif, chains_a, chains_b)` | BoltzGen `refold_cif/*.cif` design |
| `score_many(paths, chains_a, chains_b, n_workers=8)` | Many files via `ProcessPoolExecutor`, returns a `pandas.DataFrame` |

For the full metric suite, load via `load_atoms()` and call `analyze()` (or
`analyze_batch()` for many complexes). SC is included by default in
`InterfaceResult.sc`; pass `include_sc=False` to skip it for batch speed.

> **BoltzGen tip:** only `refold_cif/*.cif` files have validated full-atom
> coordinates. `intermediate_designs/` is backbone-only;
> `intermediate_designs_inverse_folded/` has side chains but is pre-Boltz.
> Use refold_cif for any meaningful analysis.

---

## Interface analysis

In addition to the Lawrence-Colman SC value, the package computes a suite of
geometry-based interface metrics. No energy function is used — every metric is
derived from atom coordinates and identities, so results depend only on the
input structure (not on any force-field choice).

### Quick start

```python
from protein_interface import load_atoms, analyze

a = load_atoms("complex.pdb", chains=["H"])
b = load_atoms("complex.pdb", chains=["A"])
result = analyze(a, b)
print(result.dsasa, result.hbonds, result.hydrophobic_fraction)
```

`analyze()` runs every metric in a single SASA pass; individual metrics are
also callable on their own (e.g. `sasa(a)`, `pi_pi_contacts(a, b)`).

### Metrics

All distances are in Å. Atoms with no entry in the sc-rs radius table are
silently skipped (matches `compute_sc`).

| Field | Definition |
|---|---|
| `dsasa` | Buried surface area on binding: `SASA(A) + SASA(B) − SASA(A+B)`. Per-atom Shrake-Rupley with 960 sphere points and 1.4 Å probe by default. |
| `n_interface_a`, `n_interface_b` | Residues with at least one heavy atom within 5 Å of the other chain. |
| `aromatic_dsasa_fraction` | Fraction of `dsasa` contributed by PHE/TYR/TRP/HIS atoms. |
| `bhsa`, `bpsa`, `bcsa` | Buried hydrophobic / polar / charged surface area (Å²). Hydrophobic = C, S; polar = N, O; charged = Asp OD\*, Glu OE\*, Lys NZ, Arg NE/NH\*, His ND1/NE2. `bhsa + bpsa ≈ dsasa`; `bcsa ⊆ bpsa`. |
| `hydrophobic_fraction` | `bhsa / dsasa`. |
| `hbonds` | Cross-interface H-bond count. Donor (any backbone N except Pro, plus polar side-chain N/O of Ser/Thr/Tyr/Asn/Gln/Lys/Arg/His/Trp) and acceptor (backbone O/OXT, plus side-chain O/N of Asp/Glu/Asn/Gln/Ser/Thr/Tyr/His) atoms within 3.5 Å. Distance-only; H positions and angles are not used. |
| `hbond_density` | `100 × hbonds / dsasa`. Useful for normalising across interfaces of different sizes. |
| `salt_bridges` | Cross-interface anion–cation pairs within 4.0 Å (Barlow & Thornton 1983). Anions: Asp OD\*, Glu OE\*. Cations: Lys NZ, Arg NE/NH\*, His ND1/NE2. |
| `pi_pi` | Aromatic ring centroid pairs (PHE/TYR/TRP/HIS) on opposite sides within 7.0 Å and with absolute angle between ring normals ≤ 90° (accepts face-to-face and T-shaped; tighten with `angle_cutoff_deg`). Ring atoms follow McGaughey et al. 1998. |
| `cation_pi` | Lys NZ or Arg CZ within 6.0 Å of an aromatic centroid on the opposite chain (Gallivan & Dougherty 1999). |
| `buried_unsat_polar` | Polar atom (donor or acceptor) whose SASA in the complex is below 1.0 Å², whose dSASA is above 1.0 Å² (i.e. binding caused the burial), and that has no complementary polar partner within 3.5 Å on the opposite chain. Geometric count of buried unsatisfied polars; intra-chain partners are not considered. |
| `planarity_rmsd` | RMS perpendicular distance of interface atoms (per-atom dSASA ≥ 0.5 Å²) from their best-fit plane. Smaller = flatter interface. |
| `elongation` | σ1 / σ2 from PCA on interface-atom coordinates. ≥ 1.0; larger = more elongated patch. |
| `planarity_ratio` | σ3 / σ2 from the same PCA. ≤ 1; smaller = flatter. |
| `hotspots_a`, `hotspots_b` | Residues with per-residue dSASA ≥ 30 Å² (configurable). Sorted by descending dSASA. Heuristic threshold corresponding to large buried-area residues (Bogan & Thorn 1998). |
| `dsasa_a`, `dsasa_b` | Per-side buried surface area (Å²). Sums to `dsasa` to within rounding. |
| `asymmetry` | `|dsasa_a − dsasa_b| / max(dsasa_a, dsasa_b)`. 0 = perfectly symmetric, 1 = totally one-sided. |
| `atomic_contacts` | Heavy-atom pairs across the interface within 5 Å. Independent of SASA and chemistry; a raw density proxy. |
| `interface_depth` | Distance between the A-side and B-side interface centroids (Å). Atoms with per-atom dSASA ≥ 0.5 Å² contribute. Small for shallow planar interfaces, larger for concave/cradled binding modes. |
| `disulfides` | Cross-interface Cys SG–SG pairs within 2.5 Å. Rare outside engineered systems. |
| `gly_pro_fraction` | (Gly + Pro) count divided by total interface residues. Flexibility/kink proxy. |
| `bb_dsasa`, `sc_dsasa` | Backbone (N/CA/C/O/OXT) and side-chain buried surface area (Å²). Sums to `dsasa`. |
| `sidechain_fraction` | `sc_dsasa / dsasa`. Side-chain-dominated interfaces (high values) are typically more designable. |
| `charge_a`, `charge_b` | Net formal charge over interface residues per side (Asp/Glu = −1, Lys/Arg = +1, His = 0). |
| `charge_complementarity` | `−(charge_a × charge_b)`. Positive = opposing signs (electrostatically complementary), negative = repulsive, zero = at least one side uncharged. |
| `mean_bfactor_interface`, `min_bfactor_interface` | Mean and minimum of the B-factor column over atoms with per-atom dSASA ≥ 0.5 Å². For AlphaFold 2/3, ESM, Boltz and BoltzGen this column is pLDDT (0–100; higher = more confident). For X-ray structures it is the thermal B-factor (Å²). NaN if `AtomArrays.bfactors` is missing. |
| `prodigy_dg` | Predicted binding free energy in kcal/mol via the empirical IC/NIS model of Vangone & Bonvin 2015 (eLife 4:e07454). More negative = predicted tighter binder. Validated against the upstream `prodigy-prot` reference: on 1ZVH we get −10.45 vs upstream −11.27 (within PRODIGY's own ~1.9 kcal/mol RMSE). Counts cross-interface residue–residue contacts at 5.5 Å split by polarity class, plus a per-residue non-interacting-surface composition; both polarity tables (`PRODIGY_IC_CLASS` and `PRODIGY_NIS_CLASS`) mirror the upstream `aa_properties` module. |

### Atomic radii

SASA, dSASA, and the burial check for `buried_unsat_polar` all use the same
MS-style atomic radii embedded in `sc-rs` (a port of the CCP4 `sc` Fortran
table; Lawrence & Colman 1993). These are slightly smaller than Bondi van
der Waals radii and are tuned for molecular surface analysis. Absolute SASA
values are therefore not directly comparable to those reported by tools using
Bondi or NACCESS radii, but relative differences across complexes are
consistent.

### Defaults and parameters

`analyze()` accepts every threshold as a keyword argument. Defaults match
common literature values:

| Parameter | Default | Source |
|---|---|---|
| `probe_radius` | 1.4 Å | Standard water probe |
| `n_points` | 92 (in `analyze()`) / 960 (in standalone functions) | Shrake-Rupley sphere points. The orchestrator defaults to 92 for batch throughput (~1 % dSASA noise); single-metric helpers default to 960 for accuracy on ad-hoc measurement. |
| `interface_cutoff` | 5.0 Å | Standard heavy-atom contact distance |
| `hbond_cutoff` | 3.5 Å | Donor-acceptor heavy-atom distance |
| `salt_bridge_cutoff` | 4.0 Å | Barlow & Thornton 1983 |
| `pi_pi_distance` | 7.0 Å | McGaughey et al. 1998 |
| `cation_pi_distance` | 6.0 Å | Gallivan & Dougherty 1999 |
| `hotspot_threshold` | 30.0 Å² | Bogan & Thorn 1998 (approximate) |
| `unsat_sasa_cutoff` | 1.0 Å² | Conservative burial cutoff |

### Validation against FreeSASA

Our SASA kernel is checked against [FreeSASA](https://freesasa.github.io/)
on `tests/data/nb_ag_test.pdb` (a nanobody–lysozyme complex, 1ZVH, ~1860 heavy
atoms across both chains). Both libraries run Shrake-Rupley with probe
radius 1.4 Å and 960 sphere points. Radii differ — we use the MS-style radii
embedded in `sc-rs` (from CCP4 `sc`, Lawrence & Colman 1993); FreeSASA defaults
to ProtOr — so absolute values differ by a few percent. Algorithm quality
shows up in the correlation:

| Metric | Value |
|---|---|
| Total SASA, ours / FreeSASA | **1.014** (within 1.5 %) |
| Atom-level Pearson r | **0.9891** |
| Residue-level Pearson r | **0.9963** |
| Mean abs diff / residue | 1.77 Å² (residues are 50–200 Å²) |

The validation runs as a pytest module ([tests/test_vs_freesasa.py](tests/test_vs_freesasa.py));
install `freesasa` (`pip install freesasa`) to enable it.

### Performance vs FreeSASA

Same workload, same n_points, single thread:

| Workload | Our `compute_sasa` | FreeSASA | Ratio |
|---|---|---|---|
| Single call, n_points=92 | 27 ms | — | — |
| Single call, n_points=960 | 241 ms | 50 ms | 4.8× slower |
| FreeSASA Lee-Richards default | — | 58 ms | comparable to us at n=92 |

Per-call we're slower — FreeSASA has ~15 years of low-level tuning we don't
match. But **batch mode flips the comparison**: a single `compute_sasa_batch`
call (Rayon-parallel, GIL released) processes 50 copies of the complex faster
than FreeSASA's per-call loop by a wide margin:

| Workload (50 copies) | Time | Per structure |
|---|---|---|
| FreeSASA Lee-Richards loop (sequential) | 3217 ms | 64 ms |
| Our `compute_sasa_batch`, n=92, parallel | **200 ms** | **4 ms** |
| Our `compute_sasa_batch`, n=92, serial | 1341 ms | 27 ms |
| Our `compute_sasa_batch`, n=960, parallel | 1750 ms | 35 ms |

For the 200 000-structure use case this means SASA work drops from ~3.5 hours
(FreeSASA loop) to ~13 minutes (`compute_sasa_batch` at `n_points=92`,
single process, Rayon across cores). Pipelining file I/O via
`ProcessPoolExecutor` over chunks gets you to a few minutes total.

### What is *not* included

These would require an energy function and are deliberately out of scope:

- Force-field binding energies (require an energy function)
- Packing-density / void-volume scores (require a Connolly surface)
- Solvation-aware H-bond satisfaction (we use distance only)
- Secondary-structure-resolved metrics (no DSSP)

### Performance

Per `analyze()` call on a ~1500-atom complex (`tests/data/nb_ag_test.pdb`):

| `n_points` | Time / call | Suitable for |
|---|---|---|
| 92 (default) | ~90 ms | Batch screening of 10⁴–10⁵ designs |
| 960 | ~650 ms | Single-structure measurement |

SASA dominates at high `n_points`; the Rust kernel uses a spatial-hash
neighbour grid so cost is approximately linear in atom count. Distance
matrices for `interface_residues`, `atomic_contacts`, and `buried_unsat_polar`
are computed with numpy broadcasts.

### Batched processing

For sweeping many complexes at once, use `analyze_batch()`:

```python
from protein_interface import analyze_batch, load_atoms

complexes = [
    (load_atoms(p, ["H"]), load_atoms(p, ["A"])) for p in pdbs
]
results = analyze_batch(complexes)  # returns a list of InterfaceResult
```

`analyze_batch()` collects every SASA computation (3·N of them) into a single
Rust call that releases the GIL and runs across Rayon threads. On a single
process this gives **~2× speedup** over a Python loop calling `analyze()`
(SASA in parallel; per-complex Python orchestration still runs serially).

The same `compute_sasa_batch()` primitive is exposed for callers who need
just per-atom SASA across many structures.

### Composing with ProcessPoolExecutor

For maximum throughput on 10⁵+ designs, split your work into chunks and run
one `analyze_batch()` per process with `parallel=False` to avoid CPU
oversubscription:

```python
from concurrent.futures import ProcessPoolExecutor

def worker(chunk):
    complexes = [(load_atoms(p, ["H"]), load_atoms(p, ["A"])) for p in chunk]
    return analyze_batch(complexes, parallel=False)

with ProcessPoolExecutor(8) as pool:
    for chunk_results in pool.map(worker, chunks_of(pdbs, 100)):
        ...
```

`parallel=False` is also the right setting for `compute_sc()`,
`compute_sasa_batch()`, and any other Rayon-aware Rust call dispatched from
inside a pool worker.

### References

- Lawrence MC & Colman PM. *J Mol Biol* 234, 946–950 (1993). Shape complementarity.
- Shrake A & Rupley JA. *J Mol Biol* 79, 351–371 (1973). SASA.
- Barlow DJ & Thornton JM. *J Mol Biol* 168, 867–885 (1983). Salt bridges.
- McGaughey GB, Gagné M, Rappé AK. *J Biol Chem* 273, 15458–15463 (1998). π-stacking geometry.
- Gallivan JP & Dougherty DA. *PNAS* 96, 9459–9464 (1999). Cation-π.
- Bogan AA & Thorn KS. *J Mol Biol* 280, 1–9 (1998). Hot spots in protein interfaces.
- Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO. *PLoS ONE* 8, e80635 (2013). Maximum allowed solvent accessibilities of residues in proteins (reference SASA values).
- Vangone A & Bonvin AMJJ. *eLife* 4, e07454 (2015). PRODIGY: contacts-based ΔG predictor for protein–protein binding affinity.

---

## Calibration recipe for design filters

Establish per-metric thresholds from a set of known good interfaces, then apply
to design candidates. Works for any field on `InterfaceResult`, not just SC.

```python
import numpy as np
from protein_interface import analyze_batch, load_atoms

def score_all(paths, chains_a, chains_b, *, parallel=True):
    complexes = [
        (load_atoms(p, chains_a), load_atoms(p, chains_b)) for p in paths
    ]
    return analyze_batch(complexes, parallel=parallel)

# Baseline from natives — pick percentile thresholds per metric.
natives = score_all(native_pdbs, ["H"], ["A"])
sc_lo         = np.percentile([r.sc for r in natives], 5)
dsasa_lo      = np.percentile([r.dsasa for r in natives], 5)
unsats_hi     = np.percentile([r.buried_unsat_polar for r in natives], 95)
sidechain_lo  = np.percentile([r.sidechain_fraction for r in natives], 5)
prodigy_hi    = np.percentile([r.prodigy_dg for r in natives], 95)  # less negative = worse

# Filter designs that look at least as good as the 5th-percentile native.
designs = score_all(design_pdbs, ["H"], ["A"])
passing = [
    r for r in designs
    if r.sc >= sc_lo
    and r.dsasa >= dsasa_lo
    and r.buried_unsat_polar <= unsats_hi
    and r.sidechain_fraction >= sidechain_lo
    and r.prodigy_dg <= prodigy_hi
]
print(f"{len(passing)} / {len(designs)} designs pass all filters")
```

For low-confidence-aware filtering on AlphaFold / Boltz / BoltzGen outputs,
also gate on `r.min_bfactor_interface` — a single low-pLDDT atom in an
otherwise-confident interface usually marks a bad contact.

---

## What this is NOT

- **Not a force field.** Every metric is purely geometric. Binding-energy and packing-density scores require an energy function or a Connolly surface and are deliberately out of scope.
- **Not a reimplementation of SC.** Shape complementarity is delegated to `sc-rs`. If `compute_sc` returns a surprising value, verify against the upstream `sc-rs` CLI first.
- **Not a design filter on its own.** Threshold selection, chain naming, and ranking belong in the consuming pipeline. `analyze()` gives you the numbers; you decide what counts as good.

---

## Acknowledgments

- [sc-rs](https://github.com/cytokineking/sc-rs) (MIT) — the Rust implementation of the Lawrence-Colman SC algorithm that this package wraps for the `sc` metric.
- [FreeSASA](https://freesasa.github.io/) (MIT) — used as the reference for SASA validation.
- The geometric criteria for the interaction metrics follow standard references: Barlow & Thornton 1983 (salt bridges), McGaughey et al. 1998 (π-stacking), Gallivan & Dougherty 1999 (cation-π), Bogan & Thorn 1998 (hot spots), Lawrence & Colman 1993 (SC). Full citations are in [Interface analysis → References](#references).
