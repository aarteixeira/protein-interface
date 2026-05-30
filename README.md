# protein_interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`protein_interface` is a Python/Rust package for coordinate-based analysis of
protein-protein interfaces. It loads PDB or mmCIF structures, splits two sides
of an interface by chain ID, and returns geometry-derived metrics from one
`analyze()` call.

The package computes:

- solvent accessible surface area (SASA) and buried surface area (dSASA)
- Lawrence-Colman shape complementarity through the upstream
  [`sc-rs`](https://github.com/cytokineking/sc-rs) Rust crate
- distance-based H-bonds, salt bridges, aromatic contacts, cation-pi contacts,
  disulfides, and atom contacts
- buried unsatisfied polar atoms
- per-side burial, backbone/side-chain burial, hydrophobic/polar/charged burial,
  interface charge summaries, interface shape descriptors, B-factor or pLDDT
  summaries, buried-area hotspots, and a PRODIGY-style empirical dG estimate

All metrics are derived from atom coordinates, residue names, atom names, and
optional B-factors. This is not a force field.

## Install

From a source checkout:

```bash
python -m pip install maturin
maturin develop --release
```

For an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

The runtime Python dependencies declared in `pyproject.toml` are `numpy` and
`biopython`. Some tests and helper paths use optional packages:

- `freesasa` for SASA comparison tests
- `prodigy_prot` for optional upstream PRODIGY comparison tests
- `biotite` for biotite intake-path tests
- `pandas` for `score_many()`

## Quick Start

```python
from protein_interface import analyze, load_atoms

side_a = load_atoms("complex.pdb", chains=["H"])
side_b = load_atoms("complex.pdb", chains=["A"])

result = analyze(side_a, side_b)

print(result.sc)
print(result.dsasa)
print(result.hbonds)
print(result.salt_bridges)
print(result.buried_unsat_polar)
print(result.prodigy_dg)
```

For many complexes:

```python
from protein_interface import analyze_batch, load_atoms

complexes = [
    (load_atoms(path, ["H"]), load_atoms(path, ["A"]))
    for path in pdb_paths
]

results = analyze_batch(complexes)
```

`analyze_batch()` batches the SASA work into one Rust call and returns one
`InterfaceResult` per input complex.

## Loading Structures

`load_atoms()` accepts `.pdb`, `.ent`, `.cif`, and `.mmcif` files:

```python
from protein_interface import load_atoms

side_a = load_atoms("complex.cif", chains=["A"])
side_b = load_atoms("complex.cif", chains=["B"])
```

Defaults:

- ATOM records are included.
- HETATM records are skipped unless `include_hetatm=True`.
- Hydrogens are skipped unless `include_hydrogens=True`.
- Altloc `A` and blank altloc are accepted.
- `model=0` selects the first model.
- B-factors are stored in `AtomArrays.bfactors`; predicted-structure workflows
  often store pLDDT in that column, but callers must know their input source.

You can also construct atom arrays directly:

```python
from protein_interface import AtomArrays

side_a = AtomArrays(
    coords=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    atom_names=["N", "CA"],
    residue_names=["ALA", "ALA"],
    residue_ids=[("A", 1), ("A", 1)],
    bfactors=[80.0, 82.0],
)
```

Residue IDs must be `(chain_id, resseq)` or `(chain_id, resseq, insertion_code)`.

## SC-Only API

If you only need Lawrence-Colman shape complementarity, use the SC-only entry
points:

| Function | Input |
|---|---|
| `from_pdb(path, chains_a, chains_b)` | PDB or mmCIF path |
| `from_structure(structure, chains_a, chains_b)` | Biopython `Structure` |
| `from_biotite(atom_array, chains_a, chains_b)` | Biotite `AtomArray` |
| `from_boltzgen_refold(refold_cif, chains_a, chains_b)` | BoltzGen `refold_cif/*.cif` |
| `score_many(paths, chains_a, chains_b, n_workers=8)` | Many files; returns a `pandas.DataFrame` |

These functions return `ScResult` with `sc`, `median_distance`, `trimmed_area`,
`atoms_a`, and `atoms_b`.

For the full metric set, call `load_atoms()` and then `analyze()`.

## Validation Behavior

The public metric functions validate inputs before computing.

With the default `strict=True`, the package raises `ValueError` for empty atom
groups, malformed atom arrays, non-finite coordinates, invalid numeric
thresholds, unsupported SASA/SC radii, and SC failures.

With `strict=False`, unsupported-radius atoms follow the low-level SASA kernel
behavior and contribute zero SASA. SC failures return `NaN`. Use this only when
you explicitly want permissive screening.

## Main API

```python
result = analyze(
    side_a,
    side_b,
    probe_radius=1.4,
    n_points=92,
    interface_cutoff=5.0,
    hbond_cutoff=3.5,
    salt_bridge_cutoff=4.0,
    pi_pi_distance=7.0,
    cation_pi_distance=6.0,
    hotspot_threshold=30.0,
    unsat_sasa_cutoff=1.0,
    min_atom_dsasa_for_shape=0.5,
    include_sc=True,
    strict=True,
    skip_metrics=None,
    metrics=None,
)
```

`analyze()` defaults to `n_points=92` for faster screening. For higher-precision
SASA or dSASA measurement, pass `n_points=960`. Standalone SASA helpers default
to `n_points=960`.

Use `include_sc=False` when you need the non-SC metrics and want to skip the
separate SC calculation.

Use `skip_metrics={...}` to disable named `InterfaceResult` fields, or
`metrics={...}` to compute only a named subset. This avoids the corresponding
metric work where the calculation is independent. For example,
`analyze(a, b, skip_metrics={"prodigy_dg", "buried_unsat_polar"})` skips the
PRODIGY pass and buried-unsat scan, while `analyze(a, b, metrics={"hbonds"})`
does not run SASA or SC at all. Unknown metric names raise `ValueError`.
Disabled fields are returned as `None`.

## Metrics

All distances are in Angstroms. Surface areas are in square Angstroms.

| Field | Calculation | Do not infer |
|---|---|---|
| `sc` | Lawrence-Colman shape complementarity from `sc-rs`. | Not an energy term. |
| `dsasa` | `SASA(A) + SASA(B) - SASA(A+B)`. | Not directly comparable across different radii tables or atom-selection rules. |
| `dsasa_a`, `dsasa_b` | Per-side buried surface from the same dSASA array. | Not a binding-energy decomposition. |
| `asymmetry` | `abs(dsasa_a - dsasa_b) / max(dsasa_a, dsasa_b)`. | Does not identify energetic contribution. |
| `n_interface_a`, `n_interface_b` | Residues with a heavy atom within `interface_cutoff` of the other side. | Can include residues with little buried area. |
| `atomic_contacts` | Cross-interface heavy-atom pairs within `interface_cutoff`. | Chemistry-independent count. |
| `aromatic_dsasa_fraction` | dSASA fraction from PHE, TYR, TRP, and HIS atoms. | HIS protonation is not modeled. |
| `bhsa`, `bpsa`, `bcsa` | Buried surface split by simple atom-name classes. | Not full atom typing. |
| `hydrophobic_fraction` | `bhsa / dsasa`. | Unstable for very small interfaces. |
| `bb_dsasa`, `sc_dsasa` | Buried surface split by backbone atom names versus all other atoms. | Depends on standard atom naming. |
| `sidechain_fraction` | `sc_dsasa / dsasa`. | Not an energetic designability score. |
| `hbonds` | Cross-interface donor/acceptor heavy-atom pairs within `hbond_cutoff`. | Hydrogens, angles, protonation, and intrachain satisfaction are not evaluated. |
| `hbond_density` | `100 * hbonds / dsasa`. | Compare only under the same H-bond definition. |
| `salt_bridges` | Acidic/basic residue pairs with at least one qualifying atom contact within `salt_bridge_cutoff`. | Protonation and angular geometry are not modeled. |
| `pi_pi` | Aromatic ring centroid pairs within `pi_pi_distance` and the angle criterion. | Default angle cutoff is permissive. |
| `cation_pi` | Lys NZ or Arg CZ within `cation_pi_distance` of an aromatic centroid. | Orientation and electrostatic strength are not modeled. |
| `buried_unsat_polar` | Buried donor/acceptor atom with no cross-interface polar partner within `hbond_cutoff`. | Waters, ligands, hydrogens, angles, and intrachain partners are not evaluated. |
| `disulfides` | Cross-interface CYS SG-SG pairs within 2.5 A. | Does not read bond records or oxidation state. |
| `planarity_rmsd` | RMS distance of dSASA-selected interface atoms from their best-fit plane. | Sensitive to the dSASA threshold. |
| `elongation` | First/second singular-value ratio from interface-atom PCA. | Shape descriptor only. |
| `planarity_ratio` | Third/second singular-value ratio from the same PCA. | Shape descriptor only. |
| `interface_depth` | Distance between side-A and side-B centroids of dSASA-selected atoms. | Not a physical penetration depth. |
| `hotspots_a`, `hotspots_b` | Residues with per-residue dSASA at least `hotspot_threshold`. | Not alanine-scanning hotspots. |
| `gly_pro_fraction` | GLY/PRO fraction among contact-defined interface residues. | Not a dynamics measurement. |
| `charge_a`, `charge_b` | Net formal charge over contact-defined interface residues. | Termini, pH, and local electrostatics are not modeled. |
| `charge_complementarity` | `-(charge_a * charge_b)`. | Not Poisson-Boltzmann electrostatic complementarity. |
| `mean_bfactor_interface`, `min_bfactor_interface` | Mean/minimum B-factor over dSASA-selected interface atoms. | The package does not decide whether B-factor means crystallographic B or pLDDT. |
| `prodigy_dg` | PRODIGY-style empirical dG from intermolecular contacts and non-interacting-surface composition. | Not a force-field, relaxation, or solvation energy. |

## Batch Processing

`analyze_batch()` processes many in-memory complexes:

```python
results = analyze_batch(complexes, parallel=True)
```

When calling from a `ProcessPoolExecutor`, pass `parallel=False` so each worker
does not also spawn Rayon worker threads:

```python
from concurrent.futures import ProcessPoolExecutor
from protein_interface import analyze_batch, load_atoms

def worker(paths):
    complexes = [(load_atoms(p, ["H"]), load_atoms(p, ["A"])) for p in paths]
    return analyze_batch(complexes, parallel=False)

with ProcessPoolExecutor(8) as pool:
    for result_chunk in pool.map(worker, chunks):
        ...
```

For SC-only file scoring, `score_many()` returns one row per input path and
reports per-file failures in `status` and `error` columns.

## Tests

Run the base suite:

```bash
python -m pytest -q
```

Optional validation paths are enabled by installing their dependencies:

```bash
python -m pip install freesasa prodigy-prot biotite pandas
python -m pytest -q
```

The test suite includes:

- unit tests for Rust SASA, H-bond, salt-bridge, and validation behavior
- parser and intake-path equivalence tests for PDB, mmCIF, Biopython, Biotite,
  and BoltzGen-shaped structures
- FreeSASA comparison tests for SASA when `freesasa` is installed
- upstream `prodigy_prot` comparison tests when `prodigy_prot` is installed

## Scope

Included:

- coordinate-derived interface descriptors
- strict input validation by default
- PDB/mmCIF loading through Biopython
- Rust kernels for SASA, H-bond counts, salt-bridge atom-pair counts, and SC
- batched SASA execution through Rayon

Not included:

- force-field binding energies
- structure relaxation
- Poisson-Boltzmann electrostatics
- ligand, water, or intrachain H-bond satisfaction
- DSSP or secondary-structure-resolved metrics
- automatic threshold selection for design campaigns

## References and Attribution

- [`sc-rs`](https://github.com/cytokineking/sc-rs), MIT license. This package
  calls `sc-rs` for the Lawrence-Colman SC metric.
- Lawrence MC and Colman PM. Shape complementarity at protein/protein
  interfaces. *Journal of Molecular Biology* 234:946-950 (1993).
  DOI: [10.1006/jmbi.1993.1648](https://doi.org/10.1006/jmbi.1993.1648).
- Vangone A and Bonvin AMJJ. Contacts-based prediction of binding affinity in
  protein-protein complexes. *eLife* 4:e07454 (2015).
  DOI: [10.7554/eLife.07454](https://doi.org/10.7554/eLife.07454).
- The PRODIGY-style implementation is validated against the upstream
  `prodigy_prot` Python package in optional tests.
- [FreeSASA](https://freesasa.github.io/), MIT license, is used by optional
  SASA comparison tests.
- Other geometric defaults follow the references cited in the source and tests:
  Barlow and Thornton 1983 for salt bridges, McGaughey et al. 1998 for
  pi-stacking geometry, Gallivan and Dougherty 1999 for cation-pi contacts,
  Bogan and Thorn 1998 for hot-spot terminology, and Tien et al. 2013 for
  reference SASA values.
