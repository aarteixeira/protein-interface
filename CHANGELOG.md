# Changelog

## Unreleased

## [0.1.2] — 2026-06-02

### Added

- Optional `protein_interface.openmm` module for OpenMM-backed structure
  minimization, potential energy calculation, and MM-GBSA-style binding-energy
  scoring between two chain groups.
- `calculate_sampled_gbsa_binding_energy()` for slower MD-sampled MM-GBSA
  scoring over trajectory frames, with a runtime warning that GPU execution is
  preferred for production use.
- `short`, `medium`, and `long` sampled-GBSA presets. `short` preserves the
  current default, while `medium` and `long` provide ns-scale protocols for
  more meaningful GPU runs.
- `environment-gpu.yml` for Linux GPU OpenMM installs, pinning CUDA 12.4 and
  OpenMM 8.2-compatible packages to avoid `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
  on NVIDIA 550-series drivers.
- README notes for a `gnode1` CUDA smoke comparison between single-structure
  and default sampled GBSA, including runtime and convergence caveats.
- `openmm` optional dependency extra. The base package still imports without
  OpenMM installed.
- The OpenMM helpers keep standard amino-acid residues and exclude waters or
  non-protein residues before setup; missing heavy atoms still surface as
  OpenMM template errors.
- Vendored, parity-tested `sc-rs` implementation with deterministic spatial
  indexing for Lawrence-Colman shape complementarity.
- Batched SC calculation through `compute_sc_batch()` and `analyze_batch()`.
- Rust Shrake-Rupley SASA kernel with spatial neighbor filtering and Rayon
  parallelism across atoms and batches.
- Rust spatial contact enumeration for `atomic_contacts`, interface-residue
  sets, charge summaries, gly/pro fraction, and PRODIGY contact bins.
- Rust salt-bridge residue-pair counting that preserves the public
  `salt_bridges()` definition.
- Benchmarks for SC, SASA, FreeSASA comparison, and contact-family metrics.
- Gated performance tests for SC, SASA, and contact-family optimized paths.

### Fixed

- `benchmark/speed.py` imports `from_boltzgen_structure` from the intended
  internal IO module.

## [0.1.1] — 2026-05-30

### Added

- `analyze()` and `analyze_batch()` now accept `metrics={...}` and
  `skip_metrics={...}` to avoid computing unneeded metrics. Disabled result
  fields are returned as `None`.

### Fixed

- Numeric parameters for Python helpers and Rust kernels now reject non-finite
  or negative radii/cutoffs instead of letting invalid values reach metric
  calculations.
- Public `AtomArrays` inputs now reject internal side-qualified residue IDs.

## [0.1.0] — 2026-05-25

### Added

- `compute_sc()`: low-level PyO3 binding to `sc-rs`'s `ScCalculator`, accepting
  pre-parsed atom coordinate arrays.
- `ScResult`: read-only Python class exposing `sc`, `median_distance`,
  `trimmed_area`, `atoms_a`, `atoms_b`.
- `from_pdb()` / `from_structure()`: biopython-based PDB/mmCIF parsing that
  mirrors the filtering logic of the `sc-rs` CLI exactly (ATOM-only, altloc A/'
  ' , heavy atoms only by default).
- `score_many()`: multiprocessing batch scorer returning a `pd.DataFrame`;
  per-file exceptions are caught and reported in a `status`/`error` column.
- Full test suite: unit tests for the Rust core, IO layer, and parity tests
  comparing `protein_interface` output against the `sc-rs` CLI binary.
- Benchmark script (`benchmark/speed.py`).
- Pinned to `sc-rs` v1.0.0 via git tag.
