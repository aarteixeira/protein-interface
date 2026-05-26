# Changelog

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
