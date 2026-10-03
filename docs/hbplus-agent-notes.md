# Handoff for HBPLUS-format work

Read [the scope and validation report](hbplus.md) before changing this engine.
`src/hbplus/` is independent Rust code; `src/hbonds.rs` remains the existing
distance-only interface metric. The CLI and Python API share the Rust engine.

The reproducible external reference is ProtonPottsMPNN commit
`09682abfa7d20e0abcdeea0490b7a4b1c190aee3`. Its saved bond table covers contacts
incident on His, Asp and Glu. Do not interpret it as a complete HBPLUS oracle.
The original HBPLUS program was not run, and its C implementation was not used.

Before changing hydrogen geometry, run `cargo test --locked` and the Python
suite. Then run `benchmark/hbplus_reference.py` in a fresh output directory.
Changes affecting contacts or geometry also need `benchmark/hbplus_ev6.py`;
its manifest prevents resuming checkpoints under a different protocol.
Keep the calibration and evaluation results distinct. The initial held-out
corpus is now observed: tuning against its five remaining discrepancies is
post-evaluation development and needs a new independent evaluation corpus.

Open issues are the two sulfur-acceptor contacts and the three internal-break
contacts listed in `hbplus-validation/holdout-discrepancies.json`. The ancillary
output fields and full HBPLUS option set remain unqualified. Do not change the
documented compatibility claim to "exact" based only on EV6 label agreement.

The new implementation follows this repository's MIT license. Software
provenance records do not decide patent questions about methods or sequences.
