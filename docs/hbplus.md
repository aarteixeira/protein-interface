# HBPLUS-format hydrogen-bond geometry

`protein-interface-hbplus` supplies the command-line interface and fixed-width
bond records used by ProtonPottsMPNN's EV6 protonation labeler. Hydrogen placement,
connectivity, bond filtering and formatting run in Rust. It is independently
written and uses no HBPLUS executable or source code.

**This is a tested compatibility subset, not an exact HBPLUS replacement.**
The held-out comparison below found five distinct contact disagreements. No EV6
state labels changed on the tested structures, but numerical probabilities did.
The existing distance-only `hbonds()` and `analyze()` metrics are unchanged.

## Use

Install the package from this checkout, or build the standalone executable:

```bash
python -m pip install .
protein-interface-hbplus -h 3.2 -d 4.0 -a 90 complex.pdb complex.pdb

cargo build --release --locked --bin protein-interface-hbplus
target/release/protein-interface-hbplus --help
```

The default output is `complex.hb2` in the current directory. `--output PATH`
chooses another destination. Existing output is replaced; input PDB files are
protected against accidental overwrite. Rust builds require Rust 1.82 or newer.

For an explicitly qualified ProtonPottsMPNN experiment:

```bash
export HBPLUS_PATH="$(command -v protein-interface-hbplus)"
```

Record that this is the protein-interface implementation, including its version
and binary hash. This setting does not constitute execution of the original
HBPLUS program. A Python API returns the same eight-header-line text:

```python
from pathlib import Path
from protein_interface.hbplus import hbplus_format

text = hbplus_format(
    Path("complex.pdb").read_text(), max_da=4.0, max_ha=3.2,
    donor_overrides=[("ASP", "OD1", 1), ("GLU", "OE1", 1)],
)
```

## Supported contract

The CLI accepts `-h/-H`, `-d/-D`, `-a`, `-A`, `-E`, `-e`, `-v/-V`, `-K/-k`,
one cleaned PDB, and optionally an original PDB supplying CONECT records. Both
the split `-E ASP ' OD1' 1` spelling and the older quoted `-E 'ASP OD1' 1`
spelling work. Positive donor overrides are limited to one hydrogen on ASP
OD1/OD2 and GLU OE1/OE2. Zero disables a donor; acceptor counts 1-3 enable it.

The parser reads the first PDB model and preserves chain IDs, insertion codes,
negative residue numbers and TER boundaries. Coordinates must be finite.
Select alternate conformers before analysis; duplicate identities, mixed residue
names and unsupported residues fail with an error. The scope is the 20 standard
amino acids and water. mmCIF, generic ligands, residue aliases, aromatic pi
acceptors, side-chain flips, `.hbplusrc`, hydrogen-coordinate export and neighbor
listing are outside this release. Unsupported switches fail explicitly.

Standard residue connectivity, peptide geometry, CONECT records and cystine
recognition define the covalent exclusions. Hydrogen loci include planar points,
hydroxyl circles and staggered amines. A circle is evaluated analytically at the
closest point to each acceptor. Supplied hydrogens are ignored unless
`--use-hydrogens` is set. Missing hydrogen geometry uses the published fallback
for selection and reports undefined hydrogen measurements as `-1`.

Output donor and acceptor identifiers occupy columns 1-13 and 15-27. D-A,
D-H-A and H-A occupy columns 28-32, 47-51 and 53-57. These are the fields consumed
by the upstream labeler. The ancillary gap, C-alpha and acceptor-angle fields
have not been independently compared to HBPLUS outputs.

## Validation

The external reference is
[ProtonPottsMPNN at 09682ab](https://github.com/christian-creator/ProtonPottsMPNN/tree/09682abfa7d20e0abcdeea0490b7a4b1c190aee3),
specifically `labeller/data/hbonds.parquet` and the deposited neutron benchmark
coordinates. The reference contains contacts involving His, Asp or Glu; it
cannot establish agreement for every hydrogen bond. No HBPLUS program was run.

The first 32 sorted PDB identifiers were reserved for calibration. Two had
insertion-code keys absent from the reference schema; 30 structures were
comparable. The remaining identifiers were evaluated after freezing the binary.
Fourteen more structures had that key ambiguity, and one coordinate file was
absent; 180 held-out structures were comparable. Exclusions are reported, not counted
as successful comparisons. D-A distances matched exactly for all common bonds.
The authors did not supply hashes of the original labeler inputs.

| Check | Calibration | Held out |
| --- | ---: | ---: |
| Comparable structures | 30 | 180 |
| Reference contacts, summed across three modes | 3,589 | 30,778 |
| Contacts recovered | 3,589 | 30,772 |
| Extra contacts | 0 | 9 |
| Common contacts outside geometry tolerance | 0 | 0 |
| Common contacts with identical printed geometry | 3,582 | 30,742 |

The three modes are default, OD1/OE1 donor overrides, and OD2/OE2 overrides;
shared contacts recur across modes. Geometry tolerances are 0.011 A for D-A/H-A
and 0.11 degrees for D-H-A. These permit one displayed rounding unit. They do not
imply byte-for-byte agreement. The six missing mode-specific records represent
two sulfur-acceptor contacts in 7JOR and 8W6X. The nine extras represent one
contact each in 6L9C, 7WNO and 7WNP, where filtering a modified residue leaves an
internal backbone break. These differences remain unresolved.

For the downstream check, both arms used the same released EV6 classifiers,
coordinates and feature code; only the bond records changed. At the checkpoint's
His 0.3 and acid 0.06 probability thresholds, **all 7,257 state labels agreed**.
The largest probability change was 0.00807; the mean absolute change was
0.00000239. This establishes label agreement on this corpus, not biological
accuracy or equivalence on a new protein.

Reproduce with external reference data and fresh output directories:

```bash
cargo test --locked
python -m pytest tests/test_hbplus.py
python benchmark/hbplus_reference.py --protonpotts /path/to/ProtonPottsMPNN \
  --binary target/release/protein-interface-hbplus --split holdout --output results/hbplus
python benchmark/hbplus_ev6.py --protonpotts /path/to/ProtonPottsMPNN \
  --binary target/release/protein-interface-hbplus \
  --parity-records results/hbplus/per_structure.csv --output results/ev6
```

The reference comparison needs pandas and pyarrow. The EV6 comparison also needs
the upstream labeler dependencies and uses per-structure checkpoints guarded by
a protocol manifest. The package itself gains no dependencies. CPU-only Slurm
jobs were used for the reported validation; shared installations were unchanged.

## Implementation provenance and limits

The specification came from the public HBPLUS manual distributed through the
[official EBI page](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/) and its
[linked repository](https://github.com/RomanLas/HBPLUS), plus the MIT-released
ProtonPottsMPNN caller and saved observations. The manual HTML SHA-256 is
`70e99ddb07faf518c03c8ed4e4da6646b2ec18c0c39e50f1f8f9d70398b219af`;
the reference parquet SHA-256 is
`2889fb9e923159825a0733e1b0083c6a211f1ac5940d3618ce4908971ffd0a22`.
No original C implementation or HBPLUS data tables were copied into this repo.

Several conventions required calibration because the manual was insufficient:
His NE2 accepts bonds in the saved observations; backbone N-H uses a 4-degree
rotation toward C-alpha; the fitted amine geometry uses 112 degrees and 1.014 A.
Histidine's H-A-antecedent filter uses the mean ring-antecedent direction, while
the reported ancillary HAA field retains the smaller individual angle. That
empirical rule reproduced the stored histidine contacts; it is not a claim about
HBPLUS internals. Carboxyl donor overrides use a planar 110-degree O-H locus.
The [calibration notes](hbplus-validation/calibration-notes.json) record the
rotation comparison and amine fit; the [validation records](hbplus-validation/)
contain input hashes, per-structure counts and residual contacts.

These contributions use the repository's MIT license. Independent implementation
does not establish freedom to operate for software methods or designed proteins.
Further qualification needs the residual contacts resolved, broader reference
coverage, and EV6 checks on additional proteins.
