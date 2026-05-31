# OpenMM Helpers

`protein_interface.openmm` is optional. The base package does not import OpenMM,
and `analyze()` never changes coordinates.

## Install

```bash
python -m pip install "protein-interface[openmm]"
```

For Linux GPU use, prefer the conda/mamba environment file:

```bash
mamba env create -f environment-gpu.yml
mamba activate protein-interface-gpu
```

The GPU environment pins `cuda-version=12.4` and `openmm<8.3`. This avoids
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` on NVIDIA 550-series drivers, where newer
conda-forge OpenMM solves can pull CUDA 12.9 runtime components that the driver
cannot JIT. If your driver supports a newer CUDA runtime, update the
`cuda-version` pin deliberately and verify `platform="CUDA"` before long jobs.

## Helpers

| Function | Purpose |
|---|---|
| `relax_structure(path, ...)` | OpenMM minimization for a whole selected structure or an interface-focused run with non-interface atoms restrained. |
| `openmm_potential_energy(path, ...)` | Potential energy for selected chains after OpenMM hydrogen addition. |
| `calculate_gbsa_binding_energy(path, chains_a, chains_b, ...)` | Single-structure MM-GBSA-style estimate: `G_complex - G_a - G_b`. |
| `calculate_sampled_gbsa_binding_energy(path, chains_a, chains_b, preset="short", ...)` | MD-sampled MM-GBSA estimate over multiple frames. |

Defaults use `amber14-all.xml` and `implicit/obc2.xml`. The module keeps
standard amino-acid residues, removes waters and non-protein residues before
OpenMM setup, and adds hydrogens with OpenMM. It does not repair missing heavy
atoms, infer ligands, add solvent boxes, mutate residues, or parameterize
nonstandard chemistry. Template and parameterization errors come from OpenMM.

Relaxation changes coordinates. Recompute geometry metrics from the relaxed
output if you need relaxed-structure descriptors.

## Sampled GBSA

The sampled GBSA helper runs minimization, optional equilibration, and MD before
scoring sampled frames. Use a GPU platform such as CUDA, OpenCL, or Metal for
large jobs.

| Preset | Equilibration | Production | Sample interval | Frames | Use |
|---|---:|---:|---:|---:|---|
| `short` | 10 ps | 100 ps | 1 ps | 100 | Smoke tests, API checks, GPU setup validation, and rough triage. Do not treat it as a stable MM-GBSA estimate. |
| `medium` | 0.5 ns | 5 ns | 20 ps | 250 | GPU screening of related variants after filtering with faster metrics. Use this preset or longer for candidate comparisons. |
| `long` | 1 ns | 20 ns | 40 ps | 500 | Ranking a small number of finalists, preferably with replicate runs. It is slower and still does not prove convergence. |

The presets assume the default 2 fs timestep. Explicit `production_steps`,
`equilibration_steps`, `sample_interval`, or `timestep_fs` arguments override
the selected preset.

GBSA outputs are force-field endpoint scores. They are not PRODIGY, experimental
affinity, or Poisson-Boltzmann PBSA. Entropy is not included.

## CUDA Check

On Linux, pass `platform="CUDA"` only after confirming OpenMM sees CUDA:

```python
from openmm import Platform

print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
```

If CUDA setup fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, recreate the
environment from `environment-gpu.yml` or pin `cuda-version` to a runtime
supported by the installed NVIDIA driver.

## Bundled GPU Smoke Comparison

`tests/data/1fyt.pdb` chains `A` vs `C` were run on `gnode1` with OpenMM 8.2.0
and `platform="CUDA"`:

| Method | Settings | Result |
|---|---|---|
| Single-structure GBSA | one endpoint evaluation | `-183.59 kJ/mol` (`-43.88 kcal/mol`) |
| Sampled GBSA default | 10 ps equilibration, 100 ps production, 100 frames | `-306.23 +/- 26.11 kJ/mol` (`-73.19 +/- 6.24 kcal/mol`) |

The sampled default took about 21 s on the `gnode1` NVIDIA RTX 4000 SFF Ada GPU
for this small fixture. Treat this as a smoke test and quick sampled estimate,
not a converged binding free energy. For reported comparisons, increase
equilibration and production length, check trajectory stability, and report
replicate runs.
