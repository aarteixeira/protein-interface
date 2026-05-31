from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

from protein_interface.openmm import (
    SAMPLED_GBSA_PRESETS,
    calculate_gbsa_binding_energy,
    calculate_sampled_gbsa_binding_energy,
    openmm_potential_energy,
    relax_structure,
)
from protein_interface import interface_residues, load_atoms


DATA_DIR = Path(__file__).parent / "data"
NB_AG = DATA_DIR / "nb_ag_test.pdb"
ONE_FYT = DATA_DIR / "1fyt.pdb"


def test_openmm_dependency_is_optional(monkeypatch):
    import protein_interface.openmm as pi_openmm

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "openmm" or name.startswith("openmm."):
            raise ImportError("blocked for test")
        return real_import_module(name, package)

    monkeypatch.setattr(pi_openmm.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="protein-interface\\[openmm\\]"):
        openmm_potential_energy(NB_AG, chains=["A"])


def test_openmm_rejects_unsupported_input_suffix(tmp_path):
    bad = tmp_path / "complex.xyz"
    bad.write_text("not a structure\n")

    with pytest.raises(ValueError, match="path must end"):
        openmm_potential_energy(bad)


def test_openmm_rejects_unsupported_output_suffix(tmp_path):
    with pytest.raises(ValueError, match="output_path must end"):
        relax_structure(NB_AG, output_path=tmp_path / "relaxed.xyz")


def test_interface_relaxation_requires_chain_groups():
    with pytest.raises(ValueError, match="requires chains_a and chains_b"):
        relax_structure(NB_AG, mode="interface")


def test_invalid_chain_raises_value_error():
    pytest.importorskip("openmm")

    with pytest.raises(ValueError, match="chain ID"):
        openmm_potential_energy(NB_AG, chains=["missing"])


def test_openmm_potential_energy_returns_finite_values():
    pytest.importorskip("openmm")

    result = openmm_potential_energy(ONE_FYT, chains=["A"])

    assert result.atom_count > 0
    assert result.chains == ("A",)
    assert result.forcefield_files == ("amber14-all.xml", "implicit/obc2.xml")
    assert math.isfinite(result.energy_kj_mol)
    assert math.isclose(result.energy_kcal_mol, result.energy_kj_mol * 0.2390057361376673)


def test_gbsa_binding_energy_returns_consistent_delta():
    pytest.importorskip("openmm")

    result = calculate_gbsa_binding_energy(ONE_FYT, chains_a=["A"], chains_b=["C"])

    expected = result.complex_energy_kj_mol - result.chain_a_energy_kj_mol - result.chain_b_energy_kj_mol
    assert result.chains_a == ("A",)
    assert result.chains_b == ("C",)
    assert result.entropy_included is False
    assert math.isfinite(result.delta_g_kj_mol)
    assert math.isclose(result.delta_g_kj_mol, expected)


def test_sampled_gbsa_binding_energy_samples_frames_and_warns(tmp_path):
    pytest.importorskip("openmm")
    sampled_fixture = _duplicated_small_chain_fixture(tmp_path)

    with pytest.warns(RuntimeWarning, match="preset='short'"):
        result = calculate_sampled_gbsa_binding_energy(
            sampled_fixture,
            chains_a=["C"],
            chains_b=["D"],
            preset="short",
            production_steps=1,
            equilibration_steps=0,
            sample_interval=1,
            timestep_fs=0.1,
            random_seed=1,
        )

    assert result.frame_count == 1
    assert result.preset == "short"
    assert result.production_steps == 1
    assert result.equilibration_steps == 0
    assert result.sample_interval == 1
    assert result.chains_a == ("C",)
    assert result.chains_b == ("D",)
    assert result.entropy_included is False
    assert len(result.frame_delta_g_kj_mol) == 1
    assert math.isfinite(result.mean_delta_g_kj_mol)
    assert math.isfinite(result.std_delta_g_kj_mol)


def test_sampled_gbsa_presets_are_named_protocols():
    assert SAMPLED_GBSA_PRESETS["short"] == {
        "equilibration_steps": 5_000,
        "production_steps": 50_000,
        "sample_interval": 500,
        "timestep_fs": 2.0,
    }
    assert SAMPLED_GBSA_PRESETS["medium"] == {
        "equilibration_steps": 250_000,
        "production_steps": 2_500_000,
        "sample_interval": 10_000,
        "timestep_fs": 2.0,
    }
    assert SAMPLED_GBSA_PRESETS["long"] == {
        "equilibration_steps": 500_000,
        "production_steps": 10_000_000,
        "sample_interval": 20_000,
        "timestep_fs": 2.0,
    }


def test_sampled_gbsa_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown sampled GBSA preset"):
        calculate_sampled_gbsa_binding_energy(
            ONE_FYT,
            chains_a=["A"],
            chains_b=["C"],
            preset="overnight",
        )


def test_whole_relaxation_reduces_energy_and_writes_output(tmp_path):
    pytest.importorskip("openmm")

    out = tmp_path / "relaxed.pdb"
    result = relax_structure(
        ONE_FYT,
        output_path=out,
        chains=["A"],
        max_iterations=10,
    )

    assert out.exists()
    assert result.output_path == str(out)
    assert result.atom_count > 0
    assert result.restrained_atom_count == 0
    assert result.free_atom_count == result.atom_count
    assert result.final_energy_kj_mol <= result.initial_energy_kj_mol + 1e-6

    reread = openmm_potential_energy(out)
    assert reread.atom_count == result.atom_count


def test_interface_relaxation_restrains_noninterface_atoms(tmp_path):
    pytest.importorskip("openmm")

    out = tmp_path / "interface_relaxed.pdb"
    result = relax_structure(
        ONE_FYT,
        output_path=out,
        mode="interface",
        chains_a=["A"],
        chains_b=["C"],
        restraint_kj_mol_nm2=1_000_000.0,
        max_iterations=10,
    )

    assert out.exists()
    assert result.atom_count > 0
    assert result.restrained_atom_count > 0
    assert result.free_atom_count > 0
    assert result.restrained_atom_count + result.free_atom_count == result.atom_count
    assert result.final_energy_kj_mol <= result.initial_energy_kj_mol + 1e-6

    assert _max_noninterface_heavy_atom_displacement(ONE_FYT, out, ["A"], ["C"]) < 0.2


def _max_noninterface_heavy_atom_displacement(
    before_path: Path,
    after_path: Path,
    chains_a: list[str],
    chains_b: list[str],
) -> float:
    before_a = load_atoms(before_path, chains_a)
    before_b = load_atoms(before_path, chains_b)
    after_a = load_atoms(after_path, chains_a)
    after_b = load_atoms(after_path, chains_b)
    interface_a, interface_b = interface_residues(before_a, before_b)

    return max(
        _side_max_displacement(before_a, after_a, interface_a),
        _side_max_displacement(before_b, after_b, interface_b),
    )


def _side_max_displacement(before, after, interface_ids: set[tuple[str, int, str]]) -> float:
    assert before.atom_names == after.atom_names
    assert before.residue_names == after.residue_names

    max_displacement = 0.0
    for i, rid in enumerate(before.residue_ids):
        if rid in interface_ids:
            continue
        dx = before.coords[i][0] - after.coords[i][0]
        dy = before.coords[i][1] - after.coords[i][1]
        dz = before.coords[i][2] - after.coords[i][2]
        max_displacement = max(max_displacement, math.sqrt(dx * dx + dy * dy + dz * dz))
    return max_displacement


def _duplicated_small_chain_fixture(tmp_path: Path) -> Path:
    src = ONE_FYT.read_text().splitlines()
    lines: list[str] = []
    serial = 1
    for line in src:
        if line.startswith("ATOM") and line[21] == "C":
            lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
    for line in src:
        if line.startswith("ATOM") and line[21] == "C":
            x = float(line[30:38]) + 8.0
            lines.append(f"{line[:6]}{serial:5d}{line[11:21]}D{line[22:30]}{x:8.3f}{line[38:]}")
            serial += 1
    lines.append("END")
    path = tmp_path / "sampled_gbsa_test.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path
