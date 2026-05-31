"""Optional OpenMM-backed relaxation and GBSA energy helpers.

This module is deliberately separate from :mod:`protein_interface.interface`:
the core ``analyze()`` path remains coordinate-only and never relaxes or
re-energies structures as a side effect.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protein_interface import interface_residues, load_atoms

DEFAULT_FORCEFIELD_FILES = ("amber14-all.xml", "implicit/obc2.xml")
KJ_TO_KCAL = 0.2390057361376673


@dataclass(frozen=True)
class RelaxationResult:
    initial_energy_kj_mol: float
    final_energy_kj_mol: float
    initial_energy_kcal_mol: float
    final_energy_kcal_mol: float
    output_path: str
    atom_count: int
    restrained_atom_count: int
    free_atom_count: int
    platform: str
    forcefield_files: tuple[str, ...]


@dataclass(frozen=True)
class EnergyResult:
    energy_kj_mol: float
    energy_kcal_mol: float
    atom_count: int
    chains: tuple[str, ...] | None
    platform: str
    forcefield_files: tuple[str, ...]


@dataclass(frozen=True)
class GBSAResult:
    delta_g_kj_mol: float
    delta_g_kcal_mol: float
    complex_energy_kj_mol: float
    complex_energy_kcal_mol: float
    chain_a_energy_kj_mol: float
    chain_a_energy_kcal_mol: float
    chain_b_energy_kj_mol: float
    chain_b_energy_kcal_mol: float
    chains_a: tuple[str, ...]
    chains_b: tuple[str, ...]
    entropy_included: bool
    platform: str
    forcefield_files: tuple[str, ...]


def relax_structure(
    path: str | Path,
    output_path: str | Path | None = None,
    chains: list[str] | tuple[str, ...] | None = None,
    mode: str = "whole",
    chains_a: list[str] | tuple[str, ...] | None = None,
    chains_b: list[str] | tuple[str, ...] | None = None,
    interface_cutoff: float = 5.0,
    forcefield_files: tuple[str, ...] = DEFAULT_FORCEFIELD_FILES,
    ph: float = 7.0,
    restraint_kj_mol_nm2: float = 1000.0,
    tolerance_kj_mol_nm: float = 10.0,
    max_iterations: int = 500,
    platform: str | None = None,
) -> RelaxationResult:
    """Minimize a structure with OpenMM.

    ``mode="whole"`` minimizes all atoms in the selected chains. ``mode="interface"``
    minimizes the selected system while restraining atoms outside the
    contact-defined interface residues from ``chains_a`` and ``chains_b``.
    """
    if mode not in {"whole", "interface"}:
        raise ValueError("mode must be 'whole' or 'interface'")
    _validate_input_suffix(path)
    _validate_nonnegative_finite("interface_cutoff", interface_cutoff)
    _validate_nonnegative_finite("restraint_kj_mol_nm2", restraint_kj_mol_nm2)
    _validate_nonnegative_finite("tolerance_kj_mol_nm", tolerance_kj_mol_nm)
    _validate_finite("ph", ph)
    _validate_max_iterations(max_iterations)

    if mode == "interface":
        if not chains_a or not chains_b:
            raise ValueError("mode='interface' requires chains_a and chains_b")
        interface_keys = _interface_residue_keys(path, chains_a, chains_b, interface_cutoff)
        selected_chains = tuple(chains) if chains is not None else _chain_union(chains_a, chains_b)
    else:
        interface_keys = set()
        selected_chains = tuple(chains) if chains is not None else None

    out = Path(output_path) if output_path is not None else _default_output_path(path)
    _validate_output_suffix(out)

    app, omm, unit = _require_openmm()
    forcefield = app.ForceField(*forcefield_files)
    topology, positions = _prepared_topology_positions(
        path, app, forcefield, ph=ph, chains=selected_chains
    )
    system = _create_system(app, topology, forcefield)

    restrained_count = 0
    if mode == "interface":
        restrained_count = _add_noninterface_restraints(
            system,
            topology,
            positions,
            interface_keys,
            restraint_kj_mol_nm2,
            omm,
        )

    simulation = _make_simulation(topology, system, omm, unit, platform)
    simulation.context.setPositions(positions)
    initial = _context_energy_kj_mol(simulation.context, unit)
    simulation.minimizeEnergy(
        tolerance=tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )
    final = _context_energy_kj_mol(simulation.context, unit)
    state = simulation.context.getState(getPositions=True)
    final_positions = state.getPositions()
    _write_structure(out, topology, final_positions, app)

    atom_count = topology.getNumAtoms()
    return RelaxationResult(
        initial_energy_kj_mol=initial,
        final_energy_kj_mol=final,
        initial_energy_kcal_mol=initial * KJ_TO_KCAL,
        final_energy_kcal_mol=final * KJ_TO_KCAL,
        output_path=str(out),
        atom_count=atom_count,
        restrained_atom_count=restrained_count,
        free_atom_count=atom_count - restrained_count,
        platform=simulation.context.getPlatform().getName(),
        forcefield_files=tuple(forcefield_files),
    )


def openmm_potential_energy(
    path: str | Path,
    chains: list[str] | tuple[str, ...] | None = None,
    forcefield_files: tuple[str, ...] = DEFAULT_FORCEFIELD_FILES,
    ph: float = 7.0,
    platform: str | None = None,
) -> EnergyResult:
    """Return OpenMM potential energy for a prepared structure."""
    _validate_input_suffix(path)
    _validate_finite("ph", ph)
    app, _, _ = _require_openmm()
    forcefield = app.ForceField(*forcefield_files)
    topology, positions = _prepared_topology_positions(path, app, forcefield, ph=ph, chains=chains)
    energy, used_platform = _energy_for_topology(topology, positions, forcefield, app, platform)
    return EnergyResult(
        energy_kj_mol=energy,
        energy_kcal_mol=energy * KJ_TO_KCAL,
        atom_count=topology.getNumAtoms(),
        chains=tuple(chains) if chains is not None else None,
        platform=used_platform,
        forcefield_files=tuple(forcefield_files),
    )


def calculate_gbsa_binding_energy(
    path: str | Path,
    chains_a: list[str] | tuple[str, ...],
    chains_b: list[str] | tuple[str, ...],
    forcefield_files: tuple[str, ...] = DEFAULT_FORCEFIELD_FILES,
    ph: float = 7.0,
    platform: str | None = None,
) -> GBSAResult:
    """Compute a single-structure MM-GBSA-style binding energy.

    The returned value is ``G_complex - G_a - G_b`` from OpenMM potential
    energies with the configured implicit-solvent force field. Entropy is not
    included, and the result should not be interpreted as an absolute affinity.
    """
    if not chains_a:
        raise ValueError("chains_a must contain at least one chain ID")
    if not chains_b:
        raise ValueError("chains_b must contain at least one chain ID")
    _validate_input_suffix(path)
    overlap = set(chains_a) & set(chains_b)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"chains_a and chains_b overlap: {names}")
    _validate_finite("ph", ph)

    app, _, _ = _require_openmm()
    forcefield = app.ForceField(*forcefield_files)
    complex_chains = _chain_union(chains_a, chains_b)
    prepared_topology, prepared_positions = _prepared_topology_positions(
        path, app, forcefield, ph=ph, chains=complex_chains
    )

    complex_energy, used_platform = _energy_for_topology(
        prepared_topology, prepared_positions, forcefield, app, platform
    )
    top_a, pos_a = _subset_topology_positions(prepared_topology, prepared_positions, chains_a, app)
    top_b, pos_b = _subset_topology_positions(prepared_topology, prepared_positions, chains_b, app)
    energy_a, _ = _energy_for_topology(top_a, pos_a, forcefield, app, platform)
    energy_b, _ = _energy_for_topology(top_b, pos_b, forcefield, app, platform)
    delta = complex_energy - energy_a - energy_b

    return GBSAResult(
        delta_g_kj_mol=delta,
        delta_g_kcal_mol=delta * KJ_TO_KCAL,
        complex_energy_kj_mol=complex_energy,
        complex_energy_kcal_mol=complex_energy * KJ_TO_KCAL,
        chain_a_energy_kj_mol=energy_a,
        chain_a_energy_kcal_mol=energy_a * KJ_TO_KCAL,
        chain_b_energy_kj_mol=energy_b,
        chain_b_energy_kcal_mol=energy_b * KJ_TO_KCAL,
        chains_a=tuple(chains_a),
        chains_b=tuple(chains_b),
        entropy_included=False,
        platform=used_platform,
        forcefield_files=tuple(forcefield_files),
    )


def _require_openmm() -> tuple[Any, Any, Any]:
    try:
        app = importlib.import_module("openmm.app")
        omm = importlib.import_module("openmm")
        unit = importlib.import_module("openmm.unit")
    except ImportError as exc:
        raise ImportError(
            "OpenMM support requires the optional OpenMM dependency. "
            "Install it with: python -m pip install 'protein-interface[openmm]'"
        ) from exc
    return app, omm, unit


def _load_openmm_structure(path: str | Path, app: Any) -> tuple[Any, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        pdb = app.PDBFile(str(path))
        return pdb.topology, pdb.positions
    if suffix in {".cif", ".mmcif"}:
        pdbx = app.PDBxFile(str(path))
        return pdbx.getTopology(), pdbx.getPositions()
    raise ValueError("path must end in .pdb, .ent, .cif, or .mmcif")


def _prepared_topology_positions(
    path: str | Path,
    app: Any,
    forcefield: Any,
    *,
    ph: float,
    chains: list[str] | tuple[str, ...] | None,
) -> tuple[Any, Any]:
    topology, positions = _load_openmm_structure(path, app)
    if chains is not None:
        topology, positions = _subset_topology_positions(topology, positions, chains, app)
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield, pH=ph)
    return modeller.topology, modeller.positions


def _subset_topology_positions(
    topology: Any,
    positions: Any,
    chains: list[str] | tuple[str, ...],
    app: Any,
) -> tuple[Any, Any]:
    wanted = set(chains)
    if not wanted:
        raise ValueError("chains must contain at least one chain ID")
    available = {chain.id for chain in topology.chains()}
    missing = wanted - available
    if missing:
        names = ", ".join(sorted(missing))
        valid = ", ".join(sorted(available))
        raise ValueError(f"chain ID(s) not found: {names}; available chains: {valid}")
    modeller = app.Modeller(topology, positions)
    modeller.delete([chain for chain in modeller.topology.chains() if chain.id not in wanted])
    return modeller.topology, modeller.positions


def _create_system(app: Any, topology: Any, forcefield: Any) -> Any:
    return forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )


def _energy_for_topology(
    topology: Any,
    positions: Any,
    forcefield: Any,
    app: Any,
    platform: str | None,
) -> tuple[float, str]:
    _, omm, unit = _require_openmm()
    system = _create_system(app, topology, forcefield)
    simulation = _make_simulation(topology, system, omm, unit, platform)
    simulation.context.setPositions(positions)
    energy = _context_energy_kj_mol(simulation.context, unit)
    return energy, simulation.context.getPlatform().getName()


def _make_simulation(topology: Any, system: Any, omm: Any, unit: Any, platform: str | None) -> Any:
    app = importlib.import_module("openmm.app")
    integrator = omm.VerletIntegrator(0.001 * unit.picoseconds)
    if platform is None:
        return app.Simulation(topology, system, integrator)
    platform_obj = omm.Platform.getPlatformByName(platform)
    return app.Simulation(topology, system, integrator, platform_obj)


def _context_energy_kj_mol(context: Any, unit: Any) -> float:
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _add_noninterface_restraints(
    system: Any,
    topology: Any,
    positions: Any,
    interface_keys: set[tuple[str, int | str, str]],
    restraint_kj_mol_nm2: float,
    omm: Any,
) -> int:
    force = omm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", restraint_kj_mol_nm2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    restrained = 0
    for atom, position in zip(topology.atoms(), positions):
        if _openmm_residue_key(atom.residue) in interface_keys:
            continue
        force.addParticle(atom.index, [position.x, position.y, position.z])
        restrained += 1
    system.addForce(force)
    return restrained


def _interface_residue_keys(
    path: str | Path,
    chains_a: list[str] | tuple[str, ...],
    chains_b: list[str] | tuple[str, ...],
    cutoff: float,
) -> set[tuple[str, int | str, str]]:
    atoms_a = load_atoms(path, list(chains_a))
    atoms_b = load_atoms(path, list(chains_b))
    int_a, int_b = interface_residues(atoms_a, atoms_b, cutoff=cutoff)
    return set(int_a) | set(int_b)


def _openmm_residue_key(residue: Any) -> tuple[str, int | str, str]:
    return (
        residue.chain.id,
        _normalise_resseq(residue.id),
        (residue.insertionCode or "").strip(),
    )


def _normalise_resseq(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _chain_union(
    chains_a: list[str] | tuple[str, ...],
    chains_b: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    out: list[str] = []
    for chain in tuple(chains_a) + tuple(chains_b):
        if chain not in out:
            out.append(chain)
    return tuple(out)


def _default_output_path(path: str | Path) -> Path:
    path = Path(path)
    suffix = path.suffix if path.suffix else ".pdb"
    return path.with_name(f"{path.stem}_relaxed{suffix}")


def _validate_output_suffix(path: Path) -> None:
    if path.suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
        raise ValueError("output_path must end in .pdb, .ent, .cif, or .mmcif")


def _validate_input_suffix(path: str | Path) -> None:
    if Path(path).suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
        raise ValueError("path must end in .pdb, .ent, .cif, or .mmcif")


def _write_structure(path: Path, topology: Any, positions: Any, app: Any) -> None:
    with path.open("w") as handle:
        if path.suffix.lower() in {".cif", ".mmcif"}:
            app.PDBxFile.writeFile(topology, positions, handle)
        else:
            app.PDBFile.writeFile(topology, positions, handle)


def _validate_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    _validate_finite(name, value)
    if float(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_max_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_iterations must be an integer >= 0")


__all__ = [
    "DEFAULT_FORCEFIELD_FILES",
    "RelaxationResult",
    "EnergyResult",
    "GBSAResult",
    "relax_structure",
    "openmm_potential_energy",
    "calculate_gbsa_binding_energy",
]
