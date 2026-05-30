use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sc_rs::sc::atomic_radii::embedded_atomic_radii;
use sc_rs::sc::{types::Atom, vector3::Vec3, ScCalculator};

mod hbonds;
mod salt_bridges;
mod sasa;

#[pyclass]
pub struct ScResult {
    #[pyo3(get)]
    pub sc: f64,
    #[pyo3(get)]
    pub median_distance: f64,
    #[pyo3(get)]
    pub trimmed_area: f64,
    #[pyo3(get)]
    pub atoms_a: usize,
    #[pyo3(get)]
    pub atoms_b: usize,
}

fn validate_nonnegative_finite(name: &str, value: f64) -> PyResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{} must be a finite non-negative number",
            name
        )));
    }
    Ok(())
}

#[pymethods]
impl ScResult {
    fn __repr__(&self) -> String {
        format!(
            "ScResult(sc={:.4}, median_distance={:.4}, trimmed_area={:.2}, atoms_a={}, atoms_b={})",
            self.sc, self.median_distance, self.trimmed_area, self.atoms_a, self.atoms_b
        )
    }
}

/// Compute Lawrence-Colman Shape Complementarity between two atom groups.
///
/// Mirrors the exact behavior of the sc-rs CLI binary. Atom radii are assigned
/// automatically from atom name + residue name; atoms without a known radius
/// are silently dropped by the upstream library.
///
/// Args:
///     coords_a, atom_names_a, residue_names_a: atoms for molecule A
///     coords_b, atom_names_b, residue_names_b: atoms for molecule B
///     parallel: enable Rayon parallelism inside sc-rs (default True;
///               set False when calling from a ProcessPoolExecutor to avoid
///               oversubscription)
#[pyfunction]
#[pyo3(signature = (coords_a, atom_names_a, residue_names_a, coords_b, atom_names_b, residue_names_b, parallel=true))]
fn compute_sc(
    coords_a: Vec<[f64; 3]>,
    atom_names_a: Vec<String>,
    residue_names_a: Vec<String>,
    coords_b: Vec<[f64; 3]>,
    atom_names_b: Vec<String>,
    residue_names_b: Vec<String>,
    parallel: bool,
) -> PyResult<ScResult> {
    let na = coords_a.len();
    let nb = coords_b.len();

    if na == 0 || nb == 0 {
        return Err(PyValueError::new_err(
            "each atom group must contain at least one atom",
        ));
    }
    if atom_names_a.len() != na || residue_names_a.len() != na {
        return Err(PyValueError::new_err(
            "coords_a, atom_names_a, residue_names_a must all have the same length",
        ));
    }
    if atom_names_b.len() != nb || residue_names_b.len() != nb {
        return Err(PyValueError::new_err(
            "coords_b, atom_names_b, residue_names_b must all have the same length",
        ));
    }

    let mut calc = ScCalculator::new();
    calc.settings_mut().enable_parallel = parallel;

    for i in 0..na {
        let mut a = Atom::new();
        a.coor = Vec3::new(coords_a[i][0], coords_a[i][1], coords_a[i][2]);
        a.atom = atom_names_a[i].clone();
        a.residue = residue_names_a[i].clone();
        calc.add_atom(0, a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    for i in 0..nb {
        let mut a = Atom::new();
        a.coor = Vec3::new(coords_b[i][0], coords_b[i][1], coords_b[i][2]);
        a.atom = atom_names_b[i].clone();
        a.residue = residue_names_b[i].clone();
        calc.add_atom(1, a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    let results = calc
        .calc()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(ScResult {
        sc: results.sc,
        median_distance: results.distance,
        trimmed_area: results.area,
        atoms_a: results.surfaces[0].n_atoms,
        atoms_b: results.surfaces[1].n_atoms,
    })
}

/// Per-atom solvent-accessible surface area (Shrake-Rupley, Å²).
///
/// Uses sc-rs's MS-style atomic radii (same lookup as compute_sc), so atoms
/// with no radius entry are assigned SASA = 0.0.
///
/// Args:
///     coords, atom_names, residue_names: parallel arrays describing the atoms
///     probe_radius: solvent probe radius in Å (default 1.4)
///     n_points: sphere points per atom (default 960; higher = more accurate, slower)
#[pyfunction]
#[pyo3(signature = (coords, atom_names, residue_names, probe_radius=1.4, n_points=960))]
fn compute_sasa(
    coords: Vec<[f64; 3]>,
    atom_names: Vec<String>,
    residue_names: Vec<String>,
    probe_radius: f64,
    n_points: usize,
) -> PyResult<Vec<f64>> {
    let n = coords.len();
    if atom_names.len() != n || residue_names.len() != n {
        return Err(PyValueError::new_err(
            "coords, atom_names, residue_names must all have the same length",
        ));
    }
    if n_points < 4 {
        return Err(PyValueError::new_err("n_points must be >= 4"));
    }
    validate_nonnegative_finite("probe_radius", probe_radius)?;
    Ok(sasa::compute(
        &coords,
        &atom_names,
        &residue_names,
        probe_radius,
        n_points,
    ))
}

/// Return atoms that have no entry in the embedded SASA/SC radius table.
#[pyfunction]
fn unknown_sasa_radius_atoms(
    atom_names: Vec<String>,
    residue_names: Vec<String>,
) -> PyResult<Vec<(usize, String, String)>> {
    if atom_names.len() != residue_names.len() {
        return Err(PyValueError::new_err(
            "atom_names and residue_names must have the same length",
        ));
    }
    Ok(sasa::unknown_radius_atoms(
        &atom_names,
        &residue_names,
        &embedded_atomic_radii(),
    ))
}

/// Count cross-interface hydrogen bonds using a distance-only criterion.
///
/// Donors and acceptors are classified by (residue, atom) name; a pair is
/// counted if their heavy atoms are within `cutoff` Å. Hydrogens and angles
/// are not used (PDB inputs typically lack H).
#[pyfunction]
#[pyo3(signature = (coords_a, atom_names_a, residue_names_a, coords_b, atom_names_b, residue_names_b, cutoff=3.5))]
fn count_hbonds(
    coords_a: Vec<[f64; 3]>,
    atom_names_a: Vec<String>,
    residue_names_a: Vec<String>,
    coords_b: Vec<[f64; 3]>,
    atom_names_b: Vec<String>,
    residue_names_b: Vec<String>,
    cutoff: f64,
) -> PyResult<usize> {
    let na = coords_a.len();
    let nb = coords_b.len();
    if atom_names_a.len() != na || residue_names_a.len() != na {
        return Err(PyValueError::new_err(
            "coords_a, atom_names_a, residue_names_a must all have the same length",
        ));
    }
    if atom_names_b.len() != nb || residue_names_b.len() != nb {
        return Err(PyValueError::new_err(
            "coords_b, atom_names_b, residue_names_b must all have the same length",
        ));
    }
    validate_nonnegative_finite("cutoff", cutoff)?;
    Ok(hbonds::count(
        &coords_a,
        &atom_names_a,
        &residue_names_a,
        &coords_b,
        &atom_names_b,
        &residue_names_b,
        cutoff,
    ))
}

/// Count cross-interface salt bridges using a distance-only criterion.
///
/// Anionic side-chain oxygens (Asp OD*, Glu OE*) and cationic side-chain
/// nitrogens (Lys NZ, Arg NE/NH*, His ND1/NE2) on opposite sides within
/// `cutoff` Å are counted. Default 4.0 Å follows Barlow & Thornton (1983).
#[pyfunction]
#[pyo3(signature = (coords_a, atom_names_a, residue_names_a, coords_b, atom_names_b, residue_names_b, cutoff=4.0))]
fn count_salt_bridges(
    coords_a: Vec<[f64; 3]>,
    atom_names_a: Vec<String>,
    residue_names_a: Vec<String>,
    coords_b: Vec<[f64; 3]>,
    atom_names_b: Vec<String>,
    residue_names_b: Vec<String>,
    cutoff: f64,
) -> PyResult<usize> {
    let na = coords_a.len();
    let nb = coords_b.len();
    if atom_names_a.len() != na || residue_names_a.len() != na {
        return Err(PyValueError::new_err(
            "coords_a, atom_names_a, residue_names_a must all have the same length",
        ));
    }
    if atom_names_b.len() != nb || residue_names_b.len() != nb {
        return Err(PyValueError::new_err(
            "coords_b, atom_names_b, residue_names_b must all have the same length",
        ));
    }
    validate_nonnegative_finite("cutoff", cutoff)?;
    Ok(salt_bridges::count(
        &coords_a,
        &atom_names_a,
        &residue_names_a,
        &coords_b,
        &atom_names_b,
        &residue_names_b,
        cutoff,
    ))
}

/// Batch per-atom SASA over many atom systems in a single FFI call.
///
/// Each entry in `structures` is `(coords, atom_names, residue_names)` for one
/// independent atom system (e.g. one chain, or a complex). Returns a parallel
/// list of per-atom SASA arrays.
///
/// With `parallel=True` (default) the inner loop runs across structures via
/// Rayon and releases the GIL — single-process throughput scales with cores.
/// When orchestrating from a `ProcessPoolExecutor` set `parallel=False` to
/// avoid CPU oversubscription, the same convention as `compute_sc`.
#[pyfunction]
#[pyo3(signature = (structures, probe_radius=1.4, n_points=960, parallel=true))]
fn compute_sasa_batch(
    py: Python<'_>,
    structures: Vec<(Vec<[f64; 3]>, Vec<String>, Vec<String>)>,
    probe_radius: f64,
    n_points: usize,
    parallel: bool,
) -> PyResult<Vec<Vec<f64>>> {
    if n_points < 4 {
        return Err(PyValueError::new_err("n_points must be >= 4"));
    }
    validate_nonnegative_finite("probe_radius", probe_radius)?;
    for (i, (c, an, rn)) in structures.iter().enumerate() {
        if an.len() != c.len() || rn.len() != c.len() {
            return Err(PyValueError::new_err(format!(
                "structure {}: coords / atom_names / residue_names have mismatched lengths",
                i
            )));
        }
    }
    let table = embedded_atomic_radii();
    let result = py.allow_threads(|| {
        if parallel {
            structures
                .par_iter()
                .map(|(c, an, rn)| sasa::compute_with_radii(c, an, rn, probe_radius, n_points, &table))
                .collect()
        } else {
            structures
                .iter()
                .map(|(c, an, rn)| sasa::compute_with_radii(c, an, rn, probe_radius, n_points, &table))
                .collect()
        }
    });
    Ok(result)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScResult>()?;
    m.add_function(wrap_pyfunction!(compute_sc, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(unknown_sasa_radius_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sasa_batch, m)?)?;
    m.add_function(wrap_pyfunction!(count_hbonds, m)?)?;
    m.add_function(wrap_pyfunction!(count_salt_bridges, m)?)?;
    Ok(())
}
