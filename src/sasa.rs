//! Shrake-Rupley solvent-accessible surface area.
//!
//! Reuses sc-rs's embedded atomic radii (MS-style, from CCP4 sc Fortran source)
//! so atom radii are consistent with the SC calculation.

use sc_rs::sc::atomic_radii::{embedded_atomic_radii, wildcard_match};
use sc_rs::sc::types::AtomRadius;

/// Generate `n` points on the unit sphere via the Fibonacci/golden-spiral method.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let mut pts = Vec::with_capacity(n);
    let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f64) / ((n - 1) as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi * (i as f64);
        pts.push([r * theta.cos(), y, r * theta.sin()]);
    }
    pts
}

/// Look up a radius for (residue, atom) using sc-rs wildcard rules.
/// Returns None if no entry matches — caller decides whether to skip or default.
fn lookup_radius(table: &[AtomRadius], residue: &str, atom: &str) -> Option<f64> {
    for r in table {
        if wildcard_match(residue, &r.residue) && wildcard_match(atom, &r.atom) {
            return Some(r.radius);
        }
    }
    None
}

/// Compute per-atom SASA in Å². Atoms with no radius entry get SASA = 0.0.
/// Loads the radii table on every call; for batch use prefer
/// [`compute_with_radii`] with a shared table.
pub fn compute(
    coords: &[[f64; 3]],
    atom_names: &[String],
    residue_names: &[String],
    probe_radius: f64,
    n_points: usize,
) -> Vec<f64> {
    compute_with_radii(
        coords,
        atom_names,
        residue_names,
        probe_radius,
        n_points,
        &embedded_atomic_radii(),
    )
}

/// Like [`compute`] but uses a caller-supplied radii table. Batch callers
/// load the table once and share it across many structures.
pub fn compute_with_radii(
    coords: &[[f64; 3]],
    atom_names: &[String],
    residue_names: &[String],
    probe_radius: f64,
    n_points: usize,
    table: &[AtomRadius],
) -> Vec<f64> {
    let n = coords.len();

    let radii: Vec<f64> = (0..n)
        .map(|i| lookup_radius(table, &residue_names[i], &atom_names[i]).unwrap_or(0.0))
        .collect();

    let inflated: Vec<f64> = radii.iter().map(|r| r + probe_radius).collect();
    let max_r = inflated.iter().cloned().fold(0.0_f64, f64::max);
    let cell = 2.0 * max_r;

    // Spatial hash grid for neighbour lookup.
    let mut min_c = [f64::INFINITY; 3];
    for c in coords {
        for k in 0..3 {
            if c[k] < min_c[k] {
                min_c[k] = c[k];
            }
        }
    }
    let key = |c: &[f64; 3]| -> (i64, i64, i64) {
        (
            ((c[0] - min_c[0]) / cell).floor() as i64,
            ((c[1] - min_c[1]) / cell).floor() as i64,
            ((c[2] - min_c[2]) / cell).floor() as i64,
        )
    };

    use std::collections::HashMap;
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, c) in coords.iter().enumerate() {
        if radii[i] > 0.0 {
            grid.entry(key(c)).or_default().push(i);
        }
    }

    let sphere = fibonacci_sphere(n_points);
    let mut sasa = vec![0.0_f64; n];

    for i in 0..n {
        let ri = radii[i];
        if ri <= 0.0 {
            continue;
        }
        let ri_inf = inflated[i];
        let area_per_pt = 4.0 * std::f64::consts::PI * ri_inf * ri_inf / (n_points as f64);
        let (kx, ky, kz) = key(&coords[i]);

        // Collect candidate neighbours from 3x3x3 surrounding cells.
        let mut nbrs: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &j in bucket {
                            if j != i {
                                nbrs.push(j);
                            }
                        }
                    }
                }
            }
        }

        let mut unburied = 0;
        for p in &sphere {
            let px = coords[i][0] + ri_inf * p[0];
            let py = coords[i][1] + ri_inf * p[1];
            let pz = coords[i][2] + ri_inf * p[2];
            let mut buried = false;
            for &j in &nbrs {
                let dx = px - coords[j][0];
                let dy = py - coords[j][1];
                let dz = pz - coords[j][2];
                let rj = inflated[j];
                if dx * dx + dy * dy + dz * dz < rj * rj {
                    buried = true;
                    break;
                }
            }
            if !buried {
                unburied += 1;
            }
        }
        sasa[i] = (unburied as f64) * area_per_pt;
    }

    sasa
}
