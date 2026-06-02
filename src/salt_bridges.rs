//! Cross-interface salt-bridge counting (distance-only geometric criterion).
//!
//! A salt bridge atom pair is counted when an anionic side-chain oxygen and a
//! cationic side-chain nitrogen on opposite sides of the interface lie within
//! `cutoff` Å (Barlow & Thornton 1983 use 4.0 Å between charged centres).

use std::collections::HashSet;

fn is_anion(residue: &str, atom: &str) -> bool {
    matches!(
        (residue, atom),
        ("ASP", "OD1") | ("ASP", "OD2") | ("GLU", "OE1") | ("GLU", "OE2")
    )
}

fn is_cation(residue: &str, atom: &str) -> bool {
    matches!(
        (residue, atom),
        ("LYS", "NZ")
            | ("ARG", "NE")
            | ("ARG", "NH1")
            | ("ARG", "NH2")
            | ("HIS", "ND1")
            | ("HIS", "NE2")
    )
}

pub fn count(
    coords_a: &[[f64; 3]],
    atom_names_a: &[String],
    residue_names_a: &[String],
    coords_b: &[[f64; 3]],
    atom_names_b: &[String],
    residue_names_b: &[String],
    cutoff: f64,
) -> usize {
    let cutoff2 = cutoff * cutoff;
    let mut count = 0usize;
    for i in 0..coords_a.len() {
        let a_an = is_anion(&residue_names_a[i], &atom_names_a[i]);
        let a_ca = is_cation(&residue_names_a[i], &atom_names_a[i]);
        if !a_an && !a_ca {
            continue;
        }
        for j in 0..coords_b.len() {
            let b_an = is_anion(&residue_names_b[j], &atom_names_b[j]);
            let b_ca = is_cation(&residue_names_b[j], &atom_names_b[j]);
            if !((a_an && b_ca) || (a_ca && b_an)) {
                continue;
            }
            let dx = coords_a[i][0] - coords_b[j][0];
            let dy = coords_a[i][1] - coords_b[j][1];
            let dz = coords_a[i][2] - coords_b[j][2];
            if dx * dx + dy * dy + dz * dz <= cutoff2 {
                count += 1;
            }
        }
    }
    count
}

pub fn count_residue_pairs(
    coords_a: &[[f64; 3]],
    atom_names_a: &[String],
    residue_names_a: &[String],
    residue_keys_a: &[String],
    coords_b: &[[f64; 3]],
    atom_names_b: &[String],
    residue_names_b: &[String],
    residue_keys_b: &[String],
    cutoff: f64,
) -> usize {
    let cutoff2 = cutoff * cutoff;
    let mut pairs: HashSet<(&str, &str)> = HashSet::new();
    for i in 0..coords_a.len() {
        let a_an = is_anion(&residue_names_a[i], &atom_names_a[i]);
        let a_ca = is_cation(&residue_names_a[i], &atom_names_a[i]);
        if !a_an && !a_ca {
            continue;
        }
        for j in 0..coords_b.len() {
            let b_an = is_anion(&residue_names_b[j], &atom_names_b[j]);
            let b_ca = is_cation(&residue_names_b[j], &atom_names_b[j]);
            if !((a_an && b_ca) || (a_ca && b_an)) {
                continue;
            }
            let dx = coords_a[i][0] - coords_b[j][0];
            let dy = coords_a[i][1] - coords_b[j][1];
            let dz = coords_a[i][2] - coords_b[j][2];
            if dx * dx + dy * dy + dz * dz <= cutoff2 {
                pairs.insert((residue_keys_a[i].as_str(), residue_keys_b[j].as_str()));
            }
        }
    }
    pairs.len()
}
