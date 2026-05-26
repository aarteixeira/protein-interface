//! Cross-interface hydrogen bond counting using a distance-only geometric criterion.
//!
//! Donors and acceptors are classified by (residue, atom) name; H-bonds are
//! counted as donor-acceptor heavy-atom pairs across the two atom groups within
//! `cutoff` Å. No hydrogens or angles are used: PDB inputs typically lack H,
//! and reconstructing them adds complexity without changing the relative
//! ranking of interfaces. Expect counts to agree with PyMOL's polar-contacts
//! definition within a few bonds.

fn is_donor(residue: &str, atom: &str) -> bool {
    // Backbone amide N (every residue except PRO).
    if atom == "N" && residue != "PRO" {
        return true;
    }
    match (residue, atom) {
        ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => true,
        ("ASN", "ND2") | ("GLN", "NE2") => true,
        ("LYS", "NZ") => true,
        ("ARG", "NE") | ("ARG", "NH1") | ("ARG", "NH2") => true,
        ("HIS", "ND1") | ("HIS", "NE2") => true,
        ("TRP", "NE1") => true,
        _ => false,
    }
}

fn is_acceptor(residue: &str, atom: &str) -> bool {
    // Backbone carbonyl O (and OXT on termini).
    if atom == "O" || atom == "OXT" {
        return true;
    }
    match (residue, atom) {
        ("ASP", "OD1") | ("ASP", "OD2") => true,
        ("GLU", "OE1") | ("GLU", "OE2") => true,
        ("ASN", "OD1") | ("GLN", "OE1") => true,
        ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => true,
        ("HIS", "ND1") | ("HIS", "NE2") => true,
        _ => false,
    }
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

    // Pair direction 1: donor in A, acceptor in B.
    // Pair direction 2: donor in B, acceptor in A.
    // A polar atom can be both donor and acceptor (e.g. SER OG) — that's
    // intended: it can form an H-bond in either direction with a partner.
    for i in 0..coords_a.len() {
        let a_is_donor = is_donor(&residue_names_a[i], &atom_names_a[i]);
        let a_is_acc = is_acceptor(&residue_names_a[i], &atom_names_a[i]);
        if !a_is_donor && !a_is_acc {
            continue;
        }
        for j in 0..coords_b.len() {
            let b_is_donor = is_donor(&residue_names_b[j], &atom_names_b[j]);
            let b_is_acc = is_acceptor(&residue_names_b[j], &atom_names_b[j]);
            let pair_ok = (a_is_donor && b_is_acc) || (a_is_acc && b_is_donor);
            if !pair_ok {
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
