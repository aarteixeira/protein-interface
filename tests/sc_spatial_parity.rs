use std::fs;
use std::path::Path;

use sc_rs::sc::{types::Atom, vector3::Vec3, ScCalculator};

const TOL: f64 = 1.0e-9;

fn is_hydrogen(atom_name: &str, element: &str) -> bool {
    let elem = element.trim().to_ascii_uppercase();
    let name = atom_name.trim();
    elem == "H"
        || name.starts_with('H')
        || name.ends_with('H')
        || (name.contains('H')
            && name
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false))
}

fn extract_atoms(path: &str, chains: &[&str]) -> Vec<Atom> {
    let text = fs::read_to_string(Path::new(path)).unwrap();
    let mut atoms = Vec::new();
    for line in text.lines() {
        if !line.starts_with("ATOM  ") {
            continue;
        }
        if line.len() < 54 {
            continue;
        }
        let altloc = line.get(16..17).unwrap_or(" ");
        if altloc != " " && altloc != "A" {
            continue;
        }
        let chain = line.get(21..22).unwrap_or(" ");
        if !chains.iter().any(|c| *c == chain) {
            continue;
        }
        let atom_name = line.get(12..16).unwrap_or("").trim();
        let residue_name = line.get(17..20).unwrap_or("").trim();
        let element = if line.len() >= 78 {
            line.get(76..78).unwrap_or("").trim()
        } else {
            ""
        };
        if is_hydrogen(atom_name, element) {
            continue;
        }
        let x: f64 = line.get(30..38).unwrap_or("").trim().parse().unwrap();
        let y: f64 = line.get(38..46).unwrap_or("").trim().parse().unwrap();
        let z: f64 = line.get(46..54).unwrap_or("").trim().parse().unwrap();
        let mut atom = Atom::new();
        atom.atom = atom_name.to_string();
        atom.residue = residue_name.to_string();
        atom.coor = Vec3::new(x, y, z);
        atoms.push(atom);
    }
    atoms
}

fn run_case(
    path: &str,
    chains_a: &[&str],
    chains_b: &[&str],
    use_spatial_index: bool,
) -> sc_rs::sc::types::Results {
    let atoms_a = extract_atoms(path, chains_a);
    let atoms_b = extract_atoms(path, chains_b);
    let mut calc = ScCalculator::new();
    calc.settings_mut().enable_parallel = true;
    calc.settings_mut().use_spatial_index = use_spatial_index;
    for atom in atoms_a {
        calc.add_atom(0, atom).unwrap();
    }
    for atom in atoms_b {
        calc.add_atom(1, atom).unwrap();
    }
    calc.calc().unwrap()
}

fn assert_same(path: &str, chains_a: &[&str], chains_b: &[&str]) {
    let legacy = run_case(path, chains_a, chains_b, false);
    let spatial = run_case(path, chains_a, chains_b, true);
    assert_eq!(legacy.surfaces[0].n_atoms, spatial.surfaces[0].n_atoms);
    assert_eq!(legacy.surfaces[1].n_atoms, spatial.surfaces[1].n_atoms);
    assert!(
        (legacy.sc - spatial.sc).abs() < TOL,
        "SC mismatch: legacy={} spatial={}",
        legacy.sc,
        spatial.sc,
    );
    assert!(
        (legacy.distance - spatial.distance).abs() < TOL,
        "distance mismatch: legacy={} spatial={}",
        legacy.distance,
        spatial.distance,
    );
    assert!(
        (legacy.area - spatial.area).abs() < TOL,
        "area mismatch: legacy={} spatial={}",
        legacy.area,
        spatial.area,
    );
}

#[test]
fn spatial_index_matches_legacy_for_single_chain_1fyt() {
    assert_same("tests/data/1fyt.pdb", &["D"], &["A"]);
}

#[test]
fn spatial_index_matches_legacy_for_multichain_1fyt() {
    assert_same("tests/data/1fyt.pdb", &["D", "E"], &["A", "B", "C"]);
}

#[test]
fn spatial_index_matches_legacy_for_nanobody_antigen() {
    assert_same("tests/data/nb_ag_test.pdb", &["A"], &["L"]);
}
