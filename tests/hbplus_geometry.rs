use _core::hbplus::{calculate, from_pdb, pdb, render, Options};
use std::collections::BTreeMap;

fn atom(n: usize, name: &str, res: &str, chain: char, id: i32, xyz: [f64; 3]) -> String {
    let el = name.chars().next().unwrap();
    format!("ATOM  {n:5}  {name:<3} {res:3} {chain}{id:4}    {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}          {el:>2}  \n",xyz[0],xyz[1],xyz[2],1.,0.)
}
fn fixture() -> String {
    [
        ("ND1", "HIS", 'A', 1, [0., 0., 0.]),
        ("CG", "HIS", 'A', 1, [-0.5, 0.866, 0.]),
        ("CE1", "HIS", 'A', 1, [-0.5, -0.866, 0.]),
        ("OD1", "ASP", 'B', 2, [2.8, 0., 0.]),
        ("CG", "ASP", 'B', 2, [4., 0., 0.]),
        ("CB", "ASP", 'B', 2, [4., 1.5, 0.]),
        ("OD2", "ASP", 'B', 2, [4.7, -1., 0.]),
    ]
    .iter()
    .enumerate()
    .map(|(n, (a, r, c, i, p))| atom(n + 1, a, r, *c, *i, *p))
    .collect()
}

#[test]
fn exact_analytic_bond_and_fixed_columns() {
    let text = fixture();
    let s = pdb::parse(&text).unwrap();
    let b = calculate(&s, &Options::default()).unwrap();
    assert_eq!(b.len(), 1);
    assert!((b[0].ha.unwrap() - 1.8).abs() < 1e-10);
    assert_eq!(b[0].dha, Some(180.));
    let out = render(&s, &b).unwrap();
    let lines: Vec<_> = out.lines().collect();
    assert_eq!(lines.len(), 9);
    let l = lines[8];
    assert_eq!(l.len(), 75);
    assert_eq!(&l[0..13], "A0001-HIS ND1");
    assert_eq!(&l[14..27], "B0002-ASP OD1");
    assert_eq!(l[27..32].trim(), "2.80");
    assert_eq!(l[46..51].trim(), "180.0");
    assert_eq!(l[52..57].trim(), "1.80");
    assert_eq!(l[58..63].trim(), "180.0");
    assert_eq!(l[64..69].trim(), "180.0");
}

#[test]
fn cutoffs_and_antecedent_angles_change_bond_membership() {
    let s = pdb::parse(&fixture()).unwrap();
    for o in [
        Options {
            max_da: 2.799,
            ..Default::default()
        },
        Options {
            max_ha: 1.799,
            ..Default::default()
        },
    ] {
        assert!(calculate(&s, &o).unwrap().is_empty());
    }
    let mut s = s;
    let cg = s
        .atoms
        .iter()
        .position(|a| a.res == "ASP" && a.name == "CG")
        .unwrap();
    s.atoms[cg].xyz.0 = 1.6; // Antecedent faces the donor: invalid D-A-AA angle.
    assert!(calculate(&s, &Options::default()).unwrap().is_empty());
}

#[test]
fn override_modes_are_directional_and_near_covalent_contacts_excluded() {
    let mut s = pdb::parse(&fixture()).unwrap();
    for atom in &mut s.atoms {
        if atom.res == "ASP" {
            atom.xyz.0 += 0.4;
        }
    }
    let baseline = calculate(&s, &Options::default()).unwrap();
    let mut o = Options {
        max_ha: 3.2,
        max_da: 4.,
        ..Default::default()
    };
    o.donors.insert(("ASP".into(), "OD1".into()), 1);
    let b = calculate(&s, &o).unwrap();
    assert!(b.len() > baseline.len());
    assert!(b
        .iter()
        .any(|b| s.atoms[b.donor].res == "ASP" && s.atoms[b.acceptor].res == "HIS"));
    assert!(!b
        .iter()
        .any(|b| s.atoms[b.donor].residue == s.atoms[b.acceptor].residue));
    o.acceptors.insert(("HIS".into(), "ND1".into()), 0);
    assert_eq!(calculate(&s, &o).unwrap().len(), 1);
}

#[test]
fn rotation_translation_invariant() {
    let mut s = pdb::parse(&fixture()).unwrap();
    let before = calculate(&s, &Options::default()).unwrap();
    for atom in &mut s.atoms {
        let p = atom.xyz;
        atom.xyz = _core::hbplus::geometry::V(-p.1 + 11., p.0 - 7., p.2 + 4.);
    }
    let after = calculate(&s, &Options::default()).unwrap();
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after) {
        assert_eq!((a.donor, a.acceptor), (b.donor, b.acceptor));
        assert!((a.da - b.da).abs() < 1e-10);
        assert!((a.ha.unwrap() - b.ha.unwrap()).abs() < 1e-10);
    }
}

#[test]
fn parser_handles_model_insertion_and_negative_numbers() {
    let mut first = fixture().replacen("A   1 ", "A  -2B", 3);
    first = format!(
        "MODEL        1\n{first}ENDMDL\nMODEL        2\n{}ENDMDL\n",
        fixture()
    );
    let s = pdb::parse(&first).unwrap();
    assert_eq!(s.atoms.len(), 7);
    assert_eq!(s.atoms[0].res_id, -2);
    assert_eq!(s.atoms[0].insertion, 'B');
    let out = render(&s, &calculate(&s, &Options::default()).unwrap()).unwrap();
    assert!(out.lines().nth(8).unwrap().starts_with("A-002BHIS"));
}

#[test]
fn malformed_and_unsupported_data_fail_closed() {
    assert!(pdb::parse("ATOM      1").is_err());
    assert!(pdb::parse(&(fixture() + &fixture())).is_err());
    let alternate = fixture().replacen(" HIS", "AHIS", 1);
    assert!(pdb::parse(&alternate).is_err());
    assert!(from_pdb(&fixture().replace("HIS", "UNK"), &Options::default()).is_err());
    assert!(from_pdb(
        &fixture(),
        &Options {
            max_da: f64::NAN,
            ..Default::default()
        }
    )
    .is_err());
    let mut donors = BTreeMap::new();
    donors.insert(("TYR".into(), "OH".into()), 3);
    assert!(from_pdb(
        &fixture(),
        &Options {
            donors,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn explicit_hydrogen_is_optional_and_changes_direction() {
    let text = fixture() + &atom(8, "HD1", "HIS", 'A', 1, [0., 0., 1.]);
    let s = pdb::parse(&text).unwrap();
    assert_eq!(calculate(&s, &Options::default()).unwrap().len(), 1);
    assert!(calculate(
        &s,
        &Options {
            use_hydrogens: true,
            ..Default::default()
        }
    )
    .unwrap()
    .is_empty());
}

#[test]
fn disulfide_sulfurs_cannot_donate() {
    let text = atom(1, "SG", "CYS", 'A', 1, [0., 0., 0.])
        + &atom(2, "CB", "CYS", 'A', 1, [-1.8, 0., 0.])
        + &atom(3, "SG", "CYS", 'B', 1, [0., 2., 0.])
        + &atom(4, "CB", "CYS", 'B', 1, [0., 3.8, 0.])
        + &atom(5, "O", "HOH", 'C', 1, [2.5, 0., 0.]);
    let s = pdb::parse(&text).unwrap();
    let b = calculate(&s, &Options::default()).unwrap();
    assert!(!b.iter().any(|b| s.atoms[b.donor].name == "SG"));
}

#[test]
fn water_undefined_fields_are_not_fabricated() {
    let text =
        atom(1, "O", "HOH", 'A', 1, [0., 0., 0.]) + &atom(2, "O", "HOH", 'B', 1, [2.8, 0., 0.]);
    let s = pdb::parse(&text).unwrap();
    let b = calculate(&s, &Options::default()).unwrap();
    assert_eq!(b.len(), 2);
    assert!(b.iter().all(|x| x.ha.is_none() && x.dha.is_none()));
}

#[test]
fn explicit_conect_excludes_cross_chain_covalent_bonds() {
    let s = pdb::parse(&(fixture() + "CONECT    1    4\n")).unwrap();
    assert!(calculate(&s, &Options::default()).unwrap().is_empty());
}

#[test]
fn bare_ter_record_preserves_segment_boundary() {
    let text = atom(1, "C", "ALA", 'A', 1, [0., 0., 0.])
        + "TER\n"
        + &atom(2, "N", "ALA", 'A', 2, [1.3, 0., 0.]);
    let s = pdb::parse(&text).unwrap();
    assert_ne!(s.atoms[0].segment, s.atoms[1].segment);
}
