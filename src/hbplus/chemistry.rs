//! Standard amino-acid connectivity, independently specified from chemical
//! structures. No HBPLUS implementation or parameter files are used.
use super::geometry::{self, Locus};
use super::pdb::Structure;
use super::Options;
use std::collections::{HashMap, HashSet};

pub struct Chemistry {
    pub edges: Vec<Vec<usize>>,
    pub donor: Vec<bool>,
    pub acceptor: Vec<bool>,
    pub loci: Vec<Locus>,
    pub ca: HashMap<usize, usize>,
}

fn side_bonds(res: &str) -> Option<&'static str> {
    Some(match res {
        "ALA" | "GLY" => "",
        "ARG" => "CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2",
        "ASN" => "CB-CG CG-OD1 CG-ND2",
        "ASP" => "CB-CG CG-OD1 CG-OD2",
        "CYS" => "CB-SG",
        "GLN" => "CB-CG CG-CD CD-OE1 CD-NE2",
        "GLU" => "CB-CG CG-CD CD-OE1 CD-OE2",
        "HIS" => "CB-CG CG-ND1 ND1-CE1 CE1-NE2 NE2-CD2 CD2-CG",
        "ILE" => "CB-CG1 CB-CG2 CG1-CD1",
        "LEU" => "CB-CG CG-CD1 CG-CD2",
        "LYS" => "CB-CG CG-CD CD-CE CE-NZ",
        "MET" => "CB-CG CG-SD SD-CE",
        "PHE" => "CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ",
        "PRO" => "CB-CG CG-CD CD-N",
        "SER" => "CB-OG",
        "THR" => "CB-OG1 CB-CG2",
        "TRP" => {
            "CB-CG CG-CD1 CG-CD2 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2"
        }
        "TYR" => "CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ CZ-OH",
        "VAL" => "CB-CG1 CB-CG2",
        "HOH" | "WAT" => "",
        _ => return None,
    })
}
fn connect(edges: &mut [Vec<usize>], i: usize, j: usize) {
    if i != j && !edges[i].contains(&j) {
        edges[i].push(j);
        edges[j].push(i);
    }
}

pub fn build(s: &Structure, o: &Options) -> Result<Chemistry, String> {
    let a = &s.atoms;
    let mut edges = vec![Vec::new(); a.len()];
    let mut index = HashMap::new();
    let mut residues: Vec<Vec<usize>> = Vec::new();
    let mut ca = HashMap::new();
    for (i, x) in a.iter().enumerate() {
        if side_bonds(&x.res).is_none() {
            return Err(format!("unsupported residue {} at {}{}; this implementation currently supports the 20 standard amino acids and water",x.res,x.chain,x.res_id));
        }
        while residues.len() <= x.residue {
            residues.push(Vec::new());
        }
        if let Some(&first) = residues[x.residue].first() {
            if a[first].res != x.res {
                return Err("mixed residue names at one residue identifier".into());
            }
        }
        residues[x.residue].push(i);
        index.insert((x.residue, x.name.as_str()), i);
        if x.name == "CA" {
            ca.insert(x.residue, i);
        }
    }
    for ids in &residues {
        let Some(&i) = ids.first() else { continue };
        let r = &a[i];
        let backbone = if matches!(r.res.as_str(), "HOH" | "WAT") {
            ""
        } else {
            "N-CA CA-C C-O C-OXT CA-CB"
        };
        for bond in backbone
            .split_whitespace()
            .chain(side_bonds(&r.res).unwrap().split_whitespace())
        {
            let (x, y) = bond.split_once('-').unwrap();
            if let (Some(&u), Some(&v)) = (index.get(&(r.residue, x)), index.get(&(r.residue, y))) {
                connect(&mut edges, u, v);
            }
        }
    }
    // TER and chain identities delimit peptide polymers; distance detects breaks.
    let mut previous: HashMap<(usize, char), usize> = HashMap::new();
    for ids in &residues {
        let Some(&i) = ids.first() else { continue };
        let r = &a[i];
        if matches!(r.res.as_str(), "HOH" | "WAT") {
            continue;
        }
        if let Some(prev) = previous.insert((r.segment, r.chain), r.residue) {
            if let (Some(&c), Some(&n)) = (index.get(&(prev, "C")), index.get(&(r.residue, "N"))) {
                if a[c].xyz.distance(a[n].xyz) <= 1.8 {
                    connect(&mut edges, c, n);
                }
            }
        }
    }
    let serials: HashMap<_, _> = a.iter().enumerate().map(|(i, x)| (x.serial, i)).collect();
    for (x, y) in &s.conect {
        if let (Some(&i), Some(&j)) = (serials.get(x), serials.get(y)) {
            connect(&mut edges, i, j);
        }
    }
    // The public manual uses 3.0 A for cystine recognition, including inter-chain S-S.
    let sulfurs: Vec<_> = a
        .iter()
        .enumerate()
        .filter(|(_, x)| x.res == "CYS" && x.name == "SG")
        .map(|(i, _)| i)
        .collect();
    let mut cystine = HashSet::new();
    for (k, &i) in sulfurs.iter().enumerate() {
        for &j in &sulfurs[k + 1..] {
            if a[i].xyz.distance(a[j].xyz) <= 3.0 {
                connect(&mut edges, i, j);
                cystine.insert(i);
                cystine.insert(j);
            }
        }
    }
    // Coordinate-linked explicit H are optional. Production labelling removes H.
    if o.use_hydrogens {
        for (i, h) in a.iter().enumerate().filter(|(_, x)| x.is_h()) {
            let nearest = residues[h.residue]
                .iter()
                .copied()
                .filter(|&j| !a[j].is_h())
                .filter(|&j| {
                    h.xyz.distance(a[j].xyz) <= if a[j].element == "S" { 1.7 } else { 1.3 }
                })
                .min_by(|&j, &k| {
                    h.xyz
                        .distance(a[j].xyz)
                        .total_cmp(&h.xyz.distance(a[k].xyz))
                });
            if let Some(j) = nearest {
                connect(&mut edges, i, j);
            }
        }
    }
    let mut donor = vec![false; a.len()];
    let mut acceptor = vec![false; a.len()];
    let mut loci = vec![Locus::Unknown; a.len()];
    for (i, x) in a.iter().enumerate() {
        if x.is_h() {
            continue;
        }
        let water = matches!(x.res.as_str(), "HOH" | "WAT") && x.element == "O";
        let heavy: Vec<_> = edges[i].iter().copied().filter(|&j| !a[j].is_h()).collect();
        let nterm = x.name == "N" && heavy.len() == 1 && a[heavy[0]].name == "CA";
        donor[i] = (x.name == "N" && x.res != "PRO")
            || water
            || matches!(
                (x.res.as_str(), x.name.as_str()),
                ("HIS", "ND1" | "NE2")
                    | ("ASN", "ND2")
                    | ("GLN", "NE2")
                    | ("ARG", "NE" | "NH1" | "NH2")
                    | ("LYS", "NZ")
                    | ("SER", "OG")
                    | ("THR", "OG1")
                    | ("TYR", "OH")
                    | ("TRP", "NE1")
            )
            || (x.res == "CYS" && x.name == "SG" && !cystine.contains(&i));
        // The manual omits NE2, but the MIT-released labeller reference tables
        // contain NE2 acceptor records in every mode. Both ring N can accept.
        acceptor[i] = matches!(x.name.as_str(), "O" | "OXT")
            || water
            || matches!(
                (x.res.as_str(), x.name.as_str()),
                ("ASP", "OD1" | "OD2")
                    | ("GLU", "OE1" | "OE2")
                    | ("ASN", "OD1")
                    | ("GLN", "OE1")
                    | ("HIS", "ND1" | "NE2")
                    | ("SER", "OG")
                    | ("THR", "OG1")
                    | ("TYR", "OH")
                    | ("CYS", "SG")
                    | ("MET", "SD")
            );
        let key = (x.res.clone(), x.name.clone());
        if let Some(&n) = o.donors.get(&key) {
            donor[i] = n > 0;
        }
        if let Some(&n) = o.acceptors.get(&key) {
            acceptor[i] = n > 0;
        }
        if !donor[i] {
            continue;
        }
        let explicit: Vec<_> = edges[i]
            .iter()
            .filter(|&&j| a[j].is_h())
            .map(|&j| a[j].xyz)
            .collect();
        if o.use_hydrogens && !explicit.is_empty() {
            loci[i] = Locus::Points(explicit);
            continue;
        }
        let d = x.xyz;
        let get = |name: &str| index.get(&(x.residue, name)).map(|&j| a[j].xyz);
        let dd = heavy.first().copied();
        let ddd = dd.and_then(|j| edges[j].iter().copied().find(|&k| k != i && !a[k].is_h()));
        if let (true, Some(dd), Some(ddd)) = (nterm || x.res == "LYS" && x.name == "NZ", dd, ddd) {
            loci[i] = geometry::tetra_three(d, a[dd].xyz, a[ddd].xyz);
        } else if x.name == "N" {
            let ca = get("CA");
            let pc = heavy
                .iter()
                .copied()
                .find(|&j| a[j].name == "C" && a[j].residue != x.residue);
            if let (Some(ca), Some(pc)) = (ca, pc) {
                if o.kabsch_sander {
                    if let Some(&oxygen) = index.get(&(a[pc].residue, "O")) {
                        if let Some(u) = (a[pc].xyz - a[oxygen].xyz).unit() {
                            loci[i] = Locus::Points(vec![d + u]);
                        }
                    }
                } else {
                    loci[i] = geometry::bisector(d, ca, a[pc].xyz, 1., 4.);
                }
            }
        } else if heavy.len() == 2 && x.element == "N" {
            loci[i] = geometry::bisector(d, a[heavy[0]].xyz, a[heavy[1]].xyz, 1., 0.);
        } else if matches!(
            (x.res.as_str(), x.name.as_str()),
            ("SER", "OG") | ("THR", "OG1") | ("CYS", "SG")
        ) {
            if let Some(j) = dd {
                loci[i] = geometry::circle(
                    d,
                    a[j].xyz,
                    if x.element == "S" { 1.33 } else { 1. },
                    if x.element == "S" { 96. } else { 110. },
                );
            }
        } else if let (Some(j), Some(k)) = (dd, ddd) {
            // Acid-O donor overrides are outside table III. The planar 110-degree
            // O-H hypothesis is explicit and must be validated before EV6 use.
            let deg = if x.element == "O" { 110. } else { 120. };
            loci[i] = geometry::planar(d, a[j].xyz, a[k].xyz, 1., deg);
        }
    }
    Ok(Chemistry {
        edges,
        donor,
        acceptor,
        loci,
        ca,
    })
}

pub fn nearly_bonded(edges: &[Vec<usize>], start: usize, end: usize, depth: usize) -> bool {
    if start == end {
        return true;
    }
    let mut seen = HashSet::from([start]);
    let mut frontier = vec![start];
    for _ in 0..depth {
        let mut next = Vec::new();
        for i in frontier {
            for &j in &edges[i] {
                if j == end {
                    return true;
                }
                if seen.insert(j) {
                    next.push(j);
                }
            }
        }
        frontier = next;
    }
    false
}
