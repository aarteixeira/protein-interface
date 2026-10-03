use super::geometry::V;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct Atom {
    pub serial: i32,
    pub name: String,
    pub res: String,
    pub chain: char,
    pub res_id: i32,
    pub insertion: char,
    pub xyz: V,
    pub element: String,
    pub hetero: bool,
    pub segment: usize,
    pub residue: usize,
}
impl Atom {
    pub fn category(&self) -> char {
        if self.hetero {
            'H'
        } else if matches!(self.name.as_str(), "N" | "CA" | "C" | "O" | "OXT") {
            'M'
        } else {
            'S'
        }
    }
    pub fn is_h(&self) -> bool {
        matches!(self.element.as_str(), "H" | "D")
    }
}
#[derive(Debug)]
pub struct Structure {
    pub atoms: Vec<Atom>,
    pub conect: Vec<(i32, i32)>,
}
fn col(s: &str, a: usize, b: usize) -> &str {
    s.get(a..b).unwrap_or("")
}
fn number<T: std::str::FromStr>(s: &str, label: &str, line: usize) -> Result<T, String> {
    s.trim()
        .parse()
        .map_err(|_| format!("line {line}: invalid {label}: {s:?}"))
}

/// First MODEL only. Reject alternate conformers instead of silently mixing
/// chemically incompatible atoms; callers must choose a coherent conformer.
pub fn parse(text: &str) -> Result<Structure, String> {
    let mut atoms = Vec::new();
    let mut conect = Vec::new();
    let mut segment = 0;
    let mut residue_map = BTreeMap::new();
    let mut identities = BTreeMap::new();
    let mut serials = BTreeMap::new();
    let mut model_seen = false;
    let mut model_done = false;
    for (line_no, line) in text.lines().enumerate() {
        if !line.is_ascii() {
            return Err(format!("line {}: PDB must be ASCII", line_no + 1));
        }
        let rec = line.get(..6).unwrap_or(line).trim();
        if rec == "MODEL" {
            if model_seen {
                model_done = true;
            }
            model_seen = true;
            continue;
        }
        if rec == "ENDMDL" {
            model_done = true;
            continue;
        }
        if rec == "CONECT" {
            let src: i32 = number(col(line, 6, 11), "CONECT serial", line_no + 1)?;
            for start in (11..line.len()).step_by(5) {
                let field = &line[start..(start + 5).min(line.len())];
                if !field.trim().is_empty() {
                    conect.push((src, number(field, "CONECT serial", line_no + 1)?));
                }
            }
        }
        if model_done {
            continue;
        }
        if rec == "TER" {
            segment += 1;
            continue;
        }
        if !matches!(rec, "ATOM" | "HETATM") {
            continue;
        }
        if line.len() < 54 {
            return Err(format!("line {}: truncated atom record", line_no + 1));
        }
        if col(line, 16, 17) != " " {
            return Err(format!(
                "line {}: select one alternate conformer and clear altLoc before analysis",
                line_no + 1
            ));
        }
        let name = col(line, 12, 16).trim().to_string();
        let res = col(line, 17, 20).trim().to_string();
        if name.is_empty() || res.is_empty() {
            return Err(format!(
                "line {}: missing atom or residue name",
                line_no + 1
            ));
        }
        let chain = line.as_bytes()[21] as char;
        let insertion = line.as_bytes()[26] as char;
        let serial = number(col(line, 6, 11), "serial", line_no + 1)?;
        let res_id = number(col(line, 22, 26), "residue number", line_no + 1)?;
        if !(-999..=9999).contains(&res_id) {
            return Err("residue number exceeds .hb2 width".into());
        }
        let xyz = V(
            number(col(line, 30, 38), "x", line_no + 1)?,
            number(col(line, 38, 46), "y", line_no + 1)?,
            number(col(line, 46, 54), "z", line_no + 1)?,
        );
        if ![xyz.0, xyz.1, xyz.2].iter().all(|v| v.is_finite()) {
            return Err(format!("line {}: non-finite coordinates", line_no + 1));
        }
        let key = (segment, chain, res_id, insertion);
        let next = residue_map.len();
        let residue = *residue_map.entry(key).or_insert(next);
        if identities
            .insert((key, name.clone()), res.clone())
            .is_some()
        {
            return Err(format!("line {}: duplicate atom identity", line_no + 1));
        }
        if serials.insert(serial, ()).is_some() {
            return Err(format!("line {}: duplicate atom serial", line_no + 1));
        }
        let mut element = col(line, 76, 78).trim().to_ascii_uppercase();
        if element.is_empty() {
            element = name
                .chars()
                .find(|c| c.is_ascii_alphabetic())
                .ok_or("invalid atom name")?
                .to_string();
        }
        atoms.push(Atom {
            serial,
            name,
            res,
            chain,
            res_id,
            insertion,
            xyz,
            element,
            hetero: rec == "HETATM",
            segment,
            residue,
        });
    }
    if atoms.is_empty() {
        return Err("no atoms in first PDB model".into());
    }
    // .hb2 has no segment identifier: repeated identities across TER are ambiguous.
    let mut unique = HashMap::new();
    for a in &atoms {
        if unique
            .insert((a.chain, a.res_id, a.insertion, &a.name), ())
            .is_some()
        {
            return Err("repeated atom identity across TER cannot be represented in .hb2".into());
        }
    }
    Ok(Structure { atoms, conect })
}
