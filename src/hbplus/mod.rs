//! Independent implementation of the published protein HBPLUS geometry and
//! the CLI/output subset used by ProtonPottsMPNN. Numerical HBPLUS parity is
//! NOT established. Existing distance-only interface metrics are unchanged.
mod chemistry;
pub mod cli;
pub mod geometry;
pub mod pdb;
use geometry::angle;
use pdb::{Atom, Structure};
use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct Options {
    pub max_da: f64,
    pub max_ha: f64,
    pub min_dha: f64,
    pub min_haa: f64,
    pub min_daa: f64,
    pub covalent_exclusion: usize,
    pub kabsch_sander: bool,
    pub use_hydrogens: bool,
    pub donors: BTreeMap<(String, String), usize>,
    pub acceptors: BTreeMap<(String, String), usize>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_da: 3.9,
            max_ha: 2.5,
            min_dha: 90.,
            min_haa: 90.,
            min_daa: 90.,
            covalent_exclusion: 2,
            kabsch_sander: false,
            use_hydrogens: false,
            donors: BTreeMap::new(),
            acceptors: BTreeMap::new(),
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        for (name, x) in [("D-A", self.max_da), ("H-A", self.max_ha)] {
            if !x.is_finite() || x <= 0. || x >= 100. {
                return Err(format!(
                    "{name} distance must be finite, positive, and below 100 A"
                ));
            }
        }
        for x in [self.min_dha, self.min_haa, self.min_daa] {
            if !x.is_finite() || !(0.0..=180.).contains(&x) {
                return Err("angles must be finite and in [0,180]".into());
            }
        }
        if self.covalent_exclusion > 10 {
            return Err("covalent exclusion >10 is unsupported".into());
        }
        for ((r, a), n) in self.donors.iter().chain(&self.acceptors) {
            if !r.is_ascii()
                || !a.is_ascii()
                || r.is_empty()
                || r.len() > 3
                || a.is_empty()
                || a.len() > 4
                || *n > 3
            {
                return Err(
                    "override requires a residue (1-3 ASCII chars), atom (1-4), and count 0-3"
                        .into(),
                );
            }
        }
        for ((r, a), n) in &self.donors {
            if *n > 0
                && !(*n == 1
                    && matches!(
                        (r.as_str(), a.as_str()),
                        ("ASP", "OD1" | "OD2") | ("GLU", "OE1" | "OE2")
                    ))
            {
                return Err("positive donor overrides currently support only ASP OD1/OD2 and GLU OE1/OE2 with count 1".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub donor: usize,
    pub acceptor: usize,
    pub da: f64,
    pub dha: Option<f64>,
    pub ha: Option<f64>,
    pub haa: Option<f64>,
    pub daa: Option<f64>,
    pub ca_distance: Option<f64>,
    pub gap: Option<usize>,
}

pub fn calculate(s: &Structure, o: &Options) -> Result<Vec<Bond>, String> {
    o.validate()?;
    let chem = chemistry::build(s, o)?;
    let a = &s.atoms;
    let mut bonds = Vec::new();
    // A spatial broadphase is unnecessary for the 267-residue design complex.
    // Keep deterministic donor/acceptor order; optimize only with a parity test.
    for (i, d) in a.iter().enumerate().filter(|(i, _)| chem.donor[*i]) {
        for (j, acceptor) in a.iter().enumerate().filter(|(j, _)| chem.acceptor[*j]) {
            let da = d.xyz.distance(acceptor.xyz);
            if da > o.max_da
                || da < 1e-10
                || chemistry::nearly_bonded(&chem.edges, i, j, o.covalent_exclusion)
            {
                continue;
            }
            let antecedents: Vec<_> = chem.edges[j]
                .iter()
                .copied()
                .filter(|&k| !a[k].is_h())
                .collect();
            let daa = antecedents
                .iter()
                .filter_map(|&k| angle(d.xyz, acceptor.xyz, a[k].xyz))
                .min_by(f64::total_cmp);
            if daa.is_some_and(|x| x < o.min_daa) {
                continue;
            }
            let h = chem.loci[i].nearest(acceptor.xyz);
            let (dha, ha, haa) = if let Some(h) = h {
                let ha = h.distance(acceptor.xyz);
                let dha = angle(d.xyz, h, acceptor.xyz);
                let haa = antecedents
                    .iter()
                    .filter_map(|&k| angle(h, acceptor.xyz, a[k].xyz))
                    .min_by(f64::total_cmp);
                // Candidate interpretation for an imidazole acceptor's single
                // outward lone-pair direction. Compare to the released contact
                // corpus; this rule is empirical, not an HBPLUS parity claim.
                let haa_filter = if acceptor.res == "HIS" && antecedents.len() == 2 {
                    angle(
                        h,
                        acceptor.xyz,
                        (a[antecedents[0]].xyz + a[antecedents[1]].xyz) / 2.,
                    )
                } else {
                    haa
                };
                if ha > o.max_ha
                    || dha.is_none_or(|x| x < o.min_dha)
                    || haa_filter.is_some_and(|x| x < o.min_haa)
                {
                    continue;
                }
                (dha, Some(ha), haa)
            } else {
                // Published fallback assumes H on D->A at 1 A, but reports
                // undefined hydrogen fields as -1, as in the manual's water example.
                if (da - 1.).abs() > o.max_ha {
                    continue;
                }
                (None, None, None)
            };
            let ca_distance = chem
                .ca
                .get(&d.residue)
                .zip(chem.ca.get(&acceptor.residue))
                .map(|(&x, &y)| a[x].xyz.distance(a[y].xyz));
            let gap = (d.chain == acceptor.chain
                && d.segment == acceptor.segment
                && !d.hetero
                && !acceptor.hetero)
                .then_some(d.res_id.abs_diff(acceptor.res_id) as usize);
            bonds.push(Bond {
                donor: i,
                acceptor: j,
                da,
                dha,
                ha,
                haa,
                daa,
                ca_distance,
                gap,
            });
        }
    }
    Ok(bonds)
}

fn atom_id(a: &Atom) -> String {
    let chain = if a.chain == ' ' { '-' } else { a.chain };
    let ins = if a.insertion == ' ' { '-' } else { a.insertion };
    let name = if a.name.len() < 4 {
        format!(" {:<3}", a.name)
    } else {
        a.name.clone()
    };
    format!("{chain}{:04}{ins}{:>3}{name}", a.res_id, a.res)
}
pub fn render(s: &Structure, bonds: &[Bond]) -> Result<String, String> {
    let mut out=String::from("protein-interface HBPLUS-format geometry (independent implementation)\nNumerical equivalence to HBPLUS has NOT been established.\nProtein/water scope; see docs/hbplus.md for limitations and provenance.\nGeometry based on the public HBPLUS manual, sections 2.5 and 3.2-3.3.\nDistances in angstroms; angles in degrees; -1 means undefined.\nDirected potential contacts; alternative H positions can be mutually exclusive.\n<---DONOR--->  <-ACCEPTOR-->   DA CAT GAP    CA   DHA    HA   HAA   DAA COUNT\n------------- ------------- ----- -- --- ----- ----- ----- ----- ----- -----\n");
    for (n, b) in bonds.iter().enumerate() {
        let d = &s.atoms[b.donor];
        let a = &s.atoms[b.acceptor];
        let gap = b.gap.map(|x| x as i64).unwrap_or(-1);
        let ca = b.ca_distance.unwrap_or(-1.);
        if gap > 999 || ca >= 100. || n >= 99999 {
            return Err("bond record exceeds legacy .hb2 field width".into());
        }
        writeln!(
            out,
            "{} {}{:5.2} {}{} {:3} {:5.2} {:5.1} {:5.2} {:5.1} {:5.1} {:5}",
            atom_id(d),
            atom_id(a),
            b.da,
            d.category(),
            a.category(),
            gap,
            ca,
            b.dha.unwrap_or(-1.),
            b.ha.unwrap_or(-1.),
            b.haa.unwrap_or(-1.),
            b.daa.unwrap_or(-1.),
            n + 1
        )
        .unwrap();
    }
    Ok(out)
}

pub fn from_pdb(text: &str, options: &Options) -> Result<String, String> {
    let s = pdb::parse(text)?;
    render(&s, &calculate(&s, options)?)
}
