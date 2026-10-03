use super::{calculate, pdb, render, Options};
use std::fs;
use std::path::PathBuf;

const HELP: &str = "protein-interface-hbplus [options] cleaned.pdb [original.pdb]
Independent protein hydrogen-bond geometry; HBPLUS numerical parity UNVALIDATED.
  -h/-H FLOAT       maximum H-A distance (2.5 A); --help for this help
  -d/-D FLOAT       maximum D-A distance (3.9 A)
  -a FLOAT          minimum DHA, HAA, DAA angles (90 degrees)
  -A DHA HAA DAA    separate minimum angles
  -E RES ATOM N     donor override (acid O: 1; any donor: 0 to disable)
  -e RES ATOM N     acceptor override (0 to disable, 1-3 to enable)
                    also accepts one quoted seven-character RES+ATOM argument
  -v/-V N          covalent-bond exclusion depth (2)
  -K / -k          Kabsch-Sander / Pauling backbone N-H placement
  --use-hydrogens   use explicit donor hydrogens when available
  --output PATH    .hb2 output (default: input basename in current directory)
  --version        print implementation identity
Unsupported options/residues fail rather than silently change results.
Existing protein-interface distance-only metrics are unaffected.";

fn take<'a>(args: &'a [String], i: &mut usize) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or("missing option argument".into())
}
fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("invalid numeric argument: {s}"))
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut o = Options::default();
    let mut paths = Vec::new();
    let mut output = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            "--version" => {
                println!(
                    "protein-interface-hbplus {} (HBPLUS parity unvalidated)",
                    env!("CARGO_PKG_VERSION")
                );
                return Ok(());
            }
            "-h" | "-H" => o.max_ha = num(take(args, &mut i)?)?,
            "-d" | "-D" => o.max_da = num(take(args, &mut i)?)?,
            "-a" => {
                let x = num(take(args, &mut i)?)?;
                o.min_dha = x;
                o.min_haa = x;
                o.min_daa = x;
            }
            "-A" => {
                o.min_dha = num(take(args, &mut i)?)?;
                o.min_haa = num(take(args, &mut i)?)?;
                o.min_daa = num(take(args, &mut i)?)?;
            }
            "-v" | "-V" => o.covalent_exclusion = num(take(args, &mut i)?)?,
            "-K" => o.kabsch_sander = true,
            "-k" => o.kabsch_sander = false,
            "--use-hydrogens" => o.use_hydrogens = true,
            "--output" => output = Some(PathBuf::from(take(args, &mut i)?)),
            "-e" | "-E" => {
                let donor = args[i] == "-E";
                let res_atom = take(args, &mut i)?;
                let (r, a) = if res_atom.len() == 7 && res_atom.is_ascii() {
                    (
                        res_atom[..3].trim().to_string(),
                        res_atom[3..].trim().to_string(),
                    )
                } else {
                    (
                        res_atom.trim().to_string(),
                        take(args, &mut i)?.trim().to_string(),
                    )
                };
                let n = num(take(args, &mut i)?)?;
                if donor {
                    o.donors.insert((r, a), n);
                } else {
                    o.acceptors.insert((r, a), n);
                }
            }
            "--" => {
                paths.extend(args[i + 1..].iter().map(PathBuf::from));
                break;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("unsupported option {arg}; use --help"))
            }
            arg => paths.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if paths.is_empty() || paths.len() > 2 {
        return Err("provide one cleaned PDB and optionally its original PDB; use --help".into());
    }
    o.validate()?;
    let input = &paths[0];
    let text = fs::read_to_string(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut s = pdb::parse(&text)?;
    if let Some(original) = paths.get(1) {
        if original != input {
            // Secondary input supplies CONECT records only, as in the public CLI.
            // Atom serials must identify the same atoms in the cleaned and original PDB.
            let original = pdb::parse(&fs::read_to_string(original).map_err(|e| e.to_string())?)?;
            for atom in &s.atoms {
                if let Some(other) = original.atoms.iter().find(|x| x.serial == atom.serial) {
                    if (other.chain, other.res_id, other.insertion, &other.name)
                        != (atom.chain, atom.res_id, atom.insertion, &atom.name)
                    {
                        return Err("cleaned/original atom serial mapping differs; normalize CONECT before use".into());
                    }
                }
            }
            s.conect.extend(original.conect);
        }
    }
    let bonds = calculate(&s, &o)?;
    let body = render(&s, &bonds)?;
    let output =
        output.unwrap_or_else(|| PathBuf::from(input.file_name().unwrap()).with_extension("hb2"));
    if output.exists()
        && paths
            .iter()
            .any(|p| fs::canonicalize(p).ok() == fs::canonicalize(&output).ok())
    {
        return Err("refusing to overwrite an input PDB".into());
    }
    fs::write(&output, body).map_err(|e| format!("{}: {e}", output.display()))?;
    eprintln!("protein-interface-hbplus: {} potential bonds -> {}; independent implementation, numerical parity UNVALIDATED",bonds.len(),output.display());
    Ok(())
}
