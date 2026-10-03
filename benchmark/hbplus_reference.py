"""Compare to MIT-released ProtonPottsMPNN HBPLUS observations, without HBPLUS.

Run with a conda Python containing pandas/pyarrow. External data are read-only.
This checks only the His/Asp/Glu incident bonds stored by the authors, not all
HBPLUS contacts. Missing inputs and coordinate discrepancies are reported.
"""
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

AA20 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
FUNCTIONAL = {("HIS", "ND1"), ("HIS", "NE2"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
MODES = {"default": [], "od1_oe1": ["-E", "ASP", " OD1", "1", "-E", "GLU", " OE1", "1"],
         "od2_oe2": ["-E", "ASP", " OD2", "1", "-E", "GLU", " OE2", "1"]}


def parse_hb2(text):
    rows = {}
    for line in text.splitlines()[8:]:
        donor = (line[0], int(line[1:5]), line[6:9].strip(), line[9:13].strip())
        acceptor = (line[14], int(line[15:19]), line[20:23].strip(), line[23:27].strip())
        if donor[2:] not in FUNCTIONAL and acceptor[2:] not in FUNCTIONAL:
            continue
        key = (*donor, *acceptor)
        if key in rows:
            raise ValueError(f"duplicate directed contact: {key}")
        values = [float(line[a:b]) for a, b in [(27, 32), (46, 51), (52, 57)]]
        rows[key] = [None if x < 0 else x for x in values]
    return rows


def reference_rows(df):
    rows = {}
    for r in df.itertuples():
        site = (r.chain, r.res_id, r.res_name, r.atom)
        partner = (r.p_chain, r.p_res_id, r.p_res_name, r.p_atom)
        key = (*site, *partner) if r.role == "donor" else (*partner, *site)
        values = [None if not math.isfinite(x) else x for x in (r.da, r.dha, r.ha)]
        if key in rows and rows[key] != values:
            raise ValueError(f"conflicting reference observations: {key}")
        rows[key] = values
    return rows


def compare(expected, actual):
    common = expected.keys() & actual.keys()
    tolerances = [0.011, 0.11, 0.011]
    geometry_bad = []
    exact = 0
    max_delta = [0.0, 0.0, 0.0]
    for key in sorted(common):
        e, a = expected[key], actual[key]
        if e == a:
            exact += 1
        bad = False
        for j, (x, y, tol) in enumerate(zip(e, a, tolerances)):
            if x is None or y is None:
                bad |= x != y
            else:
                max_delta[j] = max(max_delta[j], abs(x - y))
                bad |= abs(x - y) > tol
        if bad:
            geometry_bad.append({"key": key, "reference": e, "actual": a})
    return {"reference_bonds": len(expected), "actual_bonds": len(actual), "common_bonds": len(common),
            "missing": len(expected.keys() - actual.keys()), "extra": len(actual.keys() - expected.keys()),
            "geometry_mismatch": len(geometry_bad), "exact_geometry": exact,
            "max_da_delta": max_delta[0], "max_dha_delta": max_delta[1], "max_ha_delta": max_delta[2]}, {
                "missing": [{"key": k, "reference": expected[k]} for k in sorted(expected.keys() - actual.keys())],
                "extra": [{"key": k, "actual": actual[k]} for k in sorted(actual.keys() - expected.keys())],
                "geometry": geometry_bad,
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protonpotts", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=["calibration", "holdout", "all"], default="calibration")
    args = ap.parse_args()
    reference_path = args.protonpotts / "labeller/data/hbonds.parquet"
    ref = pd.read_parquet(reference_path)
    pdbs = sorted(ref.pdb.unique())
    # Lock 32 sorted identifiers for implementation development; the others are
    # untouched until the held-out run. This is engine parity, not ML validation.
    selected = pdbs[:32] if args.split == "calibration" else pdbs[32:] if args.split == "holdout" else pdbs
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("use a fresh output directory to preserve previous validation records")
    args.output.mkdir(parents=True, exist_ok=True)
    records, discrepancies = [], {}
    inputs = []
    for name in selected:
        path = args.protonpotts / "benchmarks/data/neutron_benchmark/heavy" / f"{name}.pdb"
        if not path.exists():
            records.append({"pdb": name, "mode": "all", "error": "structure absent"})
            continue
        inputs.append({"pdb": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        text = "\n".join(line for line in path.read_text().splitlines() if (
            line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() in AA20 and line[76:78].strip() not in {"H", "D"}) or line.startswith(("TER", "END", "MODEL"))) + "\n"
        # Reference parquet has no insertion-code key. Such inputs cannot be
        # compared without a richer reference; do not collapse their identities.
        if any(line[26:27] != " " for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))):
            records.append({"pdb": name, "mode": "all", "error": "insertion codes absent from reference keys"})
            continue
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{name}.pdb"
            p.write_text(text)
            for mode, overrides in MODES.items():
                cmd = [str(args.binary.resolve()), "-h", "3.2", "-d", "4.0", "-a", "90", *overrides, str(p), str(p)]
                result = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    records.append({"pdb": name, "mode": mode, "error": result.stderr.strip()})
                    continue
                expected = reference_rows(ref[(ref.pdb == name) & (ref["mode"] == mode)])
                actual = parse_hb2(p.with_suffix(".hb2").read_text())
                record, detail = compare(expected, actual)
                records.append({"pdb": name, "mode": mode, **record})
                discrepancies[f"{name}/{mode}"] = detail
        print(name, records[-1], flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_structure.csv", index=False)
    (args.output / "discrepancies.json").write_text(json.dumps(discrepancies, indent=2) + "\n")
    sums = {col: int(frame[col].sum()) for col in ["reference_bonds", "actual_bonds", "common_bonds", "missing", "extra", "geometry_mismatch", "exact_geometry"] if col in frame}
    summary = {"split": args.split, "selected_pdbs": list(selected), "inputs": inputs,
               "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
               "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "errors": int(frame.error.notna().sum()) if "error" in frame else 0,
               "totals": sums, "scope": "Only stored His/Asp/Glu incident bonds; all-bond parity untested",
               "coordinate_equivalence": "Inspect common-bond D-A differences; original labeller input hashes are unavailable",
               "exact_parity": False}
    summary["stored_contact_parity"] = bool(sums) and not any(sums[k] for k in ["missing", "extra", "geometry_mismatch"]) and summary["errors"] == 0
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"inputs", "selected_pdbs"}}, indent=2))


if __name__ == "__main__":
    main()
