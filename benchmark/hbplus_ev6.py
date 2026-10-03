"""Isolate EV6 label changes caused by replacing saved HBPLUS observations.

Requires the pinned ProtonPottsMPNN labeller dependencies, never HBPLUS itself.
Both arms use the same coordinates, feature extraction, and released classifiers.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.io.pdb import PDBFile

from hbplus_reference import AA20, reference_rows


def reference_pool(frame):
    pool = {}
    for mode in ["default", "od1_oe1", "od2_oe2"]:
        for key, (da, dha, ha) in reference_rows(frame[frame["mode"] == mode]).items():
            if key in pool:
                pool[key]["hbplus_modes"] = "default,override" if "default" in pool[key]["hbplus_modes"] else "override"
                continue
            dc, di, dr, dn, ac, ai, ar, an = key
            pool[key] = dict(d_chain=dc, d_resi=str(di), d_resn=dr, d_atom=dn, d_ins=" ",
                             a_chain=ac, a_resi=str(ai), a_resn=ar, a_atom=an, a_ins=" ",
                             dist=da, dha_angle=np.nan if dha is None else dha,
                             ha_dist=np.nan if ha is None else ha,
                             hbplus_modes="default" if mode == "default" else "override")
    return list(pool.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protonpotts", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--parity-records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.protonpotts.resolve() / "foundry/models/mpnn/src"))
    os.environ["HBPLUS_PATH"] = str(args.binary.resolve())
    from mpnn.transforms.ev6 import EV6Predictor
    from mpnn.transforms.ev6.features import POOL_PARAMS

    predictor = EV6Predictor(his_thr=0.3, acid_thr=0.06)
    refs = pd.read_parquet(args.protonpotts / "labeller/data/hbonds.parquet")
    counts = pd.read_csv(args.parity_records)
    names = sorted(counts.loc[counts.reference_bonds.notna(), "pdb"].unique())
    if args.limit:
        names = names[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.protonpotts / "foundry/models/mpnn/src/mpnn/transforms/ev6/weights"
    transforms = weights.parent.parent
    coordinates = args.protonpotts / "benchmarks/data/neutron_benchmark/heavy"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"binary_sha256": digest(args.binary), "reference_sha256": digest(args.protonpotts / "labeller/data/hbonds.parquet"),
                "reference_engine_script_sha256": digest(Path(__file__)),
                "reference_helper_sha256": digest(Path(__file__).with_name("hbplus_reference.py")),
                "coordinates": {name: digest(coordinates / f"{name}.pdb") for name in names},
                "upstream_code": {str(p.relative_to(transforms)): digest(p) for p in sorted(transforms.rglob("*.py"))},
                "weights": {str(p.relative_to(weights)): digest(p) for p in sorted(weights.rglob("*.pkl"))},
                "thresholds": predictor.cfg, "pdbs": names,
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "biotite", "flaml", "scikit-learn", "xgboost", "lightgbm"]}}
    manifest = args.output / "protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("output contains a different protocol; choose a new directory")
    elif any(args.output.iterdir()):
        raise ValueError("existing checkpoints have no protocol manifest; choose a new directory")
    else:
        manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for name in names:
        checkpoint = args.output / f"{name}.csv"
        if checkpoint.exists():
            records.append(pd.read_csv(checkpoint))
            continue
        path = args.protonpotts / "benchmarks/data/neutron_benchmark/heavy" / f"{name}.pdb"
        aa = PDBFile.read(path).get_structure(model=1)
        aa = aa[np.isin(aa.res_name, list(AA20)) & ~np.isin(aa.element, ["H", "D"])]
        old = predictor.predict(aa, hbond_records={"params": POOL_PARAMS, "bonds": reference_pool(refs[refs.pdb == name])})
        new = predictor.predict(aa)  # Unmodified upstream reader launches the Rust executable.
        joined = old.merge(new, on=["chain", "res_id", "res_name"], suffixes=("_reference", "_rust"), validate="one_to_one")
        assert len(joined) == len(old) == len(new), name
        joined.insert(0, "pdb", name)
        joined["token_changed"] = joined.token_reference != joined.token_rust
        joined["abs_probability_delta"] = (joined.p_protonated_reference - joined.p_protonated_rust).abs()
        joined.to_csv(checkpoint, index=False)
        records.append(joined)
        print(name, len(joined), "titratable sites; changes", int(joined.token_changed.sum()), flush=True)
    result = pd.concat(records, ignore_index=True)
    result.to_csv(args.output / "labels.csv", index=False)
    summary = {"structures": len(names), "sites": len(result), "token_changes": int(result.token_changed.sum()),
               "max_probability_delta": float(result.abs_probability_delta.max()),
               "mean_probability_delta": float(result.abs_probability_delta.mean()),
               "thresholds": predictor.cfg,
               "limitation": "Same released EV6 classifiers in both arms; this tests substitution effects, not biological accuracy"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
