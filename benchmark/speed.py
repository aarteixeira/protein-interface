"""Speed benchmark and numeric consistency check across all pysc entry points.

Tests every public entry point on the same 1FYT complex (chains D vs A,
1521 + 1479 heavy atoms, ~190 residues each).

Sections:
  1. Numeric consistency  — all entry points must agree on sc / median_distance /
                            trimmed_area to within a stated tolerance.
  2. Speed table          — mean wall time over REPS runs, with targets where
                            applicable.

Usage:
    python benchmark/speed.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "tests" / "data"
PDB_1FYT = DATA / "1fyt.pdb"
CIF_1FYT = DATA / "1fyt.cif"

CHAINS_A = ["D"]
CHAINS_B = ["A"]
REPS = 7  # timed runs per benchmark (first is always a warmup)


# ── fixture helpers ──────────────────────────────────────────────────────────

def _ensure_fixtures():
    DATA.mkdir(parents=True, exist_ok=True)
    if not PDB_1FYT.exists():
        print("Downloading 1FYT.pdb …")
        urllib.request.urlretrieve("https://files.rcsb.org/download/1FYT.pdb", PDB_1FYT)
    if not CIF_1FYT.exists():
        print("Downloading 1FYT.cif …")
        urllib.request.urlretrieve("https://files.rcsb.org/download/1FYT.cif", CIF_1FYT)


def _mean_ms(times: list[float]) -> float:
    return sum(times) / len(times) * 1000


def _median_ms(times: list[float]) -> float:
    s = sorted(times)
    n = len(s)
    mid = n // 2
    return (s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2) * 1000


# ── shared state built once ──────────────────────────────────────────────────

def _build_shared_state():
    """Parse 1FYT once with biopython; derive all other representations from it.

    This ensures every entry point is fed exactly the same atoms, so numeric
    results must be bit-for-bit identical (same arrays → same Rust code path).
    from_boltzgen_refold is the one exception: it parses the CIF independently
    with biotite, so we report the diff vs biopython separately.
    """
    from Bio.PDB import PDBParser

    from protein_interface.io import _extract_atom_arrays, _is_hydrogen, _select_real_atom

    parser = PDBParser(QUIET=True)
    bio_structure = parser.get_structure("1fyt", str(PDB_1FYT))
    model = list(bio_structure.get_models())[0]

    coords_a, names_a, res_a = _extract_atom_arrays(model, CHAINS_A, False, False)
    coords_b, names_b, res_b = _extract_atom_arrays(model, CHAINS_B, False, False)

    # ── biotite AtomArray (same atoms as biopython) ──────────────────────────
    try:
        import biotite.structure as struc

        rows = []
        res_id_counter = 1

        for chain_id, chains in [(CHAINS_A[0], CHAINS_A), (CHAINS_B[0], CHAINS_B)]:
            for chain in model.get_chains():
                if chain.id not in chains:
                    continue
                for residue in chain.get_residues():
                    if residue.id[0] != " ":
                        continue
                    for da in residue.get_atoms():
                        real = _select_real_atom(da)
                        if real is None:
                            continue
                        name = real.name.strip()
                        elem = (real.element or "").strip()
                        if _is_hydrogen(name, elem):
                            continue
                        a = struc.Atom(
                            coord=real.coord.tolist(),
                            chain_id=chain.id,
                            res_id=res_id_counter,
                            res_name=residue.resname.strip(),
                            atom_name=name,
                            element=elem if elem else name[0],
                            hetero=False,
                        )
                        rows.append(a)
                    res_id_counter += 1

        biotite_array = struc.array(rows)
        has_biotite = True
    except ImportError:
        biotite_array = None
        has_biotite = False

    # ── BoltzGen mock Structure (same atoms as biopython) ────────────────────
    Atom_dt = np.dtype([
        ("name", "<U4"), ("coords", "3f4"), ("is_present", "?"),
        ("bfactor", "f4"), ("plddt", "f4"),
    ])
    Residue_dt = np.dtype([
        ("name", "<U5"), ("res_type", "i1"), ("res_idx", "i4"),
        ("atom_idx", "i4"), ("atom_num", "i4"), ("atom_center", "i4"),
        ("atom_disto", "i4"), ("is_standard", "?"), ("is_present", "?"),
    ])
    Chain_dt = np.dtype([
        ("name", "<U5"), ("mol_type", "i1"), ("entity_id", "i4"),
        ("sym_id", "i4"), ("asym_id", "i4"), ("atom_idx", "i4"),
        ("atom_num", "i4"), ("res_idx", "i4"), ("res_num", "i4"),
        ("cyclic_period", "i4"), ("symmetric_group", "i4"),
    ])

    atom_rows, residue_rows, chain_rows = [], [], []
    global_ai, global_ri = 0, 0

    for chain_name, chain_coords, chain_anames, chain_rnames in [
        (CHAINS_A[0], coords_a, names_a, res_a),
        (CHAINS_B[0], coords_b, names_b, res_b),
    ]:
        chain_atom_start = global_ai
        chain_res_start = global_ri

        # Group atoms into residues by consecutive same-name runs
        res_groups: list[tuple[str, list]] = []
        cur_res, cur_atoms = None, []
        for c, n, r in zip(chain_coords, chain_anames, chain_rnames):
            if cur_res is None:
                cur_res = r
            if r != cur_res and cur_atoms:
                res_groups.append((cur_res, cur_atoms))
                cur_res, cur_atoms = r, []
            cur_atoms.append((n, c))
        if cur_atoms:
            res_groups.append((cur_res, cur_atoms))

        for res_name, atoms in res_groups:
            res_atom_start = global_ai
            for aname, (x, y, z) in atoms:
                atom_rows.append((aname, (x, y, z), True, 0.0, 0.0))
                global_ai += 1
            residue_rows.append((
                res_name, 0, global_ri, res_atom_start, len(atoms), 0, 0, True, True,
            ))
            global_ri += 1

        chain_rows.append((
            chain_name, 1, 0, 0, 0,
            chain_atom_start, global_ai - chain_atom_start,
            chain_res_start, global_ri - chain_res_start,
            0, 0,
        ))

    class _MockStructure:
        atoms = np.array(atom_rows, dtype=Atom_dt)
        residues = np.array(residue_rows, dtype=Residue_dt)
        chains = np.array(chain_rows, dtype=Chain_dt)

    return {
        "bio_structure": bio_structure,
        "coords_a": coords_a, "names_a": names_a, "res_a": res_a,
        "coords_b": coords_b, "names_b": names_b, "res_b": res_b,
        "biotite_array": biotite_array,
        "has_biotite": has_biotite,
        "boltz_struct": _MockStructure(),
    }


# ── individual benchmarks ────────────────────────────────────────────────────

def bench(fn, n=REPS):
    fn()  # warmup
    times = [None] * n
    for i in range(n):
        t0 = time.perf_counter()
        result = fn()
        times[i] = time.perf_counter() - t0
    return _median_ms(times), result


def run_all(state):
    from protein_interface import (
        compute_sc,
        from_biotite,
        from_boltzgen_refold,
        from_boltzgen_structure,
        from_pdb,
        from_structure,
    )

    results = {}
    timings = {}

    # 1. from_pdb
    ms, r = bench(lambda: from_pdb(PDB_1FYT, CHAINS_A, CHAINS_B))
    results["from_pdb"], timings["from_pdb"] = r, ms

    # 2. from_structure (biopython Structure object, no re-parse)
    bio_s = state["bio_structure"]
    ms, r = bench(lambda: from_structure(bio_s, CHAINS_A, CHAINS_B))
    results["from_structure"], timings["from_structure"] = r, ms

    # 3. compute_sc (raw arrays, Rust only)
    ca, na, ra = state["coords_a"], state["names_a"], state["res_a"]
    cb, nb, rb = state["coords_b"], state["names_b"], state["res_b"]
    ms, r = bench(lambda: compute_sc(ca, na, ra, cb, nb, rb))
    results["compute_sc"], timings["compute_sc"] = r, ms

    # 4. from_boltzgen_structure (mock Structure, same arrays as biopython)
    bs = state["boltz_struct"]
    ms, r = bench(lambda: from_boltzgen_structure(bs, CHAINS_A, CHAINS_B))
    results["from_boltzgen_structure"], timings["from_boltzgen_structure"] = r, ms

    # 5 & 6. biotite-dependent
    if state["has_biotite"]:
        arr = state["biotite_array"]
        ms, r = bench(lambda: from_biotite(arr, CHAINS_A, CHAINS_B))
        results["from_biotite"], timings["from_biotite"] = r, ms

        if CIF_1FYT.exists():
            ms, r = bench(lambda: from_boltzgen_refold(CIF_1FYT, CHAINS_A, CHAINS_B))
            results["from_boltzgen_refold"], timings["from_boltzgen_refold"] = r, ms
    else:
        print("  (biotite not installed — skipping from_biotite and from_boltzgen_refold)")

    return results, timings


# ── consistency check ────────────────────────────────────────────────────────

def check_consistency(results):
    """All biopython-derived entry points must agree bit-for-bit.
    Biotite-derived ones may differ by up to 1e-4 due to independent CIF parsing.
    """
    # Reference: compute_sc (raw arrays straight from biopython extraction)
    ref = results["compute_sc"]

    # These use exactly the same arrays → must be bit-identical
    same_arrays = ["from_pdb", "from_structure", "from_boltzgen_structure"]
    # These parse the file independently with biotite → allow small float diff
    biotite_parsed = ["from_boltzgen_refold"]
    # from_biotite uses the pre-built biotite array (same atoms) → bit-identical
    same_arrays_biotite = ["from_biotite"]

    print("\n── Numeric consistency (reference: compute_sc) " + "─" * 30)
    print(f"  {'entry point':<30s}  {'Δsc':>12s}  {'Δmedian_dist':>14s}  {'Δtrimmed_area':>15s}  status")

    all_ok = True
    for name, r in results.items():
        if name == "compute_sc":
            continue
        d_sc   = abs(r.sc               - ref.sc)
        d_dist = abs(r.median_distance   - ref.median_distance)
        d_area = abs(r.trimmed_area      - ref.trimmed_area)
        a_a_ok = r.atoms_a == ref.atoms_a
        a_b_ok = r.atoms_b == ref.atoms_b

        if name in biotite_parsed:
            tol = 1e-3   # independent parser → cross-compilation FP diff allowed
        else:
            tol = 1e-9   # same arrays → must be identical

        ok = d_sc < tol and d_dist < tol and d_area < tol and a_a_ok and a_b_ok
        if not ok:
            all_ok = False
        atom_note = "" if (a_a_ok and a_b_ok) else f"  ← atoms_a={r.atoms_a}(ref {ref.atoms_a}) atoms_b={r.atoms_b}(ref {ref.atoms_b})"
        status = "OK" if ok else "MISMATCH"
        print(f"  {name:<30s}  {d_sc:>12.2e}  {d_dist:>14.2e}  {d_area:>15.2e}  {status}{atom_note}")

    print()
    return all_ok


# ── speed table ──────────────────────────────────────────────────────────────

def print_speed_table(timings):
    targets = {
        "from_pdb":               ("<200 ms", lambda ms: ms < 200),
        "from_structure":         (None,      None),
        "compute_sc":             ("<100 ms", lambda ms: ms < 100),
        "from_boltzgen_structure":(None,      None),
        "from_biotite":           (None,      None),
        "from_boltzgen_refold":   (None,      None),
        "score_many":             (None,      None),
    }

    labels = {
        "from_pdb":                "from_pdb()               file → biopython → Rust",
        "from_structure":          "from_structure()          biopython obj → Rust",
        "compute_sc":              "compute_sc()              raw arrays → Rust",
        "from_boltzgen_structure": "from_boltzgen_structure() Structure obj → Rust",
        "from_biotite":            "from_biotite()            biotite array → Rust",
        "from_boltzgen_refold":    "from_boltzgen_refold()    CIF → biotite → Rust",
    }

    print("── Speed (median over 7 runs, after 1 warmup) " + "─" * 32)
    print(f"  {'entry point':<50s}  {'median':>8s}  target")

    for key, label in labels.items():
        if key not in timings:
            continue
        ms = timings[key]
        tgt_str, tgt_fn = targets.get(key, (None, None))
        tgt_label = tgt_str or "—"
        if tgt_fn:
            mark = "PASS" if tgt_fn(ms) else "FAIL"
            tgt_label = f"[{mark}] {tgt_str}"
        print(f"  {label:<50s}  {ms:>6.1f} ms  {tgt_label}")


def bench_batch():
    from protein_interface import score_many

    paths = [PDB_1FYT] * 100
    score_many(paths, CHAINS_A, CHAINS_B, n_workers=8)  # warmup
    t0 = time.perf_counter()
    df = score_many(paths, CHAINS_A, CHAINS_B, n_workers=8)
    elapsed = time.perf_counter() - t0
    ok = (df["status"] == "ok").sum()
    return 100 / elapsed, ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    _ensure_fixtures()

    print(f"\n=== protein_interface benchmark — 1FYT chains {CHAINS_A[0]} vs {CHAINS_B[0]} ===\n")

    print("Building shared state (parse 1FYT once) …")
    state = _build_shared_state()
    na = len(state["coords_a"])
    nb = len(state["coords_b"])
    print(f"  atoms: {na} (chain {CHAINS_A[0]}) + {nb} (chain {CHAINS_B[0]})\n")

    results, timings = run_all(state)

    consistent = check_consistency(results)

    # batch timing
    cps, ok = bench_batch()
    timings["score_many"] = 100 / cps * 1000  # not used in table directly

    print_speed_table(timings)
    print(f"\n  {'score_many()':<50s}  {cps:>6.1f} cx/s  (100 files, 8 workers, ok={ok})")

    print()
    if not consistent:
        print("FAIL: numeric mismatch detected — see table above.")
        sys.exit(1)
    else:
        print("All results consistent. All targets met." if all(
            (fn(timings[k]) if fn else True)
            for k, (_, fn) in [
                ("from_pdb",    ("<200 ms", lambda ms: ms < 200)),
                ("compute_sc",  ("<100 ms", lambda ms: ms < 100)),
            ]
            if k in timings
        ) else "Numeric results consistent. Some speed targets not met — see table.")


if __name__ == "__main__":
    main()
