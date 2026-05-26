"""Multiprocessing batch scoring of PDB files.

Uses ProcessPoolExecutor with the 'spawn' start method (required on macOS).
Rust-side Rayon parallelism is disabled by default to avoid oversubscription
when many worker processes are already running concurrently.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
from pathlib import Path


def _score_one(args: tuple) -> dict:
    """Top-level worker function (must be picklable — no closures)."""
    pdb_path, chains_a, chains_b, kwargs = args
    try:
        from protein_interface.io import from_pdb

        result = from_pdb(pdb_path, chains_a, chains_b, **kwargs)
        return {
            "path": str(pdb_path),
            "sc": result.sc,
            "median_distance": result.median_distance,
            "trimmed_area": result.trimmed_area,
            "atoms_a": result.atoms_a,
            "atoms_b": result.atoms_b,
            "status": "ok",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(pdb_path),
            "sc": float("nan"),
            "median_distance": float("nan"),
            "trimmed_area": float("nan"),
            "atoms_a": 0,
            "atoms_b": 0,
            "status": "error",
            "error": str(exc),
        }


def score_many(
    pdb_paths: list,
    chains_a: list[str],
    chains_b: list[str] | None = None,
    n_workers: int = 8,
    parallel: bool = False,
    **kwargs,
) -> "pd.DataFrame":
    """Score many PDB files in parallel.

    Args:
        pdb_paths:  list of file paths
        chains_a:   chain IDs for molecule A
        chains_b:   chain IDs for molecule B (None = complement of chains_a)
        n_workers:  number of worker processes
        parallel:   enable Rayon parallelism inside each worker (default False
                    to avoid oversubscription with multiple processes)
        **kwargs:   forwarded to from_pdb (model, include_hetatm, etc.)

    Returns:
        DataFrame with columns:
            path, sc, median_distance, trimmed_area, atoms_a, atoms_b,
            status ('ok' or 'error'), error (None or message string)
    """
    import pandas as pd

    kwargs["parallel"] = parallel

    args_list = [(str(p), chains_a, chains_b, kwargs) for p in pdb_paths]

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as executor:
        rows = list(executor.map(_score_one, args_list))

    return pd.DataFrame(rows)
