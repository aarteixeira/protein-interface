"""Tests for pysc.io (biopython-based PDB parsing)."""
from __future__ import annotations

from pathlib import Path

import pytest

from protein_interface import from_pdb, from_structure
from protein_interface._core import ScResult

DATA = Path(__file__).parent / "data"
FYT = DATA / "1fyt.pdb"

# 1FYT: TCR (chains D, E) vs pMHC (chains A, B, C)
CHAINS_A = ["D", "E"]
CHAINS_B = ["A", "B", "C"]


def test_from_pdb_1fyt():
    """Canonical CCP4/Lawrence-Colman test case. SC should be in [0.45, 0.65]."""
    result = from_pdb(FYT, CHAINS_A, CHAINS_B)
    assert isinstance(result, ScResult)
    assert 0.45 <= result.sc <= 0.65, f"SC={result.sc:.4f} outside expected [0.45, 0.65]"
    assert result.atoms_a > 0
    assert result.atoms_b > 0


def test_from_pdb_default_chains_b():
    """chains_b=None should resolve to all chains not in chains_a."""
    r_explicit = from_pdb(FYT, CHAINS_A, CHAINS_B)
    r_implicit = from_pdb(FYT, CHAINS_A, chains_b=None)
    assert r_explicit.sc == pytest.approx(r_implicit.sc, abs=1e-9)
    assert r_explicit.atoms_b == r_implicit.atoms_b


def test_from_structure():
    """from_structure() with an already-parsed biopython Structure."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1fyt", str(FYT))
    result = from_structure(structure, CHAINS_A, CHAINS_B)
    assert 0.45 <= result.sc <= 0.65


def test_hetatm_excluded_by_default():
    """Adding HETATM atoms to the PDB must not change the SC result."""
    import tempfile

    # Read original PDB
    lines = FYT.read_text().splitlines(keepends=True)

    # Append a fake HETATM line (SO4 sulfate ion far from the interface)
    fake_hetatm = (
        "HETATM 9999  S   SO4 A 999      99.000  99.000  99.000  1.00  0.00           S  \n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, dir=DATA
    ) as tmp:
        tmp.writelines(lines)
        tmp.write(fake_hetatm)
        tmp_path = tmp.name

    try:
        r_orig = from_pdb(FYT, CHAINS_A, CHAINS_B)
        r_mod = from_pdb(tmp_path, CHAINS_A, CHAINS_B)
        # HETATM excluded by default → same result
        assert r_orig.sc == pytest.approx(r_mod.sc, abs=1e-9)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_model_index():
    """Requesting model=0 (default) and explicitly model=0 should agree."""
    r1 = from_pdb(FYT, CHAINS_A, CHAINS_B)
    r2 = from_pdb(FYT, CHAINS_A, CHAINS_B, model=0)
    assert r1.sc == pytest.approx(r2.sc, abs=1e-9)


def test_invalid_model_index():
    """Out-of-range model index should raise ValueError."""
    with pytest.raises(ValueError, match="model index"):
        from_pdb(FYT, CHAINS_A, CHAINS_B, model=999)
