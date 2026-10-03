"""Analytical and CLI tests; these do not claim numerical HBPLUS parity."""
import subprocess
import sys

import pytest

from protein_interface.hbplus import hbplus_format


def atom(serial, name, res, chain, resid, xyz):
    x, y, z = xyz
    return (f"ATOM  {serial:5d}  {name:<3} {res:3} {chain}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{0:6.2f}          {name[0]:>2}  \n")


@pytest.fixture
def pdb_text():
    return "".join(atom(i + 1, *spec) for i, spec in enumerate([
        ("ND1", "HIS", "A", 1, (0, 0, 0)),
        ("CG", "HIS", "A", 1, (-0.5, 0.866, 0)),
        ("CE1", "HIS", "A", 1, (-0.5, -0.866, 0)),
        ("OD1", "ASP", "B", 2, (3.2, 0, 0)),
        ("CG", "ASP", "B", 2, (4.4, 0, 0)),
        ("CB", "ASP", "B", 2, (4.4, 1.5, 0)),
        ("OD2", "ASP", "B", 2, (5.1, -1, 0)),
    ]))


def run_cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "protein_interface.hbplus", *map(str, args)],
                          cwd=tmp_path, capture_output=True, text=True)


def test_upstream_column_contract(pdb_text):
    text = hbplus_format(pdb_text)
    rows = text.splitlines()[8:]
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 75
    assert (row[0], int(row[1:5]), row[6:9], row[9:13].strip()) == ("A", 1, "HIS", "ND1")
    assert (row[14], int(row[15:19]), row[20:23], row[23:27].strip()) == ("B", 2, "ASP", "OD1")
    assert [float(row[a:b]) for a, b in [(27, 32), (46, 51), (52, 57)]] == [3.2, 180.0, 2.2]


def test_cli_handles_upstream_arguments_and_spaces(tmp_path, pdb_text):
    pdb = tmp_path / "complex with spaces.pdb"
    pdb.write_text(pdb_text)
    result = run_cli(tmp_path, "-h", "3.2", "-d", "4.0", "-a", "90", "-E", "ASP", " OD1", "1", pdb, pdb)
    assert result.returncode == 0, result.stderr
    assert "UNVALIDATED" in result.stderr
    expected = hbplus_format(pdb_text, max_da=4.0, max_ha=3.2, donor_overrides=[("ASP", "OD1", 1)])
    actual = pdb.with_suffix(".hb2").read_text()
    assert actual == expected
    assert len(actual.splitlines()[8:]) == 2
    # The older one-argument RES+ATOM spelling is accepted too.
    result = run_cli(tmp_path, "-h", "3.2", "-d", "4.0", "-E", "ASP OD1", "1", pdb)
    assert result.returncode == 0
    assert pdb.with_suffix(".hb2").read_text() == expected


@pytest.mark.parametrize("args", [["-R"], ["-h", "nan"], ["-a", "181"], ["-E", "LYS", "NZ", "3"], ["-Z"], ["-d"]])
def test_unsupported_options_and_bad_values_fail(tmp_path, pdb_text, args):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(pdb_text)
    result = run_cli(tmp_path, *args, pdb)
    assert result.returncode == 2
    assert not pdb.with_suffix(".hb2").exists()


def test_refuses_to_overwrite_input(tmp_path, pdb_text):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(pdb_text)
    result = run_cli(tmp_path, "--output", pdb, pdb)
    assert result.returncode == 2
    assert pdb.read_text() == pdb_text


def test_empty_result_has_eight_headers(pdb_text):
    result = hbplus_format(pdb_text, max_da=1.0)
    assert len(result.splitlines()) == 8


def test_invalid_structures_raise_errors(pdb_text):
    for invalid in ["", "ATOM", pdb_text.replace("HIS", "UNK"), pdb_text + pdb_text]:
        with pytest.raises(ValueError):
            hbplus_format(invalid)
