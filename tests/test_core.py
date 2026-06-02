"""Tests for pysc._core (the Rust extension module)."""
import pytest

from protein_interface._core import ScResult, compute_sc, compute_sc_batch

# 10 ALA CA atoms per group, spaced 3.8 Å apart, ~5 Å between groups.
# Enough to generate surface dots; chosen to avoid empty-surface errors.
_COORDS_A = [[i * 3.8, 0.0, 0.0] for i in range(10)]
_NAMES_A = ["CA"] * 10
_RES_A = ["ALA"] * 10

_COORDS_B = [[i * 3.8, 5.0, 0.0] for i in range(10)]
_NAMES_B = ["CA"] * 10
_RES_B = ["ALA"] * 10


def test_compute_sc_smoke():
    result = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B)
    assert isinstance(result, ScResult)
    assert -1.0 <= result.sc <= 1.0
    assert result.median_distance >= 0.0
    assert result.trimmed_area >= 0.0
    assert result.atoms_a > 0
    assert result.atoms_b > 0


def test_compute_sc_parallel_false():
    """parallel=False must produce the same result as parallel=True."""
    r1 = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B, parallel=True)
    r2 = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B, parallel=False)
    assert r1.sc == pytest.approx(r2.sc, abs=1e-9)
    assert r1.median_distance == pytest.approx(r2.median_distance, abs=1e-9)
    assert r1.trimmed_area == pytest.approx(r2.trimmed_area, abs=1e-9)


def test_compute_sc_batch_matches_serial():
    """Batched SC must return the same values as serial compute_sc calls."""
    inputs = [
        (_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B),
        (_COORDS_B, _NAMES_B, _RES_B, _COORDS_A, _NAMES_A, _RES_A),
    ]
    serial = [
        compute_sc(*inputs[0], parallel=False),
        compute_sc(*inputs[1], parallel=False),
    ]
    batched = compute_sc_batch(inputs, parallel=True)
    assert len(batched) == 2
    for expected, observed in zip(serial, batched):
        assert observed.sc == pytest.approx(expected.sc, abs=1e-9)
        assert observed.median_distance == pytest.approx(expected.median_distance, abs=1e-9)
        assert observed.trimmed_area == pytest.approx(expected.trimmed_area, abs=1e-9)
        assert observed.atoms_a == expected.atoms_a
        assert observed.atoms_b == expected.atoms_b


def test_compute_sc_batch_empty():
    assert compute_sc_batch([]) == []


def test_compute_sc_length_mismatch_a():
    """coords_a length != atom_names_a length → ValueError."""
    with pytest.raises(ValueError, match="same length"):
        compute_sc(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # 2 coords
            ["CA"],  # 1 name — mismatch
            ["ALA"],
            _COORDS_B,
            _NAMES_B,
            _RES_B,
        )


def test_compute_sc_length_mismatch_b():
    """coords_b length != residue_names_b length → ValueError."""
    with pytest.raises(ValueError, match="same length"):
        compute_sc(
            _COORDS_A,
            _NAMES_A,
            _RES_A,
            [[0.0, 0.0, 0.0]],
            ["CA"],
            ["ALA", "GLY"],  # mismatch
        )


def test_compute_sc_empty_group_a():
    """Empty coords_a → ValueError."""
    with pytest.raises(ValueError):
        compute_sc([], [], [], _COORDS_B, _NAMES_B, _RES_B)


def test_compute_sc_empty_group_b():
    """Empty coords_b → ValueError."""
    with pytest.raises(ValueError):
        compute_sc(_COORDS_A, _NAMES_A, _RES_A, [], [], [])


def test_sc_result_repr():
    result = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B)
    r = repr(result)
    assert r.startswith("ScResult(")
    assert "sc=" in r


def test_determinism():
    """Same input must give bit-for-bit identical output across two calls."""
    r1 = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B)
    r2 = compute_sc(_COORDS_A, _NAMES_A, _RES_A, _COORDS_B, _NAMES_B, _RES_B)
    assert r1.sc == r2.sc
    assert r1.median_distance == r2.median_distance
    assert r1.trimmed_area == r2.trimmed_area
