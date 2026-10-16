import numpy as np
import pytest

from vcg_biomarker.mimic.design_diagnostics import design_rank_condition


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["regular", "duplicate", "zero", "near_threshold", "wide"])
def test_matches_numpy_gates(dtype, kind):
    matrix = np.random.default_rng(91).normal(size=(100, 8)).astype(dtype)
    if kind == "duplicate":
        matrix[:, -1] = matrix[:, 0]
    elif kind == "zero":
        matrix[:] = 0
    elif kind == "near_threshold":
        matrix[:, -1] *= 1e-8
    elif kind == "wide":
        matrix = matrix.T
    rank, condition = design_rank_condition(matrix)
    assert rank == np.linalg.matrix_rank(matrix)
    np.testing.assert_allclose(condition, np.linalg.cond(matrix), rtol=1e-6)
    assert (rank < matrix.shape[1] or condition > 1e8) == (
        np.linalg.matrix_rank(matrix) < matrix.shape[1] or np.linalg.cond(matrix) > 1e8)


def test_only_one_decomposition(monkeypatch):
    original = np.linalg.svd
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", counted)
    assert design_rank_condition(np.eye(4)) == (4, 1.)
    assert len(calls) == 1


@pytest.mark.parametrize("matrix", [np.zeros((0, 2)), np.zeros((2, 0)), np.zeros(3)])
def test_invalid_shape(matrix):
    with pytest.raises(ValueError, match="nonempty"):
        design_rank_condition(matrix)
