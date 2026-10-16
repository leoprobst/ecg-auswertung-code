"""Reuse one singular-value decomposition for rank and 2-norm condition."""
from __future__ import annotations

import numpy as np


def design_rank_condition(matrix: np.ndarray) -> tuple[int, float]:
    """Match NumPy's default matrix_rank tolerance and cond(p=None).

    Do not use the Gram matrix: it squares the condition number and loses
    precision precisely where the scientific abstention gate matters.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("A nonempty two-dimensional design is required")
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(singular.dtype).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return rank, condition
