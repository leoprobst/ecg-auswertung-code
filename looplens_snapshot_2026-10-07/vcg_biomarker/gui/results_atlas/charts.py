"""Small, testable chart-data transformations shared with legacy views."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg


def auc_matrix_rgba(values: np.ndarray) -> np.ndarray:
    """AUC 0.5 is neutral; missing cells are transparent, never neutral."""
    data = np.asarray(values, dtype=float)
    finite = np.isfinite(data)
    normalized = np.clip((np.where(finite, data, .5) - .5) / .5, 0., 1.)
    rgba = pg.colormap.get("viridis").map(normalized, mode="byte")
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return rgba
