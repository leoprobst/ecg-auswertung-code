"""Conservative availability checks, never cross-source patient linkage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def known_patient_mask(values: pd.Series) -> pd.Series:
    """Reject common missing-ID sentinels without inventing a replacement ID."""
    tokens = values.astype("string").str.strip().str.casefold()
    # Numeric ranges are source-specific: zero can be a legitimate identifier.
    absent = tokens.isin(("", "nan", "none", "null", "unknown", "na", "n/a", "<na>", "inf", "-inf"))
    booleans = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    return values.notna() & ~absent.fillna(True) & ~booleans
