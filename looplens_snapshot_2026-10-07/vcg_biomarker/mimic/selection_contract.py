"""Fail-closed eligibility shared by descriptive reports and score selection."""
from __future__ import annotations

import pandas as pd


def formal_selection_eligible(frame: pd.DataFrame) -> pd.Series:
    """Missing evidence remains displayable, but cannot select a winner."""
    required = {"inference_status", "assessability_gate_passed"}
    if not required.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    passed = frame["assessability_gate_passed"].astype("string").str.lower().isin(
        ["true", "1", "1.0"]
    )
    return frame["inference_status"].eq("formal_inference").fillna(False) & passed
