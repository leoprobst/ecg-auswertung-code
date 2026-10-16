"""Prespecified single-lead measurements reused from the classical ECG pass.

Not standalone lead-monitoring performance: boundaries are jointly delineated
from the twelve-channel recording. No pathology or optimal lead is inferred here.
"""
from __future__ import annotations

import numpy as np

from ..core.data_loader import LEAD_NAMES_12, LEAD_MAP

LEAD_MEASUREMENT_VERSION = "1.0.0"
# family, existing primitive, unit. Do not duplicate algebraic QRS peak-to-peak
# (R + S) or replicate the same global duration in twelve predictive models.
LEAD_MEASUREMENT_SPEC = (
    ("qrs", "r_height", "mv"), ("qrs", "s_depth", "mv"),
    ("p", "positive_peak", "mv"), ("p", "negative_peak", "mv"),
    ("p", "terminal_negative_area_mvs", "mvs"),
    ("st", "j", "mv"), ("t", "t_max", "mv"), ("t", "t_min", "mv"),
)

# Local laboratory export only; the frozen production feature schema is unchanged.
LAB_LEAD_MEASUREMENT_SPEC = LEAD_MEASUREMENT_SPEC + (
    ("st", "st20", "mv"), ("st", "st60", "mv"), ("st", "st80", "mv"),
    ("qrs", "r_peak_time", "ms"), ("qrs", "s_duration", "ms"),
)


def measurement_column(lead: str, family: str, primitive: str, unit: str) -> str:
    return f"lead_measurement_{lead}_{family}_{primitive}_{unit}"


def empty_lead_measurements(*, extended=False) -> dict[str, float]:
    specs = LAB_LEAD_MEASUREMENT_SPEC if extended else LEAD_MEASUREMENT_SPEC
    return {measurement_column(lead, *spec): np.nan for lead in LEAD_NAMES_12 for spec in specs}


def store_lead_measurements(destination: dict[str, float], ecg: np.ndarray,
                            primitives: dict, gates: dict[str, bool], *, extended=False) -> None:
    """Export finite QC-gated primitives; failed/missing inputs stay explicit NaN."""
    destination.update(empty_lead_measurements(extended=extended))
    for lead in LEAD_NAMES_12:
        signal = ecg[:, LEAD_MAP[lead]]
        # A disconnected/flat channel is not a confidently measured zero.
        if not np.isfinite(signal).all() or np.ptp(signal) <= 1e-9:
            continue
        for family, name, unit in (LAB_LEAD_MEASUREMENT_SPEC if extended else LEAD_MEASUREMENT_SPEC):
            if not gates.get(family, False):
                continue
            value = primitives.get(family, {}).get(lead, {}).get(name)
            if value is not None and np.isfinite(value):
                destination[measurement_column(lead, family, name, unit)] = float(value)
