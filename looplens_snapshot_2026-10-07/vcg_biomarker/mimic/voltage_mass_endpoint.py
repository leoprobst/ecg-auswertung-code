"""Join anatomy to the already selected endpoint index, never select again."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lead_endpoint_analysis import prepare_lead_frame, LEAD_COVARIATES
from .voltage_mass import devereux_lv_mass_g, VOLTAGE_MARKERS, DISCORDANCE_ENDPOINTS
from .voltage_mass_models import compare_voltage_mass_models

VOLTAGE_MASS_ENDPOINTS = (*DISCORDANCE_ENDPOINTS, "reduced_lvef_strict",
                         "aortic_regurgitation", "aortic_regurgitation_severe")
ECHO_GEOMETRY_COLUMNS = ("measurement_id", "septal_thickness_cm",
    "posterior_wall_thickness_cm", "lv_internal_diameter_diastole_cm")


def analyze_voltage_mass_endpoint(universe: pd.DataFrame, geometry: pd.DataFrame, *,
                                  endpoint_id: str, iterations: int, seed: int,
                                  preview: bool) -> tuple[list[dict], list[dict]]:
    if endpoint_id not in VOLTAGE_MASS_ENDPOINTS:
        return [], []
    frame = prepare_lead_frame(universe)
    # No alternate ECG can replace an index lacking anatomical evidence.
    frame["lv_mass_g"] = np.nan
    if not geometry.empty and set(ECHO_GEOMETRY_COLUMNS) <= set(geometry):
        if geometry.measurement_id.isna().any() or geometry.measurement_id.duplicated().any():
            raise ValueError("Echo geometry requires unique known measurement IDs")
        if {"nearest_echo_measurement_id", "nearest_echo_delta_hours"} <= set(frame):
            ids = frame.nearest_echo_measurement_id
            aligned = geometry.set_index("measurement_id").reindex(ids).reset_index(drop=True)
            masses = devereux_lv_mass_g(aligned.septal_thickness_cm,
                aligned.posterior_wall_thickness_cm, aligned.lv_internal_diameter_diastole_cm)
            delta = pd.to_numeric(frame.nearest_echo_delta_hours, errors="coerce").to_numpy()
            frame["lv_mass_g"] = np.where(np.isfinite(delta) & (np.abs(delta) <= 720), masses, np.nan)
    rows, models = [], []
    for marker in VOLTAGE_MARKERS:
        if marker not in frame:
            frame[marker] = np.nan
        table, fitted = compare_voltage_mass_models(frame, voltage=marker,
            covariates=LEAD_COVARIATES, iterations=iterations, seed=seed,
            confirm_association=not preview)
        table["endpoint_id"] = endpoint_id
        table["voltage_marker"] = marker
        table["patient_index_policy"] = "unchanged_reference_blind_endpoint_index"
        table["anatomy_alignment"] = "master_nearest_echo_within_30_days_no_ecg_reselection"
        table["analysis_temporality"] = "cross_sectional_not_ecg_time_prediction"
        table["component_interpretation"] = (
            "voltage_and_echo_estimated_lv_mass_not_CMR_cellular_or_extracellular_volume"
        )
        table["clinical_claim_allowed"] = False
        table["reference_kind"] = "|".join(sorted(set(frame.get("reference_kind",
            pd.Series("unknown", index=frame.index)).dropna().astype(str)))) or "unknown"
        table["sampling_scope"] = "pilot_enriched_descriptive" if preview else "development_index_cohort"
        table["probability_metrics_interpretable"] = not preview
        if preview:
            # Enriched pilot sampling does not estimate population Brier risk.
            table["validation_brier"] = np.nan
        rows.extend(table.to_dict("records"))
        models.append({"endpoint_id": endpoint_id, "voltage_marker": marker, "models": fitted})
    return rows, models
