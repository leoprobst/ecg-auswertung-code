"""Outcome-independent clinical questions for fixed anatomical 2D projections."""
from functools import lru_cache
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from vcg_biomarker.core.projection_metadata import projection_metadata
from .run_log import profiled_phase

PROJECTION_HYPOTHESIS_VERSION = "1.1.0"
PROJECTION_RESTRICTION_VERSION = "1.0.0"
P_PROJECTION_CONTEXT_COLUMNS = (
    "rhythm_context",
    "rhythm_context_confidence",
    "p_wave_coverage",
)
P_PROJECTION_ALLOWED_TECHNICAL_RHYTHMS = frozenset(
    {"regular", "wide_qrs_pattern"}
)
P_PROJECTION_MINIMUM_RHYTHM_CONFIDENCE = 0.60
P_PROJECTION_MINIMUM_WAVE_COVERAGE = 0.50
P_PROJECTION_RESTRICTION_STATUS = (
    "applied_regular_or_wide_qrs_technical_pattern_"
    "rhythm_confidence_ge_0_60_p_coverage_ge_0_50"
)

# No plane or geometry measure is selected from pilot outcomes. Every marker
# in a segment receives the same prespecified endpoint questions; multiplicity
# remains visible and is controlled by the downstream domain/global families.
SEGMENT_ENDPOINTS: dict[str, tuple[dict[str, str], ...]] = {
    "p": (
        {"endpoint_id": "echo_lae", "reference_modality": "echocardiography",
         "rationale": "atrial activation loop geometry versus independently measured left atrial enlargement",
         "restriction": "technically_evaluable_discrete_p_in_regular_or_wide_qrs_pattern",
         "restriction_operationalized": P_PROJECTION_RESTRICTION_STATUS},
    ),
    "qrs": (
        {"endpoint_id": "echo_lvh_strict", "reference_modality": "echocardiography",
         "rationale": "ventricular depolarization loop geometry versus structural hypertrophy",
         "restriction": "adequate_qrs_delineation"},
        {"endpoint_id": "reduced_lvef_strict", "reference_modality": "echocardiography",
         "rationale": "ventricular depolarization loop geometry versus systolic dysfunction",
         "restriction": "adequate_qrs_delineation"},
    ),
    "qrst": (
        {"endpoint_id": "reduced_lvef_strict", "reference_modality": "echocardiography",
         "rationale": "global depolarization-repolarization geometry versus systolic dysfunction",
         "restriction": "adequate_qrs_and_t_delineation"},
        {"endpoint_id": "severe_hypokalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "global electrical geometry versus independently measured potassium",
         "restriction": "adequate_qrs_and_t_delineation_and_prespecified_lab_window"},
    ),
    "st": (
        {"endpoint_id": "troponin_positive_any", "reference_modality": "laboratory",
         "rationale": "ST-loop geometry versus myocardial injury biomarker; not an infarction diagnosis",
         "restriction": "adequate_st_delineation_and_prespecified_lab_window"},
        {"endpoint_id": "hyperkalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "ST-loop geometry versus independently measured potassium",
         "restriction": "adequate_st_delineation_and_prespecified_lab_window"},
        {"endpoint_id": "severe_hypokalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "ST-loop geometry versus independently measured potassium",
         "restriction": "adequate_st_delineation_and_prespecified_lab_window"},
    ),
    "t": (
        {"endpoint_id": "hyperkalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "repolarization loop geometry versus independently measured potassium",
         "restriction": "adequate_t_delineation_and_prespecified_lab_window"},
        {"endpoint_id": "severe_hypokalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "repolarization loop geometry versus independently measured potassium",
         "restriction": "adequate_t_delineation_and_prespecified_lab_window"},
        {"endpoint_id": "severe_hyperkalemia_near_ecg", "reference_modality": "laboratory",
         "rationale": "repolarization loop geometry versus independently measured potassium",
         "restriction": "adequate_t_delineation_and_prespecified_lab_window"},
    ),
}

REFERENCES = {
    "p": "PMID:568646",
    "qrs": "PMID:149178",
    "qrst": "PMID:21353236",
    "st": "PMID:35458934",
    "t": "PMID:35458934",
}


def projection_hypothesis_rows(markers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker in markers:
        projection = marker.get("fixed_plane_projection")
        if not isinstance(projection, dict):
            continue
        segment = str(projection["segment"])
        for hypothesis in SEGMENT_ENDPOINTS[segment]:
            rows.append({
                "marker_id": str(marker["marker_id"]),
                "segment": segment,
                "plane": str(projection["plane"]),
                "axes": list(projection["axes"]),
                "geometry_measure": str(projection["geometry_measure"]),
                **hypothesis,
                "expected_direction": "two_sided_no_prespecified_monotonic_direction",
                "analysis_role": "prespecified_projection_hypothesis",
                "multiplicity_family": f"fixed_2d_projection__{hypothesis['endpoint_id']}",
                "reference": REFERENCES[segment],
                "clinical_threshold_validated": False,
                "monitoring_lead_recommendation_allowed": False,
                "hypothesis_version": PROJECTION_HYPOTHESIS_VERSION,
            })
    return rows


@lru_cache(maxsize=None)
def projection_hypothesis_for(marker_id: str, endpoint_id: str) -> dict[str, Any] | None:
    projection = projection_metadata(marker_id)
    if projection is None:
        return None
    segment = str(projection["segment"])
    match = next((item for item in SEGMENT_ENDPOINTS[segment] if item["endpoint_id"] == endpoint_id), None)
    if match is None:
        return None
    return {
        "projection_segment": segment,
        "projection_plane": projection["plane"],
        "projection_geometry_measure": projection["geometry_measure"],
        "projection_hypothesis_restriction": match["restriction"],
        "projection_hypothesis_restriction_operationalized": match.get(
            "restriction_operationalized",
            "true_marker_availability_and_endpoint_window",
        ),
        "projection_reference_modality": match["reference_modality"],
        "projection_multiplicity_family": f"fixed_2d_projection__{endpoint_id}",
    }


def projection_analysis_eligibility(
    frame: pd.DataFrame,
    marker_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the prespecified row mask for a fixed-plane projection.

    This is deliberately a technical signal-evaluability rule, not a rhythm
    diagnosis.  In particular, an absent device-report AF label is never
    interpreted as sinus rhythm.  P-loop inference requires positive evidence
    in the concurrently extracted delineation context.
    """

    projection = projection_metadata(marker_id)
    unrestricted: np.ndarray = np.ones(len(frame), dtype=bool)
    if projection is None or projection["segment"] != "p":
        return unrestricted, {
            "status": "not_applicable",
            "version": PROJECTION_RESTRICTION_VERSION,
            "required_columns_present": True,
        }
    missing = [column for column in P_PROJECTION_CONTEXT_COLUMNS if column not in frame]
    if missing:
        return np.zeros(len(frame), dtype=bool), {
            "status": "not_applied_missing_required_columns",
            "version": PROJECTION_RESTRICTION_VERSION,
            "required_columns_present": False,
            "missing_columns": missing,
        }
    rhythm = frame["rhythm_context"].astype("string")
    confidence = pd.to_numeric(frame["rhythm_context_confidence"], errors="coerce")
    coverage = pd.to_numeric(frame["p_wave_coverage"], errors="coerce")
    eligible = (
        rhythm.isin(P_PROJECTION_ALLOWED_TECHNICAL_RHYTHMS)
        & confidence.ge(P_PROJECTION_MINIMUM_RHYTHM_CONFIDENCE)
        & coverage.ge(P_PROJECTION_MINIMUM_WAVE_COVERAGE)
    ).fillna(False).to_numpy(dtype=bool)
    return eligible, {
        "status": P_PROJECTION_RESTRICTION_STATUS,
        "version": PROJECTION_RESTRICTION_VERSION,
        "required_columns_present": True,
    }


def projection_case_eligibility(
    marker_ids: Iterable[str],
    *,
    rhythm_context: object,
    rhythm_context_confidence: object,
    p_wave_coverage: object,
) -> tuple[bool, dict[str, Any]]:
    """Apply the cohort P-projection contract to one laboratory case.

    Non-P components are unrestricted.  When at least one component is a
    fixed-plane P projection, all three concurrent delineation measurements
    must be present and satisfy the frozen restriction.  This deliberately
    fails closed and never derives sinus rhythm from an absent diagnosis.
    """

    p_projection_ids = tuple(
        str(marker_id)
        for marker_id in marker_ids
        if (
            (metadata := projection_metadata(str(marker_id))) is not None
            and metadata["segment"] == "p"
        )
    )
    if not p_projection_ids:
        return True, {
            "status": "not_applicable",
            "version": PROJECTION_RESTRICTION_VERSION,
            "required_columns_present": True,
            "restricted_marker_ids": [],
            "abstention_reason": "",
        }
    context = pd.DataFrame(
        {
            "rhythm_context": [rhythm_context],
            "rhythm_context_confidence": [rhythm_context_confidence],
            "p_wave_coverage": [p_wave_coverage],
        }
    )
    eligible, metadata = projection_analysis_eligibility(
        context,
        p_projection_ids[0],
    )
    allowed = bool(eligible[0])
    details = {
        **metadata,
        "restricted_marker_ids": list(p_projection_ids),
        "rhythm_context": (
            None if pd.isna(rhythm_context) else str(rhythm_context)
        ),
        "rhythm_context_confidence": _finite_float_or_none(
            rhythm_context_confidence
        ),
        "p_wave_coverage": _finite_float_or_none(p_wave_coverage),
        "abstention_reason": "" if allowed else (
            "P-loop projection is not evaluable: concurrent technical "
            "rhythm evidence, rhythm confidence >=0.60 and P-wave "
            "coverage >=0.50 are required."
        ),
    }
    return allowed, details


def _finite_float_or_none(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


@profiled_phase("postprocess.projection_annotation")
def annotate_projection_hypothesis_frame(frame: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Attach the frozen role and its split-specific BH family to results."""
    output = frame.copy()
    output["projection_hypothesis_prespecified"] = False
    output["projection_hypothesis_q_value"] = np.nan
    output["projection_hypothesis_fdr_significant"] = False
    if output.empty or not {"feature_id", "endpoint_id", "p_value"} <= set(output):
        return output
    metadata = [projection_hypothesis_for(str(feature), str(endpoint))
                for feature, endpoint in zip(output.feature_id, output.endpoint_id)]
    output["projection_hypothesis_prespecified"] = [item is not None for item in metadata]
    for name in ("projection_segment", "projection_plane", "projection_geometry_measure",
                 "projection_hypothesis_restriction", "projection_hypothesis_restriction_operationalized",
                 "projection_reference_modality",
                 "projection_multiplicity_family"):
        output[name] = [item.get(name) if item else None for item in metadata]
    family_columns = ["endpoint_id"] + (["evaluation_split"] if "evaluation_split" in output else [])
    ordinary_restriction = output.projection_hypothesis_restriction_operationalized.eq(
        "true_marker_availability_and_endpoint_window"
    )
    p_restriction = (
        output.projection_hypothesis_restriction_operationalized.eq(
            P_PROJECTION_RESTRICTION_STATUS
        )
        & output.get(
            "analysis_restriction_status",
            pd.Series(None, index=output.index, dtype=object),
        ).eq(P_PROJECTION_RESTRICTION_STATUS)
        & output.get(
            "analysis_restriction_version",
            pd.Series(None, index=output.index, dtype=object),
        ).eq(PROJECTION_RESTRICTION_VERSION)
    )
    selected = output.loc[
        output.projection_hypothesis_prespecified
        & (ordinary_restriction | p_restriction)
    ]
    for _, indices in selected.groupby(family_columns, dropna=False).groups.items():
        index = list(indices)
        p = pd.to_numeric(output.loc[index, "p_value"], errors="coerce").to_numpy(float)
        valid = np.isfinite(p)
        if "inference_status" in output:
            valid &= output.loc[index, "inference_status"].eq("formal_inference").to_numpy(bool)
        if valid.any():
            rejected, q, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
            target = np.asarray(index, dtype=object)[valid]
            output.loc[target, "projection_hypothesis_q_value"] = q
            output.loc[target, "projection_hypothesis_fdr_significant"] = rejected
    output["projection_hypothesis_version"] = PROJECTION_HYPOTHESIS_VERSION
    return output
