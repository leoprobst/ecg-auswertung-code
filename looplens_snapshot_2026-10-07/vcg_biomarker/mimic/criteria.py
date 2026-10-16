from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
import pandas as pd

from ..core.biomarkers import BiomarkerRegistry
from .models import (
    ApplicabilityRule,
    CriterionCandidate,
    CriterionStage,
    EvidenceGrade,
    FrozenScoreSpec,
    ThresholdBand,
)
from .statistics import (
    MarkerModel,
    apply_frozen_score,
    bootstrap_operating_threshold_bands,
)
from .storage import content_hash
from .user_guidance import (
    biomarker_display_name,
    biomarker_tooltip,
    clinical_display_term,
)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fiducial_limitation import FiducialLimitationPolicy

CRITERION_CATALOG_VERSION = "2.2.0"
MINIMUM_REPLICATED_ROC_AUC = 0.60
MINIMUM_REPLICATED_ROC_AUC_CI_LOW = 0.50
LOCKED_TEST_ELIGIBLE_PANELS = frozenset(("primary_core", "sentinel"))

# Endpoints whose reference standard is the recorder's own measurement of the
# *same* waveform. They are kept as analytical concordance checks, never
# promoted past HYPOTHESIS, and never eligible for a diagnostic claim.
ANALYTICAL_CONCORDANCE_ENDPOINT_IDS: frozenset[str] = frozenset(
    {"qtc_gt_480", "qtc_gt_500"}
)


def is_analytical_concordance_endpoint(endpoint_id: str) -> bool:
    return str(endpoint_id) in ANALYTICAL_CONCORDANCE_ENDPOINT_IDS


def _time_to_event_primary_endpoint_ids() -> frozenset[str]:
    """Endpoints whose binary contrast is only a secondary sensitivity arm.

    Their control state requires documented observation through the horizon,
    so the binary cohort is conditioned on post-index follow-up. The
    time-to-event arm with censoring is the primary analysis; the binary
    result must not be graded as if it were.
    """

    from .label_registry import default_label_definitions

    return frozenset(
        definition.label_id
        for definition in default_label_definitions()
        if definition.primary_analysis_arm == "time_to_event"
    )


def is_time_to_event_primary_endpoint(endpoint_id: str) -> bool:
    return str(endpoint_id) in _time_to_event_primary_endpoint_ids()

OPERATING_POINT_LABELS = {
    "specificity_90": "90 % Spezifitaet",
    "specificity_95": "95 % Spezifitaet",
    "sensitivity_90": "90 % Sensitivitaet",
}


@dataclass(frozen=True)
class CriterionCatalogTables:
    candidates: pd.DataFrame
    thresholds: pd.DataFrame


def selected_single_marker_ids(
    *,
    endpoint_id: str,
    screening: pd.DataFrame,
    stability: pd.DataFrame,
    marker_models: Mapping[str, MarkerModel],
    maximum: int,
) -> tuple[str, ...]:
    """Return the exact scalar columns the criterion builder will consume.

    Validation uses this small projection to load values only after the
    statistical screen has selected the review candidates.  Keeping the
    selection here prevents the I/O optimizer and the catalog builder from
    ever drifting into two subtly different candidate sets.
    """

    rows = _validation_rows(screening, endpoint_id)
    stability_lookup = _stability_lookup(stability, endpoint_id)
    selected = _select_diverse_single_markers(
        rows,
        stability_lookup,
        marker_models,
        maximum=maximum,
    )
    return tuple(str(row["feature_id"]) for row in selected)


def build_endpoint_criterion_catalog(
    *,
    endpoint_id: str,
    train: pd.DataFrame,
    screening: pd.DataFrame,
    combinations: pd.DataFrame,
    stability: pd.DataFrame,
    marker_models: Mapping[str, MarkerModel],
    score_payloads: tuple[dict, ...],
    adjusted_results: Optional[pd.DataFrame] = None,
    development_run_id: str,
    snapshot_hash: str,
    baseline_id: str = "",
    bootstrap_iterations: int = 1000,
    random_seed: int = 20260725,
    maximum_single_candidates: int = 6,
    maximum_combination_candidates: int = 3,
) -> CriterionCatalogTables:
    """Create a compact, traceable criterion catalog for one endpoint.

    The broad screening tables remain the complete search space. This catalog
    intentionally contains only a small, diverse set that a human can review.
    Validation never changes a direction, scale or threshold learned in train.
    """
    validation_rows = _validation_rows(screening, endpoint_id)
    train_lookup = _training_row_lookup(screening, endpoint_id)
    stability_lookup = _stability_lookup(stability, endpoint_id)
    selected_single_ids = selected_single_marker_ids(
        endpoint_id=endpoint_id,
        screening=screening,
        stability=stability,
        marker_models=marker_models,
        maximum=maximum_single_candidates,
    )
    selected_by_id = {
        str(row.get("feature_id")): row
        for row in validation_rows.to_dict(orient="records")
    }
    selected_singles = [
        selected_by_id[feature_id]
        for feature_id in selected_single_ids
        if feature_id in selected_by_id
    ]
    selected_combinations = _select_combinations(
        combinations,
        endpoint_id,
        maximum=maximum_combination_candidates,
    )
    score_lookup = _score_payload_lookup(score_payloads, endpoint_id)

    candidates: list[dict] = []
    thresholds: list[dict] = []
    selected = [
        (row, marker_models.get(str(row.get("feature_id"))), None)
        for row in selected_singles
    ]
    selected.extend(
        (
            row,
            _score_model(score_lookup.get(str(row.get("feature_id")))),
            _score_spec(score_lookup.get(str(row.get("feature_id")))),
        )
        for row in selected_combinations
    )

    for selection_index, (row, model, score_spec) in enumerate(selected):
        if model is None:
            continue
        feature_id = str(row["feature_id"])
        try:
            train_values = (
                apply_frozen_score(train, score_spec)
                if score_spec is not None
                else pd.to_numeric(
                    train[feature_id], errors="coerce"
                ).to_numpy(dtype=float)
            )
        except (KeyError, TypeError, ValueError):
            continue
        try:
            uncertainty = bootstrap_operating_threshold_bands(
                train["subject_id"].to_numpy(),
                train_values,
                train["label"].to_numpy(dtype=np.int8),
                direction=model.direction,
                point_thresholds={
                    key: value
                    for key, value in model.thresholds.items()
                    if key in OPERATING_POINT_LABELS
                },
                iterations=bootstrap_iterations,
                seed=(
                    random_seed
                    + int(content_hash(f"{endpoint_id}:{feature_id}")[:8], 16)
                    + selection_index
                ),
            )
        except (ValueError, TypeError):
            continue

        train_row = train_lookup.get(
            (str(row.get("panel_id", "")), feature_id)
        )
        adjustment = _adjustment_evidence(
            adjusted_results,
            endpoint_id=endpoint_id,
            feature_id=feature_id,
        )
        evidence = _criterion_evidence(
            row,
            train_row=train_row,
            stability=stability_lookup.get(feature_id),
            adjustment=adjustment,
        )
        stage, grade = _evidence_stage(
            row,
            evidence=evidence,
            endpoint_id=endpoint_id,
        )
        components = (
            tuple(score_spec.components)
            if score_spec is not None
            else (feature_id,)
        )
        formula, unit = _formula_and_unit(feature_id, score_spec)
        applicability = _applicability_for_components(components)
        limitations = _limitations(
            row,
            score_spec,
            uncertainty,
            endpoint_id=endpoint_id,
        )
        criterion_id = "criterion_" + content_hash(
            {
                "version": CRITERION_CATALOG_VERSION,
                "endpoint_id": endpoint_id,
                "feature_id": feature_id,
                "components": components,
                "run_id": development_run_id,
                "snapshot_hash": snapshot_hash,
            }
        )[:20]
        bands = tuple(
            ThresholdBand(
                operating_point=operating_point,
                threshold=float(values["threshold_original"]),
                gray_zone_low=float(values["gray_zone_original_low"]),
                gray_zone_high=float(values["gray_zone_original_high"]),
                direction=model.direction,
                uncertainty_method=str(values["uncertainty_method"]),
                bootstrap_iterations=int(values["bootstrap_iterations"]),
                successful_bootstrap_iterations=int(
                    values["successful_bootstrap_iterations"]
                ),
            )
            for operating_point, values in uncertainty.items()
        )
        endpoint_term = clinical_display_term(endpoint_id)
        candidate = CriterionCandidate(
            criterion_id=criterion_id,
            endpoint_id=endpoint_id,
            display_name_de=(
                f"{endpoint_term.display_name_de}: "
                f"{biomarker_display_name(feature_id)}"
            ),
            formula=formula,
            component_ids=components,
            thresholds=bands,
            applicability=applicability,
            stage=stage,
            evidence_grade=grade,
            development_run_id=development_run_id,
            snapshot_hash=snapshot_hash,
            sensitivity=_optional_float(row.get("specificity_90_sensitivity")),
            specificity=_optional_float(row.get("specificity_90_specificity")),
            roc_auc=_optional_float(row.get("roc_auc")),
            confidence_interval=(
                _optional_float(row.get("roc_auc_ci_low")),
                _optional_float(row.get("roc_auc_ci_high")),
            ),
            baseline_id=baseline_id,
            limitations=limitations,
        )
        candidates.append(
            _flatten_candidate(
                candidate,
                row=row,
                feature_id=feature_id,
                formula_unit=unit,
                score_spec=score_spec,
                evidence=evidence,
            )
        )
        for band in bands:
            values = uncertainty[band.operating_point]
            thresholds.append(
                {
                    "criterion_id": criterion_id,
                    "endpoint_id": endpoint_id,
                    "feature_id": feature_id,
                    "operating_point": band.operating_point,
                    "operating_point_display_de": OPERATING_POINT_LABELS[
                        band.operating_point
                    ],
                    "threshold": band.threshold,
                    "unit": unit,
                    "direction": band.direction,
                    "positive_when": (
                        "groesser_als_grauzone"
                        if band.direction > 0
                        else "kleiner_als_grauzone"
                    ),
                    "gray_zone_low": band.gray_zone_low,
                    "gray_zone_high": band.gray_zone_high,
                    "uncertainty_method": band.uncertainty_method,
                    "bootstrap_iterations": band.bootstrap_iterations,
                    "successful_bootstrap_iterations": (
                        band.successful_bootstrap_iterations
                    ),
                    "sensitivity_validation": row.get(
                        f"{band.operating_point}_sensitivity"
                    ),
                    "specificity_validation": row.get(
                        f"{band.operating_point}_specificity"
                    ),
                    "ppv_validation": row.get(
                        f"{band.operating_point}_observed_ppv"
                    ),
                    "npv_validation": row.get(
                        f"{band.operating_point}_observed_npv"
                    ),
                    "threshold_source": "train",
                    "criterion_catalog_version": CRITERION_CATALOG_VERSION,
                }
            )
    return CriterionCatalogTables(
        candidates=pd.DataFrame(candidates),
        thresholds=pd.DataFrame(thresholds),
    )


# A criterion candidate needs a formal test behind it. Descriptive rows -- a
# marker measurable in fourteen recordings of which one is a case -- can show
# an AUC of 1.0 and used to win the catalog for every rare endpoint; they
# describe nothing a threshold could be fitted to.
CANDIDATE_INFERENCE_STATUS = "formal_inference"


def _validation_rows(frame: pd.DataFrame, endpoint_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        frame["endpoint_id"].astype(str).eq(endpoint_id)
        & frame["evaluation_split"].astype(str).eq("validation")
        & ~frame["panel_id"].astype(str).eq("classical")
    )
    from .selection_contract import formal_selection_eligible

    mask &= formal_selection_eligible(frame)
    output = frame.loc[mask].copy()
    output["roc_auc"] = pd.to_numeric(output["roc_auc"], errors="coerce")
    output["q_value"] = pd.to_numeric(
        output.get("q_value", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    )
    return output.loc[np.isfinite(output["roc_auc"])].reset_index(drop=True)


def _training_row_lookup(
    frame: pd.DataFrame,
    endpoint_id: str,
) -> dict[tuple[str, str], dict]:
    if frame.empty:
        return {}
    scoped = frame.loc[
        frame["endpoint_id"].astype(str).eq(endpoint_id)
        & frame["evaluation_split"].astype(str).eq("train")
        & ~frame["panel_id"].astype(str).eq("classical")
    ]
    return {
        (str(row.get("panel_id", "")), str(row.get("feature_id", ""))): row
        for row in scoped.to_dict(orient="records")
    }


def _stability_lookup(frame: pd.DataFrame, endpoint_id: str) -> dict[str, dict]:
    if frame.empty:
        return {}
    scoped = frame.loc[
        frame["endpoint_id"].astype(str).eq(endpoint_id)
        & frame["analysis_type"].astype(str).isin(
            ("single_marker", "exploratory_mimic_combination", "predefined_combination")
        )
    ]
    return {
        str(row["feature_id"]): row
        for row in scoped.to_dict(orient="records")
    }


def _select_diverse_single_markers(
    frame: pd.DataFrame,
    stability: Mapping[str, dict],
    models: Mapping[str, MarkerModel],
    *,
    maximum: int,
) -> list[dict]:
    if frame.empty or maximum <= 0:
        return []
    rows = frame.to_dict(orient="records")
    for row in rows:
        stable = stability.get(str(row.get("feature_id")), {})
        row["_direction_agreement"] = stable.get("direction_agreement") is True
    rows.sort(key=_candidate_sort_key)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()
    for require_new_family in (True, False):
        for row in rows:
            feature_id = str(row.get("feature_id", ""))
            family = str(row.get("conceptual_family", "unknown"))
            if feature_id not in models or feature_id in selected_ids:
                continue
            if require_new_family and family in selected_families:
                continue
            selected.append(row)
            selected_ids.add(feature_id)
            selected_families.add(family)
            if len(selected) >= maximum:
                return selected
    return selected


def _candidate_sort_key(row: dict) -> tuple:
    q_value = _optional_float(row.get("q_value"))
    ci_low = _optional_float(row.get("roc_auc_ci_low"))
    auc = _optional_float(row.get("roc_auc")) or 0.0
    missingness_value = _optional_float(row.get("missingness"))
    missingness = 1.0 if missingness_value is None else missingness_value
    return (
        not bool(row.get("fdr_significant", False)),
        not bool(row.get("_direction_agreement", False)),
        str(row.get("inference_status")) != "formal_inference",
        q_value if q_value is not None else 1.0,
        -(ci_low if ci_low is not None else 0.0),
        -auc,
        missingness,
        str(row.get("feature_id", "")),
    )


def _select_combinations(
    frame: pd.DataFrame,
    endpoint_id: str,
    *,
    maximum: int,
) -> list[dict]:
    if frame.empty or maximum <= 0:
        return []
    scoped = frame.loc[
        frame["endpoint_id"].astype(str).eq(endpoint_id)
        & frame["evaluation_split"].astype(str).eq("validation")
    ].copy()
    if scoped.empty:
        return []
    stable = pd.array(
        scoped.get(
            "validation_stable",
            pd.Series(False, index=scoped.index),
        ),
        dtype="boolean",
    ).fillna(False)
    selected = scoped.loc[stable].copy()
    if selected.empty:
        return []
    selected["roc_auc"] = pd.to_numeric(selected["roc_auc"], errors="coerce")
    selected["validation_delta_auc_vs_best_component"] = pd.to_numeric(
        selected.get(
            "validation_delta_auc_vs_best_component",
            pd.Series(np.nan, index=selected.index),
        ),
        errors="coerce",
    )
    selected = selected.sort_values(
        ["roc_auc", "validation_delta_auc_vs_best_component", "feature_id"],
        ascending=(False, False, True),
        kind="mergesort",
    )
    return selected.head(maximum).to_dict(orient="records")


def _score_payload_lookup(payloads: tuple[dict, ...], endpoint_id: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for payload in payloads:
        spec = payload.get("score_spec", {})
        if str(spec.get("endpoint_id")) == endpoint_id:
            output[str(spec.get("score_id"))] = payload
    return output


def _score_model(payload: Optional[dict]) -> Optional[MarkerModel]:
    if not payload or not payload.get("marker_model"):
        return None
    return MarkerModel.from_dict(payload["marker_model"])


def _score_spec(payload: Optional[dict]) -> Optional[FrozenScoreSpec]:
    if not payload or not payload.get("score_spec"):
        return None
    values = dict(payload["score_spec"])
    for key in ("components", "directions", "means", "standard_deviations"):
        values[key] = tuple(values[key])
    return FrozenScoreSpec(**values)


def _adjustment_evidence(
    frame: Optional[pd.DataFrame],
    *,
    endpoint_id: str,
    feature_id: str,
) -> dict[str, object]:
    missing = {
        "adjustment_confirmed": False,
        "adjustment_status": "missing_prespecified_adjusted_validation_model",
        "adjustment_model_id": None,
        "adjustment_incremental_auc": None,
        "adjustment_incremental_auc_ci_low": None,
        "adjustment_incremental_auc_ci_high": None,
    }
    if frame is None or frame.empty:
        return missing
    required = {
        "endpoint_id",
        "evaluation_split",
        "vcg_feature_id",
        "model_id",
    }
    if not required <= set(frame.columns):
        return {
            **missing,
            "adjustment_status": "adjusted_results_schema_incomplete",
        }
    scoped = frame.loc[
        frame["endpoint_id"].astype(str).eq(endpoint_id)
        & frame["evaluation_split"].astype(str).eq("validation")
        & frame["vcg_feature_id"].astype(str).eq(feature_id)
        & frame["model_id"].astype(str).eq("vcg_adjusted")
    ].copy()
    if scoped.empty:
        return missing
    row = scoped.iloc[0]
    ci_low = _optional_float(row.get("incremental_auc_ci_low"))
    ci_high = _optional_float(row.get("incremental_auc_ci_high"))
    incremental_auc = _optional_float(row.get("incremental_auc"))
    events_per_parameter = _optional_float(row.get("events_per_parameter_train"))
    confirmed = (
        str(row.get("metric_scope")) == "validation_holdout"
        and str(row.get("incremental_comparison_status")) == "formal_inference"
        and str(row.get("baseline_model_id")) == "covariates"
        and not bool(row.get("automatic_feature_selection", False))
        and events_per_parameter is not None
        and events_per_parameter >= 20.0
        and incremental_auc is not None
        and incremental_auc > 0.0
        and ci_low is not None
        and ci_low > 0.0
    )
    return {
        "adjustment_confirmed": confirmed,
        "adjustment_status": (
            "validation_incremental_auc_ci_excludes_zero"
            if confirmed
            else "adjusted_validation_increment_not_confirmed"
        ),
        "adjustment_model_id": str(row.get("model_id")),
        "adjustment_incremental_auc": incremental_auc,
        "adjustment_incremental_auc_ci_low": ci_low,
        "adjustment_incremental_auc_ci_high": ci_high,
    }


def _criterion_evidence(
    validation_row: Mapping,
    *,
    train_row: Optional[Mapping],
    stability: Optional[Mapping],
    adjustment: Mapping[str, object],
) -> dict[str, object]:
    train = train_row or {}
    train_auc = _optional_float(train.get("roc_auc"))
    validation_auc = _optional_float(validation_row.get("roc_auc"))
    validation_ci_low = _optional_float(validation_row.get("roc_auc_ci_low"))
    train_formal = str(train.get("inference_status", "")) == "formal_inference"
    validation_formal = (
        str(validation_row.get("inference_status", "")) == "formal_inference"
    )
    train_fdr = bool(train.get("fdr_significant", False))
    validation_fdr = bool(validation_row.get("fdr_significant", False))
    train_assessable = bool(train.get("assessability_gate_passed", False))
    validation_assessable = bool(
        validation_row.get("assessability_gate_passed", False)
    )
    train_missingness_sensitivity = bool(
        train.get("missingness_sensitivity_gate_passed", False)
    )
    validation_missingness_sensitivity = bool(
        validation_row.get("missingness_sensitivity_gate_passed", False)
    )
    direction_agreement = bool(
        stability and stability.get("direction_agreement") is True
    )
    eligible_role = (
        str(validation_row.get("panel_id", "")) in LOCKED_TEST_ELIGIBLE_PANELS
        and str(validation_row.get("feature_role", "")) == "candidate_biomarker"
        and str(validation_row.get("scientific_tier", "")) == "candidate"
        and str(validation_row.get("endpoint_feature_role", ""))
        != "same_ecg_source_comparator"
        and not bool(validation_row.get("post_outcome_hypothesis", False))
        and not bool(validation_row.get("exploratory_score", False))
    )
    train_effect = train_auc is not None and train_auc >= MINIMUM_REPLICATED_ROC_AUC
    validation_effect = (
        validation_auc is not None
        and validation_auc >= MINIMUM_REPLICATED_ROC_AUC
        and validation_ci_low is not None
        and validation_ci_low > MINIMUM_REPLICATED_ROC_AUC_CI_LOW
    )
    checks = {
        "train_formal_inference": train_formal,
        "validation_formal_inference": validation_formal,
        "train_fdr_significant": train_fdr,
        "validation_fdr_significant": validation_fdr,
        "direction_agreement": direction_agreement,
        "train_effect_floor_passed": train_effect,
        "validation_effect_floor_passed": validation_effect,
        "train_assessability_gate_passed": train_assessable,
        "validation_assessability_gate_passed": validation_assessable,
        "train_missingness_sensitivity_gate_passed": (
            train_missingness_sensitivity
        ),
        "validation_missingness_sensitivity_gate_passed": (
            validation_missingness_sensitivity
        ),
        "candidate_role_eligible": eligible_role,
        "adjustment_confirmed": bool(adjustment.get("adjustment_confirmed", False)),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return {
        **checks,
        **adjustment,
        "train_roc_auc": train_auc,
        "train_q_value": _optional_float(train.get("q_value")),
        "train_global_q_value": _optional_float(train.get("global_q_value")),
        "train_hierarchical_q_value": _optional_float(
            train.get("hierarchical_q_value")
        ),
        "validation_global_q_value": _optional_float(
            validation_row.get("global_q_value")
        ),
        "validation_hierarchical_q_value": _optional_float(
            validation_row.get("hierarchical_q_value")
        ),
        "train_test_yield": _optional_float(train.get("test_yield")),
        "validation_test_yield": _optional_float(
            validation_row.get("test_yield")
        ),
        "train_missingness_auc_lower_bound": _optional_float(
            train.get("missingness_auc_lower_bound")
        ),
        "validation_missingness_auc_lower_bound": _optional_float(
            validation_row.get("missingness_auc_lower_bound")
        ),
        "core_replication_evidence_passed": not failed,
        "replication_failure_reasons": "|".join(failed),
        "project_wide_multiplicity_confirmed": False,
    }


def _evidence_stage(
    row: Mapping,
    *,
    evidence: Mapping[str, object],
    endpoint_id: str = "",
) -> tuple[CriterionStage, EvidenceGrade]:
    resolved_endpoint = endpoint_id or row.get("endpoint_id", "")
    if is_analytical_concordance_endpoint(resolved_endpoint):
        # Agreement between two algorithms reading the same waveform cannot
        # advance a diagnostic claim, however strong it looks.
        return CriterionStage.HYPOTHESIS, EvidenceGrade.UNSTABLE
    if is_time_to_event_primary_endpoint(resolved_endpoint):
        # The binary contrast here is the sensitivity arm, not the primary
        # analysis, so it cannot by itself carry a replication grade.
        return CriterionStage.HYPOTHESIS, EvidenceGrade.UNSTABLE
    status = str(row.get("inference_status", ""))
    if status == "underpowered":
        return CriterionStage.TRAIN_STABLE, EvidenceGrade.UNDERPOWERED
    if bool(evidence.get("core_replication_evidence_passed", False)):
        # Project-wide global/hierarchical sensitivity is only available after
        # all endpoints have been assembled.  Finalization may promote this
        # development-stage candidate to VALIDATION_REPLICATED.
        return CriterionStage.TRAIN_STABLE, EvidenceGrade.DEVELOPMENT
    return CriterionStage.HYPOTHESIS, EvidenceGrade.UNSTABLE


def _formula_and_unit(
    feature_id: str,
    score_spec: Optional[FrozenScoreSpec],
) -> tuple[str, str]:
    if score_spec is not None:
        terms = [
            f"{direction:+d} * ({component} - {mean:.9g}) / {sd:.9g}"
            for component, direction, mean, sd in zip(
                score_spec.components,
                score_spec.directions,
                score_spec.means,
                score_spec.standard_deviations,
            )
        ]
        return (
            f"mean({'; '.join(terms)}) mit vollstaendigen Komponenten",
            "train-z score",
        )
    try:
        definition = BiomarkerRegistry.definition(feature_id)
    except ValueError:
        return feature_id, "unknown"
    return definition.formula or definition.description, definition.unit


def _applicability_for_components(
    components: tuple[str, ...],
) -> ApplicabilityRule:
    lowered = " ".join(components).casefold()
    exclusions = [
        "nicht auswertbare oder unsichere Wellengrenzen",
        "fehlende Pflichtableitungen",
    ]
    allowed_rhythms: tuple[str, ...] = ()
    if any(token in lowered for token in ("p_loop", "p_wave", "p_terminal")):
        allowed_rhythms = (
            "regular",
            "irregular",
            "wide_qrs_pattern",
        )
        exclusions.extend(("Vorhofflimmern", "Vorhofflattern ohne sichere P-Welle"))
    if any(token in lowered for token in ("st_", "t_loop", "qrs_t", "qt", "jt")):
        exclusions.append("T/U-Fusion ohne sichere T-Grenze")
    return ApplicabilityRule(
        required_leads=(
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ),
        minimum_signal_quality=0.60,
        minimum_fiducial_confidence=0.60,
        allowed_rhythm_classes=allowed_rhythms,
        exclusions=tuple(dict.fromkeys(exclusions)),
    )


def _limitations(
    row: Mapping,
    score_spec: Optional[FrozenScoreSpec],
    uncertainty: Mapping[str, Mapping],
    *,
    endpoint_id: str = "",
) -> tuple[str, ...]:
    limitations = [
        "MIMIC Train/Validation-Entwicklung; kein klinisch freigegebenes Diagnosekriterium.",
        "Grauzonen bilden die Train-Bootstrap-Unsicherheit des Grenzwerts ab.",
        "Die technische Wiederholbarkeit des Markers muss separat quantifiziert werden, bevor ein Freeze erlaubt ist.",
    ]
    if is_analytical_concordance_endpoint(
        endpoint_id or row.get("endpoint_id", "")
    ):
        limitations.append(
            "Referenzstandard ist die Geraetemessung desselben EKGs: analytische "
            "Konkordanz zweier Algorithmen auf derselben Kurve, keine "
            "diagnostische Genauigkeit gegen einen unabhaengigen Standard."
        )
    if is_time_to_event_primary_endpoint(
        endpoint_id or row.get("endpoint_id", "")
    ):
        limitations.append(
            "Binaerer Kontrast mit vollstaendiger Nachbeobachtung als "
            "Kontrollbedingung: die Kohorte ist auf Beobachtung nach dem Index "
            "konditioniert. Primaer ist das zeitabhaengige Ereignismodell mit "
            "Zensierung; dieser Arm ist nur Sensitivitaetsanalyse."
        )
    if score_spec is not None:
        limitations.append(
            "Explorativer, richtungsalignierter Score; fehlende Komponenten machen ihn nicht auswertbar."
        )
    if bool(row.get("post_outcome_hypothesis", False)):
        limitations.append(
            "Nach Outcome-Sichtung formulierter Discovery-Marker; unabhaengige Replikation erforderlich."
        )
    if any(
        values.get("uncertainty_method") == "point_only_not_estimated"
        for values in uncertainty.values()
    ):
        limitations.append(
            "Grenzwertunsicherheit konnte fuer diesen Kandidaten nicht geschaetzt werden."
        )
    return tuple(limitations)


def _flatten_candidate(
    candidate: CriterionCandidate,
    *,
    row: Mapping,
    feature_id: str,
    formula_unit: str,
    score_spec: Optional[FrozenScoreSpec],
    evidence: Mapping[str, object],
) -> dict:
    endpoint_term = clinical_display_term(candidate.endpoint_id)
    threshold_lookup = {
        band.operating_point: band for band in candidate.thresholds
    }
    primary_threshold = threshold_lookup.get("specificity_95")
    return {
        "criterion_id": candidate.criterion_id,
        "endpoint_id": candidate.endpoint_id,
        "endpoint_name_de": endpoint_term.display_name_de,
        "endpoint_category_de": endpoint_term.category_de,
        "endpoint_definition_de": endpoint_term.short_definition_de,
        "endpoint_source_de": endpoint_term.source_summary_de,
        "display_name_de": candidate.display_name_de,
        "feature_id": feature_id,
        "feature_name_de": biomarker_display_name(feature_id),
        "feature_tooltip_de": biomarker_tooltip(feature_id),
        "panel_id": row.get("panel_id"),
        "analysis_type": row.get("analysis_type"),
        "formula": candidate.formula,
        "unit": formula_unit,
        "component_ids": "|".join(candidate.component_ids),
        "component_count": len(candidate.component_ids),
        "component_ids_json": json.dumps(
            candidate.component_ids, ensure_ascii=True
        ),
        "score_spec_json": json.dumps(
            score_spec.to_dict() if score_spec is not None else None,
            ensure_ascii=True,
            sort_keys=True,
        ),
        "thresholds_json": json.dumps(
            [band.to_dict() for band in candidate.thresholds],
            ensure_ascii=True,
            sort_keys=True,
        ),
        "specificity_95_threshold": (
            primary_threshold.threshold
            if primary_threshold is not None
            else None
        ),
        "specificity_95_gray_zone_low": (
            primary_threshold.gray_zone_low
            if primary_threshold is not None
            else None
        ),
        "specificity_95_gray_zone_high": (
            primary_threshold.gray_zone_high
            if primary_threshold is not None
            else None
        ),
        "specificity_95_gray_zone": (
            f"{primary_threshold.gray_zone_low:.6g} bis "
            f"{primary_threshold.gray_zone_high:.6g}"
            if primary_threshold is not None
            else ""
        ),
        "specificity_95_direction": (
            primary_threshold.direction
            if primary_threshold is not None
            else None
        ),
        "applicability_json": json.dumps(
            candidate.applicability.to_dict(), ensure_ascii=True, sort_keys=True
        ),
        "stage": candidate.stage.value,
        "stage_de": {
            CriterionStage.HYPOTHESIS: "Hypothese",
            CriterionStage.TRAIN_STABLE: "im Training stabil",
            CriterionStage.VALIDATION_REPLICATED: "in Validation bestätigt",
            CriterionStage.FROZEN: "eingefroren",
            CriterionStage.LOCKED_TEST_VALIDATED: "im Locked Test bestätigt",
            CriterionStage.EXTERNAL_VALIDATED: "extern bestätigt",
        }[candidate.stage],
        "evidence_grade": candidate.evidence_grade.value,
        "evidence_grade_de": {
            EvidenceGrade.TECHNICAL: "technischer Vergleich",
            EvidenceGrade.DEVELOPMENT: "Entwicklung",
            EvidenceGrade.VALIDATION_REPLICATED: "Validation bestätigt",
            EvidenceGrade.FROZEN: "eingefroren",
            EvidenceGrade.LOCKED_TEST_VALIDATED: "Locked Test bestätigt",
            EvidenceGrade.EXTERNAL_VALIDATED: "extern bestätigt",
            EvidenceGrade.UNDERPOWERED: "unterpowert",
            EvidenceGrade.UNSTABLE: "instabil",
        }[candidate.evidence_grade],
        "development_run_id": candidate.development_run_id,
        "snapshot_hash": candidate.snapshot_hash,
        "sensitivity": candidate.sensitivity,
        "specificity": candidate.specificity,
        "roc_auc": candidate.roc_auc,
        "roc_auc_ci_low": candidate.confidence_interval[0],
        "roc_auc_ci_high": candidate.confidence_interval[1],
        "average_precision": row.get("average_precision"),
        "q_value": row.get("q_value"),
        "fdr_significant": row.get("fdr_significant", False),
        "global_q_value": row.get("global_q_value"),
        "global_fdr_significant": row.get("global_fdr_significant", False),
        "hierarchical_q_value": row.get("hierarchical_q_value"),
        "hierarchical_fdr_significant": row.get(
            "hierarchical_fdr_significant", False
        ),
        "inference_status": row.get("inference_status"),
        "missingness": row.get("missingness"),
        "test_yield": row.get("test_yield"),
        "positive_test_yield": row.get("positive_test_yield"),
        "negative_test_yield": row.get("negative_test_yield"),
        "test_yield_gap": row.get("test_yield_gap"),
        "assessability_status": row.get("assessability_status"),
        "assessability_gate_passed": row.get(
            "assessability_gate_passed", False
        ),
        "missingness_sensitivity_status": row.get(
            "missingness_sensitivity_status"
        ),
        "missingness_auc_lower_bound": row.get(
            "missingness_auc_lower_bound"
        ),
        "missingness_auc_upper_bound": row.get(
            "missingness_auc_upper_bound"
        ),
        "missingness_sensitivity_gate_passed": row.get(
            "missingness_sensitivity_gate_passed", False
        ),
        **evidence,
        "baseline_id": candidate.baseline_id,
        "limitations_json": json.dumps(
            candidate.limitations, ensure_ascii=True
        ),
        "threshold_source": "train",
        "evaluation_split": "validation",
        "exploratory_score": bool(row.get("exploratory_score", False)),
        "criterion_catalog_version": CRITERION_CATALOG_VERSION,
        "clinical_claim_allowed": False,
        "research_diagnosis_allowed": False,
        "locked_test_eligible": False,
    }


def finalize_criterion_catalog_evidence(
    candidates: pd.DataFrame,
    screening: pd.DataFrame,
    *,
    cross_fit_results: Optional[pd.DataFrame] = None,
    verification_bias_results: Optional[pd.DataFrame] = None,
    fiducial_limitation: Optional["FiducialLimitationPolicy"] = None,
) -> pd.DataFrame:
    """Finalize replication claims after project-wide multiplicity is known.

    Endpoint catalogs are built while validation is streamed endpoint by
    endpoint.  A candidate is therefore never promoted to
    ``VALIDATION_REPLICATED`` until this second pass has checked both Train and
    Validation against the complete project-wide global and hierarchical
    sensitivity families.

    ``fiducial_limitation`` carries the delineation gate's declared
    limitations.  A marker whose reliability collapses under a limited
    landmark's measured error is still screened, still reported and still
    available for discovery -- it just cannot be *confirmed*, because the
    measurement it rests on does not meet the external accuracy standard at
    any operating point.  Passing ``None`` means no limitation applies, which
    is the correct reading when the gate declared none.
    """
    output = candidates.copy()
    if output.empty:
        return output
    required = {
        "endpoint_id",
        "panel_id",
        "feature_id",
        "evaluation_split",
    }
    if screening.empty or not required <= set(screening.columns):
        output["project_wide_multiplicity_confirmed"] = False
        output["locked_test_eligible"] = False
        output["research_diagnosis_allowed"] = False
        return output
    # Only candidate keys can be queried below. Materializing all marker and
    # verification rows as Python dictionaries previously recreated millions
    # of wide objects just to finalize a few dozen candidate criteria.
    candidate_keys = {
        tuple(str(row.get(column, "")) for column in ("endpoint_id", "panel_id", "feature_id"))
        for row in output.to_dict(orient="records")
    }
    def candidate_rows(frame):
        if frame is None or frame.empty:
            return ()
        keys = pd.MultiIndex.from_arrays([
            frame[column].astype(str) if column in frame
            else pd.Series("", index=frame.index)
            for column in ("endpoint_id", "panel_id", "feature_id")
        ])
        return frame.loc[keys.isin(candidate_keys)].to_dict(orient="records")

    lookup = {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("panel_id", "")),
            str(row.get("feature_id", "")),
            str(row.get("evaluation_split", "")),
        ): row
        for row in candidate_rows(screening)
    }
    cross_fit_lookup = {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("panel_id", "")),
            str(row.get("feature_id", "")),
        ): row
        for row in (
            candidate_rows(cross_fit_results)
        )
    }
    verification_lookup = {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("panel_id", "")),
            str(row.get("feature_id", "")),
            str(row.get("evaluation_split", "")),
            str(row.get("operating_point", "")),
        ): row
        for row in (
            candidate_rows(verification_bias_results)
        )
    }
    records: list[dict] = []
    for candidate in output.to_dict(orient="records"):
        key = (
            str(candidate.get("endpoint_id", "")),
            str(candidate.get("panel_id", "")),
            str(candidate.get("feature_id", "")),
        )
        train = lookup.get((*key, "train"), {})
        validation = lookup.get((*key, "validation"), {})
        # Preserve the observed reference, not the endpoint registry's table
        # declaration. Candidate evidence and GUI use this same identity.
        for field in ("reference_kind", "independent_reference_passed",
                      "n_reference_eligible", "n_marker_observed", "n_marker_unavailable",
                      "availability_denominator"):
            if field in validation:
                candidate[field] = validation[field]
        cross_fit = cross_fit_lookup.get(key, {})
        verification = verification_lookup.get(
            (*key, "validation", "specificity_95"), {}
        )
        sensitivity_checks = {
            "train_global_fdr_significant": bool(
                train.get("global_fdr_significant", False)
            ),
            "validation_global_fdr_significant": bool(
                validation.get("global_fdr_significant", False)
            ),
            "train_hierarchical_fdr_significant": bool(
                train.get("hierarchical_fdr_significant", False)
            ),
            "validation_hierarchical_fdr_significant": bool(
                validation.get("hierarchical_fdr_significant", False)
            ),
        }
        multiplicity_confirmed = all(sensitivity_checks.values())
        cross_fit_checks = {
            "cross_fit_evaluated_in_both_folds": int(
                cross_fit.get("folds_available", 0) or 0
            )
            >= 2,
            "cross_fit_direction_consistent": bool(
                cross_fit.get("direction_consistent_across_folds", False)
            ),
            "cross_fit_assessability_confirmed": bool(
                cross_fit.get(
                    "confirmatory_assessability_gate_passed", False
                )
            ),
            "cross_fit_multiplicity_confirmed": bool(
                cross_fit.get("cross_fit_confirmed", False)
            ),
        }
        cross_fit_confirmed = all(cross_fit_checks.values())
        fiducial_confirmed = (
            fiducial_limitation is None
            or fiducial_limitation.marker_may_be_confirmed(
                str(candidate.get("feature_id", ""))
            )
        )
        verification_bias_confirmed = bool(
            verification.get("verification_bias_gate_passed", False)
        )
        core_confirmed = bool(
            candidate.get("core_replication_evidence_passed", False)
        )
        replicated = (
            core_confirmed
            and multiplicity_confirmed
            and cross_fit_confirmed
            and verification_bias_confirmed
            and fiducial_confirmed
        )
        candidate.update(
            {
                **sensitivity_checks,
                **cross_fit_checks,
                "fiducial_reliability_sufficient": bool(fiducial_confirmed),
                "rests_on_limited_landmark": bool(not fiducial_confirmed),
                "cross_fit_confirmed": cross_fit_confirmed,
                "verification_bias_gate_passed": (
                    verification_bias_confirmed
                ),
                "verification_rate": _optional_float(
                    verification.get("verification_rate")
                ),
                "verification_sensitivity_lower_bound": _optional_float(
                    verification.get("sensitivity_lower_bound")
                ),
                "verification_sensitivity_upper_bound": _optional_float(
                    verification.get("sensitivity_upper_bound")
                ),
                "verification_specificity_lower_bound": _optional_float(
                    verification.get("specificity_lower_bound")
                ),
                "verification_specificity_upper_bound": _optional_float(
                    verification.get("specificity_upper_bound")
                ),
                "cross_fit_p_value": _optional_float(
                    cross_fit.get("p_value")
                ),
                "cross_fit_q_value": _optional_float(
                    cross_fit.get("q_value")
                ),
                "cross_fit_global_q_value": _optional_float(
                    cross_fit.get("global_q_value")
                ),
                "cross_fit_hierarchical_q_value": _optional_float(
                    cross_fit.get("hierarchical_q_value")
                ),
                "cross_fit_pooled_auc": _optional_float(
                    cross_fit.get("pooled_effect")
                ),
                "cross_fit_pooled_auc_ci_low": _optional_float(
                    cross_fit.get("pooled_effect_ci_low")
                ),
                "cross_fit_pooled_auc_ci_high": _optional_float(
                    cross_fit.get("pooled_effect_ci_high")
                ),
                "train_global_q_value": _optional_float(
                    train.get("global_q_value")
                ),
                "validation_global_q_value": _optional_float(
                    validation.get("global_q_value")
                ),
                "train_hierarchical_q_value": _optional_float(
                    train.get("hierarchical_q_value")
                ),
                "validation_hierarchical_q_value": _optional_float(
                    validation.get("hierarchical_q_value")
                ),
                "global_q_value": _optional_float(
                    validation.get("global_q_value")
                ),
                "global_fdr_significant": sensitivity_checks[
                    "validation_global_fdr_significant"
                ],
                "hierarchical_q_value": _optional_float(
                    validation.get("hierarchical_q_value")
                ),
                "hierarchical_fdr_significant": sensitivity_checks[
                    "validation_hierarchical_fdr_significant"
                ],
                "project_wide_multiplicity_confirmed": multiplicity_confirmed,
                "stage": (
                    CriterionStage.VALIDATION_REPLICATED.value
                    if replicated
                    else candidate.get("stage")
                ),
                "stage_de": (
                    "in Validation bestaetigt"
                    if replicated
                    else candidate.get("stage_de")
                ),
                "evidence_grade": (
                    EvidenceGrade.VALIDATION_REPLICATED.value
                    if replicated
                    else candidate.get("evidence_grade")
                ),
                "evidence_grade_de": (
                    "Validation bestaetigt"
                    if replicated
                    else candidate.get("evidence_grade_de")
                ),
                "research_diagnosis_allowed": replicated,
                "locked_test_eligible": replicated,
            }
        )
        if core_confirmed and not multiplicity_confirmed:
            existing = str(
                candidate.get("replication_failure_reasons", "")
            ).strip()
            candidate["replication_failure_reasons"] = "|".join(
                value
                for value in (
                    existing,
                    "project_wide_multiplicity_sensitivity_not_confirmed",
                )
                if value
            )
        if core_confirmed and multiplicity_confirmed and not cross_fit_confirmed:
            existing = str(
                candidate.get("replication_failure_reasons", "")
            ).strip()
            candidate["replication_failure_reasons"] = "|".join(
                value
                for value in (
                    existing,
                    "bidirectional_cross_fit_confirmation_not_met",
                )
                if value
            )
        if (
            core_confirmed
            and multiplicity_confirmed
            and cross_fit_confirmed
            and not verification_bias_confirmed
        ):
            existing = str(
                candidate.get("replication_failure_reasons", "")
            ).strip()
            candidate["replication_failure_reasons"] = "|".join(
                value
                for value in (
                    existing,
                    "reference_standard_verification_bias_bounds_not_met",
                )
                if value
            )
        records.append(candidate)
    return pd.DataFrame(records)


def _optional_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
