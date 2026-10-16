from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .baseline_registry import BaselineRegistry
from .result_contract import legacy_display_grade, scientific_result_contract
from .known_ecg_associations import classify_novelty
from .feature_roles import (
    FEATURE_ROLE_POLICY_VERSION,
    ROLE_CANDIDATE,
    ROLE_CLASSICAL,
    canonical_feature_id,
    endpoint_feature_role,
    feature_profile,
)
from .pathology_groups import pathology_group_for_endpoint
from .storage import atomic_write_parquet
from .selection_contract import formal_selection_eligible
from .run_log import profiled_phase
from .user_guidance import (
    biomarker_display_name,
    biomarker_tooltip,
    clinical_display_term,
)


RESULT_EXPLORER_VERSION = "2.5.0"
# An operating point "sensitivity at 95 % specificity" is only a statement
# about that specificity when the train-fitted threshold actually reached it.
# Discrete markers (flags, octant fractions, rotation directions) often have
# no threshold anywhere near the target; the nearest one can sit at a
# specificity of 0.08, and its sensitivity of 0.96 then says nothing. Such a
# point must not rank a marker.
OPERATING_POINT_TARGET_SPECIFICITY = 0.95
OPERATING_POINT_SPECIFICITY_TOLERANCE = 1e-12
# Only rows with a formal test (a p-value the study design stands behind)
# may become an endpoint's "best" marker or a criterion candidate. The
# descriptive rows -- two positives against one negative, AUC 1.0 -- stay in
# the tables as what they are, but never at the top.
SELECTION_INFERENCE_STATUS = "formal_inference"
NO_FORMAL_MARKER_STATUS = "no formally assessable marker in this run"
ECG_DERIVED_REFERENCE_ENDPOINTS = frozenset(
    {
        "qtc_gt_480",
        "qtc_gt_500",
    }
)
_KNOWN_DIRECT_ECG_TARGET_TOKENS = (
    "bundle branch block",
    "heart block",
    "atrioventricular block",
    "pacemaker",
    "cardiac defibrillator",
    "cardiac device",
    "atrial fibrillation",
    "atrial flutter",
    "tachycardia",
    "bradycardia",
    "electrocardiogram",
)
_EVIDENCE_GRADE_NAMES_DE = {
    "A": "Klinisch validiert – alle Evidenz-Gates bestanden",
    "B": "Kriterienkandidat – interne Evidenz-Gates bestanden",
    "C": "Replizierte Assoziation – keine klinische Freigabe",
    "T": "Technischer Vergleich aus demselben ECG",
    "D": "Hypothese oder instabiles Ergebnis",
}


@dataclass(frozen=True)
class ResultExplorerTables:
    marker_pathology: pd.DataFrame
    pathology_marker: pd.DataFrame
    endpoint_summary: pd.DataFrame
    discovery_ranked: pd.DataFrame


@profiled_phase("postprocess.result_explorer")
def build_result_explorer_tables(
    screening: pd.DataFrame,
    *,
    combinations: Optional[pd.DataFrame] = None,
    comparisons: Optional[pd.DataFrame] = None,
    stability: Optional[pd.DataFrame] = None,
    broad_discovery: Optional[pd.DataFrame] = None,
    candidate_evidence: Optional[pd.DataFrame] = None,
) -> ResultExplorerTables:
    if candidate_evidence is not None and not candidate_evidence.empty:
        evidence = candidate_evidence.rename(columns={"marker_id": "feature_id"})
        keys = ["endpoint_id", "feature_id"]
        columns = [name for name in evidence if name.endswith("_passed") or name in
                   {"reference_kind", "analysis_completeness"}]
        # Ambiguous evidence rows cannot upgrade a screening association.
        evidence = evidence.loc[~evidence.duplicated(keys, keep=False), keys+columns]
        screening = screening.drop(columns=[name for name in columns if name in screening]).merge(
            evidence, on=keys, how="left", validate="many_to_one")
    marker_pathology = _marker_pathology_table(
        screening,
        comparisons=comparisons,
        stability=stability,
    )
    score_rows = _score_rows(combinations)
    pathology_marker = _pathology_marker_table(
        marker_pathology,
        score_rows,
    )
    endpoint_summary = _endpoint_summary(pathology_marker)
    discovery_ranked = rank_broad_discovery(
        broad_discovery if broad_discovery is not None else pd.DataFrame()
    )
    return ResultExplorerTables(
        marker_pathology=marker_pathology,
        pathology_marker=pathology_marker,
        endpoint_summary=endpoint_summary,
        discovery_ranked=discovery_ranked,
    )


def write_result_explorer_tables(
    run_root: Path | str,
    tables: ResultExplorerTables,
) -> dict[str, Path]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "marker_pathology": root / "marker_pathology_explorer.parquet",
        "pathology_marker": root / "pathology_marker_explorer.parquet",
        "endpoint_summary": root / "endpoint_summary_explorer.parquet",
        "discovery_ranked": root / "broad_discovery_ranked.parquet",
    }
    atomic_write_parquet(tables.marker_pathology, paths["marker_pathology"])
    atomic_write_parquet(tables.pathology_marker, paths["pathology_marker"])
    atomic_write_parquet(tables.endpoint_summary, paths["endpoint_summary"])
    atomic_write_parquet(tables.discovery_ranked, paths["discovery_ranked"])
    return paths


def _marker_pathology_table(
    screening: pd.DataFrame,
    *,
    comparisons: Optional[pd.DataFrame],
    stability: Optional[pd.DataFrame],
) -> pd.DataFrame:
    columns = _marker_columns()
    if screening is None or screening.empty:
        return pd.DataFrame(columns=columns)
    frame = _attach_training_operating_evidence(screening)
    if "evaluation_split" in frame:
        frame = frame[
            frame["evaluation_split"].astype(str).eq("validation")
        ].copy()
    if "analysis_type" in frame:
        frame = frame[
            frame["analysis_type"].astype(str).eq("single_marker")
        ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    profiles = [feature_profile(item) for item in frame["feature_id"]]
    frame["canonical_feature_id"] = [
        profile.canonical_feature_id for profile in profiles
    ]
    frame["feature_role"] = [profile.role for profile in profiles]
    frame["feature_modality"] = [profile.modality for profile in profiles]
    frame["conceptual_family"] = [
        profile.conceptual_family for profile in profiles
    ]
    frame["scientific_tier"] = [
        profile.scientific_tier for profile in profiles
    ]
    frame["post_outcome_hypothesis"] = [
        profile.post_outcome_hypothesis for profile in profiles
    ]
    frame = frame[
        frame["feature_role"].isin((ROLE_CANDIDATE, ROLE_CLASSICAL))
    ].copy()
    frame = frame[
        frame["feature_id"].astype(str)
        == frame["canonical_feature_id"].astype(str)
    ].copy()

    group_data = [
        pathology_group_for_endpoint(endpoint)
        for endpoint in frame["endpoint_id"]
    ]
    frame["pathology_group_id"] = [item.group_id for item in group_data]
    frame["pathology_group"] = [
        item.display_name_de for item in group_data
    ]
    frame["group_not_pooled"] = [
        item.navigation_only for item in group_data
    ]
    frame = _annotate_marker_display(frame)
    frame["candidate_type"] = np.where(
        frame["feature_role"].eq(ROLE_CLASSICAL),
        "classical_rule",
        np.where(
            frame["feature_modality"].eq("vcg"),
            "vcg_single",
            "ecg_single",
        ),
    )
    frame["endpoint_feature_role"] = [
        endpoint_feature_role(feature_id, endpoint_id)
        for feature_id, endpoint_id in zip(
            frame["feature_id"],
            frame["endpoint_id"],
        )
    ]
    frame["reference_independence"] = np.where(
        frame["endpoint_id"].astype(str).isin(
            ECG_DERIVED_REFERENCE_ENDPOINTS
        ),
        "same_ecg_technical_comparison",
        "reference_quality_requires_review",
    )
    frame = _annotate_baseline_applicability(frame)
    frame = frame[
        ~frame["candidate_type"].eq("classical_rule")
        | frame["baseline_applicable"].fillna(False)
    ].copy()
    frame["sensitivity_at_90_specificity"] = _numeric(
        frame,
        "specificity_90_sensitivity",
    )
    frame["sensitivity_at_95_specificity"] = _numeric(
        frame,
        "specificity_95_sensitivity",
    )
    frame["specificity_at_90_sensitivity"] = _numeric(
        frame,
        "sensitivity_90_specificity",
    )
    frame["operating_sensitivity"] = frame[
        "sensitivity_at_95_specificity"
    ]
    frame["operating_specificity"] = _numeric(
        frame,
        "specificity_95_specificity",
    )
    frame = _annotate_operating_point_attainability(frame)
    frame["selection_eligible"] = _selection_eligible(frame)
    # Old result files lack the audit metadata added in validation v3.
    frame["historical_pre_audit_result"] = (
        "canonical_feature_id" not in screening.columns
        or "feature_role" not in screening.columns
    )

    if comparisons is not None and not comparisons.empty:
        frame = _merge_comparisons(frame, comparisons)
    else:
        for column in (
            "baseline_feature_id",
            "sensitivity_delta_vs_fixed_rule",
            "specificity_delta_vs_fixed_rule",
            "delta_auc",
            "delta_auc_ci_low",
            "delta_auc_ci_high",
        ):
            frame[column] = np.nan if "delta" in column else ""
    if stability is not None and not stability.empty:
        frame = _merge_stability(frame, stability)
    else:
        frame["direction_agreement"] = np.nan
        frame["train_validation_auc_gap"] = np.nan

    frame["evidence_grade"] = frame.apply(_evidence_grade, axis=1)
    contracts = [scientific_result_contract(row).to_dict() for row in frame.to_dict("records")]
    for key in ("evidence_stage", "claim_ceiling", "analysis_scope", "failed_gates", "clinical_claim_allowed", "contract_version",
                "independent_modality_observed_n", "clock_alignment_confirmed_n", "timing_evidence_note"):
        frame[key] = [item[key] for item in contracts]
    frame["evidence_grade_de"] = frame["evidence_grade"].map(
        _EVIDENCE_GRADE_NAMES_DE
    ).fillna("Noch nicht belastbar")
    frame["evidence_rank"] = frame.apply(_evidence_rank, axis=1)
    frame["clinical_interpretation"] = frame.apply(
        _interpretation_label,
        axis=1,
    )
    frame["result_scope"] = np.where(
        frame["reference_independence"].eq(
            "same_ecg_technical_comparison"
        ),
        "same-ECG technical comparison; not an independent clinical endpoint",
        np.where(
            frame["post_outcome_hypothesis"].fillna(False),
            "post-development discovery; independent replication required",
            np.where(
                frame["historical_pre_audit_result"].fillna(False),
                "historical development result; recompute with audited feature code",
                "MIMIC validation holdout; not locked-test validated",
            ),
        ),
    )
    frame = frame.sort_values(
        [
            "selection_eligible",
            "evidence_rank",
            "roc_auc",
            "sensitivity_at_95_specificity",
            "feature_id",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    frame = frame.drop_duplicates(
        ["endpoint_id", "canonical_feature_id"],
        keep="first",
    )
    return _ensure_columns(frame, columns)


def _merge_comparisons(
    frame: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    selected = comparisons.copy()
    if "evaluation_split" in selected:
        selected = selected[
            selected["evaluation_split"].astype(str).eq("validation")
        ]
    if selected.empty:
        return frame
    selected["feature_id"] = selected.get(
        "vcg_feature_id",
        selected.get("feature_id", ""),
    )
    for column in (
        "sensitivity_delta_vs_fixed_rule",
        "specificity_delta_vs_fixed_rule",
        "delta_auc",
    ):
        selected[f"__rank_{column}"] = _numeric(selected, column).abs()
    selected["__comparison_rank"] = selected[
        [
            "__rank_sensitivity_delta_vs_fixed_rule",
            "__rank_specificity_delta_vs_fixed_rule",
            "__rank_delta_auc",
        ]
    ].max(axis=1, skipna=True)
    selected = selected.sort_values(
        "__comparison_rank",
        ascending=False,
        na_position="last",
    ).drop_duplicates(["endpoint_id", "feature_id"])
    wanted = [
        "endpoint_id",
        "feature_id",
        "baseline_feature_id",
        "baseline_registry_id",
        "baseline_source_version",
        "sensitivity_delta_vs_fixed_rule",
        "sensitivity_delta_ci_low",
        "sensitivity_delta_ci_high",
        "specificity_delta_vs_fixed_rule",
        "specificity_delta_ci_low",
        "specificity_delta_ci_high",
        "delta_auc",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "auc_delta_reason",
    ]
    wanted = [column for column in wanted if column in selected]
    return frame.merge(
        selected[wanted],
        on=["endpoint_id", "feature_id"],
        how="left",
        validate="one_to_one",
    )


def _merge_stability(
    frame: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    selected = stability.copy()
    if "analysis_type" in selected:
        selected = selected[
            selected["analysis_type"].astype(str).eq("single_marker")
        ]
    selected = selected.drop_duplicates(["endpoint_id", "feature_id"])
    wanted = [
        "endpoint_id",
        "feature_id",
        "direction_agreement",
        "train_validation_auc_gap",
        "transformation_sensitivity",
        "time_window_sensitivity",
        "validation_stable",
    ]
    wanted = [column for column in wanted if column in selected]
    return frame.merge(
        selected[wanted],
        on=["endpoint_id", "feature_id"],
        how="left",
        validate="one_to_one",
    )


def _score_rows(combinations: Optional[pd.DataFrame]) -> pd.DataFrame:
    if combinations is None or combinations.empty:
        return pd.DataFrame(columns=_marker_columns())
    frame = _attach_training_operating_evidence(combinations)
    if "evaluation_split" in frame:
        frame = frame[
            frame["evaluation_split"].astype(str).eq("validation")
        ].copy()
    if frame.empty:
        return pd.DataFrame(columns=_marker_columns())
    groups = [
        pathology_group_for_endpoint(endpoint)
        for endpoint in frame["endpoint_id"]
    ]
    frame["pathology_group_id"] = [item.group_id for item in groups]
    frame["pathology_group"] = [item.display_name_de for item in groups]
    frame["group_not_pooled"] = [item.navigation_only for item in groups]
    frame = _annotate_marker_display(frame)
    frame["canonical_feature_id"] = frame["feature_id"].astype(str)
    frame["feature_role"] = "exploratory_score"
    declared_modality = _series(frame, "feature_modality", "").fillna(
        ""
    ).astype(str)
    inferred_modality = frame.apply(_infer_score_modality, axis=1)
    frame["feature_modality"] = np.where(
        declared_modality.str.strip().ne(""),
        declared_modality,
        inferred_modality,
    )
    frame["conceptual_family"] = "small_interpretable_score"
    analysis_type = _series(frame, "analysis_type", "")
    frame["scientific_tier"] = np.where(
        analysis_type.astype(str).eq(
            "exploratory_mimic_combination"
        ),
        "development_exploratory",
        "predefined_transfer_or_score",
    )
    frame["post_outcome_hypothesis"] = analysis_type.astype(str).eq(
        "exploratory_mimic_combination"
    )
    frame["candidate_type"] = "small_score"
    frame["endpoint_feature_role"] = "exploratory_score"
    frame["reference_independence"] = np.where(
        frame["endpoint_id"].astype(str).isin(
            ECG_DERIVED_REFERENCE_ENDPOINTS
        ),
        "same_ecg_technical_comparison",
        "reference_quality_requires_review",
    )
    frame["baseline_applicable"] = False
    frame["baseline_registry_id"] = ""
    frame["baseline_display_name"] = ""
    frame["baseline_kind"] = ""
    frame["sensitivity_at_90_specificity"] = _numeric(
        frame,
        "specificity_90_sensitivity",
    )
    frame["sensitivity_at_95_specificity"] = _numeric(
        frame,
        "specificity_95_sensitivity",
    )
    frame["specificity_at_90_sensitivity"] = _numeric(
        frame,
        "sensitivity_90_specificity",
    )
    frame["operating_sensitivity"] = frame[
        "sensitivity_at_95_specificity"
    ]
    frame["operating_specificity"] = _numeric(
        frame,
        "specificity_95_specificity",
    )
    frame = _annotate_operating_point_attainability(frame)
    frame["selection_eligible"] = _selection_eligible(frame)
    frame["historical_pre_audit_result"] = (
        _series(frame, "feature_role_policy_version", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )
    frame["evidence_grade"] = frame.apply(_evidence_grade, axis=1)
    frame["evidence_grade_de"] = frame["evidence_grade"].map(
        _EVIDENCE_GRADE_NAMES_DE
    ).fillna("Noch nicht belastbar")
    frame["evidence_rank"] = frame.apply(_evidence_rank, axis=1)
    frame["clinical_interpretation"] = frame.apply(
        _interpretation_label,
        axis=1,
    )
    frame["result_scope"] = np.where(
        frame["reference_independence"].eq(
            "same_ecg_technical_comparison"
        ),
        "same-ECG technical score comparison; not clinical validation",
        "exploratory development score; recompute with audited feature code",
    )
    return _ensure_columns(frame, _marker_columns())


def _infer_score_modality(row: pd.Series) -> str:
    declared = str(row.get("component_modalities", "") or "")
    modalities = {
        item.strip()
        for item in declared.split("|")
        if item.strip()
    }
    if not modalities:
        components = [
            item.strip()
            for item in str(row.get("components", "") or "").split("|")
            if item.strip()
        ]
        modalities = {
            feature_profile(component).modality for component in components
        }
    if modalities == {"vcg"}:
        return "vcg_score"
    if "vcg" in modalities:
        return "mixed_ecg_vcg_score"
    if modalities:
        return "ecg_score"
    return "unknown_score"


def _pathology_marker_table(
    marker_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
) -> pd.DataFrame:
    frames = [frame for frame in (marker_rows, score_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_marker_columns() + ["endpoint_rank"])
    output = pd.concat(frames, ignore_index=True, sort=False)
    if "baseline_applicable" in output:
        output = output[
            ~output["candidate_type"].eq("classical_rule")
            | output["baseline_applicable"].fillna(False)
        ].copy()
    output["selection_eligible"] = (
        _series(output, "selection_eligible", False).fillna(False).astype(bool)
    )
    output = output.sort_values(
        [
            "endpoint_id",
            "selection_eligible",
            "evidence_rank",
            "roc_auc",
            "sensitivity_at_95_specificity",
        ],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    output["endpoint_rank"] = (
        output.groupby("endpoint_id").cumcount() + 1
    )
    return output


def _endpoint_summary(pathology_marker: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "pathology_group",
        "endpoint_id",
        "endpoint_name_de",
        "endpoint_definition_de",
        "best_vcg_marker",
        "best_vcg_marker_name_de",
        "best_vcg_auc",
        "best_vcg_sensitivity_at_95_specificity",
        "best_small_score",
        "best_small_score_auc",
        "best_classical_rule",
        "best_classical_auc_or_point",
        "best_evidence_grade",
        "status",
    )
    if pathology_marker.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for endpoint, group in pathology_marker.groupby("endpoint_id"):
        eligible = group[
            _series(group, "selection_eligible", False).fillna(False).astype(bool)
        ]
        vcg = eligible[eligible["candidate_type"].eq("vcg_single")]
        scores = eligible[eligible["candidate_type"].eq("small_score")]
        classical = eligible[eligible["candidate_type"].eq("classical_rule")]
        best_vcg = vcg.head(1)
        best_score = scores.head(1)
        best_classical = classical.head(1)
        top = eligible.head(1)
        status = (
            _first(top, "result_scope")
            if not top.empty
            else NO_FORMAL_MARKER_STATUS
        )
        rows.append(
            {
                "pathology_group": _first(group, "pathology_group"),
                "endpoint_id": endpoint,
                "endpoint_name_de": _first(group, "endpoint_name_de"),
                "endpoint_definition_de": _first(
                    group, "endpoint_definition_de"
                ),
                "best_vcg_marker": _first(best_vcg, "feature_id"),
                "best_vcg_marker_name_de": _first(
                    best_vcg, "feature_name_de"
                ),
                "best_vcg_auc": _first(best_vcg, "roc_auc", np.nan),
                "best_vcg_sensitivity_at_95_specificity": _first(
                    best_vcg,
                    "sensitivity_at_95_specificity",
                    np.nan,
                ),
                "best_small_score": _first(best_score, "feature_id"),
                "best_small_score_auc": _first(
                    best_score,
                    "roc_auc",
                    np.nan,
                ),
                "best_classical_rule": _first(
                    best_classical,
                    "feature_id",
                ),
                "best_classical_auc_or_point": _first(
                    best_classical,
                    "roc_auc",
                    np.nan,
                ),
                "best_evidence_grade": _first(
                    top,
                    "evidence_grade",
                ),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["pathology_group", "endpoint_id"]
    )


def rank_broad_discovery(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "evidence_rank",
        "evidence_tier",
        "discovery_class",
        "discovery_domain",
        "target_id",
        "target_name",
        "target_kind",
        "feature_id",
        "feature_name_de",
        "feature_tooltip_de",
        "canonical_feature_id",
        "conceptual_family",
        "feature_modality",
        "panel_id",
        "n_positive",
        "n_negative",
        "n_total",
        "missingness",
        "train_effect",
        "train_q_value",
        "validation_effect",
        "validation_q_value",
        "validation_auc",
        "validation_average_precision",
        "validation_ci_low",
        "validation_ci_high",
        "train_validation_gap",
        "effect_retention_ratio",
        "validation_stable",
        "adjustment_status",
        "adjusted_q_value",
        "adjusted_or_per_sd",
        "adjustment_robust",
        "likely_confounded",
        "historical_implementation",
        "confounding_control",
        "requires_endpoint_specific_adjustment",
        "direction_consistent",
        "replicated",
        "inference_status",
        "locked_test_scanned",
        "followup_priority",
        "status_de",
        "novelty_class",
        "novel_vs_literature",
        "interpretation",
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    output = frame.copy()
    profiles = [feature_profile(item) for item in output["feature_id"]]
    output["canonical_feature_id"] = [
        item.canonical_feature_id for item in profiles
    ]
    output["conceptual_family"] = [
        item.conceptual_family for item in profiles
    ]
    output["feature_modality"] = [item.modality for item in profiles]
    output["feature_name_de"] = [
        biomarker_display_name(item) for item in output["feature_id"]
    ]
    output["feature_tooltip_de"] = [
        biomarker_tooltip(item) for item in output["feature_id"]
    ]
    output["feature_role"] = [item.role for item in profiles]
    output = output[
        (
            output["feature_id"].astype(str)
            == output["canonical_feature_id"].astype(str)
        )
        & output["feature_role"].eq(ROLE_CANDIDATE)
    ].copy()
    for column in ("n_positive", "n_negative"):
        if column not in output:
            output[column] = np.nan
    replicated = output.get("replicated", False)
    if not isinstance(replicated, pd.Series):
        replicated = pd.Series(replicated, index=output.index)
    formal = _series(output, "inference_status", "").astype(str).eq(
        "formal_inference"
    )
    direction = output.get("direction_consistent", False)
    if not isinstance(direction, pd.Series):
        direction = pd.Series(direction, index=output.index)
    adjustment_robust = _series(
        output, "adjustment_robust", False
    ).fillna(False).astype(bool)
    likely_confounded = _series(
        output, "likely_confounded", False
    ).fillna(False).astype(bool)
    output["evidence_tier"] = np.select(
        [
            adjustment_robust & replicated.fillna(False) & formal,
            likely_confounded,
            replicated.fillna(False) & formal,
            replicated.fillna(False),
            direction.fillna(False),
        ],
        [
            "validation_replicated_adjusted",
            "validation_replicated_likely_confounded",
            "validation_replicated_formal",
            "validation_replicated_underpowered",
            "direction_consistent_not_replicated",
        ],
        default="train_only_or_unstable",
    )
    validation_effect = _numeric(output, "validation_effect").abs()
    validation_q = _numeric(output, "validation_q_value")
    sample_floor = np.minimum(
        _numeric(output, "n_positive"),
        _numeric(output, "n_negative"),
    )
    sample_floor = sample_floor.where(
        sample_floor > 0,
        _numeric(output, "n_total"),
    )
    stable = _series(output, "validation_stable", False).fillna(False)
    historical = ~_series(
        output,
        "feature_role_policy_version",
        "",
    ).astype(str).eq(FEATURE_ROLE_POLICY_VERSION)
    output["historical_implementation"] = historical
    output["confounding_control"] = _series(
        output,
        "confounding_control",
        "none_broad_screen",
    )
    output["requires_endpoint_specific_adjustment"] = _series(
        output,
        "requires_endpoint_specific_adjustment",
        True,
    ).fillna(True)
    output["evidence_rank"] = (
        replicated.fillna(False).astype(float) * 100.0
        + adjustment_robust.astype(float) * 45.0
        + formal.astype(float) * 20.0
        + stable.astype(float) * 15.0
        + direction.fillna(False).astype(float) * 10.0
        + validation_effect.fillna(0.0).clip(0.0, 1.0) * 20.0
        + (
            -np.log10(validation_q.clip(lower=1e-300))
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 20.0)
        + np.log10(sample_floor.clip(lower=1.0)).fillna(0.0)
    )
    output.loc[likely_confounded, "evidence_rank"] -= 70.0
    output.loc[historical, "evidence_rank"] -= 50.0
    output.loc[historical, "evidence_tier"] = (
        "historical_recompute_required"
    )
    known_direct = output["target_name"].astype(str).str.casefold().map(
        lambda value: any(
            token in value for token in _KNOWN_DIRECT_ECG_TARGET_TOKENS
        )
    )
    output["novelty_class"] = np.where(
        known_direct,
        "known_direct_ecg_or_device_association",
        "nontrivial_followup_hypothesis",
    )
    output.loc[known_direct, "evidence_rank"] -= 30.0
    absolute_effect = _numeric(output, "validation_effect").abs()
    output["followup_priority"] = np.select(
        [
            historical,
            known_direct,
            likely_confounded,
            adjustment_robust
            & formal
            & stable
            & absolute_effect.ge(0.10),
            replicated.fillna(False)
            & formal
            & stable
            & absolute_effect.ge(0.10),
            replicated.fillna(False) & formal & absolute_effect.ge(0.05),
            replicated.fillna(False),
        ],
        [
            "historical_recompute_required",
            "known_benchmark",
            "review_confounding",
            "high",
            "medium",
            "exploratory",
            "exploratory",
        ],
        default="low",
    )
    output["interpretation"] = np.where(
        output["discovery_class"].astype(str).eq("pathology_hypothesis"),
        "replizierte unadjustierte Assoziation; keine Diagnose und noch kein bestätigter Endpunkt",
        "unadjustierte Kontext-/Outcome-Assoziation; nicht als Pathologie interpretieren",
    )
    output.loc[adjustment_robust, "interpretation"] = (
        "In Validation repliziert und auf Alter, Geschlecht sowie "
        "Signalqualität adjustiert; weiterhin nur eine Hypothese für die "
        "Detailanalyse."
    )
    output.loc[likely_confounded, "interpretation"] = (
        "Unadjustiert repliziert, nach vorgegebener Adjustierung jedoch "
        "nicht robust; Confounding gezielt prüfen."
    )
    output["status_de"] = np.select(
        [
            adjustment_robust,
            likely_confounded,
            replicated.fillna(False),
            direction.fillna(False),
        ],
        [
            "repliziert",
            "wahrscheinlich confounded",
            "vielversprechend",
            "neu",
        ],
        default="verworfen",
    )
    output.loc[historical, "interpretation"] = (
        "Historische Hypothese mit veraltetem Biomarker-Code; vor "
        "inhaltlicher Verfolgung im aktuellen Train/Validation-Lauf "
        "reproduzieren."
    )
    output["locked_test_scanned"] = False
    output = output.sort_values(
        [
            "evidence_rank",
            "validation_q_value",
            "target_name",
            "feature_id",
        ],
        ascending=[False, True, True, True],
        na_position="last",
    )
    output = output.drop_duplicates(
        ["discovery_class", "target_id", "conceptual_family"],
        keep="first",
    )
    return _ensure_columns(output, columns)


def build_new_association_explorer(
    pathology_status: pd.DataFrame,
    broad_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Combine pathology hypotheses and separate context/future domains.

    This is a navigation table only. It never converts laboratory,
    medication, procedure or outcome associations into diagnoses.
    """
    columns = (
        "evidence_rank",
        "status_id",
        "status_de",
        "discovery_class",
        "discovery_domain",
        "scope_display_de",
        "cardiac_scope_id",
        "cardiac_scope_de",
        "target_id",
        "target_name",
        "target_kind",
        "feature_id",
        "feature_name_de",
        "conceptual_family",
        "feature_modality",
        "panel_id",
        "n_positive",
        "n_negative",
        "train_q_value",
        "validation_q_value",
        "statistical_strength",
        "validation_auc",
        "association_effect",
        "association_strength",
        "association_measure_de",
        "direction_consistent",
        "unadjusted_replicated",
        "inference_status",
        "adjustment_status",
        "adjusted_q_value",
        "adjusted_or_per_sd",
        "adjustment_robust",
        "likely_confounded",
        "followup_priority",
        "followup_priority_de",
        "interpretation_de",
        "automatic_promotion",
        "locked_test_scanned",
        "novel_vs_literature",
    )
    frames: list[pd.DataFrame] = []
    if pathology_status is not None and not pathology_status.empty:
        pathologies = pd.DataFrame(
            {
                "evidence_rank": _numeric(
                    pathology_status, "evidence_rank"
                )
                * 100.0
                + _numeric(
                    pathology_status, "validation_aligned_auc"
                ).fillna(0.5)
                * 20.0,
                "status_id": _series(
                    pathology_status, "status_id", "new"
                ),
                "status_de": _series(
                    pathology_status, "status_de", "neu"
                ),
                "discovery_class": "pathology_hypothesis",
                "discovery_domain": _series(
                    pathology_status, "body_system", ""
                ),
                "target_id": pathology_status["phenotype_id"].astype(str),
                "target_name": _series(
                    pathology_status, "phenotype_name", ""
                ),
                "target_kind": "retrospective_proxy",
                "feature_id": pathology_status["feature_id"].astype(str),
                "feature_name_de": [
                    biomarker_display_name(value)
                    for value in pathology_status["feature_id"]
                ],
                "conceptual_family": [
                    feature_profile(value).conceptual_family
                    for value in pathology_status["feature_id"]
                ],
                "feature_modality": [
                    feature_profile(value).modality
                    for value in pathology_status["feature_id"]
                ],
                "panel_id": pathology_status["panel_id"],
                "n_positive": _numeric(
                    pathology_status, "n_positive"
                ),
                "n_negative": _numeric(
                    pathology_status, "n_negative"
                ),
                "train_q_value": _numeric(
                    pathology_status, "train_q_value"
                ),
                "validation_q_value": _numeric(
                    pathology_status, "validation_q_value"
                ),
                "validation_auc": _numeric(
                    pathology_status, "validation_aligned_auc"
                ),
                "association_effect": _numeric(
                    pathology_status, "validation_effect"
                ),
                "association_strength": _numeric(
                    pathology_status, "association_strength"
                ),
                "direction_consistent": _series(
                    pathology_status, "direction_consistent", False
                ),
                "unadjusted_replicated": _series(
                    pathology_status, "unadjusted_replicated", False
                ),
                "inference_status": _series(
                    pathology_status,
                    "inference_status",
                    "not_evaluated",
                ),
                "adjustment_status": _series(
                    pathology_status,
                    "adjustment_status",
                    "not_requested_not_replicated",
                ),
                "adjusted_q_value": _numeric(
                    pathology_status, "adjusted_q_value"
                ),
                "adjusted_or_per_sd": _numeric(
                    pathology_status, "adjusted_or_per_sd"
                ),
                "adjustment_robust": _series(
                    pathology_status, "adjustment_robust", False
                ),
                "likely_confounded": _series(
                    pathology_status, "status_id", ""
                ).astype(str).eq("likely_confounded"),
                "followup_priority": _series(
                    pathology_status, "status_id", "new"
                ).astype(str).map(
                    {
                        "replicated": "high",
                        "promising": "medium",
                        "likely_confounded": "review_confounding",
                        "new": "exploratory",
                        "rejected": "low",
                    }
                ).fillna("low"),
                "interpretation_de": _series(
                    pathology_status, "status_reason_de", ""
                ),
                "automatic_promotion": False,
                "locked_test_scanned": False,
            }
        )
        frames.append(pathologies)
    if broad_ranked is not None and not broad_ranked.empty:
        separate = broad_ranked.loc[
            ~broad_ranked["discovery_class"].astype(str).eq(
                "pathology_hypothesis"
            )
        ].copy()
        if not separate.empty:
            separate["status_id"] = np.where(
                separate["replicated"].fillna(False),
                "replicated_association",
                "promising_association",
            )
            separate["unadjusted_replicated"] = _series(
                separate, "replicated", False
            )
            separate["association_effect"] = _numeric(
                separate, "validation_effect"
            )
            separate["association_strength"] = separate[
                "association_effect"
            ].abs()
            separate["interpretation_de"] = _series(
                separate, "interpretation", ""
            )
            separate["automatic_promotion"] = False
            frames.append(_ensure_columns(separate, columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    output = pd.concat(
        [_ensure_columns(frame, columns) for frame in frames],
        ignore_index=True,
    )
    association_effect = _numeric(output, "association_effect")
    association_strength = _numeric(output, "association_strength")
    output["association_strength"] = association_strength.where(
        association_strength.notna(),
        association_effect.abs(),
    )
    q_values = _numeric(output, "validation_q_value")
    output["statistical_strength"] = (
        -np.log10(q_values.clip(lower=1e-300))
    ).replace([np.inf, -np.inf], np.nan)
    output["association_measure_de"] = (
        "Absoluter rank-biserialer Validation-Effekt |2 x AUC - 1|"
    )
    output["scope_display_de"] = output["discovery_class"].astype(str).map(
        {
            "pathology_hypothesis": "Aktuelle Pathologie-Hypothese",
            "future_outcome_hypothesis": "Zukünftige Erkrankung",
            "non_pathology_association": "Klinischer Kontext, keine Pathologie",
        }
    ).fillna("Explorativer Zusammenhang")
    cardiac_scope = [
        _cardiac_scope(
            discovery_class=row.discovery_class,
            domain=row.discovery_domain,
            target_name=row.target_name,
        )
        for row in output.itertuples(index=False)
    ]
    output["cardiac_scope_id"] = [item[0] for item in cardiac_scope]
    output["cardiac_scope_de"] = [item[1] for item in cardiac_scope]
    priority_labels = {
        "high": "hoch",
        "medium": "mittel",
        "exploratory": "explorativ",
        "review_confounding": "Confounding prüfen",
        "known_benchmark": "bekannter Vergleich",
        "historical_recompute_required": "neu berechnen",
        "low": "niedrig",
    }
    output["followup_priority_de"] = _series(
        output, "followup_priority", "low"
    ).astype(str).map(priority_labels).fillna("niedrig")
    output["novel_vs_literature"] = [
        classify_novelty(
            feature_id=str(feature_id),
            target_id=str(target_id),
            target_name=str(target_name),
            discovery_domain=str(discovery_domain),
        )
        for feature_id, target_id, target_name, discovery_domain in zip(
            output["feature_id"],
            output["target_id"],
            output["target_name"],
            output["discovery_domain"],
        )
    ]
    return output.sort_values(
        ["evidence_rank", "validation_q_value", "target_name"],
        ascending=(False, True, True),
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _cardiac_scope(
    *,
    discovery_class: object,
    domain: object,
    target_name: object,
) -> tuple[str, str]:
    if str(discovery_class) == "non_pathology_association":
        return "not_pathology", "Kein Pathologieziel"
    searchable = f"{domain} {target_name}".casefold()
    cardiovascular_tokens = (
        "circulatory",
        "cardiovascular",
        "cardiac",
        "heart",
        "coronary",
        "myocard",
        "arrhythm",
        "valv",
        "aortic",
        "pericard",
        "endocard",
    )
    if any(token in searchable for token in cardiovascular_tokens):
        return "cardiovascular", "Kardiovaskulär"
    if searchable.strip():
        return "non_cardiovascular", "Nicht kardiovaskulär"
    return "unknown", "Nicht eindeutig zugeordnet"


def _evidence_grade(row: pd.Series) -> str:
    return legacy_display_grade(row)


def _evidence_rank(row: pd.Series) -> float:
    grade = {
        "A": 4.0,
        "B": 3.0,
        "C": 2.0,
        "T": 1.5,
        "D": 1.0,
    }.get(
        str(row.get("evidence_grade", "D")),
        0.0,
    )
    auc = _float(row.get("roc_auc")) or 0.5
    sensitivity = _float(row.get("sensitivity_at_95_specificity")) or 0.0
    q_value = _float(row.get("q_value"))
    q_strength = (
        min(20.0, -np.log10(max(q_value, 1e-300)))
        if q_value is not None and q_value > 0
        else 0.0
    )
    missingness = _float(row.get("missingness")) or 0.0
    historical_penalty = (
        15.0 if bool(row.get("historical_pre_audit_result", False)) else 0.0
    )
    post_outcome_penalty = (
        10.0 if bool(row.get("post_outcome_hypothesis", False)) else 0.0
    )
    technical_reference_penalty = (
        30.0
        if str(row.get("reference_independence", ""))
        == "same_ecg_technical_comparison"
        else 0.0
    )
    return float(
        grade * 100.0
        + max(0.0, auc - 0.5) * 100.0
        + sensitivity * 20.0
        + q_strength
        - missingness * 20.0
        - historical_penalty
        - post_outcome_penalty
        - technical_reference_penalty
    )


def _interpretation_label(row: pd.Series) -> str:
    if (
        str(row.get("reference_independence", ""))
        == "same_ecg_technical_comparison"
    ):
        return (
            "Technischer ECG-interner Vergleich; keine unabhängige "
            "klinische Validierung"
        )
    if bool(row.get("historical_pre_audit_result", False)):
        return "Historisches Entwicklungsergebnis; mit v2-Formeln neu rechnen"
    if bool(row.get("post_outcome_hypothesis", False)):
        return "Post-hoc Discovery-Hypothese; externe Replikation erforderlich"
    if str(row.get("candidate_type", "")) == "classical_rule":
        return "Klassischer ECG-Vergleichsmarker"
    return "Explorativer Forschungsmarker; keine Einzelfalldiagnose"


def _marker_columns() -> list[str]:
    return [
        "reference_kind", "evidence_stage", "claim_ceiling", "analysis_scope",
        "independent_modality_observed_n", "clock_alignment_confirmed_n", "timing_evidence_note",
        "failed_gates", "clinical_claim_allowed", "contract_version",
        "n_reference_eligible", "n_marker_observed", "n_marker_unavailable",
        "availability_denominator",
        "pathology_group",
        "pathology_group_id",
        "group_not_pooled",
        "endpoint_id",
        "endpoint_name_de",
        "endpoint_category_de",
        "endpoint_definition_de",
        "endpoint_source_de",
        "feature_id",
        "feature_name_de",
        "feature_tooltip_de",
        "canonical_feature_id",
        "candidate_type",
        "endpoint_feature_role",
        "reference_independence",
        "baseline_applicable",
        "baseline_registry_id",
        "baseline_display_name",
        "baseline_kind",
        "feature_role",
        "feature_modality",
        "conceptual_family",
        "scientific_tier",
        "panel_id",
        "evaluation_split",
        "inference_status",
        "n_positive",
        "n_negative",
        "missingness",
        "mean_signal_quality",
        "low_signal_quality_fraction",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "roc_auc_regular_rhythm",
        "n_regular_positive",
        "n_regular_negative",
        "average_precision",
        "average_precision_ci_low",
        "average_precision_ci_high",
        "sensitivity_at_90_specificity",
        "sensitivity_at_95_specificity",
        "specificity_at_90_sensitivity",
        "operating_sensitivity",
        "operating_specificity",
        "operating_point_attainable",
        "operating_point_validation_status",
        "operating_point_training_status",
        "operating_point_ranking_eligible",
        "training_operating_specificity",
        "training_operating_threshold",
        "sensitivity_at_95_specificity_reported",
        "selection_eligible",
        "q_value",
        "projection_hypothesis_prespecified",
        "projection_segment",
        "projection_plane",
        "projection_geometry_measure",
        "projection_hypothesis_restriction",
        "projection_hypothesis_restriction_operationalized",
        "projection_reference_modality",
        "projection_multiplicity_family",
        "projection_hypothesis_q_value",
        "projection_hypothesis_fdr_significant",
        "projection_hypothesis_version",
        "n_marker_raw_observed",
        "n_analysis_restriction_excluded",
        "analysis_restriction_status",
        "analysis_restriction_version",
        "analysis_restriction_required_columns_present",
        "analysis_restriction_missing_columns",
        "rank_biserial",
        "direction",
        "direction_agreement",
        "train_validation_auc_gap",
        "baseline_feature_id",
        "sensitivity_delta_vs_fixed_rule",
        "specificity_delta_vs_fixed_rule",
        "delta_auc",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "components",
        "component_count",
        "validation_stable",
        "post_outcome_hypothesis",
        "historical_pre_audit_result",
        "evidence_grade",
        "evidence_grade_de",
        "evidence_rank",
        "clinical_interpretation",
        "result_scope",
    ]


def _ensure_columns(
    frame: pd.DataFrame,
    columns,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = np.nan
    return output[list(columns)]


def _annotate_marker_display(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    terms = [clinical_display_term(value) for value in output["endpoint_id"]]
    output["endpoint_name_de"] = [item.display_name_de for item in terms]
    output["endpoint_category_de"] = [item.category_de for item in terms]
    output["endpoint_definition_de"] = [
        item.short_definition_de for item in terms
    ]
    output["endpoint_source_de"] = [item.source_summary_de for item in terms]
    output["feature_name_de"] = [
        biomarker_display_name(value) for value in output["feature_id"]
    ]
    output["feature_tooltip_de"] = [
        biomarker_tooltip(value) for value in output["feature_id"]
    ]
    return output


def _attach_training_operating_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Match one training threshold, never infer its attainability from validation."""
    columns = {
        "specificity_95_specificity": "training_operating_specificity",
        "specificity_95_threshold_aligned": "training_operating_threshold",
        "specificity_95_threshold_direction": "training_operating_direction",
    }
    output = frame.drop(columns=list(columns.values()), errors="ignore").copy()
    keys = [key for key in ("endpoint_id", "feature_id", "panel_id", "analysis_type",
                            "cohort_id", "confirmation_fold") if key in frame]
    if not {"endpoint_id", "feature_id", "evaluation_split", *columns}.issubset(frame):
        for target in columns.values():
            output[target] = np.nan
        return output
    training = frame.loc[frame.evaluation_split.eq("train"), keys + list(columns)].copy()
    training = training.loc[~training.duplicated(keys, keep=False)].rename(columns=columns)
    return output.merge(training, on=keys, how="left", validate="many_to_one")


def _annotate_operating_point_attainability(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep validation performance descriptive; rank only verified training points."""

    output = frame.copy()
    achieved = _numeric(output, "operating_specificity")
    attainable = achieved.between(0.0, 1.0) & achieved.ge(
        OPERATING_POINT_TARGET_SPECIFICITY - OPERATING_POINT_SPECIFICITY_TOLERANCE
    )
    output["operating_point_attainable"] = attainable
    output["operating_point_validation_status"] = np.where(
        ~achieved.between(0.0, 1.0), "unknown",
        np.where(attainable, "observed_target_met", "observed_target_not_met"),
    )
    reported = _numeric(output, "sensitivity_at_95_specificity")
    output["sensitivity_at_95_specificity_reported"] = reported
    training_specificity = _numeric(output, "training_operating_specificity")
    threshold = _numeric(output, "specificity_95_threshold_aligned")
    training_threshold = _numeric(output, "training_operating_threshold")
    known = (training_specificity.between(0, 1) & threshold.notna()
             & training_threshold.notna()
             & _series(output, "specificity_95_threshold_direction").isin(["above", "below"])
             & _series(output, "training_operating_direction").isin(["above", "below"]))
    same_threshold = (threshold.eq(training_threshold)
                      & _series(output, "specificity_95_threshold_direction").eq(
                          _series(output, "training_operating_direction")))
    training_pass = training_specificity.ge(OPERATING_POINT_TARGET_SPECIFICITY - 1e-12)
    rank_eligible = known & same_threshold & training_pass
    output["operating_point_ranking_eligible"] = rank_eligible
    output["operating_point_training_status"] = np.select(
        [~known, ~same_threshold, ~training_pass],
        ["unknown_training_evidence", "threshold_mismatch", "training_target_not_met"],
        default="training_target_met",
    )
    output["sensitivity_at_95_specificity"] = reported.where(rank_eligible)
    output["operating_sensitivity"] = reported
    return output


def _selection_eligible(frame: pd.DataFrame) -> pd.Series:
    return formal_selection_eligible(frame)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _series(
    frame: pd.DataFrame,
    column: str,
    default: object = np.nan,
) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _annotate_baseline_applicability(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["baseline_applicable"] = False
    output["baseline_registry_id"] = ""
    output["baseline_display_name"] = ""
    output["baseline_kind"] = ""
    registry = BaselineRegistry()
    cache: dict[str, dict[str, tuple[str, str, str]]] = {}
    for endpoint_id in output["endpoint_id"].astype(str).unique():
        applicable: dict[str, tuple[str, str, str]] = {}
        for definition in registry.for_endpoint(endpoint_id):
            if not definition.criterion_id:
                continue
            feature_id = f"classical_{definition.criterion_id}"
            applicable[feature_id] = (
                definition.baseline_id,
                definition.display_name,
                definition.kind.value,
            )
        cache[endpoint_id] = applicable
    for index, row in output.loc[
        output["candidate_type"].eq("classical_rule")
    ].iterrows():
        metadata = cache.get(str(row["endpoint_id"]), {}).get(
            str(row["feature_id"])
        )
        if metadata is None:
            continue
        output.at[index, "baseline_applicable"] = True
        output.at[index, "baseline_registry_id"] = metadata[0]
        output.at[index, "baseline_display_name"] = metadata[1]
        output.at[index, "baseline_kind"] = metadata[2]
    return output


def _float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _first(
    frame: pd.DataFrame,
    column: str,
    default="",
):
    if frame.empty or column not in frame:
        return default
    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


__all__ = [
    "RESULT_EXPLORER_VERSION",
    "ResultExplorerTables",
    "build_new_association_explorer",
    "build_result_explorer_tables",
    "rank_broad_discovery",
    "write_result_explorer_tables",
]
