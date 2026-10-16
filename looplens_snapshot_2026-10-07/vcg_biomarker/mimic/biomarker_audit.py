from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.biomarkers import BiomarkerRegistry
from ..core.biomarkers.catalog_biomarkers import EPS
from ..core.data_loader import LEAD_MAP
from ..core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    RhythmAssessment,
    RhythmClass,
    WaveStatus,
)
from ..core.preprocessing import construct_robust_morphology_record
from .classical import ClassicalECGCriteria
from .feature_roles import (
    ROLE_ALIAS,
    ROLE_CANDIDATE,
    ROLE_CLASSICAL,
    ROLE_COHORT,
    ROLE_NON_SCALAR,
    ROLE_RETIRED,
    feature_profile,
)
from .panels import ALL_CLASSICAL_FEATURE_IDS, PANEL_VERSION, default_feature_panels
from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_sha256,
)


BIOMARKER_AUDIT_VERSION = "1.10.0"
_BEST_FIT_CIRCULARITY = frozenset(f"{segment}_loop_roundness_best_fit_plane" for segment in ("p", "qrs", "qrst", "st", "t"))
_PROJECTED_CIRCULARITY = _BEST_FIT_CIRCULARITY | frozenset(
    f"{segment}_loop_{plane}_roundness"
    for segment in ("p", "qrs", "qrst", "st", "t")
    for plane in ("frontal_xy", "horizontal_xz", "sagittal_yz"))
_BOUNDED_ZERO_ONE = frozenset(
    {
        "qrs_t_directional_wasserstein",
        "qrs_t_loop_mirror_symmetry",
        "p_loop_biphasic_index",
        "t_loop_repolarization_tail_index",
        "vcg_phase_space_recurrence_rate",
        "vcg_phase_space_determinism",
        "repolarization_return_efficiency",
        "qrs_terminal_energy_fraction_20ms",
        "svg_cancellation_index",
        "qrs_t_mirror_discordant_fraction",
        "t_terminal_path_fraction",
        "qrs_beat_area_axis_coherence",
        "t_beat_area_axis_coherence",
    }
)
_BOUNDED_MINUS_ONE_ONE = frozenset(
    {
        "segmental_directional_memory",
        "qrs_initial_terminal_energy_balance",
    }
)
_ROTATION_INVARIANT = frozenset(
    {
        "qrs_loop_tortuosity",
        "qrs_t_directional_wasserstein",
        "qrs_t_loop_mirror_symmetry",
        "qrs_total_turning_angle_deg",
        "qrs_turning_angle_p95_deg",
        "qrs_terminal_energy_fraction_20ms",
        "qrs_initial_terminal_energy_balance",
        "qrs_t_normalized_hausdorff_distance",
        "qrs_t_normalized_dtw_distance",
        "qrs_t_mirror_phase_angle_median_deg",
        "qrs_t_mirror_phase_angle_p90_deg",
        "qrs_t_mirror_discordant_fraction",
        "svg_cancellation_index",
        "t_terminal_path_fraction",
        "t_terminal_direction_dispersion_deg",
        "st_t_transition_angle_deg",
        "qrs_beat_area_axis_coherence",
        "t_beat_area_axis_coherence",
        "beatwise_svg_direction_dispersion_deg",
    }
)
_SCALE_INVARIANT = frozenset(
    {
        "qrs_loop_tortuosity",
        "qrs_t_directional_wasserstein",
        "qrs_t_loop_mirror_symmetry",
        "qrs_total_turning_angle_deg",
        "qrs_turning_angle_p95_deg",
        "qrs_terminal_energy_fraction_20ms",
        "qrs_initial_terminal_energy_balance",
        "qrs_t_normalized_hausdorff_distance",
        "qrs_t_normalized_dtw_distance",
        "qrs_t_mirror_phase_angle_median_deg",
        "qrs_t_mirror_phase_angle_p90_deg",
        "qrs_t_mirror_discordant_fraction",
        "svg_cancellation_index",
        "t_terminal_path_fraction",
        "t_terminal_direction_dispersion_deg",
        "st_t_transition_angle_deg",
        "qrs_beat_area_axis_coherence",
        "t_beat_area_axis_coherence",
        "beatwise_svg_direction_dispersion_deg",
    }
)

_BOUNDED_ZERO_ONE |= _PROJECTED_CIRCULARITY
_SCALE_INVARIANT |= _PROJECTED_CIRCULARITY
# Only PCA-best-fit circularity is invariant to arbitrary 3D rotation;
# the geometry of a fixed anatomical projection can legitimately change.
_ROTATION_INVARIANT |= _BEST_FIT_CIRCULARITY


@dataclass(frozen=True)
class BiomarkerAuditResult:
    output_root: Path
    catalog_path: Path
    dynamic_path: Path
    classical_path: Path
    duplicate_path: Path
    summary_path: Path
    implementation_hash: str
    feature_count: int
    default_panel_blockers: int
    calculator_errors: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "output_root",
            "catalog_path",
            "dynamic_path",
            "classical_path",
            "duplicate_path",
            "summary_path",
        ):
            payload[key] = str(payload[key])
        return payload


def run_biomarker_audit(
    output_root: Path | str,
    *,
    dynamic: bool = True,
) -> BiomarkerAuditResult:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    panels = default_feature_panels()
    membership = {
        feature_id: panel.panel_id
        for panel in panels
        for feature_id in panel.feature_ids
    }
    catalog = _catalog_audit(membership)
    dynamic_frame = (
        _dynamic_audit(membership)
        if dynamic
        else pd.DataFrame(
            columns=(
                "feature_id",
                "case_id",
                "value",
                "confidence",
                "status",
                "notes",
            )
        )
    )
    classical_frame = (
        _classical_dynamic_audit()
        if dynamic
        else _classical_static_audit()
    )
    duplicates = _duplicate_candidates(dynamic_frame, membership)
    catalog = _merge_dynamic_summary(catalog, dynamic_frame, duplicates)
    blockers = catalog["default_panel_blocker"].fillna(False)
    calculator_errors = int(
        dynamic_frame["status"].eq("calculator_error").sum()
        if not dynamic_frame.empty
        else 0
    )
    classical_blockers = int(
        classical_frame["default_panel_blocker"].fillna(False).sum()
    )
    classical_errors = int(
        classical_frame["status"].eq("calculator_error").sum()
    )

    catalog_path = output / "biomarker_catalog_audit.parquet"
    dynamic_path = output / "biomarker_dynamic_audit.parquet"
    classical_path = output / "classical_criteria_audit.parquet"
    duplicate_path = output / "biomarker_duplicate_candidates.parquet"
    summary_path = output / "biomarker_audit_summary.json"
    atomic_write_parquet(catalog, catalog_path)
    atomic_write_parquet(dynamic_frame, dynamic_path)
    atomic_write_parquet(classical_frame, classical_path)
    atomic_write_parquet(duplicates, duplicate_path)
    implementation_hash = biomarker_audit_implementation_hash()
    summary = {
        "audit_version": BIOMARKER_AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_hash": implementation_hash,
        "panel_version": PANEL_VERSION,
        "feature_count": int(len(catalog)),
        "implemented_count": int(
            catalog["implementation_status"].eq("implemented").sum()
        ),
        "dynamically_evaluable_count": int(
            catalog["dynamic_finite_cases"].fillna(0).gt(0).sum()
        ),
        "scientifically_screenable_scalar_count": int(
            catalog["scientific_screening_eligible"].fillna(False).sum()
        ),
        "calculator_contract_status_counts": {
            str(key): int(value)
            for key, value in catalog[
                "calculator_contract_status"
            ].value_counts().items()
        },
        "role_counts": {
            str(key): int(value)
            for key, value in catalog["role"].value_counts().items()
        },
        "panel_counts": {
            panel.panel_id: int(len(panel.feature_ids))
            for panel in panels
        },
        "default_panel_blockers": int(blockers.sum()) + classical_blockers,
        "calculator_errors": calculator_errors + classical_errors,
        "biomarker_calculator_errors": calculator_errors,
        "classical_criterion_count": int(len(ALL_CLASSICAL_FEATURE_IDS)),
        "classical_contract_blockers": classical_blockers,
        "classical_calculator_errors": classical_errors,
        "classical_status_counts": {
            str(key): int(value)
            for key, value in classical_frame["status"].value_counts().items()
        },
        "synthetic_constant_candidate_count": int(
            catalog.get(
                "synthetic_constant_candidate",
                pd.Series(False, index=catalog.index),
            ).fillna(False).sum()
        ),
        "retired_invalid_proxy_count": int(
            catalog["role"].eq(ROLE_RETIRED).sum()
        ),
        "redundant_alias_count": int(
            catalog["role"].eq(ROLE_ALIAS).sum()
        ),
        "dynamic_case_count": int(
            dynamic_frame["case_id"].nunique()
            if not dynamic_frame.empty
            else 0
        ),
        "exact_dynamic_duplicate_pairs": int(
            duplicates["exact_match"].fillna(False).sum()
            if not duplicates.empty
            else 0
        ),
        "scientific_guard": (
            "This audit uses deterministic synthetic signals only. It verifies "
            "implementation contracts, not clinical validity. MIMIC locked_test "
            "and PTB fold 10 are never read."
        ),
    }
    atomic_write_json(summary_path, summary)
    return BiomarkerAuditResult(
        output_root=output,
        catalog_path=catalog_path,
        dynamic_path=dynamic_path,
        classical_path=classical_path,
        duplicate_path=duplicate_path,
        summary_path=summary_path,
        implementation_hash=implementation_hash,
        feature_count=int(len(catalog)),
        default_panel_blockers=int(blockers.sum()) + classical_blockers,
        calculator_errors=calculator_errors + classical_errors,
    )


def _classical_static_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_id": feature_id,
                "criterion_id": feature_id.removeprefix("classical_"),
                "case_id": "not_run",
                "status": "not_run",
                "evaluable": None,
                "value": None,
                "positive": None,
                "unit": "",
                "definition": "",
                "exploratory": None,
                "default_panel_blocker": False,
                "audit_issue": "",
            }
            for feature_id in ALL_CLASSICAL_FEATURE_IDS
        ]
    )


def _classical_dynamic_audit() -> pd.DataFrame:
    expected = {
        feature_id.removeprefix("classical_"): feature_id
        for feature_id in ALL_CLASSICAL_FEATURE_IDS
    }
    rows: list[dict] = []
    calculator = ClassicalECGCriteria()
    for case_id, delineation, ecg, _ in _synthetic_cases():
        try:
            results = calculator.compute(
                ecg,
                delineation,
                age=55,
                sex="M",
                # The device frontal QRS-T angle reads only the recorder's
                # axes; the normal case carries a plausible pair so the
                # contract check covers it (audit 5.1).
                **(
                    {"machine_qrs_axis_deg": 60.0, "machine_t_axis_deg": 45.0}
                    if case_id == "normal"
                    else {}
                ),
            )
            calculator_error = ""
        except Exception as exc:
            results = {}
            calculator_error = f"{type(exc).__name__}: {exc}"
        for criterion_id, feature_id in expected.items():
            criterion = results.get(criterion_id)
            issues: list[str] = []
            if calculator_error:
                status = "calculator_error"
                issues.append(calculator_error)
            elif criterion is None:
                status = "missing_criterion"
                issues.append("criterion_missing_from_calculator")
            else:
                value_finite = (
                    criterion.value is not None
                    and np.isfinite(float(criterion.value))
                )
                if criterion.evaluable and not value_finite:
                    issues.append("evaluable_criterion_without_finite_value")
                if criterion.evaluable and criterion.positive not in {
                    True,
                    False,
                    None,
                }:
                    issues.append("invalid_positive_state")
                if not criterion.evaluable and criterion.positive is not None:
                    issues.append("non_evaluable_criterion_has_positive_state")
                if not criterion.definition.strip():
                    issues.append("definition_missing")
                status = "contract_failure" if issues else (
                    "evaluable" if criterion.evaluable else "abstained"
                )
            rows.append(
                {
                    "feature_id": feature_id,
                    "criterion_id": criterion_id,
                    "case_id": case_id,
                    "status": status,
                    "evaluable": (
                        criterion.evaluable if criterion is not None else False
                    ),
                    "value": criterion.value if criterion is not None else None,
                    "positive": (
                        criterion.positive if criterion is not None else None
                    ),
                    "unit": criterion.unit if criterion is not None else "",
                    "definition": (
                        criterion.definition if criterion is not None else ""
                    ),
                    "exploratory": (
                        criterion.exploratory if criterion is not None else None
                    ),
                    "default_panel_blocker": status in {
                        "calculator_error",
                        "missing_criterion",
                        "contract_failure",
                    },
                    "audit_issue": "|".join(issues),
                }
            )
    frame = pd.DataFrame(rows)
    normal = frame.loc[frame["case_id"].eq("normal")]
    if len(normal) != len(expected) or not normal["evaluable"].fillna(False).all():
        normal_ids = set(normal["criterion_id"])
        missing = set(expected) - normal_ids
        mask = frame["case_id"].eq("normal") & ~frame["evaluable"].fillna(False)
        frame.loc[mask, "default_panel_blocker"] = True
        frame.loc[mask, "audit_issue"] = frame.loc[mask, "audit_issue"].map(
            lambda value: "|".join(
                item
                for item in (str(value).strip("|"), "normal_case_not_evaluable")
                if item
            )
        )
        if missing:
            raise RuntimeError(
                f"Classical audit failed to create expected rows: {sorted(missing)}"
            )
    return frame


def _catalog_audit(membership: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for definition in sorted(
        BiomarkerRegistry.definitions(),
        key=lambda item: (
            item.source_index if item.source_index is not None else 100_000,
            item.name,
        ),
    ):
        profile = feature_profile(
            definition.name,
            family=definition.family,
            category=definition.category,
            unit=definition.unit,
        )
        panel_id = membership.get(definition.name, "")
        issues: list[str] = []
        severity = "ok"
        if not definition.formula.strip():
            issues.append("formula_missing")
        if definition.implementation_status != "implemented":
            issues.append("calculator_not_implemented")
            severity = "warning"
        if profile.role == ROLE_ALIAS:
            issues.append(f"redundant_alias_of:{profile.canonical_feature_id}")
            severity = "info"
        if profile.role == ROLE_RETIRED:
            issues.append("historical_name_overclaimed_implementation")
            severity = "retired"
        if panel_id in {"primary_core", "sentinel"} and profile.role not in {
            ROLE_CANDIDATE,
            ROLE_CLASSICAL,
        }:
            issues.append("non_candidate_in_confirmatory_development_panel")
            severity = "blocker"
        if (
            panel_id in {"primary_core", "sentinel"}
            and not profile.is_canonical
        ):
            issues.append("alias_in_confirmatory_development_panel")
            severity = "blocker"
        if (
            panel_id in {"primary_core", "sentinel"}
            and profile.post_outcome_hypothesis
        ):
            issues.append("post_outcome_hypothesis_in_development_panel")
            severity = "blocker"
        prespecified_novel_definition = bool(
            definition.description.strip()
            and definition.formula.strip()
            and definition.source.strip()
        )
        if panel_id == "primary_core" and not definition.references:
            if prespecified_novel_definition:
                # A genuinely novel marker cannot be required to cite prior
                # clinical validation.  It may be tested as a frozen,
                # explicitly defined hypothesis, but must not be presented as
                # an established or clinically validated marker.
                issues.append(
                    "primary_core_prespecified_novel_without_"
                    "literature_reference"
                )
                if severity == "ok":
                    severity = "warning"
            else:
                issues.append("primary_core_reference_and_definition_missing")
                severity = "blocker"
        elif panel_id == "sentinel" and not definition.references:
            # A sentinel marker may be genuinely novel, but absence of a
            # reference/rationale prevents a confirmatory or clinical claim.
            issues.append("sentinel_reference_or_rationale_missing")
            if severity == "ok":
                severity = "warning"
        rows.append(
            {
                "source_index": definition.source_index,
                "feature_id": definition.name,
                "canonical_feature_id": profile.canonical_feature_id,
                "family": definition.family,
                "conceptual_family": profile.conceptual_family,
                "category": definition.category,
                "modality": profile.modality,
                "unit": definition.unit,
                "description": definition.description,
                "formula": definition.formula,
                "source": definition.source,
                "implementation_status": definition.implementation_status,
                "role": profile.role,
                "scientific_tier": profile.scientific_tier,
                "panel_id": panel_id,
                "core_sentinel_eligible": profile.core_sentinel_eligible,
                "discovery_eligible": profile.discovery_eligible,
                "combination_eligible": profile.combination_eligible,
                "post_outcome_hypothesis": profile.post_outcome_hypothesis,
                "audit_severity": severity,
                "audit_issues": "|".join(issues),
                "default_panel_blocker": severity == "blocker",
                "profile_reason": profile.reason,
                "reference_count": len(definition.references),
                "prespecified_novel_definition": prespecified_novel_definition,
                "claim_eligible": bool(definition.references)
                and severity != "blocker",
                "allowed_claim_scope": (
                    "confirmatory_candidate"
                    if definition.references and severity != "blocker"
                    else "prespecified_novel_hypothesis"
                    if prespecified_novel_definition and severity != "blocker"
                    else "exploratory_hypothesis_only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dynamic_audit(membership: dict[str, str]) -> pd.DataFrame:
    cases = _synthetic_cases()
    feature_ids = tuple(BiomarkerRegistry.available())
    rows: list[dict] = []
    for case_id, delineation, ecg, vcg in cases:
        robust_ecg, template_diagnostics = (
            construct_robust_morphology_record(
                ecg,
                delineation,
            )
        )
        robust_vcg, vcg_template_diagnostics = (
            construct_robust_morphology_record(
                vcg,
                delineation,
            )
        )
        results = BiomarkerRegistry.compute_all(
            delineation,
            robust_ecg,
            robust_vcg,
            delineation.fs,
            names=list(feature_ids),
            record_ecg_12lead=ecg,
            record_vcg_xyz=vcg,
        )
        for result in results:
            status = "finite"
            if result.value is None:
                status = (
                    "calculator_error"
                    if result.computation_notes.startswith("Error:")
                    else "not_evaluable"
                )
            elif not np.isfinite(result.value):
                status = "non_finite"
            range_issue = _range_issue(result.name, result.value)
            if range_issue:
                status = "range_violation"
            rows.append(
                {
                    "feature_id": result.name,
                    "case_id": case_id,
                    "panel_id": membership.get(result.name, ""),
                    "value": result.value,
                    "unit": result.unit,
                    "confidence": result.confidence,
                    "status": status,
                    "range_issue": range_issue,
                    "notes": result.computation_notes,
                    "morphology_template_status": (
                        template_diagnostics["status"]
                    ),
                    "vcg_template_status": (
                        vcg_template_diagnostics["status"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = _attach_invariance_checks(frame)
    return frame


def _range_issue(feature_id: str, value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if feature_id in _BOUNDED_ZERO_ONE and not (-1e-9 <= value <= 1.0 + 1e-9):
        return "expected_[0,1]"
    if feature_id in _BOUNDED_MINUS_ONE_ONE and not (
        -1.0 - 1e-9 <= value <= 1.0 + 1e-9
    ):
        return "expected_[-1,1]"
    return ""


def _attach_invariance_checks(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    pivot = output.pivot_table(
        index="feature_id",
        columns="case_id",
        values="value",
        aggfunc="first",
    )
    rotation_error: dict[str, float] = {}
    scale_error: dict[str, float] = {}
    for feature_id in pivot.index:
        normal = pivot.at[feature_id, "normal"]
        rotated = pivot.at[feature_id, "rotated"]
        scaled = pivot.at[feature_id, "amplitude_scaled"]
        if feature_id in _ROTATION_INVARIANT:
            rotation_error[feature_id] = _relative_error(normal, rotated)
        if feature_id in _SCALE_INVARIANT:
            scale_error[feature_id] = _relative_error(normal, scaled)
    output["rotation_invariance_relative_error"] = output["feature_id"].map(
        rotation_error
    )
    output["scale_invariance_relative_error"] = output["feature_id"].map(
        scale_error
    )
    output["invariance_violation"] = (
        output["rotation_invariance_relative_error"].fillna(0.0).gt(1e-5)
        | output["scale_invariance_relative_error"].fillna(0.0).gt(1e-5)
    )
    return output


def _relative_error(first: object, second: object) -> float:
    try:
        a = float(first)
        b = float(second)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b) / max(1.0, abs(a), abs(b)))


def _duplicate_candidates(
    dynamic: pd.DataFrame,
    membership: dict[str, str],
) -> pd.DataFrame:
    columns = (
        "feature_id_a",
        "feature_id_b",
        "canonical_a",
        "canonical_b",
        "panel_a",
        "panel_b",
        "exact_match",
        "pearson_correlation",
        "comparison_case_count",
        "known_alias",
    )
    if dynamic.empty:
        return pd.DataFrame(columns=columns)
    values = dynamic.pivot_table(
        index="case_id",
        columns="feature_id",
        values="value",
        aggfunc="first",
    )
    usable = [
        column
        for column in values
        if values[column].notna().sum() >= 4
        and values[column].nunique(dropna=True) >= 2
    ]
    if not usable:
        return pd.DataFrame(columns=columns)
    unit_by_feature = (
        dynamic.drop_duplicates("feature_id")
        .set_index("feature_id")["unit"]
        .astype(str)
        .to_dict()
    )
    signature_groups: dict[tuple, list[str]] = {}
    for feature_id in usable:
        profile = feature_profile(feature_id)
        if profile.role not in {
            ROLE_CANDIDATE,
            ROLE_ALIAS,
            "exploratory_composite",
        }:
            continue
        signature = tuple(
            None if not np.isfinite(value) else f"{float(value):.12g}"
            for value in values[feature_id].to_numpy(dtype=float)
        )
        key = (unit_by_feature.get(feature_id, ""), signature)
        signature_groups.setdefault(key, []).append(feature_id)

    rows: list[dict] = []
    candidate_pairs: set[tuple[str, str]] = set()
    for group in signature_groups.values():
        if len(group) < 2:
            continue
        for first_index, first in enumerate(sorted(group)):
            for second in sorted(group)[first_index + 1 :]:
                candidate_pairs.add((first, second))
    canonical_groups: dict[str, list[str]] = {}
    for feature_id in usable:
        canonical_groups.setdefault(
            feature_profile(feature_id).canonical_feature_id,
            [],
        ).append(feature_id)
    for group in canonical_groups.values():
        if len(group) < 2:
            continue
        for first_index, first in enumerate(sorted(group)):
            for second in sorted(group)[first_index + 1 :]:
                candidate_pairs.add((first, second))

    for first, second in sorted(candidate_pairs):
        canonical_a = feature_profile(first).canonical_feature_id
        canonical_b = feature_profile(second).canonical_feature_id
        known_alias = canonical_a == canonical_b
        complete = values[[first, second]].dropna()
        if len(complete) < 4:
            continue
        a = complete[first].to_numpy(dtype=float)
        b = complete[second].to_numpy(dtype=float)
        exact = bool(np.allclose(a, b, rtol=1e-10, atol=1e-12))
        if not exact and not known_alias:
            continue
        if np.std(a) <= EPS or np.std(b) <= EPS:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
        rows.append(
            {
                "feature_id_a": first,
                "feature_id_b": second,
                "canonical_a": canonical_a,
                "canonical_b": canonical_b,
                "panel_a": membership.get(first, ""),
                "panel_b": membership.get(second, ""),
                "exact_match": exact,
                "pearson_correlation": correlation,
                "comparison_case_count": len(complete),
                "known_alias": known_alias,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _merge_dynamic_summary(
    catalog: pd.DataFrame,
    dynamic: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> pd.DataFrame:
    if dynamic.empty:
        output = catalog.copy()
        for column in (
            "dynamic_finite_cases",
            "dynamic_not_evaluable_cases",
            "dynamic_error_cases",
            "dynamic_range_violations",
            "dynamic_invariance_violations",
            "dynamic_unique_values",
            "dynamic_duplicate_candidate_count",
        ):
            output[column] = 0
        output["synthetic_constant_candidate"] = False
        output["calculator_contract_status"] = "not_run"
        output["scientific_screening_eligible"] = False
        return output
    summary = (
        dynamic.groupby("feature_id", dropna=False)
        .agg(
            dynamic_finite_cases=("status", lambda values: int((values == "finite").sum())),
            dynamic_not_evaluable_cases=("status", lambda values: int((values == "not_evaluable").sum())),
            dynamic_error_cases=("status", lambda values: int((values == "calculator_error").sum())),
            dynamic_range_violations=("status", lambda values: int((values == "range_violation").sum())),
            dynamic_invariance_violations=("invariance_violation", "sum"),
            dynamic_unique_values=("value", lambda values: int(pd.Series(values).nunique(dropna=True))),
        )
        .reset_index()
    )
    output = catalog.merge(summary, on="feature_id", how="left")
    finite_count = output["dynamic_finite_cases"].fillna(0)
    error_count = output["dynamic_error_cases"].fillna(0)
    contract_failure = (
        output["dynamic_range_violations"].fillna(0).gt(0)
        | output["dynamic_invariance_violations"].fillna(0).gt(0)
    )
    output["calculator_contract_status"] = "implemented_evaluable"
    output.loc[
        output["role"].eq(ROLE_NON_SCALAR) & finite_count.gt(0),
        "calculator_contract_status",
    ] = "registered_bundle_output"
    output.loc[
        output["role"].eq(ROLE_COHORT) & finite_count.eq(0),
        "calculator_contract_status",
    ] = "requires_train_fitted_cohort_model"
    output.loc[
        output["role"].eq(ROLE_RETIRED),
        "calculator_contract_status",
    ] = "retired_no_scalar"
    output.loc[
        finite_count.eq(0)
        & ~output["role"].isin((ROLE_COHORT, ROLE_RETIRED)),
        "calculator_contract_status",
    ] = "not_evaluable_in_synthetic_audit"
    output.loc[contract_failure, "calculator_contract_status"] = (
        "contract_failure"
    )
    output.loc[error_count.gt(0), "calculator_contract_status"] = (
        "calculator_error"
    )
    output["scientific_screening_eligible"] = (
        output["role"].isin((ROLE_CANDIDATE, ROLE_CLASSICAL))
        & finite_count.gt(0)
        & ~contract_failure
        & error_count.eq(0)
    )
    duplicate_counts: dict[str, int] = {}
    if not duplicates.empty:
        for column in ("feature_id_a", "feature_id_b"):
            for feature_id, count in duplicates[column].value_counts().items():
                duplicate_counts[str(feature_id)] = (
                    duplicate_counts.get(str(feature_id), 0) + int(count)
                )
    output["dynamic_duplicate_candidate_count"] = (
        output["feature_id"].map(duplicate_counts).fillna(0).astype(int)
    )
    output["synthetic_constant_candidate"] = (
        output["role"].eq(ROLE_CANDIDATE)
        & output["dynamic_finite_cases"].fillna(0).ge(8)
        & output["dynamic_unique_values"].fillna(0).le(1)
    )
    constant_warning = (
        output["synthetic_constant_candidate"]
        & ~output["default_panel_blocker"].fillna(False)
    )
    output.loc[
        constant_warning & output["audit_severity"].eq("ok"),
        "audit_severity",
    ] = "warning"
    output.loc[constant_warning, "audit_issues"] = output.loc[
        constant_warning,
        "audit_issues",
    ].map(
        lambda value: "|".join(
            item
            for item in (
                str(value).strip("|"),
                "constant_across_synthetic_audit_cases",
            )
            if item
        )
    )
    dynamic_blocker = (
        output["panel_id"].isin(("primary_core", "sentinel"))
        & (
            output["dynamic_error_cases"].fillna(0).gt(0)
            | output["dynamic_range_violations"].fillna(0).gt(0)
            | output["dynamic_invariance_violations"].fillna(0).gt(0)
            | output["dynamic_finite_cases"].fillna(0).eq(0)
        )
    )
    output.loc[dynamic_blocker, "default_panel_blocker"] = True
    output.loc[dynamic_blocker, "audit_severity"] = "blocker"
    output.loc[dynamic_blocker, "audit_issues"] = output.loc[
        dynamic_blocker,
        "audit_issues",
    ].map(
        lambda value: "|".join(
            item
            for item in (str(value).strip("|"), "dynamic_contract_failure")
            if item
        )
    )
    return output


def _synthetic_cases() -> list[
    tuple[str, RecordDelineation, np.ndarray, np.ndarray]
]:
    fs = 500
    ecg, vcg, delineation = _synthetic_record(fs=fs)
    rotation = _rotation_matrix(0.61, -0.38, 0.27)
    noisy_rng = np.random.default_rng(20260730)
    noisy_vcg = vcg + noisy_rng.normal(0.0, 0.015, vcg.shape)
    noisy_ecg = ecg + noisy_rng.normal(0.0, 0.012, ecg.shape)
    baseline = np.sin(
        2.0 * np.pi * 0.28 * np.arange(len(vcg), dtype=float) / fs
    )[:, None]
    baseline_vector = baseline * np.asarray([[0.08, -0.05, 0.04]])
    wander_vcg = vcg + baseline_vector
    wander_ecg = wander_vcg @ _synthetic_lead_matrix().T
    inverted_t = vcg.copy()
    for beat in delineation.beats:
        inverted_t[beat.t_onset.sample : beat.t_offset.sample + 1] *= -1.0
    wide_ecg, wide_vcg, wide = _synthetic_record(fs=fs, wide_qrs=True)
    u_wave_ecg, u_wave_vcg, u_wave = _synthetic_record(
        fs=fs,
        u_wave=True,
    )
    af_ecg, af_vcg, af = _synthetic_record(fs=fs, atrial_absent=True)
    irregular_ecg, irregular_vcg, irregular = _synthetic_record(
        fs=fs,
        atrial_absent=True,
        irregular_rr=True,
    )
    variable_qt_ecg, variable_qt_vcg, variable_qt = _synthetic_record(
        fs=fs, irregular_rr=True, variable_qt=True,
    )
    flutter_ecg, flutter_vcg, flutter = _synthetic_record(
        fs=fs,
        atrial_absent=True,
    )
    flutter_wave = np.sin(
        2.0 * np.pi * 5.0 * np.arange(len(flutter_vcg), dtype=float) / fs
    )
    flutter_vcg = flutter_vcg + flutter_wave[:, None] * np.asarray(
        [[0.045, 0.025, -0.018]]
    )
    flutter_ecg = flutter_vcg @ _synthetic_lead_matrix().T
    flutter.rhythm = RhythmAssessment(
        rhythm_class=RhythmClass.FLUTTER_COMPATIBLE,
        confidence=0.92,
        rr_median_ms=1000.0,
        rr_cv=0.01,
        p_wave_coverage=0.0,
        atrial_frequency_hz=5.0,
        atrial_spectral_concentration=0.88,
        atrial_bandpower_ratio=0.31,
        flags=("continuous_flutter_like_atrial_activity",),
    )
    ectopic_vcg = vcg.copy()
    ectopic = _clone_delineation(delineation)
    target = ectopic.beats[len(ectopic.beats) // 2]
    target.morphology_outlier = True
    ectopic_rotation = _rotation_matrix(0.8, -0.45, 0.35)
    ectopic_vcg[
        target.r_onset.sample : target.t_offset.sample + 1
    ] = (
        1.35
        * ectopic_vcg[
            target.r_onset.sample : target.t_offset.sample + 1
        ]
        @ ectopic_rotation.T
    )
    ectopic_ecg = ectopic_vcg @ _synthetic_lead_matrix().T
    ectopic.rhythm = RhythmAssessment(
        rhythm_class=RhythmClass.MIXED,
        confidence=0.90,
        rr_median_ms=1000.0,
        rr_cv=0.04,
        p_wave_coverage=1.0,
        flags=("single_morphology_outlier",),
    )
    return [
        ("normal", delineation, ecg, vcg),
        ("repeat_normal", _clone_delineation(delineation), ecg.copy(), vcg.copy()),
        ("amplitude_scaled", _clone_delineation(delineation), 2.5 * ecg, 2.5 * vcg),
        ("rotated", _clone_delineation(delineation), ecg.copy(), vcg @ rotation.T),
        ("noisy", _clone_delineation(delineation), noisy_ecg, noisy_vcg),
        (
            "baseline_wander",
            _clone_delineation(delineation),
            wander_ecg,
            wander_vcg,
        ),
        (
            "low_voltage",
            _clone_delineation(delineation),
            0.12 * ecg,
            0.12 * vcg,
        ),
        ("inverted_t", _clone_delineation(delineation), ecg.copy(), inverted_t),
        ("wide_qrs", wide, wide_ecg, wide_vcg),
        ("u_wave", u_wave, u_wave_ecg, u_wave_vcg),
        ("atrial_absent", af, af_ecg, af_vcg),
        ("irregular_af_like", irregular, irregular_ecg, irregular_vcg),
        ("variable_qt_and_rr", variable_qt, variable_qt_ecg, variable_qt_vcg),
        ("flutter_like", flutter, flutter_ecg, flutter_vcg),
        ("single_morphology_outlier", ectopic, ectopic_ecg, ectopic_vcg),
    ]


def _synthetic_record(
    *,
    fs: int,
    wide_qrs: bool = False,
    atrial_absent: bool = False,
    irregular_rr: bool = False,
    u_wave: bool = False,
    variable_qt: bool = False,
) -> tuple[np.ndarray, np.ndarray, RecordDelineation]:
    duration_seconds = 10
    n = duration_seconds * fs
    time = np.arange(n, dtype=float) / fs
    vcg = np.zeros((n, 3), dtype=float)
    beats: list[BeatDelineation] = []
    if irregular_rr:
        intervals = np.asarray(
            (0.72, 1.18, 0.81, 1.27, 0.69, 1.11, 0.88, 1.22),
            dtype=float,
        )
        r_values = [fs]
        interval_index = 0
        while True:
            candidate = r_values[-1] + int(
                round(intervals[interval_index % len(intervals)] * fs)
            )
            if candidate >= n - fs // 2:
                break
            r_values.append(candidate)
            interval_index += 1
        r_samples = np.asarray(r_values, dtype=int)
    else:
        r_samples = np.arange(fs, n - fs // 2, fs)
    qrs_half = int(round((0.075 if wide_qrs else 0.045) * fs))
    for beat_index, r_peak in enumerate(r_samples):
        p_on = r_peak - int(round(0.20 * fs))
        p_peak = r_peak - int(round(0.16 * fs))
        p_off = r_peak - int(round(0.12 * fs))
        qrs_on = r_peak - qrs_half
        qrs_off = r_peak + qrs_half
        # Independently specified synthetic beat variability. Change the
        # waveform and its known timing together, not just its annotations.
        # This is a numerical audit input, not a clinical restitution model.
        t_scale = 1.0 + 0.15 * np.sin(1.37 * beat_index) if variable_qt else 1.0
        t_on = qrs_off + int(round(0.04 * t_scale * fs))
        t_peak = qrs_off + int(round(0.16 * t_scale * fs))
        t_off = qrs_off + int(round(0.32 * t_scale * fs))
        u_on = t_off + int(round(0.025 * fs))
        u_peak = t_off + int(round(0.10 * fs))
        u_off = t_off + int(round(0.18 * fs))
        if not atrial_absent:
            phase = (time - p_peak / fs) / 0.035
            p_wave = np.exp(-0.5 * phase ** 2)
            vcg[:, 0] += 0.09 * p_wave
            vcg[:, 1] += 0.05 * p_wave
            vcg[:, 2] -= 0.02 * p_wave
        qrs_phase = (time - r_peak / fs) / (0.035 if wide_qrs else 0.022)
        qrs_envelope = np.exp(-0.5 * qrs_phase ** 2)
        vcg[:, 0] += 0.85 * qrs_envelope * np.cos(1.8 * qrs_phase)
        vcg[:, 1] += 1.10 * qrs_envelope * np.sin(1.5 * qrs_phase)
        vcg[:, 2] += 0.48 * qrs_envelope * np.sin(2.1 * qrs_phase + 0.4)
        t_phase = (time - t_peak / fs) / (0.085 * t_scale)
        t_wave = np.exp(-0.5 * t_phase ** 2)
        vcg[:, 0] -= 0.30 * t_wave
        vcg[:, 1] += 0.42 * t_wave
        vcg[:, 2] -= 0.18 * t_wave
        if u_wave:
            u_phase = (time - u_peak / fs) / 0.045
            u_shape = np.exp(-0.5 * u_phase ** 2)
            vcg[:, 0] += 0.055 * u_shape
            vcg[:, 1] -= 0.080 * u_shape
            vcg[:, 2] += 0.105 * u_shape
        point = lambda sample: MarkerPoint(
            int(sample),
            0.96,
            "synthetic_audit",
            uncertainty_ms=2.0,
        )
        beats.append(
            BeatDelineation(
                r_peak=point(r_peak),
                p_onset=None if atrial_absent else point(p_on),
                p_peak=None if atrial_absent else point(p_peak),
                p_offset=None if atrial_absent else point(p_off),
                q_peak=point(qrs_on + max(1, qrs_half // 2)),
                r_onset=point(qrs_on),
                r_offset=point(qrs_off),
                s_peak=point(qrs_off - max(1, qrs_half // 2)),
                j_point=point(qrs_off),
                t_onset=point(t_on),
                t_peak=point(t_peak),
                t_offset=point(t_off),
                u_onset=point(u_on) if u_wave else None,
                u_peak=point(u_peak) if u_wave else None,
                u_offset=point(u_off) if u_wave else None,
                p_wave_absent=atrial_absent,
                p_status=(
                    WaveStatus.ABSENT
                    if atrial_absent
                    else WaveStatus.PRESENT
                ),
                qrs_status=WaveStatus.PRESENT,
                t_status=WaveStatus.PRESENT,
                u_status=(
                    WaveStatus.PRESENT
                    if u_wave
                    else WaveStatus.INDETERMINATE
                ),
                wide_qrs=wide_qrs,
            )
        )
    matrix = _synthetic_lead_matrix()
    ecg = vcg @ matrix.T
    rr_intervals_ms = np.diff(r_samples) / fs * 1000.0
    rr_median_ms = (
        float(np.median(rr_intervals_ms))
        if len(rr_intervals_ms)
        else 1000.0
    )
    rr_cv = (
        float(np.std(rr_intervals_ms, ddof=1) / np.mean(rr_intervals_ms))
        if len(rr_intervals_ms) > 1
        else 0.0
    )
    rhythm = RhythmAssessment(
        rhythm_class=(
            RhythmClass.AF_COMPATIBLE
            if atrial_absent and irregular_rr
            else RhythmClass.AF_COMPATIBLE
            if atrial_absent
            else (RhythmClass.WIDE_QRS if wide_qrs else RhythmClass.REGULAR)
        ),
        confidence=0.95,
        rr_median_ms=rr_median_ms,
        rr_cv=rr_cv,
        rr_successive_irregularity=(
            0.27 if irregular_rr else 0.01
        ),
        p_wave_coverage=0.0 if atrial_absent else 1.0,
        flags=(
            ("no_consistent_discrete_p", "irregular_rr")
            if atrial_absent and irregular_rr
            else ("no_consistent_discrete_p",)
            if atrial_absent
            else ()
        ),
    )
    delineation = RecordDelineation(
        beats=beats,
        fs=fs,
        n_samples=n,
        ecg_id=1,
        quality_score=0.97,
        source_method="synthetic_biomarker_audit",
        rhythm=rhythm,
    )
    return ecg, vcg, delineation


def _synthetic_lead_matrix() -> np.ndarray:
    return np.array(
        [
            [0.67, -0.18, 0.05],
            [0.35, 0.72, -0.10],
            [-0.32, 0.90, 0.14],
            [0.21, -0.08, 0.84],
            [0.08, 0.33, 0.73],
            [-0.12, 0.45, 0.61],
            [0.52, 0.28, -0.23],
            [-0.41, 0.56, 0.19],
            [0.18, -0.64, 0.42],
            [0.73, 0.11, -0.35],
            [-0.27, 0.39, 0.68],
            [0.44, -0.31, 0.51],
        ],
        dtype=float,
    )


def _clone_delineation(value: RecordDelineation) -> RecordDelineation:
    import copy

    return copy.deepcopy(value)


def _rotation_matrix(
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def biomarker_audit_implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "core" / "biomarkers" / "base.py",
        root / "core" / "biomarkers" / "catalog.py",
        root / "core" / "biomarkers" / "catalog_biomarkers.py",
        root / "core" / "biomarkers" / "advanced_vcg_biomarkers.py",
        root / "core" / "preprocessing.py",
        root / "core" / "delineation" / "base.py",
        root / "core" / "delineation" / "fusion_delineator.py",
        root / "mimic" / "feature_roles.py",
        root / "mimic" / "panels.py",
        root / "mimic" / "optimization.py",
        root / "mimic" / "classical.py",
        root / "mimic" / "baseline_registry.py",
    )
    payload = {
        "audit_version": BIOMARKER_AUDIT_VERSION,
        "files": {
            str(path.relative_to(root)): file_sha256(path)
            for path in paths
        },
    }
    return content_hash(payload)


__all__ = [
    "BIOMARKER_AUDIT_VERSION",
    "BiomarkerAuditResult",
    "biomarker_audit_implementation_hash",
    "run_biomarker_audit",
]
