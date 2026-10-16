"""The reporting set: which markers a vectorcardiographic report must carry.

This is the assembly point of the project's stated purpose. The screens say
which markers separate which endpoints; that is not a guideline. A guideline
says what a report contains, in what unit, with what reference, what change is
large enough to mean anything, and what the number may and may not be used to
claim. Those are five different tables and this stage builds them from
measurements the run already made, rather than from opinion.

**A marker earns its place; it is not chosen.** Four conditions, all measured
elsewhere in the pipeline, and each one is a veto:

*It must have a reference range.* From the ``reference-ranges`` stage, in at
least half the strata it could be reported in. A marker without a reference
value cannot be printed beside a patient's number.

*Its repeatability must be known.* From the ``repeatability`` stage: an
intraclass correlation above the declared reliability threshold and a finite
repeatability coefficient. Without it a follow-up measurement cannot be
compared with the first, and follow-up is most of what a report is for.

*Its transform dependence must be measured.* From ``transform-agreement``.
Not that it must be small -- a strongly transform-dependent marker is still
reportable -- but it must be known and printed, because otherwise the number
is not portable to any other laboratory.

*It must have shown something.* Either an external literature anchor, or a
replicated association in this run's own screen at the declared effect floor.

A marker failing any of the four is listed with the condition it failed. The
set is therefore auditable: a reader can see what was excluded and why, which
a hand-picked list cannot offer.

**What the report must print beside every value.** Four fields, and they are
mandatory rather than recommended, because each one was measured to change the
number: the VCG transform (twenty degrees of spatial QRS-T angle between Kors
and inverse Dower), the zero-point method (the origin estimate the loop is
measured from), the signal quality score, and the heart rate. A value without
these four is not interpretable, and this project has the measurements to say
so rather than merely assert it.

**What the algorithm may claim.** Every entry carries its claim ceiling, and
the ceiling is looked up rather than inferred, so a large area under the curve
cannot promote a claim past it. Two independent things can block a claim, and
crossing them produces the plainest statement this project can make about its
own design.

Of the 72 endpoints, **six** are verified widely enough for a diagnostic
accuracy claim to be confirmable at all -- the rest fail because the reference
standard was obtained for a few per cent of recordings, an echocardiogram
being done on one or two. Of those six, **five are read off the same tracing**:
the three new-onset atrial fibrillation horizons, and the two QTc thresholds,
which are cut from the interval the recorder measured itself.

    ================  =================  =============
    endpoint          verification rate  circularity
    ================  =================  =============
    mortality_30d               1,000    independent
    new_onset_af_30d            1,000    ECG-defined
    new_onset_af_90d            1,000    ECG-defined
    new_onset_af_1y             1,000    ECG-defined
    qtc_gt_480                  0,890    ECG-defined
    qtc_gt_500                  0,890    ECG-defined
    ================  =================  =============

**One endpoint of 72 survives both filters, and it is 30-day mortality** --
which is a prognosis, not a diagnosis. In this cohort the vectorcardiogram can
support prognostic claims and descriptive associations, and not one confirmed
diagnostic-accuracy claim about cardiac structure. That is a property of which
patients get an echocardiogram, not of any marker, and it is why the reporting
algorithm prints the ceiling beside every recommendation instead of leaving a
reader to take an absent claim for an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    scientific_config_hash,
)


REPORTING_GUIDELINE_VERSION = "1.3.0"

# Fields a report must print beside every measured value. Each was measured to
# move the number, which is why the list is short and why it is mandatory.
MANDATORY_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "vcg_transform",
        "Kors and inverse Dower differ by about twenty degrees of spatial "
        "QRS-T angle in healthy adults; a threshold without its transform "
        "cannot be applied elsewhere.",
    ),
    (
        "vcg_origin_method",
        "Every loop quantity is measured from an estimated isoelectric "
        "origin. Which estimate was available changes the value, and the "
        "estimate is not always the same one.",
    ),
    (
        "signal_quality_score",
        "Measured in this project: recording quality is more strongly "
        "associated with several diagnoses than any physiological marker "
        "is. A value from a poor tracing is a different measurement.",
    ),
    (
        "heart_rate_bpm",
        "Repolarisation markers move by a third across the physiological "
        "rate range. A value without a rate is not comparable with a "
        "reference interval that conditions on one.",
    ),
)

# Share of a marker's possible strata that must carry an estimable reference
# range before it can be reported.
MINIMUM_ESTIMABLE_STRATUM_SHARE = 0.5

# The reliability floor for a marker whose change is to be interpreted. Below
# it, the repeatability coefficient exceeds the between-subject spread and a
# follow-up difference says nothing.
MINIMUM_REPORTABLE_ICC = 0.75

_REPORTING_SET_COLUMNS = (
    "marker_id",
    "included",
    "excluded_reason",
    "unit",
    "estimable_stratum_share",
    "reference_strata",
    "icc",
    "repeatability_coefficient",
    "smallest_detectable_change",
    "transform_agreement_class",
    "transform_bias",
    "transform_limit_of_agreement_low",
    "transform_limit_of_agreement_high",
    "evidence_source",
    "best_endpoint",
    "best_validation_auc",
    "best_validation_auc_disattenuated",
    "guideline_version",
)


def disattenuated_auc(auc: float, icc: float) -> float:
    """AUC corrected for measurement error with the marker's ICC.

    Audit 5.3.5. Under the equal-variance binormal model the observed
    separation is d' = sqrt(reliability) x the true separation, with
    reliability = between-subject / total variance = ICC; so the true AUC is
    Phi(Phi^-1(AUC) / sqrt(ICC)). An upper estimate of what an error-free
    measurement of the same quantity would reach, not a claim about the
    marker as measured. NaN outside 0 < ICC <= 1 or 0 < AUC < 1.
    """

    from scipy import stats

    if not (np.isfinite(auc) and np.isfinite(icc)) or not (0.0 < icc <= 1.0) or not (0.0 < auc < 1.0):
        return float("nan")
    return float(stats.norm.cdf(stats.norm.ppf(float(auc)) / np.sqrt(float(icc))))


@dataclass(frozen=True)
class ReportingSetPolicy:
    minimum_estimable_stratum_share: float = MINIMUM_ESTIMABLE_STRATUM_SHARE
    minimum_icc: float = MINIMUM_REPORTABLE_ICC
    minimum_validation_auc: float = 0.60
    literature_anchored: tuple[str, ...] = (
        # Quantities with an external reference value, which is what lets a
        # threshold from this project be checked against somebody else's.
        "qrs_duration_ms",
        "qt_interval_ms",
        "qtc_fridericia_ms",
        "spatial_qrs_t_angle_deg",
        "svg_magnitude",
        "qrs_area_vector_magnitude_uv_ms",
        "frontal_xy_qrs_t_angle_deg",
        "tpeak_tend_ms",
        "classical_sokolow_lyon",
        "classical_cornell_product",
    )

    def to_dict(self) -> dict:
        return {
            "minimum_estimable_stratum_share": float(
                self.minimum_estimable_stratum_share
            ),
            "minimum_icc": float(self.minimum_icc),
            "minimum_validation_auc": float(self.minimum_validation_auc),
            "literature_anchored": list(self.literature_anchored),
        }


def _estimable_share(ranges: pd.DataFrame) -> pd.DataFrame:
    """Per marker: in how many of its strata a reference range exists."""

    if ranges.empty:
        return pd.DataFrame(
            columns=["marker_id", "estimable_stratum_share", "reference_strata"]
        )
    grouped = ranges.groupby("marker_id", as_index=False).agg(
        reference_strata=("estimable", "size"),
        estimable_strata=("estimable", lambda column: int(column.astype(bool).sum())),
    )
    grouped["estimable_stratum_share"] = np.where(
        grouped["reference_strata"] > 0,
        grouped["estimable_strata"] / grouped["reference_strata"],
        0.0,
    )
    return grouped[
        ["marker_id", "estimable_stratum_share", "reference_strata"]
    ]


def _best_screen_result(screening: pd.DataFrame) -> pd.DataFrame:
    # No effect floor here on purpose: the floor is applied in
    # ``build_reporting_set`` beside the other three conditions, so that a
    # marker excluded for its effect size is excluded in the same place, with
    # the same wording, as one excluded for its repeatability.
    """Strongest observed association, never itself evidence of replication."""

    if screening.empty:
        return pd.DataFrame(
            columns=["marker_id", "best_endpoint", "best_validation_auc"]
        )
    frame = screening.copy()
    if "evaluation_split" in frame.columns:
        frame = frame[frame["evaluation_split"].astype(str) == "validation"]
    if frame.empty:
        return pd.DataFrame(
            columns=["marker_id", "best_endpoint", "best_validation_auc"]
        )
    frame = frame.rename(columns={"feature_id": "marker_id"})
    frame["directed_auc"] = pd.to_numeric(
        frame.get("roc_auc"), errors="coerce"
    )
    # A marker that separates in the opposite direction is as informative; the
    # ranking is on distance from chance, not on the raw number.
    frame["distance"] = (frame["directed_auc"] - 0.5).abs()
    frame = frame.dropna(subset=["distance"])
    if frame.empty:
        return pd.DataFrame(
            columns=["marker_id", "best_endpoint", "best_validation_auc"]
        )
    best = frame.sort_values("distance", ascending=False).drop_duplicates(
        "marker_id", keep="first"
    )
    return best[["marker_id", "endpoint_id", "directed_auc"]].rename(
        columns={"endpoint_id": "best_endpoint", "directed_auc": "best_validation_auc"}
    )


def build_reporting_set(
    *,
    reference_ranges: pd.DataFrame,
    repeatability: pd.DataFrame,
    transform_agreement: pd.DataFrame,
    screening: pd.DataFrame,
    marker_units: Mapping[str, Optional[str]],
    policy: ReportingSetPolicy = ReportingSetPolicy(),
    replicated_criteria: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Decide, per marker, whether a report may carry it -- and say why not."""

    shares = _estimable_share(reference_ranges)
    best = _best_screen_result(screening)

    repeat = repeatability.copy()
    if not repeat.empty:
        rename = {}
        for source, target in (
            ("feature_id", "marker_id"),
            ("marker", "marker_id"),
        ):
            if source in repeat.columns and "marker_id" not in repeat.columns:
                rename[source] = target
        repeat = repeat.rename(columns=rename)
        keep = [
            name
            for name in (
                "marker_id",
                "icc",
                "repeatability_coefficient",
                "smallest_detectable_change",
            )
            if name in repeat.columns
        ]
        repeat = repeat[keep].drop_duplicates("marker_id")

    agreement = transform_agreement.copy()
    if not agreement.empty:
        agreement = agreement.sort_values("marker_id").drop_duplicates(
            "marker_id", keep="first"
        )
        agreement = agreement.rename(
            columns={
                "agreement_class": "transform_agreement_class",
                "bias": "transform_bias",
                "limit_of_agreement_low": "transform_limit_of_agreement_low",
                "limit_of_agreement_high": "transform_limit_of_agreement_high",
            }
        )
        agreement = agreement[
            [
                name
                for name in (
                    "marker_id",
                    "transform_agreement_class",
                    "transform_bias",
                    "transform_limit_of_agreement_low",
                    "transform_limit_of_agreement_high",
                )
                if name in agreement.columns
            ]
        ]

    candidates = sorted(
        set(shares.get("marker_id", pd.Series(dtype=str)))
        | set(repeat.get("marker_id", pd.Series(dtype=str)))
        | set(agreement.get("marker_id", pd.Series(dtype=str)))
    )
    if not candidates:
        return pd.DataFrame(columns=list(_REPORTING_SET_COLUMNS))

    table = pd.DataFrame({"marker_id": candidates})
    for other in (shares, repeat, agreement, best):
        if other is not None and not other.empty:
            table = table.merge(other, on="marker_id", how="left")

    table["unit"] = [marker_units.get(name) for name in table["marker_id"]]
    table["best_validation_auc_disattenuated"] = [
        disattenuated_auc(
            float(getattr(row, "best_validation_auc", np.nan)),
            float(getattr(row, "icc", np.nan)),
        )
        for row in table.itertuples(index=False)
    ]
    anchored = set(policy.literature_anchored)
    replicated_markers = set()
    if replicated_criteria is not None and not replicated_criteria.empty:
        from .result_contract import true
        required = {"feature_id", "stage", "core_replication_evidence_passed",
                    "project_wide_multiplicity_confirmed", "cross_fit_confirmed",
                    "verification_bias_gate_passed", "analysis_completeness"}
        if required <= set(replicated_criteria):
            eligible = replicated_criteria["stage"].isin(["validation_replicated", "frozen", "locked_test_validated", "external_validated"])
            eligible &= replicated_criteria["analysis_completeness"].eq("complete_snapshot")
            for gate in ("core_replication_evidence_passed", "project_wide_multiplicity_confirmed", "cross_fit_confirmed", "verification_bias_gate_passed"):
                eligible &= replicated_criteria[gate].map(true)
            replicated_markers = set(replicated_criteria.loc[eligible, "feature_id"])

    reasons: list[str] = []
    included: list[bool] = []
    evidence: list[str] = []
    for row in table.itertuples(index=False):
        failed: list[str] = []
        share = getattr(row, "estimable_stratum_share", np.nan)
        if not np.isfinite(share) or share < policy.minimum_estimable_stratum_share:
            failed.append("no_reference_range_in_enough_strata")
        icc = getattr(row, "icc", np.nan)
        coefficient = getattr(row, "repeatability_coefficient", np.nan)
        if not np.isfinite(icc) or icc < policy.minimum_icc:
            failed.append("repeatability_below_reportable_reliability")
        elif not np.isfinite(coefficient):
            failed.append("no_repeatability_coefficient")
        agreement_class = getattr(row, "transform_agreement_class", None)
        if not isinstance(agreement_class, str) or agreement_class in {
            "",
            "not_assessable",
        }:
            failed.append("transform_dependence_not_measured")
        auc = getattr(row, "best_validation_auc", np.nan)
        is_anchored = row.marker_id in anchored
        shows_something = (
            np.isfinite(auc)
            and abs(float(auc) - 0.5) >= (policy.minimum_validation_auc - 0.5)
        )
        if is_anchored:
            evidence.append("external_literature")
        elif row.marker_id in replicated_markers:
            evidence.append("replicated_in_this_run")
        elif shows_something:
            evidence.append("observed_association_not_replication")
            failed.append("replication_not_established_by_screening_auc")
        else:
            evidence.append("none")
            failed.append("no_external_anchor_and_no_replicated_association")
        reasons.append("|".join(failed))
        included.append(not failed)

    table["included"] = included
    table["excluded_reason"] = reasons
    table["evidence_source"] = evidence
    table["guideline_version"] = REPORTING_GUIDELINE_VERSION
    for column in _REPORTING_SET_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return (
        table[list(_REPORTING_SET_COLUMNS)]
        .sort_values(["included", "marker_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_reporting_algorithm(
    *,
    reporting_set: pd.DataFrame,
    operating_points: pd.DataFrame,
    circularity: Optional[pd.DataFrame] = None,
    reachability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per clinical question: what to read, at which cut, claiming what.

    The claim ceiling is not derived from the performance. An endpoint the
    electrocardiogram itself defines cannot be a diagnostic-accuracy claim at
    any area under the curve, and an endpoint whose reference standard was
    obtained for two per cent of recordings cannot be confirmed as one either.
    Both are looked up rather than inferred, so a strong number cannot promote
    a claim past its ceiling.
    """

    columns = [
        "endpoint_id",
        "marker_id",
        "operating_point",
        "threshold_source",
        "threshold_original",
        "sensitivity",
        "specificity",
        "claim_ceiling",
        "claim_ceiling_reason",
        "verification_rate",
        "reportable",
        "guideline_version",
    ]
    # Only the operating points are required. An empty reporting set means a
    # marker-level input has not run yet, and the claim ceiling does not depend
    # on one -- it is a property of the endpoint. Returning nothing here hid
    # the most important column of the stage behind an unrelated missing file:
    # measured on the development run, all inputs but the screen were absent
    # and the algorithm came back with zero rows instead of the ceilings it
    # could already state. Every row then carries reportable = False, which is
    # true and visible.
    if operating_points.empty:
        return pd.DataFrame(columns=columns)

    reportable: set[str] = set()
    if not reporting_set.empty and "included" in reporting_set.columns:
        reportable = set(
            reporting_set.loc[
                reporting_set["included"].astype(bool), "marker_id"
            ].astype(str)
        )
    frame = operating_points.copy()
    if "evaluation_split" in frame.columns:
        frame = frame[frame["evaluation_split"].astype(str) == "validation"]
    frame = frame.rename(columns={"feature_id": "marker_id"})
    frame["marker_id"] = frame["marker_id"].astype(str)
    frame["endpoint_id"] = frame["endpoint_id"].astype(str)

    # Classified from the endpoint identifier by default. The registry is the
    # one the phenotype screen uses, so an endpoint and a phenotype naming the
    # same thing cannot be judged differently; a caller may still pass a
    # prepared table, which then overrides individual entries.
    from .ecg_derived_targets import classify_target

    circular_map: dict[str, str] = {
        str(endpoint): classify_target(
            target_id=endpoint, target_name=endpoint
        ).classification
        for endpoint in frame["endpoint_id"].unique()
    }
    if circularity is not None and not circularity.empty:
        target_column = next(
            (
                name
                for name in ("target_id", "endpoint_id", "label_id", "phenotype_id")
                for _ in (0,)
                if name in circularity.columns
            ),
            None,
        )
        class_column = next(
            (
                name
                for name in ("circularity_class", "ecg_derived_class", "class")
                if name in circularity.columns
            ),
            None,
        )
        if target_column and class_column:
            circular_map.update(
                {
                    str(key): str(value)
                    for key, value in zip(
                        circularity[target_column], circularity[class_column]
                    )
                }
            )

    reach_map: dict[str, tuple[bool, float]] = {}
    if reachability is not None and not reachability.empty:
        key = "label_id" if "label_id" in reachability.columns else None
        if key:
            reach_map = {
                str(row[key]): (
                    bool(row.get("diagnostic_accuracy_claim_reachable", False)),
                    float(row.get("verification_rate", float("nan"))),
                )
                for _, row in reachability.iterrows()
            }

    rows = []
    for row in frame.itertuples(index=False):
        endpoint = str(row.endpoint_id)
        marker = str(row.marker_id)
        circular = circular_map.get(endpoint, "")
        known = endpoint in reach_map
        reachable, rate = reach_map.get(endpoint, (False, float("nan")))
        if circular and not circular.startswith("independent"):
            ceiling = "analytic_concordance"
            reason = f"endpoint_is_ecg_derived:{circular}"
        elif not known:
            # The default must never be the highest ceiling. Measured on the
            # development run, where ``build-labels`` had not yet written the
            # reachability table: defaulting to reachable granted
            # ``diagnostic_accuracy`` to 56.602 rows, ``aortic_stenosis`` and
            # ``significant_valvular_disease`` among them -- endpoints whose
            # reference standard exists for one or two per cent of recordings.
            # An unmeasured gate is an open gate only if it is written that
            # way.
            ceiling = "descriptive_association"
            reason = "verification_rate_not_measured_for_this_endpoint"
        elif not reachable:
            ceiling = "descriptive_association"
            reason = "reference_standard_not_obtained_for_enough_recordings"
        else:
            ceiling = "diagnostic_accuracy"
            reason = "reference_independent_and_widely_verified"
        rows.append(
            {
                "endpoint_id": endpoint,
                "marker_id": marker,
                "operating_point": getattr(row, "operating_point", None),
                "threshold_source": getattr(row, "threshold_source", None),
                "threshold_original": getattr(row, "threshold_original", np.nan),
                "sensitivity": getattr(row, "sensitivity", np.nan),
                "specificity": getattr(row, "specificity", np.nan),
                "claim_ceiling": ceiling,
                "claim_ceiling_reason": reason,
                "verification_rate": rate,
                "reportable": marker in reportable,
                "guideline_version": REPORTING_GUIDELINE_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def mandatory_context_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": field,
                "reason": reason,
                "guideline_version": REPORTING_GUIDELINE_VERSION,
            }
            for field, reason in MANDATORY_CONTEXT_FIELDS
        ]
    )


def intraindividual_change_note() -> dict:
    """The formula that turns any marker into a follow-up marker.

    The repeatability stage already produces a repeatability coefficient per
    marker: the largest difference between two measurements of an unchanged
    patient that measurement error alone explains, at 95 %. Dividing an
    observed change by it gives a number every clinician already knows how to
    read -- below one, the change is within measurement error; above one, it
    is not.

    This costs nothing to add, because the coefficients exist. It is the piece
    that makes a reporting guideline usable for follow-up rather than only for
    a single recording.
    """

    return {
        "formula": "delta_z = (value_2 - value_1) / repeatability_coefficient",
        "interpretation_below_one": (
            "within measurement error; no change may be reported"
        ),
        "interpretation_above_one": (
            "larger than measurement error explains at 95 %"
        ),
        "source": "repeatability stage, per marker",
        "requires": [
            "both recordings under the same VCG transform",
            "both recordings above the signal quality threshold",
            "rate difference reported alongside, since rate moves the value",
        ],
        "guideline_version": REPORTING_GUIDELINE_VERSION,
    }


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a missing input is reported, not raised
        return pd.DataFrame()


def run_reporting_guideline(
    config: MimicProjectConfig,
    *,
    run_id: str,
    policy: ReportingSetPolicy = ReportingSetPolicy(),
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Assemble the reporting set, the algorithm and the mandatory context."""

    run_root = config.runs_root / run_id
    output_root = run_root / "guideline"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "reporting_guideline.manifest.json"

    ranges_path = run_root / "reference_ranges" / "marker_reference_ranges.parquet"
    # Written to the run root, not into a subdirectory -- checked against
    # ``repeatability_stage`` rather than assumed, because a guideline
    # component silently missing its input would only show up as a reporting
    # set that excludes every marker for the same reason.
    repeatability_path = run_root / "marker_repeatability.parquet"
    agreement_path = output_root / "transform_agreement.parquet"
    screening_path = run_root / "screening_results.parquet"
    operating_path = run_root / "operating_point_results.parquet"
    criteria_path = run_root / "criterion_candidates.parquet"
    adjusted_path = run_root / "adjusted_model_results.parquet"
    decision_curve_path = run_root / "decision_curve_results.parquet"
    circularity_path = run_root / "phenome" / "phenome_ecg_circularity.parquet"
    reachability_path = (
        config.labels_root / "endpoint_verification_reachability.parquet"
    )

    inputs = {
        "reference_ranges": ranges_path,
        "repeatability": repeatability_path,
        "transform_agreement": agreement_path,
        "screening": screening_path,
        "operating_points": operating_path,
        "criterion_candidates": criteria_path,
        "adjusted_models": adjusted_path,
        "decision_curves": decision_curve_path,
    }
    missing = sorted(name for name, path in inputs.items() if not path.is_file())
    input_hash = content_hash(
        {
            "run_id": run_id,
            "policy": policy.to_dict(),
            "inputs": {name: str(path) for name, path in inputs.items()},
            "missing": missing,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash, config_hash, REPORTING_GUIDELINE_VERSION
        ):
            return previous

    if progress:
        progress("Berichtssatz: Eingaben lesen")

    from .marker_plausibility import marker_units

    reporting_set = build_reporting_set(
        reference_ranges=_read_optional(ranges_path),
        repeatability=_read_optional(repeatability_path),
        transform_agreement=_read_optional(agreement_path),
        screening=_read_optional(screening_path),
        marker_units=marker_units(),
        policy=policy,
        replicated_criteria=_read_optional(criteria_path),
    )
    algorithm = build_reporting_algorithm(
        reporting_set=reporting_set,
        operating_points=_read_optional(operating_path),
        circularity=_read_optional(circularity_path),
        reachability=_read_optional(reachability_path),
    )
    from .evidence_contract import build_candidate_evidence_contract

    candidate_evidence = build_candidate_evidence_contract(
        criteria=_read_optional(criteria_path),
        reporting_set=reporting_set,
        adjusted_models=_read_optional(adjusted_path),
        screening=_read_optional(screening_path),
        decision_curves=_read_optional(decision_curve_path),
    )

    set_path = output_root / "minimum_reporting_set.parquet"
    algorithm_path = output_root / "reporting_algorithm.parquet"
    context_path = output_root / "mandatory_context_fields.parquet"
    evidence_path = output_root / "candidate_evidence.parquet"
    note_path = output_root / "intraindividual_change.json"
    atomic_write_parquet(reporting_set, set_path)
    atomic_write_parquet(algorithm, algorithm_path)
    atomic_write_parquet(mandatory_context_table(), context_path)
    atomic_write_parquet(candidate_evidence, evidence_path)
    atomic_write_json(note_path, intraindividual_change_note())
    # Audit 5.3.8: STARD 2015, TRIPOD+AI and RECORD carried with the run.
    from .reporting_checklists import build_reporting_checklists

    checklist_path = output_root / "reporting_checklists.json"
    atomic_write_json(
        checklist_path,
        build_reporting_checklists(
            (run_root, config.labels_root, config.evidence_root, config.features_root)
        ),
    )

    included = (
        int(reporting_set["included"].astype(bool).sum())
        if not reporting_set.empty
        else 0
    )
    if progress:
        progress(
            f"Berichtssatz: {included:,} von {len(reporting_set):,} Markern "
            "erfuellen alle vier Bedingungen"
        )
    manifest = StageManifest.complete(
        stage="reporting-guideline",
        stage_version=REPORTING_GUIDELINE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        # An input that has not run yet must not be frozen into a reusable
        # "complete" manifest: the tables would stay empty until someone
        # forced the stage. ``matches()`` only reuses a manifest whose status
        # is exactly "complete".
        status="complete" if not missing else "incomplete_inputs",
        output_paths=(
            str(set_path.resolve()),
            str(algorithm_path.resolve()),
            str(context_path.resolve()),
            str(evidence_path.resolve()),
            str(note_path.resolve()),
            str(checklist_path.resolve()),
        ),
        row_counts={
            "candidate_markers": int(len(reporting_set)),
            "reporting_set_markers": included,
            "algorithm_rows": int(len(algorithm)),
            "candidate_evidence_rows": int(len(candidate_evidence)),
            "guideline_candidates": (
                int(candidate_evidence["evidence_tier"].eq("guideline_candidate").sum())
                if not candidate_evidence.empty
                else 0
            ),
            "clinically_validated": (
                int(candidate_evidence["evidence_tier"].eq("clinically_validated").sum())
                if not candidate_evidence.empty
                else 0
            ),
            "diagnostic_accuracy_rows": (
                int((algorithm["claim_ceiling"] == "diagnostic_accuracy").sum())
                if not algorithm.empty
                else 0
            ),
            "missing_inputs": len(missing),
        },
        notes=(
            f"missing_inputs={','.join(missing) if missing else 'none'}",
            f"minimum_icc={policy.minimum_icc}",
            (
                "mandatory_context="
                + ",".join(field for field, _ in MANDATORY_CONTEXT_FIELDS)
            ),
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "MANDATORY_CONTEXT_FIELDS",
    "MINIMUM_ESTIMABLE_STRATUM_SHARE",
    "MINIMUM_REPORTABLE_ICC",
    "REPORTING_GUIDELINE_VERSION",
    "ReportingSetPolicy",
    "build_reporting_algorithm",
    "build_reporting_set",
    "intraindividual_change_note",
    "mandatory_context_table",
    "run_reporting_guideline",
]
