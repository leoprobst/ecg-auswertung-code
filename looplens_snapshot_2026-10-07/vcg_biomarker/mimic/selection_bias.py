"""Why did this patient get an ECG at all — and what that does to the results.

Nobody in MIMIC-IV is recorded at random. Roughly 55 % of the ECG cohort
overlaps an inpatient admission and about 25 % an emergency-department visit,
which means the recorded population is selected on being unwell enough to be
investigated. Two distinct problems follow:

* **Selection bias.** Patients with an ECG differ from those without. Anything
  estimated in the recorded population describes the recorded population.
* **Verification bias.** Among the recorded, only some get the reference
  standard — an echo, a troponin. Whether they do depends on suspicion, and
  suspicion depends on the ECG. Conditioning accuracy on having been verified
  therefore inflates sensitivity and deflates specificity in a way no amount
  of data fixes.

This module does not remove either bias. It measures them:

* a coverage census comparing recorded and unrecorded patients;
* inverse-probability-of-observation weights, with the diagnostics needed to
  judge whether they are trustworthy (overlap, extreme weights, effective
  sample size);
* a verification-bias sensitivity that brackets accuracy under assumptions
  about the unverified;
* holdout splits by cart, care setting, quality and acquisition era, so that a
  result which only holds on one device or one ward is visible as such;
* prespecified positive and negative control associations.

Negative controls are the sharpest instrument here: an association that should
not exist, appearing anyway, quantifies the residual confounding and analysis
artefact that the study design failed to remove (Lipsitch et al., Epidemiology
2010).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SELECTION_BIAS_VERSION = "1.2.0"

#: Share of the reference-blind universe that must carry a known reference
#: before a diagnostic-accuracy claim may be confirmed. Declared here rather
#: than beside the validation stage because it is a property of the study
#: design, and because the label stage needs it to say per endpoint whether
#: such a claim is reachable at all. Measured on the 63 endpoints of the last
#: development run, six reach it.
MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION = 0.80

# Weight diagnostics below/above which an IPW analysis should not be trusted.
MINIMUM_OVERLAP_PROPENSITY = 0.01
MAXIMUM_STABILISED_WEIGHT = 20.0


@dataclass(frozen=True)
class ControlAssociation:
    """A prespecified association whose expected direction is known."""

    control_id: str
    kind: str  # "positive" or "negative"
    target_id: str
    rationale: str
    expected_direction: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("positive", "negative"):
            raise ValueError("Control kind must be 'positive' or 'negative'")
        if self.kind == "negative" and self.expected_direction != 0:
            raise ValueError(
                "A negative control expects no association by definition"
            )


# Prespecified before the screen, so a failure cannot be reinterpreted after
# the fact. Positive controls must reproduce; negative controls must not.
DEFAULT_CONTROL_ASSOCIATIONS: tuple[ControlAssociation, ...] = (
    ControlAssociation(
        "positive_lvh_voltage",
        "positive",
        "echo_lvh_strict",
        "Voltage criteria for LVH are established since Sokolow-Lyon (1949) "
        "and Casale (1987); a pipeline that cannot recover them is broken "
        "before any novel marker is considered.",
        expected_direction=1,
    ),
    ControlAssociation(
        "positive_hyperkalemia_t_wave",
        "positive",
        "hyperkalemia_near_ecg",
        "Peaked T waves in hyperkalemia are a textbook ECG effect with a "
        "direct ionic mechanism.",
        expected_direction=1,
    ),
    ControlAssociation(
        "positive_qt_hypokalemia",
        "positive",
        "hypokalemia_near_ecg",
        "Hypokalemia lengthens repolarization; an established drug-safety "
        "relationship.",
        expected_direction=1,
    ),
    ControlAssociation(
        "negative_marital_status",
        "negative",
        "context_marital_status",
        "Marital status has no plausible pathway to ventricular "
        "repolarization geometry. An association indicates residual "
        "confounding by care setting or socioeconomic pathway, not biology.",
    ),
    ControlAssociation(
        "negative_admission_weekday",
        "negative",
        "context_admission_weekday",
        "The day of the week carries no cardiac electrophysiology. A signal "
        "here means the analysis is picking up staffing or workflow patterns.",
    ),
    ControlAssociation(
        "negative_insurance_type",
        "negative",
        "context_insurance",
        "Insurance type cannot alter the QRS loop. Any association measures "
        "the socioeconomic-to-care-intensity pathway that also confounds real "
        "endpoints.",
    ),
    ControlAssociation(
        "negative_record_ordinal",
        "negative",
        "context_ecg_sequence_number",
        "Which number in sequence a recording happens to be is an artefact of "
        "the observation process; a marker that tracks it is tracking "
        "observation intensity.",
    ),
)


def coverage_census(
    population: pd.DataFrame,
    *,
    recorded_column: str = "has_ecg",
    covariate_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compare recorded and unrecorded patients on every covariate.

    Standardised differences rather than p-values: with hundreds of thousands
    of patients every difference is significant, and the question is how large
    it is.
    """

    if recorded_column not in population.columns:
        raise ValueError(f"Missing recorded column: {recorded_column}")
    recorded = population[recorded_column].astype(bool)
    rows: list[dict[str, object]] = []
    for column in covariate_columns:
        if column not in population.columns:
            continue
        values = pd.to_numeric(population[column], errors="coerce")
        present = values.loc[recorded.to_numpy()]
        absent = values.loc[~recorded.to_numpy()]
        mean_present = float(present.mean())
        mean_absent = float(absent.mean())
        var_present = float(present.var(ddof=1))
        var_absent = float(absent.var(ddof=1))
        pooled = np.sqrt((var_present + var_absent) / 2.0)
        standardised = (
            float((mean_present - mean_absent) / pooled)
            if np.isfinite(pooled) and pooled > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "covariate": column,
                "n_recorded": int(present.notna().sum()),
                "n_unrecorded": int(absent.notna().sum()),
                "mean_recorded": mean_present,
                "mean_unrecorded": mean_absent,
                "standardised_difference": standardised,
                "materially_different": bool(
                    np.isfinite(standardised) and abs(standardised) >= 0.1
                ),
                "selection_bias_version": SELECTION_BIAS_VERSION,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "covariate",
                "n_recorded",
                "n_unrecorded",
                "mean_recorded",
                "mean_unrecorded",
                "standardised_difference",
                "materially_different",
                "selection_bias_version",
            ]
        )
    return pd.DataFrame(rows)


def observation_propensity_weights(
    observed: Iterable[bool],
    covariates: pd.DataFrame,
    *,
    minimum_propensity: float = MINIMUM_OVERLAP_PROPENSITY,
    maximum_weight: float = MAXIMUM_STABILISED_WEIGHT,
) -> dict[str, object]:
    """Stabilised inverse-probability-of-observation weights, with diagnostics.

    A logistic propensity for being recorded, fitted on the covariates, then
    stabilised by the marginal rate. The diagnostics matter more than the
    weights: without common support the weighted estimate extrapolates into a
    region where no comparable unrecorded patient exists, and reports a number
    anyway.
    """

    from statsmodels.discrete.discrete_model import Logit

    outcome = np.asarray(list(observed), dtype=float)
    design = covariates.apply(pd.to_numeric, errors="coerce")
    complete = design.notna().all(axis=1).to_numpy() & np.isfinite(outcome)
    result: dict[str, object] = {
        "status": "not_evaluable",
        "weights": np.full(len(outcome), np.nan, dtype=float),
        "n_complete": int(complete.sum()),
        "selection_bias_version": SELECTION_BIAS_VERSION,
    }
    if complete.sum() < 50 or np.unique(outcome[complete]).size < 2:
        result["failure_reason"] = "insufficient_or_single_class"
        return result
    matrix = np.column_stack(
        (np.ones(int(complete.sum())), design.loc[complete].to_numpy())
    )
    try:
        model = Logit(outcome[complete], matrix).fit(disp=0)
        propensity = np.asarray(model.predict(matrix), dtype=float)
    except (ValueError, np.linalg.LinAlgError, Exception) as exc:  # noqa: BLE001
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result
    marginal = float(np.mean(outcome[complete]))
    clipped = np.clip(propensity, minimum_propensity, 1.0 - minimum_propensity)
    stabilised = np.where(
        outcome[complete] > 0.5,
        marginal / clipped,
        (1.0 - marginal) / (1.0 - clipped),
    )
    truncated = np.clip(stabilised, 0.0, float(maximum_weight))
    weights = np.full(len(outcome), np.nan, dtype=float)
    weights[complete] = truncated
    overlap_low = float(np.mean(propensity < minimum_propensity))
    overlap_high = float(np.mean(propensity > 1.0 - minimum_propensity))
    effective = (
        float(np.sum(truncated) ** 2 / np.sum(truncated ** 2))
        if np.sum(truncated ** 2) > 0
        else 0.0
    )
    result.update(
        {
            "status": "evaluable",
            "failure_reason": "",
            "weights": weights,
            "marginal_observation_rate": marginal,
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "share_below_overlap_floor": overlap_low,
            "share_above_overlap_ceiling": overlap_high,
            "weight_max": float(np.max(truncated)),
            "weights_truncated": bool(np.any(stabilised > maximum_weight)),
            "effective_sample_size": effective,
            "effective_sample_fraction": (
                effective / float(complete.sum()) if complete.sum() else 0.0
            ),
            "common_support_adequate": bool(
                overlap_low + overlap_high < 0.05
                and np.max(truncated) <= maximum_weight
            ),
            "interpretation": (
                "reweights the recorded population towards the full "
                "population under the assumption that observation is random "
                "given these covariates; that assumption is untestable"
            ),
        }
    )
    return result


def verification_bias_bounds(
    verified_positive: int,
    verified_negative: int,
    unverified: int,
    *,
    marker_positive_verified: int,
    marker_positive_unverified: int,
    marker_positive_verified_negative: Optional[int] = None,
    marker_missing_verified_positive: int = 0,
    marker_missing_verified_negative: int = 0,
    marker_missing_unverified: int = 0,
) -> dict[str, object]:
    """Sharp fixed-threshold bounds over verification and marker missingness.

    No missing-at-random assumption is made.  Every reference-unverified
    patient may take either outcome, while every missing marker may take
    either test result.  The extrema below are attainable jointly.  Passing
    zero missing counts recovers the historical verification-only formula.
    """

    verified = int(verified_positive) + int(verified_negative)
    total = verified + int(unverified)
    if verified <= 0 or total <= 0:
        return {
            "status": "not_evaluable",
            "verification_rate": float("nan"),
            "sensitivity_verified_only": float("nan"),
            "sensitivity_lower_bound": float("nan"),
            "sensitivity_upper_bound": float("nan"),
        }
    true_positive = int(marker_positive_verified)
    false_positive = (
        int(marker_positive_verified_negative)
        if marker_positive_verified_negative is not None
        else None
    )
    missing_verified_positive = int(marker_missing_verified_positive)
    missing_verified_negative = int(marker_missing_verified_negative)
    missing_unverified = int(marker_missing_unverified)
    unverified_marker_positive = int(marker_positive_unverified)
    unverified_marker_negative = (
        int(unverified) - unverified_marker_positive - missing_unverified
    )
    if (
        min(
            int(verified_positive),
            int(verified_negative),
            int(unverified),
            true_positive,
            unverified_marker_positive,
            unverified_marker_negative,
            missing_verified_positive,
            missing_verified_negative,
            missing_unverified,
        )
        < 0
        or true_positive + missing_verified_positive
        > int(verified_positive)
        or (
            false_positive is not None
            and (
                false_positive + missing_verified_negative
                > int(verified_negative)
                or false_positive < 0
            )
        )
    ):
        raise ValueError("Verification-bias counts are internally inconsistent")
    observed_verified_positive = (
        int(verified_positive) - missing_verified_positive
    )
    observed_verified_negative = (
        int(verified_negative) - missing_verified_negative
    )
    sensitivity_verified = (
        float(true_positive / observed_verified_positive)
        if observed_verified_positive > 0
        else float("nan")
    )
    # Sensitivity is lowest when every unverified marker-negative or missing
    # patient is diseased and every missing known-positive tests negative.
    worst_positive = (
        int(verified_positive)
        + unverified_marker_negative
        + missing_unverified
    )
    sensitivity_lower = (
        float(true_positive / worst_positive)
        if worst_positive > 0
        else float("nan")
    )
    # The upper extreme assigns every missing known-positive and every
    # unverified marker-positive/missing patient a true-positive result.
    best_numerator = (
        true_positive
        + missing_verified_positive
        + unverified_marker_positive
        + missing_unverified
    )
    best_positive = (
        int(verified_positive)
        + unverified_marker_positive
        + missing_unverified
    )
    sensitivity_upper = (
        float(best_numerator / best_positive)
        if best_positive > 0
        else float("nan")
    )
    specificity_verified = float("nan")
    specificity_lower = float("nan")
    specificity_upper = float("nan")
    ppv_lower = float("nan")
    ppv_upper = float("nan")
    npv_lower = float("nan")
    npv_upper = float("nan")
    if false_positive is not None:
        true_negative = observed_verified_negative - false_positive
        false_negative = observed_verified_positive - true_positive
        specificity_verified = (
            float(true_negative / observed_verified_negative)
            if observed_verified_negative > 0
            else float("nan")
        )
        worst_negative = (
            int(verified_negative)
            + unverified_marker_positive
            + missing_unverified
        )
        specificity_lower = (
            float(true_negative / worst_negative)
            if worst_negative > 0
            else float("nan")
        )
        best_true_negative = (
            true_negative
            + missing_verified_negative
            + unverified_marker_negative
            + missing_unverified
        )
        best_negative = (
            int(verified_negative)
            + unverified_marker_negative
            + missing_unverified
        )
        specificity_upper = (
            float(best_true_negative / best_negative)
            if best_negative > 0
            else float("nan")
        )
        positive_predictions_lower = (
            true_positive
            + false_positive
            + missing_verified_negative
            + unverified_marker_positive
            + missing_unverified
        )
        positive_predictions_upper = (
            true_positive
            + false_positive
            + missing_verified_positive
            + unverified_marker_positive
            + missing_unverified
        )
        negative_predictions_lower = (
            true_negative
            + false_negative
            + missing_verified_positive
            + unverified_marker_negative
            + missing_unverified
        )
        negative_predictions_upper = (
            true_negative
            + false_negative
            + missing_verified_negative
            + unverified_marker_negative
            + missing_unverified
        )
        ppv_lower = (
            float(true_positive / positive_predictions_lower)
            if positive_predictions_lower > 0
            else float("nan")
        )
        ppv_upper = (
            float(best_numerator / positive_predictions_upper)
            if positive_predictions_upper > 0
            else float("nan")
        )
        npv_lower = (
            float(true_negative / negative_predictions_lower)
            if negative_predictions_lower > 0
            else float("nan")
        )
        npv_upper = (
            float(best_true_negative / negative_predictions_upper)
            if negative_predictions_upper > 0
            else float("nan")
        )
    sensitivity_width = (
        float(sensitivity_upper - sensitivity_lower)
        if np.isfinite(sensitivity_upper) and np.isfinite(sensitivity_lower)
        else float("nan")
    )
    specificity_width = (
        float(specificity_upper - specificity_lower)
        if np.isfinite(specificity_upper) and np.isfinite(specificity_lower)
        else float("nan")
    )
    return {
        "status": "evaluable",
        "verification_rate": float(verified / total),
        "n_verified": verified,
        "n_unverified": int(unverified),
        "sensitivity_verified_only": sensitivity_verified,
        "sensitivity_lower_bound": sensitivity_lower,
        "sensitivity_upper_bound": sensitivity_upper,
        "sensitivity_bound_width": sensitivity_width,
        "specificity_verified_only": specificity_verified,
        "specificity_lower_bound": specificity_lower,
        "specificity_upper_bound": specificity_upper,
        "specificity_bound_width": specificity_width,
        "ppv_lower_bound": ppv_lower,
        "ppv_upper_bound": ppv_upper,
        "npv_lower_bound": npv_lower,
        "npv_upper_bound": npv_upper,
        # Compatibility alias retained for existing reports.
        "bound_width": sensitivity_width,
        "n_unverified_marker_positive": unverified_marker_positive,
        "n_unverified_marker_negative": unverified_marker_negative,
        "n_marker_missing_verified_positive": missing_verified_positive,
        "n_marker_missing_verified_negative": missing_verified_negative,
        "n_marker_missing_unverified": missing_unverified,
        "verified_only_estimate_trustworthy": bool(
            verified / total >= 0.8
        ),
        "assumption": (
            "no_missing_at_random_assumption_joint_verification_and_marker_"
            "missingness_extreme_case_bounds"
        ),
        "selection_bias_version": SELECTION_BIAS_VERSION,
    }


def endpoint_verification_reachability(
    label_summary: pd.DataFrame,
    *,
    minimum_verification_rate: float,
) -> pd.DataFrame:
    """Say per endpoint whether an accuracy claim is reachable at all.

    The confirmation gate requires that at least ``minimum_verification_rate``
    of the reference-blind universe carries a known reference before a
    diagnostic-accuracy claim may be confirmed. That is the right question, and
    for most of this project's endpoints the answer is fixed in advance by how
    hospitals work rather than by anything the analysis does.

    Measured on the 63 endpoints of the last development run, at the declared
    0.80: **six reach it.** Those six are mortality, the three new-onset atrial
    fibrillation horizons -- all four verified by construction, since no test is
    needed to know them -- and the two QTc endpoints, which are read off the
    recorder's own measurement and are circular for a different reason. Every
    echocardiographic endpoint sits between 0.003 and 0.02, because an
    echocardiogram is done on a few per cent of the recordings.

    So no structural endpoint in this study can currently be confirmed, and
    that follows from the design rather than from the data. Stating it per
    endpoint is the point of this table: a reader can then see that "not
    confirmed" means "the reference was never obtained for the other 98 %", not
    "the marker failed".

    The table does not decide what to do about it. Two answers are defensible
    -- narrowing the estimand to the verified population, where the rate is one
    by construction and transportability is carried separately by the Manski
    bounds, or leaving the gate and reporting structural endpoints as
    descriptive -- and the choice is a study-design decision, not a threshold to
    be lowered until things pass.
    """

    columns = [
        "label_id",
        "verified_ecg_count",
        "unavailable_ecg_count",
        "verification_rate",
        "minimum_verification_rate",
        "prevalence_status",
        "diagnostic_accuracy_claim_reachable",
        "unreachable_reason",
        "selection_bias_version",
    ]
    if label_summary.empty:
        return pd.DataFrame(columns=columns)

    positive = pd.to_numeric(
        label_summary.get("positive_ecg_count", 0), errors="coerce"
    ).fillna(0)
    negative = pd.to_numeric(
        label_summary.get("negative_ecg_count", 0), errors="coerce"
    ).fillna(0)
    unavailable = pd.to_numeric(
        label_summary.get("unavailable_ecg_count", label_summary.get(
            "unavailable_count", 0
        )),
        errors="coerce",
    ).fillna(0)
    verified = positive + negative
    universe = verified + unavailable
    rate = (verified / universe.replace(0, np.nan)).astype(float)
    prevalence_status = label_summary.get(
        "prevalence_status", pd.Series("unknown", index=label_summary.index)
    ).astype(str)

    rows = []
    for index, label_id in enumerate(label_summary["label_id"].astype(str)):
        observed_rate = float(rate.iloc[index])
        status = str(prevalence_status.iloc[index])
        reasons = []
        if not np.isfinite(observed_rate):
            reasons.append("no_reference_blind_universe")
        elif observed_rate < float(minimum_verification_rate):
            reasons.append("verification_rate_below_confirmation_threshold")
        if status == "high_prevalence_verification_conditioned":
            reasons.append("reference_obtained_mainly_on_suspicion")
        rows.append(
            {
                "label_id": label_id,
                "verified_ecg_count": int(verified.iloc[index]),
                "unavailable_ecg_count": int(unavailable.iloc[index]),
                "verification_rate": observed_rate,
                "minimum_verification_rate": float(minimum_verification_rate),
                "prevalence_status": status,
                "diagnostic_accuracy_claim_reachable": not reasons,
                "unreachable_reason": "|".join(reasons),
                "selection_bias_version": SELECTION_BIAS_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def holdout_strata(
    frame: pd.DataFrame,
    *,
    cart_column: str = "machine_cart_id",
    setting_column: str = "care_setting",
    quality_column: str = "signal_quality_tier",
    era_column: str = "anchor_year_group",
    minimum_stratum_size: int = 200,
) -> pd.DataFrame:
    """Enumerate device, setting, quality and era strata for holdout testing.

    A marker that only works on one recorder model, in one ward, at one signal
    quality, or in one era has not been shown to work. These strata are the
    cheapest available stand-in for external validation.
    """

    rows: list[dict[str, object]] = []
    for kind, column in (
        ("cart", cart_column),
        ("care_setting", setting_column),
        ("signal_quality", quality_column),
        ("acquisition_era", era_column),
    ):
        if column not in frame.columns:
            rows.append(
                {
                    "stratum_kind": kind,
                    "stratum_value": "",
                    "n_rows": 0,
                    "eligible_for_holdout": False,
                    "reason": "column_unavailable",
                }
            )
            continue
        counts = frame[column].astype("string").value_counts(dropna=True)
        for value, count in counts.items():
            rows.append(
                {
                    "stratum_kind": kind,
                    "stratum_value": str(value),
                    "n_rows": int(count),
                    "eligible_for_holdout": bool(
                        int(count) >= int(minimum_stratum_size)
                    ),
                    "reason": (
                        ""
                        if int(count) >= int(minimum_stratum_size)
                        else "stratum_too_small"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "stratum_kind",
                "stratum_value",
                "n_rows",
                "eligible_for_holdout",
                "reason",
            ]
        )
    output = pd.DataFrame(rows)
    output["selection_bias_version"] = SELECTION_BIAS_VERSION
    return output


def evaluate_control_associations(
    results: pd.DataFrame,
    *,
    controls: Iterable[ControlAssociation] = DEFAULT_CONTROL_ASSOCIATIONS,
    target_column: str = "target_id",
    effect_column: str = "effect_size",
    significant_column: str = "fdr_significant",
) -> pd.DataFrame:
    """Score the prespecified positive and negative controls.

    A failed positive control means the pipeline cannot see what is known to
    be there. A failed negative control means it sees what is known not to be
    there — and the size of that spurious effect is a direct estimate of the
    residual bias carried by every other result in the same screen.
    """

    rows: list[dict[str, object]] = []
    for control in controls:
        matches = (
            results.loc[
                results[target_column].astype("string").eq(control.target_id)
            ]
            if target_column in results.columns
            else results.head(0)
        )
        if matches.empty:
            rows.append(
                {
                    "control_id": control.control_id,
                    "kind": control.kind,
                    "target_id": control.target_id,
                    "status": "not_screened",
                    "passed": False,
                    "observed_effect": float("nan"),
                    "rationale": control.rationale,
                }
            )
            continue
        effect = pd.to_numeric(
            matches[effect_column],
            errors="coerce",
        ).abs().max()
        significant = (
            bool(matches[significant_column].astype(bool).any())
            if significant_column in matches.columns
            else False
        )
        if control.kind == "positive":
            passed = bool(significant)
            status = "recovered" if passed else "missed_known_association"
        else:
            passed = not significant
            status = "clean" if passed else "spurious_association_detected"
        rows.append(
            {
                "control_id": control.control_id,
                "kind": control.kind,
                "target_id": control.target_id,
                "status": status,
                "passed": passed,
                "observed_effect": float(effect) if pd.notna(effect) else float("nan"),
                "significant": significant,
                "rationale": control.rationale,
                "selection_bias_version": SELECTION_BIAS_VERSION,
            }
        )
    return pd.DataFrame(rows)


def summarize_control_associations(table: pd.DataFrame) -> Mapping[str, object]:
    """Whether the screen is fit to be believed at all."""

    if table.empty:
        return {
            "status": "not_evaluated",
            "positive_controls_recovered": 0,
            "negative_controls_clean": 0,
        }
    positives = table.loc[table["kind"].eq("positive")]
    negatives = table.loc[table["kind"].eq("negative")]
    failed_negative = negatives.loc[~negatives["passed"].astype(bool)]
    return {
        "status": "evaluated",
        "positive_controls_total": int(len(positives)),
        "positive_controls_recovered": int(positives["passed"].astype(bool).sum()),
        "negative_controls_total": int(len(negatives)),
        "negative_controls_clean": int(negatives["passed"].astype(bool).sum()),
        "residual_bias_indicated": bool(len(failed_negative) > 0),
        "failed_negative_controls": sorted(
            str(value) for value in failed_negative["control_id"]
        ),
        "interpretation": (
            "a failed negative control estimates the residual confounding "
            "and analysis artefact carried by every result in this screen"
        ),
        "selection_bias_version": SELECTION_BIAS_VERSION,
    }


__all__ = [
    "ControlAssociation",
    "DEFAULT_CONTROL_ASSOCIATIONS",
    "MAXIMUM_STABILISED_WEIGHT",
    "MINIMUM_OVERLAP_PROPENSITY",
    "SELECTION_BIAS_VERSION",
    "coverage_census",
    "evaluate_control_associations",
    "holdout_strata",
    "observation_propensity_weights",
    "summarize_control_associations",
    "verification_bias_bounds",
]
