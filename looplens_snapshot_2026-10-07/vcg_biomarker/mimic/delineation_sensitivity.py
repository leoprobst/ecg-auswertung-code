"""Comparator sensitivity against this project's own delineation bias.

The classical comparator criteria apply absolute clinical cut-offs (QTc >480
and >500 ms, QRS >=120 ms, PR >200 ms, P >=120 ms) to intervals measured by
this project's own delineation. The external LUDB benchmark documents a
systematic T-offset bias of about -20 ms and a P-onset bias of about +15 ms,
so those cut-offs do not fire where a textbook reading would place them. That
matters because the project's central claim is the *incremental* value of the
VCG markers over exactly these criteria: a comparator biased downward makes
the increment look larger than it is.

This module therefore recomputes the affected criteria from the recorder's own
global fiducials for the identical waveform and reports, per endpoint:

* how far the two boundary estimates differ (bias, MAE, limits of agreement);
* how far the two versions of the rule disagree about who is positive;
* the diagnostic performance of both versions of the comparator.

This is a sensitivity analysis, not a correction. The recorder's algorithm is
another imperfect estimator, not a reference standard, so neither version is
declared "true"; the point is whether a conclusion survives the swap. Nothing
here rescales, calibrates or offsets the project's own measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .statistics import classification_metrics


DELINEATION_SENSITIVITY_VERSION = "1.0.0"

# Agreement below this is flagged: the two boundary estimates then disagree so
# often that any incremental-value statement against the own-delineation rule
# needs the machine-rule number reported next to it.
MINIMUM_ROBUST_RULE_AGREEMENT = 0.90


@dataclass(frozen=True)
class IntervalCriterionSensitivity:
    """One absolute-threshold criterion and its recorder-side counterpart."""

    criterion_id: str
    own_value_column: str
    machine_value_kind: str
    threshold_ms: float
    inclusive: bool
    interval_label: str
    boundary_dependency: str


INTERVAL_CRITERION_SENSITIVITIES: tuple[IntervalCriterionSensitivity, ...] = (
    IntervalCriterionSensitivity(
        criterion_id="qtc_gt_480",
        own_value_column="classical_qtc_gt_480",
        machine_value_kind="qtc_fridericia",
        threshold_ms=480.0,
        inclusive=False,
        interval_label="QTc Fridericia",
        boundary_dependency="t_offset",
    ),
    IntervalCriterionSensitivity(
        criterion_id="qtc_gt_500",
        own_value_column="classical_qtc_gt_500",
        machine_value_kind="qtc_fridericia",
        threshold_ms=500.0,
        inclusive=False,
        interval_label="QTc Fridericia",
        boundary_dependency="t_offset",
    ),
    IntervalCriterionSensitivity(
        criterion_id="qrs_widening_ge_120",
        own_value_column="classical_qrs_widening_ge_120",
        machine_value_kind="qrs_duration",
        threshold_ms=120.0,
        inclusive=True,
        interval_label="QRS-Dauer",
        boundary_dependency="qrs_onset_and_offset",
    ),
    IntervalCriterionSensitivity(
        criterion_id="pr_prolongation_gt_200",
        own_value_column="classical_pr_prolongation_gt_200",
        machine_value_kind="pr_interval",
        threshold_ms=200.0,
        inclusive=False,
        interval_label="PR-Intervall",
        boundary_dependency="p_onset_and_qrs_onset",
    ),
    IntervalCriterionSensitivity(
        criterion_id="p_duration_ge_120",
        own_value_column="classical_p_duration_ge_120",
        machine_value_kind="p_duration",
        threshold_ms=120.0,
        inclusive=True,
        interval_label="P-Dauer",
        boundary_dependency="p_onset_and_p_offset",
    ),
)


def machine_interval_values(
    frame: pd.DataFrame,
    kind: str,
) -> np.ndarray:
    """Recorder-side interval in milliseconds, NaN where unavailable."""

    def column(name: str) -> pd.Series:
        return pd.to_numeric(
            frame.get(name, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )

    if kind == "qtc_fridericia":
        qt = column("machine_qt_interval_ms")
        rate = column("machine_heart_rate_bpm")
        usable = rate.between(20, 300, inclusive="both") & qt.notna()
        rr_seconds = 60.0 / rate.where(usable)
        return (qt.where(usable) / np.cbrt(rr_seconds)).to_numpy(dtype=float)
    if kind == "qrs_duration":
        direct = column("machine_qrs_duration_ms")
        derived = column("machine_qrs_end_ms") - column("machine_qrs_onset_ms")
        return direct.combine_first(derived).to_numpy(dtype=float)
    if kind == "pr_interval":
        direct = column("machine_pr_interval_ms")
        derived = column("machine_qrs_onset_ms") - column("machine_p_onset_ms")
        return direct.combine_first(derived).to_numpy(dtype=float)
    if kind == "p_duration":
        direct = column("machine_p_duration_ms")
        derived = column("machine_p_end_ms") - column("machine_p_onset_ms")
        return direct.combine_first(derived).to_numpy(dtype=float)
    raise ValueError(f"Unknown machine interval kind: {kind}")


DEVICE_PRIMARY_COMPARATOR_VERSION = "1.0.0"
OWN_COMPARATOR_PREFIX = "own_"


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(
        frame.get(name, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)


def apply_device_interval_comparators(frame: pd.DataFrame) -> pd.DataFrame:
    """Comparator criteria on the recorder's intervals (audit S2).

    The literature thresholds (QTc >480/500, QRS >=120, PR >200, P >=120,
    Cornell product >244 mV*ms) were set on device or manual measurements.
    This project's own boundaries sit systematically elsewhere (LUDB: P
    duration about -25 ms, QT about -25 ms; QRS +16 ms against the device),
    which shifted the comparators the VCG markers are claimed to beat. The
    comparator columns now carry the device version wherever the recorder
    measured the interval and the own version only where it did not
    (``*_interval_source`` says which). The own columns stay available as
    ``own_*`` for the sensitivity analysis, which is thereby reversed:
    device primary, own delineation secondary. Marker panels are untouched.
    """

    output = frame.copy()
    added: dict[str, object] = {}

    def replace(column: str, device_values: np.ndarray, device_positive, source_extra=None):
        own_values = _numeric_column(frame, column)
        own_positive = _numeric_column(frame, f"{column}_positive")
        use_device = np.isfinite(device_values)
        added[f"{OWN_COMPARATOR_PREFIX}{column}"] = own_values
        added[f"{OWN_COMPARATOR_PREFIX}{column}_positive"] = own_positive
        added[column] = np.where(use_device, device_values, own_values)
        if device_positive is not None:
            added[f"{column}_positive"] = np.where(
                use_device, device_positive, own_positive
            )
        added[f"{column}_interval_source"] = np.where(
            use_device,
            "device",
            np.where(np.isfinite(own_values), "own_delineation", None),
        )

    for definition in INTERVAL_CRITERION_SENSITIVITIES:
        device = machine_interval_values(frame, definition.machine_value_kind)
        with np.errstate(invalid="ignore"):
            positive = np.where(
                np.isfinite(device),
                _positive(device, definition.threshold_ms, definition.inclusive),
                np.nan,
            ).astype(float)
        replace(definition.own_value_column, device, positive)

    # Continuous QTc comparators from the recorder's QT and rate.
    qt = _numeric_column(frame, "machine_qt_interval_ms")
    rate = _numeric_column(frame, "machine_heart_rate_bpm")
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = np.isfinite(qt) & (rate >= 20) & (rate <= 300)
        rr = np.where(usable, 60.0 / rate, np.nan)
        corrected = {
            "classical_qtc_bazett": qt / np.sqrt(rr),
            "classical_qtc_fridericia": qt / np.cbrt(rr),
            "classical_qtc_framingham": qt + 154.0 * (1.0 - rr),
        }
    for column, values in corrected.items():
        if column in frame:
            replace(column, np.where(usable, values, np.nan), None)

    # Cornell product: own voltage (sex-adjusted) times the device QRS.
    if "classical_cornell_product" in frame:
        own_product = _numeric_column(frame, "classical_cornell_product")
        # The own QRS duration, before the device replacement above.
        own_qrs = _numeric_column(frame, "classical_qrs_widening_ge_120")
        device_qrs = machine_interval_values(frame, "qrs_duration")
        own_positive = _numeric_column(frame, "classical_cornell_product_positive")
        with np.errstate(invalid="ignore", divide="ignore"):
            voltage = np.where(own_qrs > 0, own_product / own_qrs, np.nan)
            product = voltage * device_qrs
            # Where the own rule could not be decided (sex unknown), the
            # device version is not decided either.
            positive = np.where(
                np.isfinite(product) & np.isfinite(own_positive),
                (product > 244.0).astype(float),
                np.nan,
            )
        replace("classical_cornell_product", product, positive)

    for column, values in added.items():
        output[column] = values
    output.attrs["comparator_interval_policy"] = (
        f"device_primary_v{DEVICE_PRIMARY_COMPARATOR_VERSION}"
    )
    return output


def _positive(values: np.ndarray, threshold: float, inclusive: bool) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return values >= threshold if inclusive else values > threshold


def _agreement_statistics(
    own_positive: np.ndarray,
    machine_positive: np.ndarray,
) -> dict[str, float]:
    total = int(len(own_positive))
    if total == 0:
        return {
            "rule_agreement": float("nan"),
            "cohen_kappa": float("nan"),
            "own_only_positive": 0,
            "machine_only_positive": 0,
        }
    agree = float(np.mean(own_positive == machine_positive))
    own_rate = float(np.mean(own_positive))
    machine_rate = float(np.mean(machine_positive))
    expected = own_rate * machine_rate + (1.0 - own_rate) * (1.0 - machine_rate)
    kappa = (
        (agree - expected) / (1.0 - expected)
        if 1.0 - expected > 1e-12
        else float("nan")
    )
    return {
        "rule_agreement": agree,
        "cohen_kappa": float(kappa),
        "own_only_positive": int(np.sum(own_positive & ~machine_positive)),
        "machine_only_positive": int(np.sum(~own_positive & machine_positive)),
    }


def _interval_difference_statistics(
    own_values: np.ndarray,
    machine_values: np.ndarray,
) -> dict[str, float]:
    difference = own_values - machine_values
    if not np.isfinite(difference).any():
        return {
            "interval_bias_ms": float("nan"),
            "interval_mae_ms": float("nan"),
            "interval_sd_ms": float("nan"),
            "interval_loa_low_ms": float("nan"),
            "interval_loa_high_ms": float("nan"),
        }
    bias = float(np.nanmean(difference))
    sd = float(np.nanstd(difference, ddof=1)) if len(difference) > 1 else float("nan")
    return {
        "interval_bias_ms": bias,
        "interval_mae_ms": float(np.nanmean(np.abs(difference))),
        "interval_sd_ms": sd,
        "interval_loa_low_ms": bias - 1.96 * sd,
        "interval_loa_high_ms": bias + 1.96 * sd,
    }


def evaluate_interval_criterion_sensitivity(
    frame: pd.DataFrame,
    *,
    endpoint_id: str,
    evaluation_split: str,
    definitions: Iterable[IntervalCriterionSensitivity] = (
        INTERVAL_CRITERION_SENSITIVITIES
    ),
    labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare own-delineation and recorder-side versions of each criterion.

    ``frame`` must carry the classical criterion value columns and the
    ``machine_*`` interval columns for the same rows. Rows where either side
    is unmeasurable are excluded from that criterion only, and the exclusion
    count is reported rather than silently absorbed.
    """

    label_values = (
        np.asarray(labels, dtype=float)
        if labels is not None
        else pd.to_numeric(
            frame.get("label", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
    )
    records: list[dict] = []
    for definition in definitions:
        # After ``apply_device_interval_comparators`` the comparator columns
        # hold the device version; the own measurement moved to ``own_*``.
        own_column = (
            f"{OWN_COMPARATOR_PREFIX}{definition.own_value_column}"
            if f"{OWN_COMPARATOR_PREFIX}{definition.own_value_column}" in frame
            else definition.own_value_column
        )
        own_values = pd.to_numeric(
            frame.get(
                own_column,
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).to_numpy(dtype=float)
        machine_values = machine_interval_values(
            frame,
            definition.machine_value_kind,
        )
        both = np.isfinite(own_values) & np.isfinite(machine_values)
        record: dict[str, object] = {
            "endpoint_id": endpoint_id,
            "evaluation_split": evaluation_split,
            "criterion_id": definition.criterion_id,
            "interval_label": definition.interval_label,
            "boundary_dependency": definition.boundary_dependency,
            "threshold_ms": float(definition.threshold_ms),
            "threshold_inclusive": bool(definition.inclusive),
            "n_rows": int(len(frame)),
            "n_own_measurable": int(np.sum(np.isfinite(own_values))),
            "n_machine_measurable": int(np.sum(np.isfinite(machine_values))),
            "n_both_measurable": int(np.sum(both)),
            "comparison_basis": "paired_same_waveform_two_estimators",
            "primary_comparator_intervals": (
                "device" if own_column != definition.own_value_column
                else "own_delineation"
            ),
            "reference_claim": "neither_estimator_is_a_reference_standard",
            "sensitivity_version": DELINEATION_SENSITIVITY_VERSION,
        }
        if not both.any():
            record["sensitivity_status"] = "not_evaluable"
            record["rule_agreement"] = float("nan")
            record["comparator_conclusion_robust"] = False
            records.append(record)
            continue
        own_scoped = own_values[both]
        machine_scoped = machine_values[both]
        own_positive = _positive(
            own_scoped,
            definition.threshold_ms,
            definition.inclusive,
        )
        machine_positive = _positive(
            machine_scoped,
            definition.threshold_ms,
            definition.inclusive,
        )
        record.update(_interval_difference_statistics(own_scoped, machine_scoped))
        record.update(_agreement_statistics(own_positive, machine_positive))
        record["own_positive_rate"] = float(np.mean(own_positive))
        record["machine_positive_rate"] = float(np.mean(machine_positive))
        record["positive_rate_difference"] = (
            record["own_positive_rate"] - record["machine_positive_rate"]
        )
        scoped_labels = label_values[both]
        assessable = np.isfinite(scoped_labels) & np.isin(scoped_labels, (0.0, 1.0))
        if assessable.any() and np.unique(scoped_labels[assessable]).size == 2:
            binary = scoped_labels[assessable].astype(np.int8)
            own_metrics = classification_metrics(binary, own_positive[assessable])
            machine_metrics = classification_metrics(
                binary,
                machine_positive[assessable],
            )
            for prefix, metrics in (
                ("own", own_metrics),
                ("machine", machine_metrics),
            ):
                for key in ("sensitivity", "specificity", "ppv", "npv"):
                    value = metrics.get(key)
                    record[f"{prefix}_rule_{key}"] = (
                        float(value) if value is not None else float("nan")
                    )
            record["sensitivity_difference"] = (
                record["own_rule_sensitivity"] - record["machine_rule_sensitivity"]
            )
            record["specificity_difference"] = (
                record["own_rule_specificity"] - record["machine_rule_specificity"]
            )
            record["n_labelled"] = int(np.sum(assessable))
            record["sensitivity_status"] = "evaluable"
        else:
            record["sensitivity_status"] = "agreement_only_no_usable_labels"
            record["n_labelled"] = int(np.sum(assessable))
        agreement = float(record.get("rule_agreement", float("nan")))
        record["comparator_conclusion_robust"] = bool(
            np.isfinite(agreement) and agreement >= MINIMUM_ROBUST_RULE_AGREEMENT
        )
        record["comparator_bias_direction"] = _bias_direction(
            float(record.get("positive_rate_difference", float("nan")))
        )
        records.append(record)
    return pd.DataFrame(records)


def _bias_direction(positive_rate_difference: float) -> str:
    if not np.isfinite(positive_rate_difference):
        return "unknown"
    if abs(positive_rate_difference) < 0.005:
        return "no_material_difference"
    return (
        "own_rule_fires_more_often"
        if positive_rate_difference > 0
        else "own_rule_fires_less_often"
    )


def summarize_comparator_sensitivity(
    frame: pd.DataFrame,
) -> Mapping[str, object]:
    """One-line verdict per criterion for the report layer."""

    if frame.empty:
        return {
            "criteria_evaluated": 0,
            "criteria_robust": 0,
            "criteria_not_robust": [],
            "sensitivity_version": DELINEATION_SENSITIVITY_VERSION,
        }
    evaluable = frame.loc[frame["sensitivity_status"].ne("not_evaluable")]
    not_robust = evaluable.loc[
        ~evaluable["comparator_conclusion_robust"].astype(bool)
    ]
    return {
        "criteria_evaluated": int(len(evaluable)),
        "criteria_robust": int(
            evaluable["comparator_conclusion_robust"].astype(bool).sum()
        ),
        "criteria_not_robust": sorted(
            set(not_robust["criterion_id"].astype(str))
        ),
        "minimum_rule_agreement": MINIMUM_ROBUST_RULE_AGREEMENT,
        "sensitivity_version": DELINEATION_SENSITIVITY_VERSION,
    }


__all__ = [
    "DELINEATION_SENSITIVITY_VERSION",
    "INTERVAL_CRITERION_SENSITIVITIES",
    "IntervalCriterionSensitivity",
    "MINIMUM_ROBUST_RULE_AGREEMENT",
    "evaluate_interval_criterion_sensitivity",
    "machine_interval_values",
    "apply_device_interval_comparators",
    "summarize_comparator_sensitivity",
]
