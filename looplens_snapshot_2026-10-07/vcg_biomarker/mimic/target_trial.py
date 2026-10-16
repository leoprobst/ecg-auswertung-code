"""Target-trial protocol for the ECG effects of drugs.

The existing therapy analysis compares ECGs before and after a medication was
given. That is a real and useful quantity — a pharmacodynamic association —
and it is not a causal effect. Three things separate the two:

* **Common time zero.** Before/after comparisons start the clock at the drug,
  so patients enter the analysis at whatever point in their illness the drug
  happened to be given. A trial randomises at a defined moment; an emulation
  has to construct one.
* **Confounding by indication.** Amiodarone is given to people whose ECG is
  already abnormal. The pre-post change then mixes the drug effect with the
  natural course of the arrhythmia that prompted it. An active comparator —
  another drug used for the same indication — removes most of this, which no
  amount of covariate adjustment on a placebo-free cohort can.
* **Informative follow-up.** A post-drug ECG is only recorded when somebody
  wanted one, and wanting one depends on how the patient is doing. Analysing
  the pairs that exist conditions on that decision.

This module defines the protocol elements explicitly, checks an assembled
cohort against them, and refuses to label anything a causal effect until they
are all satisfied. It emulates a trial; it does not run one, and it says so.

Reference: Hernán & Robins, *Using Big Data to Emulate a Target Trial*,
American Journal of Epidemiology 2016.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


TARGET_TRIAL_VERSION = "1.0.0"

DESIGN_PRE_POST = "pre_post_pharmacodynamic_association"
DESIGN_NEW_USER_ACTIVE_COMPARATOR = "new_user_active_comparator_emulation"

CLAIM_ASSOCIATION = "pharmacodynamic_association_not_causal"
CLAIM_EMULATED_EFFECT = "emulated_causal_effect_under_stated_assumptions"


@dataclass(frozen=True)
class TargetTrialProtocol:
    """The seven protocol elements, fixed before the data are touched."""

    protocol_id: str
    eligibility: str
    treatment_strategies: tuple[str, ...]
    assignment_procedure: str
    follow_up_start: str
    outcome: str
    causal_contrast: str
    analysis_plan: str
    active_comparator: str = ""
    new_user_required: bool = True
    grace_period_hours: float = 24.0

    def __post_init__(self) -> None:
        if len(self.treatment_strategies) < 2:
            raise ValueError(
                "A target trial compares at least two strategies; a single "
                "arm cannot support a causal contrast"
            )
        if self.grace_period_hours < 0:
            raise ValueError("grace_period_hours must not be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "eligibility": self.eligibility,
            "treatment_strategies": list(self.treatment_strategies),
            "assignment_procedure": self.assignment_procedure,
            "follow_up_start": self.follow_up_start,
            "outcome": self.outcome,
            "causal_contrast": self.causal_contrast,
            "analysis_plan": self.analysis_plan,
            "active_comparator": self.active_comparator,
            "new_user_required": self.new_user_required,
            "grace_period_hours": self.grace_period_hours,
            "target_trial_version": TARGET_TRIAL_VERSION,
        }


# One worked protocol per drug class where an active comparator genuinely
# exists in MIMIC-IV. Classes without a plausible comparator are deliberately
# absent: for those, the honest analysis is the pre-post association.
DEFAULT_PROTOCOLS: tuple[TargetTrialProtocol, ...] = (
    TargetTrialProtocol(
        protocol_id="rate_control_beta_blocker_vs_ccb",
        eligibility=(
            "Erwachsene mit dokumentiertem Vorhofflimmern/-flattern und "
            "Kammerfrequenz > 100/min, ohne Betablocker- oder "
            "Calciumkanalblocker-Exposition in den vorangehenden 30 Tagen, "
            "mit einem auswertbaren EKG vor der ersten Gabe."
        ),
        treatment_strategies=(
            "Beginn eines Betablockers innerhalb der Karenzzeit",
            "Beginn eines Calciumkanalblockers innerhalb der Karenzzeit",
        ),
        assignment_procedure=(
            "Nicht randomisiert; Zuordnung wie beobachtet, mit "
            "Propensity-Gewichtung auf vorab festgelegte Confounder."
        ),
        follow_up_start=(
            "Zeitpunkt der ersten dokumentierten Gabe (gemeinsames Time-zero "
            "fuer beide Arme)"
        ),
        outcome="Aenderung der Kammerfrequenz und der Repolarisationsmarker im Folge-EKG",
        causal_contrast="Per-Protokoll-Effekt unter Zensierung bei Strategiewechsel",
        analysis_plan=(
            "IPTW-gewichtete lineare Modelle mit robusten, patientenge"
            "clusterten Standardfehlern; Zensierungsgewichte fuer die "
            "Wahrscheinlichkeit eines Folge-EKGs."
        ),
        active_comparator="calcium_channel_blocker",
    ),
    TargetTrialProtocol(
        protocol_id="antiarrhythmic_class_iii_vs_class_i",
        eligibility=(
            "Erwachsene mit dokumentierter ventrikulaerer oder atrialer "
            "Arrhythmie, ohne Antiarrhythmikum in den vorangehenden 30 Tagen, "
            "mit auswertbarem Vor-EKG und bekanntem Kalium- und Magnesiumwert."
        ),
        treatment_strategies=(
            "Beginn eines Klasse-III-Antiarrhythmikums",
            "Beginn eines Klasse-I-Antiarrhythmikums",
        ),
        assignment_procedure=(
            "Nicht randomisiert; Propensity-Gewichtung mit Elektrolyten, "
            "Nierenfunktion, Alter, Geschlecht und Ausgangs-QTc."
        ),
        follow_up_start="Zeitpunkt der ersten dokumentierten Gabe",
        outcome="QTc-Aenderung und Tpeak-Tend-Aenderung im Folge-EKG",
        causal_contrast="Per-Protokoll-Effekt auf die Repolarisationsdauer",
        analysis_plan=(
            "IPTW plus Zensierungsgewichte; Dosis als sekundaere "
            "Expositionsdefinition."
        ),
        active_comparator="antiarrhythmic_class_i",
    ),
)


@dataclass(frozen=True)
class ProtocolComplianceResult:
    protocol_id: str
    design: str
    claim_level: str
    satisfied: tuple[str, ...] = field(default_factory=tuple)
    violated: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "design": self.design,
            "claim_level": self.claim_level,
            "satisfied": list(self.satisfied),
            "violated": list(self.violated),
            "causal_claim_permitted": self.claim_level == CLAIM_EMULATED_EFFECT,
            **dict(self.diagnostics),
            "target_trial_version": TARGET_TRIAL_VERSION,
        }


def assess_protocol_compliance(
    cohort: pd.DataFrame,
    protocol: TargetTrialProtocol,
    *,
    treatment_column: str = "treatment_arm",
    time_zero_column: str = "time_zero",
    prior_exposure_column: str = "prior_exposure_days",
    baseline_ecg_column: str = "baseline_ecg_time",
    follow_up_ecg_column: str = "follow_up_ecg_time",
    washout_days: float = 30.0,
) -> ProtocolComplianceResult:
    """Check an assembled cohort against the protocol it claims to emulate.

    Every element that fails drops the permitted claim from an emulated causal
    effect to a pharmacodynamic association. This is the whole point: the
    downgrade is automatic and recorded, not left to the analyst's discretion
    at write-up time.
    """

    satisfied: list[str] = []
    violated: list[str] = []
    diagnostics: dict[str, object] = {"n_rows": int(len(cohort))}

    if treatment_column in cohort.columns:
        arms = (
            cohort[treatment_column].astype("string").dropna().unique().tolist()
        )
        diagnostics["arms_observed"] = sorted(str(arm) for arm in arms)
        if len(arms) >= 2:
            satisfied.append("active_comparator_present")
        else:
            violated.append("single_arm_no_active_comparator")
    else:
        violated.append("treatment_arm_missing")
        diagnostics["arms_observed"] = []

    if time_zero_column in cohort.columns:
        time_zero = pd.to_datetime(cohort[time_zero_column], errors="coerce")
        share = float(time_zero.notna().mean()) if len(cohort) else 0.0
        diagnostics["time_zero_defined_share"] = share
        if share >= 0.99:
            satisfied.append("common_time_zero")
        else:
            violated.append("time_zero_undefined_for_some_rows")
    else:
        violated.append("time_zero_missing")

    if protocol.new_user_required:
        if prior_exposure_column in cohort.columns:
            prior = pd.to_numeric(
                cohort[prior_exposure_column],
                errors="coerce",
            )
            prevalent = float((prior.fillna(0) > 0).mean())
            diagnostics["prevalent_user_share"] = prevalent
            diagnostics["washout_days"] = float(washout_days)
            if prevalent <= 0.0:
                satisfied.append("new_user_design")
            else:
                violated.append("prevalent_users_included")
        else:
            violated.append("prior_exposure_unknown")

    if {baseline_ecg_column, follow_up_ecg_column} <= set(cohort.columns):
        baseline = pd.to_datetime(cohort[baseline_ecg_column], errors="coerce")
        follow_up = pd.to_datetime(cohort[follow_up_ecg_column], errors="coerce")
        paired = float((baseline.notna() & follow_up.notna()).mean())
        diagnostics["paired_ecg_share"] = paired
        # The share *without* a follow-up ECG is itself the informative
        # follow-up problem, so it is reported rather than silently dropped.
        diagnostics["follow_up_ecg_missing_share"] = float(1.0 - paired)
        if paired >= 0.8:
            satisfied.append("adequate_paired_follow_up")
        else:
            violated.append("informative_follow_up_ecg_selection")
        if time_zero_column in cohort.columns:
            time_zero = pd.to_datetime(cohort[time_zero_column], errors="coerce")
            before = float((baseline < time_zero).mean())
            diagnostics["baseline_before_time_zero_share"] = before
            if before >= 0.99:
                satisfied.append("baseline_precedes_time_zero")
            else:
                violated.append("baseline_ecg_after_time_zero")
    else:
        violated.append("paired_ecgs_missing")

    design = (
        DESIGN_NEW_USER_ACTIVE_COMPARATOR
        if not violated
        else DESIGN_PRE_POST
    )
    claim = CLAIM_EMULATED_EFFECT if not violated else CLAIM_ASSOCIATION
    return ProtocolComplianceResult(
        protocol_id=protocol.protocol_id,
        design=design,
        claim_level=claim,
        satisfied=tuple(satisfied),
        violated=tuple(violated),
        diagnostics=diagnostics,
    )


def follow_up_ecg_probability(
    cohort: pd.DataFrame,
    *,
    followed_column: str = "has_follow_up_ecg",
    covariate_columns: Sequence[str] = (),
) -> dict[str, object]:
    """Model whether a follow-up ECG happened at all, and weight for it.

    Censoring weights for informative follow-up. A patient who improved and
    went home never gets the second recording; analysing only the pairs that
    exist over-represents patients who stayed unwell.
    """

    from statsmodels.discrete.discrete_model import Logit

    if followed_column not in cohort.columns:
        return {"status": "not_evaluable", "failure_reason": "column_missing"}
    outcome = pd.to_numeric(cohort[followed_column], errors="coerce").to_numpy()
    usable = [column for column in covariate_columns if column in cohort.columns]
    if not usable:
        return {
            "status": "not_evaluable",
            "failure_reason": "no_covariates_available",
        }
    design = cohort.loc[:, usable].apply(pd.to_numeric, errors="coerce")
    complete = design.notna().all(axis=1).to_numpy() & np.isfinite(outcome)
    if complete.sum() < 50 or np.unique(outcome[complete]).size < 2:
        return {
            "status": "not_evaluable",
            "failure_reason": "insufficient_or_single_class",
        }
    matrix = np.column_stack(
        (np.ones(int(complete.sum())), design.loc[complete].to_numpy())
    )
    try:
        model = Logit(outcome[complete], matrix).fit(disp=0)
        probability = np.asarray(model.predict(matrix), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "not_evaluable",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    marginal = float(np.mean(outcome[complete]))
    clipped = np.clip(probability, 0.01, 0.99)
    weights = np.full(len(outcome), np.nan, dtype=float)
    weights[complete] = np.where(
        outcome[complete] > 0.5,
        marginal / clipped,
        np.nan,
    )
    return {
        "status": "evaluable",
        "censoring_weights": weights,
        "follow_up_rate": marginal,
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
        "covariates": list(usable),
        "informative_follow_up_suspected": bool(
            float(np.max(probability) - np.min(probability)) > 0.2
        ),
        "interpretation": (
            "weights the observed pairs back towards everyone eligible; "
            "valid only if follow-up is random given these covariates"
        ),
        "target_trial_version": TARGET_TRIAL_VERSION,
    }


def dose_exposure_levels(
    doses: Iterable[float],
    *,
    quantiles: Sequence[float] = (0.33, 0.67),
) -> dict[str, object]:
    """Turn a continuous dose into prespecified exposure categories.

    Data-driven cutpoints chosen after seeing the outcome are a known route to
    a false positive, so the quantiles are fixed in advance and reported with
    the resulting boundaries.
    """

    values = np.asarray(list(doses), dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return {
            "status": "not_evaluable",
            "levels": np.full(len(values), "unknown", dtype=object),
        }
    cuts = np.quantile(finite, list(quantiles))
    levels = np.full(len(values), "unknown", dtype=object)
    finite_mask = np.isfinite(values)
    assigned = np.digitize(values[finite_mask], cuts, right=False)
    names = np.array(["low", "medium", "high"], dtype=object)
    levels[finite_mask] = names[np.clip(assigned, 0, len(names) - 1)]
    return {
        "status": "evaluable",
        "levels": levels,
        "cut_points": [float(value) for value in cuts],
        "quantiles": list(quantiles),
        "cutpoints_prespecified": True,
        "target_trial_version": TARGET_TRIAL_VERSION,
    }


def protocol_registry_table(
    protocols: Iterable[TargetTrialProtocol] = DEFAULT_PROTOCOLS,
) -> pd.DataFrame:
    """The protocols as a reviewable table, for preregistration."""

    return pd.DataFrame([protocol.to_dict() for protocol in protocols])


__all__ = [
    "CLAIM_ASSOCIATION",
    "CLAIM_EMULATED_EFFECT",
    "DEFAULT_PROTOCOLS",
    "DESIGN_NEW_USER_ACTIVE_COMPARATOR",
    "DESIGN_PRE_POST",
    "ProtocolComplianceResult",
    "TARGET_TRIAL_VERSION",
    "TargetTrialProtocol",
    "assess_protocol_compliance",
    "dose_exposure_levels",
    "follow_up_ecg_probability",
    "protocol_registry_table",
]
