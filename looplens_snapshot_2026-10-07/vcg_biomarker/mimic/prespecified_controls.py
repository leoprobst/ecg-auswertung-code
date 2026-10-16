"""Does the screen find what is already known — and how often does it invent?

The broad screen tests millions of associations and controls the false
discovery rate, but nothing tests **the screen itself**.  Two questions stay
open after every run, and neither is answerable from the results table:

* If the pipeline cannot recover a relationship that a cardiology textbook
  states, its negative findings mean nothing.  "No association between marker X
  and phenotype Y" is only informative from an instrument that demonstrably
  detects associations of that size.
* An FDR of 1 % bounds the share of false discoveries **under the null the
  procedure assumes**.  In an intensive-care cohort the dominant error is not
  chance but confounding by acuity: a marker and a phenotype can both track how
  ill someone is without any electrophysiological path between them.  No
  multiplicity correction addresses that, because those associations are real.

This module declares both kinds of control **before** results are seen, which
is the only thing that makes them worth anything.

**Positive controls** are relationships with a mechanism, a direction and a
rough size from the literature.  They are checked individually: found, wrong
direction, too weak, or not evaluable.  "Not evaluable" is reported rather than
skipped — a control that cannot be run because the covariate is not written is
itself a finding about the pipeline (see the heart-rate control below).

**Negative controls** are phenotypes with no plausible causal path from cardiac
electrophysiology, chosen to share the measurement context: they occur in the
same hospital population and are coded the same way.  They are *not* checked
individually — some will associate, and that is the point.  What is reported is
the **rate**, which estimates the real false-positive rate of the screen with
confounding included.

Sources are named per control.  The thresholds are deliberately loose: the
question is whether the instrument works at all, not whether it reproduces a
published effect size to two digits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PRESPECIFIED_CONTROL_VERSION = "1.1.0"

EFFECT_RHO = "rank_correlation"
EFFECT_AUC = "roc_auc"

VERDICT_FOUND = "recovered"
VERDICT_WRONG_DIRECTION = "recovered_with_opposite_direction"
VERDICT_TOO_WEAK = "below_declared_effect_floor"
VERDICT_TARGET_MISSING = "not_evaluable_target_absent"
VERDICT_MARKER_MISSING = "not_evaluable_marker_absent"


@dataclass(frozen=True)
class PositiveControl:
    """A relationship the pipeline must recover to be believed on nulls."""

    control_id: str
    target_id: str
    effect_kind: str
    #: Any one of these markers counts; they are alternative measurements of
    #: the same physiological quantity, not independent chances.
    marker_ids: tuple[str, ...]
    #: +1 when the marker rises with the target, -1 when it falls.
    expected_direction: int
    minimum_effect: float
    mechanism: str
    source: str


@dataclass(frozen=True)
class NegativeControlSet:
    """Phenotypes with no plausible path from cardiac electrophysiology."""

    set_id: str
    target_ids: tuple[str, ...]
    rationale: str
    #: Above this share of associating targets the screen is reporting
    #: confounding at a rate that makes individual hits uninterpretable.
    maximum_hit_rate: float = 0.05


# The effect floors follow the project's prespecified thresholds: |rho| >= 0.10
# for continuous targets and an AUC distance >= 0.05 for binary ones.  A control
# that clears only those is still a weak recovery, which is why the electrolyte
# controls -- where the textbook effect is large -- ask for more.
POSITIVE_CONTROLS: tuple[PositiveControl, ...] = (
    PositiveControl(
        "hyperkalaemia_tall_t_waves",
        target_id="lab_potassium",
        effect_kind=EFFECT_RHO,
        marker_ids=(
            "t_vector_magnitude",
            "t_loop_diameter",
            "classical_peaked_t_wave_heuristic",
            "spatial_magnitude_p95",
        ),
        expected_direction=1,
        minimum_effect=0.10,
        mechanism=(
            "Raised extracellular potassium accelerates phase-3 repolarisation "
            "and raises the T-wave amplitude; the earliest and most reliable "
            "electrocardiographic sign of hyperkalaemia."
        ),
        source="Surawicz, Am Heart J 1967; Levis, J Emerg Med 2013",
    ),
    PositiveControl(
        "hyperkalaemia_qrs_widening",
        target_id="lab_potassium",
        effect_kind=EFFECT_RHO,
        marker_ids=("qrs_duration_ms", "filtered_qrs_duration_ms"),
        expected_direction=1,
        minimum_effect=0.10,
        mechanism=(
            "At higher concentrations sodium-channel availability falls and "
            "conduction slows, widening the QRS complex."
        ),
        source="Surawicz, Am Heart J 1967",
    ),
    PositiveControl(
        "calcium_qt_shortening",
        target_id="lab_total_calcium",
        effect_kind=EFFECT_RHO,
        marker_ids=("qt_interval_ms", "qtc_fridericia_ms", "jt_interval_ms"),
        expected_direction=-1,
        minimum_effect=0.10,
        mechanism=(
            "Extracellular calcium sets the plateau duration: hypocalcaemia "
            "lengthens the ST segment and thus the QT interval, hypercalcaemia "
            "shortens it. The relationship is monotone across the range."
        ),
        source="Bronsky, Am J Cardiol 1961; Nijjer, BMJ Case Rep 2010",
    ),
    PositiveControl(
        "ionised_calcium_qt_shortening",
        target_id="lab_ionized_calcium",
        effect_kind=EFFECT_RHO,
        marker_ids=("qt_interval_ms", "qtc_fridericia_ms", "jt_interval_ms"),
        expected_direction=-1,
        minimum_effect=0.10,
        mechanism="As above, against the physiologically active fraction.",
        source="Bronsky, Am J Cardiol 1961",
    ),
    PositiveControl(
        "digoxin_st_depression",
        target_id="medication_digoxin",
        effect_kind=EFFECT_AUC,
        marker_ids=(
            "st_depression",
            "classical_st_depression_contiguous",
            "st_deviation_vector_magnitude_uv",
        ),
        expected_direction=-1,
        minimum_effect=0.05,
        mechanism=(
            "Sodium-potassium-ATPase inhibition shortens repolarisation and "
            "produces the downsloping 'scooped' ST depression that is named "
            "after the drug."
        ),
        source="Ma, J Emerg Med 2001",
    ),
    PositiveControl(
        # Declared even though it is expected to be reported as not evaluable:
        # heart rate carries the role ``ecg_context_comparator`` and is written
        # by no panel, so the single best-established relationship in
        # electrocardiography cannot currently be checked. That is the point of
        # declaring it.
        "heart_rate_qt_dependence",
        target_id="heart_rate",
        effect_kind=EFFECT_RHO,
        marker_ids=("qt_interval_ms",),
        expected_direction=-1,
        minimum_effect=0.30,
        mechanism=(
            "The QT interval shortens as rate rises; the reference "
            "relationship every rate correction formula approximates."
        ),
        source="Bazett, Heart 1920; Fridericia, Acta Med Scand 1920",
    ),
)


NEGATIVE_CONTROLS: tuple[NegativeControlSet, ...] = (
    NegativeControlSet(
        "non_cardiac_sensory_and_dermatological",
        # Audit 2026-09-23 (5.3.3): the names are phecodeX descriptions and
        # must match them exactly. Five of the ten did not ("Refraction and
        # accommodation disorders", "Hearing loss", "Benign neoplasm of
        # skin", "Onychomycosis", "Hordeolum and chalazion") and so never
        # selected a phenotype. They are replaced by the phecodeX entries of
        # the same concepts (SO_387, SO_396, CA_138, DE_660.122, SO_360.1/2);
        # no result had been seen for any of them.
        target_ids=(
            "Cataract",
            "Disorders of refraction and accommodation",
            "Hearing impairment",
            "Benign neoplasm of the skin",
            "Actinic keratosis",
            "Seborrheic dermatitis",
            "Impacted cerumen",
            "Dermatophytosis of nail",
            "Myopia",
            "Hordeolum",
            "Chalazion",
        ),
        rationale=(
            "Common, coded in the same population, and with no pathway from "
            "cardiac electrophysiology. An association here is confounding by "
            "age or by how much medical attention a patient received -- the "
            "error a false-discovery rate does not bound."
        ),
    ),
)


def _direction_of(effect: float, effect_kind: str) -> int:
    if effect_kind == EFFECT_AUC:
        return 1 if effect >= 0.5 else -1
    return 1 if effect >= 0.0 else -1


def _magnitude_of(effect: float, effect_kind: str) -> float:
    if effect_kind == EFFECT_AUC:
        return abs(effect - 0.5)
    return abs(effect)


def evaluate_positive_controls(
    results: pd.DataFrame,
    *,
    target_column: str,
    feature_column: str = "feature_id",
    effect_column: str = "effect",
    controls: Sequence[PositiveControl] = POSITIVE_CONTROLS,
) -> pd.DataFrame:
    """Check each declared relationship against a results table."""

    columns = [
        "control_id",
        "target_id",
        "verdict",
        "best_marker",
        "best_effect",
        "observed_direction",
        "expected_direction",
        "minimum_effect",
        "mechanism",
        "source",
        "control_version",
    ]
    rows: list[dict] = []
    have_columns = (
        not results.empty
        and target_column in results.columns
        and feature_column in results.columns
        and effect_column in results.columns
    )
    for control in controls:
        row = {
            "control_id": control.control_id,
            "target_id": control.target_id,
            "verdict": VERDICT_TARGET_MISSING,
            "best_marker": None,
            "best_effect": float("nan"),
            "observed_direction": 0,
            "expected_direction": control.expected_direction,
            "minimum_effect": control.minimum_effect,
            "mechanism": control.mechanism,
            "source": control.source,
            "control_version": PRESPECIFIED_CONTROL_VERSION,
        }
        if not have_columns:
            rows.append(row)
            continue
        scoped = results.loc[
            results[target_column].astype(str) == control.target_id
        ]
        if scoped.empty:
            rows.append(row)
            continue
        scoped = scoped.loc[
            scoped[feature_column].astype(str).isin(control.marker_ids)
        ]
        if scoped.empty:
            row["verdict"] = VERDICT_MARKER_MISSING
            rows.append(row)
            continue
        effects = pd.to_numeric(scoped[effect_column], errors="coerce")
        magnitudes = effects.map(
            lambda value: _magnitude_of(value, control.effect_kind)
        )
        usable = magnitudes[np.isfinite(magnitudes)]
        if usable.empty:
            row["verdict"] = VERDICT_MARKER_MISSING
            rows.append(row)
            continue
        best_position = usable.idxmax()
        best_effect = float(effects.loc[best_position])
        observed = _direction_of(best_effect, control.effect_kind)
        row["best_marker"] = str(scoped.loc[best_position, feature_column])
        row["best_effect"] = best_effect
        row["observed_direction"] = observed
        if _magnitude_of(best_effect, control.effect_kind) < control.minimum_effect:
            row["verdict"] = VERDICT_TOO_WEAK
        elif observed != control.expected_direction:
            row["verdict"] = VERDICT_WRONG_DIRECTION
        else:
            row["verdict"] = VERDICT_FOUND
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def evaluate_negative_controls(
    results: pd.DataFrame,
    *,
    target_column: str,
    effect_column: str = "effect",
    effect_kind: str = EFFECT_AUC,
    minimum_effect: float = 0.05,
    control_sets: Sequence[NegativeControlSet] = NEGATIVE_CONTROLS,
) -> pd.DataFrame:
    """Report how often the screen associates with an implausible target."""

    columns = [
        "set_id",
        "targets_declared",
        "targets_present",
        "targets_with_association",
        "hit_rate",
        "maximum_hit_rate",
        "within_expectation",
        "associating_targets",
        "rationale",
        "control_version",
    ]
    rows: list[dict] = []
    for control_set in control_sets:
        present: list[str] = []
        associating: list[str] = []
        if not results.empty and target_column in results.columns:
            keys = results[target_column].astype(str)
            for target_id in control_set.target_ids:
                scoped = results.loc[keys == target_id]
                if scoped.empty:
                    continue
                present.append(target_id)
                effects = pd.to_numeric(
                    scoped[effect_column], errors="coerce"
                ).dropna()
                if effects.empty:
                    continue
                magnitude = max(
                    _magnitude_of(value, effect_kind) for value in effects
                )
                if magnitude >= minimum_effect:
                    associating.append(target_id)
        hit_rate = len(associating) / len(present) if present else float("nan")
        rows.append(
            {
                "set_id": control_set.set_id,
                "targets_declared": len(control_set.target_ids),
                "targets_present": len(present),
                "targets_with_association": len(associating),
                "hit_rate": hit_rate,
                "maximum_hit_rate": control_set.maximum_hit_rate,
                "within_expectation": (
                    bool(hit_rate <= control_set.maximum_hit_rate)
                    if present
                    else False
                ),
                "associating_targets": "|".join(sorted(associating)),
                "rationale": control_set.rationale,
                "control_version": PRESPECIFIED_CONTROL_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def control_summary(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
) -> dict:
    """One readable verdict on whether the screen's nulls mean anything."""

    evaluable = positive.loc[
        ~positive["verdict"].isin(
            {VERDICT_TARGET_MISSING, VERDICT_MARKER_MISSING}
        )
    ]
    recovered = int((evaluable["verdict"] == VERDICT_FOUND).sum())
    not_evaluable = int(len(positive) - len(evaluable))
    negative_ok = bool(negative["within_expectation"].all()) if len(negative) else False
    return {
        "control_version": PRESPECIFIED_CONTROL_VERSION,
        "positive_controls_declared": int(len(positive)),
        "positive_controls_evaluable": int(len(evaluable)),
        "positive_controls_recovered": recovered,
        "positive_controls_not_evaluable": not_evaluable,
        "negative_control_sets_within_expectation": negative_ok,
        # The screen is only allowed to be believed on its nulls if it
        # demonstrably finds what is known.  A partial recovery is reported as
        # such rather than rounded up.
        "negative_findings_interpretable": bool(
            len(evaluable) > 0
            and recovered == len(evaluable)
            and negative_ok
        ),
    }
