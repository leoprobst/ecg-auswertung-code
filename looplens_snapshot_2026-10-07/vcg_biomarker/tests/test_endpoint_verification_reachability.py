"""Whether an accuracy claim is reachable is fixed before the analysis starts.

The confirmation gate requires that at least 80 % of the reference-blind
universe carries a known reference.  For most of this project's endpoints the
answer to that is decided by how hospitals work, not by the marker.

Measured on the 63 endpoints of the last development run, **six reach it**:
30-day mortality and the three new-onset atrial fibrillation horizons, all
verified by construction because no test is needed to know them, plus the two
QTc endpoints, which are read off the recorder's own measurement.  Every
echocardiographic endpoint sits between 0.003 and 0.02 -- an echocardiogram is
done on a few per cent of recordings.

So no structural endpoint in this study can currently be confirmed as a
diagnostic-accuracy claim, and that is a property of the design.  These tests
pin that the table says so per endpoint, and that it distinguishes "the
reference was never obtained" from "the marker failed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.selection_bias import (
    MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION,
    endpoint_verification_reachability,
)


def _summary(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "label_id",
            "positive_ecg_count",
            "negative_ecg_count",
            "unavailable_ecg_count",
            "prevalence_status",
        ],
    )


def _reach(rows):
    return endpoint_verification_reachability(
        _summary(rows),
        minimum_verification_rate=MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION,
    ).set_index("label_id")


def test_an_endpoint_verified_for_everyone_is_reachable():
    # Mortality needs no test: every recording has a known outcome.
    row = _reach(
        [("mortality_30d", 9369, 65041, 0, "prevalence_supports_inference")]
    ).loc["mortality_30d"]
    assert row["verification_rate"] == 1.0
    assert bool(row["diagnostic_accuracy_claim_reachable"])
    assert row["unreachable_reason"] == ""


def test_an_echo_endpoint_is_not_reachable_and_says_why():
    # significant_valvular_disease: 9,701 verified against 587,749 unavailable.
    row = _reach(
        [
            (
                "significant_valvular_disease",
                9400,
                301,
                587749,
                "high_prevalence_verification_conditioned",
            )
        ]
    ).loc["significant_valvular_disease"]
    assert row["verification_rate"] < 0.02
    assert not bool(row["diagnostic_accuracy_claim_reachable"])
    assert "verification_rate_below_confirmation_threshold" in row[
        "unreachable_reason"
    ]
    # Two independent reasons, both named rather than collapsed into one.
    assert "reference_obtained_mainly_on_suspicion" in row["unreachable_reason"]


def test_a_high_prevalence_endpoint_is_flagged_even_when_widely_verified():
    # Verification alone is not enough: a reference obtained mainly on
    # suspicion produces a selected negative arm however many people got it.
    row = _reach(
        [
            (
                "troponin_positive_any",
                18581,
                1582,
                2000,
                "high_prevalence_verification_conditioned",
            )
        ]
    ).loc["troponin_positive_any"]
    assert row["verification_rate"] > 0.9
    assert not bool(row["diagnostic_accuracy_claim_reachable"])
    assert row["unreachable_reason"] == "reference_obtained_mainly_on_suspicion"


def test_an_endpoint_with_no_universe_is_not_silently_reachable():
    row = _reach([("empty", 0, 0, 0, "not_assessable")]).loc["empty"]
    assert not np.isfinite(row["verification_rate"])
    assert not bool(row["diagnostic_accuracy_claim_reachable"])
    assert row["unreachable_reason"] == "no_reference_blind_universe"


def test_the_declared_threshold_travels_with_every_row():
    table = _reach(
        [("a", 100, 100, 0, "prevalence_supports_inference")]
    )
    assert table.iloc[0]["minimum_verification_rate"] == (
        MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION
    )


def test_an_empty_summary_yields_an_empty_but_typed_table():
    table = endpoint_verification_reachability(
        pd.DataFrame(), minimum_verification_rate=0.8
    )
    assert table.empty
    assert "diagnostic_accuracy_claim_reachable" in table.columns
