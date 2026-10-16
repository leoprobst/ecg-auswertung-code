"""Recording quality must be allowed to beat physiology, and to say so.

In the last development run the strongest predictor of major depressive
disorder within 30 days was ``einthoven_relation_error_rms`` -- how badly leads
I, II and III satisfy Einthoven's law -- at rank correlation 0.51.  The quality
markers were afterwards removed from the analysis panels, which stops them being
reported as findings and simultaneously removes the only instrument that could
detect the problem.

Replaying the stored hits through this module marks 21 of 261 assessable
phenotypes as beaten by a quality marker: depression, generalized anxiety, mild
cognitive impairment, hemiplegia, wheelchair dependence -- phenotypes whose
patients move during the recording -- and, more interestingly, diaphragmatic
hernia beaten by the chest-lead misplacement proxy, which is anatomy rather than
sloppiness.  Both readings are useful; neither is available if the control is
not screened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.signal_quality_controls import (
    QUALITY_CONTROL_PANEL_ID,
    TECHNICAL_INFLUENCE_SHARE,
    VERDICT_CLEAN,
    VERDICT_DOMINATED,
    VERDICT_INFLUENCED,
    VERDICT_UNKNOWN,
    annotate_hits_with_quality_verdict,
    compare_quality_against_candidates,
    quality_control_features,
)

QUALITY = ("einthoven_relation_error_rms", "vcg_trust_score")


def _hits(rows):
    return pd.DataFrame(
        rows, columns=["phenotype_id", "feature_id", "validation_aligned_auc"]
    )


def test_only_control_markers_that_were_extracted_are_used():
    present = quality_control_features(
        ["vcg_trust_score", "qrs_loop_tortuosity", "fiducial_confidence"]
    )
    assert present == ("fiducial_confidence", "vcg_trust_score")
    assert quality_control_features([]) == ()


def test_a_quality_marker_that_wins_is_named_as_the_winner():
    hits = _hits(
        [
            ("depression", "einthoven_relation_error_rms", 0.75),
            ("depression", "qrs_loop_tortuosity", 0.62),
        ]
    )
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    row = verdicts.iloc[0]
    assert row["verdict"] == VERDICT_DOMINATED
    assert row["best_quality_feature"] == "einthoven_relation_error_rms"
    assert row["best_candidate_feature"] == "qrs_loop_tortuosity"
    assert row["quality_share"] > 1.0


def test_a_close_second_is_reported_as_influenced_not_clean():
    # 0.5 distance 0.16 for quality against 0.19 for the candidate: the
    # candidate wins, but not by enough to call the target unaffected.
    hits = _hits(
        [
            ("target", "vcg_trust_score", 0.66),
            ("target", "qrs_loop_tortuosity", 0.69),
        ]
    )
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    row = verdicts.iloc[0]
    assert row["verdict"] == VERDICT_INFLUENCED
    assert row["quality_share"] >= TECHNICAL_INFLUENCE_SHARE


def test_a_clearly_physiological_target_is_left_alone():
    hits = _hits(
        [
            ("target", "vcg_trust_score", 0.53),
            ("target", "qrs_loop_tortuosity", 0.88),
        ]
    )
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    assert verdicts.iloc[0]["verdict"] == VERDICT_CLEAN


def test_a_target_without_a_control_result_is_not_declared_clean():
    hits = _hits([("target", "qrs_loop_tortuosity", 0.88)])
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    assert verdicts.iloc[0]["verdict"] == VERDICT_UNKNOWN
    assert np.isnan(verdicts.iloc[0]["quality_share"])


def test_direction_does_not_decide_the_comparison():
    # An AUC of 0.30 is as strong as one of 0.70; the control must not lose
    # merely because its association points the other way.
    hits = _hits(
        [
            ("target", "einthoven_relation_error_rms", 0.30),
            ("target", "qrs_loop_tortuosity", 0.65),
        ]
    )
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    assert verdicts.iloc[0]["verdict"] == VERDICT_DOMINATED


def test_the_verdict_is_carried_onto_every_hit():
    hits = _hits(
        [
            ("a", "einthoven_relation_error_rms", 0.75),
            ("a", "qrs_loop_tortuosity", 0.62),
            ("b", "qrs_loop_tortuosity", 0.90),
            ("b", "vcg_trust_score", 0.51),
        ]
    )
    verdicts = compare_quality_against_candidates(
        hits, target_column="phenotype_id", quality_features=QUALITY
    )
    annotated = annotate_hits_with_quality_verdict(
        hits.copy(), verdicts, target_column="phenotype_id"
    )
    assert set(annotated.loc[annotated["phenotype_id"] == "a",
                             "quality_control_verdict"]) == {VERDICT_DOMINATED}
    assert set(annotated.loc[annotated["phenotype_id"] == "b",
                             "quality_control_verdict"]) == {VERDICT_CLEAN}


def test_empty_input_yields_an_empty_but_typed_table():
    verdicts = compare_quality_against_candidates(
        _hits([]), target_column="phenotype_id", quality_features=QUALITY
    )
    assert verdicts.empty
    assert "verdict" in verdicts.columns
    annotated = annotate_hits_with_quality_verdict(
        _hits([]), verdicts, target_column="phenotype_id"
    )
    assert "quality_control_verdict" in annotated.columns


def test_the_control_family_has_its_own_panel_identifier():
    # Sharing a family would make the control borrow power from, or lend power
    # to, the candidate markers under FDR.
    assert QUALITY_CONTROL_PANEL_ID not in {
        "primary_core",
        "sentinel",
        "discovery_reserve",
        "classical",
    }
