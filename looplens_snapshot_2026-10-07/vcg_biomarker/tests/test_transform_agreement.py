"""A threshold without its transform is not a threshold.

The X, Y and Z leads of this project are reconstructed from twelve leads, and
the two accepted reconstructions disagree.  Measured on the real store over
8.825 paired recordings, the spatial QRS-T angle is **11,6 degrees higher**
under inverse Dower than under Kors (81,6 against 93,2), which is the direction
and the order of magnitude the published Frank comparisons predict.

The mean offset is the smaller half of the finding.  The limits of agreement
run from **-26,5 to +49,7 degrees**, so for one patient the choice of transform
can move the angle by fifty degrees.  A population offset can be corrected; an
individual spread of that size means the two derivations are not
interchangeable at the bedside, and the reporting set has to say which one was
used.

The historical counts above are not current release evidence. Rank correlation
does not establish clinical interchangeability or validate a conversion.
Structural transform independence must be declared from the implementation;
numerically identical pairs alone cannot prove it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.transform_agreement import _sample_training_partitions, _partition_content_identity


def test_partition_selection_never_falls_back_to_locked_data(tmp_path):
    locked = tmp_path / "split=locked_test"
    locked.mkdir()
    (locked / "part-0.parquet").write_bytes(b"must never be opened")
    with pytest.raises(RuntimeError, match="No training"):
        _sample_training_partitions(str(tmp_path / "split=*" / "part-*.parquet"), 2)
    with pytest.raises(ValueError, match="split=train"):
        _sample_training_partitions(str(locked / "part-*.parquet"), 2)


def test_partition_identity_detects_equal_size_content_changes(tmp_path):
    train = tmp_path / "split=train"
    train.mkdir()
    path = train / "part-0.parquet"
    path.write_bytes(b"old")
    paths = _sample_training_partitions(str(tmp_path / "split=*" / "part-*.parquet"), 2)
    before = _partition_content_identity(paths)
    path.write_bytes(b"new")
    assert before != _partition_content_identity(paths)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_partition_sample_count_rejected(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        _sample_training_partitions(str(tmp_path / "split=*" / "part-*.parquet"), count)

from vcg_biomarker.mimic.panels import TRANSFORM_INDEPENDENT_FEATURE_IDS
from vcg_biomarker.mimic.transform_agreement import (
    LIMITS_OF_AGREEMENT_Z,
    AgreementPolicy,
    agreement_for_pair,
    agreement_table,
)


def _pair(marker_id: str, primary, comparison, **kwargs):
    return agreement_for_pair(
        marker_id,
        np.asarray(primary, dtype=float),
        np.asarray(comparison, dtype=float),
        primary_transform="kors",
        comparison_transform="inverse_dower",
        independent_pairs=True,
        **kwargs,
    )


def test_an_identical_pair_is_transform_independent_not_perfect_agreement():
    """The two statements are different and only one of them is true here.

    ``qt_interval_ms`` is read off the twelve leads and the delineation, so
    both arms compute the same number from the same input. Calling that
    "perfect agreement" would report robustness where the transform was never
    involved.
    """

    values = np.linspace(350.0, 460.0, 400)
    row = _pair("qt_interval_ms", values, values)
    assert row["transform_independent"] is True
    assert row["agreement_class"] == "transform_independent"
    assert row["bias"] == 0.0
    assert row["spearman_rho"] == 1.0


def test_a_constant_offset_is_reported_as_bias_with_its_interval():
    rng = np.random.default_rng(11)
    primary = rng.normal(80.0, 25.0, 2000)
    comparison = primary + 11.6
    row = _pair("spatial_qrs_t_angle_deg", primary, comparison)
    assert row["bias"] == pytest.approx(11.6, abs=1e-6)
    # An exactly constant offset has no scatter, so the interval on the bias
    # and the limits of agreement both collapse onto it. That is the correct
    # answer, not a degenerate one.
    assert row["bias_ci_low"] <= row["bias"] <= row["bias_ci_high"]
    assert row["difference_sd"] == pytest.approx(0.0, abs=1e-9)
    assert row["limit_of_agreement_low"] == pytest.approx(11.6, abs=1e-6)
    assert row["conversion_slope"] == pytest.approx(1.0, abs=1e-6)
    assert row["conversion_intercept"] == pytest.approx(11.6, abs=1e-4)
    assert not row["transform_independent"]


def test_the_limits_of_agreement_are_bias_plus_minus_two_standard_deviations():
    rng = np.random.default_rng(5)
    primary = rng.normal(80.0, 25.0, 4000)
    comparison = primary + 11.6 + rng.normal(0.0, 19.0, 4000)
    row = _pair("spatial_qrs_t_angle_deg", primary, comparison)
    expected_low = row["bias"] - LIMITS_OF_AGREEMENT_Z * row["difference_sd"]
    expected_high = row["bias"] + LIMITS_OF_AGREEMENT_Z * row["difference_sd"]
    assert row["limit_of_agreement_low"] == pytest.approx(expected_low)
    assert row["limit_of_agreement_high"] == pytest.approx(expected_high)
    # The spread, not the offset, is what makes a threshold untransferable.
    assert row["limit_of_agreement_high"] - row["limit_of_agreement_low"] > 40.0


def test_rank_correlation_does_not_establish_agreement():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, 3000)
    policy = AgreementPolicy()
    excellent = _pair("a", base, base + rng.normal(0.0, 0.05, 3000), policy=policy)
    usable = _pair("b", base, base + rng.normal(0.0, 0.6, 3000), policy=policy)
    poor = _pair("c", base, base + rng.normal(0.0, 2.5, 3000), policy=policy)
    assert excellent["agreement_class"] == "high_rank_correlation_agreement_unproven"
    assert usable["agreement_class"] == "moderate_rank_correlation_agreement_unproven"
    assert poor["agreement_class"] == "low_rank_correlation_agreement_unproven"
    assert not excellent["clinical_interchangeability_established"]


def test_perfect_correlation_with_large_bias_never_licenses_conversion():
    values = np.arange(400.)
    result = _pair("spatial_qrs_t_angle_deg", values, values*10+100)
    assert result["spearman_rho"] == pytest.approx(1)
    assert result["bias"] > 100
    assert not result["clinical_interchangeability_established"]
    assert not result["conversion_validated"]


def test_accidental_equality_is_not_structural_transform_independence():
    result = _pair("unknown_marker", np.arange(400.), np.arange(400.))
    assert result["numerically_identical"]
    assert not result["transform_independent"]


def test_broadcasting_cannot_invent_pairs():
    with pytest.raises(ValueError, match="aligned"):
        _pair("x", np.ones((300, 1)), np.ones(300))


def test_equal_constants_do_not_invent_correlation_or_conversion():
    result = _pair("unknown_marker", np.ones(400), np.ones(400))
    assert result["numerically_identical"]
    assert np.isnan(result["spearman_rho"])
    assert np.isnan(result["conversion_slope"])
    assert not result["transform_independent"]


def test_missing_patient_identity_withholds_bias_precision():
    row = agreement_for_pair("x", np.arange(400.), np.arange(400.)+1,
        primary_transform="kors", comparison_transform="inverse_dower")
    assert np.isnan(row["bias_ci_low"])
    assert row["bias"] == 1


def test_patient_index_selected_before_marker_missingness():
    frame = pd.DataFrame({"subject_id": [1, 1, 2, 2], "study_id": [10, 11, 20, 21],
        "extraction_status": ["failed", "ok", "ok", "ok"],
        "x": [np.nan, 3., 4., 5.], "sensitivity_inverse_dower__x": [np.nan, 4., 5., 6.]})
    result = agreement_table(frame, ["x"], primary_transform="kors", comparison_transform="inverse_dower",
        policy=AgreementPolicy(minimum_pairs=2))
    assert result.paired_count.iloc[0] == 1  # Failed index must not become its successful repeat.
    assert result.index_record_count.iloc[0] == 2
    assert result.extraction_ok_count.iloc[0] == 1
    assert result.paired_availability_fraction.iloc[0] == .5


def test_unknown_patient_token_does_not_authorize_independent_bias_ci():
    frame = pd.DataFrame({"subject_id": ["unknown", "known"], "study_id": [1, 2],
        "x": [1., 2.], "sensitivity_inverse_dower__x": [2., 4.]})
    result = agreement_table(frame, ["x"], primary_transform="kors", comparison_transform="inverse_dower",
        policy=AgreementPolicy(minimum_pairs=2))
    assert np.isnan(result.bias_ci_low.iloc[0])
    assert np.isnan(result.extraction_ok_count.iloc[0])
    assert result.paired_availability_fraction.iloc[0] == 1.


def test_too_few_pairs_abstains_rather_than_printing_a_bias():
    row = _pair("sparse", np.arange(20.0), np.arange(20.0) + 1.0)
    assert row["status"] == "too_few_pairs"
    assert row["agreement_class"] == "not_assessable"
    assert np.isnan(row["bias"])


def test_only_pairs_present_in_the_frame_are_reported():
    rng = np.random.default_rng(1)
    values = rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "alpha": values,
            "sensitivity_inverse_dower__alpha": values + 0.5,
            "beta": values,
        }
    )
    table = agreement_table(
        frame,
        ["alpha", "beta", "gamma"],
        primary_transform="kors",
        comparison_transform="inverse_dower",
    )
    assert list(table["marker_id"]) == ["alpha"]
    assert table.iloc[0]["comparison_transform"] == "inverse_dower"


def test_missing_values_do_not_enter_the_comparison():
    primary = np.concatenate([np.linspace(0.0, 1.0, 400), [np.nan] * 50])
    comparison = np.concatenate([np.linspace(0.0, 1.0, 400) + 2.0, [1.0] * 50])
    row = _pair("with_gaps", primary, comparison)
    assert row["paired_count"] == 400
    assert row["bias"] == pytest.approx(2.0, abs=1e-9)


def test_the_declared_independent_list_is_the_measured_one():
    """The four are a measurement, so the declaration has to match it.

    On the real store all four declared markers came back byte-identical and
    nothing else did: zero mismatches over forty pairs.
    """

    assert TRANSFORM_INDEPENDENT_FEATURE_IDS == frozenset(
        {
            "qt_interval_ms",
            "qtc_fridericia_ms",
            "tpeak_tend_ms",
            "cornell_voltage_surrogate",
        }
    )
