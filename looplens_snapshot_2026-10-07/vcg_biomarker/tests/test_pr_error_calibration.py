from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.pr_error_calibration import (
    estimate_error_moments, geometry_value, match_reference_beats,
)


def test_within_record_covariance_does_not_confuse_between_record_bias():
    rows = []
    for record in range(40):
        for error in (-1., 0., 1.):
            rows.append({"record_id": str(record), "status": "measured",
                         "pr_error_ms": record*10 + error, "geometry_error": -record + 2*error})
    result = estimate_error_moments(pd.DataFrame(rows), bootstrap=100)
    assert result["pr_error_variance_ms2"] == pytest.approx(1.)
    assert result["error_covariance"] == pytest.approx(2.)
    assert result["geometry_error_variance"] == pytest.approx(4.)
    assert result["within_record_df"] == 80
    assert result["record_cluster_bootstrap_95ci"][0] == pytest.approx([1., 1.])
    assert result["approved_for_correction"] is False


def test_many_beats_are_not_many_independent_patients():
    frame = pd.DataFrame({"record_id": "one", "status": "measured",
                          "pr_error_ms": np.arange(100), "geometry_error": np.arange(100)})
    assert estimate_error_moments(frame)["status"] == "non_estimable"


def wave(onset, peak, offset):
    return SimpleNamespace(onset=onset, peak=peak, offset=offset)


def test_reference_pairing_never_bridges_intervening_qrs_or_multiple_p():
    qrs = [wave(100, 120, 140), wave(300, 320, 340)]
    p = [wave(50, 60, 70)]
    pairs = match_reference_beats({"p": p, "qrs": qrs})
    assert len(pairs) == 1 and pairs[0][1].peak == 120
    p.append(wave(80, 85, 90))
    assert not match_reference_beats({"p": p, "qrs": qrs})


def test_geometry_kernel_is_existing_formula_and_short_segments_abstain():
    from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import _qrs_initial_terminal_energy_balance
    points = np.random.default_rng(7).normal(size=(100, 3))
    expected = _qrs_initial_terminal_energy_balance(SimpleNamespace(fs=500, vcg_segment=lambda _: points[10:61])).value
    assert geometry_value(points, 10, 60, 500) == expected
    assert geometry_value(points, 10, 11, 500) is None
