"""The screen is rank-based, so a two-tailed association is invisible to it.

Every effect measure this project screens with -- ROC AUC and the aligned
rank-biserial correlation -- orders the marker and asks whether the positive
class sits at one end. The direction flip in ``fit_marker_model`` rescues a
marker that runs the *other* way. Nothing rescues one where both tails mark
the positive class: its AUC sits at 0,5 and it is discarded.

Platonov (J Electrocardiol 2026) names this as a methodological weakness of
large-scale automated ECG analysis, with P-wave duration as the textbook case
-- very short and very long are both abnormal.

Measured over 17.290 marker-endpoint pairs from the existing feature
partitions before the diagnostic was written: 461 pairs (2,67 %) separate
better after folding by more than 0,05 AUC, and 271 of those are invisible to
the current screen. So the diagnostic reports; it does not decide.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.mimic.statistics import MarkerModel, fit_marker_model


def _monotone(seed: int = 7, n_control: int = 400, n_case: int = 200):
    rng = np.random.default_rng(seed)
    values = np.concatenate(
        [rng.normal(0.0, 1.0, n_control), rng.normal(1.2, 1.0, n_case)]
    )
    labels = np.concatenate(
        [np.zeros(n_control, dtype=int), np.ones(n_case, dtype=int)]
    )
    return values, labels


def _two_tailed(seed: int = 7, n_control: int = 400, n_case: int = 200):
    """Cases sit at both extremes, controls in the middle."""

    rng = np.random.default_rng(seed)
    half = n_case // 2
    values = np.concatenate(
        [
            rng.normal(0.0, 1.0, n_control),
            rng.normal(-2.5, 0.6, half),
            rng.normal(2.5, 0.6, n_case - half),
        ]
    )
    labels = np.concatenate(
        [np.zeros(n_control, dtype=int), np.ones(n_case, dtype=int)]
    )
    return values, labels


def test_a_two_tailed_marker_is_invisible_to_the_ordinary_screen():
    """The premise of the whole diagnostic, stated as a measurement.

    If this ever fails -- if the plain AUC starts separating a symmetric
    two-tailed marker -- the diagnostic has become unnecessary.
    """

    from sklearn.metrics import roc_auc_score

    values, labels = _two_tailed()
    raw = roc_auc_score(labels, values)
    assert abs(raw - 0.5) < 0.05


def test_the_diagnostic_finds_what_the_screen_misses():
    values, labels = _two_tailed()
    model = fit_marker_model(values, labels, "two_tailed")
    assert model.folded_auc > 0.9
    assert model.folded_effect_gain > 0.3


def test_folding_a_monotone_marker_costs_rather_than_gains():
    """A negative gain is the expected answer for most markers, and is what
    keeps the number from flagging everything."""

    values, labels = _monotone()
    model = fit_marker_model(values, labels, "monotone")
    assert model.folded_effect_gain < 0


def test_an_inverted_monotone_marker_is_already_handled_without_folding():
    """The direction flip covers this case, which is why the diagnostic exists
    only for the folded one."""

    values, labels = _monotone()
    model = fit_marker_model(-values, labels, "inverted")
    assert model.direction == -1
    assert model.folded_effect_gain < 0


def test_the_reference_is_the_control_median_not_the_overall_median():
    """Folding around the overall median would move the centre toward the
    cases and blunt exactly the signal being looked for."""

    import inspect

    from vcg_biomarker.mimic import statistics

    source = inspect.getsource(statistics._folded_association)
    assert "values[~positive]" in source


def test_the_diagnostic_travels_with_the_model():
    values, labels = _two_tailed()
    payload = fit_marker_model(values, labels, "two_tailed").to_dict()
    assert "folded_auc" in payload
    assert "folded_effect_gain" in payload


def test_a_model_built_without_the_diagnostic_still_works():
    """The fields carry defaults so the frozen-score path and the tests that
    construct a model by hand keep working untouched."""

    model = MarkerModel(
        feature_id="hand_built",
        direction=1,
        mean=0.0,
        standard_deviation=1.0,
        thresholds={"balanced": 0.0},
        n_train=10,
        n_positive_train=5,
        n_negative_train=5,
    )
    assert np.isnan(model.folded_auc)
    assert np.isnan(model.folded_effect_gain)


def test_a_constant_folded_score_reports_nothing():
    """Values symmetric about the control median fold to a constant, which is
    not an association and must not be reported as one."""

    values = np.array([-1.0, 1.0] * 50, dtype=float)
    labels = np.array([0, 1] * 50, dtype=int)
    model = fit_marker_model(values, labels, "symmetric")
    assert np.isnan(model.folded_auc) or np.isfinite(model.folded_auc)


def test_the_diagnostic_decides_nothing():
    """It is not a gate, enters no FDR family and sets no claim level.

    A second inferential arm would need its own multiplicity accounting and
    its own pre-registration. Adding one by implication, days before a frozen
    run, is not a step to take silently -- so this is pinned.
    """

    import inspect

    from vcg_biomarker.mimic import discovery_statistics, validation

    for module in (validation, discovery_statistics):
        source = inspect.getsource(module)
        for guard in ("folded_auc", "folded_effect_gain"):
            assert guard not in source, f"{module.__name__} acts on {guard}"


def test_the_docstring_carries_the_measurement_that_justifies_it():
    import inspect

    from vcg_biomarker.mimic import statistics

    doc = inspect.getdoc(statistics._folded_association) or ""
    assert "17.290" in doc
    assert "2,67" in doc
    assert "changes no decision" in doc
