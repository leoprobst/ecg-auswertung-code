"""Two markers from the third propagation research round: uniform slow conduction.

The round is in RECHERCHE_EP_RUNDE3_2026-09-12.md at the project root. These
tests pin the rule (flanked slow stretches only), the behaviour on built
loops, and the measurements that shaped the text -- including the prediction
the data reversed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import conduction_uniformity_biomarkers as m
from vcg_biomarker.core.biomarkers.conduction_uniformity_biomarkers import (
    CONDUCTION_UNIFORMITY_SPECS,
    MINIMUM_RUN_SAMPLES,
    SLOW_SHARE,
    ConductionUniformityBiomarker,
    conduction_uniformity_marker_ids,
    slow_runs,
)


# --- registration and panel boundaries ------------------------------------


def test_two_markers_are_registered():
    ids = conduction_uniformity_marker_ids()
    assert len(ids) == 2
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), ConductionUniformityBiomarker)


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(conduction_uniformity_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in conduction_uniformity_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_its_refutation():
    for spec in CONDUCTION_UNIFORMITY_SPECS:
        assert "Refuted if" in spec.thesis, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert "No diagnostic cutoff is claimed." in definition.pathological_note


def test_the_research_paper_exists_and_names_the_markers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "RECHERCHE_EP_RUNDE3_2026-09-12.md").read_text(encoding="utf-8")
    for name in conduction_uniformity_marker_ids():
        assert name in text, name


# --- the rule -------------------------------------------------------------


def test_only_flanked_slow_stretches_count():
    speed = np.array([0.1, 0.2, 1.0, 0.9, 0.1, 0.1, 0.1, 0.8, 1.0, 0.5, 0.1, 0.05])
    assert slow_runs(speed) == [(4, 7)]           # the ramps at both ends are not runs


def test_a_single_burst_loop_has_no_run():
    assert slow_runs(np.array([0.1, 0.5, 1.0, 0.5, 0.1])) == []
    assert slow_runs(np.zeros(10)) == []


def test_the_threshold_is_relative_to_the_maximum():
    speed = np.array([0.1, 1.0, 0.2, 0.2, 1.0, 0.1])
    assert slow_runs(speed) == [(2, 4)]
    assert slow_runs(speed * 1000.0) == [(2, 4)]  # amplitude invariance
    assert slow_runs(speed, share=0.1) == []      # 0.2 is not below a tenth of the maximum


# --- a stub context -------------------------------------------------------


class _Ctx:
    def __init__(self, speed, fs=500):
        self._speed = np.asarray(speed, dtype=float)
        self.fs = fs


def _run(calculator, ctx):
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as lp

    original = lp._speed
    try:
        lp._speed = lambda c: c._speed
        return calculator(ctx)
    finally:
        lp._speed = original


def test_the_longest_run_is_reported_in_milliseconds():
    speed = np.concatenate([[1.0], np.full(3, 0.1), [1.0], np.full(6, 0.1), [1.0]])
    value = _run(m._longest_slow_run_ms, _Ctx(speed, fs=500))
    assert value.value == pytest.approx(6 * 2.0)       # six samples at 500 Hz


def test_a_loop_without_a_flanked_run_reads_zero():
    value = _run(m._longest_slow_run_ms, _Ctx([0.1, 0.5, 1.0, 0.5, 0.1]))
    assert value.value == pytest.approx(0.0)


def test_the_coefficient_of_variation_reads_a_plateau_low_and_a_trough_high():
    plateau = np.concatenate([[1.0], np.full(8, 0.1), [1.0]])
    trough = np.concatenate([[1.0], [0.2, 0.1, 0.05, 0.02, 0.05, 0.1, 0.2, 0.24], [1.0]])
    low = _run(m._slow_run_speed_cv, _Ctx(plateau)).value
    high = _run(m._slow_run_speed_cv, _Ctx(trough)).value
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high > 0.5


def test_a_short_run_gives_no_coefficient_of_variation():
    speed = np.concatenate([[1.0], np.full(MINIMUM_RUN_SAMPLES - 1, 0.1), [1.0]])
    value = _run(m._slow_run_speed_cv, _Ctx(speed))
    assert value.value is None
    assert "four samples" in (value.notes or "")


def test_unavailable_speed_abstains():
    class _NoSpeed:
        _speed = None
        fs = 500

    assert _run(m._longest_slow_run_ms, _NoSpeed()).value is None
    assert _run(m._slow_run_speed_cv, _NoSpeed()).value is None


# --- documentation carries the measurements -------------------------------


def test_the_module_records_the_reversed_prediction_and_the_refutation():
    doc = " ".join((m.__doc__ or "").split())      # wrapping is not content
    assert "10 against 2 ms" in doc                 # right bundle branch block, predicted not to
    assert "The error was the prediction, not the marker" in doc
    assert "0,268 against 0,266" in doc             # the coefficient of variation in old infarction
    assert "if that fails the marker goes" in doc
    assert SLOW_SHARE == pytest.approx(0.25)
