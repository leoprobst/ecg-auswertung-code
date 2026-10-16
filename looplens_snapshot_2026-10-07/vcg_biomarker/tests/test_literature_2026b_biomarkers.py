"""The four markers that survived the extended 2026 search.

Seven were proposed. Three were already in the registry under other names and
one was withdrawn after measurement showed it 97,7 % linearly reconstructible
from its own two components. These tests pin both halves: that the four that
remain behave as described, and that the three withdrawals stay withdrawn --
because the reason each was dropped is a fact about the registry that a later
change could quietly invalidate.

Measured on 892 recordings of the training split before the module was
finished:

  repolarization_time_difference_ratio   95,5 % available, AUC 0,632 against
                                         reduced_lvef_strict, R^2 0,50
  t_azimuth_beatwise_dispersion         100,0 % available, AUC 0,440, R^2 0,34
  v6_r_peak_time_ms                      99,2 % available, R^2 0,32
  v6_v1_interpeak_interval_ms            94,6 % available, R^2 0,16
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.literature_2026b_biomarkers import (
    LITERATURE_2026B_SPECS,
    MINIMUM_BEATS_FOR_DISPERSION,
    Literature2026bBiomarker,
    literature_2026b_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_four_markers_are_registered():
    ids = literature_2026b_marker_ids()
    assert len(ids) == 4
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), Literature2026bBiomarker)


@pytest.mark.parametrize("name", ["v6_r_peak_time_ms", "v6_v1_interpeak_interval_ms"])
def test_native_timing_category_consistent_across_registry_definition_result(name, monkeypatch):
    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers as module
    from vcg_biomarker.mimic.feature_roles import feature_profile
    marker = BiomarkerRegistry.get(name)
    assert marker.category == "ecg"
    assert BiomarkerRegistry.definition(name).category == "ecg"
    assert feature_profile(name).modality == "ecg"
    # Missing numerical context deliberately exercises an abstention result;
    # category/provenance must remain correct even when no value is measurable.
    monkeypatch.setattr(module,"_context",lambda *args: None)
    result = marker.compute(None)
    assert result.category == "ecg"
    assert "published_method_equivalence=not_established" in result.computation_notes


def test_publication_numbers_not_claimed_as_implementation_validation():
    for name in literature_2026b_marker_ids():
        note = BiomarkerRegistry.definition(name).pathological_note
        assert "not a validated property of this implementation" in note


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(literature_2026b_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in literature_2026b_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_carries_a_reference_and_claims_no_cutoff():
    for spec in LITERATURE_2026B_SPECS:
        assert spec.references, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert definition.pathological_direction == "data_driven"
        assert "No diagnostic cutoff is claimed." in definition.pathological_note


# --- the three withdrawals must stay withdrawn ----------------------------


def test_the_qrs_three_dimensional_vti_is_already_extracted():
    """Withdrawn because ``magnitude_area_qrs`` is that integral.

    If it ever disappears, the Gupta reference ranges lose their anchor and the
    withdrawal has to be revisited.
    """

    definition = BiomarkerRegistry.definition("magnitude_area_qrs")
    assert "||r(t)||" in definition.formula
    assert definition.unit == "mV*s"


def test_the_pathological_q_count_is_already_extracted():
    """Withdrawn instead of the Selvester score, whose 53 criteria across six
    ECG-type variants cannot be sourced. This is the continuous substitute."""

    from vcg_biomarker.mimic.panels import CLASSICAL_FEATURE_IDS

    assert "classical_pathological_q_waves" in CLASSICAL_FEATURE_IDS


def test_the_scalar_p_terminal_force_is_already_extracted():
    from vcg_biomarker.mimic.panels import CLASSICAL_FEATURE_IDS

    assert "classical_p_terminal_force_v1" in CLASSICAL_FEATURE_IDS


def test_no_tpe_over_qt_ratio_was_added():
    """Withdrawn at R^2 = 0,977 against its own two components.

    Both components are curated panel members open to the combination search,
    so the analysis can already build it. Pinned so it is not re-added without
    re-measuring.
    """

    assert "tpeak_tend_to_qt_ratio" not in set(BiomarkerRegistry.available())
    assert "tpeak_tend_ms" in set(BiomarkerRegistry.available())
    assert "qt_interval_ms" in set(BiomarkerRegistry.available())


# --- the Salatzki ratio ---------------------------------------------------


class _Point:
    def __init__(self, sample: int, confidence: float = 1.0) -> None:
        self.sample = int(sample)
        self.confidence = float(confidence)


def _ratio(t_onset: int, r_peak: int, t_peak: int, j_point: int, fs: int = 500):
    """Drive the calculator through a stub context, so the arithmetic is
    tested without a recording."""

    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers as module

    class _Ctx:
        def __init__(self) -> None:
            self.fs = fs

        def marker_index(self, marker: str):
            return {
                "t_onset": t_onset,
                "r_peak": r_peak,
                "t_peak": t_peak,
                "j_point": j_point,
            }.get(marker)

    return module._repolarization_time_difference_ratio(_Ctx())


def test_the_ratio_is_the_quotient_of_the_two_durations():
    # R peak 100, T onset 200 -> 200 ms at 500 Hz.
    # QRS end 150, T peak 250 -> 200 ms. Quotient 1.
    value = _ratio(t_onset=200, r_peak=100, t_peak=250, j_point=150)
    assert value.value == pytest.approx(1.0)

    # Halve the denominator: T peak 200 -> 100 ms. Quotient 2.
    value = _ratio(t_onset=200, r_peak=100, t_peak=200, j_point=150)
    assert value.value == pytest.approx(2.0)


def test_both_intervals_are_taken_as_positive_durations():
    """Signed intervals would make the ratio negative for every ordinary beat,
    because the J point precedes the T peak. Pinned as a property."""

    value = _ratio(t_onset=200, r_peak=100, t_peak=250, j_point=150)
    assert value.value is not None and value.value > 0


def test_fiducials_out_of_order_abstain_rather_than_return_a_sign():
    # T onset before the R peak cannot happen in a beat; abstain.
    value = _ratio(t_onset=90, r_peak=100, t_peak=250, j_point=150)
    assert value.value is None
    assert "out of order" in (value.notes or "")


def test_a_missing_fiducial_is_named_in_the_abstention():
    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers as module

    class _Ctx:
        fs = 500

        def marker_index(self, marker: str):
            return None if marker == "t_peak" else 100

    value = module._repolarization_time_difference_ratio(_Ctx())
    assert value.value is None
    assert "T peak" in (value.notes or "")


# --- circular dispersion of the T direction -------------------------------


def _dispersion(directions):
    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers as module

    class _Ctx:
        pass

    ctx = _Ctx()
    setattr(ctx, "_literature_2026b_cache", {("beat_t_directions",): list(directions)})
    return module._t_azimuth_beatwise_dispersion(ctx)


def _unit(degrees: float) -> tuple[float, float]:
    radians = np.radians(degrees)
    return float(np.cos(radians)), float(np.sin(radians))


def test_identical_directions_disperse_by_nothing():
    value = _dispersion([_unit(40.0)] * 5)
    assert value.value == pytest.approx(0.0, abs=1e-9)


def test_opposite_directions_disperse_completely():
    value = _dispersion([_unit(0.0), _unit(180.0)] * 3)
    assert value.value == pytest.approx(1.0, abs=1e-9)


def test_the_dispersion_has_no_seam():
    """The property the whole marker turns on.

    Directions at -179 and +179 degrees are two degrees apart in the body. A
    standard deviation of the angle would call that a spread of 358; the
    resultant length must call it nearly nothing.
    """

    value = _dispersion([_unit(-179.0), _unit(179.0)] * 3)
    assert value.value is not None
    assert value.value < 0.01


def test_a_standard_deviation_of_the_angle_would_have_failed_that():
    """States the alternative that was rejected, as a measurement."""

    angles = np.array([-179.0, 179.0] * 3)
    assert float(np.std(angles)) > 170.0


def test_rotating_every_beat_does_not_change_the_dispersion():
    """A dispersion must measure spread, not position."""

    base = [_unit(10.0), _unit(50.0), _unit(90.0)]
    turned = [_unit(10.0 + 137.0), _unit(50.0 + 137.0), _unit(90.0 + 137.0)]
    assert _dispersion(base).value == pytest.approx(_dispersion(turned).value, abs=1e-9)


def test_too_few_beats_abstain():
    value = _dispersion([_unit(0.0)] * (MINIMUM_BEATS_FOR_DISPERSION - 1))
    assert value.value is None
    assert "three beats" in (value.notes or "")


def test_the_dispersion_stays_within_its_bounds():
    rng = np.random.default_rng(5)
    for _ in range(50):
        directions = [_unit(float(a)) for a in rng.uniform(-180, 180, 8)]
        value = _dispersion(directions)
        assert 0.0 <= value.value <= 1.0


# --- the R-peak timing pair -----------------------------------------------


def _lead_peak(window, lead="V6", fs=500):
    """Drive the peak finder over one synthetic lead window."""

    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers as module
    from vcg_biomarker.core.data_loader import LEAD_MAP

    samples = np.zeros((len(window), max(LEAD_MAP.values()) + 1), dtype=float)
    samples[:, LEAD_MAP[lead]] = np.asarray(window, dtype=float)

    class _Ctx:
        fs = 500
        ecg_12lead = samples

        def bounds(self, segment: str):
            return (0, len(window) - 1) if segment == "qrs" else None

    return module._lead_r_peak_sample(_Ctx(), lead)


def test_the_peak_is_the_dominant_positive_deflection():
    # A tall early R at index 3 and a small late bump at index 9.
    window = [0.0, 0.2, 0.8, 1.5, 0.6, -0.4, -0.2, 0.0, 0.1, 0.3, 0.1]
    assert _lead_peak(window) == 3


def test_a_terminal_rise_does_not_capture_the_peak():
    """The measured defect, pinned.

    Reading the *last* positive local maximum put the V6-minus-V1 interval at
    a median of -42 ms over 54 recordings -- V6 peaking before V1, which is
    backwards. The cause was ST elevation in V1 and V2 lifting the end of the
    window. Here the terminal rise is a local maximum and must be ignored.
    """

    window = [0.0, 0.5, 1.2, 0.7, 0.1, -0.3, 0.0, 0.2, 0.4, 0.3]
    assert _lead_peak(window) == 2


def test_an_entirely_negative_lead_has_no_r_peak():
    assert _lead_peak([0.0, -0.2, -1.0, -0.6, -0.1]) is None


def test_the_interpeak_interval_keeps_its_sign():
    """A magnitude would discard the direction the criterion is about."""

    spec = next(
        s for s in LITERATURE_2026B_SPECS if s.name == "v6_v1_interpeak_interval_ms"
    )
    assert "V6 R-peak time - V1 R-peak time" in spec.formula
    assert "abs" not in spec.formula.lower()


def test_v6_r_peak_time_is_the_projects_first_per_lead_activation_time():
    """Why the marker is not a duplicate of ``r_peak_timing_fraction``.

    That one is the position of the QRS *magnitude* peak as a dimensionless
    fraction -- a lead-free analogue of the intrinsicoid deflection, which no
    published cut-off in milliseconds can be held against. If it ever becomes
    a per-lead duration, this marker's justification has to be revisited.
    """

    definition = BiomarkerRegistry.definition("r_peak_timing_fraction")
    assert definition.unit == "dimensionless"
    assert BiomarkerRegistry.definition("v6_r_peak_time_ms").unit == "ms"


def test_the_pacing_markers_say_they_carry_no_threshold():
    for name in ("v6_r_peak_time_ms", "v6_v1_interpeak_interval_ms"):
        note = BiomarkerRegistry.definition(name).pathological_note
        assert "No diagnostic cutoff is claimed." in note


# --- documentation carries the measurements -------------------------------


def test_the_module_docstring_records_why_three_were_withdrawn():
    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers

    doc = literature_2026b_biomarkers.__doc__ or ""
    assert "magnitude_area_qrs" in doc
    assert "53 criteria" in doc
    assert "classical_p_terminal_force_v1" in doc
    assert "0,977" in doc


def test_the_peak_finder_records_the_measurement_that_changed_it():
    import inspect

    from vcg_biomarker.core.biomarkers import literature_2026b_biomarkers

    doc = inspect.getdoc(literature_2026b_biomarkers._lead_r_peak_sample) or ""
    assert "-42 ms" in doc
    assert "+17 ms" in doc
