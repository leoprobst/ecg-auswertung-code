"""Three markers taken from named 2026 papers, and what each one has to do.

The point of registering a marker from a publication is that the publication's
claim becomes testable. These tests pin the properties that make that possible:
the sign convention the age hypothesis depends on, the direction the visual
criterion names, and the abstention behaviour that keeps a marker from
inventing values it cannot support.

One proposal was dropped rather than built -- a binary flag for the published
SVG amyloid threshold -- because the analysis thresholds the continuous marker
itself. That decision is pinned too, so it is not silently reversed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.literature_2026_biomarkers import (
    LATE_QRS_SEGMENT,
    LATERAL_PRECORDIAL_LEADS,
    LITERATURE_2026_SPECS,
    Literature2026Biomarker,
    P_TERMINAL_FRACTION,
    _apicobasal_gradient,
    _p_terminal_progression,
    _terminal_qrs_speed_uniformity,
    literature_2026_marker_ids,
)
from vcg_biomarker.core.data_loader import LEAD_MAP
from vcg_biomarker.mimic.feature_roles import feature_profile
from vcg_biomarker.mimic.panels import default_feature_panels


BY_NAME = {spec.name: spec for spec in LITERATURE_2026_SPECS}
FS = 500


# --- a context that answers only what these markers ask ----------------------


class _Metrics:
    def __init__(self, qrst):
        self.qrst_area_vector_uv_ms = np.asarray(qrst, dtype=float)


class _Ctx:
    def __init__(
        self,
        *,
        p_by_lead=None,
        pr_baseline_mv=0.0,
        qrst=None,
        late_qrs=None,
        p_samples=60,
    ):
        self._p_by_lead = dict(p_by_lead or {})
        self._qrst = qrst
        self._late_qrs = late_qrs
        self.fs = FS
        self._p_samples = p_samples
        # One flat PR stretch per lead at the requested baseline.
        self.ecg_12lead = np.full((400, 12), float(pr_baseline_mv))

    def bounds(self, segment):
        return (0, 39) if segment == "pr" else None

    def ecg_segment(self, lead, segment):
        if segment != "p":
            return None
        amplitude = self._p_by_lead.get(lead)
        if amplitude is None:
            return None
        # A P wave whose terminal portion sits at the requested amplitude.
        values = np.zeros(self._p_samples, dtype=float)
        cut = int(round(self._p_samples * (1.0 - P_TERMINAL_FRACTION)))
        values[cut:] = amplitude / 1000.0  # the accessor returns millivolts
        return values

    def geh_area_metrics(self):
        return None if self._qrst is None else _Metrics(self._qrst)

    def vcg_segment_origin_corrected(self, segment):
        return self._late_qrs if segment == LATE_QRS_SEGMENT else None


def _flat_terminal(**amplitudes_uv):
    return _Ctx(p_by_lead=amplitudes_uv)


# --- the visual criterion -----------------------------------------------------


def test_a_fading_terminal_p_wave_gives_a_negative_slope():
    """"Poor progression" is the criterion's own wording: the left-atrial
    component fades across the lateral chest leads instead of holding up."""

    value = _p_terminal_progression(_flat_terminal(V4=60.0, V5=40.0, V6=20.0))
    assert value.value is not None
    assert value.value < 0


def test_a_rising_terminal_p_wave_gives_a_positive_slope():
    value = _p_terminal_progression(_flat_terminal(V4=20.0, V5=40.0, V6=60.0))
    assert value.value == pytest.approx(20.0, abs=1e-6)


def test_a_flat_terminal_p_wave_gives_no_slope():
    value = _p_terminal_progression(_flat_terminal(V4=50.0, V5=50.0, V6=50.0))
    assert value.value == pytest.approx(0.0, abs=1e-9)


def test_the_slope_is_fitted_not_differenced():
    """A least-squares fit over three leads lets a single noisy lead move the
    answer less than V6 minus V4 would."""

    fitted = _p_terminal_progression(_flat_terminal(V4=0.0, V5=100.0, V6=20.0))
    assert fitted.value == pytest.approx(10.0, abs=1e-6)  # not 20.0


def test_the_pr_segment_is_the_zero_line():
    """The same reference the ST markers use and the one a reader takes by eye.
    Shifting the whole tracing must not change a slope."""

    shifted = _Ctx(
        p_by_lead={"V4": 60.0, "V5": 40.0, "V6": 20.0}, pr_baseline_mv=0.5
    )
    plain = _flat_terminal(V4=60.0, V5=40.0, V6=20.0)
    assert shifted.ecg_12lead[0, 0] != plain.ecg_12lead[0, 0]
    assert _p_terminal_progression(shifted).value == pytest.approx(
        _p_terminal_progression(plain).value, abs=1e-6
    )


def test_one_missing_lateral_lead_withholds_the_answer():
    """Two leads make a line through any two points. The criterion is about a
    progression across three."""

    assert _p_terminal_progression(_flat_terminal(V4=50.0, V5=40.0)).value is None


def test_the_criterion_reads_the_lateral_precordial_leads():
    assert LATERAL_PRECORDIAL_LEADS == ("V4", "V5", "V6")
    assert all(lead in LEAD_MAP for lead in LATERAL_PRECORDIAL_LEADS)


# --- the apicobasal gradient --------------------------------------------------


def test_the_gradient_is_the_inferior_component_with_plus_y_inferior():
    """The whole age hypothesis rides on this sign.

    The project's axis convention is Frank-compatible with +Y inferior, so a
    gradient pointing at the apex is positive and an inverted one is negative.
    Reading the wrong component, or dropping the sign, would turn "inverts with
    age" into an unmeasurable statement.
    """

    from vcg_biomarker.core.vcg_geometry import VCG_AXIS_CONVENTION

    assert "+Y inferior" in VCG_AXIS_CONVENTION
    apexward = _apicobasal_gradient(_Ctx(qrst=[10.0, 250.0, -30.0]))
    baseward = _apicobasal_gradient(_Ctx(qrst=[10.0, -250.0, -30.0]))
    assert apexward.value == pytest.approx(250.0)
    assert baseward.value == pytest.approx(-250.0)


def test_the_gradient_keeps_its_sign_rather_than_a_magnitude():
    """A magnitude paired with an angle cannot state "inverted" as one ordered
    quantity, which is what the hypothesis needs."""

    values = [
        _apicobasal_gradient(_Ctx(qrst=[0.0, y, 0.0])).value
        for y in (-100.0, 0.0, 100.0)
    ]
    assert values == sorted(values)
    assert values[0] < 0 < values[2]


def test_the_gradient_equals_magnitude_times_cosine_of_elevation():
    """Stated in the docstring as the reason it is not screenable today: the
    number exists, but only inside another marker's provenance metadata."""

    from vcg_biomarker.core.vcg_geometry import vcg_elevation_deg

    vector = np.array([30.0, 120.0, -40.0])
    magnitude = float(np.linalg.norm(vector))
    elevation = vcg_elevation_deg(vector)
    reconstructed = magnitude * np.cos(np.radians(elevation))
    assert _apicobasal_gradient(_Ctx(qrst=vector)).value == pytest.approx(
        reconstructed, rel=1e-9
    )


def test_the_gradient_abstains_without_area_metrics():
    assert _apicobasal_gradient(_Ctx(qrst=None)).value is None


# --- the terminal QRS entropy -------------------------------------------------


def test_bursty_motion_scores_lower_than_a_constant_rate():
    """The direction, pinned, because the first draft had it backwards.

    A constant-rate ramp spreads its steps evenly and reaches exactly 1; noise
    on the same ramp concentrates the distance into fewer steps and lowers the
    value. Calling this "dispersion entropy" implied the reverse and would have
    had every later reader read its sign wrong.
    """

    steps = np.linspace(0.0, 1.0, 40)
    smooth = np.column_stack([steps, steps * 0.5, steps * 0.25])
    rng = np.random.default_rng(11)
    fragmented = smooth + rng.normal(0.0, 0.05, smooth.shape)
    even = _terminal_qrs_speed_uniformity(_Ctx(late_qrs=smooth)).value
    bursty = _terminal_qrs_speed_uniformity(_Ctx(late_qrs=fragmented)).value
    assert even == pytest.approx(1.0, abs=1e-9)
    assert bursty < even


def test_the_name_does_not_promise_the_opposite_direction():
    """Nothing in the marker set may be called entropy or dispersion while
    falling as its subject rises."""

    for spec in LITERATURE_2026_SPECS:
        if "uniformity" in spec.name:
            assert "entropy" not in spec.name
            assert "dispersion" not in spec.name


def test_the_entropy_stays_inside_its_declared_unit():
    rng = np.random.default_rng(3)
    for seed_shift in range(4):
        points = rng.normal(0.0, 1.0, (30, 3)) + seed_shift
        value = _terminal_qrs_speed_uniformity(_Ctx(late_qrs=points)).value
        assert value is None or 0.0 <= value <= 1.0
    assert BY_NAME["terminal_qrs_speed_uniformity"].unit == "fraction"


def test_a_still_trajectory_has_no_entropy_to_report():
    assert (
        _terminal_qrs_speed_uniformity(_Ctx(late_qrs=np.zeros((30, 3)))).value
        is None
    )


def test_the_entropy_reads_the_late_qrs_third():
    assert LATE_QRS_SEGMENT == "qrs_late"


# --- registration, roles, panels ----------------------------------------------


def test_every_marker_is_registered_once():
    available = set(BiomarkerRegistry.available())
    for name in literature_2026_marker_ids():
        assert name in available
        assert isinstance(BiomarkerRegistry.get(name), Literature2026Biomarker)


def test_three_markers_and_no_more():
    assert len(LITERATURE_2026_SPECS) == 3


def test_every_marker_names_the_paper_it_came_from():
    """A marker registered from a publication whose claim cannot be traced is
    a marker nobody can argue with."""

    for spec in LITERATURE_2026_SPECS:
        assert spec.references, spec.name
        assert any("2026" in reference for reference in spec.references), spec.name


def test_the_speculative_marker_says_it_is_speculative():
    """Its source measured synthetic injected labels. Anyone reading the name
    would assume otherwise."""

    spec = BY_NAME["terminal_qrs_speed_uniformity"]
    assert "synthetic" in spec.evidence_note
    assert "guess" in spec.evidence_note


def test_the_markers_reach_the_discovery_reserve_only():
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    for name in literature_2026_marker_ids():
        assert name in panels["discovery_reserve"], name
        assert name not in panels["primary_core"], name
        assert name not in panels["sentinel"], name


def test_the_curated_panels_keep_their_sizes():
    panels = {panel.panel_id: panel.feature_ids for panel in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240


def test_the_markers_are_declared_post_outcome():
    for name in literature_2026_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.scientific_tier == "post_outcome_discovery", name


def test_a_failing_calculator_abstains_rather_than_stopping_the_panel():
    spec = BY_NAME["p_terminal_progression_v4_v6"]
    broken = type(spec)(
        spec.name,
        spec.family,
        spec.description,
        spec.formula,
        spec.unit,
        lambda ctx: (_ for _ in ()).throw(ValueError("deliberate")),
        spec.evidence_note,
    )

    class _Delineation:
        beats = ()
        n_samples = 0
        representative_beat = None

    result = Literature2026Biomarker(broken).compute(
        delineation=_Delineation(), fs=FS
    )
    assert result.value is None
    assert "ValueError" in result.computation_notes


# --- what was deliberately not built ------------------------------------------


def test_no_binary_copy_of_the_published_svg_threshold_was_added():
    """Stabenau et al. report that no amyloid patient had an SVG magnitude
    above 50 mV*ms, and the project's own distribution puts that threshold
    between the median and the third quartile -- so the rule is testable. It is
    testable *from the continuous marker*, which the analysis already
    thresholds when it computes operating points. A binary copy would spend an
    extraction slot to encode something the analysis can apply anyway."""

    names = set(BiomarkerRegistry.available())
    for invented in (
        "svg_magnitude_amyloid_threshold",
        "svg_magnitude_below_amyloid_threshold",
        "svg_magnitude_gt_50_mv_ms",
    ):
        assert invented not in names
    assert "svg_magnitude" in names
