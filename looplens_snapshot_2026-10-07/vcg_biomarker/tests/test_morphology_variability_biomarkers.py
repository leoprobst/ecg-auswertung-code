"""Beat-to-beat morphology variability, measured continuously.

The published measure -- the share of P waves matching the dominant morphology,
90,4 % in paroxysmal fibrillation against 95,5 % in health -- cannot be carried
into this cohort. A recording here holds 13 beats at the median, so a share
moves in steps of 7,7 percentage points and the whole reported effect is 5,1.
The tests below therefore assert the properties a *continuous* substitute has
to have, and each one names the weakness it guards:

* scale-freeness, which ``p_loop_template_instability`` lacks -- the test
  measures the existing marker on the same record and asserts that it does
  move when the amplitudes double, so the fix is aimed at a real defect and
  not at a supposed one;
* a similarity that reads shape rather than offset;
* a statistic that separates two populations from even scatter, which one
  median cannot;
* measurability on the TQ window where no P wave exists, which is the only
  reason the fourth marker is worth its extraction time.

The panel tests guard the displacement rule the segment-free module ran into:
the curated panels have fixed sizes, so a marker admitted to the candidate pool
pushes one out of the sentinel panel. Everything here must land in the
discovery reserve and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _p_loop_extra
from vcg_biomarker.core.biomarkers.morphology_variability_biomarkers import (
    ATRIAL_WINDOW,
    MAXIMUM_DISSIMILARITY_RATIO,
    MINIMUM_BEATS,
    MINIMUM_BIMODALITY_BEATS,
    MINIMUM_SIMILARITY_SPAN,
    MINIMUM_TEMPLATE_SCALE_MV,
    MINIMUM_TQ_WINDOW_MS,
    MORPHOLOGY_VARIABILITY_SPECS,
    P_WINDOW,
    RESAMPLE_POINTS,
    MorphologyVariabilityBiomarker,
    _dominant_similarities,
    _p_loops,
    _template_dissimilarity_ratio,
    _tq_loops,
    morphology_variability_marker_ids,
)
from vcg_biomarker.mimic.feature_roles import (
    BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS,
    feature_profile,
)
from vcg_biomarker.mimic.panels import default_feature_panels


BY_NAME = {spec.name: spec for spec in MORPHOLOGY_VARIABILITY_SPECS}

FS = 500
BEAT_SAMPLES = 400  # 800 ms at 500 Hz
P_ONSET = 20
P_LENGTH = 40
R_ONSET = 120
T_OFFSET = 250


# --- a record the markers can read -------------------------------------------


class _Point:
    def __init__(self, sample, confidence=0.9):
        self.sample = int(sample)
        self.confidence = float(confidence)


class _Status:
    def __init__(self, value):
        self.value = value


class _Beat:
    """Only the fiducials these markers read, and their confidences."""

    def __init__(
        self,
        *,
        p_onset=None,
        p_offset=None,
        r_onset=None,
        t_offset=None,
        onset_confidence=0.9,
        t_confidence=0.9,
        t_present=True,
    ):
        self.p_onset = None if p_onset is None else _Point(p_onset)
        self.p_offset = None if p_offset is None else _Point(p_offset)
        self.r_onset = (
            None if r_onset is None else _Point(r_onset, onset_confidence)
        )
        self.t_offset = (
            None if t_offset is None else _Point(t_offset, t_confidence)
        )
        self.r_offset = None
        self.q_peak = None
        self.s_peak = None
        self.r_peak = None
        self._t_present = bool(t_present)

    def resolved_wave_status(self, wave):
        if wave == "t":
            return _Status("present" if self._t_present else "absent")
        return _Status("present")


class _Delineation:
    def __init__(self, beats):
        self.beats = list(beats)
        self.n_samples = 0
        self.representative_beat = beats[0] if beats else None


class _Ctx:
    """A context that answers exactly what these markers ask of it."""

    def __init__(self, record, beats, fs=FS):
        self._record_vcg_xyz = np.asarray(record, dtype=float)
        self.delineation = _Delineation(beats)
        self.fs = fs

    def vcg_segment(self, segment):
        """Only ``_p_loop_extra`` asks, and only to decide it may proceed."""

        if segment == "p":
            return self._record_vcg_xyz[P_ONSET : P_ONSET + P_LENGTH]
        return None


def _p_shape(kind="a", amplitude=0.1, samples=P_LENGTH):
    """A small closed P loop. ``kind`` picks between two distinct shapes."""

    t = np.linspace(0.0, 2.0 * np.pi, samples)
    if kind == "a":
        return amplitude * np.column_stack(
            [np.sin(t), 0.6 * np.sin(2.0 * t), 0.3 * np.cos(t)]
        )
    return amplitude * np.column_stack(
        [0.4 * np.sin(t), -np.cos(t), 0.9 * np.sin(3.0 * t)]
    )


def _tq_shape(kind="a", amplitude=0.05, samples=T_OFFSET + BEAT_SAMPLES - T_OFFSET):
    t = np.linspace(0.0, 2.0 * np.pi, samples)
    if kind == "a":
        return amplitude * np.column_stack(
            [np.sin(t), 0.5 * np.cos(t), 0.2 * np.sin(2.0 * t)]
        )
    return amplitude * np.column_stack(
        [np.cos(3.0 * t), -0.8 * np.sin(t), 0.6 * np.cos(t)]
    )


def _record(
    p_shapes,
    *,
    tq_shapes=None,
    with_p=True,
    onset_confidence=0.9,
    t_present=True,
    t_offset=T_OFFSET,
    fs=FS,
):
    """Build a synthetic record and its beats.

    One beat every 800 ms. The P loop sits at 40-120 ms, the QRS onset at
    240 ms and the T offset at 500 ms, so each beat's TQ window runs from its
    own T offset to the next beat's QRS onset -- 540 ms, a realistic diastole.
    """

    count = len(p_shapes)
    record = np.zeros(((count + 1) * BEAT_SAMPLES, 3), dtype=float)
    beats = []
    for index, shape in enumerate(p_shapes):
        base = index * BEAT_SAMPLES
        record[base + P_ONSET : base + P_ONSET + len(shape)] = shape
        if tq_shapes is not None:
            window = tq_shapes[index]
            start = base + t_offset
            record[start : start + len(window)] = window
        beats.append(
            _Beat(
                p_onset=(base + P_ONSET) if with_p else None,
                p_offset=(base + P_ONSET + len(shape) - 1) if with_p else None,
                r_onset=base + R_ONSET,
                t_offset=base + t_offset,
                onset_confidence=onset_confidence,
                t_present=t_present,
            )
        )
    # A closing beat, so every beat above has a QRS onset after its T offset.
    beats.append(
        _Beat(r_onset=count * BEAT_SAMPLES + R_ONSET, onset_confidence=onset_confidence)
    )
    return _Ctx(record, beats, fs=fs)


def _value(name, ctx):
    return BY_NAME[name].calculator(ctx).value


def _identical(count=8, kind="a", amplitude=0.1):
    return [_p_shape(kind, amplitude) for _ in range(count)]


def _scattered(count=8, amplitude=0.1, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    base = _p_shape("a", amplitude)
    return [base + rng.normal(0.0, spread, base.shape) for _ in range(count)]


def _two_populations(major=6, minor=3, amplitude=0.1):
    return [_p_shape("a", amplitude) for _ in range(major)] + [
        _p_shape("b", amplitude) for _ in range(minor)
    ]


# --- scale-freeness: the defect being fixed -----------------------------------


def test_the_existing_marker_moves_when_every_amplitude_doubles():
    """The weakness the ratio exists to remove, measured rather than assumed.

    ``p_loop_template_instability`` is reported in mV, so two recordings with
    identical relative variability but different P amplitude give different
    values. Without this test the new markers would be answering a supposed
    problem.
    """

    ctx = _record(_scattered(seed=1))
    doubled = _Ctx(ctx._record_vcg_xyz * 2.0, ctx.delineation.beats)
    single = _p_loop_extra(ctx, "p_loop_template_instability").value
    double = _p_loop_extra(doubled, "p_loop_template_instability").value
    assert single is not None and double is not None
    assert double == pytest.approx(2.0 * single, rel=1e-6)


def test_the_p_dissimilarity_ratio_does_not_move_when_amplitudes_double():
    ctx = _record(_scattered(seed=1))
    doubled = _Ctx(ctx._record_vcg_xyz * 2.0, ctx.delineation.beats)
    single = _value("p_loop_template_dissimilarity_ratio", ctx)
    double = _value("p_loop_template_dissimilarity_ratio", doubled)
    assert single is not None
    assert double == pytest.approx(single, rel=1e-9)


def test_the_tq_dissimilarity_ratio_does_not_move_when_amplitudes_double():
    shapes = [_tq_shape("a") for _ in range(6)]
    rng = np.random.default_rng(3)
    shapes = [shape + rng.normal(0.0, 0.004, shape.shape) for shape in shapes]
    ctx = _record(_identical(6), tq_shapes=shapes)
    doubled = _Ctx(ctx._record_vcg_xyz * 2.0, ctx.delineation.beats)
    single = _value("tq_template_dissimilarity_ratio", ctx)
    double = _value("tq_template_dissimilarity_ratio", doubled)
    assert single is not None
    assert double == pytest.approx(single, rel=1e-9)


def test_the_similarity_does_not_move_when_amplitudes_double():
    ctx = _record(_scattered(seed=2))
    doubled = _Ctx(ctx._record_vcg_xyz * 2.0, ctx.delineation.beats)
    single = _value("p_loop_dominant_similarity_mean", ctx)
    double = _value("p_loop_dominant_similarity_mean", doubled)
    assert single is not None
    assert double == pytest.approx(single, rel=1e-9)


def test_a_shared_offset_does_not_change_the_similarity():
    """What centring each loop buys: shape is compared, not position.

    Without it a common offset raises every cosine towards one and compresses
    the range the marker has to resolve -- and the offset is a property of the
    lead transform, not of the atrium.
    """

    shapes = _scattered(seed=4)
    plain = _record(shapes)
    shifted = _record([shape + np.array([0.5, -0.3, 0.2]) for shape in shapes])
    assert _value("p_loop_dominant_similarity_mean", shifted) == pytest.approx(
        _value("p_loop_dominant_similarity_mean", plain), rel=1e-9
    )


# --- the measurements themselves ---------------------------------------------


def test_identical_loops_have_no_dissimilarity():
    assert _value("p_loop_template_dissimilarity_ratio", _record(_identical())) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_identical_loops_have_a_similarity_of_one():
    assert _value("p_loop_dominant_similarity_mean", _record(_identical())) == (
        pytest.approx(1.0, abs=1e-9)
    )


def test_more_scattered_loops_give_a_larger_ratio():
    quiet = _value("p_loop_template_dissimilarity_ratio", _record(_scattered(spread=0.002, seed=5)))
    loud = _value("p_loop_template_dissimilarity_ratio", _record(_scattered(spread=0.02, seed=5)))
    assert quiet is not None and loud is not None
    assert loud > quiet


def test_similarity_falls_as_scatter_rises():
    quiet = _value("p_loop_dominant_similarity_mean", _record(_scattered(spread=0.002, seed=6)))
    loud = _value("p_loop_dominant_similarity_mean", _record(_scattered(spread=0.03, seed=6)))
    assert quiet is not None and loud is not None
    assert loud < quiet


def test_two_distinct_shapes_are_less_similar_than_one_shape_with_noise():
    split = _value("p_loop_dominant_similarity_mean", _record(_two_populations()))
    scatter = _value("p_loop_dominant_similarity_mean", _record(_scattered(count=9, seed=7)))
    assert split is not None and scatter is not None
    assert split < scatter


def test_the_similarity_is_a_mean_because_a_median_would_miss_the_minority():
    """Six loops of one shape and three of another return exactly 1,0 under a
    median: a median is blind to a minority smaller than half, and a minority
    of beats with a different atrial origin is this marker's whole subject.
    The published quantity is a share, which is a mean of indicators, so its
    continuous relaxation is a mean."""

    loops = _p_loops(_record(_two_populations(major=6, minor=3)))
    similarities = _dominant_similarities(loops)
    assert similarities is not None
    assert float(np.median(similarities)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.mean(similarities)) < 0.95


# --- diffuse scatter against two populations ---------------------------------


def test_even_scatter_stays_near_the_uniform_expectation():
    """Evenly scattered beats spread their similarities evenly.

    With n similarities the largest of the n-1 gaps is then about 1/(n-1) of
    the span. Eight beats give seven similarities, so the expectation is
    about 0,17; the bound below leaves generous room for the randomness of a
    single draw without admitting a split.
    """

    value = _value("p_loop_similarity_bimodality", _record(_scattered(count=8, seed=8)))
    assert value is not None
    assert value < 0.5


def test_two_populations_raise_the_bimodality_far_above_even_scatter():
    split = _value("p_loop_similarity_bimodality", _record(_two_populations(major=6, minor=3)))
    scatter = _value("p_loop_similarity_bimodality", _record(_scattered(count=9, seed=9)))
    assert split is not None and scatter is not None
    assert split > 0.7
    assert split > scatter


def test_a_single_deviant_beat_is_reported_as_a_split():
    """Intended, and stated so the reader is not surprised by it.

    One atrial ectopic among sinus beats is a two-population recording whose
    minor population has one member. The median-based markers beside this one
    are deliberately blind to it, which is why this one must not be.
    """

    shapes = _scattered(count=8, spread=0.002, seed=10)
    shapes[-1] = _p_shape("b")
    value = _value("p_loop_similarity_bimodality", _record(shapes))
    assert value is not None
    assert value > 0.7


def test_identical_loops_are_not_called_bimodal():
    """A span below the floor is answered with zero, not with an abstention.

    "No second population was found" is a true statement about identical
    beats; abstaining would lose it, and dividing by a span of zero would
    turn measurement noise into a maximal reading.
    """

    assert _value("p_loop_similarity_bimodality", _record(_identical())) == 0.0


def test_the_similarity_span_floor_is_small_enough_to_leave_real_scatter_alone():
    assert 0.0 < MINIMUM_SIMILARITY_SPAN <= 0.05


# --- abstention ---------------------------------------------------------------


def test_two_loops_are_not_enough_for_a_template():
    """With two beats the median loop is their midpoint and every distance is
    half their separation, which states nothing the pair does not."""

    assert _value("p_loop_template_dissimilarity_ratio", _record(_identical(2))) is None
    assert MINIMUM_BEATS == 3


def test_three_loops_are_enough():
    assert _value("p_loop_template_dissimilarity_ratio", _record(_identical(3))) is not None


def test_a_template_below_the_amplitude_floor_abstains():
    """A template of 10 uV is a baseline, and dividing by it would turn noise
    into an arbitrarily large ratio -- the one way a scale-free measure fails."""

    curves = [np.full((RESAMPLE_POINTS, 3), 0.0) for _ in range(5)]
    tiny = np.linspace(0.0, MINIMUM_TEMPLATE_SCALE_MV / 10.0, RESAMPLE_POINTS)
    curves = [np.column_stack([tiny, tiny * 0.0, tiny * 0.0]) for _ in range(5)]
    assert _template_dissimilarity_ratio(curves).value is None


def test_the_ratio_is_capped_where_it_is_formed():
    """The plausibility audit bounds the ``ratio`` unit; the cap belongs at
    the point of formation, which is what its documentation requires."""

    axis = np.linspace(-1.0, 1.0, RESAMPLE_POINTS)
    template = np.column_stack([axis * 0.02, axis * 0.0, axis * 0.0])
    displacement = np.full((RESAMPLE_POINTS, 3), 5.0)
    curves = [
        template + displacement,
        template - displacement,
        template + displacement,
        template - displacement,
        template,
    ]
    value = _template_dissimilarity_ratio(curves).value
    assert value == pytest.approx(MAXIMUM_DISSIMILARITY_RATIO)


def test_bimodality_needs_six_loops():
    assert MINIMUM_BIMODALITY_BEATS == 6
    assert _value("p_loop_similarity_bimodality", _record(_scattered(count=5, seed=11))) is None
    assert _value("p_loop_similarity_bimodality", _record(_scattered(count=6, seed=11))) is not None


def test_a_record_without_p_fiducials_yields_no_p_markers():
    ctx = _record(_identical(8), with_p=False)
    for name in ("p_loop_template_dissimilarity_ratio", "p_loop_dominant_similarity_mean", "p_loop_similarity_bimodality"):
        assert _value(name, ctx) is None


# --- the TQ window ------------------------------------------------------------


def test_the_tq_window_runs_from_the_t_offset_to_the_next_qrs_onset():
    ctx = _record(_identical(4), tq_shapes=[_tq_shape("a") for _ in range(4)])
    loops = _tq_loops(ctx)
    assert len(loops) == 4
    assert all(loop.shape == (RESAMPLE_POINTS, 3) for loop in loops)


def test_a_flat_window_is_not_a_loop():
    """A window with no deflection has no arc length and therefore no shape.

    This is also why the fixture above has to fill the windows: in sinus
    rhythm the TQ window contains the next beat's P wave, so a synthetic
    record whose last window has no successor P wave leaves that window flat.
    """

    ctx = _record(_identical(4))
    assert len(_tq_loops(ctx)) == 3


def test_an_unconfident_qrs_onset_does_not_close_the_window():
    """The onset threshold is 0,35. A window closed by a landmark the
    delineator was unsure of would end inside a depolarisation."""

    ctx = _record(_identical(4), onset_confidence=0.2)
    assert _tq_loops(ctx) == []


def test_an_absent_t_wave_yields_no_tq_window():
    """The confidence of a landmark placed on a wave that is not there says
    nothing about whether the wave was there, so the status is read too."""

    ctx = _record(_identical(4), t_present=False)
    assert _tq_loops(ctx) == []


def test_a_window_shorter_than_the_floor_is_dropped():
    floor = int(MINIMUM_TQ_WINDOW_MS * FS / 1000.0)
    record = np.zeros((1200, 3), dtype=float)
    record[300:340] = _p_shape("a")
    beats = [
        _Beat(r_onset=100, t_offset=250),
        _Beat(r_onset=250 + floor - 10, t_offset=800),
    ]
    assert len(_tq_loops(_Ctx(record, beats))) == 0

    beats[1] = _Beat(r_onset=250 + floor + 60, t_offset=800)
    assert len(_tq_loops(_Ctx(record, beats))) == 1


def test_the_last_beat_has_no_closing_onset():
    """Every beat but the last one gets a window; the last would need a QRS
    that the ten seconds do not contain."""

    ctx = _record(_identical(3))
    ctx.delineation.beats = ctx.delineation.beats[:-1]
    assert len(_tq_loops(ctx)) == 2


def test_the_tq_marker_measures_a_record_whose_p_waves_are_absent():
    """The reason the fourth marker is worth its extraction time.

    The P segment is usable in 3,7 % of recordings showing fibrillation. A
    recording with no P fiducials silences all three P markers; the TQ marker
    still reports. If this ever fails, the fourth marker has no justification
    and belongs deleted rather than defended.
    """

    shapes = [_tq_shape("a") for _ in range(6)]
    rng = np.random.default_rng(12)
    shapes = [shape + rng.normal(0.0, 0.003, shape.shape) for shape in shapes]
    ctx = _record(_identical(6), tq_shapes=shapes, with_p=False)
    assert _p_loops(ctx) == []
    assert _value("p_loop_template_dissimilarity_ratio", ctx) is None
    assert _value("tq_template_dissimilarity_ratio", ctx) is not None


def test_identical_tq_windows_have_no_dissimilarity():
    shapes = [_tq_shape("a") for _ in range(6)]
    ctx = _record(_identical(6), tq_shapes=shapes)
    assert _value("tq_template_dissimilarity_ratio", ctx) == pytest.approx(0.0, abs=1e-12)


# --- registration and roles ---------------------------------------------------


def test_every_marker_is_registered_once():
    available = set(BiomarkerRegistry.available())
    for name in morphology_variability_marker_ids():
        assert name in available
        assert isinstance(BiomarkerRegistry.get(name), MorphologyVariabilityBiomarker)


def test_four_markers_and_no_more():
    assert len(MORPHOLOGY_VARIABILITY_SPECS) == 4


def test_every_marker_declares_the_window_it_measures():
    windows = {spec.name: spec.window for spec in MORPHOLOGY_VARIABILITY_SPECS}
    assert windows["tq_template_dissimilarity_ratio"] == ATRIAL_WINDOW
    assert all(
        window == P_WINDOW
        for name, window in windows.items()
        if name.startswith("p_loop_")
    )


def test_the_similarity_is_not_declared_a_fraction():
    """A cosine similarity runs from -1 to 1. Declaring it a fraction would
    make valid negative values fail the plausibility audit, which is the
    mistake ``strip_amplitude_dispersion`` recorded before it."""

    unit = BY_NAME["p_loop_dominant_similarity_mean"].unit
    assert unit == "dimensionless"


def test_the_bimodality_is_declared_a_fraction_and_stays_inside_it():
    assert BY_NAME["p_loop_similarity_bimodality"].unit == "fraction"
    for seed in range(6):
        value = _value("p_loop_similarity_bimodality", _record(_scattered(count=9, seed=seed)))
        assert value is None or 0.0 <= value <= 1.0


def test_a_failing_calculator_abstains_rather_than_stopping_the_panel():
    spec = BY_NAME["p_loop_template_dissimilarity_ratio"]
    broken = type(spec)(
        spec.name,
        spec.family,
        spec.description,
        spec.formula,
        spec.unit,
        lambda ctx: (_ for _ in ()).throw(ValueError("deliberate")),
        spec.window,
        spec.evidence_note,
    )
    result = MorphologyVariabilityBiomarker(broken).compute(
        delineation=_Delineation([]), fs=FS
    )
    assert result.value is None
    assert "ValueError" in result.computation_notes


# --- the panels must not lose anything ---------------------------------------


def test_the_new_markers_reach_the_discovery_reserve_only():
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    for name in morphology_variability_marker_ids():
        assert name in panels["discovery_reserve"]
        assert name not in panels["primary_core"]
        assert name not in panels["sentinel"]


def test_the_curated_panels_keep_their_fixed_sizes():
    """The rule the segment-free module learned the hard way: admitting a
    marker to the candidate pool pushes another out of the sentinel panel."""

    panels = {panel.panel_id: panel.feature_ids for panel in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240


def test_no_post_outcome_hypothesis_sits_in_a_curated_panel():
    for panel in default_feature_panels():
        if panel.panel_id not in ("primary_core", "sentinel"):
            continue
        for name in panel.feature_ids:
            assert not feature_profile(name).post_outcome_hypothesis, name


def test_the_new_markers_are_declared_post_outcome_hypotheses():
    for name in morphology_variability_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis
        assert profile.scientific_tier == "post_outcome_discovery"
        assert profile.discovery_eligible


def test_the_marker_these_were_written_against_keeps_its_place():
    """The failure the TQ module produced once: adding thirteen atrial markers
    pushed eight out of the sentinel panel. ``p_loop_template_instability`` is
    the marker these four are meant to complement, and a complement that
    displaces what it complements is a replacement."""

    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    assert "p_loop_template_instability" in panels["primary_core"]


# --- the three markers that were never computed ------------------------------


def test_the_beat_to_beat_morphology_markers_are_no_longer_rhythm_context():
    """They describe the shape of the complex, not the rate. Routed to
    context they belonged to no panel, and a marker in no panel is never
    computed: the extraction asks only for the panels' feature ids."""

    for name in BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS:
        profile = feature_profile(name)
        assert profile.role != "ecg_context_comparator", name
        assert profile.discovery_eligible, name


def test_the_markers_do_not_inherit_the_consensus_gate():
    """The measured reason the new P markers reach patients the old one cannot.

    ``p_loop_template_instability`` is reached through a dispatcher that asks
    for the *consensus* P segment and returns nothing when that is
    unresolved -- although the calculation behind the gate reads the per-beat
    loops, which are often present when the consensus is not. On 1.383
    recordings drawn balanced on the machine report the marker reported on
    1,9 % of those read as fibrillation, and 559 recordings had three or more
    P loops while it stayed silent. Requiring one stable consensus P wave
    requires the absence of the phenomenon being measured.
    """

    ctx = _record(_scattered(seed=13))
    ctx.vcg_segment = lambda segment: None  # the consensus P is unresolved
    assert _p_loop_extra(ctx, "p_loop_template_instability").value is None
    assert _p_loops(ctx)  # the per-beat loops are there regardless
    assert _value("p_loop_template_dissimilarity_ratio", ctx) is not None
    assert _value("p_loop_dominant_similarity_mean", ctx) is not None


def test_the_t_side_has_no_per_beat_body_behind_its_gate():
    """How far the gate reaches, pinned so a later edit cannot widen it.

    ``_t_loop_extra`` opens with the same consensus check, but a call-graph
    walk over the module found that none of its branches reach the per-beat
    helpers -- so on the T side the gate is every marker's own precondition.
    On the P side exactly two markers have per-beat bodies. If a new branch
    starts reading per-beat data behind either gate, this fails.
    """

    import ast
    import inspect

    from vcg_biomarker.core.biomarkers import catalog_biomarkers as module

    per_beat = {
        "_beat_segment_curves",
        "_beat_segment_magnitudes",
        "_cluster_beat_area_vectors",
        "_beat_variability_feature",
        "_beat_bounds",
    }

    def called(function_name):
        tree = ast.parse(inspect.getsource(getattr(module, function_name)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                names.add(
                    func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                )
        return names

    assert not (called("_t_loop_extra") & per_beat)
    assert called("_p_loop_extra") & per_beat


def test_the_rate_markers_remain_rhythm_context():
    """The exemption must not sweep the RR measures along with them."""

    for name in ("rr_entropy/irregularity_index", "atrial_fibrillation_rr_surrogate", "rr_mean_ms/sd_ms/rmssd_ms/pnn50"):
        assert feature_profile(name, category="rhythm").role == "ecg_context_comparator"


def test_the_reclassified_markers_enter_the_reserve_and_not_the_sentinel():
    """They were never measured, so admitting them now is a decision taken
    with the interim results in hand -- post-outcome, and additive."""

    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    for name in BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS:
        assert name in panels["discovery_reserve"], name
        assert name not in panels["sentinel"], name
        assert feature_profile(name).post_outcome_hypothesis, name


def test_the_exemption_names_only_morphology_markers():
    assert BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS == frozenset(
        {"qrs_alternans_vector", "t_wave_alternans_vector", "qrs_t_angle_variability"}
    )


def test_the_four_markers_share_a_family_with_the_marker_they_complement():
    """The redundancy stage compares within conceptual families.

    Without the similarity tokens the scale-free ratio would sit in
    ``atrial_activation`` and ``p_loop_template_instability`` in
    ``beat_variability``, so the one comparison these markers were written to
    invite would never be made.
    """

    from vcg_biomarker.mimic.feature_roles import _conceptual_family

    for name in morphology_variability_marker_ids():
        assert _conceptual_family(name) == "beat_variability", name
    assert _conceptual_family("p_loop_template_instability") == "beat_variability"


def test_the_similarity_tokens_move_nothing_else():
    """Only these four markers carry the tokens, so the rule cannot reshuffle
    the curated panels behind the change it was made for."""

    carriers = sorted(
        name
        for name in BiomarkerRegistry.available()
        if any(token in name.casefold() for token in ("similarity", "bimodality"))
    )
    assert carriers == sorted(morphology_variability_marker_ids())


# --- the prose has to keep matching the code ---------------------------------


def test_the_documented_thresholds_are_the_ones_the_code_uses():
    import inspect

    from vcg_biomarker.core.biomarkers import morphology_variability_biomarkers as module

    source = inspect.getsource(module)
    assert "MINIMUM_TEMPLATE_SCALE_MV" in source
    assert str(MINIMUM_TQ_WINDOW_MS) in source
    assert "13 beats at the median" in module.__doc__


def test_the_module_says_it_does_not_replace_the_existing_marker():
    from vcg_biomarker.core.biomarkers import morphology_variability_biomarkers as module

    assert "p_loop_template_instability" in module.__doc__
    assert "replaces" in module.__doc__
