"""Beat-to-beat morphology variability, measured so that ten seconds suffice.

The literature this module follows reports that the beat-to-beat variability of
P-wave morphology separates patients with paroxysmal atrial fibrillation from
controls at AUC 0,849 (95 % CI 0,754-0,917), against 0,721 (0,613-0,813) for
the conventional P-wave indices taken together, p = 0,041 -- with the authors
stating that none of the standard indices performed acceptably on its own. The
published effect is the share of P waves matching the dominant morphology:
90,4 +- 7,8 % in paroxysmal fibrillation against 95,5 +- 3,4 % in health.

**That measure cannot be carried over, and the reason decides everything
below.** Those recordings are ten minutes of vectorcardiography at 1000 Hz;
this cohort holds ten seconds at 500 Hz. Measured on the store's own
partitions, a recording carries **13 beats at the median** (p10-p90 10-18). A
share computed over 13 beats moves in steps of 1/13 = 7,7 percentage points,
and the entire published effect is 5,1 points -- smaller than a single beat.
The share is not merely hard to measure here; it is quantised out of
existence.

A **continuous distance** carries no such floor: it spends 13 real numbers
where a share spends 13 yes/no decisions. Every marker in this module is
therefore a distance or a similarity, never a count and never a share.

**What already existed, and what it was missing.**
``p_loop_template_instability`` is in the primary core panel and is already a
continuous dissimilarity -- the median RMS distance of each P loop to the
median P loop. Four properties limit it, and the fourth was found by
measurement rather than by reading the code:

* It is reported in mV. A patient with 0,15 mV P waves shows larger distances
  than one with 0,05 mV at identical *relative* variability, so part of what
  the marker measures is P amplitude. The literature uses correlation-type,
  scale-free measures.
* One scalar cannot separate diffuse scatter -- every beat slightly different,
  as in atrial fibrosis -- from two populations, as in a wandering atrial
  pacemaker or a single ectopic focus. Both give the same median.
* It reads the P segment, so it inherits that segment's missingness. The P
  segment is usable in a small minority of recordings showing fibrillation.
  The marker is silent in the patients it was written for, which is the
  likeliest reason it reaches p = 5e-27 for a gain of AUC +0,0013.
* **It is gated on the consensus beat, although its own body is not.** The
  dispatcher that reaches it asks first for the consensus P segment and
  returns nothing when that is unresolved -- while the calculation behind the
  gate reads the *per-beat* P loops, which are often there when the consensus
  is not. Measured on 1.383 recordings drawn balanced on the machine report:
  the marker reports on **1,9 %** of recordings read as fibrillation against
  48,1 % of the rest, and in **559 of 1.383 recordings three or more P loops
  were formed while the marker stayed silent**. That gate is not a neutral
  quality rule here: requiring one stable consensus P wave requires the
  absence of the phenomenon being measured. The markers below read the
  per-beat loops directly, for the reason
  ``_distinct_p_morphology_cluster_count_raw`` already records for the P
  clustering it performs.

  **How far the gate reaches was measured, not assumed.** Of the nine markers
  behind ``_p_loop_extra`` exactly two have per-beat bodies and are therefore
  held by a precondition they do not need: ``p_loop_template_instability`` and
  ``p_loop_interbeat_axis_variability``. The other seven read the consensus
  segment themselves, so for them the gate is their own precondition.
  ``_t_loop_extra`` reaches no per-beat data at all, so the T side is
  unaffected. Neither marker is changed here -- both are prespecified and one
  is in the primary core panel -- and the ungated counterpart is added beside
  them instead.

The four markers here address those points and add the window that removes
the missingness:

``p_loop_template_dissimilarity_ratio``
    The existing quantity divided by the template's own size. Scale-free.

``p_loop_dominant_similarity_mean``
    The continuous counterpart of the published share, without its
    quantisation. A share is the mean of an indicator; this is the mean of
    the quantity the indicator was thresholding.

``p_loop_similarity_bimodality``
    Separates diffuse scatter from two populations.

``tq_template_dissimilarity_ratio``
    The same dissimilarity on the **TQ window** -- T offset to the next QRS
    onset -- rather than on the P segment. That window is available in 85,1 %
    of recordings against 37,0 % for the P segment, and in **50,3 % of all
    recordings it exists where no P wave does**. It is therefore measurable in
    precisely the patients where the three P markers abstain.

Nothing here replaces ``p_loop_template_instability``; it keeps running
unchanged beside them.

**Measured, on 1.383 recordings of the training split drawn balanced on the
machine report.** Reported so that the numbers can be argued with rather than
assumed:

=====================================  ========  ========  ======
Reported on                            mit VHF   ohne VHF  AUC
=====================================  ========  ========  ======
p_loop_template_instability             1,9 %     48,1 %   0,690
p_loop_template_dissimilarity_ratio    36,8 %     91,6 %   0,840
p_loop_dominant_similarity_mean        39,2 %     91,6 %   0,126
p_loop_similarity_bimodality           34,7 %     84,7 %   0,395
tq_template_dissimilarity_ratio        91,9 %     99,0 %   0,883
=====================================  ========  ========  ======

An AUC below 0,5 is the same separation read the other way: similarity falls
in fibrillation, and so does bimodality -- fibrillation is diffuse scatter,
not two populations, which is the distinction the third marker exists to
make and the reason it points opposite to the first two.

Among the 1.037 recordings ``p_loop_template_instability`` cannot measure at
all, the TQ ratio reports on 94,0 % of them (AUC 0,841) and the P ratio on
52,3 % (AUC 0,776).

**Two limits on those numbers, both load-bearing.** The reference is the
machine's own reading of the same tracing, the circularity this project has
measured elsewhere -- PTB-XL cardiologist labels gave AUC 0,891 against 0,68
for an independent echocardiographic reference. And the TQ ratio shares much
of its rank order with the ventricular rate: Spearman +0,54 against the RR
coefficient of variation, which by itself reaches AUC 0,864 here. Inside
quartiles of that coefficient the TQ ratio still separates at 0,746 / 0,807 /
0,898 / 0,774, so it is not a rate measure wearing another name -- but the
marginal 0,883 is not its own contribution either.

Fibrillation on the tracing is not the study's endpoint; it is the sharpest
available check that these markers measure atrial activity at all. What they
add to a *prospective* question -- new-onset fibrillation in a recording that
is still in sinus rhythm -- is untested and is what the run will answer.

The redundancy stage will say whether the normalisation earned its place. On
the 346 recordings where both exist the scale-free ratio and
``p_loop_template_instability`` correlate at Spearman +0,77, so P amplitude
was carrying part of the old marker; had it been |rho| >= 0,99, the
normalisation would have changed nothing and that would have been the result.

**These are hypotheses formed after seeing results.** They were written once
the development run had shown what the P segment's missingness costs, so they
are registered as post-outcome discovery like every other module in that
position, and they enter the discovery reserve rather than competing for
sentinel slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import (
    FeatureContext,
    FeatureValue,
    _beat_segment_curves,
    _context,
    _resample_curve,
)
from ..delineation.base import RecordDelineation, fiducial_confidence_threshold


MORPHOLOGY_VARIABILITY_VERSION = "1.0.0"
MORPHOLOGY_VARIABILITY_SOURCE = "morphology_variability_v1"

EPS = 1e-12

# Loops are compared after arc-length resampling to a common point count, the
# same convention ``_beat_segment_curves`` uses for every other beat-wise
# comparison in the project. 64 points is that function's own choice and is
# kept so the P markers here are directly comparable with it.
RESAMPLE_POINTS = 64

# A template plus at least two beats to compare against it. With two beats the
# "median loop" is the midpoint of the pair and every distance is half their
# separation, which measures nothing the pair does not already state.
MINIMUM_BEATS = 3

# The bimodality statistic reads gaps in a sorted list, and a list of three
# admits only two gaps -- the largest of which is at least half the span for
# arithmetic reasons rather than physiological ones. Six beats give five
# similarities and four interior gaps, the point at which the uniform
# expectation (1/4) is distinguishable from a genuine split.
MINIMUM_BIMODALITY_BEATS = 6

# Below this spread the beats are the same shape to within measurement noise
# and there is no structure to describe. The statistic then returns zero -- a
# true statement, not an abstention: no second population was found.
MINIMUM_SIMILARITY_SPAN = 0.02

# A template smaller than this is a baseline, not an atrial complex: a P wave
# runs 50-250 uV, and 10 uV is the noise floor of a filtered recording.
# Dividing by such a template would turn noise into an arbitrarily large
# ratio, which is the one way a scale-free measure can go wrong.
MINIMUM_TEMPLATE_SCALE_MV = 0.01

# A TQ window shorter than this cannot carry a shape worth comparing; it is
# also shorter than any P wave, so nothing is lost that the P markers would
# have caught.
MINIMUM_TQ_WINDOW_MS = 60.0

# Ratios are capped where they are formed, the convention the plausibility
# audit documents for the unit. A value at the cap says the beat-to-beat
# scatter exceeds the template many times over -- that there is no stable
# template -- rather than stating a precise multiple.
MAXIMUM_DISSIMILARITY_RATIO = 50.0

P_WINDOW = "p"
ATRIAL_WINDOW = "tq"


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_morphology_variability_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_morphology_variability_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


# ------------------------------------------------------------------ the loops


def _p_loops(ctx: FeatureContext) -> list[np.ndarray]:
    """One arc-length resampled P loop per beat that has both P fiducials."""

    return _memo(ctx, ("p_loops",), lambda: list(_beat_segment_curves(ctx, P_WINDOW)))


def _beat_sample(beat, attr: str, wave: str) -> Optional[int]:
    """One fiducial of one beat, at that landmark's calibrated strictness.

    The wave status is checked as well as the confidence, because a landmark
    the delineator placed on an absent wave carries a confidence that says
    nothing about whether the wave was there.
    """

    point = getattr(beat, attr, None)
    if point is None:
        return None
    if float(point.confidence) < fiducial_confidence_threshold(attr):
        return None
    status = getattr(beat, "resolved_wave_status", None)
    if status is not None:
        try:
            if status(wave).value != "present":
                return None
        except Exception:  # noqa: BLE001 - a missing status must not veto the beat
            pass
    return int(point.sample)


def _tq_loops(ctx: FeatureContext) -> list[np.ndarray]:
    """One resampled TQ window per beat: T offset to the next QRS onset.

    ``FeatureContext.bounds("tq")`` gives the window of the *consensus* beat
    and therefore one window per recording, which is what the spectral markers
    in the segment-free module need. A beat-to-beat comparison needs one
    window per beat, so the two landmarks are read from the record's own beats
    here. The closing QRS onset is taken as the earliest confident onset after
    the T offset rather than as "the next beat in the list", because that list
    carries no ordering guarantee and a beat whose onset was uncertain would
    end the window in the middle of a depolarisation.
    """

    def build() -> list[np.ndarray]:
        record = ctx._record_vcg_xyz
        if record is None:
            return []
        fs = float(getattr(ctx, "fs", 0) or 0)
        if fs <= 0:
            return []
        minimum_samples = int(round(MINIMUM_TQ_WINDOW_MS * fs / 1000.0))
        beats = getattr(ctx.delineation, "beats", None) or ()
        onset_threshold = fiducial_confidence_threshold("r_onset")
        onsets = sorted(
            int(beat.r_onset.sample)
            for beat in beats
            if getattr(beat, "r_onset", None) is not None
            and float(beat.r_onset.confidence) >= onset_threshold
        )
        limit = len(record) - 1
        curves: list[np.ndarray] = []
        for beat in beats:
            start = _beat_sample(beat, "t_offset", "t")
            if start is None:
                continue
            end = next((sample for sample in onsets if sample > start), None)
            if end is None:
                continue
            start = max(0, min(start, limit))
            end = max(0, min(end, limit))
            if end - start < max(minimum_samples, 4):
                continue
            curve = _resample_curve(record[start : end + 1], RESAMPLE_POINTS)
            if curve is not None:
                curves.append(curve)
        return curves

    return _memo(ctx, ("tq_loops",), build)


# ----------------------------------------------------------- the measurements


def _template_dissimilarity_ratio(curves: list[np.ndarray]) -> FeatureValue:
    """Beat-to-beat scatter expressed in units of the template's own size.

    The numerator is the quantity ``p_loop_template_instability`` already
    reports: the median over beats of the RMS distance between that beat's
    loop and the median loop. The denominator is the RMS radius of the median
    loop about its own centroid -- how large the template is as a shape, not
    where it sits. Scaling every amplitude by a constant scales both by that
    constant, so the ratio does not move; that is the whole point, and it is
    asserted directly in the tests.
    """

    if len(curves) < MINIMUM_BEATS:
        return _abstain("fewer than three comparable loops")
    stack = np.stack(curves)
    if not np.all(np.isfinite(stack)):
        return _abstain("non-finite loop samples")
    template = np.median(stack, axis=0)
    distances = np.sqrt(np.mean(np.sum((stack - template) ** 2, axis=2), axis=1))
    scale = float(
        np.sqrt(np.mean(np.sum((template - template.mean(axis=0)) ** 2, axis=1)))
    )
    if not np.isfinite(scale) or scale < MINIMUM_TEMPLATE_SCALE_MV:
        return _abstain("template below the atrial amplitude floor")
    value = float(np.median(distances)) / scale
    if not np.isfinite(value):
        return _abstain("non-finite ratio")
    return FeatureValue(float(min(value, MAXIMUM_DISSIMILARITY_RATIO)))


def _shape_vectors(curves: list[np.ndarray]) -> Optional[np.ndarray]:
    """Unit-length shape vectors, one per loop.

    Each loop is centred on its own centroid before normalisation, so the
    cosine that follows compares shapes rather than offsets. The clustering in
    ``advanced_vcg_biomarkers`` omits the centring because it only needs to
    count populations; a similarity read as a number has to be centred, or a
    shared offset inflates every value towards one and compresses the range
    the marker is meant to resolve.
    """

    vectors: list[np.ndarray] = []
    for curve in curves:
        values = np.asarray(curve, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            continue
        centred = (values - values.mean(axis=0)).ravel()
        norm = float(np.linalg.norm(centred))
        if norm <= EPS:
            continue
        vectors.append(centred / norm)
    if len(vectors) < MINIMUM_BEATS:
        return None
    return np.asarray(vectors)


def _dominant_similarities(curves: list[np.ndarray]) -> Optional[np.ndarray]:
    """Similarity of every other loop to the dominant template.

    The template is the medoid -- the loop with the highest median similarity
    to the rest -- which is the same construction the QRS and P clustering use
    elsewhere in the project, and it needs no iteration and no seed. Its own
    similarity to itself is excluded: it is one by construction and would add
    a fixed value whose weight depends on how many beats the recording
    happened to contain.
    """

    vectors = _shape_vectors(curves)
    if vectors is None:
        return None
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    medoid = int(np.argmax(np.median(similarities, axis=1)))
    row = np.delete(similarities[medoid], medoid)
    return row if len(row) >= 2 else None


def _p_template_ratio(ctx: FeatureContext) -> FeatureValue:
    return _template_dissimilarity_ratio(_p_loops(ctx))


def _tq_template_ratio(ctx: FeatureContext) -> FeatureValue:
    return _template_dissimilarity_ratio(_tq_loops(ctx))


def _p_dominant_similarity(ctx: FeatureContext) -> FeatureValue:
    """Mean similarity to the dominant template, not the median.

    The published quantity is the *share* of P waves matching the dominant
    morphology, which is the mean of an indicator; the continuous relaxation
    of a mean of indicators is the mean of what the indicator thresholded.
    The median is not that relaxation, and the difference is not academic: on
    six loops of one shape and three of another it returns exactly 1,0,
    because a median is by construction blind to a minority smaller than half.
    A minority of beats with a different atrial origin is the entire subject
    of this marker.
    """

    similarities = _memo(
        ctx, ("p_similarities",), lambda: _dominant_similarities(_p_loops(ctx))
    )
    if similarities is None:
        return _abstain("fewer than three comparable P loops")
    value = float(np.mean(similarities))
    return FeatureValue(value) if np.isfinite(value) else _abstain("non-finite")


def _p_similarity_bimodality(ctx: FeatureContext) -> FeatureValue:
    """Largest gap in the sorted similarities, as a share of their span.

    A recording whose beats scatter evenly spreads its similarities evenly,
    and the largest gap between neighbours is then about 1/(n-1) of the span.
    A recording with two populations puts a single large gap between them, and
    the statistic approaches one. It reads the shape of the distribution and
    needs no similarity threshold, which matters: counting members above a
    fixed cut would make the answer a property of the cut.

    A single deviant beat -- one atrial ectopic among sinus beats -- produces
    a high value, and that is intended. It is a two-population recording with
    one member in the minor population, and the median-based markers beside
    this one are deliberately blind to it.
    """

    similarities = _memo(
        ctx, ("p_similarities",), lambda: _dominant_similarities(_p_loops(ctx))
    )
    if similarities is None or len(similarities) < MINIMUM_BIMODALITY_BEATS - 1:
        return _abstain("fewer than six comparable P loops")
    ordered = np.sort(np.asarray(similarities, dtype=float))
    span = float(ordered[-1] - ordered[0])
    if not np.isfinite(span):
        return _abstain("non-finite similarities")
    if span < MINIMUM_SIMILARITY_SPAN:
        return FeatureValue(0.0)
    value = float(np.max(np.diff(ordered)) / span)
    if not np.isfinite(value):
        return _abstain("non-finite gap")
    return FeatureValue(float(np.clip(value, 0.0, 1.0)))


# ------------------------------------------------------------------ the specs


@dataclass(frozen=True)
class MorphologyVariabilitySpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    window: str
    evidence_note: str
    references: tuple[str, ...] = ()


MORPHOLOGY_VARIABILITY_SPECS: tuple[MorphologyVariabilitySpec, ...] = (
    MorphologyVariabilitySpec(
        "p_loop_template_dissimilarity_ratio",
        "Morphologievariabilitaet",
        "Beat-to-beat P-loop scatter relative to the size of the template loop.",
        "median_beats(RMS(loop - median loop)) / RMS radius of the median loop "
        "about its centroid.",
        "ratio",
        _p_template_ratio,
        P_WINDOW,
        "The scale-free form of p_loop_template_instability. That marker is "
        "reported in mV and therefore carries P amplitude alongside "
        "variability; this one does not.",
    ),
    MorphologyVariabilitySpec(
        "p_loop_dominant_similarity_mean",
        "Morphologievariabilitaet",
        "Mean cosine similarity of each P loop to the dominant P template.",
        "mean over beats of cos(loop, medoid loop) on centred, unit-normalised "
        "loops, the medoid excluded.",
        "dimensionless",
        _p_dominant_similarity,
        P_WINDOW,
        "The continuous counterpart of the published share of P waves matching "
        "the dominant morphology, which cannot be measured over the 13 beats a "
        "ten-second recording provides: its step of 1/13 exceeds the reported "
        "effect. A share is a mean of indicators, so the relaxation is a mean "
        "rather than a median.",
    ),
    MorphologyVariabilitySpec(
        "p_loop_similarity_bimodality",
        "Morphologievariabilitaet",
        "Largest gap between neighbouring P-loop similarities, as a share of "
        "their span.",
        "max(diff(sort(similarities))) / (max - min), zero when the span is "
        "below 0,02.",
        "fraction",
        _p_similarity_bimodality,
        P_WINDOW,
        "Separates diffuse beat-to-beat scatter from two distinct P "
        "populations. About 1/(n-1) for even scatter, near one for a split. "
        "The existing cluster count answers the same question as an integer, "
        "which over 13 beats is a coarse instrument.",
    ),
    MorphologyVariabilitySpec(
        "tq_template_dissimilarity_ratio",
        "Morphologievariabilitaet",
        "Beat-to-beat scatter of the TQ window relative to the size of its "
        "template.",
        "As p_loop_template_dissimilarity_ratio, over [T offset, next QRS "
        "onset] of each beat.",
        "ratio",
        _tq_template_ratio,
        ATRIAL_WINDOW,
        "Atrial morphology variability without requiring a P wave. The TQ "
        "window is available in 85,1 % of recordings against 37,0 % for the P "
        "segment, and exists without a P wave in 50,3 % of all recordings -- "
        "so this is measurable in the patients where the P markers abstain.",
    ),
)


class MorphologyVariabilityBiomarker(BiomarkerBase):
    def __init__(self, spec: MorphologyVariabilitySpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def references(self) -> list[str]:
        return list(self._spec.references)

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult:
        ctx = _context(delineation, ecg_12lead, vcg_xyz, fs)
        try:
            value = self._spec.calculator(ctx)
        except Exception as error:  # noqa: BLE001 - one marker must not stop a panel
            value = _abstain(f"{type(error).__name__}: {error}")
        notes = [
            f"morphology_variability_version={MORPHOLOGY_VARIABILITY_VERSION}",
            f"window={self._spec.window}",
        ]
        if value.notes:
            notes.insert(0, value.notes)
        return BiomarkerResult(
            name=self._spec.name,
            value=value.value,
            unit=self._spec.unit,
            category="vcg",
            confidence=float(value.confidence),
            description=self._spec.description,
            references=list(self._spec.references),
            computation_notes="; ".join(notes),
            family=self._spec.family,
            formula=self._spec.formula,
        )


def morphology_variability_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in MORPHOLOGY_VARIABILITY_SPECS)


def register_morphology_variability_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in MORPHOLOGY_VARIABILITY_SPECS:
        if spec.name in existing:
            if isinstance(
                BiomarkerRegistry.get(spec.name), MorphologyVariabilityBiomarker
            ):
                continue
            raise RuntimeError(
                "Morphology-variability marker collides with an existing "
                f"biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=MORPHOLOGY_VARIABILITY_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Prospective hypothesis about beat-to-beat morphology "
                    "measured continuously, because a share cannot resolve the "
                    "published effect over 13 beats. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
                references=list(spec.references),
                implementation_status="implemented",
            )
        )
        BiomarkerRegistry.register(MorphologyVariabilityBiomarker(spec))


__all__ = [
    "ATRIAL_WINDOW",
    "MAXIMUM_DISSIMILARITY_RATIO",
    "MINIMUM_BEATS",
    "MINIMUM_BIMODALITY_BEATS",
    "MINIMUM_SIMILARITY_SPAN",
    "MINIMUM_TEMPLATE_SCALE_MV",
    "MINIMUM_TQ_WINDOW_MS",
    "MORPHOLOGY_VARIABILITY_SOURCE",
    "MORPHOLOGY_VARIABILITY_SPECS",
    "MORPHOLOGY_VARIABILITY_VERSION",
    "MorphologyVariabilityBiomarker",
    "MorphologyVariabilitySpec",
    "P_WINDOW",
    "RESAMPLE_POINTS",
    "morphology_variability_marker_ids",
    "register_morphology_variability_biomarkers",
]
