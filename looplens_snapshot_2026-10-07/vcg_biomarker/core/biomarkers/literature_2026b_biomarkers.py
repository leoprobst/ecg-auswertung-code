"""Four markers the extended 2026 search found missing, after four were not.

The search that produced this module proposed seven markers. **Three were
withdrawn once the registry was measured rather than remembered**, and that
is worth stating first, because it is the more useful result:

* a QRS three-dimensional voltage-time integral already exists as
  ``magnitude_area_qrs``. Its median across the development partitions is
  0,04 mV*s = **40 uV*s** against the 38,1 uV*s Gupta & Noheria report as the
  healthy median. Similar medians do not establish transportability of healthy
  reference intervals to a hospital population. The existing marker avoids a
  duplicate formula; reference intervals still require independent validation;
* the Selvester QRS score cannot be reconstructed to a standard this project
  would accept. The 2009 version is **53 criteria across ten leads in six
  ECG-type-specific variants**, with an amplitude standard that shifts one per
  cent per year of age from 55. Transcribing that from memory would produce an
  unverifiable threshold table, which is exactly what the provenance rule
  forbids. Its continuous substitute is already extracted:
  ``classical_pathological_q_waves`` carries the pathological-Q lead count on
  90 % of recordings;
* the scalar Morris P-terminal-force index is already extracted as
  ``classical_p_terminal_force_v1`` -- a continuous value, not only the
  binary verdict beside it.

A fourth was withdrawn **after** it was written and measured. ``Tpe/QT`` is
absent from the registry and looked like a free addition, but a least-squares
fit on ``tpeak_tend_ms`` and ``qt_interval_ms`` alone reproduces it with
**R^2 = 0,977** over 762 recordings. Both of those are curated panel members
and both are open to the combination search, so the ratio is something the
analysis can investigate from its components. R^2 is explained variance, not
two-percent individual measurement error or proof of clinical redundancy.
This historical exclusion was a development choice; a future comparison must
examine agreement and incremental performance against an appropriate joint
components model, without treating inspected patients as new validation.

What follows is what survived that check.

``repolarization_time_difference_ratio``
    Salatzki et al. (J Electrocardiol 2026) screened 583 vectorcardiographic
    quantities in 245 people **against cardiac magnetic resonance**. One ratio
    of two intervals separated impaired systolic function best: **AUC 0,843,
    sensitivity 80,0 %, specificity 83,9 %** at an ejection fraction below
    40 % -- the exact threshold this project's ``reduced_lvef_strict``
    endpoint uses. Measured before the marker was written: computable on 97 %
    of recordings from fiducials the delineation already places, which is more
    than twice the availability of the P-wave markers.

``t_azimuth_beatwise_dispersion``
    The same paper's best left-ventricular-hypertrophy parameters were T-wave
    azimuth, T-wave magnitude and **azimuth variability** (AUC 0,739-0,791).
    The registry holds beat-to-beat variability of the P axis, of the T
    *amplitude* and of the SVG direction -- but nothing for the beat-to-beat
    direction of the T vector itself.

    It is computed as a **circular** dispersion, from the mean resultant
    length of the unit direction vectors, not as a standard deviation of the
    angle. A standard deviation across the seam at +/-180 degrees is the
    defect the projection module was written to remove; reintroducing it in a
    variability marker would be worse, because there the seam enters every
    beat rather than every recording.

``v6_r_peak_time_ms`` and ``v6_v1_interpeak_interval_ms``
    The two numbers by which left-bundle capture is confirmed at conduction
    system pacing -- a field that has grown steadily since 2023 and now has
    its own ultra-high-frequency literature. Neither is in the registry.

    The project gained ``pacemaker_carrier`` this week, with 57.983 paced
    recordings. **That label is capped at analytic concordance as an
    endpoint** (the device is visible on the tracing), so these two are
    written for use as a stratifier, not as a target -- which is the role the
    published criteria have anyway.

    ``v6_r_peak_time_ms`` turns out to serve an older literature as well, and
    that is worth recording because it widens the marker beyond pacing: it is
    the **intrinsicoid deflection** in V6, the classical left-ventricular
    activation time whose prolongation past 50 ms is part of the hypertrophy
    criteria. A repository-wide search found no per-lead activation time in
    the project. The nearest existing marker, ``r_peak_timing_fraction``, is
    the position of the QRS *magnitude* peak inside the complex as a
    dimensionless fraction -- a lead-free analogue, not a duration in one
    lead, and therefore not comparable with any published cut-off. Measured
    here: median 48 ms with a 95th percentile of 68 ms, which is where a
    hospital cohort carrying a great deal of hypertrophy should sit.

All four are window statistics on segments the delineation already provides.
They are registered as post-outcome discovery hypotheses: no curated panel
changes, and none of them enters the combination search.
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
from .catalog_biomarkers import FeatureContext, FeatureValue, _context
from ..data_loader import LEAD_MAP
from ..delineation.base import RecordDelineation, fiducial_confidence_threshold


LITERATURE_2026B_VERSION = "1.1.0"
LITERATURE_2026B_SOURCE = "literature_2026b_v1"

EPS = 1e-12

# A circular dispersion over fewer than three beats is the separation of a
# pair expressed as a spread, which states nothing the pair does not.
MINIMUM_BEATS_FOR_DISPERSION = 3

# A T segment shorter than this cannot carry a direction worth taking: at
# 500 Hz, eight samples is 16 ms.
MINIMUM_T_SAMPLES = 8

# Below this the T vector has no defined direction and its azimuth is noise
# rather than a measurement. In millivolt*sample units of the area vector.
MINIMUM_T_AREA_MAGNITUDE = 1e-4

# Both intervals of the Salatzki ratio must be a real duration. One sample is
# not an interval, and a zero denominator is not a ratio.
MINIMUM_INTERVAL_MS = 4.0

# The ratio is bounded because a denominator approaching zero sends it to
# infinity, which is a delineation failure rather than a physiological value.
MAXIMUM_TIME_DIFFERENCE_RATIO = 20.0


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_literature_2026b_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_literature_2026b_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _beat_sample(beat, attr: str, wave: str) -> Optional[int]:
    """One fiducial of one beat, at that landmark's calibrated strictness.

    Same rule the morphology-variability module uses: the wave status is
    checked as well as the confidence, because a landmark placed on an absent
    wave carries a confidence that says nothing about whether the wave existed.
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


# ------------------------------------------------------- the Salatzki ratio


def _repolarization_time_difference_ratio(ctx: FeatureContext) -> FeatureValue:
    """(T onset - R peak) divided by (T peak - QRS end), both as durations.

    Salatzki et al. read this off the consensus complex as two ordinary
    intervals. The numerator spans depolarisation's end and the silent stretch
    before repolarisation becomes visible; the denominator spans the rise of
    the T wave. Their quotient is therefore a shape statement about where
    repolarisation sits inside the electrical cycle, made without any
    amplitude and without any rate correction.

    Both are taken as positive durations here. Reversing both differences
    leaves the quotient unchanged. Exact equivalence of acquisition and
    fiducial definitions to the published method has not been established;
    this remains a project-specific post-outcome discovery candidate.
    """

    r_peak = ctx.marker_index("r_peak")
    t_onset = ctx.marker_index("t_onset")
    qrs_end = ctx.marker_index("j_point")
    t_peak = ctx.marker_index("t_peak")
    missing = [
        name
        for name, value in (
            ("R peak", r_peak),
            ("T onset", t_onset),
            ("QRS end", qrs_end),
            ("T peak", t_peak),
        )
        if value is None
    ]
    if missing:
        return _abstain(f"missing fiducial: {', '.join(missing)}")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    numerator = (int(t_onset) - int(r_peak)) / fs * 1000.0
    denominator = (int(t_peak) - int(qrs_end)) / fs * 1000.0
    if numerator < MINIMUM_INTERVAL_MS or denominator < MINIMUM_INTERVAL_MS:
        return _abstain("fiducials out of order or interval too short")
    ratio = numerator / denominator
    if not np.isfinite(ratio) or ratio > MAXIMUM_TIME_DIFFERENCE_RATIO:
        return _abstain("implausible ratio")
    return FeatureValue(float(ratio))


# -------------------------------------------- beat-to-beat T-wave direction


def _beat_t_directions(ctx: FeatureContext) -> list[tuple[float, float]]:
    """Unit direction of each beat's T area vector, in the transverse plane.

    Returned as (cosine, sine) pairs rather than angles. That is not a
    presentation choice: the dispersion below is a circular statistic and
    needs the unit vectors, and carrying angles instead would put the seam at
    +/-180 degrees back into the arithmetic.

    The vectors are read from the record's own beats, not from the consensus
    complex, because the consensus complex has exactly one T wave and the
    question here is how much the direction moves between beats.
    """

    def build() -> list[tuple[float, float]]:
        record = ctx._record_vcg_xyz
        if record is None:
            return []
        beats = getattr(ctx.delineation, "beats", None) or ()
        limit = len(record) - 1
        directions: list[tuple[float, float]] = []
        for beat in beats:
            start = _beat_sample(beat, "t_onset", "t")
            end = _beat_sample(beat, "t_offset", "t")
            if start is None or end is None:
                continue
            start = max(0, min(int(start), limit))
            end = max(0, min(int(end), limit))
            if end - start < MINIMUM_T_SAMPLES:
                continue
            window = np.asarray(record[start : end + 1], dtype=float)
            if window.ndim != 2 or window.shape[1] != 3:
                continue
            if not np.all(np.isfinite(window)):
                continue
            area = window.sum(axis=0)
            transverse = float(np.hypot(area[0], area[2]))
            if transverse < MINIMUM_T_AREA_MAGNITUDE:
                continue
            directions.append((area[0] / transverse, area[2] / transverse))
        return directions

    return _memo(ctx, ("beat_t_directions",), build)


def _t_azimuth_beatwise_dispersion(ctx: FeatureContext) -> FeatureValue:
    """Circular dispersion of the T direction across beats, 0 to 1.

    One minus the mean resultant length of the unit direction vectors. Zero
    means every beat's T wave points the same way; one means the directions
    cancel completely.

    **This is not a standard deviation of the azimuth, deliberately.** An
    azimuth wraps at +/-180 degrees, so a recording whose T vectors sit at
    -179 and +179 degrees -- two degrees apart in the body -- would report a
    spread of 358 degrees. The resultant length has no seam because it never
    forms an angle: it averages the unit vectors and measures how much of
    their length survives.
    """

    directions = _beat_t_directions(ctx)
    if len(directions) < MINIMUM_BEATS_FOR_DISPERSION:
        return _abstain("fewer than three beats with a measurable T direction")
    values = np.asarray(directions, dtype=float)
    if not np.all(np.isfinite(values)):
        return _abstain("non-finite T directions")
    resultant = float(np.linalg.norm(values.mean(axis=0)))
    if not np.isfinite(resultant):
        return _abstain("non-finite resultant")
    return FeatureValue(float(np.clip(1.0 - resultant, 0.0, 1.0)))


# ------------------------------------------- conduction system pacing timing


def _lead_r_peak_sample(ctx: FeatureContext, lead: str) -> Optional[int]:
    """Sample of the dominant positive deflection inside the QRS of one lead.

    **This was written the other way first, and measurement rejected it.** The
    published criterion speaks of the *last* positive peak, so as to catch an
    R-prime that follows a taller earlier R. Implemented literally -- the
    latest positive local maximum in the complex -- the V6-minus-V1 interval
    came out at a median of **-42 ms** over 54 recordings, meaning V6 peaked
    *before* V1. That is backwards: in an ordinary complex V1 is rS with an
    early small r and V6 is qR with a later tall R, so the difference must be
    positive.

    The cause is ST elevation in V1 and V2, which is common: the signal rises
    again at the end of the QRS window and the last local maximum lands on
    that rise rather than on any R wave. Raising an amplitude threshold did
    not repair it -- at half the lead's maximum the median was still only
    +6 ms and barely half the recordings were positive.

    Taking the dominant positive deflection instead gives V6 at a median of
    46 ms, V1 at 28 ms and a difference of **+17 ms**, positive in 63 % of
    recordings -- which is the expected physiology.

    The cost is stated rather than hidden: where a small late R-prime follows
    a taller R, this reads the earlier peak. Separating those two cases needs
    a paced reference set the project does not have, so the unambiguous
    reading is used and the limitation is written down.
    """

    bounds = ctx.bounds("qrs")
    source = ctx.ecg_12lead
    if bounds is None or source is None or lead not in LEAD_MAP:
        return None
    index = LEAD_MAP[lead]
    if index >= source.shape[1]:
        return None
    start, end = int(bounds[0]), int(bounds[1])
    window = np.asarray(source[start : end + 1, index], dtype=float)
    if len(window) < 3 or not np.all(np.isfinite(window)):
        return None
    if float(np.max(window)) <= 0.0:
        return None
    return start + int(np.argmax(window))


def _v6_r_peak_time(ctx: FeatureContext) -> FeatureValue:
    """QRS onset to the last positive peak in V6, in milliseconds.

    The interval by which left-bundle capture is judged. It is written as a
    plain duration and not as a verdict: the published cut-offs differ by
    baseline QRS width and by pacing mode, and a threshold this project cannot
    source is a threshold it does not state.
    """

    bounds = ctx.bounds("qrs")
    if bounds is None:
        return _abstain("no QRS window")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    peak = _lead_r_peak_sample(ctx, "V6")
    if peak is None:
        return _abstain("no positive R peak in V6")
    value = (peak - int(bounds[0])) / fs * 1000.0
    if not np.isfinite(value) or value < 0.0:
        return _abstain("R peak precedes QRS onset")
    return FeatureValue(float(value))


def _v6_v1_interpeak_interval(ctx: FeatureContext) -> FeatureValue:
    """V6 R-peak time minus V1 R-peak time, in milliseconds.

    Signed on purpose. The published reading is that the interval separates
    left-bundle capture from septal capture by *how much later* the lateral
    wall peaks than the right side; a magnitude would throw away the direction
    that carries the whole statement.
    """

    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    v6 = _lead_r_peak_sample(ctx, "V6")
    v1 = _lead_r_peak_sample(ctx, "V1")
    if v6 is None or v1 is None:
        missing = "V6" if v6 is None else "V1"
        return _abstain(f"no positive R peak in {missing}")
    value = (int(v6) - int(v1)) / fs * 1000.0
    if not np.isfinite(value):
        return _abstain("non-finite interval")
    return FeatureValue(float(value))


# ------------------------------------------------------------------- specs


@dataclass(frozen=True)
class Literature2026bSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    evidence_note: str
    references: tuple[str, ...] = ()
    category: str = "vcg"


LITERATURE_2026B_SPECS: tuple[Literature2026bSpec, ...] = (
    Literature2026bSpec(
        "repolarization_time_difference_ratio",
        "Repolarisationszeitverhaeltnis",
        "Ratio of the R-peak-to-T-onset interval over the QRS-end-to-T-peak "
        "interval.",
        "(T onset - R peak) / (T peak - QRS end), both taken as positive "
        "durations in milliseconds.",
        "ratio",
        _repolarization_time_difference_ratio,
        "Best of 583 vectorcardiographic quantities for impaired systolic "
        "function against cardiac magnetic resonance in 245 people: AUC "
        "0,843, sensitivity 80,0 %, specificity 83,9 % below an ejection "
        "fraction of 40 %. Uses no amplitude and no rate correction.",
        references=(
            "Salatzki J et al. Evaluation of new vectorcardiography "
            "algorithms for identifying left ventricular hypertrophy and "
            "impaired systolic function. J Electrocardiol 2026;97:154272.",
        ),
    ),
    Literature2026bSpec(
        "t_azimuth_beatwise_dispersion",
        "Repolarisationsgeometrie",
        "Circular dispersion of the transverse T-vector direction across "
        "beats; 0 is identical directions, 1 is complete cancellation.",
        "1 - |mean over beats of (cos(T azimuth), sin(T azimuth))|, taken "
        "from each beat's T area vector.",
        "fraction",
        _t_azimuth_beatwise_dispersion,
        "T-wave azimuth, T-wave magnitude and azimuth variability reached "
        "AUC 0,739-0,791 for left ventricular hypertrophy against cardiac "
        "magnetic resonance. The registry carries beat-to-beat variability "
        "of the P axis, of the T amplitude and of the SVG direction, but "
        "none for the T vector's own direction.",
        references=(
            "Salatzki J et al. Evaluation of new vectorcardiography "
            "algorithms for identifying left ventricular hypertrophy and "
            "impaired systolic function. J Electrocardiol 2026;97:154272.",
        ),
    ),
    Literature2026bSpec(
        "v6_r_peak_time_ms",
        "Reizleitungsstimulation",
        "QRS onset to the last positive peak in V6.",
        "(latest positive peak in V6 within the QRS - QRS onset) in ms.",
        "ms",
        _v6_r_peak_time,
        "One of the two timing criteria by which left-bundle capture is "
        "confirmed at conduction system pacing. Written as a duration, not "
        "as a verdict: the published cut-offs depend on baseline QRS width "
        "and pacing mode, and no single threshold is claimed here.",
        references=(
            "Yessenov TK et al. Ultra-high-frequency ECG quantifies residual "
            "electrical dyssynchrony during left bundle branch area pacing. "
            "J Electrocardiol 2026;95:154423.",
        ),
        category="ecg",
    ),
    Literature2026bSpec(
        "v6_v1_interpeak_interval_ms",
        "Reizleitungsstimulation",
        "Signed difference between the V6 and V1 R-peak times.",
        "V6 R-peak time - V1 R-peak time, in ms.",
        "ms",
        _v6_v1_interpeak_interval,
        "The second timing criterion for left-bundle capture. Signed, "
        "because the published reading is how much later the lateral wall "
        "peaks than the right side; a magnitude would discard that.",
        references=(
            "Yessenov TK et al. Ultra-high-frequency ECG quantifies residual "
            "electrical dyssynchrony during left bundle branch area pacing. "
            "J Electrocardiol 2026;95:154423.",
        ),
        category="ecg",
    ),
)


class Literature2026bBiomarker(BiomarkerBase):
    def __init__(self, spec: Literature2026bSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return self._spec.category

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
        notes = [f"literature_2026b_version={LITERATURE_2026B_VERSION}",
                 "published_method_equivalence=not_established",
                 "published_performance_not_transferable_to_this_implementation"]
        if value.notes:
            notes.insert(0, value.notes)
        return BiomarkerResult(
            name=self._spec.name,
            value=value.value,
            unit=self._spec.unit,
            category=self.category,
            confidence=float(value.confidence),
            description=self._spec.description,
            references=list(self._spec.references),
            computation_notes="; ".join(notes),
            family=self._spec.family,
            formula=self._spec.formula,
        )


def literature_2026b_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in LITERATURE_2026B_SPECS)


def register_literature_2026b_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in LITERATURE_2026B_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), Literature2026bBiomarker):
                continue
            raise RuntimeError(
                f"Literature marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category=spec.category,
                source=LITERATURE_2026B_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Hypothesis taken from a named publication and registered "
                    "so that publication's claim can be tested rather than "
                    "assumed. " + spec.evidence_note +
                    " Published acquisition/fiducial equivalence has not been established. "
                    "Published performance is not a validated property of this implementation. "
                    "No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(Literature2026bBiomarker(spec))
