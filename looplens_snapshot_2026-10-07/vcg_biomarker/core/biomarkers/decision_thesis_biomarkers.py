"""Five markers derived from the treatment-decision theses.

The theses are in ``THESEN_ENTSCHEIDUNGSMARKER_2026-09-12.md`` at the project
root, and they grow out of ``BEHANDLUNGSENTSCHEIDUNGEN_2026-09-12.md``: the
catheterisation-laboratory decision is made on a millimetre criterion, and
the measured decision cells -- 440 revascularised without an alert, 71 alerts
without a culprit, 3.752 unverified -- ask *why* an injury current can be
present and the criterion silent, or the criterion loud and the injury
absent. Each marker below is one physical answer to that question, written
down with its refuting observation before it was built.

``st_vector_criterion_visibility``, ``st_vector_criterion_hidden_mv``
    Thesis E1, geometry. The criterion is a direction-dependent threshold
    on a direction-independent quantity. A unit injury vector s produces the
    J-point elevation d_i . s in lead i, with d_i the Dower lead vector, and
    the criterion needs two contiguous leads over their thresholds. So each
    direction has a *visibility*: the best contiguous pair's smaller
    threshold-scaled projection. Anterior-left-inferior directions are seen
    well; posterior, rightward, superior and basal-lateral directions are
    seen poorly or as depression only. The first marker is that visibility
    as a fraction of the best possible direction; the second is the part of
    the ST magnitude the criterion cannot see, in millivolts.

``st_vector_nondiscordant_magnitude_mv``
    Thesis E2, geometric decomposition. Removes the antiparallel projection
    of ST on the QRS area direction. This does not identify which component
    is primary ischemia or secondary repolarization: either mechanism can
    contribute to the removed component or the remainder. Clinical value
    under bundle branch block requires independent acute reference evidence.

``qrs_opposition_to_st_vector_mv``
    Thesis E3, recruitment. Severe transmural ischaemia acts during the
    QRS -- the injury current and the conduction delay together -- and
    absorbs the forces that oppose the injury vector: the S wave in the
    leads that carry the elevation, Birnbaum's grade 3, and the denominator
    of the Smith-modified Sgarbossa ratio. The marker is the deepest
    excursion of the QRS loop against the ST direction, window-free. (A
    first form used the mean of the terminal 40 ms; it read zero in most
    recordings because that window still contains the main leftward loop,
    and was replaced before any endpoint was looked at.)

``initial_qrs_to_st_vector_angle_deg``
    Thesis E4, acuteness. With time from occlusion the infarcted tissue stops
    contributing initial forces, so the initial QRS vector turns *away* from
    the infarct while the ST vector keeps pointing *at* it. The angle between
    the two grows from the first hours towards the completed infarction with
    persistent elevation, and is the vector form of the T/QRS aneurysm rule.

**Measured on 2.964 recordings of the training split before this text was
finished.** The acute endpoints are nearly empty in the pilot cohort
(``acute_mi_probable`` sixteen positives, ``omi_revascularization_proxy``
five), so the acute end of every thesis waits for the full run. What could
be tested was the other end:

* Thesis E1, the mechanism: among recordings with comparable ST magnitude
  (at or above the first quartile of the criterion-positive ones, 0,123 mV)
  the criterion-negative recordings have a median visibility of **0,211
  against 0,537** for the criterion-positive, and carry **0,132 mV hidden
  against 0,080 mV**; visibility separates criterion-positive from
  criterion-negative at equal magnitude with AUC 0,756. The seventeen
  recordings flagged by the posterior-infarction pattern sit at 0,096, where
  the geometry puts a posterior vector (0,10). So the quantity measures what
  the criterion sees. In probable acute infarction (fourteen positives) the
  visibility runs *lower* than in controls (AUC 0,362) -- the direction the
  thesis predicts, at a size that decides nothing.
* Thesis E2, the secondary end, held everywhere it was testable: strict echo
  hypertrophy raises the ST magnitude (AUC 0,618) and leaves the magnitude
  beyond discordance untouched (AUC 0,499; share of the ST magnitude 0,754
  against 0,922); left bundle branch block raises the magnitude to AUC 0,758
  and the rest only to 0,577 (share **0,545 against 0,927**); pacing 0,661
  against 0,526. The block explains most of its own ST, and the marker knows
  it.
* Thesis E3: the right-bundle-branch-block half was **refuted** -- 0,43 mV
  against 0,67 in controls, the opposite direction. The reason is visible
  after the fact: the deepest excursion against the ST direction is set by
  the main loop, not by the terminal forces, so in right bundle branch block
  (ST opposing the small terminal R') the marker reads the concordant main
  loop and falls, while in left bundle branch block (1,29 against 0,62) and
  hypertrophy (1,13 against 0,58) it reads the discordant main loop and
  rises. The marker therefore measures the QRS excursion against the injury
  direction as a magnitude, which is what the acute half of the thesis
  needs; that half is untested and stands.
* Thesis E4 held at its testable end: old infarction **26,9 against 19,8
  degrees** (AUC 0,573), strict hypertrophy 22,4 against 19,7. Left bundle
  branch block, about which the thesis said nothing, is larger still (36,2),
  as every ST-direction marker is under the block.

Redundancy against the nearest neighbours, joint linear R-squared: 0,890
(visibility), 0,838 (hidden magnitude), 0,923 (magnitude beyond
discordance), 0,646 (opposition), 0,328 (initial angle); the pre-set
boundary of 0,95 removed nothing. Four markers abstain on about half of
recordings for the same reason the ischaemia thesis markers do -- the ST
vector is below five per cent of the QRS maximum -- and the initial angle on
two thirds, because the initial 20-ms vector is often below the gate too.

The lead vectors are the forward Dower matrix, which reproduces the
project's inverse Dower matrix to 0,0005 (pinned by a test); the contiguity
and thresholds are those of ``mimic/classical.py``, with the V2-V3
threshold fixed at 0,20 mV because a direction's visibility must not depend
on the patient's age or sex.
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
    _context,
    _vector_array,
    _window_vector,
)
from ..delineation.base import RecordDelineation


DECISION_THESIS_VERSION = "1.0.0"
DECISION_THESIS_SOURCE = "decision_thesis_v1"

EPS = 1e-12

# The same gate the ischaemia thesis markers use: a vector below this share
# of the QRS maximum has no direction worth reading.
MINIMUM_VECTOR_SHARE = 0.05

INITIAL_QRS_MS = 20

# Forward Dower matrix, rows in the order of INVERSE_DOWER_CHANNELS. Its
# pseudo-inverse is the project's inverse Dower matrix (test-pinned), so the
# lead vectors here are the ones the VCG was derived through.
DOWER_LEAD_VECTORS: dict[str, tuple[float, float, float]] = {
    "V1": (-0.515, 0.157, -0.917),
    "V2": (0.044, 0.164, -1.387),
    "V3": (0.882, 0.098, -1.277),
    "V4": (1.213, 0.127, -0.601),
    "V5": (1.125, 0.127, -0.086),
    "V6": (0.831, 0.076, 0.230),
    "I": (0.632, -0.235, 0.059),
    "II": (0.235, 1.066, -0.132),
}

# The criterion's contiguity, copied from mimic/classical.py (a test pins the
# copy to the original; the core package does not import from mimic).
CONTIGUOUS_GROUPS: tuple[tuple[str, str], ...] = (
    ("II", "III"),
    ("II", "aVF"),
    ("III", "aVF"),
    ("I", "aVL"),
    ("I", "V5"),
    ("I", "V6"),
    ("aVL", "V5"),
    ("aVL", "V6"),
    ("V5", "V6"),
    ("V1", "V2"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "V5"),
    ("V5", "V6"),
)

# J-point thresholds in mV. V2-V3 carries the men-over-forty value: the
# geometric visibility of a direction is a property of the lead system, not
# of the patient, so it uses one fixed threshold set.
DEFAULT_THRESHOLD_MV = 0.10
V2_V3_THRESHOLD_MV = 0.20


def _lead_vectors() -> dict[str, np.ndarray]:
    base = {name: np.asarray(vector, dtype=float) for name, vector in DOWER_LEAD_VECTORS.items()}
    one, two = base["I"], base["II"]
    base["III"] = two - one
    base["aVR"] = -(one + two) / 2.0
    base["aVL"] = one - two / 2.0
    base["aVF"] = two - one / 2.0
    return base


LEAD_VECTORS: dict[str, np.ndarray] = _lead_vectors()

THRESHOLDS_MV: dict[str, float] = {
    name: (V2_V3_THRESHOLD_MV if name in ("V2", "V3") else DEFAULT_THRESHOLD_MV)
    for name in LEAD_VECTORS
}

# Threshold-scaled lead vectors: a . s is the elevation in lead i per mV of
# injury vector, in multiples of that lead's threshold.
_SCALED: dict[str, np.ndarray] = {name: LEAD_VECTORS[name] / THRESHOLDS_MV[name] for name in LEAD_VECTORS}
_PAIRS: tuple[tuple[np.ndarray, np.ndarray], ...] = tuple(
    (_SCALED[a], _SCALED[b]) for a, b in CONTIGUOUS_GROUPS
)


def criterion_visibility(direction: np.ndarray) -> Optional[float]:
    """How much elevation the criterion sees per mV of injury vector.

    For the unit direction, the best contiguous pair's smaller projection in
    threshold units, floored at zero: a direction whose every contiguous pair
    contains a lead that sees depression has no visibility as elevation.
    """

    vector = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= EPS:
        return None
    unit = vector / norm
    best = 0.0
    for a, b in _PAIRS:
        pair = min(float(np.dot(a, unit)), float(np.dot(b, unit)))
        if pair > best:
            best = pair
    return float(best)


def _pair_ceiling(a: np.ndarray, b: np.ndarray) -> float:
    """Exact maximum over unit directions of min(a.s, b.s).

    The maximum of the smaller of two linear functions on the sphere is
    attained either where one of them alone is maximal (s along a or along
    b) or where the two are equal; the equal point is the direction of a
    projected onto the plane orthogonal to a - b.
    """

    candidates = []
    for lead in (a, b):
        norm = float(np.linalg.norm(lead))
        if norm > EPS:
            unit = lead / norm
            candidates.append(min(float(np.dot(a, unit)), float(np.dot(b, unit))))
    diff = a - b
    diff_norm2 = float(np.dot(diff, diff))
    if diff_norm2 > EPS:
        balanced = a - (float(np.dot(a, diff)) / diff_norm2) * diff
        norm = float(np.linalg.norm(balanced))
        if norm > EPS:
            unit = balanced / norm
            candidates.append(min(float(np.dot(a, unit)), float(np.dot(b, unit))))
    return max(candidates) if candidates else 0.0


VISIBILITY_CEILING: float = max(_pair_ceiling(a, b) for a, b in _PAIRS)


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_decision_thesis_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_decision_thesis_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _finite_vector(vector) -> Optional[np.ndarray]:
    if vector is None:
        return None
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return None
    return values


def _st_vector(ctx: FeatureContext) -> Optional[np.ndarray]:
    """The mean origin-corrected ST vector the registry's ST markers read."""

    return _memo(ctx, ("st_vector",), lambda: _finite_vector(_vector_array(ctx, "st", "mean")))


def _qrs_scale(ctx: FeatureContext) -> float:
    reference = _finite_vector(_vector_array(ctx, "qrs", "max"))
    return float(np.linalg.norm(reference)) if reference is not None else 0.0


def _st_with_direction(ctx: FeatureContext) -> tuple[Optional[np.ndarray], Optional[str]]:
    """The ST vector, or the reason it has no readable direction."""

    st = _st_vector(ctx)
    if st is None:
        return None, "ST vector unavailable"
    scale = _qrs_scale(ctx)
    magnitude = float(np.linalg.norm(st))
    if magnitude <= EPS:
        return None, "ST vector is zero"
    if scale > EPS and magnitude < MINIMUM_VECTOR_SHARE * scale:
        return None, "ST vector too small to carry a direction"
    return st, None


# ------------------------------------------------------------ thesis E1


def _visibility_fraction(ctx: FeatureContext) -> FeatureValue:
    st, reason = _st_with_direction(ctx)
    if st is None:
        return _abstain(reason or "ST vector unavailable")
    visibility = criterion_visibility(st)
    if visibility is None:
        return _abstain("degenerate ST vector")
    return FeatureValue(float(np.clip(visibility / VISIBILITY_CEILING, 0.0, 1.0)))


def _hidden_magnitude(ctx: FeatureContext) -> FeatureValue:
    st, reason = _st_with_direction(ctx)
    if st is None:
        return _abstain(reason or "ST vector unavailable")
    visibility = criterion_visibility(st)
    if visibility is None:
        return _abstain("degenerate ST vector")
    fraction = float(np.clip(visibility / VISIBILITY_CEILING, 0.0, 1.0))
    return FeatureValue(float(np.linalg.norm(st)) * (1.0 - fraction))


# ------------------------------------------------------------ thesis E2


def _nondiscordant_magnitude(ctx: FeatureContext) -> FeatureValue:
    st = _st_vector(ctx)
    if st is None:
        return _abstain("ST vector unavailable")
    area = _finite_vector(_vector_array(ctx, "qrs", "area"))
    if area is None:
        return _abstain("QRS area vector unavailable")
    norm = float(np.linalg.norm(area))
    if norm <= EPS:
        return _abstain("degenerate QRS area vector")
    projection = float(np.dot(st, area / norm))
    discordant = min(projection, 0.0)
    magnitude2 = float(np.dot(st, st))
    return FeatureValue(float(np.sqrt(max(magnitude2 - discordant * discordant, 0.0))))


# ------------------------------------------------------------ thesis E3


def _qrs_opposition(ctx: FeatureContext) -> FeatureValue:
    """Deepest excursion of the QRS loop against the ST vector direction."""

    st, reason = _st_with_direction(ctx)
    if st is None:
        return _abstain(reason or "ST vector unavailable")
    points = ctx.vcg_segment_origin_corrected("qrs")
    if points is None:
        return _abstain("QRS segment unavailable")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
        return _abstain("QRS segment unavailable")
    unit = st / float(np.linalg.norm(st))
    against = -(points @ unit)
    return FeatureValue(float(max(0.0, float(np.max(against)))))


# ------------------------------------------------------------ thesis E4


def _initial_qrs_to_st_angle(ctx: FeatureContext) -> FeatureValue:
    st, reason = _st_with_direction(ctx)
    if st is None:
        return _abstain(reason or "ST vector unavailable")
    initial = _finite_vector(_window_vector(ctx, "qrs", length_ms=INITIAL_QRS_MS, end_aligned=False))
    if initial is None:
        return _abstain("initial QRS window unavailable")
    scale = _qrs_scale(ctx)
    norm = float(np.linalg.norm(initial))
    if norm <= EPS or (scale > EPS and norm < MINIMUM_VECTOR_SHARE * scale):
        return _abstain("initial QRS vector too small to carry a direction")
    cosine = float(np.clip(np.dot(initial, st) / (norm * float(np.linalg.norm(st))), -1.0, 1.0))
    return FeatureValue(float(np.degrees(np.arccos(cosine))))


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class DecisionThesisSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    thesis: str
    references: tuple[str, ...] = ()


REF_THESES = (
    "THESEN_ENTSCHEIDUNGSMARKER_2026-09-12.md and "
    "BEHANDLUNGSENTSCHEIDUNGEN_2026-09-12.md, project root: own theses on why "
    "the catheterisation criterion misses and misfires, each with its "
    "refuting observation."
)
REF_DOWER = (
    "Dower GE, Machado HB, Osborne JA. On deriving the electrocardiogram from "
    "vectorcardiographic leads. Clin Cardiol 1980;3:87-95."
)
REF_UNIVERSAL_DEFINITION = (
    "Thygesen K, Alpert JS, Jaffe AS, et al. Fourth universal definition of "
    "myocardial infarction (2018). Eur Heart J 2019;40:237-269."
)
REF_SGARBOSSA = (
    "Sgarbossa EB, Pinski SL, Barbagelata A, et al. Electrocardiographic "
    "diagnosis of evolving acute myocardial infarction in the presence of "
    "left bundle-branch block. N Engl J Med 1996;334:481-487."
)
REF_SMITH_RATIO = (
    "Smith SW, Dodd KW, Henry TD, Dvorak DM, Pearce LA. Diagnosis of "
    "ST-elevation myocardial infarction in the presence of left bundle branch "
    "block with the ST-elevation to S-wave ratio in a modified Sgarbossa "
    "rule. Ann Emerg Med 2012;60:766-776."
)
REF_BIRNBAUM = (
    "Birnbaum Y, Sclarovsky S, Blum A, Mager A, Gabbay U. Prognostic "
    "significance of the initial electrocardiographic pattern in a first "
    "acute anterior wall myocardial infarction. Chest 1993;103:1681-1687."
)
REF_ANEURYSM = (
    "Smith SW. T/QRS ratio best distinguishes ventricular aneurysm from "
    "anterior myocardial infarction. Am J Emerg Med 2005;23:279-287."
)


DECISION_THESIS_SPECS: tuple[DecisionThesisSpec, ...] = (
    DecisionThesisSpec(
        "st_vector_criterion_visibility",
        "Thesenmarker Entscheidung",
        "Standardized Dower-projection visibility of the mean ST direction "
        "under a fixed threshold template; not patient-specific J-point criteria.",
        "max over contiguous lead pairs of min(d_i.s/theta_i, d_j.s/theta_j) "
        "for the unit mean ST vector s, Dower lead vectors d and fixed thresholds "
        "theta, floored at 0 and divided by the maximum over all directions.",
        "fraction",
        _visibility_fraction,
        "Thesis E1: the criterion is a direction-dependent threshold on a "
        "direction-independent quantity, and part of what it misses is "
        "geometric. Refuted if revascularised cases without an alert do not "
        "have lower visibility than revascularised cases with one.",
        references=(REF_THESES, REF_DOWER, REF_UNIVERSAL_DEFINITION),
    ),
    DecisionThesisSpec(
        "st_vector_criterion_hidden_mv",
        "Thesenmarker Entscheidung",
        "Mean ST magnitude weighted by one minus standardized Dower visibility; "
        "a geometric proxy, not reconstructed hidden injury current.",
        "|ST| * (1 - visibility fraction).",
        "mV",
        _hidden_magnitude,
        "Thesis E1, magnitude form: a large injury vector in a poorly seen "
        "direction is the geometric occlusion the criterion misses. Refuted "
        "if revascularised cases without an alert do not carry more hidden "
        "magnitude than revascularised cases with one.",
        references=(REF_THESES, REF_DOWER, REF_UNIVERSAL_DEFINITION),
    ),
    DecisionThesisSpec(
        "st_vector_nondiscordant_magnitude_mv",
        "Thesenmarker Entscheidung",
        "Geometric ST remainder after removing its antiparallel QRS-area "
        "projection; not an isolated primary ischemia measurement.",
        "sqrt(|ST|^2 - min(ST.q, 0)^2) with q the unit QRS area vector: the "
        "antiparallel projection is removed, the orthogonal and concordant "
        "parts remain.",
        "mV",
        _nondiscordant_magnitude,
        "Thesis E2, revised: geometric ST remainder may add information "
        "beyond QRS-discordant magnitude. Primary and secondary mechanisms "
        "are not identified by this decomposition; require independent "
        "reference evidence and a prespecified clinical comparator. Refuted if "
        "adequately precise independent estimates exclude the prespecified "
        "minimum incremental value, not merely when p exceeds 0.05.",
        references=(REF_THESES, REF_SGARBOSSA, REF_SMITH_RATIO),
    ),
    DecisionThesisSpec(
        "qrs_opposition_to_st_vector_mv",
        "Thesenmarker Entscheidung",
        "The deepest excursion of the QRS loop against the ST vector "
        "direction, in millivolts: the S wave as the injury vector sees it.",
        "max over the QRS of max(0, -r(t) . s) for the unit ST vector s.",
        "mV",
        _qrs_opposition,
        "Thesis E3: severe transmural ischaemia recruits the QRS and absorbs "
        "the forces opposing the injury vector. Refuted if acute occlusion "
        "does not fall below controls. The cross-prediction that right "
        "bundle branch block rises above controls was refuted in the pilot "
        "(0,43 against 0,67 mV): the excursion is set by the main loop, not "
        "the terminal forces, so the marker reads a magnitude-weighted "
        "discordance -- high under left bundle branch block and strain.",
        references=(REF_THESES, REF_BIRNBAUM, REF_SMITH_RATIO),
    ),
    DecisionThesisSpec(
        "initial_qrs_to_st_vector_angle_deg",
        "Thesenmarker Entscheidung",
        "Angle between the initial 20 ms of the QRS and the mean ST vector.",
        "angle(mean vector of the initial 20 ms of QRS, mean ST vector) in "
        "degrees, 0-180.",
        "degrees",
        _initial_qrs_to_st_angle,
        "Thesis E4, revised: initial QRS/ST geometry may vary with acute "
        "context but is not an infarct clock. Require symptom-onset uncertainty, "
        "serial ECGs, reperfusion timing and competing conduction diagnoses. "
        "Refuted if sufficiently precise independent estimates exclude the "
        "prespecified temporal association; an old-MI ICD label is not an onset time.",
        references=(REF_THESES, REF_ANEURYSM),
    ),
)


class DecisionThesisBiomarker(BiomarkerBase):
    def __init__(self, spec: DecisionThesisSpec) -> None:
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
        notes = [f"decision_thesis_version={DECISION_THESIS_VERSION}"]
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


def decision_thesis_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in DECISION_THESIS_SPECS)


def register_decision_thesis_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in DECISION_THESIS_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), DecisionThesisBiomarker):
                continue
            raise RuntimeError(
                f"Decision thesis marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=DECISION_THESIS_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Derived from an own thesis on the catheterisation "
                    "decision, written down with its refuting observation "
                    "before the marker was built. " + spec.thesis
                    + " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(DecisionThesisBiomarker(spec))
