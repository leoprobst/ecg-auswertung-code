"""Three markers the 2024-2026 literature named and this project lacked.

Each one comes from a specific paper, and each is written so the paper's claim
can be tested rather than assumed. Where the source does not give an exact
operationalisation, that is said here rather than hidden behind a formula.

``p_terminal_progression_v4_v6``
    Hoffmann et al. (Front Cardiovasc Med 2026) tested four purely
    morphological, visually assessed ECG criteria against invasive left-atrial
    voltage mapping in 78 patients read by five electrophysiologists, and
    reached **AUC 0,905** (sensitivity 85,7 %, specificity 95,4 %) for
    detecting low-voltage areas -- the fibrotic substrate behind atrial
    fibrillation recurrence. Interatrial block and **poor terminal P-wave
    progression in V4-V6** were the two most frequent predictors. The project
    already carries ``interatrial_block_vector_score``; for the second it had
    no marker at all.

``ventricular_gradient_apicobasal_uv_ms``
    Sobota et al. (J Physiol 2026) showed with electrocardiographic imaging
    that the apicobasal repolarization gradient **diminishes and inverts with
    age** (r = -0,7265, p = 0,0001), and that the inverted state carries lower
    vulnerability to re-entrant arrhythmia. This project has separately
    measured a large age effect on the spatial QRS-T angle -- 27-41 degrees in
    the young against 74,4 in the whole cohort -- without a mechanism. A
    signed apicobasal gradient is the quantity that would connect the two --
    and it is already computed on every recording, as provenance metadata
    beside ``svg_magnitude``, where no analysis can screen it.

    **Measured on 1.411 recordings of the training split before this marker
    was registered.** The published direction holds: Spearman against age
    -0,223, median 25.280 uV*ms under 50 years against 9.534 uV*ms at 70 and
    over, and the share of *inverted* (negative) gradients rising from 14,8 %
    to 32,2 % between those two age groups. The correlation is far weaker than
    the -0,73 reported from electrocardiographic imaging -- which is what a
    body-surface projection in a hospital cohort should look like beside 22
    healthy volunteers -- but it is the same phenomenon at 64 times the
    sample size.

``terminal_qrs_speed_uniformity``
    Novak et al. (Comput Methods Programs Biomed 2026) found by SHAP analysis
    that localized entropy and dispersion in the late post-R window dominated
    the detection of late-potential-like signatures. **Their labels were
    synthetic**, injected into MIMIC-IV-ECG beats because no beat-level
    adjudicated set exists, so their AUC > 0,99 says nothing about clinical
    performance. This marker is therefore explicitly a guess: cheap to
    compute, physiologically motivated, and expected to disappoint.

**One marker was proposed and then dropped.** A binary flag for the published
SVG threshold below is not built: the analysis computes operating points from
the continuous marker itself, so a binary copy would consume an extraction
slot to encode a threshold the analysis can apply anyway.
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
from ..delineation.base import RecordDelineation


LITERATURE_2026_VERSION = "1.0.0"
LITERATURE_2026_SOURCE = "literature_2026_v1"

EPS = 1e-12

# The lateral precordial leads Hoffmann's criterion is read across, in the
# order a reader sees them on the printed tracing.
LATERAL_PRECORDIAL_LEADS = ("V4", "V5", "V6")

# The share of the P wave counted as its terminal portion. The terminal P is
# the left-atrial component; 40 % is the conventional split and is stated here
# because the source describes the criterion visually rather than numerically,
# so this number is an interpretation and not a quantity taken from the paper.
P_TERMINAL_FRACTION = 0.40

# A P segment shorter than this cannot be split into a terminal portion worth
# averaging: at 500 Hz, 40 % of 40 ms is eight samples.
MINIMUM_P_SAMPLES = 20

# The late QRS window Novak's SHAP analysis pointed at. ``qrs_late`` is the
# final third of the QRS, which the delineation already provides.
LATE_QRS_SEGMENT = "qrs_late"
MINIMUM_LATE_QRS_SAMPLES = 8


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _baseline_uv(ctx: FeatureContext, lead: str) -> Optional[float]:
    """The PR segment as the zero line, in microvolts.

    The same reference the ST markers in this project use, and the same one a
    reader takes by eye: the flat stretch between the P wave and the QRS.
    """

    bounds = ctx.bounds("pr")
    source = ctx.ecg_12lead
    if bounds is None or source is None or lead not in LEAD_MAP:
        return None
    index = LEAD_MAP[lead]
    if index >= source.shape[1]:
        return None
    window = np.asarray(source[bounds[0] : bounds[1] + 1, index], dtype=float)
    if len(window) < 2 or not np.all(np.isfinite(window)):
        return None
    return float(np.mean(window)) * 1000.0


def _terminal_p_amplitude_uv(ctx: FeatureContext, lead: str) -> Optional[float]:
    """Mean amplitude of the terminal P wave in one lead, against the PR line."""

    segment = ctx.ecg_segment(lead, "p")
    if segment is None or len(segment) < MINIMUM_P_SAMPLES:
        return None
    values = np.asarray(segment, dtype=float) * 1000.0
    if not np.all(np.isfinite(values)):
        return None
    baseline = _baseline_uv(ctx, lead)
    if baseline is None:
        return None
    cut = int(round(len(values) * (1.0 - P_TERMINAL_FRACTION)))
    terminal = values[cut:]
    if len(terminal) < 2:
        return None
    return float(np.mean(terminal) - baseline)


def _p_terminal_progression(ctx: FeatureContext) -> FeatureValue:
    """Slope of the terminal P amplitude across V4, V5 and V6.

    Fitted over the lead positions 0, 1, 2 rather than differenced, so a
    single noisy lead moves the answer less than it would move V6 minus V4.
    Negative values are the "poor progression" the criterion names: the
    left-atrial component fades across the lateral chest leads instead of
    holding up.

    Read from the consensus beat, deliberately. The criterion is one a reader
    applies to the printed twelve-lead, and the representative complex is that
    reading's analogue -- which also means this marker inherits the P
    segment's missingness, as the human reader does.
    """

    amplitudes = [
        _terminal_p_amplitude_uv(ctx, lead) for lead in LATERAL_PRECORDIAL_LEADS
    ]
    if any(value is None for value in amplitudes):
        return _abstain("terminal P wave not measurable in V4-V6")
    positions = np.arange(len(amplitudes), dtype=float)
    values = np.asarray(amplitudes, dtype=float)
    slope = float(np.polyfit(positions, values, 1)[0])
    if not np.isfinite(slope):
        return _abstain("non-finite slope")
    return FeatureValue(slope)


def _apicobasal_gradient(ctx: FeatureContext) -> FeatureValue:
    """Signed cranio-caudal component of the ventricular gradient.

    The project's axis convention is Frank-compatible with **+Y inferior**, so
    the Y component of the QRST area vector is the apicobasal projection of the
    ventricular gradient: positive means the gradient points toward the apex,
    negative means it has inverted toward the base.

    **The number is already computed and then thrown away.** The dispatcher
    behind ``svg_magnitude`` writes ``svg_y_uv_ms`` into that marker's
    provenance metadata -- but metadata is not a feature column, and the
    screening stage reads columns. So the quantity exists on every recording
    and no analysis can reach it.

    It is also not reconstructible downstream: it equals ``svg_magnitude``
    times the cosine of ``svg_elevation``, and that product is not a linear
    combination of the two, so the combination search, which builds linear
    scores, cannot express it. Measured rather than argued -- a least-squares
    fit of this marker on those two reaches **R^2 = 0,62** across 1.364
    recordings, so a linear score leaves nearly two fifths of the variance
    out of reach. A magnitude paired with an angle cannot state
    "inverted" as one ordered quantity, and inversion is exactly what the age
    hypothesis is about.
    """

    metrics = ctx.geh_area_metrics()
    if metrics is None:
        return _abstain("no GEH area metrics")
    vector = getattr(metrics, "qrst_area_vector_uv_ms", None)
    if vector is None:
        return _abstain("no QRST area vector")
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return _abstain("non-finite QRST area vector")
    return FeatureValue(float(values[1]))


def _terminal_qrs_speed_uniformity(ctx: FeatureContext) -> FeatureValue:
    """How evenly the vector moves through the last third of the QRS.

    The normalised Shannon entropy of the step-to-step speed distribution.
    **The direction is the opposite of what "entropy" suggests at first
    glance, which is why this is not called one:** a trajectory that advances
    at a constant rate spreads its steps evenly and reaches entropy 1, while
    bursty, fragmented motion concentrates the distance into a few steps and
    drives the value down. Measured on a straight synthetic ramp the value is
    exactly 1,000; adding noise to that same ramp lowers it to 0,964.

    A late potential is fragmented, low-amplitude activity on the tail of
    depolarisation, so the hypothesis is that fragmentation *lowers* this
    marker. Naming it after dispersion or entropy would have implied the
    reverse and invited every later reader to misread its sign.

    Expect little. The source that motivates this window measured it against
    synthetic late potentials injected into otherwise ordinary beats, so its
    reported discrimination is a property of the injection, not of patients.
    """

    points = ctx.vcg_segment_origin_corrected(LATE_QRS_SEGMENT)
    if points is None:
        return _abstain("no late QRS window")
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        return _abstain("late QRS window is not a trajectory")
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < MINIMUM_LATE_QRS_SAMPLES:
        return _abstain("late QRS window too short")
    speed = np.linalg.norm(np.diff(values, axis=0), axis=1)
    total = float(np.sum(speed))
    if not np.isfinite(total) or total <= EPS:
        return _abstain("degenerate late QRS trajectory")
    weights = speed / total
    weights = weights[weights > EPS]
    if len(weights) < 2:
        return _abstain("speed concentrated in one step")
    entropy = float(-np.sum(weights * np.log(weights)) / np.log(len(weights)))
    if not np.isfinite(entropy):
        return _abstain("non-finite entropy")
    return FeatureValue(float(np.clip(entropy, 0.0, 1.0)))


@dataclass(frozen=True)
class Literature2026Spec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    evidence_note: str
    references: tuple[str, ...] = ()


LITERATURE_2026_SPECS: tuple[Literature2026Spec, ...] = (
    Literature2026Spec(
        "p_terminal_progression_v4_v6",
        "Vorhofsubstrat",
        "Slope of the terminal P-wave amplitude across V4, V5 and V6.",
        "Least-squares slope of mean(terminal 40 % of P, against the PR line) "
        "over the lead positions V4, V5, V6.",
        "uV",
        _p_terminal_progression,
        "One of the two most frequent predictors in a four-criterion visual "
        "model that reached AUC 0,905 against invasive left-atrial voltage "
        "mapping. Readable by eye from the printed twelve-lead, which is why "
        "it is worth having as a number.",
        references=(
            "Hoffmann R et al. Morphology-based ECG criteria for identifying "
            "left-atrial low-voltage areas. Front Cardiovasc Med 2026;13:1891500.",
        ),
    ),
    Literature2026Spec(
        "ventricular_gradient_apicobasal_uv_ms",
        "Repolarisationsgeometrie",
        "Signed apicobasal (cranio-caudal) component of the ventricular gradient.",
        "Y component of the QRST area vector, +Y inferior per the project's "
        "Frank-compatible axis convention.",
        "uV*ms",
        _apicobasal_gradient,
        "The apicobasal repolarization gradient diminishes and inverts with "
        "age (r = -0,73 by electrocardiographic imaging), and the inverted "
        "state carries lower re-entry vulnerability. A candidate mechanism "
        "for this project's own age finding on the spatial QRS-T angle.",
        references=(
            "Sobota V et al. The apicobasal dispersion of ventricular "
            "repolarization in humans is associated with age. J Physiol "
            "2026;604(13):5260-5275.",
        ),
    ),
    Literature2026Spec(
        "terminal_qrs_speed_uniformity",
        "Terminale QRS-Energie",
        "Evenness of the vector speed through the late QRS third; 1 is a "
        "constant rate, lower values are bursty motion.",
        "-sum(p log p)/log(n) over the normalised step-to-step speed of the "
        "origin-corrected trajectory in qrs_late.",
        "fraction",
        _terminal_qrs_speed_uniformity,
        "Localized entropy and dispersion in the late post-R window dominated "
        "a SHAP analysis of late-potential-like signatures. That analysis used "
        "synthetic injected labels, so this is a guess and is registered as "
        "one.",
        references=(
            "Novak A et al. Feasibility of single-beat detection of "
            "ventricular late potentials. Comput Methods Programs Biomed "
            "2026;285:109458.",
        ),
    ),
)


class Literature2026Biomarker(BiomarkerBase):
    def __init__(self, spec: Literature2026Spec) -> None:
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
        notes = [f"literature_2026_version={LITERATURE_2026_VERSION}"]
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


def literature_2026_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in LITERATURE_2026_SPECS)


def register_literature_2026_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in LITERATURE_2026_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), Literature2026Biomarker):
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
                category="vcg",
                source=LITERATURE_2026_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Hypothesis taken from a named 2026 publication and "
                    "registered so that publication's claim can be tested "
                    "rather than assumed. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
                references=list(spec.references),
                implementation_status="implemented",
            )
        )
        BiomarkerRegistry.register(Literature2026Biomarker(spec))


__all__ = [
    "LATERAL_PRECORDIAL_LEADS",
    "LATE_QRS_SEGMENT",
    "LITERATURE_2026_SOURCE",
    "LITERATURE_2026_SPECS",
    "LITERATURE_2026_VERSION",
    "Literature2026Biomarker",
    "Literature2026Spec",
    "P_TERMINAL_FRACTION",
    "literature_2026_marker_ids",
    "register_literature_2026_biomarkers",
]
