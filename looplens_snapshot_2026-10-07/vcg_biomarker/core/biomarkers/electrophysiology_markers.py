"""Electrophysiologically motivated markers filling the catalog's conceptual gaps.

The existing catalog is saturated in one region and empty in others: roughly
760 of its markers describe loop geometry and kinematics, while whole
electrophysiological concepts had no representation at all -- atrial
repolarization, terminal QRS energy, repolarization heterogeneity in its
classical forms, the geometry of the Brugada and early-repolarization
patterns, infarction under bundle branch block, and several rare patterns with
well-described pathophysiology.

Adding a 761st loop descriptor buys almost no independent information and
costs multiplicity budget. Adding a concept that was previously unmeasurable
buys a genuinely new dimension. Everything here is of the second kind.

Three honesty constraints are built in rather than documented afterwards:

* **Terminal QRS energy is not a signal-averaged ECG.** Classical late
  potentials average 200+ beats to reach sub-microvolt noise. Ten seconds
  gives 10-15 beats. These markers use the wideband path and are named as
  terminal-energy descriptors, never as SAECG or "late potentials".
* **Pattern markers are pattern *quantifications*, not diagnoses.** A
  Brugada beta angle is an angle; whether it means Brugada syndrome depends
  on lead placement, fever, drugs and genetics that no single recording
  carries.
* **Every marker abstains** when its segment is unevaluable, with the reason
  attached, rather than returning a number computed from a guess.

Literature anchors are given per marker family in the specification below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import signal as sp_signal

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import (
    EPS,
    MILLIVOLT_TO_MICROVOLT,
    FeatureContext,
    FeatureValue,
    _angle_between,
    _beat_segment_curves,
    _context,
    _path_length,
    _pca,
    _resample_curve,
    _trapz,
)
from ..data_loader import LEAD_MAP
from ..delineation.base import (
    fiducial_confidence_threshold,
    RecordDelineation,
)
from ..vcg_geometry import vcg_azimuth_deg, vcg_elevation_deg


ELECTROPHYSIOLOGY_VERSION = "electrophysiology-gap-v1.3.0-audited"
ELECTROPHYSIOLOGY_SOURCE = "Project electrophysiology gap analysis, 2026-08"

# Family-level literature anchors. Individual markers name their own where the
# definition follows one paper closely.
REFERENCES = {
    "terminal_qrs": (
        "Breithardt et al., Eur Heart J 1991 (late potential standards)",
        "Marcus et al., Circulation 2010 (ARVC Task Force criteria)",
    ),
    "atrial_repolarization": (
        "Debbas et al., J Electrocardiol 1999 (atrial repolarization wave)",
        "Bayes de Luna et al., J Electrocardiol 2012 (interatrial block)",
        "Spodick, Am J Cardiol 1974 (PR and TP segment in pericarditis)",
    ),
    "repolarization_heterogeneity": (
        "Acar et al., Med Biol Eng Comput 1999 (TMD, TCRT)",
        "Malik et al., J Cardiovasc Electrophysiol 2000 (TCRT)",
        "Priori et al., Circulation 1997 (PCA ratio of the T wave)",
    ),
    "brugada": (
        "Chevallier et al., J Am Coll Cardiol 2011 (beta angle, triangle base)",
        "Serra et al., Europace 2014 (triangle base criterion)",
    ),
    "early_repolarization": (
        "Haissaguerre et al., N Engl J Med 2008",
        "Tikkanen et al., Circulation 2011 (horizontal ST is the malignant form)",
    ),
    "bundle_branch_infarction": (
        "Sgarbossa et al., N Engl J Med 1996",
        "Smith et al., Ann Emerg Med 2012 (ST/S ratio)",
        "Cabrera and Friedland, Am Heart J 1953",
    ),
    "ischemia_vector": (
        "de Winter et al., N Engl J Med 2008",
        "de Zwaan et al., Am Heart J 1982 (Wellens)",
        "Aslanger et al., J Electrocardiol 2020",
    ),
    "electrolyte": (
        "Osborn, Am J Physiol 1953 (J wave in hypothermia)",
        "Surawicz, Am Heart J 1967 (electrolyte effects)",
    ),
    "hypertrophy": (
        "https://doi.org/10.1016/j.jacc.2008.12.015",
    ),
    "dynamic_vcg": (
        "https://doi.org/10.1371/journal.pone.0057175",
        "https://doi.org/10.1161/JAHA.114.001357",
    ),
    "loop_morphology": (
        "https://doi.org/10.1016/j.jelectrocard.2015.12.014",
    ),
}


@dataclass(frozen=True)
class _MarkerSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    interpretation: str
    calculator: Callable[[FeatureContext], FeatureValue]
    category: str = "vcg"
    reference_keys: Optional[tuple[str, ...]] = None


_REFERENCE_KEYS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "Terminale QRS-Energie": ("terminal_qrs",),
    "Atriale Repolarisation": ("atrial_repolarization",),
    "Repolarisationsheterogenitaet": ("repolarization_heterogeneity",),
    "Aktivierung-Erholung": ("repolarization_heterogeneity",),
    "ARVC/Terminale Aktivierung": ("terminal_qrs",),
    "Infarkt bei Schenkelblock": ("bundle_branch_infarction",),
    "Ischaemie-Vektorik": ("ischemia_vector",),
    "Hypertrophie-Vektorik": ("hypertrophy",),
    "Medikamenten-Signaturen": ("electrolyte",),
    "Elektrolyt-Signaturen": ("electrolyte",),
    "Atriale Geometrie": ("atrial_repolarization",),
    "QRS-T-Kopplung": ("repolarization_heterogeneity",),
    "Loop-Planaritaet": ("loop_morphology",),
    "Repolarisationslabilitaet": ("dynamic_vcg",),
}

_REFERENCE_KEYS_BY_NAME: dict[str, tuple[str, ...]] = {
    "brugada_beta_angle_deg": ("brugada",),
    "brugada_triangle_base_ms": ("brugada",),
    "brugada_ascending_limb_duration_ms": ("brugada",),
    "j_point_elevation_uv": ("early_repolarization",),
    "st_slope_after_j_uv_per_ms": ("early_repolarization",),
    "early_repolarization_malignant_score": ("early_repolarization",),
    "osborn_j_wave_amplitude_uv": ("electrolyte",),
}


def _references_for_spec(spec: _MarkerSpec) -> list[str]:
    """Return only literature that actually anchors this marker family."""

    keys = spec.reference_keys
    if keys is None:
        keys = _REFERENCE_KEYS_BY_NAME.get(
            spec.name,
            _REFERENCE_KEYS_BY_FAMILY.get(spec.family, ()),
        )
    references: list[str] = []
    for key in keys:
        for reference in REFERENCES.get(key, ()):
            if reference not in references:
                references.append(reference)
    return references


# --- shared helpers --------------------------------------------------------


def _magnitude(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) < 2:
        return None
    return np.linalg.norm(np.asarray(points, dtype=float), axis=1)


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _segment_duration_ms(ctx: FeatureContext, segment: str) -> Optional[float]:
    bounds = ctx.bounds(segment)
    if bounds is None or ctx.fs <= 0:
        return None
    start, end = bounds
    return float((end - start) * 1000.0 / ctx.fs)


def _mean_vector(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) == 0:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _area_vector(points: Optional[np.ndarray], fs: int) -> Optional[np.ndarray]:
    """Time integral of a loop, i.e. its area vector in mV*ms."""

    if points is None or len(points) < 2 or fs <= 0:
        return None
    array = np.asarray(points, dtype=float)
    step_ms = 1000.0 / float(fs)
    return np.trapezoid(array, dx=step_ms, axis=0)


def _slope_uv_per_ms(values: np.ndarray, fs: int) -> float:
    if len(values) < 3 or fs <= 0:
        return float("nan")
    time_ms = np.arange(len(values), dtype=float) * 1000.0 / float(fs)
    slope = float(np.polyfit(time_ms, values, 1)[0])
    return slope * MILLIVOLT_TO_MICROVOLT


def _tp_window(ctx: FeatureContext) -> Optional[np.ndarray]:
    """TP interval: T offset of one beat to P onset (or QRS onset) of the next.

    ``bounds()`` has no "tp" segment because TP spans two beats. It is the only
    genuinely isoelectric stretch of the cardiac cycle and therefore the right
    baseline reference, so it is reconstructed here from the beat list.
    """

    beats = getattr(ctx.delineation, "beats", None)
    vcg = ctx.vcg_segment("global_vcg")
    if not beats or len(beats) < 2 or vcg is None:
        return None
    array = np.asarray(vcg, dtype=float)
    windows: list[np.ndarray] = []
    for first, second in zip(beats[:-1], beats[1:]):
        end_point = getattr(first, "t_offset", None)
        start_attr = next(
            (
                name
                for name in ("p_onset", "r_onset", "q_peak")
                if getattr(second, name, None) is not None
            ),
            None,
        )
        start_point = (
            getattr(second, start_attr) if start_attr is not None else None
        )
        if end_point is None or start_point is None:
            continue
        # Each landmark is judged at its own wave's strictness: the fallback
        # chain above can hand back a P onset or a QRS onset, and those are
        # not calibrated alike.
        if (
            end_point.confidence < fiducial_confidence_threshold("t_offset")
            or start_point.confidence
            < fiducial_confidence_threshold(start_attr)
        ):
            continue
        start = int(end_point.sample)
        end = int(start_point.sample)
        if end - start < 4 or start < 0 or end >= len(array):
            continue
        windows.append(array[start:end])
    if not windows:
        return None
    return max(windows, key=len)


def _baseline_value(ctx: FeatureContext) -> float:
    """Isoelectric reference: TP first, then PR, then a robust low quantile.

    Falling back to zero would be wrong: the VCG magnitude is a norm and is
    therefore never negative, so a zero baseline turns every deviation into
    the raw magnitude.
    """

    tp = _magnitude(_tp_window(ctx))
    if tp is not None and len(tp) >= 3:
        return float(np.median(tp))
    pr = _magnitude(ctx.vcg_segment("pr"))
    if pr is not None and len(pr) >= 3:
        return float(np.median(pr))
    global_magnitude = _magnitude(ctx.vcg_segment("global_vcg"))
    if global_magnitude is not None and len(global_magnitude) >= 10:
        return float(np.quantile(global_magnitude, 0.10))
    return 0.0


def _representative_lead(ctx: FeatureContext, lead: str, segment: str):
    try:
        return ctx.ecg_segment(lead, segment)
    except (KeyError, ValueError, TypeError):
        return None


def _lead_baseline_mv(ctx: FeatureContext, lead: str) -> float:
    """Lead-specific isoelectric level, preferring the qualified PR segment."""

    pr = _representative_lead(ctx, lead, "pr")
    if pr is not None and len(pr) >= 3 and np.all(np.isfinite(pr)):
        return float(np.median(pr))
    return 0.0


# --- 1. terminal QRS energy (wideband) -------------------------------------


def _terminal_window(points: np.ndarray, fs: int, window_ms: float) -> np.ndarray:
    samples = max(3, int(round(window_ms * fs / 1000.0)))
    return points[-samples:] if len(points) > samples else points


def _terminal_qrs_rms40(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment_wideband("qrs")
    if points is None:
        return _abstain("no wideband source for terminal QRS energy")
    magnitude = _magnitude(points)
    if magnitude is None:
        return _abstain("QRS segment too short")
    tail = _terminal_window(magnitude, ctx.fs, 40.0)
    baseline = _baseline_value(ctx)
    return FeatureValue(
        float(np.sqrt(np.mean((tail - baseline) ** 2)) * MILLIVOLT_TO_MICROVOLT)
    )


def _low_amplitude_terminal_duration(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment_wideband("qrs")
    if points is None:
        return _abstain("no wideband source")
    magnitude = _magnitude(points)
    if magnitude is None or ctx.fs <= 0:
        return _abstain("QRS segment too short")
    baseline = _baseline_value(ctx)
    deviation = np.abs(magnitude - baseline) * MILLIVOLT_TO_MICROVOLT
    threshold = 40.0
    count = 0
    for value in deviation[::-1]:
        if value < threshold:
            count += 1
        else:
            break
    return FeatureValue(float(count * 1000.0 / ctx.fs))


def _terminal_qrs_hf_ratio(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment_wideband("qrs")
    if points is None:
        return _abstain("no wideband source")
    if len(points) < 16:
        return _abstain("QRS segment too short for a band comparison")
    nyquist = ctx.fs / 2.0
    low = 80.0
    high = 150.0
    if nyquist <= high:
        return _abstain("sampling rate does not support the fixed 80-150 Hz band")
    sos = sp_signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(points) <= padlen:
        return _abstain("QRS shorter than the zero-phase filter requirement")
    # Bandpass the signed physical axes before any nonlinear magnitude.
    filtered_xyz = sp_signal.sosfiltfilt(sos, points, axis=0)
    spatial_energy = np.sum(filtered_xyz ** 2, axis=1)
    tail = _terminal_window(spatial_energy, ctx.fs, 40.0)
    total = float(np.sum(spatial_energy))
    if total <= EPS:
        return _abstain("no high-frequency energy in the QRS")
    return FeatureValue(
        float(np.sum(tail) / total),
        notes="Axiswise XYZ 80-150 Hz bandpass followed by spatial-energy aggregation.",
    )


def _filtered_qrs_duration(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment_wideband("qrs")
    if points is None:
        return _abstain("no wideband source")
    magnitude = _magnitude(points)
    if magnitude is None or ctx.fs <= 0:
        return _abstain("QRS segment too short")
    baseline = _baseline_value(ctx)
    deviation = np.abs(magnitude - baseline) * MILLIVOLT_TO_MICROVOLT
    above = np.flatnonzero(deviation >= 40.0)
    if len(above) < 2:
        return _abstain("no samples above the 40 uV envelope threshold")
    return FeatureValue(
        float((above[-1] - above[0] + 1) * 1000.0 / ctx.fs)
    )


# --- 2. atrial repolarization ----------------------------------------------


def _ta_vector(ctx: FeatureContext) -> Optional[np.ndarray]:
    """Atrial repolarization vector, read from the PR segment deviation.

    The Ta wave is opposite in direction to the P wave and is normally buried
    under the PR segment and the QRS. What is observable in a surface
    recording is the PR-segment displacement relative to the TP baseline.
    """

    pr = ctx.vcg_segment("pr")
    if pr is None or len(pr) < 3:
        return None
    tp = _tp_window(ctx)
    reference = (
        np.median(np.asarray(tp, dtype=float), axis=0)
        if tp is not None and len(tp) >= 3
        else np.zeros(3, dtype=float)
    )
    return np.mean(np.asarray(pr, dtype=float), axis=0) - reference


def _ta_vector_magnitude(ctx: FeatureContext) -> FeatureValue:
    vector = _ta_vector(ctx)
    if vector is None:
        return _abstain("PR segment not evaluable")
    return FeatureValue(
        float(np.linalg.norm(vector) * MILLIVOLT_TO_MICROVOLT)
    )


def _ta_to_p_angle(ctx: FeatureContext) -> FeatureValue:
    vector = _ta_vector(ctx)
    p_points = ctx.vcg_segment("p")
    p_vector = _mean_vector(p_points)
    if vector is None or p_vector is None:
        return _abstain("P or PR segment not evaluable")
    angle = _angle_between(vector, p_vector)
    if angle is None:
        return _abstain("degenerate P or Ta vector")
    return FeatureValue(float(angle))


def _atrial_gradient_magnitude(ctx: FeatureContext) -> FeatureValue:
    """Atrial analogue of the ventricular gradient.

    The ventricular gradient is the sum of the QRS and T area vectors and is
    independent of activation sequence, which is what makes it a measure of
    intrinsic repolarization heterogeneity. The same construction on the
    atrium has essentially never been screened at scale.
    """

    p_area = _area_vector(ctx.vcg_segment("p"), ctx.fs)
    ta_vector = _ta_vector(ctx)
    pr_duration = _segment_duration_ms(ctx, "pr")
    if p_area is None or ta_vector is None or pr_duration is None:
        return _abstain("P or PR segment not evaluable")
    ta_area = ta_vector * float(pr_duration)
    return FeatureValue(
        float(np.linalg.norm(p_area + ta_area) * MILLIVOLT_TO_MICROVOLT)
    )


def _atrial_gradient_to_p_angle(ctx: FeatureContext) -> FeatureValue:
    p_area = _area_vector(ctx.vcg_segment("p"), ctx.fs)
    ta_vector = _ta_vector(ctx)
    pr_duration = _segment_duration_ms(ctx, "pr")
    if p_area is None or ta_vector is None or pr_duration is None:
        return _abstain("P or PR segment not evaluable")
    gradient = p_area + ta_vector * float(pr_duration)
    angle = _angle_between(gradient, p_area)
    if angle is None:
        return _abstain("degenerate atrial gradient")
    return FeatureValue(float(angle))


def _pr_segment_deviation(ctx: FeatureContext) -> FeatureValue:
    vector = _ta_vector(ctx)
    if vector is None:
        return _abstain("PR segment not evaluable")
    # Signed along the dominant P direction: negative means PR depression.
    p_vector = _mean_vector(ctx.vcg_segment("p"))
    if p_vector is None or np.linalg.norm(p_vector) < EPS:
        return FeatureValue(
            float(np.linalg.norm(vector) * MILLIVOLT_TO_MICROVOLT)
        )
    unit = p_vector / np.linalg.norm(p_vector)
    return FeatureValue(float(np.dot(vector, unit) * MILLIVOLT_TO_MICROVOLT))


def _tp_segment_slope(ctx: FeatureContext) -> FeatureValue:
    """Spodick sign: a downsloping TP segment, described in pericarditis."""

    magnitude = _magnitude(_tp_window(ctx))
    if magnitude is None or len(magnitude) < 5:
        return _abstain("no evaluable TP interval between consecutive beats")
    return FeatureValue(_slope_uv_per_ms(magnitude, ctx.fs))


# --- 3. repolarization heterogeneity ---------------------------------------


def _total_cosine_r_to_t(ctx: FeatureContext) -> FeatureValue:
    """TCRT: mean cosine between QRS and T vectors over the loops.

    Acar/Malik's descriptor of how far repolarization runs against the
    direction of depolarization. Distinct from the peak-to-peak QRS-T angle
    because it integrates over the whole loop rather than two peak vectors.
    """

    qrs = ctx.vcg_segment("qrs")
    t_points = ctx.vcg_segment("t")
    if qrs is None or t_points is None or len(qrs) < 4 or len(t_points) < 4:
        return _abstain("QRS or T loop not evaluable")
    t_mean = np.mean(np.asarray(t_points, dtype=float), axis=0)
    t_norm = np.linalg.norm(t_mean)
    if t_norm < EPS:
        return _abstain("degenerate T loop")
    qrs_array = np.asarray(qrs, dtype=float)
    norms = np.linalg.norm(qrs_array, axis=1)
    usable = norms > EPS
    if usable.sum() < 3:
        return _abstain("degenerate QRS loop")
    cosines = (qrs_array[usable] @ t_mean) / (norms[usable] * t_norm)
    return FeatureValue(float(np.mean(np.clip(cosines, -1.0, 1.0))))


def _t_morphology_dispersion(ctx: FeatureContext) -> FeatureValue:
    """TMD: mean angle between the reconstruction vectors of the T loop.

    Computed here as the mean pairwise angle between the T-loop samples and
    the loop's own principal direction, which is the vector-space equivalent
    of the lead-space formulation.
    """

    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 6:
        return _abstain("T loop not evaluable")
    array = np.asarray(t_points, dtype=float)
    _, eigenvalues, eigenvectors = _pca(array)
    if eigenvalues is None or len(eigenvalues) < 1 or eigenvalues[0] <= EPS:
        return _abstain("T loop principal direction undefined")
    principal = np.asarray(eigenvectors[:, 0], dtype=float)
    norms = np.linalg.norm(array, axis=1)
    usable = norms > EPS
    if usable.sum() < 4:
        return _abstain("degenerate T loop")
    angles = []
    for point in array[usable]:
        angle = _angle_between(point, principal)
        if angle is not None:
            angles.append(min(angle, 180.0 - angle))
    if not angles:
        return _abstain("no usable T-loop angles")
    return FeatureValue(float(np.mean(angles)))


def _t_pca_ratio(ctx: FeatureContext) -> FeatureValue:
    """PCA ratio lambda2/lambda1 of the T loop: T-wave complexity."""

    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 6:
        return _abstain("T loop not evaluable")
    _, eigenvalues, _ = _pca(np.asarray(t_points, dtype=float))
    if eigenvalues is None or len(eigenvalues) < 2 or eigenvalues[0] <= EPS:
        return _abstain("degenerate T loop covariance")
    return FeatureValue(float(eigenvalues[1] / eigenvalues[0]))


def _t_nondipolar_residual(ctx: FeatureContext) -> FeatureValue:
    # A three-dimensional VCG has no fourth through eighth dimensions and
    # therefore cannot measure the classical nondipolar T-wave residuum.  The
    # former lambda3/total implementation measured out-of-plane nonplanarity,
    # not TWR, and is retired rather than silently relabelled.
    return _abstain(
        "retired invalid definition: 3D lambda3 is loop nonplanarity, not "
        "8-lead T-wave residuum; use lead8_t_wave_residuum_fraction"
    )


def _repolarization_direction_change_rate(ctx: FeatureContext) -> FeatureValue:
    """Mean angular velocity of the T vector, in degrees per millisecond.

    Amplitude-independent by construction, which should make it markedly more
    repeatable than magnitude-based repolarization descriptors -- a property
    the repeatability layer can now actually verify.
    """

    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 5 or ctx.fs <= 0:
        return _abstain("T loop not evaluable")
    array = np.asarray(t_points, dtype=float)
    angles = []
    for first, second in zip(array[:-1], array[1:]):
        angle = _angle_between(first, second)
        if angle is not None:
            angles.append(angle)
    if not angles:
        return _abstain("degenerate T loop")
    step_ms = 1000.0 / float(ctx.fs)
    return FeatureValue(float(np.mean(angles) / step_ms))


def _t_wave_symmetry(ctx: FeatureContext) -> FeatureValue:
    """Ascending vs descending limb duration of the T loop magnitude.

    A symmetric T wave is a classical ischaemia marker; the normal T wave is
    asymmetric with a slower upstroke.
    """

    t_points = ctx.vcg_segment("t")
    magnitude = _magnitude(t_points)
    if magnitude is None or len(magnitude) < 5:
        return _abstain("T loop not evaluable")
    peak = int(np.argmax(magnitude))
    if peak <= 0 or peak >= len(magnitude) - 1:
        return _abstain("T peak at the segment boundary")
    return FeatureValue(float(peak / (len(magnitude) - 1 - peak)))


# --- 4. activation-recovery coupling ---------------------------------------


def _initial_qrs_vector(ctx: FeatureContext, window_ms: float = 20.0) -> Optional[np.ndarray]:
    qrs = ctx.vcg_segment("qrs")
    if qrs is None or ctx.fs <= 0:
        return None
    array = np.asarray(qrs, dtype=float)
    samples = max(2, int(round(window_ms * ctx.fs / 1000.0)))
    return np.mean(array[:samples], axis=0) if len(array) >= 2 else None


def _terminal_t_vector(ctx: FeatureContext, window_ms: float = 30.0) -> Optional[np.ndarray]:
    t_points = ctx.vcg_segment("t")
    if t_points is None or ctx.fs <= 0:
        return None
    array = np.asarray(t_points, dtype=float)
    samples = max(2, int(round(window_ms * ctx.fs / 1000.0)))
    return np.mean(array[-samples:], axis=0) if len(array) >= 2 else None


def _initial_qrs_to_terminal_t_angle(ctx: FeatureContext) -> FeatureValue:
    """Angle between septal activation and the last repolarizing region.

    The ventricular gradient asks whether repolarization as a whole opposes
    depolarization as a whole. This asks the sharper question: does the region
    that depolarizes *first* also repolarize *last*? In a normal heart it does,
    which is why the T wave is concordant.
    """

    initial = _initial_qrs_vector(ctx)
    terminal = _terminal_t_vector(ctx)
    if initial is None or terminal is None:
        return _abstain("QRS or T loop not evaluable")
    angle = _angle_between(initial, terminal)
    if angle is None:
        return _abstain("degenerate initial or terminal vector")
    return FeatureValue(float(angle))


def _activation_recovery_coupling_index(ctx: FeatureContext) -> FeatureValue:
    initial = _initial_qrs_vector(ctx)
    terminal = _terminal_t_vector(ctx)
    if initial is None or terminal is None:
        return _abstain("QRS or T loop not evaluable")
    norm_product = np.linalg.norm(initial) * np.linalg.norm(terminal)
    if norm_product < EPS:
        return _abstain("degenerate vectors")
    return FeatureValue(float(np.dot(initial, terminal) / norm_product))


def _septal_activation_azimuth(ctx: FeatureContext) -> FeatureValue:
    initial = _initial_qrs_vector(ctx)
    if initial is None:
        return _abstain("QRS onset not evaluable")
    return FeatureValue(float(np.degrees(np.arctan2(initial[2], initial[0]))))


# --- 5. Brugada and early repolarization geometry --------------------------


def _j_point_index(ctx: FeatureContext) -> Optional[int]:
    bounds = ctx.bounds("qrs")
    return bounds[1] if bounds is not None else None


def _brugada_beta_angle(ctx: FeatureContext) -> FeatureValue:
    """Angle between the ascending r' limb and the descending ST limb.

    Chevallier's beta angle: wider means a more triangular, more coved
    pattern. Computed on the vector magnitude around the J point rather than
    on V1/V2 alone, so lead placement matters less.
    """

    return _abstain(
        "clinical beta angle requires the S-wave upslope and r-prime "
        "downslope in V1/V2 at standard ECG calibration; VCG magnitude is "
        "not an equivalent construction"
    )


def _brugada_triangle_base(ctx: FeatureContext) -> FeatureValue:
    """Width of the ST triangle measured 0.5 mV below its apex."""

    return _abstain(
        "clinical Brugada triangle base requires an r-prime morphology in "
        "V1/V2; a spatial-magnitude ST width is not equivalent"
    )


def _brugada_ascending_limb_duration(ctx: FeatureContext) -> FeatureValue:
    return _abstain(
        "clinical Brugada limb duration requires an r-prime morphology in "
        "V1/V2; a QRS-magnitude peak-to-nadir duration is not equivalent"
    )


def _j_point_elevation(ctx: FeatureContext) -> FeatureValue:
    index = _j_point_index(ctx)
    vcg = ctx.vcg_segment("global_vcg")
    if index is None or vcg is None or index >= len(vcg):
        return _abstain("J point not evaluable")
    magnitude = float(np.linalg.norm(np.asarray(vcg[index], dtype=float)))
    baseline = _baseline_value(ctx)
    return FeatureValue(float((magnitude - baseline) * MILLIVOLT_TO_MICROVOLT))


def _st_slope_after_j(ctx: FeatureContext) -> FeatureValue:
    """ST slope in the 60 ms after the J point.

    The discriminating part of the early-repolarization literature: a
    horizontal or descending ST after the J wave carries the arrhythmic risk,
    an ascending one does not.
    """

    st = ctx.vcg_segment("st")
    magnitude = _magnitude(st)
    if magnitude is None or len(magnitude) < 4 or ctx.fs <= 0:
        return _abstain("ST segment not evaluable")
    samples = max(3, int(round(60.0 * ctx.fs / 1000.0)))
    return FeatureValue(_slope_uv_per_ms(magnitude[:samples], ctx.fs))


def _early_repolarization_pattern_score(ctx: FeatureContext) -> FeatureValue:
    elevation = _j_point_elevation(ctx)
    slope = _st_slope_after_j(ctx)
    if elevation.value is None or slope.value is None:
        return _abstain("J point or ST slope not evaluable")
    # High J point with a flat or descending ST is the malignant combination.
    malignant = float(elevation.value) * float(max(0.0, -slope.value) + 1.0)
    return FeatureValue(malignant / 1000.0)


def _osborn_j_wave_amplitude(ctx: FeatureContext) -> FeatureValue:
    """Amplitude of a J-point hump, as seen in hypothermia and hypercalcaemia."""

    qrs = ctx.vcg_segment("qrs")
    st = ctx.vcg_segment("st")
    qrs_magnitude = _magnitude(qrs)
    st_magnitude = _magnitude(st)
    if qrs_magnitude is None or st_magnitude is None or len(st_magnitude) < 4:
        return _abstain("QRS or ST segment not evaluable")
    junction = np.concatenate((qrs_magnitude[-3:], st_magnitude[: len(st_magnitude) // 2]))
    if len(junction) < 4:
        return _abstain("junction window too short")
    baseline = _baseline_value(ctx)
    return FeatureValue(
        float((np.max(junction) - baseline) * MILLIVOLT_TO_MICROVOLT)
    )


# --- 6. ARVC / terminal activation -----------------------------------------


def _terminal_activation_duration(ctx: FeatureContext) -> FeatureValue:
    """Time from the QRS magnitude nadir to the end of the QRS.

    The ARVC Task Force uses the S-wave nadir to QRS end in V1-V3; the vector
    equivalent is the last local minimum of the magnitude to the QRS offset.
    """

    if ctx.fs <= 0 or ctx.ecg_12lead is None:
        return _abstain("right-precordial ECG leads unavailable")
    durations: dict[str, float] = {}
    for lead in ("V1", "V2", "V3"):
        qrs = _representative_lead(ctx, lead, "qrs")
        if qrs is None or len(qrs) < 6:
            continue
        waveform = np.asarray(qrs, dtype=float) - _lead_baseline_mv(ctx, lead)
        if not np.all(np.isfinite(waveform)):
            continue
        span = float(np.ptp(waveform))
        if span <= EPS:
            continue
        r_peak = int(np.argmax(waveform))
        # Exclude only the final sample: allowing it made a baseline-valued
        # QRS endpoint the "S nadir" and forced TAD to zero on most records.
        search_stop = len(waveform) - 1
        if search_stop - r_peak < 2:
            continue
        s_nadir = r_peak + int(np.argmin(waveform[r_peak:search_stop]))
        negative_threshold = max(0.025, 0.05 * span)
        if waveform[s_nadir] >= -negative_threshold:
            continue
        durations[lead] = float(
            (len(waveform) - 1 - s_nadir) * 1000.0 / ctx.fs
        )
    if not durations:
        return _abstain("no qualified S-wave nadir in V1-V3")
    return FeatureValue(max(durations.values()), per_lead=durations)


def _epsilon_potential_energy_ratio(ctx: FeatureContext) -> FeatureValue:
    """Energy in the 40 ms after the QRS relative to the QRS itself.

    An epsilon wave is a small post-excitation potential; expressed as an
    energy ratio it becomes a continuous marker rather than a binary sighting.
    """

    qrs = ctx.vcg_segment_wideband("qrs")
    st = ctx.vcg_segment_wideband("st")
    qrs_magnitude = _magnitude(qrs)
    st_magnitude = _magnitude(st)
    if qrs_magnitude is None or st_magnitude is None:
        return _abstain("no wideband QRS or ST source")
    baseline = _baseline_value(ctx)
    samples = max(3, int(round(40.0 * ctx.fs / 1000.0)))
    early_st = st_magnitude[:samples] - baseline
    qrs_energy = float(np.sum((qrs_magnitude - baseline) ** 2))
    if qrs_energy <= EPS:
        return _abstain("zero QRS energy")
    return FeatureValue(float(np.sum(early_st ** 2) / qrs_energy))


# --- 7. infarction under bundle branch block -------------------------------


def _concordant_st_vector_ratio(ctx: FeatureContext) -> FeatureValue:
    """Projection of the ST vector onto the QRS direction.

    Vectorial Sgarbossa: concordant ST deviation is the criterion with the
    highest specificity for occlusion under left bundle branch block or
    ventricular pacing, and it is a projection, not a per-lead threshold.
    """

    st = ctx.vcg_segment("st")
    qrs = ctx.vcg_segment("qrs")
    st_vector = _mean_vector(st)
    qrs_vector = _mean_vector(qrs)
    if st_vector is None or qrs_vector is None:
        return _abstain("ST or QRS segment not evaluable")
    qrs_norm = np.linalg.norm(qrs_vector)
    if qrs_norm < EPS:
        return _abstain("degenerate QRS vector")
    return FeatureValue(float(np.dot(st_vector, qrs_vector) / (qrs_norm ** 2)))


def _smith_modified_st_to_s_ratio(ctx: FeatureContext) -> FeatureValue:
    """ST deviation relative to the preceding QRS excursion.

    Smith's proportional criterion, in vector form: the ST shift is judged
    against the size of the QRS that produced it rather than against a fixed
    millimetre cut-off.
    """

    if ctx.ecg_12lead is None:
        return _abstain("12-lead ECG unavailable")
    ratios: dict[str, float] = {}
    for lead in LEAD_MAP:
        qrs = _representative_lead(ctx, lead, "qrs")
        st = _representative_lead(ctx, lead, "st")
        if qrs is None or st is None or len(qrs) < 3 or len(st) < 1:
            continue
        baseline = _lead_baseline_mv(ctx, lead)
        qrs_wave = np.asarray(qrs, dtype=float) - baseline
        s_nadir = float(np.min(qrs_wave))
        r_peak = float(np.max(qrs_wave))
        # Smith's modified third criterion is lead-specific discordant ST
        # elevation in a lead whose QRS is predominantly negative.  The
        # signed denominator preserves the published negative ratio.
        if s_nadir >= -EPS or abs(s_nadir) <= max(r_peak, 0.0):
            continue
        st_elevation = float(np.mean(np.asarray(st[: max(1, int(round(0.02 * ctx.fs)))], dtype=float)) - baseline)
        if st_elevation <= 0:
            continue
        ratios[lead] = st_elevation / abs(s_nadir)
    if not ratios:
        return _abstain("no lead with discordant ST elevation and a dominant S wave")
    # Positive magnitude form; larger means more excessive discordance.
    return FeatureValue(max(ratios.values()), per_lead=ratios)


def _qrs_notch_burden_early(ctx: FeatureContext) -> FeatureValue:
    """Notching in the first half of the QRS (Cabrera and Chapman signs)."""

    qrs = ctx.vcg_segment("qrs")
    magnitude = _magnitude(qrs)
    if magnitude is None or len(magnitude) < 8:
        return _abstain("QRS segment not evaluable")
    half = magnitude[: len(magnitude) // 2]
    derivative = np.diff(half)
    sign_changes = int(np.sum(np.diff(np.sign(derivative)) != 0))
    return FeatureValue(float(sign_changes))


# --- 8. ischaemia vectors ---------------------------------------------------


def _st_deviation_vector_magnitude(ctx: FeatureContext) -> FeatureValue:
    st_vector = _mean_vector(ctx.vcg_segment("st"))
    if st_vector is None:
        return _abstain("ST segment not evaluable")
    baseline = _baseline_value(ctx)
    return FeatureValue(
        float((np.linalg.norm(st_vector) - baseline) * MILLIVOLT_TO_MICROVOLT)
    )


def _st_injury_vector_azimuth(ctx: FeatureContext) -> FeatureValue:
    """Azimuth of the ST injury vector: it points at the ischaemic region."""

    st_vector = _mean_vector(ctx.vcg_segment_origin_corrected("st"))
    if st_vector is None or np.linalg.norm(st_vector) < EPS:
        return _abstain("ST vector degenerate")
    value = vcg_azimuth_deg(st_vector)
    return (
        _abstain("ST vector azimuth undefined")
        if value is None
        else FeatureValue(float(value))
    )


def _st_injury_vector_elevation(ctx: FeatureContext) -> FeatureValue:
    st_vector = _mean_vector(ctx.vcg_segment_origin_corrected("st"))
    if st_vector is None:
        return _abstain("ST segment not evaluable")
    value = vcg_elevation_deg(st_vector)
    return (
        _abstain("ST vector elevation undefined")
        if value is None
        else FeatureValue(float(value))
    )


def _hyperacute_t_to_qrs_area_ratio(ctx: FeatureContext) -> FeatureValue:
    """T area relative to QRS area: the de Winter and hyperacute-T signature."""

    t_area = _area_vector(ctx.vcg_segment("t"), ctx.fs)
    qrs_area = _area_vector(ctx.vcg_segment("qrs"), ctx.fs)
    if t_area is None or qrs_area is None:
        return _abstain("T or QRS segment not evaluable")
    qrs_norm = float(np.linalg.norm(qrs_area))
    if qrs_norm < EPS:
        return _abstain("degenerate QRS area")
    return FeatureValue(float(np.linalg.norm(t_area) / qrs_norm))


def _terminal_t_negativity_ratio(ctx: FeatureContext) -> FeatureValue:
    """Terminal T inversion relative to the T peak: the Wellens signature."""

    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 6:
        return _abstain("T loop not evaluable")
    array = np.asarray(t_points, dtype=float)
    peak_vector = array[int(np.argmax(np.linalg.norm(array, axis=1)))]
    peak_norm = np.linalg.norm(peak_vector)
    if peak_norm < EPS:
        return _abstain("degenerate T peak")
    tail = array[-max(2, len(array) // 4):]
    projection = float(np.mean(tail @ peak_vector) / peak_norm)
    return FeatureValue(-projection / float(peak_norm))


def _aslanger_pattern_score(ctx: FeatureContext) -> FeatureValue:
    """Inferior injury with concurrent apical-lateral opposition.

    Aslanger's pattern is inferior ST elevation with lateral ST depression
    and a positive T in V1, described as inferior occlusion on top of
    multivessel disease. In vector terms: an inferiorly directed ST vector
    whose lateral projection is opposed.
    """

    st_vector = _mean_vector(ctx.vcg_segment("st"))
    t_vector = _mean_vector(ctx.vcg_segment("t"))
    if st_vector is None or t_vector is None:
        return _abstain("ST or T segment not evaluable")
    inferior = float(st_vector[1])
    lateral = float(-st_vector[0])
    anterior_t = float(t_vector[2])
    return FeatureValue(
        float(inferior * MILLIVOLT_TO_MICROVOLT)
        * float(max(0.0, -lateral) + 1.0)
        * float(1.0 + max(0.0, anterior_t))
        / 1000.0
    )


# --- 9. hypertrophy, electrolytes, drug effects -----------------------------


def _spatial_qrs_voltage_sum(ctx: FeatureContext) -> FeatureValue:
    magnitude = _magnitude(ctx.vcg_segment("qrs"))
    if magnitude is None:
        return _abstain("QRS segment not evaluable")
    return FeatureValue(float(np.max(magnitude) * MILLIVOLT_TO_MICROVOLT))


def _qrs_voltage_duration_product(ctx: FeatureContext) -> FeatureValue:
    """Vector analogue of the Cornell product."""

    magnitude = _magnitude(ctx.vcg_segment("qrs"))
    duration = _segment_duration_ms(ctx, "qrs")
    if magnitude is None or duration is None:
        return _abstain("QRS segment not evaluable")
    return FeatureValue(
        float(np.max(magnitude) * MILLIVOLT_TO_MICROVOLT * duration)
    )


def _katz_wachtel_equiphasic_index(ctx: FeatureContext) -> FeatureValue:
    """Maximum R-plus-S height in the mid-precordial leads, in uV.

    Katz-Wachtel is a lead-space voltage phenomenon (large biphasic RS in
    V2-V4), not a dimensionless VCG area/path subtraction.  Leadwise balance
    is retained as metadata so discovery models can distinguish a genuinely
    equiphasic complex from a large but one-sided excursion.
    """

    totals: dict[str, float] = {}
    metadata: dict[str, float] = {}
    for lead in ("V2", "V3", "V4"):
        qrs = _representative_lead(ctx, lead, "qrs")
        if qrs is None or len(qrs) < 3:
            continue
        waveform = np.asarray(qrs, dtype=float) - _lead_baseline_mv(ctx, lead)
        r_height = max(0.0, float(np.max(waveform)))
        s_depth = max(0.0, float(-np.min(waveform)))
        if min(r_height, s_depth) <= EPS:
            continue
        total_uv = (r_height + s_depth) * MILLIVOLT_TO_MICROVOLT
        totals[lead] = total_uv
        metadata[f"{lead}_total_rs_uv"] = total_uv
        metadata[f"{lead}_equiphasic_balance"] = 2.0 * min(r_height, s_depth) / (
            r_height + s_depth
        )
    if not totals:
        return _abstain("no biphasic RS complex in V2-V4")
    return FeatureValue(max(totals.values()), per_lead=metadata)


def _terminal_40ms_rightward_axis(ctx: FeatureContext) -> FeatureValue:
    """Rightward shift of the terminal 40 ms QRS vector.

    The vector equivalent of the terminal R wave in aVR, the classical marker
    of sodium-channel blockade -- tricyclic overdose, flecainide, cocaine.
    """

    qrs = ctx.vcg_segment("qrs")
    if qrs is None or ctx.fs <= 0:
        return _abstain("QRS segment not evaluable")
    array = np.asarray(qrs, dtype=float)
    samples = max(2, int(round(40.0 * ctx.fs / 1000.0)))
    if len(array) < samples:
        return _abstain("QRS shorter than the terminal window")
    terminal = np.mean(array[-samples:], axis=0)
    return FeatureValue(float(-terminal[0] * MILLIVOLT_TO_MICROVOLT))


def _terminal_40ms_superior_deviation(ctx: FeatureContext) -> FeatureValue:
    qrs = ctx.vcg_segment("qrs")
    if qrs is None or ctx.fs <= 0:
        return _abstain("QRS segment not evaluable")
    array = np.asarray(qrs, dtype=float)
    samples = max(2, int(round(40.0 * ctx.fs / 1000.0)))
    if len(array) < samples:
        return _abstain("QRS shorter than the terminal window")
    terminal = np.mean(array[-samples:], axis=0)
    return FeatureValue(float(-terminal[1] * MILLIVOLT_TO_MICROVOLT))


def _digitalis_st_sagging_index(ctx: FeatureContext) -> FeatureValue:
    """Concavity of the ST segment: the sagging digitalis effect."""

    magnitude = _magnitude(ctx.vcg_segment("st"))
    if magnitude is None or len(magnitude) < 5:
        return _abstain("ST segment not evaluable")
    normalized = np.linspace(0.0, 1.0, len(magnitude))
    quadratic = np.polyfit(normalized, magnitude, 2)
    return FeatureValue(float(quadratic[0] * MILLIVOLT_TO_MICROVOLT))


def _st_segment_duration(ctx: FeatureContext) -> FeatureValue:
    """ST duration: long with a normal T is the hypocalcaemia signature."""

    duration = _segment_duration_ms(ctx, "st")
    if duration is None:
        return _abstain("ST segment not evaluable")
    return FeatureValue(float(duration))


def _t_base_narrowing_index(ctx: FeatureContext) -> FeatureValue:
    """T amplitude relative to T duration: peaked T waves in hyperkalaemia."""

    magnitude = _magnitude(ctx.vcg_segment("t"))
    duration = _segment_duration_ms(ctx, "t")
    if magnitude is None or duration is None or duration <= 0:
        return _abstain("T segment not evaluable")
    return FeatureValue(
        float(np.max(magnitude) * MILLIVOLT_TO_MICROVOLT / duration)
    )


def _p_amplitude_loss_ratio(ctx: FeatureContext) -> FeatureValue:
    """P amplitude relative to QRS amplitude: atrial standstill in hyperkalaemia."""

    p_magnitude = _magnitude(ctx.vcg_segment("p"))
    qrs_magnitude = _magnitude(ctx.vcg_segment("qrs"))
    if p_magnitude is None or qrs_magnitude is None:
        return _abstain("P or QRS segment not evaluable")
    qrs_peak = float(np.max(qrs_magnitude))
    if qrs_peak < EPS:
        return _abstain("degenerate QRS")
    return FeatureValue(float(np.max(p_magnitude) / qrs_peak))


# --- 10. diastolic-motivated P-loop markers --------------------------------


def _p_loop_terminal_descent_rate(ctx: FeatureContext) -> FeatureValue:
    """Fall rate of the P vector in its final third.

    Terminal atrial activation is the left-atrial component; a slow terminal
    descent is the vector correlate of the terminal force that tracks left
    atrial pressure and size.
    """

    magnitude = _magnitude(ctx.vcg_segment("p"))
    if magnitude is None or len(magnitude) < 6 or ctx.fs <= 0:
        return _abstain("P loop not evaluable")
    tail = magnitude[-max(3, len(magnitude) // 3):]
    return FeatureValue(_slope_uv_per_ms(tail, ctx.fs))


def _p_loop_roundness(ctx: FeatureContext) -> FeatureValue:
    """Isoperimetric roundness of the P loop.

    A round P loop indicates a smooth single activation front; a flattened or
    figure-of-eight loop indicates conduction delay between the atria.
    """

    p_points = ctx.vcg_segment("p")
    if p_points is None or len(p_points) < 6:
        return _abstain("P loop not evaluable")
    array = np.asarray(p_points, dtype=float)
    perimeter = _path_length(array) + float(np.linalg.norm(array[-1] - array[0]))
    if perimeter < EPS:
        return _abstain("degenerate P loop")
    _, eigenvalues, _ = _pca(array)
    if eigenvalues is None or len(eigenvalues) < 2:
        return _abstain("degenerate P loop covariance")
    # Uniform points on an ellipse have covariance eigenvalues a^2/2 and
    # b^2/2.  The former pi*sqrt(lambda1*lambda2) estimate was therefore half
    # the ellipse area and made a perfect circle converge to 0.5, not 1.0.
    area = float(
        2.0
        * np.pi
        * np.sqrt(max(eigenvalues[0], 0.0) * max(eigenvalues[1], 0.0))
    )
    return FeatureValue(float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0)))


def _interatrial_block_vector_score(ctx: FeatureContext) -> FeatureValue:
    """P duration combined with a posterior-superior terminal P vector.

    Bayes de Luna's advanced interatrial block is a long P wave with a
    biphasic terminal component pointing back and up -- a direction, which is
    what a vector representation states directly.
    """

    duration = _segment_duration_ms(ctx, "p")
    p_points = ctx.vcg_segment("p")
    if duration is None or p_points is None or len(p_points) < 6 or ctx.fs <= 0:
        return _abstain("P loop not evaluable")
    array = np.asarray(p_points, dtype=float)
    samples = max(2, len(array) // 3)
    terminal = np.mean(array[-samples:], axis=0)
    posterior_superior = float(-terminal[2] - terminal[1])
    return FeatureValue(float(duration * max(0.0, posterior_superior)))


# --- 11. further vector constructs -----------------------------------------


def _time_to_peak_spatial_velocity(ctx: FeatureContext, segment: str) -> FeatureValue:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 5:
        return _abstain(f"{segment} segment not evaluable")
    array = np.asarray(points, dtype=float)
    velocity = np.linalg.norm(np.diff(array, axis=0), axis=1)
    if len(velocity) < 3:
        return _abstain("velocity profile too short")
    return FeatureValue(float(int(np.argmax(velocity)) / (len(velocity) - 1)))


def _loop_hysteresis_area(ctx: FeatureContext, segment: str) -> FeatureValue:
    """Area between the outgoing and returning limbs of a loop.

    A perfectly retracing loop has zero hysteresis; a wide one means the
    depolarizing and repolarizing wavefronts take different spatial routes.
    """

    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 8:
        return _abstain(f"{segment} loop not evaluable")
    array = np.asarray(points, dtype=float)
    magnitude = np.linalg.norm(array, axis=1)
    peak = int(np.argmax(magnitude))
    if peak < 3 or peak > len(array) - 4:
        return _abstain("loop peak at the boundary")
    forward = _resample_curve(array[: peak + 1], 32)
    backward = _resample_curve(array[peak:][::-1], 32)
    if forward is None or backward is None:
        return _abstain("limb resampling failed")
    return FeatureValue(
        float(np.mean(np.linalg.norm(forward - backward, axis=1)))
    )


def _qrs_upstroke_velocity_max(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("qrs")
    if points is None or len(points) < 5 or ctx.fs <= 0:
        return _abstain("QRS segment not evaluable")
    array = np.asarray(points, dtype=float)
    velocity = np.linalg.norm(np.diff(array, axis=0), axis=1) * float(ctx.fs)
    return FeatureValue(float(np.max(velocity) * MILLIVOLT_TO_MICROVOLT / 1000.0))


def _qrs_downstroke_to_upstroke_ratio(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("qrs")
    if points is None or len(points) < 8:
        return _abstain("QRS segment not evaluable")
    array = np.asarray(points, dtype=float)
    magnitude = np.linalg.norm(array, axis=1)
    peak = int(np.argmax(magnitude))
    if peak < 2 or peak > len(array) - 3:
        return _abstain("QRS peak at the boundary")
    velocity = np.linalg.norm(np.diff(array, axis=0), axis=1)
    up = float(np.max(velocity[:peak])) if peak > 0 else 0.0
    down = float(np.max(velocity[peak:])) if peak < len(velocity) else 0.0
    if up < EPS:
        return _abstain("degenerate upstroke")
    return FeatureValue(down / up)


def _jt_area_ratio(ctx: FeatureContext) -> FeatureValue:
    st_area = _area_vector(ctx.vcg_segment("st"), ctx.fs)
    t_area = _area_vector(ctx.vcg_segment("t"), ctx.fs)
    if st_area is None or t_area is None:
        return _abstain("ST or T segment not evaluable")
    total = float(np.linalg.norm(st_area) + np.linalg.norm(t_area))
    if total < EPS:
        return _abstain("degenerate repolarization area")
    return FeatureValue(float(np.linalg.norm(st_area) / total))


def _u_to_t_amplitude_ratio(ctx: FeatureContext) -> FeatureValue:
    u_magnitude = _magnitude(ctx.vcg_segment("u"))
    t_magnitude = _magnitude(ctx.vcg_segment("t"))
    if u_magnitude is None or t_magnitude is None:
        return _abstain("U or T segment not evaluable")
    t_peak = float(np.max(t_magnitude))
    if t_peak < EPS:
        return _abstain("degenerate T wave")
    return FeatureValue(float(np.max(u_magnitude) / t_peak))


def _rr_asymmetry_index(ctx: FeatureContext) -> FeatureValue:
    """Porta index: share of RR decelerations among all RR changes."""

    beats = getattr(ctx.delineation, "beats", None)
    if not beats or len(beats) < 4 or ctx.fs <= 0:
        return _abstain("fewer than four beats")
    samples = [beat.r_peak.sample for beat in beats]
    rr = np.diff(np.asarray(samples, dtype=float)) / float(ctx.fs)
    if len(rr) < 3:
        return _abstain("fewer than three RR intervals")
    differences = np.diff(rr)
    nonzero = differences[np.abs(differences) > 1e-9]
    if len(nonzero) < 2:
        return _abstain("RR sequence has no variation")
    return FeatureValue(float(np.mean(nonzero < 0)))


def _beat_axis_drift_deg(ctx: FeatureContext) -> FeatureValue:
    """Systematic drift of the QRS axis across the recording.

    A random beat-to-beat scatter and a monotone drift are different things:
    the first is noise, the second is respiration, electrode movement, or a
    genuine progressive change. Existing variability markers cannot tell them
    apart.
    """

    curves = _beat_segment_curves(ctx, "qrs")
    if len(curves) < 4:
        return _abstain("fewer than four analysable QRS complexes")
    vectors = []
    for curve in curves:
        array = np.asarray(curve, dtype=float)
        if array.ndim != 2 or len(array) < 2:
            continue
        vectors.append(array[int(np.argmax(np.linalg.norm(array, axis=1)))])
    if len(vectors) < 4:
        return _abstain("fewer than four usable beat vectors")
    reference = vectors[0]
    angles = []
    for vector in vectors[1:]:
        angle = _angle_between(reference, vector)
        if angle is not None:
            angles.append(angle)
    if len(angles) < 3:
        return _abstain("degenerate beat vectors")
    index = np.arange(len(angles), dtype=float)
    return FeatureValue(float(np.polyfit(index, angles, 1)[0]))


# --- 12. second batch: further gaps ----------------------------------------


def _tp_baseline_stability(ctx: FeatureContext) -> FeatureValue:
    """Scatter of the TP interval, the only isoelectric stretch there is.

    Doubles as a physiological marker (atrial and ventricular diastolic quiet)
    and as a quality control: a noisy TP means every baseline-relative
    measurement on that recording carries that noise.
    """

    magnitude = _magnitude(_tp_window(ctx))
    if magnitude is None or len(magnitude) < 5:
        return _abstain("no evaluable TP interval")
    return FeatureValue(float(np.std(magnitude) * MILLIVOLT_TO_MICROVOLT))


def _qrs_t_area_ratio_signed(ctx: FeatureContext) -> FeatureValue:
    """Signed ratio of T area to QRS area along the QRS direction.

    Negative means the repolarization area opposes depolarization, which is
    the discordance that defines a strain pattern and secondary T changes.
    """

    qrs_area = _area_vector(ctx.vcg_segment("qrs"), ctx.fs)
    t_area = _area_vector(ctx.vcg_segment("t"), ctx.fs)
    if qrs_area is None or t_area is None:
        return _abstain("QRS or T segment not evaluable")
    norm = float(np.linalg.norm(qrs_area))
    if norm < EPS:
        return _abstain("degenerate QRS area")
    return FeatureValue(float(np.dot(t_area, qrs_area) / (norm ** 2)))


def _ventricular_gradient_elevation(ctx: FeatureContext) -> FeatureValue:
    metrics = ctx.geh_area_metrics()
    if metrics is None:
        return _abstain("origin-corrected QRST area not evaluable")
    value = vcg_elevation_deg(metrics.qrst_area_vector_uv_ms)
    return (
        _abstain("degenerate ventricular gradient")
        if value is None
        else FeatureValue(
            float(value),
            per_lead=ctx.vcg_origin_assessment.numeric_metadata(),
            notes=(
                ctx.vcg_origin_assessment.provenance_note()
                + "; elevation from inferior +Y; SVG=complete QRST area"
            ),
        )
    )


def _qrs_loop_planarity_deviation(ctx: FeatureContext) -> FeatureValue:
    """Out-of-plane fraction of the QRS loop.

    A normal QRS loop is close to planar; conduction disease and scar bend it
    out of its own best-fit plane.
    """

    qrs = ctx.vcg_segment("qrs")
    if qrs is None or len(qrs) < 6:
        return _abstain("QRS loop not evaluable")
    _, eigenvalues, _ = _pca(np.asarray(qrs, dtype=float))
    total = float(np.sum(eigenvalues))
    if total <= EPS:
        return _abstain("degenerate QRS covariance")
    return FeatureValue(float(eigenvalues[2] / total))


def _t_loop_planarity_deviation(ctx: FeatureContext) -> FeatureValue:
    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 6:
        return _abstain("T loop not evaluable")
    _, eigenvalues, _ = _pca(np.asarray(t_points, dtype=float))
    total = float(np.sum(eigenvalues))
    if total <= EPS:
        return _abstain("degenerate T covariance")
    return FeatureValue(float(eigenvalues[2] / total))


def _qrs_initial_to_peak_angle(ctx: FeatureContext) -> FeatureValue:
    """Rotation from septal activation to peak free-wall activation."""

    initial = _initial_qrs_vector(ctx)
    qrs = ctx.vcg_segment("qrs")
    if initial is None or qrs is None or len(qrs) < 4:
        return _abstain("QRS loop not evaluable")
    array = np.asarray(qrs, dtype=float)
    peak = array[int(np.argmax(np.linalg.norm(array, axis=1)))]
    angle = _angle_between(initial, peak)
    if angle is None:
        return _abstain("degenerate QRS vectors")
    return FeatureValue(float(angle))


def _r_peak_timing_fraction(ctx: FeatureContext) -> FeatureValue:
    """Where the QRS peak sits inside the complex.

    Late intrinsicoid deflection is the classical sign of delayed activation
    of the wall under the electrode; as a fraction it needs no lead.
    """

    magnitude = _magnitude(ctx.vcg_segment("qrs"))
    if magnitude is None or len(magnitude) < 4:
        return _abstain("QRS segment not evaluable")
    return FeatureValue(float(int(np.argmax(magnitude)) / (len(magnitude) - 1)))


def _st_t_vector_angle(ctx: FeatureContext) -> FeatureValue:
    st_vector = _mean_vector(ctx.vcg_segment("st"))
    t_vector = _mean_vector(ctx.vcg_segment("t"))
    if st_vector is None or t_vector is None:
        return _abstain("ST or T segment not evaluable")
    angle = _angle_between(st_vector, t_vector)
    if angle is None:
        return _abstain("degenerate ST or T vector")
    return FeatureValue(float(angle))


def _repolarization_time_dispersion(ctx: FeatureContext) -> FeatureValue:
    """Temporal spread of the T loop, normalised by its duration.

    A high value means repolarization is spread across the segment rather than
    concentrated, i.e. a broad rather than a peaked T wave.
    """

    magnitude = _magnitude(ctx.vcg_segment("t"))
    if magnitude is None or len(magnitude) < 5:
        return _abstain("T segment not evaluable")
    total = float(np.sum(magnitude))
    if total < EPS:
        return _abstain("degenerate T wave")
    index = np.arange(len(magnitude), dtype=float)
    centroid = float(np.sum(index * magnitude) / total)
    variance = float(np.sum(((index - centroid) ** 2) * magnitude) / total)
    return FeatureValue(float(np.sqrt(variance) / len(magnitude)))


def _qrs_energy_concentration(ctx: FeatureContext) -> FeatureValue:
    """Share of QRS energy inside its densest quarter.

    A narrow, concentrated QRS gives a high value; a fragmented or widened
    complex spreads its energy out.
    """

    magnitude = _magnitude(ctx.vcg_segment("qrs"))
    if magnitude is None or len(magnitude) < 8:
        return _abstain("QRS segment not evaluable")
    energy = magnitude ** 2
    total = float(np.sum(energy))
    if total < EPS:
        return _abstain("zero QRS energy")
    window = max(2, len(energy) // 4)
    best = max(
        float(np.sum(energy[start : start + window]))
        for start in range(0, len(energy) - window + 1)
    )
    return FeatureValue(best / total)


def _beat_to_beat_qt_variability(ctx: FeatureContext) -> FeatureValue:
    """Scatter of the QT interval across beats.

    Beat-to-beat QT variability is an established repolarization-lability
    marker; ten seconds gives few beats, so this stays descriptive.
    """

    beats = getattr(ctx.delineation, "beats", None)
    if not beats or len(beats) < 4 or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        return _abstain("fewer than four beats")
    # A Q peak is not a QRS onset; mixing starts creates artificial variability.
    intervals: list[float] = []
    for beat in beats:
        onset = getattr(beat, "r_onset", None)
        offset = getattr(beat, "t_offset", None)
        if onset is None or offset is None:
            continue
        if (
            not np.isfinite(onset.confidence)
            or not np.isfinite(offset.confidence)
            or onset.confidence < fiducial_confidence_threshold("r_onset")
            or offset.confidence < fiducial_confidence_threshold("t_offset")
        ):
            continue
        if not np.isfinite(onset.sample) or not np.isfinite(offset.sample):
            continue
        duration = float((offset.sample - onset.sample) * 1000.0 / ctx.fs)
        if onset.sample < 0 or not np.isfinite(duration) or duration <= 0:
            continue
        intervals.append(duration)
    if len(intervals) < 4:
        return _abstain("fewer than four evaluable QT intervals")
    return FeatureValue(float(np.std(intervals, ddof=1)))


def _qt_variability_normalised(ctx: FeatureContext) -> FeatureValue:
    """QT scatter divided by RR scatter: the QT variability index concept."""

    beats = getattr(ctx.delineation, "beats", None)
    if not beats or len(beats) < 5 or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        return _abstain("QT or RR variability not evaluable")
    qt, rr = [], []
    for previous, current in zip(beats[:-1], beats[1:]):
        points = [(getattr(previous, "r_peak", None), "r_peak"),
                  (getattr(current, "r_peak", None), "r_peak"),
                  (getattr(current, "r_onset", None), "r_onset"),
                  (getattr(current, "t_offset", None), "t_offset")]
        if any(point is None or not np.isfinite(point.sample)
               or point.sample < 0 or not np.isfinite(point.confidence)
               or point.confidence < fiducial_confidence_threshold(name)
               for point, name in points):
            continue
        prior_r, current_r, onset, offset = [point.sample for point, _ in points]
        if not (prior_r < current_r and onset <= current_r < offset):
            continue
        qt.append((offset - onset) * 1000.0 / ctx.fs)
        rr.append((current_r - prior_r) * 1000.0 / ctx.fs)
    if len(qt) < 4:
        return _abstain("fewer than four paired QT and preceding RR intervals")
    rr_sd = float(np.std(rr, ddof=1))
    if not np.isfinite(rr_sd) or rr_sd < EPS:
        return _abstain("RR interval has no variation")
    return FeatureValue(float(np.std(qt, ddof=1)) / rr_sd)


def _p_wave_energy_fraction_terminal(ctx: FeatureContext) -> FeatureValue:
    """Share of P-wave energy in its terminal third: the left atrial part."""

    magnitude = _magnitude(ctx.vcg_segment("p"))
    if magnitude is None or len(magnitude) < 6:
        return _abstain("P segment not evaluable")
    energy = magnitude ** 2
    total = float(np.sum(energy))
    if total < EPS:
        return _abstain("zero P energy")
    tail = energy[-max(2, len(energy) // 3):]
    return FeatureValue(float(np.sum(tail) / total))


def _p_loop_direction_reversal_count(ctx: FeatureContext) -> FeatureValue:
    """Direction reversals inside the P loop: a biphasic or notched P wave."""

    p_points = ctx.vcg_segment("p")
    if p_points is None or len(p_points) < 6:
        return _abstain("P loop not evaluable")
    array = np.asarray(p_points, dtype=float)
    steps = np.diff(array, axis=0)
    if len(steps) < 3:
        return _abstain("P loop too short")
    reversals = 0
    for first, second in zip(steps[:-1], steps[1:]):
        angle = _angle_between(first, second)
        if angle is not None and angle > 90.0:
            reversals += 1
    return FeatureValue(float(reversals))


def _atrial_conduction_velocity_surrogate(ctx: FeatureContext) -> FeatureValue:
    """P loop path length divided by P duration."""

    p_points = ctx.vcg_segment("p")
    duration = _segment_duration_ms(ctx, "p")
    if p_points is None or duration is None or duration <= 0:
        return _abstain("P segment not evaluable")
    path = _path_length(np.asarray(p_points, dtype=float))
    if path < EPS:
        return _abstain("degenerate P loop")
    return FeatureValue(float(path * MILLIVOLT_TO_MICROVOLT / duration))


def _pr_interval_to_p_duration_ratio(ctx: FeatureContext) -> FeatureValue:
    """How much of the PR interval is atrial activation versus AV delay."""

    p_duration = _segment_duration_ms(ctx, "p")
    pr_duration = _segment_duration_ms(ctx, "pr")
    if p_duration is None or pr_duration is None:
        return _abstain("P or PR segment not evaluable")
    total = p_duration + pr_duration
    if total <= 0:
        return _abstain("degenerate PR interval")
    return FeatureValue(float(p_duration / total))


def _s1q3t3_vector_score(ctx: FeatureContext) -> FeatureValue:
    """Dimensionless continuous strength of the actual S1-Q3-T3 pattern."""

    if ctx.ecg_12lead is None:
        return _abstain("12-lead ECG unavailable")
    qrs_i = _representative_lead(ctx, "I", "qrs")
    qrs_iii = _representative_lead(ctx, "III", "qrs")
    t_iii = _representative_lead(ctx, "III", "t")
    if qrs_i is None or qrs_iii is None or t_iii is None:
        return _abstain("lead I/III QRS or lead III T segment unavailable")
    lead_i = np.asarray(qrs_i, dtype=float) - _lead_baseline_mv(ctx, "I")
    lead_iii = np.asarray(qrs_iii, dtype=float) - _lead_baseline_mv(ctx, "III")
    t_wave_iii = np.asarray(t_iii, dtype=float) - _lead_baseline_mv(ctx, "III")
    if min(len(lead_i), len(lead_iii), len(t_wave_iii)) < 3:
        return _abstain("lead I/III segments too short")

    s1 = max(0.0, float(-np.min(lead_i)))
    r3_index = int(np.argmax(lead_iii))
    q3 = max(0.0, float(-np.min(lead_iii[: max(r3_index, 1)])))
    t3 = max(0.0, float(-np.min(t_wave_iii)))

    scale_leads = tuple(LEAD_MAP)
    qrs_scale = max(
        (
            float(np.max(np.abs(np.asarray(values, dtype=float) - _lead_baseline_mv(ctx, lead))))
            for lead in scale_leads
            if (values := _representative_lead(ctx, lead, "qrs")) is not None
            and len(values)
        ),
        default=0.0,
    )
    t_scale = max(
        (
            float(np.max(np.abs(np.asarray(values, dtype=float) - _lead_baseline_mv(ctx, lead))))
            for lead in scale_leads
            if (values := _representative_lead(ctx, lead, "t")) is not None
            and len(values)
        ),
        default=0.0,
    )
    if qrs_scale <= EPS or t_scale <= EPS:
        return _abstain("global QRS or T amplitude scale is degenerate")
    components = np.asarray([s1 / qrs_scale, q3 / qrs_scale, t3 / t_scale])
    score = float(np.cbrt(np.prod(np.clip(components, 0.0, None))))
    return FeatureValue(
        score,
        per_lead={
            "s1_uv": s1 * MILLIVOLT_TO_MICROVOLT,
            "q3_uv": q3 * MILLIVOLT_TO_MICROVOLT,
            "negative_t3_uv": t3 * MILLIVOLT_TO_MICROVOLT,
        },
    )


def _right_precordial_t_inversion_extent(ctx: FeatureContext) -> FeatureValue:
    """Anterior component of the T vector, negated.

    Right ventricular strain and ARVC both invert the anterior T wave; as a
    vector component this needs no per-lead threshold.
    """

    t_vector = _mean_vector(ctx.vcg_segment("t"))
    if t_vector is None:
        return _abstain("T segment not evaluable")
    return FeatureValue(float(-t_vector[2] * MILLIVOLT_TO_MICROVOLT))


def _qrs_axis_to_t_axis_discordance_signed(ctx: FeatureContext) -> FeatureValue:
    qrs_vector = _mean_vector(ctx.vcg_segment("qrs"))
    t_vector = _mean_vector(ctx.vcg_segment("t"))
    if qrs_vector is None or t_vector is None:
        return _abstain("QRS or T segment not evaluable")
    norms = np.linalg.norm(qrs_vector) * np.linalg.norm(t_vector)
    if norms < EPS:
        return _abstain("degenerate vectors")
    return FeatureValue(float(np.dot(qrs_vector, t_vector) / norms))


def _strain_pattern_score(ctx: FeatureContext) -> FeatureValue:
    """High QRS voltage with an opposing, asymmetric T wave.

    The strain pattern of pressure-overload hypertrophy, expressed as the
    product of the three elements that define it.
    """

    voltage = _spatial_qrs_voltage_sum(ctx)
    discordance = _qrs_axis_to_t_axis_discordance_signed(ctx)
    symmetry = _t_wave_symmetry(ctx)
    if voltage.value is None or discordance.value is None or symmetry.value is None:
        return _abstain("voltage, discordance or symmetry not evaluable")
    opposition = max(0.0, -float(discordance.value))
    asymmetry = abs(float(symmetry.value) - 1.0)
    return FeatureValue(
        float(voltage.value) * opposition * (1.0 + asymmetry) / 1000.0
    )


def _low_voltage_with_preserved_duration(ctx: FeatureContext) -> FeatureValue:
    """Low QRS amplitude despite a normal or long QRS duration.

    The infiltrative signature: amyloid and effusion reduce voltage without
    shortening activation.
    """

    magnitude = _magnitude(ctx.vcg_segment("qrs"))
    duration = _segment_duration_ms(ctx, "qrs")
    if magnitude is None or duration is None or duration <= 0:
        return _abstain("QRS segment not evaluable")
    peak = float(np.max(magnitude)) * MILLIVOLT_TO_MICROVOLT
    if peak < EPS:
        return _abstain("degenerate QRS voltage")
    return FeatureValue(float(duration / peak))


def _qrs_area_to_perimeter_ratio(ctx: FeatureContext) -> FeatureValue:
    """Compactness of the QRS loop."""

    qrs = ctx.vcg_segment("qrs")
    if qrs is None or len(qrs) < 6:
        return _abstain("QRS loop not evaluable")
    array = np.asarray(qrs, dtype=float)
    perimeter = _path_length(array)
    if perimeter < EPS:
        return _abstain("degenerate QRS loop")
    _, eigenvalues, _ = _pca(array)
    area = float(np.pi * np.sqrt(max(eigenvalues[0], 0.0) * max(eigenvalues[1], 0.0)))
    return FeatureValue(float(area / perimeter))


def _t_loop_area_to_perimeter_ratio(ctx: FeatureContext) -> FeatureValue:
    t_points = ctx.vcg_segment("t")
    if t_points is None or len(t_points) < 6:
        return _abstain("T loop not evaluable")
    array = np.asarray(t_points, dtype=float)
    perimeter = _path_length(array)
    if perimeter < EPS:
        return _abstain("degenerate T loop")
    _, eigenvalues, _ = _pca(array)
    area = float(np.pi * np.sqrt(max(eigenvalues[0], 0.0) * max(eigenvalues[1], 0.0)))
    return FeatureValue(float(area / perimeter))


def _qrs_t_loop_scale_ratio(ctx: FeatureContext) -> FeatureValue:
    """Size of the T loop relative to the QRS loop."""

    qrs = _magnitude(ctx.vcg_segment("qrs"))
    t_points = _magnitude(ctx.vcg_segment("t"))
    if qrs is None or t_points is None:
        return _abstain("QRS or T segment not evaluable")
    qrs_peak = float(np.max(qrs))
    if qrs_peak < EPS:
        return _abstain("degenerate QRS")
    return FeatureValue(float(np.max(t_points) / qrs_peak))


def _spatial_velocity_entropy(ctx: FeatureContext, segment: str) -> FeatureValue:
    """Entropy of the spatial velocity profile.

    A smooth single-peaked wavefront has low entropy; a fragmented one with
    several bursts of movement has high entropy.
    """

    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 6:
        return _abstain(f"{segment} segment not evaluable")
    velocity = np.linalg.norm(
        np.diff(np.asarray(points, dtype=float), axis=0),
        axis=1,
    )
    total = float(np.sum(velocity))
    if total < EPS:
        return _abstain("degenerate velocity profile")
    probability = velocity / total
    probability = probability[probability > EPS]
    if len(probability) < 2:
        return _abstain("degenerate velocity distribution")
    entropy = float(-np.sum(probability * np.log(probability)))
    return FeatureValue(entropy / float(np.log(len(probability))))


def _pacing_spike_energy_ratio(ctx: FeatureContext) -> FeatureValue:
    """Very short high-amplitude transient inside the QRS.

    A pacing spike is a sub-millisecond artefact with disproportionate
    wideband slope; expressing it as a ratio keeps it continuous rather than a
    binary sighting.

    **It does not detect pacing in this cohort, and the name invites the
    opposite conclusion.** Measured on 597 recordings drawn balanced on the
    machine report -- 300 paced, 297 not -- it separates them at **AUC 0,523**,
    which is no separation at all. It is available on 94,3 % of paced
    recordings, so this is not an abstention problem: the marker returns a
    number and the number carries nothing.

    The reason is the acquisition, not the implementation. A pacing spike is
    roughly 0,5 ms wide, and this cohort is sampled at 500 Hz behind a 150 Hz
    analogue corner -- the spike is filtered away before any marker sees it.
    ``wideband`` aliases the diagnostic band deliberately for the same reason:
    at that corner there is no further bandwidth to read.

    What does separate paced from unpaced here is the consequence of pacing
    rather than the spike: QRS duration reaches AUC 0,884 and the spatial
    QRS-T angle 0,814 on the same recordings. For identifying paced
    recordings, use ``report_pacemaker_state`` from ``parse-reports`` or the
    ``pacemaker_carrier`` endpoint; both come from records rather than from a
    transient the sampling cannot hold.

    The marker is kept because the ratio is a real property of the QRS
    derivative and may carry something else, and because removing a
    registered marker mid-study is a bigger intervention than documenting it.
    """

    wideband = ctx.vcg_segment_wideband("qrs")
    if wideband is None:
        return _abstain("no wideband source")
    magnitude = _magnitude(wideband)
    if magnitude is None or len(magnitude) < 8 or ctx.fs <= 0:
        return _abstain("QRS segment too short")
    derivative = np.abs(np.diff(magnitude))
    if len(derivative) < 4:
        return _abstain("derivative too short")
    median_step = float(np.median(derivative))
    if median_step < EPS:
        return _abstain("degenerate derivative")
    return FeatureValue(float(np.max(derivative) / median_step))


def _qrs_high_frequency_reduced_amplitude_zone(ctx: FeatureContext) -> FeatureValue:
    """Depth of the deepest trough in the high-frequency QRS envelope.

    The reduced-amplitude zone of high-frequency QRS analysis, associated with
    ischaemia in the exercise-testing literature.
    """

    wideband = ctx.vcg_segment_wideband("qrs")
    if wideband is None:
        return _abstain("no wideband source")
    if len(wideband) < 16:
        return _abstain("QRS too short for an envelope")
    nyquist = ctx.fs / 2.0
    low = 80.0
    high = 150.0
    if nyquist <= high:
        return _abstain("sampling rate does not support the fixed 80-150 Hz band")
    sos = sp_signal.butter(
        3,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    padlen = 3 * (2 * len(sos) + 1)
    if len(wideband) <= padlen:
        return _abstain("QRS shorter than the filter requirement")
    filtered_xyz = sp_signal.sosfiltfilt(sos, wideband, axis=0)
    analytic_xyz = sp_signal.hilbert(filtered_xyz, axis=0)
    envelope = np.sqrt(np.sum(np.abs(analytic_xyz) ** 2, axis=1))
    if len(envelope) < 6:
        return _abstain("envelope too short")
    peak = float(np.max(envelope))
    if peak < EPS:
        return _abstain("no high-frequency content")
    interior = envelope[1:-1]
    return FeatureValue(float((peak - float(np.min(interior))) / peak))


# --- specification table ----------------------------------------------------


def _segment_marker(segment: str, function) -> Callable[[FeatureContext], FeatureValue]:
    return lambda ctx: function(ctx, segment)


ELECTROPHYSIOLOGY_SPECS: tuple[_MarkerSpec, ...] = (
    _MarkerSpec(
        "terminal_qrs_rms40_uv",
        "Terminale QRS-Energie",
        "RMS of the last 40 ms of the wideband QRS vector magnitude.",
        "sqrt(mean((|v(t)| - baseline)^2)) over the terminal 40 ms.",
        "uV",
        "Terminal conduction delay and scar substrate. Not a signal-averaged "
        "ECG: ten seconds cannot reach sub-microvolt noise.",
        _terminal_qrs_rms40,
    ),
    _MarkerSpec(
        "low_amplitude_terminal_duration_ms",
        "Terminale QRS-Energie",
        "Duration of the terminal QRS below a 40 uV envelope.",
        "Count of trailing samples with |v| - baseline < 40 uV.",
        "ms",
        "Vector analogue of LAS40; longer suggests fractionated terminal "
        "activation.",
        _low_amplitude_terminal_duration,
    ),
    _MarkerSpec(
        "filtered_qrs_duration_ms",
        "Terminale QRS-Energie",
        "QRS duration measured on the wideband envelope.",
        "Span of samples with |v| - baseline >= 40 uV.",
        "ms",
        "Wideband QRS width; sensitive to late components a 100 Hz path "
        "attenuates.",
        _filtered_qrs_duration,
    ),
    _MarkerSpec(
        "terminal_qrs_hf_energy_fraction",
        "Terminale QRS-Energie",
        "Share of axiswise 80-150 Hz QRS energy in the terminal 40 ms.",
        "sum(X/Y/Z 80-150 Hz energy over terminal 40 ms) / total QRS energy.",
        "dimensionless",
        "Localises fragmentation to the terminal portion rather than the "
        "whole complex.",
        _terminal_qrs_hf_ratio,
    ),
    _MarkerSpec(
        "ta_vector_magnitude_uv",
        "Atriale Repolarisation",
        "Magnitude of the atrial repolarization vector from the PR segment.",
        "|mean(PR loop) - median(TP loop)|.",
        "uV",
        "Atrial repolarization; raised in pericarditis and atrial infarction.",
        _ta_vector_magnitude,
    ),
    _MarkerSpec(
        "ta_to_p_angle_deg",
        "Atriale Repolarisation",
        "Angle between the atrial repolarization and P vectors.",
        "angle(Ta vector, mean P vector).",
        "degrees",
        "Normally close to opposite; a small angle indicates abnormal atrial "
        "repolarization sequence.",
        _ta_to_p_angle,
    ),
    _MarkerSpec(
        "atrial_gradient_magnitude_uv_ms",
        "Atriale Repolarisation",
        "Atrial analogue of the ventricular gradient.",
        "|P area vector + Ta area vector|.",
        "uV*ms",
        "Activation-independent atrial repolarization heterogeneity; a "
        "plausible atrial-fibrillation substrate marker.",
        _atrial_gradient_magnitude,
    ),
    _MarkerSpec(
        "atrial_gradient_to_p_angle_deg",
        "Atriale Repolarisation",
        "Angle between the atrial gradient and the P area vector.",
        "angle(P area + Ta area, P area).",
        "degrees",
        "Direction of atrial repolarization heterogeneity relative to "
        "activation.",
        _atrial_gradient_to_p_angle,
    ),
    _MarkerSpec(
        "pr_segment_signed_deviation_uv",
        "Atriale Repolarisation",
        "PR segment deviation projected on the P direction.",
        "dot(Ta vector, unit P vector).",
        "uV",
        "Negative means PR depression, the pericarditis pattern.",
        _pr_segment_deviation,
    ),
    _MarkerSpec(
        "tp_segment_slope_uv_per_ms",
        "Atriale Repolarisation",
        "Slope of the TP segment magnitude (Spodick sign).",
        "Least-squares slope of |v| across the TP segment.",
        "uV/ms",
        "Downsloping TP is described in acute pericarditis.",
        _tp_segment_slope,
    ),
    _MarkerSpec(
        "total_cosine_r_to_t",
        "Repolarisationsheterogenitaet",
        "TCRT: mean cosine between QRS loop vectors and the mean T vector.",
        "mean(cos(angle(QRS(t), mean T vector))).",
        "dimensionless",
        "Established repolarization-heterogeneity descriptor; low or negative "
        "values are associated with adverse outcome.",
        _total_cosine_r_to_t,
    ),
    _MarkerSpec(
        "t_morphology_dispersion_deg",
        "Repolarisationsheterogenitaet",
        "TMD: mean angle between T loop samples and the loop principal axis.",
        "mean(min(angle(T(t), PC1), 180 - angle)).",
        "degrees",
        "Wider dispersion indicates non-uniform repolarization.",
        _t_morphology_dispersion,
    ),
    _MarkerSpec(
        "t_pca_ratio_lambda2_lambda1",
        "Repolarisationsheterogenitaet",
        "Ratio of the second to first principal eigenvalue of the T loop.",
        "lambda2 / lambda1 of the T loop covariance.",
        "dimensionless",
        "T-wave complexity; elevated in repolarization abnormality.",
        _t_pca_ratio,
    ),
    _MarkerSpec(
        "t_nondipolar_residual_fraction",
        "Repolarisationsheterogenitaet",
        "Retired invalid 3D surrogate formerly mislabeled as nondipolar T-wave residuum.",
        "not computed; true TWR requires SVD dimensions 4-8 of I, II, V1-V6.",
        "dimensionless",
        "Always abstains; use lead8_t_wave_residuum_fraction.",
        _t_nondipolar_residual,
        reference_keys=(),
    ),
    _MarkerSpec(
        "repolarization_direction_change_rate_deg_per_ms",
        "Repolarisationsheterogenitaet",
        "Mean angular velocity of the T vector.",
        "mean(angle(T(t), T(t+1))) / sample interval.",
        "degrees/ms",
        "Amplitude-independent measure of repolarization order; expected to "
        "be more repeatable than magnitude-based descriptors.",
        _repolarization_direction_change_rate,
    ),
    _MarkerSpec(
        "t_wave_symmetry_index",
        "Repolarisationsheterogenitaet",
        "Ratio of ascending to descending T limb duration.",
        "index(T peak) / (n - 1 - index(T peak)).",
        "dimensionless",
        "A symmetric T wave near 1.0 is a classical ischaemia pattern.",
        _t_wave_symmetry,
    ),
    _MarkerSpec(
        "initial_qrs_to_terminal_t_angle_deg",
        "Aktivierung-Erholung",
        "Angle between the initial 20 ms QRS vector and the terminal T vector.",
        "angle(mean QRS first 20 ms, mean T last 30 ms).",
        "degrees",
        "Asks whether the region depolarizing first also repolarizes last, "
        "which is the basis of a concordant T wave.",
        _initial_qrs_to_terminal_t_angle,
    ),
    _MarkerSpec(
        "activation_recovery_coupling_index",
        "Aktivierung-Erholung",
        "Cosine of the initial-QRS to terminal-T angle.",
        "dot(initial QRS, terminal T) / (|initial QRS| |terminal T|).",
        "dimensionless",
        "Signed version of the coupling angle, positive for the normal "
        "sequence.",
        _activation_recovery_coupling_index,
    ),
    _MarkerSpec(
        "septal_activation_azimuth_deg",
        "Aktivierung-Erholung",
        "Azimuth of the initial 20 ms QRS vector.",
        "atan2(z, x) of the mean initial QRS vector.",
        "degrees",
        "Septal activation direction; altered in septal infarction and in "
        "left-sided conduction disease.",
        _septal_activation_azimuth,
    ),
    _MarkerSpec(
        "brugada_beta_angle_deg",
        "Brugada/J-Wave",
        "Retired invalid VCG-magnitude surrogate for the right-precordial beta angle.",
        "not computed; requires S upslope and r-prime downslope in V1/V2.",
        "degrees",
        "Always abstains because VCG magnitude is not Chevallier's beta angle.",
        _brugada_beta_angle,
        category="ecg",
    ),
    _MarkerSpec(
        "brugada_triangle_base_ms",
        "Brugada/J-Wave",
        "Retired invalid VCG-magnitude surrogate for the Brugada triangle base.",
        "not computed; requires an r-prime triangle in V1/V2.",
        "ms",
        "Always abstains; no clinical Brugada claim is made.",
        _brugada_triangle_base,
        category="ecg",
    ),
    _MarkerSpec(
        "brugada_ascending_limb_duration_ms",
        "Brugada/J-Wave",
        "Retired invalid VCG-magnitude surrogate for an r-prime limb duration.",
        "not computed; requires right-precordial r-prime fiducials.",
        "ms",
        "Always abstains; no clinical Brugada claim is made.",
        _brugada_ascending_limb_duration,
        category="ecg",
    ),
    _MarkerSpec(
        "j_point_elevation_uv",
        "Brugada/J-Wave",
        "Vector magnitude elevation at the J point above baseline.",
        "|v(J)| - PR baseline.",
        "uV",
        "J point elevation, the entry criterion of early repolarization.",
        _j_point_elevation,
    ),
    _MarkerSpec(
        "st_slope_after_j_uv_per_ms",
        "Brugada/J-Wave",
        "Slope of the ST magnitude in the 60 ms after the J point.",
        "Least-squares slope of |v| over the early ST segment.",
        "uV/ms",
        "The discriminating element of early repolarization: a horizontal or "
        "descending ST carries the arrhythmic association, an ascending one "
        "does not.",
        _st_slope_after_j,
    ),
    _MarkerSpec(
        "early_repolarization_malignant_score",
        "Brugada/J-Wave",
        "J point elevation weighted by a non-ascending ST slope.",
        "J elevation * (max(0, -ST slope) + 1) / 1000.",
        "dimensionless",
        "Combines the two elements the literature separates; high only when "
        "both are present.",
        _early_repolarization_pattern_score,
    ),
    _MarkerSpec(
        "osborn_j_wave_amplitude_uv",
        "Brugada/J-Wave",
        "Peak magnitude in the QRS-ST junction window above baseline.",
        "max(|v|) over the junction window minus baseline.",
        "uV",
        "J wave amplitude as seen in hypothermia and hypercalcaemia.",
        _osborn_j_wave_amplitude,
    ),
    _MarkerSpec(
        "terminal_activation_duration_ms",
        "ARVC/Terminale Aktivierung",
        "Maximum S-nadir-to-QRS-offset duration across V1-V3.",
        "max_lead (QRS offset - qualified S nadir) / fs, leads V1-V3.",
        "ms",
        "Lead-space ARVC terminal activation duration; not a VCG-magnitude surrogate.",
        _terminal_activation_duration,
        category="ecg",
    ),
    _MarkerSpec(
        "epsilon_potential_energy_ratio",
        "ARVC/Terminale Aktivierung",
        "Energy in the 40 ms after the QRS relative to QRS energy.",
        "sum(early ST^2) / sum(QRS^2) on the wideband path.",
        "dimensionless",
        "Continuous surrogate for a post-excitation potential.",
        _epsilon_potential_energy_ratio,
    ),
    _MarkerSpec(
        "concordant_st_vector_ratio",
        "Infarkt bei Schenkelblock",
        "Projection of the ST vector onto the QRS direction.",
        "dot(ST vector, QRS vector) / |QRS vector|^2.",
        "dimensionless",
        "Vectorial Sgarbossa concordance, the highest-specificity element for "
        "occlusion under LBBB or pacing.",
        _concordant_st_vector_ratio,
    ),
    _MarkerSpec(
        "smith_modified_st_to_s_ratio",
        "Infarkt bei Schenkelblock",
        "Largest lead-specific discordant ST-elevation/S-depth ratio.",
        "max_lead ST_elevation / abs(S_nadir) for predominantly negative QRS leads.",
        "dimensionless",
        "Continuous lead-space component of modified Sgarbossa; not a VCG magnitude ratio.",
        _smith_modified_st_to_s_ratio,
        category="ecg",
    ),
    _MarkerSpec(
        "early_qrs_notch_count",
        "Infarkt bei Schenkelblock",
        "Number of slope reversals in the first half of the QRS.",
        "Sign changes of d|v|/dt over the early QRS.",
        "count",
        "Vector analogue of the Cabrera and Chapman notching signs.",
        _qrs_notch_burden_early,
    ),
    _MarkerSpec(
        "st_deviation_vector_magnitude_uv",
        "Ischaemie-Vektorik",
        "Magnitude of the mean ST vector above baseline.",
        "|mean ST vector| - baseline.",
        "uV",
        "Overall injury current magnitude, independent of which leads show it.",
        _st_deviation_vector_magnitude,
    ),
    _MarkerSpec(
        "st_injury_vector_azimuth_deg",
        "Ischaemie-Vektorik",
        "Azimuth of the ST injury vector in the transverse plane.",
        "atan2(z, x) of the mean ST vector.",
        "degrees",
        "The injury vector points towards the ischaemic region; the angle "
        "localises it without a lead-by-lead rule.",
        _st_injury_vector_azimuth,
    ),
    _MarkerSpec(
        "st_injury_vector_elevation_deg",
        "Ischaemie-Vektorik",
        "Polar elevation of the ST injury vector from the inferior +Y axis.",
        "acos(y / |origin-corrected ST vector|), 0=inferior, 180=superior.",
        "degrees",
        "Superior-inferior component of the injury vector.",
        _st_injury_vector_elevation,
    ),
    _MarkerSpec(
        "hyperacute_t_to_qrs_area_ratio",
        "Ischaemie-Vektorik",
        "T area vector magnitude relative to QRS area vector magnitude.",
        "|T area| / |QRS area|.",
        "dimensionless",
        "Hyperacute T waves and the de Winter pattern raise this ratio.",
        _hyperacute_t_to_qrs_area_ratio,
    ),
    _MarkerSpec(
        "terminal_t_negativity_ratio",
        "Ischaemie-Vektorik",
        "Terminal T projection against the T peak direction.",
        "-mean(T tail . T peak) / |T peak|^2.",
        "dimensionless",
        "Biphasic or terminally inverted T waves, the Wellens signature.",
        _terminal_t_negativity_ratio,
    ),
    _MarkerSpec(
        "aslanger_pattern_score",
        "Ischaemie-Vektorik",
        "Inferior ST vector combined with lateral opposition and anterior T.",
        "inferior ST * (max(0, -lateral) + 1) * (1 + max(0, anterior T)).",
        "dimensionless",
        "Quantifies Aslanger's pattern: inferior occlusion on multivessel "
        "disease.",
        _aslanger_pattern_score,
    ),
    _MarkerSpec(
        "spatial_qrs_peak_voltage_uv",
        "Hypertrophie-Vektorik",
        "Peak spatial QRS magnitude.",
        "max(|QRS(t)|).",
        "uV",
        "Lead-independent voltage, avoiding the placement sensitivity of "
        "Sokolow-Lyon and Cornell.",
        _spatial_qrs_voltage_sum,
    ),
    _MarkerSpec(
        "qrs_voltage_duration_product_uv_ms",
        "Hypertrophie-Vektorik",
        "Peak spatial QRS voltage multiplied by QRS duration.",
        "max(|QRS|) * QRS duration.",
        "uV*ms",
        "Vector analogue of the Cornell product.",
        _qrs_voltage_duration_product,
    ),
    _MarkerSpec(
        "katz_wachtel_equiphasic_index",
        "Hypertrophie-Vektorik",
        "Maximum total R-plus-S height of a biphasic QRS in V2-V4.",
        "max_lead (R_height + abs(S_depth)), with equiphasic balance in metadata.",
        "uV",
        "Lead-space Katz-Wachtel voltage; pediatric, weakly validated, and not diagnostic alone.",
        _katz_wachtel_equiphasic_index,
        category="ecg",
    ),
    _MarkerSpec(
        "terminal_40ms_rightward_uv",
        "Medikamenten-Signaturen",
        "Rightward component of the terminal 40 ms QRS vector.",
        "-x component of the mean terminal 40 ms vector.",
        "uV",
        "Vector form of the terminal R wave in aVR: sodium-channel blockade.",
        _terminal_40ms_rightward_axis,
    ),
    _MarkerSpec(
        "terminal_40ms_superior_uv",
        "Medikamenten-Signaturen",
        "Superior component of the terminal 40 ms QRS vector.",
        "-y component of the mean terminal 40 ms vector.",
        "uV",
        "Superior terminal deviation accompanying sodium-channel blockade.",
        _terminal_40ms_superior_deviation,
    ),
    _MarkerSpec(
        "digitalis_st_sagging_index",
        "Medikamenten-Signaturen",
        "Quadratic curvature of the ST segment magnitude.",
        "Leading coefficient of a quadratic fit to the ST magnitude.",
        "uV",
        "Concave sagging ST, the digitalis effect.",
        _digitalis_st_sagging_index,
    ),
    _MarkerSpec(
        "st_segment_duration_ms",
        "Elektrolyt-Signaturen",
        "Duration of the ST segment.",
        "(T onset - J point) / fs.",
        "ms",
        "A long ST with a normal T is the hypocalcaemia pattern; a short one "
        "accompanies hypercalcaemia.",
        _st_segment_duration,
    ),
    _MarkerSpec(
        "t_base_narrowing_index_uv_per_ms",
        "Elektrolyt-Signaturen",
        "T peak magnitude divided by T duration.",
        "max(|T|) / T duration.",
        "uV/ms",
        "Tall narrow T waves in hyperkalaemia raise this ratio.",
        _t_base_narrowing_index,
    ),
    _MarkerSpec(
        "p_to_qrs_amplitude_ratio",
        "Elektrolyt-Signaturen",
        "P peak magnitude relative to QRS peak magnitude.",
        "max(|P|) / max(|QRS|).",
        "dimensionless",
        "Falls towards zero in atrial standstill from severe hyperkalaemia.",
        _p_amplitude_loss_ratio,
    ),
    _MarkerSpec(
        "p_loop_terminal_descent_rate_uv_per_ms",
        "Atriale Geometrie",
        "Fall rate of the P vector magnitude in its final third.",
        "Least-squares slope of |P| over the terminal third.",
        "uV/ms",
        "Terminal atrial activation is the left-atrial component; a slow "
        "descent is the vector correlate of terminal force.",
        _p_loop_terminal_descent_rate,
    ),
    _MarkerSpec(
        "p_loop_roundness_index",
        "Atriale Geometrie",
        "Isoperimetric roundness of the P loop.",
        "4*pi*ellipse area / perimeter^2.",
        "dimensionless",
        "A flattened or figure-of-eight P loop indicates interatrial "
        "conduction delay.",
        _p_loop_roundness,
    ),
    _MarkerSpec(
        "interatrial_block_vector_score",
        "Atriale Geometrie",
        "P duration weighted by a posterior-superior terminal P vector.",
        "P duration * max(0, -z - y) of the terminal third.",
        "ms*mV",
        "Advanced interatrial block is a long P wave whose terminal component "
        "points back and up.",
        _interatrial_block_vector_score,
    ),
    _MarkerSpec(
        "qrs_upstroke_peak_velocity",
        "Vektor-Kinetik",
        "Maximum spatial velocity within the QRS loop.",
        "max(|dv/dt|) over the QRS.",
        "mV/s",
        "Peak conduction velocity surrogate.",
        _qrs_upstroke_velocity_max,
    ),
    _MarkerSpec(
        "qrs_downstroke_to_upstroke_velocity_ratio",
        "Vektor-Kinetik",
        "Ratio of maximum descending to ascending QRS velocity.",
        "max velocity after the peak / max velocity before it.",
        "dimensionless",
        "Asymmetric conduction, raised where terminal activation is delayed.",
        _qrs_downstroke_to_upstroke_ratio,
    ),
    _MarkerSpec(
        "jt_area_fraction",
        "Vektor-Kinetik",
        "ST area share of the total repolarization area.",
        "|ST area| / (|ST area| + |T area|).",
        "dimensionless",
        "Distributes repolarization between its early and late phases.",
        _jt_area_ratio,
    ),
    _MarkerSpec(
        "u_to_t_amplitude_ratio",
        "Vektor-Kinetik",
        "U wave peak magnitude relative to the T peak.",
        "max(|U|) / max(|T|).",
        "dimensionless",
        "Prominent U waves accompany hypokalaemia and some channelopathies.",
        _u_to_t_amplitude_ratio,
    ),
    _MarkerSpec(
        "rr_deceleration_asymmetry_index",
        "Vektor-Kinetik",
        "Share of RR decelerations among all RR changes (Porta index).",
        "mean(diff(RR) < 0).",
        "dimensionless",
        "Heart-rate asymmetry; departures from 0.5 indicate autonomic "
        "imbalance. Limited by the ten-second window.",
        _rr_asymmetry_index,
    ),
    _MarkerSpec(
        "beat_axis_drift_deg_per_beat",
        "Vektor-Kinetik",
        "Systematic drift of the QRS axis across beats.",
        "Slope of the beat-to-reference angle against beat index.",
        "degrees/beat",
        "Separates monotone drift (respiration, electrode movement, genuine "
        "change) from random beat-to-beat scatter.",
        _beat_axis_drift_deg,
    ),
)

# Per-segment families generated rather than written out.
for _segment in ("p", "qrs", "t"):
    ELECTROPHYSIOLOGY_SPECS = ELECTROPHYSIOLOGY_SPECS + (
        _MarkerSpec(
            f"{_segment}_time_to_peak_velocity_fraction",
            "Vektor-Kinetik",
            f"Relative time of peak spatial velocity within the {_segment.upper()} loop.",
            "argmax(|dv/dt|) / (n - 1).",
            "dimensionless",
            "Where in the segment the wavefront moves fastest; a shape "
            "descriptor that is independent of amplitude.",
            _segment_marker(_segment, _time_to_peak_spatial_velocity),
        ),
        _MarkerSpec(
            f"{_segment}_loop_hysteresis_distance",
            "Vektor-Kinetik",
            f"Mean separation between the outgoing and returning {_segment.upper()} limbs.",
            "mean(|forward(s) - backward(s)|) after resampling both limbs.",
            "mV",
            "A retracing loop has near-zero hysteresis; a wide one means the "
            "wavefront takes different spatial routes.",
            _segment_marker(_segment, _loop_hysteresis_area),
        ),
    )


_SECOND_BATCH: tuple[_MarkerSpec, ...] = (
    _MarkerSpec(
        "tp_baseline_stability_uv",
        "Signalgrundlinie",
        "Scatter of the TP interval magnitude.",
        "sd(|v|) over the longest evaluable TP interval.",
        "uV",
        "The only isoelectric stretch of the cycle: physiological quiet, and "
        "simultaneously the noise floor every baseline-relative measurement "
        "on that recording inherits.",
        _tp_baseline_stability,
    ),
    _MarkerSpec(
        "qrs_t_area_ratio_signed",
        "QRS-T-Kopplung",
        "Signed projection of the T area vector onto the QRS area vector.",
        "dot(T area, QRS area) / |QRS area|^2.",
        "dimensionless",
        "Negative means repolarization opposes depolarization, the "
        "discordance that defines strain and secondary T change.",
        _qrs_t_area_ratio_signed,
    ),
    _MarkerSpec(
        "ventricular_gradient_elevation_deg",
        "QRS-T-Kopplung",
        "Polar elevation of the Wilson ventricular gradient from inferior +Y.",
        "acos(SVG_y/|SVG|), SVG=integral(QRS onset:T offset) of origin-corrected XYZ.",
        "degrees",
        "Superior-inferior orientation of activation-independent "
        "repolarization heterogeneity.",
        _ventricular_gradient_elevation,
    ),
    _MarkerSpec(
        "qrs_loop_planarity_deviation",
        "Loop-Planaritaet",
        "Out-of-plane variance fraction of the QRS loop.",
        "lambda3 / (lambda1 + lambda2 + lambda3).",
        "dimensionless",
        "A normal QRS loop is nearly planar; conduction disease and scar bend "
        "it out of its own best-fit plane.",
        _qrs_loop_planarity_deviation,
    ),
    _MarkerSpec(
        "t_loop_planarity_deviation",
        "Loop-Planaritaet",
        "Out-of-plane variance fraction of the T loop.",
        "lambda3 / (lambda1 + lambda2 + lambda3).",
        "dimensionless",
        "Non-planar repolarization; complements the non-dipolar fraction.",
        _t_loop_planarity_deviation,
    ),
    _MarkerSpec(
        "qrs_initial_to_peak_angle_deg",
        "Aktivierung-Erholung",
        "Angle from the initial 20 ms QRS vector to the QRS peak vector.",
        "angle(mean initial 20 ms, peak magnitude vector).",
        "degrees",
        "Rotation from septal to free-wall activation; widened where septal "
        "activation is lost or reversed.",
        _qrs_initial_to_peak_angle,
    ),
    _MarkerSpec(
        "r_peak_timing_fraction",
        "Aktivierung-Erholung",
        "Relative position of the QRS magnitude peak inside the complex.",
        "argmax(|QRS|) / (n - 1).",
        "dimensionless",
        "Lead-free analogue of the intrinsicoid deflection: a late peak means "
        "delayed activation.",
        _r_peak_timing_fraction,
    ),
    _MarkerSpec(
        "st_to_t_vector_angle_deg",
        "Ischaemie-Vektorik",
        "Angle between the mean ST and mean T vectors.",
        "angle(mean ST vector, mean T vector).",
        "degrees",
        "Separates a primary injury current from a secondary repolarization "
        "change, which follow different directions.",
        _st_t_vector_angle,
    ),
    _MarkerSpec(
        "repolarization_time_dispersion",
        "Repolarisationsheterogenitaet",
        "Normalised temporal spread of the T loop magnitude.",
        "sd of the magnitude-weighted time index / segment length.",
        "dimensionless",
        "Broad versus peaked repolarization, independent of amplitude.",
        _repolarization_time_dispersion,
    ),
    _MarkerSpec(
        "qrs_energy_concentration",
        "Vektor-Kinetik",
        "Share of QRS energy inside its densest quarter.",
        "max windowed sum(|v|^2) over a quarter / total sum(|v|^2).",
        "dimensionless",
        "A concentrated QRS gives a high value; fragmentation and widening "
        "spread the energy out.",
        _qrs_energy_concentration,
    ),
    _MarkerSpec(
        "beat_to_beat_qt_variability_ms",
        "Repolarisationslabilitaet",
        "Standard deviation of the QT interval across beats.",
        "sd(T offset - QRS onset) over evaluable beats.",
        "ms",
        "Repolarization lability. Ten seconds gives few beats, so this is a "
        "descriptive estimate.",
        _beat_to_beat_qt_variability,
    ),
    _MarkerSpec(
        "qt_variability_index_surrogate",
        "Repolarisationslabilitaet",
        "QT scatter divided by RR scatter.",
        "sd(QT) / sd(RR).",
        "dimensionless",
        "The QT variability index concept: repolarization instability beyond "
        "what the rate alone explains.",
        _qt_variability_normalised,
    ),
    _MarkerSpec(
        "p_terminal_energy_fraction",
        "Atriale Geometrie",
        "Share of P-wave energy in its terminal third.",
        "sum(|P|^2 over terminal third) / sum(|P|^2).",
        "dimensionless",
        "Terminal P activation is the left atrial component; a raised share "
        "is the vector correlate of terminal force.",
        _p_wave_energy_fraction_terminal,
    ),
    _MarkerSpec(
        "p_loop_direction_reversal_count",
        "Atriale Geometrie",
        "Number of direction reversals inside the P loop.",
        "Count of consecutive step vectors more than 90 degrees apart.",
        "count",
        "Biphasic or notched P waves, the surface sign of interatrial "
        "conduction delay.",
        _p_loop_direction_reversal_count,
    ),
    _MarkerSpec(
        "atrial_conduction_velocity_surrogate",
        "Atriale Geometrie",
        "P loop path length divided by P duration.",
        "path length(|P loop|) / P duration.",
        "uV/ms",
        "Slow atrial conduction traces a shorter path per unit time.",
        _atrial_conduction_velocity_surrogate,
    ),
    _MarkerSpec(
        "p_duration_to_pr_interval_ratio",
        "Atriale Geometrie",
        "P duration as a share of the total PR interval.",
        "P duration / (P duration + PR segment duration).",
        "dimensionless",
        "Splits the PR interval into atrial activation and AV nodal delay, "
        "which have different causes and different drugs.",
        _pr_interval_to_p_duration_ratio,
    ),
    _MarkerSpec(
        "s1q3t3_vector_score",
        "Rechtsherz-Vektorik",
        "Amplitude-normalized continuous strength of S in I, Q in III and negative T in III.",
        "cuberoot((S1/QRS_scale)*(Q3/QRS_scale)*(-T3/T_scale)).",
        "dimensionless",
        "Supportive lead-space right-heart-strain pattern, neither specific nor diagnostic.",
        _s1q3t3_vector_score,
        category="ecg",
        reference_keys=(),
    ),
    _MarkerSpec(
        "anterior_t_inversion_extent_uv",
        "Rechtsherz-Vektorik",
        "Negated anterior component of the mean T vector.",
        "-z of the mean T vector.",
        "uV",
        "Anterior T inversion, seen in right ventricular strain and ARVC.",
        _right_precordial_t_inversion_extent,
    ),
    _MarkerSpec(
        "qrs_t_discordance_cosine",
        "QRS-T-Kopplung",
        "Cosine between the mean QRS and mean T vectors.",
        "dot(mean QRS, mean T) / (|mean QRS| |mean T|).",
        "dimensionless",
        "Signed discordance; negative is the strain and secondary-change "
        "direction.",
        _qrs_axis_to_t_axis_discordance_signed,
    ),
    _MarkerSpec(
        "strain_pattern_score",
        "Hypertrophie-Vektorik",
        "High QRS voltage with an opposing asymmetric T wave.",
        "peak QRS voltage * max(0, -discordance) * (1 + |symmetry - 1|).",
        "dimensionless",
        "The three elements that jointly define the strain pattern of "
        "pressure-overload hypertrophy.",
        _strain_pattern_score,
    ),
    _MarkerSpec(
        "low_voltage_with_preserved_duration",
        "Hypertrophie-Vektorik",
        "QRS duration divided by peak QRS voltage.",
        "QRS duration / max(|QRS|).",
        "ms/uV",
        "The infiltrative signature: amyloid and effusion reduce voltage "
        "without shortening activation.",
        _low_voltage_with_preserved_duration,
    ),
    _MarkerSpec(
        "qrs_loop_compactness",
        "Loop-Planaritaet",
        "QRS loop ellipse area divided by its perimeter.",
        "pi*sqrt(lambda1*lambda2) / path length.",
        "mV",
        "Compact versus elongated activation trajectory.",
        _qrs_area_to_perimeter_ratio,
    ),
    _MarkerSpec(
        "t_loop_compactness",
        "Loop-Planaritaet",
        "T loop ellipse area divided by its perimeter.",
        "pi*sqrt(lambda1*lambda2) / path length.",
        "mV",
        "Compact versus elongated repolarization trajectory.",
        _t_loop_area_to_perimeter_ratio,
    ),
    _MarkerSpec(
        "t_to_qrs_loop_scale_ratio",
        "QRS-T-Kopplung",
        "Peak T magnitude relative to peak QRS magnitude.",
        "max(|T|) / max(|QRS|).",
        "dimensionless",
        "Relative repolarization amplitude; falls in low-voltage states and "
        "rises with hyperacute T waves.",
        _qrs_t_loop_scale_ratio,
    ),
    _MarkerSpec(
        "pacing_spike_slope_ratio",
        "Signalgrundlinie",
        "Maximum wideband slope relative to the median slope inside the QRS.",
        "max(|d|v||) / median(|d|v||).",
        "dimensionless",
        "A pacing spike is a sub-millisecond transient with disproportionate "
        "slope; kept continuous rather than a binary sighting. Measured on "
        "597 recordings it separates paced from unpaced at AUC 0,523 -- the "
        "0,5-ms spike does not survive 500 Hz sampling behind a 150 Hz "
        "analogue corner, so this marker must not be read as pacing "
        "detection. QRS duration reaches 0,884 on the same recordings.",
        _pacing_spike_energy_ratio,
    ),
    _MarkerSpec(
        "hf_qrs_reduced_amplitude_zone_depth",
        "Terminale QRS-Energie",
        "Relative depth of the deepest trough in the axiswise 80-150 Hz envelope.",
        "(max spatial analytic envelope - min interior) / max, after XYZ 80-150 Hz filtering.",
        "dimensionless",
        "The reduced-amplitude zone of high-frequency QRS analysis, linked to "
        "ischaemia in exercise testing.",
        _qrs_high_frequency_reduced_amplitude_zone,
    ),
)

ELECTROPHYSIOLOGY_SPECS = ELECTROPHYSIOLOGY_SPECS + _SECOND_BATCH

for _segment in ("p", "qrs", "t"):
    ELECTROPHYSIOLOGY_SPECS = ELECTROPHYSIOLOGY_SPECS + (
        _MarkerSpec(
            f"{_segment}_spatial_velocity_entropy",
            "Vektor-Kinetik",
            f"Normalised entropy of the {_segment.upper()} velocity profile.",
            "-sum(p log p) / log(n) over the normalised velocity profile.",
            "dimensionless",
            "A smooth single-peaked wavefront has low entropy; several bursts "
            "of movement raise it. Amplitude-independent by construction.",
            _segment_marker(_segment, _spatial_velocity_entropy),
        ),
    )


class ElectrophysiologyBiomarker(BiomarkerBase):
    """Adapter exposing one specification as a registry biomarker."""

    def __init__(self, spec: _MarkerSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return self._spec.category

    @property
    def references(self) -> list[str]:
        return _references_for_spec(self._spec)

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead=None,
        vcg_xyz=None,
        fs: int = 500,
    ) -> BiomarkerResult:
        context = _context(delineation, ecg_12lead, vcg_xyz, fs)
        try:
            value = self._spec.calculator(context)
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            return BiomarkerResult(
                name=self._spec.name,
                value=None,
                unit=self._spec.unit,
                category=self.category,
                confidence=0.0,
                computation_notes=f"error: {type(exc).__name__}: {exc}",
                family=self._spec.family,
                formula=self._spec.formula,
                references=self.references,
            )
        return BiomarkerResult(
            name=self._spec.name,
            value=value.value,
            unit=self._spec.unit,
            category=self.category,
            confidence=value.confidence,
            per_lead=value.per_lead,
            description=self._spec.description,
            computation_notes=value.notes,
            family=self._spec.family,
            formula=self._spec.formula,
            references=self.references,
            pathological_direction="data_driven",
            pathological_note=self._spec.interpretation,
        )


def register_electrophysiology_markers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in ELECTROPHYSIOLOGY_SPECS:
        if spec.name in existing:
            continue
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category=spec.category,
                source=ELECTROPHYSIOLOGY_SOURCE,
                pathological_direction="data_driven",
                pathological_note=spec.interpretation,
                implementation_status="implemented",
                references=_references_for_spec(spec),
                notes=(
                    f"{ELECTROPHYSIOLOGY_VERSION}; conceived after inspection "
                    "of development results and therefore Discovery Reserve "
                    "only until independent replication."
                ),
            )
        )
        BiomarkerRegistry.register(ElectrophysiologyBiomarker(spec))
        existing.add(spec.name)


__all__ = [
    "ELECTROPHYSIOLOGY_SPECS",
    "ELECTROPHYSIOLOGY_VERSION",
    "ElectrophysiologyBiomarker",
    "register_electrophysiology_markers",
]
