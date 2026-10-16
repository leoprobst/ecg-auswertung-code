"""Two markers from the third propagation research round: uniform slow conduction.

The round is documented in ``RECHERCHE_EP_RUNDE3_2026-09-12.md`` at the
project root. Its finding for these two markers is Ciaccio et al. (Heart
Rhythm 2025): in post-infarction ventricles mapped during sinus rhythm, the
isthmus of the later ventricular tachycardia is the place where conduction
is **uniformly slow** -- activation progresses linearly (r-squared 0,93) at
0,16 mm/ms along the isthmus, against irregular and fast (0,73 mm/ms)
elsewhere -- while the surrounding scar shows abrupt voltage steps. The
NIAVAS study (Reventos-Presmanes et al., Heart Rhythm 2026) reads the same
substrate non-invasively as regional activation dispersion and finds seven
of ten deceleration zones with it.

The registry already knows stalls: the fraction of time below a fifth of
the median speed in each half of the QRS, the slowest point's direction,
stop-and-go and burst indices. What it does not know is the property
Ciaccio names -- a slow stretch that is *sustained* and *even*. A normal
loop is slow only where it turns: at onset, at offset and in the trough
between two propagation bursts, and in each of those places the speed is
falling or rising, not steady. A loop that crawls through a channel of
surviving fibres is slow for a while and at a steady pace.

``qrs_longest_slow_run_ms``
    The longest contiguous stretch of the QRS during which the spatial
    speed stays below a quarter of its maximum (the share the
    velocity-landmark rule uses for a genuine burst) **and which is
    flanked by fast conduction on both sides**. Runs that touch the onset
    or the offset are the ramps every loop has and are not counted; a
    first form cut a fixed tenth from either end instead, and on thirty
    recordings it read 18-26 ms in every one of them, because the ramps
    reach past any fixed margin. In milliseconds; zero when there is no
    flanked slow stretch, which is what a single-burst loop should read.

``qrs_slow_run_speed_cv``
    The coefficient of variation of the speed inside that longest run.
    Ciaccio's uniformity: a plateau reads low, a trough between bursts
    reads high. Abstains when the run is shorter than four samples.

Both are built from the three-dimensional path's speed; no fiducial enters
beyond the QRS window itself, and both are invariant to amplitude, so the
volume conductor cannot forge them (the meta-thesis of the first theses
paper). Predictions, written before the measurement: old infarction shows
a longer run and a lower coefficient of variation than controls; left
bundle branch block a longer run (the mid-QRS plateau between the septal
and the free-wall fronts). Right bundle branch block was *not* predicted to
read longer: its delay was expected to touch the offset and be a ramp by
this rule.

**Measured on 2.964 recordings of the training split.** The run is zero in
45 % of recordings (single-burst loops, as intended), median 2 ms, and
reads longer under every conduction disturbance: left bundle branch block
**6 against 2 ms** (AUC 0,689), pacing 8 against 2 (0,751), reduced LVEF
4 against 2 (0,631) -- and right bundle branch block **10 against 2 ms**
(AUC 0,799), the opposite of what was written. The error was the
prediction, not the marker: the terminal delay of the block is a flanked
plateau followed by the late right-ventricular burst, not a ramp. Old
infarction, the thesis's own target, reads **AUC 0,579 with equal medians**
(2 against 2 ms): not refuted, not supported. The coefficient of variation
is available on 16 % of recordings (a flanked run of at least four
samples) and **does not differ in old infarction** (0,268 against 0,266,
AUC 0,510): the prediction as written is refuted. Ciaccio's population was
post-infarction patients *with* ventricular tachycardia, and "old
infarction" was a generalisation of it; the faithful test -- old
infarction with a later arrest or tachycardia against old infarction
without -- is left for the full run, and if that fails the marker goes.
Neither marker is redundant (joint R-squared 0,29 and 0,13 against the
stall, stop-go, slowing and duration neighbours), and together they cost
0,4 ms per recording, paired and order-alternated, against a spread of
45 ms in the same computation.
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
from ..delineation.base import RecordDelineation


CONDUCTION_UNIFORMITY_VERSION = "1.1.0"
CONDUCTION_UNIFORMITY_SOURCE = "conduction_uniformity_v1"

EPS = 1e-12

# Slow means below this share of the maximum spatial speed -- the share the
# velocity-landmark rule uses to tell a genuine burst from a ripple.
SLOW_SHARE = 0.25


# A coefficient of variation over fewer samples than this is not a shape.
MINIMUM_RUN_SAMPLES = 4
MINIMUM_RUN_DURATION_MS = 8.0


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_conduction_uniformity_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_conduction_uniformity_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def slow_runs(speed: np.ndarray, share: float = SLOW_SHARE) -> list[tuple[int, int]]:
    """Maximal stretches [start, end) of speed below share * max that are
    flanked by faster conduction on both sides.

    A stretch that touches the first or the last sample is an onset or
    offset ramp and is not returned.
    """

    values = np.asarray(speed, dtype=float)
    n = len(values)
    if n == 0 or not np.all(np.isfinite(values)):
        return []
    maximum = float(np.max(values))
    if maximum <= EPS:
        return []
    slow = values < share * maximum
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index in range(n):
        if slow[index]:
            if start is None:
                start = index
        elif start is not None:
            if start > 0:
                runs.append((start, index))
            start = None
    # a run still open at the end touches the offset and is a ramp
    return runs


def _longest_run(ctx: FeatureContext) -> Optional[tuple[int, int]]:
    from .loop_parametrisation_biomarkers import _speed

    def build() -> Optional[tuple[int, int]]:
        speed = _speed(ctx)
        if speed is None:
            return None
        runs = slow_runs(speed)
        if not runs:
            return (0, 0)
        return max(runs, key=lambda run: run[1] - run[0])

    return _memo(ctx, ("longest_run",), build)


def _longest_slow_run_ms(ctx: FeatureContext) -> FeatureValue:
    run = _longest_run(ctx)
    if run is None:
        return _abstain("QRS speed unavailable")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if not np.isfinite(fs) or fs <= 0:
        return _abstain("sampling rate unavailable")
    return FeatureValue(1000.0 * (run[1] - run[0]) / fs)


def _slow_run_speed_cv(ctx: FeatureContext) -> FeatureValue:
    from .loop_parametrisation_biomarkers import _speed

    run = _longest_run(ctx)
    if run is None:
        return _abstain("QRS speed unavailable")
    if run[1] - run[0] < MINIMUM_RUN_SAMPLES:
        return _abstain("no slow run of at least four samples")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if not np.isfinite(fs) or fs <= 0:
        return _abstain("sampling rate unavailable")
    if 1000.0 * (run[1] - run[0]) / fs < MINIMUM_RUN_DURATION_MS:
        return _abstain("slow run shorter than 8 ms")
    speed = _speed(ctx)
    segment = np.asarray(speed[run[0] : run[1]], dtype=float)
    mean = float(np.mean(segment))
    if mean <= EPS:
        return _abstain("slow run has no motion")
    return FeatureValue(float(np.std(segment, ddof=1) / mean))


@dataclass(frozen=True)
class ConductionUniformitySpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    thesis: str
    references: tuple[str, ...] = ()


REF_ROUND = (
    "RECHERCHE_EP_RUNDE3_2026-09-12.md, project root: third propagation "
    "research round, with the predictions written before measurement."
)
REF_CIACCIO = (
    "Ciaccio EJ, Hsia HH, Robinson D, et al. Uniform slow conduction during "
    "sinus rhythm and low voltage/low voltage gradient characterize the VT "
    "isthmus location. Heart Rhythm 2025;22:2486-2496."
)
REF_NIAVAS = (
    "Reventos-Presmanes J, Hernandez-Romero I, Pellicer-Sendra B, et al. "
    "Noninvasive assessment of the ventricular arrhythmogenic substrate using "
    "electrocardiographic imaging during sinus rhythm: the NIAVAS study. "
    "Heart Rhythm 2026;23:249-263."
)


CONDUCTION_UNIFORMITY_SPECS: tuple[ConductionUniformitySpec, ...] = (
    ConductionUniformitySpec(
        "qrs_longest_slow_run_ms",
        "Erregungsstruktur",
        "Longest contiguous stretch of the QRS with spatial speed below a "
        "quarter of its maximum, flanked by faster conduction on both sides.",
        "max over maximal runs of {t : speed(t) < 0.25 * max speed} that do "
        "not touch the QRS onset or offset, in ms; 0 if none.",
        "ms",
        _longest_slow_run_ms,
        "Uniform slow conduction in sinus rhythm marks the substrate "
        "(Ciaccio 2025). Refuted if old infarction does not show a longer "
        "run than controls.",
        references=(REF_ROUND, REF_CIACCIO, REF_NIAVAS),
    ),
    ConductionUniformitySpec(
        "qrs_slow_run_speed_cv",
        "Erregungsstruktur",
        "Coefficient of variation of the spatial speed inside the longest "
        "slow run: a plateau reads low, a trough between bursts reads high.",
        "std(speed in run, ddof=1) / mean(speed in run) over the longest slow "
        "run; abstains below four samples or 8 ms. Dimensionless CV is not "
        "a fraction constrained to [0,1].",
        "dimensionless",
        _slow_run_speed_cv,
        "Exploratory global morphology hypothesis inspired by local contact "
        "mapping, not a measurement of a VT isthmus. Similar pilot group "
        "medians (0.268 vs 0.266) do not establish equivalence or refute the "
        "local conduction mechanism. Require adequate availability, precision "
        "and independent arrhythmic reference outcomes. Refuted if adequate "
        "independent precision excludes the prespecified global prognostic "
        "effect; not a refutation of the original local mapping mechanism.",
        references=(REF_ROUND, REF_CIACCIO),
    ),
)


class ConductionUniformityBiomarker(BiomarkerBase):
    def __init__(self, spec: ConductionUniformitySpec) -> None:
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
        notes = [f"conduction_uniformity_version={CONDUCTION_UNIFORMITY_VERSION}"]
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


def conduction_uniformity_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in CONDUCTION_UNIFORMITY_SPECS)


def register_conduction_uniformity_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in CONDUCTION_UNIFORMITY_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), ConductionUniformityBiomarker):
                continue
            raise RuntimeError(
                f"Conduction uniformity marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=CONDUCTION_UNIFORMITY_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Derived from the 2025-2026 mapping literature on uniform "
                    "slow conduction, with the prediction written before the "
                    "marker was built. " + spec.thesis
                    + " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(ConductionUniformityBiomarker(spec))
