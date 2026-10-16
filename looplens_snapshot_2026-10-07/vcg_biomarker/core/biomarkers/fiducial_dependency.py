"""How much does a landmark's *measured error* move each marker?

A declared limitation in the delineation release gate -- "the P onset misses
the external accuracy standard at every operating point" -- is only honest if
it costs something.  The cost is that markers whose values are substantially
driven by that landmark's error must not carry the same evidential claim as
markers resting on a sound one.  Applying the cost needs a way to decide which
markers those are.

**Why not a hand-written list.** It would assert a moment in time: correct the
day it is written, silently wrong the first time a marker is added, and wrong
in the dangerous direction -- a marker that drops off the list keeps a claim it
no longer deserves.  Static source scanning fails too, because markers reach
their landmarks by several routes (``_sample``, direct attribute access,
segment bounds) and a scanner must know them all in advance.

**Why not plain ablation.** Deleting the landmark and seeing what moves was the
first attempt, and it over-includes to the point of uselessness: on LUDB it
made 1557 of 1851 markers depend on the P onset.  That is arithmetically true
and analytically worthless.  The P onset defines the left edge of the whole
beat window, so removing it shifts nearly every segment-derived quantity by
some amount -- including markers a 26 ms shift would barely touch.  A criterion
that demotes 84 % of the catalog separates nothing.

**What is measured instead.** The landmark is *perturbed* by its own measured
error rather than removed, and the resulting movement is compared with the
spread the marker shows between records:

    r = sigma_error / sigma_between        reliability = 1 / (1 + r^2)

This is the classical measurement-error framing.  When a landmark's error moves
a marker as much as real between-subject differences do, the marker's
reliability collapses and any association it shows is attenuated towards the
null by roughly that factor.  The demotion threshold is reliability < 0.70, the
long-standing convention for "adequate for research use", rather than a cut
invented here.

Two properties worth stating:

* **Direction of error is not guaranteed.** Attenuation is a result under
  classical independent, nondifferential additive measurement-error
  assumptions, not a general property of delineation errors. Systematic,
  rhythm-dependent or disease-dependent boundary errors can bias effects in
  either direction or create associations. This perturbation ratio does not
  test those assumptions, establish specificity, or validate a clinical
  claim. The fixed reliability threshold is a research screening convention,
  not a universal clinical acceptance criterion.

* **The map is keyed by the implementation hash.** It describes one version of
  the marker code, and reusing it across a code change is exactly the
  reuse-key mistake this project has hit before, so the recorded hash is
  checked rather than trusted.

The between-record spread is estimated from whatever records the caller passes,
which is a rough stand-in for between-subject variation in the production
cohort.  It is a benchmark-cohort estimate, and the resulting reliability is
approximate; it is used to sort markers into two tiers, not to correct them.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..delineation.base import RecordDelineation
from .base import BiomarkerRegistry


FIDUCIAL_DEPENDENCY_VERSION = "2.0.0"

# Reliability below which a marker is considered too strongly driven by the
# landmark's error to carry a confirmatory claim.  0.70 is the conventional
# floor for research use; it is not tuned to this catalog.
RELIABILITY_DEMOTION_THRESHOLD = 0.70

# The measured absolute error of each landmark on LUDB, in milliseconds, used
# as the perturbation amplitude.  These are measurements, not choices; they
# move when the delineator changes and the map is re-measured with them.
MEASURED_LANDMARK_ERROR_MS: dict[str, float] = {
    "p_onset": 26.3,
    "p_offset": 12.5,
    "r_onset": 10.2,
    "r_offset": 8.1,
    "t_onset": 21.4,
    "t_offset": 25.6,
}


def perturb_landmark(
    delineation: RecordDelineation,
    landmark: str,
    shift_samples: int,
) -> RecordDelineation:
    """A copy with ``landmark`` displaced by ``shift_samples`` on every beat.

    Deep-copied rather than mutated: the caller's delineation is also the
    baseline, and a shared beat object would make the two runs differ for the
    wrong reason.
    """

    shifted = copy.deepcopy(delineation)
    limit = int(getattr(shifted, "n_samples", 0)) - 1
    for beat in getattr(shifted, "beats", []) or []:
        point = getattr(beat, landmark, None)
        if point is None:
            continue
        moved = int(point.sample) + int(shift_samples)
        if limit > 0:
            moved = max(0, min(limit, moved))
        point.sample = moved
    return shifted


def _values(
    delineation: RecordDelineation,
    ecg_12lead: Optional[np.ndarray],
    vcg_xyz: Optional[np.ndarray],
    fs: int,
) -> dict[str, Optional[float]]:
    results = BiomarkerRegistry.compute_all(
        delineation,
        ecg_12lead=ecg_12lead,
        vcg_xyz=vcg_xyz,
        fs=fs,
        record_ecg_12lead=ecg_12lead,
        record_vcg_xyz=vcg_xyz,
    )
    values: dict[str, Optional[float]] = {}
    for result in results:
        value = getattr(result, "value", None)
        if isinstance(value, (int, float, np.floating, np.integer)):
            numeric = float(value)
            values[result.name] = numeric if np.isfinite(numeric) else None
        else:
            values[result.name] = None
    return values


@dataclass
class LandmarkSensitivity:
    """One marker's reliability under one landmark's measured error."""

    marker: str
    landmark: str
    error_sigma: float
    between_sigma: float
    reliability: float
    n_records: int

    @property
    def demoted(self) -> bool:
        return self.reliability < RELIABILITY_DEMOTION_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "landmark": self.landmark,
            "error_sigma": self.error_sigma,
            "between_sigma": self.between_sigma,
            "reliability": self.reliability,
            "n_records": int(self.n_records),
            "demoted": bool(self.demoted),
        }


@dataclass
class FiducialSensitivityMap:
    """Per-marker reliability under each perturbed landmark."""

    sensitivities: dict[str, dict[str, LandmarkSensitivity]] = field(
        default_factory=dict
    )
    landmarks_measured: tuple[str, ...] = ()
    records_measured: int = 0
    implementation_hash: str = ""

    def markers_demoted_by(self, landmarks: Iterable[str]) -> set[str]:
        """Markers whose reliability falls below the threshold for any of these."""

        wanted = {str(name) for name in landmarks}
        demoted: set[str] = set()
        for marker, per_landmark in self.sensitivities.items():
            for landmark, sensitivity in per_landmark.items():
                if landmark in wanted and sensitivity.demoted:
                    demoted.add(marker)
        return demoted

    def to_dict(self) -> dict[str, object]:
        return {
            "version": FIDUCIAL_DEPENDENCY_VERSION,
            "implementation_hash": self.implementation_hash,
            "landmarks_measured": list(self.landmarks_measured),
            "records_measured": int(self.records_measured),
            "measured_by": "perturbation_by_measured_landmark_error",
            "reliability_demotion_threshold": RELIABILITY_DEMOTION_THRESHOLD,
            "landmark_error_ms": dict(MEASURED_LANDMARK_ERROR_MS),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sensitivities": {
                marker: {
                    landmark: sensitivity.to_dict()
                    for landmark, sensitivity in sorted(per_landmark.items())
                }
                for marker, per_landmark in sorted(self.sensitivities.items())
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "FiducialSensitivityMap":
        sensitivities: dict[str, dict[str, LandmarkSensitivity]] = {}
        for marker, per_landmark in dict(
            payload.get("sensitivities") or {}
        ).items():
            for landmark, body in dict(per_landmark or {}).items():
                sensitivities.setdefault(str(marker), {})[str(landmark)] = (
                    LandmarkSensitivity(
                        marker=str(marker),
                        landmark=str(landmark),
                        error_sigma=float(body.get("error_sigma", 0.0)),
                        between_sigma=float(body.get("between_sigma", 0.0)),
                        reliability=float(body.get("reliability", 1.0)),
                        n_records=int(body.get("n_records", 0)),
                    )
                )
        return cls(
            sensitivities=sensitivities,
            landmarks_measured=tuple(
                str(item) for item in (payload.get("landmarks_measured") or [])
            ),
            records_measured=int(payload.get("records_measured", 0) or 0),
            implementation_hash=str(payload.get("implementation_hash", "")),
        )


def measure_fiducial_sensitivity(
    records: Sequence[
        tuple[RecordDelineation, Optional[np.ndarray], Optional[np.ndarray], int]
    ],
    *,
    landmarks: Iterable[str] = ("p_onset",),
    implementation_hash: str = "",
    progress: Optional[Callable[[str], None]] = None,
) -> FiducialSensitivityMap:
    """Perturb each landmark by its measured error and record the reliability.

    The perturbation is applied in both directions and the two displacements
    are pooled, so a marker that reacts asymmetrically -- common where a
    boundary can cross a peak -- is not judged on the gentler side alone.

    The shift is a **probe, not a model of the error**.  Applying the same
    displacement to every beat of a record measures the marker's slope with
    respect to that landmark; setting the amplitude to the landmark's measured
    error magnitude then makes the observed displacement an estimate of the
    marker's own error SD *across records*.  That second step is what makes
    the number meaningful, and it is why the amplitude has to be a measurement
    rather than a round figure.

    A purely constant bias -- every record shifted identically -- would not
    attenuate anything, because it preserves rank order.  What attenuates is
    error that differs between records, and 26.3 ms is exactly that: the
    typical magnitude by which a given record's P onset is misplaced, varying
    in sign and size from record to record.  Applying it within a record and
    reading it as between-record error is therefore the correct reading, not a
    conflation of the two.
    """

    landmark_list = [str(item) for item in landmarks]
    baseline_by_record: list[dict[str, Optional[float]]] = []
    deltas: dict[str, dict[str, list[float]]] = {}

    for index, (delineation, ecg, vcg, fs) in enumerate(records, start=1):
        if progress is not None:
            progress(f"Stoerung {index}/{len(records)}")
        baseline = _values(delineation, ecg, vcg, fs)
        baseline_by_record.append(baseline)
        for landmark in landmark_list:
            error_ms = MEASURED_LANDMARK_ERROR_MS.get(landmark)
            if error_ms is None:
                continue
            if not any(
                getattr(beat, landmark, None) is not None
                for beat in (getattr(delineation, "beats", []) or [])
            ):
                continue
            shift = max(1, int(round(error_ms / 1000.0 * float(fs))))
            for signed in (shift, -shift):
                moved = _values(
                    perturb_landmark(delineation, landmark, signed),
                    ecg,
                    vcg,
                    fs,
                )
                for marker, value in baseline.items():
                    other = moved.get(marker)
                    if value is None or other is None:
                        continue
                    deltas.setdefault(marker, {}).setdefault(
                        landmark, []
                    ).append(abs(float(other) - float(value)))

    sensitivities: dict[str, dict[str, LandmarkSensitivity]] = {}
    for marker, per_landmark in deltas.items():
        observed = [
            record.get(marker)
            for record in baseline_by_record
            if record.get(marker) is not None
        ]
        between = (
            float(np.std(np.asarray(observed, dtype=float), ddof=1))
            if len(observed) > 1
            else 0.0
        )
        for landmark, values in per_landmark.items():
            if not values:
                continue
            # The mean absolute displacement *is* the error-SD estimate.  If
            # the marker responds as m = m0 + k*e, shifting the landmark by
            # its own error magnitude yields |delta| = k*sigma_e, which is
            # exactly the marker's error SD.  Halving it here would flatter
            # every marker by a factor of two.
            error_sigma = float(np.mean(values))
            if between <= 0.0:
                # No between-record variation observed: the marker is constant
                # on this cohort, so reliability is undefined rather than
                # zero.  Treated as reliable, because a marker that does not
                # vary cannot have its variation destroyed.
                reliability = 1.0
            else:
                ratio = error_sigma / between
                reliability = 1.0 / (1.0 + ratio * ratio)
            sensitivities.setdefault(marker, {})[landmark] = (
                LandmarkSensitivity(
                    marker=marker,
                    landmark=landmark,
                    error_sigma=error_sigma,
                    between_sigma=between,
                    reliability=float(reliability),
                    n_records=len(baseline_by_record),
                )
            )

    return FiducialSensitivityMap(
        sensitivities=sensitivities,
        landmarks_measured=tuple(landmark_list),
        records_measured=len(records),
        implementation_hash=str(implementation_hash),
    )


def write_sensitivity_map(path: Path, sensitivity_map: FiducialSensitivityMap) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sensitivity_map.to_dict(), indent=2),
        encoding="utf-8",
    )
    return path


def read_sensitivity_map(
    path: Path,
    *,
    expected_implementation_hash: Optional[str] = None,
) -> Optional[FiducialSensitivityMap]:
    """Load a recorded map, rejecting one written by different marker code.

    Returns ``None`` rather than raising: a missing or stale map means the
    measurement has to be redone, which is a scheduling problem and not a
    reason to end a run.  What must never happen is *using* a stale one, since
    it would attach the limitation to the wrong markers.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("version", "")) != FIDUCIAL_DEPENDENCY_VERSION:
        return None
    if expected_implementation_hash is not None:
        if str(payload.get("implementation_hash", "")) != str(
            expected_implementation_hash
        ):
            return None
    return FiducialSensitivityMap.from_dict(payload)


__all__ = [
    "FIDUCIAL_DEPENDENCY_VERSION",
    "MEASURED_LANDMARK_ERROR_MS",
    "RELIABILITY_DEMOTION_THRESHOLD",
    "FiducialSensitivityMap",
    "LandmarkSensitivity",
    "measure_fiducial_sensitivity",
    "perturb_landmark",
    "read_sensitivity_map",
    "write_sensitivity_map",
]
