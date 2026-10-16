"""Carry the delineation's declared limitations into the marker results.

The release gate can declare a landmark limited: measured, reported, and known
to miss the external accuracy standard at every operating point.  The P onset
is such a landmark -- 26.3 ms against an inter-observer band of 10-11 ms.

A declaration that changes nothing downstream is a disabled test with a nicer
name.  This module is the part that makes it cost something: markers whose
reliability collapses under that landmark's measured error do not get to carry
a confirmatory claim.  They are still computed, still reported, still
available for discovery -- they simply are not presented at the same evidential
standard as markers resting on a sound landmark.

Which markers those are is measured, not declared: see
``core/biomarkers/fiducial_dependency.py``.  This module only joins the two
sides -- the gate's declared limitations and the measured sensitivity map --
and answers "may this marker be confirmed".

**Missing map is an error, not a default.**  If a limitation is declared and
the map is absent or was written by different marker code, there is no honest
answer to that question.  Continuing without the map would grant every marker
the claim the limitation was meant to withhold, and the failure would be
invisible in the results.  So it raises, with the command that fixes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.biomarkers.fiducial_dependency import (
    FIDUCIAL_DEPENDENCY_VERSION,
    RELIABILITY_DEMOTION_THRESHOLD,
    FiducialSensitivityMap,
    read_sensitivity_map,
)


FIDUCIAL_LIMITATION_VERSION = "1.0.0"
SENSITIVITY_MAP_FILENAME = "fiducial_sensitivity.json"


class MissingFiducialSensitivityMap(RuntimeError):
    """A limitation is declared but cannot be applied."""


@dataclass(frozen=True)
class FiducialLimitationPolicy:
    """Which markers may not carry a confirmatory claim, and why."""

    limited_landmarks: tuple[str, ...]
    demoted_markers: frozenset[str]
    reliability_threshold: float
    map_records_measured: int
    map_implementation_hash: str

    def marker_may_be_confirmed(self, feature_id: str) -> bool:
        return str(feature_id) not in self.demoted_markers

    def to_dict(self) -> dict[str, object]:
        return {
            "version": FIDUCIAL_LIMITATION_VERSION,
            "limited_landmarks": list(self.limited_landmarks),
            "reliability_threshold": self.reliability_threshold,
            "demoted_marker_count": len(self.demoted_markers),
            "map_records_measured": int(self.map_records_measured),
            "map_implementation_hash": self.map_implementation_hash,
        }


def sensitivity_map_path(resources_root: Path) -> Path:
    return Path(resources_root) / "delineation" / SENSITIVITY_MAP_FILENAME


def declared_limitation_landmarks(benchmark_suite_path: Path) -> tuple[str, ...]:
    """Landmarks the recorded gate declared limited.

    Read from the artifact rather than from the policy constants: the run must
    apply the limitations of the gate it actually passed, not of whatever the
    source tree currently says.
    """

    try:
        payload = json.loads(
            Path(benchmark_suite_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ()
    gate = payload.get("release_gate")
    if not isinstance(gate, dict):
        return ()
    landmarks = gate.get("affected_landmarks")
    if isinstance(landmarks, list):
        return tuple(str(item) for item in landmarks)
    return tuple(
        sorted(
            {
                str(entry.get("landmark"))
                for entry in (gate.get("declared_limitations") or [])
                if entry.get("landmark")
            }
        )
    )


def load_limitation_policy(
    *,
    resources_root: Path,
    benchmark_suite_path: Path,
    feature_implementation_hash: Optional[str] = None,
) -> FiducialLimitationPolicy:
    """Join the recorded gate with the measured sensitivity map."""

    landmarks = declared_limitation_landmarks(benchmark_suite_path)
    if not landmarks:
        # Nothing declared: no marker is demoted, and no map is required.
        return FiducialLimitationPolicy(
            limited_landmarks=(),
            demoted_markers=frozenset(),
            reliability_threshold=RELIABILITY_DEMOTION_THRESHOLD,
            map_records_measured=0,
            map_implementation_hash="",
        )

    path = sensitivity_map_path(resources_root)
    sensitivity: Optional[FiducialSensitivityMap] = read_sensitivity_map(
        path, expected_implementation_hash=feature_implementation_hash
    )
    if sensitivity is None:
        raise MissingFiducialSensitivityMap(
            "The delineation gate declares limited landmarks "
            f"({', '.join(landmarks)}) but no current fiducial sensitivity "
            f"map is available at {path}. Without it there is no way to tell "
            "which markers the limitation applies to, and continuing would "
            "grant every marker the claim the limitation exists to withhold. "
            "Re-run the validate-delineation stage, which writes the map "
            f"(expected map version {FIDUCIAL_DEPENDENCY_VERSION}"
            + (
                f", implementation hash {feature_implementation_hash}"
                if feature_implementation_hash
                else ""
            )
            + ")."
        )

    return FiducialLimitationPolicy(
        limited_landmarks=landmarks,
        demoted_markers=frozenset(sensitivity.markers_demoted_by(landmarks)),
        reliability_threshold=RELIABILITY_DEMOTION_THRESHOLD,
        map_records_measured=sensitivity.records_measured,
        map_implementation_hash=sensitivity.implementation_hash,
    )


def build_sensitivity_map(
    *,
    benchmark_database_root: Path,
    resources_root: Path,
    landmarks: tuple[str, ...],
    feature_implementation_hash: str,
    max_records: int = 24,
    progress=None,
) -> Optional[Path]:
    """Measure the sensitivity map on LUDB and write it beside the benchmark.

    Produced by the same stage that measures the gate, so the map can never
    describe a different delineator than the one just certified.  Returns None
    when no landmark is limited, because then there is nothing to apply.
    """

    if not landmarks:
        return None

    # Imported here: this is the one place the delineation benchmark and the
    # marker catalog meet, and a module-level import would tie every consumer
    # of the limitation policy to the benchmark's dependencies.
    from ..core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
        write_sensitivity_map,
    )
    from ..core.delineation.benchmark import (
        _load_ludb_signal,
        resolve_ludb_root,
    )
    from ..core.delineation.fusion_delineator import FusionDelineator
    from ..core.preprocessing import (
        detect_r_peaks_multilead,
        prepare_signal_paths,
    )
    from ..core.vcg_transform import VCGTransformRegistry

    root = resolve_ludb_root(Path(benchmark_database_root))
    names = [
        line.strip()
        for line in (root / "RECORDS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: int(max_records)]

    delineator = FusionDelineator()
    prepared = []
    for index, relative in enumerate(names, start=1):
        if progress is not None:
            progress(f"Fiducial-Sensitivitaet {index}/{len(names)}")
        try:
            signal, fs = _load_ludb_signal(root / relative)
            paths = prepare_signal_paths(signal, fs, powerline=50.0)
            detected = detect_r_peaks_multilead(paths["delineation"], fs)
            vcg = VCGTransformRegistry.transform(paths["delineation"], "kors")
            delineation = delineator.delineate_record(
                paths["delineation"],
                detected,
                fs,
                ecg_id=int(Path(relative).name),
                vcg_xyz=vcg.xyz,
            )
        except Exception:
            # One unreadable record must not cost the map; the measurement is
            # a union over records and simply sees one fewer.
            continue
        prepared.append((delineation, paths["delineation"], vcg.xyz, fs))

    if not prepared:
        return None

    sensitivity = measure_fiducial_sensitivity(
        prepared,
        landmarks=landmarks,
        implementation_hash=feature_implementation_hash,
        progress=progress,
    )
    return write_sensitivity_map(
        sensitivity_map_path(resources_root), sensitivity
    )


__all__ = [
    "FIDUCIAL_LIMITATION_VERSION",
    "SENSITIVITY_MAP_FILENAME",
    "FiducialLimitationPolicy",
    "MissingFiducialSensitivityMap",
    "build_sensitivity_map",
    "declared_limitation_landmarks",
    "load_limitation_policy",
    "sensitivity_map_path",
]
