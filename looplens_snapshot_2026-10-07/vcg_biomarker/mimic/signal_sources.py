"""Which signal each marker family is computed from, declared explicitly.

Four different signals reach the feature layer, and they are not
interchangeable:

* ``delineation`` (0.5-40 Hz) exists only to place fiducials;
* ``morphology`` (0.05-100 Hz) carries clinically relevant amplitude and
  loop geometry;
* ``wideband`` (0.05 - min(250 Hz, 0.45 fs)) is the only path that still
  contains the 80-250 Hz band;
* the fiducial-warped **median beat** built from ``morphology`` suppresses
  beat-to-beat jitter, which is what makes it robust for geometry and what
  makes it wrong for anything defined *by* that jitter.

The bug this module exists to prevent was concrete: high-frequency QRS energy
was analysed in 80-225 Hz on the 100 Hz morphology path. Measured on a real
MIMIC record, that path retains roughly a fifteenth of the 100-225 Hz energy
the wideband path carries, so the marker was reading a Butterworth roll-off.

Declaring the source per marker does not by itself make a marker valid. It
makes the choice visible and testable, and it lets the feature table state
what a value was computed from instead of leaving it to be inferred.
"""

from __future__ import annotations

import re
from typing import Mapping


SIGNAL_SOURCE_POLICY_VERSION = "1.0.0"

# Fiducial-warped median beat from the 0.05-100 Hz morphology path.
SOURCE_MEDIAN_BEAT = "median_beat_morphology_0p05_100hz"
# Full record on the same 0.05-100 Hz path (no beat averaging).
SOURCE_RECORD_MORPHOLOGY = "record_morphology_0p05_100hz"
# Full record, 0.05 Hz to min(250 Hz, 0.45 fs).
SOURCE_WIDEBAND = "record_wideband_0p05_to_250hz"
# Individual delineated beats, i.e. the jitter itself is the signal.
SOURCE_INDIVIDUAL_BEATS = "individual_delineated_beats"
# Interval measurements taken from the delineation path's fiducials.
SOURCE_FIDUCIAL_INTERVALS = "delineation_fiducials_0p5_40hz"

VALID_SIGNAL_SOURCES: frozenset[str] = frozenset(
    {
        SOURCE_MEDIAN_BEAT,
        SOURCE_RECORD_MORPHOLOGY,
        SOURCE_WIDEBAND,
        SOURCE_INDIVIDUAL_BEATS,
        SOURCE_FIDUCIAL_INTERVALS,
    }
)

# Markers whose analysis band lies above the morphology path's corner. These
# must abstain rather than silently fall back -- see catalog_biomarkers.
WIDEBAND_REQUIRED_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "qrs_fragmentation_high_frequency_energy",
    }
)

# Markers defined by beat-to-beat differences. A median beat has no
# beat-to-beat variation by construction, so computing these from it would
# return a constant.
_INDIVIDUAL_BEAT_PATTERN = re.compile(
    r"alternans|variability|beat_to_beat|^rr_|_rr_|dispersion_beat|"
    r"irregularity|poincare|template_instability",
    re.IGNORECASE,
)

# Interval and duration markers read fiducial positions, not waveform shape.
_INTERVAL_PATTERN = re.compile(
    r"_ms$|_interval|duration|qtc|jtc|tpeak_tend|pr_prolong|^heart_rate",
    re.IGNORECASE,
)

# Markers explicitly documented as whole-record rather than single-beat.
_RECORD_PATTERN = re.compile(
    r"^global_|rhythm_|_coverage$|breathing_|baseline_wander",
    re.IGNORECASE,
)


def signal_source_for_feature(feature_id: object) -> str:
    """Declare which signal a marker is computed from.

    Classification is by explicit set first, then by name pattern, then the
    median-beat default. It describes the current implementation; it is not a
    switch that changes it. ``assert_signal_source_policy_consistent`` keeps
    the two from drifting apart.
    """

    name = str(feature_id or "")
    if ("direct_8lead_beat_" in name or name.startswith("postectopic_direct8_")
            or (name.startswith("direct8_") and "_long_" in name)):
        return SOURCE_INDIVIDUAL_BEATS
    if name in WIDEBAND_REQUIRED_FEATURE_IDS:
        return SOURCE_WIDEBAND
    if _INDIVIDUAL_BEAT_PATTERN.search(name):
        return SOURCE_INDIVIDUAL_BEATS
    if _RECORD_PATTERN.search(name):
        return SOURCE_RECORD_MORPHOLOGY
    if _INTERVAL_PATTERN.search(name):
        return SOURCE_FIDUCIAL_INTERVALS
    return SOURCE_MEDIAN_BEAT


def signal_source_caveat(source: str) -> str:
    """The interpretation limit that comes with each source."""

    if source == SOURCE_MEDIAN_BEAT:
        return (
            "Fiducial-gewarpter Medianschlag: unterdrueckt Schlag-zu-Schlag-"
            "Jitter. Geschwindigkeits-, Beschleunigungs- und Jerk-Betraege "
            "sind dadurch systematisch gedaempft und nicht mit "
            "Einzelschlagwerten vergleichbar."
        )
    if source == SOURCE_WIDEBAND:
        return (
            "Breitbandpfad bis min(250 Hz, 0,45 fs): erforderlich fuer "
            "Hochfrequenzbaender, enthaelt aber auch mehr Muskel- und "
            "Geraeterauschen als der Morphologiepfad."
        )
    if source == SOURCE_INDIVIDUAL_BEATS:
        return (
            "Einzelschlaege: die Variabilitaet selbst ist das Signal, "
            "entsprechend empfindlich gegenueber Fehldelineation einzelner "
            "Schlaege."
        )
    if source == SOURCE_RECORD_MORPHOLOGY:
        return (
            "Gesamtaufzeichnung ohne Schlagmittelung, 0,05-100 Hz."
        )
    if source == SOURCE_FIDUCIAL_INTERVALS:
        return (
            "Fiducial-Positionen des Delineationspfads (0,5-40 Hz); traegt "
            "dessen Grenzfehler, siehe delineation_sensitivity."
        )
    return ""


def signal_source_table(feature_ids: object) -> list[dict[str, str]]:
    """Per-feature source declaration for the audit and manifest layers."""

    rows: list[dict[str, str]] = []
    for feature_id in feature_ids:
        source = signal_source_for_feature(feature_id)
        rows.append(
            {
                "feature_id": str(feature_id),
                "signal_source": source,
                "signal_source_caveat": signal_source_caveat(source),
                "wideband_required": str(
                    str(feature_id) in WIDEBAND_REQUIRED_FEATURE_IDS
                ),
                "policy_version": SIGNAL_SOURCE_POLICY_VERSION,
            }
        )
    return rows


def assert_signal_source_policy_consistent(
    implementation_sources: Mapping[str, str],
) -> None:
    """Fail when the declared source and the implemented one disagree.

    ``implementation_sources`` maps feature_id to the source its code actually
    reads. Keeping this check in the test suite is what stops the declaration
    from decaying into documentation nobody verifies.
    """

    mismatches = [
        (feature_id, declared, actual)
        for feature_id, actual in implementation_sources.items()
        if (declared := signal_source_for_feature(feature_id)) != actual
    ]
    if mismatches:
        rendered = "; ".join(
            f"{feature_id}: declared={declared} implemented={actual}"
            for feature_id, declared, actual in mismatches
        )
        raise AssertionError(
            f"Signal-source policy disagrees with the implementation: {rendered}"
        )


__all__ = [
    "SIGNAL_SOURCE_POLICY_VERSION",
    "SOURCE_FIDUCIAL_INTERVALS",
    "SOURCE_INDIVIDUAL_BEATS",
    "SOURCE_MEDIAN_BEAT",
    "SOURCE_RECORD_MORPHOLOGY",
    "SOURCE_WIDEBAND",
    "VALID_SIGNAL_SOURCES",
    "WIDEBAND_REQUIRED_FEATURE_IDS",
    "assert_signal_source_policy_consistent",
    "signal_source_caveat",
    "signal_source_for_feature",
    "signal_source_table",
]
