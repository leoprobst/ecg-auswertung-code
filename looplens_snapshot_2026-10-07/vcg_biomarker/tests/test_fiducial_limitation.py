"""A declared limitation has to cost something, or it is a disabled test.

The delineation gate can declare a landmark limited -- measured, and known to
miss the external accuracy standard at every operating point. These tests pin
the two halves that turn that declaration into a consequence:

* the sensitivity map, which decides *which* markers are affected by measuring
  what the landmark's own error does to them, rather than by asserting a list;
* the join, which refuses to guess when the map is missing.

The measurement matters more than it looks. Plain ablation -- delete the
landmark, see what moves -- made 1557 of 1851 markers depend on the P onset,
because the P onset defines the left edge of the whole beat window. A rule
that demotes 84 % of the catalog separates nothing. Perturbing by the measured
error instead and comparing against between-subject spread demotes 11.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.core.biomarkers.fiducial_dependency import (
    MEASURED_LANDMARK_ERROR_MS,
    RELIABILITY_DEMOTION_THRESHOLD,
    FiducialSensitivityMap,
    LandmarkSensitivity,
    perturb_landmark,
    read_sensitivity_map,
    write_sensitivity_map,
)
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)
from vcg_biomarker.mimic.fiducial_limitation import (
    MissingFiducialSensitivityMap,
    declared_limitation_landmarks,
    load_limitation_policy,
    sensitivity_map_path,
)


def _sensitivity(marker, landmark, reliability):
    return LandmarkSensitivity(
        marker=marker,
        landmark=landmark,
        error_sigma=1.0,
        between_sigma=1.0,
        reliability=reliability,
        n_records=24,
    )


def _map(**markers):
    return FiducialSensitivityMap(
        sensitivities={
            marker: {"p_onset": _sensitivity(marker, "p_onset", reliability)}
            for marker, reliability in markers.items()
        },
        landmarks_measured=("p_onset",),
        records_measured=24,
        implementation_hash="hash-a",
    )


# --------------------------------------------------------------- measurement


def test_perturbation_moves_the_landmark_and_leaves_the_rest_alone():
    beat = BeatDelineation(
        p_onset=MarkerPoint(sample=50, confidence=1.0, source="test"),
        r_peak=MarkerPoint(sample=110, confidence=1.0, source="test"),
    )
    delineation = RecordDelineation(
        beats=[beat], fs=500, n_samples=300, ecg_id=1
    )
    shifted = perturb_landmark(delineation, "p_onset", 13)
    assert shifted.beats[0].p_onset.sample == 63
    assert shifted.beats[0].r_peak.sample == 110
    # The caller's delineation is also the baseline and must be untouched.
    assert delineation.beats[0].p_onset.sample == 50


def test_perturbation_stays_inside_the_record():
    beat = BeatDelineation(
        p_onset=MarkerPoint(sample=2, confidence=1.0, source="test"),
        r_peak=MarkerPoint(sample=110, confidence=1.0, source="test"),
    )
    delineation = RecordDelineation(
        beats=[beat], fs=500, n_samples=300, ecg_id=1
    )
    assert perturb_landmark(delineation, "p_onset", -50).beats[0].p_onset.sample == 0
    assert perturb_landmark(delineation, "p_onset", 10_000).beats[0].p_onset.sample == 299


def test_the_perturbation_amplitude_is_the_measured_error():
    """Not a round number someone picked."""

    assert MEASURED_LANDMARK_ERROR_MS["p_onset"] == pytest.approx(26.3)
    assert (
        MEASURED_LANDMARK_ERROR_MS["p_offset"]
        < MEASURED_LANDMARK_ERROR_MS["p_onset"]
    )


def test_demotion_follows_the_conventional_reliability_floor():
    assert RELIABILITY_DEMOTION_THRESHOLD == pytest.approx(0.70)
    assert _sensitivity("m", "p_onset", 0.69).demoted
    assert not _sensitivity("m", "p_onset", 0.71).demoted


def test_only_markers_below_the_floor_are_demoted():
    sensitivity_map = _map(p_duration_ms=0.653, pr_interval_ms=0.843)
    assert sensitivity_map.markers_demoted_by(["p_onset"]) == {"p_duration_ms"}


def test_a_landmark_that_was_not_declared_demotes_nothing():
    sensitivity_map = _map(p_duration_ms=0.10)
    assert sensitivity_map.markers_demoted_by(["t_offset"]) == set()


# ------------------------------------------------------------- serialisation


def test_a_map_survives_a_round_trip(tmp_path):
    original = _map(p_duration_ms=0.653, pr_interval_ms=0.843)
    path = write_sensitivity_map(tmp_path / "sens.json", original)
    restored = read_sensitivity_map(path, expected_implementation_hash="hash-a")
    assert restored is not None
    assert restored.markers_demoted_by(["p_onset"]) == {"p_duration_ms"}


def test_a_map_from_other_marker_code_is_rejected(tmp_path):
    """The reuse-key mistake, in the one place it would silently mislabel."""

    path = write_sensitivity_map(tmp_path / "sens.json", _map(m=0.1))
    assert read_sensitivity_map(path, expected_implementation_hash="hash-b") is None


def test_a_map_from_an_older_format_is_rejected(tmp_path):
    path = tmp_path / "sens.json"
    write_sensitivity_map(path, _map(m=0.1))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = "0.0.1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_sensitivity_map(path) is None


def test_a_truncated_map_is_rejected_rather_than_fatal(tmp_path):
    path = tmp_path / "sens.json"
    path.write_text('{"version": "2.0.0", "sensi', encoding="utf-8")
    assert read_sensitivity_map(path) is None


# ---------------------------------------------------------------- the join


def _write_gate(tmp_path, landmarks):
    path = tmp_path / "benchmark_suite.json"
    path.write_text(
        json.dumps(
            {
                "release_gate": {
                    "passed": True,
                    "affected_landmarks": list(landmarks),
                    "declared_limitations": [
                        {"criterion": "ludb:p_onset_p95_ms", "landmark": item}
                        for item in landmarks
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_landmarks_come_from_the_recorded_gate_not_the_source_tree(tmp_path):
    path = _write_gate(tmp_path, ["p_onset"])
    assert declared_limitation_landmarks(path) == ("p_onset",)


def test_a_missing_gate_file_declares_nothing(tmp_path):
    assert declared_limitation_landmarks(tmp_path / "absent.json") == ()


def test_no_declaration_means_no_map_is_required(tmp_path):
    policy = load_limitation_policy(
        resources_root=tmp_path,
        benchmark_suite_path=_write_gate(tmp_path, []),
        feature_implementation_hash="hash-a",
    )
    assert policy.demoted_markers == frozenset()
    assert policy.marker_may_be_confirmed("anything")


def test_a_declared_limitation_without_a_map_raises(tmp_path):
    """Never silently grant the claim the limitation exists to withhold."""

    with pytest.raises(MissingFiducialSensitivityMap) as excinfo:
        load_limitation_policy(
            resources_root=tmp_path,
            benchmark_suite_path=_write_gate(tmp_path, ["p_onset"]),
            feature_implementation_hash="hash-a",
        )
    message = str(excinfo.value)
    assert "p_onset" in message
    assert "validate-delineation" in message


def test_a_stale_map_raises_rather_than_mislabelling(tmp_path):
    write_sensitivity_map(sensitivity_map_path(tmp_path), _map(m=0.1))
    with pytest.raises(MissingFiducialSensitivityMap):
        load_limitation_policy(
            resources_root=tmp_path,
            benchmark_suite_path=_write_gate(tmp_path, ["p_onset"]),
            feature_implementation_hash="a-different-hash",
        )


def test_the_policy_demotes_exactly_the_measured_markers(tmp_path):
    write_sensitivity_map(
        sensitivity_map_path(tmp_path),
        _map(p_duration_ms=0.653, pr_interval_ms=0.843),
    )
    policy = load_limitation_policy(
        resources_root=tmp_path,
        benchmark_suite_path=_write_gate(tmp_path, ["p_onset"]),
        feature_implementation_hash="hash-a",
    )
    assert policy.demoted_markers == {"p_duration_ms"}
    assert not policy.marker_may_be_confirmed("p_duration_ms")
    assert policy.marker_may_be_confirmed("pr_interval_ms")
    assert policy.to_dict()["demoted_marker_count"] == 1


# ------------------------------------------------------------- enforcement


def _candidate_frames(feature_id="p_duration_ms"):
    import pandas as pd

    candidates = pd.DataFrame(
        [
            {
                "endpoint_id": "e",
                "panel_id": "p",
                "feature_id": feature_id,
                "core_replication_evidence_passed": True,
                "stage": "TRAIN_STABLE",
            }
        ]
    )
    screening = pd.DataFrame(
        [
            {
                "endpoint_id": "e",
                "panel_id": "p",
                "feature_id": feature_id,
                "evaluation_split": split,
                "global_fdr_significant": True,
                "hierarchical_fdr_significant": True,
            }
            for split in ("train", "validation")
        ]
    )
    return candidates, screening


def test_a_demoted_marker_is_flagged_in_the_results():
    """The flag has to be visible, not only effective."""

    from vcg_biomarker.mimic.criteria import finalize_criterion_catalog_evidence
    from vcg_biomarker.mimic.fiducial_limitation import FiducialLimitationPolicy

    candidates, screening = _candidate_frames()
    policy = FiducialLimitationPolicy(
        limited_landmarks=("p_onset",),
        demoted_markers=frozenset({"p_duration_ms"}),
        reliability_threshold=0.70,
        map_records_measured=24,
        map_implementation_hash="hash-a",
    )
    row = finalize_criterion_catalog_evidence(
        candidates, screening, fiducial_limitation=policy
    ).iloc[0]
    assert bool(row["rests_on_limited_landmark"])
    assert not bool(row["fiducial_reliability_sufficient"])


def test_an_unaffected_marker_keeps_its_claim():
    from vcg_biomarker.mimic.criteria import finalize_criterion_catalog_evidence
    from vcg_biomarker.mimic.fiducial_limitation import FiducialLimitationPolicy

    candidates, screening = _candidate_frames("pr_interval_ms")
    policy = FiducialLimitationPolicy(
        limited_landmarks=("p_onset",),
        demoted_markers=frozenset({"p_duration_ms"}),
        reliability_threshold=0.70,
        map_records_measured=24,
        map_implementation_hash="hash-a",
    )
    row = finalize_criterion_catalog_evidence(
        candidates, screening, fiducial_limitation=policy
    ).iloc[0]
    assert not bool(row["rests_on_limited_landmark"])
    assert bool(row["fiducial_reliability_sufficient"])


def test_without_a_policy_nothing_is_demoted():
    """Endpoints whose gate declared no limitation must be unaffected."""

    from vcg_biomarker.mimic.criteria import finalize_criterion_catalog_evidence

    candidates, screening = _candidate_frames()
    row = finalize_criterion_catalog_evidence(
        candidates, screening, fiducial_limitation=None
    ).iloc[0]
    assert not bool(row["rests_on_limited_landmark"])


# ------------------------------------------------------- the measurement itself


class _StubResult:
    def __init__(self, name, value):
        self.name = name
        self.value = value


def _stub_registry(monkeypatch, response):
    """Replace the catalog with a marker whose value we control.

    ``response`` maps the P onset's sample position to a marker value, so a
    test can state exactly how strongly a marker reacts to the landmark.
    """

    from vcg_biomarker.core.biomarkers import fiducial_dependency

    def compute_all(delineation, **kwargs):
        sample = delineation.beats[0].p_onset.sample
        return [
            _StubResult(name, values(sample))
            for name, values in response.items()
        ]

    monkeypatch.setattr(
        fiducial_dependency.BiomarkerRegistry, "compute_all", compute_all
    )


def _records(baseline_samples):
    """One record per baseline P onset position, so a spread exists."""

    records = []
    for index, sample in enumerate(baseline_samples):
        beat = BeatDelineation(
            p_onset=MarkerPoint(sample=sample, confidence=1.0, source="test"),
            r_peak=MarkerPoint(sample=200, confidence=1.0, source="test"),
        )
        records.append(
            (
                RecordDelineation(
                    beats=[beat], fs=500, n_samples=600, ecg_id=index
                ),
                None,
                None,
                500,
            )
        )
    return records


def test_a_marker_ignoring_the_landmark_stays_fully_reliable(monkeypatch):
    from vcg_biomarker.core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
    )

    _stub_registry(monkeypatch, {"constant_marker": lambda sample: 42.0})
    result = measure_fiducial_sensitivity(
        _records([100, 120, 140, 160]), landmarks=("p_onset",)
    )
    # No movement under perturbation, so nothing can be destroyed.
    assert result.sensitivities["constant_marker"]["p_onset"].reliability == 1.0
    assert result.markers_demoted_by(["p_onset"]) == set()


def test_a_marker_tracking_the_landmark_one_to_one_is_demoted(monkeypatch):
    """Movement equal to the between-record spread must fail the floor."""

    from vcg_biomarker.core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
    )

    # Value follows the P onset exactly, and the records sit closer together
    # than the 26.3 ms (13 sample) perturbation displaces them.
    _stub_registry(monkeypatch, {"follows": lambda sample: float(sample)})
    result = measure_fiducial_sensitivity(
        _records([100, 105, 110, 115]), landmarks=("p_onset",)
    )
    sensitivity = result.sensitivities["follows"]["p_onset"]
    assert sensitivity.error_sigma > sensitivity.between_sigma
    assert sensitivity.demoted


def test_a_widely_varying_marker_survives_the_same_movement(monkeypatch):
    """The discriminator is the ratio, not the absolute error."""

    from vcg_biomarker.core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
    )

    _stub_registry(monkeypatch, {"follows": lambda sample: float(sample)})
    # Same absolute movement, but the records now span 300 samples.
    result = measure_fiducial_sensitivity(
        _records([100, 200, 300, 400]), landmarks=("p_onset",)
    )
    sensitivity = result.sensitivities["follows"]["p_onset"]
    assert sensitivity.between_sigma > sensitivity.error_sigma
    assert not sensitivity.demoted


def test_the_landmark_is_restored_between_records(monkeypatch):
    """A perturbation must not leak into the next record's baseline."""

    from vcg_biomarker.core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
    )

    seen = []

    from vcg_biomarker.core.biomarkers import fiducial_dependency

    def compute_all(delineation, **kwargs):
        seen.append(delineation.beats[0].p_onset.sample)
        return [_StubResult("m", 1.0)]

    monkeypatch.setattr(
        fiducial_dependency.BiomarkerRegistry, "compute_all", compute_all
    )
    measure_fiducial_sensitivity(_records([100, 300]), landmarks=("p_onset",))
    # Baselines are the untouched positions; each appears exactly once.
    assert seen[0] == 100
    assert 300 in seen


def test_a_landmark_without_a_measured_error_is_skipped(monkeypatch):
    """Perturbing by a made-up amplitude would invent the answer."""

    from vcg_biomarker.core.biomarkers.fiducial_dependency import (
        measure_fiducial_sensitivity,
    )

    _stub_registry(monkeypatch, {"m": lambda sample: float(sample)})
    result = measure_fiducial_sensitivity(
        _records([100, 200]), landmarks=("j_point",)
    )
    assert result.sensitivities == {}
