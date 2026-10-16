import json
import pytest
from types import SimpleNamespace

from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import ndat_biomarkers as module
from vcg_biomarker.mimic.feature_roles import feature_profile
from vcg_biomarker.tests.test_ndat import record


def delineation(bounds):
    beats = [SimpleNamespace(r_onset=SimpleNamespace(sample=start),
             r_offset=SimpleNamespace(sample=end), segment_confidence=lambda _: 1.) for start, end in bounds]
    return SimpleNamespace(beats=beats, representative_beat=beats[0])


def test_registry_computes_once_and_preserves_discovery_roles(monkeypatch):
    signal, bounds = record()
    original = module.negative_downslope_times
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "negative_downslope_times", counted)
    results = BiomarkerRegistry.compute_all(delineation(bounds), signal, fs=500,
        names=list(module.NDAT_MARKER_IDS), record_raw_ecg_12lead=signal)
    assert len(calls) == 1
    assert len(results) == 7
    assert all(result.value is not None for result in results)
    assert json.loads(results[0].computation_notes)["pacing_status"] == "unknown"
    for name in module.NDAT_MARKER_IDS:
        profile = feature_profile(name)
        assert profile.modality == "ecg"
        assert profile.post_outcome_hypothesis
        assert not profile.core_sentinel_eligible
        assert not profile.combination_eligible


def test_registry_never_substitutes_filtered_signal_and_honors_known_pacing():
    signal, bounds = record()
    d = delineation(bounds)
    result = BiomarkerRegistry.compute_all(d, signal, names=[module.NDAT_MARKER_IDS[0]])[0]
    assert result.value is None
    assert "explicit_raw_record_required" in result.computation_notes
    d.pacing_present = True
    result = BiomarkerRegistry.compute_all(d, signal, names=[module.NDAT_MARKER_IDS[0]],
        record_raw_ecg_12lead=signal)[0]
    assert result.value is None
    assert "pacing_requires" in result.computation_notes


@pytest.mark.parametrize("metadata,reason", [
    ({"original_sampling_rate_hz": 100}, "insufficient_or_invalid_original"),
    ({"upper_bandwidth_hz": 40}, "insufficient_or_invalid_upper"),
    ({"pacing_present": True}, "pacing_requires"),
    ({"upper_bandwidth_hz": "bad"}, "invalid_upper"),
])
def test_acquisition_constraints_survive_upsampling(metadata, reason):
    signal, bounds = record()
    result = BiomarkerRegistry.compute_all(delineation(bounds), signal, fs=500,
        names=[module.NDAT_MARKER_IDS[0]], record_raw_ecg_12lead=signal,
        record_acquisition=metadata)[0]
    assert result.value is None
    assert reason in result.computation_notes


def test_global_quality_abstention_clears_order_and_explains_each_lead():
    signal, bounds = record()
    d = delineation(bounds)
    d.beats.extend([SimpleNamespace(r_onset=None, r_offset=None)] * 2)
    result = BiomarkerRegistry.compute_all(d, signal, names=[module.NDAT_MARKER_IDS[0]],
        record_raw_ecg_12lead=signal)[0]
    payload = json.loads(result.computation_notes)
    assert payload["order"] == []
    assert payload["n_qrs_quality_eligible"] == 3
    assert payload["n_delineated_beats"] == 5
    assert all("insufficient_quality_eligible_beats" in x["reasons"] for x in payload["leads"].values())
