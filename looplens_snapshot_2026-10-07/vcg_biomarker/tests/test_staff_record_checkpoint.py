from types import SimpleNamespace
import numpy as np
from vcg_biomarker.core.delineation import benchmark_suite as m


def test_staff_checkpoint_reuses_completed_record_without_redelineation(tmp_path, monkeypatch):
    source = tmp_path / "record.dat"
    source.write_bytes(b"synthetic input")
    monkeypatch.setattr(m, "_record_names", lambda _: ["record"])
    monkeypatch.setattr(m.wfdb, "rdheader", lambda _: SimpleNamespace(fs=100, sig_len=30000))
    monkeypatch.setattr(m, "_read_event_samples", lambda _: [12000])
    monkeypatch.setattr(m, "_read_wfdb_signal_mv", lambda *args, **kwargs: (np.zeros((12000, 1)), 100, ("II",)))
    monkeypatch.setattr(m, "prepare_signal_paths", lambda signal, *args, **kwargs: {"delineation": signal})
    monkeypatch.setattr(m, "detect_r_peaks_multilead", lambda *args: np.array([10, 100]))
    calls = []
    def delineate(*args, **kwargs):
        calls.append(kwargs["ecg_id"])
        return SimpleNamespace(quality_score=.9, is_segment_evaluable=lambda _: True,
                               segment_confidence=lambda _: .8, review_required=False)
    monkeypatch.setattr(m, "FusionDelineator", lambda: SimpleNamespace(delineate_record=delineate))
    checkpoint = tmp_path / "checkpoints"
    first = m.run_staffiii_event_stability_benchmark(tmp_path, checkpoint_root=checkpoint)
    assert len(calls) == 2
    second = m.run_staffiii_event_stability_benchmark(tmp_path, checkpoint_root=checkpoint)
    assert first == second and len(calls) == 2
    source.write_bytes(b"changed synthetic input")
    third = m.run_staffiii_event_stability_benchmark(tmp_path, checkpoint_root=checkpoint)
    assert third == first and len(calls) == 4
