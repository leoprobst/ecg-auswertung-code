from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import population_cohorts as population


@pytest.mark.parametrize("sex", [pd.NA, None, np.nan, "M", "F"])
def test_original_rate_and_unknown_sex_reach_consumers(monkeypatch, sex):
    signal = np.zeros((4000, 12))
    record = SimpleNamespace(exam_id=1, patient_id=2, source="code15",
        original_sampling_rate_hz=400., sampling_rate_hz=500.,
        trimmed_samples=3200, padded_samples=0, signal_millivolts=signal)
    quality = SimpleNamespace(usable=True, score=1., to_dict=lambda: {})
    delineation = SimpleNamespace(beats=[object()], n_beats=5, quality_score=1.)
    monkeypatch.setattr(population, "assess_signal_quality", lambda *a, **kw: quality)
    monkeypatch.setattr(population, "prepare_signal_paths", lambda *a, **kw:
        {name: signal for name in ("delineation", "morphology", "diagnostic")})
    monkeypatch.setattr(population, "detect_r_peaks_multilead", lambda *a: np.arange(500, 3000, 500))
    monkeypatch.setattr(population, "_thread_delineator", lambda: SimpleNamespace(
        delineate_record=lambda **kw: delineation))
    monkeypatch.setattr(population, "construct_robust_morphology_record", lambda *a: (signal, {}))
    transformed = SimpleNamespace(xyz=signal[:, :3])
    monkeypatch.setattr(population.VCGTransformRegistry, "transform", lambda *a: transformed)
    monkeypatch.setattr(population, "compute_vcg_sensitivity_transforms", lambda *a, **kw:
        {"inverse_dower": transformed, "pca_kors_aligned": transformed})
    calls, sexes = [], []
    def markers(**kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(population.BiomarkerRegistry, "compute_all", markers)
    def classical(*args, **kwargs):
        sexes.append(kwargs["sex"])
        return {}
    monkeypatch.setattr(population, "ClassicalECGCriteria", lambda: SimpleNamespace(compute=classical))
    result = population._analyze_record(record, {"age": 55, "sex": sex}, (), (),
        powerline_hz=60., minimum_signal_quality=0.)
    assert result["extraction_status"] == "ok"
    assert len(calls) == 4  # main, Kors fallback, two sensitivity transforms
    assert all(call["fs"] == 500 for call in calls)
    assert all(call["record_acquisition"]["original_sampling_rate_hz"] == 400 for call in calls)
    assert sexes == ([sex] if pd.notna(sex) else [""])
    assert result["original_sampling_rate_hz"] == 400
