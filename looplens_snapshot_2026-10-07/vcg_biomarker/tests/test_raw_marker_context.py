from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext


@pytest.mark.parametrize("raise_error", [False, True])
def test_raw_context_is_explicit_and_cleared_even_on_failure(monkeypatch, raise_error):
    raw = np.ones((1500, 12))
    morphology = np.zeros_like(raw)
    delineation = SimpleNamespace(representative_beat=None)
    seen = []

    def compute(cls, biomarkers, delineation, ecg, vcg, fs):
        ctx = FeatureContext(delineation, ecg, vcg, fs)
        seen.append(ctx._record_raw_ecg_12lead)
        assert ctx._record_raw_ecg_12lead is raw
        assert ctx._record_ecg_12lead is morphology
        if raise_error:
            raise RuntimeError("synthetic failure")
        return []

    monkeypatch.setattr(BiomarkerRegistry, "_compute_biomarkers", classmethod(compute))
    kwargs = dict(delineation=delineation, ecg_12lead=morphology, names=[],
                  record_ecg_12lead=morphology, record_raw_ecg_12lead=raw)
    if raise_error:
        with pytest.raises(RuntimeError, match="synthetic"):
            BiomarkerRegistry.compute_all(**kwargs)
    else:
        BiomarkerRegistry.compute_all(**kwargs)
    assert len(seen) == 1
    assert FeatureContext(delineation, morphology, None, 500)._record_raw_ecg_12lead is None


def test_no_raw_fallback_from_morphology():
    ctx = FeatureContext(SimpleNamespace(representative_beat=None), np.ones((100, 12)), None, 500)
    assert ctx._record_raw_ecg_12lead is None
