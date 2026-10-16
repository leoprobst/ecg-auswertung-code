import numpy as np
import pytest
from types import SimpleNamespace

from vcg_biomarker.mimic.biomarker_audit import _synthetic_record
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _literature_feature


def test_variable_qt_audit_reference_has_real_paired_variation():
    ecg, vcg, delineation = _synthetic_record(fs=500, irregular_rr=True, variable_qt=True)
    _, baseline, _ = _synthetic_record(fs=500, irregular_rr=True)
    assert np.isfinite(ecg).all() and np.isfinite(vcg).all()
    assert not np.array_equal(vcg, baseline)
    pairs = delineation.qt_rr_measurements()
    assert len(pairs) >= 4
    qt = np.array([p.qt_ms / 1000 for p in pairs])
    rr = np.array([p.preceding_rr_seconds for p in pairs])
    assert np.var(qt) > 0 and np.var(rr) > 0
    context = SimpleNamespace(delineation=delineation)
    for name, expected in (
        ("qt_rr_hysteresis_surrogate", np.corrcoef(qt, rr)[0, 1]),
        ("qt_variability_index_vector", np.log10((np.var(qt)/np.mean(qt)**2) /
                                               (np.var(rr)/np.mean(rr)**2))),
    ):
        result = _literature_feature(context, name)
        assert result.value == pytest.approx(expected)
