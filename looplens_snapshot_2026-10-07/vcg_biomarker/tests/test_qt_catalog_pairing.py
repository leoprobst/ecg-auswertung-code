from types import SimpleNamespace as NS
import numpy as np
import pytest
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _literature_feature


@pytest.mark.parametrize("name", ["qt_rr_hysteresis_surrogate", "qt_variability_index_vector"])
def test_catalog_uses_explicit_pairs_and_abstains_for_constant_rr(name):
    qt = np.array([300., 330., 310., 350.])
    rr = np.array([.8, 1., .9, 1.1])
    pairs = [NS(qt_ms=q, preceding_rr_seconds=r, confidence=1.) for q, r in zip(qt, rr)]
    ctx = NS(delineation=NS(qt_rr_measurements=lambda: pairs))
    expected = (np.corrcoef(qt, rr)[0, 1] if "hysteresis" in name else
                np.log10((np.var(qt)/np.mean(qt)**2)/(np.var(rr)/np.mean(rr)**2)))
    assert _literature_feature(ctx, name).value == pytest.approx(expected)
    for item in pairs:
        item.preceding_rr_seconds = 1.
    assert _literature_feature(ctx, name).value is None
