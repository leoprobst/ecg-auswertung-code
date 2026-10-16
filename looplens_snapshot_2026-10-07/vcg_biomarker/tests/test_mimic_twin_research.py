import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.simulation.mimic_twin_research import eligible_pair, fit_delta, evaluate_pairs


def test_exact_distinct_past_measurements_and_clock_stress():
    frame = pd.DataFrame({'value0': [4., 4., 4., 4.], 'value1': [5., 5., 5., 5.],
        'unit0': ['mEq/L']*4, 'unit1': ['mmol/L']*4, 'censor0': [None]*4,
        'censor1': [None, 'above_limit', None, None], 'delta0': [-.5]*4,
        'delta1': [-.5, -.5, .2, -.5], 'event0': [1]*4, 'event1': [2, 2, 2, 1],
        'match0': ['same_admission']*4, 'match1': ['same_admission']*4})
    difference, mask = eligible_pair(frame, 'potassium')
    assert difference.tolist() == [1.]*4
    assert mask.tolist() == [True, False, False, False]
    assert not eligible_pair(frame, 'potassium', 6)[1].any()


def test_known_delta_effect_and_abstention():
    rng = np.random.default_rng(42)
    x, covariates = rng.normal(size=1200), rng.normal(size=(1200, 4))
    result = fit_delta(x, 3*x+covariates.sum(1)+rng.normal(size=1200)*.1,
                       covariates, np.ones(1200, bool))
    assert result['effect_ms_per_lab_unit'] == pytest.approx(3, abs=.02)
    assert result['ci95'][0] < 3 < result['ci95'][1]
    assert fit_delta(x, x, np.zeros((1200, 4)), np.ones(1200, bool))['status'] == 'non_estimable'
    with pytest.raises(ValueError, match='train'):
        evaluate_pairs(pd.DataFrame({'subject_id': [1], 'source_split': ['test']}), pd.DataFrame())
