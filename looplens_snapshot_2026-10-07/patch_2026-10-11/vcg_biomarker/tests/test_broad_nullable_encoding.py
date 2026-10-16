import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import broad_axis_adjustment as adjustment, broad_axis_stage as stage
from vcg_biomarker.tests.test_broad_axis_adjustment import _frame


def old_adjustment():
    spec=importlib.util.spec_from_file_location('vcg_biomarker.mimic._encoding_reference',
        Path(__file__).parent/'fixtures/broad_adjustment_before_preparation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind',['binary','continuous'])
def test_prepared_nullable_matches_original_numeric_boundary_and_default_unchanged(kind,monkeypatch):
    frame=_frame(n=800)
    frame['age']=frame.age.round().astype('Int64');frame.loc[17,'age']=pd.NA
    if kind=='continuous': frame['target_value']=np.random.default_rng(9).normal(size=len(frame))
    old=old_adjustment()
    expected_failure=old.fit_adjusted_association(frame,target_kind=kind)
    actual_default=adjustment.fit_adjusted_association(frame,target_kind=kind)
    assert actual_default['reason']==expected_failure['reason']=='fit_failed:ValueError'
    pd.testing.assert_frame_equal(pd.DataFrame([actual_default]),pd.DataFrame([expected_failure]),check_exact=True)
    import statsmodels.api as sm
    model_name='GLM' if kind=='binary' else 'OLS'
    original=getattr(sm,model_name)
    def numeric_boundary(target,exog,*args,**kwargs):
        return original(target,exog.astype(float),*args,**kwargs)
    monkeypatch.setattr(sm,model_name,numeric_boundary)
    expected=old.fit_adjusted_association(frame,target_kind=kind)
    monkeypatch.setattr(sm,model_name,original)
    actual=adjustment.fit_adjusted_association(frame,target_kind=kind,_prepared=True)
    assert expected['status']==actual['status']=='estimable'
    pd.testing.assert_frame_equal(pd.DataFrame([actual]),pd.DataFrame([expected]),check_exact=True)


def test_only_dtype_failures_recomputed_and_valid_results_preserved(monkeypatch):
    frame=_frame(n=800)
    frame['age']=frame.age.round().astype('Int64');frame.loc[17,'age']=pd.NA
    frame['evaluation_split']=np.where(frame.study_id<400,'train','validation')
    rng=np.random.default_rng(7)
    frame['rhythm_context']=rng.choice(['sinus','af','other'],len(frame))
    frame['qrs_duration_ms']=rng.normal(100,10,len(frame))
    kwargs=dict(axis='procedure_code',target_id='a',target_label='A',temporal_relation='undated',marker_id='m',target_kind='binary')
    legacy=pd.DataFrame(stage._fit_adjusted_candidate(frame,**kwargs))
    assert legacy.reason.eq('fit_failed:ValueError').sum()==4
    original=stage.fit_adjusted_association
    calls=[]
    def observed(*args,**kwargs):
        calls.append(kwargs['model_id']);return original(*args,**kwargs)
    monkeypatch.setattr(stage,'fit_adjusted_association',observed)
    connection=duckdb.connect();connection.register('input',frame.rename(columns={'marker_value':'__marker'}))
    actual=stage._compute_adjustment_packet(connection,query='SELECT * FROM input',block=['m'],aliases={'m':'__marker'},
        axis='procedure_code',target_id='a',label='A',relation='undated',target_kind='binary',old_frame=legacy)
    assert calls==['primary','rhythm_conduction_sensitivity']*2
    assert not actual.reason.eq('fit_failed:ValueError').any()
    assert actual.status.eq('estimable').all()
    pd.testing.assert_frame_equal(actual.loc[actual.model_id.eq('unadjusted_regression')].reset_index(drop=True),
        legacy.loc[legacy.model_id.eq('unadjusted_regression')].reset_index(drop=True),check_exact=True)
    connection.close()
