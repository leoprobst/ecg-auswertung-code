import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from vcg_biomarker.mimic import phenome


def reference(name):
    path=Path(__file__).parent/'fixtures/phenome_bound_overlay_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='2adefcd8193f715054cdfd1bf82ea5a372c46b27797ef99988a26769fedb1b9c'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(phenome.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


@pytest.mark.parametrize('empty',[False,True])
def test_overlay_is_exactly_equal_to_bound_implementation(empty):
    validation=pd.DataFrame(dict(phenotype_id=['p']*5,feature_id=list('abcde'),panel_id=['x']*5,
        replicated=[True,False,True,False,True],note=pd.Series(['a',None,'c','d','e'],dtype='string')))
    for name in ('auc','effect','ppv','npv'):
        validation[name+'_ci_low']=np.nan;validation[name+'_ci_high']=np.nan
    bootstrap=validation.iloc[[4,0,2]].copy()
    for name in ('auc','effect','ppv','npv'):
        bootstrap[name+'_ci_low']=[.1,.2,.3];bootstrap[name+'_ci_high']=[.5,.6,.7]
    bootstrap['bootstrap_iterations']=pd.Series([1000,1000,1000],index=bootstrap.index,dtype='Int64')
    if empty:bootstrap=bootstrap.iloc[:0]
    expected=reference('_overlay_replicated_bootstraps')(validation,bootstrap)
    actual=phenome._overlay_replicated_bootstraps(validation,bootstrap)
    assert_frame_equal(actual,expected,check_exact=True)
    assert actual.feature_id.tolist()==list('abcde')


def test_completed_adjustment_blocks_skip_raw_loading_and_model_fits(tmp_path,monkeypatch):
    candidates=pd.DataFrame(dict(phenotype_id=['p2','p1'],feature_id=['f2','f1'],panel_id=['x','x']))
    monkeypatch.setattr(phenome,'_feature_blocks',lambda ids:tuple((i,) for i in ids))
    monkeypatch.setattr(phenome,'_load_phenome_feature_block',lambda *a,**k:pd.DataFrame())
    monkeypatch.setattr(phenome,'_combine_phenome_feature_block',lambda *a,**k:pd.DataFrame())
    monkeypatch.setattr(phenome,'confirm_replicated_phenome_hits',lambda selected,*a,**k:selected.assign(effect=.25))
    monkeypatch.setattr(phenome,'finalize_phenome_adjustment_output',lambda frame,policy:frame)
    args=dict(feature_glob='unused',index_path=tmp_path/'index',candidates=candidates,
        feature_index=pd.DataFrame(dict(split=['validation'])),current=pd.DataFrame(),history_body=pd.DataFrame(),
        policy=None,checkpoint_root=tmp_path/'checkpoints',checkpoint_input_hash='same',progress=None)
    expected=reference('_checkpointed_adjusted_confirmation_streaming')(None,**args)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in args['checkpoint_root'].rglob('part-*')}
    def forbidden(*a,**k):raise AssertionError('Completed adjustment loaded or recomputed')
    monkeypatch.setattr(phenome,'_load_phenome_feature_block',forbidden)
    monkeypatch.setattr(phenome,'confirm_replicated_phenome_hits',forbidden)
    actual=phenome._checkpointed_adjusted_confirmation_streaming(None,**args)
    assert_frame_equal(actual,expected,check_exact=True)
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in args['checkpoint_root'].rglob('part-*')}
    next(args['checkpoint_root'].rglob('part-*.parquet')).write_bytes(b'damaged')
    with pytest.raises(RuntimeError):phenome._checkpointed_adjusted_confirmation_streaming(None,**args)
