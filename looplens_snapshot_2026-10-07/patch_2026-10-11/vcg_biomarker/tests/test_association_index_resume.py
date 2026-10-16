import json
from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vcg_biomarker.mimic import association_domains as module
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.tests.test_performance_hardening import _config


def inputs(tmp_path):
    config = replace(_config(tmp_path, workers=2), duckdb_memory_limit_gb=1)
    context = tmp_path/'context.parquet'; features=tmp_path/'features.parquet'
    frame=pd.DataFrame(dict(subject_id=[1,1,2,3], study_id=[10,11,20,30],
        split=['train','train','validation','test'], ecg_time=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-01','2020-01-01']),
        value=[2.,3.,4.,5.]))
    frame.to_parquet(context,index=False)
    frame.assign(extraction_status='ok').to_parquet(features,index=False)
    definitions=(module.ContextAssociationDefinition('a','A','laboratory','continuous','c.value','TRUE','current','first'),)
    args=dict(feature_glob=str(features),context_path=context,definitions=definitions,output_path=tmp_path/'index.parquet',
              checkpoint_root=tmp_path/'checkpoints',checkpoint_input_hash='same')
    return config,args


def test_index_resume_is_exact_and_does_not_repeat_join(tmp_path,monkeypatch):
    config,args=inputs(tmp_path)
    reference=tmp_path/'reference.parquet'
    module._build_association_index(config,**{k:v for k,v in args.items() if k not in ('checkpoint_root','checkpoint_input_hash','output_path')},output_path=reference)
    messages=[];module._ensure_association_index(config,**args,progress=messages.append)
    assert_frame_equal(pd.read_parquet(args['output_path']),pd.read_parquet(reference))
    assert pd.read_parquet(reference).subject_id.tolist()==[1,2]
    def forbidden(*a,**k):raise AssertionError('Repeated JOIN')
    monkeypatch.setattr(module,'_build_association_index',forbidden)
    module._ensure_association_index(config,**args,progress=messages.append)
    assert 'kein erneuter JOIN' in messages[-1]


@pytest.mark.parametrize('damage',['data','receipt','missing'])
def test_damaged_index_is_rejected_without_overwriting_it(tmp_path,monkeypatch,damage):
    config,args=inputs(tmp_path);module._ensure_association_index(config,**args)
    if damage=='data':args['output_path'].write_bytes(b'damaged')
    elif damage=='missing':args['output_path'].unlink()
    else:
        receipt=args['checkpoint_root']/'association_index_v1.json'
        value=json.loads(receipt.read_text());value['key']='changed';receipt.write_text(json.dumps(value))
    def forbidden(*a,**k):raise AssertionError('Damaged output silently rebuilt')
    monkeypatch.setattr(module,'_build_association_index',forbidden)
    with pytest.raises((RuntimeError,OSError)):module._ensure_association_index(config,**args)


def test_changed_input_rebuilds_index(tmp_path,monkeypatch):
    config,args=inputs(tmp_path);module._ensure_association_index(config,**args)
    original=module._build_association_index;calls=[]
    def record(config,**kwargs):calls.append(config.workers);return original(config,**kwargs)
    monkeypatch.setattr(module,'_build_association_index',record)
    module._ensure_association_index(config,**dict(args,checkpoint_input_hash='new'))
    assert calls==[1]


def test_definitions_and_context_are_bound_to_index_key(tmp_path):
    _,args=inputs(tmp_path);kwargs={k:args[k] for k in ('feature_glob','context_path','definitions','checkpoint_input_hash')}
    before=module._association_index_key(**kwargs)
    assert module._association_index_key(**dict(kwargs,definitions=(replace(args['definitions'][0],value_sql='c.value*2'),)))!=before
    frame=pd.read_parquet(args['context_path']);frame.loc[0,'value']=99;frame.to_parquet(args['context_path'],index=False)
    assert module._association_index_key(**kwargs)!=before
