"""Transient binary batches preserve old science and outer resume identities."""
import ast
import hashlib
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import broad_axis_stage as current

FIXTURE = Path(__file__).parent / 'fixtures/broad_axis_before_binary_memory.py'

def pinned():
    module = types.ModuleType('vcg_biomarker.mimic._pinned_binary_memory')
    module.__package__ = 'vcg_biomarker.mimic'
    module.__file__ = str(FIXTURE)
    exec(compile(FIXTURE.read_text(encoding='utf-8'),str(FIXTURE),'exec'),module.__dict__)
    return module

def test_science_and_outer_keys_unchanged():
    old = ast.parse(FIXTURE.read_text(encoding='utf-8'))
    new = ast.parse(Path(current.__file__).read_text(encoding='utf-8'))
    for name in ['_screen_binary','_screen_continuous','_marker_blocks','_axis_block_key','_axis_block_path','_load_axis_block','_save_axis_block']:
        a = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b), name

def test_default_one_batch_and_capped_materialization(monkeypatch):
    markers = tuple(f'm{i}' for i in range(2000))
    monkeypatch.delenv('VCG_JOB_MEMORY_GB',raising=False)
    assert current._binary_marker_batches(markers,100_000)==(markers,)
    monkeypatch.setenv('VCG_JOB_MEMORY_GB','4')
    batches = current._binary_marker_batches(markers,100_000)
    assert len(batches)==10
    assert sum(batches,())==markers
    assert all(len(batch)*100_000<=20_000_000 for batch in batches)

@pytest.mark.parametrize('value',['0','-1','nan','inf','invalid'])
def test_invalid_memory_cap_rejected(monkeypatch,value):
    monkeypatch.setenv('VCG_JOB_MEMORY_GB',value)
    with pytest.raises(ValueError):
        current._binary_marker_batches(('m',),100)

def test_batched_binary_exactly_matches_pinned_results(monkeypatch):
    old = pinned()
    rng = np.random.default_rng(77)
    n = 600
    markers = tuple(f'm{i}' for i in range(17))
    frame = pd.DataFrame({'study_id':np.arange(n), 'split':['train']*300+['validation']*300})
    for i,m in enumerate(markers):
        frame[m] = rng.integers(0,9,n).astype(float) if i%2 else rng.normal(size=n)
    frame.loc[::9,'m1'] = np.nan
    presence = pd.DataFrame([{'study_id':int(i),'target_id':t,'temporal_relation':r}
        for t,r in [('a','before'),('a','after'),('b','before')]
        for i in rng.choice(n,180,replace=False)])
    policy = current.policy_for('procedure_code')
    expected = old._ordered_records(old._screen_binary(frame,presence,'procedure_code',policy,{},markers,already_indexed=True),markers)
    assert expected
    # Force multiple materializations at the real cohort size of this fixture.
    monkeypatch.setenv('VCG_JOB_MEMORY_GB','0.00036')
    batches = current._binary_marker_batches(markers,n)
    assert len(batches)>1
    actual = []
    for batch in batches:
        actual.extend(current._screen_binary(frame[['study_id','split',*batch]],presence,'procedure_code',policy,{},batch,already_indexed=True))
    pd.testing.assert_frame_equal(pd.DataFrame(current._ordered_records(actual,markers)),pd.DataFrame(expected),check_exact=True)

def test_old_checkpoint_reused_without_writing(monkeypatch,tmp_path):
    old = pinned()
    block = ('m1','m2')
    key = old._axis_block_key('stage','ccsr_category','binary',block,600,('a',))
    assert current._axis_block_key('stage','ccsr_category','binary',block,600,('a',)) == key
    old._save_axis_block(tmp_path,'ccsr_category','binary',1,key,[{'feature_id':'m1','effect':0.75}])
    path = old._axis_block_path(tmp_path,'ccsr_category','binary',1)
    before = (path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setenv('VCG_JOB_MEMORY_GB','4')
    assert current._load_axis_block(tmp_path,'ccsr_category','binary',1,key)==[{'feature_id':'m1','effect':0.75}]
    assert (path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest())==before
    assert current._load_axis_block(tmp_path,'ccsr_category','binary',1,'wrong-key') is None

def test_actual_axis_resumes_saved_memory_batch_exactly(monkeypatch,tmp_path):
    import duckdb
    old = pinned()
    n = 600
    markers = tuple(f'm{i}' for i in range(9))
    rng = np.random.default_rng(101)
    features = pd.DataFrame({'study_id':np.arange(n), 'subject_id':np.arange(n)+1,
        'ecg_time':pd.Timestamp('2020-01-01'), 'split':['train']*300+['validation']*300})
    for m in markers:
        features[m] = rng.integers(0,10,n).astype(float)
    features.loc[::11,'m1'] = np.nan
    presence = pd.DataFrame({'study_id':np.r_[np.arange(100),np.arange(300,400)],
        'target_id':'a','temporal_relation':'before'})
    source = {'counts_sql':lambda:"SELECT 'a' entity_id, 'A' entity_label, 200 n_subjects, 200 n_observations",
        'relation':'presence','entity_column':'target_id'}
    connection = duckdb.connect()
    connection.register('features',features)
    connection.register('presence',presence)
    connection.execute(f'CREATE TEMP TABLE {current._INDEX_COHORT_TABLE} AS SELECT study_id,subject_id,ecg_time,split FROM features')
    for module in (old,current):
        monkeypatch.setattr(module,'_feature_parquet',lambda value:'features')
        monkeypatch.setattr(module,'binary_axis_values_sql',lambda *args,**kwargs:'SELECT * FROM presence')
    kwargs = dict(axis='procedure_code',policy=current.policy_for('procedure_code'),source=source,
        master_relation='features',feature_glob='features',marker_ids=markers,
        markers_prevalidated=True,checkpoint_key='same-stage')
    monkeypatch.delenv('VCG_JOB_MEMORY_GB',raising=False)
    expected = old._screen_one_axis(connection,progress=None,checkpoint_root=None,**kwargs)[1]
    assert not expected.empty
    monkeypatch.setenv('VCG_JOB_MEMORY_GB','0.00036')
    def pause_after_saved(message):
        if 'Speicherpaket 1/3 gesichert' in message:
            raise RuntimeError('cooperative pause after durable batch')
    with pytest.raises(RuntimeError,match='cooperative pause'):
        current._screen_one_axis(connection,progress=pause_after_saved,checkpoint_root=tmp_path,**kwargs)
    saved = list(tmp_path.rglob('*.pkl'))
    assert len(saved)==1
    path = saved[0]
    before = (path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest())
    screened = []
    original = current._screen_binary
    def observed(*args,**kwargs):
        screened.extend(args[5])
        return original(*args,**kwargs)
    monkeypatch.setattr(current,'_screen_binary',observed)
    actual = current._screen_one_axis(connection,progress=None,checkpoint_root=tmp_path,**kwargs)[1]
    assert set(screened)==set(markers[3:]), 'Saved first memory batch must not be recomputed'
    assert before==(path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest())
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    connection.close()
