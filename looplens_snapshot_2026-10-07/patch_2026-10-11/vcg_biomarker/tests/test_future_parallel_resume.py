import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from vcg_biomarker.mimic import frame_checkpoints as cp, future_phenome as f


def reference(name):
    path = Path(__file__).parent/'fixtures/future_parallel_bound_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '13d6611290359f76ada129d66741c418cbbc0fb7483a457d1ae61dd916fc8d0d'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    namespace = dict(f.__dict__)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace[name]


def test_two_processes_compute_one_chunk_and_reuse_it(tmp_path):
    script = '''import sys,time
from pathlib import Path
import pandas as pd
from vcg_biomarker.mimic.frame_checkpoints import checkpointed_frame_chunk_path
r=Path(sys.argv[1])
def compute():
 with (r/'computations.txt').open('a') as out: out.write('computed\\n')
 time.sleep(.5)
 return pd.DataFrame({'marker':['m'],'value':[.125]})
checkpointed_frame_chunk_path(r,namespace='same',chunk_index=0,input_hash='i',item_keys=['m'],compute=compute)
'''
    children = [subprocess.Popen([sys.executable, '-c', script, str(tmp_path)]) for _ in range(2)]
    try:
        assert [p.wait(timeout=15) for p in children] == [0, 0]
    finally:
        for p in children:
            if p.poll() is None: p.kill(); p.wait()
    assert (tmp_path/'computations.txt').read_text().splitlines() == ['computed']
    path = tmp_path/'same/part-000000.parquet'
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    actual, reused = cp.checkpointed_frame_chunk_path(tmp_path, namespace='same', chunk_index=0,
        input_hash='i', item_keys=['m'], compute=lambda: pytest.fail('recomputed'))
    assert reused and actual == path
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(RuntimeError):
        cp.checkpointed_frame_chunk_path(tmp_path, namespace='same', chunk_index=0,
            input_hash='changed', item_keys=['m'], compute=lambda: pytest.fail('overwritten'))


def test_process_death_releases_claim(tmp_path):
    script = '''import sys,time
from pathlib import Path
from vcg_biomarker.mimic.frame_checkpoints import frame_chunk_lock
r=Path(sys.argv[1])
with frame_chunk_lock(r,namespace='same',chunk_index=0):
 (r/'ready').write_text('ready')
 time.sleep(30)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)])
    try:
        end = time.monotonic()+10
        while not (tmp_path/'ready').exists() and time.monotonic()<end: time.sleep(.05)
        assert (tmp_path/'ready').exists()
    finally:
        child.terminate(); child.wait(timeout=10)
    with cp.frame_chunk_lock(tmp_path, namespace='same', chunk_index=0):
        pass


def data():
    n = 80; rng=np.random.default_rng(42); end=pd.Timestamp('2020-01-01')
    frame=pd.DataFrame(dict(subject_id=np.arange(n),split=['train']*n,
        index_end_time=[end]*n,last_observed_time=[end+pd.Timedelta(days=50)]*n,
        date_of_death=[pd.NaT]*n,age=rng.uniform(30,80,n),sex=['M','F']*(n//2),
        ecg_time=[end-pd.Timedelta(hours=12)]*n,heart_rate_bpm=rng.uniform(50,100,n),
        machine_heart_rate_bpm=rng.uniform(50,100,n),signal_quality_score=rng.uniform(.7,1,n),
        a=rng.normal(size=n),b=rng.normal(size=n)))
    frame.loc[[2,7,10],'b']=np.nan
    events=pd.Series([end+pd.Timedelta(days=int(i%25+1)) for i in range(40)],index=np.arange(40))
    definitions=pd.DataFrame(dict(phenotype_id=['p'],display_name=['P'],body_system=['body'],
        male_only=[False],female_only=[False])).set_index('phenotype_id')
    return frame,events,definitions


def test_screen_and_tie_mask_reuse_match_bound_values_and_dtypes():
    frame,events,definitions=data()
    args=dict(selected_targets=(('p',30),),event_lookup={'p':events},definition_lookup=definitions,
        feature_ids=('a','b'),panel_map={'a':'x','b':'x'},min_inference=5,min_formal=10,progress=None)
    expected=reference('_screen_future_targets')(frame,**args)
    actual=f._screen_future_targets(frame,**args)
    assert_frame_equal(actual,expected,check_exact=True)
    frame.loc[2,'date_of_death']=events.loc[2]
    masks=f._future_state_masks(frame,events,definitions.loc['p'],horizon_days=30)
    args=dict(horizon_days=30,primary_direction=1)
    assert f._same_day_tie_sensitivity(frame,frame.a.to_numpy(),events,definitions.loc['p'],
        primary_masks=masks,**args)==reference('_same_day_tie_sensitivity')(
            frame,frame.a.to_numpy(),events,definitions.loc['p'],**args)


def test_completed_future_block_skips_raw_join_and_preserves_checkpoint(tmp_path,monkeypatch):
    frame,events,definitions=data()
    root=tmp_path/'cp'
    expected=pd.DataFrame(dict(phenotype_id=['p','p'],horizon_days=[30,30],feature_id=['a','b'],effect=[.1,.2]))
    cp.checkpointed_frame_chunk_path(root,namespace='future_train_stream_v1/block-0000',chunk_index=0,
        input_hash='i',item_keys=(('p',30,'a'),('p',30,'b')),compute=lambda:expected)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('part-*.parquet')}
    monkeypatch.setattr(f,'_load_future_feature_block',lambda *a,**k:pytest.fail('raw data loaded'))
    actual=f._checkpointed_future_train_screen_streaming(root,config=None,feature_glob='unused',
        index_path=tmp_path/'unused',checkpoint_input_hash='i',feature_index=frame,
        selected_targets=(('p',30),),event_lookup={'p':events},definition_lookup=definitions,
        feature_ids=('a','b'),panel_map={'a':'x','b':'x'},min_inference=5,min_formal=10,progress=None)
    assert_frame_equal(actual,expected,check_exact=True)
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('part-*.parquet')}
