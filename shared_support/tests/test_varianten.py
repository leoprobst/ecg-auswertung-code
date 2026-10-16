"""AW-023: ausschließlich synthetische Signale/IDs; keine realen Testdaten."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from werkzeuge.varianten.vertrag import development_file,draw,validate_rows
from werkzeuge.varianten.metriken import repaired_qrs,graph_statistics,near_p5_baseline,form_distance
from werkzeuge.varianten.vergleich import paired_auc,bootstrap,simultaneous,icc_sdc,log_sd_ratio,invariance,fold_consistent,agreement

def population():
    rows=[]
    for fold,start in [('train',0),('validation',10000)]:
        for i in range(4000):
            rows.append(dict(subject_id=start+i,study_id=start+i,split=fold,ecg_time='2020-01-01',waveform_path='synthetic',label=i%2,qrs_ms=100 if i%3 else 140,rhythm_stratum='known' if i%5 else 'unknown',reference_time='2020-01-01'))
    return pd.DataFrame(rows)

def test_split_whitelist():
    for path,fold in [(str(data_path("b.locked_test_part_0_parquet")),'train'),(str(data_path("b.x_part_0_parquet")),'test')]:
        with pytest.raises(ValueError):development_file(path,fold)
    frame=population();frame.loc[1,'subject_id']=10000
    with pytest.raises(ValueError):validate_rows(frame)

def test_draw_deterministic_stratified_no_markers():
    frame=population();a=draw(frame);b=draw(frame.sample(frac=1,random_state=13).assign(marker=np.arange(len(frame))))
    assert a.study_id.tolist()==b.study_id.tolist()
    assert a.groupby(['split','label']).size().to_dict()=={('train',0):2000,('train',1):706,('validation',0):2000,('validation',1):706}
    assert a.subject_id.nunique()==len(a)

def test_graph_components_and_null():
    adjacency=np.zeros((6,6),bool)
    for offset in (0,3):
        adjacency[offset:offset+3,offset:offset+3]=True
    np.fill_diagonal(adjacency,False)
    result=graph_statistics(adjacency)
    assert result=={'edges':6,'vertices':6,'components':2,'beta':2}
    assert graph_statistics(np.zeros((4,4),bool))['beta']==0
    with pytest.raises(ValueError):graph_statistics(np.ones((3,3),bool))

@pytest.mark.parametrize('n',[8,64,128])
def test_near_p5_analytic_count(n):
    result=near_p5_baseline(n);m=result['pairs'];distinct=np.arange(m,dtype=float)
    assert np.count_nonzero(distinct<=np.percentile(distinct,5))==result['distinct_count']

def curve(n,kind='ellipse',nonuniform=False):
    u=np.linspace(0,1,n)
    if nonuniform:u=u**1.3
    if kind=='ellipse':return np.column_stack([np.cos(1.7*np.pi*u),.6*np.sin(1.7*np.pi*u),np.zeros(n)])
    if kind=='circle':return np.column_stack([np.cos(2*np.pi*u),np.sin(2*np.pi*u),np.zeros(n)])
    if kind=='hairpin':return np.column_stack([np.sin(np.pi*u),.04*np.cos(np.pi*u),np.zeros(n)])
    return np.column_stack([u,np.where(u<.5,.02,-.02),np.zeros(n)])

def test_scale_translation_and_degeneracy():
    x=curve(256);ref=repaired_qrs(x)
    for scale in (.5,1,2):
        actual=repaired_qrs(x*scale+7)
        for key in ('near','graph_p10','graph_fixed','events'):assert actual[key]==ref[key]
    assert repaired_qrs(np.ones((64,3))) is None
    assert form_distance(x,-x)>.1  # Separate RMS-Normierung entfernt keine Rotation.

def test_frechet_rotation_shared_time():
    q=curve(100);rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    assert form_distance(q,q@rotation,'rotation')<1e-8
    assert form_distance(q,q@rotation,'arc')>.1
    assert form_distance(q,q*2,'arc')<1e-8
    assert form_distance(q,q*2,'shared')>.1
    assert form_distance(q,q,'time')<1e-8

def test_paired_auc_common_cases():
    y=[0,0,1,1];a=[0,np.nan,1,2];b=[2,1,0,0]
    assert paired_auc(y,a,b)==1
    assert np.isnan(paired_auc([0,0],[1,2],[2,1]))

def test_bootstrap_cluster_common_weights():
    clusters=np.repeat(np.arange(24),2);seen=[]
    def estimate(w):
        assert np.array_equal(w[::2],w[1::2]);seen.append(w)
        return np.array([np.average(np.arange(48),weights=w),np.average(np.arange(48)**2,weights=w)])
    result=bootstrap(clusters,estimate,2,replicates=120)
    assert result['valid_draws']==120
    assert np.all(result['lower']<result['point'])
    with pytest.raises(ValueError):bootstrap([1,1],estimate,2,strata=[0,1],replicates=2)

def test_simultaneous_and_null_se():
    rng=np.random.default_rng(5);z=rng.normal(size=(1000,48));point=np.zeros(48)
    result=simultaneous(point,z,48)
    assert result['critical']>2.5
    with pytest.raises(ValueError):simultaneous(np.zeros(48),np.zeros((100,48)),48)
    with pytest.raises(ValueError):simultaneous(point,z[:,:47],48)
    bad=z.copy();bad[:100,0]=np.nan
    with pytest.raises(ValueError):simultaneous(point,bad,48)

def test_icc_absolute_bias_sdc_and_scale_guard():
    x=np.column_stack([np.arange(10),np.arange(10)+.5])
    assert icc_sdc(x)['icc_a1']<1
    assert icc_sdc(np.column_stack([np.arange(10)]*2))['icc_a1']==1
    assert log_sd_ratio(x*.1,x,1,10)==pytest.approx(0)
    assert agreement(x[:,0],x[:,1])['bias']==.5
    assert invariance(.5,.505,fraction=True)
    assert not invariance(100,106)
    assert fold_consistent(.01,.02) and not fold_consistent(-.01,.02)

def test_equivalence_failure_missingness_and_rounding():
    from werkzeuge.varianten.adapter import equivalent,ORIGINALS
    actual={k:1. for k in ORIGINALS};expected=actual.copy()
    assert equivalent(actual,expected)
    actual[ORIGINALS[0]]=1+5e-9;assert equivalent(actual,expected)
    actual[ORIGINALS[0]]=1+2e-8
    with pytest.raises(RuntimeError):equivalent(actual,expected)
    actual[ORIGINALS[0]]=None
    with pytest.raises(RuntimeError):equivalent(actual,expected)

def test_aw023b_mapping_and_raw_path_barriers():
    from werkzeuge.varianten.aw023b import validate_mapping,resolve_waveform
    frame=pd.DataFrame([dict(subject_id=1,study_id=2,split='train',relative_waveform_path='files/synthetic',label_state='positive')])
    validate_mapping(frame)
    for field,value in [('split','test'),('label_state','unavailable'),('study_id',None)]:
        bad=frame.copy();bad[field]=value
        with pytest.raises(ValueError):validate_mapping(bad)
    duplicate=pd.concat([frame,frame.assign(study_id=3,split='validation')])
    with pytest.raises(ValueError):validate_mapping(duplicate)
    for relative in ['../outside',str(data_path("b.outside_record")),'files/../../outside']:
        with pytest.raises(ValueError):resolve_waveform(relative)
    assert 'files' in resolve_waveform('files/synthetic')

def test_aw023b_equivalence_metadata_stop():
    from werkzeuge.varianten.adapter import equivalent,ORIGINALS
    actual={m:None for m in ORIGINALS};expected=actual.copy()
    actual['feature_confidence_json']=json.dumps({ORIGINALS[0]:0.5})
    expected['feature_confidence_json']=json.dumps({ORIGINALS[0]:0.6})
    with pytest.raises(RuntimeError,match='Metadatenabweichung'):equivalent(actual,expected)

def test_aw023b_cli_stop_has_failure_exit(monkeypatch):
    from werkzeuge.varianten import aw023b
    monkeypatch.setattr(aw023b,'memory_guard',lambda: {})
    monkeypatch.setattr(aw023b,'smoke_run',lambda: {'status':'stop_gate'})
    with pytest.raises(SystemExit) as exc:aw023b.main(['smoke'])
    assert exc.value.code==2
    monkeypatch.setattr(aw023b,'smoke_run',lambda: {'status':'passed'})
    aw023b.main(['smoke'])

def test_aw023c_context_query_gating_and_id_filter():
    from werkzeuge.varianten.aw023c import context_sql,MASTER,GATED,IDENTITY,CONTEXT
    query=context_sql(MASTER)
    assert 'WHERE study_id IN (SELECT study_id FROM wanted)' in query
    assert "split IN ('train','validation')" in query
    assert len(IDENTITY+CONTEXT)==20
    for column in GATED:
        assert f'CASE WHEN machine_metadata_usable_for_primary THEN {column} ELSE NULL END AS {column}' in query
    assert 'CASE WHEN machine_metadata_usable_for_primary THEN machine_filtering' not in query
    assert 'label' not in query and 'locked_test' not in query

def test_aw023c_block_output():
    import pyarrow.parquet as pq
    from werkzeuge.varianten.aw023c import write_block
    from werkzeuge.varianten.audit import OUT
    import uuid
    target=OUT/'AW023c'/'synthetik'/f'block_{uuid.uuid4().hex}.parquet'
    rows=[{'study_id':i,'variant':f'synthetic-{j}','value':None if j==0 else float(j),'abstention':'synthetic' if j==0 else ''} for i in range(20) for j in range(16)]
    write_block(target,rows)
    result=pq.read_table(target).to_pandas()
    assert len(result)==320 and result.study_id.nunique()==20
    assert result.value.isna().sum()==20

def test_atomic_checkpoint_freeze():
    from werkzeuge.varianten.lauf import checkpoint
    from werkzeuge.varianten.audit import OUT
    with pytest.raises(ValueError):checkpoint(Path(str(data_path("b.outside_json"))),{'n':1})
    path=OUT/'synthetik'/'checkpoint_test.json'
    checkpoint(path,{'sample_hash':'synthetic','completed':[1]})
    checkpoint(path,{'sample_hash':'synthetic','completed':[1,2]})
    assert json.loads(path.read_text(encoding='utf-8'))['completed']==[1,2]

def test_200_block_resume_freeze():
    from werkzeuge.varianten.lauf import BlockCheckpoint
    from werkzeuge.varianten.audit import OUT
    import uuid
    path=OUT/'synthetik'/f'block_{uuid.uuid4().hex}.json'
    block=BlockCheckpoint(path,'synthetic')
    for i in range(199):block.mark(i)
    assert not path.exists()
    block.mark(199)
    resumed=BlockCheckpoint(path,'synthetic');assert len(resumed.completed)==200
    resumed.mark(200);resumed.flush()
    assert len(BlockCheckpoint(path,'synthetic').completed)==201
    with pytest.raises(ValueError):BlockCheckpoint(path,'wrong')

def test_memory_below_threshold_stops(monkeypatch):
    from werkzeuge.varianten import audit
    def low(pointer):
        pointer._obj.availPhys=3*1024**3;pointer._obj.availPage=10*1024**3;return True
    monkeypatch.setattr(audit.ctypes.windll.kernel32,'GlobalMemoryStatusEx',low)
    with pytest.raises(RuntimeError):audit.memory_guard()

def test_huber_train_fixed_and_oof():
    from werkzeuge.varianten.residuen import Residualizer,oof_residual
    rng=np.random.default_rng(8);x=rng.normal(size=(100,4));sex=np.array(['F','M']*50);y=x@np.array([1,2,3,4])+rng.normal(scale=.01,size=100)
    fit=Residualizer().fit(x,sex,y)
    assert np.std(fit.residual(x,sex,y))<.02
    assert np.std(oof_residual(x,sex,y,np.arange(100)))<.03
    with pytest.raises(ValueError):Residualizer().fit(np.zeros((10,4)),sex[:10],y[:10])

def test_release_synthetic_context_all_active():
    from werkzeuge.varianten.adapter import release_import,variants,prepared_originals,ORIGINALS
    release_import()
    from vcg_biomarker.core.delineation.base import MarkerPoint,BeatDelineation,RecordDelineation
    from types import SimpleNamespace
    n=2000;signal=np.zeros((n,12));beats=[]
    rng=np.random.default_rng(18)
    for r in (300,800,1300):
        def p(sample):return MarkerPoint(sample,1.,'synthetic')
        beat=BeatDelineation(r_peak=p(r),r_onset=p(r-30),r_offset=p(r+35),t_onset=p(r+90),t_peak=p(r+150),t_offset=p(r+230))
        beat.morphology_cluster_id=0;beats.append(beat)
    signal=rng.normal(size=(n,12))*.01
    delineation=RecordDelineation(beats=beats,fs=500,n_samples=n,ecg_id=0,quality_score=1.)
    package={'loaded':SimpleNamespace(signal_millivolts=signal),'paths':{'morphology':signal},'robust':signal,'delineation':delineation,'fs':500,'acquisition':{}}
    out,qc=variants(package)
    assert len(out)==16
    originals=prepared_originals(package)
    assert all(k in originals for k in ORIGINALS)
    assert out['VB-VF01-08']['value'] is None
    assert out['VB-VF02-08']['value'] is None


def test_aw025_wait_instead_of_abort(monkeypatch):
    from werkzeuge.varianten import aw025
    seen=[]
    def memory(pointer):
        pointer._obj.availPhys=8*1024**3
        pointer._obj.availPage=(3 if not seen else 5)*1024**3
        return True
    monkeypatch.setattr(aw025.ctypes.windll.kernel32,'GlobalMemoryStatusEx',memory)
    monkeypatch.setattr(aw025.time,'sleep',lambda seconds:seen.append(seconds))
    result=aw025.wait_memory()
    assert seen==[60] and result['available_commit_bytes']==5*1024**3


def test_aw025_weighted_auc_ties_missing_cluster_draws():
    from sklearn.metrics import roc_auc_score
    from werkzeuge.varianten.aw025_auswerten import WeightedAUC,paired_estimators,cluster_draws,correction_contract
    y=np.array([0,1,0,1,0,1]);x=np.array([.2,.2,.4,.8,np.nan,.8]);weights=np.array([1,3,2,4,5,2])
    keep=np.isfinite(x)
    assert WeightedAUC(y,x)(weights)==pytest.approx(roc_auc_score(y[keep],x[keep],sample_weight=weights[keep]))
    a,b=paired_estimators(y,x,np.array([0,1,np.nan,1,0,1]))
    common=keep&np.array([True,True,False,True,True,True])
    assert a(weights)==pytest.approx(roc_auc_score(y[common],x[common],sample_weight=weights[common]))
    point,draws=cluster_draws(y,[(a,b)],replicates=20)
    assert draws.shape==(20,1) and point.shape==(1,)
    missing=np.full((100,48),np.nan)
    result=correction_contract(np.full(48,np.nan),missing)
    assert result['status']=='nicht entscheidbar' and result['expected_contrasts']==48
