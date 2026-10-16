"""AW143: ausschließlich train, explizite Mess-/Metadatenspalten, Releasefunktionen."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, hashlib, csv, copy, time, argparse, ctypes
from pathlib import Path
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
OUT=Path(str(data_path("b.aw143")))
RELEASE=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
SPEC=PROJECT/'nachrechnung/SPEZIFIKATION_B_STOERTEST_v1.md'
SPEC_HASH='fe21bd06af5bfa617665a0ebc3b1324c3e9384530a8309657241f62ff147d4d6'
IDS=['qrs_t_normalized_discrete_frechet','qrs_t_frechet_beat_median','qrs_t_normalized_dtw_distance','spatial_qrs_t_angle_deg','qrs_duration_ms','qtc_fridericia_ms']
META=['subject_id','study_id','split','ecg_time','age','sex','waveform_path','machine_filtering','machine_bandwidth','machine_metadata_usable_for_primary','machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']
SEED=20261012
OPTIONS={'vcg_feature_ids':IDS,'classical_feature_ids':[], 'sensitivity_feature_ids':[], 'sensitivity_vcg_methods':[], 'powerline_hz':60.,'delineation_weights':{'neurokit':.4,'multilead':.35,'vcg':.25},'delineation_conflict_threshold_ms':20.}
S=[(attr,sign,sd) for attr,sd in [('r_onset',13.5),('r_offset',11.6),('t_offset',32.8)] for sign in [-1,1]]
SOURCES={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')
def bind(p):
    p=Path(p)
    if any(s in str(p).lower() for s in ['locked_test','split=test','holdout','labels','clinical_context','reference']):raise PermissionError('Verbotene Eingabe')
    SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
def readp(p,columns,filter_expr=None):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    bind(p);schema=pq.ParquetFile(p).schema_arrow
    if not set(columns)<=set(schema.names):raise ValueError('Fehlende Spalten')
    with (OUT/'access_private.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'path':str(p),'columns':columns,'filter':str(filter_expr)})+'\n')
    return ds.dataset(str(p),format='parquet').to_table(columns=columns,filter=filter_expr).to_pandas()
def prepare():
    import numpy as np
    import pyarrow.dataset as ds
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e_split_train_features_parquet")))
    f=readp(p,['subject_id','study_id']+IDS)
    n=len(f);f=f.loc[np.isfinite(f[IDS].to_numpy(dtype=float)).all(axis=1)].sort_values(['subject_id','study_id']).reset_index(drop=True)
    finite_n=len(f)
    # Zufällige EKG-Reihenfolge, erstes je Person; dann erste 500. Kein Endpunktfilter.
    rng=np.random.default_rng(SEED)
    f=f.iloc[rng.permutation(len(f))].drop_duplicates('subject_id').head(500).sort_values('study_id')
    if len(f)!=500:raise ValueError('Weniger als 500 Personen')
    master=Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")))
    meta=readp(master,META,(ds.field('split')=='train') & ds.field('study_id').isin(f.study_id.tolist()))
    if len(meta)!=500 or not meta.split.eq('train').all() or meta.subject_id.duplicated().any():raise ValueError('Train/Personengate')
    usable=meta.machine_metadata_usable_for_primary.eq(True).fillna(False)
    for k in META:
        if k in ['machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']:
            meta[k]=meta[k].where(usable,None)
    joined=meta.merge(f,on=['subject_id','study_id'],validate='one_to_one').sort_values('study_id')
    pq.write_table(pa.Table.from_pandas(joined,preserve_index=False),OUT/'selection_private.parquet',compression='zstd')
    save('selection_aggregate.json',{'seed':SEED,'train_rows':n,'all_six_finite_rows':finite_n,'selected_ecgs':500,'selected_persons':500,'sampling':'default_rng permutation finite ECGs sorted subject_id/study_id; first ECG per person, first 500; final study_id order','sources':SOURCES,'selection_sha256':sha(OUT/'selection_private.parquet')})
def compute(package,delineation):
    import numpy as np
    from werkzeuge.varianten.adapter import release_import
    fr=release_import()
    record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],'kors').xyz
    robust=fr.VCGTransformRegistry.transform(package['robust'],'kors').xyz
    values=fr.BiomarkerRegistry.compute_all(delineation=delineation,ecg_12lead=package['robust'],vcg_xyz=robust,fs=package['fs'],names=IDS,record_ecg_12lead=package['paths']['morphology'],record_vcg_xyz=record,record_raw_ecg_12lead=package['loaded'].signal_millivolts,record_acquisition=package['acquisition'])
    result={};failures=[]
    fr._store_biomarker_results(result,values,failures,metadata={})
    return [float(result.get(k)) if result.get(k) is not None else float('nan') for k in IDS]
def shifted(d,attr,samples):
    z=copy.deepcopy(d)
    # Alle vorhandenen Schlaggrenzen; Repräsentativgrenzen werden durch den Release neu aufgebaut.
    seen=set()
    for beat in z.beats:
        point=getattr(beat,attr,None)
        if point is not None and id(point) not in seen:
            point.sample+=samples;seen.add(id(point))
    return z
def run(limit=None):
    import numpy as np
    from werkzeuge.varianten.adapter import prepare_record
    f=readp(OUT/'selection_private.parquet',META+IDS)
    if limit:f=f.head(limit)
    start=time.monotonic();done=0
    for row in f.to_dict('records'):
        p=OUT/'records'/f"{int(row['study_id'])}.json"
        if p.exists():
            obj=json.loads(p.read_text(encoding='utf8'))
            if obj['selection_sha256']!=sha(OUT/'selection_private.parquet') or obj['script_sha256']!=sha(__file__):raise ValueError('Checkpointbindung')
            done+=1;continue
        package=prepare_record(row,OPTIONS)
        baseline=np.array(compute(package,package['delineation']))
        expected=np.array([row[k] for k in IDS],dtype=float)
        equivalent=np.isclose(baseline,expected,rtol=1e-8,atol=0,equal_nan=False)
        obj={'selection_sha256':sha(OUT/'selection_private.parquet'),'script_sha256':sha(__file__),'equivalent':equivalent.tolist(),'baseline':[float(v) if np.isfinite(v) else None for v in baseline],'expected':expected.tolist()}
        if not equivalent.all():
            save('equivalence_failure_private.json',obj)
            save('status.json',{'status':'nicht_schaetzbar','reason':'Äquivalenz ohne Störung gescheitert','done':done,'failed_markers':[k for k,e in zip(IDS,equivalent) if not e]})
            return
        results=[]
        for attr,sign,sd in S:
            samples=round(sign*sd*package['fs']/1000)
            v=compute(package,shifted(package['delineation'],attr,samples))
            results.append({'boundary':attr,'sign':sign,'samples':samples,'actual_ms':samples*1000/package['fs'],'values':[x if np.isfinite(x) else None for x in v]})
        obj['S']=results
        save('records/'+p.name,obj);done+=1
        save('status.json',{'status':'running' if done<len(f) else 'S_complete','done':done,'total':len(f),'seconds':time.monotonic()-start})
        if done%10==0:print(json.dumps({'done':done,'total':len(f),'seconds':round(time.monotonic()-start,1)}),flush=True)
def audit_histogram():
    p=PROJECT/'publikationen/PUB-01_messvaliditaet/abbildungen/v10/LUDB_AW140_boundary_histogram.csv';bind(p)
    groups={}
    with p.open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['landmark'] not in ['r_onset','r_offset','t_offset']:continue
            g=groups.setdefault(r['landmark'],{'n':0,'declared_n':int(r['n_total_paired']),'open_bin_count':0,'occupied_open_bins':[]})
            g['n']+=int(r['n'])
            if int(r['n']) and (not __import__('math').isfinite(float(r['bin_left_ms'])) or not __import__('math').isfinite(float(r['bin_right_ms']))):
                g['open_bin_count']+=int(r['n']);g['occupied_open_bins'].append(r)
    for g in groups.values():
        if g['n']!=g['declared_n']:raise ValueError('Histogrammnenner')
    save('histogram_audit.json',{'groups':groups,'sources':SOURCES,'E_status':'nicht_schaetzbar','reason':'Keine vorab festgelegte Binnenklassenregel; t_offset hat besetztes offenes Intervall (200, inf). Kein endlicher Fehlerwert daraus ableitbar.'})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);parser.add_argument('--limit',type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if sha(SPEC)!=SPEC_HASH:raise RuntimeError('STOP Spezifikationshash')
    from werkzeuge.nachrechnung.resources import available,Budget
    while min(available().values())<1.5*1024**3:time.sleep(5)
    save('resources_start_'+args.phase+'.json',available())
    with Budget(OUT) as budget:
        audit_histogram()
        if args.phase=='prepare':prepare()
        else:run(args.limit)
    save('resources_end_'+args.phase+'.json',budget.summary())
if __name__=='__main__':main()
