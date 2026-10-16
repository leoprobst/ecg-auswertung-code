"""AW143 Nachtrag 1: E, ohne Neuberechnung/Mutation der S-Checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json, csv, argparse, time, copy, datetime
from werkzeuge.b_stoertest.run import OUT, PROJECT, IDS, META, OPTIONS, SEED, SPEC, SPEC_HASH, sha, save, readp, compute
ADD=PROJECT/'nachrechnung/SPEZIFIKATION_B_STOERTEST_v1_NACHTRAG1.md'
ADD_HASH='5f66b933d9342479d7c20b3de93ffc89b40debc882f51fd9832578b232b4c7b3'
ERROR_ROOT=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_beat_boundaries_20261012_v1")))
ATTRS=['r_onset','r_offset','t_offset']

def errors(inspect=False):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import numpy as np
    paths=sorted(ERROR_ROOT.glob('errors-*.parquet'))
    if not paths:raise RuntimeError('Keine Fehlerdateien')
    schema=pq.ParquetFile(paths[0]).schema_arrow
    if inspect:
        print(str(schema),flush=True)
        t=ds.dataset(str(paths[0]),format='parquet').to_table(columns=['scope','source','landmark'])
        print({k:sorted(set(t[k].to_pylist())) for k in t.column_names},flush=True)
        return
    # Speicher-/Schema-/Spaltenprüfung je Datei, keine Referenzpositionen laden.
    arrays={k:[] for k in ATTRS}; sources={}
    for p in paths:
        schema=pq.ParquetFile(p).schema_arrow
        cols=['scope','source','landmark','error_ms']
        assert set(cols)<=set(schema.names)
        filt=(ds.field('scope')=='fused_global') & (ds.field('source')=='fused') & ds.field('landmark').isin(ATTRS)
        tab=ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=filt).to_pandas()
        sources[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        for attr in ATTRS:
            a=tab.loc[tab.landmark.eq(attr),'error_ms'].to_numpy(float)
            arrays[attr].extend(a[np.isfinite(a)].tolist())
    values={k:np.array(v,float) for k,v in arrays.items()}
    assert [len(values[k]) for k in ATTRS]==[1831,1831,1642]
    audit={'filter':"scope=fused_global; source=fused; landmark in r_onset,r_offset,t_offset; finite error_ms",
           'landmark_alias':'Auftrag qrs_onset/qrs_offset entspricht gespeicherten r_onset/r_offset; keine anderen Landmarks.',
           'groups':{k:{'n':len(v),'min_ms':float(v.min()),'max_ms':float(v.max()),'sd_ms_ddof1':float(v.std(ddof=1)),'above_200_ms':int((v>200).sum())} for k,v in values.items()},
           'sources':sources}
    # Reproduziere ALLE Histogrammklassen einschließlich der offenen Klassen.
    hist=PROJECT/'publikationen/PUB-01_messvaliditaet/abbildungen/v10/LUDB_AW140_boundary_histogram.csv'
    with hist.open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['landmark'] not in ATTRS:continue
            a=values[r['landmark']];left=float(r['bin_left_ms']);right=float(r['bin_right_ms'])
            count=int(((a>=left)&(a<right)).sum())
            if count!=int(r['n']):raise RuntimeError('Histogrammklasse abweichend: '+str(r))
    audit['histogram_all_bins_identical']=True
    save('E_errors_audit.json',audit)
    return values

def shifted_all(d, samples):
    z=copy.deepcopy(d)
    for attr,shift in zip(ATTRS,samples):
        seen=set()
        for beat in z.beats:
            point=getattr(beat,attr,None)
            if point is not None and id(point) not in seen:
                point.sample+=int(shift);seen.add(id(point))
    return z

def run(limit=None):
    import numpy as np
    from werkzeuge.varianten.adapter import prepare_record
    # Adapterguard aus früherem Auftrag fordert 4 GiB. Hier gilt explizit AE-130.
    import werkzeuge.varianten.adapter as adapter
    from werkzeuge.nachrechnung.resources import available
    def local_gate():
        while min(available().values())<1.5*1024**3:time.sleep(5)
    adapter.memory_guard=local_gate
    err=errors()
    f=readp(OUT/'selection_private.parquet',META+IDS).sort_values('study_id')
    assert len(f)==500 and f.subject_id.nunique()==500 and f.split.eq('train').all()
    # Feste Reihenfolge EKG -> Zug -> Grenze; Wiederaufnahme verändert Ziehungen nicht.
    rng=np.random.default_rng(SEED)
    shifts=np.empty((500,20,3),dtype=np.int64)
    for i in range(500):
        for j in range(20):
            for a,k in enumerate(ATTRS):shifts[i,j,a]=round(-float(rng.choice(err[k]))/2)
    np.save(OUT/'E_shifts_private.npy',shifts)
    binding={'selection_sha256':sha(OUT/'selection_private.parquet'),'script_sha256':sha(__file__),
             'adapter_sha256':sha(PROJECT/'werkzeuge/b_stoertest/run.py'),'nachtrag_sha256':sha(ADD),
             'errors_audit_sha256':sha(OUT/'E_errors_audit.json'),'shifts_sha256':sha(OUT/'E_shifts_private.npy')}
    save('E_run_manifest.json',dict(binding,seed=SEED,draws_per_ecg=20,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),draw_order='study_id ascending; ECG/draw/boundary; default_rng choice; negative error; round(error/2)'))
    start=time.monotonic();done=0
    for i,row in enumerate(f.to_dict('records')):
        if limit and i>=limit:break
        target=OUT/'records_E'/f"{int(row['study_id'])}.json"
        old=json.loads((OUT/'records'/target.name).read_text(encoding='utf8'))
        assert old['script_sha256']==binding['adapter_sha256'] and old['selection_sha256']==binding['selection_sha256'] and all(old['equivalent'])
        if target.exists():
            obj=json.loads(target.read_text(encoding='utf8'));assert obj['binding']==binding and len(obj['E'])==20
            done+=1;continue
        local_gate();package=prepare_record(row,OPTIONS)
        assert package['fs']==500
        baseline=np.array(compute(package,package['delineation']),float)
        expected=np.array(old['baseline'],float)
        if not np.isclose(baseline,expected,rtol=1e-8,atol=0,equal_nan=False).all():
            save('E_status.json',{'status':'nicht_schaetzbar','reason':'Äquivalenz gescheitert','done':done});raise RuntimeError('Äquivalenz gescheitert')
        result=[]
        for shift in shifts[i]:
            v=compute(package,shifted_all(package['delineation'],shift))
            result.append({'samples':shift.tolist(),'values':[x if np.isfinite(x) else None for x in v]})
        save('records_E/'+target.name,{'binding':binding,'baseline':baseline.tolist(),'equivalent':True,'E':result})
        done+=1
        save('E_status.json',{'status':'E_complete' if done==500 else 'running','done':done,'total':500,'seconds':time.monotonic()-start})
        if done%5==0 or done==1:print(json.dumps({'done':done,'total':500,'seconds':round(time.monotonic()-start,1)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inspect','run','report','verify']);p.add_argument('--limit',type=int);a=p.parse_args()
    if sha(SPEC)!=SPEC_HASH or sha(ADD)!=ADD_HASH:raise RuntimeError('STOP Spezifikationshash')
    from werkzeuge.nachrechnung.resources import available,Budget
    if min(available().values())<1.5*1024**3:raise RuntimeError('Speichergate')
    save('resources_start_E_'+a.phase+'.json',available())
    with Budget(OUT) as budget:
        if a.phase=='inspect':errors(True)
        elif a.phase=='run':run(a.limit)
        else:
            from werkzeuge.b_stoertest.empirical_report import report,verify
            (report if a.phase=='report' else verify)()
    save('resources_end_E_'+a.phase+'.json',budget.summary())
if __name__=='__main__':main()
