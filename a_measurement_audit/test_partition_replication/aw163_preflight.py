"""AW163: ausschließlich Entwicklung; unveränderliche Vorprüfungsbindung."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys, json, hashlib, re, ctypes
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("a.aw163")))
REL=Path(str(data_path("a.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
RUN=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e")))
EXCLUDED={'beatwise_p_loop_axis_stability_deg','t_loop_to_qrs_loop_complexity_ratio'}
PRELOADED={'svg_qrs_area_ratio','spatial_qrs_t_angle_deg','qrs_t_hausdorff_distance'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(n,x):
    p=OUT/n
    if p.exists():raise RuntimeError('Vorhandene Bindung nicht überschreiben')
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def main():
    import pandas as pd, numpy as np
    OUT.mkdir(exist_ok=True)
    specs={'SPEZIFIKATION_A_TESTREPLIKATION_v1.md':'4cdb6d32c151a758cc947bf0e679b1d1b455ae714993dd36d8e3930e4f60dd28','SPEZIFIKATION_A_TESTREPLIKATION_v1_NACHTRAG1.md':'0d2a24a18fc8e0dcbb10e992800129c4cf87bf096b2aad7a6729bce1774e4ca7'}
    assert all(sha(ROOT/'nachrechnung'/n)==h for n,h in specs.items())
    checks=[]
    for line in (ROOT/'nachrechnung/SPEZIFIKATION_A_TESTREPLIKATION_v1.md').read_text(encoding='utf8').splitlines():
        cells=line.split('|')
        if len(cells)<4:continue
        pm=re.search(r'`([^`]+)`',cells[1]);hm=re.search(r'`([a-f0-9]{64})`',line)
        if not pm or not hm:continue
        p=pm[1]
        path=REL/p[4:] if p.startswith('REL/') else RUN/'publication_preparation'/p if p.startswith('research-') else Path(p) if re.match(r'^[A-Z]:',p) else ROOT/p
        actual=sha(path) if path.is_file() else None
        checks.append({'path':str(path),'expected':hm[1],'actual':actual,'passed':actual==hm[1]})
    assert len(checks)==30
    save('source_hash_checks.json',checks)
    if not all(x['passed'] for x in checks):
        save('status_preflight.json',{'status':'STOP_SOURCE_HASH','test_values_opened':False,'failed':[x for x in checks if not x['passed']]})
        return
    cat=pd.read_csv(ROOT/'nachrechnung/transform_katalog.csv')
    metrics=Path(str(data_path("a.rv_a1_repeat_marker_metrics_csv")))
    w=pd.read_csv(metrics)
    summaries={}
    for g in ('gesamt','proxy'):
        a=w.loc[w.population.eq(g)&~w.marker_id.isin(EXCLUDED)]
        # Preserve all native fields and original fixed development lists.
        a.to_csv(OUT/f'development_W_{g}.csv',index=False)
        icc=a['icc'] if 'icc' in a else a['icc_a1']
        summaries[g]={'universe':len(a),'finite_icc':int(np.isfinite(icc).sum()),'median_icc':float(np.nanmedian(icc))}
    m=cat.loc[~cat.marker_id.isin(EXCLUDED|PRELOADED)]
    m.to_csv(OUT/'development_M.csv',index=False)
    summaries['M']={'universe':len(m),'classes':{k:int(v) for k,v in m['class'].value_counts().items()},'transform_dependent':int(m['class'].eq('transformabhaengig').sum()),'proportion':float(m['class'].eq('transformabhaengig').mean())}
    old=json.loads(Path(str(data_path("a.aw158_marker_universes_json"))).read_text(encoding='utf8'))
    new={k:([x for x in v if x not in EXCLUDED] if isinstance(v,list) and k!='M_excluded' else v) for k,v in old.items()}
    new['technical_excluded']=sorted(EXCLUDED)
    save('marker_universes.json',new)
    save('development_benchmarks.json',{'specifications':specs,'sources':{str(metrics):sha(metrics),str(ROOT/'nachrechnung/transform_katalog.csv'):sha(ROOT/'nachrechnung/transform_katalog.csv')},'summaries':summaries,'lists':{k:len(v) for k,v in new.items() if isinstance(v,list)},'test_values_opened':False,'generator_sha256':sha(__file__)})
    save('status_preflight.json',{'status':'DEVELOPMENT_BENCHMARKS_BOUND','test_values_opened':False})
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
