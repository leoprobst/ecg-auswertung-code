"""Vorab-Messvertrag als Ergebnis, einschließlich Nichtbestehen; keine EKGs."""
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import json,time,hashlib
from pathlib import Path
from .audit import OUT,memory_guard,ROOT

def main():
    memory=memory_guard();started=time.perf_counter()
    import numpy as np
    from scipy.spatial.distance import cdist
    from .metriken import repaired_qrs,graph_statistics
    from .vergleich import invariance,simultaneous
    from .lauf import Budget
    from werkzeuge.tests.test_varianten import curve
    records=[]
    with Budget() as budget:
        for kind in ('circle','ellipse','hairpin','branches'):
            reference=repaired_qrs(curve(8192,kind))
            for n in (32,64,128,256,512):
                for irregular in (False,True):
                    for scale in (.5,1,2):
                        x=curve(n,kind,irregular)*scale
                        result=repaired_qrs(x)
                        d=cdist(x,x);i,j=np.triu_indices(n,5);dist=d[i,j]
                        radius=np.percentile(dist,10)
                        original=graph_statistics((d>0)&(d<radius))
                        row={'curve':kind,'input_n':n,'nonuniform':irregular,'global_scale':scale,'reference':reference,'repaired':result,'original_p5_count':int(np.sum(dist<=np.percentile(dist,5))),'original_graph_plus1':max(0,original['edges']-n+1),'original_components':original['components']}
                        row['passes']={key:invariance(reference[key],result[key],fraction=key in ('near','graph_fixed')) for key in ('near','graph_p10','graph_fixed','events')}
                        records.append(row)
        # Gemeinsame inferentielle Familien in fester Dimension, keine reale AUC.
        rng=np.random.default_rng(20261008);families={}
        for k in (48,64):
            common=rng.normal(size=(10000,1));draws=.005*(common+rng.normal(size=(10000,k)))
            result=simultaneous(np.zeros(k),draws,k)
            families[str(k)]={'draws':result['valid_draws'],'alpha':result['alpha'],'critical':result['critical'],'all_intervals_contain_null':bool(np.all((result['lower']<=0)&(result['upper']>=0)))}
        # Zeitdehnung verändert bei identischer Geometrie nur die Zeitmetadaten.
        report={'memory_start':memory,'synthetic_only':True,'synthetic_cases':len(records),'failed_cases':sum(not all(r['passes'].values()) for r in records),'cases':records,'families':families,'time_scale_control':[.5,1,2],'time_scale_explanation':'Nur Metadaten, Geometrie gleich; kein klinischer Dauernachweis','seconds':time.perf_counter()-started,'peak_rss_bytes':budget.peak}
    (OUT/'AW023_SYNTHETIK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Quellbindung ohne Patientenidentitäten.
    manifest={'local_sources':{},'release_sources':{},'plans':{}}
    for file in sorted(Path('werkzeuge/varianten').glob('*.py'))+[Path('werkzeuge/tests/test_varianten.py')]:manifest['local_sources'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    for relative in ('mimic/feature_record.py','mimic/signals.py','core/preprocessing.py','core/vcg_transform.py','core/biomarkers/catalog_biomarkers.py','core/biomarkers/discovery_v3_biomarkers.py','core/biomarkers/qrs_t_coupling_extension_biomarkers.py'):
        file=ROOT/'vcg_biomarker'/relative;manifest['release_sources'][relative]=hashlib.sha256(file.read_bytes()).hexdigest()
    import importlib.metadata
    manifest['packages']={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','pyarrow','scikit-learn','wfdb','neurokit2','psutil','pytest')}
    for file in (Path('verbesserung/VF-01_qrs-t-kopplung/VERBESSERUNG.md'),Path('verbesserung/VF-02_zaehlproxy-reparatur/VERBESSERUNG.md')):manifest['plans'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest['full_run_authorized']=False
    (OUT/'AW023_QUELLEN.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
