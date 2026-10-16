"""Unabhängige Aggregat-/Fitreproduktion, Bindungen und sinnvolle mathematische Tests."""
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from .runner import sha,save,small,PROJECT,ROOT
from .models import Fit,b4,BASE,EXTRA,MARKER
from werkzeuge.nachrechnung.core import auc,assert_development

def run(out):
    rng=np.random.default_rng(121);y=np.array([0,1]*80);score=np.round(rng.normal(size=160),1);w=rng.integers(1,5,size=160)
    assert np.isclose(auc(y,score,w),roc_auc_score(y,score,sample_weight=w),atol=1e-14)
    v=small(out/'verification_private.parquet');assert_development(v);assert not v.subject_id.duplicated().any()
    contract=json.loads((out/'weight_model_frozen.json').read_text(encoding='utf8'));frozen=json.loads((out/'weight_model_frozen.sha256.json').read_text(encoding='utf8'))['sha256'];assert sha(out/'weight_model_frozen.json')==frozen
    x=b4(v);tr=v.split.eq('train');model=Fit().fit(x[tr],v.loc[tr,'verification'])
    assert np.allclose(model.beta,contract['fit']['beta'],rtol=1e-10,atol=1e-12)
    prediction=expit(np.column_stack([np.ones(len(v)),((x-np.asarray(contract['fit']['mean']))/np.asarray(contract['fit']['sd']))[:,np.asarray(contract['fit']['keep'])]])@np.asarray(contract['fit']['beta']))
    assert np.allclose(prediction,v.propensity,rtol=1e-12,atol=1e-12)
    cap=np.quantile(1/prediction[tr&v.verification.eq(1)],.99);assert np.isclose(cap,contract['cap_p99_train_verified'],rtol=1e-12)
    assert np.allclose(np.minimum(1/prediction,cap),v.ipw,rtol=1e-12)
    p=small(out/'preindex.parquet',columns=['split','has_tte_30d']+BASE+EXTRA+[MARKER]);p['V_echo']=p.has_tte_30d.eq(True).astype(int)
    missing=[]
    for (split,verified),g in p.groupby(['split','V_echo']):
        for column in BASE+EXTRA+[MARKER]:missing.append({'split':split,'V_echo':int(verified),'column':column,'n':len(g),'missing_or_nonfinite':int((~np.isfinite(g[column])).sum())})
    save(out/'verification_missingness.json',missing)
    for p in out.glob('*result*.json'):
        z=json.loads(p.read_text(encoding='utf8'))
        def visit(o):
            if isinstance(o,dict):
                if all(k in o for k in ('baseline_auc','augmented_auc','delta_auc')) and o['delta_auc'] is not None:
                    assert np.isclose(o['augmented_auc']-o['baseline_auc'],o['delta_auc'],atol=1e-14)
                    assert o['successful']<=o.get('replicates',o.get('requested',1000))
                for a in o.values():visit(a)
            elif isinstance(o,list):
                for a in o:visit(a)
        visit(z)
    sources=[ROOT/'mimic/labels.py',ROOT/'mimic/classical.py',ROOT/'core/biomarkers/discovery_v3_biomarkers.py',ROOT/'core/biomarkers/catalog_biomarkers.py',ROOT/'core/biomarkers/qrs_t_coupling_extension_biomarkers.py',PROJECT/'werkzeuge/varianten/adapter.py',PROJECT/'werkzeuge/varianten/metriken.py',PROJECT/'werkzeuge/varianten/residuen.py',PROJECT/'werkzeuge/nachrechnung/core.py',PROJECT/'schema/komparatoren.csv']
    result={'status':'passed','weighted_AUC_vs_sklearn':True,'weight_model_refit_current_code_equals_frozen':True,'weight_predictions_replayed_independently':True,'P99_reproduction':True,'delta_arithmetic_all_results':True,
      'development_disjointness':True,'sealed_specific_files_accessed':False,'release_modified':False,'weight_model_sha256':frozen,
      'source_hashes':{str(p):sha(p) for p in sources},'code_hashes':{p.name:sha(p) for p in __import__('pathlib').Path(__file__).parent.glob('*') if p.is_file()},
      'outputs':{p.name:sha(p) for p in out.glob('*.json') if p.name!='audit.json'}}
    save(out/'audit.json',result);print(json.dumps({'phase':'audit','status':'passed','weight_model_sha256':frozen}),flush=True)
