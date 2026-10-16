"""Nur explizite Entwicklungsprojektionen; keine Suche in Testverzeichnissen."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import check_reference
import argparse
import json
from pathlib import Path
from werkzeuge.nachrechnung.resources import guard, Budget
from werkzeuge.b_vorbereitung.runner import sha

OUT=Path(str(data_path("b.aw131")))
OLD=Path(str(data_path("b.b_vorbereitung_aw121_a155_2026_10_04_d6e70e")))
BASE=Path(str(data_path("b.b_vorbereitung")))

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def inspect():
    import pyarrow.parquet as pq
    sources=[OLD/'reduced_lvef_strict_private.parquet',OLD/'echo_evidence_private.parquet',OLD/'preindex.parquet']
    sources += [BASE/f'{bef}_old.parquet' for bef in ('BEF-0042','BEF-0081','BEF-0049','BEF-0075','BEF-0087')]
    result={}
    for p in sources:
        guard(); pf=pq.ParquetFile(p)
        result[str(p)]={'columns':pf.schema_arrow.names,'rows':pf.metadata.num_rows,'sha256':sha(p)}
    save(OUT/'schemas.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def projected(p,columns=None):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    pf=pq.ParquetFile(p);cols=pf.schema_arrow.names if columns is None else columns
    guard()
    return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=ds.field('split').isin(['train','validation'])).to_pandas()

def enrich(f,manifest):
    import pyarrow.parquet as pq
    targets={'features.parquet':['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive'],
             'context.parquet':['sinus_selector','ntprobnp_nearest','ntprobnp_reference_upper','ntprobnp_delta_hours','ntprobnp_nearest_censoring','ntprobnp_unit']}
    origins=[]
    for filename,wanted in targets.items():
        p=BASE/filename;expected=manifest['cache_sha256'][str(p)]
        if sha(p)!=expected:raise ValueError('Zusatzprojektion geändert')
        schema=pq.ParquetFile(p).schema_arrow.names
        cols=[c for c in wanted if c not in f and c in schema]
        if not cols:continue
        z=projected(p,['subject_id','study_id','split']+cols)
        f=f.merge(z,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
        origins.append({'file':str(p),'sha256':expected,'columns':cols,'join':'left one_to_one subject_id/study_id/split; train+validation'})
    return f,origins

def export():
    import gc, hashlib
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .daten import HYPOTHESES,masks,classify_lvef
    from .statistik import scaling,raw_design,fixed_design,separation
    from werkzeuge.nachrechnung.core import assert_development,logistic_fit
    from scipy.special import expit
    project=Path(__file__).resolve().parents[2]
    spec=project/'nachrechnung/SPEZIFIKATION_B_FREEZE_v3.md'
    assert sha(spec)=='ad05f61de954968c5d2d584687f461282bf7a73b40aca2e6fa6f658ec60285a1'
    binding=json.loads((OLD/'clinical_binding.json').read_text(encoding='utf8'))
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf8'))
    data={};flows={};origins={}
    for bef,ep,marker,comp,direction in HYPOTHESES:
        source=OLD/'reduced_lvef_strict_private.parquet' if bef=='BEF-0063' else BASE/f'{bef}_old.parquet'
        expected=binding['projections_sha256'][source.name] if bef=='BEF-0063' else manifest['cache_sha256'][str(source)]
        assert sha(source)==expected
        f=projected(source);assert_development(f)
        # Alte RV-Projektion hat keine Echo-ID: statt unvollständigem Ersatz
        # den bereits attestierten AW121-RV-Index verwenden.
        if bef=='BEF-0087':
            source=OLD/'echo_rv_systolic_dysfunction_private.parquet'
            expected=binding['projections_sha256'][source.name];assert sha(source)==expected
            f=projected(source);assert_development(f)
        f,enrichment=enrich(f,manifest)
        z,good,flow=masks(f,ep,marker,comp)
        xx,names=raw_design(z.loc[good],marker,comp)
        try:
            params=scaling(xx,names)
            params['status']='gebunden'
        except ValueError as exc:
            params={'status':'nicht_schaetzbar','reason':str(exc),'n':int(good.sum())}
        strong_good=good&False
        try:
            _,strong_good,strong_flow=masks(f,ep,marker,comp,strong=True)
            xs,snames=raw_design(z.loc[strong_good],marker,comp,strong=True)
            strong_params=scaling(xs,snames);strong_params['status']='gebunden'
            # Primäre Marker-SD bleibt auch auf B4-Fallbasis fest.
            if params.get('status')=='gebunden':
                strong_params['mean'][-1]=params['mean'][-1]
                strong_params['sd'][-1]=params['sd'][-1]
        except (ValueError,KeyError,AttributeError) as exc:
            strong_params={'status':'nicht_schaetzbar','reason':str(exc)}
            strong_flow={'status':'nicht_schaetzbar','reason':str(exc)}
        private=OUT/f'{bef}_entwicklung.parquet'
        pq.write_table(pa.Table.from_pandas(f,preserve_index=False),private,row_group_size=4096)
        origin={'run':'A155-2026-10-04-d6e70e','file':str(source),'sha256':expected,
                'filter':'train+validation; age>=18; first_verified_ecg_reference_result_blind; eligible/known; Zeitgate v3 nur Stundenendpunkte; Complete-case B3+marker; endpunktspezifischer Referenzvertrag',
                'n_index':len(f),'n_complete':int(good.sum()),'script_sha256':sha(__file__),'additional_sources':enrichment}
        params['origin']=origin;origins[bef]=origin
        strong_params['origin']=dict(origin,filter=origin['filter'].replace('B3+marker','B4+marker'),n_complete=strong_params.get('n',0))
        flows[bef]={'B3':flow,'B4':strong_flow,'counts_by_split':{s:{'B3_n':int((good&f.split.eq(s)).sum()),'B4_n':int((strong_good&f.split.eq(s)).sum()),'positive':int((good&f.split.eq(s)&(f.preferred_lvef.lt(40) if ep=='reduced_lvef_strict' else f.label_state.eq('positive'))).sum()),'B4_positive':int((strong_good&f.split.eq(s)&(f.preferred_lvef.lt(40) if ep=='reduced_lvef_strict' else f.label_state.eq('positive'))).sum())} for s in ('train','validation')}}
        data[bef]={'endpoint':ep,'marker':marker,'comparator':comp,'direction':direction,'scaling_B3':params,'scaling_B4':strong_params,
                   'input_development':{'path':str(private),'sha256':sha(private)},
                   'measurement_attestation':{'status':'STOP','reason':'vollständige claimbezogene Messgateakte/Quellenattestation nach §12(c) nicht als ausführbares Attest vorgelegt'}}
        del f,z,xx;gc.collect()
    # Verifikationsfit ohne Marker-Missingnessfilter: der AW121-Fit war auf
    # Marker-vollständiger Basis erstellt; seine Parameter sind nicht austauschbar.
    cols=['subject_id','study_id','split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',
          'classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude',
          'classical_right_axis_deviation','spatial_qrs_t_angle_deg','has_tte_30d','has_tte_7d','nearest_tte_measurement_id','nearest_tte_delta_hours']
    assert sha(OLD/'preindex.parquet')==json.loads((OLD/'index_flow.json').read_text(encoding='utf8'))['preindex_sha256']
    pre=projected(OLD/'preindex.parquet',cols);assert_development(pre);n_pre=len(pre)
    assert not pre.subject_id.duplicated().any()
    null=pre.has_tte_30d.isna()
    if (null&(pre.nearest_tte_measurement_id.notna()|pre.nearest_tte_delta_hours.notna())).any():raise ValueError('unbelegte sparse Echo-Nulls')
    # raw_design verlangt letzten Marker; temporärer Echo-Indikator wird als
    # Platzhalter angehängt und sofort wieder entfernt, niemals als Prädiktor.
    pre['__verification']=pre.has_tte_30d.eq(True).astype(float)
    x,names=raw_design(pre,'__verification',strong=True);x=x[:,:-1];names=names[:-1]
    good=np.isfinite(x).all(1);x=x[good];pre=pre.loc[good].copy();y=pre.__verification.to_numpy(float)
    train=pre.split.eq('train').to_numpy();mean=x[train].mean(0);sd=x[train].std(0);keep=sd>0
    zz=np.column_stack([np.ones(len(x)),(x[:,keep]-mean[keep])/sd[keep]])
    ver_origin={'run':'A155-2026-10-04-d6e70e','file':str(OLD/'preindex.parquet'),'sha256':sha(OLD/'preindex.parquet'),
                'filter':'first technical adult index; complete B4 without marker; fit train; P99 verified train only; sparse missing TTE join=0',
                'n_index':n_pre,'n_complete':len(pre),'n_train_fit':int(train.sum()),'n_train_verified':int((train&(y==1)).sum()),'script_sha256':sha(__file__)}
    ver={'status':'gebunden','names':names,'mean':mean.tolist(),'sd':sd.tolist(),'keep':keep.tolist(),'origin':ver_origin}
    if separation(zz[train],y[train])!='keine':raise ValueError('Separation Verifikation')
    beta=logistic_fit(zz[train],y[train],np.ones(train.sum()));prop=expit(zz@beta)
    if (prop<=0).any() or not np.isfinite(prop).all():raise ValueError('Propensity nicht positiv/endlich')
    cap=float(np.quantile(1/prop[train&(y==1)],.99,method='linear'))
    ver.update({'beta':beta.tolist(),'p99_train_verified':cap,'ddof':0,'marker_excluded':True,
                'fit_scope':'train','bootstrap_scope':'train verification fit and train cap refit, fixed scaling'})
    # Parameter aus gebundenem AW121-Rezept; keine Erweiterung des Signalpiloten.
    residual_path=OLD/'residual_fit_frozen.json'
    residual=json.loads(residual_path.read_text(encoding='utf8'))
    flow_signal=json.loads((OLD/'signals_result.json').read_text(encoding='utf8'))['_flow']
    assert flow_signal['common_n']==sum(flow_signal['by_fold_requested'].values())
    n_residual_train=flow_signal['by_fold_requested']['train']
    residual={'status':'gebunden_pilot','parameters':residual,'origin':{'file':str(residual_path),'sha256':sha(residual_path),
               'input_request_file':str(OLD/'signal_request_private.json'),'input_request_sha256':sha(OLD/'signal_request_private.json'),
               'input_values_sha256':sha(OLD/'signal_values_private.json'),'filter':'AW121 gemeinsames Signalset; nur train; 200 angefragte train-Personen, vollständiges Set 200 laut signals_result.json/_flow',
               'n_train':n_residual_train,'aggregate_flow_file':str(OLD/'signals_result.json'),'aggregate_flow_sha256':sha(OLD/'signals_result.json')},
               'limitation':'Signalpilot; keine implizite Erweiterung auf Entwicklungs-Vollnenner; Anwendung nur bei attestiertem Signalvertrag'}
    package={'schema_version':'AW131-3','status':'Entwicklungsparameter; Messatteste/Testfreigabe separat erforderlich','spec_sha256':sha(spec),
             'role':'entwicklung','hypotheses':data,'verification':ver,'residual':residual,
             'seeds':{'bootstrap':1282026,'replicates':1000,'generator':'PCG64','sequence':'SeedSequence([1282026,r]), r=0..999','origin':'Freeze §6; Planwerte, keine empirischen Schätzer'},
             'test_authorization_file':str(project/'werkzeuge/b_test/FREIGABE_K.json'),
             'test_capability':'Adapter AW131 implementiert; keine Testfreigabe. Quellen-/Mess-/K-Atteste erforderlich; optionale Signalvarianten ohne vollständige Entwicklungs-SD STOP.',
             'test_inputs':{},'sources':origins}
    # Teilobjekthashes schließen sämtliche Zahlen samt Herkunft ein.
    for value in list(data.values())+[ver,residual]:
        value['binding_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    from .extras import export_extras
    package=export_extras(package)
    for entry in data.values():
        entry.pop('binding_sha256',None)
        entry['binding_sha256']=hashlib.sha256(json.dumps(entry,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    save(project/'werkzeuge/b_test/parameter_entwicklung.json',package)
    save(OUT/'parameter_fallfluss.json',flows)
    print(json.dumps({'phase':'export','complete_n':{k:v['scaling_B3']['n'] for k,v in data.items()},'verification':ver_origin,'p99':cap},ensure_ascii=False),flush=True)

def audit():
    import numpy as np
    from .daten import masks,HYPOTHESES
    from werkzeuge.b_vorbereitung.clinical import reference,eligible,complete
    project=Path(__file__).resolve().parents[2]
    package=json.loads((project/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))
    f=projected(OLD/'reduced_lvef_strict_private.parquet')
    evidence=projected(OLD/'echo_evidence_private.parquet')
    f=reference(f,evidence,'reduced_lvef_strict')
    pilot=eligible(f,'reduced_lvef_strict')&~f.lvef_conflict_gt_10.fillna(True)&np.isfinite(f.preferred_lvef)&complete(f,'qrs_t_normalized_discrete_frechet')
    computed={s:{'n':int((pilot&f.split.eq(s)).sum()),'positive_lt40':int((pilot&f.split.eq(s)&f.preferred_lvef.lt(40)).sum()),'positive_le40':int((pilot&f.split.eq(s)&f.preferred_lvef.le(40)).sum())} for s in ('train','validation')}
    check_reference("export.aw121_case_set", computed, what="AW121 case set not reproduced")
    result={'AW121_fallset_reproduziert':computed,'expected_origin':'nachrechnung/B_VORBEREITUNG_AW121.md §2; clinical_results.json strict_lt40/inclusive_le40 counts',
            'AW121_report_sha256':sha(project/'nachrechnung/B_VORBEREITUNG_AW121.md'),'AW121_results_sha256':sha(OLD/'clinical_results.json'),
            'strict_freeze':{k:v['scaling_B3']['n'] for k,v in package['hypotheses'].items()},
            'interpretation':'v3-Echofenster ohne Stundengate. AW121 war B4-vollständig; B3 kann mehr Fälle enthalten. OR/AUC unterscheiden sich durch Fitrolle und SD-Vertrag.',
            'source_sha256':sha(OLD/'reduced_lvef_strict_private.parquet'),'script_sha256':sha(__file__)}
    save(OUT/'AW121_abgleich.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

def bind():
    import sys,platform,importlib.metadata
    project=Path(__file__).resolve().parents[2]
    own=sorted((project/'werkzeuge/b_test').glob('*.py'))+sorted((project/'werkzeuge/b_test/tests').glob('*.py'))+[(project/'werkzeuge/b_test/start.ps1')]
    dependencies=[]
    for folder in ['nachrechnung','b_vorbereitung','varianten']:
        dependencies+=sorted((project/'werkzeuge'/folder).glob('*.py'))
    release=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
    dependencies += [release/x for x in ['core/biomarkers/discovery_v3_biomarkers.py','core/biomarkers/catalog_biomarkers.py','core/vcg_geometry.py','mimic/labels.py','mimic/feature_record.py']]
    dependencies += [project/'schema/komparatoren.csv',project/'schema/endpunktgruppen.csv']
    files={str(p):sha(p) for p in own+dependencies}
    packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','pyarrow','scipy','scikit-learn','pytest','psutil']}
    environment={'python':sys.version,'executable':sys.executable,'executable_sha256':sha(sys.executable),'platform':platform.platform(),'packages':packages,
                 'threads':{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}}
    envpath=project/'werkzeuge/b_test/umgebung.json';save(envpath,environment)
    files[str(envpath)]=sha(envpath)
    files[str(project/'werkzeuge/b_test/parameter_entwicklung.json')]=sha(project/'werkzeuge/b_test/parameter_entwicklung.json')
    save(project/'werkzeuge/b_test/bindung.json',{'status':'Entwicklungsbindung; KEINE Testfreigabe','spec_sha256':sha(project/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md'),'files':files})
    table='| Datei | SHA-256 |\n|---|---|\n'+'\n'.join('| '+p+' | `'+h+'` |' for p,h in files.items())+'\n'
    (OUT/'hashes.md').write_text(table,encoding='utf8')
    print(json.dumps({'phase':'bind','files':len(files),'runner_sha256':sha(project/'werkzeuge/b_test/runner.py'),'package_sha256':sha(project/'werkzeuge/b_test/parameter_entwicklung.json'),'environment_sha256':sha(envpath)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inspect','export','audit','bind']);a=p.parse_args()
    memory=guard();OUT.mkdir(parents=True,exist_ok=True)
    with Budget(OUT) as budget:
        if a.phase=='inspect':inspect()
        elif a.phase=='export':export()
        elif a.phase=='audit':audit()
        else:bind()
    save(OUT/f'resources_{a.phase}.json',{'start':memory,**budget.summary()})

if __name__=='__main__':main()
