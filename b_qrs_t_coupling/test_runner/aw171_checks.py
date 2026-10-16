"""Synthetische Integration und dokumentierte Grenzen; kein Testdatenzugriff."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json, copy, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from werkzeuge.b_test import aw171_kernels as k
from werkzeuge.b_test import aw171_mapping as mp
from werkzeuge.b_test import aw171_contracts as contract
from werkzeuge.b_test.statistik import raw_design, fixed_design
from werkzeuge.nachrechnung.core import logistic_fit, auc

def run(a):
    p=a.load(a.OUT/'parameters.json'); mapping=a.load(a.OUT/'mappings.json')
    ledger_before=a.sha(a.ROOT/'koordination/TESTZUGRIFFE.md')
    out=a.OUT/'synthetic'; out.mkdir(exist_ok=True)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1712026,0])))
    n=1400
    f=pd.DataFrame({'unit_index':np.arange(n),'split':'synthetic_testschema','age':rng.uniform(25,90,n),
                    'sex_numeric':rng.binomial(1,.5,n),'heart_rate':rng.normal(80,15,n),
                    'qrs_duration_ms':rng.uniform(80,150,n),'qtc_fridericia_ms':rng.normal(425,25,n),
                    a.A:rng.uniform(20,150,n),'marker':rng.normal(1.6,.6,n)})
    f['marker2']=.6*f.marker+.4*rng.normal(1.6,.6,n)
    f[a.F]=f.marker
    h4scale=p['endpoints']['H4']['markers'][a.M]['scale']
    f[a.M]=h4scale['mean'][-1]+(f.marker2-1.6)/.6*h4scale['sd'][-1]
    f['y']=rng.binomial(1,expit(-.3+.4*(f.marker-1.6)+.3*(f[a.A]-80)/40)).astype(float)
    f['preferred_lvef']=np.where(f.y==1,rng.uniform(20,39.9,n),rng.uniform(40,70,n))
    f['echo_available']=rng.random(n)>.15
    f['sinus']=rng.random(n)>.3; f['af']=rng.random(n)<.3
    f['rbbb']=rng.random(n)<.25; f['lbbb']=rng.random(n)<.25
    f['setting']=rng.choice(['ED','inpatient','both','outside','unknown'],n)
    f['anchor_year_group']=rng.choice(['2008-2010','2011-2013','unknown'],n)
    f['pacing']=[mp.pacing(t) for t in rng.choice(['sinus rhythm','ventricular paced rhythm','no pacemaker','history of pacemaker','possible pacing'],n)]
    for cl in p['medication_patterns']:
        f[cl]=rng.choice(['exposed','observed_no_record','unknown'],n,p=[.3,.5,.2])
    # E: only; no synthetic row data in project.
    f.to_parquet(out/'input.parquet',index=False)
    checks=[]; tables={}; start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    def check(name,fn):
        fn(); checks.append({'name':name,'status':'bestanden'})
    def stops(name,fn,reason):
        try:fn()
        except (k.Stop,ValueError) as ex:
            assert reason in str(ex),(name,str(ex)); checks.append({'name':name,'status':'bestanden','expected_stop':str(ex)}); return
        raise AssertionError('STOP fehlt: '+name)
    def table(name,ep,w,marker,fn):
        a.guard()
        source=out/('schema_input.parquet' if '_schema_' in name else 'input.parquet')
        provenance={'analysis_id':'AW171_SYNTHETIC','endpoint':ep,'W':w,'marker':marker,'population':'synthetic_testschema',
                    'source':source.as_posix(),'source_sha256':a.sha(source),
                    'run':'rein_synthetisch','filter':'ein unveraenderlicher Index pro synthetischer Person',
                    'scale_sha256':a.sha(a.OUT/'parameters.json'),'script_sha256':a.sha(a.ROOT/'werkzeuge/b_test/aw171_kernels.py'),
                    'seed':'PCG64 SeedSequence([1712026,0]); R1 [1282026,r]', 'specification_sha256':a.sha(a.SPEC)}
        try:result=fn(); status='schaetzbar'
        except (k.Stop,ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
            result={'reason':type(ex).__name__+':'+str(ex)};status='STOP'
        obj={**provenance,'status':status,'result':result}
        tables[name]=obj; a.save(out/(name+'.json'),obj)
        # Long table: numeric aggregates only, no coefficient arrays/trace row leakage.
        rows=[]
        def flatten(v,key=''):
            if isinstance(v,dict):
                for kk,vv in v.items():flatten(vv,key+'/'+str(kk))
            elif isinstance(v,list):
                for i,vv in enumerate(v):flatten(vv,key+'/'+str(i))
            else:rows.append({**provenance,'status':status,'aggregate_key':key,'value':v})
        flatten(result); pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
        return obj
    # Pre-open barriers: rejected path does not exist; rejection must precede stat.
    stops('physische Sperrbarriere vor stat/hash/footer',lambda:a.bind(Path(str(data_path("b.mimic_validation_runs_never_open_test_parquet")))),'Sperrbarriere_Pfad')
    check('keine akzeptierten Rohdatensplits',lambda:k.require(set(f.split)=={'synthetic_testschema'},'Split'))
    argv=sys.argv.copy();sys.argv=[argv[0],'partition']
    try: partition_result=a.main()
    finally:sys.argv=argv
    check('Produktivstart STOP vor Eingabe',lambda:k.require(partition_result==77,'Partitionstart'))
    parser_cases={None:'unknown','sinus rhythm':'no_mention','no pacemaker':'negated','history of pacemaker':'historical',
                  'possible ventricular pacing':'suspected','ventricular paced rhythm':'positive',
                  'no pacemaker; ventricular paced rhythm':'ambiguous','history of pacemaker; currently v-paced':'positive',
                  'cannot exclude paced rhythm':'suspected'}
    check('Parser Negation/Historie/Vermutung/Klauseln',lambda:k.require(all(mp.pacing(t)==v for t,v in parser_cases.items()),'Parser'))
    check('W21 unbekannt eigene Kategorie',lambda:k.require(mp.medication(False,None)=='unknown' and mp.medication(True,None)=='exposed','Medikation'))
    check('Setting outside ist nicht ambulant',lambda:k.require(mp.setting(False,False)=='outside' and mp.setting(True,True)=='both','Setting'))
    stops('neuer Messartwert',lambda:mp.measurement('invented',mapping['measurement']),'Messart')
    check('ICD Normalisierung/Version',lambda:k.require('heart_failure' in mp.icd(' i50.1 ',10,mapping['icd_groups']),'ICD'))
    stops('ICD unbekannte Version',lambda:mp.icd('I50',11,mapping['icd_groups']),'ICD_Version')
    # Vollständiges Testschema am eigenen synthetischen Kandidatenpool.
    schema=f.copy();schema['subject_id']=schema.unit_index+1;schema['study_id']=schema.unit_index+10000
    schema['ecg_time']=pd.Timestamp('2000-01-01')+pd.to_timedelta(schema.unit_index,unit='s')
    schema['sex']=schema.sex_numeric.map({0:'F',1:'M'});schema['label_state']=np.where(schema.y==1,'positive','negative')
    schema['eligibility_state']='eligible';schema['extraction_status']='ok'
    schema['index_selection_strategy']='first_verified_ecg_reference_result_blind'
    schema['ehr_time_alignment_confirmatory_eligible']=False
    schema['reference_available']=True;schema['encounter_reference_available']=True
    schema['time_precision']='timestamp';schema['time_delta_hours']=0.;schema['nearest_tte_delta_hours']=0.
    schema['nearest_tte_measurement_id']=np.arange(n)+20000
    schema['evidence_ids_json']=[json.dumps([int(v)]) for v in schema.nearest_tte_measurement_id]
    schema['lvef_conflict_gt_10']=False
    schema.to_parquet(out/'schema_input.parquet',index=False)
    D1,C1,flow1=contract.case_sets(schema,'reduced_lvef_strict',a.F)
    D4,C4,flow4=contract.case_sets(schema,'cardiomyopathy_icd',a.M)
    check('vollstaendiges H1/H4 Testschema und Referenzmasken',lambda:k.require(len(C1)==n and len(C4)==n,'Testschema'))
    table('H1_schema_flow','H1','W25',a.F,lambda:flow1)
    table('H4_schema_flow','H4','W25',a.M,lambda:flow4)
    splitbad=schema.copy();splitbad['split']='test'
    stops('Testschema verweigert Rohsplit test',lambda:contract.case_sets(splitbad,'cardiomyopathy_icd',a.M),'Sperrbarriere_Split')
    stops('Index-/Complete-case-Abgleich',lambda:contract.case_sets(schema,'cardiomyopathy_icd',a.M,expected={'index_n':n,'complete_n':n-1}),'Abgleich')
    lost=schema.copy();lost.loc[0,a.M]=np.nan
    later=schema.iloc[[0]].copy();later.study_id+=200000;later.ecg_time+=pd.Timedelta(days=1)
    lost=pd.concat([lost,later],ignore_index=True)
    _,cc,_=contract.case_sets(lost,'cardiomyopathy_icd',a.M)
    check('kein Ersatzindex bei Marker-Missingness',lambda:k.require(len(cc)==n-1 and not cc.subject_id.eq(1).any(),'Ersatzindex'))
    stops('Vertrag fehlt vor Leseraufruf',lambda:contract.preflight({}),'Vertrag_')
    valid_contract={field:True for field in contract.REQUIRED_CONTRACTS}
    valid_contract.update(partition_role='previously_opened_internal',sealed=False,scaling_role='development_fixed')
    check('vollstaendiger abstrakter Vertrag',lambda:contract.preflight(valid_contract))
    stops('gesperrte Menge auch bei anderem Vertrag verweigert',lambda:contract.preflight({**valid_contract,'sealed':True}),'gesperrte_Testmenge')
    stops('Skalierung testseitig verweigert',lambda:contract.preflight({**valid_contract,'scaling_role':'test'}),'Skalierung')
    stops('Code-/Inputhashabweichung',lambda:contract.match_hash('a','b','Code'),'Hash_Code')
    stops('fehlende Quellspalte',lambda:contract.source_columns(['split'],['split','age']),'fehlende_Spalte')
    stops('nichtpositive Propensity',lambda:contract.weight_gate(np.array([0.,.4])),'Propensity')
    stops('W7 fehlende Zelle',lambda:contract.w7_gate(p['w7'][:-1]),'acht_Zellen')
    gatebad=copy.deepcopy(p['w7']);gatebad[0]['delta_auc']=-.0001
    stops('W7 Gegenrichtung/negatives Inkrement',lambda:contract.w7_gate(gatebad),'geschlossen')
    # Exactly-one-Zugriffslifecycle wird unter E: geprueft, auch bei Abbruch.
    clock=lambda:time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    for scenario in ['regular','abort']:
        ledger=out/('lifecycle_'+scenario+'.md')
        ledger.write_text('',encoding='utf8')
        try:
            with contract.access_phase(ledger,'AW171_SYNTHETIC','spec','code','params',clock):
                if scenario=='abort':raise k.Stop('synthetisch')
        except k.Stop:pass
        check('Exactly-one Beginn/Ende '+scenario,lambda ledger=ledger:k.require(ledger.read_text(encoding='utf8').count('Beginn')==1 and ledger.read_text(encoding='utf8').count('Ende')==1,'Lifecycle'))
    cp=out/'checkpoint_fixture.json';seal={'spec':'a','code':'b','parameters':'c'}
    contract.checkpoint(cp,{'next':25},seal)
    check('Checkpoint identische Fortsetzung',lambda:k.require(contract.resume(cp,seal)['next']==25,'Checkpoint'))
    stops('Checkpoint bei Hashwechsel verweigert',lambda:contract.resume(cp,{**seal,'code':'changed'}),'Hashabweichung')
    scale=copy.deepcopy(p['endpoints']['H1']['markers'][a.F]['scale'])
    raw,_=raw_design(f,'marker'); z=fixed_design(raw,scale); y=f.y.to_numpy()
    check('feste Entwicklungsskalierung',lambda:k.require(np.array_equal(z[:,1:],(raw-np.array(scale['mean']))/np.array(scale['sd'])),'Skalierung'))
    stops('kleine Strata',lambda:k.association(z[:110],y[:110]),'kleine_Strata')
    sep=np.column_stack([np.ones(400),np.r_[np.zeros(200),np.ones(200)]])
    stops('vollstaendige Separation',lambda:k.solver(sep,sep[:,1]),'Separation')
    quasi=np.column_stack([np.ones(600),np.r_[np.full(100,-1.),np.zeros(400),np.ones(100)]])
    qy=np.r_[np.zeros(300),np.ones(300)]
    stops('Quasiseparation',lambda:k.solver(quasi,qy),'Separation')
    stops('Quartil-Ties',lambda:k.quartiles(np.ones(400)),'Quartil_Ties')
    stops('Rangdefizit',lambda:k.solver(np.column_stack([z,z[:,1]]),y),'Rang')
    stops('negative Gewichte',lambda:k.solver(z,y,-np.ones(n)),'Gewichte')
    tiny=np.where(y==1,1.,1.); tiny[:]=1e-12; tiny[:100]=1.
    stops('ESS klein',lambda:k.association(z,y,tiny,ess=True),'ESS')
    old={'beta':.2,'or':float(np.exp(.2)),'auc':.7,'wald':[.1,.3],'n':1400,'weight_parameters':{'p99':3.}}
    changed=copy.deepcopy(old);changed['beta']=np.nextafter(.2,1.)
    stops('W10 Punktwert ein ULP Abweichung',lambda:k.identity_gate(old,changed,'union','union'),'Punktwertabweichung')
    stops('W10 Originalunionabweichung',lambda:k.identity_gate(old,old,'other','union'),'Originalunion')
    stops('W10 Originalunionbeleg fehlt',lambda:k.identity_gate(old,old,'union',None),'Originalunion')
    check('W10 Bitidentitaet positiv',lambda:k.identity_gate(old,copy.deepcopy(old),'union','union'))
    # Frequenzgewichte gleich expliziter Zeilenvervielfachung für Solverkoeffizienten.
    w=rng.integers(1,3,n).astype(float); b=k.solver(z,y,w); ix=np.repeat(np.arange(n),w.astype(int))
    check('Solver Gewichte vs Zeilenduplikation',lambda:k.require(np.max(np.abs(b-k.solver(z[ix],y[ix])))<2e-7,'Gewichtsfit'))
    bn,tr=k.solver(z,y,diagnostics=True); check('Score/N und Score/sum(w)',lambda:k.require(tr[-1]['score_n']<1e-7 and tr[-1]['score_weight']<1e-7,'Score'))
    check('Schwelle mit Ties konservativ',lambda:k.require(k.threshold90(np.r_[np.zeros(91),np.ones(9)],np.zeros(100))==0.,'Schwelle'))
    table('W10_solver','H1','W10',a.F,lambda:{'identity':'synthetisch bestanden; historische Union bleibt STOP',
            'diagnostics':tr,'old_beta':logistic_fit(z,y,np.ones(n)).tolist(),'new_beta':bn.tolist(),
            'historical_status':'STOP_Originalunion_Altpunktdaten_nicht_geoeffnet'})
    historic={r:{'beta':.2,'or':float(np.exp(.2))} for r in range(1000) if r!=7}
    errors={7:{'class':'Konvergenz','reason':'synthetisch injizierter Solverabbruch'}}
    def repair(r,diagnose):
        if diagnose:return {'old_failure_is_solver':True,'replicate':r,'note':'synthetisches Fehlerfixture; keine historische Ursachendiagnose'}
        return {'beta':.2,'or':float(np.exp(.2))}
    repaired=k.repair_bootstrap(old,copy.deepcopy(old),'union','union',historic,errors,repair)
    check('W10 nur fehlendes Replikat gerettet, Altpunkt unveraendert',lambda:k.require(repaired['rescued']==[7] and repaired['new_valid']==1000 and repaired['old_point']==repaired['new_point'] and historic[8]==repaired['rows'][8],'W10_Repair'))
    stops('W10 keine Solverursache',lambda:k.repair_bootstrap(old,old,'union','union',historic,{7:{'class':'Rang'}},repair),'keine_Solverursache')
    table('W10_repair_fixture','H1','W10',a.F,lambda:{key:value for key,value in repaired.items() if key not in ['rows','rescued']})
    fixed={}
    for model,size in [('B2',7),('B3',8),('B3_marker',9)]:
        beta=k.solver(z[:,:size],y);fixed[model]={'beta':beta.tolist(),'threshold':k.threshold90(z[:,:size]@beta,y)}
    cuts=k.quartiles(f.marker)
    # Smoke counts are intentionally below inferential gates, never override minimum.
    smoke=8
    for ep,primary in [('H1',a.F),('H4',a.M)]:
        scale=p['endpoints'][ep]['markers'][primary]['scale']
        rr,_=raw_design(f,primary);z=fixed_design(rr,scale)
        fixed={}
        for model,size in [('B2',7),('B3',8),('B3_marker',9)]:
            beta=k.solver(z[:,:size],y);fixed[model]={'beta':beta.tolist(),'threshold':k.threshold90(z[:,:size]@beta,y)}
        cuts=k.quartiles(f[primary])
        table(ep+'_flow',ep,'W25',primary,lambda:{'candidate_recordings':n,'candidate_persons':n,'adult':n,
              'technical':n,'reference_valid':int(f.echo_available.sum()) if ep=='H1' else n,
              'complete_B3_marker':int(f.echo_available.sum()) if ep=='H1' else n,
              'echo_parallel_h4':int(f.echo_available.sum()) if ep=='H4' else None,
              'note':'synthetische Flussflags, keine Rekonstruktion echter Vorindexpopulation'})
        table(ep+'_table1',ep,'W30',primary,lambda:{'continuous':{col:{'n':n,'missing':int(f[col].isna().sum()),'mean':float(f[col].mean()),
                'sd':float(f[col].std(ddof=1)),'median':float(f[col].median()),'q1':float(f[col].quantile(.25)),'q3':float(f[col].quantile(.75))}
                for col in ['age','heart_rate','qrs_duration_ms','qtc_fridericia_ms']},
                'categorical':{col:f[col].value_counts(dropna=False).to_dict() for col in ['sex_numeric','sinus','af','rbbb','lbbb','pacing','setting','anchor_year_group']},
                'denominator':n,'D_equals_C_in_complete_synthetic_fixture':True})
        table(ep+'_prediction',ep,'W20',primary,lambda:{'point':k.prediction(z,y,fixed),
              'bootstrap':k.bootstrap(lambda w,r:k.prediction(z,y,fixed,w),np.arange(n),n,smoke,950,a.guard),
              'thresholds':{model:v['threshold'] for model,v in fixed.items()},'fixed_synthetic_model':True})
        for mark in [a.F,a.M]:
            rr,_=raw_design(f,mark);zz=fixed_design(rr,p['endpoints'][ep]['markers'][mark]['scale'])
            table(ep+'_cross_'+('F' if mark==a.F else 'M'),ep,'W7',mark,lambda zz=zz:{'point':k.association(zz,y),
                 'bootstrap':k.bootstrap(lambda w,r:k.association(zz,y,w),np.arange(n),n,smoke,950,a.guard),'gate':'AE181_offen'})
        strata={'sinus_ohne_positiven_Stimulationsvermerk':f.sinus&f.pacing.ne('positive'),'AF':f.af,'RSB':f.rbbb,'LSB':f.lbbb,
                'QRS_ge120':f.qrs_duration_ms>=120,'Stimulationsvermerk':f.pacing.eq('positive'),
                'QRS_lt120_ohne_Block':(f.qrs_duration_ms<120)&~f.rbbb&~f.lbbb}
        table(ep+'_rhythm_counts',ep,'W24',primary,lambda:{name:{'n':int(mask.sum()),'unknown_pacing':int(f.loc[mask,'pacing'].eq('unknown').sum())} for name,mask in strata.items()})
        for name,mask in strata.items():
            pos=np.flatnonzero(mask);zz=z[pos];yy=y[pos]
            table(ep+'_stratum_'+name,ep,'W24',primary,lambda zz=zz,yy=yy,pos=pos:{'point':k.association(zz,yy),
                 'bootstrap':k.bootstrap(lambda w,r:k.association(zz,yy,w),pos,n,smoke,950,a.guard)})
        def drugs():
            extras=np.column_stack([(f[cl]==category).to_numpy(float) for cl in p['medication_patterns'] for category in ['exposed','unknown']])
            # Marker bleibt letzte Spalte für Zielkoeffizient, B3 davor plus Dummies.
            zz=np.column_stack([z[:,:-1],extras,z[:,-1]])
            return {'categories':{cl:f[cl].value_counts().to_dict() for cl in p['medication_patterns']},
                    'point':k.association(zz,y),'unadjusted_same_set':k.association(z,y),
                    'bootstrap':k.bootstrap(lambda w,r:k.association(zz,y,w),np.arange(n),n,smoke,950,a.guard),'window_hours':[-168,0]}
        table(ep+'_medication',ep,'W21',primary,drugs)
        lvef=f.preferred_lvef if ep=='H1' else f.loc[f.echo_available,'preferred_lvef']
        table(ep+'_lvef',ep,'W12a',primary,lambda:{'bins':np.histogram(lvef[lvef<100],bins=np.arange(101))[0].tolist(),
                 'hundred':int(lvef.eq(100).sum()),'exact_40':int(lvef.eq(40).sum()),'denominator':len(lvef),
                 'lt40':int(lvef.lt(40).sum()),'ge40':int(lvef.ge(40).sum()),
                 'measurement':'unknown; keine numerische Ableitung'})
        def risks(w=None):
            w=np.ones(n) if w is None else w;beta=np.asarray(k.association(z,y,w)['coefficients']);prob=[]
            for median in cuts['medians']:
                xx=z.copy();xx[:,-1]=(median-scale['mean'][-1])/scale['sd'][-1]
                prob.append(float(np.average(expit(xx@beta),weights=w)))
            return {**{f'q{i+1}':v for i,v in enumerate(prob)},'q4_minus_q1':prob[-1]-prob[0]}
        table(ep+'_standardized',ep,'W26',primary,lambda:{'quartiles':cuts,'point':risks(),
             'development_mean':scale['mean'][-1],'development_sd':scale['sd'][-1],'ddof':scale['ddof'],
             'distribution':{str(cl):{'n':int(f.y.eq(cl).sum()),'median':float(f.loc[f.y.eq(cl),primary].median()),
                    'q1':float(f.loc[f.y.eq(cl),primary].quantile(.25)),'q3':float(f.loc[f.y.eq(cl),primary].quantile(.75))} for cl in [0,1]},
             'bootstrap':k.bootstrap(lambda w,r:risks(w),np.arange(n),n,smoke,950,a.guard)})
    check('H4 hat kein neues Echogate',lambda:k.require(tables['H4_flow']['result']['reference_valid']==n and tables['H1_flow']['result']['reference_valid']<n,'H4_Echogate'))
    scale=p['endpoints']['H1']['markers'][a.F]['scale']
    raw,_=raw_design(f,a.F);z=fixed_design(raw,scale)
    # W13 requires K list in real data; synthetic fixture explicitly not a clinical list.
    codes=[(['I42.0','I42.1'] if i%7==0 else ['I42.0'] if i%3==0 else ['I42.1']) if y[i]==1 else [] for i in range(n)]
    table('H4_icd_subcodes','H4','W13',a.M,lambda:{'I420_synthetic_n':sum('I42.0' in row for row in codes),
            'I421_synthetic_n':sum('I42.1' in row for row in codes),'overlap_n':sum(len(row)>1 for row in codes),'denominator':int(y.sum()),
            'normalization_test':'I42.0 -> I420; Mehrfachcodes nicht summieren'})
    for name in ['S1','S2']:
        s1=(y==1)|((np.arange(n)%11!=0)&(np.arange(n)%13!=0))
        selected=s1 if name=='S1' else ((y==1)|(s1&f.echo_available.to_numpy()&f.preferred_lvef.ge(50).to_numpy()))
        pos=np.flatnonzero(selected)
        rr,_=raw_design(f,a.M);z4=fixed_design(rr,p['endpoints']['H4']['markers'][a.M]['scale'])
        table('H4_restricted_'+name,'H4','W13',a.M,lambda pos=pos:{'fixture':'rein synthetisch; keine klinische Ausschlussliste',
             'point':k.association(z4[pos],y[pos]),'production_status':'STOP_ICD_Zusatzliste',
             'bootstrap':k.bootstrap(lambda w,r:k.association(z4[pos],y[pos],w),pos,n,smoke,950,a.guard)})
    def verification():
        # Nur synthetisches train-Verifikationsuniversum, eigenes technisches Indexset.
        v=rng.binomial(1,expit(.2+.3*z[:,1]));train=np.column_stack([z[:,:8],z[:,-1]])
        outcome=v==1
        result=k.verification(train,v,z[outcome],y[outcome],train[outcome])
        result['bootstrap']=k.verification_bootstrap(train,v,z[outcome],y[outcome],train[outcome],np.arange(n),n,np.flatnonzero(outcome),n,smoke,a.guard)
        return {'train_n':n,**result,
                'production_status':'STOP_train_Verifikationsuniversum_mit_Marker_fehlt'}
    table('H1_verification','H1','W19',a.F,verification)
    mf=f.copy(); mf.loc[rng.random(n)<.05,'heart_rate']=np.nan;mf.loc[rng.random(n)<.05,'qtc_fridericia_ms']=np.nan
    table('H1_missingness','H1','W18',a.F,lambda:{col:{'missing':int(mf[col].isna().sum()),'n':n} for col in ['heart_rate','qtc_fridericia_ms']})
    table('H1_no_qtc','H1','W18',a.F,lambda:k.association(np.delete(z,6,axis=1),y))
    def indicator():
        mm=mf.copy();extras=np.column_stack([mm[c].isna().to_numpy(float) for c in ['heart_rate','qtc_fridericia_ms']])
        for c in ['heart_rate','qtc_fridericia_ms']:mm[c]=mm[c].fillna(p['endpoints']['H1']['mi']['start'][c])
        rr,_=raw_design(mm,'marker');zz=fixed_design(rr,scale);zz=np.column_stack([zz[:,:-1],extras,zz[:,-1]])
        return k.association(zz,y)
    table('H1_missing_indicator','H1','W18',a.F,indicator)
    mi_recipe=p['endpoints']['H1']['mi']
    bad=mf.copy();bad.loc[0,'marker']=np.nan
    stops('MI imputiert Marker nie',lambda:k.impute(bad,mi_recipe,1),'MI_Marker_Winkel_Klasse')
    table('H1_MI','H1','W18',a.F,lambda:{'point':k.mi_pool(mf,scale,mi_recipe,1,guard=a.guard),
           'bootstrap':k.bootstrap(lambda w,r:k.mi_pool(mf,scale,mi_recipe,1,r+1,w,a.guard),np.arange(n),n,1,190,a.guard)})
    check('MI beobachtete Werte und Marker/Winkel bleiben identisch',lambda:k.require(
        all(np.array_equal(k.impute(mf,mi_recipe,1)[0].loc[mf[c].notna(),c].to_numpy(),mf.loc[mf[c].notna(),c].to_numpy()) for c in ['heart_rate','qtc_fridericia_ms','marker',a.A]),'MI_Originalwerte'))
    h4f=mf.copy();h4f['marker']=h4f[a.M];h4mi=p['endpoints']['H4']['mi']
    table('H4_missingness','H4','W18',a.M,lambda:{col:{str(cl):{'missing':int(h4f.loc[h4f.y.eq(cl),col].isna().sum()),
          'n':int(h4f.y.eq(cl).sum())} for cl in [0,1]} for col in ['heart_rate','qtc_fridericia_ms']})
    h4scale=p['endpoints']['H4']['markers'][a.M]['scale']
    table('H4_no_qtc','H4','W18',a.M,lambda:k.association(np.delete(z4,6,axis=1),y))
    def h4_indicator():
        mm=h4f.copy();extra=np.column_stack([mm[c].isna().to_numpy(float) for c in ['heart_rate','qtc_fridericia_ms']])
        for c in ['heart_rate','qtc_fridericia_ms']:mm[c]=mm[c].fillna(h4mi['start'][c])
        rr,_=raw_design(mm,'marker');zz=fixed_design(rr,h4scale)
        return k.association(np.column_stack([zz[:,:-1],extra,zz[:,-1]]),y)
    table('H4_missing_indicator','H4','W18',a.M,h4_indicator)
    table('H4_MI','H4','W18',a.M,lambda:{'point':k.mi_pool(h4f,h4scale,h4mi,4,guard=a.guard),
           'bootstrap':k.bootstrap(lambda w,r:k.mi_pool(h4f,h4scale,h4mi,4,r+1,w,a.guard),np.arange(n),n,1,190,a.guard)})
    # Fehlerpaket W16 nur H1: kein Transport auf inkompatiblen H4-Marker.
    error=p['error'];spline=p['endpoints']['H1']['markers'][a.F]['spline']
    for variant in p['simex']['variants']:
        code=p['simex']['parameter_codes'][variant]
        def corr(w=None,rplus=0,variant=variant,code=code):
            w=np.ones(n) if w is None else w
            return k.corrected(z,y,w,scale,error,variant,np.random.SeedSequence([1692026,16,1,code,rplus]),20 if rplus==0 else 10,spline)
        table('H1_error_'+variant,'H1','W16',a.F,lambda corr=corr:{'point':corr(),
              'bootstrap':k.bootstrap(lambda w,r:corr(w,r+1),np.arange(n),n,2,190,a.guard)})
    h4spline=p['endpoints']['H4']['markers'][a.M]['spline']
    table('H4_error_rc','H4','W16',a.M,lambda:k.corrected(z4,y,np.ones(n),h4scale,error,'rc',np.random.SeedSequence([1692026,16,4,5,0]),20,h4spline))
    for variant in ['joint_rc','joint','raw','clipped','logit','spline']:
        table('H4_error_'+variant,'H4','W16',a.M,lambda:k.require(False,'H4_passendes_Entwicklungsfehlerpaket_fehlt'))
    impossible=copy.deepcopy(error);impossible['angle_variance']=1e9
    stops('RC Reliabilitaet nichtpositiv',lambda:k.corrected(z,y,np.ones(n),scale,impossible,'rc',np.random.SeedSequence([1]),20,spline),'conditional_reliability')
    impossible['covariance']=[[1e9,0],[0,1e9]]
    stops('gemeinsame RC latente Kovarianz',lambda:k.corrected(z,y,np.ones(n),scale,impossible,'joint_rc',np.random.SeedSequence([1]),20,spline),'latente_Kovarianz')
    # Bootstrap truthfully respects thresholds, reproducible common union weights.
    boot=k.bootstrap(lambda w,r:{'mean':float(np.average(f.marker,weights=w))},np.arange(n),n,8,950)
    check('keine KI unter Mindestzahl',lambda:k.require(boot['valid']==8 and not boot['ci'],'KI_Gate'))
    r=4;rr=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,r])))
    mult=np.bincount(rr.integers(0,n,n),minlength=n); pos1=np.arange(300);pos2=np.arange(100,400)
    check('gepaarte Unionmultiplizitaeten',lambda:k.require(np.array_equal(mult[pos1[100:]],mult[pos2[:200]]),'Paarung'))
    optimistic=[]
    for r in range(8):
        rg=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1692026,20,r])))
        w=np.bincount(rg.integers(0,n,n),minlength=n)
        models={name:k.solver(z[:,:size],y,w) for name,size in [('B2',7),('B3',8),('B3_marker',9)]}
        optimistic.append({name:auc(y,z[:,:len(beta)]@beta,w)-auc(y,z[:,:len(beta)]@beta) for name,beta in models.items()})
    table('development_optimism','Entwicklung','W20',a.F,lambda:{'attempted':8,'minimum':190,'status':'KI_nicht_schaetzbar',
             'smoke_mean_optimism':{name:float(np.mean([r[name] for r in optimistic])) for name in fixed}})
    status=[{'table':name,'W':obj['W'],'endpoint':obj['endpoint'],'status':obj['status'],
             'reason':obj['result'].get('reason') if isinstance(obj['result'],dict) else None} for name,obj in tables.items()]
    pd.DataFrame(status).to_csv(out/'status_matrix.csv',index=False)
    files={str(q.relative_to(a.OUT)):a.sha(q) for q in out.iterdir() if q.suffix in ['.json','.csv','.parquet'] and q.name not in ['checks.json','manifest.json']}
    check('keine TESTZUGRIFFE-Aenderung',lambda:k.require(a.sha(a.ROOT/'koordination/TESTZUGRIFFE.md')==ledger_before,'Ledger'))
    a.save(out/'checks.json',{'checks':checks,'count':len(checks),'status':'bestanden','test_access':False,
               'integration_limits':['verkürzte Bootstrapläufe, keine vollen Produktions-KI',
                  'Produktiver physischer Leseadapter benötigt gesondert attestierte Eingabe-/Schreibziele; Schema-/Index-/Referenz- und Exactly-one-Lifecyclekerne synthetisch geprüft',
                  'H4 Fehlerkovarianz nicht verfügbar; W16 gemeinsame H4-Verfahren STOP',
                  'Parser nur synthetisch geprüft, kein reales Quellenattest']})
    a.save(out/'manifest.json',{'utc_begin':start,'utc_end':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'files':files,'parameters_sha256':a.sha(a.OUT/'parameters.json'),'mapping_sha256':a.sha(a.OUT/'mappings.json'),
             'specification_sha256':a.sha(a.SPEC),'environment':a.environment(),'memory':a.memory(),'peak':a.PEAK,
             'test_access':False,'bootstrap_smoke':8,'w16_bootstrap_smoke':2,'checks_sha256':a.sha(out/'checks.json')})
    print(f'SYNTHETIC_COMPLETE checks={len(checks)} tables={len(tables)}; Testzugriff=False')

def finalize_report(a):
    checks=a.load(a.OUT/'synthetic/checks.json');p=a.load(a.OUT/'parameters.json')
    synthetic_manifest=a.load(a.OUT/'synthetic/manifest.json')
    for relative,expected in synthetic_manifest['files'].items():
        k.require(a.sha(a.OUT/relative)==expected,'Synthetik_Ausgabehash')
    k.require(synthetic_manifest['parameters_sha256']==a.sha(a.OUT/'parameters.json'),'Synthetik_Parameterhash')
    k.require(synthetic_manifest['mapping_sha256']==a.sha(a.OUT/'mappings.json'),'Synthetik_Mappinghash')
    matrix=pd.read_csv(a.OUT/'synthetic/status_matrix.csv')
    inputs={'status':'kein_Lesevertrag_keine_Freigabe','run':'A155-2026-10-04-d6e70e',
            'development':a.load(a.OUT/'development_manifest.json')['inputs'],
            'prior_internal_manifest_reference':'<local path> (nur Codeverweis; NICHT geoeffnet/gehasht)',
            'required_projection_roles':[
                {'role':'candidate_pool','physical_path':None,'required':['subject_id','study_id','split','ecg_time','age','extraction_status','reference_available'],
                 'filter':'attestierte vorbelastete interne Partition; vor select_index; keine gesperrte Testmenge'},
                {'role':'reference_index','physical_path':None,'required':['label_state','eligibility_state','index_selection_strategy','time_precision','time_delta_hours',
                    'preferred_lvef','nearest_tte_delta_hours','nearest_tte_measurement_id','evidence_ids_json','lvef_conflict_gt_10'],
                 'filter':'H1 Echo +/-30 Tage; H4 Index-Encounter ohne neues Echoeinschlussgate'},
                {'role':'machine_report_flags','physical_path':None,'required':['machine_report','machine_metadata_usable_for_primary','machine_heart_rate_bpm','sinus_selector',
                    'current_report_af_state','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive'],'filter':'zeitgleiches Maschinen-EKG; nur Strata/gebundene Geraete-HF'},
                {'role':'encounter_icd','physical_path':None,'required':['subject_id','hadm_id','icd_code','icd_version'],'filter':'Index-Encounter; Codeliste/Erfassungsnenner attestieren'},
                {'role':'setting_anchor_measurement','physical_path':None,'required':['ecg_time','ed_intime','ed_outtime','admittime','dischtime','ed_match_ambiguous',
                    'admission_match_ambiguous','anchor_year_group','preferred_lvef_method'],'filter':'AW168-Zeitfenster/ambiguous; Messart preferred LVEF'},
                {'role':'medication_7d','physical_path':None,'required':['source_table','start_time','medication_raw','has_prescription_coverage'],
                 'filter':'prescriptions/pharmacy; -168h<=start_time-ecg_time<=0; Klassen unabhaengig; unknown separat'},
                {'role':'verification_train_first_technical','physical_path':None,'required':['split','subject_id','has_tte_30d','B4_Rohpraediktoren',a.F],
                 'filter':'nur train; V=0 UND V=1, kein Outcome-/Echo-/Markerfilter bei Indexwahl'},
                {'role':'W10_historical_unions_checkpoints','physical_path':None,'required':['ungürundete_Altpunktwerte','alle_gueltigen_Replikate','Fehlreplikatklassen',
                    'Hauptunion','IPW_train_Universum','IPW_technisches_Universum'],'filter':'nur separat erlaubte historische Rekonstruktion; unabhaengig von neuer R1-Union'}],
            'units':{'age':'Jahre','heart_rate':'bpm','qrs_duration_ms':'ms','qtc_fridericia_ms':'ms',a.A:'Grad','preferred_lvef':'Prozent'},
            'attestation':'K muss physische Pfade, Projektionserzeuger/Release, Hashes, Pflichtspalten und Grenzen vor Leseadapter binden; null ist STOP'}
    a.save(a.OUT/'input_contracts.json',inputs)
    files=list((a.ROOT/'werkzeuge/b_test').glob('aw171_*'))
    dependency=[a.ROOT/'werkzeuge/b_test/statistik.py',a.ROOT/'werkzeuge/b_test/aw157_models.py',
                a.ROOT/'werkzeuge/b_test/daten.py',a.ROOT/'werkzeuge/b_test/__init__.py',
                a.ROOT/'werkzeuge/nachrechnung/__init__.py',a.ROOT/'werkzeuge/nachrechnung/core.py',a.ROOT/'werkzeuge/nachrechnung/aw054.py',
                a.ROOT/'werkzeuge/nachrechnung/runner.py',a.ROOT/'werkzeuge/nachrechnung/b3.py',a.ROOT/'werkzeuge/nachrechnung/resources.py',
                a.ROOT/'werkzeuge/b_test/runner.py',a.ROOT/'werkzeuge/b_test/adapter.py',a.ROOT/'werkzeuge/b_test/aw156_run.py']
    hashes={str(f.relative_to(a.ROOT)):a.sha(f) for f in files+dependency if f.is_file()}
    codehash=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    env=a.environment();envhash=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.save(a.OUT/'code_manifest.json',{'files':hashes,'aggregate_sha256':codehash,'environment':env,'environment_sha256':envhash})
    text=a.SPEC.read_text(encoding='utf8')
    if '\n## 22. Bereit zum Freeze' in text:text=text.split('\n## 22. Bereit zum Freeze')[0]
    appendix='\n## 22. Bereit zum Freeze - Prüfkandidat, noch nicht freigabefähig\n\n'
    appendix+='**Status: nicht ausführungsfreigegeben.** K kann diesen dokumentierten Stand prüfen; vollständiger produktiver Freeze ist wegen der unten sichtbaren Vertrags-/Integrationslücken noch nicht möglich. Ein synthetischer Erfolg ersetzt keine Freigabe oder Quellenattestation.\n\n'
    for path,value in hashes.items():appendix+=f'- `{path}`: SHA-256 `{value}`.\n'
    for name in ['parameters.json','mappings.json','input_contracts.json','development_manifest.json','code_manifest.json','synthetic/checks.json','synthetic/manifest.json','synthetic/status_matrix.csv']:
        appendix+=f'- `{a.OUT.as_posix()}/{name}`: SHA-256 `{a.sha(a.OUT/name)}`.\n'
    appendix+=f'- Gesamtcodebindung (sortiertes kanonisches JSON der Dateihashes): `{codehash}`.\n'
    appendix+=f'- Umgebungshash (kanonisches JSON von Plattform, Interpreter, Bibliotheksversionen/Threads): `{envhash}`.\n'
    appendix+='- Die physische Eingabe-/Spalten-/Einheiten-/Filterliste steht in `input_contracts.json`. Entwicklungsquellen sind konkret gehasht; Auswertungspfade bleiben `null`/STOP, weil AW171 auch das Testmanifest nicht öffnen darf. K muss diese externen Bindungen attestieren.\n'
    appendix+='- Spezifikationshash: SHA-256 der **gesamten finalen UTF-8-Datei**, nach Schreiben extern in `<local path>` und Abgabe AW-171. Ein identischer Volltexthash kann nicht in seinem eigenen gehashten Volltext stehen; deshalb keine selbstreferenzielle Scheinbindung. Der Freezeauftrag muss exakt diesen externen Hash übernehmen.\n\n'
    appendix+=f'Synthetische Prüfung: {checks["count"]} explizite Prüfungen bestanden; vollständige Tabellen-/Hash-/Statusspur auf E:. Integrationseinschränkungen:\n\n'
    for item in checks['integration_limits']:appendix+='- '+item+'.\n'
    appendix+='\n**K muss vor einem ausführbaren Freeze entscheiden/binden:**\n\n'
    for item in p['missing_contracts']:appendix+='- '+item+'.\n'
    appendix+='- Den physischen Leseadapter mit attestiertem Vorindex-/Referenzuniversum in einem separaten Ausführungsauftrag binden; Schema-/Index-/Referenzkerne, MI-/W19-Bootstrap und exactly-one Beginn/Ende sind synthetisch geprüft. Keine Statuspromotion allein durch Hashregistrierung.\n'
    appendix+='- W16-H4-Umfang ausdrücklich auf verfügbare univariate Winkel-RC begrenzen oder ein passendes Entwicklungsfehlerpaket beauftragen; H1-Fehlerkovarianz nicht auf rohen H4-Marker übertragen.\n'
    appendix+='- Quellenwerte von Setting/Messart/Rhythmus bestätigen; Parser-Kandidat prüfen lassen und benötigte Zusatzcodes fachlich festlegen. Kein stilles ambulant-Mapping.\n'
    a.SPEC.write_text(text+appendix,encoding='utf8')
    finalhash=a.sha(a.SPEC)
    a.save(a.OUT/'freeze_candidate.json',{'specification_sha256':finalhash,'code_sha256':codehash,
              'parameter_sha256':a.sha(a.OUT/'parameters.json'),'mapping_sha256':a.sha(a.OUT/'mappings.json'),
              'status':'nicht_ausfuehrungsfreigegeben','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'test_specification_base_sha256':synthetic_manifest['specification_sha256'],
              'limitations':checks['integration_limits'],'test_access':False})
    report='# AW-171 - Abgabe MS/PR\n\nStand 13.10.2026. AGENTS.md vollständig gelesen; keine Recherche; ausschließlich AW168-Entwicklung und eigene Synthetik.\n\n## Ergebnis\n\n'
    report+='Spezifikation v1 als neue Datei mit numerischem AW168-Übernahmeblock, W7-Gate gemäß AE-181, W21-7-Tage-Rezept, W10-Zeilenliste, Mapping-/Parameter-/Codebindung erstellt. Entwurf und ursprüngliche Parameterdatei unverändert. Entwicklungsfixe B2/B3/B3+Marker-Fits, 90%-Kontrollschwellen, Quartile, MI-Startwerte und Splinebasis numerisch exportiert.\n\n'
    report+='Ein sicher gesperrter Wrapper und ausführbare Rechenkerne einschließlich MI-/W19-Bootstrap wurden erstellt und synthetisch geprüft. Der spätere Lese-/Ledgeradapter bleibt an den gesonderten Ausführungsauftrag gebunden. Der Freeze-Kandidat ist ausdrücklich **nicht ausführungsfreigegeben und wegen fehlender Originalunion-/Quellenverträge noch nicht freigabefähig**. Synthetische Tabellen und K-Hashregistrierung ersetzen diese Verträge nicht.\n\n'
    report+='## Neue Dateien\n\n- `nachrechnung/SPEZIFIKATION_B_TEST_EXPLORATIV_v1.md`.\n'
    for path in hashes:
        if Path(path).name.startswith('aw171_'):report+='- `'+path+'`.\n'
    report+='- `<local path>`: Parameter-, Mapping-, Entwicklungs-, Code- und Freeze-Manifeste; synthetische Eingabe, Ausgabetabellen, Prüfprotokoll und Statusmatrix.\n- `koordination/abgaben/AW-171.md` (diese Abgabe).\n\n## Belege und Reproduktionsbefehle\n\n'
    report+='PowerShell: `werkzeuge/b_test/aw171_start.ps1 prepare`, danach `synthetic`, danach `finalize`. Starter prüft vor jedem Python-Start freien RAM/Commit ≥1,5 GiB; eigene Auswertungsvenv, native Threads=1; Python-Ressourcengates vor Projektionen/Analysen und höchstens 25 Replikaten. Keine Installation/Löschung/Prozessbeendigung. Entwicklung: Footer vor Projektion, ausschließlich benötigte Spalten und train/validation-Filter. Kein Zugriff auf Auswertungspartition, keine Testfooter, Testmanifeste oder historischen Testcheckpointdaten; kein TESTZUGRIFFE-Eintrag.\n\n'
    report+=f'Prüfprotokoll `{a.OUT.as_posix()}/synthetic/checks.json`: {checks["count"]} Prüfungen bestanden, SHA-256 `{a.sha(a.OUT/"synthetic/checks.json")}`. W10-Abweichung um ein ULP, fehlender/falscher Originalunionhash, vollständige und Quasiseparation, kleine Strata, ESS, Rang, Quartil-Ties, Marker-MI-Verbot, Parserfälle, feste Skalen/Schwellen, H4 ohne Echogate, gepaarte Multiplizitäten und produktiver Start vor Öffnung geprüft. Synthetische Bootstrapzahlen sind verkürzt (8, W16 2), keine gültigen Produktions-KI behauptet. Ausgaben enthalten sichtbare STOPs und KI-Nichtschätzbarkeit.\n\n'
    report+=f'{len(matrix)} Tabellenzweige erzeugt, davon {int(matrix.status.eq("STOP").sum())} explizite STOP-Zweige. Die Tabellen mit Punktwerten können trotzdem Bootstrap-KI ohne ausreichende gültige Ziehungen enthalten. MI je Punkt und je geprüftem Bootstrap: 20 Imputationen × 20 Zyklen; MI-Bootstrap-Smoke je Endpunkt 1 Ziehung. Erfasster Peak RSS {synthetic_manifest["peak"]["rss"]/1024**2:.2f} MiB, private {synthetic_manifest["peak"]["private"]/1024**2:.2f} MiB; Herkunft `synthetic/manifest.json::peak`, Ressourcengates `aw171_run.py::guard`.\n\n'
    report+='Finalisierung prüft sämtliche synthetischen Ausgabedateihashes sowie Parameter-/Mappinggleichheit. Der synthetische Spezifikationshash bezeichnet den geprüften Rezepttext vor Anhängen des abschließenden Freeze-/Hashabschnitts; beide Hashes sind im Freeze-Kandidaten getrennt dokumentiert. Der abschließende Abschnitt ändert keine numerischen Rezepte. Entwurfs-SHA `60dd633e5ee3c4a1c1e35cd4bfd6a2be306c9fb9fd53850caa9c8ae5e37925d4` und ursprünglicher Parameter-SHA `a79d25e174008791f9533f2eeac890c8c1e04cbf956bb908d966f90017dd2b1b` wurden unverändert nachgeprüft.\n\n'
    for label,path in [('Spezifikation',a.SPEC),('Parameter',a.OUT/'parameters.json'),('Mappings',a.OUT/'mappings.json'),('Code-Manifest',a.OUT/'code_manifest.json'),('Freeze-Kandidat',a.OUT/'freeze_candidate.json')]:
        report+=f'- {label}: SHA-256 `{a.sha(path)}`.\n'
    report+=f'- Gesamtcodehash: `{codehash}`.\n\n'
    report+='Numerische Provenienz: `parameters.json::error` aus AW168 `reliability.json`/`simex.json`; W7 acht Zellen `models.json`, point=W7; Entwicklungsfit-/Quartil-/MI-Werte `endpoints/H1,H4`, aus `_private.parquet` mit `eligible_reference` (H1) bzw. Erwachsenen-/Index-/Eligible-/Known-Gates (H4), danach endpunktspezifischer Complete-case/known-y. Hash-/Spalten-/Filterspur `development_manifest.json`, Erzeuger `aw171_run.py::prepare`. Lauf A155-2026-10-04-d6e70e. W10 historische Punktfitzeilen aus AW142/AW156-Projektberichten, keine Testdatenreproduktion.\n\n## Offene Fragen und erforderliche Folgearbeiten\n\n'
    for item in checks['integration_limits']+p['missing_contracts']:report+='- '+item+'.\n'
    report+='\n## Nebenbefunde\n\nW15-Winkelvarianz und gemeinsame W16-Winkelvarianz stammen aus verschiedenen Paarvollständigkeitsnennern; getrennt gebunden. Das AW168-Fehlerpaket ist für normalisierten H1-Marker, kein passendes gemeinsames H4-Fehlermodell. Historische IPW-Replikate nutzen zwei getrennte Universen mit fortlaufendem RNG (train zuerst, technisches Auswertungsset danach), nicht die große H1/H4/RV-Union; §20 bindet diesen Unterschied ausdrücklich. `outside` beim Setting beweist ambulant nicht. Stundenattest-Ausnahme W21 ist gemäß AE-181 eng auf 7 Tage begrenzt.\n\n## F1-F8 / Fachstatus\n\n'
    report+='F1 eingeschränkt (geometrische Kopplung, Fehlerannahmen kein Mechanismusbeweis); F2 eingeschränkt (selektierte Krankenhaus-/Referenzpopulation, keine Populationsrisiken); F3 eingeschränkt (Schwellen entwicklungsfix, synthetisch geprüft); F4 eingeschränkt (post hoc, vorbelastete interne Partition, kein neues Erfolgsetikett); F5 offen (MAR-/Missingnessannahmen bleiben unbelegt, MI-Punkt-/Bootstrapkerne synthetisch geprüft); F6 eingeschränkt (B2/B3/Spline, keine globale Optimalität); F7 offen (Verifikationsbias/W19-Paket fehlt); F8 offen (Parser-/Flagfehlklassifikation und Quellenatteste). Rechen-/Dokumentenstand ist kein klinischer Befund und keine Publikationsfreigabe. Widerlegung/Grenze: abweichende Identitäts-/Referenzbindung, nichtschätzbare Modelle, fehlende Originalunion, inkompatible Fehlerskalen, schlechter gehaltene Leistung oder informative Missingness.\n\n## Quellenfehler-Kandidaten\n\nKeine Literaturrecherche; keine Literaturquellenfehler geprüft.\n\n## Selbsteinschätzung\n\nHohe Sicherheit der Zugriffsbegrenzung und der aus gelesenen Entwicklungsaggregaten gebundenen Parameter. Mittlere Sicherheit der synthetisch geprüften Rechenkerne; begrenzte Sicherheit für einen späteren kompletten Produktivlauf, weil die genannten Integrationen/Quellenatteste fehlen. Kein Claim „vollständig ausführungsbereit“.\n'
    (a.ROOT/'koordination/abgaben/AW-171.md').write_text(report,encoding='utf8')
    k.require(a.sha(a.DRAFT)=='60dd633e5ee3c4a1c1e35cd4bfd6a2be306c9fb9fd53850caa9c8ae5e37925d4','Entwurf_veraendert')
    k.require(a.sha(a.ROOT/'werkzeuge/b_test/parameter_entwicklung.json')=='a79d25e174008791f9533f2eeac890c8c1e04cbf956bb908d966f90017dd2b1b','Originalparameter_veraendert')
    a.save(a.OUT/'verification.json',{'status':'bestanden','checks':checks['count'],'table_branches':len(matrix),
         'all_synthetic_output_hashes_match':True,'original_draft_unchanged':True,'original_parameters_unchanged':True,
         'specification_sha256':a.sha(a.SPEC),'submission_sha256':a.sha(a.ROOT/'koordination/abgaben/AW-171.md'),
         'code_manifest_sha256':a.sha(a.OUT/'code_manifest.json'),'test_access':False})
    print('FINALIZE_COMPLETE spec_sha256='+finalhash+'; Status=nicht_ausfuehrungsfreigegeben')
