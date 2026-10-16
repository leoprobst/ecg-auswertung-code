"""AW171: Entwicklungsbindung und synthetischer Testschema-Lauf. Keine Testlesefunktion.

Der spätere Zugriff erfordert einen separaten Auftrag/Adapter. 'partition' stoppt
vor jeglicher Eingabeöffnung; weder Splitname noch selbst erstelltes Manifest
können die gesperrte Testmenge entsperren.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
import sys, json, hashlib, re, platform, ctypes, time, ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import psutil
from scipy.special import expit, logit
from werkzeuge.b_test import aw171_kernels as k
from werkzeuge.b_test.statistik import raw_design, fixed_design, scaling

OUT=Path(str(data_path("b.aw171")))
DEV=Path(str(data_path("b.aw168")))
SPEC=ROOT/'nachrechnung/SPEZIFIKATION_B_TEST_EXPLORATIV_v1.md'
DRAFT=SPEC.with_name('SPEZIFIKATION_B_TEST_EXPLORATIV_v1_ENTWURF.md')
F='qrs_t_normalized_discrete_frechet'; M='qrs_t_frechet_beat_median'; A='spatial_qrs_t_angle_deg'
INPUTS={}; PEAK={'rss':0,'private':0}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict): return {str(a):clean(b) for a,b in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(a) for a in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def save(path,x):
    path=Path(path)
    k.require(path.resolve().is_relative_to(OUT.resolve()),'Schreibziel')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def memory():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ['total','free','page','commit','vtotal','vfree','extended']]
    s=Status(); s.length=ctypes.sizeof(s)
    k.require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))),'Ressource_Messung')
    return {'ram_free':s.free,'commit_free':s.commit}

def guard():
    info=psutil.Process().memory_info()
    PEAK['rss']=max(PEAK['rss'],info.rss); PEAK['private']=max(PEAK['private'],info.private)
    k.require(max(PEAK.values())<=1.5*1024**3,'Ressource_Prozess')
    k.require(min(memory().values())>=1.5*1024**3,'Ressource_Warten_Fortsetzen')

def bind(path):
    path=Path(path).resolve()
    # Whitelist VOR stat/hash/footer, keine Auswertungspartitionsquellen.
    safe=path.is_relative_to(DEV.resolve()) or path.is_relative_to(OUT.resolve()) or path.is_relative_to(ROOT.resolve())
    k.require(safe,'Sperrbarriere_Pfad')
    if path.is_relative_to(ROOT.resolve()):
        k.require(path.suffix in {'.py','.ps1','.md','.json','.csv'},'Sperrbarriere_Projekt')
    k.require(path.stat().st_size<20*1024**2 or path.suffix=='.parquet','JSON_20MB')
    if path.suffix=='.parquet':
        # Schema/Footer VOR dem vollständigen Inhalts-Hashstream.
        pf=pq.ParquetFile(path)
        footer={'rows':pf.metadata.num_rows,'columns':pf.schema_arrow.names}
    else:footer=None
    INPUTS[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
    if footer is not None:INPUTS[str(path)]['footer']=footer
    return path

def load(path): return json.loads(bind(path).read_text(encoding='utf8'))

def development(path,columns):
    guard(); path=bind(path)
    k.require(path.parent==DEV.resolve(),'Sperrbarriere_Entwicklung')
    pf=pq.ParquetFile(path); names=pf.schema_arrow.names
    k.require(set(columns)<=set(names) and 'split' in names,'Schema')
    table=ds.dataset(str(path),format='parquet').to_table(columns=columns,filter=ds.field('split').isin(['train','validation']))
    f=table.to_pandas()
    k.require(set(f.split)<= {'train','validation'},'Sperrbarriere_Split')
    INPUTS[str(path)]['columns']=columns; INPUTS[str(path)]['filter']='split IN (train,validation)'
    INPUTS[str(path)]['footer_rows']=pf.metadata.num_rows
    return f

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    reliability=load(DEV/'reliability.json'); sx=load(DEV/'simex.json'); models=load(DEV/'models.json')
    w7=[{key:r[key] for key in ['key','endpoint','marker','fold','n','cases','beta','or','delta_auc','sd']} for r in models if r.get('point')=='W7']
    k.require(len(w7)==8 and all(r['beta']>0 and r['delta_auc']>=0 for r in w7),'W7_Gate')
    patterns=load(DEV/'medication_inventory.json')['patterns']
    knots=load(DEV/'comparator_candidates.json')['spline_knots']
    params={'status':'nicht_ausfuehrungsfreigegeben','run':'A155-2026-10-04-d6e70e','role':'entwicklung',
            'w7':w7,'medication_patterns':patterns,'endpoints':{},
            'error':{'angle_variance':reliability['angle_error'],'angle_pairs':reliability['angle_error_pairs'],
                     'angle_persons':reliability['angle_error_subjects'],'covariance':reliability['error_covariance'],
                     'joint_pairs':reliability['error_pairs'],'joint_persons':reliability['error_subjects'],
                     'order':[A,F], 'units':['Grad','normalisierte Frechet-Einheit'],
                     'logit_variance':next(r['error'][0][0] for r in sx if r['mode']=='logit'),
                     'pool':'train+validation; Differenzkovarianz ddof=1 / 2; keine Mittelung von ICCs'},
            'simex':{'lambda':[0,.5,1,1.5,2],'point_draws':20,'bootstrap_draws':10,'orders':[1,2],
                     'logit_clip':[.5,179.5],'raw_clip':None,'clipped_bounds':[0,180],
                     'variants':['clipped','joint','joint_rc','logit','raw','rc','spline'],
                     'parameter_codes':{v:i for i,v in enumerate(['clipped','joint','joint_rc','logit','raw','rc','spline'])},
                     'seed':[1692026,16,'endpoint_code','parameter_code','r_plus_1']},
            'recipes':{'R1':{'planned':1000,'minimum':950,'seed':[1282026,'r'],'rng':'PCG64'},
                       'R16_R18':{'planned':200,'minimum':190},'optimism':{'planned':200,'minimum':190,'seed':[1692026,20,'r']},
                       'MI':{'m':20,'cycles':20,'pmm_donors':5,'binary':'logistic posterior draw',
                             'seed':[1692026,18,'endpoint','r_plus_1','m'],'no_imputation':['marker',A,'y'],
                             'pool':'Rubin beta + CR1; Student-t df; delta_auc MI-Mittel'},
                       'quartiles':'linear; <=Q25, >Q25..Q50, >Q50..Q75, >Q75; feste Entwicklungsmediane',
                       'threshold':'Score > kleinste beobachtete Kontrollschwelle mit Spezifitaet >=.90'},
            'w10':{'original_union_hash':None,'status':'STOP_bis_Originalunion_Attest',
                   'dtype':'historischer Identitaetstyp durch spaeteren Auftrag zu attestieren; np.unique sortiert',
                   'AW142_recipe':'np.unique(concatenate(Indexpersonen aller geladenen HYPOTHESES + technische Indexpersonen falls verfuegbar))',
                   'AW156_recipe':'np.unique(concatenate(H1,H4,RV first_verified Indexpersonen, first_technical H1-Personen))',
                   'seed':[1282026,'r'],'n':1000,'ridge':1e-6,'old_steps':60,'new_steps':200,
                   'train_sampling':'Original-IPW adapter.py::ipw_replicate: ein PCG64 SeedSequence([1282026,r]); zuerst sortierte vollstaendige train-Verifikationspersonen, danach sortierte B4-vollstaendige technische Auswertungspersonen mit demselben RNG; P99 neu aus verifizierten train-Multiplizitaeten. Diese beiden Universen separat hashen; die uebergeordnete H1/H4/RV-Union wird fuer IPW NICHT als Ersatz verwendet.'},
            'missing_contracts':['physische attestierte Auswertungseingaben','Originalunionhash/Identitaetstyp W10',
                                 'W10 ungerundete Altpunkte und Fehlreplikate','ICD-Zusatzliste kardiomyopathienah von K',
                                 'Quellenattest W24 Flags/Parser und Setting/Messart','train Verifikationsuniversum mit H1-Marker W19',
                                 'H4 Wiederholungsfehlerpaket fuer rohen Marker W16','spaeterer Lese-/Schreibauftrag samt Ledgerziel']}
    for bef,ep,marker in [('BEF-0063','H1',F),('BEF-0049','H4',M)]:
        entry=load(DEV/(bef+'_entry.json'))
        cols=['split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F,M,'label_state']
        if ep=='H1': cols+=['preferred_lvef','eligible_reference']
        else: cols+=['eligibility_state','index_selection_strategy']
        f=development(DEV/(bef+'_private.parquet'),cols)
        source_n=len(f)
        ref=f.eligible_reference.eq(True) if ep=='H1' else (f.age.ge(18)&f.label_state.isin(['positive','negative'])&f.eligibility_state.eq('eligible')&f.index_selection_strategy.eq('first_verified_ecg_reference_result_blind'))
        f=f.loc[ref].copy()
        f['y']=(f.preferred_lvef<40).astype(float) if ep=='H1' else f.label_state.map({'positive':1.,'negative':0.})
        endpoint={'original_scale':entry['scaling_B3'],'source_rows':source_n,'markers':{}}
        mi_start={col:float(f.loc[f.split.eq('train'),col].dropna().median()) for col in ['age','heart_rate','qrs_duration_ms','qtc_fridericia_ms']}
        mi_start['sex_numeric']=float(f.loc[f.split.eq('train'),'sex_numeric'].dropna().mode().iloc[0])
        endpoint['mi']={'start':mi_start,'source':'train Referenzindex, vor B3-Vollstaendigkeit'}
        for mark in [F,M]:
            raw,names=raw_design(f,mark); good=np.isfinite(raw).all(1)&np.isfinite(f.y)
            complete=f.loc[good]; raw=raw[good]
            scale=entry['scaling_B3'] if mark==marker else scaling(raw,names)
            z=fixed_design(raw,scale); y=complete.y.to_numpy(float)
            observed=next(r for r in w7 if r['endpoint']==entry['endpoint'] and r['marker']==mark and r['fold']=='train')
            k.require(abs(scale['sd'][-1]-observed['sd'])<1e-12,'AW168_Kreuzskala')
            fixed={}
            for model,size in [('B2',7),('B3',8),('B3_marker',9)]:
                b=k.solver(z[:,:size],y); lp=z[:,:size]@b
                fixed[model]={'beta':b.tolist(),'threshold':k.threshold90(lp,y),'n':len(y),'cases':int(y.sum()),'fit_population':'train+validation C_e; feste Original-/AW168-Skala'}
            basis=k.rcs(complete[A].to_numpy(),knots); train=complete.split.eq('train').to_numpy()
            spline={'knots':knots,'mean':basis[train].mean(0).tolist(),'sd':basis[train].std(0).tolist(),'population':'train C_e; ddof=0'}
            endpoint['markers'][mark]={'scale':scale,'models':fixed,'quartiles':k.quartiles(complete[mark]),'spline':spline,
                                     'n':len(complete),'filter':'eligible_reference and finite raw_design and known y; train+validation'}
        params['endpoints'][ep]=endpoint
        del f
    # Historische Berichtsschluessel lesen; keine historischen Testartefakte oeffnen.
    report=bind(ROOT/'nachrechnung/B_TESTERGEBNIS_AW142.md').read_text(encoding='utf8')
    rows=[]
    for line in report.splitlines():
        parts=[x.strip() for x in line.split('|')]
        if len(parts)>=10 and parts[6].isdigit() and int(parts[6])<950:
            key=re.search(r'`([^`]+)`',parts[-2])
            if key and key.group(1).startswith('ergebnis.json/'):
                rows.append({'run':'AW142','key':key.group(1),'name':parts[1],'valid':int(parts[6]),'report_sha256':sha(ROOT/'nachrechnung/B_TESTERGEBNIS_AW142.md')})
    for key in ['rank_3.json/unweighted_same_first_technical','rank_3.json/IPW']:
        rows.append({'run':'AW156','key':key,'name':'urspruengliches IPW-Paar','valid':777,'report_sha256':sha(ROOT/'nachrechnung/B_FORTSETZUNG_AW156.md')})
    params['w10']['rows']=rows
    release=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker_mimic_mappings_py")))
    # Nur gebundener Quellcode, keine Imports mit Nebenwirkungen.
    tree=ast.parse(release.read_text(encoding='utf8')); groups=[]
    for node in tree.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='ICD_GROUPS':
            for call in node.value.elts:
                groups.append({'group':ast.literal_eval(call.args[0]),'icd9':list(ast.literal_eval(call.args[1])) if len(call.args)>1 else [],
                               'icd10':list(ast.literal_eval(call.args[2])) if len(call.args)>2 else []})
    mapping={'release_path':str(release),'release_sha256':sha(release),'icd_groups':groups,
             'normalization':'strip.upper.replace(dot,empty); version 9/10 only; retain leading characters',
             'w13_additional_exclusion':None,'heart_failure':{'9':['428'],'10':['I50']},
             'rhythm':{'sinus_flag':'sinus_selector nach AW054; kein neuer Sinusparser',
                       'af_flag':'rhythm_context: explizites AF/atrial_fibrillation; Quellenwerte vor Zugriff attestieren',
                       'rbbb_flag':'classical_rbbb_pattern_positive','lbbb_flag':'classical_lbbb_pattern_positive'},
             'setting':{'input':'relative Zeitfenster/ambiguity wie AW168','both':'ED_und_stationaer','ED':'ED','inpatient':'stationaer',
                        'outside':'ausserhalb_beobachteter_Fenster; NICHT automatisch ambulant','unknown':'unbekannt'},
             'measurement':{'3d':'3D','biplane':'Biplan','generic':'generisch','other':'sonstige',
                            'unknown':'unbekannt','unmapped':'STOP_neuer_Quellwert; kein Zahlenmusterparser'},
             'pacing_parser':'aw171_mapping.py; Kandidat ohne reale Befundtextvalidierung',
             'medication':{'patterns':patterns,'window_hours':[-168,0],'sources':['prescriptions','pharmacy'],
                           'time_field':'start_time','coverage':'has_prescription_coverage',
                           'categories':['exposed','observed_no_record','unknown'],'unknown_dummy':True,
                           'around24':'nur_deskriptiv; nicht im Sensitivitaetsfit'}}
    save(OUT/'mappings.json',mapping); save(OUT/'parameters.json',params)
    save(OUT/'development_manifest.json',{'inputs':INPUTS,'memory':memory(),'peak':PEAK,'test_access':False,
        'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'environment':environment()})
    build_spec(params,mapping)
    print('PREPARE_COMPLETE; Entwicklungsparameter, Mappings und Spezifikation geschrieben; Testzugriff=False')

def environment():
    import scipy, pyarrow
    return {'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
            'pyarrow':pyarrow.__version__,'psutil':psutil.__version__,'threads':1,'executable':sys.executable}

def build_spec(p,mapping):
    text=DRAFT.read_text(encoding='utf8')
    text=text.replace('# Paper B - gebündelte explorative Auswertung, Spezifikationsentwurf v1','# Paper B - gebündelte explorative Auswertung, Spezifikation v1 (Freeze-Kandidat)')
    text=text.replace('AW-169 · 13.10.2026 · MS/PR · Dokumentenspur ohne Datenzugriff.','AW-171 · 13.10.2026 · MS/PR · Entwicklungsbindung und synthetische Tests; kein Auswertungspartitionszugriff.\n\n**Status: nicht ausführungsfreigegeben.** §§18-22 ergänzen die übernommenen §§1-17; bei historischen Formulierungen über noch fehlendes AW-168 gilt §18. Offene Verträge werden nicht als erledigt behandelt.')
    text=text.replace('Dieser Entwurf ändert','Diese Spezifikation ändert')
    text=text.replace('AW-168-Bericht/Abgabe waren zum Entwurfszeitpunkt noch nicht vorhanden; keine Parameter daraus erfunden.','AW-168-Bericht und Parameter sind jetzt in §18 numerisch gebunden.')
    text=text.replace('AW-168-Werkzeuge und ihr Parameterpaket werden mit konkreten Pfaden/Hashes ergänzt, sobald vorhanden.','AW-168-Werkzeuge und Parameterpaket sind in §§18/20 konkret gebunden.')
    text=text.replace('Dieser Auftrag erstellt oder testet keinen Runner.','AW-171 erstellt und prüft den in §21 beschriebenen gesperrten Wrapper.')
    text+='\n\n## 18. Numerischer AW-168-Übernahmeblock und W7\n\n'
    text+='Quelle: `<local path>`; Ergebnisbericht `nachrechnung/B_ENTWICKLUNG_W_AW168.md`; Lauf A155-2026-10-04-d6e70e; Erzeuger `aw168_analysen.py`. Export durch `aw171_run.py prepare`, Splitfilter ausschließlich train/validation. Alle Zahlen unten sind aus benannten JSON-Schlüsseln oder festen Rezepten exportiert.\n\n'
    for name in ['reliability.json','simex.json','models.json','comparator_candidates.json','medication_inventory.json','simulation.json','prepare_manifest.json']:
        text+=f'- `{DEV.as_posix()}/{name}`: SHA-256 `{sha(DEV/name)}`.\n'
    for name in ['aw168_protokoll.md','aw168_nachtrag1.md','aw168_analysen.py','aw168_run.py','aw168_verify.py']:
        text+=f'- `werkzeuge/b_test/{name}`: SHA-256 `{sha(ROOT/"werkzeuge/b_test"/name)}`.\n'
    for name in ['B_ENTWICKLUNG_W_AW168.md','b_entwicklung_w.csv']:
        path=ROOT/'nachrechnung'/name
        if path.exists(): text+=f'- `nachrechnung/{name}`: SHA-256 `{sha(path)}`.\n'
    e=p['error']
    text+=f'\n**W15/univariate RC:** Winkelvarianz {e["angle_variance"]!r} Grad² aus {e["angle_pairs"]} Paaren/{e["angle_persons"]} Personen (`reliability.json::angle_error*`, gepoolte Entwicklung; Quelle AW155-Paare).\n\n'
    text+=f'**Gemeinsames W16:** Fehlerkovarianz {e["covariance"]!r}; Reihenfolge Winkel in Grad, normalisierter Fréchet in Originaleinheit; {e["joint_pairs"]} Paare/{e["joint_persons"]} Personen (`reliability.json::error_covariance/error_pairs/error_subjects`, AW168-Paare train+validation). Transformierte Winkelvarianz {e["logit_variance"]!r} in logit(θ/180)² (`simex.json`, mode=logit, error[0][0]); θ wird vor Transformation auf [0,5;179,5] begrenzt, auch 0/180. Differenzvarianz/-kovarianz ddof=1 geteilt durch 2; kein ICC-Mittel.\n\n'
    text+='AW168 transportiert in beide Folds dasselbe gepoolte Fehlerpaket. Es gibt keine separat geschätzten train-/validation-Fehlerszenarien, die erfunden werden dürften. **Dieses gemeinsame Paket betrifft H1 und den normalisierten Marker; es ist kein Fehlermodell für den rohen H4-Marker.** H4 univariate Winkel-RC ist definiert; gemeinsame RC/SIMEX mit rohem H4-Marker STOP bis eigenes Entwicklungsfehlerpaket vorliegt. §6 gilt insoweit eingeschränkt.\n\n'
    text+=f'SIMEX: λ={p["simex"]["lambda"]}; Punkt 20, Bootstrap 10 antithetische Ziehungen pro positivem λ; lineare und quadratische Extrapolation nach −1. Rohwinkel ohne Clipping, clipped auf [0;180], logit wie oben. Spline: Winkelstörung **vor** Basisbildung. Feste Knoten {p["endpoints"]["H1"]["markers"][F]["spline"]["knots"]}. Basis-Mittel/SD je gebundenem Endpunkt/Marker explizit in `parameters.json::endpoints/.../spline`, train, ddof=0. Keine Ergebnisvariantenwahl.\n\n'
    text+='Historische AW168-Seeds: 1682026; W16 Punkt SeedSequence([1682026,16,fold_code]), Bootstrap-Union [1682026,161,fold_code,r], MC [1682026,162,fold_code,r], fold_code train=0/validation=1. Simulation [1682026,scenario_code,r], MC [1682026,163,scenario_code,r], scenario null=0/true=1; 200 Läufe à 4.000 synthetische Personen. W7 SeedSequence([1682026,int(SHA256(key)[:8],16),r]), 200 Ziehungen. Diese historischen Seeds ersetzen nicht die neuen §2/§6-Seeds. Neue W16-Parametercodes lexikografisch: '+json.dumps(p['simex']['parameter_codes'])+'.\n\n'
    simulation=load(DEV/'simulation.json')
    text+='Null-/Zusatzeffektsimulation: '+', '.join(f'{r["scenario"]}: eigener Effekt={r["effect"]}, next={r["next"]}, n_per_run={r["n_per_run"]}' for r in simulation)+'. Deutung gemäß AW168-Bericht/AE-181: diese klassische Nullkonstellation erklärt die beobachtete Entwicklungsgröße nicht; persistente/differenzielle Fehler und biologische Eigenständigkeit sind dadurch nicht geklärt. Keine allgemeine Validierung durch SIMEX.\n\n'
    text+='**W7-Gate offen**, Quelle `koordination/ENTSCHEIDUNGEN.md::AE-181`, 13.10.2026. Acht Zellen aus `models.json`, point=W7:\n\n| Schlüssel | N | Fälle | β | OR | apparente ΔAUC |\n|---|---:|---:|---:|---:|---:|\n'
    for r in p['w7']:text+=f'| {r["key"]} | {r["n"]} | {r["cases"]} | {r["beta"]:.12g} | {r["or"]:.12g} | {r["delta_auc"]:.12g} |\n'
    text+='\nAlle acht β>0 und ΔAUC≥0; keine Signifikanzregel. Entwicklungs-/Bestätigungsfold bleiben getrennt, keine unabhängige Validierung. Kreuzskalen numerisch und mit Population in `parameters.json`.\n\n## 19. W21 - ergänzte Medikationssensitivität\n\n'
    text+='Gemäß AE-181 ausschließlich **7 Tage vor dem EKG**, einschließlich Grenzen −168 h und 0 h, Verordnungsbeginn `start_time` in `prescriptions`/`pharmacy`; identisches AW168-Rezept, keine nachträgliche Interpretation als lückenlose Einnahme oder aktive Exposition. Klassen unabhängig aus Rohbezeichnungen, Überlappung zulässig: '+', '.join(p['medication_patterns'])+'. Vollständige Namensmuster in `mappings.json::medication/patterns` mit AW168-Herkunft.\n\n'
    text+='Je Klasse Kategorien `exposed`, `observed_no_record`, `unknown`: positiver zeitgeeigneter Eintrag hat Vorrang; sonst belegte `has_prescription_coverage` → beobachtet ohne Eintrag; sonst unbekannt. Eigene Dummies exponiert/unbekannt, Referenz beobachtet ohne Eintrag. B3+Marker plus alle Klassenindikatoren; H1/H4 C_e, gleiche Personen für Vergleich mit unadjustiertem B3+Marker; R1, feste Entwicklungsskalierung. Konstante/neue Kategorien werden nicht nach Ergebnis entfernt: Rang-STOP. Nenner/Kategorien einschließlich unknown, OR, apparente ΔAUC, Wald-/Bootstrap-KI und Ausfälle ausgeben. ±24 h ausschließlich R0-deskriptiv, niemals Regressionssensitivität. Datensatzzeitannahme/Erfassungsrisiko benennen; fehlendes Stundenattest bedeutet nach AE-181 keinen 7-Tage-STOP.\n\n'
    text+='## 20. Gebundene Mappings, W10 und neue Rezepte\n\n'
    text+=f'Parameterpaket `{OUT.as_posix()}/parameters.json`: SHA-256 `{sha(OUT/"parameters.json")}`. Mappingpaket `{OUT.as_posix()}/mappings.json`: SHA-256 `{sha(OUT/"mappings.json")}`. Release-ICD-Code `{mapping["release_path"]}`, SHA-256 `{mapping["release_sha256"]}`. Mappings sind operative Rezepte, keine klinische Quellenvalidierung.\n\n'
    text+='ICD: explizite versionsgetrennte Präfixe sämtlicher Release-Gruppen; Großschreibung/Punkte entfernen, führende Stellen behalten. W13-Herzinsuffizienz 428*/I50*. Zusatzliste kardiomyopathienah bleibt **K-Entscheidung**, bis dahin STOP S1/S2, keine improvisierte Liste. Parser: `aw171_mapping.py` behandelt Negation, Historie, Vermutung, fehlend, positiven zeitgleichen Hinweis, gemischte Klauseln/Mehrdeutigkeit getrennt; textliche Abwesenheit ist kein Pacingausschluss. Sinusflag exakt AW054, AF/Blockflags nur aus attestiertem vorhandenen Mapping, nicht aus ICD als Rhythmusersatz. Neue Quellwerte stoppen betroffene Strata.\n\n'
    text+='Setting wie AW168: eindeutige inklusive ED-/Aufenthaltsfenster, beide Fenster gesondert, ambiguous/unvollständig unbekannt. `outside` bleibt außerhalb beobachteter Fenster, **keine automatische ambulante Einstufung**; ambulant nur mit eigenem Attest. Messart nur exakte vorhandene Kategorien 3d/biplane/generic/other/unknown (Quellwerte erst attestieren); sonst STOP unbekannter Quellwert, keine Ableitung aus LVEF.\n\n'
    text+='Neue Entwicklungsfits B2/B3/B3+Marker werden von `prepare` auf gepooltem C_e mit unveränderter Original-Skalierung gerechnet; feste Koeffizienten/90%-Kontrollschwellen, Quartilgrenzen/-mediane, train-Startwerte MI und Splinebasisstandardisierung stehen numerisch im Parameterpaket. W19 bleibt STOP: AW168 exportierte keinen markerhaltigen ersten-technischen train-Verifikationsfit; C_e ist dafür kein Ersatzuniversum. MI ausführbar in `aw171_kernels.py::impute/mi_pool` (PMM-5, Posteriorziehungen, 20×20, Rohalterbasis erst danach, Rubin-Student-t; keine Outcome-/Marker-/Winkelimputation). W20 Entwicklungsoptimismus erfordert 200 eigene Entwicklungsbootstrap-Neufits und wird gesondert gerechnet, nie aus Auswertungsbootstrap abgeleitet.\n\n'
    text+='**W10 vollständige Liste von tatsächlich punktgeschätzten Zeilen mit unzureichendem Bootstrap**, Quelle AW142/AW156-Berichte; zeilenweise alle Größen des Fits (β/OR/AUC/ΔAUC/Marker-AUC), nicht bloß ΔAUC:\n\n| Lauf | Berichtsschlüssel | gültig/1000 |\n|---|---|---:|\n'
    for r in p['w10']['rows']:text+=f'| {r["run"]} | `{r["key"]}` | {r["valid"]} |\n'
    text+='\nNicht punktgeschätzte Vertrags-/Zeitgate-STOPs sind keine Solverkorrekturzeilen. AW142-Union: `np.unique(concatenate(Indexpersonen aller geladenen HYPOTHESES, technische Indexpersonen falls vorhanden))`; AW156-Union für allgemeine Fits/3b: H1/H4/RV first_verified plus H1 first_technical, `np.unique`. Codequellen `adapter.py::run_frames` (Zeile 266), `aw156_run.py:157`, in Codehashmanifest gebunden. **Original-IPW weicht davon ab:** `adapter.py::ipw_replicate` zieht mit einem PCG64 SeedSequence([1282026,r]) zuerst auf den sortierten train-Verifikationspersonen, dann mit demselben RNG auf den sortierten B4-vollständigen technischen Auswertungspersonen. P99 wird aus den verifizierten train-Personen mit Frequenzgewichten neu geschätzt. Beide Personenuniversen brauchen eigene Inhalts-/Typ-/Sortierbindung; die große H1/H4/RV-Union darf diese IPW-Universen nicht ersetzen. Ungewichtet/gewichtet verwenden dieselben resultierenden Outcomemultiplizitäten und gemeinsame gültige Replikate. Originalunion-**Inhalts**hash, ursprünglicher Identitätstyp, ungerundete Altpunktwerte und kleinstes konvergenzfehlgeschlagenes IPW-Replikat sind unter dem AW171-Leseverbot nicht überprüfbar. Hier ausdrücklich `null`/STOP statt erfundener Hash. K muss den Rekonstruktionszugriff einschließlich RV-Indexuniversum separat erlauben. Es wurden weder alte Testcheckpointdateien noch Testmanifeste geöffnet/gehasht.\n\n'
    text+='## 21. Wrapper, STOP-Matrix und synthetische Prüfung\n\n'
    text+='Wrapper `aw171_run.py`, reine Rechenkerne `aw171_kernels.py`, Parser `aw171_mapping.py`, Starter `aw171_start.ps1`; keine alten Runnerimporte/Top-Level-Lesefunktionen. Physische AW171-Eingänge strikt AW168-Entwicklung oder eigene Synthetik; Whitelist vor Hash/Footer. Jeder Produktivstart `partition` stoppt **vor Öffnung**, bis separater Auftrag einen attestierten Leseadapter liefert. Ein K-Hash allein erlaubt nie die gesperrte Testmenge. Dieser sichere Zustand ist absichtlich nicht ausführungsfreigegeben.\n\n'
    text+='Ausgabetabellen und Protokoll ausschließlich <local path>. Jede Tabelle führt W-Punkt/Endpunkt/Marker/Fallset/Skalenbindung/Seed-/Replikatregime/Status; Manifest Quelle/Filter/Hashes/Umgebung/Ressourcen. Synthetischer Integrationstest verwendet verkürzte Bootstrapzahlen zur Funktionsprüfung; KI bleiben bei <950/<190 gültigen ausdrücklich nicht schätzbar. Er ist kein Produktions-/Ressourcenleistungsnachweis für 1000 Replikate und keine biologische Analyse. Das Prüfprotokoll listet bestandene STOP-Fälle und die verbleibenden Integrationslücken.\n\n'
    text+='STOP-Matrix: Gesamt vor Öffnung bei Split-/Partitions-/Hash-/Freigabevertragsfehler; betroffener Zweig bei Index-/Referenzabgleich, fehlenden Quellen/Spalten, unbekannten Mappings, nichtpositiven/nonfiniten Gewichten, Klasse/ESS<100, Rang, vollständiger/Quasiseparation, Score-/Konvergenz-/CR1-Fehler, Quartil-Ties/Leerquartilen, MI-Donor-/Modell-/Marker-/Winkelfehler, nichtpositiver RC-Reliabilität/latenter Kovarianz, Fehlerkovarianz/Extrapolation, fehlenden train-Verifikationsparametern. W10 Altpunkt-/Unionabweichung stoppt gesamten W10-Zweig, kein punktseitiger neuer Solver. Ausfallursachen zählen, kein Firth/Ridge-/PSD-/Neuskalierungsretten. W7 nur alle acht Entwicklungszellen; Ressource checkpointfähig unter E: und STOP/Warten ohne neue Datenöffnung.\n'
    SPEC.write_text(text,encoding='utf8')

def synthetic():
    from werkzeuge.b_test.aw171_checks import run
    run(sys.modules[__name__])

def finalize():
    from werkzeuge.b_test.aw171_checks import finalize_report
    finalize_report(sys.modules[__name__])

def main():
    phase=sys.argv[1] if len(sys.argv)>1 else 'synthetic'
    # Dieser Zweig liegt absichtlich VOR Load/Hash/Footer und erzeugt keinen Ledger.
    if phase=='partition':
        print('STOP: kein spaeterer attestierter Ausfuehrungsauftrag/Leseadapter; keine Eingabe geoeffnet')
        return 77
    guard()
    OUT.mkdir(parents=True,exist_ok=True)
    if phase not in ['prepare','synthetic','finalize']: raise k.Stop('Phase')
    save(OUT/'status.json',{'phase':phase,'state':'running','pid':os.getpid(),'test_access':False})
    globals()[phase]()
    save(OUT/'status.json',{'phase':phase,'state':'complete','pid':os.getpid(),'peak':PEAK,'test_access':False})
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except (k.Stop,ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
        # Nur kontrollierte technische Fehler, keine Patienteneinzelwerte ausgeben.
        print('AW171 STOP:',type(ex).__name__)
        if OUT.exists(): save(OUT/'failure.json',{'type':type(ex).__name__,'detail':str(ex),'phase':sys.argv[1:]})
        sys.exit(1)
