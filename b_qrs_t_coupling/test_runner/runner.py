"""AW136 v4-Einstieg. K-/Hash-/Quellen-/Messgates vor jedem Testzugriff.

Entwicklung ist ausdrücklich keine unabhängige Bestätigung. Keine Testrechnung
im Auftrag AW136; künftige Testeingänge benötigen eine separate K-Freigabe.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
# python -m executes this file as __main__. Adapter imports must reference the
# same module, otherwise they lose the active checkpoint directory.
if __name__=='__main__':sys.modules['werkzeuge.b_test.runner']=sys.modules[__name__]

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(str(data_path("b.aw131")))
OUTPUT_ROOT=Path(str(data_path("b.aw136")))
SPEC_HASH='f3165417cabac8479f63b2d1d14445ad5b43ec7ca21582ecf142fbc6eb4156a4'
ECHONEXT_EXCLUDED={'status':'ausgeschlossen','reason':'§0 Nachtrag 2(c): EchoNext nicht Bestandteil dieses Testlaufs; AW123-Vorstufe gesperrt'}
TEST_IMPLEMENTED=True
CHECKPOINT_ROOT=None
NUMERIC_BINDING=None
ACCESS_LOG=None

def access(record):
    if ACCESS_LOG is not None:
        with ACCESS_LOG.open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
FALLEN={'F1':'eingeschraenkt','F2':'eingeschraenkt','F3':'nicht_zutreffend','F4':'eingeschraenkt','F5':'offen','F6':'eingeschraenkt','F7':'offen','F8':'offen'}
SENSITIVITIES=[
    ('1','B4 und B3 auf identischem B4-vollständigem Fallset','B4-Skalierung aus zulässiger Entwicklung nicht verfügbar'),
    ('1b','Grenzsensitivität Echo <=29 Tage','Messtor nicht attestiert'),
    ('2a','Echo ±7 Tage am unveränderten Index','primäres Entwicklungsfallset/Skalierung fehlt'),
    ('2b','LVEF <=40 gegen >40 auf primärem Fallset','primäres Entwicklungsfallset/Skalierung fehlt'),
    ('3','IPW und ungewichtete Analyse desselben ersten technischen Indexsets','Outcome-Skalierung/konfirmatorische Referenzzeit fehlt; Verifikationsfit separat gebunden'),
    ('4','LVEF <=40 gegen LVH positiv/LVEF >=50, identisches TTE','primäre Skalierung/Referenzzeit fehlt; LVH bei LSB weiterhin STOP'),
    ('5','RV bei LVEF >=50, DTW und Fréchet separat','RV-Skalierung/konfirmatorische Referenzzeit fehlt'),
    ('6a','QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster','primäre Skalierung/Referenzzeit fehlt'),
    ('6b','Sinusselektor nach AW054-Regex','primäre Skalierung/Referenzzeit fehlt'),
    ('6c','Alter <65 / >=65, getrennt','primäre Skalierung/Referenzzeit fehlt'),
    ('6d','Geschlecht F / M, getrennt','primäre Skalierung/Referenzzeit fehlt'),
    ('6e','Gemeinsame Echo-/NT-Referenz, log1p(NT-proBNP/URL)','primäre Skalierung/Referenzzeit und vollständiger Einheiten-/Zensierungsvertrag fehlt'),
    ('7','Inverse Dower auf gleicher vollständiger Signalbasis','Vollpopulations-Signaladapter nicht implementiert/attestiert; AW121-Pilot kein Ersatz'),
    ('8a','direct8 centered-RMS','Vollpopulations-Signaladapter nicht implementiert/attestiert'),
    ('8b','direct8_qrs_max_centered und kors_qrs_max_centered getrennt','Vollpopulations-Signaladapter nicht implementiert/attestiert'),
    ('9','Original / train-fixes Huber-Residuum / SO(3)-Formrest, gemeinsame Signalbasis','Vollpopulations-Signaladapter/primäre SD fehlen; Pilot-Residualparameter separat gebunden')]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def now():return datetime.now(timezone.utc).isoformat()

def authorize(role,parameter_path):
    """Vor jeglichem Testinputzugriff; K-Hash plus Gesamtbindung erforderlich."""
    if role!='test':return
    release=Path(__file__).with_name('FREIGABE_K.json')
    if not release.is_file():raise PermissionError('STOP: test ohne K-Freigabedatei verweigert')
    att=json.loads(release.read_text(encoding='utf8'))
    if att.get('runner_sha256')!=sha(__file__) or att.get('parameter_sha256')!=sha(parameter_path):raise PermissionError('STOP: K-Runner-/Parameterhash stimmt nicht')
    if att.get('freigegeben_von')!='K' or not att.get('utc_blindheitsattestation') or att.get('spec_sha256')!=SPEC_HASH:raise PermissionError('STOP: K-Attestation unvollständig')
    if not TEST_IMPLEMENTED:raise PermissionError('STOP: Testadapter/Gesamtparameter fehlen; dieser Entwicklungsrunner ist nicht testfreigabefähig')
    if not att.get('input_manifest_path') or not att.get('input_manifest_sha256'):raise PermissionError('STOP: K-Manifestbindung fehlt')
    if sha(att['input_manifest_path'])!=att['input_manifest_sha256']:raise PermissionError('STOP: K-Manifesthash stimmt nicht')
    try:
        stamp=datetime.fromisoformat(att['utc_blindheitsattestation'].replace('Z','+00:00'))
        if stamp.utcoffset().total_seconds()!=0:raise ValueError('nicht UTC')
    except (ValueError,TypeError,AttributeError):raise PermissionError('STOP: UTC-Blindheitsattest ungültig')
    ledger=PROJECT/'koordination/TESTZUGRIFFE.md'
    if att['input_manifest_sha256'] not in ledger.read_text(encoding='utf8'):raise PermissionError('STOP: vorheriger Testzugriffseintrag mit Manifesthash fehlt')

def bootstrap(z,y,subjects,union,weights=None):
    """1000 Replikate, keine Nachziehung, Skalierung vorab gebunden."""
    import numpy as np
    from .statistik import fit,draws
    from werkzeuge.nachrechnung.resources import guard
    values={k:[] for k in ('beta','or','auc_G','auc_G_marker','delta_auc','auc_marker_original_steigend')};failed={};start=0;checkpoint=None
    if CHECKPOINT_ROOT is not None:
        h=hashlib.sha256()
        for a in (z,y,subjects,union,np.ones(len(y)) if weights is None else weights):
            a=np.asarray(a);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        h.update(sha(__file__).encode());h.update(sha(Path(__file__).with_name('statistik.py')).encode());h.update(str(NUMERIC_BINDING).encode())
        CHECKPOINT_ROOT.mkdir(parents=True,exist_ok=True);checkpoint=CHECKPOINT_ROOT/(h.hexdigest()+'.json')
        if checkpoint.exists():
            state=json.loads(checkpoint.read_text(encoding='utf8'));values=state['values'];failed=state['failed'];start=state['next_r']
    for r in range(start,1000):
        if r%25==0:guard()
        w=draws(union,subjects,r)
        if weights is not None:w=w*np.asarray(weights)
        try:
            result=fit(z,y,subjects,weights=w,wald=False)
            for k in values:values[k].append(result[k])
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            reason=str(exc);failed[reason]=failed.get(reason,0)+1
        if checkpoint is not None and ((r+1)%25==0 or r==999):
            checkpoint.write_text(json.dumps({'next_r':r+1,'values':values,'failed':failed},allow_nan=False),encoding='utf8')
            if (r+1)%250==0:print(json.dumps({'phase':'bootstrap','completed':r+1,'valid':len(values['beta'])}),flush=True)
    return {'planned':1000,'seed':1282026,'valid':len(values['beta']),'failed':failed,
            'ci95':{k:np.quantile(v,[.025,.975],method='linear').tolist() if len(v)>=950 else None for k,v in values.items()},
            'status':'schaetzbar' if len(values['beta'])>=950 else 'nicht_schaetzbar'}

def development(package,out):
    import numpy as np
    from .daten import HYPOTHESES,masks,classify_lvef
    from .statistik import raw_design,fixed_design,fit,holm
    from .export import projected
    from werkzeuge.nachrechnung.resources import guard
    frames={};union=[];flows={};results={};log=[]
    for bef,ep,marker,comp,direction in HYPOTHESES:
        print(json.dumps({'phase':'development_load','endpoint':bef}),flush=True)
        guard();entry=package['hypotheses'][bef];source=entry['input_development'];p=Path(source['path'])
        if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Entwicklungsinput außerhalb AW129-Projektionen')
        if sha(p)!=source['sha256']:raise ValueError('Entwicklungsinputhash geändert')
        log.append({'utc':now(),'role':'entwicklung','path':str(p),'sha256':source['sha256'],'filter':"split='validation'"})
        # Footer zuerst; Entwicklungsdatei enthält ausschließlich train/validation.
        # Ausgabe/Neufit nur validation. Union vor Complete-case-Gates binden.
        f=projected(p);f=f.loc[f.split.eq('validation')].copy()
        union.extend(f.subject_id.dropna().tolist());frames[bef]=f
    from .adapter import run_frames
    from .echonext import STATUS
    technical=None;verification_train=None
    for key in ('technical_development','verification_train'):
        source=package.get(key)
        if source is None:continue
        p=Path(source['path'])
        if not p.resolve().is_relative_to(ROOT.resolve()) or sha(p)!=source['sha256']:raise ValueError('Zusatzinputbindung verändert')
        log.append({'utc':now(),'role':'entwicklung','path':str(p),'sha256':source['sha256']})
        z=projected(p)
        if key=='technical_development':technical=z.loc[z.split.eq('validation')].copy()
        else:verification_train=z
    result=run_frames(frames,package,technical,verification_train,development=True)
    result.update(access_log=log,EchoNext=ECHONEXT_EXCLUDED,claim='Entwicklungs-Trockenlauf; keine unabhängige Prüfung',limitations=['Mess-/Quellenatteste bleiben gesonderte Teststarttore; Entwicklungsfits bestehen diese Tore nicht','Signalvarianten: nicht schätzbar (Skalierung nicht gebunden)'])
    return result

def test_analysis(package):
    """Einzige künftige Testlesestelle, nur nach K-/Hash-/Attestprüfung."""
    import pyarrow.parquet as pq
    import numpy as np
    from .adapter import run_frames
    from .daten import select_index
    from .echonext import STATUS
    att=json.loads(Path(__file__).with_name('FREIGABE_K.json').read_text(encoding='utf8'))
    manifest_path=Path(att['input_manifest_path'])
    if sha(manifest_path)!=att['input_manifest_sha256']:raise PermissionError('Testmanifesthash verändert')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    from .vorbereitung_aw136 import endpoint_stops
    stops=endpoint_stops(manifest,package)
    frames={};log=[]
    def load(source,role):
        p=Path(source['path'])
        if any(s in str(p).lower() for s in ('publication_preparation','diagnostics','locked_')):raise PermissionError('Testergebnisse sind kein Runnerinput')
        access({'utc':now(),'role':role,'path':str(p),'expected_sha256':source['sha256'],'action':'input_access_started'})
        if sha(p)!=source['sha256']:raise PermissionError('Testinputhash verändert')
        schema=pq.ParquetFile(p).schema_arrow.names
        cols=source['columns']
        if set(cols)-set(schema):raise ValueError('Testprojektionsschema fehlt')
        import pyarrow.dataset as ds
        f=ds.dataset(str(p),format='parquet').to_table(columns=cols).to_pandas()
        if set(f.split)!={role}:raise ValueError('physisch getrennte Partition erforderlich')
        log.append({'utc':now(),'role':role,'path':str(p),'sha256':source['sha256'],'columns':cols})
        return f
    for bef,entry in package['hypotheses'].items():
        if bef in stops:
            import pandas as pd
            frames[bef]=pd.DataFrame({'subject_id':np.array([],dtype=np.int64)})
            entry['measurement_attestation']={'status':'STOP','reason':'; '.join(stops[bef])}
            continue
        f=load(manifest['hypotheses'][bef],'test')
        if {'machine_metadata_usable_for_primary','machine_heart_rate_bpm'}-set(f):raise ValueError('testseitiger Geräte-HF-Vertrag fehlt')
        frames[bef]=select_index(f,entry['endpoint'])
        # K attests the existing frozen recipe; no empirical parameters supplied.
        entry['measurement_attestation']=manifest['hypotheses'][bef]['measurement_attestation']
    # §0 Nachtrag 2(b): kein Signalzugriff, auch bei altem Pilot-Attest.
    tech=None;dev=None
    if 'technical' in manifest and not stops and manifest.get('technical_attested') is True:
        tech=select_index(load(manifest['technical'],'test'),'reduced_lvef_strict',technical=True)
        source=package['verification_train'];p=Path(source['path'])
        if sha(p)!=source['sha256']:raise PermissionError('Verifikationsentwicklung verändert')
        from .export import projected
        dev=projected(p)
        if set(dev.split)!={'train'}:raise ValueError('Verifikationsfit nur train')
    # Unabhängigkeit nicht nur als Manifestbehauptung übernehmen: attestierte
    # Entwicklungsprojektionen lesen und die Personenunion intern vergleichen.
    development_ids=[]
    from .export import projected
    for entry in (package['hypotheses'].values() if any(len(f) for f in frames.values()) else []):
        source=entry['input_development'];p=Path(source['path'])
        if sha(p)!=source['sha256']:raise PermissionError('Entwicklungsidentitätsbindung verändert')
        development_ids.extend(projected(p,['subject_id','split']).subject_id.dropna().tolist())
    test_ids=np.unique(np.concatenate([f.subject_id.to_numpy() for f in frames.values()]+([] if tech is None else [tech.subject_id.to_numpy()])))
    if np.intersect1d(np.unique(development_ids),test_ids).size:raise PermissionError('Test-/Entwicklungsidentität überschneidet sich')
    result=run_frames(frames,package,tech,dev)
    result.update(access_log=log,EchoNext=ECHONEXT_EXCLUDED,input_stops=stops,claim='einmalige MIMIC-K1-Prüfung; keine diagnostische Freigabe')
    return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--datenrolle',required=True,choices=['entwicklung','test']);p.add_argument('--parameterpaket',required=True,type=Path);p.add_argument('--ausgabeordner',required=True,type=Path);a=p.parse_args(argv)
    authorize(a.datenrolle,a.parameterpaket)
    if not a.ausgabeordner.resolve().is_relative_to(OUTPUT_ROOT.resolve()):raise ValueError('Ausgabe nur unter <local path>')
    if sha(PROJECT/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md')!=SPEC_HASH:raise ValueError('Freezehash verändert')
    binding=json.loads(Path(__file__).with_name('bindung.json').read_text(encoding='utf8'))
    if binding['spec_sha256']!=SPEC_HASH:raise ValueError('Codebindung gehört zu anderem Freeze')
    for path,expected in binding['files'].items():
        if sha(path)!=expected:raise ValueError('Abhängigkeits-/Parameterhash geändert: '+path)
    package=json.loads(a.parameterpaket.read_text(encoding='utf8'))
    if package['spec_sha256']!=SPEC_HASH or package['role']!='entwicklung':raise ValueError('Parameterpaketvertrag falsch')
    from werkzeuge.nachrechnung.resources import guard,Budget
    mem=guard();a.ausgabeordner.mkdir(parents=True,exist_ok=True)
    target=a.ausgabeordner/'ergebnis.json'
    if target.exists():raise FileExistsError('einmaliger Trockenlauf: vorhandenes Ergebnis nicht überschreiben')
    global CHECKPOINT_ROOT,NUMERIC_BINDING,ACCESS_LOG
    CHECKPOINT_ROOT=a.ausgabeordner/'checkpoints'
    # Bind every computational dependency, not only the visible entry module.
    numeric={k:v for k,v in binding['files'].items() if k.endswith('.py') and '/tests/' not in k.replace('\\','/') and not any(s in k for s in ('bericht_aw131','abgabe.py','migrate_aw131','export.py','extras.py'))}
    NUMERIC_BINDING=hashlib.sha256(json.dumps(numeric,sort_keys=True).encode()).hexdigest()
    ACCESS_LOG=a.ausgabeordner/'zugriffe.jsonl'
    # Keine zeitgesteuerten nativen Stackdumps: deren Abschaltung hing im Entwicklungslauf.
    with Budget(a.ausgabeordner) as budget:
        result=development(package,a.ausgabeordner) if a.datenrolle=='entwicklung' else test_analysis(package)
    result.update({'utc':now(),'spec_sha256':SPEC_HASH,'runner_sha256':sha(__file__),'parameter_sha256':sha(a.parameterpaket),'resources':{'start':mem,**budget.summary()}})
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=lambda value:value.item()),encoding='utf8')
    print(json.dumps({'role':a.datenrolle,'H1':result['results']['BEF-0063']['urteil'],'output':str(target),'test_access':a.datenrolle=='test'},ensure_ascii=False))

if __name__=='__main__':main()
