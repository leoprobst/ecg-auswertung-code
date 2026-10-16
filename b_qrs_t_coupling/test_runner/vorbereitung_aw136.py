"""AW136: technische Bindung; Testdaten ausschließlich Identität/Split.

Kein Label-/Marker-/Outcomezugriff, kein Testfit. Quellenbytes werden nur
gehasht. Das Manifest attestiert fehlende Testprojektionen ausdrücklich false.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
import copy
import json
from pathlib import Path
from .runner import sha,now,PROJECT,SPEC_HASH
from werkzeuge.nachrechnung.resources import guard,Budget

OUT=Path(str(data_path("b.aw136")))
RELEASE=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
DATA=Path(str(data_path("b.mimic_validation")))
GATES=('person_disjoint','preferred_lvef_bound','label_mapping_bound','units_time_bound','release_input_bound','measurement_gates_bound')

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def endpoint_stops(manifest,package):
    """Nicht attestierte Teile werden vor jeder Patientenprojektion gestoppt."""
    stops={}
    for bef,entry in package['hypotheses'].items():
        reasons=[]
        for gate in GATES:
            evidence=manifest.get('attestations',{}).get(gate,{})
            affected=evidence.get('affected_parts',list(package['hypotheses']))
            if gate=='person_disjoint':affected=list(package['hypotheses'])
            if manifest.get(gate) is not True and bef in affected:
                reasons.append(gate+': '+evidence.get('reason','Attestat fehlt'))
            elif manifest.get(gate) is True and not evidence.get('evidence'):
                reasons.append(gate+': Belegkette fehlt')
        src=manifest.get('hypotheses',{}).get(bef,{})
        if not src.get('path') or not src.get('sha256') or not src.get('columns'):
            reasons.append('physisch getrennte gebundene Testeingabe fehlt')
        if reasons:stops[bef]=reasons
    return stops

def code_evidence(path,needle):
    lines=path.read_text(encoding='utf8').splitlines()
    return {'file':str(path),'sha256':sha(path),'code_location':[i+1 for i,s in enumerate(lines) if needle in s],
            'needle':needle,'check_script':str(Path(__file__).resolve()),'check_script_sha256':sha(__file__)}

def identities():
    """Einzige Testprojektion in AW136: exakt subject_id/study_id/split."""
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import numpy as np
    p=DATA/'master/ecg_master_index.parquet'
    schema=pq.ParquetFile(p).schema_arrow.names;before=sha(p)
    cols=['subject_id','study_id','split']
    def log_identity_access(source,source_hash):
        with (OUT/'identity_access_log.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps({'utc':now(),'auftrag':'AW-136','scope':'ausschließlich Disjunktheitsabgleich',
                'path':str(source),'sha256':source_hash,'columns':cols,'action':'identity_projection_start'},ensure_ascii=False)+'\n')
    if set(cols)-set(schema):
        result={'passed':False,'reason':'Identitätsschema fehlt','file':str(p),'sha256':before,'columns':cols}
        save(OUT/'identity_attestation.json',result)
        return result
    people={k:set() for k in ('train','validation','test')};studies={k:set() for k in people};nulls=0;unexpected=set()
    log_identity_access(p,before)
    scanner=ds.dataset(str(p),format='parquet').scanner(columns=cols,batch_size=32768)
    for batch in scanner.to_batches():
        guard();f=batch.to_pandas()
        f['split']=f['split'].replace({'locked_test':'test'})
        nulls+=int(f[cols].isna().any(axis=1).sum())
        unexpected.update(str(v) for v in f.split.unique() if v not in people)
        for role in people:
            z=f.loc[f.split.eq(role)];people[role].update(z.subject_id.dropna().tolist());studies[role].update(z.study_id.dropna().tolist())
    overlap={k:len(people['test']&people[k]) for k in ('train','validation')}
    same_study={k:len(studies['test']&studies[k]) for k in ('train','validation')}
    stable=sha(p)==before
    result={'utc':now(),'passed':bool(stable and not nulls and not unexpected and people['test'] and not any(overlap.values()) and not any(same_study.values())),
            'file':str(p),'sha256':before,'hash_stable':stable,'columns':cols,'filter':'alle Splits, ausschließlich Identität; keine Labels/Marker/Outcomes',
            'person_overlap_test':overlap,'study_overlap_test':same_study,'missing_identity_rows':nulls,'unexpected_splits':sorted(unexpected),
            'person_counts':{k:len(v) for k,v in people.items()},'study_counts':{k:len(v) for k,v in studies.items()},
            'check_script':str(Path(__file__).resolve()),'check_script_sha256':sha(__file__), 'code_location':'identities'}
    # Auch tatsächlich gebundene Entwicklungsprojektionen gegen Test prüfen.
    package=json.loads((PROJECT/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))
    witnesses=[]
    for entry in package['hypotheses'].values():
        src=entry['input_development'];q=Path(src['path']);qs=pq.ParquetFile(q).schema_arrow.names
        if sha(q)!=src['sha256'] or set(cols)-set(qs):raise ValueError('Entwicklungsidentitätsbindung verändert')
        log_identity_access(q,src['sha256'])
        ids=set();roles=set()
        for batch in ds.dataset(str(q),format='parquet').scanner(columns=cols,batch_size=32768).to_batches():
            guard();f=batch.to_pandas();ids.update(f.subject_id.dropna().tolist());roles.update(f.split.unique())
        hit=len(ids&people['test'])
        witnesses.append({'file':str(q),'sha256':src['sha256'],'columns':cols,'test_overlap':hit,'development_only':roles<=set(['train','validation'])})
        result['passed']=result['passed'] and hit==0 and roles<=set(['train','validation'])
    result['development_projection_witnesses']=witnesses
    sealed=DATA/'labels/sealed/locked_endpoint_index_cohorts.parquet'
    # Explizite AW136-Ausnahme: auch hier nur Identitäten, niemals label_id,
    # label_state, Marker, Zeit-/Outcomeinformationen.
    if sealed.is_file():
        ss=pq.ParquetFile(sealed).schema_arrow.names;before_sealed=sha(sealed)
        if set(cols)-set(ss):raise ValueError('Testkohorten-Identitätsschema fehlt')
        log_identity_access(sealed,before_sealed)
        cohort_ids=set();cohort_studies=set();bad=0;roles=set()
        for batch in ds.dataset(str(sealed),format='parquet').scanner(columns=cols,batch_size=32768).to_batches():
            guard();f=batch.to_pandas();roles.update(f.split.unique());bad+=int(f[cols].isna().any(axis=1).sum())
            cohort_ids.update(f.subject_id.dropna().tolist());cohort_studies.update(f.study_id.dropna().tolist())
        overlap_cohort={k:len(cohort_ids&people[k]) for k in ('train','validation')}
        passed=(roles<=set(['locked_test','test']) and not bad and not any(overlap_cohort.values())
                and cohort_ids<=people['test'] and cohort_studies<=studies['test'] and sha(sealed)==before_sealed)
        result['sealed_cohort_identity_witness']={'file':str(sealed),'sha256':before_sealed,'columns':cols,'passed':passed,
            'test_vs_development_overlap':overlap_cohort,'identities_contained_in_master_test':cohort_ids<=people['test'],
            'studies_contained_in_master_test':cohort_studies<=studies['test'],'missing_identity_rows':bad,'split_values':sorted(roles)}
        result['passed']=result['passed'] and passed
    else:result.update(passed=False,reason='bekannte getrennte Testkohortenquelle fehlt')
    result['split_mapping']={'locked_test':'test','train':'train','validation':'validation'}
    save(OUT/'identity_attestation.json',result)
    return result

def manifest():
    ids=identities()
    package=json.loads((PROJECT/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))
    parts=list(package['hypotheses'])
    sources=[DATA/'master/ecg_master_index.parquet',DATA/'labels/sealed/locked_endpoint_index_cohorts.parquet',DATA/'labels/endpoint_index_cohorts.parquet',DATA/'labels/clinical_label_evidence.parquet',
             DATA/'evidence/clinical_context_features.parquet',DATA/'evidence/echo_measurements_wide.parquet',DATA/'evidence/ecg_lab_features.parquet',
             DATA/'labels/labels.manifest.json',DATA/'labels/development_label_attestation.json',DATA/'evidence/evidence.manifest.json',DATA/'evidence/evidence_provenance.json']
    cache=DATA/'runs/A155-2026-10-04-d6e70e/analysis_feature_cache/cache-89d6aaf0a6b5957e/split=test/features.parquet'
    sources.append(cache)
    source_records=[]
    for p in sources:
        guard();exists=p.is_file();source_records.append({'path':str(p),'exists':exists,'sha256':sha(p) if exists else None,
            'read_policy':'nur Identitätsspalten für Disjunktheit' if p.name in ('ecg_master_index.parquet','locked_endpoint_index_cohorts.parquet') else 'nur Dateibytes gehasht; keine Inhaltsprojektion',
            'role':'Quellenkandidat; keine physisch getrennte Runner-Testprojektion'})
    preferred=code_evidence(RELEASE/'mimic/evidence.py','AS preferred_lvef')
    label=code_evidence(RELEASE/'mimic/labels.py','preferred_lvef < 40')
    time=code_evidence(PROJECT/'werkzeuge/b_test/daten.py','strict_alignment=ep in')
    release=code_evidence(RELEASE/'mimic/feature_record.py','MimicWaveformLoader')
    measure=code_evidence(PROJECT/'werkzeuge/b_test/adapter.py','_require_measurement')
    records={
        'person_disjoint':{'value':bool(ids['passed']),'reason':'Identitätsabgleich Master und gebundene Entwicklungsprojektionen; gilt nur für diese Quellen' if ids['passed'] else 'Identitätsabgleich nicht bestanden',
                          'evidence':[{'file':str(OUT/'identity_attestation.json'),'sha256':sha(OUT/'identity_attestation.json'),'code_location':'identities','check_script':str(Path(__file__).resolve()),'check_script_sha256':sha(__file__)}]},
        'preferred_lvef_bound':{'value':False,'reason':'Release-Rezept identifiziert; keine physisch getrennte Testprojektion mit belegter Preferred-/Echo-ID-/Konfliktflag-Provenienz. Quellhash allein ersetzt diese nicht.','evidence':[preferred],'affected_parts':['BEF-0063','BEF-0087']},
        'label_mapping_bound':{'value':False,'reason':'Labelrezept gebunden, aber Test-Projektions-/Index-/Negativmapping für die sechs Endpunkte nicht vorgelegt. development_label_attestation.json gilt ausdrücklich nur train_validation_only und locked_test_semantics_attested=false.','evidence':[label,code_evidence(DATA/'labels/development_label_attestation.json','locked_test_semantics_attested')]},
        'units_time_bound':{'value':False,'reason':'Runner-Zeitregeln gebunden; keine attestierte testseitige Einheiten-/Assay-/URL-/Evidenz-ID-/Referenzzeitprojektion.','evidence':[time]},
        'release_input_bound':{'value':False,'reason':'Keine physisch getrennten B-Testeingaben; Test-Featurecache-Kandidat '+('vorhanden, Erzeugerbindung nicht belegt' if cache.is_file() else 'fehlt')+'. Historische Release-/Inputäquivalenz aus Dateihashes allein nicht belegbar.','evidence':[release]},
        'measurement_gates_bound':{'value':False,'reason':'AW131-Parameter nennen Messattestation STOP; vollständige claimbezogene Test-Messgateakte fehlt. EchoNext ausgeschlossen; Signal-Skalierung nicht gebunden.','evidence':[measure]}}
    for r in records.values():r.setdefault('affected_parts',parts)
    remedies={
        'person_disjoint':'Bei einer späteren B-Testprojektion deren subject_id/study_id/split gegen dieselben gebundenen train-/validation-Identitäten prüfen; dieses Attestat deckt nur die hier genannten Quellen.',
        'preferred_lvef_bound':'Labelblinde Erzeugerakte der Echoquelle mit Preferred-Methodenrangfolge, Konfliktflag, TTE-ID und Zeitzuordnung; Hashkette bis zur getrennten H1-/RV-Testprojektion.',
        'label_mapping_bound':'Zusätzliche mögliche Belegquelle: Erzeugungscode und gehashte, labelblinde Provenienz-/Semantikattestation des Forschungs-Holdouts v2 (Existenz und Äquivalenz hier nicht attestiert). Für jeden Endpunkt positive/negative/unknown-Regeln und wertblinde Indexwahl zum gebundenen Release vergleichen; keine Testergebnisse verwenden.',
        'units_time_bound':'Erzeugerakte mit Echo-Prozenteinheit/Zeiteinheit, ICD-Encounterzuordnung, NT-/HF-Labor- und Troponin-Assay-/URL-/Zensierungsregeln. Stundenendpunkte bleiben bei fehlender Uhrzeit-Eignung unzulässig, selbst wenn die Provenienz später attestiert ist.',
        'release_input_bound':'Gehashtes Input-/Erzeugungsmanifest der gespeicherten Testfeatures mit tatsächlichem Releasecode, Konfiguration, Loader/Filter/Delineation und physisch getrennten B-Projektionen; heutige Quellhashes allein genügen nicht.',
        'measurement_gates_bound':'Claimbezogene Messgateakte für gespeicherte Marker, Geräte-HF, QRS/QTcF, Winkel und Komparatoren; B4 zusätzlich Amplituden-/Achsenvertrag. Signal-Skalierung bleibt separat gesperrt und EchoNext ausgeschlossen.'}
    impacts={
        'person_disjoint':'H1-H4, Troponin, RV und sämtliche personenbasierten Sensitivitäten.',
        'preferred_lvef_bound':'H1 (BEF-0063), H1-Sensitivitäten 1/1b/2a/2b/3/4/6a-6e sowie RV (BEF-0087) einschließlich Rang 5 mit LVEF-Kontrolle. Rang 4 benötigt zusätzlich gebundene LVH-Referenz.',
        'label_mapping_bound':'Alle sechs Endpunkte: H1 BEF-0063, H2 BEF-0042, H3 BEF-0081, H4 BEF-0049, Troponin BEF-0075, RV BEF-0087; sämtliche zugehörigen Label-/Referenzsensitivitäten. H1 ist trotz numerischer Neuklassifikation auf gültige Echo-/Indexsemantik angewiesen. Signalvarianten sind unabhängig davon gesperrt.',
        'units_time_bound':'Alle sechs Endpunkte und ihre Zeit-/Referenzsensitivitäten; NT/HF/Troponin zusätzlich Stundenzeitgate, Rang 6e NT-Einheiten-/URL-Vertrag.',
        'release_input_bound':'Alle sechs gespeicherten Marker-/Modellanalysen, B3/B4, Strata und IPW/Verifikationsanalyse; keine Inputsubstitution.',
        'measurement_gates_bound':'Alle sechs Marker-/Modellteile und Sensitivitäten einschließlich B4/Verifikation; Signalvarianten 7/8a/8b/9 separat nicht schätzbar, EchoNext ausgeschlossen.'}
    for gate,r in records.items():
        r['affected_analyses']=impacts[gate]
        r['additional_evidence_needed']=remedies[gate]
    obj={'schema_version':'AW136-1','utc':now(),'run':'A155-2026-10-04-d6e70e','spec_sha256':SPEC_HASH,
         'test_access_scope':'nur Identitäten für Disjunktheit und SHA-256-Bytehashes; keine Testergebnisse',
         'attestations':records,**{k:v['value'] for k,v in records.items()},'source_candidates':source_records,
         'hypotheses':{k:{'path':None,'sha256':None,'columns':[], 'measurement_attestation':{'status':'STOP','reason':records['measurement_gates_bound']['reason']}} for k in parts},
         'signal_adapter_attested':False,'EchoNext':'ausgeschlossen §0 Nachtrag 2(c)','check_script_sha256':sha(__file__)}
    obj['stopped_parts']=endpoint_stops(obj,package)
    save(OUT/'input_manifest.json',obj)
    print(json.dumps({'phase':'manifest','attestations':{k:obj[k] for k in GATES},'stopped_parts':list(obj['stopped_parts'])}),flush=True)

def parameters():
    p=PROJECT/'werkzeuge/b_test/parameter_entwicklung.json'
    old=json.loads(p.read_text(encoding='utf8'))
    if not (OUT/'parameter_aw131_archive.json').exists():(OUT/'parameter_aw131_archive.json').write_bytes(p.read_bytes())
    new=copy.deepcopy(old);new.update(spec_sha256=SPEC_HASH,schema_version='AW136-4',status='v4-Bindung; empirische Entwicklungsparameter aus AW131 unverändert')
    new['signal_variants_policy']='nicht schätzbar (Skalierung nicht gebunden)'
    new['EchoNext_policy']='ausgeschlossen'
    save(p,new)
    numeric_keys=['hypotheses','verification','residual','variants','seeds','sources','technical_development','verification_train']
    assert all(old.get(k)==new.get(k) for k in numeric_keys)
    save(OUT/'parameter_migration.json',{'source_sha256':sha(OUT/'parameter_aw131_archive.json'),'target_sha256':sha(p),'numeric_objects_unchanged':numeric_keys,'spec_sha256':SPEC_HASH,'utc':now()})

def bind():
    from . import export
    # Binder wiederverwenden, nur v4 und neues erlaubtes Ausgabeziel.
    export.OUT=OUT
    export.bind()
    b=PROJECT/'werkzeuge/b_test/bindung.json'
    obj=json.loads(b.read_text(encoding='utf8'));obj['spec_sha256']=SPEC_HASH
    for p in [OUT/'input_manifest.json',OUT/'identity_attestation.json']:
        obj['files'][str(p)]=sha(p)
    tests=OUT/'tests/junit.xml'
    if tests.exists():obj['files'][str(tests)]=sha(tests)
    obj['status']='AW136 v4 technische Bindung; KEINE Testfreigabe; fehlende Attestate stoppen betroffene Teile'
    save(b,obj)
    files={**obj['files'],str(b):sha(b),str(PROJECT/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md'):SPEC_HASH}
    save(OUT/'hashes.json',files)
    (OUT/'hashes.md').write_text('| Datei | SHA-256 |\n|---|---|\n'+'\n'.join('| '+p+' | `'+h+'` |' for p,h in files.items())+'\n',encoding='utf8')
    draft={'status':'ENTWURF, keine Freigabe; Blindheitsattest durch K zu ergänzen','runner_sha256':sha(PROJECT/'werkzeuge/b_test/runner.py'),
           'parameter_sha256':sha(PROJECT/'werkzeuge/b_test/parameter_entwicklung.json'),'spec_sha256':SPEC_HASH,
           'input_manifest_path':str(OUT/'input_manifest.json'),'input_manifest_sha256':sha(OUT/'input_manifest.json'),
           'utc_blindheitsattestation':None,'binding_path':str(b),'binding_sha256':sha(b),'EchoNext_excluded':True,
           'access_register_path':str(PROJECT/'koordination/TESTZUGRIFFE.md'),'access_register_requirement':'K trägt autorisierten Start vor erstem Testzugriff ein; AW136 erlaubt nur Identitätsabgleich',
           'not_estimable_parts':list(json.loads((OUT/'input_manifest.json').read_text(encoding='utf8'))['stopped_parts'])}
    save(OUT/'FREIGABE_K_entwurf.json',draft)

def report():
    import xml.etree.ElementTree as ET
    m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf8'))
    rpath=OUT/'trockenlauf_validation/ergebnis.json'
    r=json.loads(rpath.read_text(encoding='utf8'))
    hashes=json.loads((OUT/'hashes.json').read_text(encoding='utf8'))
    tests=ET.parse(OUT/'tests/junit.xml').getroot()
    suite=tests.find('testsuite') if tests.tag=='testsuites' else tests
    assert int(suite.attrib.get('failures',0))==0 and int(suite.attrib.get('errors',0))==0
    assert r['role']=='entwicklung' and r['partition']=='validation'
    assert r['EchoNext']['status']=='ausgeschlossen'
    assert all(z['reason']=='nicht schätzbar (Skalierung nicht gebunden)' for z in r['sensitivities'] if z['rank'] in ['7','8a','8b','9'])
    for p,h in hashes.items():assert sha(p)==h,p
    draft=OUT/'FREIGABE_K_entwurf.json'
    assert 'freigegeben_von' not in json.loads(draft.read_text(encoding='utf8'))
    rows=[]
    for k,v in r['results'].items():
        rows.append('| '+k+' | '+v['status']+' | '+str(v.get('N','-'))+' | '+str(v.get('or','-'))+' | '+str(v.get('bootstrap',{}).get('valid','-'))+' | '+v.get('reason','')+' |')
    strata=[]
    for v in r['sensitivities']:
        if v['rank']=='6d':strata.append('| '+v['analysis']+' | '+v['status']+' | '+str(v.get('N','-'))+' | '+str(v.get('or','-'))+' | '+v.get('reason','sex_numeric strukturell ausgelassen')+' |')
    att='\n'.join('| '+k+' | '+str(m[k]).lower()+' | '+m['attestations'][k]['reason']+' |' for k in GATES)
    followup='\n'.join('| '+k+' | '+m['attestations'][k]['affected_analyses']+' | '+m['attestations'][k]['additional_evidence_needed']+' |' for k in GATES)
    resources=[]
    for p in sorted(OUT.glob('resources_*.json'))+[OUT/'tests/resources.json']:
        x=json.loads(p.read_text(encoding='utf8'));resources.append('| '+str(p)+' | '+str(x['start']['ram_free_bytes'])+' | '+str(x['start']['commit_free_bytes'])+' | '+str(x['peak_rss_bytes'])+' | '+str(x['peak_private_commit_bytes'])+' |')
    x=r['resources'];resources.append('| '+str(rpath)+'/resources | '+str(x['start']['ram_free_bytes'])+' | '+str(x['start']['commit_free_bytes'])+' | '+str(x['peak_rss_bytes'])+' | '+str(x['peak_private_commit_bytes'])+' |')
    text=f'''# AW-136 - B-Teststart, v4-Bindung ohne Testrechnung

{now()} · MS/PR · Auftrag AW-136, AGENTS.md vollständig gelesen. Keine Recherche, Installation oder Außenkommunikation.

## Ergebnis und Startstatus

Nachtrag 2(a)-(c) umgesetzt und synthetisch geprüft. Geschlechtsstrata F/M entfernen ausschließlich `sex_numeric` vor Fit und Bootstrap; alle anderen Regressoren und Entwicklungs-Skalierungen bleiben erhalten. Keine testseitige Konstantenentfernung oder Neuskalierung. Signalvarianten 7/8a/8b/9 werden stets als **nicht schätzbar (Skalierung nicht gebunden)** ausgegeben, auch bei vorhandenen Pilotspalten. Der Runner liest hierfür keine Signale. EchoNext ist ausdrücklich ausgeschlossen; der separate Adapter bleibt für diesen Lauf unbenutzt und benötigt weiterhin AW123.

**Keine Testrechnung, keine Testlabels/-marker/-outcomes und keine Testergebnisse gelesen.** Ausschließlich Identitätsspalten wurden für Disjunktheit projiziert; weitere Quellen nur byteweise gehasht. Die dokumentierten Quellenkandidaten sind keine gültigen physisch getrennten B-Testinputs. Fehlende Attestate bleiben false; alle sechs B-Endpunktteile sind derzeit gestoppt. Dies ist eine technische Nichtschätzbarkeit und kein biologischer Nullbefund.

## Tests und Entwicklungstrockenlauf

Alle {suite.attrib['tests']} Tests bestanden (Failures/Errors jeweils 0; JUnit `{OUT/'tests/junit.xml'}`, SHA-256 `{sha(OUT/'tests/junit.xml')}`). Neue Prüfungen: reale synthetische B3-/B4-Stratumfits F/M, feste Marker-SD und unveränderte Parameter, Konstante ohne Stratumregel STOP, andere konstante Kovariate STOP, gemischtes Stratum STOP, Signal-STOP trotz Pilotwerten, endpunktbezogene false-Atteste, vollständiger STOP ohne Inputlesen, EchoNext-Ausschluss. Vorhandene Split-/CR1-/Separation-/Holm-/Bootstrap-/Referenzzeit-/Grenztests ebenfalls bestanden.

Beim ersten Testlauf bestanden 80 Tests, ein alter AW131-Manifesttest scheiterte: seine synthetischen true-Attestate enthielten keine Belegkette, die v4 nun verlangt. Der Test wurde um ausdrücklich synthetische Belege ergänzt; das produktive Beleggate blieb unverändert. Der anschließende vollständige Lauf ist der oben gehashte Nachweis.

Trockenlauf ausschließlich **validation, Entwicklung**, Quellen A155-2026-10-04-d6e70e, gebundene AW131-Projektionen unter <local path>. Je Datei vor Lesen SHA-Abgleich, Footer zuerst, train/validation-Filter; Fits ausschließlich validation. Parameterpaket-Hauptobjekte einschließlich Hypothesen, Verifikation, Residuum, Varianten und Seeds unverändert übernommen; nur v4-Metadaten/Signal-/EchoNext-Politik ergänzt. Beleg `{OUT/'parameter_migration.json'}` und archiviertes AW131-Paket. Keine unabhängige Bestätigung und keine neue Erfolgs-/Parameterwahl.

Quelle jeder folgenden Zahl: `{rpath}`, SHA-256 `{sha(rpath)}`, Schlüssel results/BEF bzw. sensitivities/rank=6d/analysis; Skript `werkzeuge/b_test/runner.py` → `adapter.py`, Filter validation. OR je feste Entwicklungs-SD, apparenter Assoziationsneufit. Bootstrap-Nichtschätzbarkeit bleibt vom Waldtest getrennt.

| Endpunkt | Status | N | OR | gültige Bootstrapreplikate | Stoppgrund |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

| Geschlechtsstratum | Status | N | OR | Strukturregel/Stoppgrund |
|---|---|---:|---:|---|
{chr(10).join(strata)}

## Eingabemanifest und Attestate

Manifest `{OUT/'input_manifest.json'}`, SHA-256 `{sha(OUT/'input_manifest.json')}`. Jedes Attestat nennt affected_parts und evidence mit Datei, SHA-256, Codezeile/-funktion sowie Prüfskript/Hash. Quellenkandidaten einschließlich fehlendem Test-Featurecache werden einzeln mit Pfad/Existenz/Hash geführt; fehlende Dateien erhalten null statt erfundenem Hash. Runner-Hypotheseneingaben bleiben null, solange keine legitime getrennte Testprojektion vorliegt. Keine Projektion von Labels oder Markern zu ihrer Herstellung in AW136.

| Attestat | Wert | Begründung |
|---|---|---|
{att}

Die folgende Zuordnung trennt den derzeitigen Stopp vom möglichen zusätzlichen Nachweis. Angaben zu Forschungs-Holdout v2 sind eine Beleganforderung, keine Behauptung seiner Existenz oder Gleichwertigkeit.

| Attestat | Betroffene B-Teile | Zusätzliche Belegquelle / erforderlicher Nachweis |
|---|---|---|
{followup}

Disjunktheitsbelege in `{OUT/'identity_attestation.json'}`: Master und alle sechs tatsächlich gebundenen Entwicklungsprojektionen; ausschließlich subject_id, study_id, split, Batchgröße 32768, keine individuellen Identitäten in Projekt oder Protokoll. Test gegen train/validation, Personen- und Studienüberschneidung, fehlende Identität, Quellhash vor/nach Prüfung. Das positive Attestat gilt nur für die konkret geprüften Quellen und attestiert keine fehlende spätere B-Testprojektion.

Das vorhandene `development_label_attestation.json` attestiert ausdrücklich train_validation_only und locked_test_semantics_attested=false. Ein Release-Rezept und heutige Hashes beweisen keine testseitige Erzeugeridentität, Preferred-/Echozuordnung, Assay-/Einheitenprovenienz oder bestandene claimbezogene Messgates. Deshalb keine pauschale Umdeutung auf true. Endpunktbezogener STOP wird **vor** Inputprojektion geprüft; fehlende optionale Tore verkleinern die Holm-Familie nicht. Fehlt H1, bleibt das konfirmatorische Sekundärgate geschlossen.

## Freigabeentwurf und Wiederaufnahme

`{draft}` (SHA-256 `{sha(draft)}`) enthält runner_sha256, parameter_sha256, spec_sha256, input_manifest_path/-sha256, utc_blindheitsattestation, Bindung und Startregisterhinweis; das Feld **freigegeben_von fehlt vollständig**. UTC-Blindheitsattestation ist null und von K selbst zu ergänzen. Keine Datei FREIGABE_K.json erzeugt. K muss vor autorisiertem Zugriff die verbleibenden Bindungen herstellen, die neue Gesamtbindung prüfen und den Manifesthash im Testzugriffsregister eintragen. Der Runner verweigert Testmodus ohne passende K-Datei, UTC-Attest, Manifesthash und vorherigen Registereintrag.

Ein technischer Abbruch wird mit demselben Kommando, Quellen, Parametern, 1000 Replikaten und Seed1282026 fortgesetzt. Checkpoints alle 25 Replikate binden Design/Labels/Identität/Gewichte, Runner-/Statistikhash und sämtliche numerischen Abhängigkeiten; IPW bindet getrennte Universen/Parameter. Keine Nachziehung fehlgeschlagener Replikate. Vorhandenes finales ergebnis.json wird nicht überschrieben. AW131-Serialisierungs-Recovery ist nicht auf v4 übertragbar; ihre alten Hashprüfungen verweigern diese Verwendung.

**Technische AW136-Recovery:** Der erste Entwicklungslauf schloss alle 34 Modell-Checkpoints ab, schrieb aber kein finales Ergebnis. Die eigene CPU-Zeit blieb bei zwei Abfragen unverändert (1454,15625 s); die zeitgesteuerte native Stackdiagnose endete partiell. Ein Stillstand beim Abschalten der Diagnose ist plausibel, jedoch nicht nativ bewiesen. Ausschließlich eigener PID 66748 wurde beendet. `recovery_aw136.py` archiviert den alten Runner und die Bindung, weist bytegenau die ausschließliche Entfernung der zeitgesteuerten Diagnose nach und prüft alle numerischen Abhängigkeiten. Die Recovery ist ausschließlich Entwicklung: aktueller CLI-/Gate-/Punktfit-/Exportpfad, exakt archivierter Bootstrap mit ursprünglichem Runnerhash und numerischem Bindungshash, dieselben vollständigen Checkpoints. Berichtsfunktion-Änderungen werden per AST auf reine Dokumentation begrenzt. Der normale künftige Runner hat keine zeitgesteuerte Stackdiagnose mehr und erhält neue Hashes. Es wurden Entwicklungsergebnisse und Bootstrapzählungen gesehen, keine Testergebnisse; die technische Korrektur ändert kein Estimand und keine Rechenregel.

Quelle: ergebnis.json/diagnostic_recovery und <local path>; alter/new_runner_sha256, parameter_sha256 und numeric_binding_before dort. Quellen, Replikate, Seeds, feste Skalierungen und fehlgeschlagene Replikate unverändert; kein Neulauf mit geänderten analytischen Einstellungen.

## Befehle und Ressourcen

Aus Projektwurzel, jeweils start.ps1 mit GlobalMemoryStatusEx vor Python und nativen Threads=1:

```powershell
./werkzeuge/b_test/start.ps1 -Mode aw136 -Role parameters
./werkzeuge/b_test/start.ps1 -Mode aw136 -Role manifest
./werkzeuge/b_test/start.ps1 -Mode tests
./werkzeuge/b_test/start.ps1 -Mode aw136 -Role bind
./werkzeuge/b_test/start.ps1 -Mode runner -Role entwicklung -Output <local path>
./werkzeuge/b_test/start.ps1 -Mode aw136 -Role report
```

Nach dem dokumentierten technischen Stillstand zusätzlich: `start.ps1 -Mode recovery136_prepare` (vor Diagnosekorrektur); danach Tests, Manifest und Bindung erneut, `start.ps1 -Mode recovery136_resume`; abschließend `start.ps1 -Mode aw136 -Role report`. Die archivierte Bindung hält den ursprünglichen Rechenstand fest.

Ein Pythonprozess; vor jedem Start ≥4 GiB freier RAM und Commit, Prozessbudget 1,5 GiB, erneute Guards je Projektion/Bootstrapblock. Alle privaten Zwischenstände nur <local path>; keine Patientendaten im Projekt. Herkunft Ressourcen: die jeweils genannte JSON-Datei, start/peak-Felder, werkzeuge/nachrechnung/resources.py und start.ps1. Bytes:

| Datei/Phase | RAM frei Start | Commit frei Start | Peak RSS | Peak privater Commit |
|---|---:|---:|---:|---:|
{chr(10).join(resources)}

## Prüfkette F1-F8, Grenzen und offene Fragen

| Falle | Status und Grenze |
|---|---|
| F1 | eingeschränkt: Echo/Labor/ICD statt Markerdefinition; Orientierungsbeziehung und HF/ICD-Proxys begrenzen Interpretation |
| F2 | eingeschränkt: verifizierte Krankenhauspopulation, kein Bevölkerungs-/PPV-/NPV-Claim |
| F3 | nicht_zutreffend: feste K1-Grenzen, keine Arbeitspunktoptimierung |
| F4 | eingeschränkt: Entwicklungsauswahl bleibt post hoc; AW136 ist keine unabhängige Prüfung; K-Blindheit separat erforderlich |
| F5 | offen: informative Complete-case-/Referenz-/Signal-Missingness; STOP ist kein Nullbefund |
| F6 | eingeschränkt: B3-Winkel obligat; B4 gepaart, kein vollständiger Amplituden-/Confoundingbeweis |
| F7 | offen: Echoindikation, Uhrzeit und Verifikation; IPW kein MAR-Beweis, keine nominale Stundenersatzrechnung |
| F8 | offen: Filter/Delineation/Gerät/Pacing; fehlende Messgateakte, Signalvarianten gesperrt, EchoNext ausgeschlossen |

P01a heutige Inhaltsbindung, keine historische Erzeugerattestation. P01b Entwicklungstrockenlauf ausgeführt; P01c keine reale individuelle Signalneuberechnung. P05 Rollentrennung/Identitätsabgleich. P08 CR1/Patientenbootstrap unverändert; keine unabhängige Testinferenz. P18 Strukturregel synthetisch und in validation geprüft. P20 Kalibrierung unter K1 nicht konfirmatorisch prüfbar. P23 keine neue Literatur.

Offen für K: getrennte legitime Testeingaben und deren Release-/Referenz-/Label-/Einheiten-/Messatteste; eigener UTC-Blindheitsvermerk und Testzugriffsregistereintrag. Unveränderte technische Bindung muss vor Testfreigabe erneut verifiziert werden. Keine fachliche Promotion. Widerlegen/begrenzen würden Gegenrichtung, kein Zusatz auf gleicher stärkerer Basis, IPW-Richtungsumkehr oder ungültige Mess-/Referenzprovenienz.

Quellenfehler-Kandidaten: keine neuen Literaturquellen geprüft. Nebenbefund: Entwicklungs-Labelattest deckt Testsemantik ausdrücklich nicht; keine Änderung außerhalb des Auftrags.

Sicherheit: hoch für getestete Strukturregel/STOPs, unveränderte numerische Parameter und heutige Hashbindung; begrenzt für historische Releaseprovenienz, Testreferenzen und Messgates, daher false. Klinische Aussagen bleiben Vorschläge, keine ärztliche Freigabe.

## Hash-Tabelle

Bindung, Modul-/Rezept-/Parameter-/Test-/Umgebung-/Manifesthashes: `{OUT/'hashes.json'}` und `{OUT/'hashes.md'}`. Heutige Inhaltsnachweise; keine vollständige Wheel-/DLL-Binärbindung oder historische Unveränderlichkeit.

{(OUT/'hashes.md').read_text(encoding='utf8')}
'''
    report_path=PROJECT/'nachrechnung/B_TESTSTART_AW136.md'
    report_path.write_text(text,encoding='utf8')
    files=['adapter.py','runner.py','start.ps1','export.py','vorbereitung_aw136.py','recovery_aw136.py','parameter_entwicklung.json','bindung.json','umgebung.json','tests/conftest.py','tests/test_aw131.py','tests/test_aw136.py']
    delivery=f'''# Abgabe AW-136

{now()} · MS/PR · Deutsch · Auftrag bearbeitet, K-Abnahme offen.

## Ergebnis

Nachtrag 2(a)-(c) umgesetzt; alle {suite.attrib['tests']} Tests bestanden; neuer validation-Trockenlauf abgeschlossen (Entwicklung). Eingabemanifest, Identitätsabgleich, neue v4-Code-/Parameter-/Umgebungs-/Testbindung und Freigabeentwurf liegen vor. Keine Testrechnung, keine Testlabels/Marker/Outcomes/Ergebnisse gelesen. Testidentitäten nur zum erlaubten Disjunktheitsabgleich projiziert. **Alle sechs B-Testteile derzeit nicht schätzbar wegen fehlender Eingabebindungen; kein biologischer Nullbefund.**

## Dateien

Geändert/neu unter werkzeuge/b_test: {', '.join(files)}. Neu nachrechnung/B_TESTSTART_AW136.md und diese Abgabe. Private Prüfbelege/Zwischenstände unter <local path>: input_manifest.json, identity_attestation.json, parameter_aw131_archive.json, parameter_migration.json, FREIGABE_K_entwurf.json, hashes.json/md, tests/, resources_*.json und trockenlauf_validation/ einschließlich Checkpoints. Keine Änderungen an Koordinationsregister, Spezifikation, gemeinsamen Werkzeugen, laufenden Analyseprozessen oder Pipelinequellen.

## Belege

Vollständige Befehle, Resultate mit Quelldatei/Hash/Filter/Schlüssel/Skript, F1-F8 und Hash-Tabelle: nachrechnung/B_TESTSTART_AW136.md, SHA-256 `{sha(report_path)}`. JUnit `{sha(OUT/'tests/junit.xml')}`; validation-ergebnis.json `{sha(rpath)}`; Manifest `{sha(OUT/'input_manifest.json')}`; Freigabeentwurf `{sha(draft)}`. Empirische AW131-Parameterobjekte unverändert, nachgewiesen in parameter_migration.json. Kein Feld freigegeben_von im Entwurf, keine FREIGABE_K.json angelegt.

## Offene Fragen / Auflagen für K

Manifest enthält die sechs Attestate mit Einzelbelegen/affected_parts; nicht belegte Tore false. K benötigt physisch getrennte B-Testeingaben und deren komplette Provenienz/Messgateakte sowie eigenen UTC-Blindheitsvermerk und vorherigen Zugriffsregistereintrag. Ein positives Quellen-/Identitätsattest ersetzt keine fehlende B-Testprojektion. Freigabeentwurf ist keine Startfreigabe. EchoNext bleibt ausgeschlossen; Signalvarianten benötigen vor einer Nachholung eigenen gebundenen Auftrag.

Konkrete Auswirkungen und mögliche Zusatzbelege je Attestat stehen im Bericht und im Manifest unter affected_analyses/additional_evidence_needed. Insbesondere betrifft locked_test_semantics_attested=false H1-H4, Troponin, RV und deren Referenzsensitivitäten. Als zusätzliche Belegquelle kommt die labelblinde Erzeuger-/Semantikakte des Forschungs-Holdouts v2 in Betracht; Existenz, Pfad und Äquivalenz sind hier nicht attestiert. Kein Ersatz durch Entwicklungsattest oder Testergebnissicht.

## Nebenbefunde und Quellenfehler-Kandidaten

development_label_attestation.json gilt nur train_validation_only; locked_test_semantics_attested=false. Dokumentierter Test-Featurecache-Kandidat fehlt. Keine neuen Literaturquellen geprüft, keine Literaturfehler-Kandidaten. Entwicklung ist keine unabhängige Prüfung; alle technische STOPs mit F1-F8 im Bericht.

Technischer Stillstand nach abgeschlossenen Checkpoints: eigener Prozess beendet, ausschließlich zeitgesteuerte native Stackdiagnose entfernt; hash-/quellbelegte Entwicklungs-Recovery mit altem Bootstrap und ursprünglicher numerischer Bindung. Vollständige Dokumentation und ursprüngliche/neue Hashes im Bericht und ergebnis.json/diagnostic_recovery. Keine analytische Änderung nach Entwicklungssicht, keine Testsicht.

## Sicherheit

Hoch für getestete Implementierung und aktuelle Inhaltsbindung; historische Input-/Releaseprovenienz und vollständige Test-Messgates nicht belegt, daher false. Keine klinische Promotion oder Testfreigabe.
'''
    (PROJECT/'koordination/abgaben/AW-136.md').write_text(delivery,encoding='utf8')
    save(OUT/'final_verification.json',{'utc':now(),'tests':suite.attrib,'all_binding_hashes_verified':True,'no_freigegeben_von':True,'report_sha256':sha(report_path),'delivery_sha256':sha(PROJECT/'koordination/abgaben/AW-136.md')})
    print(json.dumps({'phase':'report','tests':suite.attrib['tests'],'delivery':'koordination/abgaben/AW-136.md'}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['parameters','manifest','bind','report']);a=parser.parse_args()
    start=guard();OUT.mkdir(parents=True,exist_ok=True)
    if sha(PROJECT/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md')!=SPEC_HASH:raise ValueError('v4-Hash falsch')
    with Budget(OUT) as b:globals()[a.phase]()
    save(OUT/('resources_'+a.phase+'.json'),{'start':start,**b.summary()})

if __name__=='__main__':main()
