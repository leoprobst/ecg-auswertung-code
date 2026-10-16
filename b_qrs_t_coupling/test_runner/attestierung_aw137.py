"""Technische Manifest-/Identitätsprüfung AW137, ohne Marker-Endpunkt-Rechnung."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
from pathlib import Path
from datetime import datetime,timezone
from . import testeingaben_aw137 as a

def evidence(path,location):
    return {'file':str(path),'sha256':a.sha(path),'code_location':location,
       'check_script':str(Path(__file__).resolve()),'check_script_sha256':a.sha(__file__),
       'projection_script':str(Path(a.__file__).resolve()),'projection_script_sha256':a.sha(a.__file__)}

def identities(projections,package):
    import pyarrow.dataset as ds
    from werkzeuge.nachrechnung.resources import guard
    cols=['subject_id','study_id','split'];master=a.ROOT/'master/ecg_master_index.parquet'
    people={s:set() for s in ('train','validation','locked_test')};studies={s:set() for s in people}
    before=a.sha(master);a.footer(master);a.record(master,cols,None,'identity_scan');bad=0
    for batch in ds.dataset(str(master),format='parquet').scanner(columns=cols,batch_size=32768).to_batches():
        guard();f=batch.to_pandas();bad+=int(f[cols].isna().any(axis=1).sum())
        if not set(f.split)<=set(people):raise ValueError('Unerwartete Masterpartition')
        for s in people:
            part=f.loc[f.split.eq(s)];people[s].update(part.subject_id.tolist());studies[s].update(part.study_id.tolist())
    development=[]
    for bef,entry in package['hypotheses'].items():
        source=entry['input_development'];p=Path(source['path']);a.footer(p)
        if a.sha(p)!=source['sha256']:raise ValueError('Entwicklungsquelle geändert')
        a.record(p,cols,'train+validation; nur Identität')
        ids=set();roles=set()
        for batch in ds.dataset(str(p),format='parquet').scanner(columns=cols,batch_size=32768).to_batches():
            guard();f=batch.to_pandas();ids.update(f.subject_id.dropna().tolist());roles.update(f.split.unique())
        development.append({'bef':bef,'path':str(p),'sha256':source['sha256'],
            'test_overlap':len(ids&people['locked_test']),'development_only':roles<=set(['train','validation'])})
    witnesses=[]
    for bef,source in list(projections['hypotheses'].items())+[('technical',projections['technical'])]:
        p=Path(source['path']);a.footer(p)
        if a.sha(p)!=source['sha256']:raise ValueError('B-Testprojektion geändert')
        f=a.projected(p,cols+['source_split'])
        ids=set(f.subject_id);keys=set(f.study_id)
        item={'part':bef,'path':str(p),'sha256':source['sha256'],'rows':len(f),'persons':len(ids),
          'person_overlap_train':len(ids&people['train']),'person_overlap_validation':len(ids&people['validation']),
          'study_overlap_train':len(keys&studies['train']),'study_overlap_validation':len(keys&studies['validation']),
          'test_persons_only':ids<=people['locked_test'],'test_studies_only':keys<=studies['locked_test'],
          'physical_split_test_only':set(f.split)=={'test'},'source_split_locked_test_only':set(f.source_split)=={'locked_test'},
          'missing_identity_rows':int(f[cols].isna().any(axis=1).sum()),
          'duplicate_subject_study_rows':int(f.duplicated(['subject_id','study_id']).sum())}
        witnesses.append(item)
    stable=a.sha(master)==before
    passed=stable and bad==0 and not any(people['locked_test']&people[s] for s in ('train','validation'))
    passed &= all(x['test_overlap']==0 and x['development_only'] for x in development)
    passed &= all(all(x[k] for k in ('test_persons_only','test_studies_only','physical_split_test_only','source_split_locked_test_only')) and
       not any(x[k] for k in ('person_overlap_train','person_overlap_validation','study_overlap_train','study_overlap_validation','missing_identity_rows','duplicate_subject_study_rows')) for x in witnesses)
    result={'passed':bool(passed),'master_sha256':before,'master_stable':stable,'master_missing_identity_rows':bad,
       'master_person_counts':{k:len(v) for k,v in people.items()},'development_witnesses':development,'projection_witnesses':witnesses,
       'person_overlap_master':{s:len(people['locked_test']&people[s]) for s in ('train','validation')},
       'script_sha256':a.sha(__file__)}
    a.save('person_disjoint_attestation.json',result)
    return result

def build():
    package=a.read_json(a.PROJECT/'werkzeuge/b_test/parameter_entwicklung.json')
    projections=a.read_json(a.OUT/'projections.json');ids=identities(projections,package)
    sem=a.read_json(a.OUT/'label_semantics.json');pref=a.read_json(a.OUT/'preferred_lvef_attestation.json')
    inventory=a.read_json(a.OUT/'feature_inventory.json');cov=a.read_json(a.OUT/'index_coverage.json')
    provenance=a.read_json(a.OUT/'provenance.json');old=a.read_json(Path(str(data_path("b.aw136_input_manifest_json"))))
    parts=list(package['hypotheses']);hours=['BEF-0042','BEF-0081','BEF-0075']
    old.update(schema_version='AW137-2',utc=datetime.now(timezone.utc).isoformat(),
        test_access_scope='AW137: Testmerkmale/Testreferenzen/Testindizes; technische Aggregate ohne Marker-Endpunkt-Zusammenhänge; keine Ergebnis-/Fitdateien',
        hypotheses=projections['hypotheses'],technical=projections['technical'],technical_attested=False,
        source_candidates=[],attestations={},locked_test_semantics_attested={ep:v['locked_test_semantics_attested'] for ep,v in sem['endpoints'].items()},
        parent_manifest={'path':str(data_path("b.aw136_input_manifest_json")),'sha256':a.sha(str(data_path("b.aw136_input_manifest_json")))},
        projector_sha256=a.sha(a.__file__),attestation_script_sha256=a.sha(__file__))
    def gate(name,value,reason,affected,evidences,needed):
        old[name]=bool(value)
        old['attestations'][name]={'value':bool(value),'reason':reason,'affected_parts':affected,
          'affected_analyses':'B3/B4 und zugehörige Referenz-/Stratumssensitivitäten der genannten Teile',
          'evidence':evidences,'additional_evidence_needed':needed}
    gate('person_disjoint',ids['passed'],'Alle physisch getrennten B-Projektionen gegen gesamten Master und sechs hashgebundene Entwicklungsprojektionen geprüft.',parts,
        [evidence(a.OUT/'person_disjoint_attestation.json','identities')],'Kein weiterer Identitätsnachweis für diese Hashversion nötig; Änderungen neu prüfen.')
    gate('preferred_lvef_bound',pref['passed'],'Preferred-Rangfolge 3d/biplane/generic, Methodenname, Konfliktflag und Echo-ID/Zeit auf Testquellen ohne Abweichung geprüft.',
        ['BEF-0063','BEF-0087'],[evidence(a.OUT/'preferred_lvef_attestation.json','projection_build'),evidence(a.RELEASE/'mimic/evidence.py','2709-2749')],
        'Keine Änderung der Referenzwahl; Indexabdeckung/Messgates bleiben eigene Stopps.')
    gate('label_mapping_bound',all(sem['endpoints'][e]['locked_test_semantics_attested'] for _,e,_,_ in a.HYPOTHESES),
        'Alle sechs Release-Labelregeln auf vollständiger Testreferenzquelle reproduziert; sämtliche Felder und native Indizes exakt identisch. Keine Umbenennung des Entwicklungsattests.',parts,
        [evidence(a.OUT/'label_semantics.json','labels_build'),evidence(a.OUT/'label_regeneration.sql','_long_label_query'),evidence(a.OUT/'index_regeneration.sql','_index_cohort_query')],
        'Kein weiterer Labelvergleich nötig; natives Referenzindexrezept allein garantiert keine technische B-Indexpolitik.')
    gate('units_time_bound',False,'Echo-IDs/-Prozenteinheit/-Zeitdelta und ICD-Encountervertrag gebunden. Stundenendpunkte haben kein bestätigtes Geräte/EHR-Uhrzeitgate; keine nominale Ersatzanalyse.',hours,
        [evidence(a.OUT/'preferred_lvef_attestation.json','echo_time_delta'),evidence(a.OUT/'index_coverage.json','reference_candidates_clock_eligible'),
         evidence(a.RELEASE/'mimic/labels.py','459-488 / 938-955 / 1764-1766')],
        'Stundenendpunkte im v4-Lauf als nicht schätzbar berichten; zeilenspezifische Assay/URL/Zensierungsprovenienz für eventuellen gesonderten Vertrag vervollständigen.')
    gate('release_input_bound',False,'Feature-Implementierungshash entspricht Release, Protokoll und alle Fragmenthashes passen; vollständiger B-Eingabevertrag fehlt: exakte qtc_fridericia_ms überall, Primärmarker/B4/ST-mm und teilweise technische Indexabdeckung.',parts,
        [evidence(a.OUT/'provenance.json','provenance'),evidence(a.OUT/'feature_inventory.json','features_build'),
         evidence(a.OUT/'feature_parts_hashes.json','features_build'),evidence(a.OUT/'index_coverage.json','coverage'),evidence(a.OUT/'projections.json','projection_build')],
        'Separater Auftrag: fehlende exakte B-Marker/Kovariaten mit unverändertem Release extrahieren und unextrahierte notwendige Indexkandidaten technisch prüfen; keine Aliase/Ersatzindizes.')
    gate('measurement_gates_bound',False,'Claimbezogene P01a/b/c-/Wiederholbarkeits-/Delineations-/Geräte-/Transform-Messgateakte fehlt. Hashgleichheit und ursprüngliche 500 Hz sind kein vollständiges Messtor.',parts,
        [evidence(a.OUT/'provenance.json','provenance'),evidence(a.OUT/'feature_inventory.json','sampling_rate_unconditional'),
         evidence(a.PROJECT/'werkzeuge/b_test/adapter.py','_require_measurement / estimate')],
        'K benötigt vollständige claimbezogene Messgateakte je exaktem Marker und Modellbasis; B4 zusätzlich Spannungs-/Achsenvertrag. Signalvarianten und EchoNext bleiben ausgeschlossen.')
    old['source_release_equivalence']={'label':True,'available_feature_ids':provenance['feature_implementation_hash_release']==provenance['feature_implementation_hash_holdout'],
       'historical_full_runtime_tree_verified':False,'runtime_access':'Dateizugriff auf Laufzeitsnapshot verweigert; nicht umgangen',
       'evidence_scope':'Scoped Feature-Implementierungshash umfasst core tree + sechs Messabhängigkeiten; vollständige Eingabedeckung separat false.'}
    for bef,ep,marker,comp in a.HYPOTHESES:
        entry=old['hypotheses'][bef]
        entry['locked_test_semantics_attested']=sem['endpoints'][ep]['locked_test_semantics_attested']
        entry['missing_required_B3_ids']=sorted(k for k in set(a.B3+[marker,comp]) if k in inventory['absent_columns'])
        entry['missing_required_B4_ids']=[k for k in a.B4 if k in inventory['absent_columns']]
        entry['units_time_attested']=bef not in hours
        entry['index_coverage']=cov['hypotheses'][bef]
        entry['affected_analyses']='B3 alle; B4 alle; H1-Fréchet/RV-Fréchet zusätzlich Primärmarker; 6a Blockflags fehlen; 6b Sinusselektor fehlt; IPW technisches Universum nicht vollständig attestiert.'
    old['independent_stops']={'signal_variants':'nicht schätzbar (Skalierung nicht gebunden), v4 Nachtrag 2(b)',
        'EchoNext':'ausgeschlossen, v4 Nachtrag 2(c)','IPW':'61 ungeklärte Personen im ersten technischen Erwachsenenuniversum; B4-Basis unvollständig',
        'marker_substitution':'Keine Aliase/Polungsänderungen: classical_qtc_fridericia und classical_st_elevation_contiguous werden nicht an Stelle gebundener IDs verwendet.'}
    from .vorbereitung_aw136 import endpoint_stops
    old['stopped_parts']=endpoint_stops(old,package)
    old['ready_for_association_run']=False;old['signal_adapter_attested']=False
    old['check_script_sha256']=a.sha(__file__)
    sources={}
    for name in ('sources_labels.json','sources_features.json','sources_projection.json'):
        for item in a.read_json(a.OUT/name):sources[item['path']]=item
    sources.update(a.SOURCES)
    # Alle kleinen/gebundenen Quellen nachprüfen; Features bereits vor Projektion fragmentweise verifiziert.
    integrity=[]
    for item in sources.values():
        p=Path(item['path'])
        if p.parent==a.HOLD/'features':continue
        integrity.append({'file':str(p),'sha256':item['sha256'],'stable':a.sha(p)==item['sha256']})
    if not all(x['stable'] for x in integrity):raise ValueError('Quelle während Bearbeitung geändert')
    a.save('source_stability.json',integrity);old['source_candidates']=list(sources.values())
    a.save('input_manifest_v2.json',old)
    draft=a.read_json(Path(str(data_path("b.aw136_freigabe_k_entwurf_json"))))
    draft.update(status='ENTWURF AW137; alle B-Teile gestoppt, keine Freigabe',
       input_manifest_path=str(a.OUT/'input_manifest_v2.json'),input_manifest_sha256=a.sha(a.OUT/'input_manifest_v2.json'),
       utc_blindheitsattestation=None,not_estimable_parts=list(old['stopped_parts']),
       access_register_requirement='AW137 erlaubt technische Eingabenzugriffe. K muss einen künftigen Assoziationsstart mit finalem Manifesthash und eigenem UTC-Blindheitsattest gesondert binden.')
    draft.pop('freigegeben_von',None);a.save('FREIGABE_K_entwurf.json',draft)
    print(json.dumps({'gates':{k:old[k] for k in old['attestations']},'stopped_parts':list(old['stopped_parts'])}),flush=True)

if __name__=='__main__':
    from werkzeuge.nachrechnung.resources import guard,Budget
    a.OUT.mkdir(parents=True,exist_ok=True);start=guard()
    snapshot=a.OUT/'executed_source'/a.sha(__file__)/Path(__file__).name
    snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    with Budget(a.OUT) as budget:build()
    a.save('resources_attestation.json',{'start':start,**budget.summary(),'script_sha256':a.sha(__file__)})
