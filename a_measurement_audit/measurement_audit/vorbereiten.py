import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from common import *
master=Path(str(data_path("a.mimic_validation_master_ecg_master_index_parquet")))
wide=Path(str(data_path("a.mimic_validation_labels_clinical_labels_wide_parquet")))
attp=wide.with_name('development_label_attestation.json');att=json.loads(attp.read_text())
assert att['status']=='verified' and att['scope']=='train_validation_only'
assert sha(wide)==att['development_artifacts']['clinical_labels_wide']['sha256']
footer(master);footer(wide)
c=conn()
c.execute(f"CREATE TABLE dev AS SELECT subject_id,study_id,split FROM read_parquet('{sqlp(wide)}')")
assert set(x[0] for x in c.execute('SELECT DISTINCT split FROM dev').fetchall())<={'train','validation'}
cols=['waveform_path','machine_filtering','machine_bandwidth','machine_report','machine_metadata_usable_for_primary','machine_cart_id','ecg_time','age','sex','hadm_id','admission_match_ambiguous','machine_qrs_duration_ms']
q=f"SELECT m.subject_id,m.study_id,m.split,{','.join('m.'+x for x in cols)} FROM read_parquet('{sqlp(master)}') m JOIN dev d USING(subject_id,study_id,split)"
m=c.execute(q).fetchdf();m.to_parquet(OUT/'master_dev_privat.parquet',index=False)
normal="machine_metadata_usable_for_primary AND age>=18 AND regexp_matches(lower(machine_report),'normal ecg|within normal limits') AND regexp_matches(lower(machine_report),'sinus rhythm') AND NOT regexp_matches(lower(machine_report),'abnormal|infarct|hypertroph|bundle|fascic|axis deviation|atrial fibrillation|atrial flutter|paced|pacemaker')"
rq=f"WITH one AS (SELECT *,row_number() OVER(PARTITION BY subject_id ORDER BY ecg_time,study_id) r FROM read_parquet('{sqlp(OUT/'master_dev_privat.parquet')}') WHERE {normal}) SELECT * EXCLUDE(r) FROM one WHERE r=1 ORDER BY md5('AW047-R-v1:'||study_id::VARCHAR),study_id LIMIT 1500"
ref=c.execute(rq).fetchdf();ref.to_parquet(OUT/'referenz_privat.parquet',index=False)
(OUT/'auswahl.sql').write_text(q+';\n'+rq,encoding='utf-8')
features=list((RUN/'analysis_feature_cache').glob('*/split=*/features.parquet'));assert len(features)==2
for p in features:footer(p)
keys=json.loads((OUT/'befund_schluessel.json').read_text());markers=sorted(set(x['feature_id'] for x in keys))
proj=['subject_id','study_id','ecg_time','p_duration_ms','rhythm_context']+markers
for p in features:
    c.execute(f"COPY (SELECT {','.join(proj)},'{p.parent.name.split('=')[1]}' AS split FROM read_parquet('{sqlp(p)}')) TO '{sqlp(OUT/(p.parent.name+'_klein_privat.parquet'))}' (FORMAT PARQUET)")
fglob=sqlp(OUT/'split=*_klein_privat.parquet')
refp=c.execute(f"SELECT r.*,f.p_duration_ms,CASE WHEN f.p_duration_ms<120 THEN 'kurz' WHEN f.p_duration_ms<=160 THEN 'mittel' WHEN f.p_duration_ms>160 THEN 'lang' ELSE 'fehlend' END p_stratum FROM read_parquet('{sqlp(OUT/'referenz_privat.parquet')}') r JOIN read_parquet('{fglob}') f USING(subject_id,study_id,split)").fetchdf()
refp.to_parquet(OUT/'referenz_privat.parquet',index=False)
pqry=f"WITH r AS (SELECT *,row_number() OVER(PARTITION BY p_stratum ORDER BY md5('AW047-P-v1:'||study_id::VARCHAR),study_id) rn FROM read_parquet('{sqlp(OUT/'referenz_privat.parquet')}') WHERE p_stratum!='fehlend') SELECT * EXCLUDE(rn) FROM r WHERE rn<=100"
sample=c.execute(pqry).fetchdf();sample.to_parquet(OUT/'p_auswahl_privat.parquet',index=False)
pairdir=RUN/'repeatability/replicate_pairs';parts=sorted(pairdir.glob('*.parquet'))
for p in parts:footer(p)
basecols=['subject_id','split','first_study_id','second_study_id','first_ecg_time','second_ecg_time','hours_between','intervening_intervention_checked','signal_quality_first','signal_quality_second']
c.execute(f"CREATE TABLE pp AS SELECT {','.join(basecols)} FROM read_parquet('{sqlp(parts[0])}') WHERE split IN ('train','validation')")
first=','.join('a."'+x+'" AS "'+x+'_first"' for x in markers);second=','.join('b."'+x+'" AS "'+x+'_second"' for x in markers)
pq2=f"SELECT p.*,{first},{second} FROM pp p LEFT JOIN read_parquet('{fglob}') a ON p.first_study_id=a.study_id AND p.subject_id=a.subject_id AND p.split=a.split LEFT JOIN read_parquet('{fglob}') b ON p.second_study_id=b.study_id AND p.subject_id=b.subject_id AND p.split=b.split"
c.execute(f"COPY ({pq2}) TO '{sqlp(OUT/'paare_ergaenzt_privat.parquet')}' (FORMAT PARQUET)")
c.close()
jsave(OUT/'vorbereitung.json',{'n_dev':len(m),'n_referenz':len(refp),'reference_sex':refp.sex.value_counts().to_dict(),'reference_split':refp.split.value_counts().to_dict(),'p_strata':refp.p_stratum.value_counts().to_dict(),'p_auswahl':sample.p_stratum.value_counts().to_dict(),'pair_block_markers':{p.name:len([x for x in footer(p) if x.endswith('_first')])-1 for p in parts},'B_in_original_pairs':{x:[p.name for p in parts if x+'_first' in footer(p)] for x in markers},'ram':memory(),'rss_peak':peak})
bind([master,wide,attp,*features,*parts,ROOT/'werkzeuge/messaudit/vorbereiten.py',OUT/'referenz_privat.parquet',OUT/'p_auswahl_privat.parquet',OUT/'paare_ergaenzt_privat.parquet'],'vorbereitung')
print((OUT/'vorbereitung.json').read_text(),flush=True)
