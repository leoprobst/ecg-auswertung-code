"""Nur aggregierte Diagnose der Referenz- und Verfügbarkeitsgates."""
import json,re
from .runner import small,save
from .clinical import reference

def run(out):
    f=small(out/'reduced_lvef_strict_private.parquet');e=small(out/'echo_evidence_private.parquet');f=reference(f,e,'reduced_lvef_strict');s='reduced_lvef_strict__'
    report={'index_strategy':f.index_selection_strategy.value_counts(dropna=False).to_dict(),'states':f[s+'label_state'].value_counts(dropna=False).to_dict(),
      'eligibility':f[s+'eligibility_state'].value_counts(dropna=False).to_dict(),'precision':f[s+'time_precision'].value_counts(dropna=False).to_dict(),
      'conflict':f.lvef_conflict_gt_10.value_counts(dropna=False).to_dict(),'tte_id_types':f.nearest_tte_measurement_id.dropna().map(lambda x:type(x).__name__).value_counts().to_dict(),
      'alignment':f[s+'ehr_time_alignment_confirmatory_eligible'].value_counts(dropna=False).to_dict()}
    z=f[f[s+'label_state'].isin(['positive','negative'])]
    report['evidence_id_shapes']=[re.sub(r'\d','N',str(x)) for x in z[s+'evidence_ids_json'].dropna().head(3)]
    report['tte_id_shapes']=[re.sub(r'\d','N',str(x)) for x in z.nearest_tte_measurement_id.dropna().head(3)]
    p=small(out/'preindex.parquet',columns=['has_tte_30d','has_tte_7d','nearest_tte_measurement_id','nearest_tte_delta_hours'])
    report['null_flag']={'n':int(p.has_tte_30d.isna().sum()),'with_tte_id':int((p.has_tte_30d.isna()&p.nearest_tte_measurement_id.notna()).sum()),'with_tte_time':int((p.has_tte_30d.isna()&p.nearest_tte_delta_hours.notna()).sum())}
    # Convert nullable keys to strings for JSON.
    report={k:{str(a):int(b) for a,b in v.items()} if isinstance(v,dict) else v for k,v in report.items()}
    save(out/'gate_diagnosis.json',report);print(json.dumps(report),flush=True)
