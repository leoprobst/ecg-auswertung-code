"""Measurement projection of the release worker without unneeded measurement-capsule outputs.

Loader, filter, fusion, median morphology, Kors, registry and ClassicalECGCriteria
are called directly from the bound release. No marker formula of its own.
Technical index states are attested separately by the complete release worker.
"""
import hashlib
import numpy as np
from werkzeuge.varianten.adapter import prepare_record, release_import

def measure(row,options):
    fr=release_import()
    try:
        package=prepare_record(row,options)
        record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],'kors').xyz
        robust=fr.VCGTransformRegistry.transform(package['robust'],'kors').xyz
        result={'subject_id':int(row['subject_id']),'study_id':int(row['study_id']),
          'split':row['split'],'ecg_time':row['ecg_time'],'extraction_status':'ok',
          'original_sampling_rate':package['loaded'].original_sampling_rate,
          'signal_sha256':hashlib.sha256(np.ascontiguousarray(package['loaded'].signal_millivolts,dtype=np.float64).tobytes()).hexdigest()}
        values=fr.BiomarkerRegistry.compute_all(delineation=package['delineation'],ecg_12lead=package['robust'],
          vcg_xyz=robust,fs=package['fs'],names=options['vcg_feature_ids'],
          record_ecg_12lead=package['paths']['morphology'],record_vcg_xyz=record,
          record_raw_ecg_12lead=package['loaded'].signal_millivolts,record_acquisition=package['acquisition'])
        failures=[];metadata={}
        fr._store_biomarker_results(result,values,failures,metadata=metadata)
        report=fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''))
        paced=report['pacemaker'].state==fr.ReportMentionState.POSITIVE
        block=report['lbbb'].state==fr.ReportMentionState.POSITIVE
        classical=fr.ClassicalECGCriteria().compute(package['paths']['diagnostic'],package['delineation'],
          age=fr._optional_int(row.get('age')),sex=row.get('sex'),
          machine_qrs_axis_deg=fr._optional_float(row.get('machine_qrs_axis_deg')),
          machine_t_axis_deg=fr._optional_float(row.get('machine_t_axis_deg')),paced_or_lbbb=paced or block,paced=paced)
        reasons={}
        for k in options['classical_feature_ids']:
            c=classical.get(k.removeprefix('classical_'))
            result[k]=c.value if c is not None and c.evaluable else np.nan
            result[k+'_positive']=c.positive if c is not None and c.evaluable else None
            if c is None or not c.evaluable:reasons[k]='criterion_missing' if c is None else str(c)
        fr._finalize_biomarker_failure_provenance(result,failures)
        fr._store_feature_metadata(result,metadata)
        import json
        result['classical_unevaluable_reason_json']=json.dumps(reasons,sort_keys=True)
        return result
    except Exception:
        # Identical error/QC status instead of reinterpreting abstentions.
        return fr._extract_record_worker((row,options))
