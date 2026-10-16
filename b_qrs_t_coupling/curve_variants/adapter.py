"""Versionierter, nur lesender A155-Adapter. Noch kein realer Äquivalenznachweis."""
import os,sys,copy,hashlib
from pathlib import Path
import numpy as np
from .audit import ROOT,memory_guard
from .metriken import form_distance,repaired_qrs

ORIGINALS=('qrs_t_normalized_discrete_frechet','qrs_t_loop_mirror_symmetry','loop_near_self_approach_count','loop_graph_cycle_count_proxy')
COMPARATORS=('spatial_qrs_t_angle_deg','frontal_xy_qrs_t_angle_deg')
OPTIONS={'vcg_feature_ids':list(ORIGINALS)+list(COMPARATORS),'sensitivity_feature_ids':list(ORIGINALS),'classical_feature_ids':[], 'sensitivity_vcg_methods':('inverse_dower',),'powerline_hz':60.,'delineation_weights':{'neurokit':.4,'multilead':.35,'vcg':.25},'delineation_conflict_threshold_ms':20.}

def release_import():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT))
    import vcg_biomarker.mimic.feature_record as fr
    for name,module in tuple(sys.modules.items()):
        if name.startswith('vcg_biomarker') and getattr(module,'__file__',None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('Falscher Importstand')
    return fr

def prepare_record(row,options=OPTIONS):
    memory_guard();fr=release_import()
    loaded=fr.MimicWaveformLoader(target_sampling_rate=500).load(row['waveform_path'],expected_subject_id=int(row['subject_id']),expected_study_id=int(row['study_id']))
    if not loaded.quality.usable or loaded.quality.limb_lead_reversal_suspected:raise ValueError('Original-QC-Abstention')
    paths=fr.MimicWaveformLoader(target_sampling_rate=500).prepare_paths(loaded,powerline_hz=float(options['powerline_hz']),device_filtering=row.get('machine_filtering'))
    peaks=fr.detect_r_peaks_multilead(paths['delineation'],loaded.sampling_rate)
    if len(peaks)<2:raise ValueError('Original-R-Peak-Gate')
    xyz=fr.VCGTransformRegistry.transform(paths['delineation'],'kors').xyz
    delineation=fr.FusionDelineator(weights=options['delineation_weights'],conflict_threshold=options['delineation_conflict_threshold_ms']).delineate_record(ecg_12lead=paths['delineation'],r_peaks=peaks,fs=loaded.sampling_rate,ecg_id=int(row['study_id']),vcg_xyz=xyz)
    if not delineation.beats:raise ValueError('Original-Delineationsgate')
    report=fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''))
    if report['af'].state==fr.ReportMentionState.POSITIVE:
        delineation.suppress_discrete_p('discrete_p_suppressed_by_report_atrial_fibrillation_flutter')
    robust,diagnostics=fr.construct_robust_morphology_record(paths['morphology'],delineation)
    supported,upper,reason=fr._wideband_acquisition_support(row.get('machine_bandwidth'))
    acquisition={'original_sampling_rate_hz':loaded.original_sampling_rate,'upper_bandwidth_hz':upper,'pacing_present':True if report['pacemaker'].state==fr.ReportMentionState.POSITIVE else None,'pacing_source':'machine_report_positive_only_not_independent_adjudication'}
    return {'loaded':loaded,'paths':paths,'robust':robust,'delineation':delineation,'fs':loaded.sampling_rate,'acquisition':acquisition,'diagnostics':diagnostics,'wideband_supported':supported}

def context(package,method='kors',delineation=None):
    fr=release_import()
    from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext,reset_feature_context,set_feature_context_record_sources
    reset_feature_context()
    record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],method).xyz
    robust=fr.VCGTransformRegistry.transform(package['robust'],method).xyz
    set_feature_context_record_sources(package['paths']['morphology'],record,None,package['loaded'].signal_millivolts,package['acquisition'])
    try:
        return FeatureContext(delineation or package['delineation'],package['robust'],robust,package['fs'])
    finally:reset_feature_context()

def original_record(row):
    return release_import()._extract_record_worker((row,OPTIONS))

def prepared_originals(package):
    """Originalregistry auf exakt den vom lokalen Adapter vorbereiteten Quellen."""
    fr=release_import()
    record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],'kors').xyz
    robust=fr.VCGTransformRegistry.transform(package['robust'],'kors').xyz
    results=fr.BiomarkerRegistry.compute_all(delineation=package['delineation'],ecg_12lead=package['robust'],vcg_xyz=robust,fs=package['fs'],names=list(ORIGINALS),record_ecg_12lead=package['paths']['morphology'],record_vcg_xyz=record,record_raw_ecg_12lead=package['loaded'].signal_millivolts,record_acquisition=package['acquisition'])
    destination={};metadata={};failures=[]
    fr._store_biomarker_results(destination,results,calculation_failures=failures,metadata=metadata)
    fr._store_feature_metadata(destination,metadata)
    if failures:raise RuntimeError('Originalregistry-Rechenfehler')
    return destination

def equivalent(actual,expected):
    """Missingness exakt; rtol 1e-8, kein verdecktes absolutes Toleranzband."""
    from numbers import Real
    for marker in ORIGINALS:
        a=actual.get(marker);b=expected.get(marker)
        missing_a=a is None or (isinstance(a,Real) and not np.isfinite(a))
        missing_b=b is None or (isinstance(b,Real) and not np.isfinite(b))
        if missing_a!=missing_b or (not missing_a and not np.isclose(a,b,rtol=1e-8,atol=0)):
            raise RuntimeError('Originalabweichung: '+marker)
    import json
    for key in ('feature_confidence_json','feature_unevaluable_reason_json'):
        a=json.loads(actual.get(key) or '{}');b=json.loads(expected.get(key) or '{}')
        for marker in ORIGINALS:
            if a.get(marker)!=b.get(marker):raise RuntimeError('Metadatenabweichung: '+marker)
    return True

def variants(package):
    from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import _qrs_t_distances
    from vcg_biomarker.core.biomarkers.qrs_t_coupling_extension_biomarkers import _mirror_symmetry_direct_8lead,_frechet_beat_median,_frechet_beat_count,_beat_sample
    out={};qc={}
    def put(family,i,value,reason=''):
        out[f'VB-VF{family:02d}-{i:02d}']={'value':value,'abstention':reason if value is None else ''}
    ctx=context(package)
    bundle=_qrs_t_distances(context(package,'inverse_dower'))
    put(1,1,None if bundle is None else bundle['normalized_discrete_frechet'],'Originalbündel nicht messbar')
    mirror=_mirror_symmetry_direct_8lead(context(package));put(1,2,mirror.value,mirror.notes)
    beat=_frechet_beat_median(context(package));put(1,3,beat.value,beat.notes)
    qc['frechet_beat_count']=_frechet_beat_count(context(package)).value
    q=ctx.vcg_segment('qrs');t=ctx.vcg_segment('t')
    original_gate=_qrs_t_distances(context(package)) is not None
    for i,mode in ((4,'time'),(5,'shared'),(7,'rotation')):
        put(1,i,form_distance(q,t,mode) if original_gate else None,'Originalgate/degenerierte Kurve')
    distances=[]
    for attr,delta in ((None,0),('r_offset',-10),('r_offset',10),('t_offset',-20),('t_offset',20)):
        delineation=copy.deepcopy(package['delineation'])
        if attr:
            beat=delineation.representative_beat
            point=getattr(beat,attr,None)
            if point is None:continue
            point.sample+=round(delta*package['fs']/1000)
        c=context(package,delineation=delineation)
        bounds=[c.bounds(s) for s in ('qrs','t')]
        if any(b is None for b in bounds) or not (0<=bounds[0][0]<bounds[0][1]<=bounds[1][0]<bounds[1][1]<c.n_samples):continue
        b=_qrs_t_distances(c)
        if b is not None:distances.append(b['normalized_discrete_frechet'])
    put(1,6,float(np.median(distances)) if len(distances)>=3 else None,'Weniger als drei gültige Grenzen')
    qc['boundary_valid']=len(distances);qc['boundary_range']=float(np.ptp(distances)) if distances else None
    put(1,8,None,'Train-Fit/OOF erforderlich; im Rauchtest nicht schätzbar')
    repaired=repaired_qrs(q) if q is not None else None
    for i,key in ((1,'near'),(2,'graph_p10'),(3,'graph_fixed'),(4,'events')):
        put(2,i,repaired[key] if repaired else None,'QRS nicht messbar/degeneriert')
    qc['qrs']=repaired
    values=[]
    for beat in package['delineation'].beats:
        if getattr(beat,'morphology_cluster_id',None)!=0:continue
        a=_beat_sample(beat,'r_onset','qrs');b=_beat_sample(beat,'r_offset','qrs')
        if a is None or b is None or not 0<=a<b<len(ctx._record_vcg_xyz) or b-a<8:continue
        r=repaired_qrs(ctx._record_vcg_xyz[a:b+1])
        if r:values.append(r['near'])
    put(2,5,float(np.median(values)) if len(values)>=3 else None,'Weniger als drei QRS-Schläge')
    qc['near_beat_count']=len(values)
    trim=round(.010*package['fs']);core=q[trim:-trim] if q is not None and len(q)>2*trim else None
    r=repaired_qrs(core) if core is not None else None
    put(2,6,r['near'] if r else None,'Gekürztes QRS nicht messbar')
    c=context(package,'inverse_dower');segment=c.vcg_segment('qrs');r=repaired_qrs(segment) if segment is not None else None
    put(2,7,r['graph_fixed'] if r else None,'Dower-QRS nicht messbar')
    put(2,8,None,'Train-Fit/OOF erforderlich; im Rauchtest nicht schätzbar')
    return dict(sorted(out.items())),qc
