"""Delineation with a swapped vcg_xyz input; the release is read-only and no bytecode is written."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, time, hashlib, ctypes
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(str(data_path("a.release_sources_a155_survival_nullable_20261007")))
OUT = Path(str(data_path("a.aw196")))
POINTS = ('p_onset', 'p_offset', 'qrs_onset', 'qrs_offset', 't_offset')
MARKERS = ('qrs_duration_ms', 'qt_interval_ms', 'qtc_fridericia_ms', 'pr_interval_ms', 'p_duration_ms')
WEIGHTS = {'neurokit': .4, 'multilead': .35, 'vcg': .25}
CONFLICT = 20.


class _Mem(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ('totalPhys', 'availPhys', 'totalPage', 'availPage', 'totalVirtual', 'availVirtual', 'availExtended')]


def memory_wait(min_gb=1.5, label=''):
    """Small job: waits while less than min_gb of memory is available."""
    while True:
        m = _Mem(); m.length = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        free = min(m.availPhys, m.availPage) / 1024 ** 3
        if free >= min_gb:
            return free
        print(json.dumps({'warten': label, 'frei_gb': round(free, 2)}), flush=True)
        time.sleep(30)


def release():
    sys.path.insert(0, str(ROOT))
    import vcg_biomarker.mimic.feature_record as fr
    for name, module in tuple(sys.modules.items()):
        if name.startswith('vcg_biomarker') and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError('Wrong import state')
    return fr


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def delineate(fr, ecg_12lead, peaks, fs, ecg_id, xyz):
    """Production call; only xyz (None, Kors, Dower, Frank) varies."""
    return fr.FusionDelineator(weights=WEIGHTS, conflict_threshold=CONFLICT).delineate_record(
        ecg_12lead=ecg_12lead, r_peaks=peaks, fs=fs, ecg_id=ecg_id, vcg_xyz=xyz)


def fiducials_and_markers(fr, delineation, paths, loaded_signal_mv, acquisition, fs, row_report=None, apply_report=True):
    """Fiducial points (samples) and native markers from the representative consensus beat."""
    from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext, reset_feature_context, set_feature_context_record_sources
    if not delineation.beats:
        return {'status': 'no_beats', 'points': {k: None for k in POINTS}, 'markers': {k: None for k in MARKERS}, 'n_beats': 0}
    if apply_report and row_report is not None:
        report = fr._REPORT_PARSER.parse(str(row_report or ''))
        if report['af'].state == fr.ReportMentionState.POSITIVE:
            delineation.suppress_discrete_p('discrete_p_suppressed_by_report_atrial_fibrillation_flutter')
    robust, diagnostics = fr.construct_robust_morphology_record(paths['morphology'], delineation)
    record_xyz = fr.VCGTransformRegistry.transform(paths['morphology'], 'kors').xyz
    robust_xyz = fr.VCGTransformRegistry.transform(robust, 'kors').xyz
    reset_feature_context()
    set_feature_context_record_sources(paths['morphology'], record_xyz, None, loaded_signal_mv, acquisition)
    try:
        ctx = FeatureContext(delineation, robust, robust_xyz, fs)
        pts = {
            'p_onset': ctx._sample('p_onset'), 'p_offset': ctx._sample('p_offset'),
            'qrs_onset': ctx._qrs_start(), 'qrs_offset': ctx._qrs_end(), 't_offset': ctx._sample('t_offset'),
        }
    finally:
        reset_feature_context()
    pts = {k: (None if v is None else int(v)) for k, v in pts.items()}
    available = set(fr.BiomarkerRegistry.available())
    names = [n for n in MARKERS if n in available]
    results = fr.BiomarkerRegistry.compute_all(
        delineation=delineation, ecg_12lead=robust, vcg_xyz=robust_xyz, fs=fs, names=names,
        record_ecg_12lead=paths['morphology'], record_vcg_xyz=record_xyz, record_vcg_wideband_xyz=None,
        record_raw_ecg_12lead=loaded_signal_mv, record_acquisition=acquisition)
    dest = {}
    fr._store_biomarker_results(dest, results, [], metadata={})
    markers = {}
    for n in MARKERS:
        v = dest.get(n)
        markers[n] = None if v is None or not np.isfinite(float(v)) else float(v)
    rep = delineation.representative_beat
    beats = []
    for b in delineation.beats:
        def g(a):
            m = getattr(b, a, None)
            return -1 if m is None else int(m.sample)
        beats.append([int(b.r_peak.sample), g('p_onset'), g('p_offset'), g('r_onset'), g('r_offset'), g('t_offset'),
                      1 if b.resolved_wave_status('p').value == 'present' else 0])
    return {'status': 'ok', 'points': pts, 'markers': markers, 'n_beats': len(delineation.beats),
            'rep_r': None if rep is None else int(rep.r_peak.sample), 'beats': beats,
            'robust_status': diagnostics.get('status') if isinstance(diagnostics, dict) else None}
