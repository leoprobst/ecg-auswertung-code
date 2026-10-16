"""MIMIC arm: 500 development ECGs, delineation with Kors, inverse Dower and without VCG.

One process, sequential, checkpoint after each ECG (jsonl in the private output folder, which holds study_id only there).
Usage: python -B -m <package>.mimic_run [--limit N]  (run as a module of its package)
"""
import argparse
import os, sys, json, time, hashlib
from pathlib import Path
from . import kern
from .kern import OUT, np

PRE = OUT / 'praereg'
PRIV = OUT / 'privat'
VARIANTS = ('kors', 'inverse_dower', 'keine_vcg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='mimic_ergebnisse.jsonl')
    args = ap.parse_args()
    import psutil
    manifest = json.loads((PRE / 'praereg_manifest.json').read_text(encoding='utf8'))
    assert kern.sha_file(PRE / 'stichprobe_mimic_privat.json') == manifest['listen_sha256']['stichprobe_mimic_privat.json']
    assert kern.sha_file(PRE / 'f_a2_kriterium.json') == manifest['kriterium_sha256']
    sample = json.loads((PRE / 'stichprobe_mimic_privat.json').read_text(encoding='utf8'))
    wanted = {s['study_id']: s for s in sample}
    from werkzeuge.transform_katalog.fortsetzung import rows
    allrows = {r['study_id']: r for r in rows() if r['study_id'] in wanted}
    assert len(allrows) == 500 and all(allrows[k]['split'] == wanted[k]['split'] for k in wanted)
    order = [s['study_id'] for s in sample]
    if args.limit:
        order = order[:args.limit]
    PRIV.mkdir(parents=True, exist_ok=True)
    target = PRIV / args.out
    done = set()
    if target.exists():
        for line in target.read_text(encoding='utf8').splitlines():
            if line.strip():
                done.add(json.loads(line)['study_id'])
    fr = kern.release()
    proc = psutil.Process()
    peak = 0
    start = time.monotonic()
    n_new = 0
    for i, sid in enumerate(order, 1):
        if sid in done:
            continue
        kern.memory_wait(1.5, 'aw196_mimic')
        row = allrows[sid]
        tic = time.monotonic()
        rec = {'study_id': sid, 'split': row['split']}
        try:
            loader = fr.MimicWaveformLoader(target_sampling_rate=500)
            loaded = loader.load(row['waveform_path'], expected_subject_id=int(row['subject_id']), expected_study_id=int(row['study_id']))
            if not loaded.quality.usable or loaded.quality.limb_lead_reversal_suspected:
                raise ValueError('Original-QC-Abstention')
            paths = loader.prepare_paths(loaded, powerline_hz=60., device_filtering=row.get('machine_filtering'))
            peaks = fr.detect_r_peaks_multilead(paths['delineation'], loaded.sampling_rate)
            if len(peaks) < 2:
                raise ValueError('Original-R-Peak-Gate')
            fs = loaded.sampling_rate
            ecg = paths['delineation']
            xyz = {'kors': fr.VCGTransformRegistry.transform(ecg, 'kors').xyz,
                   'inverse_dower': fr.VCGTransformRegistry.transform(ecg, 'inverse_dower').xyz,
                   'keine_vcg': None}
            _, upper, _ = fr._wideband_acquisition_support(row.get('machine_bandwidth'))
            report = fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''))
            acquisition = {'original_sampling_rate_hz': loaded.original_sampling_rate, 'upper_bandwidth_hz': upper,
                           'pacing_present': True if report['pacemaker'].state == fr.ReportMentionState.POSITIVE else None,
                           'pacing_source': 'machine_report_positive_only_not_independent_adjudication'}
            rec['fs'] = int(fs)
            rec['n_rpeaks'] = int(len(peaks))
            rec['status'] = 'ok'
            for v in VARIANTS:
                d = kern.delineate(fr, ecg, peaks, fs, int(row['study_id']), xyz[v])
                rec[v] = kern.fiducials_and_markers(fr, d, paths, loaded.signal_millivolts, acquisition, fs, row.get('machine_report'))
                rec[v]['n_beats_total'] = len(d.beats)
        except Exception as exc:
            rec['status'] = f'{type(exc).__name__}: {exc}'[:160]
        rec['seconds'] = time.monotonic() - tic
        with open(target, 'a', encoding='utf8') as f:
            f.write(json.dumps(rec) + '\n')
        n_new += 1
        rss = proc.memory_info().rss
        peak = max(peak, rss)
        if n_new % 10 == 0 or args.limit:
            print(json.dumps({'fortschritt': i, 'von': len(order), 'neu': n_new, 'sek_je_ekg': round((time.monotonic() - start) / n_new, 2),
                              'rss_mb': round(rss / 2 ** 20), 'peak_mb': round(peak / 2 ** 20), 'pid': os.getpid()}), flush=True)
    (PRIV / (args.out + '.fertig')).write_text(json.dumps({'n': len(order), 'peak_rss_mb': round(peak / 2 ** 20), 'sekunden': time.monotonic() - start}), encoding='utf8')


if __name__ == '__main__':
    main()
