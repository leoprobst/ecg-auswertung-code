"""LUDB arm: 200 recordings, delineation with Kors, inverse Dower and without VCG; expert boundaries as in the preceding analysis stage.

Procedure per recording as in the earlier beat-boundary run (prepare_signal_paths at 50 Hz, detect_r_peaks_multilead, FusionDelineator),
only vcg_xyz is swapped. Results (beats per variant, expert clusters) go to a jsonl file in the output folder (public recording numbers).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
import os, sys, json, time
from pathlib import Path
from . import kern
from .kern import OUT, np

DATA = Path(str(data_path("a.lobachevsky_university_electrocardiography_database_1_0_1")))
LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
PRE = OUT / 'praereg'
PRIV = OUT / 'privat'
VARIANTS = ('kors', 'inverse_dower', 'keine_vcg')


def clusters(items, tolerance):
    """As in the earlier expert-cluster definition: anchored at the first peak, each lead at most once."""
    groups = []
    for item in sorted(items, key=lambda v: (v['peak'], v['lead'])):
        if groups and item['peak'] - groups[-1][0]['peak'] <= tolerance and item['lead'] not in {x['lead'] for x in groups[-1]}:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def expert(record):
    import wfdb
    waves = {k: [] for k in ('p', 'N', 't')}
    for lead in LEADS:
        ann = wfdb.rdann(str(DATA / 'data' / record), lead)
        sym = list(ann.symbol); smp = list(ann.sample)
        for i, kind in enumerate(sym):
            if kind in waves and 0 < i < len(sym) - 1 and sym[i - 1] == '(' and sym[i + 1] == ')' and smp[i - 1] < smp[i] < smp[i + 1]:
                waves[kind].append(dict(lead=lead, onset=int(smp[i - 1]), peak=int(smp[i]), end=int(smp[i + 1])))
    out = {}
    for kind in waves:
        groups = [g for g in clusters(waves[kind], 40) if len(g) >= 3]
        out[kind] = [dict(onset=min(v['onset'] for v in g), peak=float(np.median([v['peak'] for v in g])), end=max(v['end'] for v in g),
                          onset_med=float(np.median([v['onset'] for v in g])), end_med=float(np.median([v['end'] for v in g])), n_leads=len(g)) for g in groups]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='ludb_ergebnisse.jsonl')
    args = ap.parse_args()
    import wfdb, psutil
    manifest = json.loads((PRE / 'praereg_manifest.json').read_text(encoding='utf8'))
    assert kern.sha_file(PRE / 'liste_ludb.json') == manifest['listen_sha256']['liste_ludb.json']
    assert kern.sha_file(PRE / 'f_a2_kriterium.json') == manifest['kriterium_sha256']
    records = json.loads((PRE / 'liste_ludb.json').read_text())
    if args.limit:
        records = records[:args.limit]
    sums = {}
    for line in (DATA / 'SHA256SUMS.txt').read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip('*')] = h
    PRIV.mkdir(parents=True, exist_ok=True)
    target = PRIV / args.out
    done = set()
    if target.exists():
        for line in target.read_text(encoding='utf8').splitlines():
            if line.strip():
                done.add(json.loads(line)['record'])
    fr = kern.release()
    from vcg_biomarker.core.preprocessing import prepare_signal_paths
    proc = psutil.Process(); peak = 0; start = time.monotonic(); n_new = 0
    for i, record in enumerate(records, 1):
        if record in done:
            continue
        kern.memory_wait(1.5, 'aw196_ludb')
        tic = time.monotonic()
        rec = {'record': record}
        try:
            for ext in ['hea', 'dat', *LEADS]:
                p = DATA / 'data' / f'{record}.{ext}'
                if sums['data/' + p.name] != kern.sha_file(p):
                    raise RuntimeError('LUDB file differs from the checksum')
            sig = wfdb.rdrecord(str(DATA / 'data' / record)); fs = int(sig.fs)
            if fs != 500:
                raise RuntimeError('fs != 500')
            names = [n.lower() for n in sig.sig_name]
            signal = np.asarray(sig.p_signal, float)[:, [names.index(n) for n in LEADS]]
            rec['expert'] = expert(record)
            paths = prepare_signal_paths(signal, fs, powerline=50.)
            ecg = paths['delineation']
            peaks = fr.detect_r_peaks_multilead(ecg, fs)
            xyz = {'kors': fr.VCGTransformRegistry.transform(ecg, 'kors').xyz,
                   'inverse_dower': fr.VCGTransformRegistry.transform(ecg, 'inverse_dower').xyz,
                   'keine_vcg': None}
            acquisition = {'original_sampling_rate_hz': float(fs)}
            rec['fs'] = fs; rec['n_rpeaks'] = int(len(peaks)); rec['status'] = 'ok'
            for v in VARIANTS:
                d = kern.delineate(fr, ecg, peaks, fs, int(record), xyz[v])
                rec[v] = kern.fiducials_and_markers(fr, d, paths, signal, acquisition, fs, apply_report=False)
        except Exception as exc:
            rec['status'] = f'{type(exc).__name__}: {exc}'[:160]
        rec['seconds'] = time.monotonic() - tic
        with open(target, 'a', encoding='utf8') as f:
            f.write(json.dumps(rec) + '\n')
        n_new += 1; peak = max(peak, proc.memory_info().rss)
        if n_new % 5 == 0 or args.limit:
            print(json.dumps({'fortschritt': i, 'von': len(records), 'neu': n_new, 'sek_je_aufnahme': round((time.monotonic() - start) / n_new, 2),
                              'peak_mb': round(peak / 2 ** 20), 'pid': os.getpid()}), flush=True)
    (PRIV / (args.out + '.fertig')).write_text(json.dumps({'n': len(records), 'peak_rss_mb': round(peak / 2 ** 20), 'sekunden': time.monotonic() - start}), encoding='utf8')


if __name__ == '__main__':
    main()
