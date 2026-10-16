"""PTB arm: per person the lexicographically first recording with Frank leads; delineation with Kors, inverse Dower,
without VCG and with the measured Frank VCG as input.

Phase corr: sign of the Frank axes from the signals (median Pearson correlation of Frank against Kors per axis), no delineation.
Phase delineate: the four variants (plus the raw Frank sign if one axis is negative).
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

DATA = Path(str(data_path("a.research_expansion_20261012_ptbdb_1_0_0")))
PRE = OUT / 'praereg'
PRIV = OUT / 'privat'
NAMES = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'vx', 'vy', 'vz')


def load(record):
    import wfdb
    from vcg_biomarker.core.preprocessing import resample_signal, prepare_signal_paths
    from vcg_biomarker.mimic.signals import _convert_to_millivolts
    raw_rec = wfdb.rdrecord(str(DATA / record))
    names = [n.casefold() for n in raw_rec.sig_name]
    inds = [names.index(n.casefold()) for n in NAMES]
    raw = _convert_to_millivolts(raw_rec.p_signal[:, inds], tuple(raw_rec.units[j] or '' for j in inds))
    raw = resample_signal(raw, float(raw_rec.fs), 500)
    paths = prepare_signal_paths(raw, 500, powerline=50.)
    return raw, paths, float(raw_rec.fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['corr', 'delineate'], required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--shard', default=None, help='k/n: only recordings with index mod n == k')
    args = ap.parse_args()
    import psutil
    manifest = json.loads((PRE / 'praereg_manifest.json').read_text(encoding='utf8'))
    assert kern.sha_file(PRE / 'liste_ptb_privat.json') == manifest['listen_sha256']['liste_ptb_privat.json']
    assert kern.sha_file(PRE / 'f_a2_kriterium.json') == manifest['kriterium_sha256']
    records = json.loads((PRE / 'liste_ptb_privat.json').read_text())
    if args.limit:
        records = records[:args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split('/'))
        records = [r for j, r in enumerate(records) if j % n == k]
    PRIV.mkdir(parents=True, exist_ok=True)
    out_name = args.out or ('ptb_korrelation.jsonl' if args.phase == 'corr' else 'ptb_ergebnisse.jsonl')
    target = PRIV / out_name
    done = set()
    if target.exists():
        for line in target.read_text(encoding='utf8').splitlines():
            if line.strip():
                done.add(json.loads(line)['record'])
    fr = kern.release()
    signs = None
    raw_sign_needed = False
    if args.phase == 'delineate':
        corr = [json.loads(l) for l in (PRIV / 'ptb_korrelation.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
        corr = [c for c in corr if c.get('status') == 'ok']
        med = [float(np.median([c['r'][k] for c in corr])) for k in range(3)]
        signs = np.array([1. if m >= 0 else -1. for m in med])
        raw_sign_needed = bool((signs < 0).any())
        (PRIV / 'ptb_achsenregel.json').write_text(json.dumps({'median_korrelation_xyz': med, 'vorzeichen_xyz': signs.tolist(), 'n': len(corr), 'rohes_vorzeichen_variante': raw_sign_needed}), encoding='utf8')
        print(json.dumps({'achsenregel': med, 'vorzeichen': signs.tolist()}), flush=True)
    proc = psutil.Process(); peak = 0; start = time.monotonic(); n_new = 0
    for i, record in enumerate(records, 1):
        if record in done:
            continue
        kern.memory_wait(1.5, 'aw196_ptb')
        tic = time.monotonic()
        rec = {'record': record}
        try:
            raw, paths, fs0 = load(record)
            ecg_all = paths['delineation']
            ecg = ecg_all[:, :12]
            fs = 500
            kors = fr.VCGTransformRegistry.transform(ecg, 'kors').xyz
            frank = ecg_all[:, 12:15]
            if args.phase == 'corr':
                rec['r'] = [float(np.corrcoef(frank[:, k], kors[:, k])[0, 1]) for k in range(3)]
                rec['status'] = 'ok'
            else:
                peaks = fr.detect_r_peaks_multilead(ecg, fs)
                if len(peaks) < 2:
                    raise ValueError('R-Zacken <2')
                inputs = {'kors': kors, 'inverse_dower': fr.VCGTransformRegistry.transform(ecg, 'inverse_dower').xyz,
                          'keine_vcg': None, 'frank_gemessen': frank * signs[None, :]}
                if raw_sign_needed:
                    inputs['frank_roh'] = frank
                paths12 = {k: (x[:, :12] if getattr(x, 'ndim', 0) == 2 and x.shape[1] == 15 else x) for k, x in paths.items()}
                acquisition = {'original_sampling_rate_hz': fs0}
                rec['n_rpeaks'] = int(len(peaks)); rec['dauer_s'] = round(len(ecg) / fs, 1); rec['status'] = 'ok'
                for v, x in inputs.items():
                    d = kern.delineate(fr, ecg, peaks, fs, i, x)
                    rec[v] = kern.fiducials_and_markers(fr, d, paths12, raw[:, :12], acquisition, fs, apply_report=False)
        except Exception as exc:
            rec['status'] = f'{type(exc).__name__}: {exc}'[:160]
        rec['seconds'] = time.monotonic() - tic
        with open(target, 'a', encoding='utf8') as f:
            f.write(json.dumps(rec) + '\n')
        n_new += 1; peak = max(peak, proc.memory_info().rss)
        if n_new % 5 == 0 or args.limit:
            print(json.dumps({'phase': args.phase, 'fortschritt': i, 'von': len(records), 'neu': n_new, 'sek_je_aufnahme': round((time.monotonic() - start) / n_new, 2),
                              'peak_mb': round(peak / 2 ** 20), 'pid': os.getpid()}), flush=True)
    (PRIV / (out_name + '.fertig')).write_text(json.dumps({'n': len(records), 'peak_rss_mb': round(peak / 2 ** 20), 'sekunden': time.monotonic() - start}), encoding='utf8')


if __name__ == '__main__':
    main()
