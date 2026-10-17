"""AW-255 S3 Machbarkeit (MC-MED): Signalguete und Messbarkeit an Ableitung II, nur Train, keine Assoziation.

Eingabe (privat, <local path>, nie <cloud folder>): cand.csv (CSN, t_order, t_result, k, serum_only), cand_seg.csv (Segmentkoepfe).
Ausgabe (privat): signal_guete.csv (je Entnahme), messbar.csv (je Entnahme und Fensterlaenge). Aggregation getrennt (s3_aggregat.py).

Modi (Python: <local path> -I):
  guete    [N_WORKER]   Blockweise Guete (10 s) im Fenster um jede Entnahme; D1 = +-15 min um Auftragszeit, D2 = +-15 min um Mitte
                        zwischen Auftrag und Ergebnis, D3 = [Auftrag-15 min, Ergebnis+15 min]
  kalib                 HF-Rauschschwelle aus 100 zufaelligen Train-Besuchen (ohne Bezug zu Kalium), nach hf_kalib.json
  messen   [N] [N_WORKER]  Messkette AW-203 v2 (werkzeuge/n3) auf einer Zufallsstichprobe von N Entnahmen, Fenster 30/60/300 s um D1
  paare    [N_WORKER]  Messkette auf allen Entnahmen der Paare mit sauberem D1 beidseits (paar_draws.csv), Fenster 60/300 s um den besten sauberen Block

Regeln: nie split_*_test.csv; keine Kennungen in Ausgaben ausser den privaten Dateien; Einzelprozesse unter 0,5 GB; keine Loeschbefehle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time
sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd
import scipy.signal as ss
from pathlib import Path

W = Path(str(data_path("s3.work_dir")))
WAVE = Path(str(data_path("s3.mcmed_waveforms_dir")))
FS = 500
BLOCK_S = 10
SAT = 4.5
SOS_BP = ss.butter(2, [0.5, 40], 'bp', fs=FS, output='sos')
SOS_QRS = ss.butter(2, [5, 20], 'bp', fs=FS, output='sos')


def _segs():
    s = pd.read_csv(W / 'cand_seg.csv')
    d = {}
    for r in s.itertuples():
        d.setdefault(int(r.CSN), []).append((int(r.seg), float(r.fs), int(r.n), float(r.t0), float(r.t1)))
    return d


def lies(csn, seg, t0_seg, fs, n, a_t, b_t):
    """Roh-EKG II (mV) von a_t bis b_t (Sekunden, absolut). Rueckgabe: (Beginn absolut, Array) oder None."""
    import wfdb
    a = max(0, int(np.floor((a_t - t0_seg) * fs)))
    b = min(n, int(np.ceil((b_t - t0_seg) * fs)))
    if b - a < 10 * fs:
        return None
    name = str(WAVE / str(csn)[-3:] / str(csn) / 'II' / f'{csn}_{seg}')
    rec = wfdb.rdrecord(name, sampfrom=a, sampto=b, physical=True, channels=[0])
    return t0_seg + a / fs, rec.p_signal[:, 0].astype(float)


def block_gut(xr, xf, xq):
    """10-s-Block: Rohsignal xr (mV), Bandpass 0,5-40 Hz xf, QRS-Band xq. Rueckgabe (sauber, sqi, hf_bpm, n_schlaege, grund)."""
    if not np.isfinite(xr).all():
        return False, 0.0, np.nan, 0, 'nan'
    if np.abs(xr).max() > SAT:
        return False, 0.0, np.nan, 0, 'saettigung'
    dz = (np.diff(xr) == 0).astype(int)
    if dz.sum() >= 30:
        run = np.convolve(dz, np.ones(30, int), 'valid')       # 60 ms ohne Aenderung
        if (run >= 30).any():
            return False, 0.0, np.nan, 0, 'flach'
    if np.ptp(xf) < 0.3:
        return False, 0.0, np.nan, 0, 'amplitude'
    env = np.convolve(np.gradient(xq) ** 2, np.ones(int(0.1 * FS)) / int(0.1 * FS), 'same')
    thr = 0.35 * np.percentile(env, 98)
    pk, _ = ss.find_peaks(env, height=thr, distance=int(0.25 * FS))
    ref = []
    for p in pk:
        lo, hi = max(0, p - 20), min(len(xf), p + 21)
        ref.append(lo + int(np.argmax(xf[lo:hi])))
    R = np.unique(np.array(ref, int))
    if len(R) < 5 or len(R) > 37:
        return False, 0.0, np.nan, len(R), 'schlaege'
    rr = np.diff(R) / FS
    if not (0.27 <= np.median(rr) <= 2.0):
        return False, 0.0, np.nan, len(R), 'hf'
    pre, post = int(0.2 * FS), int(0.4 * FS)
    ok = [r for r in R if r - pre >= 0 and r + post <= len(xf)]
    if len(ok) < 5:
        return False, 0.0, np.nan, len(R), 'rand'
    B = np.array([xf[r - pre:r + post] for r in ok])
    med = np.median(B, axis=0)
    if np.ptp(med) < 0.15:
        return False, 0.0, np.nan, len(R), 'qrs_klein'
    rho = np.array([np.corrcoef(b, med)[0, 1] for b in B])
    sqi = float(np.median(rho))
    gut = sqi >= 0.90 and (rho >= 0.80).mean() >= 0.80
    return bool(gut), sqi, 60.0 / float(np.median(rr)), len(R), 'ok' if gut else 'vorm'


def verarbeite_guete(args):
    csn, t_order, t_result, segs = args
    tr = t_result if (t_result == t_result) else t_order + 3600.0
    tmin, tmax = min(t_order, tr), max(t_order, tr)
    lo, hi = tmin - 900.0, min(tmax + 900.0, tmin + 3 * 3600.0 + 900.0)
    mid = (t_order + tr) / 2.0
    out = dict(CSN=csn, t_order=t_order, n_blocks=0, d1=0, d2=0, d3=0, n_d1=0, n_d2=0, n_d3=0, best_sqi_d1=np.nan, best_t_d1=np.nan, hr_d1=np.nan,
               fehler='')
    gruende = {}
    sqis_d1 = []
    for seg, fs, n, t0, t1 in segs:
        if t1 < lo or t0 > hi or fs != FS:
            continue
        try:
            r = lies(csn, seg, t0, fs, n, lo, hi)
        except Exception as e:
            out['fehler'] = type(e).__name__
            continue
        if r is None:
            continue
        ts, xr = r
        xr = np.asarray(xr)
        nanm = ~np.isfinite(xr)
        xf_in = np.where(nanm, 0.0, xr)
        xf = ss.sosfiltfilt(SOS_BP, xf_in) if len(xf_in) > 200 else xf_in
        xq = ss.sosfiltfilt(SOS_QRS, xf_in) if len(xf_in) > 200 else xf_in
        L = BLOCK_S * FS
        t_end = ts + len(xr) / FS
        t = max(ts, lo)
        while t + BLOCK_S <= min(t_end, hi) + 1e-9:
            inD1 = (t_order - 900 <= t) and (t + BLOCK_S <= t_order + 900)
            i = int(round((t - ts) * FS))
            if i + L <= len(xr):
                gut, sqi, hrb, nb, grund = block_gut(np.where(nanm[i:i + L], np.nan, xr[i:i + L]), xf[i:i + L], xq[i:i + L])
                out['n_blocks'] += 1
                gruende[grund] = gruende.get(grund, 0) + 1
                if gut:
                    inD2 = (mid - 900 <= t) and (t + BLOCK_S <= mid + 900)
                    inD3 = (min(t_order, tr) - 900 <= t) and (t + BLOCK_S <= max(t_order, tr) + 900)
                    if inD1:
                        out['d1'] = 1
                        out['n_d1'] += 1
                        if not (sqi <= out['best_sqi_d1']):
                            out['best_sqi_d1'], out['best_t_d1'], out['hr_d1'] = sqi, t, hrb
                    if inD2:
                        out['d2'] = 1
                        out['n_d2'] += 1
                    if inD3:
                        out['d3'] = 1
                        out['n_d3'] += 1
            t += 30.0 if inD1 else 60.0
    out['gruende'] = json.dumps(gruende)
    return out


def modus_guete(nw=2):
    from multiprocessing import Pool
    cand = pd.read_csv(W / 'cand.csv')
    sg = _segs()
    arbeit = [(int(r.CSN), float(r.t_order), float(r.t_result) if r.t_result == r.t_result else float('nan'), sg.get(int(r.CSN), [])) for r in cand.itertuples()]
    t0 = time.time()
    res = []
    with Pool(nw) as p:
        for i, o in enumerate(p.imap(verarbeite_guete, arbeit, chunksize=4)):
            res.append(o)
            if i % 100 == 0:
                print(json.dumps({'fertig': i, 'von': len(arbeit), 's': round(time.time() - t0)}), flush=True)
    pd.DataFrame(res).to_csv(W / 'signal_guete.csv', index=False)
    print('fertig', round(time.time() - t0), 's')


# ----------------------------------------------------------------------------------------------- Messbarkeit
def _init_messen():
    sys.path.insert(0, str(data_path("s3.n3_dir")))
    global V, M
    import aw197_messung as M
    import aw203_messung as V
    M.CFG['min_cover_s'] = 20.0       # nur fuer die Machbarkeit; die gebundene Kette verlangt 60 s


def modus_kalib():
    _init_messen()
    import wfdb
    import duckdb
    c = duckdb.connect(str(W / 's3.duckdb'), read_only=True)
    c.execute("SET memory_limit='500MB'; SET threads=1")
    rng = np.random.default_rng(20261016)
    csns = [int(x[0]) for x in c.execute("select distinct CSN from seg where t1-t0>=600 order by CSN").fetchall()]
    # Kalibrierung ohne Bezug zu Kalium: zufaellige Train-Besuche
    ziehen = rng.choice(csns, size=min(100, len(csns)), replace=False)
    hf = []
    for csn in ziehen:
        seg, fs, n, t0, t1 = c.execute("select seg,fs,n,t0,t1 from seg where CSN=? order by (t1-t0) desc limit 1", [int(csn)]).fetchone()
        if fs != FS:
            continue
        a_t = t0 + (t1 - t0) / 2 - 30
        r = lies(int(csn), seg, t0, fs, n, a_t, a_t + 60)
        if r is None:
            continue
        x = np.nan_to_num(r[1])
        try:
            R = M.detect_r(x)
            hv, _ = M.hf_beats(x, R)
            hf.extend(list(hv[np.isfinite(hv)]))
        except Exception:
            pass
    hf = np.array(hf)
    sw = float(np.percentile(hf, 99))
    (W / 'hf_kalib.json').write_text(json.dumps({'n_schlaege': int(len(hf)), 'schwelle_p99': sw, 'median': float(np.median(hf)), 'quelle': 'zufaellige Train-Besuche, 60 s Mitte laengstes Segment, ohne Kaliumbezug'}))
    print({'n_schlaege': len(hf), 'schwelle_p99': sw, 'median': float(np.median(hf))})


def verarbeite_messen(args):
    csn, t_order, segs, hf = args
    out = []
    for dur in (30, 60, 300):
        t0w, t1w = t_order - dur / 2, t_order + dur / 2
        best = None
        for seg, fs, n, t0, t1 in segs:
            ov = min(t1, t1w + 3) - max(t0, t0w - 3)
            if fs == FS and ov > 0 and (best is None or ov > best[0]):
                best = (ov, seg, fs, n, t0, t1)
        row = dict(CSN=csn, t_order=t_order, dur=dur, reason='kein_segment')
        if best is not None:
            _, seg, fs, n, t0, t1 = best
            try:
                r = lies(csn, seg, t0, fs, n, t0w - 4, t1w + 4)
                if r is None:
                    row['reason'] = 'zu_kurz'
                else:
                    ts, xr = r
                    row = dict(CSN=csn, t_order=t_order, dur=dur)
                    row.update(V.measure_window_v2(xr, ts, t0w, t1w, hf, splithalf=True, do_dwt=True))
            except Exception as e:
                row['reason'] = 'fehler_' + type(e).__name__
        out.append(row)
    return out


def modus_messen(N=500, nw=2):
    from multiprocessing import Pool
    hf = json.loads((W / 'hf_kalib.json').read_text())['schwelle_p99']
    cand = pd.read_csv(W / 'cand.csv')
    g = pd.read_csv(W / 'signal_guete.csv')
    cand = cand.merge(g[['CSN', 't_order', 'd1']], on=['CSN', 't_order'])
    cand = cand[cand.d1 == 1]                    # nur Entnahmen mit mindestens einem sauberen 10-s-Block in D1
    rng = np.random.default_rng(20261017)
    idx = rng.choice(len(cand), size=min(N, len(cand)), replace=False)
    sub = cand.iloc[sorted(idx)]
    sg = _segs()
    arbeit = [(int(r.CSN), float(r.t_order), sg.get(int(r.CSN), []), hf) for r in sub.itertuples()]
    t0 = time.time()
    res = []
    with Pool(nw, initializer=_init_messen) as p:
        for i, o in enumerate(p.imap(verarbeite_messen, arbeit, chunksize=2)):
            res.extend(o)
            if i % 20 == 0:
                print(json.dumps({'fertig': i, 'von': len(arbeit), 's': round(time.time() - t0)}), flush=True)
    pd.DataFrame(res).to_csv(W / 'messbar.csv', index=False)
    print('fertig', round(time.time() - t0), 's')


def verarbeite_paar(args):
    csn, t_order, t_best, segs, hf = args
    out = []
    for dur in (60, 300):
        tc = t_best + BLOCK_S / 2.0
        t0w, t1w = tc - dur / 2, tc + dur / 2
        best = None
        for seg, fs, n, t0, t1 in segs:
            ov = min(t1, t1w + 3) - max(t0, t0w - 3)
            if fs == FS and ov > 0 and (best is None or ov > best[0]):
                best = (ov, seg, fs, n, t0, t1)
        row = dict(CSN=csn, t_order=t_order, dur=dur, reason='kein_segment')
        if best is not None:
            _, seg, fs, n, t0, t1 = best
            try:
                r = lies(csn, seg, t0, fs, n, t0w - 4, t1w + 4)
                if r is None:
                    row['reason'] = 'zu_kurz'
                else:
                    ts, xr = r
                    row = dict(CSN=csn, t_order=t_order, dur=dur)
                    row.update(V.measure_window_v2(xr, ts, t0w, t1w, hf, splithalf=True, do_dwt=True))
            except Exception as e:
                row['reason'] = 'fehler_' + type(e).__name__
        out.append(row)
    return out


def modus_paare(nw=3):
    from multiprocessing import Pool
    hf = json.loads((W / 'hf_kalib.json').read_text())['schwelle_p99']
    d = pd.read_csv(W / 'paar_draws.csv')
    sg = _segs_alle(set(d.CSN.astype(int)))
    arbeit = [(int(r.CSN), float(r.t_order), float(r.best_t_d1), sg.get(int(r.CSN), []), hf) for r in d.itertuples()]
    t0 = time.time()
    res = []
    with Pool(nw, initializer=_init_messen) as p:
        for i, o in enumerate(p.imap(verarbeite_paar, arbeit, chunksize=2)):
            res.extend(o)
            if i % 50 == 0:
                print(json.dumps({'fertig': i, 'von': len(arbeit), 's': round(time.time() - t0)}), flush=True)
    pd.DataFrame(res).to_csv(W / 'messbar_paare.csv', index=False)
    print('fertig', round(time.time() - t0), 's')


def _segs_alle(csns):
    import duckdb
    c = duckdb.connect(str(W / 's3.duckdb'), read_only=True)
    c.execute("SET memory_limit='400MB'; SET threads=1")
    d = {}
    for csn, seg, fs, n, t0, t1 in c.execute('select CSN, seg, fs, n, t0, t1 from seg').fetchall():
        if int(csn) in csns:
            d.setdefault(int(csn), []).append((int(seg), float(fs), int(n), float(t0), float(t1)))
    return d


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    if m == 'guete':
        modus_guete(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif m == 'kalib':
        modus_kalib()
    elif m == 'paare':
        modus_paare(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif m == 'messen':
        modus_messen(int(sys.argv[2]) if len(sys.argv) > 2 else 500, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    else:
        print(__doc__)
