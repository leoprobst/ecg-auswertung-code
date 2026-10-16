"""AW-242 S5-R5 Teil 2: Paarbeschraenkung bei unveraendertem Analysekollektiv.

Die Vollabdeckungsregel (Referenz an allen 289 Offsets im Suchfenster) ist mit strengen Referenzbedingungen unvereinbar
(Teil 1: 0 Personen). Hier wird deshalb das Kollektiv der Basisrechnung (Personen mit mindestens einem vollstaendigen EKG)
beibehalten, und je (EKG, Offset)-Paar wird nur gezaehlt, wenn die in der Basisrechnung gewaehlte naechste Referenz
die Bedingung erfuellt (Speicherverzug, Minute, Abstand, Rhythmus). Personenkurve = Median ueber gueltige Paare (>= 1),
Populationskurve = Median ueber Personen mit gueltigem Wert (>= 100 Personen je Offset). Keine Identifizierbarkeit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, pickle, time, re
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r5b')
import numpy as np, pandas as pd, duckdb
from aw242_kern import D, K, SHIFT, I0, chart_slices
from aw242_data import load_audit
import r5_ref as R5

OUT = G.OUT


def pair_arrays(ecg, chart, kept, window=1800.0):
    """Je Person: V (n,K), h (n,), gi (n,K) globale Chartindizes der gewaehlten Referenz, dt (n,K)."""
    cs = chart['subject_id'].to_numpy(); cst = chart['stay'].to_numpy(); ct = chart['t'].to_numpy(float)
    sl = chart_slices(cs, cst)
    t = ecg['t'].to_numpy(float); stay = ecg['stay'].to_numpy(); st = ecg['start'].to_numpy(float); en = ecg['end'].to_numpy(float)
    sub = ecg['subject_id'].to_numpy()
    out = {}
    cache = {}
    for p, k in kept.items():
        GI = []; DT = []
        for i in k['idx']:
            key = (p, int(stay[i]))
            if key not in cache:
                a, b = sl[key]
                tt = ct[a:b]; m = (tt >= st[i]) & (tt <= en[i])
                cache[key] = (tt[m], np.arange(a, b)[m])
            tt, gidx = cache[key]
            tg = t[i] + SHIFT
            kk = np.searchsorted(tt, tg)
            left = np.clip(kk - 1, 0, len(tt) - 1); right = np.clip(kk, 0, len(tt) - 1)
            pick = np.where(np.abs(tt[left] - tg) <= np.abs(tt[right] - tg), left, right)
            GI.append(gidx[pick]); DT.append(np.abs(tt[pick] - tg))
        out[p] = (np.vstack(GI), np.vstack(DT))
    return out


def main():
    t0 = time.time()
    data = load_audit()
    ecg, chart = data['ecg'], data['chart']
    kept = pickle.load(open(OUT / 'base_persons.pkl', 'rb'))['kept']
    persons = pickle.load(open(OUT / 'base_persons.pkl', 'rb'))['persons'].set_index('subject_id')
    c = duckdb.connect(); c.execute("SET memory_limit='900MB'; SET threads=2")
    c.execute(("ATTACH '" + str(data_path("s5.uhrzeit_aw110_audit_duckdb")) + "' AS a (READ_ONLY)"))
    aux = c.execute(f"""SELECT epoch(h.t) AS t, epoch(r.storetime)-epoch(r.charttime) AS delay_s, minute(h.t) AS mi
        FROM a.hf h LEFT JOIN read_parquet('{(OUT / 'raw_hr.parquet').as_posix()}') r ON r.subject_id=h.subject_id AND r.stay_id=h.icustay_id AND r.charttime=h.t
        ORDER BY h.subject_id, h.icustay_id, h.t""").fetchnumpy()
    assert np.array_equal(aux['t'], chart['t'].to_numpy(float))
    delay = aux['delay_s'].astype(float); minute = aux['mi']
    rep = c.execute(f"SELECT m.study_id, m.machine_report FROM read_parquet('{R5.MASTER}') m JOIN (SELECT DISTINCT study_id FROM a.eligible) e USING(study_id)").df()
    c.close()
    sinus_fn = R5.load_report_sinus()
    rep['sinus'] = rep.machine_report.map(sinus_fn); rep['pace'] = rep.machine_report.fillna('').map(lambda s: R5.PACE.search(s) is not None)
    e2 = ecg.merge(rep[['study_id', 'sinus', 'pace']], on='study_id', how='left', validate='one_to_one')
    ecg_sinus = (e2.sinus.fillna(False) & ~e2.pace.fillna(True)).to_numpy()
    pa = pair_arrays(ecg, chart, kept)
    conds = {
        'V0 alle Paare (Basis)': lambda gi, dt, es: np.ones_like(dt, bool),
        'Va Sinusrhythmus ohne Stimulation': lambda gi, dt, es: np.broadcast_to(es[:, None], dt.shape),
        'Vb5 Speicherverzug 0-5 min': lambda gi, dt, es: (delay[gi] >= 0) & (delay[gi] <= 300),
        'Vb15 Speicherverzug 0-15 min': lambda gi, dt, es: (delay[gi] >= 0) & (delay[gi] <= 900),
        'Vc Minute ungleich 0': lambda gi, dt, es: minute[gi] != 0,
        'Vd10 Abstand hoechstens 10 min': lambda gi, dt, es: dt <= 600,
        'Vd15 Abstand hoechstens 15 min': lambda gi, dt, es: dt <= 900,
        'Ve1 Sinus + Verzug 0-15 + Abstand 15': lambda gi, dt, es: np.broadcast_to(es[:, None], dt.shape) & (delay[gi] >= 0) & (delay[gi] <= 900) & (dt <= 900),
        'Ve2 Minute ungleich 0 + Verzug 0-15 + Abstand 15': lambda gi, dt, es: (minute[gi] != 0) & (delay[gi] >= 0) & (delay[gi] <= 900) & (dt <= 900),
    }
    fold_of = persons['fold'].to_dict()
    rows = []
    for name, fn in conds.items():
        acc = {'train': [], 'validation': []}
        tot = {'train': [0, 0], 'validation': [0, 0]}
        for p, k in kept.items():
            gi, dt = pa[p]
            es = ecg_sinus[k['idx']]
            M = fn(gi, dt, es)
            absd = np.abs(k['V'] - k['h'][:, None])
            nv = M.sum(axis=0)
            with np.errstate(all='ignore'):
                L = np.where(nv >= 1, np.nanmedian(np.where(M, absd, np.nan), axis=0), np.nan)
            f = fold_of[p]
            acc[f].append(L)
            tot[f][0] += int(M.sum()); tot[f][1] += int(M.size)
        for f in ('train', 'validation'):
            A = np.vstack(acc[f])
            ncontrib = np.sum(np.isfinite(A), axis=0)
            with np.errstate(all='ignore'):
                pop = np.where(ncontrib >= 100, np.nanmedian(A, axis=0), np.nan)
            ok = np.isfinite(pop)
            amin = int(D[np.nanargmin(np.where(ok, pop, np.inf))]) if ok.any() else None
            rows.append(dict(variante=name, fold=f, personen_basis=len(acc[f]), gueltige_paare_anteil=tot[f][0] / tot[f][1],
                             personen_mit_wert_bei_offset0=int(ncontrib[I0]), populationskurve_minimum_min=amin,
                             populationswert_offset0=float(pop[I0]) if ok[I0] else None, populationswert_minimum=float(np.nanmin(np.where(ok, pop, np.inf))) if ok.any() else None,
                             offsets_mit_mind100_personen=int(ok.sum()),
                             populationswert_offset_plus60=float(pop[np.where(D == 60)[0][0]]) if ok[np.where(D == 60)[0][0]] else None,
                             populationswert_offset_minus60=float(pop[np.where(D == -60)[0][0]]) if ok[np.where(D == -60)[0][0]] else None))
        G.status('r5b', name)
    pd.DataFrame(rows).to_csv(OUT / 'r5b_raw.csv', index=False)
    print('R5b fertig', round(time.time() - t0, 1), 's, RSS', G.peak())


if __name__ == '__main__':
    main()
