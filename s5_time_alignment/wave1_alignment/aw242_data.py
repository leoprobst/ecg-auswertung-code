"""AW-242: Laden der AW-110-Arbeitsdatenbank (nur lesend) und Personenrechnung ueber alle Personen.

Private Personenzeilen bleiben im Arbeitsspeicher bzw. unter <local path>. Keine Kennungen in Ausgaben.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
from aw242_kern import (D, K, SHIFT, ecg_matrix, estimate_from_loss, loss_curve, chart_slices)

AUDIT = str(data_path("s5.uhrzeit_aw110_audit_duckdb"))


def load_audit(path=AUDIT):
    import duckdb
    c = duckdb.connect(path, read_only=True)
    c.execute("SET memory_limit='700MB'; SET threads=2")
    ecg = c.execute("""SELECT subject_id, study_id, epoch(ecg_time) AS t, hf, icustay_id AS stay, epoch(intime) AS start,
                       epoch(outtime) AS "end", fold, first_careunit AS unit, anchor_year_group AS year, machine_cart_id AS cart
                       FROM eligible ORDER BY subject_id, study_id""").df()
    prim = c.execute("""SELECT subject_id, study_id, epoch(ecg_time) AS t, icustay_id AS stay, epoch(intime) AS start,
                        epoch(outtime) AS "end", fold, first_careunit AS unit FROM primary_ecg ORDER BY subject_id, study_id""").df()
    people = c.execute('SELECT subject_id, fold FROM primary_people ORDER BY subject_id').df()
    ch = c.execute('SELECT subject_id, icustay_id AS stay, epoch(t) AS t, hf FROM hf ORDER BY subject_id, icustay_id, t').fetchnumpy()
    c.close()
    chart = pd.DataFrame({k: v for k, v in ch.items()})
    return dict(ecg=ecg, prim=prim, people=people, chart=chart)


def compute_persons(ecg, chart, window=1800.0, chart_mask=None, ecg_mask=None, keep=False, people=None):
    """Personenrechnung. ecg: DataFrame (sortiert nach subject_id), chart: DataFrame (sortiert subject, stay, t).

    chart_mask/ecg_mask: boolesche Arrays zur Variantenbildung (gleiche Laenge wie die Tabellen).
    Rueckgabe: DataFrame je Person mit mindestens einem EKG (alle Felder), plus optional Kurvenmatrizen.
    """
    if chart_mask is not None:
        chart = chart.loc[chart_mask]
    if ecg_mask is not None:
        ecg = ecg.loc[ecg_mask]
    cs = chart['subject_id'].to_numpy(); cst = chart['stay'].to_numpy()
    ct = chart['t'].to_numpy(float); chf = chart['hf'].to_numpy(float)
    sl = chart_slices(cs, cst)
    cache = {}

    def stay_arrays(subj, stay, start, end):
        key = (subj, stay)
        if key in cache:
            return cache[key]
        if key not in sl:
            cache[key] = None
            return None
        a, b = sl[key]
        tt = ct[a:b]; vv = chf[a:b]
        m = (tt >= start) & (tt <= end)
        res = (tt[m], vv[m]) if m.any() else None
        cache[key] = res
        return res

    sub = ecg['subject_id'].to_numpy(); t = ecg['t'].to_numpy(float); hf = ecg['hf'].to_numpy(float)
    stay = ecg['stay'].to_numpy(); st = ecg['start'].to_numpy(float); en = ecg['end'].to_numpy(float)
    bounds = np.r_[0, np.flatnonzero(sub[1:] != sub[:-1]) + 1, len(sub)]
    rows = []; kept = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = int(sub[a])
        n_elig = b - a; n_bound = 0; n_chart = 0; Vs = []; hs = []; idx = []
        for i in range(a, b):
            boundary_ok = bool(t[i] - 720 * 60 >= st[i] and t[i] + 720 * 60 <= en[i])
            n_bound += boundary_ok
            arr = stay_arrays(p, int(stay[i]), st[i], en[i])
            if arr is None:
                continue
            n_chart += 1
            tt, vv = arr
            V, dt, ok = _matrix(t[i], st[i], en[i], tt, vv, window)
            if ok:
                Vs.append(V); hs.append(hf[i]); idx.append(i)
        rec = dict(subject_id=p, fold=ecg['fold'].iat[a], n_eligible=int(n_elig), n_boundary_ok=int(n_bound),
                   n_stay_has_chart=int(n_chart), n_complete=len(Vs))
        if Vs:
            Vm = np.vstack(Vs); hm = np.array(hs)
            loss = loss_curve(Vm, hm)
            rec.update(estimate_from_loss(loss, len(Vs)))
            rec['pop_loss'] = loss
            if keep:
                kept[p] = dict(V=Vm, h=hm, idx=np.array(idx))
        rows.append(rec)
    return pd.DataFrame(rows), kept


def _matrix(t, start, end, tt, vv, window):
    targets = t + SHIFT
    k = np.searchsorted(tt, targets)
    left = np.clip(k - 1, 0, len(tt) - 1)
    right = np.clip(k, 0, len(tt) - 1)
    pick = np.where(np.abs(tt[left] - targets) <= np.abs(tt[right] - targets), left, right)
    dt = np.abs(tt[pick] - targets)
    ok = bool(np.all((dt <= window) & (targets >= start) & (targets <= end)))
    return vv[pick], dt, ok


def summarize(df, denominators, folds=('train', 'validation')):
    """Kennzahlen je Fold (Parity zu icu_results.json)."""
    out = {}
    for f in folds:
        s = df[df.fold == f]
        good = s[s.ok.fillna(False).astype(bool)] if 'ok' in s else s.iloc[0:0]
        offs = good.offset.to_numpy() if len(good) else np.array([])
        r = dict(primary_people=int(denominators[f]), complete_people=int((s.n_complete > 0).sum()),
                 complete_ecgs=int(s.n_complete.sum()), at_least_3_people=int((s.n_complete >= 3).sum()),
                 identifiable_people=int(len(good)))
        if len(good):
            r['offset_median'] = float(np.median(offs)); r['width_median'] = float(np.median(good.width))
            r['within'] = {w: int(np.sum(np.abs(offs) <= w)) for w in (15, 60, 120, 360, 720)}
        pops = [x for x in s['pop_loss'].dropna()] if 'pop_loss' in s else []
        if pops:
            curve = np.median(np.vstack(pops), axis=0)
            r['population_optimum_minutes'] = int(D[np.argmin(curve)])
        out[f] = r
    return out
