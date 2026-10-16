"""AW-242 S5-R1: unabhaengige Gegenrechnung der Zeitausrichtungszahlen (296/392 usw.).

Eigener Code, nur aus Rohquellen: A155-Featurecache (nur subject_id/study_id der Entwicklungsfolds), Masterindex, icustays,
machine_measurements und die Roh-CSV-Extraktion der Herzfrequenz (raw_hr.parquet, aus MIMIC-IV 3.1 chartevents.csv.gz).
Keine Nutzung von audit.duckdb, rechnen.py, aw242_kern.py oder aw242_data.py. Nur Entwicklungsfolds.
Der Personenvergleich mit der Basisrechnung (offset je Person) wird als Zaehler der Abweichungen ausgegeben.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, time, pickle
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r1')
import numpy as np, pandas as pd, duckdb, pyarrow.parquet as pq

B = Path(str(data_path("s5.mimic_validation")))
OUT = G.OUT
GRID = np.arange(-720, 721, 5)          # Minuten
NG = GRID.size


def whitelist(con):
    con.execute('CREATE TABLE wl(subject_id BIGINT, study_id BIGINT, fold VARCHAR)')
    for fold in ('train', 'validation'):
        p = B / f'runs/A155-2026-10-04-d6e70e/analysis_feature_cache/cache-89d6aaf0a6b5957e/split={fold}/features.parquet'
        for batch in pq.ParquetFile(p).iter_batches(columns=['subject_id', 'study_id'], batch_size=16384, use_threads=False):
            con.register('b', batch)
            con.execute(f"INSERT INTO wl SELECT subject_id, study_id, '{fold}' FROM b")


def ecg_table(con):
    m = (B / 'master/ecg_master_index.parquet').as_posix()
    con.execute(f"CREATE TABLE ecg AS SELECT w.subject_id, w.study_id, w.fold, m.ecg_time FROM wl w JOIN read_parquet('{m}') m ON m.subject_id=w.subject_id AND m.study_id=w.study_id AND m.split=w.fold")
    icu = (B / 'standardized/icustays.parquet').as_posix()
    # ICU-Zuordnung: genau ein Aufenthalt, ecg_time innerhalb [intime, outtime]
    con.execute(f"""CREATE TABLE icu_ecg AS SELECT e.subject_id, e.study_id, e.fold, e.ecg_time, any_value(i.icustay_id) AS stay, any_value(i.intime) AS t_in, any_value(i.outtime) AS t_out
                    FROM ecg e JOIN read_parquet('{icu}') i ON i.subject_id=e.subject_id AND e.ecg_time>=i.intime AND e.ecg_time<=i.outtime
                    GROUP BY e.subject_id, e.study_id, e.fold, e.ecg_time HAVING count(*)=1""")
    mm = (B / 'standardized/machine_measurements.parquet').as_posix()
    con.execute(f"""CREATE TABLE icu_rr AS SELECT c.*, 60000.0/r.rr_interval_ms AS hr FROM icu_ecg c JOIN read_parquet('{mm}') r ON r.subject_id=c.subject_id AND r.study_id=c.study_id
                    WHERE r.rr_interval_ms>0 AND isfinite(r.rr_interval_ms) AND 60000.0/r.rr_interval_ms BETWEEN 30 AND 220 AND abs(epoch(r.machine_ecg_time)-epoch(c.ecg_time))<=60""")


def chart_table(con):
    raw = (OUT / 'raw_hr.parquet').as_posix()
    con.execute(f"""CREATE TABLE chart AS SELECT r.subject_id, r.stay_id AS stay, r.charttime AS t, median(r.valuenum) AS hr FROM read_parquet('{raw}') r
                    WHERE r.subject_id IN (SELECT DISTINCT subject_id FROM icu_ecg) AND r.charttime IS NOT NULL AND r.valuenum BETWEEN 30 AND 220 GROUP BY r.subject_id, r.stay_id, r.charttime""")


def nearest_values(times, vals, targets):
    """Naechster Wert je Ziel; bei gleichem Abstand der fruehere. side='right'-Konvention."""
    n = len(times)
    j = np.searchsorted(times, targets, side='right')      # times[j-1] <= Ziel < times[j]
    lo = np.clip(j - 1, 0, n - 1)
    hi = np.clip(j, 0, n - 1)
    d_lo = targets - times[lo]
    d_hi = times[hi] - targets
    pick = np.where(j == 0, hi, np.where(j == n, lo, np.where(d_lo <= d_hi, lo, hi)))
    dist = np.abs(times[pick] - targets)
    return vals[pick], dist


def analyse_person(ecg_rows, charts):
    """ecg_rows: Liste (t_s, hr, stay, t_in_s, t_out_s); charts: stay -> (times_s, vals). Rueckgabe Dict oder None."""
    curves = []
    for t, hr, stay, tin, tout in ecg_rows:
        if stay not in charts:
            continue
        tt, vv = charts[stay]
        keep = (tt >= tin) & (tt <= tout)
        tt, vv = tt[keep], vv[keep]
        if tt.size == 0:
            continue
        targets = t + GRID * 60.0
        ref, dist = nearest_values(tt, vv, targets)
        inside = (targets >= tin) & (targets <= tout)
        if np.all(inside) and np.all(dist <= 1800.0):
            curves.append(np.abs(ref - hr))
    if not curves:
        return None
    M = np.array(curves)                              # (n, NG)
    loss = np.median(M, axis=0)
    mn = loss.min()
    zone_idx = np.flatnonzero(loss <= mn + 1.0)
    ties = np.flatnonzero(np.abs(loss - mn) <= 1e-10)
    best = min(ties, key=lambda ix: (abs(GRID[ix]), GRID[ix]))
    contiguous = bool(np.all(np.diff(zone_idx) == 1))
    width = int(GRID[zone_idx[-1]] - GRID[zone_idx[0]])
    touches_edge = bool(zone_idx[0] == 0 or zone_idx[-1] == NG - 1)
    ident = (M.shape[0] >= 3) and contiguous and width <= 120 and mn <= 5.0 and not touches_edge
    return dict(n=M.shape[0], offset=int(GRID[best]), width=width, ident=bool(ident), loss=loss)


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='800MB'; SET threads=2")
    whitelist(con); ecg_table(con); chart_table(con)
    basis = con.execute('SELECT fold, count(*), count(DISTINCT subject_id) FROM ecg GROUP BY fold ORDER BY fold').fetchall()
    icu_stats = con.execute('SELECT fold, count(*), count(DISTINCT subject_id) FROM icu_ecg GROUP BY fold ORDER BY fold').fetchall()
    rr_stats = con.execute('SELECT fold, count(*), count(DISTINCT subject_id) FROM icu_rr GROUP BY fold ORDER BY fold').fetchall()
    ep = con.execute("SELECT subject_id, fold, epoch(ecg_time) AS t, hr, stay, epoch(t_in) AS tin, epoch(t_out) AS tout FROM icu_rr ORDER BY subject_id, ecg_time").fetchnumpy()
    ch = con.execute("SELECT subject_id, stay, epoch(t) AS t, hr FROM chart ORDER BY subject_id, stay, t").fetchnumpy()
    pop_people = con.execute('SELECT fold, count(DISTINCT subject_id) FROM icu_ecg GROUP BY fold ORDER BY fold').fetchall()
    con.close()
    # Chart nach Person/Aufenthalt gruppieren
    csub, cstay = ch['subject_id'], ch['stay']
    brk = np.flatnonzero((csub[1:] != csub[:-1]) | (cstay[1:] != cstay[:-1])) + 1
    starts = np.r_[0, brk]; ends = np.r_[brk, len(csub)]
    chart_idx = {}
    for s, e in zip(starts, ends):
        chart_idx.setdefault(int(csub[s]), {})[int(cstay[s])] = (ch['t'][s:e].astype(float), ch['hr'][s:e].astype(float))
    sub = ep['subject_id']
    pb = np.r_[0, np.flatnonzero(sub[1:] != sub[:-1]) + 1, len(sub)]
    res = {}
    for a, b in zip(pb[:-1], pb[1:]):
        p = int(sub[a])
        rows = [(float(ep['t'][i]), float(ep['hr'][i]), int(ep['stay'][i]), float(ep['tin'][i]), float(ep['tout'][i])) for i in range(a, b)]
        r = analyse_person(rows, chart_idx.get(p, {}))
        r_fold = ep['fold'][a]
        if r is not None:
            r['fold'] = r_fold
            res[p] = r
    out = {}
    den = {f: n for f, n in pop_people}
    for f in ('train', 'validation'):
        sel = [r for r in res.values() if r['fold'] == f]
        idn = [r for r in sel if r['ident']]
        offs = np.array([r['offset'] for r in idn])
        pops = np.vstack([r['loss'] for r in sel])
        popmin = int(GRID[int(np.argmin(np.median(pops, axis=0)))])
        out[f] = dict(basis_ecgs=[x[1] for x in basis if x[0] == f][0], basis_people=[x[2] for x in basis if x[0] == f][0],
                      icu_ecgs=[x[1] for x in icu_stats if x[0] == f][0], icu_people=den[f],
                      rr_ecgs=[x[1] for x in rr_stats if x[0] == f][0], rr_people=[x[2] for x in rr_stats if x[0] == f][0],
                      complete_people=len(sel), complete_ecgs=int(sum(r['n'] for r in sel)), at_least_3=sum(r['n'] >= 3 for r in sel), identifiable=len(idn),
                      within={w: int(np.sum(np.abs(offs) <= w)) for w in (15, 60, 120, 360, 720)}, pop_min=popmin, width_median=float(np.median([r['width'] for r in idn])),
                      offset_median=float(np.median(offs)))
    # Personenvergleich mit der Basisrechnung
    base = pickle.load(open(OUT / 'base_persons.pkl', 'rb'))['persons'].set_index('subject_id')
    mism = 0; missing = 0
    for p, r in res.items():
        if p not in base.index or int(base.at[p, 'n_complete']) != r['n'] or bool(base.at[p, 'ok']) != r['ident'] or int(base.at[p, 'offset']) != r['offset']:
            mism += 1
    missing = int((base.n_complete > 0).sum()) - len(res)
    out['personenvergleich'] = dict(personen_unabhaengig=len(res), personen_basis=int((base.n_complete > 0).sum()), abweichende_personen=mism, fehlende_personen=missing)
    out['seconds'] = time.time() - t0; out['peak_rss'] = G.peak()
    (OUT / 'r1_result.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
