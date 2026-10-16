"""S5-MC-MED Stufe 0b: reine Zaehlung (Nenner und Verfuegbarkeit), keine Offsets, keine Herzfrequenz-Vergleiche.

Betriebsregeln (Auftrag K, 15.10.2026):
- nur split_random_train und split_random_val; split_*_test.csv wird nie geoeffnet (hartes Verbot im Code)
- keine Kennungen (CSN, MRN) in Ausgaben; private Tabellen nur auf <local path> (nicht <cloud folder>)
- Kleinzellenregel: Zaehlungen 1 bis 9 erscheinen als "<10"
- DuckDB memory_limit 700 MB, 2 Threads (unter 1,5 GB, keine Reservierung noetig)

Aufruf (Python: <local path> -I):
  s5_mcmed_zaehlung.py hash       Hashes der Eingaben gegen SHA256SUMS.txt
  s5_mcmed_zaehlung.py tabellen   privaten Besuchs-/Numerics-Bestand aufbauen, Zaehlungen Tabellen
  s5_mcmed_zaehlung.py koepfe     WFDB-Koepfe (Ableitung II) lesen, Zaehlungen Koepfe und Abdeckung
  s5_mcmed_zaehlung.py nenner     zeitliche Vollstaendigkeit (ohne HF-Werte) je G und Beziehung
  s5_mcmed_zaehlung.py wellenhash [N]  .hea/.dat (Ableitung II, train/val) gegen SHA256SUMS.txt, optional Stichprobe N
Ausgabe: <local path><modus>.json (maskiert, ohne Kennungen)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os, json, time, hashlib, math, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import duckdb

ROOT = Path(str(data_path("s5.analyse_mc_med_1_0_1")))
DATA = ROOT / 'data'
WAVE = DATA / 'waveforms'
PRIV = Path(str(data_path("s5.analyse_mc_med_s5")))
OUT = PRIV / 'zaehlung'
SPLITS = {'val': DATA / 'split_random_val.csv', 'train': DATA / 'split_random_train.csv'}
G_LIST = [15, 30, 60, 120, 180, 360]  # Minuten, Gitterhalbweite (Entwurf 5.1 nennt 60 bis 360; 15 und 30 zusaetzlich)
A_GRIDS = [(30, 1), (60, 2), (120, 2)]   # Beziehung A: (G in min, Schritt in min); primaer G=60, Schritt 2
MAXREF_B_VAR = [1800, 3600]           # naechste Chart-Referenz innerhalb 30 min (S5) bzw. 60 min (Sensitivitaet)
EXCL = 600                            # s, Ausschluss erste/letzte 10 min
UNIT_STEP = 300                       # s, Rasterabstand der Einheiten (primaer 900 = jede dritte)
MAXREF_S = 1800                       # S5: naechste Referenz innerhalb 30 min
MAXREF_A = 90                         # Beziehung A (Monitor, dicht)


def maske(n):
    """Kleinzellenregel: 1..9 -> '<10'."""
    if n is None:
        return None
    if isinstance(n, (int, np.integer)) and 1 <= int(n) <= 9:
        return '<10'
    return int(n) if isinstance(n, (int, np.integer)) else n


def mask_tree(o):
    if isinstance(o, dict):
        return {k: mask_tree(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [mask_tree(v) for v in o]
    if isinstance(o, (np.integer,)):
        return maske(int(o))
    if isinstance(o, int) and not isinstance(o, bool):
        return maske(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def sha(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def verbinden():
    PRIV.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(str(PRIV / 's5_privat.duckdb'))
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.execute(f"SET temp_directory='{(PRIV / 'spill').as_posix()}'")
    return c


def schreibe(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(mask_tree(obj), indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return p


def quant(c, sql):
    r = c.execute(sql).fetchone()
    return [None if x is None else float(x) for x in r[0]] if r and r[0] is not None else None


# --------------------------------------------------------------------------- hash
def modus_hash():
    soll = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            soll[parts[1].strip()] = parts[0].lower()
    ziel = ['LICENSE.txt', 'README.md', 'RECORDS', 'data-dictionary.md', 'data/visits.csv', 'data/numerics.csv',
            'data/waveform_summary.csv', 'data/split_random_train.csv', 'data/split_random_val.csv']
    res = {}
    for z in ziel:
        h = sha(ROOT / z)
        res[z] = {'sha256': h, 'in_sums': z in soll, 'gleich': soll.get(z) == h}
    res['SHA256SUMS.txt'] = {'sha256': sha(ROOT / 'SHA256SUMS.txt'), 'zeilen': len(soll)}
    res['hinweis'] = 'split_random_test.csv bewusst nicht gehasht oder geoeffnet'
    schreibe('hash', res)
    print(json.dumps({k: v.get('gleich', v.get('sha256')) for k, v in res.items() if isinstance(v, dict)}, indent=1))


# --------------------------------------------------------------------------- tabellen
VISIT_COLS = {'MRN': 'BIGINT', 'CSN': 'BIGINT', 'Visit_no': 'INT', 'Visits': 'INT', 'Age': 'DOUBLE', 'Gender': 'VARCHAR',
              'Race': 'VARCHAR', 'Ethnicity': 'VARCHAR', 'Means_of_arrival': 'VARCHAR', 'Triage_Temp': 'DOUBLE',
              'Triage_HR': 'DOUBLE', 'Triage_RR': 'DOUBLE', 'Triage_SpO2': 'DOUBLE', 'Triage_SBP': 'DOUBLE',
              'Triage_DBP': 'DOUBLE', 'Triage_acuity': 'VARCHAR', 'CC': 'VARCHAR', 'ED_dispo': 'VARCHAR',
              'Hours_to_next_visit': 'DOUBLE', 'Dispo_class_next_visit': 'VARCHAR', 'ED_LOS': 'DOUBLE',
              'Hosp_LOS': 'DOUBLE', 'DC_dispo': 'VARCHAR', 'Payor_class': 'VARCHAR', 'Admit_service': 'VARCHAR',
              'Dx_ICD9': 'VARCHAR', 'Dx_ICD10': 'VARCHAR', 'Dx_name': 'VARCHAR', 'Arrival_time': 'TIMESTAMP',
              'Roomed_time': 'TIMESTAMP', 'Dispo_time': 'TIMESTAMP', 'Admit_time': 'TIMESTAMP',
              'Departure_time': 'TIMESTAMP'}


def aufbau(c):
    """Privater Bestand: nur train und val (Test wird nie gelesen)."""
    for s, p in SPLITS.items():
        assert 'test' not in p.name
    c.execute('CREATE OR REPLACE TABLE sp(CSN BIGINT, split VARCHAR)')
    for s, p in SPLITS.items():
        c.execute(f"INSERT INTO sp SELECT column0,'{s}' FROM read_csv('{p.as_posix()}',header=false,columns={{'column0':'BIGINT'}})")
    cols = ','.join(f"'{k}':'{v}'" for k, v in VISIT_COLS.items())
    c.execute(f"""CREATE OR REPLACE TABLE visit AS SELECT v.CSN,v.MRN,sp.split,v.Age,v.Gender,v.ED_LOS,
        epoch_us(v.Arrival_time)/1e6 a, epoch_us(v.Roomed_time)/1e6 r, epoch_us(v.Departure_time)/1e6 d
        FROM read_csv('{(DATA / 'visits.csv').as_posix()}',header=true,columns={{{cols}}}) v JOIN sp USING(CSN)""")
    c.execute(f"""CREATE OR REPLACE TABLE wsum AS SELECT w.CSN,w.Type,w.Segments,w.Duration
        FROM read_csv('{(DATA / 'waveform_summary.csv').as_posix()}',header=true) w JOIN sp USING(CSN)""")
    c.execute(f"""CREATE OR REPLACE TABLE num AS SELECT n.CSN,n.Source,n.Measure,n.Value,epoch_us(n.Time)/1e6 t
        FROM read_csv('{(DATA / 'numerics.csv').as_posix()}',header=true,columns={{'CSN':'BIGINT','Source':'VARCHAR','Measure':'VARCHAR','Value':'DOUBLE','Time':'TIMESTAMP'}}) n
        JOIN sp USING(CSN) WHERE n.Measure IN ('HR','1min_HRV','5min_HRV')""")


def modus_tabellen():
    c = verbinden()
    t0 = time.time()
    aufbau(c)
    R = {'aufbau_sekunden': round(time.time() - t0), 'splits': {}}
    for s in SPLITS:
        S = {}
        S['besuche'] = c.execute(f"SELECT count(*) FROM visit WHERE split='{s}'").fetchone()[0]
        S['patienten'] = c.execute(f"SELECT count(DISTINCT MRN) FROM visit WHERE split='{s}'").fetchone()[0]
        S['mit_roomed_und_departure'] = c.execute(f"SELECT count(*) FROM visit WHERE split='{s}' AND r IS NOT NULL AND d IS NOT NULL").fetchone()[0]
        S['ed_los_h_quantile_5_25_50_75_95'] = quant(c, f"SELECT quantile_cont(ED_LOS,[.05,.25,.5,.75,.95]) FROM visit WHERE split='{s}'")
        S['alter_median'] = c.execute(f"SELECT median(Age) FROM visit WHERE split='{s}'").fetchone()[0]
        S['geschlecht_F'] = c.execute(f"SELECT count(*) FROM visit WHERE split='{s}' AND Gender='F'").fetchone()[0]
        S['geschlecht_M'] = c.execute(f"SELECT count(*) FROM visit WHERE split='{s}' AND Gender='M'").fetchone()[0]
        # Wellenform II laut Zusammenfassung
        S['mit_ii'] = c.execute(f"SELECT count(*) FROM wsum JOIN visit USING(CSN) WHERE split='{s}' AND Type='II'").fetchone()[0]
        S['ii_dauer_s_quantile_5_25_50_75_95'] = quant(c, f"SELECT quantile_cont(Duration,[.05,.25,.5,.75,.95]) FROM wsum JOIN visit USING(CSN) WHERE split='{s}' AND Type='II'")
        S['ii_segmente_quantile_25_50_75_95'] = quant(c, f"SELECT quantile_cont(Segments,[.25,.5,.75,.95]) FROM wsum JOIN visit USING(CSN) WHERE split='{s}' AND Type='II'")
        S['ii_gesamtdauer_stunden'] = c.execute(f"SELECT sum(Duration)/3600 FROM wsum JOIN visit USING(CSN) WHERE split='{s}' AND Type='II'").fetchone()[0]
        # Numerics: Besuche mit HR
        fenster = "n.t BETWEEN v.r AND v.d"
        c.execute(f"""CREATE OR REPLACE TEMP TABLE bc AS SELECT v.CSN,
            count(*) FILTER (WHERE n.Source='Chart' AND n.Measure='HR' AND n.Value BETWEEN 30 AND 220) chart_all,
            count(*) FILTER (WHERE n.Source='Chart' AND n.Measure='HR' AND n.Value BETWEEN 30 AND 220 AND {fenster}) chart_fenster,
            count(*) FILTER (WHERE n.Source='Monitor' AND n.Measure='HR' AND n.Value BETWEEN 30 AND 220) mon_all,
            count(*) FILTER (WHERE n.Source='Monitor' AND n.Measure='HR' AND n.Value BETWEEN 30 AND 220 AND {fenster}) mon_fenster,
            count(*) FILTER (WHERE n.Measure='5min_HRV') hrv5,
            (SELECT count(*) FROM wsum w WHERE w.CSN=v.CSN AND w.Type='II') ii
            FROM visit v LEFT JOIN num n USING(CSN) WHERE v.split='{s}' GROUP BY v.CSN""")
        g = lambda w: c.execute(f"SELECT count(*) FROM bc WHERE {w}").fetchone()[0]
        S['mit_chart_hr'] = g('chart_all>=1')
        S['mit_chart_hr_im_fenster_mind3'] = g('chart_fenster>=3')
        S['mit_monitor_hr'] = g('mon_all>=1')
        S['mit_monitor_hr_im_fenster_mind30'] = g('mon_fenster>=30')
        S['mit_5min_hrv'] = g('hrv5>=1')
        S['ii_und_monitor_hr'] = g('ii>0 AND mon_all>=1')
        S['ii_und_chart_mind3_im_fenster'] = g('ii>0 AND chart_fenster>=3')
        S['ii_und_monitor_und_chart_mind3'] = g('ii>0 AND mon_all>=1 AND chart_fenster>=3')
        S['chart_hr_je_besuch_im_fenster_quantile_25_50_75_95'] = quant(c, "SELECT quantile_cont(chart_fenster,[.25,.5,.75,.95]) FROM bc WHERE chart_fenster>=1")
        S['chart_hr_zeilen_im_fenster'] = c.execute("SELECT sum(chart_fenster) FROM bc").fetchone()[0]
        S['monitor_hr_zeilen_im_fenster'] = c.execute("SELECT sum(mon_fenster) FROM bc").fetchone()[0]
        R['splits'][s] = S
    # G3: Sekundenphase und Raster, nur Besuche mit Chart und Monitor (Phase relativ zum Monitor)
    c.execute("""CREATE OR REPLACE TEMP TABLE ph AS
        SELECT n.CSN,n.Source,n.t,n.Value, (n.t - floor(n.t/60)*60) sec FROM num n WHERE n.Measure='HR'""")
    c.execute("""CREATE OR REPLACE TEMP TABLE monph AS SELECT CSN, median(sec) mphase, count(*) nm FROM ph WHERE Source='Monitor' GROUP BY CSN""")
    for s in SPLITS:
        r = c.execute(f"""SELECT count(*), count(DISTINCT p.CSN),
            avg((abs(((p.sec - m.mphase + 30) % 60 + 60) % 60 - 30) <= 1.0)::int)
            FROM ph p JOIN monph m USING(CSN) JOIN visit v USING(CSN) WHERE v.split='{s}' AND p.Source='Chart'""").fetchone()
        R['splits'][s]['g3_chart_zeilen_mit_monitor_besuch'] = r[0]
        R['splits'][s]['g3_besuche_mit_chart_und_monitor'] = r[1]
        R['splits'][s]['g3_anteil_chart_auf_monitor_phase_1s'] = r[2]
        # G4: Chart-Wert identisch mit Monitorwert desselben Zeitstempels (+-1 s) bzw. naechstes Monitorminute (+-60 s)
        r = c.execute(f"""WITH ch AS (SELECT p.CSN,p.t,p.Value FROM ph p JOIN visit v USING(CSN) WHERE v.split='{s}' AND p.Source='Chart' AND p.Value BETWEEN 30 AND 220 AND p.CSN IN (SELECT CSN FROM monph)),
            mo AS (SELECT CSN,t,Value FROM ph WHERE Source='Monitor')
            SELECT count(*), sum(ex::int), sum(nah::int), sum((vorhanden)::int) FROM (
              SELECT ch.CSN,ch.t,
                bool_or(abs(mo.t-ch.t)<=1 AND mo.Value=ch.Value) ex,
                bool_or(abs(mo.t-ch.t)<=60 AND mo.Value=ch.Value) nah,
                bool_or(abs(mo.t-ch.t)<=60) vorhanden
              FROM ch JOIN mo ON ch.CSN=mo.CSN AND mo.t BETWEEN ch.t-61 AND ch.t+61 GROUP BY ch.CSN,ch.t)""").fetchone()
        R['splits'][s]['g4_chart_zeilen_mit_monitorwert_in_61s'] = r[3]
        R['splits'][s]['g4_gleicher_wert_gleicher_zeitstempel_1s'] = r[1]
        R['splits'][s]['g4_gleicher_wert_innerhalb_60s'] = r[2]
    R['hinweis'] = ('Zeitverschiebung je Patient in beliebigen Sekunden: Sekundenphase je Besuch konstant; absolute Minutenstruktur '
                    'nicht beurteilbar, nur Phase relativ zum Monitor.')
    schreibe('tabellen', R)
    print(json.dumps(mask_tree(R), indent=1)[:6000])


# --------------------------------------------------------------------------- koepfe
def lies_kopf(path):
    L = Path(path).read_text().splitlines()
    a = L[0].split()
    b = L[1].split()
    tt = a[4] + ' ' + a[5]
    d = dt.datetime.strptime(tt, '%H:%M:%S.%f %d/%m/%Y') if '.' in a[4] else dt.datetime.strptime(tt, '%H:%M:%S %d/%m/%Y')
    t0 = d.replace(tzinfo=dt.timezone.utc).timestamp()
    gain = float(b[2].split('(')[0].split('/')[0]) if b[2][0].isdigit() else float('nan')
    return int(a[1]), float(a[2]), int(a[3]), t0, b[1], gain


def modus_koepfe():
    c = verbinden()
    sp = dict(c.execute('SELECT CSN,split FROM sp').fetchall())
    zeilen = []
    ordner_ii = 0
    for suf in sorted(x for x in os.listdir(WAVE) if x.isdigit()):
        for csn in os.listdir(WAVE / suf):
            if not csn.isdigit() or int(csn) not in sp:
                continue
            p = WAVE / suf / csn / 'II'
            if not p.is_dir():
                continue
            ordner_ii += 1
            for f in os.listdir(p):
                if f.endswith('.hea'):
                    try:
                        ns, fs, n, t0, fmt, gain = lies_kopf(p / f)
                    except Exception:
                        continue
                    seg = int(f[:-4].rsplit('_', 1)[1])
                    zeilen.append((int(csn), seg, fs, n, t0, fmt, gain))
    c.execute('CREATE OR REPLACE TABLE hd(CSN BIGINT, seg INT, fs DOUBLE, n BIGINT, t0 DOUBLE, fmt VARCHAR, gain DOUBLE)')
    c.executemany('INSERT INTO hd VALUES (?,?,?,?,?,?,?)', zeilen)
    c.execute('CREATE OR REPLACE TABLE hd2 AS SELECT *, t0 + n/fs t1 FROM hd')
    R = {'ordner_ii_train_val': ordner_ii, 'segmente': len(zeilen), 'splits': {}}
    erwartet = c.execute("SELECT split,count(*) FROM wsum JOIN visit USING(CSN) WHERE Type='II' GROUP BY 1").fetchall()
    for s, ne in erwartet:
        S = {'besuche_mit_ii_laut_zusammenfassung': ne}
        S['besuche_mit_kopf_auf_platte'] = c.execute(f"SELECT count(DISTINCT CSN) FROM hd JOIN visit USING(CSN) WHERE split='{s}'").fetchone()[0]
        S['segmente'] = c.execute(f"SELECT count(*) FROM hd JOIN visit USING(CSN) WHERE split='{s}'").fetchone()[0]
        S['abtastrate_hz'] = c.execute(f"SELECT fs,count(*) FROM hd JOIN visit USING(CSN) WHERE split='{s}' GROUP BY 1 ORDER BY 2 DESC").fetchall()
        S['format'] = c.execute(f"SELECT fmt,count(*) FROM hd JOIN visit USING(CSN) WHERE split='{s}' GROUP BY 1 ORDER BY 2 DESC").fetchall()
        S['segment_dauer_s_quantile_5_50_95'] = quant(c, f"SELECT quantile_cont(n/fs,[.05,.5,.95]) FROM hd JOIN visit USING(CSN) WHERE split='{s}'")
        S['segmente_vollstaendig_in_roomed_departure'] = c.execute(f"SELECT count(*) FROM hd2 JOIN visit v USING(CSN) WHERE v.split='{s}' AND t0>=v.r AND t1<=v.d").fetchone()[0]
        S['segmente_start_in_ankunft_departure'] = c.execute(f"SELECT count(*) FROM hd2 JOIN visit v USING(CSN) WHERE v.split='{s}' AND t0>=v.a AND t0<=v.d").fetchone()[0]
        S['segmente_start_vor_ankunft'] = c.execute(f"SELECT count(*) FROM hd2 JOIN visit v USING(CSN) WHERE v.split='{s}' AND t0<v.a").fetchone()[0]
        S['segmente_start_nach_departure'] = c.execute(f"SELECT count(*) FROM hd2 JOIN visit v USING(CSN) WHERE v.split='{s}' AND t0>v.d").fetchone()[0]
        S['start_minus_roomed_s_quantile_1_10_50_90_99'] = quant(c, f"SELECT quantile_cont(t0-v.r,[.01,.1,.5,.9,.99]) FROM hd2 JOIN visit v USING(CSN) WHERE v.split='{s}'")
        # Zeitbasis-Vorpruefung (nur Kopfzeit gegen erstes Monitor-Zeitstempel, ohne HF-Werte)
        S['zeitbasis_erster_monitorstempel_minus_kopfstart_s_quantile_5_25_50_75_95'] = quant(c, f"""
            SELECT quantile_cont(m.t1 - h.t0,[.05,.25,.5,.75,.95]) FROM
              (SELECT CSN,min(t0) t0 FROM hd2 WHERE seg=(SELECT min(seg) FROM hd2 x WHERE x.CSN=hd2.CSN) GROUP BY 1) h
              JOIN (SELECT CSN,min(t) t1 FROM num WHERE Source='Monitor' AND Measure='HR' GROUP BY 1) m USING(CSN) JOIN visit v USING(CSN) WHERE v.split='{s}'""")
        R['splits'][s] = S
    schreibe('koepfe', R)
    print(json.dumps(mask_tree(R), indent=1)[:5000])


# --------------------------------------------------------------------------- nenner
def einheiten_zeiten(t0, t1, r, d, step=UNIT_STEP):
    """Einheitenzeiten: Raster t0+60+k*step, Fenster [t-60,t+5] im Segment, t in [r+EXCL, d-EXCL]."""
    if t1 - t0 < 66:
        return np.empty(0)
    ts = np.arange(t0 + 60, t1 - 5 + 1e-9, step)
    return ts[(ts >= r + EXCL) & (ts <= d - EXCL)]


def vollstaendig(ts, ref, r, d, G_s, step_s, maxref):
    """Zeitliche Vollstaendigkeit wie S5 (ohne Werte): Ziel und Referenz innerhalb [r+EXCL,d-EXCL], naechste Referenz <= maxref."""
    if len(ts) == 0 or len(ref) == 0:
        return np.zeros(len(ts), bool)
    lo, hi = r + EXCL, d - EXCL
    ref = ref[(ref >= lo) & (ref <= hi)]
    if len(ref) == 0:
        return np.zeros(len(ts), bool)
    offs = np.arange(-G_s, G_s + 1, step_s)
    tg = ts[:, None] + offs[None, :]
    k = np.searchsorted(ref, tg)
    l = np.clip(k - 1, 0, len(ref) - 1)
    rr = np.clip(k, 0, len(ref) - 1)
    dist = np.minimum(np.abs(ref[l] - tg), np.abs(ref[rr] - tg))
    return np.all((dist <= maxref) & (tg >= lo) & (tg <= hi), axis=1)


def modus_nenner():
    c = verbinden()
    # Referenzen je Besuch laden (nur train/val, nur HR-Zeilen mit Wert 30..220; Werte werden sonst nicht benutzt)
    vis = {csn: (s, r, d) for csn, s, r, d in c.execute('SELECT CSN,split,r,d FROM visit WHERE r IS NOT NULL AND d IS NOT NULL').fetchall()}
    chart, mon = {}, {}
    for src, store in [('Chart', chart), ('Monitor', mon)]:
        cur = c.execute(f"SELECT CSN,t FROM num WHERE Measure='HR' AND Source='{src}' AND Value BETWEEN 30 AND 220 ORDER BY CSN,t")
        last, buf = None, []
        while True:
            rows = cur.fetchmany(500000)
            if not rows:
                break
            for csn, t in rows:
                store.setdefault(csn, []).append(t)
        for k in list(store):
            store[k] = np.unique(np.array(store[k]))
    seg = {}
    for csn, s0, t0, t1 in c.execute('SELECT CSN,seg,t0,t1 FROM hd2 ORDER BY CSN,seg').fetchall():
        seg.setdefault(csn, []).append((t0, t1))
    R = {'hinweis': ('zeitliche Vollstaendigkeit ohne Signalqualitaet und ohne HF-Werte; obere Grenze der Nenner. Nur Besuche, deren Koepfe '
                     'auf der Platte liegen (Abdeckung siehe koepfe.json).'), 'einheitenabstand_s': UNIT_STEP, 'splits': {}}
    for s in SPLITS:
        S = {'besuche_mit_kopf': 0, 'besuche_mit_einheit': 0, 'beziehung_B_chart': {}, 'beziehung_A_monitor': {}}
        for G in G_LIST:
            for mr in MAXREF_B_VAR:
                S['beziehung_B_chart'][f'G{G}_ref{mr // 60}min'] = {'ge3_alle_einheiten': 0, 'ge3_alle_einheiten_abstand_900': 0, 'ge1': 0, 'einheiten_gesamt': 0}
        S['beziehung_A_monitor'] = {f'G{g}_schritt{st}': {'ge3_alle_einheiten': 0, 'ge3_alle_einheiten_abstand_900': 0, 'ge1': 0, 'einheiten_gesamt': 0} for g, st in A_GRIDS}
        R['splits'][s] = S
    for csn, segs in seg.items():
        if csn not in vis:
            continue
        s, r, d = vis[csn]
        S = R['splits'][s]
        S['besuche_mit_kopf'] += 1
        ts_all = [einheiten_zeiten(a, b, r, d) for a, b in segs]
        ts = np.concatenate([x for x in ts_all]) if ts_all else np.empty(0)
        if len(ts) == 0:
            continue
        S['besuche_mit_einheit'] += 1
        # jede dritte Einheit je Segment (Abstand 900 s)
        ts900 = np.concatenate([x[::3] for x in ts_all if len(x)])
        ch = chart.get(csn, np.empty(0))
        mo = mon.get(csn, np.empty(0))
        for G in G_LIST:
          for mr in MAXREF_B_VAR:
            T = S['beziehung_B_chart'][f'G{G}_ref{mr // 60}min']
            for nm, arr in (('', ts), ('_abstand_900', ts900)):
                v = vollstaendig(arr, ch, r, d, G * 60, 300, mr)
                if nm == '':
                    T['einheiten_gesamt'] += int(v.sum())
                    T['ge1'] += int(v.sum() >= 1)
                    T['ge3_alle_einheiten'] += int(v.sum() >= 3)
                else:
                    T['ge3_alle_einheiten_abstand_900'] += int(v.sum() >= 3)
        for g, st in A_GRIDS:
            T = S['beziehung_A_monitor'][f'G{g}_schritt{st}']
            for nm, arr in (('', ts), ('_abstand_900', ts900)):
                v = vollstaendig(arr, mo, r, d, g * 60, st * 60, MAXREF_A)
                if nm == '':
                    T['einheiten_gesamt'] += int(v.sum())
                    T['ge1'] += int(v.sum() >= 1)
                    T['ge3_alle_einheiten'] += int(v.sum() >= 3)
                else:
                    T['ge3_alle_einheiten_abstand_900'] += int(v.sum() >= 3)
    schreibe('nenner', R)
    print(json.dumps(mask_tree(R), indent=1)[:5000])


# --------------------------------------------------------------------------- wellenhash
def modus_wellenhash(stichprobe=None):
    """Prueft .hea/.dat der Ableitung II (nur train/val) gegen SHA256SUMS.txt. Fehlende Dateien (Download laeuft) zaehlen getrennt.
    --stichprobe N: zufaellige Auswahl (Seed 270200), sonst alle. Ausgabe ohne Kennungen."""
    c = verbinden()
    sp = {int(r[0]) for r in c.execute('SELECT CSN FROM sp').fetchall()}
    soll = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and '/II/' in parts[1]:
            pf = parts[1].strip().split('/')          # data/waveforms/<suf>/<csn>/II/<datei>
            if len(pf) >= 6 and pf[3].isdigit() and int(pf[3]) in sp:
                soll[parts[1].strip()] = parts[0].lower()
    keys = sorted(soll)
    if stichprobe:
        rng = np.random.default_rng(270200)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), size=min(stichprobe, len(keys)), replace=False))]
    ok = bad = fehlt = 0
    for k in keys:
        f = ROOT / k
        if not f.exists():
            fehlt += 1
        elif sha(f) == soll[k]:
            ok += 1
        else:
            bad += 1
    R = {'dateien_laut_summen_train_val': len(soll), 'geprueft': len(keys), 'stichprobe': bool(stichprobe), 'gleich': ok, 'abweichend': bad,
         'noch_nicht_auf_platte': fehlt}
    schreibe('wellenhash', R)
    print(json.dumps(mask_tree(R), indent=1))


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    if m == 'hash':
        modus_hash()
    elif m == 'tabellen':
        modus_tabellen()
    elif m == 'koepfe':
        modus_koepfe()
    elif m == 'nenner':
        modus_nenner()
    elif m == 'wellenhash':
        modus_wellenhash(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
        sys.exit(2)
