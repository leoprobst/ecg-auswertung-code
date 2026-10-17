"""S3 MC-MED: Werkzeug fuer die kaliumblinde Referenzannotation der T-Amplitude (Ableitung II, Anwender misst von Hand).

Das Skript enthaelt KEINE Daten. Es baut aus den MC-MED-Wellenformen (nur Train und Val, der Testsplit wird nie gelesen) eine einzelne lokale HTML-Datei
ohne Netzwerkzugriff. Alle Ausgaben mit Daten liegen unter <local path> (nie in <cloud folder>).

Modi (Python: <local path> -I):
  selbsttest            Kunstdaten-Tests der Ziehung (Schichtung, Wiederholungen, Reihenfolge, Determinismus), keine echten Daten
  auswahl [--arbeiter N]  Grundgesamtheit bestimmen, ziehen, AUSWAHL.json und SCHLUESSEL.csv schreiben (verweigert, wenn AUSWAHL.json schon existiert)
  seite                 REFERENZANNOTATION_S3.html aus SCHLUESSEL.csv bauen (prueft den Hash gegen AUSWAHL.json)
  testseite             TESTSEITE.html mit 3 synthetischen Streifen bekannter T-Amplitude (keine Daten)

Auswahlregel (vor jeder Messung festgelegt, siehe AUSWAHL.json):
  Grundgesamtheit = saubere 10-s-Bloecke Ableitung II (s3_signal.block_gut), die als Enden konsekutiver Serum-Kaliumpaare (Abstand 1 bis 12 h, beide
  Ergebniszeiten vorhanden, keine zensierten Werte) im D3-Fenster der Entnahme der naechste saubere Block sind (Blockwahl der Messkette, Funktion
  s3_auswertung_v2.naechster_block), mit Messfenster (60 s) und sicherer QRS-Polaritaet (Medianschlag) an beiden Enden, gleiche Polaritaet an beiden Enden.
  Nur split_random_train und split_random_val. Kalium, Behandlung und Messwerte der Pipeline gehen in die Auswahl NICHT ein.
  Ziehung: ein Block je Person, 90 Bloecke, 63 mit positiver und 27 mit negativer QRS-Hauptrichtung (Seed 255010), 10 zufaellige davon doppelt
  (Wiederholbarkeit), 100 Ansichten in zufaelliger Reihenfolge (Abstand der beiden Vorlagen mindestens 15 Ansichten).
Regeln: keine Kennungen in der Seite; Einzelprozess unter 1,5 GB; keine Loeschbefehle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, hashlib, base64, argparse, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
DATA = Path(str(data_path("s3.mcmed_data_dir")))
ANN = Path(str(data_path("s3.annotation_dir")))
SEED = 255010          # wie CFG['seeds']['annotation'] in s3_auswertung_v2.py
N_BLOECKE = 90
ANTEIL_POS = 0.70
N_WDH = 10
MIN_ABSTAND_WDH = 15   # Ansichten zwischen Original und Wiederholung (mindestens)
VERSION = 'annot-v1'
FS = 500
N_SAMPLES = 10 * FS

_V = None


def v2():
    """Pipeline-Funktionen (Blockwahl, Polaritaet, D3) aus s3_auswertung_v2; Import hat keine Seiteneffekte (kein Datenzugriff)."""
    global _V
    if _V is None:
        import s3_auswertung_v2 as V
        _V = V
    return _V


def sha_datei(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


# =========================================================================== Ziehung (reine Funktion, ohne Daten testbar)
def ziehe_auswahl(pop, seed=SEED, n=N_BLOECKE, anteil_pos=ANTEIL_POS, n_wdh=N_WDH, min_abstand=MIN_ABSTAND_WDH):
    """pop: DataFrame mit MRN, CSN, split, t_order, c, pol (+1/-1), je Zeile ein zulaessiger Block. Ein Block je Person (zufaellig), danach geschichtet nach pol
    (n_pos = round(anteil_pos * n), Rest negativ), dann n_wdh der gezogenen Bloecke zufaellig doppelt, Reihenfolge zufaellig mit Mindestabstand min_abstand zwischen
    Original und Wiederholung. Rueckgabe: DataFrame der Ansichten (ansicht_nr 1..n+n_wdh, wiederholung_von = ansicht_nr des Originals oder 0)."""
    rng = np.random.default_rng(seed)
    p = pop.sort_values(['MRN', 'CSN', 't_order']).reset_index(drop=True)
    # 1. ein Block je Person
    grp = p.groupby('MRN', sort=True).indices
    personen = sorted(grp)
    wahl = []
    for m in personen:
        idx = grp[m]
        wahl.append(int(idx[int(rng.integers(0, len(idx)))]))
    q = p.iloc[wahl].reset_index(drop=True)
    # 2. Schichten ziehen
    n_pos = int(round(anteil_pos * n))
    n_neg = n - n_pos
    perm = rng.permutation(len(q))
    pos = [i for i in perm if q.pol.iloc[i] > 0][:n_pos]
    neg = [i for i in perm if q.pol.iloc[i] < 0][:n_neg]
    if len(pos) < n_pos or len(neg) < n_neg:
        raise ValueError(f'zu wenige Personen je Schicht: positiv {len(pos)} von {n_pos}, negativ {len(neg)} von {n_neg}')
    s = q.iloc[pos + neg].reset_index(drop=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    # 3. Wiederholungen
    wdh = sorted(int(x) for x in rng.choice(len(s), size=n_wdh, replace=False))
    items = [(i, 0) for i in range(len(s))] + [(i, 1) for i in wdh]     # (Index in s, ist_wiederholung)
    # 4. Reihenfolge mit Mindestabstand (Zurueckweisungsverfahren, fester Zufallsstrom)
    for _ in range(100000):
        order = [items[j] for j in rng.permutation(len(items))]
        pos_ori = {i: k for k, (i, w) in enumerate(order) if w == 0}
        pos_w = {i: k for k, (i, w) in enumerate(order) if w == 1}
        if all(abs(pos_w[i] - pos_ori[i]) >= min_abstand for i in wdh):
            break
    else:
        raise ValueError('keine Reihenfolge mit Mindestabstand gefunden')
    rows = []
    nr_ori = {}
    for k, (i, w) in enumerate(order):
        r = s.iloc[i].to_dict()
        r['ansicht_nr'] = k + 1
        if w == 0:
            nr_ori[i] = k + 1
        r['_i'], r['_w'] = i, w
        rows.append(r)
    out = pd.DataFrame(rows)
    out['wiederholung_von'] = [nr_ori[int(i)] if int(w) == 1 else 0 for i, w in zip(out._i, out._w)]
    return out.drop(columns=['_i', '_w'])


def auswahl_hash(views):
    """Hash der Auswahl: geordnete Liste (ansicht_nr | CSN | Blockmitte | Wiederholung von)."""
    zeilen = [f'{int(r.ansicht_nr)}|{int(r.CSN)}|{float(r.c):.3f}|{int(r.wiederholung_von)}' for r in views.itertuples()]
    return sha_text('\n'.join(zeilen))


# =========================================================================== Grundgesamtheit (Daten)
def lade_entnahmen(log):
    """Serum-Kaliumentnahmen der Besuche in Train und Val mit Ableitung-II-Wellenform. Der Testsplit wird nicht gelesen. Rueckgabe DataFrame."""
    import duckdb
    sp = []
    for name in ('train', 'val'):
        p = DATA / f'split_random_{name}.csv'
        assert 'test' not in p.name
        t = pd.read_csv(p, header=None)[0].astype('int64')
        sp.append(pd.DataFrame({'CSN': t, 'split': name}))
    sp = pd.concat(sp, ignore_index=True)
    ws = pd.read_csv(DATA / 'waveform_summary.csv')
    ii = set(ws[ws.Type == 'II'].CSN.astype('int64'))
    sp = sp[sp.CSN.isin(ii)].drop_duplicates('CSN').reset_index(drop=True)
    log(f'Besuche in Train und Val mit Ableitung II: {len(sp)}')
    c = duckdb.connect()
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.register('sp_df', sp)
    c.execute('CREATE TABLE sp AS SELECT CSN, split FROM sp_df')
    allv = 'all_varchar=true,strict_mode=false,null_padding=true,ignore_errors=true'
    c.execute(f"""CREATE TABLE visit AS SELECT CAST(v.CSN AS BIGINT) CSN, CAST(v.MRN AS BIGINT) MRN, sp.split
        FROM read_csv('{(DATA / 'visits.csv').as_posix()}', header=true, {allv}) v JOIN sp ON sp.CSN = try_cast(v.CSN AS BIGINT)""")
    c.execute(f"""CREATE TABLE labs AS SELECT try_cast(l.CSN AS BIGINT) CSN, epoch_us(try_cast(l.Order_time AS TIMESTAMP))/1e6 t_order,
        epoch_us(try_cast(l.Result_time AS TIMESTAMP))/1e6 t_result, l.Component_value cval
        FROM read_csv('{(DATA / 'labs.csv').as_posix()}', header=true, {allv}) l JOIN sp ON sp.CSN = try_cast(l.CSN AS BIGINT)
        WHERE l.Component_name = 'POTASSIUM' AND l.Component_units = 'mmol/L'""")
    D = c.execute("""SELECT k.CSN, v.MRN, v.split, k.t_order, k.t_result, k.k, k.nk, k.cens FROM
        (SELECT CSN, t_order, min(t_result) t_result, avg(k) k, count(DISTINCT k) nk, coalesce(bool_or(cens), false) cens FROM
          (SELECT CSN, t_order, t_result, try_cast(regexp_replace(cval,'^[<>]','') AS DOUBLE) k, (cval LIKE '>%' OR cval LIKE '<%') cens FROM labs)
         WHERE k IS NOT NULL AND t_order IS NOT NULL GROUP BY CSN, t_order) k JOIN visit v USING(CSN)""").df()
    c.close()
    log(f'Serum-Kaliumentnahmen: {len(D)} in {D.CSN.nunique()} Besuchen')
    return D


def paare_aus_entnahmen(D, log):
    """Paare wie in der Spezifikation (konsekutiv, 1 bis 12 h), Entnahmen mit widerspruechlichen Werten entfallen (nk > 1). Zensierte Werte bleiben bei der Paarbildung
    erhalten (Konsekutivitaet) und entfernen das Paar danach. Rueckgabe (Paare, Zaehlung)."""
    V = v2()
    d = D[D.nk == 1].copy()
    d['cens'] = d.cens.astype(bool)
    for col in ('age', 'sex', 'na', 'hco3', 'glc'):
        d[col] = np.nan
    P = V.paare_bauen(d)
    z = {'paare_gesamt': int(len(P))}
    P = P[~P.ohne_ergebnis]
    z['paare_mit_ergebniszeiten'] = int(len(P))
    P = P[~P.zensiert]
    z['paare_nicht_zensiert'] = int(len(P))
    log(f'Paare: {z}')
    return P.reset_index(drop=True), z


def lies_segmente(csns, log):
    """Ableitung-II-Segmente (nur mV, fs 500) je Besuch aus den WFDB-Koepfen: {CSN: [(seg, fs, n, t0, t1), ...]}."""
    V = v2()
    out = {}
    for k, csn in enumerate(sorted(csns)):
        p = V.WAVE / str(csn)[-3:] / str(csn) / 'II'
        if not p.is_dir():
            continue
        for f in sorted(os.listdir(p)):
            if not f.endswith('.hea'):
                continue
            try:
                ns, fs, n, t0, einheit = V.lies_kopf(p / f)
            except Exception:
                continue
            if str(einheit).lower() != 'mv' or fs != FS:
                continue
            out.setdefault(int(csn), []).append((int(f[:-4].rsplit('_', 1)[1]), float(fs), int(n), float(t0), float(t0 + n / fs)))
        if (k + 1) % 5000 == 0:
            log(f'Koepfe gelesen: {k + 1} von {len(csns)}')
    return out


def _arbeit(a):
    """Eine Entnahme: naechster sauberer 10-s-Block im D3-Fenster (Blockwahl der Messkette) und QRS-Polaritaet aus dem 60-s-Messfenster (wie messe_fenster)."""
    csn, t_order, lo, hi, segs = a
    V = v2()
    out = {'CSN': csn, 't_order': t_order, 'blk': 0, 'c': np.nan, 'seg': -1, 'abst': np.nan, 'fenster': 0, 'pol': np.nan, 'pol_unsicher': 1, 'fehler': ''}
    try:
        b = V.naechster_block(csn, segs, t_order, lo, hi)
        if b is None:
            return out
        d, c, seg = b
        sn, fs, n, t0, t1 = seg
        out.update(blk=1, c=float(c), seg=int(sn), abst=float(c - t_order))
        fe = V.messfenster_um(c, t0, t1)
        if fe is None:
            return out
        r = V.lies_roh(csn, seg, fe[0] - 4.0, fe[1] + 4.0)
        if r is None:
            return out
        ts_, x = r
        i0, i1 = int(round((fe[0] - ts_) * FS)), int(round((fe[1] - ts_) * FS))
        if i0 < 0 or i1 > len(x) or i1 - i0 < 20 * FS:
            return out
        out['fenster'] = 1
        pol, uns, nb = V.polaritaet(np.asarray(x[i0:i1], float), FS)
        if pol is not None:
            out['pol'], out['pol_unsicher'] = int(pol), int(uns)
    except Exception as e:
        out['fehler'] = type(e).__name__
    return out


def bestimme_grundgesamtheit(nw, log):
    """Gesamte Kette bis zur Liste zulaessiger Bloecke. Rueckgabe (pop, zaehlung). Zwischenstand <local path> (wiederverwendbar)."""
    V = v2()
    z = {}
    D = lade_entnahmen(log)
    z['entnahmen_serum'] = int(len(D))
    P, zp = paare_aus_entnahmen(D, log)
    z.update(zp)
    csns = set(P.CSN.astype('int64'))
    segs = lies_segmente(csns, log)
    z['besuche_mit_paaren'] = int(len(csns))
    z['besuche_mit_segment_mv'] = int(len(segs))
    # Entnahmen an Paarenden mit zeitlicher Abdeckung im D3-Fenster (>= 10 s Signal)
    ends = {}
    for r in P.itertuples():
        ends[(int(r.CSN), float(r.ta))] = float(r.ra)
        ends[(int(r.CSN), float(r.tb))] = float(r.rb)
    arbeit = []
    for (csn, t_order), t_res in sorted(ends.items()):
        g = V.d3_grenzen(t_order, t_res)
        if g is None or csn not in segs:
            continue
        ov = [min(s[4], g[1]) - max(s[3], g[0]) for s in segs[csn]]
        if ov and max(ov) >= 10.0:
            arbeit.append((csn, t_order, g[0], g[1], segs[csn]))
    z['entnahmen_an_paarenden'] = int(len(ends))
    z['entnahmen_mit_abdeckung'] = int(len(arbeit))
    log(f'zu pruefende Entnahmen: {len(arbeit)} von {len(ends)}')
    zw = ANN / '_entnahmebloecke.csv'
    if zw.exists():
        E = pd.read_csv(zw)
        log(f'Zwischenstand gelesen: {len(E)} Zeilen')
    else:
        res = []
        t0 = time.time()
        if nw > 1:
            from multiprocessing import Pool
            with Pool(nw) as pool:
                for k, o in enumerate(pool.imap(_arbeit, arbeit, chunksize=4)):
                    res.append(o)
                    if (k + 1) % 200 == 0:
                        log(f'Bloecke gesucht: {k + 1} von {len(arbeit)} ({time.time() - t0:.0f} s)')
        else:
            for k, a in enumerate(arbeit):
                res.append(_arbeit(a))
                if (k + 1) % 200 == 0:
                    log(f'Bloecke gesucht: {k + 1} von {len(arbeit)} ({time.time() - t0:.0f} s)')
        E = pd.DataFrame(res)
        E.to_csv(zw, index=False)
    z['mit_sauberem_block'] = int(E.blk.sum())
    z['mit_messfenster'] = int(E.fenster.sum())
    z['mit_bestimmter_polaritaet'] = int(E.pol.notna().sum())
    z['polaritaet_unsicher'] = int(((E.pol.notna()) & (E.pol_unsicher == 1)).sum())
    Ek = E.set_index(['CSN', 't_order'])
    sicher = E[(E.fenster == 1) & E.pol.notna() & (E.pol_unsicher == 0)].set_index(['CSN', 't_order'])
    elig = []
    n_beide = n_pol = 0
    for r in P.itertuples():
        ka, kb = (int(r.CSN), float(r.ta)), (int(r.CSN), float(r.tb))
        if ka in Ek.index and kb in Ek.index and Ek.loc[ka, 'blk'] == 1 and Ek.loc[kb, 'blk'] == 1:
            n_beide += 1
            if ka in sicher.index and kb in sicher.index and sicher.loc[ka, 'pol'] == sicher.loc[kb, 'pol']:
                n_pol += 1
                elig.append((ka, int(r.MRN), r.split))
                elig.append((kb, int(r.MRN), r.split))
    z['paare_sauberer_block_beide_enden'] = int(n_beide)
    z['paare_zulaessig_polaritaet_gleich_sicher'] = int(n_pol)
    rows, seen = [], set()
    for (key, mrn, spl) in elig:
        if key in seen:
            continue
        seen.add(key)
        e = sicher.loc[key]
        rows.append({'MRN': mrn, 'CSN': key[0], 'split': spl, 't_order': key[1], 'c': float(e.c), 'seg': int(e.seg), 'pol': int(e.pol), 'abst': float(e.abst)})
    pop = pd.DataFrame(rows).sort_values(['MRN', 'CSN', 't_order']).reset_index(drop=True)
    z['bloecke_grundgesamtheit'] = int(len(pop))
    z['personen_grundgesamtheit'] = int(pop.MRN.nunique())
    z['bloecke_positiv'] = int((pop.pol > 0).sum())
    z['bloecke_negativ'] = int((pop.pol < 0).sum())
    z['grundgesamtheit_hash'] = sha_text('\n'.join(f'{int(r.CSN)}|{r.t_order:.3f}|{r.c:.3f}|{int(r.pol)}' for r in pop.itertuples()))
    return pop, z


# =========================================================================== Signale lesen
def lies_block(csn, seg_nr, c):
    """10-s-Block (Blockmitte c, Rohsignal Ableitung II in mV) als 5000 Abtastwerte. Rueckgabe ndarray."""
    V = v2()
    p = V.WAVE / str(csn)[-3:] / str(csn) / 'II' / f'{csn}_{seg_nr}'
    ns, fs, n, t0, einheit = V.lies_kopf(str(p) + '.hea')
    assert str(einheit).lower() == 'mv' and fs == FS
    a = int(round((c - 5.0 - t0) * fs))
    assert 0 <= a and a + N_SAMPLES <= n
    import wfdb
    rec = wfdb.rdrecord(str(p), sampfrom=a, sampto=a + N_SAMPLES, physical=True, channels=[0])
    x = rec.p_signal[:, 0].astype(float)
    assert len(x) == N_SAMPLES and np.isfinite(x).all()
    return x


def kodiere(x):
    """mV -> ganze Mikrovolt (int16, little endian), Base64; Pruefsumme sum(u * ((i mod 7) + 1))."""
    u = np.clip(np.round(np.asarray(x, float) * 1000.0), -32768, 32767).astype('<i2')
    crc = int(np.sum(u.astype(np.int64) * (np.arange(len(u)) % 7 + 1)))
    return {'n': int(len(u)), 'crc': crc, 'b64': base64.b64encode(u.tobytes()).decode('ascii')}


# =========================================================================== Ausgabe: AUSWAHL, SCHLUESSEL
def schreibe_auswahl(pop, z, views, ausgabe=ANN):
    ausgabe = Path(ausgabe)
    ausgabe.mkdir(parents=True, exist_ok=True)
    pa, ps = ausgabe / 'AUSWAHL.json', ausgabe / 'SCHLUESSEL.csv'
    if pa.exists():
        raise SystemExit('AUSWAHL.json existiert bereits; die Auswahl wird nicht neu gezogen: ' + str(pa))
    views = views.copy()
    views['pol_auto'] = views.pol.map({1: 'positiv', -1: 'negativ'})
    spalten = ['ansicht_nr', 'wiederholung_von', 'CSN', 'MRN', 'split', 't_order', 'c', 'seg', 'pol_auto', 'abst']
    views[spalten].rename(columns={'c': 't_mitte', 'abst': 'abstand_zur_auftragszeit_s'}).to_csv(ps, index=False, float_format='%.3f')
    ah = auswahl_hash(views)
    ori = views[views.wiederholung_von == 0]
    info = {
        'art': 'Auswahl der Bloecke fuer die kaliumblinde Referenzannotation der T-Amplitude (S3 MC-MED)',
        'erstellt': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'version': VERSION,
        'seed': SEED,
        'regel': [
            'Grundgesamtheit: saubere 10-s-Bloecke Ableitung II (s3_signal.block_gut), je Entnahme der im D3-Fenster naechste saubere Block (Blockwahl der Messkette, '
            's3_auswertung_v2.naechster_block), Entnahme ist Ende eines konsekutiven Serum-Kaliumpaares (Abstand 1 bis 12 h, beide Ergebniszeiten vorhanden, keine '
            'zensierten Werte), 60-s-Messfenster vorhanden, sichere QRS-Polaritaet (Medianschlag, Verhaeltnis < 0,70), gleiche Polaritaet an beiden Paarenden.',
            'Nur split_random_train und split_random_val; der Testsplit wurde nicht gelesen.',
            'Kalium, Behandlung, Alter, Geschlecht und Messwerte der Pipeline gehen nicht in die Auswahl ein (Kalium nur ueber die Existenz von Paaren).',
            f'Ziehung: ein zufaelliger Block je Person (MRN), danach geschichtet nach automatisch bestimmter Polaritaet: {int(round(ANTEIL_POS * N_BLOECKE))} positiv, '
            f'{N_BLOECKE - int(round(ANTEIL_POS * N_BLOECKE))} negativ; numpy default_rng(Seed).',
            f'{N_WDH} der {N_BLOECKE} Bloecke werden ein zweites Mal vorgelegt (Wiederholbarkeit); 100 Ansichten in zufaelliger Reihenfolge, Wiederholung mindestens '
            f'{MIN_ABSTAND_WDH} Ansichten Abstand zwischen beiden Vorlagen (die zweite Vorlage traegt in wiederholung_von die Ansichtsnummer der ersten; in der Reihenfolge kann jede zuerst kommen).',
        ],
        'grundgesamtheit': {k: z[k] for k in z},
        'ziehung': {'bloecke': int(len(ori)), 'wiederholungen': int((views.wiederholung_von > 0).sum()), 'ansichten': int(len(views)),
                    'positiv': int((ori.pol > 0).sum()), 'negativ': int((ori.pol < 0).sum()), 'personen': int(ori.MRN.nunique()),
                    'train': int((ori.split == 'train').sum()), 'val': int((ori.split == 'val').sum())},
        'auswahl_hash': ah,
        'auswahl_hash_bildung': 'SHA-256 ueber die Zeilen "ansicht_nr|CSN|Blockmitte (3 Nachkommastellen)|wiederholung_von", durch Zeilenumbruch getrennt, in Ansichtsreihenfolge',
        'schluessel_datei_sha256': sha_datei(ps),
        'software_sha256': {p.name: sha_datei(p) for p in [Path(__file__), HIER / 's3_auswertung_v2.py', HIER / 's3_signal.py']},
    }
    pa.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    return info


# =========================================================================== Seite
HTML_VORLAGE = r'''<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Referenzannotation T-Amplitude</title>
<style>
:root{--bg:#eef1f4;--panel:#ffffff;--ink:#1d2733;--muted:#5b6876;--line:#d3dae1;--accent:#2a5d8f;--accent2:#e8f0f8;--warn:#7a5200;--warnbg:#fff4d6;--ok:#1f6b47;--okbg:#e5f4ec;--bad:#9b2c2c;--badbg:#fbe9e9}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:15px/1.4 "Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:8px 24px;align-items:center}
header h1{font-size:16px;font-weight:600;margin:0}
.prog{display:flex;align-items:center;gap:12px;flex:1 1 320px;min-width:260px}
.bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--accent)}
main{padding:10px 16px 24px;max-width:2000px;margin:0 auto}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0 8px}
.toolbar .sp{flex:1}
button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid #aeb8c2;border-radius:6px;padding:5px 12px;cursor:pointer}
button:hover{background:var(--accent2)}
button:disabled{opacity:.45;cursor:default;background:var(--panel)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{background:#21496f}
#hint{background:var(--accent2);border:1px solid #c5d6e8;border-radius:6px;padding:8px 12px;font-size:16px;margin:0 0 8px}
#hint.warn{background:var(--warnbg);border-color:#e6cf8f;color:var(--warn)}
#stripwrap{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:auto;max-height:68vh;position:relative}
#stage{position:relative;line-height:0}
#stage canvas{display:block;position:absolute;left:0;top:0}
#ov{cursor:crosshair}
#readout{color:var(--muted);font-size:13px;height:20px;margin:4px 2px 8px;font-variant-numeric:tabular-nums}
.panel{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 12px;border-left:5px solid var(--c,#999)}
.card.cur{box-shadow:0 0 0 2px var(--c,#999) inset}
.card h3{margin:0 0 2px;font-size:15px}
.card .st{color:var(--muted);font-size:13px;min-height:19px}
.card .val{font-size:18px;font-variant-numeric:tabular-nums;min-height:26px}
.card .val small{font-size:13px;color:var(--muted)}
.card .btns{margin-top:4px}
.row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 12px;flex:1 1 300px}
.box h4{margin:0 0 4px;font-size:13px;color:var(--muted);font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.box label{display:inline-block;margin-right:14px;cursor:pointer}
.box input[type=radio],.box input[type=checkbox]{margin-right:5px}
textarea{width:100%;height:54px;font:inherit;border:1px solid #aeb8c2;border-radius:6px;padding:6px;resize:vertical}
.nav{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.nav .sp{flex:1}
.mean{font-size:15px;margin:0 0 10px}
.banner{border-radius:6px;padding:8px 12px;margin:0 0 8px;font-size:14px}
.banner.warn{background:var(--warnbg);color:var(--warn);border:1px solid #e6cf8f}
.banner.bad{background:var(--badbg);color:var(--bad);border:1px solid #e3b4b4}
.banner.ok{background:var(--okbg);color:var(--ok);border:1px solid #b9dcc9}
.hidden{display:none}
.legend{color:var(--muted);font-size:13px;margin:8px 2px}
@media (max-width:900px){.panel{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1 id="titel">Referenzannotation T-Amplitude</h1>
  <div class="prog">
    <div>Ansicht <b id="posnr">1</b> von <span id="ntotal"></span></div>
    <div class="bar"><i id="barfill"></i></div>
    <div><span id="ndone">0</span> abgeschlossen</div>
  </div>
</header>
<main>
  <div id="banners"></div>
  <div class="toolbar">
    <button id="bUndo" title="Letzten Klick zurücknehmen (Strg+Z oder Rücktaste)">Rückgängig</button>
    <span style="width:8px"></span>
    <button id="bZm" title="Verkleinern (-)">Zoom -</button>
    <button id="bZp" title="Vergrößern (+)">Zoom +</button>
    <button id="bZf" title="An Fensterbreite anpassen">Einpassen</button>
    <span id="zoomtxt" style="color:var(--muted);font-size:13px"></span>
    <span class="sp"></span>
    <button id="bImp" title="Eine zuvor exportierte CSV-Datei einlesen">Sicherung einlesen</button>
    <button id="bExp" class="primary" title="Alle Ergebnisse als CSV-Datei speichern">Ergebnisse exportieren (CSV)</button>
    <input type="file" id="fImp" accept=".csv,text/csv" class="hidden">
  </div>
  <div id="hint"></div>
  <div id="stripwrap"><div id="stage"><canvas id="cv"></canvas><canvas id="ov"></canvas></div></div>
  <div id="readout"></div>
  <div class="panel" id="cards"></div>
  <p class="mean" id="mean"></p>
  <div class="row">
    <div class="box" style="flex:0 1 340px">
      <h4>QRS-Hauptrichtung (Ableitung II)</h4>
      <label><input type="radio" name="qrs" value="positiv">positiv</label>
      <label><input type="radio" name="qrs" value="negativ">negativ</label>
      <label><input type="radio" name="qrs" value="isoelektrisch">isoelektrisch</label>
    </div>
    <div class="box" style="flex:0 1 460px">
      <h4>Beurteilbarkeit</h4>
      <label><input type="checkbox" id="nb">Nicht beurteilbar</label>
      <div id="grund" class="hidden" style="margin-top:4px">
        <span style="color:var(--muted)">Grund:</span>
        <label><input type="radio" name="grund" value="artefakt">Artefakt</label>
        <label><input type="radio" name="grund" value="rhythmus">Rhythmus</label>
        <label><input type="radio" name="grund" value="flache_T">flache T</label>
        <label><input type="radio" name="grund" value="sonstiges">Sonstiges</label>
      </div>
    </div>
    <div class="box" style="flex:2 1 300px">
      <h4>Bemerkung</h4>
      <textarea id="bem" placeholder="freies Feld, optional"></textarea>
    </div>
  </div>
  <div class="nav">
    <button id="bBack">Zurück</button>
    <button id="bNext" class="primary">Weiter</button>
    <button id="bOpen">Nächste offene Ansicht</button>
    <span class="sp"></span>
    <span id="storetxt" style="color:var(--muted);font-size:13px"></span>
  </div>
  <p class="legend">Maßstab 25 mm/s und 10 mm/mV, Kalibrierzacke 1 mV am Anfang. Pro Ansicht drei aufeinanderfolgende, gut beurteilbare Schläge messen: zuerst die isoelektrische Linie (PR- oder TP-Segment), dann die T-Spitze (bei negativem T das tiefste Ende). Der Zwischenstand wird laufend im Browser gesichert. Zum Abschluss exportieren.</p>
</main>
<script>
(function(){
'use strict';
var GEO={PXMM:96/25.4,LEAD:12,MMS:25,MMMV:10,FS:500,SEC:10,RIGHT:8};
GEO.W=GEO.LEAD+GEO.MMS*GEO.SEC+GEO.RIGHT;
var HASH='__HASH__',KEY='__KEY__',TESTMODUS=__TESTMODUS__,VERSION='__VERSION__',TITEL='__TITEL__';
var ANS=__DATEN__;
var N=ANS.length;
var COL=['#1c5fa6','#0d7a5f','#8a3fa0'];
function $(id){return document.getElementById(id);}
function iso(){return new Date().toISOString();}
function blank(){return {bt:null,bv:null,pt:null,pv:null};}
function fmt(x,d){var s=x.toFixed(d).replace('.',',');return s;}
function sgn(x,d){return (x>=0?'+':'')+fmt(x,d);}

/* ---------- Zustand und Sicherung ---------- */
var storageOK=true;
function load(){
  try{var s=localStorage.getItem(KEY);if(s){var o=JSON.parse(s);if(o&&o.hash===HASH&&o.recs)return o;}}catch(e){storageOK=false;}
  return null;
}
var S=load()||{v:1,hash:HASH,pos:0,recs:{},zoomMode:'fit',zoom:1.5,lastExport:null,doneAtExport:0};
function save(){
  try{localStorage.setItem(KEY,JSON.stringify(S));storageOK=true;}catch(e){storageOK=false;}
  var t=$('storetxt');
  t.textContent=storageOK?'Zwischenstand im Browser gesichert':'Automatisches Sichern im Browser nicht möglich';
}
function rec(nr){
  var k=String(nr);
  if(!S.recs[k])S.recs[k]={b:[blank(),blank(),blank()],log:[],cur:0,qrs:'',nb:false,grund:'',bem:'',t0:null,t1:null,secs:0};
  return S.recs[k];
}
function touch(r){var n=iso();if(!r.t0)r.t0=n;r.t1=n;}
var lastAct=performance.now();
function tick(r){var n=performance.now();r.secs=(r.secs||0)+Math.min((n-lastAct)/1000,30);lastAct=n;}
function beatDone(b){return b.bt!==null&&b.pt!==null;}
function nDoneBeats(r){var n=0;for(var i=0;i<3;i++)if(beatDone(r.b[i]))n++;return n;}
function status(r){
  if(r.nb&&r.grund)return 'nicht_beurteilbar';
  if(nDoneBeats(r)===3&&r.qrs)return 'gemessen';
  return 'offen';
}
function ampOf(b){return beatDone(b)?(b.pv-b.bv):null;}
function meanAmp(r){var s=0,n=0;for(var i=0;i<3;i++){var a=ampOf(r.b[i]);if(a!==null){s+=a;n++;}}return n?s/n:null;}
function countDone(){var n=0;for(var i=0;i<N;i++){var r=S.recs[String(ANS[i].nr)];if(r&&status(r)!=='offen')n++;}return n;}
function firstIncomplete(r){for(var i=0;i<3;i++)if(!beatDone(r.b[i]))return i;return 3;}

/* ---------- Daten und Geometrie ---------- */
var cache={};
function decode(a){
  var bin=atob(a.b64),n=bin.length>>1,v=new Float64Array(n),crc=0;
  for(var i=0;i<n;i++){var u=bin.charCodeAt(2*i)|(bin.charCodeAt(2*i+1)<<8);if(u>=32768)u-=65536;v[i]=u/1000;crc+=u*((i%7)+1);}
  return {v:v,ok:(n===a.n&&crc===a.crc)};
}
function median(a){var s=Array.prototype.slice.call(a).sort(function(x,y){return x-y;});var m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2;}
function makeLayout(v){
  var vref=median(v),vmin=Infinity,vmax=-Infinity;
  for(var i=0;i<v.length;i++){if(v[i]<vmin)vmin=v[i];if(v[i]>vmax)vmax=v[i];}
  var upMV=Math.max(vmax,vref+1.0)-vref+0.3,dnMV=vref-Math.min(vmin,vref)+0.3;
  var up=Math.max(25,Math.ceil(upMV*GEO.MMMV/5)*5),dn=Math.max(15,Math.ceil(dnMV*GEO.MMMV/5)*5),foot=5;
  return {vref:vref,up:up,dn:dn,foot:foot,H:up+dn+foot,Y0:up};
}
function getView(i){
  if(!cache[i]){var d=decode(ANS[i]);cache[i]={v:d.v,ok:d.ok,L:makeLayout(d.v)};}
  return cache[i];
}
function mmToTV(L,xmm,ymm){return {t:(xmm-GEO.LEAD)/GEO.MMS,v:L.vref+(L.Y0-ymm)/GEO.MMMV};}
function tvToMM(L,t,v){return {x:GEO.LEAD+GEO.MMS*t,y:L.Y0-GEO.MMMV*(v-L.vref)};}

/* ---------- Zoom ---------- */
var STEPS=[0.5,0.6,0.8,1,1.25,1.5,1.75,2,2.5,3,4];
function fitZoom(){
  var avail=$("stripwrap").clientWidth-22;
  var z=avail/(GEO.W*GEO.PXMM);
  return Math.max(0.5,Math.min(4,Math.floor(z*100)/100));
}
function curZoom(){return S.zoomMode==='fit'?fitZoom():S.zoom;}
function stepZoom(dir){
  var z=curZoom(),i;
  if(dir>0){for(i=0;i<STEPS.length;i++)if(STEPS[i]>z+0.005)break;z=STEPS[Math.min(i,STEPS.length-1)];}
  else{for(i=STEPS.length-1;i>=0;i--)if(STEPS[i]<z-0.005)break;z=STEPS[Math.max(i,0)];}
  S.zoomMode='fix';S.zoom=z;save();layoutCanvas();
}

/* ---------- Zeichnen ---------- */
var cv=$('cv'),ov=$('ov'),ctxB=cv.getContext('2d'),ctxO=ov.getContext('2d');
var pxmm=GEO.PXMM,curL=null,curI=0;
function layoutCanvas(){
  var view=getView(curI);curL=view.L;
  var z=curZoom();pxmm=GEO.PXMM*z;
  var cw=GEO.W*pxmm,ch=curL.H*pxmm,dpr=window.devicePixelRatio||1;
  var stage=$('stage');stage.style.width=cw+'px';stage.style.height=ch+'px';
  [cv,ov].forEach(function(c){c.style.width=cw+'px';c.style.height=ch+'px';c.width=Math.round(cw*dpr);c.height=Math.round(ch*dpr);});
  $('zoomtxt').textContent=(S.zoomMode==='fit'?'angepasst, ':'')+'Vergrößerung '+Math.round(z*100)+' %';
  drawBase();drawOverlay(null);
}
function setMM(ctx,c){ctx.setTransform(c.width/GEO.W,0,0,c.height/curL.H,0,0);}
function drawBase(){
  var L=curL,ctx=ctxB,W=GEO.W,H=L.H,v=getView(curI).v,px=1/pxmm;
  setMM(ctx,cv);
  ctx.fillStyle='#fff7f5';ctx.fillRect(0,0,W,H);
  ctx.lineCap='butt';
  ctx.strokeStyle='#f2c8c3';ctx.lineWidth=0.9*px;ctx.beginPath();
  for(var x=0;x<=W+1e-9;x++){ if(x%5!==0){ctx.moveTo(x,0);ctx.lineTo(x,H);} }
  for(var y=0;y<=H+1e-9;y++){ if(y%5!==0){ctx.moveTo(0,y);ctx.lineTo(W,y);} }
  ctx.stroke();
  ctx.strokeStyle='#e08f88';ctx.lineWidth=1.5*px;ctx.beginPath();
  for(x=0;x<=W+1e-9;x+=5){ctx.moveTo(x,0);ctx.lineTo(x,H);}
  for(y=0;y<=H+1e-9;y+=5){ctx.moveTo(0,y);ctx.lineTo(W,y);}
  ctx.stroke();
  /* Kalibrierzacke 1 mV (10 mm), 0,2 s (5 mm) */
  ctx.strokeStyle='#111';ctx.lineWidth=1.5*px;ctx.lineJoin='miter';ctx.beginPath();
  ctx.moveTo(1.5,L.Y0);ctx.lineTo(3,L.Y0);ctx.lineTo(3,L.Y0-GEO.MMMV);ctx.lineTo(8,L.Y0-GEO.MMMV);ctx.lineTo(8,L.Y0);ctx.lineTo(10,L.Y0);ctx.stroke();
  ctx.fillStyle='#111';ctx.font='3px "Segoe UI",Arial,sans-serif';ctx.textBaseline='alphabetic';
  ctx.fillText('1 mV',3.2,L.Y0-GEO.MMMV-1.2);
  /* Spur */
  ctx.strokeStyle='#0b0b0b';ctx.lineWidth=1.4*px;ctx.lineJoin='round';ctx.beginPath();
  var step=GEO.MMS/GEO.FS;
  for(var i=0;i<v.length;i++){var X=GEO.LEAD+i*step,Y=L.Y0-GEO.MMMV*(v[i]-L.vref);if(i===0)ctx.moveTo(X,Y);else ctx.lineTo(X,Y);}
  ctx.stroke();
  /* Zeitmarken */
  ctx.fillStyle='#444';ctx.strokeStyle='#444';ctx.lineWidth=1.2*px;ctx.font='3px "Segoe UI",Arial,sans-serif';
  var yb=L.H-L.foot;
  for(var s=0;s<=GEO.SEC;s++){var xs=GEO.LEAD+GEO.MMS*s;ctx.beginPath();ctx.moveTo(xs,yb);ctx.lineTo(xs,yb+1.6);ctx.stroke();ctx.fillText(s+' s',xs+0.6,H-1.1);}
  ctx.font='2.6px "Segoe UI",Arial,sans-serif';ctx.fillText('25 mm/s, 10 mm/mV',1.2,3.4);
}
function evToMM(e){
  var r=ov.getBoundingClientRect();
  return {x:(e.clientX-r.left)/pxmm,y:(e.clientY-r.top)/pxmm};
}
function drawOverlay(cursor){
  var ctx=ctxO,L=curL,px=1/pxmm,r=rec(ANS[curI].nr);
  setMM(ctx,ov);ctx.clearRect(0,0,GEO.W,L.H);
  for(var k=0;k<3;k++){
    var b=r.b[k],c=COL[k],act=(r.cur===k);
    if(b.bt===null)continue;
    var pb=tvToMM(L,b.bt,b.bv),pp=b.pt!==null?tvToMM(L,b.pt,b.pv):null;
    ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=(act?1.8:1.4)*px;
    var x0=pb.x-6,x1=pb.x+6;
    if(pp){x0=Math.min(pb.x,pp.x)-5;x1=Math.max(pb.x,pp.x)+5;}
    ctx.setLineDash([2.5,1.5]);ctx.beginPath();ctx.moveTo(x0,pb.y);ctx.lineTo(x1,pb.y);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(pb.x-1.4,pb.y-1.4);ctx.lineTo(pb.x+1.4,pb.y+1.4);ctx.moveTo(pb.x-1.4,pb.y+1.4);ctx.lineTo(pb.x+1.4,pb.y-1.4);ctx.stroke();
    if(pp){
      ctx.beginPath();ctx.moveTo(pp.x,pb.y);ctx.lineTo(pp.x,pp.y);ctx.stroke();
      ctx.beginPath();ctx.arc(pp.x,pp.y,1.3,0,2*Math.PI);ctx.stroke();
      var a=b.pv-b.bv,up=pp.y<pb.y;
      ctx.font='bold 3.4px "Segoe UI",Arial,sans-serif';ctx.textBaseline='alphabetic';
      ctx.fillText((k+1)+': '+sgn(a,2)+' mV',pp.x+2,up?pp.y-1.2:pp.y+4.4);
    }else{
      ctx.font='bold 3.4px "Segoe UI",Arial,sans-serif';ctx.fillText(String(k+1),pb.x+2,pb.y-1.2);
    }
  }
  if(cursor){
    ctx.strokeStyle='rgba(30,40,60,0.45)';ctx.lineWidth=0.8*px;ctx.setLineDash([1.2,1.2]);
    ctx.beginPath();ctx.moveTo(cursor.x,0);ctx.lineTo(cursor.x,L.H);ctx.moveTo(0,cursor.y);ctx.lineTo(GEO.W,cursor.y);ctx.stroke();ctx.setLineDash([]);
  }
}

/* ---------- Bedienung ---------- */
var lastClick=0;
function setHint(msg,warn){var h=$('hint');h.textContent=msg;h.className=warn?'warn':'';}
function advance(r){r.cur=firstIncomplete(r);}
function onClick(e){
  var now=performance.now();if(now-lastClick<180)return;lastClick=now;
  var nr=ANS[curI].nr,r=rec(nr),p=evToMM(e),tv=mmToTV(curL,p.x,p.y);
  if(tv.t<0||tv.t>GEO.SEC){setHint('Bitte innerhalb der Aufzeichnung klicken (rechts neben der Kalibrierzacke).',true);return;}
  tick(r);touch(r);
  var k=r.cur;
  if(k>=3){setHint('Alle drei Schläge sind gemessen. Zum Ändern bei dem Schlag "Neu messen" wählen.',true);save();return;}
  var b=r.b[k],warn='';
  if(b.bt===null){b.bt=tv.t;b.bv=tv.v;r.log.push([k,'b']);}
  else{
    b.pt=tv.t;b.pv=tv.v;r.log.push([k,'p']);
    for(var j=0;j<3;j++){if(j!==k&&beatDone(r.b[j])&&Math.abs(r.b[j].pt-b.pt)<0.3)warn='Diese T-Spitze liegt sehr nah an der von Schlag '+(j+1)+'.';}
    advance(r);
  }
  save();render(warn);
}
function onMove(e){
  var p=evToMM(e),tv=mmToTV(curL,p.x,p.y),r=rec(ANS[curI].nr),txt='';
  if(tv.t>=0&&tv.t<=GEO.SEC){
    txt='Zeit '+fmt(tv.t,2)+' s';
    var b=r.b[Math.min(r.cur,2)];
    if(r.cur<3&&b&&b.bt!==null&&b.pt===null){var d=tv.v-b.bv;txt+='   |   Abstand zur Basislinie '+sgn(d,2)+' mV ('+sgn(d*GEO.MMMV,1)+' mm)';}
  }
  $('readout').textContent=txt;
  drawOverlay(p);
}
function undo(){
  var r=rec(ANS[curI].nr);if(!r.log.length)return;
  tick(r);touch(r);
  var l=r.log.pop(),b=r.b[l[0]];
  if(l[1]==='p'){b.pt=null;b.pv=null;}else{b.bt=null;b.bv=null;}
  r.cur=l[0];save();render();
}
function remeasure(k){
  var r=rec(ANS[curI].nr);tick(r);touch(r);
  r.b[k]=blank();r.log=r.log.filter(function(l){return l[0]!==k;});r.cur=k;save();render();
}
function go(i){
  if(i<0||i>=N)return;
  var r=rec(ANS[curI].nr);tick(r);
  curI=i;S.pos=i;rec(ANS[i].nr);save();lastAct=performance.now();
  layoutCanvas();render();
  $('stripwrap').scrollLeft=0;
}
function nextOpen(){
  for(var d=1;d<=N;d++){var i=(curI+d)%N,r=S.recs[String(ANS[i].nr)];if(!r||status(r)==='offen'){go(i);return;}}
  banner('ok','Alle Ansichten sind bearbeitet. Bitte jetzt exportieren.');
}

/* ---------- Anzeige ---------- */
function banner(kind,msg){var b=$('banners');var d=document.createElement('div');d.className='banner '+kind;d.textContent=msg;b.appendChild(d);}
function render(warn){
  var nr=ANS[curI].nr,r=rec(nr),st=status(r),done=countDone();
  $('posnr').textContent=curI+1;$('ntotal').textContent=N;$('ndone').textContent=done;
  $('barfill').style.width=(100*done/N)+'%';
  /* Hinweis */
  var msg='',w=!!warn;
  if(warn)msg=warn;
  else if(r.nb&&r.grund)msg='Als nicht beurteilbar markiert. Messungen sind dann nicht nötig.';
  else if(r.cur<3){
    var b=r.b[r.cur];
    msg='Schlag '+(r.cur+1)+': '+(b.bt===null?'Klicken Sie auf die isoelektrische Linie (PR- oder TP-Segment).':'Klicken Sie auf die T-Spitze (bei negativem T auf das tiefste Ende).');
  }else if(!r.qrs)msg='Alle drei Schläge gemessen. Bitte die QRS-Hauptrichtung wählen.';
  else msg='Ansicht vollständig. Weiter zur nächsten Ansicht.';
  setHint(msg,w);
  /* Karten */
  var cards=$('cards');cards.innerHTML='';
  for(var k=0;k<3;k++){
    var bb=r.b[k],c=document.createElement('div');c.className='card'+(r.cur===k?' cur':'');c.style.setProperty('--c',COL[k]);
    var stt=beatDone(bb)?'gemessen':(bb.bt!==null?'Basislinie gesetzt, T-Spitze fehlt':'noch nicht gemessen');
    var a=ampOf(bb);
    c.innerHTML='<h3>Schlag '+(k+1)+'</h3><div class="st"></div><div class="val"></div><div class="btns"></div>';
    c.querySelector('.st').textContent=stt;
    c.querySelector('.val').innerHTML=a!==null?(sgn(a,2)+' mV <small>('+sgn(a*GEO.MMMV,1)+' mm)</small>'):'';
    var btn=document.createElement('button');btn.textContent='Neu messen';btn.disabled=(bb.bt===null);
    (function(kk){btn.onclick=function(){remeasure(kk);};})(k);
    c.querySelector('.btns').appendChild(btn);
    cards.appendChild(c);
  }
  var m=meanAmp(r),nb=nDoneBeats(r);
  $('mean').textContent=m!==null?('Mittelwert der '+nb+' gemessenen Schläge: '+sgn(m,2)+' mV ('+sgn(m*GEO.MMMV,1)+' mm)'):'';
  /* Auswahlfelder */
  Array.prototype.forEach.call(document.getElementsByName('qrs'),function(x){x.checked=(x.value===r.qrs);});
  $('nb').checked=!!r.nb;$('grund').className=r.nb?'':'hidden';
  Array.prototype.forEach.call(document.getElementsByName('grund'),function(x){x.checked=(x.value===r.grund);});
  if(document.activeElement!==$('bem'))$('bem').value=r.bem||'';
  $('bUndo').disabled=!r.log.length;$('bBack').disabled=(curI===0);$('bNext').disabled=(curI===N-1);
  drawOverlay(null);
  exportHinweis();
  if(!getView(curI).ok&&!$('banners').querySelector('.bad'))banner('bad','Die Daten dieser Ansicht sind beschädigt. Bitte die Seite neu aus der Originaldatei öffnen.');
}
function exportHinweis(){
  var d=countDone();
  var old=document.getElementById('expw');if(old)old.remove();
  if(d===N){var b=document.createElement('div');b.id='expw';b.className='banner ok';b.textContent='Alle Ansichten sind bearbeitet. Bitte jetzt exportieren und die Datei in den Ordner s3\\annotation\\ legen.';$('banners').appendChild(b);}
  else if(d-(S.doneAtExport||0)>=20||(!storageOK&&d>0)){
    var b2=document.createElement('div');b2.id='expw';b2.className='banner warn';
    b2.textContent=storageOK?('Seit dem letzten Export sind '+(d-(S.doneAtExport||0))+' Ansichten dazugekommen. Bitte zwischendurch exportieren.'):'Das Sichern im Browser ist nicht möglich. Bitte häufig exportieren.';
    $('banners').appendChild(b2);
  }
}

/* ---------- CSV ---------- */
var HEAD=['auswahl_hash','ansicht_nr','status','qrs_richtung','grund_nicht_beurteilbar','n_schlaege'];
for(var q=1;q<=3;q++)HEAD.push('s'+q+'_basis_t_s','s'+q+'_basis_mv','s'+q+'_spitze_t_s','s'+q+'_spitze_mv','s'+q+'_amp_mv','s'+q+'_amp_mm');
HEAD.push('mittel_amp_mv','mittel_amp_mm','bemerkung','bearbeitungszeit_s','zuerst_geoeffnet','zuletzt_geaendert');
function cq(x){var s=(x===null||x===undefined)?'':String(x);s=s.replace(/[\r\n]+/g,' ');if(/[",;]/.test(s))s='"'+s.replace(/"/g,'""')+'"';return s;}
function num(x,d){return x===null||x===undefined?'':Number(x).toFixed(d);}
function buildCSV(){
  var out=[HEAD.join(',')];
  for(var i=0;i<N;i++){
    var nr=ANS[i].nr,r=S.recs[String(nr)]||{b:[blank(),blank(),blank()],qrs:'',nb:false,grund:'',bem:'',t0:null,t1:null,secs:0},st=status(r);
    var row=[HASH,nr,st,r.qrs||'',r.nb?(r.grund||''):'',nDoneBeats(r)];
    for(var k=0;k<3;k++){
      var b=r.b[k],a=ampOf(b);
      row.push(num(b.bt,4),num(b.bv,4),num(b.pt,4),num(b.pv,4),a===null?'':num(a,4),a===null?'':num(a*GEO.MMMV,3));
    }
    var m=meanAmp(r);
    row.push(m===null?'':num(m,4),m===null?'':num(m*GEO.MMMV,3),r.bem||'',num(r.secs||0,1),r.t0||'',r.t1||'');
    out.push(row.map(cq).join(','));
  }
  return out.join('\r\n')+'\r\n';
}
function parseCSV(t){
  var rows=[],row=[],f='',q=false;
  for(var i=0;i<t.length;i++){
    var c=t[i];
    if(q){if(c==='"'){if(t[i+1]==='"'){f+='"';i++;}else q=false;}else f+=c;}
    else if(c==='"')q=true;
    else if(c===','){row.push(f);f='';}
    else if(c==='\n'){row.push(f);rows.push(row);row=[];f='';}
    else if(c!=='\r')f+=c;
  }
  if(f.length||row.length){row.push(f);rows.push(row);}
  return rows;
}
function pad2(x){return (x<10?'0':'')+x;}
function doExport(){
  var csv=buildCSV(),d=new Date();
  var name='referenzannotation_s3_ergebnisse_'+d.getFullYear()+pad2(d.getMonth()+1)+pad2(d.getDate())+'_'+pad2(d.getHours())+pad2(d.getMinutes())+'.csv';
  var blob=new Blob([csv],{type:'text/csv;charset=utf-8'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();
  setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},1000);
  S.lastExport=iso();S.doneAtExport=countDone();save();render();
  setHint('Exportiert als '+name+' (im Download-Ordner). Bitte nach s3\\annotation\\ legen.',false);
}
function doImport(text){
  var rows=parseCSV(text);if(rows.length<2){alert('Die Datei enthält keine Daten.');return;}
  var h=rows[0],ix={};h.forEach(function(x,i){ix[x]=i;});
  if(ix.auswahl_hash===undefined||ix.ansicht_nr===undefined){alert('Das ist keine Sicherung dieser Seite.');return;}
  var nrs={};ANS.forEach(function(a,i){nrs[a.nr]=i;});
  var neu={},bad=0;
  for(var i=1;i<rows.length;i++){
    var w=rows[i];if(w.length<h.length)continue;
    if(w[ix.auswahl_hash]!==HASH){bad++;continue;}
    var nr=parseInt(w[ix.ansicht_nr],10);if(nrs[nr]===undefined)continue;
    var r={b:[blank(),blank(),blank()],log:[],cur:0,qrs:w[ix.qrs_richtung]||'',nb:w[ix.status]==='nicht_beurteilbar',grund:w[ix.grund_nicht_beurteilbar]||'',bem:w[ix.bemerkung]||'',t0:w[ix.zuerst_geoeffnet]||null,t1:w[ix.zuletzt_geaendert]||null,secs:parseFloat(w[ix.bearbeitungszeit_s])||0};
    for(var k=0;k<3;k++){
      var p='s'+(k+1)+'_',g=function(s){var x=w[ix[p+s]];return x===''||x===undefined?null:parseFloat(x);};
      var b=r.b[k];b.bt=g('basis_t_s');b.bv=g('basis_mv');b.pt=g('spitze_t_s');b.pv=g('spitze_mv');
      if(b.bt!==null)r.log.push([k,'b']);if(b.pt!==null)r.log.push([k,'p']);
    }
    advance(r);neu[String(nr)]=r;
  }
  if(bad){alert('Achtung: '+bad+' Zeilen gehören zu einer anderen Auswahl und wurden übersprungen.');}
  var n=Object.keys(neu).length;if(!n)return;
  if(!confirm('Den aktuellen Zwischenstand durch die Datei ersetzen ('+n+' Ansichten)?'))return;
  S.recs=neu;save();go(S.pos);render();
}

/* ---------- Ereignisse ---------- */
ov.addEventListener('click',onClick);
ov.addEventListener('mousemove',onMove);
ov.addEventListener('mouseleave',function(){$('readout').textContent='';drawOverlay(null);});
$('bUndo').onclick=undo;
$('bBack').onclick=function(){go(curI-1);};
$('bNext').onclick=function(){go(curI+1);};
$('bOpen').onclick=nextOpen;
$('bZp').onclick=function(){stepZoom(1);};
$('bZm').onclick=function(){stepZoom(-1);};
$('bZf').onclick=function(){S.zoomMode='fit';save();layoutCanvas();};
$('bExp').onclick=doExport;
$('bImp').onclick=function(){$('fImp').click();};
$('fImp').addEventListener('change',function(e){
  var f=e.target.files[0];if(!f)return;var rd=new FileReader();
  rd.onload=function(){doImport(String(rd.result));};rd.readAsText(f,'utf-8');e.target.value='';
});
Array.prototype.forEach.call(document.getElementsByName('qrs'),function(x){x.addEventListener('change',function(){var r=rec(ANS[curI].nr);tick(r);touch(r);r.qrs=x.value;save();render();});});
$('nb').addEventListener('change',function(){var r=rec(ANS[curI].nr);tick(r);touch(r);r.nb=$('nb').checked;if(!r.nb)r.grund='';save();render();});
Array.prototype.forEach.call(document.getElementsByName('grund'),function(x){x.addEventListener('change',function(){var r=rec(ANS[curI].nr);tick(r);touch(r);r.grund=x.value;save();render();});});
$('bem').addEventListener('input',function(){var r=rec(ANS[curI].nr);tick(r);touch(r);r.bem=$('bem').value;save();});
document.addEventListener('keydown',function(e){
  var tag=(e.target&&e.target.tagName)||'';
  if(tag==='TEXTAREA'||tag==='INPUT'&&e.target.type==='text')return;
  if((e.ctrlKey&&(e.key==='z'||e.key==='Z'))||e.key==='Backspace'){e.preventDefault();undo();}
  else if(e.key==='+'||e.key==='='){stepZoom(1);}
  else if(e.key==='-'){stepZoom(-1);}
});
window.addEventListener('resize',function(){if(S.zoomMode==='fit')layoutCanvas();});

/* ---------- Start ---------- */
if(TESTMODUS){
  $('titel').textContent='Referenzannotation T-Amplitude (TESTSEITE, synthetische Streifen)';
  banner('warn','Testseite mit synthetischen Streifen. Es sind keine Daten enthalten. Eingaben hier haben keine Bedeutung.');
}
if(S.pos<0||S.pos>=N)S.pos=0;
curI=S.pos;
save();
layoutCanvas();render();
if(TESTMODUS){
  window.__ref={GEO:GEO,ANS:ANS,S:function(){return S;},rec:rec,go:go,layout:function(){return curL;},pxmm:function(){return pxmm;},
    setZoom:function(z){S.zoomMode='fix';S.zoom=z;layoutCanvas();},view:function(){return curI;},csv:buildCSV,imp:doImport,parse:parseCSV,status:status,mmToTV:mmToTV,tvToMM:tvToMM};
}
})();
</script>
</body>
</html>
'''


def baue_html(ansichten, hash_, key, titel='Referenzannotation T-Amplitude', testmodus=False):
    """ansichten: Liste von dict(nr, n, crc, b64 [, test]). Rueckgabe HTML-Text. Keine externen Ressourcen."""
    daten = json.dumps(ansichten, separators=(',', ':'))
    h = HTML_VORLAGE
    for a, b in (('__HASH__', hash_), ('__KEY__', key), ('__TESTMODUS__', 'true' if testmodus else 'false'), ('__VERSION__', VERSION), ('__TITEL__', titel),
                 ('__DATEN__', daten)):
        assert a in h
        h = h.replace(a, b)
    assert 'http://' not in h.replace('xmlns', '') and 'https://' not in h and '<script src' not in h
    return h


def modus_seite(args):
    pa, ps = ANN / 'AUSWAHL.json', ANN / 'SCHLUESSEL.csv'
    if not pa.exists() or not ps.exists():
        raise SystemExit('AUSWAHL.json oder SCHLUESSEL.csv fehlt; zuerst Modus auswahl')
    info = json.loads(pa.read_text(encoding='utf-8'))
    if sha_datei(ps) != info['schluessel_datei_sha256']:
        raise SystemExit('SCHLUESSEL.csv weicht von AUSWAHL.json ab')
    views = pd.read_csv(ps)
    views = views.rename(columns={'t_mitte': 'c'})
    if auswahl_hash(views) != info['auswahl_hash']:
        raise SystemExit('Auswahl-Hash stimmt nicht')
    # gleiche Bloecke (Wiederholung) nur einmal lesen
    gelesen = {}
    ans = []
    for r in views.itertuples():
        k = (int(r.CSN), int(r.seg), round(float(r.c), 3))
        if k not in gelesen:
            gelesen[k] = lies_block(int(r.CSN), int(r.seg), float(r.c))
        d = kodiere(gelesen[k])
        ans.append({'nr': int(r.ansicht_nr), **d})
    h = baue_html(ans, info['auswahl_hash'], 's3ref_' + info['auswahl_hash'][:16] + '_' + VERSION)
    ziel = ANN / 'REFERENZANNOTATION_S3.html'
    ziel.write_text(h, encoding='utf-8')
    print('geschrieben', ziel, f'{len(h) / 1e6:.2f} MB', 'Ansichten', len(ans), 'Hash', info['auswahl_hash'])


# =========================================================================== Testseite (synthetisch)
def kunst_streifen(t_amp, qrs_amp, offset, hr=60.0, rausch=0.004, seed=1):
    """Synthetischer 10-s-Streifen (500 Hz): Gauss-Komponenten P, QRS (schmal), T; Grundlinie = offset. Rueckgabe (x, r_zeiten). T-Spitze bei R + 0,28 s mit Amplitude
    t_amp (vorzeichenbehaftet, gegen offset), isoelektrische Strecke bei R + 0,50 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    rr = 60.0 / hr
    rs = np.arange(1.5, 10.0, rr)
    x = np.full_like(t, offset)
    g = lambda t0, s, a: a * np.exp(-0.5 * ((t - t0) / s) ** 2)
    for r in rs:
        x += g(r - 0.16, 0.025, 0.10) + g(r, 0.012, qrs_amp) + g(r + 0.28, 0.045, t_amp)
    x += rng.normal(0, rausch, len(t))
    return x, rs


def modus_testseite():
    faelle = [(0.30, 1.20, 0.05), (-0.20, 1.00, -0.10), (0.50, -1.10, 0.00)]
    ans = []
    for k, (ta, qa, off) in enumerate(faelle):
        x, rs = kunst_streifen(ta, qa, off, hr=60.0, seed=11 + k)
        d = kodiere(x)
        d['nr'] = k + 1
        d['test'] = {'t_amp_mv': ta, 'qrs_mv': qa, 'offset_mv': off, 'r_s': [float(r) for r in rs], 'spitze_nach_r_s': 0.28, 'basis_nach_r_s': 0.50}
        ans.append(d)
    h = baue_html(ans, sha_text('testseite'), 's3ref_testseite_' + VERSION, testmodus=True)
    ANN.mkdir(parents=True, exist_ok=True)
    ziel = ANN / 'TESTSEITE.html'
    ziel.write_text(h, encoding='utf-8')
    print('geschrieben', ziel, f'{len(h) / 1e6:.2f} MB')


# =========================================================================== Selbsttest der Ziehung
def selbsttest():
    rng = np.random.default_rng(5)
    n = 700
    mrn = rng.integers(1, 400, n)
    pop = pd.DataFrame({'MRN': mrn, 'CSN': np.arange(n) + 1000, 'split': rng.choice(['train', 'val'], n), 't_order': rng.uniform(1e9, 2e9, n),
                        'c': rng.uniform(1e9, 2e9, n), 'seg': 1, 'pol': np.where(rng.random(n) < 0.2, -1, 1), 'abst': 0.0})
    ok = 0
    a = ziehe_auswahl(pop)
    b = ziehe_auswahl(pop)
    assert a.equals(b), 'nicht deterministisch'
    ok += 1
    assert len(a) == 100 and sorted(a.ansicht_nr) == list(range(1, 101))
    ok += 1
    ori = a[a.wiederholung_von == 0]
    wdh = a[a.wiederholung_von > 0]
    assert len(ori) == 90 and len(wdh) == 10
    ok += 1
    assert ori.MRN.is_unique, 'mehrere Bloecke je Person'
    ok += 1
    assert (ori.pol > 0).sum() == 63 and (ori.pol < 0).sum() == 27
    ok += 1
    for r in wdh.itertuples():
        o = a[a.ansicht_nr == r.wiederholung_von].iloc[0]
        assert o.CSN == r.CSN and o.c == r.c and o.wiederholung_von == 0
        assert r.ansicht_nr - r.wiederholung_von >= MIN_ABSTAND_WDH or r.wiederholung_von - r.ansicht_nr >= MIN_ABSTAND_WDH
    ok += 1
    assert auswahl_hash(a) == auswahl_hash(b) and auswahl_hash(a) != auswahl_hash(ziehe_auswahl(pop, seed=SEED + 1))
    ok += 1
    zu_klein = pop[pop.pol > 0].head(60)
    try:
        ziehe_auswahl(zu_klein)
        raise AssertionError('zu kleine Schicht nicht erkannt')
    except ValueError:
        ok += 1
    # Kodierung: Rundlauf
    x = np.sin(np.linspace(0, 20, N_SAMPLES)) * 2.3
    d = kodiere(x)
    u = np.frombuffer(base64.b64decode(d['b64']), '<i2')
    assert len(u) == N_SAMPLES and np.max(np.abs(u / 1000.0 - x)) <= 0.0006
    ok += 1
    # Vorlage: keine externen Ressourcen, keine langen Gedankenstriche
    h = baue_html([{'nr': 1, **d}], 'h', 'k')
    assert '\u2014' not in h and '\u2013' not in h
    ok += 1
    print(f'Selbsttest {ok} von {ok} bestanden')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('modus', choices=['selbsttest', 'auswahl', 'seite', 'testseite'])
    ap.add_argument('--arbeiter', type=int, default=3)
    a = ap.parse_args(argv)
    if a.modus == 'selbsttest':
        selbsttest()
    elif a.modus == 'testseite':
        modus_testseite()
    elif a.modus == 'seite':
        modus_seite(a)
    else:
        ANN.mkdir(parents=True, exist_ok=True)
        if (ANN / 'AUSWAHL.json').exists():
            raise SystemExit('AUSWAHL.json existiert bereits; die Auswahl wird nicht neu gezogen')
        logf = ANN / 'auswahl.log'

        def log(s):
            line = f'{time.strftime("%H:%M:%S")} {s}'
            print(line, flush=True)
            with open(logf, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

        pop, z = bestimme_grundgesamtheit(a.arbeiter, log)
        log('Grundgesamtheit: ' + json.dumps({k: v for k, v in z.items() if k != 'grundgesamtheit_hash'}))
        views = ziehe_auswahl(pop)
        info = schreibe_auswahl(pop, z, views)
        log('Auswahl geschrieben; Hash ' + info['auswahl_hash'])
        print('auswahl_hash', info['auswahl_hash'])


if __name__ == '__main__':
    main(sys.argv[1:])
