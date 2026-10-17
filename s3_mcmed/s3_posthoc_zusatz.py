"""POST HOC, BESCHREIBEND, NICHT GEBUNDEN. Zusatzfelder der Besuche fuer die Basistabelle S3 (Auftrag K 17.10.2026).

Modus laden: baut die Gruppen A, B0, C*, D* wie s3_posthoc_basis.py (gebundene Funktionen, Bestand nur lesend), liest aus der Rohdatei visits.csv (MC-MED 1.0.1, nur lesend)
Triage_acuity, ED_dispo, ED_LOS und Dx_ICD10 der Besuche dieser Gruppen und schreibt sie mit der Gruppenzuordnung in die eigene Datei zusatz.duckdb (privat, mit Kennungen,
nie nach <cloud folder>). Der Bestand s3_gesamt.duckdb wird nicht veraendert.
Modus werte: Verteilung der Werte (nur Haeufigkeiten), zur Festlegung der Klassen.
Modus tabelle: Zusammenfassung je Gruppe (Personenebene, erstes Paar), private JSON und oeffentlich taugliche Fragmente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, json, re, time, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
WERK = Path(str(data_path("s3.code_dir")))
sys.path.insert(0, str(WERK))
import numpy as np
import pandas as pd
import s3_posthoc_basis as P

OUT = P.OUT
ZUS = OUT / 'zusatz.duckdb'
VISITS = Path(str(data_path("s3.mcmed_visits_csv")))
GR = ['A', 'B0', 'C_ana', 'D_ana']


def laden():
    import duckdb
    log = P.H.Protokoll(OUT / 'protokoll.log')
    log('Zusatz laden; Skript', P.B.sha(Path(__file__))[:16], 'visits.csv Groesse', VISITS.stat().st_size)
    w = P.Waechter(log)
    w.start()
    with P.RamReservierung(1.5, 'S3 Basistabelle Zusatzfelder (1 Prozess)', log):
        soll = json.loads(P.JSON_GEBUNDEN.read_text(encoding='utf-8'))
        T, _sp, _d, _dc = P.tabellen_lesend(log)
        G, info = P.gruppen_bauen(T['Q'], P.CFG, soll, log)
        rows = []
        for g in GR:
            d = G[g][['CSN', 'MRN', 'ta', 'tb']].copy()
            d['gruppe'] = g
            rows.append(d)
        M = pd.concat(rows, ignore_index=True)
        del T
        c = duckdb.connect(str(ZUS))
        c.execute("SET memory_limit='500MB'; SET threads=1")
        c.register('m_df', M)
        c.execute('CREATE OR REPLACE TABLE gruppe_paare AS SELECT gruppe, CAST(CSN AS BIGINT) CSN, CAST(MRN AS BIGINT) MRN, ta, tb FROM m_df')
        csns = pd.DataFrame({'CSN': sorted(M.CSN.unique().astype('int64'))})
        c.register('csn_df', csns)
        allv = 'all_varchar=true,strict_mode=false,null_padding=true,ignore_errors=true'
        c.execute(f"""CREATE OR REPLACE TABLE visit_zusatz AS SELECT CAST(v.CSN AS BIGINT) CSN, v.Triage_acuity, v.ED_dispo, try_cast(v.ED_LOS AS DOUBLE) ED_LOS, v.Dx_ICD10
            FROM read_csv('{VISITS.as_posix()}', header=true, {allv}) v JOIN csn_df ON csn_df.CSN = try_cast(v.CSN AS BIGINT)""")
        n_csn, n_vis = len(csns), c.execute('SELECT count(*) FROM visit_zusatz').fetchone()[0]
        n_dup = c.execute('SELECT count(*) - count(DISTINCT CSN) FROM visit_zusatz').fetchone()[0]
        log('Besuche gesucht', n_csn, 'gefunden', n_vis, 'doppelt', n_dup)
        c.close()
    w.halt = True
    log('Zusatz fertig; RSS-Spitze MB', int(w.spitze))


def werte():
    import duckdb
    c = duckdb.connect(str(ZUS), read_only=True)
    for col in ('Triage_acuity', 'ED_dispo'):
        print(col, c.execute(f"SELECT coalesce({col}, '(NULL)') v, count(*) n FROM visit_zusatz GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print('ED_LOS', c.execute('SELECT count(ED_LOS), min(ED_LOS), median(ED_LOS), max(ED_LOS) FROM visit_zusatz').fetchone())
    print('Dx Anzahl NULL/leer', c.execute("SELECT count(*) FILTER (WHERE Dx_ICD10 IS NULL OR Dx_ICD10 = '') FROM visit_zusatz").fetchone())
    print('Dx Muster (Beispiele, keine Personen):', c.execute("SELECT Dx_ICD10 FROM visit_zusatz WHERE Dx_ICD10 LIKE '%N18%' OR Dx_ICD10 LIKE '%E87%' LIMIT 6").fetchall())
    print('Trennzeichen:', c.execute("SELECT count(*) FILTER (WHERE Dx_ICD10 LIKE '%;%'), count(*) FILTER (WHERE Dx_ICD10 LIKE '%,%'), count(*) FILTER (WHERE Dx_ICD10 LIKE '%|%') FROM visit_zusatz").fetchone())


DX = {'nierenkrank_dialyse': ('N18', 'Z992'), 'aki': ('N17',), 'diabetes': ('E10', 'E11', 'E12', 'E13', 'E14'), 'herzinsuffizienz': ('I50',), 'hyperkaliaemie': ('E875',)}
DISPO = {'Discharge': 'entlassen', 'Observation': 'beobachtung', 'Inpatient': 'stationaer', 'ICU': 'intensiv'}


def klasse_triage(x):
    if x is None or str(x)[:1] not in '12345':
        return 'fehlend'
    return {'1': 'stufe1', '2': 'stufe2', '3': 'stufe3'}.get(str(x)[:1], 'stufe4_5')


def dx_gruppen(s):
    tok = [re.sub(r'[^A-Z0-9]', '', t.upper()) for t in str(s or '').split(',')]
    return {k: any(t.startswith(pf) for t in tok for pf in pfs) for k, pfs in DX.items()}


def personen_frame():
    import duckdb
    c = duckdb.connect(str(ZUS), read_only=True)
    d = c.execute('SELECT g.gruppe, g.CSN, g.MRN, g.ta, g.tb, v.Triage_acuity, v.ED_dispo, v.ED_LOS, v.Dx_ICD10 FROM gruppe_paare g LEFT JOIN visit_zusatz v USING (CSN)').df()
    c.close()
    d = d.sort_values(['gruppe', 'MRN', 'ta', 'tb', 'CSN'], kind='mergesort').drop_duplicates(['gruppe', 'MRN'], keep='first')
    d['triage'] = [klasse_triage(x) for x in d.Triage_acuity]
    d['dispo'] = [DISPO.get(x, 'fehlend') for x in d.ED_dispo]
    dg = pd.DataFrame([dx_gruppen(x) for x in d.Dx_ICD10], index=d.index)
    return pd.concat([d, dg], axis=1)


def zusammenfassen(d):
    n = int(len(d))
    cnt = lambda col, vals: {v: int((d[col] == v).sum()) for v in vals}
    return {'personen': n, 'triage': cnt('triage', ['stufe1', 'stufe2', 'stufe3', 'stufe4_5', 'fehlend']),
            'dispo': cnt('dispo', ['entlassen', 'beobachtung', 'stationaer', 'intensiv', 'fehlend']),
            'ed_los': P.q3(d.ED_LOS), 'dx': {k: int(d[k].sum()) for k in DX}, 'dx_fehlt': int((d.Dx_ICD10.isna() | d.Dx_ICD10.eq('')).sum())}


def tabelle_roh():
    d = personen_frame()
    S = {g: zusammenfassen(d[d.gruppe == g]) for g in GR}
    O = json.loads((OUT / 'oeffentlich.json').read_text(encoding='utf-8'))
    Z = {'kopf': 'post hoc, beschreibend, nicht gebunden; ED-Diagnosen, keine Komorbiditaetsanamnese', 'gruppen': {}}
    for g in GR:
        s = S[g]
        G = O['gruppen'][g]
        n, np_ = s['personen'], P.floor10(s['personen'])
        tri = dict(s['triage'])
        tri['stufe4_5_oder_fehlend'] = tri.pop('stufe4_5') + tri.pop('fehlend')
        dis = dict(s['dispo'])
        dis['beobachtung_oder_fehlend'] = dis.pop('beobachtung') + dis.pop('fehlend')
        Z['gruppen'][g] = {'titel': G['titel'], 'personen_spanne_oeffentlich': G['personen_spanne_oeffentlich'],
                           'triage': P.pub_familie(tri, n, np_, grob=np_ < 500), 'dispo': P.pub_familie(dis, n, np_, grob=np_ < 500),
                           'ed_los_verfuegbar': P.pub_anteil(s['ed_los']['n'], n, np_), 'ed_los': P.pub_stat(s['ed_los'], 1),
                           'dx': {k: P.pub_anteil(v, n, np_) for k, v in s['dx'].items()}}
    return S, Z


NM_TRI = {'stufe1': 'Triage 1 (Reanimation)', 'stufe2': 'Triage 2 (Emergent)', 'stufe3': 'Triage 3 (Urgent)', 'stufe4_5_oder_fehlend': 'Triage 4 und 5 oder ohne Angabe'}
NM_DIS = {'entlassen': 'Entlassung nach Hause', 'stationaer': 'stationäre Aufnahme', 'intensiv': 'Intensivstation', 'beobachtung_oder_fehlend': 'Beobachtung (sonstige) oder ohne Angabe'}
NM_DX = {'nierenkrank_dialyse': 'ED-Diagnose chronische Nierenkrankheit oder Dialyse (N18, Z99.2)', 'aki': 'ED-Diagnose akute Nierenschädigung (N17)',
         'diabetes': 'ED-Diagnose Diabetes (E10 bis E14)', 'herzinsuffizienz': 'ED-Diagnose Herzinsuffizienz (I50)', 'hyperkaliaemie': 'ED-Diagnose Hyperkaliämie (E87.5)'}


def zusatz_md(Z):
    gs = GR
    kopf = ['Merkmal'] + [Z['gruppen'][g]['titel'] for g in gs]
    z = []
    for k, nm in NM_TRI.items():
        z.append([nm] + [Z['gruppen'][g]['triage'][k]['anzeige'] for g in gs])
    for k, nm in NM_DIS.items():
        z.append([nm] + [Z['gruppen'][g]['dispo'][k]['anzeige'] for g in gs])
    z.append(['Aufenthaltsdauer in der Notaufnahme (ED_LOS) vorhanden'] + [Z['gruppen'][g]['ed_los_verfuegbar']['anzeige'] for g in gs])
    z.append(['Aufenthaltsdauer in der Notaufnahme (ED_LOS), Rohdateieinheit: Median (Q1 bis Q3)'] + [P.zelle_stat(Z['gruppen'][g]['ed_los'], 1) for g in gs])
    for k, nm in NM_DX.items():
        z.append([nm] + [Z['gruppen'][g]['dx'][k]['anzeige'] for g in gs])
    t = ['| ' + ' | '.join(kopf) + ' |', '|' + '---|' * len(kopf)]
    return '\n'.join(t + ['| ' + ' | '.join(r) + ' |' for r in z])


def tabelle():
    S, Z = tabelle_roh()
    R = {'erstellt': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'skript_sha256': P.B.sha(Path(__file__)), 'zusammenfassung_privat': S}
    (OUT / 'zusatz_privat.json').write_text(json.dumps(P.sicher(R), indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'zusatz_oeffentlich.json').write_text(json.dumps(P.sicher(Z), indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'zusatz_fragment.md').write_text(zusatz_md(Z) + '\n', encoding='utf-8')
    print(zusatz_md(Z))


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    {'laden': laden, 'werte': werte, 'tabelle': tabelle}.get(m, lambda: print(__doc__))()
