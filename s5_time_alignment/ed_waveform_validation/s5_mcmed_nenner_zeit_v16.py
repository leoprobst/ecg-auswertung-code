"""S5-MC-MED: Nennerzaehlung nach den Regeln von v1.6 (AE-335), NUR AUS ZEITEN (keine HF-Werte, keine Wellenform, keine Offsets, keine Ergebnisgroessen).

Regeln v1.6: je Besuch nur das LAENGSTE Einzelsegment nach Zeitmetadaten (Dauer t1 - t0 aus dem Kopf; Gleichstand: das fruehere); mindestens 3 zeitlich
vollstaendige Einheiten (Beziehung B: Gitter +-15 min, Schritt 5 min, naechste Chart-Referenz <= 30 min, Besuchsfenster E = 600 s); Ankerminimum bei D = 0 ist 2
(drei Anker als Sensitivitaet); Rotationsregeln unveraendert (hier nur die zeitlichen: Raster 300 s, Luecke <= 900 s, mindestens 5 zulaessige Rotationen mit
Verlagerung >= 60 min; Annahme: alle Einheiten des Segments haben eine gueltige HF). NICHT geprueft werden Signalqualitaet, Stationaritaet und Nahtsprung (brauchen
HF-Werte). Die Zahlen sind deshalb OBERGRENZEN; die exakte Zaehlung liefert nenner_b (v16) nach der Extraktion. Beziehung A (Kontrolle) wird auf demselben Segment gezaehlt.
Die Ankerverteilung verwendet die Klassen 1, 2 und 3 oder mehr, damit keine Klasse unter 10 Besuche enthaelt und die sichtbaren Klassen sich zur Gesamtzahl addieren
(in v15 war die Klasse 6 oder mehr mit weniger als 10 Besuchen maskiert; deshalb summierten die sichtbaren Klassen auf 3.957 statt 3.965).
Nur train und val (split_*_test.csv wird nie geoeffnet); Ausgabe maskiert (Zaehlungen 1 bis 9 als "<10"), ohne Kennungen, nur nach <local path>.
Aufruf: <local path> -I s5_mcmed_nenner_zeit_v16.py
"""
import sys, json, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np

HIER = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('s5z', HIER / 's5_mcmed_zaehlung.py')
Z = importlib.util.module_from_spec(sp)
sp.loader.exec_module(Z)

G_S, STEP_S, MAXREF_S = 15 * 60, 300, 1800
RASTER_S, LUECKE_S, ROT_S, MIN_ROT = 300, 900, 3600, 5
ZWEIG = {'z1': (5000, 2500), 'z2': (1000, 500)}


def laengstes_segment(segs):
    """Wie im Skript v16: Dauer t1 - t0, Gleichstand das fruehere (t0, dann Segmentnummer). segs: (seg, t0, t1)."""
    return sorted(segs, key=lambda x: (-(x[2] - x[1]), x[1], x[0]))[0]


def rotationen_zulaessig(ts):
    t = np.sort(np.asarray(ts, float))
    m = len(t)
    if m < 2:
        return 'zu_wenige_einheiten'
    gaps = np.diff(t)
    q = gaps / RASTER_S
    if np.any(np.abs(q - np.round(q)) * RASTER_S > 1.0):
        return 'raster'
    if gaps.max() > LUECKE_S:
        return 'luecke'
    base = np.arange(m)
    n = 0
    for k in range(1, m):
        if np.min(np.abs(t[(base + k) % m] - t)) >= ROT_S:
            n += 1
    return 'ok' if n >= MIN_ROT else 'zu_wenige_rotationen'


def anker_d0(ts, ch, r, d):
    """Verschiedene Chart-Anker bei D = 0 ueber die zeitlich vollstaendigen Einheiten; (Zahl vollstaendiger Einheiten, Zahl Anker)."""
    v = Z.vollstaendig(ts, ch, r, d, G_S, STEP_S, MAXREF_S)
    if v.sum() < 3:
        return int(v.sum()), 0
    lo, hi = r + Z.EXCL, d - Z.EXCL
    ref = ch[(ch >= lo) & (ch <= hi)]
    tv = ts[v]
    kk = np.searchsorted(ref, tv)
    l = np.clip(kk - 1, 0, len(ref) - 1)
    rr = np.clip(kk, 0, len(ref) - 1)
    pick = np.where(np.abs(ref[l] - tv) <= np.abs(ref[rr] - tv), l, rr)          # Gleichstand: frueher
    return int(v.sum()), int(len(np.unique(pick)))


def leer_split():
    return {'besuche_mit_kopf': 0, 'besuche_mit_einheit_im_laengsten_segment': 0, 'B_ge3_vollstaendig': 0, 'B_ge3_vollstaendig_patienten': set(),
            'B_anker_d0_verteilung_ge3': {'1': 0, '2': 0, '3_oder_mehr': 0},
            'B_nenner_anker2': 0, 'B_nenner_anker2_patienten': set(), 'B_rotierbar_anker2': 0, 'B_rotierbar_anker2_patienten': set(), 'B_nicht_rotierbar_anker2_gruende': {},
            'B_nenner_anker3_sensitivitaet': 0, 'B_rotierbar_anker3_sensitivitaet': 0, 'B_rotierbar_anker3_sensitivitaet_patienten': set(),
            'A_ge3_G60_schritt2': 0, 'A_ge3_G30_schritt1': 0}


def main():
    c = Z.verbinden()
    vis = {csn: (s, r, d, mrn) for csn, s, r, d, mrn in c.execute('SELECT CSN,split,r,d,MRN FROM visit WHERE r IS NOT NULL AND d IS NOT NULL').fetchall()}
    chart, mon = {}, {}
    for src, store in (('Chart', chart), ('Monitor', mon)):
        cur = c.execute(f"SELECT CSN,t FROM num WHERE Measure='HR' AND Source='{src}' AND Value BETWEEN 30 AND 220 ORDER BY CSN,t")
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
        seg.setdefault(csn, []).append((s0, t0, t1))
    R = {'hinweis': 'Zeitbasierte Obergrenzen nach v1.6 (laengstes Einzelsegment, Ankerminimum 2); ohne Signalqualitaet, Stationaritaet und Nahtsprung.', 'splits': {}}
    for s in Z.SPLITS:
        R['splits'][s] = leer_split()
    for csn, segs in seg.items():
        if csn not in vis:
            continue
        s, r, d, mrn = vis[csn]
        S = R['splits'][s]
        S['besuche_mit_kopf'] += 1
        sg = laengstes_segment(segs)
        ts = Z.einheiten_zeiten(sg[1], sg[2], r, d)
        if len(ts) == 0:
            continue
        S['besuche_mit_einheit_im_laengsten_segment'] += 1
        # Beziehung A (Kontrolle) auf demselben Segment
        mo = mon.get(csn, np.empty(0))
        S['A_ge3_G60_schritt2'] += int(Z.vollstaendig(ts, mo, r, d, 60 * 60, 120, Z.MAXREF_A).sum() >= 3)
        S['A_ge3_G30_schritt1'] += int(Z.vollstaendig(ts, mo, r, d, 30 * 60, 60, Z.MAXREF_A).sum() >= 3)
        nv, na = anker_d0(ts, chart.get(csn, np.empty(0)), r, d)
        if nv < 3:
            continue
        S['B_ge3_vollstaendig'] += 1
        S['B_ge3_vollstaendig_patienten'].add(mrn)
        key = '3_oder_mehr' if na >= 3 else str(na)
        S['B_anker_d0_verteilung_ge3'][key] += 1
        grund = rotationen_zulaessig(ts)           # Rotationsregeln gelten fuer die Einheiten des gewaehlten Segments
        if na >= 2:
            S['B_nenner_anker2'] += 1
            S['B_nenner_anker2_patienten'].add(mrn)
            if grund == 'ok':
                S['B_rotierbar_anker2'] += 1
                S['B_rotierbar_anker2_patienten'].add(mrn)
            else:
                S['B_nicht_rotierbar_anker2_gruende'][grund] = S['B_nicht_rotierbar_anker2_gruende'].get(grund, 0) + 1
        if na >= 3:
            S['B_nenner_anker3_sensitivitaet'] += 1
            if grund == 'ok':
                S['B_rotierbar_anker3_sensitivitaet'] += 1
                S['B_rotierbar_anker3_sensitivitaet_patienten'].add(mrn)

    def zweig(n, p):
        return 'Z1' if (n >= ZWEIG['z1'][0] and p >= ZWEIG['z1'][1]) else ('Z2' if (n >= ZWEIG['z2'][0] and p >= ZWEIG['z2'][1]) else 'Z3')
    for s, S in R['splits'].items():
        for k in ('B_ge3_vollstaendig_patienten', 'B_nenner_anker2_patienten', 'B_rotierbar_anker2_patienten', 'B_rotierbar_anker3_sensitivitaet_patienten'):
            S[k] = len(S[k])
        S['B_anker_verteilung_summe_ist_gleich_ge3'] = bool(sum(S['B_anker_d0_verteilung_ge3'].values()) == S['B_ge3_vollstaendig'])
        S['zweig_wenn_signal_und_stationaritaet_ok_anker2'] = zweig(S['B_rotierbar_anker2'], S['B_rotierbar_anker2_patienten'])
        S['zweig_wenn_signal_und_stationaritaet_ok_anker3_sensitivitaet'] = zweig(S['B_rotierbar_anker3_sensitivitaet'], S['B_rotierbar_anker3_sensitivitaet_patienten'])
    Z.schreibe('nenner_b_zeit_v16', R)
    print(json.dumps(Z.mask_tree(R), indent=1))
    return R


if __name__ == '__main__':
    main()
