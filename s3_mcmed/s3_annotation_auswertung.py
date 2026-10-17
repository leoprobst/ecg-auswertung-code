"""S3 MC-MED: beschreibender Vergleich der kaliumblinden Referenzannotation (T-Amplitude, Ableitung II, Anwender) mit der gebundenen Messkette aw255-v3.

Spezifikation SPEZIFIKATION_S3_MCMED_v2_2.md, Abschnitt 10 Punkt 3 (beschreibend, ohne Tor). Auftrag von K, 17.10.2026.

Aufruf: <local path> -I <local path>

Die gebundene Kette wird UNVERAENDERT ueber den regulaeren Weg aufgerufen: pruefe_start(lock = spec_lock_s3_v2_2.json) -> Freigabe -> setze_bindung -> messe_bei.
Keine gebundene Datei wird veraendert oder kopiert. Es werden keine Daten ausserhalb von <local path> geschrieben (ausser dem Aggregatbericht, siehe schreibe_md.py-Schritt).
Nur Train/Val-Besuche (Pruefung im Skript); der Testsplit wird nicht gelesen. Ein Prozess, kleiner Speicherbedarf. Keine Loeschbefehle.
Ausgaben (privat) in <local path>: ansichten_vergleich.csv, aggregate.json, protokoll.txt, abbildung_s3_annotation.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, hashlib, math, time, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

ANN = Path(str(data_path("s3.annotation_dir")))
OUT = ANN / 'auswertung'
LOCK = Path(str(data_path("s3.spec_lock_json")))
CSV_ANW = ANN / 'referenzannotation_s3_ergebnisse_20261017_1056.csv'
HIER_KETTE = Path(str(data_path("s3.code_dir")))
GESAMT_DB = Path(str(data_path("s3.full_database_duckdb")))
FS = 500
LOGF = OUT / 'protokoll.txt'


def log(*a):
    s = dt.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, 'a', encoding='utf-8') as f:
        f.write(s + '\n')


def sha_datei(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- Statistik (beschreibend)
def vergleich(u, k):
    """u = Anwender, k = Kette (gepaart, endlich). Differenz d = u - k. Rueckgabe dict."""
    u, k = np.asarray(u, float), np.asarray(k, float)
    ok = np.isfinite(u) & np.isfinite(k)
    u, k = u[ok], k[ok]
    n = len(u)
    if n < 3:
        return {'n': int(n)}
    d = u - k
    sd = float(np.std(d, ddof=1))
    bias = float(d.mean())
    se = sd / math.sqrt(n)
    tq = float(tdist.ppf(0.975, n - 1))
    m = (u + k) / 2
    msb = 2 * np.var(m, ddof=1)                    # ICC(1,1) fuer k = 2 Bewertungen je Ansicht
    msw = float(np.sum(d ** 2) / (2 * n))
    icc = float((msb - msw) / (msb + msw))
    sl = float(np.polyfit(m, d, 1)[0]) if np.ptp(m) > 0 else float('nan')
    return {'n': int(n), 'bias': bias, 'bias_ci': [bias - tq * se, bias + tq * se], 'sd_diff': sd,
            'loa': [bias - 1.96 * sd, bias + 1.96 * sd], 'icc11': icc, 'spearman': float(spearmanr(u, k).correlation),
            'innerhalb_005': int((np.abs(d) <= 0.05).sum()), 'anteil_innerhalb_005': float((np.abs(d) <= 0.05).mean()),
            'median_diff': float(np.median(d)), 'median_abs_diff': float(np.median(np.abs(d))), 'steigung_diff_auf_mittel': sl,
            'mittel_anwender': float(u.mean()), 'mittel_kette': float(k.mean()), 'sd_anwender': float(np.std(u, ddof=1)), 'sd_kette': float(np.std(k, ddof=1))}


def wiederholbarkeit(a, b):
    """Zwei Messungen derselben Ansicht: b - a (zweite minus erste Vorlage in Anzeigereihenfolge). SD innerhalb, RC = 1,96*sqrt(2)*SD innerhalb, Bias."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 2:
        return {'n': int(n)}
    d = b - a
    sw = float(math.sqrt(np.sum(d ** 2) / (2 * n)))
    return {'n': int(n), 'sd_innerhalb': sw, 'rc': float(1.96 * math.sqrt(2) * sw), 'bias': float(d.mean()), 'max_abs_diff': float(np.max(np.abs(d)))}


# --------------------------------------------------------------------------- Hauptablauf
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log('Start; Python', sys.version.split()[0], '; Skript', Path(__file__).name, 'sha256', sha_datei(__file__)[:16])

    # ---- Eingaben und Pruefungen
    info = json.loads((ANN / 'AUSWAHL.json').read_text(encoding='utf-8'))
    assert sha_datei(ANN / 'SCHLUESSEL.csv') == info['schluessel_datei_sha256'], 'SCHLUESSEL.csv weicht von AUSWAHL.json ab'
    key = pd.read_csv(ANN / 'SCHLUESSEL.csv')
    anw = pd.read_csv(CSV_ANW)
    assert len(key) == 100 and len(anw) == 100 and anw.auswahl_hash.nunique() == 1 and anw.auswahl_hash.iloc[0] == info['auswahl_hash'], 'Auswahl-Hash der Anwender-CSV stimmt nicht'
    log('Auswahl-Hash und Schluessel-Datei geprueft; Anwender-CSV sha256', sha_datei(CSV_ANW)[:16])
    assert set(key.split.unique()) <= {'train', 'val'}, 'Testsplit in der Auswahl'
    log('Split in der Auswahl:', key.split.value_counts().to_dict(), '-> kein Testsplit')
    assert (np.sort(key.ansicht_nr.values) == np.arange(1, 101)).all() and (np.sort(anw.ansicht_nr.values) == np.arange(1, 101)).all()
    D = key.merge(anw.drop(columns=['auswahl_hash']), on='ansicht_nr', validate='1:1').sort_values('ansicht_nr').reset_index(drop=True)
    D['gemessen'] = (D.status == 'gemessen').astype(int)
    assert (D.loc[D.gemessen == 1, 'n_schlaege'] == 3).all()

    # ---- gebundene Kette ueber den regulaeren Weg
    sys.path.insert(0, str(HIER_KETTE))
    import s3_auswertung_v2_2 as V
    fr = V.pruefe_start(LOCK)                                       # prueft Spezifikation, Skript, Module, Konfiguration, Vertrag gegen die Bindung
    V.setze_bindung(fr)
    lockjs = json.loads(LOCK.read_text(encoding='utf-8'))
    log('Bindung geprueft (pruefe_start): Pipeline', V.PIPELINE, '; Freigabe-Hash', fr.hash[:16], '; Modulhashes', {k: v[:12] for k, v in fr.abhaengig.items()})
    assert not hasattr(V, '_ungueltig')
    CFG = V.CFG
    import wfdb

    zeilen = []
    gelesen = {}
    t_start = time.time()
    for r in D.itertuples():
        csn, sno, c = int(r.CSN), int(r.seg), float(r.t_mitte)
        p = V.WAVE / str(csn)[-3:] / str(csn) / 'II' / f'{csn}_{sno}'
        ns, fs, n, t0, einheit = V.lies_kopf(str(p) + '.hea')
        assert fs == FS and str(einheit).lower() == 'mv'
        seg = (sno, float(fs), int(n), float(t0), float(t0 + n / fs))
        m = V.messe_bei(csn, seg, c, CFG, V.WAVE)                    # exakt die Pruefung der Pipeline: 60-s-Fenster um die Blockmitte
        z = {'ansicht_nr': int(r.ansicht_nr), 'k_gemessen': m is not None}
        # Block des Anwenders (wie lies_block der Annotation): 10 s ab c - 5 s, physische Einheiten
        a0 = int(round((c - 5.0 - t0) * fs))
        rec = wfdb.rdrecord(str(p), sampfrom=a0, sampto=a0 + 10 * FS, physical=True, channels=[0])
        xb = rec.p_signal[:, 0].astype(float)
        xq = np.round(xb * 1000.0) / 1000.0                          # wie in der Seite (ganze Mikrovolt)
        z['block_a0'] = a0
        if m is not None:
            w0, w1 = m['w0'], m['w1']
            z['block_im_fenster'] = bool(w0 <= c - 5.0 + 0.002 and c + 5.0 <= w1 + 0.002)      # Toleranz 2 ms (Gleitkomma bei 1e9-Sekunden-Zeiten)
            # Kontrolle: Rohsignal, das die Kette liest, stimmt am Block mit dem Signal der Annotation ueberein
            rd = V.lies_roh(csn, seg, w0 - 4.0, w1 + 4.0, V.WAVE)
            ts_, xr = rd
            i0 = int(round((c - 5.0 - ts_) * FS))
            # Kontrolle der Ausrichtung: Versatz in Abtastwerten (-2 bis +2), der die Abweichung minimiert; Rohdaten sind dieselben Dateien
            best = min(((float(np.max(np.abs(xr[i0 + o:i0 + o + 10 * FS] - xb))) * 1000.0, o) for o in range(-2, 3) if i0 + o >= 0 and i0 + o + 10 * FS <= len(xr)))
            z['block_signal_maxabs_diff_uv'] = best[0]
            z['block_versatz_abtastwerte'] = int(best[1])
            for k in ('valid', 'reason', 'pol', 'pol_unsicher', 't_amp', 't_roh', 'steil', 'qrs_ms', 'pr_ms', 'hr', 'w0', 'w1'):
                z['k_' + k] = m.get(k)
        zeilen.append(z)
        if len(zeilen) % 20 == 0:
            log(f'Kette: {len(zeilen)} von 100 Ansichten ({time.time() - t_start:.0f} s)')
    K = pd.DataFrame(zeilen)
    D = D.merge(K, on='ansicht_nr', validate='1:1')
    log('Kette fertig:', f'{time.time() - t_start:.0f} s', '; Fenster gefunden:', int(D.k_gemessen.sum()), 'von 100; Block im Fenster:', int(D.get('block_im_fenster', pd.Series(dtype=bool)).fillna(False).sum()),
        '; max Signalabweichung (uV):', float(D.block_signal_maxabs_diff_uv.max()), '; Versatz (Abtastwerte) Verteilung:', D.block_versatz_abtastwerte.value_counts().to_dict())

    # ---- Zuordnung der Groessen
    D['k_valid'] = D.k_valid.fillna(0).astype(int)
    D['u_roh'] = D.mittel_amp_mv                                     # Anwender: Spitze - Basis, vorzeichenbehaftet im gezeigten (rohen) Ableitung-II-Signal
    qsign = D.qrs_richtung.map({'positiv': 1.0, 'negativ': -1.0})    # isoelektrisch -> NaN
    D['u_rel_userqrs'] = D.u_roh * qsign                             # relativ zur QRS-Richtung des Anwenders
    D['u_rel_auto'] = D.u_roh * D.pol_auto.map({'positiv': 1.0, 'negativ': -1.0})   # relativ zur automatischen Polaritaet (Bezugssystem der Kette)
    D['k_roh'] = np.where(D.k_valid == 1, D.k_t_roh, np.nan)
    D['k_rel'] = np.where(D.k_valid == 1, D.k_t_amp, np.nan)
    D['d_roh'] = D.u_roh - D.k_roh
    D['d_rel_auto'] = D.u_rel_auto - D.k_rel
    D['d_rel_userqrs'] = D.u_rel_userqrs - D.k_rel
    # Plausibilitaet: pol der Kette == pol_auto der Auswahl?
    D['pol_kette_gleich_auto'] = (D.k_pol.map({1.0: 'positiv', -1.0: 'negativ'}) == D.pol_auto)
    log('pol der Kette gleich pol_auto der Auswahl:', int(D.pol_kette_gleich_auto.sum()), 'von', int(D.k_gemessen.sum()))

    # ---- Kontrolle gegen den gebundenen Gesamtlauf (nur lesend, nur die Besuche dieser Auswahl)
    quer = {}
    try:
        import duckdb
        con = duckdb.connect(str(GESAMT_DB), read_only=True)
        con.execute("SET memory_limit='500MB'; SET threads=1")
        cols = [c_[0] for c_ in con.execute('DESCRIBE mess_v2').fetchall()]
        if all(x in cols for x in ('c_d3', 't_roh_d3', 'valid_d3', 't_amp_d3')):
            G = con.execute('SELECT CSN, c_d3, valid_d3, t_roh_d3, t_amp_d3 FROM mess_v2 WHERE c_d3 IS NOT NULL AND CSN IN (SELECT unnest(?))', [[int(x) for x in D.CSN.unique()]]).df()
            n_match = 0
            dif = []
            for r in D.itertuples():
                g = G[(G.CSN == r.CSN) & (np.abs(G.c_d3 - r.t_mitte) < 0.01)]
                if len(g) >= 1:
                    n_match += 1
                    g = g.iloc[0]
                    if int(g.valid_d3 or 0) == r.k_valid and (r.k_valid == 0 or abs(g.t_roh_d3 - r.k_t_roh) < 1e-9):
                        dif.append(0)
                    else:
                        dif.append(1)
            quer = {'gefunden': n_match, 'abweichend': int(sum(dif))}
            log('Gesamtlauf-Kontrolle (mess_v2, lesend): Ansichten mit gespeicherter Messung am selben Block:', n_match, '; abweichend:', int(sum(dif)))
        else:
            log('Gesamtlauf-Kontrolle uebersprungen: Spalten fehlen', cols[:10])
        con.close()
    except Exception as e:
        log('Gesamtlauf-Kontrolle uebersprungen:', repr(e)[:200])

    # ---- Auswertung
    A = {'kontrolle_gesamtlauf': quer}
    G = D[D.gemessen == 1]
    A['n_ansichten'] = int(len(D))
    A['n_gemessen'] = int(len(G))
    A['n_nicht_beurteilbar'] = int((D.gemessen == 0).sum())
    A['kette_gueltig_unter_gemessen'] = int((G.k_valid == 1).sum())
    A['kette_ungueltig_gruende_unter_gemessen'] = G[G.k_valid == 0].k_reason.value_counts().to_dict()
    A['kette_fenster_nicht_gefunden'] = int((D.k_gemessen == False).sum())
    for nm, uc, kc, dd in (('roh', 'u_roh', 'k_roh', G), ('rel_auto', 'u_rel_auto', 'k_rel', G), ('rel_userqrs', 'u_rel_userqrs', 'k_rel', G)):
        A[f'gesamt_{nm}'] = vergleich(dd[uc], dd[kc])
        for q in ('positiv', 'negativ'):
            s = dd[dd.qrs_richtung == q]
            A[f'{q}_{nm}'] = vergleich(s[uc], s[kc])
    # Sensitivitaet: ein Wert je Block (die 10 doppelten Ansichten gehen sonst zweimal ein); Anwender = Mittel der gemessenen Vorlagen, Kette identisch
    D['block_id'] = D.CSN.astype(str) + '_' + D.seg.astype(str) + '_' + D.t_mitte.round(3).astype(str)
    B = G.assign(block_id=D.loc[G.index, 'block_id']).groupby('block_id').agg(u_roh=('u_roh', 'mean'), k_roh=('k_roh', 'first'), k_rel=('k_rel', 'first'), u_rel_auto=('u_rel_auto', 'mean'),
                                                                        qrs=('qrs_richtung', 'first'), nview=('ansicht_nr', 'count')).reset_index()
    A['bloecke_gemessen'] = int(len(B))
    A['block_roh'] = vergleich(B.u_roh, B.k_roh)
    A['block_rel_auto'] = vergleich(B.u_rel_auto, B.k_rel)
    for q in ('positiv', 'negativ'):
        sb = B[B.qrs == q]
        A[f'block_{q}_roh'] = vergleich(sb.u_roh, sb.k_roh)
    A['block_vorzeichen_gleich_roh'] = int((np.sign(B.u_roh[B.k_roh.notna()]) == np.sign(B.k_roh[B.k_roh.notna()])).sum())
    A['block_n_vorzeichen_roh'] = int(B.k_roh.notna().sum())
    # Abweichung nach QRS-Dauer der Kette
    for nm, msk in (('qrs_ge_120', G.k_qrs_ms >= 120), ('qrs_lt_120', G.k_qrs_ms < 120), ('qrs_nan', G.k_qrs_ms.isna())):
        sq = G[msk & G.d_roh.notna()]
        A['abw_' + nm] = {'n': int(len(sq)), 'median_abs_diff': float(sq.d_roh.abs().median()) if len(sq) else None, 'n_abs_gt_015': int((sq.d_roh.abs() > 0.15).sum())}
    # Zusatz: roh, getrennt nach pol_auto (Bezug der Kette)
    for q in ('positiv', 'negativ'):
        s = G[G.pol_auto == q]
        A[f'polauto_{q}_roh'] = vergleich(s.u_roh, s.k_roh)
    # Vorzeichenuebereinstimmung der T-Amplitude (roh)
    ok = G.u_roh.notna() & G.k_roh.notna()
    A['vorzeichen_gleich_roh'] = int((np.sign(G.u_roh[ok]) == np.sign(G.k_roh[ok])).sum())
    A['n_vorzeichen_roh'] = int(ok.sum())

    # QRS-Richtung Anwender vs. pol_auto (alle 100, nur positiv/negativ im Anwenderurteil)
    kt = pd.crosstab(D.qrs_richtung, D.pol_auto)
    A['qrs_kreuztabelle_alle'] = {str(i): {str(j): int(kt.loc[i, j]) for j in kt.columns} for i in kt.index}
    Gq = G[G.qrs_richtung.isin(['positiv', 'negativ'])]
    A['qrs_gemessene_n'] = int(len(Gq))
    A['qrs_gemessene_gleich_auto'] = int((Gq.qrs_richtung == Gq.pol_auto).sum())
    Dq = D[D.qrs_richtung.isin(['positiv', 'negativ'])]
    A['qrs_alle_n'] = int(len(Dq))
    A['qrs_alle_gleich_auto'] = int((Dq.qrs_richtung == Dq.pol_auto).sum())
    A['qrs_kreuz_gemessene'] = {i: {j: int(((Gq.qrs_richtung == i) & (Gq.pol_auto == j)).sum()) for j in ('positiv', 'negativ')} for i in ('positiv', 'negativ')}
    A['qrs_isoelektrisch_n'] = int((D.qrs_richtung == 'isoelektrisch').sum())

    # Wiederholbarkeit an den 10 doppelten Ansichten: erste Vorlage = kleinere Ansichtsnummer
    zweit = D[D.wiederholung_von > 0]
    paare = []
    for r in zweit.itertuples():
        o = D[D.ansicht_nr == r.wiederholung_von].iloc[0]
        a, b = (o, D[D.ansicht_nr == r.ansicht_nr].iloc[0])
        if a.ansicht_nr > b.ansicht_nr:
            a, b = b, a
        paare.append({'a': int(a.ansicht_nr), 'b': int(b.ansicht_nr), 'u_a': a.u_roh, 'u_b': b.u_roh, 'k_a': a.k_roh, 'k_b': b.k_roh,
                      'urel_a': a.u_rel_auto, 'urel_b': b.u_rel_auto, 'krel_a': a.k_rel, 'krel_b': b.k_rel,
                      'qrs_gleich': a.qrs_richtung == b.qrs_richtung, 'status_gleich': a.status == b.status, 'gem_beide': (a.gemessen == 1 and b.gemessen == 1),
                      'abst_vorlagen': int(b.ansicht_nr - a.ansicht_nr)})
    P = pd.DataFrame(paare)
    A['paare_n'] = int(len(P))
    A['paare_beide_gemessen'] = int(P.gem_beide.sum())
    A['paare_qrs_gleich'] = int(P.qrs_gleich.sum())
    A['paare_status_gleich'] = int(P.status_gleich.sum())
    A['paare_abstand_min'] = int(P.abst_vorlagen.min())
    A['wh_anwender_roh'] = wiederholbarkeit(P.u_a, P.u_b)
    A['wh_kette_roh'] = wiederholbarkeit(P.k_a, P.k_b)
    A['wh_anwender_rel'] = wiederholbarkeit(P.urel_a, P.urel_b)
    A['wh_kette_rel'] = wiederholbarkeit(P.krel_a, P.krel_b)
    # Kette an Paaren identisch? (gleicher Block -> identische Eingabe)
    pk = P[P.k_a.notna() & P.k_b.notna()]
    A['wh_kette_identisch'] = bool(np.all(pk.k_a.values == pk.k_b.values)) if len(pk) else None
    A['wh_anwender_icc_bland'] = wiederholbarkeit(P.u_a, P.u_b)
    # Mittelwert Anwender in den Paaren gegen die Kette (informativ): Anwender-Mittel der beiden Vorlagen vs. Kette
    # nicht beurteilbare Ansichten
    N = D[D.gemessen == 0]
    A['nb_gruende'] = {str(k): int(v) for k, v in N.grund_nicht_beurteilbar.value_counts().items()}
    A['nb_kette_gueltig'] = int((N.k_valid == 1).sum())
    A['nb_kette_ungueltig'] = int((N.k_valid == 0).sum())
    A['nb_kette_ungueltig_gruende'] = N[N.k_valid == 0].k_reason.value_counts().to_dict()
    A['nb_pol_unsicher'] = int((N.k_pol_unsicher == 1).sum())
    A['nb_gruende_kette'] = {str(g): {'gueltig': int(((N.grund_nicht_beurteilbar == g) & (N.k_valid == 1)).sum()), 'ungueltig': int(((N.grund_nicht_beurteilbar == g) & (N.k_valid == 0)).sum())}
                             for g in N.grund_nicht_beurteilbar.dropna().unique()}
    A['nb_kette_t_roh_mittel'] = float(N.k_roh.mean()) if N.k_roh.notna().any() else None
    A['nb_kette_t_amp_betrag_median'] = float(N.k_rel.abs().median()) if N.k_rel.notna().any() else None
    A['gem_kette_t_amp_betrag_median'] = float(G.k_rel.abs().median()) if G.k_rel.notna().any() else None

    # Auffaelligkeiten: groesste Abweichungen (privat), Verteilung des Anwenders
    D['absd'] = D.d_roh.abs()
    A['gross_abweichend_roh_gt_015'] = int((G.d_roh.abs() > 0.15).sum())
    A['gross_abweichend_roh_gt_010'] = int((G.d_roh.abs() > 0.10).sum())
    A['u_negativ_t_roh_anwender'] = int((G.u_roh < 0).sum())
    A['k_negativ_t_roh_kette'] = int((G.k_roh < 0).sum())
    A['k_t_amp_negativ'] = int((G.k_rel < 0).sum())
    A['u_rel_auto_negativ'] = int((G.u_rel_auto < 0).sum())
    A['bemerkungen_n'] = int(anw.bemerkung.notna().sum())
    A['bearbeitungszeit_median_s'] = float(anw.bearbeitungszeit_s.median())
    A['sd_anwender_schlaege'] = float(np.nanmean([np.std([r_[f's{q}_amp_mv'] for q in (1, 2, 3)], ddof=1) for _, r_ in G.iterrows()]))
    A['hr_median_kette'] = float(G.k_hr.median())
    # Nicht-Normalitaet / Mittelwerte
    A['mittel_diff_kontrolle_mm'] = None

    # ---- Schreiben (privat)
    ausg = D.drop(columns=['absd'])
    ausg.to_csv(OUT / 'ansichten_vergleich.csv', index=False, float_format='%.5f')
    (OUT / 'aggregate.json').write_text(json.dumps(A, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
    P.to_csv(OUT / 'paare_wiederholung.csv', index=False, float_format='%.5f')
    log('Zeilen pro Ansicht, Paare und Aggregate geschrieben')

    # ---- Abbildung (nur F:)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    cols = {'positiv': '#1f6fb4', 'negativ': '#d9541e'}
    ok = G.u_roh.notna() & G.k_roh.notna()
    for q in ('positiv', 'negativ'):
        s = G[ok & (G.qrs_richtung == q)]
        ax[0].scatter(s.k_roh, s.u_roh, s=26, color=cols[q], alpha=0.85, label=f'QRS {q} (Anwender), n={len(s)}', edgecolor='white', linewidth=0.5)
        ax[1].scatter((s.u_roh + s.k_roh) / 2, s.u_roh - s.k_roh, s=26, color=cols[q], alpha=0.85, label=f'QRS {q}', edgecolor='white', linewidth=0.5)
    lim = [min(G.u_roh.min(), G.k_roh.min()) - 0.05, max(G.u_roh.max(), G.k_roh.max()) + 0.05]
    ax[0].plot(lim, lim, color='#444', lw=1, ls='--', label='Identität')
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel('Messkette, T-Amplitude roh (mV)'); ax[0].set_ylabel('Anwender, T-Amplitude (mV)')
    ax[0].set_title('Streudiagramm (rohe, vorzeichenbehaftete Amplitude, Ableitung II)', fontsize=10)
    ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.25)
    g = A['gesamt_roh']
    ax[1].axhline(g['bias'], color='#222', lw=1.2, label=f"Bias {g['bias']:+.3f} mV")
    for v in g['loa']:
        ax[1].axhline(v, color='#222', lw=1, ls=':')
    ax[1].axhline(0, color='#888', lw=0.6)
    ax[1].axhspan(-0.05, 0.05, color='#2a9d8f', alpha=0.10, label='±0,05 mV')
    ax[1].set_xlabel('Mittel aus Anwender und Kette (mV)'); ax[1].set_ylabel('Anwender minus Kette (mV)')
    ax[1].set_title(f"Bland-Altman: Grenzen {g['loa'][0]:+.3f} bis {g['loa'][1]:+.3f} mV, n={g['n']}", fontsize=10)
    ax[1].legend(fontsize=8, loc='upper left'); ax[1].grid(alpha=0.25)
    fig.suptitle('S3 MC-MED: Referenzannotation (Anwender) gegen Messkette aw255-v3, beschreibend', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'abbildung_s3_annotation.png', dpi=150)
    plt.close(fig)
    log('Abbildung geschrieben')
    log('Fertig')


if __name__ == '__main__':
    main()
