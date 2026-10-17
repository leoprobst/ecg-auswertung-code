"""S3 MC-MED: Auswertung der zweiten Annotation (nachtraegliche Pruefung der einflussreichen H1-Primaerpaare). POST HOC, BESCHREIBEND, NICHT GEBUNDEN, KEINE BESTAETIGUNG.

Eingaben (alle privat unter <local path>): AUSWAHL_EINFLUSS.json, SCHLUESSEL_EINFLUSS.csv, die vom Anwender exportierte CSV
referenzannotation_s3_einfluss_<Datum>_<Uhrzeit>.csv; dazu der Zwischenspeicher des gebundenen Gesamtlaufs (tabellen_cache.pkl, enthaelt den Testsplit-Anteil; der SHA-256
dieses Skripts muss vor dem Lauf in koordination\\TESTZUGRIFFE.md stehen) und, falls vorhanden, die Auswertung der ersten Annotation (aggregate.json) als Massstab.

Was gerechnet wird:
  1. Leser gegen Kette JE ENDE: T-Amplitude des Lesers (Mittel der drei Schlaege, Spitze minus Linie) im Bezugssystem der Kette (QRS-relativ: mal pol der Kette) gegen t_amp der
     Kette. Verglichen werden Bias, Grenzen der Uebereinstimmung, SD der Differenz, ICC, Rangkorrelation, getrennt fuer einflussreiche Enden und Fuellenden.
  2. Leser gegen Kette JE PAAR: dT des Lesers (Ende b minus Ende a) gegen dT der Kette; Differenz, Vorzeichen, Anteil ausserhalb der Spanne, die aus der Uebereinstimmung der ersten
     Annotation zu erwarten ist (sqrt(2) mal SD der Differenz je Ende). Getrennt fuer einflussreiche Paare und Fuellpaare.
  3. H1 neu gerechnet, dT des Lesers an den einflussreichen Paaren statt dT der Kette (alles andere unveraendert: Primaerrahmen, freier Achsenabschnitt, HF-Spline, Personen-Cluster-
     bootstrap, Seed 255001, B = 2000). Varianten: Kette (Gate gegen die Bindung), ohne einflussreiche (Gate), Leser an einflussreichen, Leser an einflussreichen und nicht Lesbare
     entfallen, Leser an allen gelesenen Paaren (einflussreiche und Fuellpaare). Zu jeder Variante: wie viele Paare danach noch einflussreich sind (gleiche DFBETAS-Regel).

Aufruf (Python: <local path> -I):
  selbsttest                       Kunstdaten, kein Zugriff auf MC-MED
  auswerten [--csv PFAD]           die Auswertung; ohne --csv die juengste referenzannotation_s3_einfluss_*.csv im Ordner
Ausgaben (privat, Unterordner auswertung_einfluss\\): ende_vergleich.csv, paare_vergleich.csv, ergebnisse_einfluss.json, abbildung_einfluss.png, protokoll.txt.
Die Ergebnisse sind nur Beschreibung. Der Leser sieht 10 s, die Kette misst ein 60-s-Fenster; Unterschiede beider Verfahren sind nicht nur Messfehler der Kette.
Regeln: ein Prozess, kleiner Speicherbedarf, keine Loeschbefehle, keine Kennungen in den Ausgaben mit Zusammenfassung (die privaten CSV tragen Paar-IDs, keine Besuchskennungen).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, math, hashlib, argparse, time, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import s3_annotation_einfluss as E          # Auswahlregel, Primaerrahmen, DFBETAS-Regel (gleiche Rechenvorschrift wie bei der Ziehung)

ANN = E.ANN
OUT = ANN / 'auswertung_einfluss'
ERSTE_AGG = Path(str(data_path("s3.annotation_aggregate_json")))
SEED_H1 = 255001
LOGF = None


def log(*a):
    s = dt.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        with open(LOGF, 'a', encoding='utf-8') as f:
            f.write(s + '\n')


def sicher(o):
    if isinstance(o, dict):
        return {str(k): sicher(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sicher(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


# =========================================================================== Statistik (beschreibend)
def vergleich(u, k):
    """u = Leser, k = Kette (gepaart, endlich); d = u - k."""
    from scipy.stats import spearmanr, t as tdist
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
    msb = 2 * np.var(m, ddof=1)
    msw = float(np.sum(d ** 2) / (2 * n))
    icc = float((msb - msw) / (msb + msw)) if (msb + msw) > 0 else float('nan')
    return {'n': int(n), 'bias': bias, 'bias_ci': [bias - tq * se, bias + tq * se], 'sd_diff': sd, 'loa': [bias - 1.96 * sd, bias + 1.96 * sd], 'icc11': icc,
            'spearman': float(spearmanr(u, k).correlation) if n >= 3 and np.ptp(u) > 0 and np.ptp(k) > 0 else float('nan'),
            'median_diff': float(np.median(d)), 'median_abs_diff': float(np.median(np.abs(d))), 'max_abs_diff': float(np.max(np.abs(d))),
            'innerhalb_005': int((np.abs(d) <= 0.05).sum()), 'mittel_leser': float(u.mean()), 'mittel_kette': float(k.mean())}


# =========================================================================== Einlesen
def lies_leser(pfad, info, n_ansichten):
    """Exportierte CSV der Seite: Hash, Vollstaendigkeit, Statusspalten; Rueckgabe DataFrame je Ansicht mit u_roh (Mittel der drei Schlaege, mV) oder NaN."""
    a = pd.read_csv(pfad)
    if a.auswahl_hash.nunique() != 1 or a.auswahl_hash.iloc[0] != info['auswahl_hash']:
        raise SystemExit('Auswahl-Hash der Leser-CSV stimmt nicht mit AUSWAHL_EINFLUSS.json ueberein')
    if sorted(a.ansicht_nr.tolist()) != list(range(1, n_ansichten + 1)):
        raise SystemExit(f'Leser-CSV enthaelt nicht genau die Ansichten 1 bis {n_ansichten}')
    a = a.sort_values('ansicht_nr').reset_index(drop=True)
    a['gemessen'] = (a.status == 'gemessen') & (a.n_schlaege == 3) & a.mittel_amp_mv.notna()
    a['u_roh'] = np.where(a.gemessen, a.mittel_amp_mv, np.nan)
    return a


# =========================================================================== Ende- und Paarvergleich
def enden_tabelle(K, L):
    """Eine Zeile je Paar und Ende: Leser im Bezugssystem der Kette (mal pol_kette) gegen t_amp der Kette."""
    cols = ['ansicht_nr', 'status', 'qrs_richtung', 'grund_nicht_beurteilbar', 'gemessen', 'u_roh', 'bearbeitungszeit_s']
    En = K.merge(L[cols], on='ansicht_nr', validate='m:1')
    En['u_rel'] = En.u_roh * En.pol_kette
    En['k_rel'] = En.t_amp_kette
    En['d_ende'] = En.u_rel - En.k_rel
    En['qrs_leser_gleich_pol_kette'] = np.where(En.qrs_richtung.isin(['positiv', 'negativ']), En.qrs_richtung.map({'positiv': 1, 'negativ': -1}) == En.pol_kette, np.nan)
    return En


def paare_tabelle(En, sd_ende_erste=None):
    """Eine Zeile je Paar: dT des Lesers (Ende b minus Ende a) gegen dT der Kette."""
    zeilen = []
    for pid, g in En.groupby('paar_id', sort=True):
        a, b = g[g.ende == 'a'].iloc[0], g[g.ende == 'b'].iloc[0]
        dl = b.u_rel - a.u_rel
        r = {'paar_id': pid, 'einflussreich': a.einflussreich, 'CSN': a.CSN, 'ta': a.ta, 'tb': a.tb, 'split': a.split, 'dk': a.dk, 'dfbetas': a.dfbetas,
             'ansicht_a': a.ansicht_nr, 'ansicht_b': b.ansicht_nr, 'k_rel_a': a.k_rel, 'k_rel_b': b.k_rel, 'u_rel_a': a.u_rel, 'u_rel_b': b.u_rel,
             'dT_kette': a.dT_kette, 'dT_leser': dl, 'beide_gelesen': bool(a.gemessen and b.gemessen), 'diff_dT': dl - a.dT_kette,
             'vorzeichen_gleich': (np.sign(dl) == np.sign(a.dT_kette)) if np.isfinite(dl) else np.nan}
        if sd_ende_erste:
            spanne = 1.96 * math.sqrt(2) * sd_ende_erste
            r['spanne_erwartet'] = spanne
            r['ausserhalb_spanne'] = bool(abs(r['diff_dT']) > spanne) if np.isfinite(dl) else np.nan
        zeilen.append(r)
    return pd.DataFrame(zeilen)


def zusammenfassung_paare(P, sd_ende_erste=None):
    out = {}
    for nm, g in (('einflussreich', P[P.einflussreich == 'ja']), ('fuell', P[P.einflussreich == 'nein']), ('alle', P)):
        gg = g[g.beide_gelesen]
        o = {'paare': int(len(g)), 'beide_enden_gelesen': int(len(gg)), 'vergleich_dT': vergleich(gg.dT_leser, gg.dT_kette)}
        if len(gg):
            o['vorzeichen_gleich'] = int(gg.vorzeichen_gleich.astype(bool).sum())
            o['median_abs_diff_dT'] = float(gg.diff_dT.abs().median())
            o['mittel_diff_dT'] = float(gg.diff_dT.mean())
            if sd_ende_erste:
                o['ausserhalb_erwarteter_spanne'] = int(gg.ausserhalb_spanne.astype(bool).sum())
                o['spanne_erwartet_dT'] = float(1.96 * math.sqrt(2) * sd_ende_erste)
        out[nm] = o
    return out


# =========================================================================== H1 neu mit dT des Lesers
def zeilen_der_paare(q, P):
    """Zeile im Primaerrahmen je Paar (CSN gleich, ta und tb auf 10 ms)."""
    idx = {}
    for r in P.itertuples():
        m = q[(q.CSN == r.CSN) & ((q.ta - r.ta).abs() < 0.01) & ((q.tb - r.tb).abs() < 0.01)]
        if len(m) != 1:
            raise ValueError(f'Paar {r.paar_id}: {len(m)} Zeilen im Primaerrahmen (erwartet 1)')
        idx[r.paar_id] = int(m.index[0])
    return idx


def h1_variante(q, kov, B, dT_neu, behalten=None, n_boot=None):
    """H1-Primaermodell auf q mit ersetzter Ergebnisgroesse dT_neu (Serie, Index wie q); behalten: boolesche Maske der Zeilen. Rueckgabe dict."""
    qq = q.copy()
    qq['dT'] = dT_neu
    if behalten is not None:
        qq = qq[behalten].reset_index(drop=True)
    e = B.schaetze(qq, 'dT', 'dk', kov, B.CFG, B=n_boot) if n_boot else B.schaetze(qq, 'dT', 'dk', kov, B.CFG)
    dfb, gr = E.dfbetas_regel(qq, kov)
    e = dict(e)
    e.update(n_einflussreich_danach=int((dfb > gr).sum()), dfbetas_max_danach=float(dfb.max()), grenze_danach=gr)
    return e


def h1_neu(q, kov, B, P, n_boot=None, n_boot_ohne=None):
    """Alle Varianten. P: Paartabelle mit paar_id, einflussreich, dT_leser, beide_gelesen, CSN, ta, tb."""
    zl = zeilen_der_paare(q, P)
    P = P.assign(zeile=P.paar_id.map(zl))
    infl = P[P.einflussreich == 'ja']
    dfb, gr = E.dfbetas_regel(q, kov)
    assert set(infl.zeile) == set(int(i) for i in np.where(dfb > gr)[0]), 'einflussreiche Paare im Schluessel und im Primaerrahmen stimmen nicht ueberein'
    res = {}
    res['kette'] = h1_variante(q, kov, B, q.dT, n_boot=n_boot)
    keep = pd.Series(True, index=q.index)
    keep[infl.zeile.to_numpy()] = False
    res['ohne_einflussreiche_kette'] = h1_variante(q, kov, B, q.dT, behalten=keep, n_boot=n_boot_ohne)
    ersatz = q.dT.copy()
    les = infl[infl.beide_gelesen]
    ersatz[les.zeile.to_numpy()] = les.dT_leser.to_numpy()
    res['leser_an_einflussreichen'] = h1_variante(q, kov, B, ersatz, n_boot=n_boot)
    res['leser_an_einflussreichen']['paare_ersetzt'] = int(len(les))
    res['leser_an_einflussreichen']['paare_einflussreich'] = int(len(infl))
    keep2 = pd.Series(True, index=q.index)
    keep2[infl[~infl.beide_gelesen].zeile.to_numpy()] = False
    res['leser_an_einflussreichen_nicht_lesbare_entfallen'] = h1_variante(q, kov, B, ersatz, behalten=keep2, n_boot=n_boot)
    allg = P[P.beide_gelesen]
    ersatz2 = q.dT.copy()
    ersatz2[allg.zeile.to_numpy()] = allg.dT_leser.to_numpy()
    res['leser_an_allen_gelesenen_paaren'] = h1_variante(q, kov, B, ersatz2, n_boot=n_boot)
    res['leser_an_allen_gelesenen_paaren']['paare_ersetzt'] = int(len(allg))
    return res


# =========================================================================== Abbildung
def abbildung(En, P, pfad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    farben = {'ja': '#d9541e', 'nein': '#1f6fb4'}
    namen = {'ja': 'einflussreiche Paare', 'nein': 'Fuellpaare'}
    for k, g in En[En.gemessen].groupby('einflussreich'):
        ax[0].scatter(g.k_rel, g.u_rel, s=26, color=farben[k], alpha=0.85, edgecolor='white', linewidth=0.5, label=f'Enden {namen[k]}, n={len(g)}')
    lim = [np.nanmin([En.k_rel.min(), En.u_rel.min()]) - 0.05, np.nanmax([En.k_rel.max(), En.u_rel.max()]) + 0.05]
    ax[0].plot(lim, lim, color='#444', lw=1, ls='--', label='Identität')
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel('Messkette, T-Amplitude QRS-relativ (mV)')
    ax[0].set_ylabel('Leser, T-Amplitude QRS-relativ (mV)')
    ax[0].set_title('Je Ende', fontsize=10)
    ax[0].legend(fontsize=8, loc='upper left')
    ax[0].grid(alpha=0.25)
    for k, g in P[P.beide_gelesen].groupby('einflussreich'):
        ax[1].scatter(g.dT_kette, g.dT_leser, s=34, color=farben[k], alpha=0.85, edgecolor='white', linewidth=0.5, label=f'{namen[k]}, n={len(g)}')
    v = P[['dT_kette', 'dT_leser']].to_numpy(float)
    lim2 = [np.nanmin(v) - 0.05, np.nanmax(v) + 0.05]
    ax[1].plot(lim2, lim2, color='#444', lw=1, ls='--', label='Identität')
    ax[1].axhline(0, color='#888', lw=0.6)
    ax[1].axvline(0, color='#888', lw=0.6)
    ax[1].set_xlim(lim2)
    ax[1].set_ylim(lim2)
    ax[1].set_xlabel('dT Messkette (mV)')
    ax[1].set_ylabel('dT Leser (mV)')
    ax[1].set_title('Je Paar (Ende b minus Ende a)', fontsize=10)
    ax[1].legend(fontsize=8, loc='upper left')
    ax[1].grid(alpha=0.25)
    fig.suptitle('S3 MC-MED: nachträgliche Prüfung der einflussreichen Paare, Leser gegen Messkette (beschreibend, post hoc)', fontsize=11)
    fig.tight_layout()
    fig.savefig(pfad, dpi=150)
    plt.close(fig)


# =========================================================================== Ablauf
def auswerten_kern(K, L, q, kov, B, sd_ende_erste=None, n_boot=None, n_boot_ohne=None):
    En = enden_tabelle(K, L)
    P = paare_tabelle(En, sd_ende_erste)
    A = {'enden': {}, 'paare': zusammenfassung_paare(P, sd_ende_erste)}
    for nm, g in (('einflussreich', En[En.einflussreich == 'ja']), ('fuell', En[En.einflussreich == 'nein']), ('alle', En)):
        gg = g[g.gemessen]
        A['enden'][nm] = {'enden': int(len(g)), 'gelesen': int(len(gg)), 'nicht_beurteilbar_oder_offen': int(len(g) - len(gg)), 'vergleich': vergleich(gg.u_rel, gg.k_rel),
                          'qrs_leser_gleich_pol_kette': int(np.nansum(g.qrs_leser_gleich_pol_kette.astype(float))), 'qrs_leser_angegeben': int(g.qrs_leser_gleich_pol_kette.notna().sum())}
    A['h1'] = h1_neu(q, kov, B, P, n_boot=n_boot, n_boot_ohne=n_boot_ohne)
    return En, P, A


def pruefe_eintrag_eigen():
    h = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    zeilen = [z for z in E.TESTZUGRIFFE.read_text(encoding='utf-8').splitlines() if h in z and 'Beginn Auswertung zweite Annotation' in z]
    if not zeilen:
        raise SystemExit(f'Kein Eintrag "Beginn Auswertung zweite Annotation" mit dem Skript-Hash {h} in {E.TESTZUGRIFFE}; zuerst anhaengen '
                         '(der Zwischenspeicher enthaelt den Testsplit-Anteil)')
    return h


def lauf(csv_pfad):
    global LOGF
    OUT.mkdir(parents=True, exist_ok=True)
    LOGF = OUT / 'protokoll.txt'
    h = pruefe_eintrag_eigen()
    info, K = E._lies_schluessel()
    if csv_pfad is None:
        c = sorted(ANN.glob('referenzannotation_s3_einfluss_*.csv'))
        if not c:
            raise SystemExit('keine referenzannotation_s3_einfluss_*.csv im Ordner ' + str(ANN))
        csv_pfad = c[-1]
    csv_pfad = Path(csv_pfad)
    log('Start; Skript sha256', h[:16], '; Leser-CSV', csv_pfad.name, 'sha256', E.sha_datei(csv_pfad)[:16])
    L = lies_leser(csv_pfad, info, int(K.ansicht_nr.nunique()))
    log('Hash und Vollstaendigkeit der Leser-CSV geprueft; gelesen', int(L.gemessen.sum()), 'von', len(L), 'Ansichten')
    import s3_auswertung_v2_2 as B
    T = E.lade_tabellen()
    q, kov, _z = E.primaerrahmen(T['Q'], B)
    soll = json.loads(E.GEBUNDEN_JSON.read_text(encoding='utf-8'))
    sd1 = None
    if ERSTE_AGG.exists():
        try:
            sd1 = json.loads(ERSTE_AGG.read_text(encoding='utf-8'))['gesamt_rel_auto']['sd_diff']
            log('Massstab aus der ersten Annotation: SD der Differenz je Ende (QRS-relativ)', sd1)
        except Exception as e:
            log('Massstab der ersten Annotation nicht lesbar:', repr(e)[:120])
    En, P, A = auswerten_kern(K, L, q, kov, B, sd_ende_erste=sd1, n_boot=None, n_boot_ohne=max(200, B.CFG['bootstrap_n'] // 4))
    # Gates gegen die Bindung
    sp, sh = soll['h1']['primaer'], soll['h1']['sensitivitaeten']['einfluss']['ohne_einflussreiche']
    g1, g2 = A['h1']['kette'], A['h1']['ohne_einflussreiche_kette']
    A['gate'] = {'kette_gleich_gebunden': bool(g1['n'] == sp['n'] and abs(g1['beta'] - sp['beta']) < 1e-12 and abs(g1['ci_lo'] - sp['ci_lo']) < 1e-12 and abs(g1['ci_hi'] - sp['ci_hi']) < 1e-12),
                 'ohne_einflussreiche_gleich_gebunden': bool(g2['n'] == sh['n'] and abs(g2['beta'] - sh['beta']) < 1e-12 and abs(g2['ci_lo'] - sh['ci_lo']) < 1e-12 and abs(g2['ci_hi'] - sh['ci_hi']) < 1e-12)}
    log('Gate gegen die Bindung:', A['gate'])
    if not all(A['gate'].values()):
        log('ACHTUNG: Gate nicht bestanden; die Varianten sind nicht auf die Bindung bezogen')
    A['kopf'] = 'post hoc, nach Sicht der Ergebnisse, beschreibend, nicht gebunden, keine Bestaetigung'
    A['skript_sha256'] = h
    A['leser_csv_sha256'] = E.sha_datei(csv_pfad)
    A['auswahl_hash'] = info['auswahl_hash']
    A['bearbeitungszeit_summe_min'] = float(L.bearbeitungszeit_s.sum() / 60.0)
    En.to_csv(OUT / 'ende_vergleich.csv', index=False, float_format='%.5f')
    P.to_csv(OUT / 'paare_vergleich.csv', index=False, float_format='%.5f')
    (OUT / 'ergebnisse_einfluss.json').write_text(json.dumps(sicher(A), indent=1, ensure_ascii=False), encoding='utf-8')
    abbildung(En, P, OUT / 'abbildung_einfluss.png')
    for k, v in A['h1'].items():
        log(f'H1 {k}: n={v["n"]} beta={v["beta"]:.5f} CI [{v["ci_lo"]:.5f}; {v["ci_hi"]:.5f}] p={v["p_boot"]:.3f} einflussreich danach={v["n_einflussreich_danach"]}')
    log('Fertig; Ausgaben in', OUT)


# =========================================================================== Selbsttest (Kunstdaten)
def synth(n_boot=200):
    """Kunstdaten: Kette liefert an den Hebelpunkten falsche Aenderungen; der Leser liest die wahre (Null-)Aenderung."""
    import s3_auswertung_v2_2 as B
    d, kov = E.kunst_rahmen()
    # t_amp so setzen, dass dT = t_amp_b - t_amp_a
    d['t_amp_a'] = 0.30
    d['t_amp_b'] = d.t_amp_a + d.dT
    d['t_roh_a'], d['t_roh_b'] = d.t_amp_a, d.t_amp_b
    dfb, gr = E.dfbetas_regel(d, kov)
    r1 = pd.DataFrame({'CSN': [0], 't_mitte': [0.0]})
    sel = E.waehle(d, dfb, gr, r1)
    rows = []
    for r in sel['enden'].itertuples():
        i = int(r.zeile)
        e = r.ende
        pr = sel['paare'].set_index('paar_id').loc[r.paar_id]
        rows.append({'ansicht_nr': int(r.ansicht_nr), 'paar_id': r.paar_id, 'ende': e, 'einflussreich': r.einflussreich, 'CSN': int(r.CSN), 'MRN': int(r.MRN), 'split': r.split,
                     'ta': float(d.ta[i]), 'tb': float(d.tb[i]), 'seg': 1, 't_mitte': r.t_mitte, 'w0': float(d['w0_' + e][i]), 'w1': float(d['w1_' + e][i]), 'c_kette': float(d['c_' + e][i]),
                     'pol_kette': int(d['pol_' + e][i]), 't_amp_kette': float(d['t_amp_' + e][i]), 't_roh_kette': float(d['t_roh_' + e][i]), 'qrs_ms_kette': 90.0, 'hf': 70.0,
                     'ka': float(d.ka[i]), 'kb': float(d.kb[i]), 'dk': float(d.dk[i]), 'dT_kette': float(d.dT[i]), 'dfbetas': float(pr.dfbetas),
                     'ueberschneidung_runde1': r.ueberschneidung_runde1, 'besuch_in_runde1': r.besuch_in_runde1})
    K = pd.DataFrame(rows, columns=E.SPALTEN)
    return B, d, kov, K, sel


def leser_csv_kunst(K, wahrheit, rausch=0.01, seed=5, nicht_beurteilbar=()):
    """Kuenstliche Leser-Ergebnisse: u_rel = wahrheit(Zeile) + Rauschen; u_roh = u_rel / pol; fuer ansicht_nr in nicht_beurteilbar: Status nicht_beurteilbar."""
    rng = np.random.default_rng(seed)
    z = []
    for nr, g in K.groupby('ansicht_nr'):
        r = g.iloc[0]
        u_rel = wahrheit(r) + rng.normal(0, rausch)
        nb = nr in nicht_beurteilbar
        z.append({'auswahl_hash': 'h' * 64, 'ansicht_nr': int(nr), 'status': 'nicht_beurteilbar' if nb else 'gemessen', 'qrs_richtung': 'positiv' if r.pol_kette > 0 else 'negativ',
                  'grund_nicht_beurteilbar': 'artefakt' if nb else '', 'n_schlaege': 0 if nb else 3, 'mittel_amp_mv': np.nan if nb else u_rel * r.pol_kette, 'bearbeitungszeit_s': 20.0})
    return pd.DataFrame(z)


def selbsttest():
    ok = 0
    B, d, kov, K, sel = synth()
    infl_enden = K[K.einflussreich == 'ja']
    # Wahrheit: es gibt keine Aenderung der T-Amplitude -> Ende a: 0,30; Ende b: 0,30 (die Kette liest an den Hebelpunkten falsch)
    wahr = lambda r: 0.30
    L = leser_csv_kunst(K, wahr)
    L = lies_leser_aus_frame(L)
    # 1. Ende- und Paarvergleich
    En, P, A = auswerten_kern(K, L, d.reset_index(drop=True), kov, B, sd_ende_erste=0.05, n_boot=200, n_boot_ohne=200)
    assert len(En) == len(K) and len(P) == K.paar_id.nunique()
    assert A['enden']['einflussreich']['gelesen'] == len(infl_enden)
    pi = P[P.einflussreich == 'ja']
    heb = pi[pi.dT_kette.abs() > 0.5]
    assert (pi.dT_leser.abs() < 0.08).all() and len(heb) >= 5, 'Leser liest Null, Kette liest die Hebel'
    assert np.allclose(pi.diff_dT, pi.dT_leser - pi.dT_kette)
    assert A['paare']['einflussreich']['ausserhalb_erwarteter_spanne'] >= len(heb)
    ok += 1
    # 2. H1: Leser an einflussreichen nimmt den Hebel weg und liegt nahe "ohne einflussreiche"
    h = A['h1']
    assert h['kette']['beta'] > 0.05 and abs(h['leser_an_einflussreichen']['beta']) < 0.05 and abs(h['ohne_einflussreiche_kette']['beta']) < 0.05
    assert h['leser_an_einflussreichen']['paare_ersetzt'] == len(pi) and h['leser_an_einflussreichen']['n'] == h['kette']['n']
    assert h['ohne_einflussreiche_kette']['n'] == h['kette']['n'] - len(pi)
    assert h['kette']['n_einflussreich_danach'] == len(pi) and h['leser_an_einflussreichen']['n_einflussreich_danach'] < len(pi)
    ok += 1
    # 3. Ein einflussreiches Paar nicht lesbar: nur dieses bleibt bei der Kette bzw. entfaellt
    nb_nr = int(infl_enden.ansicht_nr.iloc[0])
    L2 = lies_leser_aus_frame(leser_csv_kunst(K, wahr, nicht_beurteilbar=(nb_nr,)))
    En2, P2, A2 = auswerten_kern(K, L2, d.reset_index(drop=True), kov, B, n_boot=200, n_boot_ohne=200)
    h2 = A2['h1']
    assert h2['leser_an_einflussreichen']['paare_ersetzt'] == len(pi) - 1 and h2['leser_an_einflussreichen_nicht_lesbare_entfallen']['n'] == h2['kette']['n'] - 1
    assert A2['paare']['einflussreich']['beide_enden_gelesen'] == len(pi) - 1
    ok += 1
    # 4. Falsche Zuordnung (Paar-Schluessel passt nicht zum Primaerrahmen) wird erkannt
    P_bad = P.copy()
    P_bad.loc[0, 'ta'] += 100.0
    try:
        zeilen_der_paare(d.reset_index(drop=True), P_bad)
        raise AssertionError('Fehler erwartet')
    except ValueError:
        ok += 1
    # 5. Leser stimmt mit der Kette ueberein (Ende-Ebene): Bias 0, kleine SD
    L3 = lies_leser_aus_frame(leser_csv_kunst(K, lambda r: r.t_amp_kette, rausch=0.01))
    En3, P3, A3 = auswerten_kern(K, L3, d.reset_index(drop=True), kov, B, n_boot=200, n_boot_ohne=200)
    v = A3['enden']['alle']['vergleich']
    assert abs(v['bias']) < 0.01 and v['sd_diff'] < 0.02 and v['icc11'] > 0.9
    assert abs(A3['h1']['leser_an_einflussreichen']['beta'] - A3['h1']['kette']['beta']) < 0.05
    ok += 1
    # 6. Hash und Vollstaendigkeit der CSV werden geprueft
    import tempfile
    info = {'auswahl_hash': 'h' * 64}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x.csv'
        c = leser_csv_kunst(K, wahr)
        c.to_csv(p, index=False)
        assert len(lies_leser(p, info, int(K.ansicht_nr.nunique()))) == K.ansicht_nr.nunique()
        c.assign(auswahl_hash='x' * 64).to_csv(p, index=False)
        try:
            lies_leser(p, info, int(K.ansicht_nr.nunique()))
            raise AssertionError('Fehler erwartet')
        except SystemExit:
            pass
        c.iloc[:-1].to_csv(p, index=False)
        try:
            lies_leser(p, info, int(K.ansicht_nr.nunique()))
            raise AssertionError('Fehler erwartet')
        except SystemExit:
            pass
    ok += 1
    # 7. Abbildung entsteht, JSON-tauglich
    with tempfile.TemporaryDirectory() as td:
        abbildung(En, P, Path(td) / 'a.png')
        assert (Path(td) / 'a.png').stat().st_size > 5000
        json.dumps(sicher(A))
    ok += 1
    print(f'Selbsttest {ok} von {ok} bestanden')


def lies_leser_aus_frame(a):
    """Wie lies_leser, aber aus einem DataFrame (Kunstdaten ohne Datei)."""
    a = a.sort_values('ansicht_nr').reset_index(drop=True)
    a['gemessen'] = (a.status == 'gemessen') & (a.n_schlaege == 3) & a.mittel_amp_mv.notna()
    a['u_roh'] = np.where(a.gemessen, a.mittel_amp_mv, np.nan)
    return a


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('modus', choices=['selbsttest', 'auswerten'])
    ap.add_argument('--csv', default=None)
    a = ap.parse_args(argv)
    if a.modus == 'selbsttest':
        selbsttest()
    else:
        lauf(a.csv)


if __name__ == '__main__':
    main(sys.argv[1:])
