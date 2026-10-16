#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S5-MC-MED Diagnose AW-251 (post hoc, nicht gebunden): Warum sind in Beziehung A nur 28,3 % der Besuche identifizierbar?

Ändert weder das gebundene Urteil (H0 verfehlt, B gesperrt) noch eine gebundene Datei. Das gebundene Skript v17 wird nur importiert (Funktionen werte_aus,
schaetze, nk2_zuordnung, anteil_bootstrap, ...), nie verändert. Gerechnet wird ausschließlich auf Train (split = 'train'); Val und Test werden nicht gelesen.
Die Regeln stehen vor jeder Rechnung in nachrechnung/S5_MCMED_DIAGNOSE_AW251_REGELN.md (mit SHA-256 dieses Skripts); ohne passenden Eintrag verweigert
das Skript jeden Datenmodus.

Aufruf (Python: <local path> -I):
  selbsttest                          Kunstdatentests (Sol-Fälle, Rand, Mindestfehler, Breite, Zufallskurven gegen v17.schaetze), keine echten Daten
  rechnen --phasen haupt,nk1,nk2      Hauptlauf, NK1, NK2 (je Phase Zwischenstand in <local path>)
  rechnen --phasen nk3                NK3 (6 Dichtestufen x 20 Wiederholungen, Rng-Zustand je Stufe gesichert, wiederaufnehmbar)
  auswerten                           Aggregation aller Zwischenstände -> ergebnis.json (maskiert), tabellen.md
  bericht --prosa DATEI               S5_MCMED_DIAGNOSE_AW251.md aus tabellen.md und einem Prosa-Text zusammensetzen
Ausgaben: nur Aggregate; Zählungen von 1 bis 9 als '<10'; keine Besuchs- oder Patientenkennungen. Keine Löschbefehle.

Nachträge N1 (Lage des gewählten Minimums ohne Zonenregel, auch in den Nullkontrollen und je Dichtestufe) und N2 (Abstand genau 4 min) wurden NACH der ersten Sicht der
Aggregate ergänzt; sie ändern keine Rechenphase, nur die Aggregation (Regeldatei, Abschnitt 9). Die Rechenphasen haupt, nk1, nk2, nk3 sind seit der Regelfestlegung unverändert.

Zerlegungsbits je Besuch (uint8): 1 = (i) Mindestfehler > 5/min, 2 = (ii) Zone berührt Rand, 4 = (iii) Zone nicht zusammenhängend, 8 = (iv) Breite > 120 min,
16 = identifizierbar unter Variante C1, 32 = identifizierbar unter Variante C2, 64 = Lage |Offset| <= 2 min, 128 = im Nenner (>= 3 vollständige Einheiten).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os, json, time, hashlib, importlib.util, argparse, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np

HIER = Path(__file__).resolve().parent
V17 = HIER / 's5_mcmed_auswertung_v17.py'
PRIV = Path(str(data_path("s5.analyse_mc_med_s5")))
FREEZE = PRIV / 'freeze_v16.json'
BOUND_JSON = PRIV / 'auswertung_v16' / 'bestaetigung.json'      # nur lesen (Nachvollzug)
OUT = PRIV / 'diagnose_aw251'
ZW = OUT / 'zwischen'
NACH = Path(str(data_path("s5.nachrechnung")))
REGELN = NACH / 'S5_MCMED_DIAGNOSE_AW251_REGELN.md'
ERGEBNIS_MD = NACH / 'S5_MCMED_DIAGNOSE_AW251.md'

GA = 60                                   # Gitterhalbweite A (min), Schritt 2 min (gebunden)
B1, B2, B3, B4, OK1, OK2, LAGE, NENN = 1, 2, 4, 8, 16, 32, 64, 128
GRUND = ['i', 'ii', 'iii', 'iv']
GRUND_TEXT = {'i': 'Mindestfehler > 5/min', 'ii': 'Zone berührt Rand', 'iii': 'Zone nicht zusammenhängend', 'iv': 'Breite > 120 min', 'v': 'identifizierbar'}
LAGE_MIN = 2.0                            # Lagefenster A (min), wie gebunden


# =========================================================================== v17 laden (nur Import)
def lade_v17():
    sp = importlib.util.spec_from_file_location('s5v17', V17)
    A = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(A)
    fz = json.loads(FREEZE.read_text(encoding='utf-8'))
    for k in ('rr_cv_max', 'abdeckung_A_min_s'):
        A.CFG[k] = fz['cfg'][k]                       # eingefrorene Erkennerwerte (wie die gebundene Pipeline)
    if A.erkenner_version() != fz['erkenner_version']:
        raise SystemExit('Erkennerversion weicht vom Einfrieren ab.')
    if A.code_hash() != fz['code_sha256']:
        raise SystemExit('v17 weicht vom eingefrorenen Hash ab.')
    return A, fz


def sha_datei(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pruefe_regeln():
    """Datenmodi nur, wenn die Regeldatei existiert und den SHA-256 dieses Skripts enthält."""
    if not REGELN.exists():
        raise SystemExit('REGELN fehlen: erst S5_MCMED_DIAGNOSE_AW251_REGELN.md schreiben.')
    s = sha_datei(__file__)
    if s not in REGELN.read_text(encoding='utf-8'):
        raise SystemExit('REGELN passen nicht zum Skript (SHA-256 fehlt in der Regeldatei).')
    return s


# =========================================================================== Zerlegung einer Verlustkurve
def zerlege(loss, D, cfg):
    """Zerlegung einer Median-Fehlerkurve (nur Besuche im Nenner). Spiegelt v17.schaetze und ergänzt die Komponente um das gewählte Minimum.
    Rückgabe: dict mit Bits und Zusatzgrößen."""
    rg = cfg['A']
    step = float(D[1] - D[0])
    loss = np.asarray(loss, float)
    lo = float(np.min(loss))
    nm = loss <= lo + cfg['nahe_optimal_bpm']
    idx = np.flatnonzero(nm)
    near = D[idx]
    opts = D[np.isclose(loss, lo, rtol=0, atol=1e-10)]
    offset = float(sorted(opts, key=lambda x: (abs(x), x))[0])
    ic = int(np.flatnonzero(D == offset)[0])
    breite = float(near[-1] - near[0])
    zus = bool(np.all(np.isclose(np.diff(near), step)))
    r1 = lo > rg['minfehler_max']
    r2 = not bool(near[0] > D[0] and near[-1] < D[-1])
    r3 = not zus
    r4 = breite > rg['breite_max_min']
    a = ic
    while a > 0 and nm[a - 1]:
        a -= 1
    b = ic
    while b < len(D) - 1 and nm[b + 1]:
        b += 1
    cw = float(D[b] - D[a])
    ok_bound = not (r1 or r2 or r3 or r4)
    ok_c1 = not (r1 or r2 or r4)                                              # nur die Zusammenhangsregel entfällt
    ok_c2 = bool((not r1) and D[a] > D[0] and D[b] < D[-1] and cw <= rg['breite_max_min'])   # alles auf der Komponente um das gewählte Minimum
    ncomp = 1 + int(np.sum(np.diff(idx) > 1))
    fremd = idx[(idx < a) | (idx > b)]
    abst2 = float(np.min(np.abs(D[fremd] - offset))) if len(fremd) else float('nan')
    bits = (B1 if r1 else 0) | (B2 if r2 else 0) | (B3 if r3 else 0) | (B4 if r4 else 0)
    bits |= (OK1 if ok_c1 else 0) | (OK2 if ok_c2 else 0) | (LAGE if abs(offset) <= LAGE_MIN else 0) | NENN
    return {'bits': bits, 'ok_bound': ok_bound, 'ok_c1': ok_c1, 'ok_c2': ok_c2, 'lo': lo, 'offset': offset, 'breite': breite, 'cw': cw,
            'ncomp': ncomp, 'abst2': abst2, 'zus': zus}


def erster_grund(bits):
    """Erster verletzter Grund in fester Reihenfolge i, ii, iii, iv; sonst 'v' (identifizierbar). bits: Zerlegungsbits."""
    for g, b in zip(GRUND, (B1, B2, B3, B4)):
        if bits & b:
            return g
    return 'v'


def werte_zerlegt(A, besuche, extras=False, **kw):
    """Ruft v17.werte_aus (A, G = 60) auf und zerlegt jeden Besuch im Nenner. Stellt je Besuch sicher, dass die gebundene Entscheidung r['ok'] reproduziert wird."""
    res, D = A.werte_aus(besuche, 'A', GA, A.CFG, **kw)
    N = len(res)
    code = np.zeros(N, np.uint8)
    ex = None
    if extras:
        ex = {k: np.full(N, np.nan) for k in ('hf_sd', 'lo', 'offset', 'breite', 'cw', 'ncomp', 'abst2', 'n')}
    for i, r in enumerate(res):
        if not r.get('nenner'):
            continue
        z = zerlege(r['_loss'], D, A.CFG)
        if bool(z['ok_bound']) != bool(r['ok']):
            raise SystemExit('Zerlegung weicht von v17.schaetze ab (interner Fehler).')
        if bool(z['zus']) != (z['ncomp'] == 1):
            raise SystemExit('Zusammenhang und Komponentenzahl widersprüchlich.')
        code[i] = z['bits']
        if extras:
            for k in ('lo', 'offset', 'breite', 'cw', 'ncomp', 'abst2'):
                ex[k][i] = z[k]
            ex['hf_sd'][i] = r['hf_sd']
            ex['n'][i] = r['n']
    return code, ex


# =========================================================================== Daten (nur Train, nur Beziehung A)
def lade_train(A, ev):
    """Wie v17.besuche_aus_duckdb für split = 'train', aber nur Monitorreferenz und Einheiten (Beziehung A braucht keine Chartwerte); gleiche Besuchsmenge
    und Reihenfolge (nach CSN). Die DuckDB wird schreibgeschützt geöffnet."""
    import duckdb
    c = duckdb.connect(str(PRIV / 's5_privat.duckdb'), read_only=True)
    c.execute("SET memory_limit='900MB'; SET threads=2")
    cfg = A.CFG
    split = 'train'
    vis = c.execute("SELECT CSN,MRN,r,d FROM visit WHERE split=? AND r IS NOT NULL AND d IS NOT NULL AND CSN IN "
                    "(SELECT DISTINCT CSN FROM units_v16 WHERE ev=?) ORDER BY CSN", [split, ev]).fetchall()
    ids = np.array([v[0] for v in vis], dtype=np.int64)
    d = c.execute("SELECT CSN,t,Value FROM num WHERE Measure='HR' AND Source='Monitor' AND Value BETWEEN ? AND ? "
                  "AND CSN IN (SELECT CSN FROM visit WHERE split=?) ORDER BY CSN,t", [cfg['ref_wert_min'], cfg['ref_wert_max'], split]).fetchnumpy()
    csn = np.asarray(d['CSN'], dtype=np.int64)
    keep = np.isin(csn, ids)
    csn, tt_all, vv_all = csn[keep], np.asarray(d['t'], float)[keep], np.asarray(d['Value'], float)[keep]
    del d
    mon = {}
    if len(csn):
        grenzen = np.flatnonzero(np.diff(csn) != 0) + 1
        starts = np.concatenate([[0], grenzen])
        ends = np.concatenate([grenzen, [len(csn)]])
        for s, e in zip(starts, ends):
            t, v = tt_all[s:e], vv_all[s:e]
            tu, inv = np.unique(t, return_inverse=True)           # doppelte Zeiten -> Median (wie v17)
            vu = np.array([np.median(v[inv == i]) for i in range(len(tu))]) if len(tu) != len(t) else v
            mon[int(csn[s])] = (tu, vu)
    del tt_all, vv_all, csn
    u = c.execute("SELECT CSN,seg,t,hfA,hfB FROM units_v16 WHERE ev=? AND CSN IN (SELECT CSN FROM visit WHERE split=?) ORDER BY CSN,t", [ev, split]).fetchnumpy()
    num = lambda x: (np.ma.filled(x.astype(float), np.nan) if isinstance(x, np.ma.MaskedArray) else np.asarray(x, float))
    ucsn = np.asarray(u['CSN'], dtype=np.int64)
    arr = np.column_stack([num(u['t']), num(u['hfA']), num(u['hfB']), num(u['seg'])])
    un = {}
    if len(ucsn):
        grenzen = np.flatnonzero(np.diff(ucsn) != 0) + 1
        starts = np.concatenate([[0], grenzen])
        ends = np.concatenate([grenzen, [len(ucsn)]])
        for s, e in zip(starts, ends):
            un[int(ucsn[s])] = arr[s:e]
    del u, arr, ucsn
    out = []
    leer = (np.empty(0), np.empty(0))
    for csn_, mrn, r, dd in vis:
        x = un.get(int(csn_), np.empty((0, 4)))
        m = mon.get(int(csn_), leer)
        out.append(A.Besuch(id=csn_, mrn=mrn, lo=r + cfg['excl_s'], hi=dd - cfg['excl_s'], ts=x[:, 0], hfA=x[:, 1], hfB=x[:, 2], seg=x[:, 3].astype(int),
                            chart_t=leer[0], chart_v=leer[1], mon_t=m[0], mon_v=m[1], hrv5=None))
    c.close()
    return out


def patientenindex(besuche):
    """Patienten als laufende Indizes (nicht umkehrbar gespeichert)."""
    u, inv = np.unique(np.array([b.mrn for b in besuche]), return_inverse=True)
    return inv.astype(np.int64)


def logzeile(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    zeile = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {text}"
    with open(OUT / f'lauf_{name}.log', 'a', encoding='utf-8') as f:
        f.write(zeile + '\n')
    print(zeile, flush=True)


# =========================================================================== Rechenphasen
def phase_haupt(A, besuche, pat):
    t0 = time.time()
    code, ex = werte_zerlegt(A, besuche, extras=True)
    np.savez(ZW / 'haupt.npz', code=code, pat=pat, **ex)
    nen = (code & NENN) > 0
    ident = nen & ((code & 15) == 0)
    logzeile('haupt', f'haupt fertig: Nenner {int(nen.sum())}, identifizierbar {int(ident.sum())}, {time.time() - t0:.0f} s')


def phase_nk1(A, besuche):
    t0 = time.time()
    rng = np.random.default_rng(A.CFG['seeds']['nk1_permutation'])
    codes = np.zeros((A.CFG['nk1_n'], len(besuche)), np.uint8)
    for k in range(A.CFG['nk1_n']):
        codes[k], _ = werte_zerlegt(A, besuche, permutiere=rng)
        logzeile('nk1', f'NK1 Wiederholung {k + 1}/{A.CFG["nk1_n"]}, {time.time() - t0:.0f} s')
    np.savez(ZW / 'nk1.npz', codes=codes)


def phase_nk2(A, besuche):
    t0 = time.time()
    rng = np.random.default_rng(A.CFG['seeds']['nk2_shuffle'])
    getr = lambda b: (b.mon_t, b.mon_v)
    codes = np.zeros((A.CFG['nk2_n'], len(besuche)), np.uint8)
    ohne = []
    for k in range(A.CFG['nk2_n']):
        zuord, o = A.nk2_zuordnung(besuche, 'A', rng)
        ohne.append(o)

        def ov(b, zuord=zuord):
            p = zuord.get(b.id)
            if p is None:
                return np.empty(0), np.empty(0)
            t, v = getr(p)
            t2 = t + (b.lo - p.lo)
            m = (t2 >= b.lo) & (t2 <= b.hi)
            return t2[m], v[m]
        codes[k], _ = werte_zerlegt(A, besuche, ref_override=ov)
        logzeile('nk2', f'NK2 Wiederholung {k + 1}/{A.CFG["nk2_n"]}, {time.time() - t0:.0f} s')
    np.savez(ZW / 'nk2.npz', codes=codes, ohne=np.array(ohne))


def phase_nk3(A, besuche):
    t0 = time.time()
    rng = np.random.default_rng(A.CFG['seeds']['nk3_ausduennen'])
    n = A.CFG['nk3_n']
    for ab in A.CFG['dichte_min']:
        f = ZW / f'nk3_{ab}.npz'
        if f.exists():
            rng.bit_generator.state = json.loads(str(np.load(f)['rngstate']))
            logzeile('nk3', f'NK3 Stufe {ab} min vorhanden, Rng-Zustand übernommen')
            continue
        codes = np.zeros((n, len(besuche)), np.uint8)
        for k in range(n):
            codes[k], _ = werte_zerlegt(A, besuche, dichte=(ab, rng))
            logzeile('nk3', f'NK3 Stufe {ab} min, Wiederholung {k + 1}/{n}, {time.time() - t0:.0f} s')
        np.savez(f, codes=codes, rngstate=np.array(json.dumps(rng.bit_generator.state)))


def rechnen(phasen):
    pruefe_regeln()
    A, fz = lade_v17()
    OUT.mkdir(parents=True, exist_ok=True)
    ZW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    besuche = lade_train(A, fz['erkenner_version'])
    pat = patientenindex(besuche)
    logzeile('_'.join(phasen), f'geladen: {len(besuche)} Besuche, {time.time() - t0:.0f} s')
    for ph in phasen:
        if ph == 'haupt':
            phase_haupt(A, besuche, pat)
        elif ph == 'nk1':
            phase_nk1(A, besuche)
        elif ph == 'nk2':
            phase_nk2(A, besuche)
        elif ph == 'nk3':
            phase_nk3(A, besuche)
        else:
            raise SystemExit(f'unbekannte Phase {ph}')
    logzeile('_'.join(phasen), f'Phasen {phasen} fertig, {time.time() - t0:.0f} s')


# =========================================================================== Aggregation
def boot(A, pat, wn, wz, seed):
    pseudo = [{'mrn': int(p), 'n': float(a), 'z': float(b)} for p, a, b in zip(pat, wn, wz)]
    pt, lo, hi, zs, ns = A.anteil_bootstrap(pseudo, lambda r: r['n'], lambda r: r['z'], seed)
    ganz = lambda x: int(round(x)) if abs(x - round(x)) < 1e-9 else x
    return {'wert': pt, 'ci95': [lo, hi], 'zaehler': ganz(zs), 'nenner': ganz(ns)}


def terzile(hf_sd, nen):
    """Terzile der HF-Standardabweichung im Nenner (wie v17.h0_variabilitaet). Rückgabe Label-Array ('' außerhalb des Nenners) und Grenzen."""
    sd = hf_sd[nen]
    c1, c2 = np.nanquantile(sd, [1 / 3, 2 / 3])
    lab = np.full(len(hf_sd), '', dtype='<U6')
    for i in np.flatnonzero(nen):
        lab[i] = 'unten' if hf_sd[i] <= c1 else ('mitte' if hf_sd[i] <= c2 else 'oben')
    return lab, [float(c1), float(c2)]


def kategorien(code):
    """Erster verletzter Grund je Besuch ('' außerhalb des Nenners)."""
    kat = np.full(len(code), '', dtype='<U3')
    for i in np.flatnonzero((code & NENN) > 0):
        kat[i] = erster_grund(int(code[i]))
    return kat


def zerlegungstabelle(kat, nen, mask):
    return {g: int(np.sum((kat == g) & mask)) for g in GRUND + ['v']}


def aggregiere():
    A, fz = lade_v17()
    cfg = A.CFG
    seed = cfg['seeds']['bootstrap']
    R = {}
    h = np.load(ZW / 'haupt.npz')
    code, pat = h['code'], h['pat']
    hf_sd = h['hf_sd']
    nen = (code & NENN) > 0
    N = int(nen.sum())
    ident = nen & ((code & 15) == 0)
    R['haupt'] = {'besuche_gesamt': int(len(code)), 'nenner': N, 'identifizierbar': int(ident.sum())}
    kat = kategorien(code)
    R['zerlegung'] = {'erster_grund': {}, 'verletzt_unabhaengig': {}, 'ueberlappung': {}, 'terzile': {}, 'komponenten': {}, 'zweitkomponente': {}}
    for g in GRUND + ['v']:
        R['zerlegung']['erster_grund'][g] = boot(A, pat, nen, (kat == g), seed)
    for g, b in zip(GRUND, (B1, B2, B3, B4)):
        k = int(np.sum(nen & ((code & b) > 0)))
        R['zerlegung']['verletzt_unabhaengig'][g] = {'anzahl': k, 'anteil': k / N}
    for m in range(16):
        k = int(np.sum(nen & ((code & 15) == m)))
        name = '+'.join(g for g, b in zip(GRUND, (B1, B2, B3, B4)) if m & b) or 'keiner'
        R['zerlegung']['ueberlappung'][name] = k
    lab, grenzen = terzile(hf_sd, nen)
    R['zerlegung']['terzilgrenzen_bpm'] = grenzen
    for t in ('unten', 'mitte', 'oben'):
        mk = lab == t
        R['zerlegung']['terzile'][t] = {
            'besuche': int(mk.sum()),
            'erster_grund': zerlegungstabelle(kat, nen, mk),
            'verletzt_unabhaengig': {g: int(np.sum(mk & ((code & b) > 0))) for g, b in zip(GRUND, (B1, B2, B3, B4))},
            'ueberlappung': {('+'.join(g for g, b in zip(GRUND, (B1, B2, B3, B4)) if m & b) or 'keiner'): int(np.sum(mk & ((code & 15) == m))) for m in range(16)},
            'ident_c1': int(np.sum(mk & ((code & OK1) > 0))), 'ident_c2': int(np.sum(mk & ((code & OK2) > 0))),
            'in_lage_unter_ident': int(np.sum(mk & ident & ((code & LAGE) > 0)))}
    nc = h['ncomp']
    R['zerlegung']['komponenten'] = {'1': int(np.sum(nen & (nc == 1))), '2': int(np.sum(nen & (nc == 2))), '3': int(np.sum(nen & (nc == 3))), '4_plus': int(np.sum(nen & (nc >= 4)))}
    ab = h['abst2']
    iii = nen & ((code & B3) > 0)
    R['zerlegung']['zweitkomponente'] = {
        'besuche_mit_zweitkomponente': int(np.sum(nen & np.isfinite(ab))),
        'abstand_genau_4': int(np.sum(iii & (ab == 4))), 'abstand_bis_10': int(np.sum(iii & (ab <= 10))), 'abstand_10_bis_30': int(np.sum(iii & (ab > 10) & (ab <= 30))), 'abstand_ueber_30': int(np.sum(iii & (ab > 30))),
        'abstand_median_min': float(np.median(ab[iii])) if iii.any() else None, 'abstand_quartile_min': np.quantile(ab[iii], [.25, .75]).tolist() if iii.any() else None}
    R['zerlegung']['minimum'] = {'mediane_zonenbreite_gebunden_ident': float(np.median(h['breite'][ident])),
                                 'median_lo_nenner': float(np.median(h['lo'][nen])), 'einheiten_median': float(np.median(h['n'][nen]))}
    # ---- Variante C1 / C2 (Daten)
    var = {}
    base_ident = ident
    for name, bit in (('gebunden', None), ('C1', OK1), ('C2', OK2)):
        idv = base_ident if bit is None else (nen & ((code & bit) > 0))
        # Lageanteil wie v17: Patienten-Bootstrap nur über die Identifizierbaren (Seed + 1)
        lage = boot(A, pat[idv], np.ones(int(idv.sum())), ((code & LAGE) > 0)[idv], seed + 1) if idv.any() else None
        breite = h['breite'] if name != 'C2' else h['cw']
        neu = idv & ~base_ident
        var[name] = {'identifizierbar': int(idv.sum()), 'anteil': boot(A, pat, nen, idv, seed), 'lage_anteil': lage,
                     'in_lage': int(np.sum(idv & ((code & LAGE) > 0))),
                     'breite_median_min': float(np.median(breite[idv])) if idv.any() else None,
                     'neu_identifizierbar': int(neu.sum()), 'neu_in_lage': int(np.sum(neu & ((code & LAGE) > 0))),
                     'offset_median_min': float(np.median(h['offset'][idv])) if idv.any() else None,
                     'terzile': {t: {'besuche': int(np.sum(lab == t)), 'identifizierbar': int(np.sum(idv & (lab == t)))} for t in ('unten', 'mitte', 'oben')}}
    R['variante_daten'] = var
    # Nachtrag N1: Lage ohne Zonenregel (Anteil der Besuche im Nenner, deren gewähltes Minimum innerhalb +-2 min liegt)
    lg = nen & ((code & LAGE) > 0)
    R['lage_alle'] = {'daten': boot(A, pat, nen, lg, seed),
                      'terzile': {t: {'anzahl': int(np.sum(lg & (lab == t))), 'besuche': int(np.sum(lab == t))} for t in ('unten', 'mitte', 'oben')}}
    # ---- Nullkontrollen NK1 / NK2 unter denselben Varianten
    for nk in ('nk1', 'nk2'):
        z = np.load(ZW / f'{nk}.npz')
        codes = z['codes']
        nz = (codes & NENN) > 0
        nn = nz.sum(axis=1).astype(float)
        out = {'wiederholungen': int(codes.shape[0]), 'nenner_mittel': float(nn.mean())}
        for name, bit in (('gebunden', None), ('C1', OK1), ('C2', OK2)):
            idk = (nz & ((codes & 15) == 0)) if bit is None else (nz & ((codes & bit) > 0))
            zahl = idk.sum(axis=1).astype(float)
            anteile = zahl / np.where(nn > 0, nn, np.nan)
            lu = [float((idk[k] & ((codes[k] & LAGE) > 0)).sum() / idk[k].sum()) for k in range(codes.shape[0]) if idk[k].sum() > 0]
            out[name] = {'lage_unter_ident_mittel': float(np.mean(lu)) if lu else None,
                         'identifizierbar_mittel': float(zahl.mean()), 'identifizierbar_min': int(zahl.min()), 'identifizierbar_max': int(zahl.max()),
                         'anteil_mittel': float(np.nanmean(anteile)), 'anteil_min': float(np.nanmin(anteile)), 'anteil_max': float(np.nanmax(anteile)),
                         'anteil_max_zaehler_durch_mittleren_nenner': float(zahl.max() / nn.mean())}
        la = (nz & ((codes & LAGE) > 0)).sum(axis=1) / np.where(nn > 0, nn, np.nan)
        out['lage_alle'] = {'anteil_mittel': float(np.nanmean(la)), 'anteil_min': float(np.nanmin(la)), 'anteil_max': float(np.nanmax(la))}
        if nk == 'nk2':
            out['ohne_partner_mittel'] = float(np.mean(z['ohne']))
        R[nk] = out
    # ---- Nachvollzug der gebundenen Ergebnisse
    b = json.loads(BOUND_JSON.read_text(encoding='utf-8'))['A']
    bz = b['zusammenfassung']
    nv = {'nenner': (N, bz['nenner']), 'identifizierbar': (int(ident.sum()), bz['identifizierbar']),
          'anteil': (R['variante_daten']['gebunden']['anteil']['wert'], bz['identifizierbar_anteil']['wert']),
          'anteil_ci_unten': (R['variante_daten']['gebunden']['anteil']['ci95'][0], bz['identifizierbar_anteil']['ci95'][0]),
          'anteil_ci_oben': (R['variante_daten']['gebunden']['anteil']['ci95'][1], bz['identifizierbar_anteil']['ci95'][1]),
          'innerhalb_2': (R['variante_daten']['gebunden']['in_lage'], bz['innerhalb_min']['2']['anzahl']),
          'terzil_grenze_1': (grenzen[0], b['h0_variabilitaet']['terzilgrenzen_bpm'][0]), 'terzil_grenze_2': (grenzen[1], b['h0_variabilitaet']['terzilgrenzen_bpm'][1])}
    for t in ('unten', 'mitte', 'oben'):
        nv[f'terzil_{t}_ident'] = (int(np.sum(ident & (lab == t))), b['h0_variabilitaet'][t]['identifizierbar'])
    nv['nk1_ident_max'] = (R['nk1']['gebunden']['identifizierbar_max'], b['nk1']['identifizierbar_max'])
    nv['nk1_ident_min'] = (R['nk1']['gebunden']['identifizierbar_min'], b['nk1']['identifizierbar_min'])
    nv['nk2_anteil_mittel'] = (R['nk2']['gebunden']['anteil_mittel'], b['nk2']['falsch_identifiziert_anteil_mittel'])
    nv['nk2_nenner_mittel'] = (R['nk2']['nenner_mittel'], b['nk2']['nenner_mittel'])
    R['nachvollzug_vorlaeufig'] = nv
    # ---- NK3 auf gemeinsamem Nenner
    stufen = cfg['dichte_min']
    codes3 = np.stack([np.load(ZW / f'nk3_{s}.npz')['codes'] for s in stufen])            # (S, K, N)
    S, K, _ = codes3.shape
    nz3 = (codes3 & NENN) > 0
    gem = nz3.all(axis=0)                                                                 # (K, N): in allen Stufen im Nenner derselben Wiederholung
    w = gem.mean(axis=0)
    strikt = gem.all(axis=0)
    mindestens_eine = nz3.any(axis=0)
    s3 = cfg['seeds']['nk3_bootstrap']
    n3 = {'wiederholungen': int(K), 'gemeinsam_mittel': float(gem.sum(axis=1).mean()), 'mindestens_eine_stufe_mittel': float(mindestens_eine.sum(axis=1).mean()),
          'strikt_alle_wiederholungen': int(strikt.sum()), 'stufen': {}}
    ok_main = ident
    for si, sname in enumerate(stufen):
        cs = codes3[si]
        nzs = nz3[si]
        identk = nzs & ((cs & 15) == 0)
        eigen = {'nenner_mittel': float(nzs.sum(axis=1).mean()), 'ident_mittel': float(identk.sum(axis=1).mean()),
                 'anteil': boot(A, pat, nzs.mean(axis=0), identk.mean(axis=0), s3)}
        katk = np.array([[erster_grund(int(c)) if c & NENN else '' for c in row] for row in cs])
        gemein = {'nenner': float(w.sum())}
        for g in GRUND + ['v']:
            gemein[g] = boot(A, pat, w, (gem & (katk == g)).mean(axis=0), s3)
        gemein['C1'] = boot(A, pat, w, (gem & ((cs & OK1) > 0)).mean(axis=0), s3)
        gemein['C2'] = boot(A, pat, w, (gem & ((cs & OK2) > 0)).mean(axis=0), s3)
        gemein['lage_alle'] = boot(A, pat, w, (gem & ((cs & LAGE) > 0)).mean(axis=0), s3)
        gemein['lage_unter_ident'] = float((gem & identk & ((cs & LAGE) > 0)).sum() / max(1, (gem & identk).sum()))
        st = {'eigen': eigen, 'gemeinsam': gemein,
              'strikt': {'nenner': int(strikt.sum()), 'identifizierbar_mittel': float((gem & identk)[:, strikt].sum(axis=1).mean()) if strikt.any() else None}}
        n3['stufen'][str(sname)] = st
    n3['ungeduennt_auf_gemeinsam'] = {'anteil': boot(A, pat, w, w * ok_main, s3),
                                     'davon_nicht_im_nenner_des_hauptlaufs': float((w * (~nen)).sum())}
    # Nachvollzug NK3
    bn = b['nk3']
    n3['nachvollzug'] = {str(s): {'nenner_mittel': (n3['stufen'][str(s)]['eigen']['nenner_mittel'], bn[str(s)]['nenner_mittel']),
                                  'anteil': (n3['stufen'][str(s)]['eigen']['anteil']['wert'], bn[str(s)]['anteil']['wert']),
                                  'ci_unten': (n3['stufen'][str(s)]['eigen']['anteil']['ci95'][0], bn[str(s)]['anteil']['ci95'][0]),
                                  'ci_oben': (n3['stufen'][str(s)]['eigen']['anteil']['ci95'][1], bn[str(s)]['anteil']['ci95'][1])} for s in stufen}
    R['nk3'] = n3
    return R


# =========================================================================== Berichtstabellen
def mk(n):
    """Kleinzellenregel: 1 bis 9 -> '<10'."""
    if n is None:
        return 'n/a'
    n = int(round(n)) if abs(n - round(n)) < 1e-9 else n
    if isinstance(n, int):
        return '<10' if 1 <= n <= 9 else str(n)
    return f'{n:.1f}'.replace('.', ',')


def fz(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{d}f}'.replace('.', ',')


def pci(a, d=3):
    return f"{fz(a['wert'], d)} ([{fz(a['ci95'][0], d)}; {fz(a['ci95'][1], d)}])"


def tab(kopf, zeilen):
    t = ['| ' + ' | '.join(kopf) + ' |', '|' + '|'.join(['---'] * len(kopf)) + '|']
    t += ['| ' + ' | '.join(str(c) for c in z) + ' |' for z in zeilen]
    return '\n'.join(t)


def tabellen(R, kopf_info):
    L = []
    h = R['haupt']
    N = h['nenner']
    z = R['zerlegung']
    L.append('### 3.1 Zerlegung nach dem ersten verletzten Grund (Train, Beziehung A)\n')
    L.append(f"Nenner {mk(N)} Besuche (mindestens 3 vollständige Einheiten, wie gebunden). Anteil mit Patienten-Bootstrap (1000 Ziehungen, Seed der gebundenen Auswertung).\n")
    L.append(tab(['Rang', 'Grund (erster verletzter)', 'Besuche', 'Anteil (95-%-Intervall)'],
                 [[i + 1, f"({g}) {GRUND_TEXT[g]}", mk(z['erster_grund'][g]['zaehler']), pci(z['erster_grund'][g])] for i, g in enumerate(GRUND + ['v'])]))
    L.append('\n### 3.2 Jeder Grund unabhängig von den anderen (Besuche können mehrere verletzen)\n')
    L.append(tab(['Grund', 'Besuche, die ihn verletzen', 'Anteil am Nenner'],
                 [[f"({g}) {GRUND_TEXT[g]}", mk(z['verletzt_unabhaengig'][g]['anzahl']), fz(z['verletzt_unabhaengig'][g]['anteil'])] for g in GRUND]))
    L.append('\n### 3.3 Überlappung (alle Kombinationen verletzter Gründe)\n')
    L.append(tab(['Verletzte Gründe', 'Besuche'], [[k, mk(v)] for k, v in z['ueberlappung'].items()]))
    L.append('\n### 3.4 Terzile der HF-Variabilität (Standardabweichung der Einheiten-HF im Nenner; Grenzen ' +
             f"{fz(z['terzilgrenzen_bpm'][0], 2)} und {fz(z['terzilgrenzen_bpm'][1], 2)} bpm)\n")
    rows = []
    for t in ('unten', 'mitte', 'oben'):
        d = z['terzile'][t]
        eg = d['erster_grund']
        rows.append([t, mk(d['besuche']), mk(eg['i'] + eg['ii']), mk(eg['iii']), mk(eg['iv']), mk(eg['v'])])
    L.append('Erster verletzter Grund (Besuche); (i) und (ii) sind zusammengefasst, weil (i) je Terzil weniger als 10 Besuche betrifft:\n')
    L.append(tab(['Terzil', 'Besuche', '(i) oder (ii)', '(iii) Zusammenhang', '(iv) Breite', '(v) identifizierbar'], rows))
    rows = []
    for t in ('unten', 'mitte', 'oben'):
        d = z['terzile'][t]
        rows.append([t] + [f"{mk(d['verletzt_unabhaengig'][g])} ({fz(d['verletzt_unabhaengig'][g] / d['besuche'])})" for g in ('ii', 'iii', 'iv')])
    L.append('\nGrund (ii), (iii) und (iv) jeweils unabhängig von den übrigen (Besuche, Anteil im Terzil):\n')
    L.append(tab(['Terzil', '(ii) Rand', '(iii) Zusammenhang', '(iv) Breite'], rows))
    rows = []
    for k_, bez in (('keiner', 'keiner (identifizierbar)'), ('ii', 'nur (ii)'), ('iii', 'nur (iii)'), ('ii+iii', '(ii) und (iii)')):
        rows.append([bez] + [mk(z['terzile'][t]['ueberlappung'][k_]) for t in ('unten', 'mitte', 'oben')])
    L.append('\nÜberlappung je Terzil (Besuche); Kombinationen mit (i) oder (iv) sind nicht aufgeführt ((iv) tritt nie auf, (i) je Terzil bei weniger als 10 Besuchen):\n')
    L.append(tab(['Verletzte Gründe', 'unten', 'mitte', 'oben'], rows))
    k = z['komponenten']
    zk = z['zweitkomponente']
    L.append('\n### 3.5 Aufbau der nahe-optimalen Menge (Fehler <= Minimum + 1/min)\n')
    L.append(tab(['Zahl getrennter Komponenten', 'Besuche'], [['1', mk(k['1'])], ['2', mk(k['2'])], ['3', mk(k['3'])], ['4 oder mehr', mk(k['4_plus'])]]))
    L.append('\nAbstand der nächsten fremden Komponente zum gewählten Minimum, nur Besuche, die (iii) verletzen:\n')
    L.append(tab(['Abstand', 'Besuche'], [['genau 4 min (Lücke von einem Gitterpunkt, Nachtrag N2)', mk(zk['abstand_genau_4'])], ['bis 10 min (einschließlich 4 min)', mk(zk['abstand_bis_10'])], ['über 10 bis 30 min', mk(zk['abstand_10_bis_30'])], ['über 30 min', mk(zk['abstand_ueber_30'])]]))
    L.append(f"\nMedian {fz(zk['abstand_median_min'], 1)} min (Quartile {fz(zk['abstand_quartile_min'][0], 1)}; {fz(zk['abstand_quartile_min'][1], 1)}).\n")
    L.append('\n### 4.1 Zusammenhangsvarianten auf den Daten (Train, Nenner ' + mk(N) + ')\n')
    L.append('Gebunden: ganze nahe-optimale Menge muss zusammenhängend und randfern sein. C1: nur die Zusammenhangsregel entfällt (Rand und Breite weiter auf der ganzen Menge). '
             'C2: alle Regeln nur auf der zusammenhängenden Komponente um das gewählte Minimum.\n')
    v = R['variante_daten']
    L.append(tab(['Regel', 'identifizierbar', 'Anteil (Intervall)', 'Lage innerhalb ±2 min, Anteil unter den Identifizierbaren (Intervall)', 'zusätzlich gegenüber gebunden', 'davon innerhalb ±2 min', 'mediane Zonenbreite (min)'],
                 [[n, mk(v[n]['identifizierbar']), pci(v[n]['anteil']), pci(v[n]['lage_anteil']), mk(v[n]['neu_identifizierbar']), mk(v[n]['neu_in_lage']), fz(v[n]['breite_median_min'], 1)]
                  for n in ('gebunden', 'C1', 'C2')]))
    L.append('\nIdentifizierbare je Terzil der HF-Variabilitä<local path>')
    L.append(tab(['Regel', 'unten', 'mitte', 'oben'],
                 [[n] + [f"{mk(v[n]['terzile'][t]['identifizierbar'])} von {mk(v[n]['terzile'][t]['besuche'])} ({fz(v[n]['terzile'][t]['identifizierbar'] / v[n]['terzile'][t]['besuche'])})" for t in ('unten', 'mitte', 'oben')]
                  for n in ('gebunden', 'C1', 'C2')]))
    L.append('\n### 4.2 Nullkontrollen unter derselben Variante (je 20 Wiederholungen, Seeds und Ziehungsreihenfolge der gebundenen Auswertung)\n')
    rows = []
    for n in ('gebunden', 'C1', 'C2'):
        a1, a2 = R['nk1'][n], R['nk2'][n]
        rows.append([n, fz(v[n]['anteil']['wert']), f"{fz(a1['anteil_mittel'])} (max. {fz(a1['anteil_max'])}; Zähler {mk(a1['identifizierbar_min'])} bis {mk(a1['identifizierbar_max'])})",
                     f"{fz(a2['anteil_mittel'])} (min. {fz(a2['anteil_min'])}; max. {fz(a2['anteil_max'])})",
                     fz(v[n]['anteil']['wert'] - a2['anteil_mittel']), fz(v[n]['anteil']['wert'] / a2['anteil_mittel'], 2) if a2['anteil_mittel'] else 'n/a'])
    L.append(tab(['Regel', 'Daten: Anteil', 'NK1 Permutation: Anteil identifizierbar', 'NK2 anderer Patient: Anteil identifizierbar', 'Daten minus NK2', 'Daten durch NK2'], rows))
    L.append(f"\nNK1-Nenner im Mittel {mk(R['nk1']['nenner_mittel'])}; NK2-Nenner im Mittel {mk(R['nk2']['nenner_mittel'])} (Referenzgeometrie und Nenner weichen von den Daten ab), ohne Partner im Mittel {mk(R['nk2'].get('ohne_partner_mittel'))}.\n")
    L.append('\n### 4.3 Nachtrag N1: Lage des gewählten Minimums ohne Zonenregel\n')
    la = R['lage_alle']
    L.append(f"Anteil der Besuche im Nenner, deren gewähltes Minimum innerhalb ±2 min liegt, unabhängig von jeder Zonenregel. Bei gleichverteilter Lage über das Gitter (61 Punkte, davon 3 im Fenster) wäre der Anteil {fz(3 / 61)}.\n")
    rows = [['Daten', pci(la['daten'])] + [fz(v[n]['lage_anteil']['wert']) for n in ('gebunden', 'C1', 'C2')]]
    for nk, nm_ in (('nk1', 'NK1 Permutation'), ('nk2', 'NK2 anderer Patient')):
        a = R[nk]
        rows.append([nm_, f"{fz(a['lage_alle']['anteil_mittel'])} (Bereich {fz(a['lage_alle']['anteil_min'])} bis {fz(a['lage_alle']['anteil_max'])})"] +
                    [fz(a[n]['lage_unter_ident_mittel']) for n in ('gebunden', 'C1', 'C2')])
    L.append(tab(['Gruppe', 'Lage innerhalb ±2 min, alle Besuche im Nenner', 'unter den Identifizierbaren, gebundene Regel', 'unter den Identifizierbaren, C1', 'unter den Identifizierbaren, C2'], rows))
    L.append('\nDaten je Terzil der HF-Variabilität (alle Besuche im Nenner):\n')
    L.append(tab(['Terzil', 'Besuche', 'Minimum innerhalb ±2 min', 'Anteil'],
                 [[t, mk(la['terzile'][t]['besuche']), mk(la['terzile'][t]['anzahl']), fz(la['terzile'][t]['anzahl'] / la['terzile'][t]['besuche'])] for t in ('unten', 'mitte', 'oben')]))
    L.append('\n### 5.1 NK3 auf gemeinsamem Nenner (Train)\n')
    n3 = R['nk3']
    L.append(f"Gemeinsamer Nenner: Besuche, die in derselben Wiederholung bei allen sechs Dichtestufen im Nenner sind (mindestens 3 vollständige Einheiten); Gewicht = Anteil der 20 Wiederholungen. "
             f"Größe im Mittel {mk(n3['gemeinsam_mittel'])} Besuche; Besuche mit Nenner in mindestens einer Stufe im Mittel {mk(n3['mindestens_eine_stufe_mittel'])}; "
             f"strenger Nenner (in allen 20 Wiederholungen und allen Stufen): {mk(n3['strikt_alle_wiederholungen'])} Besuche.\n")
    st = n3['stufen']
    rows = []
    for s in st:
        e, g = st[s]['eigen'], st[s]['gemeinsam']
        rows.append([s, mk(e['nenner_mittel']), pci(e['anteil']), mk(g['nenner']), pci(g['v'])])
    L.append(tab(['Referenzabstand (min)', 'Eigener Nenner der Stufe (Mittel)', 'Identifizierbar, eigener Nenner (Intervall)', 'Gemeinsamer Nenner (Mittel)', 'Identifizierbar, gemeinsamer Nenner (Intervall)'], rows))
    rows = [['ungedünnt (Hauptlauf)', mk(n3['ungeduennt_auf_gemeinsam']['anteil']['nenner']), pci(n3['ungeduennt_auf_gemeinsam']['anteil'])]]
    L.append('\nReferenzpunkt: gebundene, ungedünnte Auswertung auf demselben gemeinsamen Nenner (Besuche außerhalb des Nenners des Hauptlaufs zählen als nicht identifizierbar; Anzahl im Mittel ' +
             f"{mk(n3['ungeduennt_auf_gemeinsam']['davon_nicht_im_nenner_des_hauptlaufs'])}):\n")
    L.append(tab(['Auswertung', 'Gemeinsamer Nenner', 'Identifizierbar (Intervall)'], rows))
    L.append('\n### 5.2 Ausschlussgründe je Dichtestufe auf dem gemeinsamen Nenner (Anteil am gemeinsamen Nenner, erster verletzter Grund)\n')
    rows = []
    for s in st:
        g = st[s]['gemeinsam']
        rows.append([s] + [pci(g[x]) for x in ('i', 'ii', 'iii', 'iv', 'v')])
    L.append(tab(['Referenzabstand (min)', '(i) Mindestfehler', '(ii) Rand', '(iii) Zusammenhang', '(iv) Breite', '(v) identifizierbar'], rows))
    L.append('\n### 5.3 Zusammenhangsvarianten je Dichtestufe auf dem gemeinsamen Nenner\n')
    rows = []
    for s in st:
        g = st[s]['gemeinsam']
        rows.append([s, pci(g['v']), pci(g['C1']), pci(g['C2'])])
    L.append(tab(['Referenzabstand (min)', 'gebunden', 'C1 (ohne Zusammenhangsregel)', 'C2 (Komponente um das Minimum)'], rows))
    L.append('\n### 5.4 Nachtrag N1: Lage des gewählten Minimums innerhalb ±2 min je Dichtestufe (gemeinsamer Nenner, ohne Zonenregel)\n')
    L.append(tab(['Referenzabstand (min)', 'Minimum innerhalb ±2 min, alle Besuche im gemeinsamen Nenner (Intervall)', 'innerhalb ±2 min unter den nach gebundener Regel Identifizierbaren', 'identifizierbar nach gebundener Regel (zum Vergleich)'],
                 [[s_, pci(st[s_]['gemeinsam']['lage_alle']), fz(st[s_]['gemeinsam']['lage_unter_ident']), fz(st[s_]['gemeinsam']['v']['wert'])] for s_ in st]))
    L.append('\n### 5.5 Strenger gemeinsamer Nenner (Besuche, die in allen 20 Wiederholungen und allen Stufen im Nenner sind)\n')
    rows = [[s, mk(st[s]['strikt']['nenner']), fz(st[s]['strikt']['identifizierbar_mittel'] / st[s]['strikt']['nenner']) if st[s]['strikt']['nenner'] and st[s]['strikt']['identifizierbar_mittel'] is not None else 'n/a'] for s in st]
    L.append(tab(['Referenzabstand (min)', 'Besuche', 'Anteil identifizierbar (Mittel über Wiederholungen)'], rows))
    L.append('\n### 6 Nachvollzug der gebundenen Auswertung durch dieses Skript (Wiedergabe; gleiche Daten, gleiche Seeds)\n')
    nv = R['nachvollzug_vorlaeufig']
    rows = [[k, (mk(a) if isinstance(a, int) else fz(a, 4)), (mk(b) if isinstance(b, int) else fz(b, 4)), 'ja' if (a == b if isinstance(a, int) else abs(a - b) < 1e-9) else 'NEIN'] for k, (a, b) in nv.items()]
    L.append(tab(['Größe', 'dieses Skript', 'gebunden', 'gleich'], rows))
    L.append('\nNK3 (eigene Nenner der Stufen):\n')
    rows = []
    for s, d in n3['nachvollzug'].items():
        rows.append([s, fz(d['nenner_mittel'][0], 2), fz(d['nenner_mittel'][1], 2), fz(d['anteil'][0], 6), fz(d['anteil'][1], 6),
                     'ja' if (abs(d['nenner_mittel'][0] - d['nenner_mittel'][1]) < 1e-9 and abs(d['anteil'][0] - d['anteil'][1]) < 1e-9 and abs(d['ci_unten'][0] - d['ci_unten'][1]) < 1e-9) else 'NEIN'])
    L.append(tab(['Stufe (min)', 'Nenner Skript', 'Nenner gebunden', 'Anteil Skript', 'Anteil gebunden', 'gleich'], rows))
    return '\n'.join(L)


def auswerten():
    pruefe_regeln()
    R = aggregiere()
    OUT.mkdir(parents=True, exist_ok=True)
    A, _ = lade_v17()
    konv = lambda o: int(o) if isinstance(o, np.integer) else (bool(o) if isinstance(o, np.bool_) else float(o))
    (OUT / 'ergebnis_intern.json').write_text(json.dumps(R, indent=1, default=konv), encoding='utf-8')
    (OUT / 'ergebnis.json').write_text(json.dumps(A.Z.mask_tree(R), indent=1, default=konv), encoding='utf-8')
    (OUT / 'tabellen.md').write_text(tabellen(R, None), encoding='utf-8')
    print('geschrieben:', OUT / 'tabellen.md')


def bericht(prosa):
    s = pruefe_regeln()
    kopf = (f"Erzeugt {dt.datetime.now().strftime('%d.%m.%Y %H:%M')} aus `werkzeuge\\s5_mcmed\\s5_mcmed_diagnose_aw251.py` (SHA-256 `{s}`) "
            f"und den maskierten Aggregaten in `<local path>`. Regeln: `nachrechnung\\S5_MCMED_DIAGNOSE_AW251_REGELN.md`.\n")
    txt = Path(prosa).read_text(encoding='utf-8')
    tabs = (OUT / 'tabellen.md').read_text(encoding='utf-8')
    ERGEBNIS_MD.write_text(txt.replace('{{KOPF}}', kopf).replace('{{TABELLEN}}', tabs), encoding='utf-8')
    print('geschrieben:', ERGEBNIS_MD)


# =========================================================================== Selbsttest (Kunstdaten, keine echten Daten)
def selbsttest():
    sp = importlib.util.spec_from_file_location('s5v17', V17)
    A = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(A)
    cfg = A.CFG
    rg = cfg['A']
    D = A.raster(GA, 2)
    nD = len(D)
    i0 = int(np.flatnonzero(D == 0)[0])
    basis = 3.0
    hoch = basis + 6.0

    def kurve(minimum=basis, punkte=None):
        c = np.full(nD, minimum + 6.0)
        for off, wert in (punkte or {}).items():
            c[int(np.flatnonzero(D == off)[0])] = wert
        return c

    def drei(c):
        return np.vstack([c, c, c])                                       # drei identische Fehlerkurven

    def echt(c, regel=rg, Dx=D):
        return A.schaetze(drei(c), Dx, regel)

    # Fall 1: Minimum nur bei 0 -> identifizierbar, Breite 0
    c1 = kurve(punkte={0: basis})
    e = echt(c1)
    assert e['ok'] and e['breite'] == 0.0 and e['offset'] == 0.0 and e['near_zusammenhaengend']
    z = zerlege(c1, D, cfg)
    assert z['ok_bound'] and z['ok_c1'] and z['ok_c2'] and z['bits'] & 15 == 0 and erster_grund(z['bits']) == 'v'
    # Fall 2 (Sol): zusätzlicher isolierter Punkt bei +20 min mit Fehler Minimum + 0,8 -> nicht identifizierbar (Zusammenhang), Minimum bei 0, Breite 20
    c2 = kurve(punkte={0: basis, 20: basis + 0.8})
    e = echt(c2)
    assert (not e['ok']) and (not e['near_zusammenhaengend']) and e['offset'] == 0.0 and e['breite'] == 20.0 and e['near_lo'] == 0.0 and e['near_hi'] == 20.0
    z = zerlege(c2, D, cfg)
    assert (not z['ok_bound']) and z['ok_c1'] and z['ok_c2'] and (z['bits'] & 15) == B3 and erster_grund(z['bits']) == 'iii' and z['ncomp'] == 2 and z['abst2'] == 20.0
    assert z['cw'] == 0.0
    # Toleranzgrenze: Punkt genau bei Minimum + 1,0 gehört dazu, bei + 1,0 + 1e-6 nicht
    assert not echt(kurve(punkte={0: basis, 20: basis + 1.0}))['ok']
    assert echt(kurve(punkte={0: basis, 20: basis + 1.0 + 1e-6}))['ok']
    # Fall 3 (Rand): Minimum bei 56, Nachbarn 58 und 60 innerhalb der Toleranz -> zusammenhängend, berührt den Rand
    c3 = kurve(punkte={56: basis, 58: basis + 0.5, 60: basis + 0.9})
    e = echt(c3)
    assert (not e['ok']) and e['near_zusammenhaengend'] and e['near_hi'] == 60.0 and e['offset'] == 56.0
    z = zerlege(c3, D, cfg)
    assert (z['bits'] & 15) == B2 and erster_grund(z['bits']) == 'ii' and (not z['ok_c1']) and (not z['ok_c2'])
    # Randpunkt isoliert bei -60: zählt für (ii) und (iii); erster Grund (ii)
    c3b = kurve(punkte={0: basis, -60: basis + 0.9})
    z = zerlege(c3b, D, cfg)
    assert (z['bits'] & 15) == (B2 | B3) and erster_grund(z['bits']) == 'ii' and z['ok_c2'] and (not z['ok_c1'])
    # Fall 4 (Mindestfehler): Minimum 6,0 bei 0 -> nicht identifizierbar nur wegen des Mindestfehlers
    c4 = kurve(minimum=6.0, punkte={0: 6.0})
    e = echt(c4)
    assert (not e['ok']) and e['near_zusammenhaengend'] and e['minimum'] == 6.0
    z = zerlege(c4, D, cfg)
    assert (z['bits'] & 15) == B1 and erster_grund(z['bits']) == 'i' and not z['ok_c1'] and not z['ok_c2']
    # genau 5,0 ist erlaubt (<=)
    assert echt(kurve(minimum=5.0, punkte={0: 5.0}))['ok']
    # Fall 5 (Breite): auf einem größeren Gitter (+-70) mit zusammenhängender Zone von -62 bis +62 (124 min), randfern -> nur (iv)
    D70 = A.raster(70, 2)
    c5 = np.full(len(D70), basis + 6.0)
    c5[(D70 >= -62) & (D70 <= 62)] = basis + 0.5
    c5[int(np.flatnonzero(D70 == 0)[0])] = basis
    e = A.schaetze(drei(c5), D70, rg)
    assert (not e['ok']) and e['near_zusammenhaengend'] and e['breite'] == 124.0
    z = zerlege(c5, D70, cfg)
    assert (z['bits'] & 15) == B4 and erster_grund(z['bits']) == 'iv' and z['cw'] == 124.0 and not z['ok_c2']
    # Reihenfolge: (i) vor (ii) vor (iii) vor (iv)
    assert erster_grund(B1 | B2 | B3 | B4) == 'i' and erster_grund(B2 | B3) == 'ii' and erster_grund(B3 | B4) == 'iii' and erster_grund(B4) == 'iv' and erster_grund(0) == 'v'
    # Gleichstand: zwei gleiche Minima -> kleinster Absolutversatz
    ct = kurve(punkte={-4: basis, 6: basis})
    assert echt(ct)['offset'] == -4.0 and zerlege(ct, D, cfg)['offset'] == -4.0
    # Zufallskurven: Zerlegung muss die echte v17-Entscheidung reproduzieren (und Bits, Komponenten konsistent)
    rng = np.random.default_rng(251)
    nf = 0
    for _ in range(3000):
        c = 2.0 + np.cumsum(rng.normal(0, rng.choice([0.2, 0.6, 1.5]), nD))
        c += rng.choice([0.0, 4.0]) * (rng.random(nD) < 0.3)
        c = np.abs(c)
        e = A.schaetze(drei(c), D, rg)
        z = zerlege(c, D, cfg)
        assert bool(e['ok']) == bool(z['ok_bound']) and e['offset'] == z['offset'] and e['breite'] == z['breite'] and e['near_zusammenhaengend'] == z['zus']
        assert z['zus'] == (z['ncomp'] == 1)
        assert (not z['ok_bound']) or (z['ok_c1'] and z['ok_c2'])                    # gebunden impliziert C1 und C2
        assert (not z['ok_c1']) or z['ok_c2']                                        # C1 impliziert C2
        nf += int(e['ok'])
    assert 0 < nf < 3000
    # Integration: werte_zerlegt auf Kunstbesuchen reproduziert r['ok'] (A, Sollversatz 0), Dichte- und Permutationspfad ebenfalls
    bes = A._kunst_besuche(n_besuche=40, offset_min=0.0, seed=5)
    code, ex = werte_zerlegt(A, bes, extras=True)
    assert int(((code & NENN) > 0).sum()) > 0
    rngp = np.random.default_rng(1)
    werte_zerlegt(A, bes, permutiere=rngp)
    werte_zerlegt(A, bes, dichte=(5, np.random.default_rng(2)))
    # gemeinsamer Nenner: Logik an einem kleinen Beispiel
    nz3 = np.zeros((2, 3, 4), bool)                       # Stufen, Wiederholungen, Besuche
    nz3[:, :, 0] = True                                   # Besuch 0 immer
    nz3[0, :, 1] = True                                   # Besuch 1 nur in Stufe 0
    nz3[:, 0, 2] = True                                   # Besuch 2 nur in Wiederholung 0
    gem = nz3.all(axis=0)
    assert gem.sum(axis=1).tolist() == [2, 1, 1] and np.allclose(gem.mean(axis=0), [1, 0, 1 / 3, 0]) and gem.all(axis=0).tolist() == [True, False, False, False]
    # Kleinzellenmaske
    assert mk(0) == '0' and mk(1) == '<10' and mk(9) == '<10' and mk(10) == '10'
    print('SELBSTTEST OK (Kunstdaten-asserts bestanden, 3000 Zufallskurven gegen v17.schaetze)')
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('modus', choices=['selbsttest', 'rechnen', 'auswerten', 'bericht'])
    ap.add_argument('--phasen', default='haupt,nk1,nk2')
    ap.add_argument('--prosa')
    a = ap.parse_args()
    if a.modus == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    elif a.modus == 'rechnen':
        rechnen([p for p in a.phasen.split(',') if p])
    elif a.modus == 'auswerten':
        auswerten()
    else:
        bericht(a.prosa)
