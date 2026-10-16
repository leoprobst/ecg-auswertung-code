"""AW-241: unabhaengige Gegenrechnung der Fehlerkorrektur B-R1 (Bootstrap-Gueltigkeitsregel).

Eigenstaendig geschrieben: Logit-Anpassung (gedaempftes Newton-Verfahren mit Ridge 1e-6 auf alle
Nicht-Interzept-Koeffizienten), gewichtete AUC, Separationspruefung (LP), Bootstrap-Schleife,
Gueltigkeitsregeln, Perzentilintervalle. Aus den gebundenen Originalen werden nur Datenleser bzw.
Designbauer verwendet: werkzeuge.b_test.statistik (raw_design, fixed_design, draws nur als
Kontrollvergleich) und werkzeuge.b_test.daten (select_index, masks).

Aufruf:
  python b_r1_gegenrechnung.py --test
  python b_r1_gegenrechnung.py --run --freigabe-eingetragen
Keine Kennungen in Ausgaben. Persoenliche Zwischenwerte nur unter <local path>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
import sys
import json
import time
import hashlib
import threading
import argparse
import numpy as np
import psutil
from scipy.special import expit
from scipy.optimize import linprog

ROOT = str(data_path("b.auswertung"))
PRIVAT = str(data_path("b.aw241_gegenrechnung"))
OUT_TXT = ROOT + '/nachrechnung/AW241/B_R1_GEGENRECHNUNG_ausgabe.txt'
SEED = 1282026
RIDGE = 1e-6
TOL = 1e-7
SEP_TOL = 1e-6          # mittlere vorzeichenbehaftete Marge der LP-Loesung, ab der Separation gilt
SEP_BORDER = 1e-8       # Diagnose: Marge zwischen SEP_BORDER und SEP_TOL wird gezaehlt
RAM_PROZESS = 1.0 * 2 ** 30
RAM_FREI_MIN = 1.5 * 2 ** 30
PEAK = {'rss': 0}


# ----------------------------------------------------------------------------------------------
# Speicherwaechter
# ----------------------------------------------------------------------------------------------
def start_waechter():
    proc = psutil.Process()

    def loop():
        while True:
            rss = proc.memory_info().rss
            avail = psutil.virtual_memory().available
            PEAK['rss'] = max(PEAK['rss'], rss)
            if rss > RAM_PROZESS or avail < RAM_FREI_MIN:
                sys.stderr.write('WAECHTER: Abbruch (RSS %.2f GiB, frei %.2f GiB)\n' % (rss / 2 ** 30, avail / 2 ** 30))
                sys.stderr.flush()
                os._exit(3)
            time.sleep(0.25)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


# ----------------------------------------------------------------------------------------------
# Eigener Kern
# ----------------------------------------------------------------------------------------------
class Invalid(Exception):
    pass


def fit_logit(z, y, w, max_steps=200, ridge=RIDGE):
    """Gewichtete Logit-Anpassung, Newton mit Schrittverkuerzung. Rueckgabe (beta, konvergiert, Schritte).

    Zielfunktion: sum w*(log(1+exp(eta)) - y*eta) + 0.5*ridge*|beta_ohne_Interzept|^2.
    Konvergenz: max|g| / sum(w) < 1e-7 und max|g| / N < 1e-7, g = X'(w(p-y)) + pen*beta, N = Zeilenzahl.
    """
    n, p = z.shape
    pen = np.r_[0.0, np.full(p - 1, ridge)]
    sw = float(np.sum(w))
    m = float(np.sum(w * y) / sw)
    b = np.zeros(p)
    b[0] = np.log(m / (1.0 - m))

    def obj(bb):
        eta = z @ bb
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) + 0.5 * np.sum(pen * bb * bb))

    f = obj(b)
    steps = 0
    while True:
        pr = expit(z @ b)
        g = z.T @ (w * (pr - y)) + pen * b
        gmax = float(np.max(np.abs(g)))
        if gmax / sw < TOL and gmax / n < TOL:
            return b, True, steps
        if steps >= max_steps:
            return b, False, steps
        h = z.T @ ((w * pr * (1.0 - pr))[:, None] * z) + np.diag(pen)
        d = np.linalg.solve(h, g)
        gd = float(g @ d)
        t = 1.0
        while True:
            bn = b - t * d
            fn = obj(bn)
            if np.isfinite(fn) and fn <= f - 1e-4 * t * gd + 1e-12 * abs(f):
                break
            t *= 0.5
            if t < 1e-12:
                return b, False, steps
        b, f = bn, fn
        steps += 1


def auc_w(y, s, w):
    """Gewichtete AUC = P(s_Fall > s_Kontrolle) + 0.5 P(gleich), Gewichte als Personenmultiplizitaeten."""
    u, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    neg = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    below = np.cumsum(neg) - neg
    return float(np.sum(pos * (below + 0.5 * neg)) / (pos.sum() * neg.sum()))


def separation_margin(z, y):
    """Albert-Anderson per LP: maximiere mittlere Marge a'b, a = z*(2y-1), unter a'b >= 0, |b|<=1.

    Positive Marge (> SEP_TOL) heisst: es gibt eine nichttriviale Richtung ohne Fehlklassifikation
    (vollstaendige oder Quasi-Separation; bei vollem Rang ist a'b = 0 fuer alle Zeilen ausgeschlossen)."""
    a = z * (2.0 * y - 1.0)[:, None]
    res = linprog(-a.mean(0), A_ub=-a, b_ub=np.zeros(len(y)), bounds=[(-1.0, 1.0)] * z.shape[1],
                  method='highs')
    if res.status != 0 or not np.isfinite(res.fun):
        raise Invalid('Separations-LP numerisch nicht loesbar')
    return float(-res.fun)


def evaluate(z, y, w, max_steps=200):
    """Gueltigkeitsregeln fuer eine Anpassung (Originalstichprobe oder Ziehung) und Kennzahlen.

    z: Design mit Interzept in Spalte 0, Marker in der letzten Spalte; nur Zeilen mit w > 0."""
    if not (np.isfinite(z).all() and np.isfinite(y).all() and np.isfinite(w).all()) or (w <= 0).any():
        raise Invalid('nicht endlich')
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise Invalid('Klasse fehlt')
    n, p = z.shape
    if n <= p:
        raise Invalid('N<=p')
    try:
        if np.linalg.matrix_rank(z) != p:
            raise Invalid('Rangdefizit')
        margin = separation_margin(z, y)
        if margin > SEP_TOL:
            raise Invalid('Separation')
        b1, ok1, s1 = fit_logit(z, y, w, max_steps)
        if not ok1 or not np.isfinite(b1).all():
            raise Invalid('Konvergenz Vollmodell')
        base = z[:, :-1]
        b0, ok0, s0 = fit_logit(base, y, w, max_steps)
        if not ok0 or not np.isfinite(b0).all():
            raise Invalid('Konvergenz Basismodell')
    except np.linalg.LinAlgError:
        raise Invalid('LinAlg')
    a1 = auc_w(y, z @ b1, w)
    a0 = auc_w(y, base @ b0, w)
    if not (np.isfinite(a1) and np.isfinite(a0)):
        raise Invalid('AUC nicht endlich')
    return {'beta': float(b1[-1]), 'or': float(np.exp(b1[-1])), 'dauc': float(a1 - a0), 'auc1': a1, 'auc0': a0,
            'steps': max(s1, s0), 'margin': margin}


def multiplicities(G, r):
    """Ziehungsrezept: PCG64 mit SeedSequence([1282026, r]); bincount der G Ziehungen aus G Personen."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, int(r)])))
    return np.bincount(rng.integers(0, G, G), minlength=G)


def bootstrap(z, y, subjects, union, nboot=1000, max_steps=200, label=''):
    union = np.asarray(union)
    subjects = np.asarray(subjects)
    if len(union) == 0 or not np.array_equal(union, np.unique(union)):
        raise ValueError('Personenunion nicht sortiert/eindeutig')
    pos = np.searchsorted(union, subjects)
    if (pos >= len(union)).any() or not np.array_equal(union[pos], subjects):
        raise ValueError('Person fehlt in Union')
    G = len(union)
    rows = {k: [] for k in ('r', 'beta', 'or', 'dauc', 'steps', 'margin')}
    fails = {}
    altregel = 0
    border = 0
    t0 = time.time()
    for r in range(nboot):
        m = multiplicities(G, r)[pos].astype(float)
        act = m > 0
        ya = y[act]
        if min(len(np.unique(subjects[act][ya == 0])), len(np.unique(subjects[act][ya == 1]))) < 100:
            altregel += 1
        try:
            e = evaluate(z[act], ya, m[act], max_steps)
        except Invalid as exc:
            fails[str(exc)] = fails.get(str(exc), 0) + 1
            continue
        if SEP_BORDER < e['margin'] <= SEP_TOL:
            border += 1
        rows['r'].append(r)
        for k in ('beta', 'or', 'dauc', 'steps', 'margin'):
            rows[k].append(e[k])
        if (r + 1) % 100 == 0:
            print(json.dumps({'analyse': label, 'ziehungen': r + 1, 'gueltig': len(rows['r']), 'sek': round(time.time() - t0)}), flush=True)
    return {k: np.asarray(v) for k, v in rows.items()}, fails, altregel, border


def intervall(v):
    return np.quantile(v, [0.025, 0.975], method='linear')


def zusammenfassung(name, n, n1, n0, g, punkt, rows, fails, altregel, border, nboot=1000):
    valid = len(rows['r'])
    out = {'name': name, 'N': n, 'faelle': n1, 'kontrollen': n0, 'G_union': g, 'punkt': punkt, 'gueltig': valid,
           'ausfaelle': fails, 'altregel_verletzt': altregel, 'sep_grenzfaelle': border,
           'schritte_max': int(rows['steps'].max()) if valid else None,
           'gueltig_60': int(np.sum(rows['steps'] <= 60)) if valid else 0}
    if valid >= 950:
        out['or_ci'] = intervall(rows['or']).tolist()
        out['beta_ci'] = intervall(rows['beta']).tolist()
        out['dauc_ci'] = intervall(rows['dauc']).tolist()
    else:
        out['or_ci'] = out['beta_ci'] = out['dauc_ci'] = None
    if out['gueltig_60'] >= 950:
        v60 = rows['steps'] <= 60
        out['or_ci_60'] = intervall(rows['or'][v60]).tolist()
        out['dauc_ci_60'] = intervall(rows['dauc'][v60]).tolist()
    else:
        out['or_ci_60'] = out['dauc_ci_60'] = None
    return out


# ----------------------------------------------------------------------------------------------
# Datenaufbereitung (nur Leser; gebundene Originalfunktionen)
# ----------------------------------------------------------------------------------------------
def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def lies(path, cols):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pandas()


IDX_COLS = ['subject_id', 'study_id', 'split', 'ecg_time', 'age', 'extraction_status', 'reference_available']


def daten_h1_rbbb():
    from werkzeuge.b_test.statistik import raw_design, fixed_design, draws
    from werkzeuge.b_test.daten import select_index
    base = (str(data_path("b.aw172")) + "/")
    h1 = select_index(lies(base + 'H1_candidate_private.parquet', IDX_COLS), 'reduced_lvef_strict').subject_id.to_numpy()
    h4 = select_index(lies(base + 'H4_candidate_private.parquet', IDX_COLS), 'cardiomyopathy_icd').subject_id.to_numpy()
    tech = lies(base + 'technical_index_private.parquet', ['subject_id']).subject_id.to_numpy()
    union = np.unique(np.concatenate([h1, h4, tech]))
    hu = hashlib.sha256(union.tobytes()).hexdigest()
    soll = 'e4494075d5d4a9e28fb6f683c551c6260d7f8e1687607212a86fa26cdd58d1a2'
    if hu != soll:
        raise SystemExit('GATE verletzt: U172-Hash %s' % hu)
    ph = sha256_datei(str(data_path("b.aw171_parameters_json")))
    if ph != '17784e0a21811d2f61e9def653d7b1a5d9dfbb8cd1bf2088b508da0aa0938b72':
        raise SystemExit('GATE verletzt: Parameterhash %s' % ph)
    marker = 'qrs_t_normalized_discrete_frechet'
    scale = json.load(open(str(data_path("b.aw171_parameters_json"))))['endpoints']['H1']['markers'][marker]['scale']
    cols = ['subject_id', 'y', 'classical_rbbb_pattern_positive', 'age', 'sex_numeric', 'heart_rate', 'qrs_duration_ms',
            'qtc_fridericia_ms', 'spatial_qrs_t_angle_deg', marker]
    c = lies(base + 'H1_C_private.parquet', cols)
    c = c[c.classical_rbbb_pattern_positive == True].copy()  # noqa: E712
    x, names = raw_design(c, marker)
    if names != scale['names']:
        raise SystemExit('GATE verletzt: Designspalten')
    z = fixed_design(x, scale)
    y = c.y.to_numpy(float)
    ids = c.subject_id.to_numpy()
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    if (len(y), n1, n0) != (791, 146, 645):
        raise SystemExit('GATE verletzt: Stratum N/Faelle/Kontrollen %s' % ((len(y), n1, n0),))
    if len(np.unique(ids)) != len(ids):
        raise SystemExit('GATE verletzt: Personen im Stratum nicht eindeutig')
    for r in (0, 1, 999):
        if not np.array_equal(draws(union, ids, r), multiplicities(len(union), r)[np.searchsorted(union, ids)].astype(float)):
            raise SystemExit('GATE verletzt: Ziehungsrezept weicht vom Original ab (r=%d)' % r)
    return z, y, ids, union, {'U172_sha256': hu, 'parameter_sha256': ph}


def daten_s3b_049():
    from werkzeuge.b_test.statistik import raw_design, fixed_design, draws
    from werkzeuge.b_test.daten import select_index, masks
    mp = str(data_path("b.b_freigabe_input_manifest_v5_json"))
    mh = sha256_datei(mp)
    if mh != '5bfa03ae40fb812f6d037b618c8e7b5a76512b656a5bf3ab1f02265a8223b49e':
        raise SystemExit('GATE verletzt: Manifesthash')
    man = json.load(open(mp))['hypotheses']
    ids_union = []
    frame049 = None
    cols049 = IDX_COLS + ['sex', 'machine_heart_rate_bpm', 'machine_metadata_usable_for_primary', 'qrs_duration_ms',
                          'qtc_fridericia_ms', 'spatial_qrs_t_angle_deg', 'qrs_t_frechet_beat_median', 'label_state',
                          'eligibility_state', 'ehr_time_alignment_confirmatory_eligible']
    for bef, ep in (('BEF-0063', 'reduced_lvef_strict'), ('BEF-0049', 'cardiomyopathy_icd'), ('BEF-0087', 'echo_rv_systolic_dysfunction')):
        e = man[bef]
        pth = e['path']
        if 'locked_test' in pth.lower():
            raise SystemExit('ABBRUCH: Pfad mit locked_test')
        if sha256_datei(pth) != e['sha256']:
            raise SystemExit('GATE verletzt: Dateihash ' + bef)
        f = lies(pth, cols049 if bef == 'BEF-0049' else IDX_COLS)
        ids_union.append(select_index(f, ep).subject_id.to_numpy())
        if bef == 'BEF-0049':
            frame049 = f
            comp = e['comparator']
            marker = e['marker']
    union = np.unique(np.concatenate(ids_union))
    pe = json.load(open(ROOT + '/werkzeuge/b_test/parameter_entwicklung.json'))
    params = pe['hypotheses']['BEF-0049']['scaling_B3']
    if params.get('status') != 'gebunden':
        raise SystemExit('GATE verletzt: Skalierung ungebunden')
    f = select_index(frame049, 'cardiomyopathy_icd')
    zf, good, flow = masks(f, 'cardiomyopathy_icd', marker, comp)
    good = good & zf.sex.eq('F')
    sub = zf.loc[good]
    x, names = raw_design(sub, marker, comp)
    if names != params['names']:
        raise SystemExit('GATE verletzt: Designspalten')
    idx = [i for i, nm in enumerate(names) if nm != 'sex_numeric']
    x = x[:, idx]
    params = dict(params, **{k: [params[k][i] for i in idx] for k in ('names', 'mean', 'sd', 'keep')})
    z = fixed_design(x, params)
    y = sub.label_state.eq('positive').to_numpy(float)
    ids = sub.subject_id.to_numpy()
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    if (len(y), n1, n0) != (8851, 145, 8706):
        raise SystemExit('GATE verletzt: Stratum N/Faelle/Kontrollen %s' % ((len(y), n1, n0),))
    if len(np.unique(ids)) != len(ids):
        raise SystemExit('GATE verletzt: Personen im Stratum nicht eindeutig')
    for r in (0, 1, 999):
        if not np.array_equal(draws(union, ids, r), multiplicities(len(union), r)[np.searchsorted(union, ids)].astype(float)):
            raise SystemExit('GATE verletzt: Ziehungsrezept weicht vom Original ab (r=%d)' % r)
    return z, y, ids, union, {'manifest_sha256': mh}


# ----------------------------------------------------------------------------------------------
# Lauf
# ----------------------------------------------------------------------------------------------
def analysiere(name, z, y, ids, union, nboot=1000):
    punkt = evaluate(z, y, np.ones(len(y)))
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    # Eignung der Originalstichprobe: mindestens 100 Personen je Klasse
    if min(len(np.unique(ids[y == 1])), len(np.unique(ids[y == 0]))) < 100:
        raise SystemExit('Eignung der Originalstichprobe verletzt')
    rows, fails, altregel, border = bootstrap(z, y, ids, union, nboot, 200, name)
    res = zusammenfassung(name, len(y), n1, n0, len(union), punkt, rows, fails, altregel, border, nboot)
    os.makedirs(PRIVAT, exist_ok=True)
    with open('%s/draws_%s.csv' % (PRIVAT, name.replace('/', '_')), 'w', encoding='utf8') as fh:
        fh.write('r,beta,or,dauc,schritte,marge\n')
        for i in range(len(rows['r'])):
            fh.write('%d,%.17g,%.17g,%.17g,%d,%.6g\n' % (rows['r'][i], rows['beta'][i], rows['or'][i], rows['dauc'][i], rows['steps'][i], rows['margin'][i]))
    return res


def fmt_ci(ci, nd=4):
    return 'nicht berechnet (weniger als 950 gueltig)' if ci is None else '[%.*f; %.*f]' % (nd, ci[0], nd, ci[1])


def schreibe(ergebnisse, kopf, laufzeit, peak):
    zeilen = ['AW-241 Gegenrechnung B-R1 (unabhaengig, eigener Code), Skript b_r1_gegenrechnung.py',
              'Skript-SHA-256: %s' % kopf['skript_sha'],
              'Laufzeit gesamt: %.1f s; Speicherspitze (Arbeitssatz): %.0f MiB' % (laufzeit, peak / 2 ** 20),
              'Gate-Hashes: ' + json.dumps(kopf['gates'], sort_keys=True),
              'Regeln: 1.000 Ziehungen, Seed-Rezept [1282026, r], Gueltigkeit je Ziehung = endlich, beide Klassen, N>p, voller Rang,',
              '  keine (Quasi-)Separation (LP), Konvergenz (Newton, Ridge 1e-6, max. 200 Schritte, Score/sum(w) und Score/N < 1e-7);',
              '  Intervall nur bei mindestens 950 gueltigen Ziehungen; Perzentile 2,5/97,5 % (linear).', '']
    for r in ergebnisse:
        p = r['punkt']
        zeilen += ['== %s ==' % r['name'],
                   'N = %d, Faelle = %d, Kontrollen = %d, Personen der Union G = %d' % (r['N'], r['faelle'], r['kontrollen'], r['G_union']),
                   'Punkt: beta = %.6f, OR je Entwicklungs-SD = %.6f, Delta-AUC = %.6f (AUC mit Marker %.6f, ohne %.6f)' % (p['beta'], p['or'], p['dauc'], p['auc1'], p['auc0']),
                   'Gueltige Ziehungen: %d von 1000; Ausfaelle (Anzahl je Grund): %s' % (r['gueltig'], json.dumps(r['ausfaelle'], ensure_ascii=False) if r['ausfaelle'] else 'keine'),
                   'OR-Bootstrap-Perzentilintervall: %s' % fmt_ci(r['or_ci']),
                   'beta-Bootstrap-Perzentilintervall: %s' % fmt_ci(r['beta_ci']),
                   'Delta-AUC-Bootstrap-Perzentilintervall: %s' % fmt_ci(r['dauc_ci']),
                   'Diagnose: Ziehungen mit weniger als 100 verschiedenen Personen in mindestens einer Klasse (Altregel haette verworfen): %d' % r['altregel_verletzt'],
                   'Diagnose: Separations-Grenzfaelle (Marge zwischen 1e-8 und 1e-6, als nicht separiert gewertet): %d' % r['sep_grenzfaelle'],
                   'Diagnose: maximale Newton-Schritte in gueltigen Ziehungen: %s; gueltige Ziehungen bei Grenze 60 Schritte: %d' % (r['schritte_max'], r['gueltig_60']),
                   'Diagnose 60-Schritte-Variante: OR-Intervall %s, Delta-AUC-Intervall %s' % (fmt_ci(r['or_ci_60']), fmt_ci(r['dauc_ci_60'])), '']
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf8', newline='\n') as fh:
        fh.write('\n'.join(zeilen) + '\n')


def lauf(args):
    t0 = time.time()
    skript_sha = sha256_datei(os.path.abspath(__file__))
    ergebnisse = []
    gates = {}
    z, y, ids, union, g = daten_h1_rbbb()
    gates.update(g)
    ergebnisse.append(analysiere('S15i_H1_rbbb', z, y, ids, union))
    del z, y, ids, union
    z, y, ids, union, g = daten_s3b_049()
    gates.update(g)
    ergebnisse.append(analysiere('S3b_049_BEF-0049_F', z, y, ids, union))
    peak = max(PEAK['rss'], psutil.Process().memory_info().peak_wset)
    schreibe(ergebnisse, {'skript_sha': skript_sha, 'gates': gates}, time.time() - t0, peak)
    json.dump(ergebnisse, open(PRIVAT + '/ergebnisse.json', 'w', encoding='utf8'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('fertig', OUT_TXT, 'sha256', sha256_datei(OUT_TXT))


# ----------------------------------------------------------------------------------------------
# Test an synthetischen Daten
# ----------------------------------------------------------------------------------------------
def test(args):
    from scipy.optimize import minimize
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    ok = True

    def check(name, cond, info=''):
        nonlocal ok
        ok &= bool(cond)
        print(('OK    ' if cond else 'FEHLER'), name, info)

    n, p = 3000, 6
    X = rng.normal(size=(n, p))
    true = np.array([-1.5, 0.4, -0.3, 0.2, 0.0, 0.7, 0.5])   # letzter = Marker, OR = exp(0.5)
    Z = np.column_stack([np.ones(n), X])
    y = (rng.uniform(size=n) < expit(Z @ true)).astype(float)
    w = np.ones(n)
    b, conv, st = fit_logit(Z, y, w)
    pen = np.r_[0, np.full(Z.shape[1] - 1, RIDGE)]
    fobj = lambda bb: np.sum(w * (np.logaddexp(0, Z @ bb) - y * (Z @ bb))) + 0.5 * np.sum(pen * bb * bb)
    gobj = lambda bb: Z.T @ (w * (expit(Z @ bb) - y)) + pen * bb
    ref = minimize(fobj, np.zeros(Z.shape[1]), jac=gobj, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
    check('Newton gegen scipy-BFGS (Ridge)', conv and np.max(np.abs(b - ref.x)) < 1e-6, 'max|diff|=%.2e Schritte=%d' % (np.max(np.abs(b - ref.x)), st))
    glm = sm.GLM(y, Z, family=sm.families.Binomial()).fit(tol=1e-12)
    check('Newton gegen statsmodels-GLM (ungestraft, Ridge 1e-6 vernachlaessigbar)', np.max(np.abs(b - glm.params)) < 1e-4, 'max|diff|=%.2e' % np.max(np.abs(b - glm.params)))
    check('Marker-OR nahe wahrem OR', abs(b[-1] - 0.5) < 0.15, 'beta=%.3f' % b[-1])
    # Gewichte = Wiederholung von Zeilen
    m = rng.integers(0, 4, size=n).astype(float)
    act = m > 0
    b_w, c1, _ = fit_logit(Z[act], y[act], m[act])
    rep = np.repeat(np.arange(n), m.astype(int))
    b_r, c2, _ = fit_logit(Z[rep], y[rep], np.ones(len(rep)))
    check('Multiplizitaeten gleich Zeilenwiederholung', c1 and c2 and np.max(np.abs(b_w - b_r)) < 1e-8, 'max|diff|=%.2e' % np.max(np.abs(b_w - b_r)))
    # Penalisierte Zielfunktion mit Gewichten gegen BFGS
    w2 = m[act]
    Z2, y2 = Z[act], y[act]
    f2 = lambda bb: np.sum(w2 * (np.logaddexp(0, Z2 @ bb) - y2 * (Z2 @ bb))) + 0.5 * np.sum(pen * bb * bb)
    g2 = lambda bb: Z2.T @ (w2 * (expit(Z2 @ bb) - y2)) + pen * bb
    ref2 = minimize(f2, np.zeros(Z2.shape[1]), jac=g2, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
    check('gewichteter Newton gegen BFGS', np.max(np.abs(b_w - ref2.x)) < 1e-6)
    # AUC
    for tie in (False, True):
        s = Z @ b
        if tie:
            s = np.round(s, 1)
        a_mine = auc_w(y, s, m)
        a_ref = roc_auc_score(y, s, sample_weight=m)
        check('gewichtete AUC gegen sklearn (Ties=%s)' % tie, abs(a_mine - a_ref) < 1e-12, 'diff=%.1e' % abs(a_mine - a_ref))
    # Separation
    xs = rng.normal(size=(400, 2))
    ys = (xs[:, 0] > 0).astype(float)
    Zs = np.column_stack([np.ones(400), xs])
    check('vollstaendige Separation erkannt', separation_margin(Zs, ys) > SEP_TOL, 'Marge=%.3g' % separation_margin(Zs, ys))
    # Quasi-Separation: Klassen sind ueberlappungsfrei bis auf Punkte auf der Trennebene
    xq = np.vstack([xs, np.zeros((6, 2))])
    yq = np.r_[ys, [0, 1, 0, 1, 0, 1]].astype(float)
    xq[-6:, 0] = 0.0
    Zq = np.column_stack([np.ones(len(xq)), xq])
    Zq[:, 1] = np.where(xq[:, 0] == 0.0, 0.0, Zq[:, 1])
    mq = separation_margin(Zq, yq)
    check('Quasi-Separation erkannt', mq > SEP_TOL, 'Marge=%.3g' % mq)
    mn = separation_margin(Z, y)
    check('keine Separation bei ueberlappenden Daten', mn <= SEP_TOL, 'Marge=%.3g' % mn)
    # evaluate: Ablehnungsgruende
    for nm, fn, soll in (('Klasse fehlt', lambda: evaluate(Z, np.zeros(n), w), 'Klasse fehlt'),
                         ('Separation', lambda: evaluate(Zs, ys, np.ones(400)), 'Separation'),
                         ('Rangdefizit', lambda: evaluate(np.column_stack([Z, Z[:, 1]]), y, w), 'Rangdefizit'),
                         ('N<=p', lambda: evaluate(Z[:5], np.array([0.0, 1.0, 0.0, 1.0, 0.0]), w[:5]), 'N<=p'),
                         ('nicht endlich', lambda: evaluate(np.where(np.arange(n)[:, None] == 0, np.nan, Z), y, w), 'nicht endlich')):
        try:
            fn()
            check('Ablehnung ' + nm, False, 'nicht abgelehnt')
        except Invalid as e:
            check('Ablehnung ' + nm, str(e) == soll, str(e))
    e = evaluate(Z, y, w)
    # Delta-AUC gegen sklearn: Basis ohne Marker
    b0, _, _ = fit_logit(Z[:, :-1], y, w)
    d_ref = roc_auc_score(y, Z @ b) - roc_auc_score(y, Z[:, :-1] @ b0)
    check('Delta-AUC gegen sklearn', abs(e['dauc'] - d_ref) < 1e-12, 'dAUC=%.5f' % e['dauc'])
    # Schrittzaehler: sehr strenge Grenze erzwingt Nichtkonvergenz
    _, c3, _ = fit_logit(Z, y, w, max_steps=1)
    check('Schrittgrenze greift', not c3)
    # Ziehungsrezept: Summe der Multiplizitaeten = G, Reproduzierbarkeit
    m1, m2 = multiplicities(500, 3), multiplicities(500, 3)
    check('Ziehungsrezept reproduzierbar, Summe = G', np.array_equal(m1, m2) and m1.sum() == 500 and not np.array_equal(m1, multiplicities(500, 4)))
    # Mini-Bootstrap: wahrer OR im Intervall, alle gueltig
    union = np.arange(n) + 1000
    rows, fails, alt, bord = bootstrap(Z, y, union, union, nboot=60, label='synth')
    lo, hi = intervall(rows['beta'])
    check('Mini-Bootstrap: 60 Ziehungen gueltig, wahres beta im Intervall', len(rows['r']) == 60 and lo < 0.5 < hi, 'gueltig=%d CI=[%.3f;%.3f]' % (len(rows['r']), lo, hi))
    # Linear-Quantile gegen manuelle Interpolation
    v = np.sort(rng.normal(size=1000))
    k = 0.025 * 999
    man = v[int(k)] + (k - int(k)) * (v[int(k) + 1] - v[int(k)])
    check('Perzentil linear', abs(intervall(v)[0] - man) < 1e-12)
    # Zeitprobe fuer LP bei N ~ 9000
    big = rng.normal(size=(8851, 7))
    Zb = np.column_stack([np.ones(8851), big])
    yb = (rng.uniform(size=8851) < expit(-4.0 + 0.5 * big[:, -1])).astype(float)
    t0 = time.time()
    evaluate(Zb, yb, np.ones(8851))
    print('Zeit einer Auswertung bei N=8851: %.2f s' % (time.time() - t0))
    print('ALLE TESTS BESTANDEN' if ok else 'TESTS FEHLGESCHLAGEN')
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--freigabe-eingetragen', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    start_waechter()
    if a.test:
        sys.exit(test(a))
    if a.run:
        if not a.freigabe_eingetragen:
            raise SystemExit('Beginn-Eintrag in TESTZUGRIFFE.md fehlt (--freigabe-eingetragen)')
        lauf(a)
    else:
        ap.print_help()
