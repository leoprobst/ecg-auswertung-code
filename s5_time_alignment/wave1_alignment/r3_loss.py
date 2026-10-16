"""AW-242 S5-R3: exklusive Verlustzerlegung und Vergleich Eingeschlossene/Ausgeschlossene (post hoc, beschreibend).

Eingabe: AW-110-Arbeitsdatenbank (lesend), Masterindex (nur Spalten age, sex, hospital_expire_flag, anchor_year_group).
Nur Entwicklungspartitionen. Ausgaben ohne Kennungen: nachrechnung/AW242/S5_R3_*.csv, private Zwischenstaende unter <local path>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r3')
import numpy as np, pandas as pd, duckdb
from aw242_kern import D, K, SHIFT, wilson, smd_cont, smd_bin, chart_slices
from aw242_data import load_audit

OUT = G.OUT
ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW242'; PUB.mkdir(parents=True, exist_ok=True)
MASTER = str(data_path("s5.mimic_validation_master_ecg_master_index_parquet"))


def kz(n):
    n = int(n)
    return '<10' if 0 < n < 10 else str(n)


def ecg_flags(ecg, chart, window=1800.0):
    """Je RR-geeignetem ICU-EKG: Randbedingung, Chartzeilen im Aufenthalt, Vollabdeckung, Abdeckungsanteil."""
    cs = chart['subject_id'].to_numpy(); cst = chart['stay'].to_numpy(); ct = chart['t'].to_numpy(float)
    sl = chart_slices(cs, cst)
    sub = ecg['subject_id'].to_numpy(); stay = ecg['stay'].to_numpy(); t = ecg['t'].to_numpy(float)
    st = ecg['start'].to_numpy(float); en = ecg['end'].to_numpy(float)
    n = len(ecg)
    b_ok = np.zeros(n, bool); has = np.zeros(n, bool); comp = np.zeros(n, bool); cov = np.zeros(n)
    cache = {}
    for i in range(n):
        b_ok[i] = (t[i] - 720 * 60 >= st[i]) and (t[i] + 720 * 60 <= en[i])
        key = (int(sub[i]), int(stay[i]))
        if key not in cache:
            if key in sl:
                a, b = sl[key]
                tt = ct[a:b]; m = (tt >= st[i]) & (tt <= en[i]); cache[key] = tt[m] if m.any() else None
            else:
                cache[key] = None
        tt = cache[key]
        if tt is None:
            continue
        has[i] = True
        tg = t[i] + SHIFT
        k = np.searchsorted(tt, tg)
        left = np.clip(k - 1, 0, len(tt) - 1); right = np.clip(k, 0, len(tt) - 1)
        dt = np.minimum(np.abs(tt[left] - tg), np.abs(tt[right] - tg))
        good = (dt <= window) & (tg >= st[i]) & (tg <= en[i])
        cov[i] = good.mean(); comp[i] = bool(good.all())
    return b_ok, has, comp, cov


def main():
    t0 = time.time()
    data = load_audit()
    ecg, prim, people, chart = data['ecg'], data['prim'], data['people'], data['chart']
    import pickle
    base = pickle.load(open(OUT / 'base_persons.pkl', 'rb'))['persons']
    b_ok, has, comp, cov = ecg_flags(ecg, chart)
    ecg = ecg.assign(b_ok=b_ok, has=has, comp=comp, cov=cov)
    # Paritaet: vollstaendige EKG
    assert int(ecg.comp.sum()) == int(base.n_complete.sum()), (int(ecg.comp.sum()), int(base.n_complete.sum()))
    # Personentabelle (alle ICU-verknuepften Personen)
    P = people.rename(columns={'fold': 'fold'}).set_index('subject_id')
    g = ecg.groupby('subject_id')
    pe = pd.DataFrame({'n_rr': g.size(), 'n_bound': g.b_ok.sum(), 'n_has_chart': g.has.sum(), 'n_comp': g.comp.sum(),
                       'n_bound_has': g.apply(lambda d: int((d.b_ok & d.has).sum())),
                       'best_cov_bound': g.apply(lambda d: float(d.loc[d.b_ok, 'cov'].max()) if d.b_ok.any() else np.nan),
                       'hr_sd': g.hf.apply(lambda s: float(np.std(s.to_numpy(), ddof=1)) if len(s) > 1 else np.nan),
                       'hr_mean': g.hf.mean()})
    gp = prim.assign(los_h=(prim['end'] - prim['start']) / 3600.0).groupby('subject_id')
    pp = pd.DataFrame({'n_icu_ecg': gp.size(), 'los_max_d': gp.los_h.max() / 24.0,
                       'any_bound_all': gp.apply(lambda d: bool(((d.t - 720 * 60 >= d.start) & (d.t + 720 * 60 <= d['end'])).any())),
                       'unit': gp.unit.agg(lambda s: sorted(s.value_counts().items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0])})
    P = P.join(pp).join(pe)
    for c_ in ('n_rr', 'n_bound', 'n_has_chart', 'n_comp', 'n_bound_has'):
        P[c_] = P[c_].fillna(0).astype(int)
    # Schaetzer aus der Basisrechnung
    B = base.set_index('subject_id')[['n_complete', 'ok', 'offset', 'width', 'minimum', 'c_n', 'c_w', 'c_m', 'c_edge', 'c_conn']]
    P = P.join(B)
    P['ok'] = P['ok'].fillna(False).astype(bool)
    # Stammdaten aus dem Masterindex
    c = duckdb.connect()
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.execute(("ATTACH '" + str(data_path("s5.uhrzeit_aw110_audit_duckdb")) + "' AS a (READ_ONLY)"))
    md = c.execute(f"""SELECT p.subject_id, min(m.age) AS age0, any_value(m.sex) AS sex, max(m.hospital_expire_flag) AS died,
                       arg_min(m.anchor_year_group, m.ecg_time) AS year, count(*) AS n_master
                       FROM read_parquet('{MASTER}') m JOIN a.primary_ecg p USING(subject_id, study_id) GROUP BY p.subject_id""").df().set_index('subject_id')
    allecg = c.execute('SELECT subject_id, count(*) AS n_all_ecg FROM a.ecg GROUP BY subject_id').df().set_index('subject_id')
    c.close()
    P = P.join(md).join(allecg)
    P.to_pickle(OUT / 'r3_persons.pkl')

    rows_wf = []; rows_cmp = []; rows_cond = []
    for f in ('train', 'validation'):
        S = P[P.fold == f]
        n1 = len(S)
        L = {}
        L['L1'] = n1
        L['L2'] = int((S.n_rr > 0).sum())
        L['L3'] = int((S.n_bound > 0).sum())
        L['L4'] = int((S.n_comp > 0).sum())
        L['L5'] = int((S.n_comp >= 3).sum())
        L['L6'] = int(S.ok.sum())
        names = [('L1', 'ECG nominal in eindeutigem ICU-Aufenthalt'), ('L2', 'mindestens ein RR-geeignetes ICU-EKG'),
                 ('L3', 'mindestens ein EKG mit je 12 h Abstand zu beiden Aufenthaltsgrenzen'), ('L4', 'mindestens ein EKG mit Referenz in jedem der 289 Offsets'),
                 ('L5', 'mindestens drei vollstaendige EKG'), ('L6', 'Offset identifizierbar (alle Kurvenkriterien)')]
        prev = None
        for key, name in names:
            rows_wf.append(dict(fold=f, stufe=key, beschreibung=name, personen=L[key], anteil_an_L1=L[key] / n1, verlust_zur_Vorstufe=(prev - L[key]) if prev is not None else None,
                                verlust_anteil_an_L1=((prev - L[key]) / n1) if prev is not None else None))
            prev = L[key]
        # Zerlegung der Verluste nach Gruende
        extra = {}
        # L1 -> L2: ICU-Person ohne RR-geeignetes EKG
        extra['L1_L2_kein_RR_EKG'] = int((S.n_rr == 0).sum())
        # L2 -> L3: Aufenthaltsdauer/Randlage
        s2 = S[S.n_rr > 0]
        no_b = s2[s2.n_bound == 0]
        extra['L2_L3_gesamt'] = len(no_b)
        extra['L2_L3_alle_Aufenthalte_unter_24h'] = int((no_b.los_max_d < 1.0).sum())
        extra['L2_L3_Aufenthalt_ab_24h_aber_EKG_am_Rand'] = int((no_b.los_max_d >= 1.0).sum())
        # L3 -> L4: Referenzluecken
        s3 = S[S.n_bound > 0]
        no_c = s3[s3.n_comp == 0]
        extra['L3_L4_gesamt'] = len(no_c)
        extra['L3_L4_kein_Chartwert_im_Aufenthalt_fuer_Randfreie_EKG'] = int((no_c.n_bound_has == 0).sum())
        extra['L3_L4_Chartwerte_vorhanden_aber_Luecke_ueber_30min'] = int((no_c.n_bound_has > 0).sum())
        extra['L3_L4_beste_Abdeckung_ab_90_Prozent_der_Offsets'] = int((no_c.best_cov_bound >= 0.9).sum())
        extra['L3_L4_beste_Abdeckung_unter_50_Prozent'] = int((no_c.best_cov_bound < 0.5).sum())
        # L4 -> L5: EKG-Anzahl
        s4 = S[S.n_comp > 0]
        extra['L4_L5_gesamt'] = int((s4.n_comp < 3).sum())
        extra['L4_L5_genau_1_vollst_EKG'] = int((s4.n_comp == 1).sum())
        extra['L4_L5_genau_2_vollst_EKG'] = int((s4.n_comp == 2).sum())
        extra['L4_L5_davon_mit_mind3_randfreien_EKG'] = int(((s4.n_comp < 3) & (s4.n_bound >= 3)).sum())
        # L5 -> L6: Kurvenkriterien (exklusive Reihenfolge Minimum, Breite, Zusammenhang, Rand) und einzeln
        s5 = S[S.n_comp >= 3]
        fail_m = ~s5.c_m.astype(bool)
        fail_w = ~s5.c_w.astype(bool)
        fail_c = ~s5.c_conn.astype(bool)
        fail_e = ~s5.c_edge.astype(bool)
        extra['L5_L6_gesamt'] = int((~s5.ok).sum())
        extra['L5_L6_Reihenfolge1_Minimum_ueber_5'] = int(fail_m.sum())
        r1 = ~fail_m
        extra['L5_L6_Reihenfolge2_Breite_ueber_120'] = int((r1 & fail_w).sum())
        r2 = r1 & ~fail_w
        extra['L5_L6_Reihenfolge3_Zone_nicht_zusammenhaengend'] = int((r2 & fail_c).sum())
        r3 = r2 & ~fail_c
        extra['L5_L6_Reihenfolge4_Zone_am_Suchrand'] = int((r3 & fail_e).sum())
        extra['L5_einzeln_Minimum_ueber_5'] = int(fail_m.sum())
        extra['L5_einzeln_Breite_ueber_120'] = int(fail_w.sum())
        extra['L5_einzeln_Zone_nicht_zusammenhaengend'] = int(fail_c.sum())
        extra['L5_einzeln_Zone_am_Suchrand'] = int(fail_e.sum())
        extra['L5_median_Breite_nicht_identifizierbar'] = float(s5.loc[~s5.ok, 'width'].median())
        # Bedingte Ausbeute
        cond = []
        for lab, num, den in (('identifizierbar / mindestens 3 vollstaendige EKG', L['L6'], L['L5']), ('identifizierbar / mindestens 1 vollstaendiges EKG', L['L6'], L['L4']),
                              ('identifizierbar / mindestens 1 randfreies EKG', L['L6'], L['L3']), ('identifizierbar / ICU-verknuepfte Personen', L['L6'], L['L1']),
                              ('mindestens 3 vollstaendige EKG / mindestens 1 vollstaendiges EKG', L['L5'], L['L4'])):
            lo, hi = wilson(num, den)
            rows_cond.append(dict(fold=f, groesse=lab, zaehler=num, nenner=den, anteil=num / den, wilson_untere=lo, wilson_obere=hi))
        for k_, v_ in extra.items():
            rows_wf.append(dict(fold=f, stufe='Zerlegung', beschreibung=k_, personen=v_, anteil_an_L1=(v_ / n1) if not isinstance(v_, float) or v_ > 1000 else None))
    wf = pd.DataFrame(rows_wf)
    wf['personen'] = wf['personen'].map(lambda v: v)
    wf.to_csv(OUT / 'r3_waterfall_raw.csv', index=False)
    pd.DataFrame(rows_cond).to_csv(OUT / 'r3_cond_raw.csv', index=False)

    # Vergleiche
    def describe(S, name):
        out = {}
        out['n'] = len(S)
        out['age_mean'] = S.age0.mean(); out['age_sd'] = S.age0.std(); out['age_median'] = S.age0.median()
        out['male'] = float((S.sex == 'M').mean()); out['died'] = float(S.died.fillna(0).astype(float).gt(0).mean())
        out['los_median_d'] = S.los_max_d.median(); out['los_mean_d'] = S.los_max_d.mean()
        out['n_icu_ecg_median'] = S.n_icu_ecg.median(); out['n_rr_median'] = S.n_rr.median()
        out['n_all_ecg_median'] = S.n_all_ecg.median()
        out['hr_sd_median'] = S.hr_sd.median(); out['hr_mean_mean'] = S.hr_mean.mean()
        return out

    def compare(A, B, la, lb, fold, vergleich):
        res = []
        def add(var, level, a_val, b_val, smd, kind):
            res.append(dict(fold=fold, vergleich=vergleich, variable=var, stufe=level, gruppe_A=la, gruppe_B=lb, wert_A=a_val, wert_B=b_val, smd=smd, art=kind))
        # kontinuierlich
        for var, col in (('Alter (Jahre) beim ersten ICU-EKG', 'age0'), ('laengster ICU-Aufenthalt (Tage)', 'los_max_d'), ('Anzahl ICU-verknuepfter EKG', 'n_icu_ecg'),
                         ('Anzahl RR-geeigneter ICU-EKG', 'n_rr'), ('Anzahl aller EKG der Person (Entwicklungsfold)', 'n_all_ecg'),
                         ('SD der EKG-Herzfrequenz (min^-1, mind. 2 EKG)', 'hr_sd'), ('mittlere EKG-Herzfrequenz (min^-1)', 'hr_mean')):
            a = A[col].astype(float); b = B[col].astype(float)
            add(var, 'Median [Q1; Q3]', f'{a.median():.2f} [{a.quantile(.25):.2f}; {a.quantile(.75):.2f}]', f'{b.median():.2f} [{b.quantile(.25):.2f}; {b.quantile(.75):.2f}]', smd_cont(a, b), 'kontinuierlich')
        for var, col, pos in (('Maennliches Geschlecht (MIMIC-Feld)', 'sex', 'M'),):
            pa = float((A[col] == pos).mean()); pb = float((B[col] == pos).mean())
            add(var, 'Anteil', pa, pb, smd_bin(pa, pb), 'binaer')
        da = A.died.fillna(0).astype(float).gt(0); db = B.died.fillna(0).astype(float).gt(0)
        add('Krankenhaussterblichkeit (verknuepfte Aufnahme)', 'Anteil', float(da.mean()), float(db.mean()), smd_bin(float(da.mean()), float(db.mean())), 'binaer')
        for var, col in (('erste ICU-Einheit (haeufigste)', 'unit'), ('Ankerjahr-Gruppe', 'year')):
            levels = sorted(set(A[col].dropna()) | set(B[col].dropna()), key=str)
            for lv in levels:
                pa = float((A[col] == lv).mean()); pb = float((B[col] == lv).mean())
                na = int((A[col] == lv).sum()); nb = int((B[col] == lv).sum())
                if 0 < min(na, nb) < 10 or 0 < na < 10 or 0 < nb < 10:
                    add(var, str(lv), '<10 Personen in mindestens einer Gruppe', None, None, 'kategorial')
                else:
                    add(var, str(lv), pa, pb, smd_bin(pa, pb), 'kategorial')
        return res

    for f in ('train', 'validation'):
        S = P[P.fold == f]
        rows_cmp += compare(S[S.n_comp >= 3], S[S.n_comp < 3], 'mind. 3 vollstaendige EKG', 'alle uebrigen ICU-Personen', f, 'eingeschlossen (L5) gegen ausgeschlossen')
        rows_cmp += compare(S[S.ok], S[(S.n_comp >= 3) & ~S.ok], 'identifizierbar (L6)', 'mind. 3 vollst. EKG, nicht identifizierbar', f, 'identifizierbar gegen nicht identifizierbar')
        rows_cmp += compare(S[S.n_comp > 0], S[S.n_comp == 0], 'mind. 1 vollstaendiges EKG', 'ohne vollstaendiges EKG', f, 'L4 gegen Rest')
    cmp_df = pd.DataFrame(rows_cmp)
    cmp_df.to_csv(OUT / 'r3_compare_raw.csv', index=False)
    (OUT / 'r3_done.json').write_text(json.dumps({'seconds': time.time() - t0, 'peak_rss': G.peak(), 'persons': len(P)}))
    print('R3 fertig', round(time.time() - t0, 1), 's, RSS', G.peak())


if __name__ == '__main__':
    main()
