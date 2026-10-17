"""AW-255 S3 Machbarkeit: Aggregation der Signalguete (signal_guete.csv) und der Messbarkeit (messbar.csv) mit den Paar- und Extremwert-Tabellen.
Keine Assoziation Elektrolyt-EKG. Ausgabe maskiert (1 bis 9 -> "<10"), keine Kennungen.
Aufruf: <local path> -I s3_aggregat.py <ausgabe.json>

Saubere Entnahme (Dk) = mindestens ein 10-s-Block in Fenster Dk mit: kein NaN, keine Saettigung (>4,5 mV), keine Flachlinie (>=60 ms), Spannweite >= 0,3 mV,
5 bis 37 Schlaege, mittlerer RR 0,27 bis 2,0 s, Medianschlag >= 0,15 mV, Schlag-Schablonen-Korrelation: Median >= 0,90 und >= 80 % der Schlaege >= 0,80.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, json
sys.dont_write_bytecode = True
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

W = Path(str(data_path("s3.work_dir")))
c = duckdb.connect(str(W / 's3.duckdb'))
c.execute("SET memory_limit='700MB'; SET threads=2")
g = pd.read_csv(W / 'signal_guete.csv')
c.register('g_df', g[['CSN', 't_order', 'd1', 'd2', 'd3', 'n_blocks', 'best_sqi_d1', 'hr_d1', 'gruende']])
c.execute("create or replace table gu as select CAST(CSN as BIGINT) CSN, t_order, d1, d2, d3, n_blocks, best_sqi_d1, hr_d1, gruende from g_df")
one = lambda s: c.execute(s).fetchone()
q = lambda s: c.execute(s).fetchall()


def maske(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, np.integer)):
        return '<10' if 1 <= int(x) <= 9 else int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        return {k: maske(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [maske(v) for v in x]
    return x


R = {'version': 's3_aggregat v1 (AW-255)'}
R['kandidaten_entnahmen'] = one('select count(*) from gu')[0]
R['entnahmen_mit_temporaler_abdeckung_geprueft_sauber_D1_D2_D3'] = one('select sum(d1), sum(d2), sum(d3) from gu')
R['bloecke_je_entnahme_median'] = one('select median(n_blocks) from gu')[0]
# Gruende der Block-Ablehnung (Summe ueber alle Bloecke)
gr = {}
for s in g['gruende'].dropna():
    for k, v in json.loads(s).items():
        gr[k] = gr.get(k, 0) + v
R['block_gruende'] = gr
R['hr_sauberer_block_D1_quantile_05_50_95'] = one('select quantile_cont(hr_d1,[.05,.5,.95]) from gu where d1=1')[0]

# Paare konsekutiv
R['paare'] = {}
for n in ('kdraw_serum', 'kdraw_all'):
    t = f'{n}_cons_c_t'
    S = {}
    for d in (1, 2, 3):
        c.execute(f"""create or replace table pp as select p.*, ga.d{d} ga, gb.d{d} gb from {t} p
           left join gu ga on ga.CSN=p.CSN and ga.t_order=p.ta left join gu gb on gb.CSN=p.CSN and gb.t_order=p.tb""")
        for thr in (0.0, 0.5, 1.0):
            w = f"abs(dk)>={thr}"
            S[f'D{d}_dK_ab_{thr}'] = {
                'zeitlich_beidseits_paare': one(f'select count(*) from pp where {w} and a{d}>=10 and b{d}>=10')[0],
                'sauber_beidseits_paare_besuche_personen': one(f'select count(*), count(distinct CSN), count(distinct MRN) from pp where {w} and ga=1 and gb=1'),
                'sauber_und_ohne_behandlung_und_kaliumgabe': one(f'select count(*) from pp where {w} and ga=1 and gb=1 and n_kl=0 and n_kgabe=0')[0],
                'sauber_und_mit_k_senkender_behandlung': one(f'select count(*) from pp where {w} and ga=1 and gb=1 and n_kl>0')[0],
                'sauber_und_HF_beidseits': one(f'select count(*) from pp where {w} and ga=1 and gb=1 and hr_a is not null and hr_b is not null')[0],
                'sauber_und_abs_dHF_unter_10': one(f'select count(*) from pp where {w} and ga=1 and gb=1 and abs(hr_b-hr_a)<10')[0],
                'sauber_abfall_anstieg': one(f'select count(*) filter (where dk<0), count(*) filter (where dk>0) from pp where {w} and ga=1 and gb=1'),
            }
        if d == 1:
            S['D1_sauber_dK_sd_und_quantile'] = one('select stddev(dk), quantile_cont(dk,[.05,.25,.5,.75,.95]), count(*) from pp where ga=1 and gb=1')
            S['D1_sauber_dK_ab_0.5_sd'] = one('select stddev(dk), count(*) from pp where ga=1 and gb=1 and abs(dk)>=0.5')
            S['D1_sauber_abstand_h_quantile_05_50_95'] = one('select quantile_cont(gap_h,[.05,.5,.95]) from pp where ga=1 and gb=1')[0]
            S['D1_sauber_erste_K_bereiche'] = one('select count(*) filter (where ka<=3.5), count(*) filter (where ka>3.5 and ka<5.5), count(*) filter (where ka>=5.5) from pp where ga=1 and gb=1')
    R['paare'][n] = S

# Extremwerte
R['extremwerte'] = {}
CL = "('insulin_regular','glukose_bolus','calcium_iv','bikarbonat_iv','albuterol','k_binder')"
for n in ('kdraw_serum', 'kdraw_all'):
    E = {}
    for thr in (6.0, 6.5):
        c.execute(f"""create or replace table ex as select e.CSN, e.t_order, e.k, v.MRN,
           (select min(t_order) from {n} x where x.CSN=e.CSN and x.t_order>e.t_order+3600 and x.t_order<=e.t_order+12*3600) tn
           from {n} e join visit v on v.CSN=e.CSN where e.k>={thr}""")
        c.execute(f"""create or replace table ex2 as select x.*, ga.d1 a1, ga.d2 a2, ga.d3 a3, gb.d1 b1, gb.d2 b2, gb.d3 b3, kn.k kn,
           (select count(*) from trt t where t.CSN=x.CSN and t.t>x.t_order-3600 and t.t<=x.tn and t.cls in {CL}) n_trt
           from ex x left join gu ga on ga.CSN=CAST(x.CSN as BIGINT) and ga.t_order=x.t_order left join gu gb on gb.CSN=CAST(x.CSN as BIGINT) and gb.t_order=x.tn left join {n} kn on kn.CSN=x.CSN and kn.t_order=x.tn""")
        X = {'entnahmen': one('select count(*) from ex')[0]}
        for d in (1, 2, 3):
            X[f'sauber_D{d}_entnahmen_besuche_personen'] = one(f'select count(*), count(distinct CSN), count(distinct MRN) from ex2 where a{d}=1')
            X[f'sauber_D{d}_und_spaeterer_wert_1_12h_besuche'] = one(f'select count(distinct CSN) from ex2 where a{d}=1 and tn is not null')[0]
            X[f'sauber_D{d}_an_beiden_enden_besuche'] = one(f'select count(distinct CSN) from ex2 where a{d}=1 and b{d}=1')[0]
            X[f'sauber_D{d}_an_beiden_enden_mit_k_senkender_behandlung'] = one(f'select count(distinct CSN) from ex2 where a{d}=1 and b{d}=1 and n_trt>0')[0]
            X[f'sauber_D{d}_an_beiden_enden_ohne_behandlung'] = one(f'select count(distinct CSN) from ex2 where a{d}=1 and b{d}=1 and n_trt=0')[0]
            X[f'sauber_D{d}_an_beiden_enden_spaeterer_wert_unter_5_5'] = one(f'select count(distinct CSN) from ex2 where a{d}=1 and b{d}=1 and kn<5.5')[0]
        E[f'K_ab_{thr}'] = X
    R['extremwerte'][n] = E

# Messbarkeit
mp = W / 'messbar.csv'
if mp.exists():
    m = pd.read_csv(mp)
    M = {}
    for dur, grp in m.groupby('dur'):
        n = len(grp)
        gueltig = int((grp.get('valid', pd.Series([0] * n)) == 1).sum()) if 'valid' in grp else 0
        d = {'fenster': n, 'valid': gueltig, 'anteil_valid': gueltig / n if n else None,
             'gruende': grp['reason'].fillna('').value_counts().to_dict()}
        v = grp[grp.get('valid', 0) == 1] if 'valid' in grp else grp.iloc[0:0]
        for k in ('t_amp', 't_slope', 'qrs_ms', 'pr_ms', 'p_amp', 'tpe_ms', 'qt_ms', 'jt_ms', 't_over_r'):
            if k in v:
                x = v[k]
                d[k] = {'anteil_vorhanden_an_valid': float(x.notna().mean()) if len(v) else None, 'median': float(x.median()) if x.notna().any() else None,
                        'q05': float(x.quantile(.05)) if x.notna().any() else None, 'q95': float(x.quantile(.95)) if x.notna().any() else None}
        # Halbierungs-Wiederholbarkeit (Schlaege abwechselnd): SD der Differenz / sqrt(2), ICC ueber Entnahmen
        for k in ('t_amp', 't_slope', 'qrs_ms', 'pr_ms', 'p_amp'):
            a, b = f'{k}_a', f'{k}_b'
            if a in v and b in v:
                z = v[[a, b]].dropna()
                if len(z) >= 30:
                    dif = z[a] - z[b]
                    sw = float(dif.std() / np.sqrt(2))
                    tot = float(((z[a] + z[b]) / 2).std())
                    d[f'{k}_halbierung'] = {'n': int(len(z)), 'sd_innerhalb_aus_halbierung': sw, 'sd_zwischen_entnahmen': tot,
                                           'icc_halbierung_naeherung': float(max(0.0, 1 - (sw ** 2 / 2) / (tot ** 2))) if tot > 0 else None}
        M[str(dur)] = d
    R['messbarkeit_aw203_kette'] = M

# Messbarkeit auf Paarebene (Entnahmen aus Paaren mit sauberem D1 beidseits; Fenster um den besten sauberen Block)
mpp = W / 'messbar_paare.csv'
if mpp.exists():
    mm = pd.read_csv(mpp)
    mm['CSN'] = mm['CSN'].astype('int64')
    P = {}
    base = c.execute("""select p.CSN, p.ta, p.tb, p.ka, p.dk, p.gap_h, p.n_kl, p.n_kgabe, p.hr_a mhr_a, p.hr_b mhr_b from kdraw_serum_cons_c_t p
        join gu ga on ga.CSN=p.CSN and ga.t_order=p.ta and ga.d1=1 join gu gb on gb.CSN=p.CSN and gb.t_order=p.tb and gb.d1=1""").df()
    base['CSN'] = base['CSN'].astype('int64')
    for dur in (60, 300):
        m = mm[mm.dur == dur].copy()
        for k in ('valid', 't_amp', 't_slope', 'qrs_ms', 'pr_ms', 'qt_ms', 'tpe_ms', 'p_amp'):
            if k not in m:
                m[k] = np.nan
        m['valid'] = m['valid'].fillna(0)
        a = m.rename(columns=lambda x: x if x in ('CSN',) else x + '_a')
        b = m.rename(columns=lambda x: x if x in ('CSN',) else x + '_b')
        j = base.merge(a, left_on=['CSN', 'ta'], right_on=['CSN', 't_order_a']).merge(b, left_on=['CSN', 'tb'], right_on=['CSN', 't_order_b'])
        both = j[(j.valid_a == 1) & (j.valid_b == 1)]
        D = {'paare_sauber_D1': int(len(j)), 'entnahmen_valid_anteil': float((m.valid == 1).mean()),
             'entnahmen_gruende': m['reason'].fillna('').value_counts().to_dict(), 'paare_beide_valid': int(len(both)),
             'paare_beide_valid_personen': None}
        mo = {}
        for thr in (0.0, 0.5, 1.0):
            sel = both[both.dk.abs() >= thr]
            mo[f'dK_ab_{thr}'] = {'paare': int(len(sel)), 'ohne_behandlung_und_kaliumgabe': int(((sel.n_kl == 0) & (sel.n_kgabe == 0)).sum()),
                                  'mit_k_senkender_behandlung': int((sel.n_kl > 0).sum()),
                                  'hf_beidseits': int((sel.mhr_a.notna() & sel.mhr_b.notna()).sum())}
        D['nach_dK'] = mo
        # Marginale Streuung der Aenderung innerhalb Person (alle Paare, NICHT nach dK geteilt; keine Assoziation)
        for k, sc in (('t_amp', 1.0), ('t_slope', 1.0), ('qrs_ms', 1.0), ('pr_ms', 1.0), ('qt_ms', 1.0), ('p_amp', 1.0)):
            x = both[f'{k}_b'] - both[f'{k}_a']
            x = x.dropna()
            D[f'marginal_sd_aenderung_{k}'] = {'n': int(len(x)), 'sd': float(x.std()) if len(x) > 3 else None, 'mad_sd': float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 3 else None}
        # Vorhandensein beider Endpunkte
        D['beide_enden_vorhanden'] = {k: int((both[f'{k}_a'].notna() & both[f'{k}_b'].notna()).sum()) for k in ('t_amp', 't_slope', 'qrs_ms', 'pr_ms', 'qt_ms', 'tpe_ms', 'p_amp')}
        # Planungsrechnung (kein Befund): minimal nachweisbare Steigung bei 80 % Power, zweiseitig 5 %, aus marginaler SD der T-Amplituden-Aenderung (konservativ)
        sdK = float(both.dk.std()) if len(both) > 3 else None
        sdT = D['marginal_sd_aenderung_t_amp']['sd']
        sdT_robust = D['marginal_sd_aenderung_t_amp']['mad_sd']
        if sdK and sdT:
            pl = {}
            for lab, n_ in (('alle_beide_valid', len(both)), ('dK_ab_0.5', mo['dK_ab_0.5']['paare']), ('dK_ab_1.0', mo['dK_ab_1.0']['paare']), ('ohne_behandlung_dK_alle', mo['dK_ab_0.0']['ohne_behandlung_und_kaliumgabe'])):
                if n_ > 5:
                    sdk_sub = float(both[both.dk.abs() >= (0.5 if 'ab_0.5' in lab else 1.0 if 'ab_1.0' in lab else 0.0)].dk.std())
                    pl[lab] = {'n': int(n_), 'sd_dK': sdk_sub, 'mdb_beta_mV_je_mmol_80pct': float(2.8 * sdT / (sdk_sub * np.sqrt(n_))),
                               'mdb_beta_robust_sd': float(2.8 * sdT_robust / (sdk_sub * np.sqrt(n_)))}
            D['planung_mindestens_nachweisbare_steigung'] = pl
        P[str(dur)] = D
    R['paarebene_messbarkeit'] = P

out = Path(sys.argv[1]) if len(sys.argv) > 1 else W / 'aggregat_s3.json'
out.write_text(json.dumps(maske(R), indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print('geschrieben', out)
