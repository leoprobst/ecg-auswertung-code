"""AW-242: Aggregation der privaten Rohergebnisse zu veroeffentlichbaren Aggregat-CSV (ohne Kennungen, Kleinzellenregel).

Liest <local path>*_raw.* ; schreibt nachrechnung/AW242/S5_*.csv und <local path>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

E = Path(str(data_path("s5.aw242")))
ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW242'; PUB.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(242999)
N = {}


def kzv(v):
    try:
        x = int(v)
    except (TypeError, ValueError):
        return v
    return '<10' if 0 < x < 10 else x


def write(df, name):
    df.to_csv(PUB / name, index=False, encoding='utf-8')


# ---------------- R2 ----------------
main = pd.read_csv(E / 'r2_main.csv')
perm = pd.read_csv(E / 'r2_perm.csv')
cols = ['label', 'delay', 'delay_min', 'tau_min', 'fold', 'reps', 'sigma_eps', 'p_ident', 'p_ident_sd', 'bias_vs_expected', 'median_dev', 'mad', 'share15_given_ident',
        'share60_given_ident', 'p_ident_and_within60', 'share_abs_est_le60_given_ident', 'p_flag_large', 'median_of_person_minimum']
sim = main[cols].copy().round(4)
write(sim, 'S5_R2_simulation.csv')
N3 = {'train': 1482, 'validation': 497}
OBS = {'train': dict(n=1482, ident=392, w60=263), 'validation': dict(n=497, ident=145, w60=102)}
OBS['gesamt'] = {k: OBS['train'][k] + OBS['validation'][k] for k in ('n', 'ident', 'w60')}
nullrows = []
pg = perm.copy()
tot = pg.groupby('rep')[['n', 'ident', 'within60', 'abs_gt60']].sum().reset_index(); tot['fold'] = 'gesamt'
pg = pd.concat([pg, tot], ignore_index=True)
null_stats = {}
for f in ('train', 'validation', 'gesamt'):
    s = pg[pg.fold == f]
    n = s.n.iloc[0]
    null_stats[f] = dict(n=int(n), ident_mean=s.ident.mean(), ident_lo=s.ident.quantile(.025), ident_hi=s.ident.quantile(.975), ident_min=s.ident.min(), ident_max=s.ident.max(),
                         w60_mean=s.within60.mean(), w60_lo=s.within60.quantile(.025), w60_hi=s.within60.quantile(.975), gt60_mean=s.abs_gt60.mean(), gt60_lo=s.abs_gt60.quantile(.025), gt60_hi=s.abs_gt60.quantile(.975))
    o = OBS[f]
    r_obs = o['ident'] / o['n']; r0 = null_stats[f]['ident_mean'] / n
    a = (r_obs - r0) / (1 - r0)
    # Intervall: Beobachtung binomial (Personen), Null aus den 200 Permutationen
    draws = []
    for _ in range(10000):
        ro = rng.binomial(o['n'], r_obs) / o['n']
        r0d = s.ident.iloc[rng.integers(0, len(s))] / n
        draws.append((ro - r0d) / (1 - r0d))
    exc_w60 = o['w60'] - null_stats[f]['w60_mean']
    exc_draws = [rng.binomial(o['n'], o['w60'] / o['n']) - s.within60.iloc[rng.integers(0, len(s))] for _ in range(10000)]
    nullrows.append(dict(fold=f, personen_mind3=int(n), beobachtet_identifizierbar=o['ident'], null_identifizierbar_mittel=round(null_stats[f]['ident_mean'], 1),
                         null_identifizierbar_2_5=round(null_stats[f]['ident_lo'], 1), null_identifizierbar_97_5=round(null_stats[f]['ident_hi'], 1),
                         null_rate=round(r0, 4), beob_rate=round(r_obs, 4), mischungsanteil_a=round(a, 4), a_ci_lo=round(float(np.quantile(draws, .025)), 4), a_ci_hi=round(float(np.quantile(draws, .975)), 4),
                         a_personen=round(a * n, 1), beobachtet_innerhalb60=o['w60'], null_innerhalb60_mittel=round(null_stats[f]['w60_mean'], 1), null_innerhalb60_2_5=round(null_stats[f]['w60_lo'], 1),
                         null_innerhalb60_97_5=round(null_stats[f]['w60_hi'], 1), ueberschuss_innerhalb60=round(exc_w60, 1), ueberschuss_ci_lo=round(float(np.quantile(exc_draws, .025)), 1),
                         ueberschuss_ci_hi=round(float(np.quantile(exc_draws, .975)), 1), ueberschuss_anteil_an_beobachtet_innerhalb60=round(exc_w60 / o['w60'], 3),
                         beobachtet_ausserhalb60=o['ident'] - o['w60'], null_ausserhalb60_mittel=round(null_stats[f]['gt60_mean'], 1)))
write(pd.DataFrame(nullrows), 'S5_R2_null_und_mischung.csv')
N['null'] = nullrows

# Drei-Klassen-Mischung (innerhalb 60 / ausserhalb 60 / nicht identifizierbar): Beobachtung = w * idealisierte Simulation(tau=0) + (1-w) * Permutationsnull
b0 = main[(main.label == 'B_haupt') & (main.delay == '0') & (main.tau_min == 0)].set_index('fold')
a0 = main[(main.label == 'A_einfach') & (main.tau_min == 0)].set_index('fold')
mix = []
for f in ('train', 'validation', 'gesamt'):
    n = null_stats[f]['n']; o = OBS[f]
    c_obs = np.array([o['w60'], o['ident'] - o['w60'], n - o['ident']], float) / n
    N0 = np.array([null_stats[f]['w60_mean'], null_stats[f]['gt60_mean'], n - null_stats[f]['ident_mean']]) / n
    for nm, S in (('Modell B, Verzug 0', b0), ('Modell A', a0)):
        p_id = S.loc[f, 'p_ident']; p_fl = S.loc[f, 'p_flag_large']
        H0 = np.array([p_id - p_fl, p_fl, 1 - p_id])
        def solve(c):
            d = H0 - N0
            return float(np.clip(np.dot(c - N0, d) / np.dot(d, d), 0, 1))
        w = solve(c_obs)
        ws = []
        for _ in range(5000):
            cnt = rng.multinomial(n, c_obs / c_obs.sum()) / n
            ws.append(solve(cnt))
        mix.append(dict(fold=f, simulation=nm, anteil_wie_idealisiert=round(w, 3), ci_lo=round(float(np.quantile(ws, .025)), 3), ci_hi=round(float(np.quantile(ws, .975)), 3),
                        beobachtet_innerhalb60=round(c_obs[0], 4), beobachtet_ausserhalb60=round(c_obs[1], 4), beobachtet_nicht_ident=round(c_obs[2], 4),
                        sim_innerhalb60=round(H0[0], 4), sim_ausserhalb60=round(H0[1], 4), sim_nicht_ident=round(H0[2], 4),
                        null_innerhalb60=round(N0[0], 4), null_ausserhalb60=round(N0[1], 4), null_nicht_ident=round(N0[2], 4)))
write(pd.DataFrame(mix), 'S5_R2_dreiklassenmischung.csv')
N['mix'] = mix

# PPV der Kennzeichnung "identifizierbar und |Schaetzer| > 60 min"
g = main[(main.label == 'B_haupt') & (main.delay == '0') & (main.fold == 'gesamt')].set_index('tau_min')
sens = float(np.mean([g.loc[t, 'p_flag_large'] for t in (120, -120, 240, -240)]))
fpr_sim = float(np.mean([g.loc[t, 'p_flag_large'] for t in (0, 30, -30)]))
fpr_null = null_stats['gesamt']['gt60_mean'] / null_stats['gesamt']['n']
ppv = []
for prev in (0.01, 0.05, 0.20):
    for nm, fpr in (('Simulation tau 0 bis +-30 min', fpr_sim), ('Permutationsnull', fpr_null)):
        ppv.append(dict(praevalenz_grosser_versatz=prev, falsch_positiv_quelle=nm, sensitivitaet=round(sens, 4), falsch_positiv_rate=round(fpr, 4), ppv=round(sens * prev / (sens * prev + fpr * (1 - prev)), 4)))
write(pd.DataFrame(ppv), 'S5_R2_ppv.csv')
N['ppv'] = ppv; N['sens'] = sens; N['fpr_sim'] = fpr_sim; N['fpr_null'] = fpr_null

# ---------------- R3 ----------------
w = pd.read_csv(E / 'r3_waterfall_raw.csv')
c3 = pd.read_csv(E / 'r3_cond_raw.csv')
cmp_ = pd.read_csv(E / 'r3_compare_raw.csv')
wf = w[w.stufe != 'Zerlegung'][['fold', 'stufe', 'beschreibung', 'personen', 'anteil_an_L1', 'verlust_zur_Vorstufe', 'verlust_anteil_an_L1']].copy()
write(wf.round(4), 'S5_R3_wasserfall.csv')
z = w[w.stufe == 'Zerlegung'][['fold', 'beschreibung', 'personen']].copy()
rows = []
for f in ('train', 'validation'):
    s = z[z.fold == f].set_index('beschreibung').personen
    def v(k): return int(s[k])
    one = v('L3_L4_kein_Chartwert_im_Aufenthalt_fuer_Randfreie_EKG')
    edge4 = v('L5_L6_Reihenfolge4_Zone_am_Suchrand')
    rows += [
        (f, 'L1 zu L2', 'kein RR-geeignetes ICU-EKG', v('L1_L2_kein_RR_EKG')),
        (f, 'L2 zu L3', 'Verlust insgesamt (Aufenthaltsdauer/Randlage)', v('L2_L3_gesamt')),
        (f, 'L2 zu L3', 'davon alle Aufenthalte unter 24 h (Suchfenster passt nie hinein)', v('L2_L3_alle_Aufenthalte_unter_24h')),
        (f, 'L2 zu L3', 'davon laengerer Aufenthalt, aber alle EKG naeher als 12 h an einer Grenze', v('L2_L3_Aufenthalt_ab_24h_aber_EKG_am_Rand')),
        (f, 'L3 zu L4', 'Verlust insgesamt (Referenzluecken)', v('L3_L4_gesamt')),
        (f, 'L3 zu L4', 'davon ohne jeden Chartwert im Aufenthalt', kzv(one)),
        (f, 'L3 zu L4', 'davon mit Chartwerten, aber Luecke > 30 min an mindestens einem Offset', f">{v('L3_L4_gesamt') - 10}" if 0 < one < 10 else v('L3_L4_Chartwerte_vorhanden_aber_Luecke_ueber_30min')),
        (f, 'L3 zu L4', 'davon beste Abdeckung >= 90 % der Offsets', v('L3_L4_beste_Abdeckung_ab_90_Prozent_der_Offsets')),
        (f, 'L3 zu L4', 'davon beste Abdeckung < 50 % der Offsets', kzv(v('L3_L4_beste_Abdeckung_unter_50_Prozent'))),
        (f, 'L4 zu L5', 'Verlust insgesamt (weniger als 3 vollstaendige EKG)', v('L4_L5_gesamt')),
        (f, 'L4 zu L5', 'davon genau 1 vollstaendiges EKG', v('L4_L5_genau_1_vollst_EKG')),
        (f, 'L4 zu L5', 'davon genau 2 vollstaendige EKG', v('L4_L5_genau_2_vollst_EKG')),
        (f, 'L4 zu L5', 'davon mit mindestens 3 randfreien EKG (Referenzluecken verhindern Vollabdeckung)', v('L4_L5_davon_mit_mind3_randfreien_EKG')),
        (f, 'L5 zu L6', 'Verlust insgesamt (Kurvenkriterien)', v('L5_L6_gesamt')),
        (f, 'L5 zu L6 (feste Reihenfolge)', '1 Minimum > 5/min', v('L5_L6_Reihenfolge1_Minimum_ueber_5')),
        (f, 'L5 zu L6 (feste Reihenfolge)', '2 Zonenbreite > 120 min', v('L5_L6_Reihenfolge2_Breite_ueber_120')),
        (f, 'L5 zu L6 (feste Reihenfolge)', '3 und 4 Zone nicht zusammenhaengend oder am Suchrand', v('L5_L6_Reihenfolge3_Zone_nicht_zusammenhaengend') + edge4),
        (f, 'L5 einzeln', 'Minimum > 5/min', v('L5_einzeln_Minimum_ueber_5')),
        (f, 'L5 einzeln', 'Zonenbreite > 120 min', v('L5_einzeln_Breite_ueber_120')),
        (f, 'L5 einzeln', 'Zone nicht zusammenhaengend', v('L5_einzeln_Zone_nicht_zusammenhaengend')),
        (f, 'L5 einzeln', 'Zone am Suchrand', v('L5_einzeln_Zone_am_Suchrand')),
    ]
zer = pd.DataFrame(rows, columns=['fold', 'schritt', 'grund', 'personen'])
den = {'train': 17067, 'validation': 5846}
zer['anteil_an_L1'] = [round(int(p) / den[f], 4) if not str(p).startswith(('<', '>')) else None for f, p in zip(zer.fold, zer.personen)]
write(zer, 'S5_R3_verlustgruende.csv')
N['zer'] = zer.to_dict('records')
c3['anteil'] = c3.anteil.round(4); c3['wilson_untere'] = c3.wilson_untere.round(4); c3['wilson_obere'] = c3.wilson_obere.round(4)
write(c3, 'S5_R3_bedingte_ausbeute.csv')
cmp_['smd'] = cmp_['smd'].round(3)
write(cmp_[['fold', 'vergleich', 'variable', 'stufe', 'gruppe_A', 'gruppe_B', 'wert_A', 'wert_B', 'smd', 'art']], 'S5_R3_vergleich_gruppen.csv')

# ---------------- R4 ----------------
cause = pd.read_csv(E / 'r4_cause_raw.csv'); shift = pd.read_csv(E / 'r4_shift_raw.csv'); cens = pd.read_csv(E / 'r4_census_raw.csv')
src = pd.read_csv(E / 'r4_source_raw.csv'); gap = pd.read_csv(E / 'r4_gap_raw.csv'); dist = pd.read_csv(E / 'r4_dist_raw.csv'); flow = pd.read_csv(E / 'r4_flow_raw.csv')
# Kleinzellenregel: Validierungsfold hat <10 inkompatible Landmarken; Gesamtzahlen und Komplemente vergroebern
reference_flow_totals = reference("agg_publish.validation_flow_totals")
if reference_flow_totals is None: raise SystemExit("agg_publish: local reference value 'agg_publish.validation_flow_totals' is missing; the validation totals cannot be coarsened (small-cell rule), so the aggregation cannot run.")
cause = cause.astype({'n': object}); flow = flow.astype({'n': object})
for i_ in cause.index:
    if cause.fold[i_] == 'validation' and cause.ursache[i_].startswith('1 ausgewaehlte'):
        cause.at[i_, 'n'] = '>9745'
    if cause.fold[i_] == 'validation' and cause.ursache[i_].startswith('2 inkompatible'):
        cause.at[i_, 'n'] = '<10'
    if cause.fold[i_] == 'validation' and cause.ursache[i_].startswith('3 kompatibel'):
        cause.at[i_, 'n'] = 9745
for i_ in flow.index:
    if flow.fold[i_] == 'validation' and int(flow.n[i_]) in reference_flow_totals:
        flow.at[i_, 'n'] = '>9745'
dist['n_risikopersonen_archiviert'] = dist['n_risikopersonen_archiviert'].map(kzv)
write(cause, 'S5_R4_nullzeit_ursachen.csv'); write(shift, 'S5_R4_null_positiv_verschiebung.csv'); write(src, 'S5_R4_kontaktquellen.csv'); write(dist, 'S5_R4_kontaktzeit_verteilung.csv'); write(flow, 'S5_R4_tore.csv')
c2 = cens.copy()
c2['tied'] = c2.tied.map(kzv)
c2['horizon_end'] = [f'>{int(h) - 10}' if t == '<10' else h for h, t in zip(cens.horizon_end, c2.tied)]
write(c2, 'S5_R4_zensus.csv')
g2 = gap.copy().astype({'gleichstand': object, 'horizontende': object})
for i_ in g2.index:
    hz_, ti_ = int(gap.horizontende[i_]), int(gap.gleichstand[i_])
    if 0 < ti_ < 10 and 0 < hz_ < 10:
        g2.at[i_, 'horizontende'] = '<20 (Summe mit Gleichstand)'; g2.at[i_, 'gleichstand'] = '<20 (Summe mit Horizontende)'
    elif 0 < ti_ < 10:
        g2.at[i_, 'gleichstand'] = '<10'; g2.at[i_, 'horizontende'] = f'>{hz_ - 10}'
    elif 0 < hz_ < 10:
        g2.at[i_, 'horizontende'] = '<10'; g2.at[i_, 'gleichstand'] = f'>{ti_ - 10}' if ti_ >= 10 else ti_
write(g2.round(4), 'S5_R4_kontaktluecken.csv')
N['cens'] = cens.to_dict('records'); N['gap'] = gap.to_dict('records')

# ---------------- R5 ----------------
r5 = pd.read_csv(E / 'r5_raw.csv')
for c_ in ('mind1_vollst', 'mind3_vollst', 'identifizierbar', 'innerhalb15', 'innerhalb60', 'innerhalb120', 'innerhalb360'):
    r5[c_] = r5[c_].map(kzv)
for c_ in ('ident_anteil_primaer', 'ident_anteil_mind3', 'anteil60', 'ci60_unten', 'ci60_oben', 'null_ident_mean', 'null_within60_mean'):
    r5[c_] = r5[c_].round(3)
write(r5, 'S5_R5_varianten_vollabdeckung.csv')
r5b = pd.read_csv(E / 'r5b_raw.csv').round(3)
r5b['personen_mit_wert_bei_offset0'] = r5b.personen_mit_wert_bei_offset0.map(kzv)
write(r5b, 'S5_R5_paarbeschraenkung.csv')
e1 = json.loads((E / 'e1_delay_stats.json').read_text()); meta = json.loads((E / 'r5_meta.json').read_text())
reference_rows_d5 = reference("agg_publish.rows_delay_0_5"); reference_rows_d15 = reference("agg_publish.rows_delay_0_15")
if reference_rows_d5 is None or reference_rows_d15 is None: raise SystemExit("agg_publish: local reference values 'agg_publish.rows_delay_0_5' / 'agg_publish.rows_delay_0_15' are missing; the origin table cannot be written.")
orig = [dict(merkmal='Item', wert='220045 Heart Rate (chartevents, ICU, MetaVision)'), dict(merkmal='Zeilen Primaerpersonen 30-220/min', wert=e1['delay']['n']),
        dict(merkmal='Minute der Charttime = 0', wert=round(e1['minute']['minute0'], 3)), dict(merkmal='Sekunde = 0', wert=e1['minute']['second0']),
        dict(merkmal='Speicherverzug storetime-charttime, Median (min)', wert=e1['delay']['quantiles_min']['0.5']), dict(merkmal='Verzug Q25 / Q75 (min)', wert=f"{e1['delay']['quantiles_min']['0.25']} / {e1['delay']['quantiles_min']['0.75']}"),
        dict(merkmal='Verzug Q90 / Q99 (min)', wert=f"{e1['delay']['quantiles_min']['0.9']} / {e1['delay']['quantiles_min']['0.99']}"), dict(merkmal='Anteil Verzug 0 bis 5 min (Bedingung 0<=d<=5)', wert=round(reference_rows_d15 and reference_rows_d5 / e1['delay']['n'], 3)),
        dict(merkmal='Anteil Verzug 0 bis 15 min', wert=round(reference_rows_d15 / e1['delay']['n'], 3)), dict(merkmal='Anteil Verzug <= 60 min (inkl. negativer)', wert=round(e1['delay']['le60'], 3)),
        dict(merkmal='negativer Verzug (storetime vor charttime), Zeilen', wert=e1['delay']['negative']), dict(merkmal='Anteil warning = 1', wert=round(e1['warning_share'], 4)),
        dict(merkmal='RR-geeignete ICU-EKG', wert=meta['report_stats']['eligible_ecgs']), dict(merkmal='davon berichteter Sinusrhythmus ohne Stimulationstext', wert=meta['report_stats']['sinus_no_pacing']),
        dict(merkmal='davon mit Stimulationstext im Maschinenbefund', wert=meta['report_stats']['any_pacing_text'])]
write(pd.DataFrame(orig), 'S5_R5_herkunft_referenz.csv')
N['e1'] = e1

(E / 'agg_numbers.json').write_text(json.dumps(N, indent=1, default=str))
print('Aggregation fertig')
print(pd.DataFrame(nullrows).T.to_string())
print(pd.DataFrame(mix).to_string())
print(pd.DataFrame(ppv).to_string())
