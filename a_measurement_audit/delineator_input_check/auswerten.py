"""Aggregation and classification following the pre-registered criterion (aggregates only, no identifiers in the result)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True

OUT = Path(str(data_path("a.aw196")))
PRIV = OUT / 'privat'
RES = OUT / 'ergebnis'
HERE = Path(__file__).resolve().parent
CRIT = json.loads((HERE / 'f_a2_kriterium.json').read_text(encoding='utf8'))
S_POINT = CRIT['referenzskalen_ms']['messpunkte_sd']
MAE_A = CRIT['referenzskalen_ms']['messpunkte_mae']
S_MARK = dict(CRIT['referenzskalen_ms']['marker_S'])
S_MARK['qtc_fridericia_ms'] = S_MARK['qt_interval_ms']
POINTS = ('p_onset', 'p_offset', 'qrs_onset', 'qrs_offset', 't_offset')
MARKERS = ('qrs_duration_ms', 'qt_interval_ms', 'qtc_fridericia_ms', 'pr_interval_ms', 'p_duration_ms')
B = 2000
SEED = 20261014


def klasse(n, bias, h, disc, S, min_n):
    if n < min_n or not np.isfinite(bias) or not np.isfinite(h):
        return 'nicht beurteilbar'
    if abs(bias) >= .25 * S or h >= 1.0 * S or disc >= 5.0:
        return 'praktisch abhaengig'
    if abs(bias) < .10 * S and h < .5 * S and disc < 2.0:
        return 'praktisch unabhaengig'
    return 'begrenzt abhaengig'


def schwellenlage(blo, bhi, hlo, hhi, S):
    out = []
    for f in (.10, .25):
        if blo is not None and (blo <= f * S <= bhi or blo <= -f * S <= bhi or (np.sign(blo) != np.sign(bhi) and False)):
            out.append(f'Bias {f}S')
    for f in (.5, 1.0):
        if hlo is not None and hlo <= f * S <= hhi:
            out.append(f'H {f}S')
    return '; '.join(out)


def boot_indices(n, rng):
    return rng.integers(0, n, size=(B, n))


def paired(a, b, S, min_n, idx, n_total=None, label=None):
    """a = reference variant (Kors), b = variant; arrays with NaN for 'not available', one entry per unit (ECG/recording)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n_total = len(a) if n_total is None else n_total
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.where(both, b - a, np.nan)
    dd = d[both]
    n = int(both.sum())
    only_v = int((~fa & fb).sum()); only_k = int((fa & ~fb).sum())
    disc = 100. * (only_v + only_k) / n_total if n_total else np.nan
    row = dict(n_einheiten=n_total, n_paare=n, n_beide_fehlend=int((~fa & ~fb).sum()), n_nur_variante=only_v, n_nur_kors=only_k,
               verfuegbarkeit_kors_pct=100. * fa.sum() / n_total, verfuegbarkeit_variante_pct=100. * fb.sum() / n_total,
               diskordant_pct=disc)
    if n >= 2:
        sd = float(dd.std(ddof=1)); bias = float(dd.mean()); h = 1.96 * sd
        row.update(anteil_identisch_pct=100. * float((np.abs(dd) < 1e-9).mean()), bias_ms=bias, sd_ms=sd, loa_unten_ms=bias - h, loa_oben_ms=bias + h,
                   H_ms=h, mad_ms=float(np.abs(dd).mean()), p95_abs_ms=float(np.percentile(np.abs(dd), 95)),
                   median_d_ms=float(np.median(dd)), emp_loa_unten_ms=float(np.percentile(dd, 2.5)), emp_loa_oben_ms=float(np.percentile(dd, 97.5)))
        sub = d[idx]; ok = np.isfinite(sub); cnt = ok.sum(1)
        s1 = np.where(ok, sub, 0).sum(1); mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        s2 = (np.where(ok, sub - mean[:, None], 0) ** 2).sum(1)
        sdb = np.sqrt(s2 / np.maximum(cnt - 1, 1)); hb = 1.96 * sdb
        valid = cnt >= 2
        if valid.sum() > 100:
            blo, bhi = np.percentile(mean[valid], [2.5, 97.5]); hlo, hhi = np.percentile(hb[valid], [2.5, 97.5])
            row.update(bias_ci_unten=float(blo), bias_ci_oben=float(bhi), H_ci_unten=float(hlo), H_ci_oben=float(hhi))
            row['schwellenlage'] = schwellenlage(blo, bhi, hlo, hhi, S)
        row['S_ms'] = S
        row['klasse'] = klasse(n, bias, h, disc, S, min_n)
    else:
        row['S_ms'] = S; row['klasse'] = 'nicht beurteilbar'
    return row


def vec(records, variant, key, name, fs=500., rel=False):
    out = np.full(len(records), np.nan)
    for i, r in enumerate(records):
        v = r.get(variant)
        if not v or v.get('status') != 'ok':
            continue
        if key == 'markers':
            x = v['markers'].get(name)
            if x is not None:
                out[i] = x
        else:
            x = v['points'].get(name)
            if x is not None and v.get('rep_r') is not None:
                out[i] = (x - v['rep_r']) * 1000. / fs
    return out


def load(name, key):
    rows = []
    for l in (PRIV / name).read_text(encoding='utf8').splitlines():
        if l.strip():
            rows.append(json.loads(l))
    return rows


def ecg_level(records, variants, min_n, arm, rng, contrasts):
    """Marker and fiducial-point contrasts per unit (representative consensus beat)."""
    ok = [r for r in records if r['status'] == 'ok']
    idx = boot_indices(len(ok), rng)
    rows = []
    for ref, var in contrasts:
        for name in MARKERS:
            row = paired(vec(ok, ref, 'markers', name), vec(ok, var, 'markers', name), S_MARK[name], min_n, idx)
            row.update(arm=arm, ebene='Marker', groesse=name, kontrast=f'{var} minus {ref}', einheit='ms')
            rows.append(row)
        for name in POINTS:
            row = paired(vec(ok, ref, 'points', name, rel=True), vec(ok, var, 'points', name, rel=True), S_POINT[name], min_n, idx)
            row.update(arm=arm, ebene='Messpunkt (zum R-Gipfel des Konsensschlags)', groesse=name, kontrast=f'{var} minus {ref}', einheit='ms')
            rows.append(row)
    return rows, len(ok)


def beat_level(records, contrasts, arm, rng, min_beats=100):
    """Addition (not part of the pre-registered criterion): beat-wise fiducial points, beats paired by identical R-peak sample; cluster bootstrap over units."""
    ok = [r for r in records if r['status'] == 'ok']
    rows = []
    for ref, var in contrasts:
        for pi, name in enumerate(POINTS):
            col = {'p_onset': 1, 'p_offset': 2, 'qrs_onset': 3, 'qrs_offset': 4, 't_offset': 5}[name]
            per = []
            n_beats = 0; only_v = 0; only_k = 0; both_missing = 0
            for r in ok:
                a = {b[0]: b for b in r[ref]['beats']}; b = {b[0]: b for b in r[var]['beats']}
                common = sorted(set(a) & set(b))
                ds = []
                for k in common:
                    pa, pb = a[k][col], b[k][col]
                    n_beats += 1
                    if pa >= 0 and pb >= 0:
                        ds.append((pb - pa) * 2.0)
                    elif pa < 0 and pb >= 0:
                        only_v += 1
                    elif pa >= 0 and pb < 0:
                        only_k += 1
                    else:
                        both_missing += 1
                ds = np.array(ds, float)
                per.append((len(ds), ds.sum(), (ds ** 2).sum(), len(common), (np.abs(ds) < 1e-9).sum(), np.abs(ds).sum()))
            per = np.array(per, float)
            n, s1, s2 = per[:, 0].sum(), per[:, 1].sum(), per[:, 2].sum()
            if n < 2:
                continue
            bias = s1 / n; sd = np.sqrt(max(s2 - n * bias ** 2, 0) / (n - 1)); h = 1.96 * sd
            bi = rng.integers(0, len(per), size=(B, len(per)))
            nb = per[bi, 0].sum(1); sb1 = per[bi, 1].sum(1); sb2 = per[bi, 2].sum(1)
            mb = sb1 / np.maximum(nb, 1); sdb = np.sqrt(np.maximum(sb2 - nb * mb ** 2, 0) / np.maximum(nb - 1, 1)); hb = 1.96 * sdb
            S = S_POINT[name]
            disc = 100. * (only_v + only_k) / max(n_beats, 1)
            rows.append(dict(arm=arm, ebene='Messpunkt je Schlag (Zusatz)', groesse=name, kontrast=f'{var} minus {ref}', einheit='ms', n_schlaege=int(n_beats), n_paare=int(n),
                             n_nur_variante=int(only_v), n_nur_kors=int(only_k), diskordant_pct=disc,
                             anteil_identisch_pct=100. * per[:, 4].sum() / n, bias_ms=bias, sd_ms=sd, H_ms=h, loa_unten_ms=bias - h, loa_oben_ms=bias + h,
                             mad_ms=per[:, 5].sum() / n, bias_ci_unten=float(np.percentile(mb, 2.5)), bias_ci_oben=float(np.percentile(mb, 97.5)),
                             H_ci_unten=float(np.percentile(hb, 2.5)), H_ci_oben=float(np.percentile(hb, 97.5)), S_ms=S,
                             klasse=klasse(n, bias, h, disc, S, min_beats)))
    return rows


def one_to_one(reference, predicted, tolerance):
    from scipy.optimize import linear_sum_assignment
    n = len(reference); m = len(predicted)
    if not n or not m:
        return []
    distance = np.abs(np.asarray(reference)[:, None] - np.asarray(predicted)[None, :]); penalty = (n + m + 1) * (tolerance + 1)
    cost = np.zeros((n + m, n + m)); cost[:n, :m] = np.where(distance <= tolerance, distance, 3 * penalty); cost[:n, m:] = penalty; cost[n:, :m] = penalty
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if i < n and j < m and distance[i, j] <= tolerance]


LANDMARK = {'p_onset': ('p', 'onset', 'onset_med', 1), 'p_offset': ('p', 'end', 'end_med', 2), 'qrs_onset': ('N', 'onset', 'onset_med', 3),
            'qrs_offset': ('N', 'end', 'end_med', 4), 't_offset': ('t', 'end', 'end_med', 5)}


def ludb_errors(rec, variant, fs=500):
    """Error per reference beat and fiducial point (variant minus expert, ms), extreme and median reference; -> dict[(i,name)] = (err_ext, err_med, sample)."""
    ex = rec['expert']; refs = ex['N']
    beats = rec[variant]['beats']
    pairs = one_to_one([v['peak'] for v in refs], [b[0] for b in beats], round(.08 * fs))
    out = {}
    matched = {}
    for i, j in pairs:
        q = refs[i]['peak']
        e = {'N': refs[i]}
        pcs = [v for v in ex['p'] if 0 < q - v['peak'] < .4 * fs]; tcs = [v for v in ex['t'] if 0 < v['peak'] - q < .7 * fs]
        if pcs:
            e['p'] = max(pcs, key=lambda v: v['peak'])
        if tcs:
            e['t'] = min(tcs, key=lambda v: v['peak'])
        matched[i] = j
        for name, (wave, f_ext, f_med, col) in LANDMARK.items():
            if wave not in e:
                continue
            s = beats[j][col]
            out[(i, name)] = (None, None, None) if s < 0 else ((s - e[wave][f_ext]) * 1000. / fs, (s - e[wave][f_med]) * 1000. / fs, s)
    return out, matched, len(pairs)


def ludb_expert_intervals(ex, fs=500):
    out = {}
    Ns, ps, ts = ex['N'], ex['p'], ex['t']
    if Ns:
        out['qrs_duration_ms'] = float(np.median([(v['end'] - v['onset']) * 1000. / fs for v in Ns]))
    if ps:
        out['p_duration_ms'] = float(np.median([(v['end'] - v['onset']) * 1000. / fs for v in ps]))
        pr = []
        for q in Ns:
            c = [v for v in ps if 0 < q['onset'] - v['onset'] < .4 * fs]
            if c:
                pr.append((q['onset'] - max(c, key=lambda v: v['onset'])['onset']) * 1000. / fs)
        if pr:
            out['pr_interval_ms'] = float(np.median(pr))
    qt = []
    for q in Ns:
        c = [v for v in ts if 0 < v['peak'] - q['peak'] < .7 * fs]
        if c:
            qt.append((min(c, key=lambda v: v['peak'])['end'] - q['onset']) * 1000. / fs)
    if qt:
        out['qt_interval_ms'] = float(np.median(qt))
    return out


def err_stats(e):
    e = np.asarray([x for x in e if x is not None and np.isfinite(x)], float)
    if len(e) < 2:
        return dict(n=len(e))
    return dict(n=len(e), mean_ms=float(e.mean()), sd_ms=float(e.std(ddof=1)), median_ms=float(np.median(e)), mae_ms=float(np.abs(e).mean()), p95_abs_ms=float(np.percentile(np.abs(e), 95)))


def ludb_arm(rng):
    recs = [r for r in load('ludb_ergebnisse.jsonl', 'ludb')]
    ok = [r for r in recs if r['status'] == 'ok']
    variants = ('kors', 'inverse_dower', 'keine_vcg')
    errs = {v: [ludb_errors(r, v) for r in ok] for v in variants}
    # 1) Error against the expert per variant
    rows = []
    for v in variants:
        for name in POINTS:
            ext = [t[0] for (o, m, n) in errs[v] for (i, nm), t in o.items() if nm == name]
            med = [t[1] for (o, m, n) in errs[v] for (i, nm), t in o.items() if nm == name]
            n_ref = len(ext)
            r1 = err_stats(ext); r2 = err_stats(med)
            rows.append(dict(arm='LUDB', variante=v, groesse=name, n_referenzzeilen=n_ref, n_gepaart=r1.get('n'), **{'ext_' + k: x for k, x in r1.items() if k != 'n'},
                             **{'med_' + k: x for k, x in r2.items() if k != 'n'}))
    err_tab = pd.DataFrame(rows)
    # 2) Shifts against Kors per reference beat, paired
    shift_rows = []
    for v in ('inverse_dower', 'keine_vcg'):
        for name in POINTS:
            a_all = []; b_all = []; ea = []; eb = []; ma = []; mb_ = []
            for k in range(len(ok)):
                oa, mapa, _ = errs['kors'][k]; ob, mapb, _ = errs[v][k]
                common = sorted(set(mapa) & set(mapb))
                for i in common:
                    ta = oa.get((i, name)); tb = ob.get((i, name))
                    if ta is None and tb is None:
                        continue
                    sa = np.nan if ta is None or ta[2] is None else ta[2]; sb = np.nan if tb is None or tb[2] is None else tb[2]
                    a_all.append(sa); b_all.append(sb)
                    ea.append(np.nan if ta is None or ta[0] is None else ta[0]); eb.append(np.nan if tb is None or tb[0] is None else tb[0])
                    ma.append(np.nan if ta is None or ta[1] is None else ta[1]); mb_.append(np.nan if tb is None or tb[1] is None else tb[1])
            a = np.array(a_all) * 2.0; b = np.array(b_all) * 2.0   # samples -> ms (500 Hz); only the difference counts
            idx = rng.integers(0, len(a), size=(B, len(a)))
            row = paired(a, b, S_POINT[name], 30, idx)
            row.update(arm='LUDB', ebene='Messpunkt je Referenzschlag', groesse=name, kontrast=f'{v} minus kors', einheit='ms')
            ea, eb, ma, mb_ = map(np.array, (ea, eb, ma, mb_))
            both = np.isfinite(ea) & np.isfinite(eb)
            if both.sum() > 2:
                row.update(delta_bias_gegen_experte_ms=float(eb[both].mean() - ea[both].mean()), delta_mae_gegen_experte_ms=float(np.abs(eb[both]).mean() - np.abs(ea[both]).mean()),
                           bias_kors_gegen_experte_ms=float(ea[both].mean()), bias_variante_gegen_experte_ms=float(eb[both].mean()),
                           mae_kors_gegen_experte_ms=float(np.abs(ea[both]).mean()), mae_variante_gegen_experte_ms=float(np.abs(eb[both]).mean()),
                           delta_bias_medianreferenz_ms=float(mb_[both].mean() - ma[both].mean()),
                           delta_mae_medianreferenz_ms=float(np.abs(mb_[both]).mean() - np.abs(ma[both]).mean()))
                row['fehlerrelevant'] = bool(abs(row['delta_bias_gegen_experte_ms']) >= .25 * S_POINT[name] or row['delta_mae_gegen_experte_ms'] >= .10 * MAE_A[name])
            shift_rows.append(row)
    # 3) Marker per recording (representative consensus beat) against the expert interval and against Kors
    ref_int = [ludb_expert_intervals(r['expert']) for r in ok]
    marker_rows = []; marker_err_rows = []
    idx = rng.integers(0, len(ok), size=(B, len(ok)))
    for v in ('inverse_dower', 'keine_vcg'):
        for name in MARKERS:
            row = paired(vec(ok, 'kors', 'markers', name), vec(ok, v, 'markers', name), S_MARK[name], 30, idx)
            row.update(arm='LUDB', ebene='Marker je Aufnahme', groesse=name, kontrast=f'{v} minus kors', einheit='ms')
            marker_rows.append(row)
    for v in variants:
        for name in ('qrs_duration_ms', 'qt_interval_ms', 'pr_interval_ms', 'p_duration_ms'):
            pred = vec(ok, v, 'markers', name); ref = np.array([x.get(name, np.nan) for x in ref_int], float)
            e = (pred - ref)[np.isfinite(pred) & np.isfinite(ref)]
            marker_err_rows.append(dict(arm='LUDB', variante=v, groesse=name, n_aufnahmen=len(ok), n_referenz=int(np.isfinite(ref).sum()), n_vorhersage=int(np.isfinite(pred).sum()), **{('fehler_' + k): x for k, x in err_stats(e).items()}))
    status = {'n_aufnahmen_gesamt': len(recs), 'n_ok': len(ok), 'status_nicht_ok': [r['status'] for r in recs if r['status'] != 'ok']}
    return err_tab, pd.DataFrame(shift_rows), pd.DataFrame(marker_rows), pd.DataFrame(marker_err_rows), status, ok


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {}
    # MIMIC
    mimic = load('mimic_ergebnisse.jsonl', 'mimic')
    status_counts = pd.Series([r['status'] if r['status'] == 'ok' else 'abstention:' + r['status'].split(':')[0] + (':' + r['status'].split(':')[1].strip()[:30] if ':' in r['status'] else '') for r in mimic]).value_counts().to_dict()
    ok = [r for r in mimic if r['status'] == 'ok']
    summary['mimic'] = {'n_gerechnet': len(mimic), 'n_ok': len(ok), 'status': status_counts,
                        'folds_ok': pd.Series([r['split'] for r in ok]).value_counts().to_dict(),
                        'n_ohne_schlaege': {v: sum(1 for r in ok if r[v]['status'] != 'ok') for v in ('kors', 'inverse_dower', 'keine_vcg')}}
    rows, _ = ecg_level(mimic, None, 2, 'MIMIC', rng, [('kors', 'inverse_dower'), ('kors', 'keine_vcg')])
    pd.DataFrame(rows).to_csv(RES / 'mimic_kontraste_konsensschlag.csv', index=False)
    pd.DataFrame(beat_level(mimic, [('kors', 'inverse_dower'), ('kors', 'keine_vcg')], 'MIMIC', rng)).to_csv(RES / 'mimic_kontraste_je_schlag_zusatz.csv', index=False)
    # MIMIC gate: reproduction of the Kors variant against the earlier catalogue run
    summary['mimic_gate'] = mimic_gate(ok)
    # Overall verdict
    df = pd.DataFrame(rows)
    d_mark = df[(df.ebene == 'Marker') & (df.kontrast == 'inverse_dower minus kors') & df.groesse.isin(['qrs_duration_ms', 'qt_interval_ms', 'pr_interval_ms', 'p_duration_ms'])]
    n_dep = int((d_mark.klasse == 'praktisch abhaengig').sum()); n_ind = int((d_mark.klasse == 'praktisch unabhaengig').sum())
    urteil = 'native markers are practically dependent on the transformation' if n_dep >= 2 else ('native markers are practically independent of the transformation' if n_ind == 4 else 'limited/partly dependent')
    summary['f_a2_urteil_primaer'] = {'klassen': d_mark.set_index('groesse').klasse.to_dict(), 'n_abhaengig': n_dep, 'n_unabhaengig': n_ind, 'urteil': urteil}
    # LUDB
    if (PRIV / 'ludb_ergebnisse.jsonl').exists():
        err_tab, shift, mk, mke, st, _ = ludb_arm(rng)
        err_tab.to_csv(RES / 'ludb_fehler_gegen_experte.csv', index=False)
        shift.to_csv(RES / 'ludb_kontraste_messpunkte.csv', index=False)
        mk.to_csv(RES / 'ludb_kontraste_marker.csv', index=False)
        mke.to_csv(RES / 'ludb_marker_gegen_experte.csv', index=False)
        summary['ludb'] = st
    # PTB
    shards = sorted(PRIV.glob('ptb_ergebnisse_*.jsonl'))
    if shards:
        ptb = [r for s in shards for r in load(s.name, 'ptb')]
        okp = [r for r in ptb if r['status'] == 'ok']
        variants = [v for v in ('inverse_dower', 'keine_vcg', 'frank_gemessen', 'frank_roh') if okp and v in okp[0]]
        rows_p, _ = ecg_level(ptb, None, 2, 'PTB', rng, [('kors', v) for v in variants] + [('inverse_dower', 'frank_gemessen')])
        pd.DataFrame(rows_p).to_csv(RES / 'ptb_kontraste_konsensschlag.csv', index=False)
        pd.DataFrame(beat_level(ptb, [('kors', v) for v in variants], 'PTB', rng)).to_csv(RES / 'ptb_kontraste_je_schlag_zusatz.csv', index=False)
        summary['ptb'] = {'n_gerechnet': len(ptb), 'n_ok': len(okp), 'status': pd.Series([r['status'][:40] for r in ptb if r['status'] != 'ok']).value_counts().to_dict(), 'varianten': variants}
        ar = PRIV / 'ptb_achsenregel.json'
        if ar.exists():
            summary['ptb']['achsenregel'] = json.loads(ar.read_text())
    (RES / 'zusammenfassung.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf8')
    print(json.dumps(summary, indent=2, default=str))


def mimic_gate(ok):
    import pyarrow.parquet as pq
    ids = {r['study_id']: r for r in ok}
    cols = ['study_id'] + ['kors__' + m for m in MARKERS]
    ref = {}
    for f in sorted(glob.glob(str(data_path("a.transform_katalog_fortsetzung_v3_bloecke")) + "/block_*.parquet")):
        t = pq.read_table(f, columns=cols).to_pandas()
        t = t[t.study_id.isin(ids)]
        for _, row in t.iterrows():
            ref[int(row.study_id)] = row
    out = {'n_gefunden_in_aw067': len(ref), 'n_ok_aw196': len(ids)}
    for m in MARKERS:
        same = 0; tot = 0; miss_diff = 0; val_diff = 0
        for sid, r in ids.items():
            if sid not in ref:
                continue
            tot += 1
            a = r['kors']['markers'].get(m); b = ref[sid]['kors__' + m]
            fa = a is not None; fb = bool(np.isfinite(b))
            if fa != fb:
                miss_diff += 1
            elif fa and not np.isclose(a, b, rtol=1e-9, atol=0):
                val_diff += 1
            else:
                same += 1
        out[m] = {'n': tot, 'identisch': same, 'verfuegbarkeit_abweichend': miss_diff, 'wert_abweichend': val_diff, 'anteil_identisch_pct': 100. * same / tot if tot else None}
    return out


if __name__ == '__main__':
    main()
