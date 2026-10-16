"""AW-242 R2: Simulation bekannter wahrer Versaetze, Nullkalibrierung (200 Permutationen), Mischungsschaetzung, PPV.

Aufruf: python r2_run.py <stufe>   mit stufe in {prep, cal, main, perm, emp}
Alle Rechnungen nachtraeglich (post hoc). Entwicklungspartitionen train und validation. Keine Testpartition.
Private Zwischenstaende: <local path>*.  Aggregate (ohne Kennungen): nachrechnung/AW242/.
"""
import sys, json, pickle, time, os
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r2')
import numpy as np, pandas as pd
from aw242_kern import D, K, estimate_from_loss, loss_curve, chart_slices
from aw242_data import load_audit
from r2_sim_lib import variogram, fit_ou, run_scenario, person_sim

OUT = G.OUT
ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW242'; PUB.mkdir(parents=True, exist_ok=True)
STAGE = sys.argv[1]
TAUS = [0, 30, -30, 60, -60, 120, -120, 240, -240]
SEED = 242


def seeds(*key):
    return np.random.SeedSequence([SEED, *key])


def prep():
    """Personenliste (>=3 vollstaendige EKG) mit V-Matrix, Zeiten, Chartreihen je Aufenthalt; Variogramm; OU-Fit."""
    base = pickle.load(open(OUT / 'base_persons.pkl', 'rb'))
    df = base['persons']; kept = base['kept']
    data = load_audit()
    ecg = data['ecg']; chart = data['chart']
    sl = chart_slices(chart['subject_id'].to_numpy(), chart['stay'].to_numpy())
    ct = chart['t'].to_numpy(float); chf = chart['hf'].to_numpy(float)
    sub = ecg['subject_id'].to_numpy()
    first = {int(p): int(i) for p, i in zip(*np.unique(sub, return_index=True))}
    t_all = ecg['t'].to_numpy(float); stay_all = ecg['stay'].to_numpy(); st_all = ecg['start'].to_numpy(float); en_all = ecg['end'].to_numpy(float)
    h_all = ecg['hf'].to_numpy(float)
    persons = []; stays_for_fit = []
    sel = df[df.n_complete >= 3]
    for r in sel.itertuples():
        p = int(r.subject_id); k = kept[p]; idx = k['idx']
        stays = {}
        for s in np.unique(stay_all[idx]):
            a, b = sl[(p, int(s))]
            st = st_all[idx][stay_all[idx] == s][0]; en = en_all[idx][stay_all[idx] == s][0]
            tt = ct[a:b]; vv = chf[a:b]
            m = (tt >= st) & (tt <= en)
            tt, vv = tt[m], vv[m]
            stays[int(s)] = (tt, vv, float(vv.mean()))
            stays_for_fit.append((tt, vv))
        n_el = int((sub == p).sum())
        persons.append(dict(subject_id=p, fold=r.fold, t=t_all[idx], stay_ids=stay_all[idx].astype(int), stays=stays, V=k['V'], h_real=k['h'],
                            minimum=float(r.minimum), ok=bool(r.ok), offset=int(r.offset), width=int(r.width), all_h=h_all[first[p]:first[p] + n_el],
                            local_idx=idx - first[p], n_eligible=n_el))
    lag, g, c = variogram(stays_for_fit)
    fit = fit_ou(lag, g, c)
    pickle.dump(persons, open(OUT / 'r2_persons.pkl', 'wb'), protocol=4)
    res = dict(n_persons=len(persons), n_stays=len(stays_for_fit), variogram=dict(lag_mid_min=(lag / 60).tolist(), gamma=g.tolist(), count=c.tolist()), ou_fit=fit)
    (OUT / 'r2_prep.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'n': len(persons), 'fit': fit}))


def load_persons():
    return pickle.load(open(OUT / 'r2_persons.pkl', 'rb')), json.loads((OUT / 'r2_prep.json').read_text())['ou_fit']


def stats_run(res):
    """Kennzahlen eines Durchlaufs je Fold."""
    out = {}
    for f in ('train', 'validation'):
        r = [x for x in res if x['fold'] == f]
        ok = [x for x in r if x['ok']]
        out[f] = dict(n=len(r), ident=len(ok), off=[x['offset'] for x in ok], width=[x['width'] for x in ok], minimum=[x['minimum'] for x in r])
    return out


def aggregate(runs, tau, d_min):
    """Aggregation ueber Wiederholungen -> Zeilen je Fold und gesamt."""
    rows = []
    for f in ('train', 'validation', 'gesamt'):
        n = []; ident = []; off_all = []; err15 = []; err60 = []; flag = []; med_min = []; pid = []; share60_zero = []
        for run in runs:
            fs = ('train', 'validation') if f == 'gesamt' else (f,)
            nn = sum(run[x]['n'] for x in fs); ii = sum(run[x]['ident'] for x in fs)
            off = np.array(sum((run[x]['off'] for x in fs), []), float)
            n.append(nn); ident.append(ii)
            off_all.append(off)
            exp_ = tau + d_min
            err15.append(np.sum(np.abs(off - exp_) <= 15)); err60.append(np.sum(np.abs(off - exp_) <= 60))
            flag.append(np.sum(np.abs(off) > 60)); share60_zero.append(np.sum(np.abs(off) <= 60))
            med_min.append(np.median(sum((run[x]['minimum'] for x in fs), [])))
        n = np.array(n, float); ident = np.array(ident, float)
        offs = np.concatenate(off_all) if off_all else np.array([])
        e = offs - (tau + d_min)
        rows.append(dict(fold=f, reps=len(runs), persons=float(n.mean()), ident_mean=float(ident.mean()), p_ident=float((ident / n).mean()),
                         p_ident_sd=float((ident / n).std(ddof=1)) if len(runs) > 1 else None,
                         bias_vs_expected=float(e.mean()) if len(e) else None, median_dev=float(np.median(e)) if len(e) else None,
                         mad=float(np.median(np.abs(e))) if len(e) else None,
                         share15_given_ident=float(np.sum(err15) / np.sum(ident)) if ident.sum() else None,
                         share60_given_ident=float(np.sum(err60) / np.sum(ident)) if ident.sum() else None,
                         p_ident_and_within60=float((np.array(err60) / n).mean()),
                         share_abs_est_le60_given_ident=float(np.sum(share60_zero) / np.sum(ident)) if ident.sum() else None,
                         p_flag_large=float((np.array(flag) / n).mean()),
                         median_of_person_minimum=float(np.mean(med_min))))
    return rows


def calibrate(model, persons, ou, scale_fn=None, tag=''):
    real = np.array([p['minimum'] for p in persons])
    target = float(np.median(real))
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    pts = []
    for sg in grid:
        res = run_scenario(persons, model, 0, ('const', 0.0), ou, sg, seeds(1, {'A': 1, 'B': 2}[model] + (10 if scale_fn else 0), int(sg * 10)), scale_fn)
        m = float(np.median([x['minimum'] for x in res])); fi = float(np.mean([x['ok'] for x in res]))
        pts.append((sg, m, fi))
    sg_arr = np.array([a for a, b, c in pts]); m_arr = np.array([b for a, b, c in pts])
    if target <= m_arr[0]:
        sg = 0.0
    elif target >= m_arr[-1]:
        sg = float(sg_arr[-1])
    else:
        sg = float(np.interp(target, m_arr, sg_arr))
    return dict(model=model, tag=tag, target_median_minimum=target, real_frac_min_le5=float(np.mean(real <= 5)), sigma_eps=sg, grid=pts)


def main_sim():
    persons, ou = load_persons()
    cal = json.loads((OUT / 'r2_cal.json').read_text())
    results = []
    sc_id = 0
    def block(model, caltag, delays, taus, reps, scale_fn=None, mult=1.0, label=''):
        nonlocal sc_id
        sg = cal[caltag]['sigma_eps'] * mult
        for dname, dspec, d_min in delays:
            for tau in taus:
                runs = []
                for rep in range(reps):
                    sc_id += 1
                    res = run_scenario(persons, model, tau, dspec, ou, sg, seeds(2, sc_id, rep), scale_fn)
                    runs.append(stats_run(res))
                rows = aggregate(runs, tau, d_min)
                for r in rows:
                    r.update(model=model, label=label, sigma_eps=sg, tau_min=tau, delay=dname, delay_min=d_min)
                    results.append(r)
                G.status('r2', f'{label} d={dname} tau={tau}')
    DEL = [('0', ('const', 0.0), 0.0), ('5', ('const', 300.0), 5.0), ('15', ('const', 900.0), 15.0)]
    Z = [('0', ('const', 0.0), 0.0)]
    block('B', 'B', DEL, TAUS, 20, label='B_haupt')
    samp = np.load(OUT / 'delay_sample_s.npy')
    block('B', 'B', [('empirisch', ('emp', samp), float(np.median(samp) / 60.0))], TAUS, 10, label='B_verzug_empirisch')
    block('A', 'A', Z, TAUS, 20, label='A_einfach')
    med = float(np.median([p['minimum'] for p in persons]))
    sfn = lambda p: max(p['minimum'], 0.5) / med
    block('B', 'B_person', Z, [0, 60, -60, 240, -240], 10, scale_fn=sfn, label='B_personenbezogen')
    block('B', 'B', Z, [0, 60, -60, 240, -240], 10, mult=0.5, label='B_rauschen_0.5x')
    block('B', 'B', Z, [0, 60, -60, 240, -240], 10, mult=2.0, label='B_rauschen_2x')
    pd.DataFrame(results).to_csv(OUT / 'r2_main.csv', index=False)
    print('main fertig', len(results))


def perm_null():
    persons, ou = load_persons()
    R = 200
    rows = []
    for rep in range(R):
        rng = np.random.default_rng(seeds(3, rep))
        per = {'train': dict(n=0, ident=0, within60=0, within120=0, off=[]), 'validation': dict(n=0, ident=0, within60=0, within120=0, off=[])}
        for p in persons:
            # Permutation der HF-Werte ueber alle verwertbaren EKG der Person (wie AW-181), dann nur vollstaendige EKG
            vals = rng.permutation(p['all_h'])
            h = vals[p['local_idx']]
            e = estimate_from_loss(loss_curve(p['V'], h), len(h))
            q = per[p['fold']]
            q['n'] += 1
            if e['ok']:
                q['ident'] += 1; q['within60'] += abs(e['offset']) <= 60; q['within120'] += abs(e['offset']) <= 120; q['off'].append(e['offset'])
        for f, q in per.items():
            rows.append(dict(rep=rep, fold=f, n=q['n'], ident=q['ident'], within60=int(q['within60']), within120=int(q['within120']),
                             abs_gt60=int(sum(abs(o) > 60 for o in q['off']))))
        if rep % 20 == 0:
            G.status('r2', f'perm {rep}/{R}')
    pd.DataFrame(rows).to_csv(OUT / 'r2_perm.csv', index=False)
    print('perm fertig')


STAGES = dict(prep=prep, main=main_sim, perm=perm_null)

if __name__ == '__main__':
    t0 = time.time()
    if STAGE == 'cal':
        persons, ou = load_persons()
        med = float(np.median([p['minimum'] for p in persons]))
        sfn = lambda p: max(p['minimum'], 0.5) / med
        cal = {'B': calibrate('B', persons, ou, None, 'B'), 'A': calibrate('A', persons, ou, None, 'A'),
               'B_person': calibrate('B', persons, ou, sfn, 'B_person')}
        (OUT / 'r2_cal.json').write_text(json.dumps(cal, indent=1))
        print(json.dumps({k: {'sigma_eps': v['sigma_eps'], 'target': v['target_median_minimum']} for k, v in cal.items()}))
    else:
        STAGES[STAGE]()
    print('Dauer', round(time.time() - t0, 1), 's, Spitzen-RSS', G.peak())
