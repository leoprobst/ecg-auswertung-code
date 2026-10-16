"""AW-242 S5-R5: Referenzfrequenz-Sensitivitaet mit vergleichbaren Messbedingungen (post hoc).

Varianten (Verfahren sonst unveraendert): a) nur EKG mit maschinell berichtetem Sinusrhythmus ohne Stimulationshinweis;
b) nur Chartwerte mit 0 <= storetime-charttime <= 5 bzw. 15 min; c) nur Chartwerte mit Minute ungleich 0;
d) Suchfenster 10 bzw. 15 min; e) Kombinationen. Je Variante 20 Permutationen als Nullrate (Seed festgehalten).
Eingabe: AW-110-Arbeitsdatenbank (lesend), <local path> (storetime), Masterindex (machine_report).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, re, time, ast
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r5')
import numpy as np, pandas as pd, duckdb
from aw242_kern import D, K, estimate_from_loss, loss_curve, wilson
from aw242_data import load_audit, compute_persons, summarize

OUT = G.OUT
ROOT = Path(__file__).resolve().parents[2]
MASTER = str(data_path("s5.mimic_validation_master_ecg_master_index_parquet"))
SRC = Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_documented_af_landmark_20261012_v2_executed_source_a155_research_documented_af_py")))
PACE = re.compile(r'pace|pacing|pacemaker|paced', re.I)


def load_report_sinus():
    code = SRC.read_text(encoding='utf-8-sig')
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'report_sinus']
    ns = {'np': np, 'pd': pd, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<AF>', 'exec'), ns)
    return ns['report_sinus']


def perm_null(kept, df, ecg, reps, seed):
    """Nullrate: HF-Werte innerhalb der Person ueber alle verwertbaren (hier: verbleibenden) EKG permutieren."""
    sub = ecg['subject_id'].to_numpy(); hf = ecg['hf'].to_numpy(float)
    first = {int(p): int(i) for p, i in zip(*np.unique(sub, return_index=True))}
    cnt = pd.Series(sub).value_counts().to_dict()
    sel = [(p, k) for p, k in kept.items() if len(k['idx']) >= 3]
    fold_of = df.set_index('subject_id')['fold'].to_dict()
    rng = np.random.default_rng(seed)
    res = {'train': [], 'validation': []}
    for rep in range(reps):
        acc = {'train': [0, 0, 0], 'validation': [0, 0, 0]}
        for p, k in sel:
            a = first[p]; n = cnt[p]
            vals = rng.permutation(hf[a:a + n])
            h = vals[k['idx'] - a]
            e = estimate_from_loss(loss_curve(k['V'], h), len(h))
            q = acc[fold_of[p]]
            q[0] += 1
            if e['ok']:
                q[1] += 1; q[2] += abs(e['offset']) <= 60
        for f in acc:
            res[f].append(acc[f])
    return {f: dict(n3=int(v[0][0]), ident_mean=float(np.mean([x[1] for x in v])), within60_mean=float(np.mean([x[2] for x in v])),
                    ident_min=int(min(x[1] for x in v)), ident_max=int(max(x[1] for x in v))) for f, v in res.items()}


def row_for(name, df, den, nulls, rng):
    out = []
    summ = summarize(df, den)
    for f in ('train', 'validation'):
        s = summ[f]; sub = df[(df.fold == f)]
        good = sub[sub.ok.fillna(False).astype(bool)]
        offs = good.offset.to_numpy(float)
        n_id = len(offs)
        bs = []
        if n_id:
            for _ in range(1000):
                b = offs[rng.integers(0, n_id, n_id)]
                bs.append(np.mean(np.abs(b) <= 60))
        r = dict(variante=name, fold=f, primaer=int(den[f]), mind1_vollst=s['complete_people'], mind3_vollst=s['at_least_3_people'], identifizierbar=s['identifiable_people'],
                 ident_anteil_primaer=s['identifiable_people'] / den[f],
                 ident_anteil_mind3=(s['identifiable_people'] / s['at_least_3_people']) if s['at_least_3_people'] else None,
                 innerhalb15=s['within'][15] if n_id else 0, innerhalb60=s['within'][60] if n_id else 0, innerhalb120=s['within'][120] if n_id else 0, innerhalb360=s['within'][360] if n_id else 0,
                 anteil60=(s['within'][60] / n_id) if n_id else None, ci60_unten=float(np.quantile(bs, .025)) if bs else None, ci60_oben=float(np.quantile(bs, .975)) if bs else None,
                 popmin_min=s.get('population_optimum_minutes'), breite_median=float(np.median(good.width)) if n_id else None,
                 null_ident_mean=nulls[f]['ident_mean'] if nulls else None, null_ident_min=nulls[f]['ident_min'] if nulls else None, null_ident_max=nulls[f]['ident_max'] if nulls else None,
                 null_within60_mean=nulls[f]['within60_mean'] if nulls else None)
        out.append(r)
    return out


def main():
    t0 = time.time()
    data = load_audit()
    ecg, chart, people = data['ecg'], data['chart'], data['people']
    den = people.groupby('fold').size().to_dict()
    # Chart-Zusatzmerkmale in derselben Reihenfolge
    c = duckdb.connect(); c.execute("SET memory_limit='900MB'; SET threads=2")
    c.execute(("ATTACH '" + str(data_path("s5.uhrzeit_aw110_audit_duckdb")) + "' AS a (READ_ONLY)"))
    aux = c.execute(f"""SELECT epoch(h.t) AS t, epoch(r.storetime)-epoch(r.charttime) AS delay_s, minute(h.t) AS mi
        FROM a.hf h LEFT JOIN read_parquet('{(OUT / 'raw_hr.parquet').as_posix()}') r ON r.subject_id=h.subject_id AND r.stay_id=h.icustay_id AND r.charttime=h.t
        ORDER BY h.subject_id, h.icustay_id, h.t""").fetchnumpy()
    assert len(aux['t']) == len(chart) and np.array_equal(aux['t'], chart['t'].to_numpy(float))
    delay = aux['delay_s'].astype(float); minute = aux['mi']
    assert not np.isnan(delay).any()
    # EKG-Berichtsmerkmale
    rep = c.execute(f"SELECT m.study_id, m.machine_report FROM read_parquet('{MASTER}') m JOIN (SELECT DISTINCT study_id FROM a.eligible) e USING(study_id)").df()
    c.close()
    sinus_fn = load_report_sinus()
    rep['sinus'] = rep.machine_report.map(sinus_fn)
    rep['pace'] = rep.machine_report.fillna('').map(lambda s: PACE.search(s) is not None)
    e2 = ecg.merge(rep[['study_id', 'sinus', 'pace']], on='study_id', how='left', validate='one_to_one')
    assert (e2.subject_id.to_numpy() == ecg.subject_id.to_numpy()).all()
    ecg_sinus = (e2.sinus.fillna(False) & ~e2.pace.fillna(True)).to_numpy()
    report_stats = {'eligible_ecgs': int(len(ecg)), 'sinus_no_pacing': int(ecg_sinus.sum()), 'any_pacing_text': int(e2.pace.fillna(False).sum()), 'sinus_text': int(e2.sinus.fillna(False).sum())}
    V = {
        'V0 Basis (AW-110, Fenster 30 min)': dict(),
        'Va nur Sinusrhythmus ohne Stimulationshinweis': dict(ecg_mask=ecg_sinus),
        'Vb5 Chart: Speicherverzug 0 bis 5 min': dict(chart_mask=(delay >= 0) & (delay <= 300)),
        'Vb15 Chart: Speicherverzug 0 bis 15 min': dict(chart_mask=(delay >= 0) & (delay <= 900)),
        'Vc Chart: Minute ungleich 0': dict(chart_mask=(minute != 0)),
        'Vd10 Suchfenster 10 min': dict(window=600.0),
        'Vd15 Suchfenster 15 min': dict(window=900.0),
        'Ve1 Sinus + Verzug 0-15 min + Fenster 15 min': dict(ecg_mask=ecg_sinus, chart_mask=(delay >= 0) & (delay <= 900), window=900.0),
        'Ve2 Minute ungleich 0 + Verzug 0-15 min + Fenster 15 min': dict(chart_mask=(minute != 0) & (delay >= 0) & (delay <= 900), window=900.0),
    }
    rows = []; chart_counts = {}
    rng = np.random.default_rng(242005)
    for i, (name, kw) in enumerate(V.items()):
        G.status('r5', name)
        em = kw.get('ecg_mask'); cm = kw.get('chart_mask'); w = kw.get('window', 1800.0)
        df, kept = compute_persons(ecg, chart, window=w, chart_mask=cm, ecg_mask=em, keep=True)
        for col_, val_ in (('ok', False), ('offset', 0), ('width', 0), ('n_complete', 0)):
            if col_ not in df.columns:
                df[col_] = val_
        sub_ecg = ecg.loc[em] if em is not None else ecg
        chart_counts[name] = dict(chart_rows=int(cm.sum()) if cm is not None else int(len(chart)), ecg_rows=int(len(sub_ecg)))
        # Personen ohne verbleibendes EKG fehlen in df: Nenner bleibt die Primaermenge
        nulls = perm_null(kept, df, sub_ecg.reset_index(drop=True), 20, 242100 + i)
        # perm_null braucht idx relativ zu sub_ecg: kept['idx'] bezieht sich bereits auf die gefilterte Tabelle
        rows += row_for(name, df, den, nulls, rng)
        if i == 0:
            summ0 = summarize(df, den)
            ref = json.loads(Path(str(data_path("s5.uhrzeit_aw110_icu_results_json"))).read_text(encoding='utf-8'))
            for f in ('train', 'validation'):
                assert summ0[f]['identifiable_people'] == ref['folds'][f]['identifiable_people'] and summ0[f]['within'][60] == ref['folds'][f]['within']['60']['count']
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'r5_raw.csv', index=False)
    (OUT / 'r5_meta.json').write_text(json.dumps({'report_stats': report_stats, 'chart_counts': chart_counts, 'seconds': time.time() - t0, 'peak_rss': G.peak()}, indent=1))
    print('R5 fertig', round(time.time() - t0, 1), 's, RSS', G.peak())


if __name__ == '__main__':
    main()
