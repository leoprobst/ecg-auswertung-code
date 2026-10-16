"""AW-243 Teil 4: Aggregation der Nullverteilungen (zirkulaere Verschiebung n1/n2/n3) und Vergleich mit der Permutationsnull (AW-242)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'nachrechnung/AW243'
P3 = Path(str(data_path("s5.aw243"))); P2 = Path(str(data_path("s5.aw242")))
OBS = {'gesamt': dict(n=1979, ident=537, w60=365, gt60=172, w15=None, w120=None),
       'train': dict(n=1482, ident=392, w60=263, gt60=129), 'validation': dict(n=497, ident=145, w60=102, gt60=43)}
kz = lambda n: '<10' if 0 < n < 10 else str(int(n))


def per_rep(df, fold):
    x = df if fold == 'gesamt' else df[df.fold == fold]
    return x.groupby('rep')[[c for c in x.columns if c not in ('rep', 'fold')]].sum()


def summarize(name, df, fold, w60col='w60'):
    g = per_rep(df, fold)
    o = OBS[fold]
    res = dict(null=name, fold=fold, reps=len(g), n=int(g['n'].iloc[0]))
    for col, key in (('ident', 'ident'), (w60col, 'w60'), ('gt60' if 'gt60' in g else 'abs_gt60', 'gt60')):
        v = g[col].to_numpy(float)
        res[key + '_mittel'] = round(float(v.mean()), 1)
        res[key + '_2_5'] = round(float(np.percentile(v, 2.5)), 1); res[key + '_97_5'] = round(float(np.percentile(v, 97.5)), 1)
        res[key + '_beobachtet'] = o[key]
    v = g['ident'].to_numpy(float)
    res['p_ident_null'] = round(float(v.mean() / res['n']), 4)
    res['p_ident_beobachtet'] = round(o['ident'] / res['n'], 4)
    res['mischungsanteil'] = round((o['ident'] / res['n'] - v.mean() / res['n']) / (1 - v.mean() / res['n']), 4)
    lo_r = float(np.percentile(v, 2.5)) / res['n']; hi_r = float(np.percentile(v, 97.5)) / res['n']
    r_obs = o['ident'] / res['n']
    res['mischungsanteil_untere_grenze'] = round((r_obs - hi_r) / (1 - hi_r), 4)  # Null an der oberen Grenze
    res['mischungsanteil_obere_grenze'] = round((r_obs - lo_r) / (1 - lo_r), 4)
    # empirische einseitige Anteile (Wiederholungen mit Wert mindestens so gross wie beobachtet, bzw. hoechstens so gross fuer 'gt60')
    res['anteil_null_ident_ge_beob'] = round(float(((v >= o['ident']).sum() + 1) / (len(v) + 1)), 4)
    w = g[w60col].to_numpy(float)
    res['anteil_null_w60_ge_beob'] = round(float(((w >= o['w60']).sum() + 1) / (len(w) + 1)), 4)
    gt = g['gt60' if 'gt60' in g else 'abs_gt60'].to_numpy(float)
    res['anteil_null_gt60_le_beob'] = round(float(((gt <= o['gt60']).sum() + 1) / (len(gt) + 1)), 4)
    res['ueberschuss_w60'] = round(o['w60'] - float(w.mean()), 1)
    res['ueberschuss_w60_anteil_der_beobachteten'] = round((o['w60'] - float(w.mean())) / o['w60'], 3)
    return res


def main():
    rows = []
    perm = pd.read_csv(P2 / 'r2_perm.csv')
    perm = perm.rename(columns={'within60': 'w60', 'abs_gt60': 'gt60', 'within120': 'w120'})
    nulls = [('Permutation innerhalb der Person (AW-242)', perm)]
    for v, lab in (('n1', 'Zirkulaere Verschiebung der Chartreihe, mindestens 24 h (N1)'), ('n2', 'Zirkulaere Verschiebung, mindestens 72 h (N2)'),
                   ('n3', 'Zirkulaere Verschiebung um Vielfache von 24 h, erhaelt die Tagesphase (N3, Diagnosevariante)')):
        f = P3 / f'null_{v}.csv'
        if f.exists():
            nulls.append((lab, pd.read_csv(f)))
    for name, df in nulls:
        for fold in ('gesamt', 'train', 'validation'):
            rows.append(summarize(name, df, fold))
    out = pd.DataFrame(rows)
    out.to_csv(PUB / 'S5_NACHTRAG_null_zirkulaer.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(out[out.fold == 'gesamt'].T.to_string())
    print(out[out.fold != 'gesamt'][['null', 'fold', 'ident_mittel', 'ident_2_5', 'ident_97_5', 'ident_beobachtet', 'w60_mittel', 'w60_beobachtet', 'gt60_mittel', 'gt60_beobachtet', 'mischungsanteil']].to_string())


if __name__ == '__main__':
    main()
