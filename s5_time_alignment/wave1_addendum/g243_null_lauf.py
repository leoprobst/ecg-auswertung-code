"""AW-243 Teil 4: strengere Null (zirkulaer verschobene Chartreihe) fuer S5-R2, nachtraeglich (post hoc).

Aufruf: python g243_null_lauf.py <variante: n0|n1|n2|n3> <wiederholungen>
Eingabe: <local path> (Personen mit mindestens 3 vollstaendigen EKG, Chartreihen je Aufenthalt, nur
Entwicklungsfolds). Ausgabe: <local path><variante>.csv (Zaehlwerte je Wiederholung und Fold, keine Kennungen).
Zeitpunkte der Chartwerte und Aufenthaltsgrenzen bleiben unveraendert, daher bleibt die Vollabdeckung je EKG unveraendert.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, time, pickle, json
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import g243_guard as G
G.start('null')
import numpy as np, pandas as pd
import g243_null_lib as L

VAR = sys.argv[1]; R = int(sys.argv[2])
SEED = 243
VAR_ID = {'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3}[VAR]


def prepare_persons():
    P = pickle.load(open(str(data_path("s5.aw242_r2_persons_pkl")), 'rb'))
    out = []
    for p in P:
        stays = sorted(p['stays'].keys())
        picks = []
        for i, (t, s) in enumerate(zip(p['t'], p['stay_ids'])):
            tt, vv, _ = p['stays'][int(s)]
            picks.append(L.pick_matrix(np.array([t]), tt)[0])
        out.append(dict(fold=p['fold'], stays=stays, S=p['stays'], stay_ids=p['stay_ids'].astype(int), picks=np.vstack(picks), h=p['h_real'],
                        V=p['V'], ok=p['ok'], offset=p['offset'], width=p['width']))
    return out


def values(person, shifts):
    """Referenzmatrix V (n,K) aus (ggf. verschobenen) Chartreihen je Aufenthalt."""
    rows = []
    for i, s in enumerate(person['stay_ids']):
        vv = shifts[int(s)] if shifts else person['S'][int(s)][1]
        rows.append(vv[person['picks'][i]])
    return np.vstack(rows)


def main():
    t0 = time.time()
    persons = prepare_persons()
    # Paritaet: Matrix aus den Chartreihen gleich der gespeicherten V; Identifizierbarkeit gleich der gespeicherten
    mism = 0; ident_mism = 0; n_ident = 0
    for p in persons:
        V = values(p, None)
        if not np.array_equal(V, p['V']):
            mism += 1
        ok, off, width, lo = L.estimate(V, p['h'])
        n_ident += ok
        if (ok != p['ok']) or (ok and off != p['offset']):
            ident_mism += 1
    par = dict(personen=len(persons), matrix_abweichend=mism, identifizierbarkeit_abweichend=ident_mism, identifizierbar_beobachtet=int(n_ident))
    print('Paritaet', par, flush=True)
    rows = []
    for rep in range(R):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, VAR_ID, rep]))
        per = {f: dict(n=0, ident=0, w15=0, w60=0, w120=0, gt60=0, shifts=[]) for f in ('train', 'validation')}
        for p in persons:
            if VAR == 'n0':
                shifts = None
            else:
                shifts = {}
                for s in p['stays']:
                    tt, vv, _ = p['S'][s]
                    sh_s = L.draw_shift(rng, tt, VAR)
                    shifts[s], _ = L.circ_shift_values(tt, vv, sh_s)
            V = values(p, shifts)
            ok, off, width, lo = L.estimate(V, p['h'])
            q = per[p['fold']]
            q['n'] += 1
            if ok:
                q['ident'] += 1; q['w15'] += abs(off) <= 15; q['w60'] += abs(off) <= 60; q['w120'] += abs(off) <= 120; q['gt60'] += abs(off) > 60
        for f, q in per.items():
            rows.append(dict(rep=rep, fold=f, n=q['n'], ident=q['ident'], w15=int(q['w15']), w60=int(q['w60']), w120=int(q['w120']), gt60=int(q['gt60'])))
        if rep % 10 == 0:
            print('rep', rep, 'ident', sum(r['ident'] for r in rows[-2:]), round(time.time() - t0, 1), 's', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / f'null_{VAR}.csv', index=False)
    (G.OUT / f'null_{VAR}_meta.json').write_text(json.dumps(dict(variant=VAR, reps=R, seed=SEED, parity=par, dauer_s=round(time.time() - t0, 1), rss_gb=round(G.peak() / 2**30, 2)), indent=1))
    print('fertig', round(time.time() - t0, 1), 's, RSS', round(G.peak() / 2**30, 2), 'GB')


if __name__ == '__main__':
    main()
