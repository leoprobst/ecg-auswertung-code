"""AW-242 R0: Basisrechnung des Kerns ueber alle Personen, Parity gegen AW-110 (icu_results.json).

Schreibt private Personentabelle und Kurvenmatrizen nach <local path> (Pickle). Keine Testpartition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, pickle, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('r0')
import numpy as np, pandas as pd
from aw242_data import load_audit, compute_persons, summarize

OUT = G.OUT
t0 = time.time()
G.status('r0', 'lade')
data = load_audit()
den = data['people'].groupby('fold').size().to_dict()
G.status('r0', 'rechne')
df, kept = compute_persons(data['ecg'], data['chart'], keep=True)
summ = summarize(df, den)
ref = json.loads(Path(str(data_path("s5.uhrzeit_aw110_icu_results_json"))).read_text(encoding='utf-8'))
checks = {}
for f in ('train', 'validation'):
    r = ref['folds'][f]; s = summ[f]
    pairs = {'primary_people': (r['primary_people'], s['primary_people']), 'complete_people': (r['complete_people'], s['complete_people']),
             'complete_ecgs': (r['complete_ecgs'], s['complete_ecgs']), 'at_least_3_people': (r['at_least_3_people'], s['at_least_3_people']),
             'identifiable_people': (r['identifiable_people'], s['identifiable_people']),
             'within60': (r['within']['60']['count'], s['within'][60]), 'within15': (r['within']['15']['count'], s['within'][15]),
             'within360': (r['within']['360']['count'], s['within'][360]),
             'popopt': (r['population_optimum_minutes'], s['population_optimum_minutes']),
             'width_median': (r['width_median'], s['width_median'])}
    checks[f] = {k: {'ref': a, 'neu': b, 'gleich': bool(a == b)} for k, (a, b) in pairs.items()}
    assert all(v['gleich'] for v in checks[f].values()), (f, checks[f])
(OUT / 'r0_parity.json').write_text(json.dumps({'summary': summ, 'parity': checks, 'seconds': time.time() - t0, 'peak_rss': G.peak()}, indent=1, default=str))
with open(OUT / 'base_persons.pkl', 'wb') as fh:
    pickle.dump({'persons': df, 'kept': kept}, fh, protocol=4)
G.status('r0', 'fertig')
print(json.dumps({'summary': summ, 'seconds': time.time() - t0, 'peak_rss': G.peak()}, default=str))
