"""AW-243 Teil 2: Rohzaehlung -> veroeffentlichbare Tabelle (Kleinzellenregel 1 bis 9 als <10)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
raw = pd.read_csv(str(data_path("s5.aw243_audit_zaehlung_roh_csv")))
base = ['werk', 'stelle', 'groesse', 'fold', 'n']
meas = [c for c in raw.columns if c not in base and c != 'hinweis' and c != 'tor_ueberhaupt_erreicht']
rows = []
for r in raw.itertuples(index=False):
    d = r._asdict()
    for c in meas:
        v = d[c]
        if pd.isna(v):
            continue
        v = int(v)
        rows.append(dict(werk=d['werk'], stelle=d['stelle'], groesse=d['groesse'], fold={'train': 'Entwicklung', 'validation': 'Interne Pruefung'}[d['fold']],
                         nenner=int(d['n']), kennzahl=c, wert=('<10' if 0 < v < 10 else str(v))))
out = pd.DataFrame(rows)
mask = out.kennzahl.isin(['personen_ausgewaehlt_archiv', 'personen_ausgewaehlt_exakt']) & (out.fold == 'Interne Pruefung')
out.loc[mask, 'wert'] = '>9745'  # Gesamtzahl der Landmarken der internen Pruefung nicht veroeffentlichen (Differenz < 10)
out.to_csv(ROOT / 'nachrechnung/AW243/AUDIT_ZEIT_zaehlung.csv', index=False)
print(out.to_string(max_colwidth=60))
