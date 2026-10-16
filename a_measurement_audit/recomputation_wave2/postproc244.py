"""Post-processing of the public aggregates after the small-cell check (kleinzellen_scan.py).
- b_r8_or_gitter.csv: columns n_available/cases_available removed (small counts of missing cases below the disclosure threshold could be back-calculated); the information is given in b_r8_meta.json as '<10'.
- a_r5_bereichsscreen.csv: column n removed; a_r5_binaer_matrix.csv: n_paare rounded to 10 (differences to the column maximum below the disclosure threshold could be back-calculated).
- a_r6_zusammensetzung.csv: rows of the (date-shifted) recording years removed (not informative).
The unredacted versions are kept in a private location outside the repository.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import shutil
import sys
from pathlib import Path

import pandas as pd

PUB = Path(str(data_path("a.nachrechnung_aw244")))
PRIV = Path(str(data_path("a.aw244_ohne_kleinzellenbereinigung")))
PRIV.mkdir(exist_ok=True)
for name in ("b_r8_or_gitter.csv", "a_r6_zusammensetzung.csv", "a_r5_bereichsscreen.csv", "a_r5_binaer_matrix.csv"):
    if not (PRIV / name).exists():
        shutil.copy2(PUB / name, PRIV / name)
d = pd.read_csv(PRIV / "b_r8_or_gitter.csv")
d.drop(columns=["n_available", "cases_available"]).to_csv(PUB / "b_r8_or_gitter.csv", index=False)
c = pd.read_csv(PRIV / "a_r6_zusammensetzung.csv")
c[c.variable != "year"].to_csv(PUB / "a_r6_zusammensetzung.csv", index=False)
b = pd.read_csv(PRIV / "a_r5_bereichsscreen.csv")
b.drop(columns=["n"]).to_csv(PUB / "a_r5_bereichsscreen.csv", index=False)
m = pd.read_csv(PRIV / "a_r5_binaer_matrix.csv")
m["n_paare"] = (m["n_paare"] / 10).round() * 10
m.to_csv(PUB / "a_r5_binaer_matrix.csv", index=False)
print("postproc ok")
