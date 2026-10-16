"""AW-244 A-R5 Nachsatz (nachtraeglich, post hoc): Katalog-Medianwerte der Tabelle 1, wenn die 25 nicht-konstanten periodischen Marker
mit den gewickelten ICC-Werten (Variante C aus a_r5.py) statt der Rohwerte eingehen. Reine Neuzusammenstellung vorhandener
Punktschaetzer (<local path>, nachrechnung/AW244/a_r5_wiederholbarkeit_periodisch.csv); keine Intervalle
(die Personenziehungen der Katalogmediane liegen nicht neu vor), keine neuen Daten.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def recompose(base, repl, pop):
    """base: Serie icc je marker_id (Population), repl: Serie ersetzter icc; Rueckgabe Kennzahlen."""
    new = base.copy()
    new.loc[repl.index] = repl.values
    est_old, est_new = base.dropna(), new.dropna()
    return {"population": pop, "markers_estimable_roh": int(est_old.size), "markers_estimable_neu": int(est_new.size),
            "median_roh": float(est_old.median()), "median_neu": float(est_new.median()),
            "anteil_ge_075_roh": float((est_old >= 0.75).mean()), "anteil_ge_075_neu": float((est_new >= 0.75).mean()),
            "zahl_ge_075_roh": int((est_old >= 0.75).sum()), "zahl_ge_075_neu": int((est_new >= 0.75).sum())}


def test():
    base = pd.Series([0.1, 0.8, 0.5, np.nan, 0.3], index=list("abcde"))
    repl = pd.Series([0.9], index=["a"])
    r = recompose(base, repl, "t")
    assert r["markers_estimable_neu"] == 4 and abs(r["median_roh"] - 0.4) < 1e-12 and abs(r["median_neu"] - 0.65) < 1e-12
    assert r["zahl_ge_075_neu"] == 2 and r["zahl_ge_075_roh"] == 1
    print("TEST_OK")


def run():
    shash = h244.require_stamp(Path(__file__))
    am = pd.read_csv(str(data_path("a.aw149_all_metrics_csv")))
    rep = pd.read_csv(h244.PUB / "a_r5_wiederholbarkeit_periodisch.csv")
    rows = []
    for pop in ("gesamt", "proxy"):
        base = am[am.population == pop].set_index("marker_id")["icc"]
        r = rep[(rep.population == pop)].set_index("marker_id")["icc_gew"].dropna()
        # delta_wave_frontal_axis_deg: ICC aus 497 Paaren; alle uebrigen wie berichtet
        out = recompose(base, r, pop)
        out["ersetzte_marker"] = int(r.size)
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(h244.PUB / "a_r5_katalogmedian.csv", index=False)
    (h244.PUB / "A_R5_katalogmedian_ausgabe.txt").write_text(f"Skript-SHA-256 {shash}\n" + df.to_string(index=False), encoding="utf8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
