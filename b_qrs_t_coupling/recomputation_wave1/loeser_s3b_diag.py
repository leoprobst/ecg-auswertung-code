"""AW-241 Diagnose zur Loeserkontrolle: Warum scheitert der Originalfit in einem Teil der Ziehungen mit 'Score/N-Konvergenz nicht erfuellt'?
Hypothese: logistic_fit bricht ab, sobald max|Gradient|/sum(w) < 1e-7; die Annahmepruefung in statistik.fit verlangt max|Gradient|/N < 1e-7
(N = Zahl der Zeilen mit Gewicht > 0, in Ziehungen etwa 63 % von sum(w)). Fuer Ziehungen, die der Originalfit ablehnt, wird beides berechnet.
Liest dieselben Testprojektionen wie loeser_s3b.py (innerhalb desselben Beginn/Ende-Fensters von AW-241 Phase L). Ausgabe nur Aggregate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
import numpy as np  # noqa: E402
import loeser_s3b as L  # noqa: E402
from scipy.special import expit  # noqa: E402
from werkzeuge.b_test import statistik as st  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

ROWS = ["025", "062", "046"]


def main():
    frames, package = L.load_frames()
    union, rows = L.capture_designs(frames, package)
    out = {}
    for name in ROWS:
        d = rows[name]
        z, y, ids = d["z"], d["y"], d["ids"]
        state = json.loads((L.OUT / f"row_{name}.json").read_text(encoding="utf8"))
        failed = [r["r"] for r in state["reps"] if not r["old_ok"]]
        okd = [r["r"] for r in state["reps"] if r["old_ok"]][:150]
        pen = np.r_[0, np.full(z.shape[1] - 1, 1e-6)]
        res = {}
        for label, draws in (("abgelehnt", failed), ("angenommen_stichprobe", okd)):
            ratio_w, ratio_n, frac_active = [], [], []
            for r in draws:
                w = st.draws(union, ids, r)
                a = w > 0
                zz, yy, ww = z[a], y[a], w[a]
                b = logistic_fit(zz, yy, ww)
                g = np.max(np.abs(zz.T @ (ww * (expit(zz @ b) - yy)) + pen * b))
                ratio_w.append(g / ww.sum())
                ratio_n.append(g / len(yy))
                frac_active.append(len(yy) / ww.sum())
            res[label] = dict(n=len(draws), max_score_over_sumw=float(np.max(ratio_w)), min_score_over_sumw=float(np.min(ratio_w)),
                              max_score_over_N=float(np.max(ratio_n)), share_N_criterion_violated=float(np.mean(np.array(ratio_n) >= 1e-7)),
                              share_sumw_criterion_violated=float(np.mean(np.array(ratio_w) >= 1e-7)), mean_active_rows_over_sumw=float(np.mean(frac_active)))
        out[name] = res
        print(name, json.dumps(res), flush=True)
    Path(str(data_path("b.nachrechnung_aw241_loeser_s3b_diagnose_ausgabe_txt"))).write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
