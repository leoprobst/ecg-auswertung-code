"""AW-241 B-R6 Teil 2: Echoabstand, Richtung und Stimulationshinweis in der internen Assessmentpartition (Testprojektion BEF-0063).
Nur nach Beginn-Eintrag in koordination/TESTZUGRIFFE.md. Liest ausschliesslich die Projektion laut input_manifest_v5 (Hashpruefung);
locked_test wird nicht beruehrt. Wald-Intervalle (CR1), keine Bootstrapintervalle. Ausgabe nur Aggregate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
import numpy as np  # noqa: E402
import loeser_s3b as L  # noqa: E402
from werkzeuge.b_test import adapter  # noqa: E402

AGG = Path(str(data_path("b.nachrechnung_aw241")))
WINDOWS = [("<=24h", 24), ("<=7d", 168), ("<=14d", 336), ("<=30d", 720)]


def sup(n):
    return int(n) if n >= 10 else "<10"


def main():
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from werkzeuge.b_test.daten import select_index

    for key, p in (("manifest", L.MANIFEST), ("params", L.PARAMS), ("aw142", L.AW142)):
        if L.sha(p) != L.EXPECT[key]:
            raise RuntimeError("Hashabweichung " + key)
    manifest = json.loads(L.MANIFEST.read_text(encoding="utf8"))
    package = json.loads(L.PARAMS.read_text(encoding="utf8"))
    src = manifest["hypotheses"]["BEF-0063"]
    p = Path(src["path"])
    if any(s in str(p).lower() for s in ("locked_", "diagnostics", "publication_preparation")):
        raise PermissionError("gesperrter Pfad")
    if L.sha(p) != src["sha256"]:
        raise RuntimeError("Hashabweichung BEF-0063")
    names = pq.ParquetFile(p).schema_arrow.names
    cols = src["columns"]
    has_pace = "current_report_pacemaker_state" in names
    cols = list(cols) + (["current_report_pacemaker_state"] if has_pace and "current_report_pacemaker_state" not in cols else [])
    f = ds.dataset(str(p), format="parquet").to_table(columns=cols).to_pandas()
    if set(f.split) != {"test"}:
        raise ValueError("Partition nicht 'test'")
    f63 = select_index(f, package["hypotheses"]["BEF-0063"]["endpoint"])
    del f
    h1 = package["hypotheses"]["BEF-0063"]
    union = np.array([0])
    old = json.loads(L.AW142.read_text(encoding="utf8"))
    ref7 = [x for x in old["sensitivities"] if x["rank"] == "2a"][0]
    # Gate: Echo +-7 Tage am unveraenderten Index
    g = adapter.estimate(f63, h1, union, lambda z: z.time_delta_hours.abs().le(168), bootstrap=False)
    for key in ("beta", "or", "delta_auc", "auc_G", "auc_G_marker"):
        if not np.isclose(g[key], ref7[key], rtol=1e-9, atol=0):
            raise RuntimeError(f"Gate +-7 Tage: {key} weicht ab")
    if g["N"] != ref7["N"]:
        raise RuntimeError("Gate N")
    lines = [f"AW-241 B-R6 Teil 2 (interne Assessmentpartition, post hoc). Skripthash br6_assess.py: {L.sha(__file__)}",
             f"Gate bestanden: +-7 Tage reproduziert OR {g['or']:.3f} (N {g['N']}), AW-142 gleich auf 1e-9.", f"Pacemaker-Spalte in der Projektion vorhanden: {has_pace}", ""]
    rows = []

    def add(label, win, dirn, sel):
        r = adapter.estimate(f63, h1, union, sel, bootstrap=False)
        if r.get("status") != "schaetzbar":
            rows.append(dict(menge=label, fenster=win, richtung=dirn, status="nicht schaetzbar: " + str(r.get("reason"))[:60]))
            return
        cp = r["class_persons"]
        rows.append(dict(menge=label, fenster=win, richtung=dirn, status="schaetzbar", N=int(r["N"]), faelle=sup(cp["1"]), kontrollen=sup(cp["0"]), OR=round(r["or"], 3),
                         ci=f"{r['or_ci95_wald'][0]:.3f}; {r['or_ci95_wald'][1]:.3f}", dAUC=round(r["delta_auc"], 4)))

    for wname, h in WINDOWS:
        for dirn, fn in (("alle", lambda z: z.time_delta_hours.abs().le(h)), ("Echo vor ECG", lambda z: z.time_delta_hours.abs().le(h) & z.time_delta_hours.lt(0)),
                         ("Echo nach ECG", lambda z: z.time_delta_hours.abs().le(h) & z.time_delta_hours.ge(0))):
            add("alle ECG", wname, dirn, fn)
    if has_pace:
        nopace = lambda z: ~z.current_report_pacemaker_state.eq("positive")  # noqa: E731
        add("ohne Berichtshinweis Stimulation (P)", "<=30d", "alle", nopace)
        add("ohne Berichtshinweis Stimulation (P)", "<=7d", "alle", lambda z: nopace(z) & z.time_delta_hours.abs().le(168))
        n_p = int(f63.current_report_pacemaker_state.eq("positive").sum())
        lines.append(f"Index-ECG mit Berichtshinweis Stimulation in der Projektion (alle Personen des Index): {sup(n_p)} von {len(f63)}")
    with (AGG / "b_r6_assessment.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list({k: 1 for r in rows for k in r}.keys()))
        w.writeheader()
        w.writerows(rows)
    lines += [str(r) for r in rows]
    (AGG / "B_R6_assessment_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
