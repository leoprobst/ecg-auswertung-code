"""AW-241 B-R6 Teil 1 (Entwicklung): Echoabstand mit Richtung, Stimulationshinweis, LVEF-Methode. Aufrufe: stamp | run.
Vertrag: nachrechnung/AW241/B_R6_PROTOKOLL.md.  Nur train/validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design, fit as st_fit  # noqa: E402

OUT = Path(str(data_path("b.aw241_r6")))
AGG = ROOT / "nachrechnung/AW241"
PROTOKOLL = AGG / "B_R6_PROTOKOLL.md"
CTX = str(data_path("b.mimic_validation_evidence_clinical_context_features_parquet"))
F = C.F
FOLDS = ["train", "validation"]
WINDOWS = [("<=24h", 24), ("<=7d", 168), ("<=14d", 336), ("<=30d", 720)]


def sup(n):
    return int(n) if n >= 10 else "<10"


def stamp():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "protokoll_hash.json"
    if p.exists():
        raise SystemExit("bereits gebunden")
    C.dump(p, {"path": str(PROTOKOLL), "sha256": C.sha(PROTOKOLL), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "before_computation": True})
    print("STAMP", C.sha(PROTOKOLL))


def estimate(z, y, ids):
    try:
        r = st_fit(z, y, ids)
        return dict(status="schaetzbar", N=int(r["N"]), cases=sup(int((y == 1).sum())), controls=sup(int((y == 0).sum())), OR=round(r["or"], 3),
                    ci=f"{r['or_ci95_wald'][0]:.3f}; {r['or_ci95_wald'][1]:.3f}", dAUC=round(r["delta_auc"], 4))
    except Exception as ex:  # noqa: BLE001
        return dict(status="nicht schaetzbar: " + str(ex)[:60], N=sup(len(y)), cases=sup(int((y == 1).sum())), controls=sup(int((y == 0).sum())), OR="n/a", ci="n/a", dAUC="n/a")


def write(name, rows):
    with (AGG / name).open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list({k: 1 for r in rows for k in r}.keys()))
        w.writeheader()
        w.writerows(rows)


def run():
    s = C.load(OUT / "protokoll_hash.json")
    assert s["sha256"] == C.sha(PROTOKOLL)
    f, info, scale = C.outcome_frame()
    g = f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=2")
    ctx = con.execute(f"SELECT study_id, current_report_pacemaker_state, dx_cardiac_device_status, proc_pacemaker_or_icd, proc_permanent_pacemaker_implant, preferred_lvef_method, nearest_tte_delta_hours AS delta_ctx "
                      f"FROM read_parquet('{CTX}') WHERE split IN ('train','validation')").df()
    con.close()
    dl, _ = C.read(C.POP, ["study_id", "split", "nearest_tte_delta_hours"])
    g = g.merge(dl[["study_id", "nearest_tte_delta_hours"]], on="study_id", how="left", validate="one_to_one")
    g = g.merge(ctx, on="study_id", how="left", validate="one_to_one")
    assert g.study_id.notna().all() and set(g.split.unique()) <= {"train", "validation"}
    delta = g.nearest_tte_delta_hours.to_numpy(float)
    assert np.isfinite(delta).all() and np.allclose(delta, g.delta_ctx.to_numpy(float), equal_nan=False)
    raw, _ = raw_design(g, F)
    Z = fixed_design(raw, scale)
    y = classify_lvef(g.preferred_lvef.to_numpy(float))
    ids = g.subject_id.to_numpy()
    rep = g.current_report_pacemaker_state.eq("positive").to_numpy()
    t3 = ["dx_cardiac_device_status", "proc_pacemaker_or_icd", "proc_permanent_pacemaker_implant"]
    dev = np.zeros(len(g), bool)
    for c_ in t3:
        dev |= g[c_].eq(True).fillna(False).to_numpy(dtype=bool)
    anyknown = g[t3].notna().any(axis=1).to_numpy(dtype=bool)
    cls = np.where(rep, "P", np.where(dev, "D", np.where(~anyknown, "U", "N")))
    lines = ["AW-241 B-R6 Teil 1 (Entwicklung, post hoc, nur Aggregate). Skripthash br6_run.py: " + C.sha(__file__), ""]
    rows_dist, rows_win, rows_pace, rows_cls, rows_meth = [], [], [], [], []
    for fold in FOLDS:
        m = g.split.eq(fold).to_numpy()
        d = delta[m]
        yy = y[m]
        # (a) Verteilung
        for grp, gv in (("Faelle", 1.0), ("Kontrollen", 0.0)):
            for dirn, dm in (("Echo vor ECG", d < 0), ("Echo nach ECG", d >= 0), ("alle", np.ones(len(d), bool))):
                sel = (yy == gv) & dm
                if sel.sum() < 10:
                    rows_dist.append(dict(fold=fold, gruppe=grp, richtung=dirn, n="<10"))
                    continue
                days = np.abs(d[sel]) / 24
                rows_dist.append(dict(fold=fold, gruppe=grp, richtung=dirn, n=int(sel.sum()), median_tage=round(float(np.median(days)), 2), q1=round(float(np.quantile(days, .25)), 2),
                                      q3=round(float(np.quantile(days, .75)), 2), **{f"anteil_{w}": round(float((days <= h / 24).mean()), 3) for w, h in WINDOWS}))
        # (a) Fenster und Richtung
        for wname, h in WINDOWS:
            for dirn, dm in (("alle", np.ones(len(d), bool)), ("Echo vor ECG", d < 0), ("Echo nach ECG", d >= 0)):
                sel = (np.abs(d) <= h) & dm
                e = estimate(Z[m][sel], yy[sel], ids[m][sel])
                rows_win.append(dict(fold=fold, fenster=wname, richtung=dirn, **e))
        # (b) Stimulationshinweis
        cm = cls[m]
        for lab in ("P", "D", "U", "N"):
            n = (cm == lab).sum()
            rows_cls.append(dict(fold=fold, klasse=lab, n=sup(n), faelle=sup(int(((cm == lab) & (yy == 1)).sum())), kontrollen=sup(int(((cm == lab) & (yy == 0)).sum())),
                                 anteil=round(float(n / len(cm)), 4)))
        for label, keep in (("alle", np.ones(len(cm), bool)), ("ohne P", cm != "P"), ("ohne P und D", ~np.isin(cm, ["P", "D"])), ("nur N (streng)", cm == "N"), ("nur P", cm == "P"), ("nur D", cm == "D")):
            e = estimate(Z[m][keep], yy[keep], ids[m][keep])
            rows_pace.append(dict(fold=fold, menge=label, **e))
        # (c) Methode
        meth = g.preferred_lvef_method.to_numpy()[m]
        for me in ("generic", "biplane", "3d"):
            nn, nc, nk = int((meth == me).sum()), int(((meth == me) & (yy == 1)).sum()), int(((meth == me) & (yy == 0)).sum())
            if min(nc, nk) < 10:   # Kleinzellenregel: Komplemente vergroebern
                lo = nn // 10 * 10
                rows_meth.append(dict(fold=fold, methode=me, n=f"{lo}-{lo + 9}", faelle=sup(nc), kontrollen=f"{(nn - 9) // 10 * 10}-{(nn - 9) // 10 * 10 + 19}" if nc < 10 else sup(nk)))
            else:
                rows_meth.append(dict(fold=fold, methode=me, n=nn, faelle=nc, kontrollen=nk))
        ng = np.isin(meth, ["biplane", "3d"])
        e = estimate(Z[m][ng], yy[ng], ids[m][ng])
        nn, nc, nk = int(ng.sum()), int((ng & (yy == 1)).sum()), int((ng & (yy == 0)).sum())
        coarse = any(min(int(((meth == me) & (yy == 1)).sum()), int(((meth == me) & (yy == 0)).sum())) < 10 for me in ("biplane", "3d"))
        if coarse:   # Komplement einer unterdrueckten Zelle (Teilmenge) darf nicht ableitbar sein
            rows_meth.append(dict(fold=fold, methode="nicht-generisch (biplan+3D)", n=f"{nn // 20 * 20}-{nn // 20 * 20 + 19}", faelle=f"{nc // 10 * 10}-{nc // 10 * 10 + 9}",
                                  kontrollen=f"{nk // 20 * 20}-{nk // 20 * 20 + 19}", OR=e["OR"], ci=e["ci"], status=e["status"]))
        else:
            rows_meth.append(dict(fold=fold, methode="nicht-generisch (biplan+3D)", n=nn, faelle=nc, kontrollen=nk, OR=e["OR"], ci=e["ci"], status=e["status"]))
    write("b_r6_verteilung.csv", rows_dist)
    write("b_r6_fenster_richtung.csv", rows_win)
    write("b_r6_stimulation_klassen.csv", rows_cls)
    write("b_r6_stimulation_sensitivitaet.csv", rows_pace)
    write("b_r6_methode.csv", rows_meth)
    for nm, rows in (("Fenster/Richtung", rows_win), ("Stimulationsklassen", rows_cls), ("Stimulationssensitivitaet", rows_pace), ("Methode", rows_meth)):
        lines.append("== " + nm)
        lines += [str(r) for r in rows]
    (AGG / "B_R6_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"stamp": stamp, "run": run}[sys.argv[1]]()
