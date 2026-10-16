# -*- coding: utf-8 -*-
"""AW-245 S5-R6, Lauf (Version 1.0): individuelle Unsicherheit des Uhrversatzes. Nachtraeglich (post hoc), explorativ, nur Entwicklungsfolds train und validation.

Aufruf: python -I -B s5_r6_lauf.py [--test]   (--test: 60 Personen, R=100, 3 Pseudo-Datensaetze)
Eingabe: <local path> (1.979 Personen mit mindestens 3 vollstaendigen EKG, nur lesen; Hash wird eingetragen).
Ausgabe privat (Personenzeilen, Kennungen): <local path>
Ausgabe Aggregate (ohne Kennungen, Kleinzellen 1 bis 9 als "<10"): nachrechnung/AW245/S5_R6_*.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import hashlib
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "aw242"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import s5_r6_lib as L  # noqa: E402
from aw242_kern import D, estimate_from_loss, loss_curve  # noqa: E402

TEST = "--test" in sys.argv
ROOT = HERE.parents[1]
PRIV = Path(str(data_path("s5.aw245")))
PUB = ROOT / "nachrechnung/AW245"
PRIV.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)
PKL = Path(str(data_path("s5.aw242_r2_persons_pkl")))
KERN_HASH = "8ebb2b92629fd8d61b3967502333839f363789a5ed75cb50d160bc2aa7f24c83"
R_OBS = 100 if TEST else 1000
R_NULL = 50 if TEST else 200
K_NULL = 3 if TEST else 20
DENOM_ICU = {"train": 17067, "validation": 5846}
FOLD_NAME = {"train": "Entwicklung", "validation": "interne Pruefung"}
PEAK = [0]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def guard(min_free_gb=1.5, rss_gb=1.4):
    import psutil

    def check():
        PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss)
        if psutil.virtual_memory().available < min_free_gb * 2 ** 30 or PEAK[0] > rss_gb * 2 ** 30:
            (PRIV / "s5_r6_stop.json").write_text(json.dumps({"reason": "Ressourcentor", "rss": PEAK[0]}))
            os._exit(75)

    check()

    def loop():
        while True:
            check()
            time.sleep(1)

    threading.Thread(target=loop, daemon=True).start()


def status(msg):
    (PRIV / "s5_r6_status.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "msg": msg, "peak_rss": PEAK[0]}))
    print(msg, flush=True)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def bh_reject(p, q=0.05):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    thr = q * (np.arange(1, m + 1) / m)
    ok = p[order] <= thr
    out = np.zeros(m, bool)
    if ok.any():
        kmax = np.max(np.flatnonzero(ok))
        out[order[:kmax + 1]] = True
    return out


def kz(x):
    """Kleinzellenregel: 1 bis 9 als '<10'."""
    x = int(x)
    return "<10" if 1 <= x <= 9 else str(x)


def main():
    guard()
    t0 = time.time()
    kh = sha(HERE.parent / "aw242/aw242_kern.py")
    if kh != KERN_HASH:
        raise SystemExit("aw242_kern.py weicht ab: " + kh)
    persons = pickle.load(open(PKL, "rb"))
    if TEST:
        persons = persons[:60]
    pkl_hash = sha(PKL)
    status(f"geladen {len(persons)} Personen, pkl {pkl_hash[:12]}")

    # ---- Beobachtete Personen ----
    rows = []
    for i, p in enumerate(persons):
        rng = np.random.default_rng(np.random.SeedSequence([245, 6, 1, i]))
        loss, e = L.curve_estimate(p["V"], p["h_real"])
        pa, na, ex_a = L.p_all_eligible(p, e["minimum"], rng, R_OBS)
        pc, nc, ex_c = L.p_complete_only(p, e["minimum"], rng, R_OBS, p["h_real"])
        offs, oks, dev = L.loo(p["V"], p["h_real"], e["offset"])
        rows.append(dict(subject_id=p["subject_id"], fold=p["fold"], n=len(p["h_real"]), m=len(p["all_h"]), ok=bool(e["ok"]), offset=e["offset"], width=e["width"],
                         minimum=e["minimum"], p_all=pa, nperm_all=na, exact_all=ex_a, p_comp=pc, nperm_comp=nc, exact_comp=ex_c,
                         loo_maxdev=dev, loo_ok_share=float(np.mean(oks)), loo_min=int(offs.min()), loo_max=int(offs.max())))
        if i % 200 == 0:
            status(f"beobachtet {i}/{len(persons)}")
    df = pd.DataFrame(rows)
    df.to_csv(PRIV / ("s5_r6_personen_TEST.csv" if TEST else "s5_r6_personen.csv"), index=False)
    # Parität
    par = dict(personen=len(df), identifizierbar=int(df.ok.sum()), je_fold={f: int(df[(df.fold == f)].ok.sum()) for f in ("train", "validation")})
    status("Paritaet " + json.dumps(par))

    # ---- Pseudo-Datensaetze (kein Zeitsignal) ----
    null_rows = []
    for k in range(K_NULL):
        rng = np.random.default_rng(np.random.SeedSequence([245, 6, 2, k]))
        for i, p in enumerate(persons):
            vals = rng.permutation(p["all_h"])
            h = vals[p["local_idx"]]
            loss, e = L.curve_estimate(p["V"], h)
            pa, na, ex = L.p_value(e["minimum"], p["V"], p["all_h"], p["local_idx"], R_NULL, rng)
            dev = None
            if e["ok"]:
                dev = L.loo(p["V"], h, e["offset"])[2]
            null_rows.append(dict(k=k, fold=p["fold"], n=len(h), m=len(p["all_h"]), ok=bool(e["ok"]), offset=e["offset"], p_all=pa, loo_maxdev=dev))
        status(f"Pseudo-Datensatz {k + 1}/{K_NULL}")
    nd = pd.DataFrame(null_rows)
    nd.to_csv(PRIV / ("s5_r6_null_TEST.csv" if TEST else "s5_r6_null.csv"), index=False)

    # ---- Aggregation ----
    out = []
    def kennzahlen(d, label, groups, null_d=None):
        for gname, sel in groups:
            s = d[sel(d)]
            ident = s[s.ok]
            row = dict(satz=label, gruppe=gname, personen=len(s), identifizierbar=len(ident))
            for thr in (0.05, 0.01):
                row[f"ident_p<{thr}"] = int((ident.p_all < thr).sum())
            row["ident_loo<=15"] = int((ident.loo_maxdev <= 15).sum())
            row["ident_loo<=60"] = int((ident.loo_maxdev <= 60).sum())
            row["ident_p<0.05_und_loo<=15"] = int(((ident.p_all < 0.05) & (ident.loo_maxdev <= 15)).sum())
            row["ident_p<0.05_und_loo<=15_und_abs_offset<=60"] = int(((ident.p_all < 0.05) & (ident.loo_maxdev <= 15) & (ident.offset.abs() <= 60)).sum())
            row["alle_p<0.05"] = int((s.p_all < 0.05).sum())
            out.append(row)
    groups = [("gesamt", lambda d: np.ones(len(d), bool)), ("train", lambda d: d.fold == "train"), ("validation", lambda d: d.fold == "validation"),
              ("n=3", lambda d: d.n == 3), ("n=4 bis 5", lambda d: (d.n >= 4) & (d.n <= 5)), ("n>=6", lambda d: d.n >= 6)]
    kennzahlen(df, "beobachtet", groups)
    # Nullmittel je Gruppe
    for gname, sel in groups:
        reps = []
        for k in range(K_NULL):
            s = nd[(nd.k == k)]
            s = s[sel(s)]
            ident = s[s.ok]
            reps.append(dict(personen=len(s), identifizierbar=len(ident), p05=int((ident.p_all < 0.05).sum()), p01=int((ident.p_all < 0.01).sum()),
                             loo15=int((ident.loo_maxdev <= 15).sum()), loo60=int((ident.loo_maxdev <= 60).sum()),
                             p05loo15=int(((ident.p_all < 0.05) & (ident.loo_maxdev <= 15)).sum()),
                             p05loo15off60=int(((ident.p_all < 0.05) & (ident.loo_maxdev <= 15) & (ident.offset.abs() <= 60)).sum()),
                             alle_p05=int((s.p_all < 0.05).sum())))
        r = pd.DataFrame(reps)
        row = dict(satz="Null (Permutation, Mittel ueber Pseudo-Datensaetze)", gruppe=gname, personen=float(r.personen.mean()), identifizierbar=float(r.identifizierbar.mean()))
        for key, lab in (("p05", "ident_p<0.05"), ("p01", "ident_p<0.01"), ("loo15", "ident_loo<=15"), ("loo60", "ident_loo<=60"),
                         ("p05loo15", "ident_p<0.05_und_loo<=15"), ("p05loo15off60", "ident_p<0.05_und_loo<=15_und_abs_offset<=60"), ("alle_p05", "alle_p<0.05")):
            row[lab] = float(r[key].mean())
            row[lab + "_q025"] = float(r[key].quantile(0.025))
            row[lab + "_q975"] = float(r[key].quantile(0.975))
        out.append(row)
    agg = pd.DataFrame(out)

    # BH-Anteil
    bh = bh_reject(df.p_all.to_numpy(), 0.05)
    df["bh"] = bh
    bh_ident = int((df.ok & df.bh).sum())
    # Verteilung der p-Werte der Identifizierbaren
    ident = df[df.ok]
    pbins = [0, 0.01, 0.05, 0.10, 0.25, 0.5, 1.0001]
    pdist = pd.cut(ident.p_all, pbins, right=False).value_counts().sort_index()
    nullp = nd[nd.ok]
    pdist_null = pd.cut(nullp.p_all, pbins, right=False).value_counts().sort_index() / K_NULL
    pd_tab = pd.DataFrame({"p_bereich": [str(x) for x in pdist.index], "identifizierbar_beobachtet": pdist.values, "identifizierbar_null_mittel": pdist_null.values})
    # Erreichbarkeit
    erreichbar = dict(m_eq_3=int((df.m == 3).sum()), m_ge_4=int((df.m >= 4).sum()), ident_m_eq_3=int(((df.m == 3) & df.ok).sum()), ident_m_ge_4=int(((df.m >= 4) & df.ok).sum()),
                      exakt_all=int(df.exact_all.sum()), min_p_exakt=float(df[df.exact_all].p_all.min()) if df.exact_all.any() else None)
    # Sensitivitaet p_comp
    sens = dict(ident_p_comp_lt05=int((ident.p_comp < 0.05).sum()), ident_p_all_lt05=int((ident.p_all < 0.05).sum()),
                korrelation_logp=float(np.corrcoef(np.log(df.p_all), np.log(df.p_comp))[0, 1]))
    # LOO Verteilung (Identifizierbare)
    loo_tab = pd.cut(ident.loo_maxdev, [-1, 0, 5, 15, 30, 60, 120, 1e9], labels=["0", "1-5", "6-15", "16-30", "31-60", "61-120", ">120"]).value_counts().sort_index()
    looN = nd[nd.ok & nd.loo_maxdev.notna()]
    loo_null = pd.cut(looN.loo_maxdev, [-1, 0, 5, 15, 30, 60, 120, 1e9], labels=["0", "1-5", "6-15", "16-30", "31-60", "61-120", ">120"]).value_counts().sort_index() / K_NULL
    loo_df = pd.DataFrame({"loo_maxdev_min": loo_tab.index.astype(str), "identifizierbar_beobachtet": loo_tab.values, "identifizierbar_null_mittel": loo_null.values})

    # Ausgabe
    sfx = "_TEST" if TEST else ""
    # Kleinzellen in Aggregat: nur ganze Zahlen maskieren
    aggm = agg.copy()
    for c in aggm.columns:
        if c in ("satz", "gruppe"):
            continue
        if aggm.loc[aggm.satz == "beobachtet", c].dtype.kind in "iu":
            aggm[c] = aggm[c].astype(object)
            aggm.loc[aggm.satz == "beobachtet", c] = [kz(v) for v in agg.loc[agg.satz == "beobachtet", c]]
    aggm.to_csv(PUB / f"S5_R6_kennzahlen{sfx}.csv", index=False, encoding="utf-8", float_format="%.4g")
    pd_tab.to_csv(PUB / f"S5_R6_pverteilung{sfx}.csv", index=False, encoding="utf-8", float_format="%.4g")
    loo_df.to_csv(PUB / f"S5_R6_loo{sfx}.csv", index=False, encoding="utf-8", float_format="%.4g")
    ext = dict(paritaet=par, bh_identifizierbar_q05=bh_ident, bh_alle=int(bh.sum()), erreichbar=erreichbar, sensitivitaet=sens, R_obs=R_OBS, R_null=R_NULL, K_null=K_NULL,
               kern_hash=kh, pkl_hash=pkl_hash, laufzeit_s=round(time.time() - t0, 1), spitzen_rss_gb=round(PEAK[0] / 2 ** 30, 3), icu_nenner=DENOM_ICU)
    (PUB / f"S5_R6_lauf{sfx}.json").write_text(json.dumps(ext, indent=1, default=str))
    print(json.dumps(ext, indent=1, default=str))
    print(aggm.to_string())
    print(pd_tab.to_string())
    print(loo_df.to_string())


if __name__ == "__main__":
    main()
