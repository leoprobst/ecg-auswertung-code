"""AW-244 B-R9 (nachtraeglich, post hoc): Code-Negativkontrollen getrennt.

Frage (Zweitmeinung B, Sollte 4): Kontrollen des Kardiomyopathie-Codeendpunkts (ICD-9 425, ICD-10 I42/I43) sind im
Manuskript alle Aufenthalte ohne passenden Code; ein fehlender Diagnosebefund wurde als "Code abwesend" behandelt.
Hier werden Kontrollen mit dokumentierter Diagnoseliste (mindestens eine Zeile in diagnoses_icd) von Kontrollen ohne
jede dokumentierte Diagnose getrennt.

Daten: nur Entwicklung (train, validation); Eingabe BEF-0049_entwicklung.parquet (AW-131), Aufnahme-Zuordnung aus
master_dev_privat.parquet, Diagnosezeilen aus MIMIC-IV 3.1 hosp/diagnoses_icd (lokal). Kein Testzugriff.
Modell: festes B3-Design der Arbeit (Alter natuerliche Basis, Geschlecht, HF, QRS, QTcF, Winkel, Marker
qrs_t_frechet_beat_median), Skalierung aus den vollstaendigen Entwicklungsfaellen (Gate: Marker-SD 0.65940801).
Analysen: A0 alle Kontrollen; A1 nur Kontrollen mit dokumentierter Diagnoseliste; A2 nur Kontrollen mit mindestens
T1 Diagnosen (T1 = 33. Perzentil der Diagnosezahl unter Kontrollen; Dokumentationsintensitaet).
Bootstrap: Personen (eine ECG je Person) mit Zeilenneuziehung, 1.000 Ziehungen, Seed SeedSequence([244090, k]),
Intervall bei mindestens 950 gueltigen Fits.
Ausgabe: nur Aggregate (nachrechnung/AW244/b_r9_*.csv/.txt), keine Kennungen.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

sys.path.insert(0, str(h244.AUSW))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MARKER = "qrs_t_frechet_beat_median"
B = 1000
COLS = ["subject_id", "study_id", "split", "label", "age", "sex_numeric", "heart_rate", "qrs_duration_ms",
        "qtc_fridericia_ms", "spatial_qrs_t_angle_deg", MARKER]


def partition(df, counts):
    """df: Spalten hadm_id; counts: hadm_id, n (Diagnosezeilen). Rueckgabe df mit ndx (NaN = keine Diagnose)."""
    x = df.merge(counts[["hadm_id", "n"]], on="hadm_id", how="left", validate="many_to_one")
    x["ndx"] = x["n"].fillna(0).astype(int)
    return x


def fit_models(x, label="label"):
    from werkzeuge.b_test.statistik import raw_design, scaling, fixed_design, fit
    return raw_design, scaling, fixed_design, fit


def analyse(x, params, mask, name, rng_seed_base, nboot=B, log=print):
    from werkzeuge.b_test.statistik import raw_design, fixed_design, fit
    from werkzeuge.nachrechnung.core import logistic_fit
    from scipy.special import expit
    sub = x[mask]
    xx, names = raw_design(sub, MARKER)
    z = fixed_design(xx, params)
    y = sub["label"].to_numpy(float)
    out = fit(z, y, sub["subject_id"].to_numpy())
    row = {"analyse": name, "n": int(len(y)), "cases": int(y.sum()), "controls": int((y == 0).sum()),
           "or_per_sd": out["or"], "wald_lo": out["or_ci95_wald"][0], "wald_hi": out["or_ci95_wald"][1],
           "delta_auc": out["delta_auc"], "auc_base": out["auc_G"], "auc_marker": out["auc_G_marker"]}
    boots = []
    rng = np.random.default_rng(np.random.SeedSequence([rng_seed_base, 0]))
    n = len(y)
    for k in range(nboot):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        try:
            b = logistic_fit(z, y, c)
            sc = np.max(np.abs(z.T @ (c * (expit(z @ b) - y)) + np.r_[0, np.full(z.shape[1] - 1, 1e-6)] * b)) / c.sum()
            if np.isfinite(b).all() and sc < 1e-7:
                boots.append(float(np.exp(b[-1])))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass
        if k % 100 == 0:
            h244.guard("b_r9")
    row["boot_valid"] = len(boots)
    row["boot_lo"], row["boot_hi"] = h244.ci(boots, minn=int(0.95 * nboot))
    return row


def synthetic(n=12000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "subject_id": np.arange(n), "study_id": np.arange(n) + 10 ** 6, "split": np.where(rng.random(n) < 0.75, "train", "validation"),
        "age": rng.uniform(20, 90, n), "sex_numeric": rng.integers(0, 2, n).astype(float),
        "heart_rate": rng.normal(80, 20, n), "qrs_duration_ms": rng.normal(100, 20, n),
        "qtc_fridericia_ms": rng.normal(430, 35, n), "spatial_qrs_t_angle_deg": rng.uniform(5, 170, n),
        MARKER: rng.normal(1.0, 0.65, n)})
    lp = -3.0 + np.log(2.0) * (df[MARKER] - 1.0) / 0.65
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    df["hadm_id"] = np.arange(n) + 5 * 10 ** 6
    cnt = pd.DataFrame({"hadm_id": df.hadm_id.to_numpy(), "n": rng.integers(1, 25, n)})
    drop = rng.choice(n, 300, replace=False)
    cnt = cnt.drop(index=drop)          # 300 Aufenthalte ohne Diagnosezeile
    return df, cnt, set(df.hadm_id.to_numpy()[drop])


def test():
    h244.start_watch()
    df, cnt, dropped = synthetic()
    x = partition(df, cnt)
    miss = x.loc[x.ndx == 0, "hadm_id"]
    assert set(miss) == dropped and len(miss) == 300, "Partition: Aufenthalte ohne Diagnose falsch erkannt"
    assert (x.loc[x.ndx > 0, "ndx"].between(1, 24)).all()
    from werkzeuge.b_test.statistik import raw_design, scaling
    xx, names = raw_design(df, MARKER)
    params = scaling(xx, names)
    allc = np.ones(len(x), bool)
    r0 = analyse(x, params, allc, "A0", 1, nboot=60)
    # bekannter Effekt: log(2) je SD (SD der Stichprobe ~0.65) -> OR nahe 2
    assert 1.7 < r0["or_per_sd"] < 2.4, r0
    assert r0["boot_valid"] >= 55
    mask_a = ((x.label == 1) | (x.ndx > 0)).to_numpy()
    r1 = analyse(x, params, mask_a, "A1", 2, nboot=60)
    assert r1["n"] == int(mask_a.sum()) and r1["controls"] == int(((x.label == 0) & (x.ndx > 0)).sum())
    print("TEST_OK", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r0.items()})


def run():
    import duckdb
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("b_r9")
    h244.start_watch()
    src = Path(str(data_path("b.aw131_bef_0049_entwicklung_parquet")))
    master = h244.MA / "master_dev_privat.parquet"
    dxcsv = Path(str(data_path("b.mimic_iv_3_1_hosp_diagnoses_icd_csv_gz")))
    inputs = {str(p): h244.sha(p) for p in (src, master)}
    df = pd.read_parquet(src, columns=COLS)
    assert set(df.split) == {"train", "validation"}
    m = pd.read_parquet(master, columns=["study_id", "hadm_id", "split"])
    assert set(m.split) <= {"train", "validation"}
    df = df.merge(m[["study_id", "hadm_id"]], on="study_id", how="left", validate="one_to_one")
    assert df.hadm_id.notna().all()
    con = duckdb.connect()
    con.execute("set memory_limit='1GB'")
    con.execute('set threads=1')
    cnt = con.execute("select hadm_id, count(*) as n from read_csv_auto('" + str(dxcsv).replace(chr(92), '/') + "') group by hadm_id").df()
    x = partition(df, cnt)
    from werkzeuge.b_test.statistik import raw_design, scaling
    complete = np.isfinite(raw_design(x, MARKER)[0]).all(1)
    x = x[complete].reset_index(drop=True)
    xx, names = raw_design(x, MARKER)
    params = scaling(xx, names)
    sd_marker = params["sd"][-1]
    assert abs(sd_marker - 0.65940801) < 5e-4, f"Skalierungsgate: SD {sd_marker}"
    ctrl = x.label == 0
    nd = x.loc[ctrl, "ndx"]
    t1 = float(np.percentile(nd[nd > 0], 33))
    cap = {
        "n_complete": int(len(x)), "cases": int(x.label.sum()), "controls": int(ctrl.sum()),
        "controls_without_diagnosis_rows": int((ctrl & (x.ndx == 0)).sum()),
        "cases_without_diagnosis_rows": int(((x.label == 1) & (x.ndx == 0)).sum()),
        "marker_sd": sd_marker, "diagnosis_count_controls_median": float(nd.median()),
        "diagnosis_count_controls_p33": t1, "diagnosis_count_controls_p25": float(np.percentile(nd, 25)),
    }
    allc = np.ones(len(x), bool)
    mask_a = ((x.label == 1) | (x.ndx > 0)).to_numpy()
    mask_q = ((x.label == 1) | (x.ndx >= t1)).to_numpy()
    rows = []
    for name, mask, seed in (("A0_alle_Kontrollen", allc, 244091), ("A1_nur_dokumentierte_Diagnoseliste", mask_a, 244092),
                             ("A2_Kontrollen_mit_mindestens_T1_Diagnosen", mask_q, 244093)):
        r = analyse(x, params, mask, name, seed)
        print(r, flush=True)
        rows.append(r)
    # Kleinzellenregel: Zaehler 1-9 -> "<10"
    cap_out = {k: (h244.small(v) if k.startswith(("controls_without", "cases_without")) else v) for k, v in cap.items()}
    out = pd.DataFrame(rows)
    out.to_csv(h244.PUB / "b_r9_ergebnisse.csv", index=False)
    h244.dump(h244.PUB / "b_r9_zaehlung.json", cap_out)
    lines = ["B-R9 Code-Negativkontrollen getrennt (post hoc, nur Entwicklung)", f"Skript-SHA-256 {shash}",
             "Eingaben " + str(inputs), "Zaehlung " + str(cap_out), out.to_string(index=False)]
    (h244.PUB / "B_R9_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
