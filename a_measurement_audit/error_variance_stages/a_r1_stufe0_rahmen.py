"""AW-238 A-R1, Stufe 0: Analyserahmen der These T2 (i) aus dem gebundenen Original-Code nachbauen
und die Algebra der standardisierten Deming-Steigung kontrollieren (SD(Y)/SD(X)).

Liest ausschliesslich Entwicklungspartitionen (train, validation) lokal (<local path>).
Schreibt den personenbezogenen Rahmen NUR nach <local path> (privat, nicht <cloud folder>).
Ausgabe: nur Aggregate (n, SD, Verhaeltnis).
Nachbau der Auswahl aus <local path>
(Zeilen 245-316) und thesis_data.py::ThesisData.base/context.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import glob, hashlib, json, os, sys
import duckdb
import numpy as np
import pandas as pd

ROOT = str(data_path("a.mimic_validation"))
FEAT = ROOT + "/features/primary_core_sentinel_discovery_reserve_classical_criteria/development/split=*/part-*.parquet"
MASTER = ROOT + "/master/ecg_master_index.parquet"
REPORTS = ROOT + "/evidence/machine_report_labels.parquet"
CONTEXT = ROOT + "/evidence/clinical_context_features.parquet"
OUT = str(data_path("a.aw238"))
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA memory_limit='1500MB'")
con.execute("PRAGMA threads=2")
files = sorted(glob.glob(FEAT))
print("Merkmalsdateien:", len(files))
for probe in (files[0], files[-1]):
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{probe}')").fetchall()}
    print("Spalten vorhanden (", os.path.basename(os.path.dirname(probe)), "):", all(c in cols for c in ("v6_r_peak_time_ms", "qrs_duration_ms", "extraction_status")))

q = f"""
SELECT f.subject_id, f.study_id, f.split, f.qrs_duration_ms, f.v6_r_peak_time_ms,
       m.ecg_time, r.report_pacemaker_state, r.report_lbbb_state
FROM read_parquet('{FEAT}') f
JOIN read_parquet('{MASTER}') m USING (subject_id, study_id)
LEFT JOIN read_parquet('{REPORTS}') r USING (subject_id, study_id)
WHERE f.extraction_status = 'ok' AND f.split IN ('train', 'validation')
"""
base = con.execute(q).df()
print("Basis-EKG (extraction ok, Entwicklung):", len(base))


def positive(s):
    return s.astype("string").str.lower().eq("positive").fillna(False)


keep = ~positive(base["report_pacemaker_state"]) & ~positive(base["report_lbbb_state"])
base = base.loc[keep]
ctx = con.execute(f"""SELECT subject_id, study_id, has_tte_30d, posterior_wall_thickness_cm, history_old_mi_before_ecg
FROM read_parquet('{CONTEXT}') WHERE split IN ('train','validation')""").df()


def flag(s):
    return s.astype("boolean").fillna(False).astype(bool)


frame = base.merge(ctx, on=["subject_id", "study_id"], how="inner")
qrs = pd.to_numeric(frame["qrs_duration_ms"], errors="coerce")
frame = frame.loc[flag(frame["has_tte_30d"]) & ~flag(frame["history_old_mi_before_ecg"]) & (qrs < 120)]
frame = frame.sort_values(["subject_id", "ecg_time", "study_id"], kind="stable").drop_duplicates("subject_id", keep="first")
frame["wall_mm"] = pd.to_numeric(frame["posterior_wall_thickness_cm"], errors="coerce") * 10.0
frame["v6"] = pd.to_numeric(frame["v6_r_peak_time_ms"], errors="coerce")
n_before = len(frame)
frame = frame.dropna(subset=["wall_mm", "v6"])
print("Personen vor dropna:", n_before, " Analyserahmen:", len(frame), "(Original 27.810)")
print("nach Split:", frame["split"].value_counts().to_dict())
x, y = frame["wall_mm"].to_numpy(float), frame["v6"].to_numpy(float)
sx, sy = x.std(ddof=1), y.std(ddof=1)
print(f"SD(X) = {sx:.5f} mm   SD(Y) = {sy:.5f} ms   SD(Y)/SD(X) = {sy/sx:.5f} ms/mm   (Original: -3.347 = -SD(Y)/SD(X)?)")
for sp in ("train", "validation"):
    f = frame.loc[frame["split"] == sp]
    print(sp, len(f), f"SD(Y)/SD(X) = {f['v6'].std(ddof=1)/f['wall_mm'].std(ddof=1):.5f}")
print("Aufloesung wall_mm (Anzahl verschiedener Werte, kleinste Schrittweite):", int(np.unique(x).size), float(np.min(np.diff(np.unique(x)))))
print("Aufloesung v6 (kleinste Schrittweite):", float(np.min(np.diff(np.unique(y)))))
frame[["subject_id", "split", "wall_mm", "v6", "qrs_duration_ms"]].to_parquet(OUT + "/t2_rahmen_privat.parquet", index=False)
print("geschrieben:", OUT + "/t2_rahmen_privat.parquet")
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
