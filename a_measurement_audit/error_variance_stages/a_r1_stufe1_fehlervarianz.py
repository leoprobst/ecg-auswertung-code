"""AW-238 A-R1, Stufe 1: Fehlervarianzen fuer den Fehlerquotienten der Deming-Regression.

Bestimmt NUR die Innerpersonen-Varianzen der beiden Messgroessen aus kurzen Wiederholungspaaren (kein Zugriff auf die
gemeinsame Verteilung X-Y im Analyserahmen ausser zur Festlegung der Personenmenge):
  Y = V6-R-Zackenzeit (ms), Paare aus aufeinanderfolgenden EKG derselben Person (Entwicklung), Regeln wie im T2-Rahmen
      (kein Schrittmacher, kein LSB, QRS < 120 ms), Abstand <= 24 h (Sensitivitaet 6 h, 72 h).
  X = Hintere Wanddicke (mm), Paare aus TTE-Berichten derselben Person (Entwicklung), Abstand <= 7 d (Sensitivitaet 1 d, 30 d).
sigma_w^2 = Summe(d^2) / (2 n) (je Person das erste Paar), robust: (1.4826 * MAD(d))^2 / 2.
lambda0 = sigma_w,Y^2 / sigma_w,X^2 in ms^2/mm^2 (Konvention des Original-Codes: var(Fehler y)/var(Fehler x)).
Ausgabe nur Aggregate. Personenbezogene Daten bleiben unter <local path>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import glob, hashlib, json, os
import duckdb
import numpy as np
import pandas as pd

ROOT = str(data_path("a.mimic_validation"))
FEAT = ROOT + "/features/primary_core_sentinel_discovery_reserve_classical_criteria/development/split=*/part-*.parquet"
MASTER = ROOT + "/master/ecg_master_index.parquet"
REPORTS = ROOT + "/evidence/machine_report_labels.parquet"
ECHO = ROOT + "/evidence/echo_measurements_wide.parquet"
SPLITS = ROOT + "/master/subject_splits.parquet"
OUT = str(data_path("a.aw238"))
con = duckdb.connect()
con.execute("PRAGMA memory_limit='1500MB'")
con.execute("PRAGMA threads=2")

frame = pd.read_parquet(OUT + "/t2_rahmen_privat.parquet")
frame_ids = set(frame["subject_id"].tolist())

q = f"""
SELECT f.subject_id, f.study_id, f.qrs_duration_ms, f.v6_r_peak_time_ms, m.ecg_time,
       r.report_pacemaker_state, r.report_lbbb_state
FROM read_parquet('{FEAT}') f
JOIN read_parquet('{MASTER}') m USING (subject_id, study_id)
LEFT JOIN read_parquet('{REPORTS}') r USING (subject_id, study_id)
WHERE f.extraction_status = 'ok' AND f.split IN ('train', 'validation')
"""
e = con.execute(q).df()
pos = lambda s: s.astype("string").str.lower().eq("positive").fillna(False)
e = e.loc[~pos(e["report_pacemaker_state"]) & ~pos(e["report_lbbb_state"])]
e["v6"] = pd.to_numeric(e["v6_r_peak_time_ms"], errors="coerce")
e["qrs"] = pd.to_numeric(e["qrs_duration_ms"], errors="coerce")
e = e.loc[e["qrs"].lt(120) & e["v6"].notna()].sort_values(["subject_id", "ecg_time", "study_id"], kind="stable")
e["t"] = pd.to_datetime(e["ecg_time"])


def first_pairs(df, tcol, vcol, max_hours, min_hours=0.0):
    """Je Person das erste Paar aufeinanderfolgender Messungen mit min_hours < Abstand <= max_hours."""
    df = df.sort_values(["subject_id", tcol]).copy()
    g = df.groupby("subject_id")
    df["t_next"] = g[tcol].shift(-1)
    df["v_next"] = g[vcol].shift(-1)
    df["gap_h"] = (df["t_next"] - df[tcol]).dt.total_seconds() / 3600.0
    ok = df.loc[df["gap_h"].gt(min_hours) & df["gap_h"].le(max_hours) & df["v_next"].notna()]
    ok = ok.drop_duplicates("subject_id", keep="first")
    d = (ok["v_next"] - ok[vcol]).to_numpy(float)
    return ok["subject_id"].to_numpy(), d


def sig(d):
    d = np.asarray(d, float)
    n = len(d)
    classic = float(np.sum(d ** 2) / (2 * n)) if n else float("nan")
    mad = float(np.median(np.abs(d - np.median(d))))
    robust = float((1.4826 * mad) ** 2 / 2) if n else float("nan")
    return n, classic, robust, float(np.mean(d)) if n else float("nan")


res = {}
print("== Y: V6-R-Zackenzeit aus EKG-Paaren ==")
for hours in (6, 24, 72):
    for name, sel in (("alle Entwicklungspersonen", None), ("Personen des T2-Rahmens", frame_ids)):
        ids, d = first_pairs(e, "t", "v6", hours)
        if sel is not None:
            m = np.array([i in sel for i in ids])
            d = d[m]
        n, c, r, bias = sig(d)
        res[("Y", hours, name)] = (n, c, r, bias)
        print(f"  Abstand <= {hours:>2} h | {name:<26} | n Paare {n:>6} | sigma_w^2 klassisch {c:8.2f} ms^2 (SD {np.sqrt(c):5.2f}) | robust {r:8.2f} (SD {np.sqrt(r):5.2f}) | mittl. Differenz {bias:+.2f}")

print("== X: Hintere Wanddicke aus TTE-Paaren ==")
echo = con.execute(f"""
SELECT w.subject_id, w.echo_time, w.test_type, w.posterior_wall_thickness_cm AS pw
FROM read_parquet('{ECHO}') w
JOIN read_parquet('{SPLITS}') s USING (subject_id)
WHERE s.split IN ('train', 'validation') AND w.posterior_wall_thickness_cm IS NOT NULL
""").df()
print("  Echo-Zeilen mit Wanddicke:", len(echo), " test_type:", echo["test_type"].value_counts().head(5).to_dict())
echo["t"] = pd.to_datetime(echo["echo_time"])
echo["pw_mm"] = pd.to_numeric(echo["pw"], errors="coerce") * 10.0
echo = echo.dropna(subset=["pw_mm"])
echo = echo.loc[echo["pw_mm"].between(2, 40)]
print("  Spanne mm (Min, Max):", float(echo["pw_mm"].min()), float(echo["pw_mm"].max()))
for hours in (24, 168, 720):
    for name, sel in (("alle Entwicklungspersonen", None), ("Personen des T2-Rahmens", frame_ids)):
        ids, d = first_pairs(echo, "t", "pw_mm", hours, min_hours=0.0)
        if sel is not None:
            m = np.array([i in sel for i in ids])
            d = d[m]
        n, c, r, bias = sig(d)
        res[("X", hours, name)] = (n, c, r, bias)
        print(f"  Abstand <= {hours:>3} h ({hours/24:>4.1f} d) | {name:<26} | n Paare {n:>6} | sigma_w^2 klassisch {c:6.3f} mm^2 (SD {np.sqrt(c):5.3f}) | robust {r:6.3f} (SD {np.sqrt(r):5.3f}) | mittl. Differenz {bias:+.3f}")

# Aufloesungsvarianz der Dokumentation (0.1 cm = 1 mm Raster): Gleichverteilung -> 1/12 mm^2
print("Rundungsvarianz Wanddicke (1-mm-Raster): ", 1 / 12, "mm^2")
print("Rundungsvarianz V6-R-Zeit (2-ms-Raster): ", 4 / 12, "ms^2")

# lambda0 (Vorabregel): Y aus 24-h-Paaren der Personen des T2-Rahmens, X aus 7-Tage-Paaren der Personen des T2-Rahmens;
# Auffangregel, falls n < 200: alle Entwicklungspersonen
def pick(axis, hours):
    n_f = res[(axis, hours, "Personen des T2-Rahmens")][0]
    return (axis, hours, "Personen des T2-Rahmens") if n_f >= 200 else (axis, hours, "alle Entwicklungspersonen")
ky, kx = pick("Y", 24), pick("X", 168)
sy2, sx2 = res[ky][1], res[kx][1]
print("\nPrimaerquelle Y:", ky, "n", res[ky][0], "sigma_w^2", round(sy2, 3))
print("Primaerquelle X:", kx, "n", res[kx][0], "sigma_w^2", round(sx2, 4))
print(f"lambda0 (klassisch) = {sy2:.3f} / {sx2:.4f} = {sy2 / sx2:.2f} ms^2/mm^2")
for hy in (6, 72):
    for hx in (24, 720):
        a = res[("Y", hy, "Personen des T2-Rahmens")][1]
        b = res[("X", hx, "Personen des T2-Rahmens")][1]
        print(f"  Sensitivitaet Y<= {hy} h, X<= {hx/24:.0f} d: lambda = {a/b:.2f}")
ry = res[ky][2] / res[kx][2]
print(f"lambda robust (MAD) = {res[ky][2]:.3f} / {res[kx][2]:.4f} = {ry:.2f}")
# Reliabilitaeten im T2-Rahmen (nur aus Varianzen, nicht aus X-Y-Kovarianz)
vy, vx = float(frame["v6"].var(ddof=1)), float(frame["wall_mm"].var(ddof=1))
print(f"Gesamtvarianz im Rahmen: Y {vy:.2f} ms^2, X {vx:.3f} mm^2 -> Reliabilitaet 1 - sigma_w^2/Var: Y {1 - sy2 / vy:.3f}, X {1 - sx2 / vx:.3f}")
def _d(axis_key, df, tcol, vcol, hours):
    ids, d = first_pairs(df, tcol, vcol, hours)
    m = np.array([i in frame_ids for i in ids])
    return d[m]
np.savez(OUT + "/stufe1_paardifferenzen_privat.npz", dy=_d("Y", e, "t", "v6", 24), dx=_d("X", echo, "t", "pw_mm", 168))
json.dump({"lambda0": sy2 / sx2, "sigma2_y": sy2, "sigma2_x": sx2, "quelle_y": list(ky), "quelle_x": list(kx),
           "n_y": res[ky][0], "n_x": res[kx][0], "lambda_robust": ry,
           "raw": {f"{k[0]}|{k[1]}|{k[2]}": list(v) for k, v in res.items()}},
          open(OUT + "/stufe1_fehlervarianz.json", "w"), indent=1)
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
