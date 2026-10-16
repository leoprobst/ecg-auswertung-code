"""AW-241 (Sollte 3, Zweitmeinung v11): Welche Kovariaten aendert die T-Ende-Rekonstruktion (S4/S17) wirklich?
Entwicklungsdaten (train/validation, Stichprobe von AW-192), Produktionskern wie aw192_run.signals. Je Aufzeichnung werden fuer Verschiebungen
des T-Endes aller Schlaege (0, +-16, +-32 ms) die Produktionsmerkmale qrs_duration_ms, qt_interval_ms, qtc_fridericia_ms, heart_rate_bpm
sowie der Winkel und der normierte Fréchet-Marker aus demselben Kontext berechnet. Ausgabe nur Aggregate (keine Einzelwerte).
Aufruf: python tende_kovariaten.py [N]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

OUT = Path(str(data_path("b.aw241_tende")))
OUT.mkdir(parents=True, exist_ok=True)
AGG = ROOT / "nachrechnung/AW241"
OLD = Path(str(data_path("b.aw166")))
S192 = Path(str(data_path("b.aw192")))
SHIFTS = [-32, -16, 0, 16, 32]
FEATS = ["qrs_duration_ms", "qt_interval_ms", "qtc_fridericia_ms", "heart_rate_bpm", "qrs_t_normalized_discrete_frechet", "spatial_qrs_t_angle_deg"]


def guard():
    import ctypes
    import psutil

    class M(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ["total", "free", "page", "commit", "vtotal", "vfree", "extended"]]

    m = M()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    if min(m.free, m.commit) < 1.5 * 1024 ** 3:
        raise RuntimeError("WAIT_MEMORY")
    if psutil.Process().memory_info().rss > 1.3 * 1024 ** 3:
        raise RuntimeError("RSS_LIMIT")


def main(n):
    source = (ROOT / "werkzeuge/b_test/aw166_signals.py").read_text(encoding="utf8")
    body = source[source.index("    from werkzeuge.varianten.adapter"):source.index("    f=pd.read_parquet")]
    scope = {"np": np}
    exec("def kernel():\n" + body + "    return compute,compute_full,prepare_record,context,shifted,OPTIONS,verified_release_import,fr\n", scope)
    compute, full, prepare, context, shifted, options, verify, fr = scope["kernel"]()
    import werkzeuge.varianten.adapter as adapter
    adapter.memory_guard = guard
    f = ds.dataset(str(S192 / "sample_private.parquet")).to_table().to_pandas()
    assert set(f.split) <= {"train", "validation"}
    rng = np.random.default_rng(24120261016)
    pick = np.sort(rng.choice(len(f), n, replace=False))
    f = f.iloc[pick]
    meta = ds.dataset(str(OLD / "meta_private.parquet")).to_table(filter=ds.field("study_id").isin(f.study_id.tolist())).to_pandas()
    f = meta.merge(f[["study_id", "y"]], on="study_id", validate="one_to_one")
    rows = []
    t0 = time.time()
    for i, row in enumerate(f.to_dict("records")):
        guard()
        try:
            p = prepare(row, options)
            rec = {"y": int(row["y"])}
            for ms in SHIFTS:
                d = p["delineation"] if ms == 0 else shifted(p["delineation"], "t_offset", ms // 2)
                v = full(p, d, FEATS)
                rec[str(ms)] = [None if x is None else float(getattr(x, "value", x)) if np.isfinite(float(getattr(x, "value", x))) else None for x in v]
            rows.append(rec)
        except Exception as ex:  # noqa: BLE001
            rows.append({"y": int(row["y"]), "error": str(ex)[:80]})
        if i % 10 == 0:
            print(i, round(time.time() - t0), flush=True)
            (OUT / "records.json").write_text(json.dumps(rows), encoding="utf8")
    (OUT / "records.json").write_text(json.dumps(rows), encoding="utf8")
    # Aggregation
    ok = [r for r in rows if "error" not in r]
    lines = [f"AW-241 T-Ende-Kovarianz (Entwicklung, Stichprobe von AW-192, {len(ok)} von {len(rows)} Aufzeichnungen ausgewertet; Skripthash "+__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()+")"]
    out = {}
    for ms in SHIFTS:
        if ms == 0:
            continue
        d = {}
        for j, name in enumerate(FEATS):
            base = np.array([np.nan if r["0"][j] is None else r["0"][j] for r in ok], float)
            new = np.array([np.nan if r[str(ms)][j] is None else r[str(ms)][j] for r in ok], float)
            both = np.isfinite(base) & np.isfinite(new)
            diff = new[both] - base[both]
            d[name] = dict(n_beide_endlich=int(both.sum()), n_basis_nur=int((np.isfinite(base) & ~np.isfinite(new)).sum()), anteil_geaendert=float(np.mean(np.abs(diff) > 1e-9)) if both.sum() else None,
                           mittel=float(diff.mean()) if both.sum() else None, median=float(np.median(diff)) if both.sum() else None, abs_max=float(np.abs(diff).max()) if both.sum() else None)
        out[str(ms)] = d
        lines.append(f"Verschiebung {ms} ms: " + json.dumps(d))
    (AGG / "TENDE_KOVARIATEN_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
