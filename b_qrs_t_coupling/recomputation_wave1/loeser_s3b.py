"""AW-241 Aufgabe 2: Loeserkontrolle (200 statt 60 Newton-Schritte, gleiches Score-Kriterium) fuer die S3b-Zeilen
006, 015, 021, 025, 027, 029, 046, 062 (Bootstrap im Originallauf AW-142 nicht schaetzbar wegen Konvergenzausfaellen).
Nachtraeglich (post hoc). Vorgehen wie AW-238 (Zeilen 045, 047, 049), Originalskripte unveraendert.

Aufrufe:
  python loeser_s3b.py test            synthetische Funktionspruefung (kein Datenzugriff)
  python loeser_s3b.py run [--reps N]  Designs aus den Testprojektionen (interne Assessmentpartition) bauen, Punkt-Gate gegen AW-142,
                                       je Zeile 1000 Ziehungen mit (a) Originalregel+Originalloeser, (b) Loeserkontrolle (AW-171-Kern,
                                       200 Schritte, Eignung nur an der Stichprobe). Checkpoint alle 50 Ziehungen.
  python loeser_s3b.py report          Aggregate nach nachrechnung/AW241/LOESER_S3B_ausgabe.txt und loeser_s3b_ergebnisse.csv
Rezept der Ziehungen: werkzeuge/b_test/statistik.py::draws (SeedSequence([1282026, r]), Union der Indexpersonen H1/H4/RV).
Private Zwischenergebnisse (Personenkennungen im Design): <local path> ; in <cloud folder> nur Aggregate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = str(data_path("b.auswertung"))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/werkzeuge/aw238")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(str(data_path("b.aw241_loeser")))
AGG = Path(ROOT + "/nachrechnung/AW241")
AW142 = Path(str(data_path("b.aw136_b_testlauf_aw142_ergebnis_json")))
MANIFEST = Path(str(data_path("b.b_freigabe_input_manifest_v5_json")))
PARAMS = Path(ROOT + "/werkzeuge/b_test/parameter_entwicklung.json")
EXPECT = {
    "aw142": "f17ef84b6533f71d3d7ed0099e81ed53b684e4e45f8539e3e0a17d0a0e3301a7",
    "manifest": "5bfa03ae40fb812f6d037b618c8e7b5a76512b656a5bf3ab1f02265a8223b49e",
    "params": "a79d25e174008791f9533f2eeac890c8c1e04cbf956bb908d966f90017dd2b1b",
    "u142": "233f2d8ef71b5354e4dc7f9c6d29121abe8f5e41dfa9723fc10cb43599d8ad5e",
}
PLANNED = 1000
MIN_VALID = 950
ROWS = ["006", "015", "021", "025", "027", "029", "046", "062"]
LABEL = {
    "006": "RV-Hypothese BEF-0087, Hauptmodell",
    "015": "RV BEF-0087, B3 auf identischem B4-Set (Rang 1)",
    "021": "LVEF<=40 gegen LVH positiv/LVEF>=50 (Rang 4, H1)",
    "025": "H1, QRS<120 ms ohne Block (Rang 6a)",
    "027": "H1, Alter<65 (Rang 6c)",
    "029": "H1, Geschlecht F (Rang 6d)",
    "046": "H4 Kardiomyopathie, Sinusselektor (Rang 6b)",
    "062": "RV BEF-0087, Geschlecht M (Rang 6d)",
}
PEAK = [0]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean(v):
    if isinstance(v, dict):
        return {str(a): clean(b) for a, b in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(a) for a in v]
    if hasattr(v, "tolist"):
        return clean(v.tolist())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def dump(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf8")


def free_mem_ok():
    import ctypes

    class Mem(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k_, ctypes.c_ulonglong) for k_ in ("total", "free", "page", "commit", "vtotal", "vfree", "extended")]

    m = Mem()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.free, m.commit) > 1.5 * 1024 ** 3


def ram_guard():
    while not free_mem_ok():
        time.sleep(30)


def start_watch(limit_gb=1.4):
    import psutil

    def watch():
        while True:
            PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss)
            if PEAK[0] > limit_gb * 1024 ** 3:
                os._exit(75)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


# ------------------------------------------------------------------ Bootstrap je Zeile

def replicate(design, y, ids, union, r, st, k, bk):
    w = st.draws(union, ids, r)
    rec = dict(r=r)
    try:  # (a) Originalregel und Originalloeser (60 Schritte), wie im Originallauf
        res = st.fit(design, y, ids, weights=w, wald=False)
        rec.update(old_ok=True, old=dict(beta=float(res["beta"]), delta_auc=float(res["delta_auc"])))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        rec.update(old_ok=False, old_reason=str(exc))
    try:  # (b) Loeserkontrolle: AW-171-Kern, 200 Schritte, gleiches Score-Kriterium, Eignung nur an der Stichprobe
        with bk.replicate_rule():
            v = k.association(design, y, w)
        rec.update(v_ok=True, v=dict(beta=float(v["beta"]), delta_auc=float(v["delta_auc"])))
    except Exception as exc:  # noqa: BLE001
        rec.update(v_ok=False, v_reason=type(exc).__name__ + ":" + str(exc))
    return rec


def bootstrap_row(name, design, y, ids, union, reps, st, k, bk, target_dir, log=True):
    target = target_dir / f"row_{name}.json"
    state = json.loads(target.read_text(encoding="utf8")) if target.exists() else dict(row=name, next=0, reps=[])
    for r in range(state["next"], reps):
        if r % 25 == 0:
            ram_guard()
        state["reps"].append(replicate(design, y, ids, union, r, st, k, bk))
        state["next"] = r + 1
        if (r + 1) % 50 == 0 or r + 1 == reps:
            dump(target, dict(state, utc=now()))
            if log:
                print(json.dumps(dict(row=name, done=r + 1, utc=now())), flush=True)
    return state


# ------------------------------------------------------------------ Test mit Kunstdaten

def synthetic_test():
    import tempfile

    from werkzeuge.b_test import aw171_kernels as k
    from werkzeuge.b_test import statistik as st
    import b_r1_kern as bk

    rng = np.random.default_rng(5)
    n = 3000
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 8))])
    beta = np.array([-2.6, .3, .1, 0, .2, .1, 0, .2, .5])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x @ beta)))).astype(float)
    ids = np.arange(1000, 1000 + n)
    union = np.unique(np.concatenate([ids, np.arange(5000, 5400)]))
    d = Path(tempfile.mkdtemp(prefix="aw241_syn_"))
    s = bootstrap_row("syn", x, y, ids, union, 30, st, k, bk, d, log=False)
    ok_old = [r for r in s["reps"] if r["old_ok"]]
    ok_v = [r for r in s["reps"] if r["v_ok"]]
    both = [r for r in s["reps"] if r["old_ok"] and r["v_ok"]]
    diff = max(abs(r["old"]["beta"] - r["v"]["beta"]) for r in both) if both else None
    print("synthetisch: alt gueltig", len(ok_old), "von", len(s["reps"]), "| Kontrolle gueltig", len(ok_v), "| max |beta alt - beta Kontrolle|", diff)
    pt = k.association(x, y, np.ones(n))
    ref = st.fit(x, y, ids, wald=False)
    print("Punkt: Kontrolle beta", pt["beta"], "Original beta", ref["beta"], "delta_auc", pt["delta_auc"], ref["delta_auc"])
    assert abs(pt["beta"] - ref["beta"]) < 1e-6 and abs(pt["delta_auc"] - ref["delta_auc"]) < 1e-9
    assert len(ok_v) >= len(ok_old)
    print("TEST OK", d)


# ------------------------------------------------------------------ Designs aus den Testprojektionen

def load_frames():
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from werkzeuge.b_test.daten import select_index

    for key, p in (("aw142", AW142), ("manifest", MANIFEST), ("params", PARAMS)):
        if sha(p) != EXPECT[key]:
            raise RuntimeError("Hashabweichung " + key)
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    package = json.loads(PARAMS.read_text(encoding="utf8"))
    frames = {}
    for bef in ("BEF-0063", "BEF-0049", "BEF-0087"):
        src = manifest["hypotheses"][bef]
        p = Path(src["path"])
        if any(s in str(p).lower() for s in ("locked_", "diagnostics", "publication_preparation")):
            raise PermissionError("gesperrter Pfad: " + str(p))
        if sha(p) != src["sha256"]:
            raise RuntimeError("Hashabweichung " + bef)
        cols = src["columns"]
        if set(cols) - set(pq.ParquetFile(p).schema_arrow.names):
            raise ValueError("Schema")
        f = ds.dataset(str(p), format="parquet").to_table(columns=cols).to_pandas()
        if set(f.split) != {"test"}:
            raise ValueError("Partition nicht 'test' (interne Assessmentpartition)")
        frames[bef] = select_index(f, package["hypotheses"][bef]["endpoint"])
        del f
    return frames, package


CAP = {}


def capture_designs(frames, package):
    from werkzeuge.b_test import adapter, daten

    orig = adapter.fit

    def cap_fit(z, y, subjects, direction=1, weights=None, wald=True):
        CAP.update(z=np.asarray(z, float), y=np.asarray(y, float), ids=np.asarray(subjects))
        return orig(z, y, subjects, direction, weights=weights, wald=wald)

    adapter.fit = cap_fit
    H = package["hypotheses"]
    f63, f87, f49 = frames["BEF-0063"], frames["BEF-0087"], frames["BEF-0049"]
    union = np.unique(np.concatenate([f.subject_id.to_numpy() for f in (f63, f49, f87)]))
    if hashlib.sha256(union.tobytes()).hexdigest() != EXPECT["u142"]:
        raise RuntimeError("Union U142 stimmt nicht mit AW-238 ueberein")
    out = {}

    def run(name, fn):
        CAP.clear()
        res = fn()
        if res.get("status") != "schaetzbar":
            raise RuntimeError(f"Zeile {name} nicht schaetzbar: {res.get('reason')}")
        out[name] = dict(res=res, z=CAP["z"], y=CAP["y"], ids=CAP["ids"])

    e87, h1, e49 = H["BEF-0087"], H["BEF-0063"], H["BEF-0049"]
    run("006", lambda: adapter.estimate(f87, e87, union, bootstrap=False))
    _, good, _ = daten.masks(f87, e87["endpoint"], e87["marker"], e87["comparator"], strong=True)
    run("015", lambda: adapter.estimate(f87, e87, union, lambda z: good, bootstrap=False))
    run("021", lambda: adapter.estimate(f63, h1, union, adapter.lvh_selector, inclusive=True, bootstrap=False))
    sel6a = lambda z: z.qrs_duration_ms.lt(120) & z.classical_lbbb_pattern_positive.eq(False) & z.classical_rbbb_pattern_positive.eq(False)  # noqa: E731
    run("025", lambda: adapter.estimate(f63, h1, union, sel6a, bootstrap=False))
    run("027", lambda: adapter.estimate(f63, h1, union, lambda z: z.age.lt(65), bootstrap=False))
    run("029", lambda: adapter.estimate(f63, h1, union, lambda z: z.sex.eq("F"), bootstrap=False, sex_stratum="F"))
    run("046", lambda: adapter.estimate(f49, e49, union, lambda z: daten.truth(z.sinus_selector), bootstrap=False))
    run("062", lambda: adapter.estimate(f87, e87, union, lambda z: z.sex.eq("M"), bootstrap=False, sex_stratum="M"))
    adapter.fit = orig
    return union, out


def expected_row(name):
    d = json.loads(AW142.read_text(encoding="utf8"))
    s = d["sensitivities"]
    if name == "006":
        return d["results"]["BEF-0087"]
    if name == "015":
        e = [x for x in s if x["rank"] == "1" and x["analysis"].endswith("BEF-0087")][0]
        return e["B3_same_set"]
    want = {"021": ("4", None), "025": ("6a", "QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster"), "027": ("6c", "lt65"), "029": ("6d", "F"),
            "046": ("6b", "Sinusselektor nach AW054-Regex / BEF-0049"), "062": ("6d", "Geschlecht M / BEF-0087")}[name]
    cand = [x for x in s if x["rank"] == want[0] and (want[1] is None or x["analysis"] == want[1])]
    if len(cand) != 1:
        raise RuntimeError("Originalzeile nicht eindeutig: " + name)
    return cand[0]


def gate_point(name, res):
    e = expected_row(name)
    for key in ("beta", "or", "delta_auc", "auc_G", "auc_G_marker"):
        if not np.isclose(res[key], e[key], rtol=1e-9, atol=0):
            raise RuntimeError(f"Punktschaetzer weicht ab {name}: {key} {res[key]!r} {e[key]!r}")
    if res["N"] != e["N"] or res["class_persons"] != e["class_persons"]:
        raise RuntimeError("N/Klassenpersonen weichen ab " + name)
    return e


def run(reps):
    start_watch()
    from werkzeuge.b_test import aw171_kernels as k
    from werkzeuge.b_test import statistik as st
    import b_r1_kern as bk

    OUT.mkdir(parents=True, exist_ok=True)
    ram_guard()
    stamp = dict(script=sha(__file__), kern=sha(ROOT + "/werkzeuge/aw238/b_r1_kern.py"), utc=now())
    dump(OUT / "run_stamp.json", stamp)
    frames, package = load_frames()
    union, rows = capture_designs(frames, package)
    del frames
    meta = {}
    for name in ROWS:
        e = gate_point(name, rows[name]["res"])
        meta[name] = dict(N=int(rows[name]["res"]["N"]), class_persons=rows[name]["res"]["class_persons"], or_point=rows[name]["res"]["or"],
                          wald=rows[name]["res"]["or_ci95_wald"], delta_auc=rows[name]["res"]["delta_auc"],
                          old_valid=e["bootstrap"]["valid"], old_failed=e["bootstrap"]["failed"], gate="Punktschaetzer gleich AW-142 (rtol 1e-9)")
        print(json.dumps(dict(gate=name, N=meta[name]["N"], ok=True)), flush=True)
    dump(OUT / "row_meta.json", meta)
    print("PEAK_MB_nach_Designs", PEAK[0] // 2 ** 20, flush=True)
    for name in ROWS:
        d = rows[name]
        bootstrap_row(name, d["z"], d["y"], d["ids"], union, reps, st, k, bk, OUT)
    dump(OUT / "run_done.json", dict(utc=now(), peak_mb=PEAK[0] // 2 ** 20, reps=reps))


# ------------------------------------------------------------------ Bericht

def q(v, p=(0.025, 0.975)):
    return np.quantile(np.asarray(v, float), p, method="linear").tolist()


def report():
    meta = json.loads((OUT / "row_meta.json").read_text(encoding="utf8"))
    lines = ["AW-241 Aufgabe 2: Loeserkontrolle S3b (post hoc). Aggregate, keine Personenwerte.",
             f"Skripthash loeser_s3b.py: {sha(__file__)}", ""]
    rows_csv = []
    for name in ROWS:
        s = json.loads((OUT / f"row_{name}.json").read_text(encoding="utf8"))
        reps = s["reps"]
        m = meta[name]
        old_ok = [r for r in reps if r["old_ok"]]
        v_ok = [r for r in reps if r["v_ok"]]
        old_reasons = {}
        for r in reps:
            if not r["old_ok"]:
                old_reasons[r["old_reason"]] = old_reasons.get(r["old_reason"], 0) + 1
        v_reasons = {}
        for r in reps:
            if not r["v_ok"]:
                v_reasons[r["v_reason"]] = v_reasons.get(r["v_reason"], 0) + 1
        both = [r for r in reps if r["old_ok"] and r["v_ok"]]
        diff = max(abs(r["old"]["beta"] - r["v"]["beta"]) for r in both) if both else float("nan")
        gate_old = (len(old_ok) == m["old_valid"]) and (old_reasons == {k_: v for k_, v in m["old_failed"].items()})
        or_ci = np.exp(q([r["v"]["beta"] for r in v_ok])).tolist() if len(v_ok) >= MIN_VALID else None
        da_ci = q([r["v"]["delta_auc"] for r in v_ok]) if len(v_ok) >= MIN_VALID else None
        cases = m["class_persons"].get("1")
        controls = m["class_persons"].get("0")
        lines += [f"Zeile {name} ({LABEL[name]}): N {m['N']}, Faelle {cases}, Kontrollen {controls}; OR {m['or_point']:.3f} (Wald {m['wald'][0]:.3f} bis {m['wald'][1]:.3f}); Delta-AUC {m['delta_auc']:.4f}",
                  f"  alt (Originalloeser, 60 Schritte): gueltig {len(old_ok)}/1000, Ausfaelle {old_reasons}; Gate gegen veroeffentlichte Zahl (gueltig {m['old_valid']}, Ausfaelle {m['old_failed']}): {'erfuellt' if gate_old else 'NICHT erfuellt'}",
                  f"  neu (Loeserkontrolle, 200 Schritte): gueltig {len(v_ok)}/1000, Ausfaelle {v_reasons}",
                  "  OR-Bootstrap-Perzentilintervall: " + (f"[{or_ci[0]:.3f}; {or_ci[1]:.3f}]" if or_ci else "nicht schaetzbar (unter 950)") + "; Delta-AUC-Intervall: " + (f"[{da_ci[0]:.4f}; {da_ci[1]:.4f}]" if da_ci else "nicht schaetzbar"),
                  f"  groesste Abweichung der beta zwischen 60- und 200-Schritte-Fit auf {len(both)} gemeinsam gueltigen Ziehungen: {diff:.2e}", ""]
        rows_csv.append(dict(zeile=name, analyse=LABEL[name], N=m["N"], faelle=cases, kontrollen=controls, or_punkt=round(m["or_point"], 4),
                             or_wald_low=round(m["wald"][0], 4), or_wald_high=round(m["wald"][1], 4), delta_auc=round(m["delta_auc"], 5),
                             alt_gueltig=len(old_ok), alt_ausfaelle=json.dumps(old_reasons, ensure_ascii=False), gate_alt_reproduziert=gate_old,
                             neu_gueltig=len(v_ok), neu_ausfaelle=json.dumps(v_reasons, ensure_ascii=False),
                             or_boot_low=round(or_ci[0], 4) if or_ci else "n/a", or_boot_high=round(or_ci[1], 4) if or_ci else "n/a",
                             dauc_boot_low=round(da_ci[0], 5) if da_ci else "n/a", dauc_boot_high=round(da_ci[1], 5) if da_ci else "n/a",
                             max_abw_beta_60_200=f"{diff:.2e}"))
    AGG.mkdir(exist_ok=True)
    (AGG / "LOESER_S3B_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    with (AGG / "loeser_s3b_ergebnisse.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "run", "report"])
    ap.add_argument("--reps", type=int, default=PLANNED)
    a = ap.parse_args()
    if a.cmd == "test":
        synthetic_test()
    elif a.cmd == "run":
        run(a.reps)
    else:
        report()


if __name__ == "__main__":
    main()
