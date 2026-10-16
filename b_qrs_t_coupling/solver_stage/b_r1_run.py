"""AW-238 B-R1: korrigierte Bootstrap-Intervalle (Welle 0, Fehlerkorrektur), nachtraeglich (post hoc).

Aufrufe:
  python b_r1_run.py prepare                 Personenunionen U142 (Original AW-142) und U172 (AW-172) rekonstruieren und pruefen
  python b_r1_run.py task <name> [--reps N]  eine Analyse mit 1000 Ziehungen (Checkpoint alle 50)
  python b_r1_run.py mi                      Nenner der multiplen Imputation (keine Ziehungen)
  python b_r1_run.py synthetic               Funktionspruefung mit synthetischen Daten (kein Zugriff auf echte Daten)
Aufgaben:
  A_6a, A_6c_lt65, A_6d_F                   S3b-Zeilen 045, 047, 049 (Motor: Original statistik.fit, Union U142), dazu Loeserkontrolle (AW-171-Kern)
  S15_<H1|H4>_<no_qtc_extended|no_qtc_C|missing_indicator_extended|standardized|cross_F|cross_M>, S15_H1_W24_rbbb   (Motor: aw171_kernels, Union U172)
Alle Ziehungen verwenden das Originalrezept SeedSequence([1282026, r]), r = 0..999; kein neuer Zufall.
Private Zwischenergebnisse: <local path> ; in <cloud folder> nur Aggregate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
import b_r1_kern as bk  # noqa: E402
from b_r1_kern import k, st  # noqa: E402

OUT = Path(str(data_path("b.aw238")))
AW172 = Path(str(data_path("b.aw172")))
PARAMS171 = Path(str(data_path("b.aw171_parameters_json")))
PARAM_B = Path(bk.ROOT + "/werkzeuge/b_test/parameter_entwicklung.json")
AW142 = Path(str(data_path("b.aw136_b_testlauf_aw142_ergebnis_json")))
MANIFEST = Path(str(data_path("b.b_freigabe_input_manifest_v5_json")))
EXPECT = {
    "params171": "17784e0a21811d2f61e9def653d7b1a5d9dfbb8cd1bf2088b508da0aa0938b72",
    "aw142": "f17ef84b6533f71d3d7ed0099e81ed53b684e4e45f8539e3e0a17d0a0e3301a7",
    "manifest": "5bfa03ae40fb812f6d037b618c8e7b5a76512b656a5bf3ab1f02265a8223b49e",
    "u172": "e4494075d5d4a9e28fb6f683c551c6260d7f8e1687607212a86fa26cdd58d1a2",
}
F = "qrs_t_normalized_discrete_frechet"
M = "qrs_t_frechet_beat_median"
A = "spatial_qrs_t_angle_deg"
PLANNED = 1000
MIN_VALID = 950


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


def ram_guard():
    import ctypes

    class Mem(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k_, ctypes.c_ulonglong) for k_ in ("total", "free", "page", "commit", "vtotal", "vfree", "extended")]

    while True:
        m = Mem()
        m.length = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        if min(m.free, m.commit) > 1.5 * 1024 ** 3:
            return
        time.sleep(30)


# ------------------------------------------------------------------ Daten

def synthetic_ctx():
    rng = np.random.default_rng(11)
    n = 4600
    ids = np.arange(1, n + 1)

    def frame(shift):
        age = rng.integers(25, 90, n).astype(float)
        sex = rng.choice(["F", "M"], n)
        hr = rng.normal(78, 14, n)
        qrs = rng.normal(108, 18, n)
        qtc = rng.normal(430, 28, n)
        ang = rng.uniform(0, 180, n)
        mk = rng.gamma(4, 0.4, n)
        mk2 = mk * 0.8 + rng.gamma(2, 0.2, n)
        lin = -2.75 + 0.016 * (age - 60) + 0.02 * (qrs - 108) + 0.5 * (mk - 1.6) / 0.8
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(lin + shift)))).astype(float)
        d = pd.DataFrame(dict(subject_id=ids, age=age, sex=sex, sex_numeric=np.where(sex == "F", 0.0, 1.0), heart_rate=hr,
                              qrs_duration_ms=qrs, qtc_fridericia_ms=qtc, y=y,
                              classical_lbbb_pattern_positive=rng.uniform(size=n) < 0.02,
                              classical_rbbb_pattern_positive=rng.uniform(size=n) < 0.45))
        d[A] = ang
        d[F] = mk
        d[M] = mk2
        d.loc[rng.uniform(size=n) < 0.03, "qtc_fridericia_ms"] = np.nan
        d.loc[rng.uniform(size=n) < 0.02, "heart_rate"] = np.nan
        return d

    ctx = dict(C={}, D={}, u172=np.arange(1, n + 1), u142=np.arange(1, n + 1), params=dict(endpoints={}))
    for ep, marker, other in (("H1", F, M), ("H4", M, F)):
        D = frame(0.0 if ep == "H1" else 0.05)
        C = D.loc[np.isfinite(D[["age", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", A, F, M]].to_numpy(float)).all(1)].copy()
        ctx["D"][ep], ctx["C"][ep] = D, C
        markers = {}
        for mk in (F, M):
            raw, names = st.raw_design(C, mk)
            sc = st.scaling(raw, names)
            sc.update(status="gebunden")
            medians = np.quantile(C[mk], [0.125, 0.375, 0.625, 0.875]).tolist()
            markers[mk] = dict(scale=sc, quartiles=dict(medians=medians), models={})
        ctx["params"]["endpoints"][ep] = dict(markers=markers, mi=dict(start=dict(age=64.0, heart_rate=77.0, qrs_duration_ms=108.0, qtc_fridericia_ms=430.0, sex_numeric=1.0)))
    ctx["pkg_scaling_H4"] = ctx["params"]["endpoints"]["H4"]["markers"][M]["scale"]
    return ctx


def load_ctx(real=True):
    if not real:
        return synthetic_ctx()
    for key, path in (("params171", PARAMS171), ("aw142", AW142), ("manifest", MANIFEST)):
        if sha(path) != EXPECT[key]:
            raise RuntimeError("Hashabweichung " + key)
    ctx = dict(C={}, D={})
    for ep in ("H1", "H4"):
        ctx["C"][ep] = pd.read_parquet(AW172 / f"{ep}_C_private.parquet")
        ctx["D"][ep] = pd.read_parquet(AW172 / f"{ep}_D_private.parquet")
    ctx["params"] = json.loads(PARAMS171.read_text(encoding="utf8"))
    pkg = json.loads(PARAM_B.read_text(encoding="utf8"))
    ctx["pkg_scaling_H4"] = pkg["hypotheses"]["BEF-0049"]["scaling_B3"]
    u = np.load(OUT / "unions_private.npz")
    ctx["u172"], ctx["u142"] = u["u172"], u["u142"]
    return ctx


# ------------------------------------------------------------------ prepare

def prepare():
    import pyarrow.dataset as ds
    from werkzeuge.b_test.daten import select_index

    OUT.mkdir(parents=True, exist_ok=True)
    for key, path in (("params171", PARAMS171), ("aw142", AW142), ("manifest", MANIFEST)):
        if sha(path) != EXPECT[key]:
            raise RuntimeError("Hashabweichung " + key)
    cand = {ep: pd.read_parquet(AW172 / f"{ep}_candidate_private.parquet") for ep in ("H1", "H4")}
    idx1 = select_index(cand["H1"], "reduced_lvef_strict")
    idx4 = select_index(cand["H4"], "cardiomyopathy_icd")
    tech = pd.read_parquet(AW172 / "technical_index_private.parquet", columns=["subject_id"])
    u172 = np.unique(np.concatenate([idx1.subject_id.to_numpy(), idx4.subject_id.to_numpy(), tech.subject_id.to_numpy()]))
    ok172 = hashlib.sha256(u172.tobytes()).hexdigest() == EXPECT["u172"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    entry = manifest["hypotheses"]["BEF-0087"]
    if sha(entry["path"]) != entry["sha256"]:
        raise RuntimeError("Hashabweichung BEF-0087")
    cols = ["subject_id", "study_id", "split", "ecg_time", "age", "extraction_status", "reference_available"]
    rv = ds.dataset(entry["path"], format="parquet").to_table(columns=cols).to_pandas()
    idxrv = select_index(rv, "echo_rv_systolic_dysfunction")
    u142 = np.unique(np.concatenate([idx1.subject_id.to_numpy(), idx4.subject_id.to_numpy(), idxrv.subject_id.to_numpy()]))
    np.savez(OUT / "unions_private.npz", u172=u172, u142=u142)
    info = dict(utc=now(), n_h1_index=len(idx1), n_h4_index=len(idx4), n_tech=len(tech), n_rv_index=len(idxrv), n_u172=len(u172), u172_hash_matches_aw172=ok172,
                n_u142=len(u142), u142_sha256=hashlib.sha256(u142.tobytes()).hexdigest(),
                rv_subset_of_tech_or_h1h4=bool(np.isin(idxrv.subject_id.to_numpy(), u172).all()))
    dump(OUT / "prepare.json", info)
    print(json.dumps(info))


# ------------------------------------------------------------------ Motor A (S3b-Zeilen)

A_TASKS = {
    "A_6a": dict(rank="6a", analysis_prefix="QRS <120 ms", stratum=None),
    "A_6c_lt65": dict(rank="6c", analysis_prefix="Alter <65", stratum=None),
    "A_6d_F": dict(rank="6d", analysis_prefix="Geschlecht F", stratum="F"),
}


def a_selector(name, C):
    if name == "A_6a":
        return C.qrs_duration_ms.lt(120) & C.classical_lbbb_pattern_positive.eq(False) & C.classical_rbbb_pattern_positive.eq(False)
    if name == "A_6c_lt65":
        return C.age.lt(65)
    return C.sex.eq("F")


def task_A(name, reps, ctx, synthetic=False):
    spec = A_TASKS[name]
    C = ctx["C"]["H4"]
    z = C.loc[a_selector(name, C)].copy()
    x, names = st.raw_design(z, M, "qrs_duration_ms")
    params = ctx["pkg_scaling_H4"]
    if names != params["names"]:
        raise RuntimeError("Designspalten")
    if spec["stratum"] is not None:
        if not z.sex.eq(spec["stratum"]).all():
            raise RuntimeError("Stratum nicht homogen")
        idx = [i for i, n in enumerate(names) if n != "sex_numeric"]
        x = x[:, idx]
        params = dict(params, **{kk: [params[kk][i] for i in idx] for kk in ("names", "mean", "sd", "keep")})
    design = st.fixed_design(x, params)
    y = z.y.to_numpy(float)
    ids = z.subject_id.to_numpy()
    point = st.fit(design, y, ids, 1)
    expected = None
    if not synthetic:
        old = json.loads(AW142.read_text(encoding="utf8"))
        label = {"A_6a": "QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster / BEF-0049",
                 "A_6c_lt65": "Alter <65 / BEF-0049", "A_6d_F": "Geschlecht F / BEF-0049"}[name]
        for s in old["sensitivities"]:
            if s.get("rank") == spec["rank"] and s.get("analysis") == label and s.get("N") == len(z):
                expected = s
                break
        if expected is None:
            raise RuntimeError("Originalzeile nicht gefunden")
        for key in ("beta", "or", "delta_auc", "auc_G", "auc_G_marker"):
            if not np.isclose(point[key], expected[key], rtol=1e-9, atol=0):
                raise RuntimeError(f"Punktschaetzer weicht ab: {key} {point[key]!r} {expected[key]!r}")
        if point["class_persons"] != expected["class_persons"]:
            raise RuntimeError("Klassenpersonen weichen ab")
    target = OUT / f"{name}.json"
    state = json.loads(target.read_text(encoding="utf8")) if target.exists() else dict(task=name, next=0, reps=[])
    union = ctx["u142"]
    for r in range(state["next"], reps):
        if r % 25 == 0:
            ram_guard()
        w = st.draws(union, ids, r)
        act = w > 0
        rec = dict(r=r, distinct_cases=int(np.unique(ids[act & (y == 1)]).size), distinct_controls=int(np.unique(ids[act & (y == 0)]).size))
        try:
            with bk.replicate_rule():
                res = st.fit(design, y, ids, weights=w, wald=False)
            rec.update(ok=True, m={kk: float(res[kk]) for kk in ("beta", "or", "auc_G", "auc_G_marker", "delta_auc", "auc_marker_original_steigend")})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            rec.update(ok=False, reason=str(exc))
        # Loeserkontrolle: AW-171-Kern (200 Schritte, relative Toleranz) auf denselben Ziehungen
        try:
            with bk.replicate_rule():
                v = k.association(design, y, w)
            rec.update(v_ok=True, v={kk: float(v[kk]) for kk in ("beta", "or", "delta_auc", "auc_base", "auc_marker")})
        except Exception as exc:  # noqa: BLE001
            rec.update(v_ok=False, v_reason=type(exc).__name__ + ":" + str(exc))
        state["reps"].append(rec)
        state["next"] = r + 1
        if (r + 1) % 50 == 0 or r + 1 == reps:
            dump(target, dict(state, point={kk: point[kk] for kk in ("beta", "or", "delta_auc", "auc_G", "auc_G_marker", "N", "class_persons")},
                              expected_gate="exakt gleich Originalzeile" if expected is not None else "synthetisch", utc=now(), union_n=len(union)))
            print(json.dumps(dict(task=name, done=r + 1, utc=now())), flush=True)
    return state


# ------------------------------------------------------------------ Motor B (S15-Zweige)

def build_B(ep, kind, ctx):
    marker = F if ep == "H1" else M
    C, D, params, U = ctx["C"][ep], ctx["D"][ep], ctx["params"], ctx["u172"]
    pack = params["endpoints"][ep]["markers"][marker]
    scale = pack["scale"]
    raw, _ = st.raw_design(C, marker)
    z = st.fixed_design(raw, scale)
    y = C.y.to_numpy(float)
    pos = np.searchsorted(U, C.subject_id.to_numpy())
    if kind == "no_qtc_C":
        zz = np.delete(z, 6, axis=1)
        return (lambda w, r: k.association(zz, y, w)), pos
    if kind == "no_qtc_extended":
        start = params["endpoints"][ep]["mi"]["start"]
        base_raw, _ = st.raw_design(D, marker)
        good = np.isfinite(np.delete(base_raw, 5, axis=1)).all(1)
        reduced = D.loc[good].copy()
        rr = base_raw[good].copy()
        rr[:, 5] = start["qtc_fridericia_ms"]
        zr = np.delete(st.fixed_design(rr, scale), 6, axis=1)
        yr = reduced.y.to_numpy(float)
        return (lambda w, r: k.association(zr, yr, w)), np.searchsorted(U, reduced.subject_id.to_numpy())
    if kind == "missing_indicator_extended":
        start = params["endpoints"][ep]["mi"]["start"]
        mi_good = np.isfinite(D[[marker, A]].to_numpy(float)).all(1)
        mf = D.loc[mi_good].copy()
        mf["marker"] = mf[marker]
        ind_good = np.isfinite(mf[["age", "sex_numeric", "qrs_duration_ms", A, marker]].to_numpy(float)).all(1)
        ind = mf.loc[ind_good].copy()
        extras = np.column_stack([ind[c].isna().to_numpy(float) for c in ["heart_rate", "qtc_fridericia_ms"]])
        for c in ["heart_rate", "qtc_fridericia_ms"]:
            ind[c] = ind[c].fillna(start[c])
        ir, _ = st.raw_design(ind, marker)
        iz = st.fixed_design(ir, scale)
        iz = np.column_stack([iz[:, :-1], extras, iz[:, -1]])
        iy = ind.y.to_numpy(float)
        return (lambda w, r: k.association(iz, iy, w)), np.searchsorted(U, ind.subject_id.to_numpy())
    if kind == "standardized":
        cuts = pack["quartiles"]

        def risks(w, r):
            ww = np.ones(len(y)) if w is None else w
            b = np.asarray(k.association(z, y, ww)["coefficients"])
            prob = []
            for median in cuts["medians"]:
                xx = z.copy()
                xx[:, -1] = (median - scale["mean"][-1]) / scale["sd"][-1]
                prob.append(float(np.average(expit(xx @ b), weights=ww)))
            return {**{f"q{i + 1}": v for i, v in enumerate(prob)}, "q4_minus_q1": prob[-1] - prob[0], "n": len(y), "cases": int(y.sum()), "controls": int((y == 0).sum())}
        return risks, pos
    if kind in ("cross_F", "cross_M"):
        cross = F if kind == "cross_F" else M
        cg = np.isfinite(C[[F, M]].to_numpy(float)).all(1)
        ch = C.loc[cg]
        raw2, _ = st.raw_design(ch, cross)
        z2 = st.fixed_design(raw2, params["endpoints"][ep]["markers"][cross]["scale"])
        y2 = ch.y.to_numpy(float)
        return (lambda w, r: k.association(z2, y2, w)), np.searchsorted(U, ch.subject_id.to_numpy())
    if kind == "W24_rbbb":
        ch = C.loc[C.classical_rbbb_pattern_positive.eq(True)]
        raw2, _ = st.raw_design(ch, marker)
        z2 = st.fixed_design(raw2, scale)
        y2 = ch.y.to_numpy(float)
        return (lambda w, r: k.association(z2, y2, w)), np.searchsorted(U, ch.subject_id.to_numpy())
    raise ValueError(kind)


def stored_key(ep, kind):
    return f"{ep}_W24_rbbb" if kind == "W24_rbbb" else f"{ep}_{kind}"


def task_B(name, reps, ctx, synthetic=False):
    _, ep, kind = name.split("_", 2)
    fn, positions = build_B(ep, kind, ctx)
    union_n = len(ctx["u172"])
    target = OUT / f"{name}.json"
    state = json.loads(target.read_text(encoding="utf8")) if target.exists() else dict(task=name, next=0, reps=[], errors={})
    ram_guard()
    point = bk.numeric(fn(None, -1))
    if not synthetic:
        stored = json.loads((AW172 / f"{stored_key(ep, kind)}.json").read_text(encoding="utf8"))["point"]
        for key in ("beta", "or", "delta_auc"):
            if key in stored and key in point and not np.isclose(stored[key], point[key], rtol=1e-9, atol=0):
                raise RuntimeError(f"Punktschaetzer weicht von AW-172 ab: {key} {point[key]!r} {stored[key]!r}")
    for r in range(state["next"], reps):
        if r % 25 == 0:
            ram_guard()
        w = bk.multiplicities(union_n, r)[positions].astype(float)
        rec = dict(r=r)
        try:
            with bk.replicate_rule():
                rec["m"] = bk.numeric(fn(w, r))
            rec["ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, reason=type(exc).__name__ + ":" + str(exc))
        state["reps"].append(rec)
        state["next"] = r + 1
        if (r + 1) % 50 == 0 or r + 1 == reps:
            dump(target, dict(state, point=point, utc=now(), union_n=union_n))
            print(json.dumps(dict(task=name, done=r + 1, utc=now())), flush=True)
    return state


# ------------------------------------------------------------------ MI-Nenner

def mi_denominators(ctx):
    out = {}
    cols = ["age", "sex_numeric", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms"]
    for ep, marker in (("H1", F), ("H4", M)):
        D = ctx["D"][ep]
        good = np.isfinite(D[[marker, A]].to_numpy(float)).all(1)
        mf = D.loc[good]
        comp = np.isfinite(mf[cols].to_numpy(float)).all(1)

        def n3(df):
            return dict(n=int(len(df)), cases=int((df.y == 1).sum()), controls=int((df.y == 0).sum()))
        out[ep] = dict(reference_valid_D=n3(D), mi_set=n3(mf), complete_cases=n3(mf.loc[comp]), imputed_persons=n3(mf.loc[~comp]),
                       missing_by_variable={c: dict(cases=int(mf.loc[mf.y == 1, c].isna().sum()), controls=int(mf.loc[mf.y == 0, c].isna().sum())) for c in cols},
                       imputations=20)
    dump(OUT / "mi_denominators.json", out)
    print(json.dumps(out))
    return out


# ------------------------------------------------------------------ Hauptprogramm

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "task", "mi", "synthetic"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--reps", type=int, default=PLANNED)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "mi":
        mi_denominators(load_ctx(True))
    elif a.cmd == "task":
        ctx = load_ctx(True)
        if a.name in A_TASKS:
            task_A(a.name, a.reps, ctx)
        else:
            task_B(a.name, a.reps, ctx)
    else:
        import tempfile
        OUT = Path(tempfile.mkdtemp(prefix="aw238_syn_"))
        ctx = synthetic_ctx()
        for name in A_TASKS:
            s = task_A(name, a.reps, ctx, synthetic=True)
            ok = sum(1 for r in s["reps"] if r["ok"])
            old_ok = sum(1 for r in s["reps"] if r["ok"] and min(r["distinct_cases"], r["distinct_controls"]) >= 100)
            print(name, "corrected valid", ok, "of", len(s["reps"]), "old-rule valid", old_ok, "variant ok", sum(1 for r in s["reps"] if r["v_ok"]))
        for ep in ("H1", "H4"):
            for kind in ("no_qtc_extended", "no_qtc_C", "missing_indicator_extended", "standardized", "cross_F", "cross_M", "W24_rbbb"):
                if ep == "H4" and kind == "W24_rbbb":
                    continue
                s = task_B(f"S15_{ep}_{kind}", min(a.reps, 8), ctx, synthetic=True)
                print(ep, kind, "valid", sum(1 for r in s["reps"] if r["ok"]), "of", len(s["reps"]), {r.get("reason") for r in s["reps"] if not r["ok"]})
        print("synthetic done", OUT)


if __name__ == "__main__":
    main()
