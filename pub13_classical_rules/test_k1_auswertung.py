"""Tests fuer k1_auswertung.py (AW-236): ausschliesslich synthetische Daten, keine Patientendaten.
Aufruf: python -I -m pytest -q test_k1_auswertung.py -p no:cacheprovider
"""
import gzip
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k1_auswertung as K  # noqa: E402
import k1_geruest as G  # noqa: E402

CODELIST = K.CFG["codelist"]


# ----------------------------------------------------------------------------- Statistik-Bausteine
def test_rcs_basis_shape_and_linear_tails():
    kn = [20, 40, 60, 80]
    x = np.linspace(0, 120, 241)
    B = K.rcs_basis(x, kn)
    assert B.shape == (241, 3)
    # natuerlicher Spline: jenseits des letzten Knotens linear (zweite Differenz 0)
    tail = B[x > 80]
    d2 = np.diff(tail, n=2, axis=0)
    assert np.max(np.abs(d2)) < 1e-9


def test_logit_recovers_coefficients_and_ridge_on_separation():
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.normal(size=(n, 3))
    beta = np.array([-0.5, 0.8, -0.4, 0.3])
    Z = np.column_stack([np.ones(n), X])
    y = (rng.uniform(size=n) < K.expit(Z @ beta)).astype(float)
    b, info = K.fit_logit_safe(Z, y)
    assert not info["ridge"] and np.max(np.abs(b - beta)) < 0.08
    # perfekte Separation -> Ridge mit festem lambda
    xs = rng.normal(size=500)
    ys = (xs > 0).astype(float)
    Zs = np.column_stack([np.ones(500), xs])
    b2, info2 = K.fit_logit_safe(Zs, ys)
    assert info2["ridge"] and np.all(np.isfinite(b2)) and b2[1] > 0


def make_selection_world(n=30000, seed=3):
    """T beeinflusst die Verifikation (Begg-Greenes), D haengt von T und X ab, V unabhaengig von D gegeben T, X (MAR)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, n)
    x = (age - 65) / 12
    female = (rng.uniform(size=n) < 0.5).astype(float)
    p_d = K.expit(-1.2 + 0.7 * x + 0.3 * female)
    D = (rng.uniform(size=n) < p_d).astype(float)
    # Regel: sens 0.55, spec 0.92 (unabhaengig von X)
    T = np.where(D == 1, rng.uniform(size=n) < 0.55, rng.uniform(size=n) < 0.08).astype(float)
    p_v = K.expit(-1.0 + 1.5 * T + 0.4 * x)  # Verifikation haengt von T und X ab, nicht von D
    V = (rng.uniform(size=n) < p_v).astype(int)
    truth = dict(sens=float(T[D == 1].mean()), spec=float(1 - T[D == 0].mean()))
    return age, female, T, D, V, truth


def basis_for(age, female):
    kn = K.rcs_knots(age)
    A = K.rcs_basis(age, kn)
    A = (A - A.mean(0)) / A.std(0)
    return np.column_stack([A, female])


def test_ipw_removes_verification_bias_naive_is_biased():
    age, female, T, D, V, truth = make_selection_world()
    Xb = basis_for(age, female)
    band = K.age_band(age)
    Dm = np.where(V == 1, D, np.nan)
    u = K.IPWUnit({"s": T}, Dm, V, Xb, female, band, do_kipp=True, weighted=True)
    st = u.stats(np.ones(len(V)))["s"]
    naive = K.two_by_two_ipw(T, Dm, V, np.ones(len(V)))
    assert abs(st["sens"] - truth["sens"]) < 0.025 and abs(st["spec"] - truth["spec"]) < 0.01
    assert abs(naive["sens"] - truth["sens"]) > 0.04  # Verifikationsverzerrung ist da (Sens hoeher bei Verifizierten)
    m = st["manski"]
    assert m["sens_lo"] <= truth["sens"] <= m["sens_hi"] and m["spec_lo"] <= truth["spec"] <= m["spec_hi"]
    # Kipppunkt: bei kappa = 1 (MAR) nahe an der Wahrheit, bei kappa+ = 2 steigt die Sensitivitaet
    k11 = st["kipp"]["1.0|1.0"]
    assert abs(k11[0] - truth["sens"]) < 0.03
    assert st["kipp"]["2.0|1.0"][0] > k11[0] > st["kipp"]["0.5|1.0"][0]
    assert 0.4 < st["ess_frac"] <= 1.0 and 0 < st["trivial_bound"] < 1


def test_bootstrap_bounds_cover_truth_and_are_ordered():
    age, female, T, D, V, truth = make_selection_world(n=8000, seed=11)
    Dm = np.where(V == 1, D, np.nan)
    u = K.IPWUnit({"s": T}, Dm, V, basis_for(age, female), female, K.age_band(age), do_kipp=True)
    rng = K.seed_stream(12345, 2, "primary", "dev")
    point, reps = K.run_bootstrap(u, rng, 60, every=1000)
    itv = K.interval(reps["s"]["sens"], 0.2, 60)
    assert itv["lo"] < point["s"]["sens"] < itv["hi"] and itv["lo95"] <= itv["hi95"]
    assert itv["lo"] < truth["sens"] + 0.08 and itv["hi"] > truth["sens"] - 0.08
    assert "kipp_sens_0.5|2.0" in reps["s"] and len(reps["s"]["spec"]) == 60
    # reproduzierbar (gleicher Seed, gleicher Strom)
    u2 = K.IPWUnit({"s": T}, Dm, V, basis_for(age, female), female, K.age_band(age), do_kipp=True)
    _p2, reps2 = K.run_bootstrap(u2, K.seed_stream(12345, 2, "primary", "dev"), 60, every=1000)
    assert np.allclose(reps["s"]["sens"], reps2["s"]["sens"])


def test_streams_differ_between_modes_and_uses():
    a = K.seed_stream(1, 2, "primary", "dev").integers(0, 10**9, 3).tolist()
    b = K.seed_stream(1, 2, "primary", "test").integers(0, 10**9, 3).tolist()
    c = K.seed_stream(1, 2, "concord", "dev").integers(0, 10**9, 3).tolist()
    assert a != b and a != c


def test_judge_e_classes_and_unit_priority():
    comps = [dict(id="a_spec", T="s", metric="spec", kind="ge", theta=0.9, claim="x"),
             dict(id="b_sens", T="s", metric="sens", kind="le", theta=0.1, claim="y")]
    rng = np.random.default_rng(2)

    def reps(spec_mu, sens_mu):
        d = {"spec": rng.normal(spec_mu, 0.005, 4000), "sens": rng.normal(sens_mu, 0.01, 4000)}
        for gk in [f"{a}|{b}" for a in K.KAPPA_GRID for b in K.KAPPA_GRID]:
            d[f"kipp_spec_{gk}"] = d["spec"].copy()
            d[f"kipp_sens_{gk}"] = d["sens"].copy()
        return {"s": d, "_meta": {"B": 4000, "failed": 0}}

    pt = {"s": dict(spec=0.97, sens=0.05, trivial_bound=0.95, ess_frac=0.8, kipp={f"{a}|{b}": (0.05, 0.97) for a in K.KAPPA_GRID for b in K.KAPPA_GRID})}
    j = K.judge_e(pt, reps(0.97, 0.05), comps, "E", with_kipp=True)
    assert j["unit_class"] == G.HAELT and j["kipp"]["_robust"] is True
    assert "durch Positivrate vorbestimmt" in j["components"][0]["hinweise"]
    j2 = K.judge_e(pt, reps(0.80, 0.05), comps, "E", with_kipp=False)
    assert j2["unit_class"] == G.HAELT_NICHT  # eine verfehlte Komponente genuegt
    j3 = K.judge_e(pt, reps(0.9, 0.05), comps, "E", with_kipp=False)
    assert j3["unit_class"] == G.UNENTSCHIEDEN


def test_masked_copy_complement_coarsening():
    res = {"counts": {"n_pop": 1234, "cases": 7, "cells": {"tp": 7, "fp": 181, "fn": 53, "tn": 993}}, "x": 0.5}
    m = K.masked_copy(res)
    assert m["counts"]["cases"] == "<10" and m["counts"]["n_pop"] == 1200 and m["counts"]["cells"]["fp"] == 200
    assert m["x"] == 0.5
    ok = K.masked_copy({"counts": {"n_pop": 1234, "cases": 77}})
    assert ok["counts"]["n_pop"] == 1230 and ok["counts"]["cases"] == 80


def test_wald_and_rr_helpers():
    w = K.wald_hr(np.log(1.5), 0.1, 0.025 / 4)
    assert abs(w["hr"] - 1.5) < 1e-9 and w["lo"] < 1.5 < w["hi"] and w["lo95"] > w["lo"]
    r = K.rr_wald(100, 1000, 50, 1000, 0.025 / 3)
    assert abs(r["rr"] - 2.0) < 1e-9 and r["lo"] > 1.0
    cls, hint = G.classify_marker(r["lo"], r["hi"])
    assert cls == G.HAELT


# ----------------------------------------------------------------------------- Regelauswertung und Maschinenflags
def synth_idx(n=6, with_cache=True):
    rows = []
    for i in range(n):
        r = dict(subject_id=i, study_id=100 + i, ecg_time=pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(i)), hadm_id=np.nan,
                 ed_stay_id=np.nan, ed_intime=pd.NaT, icustay_id=np.nan, icu_intime=pd.NaT, icu_outtime=pd.NaT,
                 sex="M" if i % 2 == 0 else "F", age=60 + i, chief_complaint=None, hr_bpm=70.0, pr_ms=160.0, qrs_ms=90.0, qt_ms=400.0,
                 split="train", usable=True, report_af_state="not_mentioned", report_lbbb_state="not_mentioned",
                 report_rbbb_state="not_mentioned", report_infarct_state="not_mentioned", report_lvh_state="not_mentioned",
                 report_pacemaker_state="not_mentioned", report_low_voltage_state="not_mentioned", rr_ms=860.0,
                 report_text="Sinus rhythm | Normal ECG")
        for l in K.AMP_LEADS:
            r[f"lead_measurement_{l}_qrs_r_height_mv"] = 0.6
            r[f"lead_measurement_{l}_qrs_s_depth_mv"] = 0.4
        r.update(signal_quality_usable=True, classical_pathological_q_waves_positive=False, classical_rbbb_pattern_positive=False,
                 classical_qrs_widening_ge_120=92.0, pr_interval_ms=165.0)
        rows.append(r)
    return pd.DataFrame(rows)


def test_machine_flags_text_rules():
    idx = synth_idx(5)
    idx.loc[0, "report_text"] = "Sinus rhythm | Normal ECG"
    idx.loc[1, "report_text"] = "Atrial fibrillation | Abnormal ECG"
    idx.loc[2, "report_text"] = "Sinus rhythm | Otherwise normal ECG | Borderline ECG"
    idx.loc[3, "report_text"] = "Sinus rhythm | 1st degree A-V block | Left anterior fascicular block | Abnormal ECG"
    idx.loc[4, "report_text"] = None
    idx.loc[4, "report_rbbb_state"] = "positive"
    idx.loc[1, "report_af_state"] = "positive"
    mf = K.machine_flags(idx)
    assert mf.loc[0, "normal"] == 1 and mf.loc[0, "sinus"] == 1 and mf.loc[0, "mcat"] == 0
    assert mf.loc[1, "sinus"] == 0 and mf.loc[1, "normal"] == 0 and mf.loc[1, "mcat"] == 2
    assert mf.loc[2, "normal"] == 0 and mf.loc[2, "mcat"] == 1
    assert mf.loc[3, "avb"] == 1 and mf.loc[3, "lafb"] == 1
    assert np.isnan(mf.loc[4, "normal"]) and np.isnan(mf.loc[4, "sinus"])
    # R09: Exposition RSB positiv, Vergleich nur ohne Ausschluesse
    assert mf.loc[0, "R09_m"] == 0 and np.isnan(mf.loc[1, "R09_m"]) and np.isnan(mf.loc[3, "R09_m"]) and mf.loc[4, "R09_m"] == 1


def test_evaluate_rules_voltage_and_gating():
    idx = synth_idx(4)
    idx.loc[0, "lead_measurement_V1_qrs_s_depth_mv"] = 2.0
    idx.loc[0, "lead_measurement_V5_qrs_r_height_mv"] = 2.0   # SL = 4,0 > 3,5
    idx.loc[1, "report_pacemaker_state"] = "positive"
    idx.loc[2, "signal_quality_usable"] = False
    for l in K.R.LIMB:
        idx.loc[3, f"lead_measurement_{l}_qrs_r_height_mv"] = 0.2
        idx.loc[3, f"lead_measurement_{l}_qrs_s_depth_mv"] = 0.1
    idx = pd.concat([idx, K.machine_flags(idx)], axis=1)
    r = K.evaluate_rules(idx, True)
    assert r.loc[0, "R02_gt"] == 1 and r.loc[3, "R02_gt"] == 0
    assert np.isnan(r.loc[1, "R02_gt"]) and np.isnan(r.loc[2, "R02_gt"]) and np.isnan(r.loc[2, "R06_a"])
    assert r.loc[3, "R06_a"] == 1 and r.loc[3, "R07_a"] == 1 and r.loc[0, "R06_a"] == 0
    assert r.loc[0, "lvh_volt"] == 1 and np.isnan(r.loc[0, "R14_a155"])  # LVH positiv -> R14 nicht auswertbar
    assert r.loc[3, "R14_a155"] == 0 and r.loc[3, "R14_o2"] == 0
    assert r.loc[0, "R08_baz"] == 0 and abs(r.loc[0, "qtc_baz"] - 400 / np.sqrt(0.86)) < 1e-6
    assert r.loc[0, "R18_m"] == 0 and r.loc[0, "qrs_machine"] == 90.0
    r2 = K.evaluate_rules(idx, False)  # Testpartition: kein Cache
    assert r2["R02_gt"].isna().all() and r2["R08_baz"].notna().all()


# ----------------------------------------------------------------------------- Einheiten an synthetischen Daten
def synth_data(n=6000, seed=5, hr_effect=0.0, qrs_shift=0.0):
    """Data-Objekt mit Regeln, Referenzen und Kovariaten, ohne Dateien."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(62, 15, n), 18, 95)
    female = (rng.uniform(size=n) < 0.5).astype(float)
    adm = rng.choice(["icu", "ed", "inpatient", "other"], size=n, p=[0.1, 0.3, 0.4, 0.2])
    cov = pd.DataFrame({"age": age, "female": female, "sex_known": 1.0, "adm": adm, "hr": rng.normal(80, 14, n), "hr_missing": 0.0})
    for h in K.Design.HX + ["hx_cm", "hx_pe", "hx_ph"]:
        cov[h] = (rng.uniform(size=n) < 0.08).astype(float)
    cov["claim_pop"] = (rng.uniform(size=n) < 0.6).astype(float)
    cov["has_prior"] = (rng.uniform(size=n) < 0.7).astype(float)
    cov["mcat_border"] = (rng.uniform(size=n) < 0.2).astype(float)
    cov["mcat_abn"] = (rng.uniform(size=n) < 0.4).astype(float)
    for m in K.COX_MED:
        cov[m] = (rng.uniform(size=n) < 0.15).astype(float)
    cov["med_assessed"] = 1.0
    cov["med_qt_prolonging_drug"] = (rng.uniform(size=n) < 0.1).astype(float)
    for c in ("lab_potassium", "lab_magnesium", "lab_total_calcium"):
        cov[c] = rng.normal(4, 0.4, n)
    # LVH, Regeln R02/R03/R04 (sens ~0.4/0.45, spec ~0.95/0.93), Referenz nur bei Echo
    lvh = (rng.uniform(size=n) < K.expit(-1.3 + 0.02 * (age - 62))).astype(float)
    t_sl = np.where(lvh == 1, rng.uniform(size=n) < 0.30, rng.uniform(size=n) < 0.03).astype(float)
    t_co = np.where(lvh == 1, rng.uniform(size=n) < 0.40, rng.uniform(size=n) < 0.07).astype(float)
    t_pe = np.where(lvh == 1, rng.uniform(size=n) < 0.55, rng.uniform(size=n) < 0.16).astype(float)
    pv = K.expit(-1.0 + 0.8 * t_sl + 0.3 * (age - 62) / 15)
    ver = rng.uniform(size=n) < pv
    flip = lambda t, pr: np.where(rng.uniform(size=n) < pr, 1.0 - t, t)  # noqa: E731
    r = pd.DataFrame({"R02_gt": t_sl, "R02_ge": flip(t_sl, 0.04), "R03_gt": t_co, "R03_ge": flip(t_co, 0.04), "R04_ge": t_pe,
                      "R04_gt": flip(t_pe, 0.04), "R03_prod": t_co, "R03_fixM": flip(t_co, 0.1), "R03_fixF": flip(t_co, 0.1),
                      "R04_fixM": flip(t_pe, 0.1), "R04_fixF": flip(t_pe, 0.1)})
    peri = (rng.uniform(size=n) < 0.04).astype(float)
    r["R06_a"] = np.where(peri == 1, rng.uniform(size=n) < 0.1, rng.uniform(size=n) < 0.01).astype(float)
    r["R06_c"] = flip(r["R06_a"].values, 0.1)
    r["R06_b"] = flip(r["R06_a"].values, 0.1)
    r["lafb"] = (rng.uniform(size=n) < 0.05).astype(float)
    r["wpw"] = (rng.uniform(size=n) < 0.01).astype(float)
    amy = (rng.uniform(size=n) < 0.05)
    r["R07_a"] = np.where(amy, rng.uniform(size=n) < 0.3, rng.uniform(size=n) < 0.05).astype(float)
    r["R07_b"] = flip(r["R07_a"].values, 0.2)
    r["R07_c"] = flip(r["R07_a"].values, 0.2)
    r["R14_a155"] = (rng.uniform(size=n) < 0.12).astype(float)
    r["R14_o2"] = np.where(r["R14_a155"] == 1, rng.uniform(size=n) < 0.7, rng.uniform(size=n) < 0.1).astype(float)
    # QRS, QTc, Maschinenflags
    sex_f = female == 1
    qrs = np.clip(rng.normal(100, 16, n) + qrs_shift * (~sex_f), 70, 200)
    r["qrs_machine"] = qrs
    r["qrs_own"] = qrs + rng.normal(0, 4, n)
    r["pr_machine"] = np.clip(rng.normal(165, 30, n), 80, 320)
    r["R18_m"] = (r["pr_machine"] > 200).astype(float)
    r["R18_own"] = flip(r["R18_m"].values, 0.05)
    r["R09_m"] = np.where(rng.uniform(size=n) < 0.9, (rng.uniform(size=n) < 0.06).astype(float), np.nan)
    r["R09_a155"] = np.where(np.isfinite(r["R09_m"]), flip(np.nan_to_num(r["R09_m"].values), 0.05), np.nan)
    r["R08_baz"] = (rng.uniform(size=n) < 0.05).astype(float)
    r["R08_fri"] = flip(r["R08_baz"].values, 0.1)
    r["R08_lin"] = r["R08_baz"]
    r["qtc_cat"] = rng.integers(0, 4, n).astype(float)
    r["qtc_baz"] = 430 + 30 * r["qtc_cat"]
    r["lvh_volt"] = rng.choice([0.0, 1.0, np.nan], size=n, p=[0.8, 0.12, 0.08])
    r["lbbb_any"] = (rng.uniform(size=n) < 0.04).astype(float)
    r["sinus"] = (rng.uniform(size=n) < 0.8).astype(float)
    r["af_any"] = (rng.uniform(size=n) < 0.05).astype(float)
    r["normal"] = (rng.uniform(size=n) < 0.2).astype(float)
    r["normal_add"] = np.fmax(r["normal"], (rng.uniform(size=n) < 0.1).astype(float))
    r["no_stt"] = (rng.uniform(size=n) < 0.3).astype(float)
    # Referenzen
    f = pd.DataFrame(index=range(n))
    has_tte = rng.uniform(size=n) < K.expit(-0.6 + 0.7 * t_sl)
    f["lvh_30d"] = np.where(ver, lvh, np.nan)
    f["lvh_7d"] = np.where(ver & (rng.uniform(size=n) < 0.4), lvh, np.nan)
    f["lvh_24h"] = np.where(ver & (rng.uniform(size=n) < 0.2), lvh, np.nan)
    f["peri_30d"] = np.where(has_tte, peri, np.nan)
    f["peri_7d"], f["peri_24h"] = f["peri_30d"], f["peri_30d"]
    f["idx_AMYLOID"] = amy.astype(int)
    f["idx_KARDIOMYOPATHIE_HINWEIS"] = (amy & (rng.uniform(size=n) < 0.4)).astype(int)
    iv = np.where(amy & has_tte, (rng.uniform(size=n) < 0.5).astype(float), np.nan)
    f["ivs12_30d"], f["ivs12_7d"], f["ivs12_24h"] = iv, iv, iv
    f["n_prior_dx"] = (rng.uniform(size=n) < 0.7).astype(int)
    f["idx_ACUTE_MI"] = (rng.uniform(size=n) < 0.03).astype(int)
    f["idx_OLD_MI"] = (rng.uniform(size=n) < 0.05).astype(int)
    f["prior_OLD_MI"] = np.where(r["R14_a155"] == 1, rng.uniform(size=n) < 0.4, rng.uniform(size=n) < 0.05).astype(int)
    f["prior_ACUTE_MI"] = (rng.uniform(size=n) < 0.03).astype(int)
    f["prior_pci_cabg"] = (rng.uniform(size=n) < 0.04).astype(int)
    f["idx_NSTEMI"] = (rng.uniform(size=n) < 0.02).astype(int)
    f["ecg_stay_day"] = rng.uniform(0, 5, n)
    # lvef, Todes- und VHF-Zeiten
    low_ef = (rng.uniform(size=n) < K.expit(-2.0 + 0.04 * (qrs - 100) + 0.3 * (~sex_f))).astype(float)
    f["lvef35_30d"] = np.where(rng.uniform(size=n) < 0.35, low_ef, np.nan)
    f["lvef35_7d"], f["lvef35_24h"] = f["lvef35_30d"], f["lvef35_30d"]
    expo = r["R09_m"].fillna(0).values
    lam = 0.05 * np.exp(0.03 * (age - 62) + np.log(1.0 + hr_effect) * expo)
    tt = rng.exponential(1.0 / lam)
    cens = rng.uniform(0.3, 1.2, n)
    f["death_t"] = np.minimum(tt, np.minimum(cens, 1.0)) * 365.25
    f["death_ev"] = (tt <= np.minimum(cens, 1.0)).astype(float)
    ta = rng.exponential(1.0 / (0.04 * np.exp(0.02 * (age - 62))), n)
    f["af_t"] = np.minimum(ta, np.minimum(cens, 1.0)) * 365.25
    f["af_ev"] = (ta <= np.minimum(cens, 1.0)).astype(float)
    f["af_prev"] = 0.0
    f["followup_contact"] = 1.0
    for k, p in (("vtvf", 0.03), ("arrest", 0.01), ("mort30", 0.04), ("mi_prob", 0.08), ("trop_pos", 0.2), ("cond1y", 0.03),
                 ("rv_dil", 0.1), ("ph_echo", 0.1)):
        v = (rng.uniform(size=n) < p).astype(float)
        v[rng.uniform(size=n) < 0.05] = np.nan
        f["lab_" + k] = v
    d = K.Data.__new__(K.Data)
    d.keys = pd.DataFrame({"subject_id": np.arange(n), "study_id": np.arange(n)})
    d.rules, d.cov, d.refs, d.n = r, cov, f, n
    return d


def test_run_e_unit_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)  # kleines B im Test
    d = synth_data()
    design = K.Design(d.cov)
    K.B_SENS = 15
    res = K.run_e("R02", d, design, 4242, "dev", 25, tmp_path)
    assert res["status"] == "ok" and res["final"]["klasse"] in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN)
    c = res["judge"]["components"][0]
    assert c["lo"] <= c["hi"] and 0 < c["lo95"] <= c["hi95"] < 1
    assert 0.9 < c["estimate"] < 1.0 and res["counts"]["n_verified"] > 500
    dg = res["diagnostik"]
    assert dg["gewichtet"] and 0 < dg["c_statistik"] < 1 and dg["n_spalten"] > 5 and dg["ess_frac"] > 0
    assert res["abdeckung"]["n_index"] == d.n
    assert "LR+>=5" in res["hinweisurteile"] and res["concordance"]["variante_klasse"] in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN)
    assert set(res["sensitivitaeten"]) >= {"echo_7d", "echo_24h_ungewichtet", "regel_NULL_als_negativ", "regel_NULL_als_positiv"}
    assert all(v.get("status") == "ok" for v in res["sensitivitaeten"].values()), {k: v.get("status") for k, v in res["sensitivitaeten"].items()}
    assert res["geschlecht"]["F"]["n_pop"] + res["geschlecht"]["M"]["n_pop"] == res["counts"]["n_pop"]
    assert (tmp_path / "dev" / "units" / "R02.json").exists()
    r4 = K.run_e("R04", d, design, 4242, "dev", 20, tmp_path)
    ids = [c["id"] for c in r4["judge"]["components"]]
    assert ids == ["R04_T1_spec", "R04_T2_dsens"]
    for u in ("R06", "R07", "R14"):
        out = K.run_e(u, d, design, 4242, "dev", 20, tmp_path)
        assert out["status"] == "ok", (u, out.get("status"))
    # R14 ist ungewichtet (Verifikation = Referenz bestimmbar)
    assert K.run_e("R14", d, design, 4242, "dev", 20, tmp_path)["judge"]["kipp"] is None


def test_run_p_recovers_planted_hazard(tmp_path):
    d = synth_data(n=9000, hr_effect=0.8, seed=21)
    design = K.Design(d.cov)
    res = K.run_p("R09", d, design, 7, "dev", tmp_path)
    p = res["primaer"]["death"]
    assert p["status"] == "ok" and 1.2 < p["hr"] < 2.6, p
    d0 = synth_data(n=30000, hr_effect=0.0, seed=22)
    res0 = K.run_p("R18", d0, K.Design(d0.cov), 7, "dev", tmp_path)
    assert res0["primaer"]["death"]["status"] == "ok" and 0.7 < res0["primaer"]["death"]["hr"] < 1.45
    assert res0["final"]["klasse"] in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN)
    assert "gesamtkohorte" in res0["sekundaer"] and "ohne_PR_unter_120" in res0["sekundaer"]


def test_run_r08_and_r20_and_r19(tmp_path):
    d = synth_data(n=9000, seed=31, qrs_shift=14.0)
    design = K.Design(d.cov)
    r8 = K.run_r08(d, design, 1, "dev", tmp_path)
    assert set(r8) == {"R08_VTVF", "R08_Tod"}  # v3: Arrhythmie und Tod getrennt
    assert r8["R08_Tod"]["primaer"]["Tod_30_Tage"]["status"] == "ok"
    pt, kt = r8["R08_Tod"]["primaer"]["Tod_30_Tage"], r8["R08_Tod"]["konkordanz"]["Tod_30_Tage"]
    expect = pt["klasse"] if kt.get("klasse") == pt["klasse"] else G.UNENTSCHIEDEN  # Konkordanzregel: nur bei gleicher Klasse
    assert r8["R08_Tod"]["final"]["klasse"] == expect and r8["R08_Tod"]["final"]["primaer"] == pt["klasse"]
    assert any("gleichzeitige Assoziation" in h for h in r8["R08_VTVF"]["final"]["hinweise"])
    assert "Dosis_Wirkung_Stufe_0_bis_3_RR_je_Stufe" in r8["R08_Tod"]["sekundaer"]["Tod_30_Tage"]
    r19 = K.run_r19(d, design, 1, "dev", 25, tmp_path)
    v = r19["varianten"]["primaer_Maschinenwert"]
    assert v["status"] == "ok" and (v["delta_status"] == "ok" or v["anzeige"] == "nicht pruefbar")
    assert r19["varianten"]["konkordanz_eigene_Dauer"]["status"] == "ok" and r19["varianten"]["konkordanz_eigene_Dauer"]["n_pop"] == v["n_pop"]
    ed = synth_data(n=40000, seed=41)
    r20 = K.run_r20(ed, 3, "dev", 50, tmp_path)
    a1, a2 = r20["R20_A1"]["primaer"], r20["R20_A2"]["primaer"]
    assert r20["R20_A1"]["status"] == "ok" and 0 < a1["estimate"] < 1 and a1["n"] == int(np.nansum(ed.rules["normal"] == 1) - np.sum((ed.rules["normal"].values == 1) & ~np.isfinite(ed.refs["lab_mi_prob"].values)))
    assert a2["n"] > 10 and 0 <= a2["events"] <= a2["n"] and 0 < a2["estimate"] < 1
    assert "interaktion_F_minus_M" in r20["R20_A1"]["geschlecht"]


def test_r19_delta_detects_sex_shift():
    """Frauen-Risikokurve ist um 12 ms nach links verschoben (gleiches Risiko bei kuerzerem QRS): Delta ~ -12."""
    rng = np.random.default_rng(8)
    n = 40000
    fem = (rng.uniform(size=n) < 0.5).astype(float)
    q = np.clip(rng.normal(115, 25, n), 80, 190)
    shift = 12.0
    risk = K.expit(-6.0 + 0.045 * (q + shift * fem))
    D = (rng.uniform(size=n) < risk).astype(float)
    V = (rng.uniform(size=n) < 0.5).astype(int)
    knots = list(G.QRS_KNOTS)
    qs = K.rcs_basis(q, knots)
    mu, sd = qs.mean(0), qs.std(0)
    qs_s = (qs - mu) / sd
    Xo = np.column_stack([rng.normal(size=(n, 3)) * 0, np.zeros((n, 3)), np.zeros((n, 3))])
    inp = dict(q=q, D=np.where(V == 1, D, np.nan), V=V, fem=fem, mu=mu, sd=sd, knots=knots, n=n, nq=4, Xo=Xo,
               Zv=np.column_stack([np.ones(n), fem, qs_s, qs_s * fem[:, None]]), Zo=np.column_stack([np.ones(n), qs_s, Xo]))
    std = np.arange(0, n, 8)
    out, _w = K.r19_deltas(inp, np.ones(n), std)
    delta, status, risks = out["p"]
    assert status == "ok" and -16.0 < delta < -8.0, (delta, status)


def test_assemble_and_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    d = synth_data()
    design = K.Design(d.cov)
    K.B_SENS = 10
    for mode in ("dev", "test"):
        K.run_e("R02", d, design, 1, mode, 20, tmp_path)
        K.run_p("R09", d, design, 1, mode, tmp_path)
    units = K.assemble("test", tmp_path, tmp_path / "pub")
    assert units["R02"]["replikation"]["urteil"] in ("repliziert", "Replikation abweichend", "nicht informativ (beide unentschieden)")
    rep = (tmp_path / "pub" / "report_test_aggregate.md").read_text(encoding="utf-8")
    assert "R02" in rep and "R09" in rep
    j = json.loads((tmp_path / "pub" / "results_test_aggregate.json").read_text(encoding="utf-8"))
    assert isinstance(j["R02"]["counts"]["n_pop"], (int, str))
    assert K.dev_results_hash(tmp_path) != K.result_hash(tmp_path, "test")


# ----------------------------------------------------------------------------- Waechter und Testregister
def test_guard_blocks_test_without_register(tmp_path):
    con = duckdb.connect()
    with pytest.raises(PermissionError):
        K.load_base(con, dict(K.CFG, root=str(tmp_path)), "dev", {}) if False else G.partition_sql("x", "locked_test", "dev")
    with pytest.raises(PermissionError):
        G.partition_sql("x", "locked_test", "test")  # ohne Registerargumente
    reg = tmp_path / "TESTZUGRIFFE.md"
    reg.write_text("| anderes |\n", encoding="utf-8")
    H = {k: c * 64 for k, c in zip("SCMD", "abcd")}
    args = dict(register_path=reg, spec_hash=H["S"], script_hash=H["C"], manifest_hash=H["M"], dev_results_hash=H["D"])
    with pytest.raises(PermissionError):
        G.partition_sql("x", "locked_test", "test", **args)
    G.append_register(reg, G.register_line("Beginn", "AW-236", H["S"], H["C"], H["M"], H["D"], "locked_test"))
    assert "locked_test" in G.partition_sql("x", "locked_test", "test", **args)
    # Entwicklungsmodus bleibt gesperrt
    with pytest.raises(PermissionError):
        G.partition_sql("x", "locked_test", "dev", **args)


def test_bound_file_hashes():
    K.check_bound_files()


# ----------------------------------------------------------------------------- Ende-zu-Ende mit synthetischer Datenwelt
def make_world(root: Path, n_persons=2600, seed=0):
    """Kleine synthetische Welt mit allen Eingabetabellen im Schema der echten Dateien (keine echten Daten)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "sealed").mkdir(parents=True, exist_ok=True)
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    codes = pd.read_csv(CODELIST, dtype={"icd_code": str})
    by_list = {k: v[["icd_version", "icd_code"]].values.tolist() for k, v in codes.groupby("list_id")}
    ccf, mrl, mm, cache, echo_m, echo_e, labf, medf, labs_w, adm_rows, dx_rows, tl, pch = ([] for _ in range(13))
    sid, hid, mid = 100000, 500000, 900000
    base0 = pd.Timestamp("2140-01-01")
    for i in range(n_persons):
        subj = 1000 + i
        split = ("train", "validation", "locked_test")[0 if i % 5 < 3 else (1 if i % 5 == 3 else 2)]
        female = rng.uniform() < 0.5
        sexu = rng.uniform() < 0.01  # unbekanntes Geschlecht
        age0 = int(np.clip(rng.normal(62, 15), 19, 90))
        lvh = rng.uniform() < 0.22
        amy = rng.uniform() < 0.05
        eff = rng.uniform() < 0.05
        lowv = amy or (eff and rng.uniform() < 0.5)
        hf = rng.uniform() < 0.25
        n_prior = rng.choice([0, 1, 2], p=[0.35, 0.4, 0.25])
        t_first = base0 + pd.Timedelta(days=int(rng.integers(400, 2500)), hours=int(rng.integers(0, 24)))
        for k in range(n_prior):
            hid += 1
            adm = t_first - pd.Timedelta(days=int(rng.integers(30, 300)) * (k + 1))
            adm_rows.append((subj, hid, adm, adm + pd.Timedelta(days=3)))
            for lid, p in (("OLD_MI", 0.10), ("H_HERZINSUFFIZIENZ", 0.15 if hf else 0.03), ("K_HYPERTONIE", 0.4), ("K_DIABETES", 0.2),
                           ("H_VHF", 0.06), ("ACUTE_MI", 0.03)):
                if rng.uniform() < p:
                    v, c = by_list[lid][int(rng.integers(0, len(by_list[lid])))]
                    dx_rows.append((subj, hid, 1, c, v))
            dx_rows.append((subj, hid, 9, "78900", 9))
        n_ecg = int(min(4, 1 + rng.poisson(0.7)))
        t = t_first
        last_t = t
        for e in range(n_ecg):
            sid += 1
            ed = rng.uniform() < 0.4
            has_h = rng.uniform() < 0.6 or (ed and rng.uniform() < 0.5)
            hadm = np.nan
            if has_h:
                hid += 1
                hadm = hid
                adm = t - pd.Timedelta(hours=float(rng.uniform(0, 30)))
                adm_rows.append((subj, hid, adm, adm + pd.Timedelta(days=float(rng.uniform(1, 9)))))
                lst = [("K_HYPERTONIE", 0.4), ("H_HERZINSUFFIZIENZ", 0.2 if hf else 0.02), ("NSTEMI", 0.04), ("ACUTE_MI", 0.04)]
                if amy:
                    lst += [("AMYLOID", 0.9), ("KARDIOMYOPATHIE_HINWEIS", 0.4)]
                for lid, p in lst:
                    if rng.uniform() < p:
                        v, c = by_list[lid][int(rng.integers(0, len(by_list[lid])))]
                        dx_rows.append((subj, hid, 1, c, v))
                dx_rows.append((subj, hid, 8, "78900", 9))
            icu = has_h and rng.uniform() < 0.1
            cc = None
            ed_stay = np.nan
            ed_in = pd.NaT
            if ed:
                ed_stay = 700000 + sid
                ed_in = t - pd.Timedelta(hours=float(rng.uniform(0.1, 3.0)))
                cc = rng.choice(["Chest pain", "CHEST PAIN", "Dyspnea", "Abd pain", "Chest pain, Dyspnea"], p=[0.25, 0.1, 0.25, 0.2, 0.2])
            qrs = float(np.clip(rng.normal(98, 15) + (10 if hf else 0), 70, 190))
            pr = float(np.clip(rng.normal(165, 28), 80, 320))
            qt = float(np.clip(rng.normal(405, 35), 300, 600))
            hr = float(np.clip(rng.normal(78, 14), 40, 150))
            rr = 60000.0 / hr
            af = rng.uniform() < 0.07
            lb = rng.uniform() < 0.03
            rb = rng.uniform() < 0.06
            pace = rng.uniform() < 0.01
            infm = rng.uniform() < 0.1
            usable = rng.uniform() < 0.97
            txt = ["Atrial fibrillation" if af else "Sinus rhythm"]
            if rb:
                txt.append("Right bundle branch block")
            if lb:
                txt.append("Left bundle branch block")
            if infm:
                txt.append("Anteroseptal infarct - age undetermined")
            if not (af or lb or rb or infm) and rng.uniform() < 0.5:
                txt.append("Normal ECG")
            else:
                txt.append("Abnormal ECG" if (af or lb or rb or infm) else "Borderline ECG")
            ccf.append(dict(subject_id=subj, study_id=sid, ecg_time=t, hadm_id=hadm, ed_stay_id=ed_stay, ed_intime=ed_in,
                            icustay_id=(800000 + sid) if icu else np.nan, icu_intime=t - pd.Timedelta(hours=1) if icu else pd.NaT,
                            icu_outtime=t + pd.Timedelta(days=2) if icu else pd.NaT, sex=("U" if sexu else ("F" if female else "M")), age=age0 + e // 3,
                            chief_complaint=cc, machine_heart_rate_bpm=hr, machine_pr_interval_ms=pr, machine_qrs_duration_ms=qrs,
                            machine_qt_interval_ms=qt, split=split))
            mrl.append(dict(subject_id=subj, study_id=sid, split=split, machine_metadata_usable=bool(usable),
                            report_af_state="positive" if af else "not_mentioned", report_lbbb_state="positive" if lb else "not_mentioned",
                            report_rbbb_state="positive" if rb else "not_mentioned", report_infarct_state="positive" if infm else "not_mentioned",
                            report_lvh_state="positive" if (lvh and rng.uniform() < 0.4) else "not_mentioned",
                            report_pacemaker_state="positive" if pace else "not_mentioned",
                            report_low_voltage_state="not_mentioned"))
            row = dict(subject_id=subj, study_id=sid, cart_id=1, ecg_time=str(t), rr_interval=rr)
            for j in range(18):
                row[f"report_{j}"] = txt[j] if j < len(txt) else None
            mm.append(row)
            crow = dict(study_id=sid, signal_quality_usable=bool(rng.uniform() < 0.97),
                        classical_pathological_q_waves_positive=bool(rng.uniform() < 0.12),
                        classical_rbbb_pattern_positive=bool(rb), classical_qrs_widening_ge_120=qrs + float(rng.normal(0, 3)),
                        pr_interval_ms=pr + float(rng.normal(0, 3)), context_machine_qrs_duration_ms=qrs, context_machine_pr_interval_ms=pr,
                        context_machine_qt_interval_ms=qt)
            for l in K.AMP_LEADS:
                rr_ = abs(rng.normal(0.9 if not lowv else 0.15, 0.25 if not lowv else 0.05))
                ss_ = abs(rng.normal(0.5 if not lowv else 0.1, 0.2 if not lowv else 0.04))
                if lvh and l in ("V5", "V6"):
                    rr_ += 1.3
                if lvh and l in ("V1", "V3"):
                    ss_ += 1.1
                if lowv and l.startswith("V"):
                    rr_ += 0.2
                crow[f"lead_measurement_{l}_qrs_r_height_mv"] = float(rr_)
                crow[f"lead_measurement_{l}_qrs_s_depth_mv"] = float(ss_)
            cache.append((split, crow))
            if rng.uniform() < 0.4:
                mid += 1
                dh = float(rng.choice([-1, 1]) * rng.uniform(1, 700)) if rng.uniform() < 0.7 else float(rng.uniform(-20, 20))
                echo_m.append(dict(subject_id=subj, study_id=sid, nearest_tte_measurement_id=mid, nearest_tte_delta_hours=dh))
                ivs = 1.0 + (0.4 if lvh else 0.0) + (0.4 if amy else 0.0) + float(rng.normal(0, 0.12))
                pw = 0.95 + (0.3 if lvh else 0.0) + float(rng.normal(0, 0.1))
                eff_s = None
                if eff:
                    eff_s = rng.choice(["Moderate (1.0-2.0cm)", "Small (<1.0cm)", "Large (>2.0cm)", "Small-moderate"])
                echo_e.append(dict(measurement_id=mid, subject_id=subj, preferred_lvef=float(np.clip(rng.normal(32 if hf else 57, 10), 10, 75)),
                                   septal_thickness_cm=ivs, posterior_wall_thickness_cm=pw,
                                   lv_internal_diameter_diastole_cm=float(rng.normal(4.8, 0.5)), echo_body_surface_area_m2=float(rng.normal(1.9, 0.2)),
                                   pericardial_effusion=eff_s, test_type="tte"))
            labf.append(dict(subject_id=subj, study_id=sid, troponin_count_24h=float(rng.poisson(2.0)) if ed else float(rng.poisson(0.3)),
                             potassium_nearest=float(rng.normal(4.1, 0.4)), potassium_delta_hours=float(rng.uniform(-30, 30)),
                             magnesium_nearest=float(rng.normal(2.0, 0.3)), magnesium_delta_hours=float(rng.uniform(-30, 30)),
                             total_calcium_nearest=float(rng.normal(9.2, 0.5)), total_calcium_delta_hours=float(rng.uniform(-30, 30))))
            if rng.uniform() < 0.8:
                mrow = dict(subject_id=subj, study_id=sid)
                for c in ["med_beta_blocker", "med_calcium_channel_blocker", "med_digoxin", "med_antiarrhythmic_class_i_iii", "med_diuretic", "med_qt_prolonging_drug"]:
                    mrow[c] = bool(rng.uniform() < 0.15)
                    mrow[c + "_home_reported"] = bool(rng.uniform() < 0.1)
                medf.append(mrow)

            def st(p):
                u = rng.uniform()
                return "positive" if u < p else ("negative" if u < 0.9 else "indeterminate")
            labs_w.append(dict(subject_id=subj, study_id=sid, split=split, vt_vf_icd_or_event_state=st(0.04), cardiac_arrest_state=st(0.02),
                               mortality_30d_state=st(0.05), acute_mi_probable_state=st(0.10), troponin_positive_any_state=st(0.25),
                               conduction_progression_1y_state=st(0.03), echo_rv_dilation_state=st(0.1), echo_pulmonary_hypertension_state=st(0.1), pericardial_effusion_state=st(0.03)))
            last_t = t
            t = t + pd.Timedelta(days=float(rng.uniform(5, 200)))
        dead = rng.uniform() < 0.12
        dod = (last_t + pd.Timedelta(days=float(rng.uniform(1, 500)))).normalize() if dead else pd.NaT
        af_t = last_t + pd.Timedelta(days=float(rng.uniform(-100, 400))) if rng.uniform() < 0.1 else pd.NaT
        lo = last_t + pd.Timedelta(days=float(rng.uniform(0, 600)))
        tl.append(dict(subject_id=subj, first_af_event_time=af_t, last_observed_time=lo,
                       vital_status_known_until=lo + pd.Timedelta(days=365), date_of_death=dod))
        pch.append(dict(subject_id=subj, first_pci_date=(t_first - pd.Timedelta(days=200)) if rng.uniform() < 0.04 else pd.NaT,
                        first_cabg_date=pd.NaT))
    pd.DataFrame(ccf).to_parquet(root / "evidence/clinical_context_features.parquet")
    pd.DataFrame(mrl).to_parquet(root / "evidence/machine_report_labels.parquet")
    pd.DataFrame(echo_m).to_parquet(root / "evidence/ecg_echo_matches.parquet")
    pd.DataFrame(echo_e).to_parquet(root / "evidence/echo_measurements_wide.parquet")
    pd.DataFrame(labf).to_parquet(root / "evidence/ecg_lab_features.parquet")
    pd.DataFrame(medf).to_parquet(root / "evidence/ecg_medication_features.parquet")
    pd.DataFrame(tl).to_parquet(root / "evidence/patient_clinical_timeline.parquet")
    pch_df = pd.DataFrame(pch)
    pch_df["first_pci_date"] = pd.to_datetime(pch_df["first_pci_date"]).dt.date
    pch_df["first_cabg_date"] = pd.to_datetime(pch_df["first_cabg_date"]).dt.date
    pch_df.to_parquet(root / "evidence/patient_code_history.parquet")
    lw = pd.DataFrame(labs_w)
    lw[lw["split"] != "locked_test"].to_parquet(root / "labels/clinical_labels_wide.parquet")
    lw[lw["split"] == "locked_test"].drop(columns=["split"]).to_parquet(root / "labels/sealed/locked_clinical_labels_wide.parquet")
    (root / "labels/label_definitions.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(mm).to_csv(root / "machine_measurements.csv", index=False)
    with gzip.open(root / "hosp/admissions.csv.gz", "wt", encoding="utf-8", newline="") as f:
        f.write("subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admit_provider_id,admission_location,discharge_location,insurance,language,marital_status,race,edregtime,edouttime,hospital_expire_flag\n")
        for s_, h_, a_, d_ in adm_rows:
            f.write(f"{s_},{h_},{a_:%Y-%m-%d %H:%M:%S},{d_:%Y-%m-%d %H:%M:%S},,URGENT,P1,ER,HOME,Other,English,SINGLE,WHITE,,,0\n")
    with gzip.open(root / "hosp/d_icd_diagnoses.csv.gz", "wt", encoding="utf-8", newline="") as f:
        f.write("icd_code,icd_version,long_title" + chr(10) + "0010,9,Cholera" + chr(10))
    with gzip.open(root / "hosp/diagnoses_icd.csv.gz", "wt", encoding="utf-8", newline="") as f:
        f.write("subject_id,hadm_id,seq_num,icd_code,icd_version\n")
        for r in dx_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    for s in ("train", "validation"):
        d = root / f"cache/split={s}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([c for sp, c in cache if sp == s]).to_parquet(d / "features.parquet")
    con = duckdb.connect(str(root / "counts.duckdb"))
    df = pd.DataFrame(ccf)
    df = df[df["split"] != "locked_test"][["subject_id", "study_id", "split"]].rename(columns={"split": "fold"})
    con.register("w", df)
    con.execute("CREATE TABLE ecg AS SELECT * FROM w")
    con.close()
    exams = []
    for i in range(3000):
        age = int(rng.integers(20, 90))
        rb, d1 = rng.uniform() < 0.07, rng.uniform() < 0.07
        dead = rng.uniform() < (0.03 * (1.5 if rb else 1.0))
        exams.append({"exam_id": i, "age": age, "is_male": bool(rng.uniform() < 0.4), "nn_predicted_age": age, "1dAVb": d1, "RBBB": rb,
                      "LBBB": rng.uniform() < 0.03, "SB": rng.uniform() < 0.03, "ST": rng.uniform() < 0.03, "AF": rng.uniform() < 0.03,
                      "patient_id": i, "death": float(dead), "timey": float(rng.uniform(0.5, 6)), "normal_ecg": bool(rng.uniform() < 0.5),
                      "trace_file": "x.hdf5"})
    pd.DataFrame(exams).to_csv(root / "exams.csv", index=False)
    return dict(K.CFG, root=str(root), counts_db=str(root / "counts.duckdb"), cache=str(root / "cache/split={s}/features.parquet"),
                machine_csv=str(root / "machine_measurements.csv"), hosp=str(root / "hosp"),
                sealed_labels=str(root / "labels/sealed/locked_clinical_labels_wide.parquet"),
                register=str(root / "TESTZUGRIFFE.md"), code15=str(root / "exams.csv"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("welt")
    return make_world(root), root


def test_prepare_dev_and_run_all_units_on_synthetic_world(world, tmp_path):
    cfg, root = world
    K.B_SENS = 10
    out = tmp_path / "aw236"
    summ = K.prepare(cfg, "dev", out, {})
    assert summ["n_index_main"] > 1000 and summ["n_index_ed"] > 30
    w = K.work_dir(out, "dev")
    d = K.Data(w, "main")
    assert d.n == summ["n_index_main"] and d.keys["split"].isin(["train", "validation"]).all()
    # Trennschutz: Regeltabelle enthaelt keine Referenzspalten
    assert not any(c.startswith(("lvh_30d", "lab_", "death_", "af_t", "peri_")) for c in d.rules.columns)
    assert any(c.startswith("lvh_") for c in d.refs.columns)
    assert d.keys["subject_id"].is_unique
    design = K.Design(d.cov)
    (out / "dev").mkdir(parents=True, exist_ok=True)
    (out / "dev" / "design_dev.json").write_text(json.dumps(design.state()), encoding="utf-8")
    for u in ("R02", "R03", "R04", "R06", "R07", "R14"):
        res = K.run_e(u, d, design, 99, "dev", 12, out)
        assert res["status"] == "ok", (u, res.get("status"))
    K.run_r20(K.Data(w, "ed"), 99, "dev", 30, out)
    for u in ("R09", "R18"):
        assert K.run_p(u, d, design, 99, "dev", out)["status"] == "ok"
    assert all(v["status"] == "ok" for v in K.run_r08(d, design, 99, "dev", out).values())
    assert K.run_r19(d, design, 99, "dev", 15, out)["status"] == "ok"
    units = K.assemble("dev", out, out / "pub")
    assert set(units) >= {"R02", "R03", "R04", "R06", "R07", "R14", "R20_A1", "R20_A2", "R09", "R18", "R08_VTVF", "R08_Tod", "R19"}
    txt = (out / "pub" / "results_dev_aggregate.json").read_text(encoding="utf-8")
    assert "subject_id" not in txt and "study_id" not in txt and "hadm_id" not in txt
    # Testmodus: ohne Registerzeile nicht startbar; mit Zeile: Spannungsregeln nicht verfuegbar, Maschinenregeln vorhanden
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    spec_h, script_h = K.SPEC_SHA256, G.sha256_file(K.__file__)
    man_h = G.sha256_file(out / "manifest.json")
    dev_h = K.dev_results_hash(out)
    ga = dict(register_path=cfg["register"], spec_hash=spec_h, script_hash=script_h, manifest_hash=man_h, dev_results_hash=dev_h)
    with pytest.raises(PermissionError):
        K.prepare(cfg, "test", out, ga)
    G.append_register(cfg["register"], G.register_line("Beginn", "AW-236", spec_h, script_h, man_h, dev_h, "locked_test"))
    # Testlabels ohne eine Spalte: kein Absturz, Spalte fehlt (v3)
    sl = pd.read_parquet(cfg["sealed_labels"])
    sl.drop(columns=["pericardial_effusion_state"]).to_parquet(cfg["sealed_labels"])
    K.prepare(cfg, "test", out, ga)
    wt = K.work_dir(out, "test")
    dt = K.Data(wt, "main")
    assert dt.keys["split"].eq("locked_test").all() and dt.rules["R02_gt"].isna().all() and dt.rules["R08_baz"].notna().any()
    designt = K.Design(dt.cov, json.loads((out / "dev" / "design_dev.json").read_text(encoding="utf-8")))
    rt = K.run_e("R02", dt, designt, 99, "test", 10, out)
    assert rt["status"].startswith("nicht auswertbar")
    pt = K.run_p("R09", dt, designt, 99, "test", out)
    assert pt["status"] == "ok" and any("nicht berechenbar" in h for h in pt["final"]["hinweise"])
    K.run_r08(dt, designt, 99, "test", out)
    K.run_r20(K.Data(wt, "ed"), 99, "test", 20, out)
    K.run_r19(dt, designt, 99, "test", 10, out)
    ut = K.assemble("test", out, out / "pub")
    assert ut["R09"]["replikation"]["urteil"] in ("repliziert", "Replikation abweichend", "nicht informativ (beide unentschieden)")


def test_code15_reproduction_on_synthetic_exams(world, tmp_path):
    cfg, root = world
    res = K.run_code15(cfg, 1, tmp_path)
    assert res["vorpruefung"]["n_patients"] == 3000
    r = res["ergebnisse"]["R09_RSB_primaer"]
    assert 0.2 < r["hr"] < 8 and r["klasse"] in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN)
    assert "R18_1dAVb_sens_normal_ecg" in res["ergebnisse"]


def test_cox_robust_matches_lifelines_with_ties():
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(5)
    n = 2500
    x1 = rng.normal(size=n)
    x2 = (rng.uniform(size=n) < 0.3).astype(float)
    expo = (rng.uniform(size=n) < 0.25).astype(float)
    lam = 0.08 * np.exp(0.4 * expo + 0.5 * x1 + 0.3 * x2)
    t = rng.exponential(1 / lam)
    c = rng.uniform(0.2, 3.0, n)
    T = np.maximum(np.round(np.minimum(t, c) * 365.25), 1)
    E = (t <= c).astype(float)
    b, se, info = K.cox_expo(T, E, expo, np.column_stack([x1, x2]), ["x1", "x2"])
    df = pd.DataFrame({"expo": expo, "x1": x1, "x2": x2, "T": T / 365.25, "E": E})
    cph = CoxPHFitter().fit(df, "T", "E", robust=True)
    s = cph.summary.loc["expo"]
    assert abs(b - s["coef"]) < 0.01 and abs(se - s["se(coef)"]) < 0.005


def test_logit_converges_without_ridge_on_large_sample():
    """v2: Konvergenz bei grossem n (Rundungsrauschen der Zielfunktion darf nicht zum Ridge-Rueckfall fuehren)."""
    rng = np.random.default_rng(9)
    n = 150000
    X = rng.normal(size=(n, 20))
    beta = rng.normal(0, 0.15, 21)
    Z = np.column_stack([np.ones(n), X])
    beta[0] = -1.6
    y = (rng.uniform(size=n) < K.expit(Z @ beta)).astype(float)
    c = K.bootstrap_counts(rng, n)
    b, info = K.fit_logit_safe(Z[c > 0], y[c > 0], c[c > 0])
    assert info["ridge"] is False and np.max(np.abs(b - beta)) < 0.1


def test_cox_with_rare_covariate_and_tied_times_is_finite():
    """v2: seltene Kovariate ohne Ereignisse (quasi-Separation) und viele Bindungen am Minimum fuehren nicht zu NaN."""
    rng = np.random.default_rng(4)
    n = 30000
    x1 = rng.normal(size=n)
    rare = (rng.uniform(size=n) < 0.0008).astype(float)
    expo = (rng.uniform(size=n) < 0.1).astype(float)
    t = np.where(rng.uniform(size=n) < 0.4, 0.5, rng.uniform(1, 365, n))
    e = ((rng.uniform(size=n) < 0.05) & (rare == 0)).astype(float)
    b, se, info = K.cox_expo(t, e, expo, np.column_stack([x1, rare]), ["x1", "rare"])
    assert np.isfinite(b) and np.isfinite(se) and se > 0, info


def test_cli_flow_dev_register_test_on_synthetic_world(world, tmp_path, monkeypatch):
    """Kommandozeile Ende zu Ende: manifest, prepare/run/assemble dev, register Beginn, prepare/run/assemble test, register Ende."""
    cfg, root = world
    cfgj = tmp_path / "cfg.json"
    cfgj.write_text(json.dumps({k: v for k, v in cfg.items() if isinstance(v, (str, int, float))}), encoding="utf-8")
    monkeypatch.setenv("K1_CFG_JSON", str(cfgj))
    out = tmp_path / "out"
    base = ["--out", str(out), "--no-checks", "--pub", str(out / "pub"), "--b-sens", "8", "--B", "12"]
    K.main(["manifest", "--out", str(out), "--no-checks"])
    assert (out / "manifest.json").exists()
    K.main(["prepare", "--mode", "dev"] + base)
    K.main(["run", "--mode", "dev", "--units", "R02,R20,R09,R08"] + base)
    K.main(["assemble", "--mode", "dev"] + base)
    with pytest.raises(PermissionError):
        K.main(["prepare", "--mode", "test"] + base)  # ohne Registerzeile
    K.main(["register", "--event", "Beginn"] + base)
    K.main(["prepare", "--mode", "test"] + base)
    K.main(["run", "--mode", "test", "--units", "R02,R20,R09,R08"] + base)
    K.main(["assemble", "--mode", "test"] + base)
    K.main(["register", "--event", "Ende"] + base)
    reg = Path(cfg["register"]).read_text(encoding="utf-8")
    assert "Beginn" in reg and "Ende" in reg and "E `" in reg
    assert (out / "pub" / "report_test_aggregate.md").exists()


# ----------------------------------------------------------------------------- v3: je Sol-Fehler und Astra-Muss ein Test
def test_v3_masking_covers_all_integer_outputs_and_complement():
    """Sol-Fehler 1: Abdeckungs- und Ausschlusszahlen werden maskiert; technische Zaehler bleiben."""
    res = {"abdeckung": {"n_population": 7, "n_basis": 1234, "n_lvh_positiv": 5432}, "bootstrap": {"B": 4000, "failed": 0},
           "dauer_s": 42, "judge": {"components": [{"estimate": 0.9, "nboot": 4000}]}}
    m = K.masked_copy(res)
    assert m["abdeckung"]["n_population"] == "<10" and m["abdeckung"]["n_basis"] == 1200 and m["abdeckung"]["n_lvh_positiv"] == 5400
    assert m["bootstrap"]["B"] == 4000 and m["dauer_s"] == 42 and m["judge"]["components"][0]["nboot"] == 4000
    ok = K.masked_copy({"abdeckung": {"n_population": 77, "n_basis": 1234}})
    assert ok["abdeckung"]["n_population"] == 80 and ok["abdeckung"]["n_basis"] == 1230


def test_v3_r14_requires_pacemaker_and_lbbb_exclusion_and_sensitivity_population():
    """Sol-Fehler 2 und 3."""
    idx = synth_idx(4)
    idx.loc[1, "report_pacemaker_state"] = "positive"
    idx.loc[2, "report_lbbb_state"] = "uncertain"
    idx = pd.concat([idx, K.machine_flags(idx)], axis=1)
    r = K.evaluate_rules(idx, True)
    assert r.loc[0, "R14_a155"] == 0
    assert np.isnan(r.loc[1, "R14_a155"]) and np.isnan(r.loc[2, "R14_a155"])  # Schrittmacher und unsicherer LSB schliessen aus
    d = synth_data(n=6000, seed=77)
    rf = d.refs
    base = K.r14_reference(rf)
    ap = K.r14_reference(rf, acute_pos=True)
    assert np.isfinite(ap).sum() > np.isfinite(base).sum()  # Indexinfarkte werden wieder aufgenommen
    K.B_SENS = 10
    s = K.e_sensitivity("R14", d, K.Design(d.cov), 1, "dev", 10, dict(ref14="acute_pos"), 14, "referenz_akuter_Infarkt_im_Index_positiv")
    p = K.build_ipw(K.e_setup("R14", d), d, K.Design(d.cov), do_kipp=False)
    assert s["n_pop"] > p.n


def test_v3_r14_reference_states_pci_alone_is_unknown():
    rf = pd.DataFrame({"prior_OLD_MI": [0, 1, 0, 0, 0], "prior_ACUTE_MI": [0, 0, 0, 0, 0], "prior_pci_cabg": [0, 0, 1, 0, 0],
                       "n_prior_dx": [2, 1, 1, 0, 1], "idx_ACUTE_MI": [0, 0, 0, 0, 1], "idx_OLD_MI": [0, 0, 0, 0, 0]})
    D = K.r14_reference(rf)
    assert D[0] == 0 and D[1] == 1 and np.isnan(D[2]) and np.isnan(D[3]) and np.isnan(D[4])  # nur PCI/CABG, ohne Vorgeschichte, Indexinfarkt
    assert K.r14_reference(rf, acute_pos=True)[4] == 1


def test_v3_af_risk_time_never_exceeds_last_contact():
    """Sol-Fehler 4: letzter Kontakt Tag 10, Tod Tag 100, kein VHF -> Risikozeit 10 Tage."""
    t, ev = K.af_risk_times([np.nan, 50.0, 5.0, np.nan], [10.0, 10.0, 10.0, 400.0], [100.0, np.nan, np.nan, 200.0], [True, False, False, True])
    assert abs(t[0] - 10.0) < 1e-9 and not ev[0]
    assert abs(t[1] - 10.0) < 1e-9 and not ev[1]  # VHF nach dem Kontaktende zaehlt nicht
    assert abs(t[2] - 5.0) < 1e-9 and ev[2]
    assert abs(t[3] - 200.0) < 1e-9 and not ev[3]  # konkurrierender Tod beendet die Risikozeit


def test_v3_first_hospital_day_is_calendar_day():
    """Sol-Fehler 5: Aufnahme Montag 23 Uhr, EKG Dienstag 22 Uhr liegt am zweiten Kalendertag."""
    a = pd.Series(pd.to_datetime(["2150-01-05 22:00", "2150-01-05 00:30"]))
    b = pd.Series(pd.to_datetime(["2150-01-04 23:00", "2150-01-05 23:50"]))
    d = K.calendar_day_diff(a, b)
    assert d[0] == 1.0 and d[1] == 0.0


def test_v3_r04_second_part_is_strict():
    """Sol-Fehler 6: Untere Schranke exakt 0 gilt nicht als 'haelt'."""
    assert K.classify_gt(0.0, 0.05, 0.0) == G.UNENTSCHIEDEN
    assert K.classify_gt(0.001, 0.05, 0.0) == G.HAELT
    assert K.classify_gt(-0.05, 0.0, 0.0) == G.HAELT_NICHT
    comps = K.e_setup("R04", synth_data(n=3000, seed=2))["comps"]
    assert [c["kind"] for c in comps] == ["ge", "gt"]


def test_v3_lr_plus_infinite_is_kept_and_zero_over_zero_is_undefined():
    """Sol-Fehler 7."""
    m = K.two_by_two_ipw(np.array([1, 0, 1, 0.0]), np.array([1, 0, 1, 0.0]), np.ones(4), np.ones(4))
    assert m["spec"] == 1.0 and m["lrp"] == float("inf")
    m0 = K.two_by_two_ipw(np.array([0, 0, 0, 0.0]), np.array([1, 0, 1, 0.0]), np.ones(4), np.ones(4))
    assert np.isnan(m0["lrp"])  # sens = 0 und spec = 1 -> 0/0
    arr = np.concatenate([np.full(200, np.inf), np.linspace(3, 9, 3800)])
    itv = K.interval(arr, 0.025, 4000)
    assert itv["hi"] == float("inf") and np.isfinite(itv["lo"])


def test_v3_failed_fits_are_counted_and_block_inference(monkeypatch):
    """Sol-Fehler 8: nicht konvergierte Anpassungen fliessen nicht ein und werden gezaehlt; zu viele ergeben keine Inferenz."""
    age, female, T, D, V, truth = make_selection_world(n=3000, seed=4)
    Dm = np.where(V == 1, D, np.nan)
    u = K.IPWUnit({"s": T}, Dm, V, basis_for(age, female), female, K.age_band(age), do_kipp=False)
    calls = {"n": 0}
    real = K.fit_logit_safe

    def flaky(*a, **k):
        calls["n"] += 1
        b, info = real(*a, **k)
        if calls["n"] > 1 and calls["n"] % 2 == 0:  # jede zweite Wiederholung scheitert
            info = dict(info, failed=True)
        return b, info

    monkeypatch.setattr(K, "fit_logit_safe", flaky)
    point, reps = K.run_bootstrap(u, np.random.default_rng(1), 40, every=1000)
    assert reps["_meta"]["failed"] > 10 and len(reps["s"]["sens"]) == 40 - reps["_meta"]["failed"]
    itv = K.interval(reps["s"]["sens"], 0.2, reps["_meta"]["B"])
    assert itv["keine_inferenz"] is True and np.isnan(itv["lo"]) and itv["n_undefiniert"] > 0  # 20 von 40 gueltig: keine Inferenz
    # 95-%-Regel isoliert: Randbesetzung reichlich (ac=0,025, B=1000), nur der Anteil gueltiger Wiederholungen entscheidet
    rg = np.random.default_rng(5)
    ok_vals = rg.normal(size=1000)
    low = ok_vals.copy()
    low[:100] = np.nan  # 90 % gueltig
    high = ok_vals.copy()
    high[:40] = np.nan  # 96 % gueltig
    assert K.interval(low, 0.025, 1000)["keine_inferenz"] is True
    assert K.interval(high, 0.025, 1000)["keine_inferenz"] is False
    # Punktschaetzung scheitert -> keine Inferenz
    monkeypatch.setattr(K, "fit_logit_safe", lambda *a, **k: (np.zeros(a[0].shape[1]), {"failed": True, "ridge": True}))
    point2, reps2 = K.run_bootstrap(u, np.random.default_rng(1), 5, every=1000)
    assert point2 is None and reps2["_meta"]["punktschaetzung_fehlgeschlagen"]


def test_v3_test_register_is_a_phase_sequence(tmp_path):
    """Sol-Fehler 9: nach 'Ende' ist derselbe Durchgang gesperrt; 'Fortsetzung' vor 'Ende' bleibt erlaubt."""
    reg = tmp_path / "TESTZUGRIFFE.md"
    reg.write_text("", encoding="utf-8")
    H = {k: c * 64 for k, c in zip("SCMD", "abcd")}
    ga = dict(register_path=reg, spec_hash=H["S"], script_hash=H["C"], manifest_hash=H["M"], dev_results_hash=H["D"])
    with pytest.raises(PermissionError):
        K.guard_phase("test", "locked_test", **ga)
    G.append_register(reg, G.register_line("Beginn", "AW", H["S"], H["C"], H["M"], H["D"], "x"))
    K.guard_phase("test", "locked_test", **ga)
    G.append_register(reg, G.register_line("Fortsetzung", "AW", H["S"], H["C"], H["M"], H["D"], "x"))
    K.guard_phase("test", "locked_test", **ga)
    assert K.test_phase(**ga) == "offen"
    G.append_register(reg, G.register_line("Ende", "AW", H["S"], H["C"], H["M"], H["D"], "x", result_hash="e" * 64))
    assert K.test_phase(**ga) == "abgeschlossen"
    with pytest.raises(PermissionError):
        K.guard_phase("test", "locked_test", **ga)
    with pytest.raises(PermissionError):
        K.part_sql("rel", "locked_test", "test", **ga)
    K.guard_phase("dev", "train")  # Entwicklung bleibt frei


def test_v3_r20_index_chosen_before_troponin_filter():
    """Sol-Verdacht 4: das erste EKG des geeigneten Aufenthalts bleibt Index, auch wenn nur ein spaeteres EKG eine Troponinzaehlung traegt."""
    t0 = pd.Timestamp("2150-03-01 08:00")
    base = pd.DataFrame({"subject_id": [1, 1, 2], "study_id": [10, 11, 20], "ecg_time": [t0 + pd.Timedelta(minutes=20), t0 + pd.Timedelta(hours=2), t0],
                         "ed_stay_id": [5.0, 5.0, 6.0], "ed_intime": [t0, t0, t0], "chief_complaint": ["Chest pain", "Chest pain", "Chest pain"],
                         "usable": [True, True, True]})
    trop = pd.DataFrame({"study_id": [10, 11, 20], "trop_count": [0.0, 3.0, 0.0]})
    idx = K.choose_index_ed(base, trop)
    assert list(idx["study_id"]) == [10]  # Person 2 hat im Aufenthalt keine Troponinmessung


def test_v3_unknown_sex_is_excluded_not_male(world, tmp_path):
    cfg, root = world
    out = tmp_path / "o"
    summ = K.prepare(cfg, "dev", out, {})
    assert summ["n_ecg_geschlecht_unbekannt_ausgeschlossen"] > 0
    d = K.Data(K.work_dir(out, "dev"), "main")
    ccf = pd.read_parquet(Path(cfg["root"]) / "evidence/clinical_context_features.parquet", columns=["subject_id", "sex"])
    unknown = set(ccf.loc[~ccf["sex"].isin(["M", "F"]), "subject_id"])
    assert len(unknown) > 0 and not (set(d.keys["subject_id"]) & unknown)  # ausgeschlossen, nicht als maennlich gewertet
    pc = summ["pruefungen_main"]
    assert pc["amplituden_minimum_mv"] >= 0 and 60 <= pc["qrs_eigen_median_ms"] <= 200
    # leeres Echo-Feld = unbekannt (strikt), als negativ nur in der Sensitivitaet
    assert np.isfinite(d.refs["periL_30d"]).sum() > np.isfinite(d.refs["peri_30d"]).sum()
    assert (d.refs["ecg_stay_day"].dropna() >= 0).mean() > 0.5


def test_v3_cache_contract_failure_stops_prepare():
    idx = synth_idx(4)
    idx["classical_qrs_widening_ge_120"] = [0.0, 1.0, 1.0, 0.0]  # Schwellenflag statt ms-Dauer
    with pytest.raises(SystemExit):
        K.plausibility_checks(idx)
    idx2 = synth_idx(4)
    idx2.loc[0, "lead_measurement_V1_qrs_s_depth_mv"] = -0.4  # Vorzeichenbruch
    with pytest.raises(SystemExit):
        K.plausibility_checks(idx2)
    assert K.plausibility_checks(synth_idx(4))["qrs_eigen_median_ms"] > 60


def test_v3_spline_knots_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        K.rcs_basis(np.linspace(0, 100, 10), [20, 40, 40, 80])
    with pytest.raises(ValueError):
        K.rcs_basis(np.linspace(0, 100, 10), [20, 40])


def test_v3_r19_requires_events_per_sex():
    rng = np.random.default_rng(3)
    n = 3000
    fem = (rng.uniform(size=n) < 0.5).astype(float)
    q = rng.normal(110, 20, n)
    D = np.where(fem == 1, (rng.uniform(size=n) < 0.002).astype(float), (rng.uniform(size=n) < 0.2).astype(float))
    knots = list(G.QRS_KNOTS)
    qs = K.rcs_basis(q, knots)
    inp = dict(q=q, D=D, V=np.ones(n, dtype=int), fem=fem, knots=knots, n=n, nq=4, conc_avail=False, Zv=np.ones((n, 2)), Zo=np.ones((n, 2)), Xo=np.zeros((n, 3)))
    out = K.r19_run_one(inp, np.random.default_rng(1), np.random.default_rng(2), 5, "t")
    assert out["status"] == "nicht schaetzbar" and out["ereignisse_je_geschlecht"]["F"] < K.R19_MIN_EVENTS_PER_SEX


def test_v31_concordance_uses_the_primary_weights_on_the_same_population():
    """Sol-Abweichung 1 / Punkt 8: unterschiedliche Varianten, Gewichte tatsaechlich aus der Primaerfassung uebernommen."""
    d = synth_data(n=12000, seed=12)
    setup = K.e_setup("R02", d)
    assert (setup["Tprim"]["s"] != setup["Tconc"]["s"]).sum() > 100  # die Varianten unterscheiden sich wirklich
    design = K.Design(d.cov)
    u = K.build_ipw(setup, d, design, with_conc=True, do_kipp=False)
    assert set(u.T) == {"s", "s__c"} and u.weights_from == {"s__c": "s"} and list(u.Z) == ["s"]
    c = np.ones(u.n)
    st = u.stats(c)
    # unabhaengige Nachrechnung mit den Gewichten des Primaermodells
    Z = u.Z["s"]
    bv, info = K.fit_logit_safe(Z, u.V, c)
    w = K.stabilised_weights(K.expit(Z @ bv), u.V, c)
    ref_c = K.two_by_two_ipw(u.T["s__c"], u.D, u.V, w * c)
    ref_p = K.two_by_two_ipw(u.T["s"], u.D, u.V, w * c)
    assert abs(st["s__c"]["sens"] - ref_c["sens"]) < 1e-9 and abs(st["s__c"]["spec"] - ref_c["spec"]) < 1e-9
    assert abs(st["s"]["sens"] - ref_p["sens"]) < 1e-9
    assert abs(st["s__c"]["sens"] - st["s"]["sens"]) > 1e-6  # unterschiedliche Varianten, unterschiedliche Kennzahlen
    # eine Neuanpassung des Gewichtsmodells fuer die Variante ergaebe andere Zahlen
    Zc = K.design(u.Xbase, u.T["s__c"], u.female, u.band)
    bc, _ = K.fit_logit_safe(Zc, u.V, c)
    wc = K.stabilised_weights(K.expit(Zc @ bc), u.V, c)
    assert abs(K.two_by_two_ipw(u.T["s__c"], u.D, u.V, wc * c)["sens"] - st["s__c"]["sens"]) > 1e-9


def test_v3_p_family_without_concordance_has_no_final_verdict(tmp_path):
    """Sol-Abweichung 2: fehlende Konkordanz -> Endurteil unentschieden."""
    d = synth_data(n=9000, seed=5, hr_effect=0.9)
    d.rules["R09_a155"] = np.nan  # Testpartition: Cache fehlt
    res = K.run_p("R09", d, K.Design(d.cov), 1, "test", tmp_path)
    assert res["final"]["klasse"] == G.UNENTSCHIEDEN and any("nicht berechenbar" in h for h in res["final"]["hinweise"])
    assert res["primaer"]["death"]["status"] == "ok"


def test_v3_r08_assessment_logic_refutation_only_if_upper_bound_not_above_one():
    """Astra Muss 1: [0,98; 1,20] widerlegt nicht."""
    assert K.classify_marker3(0.98, 1.20)[0] == G.UNENTSCHIEDEN
    assert K.classify_marker3(0.70, 1.00)[0] == G.HAELT_NICHT
    assert K.classify_marker3(1.05, 1.20) == (G.HAELT, "klein erhoeht (obere Schranke < 1,25)")
    assert K.classify_marker3(1.05, 1.60)[0] == G.HAELT


def test_v3_replication_requires_informative_agreement():
    """Astra Sollte 7."""
    mk = lambda c: {"final": {"klasse": c}, "family": "E", "judge": {"components": [{"id": "a", "estimate": 0.9, "lo95": 0.88, "hi95": 0.92}]}}  # noqa: E731
    both_u = K.replication_check(mk(G.UNENTSCHIEDEN), mk(G.UNENTSCHIEDEN))
    assert both_u["urteil"] == "nicht informativ (beide unentschieden)"
    assert K.replication_check(mk(G.HAELT), mk(G.HAELT))["urteil"] == "repliziert"
    assert K.replication_check(mk(G.HAELT), mk(G.HAELT_NICHT))["urteil"] == "Replikation abweichend"
    t = mk(G.HAELT)
    t["judge"]["components"][0].update(estimate=0.5, lo95=0.45, hi95=0.55)
    dd = K.replication_check(mk(G.HAELT), t)["effekt_differenz"]["a"]
    assert dd["vertraeglich"] is False and dd["differenz"] < 0


def test_v3_proportion_edge_case_uses_exact_bounds():
    """Astra Muss 6: null Ereignisse -> Clopper-Pearson statt [0; 0]."""
    x = np.zeros(200)
    est, reps = K.prop_boot(x, np.random.default_rng(1), 400)
    itv = K.prop_interval(x, reps, 0.0125, 400)
    assert itv["lo"] == 0.0 and 0.01 < itv["hi"] < 0.05 and "Clopper" in itv["randfall"]
    lo, hi = K.cp_bounds(200, 200, 0.0125)
    assert 0.95 < lo < 1.0 and hi == 1.0


def test_v3_unit_final_caps_when_selection_not_robust():
    """Astra Muss 4: nicht robust gegenueber den neun Szenarien -> Haupturteil eingeschraenkt."""
    judged = {"unit_class": G.HAELT, "kipp": {"_robust": False}, "components": []}
    final, hints = K.unit_final("R02", judged, {"unit_class": G.HAELT}, {"identifizierbarkeit_begrenzt": False}, {"B": 10, "failed": 0})
    assert final == G.UNENTSCHIEDEN and any("durch Selektion nicht gedeckt" in h for h in hints)
    judged["kipp"] = {"_robust": True}
    final2, _h = K.unit_final("R02", judged, {"unit_class": G.HAELT}, {"identifizierbarkeit_begrenzt": False}, {"B": 10, "failed": 0})
    assert final2 == G.HAELT
    final3, h3 = K.unit_final("R14", dict(judged, kipp=None), {"unit_class": G.HAELT}, {}, {"B": 10, "failed": 2})
    assert any("operationalisierungsabhaengig" in h for h in h3) and any("2 von 10" in h for h in h3)


def test_v3_code15_family_has_two_hypotheses(world, tmp_path):
    cfg, root = world
    res = K.run_code15(cfg, 1, tmp_path)
    assert abs(res["alpha_c"] - 0.0125) < 1e-12


def test_v3_weighted_secondary_odds_ratio_runs():
    d = synth_data(n=9000, seed=9)
    st = K.p_setup("R09", d)
    r = K.weighted_or(d, K.Design(d.cov), st["mask"], st["Tprim"], d.refs["lab_rv_dil"].values)
    assert r["status"] == "ok" and r["gewichtet"] is True and r["lo95"] < r["OR"] < r["hi95"]


def test_v3_monte_carlo_tail_rule():
    a = np.random.default_rng(1).normal(size=4000)
    assert K.interval(a, 0.003125, 4000)["keine_inferenz"] is False  # 12 Wiederholungen im Rand
    assert K.interval(a[:2500], 0.003125, 2500)["keine_inferenz"] is True  # nur 7 im Rand
    assert K.interval(a[:3000], 0.003125, 4000)["keine_inferenz"] is True  # weniger als 95 % gueltig


# ----------------------------------------------------------------------------- v3.1 (Sol-Diffpruefung, Punkt 8)
def test_v31_lr_plus_without_reference_negatives_is_undefined_not_infinite():
    """Sol-Punkt 1: ohne Referenznegative (spec nicht definiert) ist LR+ undefiniert, nicht +inf."""
    m = K.two_by_two_ipw(np.array([1, 1, 0.0]), np.array([1, 1, 1.0]), np.ones(3), np.ones(3))
    assert np.isnan(m["spec"]) and np.isnan(m["lrp"])
    mi = K.two_by_two_ipw(np.array([1, 0, 1, 0.0]), np.array([1, 0, 1, 0.0]), np.ones(4), np.ones(4))
    assert mi["lrp"] == float("inf")  # echte Spezifitaet 1 bleibt +inf


def test_v31_infinite_bound_is_classified_not_unentschieden():
    """Sol-Punkt 1: untere Schranke endlich, obere +inf -> Klassifikation moeglich (nur NaN blockiert)."""
    assert K.class_from("gt", 3.0, float("inf"), 2.0) == G.HAELT
    assert K.class_from("gt", float("nan"), float("inf"), 2.0) == G.UNENTSCHIEDEN


def test_v31_complete_bootstrap_failure_keeps_keys_and_blocks_inference(monkeypatch):
    """Sol-Punkt 8: alle Wiederholungen scheitern -> Schluessel vorhanden, Ausfaelle gezaehlt, keine Inferenz, kein Absturz."""
    age, female, T, D, V, _ = make_selection_world(n=3000, seed=6)
    u = K.IPWUnit({"s": T}, np.where(V == 1, D, np.nan), V, basis_for(age, female), female, K.age_band(age), do_kipp=False)
    real = K.fit_logit_safe
    calls = {"n": 0}

    def all_but_first(*a, **k):
        calls["n"] += 1
        b, info = real(*a, **k)
        return (b, info) if calls["n"] == 1 else (b, dict(info, failed=True))

    monkeypatch.setattr(K, "fit_logit_safe", all_but_first)
    point, reps = K.run_bootstrap(u, np.random.default_rng(3), 12, every=1000)
    assert point is not None and reps["_meta"]["failed"] == 12
    assert "sens" in reps["s"] and len(reps["s"]["sens"]) == 0 and "manski_sens_lo" in reps["s"]
    itv = K.interval(reps["s"]["sens"], 0.0125, 12)
    assert itv["keine_inferenz"] is True and np.isnan(itv["lo"])
    summ = K.summary_metrics(point, reps)
    assert summ["s"]["sens"]["est"] == point["s"]["sens"]


def test_v32_undefined_replicates_are_counted_including_manski_and_kipp(monkeypatch):
    """Sol-Punkt 2/5: tatsaechlich undefinierte Wiederholungen (Sensitivitaet, Worst-Case-, Kipppunktwerte) erscheinen in den Metadaten."""
    age, female, T, D, V, _ = make_selection_world(n=3000, seed=7)
    u = K.IPWUnit({"s": T}, np.where(V == 1, D, np.nan), V, basis_for(age, female), female, K.age_band(age), do_kipp=True)
    calls = {"tb": 0, "mk": 0, "cm": 0}
    real_tb, real_mk, real_cm = K.two_by_two_ipw, K.manski_bounds, K.completed_metrics

    def tb(*a, **k):
        calls["tb"] += 1
        m = real_tb(*a, **k)
        if calls["tb"] > 1 and calls["tb"] % 2 == 0:
            m["sens"] = float("nan")
        return m

    def mk(*a, **k):
        calls["mk"] += 1
        m = real_mk(*a, **k)
        if calls["mk"] > 1 and calls["mk"] % 2 == 0:
            m["sens_lo"] = float("nan")
        return m

    def cm(*a, **k):
        calls["cm"] += 1
        r = real_cm(*a, **k)
        return (float("nan"), r[1]) if calls["cm"] % 9 == 0 else r

    monkeypatch.setattr(K, "two_by_two_ipw", tb)
    monkeypatch.setattr(K, "manski_bounds", mk)
    monkeypatch.setattr(K, "completed_metrics", cm)
    point, reps = K.run_bootstrap(u, np.random.default_rng(4), 8, every=1000)
    und = reps["_meta"]["undefiniert_je_kennzahl"]
    assert und["s.sens"] > 0 and und["s.manski_sens_lo"] > 0
    assert any(k.startswith("s.kipp_sens_") for k in und)
    assert "s.spec" not in und and all(isinstance(v, int) and v > 0 for v in und.values())
    assert reps["_meta"]["B"] == 8 and reps["_meta"]["failed"] == 0


def test_v32_technical_counters_are_not_masked_and_person_counts_still_are():
    """Sol-Punkt 2: Ausfallzaehler (R19, Bootstrap-Metadaten) bleiben exakt, Personenzahlen werden maskiert."""
    res = {"n_pop": 1234, "n_risikozeit_null": 4, "bootstrap_fit_fehlgeschlagen": 7, "bootstrap_kein_eindeutiger_schnittpunkt": 3,
           "bootstrap": {"B": 1000, "failed": 2, "undefiniert_je_kennzahl": {"s.sens": 5, "s.kipp_sens_0.5|0.5": 3}}, "fit_fehler": 6}
    m = K.masked_copy(res)
    assert m["bootstrap_fit_fehlgeschlagen"] == 7 and m["bootstrap_kein_eindeutiger_schnittpunkt"] == 3 and m["fit_fehler"] == 6
    assert m["bootstrap"]["undefiniert_je_kennzahl"] == {"s.sens": 5, "s.kipp_sens_0.5|0.5": 3} and m["bootstrap"]["failed"] == 2
    assert m["n_risikozeit_null"] == "<10" and m["n_pop"] == 1200  # Komplementvergroeberung auf 100 wegen der kleinen Zelle


def _r19_world(n=6000, seed=21):
    d = synth_data(n=n, seed=seed, qrs_shift=14.0)
    return d, K.Design(d.cov)


def test_v32_r19_concordance_fit_failure_keeps_the_primary_estimate(monkeypatch):
    """Sol-Punkt 1: Anpassungsfehler nur der Konkordanzvariante (Punkt und Bootstrap) verwerfen die Primaerschaetzung nicht."""
    d, design = _r19_world()
    inp = K.r19_inputs(d, design)
    assert inp["conc_avail"]
    inp["Zoc"] = np.hstack([inp["Zoc"], np.zeros((inp["n"], 1))])
    wide = inp["Zoc"].shape[1]
    real = K.fit_logit_safe
    calls = {"wide": 0, "mode": "bootstrap"}

    def fail_conc(*a, **k):
        if a[0].shape[1] == wide:  # Modell der Konkordanzvariante (die zusaetzliche Nullspalte dient nur der Erkennung)
            b, info = real(a[0][:, :-1], *a[1:], **k)
            calls["wide"] += 1
            if calls["mode"] == "point" or calls["wide"] > 2:  # Punktschaetzung (zwei Modelle) gelingt, Wiederholungen scheitern
                info = dict(info, failed=True)
            return b, info
        return real(*a, **k)

    monkeypatch.setattr(K, "fit_logit_safe", fail_conc)
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    run = K.r19_run_one(inp, np.random.default_rng(1), np.random.default_rng(2), 6, "t")
    assert run["status"] == "ok" and run["p"] is not None and run["c"] is not None
    assert run["c"]["fit_fehler"] == 6 and run["p"]["fit_fehler"] == 0
    assert np.isnan(run["c"]["reps"]).all() and not np.isnan(run["p"]["reps"]).all()
    # Punktschaetzung der Konkordanz scheitert: Primaer bleibt, Konkordanz fehlt mit Kennzeichnung
    calls["mode"] = "point"
    run2 = K.r19_run_one(inp, np.random.default_rng(1), np.random.default_rng(2), 4, "t")
    assert run2["status"] == "ok" and run2["p"] is not None and run2["c"] is None and run2["c_punktschaetzung_fehlgeschlagen"] is True
    assert run2["p"]["fit_fehler"] == 0 and np.isfinite(run2["p"]["d0"])


def test_v32_run_r19_reports_primary_class_when_concordance_is_missing(tmp_path, monkeypatch):
    """Sol-Punkt 1: Primaerschaetzung und -klasse bleiben sichtbar, nur das Endurteil wird eingeschraenkt."""
    d, design = _r19_world(n=8000, seed=23)
    orig = K.r19_deltas

    def no_conc(*a, **k):
        out, warm = orig(*a, **k)
        if out:
            out["c"] = None
        return out, warm

    monkeypatch.setattr(K, "r19_deltas", no_conc)
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    monkeypatch.setattr(K, "B_SENS", 6)
    res = K.run_r19(d, design, 1, "dev", 12, tmp_path)
    pr, kc = res["varianten"]["primaer_Maschinenwert"], res["varianten"]["konkordanz_eigene_Dauer"]
    assert pr["status"] == "ok" and pr["klasse"] in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN) and "delta" in pr
    assert pr["delta_status"] == "ok" or pr["delta_status"].startswith("nicht pruefbar")
    assert pr["bootstrap_fit_fehlgeschlagen"] == 0
    assert kc["status"].startswith("nicht schaetzbar")
    assert res["final"]["klasse"] == G.UNENTSCHIEDEN and res["final"]["primaer"] == pr["klasse"]
    assert any("Konkordanzvariante nicht berechenbar" in h for h in res["final"]["hinweise"])


def test_v32_k2_results_and_observed_positive_rate_reach_output_and_report(tmp_path, monkeypatch):
    """Sol-Punkt 3: p_t_beobachtet und die geschlechtsgetrennten K2-Ergebnisse stehen in JSON und Zusatzbericht."""
    import k1_bericht as BR
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    monkeypatch.setattr(K, "B_SENS", 8)
    d = synth_data(n=9000, seed=19)
    res = K.run_e("R03", d, K.Design(d.cov), 1, "dev", 12, tmp_path)
    assert "p_t_beobachtet" in res["metrics"]["s"] and 0 < res["metrics"]["s"]["p_t_beobachtet"]["est"] < 1
    k2 = [v for k, v in res["sensitivitaeten"].items() if k.startswith("K2_")]
    assert k2 and all("geschlecht" in v and set(v["geschlecht"]) >= {"F", "M"} for v in k2 if v.get("status") == "ok")
    K.assemble("dev", tmp_path, None)
    txt = BR.build("dev", tmp_path)
    assert "beobachtet" in txt and "Geschlecht F" in txt and "Geschlecht M" in txt and "Differenz Frauen minus Männer" in txt


def test_v32_report_shows_absolute_risks_and_new_replication_field(tmp_path, monkeypatch):
    """Sol-Punkt 4: absolute Risiken erscheinen mit Werten; die Replikationszeile nutzt effekt_differenz statt des entfallenen Feldes."""
    import k1_bericht as BR
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    d = synth_data(n=9000, seed=5, hr_effect=0.9)
    design = K.Design(d.cov)
    for mode in ("dev", "test"):
        K.run_p("R09", d, design, 1, mode, tmp_path)
    units = K.assemble("test", tmp_path, None)
    assert "effekt_differenz" in units["R09"]["replikation"]
    txt = BR.build("test", tmp_path)
    line = [x for x in txt.splitlines() if "Absolutes 365-Tage-Risiko" in x]
    assert line and all("None" not in x and "n/a" not in x for x in line)
    assert "Effektdifferenz Test minus Entwicklung" in txt and "effekt_konsistent" not in txt


def test_v32_lr_plus_ge_branch_with_infinite_upper_bound():
    """Sol-Punkt 5: produktiver LR+-Zweig 'ge' (Hinweisurteil LR+ >= 5) mit unendlicher oberer Schranke."""
    assert K.class_from("ge", 6.0, float("inf"), 5.0) == G.HAELT
    assert K.class_from("ge", 3.0, float("inf"), 5.0) == G.UNENTSCHIEDEN
    assert K.class_from("ge", float("nan"), float("inf"), 5.0) == G.UNENTSCHIEDEN
    arr = np.concatenate([np.full(300, np.inf), np.linspace(6, 20, 3700)])
    itv = K.interval(arr, 0.025, 4000)
    assert itv["hi"] == float("inf") and K.class_from("ge", itv["lo"], itv["hi"], 5.0) == G.HAELT


def test_v32_changed_test_labels_stop_a_continued_prepare(tmp_path):
    """Sol-Punkt 5: geaenderter Hash der Testlabels bei Fortsetzung bricht prepare ab."""
    root = tmp_path / "w"
    root.mkdir()
    cfg = make_world(root)
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    spec_h, script_h = K.SPEC_SHA256, G.sha256_file(K.__file__)
    man_h, dev_h = G.sha256_file(out / "manifest.json"), K.dev_results_hash(out)
    ga = dict(register_path=cfg["register"], spec_hash=spec_h, script_hash=script_h, manifest_hash=man_h, dev_results_hash=dev_h)
    G.append_register(cfg["register"], G.register_line("Beginn", "AW-236", spec_h, script_h, man_h, dev_h, "locked_test"))
    K.prepare(cfg, "test", out, ga)
    assert (K.work_dir(out, "test") / "eingaben_test.json").exists()
    K.prepare(cfg, "test", out, ga)  # unveraenderte Testlabels: Fortsetzung erlaubt
    sl = pd.read_parquet(cfg["sealed_labels"])
    sl.drop(columns=[sl.columns[-1]]).to_parquet(cfg["sealed_labels"])
    with pytest.raises(SystemExit):
        K.prepare(cfg, "test", out, ga)


def test_v31_risk_time_has_no_floor_and_zero_risk_time_is_counted():
    """Sol-Punkt 3 / K-Entscheidung: Risikozeit = Minimum aus Kontaktende, Tod, 365 Tage; Risikozeit 0 wird gezaehlt und ausgeschlossen."""
    t, ev = K.af_risk_times([np.nan, np.nan, 0.2], [0.0, 3.0, 3.0], [np.nan, 0.0, np.nan], [False, True, False])
    assert t[0] == 0.0 and not ev[0] and t[1] == 0.0 and t[2] == 0.2 and ev[2]
    d = synth_data(n=6000, seed=11, hr_effect=0.5)
    st = K.p_setup("R09", d)
    d.refs["death_t"] = np.asarray(d.refs["death_t"].values, dtype=float).copy()
    pos = np.where(st["mask"] & np.isfinite(st["Tprim"]["s"] if isinstance(st["Tprim"], dict) else st["Tprim"]))[0][:7]
    d.refs.loc[d.refs.index[pos], "death_t"] = 0.0
    T = st["Tprim"]["s"] if isinstance(st["Tprim"], dict) else st["Tprim"]
    ok, tt, ee, nz = K.p_data(d, st["mask"], T, "death")
    assert nz >= 1 and not ok[pos].any() and np.all(tt[ok] > 0)
    r = K.absolute_risks(d, st["mask"], T, "death")
    assert r["n_risikozeit_null"] == nz


def test_v31_min_case_counts_restored_to_v1_v2_values():
    """K-Entscheidung zu Punkt 6: 100 vollstaendige Faelle (gewichtetes OR), R19 mit 200 (Primaer) bzw. 150 (Strata) Verifizierten."""
    d = synth_data(n=6000, seed=13)
    st = K.p_setup("R09", d)
    y = np.full(len(d.rules), np.nan)
    idx = np.where(st["mask"])[0]
    y[idx[:99]] = (np.arange(99) % 3 == 0).astype(float)
    assert K.weighted_or(d, K.Design(d.cov), st["mask"], st["Tprim"], y)["status"] == "nicht schaetzbar"
    n = 400
    rng = np.random.default_rng(3)
    fem = (np.arange(n) % 2).astype(float)
    V = np.zeros(n, dtype=int)
    V[:160] = 1
    D = (rng.uniform(size=n) < 0.4).astype(float)
    inp = dict(D=D, V=V, fem=fem, n=n)
    out = K.r19_run_one(inp, np.random.default_rng(1), np.random.default_rng(2), 3, "t")
    assert out["status"] == "nicht schaetzbar"  # 160 < 200 (Primaer)
    called = {}
    orig = K.r19_deltas
    K.r19_deltas = lambda *a, **k: called.setdefault("x", True) and (None, None)
    try:
        out2 = K.r19_run_one(inp, np.random.default_rng(1), np.random.default_rng(2), 3, "t", min_verified=150)
    finally:
        K.r19_deltas = orig
    assert called.get("x") and "Anpassung" in out2["status"]  # Schwelle 150 erreicht, erst die Anpassung scheitert


def test_v31_register_guard_is_phase_aware_also_for_direct_gerust_calls(tmp_path):
    """Sol-Punkt 7: auch G.check_test_register (direkte Geruestaufrufe) liefert nach 'Ende' falsch."""
    reg = tmp_path / "TESTZUGRIFFE.md"
    reg.write_text("", encoding="utf-8")
    H = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
    assert G.check_test_register(reg, *H) is False
    G.append_register(reg, G.register_line("Beginn", "AW", *H, "x"))
    assert G.check_test_register(reg, *H) is True
    G.append_register(reg, G.register_line("Ende", "AW", *H, "x", result_hash="e" * 64))
    assert G.check_test_register(reg, *H) is False


def test_v31_report_and_json_show_the_same_replication_verdict(tmp_path, monkeypatch):
    """Sol-Punkt 2: Der Bericht uebernimmt das Replikationsurteil aus dem JSON (replication_check), nicht eine eigene Rechnung."""
    monkeypatch.setattr(K, "MIN_TAIL_REPS", 0)
    d = synth_data()
    design = K.Design(d.cov)
    K.B_SENS = 10
    for mode in ("dev", "test"):
        K.run_e("R02", d, design, 1, mode, 20, tmp_path)
    units = K.assemble("test", tmp_path, tmp_path / "pub")
    rp = units["R02"]["replikation"]
    rep = (tmp_path / "pub" / "report_test_aggregate.md").read_text(encoding="utf-8")
    assert f"Entwicklung {rp['entwicklung']} / Test {rp['test']} -> {rp['urteil']}" in rep
    assert units["R02"]["replikation"] == json.loads((tmp_path / "pub" / "results_test_aggregate.json").read_text(encoding="utf-8"))["R02"]["replikation"]
