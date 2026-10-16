"""AW-248 A-R5-Kreis (nachtraeglich, post hoc): Wiederholbarkeit periodischer Marker mit kuerzestem Winkelabstand.

Ersetzt die in A_R5_CODEREVIEW_SOL.md als nicht tragfaehig bewertete Ersatz-ICC-Konstruktion (x + wrap(y - x)) durch:
 (1) Periode je Marker aus der Code-/Katalogdefinition (Tabelle PERIODEN unten, mit Fundstelle); 360 Grad = gerichteter Winkel,
     180 Grad = ungerichtete Achse; nicht periodische Marker werden ausgewiesen und nicht gewickelt.
 (2) Differenz s = Wicklung(y - x) auf (-P/2, P/2], Abstand |s| = kuerzester Winkelabstand mit richtiger Periode P.
     RC = empirisches 95-%-Quantil von |s| (inverse Verteilungsfunktion, gewichtet), Bootstrap ueber Personen
     (gemeinsame Gewichte der Hauptanalyse: 1.000 Ziehungen, Seed 77010); Abdeckung = Anteil |s| <= RC bzw. Anteil, den die
     parametrische Regel 1,96 x SD (um den linearen Mittelwert von s) tatsaechlich abdeckt.
     Bias = Kreismittel der Differenz s (Winkel mit Periode P, d.h. fuer P=180 auf den verdoppelten Winkel bezogen).
     Linearer ICC fuer periodische Marker: 'nicht definiert' (keine Ersatzkonstruktion).
 (3) Katalog-Median-ICC und Matrixklassenzaehlung zusaetzlich OHNE die periodischen Marker (Sensitivitaet); die Originalwerte
     bleiben die Hauptangabe (Gate: Originalwerte werden reproduziert).
Rohwerte (linearer ICC und RC der ungewickelten Differenz) werden nur als Datenidentitaets-Gate gegen aw149/all_metrics.csv
berechnet (wie in A-R5), nicht als Ergebnis ausgewiesen.
Daten: nur Entwicklung (train, validation). Nur Aggregate, keine Kennungen in den Ausgaben.
Aufruf: <local path> -I a_r5k.py test|run   (Stempel: h244.py stamp a_r5k.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aw244"))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import a_r4 as r4  # noqa: E402  (nur wstats, unveraendert; fuer das Datenidentitaets-Gate)

B = 1000
SEED = 77010
OUT = h244.AUSW / "nachrechnung" / "AW248"
CAT = h244.AUSW / "nachrechnung" / "transform_katalog.csv"
AM = Path(str(data_path("a.aw149_all_metrics_csv")))
KEY = ["subject_id", "split", "first_study_id", "second_study_id"]

BIO = "vcg_biomarker/core/biomarkers/"
GEO = "vcg_biomarker/core/vcg_geometry.py:122-128 (vcg_azimuth_deg = degrees(arctan2(Z, X)), Bereich (-180, 180])"
AZ_DIRECTION = "gerichteter Winkel (Richtungsvektor in der Transversalebene)"

# marker -> (Periode in Grad oder None, Art, Fundstelle). Perioden aus Code gelesen, nicht aus dem Katalogkennzeichen abgeleitet.
PERIODEN = {
    "p_area_vector_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v2_biomarkers.py:134-160 (_area_vector_mv_ms, azimuth = _azimuth(vector)), Segment 'p' ueber :1036-1060; _azimuth -> {BIO}catalog_biomarkers.py:675-676 -> {GEO}"),
    "qrs_area_vector_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v2_biomarkers.py:134-160, Segment 'qrs'; ebenso {BIO}circular_projection_biomarkers.py:89; {GEO}"),
    "st_area_vector_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v2_biomarkers.py:134-160, Segment 'st'; ebenso circular_projection_biomarkers.py:100; {GEO}"),
    "t_area_vector_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v2_biomarkers.py:134-160, Segment 't'; ebenso circular_projection_biomarkers.py:107; {GEO}"),
    "peak_svg_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v2_biomarkers.py:204-220 (_peak_svg_component, azimuth = _azimuth(vector)); circular_projection_biomarkers.py:88; {GEO}"),
    "svg_azimuth": (360, AZ_DIRECTION, f"{BIO}vcg_biomarkers.py:359-364 (SVGAzimuthBiomarker) mit :328 vcg_azimuth_deg; {GEO}"),
    "qrs_max_vector_azimuth": (360, AZ_DIRECTION, f"{BIO}vcg_biomarkers.py:406-411 (QRSMaxVectorAzimuthBiomarker) mit :386 vcg_azimuth_deg; {GEO}"),
    "t_max_vector_azimuth": (360, AZ_DIRECTION, f"{BIO}vcg_biomarkers.py:424-429 (TMaxVectorAzimuthBiomarker) mit :386 vcg_azimuth_deg; {GEO}"),
    "septal_activation_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}electrophysiology_markers.py:666-670 (arctan2(z, x) des initialen QRS-Vektors, Deklaration :2052-2059); circular_projection_biomarkers.py:84-85,99"),
    "st_injury_vector_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}electrophysiology_markers.py:913-924 (vcg_azimuth_deg des mittleren ST-Vektors, Deklaration :2188-2195); {GEO}"),
    "t_early_trajectory_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v3_biomarkers.py:695-711 (_t_phase_metrics, vcg_azimuth_deg), Spezifikation :1407; {GEO}"),
    "t_late_trajectory_azimuth_deg": (360, AZ_DIRECTION, f"{BIO}discovery_v3_biomarkers.py:695-711 (_t_phase_metrics, vcg_azimuth_deg), Spezifikation :1407; {GEO}"),
    "secondary_morphology_frontal_axis_deg": (360, "gerichteter Winkel (Frontalachse des mittleren QRS-Flaechenvektors)", f"{BIO}advanced_vcg_biomarkers.py:751-758 (degrees(arctan2(mean_y, mean_x)), Bereich (-180, 180]); Deklaration :3120-3127"),
    "delta_wave_frontal_axis_deg": (360, "gerichteter Winkel (Frontalachse der initialen QRS-Kraefte, hexaxiale Konvention I=0, aVF=+90)", f"{BIO}advanced_vcg_biomarkers.py:2141-2155 (degrees(arctan2(aVF, I)), Bereich (-180, 180]); Deklaration :3861-3870"),
}
for _a in ("initial", "terminal"):
    for _w in (10, 20, 30, 40):
        PERIODEN[f"qrs_{_a}_{_w}ms_force_azimuth_deg"] = (
            360, AZ_DIRECTION,
            f"{BIO}discovery_v3_biomarkers.py:560-585 (_window_vector, vcg_azimuth_deg des mittleren {_a}-Fensters), Spezifikation :1309; {GEO}")
for _o in (20, 40, 60, 80):
    PERIODEN[f"st_j_plus_{_o}ms_azimuth_deg"] = (
        360, AZ_DIRECTION,
        f"{BIO}discovery_v3_biomarkers.py:660-690 (_st_offset_vector, vcg_azimuth_deg), Spezifikation :1387; {GEO}")
PERIODEN["loop_ellipse_orientation_deg"] = (
    180, "ungerichtete Achse (Eigenvektor, Vorzeichen beliebig)",
    f"{BIO}catalog_biomarkers.py:2893-2898 (degrees(arctan2(vecs[1,0], vecs[0,0])) des ersten Eigenvektors) mit :697-706 (_pca, numpy eigh, Vorzeichen nicht festgelegt); Begruendung 180 Grad: {BIO}circular_projection_biomarkers.py:43-52 (12,8 % der Aufnahmen auf dem Gegenzweig)")
PERIODEN["classical_left_axis_deviation"] = (
    360, "gerichteter Winkel (Frontalachse, Geraetewert bevorzugt oder arctan2(aVF, I))",
    "vcg_biomarker/mimic/classical.py:2079-2093 (_frontal_axis_surrogate: Geraetewert in [-180, 180] oder degrees(arctan2(y, x))) und :787-797 (Kriteriumswert = qrs_axis); nicht im Katalog als periodisch gekennzeichnet")
PERIODEN["classical_right_axis_deviation"] = (
    360, "gerichteter Winkel (Frontalachse; Wert identisch mit classical_left_axis_deviation)",
    "vcg_biomarker/mimic/classical.py:2079-2093 und :776-785 (Kriteriumswert = qrs_axis); nicht im Katalog als periodisch gekennzeichnet")

# nicht periodisch (ausgewiesen)
NICHT_PERIODISCH = {
    "peak_area_qrs_azimuth_difference_deg": (
        "unsignierter kuerzester Winkelabstand, Bereich [0, 180]: min(|a-b| mod 360, 360 - |a-b| mod 360); 0 und 180 sind verschieden (parallel gegen antiparallel), keine Wicklungsnaht",
        f"{BIO}discovery_v4_biomarkers.py:405-411 (difference = abs(azimuth_peak - azimuth_area) % 360; min(difference, 360 - difference)); Deklaration :1713-1714"),
    "qrs_holonomy_planar_departure_deg": ("kumulierter Raumwinkel in Grad umgerechnet (|Holonomie| - 2 pi), Bereich ueber +-360 Grad hinaus; kein Winkel auf dem Kreis",
                                          f"{BIO}discovery_v4_biomarkers.py:866-868"),
    "qrst_holonomy_planar_departure_deg": ("wie oben", f"{BIO}discovery_v4_biomarkers.py:866-868"),
    "t_holonomy_planar_departure_deg": ("wie oben", f"{BIO}discovery_v4_biomarkers.py:866-868"),
    "qrs_t_area_angle_direct_8lead_tp_minus_pr_deg": ("Differenz zweier unsignierter QRS-T-Winkel in [0, 180]; linear, Bereich [-180, 180] ohne Wicklung",
                                                      f"{BIO}a108_robustness.py:398-399"),
    "qrs_t_integral_angle_direct_8lead_tp_minus_pr_deg": ("wie oben", f"{BIO}a108_robustness.py:398-399"),
}


# ------------------------------------------------------------------ Kernfunktionen
def wrap_period(d, P):
    """Wicklung der Differenz auf (-P/2, P/2]."""
    d = np.asarray(d, float)
    return P / 2.0 - np.mod(P / 2.0 - d, P)


def short_diff(x, y, P):
    return wrap_period(np.asarray(y, float) - np.asarray(x, float), P)


def wquantile_sorted(sorted_vals, cumw, q):
    """Inverse Verteilungsfunktion (Hyndman-Fan Typ 1): kleinster Wert mit gewichtetem Anteil >= q; cumw: Zeilen (c, n) oder (n,)."""
    cumw = np.atleast_2d(cumw)
    thr = q * cumw[:, -1] * (1 - 1e-12)
    idx = (cumw >= thr[:, None]).argmax(axis=1)
    return sorted_vals[idx]


def wquantile(v, w, q):
    order = np.argsort(v, kind="stable")
    return float(wquantile_sorted(np.asarray(v)[order], np.cumsum(np.asarray(w, float)[order]), q)[0])


def circ_mean_diff(s, w, P):
    """Kreismittel der Differenz (Winkel mit Periode P) in (-P/2, P/2]."""
    k = 2 * np.pi / P
    w = np.asarray(w, float)
    return float(np.arctan2((w * np.sin(k * s)).sum(), (w * np.cos(k * s)).sum()) / k)


def rbar(s, w, P):
    k = 2 * np.pi / P
    w = np.asarray(w, float)
    return float(np.hypot((w * np.sin(k * s)).sum(), (w * np.cos(k * s)).sum()) / w.sum())


def rule_cover(s, w):
    """Parametrische Regel 1,96 x SD (um den linearen Mittelwert von s, Haeufigkeitsgewichte): Anteil der Paare innerhalb der Regel, Regelwert."""
    w = np.asarray(w, float)
    tot = w.sum()
    m = (w * s).sum() / tot
    sd = np.sqrt((w * (s - m) ** 2).sum() / max(tot - 1, 1e-300))
    return float((w * (np.abs(s - m) <= 1.96 * sd)).sum() / tot), float(1.96 * sd), float(sd)


def point_stats(s, w, P):
    a = np.abs(s)
    rc = wquantile(a, w, 0.95)
    cov_emp = float((np.asarray(w, float) * (a <= rc)).sum() / np.asarray(w, float).sum())
    cov_rule, rule_val, sd = rule_cover(s, w)
    return {"bias": circ_mean_diff(s, w, P), "rc_q95": rc, "abdeckung_rc_q95": cov_emp, "abdeckung_regel_196sd": cov_rule,
            "regel_196sd": rule_val, "sd_linear": sd, "rbar": rbar(s, w, P),
            "median_abs": wquantile(a, w, 0.5)}


def boot_stats(s, wpair, pidx_sel, bootw, P, chunk=100):
    """Personenbootstrap: Ziehungs-Gewichte bootw[b, person] * Paargewicht; Rueckgabe dict Arrays (B,)."""
    a = np.abs(s)
    order = np.argsort(a, kind="stable")
    a_sorted = a[order]
    k = 2 * np.pi / P
    sn, cs = np.sin(k * s), np.cos(k * s)
    keys = ["bias", "rc_q95", "abdeckung_regel_196sd", "regel_196sd", "rbar", "median_abs"]
    res = {kk: np.empty(bootw.shape[0]) for kk in keys}
    for lo in range(0, bootw.shape[0], chunk):
        W = bootw[lo:lo + chunk][:, pidx_sel] * wpair[None, :]
        tot = W.sum(1)
        cw = np.cumsum(W[:, order], axis=1)
        res["rc_q95"][lo:lo + chunk] = wquantile_sorted(a_sorted, cw, 0.95)
        res["median_abs"][lo:lo + chunk] = wquantile_sorted(a_sorted, cw, 0.5)
        S, C = W @ sn, W @ cs
        res["bias"][lo:lo + chunk] = np.arctan2(S, C) / k
        res["rbar"][lo:lo + chunk] = np.hypot(S, C) / tot
        m = (W @ s) / tot
        sd = np.sqrt(((W * (s[None, :] - m[:, None]) ** 2).sum(1)) / np.maximum(tot - 1, 1e-300))
        res["regel_196sd"][lo:lo + chunk] = 1.96 * sd
        res["abdeckung_regel_196sd"][lo:lo + chunk] = (W * (np.abs(s[None, :] - m[:, None]) <= 1.96 * sd[:, None])).sum(1) / tot
    return res


def catalog_sensitivity(am_icc, cat_class, excluded, pop):
    """Katalog-Median-ICC (schaetzbare Marker) mit und ohne die ausgeschlossene Menge."""
    base = am_icc.dropna()
    keep = base.drop(index=[m for m in excluded if m in base.index])
    return {"population": pop, "markers_schaetzbar": int(base.size), "median_icc": float(base.median()),
            "anteil_ge_075": float((base >= .75).mean()), "zahl_ge_075": int((base >= .75).sum()),
            "ausgeschlossen_schaetzbar": int(base.size - keep.size), "markers_schaetzbar_ohne": int(keep.size),
            "median_icc_ohne": float(keep.median()), "anteil_ge_075_ohne": float((keep >= .75).mean()),
            "zahl_ge_075_ohne": int((keep >= .75).sum())}


def class_counts(cat, excluded):
    cl = cat.set_index("marker_id")["class"]
    out = {}
    for name, sel in (("original", cl), ("ohne", cl.drop(index=[m for m in excluded if m in cl.index]))):
        vc = sel.value_counts()
        dep, cond = int(vc.get("transformabhaengig", 0)), int(vc.get("bedingt", 0))
        out[name] = {"abhaengig": dep, "bedingt": cond, "stabil": int(vc.get("stabil", 0)),
                     "nicht_schaetzbar": int(vc.get("nicht_schaetzbar", 0)), "konstant_identisch": int(vc.get("konstant_identisch", 0)),
                     "nicht_pruefbar": int(vc.get("nicht_pruefbar", 0)), "summe_abhaengig_bedingt": dep + cond,
                     "anteil_abhaengig": dep / (dep + cond) if dep + cond else float("nan"),
                     "anteil_bedingt": cond / (dep + cond) if dep + cond else float("nan")}
    return out


# ------------------------------------------------------------------ Test an Kunstdaten
def test():
    h244.start_watch()
    # 1 Wicklung
    assert np.allclose(wrap_period([-190, 190, 180, -180, 0, 540, 725], 360), [170, -170, 180, 180, 0, 180, 5])
    assert np.allclose(wrap_period([100, -90, 90, 270, 180, 0], 180), [-80, 90, 90, 90, 0, 0])
    # 2 gewichtetes Quantil == Quantil der wiederholten Stichprobe
    rng = np.random.default_rng(1)
    for n in (37, 101, 1000):
        v = rng.normal(size=n)
        w = rng.integers(0, 4, n) + 0.0
        w[0] = 1
        ref = np.quantile(np.repeat(v, w.astype(int)), 0.95, method="inverted_cdf")
        assert abs(wquantile(v, w, 0.95) - ref) < 1e-12, (n, wquantile(v, w, 0.95), ref)
    # 3 exakt antipodale Paare: RC = 180, |Bias| = 180 (Kreismittel), lineares Mittel waere fehlleitend
    n = 4000
    x = rng.uniform(-180, 180, n)
    y = wrap_period(x + 180.0, 360)          # Bereich (-180, 180]
    s = short_diff(x, y, 360)
    assert np.abs(s).min() > 179.999999
    st = point_stats(s, np.ones(n), 360)
    assert abs(st["rc_q95"] - 180) < 1e-5 and abs(abs(st["bias"]) - 180) < 1e-5 and st["rbar"] > 0.999999
    # antipodal mit Rauschen, Vorzeichen von s wechselt zwischen +-180
    y2 = wrap_period(x + 180.0 + rng.normal(0, 0.5, n), 360)
    s2 = short_diff(x, y2, 360)
    assert (s2 > 0).mean() > 0.4 and (s2 < 0).mean() > 0.4, "Test braucht beide Vorzeichen bei +-180"
    st2 = point_stats(s2, np.ones(n), 360)
    assert abs(abs(st2["bias"]) - 180) < 0.2 and abs(s2.mean()) < 30 and st2["rc_q95"] > 179
    # 4 ungerichtete 180-Grad-Achsen: mit P=180 kleiner Fehler, mit P=360 grosser Fehler
    sig = 3.0
    k = rng.integers(0, 2, n)
    noise = rng.normal(0, sig, n)
    y3 = wrap_period(x + 180.0 * k + noise, 360)
    s180 = short_diff(x, y3, 180)
    s360 = short_diff(x, y3, 360)
    ref180 = np.quantile(np.abs(noise), 0.95, method="inverted_cdf")
    st180, st360 = point_stats(s180, np.ones(n), 180), point_stats(s360, np.ones(n), 360)
    assert abs(st180["rc_q95"] - ref180) < 1e-9, (st180["rc_q95"], ref180)
    assert abs(st180["bias"] - noise.mean()) < 0.3 and abs(st180["bias"]) < 0.3
    assert st360["rc_q95"] > 170
    # 5 Naht bei +-180: kleine Abweichung, Verteilung um 175 Grad
    x4 = wrap_period(rng.normal(175, 15, n), 360)
    noise4 = rng.normal(2.0, 4.0, n)
    y4 = wrap_period(x4 + noise4, 360)
    assert (np.abs(y4 - x4) > 180).sum() > 100, "Test braucht Spruenge ueber die Naht"
    s4 = short_diff(x4, y4, 360)
    st4 = point_stats(s4, np.ones(n), 360)
    assert np.allclose(s4, noise4, atol=1e-9)
    assert abs(st4["rc_q95"] - np.quantile(np.abs(noise4), 0.95, method="inverted_cdf")) < 1e-9
    assert abs(st4["bias"] - noise4.mean()) < 0.01
    # 6 Abdeckung der Regel 1,96 SD: normal etwa 0,95; Mischung mit 10 % antipodalen Paaren deutlich darunter, RC_q95 = 180
    assert abs(st4["abdeckung_regel_196sd"] - 0.95) < 0.015
    mix = np.where(rng.random(n) < 0.10, 180.0, rng.normal(0, 3, n))
    y5 = wrap_period(x + mix, 360)
    st5 = point_stats(short_diff(x, y5, 360), np.ones(n), 360)
    assert 0.86 < st5["abdeckung_regel_196sd"] < 0.93 and st5["rc_q95"] > 175 and st5["abdeckung_rc_q95"] >= 0.95
    # 7 Personenbootstrap: Intervall enthaelt den wahren RC (sigma 10 -> 19.6), Zeilen-Gewichte konsistent
    npers = 500
    pid = np.repeat(np.arange(npers), rng.integers(1, 4, npers))
    nn = len(pid)
    xx = rng.uniform(-180, 180, nn)
    nz = rng.normal(0, 10, nn)
    sp = short_diff(xx, wrap_period(xx + nz, 360), 360)
    bw = np.empty((300, npers))
    for b in range(300):
        bw[b] = np.bincount(rng.integers(0, npers, npers), minlength=npers)
    # Punktschaetzer aus Bootstrapfunktion mit Einheitsgewichten == Punktschaetzer
    bs1 = boot_stats(sp, np.ones(nn), pid, np.ones((1, npers)), 360)
    ps = point_stats(sp, np.ones(nn), 360)
    assert abs(bs1["rc_q95"][0] - ps["rc_q95"]) < 1e-12 and abs(bs1["bias"][0] - ps["bias"]) < 1e-9
    assert abs(bs1["abdeckung_regel_196sd"][0] - ps["abdeckung_regel_196sd"]) < 1e-12 and abs(bs1["rbar"][0] - ps["rbar"]) < 1e-12
    bs = boot_stats(sp, np.ones(nn), pid, bw, 360)
    lo, hi = h244.ci(bs["rc_q95"], minn=285)
    assert lo < 19.6 < hi, (lo, hi)
    # Gewichte: Bootstrap mit Ziehung == Quantil der wiederholten Paare
    b0 = bw[0]
    rep = np.repeat(np.arange(nn), b0[pid].astype(int))
    assert abs(boot_stats(sp, np.ones(nn), pid, b0[None, :], 360)["rc_q95"][0] - np.quantile(np.abs(sp)[rep], 0.95, method="inverted_cdf")) < 1e-12
    # 7b Zufallsreferenz: unabhaengige Gleichverteilung -> Q95 etwa 171 (P=360), 85,5 (P=180)
    xu, yu = rng.uniform(-180, 180, 20000), rng.uniform(-180, 180, 20000)
    assert abs(wquantile(np.abs(short_diff(xu, yu, 360)), np.ones(20000), 0.95) - 171) < 3
    assert abs(wquantile(np.abs(short_diff(xu, yu, 180)), np.ones(20000), 0.95) - 85.5) < 2
    # 8 Katalog-Sensitivitaet
    base = pd.Series([0.1, 0.8, 0.5, np.nan, 0.3, 0.9], index=list("abcdef"))
    r = catalog_sensitivity(base, None, ["a", "f", "zz"], "t")
    assert r["markers_schaetzbar"] == 5 and abs(r["median_icc"] - 0.5) < 1e-12 and r["ausgeschlossen_schaetzbar"] == 2
    assert r["markers_schaetzbar_ohne"] == 3 and abs(r["median_icc_ohne"] - 0.5) < 1e-12 and r["zahl_ge_075_ohne"] == 1 and r["zahl_ge_075"] == 2
    cat = pd.DataFrame({"marker_id": list("abcdef"), "class": ["transformabhaengig", "bedingt", "transformabhaengig", "stabil", "bedingt", "transformabhaengig"]})
    cc = class_counts(cat, ["a", "e"])
    assert cc["original"]["abhaengig"] == 3 and cc["original"]["bedingt"] == 2 and cc["ohne"]["abhaengig"] == 2 and cc["ohne"]["bedingt"] == 1
    print("TEST_OK", round(st180["rc_q95"], 3), round(st360["rc_q95"], 1), round(st2["bias"], 2), round(st4["rc_q95"], 3), round(st5["abdeckung_regel_196sd"], 3), (round(lo, 2), round(hi, 2)))


# ------------------------------------------------------------------ Lauf
def run():
    import pyarrow.parquet as pq
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("a_r5k")
    h244.start_watch()
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(CAT)
    adm = cat[cat.admitted]
    flagged = adm[adm.train_wrapped_rule == "Release_Signed_Grad_oder_atan2"].marker_id.tolist()
    assert len(flagged) == 27, len(flagged)
    for m in flagged:
        assert m in PERIODEN or m in NICHT_PERIODISCH, m
    periodic_ids = [m for m in PERIODEN if m in flagged] + ["loop_ellipse_orientation_deg", "classical_left_axis_deviation", "classical_right_axis_deviation"]
    assert set(PERIODEN) == set(periodic_ids), set(PERIODEN) ^ set(periodic_ids)
    assert sum(m in flagged for m in PERIODEN) == 26 and "peak_area_qrs_azimuth_difference_deg" in flagged and "peak_area_qrs_azimuth_difference_deg" in NICHT_PERIODISCH
    # Mengen
    set_auftrag = flagged + ["loop_ellipse_orientation_deg"]                      # 27 gekennzeichnete + Ellipsenmarker = 28
    set_belegt = list(PERIODEN)                                                     # 26 belegt periodisch (360) + Ellipse (180) + 2 Frontalachsen (360) = 29
    assert len(set_auftrag) == 28 and len(set_belegt) == 29
    # Daten
    z = pd.read_parquet(h244.MA / "repeatability" / "paargruppen_privat.parquet", columns=KEY + ["stable"])
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    rng = np.random.default_rng(SEED)
    bootw = np.empty((B, len(ids)))
    for b in range(B):
        bootw[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    pops = {"gesamt": np.ones(len(z)), "proxy": z.stable.to_numpy().astype(float)}
    am = pd.read_csv(AM)
    parts = sorted((h244.RUN / "repeatability" / "replicate_pairs").glob("part-*.parquet"))
    rows, gate_rows, seen = [], [], set()
    for part in parts:
        pcols = pq.ParquetFile(part).schema_arrow.names
        have = [m for m in PERIODEN if m + "_first" in pcols and m + "_second" in pcols and m not in seen]
        if not have:
            continue
        f = pd.read_parquet(part, columns=KEY + [m + s for m in have for s in ("_first", "_second")])
        f = f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]))
        for m in have:
            seen.add(m)
            P = PERIODEN[m][0]
            x = f[m + "_first"].to_numpy(float)
            y = f[m + "_second"].to_numpy(float)
            pops_done = []
            for pn, w in pops.items():
                pops_done.append(pn)
                sel = (w > 0) & np.isfinite(x) & np.isfinite(y)
                nper = int(len(np.unique(pidx[sel])))
                # Datenidentitaets-Gate (roh, nur Pruefung)
                raw, _, nsr, nnr = r4.wstats(x[:, None], y[:, None], pidx, len(ids), w, None)
                a = am[(am.population == pn) & (am.marker_id == m)]
                g = {"marker_id": m, "population": pn, "icc_roh_gate": float(raw[0, 0]), "rc_roh_gate": float(raw[0, 3]),
                     "icc_all_metrics": float(a.icc.iloc[0]) if len(a) else float("nan"), "rc_all_metrics": float(a.rc.iloc[0]) if len(a) else float("nan")}
                gate_rows.append(g)
                row = {"marker_id": m, "population": pn, "periode_grad": P, "n_paare": int(sel.sum()), "n_personen": nper,
                       "icc_linear": "nicht definiert"}
                if nper < 30 or sel.sum() < 2:
                    rows.append(row)
                    continue
                s = short_diff(x[sel], y[sel], P)
                wp = w[sel]
                ps = point_stats(s, wp, P)
                bs = boot_stats(s, wp, pidx[sel], bootw, P)
                row.update(ps)
                # Zufallsreferenz (beschreibend): erste und zweite Werte zufaellig neu gepaart, 20 Permutationen, Median
                prng = np.random.default_rng(np.random.SeedSequence([248010, list(PERIODEN).index(m), len(pops_done)]))
                zq, zm = [], []
                for _ in range(20):
                    sperm = short_diff(x[sel], prng.permutation(y[sel]), P)
                    zq.append(wquantile(np.abs(sperm), wp, 0.95))
                    zm.append(wquantile(np.abs(sperm), wp, 0.5))
                row["rc_q95_zufallspaarung"], row["median_abs_zufallspaarung"] = float(np.median(zq)), float(np.median(zm))
                for kk in ("bias", "rc_q95", "abdeckung_regel_196sd", "regel_196sd", "rbar", "median_abs"):
                    lo, hi = h244.ci(bs[kk], minn=int(0.95 * B))
                    row[kk + "_lo"], row[kk + "_hi"] = lo, hi
                row["anteil_abs_ge_haelfte_max"] = float((np.abs(s) >= P / 4).mean())   # Anteil mit Abstand >= 90 (P=360) bzw. 45 (P=180) Grad
                rows.append(row)
        del f
        h244.guard("a_r5k")
        print("Block", part.name, flush=True)
    missing = [m for m in PERIODEN if m not in seen]
    gate = pd.DataFrame(gate_rows)
    gate["abw_icc"] = (gate.icc_roh_gate - gate.icc_all_metrics).abs()
    gate["abw_rc_rel"] = (gate.rc_roh_gate - gate.rc_all_metrics).abs() / np.maximum(gate.rc_all_metrics.abs(), 1e-12)
    ok = gate.dropna(subset=["abw_icc"])
    assert len(ok) > 0 and (ok.abw_icc < 1e-9).all() and (ok.abw_rc_rel < 1e-9).all(), ok.sort_values("abw_icc").tail()
    rep = pd.DataFrame(rows)
    # Periodentabelle
    per_rows = []
    for m, (P, art, src) in PERIODEN.items():
        per_rows.append({"marker_id": m, "im_katalog_als_periodisch_gekennzeichnet": m in flagged, "im_auftrag": m in set_auftrag, "periodisch": True,
                         "periode_grad": P, "art": art, "fundstelle": src, "wiederholungsdaten_vorhanden": m in seen})
    for m, (art, src) in NICHT_PERIODISCH.items():
        per_rows.append({"marker_id": m, "im_katalog_als_periodisch_gekennzeichnet": m in flagged, "im_auftrag": m in set_auftrag, "periodisch": False,
                         "periode_grad": None, "art": art, "fundstelle": src, "wiederholungsdaten_vorhanden": None})
    per = pd.DataFrame(per_rows)
    # Zusammenfassung
    summ = []
    for pn in ("gesamt", "proxy"):
        for lab, ids_ in (("360 Grad gerichtet (26 gekennzeichnete + 2 klassische Frontalachsen)", [m for m, v in PERIODEN.items() if v[0] == 360]),
                          ("360 Grad gerichtet, nur die 26 aus dem Katalog gekennzeichneten", [m for m, v in PERIODEN.items() if v[0] == 360 and m in flagged]),
                          ("180 Grad Achse", [m for m, v in PERIODEN.items() if v[0] == 180])):
            d = rep[(rep.population == pn) & rep.marker_id.isin(ids_)].dropna(subset=["rc_q95"])
            if not len(d):
                continue
            summ.append({"population": pn, "gruppe": lab, "n_marker": int(len(d)), "rc_q95_min": float(d.rc_q95.min()), "rc_q95_median": float(d.rc_q95.median()),
                         "rc_q95_max": float(d.rc_q95.max()), "bias_min": float(d.bias.min()), "bias_max": float(d.bias.max()),
                         "abdeckung_regel_min": float(d.abdeckung_regel_196sd.min()), "abdeckung_regel_max": float(d.abdeckung_regel_196sd.max()),
                         "regel_196sd_min": float(d.regel_196sd.min()), "regel_196sd_max": float(d.regel_196sd.max()),
                         "rbar_min": float(d.rbar.min()), "rbar_max": float(d.rbar.max())})
    summ = pd.DataFrame(summ)
    # Katalog-Sensitivitaet
    sens, cls = [], []
    for lab, excl in (("A_Auftragsmenge_28", set_auftrag), ("B_belegt_periodisch_29", set_belegt)):
        for pn in ("gesamt", "proxy"):
            icc = am[am.population == pn].set_index("marker_id")["icc"]
            r = catalog_sensitivity(icc, None, excl, pn)
            r["ausgeschlossene_menge"] = lab
            r["n_ausgeschlossen_gesamt"] = len(excl)
            sens.append(r)
        cc = class_counts(cat, excl)
        for k in ("original", "ohne"):
            cls.append({"ausgeschlossene_menge": lab, "variante": k, **cc[k]})
    sens = pd.DataFrame(sens)
    cls = pd.DataFrame(cls)
    # Gate Originalwerte (Hauptangabe): Median 0,46652 / 0,52421; 1.071 abhaengig / 684 bedingt
    g1 = sens[sens.population == "gesamt"].iloc[0]
    g2 = sens[sens.population == "proxy"].iloc[0]
    assert abs(g1.median_icc - 0.46652) < 5e-6 and abs(g2.median_icc - 0.52421) < 5e-6 and g1.markers_schaetzbar == 1987 and g2.markers_schaetzbar == 1974
    assert int(cls.iloc[0].abhaengig) == 1071 and int(cls.iloc[0].bedingt) == 684, cls.iloc[0].to_dict()
    # Ausgabe
    per.to_csv(OUT / "a_r5k_perioden.csv", index=False)
    rep.to_csv(OUT / "a_r5k_wiederholbarkeit.csv", index=False)
    summ.to_csv(OUT / "a_r5k_zusammenfassung.csv", index=False)
    sens.to_csv(OUT / "a_r5k_katalog_icc.csv", index=False)
    cls.to_csv(OUT / "a_r5k_matrixklassen.csv", index=False)
    gate.to_csv(OUT / "a_r5k_gate_daten.csv", index=False)
    meta = {"skript_sha256": shash, "a_r4_sha256": h244.sha(Path(r4.__file__)), "h244_sha256": h244.sha(Path(h244.__file__)),
            "katalog_sha256": h244.sha(CAT), "all_metrics_sha256": h244.sha(AM), "B": B, "seed": SEED, "marker_ohne_wiederholungsdaten": missing,
            "gate_max_abw_icc": float(ok.abw_icc.max()), "gate_max_abw_rc_rel": float(ok.abw_rc_rel.max()), "gate_n": int(len(ok)), "peak_rss": h244.peak()}
    h244.dump(OUT / "a_r5k_meta.json", meta)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    txt = ["A-R5-Kreis (AW-248, post hoc, nur Entwicklung)", f"Skript-SHA-256 {shash}", "Gate", json.dumps(meta, indent=1, default=str),
           "Wiederholbarkeit", rep.drop(columns=["icc_linear"]).to_string(index=False), "Zusammenfassung", summ.to_string(index=False),
           "Katalog-ICC", sens.to_string(index=False), "Matrixklassen", cls.to_string(index=False)]
    (OUT / "A_R5K_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), "fehlend", missing, flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
