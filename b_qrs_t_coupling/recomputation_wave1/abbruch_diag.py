"""AW-241 Nachtrag Abbruchkriterium: Prueft an den abgelehnten Ziehungen von S3b 045, 047, 049 und S8 (Rang 3, 777 von 1.000 gueltig),
ob die Ablehnung 'Score/N-Konvergenz nicht erfuellt' auf dem Widerspruch zwischen Abbruchregel von logistic_fit (max|Gradient|/sum(w) < 1e-7)
und Annahmeregel von statistik.fit (max|Gradient|/N < 1e-7, N = Zeilen mit Gewicht > 0) beruht. Nur Zaehlung und Ursache, keine neuen Effektgroessen.
Liest Testprojektionen der internen Assessmentpartition (manifest v5) und die Entwicklungsprojektion der Verifikation; locked_test nicht beruehrt.
Aufruf: python abbruch_diag.py s3b | s8
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
import numpy as np  # noqa: E402
import loeser_s3b as L  # noqa: E402
from scipy.special import expit  # noqa: E402
from werkzeuge.b_test import adapter, daten  # noqa: E402
from werkzeuge.b_test import statistik as st  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

AGG = Path(str(data_path("b.nachrechnung_aw241")))
PEN = 1e-6
ROWS = {
    "045": ("6a", "QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster / BEF-0049", 244, 2),
    "047": ("6c", "Alter <65 / BEF-0049", 181, 4),
    "049": ("6d", "Geschlecht F / BEF-0049", 2, 905),
}


def ratios(z, y, w):
    """Score/sum(w) und Score/N fuer den Originalloeser am Stoppunkt; N = Zeilen mit Gewicht > 0."""
    a = w > 0
    zz, yy, ww = z[a], y[a], w[a]
    b = logistic_fit(zz, yy, ww)
    pen = np.r_[0, np.full(zz.shape[1] - 1, PEN)]
    g = np.max(np.abs(zz.T @ (ww * (expit(zz @ b) - yy)) + pen * b))
    return g / ww.sum(), g / len(yy), len(yy) / ww.sum()


def summarize(label, recs):
    if not recs:
        return dict(menge=label, n=0)
    rw = np.array([r[0] for r in recs])
    rn = np.array([r[1] for r in recs])
    fr = np.array([r[2] for r in recs])
    return dict(menge=label, n=len(recs), score_sumw_min=float(rw.min()), score_sumw_max=float(rw.max()), anteil_sumw_kriterium_verletzt=float((rw >= 1e-7).mean()),
                anteil_N_kriterium_verletzt=float((rn >= 1e-7).mean()), N_ueber_sumw_mittel=float(fr.mean()), N_ueber_sumw_min=float(fr.min()), N_ueber_sumw_max=float(fr.max()))


def s3b():
    frames, package = L.load_frames()
    f49 = frames["BEF-0049"]
    e49 = package["hypotheses"]["BEF-0049"]
    union = np.unique(np.concatenate([f.subject_id.to_numpy() for f in frames.values()]))
    if __import__("hashlib").sha256(union.tobytes()).hexdigest() != L.EXPECT["u142"]:
        raise RuntimeError("Union U142")
    old = json.loads(L.AW142.read_text(encoding="utf8"))
    sel = {"045": lambda z: z.qrs_duration_ms.lt(120) & z.classical_lbbb_pattern_positive.eq(False) & z.classical_rbbb_pattern_positive.eq(False),
           "047": lambda z: z.age.lt(65), "049": lambda z: z.sex.eq("F")}
    cap = {}
    orig = adapter.fit

    def cap_fit(z, y, subjects, direction=1, weights=None, wald=True):
        cap.update(z=np.asarray(z, float), y=np.asarray(y, float), ids=np.asarray(subjects))
        return orig(z, y, subjects, direction, weights=weights, wald=wald)

    adapter.fit = cap_fit
    out = {}
    for name, (rank, label, n_conv, n_dist) in ROWS.items():
        res = adapter.estimate(f49, e49, union, sel[name], bootstrap=False, sex_stratum="F" if name == "049" else None)
        ref = [x for x in old["sensitivities"] if x["rank"] == rank and x["analysis"] == label][0]
        for key in ("beta", "or", "delta_auc", "auc_G", "auc_G_marker"):
            if not np.isclose(res[key], ref[key], rtol=1e-9, atol=0):
                raise RuntimeError(f"Punktgate {name} {key}")
        z, y, ids = cap["z"], cap["y"], cap["ids"]
        conv, dist, other, ok = [], [], {}, []
        for r in range(1000):
            w = st.draws(union, ids, r)
            try:
                st.fit(z, y, ids, weights=w, wald=False)
                if len(ok) < 150:
                    ok.append(ratios(z, y, w))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                msg = str(exc)
                if msg.startswith("Score/N"):
                    conv.append(ratios(z, y, w))
                elif msg.startswith("weniger als 100"):
                    dist.append(r)
                else:
                    other[msg] = other.get(msg, 0) + 1
        out[name] = dict(label=label, N=int(res["N"]), alt_erwartet=dict(score_N=n_conv, weniger_als_100=n_dist), alt_gezaehlt=dict(score_N=len(conv), weniger_als_100=len(dist), sonstige=other),
                         abgelehnt_score_N=summarize("abgelehnt (Score/N)", conv), angenommen_stichprobe=summarize("angenommen (erste 150)", ok))
        print(name, json.dumps(out[name]), flush=True)
    adapter.fit = orig
    (AGG / "ABBRUCH_S3B_ausgabe.txt").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


def s8():
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from werkzeuge.b_test.export import projected

    manifest = json.loads(L.MANIFEST.read_text(encoding="utf8"))
    package = json.loads(L.PARAMS.read_text(encoding="utf8"))
    for key, p in (("manifest", L.MANIFEST), ("params", L.PARAMS)):
        if L.sha(p) != L.EXPECT[key]:
            raise RuntimeError("Hashabweichung " + key)

    def load(src):
        p = Path(src["path"])
        if any(s in str(p).lower() for s in ("locked_", "diagnostics", "publication_preparation")):
            raise PermissionError(str(p))
        if L.sha(p) != src["sha256"]:
            raise RuntimeError("Hash " + p.name)
        f = ds.dataset(str(p), format="parquet").to_table(columns=src["columns"]).to_pandas()
        if set(f.split) != {"test"}:
            raise ValueError("Partition")
        return f

    h1 = package["hypotheses"]["BEF-0063"]
    f63 = daten.select_index(load(manifest["hypotheses"]["BEF-0063"]), h1["endpoint"])
    f49ids = daten.select_index(load(manifest["hypotheses"]["BEF-0049"]), "cardiomyopathy_icd").subject_id.to_numpy()
    rvids = daten.select_index(load(manifest["hypotheses"]["BEF-0087"]), "echo_rv_systolic_dysfunction").subject_id.to_numpy()
    tech = daten.select_index(load(manifest["technical"]), "reduced_lvef_strict", technical=True)
    devsource = package["verification_train"]
    p = Path(devsource["path"])
    if L.sha(p) != devsource["sha256"]:
        raise RuntimeError("Hash dev")
    dev = projected(p, ["subject_id", "split", "age", "sex_numeric", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", "spatial_qrs_t_angle_deg", "classical_cornell_product",
                        "classical_sokolow_lyon", "magnitude_area_qrs", "spatial_low_voltage_qrs_peak_magnitude", "classical_right_axis_deviation", "has_tte_30d"])
    if set(dev.split) != {"train"}:
        raise ValueError("dev split")
    union = np.unique(np.concatenate([f63.subject_id.to_numpy(), f49ids, rvids, tech.subject_id.to_numpy()]))
    # Vorbereitung wie adapter.ipw_analysis
    parameters = package["verification"]
    technical = tech.sort_values("subject_id").copy()
    development = dev.sort_values("subject_id").copy()
    x, _ = st.raw_design(technical.assign(__verification=0.), "__verification", strong=True)
    technical = technical.loc[np.isfinite(x[:, :-1]).all(axis=1)].copy()
    vx = adapter.verification_design(technical, parameters)
    dx = adapter.verification_design(development, parameters)
    dy = development.has_tte_30d.eq(True).to_numpy(float)
    z, good, flow = daten.masks(technical.assign(index_selection_strategy="first_verified_ecg_reference_result_blind"), h1["endpoint"], h1["marker"], h1["comparator"])
    x, names = st.raw_design(z.loc[good], h1["marker"], h1["comparator"])
    design = st.fixed_design(x, h1["scaling_B3"])
    y = daten.classify_lvef(z.loc[good, "preferred_lvef"])
    ids = z.loc[good, "subject_id"].to_numpy()
    prop, w = adapter.ipw_weights(vx, parameters)
    weights = w[good.to_numpy()]
    adapter.ess(weights, y)
    unw = st.fit(design, y, ids)
    wtd = st.fit(design, y, ids, weights=weights)
    if not (abs(unw["or"] - 2.6457223) < 1e-6 and abs(wtd["or"] - 2.9678222) < 1e-6 and unw["N"] == 4844):
        raise RuntimeError(f"Punktgate S8: {unw['or']} {wtd['or']} {unw['N']}")
    fails, conv_a, conv_b, ok_a, ok_b, other = {}, [], [], [], [], {}
    which = {"unweighted_zuerst": 0, "weighted": 0}
    for r in range(1000):
        try:
            mult, rw, _ = adapter.ipw_replicate(dx, dy, development.subject_id.to_numpy(), vx, technical.subject_id.to_numpy(), ids, r, parameters)
            adapter.ess(mult * rw, y)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            other["ipw_replicate/ess: " + str(exc)[:60]] = other.get("ipw_replicate/ess: " + str(exc)[:60], 0) + 1
            continue
        failed = False
        for tag, ww in (("a", mult), ("b", mult * rw)):
            try:
                st.fit(design, y, ids, weights=ww, wald=False)
                if len(ok_a if tag == "a" else ok_b) < 150:
                    (ok_a if tag == "a" else ok_b).append(ratios(design, y, ww))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                msg = str(exc)
                if msg.startswith("Score/N"):
                    (conv_a if tag == "a" else conv_b).append(ratios(design, y, ww))
                else:
                    other[msg[:60]] = other.get(msg[:60], 0) + 1
                failed = True
                break
        fails[r] = failed
    n_fail = sum(fails.values())
    out = dict(ipw_gate=dict(or_unweighted=unw["or"], or_weighted=wtd["or"], N=int(unw["N"])), alt_erwartet=dict(abgelehnt=223, gueltig=777), nachgezaehlt=dict(abgelehnt_gesamt=n_fail, gueltig=1000 - n_fail - sum(other.values())),
               erster_fehlschlag_ungewichtet=len(conv_a), erster_fehlschlag_gewichtet=len(conv_b), sonstige=other,
               abgelehnt_ungewichtet=summarize("abgelehnt (Fit mit Multiplizitaeten)", conv_a), abgelehnt_gewichtet=summarize("abgelehnt (Fit mit Multiplizitaeten und IPW)", conv_b),
               angenommen_ungewichtet=summarize("angenommen (erste 150)", ok_a), angenommen_gewichtet=summarize("angenommen (erste 150)", ok_b))
    print(json.dumps(out), flush=True)
    (AGG / "ABBRUCH_S8_ausgabe.txt").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    {"s3b": s3b, "s8": s8}[sys.argv[1]]()
