"""AW-244 A-R7 (nachtraeglich, post hoc): PTB-XL, LoopLens und BRAVEHEART auf identischer markerweiser Paarmenge.

Ausgang: AW-223 (Bericht A_EXTERN_PTBXL_ERGEBNIS_AW223.md, Sensitivitaet S9). Dort rechnete BRAVEHEART auf eigenem
Primaerfenster (qrs_ok_bh) oder auf den LoopLens-Primaerpaaren; die Spalte LoopLens stand auf einer anderen
Paarmenge. Hier: je Marker die Schnittmenge der Paare, in denen BEIDE Werkzeuge an BEIDEN Aufnahmen einen endlichen Wert
haben; RC, ICC, Bias und R (RC / RC_A aus A) fuer beide Werkzeuge auf genau dieser Menge. Kein Werkzeugvergleich der
Richtigkeit (kein Referenzstandard); das gepaarte RC-Verhaeltnis BRAVEHEART / LoopLens (gemeinsame Bootstrap-Gewichte) misst nur
Unterschiede der Wiederholbarkeit auf gleichen Paaren.
Varianten der Basis: V1 = Paare bis 24 h, ohne PT-Paare, Signale verschieden, markerweise beide Werkzeuge gueltig;
V2 = zusaetzlich qrs_ok_ll und qrs_ok_bh (Primaerpopulation von AW-223 und Statusmerker beider Werkzeuge).
Rechenvertrag unveraendert aus AW-223 (aw223_auswertung.py v1.2, Hash geprueft; estimate/moments aus rv_a1, Hash geprueft),
2.000 Personenziehungen, Seed 223019 (Analyse S9) fuer die Gewichte, gemeinsam fuer beide Werkzeuge; R-Unsicherheit
lognormal wie in AW-223 (A-Bootstrap-Intervall). Nur Aggregate. PTB-XL (offen). Keine MIMIC-Daten.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AW223 = h244.AUSW / "werkzeuge" / "benchmark" / "aw223_auswertung.py"
AW223_SHA = "e62aaa01855922522b9c50daec47e62963dd0c0d97bc9ea41b89642234654b18"
SEED = 223019


def load_aw223():
    if h244.sha(AW223) != AW223_SHA:
        raise SystemExit("aw223_auswertung.py weicht vom gebundenen Hash ab")
    spec = importlib.util.spec_from_file_location("aw223_auswertung", str(AW223))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aw223_auswertung"] = mod
    spec.loader.exec_module(mod)
    return mod


def intersect_sets(P, V_ll, V_bh, ll_key, variant):
    base = (P.abstand_h <= 24) & ~P.pt_paar & ~P.identisch
    if variant == "V2":
        base = base & P.qrs_ok_ll & P.qrs_ok_bh
    df = P[base]
    a1 = V_ll.reindex(df.ecg_id_1.to_numpy())[ll_key].to_numpy(float)
    a2 = V_ll.reindex(df.ecg_id_2.to_numpy())[ll_key].to_numpy(float)
    b1 = V_bh.reindex(df.ecg_id_1.to_numpy())[ll_key].to_numpy(float)
    b2 = V_bh.reindex(df.ecg_id_2.to_numpy())[ll_key].to_numpy(float)
    ok = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(b1) & np.isfinite(b2)
    return df[ok]


def analyse(mod, moments_fn, P, V_ll, V_bh, ziel=None, nboot=None, variants=("V1", "V2"), keys=None, log=print):
    nboot = nboot or mod.NBOOT
    rows = []
    keys = keys or mod.BH_MARKER
    for variant in variants:
        for key in keys:
            df = intersect_sets(P, V_ll, V_bh, key, variant)
            nper = int(df.patient_id.nunique())
            base = {"variante": variant, "marker": key, "marker_name": mod.MARKER_NAMEN[key], "n_paare": int(len(df)), "n_personen": nper}
            if nper < mod.MIN_PERSONEN_SCHAETZBAR:
                rows.append({**base, "werkzeug": "beide", "status": "nicht schaetzbar (weniger als 30 Personen)"})
                continue
            S = {}
            for tool, V in (("looplens", V_ll), ("braveheart", V_bh)):
                S[tool] = mod.schaetze(df, V, [key], SEED, moments_fn, nboot)
            cfg = [m for m in mod.MARKER if m[1] == key][0]
            k, _, aid, div, rolle, _u = cfg
            for tool in ("looplens", "braveheart"):
                s = S[tool]
                pt, bt = s["point"][0], s["boot"][:, 0, :]
                icc_lo, icc_hi, _ = mod.ci_perzentil(bt[:, 0])
                b_lo, b_hi, _ = mod.ci_perzentil(bt[:, 1])
                rc_lo, rc_hi, rc_n = mod.ci_perzentil(bt[:, 3])
                row = {**base, "werkzeug": tool, "status": "ok" if s["est"][0] else "nicht schaetzbar",
                       "icc": float(pt[0]), "icc_lo": icc_lo, "icc_hi": icc_hi, "bias": float(pt[1]), "bias_lo": b_lo, "bias_hi": b_hi,
                       "rc": float(pt[3]), "rc_lo": rc_lo, "rc_hi": rc_hi, "rc_n_gueltig": rc_n,
                       "zwischen_sd": float(s["bsd"][0]), "n_paare_schaetz": int(s["npair"][0]), "n_personen_schaetz": int(s["nper"][0])}
                if ziel is not None and s["est"][0]:
                    t = ziel[key]
                    r = mod.kombiniertes_R(bt[:, 3], float(pt[3]), t["rc"], t["rc_lo"], t["rc_hi"], mod.ANALYSEN["S9"][0], k)
                    row.update({"R": r["R"], "R_lo": r["R_lo"], "R_hi": r["R_hi"], "R_n_gueltig": r["R_n_gueltig"], "rc_a": t["rc"]})
                rows.append(row)
            # gepaartes Verhaeltnis der RC (gemeinsame Gewichte)
            rb = S["braveheart"]["boot"][:, 0, 3] / S["looplens"]["boot"][:, 0, 3]
            lo, hi, n = mod.ci_perzentil(rb)
            ib = S["braveheart"]["boot"][:, 0, 0] - S["looplens"]["boot"][:, 0, 0]
            ilo, ihi, _ = mod.ci_perzentil(ib)
            rows.append({**base, "werkzeug": "Verhaeltnis_BH_zu_LL", "status": "ok",
                         "rc_verhaeltnis": float(S["braveheart"]["point"][0][3] / S["looplens"]["point"][0][3]),
                         "rc_verhaeltnis_lo": lo, "rc_verhaeltnis_hi": hi, "rc_verhaeltnis_n_gueltig": n,
                         "icc_differenz_bh_minus_ll": float(S["braveheart"]["point"][0][0] - S["looplens"]["point"][0][0]),
                         "icc_diff_lo": ilo, "icc_diff_hi": ihi})
        log(f"{variant} fertig")
    return rows


def synthetic(mod, n_pat=400, seed=3):
    rng = np.random.default_rng(seed)
    pairs = []
    eid = 1
    for p in range(n_pat):
        e1, e2 = eid, eid + 1
        eid += 2
        pairs.append({"patient_id": p, "ecg_id_1": e1, "ecg_id_2": e2, "abstand_h": rng.uniform(0, 30)})
    P = pd.DataFrame(pairs)
    P["pt_paar"] = rng.random(len(P)) < 0.05
    P["identisch"] = False
    P["qrs_ok_ll"] = True
    P["qrs_ok_bh"] = rng.random(len(P)) < 0.97
    ecg_ids = np.r_[P.ecg_id_1.to_numpy(), P.ecg_id_2.to_numpy()]
    truth = {k: rng.normal(0, 1, n_pat) for k in mod.BH_MARKER}
    V_ll = pd.DataFrame(index=np.sort(ecg_ids), columns=mod.LL_ALLE, dtype=float)
    V_bh = pd.DataFrame(index=np.sort(ecg_ids), columns=mod.LL_ALLE, dtype=float)
    for k in mod.BH_MARKER:
        t = np.repeat(truth[k], 2)
        order = np.r_[P.ecg_id_1.to_numpy(), P.ecg_id_2.to_numpy()]
        # beide ECG einer Person tragen dieselbe Wahrheit
        tt = pd.Series(np.r_[truth[k], truth[k]], index=order)
        V_ll[k] = tt.reindex(V_ll.index) + rng.normal(0, 0.5, len(V_ll))     # LoopLens: Fehler-SD 0.5
        V_bh[k] = tt.reindex(V_bh.index) + rng.normal(0, 1.0, len(V_bh))     # BRAVEHEART: Fehler-SD 1.0
    # fehlende Werte
    V_ll.iloc[rng.choice(len(V_ll), 60, replace=False), 0] = np.nan
    V_bh.iloc[rng.choice(len(V_bh), 90, replace=False), 1] = np.nan
    return P, V_ll, V_bh


def test():
    h244.start_watch()
    mod = load_aw223()
    _e, moments_fn = mod.lade_rv(nboot=200)
    P, V_ll, V_bh = synthetic(mod)
    keys = mod.BH_MARKER[:3]
    rows = analyse(mod, moments_fn, P, V_ll, V_bh, ziel=None, nboot=200, variants=("V1", "V2"), keys=keys, log=lambda s: None)
    df = pd.DataFrame(rows)
    # Schnittmenge: beide Werkzeuge gueltig
    for key in keys:
        s = intersect_sets(P, V_ll, V_bh, key, "V1")
        for col, V in (("e1", V_ll), ("e2", V_bh)):
            assert np.isfinite(V.reindex(s.ecg_id_1.to_numpy())[key].to_numpy(float)).all()
            assert np.isfinite(V.reindex(s.ecg_id_2.to_numpy())[key].to_numpy(float)).all()
        assert (s.abstand_h <= 24).all() and (~s.pt_paar).all()
        s2 = intersect_sets(P, V_ll, V_bh, key, "V2")
        assert len(s2) <= len(s) and s2.qrs_ok_bh.all()
    r = df[(df.variante == "V1") & (df.marker == keys[2]) & (df.werkzeug == "looplens")].iloc[0]
    rb = df[(df.variante == "V1") & (df.marker == keys[2]) & (df.werkzeug == "braveheart")].iloc[0]
    # wahres RC: 1.96*sqrt(2)*Fehler-SD  -> LL 1.386, BH 2.772; wahres ICC 1/(1+0.25)=0.8 bzw. 0.5
    assert abs(r.rc - 1.96 * np.sqrt(2) * 0.5) < 0.25, r.rc
    assert abs(rb.rc - 1.96 * np.sqrt(2) * 1.0) < 0.5, rb.rc
    assert abs(r.icc - 0.8) < 0.07 and abs(rb.icc - 0.5) < 0.1, (r.icc, rb.icc)
    rv = df[(df.variante == "V1") & (df.marker == keys[2]) & (df.werkzeug == "Verhaeltnis_BH_zu_LL")].iloc[0]
    assert 1.5 < rv.rc_verhaeltnis < 2.6
    print("TEST_OK", round(r.rc, 3), round(rb.rc, 3), round(r.icc, 3), round(rb.icc, 3), round(rv.rc_verhaeltnis, 3))


def run():
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("a_r7")
    h244.start_watch()
    mod = load_aw223()
    prot = mod.pruefe_bindung()
    step0 = mod.lade_schritt0()
    _e, moments_fn = mod.lade_rv()
    mod.selbsttest_rv(moments_fn, print)
    ziel = mod.lade_zielwerte()
    meta = mod.lade_metadaten()
    P = mod.lade_paarliste()
    P = mod.paar_merkmale(P, meta)
    cache = mod.OUT_INTERN / "signalhashes_ptbxl_aw223.csv"
    d = pd.read_csv(cache, dtype={"sha256": str})
    have = dict(zip(d.ecg_id.astype(int), d.sha256))
    need = pd.unique(np.r_[P[P.abstand_h <= 168].ecg_id_1.to_numpy(), P[P.abstand_h <= 168].ecg_id_2.to_numpy()])
    assert all(int(i) in have for i in need), "Signalhash-Cache unvollstaendig"
    P["identisch"] = [bool(a in have and b in have and have[a] == have[b]) for a, b in zip(P.ecg_id_1, P.ecg_id_2)]
    V_ll, _s1, einh_ll, _d1 = mod.lade_langformat(mod.EINGABEN["langformat_looplens"][0], "looplens", True, None, "aufnahme")
    V_ll = V_ll[mod.LL_ALLE]
    V_bh, _s2, einh_bh, _d2 = mod.lade_langformat(mod.EINGABEN["langformat_braveheart"][0], "braveheart", True, mod.BH_ZU_LL, "medianschlag")
    V_bh = V_bh.reindex(columns=mod.LL_ALLE)
    # Gate: Primaerpopulation von AW-223 (312 Paare / 259 Personen) und S9b (296 / 246) reproduzieren
    prim = (P.abstand_h <= 24) & ~P.pt_paar & P.qrs_ok_ll & ~P.identisch
    assert int(prim.sum()) == 312 and int(P[prim].patient_id.nunique()) == 259, "Gate PRIM"
    s9b = prim & P.qrs_ok_bh
    assert int(s9b.sum()) == 296 and int(P[s9b].patient_id.nunique()) == 246, "Gate S9b"
    rows = analyse(mod, moments_fn, P, V_ll, V_bh, ziel=ziel)
    out = pd.DataFrame(rows)
    out.to_csv(h244.PUB / "a_r7_ergebnisse.csv", index=False)
    txt = ["A-R7 PTB-XL: LoopLens und BRAVEHEART auf identischer markerweiser Paarmenge (post hoc)", f"Skript-SHA-256 {shash}",
           f"aw223_auswertung.py {AW223_SHA}", "Gates: PRIM 312/259, S9b 296/246 bestanden", out.to_string(index=False)]
    (h244.PUB / "A_R7_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
