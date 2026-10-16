"""AW-240 / A-R3: Praezisionsrechnung der Bandgrenzen (AW-223, PTB-XL-Paare bis 24 h) mit 10 x 2.000 Ziehungen.

Verwendet das gebundene Skript aw223_auswertung.py (v1.2) unveraendert (Funktionen schaetze, lade_rv, lade_zielwerte,
ci_perzentil, klassifiziere, rolle_effektiv, zeilen_fuer_population); nur Seeds der Laeufe 1 bis 9 sind neu.
Vorabfestlegung: nachrechnung/AW240/A_R3_VORABFESTLEGUNG.md.

  python a_r3_praezision.py --test
  python a_r3_praezision.py --run <ausgabeordner>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, math, hashlib, importlib.util
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd

BOUND = Path(str(data_path("a.werkzeuge_benchmark_aw223_auswertung_py")))
BOUND_SHA = "e62aaa01855922522b9c50daec47e62963dd0c0d97bc9ea41b89642234654b18"
INTERN = Path(str(data_path("a.aw_223_intern_einzelpaare_bis168h_intern_csv_gz")))
FROZEN = Path(str(data_path("a.aw_223_intern_aw223_kennzahlen_unmaskiert_csv")))
N_RUNS = 10
NBOOT = 2000
W_SEED0 = 223010
W_SEED_BASE = 240300
Z_BASE_NEW = 240910


def lade_gebunden():
    sha = hashlib.sha256(BOUND.read_bytes()).hexdigest()
    if sha != BOUND_SHA:
        raise RuntimeError("gebundenes Skript weicht ab: " + sha)
    spec = importlib.util.spec_from_file_location("aw223_gebunden", str(BOUND))
    m = importlib.util.module_from_spec(spec)
    sys.modules["aw223_gebunden"] = m
    spec.loader.exec_module(m)
    return m


def baue_V_und_df(tab, keys):
    """Wertematrix V (Index ecg_id) und Paar-Tabelle df aus der Einzelpaartabelle."""
    a = tab[["ecg_id_1"] + [k + "_1" for k in keys]].copy()
    a.columns = ["ecg_id"] + keys
    b = tab[["ecg_id_2"] + [k + "_2" for k in keys]].copy()
    b.columns = ["ecg_id"] + keys
    allv = pd.concat([a, b], ignore_index=True)
    g = allv.groupby("ecg_id")
    # Konsistenz: derselbe ecg_id muss in beiden Rollen denselben Wert tragen
    for k in keys:
        nun = g[k].nunique(dropna=False)
        if (nun > 1).any():
            raise RuntimeError("inkonsistente Werte je ecg_id: " + k)
    V = g.first()
    df = tab[["patient_id", "ecg_id_1", "ecg_id_2", "abstand_h"]].copy()
    return V, df


def z_lauf(j, a, k, n):
    if j == 0:
        return np.random.default_rng(np.random.SeedSequence([223910, a, k])).standard_normal(n)
    return np.random.default_rng(np.random.SeedSequence([Z_BASE_NEW, j, a, k])).standard_normal(n)


def lauf_j(M, j, df, V, keys, ziel, n_boot, moments_fn):
    seed = W_SEED0 if j == 0 else W_SEED_BASE + j
    S = M.schaetze(df, V, keys, seed, moments_fn, nboot=n_boot)
    out = {}
    for jj, key in enumerate(keys):
        k = [m for m in M.MARKER if m[1] == key][0][0]
        t = ziel[key]
        bt = S["boot"][:, jj, 3]
        rc = float(S["point"][jj][3])
        sigma = (math.log(t["rc_hi"]) - math.log(t["rc_lo"])) / (2 * 1.96)
        z = z_lauf(j, 0, k, len(bt))
        rb = bt / (t["rc"] * np.exp(sigma * z))
        out[key] = {"rc_boot": bt, "R_boot": rb, "rc": rc, "est": bool(S["est"][jj]), "n_personen": int(S["nper"][jj]),
                    "n_paare": int(S["npair"][jj]), "sigma": sigma}
    return out, S


def auswerten(M, laeufe, keys, ziel, step0, n_boot):
    """laeufe: Liste von dict key -> {...}. Rueckgabe Zeilen je Marker."""
    zeilen = []
    n_runs = len(laeufe)
    for key in keys:
        cfg = [m for m in M.MARKER if m[1] == key][0]
        rolle = cfg[4]
        rolle_eff = M.rolle_effektiv(rolle, key, step0)
        rc_pt = laeufe[0][key]["rc"]
        t = ziel[key]
        npers = laeufe[0][key]["n_personen"]
        if not laeufe[0][key]["est"]:
            continue
        mind_run = int(0.95 * n_boot)
        # je Lauf
        lo_r, hi_r, plo_r, phi_r, rclo_r, rchi_r, klassen = [], [], [], [], [], [], []
        for L in laeufe:
            rb = L[key]["R_boot"]
            lo, hi, ng = M.ci_perzentil(rb, mind=mind_run)
            lo_p, hi_p, _ = M.ci_perzentil(L[key]["rc_boot"] / t["rc"], mind=mind_run)
            lo_c, hi_c, _ = M.ci_perzentil(L[key]["rc_boot"], mind=mind_run)
            lo_r.append(lo); hi_r.append(hi); plo_r.append(lo_p); phi_r.append(hi_p); rclo_r.append(lo_c); rchi_r.append(hi_c)
            klassen.append(M.klassifiziere(lo, hi, npers, ng, rolle_eff, True))
        rb_all = np.concatenate([L[key]["R_boot"] for L in laeufe])
        rc_all = np.concatenate([L[key]["rc_boot"] for L in laeufe])
        mind_all = int(0.95 * len(rb_all))
        lo, hi, ng = M.ci_perzentil(rb_all, mind=mind_all)
        lo_p, hi_p, _ = M.ci_perzentil(rc_all / t["rc"], mind=mind_all)
        rc_lo, rc_hi, _ = M.ci_perzentil(rc_all, mind=mind_all)
        klasse_pool = M.klassifiziere(lo, hi, npers, ng, rolle_eff, True)
        lo_r, hi_r = np.array(lo_r), np.array(hi_r)
        se_lo = float(lo_r.std(ddof=1) / math.sqrt(n_runs))
        se_hi = float(hi_r.std(ddof=1) / math.sqrt(n_runs))
        from collections import Counter
        zeilen.append({
            "marker": key, "marker_name": M.MARKER_NAMEN[key], "rolle_eff": rolle_eff, "n_personen": npers,
            "n_paare": laeufe[0][key]["n_paare"], "R_punkt": rc_pt / t["rc"], "rc": rc_pt, "rc_a": t["rc"],
            "n_ziehungen": len(rb_all), "n_gueltig": ng,
            "R_lo_gepoolt": lo, "R_hi_gepoolt": hi,
            "R_lo_lauf0_eingefroren": float(lo_r[0]), "R_hi_lauf0_eingefroren": float(hi_r[0]),
            "R_lo_mittel_laeufe": float(lo_r.mean()), "R_hi_mittel_laeufe": float(hi_r.mean()),
            "R_lo_min_laeufe": float(lo_r.min()), "R_lo_max_laeufe": float(lo_r.max()),
            "R_hi_min_laeufe": float(hi_r.min()), "R_hi_max_laeufe": float(hi_r.max()),
            "MC_SD_lauf_R_lo": float(lo_r.std(ddof=1)), "MC_SD_lauf_R_hi": float(hi_r.std(ddof=1)),
            "MC_SE_gepoolt_R_lo": se_lo, "MC_SE_gepoolt_R_hi": se_hi,
            "abstand_lo_zu_0.80_in_SE": (lo - M.BAND_LO) / se_lo if se_lo > 0 else float("nan"),
            "abstand_hi_zu_1.25_in_SE": (hi - M.BAND_HI) / se_hi if se_hi > 0 else float("nan"),
            "R_lo_nur_ptb_gepoolt": lo_p, "R_hi_nur_ptb_gepoolt": hi_p,
            "rc_lo_gepoolt": rc_lo, "rc_hi_gepoolt": rc_hi,
            "klasse_gepoolt": klasse_pool, "klasse_lauf0": klassen[0],
            "klassen_verteilung_laeufe": "; ".join(f"{k}: {v}" for k, v in sorted(Counter(klassen).items())),
            "sigma_a": laeufe[0][key]["sigma"],
        })
    return zeilen


def parametrisierung(M, ziel):
    rows = []
    for k, key, aid, div, rolle, unit in M.MARKER:
        t = ziel[key]
        sigma = (math.log(t["rc_hi"]) - math.log(t["rc_lo"])) / (2 * 1.96)
        mu = math.log(t["rc"])
        mean = math.exp(mu + sigma ** 2 / 2)
        sd = mean * math.sqrt(math.exp(sigma ** 2) - 1)
        asym_lo = math.log(t["rc"] / t["rc_lo"])
        asym_hi = math.log(t["rc_hi"] / t["rc"])
        # Kontrolle: Perzentile der Lognormalverteilung durch Ziehung
        z = np.random.default_rng(np.random.SeedSequence([240999, k])).standard_normal(400000)
        x = np.exp(mu + sigma * z)
        q = np.percentile(x, [2.5, 50, 97.5])
        rows.append({"marker": key, "marker_name": M.MARKER_NAMEN[key], "RC_A_Median": t["rc"], "L_A": t["rc_lo"], "U_A": t["rc_hi"],
                     "mu_ln": mu, "sigma_A": sigma, "Lognormal_Mittel": mean, "Lognormal_SD": sd,
                     "ln_RC_minus_ln_L": asym_lo, "ln_U_minus_ln_RC": asym_hi,
                     "Asymmetrie_Verhaeltnis": asym_hi / asym_lo if asym_lo > 0 else float("nan"),
                     "lognormal_p2.5_theoretisch": math.exp(mu - 1.96 * sigma), "lognormal_p97.5_theoretisch": math.exp(mu + 1.96 * sigma),
                     "lognormal_p2.5_durch_L_A": math.exp(mu - 1.96 * sigma) / t["rc_lo"],
                     "lognormal_p97.5_durch_U_A": math.exp(mu + 1.96 * sigma) / t["rc_hi"],
                     "kontrolle_p2.5": float(q[0]), "kontrolle_p50": float(q[1]), "kontrolle_p97.5": float(q[2])})
    return rows


# --------------------------------------------------------------------------- Selbsttest
def selftest():
    M = lade_gebunden()
    fehler = []

    def chk(name, ok):
        print(("ok    " if ok else "FEHLER"), name)
        if not ok:
            fehler.append(name)

    _, moments_fn = None, None
    est_fn, moments_fn = M.lade_rv(nboot=NBOOT)
    keys = M.LL_KEYS
    rng = np.random.default_rng(99)
    n_pers = 260
    rows = []
    eid = 1
    pid = 5000
    for p in range(n_pers):
        tr = {k: rng.normal(60 + 3 * i, 10 + i) for i, k in enumerate(keys)}
        for _ in range(1 + (rng.random() < 0.15)):
            a, b = eid, eid + 1
            eid += 2
            row = {"patient_id": pid + p, "ecg_id_1": a, "ecg_id_2": b, "abstand_h": 5.0}
            for i, k in enumerate(keys):
                s = 3 + 0.5 * i
                row[k + "_1"] = tr[k] + rng.normal(0, s)
                row[k + "_2"] = tr[k] + rng.normal(0, s)
            rows.append(row)
    tab = pd.DataFrame(rows)
    V, df = baue_V_und_df(tab, keys)
    ziel = M.lade_zielwerte()
    step0 = M.lade_schritt0()
    # (a) Lauf 0 gegen gebundenes zeilen_fuer_population
    pop = {"analyse": "PRIM", "df": df, "name": "PRIM", "tool": "looplens"}
    zl, S = M.zeilen_fuer_population(pop, V, keys, ziel, step0, moments_fn)
    L0, _ = lauf_j(M, 0, df, V, keys, ziel, NBOOT, moments_fn)
    ok = True
    for z in zl:
        key = z["marker"]
        rb = L0[key]["R_boot"]
        lo, hi, _ = M.ci_perzentil(rb)
        ok &= abs(lo - z["R_lo"]) < 1e-9 and abs(hi - z["R_hi"]) < 1e-9
    chk("Lauf 0 gleich gebundenes zeilen_fuer_population (13 Marker, R-Grenzen 1e-9)", ok)
    # (b) Pool-Quantile = Quantile der Aneinanderhaengung (mehrere Laeufe)
    laeufe = [L0] + [lauf_j(M, j, df, V, keys, ziel, NBOOT, moments_fn)[0] for j in range(1, N_RUNS)]
    zeil = auswerten(M, laeufe, keys, ziel, step0, NBOOT)
    z0 = [z for z in zeil if z["marker"] == "qtcf"][0]
    rb_all = np.concatenate([L["qtcf"]["R_boot"] for L in laeufe])
    lo, hi = np.quantile(rb_all[np.isfinite(rb_all)], [0.025, 0.975])
    chk("gepoolte Grenzen gleich Quantile der Aneinanderhaengung", abs(lo - z0["R_lo_gepoolt"]) < 1e-12 and abs(hi - z0["R_hi_gepoolt"]) < 1e-12)
    chk("20.000 Ziehungen gepoolt", z0["n_ziehungen"] == 20000)
    # (c) Lauf-SD der Grenze gegen theoretischen Standardfehler eines Quantils (aus der gepoolten Dichte)
    ratios = []
    for z in zeil:
        rb = np.concatenate([L[z["marker"]]["R_boot"] for L in laeufe])
        rb = rb[np.isfinite(rb)]
        q = z["R_lo_gepoolt"]
        h = 0.05 * (np.quantile(rb, 0.975) - np.quantile(rb, 0.025))
        dens = np.mean(np.abs(rb - q) < h) / (2 * h)
        se_theo = math.sqrt(0.025 * 0.975 / NBOOT) / dens
        ratios.append(z["MC_SD_lauf_R_lo"] / se_theo)
    chk(f"Lauf-SD der unteren Grenze gleich theor. SE (Median-Verhaeltnis {np.median(ratios):.2f} in 0,7 bis 1,4)", 0.7 <= np.median(ratios) <= 1.4)
    # (d) Parametrisierung
    par = parametrisierung(M, ziel)
    ok = all(abs(r["kontrolle_p2.5"] / r["lognormal_p2.5_theoretisch"] - 1) < 0.005 and abs(r["kontrolle_p97.5"] / r["lognormal_p97.5_theoretisch"] - 1) < 0.005 for r in par)
    chk("gezogene Lognormal-Perzentile gleich exp(mu -/+ 1,96 sigma) (0,5 %)", ok)
    chk("Verhaeltnis p97.5/p2.5 gleich U_A/L_A (1e-9)", all(abs(r["lognormal_p97.5_theoretisch"] / r["lognormal_p2.5_theoretisch"] / (r["U_A"] / r["L_A"]) - 1) < 1e-9 for r in par))
    chk("ln(p2.5/L_A) = ln(p97.5/U_A) = (Asym_lo - Asym_hi)/2 (1e-9)", all(
        abs(math.log(r["lognormal_p2.5_durch_L_A"]) - (r["ln_RC_minus_ln_L"] - r["ln_U_minus_ln_RC"]) / 2) < 1e-9 and
        abs(math.log(r["lognormal_p97.5_durch_U_A"]) - (r["ln_RC_minus_ln_L"] - r["ln_U_minus_ln_RC"]) / 2) < 1e-9 for r in par))
    chk("Lognormal-Median gleich RC_A (0,5 %)", all(abs(r["kontrolle_p50"] / r["RC_A_Median"] - 1) < 0.005 for r in par))
    print("Selbsttest:", "BESTANDEN" if not fehler else f"FEHLER {fehler}")
    return not fehler


# --------------------------------------------------------------------------- Lauf
def lauf(ordner):
    M = lade_gebunden()
    est_fn, moments_fn = M.lade_rv(nboot=NBOOT)
    keys = M.LL_KEYS
    prot = {"bound_script_sha": BOUND_SHA}
    for name, (pfad, soll) in (("rechenvertrag", M.RV), ("zielwerte", M.EINGABEN["zielwerte_a"]), ("schritt0_json", M.STEP0_JSON)):
        ist = hashlib.sha256(Path(pfad).read_bytes()).hexdigest()
        if ist != soll:
            raise RuntimeError("Hash weicht ab: " + name)
        prot[name] = ist
    prot["intern_tabelle_sha"] = hashlib.sha256(INTERN.read_bytes()).hexdigest()
    prot["eingefrorene_kennzahlen_sha"] = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    tab = pd.read_csv(INTERN)
    tab = tab[tab["abstand_h"] <= 24.0].reset_index(drop=True)
    assert len(tab) == 312 and tab["patient_id"].nunique() == 259, (len(tab), tab["patient_id"].nunique())
    V, df = baue_V_und_df(tab, keys)
    ziel = M.lade_zielwerte()
    step0 = M.lade_schritt0()
    # Gate: Lauf 0 reproduziert die eingefrorenen Zahlen
    fr = pd.read_csv(FROZEN)
    sub = fr[(fr["analyse"] == "PRIM") & (fr["werkzeug"] == "looplens")].set_index("marker")
    laeufe = []
    gate = []
    for j in range(N_RUNS):
        L, S = lauf_j(M, j, df, V, keys, ziel, NBOOT, moments_fn)
        laeufe.append(L)
        if j == 0:
            for key in keys:
                lo, hi, _ = M.ci_perzentil(L[key]["R_boot"])
                if L[key]["est"] and key in sub.index and pd.notna(sub.loc[key, "R_lo"]):
                    # eingefrorene Datei ist auf 10 signifikante Stellen gerundet -> relative Abweichung
                    d = max(abs(lo / sub.loc[key, "R_lo"] - 1), abs(hi / sub.loc[key, "R_hi"] - 1), abs(L[key]["rc"] / sub.loc[key, "rc"] - 1))
                    gate.append((key, d))
    gate_max = max(d for _, d in gate)
    print("Gate Lauf 0 gegen eingefrorene Kennzahlen: maximale relative Abweichung", gate_max, "ueber", len(gate), "Marker")
    if gate_max > 1e-9:
        raise RuntimeError("Lauf 0 reproduziert die eingefrorenen Zahlen nicht")
    zeil = auswerten(M, laeufe, keys, ziel, step0, NBOOT)
    par = parametrisierung(M, ziel)
    os.makedirs(ordner, exist_ok=True)
    pd.DataFrame(zeil).to_csv(os.path.join(ordner, "a_r3_gepoolt.csv"), index=False, encoding="utf-8")
    pd.DataFrame(par).to_csv(os.path.join(ordner, "a_r3_lognormal_parametrisierung.csv"), index=False, encoding="utf-8")
    # Einzellaeufe (Grenzen je Lauf fuer alle Marker)
    rows = []
    for j, L in enumerate(laeufe):
        for key in keys:
            if not L[key]["est"]:
                continue
            lo, hi, ng = M.ci_perzentil(L[key]["R_boot"])
            rows.append({"lauf": j, "seed_gewichte": W_SEED0 if j == 0 else W_SEED_BASE + j, "marker": key, "R_lo": lo, "R_hi": hi, "n_gueltig": ng})
    pd.DataFrame(rows).to_csv(os.path.join(ordner, "a_r3_einzellaeufe.csv"), index=False, encoding="utf-8")
    json.dump({"protokoll": prot, "gate_max_abweichung": gate_max, "n_gate_marker": len(gate)},
              open(os.path.join(ordner, "a_r3_protokoll.json"), "w"), indent=1)
    return zeil, par, prot, gate_max


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--run" in sys.argv:
        ordner = sys.argv[sys.argv.index("--run") + 1]
        zeil, par, prot, gm = lauf(ordner)
        print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
        print(json.dumps(prot, indent=1))
        pd.set_option("display.width", 250)
        d = pd.DataFrame(zeil)
        cols = ["marker_name", "rolle_eff", "n_personen", "R_punkt", "R_lo_gepoolt", "R_hi_gepoolt", "R_lo_lauf0_eingefroren", "R_hi_lauf0_eingefroren",
                "MC_SD_lauf_R_lo", "MC_SD_lauf_R_hi", "MC_SE_gepoolt_R_lo", "MC_SE_gepoolt_R_hi", "abstand_lo_zu_0.80_in_SE", "abstand_hi_zu_1.25_in_SE",
                "klasse_gepoolt", "klasse_lauf0", "klassen_verteilung_laeufe"]
        print(d[cols].to_string())
