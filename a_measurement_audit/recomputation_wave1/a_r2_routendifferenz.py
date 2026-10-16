"""AW-240 / A-R2: gepaarte Differenz der Arzneimittelkontraste zwischen Produktions- und Quellmessweg
(ECGRDVQ, oeffentliche Daten; nachtraeglich, explorativ).

Eingabe: <local path> (Spalten mode, drug, marker, RANDID, dd).
  dd = Teilnehmerwert der Doppeldifferenz (Wirkstoff post minus Baseline, minus Placebo post minus Baseline), 2 h.
Je Wirkstoff und Marker: D_i = dd_Produktion,i - dd_Quelle,i je Teilnehmer i (nur Teilnehmer mit beiden Werten).
Intervalle: Teilnehmer-Bootstrap (10.000 Ziehungen, Perzentil) und gepaartes t-Intervall (Sensitivitaet).
Multiplizitaet: BH ueber die 16 Sekundaerkontraste (nicht schaetzbare mit p = 1), Holm ueber die 4 QTcF-Kontraste.

  python a_r2_routendifferenz.py --test
  python a_r2_routendifferenz.py --run <ausgabeordner>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os, json, hashlib
import numpy as np
from math import sqrt

SEED = 240201
N_BOOT = 10000
EINGABE = str(data_path("a.aw152_person_dd_2h_csv"))
WIRKSTOFFE = ["Dofetilide", "Quinidine Sulph", "Ranolazine", "Verapamil HCL"]
NAMEN = {"Dofetilide": "Dofetilide", "Quinidine Sulph": "Quinidine", "Ranolazine": "Ranolazine", "Verapamil HCL": "Verapamil"}
PRIMAER = [("qtc_fridericia", "QTcF (ms)")]
SEKUNDAER = [("spatial_qrs_t_angle_mean", "Area angle QRS-T (deg)"), ("tpeak_tend_ms", "Tpeak-Tend (ms)"),
             ("qrs_duration", "QRS duration (ms)"), ("qrs_t_normalized_discrete_frechet", "Normalized Frechet")]


# ------------------------------------------------------------------ Statistik (eigene Implementierung)
def t_quantil(df, p=0.975):
    from scipy import stats
    return float(stats.t.ppf(p, df))


def t_pwert(mean, se, df):
    from scipy import stats
    return float(2 * stats.t.sf(abs(mean / se), df))


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def boot_mittel(d, n_boot, seed_seq):
    rng = np.random.default_rng(seed_seq)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    # zweiseitiger Bootstrap-p (Perzentilmethode, Anteil der Ziehungen jenseits 0 mit Untergrenze 1/(B+1))
    p_lo = (np.sum(m <= 0) + 1) / (n_boot + 1)
    p_hi = (np.sum(m >= 0) + 1) / (n_boot + 1)
    p = min(1.0, 2 * min(p_lo, p_hi))
    return float(lo), float(hi), float(p), m


def kontrast(prod, quel, k, n_boot=N_BOOT, seed=SEED):
    """prod, quel: dict RANDID -> dd. Rueckgabe dict mit allen Kennzahlen (k = laufender Kontrastindex)."""
    ids = sorted(set(prod) & set(quel))
    out = {"n_prod": len(prod), "n_quelle": len(quel), "n_gepaart": len(ids)}
    if len(ids) < 3:
        out.update(schaetzbar=False)
        return out
    a = np.array([prod[i] for i in ids], float)
    b = np.array([quel[i] for i in ids], float)
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / sqrt(n)
    tq = t_quantil(n - 1)
    lo, hi, pb, _ = boot_mittel(d, n_boot, np.random.SeedSequence([seed, k]))
    out.update(schaetzbar=True, mean_prod=float(a.mean()), mean_quelle=float(b.mean()),
               D=float(d.mean()), D_median=float(np.median(d)), SD_D=sd,
               boot_lo=lo, boot_hi=hi, p_boot=pb,
               t_lo=float(d.mean() - tq * se), t_hi=float(d.mean() + tq * se),
               p_t=t_pwert(d.mean(), se, n - 1),
               korr_routen=float(np.corrcoef(a, b)[0, 1]),
               SD_prod=float(a.std(ddof=1)), SD_quelle=float(b.std(ddof=1)))
    return out


def lade(pfad):
    import csv
    daten = {}
    with open(pfad, newline="", encoding="utf-8") as f:
        for z in csv.DictReader(f):
            if z["dd"] in ("", "nan", "NaN"):
                continue
            daten.setdefault((z["mode"], z["drug"], z["marker"]), {})[z["RANDID"]] = float(z["dd"])
    return daten


def auswerten(daten, n_boot=N_BOOT, seed=SEED):
    zeilen = []
    k = 0
    for fam, marker_liste in (("primaer", PRIMAER), ("sekundaer", SEKUNDAER)):
        for marker, label in marker_liste:
            for w in WIRKSTOFFE:
                prod = daten.get(("production", w, marker), {})
                quel = daten.get(("source", w, marker), {})
                r = kontrast(prod, quel, k, n_boot, seed)
                r.update(familie=fam, marker=marker, label=label, wirkstoff=NAMEN[w], k=k)
                zeilen.append(r)
                k += 1
    # Multiplizitaet
    sek = [z for z in zeilen if z["familie"] == "sekundaer"]
    ps = [z["p_t"] if z["schaetzbar"] else 1.0 for z in sek]
    pb = [z["p_boot"] if z["schaetzbar"] else 1.0 for z in sek]
    for z, q1, q2 in zip(sek, bh(ps), bh(pb)):
        z["q_BH16_t"], z["q_BH16_boot"] = float(q1), float(q2)
    pri = [z for z in zeilen if z["familie"] == "primaer"]
    for z, q1, q2 in zip(pri, holm([z["p_t"] for z in pri]), holm([z["p_boot"] for z in pri])):
        z["q_Holm4_t"], z["q_Holm4_boot"] = float(q1), float(q2)
    return zeilen


def zusammenfassung(zeilen):
    sch = [z for z in zeilen if z["schaetzbar"]]
    s = {"kontraste_gesamt": len(zeilen), "schaetzbar": len(sch),
         "nicht_schaetzbar": len(zeilen) - len(sch),
         "boot_ci_ohne_0": sum(1 for z in sch if z["boot_lo"] > 0 or z["boot_hi"] < 0),
         "t_ci_ohne_0": sum(1 for z in sch if z["t_lo"] > 0 or z["t_hi"] < 0)}
    for fam in ("primaer", "sekundaer"):
        sf = [z for z in sch if z["familie"] == fam]
        s[fam + "_schaetzbar"] = len(sf)
        s[fam + "_boot_ci_ohne_0"] = sum(1 for z in sf if z["boot_lo"] > 0 or z["boot_hi"] < 0)
        s[fam + "_t_ci_ohne_0"] = sum(1 for z in sf if z["t_lo"] > 0 or z["t_hi"] < 0)
    sek = [z for z in sch if z["familie"] == "sekundaer"]
    s["sekundaer_BH_q_unter_0.05_t"] = sum(1 for z in sek if z["q_BH16_t"] < 0.05)
    s["sekundaer_BH_q_unter_0.05_boot"] = sum(1 for z in sek if z["q_BH16_boot"] < 0.05)
    pri = [z for z in sch if z["familie"] == "primaer"]
    s["primaer_Holm_unter_0.05_t"] = sum(1 for z in pri if z["q_Holm4_t"] < 0.05)
    s["primaer_Holm_unter_0.05_boot"] = sum(1 for z in pri if z["q_Holm4_boot"] < 0.05)
    return s


# ------------------------------------------------------------------ Selbsttest
def selftest():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(5)
    fehler = []

    def chk(name, ok):
        print(("ok    " if ok else "FEHLER"), name)
        if not ok:
            fehler.append(name)

    # BH / Holm gegen statsmodels
    p = rng.uniform(0, 1, 16) ** 3
    chk("BH gleich statsmodels", np.allclose(bh(p), multipletests(p, method="fdr_bh")[1]))
    chk("Holm gleich statsmodels", np.allclose(holm(p[:4]), multipletests(p[:4], method="holm")[1]))
    # Kunstdaten: bekannte Verschiebung D = 3, SD(D) = 4, n = 22
    n = 22
    ids = [str(1000 + i) for i in range(n)]
    quelle = {i: rng.normal(0, 5) for i in ids}
    prod = {i: quelle[i] + 3.0 + rng.normal(0, 4) for i in ids}
    r = kontrast(prod, quelle, 0, 10000, 11)
    d = np.array([prod[i] - quelle[i] for i in ids])
    chk("D gleich direktem Mittel", abs(r["D"] - d.mean()) < 1e-12)
    se = d.std(ddof=1) / sqrt(n)
    chk("t-Intervall gleich Handformel", abs(r["t_lo"] - (d.mean() - t_quantil(n - 1) * se)) < 1e-9)
    chk("Bootstrap-Intervall nah am t-Intervall (n=22: +-20% der Breite)",
        abs(r["boot_lo"] - r["t_lo"]) < 0.2 * (r["t_hi"] - r["t_lo"]) and abs(r["boot_hi"] - r["t_hi"]) < 0.2 * (r["t_hi"] - r["t_lo"]))
    # Teilnehmermenge verschieden -> nur Schnittmenge
    q2 = dict(quelle)
    q2.pop(ids[0])
    q2["9999"] = 1.0
    r2 = kontrast(prod, q2, 0, 2000, 11)
    chk("nur Schnittmenge gepaart (21)", r2["n_gepaart"] == 21)
    # keine Differenz: Intervall deckt 0, p gross
    prod0 = {i: quelle[i] + rng.normal(0, 4) for i in ids}
    r0 = kontrast(prod0, quelle, 1, 10000, 12)
    # Deckung der Nullhypothese in 300 Wiederholungen (t-Intervall, 95 %)
    dec = 0
    for rep in range(300):
        q = {i: rng.normal(0, 5) for i in ids}
        pr = {i: q[i] + rng.normal(0, 4) for i in ids}
        rr = kontrast(pr, q, 0, 200, rep)
        dec += (rr["t_lo"] <= 0 <= rr["t_hi"])
    chk("t-Intervall-Deckung der Wahrheit 0 in 300 Laeufen 91 bis 98 %", 0.91 <= dec / 300 <= 0.98)
    # nicht schaetzbar bei leerem Quellweg
    r3 = kontrast(prod, {}, 0, 100, 1)
    chk("leerer Quellweg -> nicht schaetzbar", r3["schaetzbar"] is False)
    # Gesamtlauf mit Kunstdaten
    daten = {}
    for w in WIRKSTOFFE:
        for m, _ in PRIMAER + SEKUNDAER:
            daten[("production", w, m)] = {i: rng.normal(1, 3) for i in ids}
            if m != "qrs_t_normalized_discrete_frechet":
                daten[("source", w, m)] = {i: rng.normal(0, 3) for i in ids}
    z = auswerten(daten, 500, 3)
    s = zusammenfassung(z)
    chk("16+4 Kontraste, 4 nicht schaetzbar (Frechet)", s["kontraste_gesamt"] == 20 and s["nicht_schaetzbar"] == 4)
    chk("BH-q der nicht schaetzbaren = 1 eingerechnet (alle q >= p)", all(x["q_BH16_t"] >= x["p_t"] - 1e-12 for x in z if x["schaetzbar"] and x["familie"] == "sekundaer"))
    print("Selbsttest:", "BESTANDEN" if not fehler else f"FEHLER {fehler}")
    return not fehler


def f3(x):
    return f"{x:.3f}"


def schreibe(zeilen, ordner, quelle_sha):
    import csv
    os.makedirs(ordner, exist_ok=True)
    felder = ["familie", "wirkstoff", "marker", "label", "n_prod", "n_quelle", "n_gepaart", "mean_prod", "mean_quelle", "D", "boot_lo", "boot_hi",
              "t_lo", "t_hi", "D_median", "SD_D", "korr_routen", "p_t", "p_boot", "q_BH16_t", "q_BH16_boot", "q_Holm4_t", "q_Holm4_boot"]
    with open(os.path.join(ordner, "a_r2_kontraste.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(felder)
        for z in zeilen:
            w.writerow([("nicht schaetzbar" if (not z.get("schaetzbar", False) and k not in ("familie", "wirkstoff", "marker", "label", "n_prod", "n_quelle", "n_gepaart")) else z.get(k, "")) for k in felder])


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--run" in sys.argv:
        ordner = sys.argv[sys.argv.index("--run") + 1]
        sha = hashlib.sha256(open(EINGABE, "rb").read()).hexdigest()
        daten = lade(EINGABE)
        zeilen = auswerten(daten)
        schreibe(zeilen, ordner, sha)
        s = zusammenfassung(zeilen)
        print("eingabe sha256", sha)
        print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
        print(json.dumps(s, indent=1))
        for z in zeilen:
            if z["schaetzbar"]:
                q = z.get("q_BH16_t", z.get("q_Holm4_t"))
                print(f'{z["familie"][:3]} {z["wirkstoff"]:10s} {z["label"]:26s} n={z["n_gepaart"]} prod {f3(z["mean_prod"]):>8} quelle {f3(z["mean_quelle"]):>8} '
                      f'D {f3(z["D"]):>8} boot[{f3(z["boot_lo"])};{f3(z["boot_hi"])}] t[{f3(z["t_lo"])};{f3(z["t_hi"])}] r={z["korr_routen"]:.2f} p_t={z["p_t"]:.3g} q={q:.3g}')
            else:
                print(f'{z["familie"][:3]} {z["wirkstoff"]:10s} {z["label"]:26s} nicht schaetzbar (n_quelle={z["n_quelle"]})')
