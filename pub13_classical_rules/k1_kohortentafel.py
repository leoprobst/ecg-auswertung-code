"""PUB-13 Schritt 3: Merkmalstafel nach QTc-Gruppe, Herzfrequenz-Schichten, Schwellen 480/500/520 ms.

Beschreibend und nachtraeglich (nach Ergebnissicht von K1 festgelegt); nur Entwicklungsdaten (train, validation).
Liest ausschliesslich die vom K1-Lauf erzeugten Arbeitstabellen der Entwicklung (main_cov, main_rules, main_refs unter
<local path>) und rechnet nichts neu aus den Rohdaten. Keine Testpartition, keine Kennungen.
Gebundene Dateien (k1_auswertung.py, regeln.py, k1_geruest.py, adapter_cache.py) werden nicht veraendert; k1_geruest.py wird
nur fuer die Kleinzellenregel importiert (coarsen_counts).

Aufruf: python k1_kohortentafel.py [--out-md PFAD] [--out-json PFAD]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import k1_geruest as G  # noqa: E402

WORK = Path(str(data_path("pub13.aw236_v3_dev_work")))
UNITS = Path(str(data_path("pub13.aw236_v3_dev_units")))
PUB = Path(str(data_path("pub13.publikationen_pub_13_kardiologie")))
OUT_JSON = Path(str(data_path("pub13.aw236_v3_kohortentafel_kohortentafel_exakt_json")))
OUT_MD = PUB / "K1_KOHORTENTAFEL.md"
THRESHOLDS = (480.0, 500.0, 520.0)
HR_BINS = (("HF < 60/min", -np.inf, 60.0, "lt"), ("HF 60 bis 100/min", 60.0, 100.0, "mid"), ("HF > 100/min", 100.0, np.inf, "gt"))


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load():
    cov = pd.read_parquet(WORK / "main_cov.parquet", columns=["age", "female", "adm", "hr", "hr_missing", "hx_hf", "med_assessed", "med_qt_prolonging_drug", "has_prior"])
    rul = pd.read_parquet(WORK / "main_rules.parquet", columns=["R08_baz", "R08_fri", "qtc_baz", "qrs_machine"])
    ref = pd.read_parquet(WORK / "main_refs.parquet", columns=["lab_vtvf", "lab_arrest", "lab_mort30"])
    assert len(cov) == len(rul) == len(ref)
    return cov, rul, ref


# ------------------------------------------------------------------ Kleinzellenregel
def cnt(cells: dict, totals: dict | None = None) -> dict:
    return G.coarsen_counts({k: int(v) for k, v in cells.items()}, {k: int(v) for k, v in (totals or {}).items()} or None)


def pct(c, tot):
    if isinstance(c, str) or isinstance(tot, str) or not tot:
        return "n/a"
    return f"{100.0 * c / tot:.1f}".replace(".", ",")


def cell(c, tot):
    """Anzahl (Prozent) aus bereits vergroeberten Werten."""
    if isinstance(c, str):
        return c
    return f"{c:,}".replace(",", ".") + f" ({pct(c, tot)} %)"


def num_fmt(x):
    if isinstance(x, str):
        return x
    return f"{x:,}".replace(",", ".")


def med_iqr(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    q1, m, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return f"{m:.0f} ({q1:.0f} bis {q3:.0f})"


# ------------------------------------------------------------------ Tafel
def merkmalstafel(cov, rul, ref, pop):
    g1 = pop & (rul["R08_baz"].values == 1)
    g0 = pop & (rul["R08_baz"].values == 0)
    groups = {"QTc > 500 ms": g1, "QTc <= 500 ms": g0, "alle auswertbar": pop}
    ex = {}
    tot_exact = {k: int(m.sum()) for k, m in groups.items()}
    rows = []

    def binrow(name, flag, denom=None):
        """flag: bool-Array; je Gruppe Zellen (ja, nein) gemeinsam vergroebert."""
        r = {}
        for gk, m in groups.items():
            base = m if denom is None else (m & denom)
            yes = int((base & flag).sum())
            no = int((base & ~flag).sum())
            c = cnt({"ja": yes, "nein": no}, {"n": yes + no})
            r[gk] = cell(c["ja"], c["_totals"]["n"])
            ex.setdefault(name, {})[gk] = {"ja": yes, "nein": no}
        rows.append((name, r))

    hr = cov["hr"].values.astype(float)
    hr_ok = (cov["hr_missing"].values == 0)
    rows.append(("Personen (Index-EKG)", {gk: num_fmt(cnt({"n": int(m.sum())})["n"]) for gk, m in groups.items()}))
    rows.append(("Alter, Jahre, Median (Quartile)", {gk: med_iqr(cov["age"].values[m]) for gk, m in groups.items()}))
    binrow("Frauen", cov["female"].values == 1)
    for code, label in (("icu", "Versorgungsebene: Intensivstation (EKG im ICU-Zeitraum)"), ("ed", "Versorgungsebene: Notaufnahme (nicht ICU)"),
                        ("inpatient", "Versorgungsebene: Station (stationär, nicht ICU, nicht Notaufnahme)"), ("other", "Versorgungsebene: übrige (ohne Aufenthalt)")):
        binrow(label, cov["adm"].values == code)
    prior = cov["has_prior"].values == 1
    assessed = cov["med_assessed"].values == 1
    binrow("Frühere Diagnosecodes verfügbar (Vorgeschichte dokumentiert; Anteil aller Personen)", prior)
    binrow("Herzinsuffizienz vor dem Index, nur Personen mit verfügbarer Vorgeschichte (Anteil dieser Personen)", cov["hx_hf"].values == 1, denom=prior)
    binrow("Medikationsdaten verfügbar (Anteil aller Personen)", assessed)
    binrow("QT-verlängernde Medikation, nur Personen mit Medikationsdaten (Anteil dieser Personen)", cov["med_qt_prolonging_drug"].values == 1, denom=assessed)
    rows.append(("Herzfrequenz, /min, Median (Quartile)", {gk: med_iqr(np.where(hr_ok, hr, np.nan)[m]) for gk, m in groups.items()}))
    binrow("Herzfrequenz fehlt (Maschinenwert und RR fehlen)", ~hr_ok)
    q = rul["qrs_machine"].values.astype(float)
    rows.append(("QRS-Dauer (Maschine), ms, Median (Quartile)", {gk: med_iqr(q[m]) for gk, m in groups.items()}))
    binrow("QRS >= 120 ms (Maschine)", q >= 120, denom=np.isfinite(q))
    return rows, ex, tot_exact


# ------------------------------------------------------------------ Risikoverhaeltnisse
def rr_counts(T, D, mask):
    ok = mask & np.isfinite(T) & np.isfinite(D)
    return (int(np.sum(ok & (T == 1) & (D == 1))), int(np.sum(ok & (T == 1))),
            int(np.sum(ok & (T == 0) & (D == 1))), int(np.sum(ok & (T == 0))))


def rr_wald(a, n1, c, n0):
    if min(a, c) == 0 or n1 == 0 or n0 == 0:
        return None
    lr = math.log((a / n1) / (c / n0))
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return math.exp(lr), math.exp(lr - 1.959964 * se), math.exp(lr + 1.959964 * se)


def rr_row(label, T, D, mask):
    a, n1, c, n0 = rr_counts(T, D, mask)
    cc = cnt({"a": a, "c": c}, {"n1": n1, "n0": n0})
    tot = cc["_totals"]
    small = isinstance(cc["a"], str) or isinstance(cc["c"], str)
    w = None if small else rr_wald(a, n1, c, n0)
    r = {"label": label,
         "exposed": cell(cc["a"], tot["n1"]) + f" von {num_fmt(tot['n1'])}",
         "ref": cell(cc["c"], tot["n0"]) + f" von {num_fmt(tot['n0'])}",
         "rr": "n/a (Zelle < 10)" if small else ("n/a" if w is None else f"{w[0]:.2f} ({w[1]:.2f} bis {w[2]:.2f})".replace(".", ","))}
    return r, {"a": a, "n1": n1, "c": c, "n0": n0, "rr": None if w is None else list(w)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    args = ap.parse_args(argv)
    cov, rul, ref = load()
    n = len(cov)
    both = np.isfinite(rul["R08_baz"].values) & np.isfinite(rul["R08_fri"].values)  # Population von K1 (R08: Bazett und Fridericia auswertbar)
    baz_only = np.isfinite(rul["R08_baz"].values)
    # Konsistenz: R08_baz == (qtc_baz > 500) wo auswertbar
    qb = rul["qtc_baz"].values.astype(float)
    assert np.array_equal(np.isfinite(qb), baz_only), "qtc_baz und R08_baz nicht gleich auswertbar"
    assert np.all((qb[baz_only] > 500.0) == (rul["R08_baz"].values[baz_only] == 1)), "R08_baz weicht von qtc_baz > 500 ab"
    vtvf, arr = ref["lab_vtvf"].values.astype(float), ref["lab_arrest"].values.astype(float)
    DA = np.where((vtvf == 1) | (arr == 1), 1.0, np.where((vtvf == 0) & (arr == 0), 0.0, np.nan))
    DB = ref["lab_mort30"].values.astype(float)
    outcomes = {"VT/VF oder Herzstillstand im Aufenthalt": DA, "Tod innerhalb 30 Tage": DB}

    # --- Tor: Reproduktion der K1-Entwicklungszahlen (gleiche Zellen)
    gate = {}
    for uid, nm, D in (("R08_VTVF", "VTVF_Herzstillstand", DA), ("R08_Tod", "Tod_30_Tage", DB)):
        k = json.loads((UNITS / f"{uid}.json").read_text(encoding="utf-8"))
        ref_cells = tuple(k["primaer"][nm][x] for x in ("a", "n1", "c", "n0"))
        mine = rr_counts(rul["R08_baz"].values.astype(float), D, both)
        gate[uid] = {"k1": ref_cells, "hier": mine, "gleich": ref_cells == mine, "n_auswertbar_k1": k["kohorte"]["n_regel_auswertbar"]}
        assert ref_cells == mine, f"Tor verletzt: {uid} {ref_cells} != {mine}"
    assert int(both.sum()) == gate["R08_VTVF"]["n_auswertbar_k1"]

    # --- Merkmalstafel
    rows, ex_tab, tot_exact = merkmalstafel(cov, rul, ref, both)

    # --- Herzfrequenz-Schichten (Schwelle 500, Bazett)
    hr = cov["hr"].values.astype(float)
    hr_ok = cov["hr_missing"].values == 0
    T500 = rul["R08_baz"].values.astype(float)
    strata_rows, strata_ex = [], {}
    for lab, lo, hi, key in HR_BINS:
        if key == "lt":
            m = hr < hi
        elif key == "mid":
            m = (hr >= lo) & (hr <= hi)
        else:
            m = hr > lo
        m = both & hr_ok & m
        c = cnt({"n": int(m.sum()), "pos": int((m & (T500 == 1)).sum())})
        prev = "n/a" if isinstance(c["pos"], str) or not c["n"] else f"{100.0 * c['pos'] / c['n']:.1f}".replace(".", ",")
        entry = {"label": lab, "n": c["n"], "qtc_pos": c["pos"], "anteil_qtc_pos": prev, "ergebnisse": []}
        ex_o = {}
        for on, D in outcomes.items():
            r, e = rr_row(on, T500, D, m)
            entry["ergebnisse"].append(r)
            ex_o[on] = e
        strata_rows.append(entry)
        strata_ex[lab] = {"n": int(m.sum()), "pos": int((m & (T500 == 1)).sum()), "ergebnisse": ex_o}
    n_hr_missing = int((both & ~hr_ok).sum())

    # --- Schwellen 480, 500, 520 (Bazett, gleiche Population)
    thr_rows, thr_ex = [], {}
    for thr in THRESHOLDS:
        T = np.where(np.isfinite(qb), (qb > thr).astype(float), np.nan)
        for on, D in outcomes.items():
            r, e = rr_row(f"QTc > {thr:.0f} ms", T, D, both)
            r["ergebnis"] = on
            thr_rows.append(r)
            thr_ex[f"{thr:.0f}|{on}"] = e
    # Kontrolle: Schwelle 500 = K1
    assert thr_ex["500|VT/VF oder Herzstillstand im Aufenthalt"]["a"] == gate["R08_VTVF"]["k1"][0]
    assert thr_ex["500|Tod innerhalb 30 Tage"]["c"] == gate["R08_Tod"]["k1"][2]

    # --- Ausgabe
    L = []
    L.append("# K1 Kohortentafel (PUB-13, Schritt 3), beschreibend, nachträglich festgelegt\n")
    L.append(f"Skript `werkzeuge/pub13/k1_kohortentafel.py` SHA-256 `{sha256(Path(__file__))}`; Regeldatei `K1_KOHORTENTAFEL_REGEL.md`. Eingaben: Arbeitstabellen der K1-Entwicklung (`main_cov`, `main_rules`, `main_refs` unter `<local path>`; SHA-256 {sha256(WORK / 'main_cov.parquet')[:12]}…, {sha256(WORK / 'main_rules.parquet')[:12]}…, {sha256(WORK / 'main_refs.parquet')[:12]}…). **Nur Entwicklungsdaten (train und validation); die Testpartition wurde nicht gelesen.** Alle Tabellen sind beschreibend; sie wurden nach Sicht der K1-Ergebnisse festgelegt und gehen in kein Urteil ein.\n")
    L.append("**Population.** Index-EKG je Person wie im K1-Lauf (n = %s), davon QTc nach Bazett und Fridericia aus Maschinenwerten auswertbar, ohne Vorhofflimmern im Maschinenbefund (n = %s; Bazett allein auswertbar: %s). Gruppen: QTc (Bazett, Maschinen-QT und Maschinen-RR) > 500 ms gegen <= 500 ms. Anzahlen auf 10 gerundet; Zellen von 1 bis 9 als \"<10\"; bei einer maskierten Zelle werden alle Zellen der Variablen auf 100 gerundet. Prozentwerte sind aus den gerundeten Anzahlen berechnet.\n" % (num_fmt(cnt({"n": n})["n"]), num_fmt(cnt({"n": int(both.sum())})["n"]), num_fmt(cnt({"n": int(baz_only.sum())})["n"])))
    L.append("**Tor.** Die Zellen (a, n1, c, n0) der Schwelle 500 ms stimmen für beide Ergebnisse exakt mit dem K1-Entwicklungslauf überein (VT/VF: %s; Tod 30 Tage: %s).\n" % (gate["R08_VTVF"]["hier"], gate["R08_Tod"]["hier"]))
    L.append("## Tabelle 1. Merkmale nach QTc-Gruppe\n")
    L.append("| Merkmal | QTc > 500 ms | QTc <= 500 ms | alle auswertbar |")
    L.append("|---|---|---|---|")
    for name, r in rows:
        L.append(f"| {name} | {r['QTc > 500 ms']} | {r['QTc <= 500 ms']} | {r['alle auswertbar']} |")
    L.append("")
    L.append("Versorgungsebene: Intensivstation, wenn das EKG in einen ICU-Aufenthalt fällt; sonst Notaufnahme, wenn ein ED-Aufenthalt verknüpft ist; sonst Station, wenn ein Krankenhausaufenthalt verknüpft ist; sonst übrige (K1-Kovariate `adm`). Herzinsuffizienz: Codeliste H_HERZINSUFFIZIENZ vor dem Index; berichtet nur für Personen mit früheren Diagnosecodes (Kovariate `has_prior`), weil ohne frühere Aufnahme keine Vorgeschichte besteht und der Anteil sonst unterschätzt wird. QT-Medikation: Verordnung oder berichtete Hausmedikation (Kovariate `med_qt_prolonging_drug`); berichtet nur für Personen mit Medikationsdaten (`med_assessed`), da ohne Daten die Medikation in K1 als nicht vorhanden zählt. Herzfrequenz: Maschinenwert, sonst aus RR, auf 30 bis 220/min begrenzt.\n")
    L.append("## Tabelle 2. Herzfrequenz-Schichten (QTc > 500 ms gegen <= 500 ms, Bazett)\n")
    L.append("Beschreibend; Risikoverhältnis mit Wald-Intervall (95 %), nicht für Mehrfachprüfung korrigiert, ohne Urteil. Ereignis in der QTc-Gruppe jeweils als Anzahl (Anteil) von der Zahl mit bekanntem Ergebnis.\n")
    L.append("| Schicht | Personen | davon QTc > 500 ms (Anteil) | Ergebnis | QTc > 500 ms: Ereignisse | QTc <= 500 ms: Ereignisse | Risikoverhältnis (95 %-Intervall) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in strata_rows:
        prevtxt = f"{num_fmt(s['qtc_pos'])} ({s['anteil_qtc_pos']} %)" if not isinstance(s["qtc_pos"], str) else s["qtc_pos"]
        for i, r in enumerate(s["ergebnisse"]):
            L.append(f"| {s['label'] if i == 0 else ''} | {num_fmt(s['n']) if i == 0 else ''} | {prevtxt if i == 0 else ''} | {r['label']} | {r['exposed']} | {r['ref']} | {r['rr']} |")
    L.append("")
    L.append(f"Herzfrequenz fehlt bei {num_fmt(cnt({'n': n_hr_missing})['n'])} der auswertbaren Personen (nicht in den Schichten).\n")
    L.append("## Tabelle 3. Schwellen 480, 500 und 520 ms (Bazett, gleiche Population)\n")
    L.append("Beschreibend. Die Zeile 500 ms ist die Primärdefinition von K1 (dort mit Bonferroni-Schranken; hier Wald-Intervall 95 %).\n")
    L.append("| Schwelle | Ergebnis | exponiert: Ereignisse | nicht exponiert: Ereignisse | Risikoverhältnis (95 %-Intervall) |")
    L.append("|---|---|---|---|---|")
    for r in thr_rows:
        L.append(f"| {r['label']} | {r['ergebnis']} | {r['exposed']} | {r['ref']} | {r['rr']} |")
    L.append("")
    L.append("Grenzen: Schwellen und Schichten wurden nach Ergebnissicht festgelegt. Das Risikoverhältnis ist unadjustiert. VT/VF und Herzstillstand sind gleichzeitig zum Aufenthalt erfasst (keine zeitliche Ordnung). Die Maschinen-QT-Ableitung ist nicht dokumentiert.\n")
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"gate": gate, "gruppen_exakt": tot_exact, "tafel_exakt": ex_tab, "schichten_exakt": strata_ex,
                                               "schwellen_exakt": thr_ex, "hr_fehlt_exakt": n_hr_missing}, indent=1, default=str), encoding="utf-8")
    print("fertig; Tor erfuellt:", {k: v["gleich"] for k, v in gate.items()})


if __name__ == "__main__":
    main()
