"""AW-238 B-R1: Auswertung der Laeufe (nur Aggregate in der Ausgabe, Kleinzellenregel beachtet).

Liest <local path>*.json (eigene private Replikatlisten), die gespeicherten AW-172-Replikatlisten
(<local path><Schluessel>_bootstrap.json) und die AW-142-Ergebnisse. Schreibt
nachrechnung/AW238/B_R1_ergebnisse.csv und gibt die Tabellen als Markdown aus.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(str(data_path("b.aw238")))
AW172 = Path(str(data_path("b.aw172")))
AW142 = Path(str(data_path("b.aw136_b_testlauf_aw142_ergebnis_json")))
CSV = Path(str(data_path("b.nachrechnung_aw238_b_r1_ergebnisse_csv")))
MIN_VALID = 950
A_TASKS = {
    "A_6a": ("045", "6a QRS <120 ms, ohne Block", "QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster / BEF-0049", 754, {"conv": 244, "distinct": 2}),
    "A_6c_lt65": ("047", "6c Alter <65", "Alter <65 / BEF-0049", 815, {"conv": 181, "distinct": 4}),
    "A_6d_F": ("049", "6d Geschlecht F", "Geschlecht F / BEF-0049", 93, {"conv": 2, "distinct": 905}),
}
B_TASKS = [
    ("S15_H1_no_qtc_extended", "H1/no_qtc_extended"), ("S15_H1_no_qtc_C", "H1/no_qtc_C"), ("S15_H1_missing_indicator_extended", "H1/missing_indicator_extended"),
    ("S15_H4_no_qtc_extended", "H4/no_qtc_extended"), ("S15_H4_no_qtc_C", "H4/no_qtc_C"), ("S15_H4_missing_indicator_extended", "H4/missing_indicator_extended"),
    ("S15_H1_standardized", "H1/standardized"), ("S15_H4_standardized", "H4/standardized"),
    ("S15_H1_cross_F", "H1/cross F"), ("S15_H1_cross_M", "H1/cross M"), ("S15_H4_cross_F", "H4/cross F"), ("S15_H4_cross_M", "H4/cross M"),
    ("S15_H1_W24_rbbb", "H1/RBBB-Stratum"),
]


def ci(vals):
    v = np.asarray(vals, float)
    return None if len(v) < MIN_VALID else np.quantile(v, [0.025, 0.975], method="linear").tolist()


def fmt(c, d=3):
    return "nicht schaetzbar" if c is None else f"[{c[0]:.{d}f}; {c[1]:.{d}f}]"


def stored_name(ep, kind):
    return f"{ep}_W24_rbbb" if kind == "W24_rbbb" else f"{ep}_{kind}"


def main():
    rows = []
    md = []
    old = json.loads(AW142.read_text(encoding="utf8"))
    md.append("### Teil I: S3b-Zeilen (Motor: Original statistik.fit, Union U142)\n")
    md.append("| S3b-Zeile | Analyse | Punktschaetzer OR (Wald-KI, alt) | gueltig alt (veroeffentlicht) | gueltig alt (hier mit alter Regel rekonstruiert) | Ausfaelle alt rekonstruiert | gueltig neu | Ausfaelle neu | OR Bootstrap-KI neu | Delta-AUC Bootstrap-KI neu | Loeserkontrolle gueltig | OR-KI Loeserkontrolle | Delta-AUC-KI Loeserkontrolle |")
    md.append("|---|---|---|---:|---:|---|---:|---|---|---|---:|---|---|")
    for name, (row, label, orig, pub_valid, pub_fail) in A_TASKS.items():
        p = OUT / f"{name}.json"
        if not p.exists():
            md.append(f"| {row} | {label} | fehlt | | | | | | | | | | |")
            continue
        s = json.loads(p.read_text(encoding="utf8"))
        reps = s["reps"]
        n = len(reps)
        ok = [r for r in reps if r["ok"]]
        old_valid = [r for r in ok if min(r["distinct_cases"], r["distinct_controls"]) >= 100]
        old_distinct = sum(1 for r in reps if min(r["distinct_cases"], r["distinct_controls"]) < 100)
        old_conv = sum(1 for r in reps if min(r["distinct_cases"], r["distinct_controls"]) >= 100 and not r["ok"])
        fails = {}
        for r in reps:
            if not r["ok"]:
                fails[r["reason"]] = fails.get(r["reason"], 0) + 1
        vok = [r for r in reps if r.get("v_ok")]
        orci, aci = ci([r["m"]["or"] for r in ok]), ci([r["m"]["delta_auc"] for r in ok])
        vor, vauc = ci([r["v"]["or"] for r in vok]), ci([r["v"]["delta_auc"] for r in vok])
        pt = s["point"]
        wald = [x for x in old["sensitivities"] if x.get("analysis") == orig and x.get("rank") == {"A_6a": "6a", "A_6c_lt65": "6c", "A_6d_F": "6d"}[name]][0]["or_ci95_wald"]
        md.append(f"| {row} | {label} | {pt['or']:.2f} ({wald[0]:.2f}-{wald[1]:.2f}) | {pub_valid} | {len(old_valid)} | Konvergenz {old_conv}, <100 verschiedene {old_distinct} (veroeffentlicht: Konvergenz {pub_fail['conv']}, <100: {pub_fail['distinct']}) | {len(ok)} von {n} | {json.dumps(fails, ensure_ascii=False)} | {fmt(orci)} | {fmt(aci, 4)} | {len(vok)} | {fmt(vor)} | {fmt(vauc, 4)} |")
        rows.append(dict(teil="I", task=name, s3b_zeile=row, alt_gueltig_veroeffentlicht=pub_valid, alt_gueltig_rekonstruiert=len(old_valid), neu_gueltig=len(ok), geplant=n,
                         or_punkt=pt["or"], or_ci_neu=fmt(orci), dauc_punkt=pt["delta_auc"], dauc_ci_neu=fmt(aci, 4), loeser_gueltig=len(vok), or_ci_loeser=fmt(vor), dauc_ci_loeser=fmt(vauc, 4),
                         ausfaelle_neu=json.dumps(fails, ensure_ascii=False)))
    md.append("\n### Teil II: S15-Zweige (Motor: aw171_kernels, Union U172, 1.000 statt 500 Ziehungen)\n")
    md.append("| Zweig | Punkt OR | gueltig alt (veroeffentlicht) | gueltig neu | Ausfaelle neu | OR Bootstrap-KI neu | Delta-AUC Bootstrap-KI neu | Reproduktion r<500 (gleich/abweichend/nur neu/nur alt) |")
    md.append("|---|---:|---:|---:|---|---|---|---|")
    for name, label in B_TASKS:
        p = OUT / f"{name}.json"
        if not p.exists():
            md.append(f"| {label} | fehlt | | | | | | |")
            continue
        s = json.loads(p.read_text(encoding="utf8"))
        _, ep, kind = name.split("_", 2)
        reps = s["reps"]
        ok = [r for r in reps if r["ok"]]
        fails = {}
        for r in reps:
            if not r["ok"]:
                fails[r["reason"]] = fails.get(r["reason"], 0) + 1
        st = json.loads((AW172 / f"{stored_name(ep, kind)}_bootstrap.json").read_text(encoding="utf8"))
        stored = {r["replicate"]: r["metrics"] for r in st["rows"]}
        mine = {r["r"]: r["m"] for r in ok}
        same = diff = only_new = only_old = 0
        for r in range(st["next"]):
            a, b = stored.get(r), mine.get(r)
            if a is None and b is None:
                continue
            if a is None:
                only_new += 1
            elif b is None:
                only_old += 1
            else:
                keys = [kk for kk in a if kk in b]
                good = all(np.isclose(a[kk], b[kk], rtol=1e-9, atol=1e-12) for kk in keys)
                same += int(good)
                diff += int(not good)
        pt = s["point"]
        entry = dict(teil="II", task=name, zweig=label, alt_gueltig_veroeffentlicht=len(stored), neu_gueltig=len(ok), geplant=len(reps), or_punkt=pt.get("or"),
                     reproduktion_gleich=same, reproduktion_abweichend=diff, nur_neu=only_new, nur_alt=only_old, ausfaelle_neu=json.dumps(fails, ensure_ascii=False))
        if kind == "standardized":
            cells = {m_: ci([r["m"][m_] for r in ok]) for m_ in ("q1", "q2", "q3", "q4", "q4_minus_q1")}
            md.append(f"| {label} | Q1 {pt['q1']:.4f}, Q4 {pt['q4']:.4f} | {len(stored)} | {len(ok)} | {json.dumps(fails, ensure_ascii=False)} | Q1 {fmt(cells['q1'], 4)}; Q2 {fmt(cells['q2'], 4)}; Q3 {fmt(cells['q3'], 4)}; Q4 {fmt(cells['q4'], 4)}; Q4-Q1 {fmt(cells['q4_minus_q1'], 4)} | | {same}/{diff}/{only_new}/{only_old} |")
            for m_, c_ in cells.items():
                entry[f"ci_{m_}"] = fmt(c_, 4)
            entry.update(q1_punkt=pt["q1"], q2_punkt=pt["q2"], q3_punkt=pt["q3"], q4_punkt=pt["q4"], q4_minus_q1_punkt=pt["q4_minus_q1"])
        else:
            orci, aci = ci([r["m"]["or"] for r in ok]), ci([r["m"]["delta_auc"] for r in ok])
            md.append(f"| {label} | {pt['or']:.3f} | {len(stored)} | {len(ok)} | {json.dumps(fails, ensure_ascii=False)} | {fmt(orci)} | {fmt(aci, 4)} | {same}/{diff}/{only_new}/{only_old} |")
            entry.update(or_ci_neu=fmt(orci), dauc_punkt=pt.get("delta_auc"), dauc_ci_neu=fmt(aci, 4))
        rows.append(entry)
    if (OUT / "mi_denominators.json").exists():
        mi = json.loads((OUT / "mi_denominators.json").read_text(encoding="utf8"))
        md.append("\n### Teil III: Nenner der multiplen Imputation (M = 20)\n")
        md.append("| Endpunkt | Referenz-gueltig D (n / Faelle / Kontrollen) | MI-Menge (Marker und Winkel bekannt) | vollstaendige Faelle | imputierte Personen |")
        md.append("|---|---|---|---|---|")

        def sup(v):
            return "<10" if 0 < v < 10 else str(v)
        for ep, d in mi.items():
            def t(x):
                return f"{x['n']:,} / {sup(x['cases'])} / {sup(x['controls'])}"
            md.append(f"| {ep} | {t(d['reference_valid_D'])} | {t(d['mi_set'])} | {t(d['complete_cases'])} | {t(d['imputed_persons'])} |")
            rows.append(dict(teil="III", task="MI_" + ep, **{k_: json.dumps(v_) for k_, v_ in d.items() if k_ != "missing_by_variable"}))
    keys = sorted({k_ for r in rows for k_ in r})
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n".join(md))
    print("\nCSV", CSV, hashlib.sha256(CSV.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
