"""AW-241 (Muss 5 der Zweitmeinung v11): Modellspezifikation von B als Tabellen (nur Konstanten und Aggregate).
Quellen: werkzeuge/b_test/parameter_entwicklung.json (gebundene Entwicklungsskalierung), <local path> (S15b-Modelle,
Schwellen, Quartile, Winkelknoten), nachrechnung/core.py (Altersbasis). Koeffizienten der Entwicklungsmodelle (S13a) werden mit dem gebundenen Loeser
neu angepasst (nur train/validation) und gegen Tabelle S45 geprueft. Ausgabe nachrechnung/B_MODELLSPEZIFIKATION_AW241.md.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.auswertung")))
import numpy as np  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design  # noqa: E402
from werkzeuge.nachrechnung import core  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

ROOT = Path(str(data_path("b.auswertung")))
PKG = ROOT / "werkzeuge/b_test/parameter_entwicklung.json"
PAR = Path(str(data_path("b.aw171_parameters_json")))
OUT = ROOT / "nachrechnung/B_MODELLSPEZIFIKATION_AW241.md"


def g(x, n=7):
    return f"{x:.{n}g}"


def table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
    return s + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def scale_table(sc):
    return table(["Spalte (Reihenfolge im Design)", "Zentrierung (Mittel)", "SD (ddof=0)", "im Entwicklungsdesign enthalten"],
                 [[n, g(m), g(s), "ja" if k else "nein (in Entwicklung konstant)"] for n, m, s, k in zip(sc["names"], sc["mean"], sc["sd"], sc["keep"])])


def main():
    pkg = json.loads(PKG.read_text(encoding="utf8"))
    par = json.loads(PAR.read_text(encoding="utf8"))
    lines = ["# Modellspezifikation B (AW-241, Muss 5 der Zweitmeinung v11)", "",
             "16.10.2026. Nur Konstanten und Aggregate, keine Personenwerte. Entstanden aus Code und gespeicherten Ausgaben; keine Manuskripte geändert. "
             "Quellen: `werkzeuge/b_test/parameter_entwicklung.json` (SHA-256 `%s`), `<local path>` (SHA-256 `%s`), `werkzeuge/nachrechnung/core.py::age_basis`. "
             "Skript `werkzeuge/aw241/modellspez.py` (SHA-256 `%s`)." % (C.sha(PKG), C.sha(PAR), C.sha(__file__)), "",
             "## 1. Aufbau des Designs und Anwendung", "",
             "Die Zeilen eines Modells sind: Achsenabschnitt, dann die Spalten in der Reihenfolge der Tabellen unten. Jede Spalte x wird als z = (x − Zentrierung) / SD eingesetzt, der Achsenabschnitt als 1. "
             "Lineares Prädiktorprodukt = Summe aus Koeffizient mal z; Wahrscheinlichkeit = 1 / (1 + exp(−Prädiktor)). Die Koeffizienten stehen auf der standardisierten Skala; ein Odds-Ratio je Entwicklungs-SD des Markers ist exp(Koeffizient). "
             "Alle Mittel und SD stammen aus den kombinierten Entwicklungsfolds (train und validation), vollständige Fälle, ddof 0, und blieben in Assessment, Strata und Bootstrap fest. "
             "Geschlecht: F = 0, M = 1 (aufgezeichnet). Alter: Basis siehe Abschnitt 2. Herzfrequenz: im Assessment die zeitlich geeignete Geräte-Herzfrequenz, QRS-Dauer und QTc (Fridericia) aus der Messkette, Winkel in Grad, Marker dimensionslos. "
             "Logit-Fit mit Ridge 1e-6 auf alle Koeffizienten außer dem Achsenabschnitt (Newton, Zeilengewichte). Strata nach Geschlecht lassen die Geschlechtsspalte weg (strukturell konstant).", "",
             "## 2. Altersbasis (kubischer natürlicher Spline, feste Knoten)", "",
             "Quelle `core.age_basis`: a = Alter / 20; d(k) = ((a − k)₊³ − (a − 4)₊³) / (4 − k); die beiden Designspalten sind `age_linear` = a und `age_natural` = d(2) − d(3). "
             "Die Knoten liegen bei a = 2, 3 und 4, also bei **40, 60 und 80 Jahren**; Supplement S0 sagt fälschlich, die Knoten seien nicht tabelliert (S3 nennt sie). Quelltext:", "",
             "```", inspect.getsource(core.age_basis).strip(), "```", ""]
    lines += ["## 3. Zentrierungen und Skalierungen der Assoziationsmodelle (gebundene Entwicklungsskalierung `scaling_B3`)", ""]
    for bef, text in (("BEF-0063", "LVEF < 40 % (H1), Marker `qrs_t_normalized_discrete_frechet`, Komparator QRS-Dauer"),
                      ("BEF-0049", "Kardiomyopathie-Code (H4), Marker `qrs_t_frechet_beat_median`, Komparator QRS-Dauer"),
                      ("BEF-0087", "rechtsventrikuläre Dysfunktion (nur Kontext), Marker `qrs_t_normalized_dtw_distance`, Komparator `classical_dominant_r_v1`")):
        sc = pkg["hypotheses"][bef]["scaling_B3"]
        lines += [f"### {bef}: {text} (n vollständige Entwicklungsfälle {sc['n']})", "", scale_table(sc)]
    sc4 = pkg["hypotheses"]["BEF-0063"]["scaling_B4"]
    lines += [f"### B4 (stärkeres Orientierungsmodell) für BEF-0063 (n {sc4['n']})", "", scale_table(sc4)]
    lines += ["Hinweis: `axis_sin` und `axis_cos` sind sin und cos der frontalen QRS-Achse (Grad in Bogenmaß).", ""]
    # S15b-Modelle
    lines += ["## 4. Entwicklungsfeste Modelle und Schwellen (S15b)", "",
              "Anpassung auf den kombinierten Entwicklungsfolds, Population `eligible_reference` mit endlichem Design und bekanntem Endpunkt (n siehe Tabelle). Schwellen: %s." % par["recipes"]["threshold"], ""]
    for ep, lab in (("H1", "LVEF < 40 %"), ("H4", "Kardiomyopathie-Code")):
        for mk, mv in par["endpoints"][ep]["markers"].items():
            names = mv["scale"]["names"]
            core_names = names[:-3] + ([] if False else [])
            cols_b2 = ["Achsenabschnitt"] + [n for n in names if n not in ("spatial_qrs_t_angle_deg", mk)]
            cols_b3 = ["Achsenabschnitt"] + [n for n in names if n != mk]
            cols_bm = ["Achsenabschnitt"] + names
            rows = []
            for i in range(len(cols_bm)):
                rows.append([cols_bm[i], g(mv["scale"]["mean"][i - 1]) if i else "n/a", g(mv["scale"]["sd"][i - 1]) if i else "n/a"]
                            + [g(mv["models"][mname]["beta"][cols.index(cols_bm[i])]) if cols_bm[i] in cols else "n/a" for mname, cols in (("B2", cols_b2), ("B3", cols_b3), ("B3_marker", cols_bm))])
            rows.append(["Schwelle (linearer Prädiktor)", "", ""] + [g(mv["models"][m]["threshold"]) for m in ("B2", "B3", "B3_marker")])
            rows.append(["n / Fälle", "", ""] + [f"{mv['models'][m]['n']} / {mv['models'][m]['cases']}" for m in ("B2", "B3", "B3_marker")])
            lines += [f"### {ep} ({lab}), Marker `{mk}` (n Skalierung {mv['scale']['n']}; {mv['filter']})", "", table(["Spalte", "Zentrierung", "SD", "B2", "B3", "B3 + Marker"], rows)]
            q = mv["quartiles"]
            lines += [f"Quartilgrenzen des Markers (S15g, {q['method']}, Entwicklung; Rohskala): " + ", ".join(g(b) for b in q["bounds"]) + "; Mediane der vier Klassen: " + ", ".join(g(b) for b in q["medians"]) + "; Klassengrößen " + ", ".join(map(str, q["counts"])) + ".", ""]
    sp = par["endpoints"]["H1"]["markers"]["qrs_t_normalized_discrete_frechet"]["spline"]
    lines += ["## 5. Winkelspline (flexibler Komparator, S13d)", "",
              "Eingeschränkter kubischer Spline des Winkels mit vier Knoten bei den 5/35/65/95-%-Quantilen der Entwicklung (train): " + ", ".join(g(k) for k in sp["knots"]) + " Grad. Drei Basisspalten (linear plus zwei nichtlineare); Zentrierung " + ", ".join(g(m) for m in sp["mean"]) + "; SD " + ", ".join(g(s) for s in sp["sd"]) + ". Population: " + sp["population"] + ".", ""]
    # Verifikationsmodell
    ver = pkg["verification"]
    vrows = [["Achsenabschnitt", "n/a", "n/a", g(ver["beta"][0])]] + [[n, g(m), g(s), g(b)] for n, m, s, b in zip(ver["names"], ver["mean"], ver["sd"], ver["beta"][1:])]
    lines += ["## 6. Verifikationsmodell (markerfreie Gewichte, S8)", "",
              "Logit der Echoverfügbarkeit (±30 Tage) auf den B4-Basisspalten ohne Marker, nur train; Gewichte = 1 / Wahrscheinlichkeit, gekappt bei `%s` (99. Perzentil der Gewichte der verifizierten train-Personen)." % g(ver["p99_train_verified"]), "",
              table(["Spalte", "Zentrierung", "SD", "Koeffizient"], vrows)]
    # Entwicklungs-Koeffizienten (S13a) neu angepasst
    f, finfo, scale = C.outcome_frame()
    gm = f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    lines += ["## 7. Koeffizienten der Entwicklungsmodelle (S13a, B3 mit und ohne Marker, LVEF < 40 %)", "",
              "Neu angepasst mit dem gebundenen Löser auf den Fällen `eligible_B3` je Fold und der Skalierung aus Abschnitt 3 (BEF-0063); Prüfung gegen Tabelle S45: Marker-OR train 2,480 und validation 2,392.", ""]
    for fold in ("train", "validation"):
        sub = gm.loc[gm.split.eq(fold)]
        y = classify_lvef(sub.preferred_lvef.to_numpy(float))
        raw, names = raw_design(sub, C.F)
        z = fixed_design(raw, scale)
        w = np.ones(len(y))
        b1 = logistic_fit(z, y, w)
        b0 = logistic_fit(z[:, :-1], y, w)
        rows = [[n, g(b0[i]), g(np.exp(b0[i])), g(b1[i]), g(np.exp(b1[i]))] for i, n in enumerate(["Achsenabschnitt"] + names[:-1])]
        rows.append([names[-1], "n/a", "n/a", g(b1[-1]), g(np.exp(b1[-1]))])
        lines += [f"### {fold} (n {len(y)}, Fälle {int(y.sum())})", "", table(["Spalte", "Koeffizient ohne Marker", "exp", "Koeffizient mit Marker", "exp"], rows)]
        if abs(np.exp(b1[-1]) - {"train": 2.480, "validation": 2.392}[fold]) > 0.002:
            raise RuntimeError("Prüfgate S45 verletzt")
    lines += ["## 8. Was weiterhin fehlt", "",
              "Rechenschritte der Messkette (LoopLens-Release, Kors-Transformation, Delineation) sind durch den Code-Verweis des Manuskripts gebunden und hier nicht tabelliert. Die Koeffizienten der Assessment-Anpassungen (Neuanpassung im Assessment je Hypothese) werden nur als OR je SD mit Intervall berichtet; "
              "eine Koeffiziententabelle der Assessment-Fits liegt als Aggregat nicht vor und müsste aus den Assessment-Daten neu erzeugt werden (Testzugriff mit Eintrag).", ""]
    OUT.write_text("\n".join(lines), encoding="utf8")
    print(f"geschrieben: {OUT}, {len(lines)} Zeilen")


if __name__ == "__main__":
    main()
