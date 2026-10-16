"""PUB-13 K1 (AW-236): erzeugt die Codelisten aus Praefixregeln gegen die ICD-Tabelle d_icd_diagnoses (nur Code und
Titel, keine Patientenzeilen). Ausgabe: codelisten_k1.csv (list_id, icd_version, icd_code, long_title).
Aufruf: python -I k1_codelisten.py <d_icd_diagnoses.csv.gz> <ausgabe.csv>
"""
import csv
import gzip
import re
import sys

# list_id -> (Version 10 Praefixe, Version 9 Praefixe); Codes ohne Punkt wie in MIMIC-IV
RULES = {
    # Referenzen
    "AMYLOID": (["E85"], ["2773"]),
    "KARDIOMYOPATHIE_HINWEIS": (["I42", "I43", "E8582"], ["425"]),
    "OLD_MI": (["I252"], ["412"]),
    "ACUTE_MI": (["I21", "I22"], ["410"]),
    "NSTEMI": (["I214"], ["4107"]),
    # Population der Behauptung (Herz- oder Lungenerkrankung vor dem Index)
    "H_HERZINSUFFIZIENZ": (["I50", "I110", "I130", "I132", "I0981"],
                           ["428", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493"]),
    "H_KHK_INFARKT": (["I20", "I21", "I22", "I23", "I24", "I25"], ["410", "411", "412", "413", "414"]),
    "H_KARDIOMYOPATHIE": (["I42", "I43"], ["425"]),
    "H_KLAPPEN": (["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37", "I38", "I39"],
                  ["394", "395", "396", "397", "424"]),
    "H_VHF": (["I48"], ["4273"]),
    "H_COPD": (["J41", "J42", "J43", "J44"], ["491", "492", "496"]),
    "H_LUNGENEMBOLIE": (["I26", "I2782"], ["4151"]),
    "H_PULM_HYPERTONIE": (["I270", "I272"], ["4160", "4168"]),
    # weitere Vorgeschichte-Kovariaten des Selektionsmodells
    "K_HYPERTONIE": (["I10", "I11", "I12", "I13", "I15", "I16"], ["401", "402", "403", "404", "405"]),
    "K_DIABETES": (["E08", "E09", "E10", "E11", "E13"], ["250"]),
    "K_NIERE": (["N17", "N18", "N19"], ["584", "585", "586"]),
    "K_KREBS": ([f"C{i:02d}" for i in range(0, 98)], [str(i) for i in range(140, 209)]),
    "K_SCHLAGANFALL": (["I63", "I64"], ["434", "436"]),
}
# Titelpruefung gegen d_icd_diagnoses: ICD-9 277.31 ist Familiaeres Mittelmeerfieber, keine Amyloidose
EXCLUDE = {("AMYLOID", 9, "27731")}
EXTRA_REGEX = {"K_SCHLAGANFALL": {9: re.compile(r"^433.1$")}}  # 433.x1: Verschluss mit Hirninfarkt


def main(src, dst):
    rows = []
    with gzip.open(src, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ver = int(r["icd_version"])
            code = r["icd_code"].strip()
            for lid, (p10, p9) in RULES.items():
                pref = p10 if ver == 10 else p9
                hit = any(code.startswith(p) for p in pref)
                if not hit:
                    rx = EXTRA_REGEX.get(lid, {}).get(ver)
                    hit = bool(rx and rx.match(code))
                if hit and (lid, ver, code) not in EXCLUDE:
                    rows.append((lid, ver, code, r["long_title"].strip()))
    rows.sort()
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["list_id", "icd_version", "icd_code", "long_title"])
        w.writerows(rows)
    from collections import Counter
    c = Counter(r[0] for r in rows)
    for k in RULES:
        print(k, c.get(k, 0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
