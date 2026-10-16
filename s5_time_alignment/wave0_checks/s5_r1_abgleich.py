"""AW-238 S5-R1: Prozentangaben des S5-Manuskripts (v6) und des Supplements (v4) gegen Aggregate abgleichen.
Keine Patientendaten, nur Aggregatdateien (nachrechnung/*.csv) und die Manuskripttexte. Keine Schreibzugriffe ausser Ausgabe nach stdout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference
import csv, re, json, hashlib
from decimal import Decimal, ROUND_HALF_UP

BASE = (str(data_path("s5.root")) + "/")
MS = BASE + "publikationen/PUB-08_zeitausrichtung/MANUSKRIPT_PREPRINT_S5_v6.md"
SUP = BASE + "publikationen/PUB-08_zeitausrichtung/SUPPLEMENT_S5_v4.md"


def rd(path, delim):
    return list(csv.reader(open(path, encoding="utf-8-sig"), delimiter=delim))


A110 = {r[1]: r[2] for r in rd(BASE + "nachrechnung/uhrzeit_aw110.csv", ",")[1:]}
A115 = {(r[1], r[2], r[3]): r[4] for r in rd(BASE + "nachrechnung/uhrzeit_aw115.csv", ";")[1:]}
E = {}
for r in csv.DictReader(open(BASE + "nachrechnung/s5_ergaenzung.csv", encoding="utf-8-sig")):
    E[(r["aussage"], r["partition"])] = r["wert"]


def a110(k):
    return float(A110["icu_results.folds." + k])


def a115(fold, k):
    return float(A115[("primary", fold, k)])


def ee(k, p):
    return float(E[(k, p)])


def pct(num, den, dec=1):
    q = Decimal(1).scaleb(-dec)
    return (Decimal(repr(float(num))) * 100 / Decimal(repr(float(den)))).quantize(q, rounding=ROUND_HALF_UP), float(num) * 100 / float(den)


checks = []


def add(ort, text, num, den, dec, src):
    checks.append((ort, text, num, den, dec, src))


tr, va = "train", "validation"
add("Abstract/3.1 Entwicklung", "2.3%", a110("train.identifiable_people"), a110("train.primary_people"), 1, "aw110 identifiable / primary_people")
add("Abstract/3.1 interne Pruefung", "2.5%", a110("validation.identifiable_people"), a110("validation.primary_people"), 1, "aw110 identifiable / primary_people")
add("3.1 Entwicklung", "0.41%", a110("train.identifiable_people"), float(A110["prepare.basis.rows[0][2]"]), 2, "aw110 identifiable / prepare.basis Personen (96,680)")
add("3.1 interne Pruefung", "0.45%", a110("validation.identifiable_people"), float(A110["prepare.basis.rows[1][2]"]), 2, "aw110 identifiable / prepare.basis Personen (32,579)")
add("3.1 Entwicklung RR-geeignete ICU-EKG", "8.8%", float(A110["prepare.rr.rows[0][1]"]), float(A110["prepare.basis.rows[0][1]"]), 1, "aw110 prepare.rr / prepare.basis (EKG)")
add("3.1 interne Pruefung RR-geeignete ICU-EKG", "8.7%", float(A110["prepare.rr.rows[1][1]"]), float(A110["prepare.basis.rows[1][1]"]), 1, "aw110 prepare.rr / prepare.basis (EKG)")
for f, lab, t60, t15, t360 in ((tr, "Entwicklung", "67.1%", "46.2%", "88.0%"), (va, "interne Pruefung", "70.3%", "47.6%", "91.0%")):
    n = a110(f + ".identifiable_people")
    add(f"Abstract/3.1 {lab} +-60", t60, a110(f + ".within.60.count"), n, 1, "aw110 within.60.count / identifiable")
    add(f"3.1 {lab} +-15", t15, a110(f + ".within.15.count"), n, 1, "aw110 within.15.count / identifiable")
    add(f"3.1 {lab} +-360", t360, a110(f + ".within.360.count"), n, 1, "aw110 within.360.count / identifiable")
add("3.1 Entwicklung +-120", "75.0%", ee("offset.ties[0].within120", tr), a110("train.identifiable_people"), 1, "s5_ergaenzung offset.ties[0].within120 / identifiable (296/392)")
add("3.1 interne Pruefung +-120", "80.0%", ee("offset.ties[8].within120", va), a110("validation.identifiable_people"), 1, "s5_ergaenzung offset.ties[8].within120 / identifiable (116/145)")
for f, t in ((tr, ("62.2%", "71.6%")), (va, ("62.4%", "77.3%"))):
    lo = float(A110[f"icu_results.folds.{f}.bootstrap_95ci.conditional_within60[0]"])
    hi = float(A110[f"icu_results.folds.{f}.bootstrap_95ci.conditional_within60[1]"])
    add(f"3.1 CI untere Grenze {f}", t[0], lo * 1e6, 1e6, 1, "aw110 bootstrap_95ci.conditional_within60[0]")
    add(f"3.1 CI obere Grenze {f}", t[1], hi * 1e6, 1e6, 1, "aw110 bootstrap_95ci.conditional_within60[1]")
add("3.1 Entwicklung Tie-Mittelpunkt +-60", "63.8%", 250, 392, 1, "Supplement Tab. S1.6 (250/392)")
add("3.1 interne Pruefung Tie-Mittelpunkt +-60", "68.3%", 99, 145, 1, "Supplement Tab. S1.6 (99/145)")
add("3.1 Entwicklung ganze Zone innerhalb +-60", "54.8%", 215, 392, 1, "Supplement Tab. S1.6 (215/392)")
add("3.1 interne Pruefung ganze Zone innerhalb +-60", "57.2%", 83, 145, 1, "Supplement Tab. S1.6 (83/145)")
add("3.1 Stays Entwicklung Halbierungsuebereinstimmung", "42.1%", a115(tr, "half_agree_units"), a115(tr, "halves_estimable_units"), 1, "aw115 half_agree / halves_estimable (32/76)")
for f, lab, cl, ch in ((tr, "Entwicklung", "60.3%", "70.0%"), (va, "interne Pruefung", "56.8%", "73.0%")):
    add(f"3.1 Stays {lab} +-60", "65.1%", a115(f, "within60_units"), a115(f, "identifiable_units"), 1, "aw115 within60_units / identifiable_units")
    add(f"3.1 Stay-CI untere {lab}", cl, a115(f, "bootstrap_95ci.within60[0]") * 1e6, 1e6, 1, "aw115 bootstrap_95ci.within60[0]")
    add(f"3.1 Stay-CI obere {lab}", ch, a115(f, "bootstrap_95ci.within60[1]") * 1e6, 1e6, 1, "aw115 bootstrap_95ci.within60[1]")
add("3.3 Verlust Entwicklung", "34.8%", 23151 - 15092, 23151, 1, "Supplement Flow S3.1 (23,151 -> 15,092)")
add("3.3 Verlust interne Pruefung", "33.8%", 7945 - 5257, 7945, 1, "Supplement Flow S3.1 (7,945 -> 5,257)")
add("3.3 Kontaktende Entwicklung vor Vollst.", "43.6%", ee("AF[4].non_AF_contact_end", tr), ee("AF[4].n", tr) - ee("AF[4].events", tr), 1, "s5_ergaenzung AF[4]: 9,769 / (23,151-767)")
add("3.3 Kontaktende interne Pruefung vor Vollst.", "43.7%", ee("AF[10].non_AF_contact_end", va), ee("AF[10].n", va) - ee("AF[10].events", va), 1, "s5_ergaenzung AF[4]: 3,356 / (7,945-274)")
add("3.3 Kontaktende Entwicklung nach Vollst.", "43.5%", ee("AF[5].non_AF_contact_end", tr), ee("AF[5].n", tr) - ee("AF[5].events", tr), 1, "s5_ergaenzung AF[5]: non_AF_contact_end / (n - events)")
add("3.3 Kontaktende interne Pruefung nach Vollst.", "44.0%", ee("AF[11].non_AF_contact_end", va), ee("AF[11].n", va) - ee("AF[11].events", va), 1, "s5_ergaenzung AF[5]: non_AF_contact_end / (n - events)")
add("4 Minute null Entwicklung", "87.2%", 2564374, 2941325, 1, "Supplement S1.5 (2,564,374/2,941,325)")
add("4 Minute null interne Pruefung", "87.2%", 873796, 1001797, 1, "Supplement S1.5 (873,796/1,001,797)")
if reference("s5_r1_abgleich.s33_tod_nach_kontakt_entwicklung_n") is None: raise SystemExit("Local reference value s5_r1_abgleich.s33_tod_nach_kontakt_entwicklung_n is missing; the numerator of the Supplement S3.3 check is not published.")
add("S3.3 Tod nach Kontakt Entwicklung", "75.4%", reference("s5_r1_abgleich.s33_tod_nach_kontakt_entwicklung_n"), 997, 1, "Supplement S3.3")
add("S3.3 Tod nach Kontakt interne Pruefung", "73.6%", 240, 326, 1, "Supplement S3.3")
add("S1.3 Stays Identifizierbarkeit Entwicklung", "7.8%", a115(tr, "identifiable_units"), a115(tr, "rr_at_least3_units"), 1, "aw115 350/4467")
add("S1.3 Stays Identifizierbarkeit interne Pruefung", "8.7%", a115(va, "identifiable_units"), a115(va, "rr_at_least3_units"), 1, "aw115 129/1477")
add("S1.3 Halbierungsuebereinstimmung Kriteriumsnenner Entwicklung", "9.1%", a115(tr, "half_agree_units"), a115(tr, "identifiable_units"), 1, "aw115 32/350")

bad = 0
print("| Ort | Textwert | Zaehler | Nenner | exakt | gerundet | Ergebnis | Abweichung (Pp) | Quelle |")
print("|---|---|---:|---:|---:|---:|---|---:|---|")
for ort, text, num, den, dec, src in checks:
    r, raw = pct(num, den, dec)
    t = Decimal(text.rstrip("%"))
    ok = (r == t)
    if not ok:
        bad += 1
    nn = f"{num:.0f}" if abs(num) < 1e7 and float(num).is_integer() else f"{num:.1f}"
    print(f"| {ort} | {text} | {nn} | {den:.0f} | {raw:.3f}% | {r}% | {'ok' if ok else 'ABWEICHUNG'} | {raw - float(t):+.2f} | {src} |")
print("geprueft:", len(checks), "Abweichungen:", bad)

# Alle Prozentangaben im Haupttext, die nicht in checks stehen
txt = open(MS, encoding="utf-8").read().split("\n")[:196]
covered = {c[1] for c in checks}
print("\nNICHT als Zaehler/Nenner geprueft (Schwellen, Kriterien, Intervallgrenzen u.a.):")
for i, l in enumerate(txt, 1):
    for m in re.finditer(r"(?<![\d.,])(\d{1,3}(?:\.\d+)?)%", l):
        if m.group(0) not in covered:
            print(i, m.group(0), "|", l[max(0, m.start() - 50):m.end() + 20])
for p in (MS, SUP):
    print(p.split("/")[-1], hashlib.sha256(open(p, "rb").read()).hexdigest())
print("script", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
