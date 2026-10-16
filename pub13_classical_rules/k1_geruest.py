"""PUB-13 K1: Geruest des Auswertungsskripts (Stand v3.2). Rechnet NICHTS gegen Referenzen.

Rahmenfunktionen, die das spaetere Auswertungsskript benutzt:
  - Seed aus dem Hash der gebundenen Spezifikation (Errata aendern ihn nicht), je Regel und Verwendung ein Strom,
  - Pruefung aller Eingabe-Hashes gegen ein Manifest vor dem Lauf,
  - Moduswaechter mit Registerpruefung: `locked_test` nur im Modus "test" und nur, wenn `TESTZUGRIFFE.md` eine
    Zeile "Beginn" mit den Hashes S, C, M, D enthaelt; ein einziger Ladepfad filtert den Split per WHERE-Klausel,
  - Kleinzellenregel: Rundung auf 10, "<10" fuer 1 bis 9, Komplementvergroeberung,
  - Entscheidungstabellen je Familie (disjunkte, erschoepfende Ausgaenge, feste Vorrangfolge), Bonferroni-
    angepasste einseitige Schranken (alpha/m), Spezifitaets-Trivialschranke, Gleich-Risiko-QRS per Inversion.
Das Waechterkonzept schuetzt gegen Versehen, nicht gegen Absicht (so im Paper beschrieben).

Aufruf (Selbsttest): python -I k1_geruest.py <pfad zur Spezifikation>
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

DEV_SPLITS = ("train", "validation")
TEST_SPLIT = "locked_test"
B_BOOT = 4000
ALPHA_FAMILY = 0.025  # einseitig, je Familie
SMALL_CELL = 10

# Komponentenhypothesen je Familie (Bonferroni-Nenner m, v3.2 Abschnitt 8)
FAMILY_M = {"E": 8, "A": 2, "P": 4, "S": 3}

HAELT, HAELT_NICHT, UNENTSCHIEDEN = "haelt", "haelt nicht", "unentschieden"


# ------------------------------------------------------------------ Hash, Seed, Manifest
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_from_spec(spec_hash_hex: str) -> int:
    """Erste 8 Hexstellen des SHA-256 der GEBUNDENEN Spezifikation; Errata nach dem Freeze aendern den Seed
    nicht (AE-309 C10)."""
    return int(spec_hash_hex[:8], 16)


def rng_for(seed: int, rule_index: int, stream: int = 0) -> np.random.Generator:
    """Strom 0 Bootstrap, 1 Gewichtsmodell, 2 Test."""
    return np.random.default_rng(np.random.SeedSequence([seed, rule_index, stream]))


def verify_manifest(manifest_path: str | Path) -> list[str]:
    problems = []
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for path, expected in man.items():
        p = Path(path)
        if not p.exists():
            problems.append(f"fehlt: {path}")
        elif sha256_file(p) != expected:
            problems.append(f"Hash abweichend: {path}")
    return problems


# ------------------------------------------------------------------ Testregister und Waechter (M9)
def check_test_register(register_path: str | Path, spec_hash: str, script_hash: str, manifest_hash: str,
                        dev_results_hash: str) -> bool:
    """True, wenn TESTZUGRIFFE.md eine Zeile mit dem Ereignis 'Beginn' enthaelt, die S, C, M und D (volle
    SHA-256) nennt. Das Register wird gelesen, nicht nur eine Marke verglichen."""
    p = Path(register_path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        if "Beginn" in line and all(h and h in line for h in (spec_hash, script_hash, manifest_hash, dev_results_hash)):
            return True
    return False


def register_line(event: str, auftrag: str, spec_hash: str, script_hash: str, manifest_hash: str,
                  dev_results_hash: str, scope: str, result_hash: str | None = None,
                  now: _dt.datetime | None = None) -> str:
    """Tabellenzeile fuer TESTZUGRIFFE.md ('Beginn', 'Fortsetzung', 'Ende'); Keine Kennungen."""
    t = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    regs = f"S `{spec_hash}`; C `{script_hash}`; M `{manifest_hash}`; D `{dev_results_hash}`"
    if result_hash:
        regs += f"; E `{result_hash}`"
    return f"| {t} | {auftrag} | {event}: einmalige Testrechnung K1 | {regs} | {scope} |"


def append_register(register_path: str | Path, line: str) -> None:
    with open(register_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def guard_split(mode: str, split: str, register_path: str | Path | None = None, spec_hash: str | None = None,
                script_hash: str | None = None, manifest_hash: str | None = None,
                dev_results_hash: str | None = None) -> None:
    """Dev-Modus: nur train und validation. Test-Modus: locked_test nur mit Registereintrag 'Beginn'."""
    if split in DEV_SPLITS:
        return
    if split == TEST_SPLIT and mode == "test" and register_path and all(
            (spec_hash, script_hash, manifest_hash, dev_results_hash)):
        if check_test_register(register_path, spec_hash, script_hash, manifest_hash, dev_results_hash):
            return
    raise PermissionError(f"Split {split!r} im Modus {mode!r} nicht freigegeben")


def partition_sql(relation: str, split: str, mode: str, split_column: str = "split", **guard_args) -> str:
    """Einziger Ladepfad: ruft den Waechter und liefert eine SELECT-Anweisung, die den Split per WHERE-Klausel
    festlegt. Alle Partitionszugriffe des Skripts laufen darueber (Test: Dev-Modus mit locked_test scheitert)."""
    guard_split(mode, split, **guard_args)
    return f"SELECT * FROM {relation} WHERE {split_column} = '{split}'"


# ------------------------------------------------------------------ Kleinzellenregel (M10)
def mask_small(count: int) -> str | int:
    """1 bis 9 als '<10', sonst auf 10 gerundet."""
    if 0 < count < SMALL_CELL:
        return "<10"
    return int(round(count / 10.0) * 10)


def coarsen_counts(cells: dict[str, int], totals: dict[str, int] | None = None) -> dict:
    """Kleinzellenregel mit Komplementvergroeberung: Zellen auf 10 gerundet, 1 bis 9 als '<10'. Ist mindestens
    eine Zelle maskiert, werden alle uebrigen Zellen und alle Randsummen auf 100 gerundet, damit die maskierte
    Zelle nicht aus Summen zurueckgerechnet werden kann. Werte 0 bleiben 0."""
    any_masked = any(0 < v < SMALL_CELL for v in cells.values())
    unit = 100 if any_masked else 10
    out = {}
    for k, v in cells.items():
        out[k] = "<10" if 0 < v < SMALL_CELL else int(round(v / unit) * unit)
    if totals is not None:
        out["_totals"] = {k: int(round(v / unit) * unit) for k, v in totals.items()}
    return out


# ------------------------------------------------------------------ Schranken und Entscheidungstabellen (B3)
def alpha_component(family: str, alpha: float = ALPHA_FAMILY) -> float:
    """Bonferroni: einseitiges Niveau je Komponentenhypothese = alpha / m der Familie."""
    return alpha / FAMILY_M[family]


def one_sided_bounds(boot: Sequence[float], alpha_c: float) -> tuple[float, float]:
    """Untere und obere einseitige Perzentilschranke zum Niveau 1 - alpha_c aus Bootstrap-Wiederholungen."""
    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    return float(np.quantile(b, alpha_c)), float(np.quantile(b, 1.0 - alpha_c))


def classify_ge(lower: float, upper: float, theta: float) -> str:
    """Behauptung 'Wert >= theta' (Spezifitaet, Anteil, Risiko): haelt, wenn das Intervall ganz im
    Behauptungsbereich liegt (lower >= theta); haelt nicht, wenn es ganz ausserhalb liegt (upper < theta)."""
    if lower >= theta:
        return HAELT
    if upper < theta:
        return HAELT_NICHT
    return UNENTSCHIEDEN


def classify_le(lower: float, upper: float, theta: float) -> str:
    """Behauptung 'Wert <= theta' (Sensitivitaet 'wenig sensitiv'): haelt, wenn upper <= theta; haelt nicht, wenn
    lower > theta."""
    if upper <= theta:
        return HAELT
    if lower > theta:
        return HAELT_NICHT
    return UNENTSCHIEDEN


def classify_hr_nonharm(lower: float, upper: float, margin: float = 1.20) -> tuple[str, str]:
    """Familie P ('harmlos'): Behauptungsbereich HR < margin. haelt: upper < margin; haelt nicht: lower > margin;
    sonst unentschieden. Hinweisklasse (kein Ausgang): 'Erhoehung nachgewiesen', wenn lower > 1,0."""
    if upper < margin:
        cls = HAELT
    elif lower > margin:
        cls = HAELT_NICHT
    else:
        cls = UNENTSCHIEDEN
    hint = "Erhoehung nachgewiesen (untere Schranke > 1,0)" if lower > 1.0 else ""
    return cls, hint


def classify_marker(lower: float, upper: float, null: float = 1.0, equiv_upper: float = 1.25) -> tuple[str, str]:
    """R08 ('QTc > 500 ist ein Risikomarker'): haelt, wenn lower > 1,0 (Vorrang); haelt nicht, wenn nicht 'haelt'
    und upper < 1,25; sonst unentschieden. Hinweisklasse 'klein erhoeht', wenn lower > 1,0 und upper < 1,25."""
    if lower > null:
        hint = "klein erhoeht (obere Schranke < 1,25)" if upper < equiv_upper else ""
        return HAELT, hint
    if upper < equiv_upper:
        return HAELT_NICHT, ""
    return UNENTSCHIEDEN, ""


def classify_equiv(lower: float, upper: float, margin: float = 10.0) -> str:
    """R19 (gleiche Schwelle): Behauptungsbereich (-margin, +margin). haelt: Intervall ganz darin; haelt nicht:
    Intervall ganz ausserhalb (upper < -margin oder lower > margin); sonst unentschieden."""
    if lower > -margin and upper < margin:
        return HAELT
    if upper < -margin or lower > margin:
        return HAELT_NICHT
    return UNENTSCHIEDEN


def combine(classes: Iterable[str], mode: str = "all") -> str:
    """Einheit aus mehreren Komponenten mit fester Vorrangfolge. mode 'all' (Behauptung gilt, wenn ALLE
    Komponenten halten): haelt nicht hat Vorrang vor unentschieden vor haelt (eine verfehlte Komponente genuegt
    fuer 'haelt nicht'). Die Ausgaenge sind disjunkt und erschoepfend."""
    cs = list(classes)
    if not cs:
        return UNENTSCHIEDEN
    if mode != "all":
        raise ValueError(mode)
    if any(c == HAELT_NICHT for c in cs):
        return HAELT_NICHT
    if all(c == HAELT for c in cs):
        return HAELT
    return UNENTSCHIEDEN


def trivial_specificity_bound(p_test_pos: float, p_disease_neg: float) -> float:
    """Untere Schranke der Spezifitaet allein aus der Positivrate: spec >= 1 - P(T+)/P(D-) (B6)."""
    if p_disease_neg <= 0:
        return float("nan")
    return max(0.0, 1.0 - p_test_pos / p_disease_neg)


def predetermined_by_positive_rate(p_test_pos: float, p_disease_neg: float, theta: float) -> bool:
    """True, wenn die Trivialschranke schon ueber theta liegt ('haelt (durch Positivrate vorbestimmt)')."""
    b = trivial_specificity_bound(p_test_pos, p_disease_neg)
    return bool(np.isfinite(b) and b >= theta)


# ------------------------------------------------------------------ R19 Gleich-Risiko-QRS (M12)
QRS_KNOTS = (90.0, 110.0, 130.0, 150.0, 170.0)  # feste Knoten des natuerlichen Splines
QRS_GRID = (90, 200)  # ms, 1-ms-Raster


def equal_risk_qrs(grid: Sequence[float], risk_f: Sequence[float], target: float):
    """Inversion: QRS-Wert, bei dem die Frauenkurve das Zielrisiko (Maennerrisiko bei 150 ms) erreicht.
    Rueckgabe (wert, status): status 'ok' bei genau einem Schnittpunkt (lineare Interpolation auf dem Raster),
    'nicht pruefbar (kein Schnittpunkt)' oder 'nicht pruefbar (mehrfacher Schnittpunkt)'."""
    g = np.asarray(grid, dtype=float)
    r = np.asarray(risk_f, dtype=float) - target
    crossings = []
    for i in range(len(g) - 1):
        a, b = r[i], r[i + 1]
        if a == 0:
            crossings.append(g[i])
        elif a * b < 0:
            crossings.append(g[i] + (g[i + 1] - g[i]) * (0 - a) / (b - a))
    if r[-1] == 0:
        crossings.append(g[-1])
    uniq = sorted(set(round(c, 6) for c in crossings))
    if not uniq:
        return None, "nicht pruefbar (kein Schnittpunkt)"
    if len(uniq) > 1:
        return None, "nicht pruefbar (mehrfacher Schnittpunkt)"
    return float(uniq[0]), "ok"


PLAN = [
    ("1 kohorte", "build_cohort(mode) -> Personenliste, Index-EKG je Person (erstes Entwicklungs-EKG; R20: erstes EKG im ersten ED-Aufenthalt mit Brustschmerz-Triage und Troponin am selben Kalendertag), Ausschluesse aus EKG-Messgroessen und Maschinenaussagen"),
    ("2 regeln", "evaluate_rules(cohort) -> Regelzustand True/False/None je Regel und Variante (regeln.py plus adapter_cache.py; maschine_normal, maschine_sinus)"),
    ("3 referenzen", "build_references(cohort) -> Referenzen und Fenster nach Abschnitt 4; Trennschutz: keine Referenzspalte vor Stufe 5 mit Regelspalten verbunden"),
    ("4 gewichte", "fit_weights(cohort, rule) -> stabilisierte IPW je Regel (Verifikation = Referenz berechenbar, Abschnitt 5.1), Kappung 1./99. Perzentil, Positivitaet"),
    ("5 schaetzer", "estimate(rule, data, weights, rng) -> Kennzahl, Bootstrap B=4000 mit Neuanpassung der Gewichte; Schranken per one_sided_bounds(alpha_component(Familie))"),
    ("6 urteile", "verdict(rule, estimate) -> classify_* und combine nach Abschnitt 6.A; Hinweisklassen; Trivialschranke"),
    ("7 ausgabe", "write_outputs(results) -> nur Aggregate, coarsen_counts, Hashes; Zielordner E:\\VCG\\auswertung\\pub13_k1\\, nicht OneDrive"),
]


if __name__ == "__main__":
    spec = Path(sys.argv[1])
    h = sha256_file(spec)
    s = seed_from_spec(h)
    print("Spezifikation:", spec.name, h)
    print("Seed:", s)
    print("Strom 0 Regel 0:", rng_for(s, 0, 0).integers(0, 10**6, 3).tolist())
    guard_split("dev", "train")
    for bad in ("locked_test", "sealed"):
        try:
            guard_split("dev", bad)
        except PermissionError as e:
            print("Waechter ok:", e)
    print("Kleinzellen:", mask_small(0), mask_small(3), mask_small(14), mask_small(1234))
    for k, v in PLAN:
        print(k, "|", v)
