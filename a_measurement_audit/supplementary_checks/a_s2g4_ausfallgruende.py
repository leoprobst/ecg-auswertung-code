#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""a_s2g4_ausfallgruende.py: Aggregation der beiden 31er-Gruppen "nicht schaetzbar" (Paper A, Supplement S2g-4).

Zweck
    Belegt aus den veroeffentlichten Marker-Aggregaten, warum 31 Ausgaben im Gesamtvergleich (Gruppe a) und
    31 Ausgaben der historischen 1.755er-Menge im +-4-ms-Kors/Kors-Grenzversuch (Gruppe b) nicht schaetzbar
    sind. Nur Lesen; keine Neuberechnung von Markerwerten, keine Patientenebene.

Eingabe (Standard: Ordner PUB-01_messvaliditaet, anderer Pfad als 1. Argument)
    supplementary_data_S2_marker_matrix_public_v4.csv (eine Zeile je Marker, 2.005 Zeilen)

Regeln (Prioritaet von oben nach unten, wie im Text von S2g-4)
    1. Historische Menge: technical_class in {dependent, conditional} (1.071 + 684 = 1.755).
    2. Je Marker 12 Zellen: sechs Verschiebungen (QRS-Beginn, QRS-Ende, T-Ende, je -4 und +4 ms) x zwei Falten;
       Zellklasse = Spalte "{Verschiebung}_development_fold_{1,2}_class".
    3. Ein Marker mit mindestens einer Zelle, die nicht stable, conditional oder dependent ist
       (not_estimable oder constant_identical), gilt als "nicht schaetzbar", auch wenn andere Zellen
       classifizierbar sind (diese Zellen werden dann nicht gewertet).
    4. Sonst gilt die schlechteste Klasse ueber die zwoelf Zellen (stable < conditional < dependent).
    5. Ausfallgrund einer nicht klassifizierbaren Zelle: weniger als 100 endliche Paare (Spalte *_finite_pairs;
       leer = unterdrueckt, also weniger als 10, zaehlt als weniger als 100) oder constant_identical.
    Gruppe (a): technical_class == not_estimable; Grund: weniger als 100 endliche Paare in einer Falte
    (Spalten development_fold_{1,2}_finite_pairs). Zirkulaer: Spalte circular_marker.

Erwartung (Gate, bricht bei Abweichung ab): 295 / 933 / 496 / 31 und Gruppe (a) = 31.
Aufruf: python -I -B a_s2g4_ausfallgruende.py [Ordner]   (kein Bytecode, nichts geschrieben)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import collections
import os
import sys

ordner = sys.argv[1] if len(sys.argv) > 1 else str(data_path("a.publikationen_pub_01_messvaliditaet"))
datei = os.path.join(ordner, "supplementary_data_S2_marker_matrix_public_v4.csv")
with open(datei, encoding="utf-8", newline="") as f:
    zeilen = list(csv.DictReader(f))
assert len(zeilen) == 2005, len(zeilen)

SHIFTS = ["qrs_onset_minus_4_ms", "qrs_onset_plus_4_ms", "qrs_offset_minus_4_ms",
          "qrs_offset_plus_4_ms", "t_offset_minus_4_ms", "t_offset_plus_4_ms"]
RANG = {"stable": 0, "conditional": 1, "dependent": 2}


def zahl(s):
    return None if s == "" else float(s)


historisch = [r for r in zeilen if r["technical_class"] in ("dependent", "conditional")]
print("historische Menge:", len(historisch))

summary = collections.Counter()
nicht = []
for r in historisch:
    zellen = []
    for s in SHIFTS:
        for fold in (1, 2):
            zellen.append((r[f"{s}_development_fold_{fold}_class"], zahl(r[f"{s}_development_fold_{fold}_finite_pairs"])))
    if any(k not in RANG for k, _ in zellen):
        summary["nicht schaetzbar"] += 1
        nicht.append((r, zellen))
    else:
        summary[max((k for k, _ in zellen), key=lambda k: RANG[k])] += 1
print("Grenzversuch:", dict(summary))
assert (summary["stable"], summary["conditional"], summary["dependent"], summary["nicht schaetzbar"]) == (295, 933, 496, 31), summary
print("schaetzbar:", 295 + 933 + 496)

grund = collections.Counter()
ueberschrieben = collections.Counter()
familie_b = collections.Counter()
mit_null_paaren = 0
for r, zellen in nicht:
    kleiner100 = False
    konstant = False
    andere = False
    null = False
    for k, p in zellen:
        if k == "constant_identical":
            konstant = True
        elif k == "not_estimable":
            if p is None or p < 100:
                kleiner100 = True
                if p == 0:
                    null = True
            else:
                andere = True
        else:
            ueberschrieben[k] += 1
    grund[(kleiner100, konstant, andere)] += 1
    mit_null_paaren += null
    familie_b[r["family"]] += 1
print("Gruppe b Gruende (weniger_als_100, konstant, andere):", dict(grund))
print("Gruppe b: Marker mit mindestens einer Zelle ohne endliche Paare:", mit_null_paaren)
print("Gruppe b: klassifizierbare Zellen der nicht schaetzbaren Marker (nicht gewertet):", dict(ueberschrieben))
print("Gruppe b Familien:", dict(familie_b))

gruppe_a = [r for r in zeilen if r["technical_class"] == "not_estimable"]
assert len(gruppe_a) == 31
in_mind_einer = sum(1 for r in gruppe_a if any((zahl(r[f"development_fold_{k}_finite_pairs"]) or 0) < 100 for k in (1, 2)))
in_beiden = sum(1 for r in gruppe_a if all((zahl(r[f"development_fold_{k}_finite_pairs"]) or 0) < 100 for k in (1, 2)))
print("Gruppe a: weniger als 100 Paare in mindestens einer Falte:", in_mind_einer, "in beiden:", in_beiden)
print("Gruppe a: zirkulaer:", collections.Counter(r["circular_marker"] for r in gruppe_a))
print("Gruppe a Familien:", dict(collections.Counter(r["family"] for r in gruppe_a)))
ids_a = {r["marker_identifier"] for r in gruppe_a}
ids_b = {r["marker_identifier"] for r, _ in nicht}
print("Schnittmenge a und b:", len(ids_a & ids_b))
