"""Patientenbezogene Vorannahmen für die Simulation (Alter, Geschlecht, Größe, Gewicht).

Quellen:
* Eigene Regression an 75.755 maschinell als normal befundeten MIMIC-IV-ECG-Aufzeichnungen mit
  BMI aus MIMIC-IV omr (2026-09-22): QRS-Achse = 83,9 − 0,50·Alter + 3,0·weiblich − 0,92·BMI (R² 0,13);
  QRS-Dauer = 83,0 + 0,03·Alter − 5,9·weiblich + 0,22·BMI ms (R² 0,11). Bezug: Alter 57,5, BMI 29,1, Achse 29,9°.
* Sathananthan 2015 (*Clin Anat* 28:487): anatomische Herzachse bei Adipösen 25° vs. 38° (−13°), bei Älteren
  nur die elektrische Achse linkslastiger; Engblom 2005 (*Am Heart J* 150:507): elektrische und anatomische
  Achse nur schwach korreliert (r 0,3 frontal) – die Achsenvorannahme ist daher eine elektrische, keine anatomische.
* Qian et al. 2025 (*Nat Cardiovasc Res* 4:624), 3.461 digitale Zwillinge: QRS-Dauer bei Männern 9,4 % länger
  allein durch Herzmasse (181 vs. 131 g), Leitungsgeschwindigkeit geschlechtsgleich (0,58 m/s), mit BMI leicht
  höher (0,569 → 0,587), ab 65 J. niedriger (0,586 → 0,569); QTc bei Frauen 3 % länger (G_KrKs −18 %).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

REFERENCE = dict(age=57.5, bmi=29.1, axis_deg=29.9)
AXIS_BETA = (83.9, -0.50, 3.0, -0.92)  # Intercept, Alter, weiblich, BMI
QRSD_BETA = (83.0, 0.03, -5.9, 0.22)


@dataclass
class PatientPrior:
    axis_shift_deg: float = 0.0  # erwartete Verschiebung der frontalen QRS-Achse gegenüber der Referenzpopulation
    duration_factor: float = 1.0  # erwartete QRS-Dauer relativ zur Referenz (Herzgröße/Leitung)
    cv_factor: float = 1.0  # Leitungsgeschwindigkeit relativ (Qian 2025)
    qtc_factor: float = 1.0
    note: str = "keine Patientendaten – Bezugspopulation"


def patient_prior(age: Optional[float] = None, sex: Optional[str] = None, height_cm: Optional[float] = None,
                  weight_kg: Optional[float] = None) -> PatientPrior:
    def bounded(value, low, high):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) and low <= number <= high else None

    a_observed = bounded(age, 18, 110)
    sex_key = str(sex).strip().lower() if sex is not None else ""
    female = (1.0 if sex_key in {"f", "female", "weiblich", "w"} else
              0.0 if sex_key in {"m", "male", "männlich"} else 0.5)
    bmi = None
    height = bounded(height_cm, 120, 220)
    weight = bounded(weight_kg, 30, 250)
    if height is not None and weight is not None:
        bmi = bounded(weight / (height / 100.0) ** 2, 15, 55)
    if a_observed is None and female == 0.5 and bmi is None:
        return PatientPrior()
    a = a_observed if a_observed is not None else REFERENCE["age"]
    b = float(bmi) if bmi is not None else REFERENCE["bmi"]
    axis = AXIS_BETA[0] + AXIS_BETA[1] * a + AXIS_BETA[2] * female + AXIS_BETA[3] * b
    # Referenz für die Geschlechtsmischung: halber Effekt
    axis_ref = AXIS_BETA[0] + AXIS_BETA[1] * REFERENCE["age"] + AXIS_BETA[2] * 0.5 + AXIS_BETA[3] * REFERENCE["bmi"]
    qrsd = QRSD_BETA[0] + QRSD_BETA[1] * a + QRSD_BETA[2] * female + QRSD_BETA[3] * b
    qrsd_ref = QRSD_BETA[0] + QRSD_BETA[1] * REFERENCE["age"] + QRSD_BETA[2] * 0.5 + QRSD_BETA[3] * REFERENCE["bmi"]
    cv = 1.0
    if bmi is not None:
        cv *= 1.0 + 0.03 * float(np.clip((b - 25.0) / 5.0, 0.0, 1.0))  # Qian: 0,569 -> 0,587 mit BMI
    if a_observed is not None and a >= 65:
        cv *= 0.97
    qtc = 1.03 if female == 1.0 else 1.0
    parts = []
    if a_observed is not None:
        parts.append(f"{a:.0f} J")
    if female != 0.5:
        parts.append("weiblich" if female == 1.0 else "männlich")
    if bmi is not None:
        parts.append(f"BMI {b:.1f}")
    note = (", ".join(parts) + f" → explorative Achsen-Vorannahme {axis - axis_ref:+.0f}°, "
            f"QRS-Dauer ×{qrsd / qrsd_ref:.2f}; Leitungstrend ×{cv:.2f} nur Sensitivität, "
            "nicht zusätzlich angewandt; keine individuelle Anatomie")
    return PatientPrior(float(np.clip(axis - axis_ref, -30.0, 30.0)), float(np.clip(qrsd / qrsd_ref, 0.85, 1.15)), cv, qtc, note)
