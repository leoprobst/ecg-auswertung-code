"""Eigenständige Prüf-GUI für die Herzillustration (ohne das ECG-Labor).

Start:  py -3.12 -m vcg_biomarker.gui.heart3d.demo
        py -3.12 -m vcg_biomarker.gui.heart3d.demo --wfdb PFAD/record   (ohne Endung)
        py -3.12 -m vcg_biomarker.gui.heart3d.demo --csv PFAD/12leads.csv --fs 500

Synthetische Szenarien erzeugen 12-Kanal-Signale aus Dipolvektoren (Dower-Rückprojektion
aus Kors), damit Rhythmusstreifen, Vektor-basierte Ursprungsbestimmung und Muster
zusammenpassen. Sie sind Lehrbeispiele, keine Patientendaten.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget)

from .timeline import KORS, LEADS_12, Beat, BeatTimeline, qrs_vector_from_signal
from .widgets import MiniHeartWidget

_PINV = np.linalg.pinv(KORS)  # (8, 3): XYZ -> I, II, V1..V6 (kleinste Norm)


def _leads12_from_8(eight: np.ndarray) -> np.ndarray:
    """(n, 8: I, II, V1..V6) -> (n, 12) in Standardreihenfolge."""
    I, II = eight[:, 0], eight[:, 1]
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    return np.column_stack([I, II, III, aVR, aVL, aVF, eight[:, 2:]])


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


class Scenario:
    def __init__(self, name: str, duration: float = 10.0, fs: float = 500.0):
        self.name = name
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0.0, duration, 1.0 / fs)
        self.xyz = np.zeros((len(self.t), 3))
        self.beats: list[Beat] = []
        self.atrial_events: Optional[list[tuple[float, float]]] = None
        self.rhythm_class = "regular"
        self.hints: dict = {}
        self.label = name

    # ---- Bausteine (Vektoren im VCG-Raum: X links, Y inferior, Z posterior)
    def add_p(self, t0: float, duration: float = 0.10, vector=(0.10, 0.12, 0.00)) -> None:
        self.xyz += np.outer(_gauss(self.t, t0 + duration / 2, duration / 4.5), vector)

    def add_qrs(self, t0: float, duration: float = 0.09, vector=(1.3, 1.0, -0.4), septal=(-0.15, -0.05, 0.25)) -> None:
        self.xyz += np.outer(_gauss(self.t, t0 + 0.012, 0.007), septal)
        self.xyz += np.outer(_gauss(self.t, t0 + duration * 0.45, duration / 5.0), vector)
        self.xyz += np.outer(_gauss(self.t, t0 + duration * 0.78, duration / 11.0), np.array(vector) * -0.25)

    def add_t(self, t0: float, duration: float = 0.18, vector=(0.45, 0.40, -0.15)) -> None:
        self.xyz += np.outer(_gauss(self.t, t0 + duration / 2, duration / 4.0), vector)

    def add_st(self, qrs_off: float, t_end: float, vector) -> None:
        """ST-Plateau (Verletzungsstrom) vom QRS-Ende bis in die T-Welle, weich ein-/ausgeblendet."""
        from scipy.special import expit
        rise = expit((self.t - qrs_off + 0.01) / 0.006)
        fall = expit(-(self.t - t_end + 0.04) / 0.02)
        self.xyz += np.outer(rise * fall, vector)

    def add_fibrillation(self, amplitude: float = 0.04) -> None:
        rng = np.random.default_rng(3)
        f = 6.0
        wave = amplitude * np.sin(2 * np.pi * f * self.t + 2 * np.pi * rng.random()) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * self.t))
        wave += amplitude * 0.5 * rng.standard_normal(len(self.t))
        from scipy.signal import butter, filtfilt
        b, a = butter(2, [3 / (self.fs / 2), 10 / (self.fs / 2)], btype="band")
        wave = filtfilt(b, a, wave)
        self.xyz += np.outer(wave, (0.6, 0.8, 0.3))

    def add_flutter(self, rate_per_min: float = 300.0) -> None:
        period = 60.0 / rate_per_min
        saw = (np.mod(self.t, period) / period) - 0.5
        self.xyz += np.outer(-0.12 * saw, (0.3, 1.0, 0.2))

    def beat(self, index: int, qrs_on: float, qrs_dur: float, p_on: Optional[float], rr: Optional[float],
             beat_type: str = "N", qt: float = 0.38) -> Beat:
        b = Beat(index=index, r_time=qrs_on + qrs_dur * 0.45, qrs_on=qrs_on, qrs_off=qrs_on + qrs_dur,
                 p_on=p_on, p_off=(p_on + 0.10) if p_on is not None else None,
                 t_on=qrs_on + qrs_dur + 0.05, t_end=qrs_on + qt, rr_prev=rr, beat_type=beat_type, source="payload")
        self.beats.append(b)
        return b

    def timeline(self) -> BeatTimeline:
        eight = self.xyz @ _PINV.T
        signal = _leads12_from_8(eight)
        for b in self.beats:
            b.qrs_vector = qrs_vector_from_signal(signal, self.fs, b.qrs_on, b.qrs_off, LEADS_12)
        tl = BeatTimeline(self.beats, self.duration, self.fs, rhythm_class=self.rhythm_class,
                          conduction_hints=self.hints, source="payload", signal=signal, lead_names=LEADS_12,
                          atrial_events=self.atrial_events, label=self.label)
        return tl


def scenario_sinus(rate: float = 72.0, qrs_dur: float = 0.09, hints: Optional[dict] = None, name: str = "Sinusrhythmus") -> Scenario:
    sc = Scenario(name)
    rr = 60.0 / rate
    k = 0
    t0 = 0.3
    prev = None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.16
        sc.add_p(p_on)
        sc.add_qrs(t0, qrs_dur)
        sc.add_t(t0 + qrs_dur + 0.06)
        sc.beat(k, t0, qrs_dur, p_on, (t0 - prev) if prev is not None else rr)
        prev = t0
        t0 += rr
        k += 1
    sc.hints = dict(hints or {})
    return sc


def scenario_lbbb() -> Scenario:
    sc = Scenario("Sinus mit Linksschenkelblock")
    rr = 60.0 / 68
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.17
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.15, vector=(1.6, 0.6, 0.6), septal=(-0.05, 0.0, 0.05))
        sc.add_t(t0 + 0.15 + 0.05, vector=(-0.5, -0.2, -0.4))
        sc.beat(k, t0, 0.15, p_on, (t0 - prev) if prev is not None else rr, qt=0.44)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.hints = {"lbbb": True, "rbbb": False}
    return sc


def scenario_rbbb() -> Scenario:
    sc = Scenario("Sinus mit Rechtsschenkelblock")
    rr = 60.0 / 70
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.16
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.14, vector=(1.0, 0.8, -0.3))
        sc.xyz += np.outer(_gauss(sc.t, t0 + 0.11, 0.015), (-0.7, 0.1, -0.8))  # später Rechtsvektor
        sc.add_t(t0 + 0.14 + 0.05, vector=(0.5, 0.3, 0.3))
        sc.beat(k, t0, 0.14, p_on, (t0 - prev) if prev is not None else rr, qt=0.42)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.hints = {"lbbb": False, "rbbb": True}
    return sc


def scenario_af() -> Scenario:
    sc = Scenario("Vorhofflimmern")
    rng = np.random.default_rng(7)
    sc.add_fibrillation()
    k, t0, prev = 0, 0.4, None
    while t0 + 0.5 < sc.duration:
        sc.add_qrs(t0, 0.09)
        sc.add_t(t0 + 0.09 + 0.06)
        sc.beat(k, t0, 0.09, None, (t0 - prev) if prev is not None else None)
        rr = float(np.clip(rng.normal(0.72, 0.18), 0.4, 1.4))
        prev, t0, k = t0, t0 + rr, k + 1
    sc.rhythm_class = "af_compatible_pattern"
    return sc


def scenario_bigeminy() -> Scenario:
    sc = Scenario("Ventrikuläre Extrasystolen (Bigeminus, Ursprung LV-lateral)")
    rr = 60.0 / 66
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.16
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.09)
        sc.add_t(t0 + 0.15)
        sc.beat(k, t0, 0.09, p_on, (t0 - prev) if prev is not None else rr)
        k += 1
        tv = t0 + 0.46
        # Fokus LV-lateral/basal: Vektor zeigt nach rechts-anterior-inferior (vom Fokus weg)
        sc.add_qrs(tv, 0.15, vector=(-1.4, 0.9, -0.9), septal=(0.0, 0.0, 0.0))
        sc.add_t(tv + 0.15 + 0.04, vector=(0.7, -0.3, 0.5))
        sc.beat(k, tv, 0.15, None, tv - t0, beat_type="V", qt=0.42)
        k += 1
        prev = tv
        t0 = t0 + rr + 0.34  # kompensatorische Pause
    sc.rhythm_class = "mixed_or_ectopic_pattern"
    return sc


def scenario_flutter() -> Scenario:
    sc = Scenario("Vorhofflattern 2:1")
    sc.add_flutter(300.0)
    rr = 0.4
    k, t0, prev = 0, 0.35, None
    while t0 + 0.5 < sc.duration:
        sc.add_qrs(t0, 0.09)
        sc.add_t(t0 + 0.15, vector=(0.3, 0.3, -0.1))
        sc.beat(k, t0, 0.09, None, (t0 - prev) if prev is not None else rr, qt=0.34)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.rhythm_class = "flutter_compatible_pattern"
    return sc


def scenario_av_block() -> Scenario:
    sc = Scenario("AV-Block III (Vorhof 80/min, Kammerersatz 38/min)")
    atrial = []
    t = 0.2
    while t + 0.2 < sc.duration:
        sc.add_p(t)
        atrial.append((t, t + 0.10))
        t += 60.0 / 80
    rr = 60.0 / 38
    k, t0, prev = 0, 0.9, None
    while t0 + 0.5 < sc.duration:
        sc.add_qrs(t0, 0.14, vector=(1.2, 0.3, 0.5), septal=(0.0, 0.0, 0.0))
        sc.add_t(t0 + 0.14 + 0.06, vector=(-0.3, -0.2, -0.3))
        sc.beat(k, t0, 0.14, None, (t0 - prev) if prev is not None else rr, beat_type="V", qt=0.48)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.atrial_events = atrial
    sc.rhythm_class = "regular"
    sc.label = sc.name
    return sc


def scenario_wpw() -> Scenario:
    sc = Scenario("Präexzitation (linkslaterale Bahn)")
    rr = 60.0 / 74
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.10
        sc.add_p(p_on)
        sc.xyz += np.outer(_gauss(sc.t, t0 + 0.02, 0.02), (-0.3, 0.2, 0.1))  # Delta
        sc.add_qrs(t0 + 0.03, 0.10, vector=(1.2, 0.9, -0.3), septal=(0.0, 0.0, 0.0))
        sc.add_t(t0 + 0.13 + 0.05, vector=(-0.3, 0.2, -0.2))
        sc.beat(k, t0, 0.13, p_on, (t0 - prev) if prev is not None else rr, qt=0.40)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.hints = {"wpw": True, "delta_axis_deg": 60.0}
    return sc


def scenario_paced() -> Scenario:
    sc = Scenario("Rechtsventrikuläre Stimulation (VVI 70/min)")
    rr = 60.0 / 70
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        sc.xyz += np.outer(_gauss(sc.t, t0, 0.0015), (0.0, -2.0, 0.0))  # Spike
        sc.add_qrs(t0 + 0.005, 0.16, vector=(1.4, -0.6, 0.8), septal=(0.0, 0.0, 0.0))
        sc.add_t(t0 + 0.17 + 0.05, vector=(-0.5, 0.3, -0.4))
        sc.beat(k, t0, 0.16, None, (t0 - prev) if prev is not None else rr, beat_type="P", qt=0.46)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.hints = {"paced": True}
    return sc


def scenario_stemi_inferior() -> Scenario:
    sc = Scenario("Inferiorer STEMI (ST-Vektor nach inferior, Spiegelbild in aVL)")
    rr = 60.0 / 64
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.16
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.09)
        sc.add_st(t0 + 0.09, t0 + 0.40, (0.05, 0.32, 0.05))
        sc.add_t(t0 + 0.15, vector=(0.3, 0.7, -0.1))
        sc.beat(k, t0, 0.09, p_on, (t0 - prev) if prev is not None else rr, qt=0.40)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_stemi_anterior() -> Scenario:
    sc = Scenario("Anteriorer STEMI (ST-Vektor nach anterior, LAD)")
    rr = 60.0 / 88
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.15
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.09, vector=(1.0, 0.8, -0.1))
        sc.add_st(t0 + 0.09, t0 + 0.38, (0.10, 0.05, -0.38))
        sc.add_t(t0 + 0.15, vector=(0.4, 0.3, -0.6))
        sc.beat(k, t0, 0.09, p_on, (t0 - prev) if prev is not None else rr, qt=0.38)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_subendocardial() -> Scenario:
    sc = Scenario("Subendokardiale Ischämie (diffuse ST-Senkung, Hebung in aVR)")
    rr = 60.0 / 96
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.15
        sc.add_p(p_on)
        sc.add_qrs(t0, 0.09)
        sc.add_st(t0 + 0.09, t0 + 0.36, (-0.16, -0.16, 0.10))
        sc.add_t(t0 + 0.15, vector=(0.2, 0.15, -0.05))
        sc.beat(k, t0, 0.09, p_on, (t0 - prev) if prev is not None else rr, qt=0.36)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_old_anterior_mi() -> Scenario:
    sc = Scenario("Alter anteriorer Infarkt (Q-Zacken V1–V4, verlorene anteriore Kräfte)")
    rr = 60.0 / 70
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.16
        sc.add_p(p_on)
        # Initialkräfte zeigen nach posterior (weg von der anterioren Narbe), danach normaler Hauptvektor
        sc.xyz += np.outer(_gauss(sc.t, t0 + 0.02, 0.012), (-0.2, 0.1, 0.9))
        sc.xyz += np.outer(_gauss(sc.t, t0 + 0.05, 0.016), (1.0, 0.9, 0.3))
        sc.add_t(t0 + 0.15, vector=(-0.2, 0.1, 0.4))
        sc.beat(k, t0, 0.10, p_on, (t0 - prev) if prev is not None else rr)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_lvh() -> Scenario:
    sc = Scenario("Linksventrikuläre Hypertrophie mit Schädigungsmuster")
    rr = 60.0 / 66
    k, t0, prev = 0, 0.3, None
    while t0 + 0.5 < sc.duration:
        p_on = t0 - 0.18
        sc.add_p(p_on, vector=(0.12, 0.10, 0.12))  # P mitrale: terminal posterior
        sc.add_qrs(t0, 0.10, vector=(2.9, 1.6, -0.7))
        sc.add_st(t0 + 0.10, t0 + 0.40, (-0.10, -0.06, 0.06))
        sc.add_t(t0 + 0.16, vector=(-0.6, -0.4, 0.3))
        sc.beat(k, t0, 0.10, p_on, (t0 - prev) if prev is not None else rr, qt=0.42)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_junctional() -> Scenario:
    sc = Scenario("Junktionaler Ersatzrhythmus 45/min (schmales QRS, keine P-Welle)")
    rr = 60.0 / 45
    k, t0, prev = 0, 0.4, None
    while t0 + 0.5 < sc.duration:
        sc.add_qrs(t0, 0.09)
        sc.add_p(t0 + 0.10, vector=(-0.08, -0.10, 0.02))  # retrograde P nach dem QRS (negativ in II)
        sc.add_t(t0 + 0.17)
        sc.beat(k, t0, 0.09, None, (t0 - prev) if prev is not None else rr)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_ventricular_escape() -> Scenario:
    sc = Scenario("Ventrikulärer Ersatzrhythmus 35/min (breites QRS)")
    rr = 60.0 / 35
    k, t0, prev = 0, 0.5, None
    while t0 + 0.5 < sc.duration:
        sc.add_qrs(t0, 0.16, vector=(1.3, -0.5, 0.7), septal=(0.0, 0.0, 0.0))
        sc.add_t(t0 + 0.22, vector=(-0.5, 0.3, -0.4))
        sc.beat(k, t0, 0.16, None, (t0 - prev) if prev is not None else rr, beat_type="V", qt=0.5)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_avnrt() -> Scenario:
    sc = Scenario("AV-Knoten-Reentry-Tachykardie 170/min (P im QRS verborgen)")
    rr = 60.0 / 170
    k, t0, prev = 0, 0.3, None
    while t0 + 0.4 < sc.duration:
        sc.add_qrs(t0, 0.085)
        sc.add_t(t0 + 0.13, duration=0.14)
        sc.beat(k, t0, 0.085, None, (t0 - prev) if prev is not None else rr, qt=0.28)
        prev, t0, k = t0, t0 + rr, k + 1
    return sc


def scenario_vt() -> Scenario:
    sc = Scenario("Kammertachykardie 160/min (Kreiserregung, Austritt LV-lateral)")
    rr = 60.0 / 160
    k, t0, prev = 0, 0.3, None
    while t0 + 0.4 < sc.duration:
        sc.add_qrs(t0, 0.16, vector=(-1.4, 0.9, -0.9), septal=(0.0, 0.0, 0.0))
        sc.add_t(t0 + 0.2, duration=0.14, vector=(0.7, -0.3, 0.5))
        sc.beat(k, t0, 0.16, None, (t0 - prev) if prev is not None else rr, beat_type="V", qt=0.34)
        prev, t0, k = t0, t0 + rr, k + 1
    sc.rhythm_class = "wide_qrs_pattern"
    return sc


SCENARIOS = {
    "Sinusrhythmus 72/min": lambda: scenario_sinus(),
    "Sinustachykardie 120/min": lambda: scenario_sinus(120.0, 0.08, name="Sinustachykardie"),
    "Sinusbradykardie 45/min": lambda: scenario_sinus(45.0, name="Sinusbradykardie"),
    "Linksschenkelblock": scenario_lbbb,
    "Rechtsschenkelblock": scenario_rbbb,
    "Vorhofflimmern": scenario_af,
    "VES-Bigeminus (LV-lateral)": scenario_bigeminy,
    "Vorhofflattern 2:1": scenario_flutter,
    "AV-Block III": scenario_av_block,
    "Präexzitation": scenario_wpw,
    "RV-Stimulation": scenario_paced,
    "Inferiorer STEMI": scenario_stemi_inferior,
    "Anteriorer STEMI": scenario_stemi_anterior,
    "Subendokardiale Ischämie": scenario_subendocardial,
    "Alter anteriorer Infarkt": scenario_old_anterior_mi,
    "LV-Hypertrophie": scenario_lvh,
    "Junktionaler Ersatzrhythmus": scenario_junctional,
    "Ventrikulärer Ersatzrhythmus": scenario_ventricular_escape,
    "AVNRT 170/min": scenario_avnrt,
    "Kammertachykardie (Reentry)": scenario_vt,
}


def load_wfdb(path: str) -> BeatTimeline:
    import wfdb
    from .timeline import timeline_from_signal
    record = wfdb.rdrecord(path)
    signal = np.asarray(record.p_signal, dtype=np.float64)
    names = tuple(str(n).upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in record.sig_name)
    tl = timeline_from_signal(signal, float(record.fs), names)
    tl.label = path
    return tl


def load_csv(path: str, fs: float) -> BeatTimeline:
    from .timeline import timeline_from_signal
    data = np.genfromtxt(path, delimiter=",", skip_header=0)
    if data.ndim == 1:
        data = data[:, None]
    if np.isnan(data[0]).any():
        data = data[1:]
    tl = timeline_from_signal(data, fs, LEADS_12[:data.shape[1]])
    tl.label = path
    return tl


class DemoWindow(QMainWindow):
    def __init__(self, initial: Optional[BeatTimeline] = None):
        super().__init__()
        self.setWindowTitle("Herzillustration – Prüf-GUI")
        central = QWidget()
        layout = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(QLabel("Szenario"))
        self.combo = QComboBox()
        self.combo.addItems(list(SCENARIOS))
        row.addWidget(self.combo, 1)
        start = QPushButton("Laden")
        start.clicked.connect(self._load_scenario)
        row.addWidget(start)
        wfdb_button = QPushButton("WFDB-Datei …")
        wfdb_button.clicked.connect(self._pick_wfdb)
        row.addWidget(wfdb_button)
        layout.addLayout(row)
        hint = QLabel("Die kleine Ansicht entspricht dem geplanten Eckfenster im ECG-Labor; „Vergrößern“ öffnet das Steuerfenster "
                      "mit Rhythmusstreifen, Geschwindigkeit, Ebenen und Musterwahl.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.mini = MiniHeartWidget(self)
        layout.addWidget(self.mini, 0)
        layout.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.resize(560, 420)
        if initial is not None:
            self.mini.set_timeline(initial)
            self.status.setText(initial.label)
        else:
            self._load_scenario()

    def _load_scenario(self) -> None:
        name = self.combo.currentText()
        scenario = SCENARIOS[name]()
        tl = scenario.timeline()
        self.mini.set_timeline(tl)
        self.status.setText(f"{tl.label} · {len(tl.beats)} Schläge · Rhythmusklasse {tl.rhythm_class} · Hinweise {tl.conduction_hints or '–'}")

    def _pick_wfdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WFDB-Header wählen", "", "WFDB (*.hea)")
        if not path:
            return
        try:
            tl = load_wfdb(path[:-4] if path.endswith(".hea") else path)
        except Exception as exc:
            self.status.setText(f"Laden fehlgeschlagen: {exc}")
            return
        self.mini.set_timeline(tl)
        self.status.setText(f"{tl.label} · {len(tl.beats)} Schläge (Schätzung; Analyse des Hauptprojekts läuft im Hintergrund)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prüf-GUI für die Herzillustration")
    parser.add_argument("--wfdb", help="WFDB-Record (Pfad ohne Endung)")
    parser.add_argument("--csv", help="CSV mit 12 Spalten (I, II, III, aVR, aVL, aVF, V1..V6) in mV")
    parser.add_argument("--fs", type=float, default=500.0)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    args = parser.parse_args(argv)
    if QApplication.instance() is None:
        from PyQt6.QtCore import Qt
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)  # wie main.py: mehrere GL-Ansichten
    app = QApplication.instance() or QApplication(sys.argv)
    initial = None
    if args.wfdb:
        initial = load_wfdb(args.wfdb)
    elif args.csv:
        initial = load_csv(args.csv, args.fs)
    elif args.scenario:
        initial = SCENARIOS[args.scenario]().timeline()
    window = DemoWindow(initial)
    window.show()
    window.mini.open_window()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
