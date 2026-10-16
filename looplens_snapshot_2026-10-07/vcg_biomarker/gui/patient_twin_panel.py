"""Patient constraints and quantitative cellular assays in the research workspace."""
from dataclasses import asdict
import json
import hashlib
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextBrowser, QVBoxLayout, QWidget, QSlider, QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer

from ..simulation.patient import FIELDS, Observation, PatientProfile, twin_readiness

LABELS = ("Alter", "Größe", "Gewicht", "Thoraxbreite", "Thoraxtiefe", "Herzfrequenz",
          "LV-Innendurchmesser diastolisch", "Septumdicke", "Hinterwanddicke", "LV-EDV",
          "Kalium", "Natrium", "Ionisiertes Calcium")


class PatientTwinPanel(QWidget):
    def __init__(self, launch, parent=None):
        super().__init__(parent)
        self.launch = launch
        self.bundle = None
        self.geometry_source = "generic"
        layout = QVBoxLayout(self)
        note = QLabel("PATIENTENZWILLING · Forschungsparameter\nGemessene Werte und gelernte Verteilungen bleiben getrennt. "
                      "Körpermaße identifizieren keine individuelle Leitungsbahn. Leere Eingaben bleiben unbekannt.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.sex = QComboBox()
        for label, value in (("Unbekannt", "unknown"), ("Weiblich (Dokumentation)", "female"), ("Männlich (Dokumentation)", "male")):
            self.sex.addItem(label, value)
        form.addRow("Dokumentiertes Geschlecht", self.sex)
        self.index = QLineEdit()
        self.index.setPlaceholderText("ISO-Zeit des EKGs, z. B. 2026-09-23T10:00:00+02:00")
        form.addRow("Indexzeit", self.index)
        self.aligned = QCheckBox("Zeitabgleich der Messungen tatsächlich verifiziert")
        form.addRow(self.aligned)
        layout.addLayout(form)
        self.table = QTableWidget(len(FIELDS), 6)
        self.table.setHorizontalHeaderLabels(["Messung", "Wert", "Einheit", "Quelle", "Messzeit (ISO)", "Unsicherheit (SD)"])
        for row, (key, (unit, _, _)) in enumerate(FIELDS.items()):
            for col, value in enumerate((LABELS[row], "", unit, "", "", "")):
                item = QTableWidgetItem(value)
                if col in (0, 2):
                    from PyQt6.QtCore import Qt
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 65)
        self.table.setColumnWidth(2, 70)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 2)
        buttons = QHBoxLayout()
        for label, callback, tip in (
                ("Profil öffnen", self.open_profile, "Lokal gespeichertes Forschungsprofil (JSON) laden"),
                ("Profil speichern", self.save_profile, "Aktuelles Profil als neue JSON-Datei sichern; bestehende Dateien werden nicht überschrieben"),
                ("Kalibrierung laden", self.load_priors, "Populationskalibrierung laden und daraus Vorannahmen für dieses Profil schätzen"),
                ("Parameter prüfen", self.evaluate, "Messwerte und populationsbasierte Vorannahmen im Bericht gegenüberstellen")):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.report = QTextBrowser()
        self.report.setPlainText("Noch kein Patientenprofil geprüft. Keine automatischen Alters-/BMI-Multiplikatoren für Ionenkanäle.")
        layout.addWidget(self.report, 1)

    def profile(self):
        observations = {}
        for row, (key, (unit, _, _)) in enumerate(FIELDS.items()):
            value = self.table.item(row, 1).text().strip()
            if not value:
                continue
            source, stamp, uncertainty = (self.table.item(row, col).text().strip() for col in (3, 4, 5))
            observations[key] = Observation(float(value.replace(",", ".")), unit, source,
                                            stamp or None, float(uncertainty.replace(",", ".")) if uncertainty else None)
        profile = PatientProfile(observations, self.sex.currentData(), self.index.text().strip() or None,
                                 self.aligned.isChecked(), self.geometry_source)
        profile.validate()
        return profile

    def evaluate(self):
        try:
            profile = self.profile()
            lines = ["MESSWERTE (werden nicht durch Prior-Schätzungen überschrieben)"]
            for key, obs in profile.observations.items():
                lines.append(f"{key}: {obs.value:g} {obs.unit} · {obs.source}")
            lines.append("\nPOPULATIONSBASIERTE VORANNAHMEN · keine individuellen Normalwerte")
            if self.bundle:
                from ..simulation.calibration import predict_priors
                lines.append("Verwendete Eingaben: " + ", ".join(self.bundle.get("features", [])))
                for key, estimate in predict_priors(self.bundle, profile).items():
                    if estimate["status"] == "population_prior":
                        lo, hi = estimate["interval_90"]
                        lines.append(f"{key}: {estimate['center']:.1f} [{lo:.1f}, {hi:.1f}] {estimate['unit']} · 90-%-Intervall")
                    else:
                        lines.append(f"{key}: {estimate.get('reason', estimate['status'])}")
            else:
                lines.append("Kein versioniertes Kalibrierungsartefakt geladen.")
            lines.extend(["\nEIGNUNG", json.dumps(twin_readiness(profile), ensure_ascii=False, indent=2),
                          "\nZEITLICH ZUORDENBARE ELEKTROLYTE", json.dumps(profile.ionic_constraints(), ensure_ascii=False, indent=2)])
            self.report.setPlainText("\n".join(lines))
        except (ValueError, TypeError, KeyError) as exc:
            self.report.setPlainText("Profil nicht verwendbar: " + str(exc))

    def save_profile(self):
        try:
            profile = self.profile()
            path, _ = QFileDialog.getSaveFileName(self, "Lokales Forschungsprofil", "patient-profile.json", "JSON (*.json)")
            if path:
                with Path(path).open("x", encoding="utf-8") as handle:
                    json.dump({"schema": "patient-twin-profile/1", "profile": profile.to_dict()}, handle, indent=2)
        except (ValueError, OSError) as exc:
            self.report.setPlainText(str(exc))

    def open_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Forschungsprofil", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("schema") != "patient-twin-profile/1":
                raise ValueError("Unbekanntes Profilformat")
            profile = PatientProfile.from_dict(data["profile"])
            self.sex.setCurrentIndex(self.sex.findData(profile.sex_recorded))
            self.index.setText(profile.index_time or "")
            self.aligned.setChecked(profile.clock_alignment_verified)
            self.geometry_source = profile.geometry_source
            for row, key in enumerate(FIELDS):
                obs = profile.observations.get(key)
                values = (str(obs.value), obs.source, obs.measured_at or "", str(obs.uncertainty) if obs.uncertainty else "") if obs else ("", "", "", "")
                for col, value in zip((1, 3, 4, 5), values):
                    self.table.item(row, col).setText(value)
            self.evaluate()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.report.setPlainText(str(exc))

    def load_priors(self):
        path, _ = QFileDialog.getOpenFileName(self, "Populationskalibrierung", "artifacts/patient_twin", "JSON (*.json)")
        if not path:
            return
        try:
            from ..simulation.calibration import predict_priors
            bundle = json.loads(Path(path).read_text(encoding="utf-8"))
            predict_priors(bundle, self.profile())
            self.bundle = bundle
            self.evaluate()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.report.setPlainText(str(exc))


class IonicCellPanel(QWidget):
    def __init__(self, launch, patient_panel, parent=None):
        super().__init__(parent)
        self.launch, self.patient_panel = launch, patient_panel
        self.tissue_data = None
        self.tissue_gl = None
        self.channel_scales = {}
        self.katp_open_fraction = 0.
        self.genetic_variant = "reference"
        self.precondition_max_beats = 0
        self.regional_scenario = False
        self.regional_k = 7.2
        self.s2_onset_ms = 0.
        self.tissue_duration_ms = 60.
        self.tissue_units = "mV"
        layout = QVBoxLayout(self)
        note = QLabel("IONENLABOR · Quantitative Einzelzellmodelle\nVeröffentlichte menschliche Ionenkanalmodelle, "
                      "adaptive Integration. Kein Ganzherz-EKG und keine automatisch patientenspezifische Zellphysiologie.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.model = QComboBox()
        for name, key in (("Ventrikel · Ten Tusscher–Panfilov 2006", "tentusscher-2006"),
                          ("Ventrikel · O’Hara–Rudy 2011 (inkl. spätem INa)", "ohara-2011"),
                          ("Vorhof · Courtemanche 1998", "courtemanche-1998"),
                          ("Purkinje · Stewart 2009", "stewart-2009"), ("Sinusknoten · Fabbri 2017", "fabbri-2017")):
            self.model.addItem(name, key)
        form.addRow("Zellmodell", self.model)
        self.phenotype = QComboBox()
        self.phenotype.addItems(["Endokard", "Epikard", "M-Zelle (Modellannahme)"])
        self.phenotype.setCurrentIndex(1)
        form.addRow("Ventrikeltyp (TP06 / O’Hara)", self.phenotype)
        self.cycle = QDoubleSpinBox()
        self.cycle.setRange(250, 2500)
        self.cycle.setValue(800)
        self.cycle.setSuffix(" ms")
        form.addRow("Zyklus / SAN-Beobachtungsfenster", self.cycle)
        rate_button = QPushButton("Gemessene Herzfrequenz als Pacing-Vorschlag übernehmen")
        rate_button.setToolTip("Zykluslänge aus der gemessenen Herzfrequenz des Profils setzen; Stimulationsprotokoll, keine Sinusknotenphysiologie")
        rate_button.clicked.connect(self.apply_rate)
        form.addRow(rate_button)
        self.beats = QSpinBox()
        self.beats.setRange(1, 20)
        self.beats.setValue(3)
        form.addRow("Zyklen (keine Steady-State-Garantie)", self.beats)
        self.use_electrolytes = QCheckBox("Verifizierte Elektrolyte aus Patientenprofil als Szenario übernehmen")
        form.addRow(self.use_electrolytes)
        layout.addLayout(form)
        setup = QPushButton("Kanäle, regionales Substrat und S1/S2 …")
        setup.setToolTip("Kanalmultiplikatoren, künstliches regionales Substrat und S1/S2-Protokoll einstellen")
        setup.clicked.connect(self.configure_mechanisms)
        layout.addWidget(setup)
        self.run_button = QPushButton("Zellphysiologie berechnen")
        self.run_button.setToolTip("Einzelzellversuch mit den gewählten Einstellungen im Hintergrund rechnen")
        self.run_button.clicked.connect(self.start)
        layout.addWidget(self.run_button)
        self.metabolism_button = QPushButton("Stoffwechsel und mitochondriale Energetik …")
        self.metabolism_button.setToolTip("Stoffwechsel- und Mitochondrienversuch einrichten und rechnen")
        self.metabolism_button.clicked.connect(self.start_metabolism)
        layout.addWidget(self.metabolism_button)
        actions = QHBoxLayout()
        self.tissue_button = QPushButton("3D-Ionengewebe berechnen")
        self.tissue_button.setToolTip("Gewebeversuch auf dem eingebauten Blocknetz mit publizierten Elektrolytwerten rechnen")
        self.tissue_button.clicked.connect(lambda: self.start_tissue(False))
        self.mesh_button = QPushButton("Eigenes Tetraedernetz …")
        self.mesh_button.setToolTip("Gewebeversuch auf einem eigenen Tetraedernetz (NPZ, mm, höchstens 2000 Knoten) rechnen")
        self.mesh_button.clicked.connect(lambda: self.start_tissue(True))
        actions.addWidget(self.tissue_button)
        actions.addWidget(self.mesh_button)
        layout.addLayout(actions)
        rings = QHBoxLayout()
        self.ring_button = QPushButton("Kreisende Erregung berechnen")
        self.ring_button.setToolTip("Kreisende Erregung auf einem Ring über 1800 ms rechnen")
        self.ring_button.clicked.connect(lambda: self.start_ring(False))
        self.ablation_button = QPushButton("Ring mit Unterbrechung bei 900 ms")
        self.ablation_button.setToolTip("Denselben Ringversuch rechnen, mit Leitungsunterbrechung bei 900 ms")
        self.ablation_button.clicked.connect(lambda: self.start_ring(True))
        rings.addWidget(self.ring_button)
        rings.addWidget(self.ablation_button)
        layout.addLayout(rings)
        learning = QPushButton("Hintergrund-Lernlauf anzeigen …")
        learning.setToolTip("Status eines Lernlaufs (Ordner mit manifest.json) anzeigen")
        learning.clicked.connect(self.show_learning)
        layout.addWidget(learning)
        self.tissue_slider = QSlider(Qt.Orientation.Horizontal)
        self.tissue_slider.setRange(0, 0)
        self.tissue_slider.valueChanged.connect(self.draw_tissue)
        layout.addWidget(self.tissue_slider)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Zeit", units="ms")
        self.plot.setLabel("left", "Membranpotential", units="mV")
        layout.addWidget(self.plot, 2)
        self.report = QTextBrowser()
        self.report.setPlainText("Noch keine Zellrechnung. Fehlende Laborwerte werden nicht durch Normalwerte ersetzt; ohne Übernahme gelten veröffentlichte Modellparameter.")
        layout.addWidget(self.report, 1)
        self.model.currentIndexChanged.connect(self.reset_model_mechanisms)

    def reset_model_mechanisms(self):
        self.channel_scales = {}
        self.katp_open_fraction = 0.
        self.genetic_variant = "reference"
        self.regional_scenario = False
        self.report.setPlainText("Modell gewechselt: Kanal-/KATP-/Regionaländerungen zurückgesetzt. "
                                 "Vorhandene Kurve bleibt ein historisches Ergebnis, keine Rechnung des neu gewählten Modells.")
    def configure_mechanisms(self):
        from ..simulation.ionic import CHANNELS
        from ..simulation.variants import VARIANTS
        dialog = QDialog(self)
        dialog.setWindowTitle("Mechanismusversuche · keine Genotypdiagnosen")
        form = QFormLayout(dialog)
        note = QLabel("Multiplikatoren verändern den publizierten Stromparameter, nicht eine bestimmte Mutation. "
                      "Der regionale X-Gradient ist ein künstliches Substrat, kein bestimmtes Koronargefäß; ATP/pH werden nicht gelöst.")
        note.setWordWrap(True)
        form.addRow(note)
        controls = {}
        for name in CHANNELS[self.model.currentData()]:
            spin = QDoubleSpinBox()
            spin.setRange(0, 3)
            spin.setSingleStep(.1)
            spin.setValue(self.channel_scales.get(name, 1.))
            controls[name] = spin
            form.addRow(name+" ×", spin)
        variant = QComboBox()
        for key, label in VARIANTS.items():
            variant.addItem(label, key)
        variant.setCurrentIndex(variant.findData(self.genetic_variant))
        variant.setEnabled(self.model.currentData() == "fabbri-2017")
        form.addRow("Quellengebundene Variante (nur Sinusknoten)", variant)

        def variant_changed():
            for key in ("INa", "INaL"):
                if key in controls:
                    controls[key].setEnabled(variant.currentData() == "reference")
                    if variant.currentData() != "reference":
                        controls[key].setValue(1.)
        variant.currentIndexChanged.connect(variant_changed)
        variant_changed()
        katp = QDoubleSpinBox()
        katp.setDecimals(3)
        katp.setRange(0, 1.)
        katp.setSingleStep(.05)
        katp.setSuffix(" %")
        katp.setValue(100*self.katp_open_fraction)
        katp.setEnabled(self.model.currentData() == "tentusscher-2006")
        form.addRow("Offene KATP-Kanäle (TP06; nicht ATP-Konzentration)", katp)
        prepace = QSpinBox()
        prepace.setRange(0, 1000)
        prepace.setSingleStep(10)
        prepace.setValue(self.precondition_max_beats)
        form.addRow("Max. Vorkonditionierungsschläge (Einzelzelle; 0 aus)", prepace)
        regional = QCheckBox("K/INa/ICaL/Kopplungs-Gradient in X-Richtung (TP06)")
        regional.setChecked(self.regional_scenario)
        form.addRow(regional)
        potassium = QDoubleSpinBox()
        potassium.setRange(1.5, 9)
        potassium.setValue(self.regional_k)
        form.addRow("K am rechten X-Rand (mmol/l)", potassium)
        duration = QDoubleSpinBox()
        duration.setRange(5, 1000)
        duration.setValue(self.tissue_duration_ms)
        form.addRow("Gewebedauer (ms)", duration)
        s2 = QDoubleSpinBox()
        s2.setRange(0, 999)
        s2.setValue(self.s2_onset_ms)
        form.addRow("S2 gegenüberliegende Seite (ms; 0 aus)", s2)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.channel_scales = {key: spin.value() for key, spin in controls.items() if spin.value() != 1.}
            self.katp_open_fraction = katp.value()/100
            self.genetic_variant = variant.currentData()
            self.precondition_max_beats = prepace.value()
            self.regional_scenario, self.regional_k = regional.isChecked(), potassium.value()
            self.s2_onset_ms, self.tissue_duration_ms = s2.value(), duration.value()
            self.report.setPlainText("Nächster Versuch: Kanalparameter " + json.dumps(self.channel_scales)
                                     + "; regionale Feld-/S2-Einstellungen gespeichert. Keine bestehende Rechnung geändert.")

    def show_learning(self):
        directory = QFileDialog.getExistingDirectory(self, "Lernlauf mit manifest.json", "artifacts/patient_twin")
        if not directory:
            return
        root = Path(directory)
        dialog = QDialog(self)
        dialog.setWindowTitle("Lernlauf · Solver-Emulator, keine Patientendiagnose")
        dialog.resize(700, 600)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        layout.addWidget(browser)

        def update():
            try:
                status = json.loads((root/"status.json").read_text(encoding="utf-8"))
                text = "GUI-unabhängiger Forschungsjob\n" + json.dumps(status, ensure_ascii=False, indent=2)
                if (root/"emulator.json").exists():
                    result = json.loads((root/"emulator.json").read_text(encoding="utf-8"))
                    summary = {k: v for k, v in result.items() if k != "coefficients"}
                    text += "\n\nAbschlussbewertung\n" + json.dumps(summary, ensure_ascii=False, indent=2)
                browser.setPlainText(text)
            except (OSError, ValueError) as exc:
                browser.setPlainText(str(exc))
        timer = QTimer(dialog)
        timer.timeout.connect(update)
        timer.start(2000)
        update()
        dialog.exec()

    def start_ring(self, ablation):
        try:
            self.launch({"kind": "reentry_ring", "duration_ms": 1800., "ablate_at_ms": 900. if ablation else None})
        except (ValueError, OSError) as exc:
            self.report.setPlainText(str(exc))

    def apply_rate(self):
        try:
            from ..simulation.patient import pacing_suggestion
            suggestion = pacing_suggestion(self.patient_panel.profile())
            if suggestion["status"] != "protocol_suggestion":
                raise ValueError("Keine gemessene Herzfrequenz mit Quelle im Patientenprofil")
            cycle = suggestion["cycle_ms"]
            if not self.cycle.minimum() <= cycle <= self.cycle.maximum():
                raise ValueError("Gemessene Frequenz außerhalb des Modellprotokolls")
            self.cycle.setValue(cycle)
            self.report.setPlainText("Pacing-Vorschlag übernommen, nicht autonome Sinusknotenphysiologie oder AF: " + suggestion["note"])
        except (ValueError, TypeError) as exc:
            self.report.setPlainText(str(exc))

    def start_tissue(self, imported=False):
        try:
            from ..simulation.ionic import CellExperiment
            spec = CellExperiment(model=self.model.currentData(), cell_type=self.phenotype.currentIndex(),
                                  cycle_ms=self.cycle.value(), beats=1, channel_scales=self.channel_scales.copy(),
                                  katp_open_fraction=self.katp_open_fraction, genetic_variant=self.genetic_variant)
            request = {"kind": "ionic_tissue", "cell_experiment": asdict(spec), "duration_ms": self.tissue_duration_ms}
            from ..simulation.monodomain import block_mesh
            mesh = block_mesh()
            if self.use_electrolytes.isChecked():
                raise ValueError("Patienten-Elektrolyte zuerst im Einzelzellversuch prüfen; Gewebedemo verwendet publizierte Werte.")
            if imported:
                path, _ = QFileDialog.getOpenFileName(self, "Forschungs-Tetraedernetz (mm; maximal 2000 Knoten)", "", "NPZ (*.npz)")
                if not path:
                    return
                from ..simulation.monodomain import load_mesh
                mesh = load_mesh(path)
                request.update(mesh_path=path, mesh_file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
            if self.regional_scenario:
                if spec.model != "tentusscher-2006":
                    raise ValueError("Regionale Felder bisher nur für TP06 geprüft")
                x = mesh.xyz_mm[:, 0]
                span = float(np.ptp(x))
                if span <= 0:
                    raise ValueError("X-Gradient bei diesem Netz nicht definiert")
                gradient = (x-x.min())/span
                request["substrate"] = {
                    "potassium_mmol_l": (5.4+(self.regional_k-5.4)*gradient).tolist(),
                    "channel_scales": {"INa": (1-.25*gradient).tolist(), "ICaL": (1-.25*gradient).tolist()},
                    "element_coupling_scale": (1-.7*gradient[mesh.tetra].mean(axis=1)).tolist(),
                    "provenance": "GUI explicit synthetic X gradient; not a coronary or complete ischemia model"}
            if self.s2_onset_ms:
                from ..simulation.monodomain import StimulusEvent
                x = mesh.xyz_mm[:, 0]
                events = [StimulusEvent(0., spec.stimulus_ms, tuple(map(int, np.flatnonzero(x == x.min())))),
                          StimulusEvent(self.s2_onset_ms, spec.stimulus_ms, tuple(map(int, np.flatnonzero(x == x.max()))))]
                for event in events:
                    event.validate(len(x), self.tissue_duration_ms)
                request["events"] = [asdict(e) for e in events]
            self.launch(request)
        except (OSError, ValueError) as exc:
            self.report.setPlainText(str(exc))

    def show_tissue(self, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            ring = metadata.get("schema") == "reentry-ring/1"
            self.tissue_units = "normalized" if ring else "mV"
            self.tissue_data = {key: data[key].copy() for key in ("xyz_mm", "times_ms")}
            self.tissue_data["values"] = data["voltage" if ring else "voltage_mv"].copy()
        self.plot.clear()
        self.plot.setLabel("left", "Membranpotential" if not ring else "Reduzierter Erregungszustand", units="mV" if not ring else "")
        for node in (0, len(self.tissue_data["xyz_mm"])-1):
            self.plot.plot(self.tissue_data["times_ms"], self.tissue_data["values"][:, node],
                           pen=pg.mkPen("#33cbbb" if node == 0 else "#ecae55", width=2))
        self.report.setPlainText(("Reduziertes Ringexperiment · normierter Zustand, kein Ionenkanalmodell.\n" if ring else
                                 "3D-Ionen/FEM-Gewebe · Membranpotential an Knoten 0 (türkis) und letztem Knoten (orange).\n")
                                 + "Kein Patientenganzherz oder Oberflächen-EKG.\n\n" + json.dumps(metadata, ensure_ascii=False, indent=2))
        self.tissue_slider.setRange(0, len(self.tissue_data["times_ms"])-1)
        try:
            import pyqtgraph.opengl as gl
            if self.tissue_gl:
                self.tissue_gl.close()
            self.tissue_gl = gl.GLViewWidget()
            self.tissue_gl.setWindowTitle("Ionisches 3D-Gewebe · Forschung · mV")
            self.tissue_gl.resize(800, 600)
            xyz = self.tissue_data["xyz_mm"]
            self.tissue_scatter = gl.GLScatterPlotItem(pos=xyz, size=12, pxMode=True)
            self.tissue_gl.addItem(self.tissue_scatter)
            from PyQt6.QtGui import QVector3D
            self.tissue_gl.opts["center"] = QVector3D(*map(float, xyz.mean(axis=0)))
            self.tissue_gl.setCameraPosition(distance=max(12, float(np.ptp(xyz, axis=0).max())*3))
            self.tissue_gl.show()
            self.draw_tissue(self.tissue_slider.value())
        except (ImportError, RuntimeError) as exc:
            self.report.append("3D nicht verfügbar; Messkurven bleiben nutzbar: " + str(exc))

    def draw_tissue(self, frame):
        if self.tissue_data is None or self.tissue_gl is None:
            return
        # Fixed physical voltage scale, not per-frame auto contrast.
        voltage = self.tissue_data["values"][frame]
        scaled = voltage if self.tissue_units == "normalized" else (voltage+100)/150
        colors = pg.colormap.get("viridis").map(np.clip(scaled, 0, 1), mode="float")
        self.tissue_scatter.setData(color=colors)
        scale = "normierter Zustand 0–1" if self.tissue_units == "normalized" else "−100 bis +50 mV"
        self.tissue_gl.setWindowTitle(f"Berechnetes Gewebe · {self.tissue_data['times_ms'][frame]:.1f} ms · {scale}")

    def set_busy(self, busy):
        for button in (self.run_button, self.tissue_button, self.mesh_button, self.ring_button, self.ablation_button, self.metabolism_button):
            button.setEnabled(not busy)

    def start_metabolism(self):
        from .metabolism_panel import MetabolismDialog
        dialog = MetabolismDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.launch(dialog.request())
            except (ValueError, OSError) as exc:
                self.report.setPlainText(str(exc))

    def show_metabolism(self, path):
        from .metabolism_panel import MetabolismResult
        self.metabolism_result = MetabolismResult(path, self)
        self.metabolism_result.show()

    def start(self):
        try:
            from ..simulation.ionic import CellExperiment
            if 0 < self.precondition_max_beats < 10:
                raise ValueError("Vorkonditionierung: 0 (aus) oder mindestens 10 Schläge wählen")
            spec = CellExperiment(model=self.model.currentData(), cell_type=self.phenotype.currentIndex(),
                                  cycle_ms=self.cycle.value(), beats=self.beats.value(), channel_scales=self.channel_scales.copy(),
                                  katp_open_fraction=self.katp_open_fraction, genetic_variant=self.genetic_variant)
            spec.validate()
            self.launch({"kind": "ionic_cell", "cell_experiment": asdict(spec),
                         "precondition_max_beats": self.precondition_max_beats,
                         "patient_profile": self.patient_panel.profile().to_dict(),
                         "use_patient_electrolytes": self.use_electrolytes.isChecked()})
        except (ValueError, OSError) as exc:
            self.report.setPlainText(str(exc))

    def show_result(self, path):
        self.plot.setLabel("left", "Membranpotential", units="mV")
        if self.tissue_gl is not None:
            self.tissue_gl.close()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("schema") != "ionic-cell-assay/1":
                raise ValueError("Unbekanntes Zellmodellformat")
            self.plot.clear()
            self.plot.plot(data["times_ms"], data["voltage_mv"], pen=pg.mkPen("#33cbbb", width=2))
        self.report.setPlainText("Nur Membranpotential, kein Oberflächen-EKG.\n\n" + json.dumps(metadata, ensure_ascii=False, indent=2))
