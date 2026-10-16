"""Native-signal expert review surface, independent of production orchestration."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import copy
import html
import json
import uuid
import logging
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QCoreApplication, QSettings, QEvent, QUrl
from PyQt6.QtGui import QKeySequence, QShortcut, QFont, QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QApplication, QSizePolicy, QFrame,
    QCheckBox, QTabWidget, QLineEdit, QTextEdit, QFormLayout, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QInputDialog, QCompleter, QHeaderView,
    QToolButton, QMenu, QScrollArea,
    QDialog,
)

from ...mimic.lab_workspace import (
    SignalReference, MeasurementAnnotation, ReportRevision, LabReviewStore, REPORT_SECTIONS,
)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LANDMARKS = ("p_onset", "p_peak", "p_offset", "qrs_onset", "r_peak", "qrs_offset", "t_onset", "t_peak", "t_offset")


class _OverlayCard(QFrame):
    """Schwebende Karte über dem Signal: am Kopf verschiebbar, einklappbar, im Farbschema des Programms."""
    moved = pyqtSignal()
    toggled = pyqtSignal(bool)  # True = aufgeklappt

    def __init__(self, title, parent=None, tooltip=""):
        super().__init__(parent)
        self.setObjectName("overlayCard")
        self._drag = None
        self.collapsed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setSpacing(4)
        self.title = QLabel(title)
        self.title.setObjectName("overlayCardTitle")
        header.addWidget(self.title, 1)
        self.collapse_button = QToolButton()
        self.collapse_button.setObjectName("overlayCardButton")
        self.collapse_button.setText("–")
        self.collapse_button.setFixedSize(18, 18)
        self.collapse_button.setToolTip(tooltip or "Karte ein- und ausklappen; am Kopf verschiebbar")
        self.collapse_button.clicked.connect(self.toggle)
        header.addWidget(self.collapse_button)
        layout.addLayout(header)
        self.content = QWidget(self)
        self.content.setObjectName("overlayCardContent")
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(2)
        layout.addWidget(self.content)

    def toggle(self):
        self.set_collapsed(not self.collapsed)
        self.toggled.emit(not self.collapsed)

    def set_collapsed(self, collapsed):
        self.collapsed = bool(collapsed)
        self.content.setVisible(not self.collapsed)
        self.collapse_button.setText("+" if self.collapsed else "–")
        self.adjustSize()

    def apply_theme(self, dark):
        """Farben aus ``gui.theme``: Fläche, Rand und Textton wie die übrigen Karten des Programms."""
        from ..theme import RADIUS_CARD, tokens_for
        t = tokens_for(bool(dark))
        surface = QColor(t.surface)
        surface.setAlpha(255)  # deckend: die Vorschauen zeichnen auf derselben Fläche, sonst bleibt eine Kante sichtbar
        self.setStyleSheet(
            f"#overlayCard {{ background: rgba({surface.red()}, {surface.green()}, {surface.blue()}, {surface.alpha()});"
            f" border: 1px solid {t.border}; border-radius: {RADIUS_CARD}px; }}"
            f" #overlayCardTitle {{ font-size: 9px; font-weight: 600; color: {t.text_muted}; background: transparent; }}"
            f" #overlayCardButton {{ border: none; color: {t.text_muted}; font-weight: 700; background: transparent; }}"
            f" #overlayCardButton:hover {{ color: {t.accent}; }}"
            " #overlayCardContent, #overlayCardBody { background: transparent; }")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 24 and self.parentWidget().objectName() != 'labViewerStrip':
            self._drag = event.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            target = event.globalPosition().toPoint() - self._drag
            parent = self.parentWidget()
            if parent is not None:
                target.setX(min(max(0, target.x()), max(0, parent.width() - self.width())))
                target.setY(min(max(0, target.y()), max(0, parent.height() - self.height())))
            self.move(target)
            self.moved.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)


class ExpertECGLab(QWidget):
    researchModeChanged = pyqtSignal(bool)
    timeSelected = pyqtSignal(float)
    applyFiducials = pyqtSignal(dict, float, str)
    resultsRequested = pyqtSignal()
    syntheticDemoReady = pyqtSignal(int, object, object)
    publicDemoReady = pyqtSignal(int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.library = None
        self.case = None
        self.signal = None
        self.reference = None
        self.payload = {}
        self.marker_favorites = set()
        self.marker_groups = {}
        self._marker_groups_read_error = False
        self._marker_favorites_read_error = False
        self.fiducials = {}
        self.manual_fiducials = False
        self.history = []
        self.redo_history = []
        self.calipers = []
        self.saved_measurements = []
        self.plots = []
        self.prior = None
        self.prior_fs = None
        self.prior_payload = {}
        self.parent_revision = ""
        self.report_analysis_id = ""
        self.report_measurements = []
        self._dirty = False
        self._loading = False
        self.recovery_timer = QTimer(self)
        self.recovery_timer.setSingleShot(True)
        self.recovery_timer.setInterval(750)
        self.recovery_timer.timeout.connect(self._save_recovery)
        if QCoreApplication.instance():
            QCoreApplication.instance().aboutToQuit.connect(self._save_recovery)
        self._dark = False
        self.manual_overrides = {}
        self.suggestion_review = {}
        self._landmark_items = []
        self.extra_research_tabs = []
        self.analysis_arrays = None
        self._caliper_ids = []
        self._synthetic_demo_run = 0
        self.syntheticDemoReady.connect(self._on_synthetic_demo_ready)
        self._public_demo_run = 0
        self.publicDemoReady.connect(self._on_public_demo_ready)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        self.lab_hero = QFrame(self)
        self.lab_hero.setObjectName("ecgLabHero")
        hero_layout = QVBoxLayout(self.lab_hero)
        hero_layout.setContentsMargins(16, 12, 16, 12)
        hero_layout.setSpacing(5)
        modebar = QHBoxLayout()
        self.lab_kicker = QLabel("ECG-LABOR  /  SIGNAL ZUERST")
        self.lab_kicker.setObjectName("ecgLabKicker")
        modebar.addWidget(self.lab_kicker)
        modebar.addStretch(1)
        self.mode = QComboBox()
        self.mode.addItems(["Befundung", "Forschung"])
        self.mode.setToolTip("Befundung und Forschung nutzen dasselbe Originalsignal; Forschungswerte sind getrennt gekennzeichnet.")
        self.blind = QCheckBox("Verblindeter Review")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.blind.toggled.connect(self._mode_changed)
        modebar.addWidget(self.mode)
        modebar.addWidget(self.blind)
        hero_layout.addLayout(modebar)
        self.lab_title = QLabel("Vom Signal zum nachvollziehbaren Befund")
        self.lab_title.setObjectName("ecgLabTitle")
        hero_layout.addWidget(self.lab_title)
        self.notice = QLabel("Originalsignal · keine automatische klinische Freigabe")
        self.notice.setWordWrap(True)
        self.notice.setObjectName("ecgLabCaseStatus")
        hero_layout.addWidget(self.notice)
        root.addWidget(self.lab_hero)
        self.setup_toggle = QToolButton(self)
        self.setup_toggle.setText("Analyse-Setup anzeigen  ▾")
        self.setup_toggle.setObjectName("ecgLabSetupToggle")
        self.setup_toggle.setCheckable(True)
        self.setup_toggle.setToolTip("Netzfilter und bestätigten Morphologiekontext für die nächste lokale Analyse einstellen.")
        root.addWidget(self.setup_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setup_panel = QFrame(self)
        self.setup_panel.setObjectName("ecgLabSetupPanel")
        setup_layout = QHBoxLayout(self.setup_panel)
        setup_layout.setContentsMargins(12, 8, 12, 8)
        self.powerline = QComboBox()
        for label, value in (("Netzfilter: Metadaten", "metadata"), ("Netzfilter: 50 Hz", "50"), ("Netzfilter: 60 Hz", "60"), ("Netzfilter: aus", "off")):
            self.powerline.addItem(label, value)
        self.powerline.setToolTip("Gilt für die nächste lokale Analyse. Unbekannte Metadaten: kein Notch. Die native Anzeige bleibt unverändert.")
        setup_layout.addWidget(self.powerline)
        self.lbbb_context = QCheckBox('LSB für nächste Analyse bestätigen')
        self.lbbb_context.setToolTip('Erfordert Bearbeiter und Begründung. Keine automatische LSB-Erkennung.')
        setup_layout.addWidget(self.lbbb_context)
        self.ischemia_context_kind = QComboBox()
        self.ischemia_context_kind.addItem('LSB', 'lbbb')
        self.ischemia_context_kind.addItem('Ventrikulär stimulierte QRS', 'ventricular_pacing')
        self.lbbb_context.setText('Morphologiekontext bestätigen')
        setup_layout.addWidget(self.ischemia_context_kind)
        setup_layout.addStretch(1)
        self.setup_panel.hide()
        self.setup_toggle.toggled.connect(self.setup_panel.setVisible)
        self.setup_toggle.toggled.connect(
            lambda shown: self.setup_toggle.setText(
                "Analyse-Setup ausblenden  ▴" if shown else "Analyse-Setup anzeigen  ▾"
            )
        )
        root.addWidget(self.setup_panel)
        self.quick_suggestion_panel = QFrame(self)
        self.quick_suggestion_panel.setObjectName('labQuickSuggestions')
        quick_layout = QVBoxLayout(self.quick_suggestion_panel)
        self.quick_suggestion_status = QLabel('Automatische Prüfung: noch kein lokaler Analysestand geladen.')
        self.quick_suggestion_status.setWordWrap(True)
        self.quick_suggestion_status.setTextFormat(Qt.TextFormat.PlainText)
        quick_layout.addWidget(self.quick_suggestion_status)
        self.quick_suggestion_buttons = QHBoxLayout()
        quick_layout.addLayout(self.quick_suggestion_buttons)
        root.addWidget(self.quick_suggestion_panel)
        self.control_deck = QFrame(self)
        self.control_deck.setObjectName("ecgLabControlDeck")
        deck_layout = QVBoxLayout(self.control_deck)
        deck_layout.setContentsMargins(13, 9, 13, 9)
        deck_layout.setSpacing(7)
        deck_header = QHBoxLayout()
        deck_title = QLabel("ANSICHT & MASSSTAB")
        deck_title.setObjectName("ecgLabSectionLabel")
        deck_header.addWidget(deck_title)
        deck_header.addStretch(1)
        deck_note = QLabel("Anzeigeoptionen · Originalsignal unverändert")
        deck_note.setObjectName("ecgLabMicrocopy")
        deck_header.addWidget(deck_note)
        deck_layout.addLayout(deck_header)
        controls = QHBoxLayout()
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(["3×4 + Rhythmus", "6×2", "12×1", "Einzelableitung"])
        self.lead_choice = QComboBox()
        self.lead_choice.addItems(LEADS)
        self.lead_choice.setCurrentText("II")
        self.speed = QComboBox()
        self.speed.addItems(["12.5", "25", "50"])
        self.speed.setCurrentText("25")
        self.gain = QComboBox()
        self.gain.addItems(["5", "10", "20"])
        self.gain.setCurrentText("10")
        self.fit = QCheckBox("An Fenster anpassen")
        self.fit.setChecked(True)
        self.view_duration = QComboBox()
        self.view_duration.addItem('10 s', 10.0)
        self.view_duration.addItem('5 s', 5.0)
        self.view_duration.addItem('2,5 s', 2.5)
        self.view_duration.addItem('1 s', 1.0)
        self.view_duration.setToolTip('Zeitfenster der EKG-Anzeige. Originalsignal und Messzeiten bleiben unverändert.')
        self.show_landmarks = QCheckBox("Landmarken")
        self.show_landmarks.setToolTip("Grenzen anzeigen; Beschriftung nur für die ausgewählte Grenze.")
        self.show_landmarks.toggled.connect(self.redraw)
        self.pixels_per_mm = QDoubleSpinBox()
        self.pixels_per_mm.setRange(1, 20)
        self.pixels_per_mm.setDecimals(3)
        self.pixels_per_mm.setValue(96 / 25.4)
        self.pixels_per_mm.setToolTip("Logische Pixel/mm: mit physischem Lineal kalibrieren. Ohne Kalibrierung ist der Bildschirmmaßstab nur nominell.")
        for label, control in (("Layout", self.layout_choice), ("Ableitung", self.lead_choice), ("mm/s", self.speed), ("mm/mV", self.gain), ("px/mm", self.pixels_per_mm)):
            controls.addWidget(QLabel(label))
            controls.addWidget(control)
        deck_layout.addLayout(controls)
        calibration = QPushButton('Bildschirm kalibrieren')
        calibration.setToolTip('Maßstab mit einem physischen Lineal kalibrieren; ohne Kalibrierung nur nominelle Bildschirmmaße.')
        calibration.clicked.connect(self.calibrate_screen)
        lead_display = QHBoxLayout()
        self.lead_order = QComboBox()
        self.lead_order.addItems(['Standard', 'Cabrera'])
        self.lead_order.setToolTip('Cabrera: aVL, I, −aVR, II, aVF, III; V1–V6 unverändert. Nur Anzeige, keine neue Ableitung oder Transformation.')
        self.avr_polarity = QComboBox()
        self.avr_polarity.addItems(['aVR', '−aVR'])
        self.avr_polarity.setToolTip('Nur Anzeigeumkehr. Signal, VCG und Messwerte bleiben im nativen aVR-Koordinatensystem. Cabrera verwendet immer −aVR.')
        self._standard_avr_polarity = 0
        lead_display.addWidget(QLabel('Ableitungsfolge'))
        lead_display.addWidget(self.lead_order)
        lead_display.addWidget(QLabel('aVR-Anzeige'))
        lead_display.addWidget(self.avr_polarity)
        self.save_display_defaults = QPushButton('Als Ansichtsstandard speichern')
        self.save_display_defaults.setToolTip('Speichert nur Layout, Ableitungsfolge, aVR-Anzeige, Rhythmusableitung, Geschwindigkeit, Verstärkung und Fensteranpassung lokal für diesen Benutzer. Keine Messungen, Filter, Patienten, Befunde oder Bildschirmkalibrierung.')
        self.save_display_defaults.clicked.connect(self._save_display_defaults)
        lead_display.addWidget(self.save_display_defaults)
        lead_display.addWidget(calibration)
        lead_display.addStretch()
        deck_layout.addLayout(lead_display)
        navigation = QHBoxLayout()
        navigation.addWidget(self.fit)
        navigation.addWidget(QLabel('Zeitfenster'))
        navigation.addWidget(self.view_duration)
        navigation.addWidget(self.show_landmarks)
        self.start = QDoubleSpinBox()
        self.start.setDecimals(3)
        self.start.setSuffix(" s")
        navigation.addWidget(QLabel("Beginn"))
        navigation.addWidget(self.start)
        for title, fn in (("← Schlag", lambda: self.navigate(-1)), ("Schlag →", lambda: self.navigate(1))):
            button = QPushButton(title)
            button.clicked.connect(fn)
            navigation.addWidget(button)
        navigation.addStretch(1)
        tools = QHBoxLayout()
        for title, fn in (("Messzirkel +", self.add_caliper), ("RR-Zirkel +", lambda: self.add_caliper(True)), ("Zirkel löschen", self.clear_calipers), ("PNG/SVG", self.export_plot)):
            button = QPushButton(title)
            button.clicked.connect(fn)
            tools.addWidget(button)
        self.presentation_button = QToolButton()
        self.presentation_button.setText("Signalreise · Vorführen ▾")
        self.presentation_button.setToolTip("Geladenen Fall oder isolierte Demofälle zeigen; keine neue Produktionsanalyse")
        presentation_menu = QMenu(self.presentation_button)
        presentation_menu.addAction("Geladenen Fall zeigen", self.show_signal_presentation)
        presentation_menu.addAction("Synthetische Demo · kein Patient", self.show_synthetic_presentation)
        presentation_menu.addAction("Öffentlicher PTB-XL-Demofall · CC BY 4.0", self.show_public_presentation)
        self.presentation_button.setMenu(presentation_menu)
        self.presentation_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tools.addWidget(self.presentation_button)
        self.spatial_preview = QCheckBox("3D/VCG-Vorschau")
        self.spatial_preview.setChecked(True)
        self.spatial_preview.setToolTip("Räumliche Vorschaukarten unter dem EKG ein-/ausblenden; Messungen und Signalreise bleiben unverändert.")
        self._spatial_preview_user_set = False
        self.spatial_preview.clicked.connect(self._spatial_preview_clicked)
        tools.addWidget(self.spatial_preview)
        tools.addStretch(1)
        deck_layout.addLayout(navigation)
        deck_layout.addLayout(tools)
        root.addWidget(self.control_deck)
        self.signal_stage = QFrame(self)
        self.signal_stage.setObjectName("ecgLabSignalStage")
        stage_layout = QVBoxLayout(self.signal_stage)
        stage_layout.setContentsMargins(12, 10, 12, 10)
        stage_layout.setSpacing(7)
        stage_header = QHBoxLayout()
        stage_title = QLabel("SIGNALBÜHNE")
        stage_title.setObjectName("ecgLabSectionLabel")
        stage_header.addWidget(stage_title)
        stage_header.addStretch(1)
        stage_note = QLabel("Zeit und Spannung bleiben am Originalsignal verankert")
        stage_note.setObjectName("ecgLabMicrocopy")
        stage_header.addWidget(stage_note)
        stage_layout.addLayout(stage_header)
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.scene().sigMouseClicked.connect(self._scene_clicked)
        self.canvas.setFixedHeight(360)
        self.canvas_empty_label = self.canvas.addLabel(
            "NOCH KEIN EKG GELADEN  ·  Fall auswählen oder Signal importieren",
            row=0, col=0, color="#8fb9bd",
        )
        stage_layout.addWidget(self.canvas)
        self.viewer_strip = QFrame(self)
        self.viewer_strip.setObjectName('labViewerStrip')
        self.viewer_strip_layout = QVBoxLayout(self.viewer_strip)
        self.viewer_strip_layout.setContentsMargins(8, 6, 8, 6)
        viewer_header = QHBoxLayout()
        viewer_title = QLabel('RÄUMLICHE VORSCHAU  /  ILLUSTRATION & VCG')
        viewer_title.setObjectName("ecgLabSectionLabel")
        viewer_header.addWidget(viewer_title)
        viewer_header.addStretch(1)
        self.viewer_signal_note = QLabel('VCG-Vorschau: Signalquelle noch nicht geladen')
        self.viewer_signal_note.setWordWrap(True)
        self.viewer_signal_note.setMaximumWidth(260)
        viewer_header.addWidget(self.viewer_signal_note)
        self.viewer_strip_layout.addLayout(viewer_header)
        self.viewer_cards_layout = QHBoxLayout()
        self.viewer_cards_layout.addStretch(1)
        self.viewer_strip_layout.addLayout(self.viewer_cards_layout)
        stage_layout.addWidget(self.viewer_strip)
        self.viewer_strip.hide()
        root.addWidget(self.signal_stage)
        self._install_heart_overlay()
        self.measurement_dock = QFrame(self)
        self.measurement_dock.setObjectName("ecgLabMeasurementDock")
        measurement_layout = QVBoxLayout(self.measurement_dock)
        measurement_layout.setContentsMargins(13, 9, 13, 9)
        measurement_header = QHBoxLayout()
        measurement_title = QLabel("MESSEN & PRÜFEN")
        measurement_title.setObjectName("ecgLabSectionLabel")
        measurement_header.addWidget(measurement_title)
        measurement_header.addStretch(1)
        measurement_layout.addLayout(measurement_header)
        self.measurement_status = QLabel("Messzirkel: Zeit in ms, Spannung in mV; native Sampleauflösung.")
        self.measurement_status.setWordWrap(True)
        measurement_layout.addWidget(self.measurement_status)
        self.landmark_editor = QDialog(self)
        self.landmark_editor.setWindowTitle('Wellengrenzen im Detail')
        editor_layout = QVBoxLayout(self.landmark_editor)
        self.landmark_closeup = pg.PlotWidget()
        self.landmark_closeup.setMinimumSize(760, 330)
        editor_layout.addWidget(self.landmark_closeup)
        edits = QHBoxLayout()
        self.landmark_choice = QComboBox()
        self.landmark_choice.addItems(LANDMARKS)
        self.landmark_choice.currentIndexChanged.connect(self.redraw)
        edits.addWidget(self.landmark_choice)
        self.landmark_time = QDoubleSpinBox()
        self.landmark_time.setDecimals(6)
        self.landmark_time.setSuffix(" s")
        edits.addWidget(self.landmark_time)
        for title, fn in (("Landmarke setzen", self.set_landmark), ("Entfernen", self.remove_landmark), ("Rückgängig", self.undo), ("Wiederholen", self.redo), ("Messungen speichern", self.save_annotations), ("Grenzen anwenden", self.apply_landmarks)):
            button = QPushButton(title)
            button.clicked.connect(fn)
            edits.addWidget(button)
        editor_layout.addLayout(edits)
        open_editor = QPushButton('Wellengrenzen vergrößern / bearbeiten')
        open_editor.setObjectName("ecgLabPrimaryAction")
        open_editor.clicked.connect(self.show_landmark_editor)
        measurement_layout.addWidget(open_editor, alignment=Qt.AlignmentFlag.AlignLeft)
        self.landmark_editor.resize(900, 520)
        self.caliper_march = QCheckBox('March-out: Zeitzirkel mit fester Breite verschieben')
        self.caliper_march.setToolTip('Eine Zeitlinie ziehen verschiebt beide Grenzen um denselben Betrag. '
            'Zum Einstellen der Breite ausschalten. Am Signalrand bleibt die Breite erhalten; '
            'RR-Hilfslinien sind keine erkannten Schläge.')
        measurement_layout.addWidget(self.caliper_march)
        self.qt_alert_note = QLabel()
        self.qt_alert_note.setWordWrap(True)
        self.qt_alert_note.setTextFormat(Qt.TextFormat.PlainText)
        measurement_layout.addWidget(self.qt_alert_note)
        self.qt_ack_button = QPushButton('QT-Hinweis geprüft – lokal quittieren')
        self.qt_ack_button.clicked.connect(self.acknowledge_qt_alert)
        measurement_layout.addWidget(self.qt_ack_button)
        self.qt_history_button = QPushButton('Quittierungshistorie dieses QT-Messstands')
        self.qt_history_button.clicked.connect(self.show_qt_review_history)
        measurement_layout.addWidget(self.qt_history_button)
        self.qt_history_button.hide()
        self.qt_alert_note.hide()
        self.qt_ack_button.hide()
        root.addWidget(self.measurement_dock)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ecgLabWorkbenchTabs")
        root.addWidget(self.tabs)
        report = QWidget()
        report_layout = QVBoxLayout(report)
        form = QFormLayout()
        report_layout.addLayout(form)
        report_layout.addStretch(1)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        def compact_row(layout):
            container = QWidget(report)
            container.setLayout(layout)
            layout.setContentsMargins(0, 0, 0, 0)
            container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            form.addRow(container)
        self.author = QLineEdit("local_user")
        self.author.textChanged.connect(self._mark_dirty)
        form.addRow("Bearbeiter", self.author)
        self.section_choice = QComboBox()
        self.section_choice.addItems(REPORT_SECTIONS)
        self.phrase = QComboBox()
        self.phrase.setEditable(True)
        self.phrase.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.phrase.setMinimumContentsLength(24)
        self.phrase.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.phrase.addItems(["", "Signalqualität eingeschränkt; Befund nur eingeschränkt beurteilbar.", "Regelmäßige Herzaktion.", "Unregelmäßige Herzaktion; Rhythmus manuell prüfen.", "Vor-EKG für Vergleich nicht vorhanden.", "Repolarisationsbeurteilung bei breitem QRS eingeschränkt."])
        self.phrase.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.phrase.lineEdit().setPlaceholderText('Baustein suchen oder eigenen Text eingeben')
        self.phrase.setToolTip('Manuelle Textauswahl, kein automatisch erkannter Befund. Teilwörter suchen; Übernahme erst per Button oder Strg+Enter in den ausgewählten Abschnitt.')
        from ..lab_phrase_library import MEASUREMENT_VARIABLES
        self.phrase.setToolTip(self.phrase.toolTip() + '\nVariablen in ms: ' +
            ', '.join('{' + key + '}' for key in MEASUREMENT_VARIABLES) +
            '\nBeispiel: QTc {qtc_fridericia} ms nach Fridericia. Vorschau vor Übernahme.')
        completer = QCompleter(self.phrase.model(), self.phrase)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.phrase.setCompleter(completer)
        phrase_shortcut = QShortcut(QKeySequence('Ctrl+Return'), self.phrase.lineEdit())
        phrase_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        phrase_shortcut.activated.connect(self.insert_phrase)
        phrasebar = QHBoxLayout()
        phrasebar.addWidget(self.section_choice)
        phrasebar.addWidget(self.phrase, 1)
        button = QPushButton("Baustein übernehmen")
        button.clicked.connect(self.insert_phrase)
        phrasebar.addWidget(button)
        compact_row(phrasebar)
        variables = QHBoxLayout()
        variables.addWidget(QLabel('Messwertvariable'))
        self.phrase_variable = QComboBox()
        variables.addWidget(self.phrase_variable, 1)
        self.phrase_variable_insert = QPushButton('Variable in Vorlage einsetzen')
        self.phrase_variable_insert.clicked.connect(self.insert_measurement_variable)
        variables.addWidget(self.phrase_variable_insert)
        compact_row(variables)
        self.blind.toggled.connect(lambda *_: self.refresh_measurement_variables(self.payload))
        self.refresh_measurement_variables({})
        user_phrases = QHBoxLayout()
        self.phrase_save = QPushButton('Eigenen Baustein speichern')
        self.phrase_save.clicked.connect(self.save_user_phrase)
        user_phrases.addWidget(self.phrase_save)
        self.phrase_manage = QPushButton('Eigene Bausteine verwalten')
        self.phrase_manage.clicked.connect(self.manage_user_phrase)
        user_phrases.addWidget(self.phrase_manage)
        self.phrase_library_note = QLabel('Eigene Bausteine: lokale Benutzervorlagen, keine automatische Befundung.')
        self.phrase_library_note.setWordWrap(True)
        user_phrases.addWidget(self.phrase_library_note, 1)
        compact_row(user_phrases)
        self.section_choice.currentTextChanged.connect(self.refresh_user_phrases)
        self.refresh_user_phrases()
        self.sections = {}
        for name in REPORT_SECTIONS:
            edit = QTextEdit()
            edit.setMaximumHeight(70)
            edit.textChanged.connect(self._mark_dirty)
            self.sections[name] = edit
            form.addRow(name, edit)
        self.report_status = QLabel("Noch kein gespeicherter Befund")
        self.report_status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        form.addRow(self.report_status)
        buttons = QHBoxLayout()
        for title, status in (("Entwurf speichern", "draft"), ("Als geprüft speichern", "reviewed"), ("Lokal abschließen", "completed")):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, state=status: self.save_report(state))
            buttons.addWidget(button)
        button = QPushButton("Befund-PDF")
        button.clicked.connect(self.export_pdf)
        buttons.addWidget(button)
        button = QPushButton("Messungen CSV/JSON")
        button.clicked.connect(self.export_measurements)
        buttons.addWidget(button)
        compact_row(buttons)
        self.pdf_research = QCheckBox("Ausgewählte Marker als getrennten Forschungsanhang exportieren")
        self.pdf_research.setToolTip("Nur ausdrücklich in der Markertabelle ausgewählte Zeilen; keine Übernahme als klinischer Befund.")
        form.addRow(self.pdf_research)
        self.tabs.addTab(report, "Befund")
        self.measurements = QTableWidget(0, 3)
        self.measurements.setHorizontalHeaderLabels(["Messung", "Wert / Definition", "Herkunft"])
        self.measurements.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.measurements.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        matrix_page = QWidget()
        ml = QVBoxLayout(matrix_page)
        ml.addWidget(self.measurements)
        from ..lab_lead_matrix import HEADERS
        self.lead_measurement_table = QTableWidget(12, len(HEADERS))
        self.lead_measurement_table.setHorizontalHeaderLabels(HEADERS)
        self.lead_measurement_table.setVerticalHeaderLabels(LEADS)
        self.lead_measurement_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lead_measurement_table.setMinimumHeight(260)
        self.lead_measurement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.lead_measurement_table.cellClicked.connect(self._select_measured_lead)
        matrix_note = QLabel('Ableitungsmatrix · gespeicherte diagnostische Messungen, gemeinsame Delineation. '
                            'Zeile anklicken: Ableitung im EKG auswählen. aVR-Werte bleiben nativ, auch bei −aVR-Anzeige. '
                            'ST-J und J+20/60/80 ms sind getrennte Messzeitpunkte; spätere Zeitpunkte können in der T-Welle liegen. '
                            'Weitere Spalten sind horizontal erreichbar; fehlende Größen werden nicht geschätzt.')
        matrix_note.setWordWrap(True)
        ml.addWidget(matrix_note)
        ml.addWidget(self.lead_measurement_table)
        lead_export = QPushButton('Ableitungsmatrix CSV/JSON')
        lead_export.clicked.connect(self.export_lead_matrix)
        ml.addWidget(lead_export)
        self.st_j_review_note = QLabel()
        self.st_j_review_note.setWordWrap(True)
        ml.addWidget(self.st_j_review_note)
        self._populate_lead_matrix({})
        self.score_item_note = QLabel("Kein auswertbares QTc-Einzelitem; kein vollständiger LQTS-Score.")
        self.score_item_note.setWordWrap(True)
        ml.addWidget(self.score_item_note)
        self.device_interval_document = None
        device_import = QPushButton('Geräteintervalle zum Signal importieren (JSON)')
        device_import.clicked.connect(self.import_device_intervals)
        ml.addWidget(device_import)
        device_reopen = QPushButton('Gespeicherten Gerätevergleich öffnen')
        device_reopen.clicked.connect(self.reopen_device_intervals)
        ml.addWidget(device_reopen)
        device_export = QPushButton('Gerätevergleich CSV/JSON exportieren')
        device_export.clicked.connect(self.export_device_comparison)
        ml.addWidget(device_export)
        self.device_interval_table = QTableWidget(0, 5)
        self.device_interval_table.setHorizontalHeaderLabels(
            ['Intervall / Formel', 'Eigene Messung (ms)', 'Gerät (ms)', 'Eigene − Gerät (ms)', 'Status'])
        self.device_interval_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.device_interval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.device_interval_table.setMaximumHeight(230)
        self.device_interval_table.hide()
        ml.addWidget(self.device_interval_table)
        self.device_interval_note = QLabel('Kein fallgebundener Gerätevergleich geladen.')
        self.device_interval_note.setWordWrap(True)
        self.device_interval_note.setTextFormat(Qt.TextFormat.PlainText)
        ml.addWidget(self.device_interval_note)
        interval_export_button = QPushButton('Automatische Intervalle CSV/JSON exportieren')
        interval_export_button.clicked.connect(self.export_automatic_intervals)
        ml.addWidget(interval_export_button)
        manual = QPushButton("Ausgewählte Messung manuell im Befund ergänzen")
        manual.clicked.connect(self.override_measurement)
        ml.addWidget(manual)
        self.override_note = QLabel("Automatische Werte bleiben unverändert; manuelle Befundwerte werden getrennt gespeichert.")
        self.override_note.setWordWrap(True)
        self.override_note.setTextFormat(Qt.TextFormat.PlainText)
        ml.addWidget(self.override_note)
        self.tabs.addTab(matrix_page, "Messmatrix")
        suggestions = QWidget()
        sl = QVBoxLayout(suggestions)
        sl.addWidget(QLabel("Regelvorschläge einzeln prüfen. Erfüllte EKG-Regel ist kein unabhängiger Krankheitsnachweis."))
        self.rhythm_review_note = QLabel('Kein gespeicherter Rhythmuskontext geladen.')
        self.rhythm_review_note.setWordWrap(True)
        self.rhythm_review_note.setTextFormat(Qt.TextFormat.PlainText)
        sl.addWidget(self.rhythm_review_note)
        self.rule_search = QLineEdit()
        self.rule_search.setPlaceholderText('Gespeicherte Regeln suchen: Name, Definition, Quelle …')
        self.rule_search.textChanged.connect(self.filter_rule_suggestions)
        sl.addWidget(self.rule_search)
        self.suggestions = QTableWidget(0, 4)
        self.suggestions.setHorizontalHeaderLabels(["Regel", "Automatischer Wert", "Beurteilbarkeit", "Review"])
        self.suggestions.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sl.addWidget(self.suggestions)
        self.suggestion_detail = QTextEdit()
        self.suggestion_detail.setReadOnly(True)
        self.suggestion_detail.setPlaceholderText('Regel auswählen: gespeicherte Definition und Herkunft prüfen.')
        self.suggestion_detail.setMaximumHeight(210)
        sl.addWidget(self.suggestion_detail)
        self.suggestions.itemSelectionChanged.connect(self.inspect_suggestion)
        visual_review = QPushButton('Ausgewählte Regel am EKG prüfen')
        visual_review.clicked.connect(self._open_selected_rule_evidence)
        sl.addWidget(visual_review)
        sb = QHBoxLayout()
        for label, decision in (("Übernehmen", "accepted"), ("Mit Änderung übernehmen", "edited"), ("Verwerfen", "rejected")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=decision: self.review_suggestion(d))
            sb.addWidget(button)
        sl.addLayout(sb)
        self.reconcile_review = QPushButton("Geänderte Übernahmen: Befundtext geprüft bestätigen")
        self.reconcile_review.clicked.connect(self.reconcile_suggestion_text)
        sl.addWidget(self.reconcile_review)
        self.suggestions_tab = self.tabs.addTab(suggestions, "Regelprüfung")
        from ..lab_criteria_catalog import CriteriaCatalogWidget
        self.criteria_catalog = CriteriaCatalogWidget()
        self.criteria_catalog.criterionRequested.connect(self.open_catalog_case_criterion)
        self.criteria_catalog_tab = self.tabs.addTab(self.criteria_catalog, 'Kriterien-Quellenkatalog')
        compare = QWidget()
        cv = QVBoxLayout(compare)
        self.comparison_id = QComboBox()
        cv.addWidget(QLabel("Vergleich nur mit belegter gemeinsamer Patientenkennung innerhalb derselben Quelle."))
        cv.addWidget(self.comparison_id)
        button = QPushButton("Vor-EKG überlagern / ausblenden")
        button.clicked.connect(self.compare_case)
        cv.addWidget(button)
        self.comparison_note = QLabel("Keine verifizierte Patientenzuordnung: kein automatischer Fallvergleich.")
        self.comparison_note.setWordWrap(True)
        cv.addWidget(self.comparison_note)
        identity_form = QFormLayout()
        self.patient_namespace = QLineEdit()
        self.patient_namespace.setPlaceholderText("z. B. eindeutiges lokales Quellsystem")
        self.patient_identifier = QLineEdit()
        self.patient_evidence = QLineEdit()
        identity_form.addRow("Quell-Namensraum", self.patient_namespace)
        identity_form.addRow("Verifizierte Patientenkennung", self.patient_identifier)
        identity_form.addRow("Zuordnungsnachweis", self.patient_evidence)
        identity_button = QPushButton("Patientenzuordnung ausdrücklich bestätigen")
        identity_button.clicked.connect(self.save_patient_identity)
        identity_form.addRow(identity_button)
        self.prior_offset = QDoubleSpinBox()
        self.prior_offset.setRange(-3600, 3600)
        self.prior_offset.setDecimals(3)
        self.prior_offset.setSuffix(" s")
        self.prior_offset.setToolTip("Manueller Zeitversatz des Vergleichs-EKGs. Keine automatische Beat-Ausrichtung oder Amplitudennormierung.")
        self.prior_offset.valueChanged.connect(self.redraw)
        identity_form.addRow("Zeitversatz Vergleich", self.prior_offset)
        cv.addLayout(identity_form)
        self.comparison_analysis = QComboBox()
        self.comparison_analysis.currentIndexChanged.connect(self._comparison_analysis_changed)
        cv.addWidget(QLabel("Analysestand des Vergleichsfalls (keine Auswahl nach Änderungsdatum):"))
        cv.addWidget(self.comparison_analysis)
        self.comparison_values = QTableWidget(0, 5)
        self.comparison_values.setHorizontalHeaderLabels(["Messung [ms]", "Aktuell", "Vergleich", "Δ aktuell − Vergleich", "Aussagegrenze"])
        self.comparison_values.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        cv.addWidget(self.comparison_values)
        from .paired_change_widget import PairedChangeWidget
        self.comparison_changes = PairedChangeWidget()
        cv.addWidget(self.comparison_changes)
        self.comparison_plot = pg.GraphicsLayoutWidget()
        self.comparison_plot.setMinimumHeight(220)
        cv.addWidget(self.comparison_plot)
        self.tabs.addTab(compare, "Vergleichs-EKG")
        research = QWidget()
        rv = QVBoxLayout(research)
        self.wpw_button = QPushButton('Akzessorische Bahn · EASY-WPW prüfen')
        self.wpw_button.setToolTip(
            'Quellengebundener Regions-Entscheidungsbaum nur nach bestätigter manifester Präexzitation; keine automatische WPW-Diagnose.'
        )
        self.wpw_button.clicked.connect(self.open_wpw_localization)
        rv.addWidget(self.wpw_button)
        self.pvc_origin_button = QPushButton('Ventrikulärer Ursprung · PVC-Ausflusstrakt prüfen')
        self.pvc_origin_button.setToolTip(
            'Publizierte V2- und V2S/V3R-Regeln nur für bestätigte idiopathische Ausflusstrakt-PVCs; Mapping bleibt Referenz.'
        )
        self.pvc_origin_button.clicked.connect(self.open_pvc_localization)
        rv.addWidget(self.pvc_origin_button)
        self.atrial_origin_button = QPushButton('Vorhof-Fokus · V1-P-Welle prüfen')
        self.atrial_origin_button.setToolTip(
            'Konservativer regionaler Hinweis nur bei bestätigter fokaler atrialer Tachykardie und klarer ektoper P-Welle.'
        )
        self.atrial_origin_button.clicked.connect(self.open_atrial_focus)
        rv.addWidget(self.atrial_origin_button)
        self.marker_search = QLineEdit()
        self.marker_search.setPlaceholderText("Marker / Familie suchen …")
        self.marker_search.textChanged.connect(self.filter_markers)
        rv.addWidget(self.marker_search)
        favorite_controls = QHBoxLayout()
        self.only_favorites = QCheckBox("Nur lokale Favoriten")
        self.only_favorites.toggled.connect(lambda _: self.filter_markers(self.marker_search.text()))
        favorite_controls.addWidget(self.only_favorites)
        favorite_button = QPushButton("Auswahl zu Favoriten / entfernen")
        favorite_button.clicked.connect(self.toggle_marker_favorites)
        favorite_controls.addWidget(favorite_button)
        rv.addLayout(favorite_controls)
        group_controls = QHBoxLayout()
        self.marker_group = QComboBox()
        self.marker_group.addItem("Alle Marker", None)
        self.marker_group.currentIndexChanged.connect(lambda _: self.filter_markers(self.marker_search.text()))
        group_controls.addWidget(self.marker_group)
        save_group = QPushButton("Auswahl als Vergleichsgruppe speichern")
        save_group.clicked.connect(self.save_marker_group)
        group_controls.addWidget(save_group)
        remove_group = QPushButton("Gruppe entfernen")
        remove_group.clicked.connect(self.remove_marker_group)
        group_controls.addWidget(remove_group)
        rv.addLayout(group_controls)
        self.markers = QTableWidget(0, 3)
        self.markers.setHorizontalHeaderLabels(["Marker", "Wert", "Einheit / Status"])
        self.markers.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.markers.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.markers.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.markers.itemSelectionChanged.connect(self.inspect_marker)
        rv.addWidget(self.markers)
        from .marker_evidence_trail import MarkerEvidenceTrail
        self.marker_evidence = MarkerEvidenceTrail(research, dark=self._dark)
        rv.addWidget(self.marker_evidence)
        self.marker_detail = QTextEdit()
        self.marker_detail.setReadOnly(True)
        rv.addWidget(self.marker_detail)
        from ..lab_provenance_tree import MarkerProvenanceTree
        self.marker_provenance_tree = MarkerProvenanceTree()
        rv.addWidget(self.marker_provenance_tree)
        self.marker_beat = QComboBox()
        self.marker_beat.setVisible(False)
        self.marker_beat.currentIndexChanged.connect(lambda _: self.inspect_marker(keep_beat=True))
        rv.addWidget(self.marker_beat)
        self.marker_beat_audit = QTableWidget(0, 3)
        self.marker_beat_audit.setHorizontalHeaderLabels(['Schlag', 'Verwendet', 'Erster Ausschlussgrund / Status'])
        self.marker_beat_audit.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.marker_beat_audit.setMaximumHeight(180)
        self.marker_beat_audit.setColumnWidth(2, 430)
        self.marker_beat_audit.setVisible(False)
        rv.addWidget(self.marker_beat_audit)
        self.marker_signal = pg.PlotWidget()
        self.marker_signal.setLabel("bottom", "Analysezeit", units="s")
        self.marker_signal.setLabel("left", "Ableitungen versetzt", units="mV")
        self.marker_signal.setMinimumHeight(260)
        rv.addWidget(self.marker_signal)
        research_scroll = QScrollArea()
        research_scroll.setWidgetResizable(True)
        research_scroll.setFrameShape(QFrame.Shape.NoFrame)
        research_scroll.setWidget(research)
        self.research_tab = self.tabs.addTab(research_scroll, "Markerinspektor")
        endpoint_page = QWidget()
        endpoint_layout = QVBoxLayout(endpoint_page)
        endpoint_note = QLabel('Endpunktübersicht gespeicherter Kriterien. Keine Mehrheitsdiagnose, keine kombinierte Wahrscheinlichkeit. Negative Kriterien schließen die Erkrankung nicht aus; fehlende Endpunkte wurden nicht beurteilt.')
        endpoint_note.setWordWrap(True)
        endpoint_layout.addWidget(endpoint_note)
        self.endpoint_review_status = QLabel('Keine endpunktbezogenen Forschungskriterien im geladenen Analysestand; keine negative Befundung.')
        self.endpoint_review_status.setWordWrap(True)
        endpoint_layout.addWidget(self.endpoint_review_status)
        from .endpoint_comparison_card import EndpointComparisonCard
        self.endpoint_card = EndpointComparisonCard(endpoint_page, dark=self._dark)
        self.endpoint_card.ruleRequested.connect(self._open_endpoint_rule)
        self.endpoint_card.markerRequested.connect(self._open_endpoint_marker)
        endpoint_layout.addWidget(self.endpoint_card)
        self.endpoint_review = QTableWidget(0, 4)
        self.endpoint_review.setHorizontalHeaderLabels(['Endpunkt', 'Klassische Kriterien', 'Forschungskriterien', 'Gegenüberstellung'])
        self.endpoint_review.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.endpoint_review.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.endpoint_review.itemSelectionChanged.connect(self._endpoint_selection_changed)
        endpoint_layout.addWidget(self.endpoint_review)
        self.endpoint_review_tab = self.tabs.addTab(endpoint_page, 'Endpunktübersicht')
        self.extra_research_tabs.append(self.endpoint_review_tab)
        from ..lab_laddergram import LaddergramWidget
        self.laddergram = LaddergramWidget()
        self.extra_research_tabs.append(self.tabs.addTab(self.laddergram, 'Laddergram'))
        self.laddergram.timeSelected.connect(self.timeSelected.emit)
        self.timeSelected.connect(self.laddergram.time.setValue)
        representative_page = QWidget()
        representative_layout = QVBoxLayout(representative_page)
        representative_controls = QHBoxLayout()
        self.representative_path = QComboBox()
        self.representative_path.addItem('Gespeicherter diagnostischer Signalpfad', 'diagnostic_ecg')
        self.representative_path.addItem('Gespeicherter Morphologiepfad', 'morphology_ecg')
        self.representative_lead = QComboBox()
        self.representative_lead.addItems(['Alle 12 Ableitungen überlagert', *LEADS])
        representative_controls.addWidget(self.representative_path)
        representative_controls.addWidget(self.representative_lead)
        representative_layout.addLayout(representative_controls)
        template_controls = QHBoxLayout()
        self.representative_kind = QComboBox()
        self.representative_kind.addItems(['Gespeicherter repräsentativer Schlag', 'Exploratives Median-Template'])
        self.template_cluster = QComboBox()
        self.template_cluster.addItem('Cluster ausdrücklich wählen', None)
        self.template_build = QPushButton('Median lokal berechnen')
        self.template_build.clicked.connect(self.build_median_template)
        template_controls.addWidget(self.representative_kind)
        template_controls.addWidget(self.template_cluster)
        template_controls.addWidget(self.template_build)
        self.template_export = QPushButton('Median + Provenienz exportieren')
        self.template_export.clicked.connect(self.export_median_template)
        template_controls.addWidget(self.template_export)
        representative_layout.addLayout(template_controls)
        self.median_template = None
        self.template_annotations = []
        self.template_region = None
        self.template_region_bounds = None
        measurement_controls = QHBoxLayout()
        self.template_interval_kind = QComboBox()
        self.template_interval_kind.addItems(['freies Intervall', 'P', 'PR', 'QRS', 'QT', 'JT'])
        measure_template = QPushButton('Manuellen Messbereich setzen')
        measure_template.clicked.connect(self.add_template_measurement)
        save_template_measurement = QPushButton('Template-Messung lokal speichern')
        save_template_measurement.clicked.connect(self.save_template_measurement)
        measurement_controls.addWidget(self.template_interval_kind)
        measurement_controls.addWidget(measure_template)
        measurement_controls.addWidget(save_template_measurement)
        withdraw_template_measurement = QPushButton('Ausgewählte Messung zurücknehmen')
        withdraw_template_measurement.clicked.connect(self.withdraw_template_measurement)
        measurement_controls.addWidget(withdraw_template_measurement)
        representative_layout.addLayout(measurement_controls)
        self.template_measurement_note = QLabel('Manuelle Template-Messungen sind getrennt von Originalsignal-Messungen; Export über Median + Provenienz.')
        self.template_measurement_note.setWordWrap(True)
        representative_layout.addWidget(self.template_measurement_note)
        self.template_measurement_table = QTableWidget(0, 5)
        self.template_measurement_table.setHorizontalHeaderLabels(['Messung', 'Dauer [ms]', 'Ableitungsbezug', 'Bearbeiter', 'Gespeichert'])
        self.template_measurement_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.template_measurement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.template_measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.template_measurement_table.setMaximumHeight(160)
        self.template_measurement_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.template_measurement_table.cellDoubleClicked.connect(self.show_template_measurement)
        representative_layout.addWidget(self.template_measurement_table)
        self.representative_kind.currentIndexChanged.connect(self.draw_representative_beat)
        self.template_cluster.currentIndexChanged.connect(self.invalidate_median_template)
        self.representative_note = QLabel('Keine gespeicherte Schlagansicht geladen.')
        self.representative_note.setWordWrap(True)
        representative_layout.addWidget(self.representative_note)
        self.representative_plot = pg.PlotWidget()
        self.representative_legend = self.representative_plot.addLegend()
        self.representative_plot.setLabel('bottom', 'Zeit relativ zu R', units='ms')
        self.representative_plot.setLabel('left', 'Native Ableitung', units='mV')
        representative_layout.addWidget(self.representative_plot)
        self.representative_tab = self.tabs.addTab(representative_page, 'Repräsentativer Schlag')
        self.extra_research_tabs.append(self.representative_tab)
        self.representative_path.currentIndexChanged.connect(self.draw_representative_beat)
        self.representative_path.currentIndexChanged.connect(self.invalidate_median_template)
        self.representative_lead.currentIndexChanged.connect(self.draw_representative_beat)
        self.diagnostics = pg.PlotWidget()
        self.diagnostics.setLabel("bottom", "Frequenz", units="Hz")
        self.diagnostics.setLabel("left", "Leistungsdichte", units="mV²/Hz")
        self.diagnostics_tab = self.tabs.addTab(self.diagnostics, "Signaldiagnostik")
        from .research_display_widget import ResearchDisplayWidget
        self.scientific_displays = ResearchDisplayWidget()
        self.scientific_display_tab = self.tabs.addTab(self.scientific_displays, "Wissenschaftliche Darstellungen")
        self.transform_page = QWidget()
        tv = QVBoxLayout(self.transform_page)
        self.transform_segment = QComboBox()
        self.transform_segment.addItems(["Gesamter Record", "P", "QRS", "T"])
        self.transform_segment.currentIndexChanged.connect(self.draw_transforms)
        tv.addWidget(self.transform_segment)
        tv.addWidget(QLabel("Kors / inverse Dower: XYZ in mV. SVD: nichtanatomische Komponenten. Direct-8 ist kein 3D-Winkel."))
        self.transform_canvas = pg.GraphicsLayoutWidget()
        self.transform_canvas.setMinimumHeight(600)
        tv.addWidget(self.transform_canvas)
        self.transform_note = QLabel("Noch keine passende Analyse")
        self.transform_note.setWordWrap(True)
        tv.addWidget(self.transform_note)
        self.transform_tab = self.tabs.addTab(self.transform_page, "Transformationen")
        gallery_page = QWidget()
        gl = QVBoxLayout(gallery_page)
        gl.addWidget(QLabel("Cluster sind Morphologiegruppen, keine gesicherten Schlagdiagnosen. Manuelle Schlaglabels bleiben lokale Forschungsannotation."))
        self.gallery = QTableWidget(0, 5)
        self.gallery.setHorizontalHeaderLabels(["Schlag", "R-Zeit (s)", "Cluster", "Qualität / Typquelle", "Manuelles Label"])
        self.gallery.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gallery.itemSelectionChanged.connect(self.select_beat)
        gl.addWidget(self.gallery)
        self.beat_annotation_note = QLabel('Manuelle Annotationen: Originalsignal und Analyse bleiben unverändert.')
        self.beat_annotation_note.setWordWrap(True)
        gl.addWidget(self.beat_annotation_note)
        self.beat_plot = pg.PlotWidget()
        self.beat_plot.setLabel("bottom", "Zeit relativ zu R", units="s")
        self.beat_plot.setLabel("left", "Native Spannung", units="mV")
        gl.addWidget(self.beat_plot)
        label_beat = QPushButton("Schlag manuell kennzeichnen")
        label_beat.clicked.connect(self.label_beat)
        gl.addWidget(label_beat)
        self.gallery_tab = self.tabs.addTab(gallery_page, "Beat-Galerie")
        self.context = QTableWidget(0, 5)
        self.context.setHorizontalHeaderLabels(["Zeit", "Modalität", "Befund / Therapie", "Quelle", "Zeitunsicherheit"])
        self.context.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.context_tab = self.tabs.addTab(self.context, "Klinischer Zeitbezug")
        fiducial_review = QWidget()
        fv = QVBoxLayout(fiducial_review)
        note = QLabel('Gespeicherte Landmarkenprovenienz. Unsicherheit und Methodenabweichung sind keine klinisch kalibrierten 95-%-Konfidenzintervalle. Einzelne Delineator-Kandidaten werden nur angezeigt, wenn tatsächlich gespeichert; hier kein erfundener Methodenkonsens. Auswahl synchronisiert den Zeitcursor, ändert keine Grenze.')
        note.setWordWrap(True)
        fv.addWidget(note)
        self.fiducial_evidence = QTableWidget(0, 8)
        self.fiducial_evidence.setHorizontalHeaderLabels(['Landmarke', 'Zeit [s]', 'Qualitätswert', 'Unsicherheit [ms]', 'Abweichung [ms]', 'Unterstützung', 'Quelle / Ableitungen', 'Status'])
        self.fiducial_evidence.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fiducial_evidence.setMaximumHeight(300)
        for column, width in enumerate((280, 90, 100, 130, 130, 110, 230, 300)):
            self.fiducial_evidence.setColumnWidth(column, width)
        self.fiducial_evidence.cellClicked.connect(self._inspect_fiducial_evidence)
        fv.addWidget(self.fiducial_evidence)
        self.fiducial_overlay = pg.PlotWidget()
        self.fiducial_overlay.setMinimumHeight(280)
        self.fiducial_overlay.setLabel('bottom', 'Native Signalzeit', units='s')
        self.fiducial_overlay.setLabel('left', 'Spannung', units='mV')
        fv.addWidget(self.fiducial_overlay)
        self.fiducial_evidence_tab = self.tabs.addTab(fiducial_review, 'Landmarken-Evidenz')
        self._contain_detail_tabs()
        for control in (self.layout_choice, self.lead_choice, self.speed, self.gain):
            control.currentIndexChanged.connect(self.redraw)
        self.lead_order.currentIndexChanged.connect(self._lead_order_changed)
        self.avr_polarity.currentIndexChanged.connect(self._avr_polarity_changed)
        self.fit.toggled.connect(self.redraw)
        self.view_duration.currentIndexChanged.connect(self.redraw)
        self.start.valueChanged.connect(self.redraw)
        self.pixels_per_mm.valueChanged.connect(self.redraw)
        for key, action in (("Ctrl+Z", lambda: self.undo_focused(False)), ("Ctrl+Shift+Z", lambda: self.undo_focused(True)), ("Alt+Right", lambda: self.navigate(1)), ("Alt+Left", lambda: self.navigate(-1))):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
        self._mode_changed()
        fullscreen = QShortcut(QKeySequence("F11"), self)
        fullscreen.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        fullscreen.activated.connect(lambda: self.window().showNormal() if self.window().isFullScreen() else self.window().showFullScreen())
        for button in self.findChildren(QPushButton):
            if not button.toolTip():
                button.setToolTip(button.text() + " – ausschließlich im lokalen Labor; Produktionsresultate bleiben unverändert.")
        self._display_settings = QSettings('VCG', 'ExpertECGLabDisplay')
        self._display_defaults_read_error = False
        self._load_display_defaults()

    def _contain_detail_tabs(self):
        """Keep research pages scrollable without making the ECG workspace enormous.

        The clinical report page retains its direct widget contract. Other pages
        may have tall plots or tables; those scroll inside the tab, leaving the
        ECG and its navigation available in the surrounding workspace.
        """
        for index in range(1, self.tabs.count()):
            page = self.tabs.widget(index)
            if isinstance(page, QScrollArea):
                continue
            label = self.tabs.tabText(index)
            tooltip = self.tabs.tabToolTip(index)
            self.tabs.removeTab(index)
            scroll = QScrollArea(self.tabs)
            scroll.setObjectName("labDetailScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            self.tabs.insertTab(index, scroll, label)
            self.tabs.setTabToolTip(index, tooltip)
        self.tabs.setMinimumHeight(360)
        self.tabs.setMaximumHeight(540)

    def calibrate_screen(self):
        from ..lab_screen_calibration import ScreenCalibrationDialog
        dialog = ScreenCalibrationDialog(self.pixels_per_mm.value(), self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.pixels_per_mm.setValue(dialog.calibration())
            self.fit.setChecked(False)

    def undo_focused(self, redo=False):
        """Text editing must not silently change signal annotations."""
        editor = QApplication.focusWidget()
        if isinstance(editor, (QTextEdit, QLineEdit)) and self.isAncestorOf(editor):
            if not editor.isReadOnly():
                (editor.redo if redo else editor.undo)()
            return
        (self.redo if redo else self.undo)()

    def _load_display_defaults(self):
        from ..lab_display_preferences import KEY, LAYOUTS, decode_display_defaults
        try:
            if self._display_settings.status() != QSettings.Status.NoError:
                raise ValueError('Anzeigevorgaben nicht lesbar')
            value = decode_display_defaults(self._display_settings.value(KEY))
            if value is None:
                return
        except (ValueError, TypeError) as exc:
            self._display_defaults_read_error = True
            self.save_display_defaults.setEnabled(False)
            self.save_display_defaults.setText('Ansichtsstandard nicht lesbar')
            self.save_display_defaults.setToolTip(str(exc) + ' · Gespeicherte Daten bleiben unverändert. Vorgaben vor einer Korrektur sichern.')
            logging.getLogger(__name__).warning('Lab display defaults rejected: %s', exc)
            return
        self.layout_choice.setCurrentIndex(LAYOUTS.index(value['layout']))
        self.lead_choice.setCurrentText(value['lead'])
        self.speed.setCurrentText(value['speed'])
        self.gain.setCurrentText(value['gain'])
        self.fit.setChecked(value['fit'])
        self.avr_polarity.setCurrentIndex(int(value['standard_avr'] == 'inverted'))
        self.lead_order.setCurrentIndex(int(value['order'] == 'cabrera'))

    def _save_display_defaults(self):
        from ..lab_display_preferences import KEY, LAYOUTS, encode_display_defaults
        if self._display_defaults_read_error:
            return
        value = dict(version='1.0.0', layout=LAYOUTS[self.layout_choice.currentIndex()],
            lead=self.lead_choice.currentText(), order='cabrera' if self.lead_order.currentIndex() == 1 else 'standard',
            standard_avr='inverted' if self._standard_avr_polarity else 'native',
            speed=self.speed.currentText(), gain=self.gain.currentText(), fit=self.fit.isChecked())
        self._display_settings.setValue(KEY, encode_display_defaults(value))
        self._display_settings.sync()
        if self._display_settings.status() != QSettings.Status.NoError:
            self.save_display_defaults.setText('Speichern fehlgeschlagen')
            logging.getLogger(__name__).error('Lab display defaults could not be saved')
        else:
            self.save_display_defaults.setText('Ansichtsstandard gespeichert')

    def _mark_dirty(self):
        if not self._loading:
            self._dirty = True
            self.report_status.setText('Ungesicherte Änderungen – erneute Prüfung erforderlich; gespeicherte Revision unverändert.')
            self.recovery_timer.start()

    def _save_recovery(self):
        if self.case is None or not self._dirty:
            return
        from ...mimic.storage import atomic_write_json
        try:
            self.store.root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.store.root / "draft_recovery.json", {
                "schema_version": "1.0.0", "case_id": self.case.case_id,
                "source_sha256": self.case.source_sha256, "parent_revision_id": self.parent_revision,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "recovery_not_reviewed",
                "sections": {name: edit.toPlainText() for name, edit in self.sections.items()},
                "author": self.author.text(), "manual_overrides": self.manual_overrides,
                "suggestion_review": self.suggestion_review, "analysis_id": self.payload.get("analysis_id", ""),
            })
        except OSError:
            logging.getLogger(__name__).exception("Local draft recovery could not be saved")
            self.report_status.setText("Entwurfsicherung fehlgeschlagen – bitte manuell speichern.")

    def _mode_changed(self, *_):
        self._refresh_qt_alert()
        research = self.mode.currentIndex() == 1 and not self.blind.isChecked()
        if not self._spatial_preview_user_set:
            self.spatial_preview.setChecked(not research)
        self.tabs.setTabVisible(self.research_tab, research)
        self.tabs.setTabVisible(self.diagnostics_tab, research)
        self.tabs.setTabVisible(self.scientific_display_tab, research)
        self.tabs.setTabVisible(self.transform_tab, research)
        self.tabs.setTabVisible(self.gallery_tab, research)
        self.tabs.setTabVisible(self.context_tab, research)
        self.tabs.setTabVisible(self.fiducial_evidence_tab, research)
        for index in self.extra_research_tabs:
            self.tabs.setTabVisible(index, research)
        self.tabs.setTabVisible(1, not self.blind.isChecked())
        self.tabs.setTabVisible(self.suggestions_tab, not self.blind.isChecked())
        self.quick_suggestion_panel.setVisible(not self.blind.isChecked())
        self.tabs.setTabVisible(self.criteria_catalog_tab, not self.blind.isChecked())
        self.comparison_values.setVisible(not self.blind.isChecked())
        self.comparison_changes.setVisible(not self.blind.isChecked())
        self.pdf_research.setEnabled(not self.blind.isChecked())
        if self.blind.isChecked():
            self.pdf_research.setChecked(False)
        self._place_heart_overlay()
        self.researchModeChanged.emit(research)
        if self.signal is not None:
            self.redraw()

    def _spatial_preview_clicked(self, _checked):
        self._spatial_preview_user_set = True
        self._place_heart_overlay()

    # ---- Räumliche Viewer unter dem Signal; Fehler dort dürfen das Labor nie anhalten
    def _install_heart_overlay(self):
        self.heart_overlay = None
        self.heart = None
        self.vcg_overlay = None
        self.vcg_mini = None
        self.vcg_window = None
        self._card_user_moved = {}
        self.heart_overlay = self._build_card(
            "heart", "Herz · Illustration", "Herzillustration ein- und ausklappen",
            lambda frame: self._create_heart_widget(frame))
        self.vcg_overlay = self._build_card(
            "vcg", "VCG · Schleife", "VCG-Vorschau ein- und ausklappen",
            lambda frame: self._create_vcg_widget(frame))
        if self.vcg_overlay is not None:
            self.vcg_overlay.set_collapsed(True)  # standardmäßig klein, damit beide Karten Platz haben
        self._apply_card_theme()

    def _build_card(self, key, title, tooltip, factory):
        try:
            card = _OverlayCard(title, self.viewer_strip, tooltip)
            widget = factory(card)
            if widget is None:
                card.deleteLater()
                return None
            widget.setObjectName("overlayCardBody")  # sonst malt die Programmfläche einen Rand in die Karte
            card.body.addWidget(widget)
            card.adjustSize()
            card.hide()
            self.viewer_cards_layout.insertWidget(self.viewer_cards_layout.count() - 1, card)
            self._card_user_moved[key] = False
            card.moved.connect(lambda k=key: self._card_user_moved.__setitem__(k, True))
            card.toggled.connect(lambda _on, k=key: self._card_expanded(k))
            return card
        except Exception as exc:  # OpenGL/Treiber: Labor bleibt ohne diese Karte voll nutzbar
            logging.getLogger(__name__).warning("Karte %s nicht verfügbar: %s", key, exc)
            return None

    def _create_heart_widget(self, frame):
        from ..heart3d.widgets import MiniHeartWidget
        self.heart = MiniHeartWidget(frame, size=150, auto_analysis=False)
        return self.heart

    def _create_vcg_widget(self, frame):
        from ..vcglab.widgets import MiniVCGWidget
        self.vcg_mini = MiniVCGWidget(frame, size=150, auto_analysis=False, dark=getattr(self, "_dark", True))
        self.vcg_mini.open_button.setToolTip("VCG-Labor in einem eigenen Fenster öffnen (Ebenen, Maße, Schleifenlesung)")
        self.vcg_mini.openRequested.disconnect()
        self.vcg_mini.openRequested.connect(self._open_vcg_window)
        return self.vcg_mini

    def show_signal_presentation(self):
        heart = getattr(self, "heart", None)
        vcg = getattr(self, "vcg_mini", None)
        if heart is None or heart.engine.timeline is None or vcg is None or vcg.record is None:
            QMessageBox.information(self, "Signalreise", "Zuerst einen Fall mit EKG- und VCG-Auswertung laden.")
            return
        from .signal_presentation import SignalPresentationDialog
        previous_playing = heart.clock.playing
        heart.clock.pause()
        dialog = SignalPresentationDialog(engine=heart.engine, model=heart.model, record=vcg.record,
                                          dark=getattr(self, "_dark", True), parent=self)
        dialog.exec()
        if previous_playing and heart.isVisible():
            from ..motion import current_motion_policy
            if not current_motion_policy().reduced:
                heart.clock.play()

    def show_synthetic_presentation(self):
        """Build a clearly synthetic presentation without reading any patient or run data."""
        if getattr(self, "heart", None) is None:
            QMessageBox.information(self, "Signalreise", "Herzmodell ist in dieser GUI-Sitzung nicht verfügbar.")
            return
        self._synthetic_demo_run += 1
        run = self._synthetic_demo_run
        self.presentation_button.setEnabled(False)
        self.presentation_button.setToolTip("Synthetische Demo wird im Hintergrund vorbereitet …")

        def prepare():
            try:
                from ..heart3d.demo import scenario_sinus
                from ..vcglab.analysis import quick_record
                timeline = scenario_sinus().timeline()
                record = quick_record(timeline.signal, timeline.fs, timeline.lead_names, "Synthetischer Sinusfall")
            except Exception as exc:
                timeline, record = None, f"{type(exc).__name__}: {exc}"
            try:
                self.syntheticDemoReady.emit(run, timeline, record)
            except RuntimeError:
                pass  # Labor wurde während der Vorbereitung geschlossen

        threading.Thread(target=prepare, name="vcg-synthetic-presentation", daemon=True).start()

    def _on_synthetic_demo_ready(self, run, timeline, record):
        if run != self._synthetic_demo_run:
            return
        self.presentation_button.setEnabled(True)
        self.presentation_button.setToolTip("Geladenen Fall oder synthetische Demo zeigen; keine neue Produktionsanalyse")
        if timeline is None:
            QMessageBox.warning(self, "Synthetische Signalreise", f"Demo nicht verfügbar: {record}")
            return
        from ..heart3d.engine import AnimationEngine
        from .signal_presentation import SignalPresentationDialog
        model = self.heart.model
        engine = AnimationEngine(model)
        engine.set_timeline(timeline)
        self._synthetic_demo_dialog = SignalPresentationDialog(
            engine=engine, model=model, record=record, dark=getattr(self, "_dark", True),
            synthetic=True, parent=self)
        self._synthetic_demo_dialog.show()
        self._synthetic_demo_dialog.raise_()

    def show_public_presentation(self):
        """Verify and analyze one bundled public ECG outside the production run."""
        if getattr(self, "heart", None) is None:
            QMessageBox.information(self, "Signalreise", "Herzmodell ist in dieser GUI-Sitzung nicht verfügbar.")
            return
        self._public_demo_run += 1
        run = self._public_demo_run
        self.presentation_button.setEnabled(False)
        self.presentation_button.setToolTip("Hashgeprüfter öffentlicher Demofall wird im Hintergrund vorbereitet …")

        def prepare():
            try:
                from ..demo_capsules import load_public_demo_signal
                from ..heart3d.timeline import timeline_from_signal
                from ..vcglab.analysis import analyze_signal
                signal, fs, names, _manifest = load_public_demo_signal()
                record = analyze_signal(signal, fs, names, "Öffentlicher PTB-XL-Demofall")
                timeline = timeline_from_signal(signal, fs, names)
            except Exception as exc:
                timeline, record = None, f"{type(exc).__name__}: {exc}"
            try:
                self.publicDemoReady.emit(run, timeline, record)
            except RuntimeError:
                pass  # Labor wurde während der Vorbereitung geschlossen

        threading.Thread(target=prepare, name="vcg-public-demo-presentation", daemon=True).start()

    def _on_public_demo_ready(self, run, timeline, record):
        if run != self._public_demo_run:
            return
        self.presentation_button.setEnabled(True)
        self.presentation_button.setToolTip("Geladenen Fall oder isolierte Demofälle zeigen; keine neue Produktionsanalyse")
        if timeline is None:
            QMessageBox.warning(self, "Öffentliche Signalreise", f"Demokapsel nicht verfügbar: {record}")
            return
        from ..heart3d.engine import AnimationEngine
        from .signal_presentation import SignalPresentationDialog
        model = self.heart.model
        engine = AnimationEngine(model)
        engine.set_timeline(timeline)
        self._public_demo_dialog = SignalPresentationDialog(
            engine=engine, model=model, record=record, dark=getattr(self, "_dark", True),
            public_demo=True, parent=self)
        self._public_demo_dialog.show()
        self._public_demo_dialog.raise_()

    def _cards(self):
        return [(key, card) for key, card in (("vcg", getattr(self, "vcg_overlay", None)),
                                              ("heart", getattr(self, "heart_overlay", None))) if card is not None]

    def _card_expanded(self, key):
        """Keep the spatial viewers in their own row, without obscuring leads."""
        self._sync_viewer_strip_height()

    def _sync_viewer_strip_height(self):
        self.viewer_strip_layout.invalidate()
        height = self.viewer_strip.sizeHint().height() if not self.viewer_strip.isHidden() else 0
        self.viewer_strip.setMinimumHeight(height)
        stage_layout = self.signal_stage.layout()
        stage_layout.invalidate()
        margins = stage_layout.contentsMargins()
        header_height = stage_layout.itemAt(0).minimumSize().height()
        required_height = (
            margins.top() + margins.bottom() + header_height
            + self.canvas.minimumHeight() + height + 2 * stage_layout.spacing()
        )
        self.signal_stage.setMinimumHeight(max(stage_layout.minimumSize().height(), required_height))
        self.signal_stage.updateGeometry()
        self.layout().invalidate()
        self.updateGeometry()
        if self.isVisible():
            self.adjustSize()

    def _apply_card_theme(self):
        dark = getattr(self, "_dark", True)
        for _key, card in self._cards():
            card.apply_theme(dark)
        if getattr(self, "vcg_mini", None) is not None:
            try:
                self.vcg_mini.set_dark(dark)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        if obj is getattr(self, "canvas", None) and event.type() == QEvent.Type.Resize:
            self._place_heart_overlay()
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        for name in ("heart", "vcg_mini"):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.stop()
                except Exception:
                    pass
        if getattr(self, "vcg_window", None) is not None:
            try:
                self.vcg_window.close()
            except Exception:
                pass
        super().closeEvent(event)

    def _heart_canvas_margin(self, reserve):
        """Rechten Rand im Signal-Canvas freihalten, damit die Karten keine Ableitung verdecken."""
        try:
            self.canvas.ci.layout.setContentsMargins(0, 0, int(reserve), 0)
        except Exception:
            pass

    def _place_heart_overlay(self):
        """The viewers are docked below the ECG; no signal area is reserved."""
        self.viewer_strip.setVisible(
            not self.blind.isChecked()
            and self.spatial_preview.isChecked()
            and any(not card.isHidden() for _, card in self._cards())
        )
        self._sync_viewer_strip_height()

    # ---- VCG-Labor (eigenes Fenster, wie die Herzillustration; rechnet erst beim Öffnen)
    def _open_vcg_window(self):
        """Vorschaukarte vergrößern: VCG-Labor in einem eigenen Fenster öffnen."""
        if getattr(self, "vcg_window", None) is None:
            try:
                from ..vcglab.widgets import VCGLabWindow
                self.vcg_window = VCGLabWindow(dark=getattr(self, "_dark", True))
                self.vcg_window.panel.recordChanged.connect(self._vcg_record_ready)
            except Exception as exc:  # OpenGL/Treiber: Labor bleibt ohne VCG-Fenster voll nutzbar
                logging.getLogger(__name__).warning("VCG-Labor nicht verfügbar: %s", exc)
                return
            self._vcg_push_source()
        self.vcg_window.show()
        self.vcg_window.raise_()
        self.vcg_window.activateWindow()

    def open_wpw_localization(self) -> None:
        if self.blind.isChecked() or self.signal is None:
            return
        from .wpw_dialog import WPWLocalizationDialog
        dialog = getattr(self, 'wpw_dialog', None)
        if dialog is None:
            dialog = WPWLocalizationDialog(self)
            self.wpw_dialog = dialog
        dialog.show()
        dialog.raise_()

    def open_pvc_localization(self) -> None:
        if self.blind.isChecked() or self.signal is None:
            return
        from .pvc_localization_dialog import PVCLocalizationDialog
        dialog = getattr(self, 'pvc_localization_dialog', None)
        if dialog is None:
            dialog = PVCLocalizationDialog(self)
            self.pvc_localization_dialog = dialog
        dialog.show()
        dialog.raise_()

    def open_atrial_focus(self) -> None:
        if self.blind.isChecked() or self.signal is None:
            return
        from .atrial_focus_dialog import AtrialFocusDialog
        dialog = getattr(self, 'atrial_focus_dialog', None)
        if dialog is None:
            dialog = AtrialFocusDialog(self)
            self.atrial_focus_dialog = dialog
        dialog.show()
        dialog.raise_()

    def _vcg_push_source(self):
        """Aktuellen Fall an ein geöffnetes VCG-Fenster geben (dort wird erst bei Sichtbarkeit gerechnet)."""
        window = getattr(self, "vcg_window", None)
        if window is None or self.signal is None:
            return
        metadata = self.case.metadata if self.case else {}
        signal, fs = self._vcg_signal_path()
        try:
            window.panel.set_source(signal, fs, LEADS, self.case.pseudonym if self.case else "",
                                    metadata.get("age"), metadata.get("sex"), self.payload or None)
        except Exception as exc:
            logging.getLogger(__name__).warning("VCG-Labor: Signal nicht übernommen: %s", exc)

    def _vcg_record_ready(self, record):
        """Gespeicherten oder zur Vorschau berechneten VCG-Datensatz übernehmen."""
        if getattr(self, "vcg_mini", None) is None or record is None:
            return
        try:
            self.vcg_mini.set_record(record)
        except Exception as exc:
            logging.getLogger(__name__).warning("VCG-Vorschau nicht aktualisiert: %s", exc)

    def _vcg_set_source(self, fs):
        signal, fs = self._vcg_signal_path()
        metadata = self.case.metadata if self.case else {}
        label = self.case.pseudonym if self.case else ""
        self._vcg_push_source()
        if getattr(self, "vcg_mini", None) is not None and signal is not None:
            try:
                if self.payload and self.payload.get('case_id') == self.case.case_id:
                    self.vcg_mini.set_analysis(self.payload, signal, fs, LEADS, label,
                                               metadata.get("age"), metadata.get("sex"))
                else:
                    self.vcg_mini.set_signal(signal, fs, LEADS, label, metadata.get("age"), metadata.get("sex"))
                self.vcg_overlay.show()
                self.viewer_signal_note.setText(
                    f'VCG: gespeicherter robuster Morphologiepfad · {fs:g}-Hz-Analyse'
                    if self.analysis_arrays is not None and signal is not self.signal else
                    'VCG: natives Originalsignal · Vorschau bis zum Laden der Analysearrays')
                self._place_heart_overlay()
            except Exception as exc:
                logging.getLogger(__name__).warning("VCG-Vorschau: Signal nicht übernommen: %s", exc)

    def _vcg_signal_path(self):
        if (self.reference is not None and isinstance(self.analysis_arrays, dict)
                and self.payload.get('analysis_id')):
            signal = self.analysis_arrays.get('robust_morphology_ecg')
            rate = self.analysis_arrays.get('sampling_rate_hz')
            if signal is not None and rate is not None:
                return signal, float(rate)
        return self.signal, self.reference.sampling_rate_hz if self.reference else 500.0

    def _heart_set_signal(self, fs):
        if getattr(self, "heart", None) is None or self.signal is None:
            return
        try:
            self.heart.set_signal(self.signal, fs, LEADS, self.case.pseudonym if self.case else "")
            self.heart_overlay.show()
        except Exception as exc:
            logging.getLogger(__name__).warning("Herzillustration: Signal nicht übernommen: %s", exc)

    def _heart_set_analysis(self, payload, analysis_fs=500.0):
        if getattr(self, "heart", None) is None:
            return
        try:
            self.heart.set_analysis(payload, analysis_fs)
        except Exception as exc:
            logging.getLogger(__name__).warning("Herzillustration: Analysestand nicht übernommen: %s", exc)

    def set_dark(self, dark):
        self._dark = bool(dark)
        self.endpoint_card.set_dark(dark)
        self.marker_evidence.set_dark(dark)
        self.laddergram.set_dark(dark)
        if getattr(self, "heart", None) is not None:
            try:
                self.heart.set_dark(dark)
            except Exception:
                pass
        if getattr(self, "vcg_window", None) is not None:
            try:
                self.vcg_window.panel.set_dark(dark)
            except Exception:
                pass
        self._apply_card_theme()
        self.canvas.setBackground("#111923" if dark else "#fffdf9")
        if self.signal is None:
            self.canvas_empty_label.setText(
                "NOCH KEIN EKG GELADEN  ·  Fall auswählen oder Signal importieren",
                color="#8fb9bd" if dark else "#52616b",
            )
        self.comparison_plot.setBackground("#111923" if dark else "#fffdf9")
        self.comparison_changes.set_dark(dark)
        self.fiducial_overlay.setBackground('#111923' if dark else '#fffdf9')
        for name in ('bottom', 'left'):
            axis = self.fiducial_overlay.getAxis(name)
            axis.setPen('#dce5ef' if dark else '#263746')
            axis.setTextPen('#dce5ef' if dark else '#263746')
        self.scientific_displays.canvas.setBackground("#111923" if dark else "#fffdf9")
        self.diagnostics.setBackground("#111923" if dark else "#fffdf9")
        self.transform_canvas.setBackground("#111923" if dark else "#fffdf9")
        self.beat_plot.setBackground("#111923" if dark else "#fffdf9")
        self.marker_signal.setBackground("#111923" if dark else "#fffdf9")
        self.representative_plot.setBackground("#111923" if dark else "#fffdf9")
        for name in ('bottom', 'left'):
            self.representative_plot.getAxis(name).setTextPen('#dce5ef' if dark else '#263746')
            self.representative_plot.getAxis(name).setPen('#dce5ef' if dark else '#263746')
        self.draw_representative_beat()
        self.redraw()

    def load_case(self, library, case_id):
        self.laddergram.persist_draft()
        next_case = library.load_case(case_id)
        if (self.case and self.case.case_id == case_id
                and self.library is not None
                and Path(self.library.root).resolve() == Path(library.root).resolve()
                and self.case.source_sha256 == next_case.source_sha256):
            return
        if self.case and self._dirty:
            if not self.save_report("draft"):
                raise ValueError("Ungesicherter Befund: Fallwechsel abgebrochen")
        next_signal, fs = library.load_signal(case_id)
        next_reference = SignalReference(case_id, next_case.source_sha256, fs, len(next_signal))
        self.library = library
        self.case = next_case
        self.signal = next_signal
        self.reference = next_reference
        self.store = LabReviewStore(library.root, case_id)
        self.laddergram.set_context(asdict(self.reference), self.store.root, self.signal)
        self.lbbb_context.setChecked(False)
        self.ischemia_context_kind.setCurrentIndex(0)
        self.marker_groups = {}
        groups_path = self.store.root / 'marker_groups.json'
        self._marker_groups_read_error = False
        if groups_path.is_file():
            try:
                data = json.loads(groups_path.read_text(encoding='utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('Gruppendatei muss ein Objekt sein')
                groups = data.get('groups')
                if data.get('schema_version') != '1.0.0' or not isinstance(groups, dict):
                    raise ValueError('Unbekanntes Gruppenformat')
                if any(not isinstance(k, str) or not k.strip() or not isinstance(v, list)
                       or not 1 <= len(v) <= 20 or not all(isinstance(n, str) and n for n in v)
                       for k, v in groups.items()):
                    raise ValueError('Ungültige Vergleichsgruppe')
                self.marker_groups = groups
            except (OSError, ValueError, TypeError) as exc:
                self._marker_groups_read_error = True
                logging.getLogger(__name__).warning('Vergleichsgruppen nicht geladen: %s', exc)
                QMessageBox.warning(self, 'Vergleichsgruppen', 'Gruppendatei nicht lesbar; Original bleibt unverändert.')
        self._refresh_marker_groups()
        self.marker_favorites = set()
        favorites_path = self.store.root / "marker_favorites.json"
        self._marker_favorites_read_error = False
        if favorites_path.is_file():
            try:
                favorites = json.loads(favorites_path.read_text(encoding="utf-8"))
                if not isinstance(favorites, dict):
                    raise ValueError('Favoritendatei muss ein Objekt sein')
                names = favorites.get("markers", [])
                if favorites.get("schema_version") != "1.0.0" or not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                    raise ValueError("Unbekanntes Favoritenformat")
                self.marker_favorites = set(names)
            except (OSError, ValueError, TypeError) as exc:
                self._marker_favorites_read_error = True
                logging.getLogger(__name__).warning("Favoriten konnten nicht geladen werden: %s", exc)
                QMessageBox.warning(self, "Favoriten", "Lokale Favoriten konnten nicht geladen werden; Originaldatei bleibt unverändert.")
        self.payload = {}
        self.device_interval_document = None
        self.refresh_device_intervals()
        self.analysis_arrays = None
        self.markers.setRowCount(0)
        self._refresh_qt_alert()
        self.measurements.setRowCount(0)
        self._populate_lead_matrix({})
        self.marker_detail.clear()
        self.marker_provenance_tree.clear()
        self.marker_evidence.set_evidence(None)
        self.marker_signal.clear()
        self.fiducials = {}
        self.marker_beat.clear()
        self.marker_beat.setVisible(False)
        self.manual_fiducials = False
        self.marker_beat_audit.setRowCount(0)
        self.marker_beat_audit.setVisible(False)
        self.history.clear()
        self.redo_history.clear()
        self.calipers.clear()
        self.saved_measurements.clear()
        self.score_item_note.setText("Kein auswertbares QTc-Einzelitem; kein vollständiger LQTS-Score.")
        self.prior = None
        self.prior_payload = {}
        self.comparison_changes.set_rows([])
        self.comparison_analysis.clear()
        self.comparison_values.setRowCount(0)
        self.comparison_plot.clear()
        self.manual_overrides = {}
        self.suggestion_review = {}
        self.suggestions.setRowCount(0)
        self.gallery.setRowCount(0)
        self.context.setRowCount(0)
        self.fiducial_evidence.setRowCount(0)
        self.fiducial_overlay.clear()
        self.beat_plot.clear()
        self._loading = True
        for edit in self.sections.values():
            edit.clear()
        reports = self.store.reports()
        self.parent_revision = ""
        self.report_analysis_id = ""
        self.report_measurements = []
        if reports:
            last = reports[-1]
            self.parent_revision = last["revision_id"]
            self.report_analysis_id = last.get("analysis_id", "")
            self.report_measurements = last.get("measurements", [])
            self.manual_overrides = last.get("manual_overrides", {})
            self.suggestion_review = last.get("suggestion_review", {})
            for name, value in last["sections"].items():
                if name in self.sections:
                    self.sections[name].setPlainText(value)
            self.report_status.setText(f"{last['status']} · {last['author']} · {last['created_at']}")
        else:
            self.report_status.setText("Noch kein gespeicherter Befund")
        self._loading = False
        self._dirty = False
        recovery_path = self.store.root / "draft_recovery.json"
        if recovery_path.is_file():
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            if (recovery.get("source_sha256") == self.case.source_sha256
                    and recovery.get("parent_revision_id") == self.parent_revision
                    and recovery.get("created_at", "") > (reports[-1]["created_at"] if reports else "")):
                self._loading = True
                for name, value in recovery.get("sections", {}).items():
                    if name in self.sections:
                        self.sections[name].setPlainText(value)
                self.author.setText(recovery.get("author", ""))
                self.manual_overrides = recovery.get("manual_overrides", {})
                self.suggestion_review = recovery.get("suggestion_review", {})
                self._loading = False
                self._dirty = True
                self.report_status.setText("Ungesicherten Entwurf wiederhergestellt – nicht geprüft.")
        self.refresh_manual_overrides()
        self.start.setMaximum(max(0, (len(self.signal) - 1) / fs))
        self.start.setValue(0)
        self.landmark_time.setMaximum((len(self.signal) - 1) / fs)
        self.refresh_comparison_cases()
        self.scientific_displays.set_data(self.signal, self.reference.sampling_rate_hz)
        self.notice.setText(f"{self.case.pseudonym} · Originalsignal {fs:g} Hz · Auflösung {1000/fs:g} ms · {len(self.signal)/fs:.2f} s · mV · keine Anzeigefilter")
        if self.case.metadata.get("source_metadata", {}).get("synthetic"):
            self.notice.setText("SYNTHETISCHE GEWEBESIMULATION · kein Patient · Amplitude nur illustrativ skaliert · "
                                f"{fs:g} Hz · {len(self.signal)/fs:.2f} s · keine klinische Interpretation")
        self._heart_set_signal(fs)
        self._vcg_set_source(fs)
        self._place_heart_overlay()
        contexts = self.case.metadata.get("clinical_context", [])
        if not isinstance(contexts, list):
            contexts = []
        self.context.setRowCount(max(1, len(contexts)))
        if not contexts:
            self.context.setItem(0, 2, QTableWidgetItem("Keine verknüpften klinischen Kontextdaten vorhanden; nicht als unauffällig interpretieren."))
        for row, event in enumerate(contexts):
            for col, key in enumerate(("time", "modality", "finding", "source", "time_uncertainty")):
                self.context.setItem(row, col, QTableWidgetItem(str(event.get(key, "unbekannt"))))
        annotations = sorted((self.store.root / "annotations").glob("*.json"))
        if annotations:
            data = json.loads(annotations[-1].read_text(encoding="utf-8"))
            if data.get("signal") == asdict(self.reference):
                self.fiducials = data.get("fiducials", {})
                self.manual_fiducials = True
                self.saved_measurements = data.get("measurements", [])
        self.redraw()

    def set_analysis(self, payload, analysis_fs=500.0):
        if not isinstance(payload, dict):
            raise ValueError('Analysestand muss ein Objekt sein')
        if not self.case or payload.get("case_id") != self.case.case_id:
            return False
        reference = payload.get("signal_reference", {})
        if not isinstance(reference, dict):
            raise ValueError('Ungültige Signalreferenz im Analysestand')
        if reference and reference.get("source_sha256") != self.case.source_sha256:
            raise ValueError("Analysestand gehört zu einem anderen Originalsignal")
        if getattr(self, 'rule_evidence_dialog', None) is not None:
            self.rule_evidence_dialog.close()
        self.scientific_displays.set_data(self.signal, self.reference.sampling_rate_hz if self.reference else None)
        self.payload = payload
        self._heart_set_analysis(payload, analysis_fs)
        self._vcg_set_source(self.reference.sampling_rate_hz if self.reference else analysis_fs)
        self.refresh_device_intervals()
        self._refresh_qt_alert()
        self.analysis_arrays = None
        self.median_template = None
        self.template_cluster.clear()
        self.template_cluster.addItem('Cluster ausdrücklich wählen', None)
        inventory = payload.get('beat_inventory', [])
        clusters = {item.get('morphology_cluster_id') for item in inventory if isinstance(item, dict)
                    and isinstance(item.get('morphology_cluster_id'), (str, int))
                    and not isinstance(item.get('morphology_cluster_id'), bool)} if isinstance(inventory, list) else set()
        for cluster in sorted(clusters, key=str):
            self.template_cluster.addItem(str(cluster), cluster)
        self.draw_representative_beat()
        if self.parent_revision and self.report_analysis_id != payload.get("analysis_id", ""):
            self._dirty = True
            self.report_status.setText("Anderer Analysestand als im gespeicherten Befund – erneute Prüfung erforderlich.")
        analysis_fs = payload.get("lab_processing", {}).get("analysis_sampling_rate_hz", analysis_fs)
        if not self.manual_fiducials:
            self.fiducials = {}
            for name, sample in payload.get("delineation", {}).get("representative_fiducials", {}).items():
                time = sample / analysis_fs
                if 0 <= time <= (self.reference.sample_count - 1) / self.reference.sampling_rate_hz:
                    self.fiducials[name] = self.reference.sample(time)
        entries = []
        intervals = payload.get("delineation", {}).get("interval_measurements", [])
        details = payload.get('delineation', {}).get('fiducial_details', {})
        details = details if isinstance(details, dict) else {}
        details = dict(details)
        candidates = payload.get('delineation', {}).get('method_candidates', {})
        if isinstance(candidates, dict) and candidates.get('schema_version') == '1.0.0':
            items = candidates.get('items', [])
            for index, candidate in enumerate(items[:100] if isinstance(items, list) else []):
                if not isinstance(candidate, dict):
                    continue
                details[f"Kandidat {candidate.get('method', '?')} · {candidate.get('landmark', '?')} · {index}"] = {
                    'landmark': candidate.get('landmark'),
                    'time_seconds': candidate.get('time_seconds'), 'confidence': candidate.get('confidence'),
                    'source': candidate.get('source'),
                    'wave_status': 'vor Fusion / manueller Korrektur; zeitlich zulässig: ' + str(candidate.get('temporally_admissible', 'unbekannt'))}
        self.fiducial_evidence.setRowCount(len(details))
        self.fiducial_overlay.clear()
        for row, (name, item) in enumerate(sorted(details.items(), key=lambda pair: (pair[0].startswith('Kandidat '), pair[0]))):
            item = item if isinstance(item, dict) else {}
            values = (name, item.get('time_seconds'), item.get('confidence'), item.get('uncertainty_ms'),
                      item.get('conflict_ms'), item.get('support_count'),
                      f"{item.get('source', 'unbekannt')} / {item.get('lead_names', [])}",
                      f"{item.get('wave_status', 'unbekannt')} · manuell geprüft: {item.get('manually_reviewed', 'unbekannt')}")
            for col, value in enumerate(values):
                cell = QTableWidgetItem('unbekannt' if value is None else str(value))
                cell.setToolTip(cell.text())
                self.fiducial_evidence.setItem(row, col, cell)
            self.fiducial_evidence.item(row, 0).setData(Qt.ItemDataRole.UserRole, item.get('landmark', name))
        from ...mimic.lab_score_items import lqts_qtc_item
        item = lqts_qtc_item(intervals if isinstance(intervals, list) else [], self.case.metadata.get('sex'))
        self.score_item_note.setText(f"ESC 2022, Tabelle 10: QTc-Bazett-Einzelitem = {item['points'] if item['points'] is not None else 'unbekannt'} Punkte. " + item['warning_de'])
        self.score_item_note.setToolTip(item['source'] + '\n' + item['reason'])
        from ..lab_qt_context import qt_context
        qt_review = qt_context(intervals, age=self.case.metadata.get('age'),
                               sex=self.case.metadata.get('sex'),
                               rhythm=payload.get('delineation', {}).get('rhythm_context'))
        self.score_item_note.setText(self.score_item_note.text() + '\n' + '\n'.join(qt_review['notes']))
        self.score_item_note.setToolTip(self.score_item_note.toolTip() + '\n' + qt_review['source'])
        for item in intervals if isinstance(intervals, list) else []:
            origin = "Automatisch · repräsentativer Schlag"
            if item.get('measurement_method') == 'same_representative_beat_qt_minus_0_485_qrs':
                origin = (f"{item.get('formula')} · {item.get('interpretation_note')} · "
                          f"Quelle: {item.get('source')} · LSB-Kontextrevision: {item.get('context_revision_id', 'unbekannt')}")
            if item.get('measurement_method') == 'beatwise_preceding_rr_median':
                origin = (f"Schlagweiser QTc-Median · n={item.get('n_pairs', 'unbekannt')} · "
                          f"vorausgehendes RR: {item.get('rr_seconds_median', 'unbekannt')} s · "
                          f"{item.get('formula', 'Formel im historischen Artefakt nicht angegeben')} · "
                          "Schlagstreuung ist kein Messfehler-KI")
            entries.append((item.get("interval", ""), f"{item.get('value_ms')} ms · {item.get('status')} · {item.get('reason', '')}", origin))
        from ..lab_pwave_summary import accepted_p_wave_summary
        p_summary = accepted_p_wave_summary(payload)
        if p_summary['status'] == 'descriptive':
            entries.append((
                'P-Dauer · weitere akzeptierte Schläge',
                f"Median {p_summary['median_ms']:.1f} ms · {p_summary['n_beats']}/{p_summary['n_accepted_cluster_beats']} im Hauptcluster",
                p_summary['reason'] + f" · Cluster {p_summary['cluster_id']}; keine klinische Ersatzmessung",
            ))
        else:
            entries.append(('P-Dauer · weitere akzeptierte Schläge', 'nicht auswertbar', p_summary['reason']))
        self._populate_lead_matrix(payload)
        from ..lab_marker_viewmodel import classical_measurement_cells
        from ..lab_axis_review import axis_review
        axis = axis_review(payload, self.case.metadata.get('age'))
        entries.append(('Frontale QRS-Achseneinordnung', axis['text'], axis['source']))
        for name, value in payload.get("classical_criteria", {}).items():
            entries.append(classical_measurement_cells(name, value))
        self.measurements.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            for col, value in enumerate(entry):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.measurements.setItem(row, col, item)
        values = payload.get("feature_values", {})
        from ..lab_marker_viewmodel import marker_row
        recorded_measurements = payload.get('feature_measurements', {})
        if not isinstance(recorded_measurements, dict):
            recorded_measurements = {}
        self.markers.setRowCount(len(values))
        for row, (name, value) in enumerate(sorted(values.items())):
            view = marker_row(name, value, recorded_measurements.get(name))
            for col, text in enumerate((name, view['value_text'], view['unit_status'])):
                self.markers.setItem(row, col, QTableWidgetItem(text))
                self.markers.item(row, col).setToolTip(view['tooltip'])
            self.markers.item(row, 0).setData(Qt.ItemDataRole.UserRole, view['search_text'])
        self.filter_markers(self.marker_search.text())
        from ..lab_marker_viewmodel import classical_rule_view
        criteria = {key: value for key, value in payload.get("classical_criteria", {}).items() if isinstance(value, dict) and value.get("exploratory") is False}
        self.suggestions.setRowCount(len(criteria))
        for row, (key, value) in enumerate(criteria.items()):
            text = [key, f"{value.get('value')} {value.get('unit', '')}", classical_rule_view(value)['status'], self.suggestion_review.get(key, {}).get("decision", "ungeprüft")]
            for col, item in enumerate(text):
                self.suggestions.setItem(row, col, QTableWidgetItem(str(item)))
            if key in self.suggestion_review and self.suggestion_review[key].get("analysis_id") != payload.get("analysis_id"):
                self.suggestions.setItem(row, 3, QTableWidgetItem("Historischer Review – anderer Analysestand"))
        self.filter_rule_suggestions(self.rule_search.text())
        self._refresh_quick_suggestions()
        self.redraw()
        return True

    def _refresh_quick_suggestions(self):
        while self.quick_suggestion_buttons.count():
            item = self.quick_suggestion_buttons.takeAt(0)
            if item.widget() is not None:
                # deleteLater() alone leaves the removed button painted at the
                # panel's origin until the next event loop pass, on top of
                # the status text.
                item.widget().hide()
                item.widget().deleteLater()
        criteria = self.payload.get('classical_criteria', {}) if isinstance(self.payload, dict) else {}
        if not isinstance(criteria, dict) or not criteria:
            self.quick_suggestion_status.setText('Automatische Prüfung: kein gespeicherter Regelstand für diesen Fall.')
            return
        from ..lab_marker_viewmodel import classical_criterion_label, classical_rule_view
        positive = [(name, rule) for name, rule in criteria.items()
                    if isinstance(rule, dict) and rule.get('positive') is True
                    and classical_rule_view(rule)['eligible']]
        exploratory = [(name, rule) for name, rule in criteria.items()
                       if isinstance(rule, dict) and rule.get('positive') is True
                       and rule.get('exploratory') is True and rule.get('evaluable') is True]
        self.quick_suggestion_status.setText(
            f'Automatische Regelprüfung · {len(positive)} beurteilbare Kriterien erfüllt'
            f' · {len(exploratory)} explorative Muster. '
            'Ein Muster ist keine Diagnose; eine kalibrierte Erkrankungswahrscheinlichkeit liegt hier nicht vor.'
        )
        for name, rule in (positive + exploratory)[:4]:
            button = QToolButton(self.quick_suggestion_panel)
            button.setText(classical_criterion_label(name) + (' · Forschung' if rule.get('exploratory') else ''))
            button.setToolTip('Gespeicherte Definition, Quellstand und aktuelles Signal nebeneinander prüfen')
            button.clicked.connect(lambda _checked=False, key=name: self.show_rule_evidence(key))
            self.quick_suggestion_buttons.addWidget(button)
        if len(positive) + len(exploratory) > 4 or not positive and not exploratory:
            button = QToolButton(self.quick_suggestion_panel)
            button.setText('Alle Regeln und Messwerte öffnen')
            button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.suggestions_tab))
            self.quick_suggestion_buttons.addWidget(button)

    def _open_selected_rule_evidence(self):
        row = self.suggestions.currentRow()
        if row >= 0 and self.suggestions.item(row, 0) is not None:
            self.show_rule_evidence(self.suggestions.item(row, 0).text())

    def show_rule_evidence(self, name):
        if self.blind.isChecked() or self.signal is None or self.reference is None:
            return
        rule = self.payload.get('classical_criteria', {}).get(name)
        if not isinstance(rule, dict):
            return
        from ..lab_marker_viewmodel import classical_rule_details
        if getattr(self, 'rule_evidence_dialog', None) is not None:
            self.rule_evidence_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle('EKG und gespeicherte Regelevidenz · ' + name)
        layout = QHBoxLayout(dialog)
        left = QVBoxLayout()
        plot = pg.PlotWidget()
        left.addWidget(plot)
        lead_names = rule.get('leads')
        documented = isinstance(lead_names, (list, tuple)) and any(item in LEADS for item in lead_names)
        lead_name = next((item for item in lead_names if item in LEADS), None) if documented else self.lead_choice.currentText()
        lead = LEADS.index(lead_name)
        fs = self.reference.sampling_rate_hz
        center = self.fiducials.get('r_peak', self.reference.sample(self.start.value() + 0.4)) / fs
        a = max(0, round((center - 0.55) * fs))
        b = min(len(self.signal), round((center + 0.75) * fs))
        plot.plot(np.arange(a, b) / fs, self.signal[a:b, lead], pen=pg.mkPen('#dfebf5' if self._dark else '#202f3b', width=1.4))
        plot.setLabel('bottom', 'Signalzeit', units='s')
        plot.setLabel('left', lead_name, units='mV')
        plot.setTitle('Aktuelles Originalsignal · Grenzen des repräsentativen Schlags')
        for key, color in (('p_onset', '#258c9a'), ('p_offset', '#258c9a'),
                           ('qrs_onset', '#c16635'), ('qrs_offset', '#c16635'),
                           ('t_onset', '#8067ac'), ('t_offset', '#8067ac')):
            if key in self.fiducials and a <= self.fiducials[key] < b:
                plot.addItem(pg.InfiniteLine(self.fiducials[key] / fs, pen=pg.mkPen(color, width=1)))
        note = QLabel('Ableitung aus gespeicherter Regel' if documented else
                      'Ableitung nicht fallbezogen gespeichert: aktuell gewählte Ableitung als Sichtkontrolle. '
                      'Diese Einzelansicht belegt das Kriterium nicht allein.')
        note.setWordWrap(True)
        left.addWidget(note)
        layout.addLayout(left, 3)
        right = QVBoxLayout()
        details = QTextEdit()
        details.setReadOnly(True)
        details.setPlainText(classical_rule_details(name, rule, self.payload.get('analysis_id')))
        right.addWidget(details)
        jump = QPushButton('Regelprüfung und manuellen Review öffnen')
        jump.clicked.connect(lambda: (self.open_catalog_case_criterion(name), dialog.close()))
        right.addWidget(jump)
        cohort_results = QPushButton('Kohorten-Ergebnisse separat öffnen')
        cohort_results.setToolTip(
            'Öffnet die Ergebnisansicht; dortige Kohortenresultate sind kein Nachweis für diesen Einzelfall.'
        )
        cohort_results.clicked.connect(lambda: (self.resultsRequested.emit(), dialog.close()))
        right.addWidget(cohort_results)
        source = next((item for item in self.criteria_catalog.entries
                       if item.get('criterion_id') == name and isinstance(item.get('source_url'), str)
                       and item['source_url'].startswith('https://')), None)
        if source:
            source_button = QPushButton('Quelle im aktuellen Register öffnen')
            source_button.setToolTip('Aktueller Registerstand; kann vom gespeicherten Analysestand abweichen.')
            source_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(source['source_url'])))
            right.addWidget(source_button)
        layout.addLayout(right, 2)
        dialog.resize(1150, 600)
        self.rule_evidence_dialog = dialog
        dialog.show()

    def filter_rule_suggestions(self, text):
        terms = str(text).casefold().split()
        for row in range(self.suggestions.rowCount()):
            item = self.suggestions.item(row, 0)
            if item is None:
                continue
            name = item.text()
            rule = self.payload.get('classical_criteria', {}).get(name, {})
            searchable = (name + ' ' + json.dumps(rule, ensure_ascii=False, default=str)).casefold()
            hidden = not all(term in searchable for term in terms)
            self.suggestions.setRowHidden(row, hidden)
            if hidden and row == self.suggestions.currentRow():
                self.suggestions.clearSelection()
                self.suggestions.setCurrentCell(-1, -1)
                self.suggestion_detail.clear()

    def clear_analysis(self):
        if getattr(self, 'rule_evidence_dialog', None) is not None:
            self.rule_evidence_dialog.close()
        self.qt_history_button.hide()
        self.device_interval_table.setRowCount(0)
        self.device_interval_table.hide()
        self.device_interval_note.setText('Keine eigene Analyse für den Gerätevergleich geladen.')
        self.qt_alert_note.hide()
        self.qt_ack_button.hide()
        self.fiducial_evidence.setRowCount(0)
        self.fiducial_overlay.clear()
        self.scientific_displays.set_data(self.signal, self.reference.sampling_rate_hz if self.reference else None)
        self.score_item_note.setText("Kein auswertbares QTc-Einzelitem; kein vollständiger LQTS-Score.")
        self.payload = {}
        self._refresh_quick_suggestions()
        self._heart_set_analysis({})
        self._vcg_set_source(self.reference.sampling_rate_hz if self.reference else 500.0)
        self.analysis_arrays = None
        self.markers.setRowCount(0)
        self.measurements.setRowCount(0)
        self._populate_lead_matrix({})
        self.suggestions.setRowCount(0)
        self.marker_detail.clear()
        self.marker_provenance_tree.clear()
        self.marker_evidence.set_evidence(None)
        self.marker_signal.clear()
        self.gallery.setRowCount(0)
        self.marker_beat.clear()
        self.marker_beat.setVisible(False)
        self.beat_plot.clear()
        self.marker_beat_audit.setRowCount(0)
        self.marker_beat_audit.setVisible(False)
        self.transform_canvas.clear()
        if not self.manual_fiducials:
            self.fiducials.clear()
        self.redraw()

    def set_analysis_arrays(self, path):
        self.analysis_arrays = None
        self.gallery.setRowCount(0)
        self.representative_plot.clear()
        self.beat_plot.clear()
        self.marker_signal.clear()
        self.marker_beat_audit.setRowCount(0)
        self.marker_beat_audit.hide()
        self.draw_transforms()
        self.scientific_displays.set_data(self.signal, self.reference.sampling_rate_hz)
        self.median_template = None
        self.template_annotations = []
        self.template_region_bounds = None
        self.template_measurement_note.setText('Neuer Signalstand: keine Template-Messungen vorhanden.')
        self.draw_representative_beat()
        with np.load(path, allow_pickle=False) as archive:
            wanted = ("diagnostic_ecg", "morphology_ecg", "robust_morphology_ecg", "record_svd_components", "sampling_rate_hz", "fiducial_names", "fiducial_samples", "segment_names", "segment_start_samples", "segment_end_samples", "r_peaks")
            arrays = {k: archive[k].copy() for k in wanted if k in archive}
        from ..lab_array_contract import validate_visualization_arrays
        fs, peaks = validate_visualization_arrays(arrays)
        self.analysis_arrays = arrays
        self._vcg_set_source(fs)
        self.draw_representative_beat()
        self.scientific_displays.set_data(self.signal, self.reference.sampling_rate_hz, np.asarray(peaks)/fs)
        inventory = {item["beat_index"]: item for item in self.payload.get("beat_inventory", [])}
        self.gallery.setRowCount(len(peaks))
        for row, sample in enumerate(peaks):
            item = inventory.get(row, {})
            texts = (str(row+1), f"{sample/fs:.6f}", str(item.get("morphology_cluster_id", "nicht dokumentiert")), json.dumps({key: item.get(key, "nicht dokumentiert") for key in ("accepted", "beat_type", "beat_type_source", "review_reasons")}, ensure_ascii=False), "unbekannt")
            for col, text in enumerate(texts):
                self.gallery.setItem(row, col, QTableWidgetItem(text))
            self.gallery.item(row, 1).setData(Qt.ItemDataRole.UserRole, float(sample/fs))
        from ..lab_beat_annotations import read_beat_labels
        labels, annotation_errors = read_beat_labels(self.store.root / 'beats', asdict(self.reference), self.payload.get('analysis_id'))
        self.beat_annotation_note.setText(
            f'{len(annotation_errors)} Annotationsdateien nicht lesbar/kompatibel; Details im Tooltip.'
            if annotation_errors else 'Manuelle Annotationen geladen; keine automatische Pipeline-Übernahme.')
        self.beat_annotation_note.setToolTip('\n'.join(annotation_errors))
        for row, sample in enumerate(peaks):
            key = round(sample/fs*self.reference.sampling_rate_hz)
            if key in labels:
                self.gallery.setItem(row, 4, QTableWidgetItem(labels[key]))
        self.draw_transforms()

    def draw_representative_beat(self, *_):
        self.refresh_template_measurement_table()
        from ..lab_representative_beat import representative_window
        self.representative_plot.clear()
        self.template_region = None
        self.representative_note.setToolTip('')
        result = (self.median_template or {'available': False, 'reason': 'Cluster wählen und Median ausdrücklich berechnen.'}) if self.representative_kind.currentIndex() == 1 else representative_window(self.analysis_arrays, self.representative_path.currentData())
        if not result['available']:
            self.representative_note.setText(result['reason'])
            return
        self.representative_note.setText(result['note'] + f" {result['sampling_rate_hz']:g} Hz · natives aVR, keine Anzeigeinversion. Änderungen manueller Grenzen erscheinen erst nach lokaler Neuberechnung.")
        if 'used_beats' in result:
            self.representative_note.setToolTip(json.dumps({key: result[key] for key in
                ('used_beats', 'excluded_beats', 'method', 'cluster', 'path')}, ensure_ascii=False, indent=2))
        selected = self.representative_lead.currentIndex()
        leads = range(12) if selected == 0 else [selected-1]
        dark = getattr(self, '_dark', False)
        text_color = '#dce5ef' if dark else '#263746'
        colors = ('#56b4e9', '#e69f00', '#009e73', '#f0e442', '#cc79a7', '#d55e00') if dark else ('#0072b2', '#985a00', '#007a59', '#695800', '#965780', '#b54300')
        for lead in leads:
            pen = pg.mkPen(colors[lead % 6], width=1.3, style=Qt.PenStyle.SolidLine if lead < 6 else Qt.PenStyle.DashLine)
            self.representative_plot.plot(result['time_ms'], result['signal'][:, lead], name=LEADS[lead], pen=pen)
        for _, label in self.representative_legend.items:
            label.setText(label.text, color=text_color)
        for index, (name, time) in enumerate(sorted(result['fiducials_ms'].items(), key=lambda item: item[1])):
            line = pg.InfiniteLine(time, angle=90, movable=False, pen=pg.mkPen('#888888', style=Qt.PenStyle.DotLine), label=name,
                                   labelOpts={'color': text_color, 'position': .92-(index % 4)*.075})
            self.representative_plot.addItem(line)
        self.representative_plot.enableAutoRange()
        if self.representative_kind.currentIndex() == 1 and self.template_region_bounds is not None:
            self.template_region = pg.LinearRegionItem(self.template_region_bounds,
                bounds=(float(result['time_ms'][0]), float(result['time_ms'][-1])),
                brush=pg.mkBrush(80, 130, 180, 35), movable=True)
            self.representative_plot.addItem(self.template_region)
            self.template_region.sigRegionChangeFinished.connect(self.template_region_changed)

    def invalidate_median_template(self, *_):
        self.median_template = None
        self.template_annotations = []
        self.template_region_bounds = None
        self.template_measurement_note.setText('Keine Template-Messungen geladen. Gespeicherte Messungen erscheinen beim Wiederherstellen des passenden Templates.')
        self.draw_representative_beat()

    def build_median_template(self):
        from ..lab_representative_beat import cluster_median
        self.template_annotations = []
        self.template_region_bounds = None
        self.template_measurement_note.setText('Neues Template: keine Messungen vorgemerkt. Originalsignal-Messungen bleiben unverändert.')
        self.median_template = cluster_median(self.analysis_arrays, self.payload.get('beat_inventory'),
                                            self.template_cluster.currentData(), self.representative_path.currentData())
        self.representative_kind.setCurrentIndex(1)
        self.draw_representative_beat()
        if self.median_template.get('available'):
            self.representative_note.setToolTip(json.dumps({key: self.median_template[key] for key in
                ('used_beats', 'excluded_beats', 'method', 'cluster', 'path')}, ensure_ascii=False, indent=2))
            if self.payload.get('analysis_id') and self.reference:
                try:
                    self.template_annotations = self._template_measurement_store().active()
                    self.refresh_template_measurement_table()
                    self.template_measurement_note.setText(f'{len(self.template_annotations)} lokal gespeicherte Messungen für dieses exakte Template geladen.')
                except (ValueError, OSError) as exc:
                    logging.getLogger(__name__).exception('Template measurement history cannot be loaded')
                    self.template_measurement_note.setText(f'Messverlauf nicht lesbar – keine Freigabe der Messhistorie: {exc}')

    def _template_measurement_store(self):
        from ..lab_template_store import TemplateMeasurementStore
        return TemplateMeasurementStore(self.store.root, self.median_template, asdict(self.reference), self.payload.get('analysis_id'))

    def export_median_template(self):
        from ..lab_representative_beat import median_export
        from ...mimic.storage import atomic_write_json
        try:
            document = median_export(self.median_template, asdict(self.reference) if self.reference else {},
                                     self.payload.get('analysis_id'))
            if any(item['template_sha256'] != document['template_sha256'] for item in self.template_annotations):
                raise ValueError('Template-Messungen gehören zu einem anderen Template.')
            document['manual_measurements'] = list(self.template_annotations)
            history = self._template_measurement_store()
            document['manual_measurements'] = history.active()
            document['measurement_history'] = history.events()
            path, _ = QFileDialog.getSaveFileName(self, 'Exploratives Median-Template exportieren',
                str(self.store.root / 'exports' / 'median_template.json'), 'JSON (*.json)')
            if not path:
                return
            if Path(path).suffix.lower() != '.json':
                raise ValueError('Für diesen Export ist die Dateiendung .json erforderlich.')
            atomic_write_json(Path(path), document)
        except (ValueError, OSError) as exc:
            logging.getLogger(__name__).warning('Median export failed: %s', exc)
            QMessageBox.warning(self, 'Median-Export nicht möglich', str(exc))

    def add_template_measurement(self):
        if self.blind.isChecked() or not self.median_template or not self.median_template.get('available'):
            return
        times = self.median_template['time_ms']
        self.template_region_bounds = (float(times[len(times)//3]), float(times[2*len(times)//3]))
        self.representative_kind.setCurrentIndex(1)
        self.draw_representative_beat()
        self.template_measurement_note.setText('Grenzen manuell verschieben. Anfangspositionen sind keine automatisch erkannten Wellengrenzen.')

    def template_region_changed(self):
        if self.template_region is not None:
            self.template_region_bounds = tuple(self.template_region.getRegion())
            self.template_measurement_note.setText('Manueller Bereich geändert; noch nicht gespeichert. Beim Speichern wird auf Samples eingerastet.')

    def save_template_measurement(self):
        if self.blind.isChecked() or self.template_region is None:
            return
        from ..lab_representative_beat import manual_template_interval
        try:
            start, end = self.template_region.getRegion()
            scope = '12-Kanal-Überlagerung' if self.representative_lead.currentIndex() == 0 else self.representative_lead.currentText()
            item = manual_template_interval(self.median_template, asdict(self.reference), self.payload.get('analysis_id'),
                start, end, self.template_interval_kind.currentText(), self.author.text(), scope)
            self.template_annotations = self._template_measurement_store().add(item)
            self.template_region_bounds = (item['start_ms'], item['end_ms'])
            self.draw_representative_beat()
            self.template_measurement_note.setText(f"{item['label']}: {item['duration_ms']:.6g} ms · {scope} · {len(self.template_annotations)} lokal gespeichert. Nur manuelle Template-Messung.")
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, 'Template-Messung nicht möglich', str(exc))

    def withdraw_template_measurement(self):
        if self.blind.isChecked() or not self.template_annotations:
            return
        row = self.template_measurement_table.currentRow()
        if row < 0:
            return
        event_id = self.template_measurement_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        reason, ok = QInputDialog.getText(self, 'Ausgewählte Template-Messung zurücknehmen', 'Begründung (Messung bleibt im Verlauf erhalten):')
        if not ok:
            return
        try:
            self.template_annotations = self._template_measurement_store().withdraw(
                event_id, self.author.text(), reason)
            self.refresh_template_measurement_table()
            self.template_measurement_note.setText(f'{len(self.template_annotations)} aktive lokale Messungen. Rücknahme im Verlauf protokolliert.')
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Rücknahme nicht möglich', str(exc))

    def refresh_template_measurement_table(self):
        table = self.template_measurement_table
        old = table.item(table.currentRow(), 0)
        selected = old.data(Qt.ItemDataRole.UserRole) if old else None
        table.setRowCount(len(self.template_annotations))
        for row, item in enumerate(self.template_annotations):
            for col, key in enumerate(('label', 'duration_ms', 'lead_scope', 'author', 'created_at')):
                cell = QTableWidgetItem(str(item[key]))
                cell.setData(Qt.ItemDataRole.UserRole, item['local_event_id'])
                cell.setToolTip(json.dumps(item, ensure_ascii=False, indent=2))
                table.setItem(row, col, cell)
            if item['local_event_id'] == selected:
                table.selectRow(row)

    def show_template_measurement(self, row, _column):
        if self.blind.isChecked() or row < 0:
            return
        cell = self.template_measurement_table.item(row, 0)
        if cell is None:
            return
        event_id = cell.data(Qt.ItemDataRole.UserRole)
        item = next((item for item in self.template_annotations if item['local_event_id'] == event_id), None)
        if item is None:
            return
        self.template_region_bounds = (item['start_ms'], item['end_ms'])
        self.representative_kind.setCurrentIndex(1)
        scope = item['lead_scope']
        self.representative_lead.setCurrentIndex(0 if scope == '12-Kanal-Überlagerung' else LEADS.index(scope)+1)
        self.draw_representative_beat()
        self.template_measurement_note.setText('Gespeicherte manuelle Grenzen angezeigt. Verschieben ändert den gespeicherten Eintrag nicht; erneut speichern erzeugt eine neue Messung.')

    def select_beat(self):
        row = self.gallery.currentRow()
        if row < 0 or self.gallery.item(row, 1) is None:
            return
        seconds = self.gallery.item(row, 1).data(Qt.ItemDataRole.UserRole)
        if seconds is None:
            return
        self.start.setValue(max(0, seconds-.3))
        self.timeSelected.emit(seconds)
        fs = self.reference.sampling_rate_hz
        a, b = max(0, round((seconds-.3)*fs)), min(len(self.signal), round((seconds+.7)*fs))
        self.beat_plot.clear()
        lead = LEADS.index(self.lead_choice.currentText())
        self.beat_plot.plot(np.arange(a, b)/fs-seconds, self.signal[a:b, lead], pen="#258c9a")
        self.beat_plot.setTitle(f"{self.lead_choice.currentText()} · Schlag {row+1} · Originalsignal")

    def label_beat(self):
        row = self.gallery.currentRow()
        if row < 0:
            return
        if not self.author.text().strip():
            QMessageBox.information(self, 'Bearbeiter erforderlich', 'Bitte vor einer Schlagannotation den Bearbeiter eintragen.')
            return
        seconds = self.gallery.item(row, 1).data(Qt.ItemDataRole.UserRole)
        if seconds is None:
            self.beat_annotation_note.setText('Keine präzise Schlagzeit verfügbar; Annotation nicht gespeichert.')
            return
        label, ok = QInputDialog.getItem(self, "Manuelle Schlagannotation", "Experteneinschätzung (keine automatische Pipeline-Übernahme):", ["unbekannt", "Sinusschlag", "atriale Extrasystole", "ventrikuläre Extrasystole", "stimuliert", "Artefakt", "sonstige"], editable=False)
        if not ok:
            return
        reason, ok = QInputDialog.getText(self, "Begründung", "Beobachtung / Referenz:")
        if not ok or not reason.strip():
            return
        from ...mimic.storage import atomic_write_json
        target = self.store.root / "beats" / f"{uuid.uuid4().hex}.json"
        try:
            native_sample = self.reference.sample(seconds)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target, {"schema_version": "1.1.0", "signal": asdict(self.reference), "analysis_id": self.payload.get("analysis_id", ""), "r_time_seconds": seconds, "native_r_sample": native_sample, "label": label, "reason": reason, "author": self.author.text().strip(), "created_at": datetime.now(timezone.utc).isoformat(), "clinical_claim_allowed": False})
        except (OSError, ValueError) as exc:
            self.beat_annotation_note.setText('Schlagannotation nicht gespeichert: ' + str(exc))
            return
        self.gallery.setItem(row, 4, QTableWidgetItem(label))
        self.beat_annotation_note.setText('Schlagannotation als neue lokale Revision gespeichert; Originalsignal unverändert.')

    def draw_transforms(self, *_):
        self.transform_canvas.clear()
        arrays = self.analysis_arrays
        if not arrays or "morphology_ecg" not in arrays:
            self.transform_note.setText("Keine kompatiblen verarbeiteten Signale vorhanden.")
            return
        from ...core.vcg_transform import VCGTransformRegistry
        signal = arrays["morphology_ecg"]
        bounds = dict(zip(arrays.get("fiducial_names", []), arrays.get("fiducial_samples", [])))
        segment = self.transform_segment.currentText()
        keys = {"P": ("p_onset", "p_offset"), "QRS": ("qrs_onset", "qrs_offset"), "T": ("t_onset", "t_offset")}
        a, b = 0, len(signal)
        if segment in keys:
            left, right = keys[segment]
            if left not in bounds or right not in bounds:
                self.transform_note.setText("Segment nicht abgegrenzt; keine Ersatzgrenzen erzeugt.")
                return
            a, b = int(bounds[left]), int(bounds[right])+1
        methods = []
        for method in ("kors", "inverse_dower"):
            methods.append((method, VCGTransformRegistry.transform(signal, method).xyz[a:b]))
        if "record_svd_components" in arrays:
            methods.append(("SVD – keine Anatomie", arrays["record_svd_components"][a:b]))
        for row, (method, points) in enumerate(methods):
            for col, (x, y, title) in enumerate(((0, 1, "XY / frontal"), (0, 2, "XZ / horizontal"), (1, 2, "YZ / sagittal"))):
                plot = self.transform_canvas.addPlot(row=row, col=col)
                plot.setTitle(f"{method} · {title if row < 2 else f'Komponenten {x+1}/{y+1}'}", size="9pt")
                plot.setAspectLocked(True)
                plot.showGrid(x=True, y=True, alpha=.15)
                colors = pg.colormap.get("viridis").getLookupTable(nPts=16, alpha=True)
                edges = np.linspace(0, len(points), 17, dtype=int)
                for chunk, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                    if right > left:
                        section = points[max(0, left-1):right]
                        plot.plot(section[:, x], section[:, y], pen=pg.mkPen(tuple(int(c) for c in colors[chunk]), width=1.3))
                if len(points):
                    plot.plot([points[0, x]], [points[0, y]], pen=None, symbol="o", symbolBrush="#258c9a")
                    plot.plot([points[-1, x]], [points[-1, y]], pen=None, symbol="t", symbolBrush="#c16635")
        self.transform_note.setText("Identisches verarbeitetes EKG und Zeitfenster für alle Methoden. Zeitfarbe: violett (früh) → gelb (spät). Kreis: Beginn; Dreieck: Ende. Gesamtrecord enthält mehrere Schläge. Ein Einzelfall liefert keine populationsweiten Bland–Altman-Grenzen.\n" + json.dumps(self.payload.get("qrs_t_method_comparison", {}), ensure_ascii=False, default=str)[:1800])

    def _lead_order_changed(self, *_):
        cabrera = self.lead_order.currentIndex() == 1
        self.avr_polarity.blockSignals(True)
        self.avr_polarity.setCurrentIndex(1 if cabrera else self._standard_avr_polarity)
        self.avr_polarity.blockSignals(False)
        self.avr_polarity.setEnabled(not cabrera)
        self.redraw()

    def _avr_polarity_changed(self, *_):
        if self.lead_order.currentIndex() == 0:
            self._standard_avr_polarity = self.avr_polarity.currentIndex()
        self.redraw()

    def redraw(self, *_):
        if self.signal is None:
            return
        if self.prior is not None:
            self._comparison_analysis_changed()
        # Retain scientific coordinates when changing display layout.
        if self.calipers:
            self.saved_measurements = self.measurement_rows()
        self.calipers = []
        self._caliper_ids = []
        self.canvas.clear()
        self.plots = []
        self._landmark_items = []
        layout = self.layout_choice.currentIndex()
        fs = self.reference.sampling_rate_hz
        start = self.start.value()
        duration = min(float(self.view_duration.currentData() or 10.0), len(self.signal) / fs - start)
        window = self.signal[max(0, round(start*fs)):min(len(self.signal), round((start+duration)*fs))]
        finite = np.abs(window[np.isfinite(window)])
        self._display_limit = max(1.1, float(np.max(finite))*1.05) if finite.size else 1.1
        columns = 4 if layout == 0 else 2 if layout == 1 else 1
        sequence = [4, 0, 3, 1, 5, 2, 6, 7, 8, 9, 10, 11] if self.lead_order.currentIndex() == 1 else list(range(12))
        # Fill each column top-to-bottom: limb leads first, then V1–V6.
        grid = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11] if layout == 0 else [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11] if layout == 1 else list(range(12))
        order = [sequence[index] for index in grid]
        if layout == 3:
            order = [LEADS.index(self.lead_choice.currentText())]
        for pos, lead in enumerate(order):
            column = pos % columns
            offset = column * duration / columns if columns > 1 else 0
            self._plot_lead(pos // columns, column, lead, start + offset, duration / columns)
        if layout == 0:
            self._plot_lead(3, 0, LEADS.index(self.lead_choice.currentText()), start, duration, span=4)
        # Plot titles, axes and the minimum view-box height require more than
        # 60 px per row. A smaller fixed canvas silently clips inferior rows.
        rows = 12 if layout == 2 else 6 if layout == 1 else 4 if layout == 0 else 1
        self.canvas.setFixedHeight(max(360, rows * 100))
        self._sync_viewer_strip_height()
        for row in list(self.saved_measurements):
            self._restore_caliper(row)
        if self.show_landmarks.isChecked() and not self.blind.isChecked() and self.plots:
            for name, sample in self.fiducials.items():
                time = sample / fs
                for plot, lead, lo, hi in self.plots:
                    if lo <= time <= hi:
                        label = name + (" · man." if self.manual_fiducials else " · auto")
                        line = pg.InfiniteLine(time, movable=True, pen=pg.mkPen("#b87915", style=Qt.PenStyle.DashLine), label=label if name == self.landmark_choice.currentText() else None)
                        line.setBounds([0, (len(self.signal)-1)/fs])
                        line.sigPositionChangeFinished.connect(lambda item, key=name: self._move_landmark(key, item.value()))
                        plot.addItem(line)
                        self._landmark_items.append(line)
        if self.mode.currentIndex() == 1 and not self.blind.isChecked():
            from scipy.signal import welch
            lead = LEADS.index(self.lead_choice.currentText())
            f, p = welch(self.signal[:, lead], fs=fs, nperseg=min(2048, len(self.signal)))
            self.diagnostics.clear()
            self.diagnostics.plot(f, p, pen="#258c9a")

    def _plot_lead(self, row, col, lead, start, duration, span=1):
        fs = self.reference.sampling_rate_hz
        plot = self.canvas.addPlot(row=row, col=col, colspan=span)
        display_sign = -1 if lead == 3 and (self.lead_order.currentIndex() == 1 or self.avr_polarity.currentIndex() == 1) else 1
        plot._native_amplitude_sign = display_sign
        plot.setTitle(f"{LEADS[lead]} · {start:.2f}–{start+duration:.2f} s", size="9pt")
        if display_sign == -1:
            plot.setTitle(f"−aVR · {start:.2f}–{start+duration:.2f} s", size="9pt")
            plot.setToolTip('Anzeige invertiert; Messungen beziehen sich weiterhin auf natives aVR. Keine Änderung am Originalsignal.')
        plot.showGrid(x=True, y=True, alpha=.17)
        plot.setLabel("bottom", "s")
        plot.setLabel("left", "mV")
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot.getAxis("left").setStyle(tickFont=QFont("Arial", 8))
        plot.getAxis("bottom").setStyle(tickFont=QFont("Arial", 8))
        mode = self.layout_choice.currentIndex()
        last_row = 3 if mode == 0 else 5 if mode == 1 else 11 if mode == 2 else 0
        if row != last_row:
            plot.hideAxis("bottom")
        plot.vb.setMinimumHeight(40)
        a = max(0, round(start * fs))
        b = min(len(self.signal), round((start + duration) * fs))
        values = self.signal[a:b, lead]
        finite = np.abs(values[np.isfinite(values)])
        local_limit = max(1.1, float(np.max(finite)) * 1.05) if finite.size else 1.1
        # Ten-second windows are already bounded. View-dependent downsampling
        # can lose narrow spikes and triggers a pyqtgraph reparenting failure.
        plot.plot(np.arange(a, b) / fs, display_sign*self.signal[a:b, lead], pen=pg.mkPen("#dfebf5" if self._dark else "#202f3b", width=1))
        # Explicit calibration, outside recorded samples, never used for measurements.
        plot.plot([start-.28, start-.25, start-.25, start-.05, start-.05, start-.02], [0, 0, 1, 1, 0, 0], pen="#258c9a")
        plot.setXRange(start-.3, start+max(.1, duration), padding=0)
        if self.prior is not None:
            offset = self.prior_offset.value()
            pa = max(0, round((start-offset)*self.prior_fs))
            pb = min(len(self.prior), round((start+duration-offset)*self.prior_fs))
            if pb > pa:
                prior_values = self.prior[pa:pb, lead]
                prior_finite = np.abs(prior_values[np.isfinite(prior_values)])
                if prior_finite.size:
                    local_limit = max(local_limit, float(np.max(prior_finite)) * 1.05)
        plot._observed_limit = local_limit
        plot.setYRange(-local_limit, local_limit, padding=0)
        plot.vb.setMouseEnabled(x=False, y=False)
        if not self.fit.isChecked():
            plot.vb.sigResized.connect(lambda *_args, p=plot, s=start: self._metric_scale(p, s))
            self._metric_scale(plot, start)
        if self.prior is not None:
            offset = self.prior_offset.value()
            pa = max(0, round((start-offset)*self.prior_fs))
            pb = min(len(self.prior), round((start+duration-offset)*self.prior_fs))
            if pb > pa:
                plot.plot(np.arange(pa, pb)/self.prior_fs+offset, display_sign*self.prior[pa:pb, lead], pen=pg.mkPen("#c16635", width=1))
        self.plots.append((plot, lead, start, start+duration))

    def _metric_scale(self, plot, start):
        width, height = plot.vb.width(), plot.vb.height()
        if width <= 0 or height <= 0:
            return
        plot.setXRange(start-.3, start-.3+width/(self.pixels_per_mm.value()*float(self.speed.currentText())), padding=0)
        nominal_half = height/(2*self.pixels_per_mm.value()*float(self.gain.currentText()))
        half = max(nominal_half, getattr(plot, '_observed_limit', 1.1))
        if half > nominal_half:
            plot.setToolTip('Ausschlag vollständig sichtbar: dieser Streifen verwendet vertikale Autoanpassung. Die gewählte mm/mV-Skalierung gilt hier nicht.')
        plot.setYRange(-half, half, padding=0)

    def _scene_clicked(self, event):
        for plot, *_ in self.plots:
            if plot.sceneBoundingRect().contains(event.scenePos()):
                self._clicked(event, plot)
                break

    def _clicked(self, event, plot):
        if plot.sceneBoundingRect().contains(event.scenePos()):
            point = plot.vb.mapSceneToView(event.scenePos())
            if 0 <= point.x() <= self.landmark_time.maximum():
                self.landmark_time.setValue(point.x())
                self.timeSelected.emit(point.x())

    def add_caliper(self, rhythm=False):
        if not self.plots:
            return
        target = next((item for item in self.plots if LEADS[item[1]] == self.lead_choice.currentText()), self.plots[0])
        plot, lead, lo, hi = target
        end = min(hi, (self.reference.sample_count-1)/self.reference.sampling_rate_hz)
        if end <= lo:
            return
        self._create_caliper(plot, lead, lo+(end-lo)*.25, lo+(end-lo)*.65, 0, .5, rhythm)

    def _create_caliper(self, plot, lead, t0, t1, y0, y1, rhythm, identifier=None):
        sign = getattr(plot, '_native_amplitude_sign', 1)
        y0, y1 = sign*y0, sign*y1
        lines = [pg.InfiniteLine(t0, movable=True, pen="#258c9a"), pg.InfiniteLine(t1, movable=True, pen="#258c9a"), pg.InfiniteLine(y0, angle=0, movable=True, pen="#c16635"), pg.InfiniteLine(y1, angle=0, movable=True, pen="#c16635")]
        for line in lines[2:]:
            line.setProperty('native_amplitude_sign', sign)
        for line in lines[:2]:
            line.setBounds([0, (self.reference.sample_count-1)/self.reference.sampling_rate_hz])
        repeat = []
        if rhythm:
            for _ in range(8):
                line = pg.InfiniteLine(0, pen=pg.mkPen("#258c9a", style=Qt.PenStyle.DotLine))
                plot.addItem(line)
                repeat.append(line)
        state = {'times': (t0, t1)}
        for index, line in enumerate(lines):
            plot.addItem(line)
            if index < 2:
                line.sigPositionChanged.connect(lambda *_args, i=index, ls=lines, s=state: self._caliper_time_changed(ls, i, s))
            else:
                line.sigPositionChanged.connect(self._caliper_changed)
        self.calipers.append((lead, lines, repeat))
        self._caliper_ids.append(identifier or uuid.uuid4().hex)
        self._caliper_changed()

    def _caliper_time_changed(self, lines, index, state):
        previous = state['times']
        if self.caliper_march.isChecked():
            delta = lines[index].value() - previous[index]
            maximum = (self.reference.sample_count - 1) / self.reference.sampling_rate_hz
            delta = max(-min(previous), min(delta, maximum - max(previous)))
            for line, value in zip(lines[:2], previous):
                was_blocked = line.blockSignals(True)
                try:
                    line.setValue(value + delta)
                finally:
                    line.blockSignals(was_blocked)
        state['times'] = tuple(line.value() for line in lines[:2])
        self._caliper_changed()

    def _restore_caliper(self, row):
        target = next((p for p in self.plots if LEADS[p[1]] == row["lead"] and p[2] <= row["start_sample"]/self.reference.sampling_rate_hz <= p[3]), None)
        if target:
            fs = self.reference.sampling_rate_hz
            self._create_caliper(target[0], target[1], row["start_sample"]/fs, row["end_sample"]/fs, row["amplitude_start_mv"], row["amplitude_end_mv"], row.get("method") == "manual_rr_caliper", row.get("measurement_id"))

    def measurement_rows(self):
        rows = {r.get("measurement_id", str(i)): r for i, r in enumerate(self.saved_measurements)}
        fs = self.reference.sampling_rate_hz
        for identifier, (lead, lines, repeat) in zip(self._caliper_ids, self.calipers):
            a, b = sorted((self.reference.sample(lines[0].value()), self.reference.sample(lines[1].value())))
            sign = lines[2].property('native_amplitude_sign') or 1
            item = MeasurementAnnotation(self.reference, LEADS[lead], a, b, sign*lines[2].value(), sign*lines[3].value(), method="manual_rr_caliper" if repeat else "manual_caliper", author=self.author.text())
            data = asdict(item)
            data.update(duration_ms=item.duration_ms, delta_mv=item.delta_mv)
            data["measurement_id"] = identifier
            rows[identifier] = data
            for i, line in enumerate(repeat):
                line.setValue((b+(i+1)*(b-a))/fs)
        return list(rows.values())

    def _caliper_changed(self, *_):
        rows = self.measurement_rows()
        self.measurement_status.setText(" | ".join(f"{r['lead']}: Δt {r['duration_ms']:.2f} ms · ΔU {r['delta_mv']:.3f} mV" for r in rows) or "Keine Messzirkel")

    def clear_calipers(self):
        self.calipers = []
        self._caliper_ids = []
        self.saved_measurements = []
        self.redraw()
        self.measurement_status.setText("Keine Messzirkel")

    def _move_landmark(self, name, time):
        self.manual_fiducials = True
        self.history.append(dict(self.fiducials))
        self.redo_history.clear()
        self.fiducials[name] = self.reference.sample(time)
        self.redraw()

    def show_landmark_editor(self):
        """Inspect the fused representative boundaries at native signal resolution."""
        if self.signal is None or self.reference is None:
            return
        self._draw_landmark_closeup()
        self.landmark_editor.show()
        self.landmark_editor.raise_()

    def _draw_landmark_closeup(self):
        if self.signal is None or self.reference is None:
            return
        plot = self.landmark_closeup
        plot.clear()
        fs = self.reference.sampling_rate_hz
        selected = self.landmark_choice.currentText()
        center = (self.fiducials[selected] / fs if selected in self.fiducials
                  else self.fiducials.get('r_peak', self.reference.sample(self.start.value() + 0.25)) / fs)
        lo = max(0.0, center - 0.45)
        hi = min((len(self.signal) - 1) / fs, center + 0.55)
        a, b = max(0, int(lo * fs)), min(len(self.signal), int(hi * fs) + 1)
        lead = LEADS.index(self.lead_choice.currentText())
        signal = self.signal[a:b, lead]
        plot.plot(np.arange(a, b) / fs, signal, pen=pg.mkPen('#dfebf5' if self._dark else '#202f3b', width=1.5))
        for start_name, end_name, color in (
            ('p_onset', 'p_offset', '#258c9a'),
            ('qrs_onset', 'qrs_offset', '#c16635'),
            ('t_onset', 't_offset', '#8067ac'),
        ):
            if start_name in self.fiducials and end_name in self.fiducials:
                left, right = self.fiducials[start_name] / fs, self.fiducials[end_name] / fs
                if left < right and right >= lo and left <= hi:
                    region = pg.LinearRegionItem((left, right), movable=False, brush=pg.mkBrush(color + '33'), pen=pg.mkPen(color))
                    region.setZValue(-1)
                    plot.addItem(region)
        if not self.blind.isChecked():
            for name, sample in self.fiducials.items():
                time = sample / fs
                if lo <= time <= hi and name in LANDMARKS:
                    line = pg.InfiniteLine(time, movable=True, pen=pg.mkPen('#c16635', width=1.2),
                                           label=name if name == selected else None)
                    line.setBounds((0, (len(self.signal) - 1) / fs))
                    line.sigPositionChangeFinished.connect(lambda item, key=name: self._closeup_landmark_moved(key, item.value()))
                    plot.addItem(line)
        plot.setXRange(lo, hi, padding=0)
        finite = np.abs(signal[np.isfinite(signal)])
        limit = max(0.2, float(np.max(finite)) * 1.1) if finite.size else 0.2
        plot.setYRange(-limit, limit, padding=0)
        plot.setLabel('bottom', 'Signalzeit', units='s')
        plot.setLabel('left', self.lead_choice.currentText(), units='mV')
        plot.setTitle('Fusionsgrenzen · lokaler Review · Änderungen erst nach Speichern/Anwenden wirksam')

    def _closeup_landmark_moved(self, name, time):
        self._move_landmark(name, time)
        self._draw_landmark_closeup()

    def set_landmark(self):
        if self.reference:
            self._move_landmark(self.landmark_choice.currentText(), self.landmark_time.value())

    def remove_landmark(self):
        self.manual_fiducials = True
        self.history.append(dict(self.fiducials))
        self.redo_history.clear()
        self.fiducials.pop(self.landmark_choice.currentText(), None)
        self.redraw()

    def undo(self):
        if self.history:
            self.redo_history.append(dict(self.fiducials))
            self.fiducials = self.history.pop()
            self.redraw()

    def redo(self):
        if self.redo_history:
            self.history.append(dict(self.fiducials))
            self.fiducials = self.redo_history.pop()
            self.redraw()

    def save_annotations(self):
        if self.reference:
            rows = self.measurement_rows() or self.saved_measurements
            path = self.store.save_annotations(self.reference, rows, self.fiducials, self.author.text())
            self.measurement_status.setText(f"Lokale Revision gespeichert: {path.name}")

    def apply_landmarks(self):
        if not self.reference or not self.fiducials:
            return
        ordered = [self.fiducials[key] for key in LANDMARKS if key in self.fiducials]
        if any(b < a for a, b in zip(ordered, ordered[1:])):
            QMessageBox.warning(self, "Landmarken", "P/QRS/T-Grenzen müssen zeitlich geordnet sein.")
            return
        self.save_annotations()
        self.applyFiducials.emit(dict(self.fiducials), self.reference.sampling_rate_hz, self.author.text())

    def navigate(self, direction):
        if not self.reference:
            return
        fs = self.payload.get("lab_processing", {}).get("analysis_sampling_rate_hz", 500)
        segments = self.payload.get("delineation", {}).get("accepted_beat_wave_segments", [])
        times = sorted({float(s["r_peak_sample"])/fs for s in segments if s.get("r_peak_sample") is not None})
        current = self.start.value()
        candidates = [t for t in times if t > current+.01] if direction > 0 else [t for t in times if t < current-.01]
        target = (min(candidates) if direction > 0 else max(candidates)) if candidates else current+direction
        self.start.setValue(max(0, min(self.start.maximum(), target)))

    def _populate_lead_matrix(self, payload):
        self.refresh_measurement_variables(payload)
        if hasattr(self, 'suggestion_detail'):
            self.suggestion_detail.clear()
        if hasattr(self, 'rhythm_review_note'):
            from ..lab_rhythm_review import rhythm_review
            rhythm = rhythm_review(payload)
            self.rhythm_review_note.setText(rhythm['text'])
            self.rhythm_review_note.setToolTip(json.dumps(rhythm, ensure_ascii=False, indent=2))
        if hasattr(self, 'representative_plot') and not payload:
            self.median_template = None
            self.template_annotations = []
            self.refresh_template_measurement_table()
            self.template_region_bounds = None
            self.template_region = None
            self.template_measurement_note.setText('Keine Template-Messung vorhanden.')
            self.representative_plot.clear()
            self.representative_note.setToolTip('')
            self.representative_note.setText('Keine gespeicherte Schlagansicht geladen.')
        if hasattr(self, 'endpoint_review'):
            from ..lab_endpoint_review import endpoint_reviews
            reviews = endpoint_reviews(payload)
            self._endpoint_reviews = reviews
            self.endpoint_review_status.setText(f'{len(reviews)} Endpunkte mit gespeicherten Kriterien. Einzelregeln und Evidenz im Tooltip.' if reviews else 'Keine endpunktbezogenen Forschungskriterien im geladenen Analysestand; keine negative Befundung.')
            self.endpoint_review.setRowCount(len(reviews))
            for index, review in enumerate(reviews):
                for column, key in enumerate(('name', 'classical_text', 'research_text', 'comparison')):
                    item = QTableWidgetItem(str(review[key]))
                    item.setToolTip(review['details'])
                    item.setData(Qt.ItemDataRole.UserRole, review['endpoint_id'])
                    self.endpoint_review.setItem(index, column, item)
            self.endpoint_review.resizeRowsToContents()
            if reviews:
                self.endpoint_review.selectRow(0)
            else:
                self.endpoint_card.set_review(None, [])
        from ..lab_lead_matrix import recorded_lead_rows, st_j_review
        for row, entry in enumerate(recorded_lead_rows(payload)):
            for column, cell in enumerate(entry['cells']):
                item = QTableWidgetItem(cell['text'])
                item.setToolTip(cell['tooltip'])
                item.setData(Qt.ItemDataRole.UserRole, cell['key'])
                self.lead_measurement_table.setItem(row, column, item)
        self.lead_measurement_table.resizeColumnsToContents()
        review = st_j_review(payload)
        self.st_j_review_note.setText(review['text'])
        self.st_j_review_note.setToolTip('Hintergrund: ESC 2023 ACS, EKG-Befunde.\n' + review['source'] +
            '\nST-J ist hier eine gespeicherte Messung, keine bestätigte ST-Streckenmorphologie. Fehlend ist nicht normal.')

    def _endpoint_selection_changed(self):
        rows = self.endpoint_review.selectionModel().selectedRows() if self.endpoint_review.selectionModel() else []
        index = rows[0].row() if rows else -1
        reviews = getattr(self, '_endpoint_reviews', [])
        review = reviews[index] if 0 <= index < len(reviews) else None
        payload = self.payload if isinstance(self.payload, dict) else {}
        findings = payload.get('research_findings', [])
        rules = payload.get('classical_criteria', {})
        markers = payload.get('feature_values', {})
        listed_markers = {item.text() for row in range(self.markers.rowCount())
                          if (item := self.markers.item(row, 0)) is not None}
        self.endpoint_card.set_review(
            review, findings if isinstance(findings, list) else [],
            available_rules=set(rules) if isinstance(rules, dict) and self.signal is not None and self.reference is not None else set(),
            available_markers=(set(markers) & listed_markers) if isinstance(markers, dict) else set(),
        )

    def _open_endpoint_rule(self, criterion_id):
        if self.blind.isChecked():
            return
        rules = self.payload.get('classical_criteria', {})
        if isinstance(rules, dict) and criterion_id in rules:
            self.show_rule_evidence(criterion_id)

    def _open_endpoint_marker(self, feature_id):
        markers = self.payload.get('feature_values', {})
        if self.blind.isChecked() or not isinstance(markers, dict) or feature_id not in markers:
            return
        self.marker_search.clear()
        self.only_favorites.setChecked(False)
        self.marker_group.setCurrentIndex(0)
        for row in range(self.markers.rowCount()):
            item = self.markers.item(row, 0)
            if item is not None and item.text() == feature_id:
                self.tabs.setCurrentIndex(self.research_tab)
                self.markers.selectRow(row)
                self.markers.scrollToItem(item)
                return

    def _select_measured_lead(self, row, column):
        if 0 <= row < len(LEADS):
            self.lead_choice.setCurrentText(LEADS[row])

    def refresh_measurement_variables(self, payload):
        from ..lab_phrase_library import MEASUREMENT_VARIABLES, render_measurement_phrase
        selected = self.phrase_variable.currentData()
        self.phrase_variable.clear()
        blind = self.blind.isChecked()
        for key, name in MEASUREMENT_VARIABLES.items():
            if blind:
                label = name + ' · im Blindmodus verborgen'
            else:
                result = render_measurement_phrase('{' + key + '}', payload)
                label = name + ' · ' + result['text'] + ' ms'
            self.phrase_variable.addItem(label, key)
        index = self.phrase_variable.findData(selected)
        if index >= 0:
            self.phrase_variable.setCurrentIndex(index)
        self.phrase_variable_insert.setEnabled(not blind)

    def insert_measurement_variable(self):
        if self.blind.isChecked():
            return
        key = self.phrase_variable.currentData()
        if key:
            self.phrase.lineEdit().insert('{' + key + '}')
            self.phrase.lineEdit().setFocus()

    def insert_phrase(self):
        text = self.phrase.currentText().strip()
        if not text:
            return
        if '{' in text or '}' in text:
            if self.blind.isChecked():
                QMessageBox.warning(self, 'Verblindete Befundung', 'Automatische Messwertvariablen sind im verblindeten Modus gesperrt.')
                return
            from ..lab_phrase_library import render_measurement_phrase
            try:
                rendered = render_measurement_phrase(text, self.payload)
            except ValueError as exc:
                QMessageBox.warning(self, 'Baustein nicht übernommen', str(exc))
                return
            preview = QMessageBox(self)
            preview.setWindowTitle('Messwertbaustein prüfen')
            preview.setTextFormat(Qt.TextFormat.PlainText)
            preview.setText(rendered['text'])
            preview.setInformativeText('Gespeicherte automatische Werte in ms; keine manuelle Korrektur. '
                'Analyse: ' + str(rendered['analysis_id'] or 'nicht vorhanden') +
                '\nFehlende Variablen: ' + (', '.join(rendered['missing']) or 'keine') +
                '\nDiesen Text ausdrücklich in den Befund übernehmen?')
            preview.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            preview.setDefaultButton(QMessageBox.StandardButton.No)
            if preview.exec() != QMessageBox.StandardButton.Yes:
                return
            text = rendered['text']
        self._append_report_text(text)

    def refresh_user_phrases(self, *_):
        from ..lab_phrase_library import decode_phrases
        current = self.phrase.currentText()
        # Keep the original fixed entries; user templates follow them.
        while self.phrase.count() > 6:
            self.phrase.removeItem(6)
        try:
            entries = decode_phrases(QSettings('VCG', 'ExpertECGLabPhrases').value('library_v1'))
        except (ValueError, TypeError) as exc:
            self.phrase_library_note.setText(f'Bausteinbibliothek nicht lesbar: {exc}. Kein Überschreiben.')
            self.phrase_save.setEnabled(False)
            self.phrase_manage.setEnabled(False)
            logging.getLogger(__name__).warning('Cannot decode local phrase library: %s', exc)
        else:
            section = self.section_choice.currentText()
            for entry in entries:
                if entry['section'] == section:
                    self.phrase.addItem(entry['text'])
            self.phrase_save.setEnabled(True)
            self.phrase_manage.setEnabled(bool(entries))
            self.phrase_library_note.setText('Eigene Bausteine des ausgewählten Abschnitts; vor jeder Übernahme prüfen.')
        self.phrase.setEditText(current)

    def save_user_phrase(self):
        from ..lab_phrase_library import decode_phrases, encode_phrases, add_phrase
        text = self.phrase.currentText().strip()
        if not text:
            return
        answer = QMessageBox.question(self, 'Persönlichen Baustein speichern',
            'Diesen Text als wiederverwendbare lokale Benutzervorlage speichern? '
            'Keine Patientendaten oder fallspezifischen Messwerte aufnehmen. '
            'Dies speichert keinen Befund und bestätigt keine Diagnose.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        settings = QSettings('VCG', 'ExpertECGLabPhrases')
        try:
            entries = decode_phrases(settings.value('library_v1'))
            updated = add_phrase(entries, self.section_choice.currentText(), text)
            settings.setValue('library_v1', encode_phrases(updated))
            settings.sync()
            if settings.status() != QSettings.Status.NoError:
                raise OSError('Lokale Einstellungen konnten nicht gespeichert werden')
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, 'Baustein nicht gespeichert', str(exc))
            return
        self.refresh_user_phrases()

    def manage_user_phrase(self):
        from ..lab_phrase_library import decode_phrases, encode_phrases, change_phrase
        settings = QSettings('VCG', 'ExpertECGLabPhrases')
        try:
            entries = decode_phrases(settings.value('library_v1'))
            section = self.section_choice.currentText()
            texts = [item['text'] for item in entries if item['section'] == section]
            if not texts:
                QMessageBox.information(self, 'Eigene Bausteine', 'In diesem Abschnitt sind keine eigenen Bausteine gespeichert.')
                return
            original, ok = QInputDialog.getItem(self, 'Vorlage auswählen', section, texts, 0, False)
            if not ok:
                return
            action, ok = QInputDialog.getItem(self, 'Vorlage verwalten',
                'Bestehende Befunde bleiben unverändert.', ['Bearbeiten', 'Entfernen'], 0, False)
            if not ok:
                return
            replacement = None
            if action == 'Bearbeiten':
                replacement, ok = QInputDialog.getMultiLineText(self, 'Vorlage bearbeiten',
                    'Keine Patientendaten oder fallspezifischen Werte speichern.', original)
                if not ok:
                    return
            elif QMessageBox.question(self, 'Vorlage entfernen',
                    'Nur diese persönliche Vorlage entfernen? Bereits eingefügte Befundtexte bleiben erhalten.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            # Re-read after dialogs: never overwrite other intervening edits.
            entries = decode_phrases(settings.value('library_v1'))
            updated = change_phrase(entries, section, original, replacement)
            settings.setValue('library_v1', encode_phrases(updated))
            settings.sync()
            if settings.status() != QSettings.Status.NoError:
                raise OSError('Lokale Einstellungen konnten nicht gespeichert werden')
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, 'Vorlage nicht geändert', str(exc))
            return
        self.refresh_user_phrases()

    def _append_report_text(self, text):
        """Append reviewed or typed text literally, including angle brackets."""
        edit = self.sections[self.section_choice.currentText()]
        cursor = edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.beginEditBlock()
        if edit.toPlainText():
            cursor.insertBlock()
        cursor.insertText(text)  # Literal text, never interpret typed HTML.
        cursor.endEditBlock()
        edit.setTextCursor(cursor)

    def override_measurement(self):
        row = self.measurements.currentRow()
        if row < 0:
            return
        author = self.author.text().strip()
        if not author:
            QMessageBox.warning(self, 'Bearbeiter fehlt', 'Eine manuelle Messwertkorrektur benötigt einen Bearbeiter.')
            return
        name = self.measurements.item(row, 0).text()
        value, ok = QInputDialog.getText(self, "Manueller Befundwert", name + " – Wert einschließlich Einheit:")
        if not ok or not value.strip():
            return
        reason, ok = QInputDialog.getText(self, "Begründung", "Messdefinition / Ableitung / Schlag und Grund der Korrektur:")
        if not ok or not reason.strip():
            return
        self.manual_overrides[name] = {
            "manual_value": value.strip(), "automatic_value": self.measurements.item(row, 1).text(),
            "reason": reason.strip(), "author": author,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_id": self.payload.get("analysis_id", ""),
            "signal_reference": asdict(self.reference) if self.reference is not None else None,
            "method": "manual_report_override_not_signal_recalculation",
        }
        self.refresh_manual_overrides()
        self._mark_dirty()

    def refresh_manual_overrides(self):
        prefix = 'Automatische Werte bleiben unverändert; manuelle Befundwerte werden getrennt gespeichert.'
        if not self.manual_overrides:
            self.override_note.setText(prefix + '\nKeine manuellen Befundwerte für diesen Fall.')
            return
        self.override_note.setText(prefix + '\n' + json.dumps(
            self.manual_overrides, ensure_ascii=False, indent=2))

    def inspect_suggestion(self):
        self.suggestion_detail.clear()
        row = self.suggestions.currentRow()
        if row < 0 or self.suggestions.item(row, 0) is None:
            return
        from ..lab_marker_viewmodel import classical_rule_details
        name = self.suggestions.item(row, 0).text()
        rule = self.payload.get('classical_criteria', {}).get(name)
        self.suggestion_detail.setPlainText(classical_rule_details(name, rule, self.payload.get('analysis_id')))
        review = self.suggestion_review.get(name)
        if isinstance(review, dict):
            self.suggestion_detail.setPlainText(self.suggestion_detail.toPlainText()
                + '\n\nGespeicherte lokale Review-Entscheidung (kann älteren Analysestand betreffen):\n'
                + json.dumps(review, ensure_ascii=False, indent=2, default=str))

    def _refresh_qt_alert(self):
        from ..lab_qt_alert import qt_alert_items, SOURCE
        items = qt_alert_items(self.payload)
        visible = bool(items) and not self.blind.isChecked()
        self.qt_alert_note.setVisible(visible)
        self.qt_ack_button.setVisible(visible)
        self.qt_history_button.setVisible(visible)
        self.qt_ack_button.setEnabled(bool(self.reference and self.payload.get('analysis_id')))
        values = '; '.join(f"{i['method']}: {i['value_ms']:.6g} ms" for i in items)
        self.qt_alert_note.setText('QTc-Prüfhinweis >500 ms: ' + values + '. Messung, Formel, QRS-Breite, Rhythmus, '
            'Medikamente und Elektrolyte prüfen. Keine automatische LQTS-/Torsade-Diagnose; kein Echtzeit-Alarmsystem.')
        self.qt_alert_note.setToolTip(SOURCE)

    def show_qt_review_history(self):
        if self.blind.isChecked() or not self.reference or not self.payload.get('analysis_id'):
            return
        from ..lab_qt_alert import qt_review_history
        from PyQt6.QtWidgets import QDialog
        entries, errors = qt_review_history(self.store.root / 'alert_reviews', asdict(self.reference), self.payload)
        dialog = QDialog(self)
        dialog.setWindowTitle('Lokale QT-Quittierungen – keine klinische Freigabe')
        layout = QVBoxLayout(dialog)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(('\n\n'.join(
            f"{item['created_at']} · {item['author']}\n{item['review_note']}" for item in entries)
            or 'Keine passende Quittierung für diesen Signal- und Messstand vorhanden.')
            + ('\n\nNicht lesbare oder unpassende Messnachweise:\n' + '\n'.join(errors) if errors else '')
            + '\n\nQuittierungen ersetzen keine klinische Abklärung und unterdrücken den Hinweis nicht.')
        layout.addWidget(view)
        dialog.resize(700, 420)
        dialog.exec()

    def acknowledge_qt_alert(self):
        from ..lab_qt_alert import qt_alert_items, SOURCE
        from ...mimic.storage import atomic_write_json, content_hash
        items = qt_alert_items(self.payload)
        if self.blind.isChecked() or not items or not self.reference or not self.payload.get('analysis_id'):
            return
        if not self.author.text().strip():
            QMessageBox.warning(self, 'Bearbeiter fehlt', 'Die Quittierung benötigt einen Bearbeiter.')
            return
        reason, ok = QInputDialog.getText(self, 'QT-Prüfung dokumentieren', 'Was wurde geprüft / Ergebnis:')
        if not ok or not reason.strip():
            return
        document = dict(schema_version='1.0.0', kind='qt_review_acknowledgement',
            signal_reference=asdict(self.reference), analysis_id=self.payload['analysis_id'],
            items=items, items_sha256=content_hash(items), source=SOURCE,
            author=self.author.text().strip(), review_note=reason.strip(),
            created_at=datetime.now(timezone.utc).isoformat(), clinical_claim_allowed=False,
            confirms_treatment=False)
        try:
            directory = self.store.root / 'alert_reviews'
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / (uuid.uuid4().hex + '.json'), document)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Nicht gespeichert', str(exc))
            return
        self.qt_alert_note.setText(self.qt_alert_note.text() + '\nLokal quittiert; ersetzt keine klinische Abklärung.')

    def open_catalog_case_criterion(self, criterion):
        if self.blind.isChecked():
            return
        for row in range(self.suggestions.rowCount()):
            item = self.suggestions.item(row, 0)
            if item is not None and item.text() == criterion:
                self.rule_search.clear()
                self.tabs.setCurrentIndex(self.suggestions_tab)
                self.suggestions.setCurrentCell(row, 0)
                self.inspect_suggestion()
                return
        self.criteria_catalog.status.setText('Dieses Kriterium ist im geöffneten Fall nicht als klassischer '
            'Review-Vorschlag gespeichert. Die aktuelle Registerdefinition ersetzt kein fehlendes Fallergebnis.')

    def review_suggestion(self, decision):
        if decision not in ('accepted', 'edited', 'rejected'):
            raise ValueError('Unbekannte Review-Entscheidung')
        if self.blind.isChecked():
            return
        if not self.author.text().strip():
            QMessageBox.warning(self, 'Bearbeiter fehlt', 'Eine Review-Entscheidung benötigt einen Bearbeiter.')
            return
        row = self.suggestions.currentRow()
        if row < 0:
            return
        name = self.suggestions.item(row, 0).text()
        value = self.payload.get("classical_criteria", {}).get(name, {})
        from ..lab_marker_viewmodel import classical_rule_view
        view = classical_rule_view(value)
        if decision != "rejected" and not view['eligible']:
            QMessageBox.warning(self, "Nicht beurteilbar", view['status'] + ". Einen eigenen manuellen Befund getrennt dokumentieren.")
            return
        text = f"{value.get('definition', name)}: {self.suggestions.item(row, 2).text()} · {value.get('value')} {value.get('unit', '')}"
        if decision == "edited":
            text, ok = QInputDialog.getText(self, "Geprüfter Befundtext", "Manuell geänderter Text:", text=text)
            if not ok or not text.strip():
                return
        previous = self.suggestion_review.get(name, {})
        from ...mimic.storage import content_hash
        evidence = {'criterion_id': name, 'recorded_rule': copy.deepcopy(value),
                    'signal_reference': asdict(self.reference) if self.reference else None,
                    'analysis_id': self.payload.get('analysis_id', '')}
        if (decision == 'accepted' and previous.get('decision') == decision
                and previous.get('text') == text and previous.get('analysis_id') == self.payload.get('analysis_id', '')
                and previous.get('evidence_sha256') == content_hash(evidence)):
            return
        self.suggestion_review[name] = {"decision": decision, "text": text, "analysis_id": self.payload.get("analysis_id", ""), "author": self.author.text(), "reviewed_at": datetime.now(timezone.utc).isoformat()}
        self.suggestion_review[name]['evidence_snapshot'] = evidence
        self.suggestion_review[name]['evidence_sha256'] = content_hash(evidence)
        history = list(previous.get('history', []))
        if previous:
            history.append({key: value for key, value in previous.items() if key != 'history'})
        self.suggestion_review[name]['history'] = history
        self.suggestion_review[name]['requires_text_reconciliation'] = bool(
            previous.get('requires_text_reconciliation') or previous.get('decision') in ('accepted', 'edited'))
        if previous.get('decision') in ('accepted', 'edited'):
            QMessageBox.information(self, 'Befundtext prüfen', 'Die neue Entscheidung ersetzt nicht automatisch bereits eingefügten oder bearbeiteten Freitext. Bitte den Bericht auf widersprüchliche ältere Aussagen prüfen.')
        if decision != "rejected":
            self._append_report_text(text)
        self.suggestions.setItem(row, 3, QTableWidgetItem(decision))
        self._dirty = True

    def reconcile_suggestion_text(self):
        from ...mimic.lab_workspace import report_text_digest, unresolved_review_conflicts, unresolved_override_conflicts
        if self.case is None or self.blind.isChecked() or not self.author.text().strip():
            return
        sections = {name: edit.toPlainText() for name, edit in self.sections.items()}
        analysis_id = self.payload.get('analysis_id', '')
        pending = unresolved_review_conflicts(self.suggestion_review, sections, analysis_id)
        overrides = unresolved_override_conflicts(self.manual_overrides, sections, analysis_id)
        if not pending and not overrides:
            QMessageBox.information(self, 'Befundabgleich', 'Keine offenen Übernahmekonflikte für diesen Textstand.')
            return
        reason, ok = QInputDialog.getText(self, 'Befundtext manuell abgleichen',
            'Regeln: ' + ', '.join(pending) + '\nManuelle Messwerte: ' + ', '.join(overrides)
            + '\nAlte Aussagen und Herkunft im aktuellen Analysestand prüfen/korrigieren. Was wurde abgeglichen?')
        if not ok or not reason.strip():
            return
        proof = {'sections_sha256': report_text_digest(sections), 'author': self.author.text().strip(),
                 'reason': reason.strip(), 'reviewed_at': datetime.now(timezone.utc).isoformat(), 'analysis_id': analysis_id}
        for name in pending:
            self.suggestion_review[name]['text_reconciliation'] = dict(proof)
        for name in overrides:
            self.manual_overrides[name]['text_reconciliation'] = dict(proof)
        self._dirty = True

    def save_report(self, status="draft"):
        if self.case is None:
            return False
        try:
            report = ReportRevision(self.case.case_id, self.case.source_sha256, self.author.text(), status=status, sections={k: e.toPlainText() for k, e in self.sections.items()}, manual_overrides=self.manual_overrides, suggestion_review=self.suggestion_review, measurements=tuple(self.measurement_rows() or self.saved_measurements), analysis_id=self.payload.get("analysis_id", ""), parent_revision_id=self.parent_revision)
            self.store.save_report(report)
            self.parent_revision = report.revision_id
            self.report_analysis_id = report.analysis_id
            self.report_measurements = list(report.measurements)
            self.report_status.setText(f"{status} · {report.author} · {report.created_at} · {report.revision_id[:8]}")
            self._dirty = False
            return True
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Befund nicht gespeichert", str(exc))
            return False

    def compare_case(self):
        if self.prior is not None:
            self.prior = None
            self.prior_payload = {}
            self.comparison_changes.set_rows([])
            self.comparison_analysis.clear()
            self.comparison_values.setRowCount(0)
            self.comparison_plot.clear()
        elif self.comparison_id.currentData():
            case_id = self.comparison_id.currentData()
            candidate = self.library.load_case(case_id)
            identity = self._patient_identity(self.case)
            if not identity or self._patient_identity(candidate) != identity:
                raise ValueError("Unverified patient linkage")
            self.prior, self.prior_fs = self.library.load_signal(case_id)
            self.comparison_analysis.blockSignals(True)
            self.comparison_analysis.clear()
            self.comparison_analysis.addItem("Analysestand ausdrücklich wählen", None)
            for path in sorted((self.library.root / case_id / "analyses").glob("*/analysis.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    aid = data.get("analysis_id", "")
                    if (data.get("case_id") != case_id or not isinstance(aid, str)
                            or not aid or path.parent.name != aid[:16]
                            or data.get("signal_reference", {}).get("source_sha256") not in (None, candidate.source_sha256)):
                        continue
                    self.comparison_analysis.addItem(f"{aid[:16]} · {data.get('created_at', 'Zeit unbekannt')}", data)
                except (OSError, ValueError, TypeError):
                    logging.getLogger(__name__).warning("Invalid comparison analysis: %s", path, exc_info=True)
            self.comparison_analysis.blockSignals(False)
            self.comparison_note.setText(f"Orange: {candidate.pseudonym}. Zeitversatz ausdrücklich manuell; keine Amplitudennormierung. Chronologische Reihenfolge und Aufnahmebedingungen gesondert prüfen.")
            self._comparison_analysis_changed()
        self.redraw()

    def _comparison_analysis_changed(self, *_):
        from ...mimic.lab_comparison import paired_intervals
        self.prior_payload = self.comparison_analysis.currentData() or {}
        rows = paired_intervals(self.payload, self.prior_payload) if self.prior_payload else []
        self.comparison_changes.set_rows(rows)
        self.comparison_changes.setVisible(not self.blind.isChecked())
        self.comparison_values.setRowCount(len(rows))
        self.comparison_values.setVisible(not self.blind.isChecked())
        for i, row in enumerate(rows):
            values = (row['measurement'], row['current'], row['comparator'], row['delta_current_minus_comparator'], row['status'])
            for j, value in enumerate(values):
                text = "unbekannt" if value is None else f"{value:.3g}" if isinstance(value, float) else str(value)
                self.comparison_values.setItem(i, j, QTableWidgetItem(text))
        self.comparison_plot.clear()
        if self.prior is None or self.signal is None:
            return
        lead = self.lead_choice.currentIndex()
        if not 0 <= lead < 12:
            lead = 1
        current = self.comparison_plot.addPlot(row=0, col=0, title=f"Aktuell · {LEADS[lead]}")
        other = self.comparison_plot.addPlot(row=0, col=1, title=f"Vergleich · {LEADS[lead]} (nicht automatisch zeitlich früher)")
        current.plot(np.arange(len(self.signal))/self.reference.sampling_rate_hz, self.signal[:, lead], pen=pg.mkPen('#0072b2'))
        other.plot(np.arange(len(self.prior))/self.prior_fs, self.prior[:, lead], pen=pg.mkPen('#c16635'))
        other.setXLink(current)
        other.setYLink(current)
        for plot in (current, other):
            plot.setLabel('bottom', 'Signalzeit', units='s')
            plot.setLabel('left', 'Spannung', units='mV')
        current.setXRange(0, min(10, len(self.signal)/self.reference.sampling_rate_hz), padding=0)

    def _patient_identity(self, case):
        data = LabReviewStore(self.library.root, case.case_id).patient_link(case.source_sha256)
        if not data:
            data = case.metadata.get("verified_patient_identity", {})
        if isinstance(data, dict) and data.get("source") and data.get("patient_id"):
            return (str(data["source"]), str(data["patient_id"]))
        return None

    def refresh_comparison_cases(self):
        self.comparison_id.clear()
        self.patient_namespace.clear()
        self.patient_identifier.clear()
        self.patient_evidence.clear()
        identity = self._patient_identity(self.case)
        if identity:
            self.patient_namespace.setText(identity[0])
            self.patient_identifier.setText(identity[1])
            for candidate in self.library.list_cases():
                if candidate.case_id != self.case.case_id and self._patient_identity(candidate) == identity:
                    self.comparison_id.addItem(candidate.pseudonym, candidate.case_id)

    def save_patient_identity(self):
        if self.case is None:
            return
        try:
            self.store.save_patient_link(self.case.source_sha256, self.patient_namespace.text(), self.patient_identifier.text(), self.patient_evidence.text(), self.author.text())
            self.prior = None
            self.refresh_comparison_cases()
            self.redraw()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Zuordnung nicht gespeichert", str(exc))

    def filter_markers(self, text):
        group = self.marker_group.currentData()
        members = self.marker_groups.get(group, []) if group else None
        for row in range(self.markers.rowCount()):
            item = self.markers.item(row, 0)
            if item is None:
                continue
            name = item.text()
            search_text = item.data(Qt.ItemDataRole.UserRole) or name.casefold()
            self.markers.setRowHidden(row, text.casefold() not in search_text
                                     or (self.only_favorites.isChecked() and name not in self.marker_favorites)
                                     or (members is not None and name not in members))

    def _inspect_fiducial_evidence(self, row, _column):
        if self.blind.isChecked() or not self.reference:
            return
        try:
            seconds = float(self.fiducial_evidence.item(row, 1).text())
        except (AttributeError, ValueError):
            return
        if np.isfinite(seconds) and 0 <= seconds <= (self.reference.sample_count-1)/self.reference.sampling_rate_hz:
            self.landmark_time.setValue(seconds)
            self.timeSelected.emit(seconds)
            self._render_fiducial_candidates(row, seconds)

    def _render_fiducial_candidates(self, row, seconds):
        self.fiducial_overlay.clear()
        aliases = {'qrs_onset': 'r_onset', 'qrs_offset': 'r_offset'}
        selected = self.fiducial_evidence.item(row, 0).data(Qt.ItemDataRole.UserRole)
        selected = aliases.get(selected, selected)
        fs = self.reference.sampling_rate_hz
        lo, hi = max(0, seconds-.2), min(len(self.signal)/fs, seconds+.2)
        a, b = max(0, int(lo*fs)), min(len(self.signal), int(hi*fs)+1)
        lead = self.lead_choice.currentIndex()
        self.fiducial_overlay.plot(np.arange(a,b)/fs, self.signal[a:b,lead], pen='#258c9a')
        foreground = '#dce5ef' if self._dark else '#263746'
        self.fiducial_overlay.setTitle(f'{LEADS[lead]} · natives Rohsignal; Kandidaten vor Fusion, keine automatische Auswahl', color=foreground)
        self.fiducial_overlay.setXRange(lo, hi, padding=0)
        colors = {'neurokit':'#0072b2', 'multilead':'#d55e00', 'vcg':'#cc79a7'}
        contract = self.payload.get('delineation', {}).get('method_candidates', {})
        items = contract.get('items', []) if isinstance(contract, dict) and contract.get('schema_version') == '1.0.0' else []
        visible_count = 0
        for candidate in items[:100] if isinstance(items, list) else []:
            if not isinstance(candidate, dict) or candidate.get('landmark') != selected:
                continue
            time = candidate.get('time_seconds')
            if not isinstance(time, (int, float)) or isinstance(time, bool) or not np.isfinite(time) or not lo <= time <= hi:
                continue
            method = str(candidate.get('method', 'unbekannt'))
            line = pg.InfiniteLine(time, movable=False, pen=pg.mkPen(colors.get(method, '#888888'), style=Qt.PenStyle.DashLine),
                label=method + (' · verworfen (Zeit)' if candidate.get('temporally_admissible') is False else ' · Vorschlag'),
                labelOpts={'position': .9 - .15 * (visible_count % 5), 'color': foreground})
            self.fiducial_overlay.addItem(line)
            visible_count += 1
        final_name = {'r_onset':'qrs_onset', 'r_offset':'qrs_offset'}.get(selected, selected)
        details = self.payload.get('delineation', {}).get('fiducial_details', {})
        detail = details.get(final_name, {}) if isinstance(details, dict) else {}
        final = detail.get('time_seconds') if isinstance(detail, dict) else None
        if isinstance(final, (int,float)) and not isinstance(final, bool) and np.isfinite(final) and lo <= final <= hi:
            self.fiducial_overlay.addItem(pg.InfiniteLine(final, movable=False, pen=pg.mkPen('#888888', width=2), label='Gespeichertes Endergebnis', labelOpts={'position': .08, 'color': foreground}))

    def next_ischemia_context(self):
        if not self.lbbb_context.isChecked():
            return None
        if not self.author.text().strip():
            raise ValueError('Bearbeiter für LSB-Bestätigung erforderlich')
        reason, accepted = QInputDialog.getText(self, 'Morphologiekontext bestätigen',
            'Begründung / Referenz für ' + self.ischemia_context_kind.currentText()
            + ' (Gerätebesitz oder atriales Pacing allein genügt nicht):')
        if not accepted or not reason.strip():
            raise ValueError('Analyse nicht gestartet: LSB-Bestätigung abgebrochen oder unbegründet')
        return {'schema_version': '1.0.0', 'case_id': self.case.case_id,
                'context': self.ischemia_context_kind.currentData(), 'author': self.author.text().strip(), 'reason': reason.strip(),
                'created_at': datetime.now(timezone.utc).isoformat(), 'revision_id': uuid.uuid4().hex}

    def _refresh_marker_groups(self, selected=None):
        self.marker_group.blockSignals(True)
        self.marker_group.clear()
        self.marker_group.addItem('Alle Marker', None)
        for name in sorted(self.marker_groups):
            self.marker_group.addItem(f'{name} ({len(self.marker_groups[name])})', name)
        self.marker_group.setCurrentIndex(max(0, self.marker_group.findData(selected)))
        self.marker_group.blockSignals(False)
        self.filter_markers(self.marker_search.text())

    def _write_marker_groups(self, groups, selected=None):
        if self._marker_groups_read_error:
            QMessageBox.warning(self, 'Speichern gesperrt', 'Nicht lesbare Gruppendatei zuerst sichern und korrigieren; sie wird nicht überschrieben.')
            return
        from ...mimic.storage import atomic_write_json
        try:
            atomic_write_json(self.store.root / 'marker_groups.json', {
                'schema_version': '1.0.0', 'groups': groups,
                'scope': 'local_case_display_only', 'changes_analysis': False})
        except OSError as exc:
            QMessageBox.warning(self, 'Vergleichsgruppen nicht gespeichert', str(exc))
            return
        self.marker_groups = groups
        self._refresh_marker_groups(selected)

    def save_marker_group(self):
        if self.case is None:
            return
        names = sorted({self.markers.item(i.row(), 0).text()
                        for i in self.markers.selectionModel().selectedRows()
                        if not self.markers.isRowHidden(i.row())})
        if not 1 <= len(names) <= 20:
            QMessageBox.information(self, 'Vergleichsgruppe', 'Bitte 1 bis 20 sichtbare Marker auswählen.')
            return
        name, accepted = QInputDialog.getText(self, 'Lokale Vergleichsgruppe', 'Name:')
        name = name.strip()
        if not accepted or not name:
            return
        if name in self.marker_groups:
            QMessageBox.information(self, 'Name bereits vorhanden', 'Gruppe zuerst ausdrücklich entfernen oder anderen Namen wählen.')
            return
        self._write_marker_groups({**self.marker_groups, name: names}, name)

    def remove_marker_group(self):
        name = self.marker_group.currentData()
        if name and self.case is not None:
            self._write_marker_groups({k: v for k, v in self.marker_groups.items() if k != name})

    def toggle_marker_favorites(self):
        if self.case is None:
            return
        if self._marker_favorites_read_error:
            QMessageBox.warning(self, 'Speichern gesperrt', 'Nicht lesbare Favoritendatei zuerst sichern und korrigieren; sie wird nicht überschrieben.')
            return
        names = {self.markers.item(index.row(), 0).text()
                 for index in self.markers.selectionModel().selectedRows()
                 if not self.markers.isRowHidden(index.row())}
        if not names:
            return
        updated = self.marker_favorites - names if names <= self.marker_favorites else self.marker_favorites | names
        from ...mimic.storage import atomic_write_json
        try:
            atomic_write_json(self.store.root / "marker_favorites.json", {
                "schema_version": "1.0.0", "markers": sorted(updated),
                "scope": "local_case_display_only", "changes_analysis": False,
            })
        except OSError as exc:
            QMessageBox.warning(self, "Favoriten nicht gespeichert", str(exc))
            return
        self.marker_favorites = updated
        self.filter_markers(self.marker_search.text())

    def inspect_marker(self, *, keep_beat=False):
        self.marker_provenance_tree.clear()
        self.marker_evidence.set_evidence(None)
        selected = self.markers.selectedItems()
        if not selected:
            return
        name = self.markers.item(selected[0].row(), 0).text()
        from ...core.biomarkers import BiomarkerRegistry
        try:
            marker = BiomarkerRegistry.get(name)
            definition = {"description": marker.description, "unit": marker.unit, "references": marker.references}
            try:
                definition.update(asdict(BiomarkerRegistry.definition(name)))
            except ValueError:
                pass
        except (KeyError, ValueError):
            definition = {"status": "Definition im aktuellen Register nicht vorhanden"}
        detail = {"marker": name, "value": self.payload.get("feature_values", {}).get(name), "definition": definition, "vcg_method": self.payload.get("vcg_method"), "analysis_id": self.payload.get("analysis_id"), "signal_quality": self.payload.get("signal_quality"), "warning": "Forschungswert, keine Diagnosewahrscheinlichkeit. Null/fehlend ist nicht normal. Direct-8-Winkel ist kein anatomischer 3D-Winkel.", "measurement_scope": "Siehe Originaldefinition; keine abgeleitete Schlag-/Ableitungszuordnung erfunden."}
        measurements = self.payload.get('feature_measurements', {})
        if not isinstance(measurements, dict):
            measurements = {}
        detail["measurement_provenance"] = measurements.get(name, {"status": "Historisches Artefakt ohne vollständige Einzelmarker-Messprovenienz"})
        detail['current_registry_reference_not_historical_proof'] = detail.pop('definition')
        recorded = measurements.get(name)
        detail['recorded_definition'] = recorded if isinstance(recorded, dict) else {
            'status': 'Im gespeicherten Artefakt nicht enthalten; aktuelles Register ersetzt keine historische Definition'}
        self.marker_detail.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
        self.marker_provenance_tree.set_evidence(detail)
        self.marker_signal.clear()
        previous_beat = self.marker_beat.currentData() if keep_beat else None
        self.marker_beat.blockSignals(True)
        self.marker_beat.clear()
        self.marker_beat.setVisible(False)
        windows = self.payload.get("direct8_measurement_windows")
        original = name in ("qrs_t_area_angle_direct_8lead_deg", "qrs_t_integral_angle_direct_8lead_deg")
        if not original or not isinstance(windows, dict):
            windows = recorded.get('measurement_windows') if isinstance(recorded, dict) else None
        from ..lab_measurement_windows import beat_window_items, beat_audit_rows, BEAT_REASON_LABELS
        audit = beat_audit_rows(windows)
        self.marker_beat_audit.setRowCount(len(audit))
        self.marker_beat_audit.setVisible(bool(audit))
        for row, entry in enumerate(audit):
            text = (str(entry['beat_index'] + 1), 'ja' if entry['included'] else 'nein',
                    BEAT_REASON_LABELS.get(entry['reason'], entry['reason']))
            for col, value in enumerate(text):
                cell = QTableWidgetItem(value)
                cell.setToolTip(entry['reason'])
                self.marker_beat_audit.setItem(row, col, cell)
        if audit:
            self.marker_beat_audit.setToolTip(f"Erster fehlgeschlagener Prüfschritt, nicht alle möglichen Ursachen. Gründe gesamt: {windows.get('reason_counts', {})}; Tabelle gekürzt: {windows.get('audit_truncated', 'unbekannt')}")
        beat_items = beat_window_items(windows)
        is_beat_summary = bool(beat_items)
        if beat_items:
            for item in beat_items:
                self.marker_beat.addItem(f"Schlag {item['beat_index'] + 1} · {item['value']:.3f}° · Morphologiecluster {item.get('morphology_cluster_id', '?')} (keine Schlagdiagnose)", item['beat_index'])
            chosen = self.marker_beat.findData(previous_beat)
            self.marker_beat.setCurrentIndex(max(0, chosen))
            self.marker_beat.setVisible(True)
            chosen_item = beat_items[self.marker_beat.currentIndex()]
            self.marker_beat.setToolTip(f"{windows.get('stored_beats')} gespeichert / {windows.get('eligible_beats')} geeignet / {windows.get('record_beats')} erkannt; Mindestzahl {windows.get('minimum_beats')}; gekürzt: {windows.get('truncated')}. MAD ist kein Konfidenzintervall.")
            windows = dict(windows, **chosen_item)
            windows['schema_version'] = '1.0.0'
        self.marker_beat.blockSignals(False)
        recorded_signal = self.payload.get('signal_reference')
        signal_bound = (isinstance(recorded_signal, dict) and self.reference is not None
                        and recorded_signal.get('source_sha256') == self.reference.source_sha256
                        and recorded_signal.get('case_id') == self.reference.case_id)
        self.marker_evidence.set_evidence(detail, windows, self.analysis_arrays,
                                          signal_bound=signal_bound)
        if windows and self.analysis_arrays:
            signal = self.analysis_arrays.get(windows.get('signal_path')) if isinstance(windows, dict) else None
            if signal is not None and signal.ndim == 2 and signal.shape[1] == 12:
                from ..lab_measurement_windows import validated_direct8_windows
                contract = validated_direct8_windows(windows, len(signal))
                if contract is None:
                    self.marker_signal.setTitle("Inkompatibler Messfenstervertrag – keine Darstellung")
                    return
                fs, baseline_key, bounds = contract
                valid = list(bounds.values())
                a, b = max(0, min(int(v[0]) for v in valid)-10), min(len(signal), max(int(v[1]) for v in valid)+11)
                leads = (0, 1, 6, 7, 8, 9, 10, 11)
                for index, lead in enumerate(leads):
                    self.marker_signal.plot(np.arange(a, b)/fs, signal[a:b, lead]-index*3, pen="#258c9a")
                self.marker_signal.getAxis("left").setTicks([[(float(-i*3), LEADS[lead]) for i, lead in enumerate(leads)]])
                for key, color in ((baseline_key, (85, 122, 149, 45)), ("qrs", (209, 73, 91, 45)), ("j_to_t_end", (183, 121, 0, 45))):
                    left, right = windows[key]
                    self.marker_signal.addItem(pg.LinearRegionItem((left/fs, (right+1)/fs), movable=False, brush=color))
                variant = 'PR' if original else str(windows.get('baseline_kind'))
                scope = 'ein verwendeter Schlag, nicht der Median/MAD' if is_beat_summary else 'repräsentatives Signal, kein Beat-Median'
                self.marker_signal.setTitle(f"Gespeicherte Messfenster · {variant}-Baseline / QRS / J–T-Ende · {scope} · kein 3D-Winkel")
                return
        self.marker_signal.setTitle("Keine exakten Messfenster für diese Variante gespeichert; keine Grenzen rekonstruiert oder erfunden.")

    def export_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Signalansicht exportieren", "ekg.png", "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        if path.lower().endswith(".svg"):
            # Qt-native SVG avoids pyqtgraph's SVG path rewriter, which is
            # incompatible with some Qt 6 path encodings (including calipers).
            from PyQt6.QtSvg import QSvgGenerator
            from PyQt6.QtCore import QSize
            from PyQt6.QtGui import QPainter
            rect = self.canvas.ci.sceneBoundingRect()
            generator = QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(QSize(max(1, int(np.ceil(rect.width()))), max(1, int(np.ceil(rect.height())))))
            generator.setViewBox(rect)
            generator.setTitle("ECG laboratory – display export, not calibrated paper ECG")
            painter = QPainter(generator)
            if not painter.isActive():
                raise OSError("SVG export could not start")
            try:
                self.canvas.scene().render(painter, rect, rect)
            finally:
                painter.end()
            return
        import pyqtgraph.exporters
        pyqtgraph.exporters.ImageExporter(self.canvas.ci).export(path)

    def export_lead_matrix(self):
        if self.reference is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Gespeicherte Ableitungsmatrix exportieren', 'ableitungsmatrix.json', 'JSON (*.json);;CSV (*.csv)')
        if not path:
            return
        from ..lab_lead_matrix import lead_matrix_export
        document = lead_matrix_export(self.payload, asdict(self.reference))
        try:
            if Path(path).suffix.lower() == '.csv':
                # Numeric negative measurements remain numbers, not text formulas.
                # Guard textual provenance if an imported identifier is unusual.
                def csv_safe(value):
                    return "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else value
                rows = [dict(row, analysis_id=document['analysis_id'],
                    source_sha256=self.reference.source_sha256,
                    coordinate_convention=document['coordinate_convention'],
                    measurement_scope=document['measurement_scope']) for row in document['rows']]
                import io
                from ...mimic.storage import atomic_write_text
                with io.StringIO(newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows({key: csv_safe(value) for key, value in row.items()} for row in rows)
                    atomic_write_text(Path(path), '\ufeff' + handle.getvalue())
            elif Path(path).suffix.lower() == '.json':
                from ...mimic.storage import atomic_write_json
                atomic_write_json(Path(path), document)
            else:
                raise ValueError('Bitte .csv oder .json als Dateiendung verwenden.')
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).exception('Lead matrix export failed')
            QMessageBox.warning(self, 'Export fehlgeschlagen', str(exc))

    def export_measurements(self):
        if self.reference is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Messungen exportieren", "messungen.json", "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        rows = self.measurement_rows() or self.saved_measurements
        try:
            if path.lower().endswith(".csv"):
                from ..lab_csv_export import write_csv_atomic
                fields = ["lead", "start_sample", "end_sample", "duration_ms", "delta_mv", "method", "author"]
                write_csv_atomic(path, ({key: row.get(key) for key in fields} for row in rows), fields)
            else:
                from ...mimic.storage import atomic_write_json
                atomic_write_json(Path(path), {"schema_version": "1.0.0", "signal": asdict(self.reference), "measurements": rows, "fiducials": self.fiducials})
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Messungen nicht exportiert', str(exc))

    def export_automatic_intervals(self):
        if self.reference is None or self.blind.isChecked():
            return
        from ..lab_interval_export import interval_export
        from ...mimic.storage import atomic_write_json
        document = interval_export(self.payload, asdict(self.reference))
        if not document['rows']:
            QMessageBox.information(self, 'Keine Intervalle', 'Keine aufgezeichneten automatischen Intervalle vorhanden.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Automatische Intervalle (ungeprüft)',
            'automatische_intervalle.json', 'JSON (*.json);;CSV (*.csv)')
        if not path:
            return
        try:
            if path.lower().endswith('.csv'):
                rows = [dict(row, analysis_id=document['analysis_id'],
                    signal_sha256=self.reference.source_sha256,
                    case_id=self.reference.case_id) for row in document['rows']]
                from ..lab_csv_export import write_csv_atomic
                write_csv_atomic(path, rows, list(rows[0]))
            else:
                atomic_write_json(Path(path), document)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Export fehlgeschlagen', str(exc))

    def import_device_intervals(self):
        if self.reference is None or self.blind.isChecked():
            return
        from ..lab_device_comparison import validate_device_intervals
        from ...mimic.storage import atomic_write_json
        path, _ = QFileDialog.getOpenFileName(self, 'Signalgebundene Geräteintervalle', '', 'JSON (*.json)')
        if not path:
            return
        try:
            source = Path(path)
            if source.stat().st_size > 1_000_000:
                raise ValueError('Geräteintervall-Datei zu groß')
            document = validate_device_intervals(json.loads(source.read_text(encoding='utf-8')), asdict(self.reference))
            target = self.store.root / 'device_intervals' / (uuid.uuid4().hex + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target, document)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Gerätevergleich nicht geladen', str(exc))
            return
        self.device_interval_document = document
        self.refresh_device_intervals()

    def export_device_comparison(self):
        if self.reference is None or self.blind.isChecked() or self.device_interval_document is None:
            return
        from ..lab_device_comparison import device_interval_comparison
        from ...mimic.storage import atomic_write_json
        try:
            rows = device_interval_comparison(self.payload, self.device_interval_document, asdict(self.reference))
            path, _ = QFileDialog.getSaveFileName(self, 'Deskriptiver Gerätevergleich',
                'geraetevergleich.json', 'JSON (*.json);;CSV (*.csv)')
            if not path:
                return
            rows = [dict(row, unit='ms', case_id=self.reference.case_id,
                         source_sha256=self.reference.source_sha256,
                         analysis_id=self.payload.get('analysis_id'),
                         clinical_claim_allowed=False) for row in rows]
            if path.lower().endswith('.csv'):
                from ..lab_csv_export import write_csv_atomic
                write_csv_atomic(path, rows, list(rows[0]))
            else:
                atomic_write_json(Path(path), dict(schema_version='1.0.0',
                    kind='descriptive_device_interval_comparison', signal_reference=asdict(self.reference),
                    analysis_id=self.payload.get('analysis_id'), rows=rows,
                    clinical_claim_allowed=False, device_intervals=self.device_interval_document))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Gerätevergleich nicht exportiert', str(exc))

    def reopen_device_intervals(self):
        if self.reference is None or self.blind.isChecked():
            return
        from ..lab_device_comparison import validate_device_intervals
        directory = self.store.root / 'device_intervals'
        path, _ = QFileDialog.getOpenFileName(
            self, 'Gespeicherte Gerätevergleichsrevision auswählen', str(directory), 'JSON (*.json)')
        if not path:
            return
        try:
            source = Path(path).resolve()
            if source.parent != directory.resolve():
                raise ValueError('Bitte eine gespeicherte Revision dieses Laborfalls auswählen; externe Dateien über Import laden.')
            if source.stat().st_size > 1_000_000:
                raise ValueError('Geräteintervall-Datei zu groß')
            document = validate_device_intervals(
                json.loads(source.read_text(encoding='utf-8')), asdict(self.reference))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Revision nicht geladen', str(exc))
            return
        self.device_interval_document = document
        self.refresh_device_intervals()

    def refresh_device_intervals(self):
        from ..lab_device_comparison import device_interval_comparison
        self.device_interval_table.setRowCount(0)
        self.device_interval_table.hide()
        if self.device_interval_document is None or self.reference is None:
            self.device_interval_note.setText('Kein fallgebundener Gerätevergleich geladen.')
            return
        try:
            rows = device_interval_comparison(self.payload, self.device_interval_document, asdict(self.reference))
        except (ValueError, TypeError):
            self.device_interval_document = None
            self.device_interval_note.setText('Gerätevergleich passt nicht zum aktuellen Signal.')
            return
        self.device_interval_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [row['interval']] + [
                'nicht verfügbar' if row[key] is None else format(row[key], '.6g')
                for key in ('own_ms', 'device_ms', 'difference_ms')]
            values.append('deskriptiver Vergleich' if row['difference_ms'] is not None else 'eigene Messung fehlt/mehrdeutig')
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(json.dumps(row, ensure_ascii=False))
                self.device_interval_table.setItem(index, column, item)
        self.device_interval_table.show()
        self.device_interval_note.setText('Quelle: ' + self.device_interval_document['device_source']
            + '\nDeskriptiver Vergleich, kein Genauigkeitsnachweis. Schlagauswahl, RR und Messkonventionen können abweichen.')

    def export_pdf(self):
        if self.reference is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Befund-PDF", "befund.pdf", "PDF (*.pdf)")
        if not path:
            return
        if (self._dirty or self.measurement_rows() != self.report_measurements) and not self.save_report("draft"):
            return
        text = f"<h1>Lokaler EKG-Befund · {html.escape(self.case.pseudonym)}</h1><p>Forschungssoftware – keine Medizinproduktfreigabe.</p><p>{html.escape(self.report_status.text())}</p>"
        for name, edit in self.sections.items():
            text += f"<h3>{html.escape(name)}</h3><p>{html.escape(edit.toPlainText()).replace(chr(10), '<br>')}</p>"
        text += f"<p>Quelle SHA256: {self.case.source_sha256}<br>Analyse: {html.escape(self.payload.get('analysis_id', 'nicht berechnet'))}<br>Native Abtastrate: {self.reference.sampling_rate_hz:g} Hz</p>"
        text += "<h3>Manuelle Messungen</h3><table><tr><th>Ableitung</th><th>Δt (ms)</th><th>ΔU (mV)</th><th>Methode</th></tr>"
        for row in self.measurement_rows():
            text += f"<tr><td>{html.escape(row['lead'])}</td><td>{row['duration_ms']:.3f}</td><td>{row['delta_mv']:.3f}</td><td>{html.escape(row['method'])}</td></tr>"
        text += "</table>"
        for name, item in self.manual_overrides.items():
            text += f"<p>Manueller Befundwert {html.escape(name)}: {html.escape(item['manual_value'])}. Grund: {html.escape(item['reason'])}; automatisch: {html.escape(item['automatic_value'])}</p>"
        from ..lab_export import export_ecg_pdf, research_appendix_html
        appendix = None
        if self.pdf_research.isChecked() and not self.blind.isChecked():
            names = [self.markers.item(index.row(), 0).text() for index in self.markers.selectionModel().selectedRows()]
            if not names:
                QMessageBox.warning(self, "Forschungsanhang", "Zuerst die gewünschten Markerzeilen auswählen. Es wurde kein PDF exportiert.")
                return
            appendix = research_appendix_html(self.payload, names)
        export_ecg_pdf(path, text, self.signal, self.reference.sampling_rate_hz, float(self.speed.currentText()), float(self.gain.currentText()), research_html=appendix,
                       cabrera=self.lead_order.currentIndex() == 1, invert_avr=self.avr_polarity.currentIndex() == 1)
