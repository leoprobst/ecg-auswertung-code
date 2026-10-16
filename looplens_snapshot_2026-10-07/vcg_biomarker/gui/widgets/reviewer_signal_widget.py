"""Blinded twelve-lead reading with explicit judgments, not automated diagnoses."""
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton

from ...core.data_loader import LEAD_NAMES_12
from ...mimic.reviewer_packet import ReviewerPacket
from ...mimic.reviewer_responses import save_review


class ReviewerSignalWidget(QWidget):
    def __init__(self, packet: ReviewerPacket, responses: Path, parent=None):
        super().__init__(parent)
        self.packet, self.responses = packet, Path(responses)
        self.loaded_case: str | None = None
        layout = QVBoxLayout(self)
        note = QLabel("Verblindete Signalbeurteilung. Keine Gerätebefunde, Marker oder Auswahlgruppen. "
            "Eine Signalbeurteilung ersetzt keinen unabhängigen klinischen Nachweis. "
            "Bei fehlender Beurteilbarkeit ausdrücklich unsicher/nicht beurteilbar wählen. "
            "Urteile sind endgültige Einzelabgaben; Korrekturen erfordern koordinierte Nachprüfung.")
        note.setWordWrap(True)
        layout.addWidget(note)
        bar = QHBoxLayout()
        self.case = QComboBox()
        self.case.addItems(packet.cases.review_case_id.tolist())
        self.scale = QComboBox()
        for value in (1.,2.,5.):
            self.scale.addItem(f"Alle Ableitungen: ±{value:g} mV", value)
        self.scale.setCurrentIndex(1)
        bar.addWidget(self.case, 2)
        bar.addWidget(self.scale)
        layout.addLayout(bar)
        self.question = QLabel()
        layout.addWidget(self.question)
        self.graph = pg.GraphicsLayoutWidget()
        self.graph.setBackground(self.palette().base().color())
        layout.addWidget(self.graph, 1)
        self.plots: list[pg.PlotItem] = []
        for i, lead in enumerate(LEAD_NAMES_12):
            plot = self.graph.addPlot(row=i//3, col=i%3, title=lead)
            foreground = self.palette().text().color()
            plot.setTitle(lead,color=foreground.name())
            for side in ("left","bottom"):
                plot.getAxis(side).setPen(pg.mkPen(foreground))
                plot.getAxis(side).setTextPen(pg.mkPen(foreground))
            plot.setLabel("left", "mV")
            plot.setLabel("bottom", "Zeit", units="s")
            plot.showGrid(x=True,y=True,alpha=.15)
            plot.setMouseEnabled(x=True,y=False)
            if self.plots:
                plot.setXLink(self.plots[0])
            self.plots.append(plot)
        controls = QHBoxLayout()
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("Eigene Beurteilerkennung (keine Patientenkennung)")
        self.verdict = QComboBox()
        for label,verdict_value in (("Urteil auswählen",None),("Positiv","positive"),("Negativ","negative"),
                            ("Unsicher","uncertain"),("Signal nicht beurteilbar","unreadable")):
            self.verdict.addItem(label,verdict_value)
        self.save = QPushButton("Urteil verbindlich speichern")
        controls.addWidget(self.reviewer,2)
        controls.addWidget(self.verdict)
        controls.addWidget(self.save)
        layout.addLayout(controls)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.case.currentIndexChanged.connect(self.load_case)
        self.scale.currentIndexChanged.connect(self.apply_scale)
        self.save.clicked.connect(self.save_verdict)
        self.load_case()

    def apply_scale(self):
        amplitude = self.scale.currentData()
        for plot in self.plots:
            plot.setYRange(-amplitude,amplitude,padding=0)

    def load_case(self):
        self.loaded_case = None
        self.save.setEnabled(False)
        self.verdict.setCurrentIndex(0)
        self.question.clear()
        for plot in self.plots:
            plot.clear()
        try:
            case = self.case.currentText()
            values, fs = self.packet.signal(case)
            row = self.packet.cases.loc[self.packet.cases.review_case_id.eq(case)].iloc[0]
            self.question.setText(f"Fragestellung: {row.endpoint_id} · {fs:g} Hz · {len(values)/fs:.2f} s")
            times = np.arange(len(values))/fs
            for i, plot in enumerate(self.plots):
                curve = plot.plot(times,values[:,i],pen=pg.mkPen("#0072B2",width=1))
                curve.setDownsampling(auto=True,method="peak")
                curve.setClipToView(True)
            self.plots[0].setXRange(0,times[-1],padding=0)
            self.apply_scale()
            self.loaded_case = case
            self.save.setEnabled(True)
            self.status.setText("Signal geprüft. Einheitliche Skalierung; Anzeigeverdichtung erhält Spitzen, Originalwerte bleiben unverändert.")
        except (OSError,ValueError,KeyError) as exc:
            self.status.setText(f"Fall nicht geladen – keine Bewertung möglich: {exc}")

    def save_verdict(self):
        if self.loaded_case is None:
            return
        try:
            save_review(self.packet,case_id=self.loaded_case,verdict=self.verdict.currentData(),
                        reviewer_id=self.reviewer.text(),response_root=self.responses)
            self.save.setEnabled(False)
            self.status.setText("Urteil gespeichert. Keine automatische Änderung von Patientenlabels.")
        except (OSError,ValueError) as exc:
            self.status.setText(f"Nicht gespeichert: {exc}")
