"""Read-only, on-demand scientific signal displays, separate from inference."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel

from ..lab_research_views import beat_amplitude_map, native_spectrogram


class ResearchDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal = None
        self.fs = None
        self.r_seconds = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(['Schlag-Amplitudenkarte', 'Zeit-Frequenz-Ansicht'])
        self.lead = QComboBox()
        self.lead.addItems(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])
        self.lead.setCurrentText('II')
        button = QPushButton('Darstellung berechnen')
        button.clicked.connect(self.render)
        for control in (self.mode, self.lead, button):
            controls.addWidget(control)
        layout.addLayout(controls)
        self.note = QLabel('Methodische Darstellung, keine Diagnose oder klinische Nutzenbestätigung. Berechnung nur auf Anforderung.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setMinimumHeight(340)
        layout.addWidget(self.canvas)
        self.mode.currentIndexChanged.connect(self.clear)
        self.lead.currentIndexChanged.connect(self.clear)

    def set_data(self, signal, fs, r_seconds=None):
        self.signal, self.fs, self.r_seconds = signal, fs, r_seconds
        self.clear()

    def clear(self, *_):
        self.canvas.clear()
        self.note.setText('Neue Auswahl: Darstellung berechnen. Originalsignal unverändert; keine automatische physiologische Interpretation.')

    def render(self):
        self.canvas.clear()
        if self.signal is None:
            self.note.setText('Kein natives Signal verfügbar.')
            return
        signal = self.signal[:, self.lead.currentIndex()]
        try:
            plot = self.canvas.addPlot(title=self.lead.currentText())
            if self.mode.currentIndex() == 0:
                if self.r_seconds is None:
                    raise ValueError('Keine zum ausgewählten Analysestand gehörenden R-Marken verfügbar.')
                result = beat_amplitude_map(signal, self.fs, self.r_seconds)
                values = result['values_mv']
                if not len(values) or not np.any(np.isfinite(values)):
                    raise ValueError('Keine vollständigen darstellbaren Schlagfenster.')
                t = result['relative_seconds']
                extent = QRectF(t[0]-.5/self.fs, -.5, len(t)/self.fs, len(values))
                label = 'Native Amplitude (mV)'
                limits = float(np.nanmax(np.abs(values))) or 1.
                levels = (-limits, limits)
                cmap = pg.ColorMap([0,.5,1], ['#0072b2','#ffffff','#d55e00'])
                plot.setLabel('bottom', 'Zeit relativ zur R-Marke', units='s')
                plot.setLabel('left', 'Angezeigter Schlagindex (nicht Zeit)')
                plot.getAxis('left').setTicks([[(i, f"#{int(result['beat_indices'][i])+1} · {result['r_seconds'][i]:.2f}s")
                                              for i in np.unique(np.linspace(0, len(values)-1, min(8,len(values)), dtype=int))]])
                self.note.setText(f"{len(values)}/{result['eligible_beats']} vollständige Schlagfenster; {result['edge_excluded']} Randfenster ausgeschlossen. "
                                  'Keine Baselinekorrektur, keine Amplitudennormierung, keine Mittelung unterschiedlicher Morphologien. '
                                  'Farbmuster sind weder Alternans-Nachweis noch Schlagdiagnose. '
                                  + ('Anzeige gleichmäßig ausgedünnt; keine Schätzung einer Arrhythmielast.' if result['sampled_for_display'] else 'Alle vollständigen Fenster angezeigt.'))
            else:
                result = native_spectrogram(signal, self.fs)
                frequencies, times = result['frequencies_hz'], result['seconds']
                if len(times) < 2:
                    raise ValueError('Zu wenige Zeitfenster für eine Zeit-Frequenz-Darstellung.')
                values = 10*np.log10(np.maximum(result['power_mv2_hz'], 1e-12))
                dt, df = times[1]-times[0], frequencies[1]-frequencies[0]
                extent = QRectF(times[0]-dt/2, frequencies[0]-df/2, len(times)*dt, len(frequencies)*df)
                levels = (float(values.max())-80, float(values.max()))
                label = 'dB re 1 mV²/Hz (Anzeige: 80 dB)'
                cmap = pg.colormap.get('viridis')
                plot.setLabel('bottom', 'Signalzeit', units='s')
                plot.setLabel('left', 'Frequenz', units='Hz')
                plot.setYRange(0, self.fs/2, padding=0)
                self.note.setText(f"Native Abtastrate {self.fs:g} Hz; erste {result['duration_seconds']:g} s. Hann-Fenster "
                                  f"{result['window_samples']/self.fs*1000:.1f} ms; 75 % Überlappung; je Fenster Mittelwert entfernt. "
                                  'Keine HF-QRS-, Spätpotential- oder Alternansdiagnostik. Frequenzen oberhalb Nyquist sind nicht rekonstruierbar.')
            image = pg.ImageItem(values, axisOrder='row-major')
            image.setRect(extent)
            plot.addItem(image)
            bar = pg.ColorBarItem(values=levels, colorMap=cmap, label=label, interactive=False)
            bar.setImageItem(image, insert_in=plot)
        except ValueError as exc:
            self.canvas.clear()
            self.note.setText(f'Nicht darstellbar: {exc}')
