from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QProgressBar, QLabel, QGroupBox,
    QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ...config import AppConfig
from ...core.pipeline import PipelineState, PipelineProgress, _format_eta


class PipelineWidget(QWidget):
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    force_recompute_requested = pyqtSignal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        config_group = QGroupBox("Pipeline Configuration")
        config_layout = QVBoxLayout(config_group)

        vcg_layout = QHBoxLayout()
        vcg_layout.addWidget(QLabel("VCG Transform:"))
        self.vcg_combo = QComboBox()
        self.vcg_combo.addItems(["kors", "inverse_dower"])
        self.vcg_combo.setCurrentText(self.config.vcg_method)
        self.vcg_combo.currentTextChanged.connect(self._on_vcg_changed)
        vcg_layout.addWidget(self.vcg_combo)
        config_layout.addLayout(vcg_layout)

        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Delineation: Fusion (NeuroKit + Multilead + VCG)"))
        config_layout.addLayout(info_layout)

        layout.addWidget(config_group)

        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout(control_group)
        self.run_btn = QPushButton("Process Records")
        self.run_btn.setToolTip("Build or update the ECG/VCG/delineation/biomarker cache.")
        self.run_btn.clicked.connect(self.run_requested.emit)
        control_layout.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn)

        self.force_recompute_btn = QPushButton("Neuberechnung erzwingen")
        self.force_recompute_btn.setToolTip(
            "Löscht den gesamten Cache (Fortschritt + Analyse) und erzwingt eine vollständige Neuberechnung."
        )
        self.force_recompute_btn.setStyleSheet("color: #c0392b;")
        self.force_recompute_btn.clicked.connect(self.force_recompute_requested.emit)
        control_layout.addWidget(self.force_recompute_btn)

        layout.addWidget(control_group)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.state_label = QLabel("State: Idle")
        progress_layout.addWidget(self.state_label)
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.detail_label = QLabel("")
        progress_layout.addWidget(self.detail_label)
        self.eta_label = QLabel("")
        progress_layout.addWidget(self.eta_label)
        layout.addWidget(progress_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    def _on_vcg_changed(self, method: str):
        self.config.vcg_method = method

    def update_progress(self, progress: PipelineProgress):
        state_text = progress.state.value.replace("_", " ").title()
        self.state_label.setText(f"State: {state_text}")
        self.progress_bar.setValue(progress.percent)
        self.detail_label.setText(progress.message)

        if progress.elapsed_seconds > 0 and progress.current_record > 0:
            rate = progress.current_record / progress.elapsed_seconds
            remaining = progress.total_records - progress.current_record
            eta = remaining / rate if rate > 0 else 0
            elapsed_str = _format_eta(progress.elapsed_seconds)
            eta_str = _format_eta(eta)
            self.eta_label.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        else:
            self.eta_label.setText("")

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.force_recompute_btn.setEnabled(not running)

    def append_log(self, message: str):
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
