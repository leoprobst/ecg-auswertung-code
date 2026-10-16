from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QToolBar,
    QFileDialog, QMessageBox, QLabel, QInputDialog,
    QDialog, QVBoxLayout, QPushButton,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QAction

from ..config import AppConfig
from ..core.pipeline import Pipeline, PipelineCancelled, PipelineProgress, ProcessedRecord
from ..core.parallel_pipeline import ParallelPipeline
from ..core.analysis.results import AnalysisRun
from ..core.cache import ParquetEngineUnavailableError
from ..core.report import ClinicalReport
from ..core.reference import ReferenceLoopStore
from .ecg_viewer_window import EKGViewerWindow


def _schedule_os_shutdown(delay_seconds: int) -> bool:
    """Schedule an OS-level shutdown. Returns True on success."""
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["shutdown", "/s", "/t", str(delay_seconds)])
        else:
            subprocess.Popen(["shutdown", "-h", f"+{max(1, delay_seconds // 60)}"])
        return True
    except Exception:
        return False


def _cancel_os_shutdown() -> None:
    """Abort a previously scheduled OS shutdown."""
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/a"], check=False,
                           capture_output=True)
        else:
            subprocess.run(["shutdown", "-c"], check=False,
                           capture_output=True)
    except Exception:
        pass


class ShutdownCountdownDialog(QDialog):
    """Non-modal countdown dialog shown after analysis when auto-shutdown is active."""

    cancelled = pyqtSignal()

    def __init__(self, delay_minutes: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Herunterfahren geplant")
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedWidth(320)
        self._remaining = delay_minutes * 60

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._label)

        btn = QPushButton("Abbrechen")
        btn.setFixedHeight(32)
        btn.clicked.connect(self._on_cancel)
        layout.addWidget(btn)

        self._update_label()

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(1000)

    def _tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._tick_timer.stop()
            self.close()
        else:
            self._update_label()

    def _update_label(self):
        mins = self._remaining // 60
        secs = self._remaining % 60
        self._label.setText(
            "Analyse abgeschlossen.\n\n"
            f"PC fährt in  <b>{mins}:{secs:02d}</b>  herunter.\n\n"
            "Nicht gespeicherte Arbeit jetzt sichern."
        )

    def _on_cancel(self):
        self._tick_timer.stop()
        self.cancelled.emit()
        self.close()

    def closeEvent(self, event):
        self._tick_timer.stop()
        super().closeEvent(event)


class PipelineWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(object)
    log = pyqtSignal(str)
    records_done = pyqtSignal(list)
    analysis_done = pyqtSignal(object)
    analysis_update = pyqtSignal(object)

    def __init__(self, pipeline: Pipeline, mode: str = "process", **kwargs):
        super().__init__()
        self.pipeline = pipeline
        self.mode = mode
        self.kwargs = kwargs

    def run(self):
        try:
            self.pipeline.set_progress_callback(lambda p: self.progress.emit(p))
            self.pipeline.set_log_callback(lambda m: self.log.emit(m))
            self.pipeline.set_analysis_update_callback(lambda r: self.analysis_update.emit(r))
            if self.mode == "process":
                self.records_done.emit(self.pipeline.process_records(**self.kwargs))
            elif self.mode == "analyze":
                self.analysis_done.emit(self.pipeline.run_analysis(**self.kwargs))
        except PipelineCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            if getattr(self.pipeline, "_cancel_requested", False):
                self.cancelled.emit("Job abgebrochen. Fortschritt wurde gespeichert.")
            else:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """
    Main application window — handles dataset management, batch pipeline
    execution and statistical analysis.

    Individual ECG inspection is done in a separate EKGViewerWindow that
    is opened on demand. Thresholds learned here are pushed to the viewer
    automatically whenever a new analysis run finishes.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        super().__init__()
        self._startup_progress = progress_callback or (lambda message, fraction: None)
        self.config = config or AppConfig()
        self.pipeline = ParallelPipeline(self.config)
        self._worker: Optional[PipelineWorker] = None
        self._thread: Optional[QThread] = None
        self._last_analysis: Optional[AnalysisRun] = None
        self._last_processed_record: Optional[ProcessedRecord] = None
        self._report = ClinicalReport()
        self._reference_store = ReferenceLoopStore()
        self._ecg_viewer: Optional[EKGViewerWindow] = None
        self._developer_dialog: Optional[QDialog] = None
        self.developer_widget = None
        self._shutdown_after_analysis: bool = False
        self._shutdown_delay_minutes: int = 5
        self._shutdown_dialog: Optional[ShutdownCountdownDialog] = None

        self.setWindowTitle("VCG Feature Scanner")
        self.setMinimumSize(1200, 800)

        self._setup_menu()
        self._setup_toolbar()
        self._setup_tabs()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_settings()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_menu(self):
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        act = QAction("Open &MIMIC Validation Project", self)
        act.setToolTip("Open the local MIMIC clinical validation workspace")
        act.triggered.connect(self._open_mimic_workspace)
        file_menu.addAction(act)

        act = QAction("&Open PTB-XL Dataset…", self)
        act.setShortcut("Ctrl+D")
        act.setToolTip("Select the root folder of the PTB-XL ECG dataset")
        act.triggered.connect(self._on_open_dataset)
        file_menu.addAction(act)

        act = QAction("Open &ECG Viewer", self)
        act.setShortcut("Ctrl+E")
        act.setToolTip("Open the ECG Viewer window for single-record inspection")
        act.triggered.connect(self._open_ecg_viewer)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("Export Results as &CSV…", self)
        act.setToolTip("Export the last analysis run to a CSV file")
        act.triggered.connect(self._on_export_csv)
        file_menu.addAction(act)

        act = QAction("Export Results as &Excel…", self)
        act.setToolTip("Export the last analysis run to an Excel workbook")
        act.triggered.connect(self._on_export_excel)
        file_menu.addAction(act)

        act = QAction("Export Evidence as &JSON...", self)
        act.setToolTip("Export the last analysis run with curves and evidence metadata")
        act.triggered.connect(self._on_export_json)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("E&xit", self)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # ── Pipeline ──────────────────────────────────────────────────
        pipeline_menu = mb.addMenu("&Pipeline")

        act = QAction("Process &ECG Records", self)
        act.setShortcut("F5")
        act.setToolTip(
            "Build or update the processed-record cache: preprocess, VCG-transform, delineate and compute biomarkers."
        )
        act.triggered.connect(self._on_run_pipeline)
        pipeline_menu.addAction(act)

        self._analyze_action = QAction("&Analyze Processed Records", self)
        self._analyze_action.setShortcut("F6")
        self._analyze_action.setToolTip(
            "Run ROC/threshold analysis on cached processed records. This can be repeated without recomputing biomarkers."
        )
        self._analyze_action.triggered.connect(self._on_run_analysis)
        pipeline_menu.addAction(self._analyze_action)

        pipeline_menu.addSeparator()

        act = QAction(f"Workers: {self.pipeline.n_workers}", self)
        act.setEnabled(False)
        pipeline_menu.addAction(act)

        act = QAction("&Cancel", self)
        act.setShortcut("Escape")
        act.setToolTip("Cancel the running pipeline or analysis job")
        act.triggered.connect(self._on_cancel)
        pipeline_menu.addAction(act)

        # ── Settings ──────────────────────────────────────────────────
        tools_menu = mb.addMenu("&Tools")

        act = QAction("&Developer Diagnostics...", self)
        act.setShortcut("Ctrl+Shift+D")
        act.setToolTip("Open the developer smoke test, warning capture and runtime/file-size estimator.")
        act.triggered.connect(self._on_run_developer_diagnostics)
        tools_menu.addAction(act)

        tools_menu.addSeparator()

        shutdown_menu = tools_menu.addMenu("Herunterfahren nach Analyse")
        shutdown_menu.setToolTipsVisible(True)

        self._shutdown_toggle_act = QAction("Aktiv — nach Analyse herunterfahren", self)
        self._shutdown_toggle_act.setCheckable(True)
        self._shutdown_toggle_act.setChecked(False)
        self._shutdown_toggle_act.setToolTip(
            "Nach Abschluss der nächsten Analyse den PC automatisch herunterfahren.\n"
            "Ein Countdown-Dialog erscheint und kann abgebrochen werden."
        )
        self._shutdown_toggle_act.toggled.connect(self._on_toggle_shutdown)
        shutdown_menu.addAction(self._shutdown_toggle_act)

        self._shutdown_delay_act = QAction("Verzögerung: 5 Minuten", self)
        self._shutdown_delay_act.setToolTip(
            "Wartezeit zwischen Analyseende und Herunterfahren festlegen (1–60 Minuten)."
        )
        self._shutdown_delay_act.triggered.connect(self._on_set_shutdown_delay)
        shutdown_menu.addAction(self._shutdown_delay_act)

        settings_menu = mb.addMenu("&Settings")

        vcg_menu = settings_menu.addMenu("VCG Transform Method")
        vcg_menu.setToolTipsVisible(True)
        for method, tip in [
            ("kors", "Kors regression-based transform (recommended)"),
            ("inverse_dower", "Inverse Dower transform"),
        ]:
            act = QAction(method, self)
            act.setCheckable(True)
            act.setChecked(method == self.config.vcg_method)
            act.setToolTip(tip)
            act.triggered.connect(lambda checked, m=method: self._set_vcg_method(m))
            vcg_menu.addAction(act)

        act = QAction("Keep Signal Data in Memory", self)
        act.setCheckable(True)
        act.setChecked(self.config.keep_signal_data)
        act.setToolTip(
            "Retain raw and filtered ECG signals in memory after processing.\n"
            "Disable to reduce RAM usage when processing large datasets."
        )
        act.triggered.connect(lambda checked: setattr(self.config, "keep_signal_data", checked))
        settings_menu.addAction(act)

        cache_menu = settings_menu.addMenu("Cache")
        cache_menu.setToolTipsVisible(True)

        act = QAction("Set Cache Directory…", self)
        act.setToolTip("Choose a folder for caching processed records between sessions")
        act.triggered.connect(self._on_set_cache_dir)
        cache_menu.addAction(act)

        act = QAction("Clear Cache", self)
        act.setToolTip("Delete all cached processing results")
        act.triggered.connect(self._on_clear_cache)
        cache_menu.addAction(act)

        act = QAction("Neuberechnung erzwingen", self)
        act.setToolTip(
            "Löscht Cache + Analyse-Ergebnisse vollständig und erzwingt eine neue vollständige Berechnung beim nächsten Start der Pipeline."
        )
        act.triggered.connect(self._on_force_recompute)
        cache_menu.addAction(act)

        # MIMIC is the primary clinical workflow. Legacy PTB processing
        # actions remain implemented for reproducibility, but are deliberately
        # absent from the primary navigation.
        for menu_action in mb.actions():
            if menu_action.text().replace("&", "") in {
                "Pipeline",
                "Tools",
                "Settings",
            }:
                menu_action.setVisible(False)
        for file_action in file_menu.actions():
            normalized = file_action.text().replace("&", "")
            file_action.setVisible(normalized == "Exit")
        # Window controls already provide the only remaining command. Keeping
        # a one-item legacy menu would add a second, unexplained exit action.
        mb.setVisible(False)

    def _setup_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        mimic_action = QAction("MIMIC Validation", self)
        mimic_action.setToolTip("Open the clinical MIMIC validation workspace")
        mimic_action.triggered.connect(self._open_mimic_workspace)
        tb.addAction(mimic_action)
        tb.addSeparator()

        # ── Dataset ───────────────────────────────────────────────────
        act = QAction("Open Dataset", self)
        act.setToolTip("Open PTB-XL dataset directory  (Ctrl+D)")
        act.triggered.connect(self._on_open_dataset)
        tb.addAction(act)

        tb.addSeparator()

        # ── Pipeline ──────────────────────────────────────────────────
        self._run_action = QAction("Process ECGs", self)
        self._run_action.setToolTip(
            "Process selected ECG records and build the reusable biomarker cache.  (F5)"
        )
        self._run_action.triggered.connect(self._on_run_pipeline)
        tb.addAction(self._run_action)

        tb.addAction(self._analyze_action)

        self._cancel_action = QAction("⏹  Cancel", self)
        self._cancel_action.setToolTip("Cancel the currently running pipeline or analysis job")
        self._cancel_action.setEnabled(False)
        self._cancel_action.triggered.connect(self._on_cancel)
        tb.addAction(self._cancel_action)

        tb.addSeparator()

        # ── ECG Viewer ────────────────────────────────────────────────
        act = QAction("🔬  ECG Viewer", self)
        act.setToolTip(
            "Open the ECG Viewer window to inspect individual records,\n"
            "view 3-D VCG loops and run single-record diagnosis  (Ctrl+E)"
        )
        act.triggered.connect(self._open_ecg_viewer)
        tb.addAction(act)

        tb.addSeparator()

        # ── Export ────────────────────────────────────────────────────
        act = QAction("📄  Export CSV", self)
        act.setToolTip("Export analysis results to CSV")
        act.triggered.connect(self._on_export_csv)
        tb.addAction(act)

        for action in tb.actions():
            if action is not mimic_action:
                action.setVisible(False)
        # The MIMIC workspace is already the only visible primary view.
        # Its guided overview is the canonical place for project actions.
        tb.setVisible(False)

    def _setup_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.setCentralWidget(self.tab_widget)

        from .widgets.data_widget import DataWidget
        from .widgets.pipeline_widget import PipelineWidget
        from .widgets.analysis_widget import AnalysisWidget
        from .widgets.evidence_widget import EvidenceExplorerWidget
        from .widgets.performance_widget import PerformanceDashboardWidget
        from .widgets.mimic_project_widget import MimicProjectWidget

        # MimicProjectWidget does essentially all of the real startup work
        # (config discovery, hashing, DB-backed refreshes) -- give it the
        # bulk of the progress range and forward a rescaled callback so the
        # splash's progress bar advances smoothly across the whole thing.
        self._startup_progress("Arbeitsbereich wird geöffnet...", 0.05)
        self.mimic_widget = MimicProjectWidget(
            self,
            progress_callback=lambda message, fraction: self._startup_progress(
                message, 0.05 + fraction * 0.80
            ),
        )
        self._startup_progress("Zusatzansichten werden vorbereitet...", 0.88)
        self.data_widget = DataWidget(self.config, self.pipeline)
        self._startup_progress("Zusatzansichten werden vorbereitet...", 0.90)
        self.pipeline_widget = PipelineWidget(self.config)
        self._startup_progress("Zusatzansichten werden vorbereitet...", 0.92)
        self.analysis_widget = AnalysisWidget(self.config)
        self._startup_progress("Zusatzansichten werden vorbereitet...", 0.94)
        self.evidence_widget = EvidenceExplorerWidget(self.config)
        self._startup_progress("Zusatzansichten werden vorbereitet...", 0.96)
        self.performance_widget = PerformanceDashboardWidget(self.config)
        self._startup_progress("Fertig.", 1.0)

        mimic_index = self.tab_widget.addTab(
            self.mimic_widget,
            "MIMIC Validation",
        )
        self.tab_widget.setTabToolTip(
            mimic_index,
            "Clinical reference validation with patient-level splits and a sealed locked test.",
        )

        data_index = self.tab_widget.addTab(self.data_widget, "1 · Data")
        self.tab_widget.setTabToolTip(data_index,
            "Load and filter the PTB-XL dataset.\n"
            "Select diagnostic classes and apply quality filters before running the pipeline."
        )

        pipeline_index = self.tab_widget.addTab(
            self.pipeline_widget,
            "2 · Pipeline",
        )
        self.tab_widget.setTabToolTip(pipeline_index,
            "Monitor pipeline progress and log output.\n"
            "Run pipeline from here or via the toolbar."
        )

        analysis_index = self.tab_widget.addTab(
            self.analysis_widget,
            "3 · Analysis",
        )
        self.tab_widget.setTabToolTip(analysis_index,
            "Configure and run the ROC / threshold analysis.\n"
            "Results are used to learn optimal diagnostic cut-offs."
        )

        evidence_index = self.tab_widget.addTab(
            self.evidence_widget,
            "4 - Evidence",
        )
        self.tab_widget.setTabToolTip(evidence_index,
            "Research-first evidence explorer with rankings, drilldowns,\n"
            "threshold tradeoffs, provenance and narrative summaries."
        )

        performance_index = self.tab_widget.addTab(
            self.performance_widget,
            "5 - Performance",
        )
        self.tab_widget.setTabToolTip(performance_index,
            "Pathology-centered performance overview with biomarker rankings\n"
            "and contextual fixed clinical ECG reference rules."
        )
        for index in range(1, self.tab_widget.count()):
            self.tab_widget.setTabVisible(index, False)
            self.tab_widget.setTabEnabled(index, False)
        # MIMIC is the single primary workspace; its five inner tabs are the
        # complete user navigation. The outer one-tab bar is redundant.
        self.tab_widget.tabBar().hide()

    def _open_mimic_workspace(self):
        if hasattr(self, "mimic_widget"):
            self.tab_widget.setCurrentWidget(self.mimic_widget)

    def _setup_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._status_label = QLabel(
            "MIMIC-Forschungsprojekt wird geprüft"
        )
        sb.addWidget(self._status_label, stretch=1)

        self._analysis_status = QLabel("")
        self._analysis_status.setStyleSheet("padding-right: 8px;")
        sb.addPermanentWidget(self._analysis_status)

    def _connect_signals(self):
        self.pipeline.set_progress_callback(self._on_pipeline_progress)
        self.pipeline.set_log_callback(self._on_pipeline_log)
        self.data_widget.dataset_loaded.connect(self._on_dataset_loaded)
        self.data_widget.view_record_requested.connect(self._on_view_record_in_viewer)
        self.pipeline_widget.run_requested.connect(self._on_run_pipeline)
        self.pipeline_widget.cancel_requested.connect(self._on_cancel)
        self.pipeline_widget.force_recompute_requested.connect(self._on_force_recompute)
        self.mimic_widget.status_changed.connect(
            self._status_label.setText
        )
        self._status_label.setText(
            self.mimic_widget.project_status.text()
        )

    # ------------------------------------------------------------------
    # ECG Viewer management
    # ------------------------------------------------------------------

    def _get_viewer(self) -> EKGViewerWindow:
        """Return the ECG viewer, creating it fresh if needed."""
        if self._ecg_viewer is None or not self._ecg_viewer.isVisible():
            self._ecg_viewer = EKGViewerWindow(
                config=self.config,
                reference_store=self._reference_store,
                pipeline=self.pipeline,
                parent=None,   # top-level window
            )
            # Push current thresholds immediately if analysis has been run
            if self._last_analysis is not None:
                self._ecg_viewer.update_thresholds(self._last_analysis)

        self._ecg_viewer.show()
        self._ecg_viewer.raise_()
        return self._ecg_viewer

    def _open_ecg_viewer(self):
        self._get_viewer()

    def _on_view_record_in_viewer(self, ecg_id: int):
        """Load a dataset record and display it in the ECG viewer window."""
        if self.pipeline.data_loader is None:
            QMessageBox.warning(
                self, "No Dataset",
                "Please load a PTB-XL dataset first (tab '1 · Data')."
            )
            return

        viewer = self._get_viewer()
        # Delegate loading to the viewer's own load-by-ID handler
        viewer._pipeline = self.pipeline
        try:
            self._status_label.setText(f"Loading ECG ID {ecg_id}…")
            record = self.pipeline.data_loader.get_record(ecg_id, load_signal=True)
            if not record.has_signal:
                QMessageBox.warning(self, "No Signal", f"ECG ID {ecg_id} has no signal data.")
                return
            pr = viewer._run_pipeline(record, label=f"ECG ID {ecg_id}")
            self._last_processed_record = pr
            viewer.display_record(pr)
            self._status_label.setText(f"ECG ID {ecg_id} opened in ECG Viewer")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load ECG ID {ecg_id}:\n{exc}")
            self._status_label.setText(f"Error loading ECG ID {ecg_id}")

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------

    def _on_pipeline_progress(self, progress: PipelineProgress):
        self._status_label.setText(progress.message)
        self.pipeline_widget.update_progress(progress)

    def _on_pipeline_log(self, message: str):
        self.pipeline_widget.append_log(message)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select PTB-XL Dataset Directory")
        if path:
            self.config.ptbxl_path = Path(path)
            self.data_widget.set_dataset_path(path)
            QSettings("VCGBiomarker", "App").setValue("ptbxl_path", path)
            self.data_widget._load_dataset()

    def _on_dataset_loaded(self, info: dict):
        n = info.get("n_records", 0)
        self._status_label.setText(
            f"Dataset loaded — {n:,} records  |  {self.pipeline.n_workers} workers"
        )

    def _on_run_pipeline(self):
        if self.pipeline.data_loader is None:
            QMessageBox.warning(
                self, "No Dataset",
                "Load a PTB-XL dataset first (tab '1 · Data' or File → Open PTB-XL Dataset)."
            )
            return

        ecg_ids = self.data_widget.get_selected_ids()
        if not ecg_ids:
            try:
                ecg_ids = self.pipeline.load_data(str(self.config.ptbxl_path))
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to load records:\n{exc}")
                return

        if not ecg_ids:
            QMessageBox.warning(
                self, "No Records",
                "No records match the current filters. Adjust the selection in '1 · Data'."
            )
            return

        self._run_in_thread(mode="process", ecg_ids=ecg_ids, vcg_method=self.config.vcg_method)

    def _on_run_analysis(self):
        if not self.pipeline.has_processed_data():
            QMessageBox.warning(
                self, "No Processed Records",
                "Run the pipeline first to process ECG records before performing analysis."
            )
            return

        self._save_analysis_settings()
        self._run_in_thread(
            mode="analyze",
            correction_method=self.analysis_widget.get_correction_method(),
            bootstrap_iterations=self.analysis_widget.get_bootstrap_iterations(),
            combination_strategy=self.analysis_widget.get_combination_strategy(),
            greedy_seeds=self.analysis_widget.get_greedy_seeds(),
            greedy_max_depth=self.analysis_widget.get_greedy_max_depth(),
            max_biomarker_combination=self.analysis_widget.get_max_combination(),
            compute_subgroups=self.analysis_widget.get_compute_subgroups(),
        )

    def _current_analysis_settings(self) -> dict:
        return {
            "correction_method": self.analysis_widget.get_correction_method(),
            "bootstrap_iterations": self.analysis_widget.get_bootstrap_iterations(),
            "combination_strategy": self.analysis_widget.get_combination_strategy(),
            "greedy_seeds": self.analysis_widget.get_greedy_seeds(),
            "greedy_max_depth": self.analysis_widget.get_greedy_max_depth(),
            "max_combination": self.analysis_widget.get_max_combination(),
            "compute_subgroups": self.analysis_widget.get_compute_subgroups(),
        }

    def _save_analysis_settings(self):
        settings = QSettings("VCGBiomarker", "App")
        current = self._current_analysis_settings()
        settings.setValue("analysis_correction_method", current["correction_method"])
        settings.setValue("analysis_bootstrap_iterations", int(current["bootstrap_iterations"]))
        settings.setValue("analysis_combination_strategy", current["combination_strategy"])
        settings.setValue("analysis_greedy_seeds", int(current["greedy_seeds"]))
        settings.setValue("analysis_greedy_max_depth", int(current["greedy_max_depth"]))
        settings.setValue("analysis_max_combination", int(current["max_combination"]))
        settings.setValue("analysis_compute_subgroups", bool(current["compute_subgroups"]))

    def _ensure_developer_dialog(self) -> QDialog:
        if self._developer_dialog is None:
            from .widgets.developer_widget import DeveloperDiagnosticsWidget

            dialog = QDialog(self)
            dialog.setWindowTitle("Developer Diagnostics")
            dialog.resize(980, 720)
            layout = QVBoxLayout(dialog)
            self.developer_widget = DeveloperDiagnosticsWidget(
                self.config,
                self.pipeline,
                records_provider=self.data_widget.get_selected_ids,
                analysis_settings_provider=self._current_analysis_settings,
                parent=dialog,
            )
            layout.addWidget(self.developer_widget)
            self._developer_dialog = dialog
        return self._developer_dialog

    def _on_run_developer_diagnostics(self):
        dialog = self._ensure_developer_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.developer_widget.run_diagnostics()

    def _on_cancel(self):
        self.pipeline.request_cancel()
        self.pipeline_widget.append_log("Abbruch angefordert - speichere aktuellen Fortschritt...")
        self._status_label.setText("Abbruch angefordert - Checkpoint wird gespeichert")

    def _on_export_csv(self):
        if self._last_analysis is None:
            QMessageBox.information(self, "No Analysis", "Run an analysis first to export results.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            self._last_analysis.export_csv(Path(path))
            self._status_label.setText(f"Exported CSV → {path}")

    def _on_export_excel(self):
        if self._last_analysis is None:
            QMessageBox.information(self, "No Analysis", "Run an analysis first to export results.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "", "Excel Files (*.xlsx)")
        if path:
            self._last_analysis.export_excel(Path(path))
            self._status_label.setText(f"Exported Excel → {path}")

    def _on_export_json(self):
        if self._last_analysis is None:
            QMessageBox.information(self, "No Analysis", "Run an analysis first to export evidence.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Evidence JSON", "", "JSON Files (*.json)")
        if path:
            self._last_analysis.export_json(Path(path), include_curves=True)
            self._status_label.setText(f"Exported evidence JSON -> {path}")

    def _on_set_cache_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if path:
            self.pipeline.cache.cache_dir = Path(path)
            self._status_label.setText(f"Cache-Verzeichnis: {path}")
            QSettings("VCGBiomarker", "App").setValue("cache_dir", path)

    def _on_clear_cache(self):
        self.pipeline.cache.clear()
        self.pipeline._processed_records = []
        self.pipeline._bm_accumulator = {}
        self.pipeline._path_accumulator = {}
        self._last_analysis = None
        self.evidence_widget.set_analysis_run(None)
        self.performance_widget.set_analysis_run(None)
        self._status_label.setText("Cache geleert")

    def _on_force_recompute(self):
        reply = QMessageBox.question(
            self, "Neuberechnung erzwingen",
            "Hiermit wird der gesamte Cache (Fortschritt + Analyse-Ergebnisse) gelöscht.\n"
            "Die nächste Pipeline-Ausführung beginnt von vorne.\n\n"
            "Fortfahren?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.pipeline.cache.clear()
        self.pipeline._processed_records = []
        self.pipeline._bm_accumulator = {}
        self.pipeline._path_accumulator = {}
        self._last_analysis = None
        self.evidence_widget.set_analysis_run(None)
        self.performance_widget.set_analysis_run(None)
        self.pipeline_widget.append_log("Cache vollständig geleert — nächste Pipeline-Ausführung startet von vorne")
        self._status_label.setText("Cache geleert — bereit für Neuberechnung")
        self._analysis_status.setText("")

    def _set_vcg_method(self, method: str):
        self.config.vcg_method = method
        self.pipeline_widget.vcg_combo.setCurrentText(method)
        QSettings("VCGBiomarker", "App").setValue("vcg_method", method)
        self._status_label.setText(f"VCG method set to: {method}")

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _run_in_thread(self, mode: str, **kwargs):
        self._thread = QThread()
        self._worker = PipelineWorker(self.pipeline, mode=mode, **kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_pipeline_progress)
        self._worker.log.connect(self._on_pipeline_log)
        self._worker.error.connect(self._on_worker_error)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.records_done.connect(self._on_records_done)
        self._worker.analysis_done.connect(self._on_analysis_done)
        self._worker.analysis_update.connect(self._on_analysis_update)
        self._set_pipeline_running(True)
        self._thread.start()

    def _set_pipeline_running(self, running: bool):
        self.pipeline_widget.set_running(running)
        self._run_action.setEnabled(not running)
        self._analyze_action.setEnabled(not running)
        self._cancel_action.setEnabled(running)

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Pipeline Error", msg)
        self._set_pipeline_running(False)

    def _on_worker_cancelled(self, msg: str):
        self.pipeline_widget.append_log(msg)
        self._status_label.setText(msg)
        self._set_pipeline_running(False)

    def _on_worker_finished(self):
        self._set_pipeline_running(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
            self.pipeline.set_analysis_update_callback(None)

    def _on_records_done(self, records: list):
        # Count from cache for accuracy (streaming mode returns [] or [last_record])
        n_cached = self.pipeline.cache.get_n_cached() if self.pipeline.cache.cache_dir else 0
        n_ok = n_cached if n_cached > 0 else sum(1 for r in records if r.error is None)
        n_err = sum(1 for r in records if r.error is not None)

        self.pipeline_widget.append_log(
            f"Verarbeitung abgeschlossen: {n_ok:,} Records OK, {n_err} Fehler"
        )

        # Records ist im Streaming-Modus leer oder enthaelt nur den letzten Record
        ok_records = [r for r in records if r.error is None]

        # Reference loops falls VCG-Daten verfuegbar (nur letzter Batch)
        if ok_records:
            try:
                norm = [r for r in ok_records if r.vcg_result and r.delineation
                        and "NORM" in r.ecg_record.diagnostic_superclass]
                if len(norm) >= 5:
                    self._reference_store.compute_from_records(norm)
            except Exception:
                pass

        last_ok = next((r for r in reversed(ok_records) if r.error is None), None)
        if last_ok:
            self._last_processed_record = last_ok

        self._status_label.setText(
            f"Pipeline abgeschlossen — {n_ok:,} Records verarbeitet. "
            "'Analyze Processed' ausfuehren fuer Schwellenwerte."
        )
        self._analysis_status.setText(f"{n_ok:,} Records bereit")

    def _on_analysis_done(self, run: AnalysisRun):
        self._last_analysis = run
        self.analysis_widget.display_results(run)
        self.evidence_widget.set_analysis_run(run)
        self.performance_widget.set_analysis_run(run)

        n_sig = sum(1 for r in run.results if r.is_significant)
        self.pipeline_widget.append_log(
            f"Analysis complete: {len(run.results)} pairs tested, {n_sig} significant"
        )
        self._status_label.setText(
            f"Analysis complete — {n_sig} significant biomarker/pathology pairs found."
        )
        self._analysis_status.setText(
            f"Thresholds learned: {n_sig} pairs  |  Open ECG Viewer to diagnose"
        )

        # Push thresholds to viewer if it is open
        if self._ecg_viewer is not None and self._ecg_viewer.isVisible():
            self._ecg_viewer.update_thresholds(run)

        if self._shutdown_after_analysis:
            self._start_shutdown_countdown()

    def _on_analysis_update(self, run: AnalysisRun):
        self._last_analysis = run
        self.analysis_widget.display_results(run)
        self.evidence_widget.set_analysis_run(run)
        self.performance_widget.set_analysis_run(run)

        n_sig = sum(1 for r in run.results if r.is_significant)
        expected = run.cohort_filter.get("n_expected_pairs")
        expected_text = f"/{expected:,}" if expected else ""
        self._analysis_status.setText(
            f"Live: {len(run.results):,}{expected_text} Paare | {n_sig} signif."
        )

        if self._ecg_viewer is not None and self._ecg_viewer.isVisible():
            self._ecg_viewer.update_thresholds(run)

    # ------------------------------------------------------------------
    # Auto-shutdown
    # ------------------------------------------------------------------

    def _on_toggle_shutdown(self, checked: bool):
        self._shutdown_after_analysis = checked
        QSettings("VCGBiomarker", "App").setValue("shutdown_after_analysis", checked)

    def _on_set_shutdown_delay(self):
        value, ok = QInputDialog.getInt(
            self,
            "Verzögerung festlegen",
            "Minuten bis zum Herunterfahren (nach Analyseende):",
            self._shutdown_delay_minutes, 1, 60,
        )
        if ok:
            self._shutdown_delay_minutes = value
            self._shutdown_delay_act.setText(f"Verzögerung: {value} Minuten")
            QSettings("VCGBiomarker", "App").setValue("shutdown_delay_minutes", value)

    def _start_shutdown_countdown(self):
        delay_secs = self._shutdown_delay_minutes * 60
        if not _schedule_os_shutdown(delay_secs):
            QMessageBox.warning(
                self, "Herunterfahren fehlgeschlagen",
                "Der OS-Shutdown-Befehl konnte nicht ausgeführt werden.\n"
                "Bitte manuell herunterfahren.",
            )
            return

        self._shutdown_dialog = ShutdownCountdownDialog(self._shutdown_delay_minutes, self)
        self._shutdown_dialog.cancelled.connect(self._cancel_shutdown_sequence)
        self._shutdown_dialog.show()

        self.pipeline_widget.append_log(
            f"Herunterfahren in {self._shutdown_delay_minutes} Minuten geplant. "
            "Countdown-Dialog zum Abbrechen verwenden."
        )
        self._status_label.setText(
            f"Herunterfahren in {self._shutdown_delay_minutes} Min. — "
            "Dialog öffnen zum Abbrechen"
        )

    def _cancel_shutdown_sequence(self):
        _cancel_os_shutdown()
        self.pipeline_widget.append_log("Herunterfahren abgebrochen.")
        self._status_label.setText("Herunterfahren abgebrochen.")
        self._shutdown_dialog = None

    # ------------------------------------------------------------------
    # Settings persistence & cache resume
    # ------------------------------------------------------------------

    def _restore_settings(self):
        settings = QSettings("VCGBiomarker", "App")

        geometry = settings.value("mimic_window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        # PTB cache, dataset and analysis settings are intentionally not
        # restored into the primary MIMIC workflow.
        return

        # Restore cache directory
        cache_dir = settings.value("cache_dir")
        if cache_dir:
            p = Path(cache_dir)
            if p.exists():
                try:
                    self.pipeline.cache.cache_dir = p
                except ParquetEngineUnavailableError as exc:
                    self.pipeline_widget.append_log(f"Cache kann nicht geladen werden: {exc}")
                    self._status_label.setText("Cache benötigt pyarrow oder fastparquet.")

        # Restore VCG method
        vcg_method = settings.value("vcg_method", "kors")
        if vcg_method in ("kors", "inverse_dower"):
            self.config.vcg_method = vcg_method
            self.pipeline_widget.vcg_combo.setCurrentText(vcg_method)

        # Restore analysis settings before cache resume messages are shown.
        correction_method = settings.value(
            "analysis_correction_method",
            self.config.multiple_testing_correction,
        )
        idx = self.analysis_widget.correction_combo.findText(str(correction_method))
        if idx >= 0:
            self.analysis_widget.correction_combo.setCurrentIndex(idx)

        try:
            bootstrap_iterations = int(settings.value(
                "analysis_bootstrap_iterations",
                self.config.bootstrap_iterations,
            ))
            self.analysis_widget.bootstrap_spin.setValue(bootstrap_iterations)
        except (TypeError, ValueError):
            pass

        combo_strategy = settings.value("analysis_combination_strategy", "none")
        strategy_map = {"none": 0, "greedy": 1, "brute_force": 2}
        if combo_strategy in strategy_map:
            self.analysis_widget.combo_strategy_combo.setCurrentIndex(strategy_map[combo_strategy])
        elif settings.value("analysis_max_combination", "1") not in ("1", 1):
            # Migrate old max_combination > 1 → brute_force
            self.analysis_widget.combo_strategy_combo.setCurrentIndex(2)

        try:
            greedy_seeds = int(settings.value("analysis_greedy_seeds", 3))
            self.analysis_widget.greedy_seeds_spin.setValue(greedy_seeds)
        except (TypeError, ValueError):
            pass

        try:
            greedy_max_depth = int(settings.value("analysis_greedy_max_depth", 2))
            self.analysis_widget.greedy_depth_spin.setValue(greedy_max_depth)
        except (TypeError, ValueError):
            pass

        compute_subgroups = settings.value("analysis_compute_subgroups", False)
        if isinstance(compute_subgroups, str):
            compute_subgroups = compute_subgroups.lower() == "true"
        self.analysis_widget.subgroups_check.setChecked(bool(compute_subgroups))

        # Restore auto-shutdown settings
        shutdown_after = settings.value("shutdown_after_analysis", False)
        if isinstance(shutdown_after, str):
            shutdown_after = shutdown_after.lower() == "true"
        self._shutdown_after_analysis = bool(shutdown_after)
        self._shutdown_toggle_act.setChecked(bool(shutdown_after))

        try:
            delay = int(settings.value("shutdown_delay_minutes", 5))
            self._shutdown_delay_minutes = max(1, min(60, delay))
            self._shutdown_delay_act.setText(f"Verzögerung: {self._shutdown_delay_minutes} Minuten")
        except (TypeError, ValueError):
            pass

        # Restore dataset path and auto-load it
        ptbxl_path = settings.value("ptbxl_path")
        if ptbxl_path:
            p = Path(ptbxl_path)
            if p.exists():
                self.config.ptbxl_path = p
                self.data_widget.set_dataset_path(ptbxl_path)
                try:
                    self.data_widget._load_dataset()
                except Exception:
                    pass

    def _try_resume_from_cache(self):
        from ..core.biomarkers.base import BiomarkerRegistry
        from ..core.pathologies.registry import PathologyRegistry

        cache = self.pipeline.cache
        if cache.cache_dir is None:
            return

        # Schnell: nur die gecachten IDs zählen, keine Records laden
        n_cached = cache.get_n_cached()
        if n_cached == 0:
            return

        # Config-Hash-Prüfung
        if not cache.check_config_valid():
            self.pipeline_widget.append_log(
                "Cache-Konfiguration geaendert — beim naechsten Lauf wird neu verarbeitet"
            )
            return

        # Pathologien- und Biomarker-Validierung
        current_bm = BiomarkerRegistry.available()
        current_path = PathologyRegistry.available()
        valid, reason = cache.check_checkpoint_valid(current_bm, current_path)

        if not valid:
            self.pipeline_widget.append_log(
                f"Cache ungueltig (Aenderungen erkannt): {reason}\n"
                "Bitte 'Neuberechnung erzwingen' verwenden."
            )
            self._status_label.setText(
                "Cache ungültig — Biomarker/Pathologien geändert. Neuberechnung erforderlich."
            )
            return

        manifest = cache.load_checkpoint_manifest()
        manifest_info = ""
        if manifest:
            manifest_info = (
                f" | VCG: {manifest.get('vcg_method', '?')} "
                f"| {manifest.get('n_records', '?')} vollstaendig verarbeitet"
            )

        self.pipeline_widget.append_log(
            f"Fortschritt gefunden: {n_cached:,} gecachte Records{manifest_info}"
        )
        self._analysis_status.setText(f"{n_cached:,} gecachte Records")

        if cache.has_analysis_checkpoint():
            checkpoint = cache.load_analysis_checkpoint()
            n_partial = len(checkpoint.get("results", [])) if checkpoint else 0
            vcg_method = checkpoint.get("vcg_method", "?") if checkpoint else "?"
            if vcg_method == self.config.vcg_method:
                self.pipeline_widget.append_log(
                    f"Analyse-Checkpoint gefunden: {n_partial:,} Teilergebnisse - "
                    "wird beim Analyse-Start geprueft und fortgesetzt"
                )
            else:
                self.pipeline_widget.append_log(
                    f"Analyse-Checkpoint gefunden: {n_partial:,} Teilergebnisse - "
                    f"passt nicht zur aktuellen VCG-Methode ({vcg_method} != {self.config.vcg_method})"
                )

        # Analyse-Ergebnisse laden falls vorhanden
        if cache.has_analysis_run():
            try:
                run = cache.load_analysis_run()
                if run is not None and len(run.results) > 0:
                    self._last_analysis = run
                    self.analysis_widget.display_results(run)
                    self.evidence_widget.set_analysis_run(run)
                    self.performance_widget.set_analysis_run(run)
                    n_sig = sum(1 for r in run.results if r.is_significant)
                    self.pipeline_widget.append_log(
                        f"Analyse wiederhergestellt: {len(run.results)} Paare, {n_sig} signifikant"
                    )
                    self._status_label.setText(
                        f"{n_cached:,} Records + Analyse geladen — bereit."
                    )
                    self._analysis_status.setText(
                        f"{n_cached:,} Records | {n_sig} signif. Paare"
                    )
                    return
                if run is not None:
                    self.pipeline_widget.append_log(
                        "Leerer Analyse-Cache ignoriert - 'Analyze Processed' berechnet neu"
                    )
            except Exception as exc:
                self.pipeline_widget.append_log(f"Analyse-Cache Fehler: {exc}")

        self._status_label.setText(
            f"{n_cached:,} gecachte Records. 'Analyze Processed' oder Pipeline fortsetzen."
        )

    def closeEvent(self, event):
        atlas = getattr(getattr(self, "mimic_widget", None), "results_atlas", None)
        if atlas is not None:
            if not getattr(self, "_atlas_shutdown_connected", False):
                atlas.idle.connect(self.close)
                self._atlas_shutdown_connected = True
            if not atlas.request_shutdown():
                # A read-only DuckDB/Qt worker is allowed to finish.  Never
                # terminate a running QThread or affect the separate run.
                event.ignore()
                return
        settings = QSettings("VCGBiomarker", "App")
        settings.setValue("mimic_window_geometry", self.saveGeometry())
        if self.pipeline.cache.cache_dir:
            settings.setValue("cache_dir", str(self.pipeline.cache.cache_dir))
        if self.config.ptbxl_path:
            settings.setValue("ptbxl_path", str(self.config.ptbxl_path))
        settings.setValue("vcg_method", self.config.vcg_method)
        self._save_analysis_settings()

        if self._thread is not None and self._thread.isRunning():
            self.pipeline.request_cancel()
            self.pipeline_widget.append_log("Fenster wird geschlossen - laufender Job wird beendet und checkpointed...")
            self._thread.quit()
            if not self._thread.wait(10000):
                self._thread.terminate()
                self._thread.wait(3000)

        event.accept()
