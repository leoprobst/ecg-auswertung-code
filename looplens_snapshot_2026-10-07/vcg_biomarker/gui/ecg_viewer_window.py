from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QToolBar, QStatusBar,
    QLabel, QMessageBox, QFileDialog, QInputDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ..config import AppConfig
from ..core.pipeline import ProcessedRecord
from ..core.reference import ReferenceLoopStore
from ..core.report import ClinicalReport


class EKGViewerWindow(QMainWindow):
    """
    Standalone window for inspecting a single ECG/VCG record.

    Receives processed records from MainWindow via display_record() and
    receives live threshold updates via update_thresholds() whenever the
    main analysis pipeline completes a new analysis run.
    """

    def __init__(
        self,
        config: AppConfig,
        reference_store: ReferenceLoopStore,
        pipeline=None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self._reference_store = reference_store
        self._pipeline = pipeline          # ParallelPipeline for "load by ID"
        self._last_processed_record: Optional[ProcessedRecord] = None
        self._last_analysis_run = None
        self._report = ClinicalReport()

        self.setWindowTitle("ECG Viewer")
        self.setMinimumSize(1300, 850)

        self._setup_toolbar()
        self._setup_tabs()
        self._setup_statusbar()
        self._connect_internal_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_toolbar(self):
        tb = QToolBar("ECG Viewer")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        load_id = QAction("Load by ID", self)
        load_id.setShortcut("Ctrl+G")
        load_id.setToolTip(
            "Load an ECG from the PTB-XL dataset by entering its numeric ECG ID  (Ctrl+G)"
        )
        load_id.triggered.connect(self._on_load_by_id)
        tb.addAction(load_id)

        open_file = QAction("Open File", self)
        open_file.setShortcut("Ctrl+O")
        open_file.setToolTip(
            "Open an ECG file from disk — supported formats: WFDB (.hea), EDF, CSV, NPY  (Ctrl+O)"
        )
        open_file.triggered.connect(self._on_open_file)
        tb.addAction(open_file)

        tb.addSeparator()

        recompute = QAction("Recompute Biomarkers", self)
        recompute.setShortcut("Ctrl+R")
        recompute.setToolTip(
            "Recompute all 34 biomarkers using the current wave-marker positions.\n"
            "Use after manually adjusting P/QRS/T markers in the ECG Annotation tab.  (Ctrl+R)"
        )
        recompute.triggered.connect(self._on_recompute_biomarkers)
        tb.addAction(recompute)

        tb.addSeparator()

        load_analysis = QAction("Load Analysis…", self)
        load_analysis.setToolTip(
            "Load a previously saved analysis run (.pkl) to enable diagnostic thresholds\n"
            "without running the full pipeline again"
        )
        load_analysis.triggered.connect(self._on_load_analysis)
        tb.addAction(load_analysis)

        report = QAction("PDF Report", self)
        report.setToolTip("Generate a clinical PDF report for the currently loaded ECG record")
        report.triggered.connect(self._on_generate_report)
        tb.addAction(report)

    def _setup_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.setCentralWidget(self.tab_widget)

        from .widgets.annotation_widget import AnnotationWidget
        from .widgets.vcg_widget import VCGWidget
        from .widgets.diagnostic_widget import DiagnosticWidget
        from .widgets.caliper_widget import CaliperWidget, CaliperManager

        self.annotation_widget = AnnotationWidget(self.config)
        self.vcg_widget = VCGWidget(self.config)
        self.diagnostic_widget = DiagnosticWidget(self.config)

        self._caliper_manager = CaliperManager()
        self.caliper_widget = CaliperWidget()
        self.caliper_widget.set_manager(self._caliper_manager)
        self.annotation_widget.set_caliper_manager(self._caliper_manager)

        self.vcg_widget.set_reference_store(self._reference_store)

        self.tab_widget.addTab(self.annotation_widget, "ECG Annotation")
        self.tab_widget.setTabToolTip(0,
            "12-lead ECG display with draggable P/QRS/T wave markers.\n"
            "Drag markers to correct delineation, then click 'Recompute Biomarkers'."
        )

        self.tab_widget.addTab(self.vcg_widget, "VCG Viewer")
        self.tab_widget.setTabToolTip(1,
            "3-D vectorcardiogram loop viewer (Frank leads X/Y/Z).\n"
            "Switch between 3D, Frontal, Horizontal and Sagittal projections."
        )

        self.tab_widget.addTab(self.diagnostic_widget, "Diagnosis")
        self.tab_widget.setTabToolTip(2,
            "Biomarker traffic-light table and pathology probability scores.\n"
            "Thresholds are learned from the dataset analysis in the main window."
        )

        self.tab_widget.addTab(self.caliper_widget, "Calipers")
        self.tab_widget.setTabToolTip(3,
            "Manual interval measurements placed directly on the ECG annotation plot."
        )

    def _setup_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._ecg_info_label = QLabel("No record loaded")
        sb.addWidget(self._ecg_info_label, stretch=1)

        self._threshold_label = QLabel("Thresholds: not yet loaded")
        self._threshold_label.setToolTip(
            "Indicates whether diagnostic thresholds have been learned from the\n"
            "dataset analysis in the main window. Run 'Analyze' there to populate."
        )
        self._threshold_label.setStyleSheet("color: #999; padding-right: 8px;")
        sb.addPermanentWidget(self._threshold_label)

    def _connect_internal_signals(self):
        self.annotation_widget.sample_selected.connect(self.vcg_widget.highlight_sample)
        self.vcg_widget.time_index_selected.connect(self.annotation_widget.highlight_sample)
        self.annotation_widget.biomarker_recompute_requested.connect(self._on_recompute_biomarkers)
        # Caliper wiring: both tabs share one CaliperManager; keep table in sync
        self.caliper_widget.add_caliper_requested.connect(self.annotation_widget._start_caliper_placement)
        self.caliper_widget.measurements_cleared.connect(self.annotation_widget.clear_calipers)
        self.annotation_widget.caliper_updated.connect(self.caliper_widget._update_table)

    # ------------------------------------------------------------------
    # Public API — called by MainWindow
    # ------------------------------------------------------------------

    def display_record(self, pr: ProcessedRecord):
        """Show a fully processed ECG record across all viewer tabs."""
        self._last_processed_record = pr
        rec = pr.ecg_record

        if pr.preprocessed and pr.delineation:
            signal = pr.preprocessed.get("cleaned", rec.ecg_signal)
            if signal is not None:
                self.annotation_widget.display_ecg(signal, rec.sampling_rate, pr.delineation)

        if pr.vcg_result and pr.delineation:
            self.vcg_widget.display_vcg(
                pr.vcg_result,
                pr.delineation,
                rec.sampling_rate,
                age=rec.age,
                sex=rec.sex,
            )

        if pr.biomarker_results:
            # Apply currently loaded analysis thresholds before setting results
            # so diagnosis runs with the correct thresholds on the first call
            if self._last_analysis_run is not None:
                self.diagnostic_widget.learn_from_analysis(self._last_analysis_run)
            self.diagnostic_widget.set_biomarker_results(
                pr.biomarker_results,
                age=rec.age,
                sex=rec.sex,
            )

        # Build status bar info string
        parts: list[str] = []
        if rec.ecg_id:
            parts.append(f"ID {rec.ecg_id}")
        if rec.age:
            parts.append(f"Age {rec.age}")
        if rec.sex:
            parts.append(f"Sex {rec.sex}")
        if rec.diagnostic_superclass:
            parts.append("Class: " + ", ".join(rec.diagnostic_superclass))
        self._ecg_info_label.setText("  ".join(parts) if parts else "Record loaded")

        label = f"ECG ID {rec.ecg_id}" if rec.ecg_id else Path(str(rec.ecg_id or "")).name or "External ECG"
        self.setWindowTitle(f"ECG Viewer — {label}")

        self.tab_widget.setCurrentIndex(0)
        self.raise_()
        self.activateWindow()

    def update_thresholds(self, run, source: str = "") -> None:
        """
        Push learned thresholds into the DiagnosticWidget.
        Called by MainWindow on analysis completion, on startup (from cache),
        or manually via 'Load Analysis…'.
        """
        self._last_analysis_run = run
        self.diagnostic_widget.learn_from_analysis(run)
        n_sig = sum(1 for r in run.results if r.is_significant)
        label = f"Thresholds: {n_sig} sig. pairs"
        if source:
            label += f"  [{source}]"
        self._threshold_label.setText(label)
        self._threshold_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding-right: 8px;")

        # Re-run diagnosis with fresh thresholds if a record is already loaded
        if self._last_processed_record and self._last_processed_record.biomarker_results:
            pr = self._last_processed_record
            self.diagnostic_widget.set_biomarker_results(
                pr.biomarker_results,
                age=pr.ecg_record.age,
                sex=pr.ecg_record.sex,
            )

    # ------------------------------------------------------------------
    # Internal actions
    # ------------------------------------------------------------------

    def _on_load_by_id(self):
        if self._pipeline is None or self._pipeline.data_loader is None:
            QMessageBox.warning(
                self,
                "No Dataset Loaded",
                "The PTB-XL dataset is not yet loaded in the main window.\n\n"
                "Please use 'Open PTB-XL Dataset' in the main window first.",
            )
            return

        ecg_id, ok = QInputDialog.getInt(
            self,
            "Load ECG by ID",
            "Enter ECG ID (1 – 21837):",
            min=1,
            max=99999,
        )
        if not ok:
            return

        self._ecg_info_label.setText(f"Loading ECG ID {ecg_id}…")
        try:
            record = self._pipeline.data_loader.get_record(ecg_id, load_signal=True)
            if not record.has_signal:
                QMessageBox.warning(
                    self, "No Signal Data",
                    f"ECG ID {ecg_id} has no signal data in the dataset."
                )
                self._ecg_info_label.setText("Load failed")
                return
            pr = self._run_pipeline(record, label=f"ECG ID {ecg_id}")
            self.display_record(pr)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load ECG ID {ecg_id}:\n{exc}")
            self._ecg_info_label.setText("Load failed")

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ECG File",
            "",
            "ECG Files (*.hea *.edf *.dcm *.dicom *.csv *.tsv *.npy *.npz *.txt);;"
            "WFDB Header (*.hea);;"
            "EDF (*.edf);;"
            "DICOM Waveform (*.dcm *.dicom);;"
            "CSV / TSV (*.csv *.tsv *.txt);;"
            "NumPy (*.npy *.npz);;"
            "All Files (*)",
        )
        if not path:
            return

        from ..core.external_loader import ExternalECGMeta, load_external_ecg, make_external_record
        from .widgets.external_ecg_dialog import ExternalECGDialog

        dlg = ExternalECGDialog(detected_fs=500, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        meta = dlg.get_meta()
        try:
            signal = load_external_ecg(path, meta)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Could not read ECG file:\n{exc}\n\n"
                "Supported formats: WFDB, EDF, DICOM Waveform, CSV/TSV and NumPy research arrays.",
            )
            return

        try:
            record = make_external_record(signal, meta)
            pr = self._run_pipeline(record, label=Path(path).name)
            self.display_record(pr)
        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", f"Pipeline failed:\n{exc}")

    def _run_pipeline(self, record, label: str = "ECG") -> ProcessedRecord:
        """Preprocess + VCG transform + delineate + compute all biomarkers."""
        from ..core.preprocessing import (
            construct_robust_morphology_record,
            preprocess_record,
        )
        from ..core.vcg_transform import VCGTransformRegistry
        from ..core.delineation.fusion_delineator import FusionDelineator
        from ..core.biomarkers.base import BiomarkerRegistry
        from ..core.pathologies.registry import PathologyRegistry

        self._ecg_info_label.setText(f"Processing {label}…")

        preprocessed = preprocess_record(record.ecg_signal, record.sampling_rate, record.r_peaks)
        vcg_delineation = VCGTransformRegistry.transform(
            preprocessed["delineation"],
            self.config.vcg_method,
        )

        delineator = FusionDelineator(
            weights=self.config.delineation_weights,
            conflict_threshold=self.config.delineation_conflict_threshold,
        )
        delineation = delineator.delineate_record(
            ecg_12lead=preprocessed["cleaned"],
            r_peaks=preprocessed["r_peaks"],
            fs=record.sampling_rate,
            ecg_id=record.ecg_id,
            vcg_xyz=vcg_delineation.xyz,
        )
        robust_morphology, template_diagnostics = (
            construct_robust_morphology_record(
                preprocessed["morphology"],
                delineation,
            )
        )
        vcg = VCGTransformRegistry.transform(
            robust_morphology,
            self.config.vcg_method,
        )
        record_vcg = VCGTransformRegistry.transform(
            preprocessed["morphology"],
            self.config.vcg_method,
        )
        preprocessed["morphology_template"] = robust_morphology
        preprocessed["morphology_template_diagnostics"] = template_diagnostics
        biomarker_results = BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=robust_morphology,
            vcg_xyz=vcg.xyz,
            fs=record.sampling_rate,
            record_ecg_12lead=preprocessed["morphology"],
            record_raw_ecg_12lead=record.ecg_signal,
            record_acquisition={"original_sampling_rate_hz": float(record.sampling_rate)},
            record_vcg_xyz=record_vcg.xyz,
        )
        pathology_labels = PathologyRegistry.check_all(record.scp_codes)

        return ProcessedRecord(
            ecg_record=record,
            preprocessed=preprocessed,
            vcg_result=vcg,
            delineation=delineation,
            biomarker_results=biomarker_results,
            biomarker_dict={br.name: br.value for br in biomarker_results},
            pathology_labels=pathology_labels,
        )

    def _on_recompute_biomarkers(self):
        """Recompute biomarkers from the current (possibly edited) delineation."""
        if self._last_processed_record is None:
            return

        pr = self._last_processed_record
        modified = self.annotation_widget.get_modified_delineation()
        if modified is None or pr.preprocessed is None:
            return

        from ..core.biomarkers.base import BiomarkerRegistry
        from ..core.preprocessing import construct_robust_morphology_record
        from ..core.vcg_transform import VCGTransformRegistry

        record_morphology = pr.preprocessed.get(
            "morphology",
            pr.ecg_record.ecg_signal,
        )
        robust_morphology, template_diagnostics = (
            construct_robust_morphology_record(
                record_morphology,
                modified,
            )
        )
        vcg = VCGTransformRegistry.transform(
            robust_morphology,
            self.config.vcg_method,
        )
        record_vcg = VCGTransformRegistry.transform(
            record_morphology,
            self.config.vcg_method,
        )

        new_results = BiomarkerRegistry.compute_all(
            delineation=modified,
            ecg_12lead=robust_morphology,
            vcg_xyz=vcg.xyz,
            fs=pr.ecg_record.sampling_rate,
            record_ecg_12lead=record_morphology,
            record_raw_ecg_12lead=pr.ecg_record.ecg_signal,
            record_acquisition={"original_sampling_rate_hz": float(pr.ecg_record.sampling_rate)},
            record_vcg_xyz=record_vcg.xyz,
        )
        pr.preprocessed["morphology_template"] = robust_morphology
        pr.preprocessed["morphology_template_diagnostics"] = (
            template_diagnostics
        )
        pr.vcg_result = vcg
        pr.biomarker_results = new_results
        pr.biomarker_dict = {br.name: br.value for br in new_results}

        self.diagnostic_widget.set_biomarker_results(
            new_results,
            age=pr.ecg_record.age,
            sex=pr.ecg_record.sex,
        )
        current = self._ecg_info_label.text().removesuffix(" [recomputed]")
        self._ecg_info_label.setText(current + " [recomputed]")

    def _on_load_analysis(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Analysis Run",
            "",
            "Analysis Files (*.pkl);;All Files (*)",
        )
        if not path:
            return
        try:
            import pickle
            with open(path, "rb") as f:
                run = pickle.load(f)
            from pathlib import Path as _Path
            self.update_thresholds(run, source=_Path(path).name)
            if self._last_processed_record and self._last_processed_record.biomarker_results:
                pr = self._last_processed_record
                self.diagnostic_widget.set_biomarker_results(
                    pr.biomarker_results,
                    age=pr.ecg_record.age,
                    sex=pr.ecg_record.sex,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load analysis file:\n{exc}")

    def _on_generate_report(self):
        if self._last_processed_record is None:
            QMessageBox.warning(self, "No Record", "No ECG record is currently loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            pr = self._last_processed_record
            result = self._report.generate(
                processed_record=pr,
                vcg_result=pr.vcg_result,
                analysis_results=None,
                output_path=Path(path),
            )
            if result:
                QMessageBox.information(self, "Report Saved", f"PDF report saved to:\n{path}")
            else:
                QMessageBox.warning(
                    self, "Report Failed",
                    "Report generation failed — matplotlib may not be installed."
                )
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", f"Could not generate report:\n{exc}")

    def closeEvent(self, event):
        event.accept()
