from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSettings,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QDesktopServices, QFontMetrics
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..runtime_status import supervisor_runtime_notice
from ..lab_controller import LabProcessController
from ..lab_marker_viewmodel import finding_validation_metrics
from ..results_atlas import ResultsAtlasWidget
from ..results_atlas.charts import auc_matrix_rgba
from .expert_ecg_lab import ExpertECGLab

from ...core.external_loader import (
    ECGImportMappingRequired,
)
from ...core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_DATASETS,
    DELINEATION_SUITE_VERSION,
    delineation_implementation_hash,
)
from ...mimic.acceptance import DEFAULT_GROUP_VCG_ENDPOINTS
from ...mimic.biomarker_audit import (
    BIOMARKER_AUDIT_VERSION,
    biomarker_audit_implementation_hash,
)
from ...mimic.config import MimicProjectConfig, load_or_discover_config
from ...mimic.ecg_lab import ECGCaseLibrary
from ...mimic.feature_extraction import (
    FEATURE_STAGE_VERSION,
    feature_implementation_hash,
)
from ...mimic.feature_roles import (
    METHODOLOGICAL_CONTROL_FEATURE_IDS,
    is_candidate_biomarker,
)
from ...mimic.group_vcg import GROUP_VCG_STAGE_VERSION, group_vcg_artifact_dir
from ...mimic.jobs import JobStore
from ...mimic.resource_grants import grant_status_text
from ...mimic.models import JobState, ResultQuery
from ...mimic.panels import default_feature_panels
from ...mimic.run_lineage import (
    AnalysisLineage,
    analysis_is_frozen,
    resolve_current_analysis,
)
from ...mimic.presets import (
    STUDY_PRESET_ID,
    get_analysis_preset,
)
from ...mimic.result_query import ResultRepository
from ...mimic.result_query import RESULT_QUERY_VERSION
from ...mimic.run_integrity import (
    assert_development_run_mutable,
    immutable_development_run_markers,
)
from ...mimic.presentation import (
    comparison_effect_frame,
    comparison_effect_labels,
)
from ...mimic.resources import (
    critical_available_memory_gb,
    current_cpu_percent,
    detect_hardware,
    detect_memory_status,
    global_memory_budget_gb,
)
from ...mimic.ptb_transfer import DEFAULT_PTB_TRANSFER_RUN_ID
from ...mimic.supervisor import ensure_supervisor
from ...mimic.services import (
    MANAGED_SERVICE_STAGE_IDS,
    MIMIC_BACKEND_CONTRACT_VERSION,
)
from ...mimic.stage_exposure import individually_runnable_stage_ids
from ...mimic.storage import StageManifest
from ...mimic.user_guidance import (
    GLOSSARY_DE,
    STAGE_GROUP_NAMES_DE,
    biomarker_display_name,
    biomarker_tooltip,
    clinical_display_term,
    clinical_display_terms,
    stage_descriptor,
)
from ...mimic.validation_provenance import validation_implementation_hash
from ...mimic.validation import VALIDATION_STAGE_VERSION
from ...mimic.phenome import PHENOME_STAGE_VERSION
from ...mimic.workflow import (
    GUIDED_WORKFLOW_VERSION,
    enqueue_guided_study,
    guided_study_spec,
    guided_service_stage_ids,
)
from ..run_progress import CalibratedJobProgress, calibrated_job_progress
from ..run_progress import development_record_count
from .external_ecg_dialog import ExternalECGDialog, LeadMappingDialog
from .paged_table import PagedParquetTable
from .recovery_reference_widget import RecoveryReferenceWidget
from .paired_effect_widget import PairedEffectWidget
from .dose_pair_widget import DosePairWidget
from .joint_lab_coverage_widget import JointLabCoverageWidget


RESEARCH_NOTICE = (
    "Forschungsmarker und klinischer Referenzvergleich. "
    "Keine Einzelfalldiagnose oder Behandlungsempfehlung."
)
LOGGER = logging.getLogger(__name__)
DEFAULT_DEVELOPMENT_RUN_ID = "mimic-clinical-development-v3"
# The "Live-Features" preview panel scans this many of the most recent
# feature-extraction partition files (part-000000.parquet, sequentially
# numbered -- see feature_extraction.py) via DuckDB on every cohort
# refresh. A long-lived project accumulates thousands of these small
# files (real observed case: 2,562 files / ~2.4 GB for one panel combo);
# scanning all of them via read_parquet(..., union_by_name=true) plus a
# QUANTILE_CONT histogram blocked the GUI thread for ~90 seconds. This
# is a *live* progress preview, not a definitive full-dataset summary,
# so capping to a recent, bounded sample keeps it fast regardless of
# how large the underlying project grows.
LIVE_FEATURE_PREVIEW_PART_LIMIT = 200
GUI_BACKEND_CONTRACT_VERSION = "1.9.0"
POPULATION_FOREST_HR_MIN = 0.125
POPULATION_FOREST_HR_MAX = 8.0
# Derived from what the service layer accepts, never hand-maintained. The
# hand-written tuple that stood here had drifted: twelve runnable stages were
# missing from the dropdown, ``marker-redundancy`` among them, and four more
# were offered under the fallback label "Technische Aufgabe" -- including
# ``screen-broad-axes``, the largest analysis in the project. Nothing failed,
# because the contract check only asks whether every offered stage is runnable,
# not whether every runnable stage is offered.
TECHNICAL_STAGE_IDS = individually_runnable_stage_ids()

TAB_PROJECT = 0
TAB_ANALYSIS = 1
TAB_RESULTS = 2
TAB_ECG_LAB = 3
TAB_REPORTS = 4


def _artifact_revision(paths: tuple[Path, ...]) -> int:
    """Cheap stable-in-process key for independently completed result stages."""
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamps.append((0, 0))
    return hash(tuple(stamps))


def _population_forest_coordinates(
    hazard_ratio: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return finite, clipped log2 coordinates for a readable forest plot.

    A single unstable Cox estimate can otherwise expand a linear axis by
    several orders of magnitude and visually collapse every clinically
    plausible estimate.  The returned masks retain that information: the GUI
    marks every clipped point or confidence interval rather than silently
    dropping it.
    """

    hazard = np.asarray(hazard_ratio, dtype=float)
    low = np.asarray(ci_low, dtype=float)
    high = np.asarray(ci_high, dtype=float)
    valid = (
        np.isfinite(hazard)
        & np.isfinite(low)
        & np.isfinite(high)
        & (hazard > 0.0)
        & (low > 0.0)
        & (high > 0.0)
    )
    hazard = hazard[valid]
    low = low[valid]
    high = high[valid]
    lower = float(np.log2(POPULATION_FOREST_HR_MIN))
    upper = float(np.log2(POPULATION_FOREST_HR_MAX))
    log_hazard = np.log2(hazard)
    log_low = np.log2(low)
    log_high = np.log2(high)
    x = np.clip(log_hazard, lower, upper)
    left_edge = np.clip(log_low, lower, upper)
    right_edge = np.clip(log_high, lower, upper)
    left = np.maximum(0.0, x - left_edge)
    right = np.maximum(0.0, right_edge - x)
    clipped_low = (log_hazard < lower) | (log_low < lower)
    clipped_high = (log_hazard > upper) | (log_high > upper)
    return valid, x, left, right, clipped_low, clipped_high


def _gwas_context_presentation(payload: dict) -> tuple[str, bool]:
    """Human-readable scope statement for the deliberately limited GWAS arm."""

    if not payload:
        return (
            "Noch kein GWAS-Machbarkeitsinventar. Die Genetik ist ein "
            "separater Kontextarm und beeinflusst den MIMIC-Pilot nicht.",
            False,
        )
    ready = bool(payload.get("source_inventory_complete"))
    source_files = int(payload.get("n_summary_files", 0) or 0)
    ancestry = ", ".join(
        str(item) for item in payload.get("available_ancestry_arms", ())
    ) or "nicht ausgewiesen"
    if not ready:
        missing = ", ".join(
            str(item) for item in payload.get("missing_expected_accessions", ())
        ) or "unbekannt"
        return (
            f"GWAS-Kontext unvollständig: {source_files} Quelldateien; "
            f"fehlende Studien: {missing}. Keine genetischen Claims.",
            False,
        )
    return (
        f"{source_files} harmonisierte Summary-Statistik-Dateien zum räumlichen "
        f"QRS-T-Winkel ({ancestry}). Nutzung nur als genetischer "
        "Architekturkontext. Mendelian Randomization und polygenetische Scores "
        "sind absichtlich nicht schätzbar: unabhängiges Outcome-GWAS, "
        "Individualgenotypen und LD-Referenz fehlen. Kausal- oder "
        "Leitlinienclaims sind gesperrt.",
        True,
    )


class _SortableTableItem(QTableWidgetItem):
    def __init__(self, value: object):
        super().__init__(_display_value(value))
        self._sort_value = value

    def __lt__(self, other) -> bool:
        if isinstance(other, _SortableTableItem):
            left = _numeric_sort_value(self._sort_value)
            right = _numeric_sort_value(other._sort_value)
            if left is not None and right is not None:
                return left < right
        return self.text().casefold() < other.text().casefold()


class MimicProjectWidget(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, progress_callback=None):
        super().__init__(parent)
        self._dark_mode: bool = QSettings("VCGBiomarker", "App").value(
            "dark_mode", False, type=bool
        )
        self._themed_plots: list = []
        startup_progress = progress_callback or (lambda message, fraction: None)
        # psutil.cpu_percent's first call in a process always returns 0.0
        # (it measures the delta since the previous call) -- prime it here
        # so the first real reading in refresh_jobs() is meaningful.
        current_cpu_percent()
        startup_progress("Konfiguration wird ermittelt...", 0.0)
        self.config = load_or_discover_config()
        self.config.ensure_directories()
        startup_progress("Code-Stand wird geprüft...", 0.10)
        self._current_feature_code_hash = feature_implementation_hash()
        self._current_validation_code_hash = validation_implementation_hash()
        self._current_biomarker_audit_hash = (
            biomarker_audit_implementation_hash()
        )
        self._current_delineation_hash = delineation_implementation_hash()
        startup_progress("Job- und Fallverzeichnisse werden geöffnet...", 0.20)
        self.job_store = JobStore.for_project(self.config)
        self.case_library = ECGCaseLibrary(self.config)
        self.result_repository = ResultRepository(
            self.config,
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        )
        self._results_gui_ready = False
        self._selected_job_id = ""
        self._active_review_path: Optional[Path] = None
        self._last_ui_artifact_signature: tuple = ()
        self._benchmark_mtime_ns = 0
        self._current_case_fiducials: dict[str, int] = {}
        self._current_case_fiducial_details: dict[str, dict] = {}
        self._current_case_segments: list[dict] = []
        self._current_case_array_path: Optional[Path] = None
        self._current_case_sample_count = 0
        self._current_case_ecg: Optional[np.ndarray] = None
        self._current_case_r_peaks: Optional[np.ndarray] = None
        self._current_case_sampling_rate: float = 0.0
        self._current_case_record_vcg: Optional[np.ndarray] = None
        self._current_case_qt_rr: tuple = (None, None, None)
        startup_progress("Oberfläche wird aufgebaut...", 0.30)
        self._setup_ui(progress=startup_progress)
        self._load_config_into_fields()
        # Switch to the broad-discovery perspective now that every widget
        # its currentChanged handler touches has been built; doing this
        # inside _results_workspace() itself would fire the handler before
        # the rest of __init__ (e.g. self._primary_result_offsets) exists.
        self._focus_discovery_perspective()
        startup_progress("Ergebnisse und Auftragsstatus werden geladen...", 0.60)
        self.refresh_all(progress=startup_progress)
        self._results_gui_ready = True
        startup_progress("Bereit.", 1.0)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_background_state)
        self._poll_timer.start(1_500)

    def _setup_ui(
        self,
        *,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.window().setStyleSheet(theme.generate_stylesheet(self._dark_mode))
        # The workspaces contain information-dense tables and plots whose
        # combined minimum height can exceed a laptop display.  Keep the
        # whole shell inside one resizeable viewport so 1366x768 and scaled
        # Windows displays remain usable instead of forcing an off-screen
        # top-level window.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_scroll = QScrollArea(self)
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setFrameShape(QFrame.Shape.NoFrame)
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header_surface = QFrame()
        header_surface.setObjectName("mimicHeaderSurface")
        header = QHBoxLayout(header_surface)
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("VCG Feature Scanner · Klinische Validierung")
        title.setObjectName("mimicHeaderTitle")
        header.addWidget(title)
        header.addStretch(1)
        # Small breathing accent dot, shown only while refresh_all() is
        # loading cohort/result data -- a lighter-weight echo of the
        # startup splash screen's pulse animation for in-app loading
        # states, not a full progress readout.
        self.busy_indicator = QLabel()
        self.busy_indicator.setObjectName("mimicBusyDot")
        self.busy_indicator.setFixedSize(9, 9)
        self.busy_indicator.hide()
        self._busy_opacity_effect = QGraphicsOpacityEffect(self.busy_indicator)
        self.busy_indicator.setGraphicsEffect(self._busy_opacity_effect)
        self._busy_animation = QPropertyAnimation(
            self._busy_opacity_effect, b"opacity", self
        )
        self._busy_animation.setDuration(1400)
        self._busy_animation.setKeyValueAt(0.0, 0.3)
        self._busy_animation.setKeyValueAt(0.5, 1.0)
        self._busy_animation.setKeyValueAt(1.0, 0.3)
        self._busy_animation.setLoopCount(-1)
        from ..motion import current_motion_policy
        current_motion_policy().changed.connect(self._on_motion_preference_changed)
        header.addWidget(self.busy_indicator)
        self.project_status = QLabel("Projekt wird geprüft")
        self.project_status.setObjectName("mimicTextMuted")
        header.addWidget(self.project_status)
        self.dark_mode_toggle = QCheckBox("Dunkel")
        self.dark_mode_toggle.setChecked(self._dark_mode)
        self.dark_mode_toggle.setToolTip(
            "Zwischen hellem und dunklem Design wechseln -- die Wahl wird "
            "gespeichert und beim nächsten Start wiederhergestellt."
        )
        self.dark_mode_toggle.toggled.connect(self.apply_theme)
        header.addWidget(self.dark_mode_toggle)
        self.motion_toggle = QCheckBox("Bewegung reduzieren")
        self.motion_toggle.setChecked(current_motion_policy().reduced)
        self.motion_toggle.setToolTip("Pausiert automatische 3D-Bewegung und dekorative Übergänge; Messwerte bleiben unverändert.")
        self.motion_toggle.toggled.connect(current_motion_policy().set_reduced)
        current_motion_policy().changed.connect(self.motion_toggle.setChecked)
        header.addWidget(self.motion_toggle)
        layout.addWidget(header_surface)

        # Frozen production jobs retain their original run identity while GUI
        # development changes the current fingerprint. Keep their read-only
        # status visible in every workspace, not only in the setup form.
        self.background_analysis_status = QLabel('Projektweite Hintergrundjobs werden geladen …')
        self.background_analysis_status.setWordWrap(True)
        self.background_analysis_status.setTextFormat(Qt.TextFormat.PlainText)
        self.background_analysis_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.background_analysis_status)

        notice = QLabel(RESEARCH_NOTICE)
        notice.setWordWrap(True)
        notice.setObjectName("mimicInfoBoxDanger")
        layout.addWidget(notice)

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.addTab(self._project_workspace(), "Projekt")
        if progress is not None:
            progress("Analyse-Arbeitsbereich wird aufgebaut...", 0.32)
        self.workspace_tabs.addTab(self._jobs_workspace(), "Analyse")
        if progress is not None:
            progress("Ergebnis-Arbeitsbereich wird aufgebaut...", 0.38)
        self.workspace_tabs.addTab(self._results_workspace(), "Ergebnisse")
        if progress is not None:
            progress("ECG-Labor wird aufgebaut...", 0.53)
        self.workspace_tabs.addTab(self._ecg_lab_workspace(), "ECG-Labor")
        if progress is not None:
            progress("Berichte werden aufgebaut...", 0.57)
        self.workspace_tabs.addTab(self._reports_workspace(), "Berichte")
        # Appended: existing project/analysis/results/lab/report indices stay stable.
        from ..simulation_workspace import SimulationWorkspace
        self.simulation_workspace = SimulationWorkspace(self, target_provider=self._simulation_target)
        self.simulation_workspace.labSignalReady.connect(self._import_simulated_ecg)
        self.workspace_tabs.addTab(self.simulation_workspace, "Simulation")
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.simulation_workspace.shutdown)
        for index, tooltip in enumerate(
            (
                "Pfade, Datengrundlage, klinische Referenzen und Qualität",
                "Geführte Analyse, laufende Arbeit und Pause/Fortsetzen",
                "Pathologien, Biomarker, Kriterien und neue Zusammenhänge",
                "Lokaler Forschungsimport einzelner 12-Kanal-ECGs",
                "Druckbare Berichte, Exporte und lokale Fallprüfung",
                "Isolierte elektrophysiologische Forschung: Gewebe, Elektroden und Modellhypothesen",
            )
        ):
            self.workspace_tabs.setTabToolTip(index, tooltip)
        layout.addWidget(self.workspace_tabs, 1)

        self._themed_plots = [
            self.window_plot,
            self.qc_plot,
            self.assessability_plot,
            self.feature_failure_plot,
            self.live_feature_plot,
            self.performance_plot,
            self.roc_plot,
            self.pr_plot,
            self.distribution_plot,
            self.operating_plot,
            self.baseline_plot,
            self.matrix_plot,
            self.stability_plot,
            self.sensitivity_plot,
            self.calibration_plot,
            self.decision_plot,
            self.evidence_ladder_plot,
            self.transform_agreement_plot,
            self.population_mortality_plot,
            self.phenome_plot,
            self.broad_continuous_plot,
            self.broad_binary_plot,
            self.transfer_plot,
            self.ecg_plot,
            self.rr_tachogram_plot,
            self.poincare_plot,
            self.beat_overlay_plot,
            self.qt_rr_plot,
            self.research_results_view.effect,
            self.research_results_view.calibration,
            self.results_atlas.endpoint_plot,
            self.results_atlas.map_view.plot,
            self.results_atlas.discovery_view.domain_plot,
            self.results_atlas.marker_view.plot,
            self.results_atlas.model_quality_view.calibration_plot,
            self.results_atlas.model_quality_view.decision_plot,
            self.results_atlas.availability_view.funnel_plot,
            self.results_atlas.transport_view.plot,
            *(item[0] for item in self.group_vcg_projection_plots.values()),
        ]
        for plot in self._themed_plots:
            theme.apply_plot_theme(plot, self._dark_mode)
        for box in self.findChildren(QGroupBox):
            shadow = QGraphicsDropShadowEffect(box)
            shadow.setBlurRadius(18)
            shadow.setXOffset(0)
            shadow.setYOffset(2)
            shadow.setColor(QColor(0, 0, 0, 60))
            box.setGraphicsEffect(shadow)
        self.workspace_scroll.setWidget(shell)
        outer_layout.addWidget(self.workspace_scroll)

    def _animate_results_view_change(self, _index: int) -> None:
        from ..motion import current_motion_policy
        if current_motion_policy().reduced:
            self._results_view_opacity.setOpacity(1.0)
            return
        anim = QPropertyAnimation(self._results_view_opacity, b"opacity", self)
        anim.setDuration(180)
        anim.setStartValue(0.35)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._results_view_anim = anim
        anim.start()

    def apply_theme(self, dark: bool) -> None:
        self._dark_mode = dark
        if hasattr(self, "expert_lab"):
            self.expert_lab.set_dark(dark)
        self.window().setStyleSheet(theme.generate_stylesheet(dark))
        for plot in self._themed_plots:
            theme.apply_plot_theme(plot, dark)
        self._refresh_backend_sync_status()
        self.refresh_jobs()
        QSettings("VCGBiomarker", "App").setValue("dark_mode", dark)

    def _project_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.project_tabs = QTabWidget()
        self.project_tabs.setDocumentMode(True)
        self.project_tabs.addTab(
            self._overview_workspace(),
            "Start und Projektpfade",
        )
        self.project_tabs.addTab(
            self._cohort_workspace(),
            "Datengrundlage",
        )
        self.project_tabs.setTabToolTip(
            0,
            "Projekt prüfen und den nächsten sinnvollen Schritt sehen",
        )
        self.project_tabs.setTabToolTip(
            1,
            "Patienten, klinische Referenzen, ECG-Qualität und zeitliche Zuordnung",
        )
        layout.addWidget(self.project_tabs)
        return page

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    def _overview_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        paths = QGroupBox("Projektassistent")
        form = QFormLayout(paths)
        self.database_root = QLineEdit()
        self.database_root.setToolTip(
            "Lokaler Stammordner mit MIMIC-IV, MIMIC-IV-ED, "
            "MIMIC-IV-ECG und den strukturierten Echo-Dateien"
        )
        self.work_root = QLineEdit()
        self.work_root.setToolTip(
            "Lokaler Arbeitsordner für Parquet, DuckDB, Jobs, Runs und Reports"
        )
        self.mimic_note_root_field = QLineEdit()
        self.mimic_note_root_field.setPlaceholderText(
            "Optional -- leer lassen, wenn keine MIMIC-IV-Notes vorliegen"
        )
        self.mimic_note_root_field.setToolTip(
            "Stammordner mit discharge.csv/radiology.csv (MIMIC-IV-Notes). "
            "Separat lizenziertes Dataset, nicht Teil von 'Rohdaten' oben."
        )
        self.code15_root_field = QLineEdit()
        self.code15_root_field.setPlaceholderText("Optional -- CODE-15%-Ordner")
        self.code15_root_field.setToolTip(
            "Primärversorgungs-ECGs für externe Messvalidierung und "
            "ECG-Referenzverteilungen; nicht mit MIMIC gepoolt."
        )
        self.sami_trop_root_field = QLineEdit()
        self.sami_trop_root_field.setPlaceholderText("Optional -- SaMi-Trop-Ordner")
        self.sami_trop_root_field.setToolTip(
            "Seropositive Chagas-Kohorte mit prospektivem Mortalitäts-Follow-up."
        )
        form.addRow("Rohdaten", self._path_row(self.database_root))
        form.addRow("Arbeitsdaten", self._path_row(self.work_root))
        form.addRow("MIMIC-Notes", self._path_row(self.mimic_note_root_field))
        form.addRow("CODE-15%", self._path_row(self.code15_root_field))
        form.addRow("SaMi-Trop", self._path_row(self.sami_trop_root_field))
        path_actions = QHBoxLayout()
        check = self._button(
            "Projekt prüfen",
            "Pfade, Releases, Pflichtdateien, Schreibzugriff und freien "
            "Speicher prüfen",
            self.check_project,
            QStyle.StandardPixmap.SP_DialogApplyButton,
        )
        path_actions.addWidget(check)
        path_actions.addStretch(1)
        form.addRow("", path_actions)
        layout.addWidget(paths)

        readiness = QGroupBox("Bereitschaft")
        readiness_layout = QVBoxLayout(readiness)
        self.readiness_text = QLabel()
        self.readiness_text.setWordWrap(True)
        self.readiness_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        readiness_layout.addWidget(self.readiness_text)
        self.overview_metrics = self._table(
            ("Bereich", "Stand", "Vollständigkeit", "Hinweis")
        )
        self.overview_metrics.setMaximumHeight(420)
        readiness_layout.addWidget(self.overview_metrics)
        self.backend_sync_status = QLabel()
        self.backend_sync_status.setWordWrap(True)
        self.backend_sync_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.backend_sync_status.setToolTip(
            "Prüft zur Laufzeit, ob alle geführten und technischen "
            "GUI-Aktionen vom aktuell importierten Backend unterstützt "
            "werden. Nach einer Softwareaktualisierung muss die GUI neu "
            "gestartet werden."
        )
        readiness_layout.addWidget(self.backend_sync_status)
        layout.addWidget(readiness)

        next_step = QGroupBox("Nächster empfohlener Schritt")
        next_layout = QHBoxLayout(next_step)
        self.next_step_label = QLabel()
        self.next_step_label.setWordWrap(True)
        next_layout.addWidget(self.next_step_label, 1)
        self.next_step_button = self._button(
            "Öffnen",
            "Zum kanonischen Arbeitsbereich für den nächsten Schritt wechseln",
            self._open_next_step,
            QStyle.StandardPixmap.SP_ArrowForward,
        )
        next_layout.addWidget(self.next_step_button)
        layout.addWidget(next_step)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _jobs_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        setup = QGroupBox("Analysevorhaben")
        setup_layout = QFormLayout(setup)
        # No purpose dropdown any more.  It offered five options of which two
        # were invisible and one was a byte-identical duplicate, and it was
        # labelled as a thoroughness dial while its real effect was scope:
        # 320 markers against 1753, a five-fold change in the number of
        # hypotheses.  The study runs at full scope, and the only question
        # that remains -- how long to keep refining -- is answered while it
        # runs, in the refinement panel below.
        self.study_scope_label = QLabel(
            f"{get_analysis_preset(STUDY_PRESET_ID).display_name_de} — "
            "es gibt nichts auszuwählen."
        )
        self.study_scope_label.setWordWrap(True)
        self.study_scope_label.setObjectName("analysisScopeLabel")
        setup_layout.addRow("Umfang", self.study_scope_label)

        # The run name used to be typed here, with the warning "change it when
        # markers, evidence or configuration have changed". That put the
        # project's hardest question on the person least able to answer it:
        # the software computes to the byte whether a change alters results,
        # while a human has to remember which of fifty edits mattered. Now the
        # identity is derived and only shown.
        self.analysis_name_label = QLabel("wird ermittelt …")
        self.analysis_name_label.setObjectName("analysisNameLabel")
        self.analysis_name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.analysis_name_label.setToolTip(
            "Der Name wird aus dem abgeleitet, was das Ergebnis bestimmt: "
            "Marker-Code, Delineation, Signalaufbereitung, Konfiguration und "
            "Panels. Gleiche Grundlage heißt gleiche Analyse -- sie wird "
            "fortgesetzt. Ändert sich etwas, entsteht eine neue Generation, "
            "und die alte bleibt vollständig lesbar."
        )
        setup_layout.addRow("Analyse", self.analysis_name_label)

        self.analysis_lineage_label = QLabel()
        self.analysis_lineage_label.setWordWrap(True)
        self.analysis_lineage_label.setObjectName("mimicTextMuted")
        setup_layout.addRow("", self.analysis_lineage_label)

        # Free text, never part of the identity, and stored in the registry
        # rather than in the run directory so a frozen analysis stays
        # annotatable.
        self.analysis_notes = QLineEdit()
        self.analysis_notes.setObjectName("analysisNotes")
        self.analysis_notes.setPlaceholderText(
            "Optionale Notiz zu dieser Analyse — woran Sie sie wiedererkennen"
        )
        self.analysis_notes.editingFinished.connect(self._save_analysis_notes)
        setup_layout.addRow("Notiz", self.analysis_notes)

        self.preset_description = QLabel()
        self.preset_description.setWordWrap(True)
        self.preset_description.setObjectName("mimicTextMuted")
        setup_layout.addRow("", self.preset_description)
        self.analysis_status_label = QLabel()
        self.analysis_status_label.setWordWrap(True)
        setup_layout.addRow("Aktueller Stand", self.analysis_status_label)
        action_row = QHBoxLayout()
        self.study_action_button = self._button(
            "Analyse starten",
            "Startet automatisch alle erforderlichen Schritte in richtiger Reihenfolge; "
            "ein pausierter Lauf wird am letzten sicheren Checkpoint fortgesetzt.",
            self.start_or_resume_analysis,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        self.study_action_button.setObjectName("mainAnalysisAction")
        action_row.addWidget(self.study_action_button)
        action_row.addStretch(1)
        setup_layout.addRow("", action_row)
        layout.addWidget(setup)

        resources = QHBoxLayout()
        self.supervisor_status = QLabel("Supervisor: nicht verbunden")
        self.supervisor_status.setWordWrap(True)
        self.supervisor_status.setToolTip(
            "MIMIC-Projektjobs laufen unabhängig vom GUI-Fenster weiter. Nur neu "
            "gestartete Snapshot-Supervisoren verwenden eine eigene Quellcodekopie. "
            "Den angezeigten Isolationsstatus beachten; einen alten Supervisor erst "
            "nach Abschluss aller Jobs kontrolliert erneuern. Während "
            "des Laufs keine Pakete in der verwendeten Python-Umgebung aktualisieren "
            "und Produktionskonfiguration/-daten nicht verändern. Windows muss "
            "eingeschaltet und wach bleiben. Dies gilt nicht für die alte lokale Pipeline."
        )
        resources.addWidget(self.supervisor_status)
        resources.addStretch(1)
        self.resource_status = QLabel()
        resources.addWidget(self.resource_status)
        resources.addWidget(QLabel("CPU"))
        self.cpu_gauge = QProgressBar()
        self.cpu_gauge.setRange(0, 100)
        self.cpu_gauge.setTextVisible(False)
        self.cpu_gauge.setFixedWidth(80)
        self.cpu_gauge.setFixedHeight(10)
        self.cpu_gauge.setToolTip(
            "Systemweite CPU-Auslastung, alle 1,5s aktualisiert. Macht "
            "ungenutzte Rechenkapazität sichtbar, die reiner Text nicht zeigt."
        )
        resources.addWidget(self.cpu_gauge)
        resources.addWidget(QLabel("RAM"))
        self.ram_gauge = QProgressBar()
        self.ram_gauge.setRange(0, 100)
        self.ram_gauge.setTextVisible(False)
        self.ram_gauge.setFixedWidth(80)
        self.ram_gauge.setFixedHeight(10)
        self.ram_gauge.setToolTip(
            "Systemweite RAM-Auslastung. Rot/Gelb folgt denselben "
            "Schwellenwerten wie der Text daneben."
        )
        resources.addWidget(self.ram_gauge)
        layout.addLayout(resources)

        self.job_group_tabs = QTabWidget()
        self.job_group_tabs.setDocumentMode(True)
        self.job_tables: dict[str, QTableWidget] = {}
        headers = (
            "Aufgabe",
            "Status",
            "Fortschritt",
            "Restzeit",
            "Ergebnis",
            "Handlungsbedarf",
        )
        for group_id, title, tooltip in (
            ("active", "Läuft", "Laufende, wartende oder gerade pausierende Arbeit"),
            ("attention", "Benötigt Aufmerksamkeit", "Pausierte, blockierte oder fehlgeschlagene Arbeit"),
            ("history", "Verlauf", "Abgeschlossene oder abgebrochene Arbeit"),
        ):
            table = self._table(headers)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            table.itemSelectionChanged.connect(
                lambda group=group_id: self._job_selected(group)
            )
            self.job_tables[group_id] = table
            self.job_group_tabs.addTab(table, title)
            self.job_group_tabs.setTabToolTip(
                self.job_group_tabs.count() - 1,
                tooltip,
            )
        self.jobs_table = self.job_tables["active"]
        layout.addWidget(self.job_group_tabs, 1)

        self.job_explanation = QLabel(
            "Wählen Sie eine Aufgabe, um Zweck, Ergebnis und wissenschaftliche "
            "Verwendbarkeit zu sehen."
        )
        self.job_explanation.setWordWrap(True)
        self.job_explanation.setObjectName("mimicSurfaceBox")
        layout.addWidget(self.job_explanation)

        controls = QHBoxLayout()
        self.pause_button = self._button(
            "",
            "Nach dem nächsten atomaren Mikro-Batch sicher pausieren",
            self.pause_selected_job,
            QStyle.StandardPixmap.SP_MediaPause,
        )
        controls.addWidget(self.pause_button)
        self.resume_button = self._button(
            "",
            "Ab dem letzten vollständigen Checkpoint fortsetzen",
            self.resume_selected_job,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        controls.addWidget(self.resume_button)
        self.retry_button = self._button(
            "",
            "Mit denselben geprüften Einstellungen erneut starten",
            self.retry_selected_job,
            QStyle.StandardPixmap.SP_BrowserReload,
        )
        controls.addWidget(self.retry_button)
        self.cancel_button = self._button(
            "",
            "Kontrolliert beenden; fertige atomare Ergebnisse bleiben erhalten",
            self.cancel_selected_job,
            QStyle.StandardPixmap.SP_BrowserStop,
        )
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        layout.addWidget(self._build_deep_search_box())

        self.job_technical_box = QGroupBox("Technische Details der gewählten Aufgabe")
        self.job_technical_box.setCheckable(True)
        self.job_technical_box.setChecked(False)
        job_technical_layout = QVBoxLayout(self.job_technical_box)
        self.job_technical_contents = QWidget()
        job_details_layout = QVBoxLayout(self.job_technical_contents)
        priority_row = QHBoxLayout()
        priority_row.addWidget(QLabel("Planungspriorität"))
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 100)
        self.priority_spin.setToolTip(
            "Beeinflusst nur die Reihenfolge lokaler Jobs, nicht die Wissenschaft."
        )
        priority_row.addWidget(self.priority_spin)
        self.priority_button = self._button(
            "Übernehmen",
            "Priorität ändern, ohne die Berechnung abzubrechen",
            self.update_selected_priority,
        )
        priority_row.addWidget(self.priority_button)
        priority_row.addStretch(1)
        job_details_layout.addLayout(priority_row)
        self.job_events = QTextEdit()
        self.job_events.setReadOnly(True)
        self.job_events.setMaximumHeight(260)
        self.job_events.setToolTip(
            "Job-ID, interne Stufe, Heartbeats, Checkpoints und Fehlermeldungen"
        )
        job_details_layout.addWidget(self.job_events)
        job_details_layout.addWidget(self._button(
            "SQL-/Ressourcen-Messaufnahme",
            "Letzte gespeicherte Messung der gewaehlten Aufgabe lesen; kein Live- oder ETA-Beweis",
            self._show_selected_job_telemetry,
        ))
        job_technical_layout.addWidget(self.job_technical_contents)
        self.job_technical_box.toggled.connect(
            self.job_technical_contents.setVisible
        )
        self.job_technical_contents.setVisible(False)
        layout.addWidget(self.job_technical_box)

        self.technical_tools_box = QGroupBox("Technische Werkzeuge")
        self.technical_tools_box.setObjectName("technicalToolsBox")
        self.technical_tools_box.setCheckable(True)
        self.technical_tools_box.setChecked(False)
        technical_outer = QVBoxLayout(self.technical_tools_box)
        self.technical_tools_contents = QWidget()
        technical_form = QFormLayout(self.technical_tools_contents)
        technical_note = QLabel(
            "Für eine normale Analyse nicht nötig: 'Analyse starten' oben "
            "rechnet alle Stufen vollständig und in der richtigen "
            "Reihenfolge.\n\nDieser Bereich ist die Rettungsleine. Bricht "
            "eine einzelne Stufe ab -- etwa wegen einer Schutzsperre oder "
            "eines Ressourcenengpasses -- lässt sie sich hier nach der "
            "Behebung einzeln nachholen, statt den gesamten mehrtägigen Lauf "
            "zu wiederholen. Außerdem liegt hier die Hardware-Neumessung."
        )
        technical_note.setWordWrap(True)
        technical_form.addRow(
            "",
            technical_note,
        )
        self.stage_combo = QComboBox()
        self.stage_combo.setObjectName("internalStageCombo")
        for stage in TECHNICAL_STAGE_IDS:
            descriptor = stage_descriptor(stage)
            group_name = STAGE_GROUP_NAMES_DE.get(descriptor.group_id, "Technik")
            self.stage_combo.addItem(
                f"{group_name}: {descriptor.display_name_de}",
                stage,
            )
        self.stage_combo.setToolTip(
            "Ein technischer Einzelschritt mit fest definierten Voraussetzungen und Ausgaben."
        )
        self.stage_combo.currentIndexChanged.connect(
            self._update_stage_parameter_visibility
        )
        self.stage_combo.currentIndexChanged.connect(
            self._update_technical_stage_explanation
        )
        technical_form.addRow("Einzelschritt", self.stage_combo)
        self.technical_stage_explanation = QLabel()
        self.technical_stage_explanation.setWordWrap(True)
        technical_form.addRow("", self.technical_stage_explanation)

        # The run name now lives in the main form; the technical tools reuse
        # the same field rather than keeping a second, divergent one.
        self.expert_toggle = QCheckBox("Ressourcen und Suchparameter anzeigen")
        self.expert_toggle.setToolTip(
            "Zeigt Worker, Speicher, Partitions- und Statistikdetails; Schutzregeln bleiben fest."
        )
        self.expert_toggle.toggled.connect(self._toggle_expert)
        technical_form.addRow("", self.expert_toggle)

        self.expert_controls = QWidget()
        expert_form = QFormLayout(self.expert_controls)
        expert_form.setContentsMargins(0, 0, 0, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setToolTip("Feature-Prozesse; BLAS läuft pro Worker mit einem Thread.")
        expert_form.addRow("Feature-Worker", self.workers_spin)
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(2, 256)
        self.memory_spin.setSuffix(" GB")
        self.memory_spin.setToolTip("DuckDB-Limit; mindestens 6 GB bleiben für System und GUI frei.")
        expert_form.addRow("DuckDB-Limit", self.memory_spin)
        self.max_records = QSpinBox()
        self.max_records.setRange(0, 800_035)
        self.max_records.setSpecialValueText("Alle")
        self.max_records.setToolTip("Nur Piloten begrenzen; null verarbeitet die vollständige Entwicklungskohorte.")
        expert_form.addRow("Max. ECGs", self.max_records)
        self.partition_size = QSpinBox()
        self.partition_size.setRange(25, 5_000)
        self.partition_size.setValue(250)
        self.partition_size.setToolTip("Größe einer atomar geschriebenen Feature-Partition.")
        expert_form.addRow("Partition", self.partition_size)
        self.micro_batch_size = QSpinBox()
        self.micro_batch_size.setRange(1, 250)
        self.micro_batch_size.setValue(25)
        self.micro_batch_size.setToolTip("Pause wird spätestens nach diesem Mikro-Batch wirksam.")
        expert_form.addRow("Mikro-Batch", self.micro_batch_size)
        self.bootstrap_iterations = QSpinBox()
        self.bootstrap_iterations.setRange(0, 10_000)
        self.bootstrap_iterations.setValue(1_000)
        self.bootstrap_iterations.setSingleStep(100)
        self.bootstrap_iterations.setToolTip("Subject-Cluster-Bootstrap; Standardwerte kommen aus dem Zweck-Preset.")
        expert_form.addRow("Bootstrap", self.bootstrap_iterations)
        self.future_horizons_combo = QComboBox()
        for label, horizons in (
            ("30 / 90 / 365 Tage (empfohlen)", (30, 90, 365)),
            ("30 / 90 Tage", (30, 90)),
            ("365 Tage", (365,)),
        ):
            self.future_horizons_combo.addItem(label, horizons)
        self.future_horizons_combo.setToolTip(
            "Follow-up-Fenster des getrennten Prognosearms; fehlende Nachbeobachtung ist nicht negativ."
        )
        expert_form.addRow("Prognosefenster", self.future_horizons_combo)
        self.future_phenotype_limit_combo = QComboBox()
        for label, limit in (
            ("100 nach Power (empfohlen)", 100),
            ("50 nach Power (schneller)", 50),
            ("Alle auswertbaren (sehr langsam)", None),
        ):
            self.future_phenotype_limit_combo.addItem(label, limit)
        self.future_phenotype_limit_combo.setToolTip(
            "Begrenzt nur Prognoseziele nach Fallzahl, niemals nach Markerleistung."
        )
        expert_form.addRow("Prognoseumfang", self.future_phenotype_limit_combo)
        self.include_promoted_candidates_checkbox = QCheckBox(
            "Zuvor promotete Kandidaten erneut screenen (opt-in)"
        )
        self.include_promoted_candidates_checkbox.setChecked(False)
        self.include_promoted_candidates_checkbox.setToolTip(
            "Sichert Phänotypen aus der kuratierten Hypothesenliste dieses "
            "Laufs einen Screening-Platz (nicht vom max_phenotypes-Deckel "
            "betroffen), statt sie zu ignorieren. Erneutes Screening ist "
            "KEINE unabhängige Replikation mehr -- Treffer werden explizit "
            "als seeded_from_promotion gekennzeichnet und als schwächere "
            "Evidenzstufe geführt, nie automatisch in die Pathologieliste "
            "übernommen."
        )
        expert_form.addRow("", self.include_promoted_candidates_checkbox)
        self.expert_controls.setVisible(False)
        technical_form.addRow("", self.expert_controls)

        technical_actions = QHBoxLayout()
        self.enqueue_button = self._button(
            "Einzelschritt starten",
            "Nur den ausgewählten technischen Schritt persistent einreihen",
            self.enqueue_selected_stage,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        technical_actions.addWidget(self.enqueue_button)
        self.benchmark_button = self._button(
            "Hardware neu messen",
            "1/2/4/6/7 Worker testen und die sichere Empfehlung speichern",
            self.enqueue_benchmark,
            QStyle.StandardPixmap.SP_ComputerIcon,
        )
        technical_actions.addWidget(self.benchmark_button)
        technical_actions.addStretch(1)
        technical_form.addRow("", technical_actions)
        technical_outer.addWidget(self.technical_tools_contents)
        self.technical_tools_box.toggled.connect(
            self.technical_tools_contents.setVisible
        )
        self.technical_tools_contents.setVisible(False)
        layout.addWidget(self.technical_tools_box)

        self._preset_changed()
        self._refresh_analysis_identity()
        self._update_technical_stage_explanation()
        self._set_job_action_state(None)
        self._set_selected_job_controls(None)
        return page

    # ------------------------------------------------------------------
    # Cohorts and QC
    # ------------------------------------------------------------------

    def _cohort_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        self.cohort_context = QLabel()
        self.cohort_context.setWordWrap(True)
        header.addWidget(self.cohort_context, 1)
        header.addWidget(
            self._button(
                "",
                "Inventar-, Label- und QC-Ansichten aus fertigen Artefakten neu laden",
                self.refresh_cohorts,
                QStyle.StandardPixmap.SP_BrowserReload,
            )
        )
        layout.addLayout(header)

        self.foundation_overview = self._table(
            ("Frage", "Antwort", "Bewertung", "Bedeutung / nächste Handlung")
        )
        self.foundation_overview.setMaximumHeight(420)
        self.foundation_overview.setToolTip(
            "Kompakte Antwort auf Umfang, Beurteilbarkeit, Qualität, zeitliche "
            "Zuordnung und patientengetrennte Entwicklung."
        )
        layout.addWidget(self.foundation_overview)

        tabs = QTabWidget()
        self.cohort_tabs = tabs
        self.inventory_view = PagedParquetTable(page_size=200)
        tabs.addTab(self.inventory_view, "Datenquellen (Details)")
        self.label_view = PagedParquetTable(page_size=200)
        tabs.addTab(self.label_view, "Klinische Referenzen")
        self.split_view = PagedParquetTable(page_size=100)
        tabs.addTab(self.split_view, "Train/Validation-Trennung")
        window_page = QWidget()
        window_layout = QVBoxLayout(window_page)
        self.window_plot = pg.PlotWidget()
        self.window_plot.setBackground("w")
        self.window_plot.showGrid(x=True, y=True, alpha=0.2)
        self.window_plot.setLabel("left", "Median |Zeitdelta|", units="h")
        self.window_plot.setLabel("bottom", "Endpunkt / Fenster")
        self.window_plot.setToolTip(
            "Medianer absoluter Abstand zwischen ECG und Referenzevidenz; "
            "jeder Punkt ist eine Endpunkt-/Fenster-/Split-Kombination."
        )
        window_layout.addWidget(self.window_plot, 1)
        self.window_view = PagedParquetTable(page_size=200)
        window_layout.addWidget(self.window_view, 2)
        tabs.addTab(window_page, "Zeitliche Zuordnung")

        qc_page = QWidget()
        qc_layout = QVBoxLayout(qc_page)
        qc_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.qc_plot = pg.PlotWidget()
        self.qc_plot.setBackground("w")
        self.qc_plot.showGrid(x=True, y=True, alpha=0.2)
        self.qc_plot.setLabel("left", "Prävalenz")
        self.qc_plot.setLabel("bottom", "Endpunkt")
        self.qc_plot.setToolTip(
            "Prävalenz ausschließlich unter beurteilbaren Positiv-/Negativfällen."
        )
        qc_splitter.addWidget(self.qc_plot)
        self.assessability_plot = pg.PlotWidget()
        self.assessability_plot.setBackground("w")
        self.assessability_plot.showGrid(x=True, y=True, alpha=0.2)
        self.assessability_plot.setLabel("left", "Anteil aller ECGs")
        self.assessability_plot.setLabel("bottom", "Endpunkt")
        self.assessability_plot.setToolTip(
            "Anteile beurteilbar, indeterminate, unavailable und ausgeschlossen. "
            "Fehlende Referenzdaten werden niemals als negativ gezählt."
        )
        qc_splitter.addWidget(self.assessability_plot)
        qc_layout.addWidget(qc_splitter, 1)

        marker_box = QGroupBox(
            "Messbarkeit der Marker nach Endpunktstatus"
        )
        marker_layout = QVBoxLayout(marker_box)
        marker_note = QLabel(
            "Anteil der ECGs, in denen der jeweilige Wellenabschnitt "
            "messbar war, getrennt nach Fällen und Kontrollen des gewählten "
            "Endpunkts. Die Diagramme darüber beziehen sich auf die "
            "Verfügbarkeit der Referenz, nicht auf die des Markers.\n\n"
            "Gleich hohe Balken bedeuten Verlust an Aussagekraft. "
            "Unterschiedlich hohe Balken bedeuten mehr: der Abschnitt fehlt "
            "dann systematisch häufiger in einer der beiden Gruppen, und der "
            "Marker bildet den Endpunkt teilweise über seine eigene "
            "Abwesenheit ab. Die Richtung der resultierenden Verzerrung ist "
            "nicht bestimmbar.\n\n"
            "Regelmäßig betroffen ist die P-Welle; QRS-Abschnitte zeigen den "
            "Effekt kaum. Die Endpunktliste ist nach dem größten Abstand "
            "sortiert."
        )
        marker_note.setWordWrap(True)
        marker_note.setObjectName("mimicInfoBoxWarning")
        marker_layout.addWidget(marker_note)

        availability_controls = QHBoxLayout()
        availability_controls.addWidget(QLabel("Endpunkt:"))
        self.availability_endpoint_combo = QComboBox()
        self.availability_endpoint_combo.setMinimumWidth(360)
        self.availability_endpoint_combo.currentIndexChanged.connect(
            self._redraw_segment_availability
        )
        availability_controls.addWidget(self.availability_endpoint_combo)
        availability_controls.addStretch(1)
        marker_layout.addLayout(availability_controls)

        self.availability_plot = pg.PlotWidget()
        self.availability_plot.setBackground("w")
        self.availability_plot.showGrid(y=True, alpha=0.2)
        self.availability_plot.setLabel("left", "messbar (% der ECGs)")
        self.availability_plot.setLabel("bottom", "Wellenabschnitt")
        self.availability_plot.setYRange(0, 100, padding=0.02)
        self.availability_plot.addLegend()
        marker_layout.addWidget(self.availability_plot, 2)

        self.availability_table = PagedParquetTable(page_size=200)
        self.availability_table.setToolTip(
            "gap_points ist die Differenz Fälle minus Kontrollen in "
            "Prozentpunkten. Negative Werte stehen für geringere "
            "Messbarkeit bei den Fällen; differential markiert Abstände ab "
            "10 Punkten."
        )
        marker_layout.addWidget(self.availability_table, 1)
        qc_layout.addWidget(marker_box, 2)

        self.qc_table = self._table(("Prüfung", "Status", "Detail"))
        self.qc_table.setMaximumHeight(420)
        qc_layout.addWidget(self.qc_table)
        tabs.addTab(qc_page, "Beurteilbarkeit & Blocker")

        failure_page = QWidget()
        failure_layout = QVBoxLayout(failure_page)
        self.feature_failure_plot = pg.PlotWidget()
        self.feature_failure_plot.setBackground("w")
        self.feature_failure_plot.showGrid(x=True, y=True, alpha=0.2)
        self.feature_failure_plot.setLabel("left", "ECGs")
        self.feature_failure_plot.setLabel("bottom", "Fehlerkategorie")
        self.feature_failure_plot.setToolTip(
            "Aggregierte Signal-QC- und Extraktionsfehler der aktuellen "
            "Feature-Artefakte."
        )
        failure_layout.addWidget(self.feature_failure_plot, 1)
        self.feature_failure_view = PagedParquetTable(page_size=200)
        failure_layout.addWidget(self.feature_failure_view, 2)
        tabs.addTab(failure_page, "ECG- und Markerqualität")

        live_page = QWidget()
        live_layout = QVBoxLayout(live_page)
        live_header = QHBoxLayout()
        self.live_feature_context = QLabel()
        self.live_feature_context.setWordWrap(True)
        live_header.addWidget(self.live_feature_context, 1)
        live_header.addWidget(QLabel("Marker"))
        self.live_marker_combo = QComboBox()
        self.live_marker_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.live_marker_combo.setMinimumContentsLength(22)
        self.live_marker_combo.setMaximumWidth(380)
        self.live_marker_combo.setToolTip(
            "Deskriptive Verteilung aus bereits atomar abgeschlossenen "
            "Partitionen; keine formale Endpunktanalyse."
        )
        live_features = tuple(
            dict.fromkeys(
                feature_id
                for panel in default_feature_panels()
                if panel.panel_id in ("primary_core", "sentinel")
                for feature_id in panel.feature_ids
            )
        )
        for feature_id in live_features:
            self.live_marker_combo.addItem(
                biomarker_display_name(feature_id),
                feature_id,
            )
            self.live_marker_combo.setItemData(
                self.live_marker_combo.count() - 1,
                biomarker_tooltip(feature_id),
                Qt.ItemDataRole.ToolTipRole,
            )
        preferred_live_marker = self.live_marker_combo.findData(
            "qrs_t_hausdorff_distance"
        )
        if preferred_live_marker >= 0:
            self.live_marker_combo.setCurrentIndex(preferred_live_marker)
        self.live_marker_combo.currentIndexChanged.connect(
            self._refresh_live_features
        )
        live_header.addWidget(self.live_marker_combo)
        live_layout.addLayout(live_header)
        live_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_feature_plot = pg.PlotWidget()
        self.live_feature_plot.setBackground("w")
        self.live_feature_plot.showGrid(x=True, y=True, alpha=0.2)
        self.live_feature_plot.addLegend(offset=(8, 8))
        self.live_feature_plot.setLabel("left", "Anteil")
        self.live_feature_plot.setLabel("bottom", "Markerwert")
        self.live_feature_plot.setToolTip(
            "Winsorisierte 1.-99.-Perzentil-Verteilung nach Development-Split. "
            "Snapshotstatus bleibt partial, solange die Extraktion läuft."
        )
        live_splitter.addWidget(self.live_feature_plot)
        self.live_feature_table = self._table(
            ("Split", "Status", "ECGs", "Anteil")
        )
        live_splitter.addWidget(self.live_feature_table)
        live_splitter.setSizes([700, 500])
        live_layout.addWidget(live_splitter, 1)
        tabs.addTab(live_page, "Laufender Teilstand")
        layout.addWidget(tabs, 1)
        self.locked_label_status = QLabel(
            "Locked Test: versiegelt; keine breite Suche und keine Auswahl."
        )
        self.locked_label_status.setObjectName("mimicTextDangerMuted")
        layout.addWidget(self.locked_label_status)
        _disable_auto_si_prefix(self.qc_plot, self.assessability_plot)
        return page

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _results_workspace(self) -> QWidget:
        page = QWidget()
        root_layout = QVBoxLayout(page)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Ergebnisse verstehen"))
        self.results_mode_combo = QComboBox()
        self.results_mode_combo.addItem("Einfach", "simple")
        self.results_mode_combo.addItem("Detailliert", "detailed")
        self.results_mode_combo.setToolTip(
            "Einfach: Kernaussagen und Grenzen. Detailliert: alle Tabellen, "
            "Methodenvergleiche und Forschungsansichten."
        )
        mode_row.addWidget(self.results_mode_combo)
        mode_row.addWidget(QLabel("Ergebnisstand"))
        self.atlas_run_combo = QComboBox()
        self.atlas_run_combo.setMinimumContentsLength(14)
        self.atlas_run_combo.setMaximumWidth(300)
        self.atlas_run_combo.setToolTip(
            "Nur aktuelle, kompatible Ergebnisstände. Laufende Analysen zeigen "
            "zunächst einen ehrlichen Leerzustand."
        )
        self.atlas_run_combo.currentIndexChanged.connect(self._atlas_run_changed)
        mode_row.addWidget(self.atlas_run_combo)
        atlas_pilot_button = QPushButton("Pilot-Testarbeitsbereich")
        atlas_pilot_button.setToolTip(
            "Archivierte Pilotresultate getrennt vom aktuellen Ergebnisatlas "
            "lesen; keine Übernahme in den Voll-Lauf."
        )
        atlas_pilot_button.clicked.connect(self._open_pilot_workspace)
        mode_row.addWidget(atlas_pilot_button)
        mode_row.addStretch(1)
        root_layout.addLayout(mode_row)
        self.results_atlas_tabs = QTabWidget()
        self.results_atlas_tabs.setDocumentMode(True)
        self.results_atlas = ResultsAtlasWidget(self.result_repository)
        self.results_atlas.endpoint_requested.connect(self._open_atlas_endpoint)
        self.results_atlas.spatial_requested.connect(self._open_atlas_spatial)
        self.results_atlas.designs_requested.connect(self._open_atlas_designs)
        self.results_atlas.scores_requested.connect(self._open_atlas_scores)
        self.results_atlas.leads_requested.connect(self._open_atlas_leads)
        self.results_atlas.review_requested.connect(self._open_atlas_review)
        self.results_atlas.detail_requested.connect(self._open_atlas_detail)
        self.results_atlas.simple_requested.connect(
            lambda: self.results_mode_combo.setCurrentIndex(0)
        )
        self.results_atlas_tabs.addTab(self.results_atlas, "Ergebnisatlas")
        self.results_legacy_page = QWidget()
        layout = QVBoxLayout(self.results_legacy_page)
        pilot_button = QPushButton("Pilot-Testarbeitsbereich öffnen …")
        pilot_button.setToolTip("Physisch gesicherte Pilotresultate und isoliertes ECG-Labor; kein Neustart alter Runs")
        pilot_button.clicked.connect(self._open_pilot_workspace)
        layout.addWidget(pilot_button)
        self.discovery_summary_banner = QLabel(
            "Noch keine Entdeckungs-Zusammenfassung verfügbar."
        )
        self.discovery_summary_banner.setWordWrap(True)
        self.discovery_summary_banner.setObjectName("mimicInfoBoxAccent")
        self.discovery_summary_banner.setToolTip(
            "Zählt replizierte Treffer aus dem breiten Scan (Pathologien, "
            "Prognosen und sonstige klinische Assoziationen) im aktuell "
            "gewählten Ergebnisstand. Öffnet 'Neue Zusammenhänge' bei Klick."
        )
        self.discovery_summary_banner.setCursor(Qt.CursorShape.PointingHandCursor)
        self.discovery_summary_banner.mousePressEvent = (
            lambda _event: self._focus_discovery_perspective()
        )
        _reserve_wrapped_label_height(self.discovery_summary_banner, 2)
        layout.addWidget(self.discovery_summary_banner)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Ergebnisstand"))
        self.results_run_combo = QComboBox()
        self.results_run_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.results_run_combo.setMinimumContentsLength(10)
        self.results_run_combo.setMaximumWidth(240)
        self.results_run_combo.setToolTip(
            "Standard ist der neueste vollständige, mit dem aktuellen Code "
            "kompatible Validation-Lauf. Alte Piloten werden nicht vermischt."
        )
        self.results_run_combo.currentIndexChanged.connect(self.refresh_results)
        self.results_run_combo.currentIndexChanged.connect(
            self._sync_atlas_run_combo
        )
        filters.addWidget(self.results_run_combo)
        scope_filters = QHBoxLayout()
        scope_filters.addWidget(QLabel("Pathologie oder klinisches Ziel"))
        self.endpoint_filter = QComboBox()
        self.endpoint_filter.setEditable(True)
        self.endpoint_filter.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.endpoint_filter.lineEdit().setPlaceholderText(
            "Name oder Synonym suchen"
        )
        self.endpoint_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.endpoint_filter.setMinimumContentsLength(12)
        self.endpoint_filter.setMaximumWidth(300)
        self.endpoint_filter.setToolTip(
            "Durchsuchbare klinische Bezeichnung. Interne IDs stehen nur in "
            "Details und Exporten. 'Übersicht' zeigt alle Ziele."
        )
        self.endpoint_filter.currentIndexChanged.connect(self.refresh_results)
        self.endpoint_filter.lineEdit().returnPressed.connect(
            self.refresh_results
        )
        scope_filters.addWidget(self.endpoint_filter, 2)
        scope_filters.addWidget(QLabel("Markergruppe"))
        self.panel_filter = QComboBox()
        self.panel_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.panel_filter.setMinimumContentsLength(9)
        self.panel_filter.setMaximumWidth(155)
        self.panel_filter.addItem("Alle Markergruppen", "")
        self.panel_filter.addItem("Kernmarker", "primary_core")
        self.panel_filter.addItem("Erweiterte Sentinel-Marker", "sentinel")
        self.panel_filter.addItem("Exploratives Discovery-Reservoir", "discovery_reserve")
        self.panel_filter.setToolTip(
            "Kernmarker sind vorab priorisiert; Sentinel erweitert Familien; "
            "Discovery ist hypothesengenerierend und strenger zu interpretieren."
        )
        self.panel_filter.currentIndexChanged.connect(self.refresh_results)
        scope_filters.addWidget(self.panel_filter)
        self.classical_only_filter = QCheckBox("Nur Leitlinienkriterien")
        self.classical_only_filter.setToolTip(
            "Zeigt nur die etablierten klassischen ECG-Leitlinienkriterien "
            "(z. B. Sokolow-Lyon, QTc, Schenkelblock-Muster) statt der "
            "datengetriebenen Discovery-Marker -- nützlich als "
            "Plausibilitätsprüfung: passen die berechneten Werte zu den "
            "bekannten Leitlinien-Sollwerten?"
        )
        self.classical_only_filter.toggled.connect(self.refresh_results)
        scope_filters.addWidget(self.classical_only_filter)
        scope_filters.addStretch(1)
        self.split_filter = QComboBox()
        self.split_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.split_filter.setMinimumContentsLength(7)
        self.split_filter.setMaximumWidth(120)
        self.split_filter.addItem("Validation", "validation")
        self.split_filter.setItemData(
            self.split_filter.count() - 1,
            GLOSSARY_DE["Validation"],
            Qt.ItemDataRole.ToolTipRole,
        )
        self.split_filter.addItem("Train", "train")
        self.split_filter.setItemData(
            self.split_filter.count() - 1,
            GLOSSARY_DE["Train"],
            Qt.ItemDataRole.ToolTipRole,
        )
        self.split_filter.setToolTip(
            "Welche Patientengruppe die Ergebnisse zeigt -- Validation ist "
            "der wissenschaftlich aussagekräftige Standard, Train nur zur "
            "Modell-/Grenzwertbildung. Locked Test wird hier nie gezeigt."
        )
        self.split_filter.currentIndexChanged.connect(self.refresh_results)
        self.inference_filter = QComboBox()
        self.inference_filter.addItem("Formal", "formal_inference")
        self.inference_filter.addItem("Unterpowert", "underpowered")
        self.inference_filter.addItem("Deskriptiv", "descriptive_only")
        self.inference_filter.addItem("Alle", "")
        self.inference_filter.setMaximumWidth(145)
        self.inference_filter.setToolTip(
            "Steuert Screening-Tabelle und Marker-Grafiken. Formal ist der "
            "wissenschaftlich empfohlene Standard; methodische QC-Kontrollen "
            "werden nicht als Biomarker gerankt."
        )
        self.inference_filter.currentIndexChanged.connect(
            self.refresh_results
        )
        filters.addStretch(1)
        self.freeze_button = self._button(
            "Freeze erstellen",
            "Nach wissenschaftlicher Prüfung einen unveränderlichen Freeze-Hash "
            "erzeugen. Der Locked Test wird dabei nicht gelesen.",
            self.freeze_selected_run,
            QStyle.StandardPixmap.SP_DialogSaveButton,
        )
        filters.addWidget(self.freeze_button)
        filters.addWidget(
            self._button(
                "",
                "Ergebnistabellen und Visualisierungen neu laden",
                self.refresh_results,
                QStyle.StandardPixmap.SP_BrowserReload,
            )
        )
        layout.addLayout(filters)
        layout.addLayout(scope_filters)

        self.result_context = QLabel()
        self.result_context.setWordWrap(True)
        self.result_context.setObjectName("mimicInfoBoxNeutral")
        layout.addWidget(self.result_context)

        primary_filters = QHBoxLayout()
        self.result_search = QLineEdit()
        self.result_search.setPlaceholderText(
            "Pathologie, Synonym, Biomarker oder Familie suchen"
        )
        self.result_search.setClearButtonEnabled(True)
        self.result_search.setToolTip(
            "Durchsucht serverseitig den vollständigen Ergebnisstand, nicht "
            "nur die aktuell sichtbare Seite."
        )
        self.result_search.returnPressed.connect(
            self._refresh_result_repository
        )
        primary_filters.addWidget(self.result_search, 2)
        self.result_evidence_filter = QComboBox()
        for label, value in (
            ("Alle Evidenzstufen", ""),
            ("Validation bestätigt", "validation_replicated"),
            ("Entwicklung", "development"),
            ("Unterpowert", "underpowered"),
            ("Instabil", "unstable"),
        ):
            self.result_evidence_filter.addItem(label, value)
        self.result_evidence_filter.setToolTip(
            "Evidenzstufe beschreibt Replikation und Stabilität; sie ist "
            "keine Diagnosewahrscheinlichkeit."
        )
        self.result_evidence_filter.currentIndexChanged.connect(
            self._refresh_result_repository
        )
        primary_filters.addWidget(self.result_evidence_filter)
        metric_filters = QHBoxLayout()
        metric_filters.addWidget(QLabel("Sensitivität ab"))
        self.minimum_sensitivity_filter = QSpinBox()
        self.minimum_sensitivity_filter.setRange(0, 100)
        self.minimum_sensitivity_filter.setSuffix(" %")
        self.minimum_sensitivity_filter.setSpecialValueText("beliebig")
        self.minimum_sensitivity_filter.setToolTip(
            "Mindest-Sensitivität am vorab festgelegten Betriebspunkt; null filtert nicht."
        )
        self.minimum_sensitivity_filter.editingFinished.connect(
            self._refresh_result_repository
        )
        metric_filters.addWidget(self.minimum_sensitivity_filter)
        metric_filters.addWidget(QLabel("Spezifität ab"))
        self.minimum_specificity_filter = QSpinBox()
        self.minimum_specificity_filter.setRange(0, 100)
        self.minimum_specificity_filter.setSuffix(" %")
        self.minimum_specificity_filter.setSpecialValueText("beliebig")
        self.minimum_specificity_filter.setToolTip(
            "Mindest-Spezifität des gleichen Validation-Betriebspunkts."
        )
        self.minimum_specificity_filter.editingFinished.connect(
            self._refresh_result_repository
        )
        metric_filters.addWidget(self.minimum_specificity_filter)
        metric_filters.addStretch(1)
        self.result_sort_combo = QComboBox()
        for label, value in (
            ("Beste Evidenz", "evidence_rank"),
            ("Höchste ROC-AUC", "roc_auc"),
            ("Höchste Sensitivität", "sensitivity_at_95_specificity"),
            ("Höchste Spezifität", "operating_specificity"),
            ("Name", "endpoint_name_de"),
        ):
            self.result_sort_combo.addItem(label, value)
        self.result_sort_combo.setToolTip(
            "Sortiert die vollständige serverseitige Ergebnismenge."
        )
        self.result_sort_combo.currentIndexChanged.connect(
            self._refresh_result_repository
        )
        primary_filters.addWidget(self.result_sort_combo)
        primary_filters.addWidget(
            self._button(
                "Zurücksetzen",
                "Alle Ergebnisfilter entfernen und zur ersten Seite wechseln",
                self._reset_primary_result_filters,
                QStyle.StandardPixmap.SP_DialogResetButton,
            )
        )
        layout.addLayout(primary_filters)
        layout.addLayout(metric_filters)

        self.active_result_filters = QLabel("Keine zusätzlichen Filter aktiv")
        self.active_result_filters.setWordWrap(True)
        self.active_result_filters.setObjectName("mimicTextMuted")
        layout.addWidget(self.active_result_filters)

        self.result_perspective_tabs = QTabWidget()
        self.result_perspective_tabs.setDocumentMode(True)
        self.result_perspective_tables: dict[str, QTableWidget] = {}
        self._discovery_perspective_tab_widget: Optional[QWidget] = None
        for perspective, title, tooltip in (
            (
                "pathologies",
                "Pathologien",
                "Für jede Pathologie: beste VCG-/ECG-Marker und klassische Vergleichsregel",
            ),
            (
                "biomarkers",
                "Biomarker",
                "Für jeden Biomarker: zugehörige Pathologien und Leistungswerte",
            ),
            (
                "criteria",
                "Kriterienkandidaten",
                "Exakte Formel, Train-Grenzwert, Grauzone, Evidenz und Anwendbarkeit",
            ),
            (
                "discovery",
                "Neue Zusammenhänge",
                "Breiter Scan mit adjustiert bestätigten, konfundierten oder verworfenen Hypothesen",
            ),
        ):
            table = self._table(("Ergebnis",))
            table.setObjectName(f"resultPerspective_{perspective}")
            table.setToolTip(tooltip)
            table.cellDoubleClicked.connect(
                lambda _row, _column, p=perspective: self._open_primary_result_detail(p)
            )
            self.result_perspective_tables[perspective] = table
            tab_widget: QWidget = table
            if perspective == "discovery":
                tab_widget = QWidget()
                discovery_layout = QVBoxLayout(tab_widget)
                discovery_note = QLabel(
                    "Dieser breite Scan sucht auch nach bisher nicht als "
                    "ECG-/VCG-Pathologie bekannten Zusammenhängen. Reine "
                    "Effektstärke, statistische Sicherheit und Confounding-"
                    "Prüfung bleiben getrennt; Treffer sind Hypothesen und "
                    "keine Diagnosen."
                )
                discovery_note.setWordWrap(True)
                discovery_note.setObjectName("mimicInfoBoxAccent")
                discovery_layout.addWidget(discovery_note)
                discovery_filters_top = QHBoxLayout()
                discovery_filters_top.addWidget(QLabel("Bereich"))
                self.discovery_scope_combo = QComboBox()
                self.discovery_scope_combo.setMaximumWidth(360)
                for label, value, item_tooltip in (
                    (
                        "Alle getrennten Bereiche",
                        "",
                        "Zeigt Treffer aus allen drei Bereichen zusammen: "
                        "aktuelle Pathologien, zukünftige Erkrankungen und "
                        "sonstige klinische Assoziationen.",
                    ),
                    (
                        "Aktuelle Pathologie-Hypothesen",
                        "pathology_hypothesis",
                        "ECG-/VCG-Marker geprüft gegen Diagnosen, die zum "
                        "Zeitpunkt des ECGs bereits bestehen (PheWAS-artiger "
                        "Scan über ICD-/PhecodeX-Diagnosen).",
                    ),
                    (
                        "Zukünftige Erkrankungen",
                        "future_outcome_hypothesis",
                        "Prospektiv: Marker geprüft gegen Diagnosen, die "
                        "erst 30/90/365 Tage NACH dem ECG auftreten -- "
                        "mögliche Frühwarn-Zusammenhänge.",
                    ),
                    (
                        "Labor, Medikamente, Prozeduren, Outcomes, "
                        "Demografie, Vitalwerte, Soziales",
                        "non_pathology_association",
                        "Kein Diagnose-Scan, sondern feste, vordefinierte "
                        "klinische Kontext-Domänen (z. B. Troponin, "
                        "Medikamentenklassen, ICU-Aufenthalt).",
                    ),
                ):
                    self.discovery_scope_combo.addItem(label, value)
                    self.discovery_scope_combo.setItemData(
                        self.discovery_scope_combo.count() - 1,
                        item_tooltip,
                        Qt.ItemDataRole.ToolTipRole,
                    )
                self.discovery_scope_combo.setToolTip(
                    "Pathologien, Prognoseziele und sonstige klinische "
                    "Assoziationen werden getrennt ausgewertet."
                )
                self.discovery_scope_combo.currentIndexChanged.connect(
                    self._discovery_filter_changed
                )
                discovery_filters_top.addWidget(self.discovery_scope_combo)
                discovery_filters_top.addWidget(QLabel("Pathologieart"))
                self.discovery_cardiac_combo = QComboBox()
                self.discovery_cardiac_combo.setMaximumWidth(330)
                for label, value, item_tooltip in (
                    (
                        "Kardiale und nicht-kardiale Ziele",
                        "",
                        "Keine Einschränkung nach Pathologie-Art.",
                    ),
                    (
                        "Nur nicht-kardiovaskuläre Pathologien",
                        "non_cardiovascular",
                        "Zeigt gezielt mögliche Zusammenhänge zwischen "
                        "ECG/VCG-Mustern und NICHT-kardialen Diagnosen "
                        "(z. B. Niere, Lunge, Stoffwechsel) -- der "
                        "eigentliche 'unbekannte Einflüsse'-Scan.",
                    ),
                    (
                        "Nur kardiovaskuläre Pathologien",
                        "cardiovascular",
                        "Nur Herz-Kreislauf-Diagnosen als Ziel.",
                    ),
                    (
                        "Nur klinischer Kontext, keine Pathologie",
                        "not_pathology",
                        "Ziele sind keine Diagnosen, sondern Kontext wie "
                        "Laborwerte, Medikamente oder Vitalwerte.",
                    ),
                ):
                    self.discovery_cardiac_combo.addItem(label, value)
                    self.discovery_cardiac_combo.setItemData(
                        self.discovery_cardiac_combo.count() - 1,
                        item_tooltip,
                        Qt.ItemDataRole.ToolTipRole,
                    )
                self.discovery_cardiac_combo.setToolTip(
                    "Erlaubt insbesondere die gezielte Sicht auf mögliche "
                    "nicht-kardiale Pathologie-ECG/VCG-Zusammenhänge."
                )
                self.discovery_cardiac_combo.currentIndexChanged.connect(
                    self._discovery_filter_changed
                )
                discovery_filters_top.addWidget(self.discovery_cardiac_combo)
                discovery_filters_top.addStretch(1)
                discovery_layout.addLayout(discovery_filters_top)
                discovery_filters_bottom = QHBoxLayout()
                discovery_filters_bottom.addWidget(QLabel("Prüfstatus"))
                self.discovery_status_combo = QComboBox()
                self.discovery_status_combo.setMaximumWidth(310)
                for label, value in (
                    ("Alle Prüfstände", ""),
                    ("Adjustiert repliziert", "replicated"),
                    ("Vielversprechend", "promising"),
                    ("Wahrscheinlich confounded", "likely_confounded"),
                    ("Neu / nur Training", "new"),
                    ("Nicht repliziert", "rejected"),
                    ("Replizierte Kontextassoziation", "replicated_association"),
                ):
                    self.discovery_status_combo.addItem(label, value)
                self.discovery_status_combo.setToolTip(
                    "Validation-Replikation und vorgegebene Adjustierung "
                    "bestimmen den Prüfstatus."
                )
                self.discovery_status_combo.currentIndexChanged.connect(
                    self._discovery_filter_changed
                )
                discovery_filters_bottom.addWidget(self.discovery_status_combo)
                discovery_filters_bottom.addStretch(1)
                discovery_layout.addLayout(discovery_filters_bottom)
                table.setToolTip(
                    "Die Sortierung oberhalb wirkt serverseitig auf die "
                    "vollständige Liste, nicht nur auf die sichtbare Seite."
                )
                discovery_layout.addWidget(table, 1)
                self._discovery_perspective_tab_widget = tab_widget
            self.result_perspective_tabs.addTab(tab_widget, title)
            self.result_perspective_tabs.setTabToolTip(
                self.result_perspective_tabs.count() - 1,
                tooltip,
            )
        # "Neue Zusammenhänge" (the broad discovery scan) is the most
        # actionable view for spotting new findings, so it is shown first
        # by default; the physical tab order above is left untouched. The
        # actual switch happens once at the end of __init__ (see
        # _focus_discovery_perspective there), once every other widget
        # this signal handler touches actually exists.
        self.result_perspective_tabs.currentChanged.connect(
            self._result_perspective_changed
        )
        layout.addWidget(self.result_perspective_tabs, 2)

        paging = QHBoxLayout()
        self.result_previous_button = self._button(
            "",
            "Vorherige Ergebnisseite",
            self._previous_primary_result_page,
            QStyle.StandardPixmap.SP_ArrowBack,
        )
        paging.addWidget(self.result_previous_button)
        self.result_page_label = QLabel()
        paging.addWidget(self.result_page_label)
        self.result_next_button = self._button(
            "",
            "Nächste Ergebnisseite",
            self._next_primary_result_page,
            QStyle.StandardPixmap.SP_ArrowForward,
        )
        paging.addWidget(self.result_next_button)
        paging.addStretch(1)
        self.result_repository_status = QLabel()
        self.result_repository_status.setWordWrap(True)
        _reserve_wrapped_label_height(self.result_repository_status, 2)
        paging.addWidget(self.result_repository_status, 1)
        layout.addLayout(paging)
        self._primary_result_offsets = {
            perspective: 0 for perspective in self.result_perspective_tables
        }
        self._primary_result_totals = {
            perspective: 0 for perspective in self.result_perspective_tables
        }

        self.advanced_results_box = QGroupBox(
            "Detailanalyse und Spezialdiagramme"
        )
        self.advanced_results_box.setCheckable(True)
        self.advanced_results_box.setChecked(True)
        advanced_layout = QVBoxLayout(self.advanced_results_box)
        self.advanced_results_contents = QWidget()
        advanced_contents_layout = QVBoxLayout(
            self.advanced_results_contents
        )
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Auswertung"))
        self.result_view_combo = QComboBox()
        self.result_view_combo.setMinimumWidth(270)
        self.result_view_combo.setToolTip(
            "Wählt genau eine Ergebnisansicht. Filter für Run, Endpunkt, "
            "Panel und Split gelten für alle Ansichten."
        )
        view_row.addWidget(self.result_view_combo)
        view_row.addWidget(QLabel("Grafik-Power"))
        view_row.addWidget(self.inference_filter)
        view_row.addStretch(1)
        advanced_contents_layout.addLayout(view_row)

        self.result_tabs = QTabWidget()
        tabs = self.result_tabs
        performance = QWidget()
        performance_layout = QVBoxLayout(performance)
        navigator_controls = QHBoxLayout()
        navigator_controls.addWidget(QLabel("Sicht"))
        self.navigator_mode_combo = QComboBox()
        self.navigator_mode_combo.addItem(
            "Pathologie → beste Marker",
            "pathology_marker",
        )
        self.navigator_mode_combo.setItemData(
            self.navigator_mode_combo.count() - 1,
            "Startet bei einem klinischen Ziel und listet die dafür "
            "leistungsfähigsten Marker -- die Frage 'welcher Marker hilft "
            "bei dieser Pathologie am meisten?'.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.navigator_mode_combo.addItem(
            "Biomarker → Pathologien",
            "marker_pathology",
        )
        self.navigator_mode_combo.setItemData(
            self.navigator_mode_combo.count() - 1,
            "Startet bei einem Marker und listet, für welche Pathologien "
            "er am aussagekräftigsten ist -- die Frage 'wofür ist dieser "
            "Marker gut?'.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.navigator_mode_combo.addItem(
            "Breite Zusammenhänge",
            "discovery_ranked",
        )
        self.navigator_mode_combo.setItemData(
            self.navigator_mode_combo.count() - 1,
            "Alle replizierten, FDR-korrigierten Treffer aus dem breiten "
            "Discovery-Scan (Pathologien, zukünftige Erkrankungen, "
            "klinischer Kontext), unabhängig von Marker oder Pathologie.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.navigator_mode_combo.setToolTip(
            "Wechselt zwischen der endpunktzentrierten Rangliste, der "
            "markerzentrierten Sicht und replizierten breiten Assoziationen."
        )
        # Consistent with the primary perspective tabs: default to the
        # broad discovery view, the most actionable for spotting new
        # findings, rather than the first-added pathology-centric view.
        # Set *before* connecting currentIndexChanged, so this initial
        # switch does not call refresh_results() while the rest of
        # __init__ has not finished building other widgets yet.
        discovery_ranked_index = self.navigator_mode_combo.findData(
            "discovery_ranked"
        )
        if discovery_ranked_index >= 0:
            self.navigator_mode_combo.setCurrentIndex(discovery_ranked_index)
        self.navigator_mode_combo.currentIndexChanged.connect(
            self.refresh_results
        )
        navigator_controls.addWidget(self.navigator_mode_combo)
        self.navigator_search = QLineEdit()
        self.navigator_search.setPlaceholderText(
            "Marker, Pathologie oder Familie suchen"
        )
        self.navigator_search.setClearButtonEnabled(True)
        self.navigator_search.setToolTip(
            "Filtert die kompakte Rangliste; Groß-/Kleinschreibung wird ignoriert."
        )
        self.navigator_search.returnPressed.connect(self.refresh_results)
        self.navigator_search.editingFinished.connect(self.refresh_results)
        navigator_controls.addWidget(self.navigator_search, 1)
        performance_layout.addLayout(navigator_controls)
        self.navigator_summary = QLabel()
        self.navigator_summary.setWordWrap(True)
        self.navigator_summary.setObjectName("mimicTextMuted")
        performance_layout.addWidget(self.navigator_summary)
        self.performance_plot = pg.PlotWidget()
        self.performance_plot.setBackground("w")
        self.performance_plot.showGrid(x=True, y=True, alpha=0.2)
        self.performance_plot.setLabel("bottom", "ROC-AUC")
        self.performance_plot.setLabel(
            "left",
            "Sensitivität am Trainingsschwellenwert (Ziel: 95 % Spezifität)",
        )
        self.performance_plot.setTitle(
            "Ergebnisnavigator: Sensitivität vs. ROC-AUC"
        )
        self.performance_plot.addLegend(offset=(8, 8))
        performance_layout.addWidget(self.performance_plot, 2)
        self.screening_table = PagedParquetTable(page_size=250)
        self.screening_table.setToolTip(
            "Spaltenköpfe sortieren die vollständige Rangliste serverseitig. "
            "Sensitivität und Spezifität stammen aus Train-fixierten "
            "Betriebspunkten auf Validation."
        )
        self.screening_table.selection_changed.connect(
            self._update_navigator_detail
        )
        performance_layout.addWidget(self.screening_table, 1)
        performance_layout.addWidget(
            self._chart_guidance_label(
                "Übersichtstabelle: verknüpft Marker und Pathologien nach "
                "Leistungsfähigkeit. Zeile anklicken, um Filter oben auf "
                "diese Kombination zu setzen. Sortierung wirkt "
                "serverseitig auf die vollständige Liste."
            )
        )
        tabs.addTab(performance, "Ergebnisnavigator")

        curves = QWidget()
        curves_layout = QVBoxLayout(curves)
        self.curve_context = QLabel(
            "Kurven werden für vorab definierte oder ausschließlich in Train "
            "gewählte repräsentative Marker erzeugt."
        )
        self.curve_context.setWordWrap(True)
        # This label's content is one line per plotted marker (see
        # _draw_roc_pr_curves, "\n".join(context_lines)) -- its line count
        # legitimately varies with the current selection, so (unlike the
        # fixed-height banners above) it's wrapped in a fixed-height
        # scroll area instead of being clipped: the box size stays
        # constant regardless of marker count, and a scrollbar appears
        # for selections with more markers than fit.
        curve_context_scroll = QScrollArea()
        curve_context_scroll.setWidgetResizable(True)
        curve_context_scroll.setWidget(self.curve_context)
        curve_context_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        curve_context_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _reserve_wrapped_label_height(curve_context_scroll, 5)
        curves_layout.addWidget(curve_context_scroll)
        curves_layout.addWidget(
            self._chart_guidance_label(
                "ROC: Sensitivität gegen 1-Spezifität, AUC nahe 1,0 = gute "
                "Trennschärfe, die Diagonale entspricht reinem Zufall. "
                "PR: Precision gegen Recall, informativer bei seltenen "
                "positiven Fällen. Das Band ist ein Bootstrap-"
                "Konfidenzintervall -- ein breites Band bei hoher "
                "Punktschätzung bedeutet trotzdem Unsicherheit. Eine hohe "
                "AUC allein impliziert keinen klinischen Nutzen."
            )
        )
        curve_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.roc_plot = pg.PlotWidget()
        self.roc_plot.setBackground("w")
        self.roc_plot.showGrid(x=True, y=True, alpha=0.2)
        self.roc_plot.setTitle("ROC-Kurve")
        self.roc_plot.setLabel("bottom", "1 - Spezifität")
        self.roc_plot.setLabel("left", "Sensitivität")
        self.roc_plot.addLegend(offset=(8, 8))
        curve_splitter.addWidget(self.roc_plot)
        self.pr_plot = pg.PlotWidget()
        self.pr_plot.setBackground("w")
        self.pr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.pr_plot.setTitle("Precision-Recall-Kurve")
        self.pr_plot.setLabel("bottom", "Recall / Sensitivität")
        self.pr_plot.setLabel("left", "Precision / PPV")
        self.pr_plot.addLegend(offset=(8, 8))
        curve_splitter.addWidget(self.pr_plot)
        curve_splitter.setSizes([600, 600])
        curves_layout.addWidget(curve_splitter, 2)
        self.curve_table = PagedParquetTable(page_size=250)
        curves_layout.addWidget(self.curve_table, 1)
        tabs.addTab(curves, "ROC & PR")

        distributions = QWidget()
        distributions_layout = QVBoxLayout(distributions)
        self.distribution_context = QLabel(
            "Richtungsalignierte Verteilung auf der ausschließlich aus Train "
            "gelernten z-Skala."
        )
        self.distribution_context.setWordWrap(True)
        distributions_layout.addWidget(self.distribution_context)
        distributions_layout.addWidget(
            self._chart_guidance_label(
                "Zeigt die Score-Verteilung getrennt nach klinisch "
                "positiver und negativer Gruppe. Je weniger sich beide "
                "Verteilungen überlappen, desto besser trennt der Marker; "
                "vollständige Überlappung bedeutet keine Trennschärfe. "
                "Die z-Skala ist ausschließlich aus Train gelernt, damit "
                "die Validation-Form nicht in die Skalierung einfließt."
            )
        )
        self.distribution_plot = pg.PlotWidget()
        self.distribution_plot.setBackground("w")
        self.distribution_plot.showGrid(x=True, y=True, alpha=0.2)
        self.distribution_plot.setTitle("Scoreverteilung nach Klasse")
        self.distribution_plot.setLabel(
            "bottom",
            "Richtungsalignierter Train-z-Score",
        )
        self.distribution_plot.setLabel("left", "Anteil innerhalb Klasse")
        self.distribution_plot.addLegend(offset=(8, 8))
        distributions_layout.addWidget(self.distribution_plot, 2)
        self.distribution_table = PagedParquetTable(page_size=250)
        distributions_layout.addWidget(self.distribution_table, 1)
        tabs.addTab(distributions, "Scoreverteilungen")

        operating = QWidget()
        operating_layout = QVBoxLayout(operating)
        self.operating_plot = pg.PlotWidget()
        self.operating_plot.setBackground("w")
        self.operating_plot.showGrid(x=True, y=True, alpha=0.2)
        self.operating_plot.setTitle("Betriebspunkte (Sensitivität/Spezifität)")
        self.operating_plot.setLabel("bottom", "Spezifität")
        self.operating_plot.setLabel("left", "Sensitivität")
        operating_layout.addWidget(self.operating_plot, 2)
        self.operating_table = PagedParquetTable(page_size=250)
        operating_layout.addWidget(self.operating_table, 1)
        operating_layout.addWidget(
            self._chart_guidance_label(
                "Sensitivität, Spezifität, PPV und NPV an konkreten, in "
                "Train fixierten Schwellenwerten -- der klinisch nutzbare "
                "Kompromiss statt nur der Gesamtkurve. PPV/NPV hängen "
                "zusätzlich von der Prävalenz in der jeweiligen Kohorte "
                "ab und sind daher nicht ohne Weiteres auf andere "
                "Populationen übertragbar."
            )
        )
        tabs.addTab(operating, "Betriebspunkte")

        baseline = QWidget()
        baseline_layout = QVBoxLayout(baseline)
        baseline_layout.addWidget(
            self._chart_guidance_label(
                "Direkter Vergleich des neuen Markers gegen die "
                "etablierte klassische ECG-Regel für dieselbe "
                "Fragestellung, mit statistischem Verdikt aus gepaarten "
                "Subject-Cluster-Bootstrap-Konfidenzintervallen (nicht "
                "aus einem einfachen p-Wert-Test). 'Nicht genügend "
                "Evidenz' bedeutet ausdrücklich Unentschieden, nicht "
                "Gleichwertigkeit."
            )
        )
        self.baseline_plot = pg.PlotWidget()
        self.baseline_plot.setBackground("w")
        self.baseline_plot.showGrid(x=True, y=True, alpha=0.2)
        self.baseline_plot.setTitle("Marker vs. klassische Regel (Forest-Plot)")
        self.baseline_plot.setMouseEnabled(x=False, y=False)
        baseline_scroll = QScrollArea()
        baseline_scroll.setWidgetResizable(True)
        baseline_scroll.setFrameShape(QFrame.Shape.NoFrame)
        baseline_scroll.setWidget(self.baseline_plot)
        baseline_layout.addWidget(baseline_scroll, 3)
        baseline_filter_row = QHBoxLayout()
        baseline_filter_row.addWidget(QLabel("Verdikt"))
        self.baseline_verdict_filter = QComboBox()
        for label, value in (
            ("Alle Verdikte", ""),
            ("Neuer Marker signifikant überlegen", "new_marker_significantly_better"),
            ("Kein signifikanter Unterschied", "equivalent"),
            ("Klassische Regel signifikant überlegen", "classical_rule_significantly_better"),
            ("Nicht genügend Evidenz", "insufficient_evidence"),
        ):
            self.baseline_verdict_filter.addItem(label, value)
        self.baseline_verdict_filter.setToolTip(
            "Filtert jede registrierte Leitmarker-gegen-Kriterium-Prüfung "
            "nach ihrem Verdikt (Work Package A: gepaarte "
            "Subject-Cluster-Bootstrap-CI)."
        )
        self.baseline_verdict_filter.currentIndexChanged.connect(
            self.refresh_results
        )
        baseline_filter_row.addWidget(self.baseline_verdict_filter)
        baseline_filter_row.addStretch(1)
        baseline_layout.addLayout(baseline_filter_row)
        self.baseline_table = PagedParquetTable(page_size=200)
        baseline_layout.addWidget(self.baseline_table, 2)
        tabs.addTab(baseline, "Leitlinienvergleich")

        matrix = QWidget()
        matrix_layout = QVBoxLayout(matrix)
        matrix_layout.addWidget(
            self._chart_guidance_label(
                "Matrixübersicht aller Marker gegen alle Endpunkte "
                "gleichzeitig -- zum schnellen Auffinden der stärksten "
                "Kombinationen über das gesamte Panel hinweg, statt "
                "Endpunkt für Endpunkt einzeln zu prüfen."
            )
        )
        self.matrix_plot = pg.PlotWidget()
        self.matrix_plot.setBackground("w")
        self.matrix_plot.setTitle("Marker × Endpunkt (Heatmap)")
        matrix_layout.addWidget(self.matrix_plot, 1)
        tabs.addTab(matrix, "Marker × Endpunkt")

        group_vcg = QWidget()
        group_vcg_layout = QVBoxLayout(group_vcg)
        group_vcg_actions = QHBoxLayout()
        self.group_vcg_context = QLabel(
            "Wählen Sie einen klinischen Endpunkt. Gruppenloops werden nur "
            "aus einem vollständigen Train-/Validation-Feature-Snapshot "
            "berechnet."
        )
        self.group_vcg_context.setWordWrap(True)
        group_vcg_actions.addWidget(self.group_vcg_context, 1)
        self.group_vcg_build_button = self._button(
            "Gruppenansicht berechnen",
            "Erzeugt R-ausgerichtete Kors-VCG-Gruppenmediane, "
            "Subject-Bootstrap-Konfidenzhüllen und Differenzvektoren für den "
            "gewählten Development-Split. Der Locked Test wird nicht gelesen.",
            self.enqueue_group_vcg,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        group_vcg_actions.addWidget(self.group_vcg_build_button)
        self.group_vcg_required_button = self._button(
            "Abnahme-Set einreihen",
            "Legt die sechs vorab festgelegten Validation-Gruppenansichten "
            "für ACS/MI, LVH, reduzierte EF, Rechtsherzbelastung, LAE und "
            "Hyperkaliämie als fortsetzbare Jobs an. Fertige oder laufende "
            "Endpunkte werden nicht doppelt gestartet.",
            self.enqueue_required_group_vcg,
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )
        group_vcg_actions.addWidget(self.group_vcg_required_button)
        group_vcg_layout.addLayout(group_vcg_actions)
        group_vcg_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.group_vcg_view = self._create_group_vcg_view()
        self.group_vcg_view.setToolTip(
            "Kors-VCG-Gruppenmediane in gemeinsamer mV-Skala. Türkis: "
            "positiv; Rot: negativ; gelbe Segmente: positive-minus-negative "
            "Differenzvektoren. Hellere Linienabschnitte liegen zeitlich früher, "
            "kräftigere später; Start/Ende sind als Punkte markiert. "
            "Marginale komponentenweise Bootstrap-Intervalle "
            "werden nicht als gemeinsame räumliche Konfidenzhülle gezeichnet."
        )
        group_vcg_splitter.addWidget(self.group_vcg_view)
        projection_tabs = QTabWidget()
        self.group_vcg_projection_plots = {}
        for projection, axes in (
            ("Frontal XY", ("X (mV)", "Y (mV)", "x", "y")),
            ("Sagittal XZ", ("X (mV)", "Z (mV)", "x", "z")),
            ("Transversal YZ", ("Y (mV)", "Z (mV)", "y", "z")),
        ):
            plot = pg.PlotWidget()
            plot.setBackground("w")
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setAspectLocked(True)
            plot.setLabel("bottom", axes[0])
            plot.setLabel("left", axes[1])
            plot.setToolTip(
                "Orthogonale Projektion der Gruppenmediane in mV. Quadrat = "
                "Beginn, Kreis = t=0 am R-Gipfel, Dreieck = Ende. Eine räumliche "
                "Konfidenzhülle wird nicht behauptet."
            )
            self.group_vcg_projection_plots[projection] = (
                plot,
                axes[2],
                axes[3],
            )
            projection_tabs.addTab(plot, projection)
        self.group_vcg_delta_plot = pg.PlotWidget()
        self.group_vcg_delta_plot.setBackground("w")
        self.group_vcg_delta_plot.showGrid(x=True, y=True, alpha=0.2)
        self.group_vcg_delta_plot.setLabel("bottom", "Zeit relativ zum R-Gipfel (ms)")
        self.group_vcg_delta_plot.setLabel("left", "|positiv − negativ| (mV)")
        self.group_vcg_delta_plot.setToolTip(
            "Betrag des Differenzvektors über der zum R-Gipfel ausgerichteten Zeit. "
            "Das Band ist ein punktweises Subject-Bootstrap-Intervall, "
            "kein simultanes Zeit- oder räumliches Konfidenzband."
        )
        projection_tabs.addTab(self.group_vcg_delta_plot, "Differenz über Zeit")
        group_vcg_splitter.addWidget(projection_tabs)
        group_vcg_splitter.setSizes([650, 550])
        group_vcg_layout.addWidget(
            self._chart_guidance_label(
                "Gemittelte VCG-Schleifen je Gruppe (z. B. positiv vs. "
                "negativ), projiziert auf die drei Ebenen -- macht die "
                "räumliche/morphologische Differenz sichtbar, die ein "
                "1D-Kurvenvergleich nicht zeigen kann. Heller bedeutet früher, "
                "kräftiger bedeutet später; Punktmarken zeigen Start und Ende. "
                "Die Backend-CIs sind "
                "komponentenweise und punktweise; daraus wird keine "
                "gemeinsame räumliche 95-%-Hülle konstruiert."
            )
        )
        group_vcg_layout.addWidget(group_vcg_splitter, 2)
        self.group_vcg_audit_table = PagedParquetTable(page_size=160)
        self.group_vcg_audit_table.setToolTip(
            "Deterministische Auswahl, Signalqualität, verwendete Schläge und "
            "Ausschlussdiagnosen der Gruppenansicht."
        )
        group_vcg_layout.addWidget(self.group_vcg_audit_table, 1)
        tabs.addTab(group_vcg, "Gruppen-VCG")
        self.group_vcg_result_tab = group_vcg

        stability = QWidget()
        stability_layout = QVBoxLayout(stability)
        stability_layout.addWidget(
            self._chart_guidance_label(
                "Train-AUC gegen Validation-AUC je Marker. Punkte nahe "
                "der Diagonale sind stabil; Punkte deutlich UNTER der "
                "Diagonale deuten auf Overfitting in Train hin -- ein "
                "hoher Train-Wert allein ist kein Beleg für einen echten "
                "Zusammenhang."
            )
        )
        self.stability_plot = pg.PlotWidget()
        self.stability_plot.setBackground("w")
        self.stability_plot.showGrid(x=True, y=True, alpha=0.2)
        self.stability_plot.setTitle("Stabilität: Train vs. Validation")
        self.stability_plot.setLabel("bottom", "Train")
        self.stability_plot.setLabel("left", "Validation")
        stability_layout.addWidget(self.stability_plot, 2)
        self.stability_table = PagedParquetTable(page_size=200)
        stability_layout.addWidget(self.stability_table, 1)
        tabs.addTab(stability, "Stabilität")

        sensitivity = QWidget()
        sensitivity_layout = QVBoxLayout(sensitivity)
        sensitivity_layout.addWidget(
            self._chart_guidance_label(
                "Prüft, ob ein Befund konsistent bleibt, wenn Fenster, "
                "Transformation oder Subgruppe (z. B. Geschlecht, Alter, "
                "Herkunft) variiert werden -- ein Fairness- und "
                "Generalisierbarkeits-Check. Stark schwankende AUC über "
                "Subgruppen hinweg ist ein Warnsignal, auch wenn die "
                "Gesamt-AUC gut aussieht."
            )
        )
        self.sensitivity_plot = pg.PlotWidget()
        self.sensitivity_plot.setBackground("w")
        self.sensitivity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sensitivity_plot.setTitle("Sensitivitätsanalyse & Subgruppen")
        self.sensitivity_plot.setLabel("bottom", "ROC-AUC")
        self.sensitivity_plot.setLabel(
            "left",
            "Fenster / Transformation / Subgruppe",
        )
        self.sensitivity_plot.setMouseEnabled(x=False, y=False)
        sensitivity_scroll = QScrollArea()
        sensitivity_scroll.setWidgetResizable(True)
        sensitivity_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sensitivity_scroll.setWidget(self.sensitivity_plot)
        sensitivity_layout.addWidget(sensitivity_scroll, 3)
        self.sensitivity_table = PagedParquetTable(page_size=250)
        sensitivity_layout.addWidget(self.sensitivity_table, 1)
        tabs.addTab(sensitivity, "Sensitivität & Subgruppen")

        models = QWidget()
        models_layout = QVBoxLayout(models)
        models_layout.addWidget(
            self._chart_guidance_label(
                "Zusätzlicher klinischer Nutzen NACH Berücksichtigung "
                "bekannter Risikofaktoren -- der inkrementelle Wert, "
                "nicht die Rohleistung. Kalibrationssteigung nahe 1,0 = "
                "vorhergesagte Risiken passen zur beobachteten Häufigkeit. "
                "Net Benefit über der 'Alle behandeln'/'Niemand "
                "behandeln'-Referenzlinie zeigt den Schwellenwertbereich, "
                "in dem der Marker tatsächlich Entscheidungen verbessert."
            )
        )
        model_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.calibration_plot = pg.PlotWidget()
        self.calibration_plot.setBackground("w")
        self.calibration_plot.showGrid(x=True, y=True, alpha=0.2)
        self.calibration_plot.setTitle("Kalibration")
        self.calibration_plot.setLabel("bottom", "Vorhergesagte Wahrscheinlichkeit")
        self.calibration_plot.setLabel("left", "Beobachtete Häufigkeit")
        model_splitter.addWidget(self.calibration_plot)
        self.decision_plot = pg.PlotWidget()
        self.decision_plot.setBackground("w")
        self.decision_plot.showGrid(x=True, y=True, alpha=0.2)
        self.decision_plot.setTitle("Decision-Curve-Analyse (Net Benefit)")
        self.decision_plot.setLabel("bottom", "Schwellenwahrscheinlichkeit")
        self.decision_plot.setLabel("left", "Net Benefit")
        model_splitter.addWidget(self.decision_plot)
        model_splitter.setSizes([600, 600])
        models_layout.addWidget(model_splitter, 2)
        self.adjusted_model_table = PagedParquetTable(page_size=200)
        models_layout.addWidget(self.adjusted_model_table, 1)
        self.decision_curve_table = PagedParquetTable(page_size=200)
        models_layout.addWidget(self.decision_curve_table, 1)
        tabs.addTab(models, "Adjustierung & Nutzen")

        phenome = QWidget()
        phenome_layout = QVBoxLayout(phenome)
        phenome_controls = QHBoxLayout()
        phenome_controls.addWidget(QLabel("Domäne"))
        self.phenome_domain_combo = QComboBox()
        for label, value in (
            ("Alle getrennten Domänen", ""),
            ("ICD/PhecodeX-Pathologien", "pathology_hypothesis"),
            (
                "Zukünftige Erkrankungen (zensiert)",
                "future_outcome_hypothesis",
            ),
            ("Laborassoziationen", "laboratory"),
            ("Medikamentenexpositionen", "medication"),
            ("Prozedurkontext", "procedure"),
            ("Outcomes", "outcome"),
        ):
            self.phenome_domain_combo.addItem(label, value)
        self.phenome_domain_combo.setToolTip(
            "Pathologie-Hypothesen und klinische Kontextassoziationen werden "
            "getrennt ausgewiesen und niemals semantisch gepoolt."
        )
        self.phenome_domain_combo.currentIndexChanged.connect(
            self.refresh_results
        )
        phenome_controls.addWidget(self.phenome_domain_combo)
        phenome_controls.addStretch(1)
        phenome_layout.addLayout(phenome_controls)
        phenome_layout.addWidget(
            self._chart_guidance_label(
                "Vulkan-Diagramm-Stil: Effektstärke (x) gegen "
                "-log10(FDR-q) (y) -- Punkte weiter oben sind "
                "statistisch sicherer, Punkte weiter rechts/links haben "
                "einen stärkeren Effekt in jeweils eine Richtung. Alle "
                "gezeigten Treffer haben bereits die zweistufige "
                "Train→Validation-FDR-Replikation bestanden; "
                "hypothesengenerierend, keine bestätigten Diagnosekriterien."
            )
        )
        self.phenome_plot = pg.PlotWidget()
        self.phenome_plot.setBackground("w")
        self.phenome_plot.showGrid(x=True, y=True, alpha=0.2)
        self.phenome_plot.setTitle("Breiter Scan: Effekt vs. Signifikanz")
        self.phenome_plot.setLabel("bottom", "Effekt")
        self.phenome_plot.setLabel("left", "-log10(q)")
        phenome_layout.addWidget(self.phenome_plot, 2)
        self.phenome_table = PagedParquetTable(page_size=250)
        phenome_layout.addWidget(self.phenome_table, 1)

        self.promotion_candidates_box = QGroupBox(
            "Kuratierte Hypothesenliste (draft_hypotheses_only)"
        )
        self.promotion_candidates_box.setCheckable(True)
        self.promotion_candidates_box.setChecked(False)
        promotion_layout = QVBoxLayout(self.promotion_candidates_box)
        promotion_note = QLabel(
            "Ein Kandidat je Phänotyp, dedupliziert auf die stärkste "
            "Marker-Assoziation, ausschließlich Biomarker-Rolle (keine "
            "QC-/Klassik-Baseline-Merkmale). Hypothesengenerierend, KEIN "
            "validiertes Diagnosekriterium und NICHT Teil der "
            "SCP-Code-Pathologieliste -- vor jeder klinischen Nutzung ist "
            "ein vorregistriertes Protokoll erforderlich."
        )
        promotion_note.setWordWrap(True)
        promotion_note.setObjectName("mimicInfoBoxWarning")
        promotion_layout.addWidget(promotion_note)
        self.promotion_candidates_summary = QLabel()
        self.promotion_candidates_summary.setWordWrap(True)
        promotion_layout.addWidget(self.promotion_candidates_summary)
        self.promotion_candidates_table = PagedParquetTable(page_size=100)
        self.promotion_candidates_table.setToolTip(
            "effect_size ist die Validation-ROC-AUC des besten Markers je "
            "Phänotyp; effect_ci_low/high sind deren Bootstrap-Konfidenz-"
            "grenzen; train_q_value/validation_q_value sind die "
            "FDR-korrigierten q-Werte aus Trainings- bzw. "
            "Replikationsstufe."
        )
        promotion_layout.addWidget(self.promotion_candidates_table, 1)
        self.promotion_candidates_box.toggled.connect(
            self.promotion_candidates_table.setVisible
        )
        self.promotion_candidates_table.setVisible(False)
        phenome_layout.addWidget(self.promotion_candidates_box)
        tabs.addTab(phenome, "Breiter Scan")

        axes = QWidget()
        axes_layout = QVBoxLayout(axes)
        axes_controls = QHBoxLayout()
        axes_controls.addWidget(QLabel("Achse"))
        self.broad_axis_combo = QComboBox()
        for label, value in (
            ("Alle Achsen", ""),
            ("ICU-Messwerte (invasiv/bettseitig)", "icu_measurement"),
            ("Laborparameter", "laboratory_item"),
            ("Prozedurencodes", "procedure_code"),
            ("DRG-Fallgruppen", "drg_case_mix"),
            ("Erreger", "microbiology_organism"),
            ("CCSR-Diagnosekategorien", "ccsr_category"),
        ):
            self.broad_axis_combo.addItem(label, value)
        self.broad_axis_combo.setToolTip(
            "Jede Achse ist eine eigene FDR-Familie. Ein Treffer auf einer "
            "Achse erhöht die Schwelle der anderen nicht."
        )
        self.broad_axis_combo.currentIndexChanged.connect(self.refresh_results)
        axes_controls.addWidget(self.broad_axis_combo)
        axes_controls.addWidget(QLabel("Zeitliche Lage"))
        self.broad_axis_relation_combo = QComboBox()
        for label, value in (
            ("Alle", ""),
            ("Ziel VOR dem ECG (Folge)", "before_ecg"),
            ("Ziel NACH dem ECG (prognostisch)", "after_ecg"),
            ("gleicher Tag", "same_day_as_ecg"),
            ("zeitgleich", "same_time_as_ecg"),
            ("undatiert / bei Entlassung", "undated_admission_context"),
        ):
            self.broad_axis_relation_combo.addItem(label, value)
        self.broad_axis_relation_combo.setToolTip(
            "Prozeduren eines Aufenthalts verteilen sich annähernd "
            "gleichmäßig auf die Zeit vor und nach dem ECG. Ein Treffer "
            "'davor' bedeutet, dass das ECG die Folge eines Eingriffs zeigt; "
            "'danach', dass es ihm vorausgeht. Beides sind gegensätzliche "
            "Aussagen und werden getrennt gerechnet."
        )
        self.broad_axis_relation_combo.currentIndexChanged.connect(
            self.refresh_results
        )
        axes_controls.addWidget(self.broad_axis_relation_combo)
        axes_controls.addStretch(1)
        axes_layout.addLayout(axes_controls)

        axes_note = QLabel(
            "Datengetriebene Achsen: gescreent wird, was oft genug vorkommt, "
            "nicht was jemand ausgewählt hat. Die Häufigkeitsschwelle steht "
            "vor dem Screen fest; was darunter liegt, wird als "
            "'unterpowert, nicht uninformativ' ausgewiesen statt "
            "weggelassen.\n\nDie ICU-Achse ist wissenschaftlich die "
            "wertvollste: dort stehen invasiv gemessene Größen "
            "(Pulmonalarteriendruck, ZVD, Herzzeitvolumen, SvO2). Ein Treffer "
            "dort ist eine Surrogatfrage -- kann ein ECG-Marker ersetzen, wofür "
            "sonst ein Katheter nötig ist -- und nicht blosse Assoziation. "
            "Alle gezeigten Treffer haben Train-Screening, "
            "Validation-Replikation und Richtungsgleichheit bestanden; "
            "hypothesengenerierend, keine bestätigten Diagnosekriterien."
        )
        axes_note.setWordWrap(True)
        axes_note.setObjectName("mimicInfoBoxWarning")
        axes_layout.addWidget(axes_note)

        self.broad_axis_summary = QLabel("Noch keine Breitachsen-Ergebnisse.")
        self.broad_axis_summary.setWordWrap(True)
        self.broad_axis_summary.setObjectName("mimicSurfaceBox")
        axes_layout.addWidget(self.broad_axis_summary)

        self.broad_axis_plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.broad_continuous_plot = pg.PlotWidget()
        self.broad_continuous_plot.setBackground("w")
        self.broad_continuous_plot.showGrid(x=True, y=True, alpha=0.18)
        self.broad_continuous_plot.setTitle("Stetige Ziele: rho vs. -log10(q)")
        self.broad_continuous_plot.setLabel("bottom", "Spearman rho")
        self.broad_continuous_plot.setLabel("left", "-log10(q)")
        self.broad_binary_plot = pg.PlotWidget()
        self.broad_binary_plot.setBackground("w")
        self.broad_binary_plot.showGrid(x=True, y=True, alpha=0.18)
        self.broad_binary_plot.setTitle("Binäre Ziele: AUC-Abstand vs. -log10(q)")
        self.broad_binary_plot.setLabel("bottom", "AUC - 0,5")
        self.broad_binary_plot.setLabel("left", "-log10(q)")
        self.broad_axis_plot_splitter.addWidget(self.broad_continuous_plot)
        self.broad_axis_plot_splitter.addWidget(self.broad_binary_plot)
        axes_layout.addWidget(self.broad_axis_plot_splitter, 2)

        self.broad_axis_table = PagedParquetTable(page_size=250)
        self.broad_axis_table.setToolTip(
            "effect ist bei stetigen Zielen die Rangkorrelation, bei binären "
            "die ROC-AUC. train_effect zeigt denselben Marker im "
            "Train-Screening; die Richtungen müssen übereinstimmen."
        )
        axes_layout.addWidget(self.broad_axis_table, 2)

        self.broad_axis_coverage_box = QGroupBox(
            "Abdeckung je Achse: was geprüft wurde und was zu selten war"
        )
        self.broad_axis_coverage_box.setCheckable(True)
        self.broad_axis_coverage_box.setChecked(False)
        coverage_layout = QVBoxLayout(self.broad_axis_coverage_box)
        self.broad_axis_coverage_table = PagedParquetTable(page_size=100)
        coverage_layout.addWidget(self.broad_axis_coverage_table)
        axes_layout.addWidget(self.broad_axis_coverage_box)

        tabs.addTab(axes, "Breitachsen")

        transfer = QWidget()
        transfer_layout = QVBoxLayout(transfer)
        transfer_note = QLabel(
            "PTB-XL bleibt eine getrennte Machbarkeits- und Transferreferenz. "
            "ECG-Befundlabels und klinische MIMIC-Endpunkte werden nicht "
            "gepoolt. Der Effektplot zeigt rohe rank-biseriale Effekte: "
            "Kreise sind Einzelmarker, Quadrate eingefrorene Scores; grün "
            "bedeutet gleiche Train-Richtung."
        )
        transfer_note.setWordWrap(True)
        transfer_layout.addWidget(transfer_note)
        self.transfer_plot = pg.PlotWidget()
        self.transfer_plot.setBackground("w")
        self.transfer_plot.showGrid(x=True, y=True, alpha=0.2)
        self.transfer_plot.setTitle("Evidenztransfer: PTB-XL vs. MIMIC")
        self.transfer_plot.setLabel(
            "bottom",
            "PTB-XL Effekt (rank-biserial)",
        )
        self.transfer_plot.setLabel(
            "left",
            "MIMIC Effekt (rank-biserial)",
        )
        self.transfer_plot.setToolTip(
            "Richtungstransfer zwischen ungleichen Referenzendpunkten. "
            "Die Achsen sind keine AUC-Differenz und dürfen nicht als "
            "gepoolte Leistung interpretiert werden."
        )
        self.transfer_plot.setXRange(-1.0, 1.0, padding=0.03)
        self.transfer_plot.setYRange(-1.0, 1.0, padding=0.03)
        transfer_layout.addWidget(self.transfer_plot, 2)
        self.transfer_table = PagedParquetTable(page_size=200)
        transfer_layout.addWidget(self.transfer_table, 1)
        tabs.addTab(transfer, "Evidenzherkunft")

        # --- Kontrolltafel ---------------------------------------------------
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_note = QLabel(
            "Prüft das Verfahren, nicht die Marker. Positivkontrollen sind "
            "vorab festgelegte Zusammenhänge mit bekannter Richtung: Kalium "
            "hebt die T-Welle, Kalzium verkürzt die QT-Zeit. Werden sie im "
            "fertigen Screen nicht wiedergefunden, sind auch dessen "
            "Negativbefunde nicht als Abwesenheit eines Effekts lesbar.\n\n"
            "Negativkontrollen sind Paarungen ohne bekannten Mechanismus. Der "
            "Anteil, der dennoch anschlägt, ist die gemessene "
            "Falsch-Positiv-Rate dieser Kohorte. Sie schließt Konfundierung "
            "ein, die eine FDR-Korrektur nicht erfasst.\n\n"
            "Bewertet wird auf dem Validierungssplit; der Trainingswert steht "
            "zum Vergleich daneben."
        )
        controls_note.setWordWrap(True)
        controls_note.setObjectName("mimicInfoBoxWarning")
        controls_layout.addWidget(controls_note)

        self.control_panel_shrinkage_table = PagedParquetTable(page_size=50)
        self.control_panel_shrinkage_table.setToolTip(
            "Anteil des Effekts, den eine Kontrolle beim Übergang von "
            "Training auf Validierung verliert. shrinkage = 0,36 bedeutet, "
            "dass ein Drittel der Marge über dem Zufall auf Anpassung "
            "entfiel. Nur definiert, wenn der Trainingseffekt die "
            "Effektuntergrenze überschreitet; darunter besteht der Nenner "
            "aus Rauschen."
        )

        self.control_panel_summary = QLabel("Noch keine Kontrolltafel.")
        self.control_panel_summary.setWordWrap(True)
        self.control_panel_summary.setObjectName("mimicSurfaceBox")
        controls_layout.addWidget(self.control_panel_summary)

        self.control_panel_table = PagedParquetTable(page_size=100)
        self.control_panel_table.setToolTip(
            "status: recovered = gefunden und richtig gerichtet; "
            "direction_only = richtige Richtung ohne Signifikanz; "
            "wrong_direction = gefunden, aber entgegengesetzt; not_tested = "
            "Paarung im Screen nicht enthalten; silent_as_expected = "
            "Negativkontrolle ohne Effekt; false_positive = Negativkontrolle "
            "mit Effekt."
        )
        controls_layout.addWidget(self.control_panel_table, 2)
        shrinkage_box = QGroupBox(
            "Effektverlust zwischen Training und Validierung"
        )
        shrinkage_box.setCheckable(True)
        shrinkage_box.setChecked(False)
        shrinkage_layout = QVBoxLayout(shrinkage_box)
        shrinkage_layout.addWidget(self.control_panel_shrinkage_table)
        controls_layout.addWidget(shrinkage_box)
        tabs.addTab(controls, "Kontrolltafel")

        # --- Zielgerichtete Designs ------------------------------------------
        designs = QWidget()
        designs_layout = QVBoxLayout(designs)
        designs_note = QLabel(
            "Fünf Auswertungen mit anderer Fragestellung als der breite "
            "Screen und entsprechend anderen Schwächen.\n\n"
            "Intraindividuell vergleicht denselben Patienten vor und nach "
            "einem Ereignis. Stabile Unterschiede werden reduziert; zeitabhängige "
            "Krankheitsschwere und Begleittherapien bleiben mögliche Störgrößen.\n\n"
            "Spannung/Masse rekonstruiert die linksventrikuläre Masse aus den "
            "Echo-Rohmaßen, die MIMIC nicht direkt ausweist, und sucht Herzen "
            "mit zu geringer Spannung für ihre Muskelmasse -- das klassische "
            "Zeichen einer infiltrativen Kardiomyopathie.\n\n"
            "Zeit bis Ereignis verwendet das ECG als Zeitnullpunkt und "
            "benötigt keinen Diagnosecode. Verschluss stellt die klassischen "
            "Infarktkriterien einer Referenz gegenüber, die das ECG nicht "
            "heranzieht. Zweistufige Auswahl führt Screening-Treffer einer "
            "bestätigenden Prüfung zu."
        )
        designs_note.setWordWrap(True)
        designs_note.setObjectName("mimicInfoBoxWarning")
        designs_layout.addWidget(designs_note)

        self.targeted_design_summary = QLabel(
            "Noch keine zielgerichteten Auswertungen."
        )
        self.targeted_design_summary.setWordWrap(True)
        self.targeted_design_summary.setObjectName("mimicSurfaceBox")
        designs_layout.addWidget(self.targeted_design_summary)

        design_tabs = QTabWidget()
        self.within_subject_table = PagedParquetTable(page_size=200)
        self.within_subject_table.setToolTip(
            "design nennt das Paardesign. spearman_rho ist die Korrelation "
            "zwischen der Änderung des Markers und der Änderung der "
            "Bezugsgröße innerhalb derselben Person. Eine Zeile auswählen und "
            "'Intraindividuell: Effekte' öffnen: dort stehen Unsicherheit, Fallzahlen "
            "und die begrenzte Timing-/Qualitäts-Sensitivität getrennt. "
            "Die Positivkontrolle ist im ausgewählten Lauf zu prüfen; ein Nullbefund "
            "beweist keine Abwesenheit eines Effekts."
        )
        within_page = QWidget()
        within_layout = QVBoxLayout(within_page)
        within_note = QLabel(
            "Ein Punkt je Paar: Änderung der Bezugsgröße (waagrecht) gegen "
            "Änderung des Markers (senkrecht), jeweils innerhalb derselben "
            "Person. Stabile Unterschiede werden reduziert, aber nicht sämtliche "
            "Störgrößen kontrolliert. Die Timing-/Qualitätsanalyse ist nur eine Sensitivität.\n\n"
            "Elektrolyte und pH werden getrennt nach Einheit dargestellt. "
            "Dialyse verändert zugleich den Volumenstatus; die Darstellung "
            "zeigt eine physiologische Assoziation, keinen isolierten "
            "Kausalnachweis."
        )
        within_note.setWordWrap(True)
        within_note.setObjectName("mimicInfoBoxWarning")
        within_layout.addWidget(within_note)
        within_controls = QHBoxLayout()
        within_controls.addWidget(QLabel("Labor: "))
        self.within_analyte_combo = QComboBox()
        self.within_analyte_combo.setMinimumWidth(190)
        self.within_analyte_combo.currentIndexChanged.connect(
            self._redraw_within_subject_scatter
        )
        within_controls.addWidget(self.within_analyte_combo)
        within_controls.addWidget(QLabel("Marker:"))
        self.within_marker_combo = QComboBox()
        self.within_marker_combo.setMinimumWidth(320)
        self.within_marker_combo.currentIndexChanged.connect(
            self._redraw_within_subject_scatter
        )
        within_controls.addWidget(self.within_marker_combo)
        within_controls.addStretch(1)
        within_layout.addLayout(within_controls)
        self.within_subject_plot = pg.PlotWidget()
        self.within_subject_plot.setBackground("w")
        self.within_subject_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.within_subject_plot.getAxis("left").enableAutoSIPrefix(False)
        self.within_subject_plot.showGrid(x=True, y=True, alpha=0.2)
        self.within_subject_plot.setLabel("bottom", "Aenderung der Bezugsgröße")
        self.within_subject_plot.setLabel("left", "Aenderung des Markers")
        within_layout.addWidget(self.within_subject_plot, 2)
        within_layout.addWidget(self.within_subject_table, 1)
        self.within_effect_view = PairedEffectWidget()
        self.within_subject_table.selection_changed.connect(
            lambda: self.within_effect_view.set_result(self.within_subject_table.selected_row())
        )
        self.within_subject_table.model.modelReset.connect(
            lambda: self.within_effect_view.set_result({})
        )
        design_tabs.addTab(within_page, "Intraindividuell")
        design_tabs.addTab(self.within_effect_view, "Intraindividuell: Effekte")
        self.dose_pair_view = DosePairWidget()
        self.dose_pair_view.open_ecg_requested.connect(self._open_dose_pair_ecg)
        design_tabs.addTab(self.dose_pair_view, "Dosis: EKG-Paare")
        self.joint_lab_coverage_view = JointLabCoverageWidget()
        design_tabs.addTab(self.joint_lab_coverage_view, "Gemeinsame Laborreferenzen")

        self.voltage_mass_table = PagedParquetTable(page_size=200)
        self.voltage_mass_table.setToolTip(
            "auc_ratio bewertet das Verhältnis Spannung/Masse, auc_voltage "
            "und auc_mass seine beiden Bestandteile. gain_over_best_part ist "
            "nur ein deskriptiver Vergleich einzelner Größen, kein adjustierter "
            "Zusatznutzen gegenüber einem gemeinsamen Komponentenmodell. "
            "population gibt an, ob nur verdickte Wände einbezogen wurden; "
            "nachträglich gewählte Untergruppen bleiben Discovery."
        )
        voltage_page = QWidget()
        voltage_layout = QVBoxLayout(voltage_page)
        voltage_note = QLabel(
            "Ein Punkt je ECG-Echo-Paar: linksventrikuläre Masse gegen "
            "QRS-Spannung. Hohe Masse bei niedriger Spannung beschreibt eine "
            "elektrisch-anatomische Diskordanz, beweist aber keine Amyloidose. "
            "Der Quotient enthält keine von seinen Komponenten unabhängige "
            "Messinformation.\n\n"
            "Gezeigt werden explorative Assoziationen. Referenzqualität, "
            "adjustierter Komponentenvergleich und unabhängige Validierung "
            "sind für Aussagen über klinischen Zusatznutzen erforderlich."
        )
        voltage_note.setWordWrap(True)
        voltage_note.setObjectName("mimicInfoBoxWarning")
        voltage_layout.addWidget(voltage_note)
        self.voltage_mass_plot = pg.PlotWidget()
        self.voltage_mass_plot.setBackground("w")
        self.voltage_mass_plot.showGrid(x=True, y=True, alpha=0.2)
        self.voltage_mass_plot.setLabel("bottom", "LV-Masse (g, Devereux)")
        self.voltage_mass_plot.setLabel("left", "QRS-Spannung")
        self.voltage_mass_plot.addLegend()
        voltage_layout.addWidget(self.voltage_mass_plot, 2)
        voltage_layout.addWidget(self.voltage_mass_table, 1)
        design_tabs.addTab(voltage_page, "Spannung/Masse")

        self.voltage_mass_models_table = PagedParquetTable(page_size=200)
        self.voltage_mass_models_table.setToolTip(
            "Gleiche Patienten je Modell: klinische Kovariaten, Einzelkomponenten, "
            "Quotient und gemeinsame Komponentenmodelle. validation_auc und "
            "delta_auc_vs_joint_quadratic mit verfügbaren Bootstrap-Intervallen. "
            "confirmation_log_or und confirmation_q prüfen die bedingte "
            "Assoziation des trainingsorientierten Quotienten, nicht den AUC-Gewinn. "
            "n_excluded_* zerlegt Ausschluesse in der Reihenfolge Referenz, "
            "Signalqualitaet, Kovariaten, Echo-Masse, Spannung. Jeder Patient "
            "wird nur beim ersten fehlgeschlagenen Gate gezaehlt. "
            "Explorativ; nicht schätzbare Modelle sind keine negativen Ergebnisse."
        )
        component_page = QWidget()
        component_layout = QVBoxLayout(component_page)
        component_note = QLabel(
            "Adjustierter Modellvergleich – Training und Validation getrennt. "
            "Die Echo-Geometrie wird am unveränderten Patientenindex ergänzt. "
            "Das ist eine Querschnittsanalyse, keine Vorhersage allein aus dem EKG. "
            "Konfidenzintervalle sind nicht multiplikitätsadjustiert. "
            "Die q-Werte gehören zur eigenen Spannung/Masse-Testfamilie und "
            "beweisen keinen Vorhersagegewinn. "
            "Alte Läufe enthalten diese Auswertung möglicherweise noch nicht."
        )
        component_note.setWordWrap(True)
        component_layout.addWidget(component_note)
        component_layout.addWidget(self.voltage_mass_models_table)
        design_tabs.addTab(component_page, "Spannung/Masse: Modelle")

        recovery_page = QWidget()
        recovery_layout = QVBoxLayout(recovery_page)
        recovery_note = QLabel(
            "Retrospektive Erholungsverläufe je Patient und Dialyseepisode. "
            "Der Halb-Peak-Übertritt ist beobachtungsabhängig, keine physiologische Halbwertszeit. "
            "Normalisierungsrestabweichungen verwenden unabhängige Laborreferenzen "
            "mit gleichem Assay und Referenzbereich. Keine kausale Therapiewirkung; "
            "Kontrollgruppenvergleich noch nicht berechnet. Fehlende oder abgelehnte "
            "Episoden sind keine negativen Befunde. Alte Läufe enthalten diese Tabelle nicht."
        )
        recovery_note.setWordWrap(True)
        recovery_layout.addWidget(recovery_note)
        self.recovery_episode_table = PagedParquetTable(page_size=200)
        self.recovery_reference_view = RecoveryReferenceWidget()
        recovery_layout.addWidget(self.recovery_reference_view)
        self.recovery_episode_table.selection_changed.connect(
            lambda: self.recovery_reference_view.set_result(self.recovery_episode_table.selected_row())
        )
        self.recovery_episode_table.model.modelReset.connect(
            lambda: self.recovery_reference_view.set_result({})
        )
        recovery_layout.addWidget(self.recovery_episode_table)
        design_tabs.addTab(recovery_page, "Erholung: Episoden")

        self.time_to_event_table = PagedParquetTable(page_size=200)
        self.time_to_event_table.setToolTip(
            "hazard_ratio_per_sd ist die Risikoerhöhung je "
            "Standardabweichung des Markers, adjustiert für Alter und "
            "Geschlecht. Jeder Marker erscheint zweimal: auf dem ersten ECG "
            "je Patient und auf allen ECGs mit clusterrobuster Varianz. "
            "Übereinstimmende Werte zeigen, dass das Ergebnis nicht von einem "
            "einzelnen Design getragen wird."
        )
        survival_page = QWidget()
        survival_layout = QVBoxLayout(survival_page)
        survival_note = QLabel(
            "Absolute Ein-Jahres-Sterblichkeit je Marker-Terzil. Die "
            "Terzilgrenzen stammen aus dem Trainingssplit und werden "
            "unverändert auf die Validierung angewandt; splitweise bestimmte "
            "Grenzen würden die Kurven bei jedem beliebigen Marker "
            "trennen.\n\n"
            "Der Reihe nach auseinanderlaufende Kurven zeigen ein mit dem "
            "Marker steigendes Risiko. Kreuzende Kurven oder ein oben "
            "liegendes unterstes Terzil zeigen einen nicht monotonen "
            "Zusammenhang; das Hazard Ratio unterstellt einen linearen "
            "Verlauf und fasst solche Marker unzureichend zusammen."
        )
        survival_note.setWordWrap(True)
        survival_note.setObjectName("mimicInfoBoxWarning")
        survival_layout.addWidget(survival_note)

        survival_controls = QHBoxLayout()
        survival_controls.addWidget(QLabel("Marker:"))
        self.survival_marker_combo = QComboBox()
        self.survival_marker_combo.setMinimumWidth(320)
        self.survival_marker_combo.currentIndexChanged.connect(
            self._redraw_survival_curves
        )
        survival_controls.addWidget(self.survival_marker_combo)
        survival_controls.addStretch(1)
        survival_layout.addLayout(survival_controls)

        survival_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.survival_plot = pg.PlotWidget()
        self.survival_plot.setBackground("w")
        self.survival_plot.showGrid(x=True, y=True, alpha=0.2)
        self.survival_plot.setLabel("left", "Kumulative Sterblichkeit (%)")
        self.survival_plot.setLabel("bottom", "Tage seit dem ECG")
        self.survival_plot.addLegend()
        survival_splitter.addWidget(self.survival_plot)

        self.hazard_forest_plot = pg.PlotWidget()
        self.hazard_forest_plot.setBackground("w")
        self.hazard_forest_plot.showGrid(x=True, alpha=0.2)
        self.hazard_forest_plot.setLabel("bottom", "Hazard Ratio je SD")
        self.hazard_forest_plot.setToolTip(
            "Je Marker zwei Schätzungen: erstes ECG je Patient und alle ECGs "
            "mit clusterrobuster Varianz. Übereinstimmende Punkte zeigen, "
            "dass das Ergebnis nicht von einem einzelnen Design getragen "
            "wird. Die senkrechte Linie bei 1,0 entspricht keinem Effekt."
        )
        survival_splitter.addWidget(self.hazard_forest_plot)
        survival_layout.addWidget(survival_splitter, 2)
        survival_layout.addWidget(self.time_to_event_table, 1)
        design_tabs.addTab(survival_page, "Zeit bis Ereignis")

        self.occlusion_yield_table = PagedParquetTable(page_size=100)
        self.occlusion_yield_table.setToolTip(
            "sensitivity ist der Anteil der Revaskularisations-Ereignisse, "
            "den das Kriterium erfasst. Der Endpunkt ist ein Surrogat ohne "
            "angiographischen Verschlussnachweis; die Werte sind nicht als "
            "verfehlte Gefäßverschlüsse lesbar."
        )
        design_tabs.addTab(self.occlusion_yield_table, "Verschluss")

        self.two_stage_table = PagedParquetTable(page_size=200)
        self.two_stage_table.setToolTip(
            "arm trennt die beiden Wege: prespecified_curated sind die vorab "
            "von Hand zugeordneten Paare, two_stage_confirmatory die aus dem "
            "Trainingssplit ausgewählten. Nur erstere tragen eine Aussage "
            "ohne Mehrfachtest-Argument; n_screened_for_endpoint nennt die "
            "Zahl der Marker, gegen die sich ein ausgewählter durchsetzen "
            "musste."
        )
        design_tabs.addTab(self.two_stage_table, "Zweistufige Auswahl")

        redundancy_page = QWidget()
        redundancy_layout = QVBoxLayout(redundancy_page)
        redundancy_note = QLabel(
            "Marker, die dieselbe Größe unter verschiedenen Namen messen. "
            "Ohne diese Gruppierung erscheint ein Befund in einer "
            "Trefferliste so oft, wie es Schreibweisen dafür gibt.\n\n"
            "Kein Marker wird entfernt. Die Gruppe weist aus, dass mehrere "
            "Zeilen einen Befund darstellen, und benennt den Vertreter, den "
            "der Screen stellvertretend prüft. Gruppen, deren Mitglieder sich "
            "ausschließlich im Wellenabschnitt unterscheiden, sind daran "
            "erkennbar, dass ihre Namen bis auf das Segmentpräfix "
            "übereinstimmen."
        )
        redundancy_note.setWordWrap(True)
        redundancy_note.setObjectName("mimicInfoBoxWarning")
        redundancy_layout.addWidget(redundancy_note)
        self.redundancy_summary = QLabel("Noch keine Redundanzgruppen.")
        self.redundancy_summary.setWordWrap(True)
        self.redundancy_summary.setObjectName("mimicSurfaceBox")
        redundancy_layout.addWidget(self.redundancy_summary)
        self.redundancy_table = PagedParquetTable(page_size=250)
        self.redundancy_table.setToolTip(
            "cluster_id fasst zusammen, was sich bei |rho| >= 0,99 nicht "
            "unterscheiden lässt. is_representative markiert den "
            "stellvertretend geprüften Marker; coverage ist der Anteil "
            "auswertbarer ECGs, nach dem der Vertreter gewählt wurde."
        )
        redundancy_layout.addWidget(self.redundancy_table, 2)
        design_tabs.addTab(redundancy_page, "Markergruppen")
        designs_layout.addWidget(design_tabs, 3)
        tabs.addTab(designs, "Zielgerichtete Designs")
        self.targeted_design_result_tab = designs

        guideline = QWidget()
        guideline_layout = QVBoxLayout(guideline)
        guideline_note = QLabel(
            "Die Bausteine der Befundungsleitlinie. Ein Screen sagt, welcher "
            "Marker welchen Endpunkt trennt; das ist noch kein Befund. Ein "
            "Befund braucht einen Referenzwert, eine bekannte Messgenauigkeit, "
            "eine gemessene Transformationsabhängigkeit und eine "
            "Anspruchsobergrenze.\n\n"
            "Referenzbereiche sind KEINE Normwerte. Die Vergleichskohorte "
            "sind Krankenhauspatienten ohne dokumentierte Herzerkrankung, ohne "
            "herzwirksames Medikament, ohne Device und ohne Intensivepisode -- "
            "nicht Gesunde. Jede Zeile trägt die Population mit. Normwerte "
            "brauchen eine ambulante Kohorte (CODE-15)."
        )
        guideline_note.setWordWrap(True)
        guideline_note.setObjectName("mimicInfoBoxWarning")
        guideline_layout.addWidget(guideline_note)
        self.guideline_tabs = QTabWidget()
        guideline_tabs = self.guideline_tabs

        evidence_page = QWidget()
        evidence_layout = QVBoxLayout(evidence_page)
        self.evidence_ladder_summary = QLabel(
            "Noch kein evidenzgestützter Kriterienkatalog."
        )
        self.evidence_ladder_summary.setWordWrap(True)
        self.evidence_ladder_summary.setObjectName("mimicSurfaceBox")
        evidence_layout.addWidget(self.evidence_ladder_summary)
        self.evidence_ladder_plot = pg.PlotWidget()
        self.evidence_ladder_plot.setBackground("w")
        self.evidence_ladder_plot.showGrid(x=False, y=True, alpha=0.16)
        self.evidence_ladder_plot.setTitle("Evidenzleiter der Kriterien")
        self.evidence_ladder_plot.setLabel("left", "Anzahl Kriterien")
        self.evidence_ladder_plot.setMouseEnabled(x=False, y=False)
        evidence_layout.addWidget(self.evidence_ladder_plot, 1)
        self.candidate_evidence_table = PagedParquetTable(page_size=250)
        self.candidate_evidence_table.setToolTip(
            "Jede Zeile nennt die Evidenzstufe und die bestandenen sowie "
            "fehlenden Gates. guideline_candidate ist keine klinische "
            "Leitlinie; clinically_validated verlangt externe Validierung."
        )
        evidence_layout.addWidget(self.candidate_evidence_table, 2)
        guideline_tabs.addTab(evidence_page, "Evidenzleiter")
        from .lead_comparison_widget import LeadComparisonWidget
        self.lead_comparison_view = LeadComparisonWidget()
        guideline_tabs.addTab(self.lead_comparison_view, "Ableitungen je Endpunkt")

        population_page = QWidget()
        population_layout = QVBoxLayout(population_page)
        self.population_cohort_summary = QLabel(
            "Noch keine CODE-15-/SaMi-Trop-Auswertung."
        )
        self.population_cohort_summary.setWordWrap(True)
        self.population_cohort_summary.setObjectName("mimicSurfaceBox")
        population_layout.addWidget(self.population_cohort_summary)
        self.population_mortality_plot = pg.PlotWidget()
        self.population_mortality_plot.setBackground("w")
        self.population_mortality_plot.showGrid(x=True, y=True, alpha=0.16)
        self.population_mortality_plot.setTitle(
            "Prospektive Mortalität: adjustierte Hazard Ratios"
        )
        self.population_mortality_plot.setLabel(
            "bottom", "Hazard Ratio je SD (logarithmisch)"
        )
        self.population_mortality_plot.setMouseEnabled(x=False, y=False)
        population_layout.addWidget(self.population_mortality_plot, 1)
        self.population_tabs = QTabWidget()
        population_tabs = self.population_tabs
        self.population_coverage_table = PagedParquetTable(page_size=250)
        population_tabs.addTab(self.population_coverage_table, "Messbarkeit")
        self.population_loader_acceptance_table = PagedParquetTable(page_size=50)
        self.population_loader_acceptance_table.setToolTip(
            "Ausgeführte Plausibilitätsprüfungen für Lead-Reihenfolge, "
            "Resampling und Delineation. Ein fehlgeschlagener Voll-Lauf-Gate "
            "sperrt alle CODE-15-Ergebnisse."
        )
        population_tabs.addTab(
            self.population_loader_acceptance_table,
            "Loader-QC",
        )
        self.code15_reference_table = PagedParquetTable(page_size=250)
        self.code15_reference_table.setToolTip(
            "Primärversorgungs-ECGs mit durchgehend normalem ECG-Befund; "
            "keine verifiziert gesunde Referenzpopulation."
        )
        population_tabs.addTab(self.code15_reference_table, "CODE-15-Referenz")
        self.population_diagnostic_table = PagedParquetTable(page_size=250)
        self.population_diagnostic_table.setToolTip(
            "Development bestimmt nur die Richtung; Validation bestätigt "
            "sie patientengetrennt. Die Labels annotieren dasselbe ECG und "
            "sind keine unabhängige klinische Referenz; normal_ecg ist laut "
            "Datensatzdokumentation eine automatische Annotation."
        )
        population_tabs.addTab(
            self.population_diagnostic_table,
            "CODE-15-Befundlabels",
        )
        self.population_agreement_table = PagedParquetTable(page_size=250)
        self.population_agreement_table.setToolTip(
            "Bland-Altman-Bias und 95%-Uebereinstimmungsgrenzen der direkten "
            "8-Lead-Winkel gegen Kors, inverse Dower und SVD/PCA. Eine gute "
            "Korrelation allein belegt keine Austauschbarkeit."
        )
        population_tabs.addTab(
            self.population_agreement_table,
            "Direct-8-Lead-Agreement",
        )
        self.population_mortality_table = PagedParquetTable(page_size=250)
        population_tabs.addTab(self.population_mortality_table, "Mortalität")
        self.population_value_table = PagedParquetTable(page_size=100)
        self.population_value_table.setToolTip("Zusätzliche bekannte Patienten, Ereignisse und Messverfügbarkeit. "
            "Mehr Validierungszeilen sind kein Nachweis einer Modellverbesserung. Unbekannte Identitäten bleiben sichtbar.")
        population_tabs.addTab(self.population_value_table, "Messbarer Datenbankmehrwert")
        self.external_value_table = PagedParquetTable(page_size=100)
        self.external_value_table.setToolTip(
            "Challenge-Kohorten: bekannte Patienten, exakte Signalkopien und Messbarkeit. "
            "Verschiedene Quelldateien sind kein Nachweis unabhängiger Patienten.")
        population_tabs.addTab(self.external_value_table, "Challenge · Datenbankmehrwert")
        self.external_annotation_table = PagedParquetTable(page_size=250)
        self.external_annotation_table.setToolTip(
            "Übereinstimmung mit EKG-Annotationen, kein unabhängiger Echo-/Labornachweis. "
            "Unbekannte Labels bleiben unbekannt; Signalkopien werden vor der Auswertung ausgeschlossen.")
        population_tabs.addTab(self.external_annotation_table, "Challenge · Befundübereinstimmung")
        self.population_chagas_table = PagedParquetTable(page_size=250)
        self.population_chagas_table.setToolTip(
            "Alters-/geschlechtsgematchter, aber standortkonfundierter "
            "SaMi-Trop-vs.-CODE-15-Kontrast; nur explorativ."
        )
        population_tabs.addTab(self.population_chagas_table, "Chagas-Kontrast")
        gwas_page = QWidget()
        gwas_layout = QVBoxLayout(gwas_page)
        self.gwas_context_summary = QLabel(
            "Noch kein GWAS-Machbarkeitsinventar."
        )
        self.gwas_context_summary.setWordWrap(True)
        self.gwas_context_summary.setObjectName("mimicInfoBoxWarning")
        gwas_layout.addWidget(self.gwas_context_summary)
        self.gwas_inventory_table = PagedParquetTable(page_size=100)
        self.gwas_inventory_table.setToolTip(
            "Quelldatei-, Schema-, Wertebereichs- und SHA-256-Prüfung. "
            "Die Tabelle ist kein MR- oder PRS-Ergebnis."
        )
        gwas_layout.addWidget(self.gwas_inventory_table, 1)
        population_tabs.addTab(gwas_page, "GWAS-Kontext")
        population_layout.addWidget(population_tabs, 2)
        guideline_tabs.addTab(population_page, "Externe Populationen")

        self.reference_range_summary = QLabel("Noch keine Referenzbereiche.")
        self.reference_range_summary.setWordWrap(True)
        self.reference_range_summary.setObjectName("mimicSurfaceBox")
        reference_page = QWidget()
        reference_layout = QVBoxLayout(reference_page)
        reference_layout.addWidget(self.reference_range_summary)
        self.reference_range_table = PagedParquetTable(page_size=250)
        self.reference_range_table.setToolTip(
            "percentile_2_5 bis percentile_97_5 ist das Referenzintervall der "
            "Schicht, subject_count die Zahl der Personen dahinter. "
            "estimable = false heißt, die Schicht ist zu klein -- die Zeile "
            "steht trotzdem da, damit eine Lücke sichtbar ist statt zu "
            "fehlen. population und vcg_transform gehören zu jedem Wert."
        )
        reference_layout.addWidget(self.reference_range_table, 3)
        self.reference_cohort_table = PagedParquetTable(page_size=50)
        self.reference_cohort_table.setToolTip(
            "Was die Vergleichskohorte gekostet hat, je Kriterium. Die erste "
            "Frage zu jedem Referenzbereich ist, wer ausgeschlossen wurde."
        )
        reference_layout.addWidget(self.reference_cohort_table, 1)
        guideline_tabs.addTab(reference_page, "Referenzbereiche")

        transform_page = QWidget()
        transform_layout = QVBoxLayout(transform_page)
        self.transform_marker_combo = QComboBox()
        self.transform_marker_combo.setAccessibleName("Marker für den Transformationsvergleich")
        self.transform_marker_combo.currentTextChanged.connect(
            lambda _text: self._draw_transform_agreement(self._transform_agreement_source, reload=False)
            if getattr(self, "_transform_agreement_source", None) is not None else None
        )
        transform_layout.addWidget(self.transform_marker_combo)
        self.transform_agreement_summary = QLabel()
        self.transform_agreement_summary.setWordWrap(True)
        transform_layout.addWidget(self.transform_agreement_summary)
        self.transform_agreement_plot = pg.PlotWidget()
        self.transform_agreement_plot.setBackground("w")
        self.transform_agreement_plot.showGrid(x=True, y=True, alpha=0.16)
        self.transform_agreement_plot.setTitle(
            "Bland-Altman: Bias und 95-%-Uebereinstimmungsgrenzen"
        )
        self.transform_agreement_plot.setLabel("bottom", "Differenz der Transformationen")
        self.transform_agreement_plot.setMouseEnabled(x=False, y=False)
        transform_layout.addWidget(self.transform_agreement_plot, 1)
        self.transform_agreement_table = PagedParquetTable(page_size=100)
        self.transform_agreement_table.setToolTip(
            "bias ist der systematische Versatz zwischen Kors und inversem "
            "Dower, limit_of_agreement_low/high der Bereich, in dem 95 % der "
            "Einzeldifferenzen liegen. Ein kleiner Bias bei weiten "
            "Übereinstimmungsgrenzen bedeutet, dass die beiden "
            "Transformationen im Mittel übereinstimmen, am einzelnen "
            "Patienten aber nicht austauschbar sind; ein Schwellenwert ist "
            "dann nicht zwischen ihnen übertragbar."
        )
        transform_layout.addWidget(self.transform_agreement_table, 2)
        guideline_tabs.addTab(transform_page, "Transformation")

        self.reporting_set_table = PagedParquetTable(page_size=250)
        self.reporting_set_table.setToolTip(
            "included sagt, ob der Marker alle vier Bedingungen erfüllt; "
            "excluded_reason nennt jede verfehlte. Ein handverlesener "
            "Berichtssatz lässt sich nicht prüfen, dieser schon."
        )
        guideline_tabs.addTab(self.reporting_set_table, "Berichtssatz")

        self.reporting_algorithm_table = PagedParquetTable(page_size=250)
        self.reporting_algorithm_table.setToolTip(
            "claim_ceiling ist die Obergrenze dessen, was die Zeile behaupten "
            "darf, und wird nachgeschlagen statt aus der Trennschärfe "
            "abgeleitet: ein ECG-definierter Endpunkt bleibt analytische "
            "Konkordanz, ein selten verifizierter bleibt deskriptiv."
        )
        guideline_tabs.addTab(
            self.reporting_algorithm_table, "Befundungsalgorithmus"
        )

        self.mandatory_context_table = PagedParquetTable(page_size=25)
        self.mandatory_context_table.setToolTip(
            "Die vier Angaben, ohne die ein Messwert nicht interpretierbar "
            "ist. Jede steht hier, weil sie gemessen den Wert verschiebt."
        )
        guideline_tabs.addTab(self.mandatory_context_table, "Pflichtangaben")

        guideline_layout.addWidget(guideline_tabs, 3)
        self.guideline_result_tab = guideline
        tabs.addTab(guideline, "Befundungsleitlinie")
        from .research_results_widget import ResearchResultsWidget
        self.research_results_view = ResearchResultsWidget()
        tabs.addTab(self.research_results_view, "Scores & Fokusforschung")

        result_view_tab_tooltips = {
            "Kontrolltafel": (
                "Prüfung des Verfahrens statt der Marker: ob der Lauf "
                "Zusammenhänge mit vorab bekannter Richtung wiederfindet. "
                "Werden sie nicht wiedergefunden, sind auch die "
                "Negativbefunde des Laufs nicht als Abwesenheit lesbar."
            ),
            "Zielgerichtete Designs": (
                "Fünf Auswertungen mit anderer Fragestellung als der breite "
                "Screen: intraindividueller Vergleich, Spannung gegen "
                "Herzmuskelmasse, Zeit bis zum Ereignis, Ausbeute der "
                "Infarktkriterien und die Auswahl der "
                "Bestätigungskandidaten."
            ),
            "Ergebnisnavigator": (
                "Übersichtstabelle, die Marker und Pathologien nach "
                "Leistungsfähigkeit verknüpft -- Ausgangspunkt für alle "
                "anderen Ansichten hier."
            ),
            "ROC & PR": (
                "ROC-Kurve (Sensitivität vs. 1-Spezifität) und "
                "Precision-Recall-Kurve mit Bootstrap-Konfidenzband. "
                "AUC nahe 1,0 = gute Trennschärfe; die Diagonale entspricht "
                "reinem Zufall."
            ),
            "Scoreverteilungen": (
                "Verteilung des Marker-Scores getrennt nach klinisch "
                "positiver und negativer Gruppe. Je weniger die beiden "
                "Verteilungen sich überlappen, desto besser trennt der "
                "Marker."
            ),
            "Betriebspunkte": (
                "Sensitivität, Spezifität, PPV und NPV an konkreten, im "
                "Training festgelegten Schwellenwerten -- der klinisch "
                "nutzbare Kompromiss statt nur der Gesamtkurve."
            ),
            "Leitlinienvergleich": (
                "Direkter Vergleich des neuen Markers gegen die etablierte "
                "klassische ECG-Regel für dieselbe Fragestellung, mit "
                "statistischem Verdikt (besser/gleichwertig/schlechter)."
            ),
            "Marker × Endpunkt": (
                "Matrixübersicht aller Marker gegen alle Endpunkte "
                "gleichzeitig -- zum schnellen Auffinden der stärksten "
                "Kombinationen."
            ),
            "Gruppen-VCG": (
                "Gemittelte VCG-Schleifen je Gruppe (z. B. positiv vs. "
                "negativ), projiziert auf die drei Ebenen -- zeigt die "
                "räumliche/morphologische Differenz visuell."
            ),
            "Stabilität": (
                "Wie sehr sich Marker-Effekt/-Rang über Resampling und "
                "Panels ändert. Hohe Stabilität spricht für einen robusten "
                "Befund statt eines Zufallsartefakts."
            ),
            "Sensitivität & Subgruppen": (
                "Prüft, ob der Befund in Untergruppen (z. B. Geschlecht, "
                "Alter, Herkunft) konsistent bleibt -- Fairness- und "
                "Generalisierbarkeits-Check."
            ),
            "Adjustierung & Nutzen": (
                "Zusätzlicher klinischer Nutzen des Markers NACH "
                "Berücksichtigung bekannter Risikofaktoren -- der "
                "inkrementelle Wert, nicht nur die Rohleistung."
            ),
            "Breiter Scan": (
                "Ergebnisse des explorativen PheWAS-/Kontext-Scans (siehe "
                "Bereich-/Pathologieart-Filter). Hypothesengenerierend, "
                "kein bestätigtes Diagnosekriterium."
            ),
            "Evidenzherkunft": (
                "Richtungstransfer der Evidenz zwischen PTB-XL und MIMIC "
                "-- keine gepoolte Leistungskennzahl."
            ),
        }
        for index in range(tabs.count()):
            title = tabs.tabText(index)
            self.result_view_combo.addItem(title, index)
            item_tooltip = result_view_tab_tooltips.get(title)
            if item_tooltip:
                self.result_view_combo.setItemData(
                    self.result_view_combo.count() - 1,
                    item_tooltip,
                    Qt.ItemDataRole.ToolTipRole,
                )
        self.result_view_combo.currentIndexChanged.connect(
            tabs.setCurrentIndex
        )
        tabs.currentChanged.connect(
            self.result_view_combo.setCurrentIndex
        )
        self._results_view_opacity = QGraphicsOpacityEffect(tabs)
        tabs.setGraphicsEffect(self._results_view_opacity)
        self._results_view_opacity.setOpacity(1.0)
        tabs.currentChanged.connect(self._animate_results_view_change)
        # "Leitlinienvergleich" (new marker vs. classical guideline rule)
        # is the most immediately relevant view, so it is shown first by
        # default; the physical tab order (and therefore the drilldown
        # from the Ergebnisnavigator, which jumps to index 0) is left
        # untouched.
        tabs.setCurrentWidget(baseline)
        tabs.tabBar().hide()
        advanced_contents_layout.addWidget(tabs, 1)
        advanced_layout.addWidget(self.advanced_results_contents)
        self.advanced_results_box.toggled.connect(
            self.advanced_results_contents.setVisible
        )
        self.advanced_results_contents.setVisible(True)
        layout.addWidget(self.advanced_results_box, 3)
        _disable_auto_si_prefix(
            self.performance_plot,
            self.roc_plot,
            self.pr_plot,
            self.distribution_plot,
            self.operating_plot,
            self.baseline_plot,
            self.matrix_plot,
            self.stability_plot,
            self.sensitivity_plot,
            self.calibration_plot,
            self.decision_plot,
            self.evidence_ladder_plot,
            self.transform_agreement_plot,
            self.population_mortality_plot,
            self.phenome_plot,
            self.transfer_plot,
            *(
                item[0]
                for item in self.group_vcg_projection_plots.values()
            ),
        )
        self.results_atlas_tabs.addTab(
            self.results_legacy_page, "Alle Forschungsansichten"
        )
        self.results_atlas_tabs.currentChanged.connect(
            self._results_atlas_tab_changed
        )
        root_layout.addWidget(self.results_atlas_tabs, 1)
        self.results_mode_combo.currentIndexChanged.connect(
            self._results_mode_changed
        )
        self._results_mode_changed()
        return page

    def _results_mode_changed(self, *_args) -> None:
        detailed = self.results_mode_combo.currentData() == "detailed"
        if not detailed:
            self.results_atlas_tabs.setCurrentIndex(0)
        self.results_atlas_tabs.setTabVisible(1, detailed)
        self.results_atlas_tabs.tabBar().setVisible(detailed)
        self.results_atlas.set_detailed(detailed)
        if detailed and getattr(self, "_results_gui_ready", False):
            self.refresh_results()

    def _results_atlas_tab_changed(self, index: int) -> None:
        if (index == 1 and self.results_mode_combo.currentData() == "detailed"
                and getattr(self, "_results_gui_ready", False)):
            self.refresh_results()

    def _atlas_run_changed(self, *_args) -> None:
        index = self.results_run_combo.findData(
            self.atlas_run_combo.currentData()
        )
        if index >= 0 and index != self.results_run_combo.currentIndex():
            self.results_run_combo.setCurrentIndex(index)

    def _sync_atlas_run_combo(self, *_args) -> None:
        index = self.atlas_run_combo.findData(
            self.results_run_combo.currentData()
        )
        if index >= 0 and index != self.atlas_run_combo.currentIndex():
            self.atlas_run_combo.blockSignals(True)
            self.atlas_run_combo.setCurrentIndex(index)
            self.atlas_run_combo.blockSignals(False)

    def _open_atlas_endpoint(self, endpoint_id: str) -> None:
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(0)
        self.results_atlas.detail_tabs.setCurrentIndex(0)
        if endpoint_id:
            index = self.endpoint_filter.findData(endpoint_id)
            if index >= 0:
                self.endpoint_filter.setCurrentIndex(index)
        self.advanced_results_box.setChecked(True)

    def _open_atlas_detail(self, index: int) -> None:
        if not 0 <= index < self.results_atlas.detail_tabs.count():
            return
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(0)
        self.results_atlas.detail_tabs.setCurrentIndex(index)
        self.advanced_results_box.setChecked(True)

    def _open_atlas_spatial(self, endpoint_id: str) -> None:
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(1)
        self.advanced_results_box.setChecked(True)
        if endpoint_id:
            index = self.endpoint_filter.findData(endpoint_id)
            if index >= 0:
                self.endpoint_filter.setCurrentIndex(index)
        validation_index = self.split_filter.findData("validation")
        if validation_index >= 0:
            self.split_filter.setCurrentIndex(validation_index)
        self.result_tabs.setCurrentWidget(self.group_vcg_result_tab)

    def _open_atlas_designs(self) -> None:
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(1)
        self.advanced_results_box.setChecked(True)
        self.result_tabs.setCurrentWidget(self.targeted_design_result_tab)

    def _open_atlas_scores(self) -> None:
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(1)
        self.advanced_results_box.setChecked(True)
        self.result_tabs.setCurrentWidget(self.research_results_view)

    def _open_atlas_leads(self) -> None:
        self.results_mode_combo.setCurrentIndex(1)
        self.results_atlas_tabs.setCurrentIndex(1)
        self.advanced_results_box.setChecked(True)
        self.result_tabs.setCurrentWidget(self.guideline_result_tab)
        self.guideline_tabs.setCurrentWidget(self.lead_comparison_view)

    def _open_atlas_review(self, run_id: str) -> None:
        """Navigate only to an explicitly matching, selectable run's review list."""
        index = self.report_run_combo.findData(run_id) if run_id else -1
        if index < 0:
            return
        self.report_run_combo.setCurrentIndex(index)
        self.workspace_tabs.setCurrentIndex(TAB_REPORTS)
        self.refresh_review()

    # ------------------------------------------------------------------
    # ECG laboratory
    # ------------------------------------------------------------------

    def _ecg_lab_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        actions = QHBoxLayout()
        self.import_case_button = self._button(
            "ECG importieren",
            "WFDB, CSV/TSV, EDF, DICOM-Waveform oder NPY/NPZ lokal importieren",
            self.import_ecg_case,
            QStyle.StandardPixmap.SP_DialogOpenButton,
        )
        actions.addWidget(self.import_case_button)
        self.practice_button = self._button(
            "EKG-Übungsmodus",
            "Verblindetes, lokal annotiertes 12-Kanal-EKG aus Challenge-Trainingsdaten; Lösung erst auf Klick",
            self.open_ecg_practice,
            QStyle.StandardPixmap.SP_DialogHelpButton,
        )
        actions.addWidget(self.practice_button)
        self.case_combo = QComboBox()
        self.case_combo.setMinimumWidth(260)
        self.case_combo.setToolTip("Lokale pseudonymisierte Fallbibliothek")
        self.case_combo.currentIndexChanged.connect(self._case_changed)
        actions.addWidget(self.case_combo, 1)
        self.analyze_case_button = self._button(
            "ECG analysieren",
            "Wellen automatisch erkennen, VCG und Marker berechnen sowie "
            "kompatible Kriterien mit Grenzwert, Grauzone und klassischen Regeln vergleichen",
            self.analyze_selected_case,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        actions.addWidget(self.analyze_case_button)
        self.edit_fiducials_button = self._button(
            "Fiduzialpunkte",
            "Automatische P/QRS/T-Grenzen lokal prüfen und als "
            "versionierte manuelle Revision speichern",
            self.edit_case_fiducials,
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )
        self.edit_fiducials_button.setEnabled(False)
        actions.addWidget(self.edit_fiducials_button)
        layout.addLayout(actions)
        self.expert_lab = ExpertECGLab()
        analysis_selector = QHBoxLayout()
        analysis_selector.addWidget(QLabel("Gespeicherter Laborstand:"))
        self.lab_analysis_combo = QComboBox()
        self.lab_analysis_combo.setToolTip("Explizite historische Analyseauswahl anhand ID und Provenienz; Dateizeitstempel entscheiden nicht über die Auswahl.")
        analysis_selector.addWidget(self.lab_analysis_combo, 1)
        layout.addLayout(analysis_selector)
        self.lab_analysis_combo.currentIndexChanged.connect(self._lab_analysis_selected)
        self._lab_analysis_choices = {}
        self.expert_lab.set_dark(self._dark_mode)
        self.lab_scroll = QScrollArea()
        self.lab_scroll.setObjectName("ecgLabScroll")
        self.lab_scroll.setWidgetResizable(True)
        self.lab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.lab_scroll.setWidget(self.expert_lab)
        layout.addWidget(self.lab_scroll, 1)
        self.lab_controller = LabProcessController(self)
        self.lab_controller.completed.connect(self._display_lab_analysis)
        self.lab_controller.failed.connect(lambda message: QMessageBox.warning(self, "Laboranalyse", message))
        self.lab_controller.status.connect(lambda text: self.case_status.setText(text))
        self.lab_controller.busyChanged.connect(self._lab_busy_changed)
        self.cancel_lab_button = QPushButton("Labor abbrechen")
        self.cancel_lab_button.setToolTip("Nur den lokalen Laborauftrag am nächsten sicheren Phasenübergang abbrechen; Produktionslauf unverändert.")
        self.cancel_lab_button.setEnabled(False)
        self.cancel_lab_button.clicked.connect(self.lab_controller.cancel)
        actions.addWidget(self.cancel_lab_button)
        self.expert_lab.applyFiducials.connect(self._apply_expert_fiducials)
        self.expert_lab.timeSelected.connect(self._set_case_cursor_time)
        self.expert_lab.resultsRequested.connect(self._show_results_from_lab)
        self.case_status = QLabel(
            "Lokale Forschungsanalyse, keine klinische Diagnose. Nur Kriterien "
            "eines vollständigen kompatiblen Validation-Laufs werden verwendet."
        )
        self.case_status.setWordWrap(True)
        self.case_status.setObjectName("mimicTextDangerMuted")
        layout.addWidget(self.case_status)
        self.case_delineation_status = QLabel(
            "Automatische Wellen- und Intervallmessung noch nicht ausgeführt."
        )
        self.case_delineation_status.setWordWrap(True)
        self.case_delineation_status.setToolTip(
            "Technischer Rhythmuskontext, Fiduzialkonfidenz und Gründe für "
            "eine erforderliche manuelle Kontrolle. Dies ist keine Diagnose."
        )
        layout.addWidget(self.case_delineation_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        signal_panel = QWidget()
        signal_panel.setObjectName("ecgLabResearchSignalPanel")
        signal_layout = QVBoxLayout(signal_panel)
        signal_layout.setContentsMargins(9, 8, 9, 8)
        signal_title = QLabel("12-KANAL-EKG  /  ZEIT")
        signal_title.setObjectName("ecgLabSectionLabel")
        signal_layout.addWidget(signal_title)
        signal_controls = QHBoxLayout()
        self.show_case_wave_regions = QCheckBox("Wellenbereiche")
        self.show_case_wave_regions.setChecked(True)
        self.show_case_wave_regions.setToolTip(
            "P-Welle, PR-Segment, QRS-Komplex, ST-Segment und T-Welle "
            "farblich hinter dem 12-Kanal-ECG hervorheben"
        )
        self.show_case_wave_regions.stateChanged.connect(
            self._redraw_current_case_arrays
        )
        signal_controls.addWidget(self.show_case_wave_regions)
        legend = QLabel(
            "<span style='color:#087e8b'>P</span>  "
            "<span style='color:#557a95'>PR</span>  "
            "<span style='color:#d1495b'>QRS</span>  "
            "<span style='color:#2a9d8f'>ST</span>  "
            "<span style='color:#b77900'>T</span>  "
            "<span style='color:#6d4c9b'>U</span>"
        )
        legend.setToolTip(
            "Nur Bereiche mit ausreichender Fiduzialkonfidenz werden "
            "eingefärbt. Gestrichelte Linien zeigen einzelne Landmarken."
        )
        signal_controls.addWidget(legend)
        signal_controls.addStretch()
        signal_layout.addLayout(signal_controls)
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setBackground("w")
        self.ecg_plot.setLabel("bottom", "Zeit", units="s")
        self.ecg_plot.setLabel("left", "12 Kanäle (versetzt)", units="mV")
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_plot.scene().sigMouseMoved.connect(self._case_mouse_moved)
        signal_layout.addWidget(self.ecg_plot, 1)
        splitter.addWidget(signal_panel)

        self.vcg_container = QWidget()
        self.vcg_container.setObjectName("ecgLabResearchSpatialPanel")
        vcg_layout = QVBoxLayout(self.vcg_container)
        vcg_layout.setContentsMargins(9, 8, 9, 8)
        vcg_title = QLabel("VEKTORKARDIOGRAMM  /  RAUM")
        vcg_title.setObjectName("ecgLabSectionLabel")
        vcg_layout.addWidget(vcg_title)
        vcg_controls = QHBoxLayout()
        self.show_vcg_cloud = QCheckBox("Ganzer Record (Wolke)")
        self.show_vcg_cloud.setToolTip(
            "Alle Beat-Schleifen der VCG-Trajektorie statt nur der einen "
            "repräsentativen Schleife -- zeigt Beat-zu-Beat-Variabilität "
            "direkt statt nur als abstrakte Kennzahl."
        )
        self.show_vcg_cloud.stateChanged.connect(
            self._redraw_current_case_arrays
        )
        vcg_controls.addWidget(self.show_vcg_cloud)
        vcg_controls.addWidget(QLabel("Segment:"))
        self.case_vcg_segment_combo = QComboBox()
        self.case_vcg_segment_combo.addItem("Gesamter Loop", "")
        self.case_vcg_segment_combo.addItem("P-Loop", "p_wave")
        self.case_vcg_segment_combo.addItem("QRS-Loop", "qrs_complex")
        self.case_vcg_segment_combo.addItem("T-Loop", "t_wave")
        self.case_vcg_segment_combo.currentIndexChanged.connect(
            self._redraw_current_case_arrays
        )
        vcg_controls.addWidget(self.case_vcg_segment_combo)
        self.case_direct_angle_label = QLabel("Direct-8-Lead: noch nicht berechnet")
        self.case_direct_angle_label.setObjectName("mimicTextMuted")
        self.case_direct_angle_label.setMaximumWidth(300)
        self.case_direct_angle_label.setWordWrap(True)
        vcg_controls.addWidget(self.case_direct_angle_label)
        vcg_controls.addStretch()
        vcg_layout.addLayout(vcg_controls)
        self.vcg_view = self._create_vcg_view()
        if self._vcg_gl is None:
            self.show_vcg_cloud.setEnabled(False)
            self.show_vcg_cloud.setToolTip(
                "3D-VCG ist ohne funktionierendes OpenGL nicht verfügbar."
            )
        vcg_layout.addWidget(self.vcg_view, 1)
        splitter.addWidget(self.vcg_container)
        splitter.setSizes([800, 500])
        # The outer workspace scroll area must not stretch the signal plots
        # to the combined minimum height of all result tabs below them.
        # QSplitter recalculates its own maximum size from its children.
        # Put the bound on an ordinary wrapper so hidden tabs with larger
        # size hints cannot stretch this signal panel beyond the viewport.
        signal_viewport = QWidget()
        signal_viewport.setMinimumHeight(360)
        signal_viewport.setMaximumHeight(480)
        signal_viewport_layout = QVBoxLayout(signal_viewport)
        signal_viewport_layout.setContentsMargins(0, 0, 0, 0)
        signal_viewport_layout.addWidget(splitter)
        layout.addWidget(signal_viewport, 2)

        self.case_pattern_summary = QLabel(
            "Noch keine Forschungsanalyse verfügbar."
        )
        self.case_pattern_summary.setWordWrap(True)
        self.case_pattern_summary.setStyleSheet(
            theme.status_box_qss("neutral", self._dark_mode)
        )
        self.case_pattern_summary.setToolTip(
            "Eine wahrscheinliche Forschungsdiagnose benötigt mindestens "
            "Validation-Evidenz, bestandene Anwendbarkeit und einen Wert "
            "außerhalb der Bootstrap-Grauzone. Es wird keine Wahrscheinlichkeit ausgegeben."
        )
        layout.addWidget(self.case_pattern_summary)
        result_tabs = QTabWidget()
        self.case_findings = self._table(
            (
                "Pathologie / klinisches Ziel",
                "Bewertung",
                "Evidenzstand",
                "Biomarker / Kriterium",
                "Wert",
                "Grenzwert",
                "Grauzone",
                "Abstand",
                "Validierung: Sensitivität",
                "Validierung: erreichte Spezifität",
                "Klassischer Vergleich",
                "Einordnung",
            )
        )
        self.case_findings.setToolTip(
            "Train-fixierte Kriterien aus genau einem kompatiblen Run. "
            "Nicht beurteilbar ist weder positiv noch negativ."
        )
        result_tabs.addTab(self.case_findings, "Forschungsbefunde")
        from .research_results_widget import CaseResearchStatusWidget
        self.case_research_status = CaseResearchStatusWidget()
        result_tabs.addTab(self.case_research_status, "Neue Scores & Fokus: Status")
        self.case_guideline_comparison = self._table(
            (
                "Endpunkt",
                "Klassisches Kriterium",
                "Klassisch: Verdikt",
                "Klassisch: Wert",
                "Neuer Marker/Score",
                "Neu: Wert",
                "Neu: Abstand zur 95-%-Spez.-Schwelle",
                "Neu: Status",
                "DB-weit: Validation-AUC (95 % CI)",
                "DB-weit: Sens. am Train-Grenzwert (95-%-Spez.-Ziel)",
                "Evidenz",
                "DB-weit: Verdikt (Leitmarker)",
            )
        )
        self.case_guideline_comparison.setToolTip(
            "Klassisches Leitlinienkriterium und neuer Forschungsmarker/-score "
            "für denselben klinischen Endpunkt nebeneinander: Verdikt/Wert für "
            "genau dieses ECG, plus die datenbankweite Validation-Zuverlässigkeit "
            "des neuen Markers (Train-fixierte Schwelle, keine Diagnose). "
            "'DB-weit: Verdikt (Leitmarker)' bezieht sich auf den für diesen "
            "Endpunkt vorab festgelegten Leitmarker aus dem gepaarten "
            "Signifikanztest gegen genau dieses klassische Kriterium — das "
            "kann ein anderer Marker sein als der in dieser Zeile gezeigte "
            "Score."
        )
        result_tabs.addTab(
            self.case_guideline_comparison,
            "Leitlinienvergleich",
        )
        self.case_scores = self._table(
            (
                "Endpunkt",
                "Forschungsmuster",
                "Evidenz",
                "Score",
                "95-%-Spez.-Schwelle",
                "Abstand",
                "Validation-AUC (95 % CI)",
                "Sens. am Train-Grenzwert (95-%-Spez.-Ziel)",
                "Klassisch positiv",
                "Release",
            )
        )
        result_tabs.addTab(self.case_scores, "Entwicklungsscores (Details)")
        self.case_single_markers = self._table(
            (
                "Endpunkt",
                "Einzelmarker",
                "Typ",
                "Rohwert",
                "Richtung",
                "95-%-Spez.-Schwelle",
                "Abstand",
                "Forschungsmuster",
                "Validation-AUC",
                "Sens. am Train-Grenzwert (95-%-Spez.-Ziel)",
                "Evidenz",
                "Familie",
            )
        )
        self.case_single_markers.setToolTip(
            "Je Endpunkt höchstens drei kanonische Core-/Sentinel-Marker aus "
            "unterschiedlichen Familien. Schwellen wurden nur in Train "
            "gelernt; die Anzeige ist keine Diagnose."
        )
        result_tabs.addTab(
            self.case_single_markers,
            "Beste Einzelmarker",
        )
        self.case_classical = self._table(
            (
                "Kriterium",
                "Relevante Endpunkte",
                "Auswertbar",
                "Positiv",
                "Wert",
                "Leitlinien-Sollwert",
                "Ausschluss",
            )
        )
        self.case_classical.setToolTip(
            "Plausibilitätsprüfung: der berechnete Wert neben dem "
            "literaturbasierten Leitlinien-Sollwert derselben Regel."
        )
        result_tabs.addTab(self.case_classical, "Klassische Regeln")
        self.case_features = self._table(
            (
                "Endpunkt",
                "Feature",
                "Rohwert",
                "Richtung",
                "Train-Mittel",
                "Train-SD",
                "z-Beitrag",
                "Einordnung",
            )
        )
        result_tabs.addTab(self.case_features, "Markerkomponenten")
        self.case_delineation = self._table(
            (
                "Welle/Intervall",
                "Beginn",
                "Ende",
                "Dauer",
                "Konfidenz",
                "Status",
            )
        )
        self.case_delineation.setToolTip(
            "Automatisch gemessene Bereiche und Zeiten des robusten "
            "repräsentativen Beats; nicht auswertbare Intervalle bleiben leer"
        )
        result_tabs.addTab(self.case_delineation, "Wellen & Zeiten")

        rr_tab = QWidget()
        rr_layout = QVBoxLayout(rr_tab)
        self.rr_tachogram_note = QLabel("Noch keine Forschungsanalyse verfügbar.")
        self.rr_tachogram_note.setObjectName("mimicTextMuted")
        self.rr_tachogram_note.setWordWrap(True)
        rr_layout.addWidget(self.rr_tachogram_note)
        rr_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.rr_tachogram_plot = pg.PlotWidget()
        self.rr_tachogram_plot.setBackground("w")
        self.rr_tachogram_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rr_tachogram_plot.setTitle("RR-Tachogramm")
        self.rr_tachogram_plot.setLabel("bottom", "Zeit", units="s")
        self.rr_tachogram_plot.setLabel("left", "RR-Intervall", units="ms")
        rr_splitter.addWidget(self.rr_tachogram_plot)
        self.poincare_plot = pg.PlotWidget()
        self.poincare_plot.setBackground("w")
        self.poincare_plot.showGrid(x=True, y=True, alpha=0.2)
        self.poincare_plot.setTitle("Poincaré-Plot (RRₙ vs. RRₙ₊₁)")
        self.poincare_plot.setLabel("bottom", "RRₙ", units="ms")
        self.poincare_plot.setLabel("left", "RRₙ₊₁", units="ms")
        self.poincare_plot.setAspectLocked(True)
        rr_splitter.addWidget(self.poincare_plot)
        rr_splitter.setSizes([650, 450])
        rr_layout.addWidget(rr_splitter, 1)
        result_tabs.addTab(rr_tab, "RR-Tachogramm & Poincaré")

        overlay_tab = QWidget()
        overlay_layout = QVBoxLayout(overlay_tab)
        overlay_controls = QHBoxLayout()
        overlay_controls.addWidget(QLabel("Ableitung"))
        self.beat_overlay_lead_combo = QComboBox()
        for index, name in enumerate(
            (
                "I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6",
            )
        ):
            self.beat_overlay_lead_combo.addItem(name, index)
        self.beat_overlay_lead_combo.setCurrentIndex(1)
        self.beat_overlay_lead_combo.currentIndexChanged.connect(
            self._draw_beat_overlay
        )
        overlay_controls.addWidget(self.beat_overlay_lead_combo)
        overlay_controls.addStretch()
        overlay_layout.addLayout(overlay_controls)
        self.beat_overlay_note = QLabel("Noch keine Forschungsanalyse verfügbar.")
        self.beat_overlay_note.setObjectName("mimicTextMuted")
        self.beat_overlay_note.setWordWrap(True)
        overlay_layout.addWidget(self.beat_overlay_note)
        self.beat_overlay_plot = pg.PlotWidget()
        self.beat_overlay_plot.setBackground("w")
        self.beat_overlay_plot.showGrid(x=True, y=True, alpha=0.2)
        self.beat_overlay_plot.setTitle("Beat-Überlagerung (R-ausgerichtet)")
        self.beat_overlay_plot.setLabel(
            "bottom", "Zeit relativ zu R-Zacke", units="ms"
        )
        self.beat_overlay_plot.setLabel("left", "Amplitude", units="mV")
        overlay_layout.addWidget(self.beat_overlay_plot, 1)
        result_tabs.addTab(overlay_tab, "Beat-Überlagerung")

        qtrr_tab = QWidget()
        qtrr_layout = QVBoxLayout(qtrr_tab)
        self.qt_rr_note = QLabel("Noch keine Forschungsanalyse verfügbar.")
        self.qt_rr_note.setObjectName("mimicTextMuted")
        self.qt_rr_note.setWordWrap(True)
        qtrr_layout.addWidget(self.qt_rr_note)
        self.qt_rr_plot = pg.PlotWidget()
        self.qt_rr_plot.setBackground("w")
        self.qt_rr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.qt_rr_plot.setTitle("QT/RR-Regression (Bazett/Fridericia)")
        self.qt_rr_plot.setLabel("bottom", "RR-Intervall", units="ms")
        self.qt_rr_plot.setLabel("left", "QT-Intervall", units="ms")
        self.qt_rr_plot.addLegend(offset=(8, 8))
        qtrr_layout.addWidget(self.qt_rr_plot, 1)
        result_tabs.addTab(qtrr_tab, "QT/RR-Regression")

        self.expert_lab.extra_research_tabs.extend([
            self.expert_lab.tabs.addTab(signal_viewport, "EKG / 3D-VCG"),
            self.expert_lab.tabs.addTab(result_tabs, "Forschungsbefunde & Dynamik"),
        ])
        self.lab_research_tabs = result_tabs
        self.lab_research_signal = signal_viewport
        self.expert_lab.researchModeChanged.connect(self._lab_mode_changed)
        self.expert_lab._mode_changed()
        self._lab_mode_changed(False)
        _disable_auto_si_prefix(
            self.rr_tachogram_plot,
            self.poincare_plot,
            self.beat_overlay_plot,
            self.qt_rr_plot,
        )
        return page

    # ------------------------------------------------------------------
    # Reports and review
    # ------------------------------------------------------------------

    def _reports_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        report_group = QGroupBox("Bericht erstellen")
        form = QFormLayout(report_group)
        self.report_run_combo = QComboBox()
        self.report_run_combo.currentIndexChanged.connect(self.refresh_review)
        self.report_run_combo.setToolTip(
            "Vollständiger, kompatibler Ergebnisstand, aus dem der Bericht gebaut wird."
        )
        form.addRow("Run", self.report_run_combo)
        self.report_type_combo = QComboBox()
        for label, value in (
            ("Studienbericht", "study"),
            ("Endpunktbericht", "endpoint"),
            ("Markerbericht", "marker"),
            ("Job- und QC-Bericht", "job_qc"),
        ):
            self.report_type_combo.addItem(label, value)
        self.report_type_combo.setToolTip(
            "Studie fasst den Lauf zusammen; Endpunkt und Marker filtern auf eine konkrete Frage."
        )
        self.report_type_combo.currentIndexChanged.connect(
            self._update_report_scope_controls
        )
        form.addRow("Typ", self.report_type_combo)

        self.report_endpoint_label = QLabel("Klinisches Ziel")
        self.report_endpoint_combo = QComboBox()
        self.report_endpoint_combo.setObjectName("reportEndpointCombo")
        self.report_endpoint_combo.setEditable(True)
        self.report_endpoint_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.report_endpoint_combo.lineEdit().setPlaceholderText(
            "Pathologie oder klinisches Ziel suchen"
        )
        self.report_endpoint_combo.addItem("Klinisches Ziel auswählen", "")
        for term in clinical_display_terms():
            self.report_endpoint_combo.addItem(
                f"{term.category_de} · {term.display_name_de}",
                term.term_id,
            )
            self.report_endpoint_combo.setItemData(
                self.report_endpoint_combo.count() - 1,
                f"{term.short_definition_de}\n{term.source_summary_de}",
                Qt.ItemDataRole.ToolTipRole,
            )
        endpoint_completer = self.report_endpoint_combo.completer()
        endpoint_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        endpoint_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.report_endpoint_combo.setToolTip(
            "Der Endpunktbericht enthält nur dieses klinische Ziel und behält interne IDs im Herkunftsnachweis."
        )
        form.addRow(self.report_endpoint_label, self.report_endpoint_combo)

        self.report_marker_label = QLabel("Biomarker")
        self.report_marker_combo = QComboBox()
        self.report_marker_combo.setObjectName("reportMarkerCombo")
        self.report_marker_combo.setEditable(True)
        self.report_marker_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.report_marker_combo.lineEdit().setPlaceholderText(
            "Biomarker oder Formelbestandteil suchen"
        )
        self.report_marker_combo.addItem("Biomarker auswählen", "")
        report_feature_ids = tuple(
            dict.fromkeys(
                feature_id
                for panel in default_feature_panels()
                if panel.panel_id in {"primary_core", "sentinel"}
                for feature_id in panel.feature_ids
            )
        )
        for feature_id in report_feature_ids:
            self.report_marker_combo.addItem(
                biomarker_display_name(feature_id),
                feature_id,
            )
            self.report_marker_combo.setItemData(
                self.report_marker_combo.count() - 1,
                biomarker_tooltip(feature_id),
                Qt.ItemDataRole.ToolTipRole,
            )
        marker_completer = self.report_marker_combo.completer()
        marker_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        marker_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.report_marker_combo.setToolTip(
            "Der Markerbericht zeigt nur diesen Marker, seine Endpunkte, Grenzwerte und Vergleiche."
        )
        form.addRow(self.report_marker_label, self.report_marker_combo)

        self.report_scope_hint = QLabel()
        self.report_scope_hint.setWordWrap(True)
        form.addRow("", self.report_scope_hint)
        self.report_language = QComboBox()
        self.report_language.addItem("Deutsch", "de")
        self.report_language.addItem("English", "en")
        form.addRow("Sprache", self.report_language)
        formats = QWidget()
        formats_layout = QHBoxLayout(formats)
        formats_layout.setContentsMargins(0, 0, 0, 0)
        self.format_checks = {}
        for name in ("PDF", "HTML", "XLSX", "Parquet", "CSV"):
            check = QCheckBox(name)
            check.setChecked(name != "CSV")
            check.setToolTip(f"{name}-Ausgabe für den lokalen Bericht")
            formats_layout.addWidget(check)
            self.format_checks[name.lower()] = check
        form.addRow("Formate", formats)
        self.include_case_appendix = QCheckBox("Review-Anhang")
        self.include_case_appendix.setToolTip(
            "Diskordanzfälle als lokalen Anhang aufnehmen"
        )
        form.addRow("", self.include_case_appendix)
        self.include_case_ids = QCheckBox("Lokale IDs einschließen")
        self.include_case_ids.setToolTip(
            "Explizites lokales Opt-in. Standardberichte sind aggregiert "
            "und anonymisiert."
        )
        form.addRow("", self.include_case_ids)
        self._update_report_scope_controls()
        controls_layout.addWidget(report_group)

        report_actions = QHBoxLayout()
        report_actions.addWidget(
            self._button(
                "Bericht einreihen",
                "Konfigurierten A4-Bericht als persistenten Hintergrundjob starten",
                self.enqueue_report,
                QStyle.StandardPixmap.SP_DialogSaveButton,
            )
        )
        report_actions.addWidget(
            self._button(
                "",
                "Lokalen Report-Ordner öffnen",
                self.open_report_folder,
                QStyle.StandardPixmap.SP_DirOpenIcon,
            )
        )
        controls_layout.addLayout(report_actions)
        self.report_files = self._table(("Bericht", "Format", "Geändert"))
        controls_layout.addWidget(self.report_files, 1)
        splitter.addWidget(controls)

        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_header = QHBoxLayout()
        review_header.addWidget(QLabel("Diskordanz-Review"))
        review_header.addStretch(1)
        review_header.addWidget(
            self._button(
                "Ausgewähltes EKG im Labor öffnen",
                "Prüft Patienten- und Studienkennung gegen den Development-Master; "
                "Locked-Test- und mehrdeutige Fälle bleiben gesperrt.",
                self._open_review_ecg,
                QStyle.StandardPixmap.SP_FileDialogDetailedView,
            )
        )
        review_header.addWidget(
            self._button(
                "",
                "Review-Kandidaten des gewählten Runs neu laden",
                self.refresh_review,
                QStyle.StandardPixmap.SP_BrowserReload,
            )
        )
        review_layout.addLayout(review_header)
        self.review_table = PagedParquetTable(page_size=150)
        self.review_table.selection_changed.connect(self._show_review_details)
        review_layout.addWidget(self.review_table, 2)
        self.review_details = QTextEdit()
        self.review_details.setReadOnly(True)
        self.review_details.setMaximumHeight(380)
        self.review_details.setToolTip(
            "Klinische Evidenz, Zeitkontext, Vergleichsregeln, "
            "Scorekomponenten und Qualitätsstatus"
        )
        review_layout.addWidget(self.review_details, 1)
        splitter.addWidget(review)
        splitter.setSizes([380, 900])
        layout.addWidget(splitter, 1)
        return page

    # ------------------------------------------------------------------
    # Project and refresh
    # ------------------------------------------------------------------

    def _load_config_into_fields(self) -> None:
        self.database_root.setText(str(self.config.mimic_iv_root.parent))
        self.work_root.setText(str(self.config.work_root))
        self.mimic_note_root_field.setText(
            str(self.config.mimic_note_root)
            if self.config.mimic_note_root is not None
            else ""
        )
        self.code15_root_field.setText(
            str(self.config.code15_root)
            if self.config.code15_root is not None
            else ""
        )
        self.sami_trop_root_field.setText(
            str(self.config.sami_trop_root)
            if self.config.sami_trop_root is not None
            else ""
        )
        self.workers_spin.setValue(self.config.workers)
        self.memory_spin.setValue(self.config.duckdb_memory_limit_gb)
        self.micro_batch_size.setValue(self.config.feature_micro_batch_size)

    # The five settings this dialog owns.  Everything else in project.json
    # belongs to whoever wrote it and must survive a "Projekt pruefen".
    _OWNED_SETTINGS = (
        "mimic_note_root",
        "code15_root",
        "sami_trop_root",
        "workers",
        "duckdb_memory_limit_gb",
        "feature_micro_batch_size",
    )

    def _save_project_fields(self) -> None:
        database_root = Path(self.database_root.text().strip())
        work_root = Path(self.work_root.text().strip())
        roots_changed = (
            database_root.resolve() != self.config.mimic_iv_root.parent.resolve()
            or work_root.resolve() != self.config.work_root.resolve()
        )
        if roots_changed:
            discovered = MimicProjectConfig.discover(database_root, work_root)
            self.config = discovered
        else:
            # ``config.save()`` writes the whole file from this object, which
            # holds whatever was on disk when the dialog opened.  Anything
            # edited outside meanwhile is then silently reverted -- measured:
            # workers 5 and an OS reserve of 7.0 GB, set while a run was
            # paused, were back at 7 and 6.0 after one "Projekt pruefen", and
            # os_memory_reserve_gb does not even have a field here.  Re-read
            # first and change only what this dialog owns.
            self._reload_config_preserving_external_edits()
        note_root_text = self.mimic_note_root_field.text().strip()
        self.config.mimic_note_root = Path(note_root_text) if note_root_text else None
        code15_text = self.code15_root_field.text().strip()
        sami_trop_text = self.sami_trop_root_field.text().strip()
        self.config.code15_root = Path(code15_text) if code15_text else None
        self.config.sami_trop_root = Path(sami_trop_text) if sami_trop_text else None
        self.config.workers = self.workers_spin.value()
        self.config.duckdb_memory_limit_gb = self.memory_spin.value()
        self.config.feature_micro_batch_size = self.micro_batch_size.value()
        self.config.save()
        self.job_store = JobStore.for_project(self.config)
        self.case_library = ECGCaseLibrary(self.config)
        self.result_repository = ResultRepository(self.config)

    def _reload_config_preserving_external_edits(self) -> None:
        """Take the file on disk as the base for this save.

        Failing to re-read is not worth refusing the save for: the previous
        behaviour (write what we had) is what happens either way, and the user
        asked for a check, not for an error dialog about a config file.
        """

        try:
            self.config = MimicProjectConfig.load(self.config.project_file)
        except Exception as exc:
            LOGGER.warning("Project configuration reload failed: %s", exc)
            return

    def check_project(self) -> None:
        try:
            self._save_project_fields()
            errors = self.config.validate_paths()
            free_gb = shutil.disk_usage(self.config.work_root).free / 1024**3
            if errors:
                self.project_status.setText(
                    f"{len(errors)} blockierende Probleme"
                )
                self.readiness_text.setText("\n".join(errors))
                QMessageBox.warning(
                    self,
                    "Projektprüfung",
                    "\n".join(errors),
                )
            else:
                self.project_status.setText(
                    f"Projekt bereit | {free_gb:.1f} GB frei"
                )
                self.readiness_text.setText(
                    "Pfade und Schreibzugriff sind bereit. Die inhaltliche "
                    "Schema- und Releaseprüfung erfolgt reproduzierbar im "
                    "Dateninventar-Job."
                )
                self.status_changed.emit(self.project_status.text())
            # _save_project_fields() just rebuilt self.result_repository
            # (fresh, empty available_runs() cache), so this refresh pays
            # the full per-run-directory cost once -- keep processing
            # events throughout so Windows doesn't flag the app as
            # unresponsive if the user clicks around while it runs.
            self.refresh_all(
                progress=lambda _message, _fraction: QApplication.processEvents()
            )
        except Exception as exc:
            QMessageBox.critical(self, "Projektprüfung", str(exc))

    def _refresh_backend_sync_status(self) -> None:
        backend_current, backend_message = _gui_backend_contract_status()
        self.backend_sync_status.setText(backend_message)
        self.backend_sync_status.setStyleSheet(
            theme.status_qss("success", self._dark_mode)
            if backend_current
            else theme.status_qss("danger", self._dark_mode)
        )

    def refresh_all(
        self,
        *,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._start_busy_pulse()
        try:
            errors = self.config.validate_paths()
            try:
                free_gb = shutil.disk_usage(self.config.work_root).free / 1024**3
            except OSError:
                free_gb = 0.0
            if errors:
                self.project_status.setText(
                    f"Projekt blockiert | {len(errors)} Probleme"
                )
                self.readiness_text.setText("\n".join(errors))
            else:
                self.project_status.setText(
                    f"Projekt bereit | {free_gb:.1f} GB frei"
                )
                self.readiness_text.setText(
                    "MIMIC-IV ist der primäre klinische Arbeitsmodus. "
                    "PTB-XL bleibt ausschließlich als getrennte Evidenzherkunft."
                )
            self._refresh_backend_sync_status()
            self._refresh_overview()
            if progress is not None:
                progress("Aufträge werden geladen...", 0.65)
            self.refresh_jobs()
            if progress is not None:
                progress("Kohorten werden geladen...", 0.70)
            self.refresh_cohorts()
            if progress is not None:
                progress("Laufauswahl wird ermittelt...", 0.74)
            self._refresh_run_choices()
            if progress is not None:
                progress("Ergebnisse werden geladen...", 0.79)
            self.refresh_results()
            if progress is not None:
                progress("Fälle werden geladen...", 0.93)
            self.refresh_cases()
            if progress is not None:
                progress("Prüfliste wird geladen...", 0.97)
            self.refresh_review()
            self._last_ui_artifact_signature = self._ui_artifact_signature()
        finally:
            self._stop_busy_pulse()

    def _start_busy_pulse(self) -> None:
        self.busy_indicator.show()
        from ..motion import current_motion_policy
        if not current_motion_policy().reduced:
            self._busy_animation.start()

    def _on_motion_preference_changed(self, reduced: bool) -> None:
        if reduced:
            self._busy_animation.stop()
            self._busy_opacity_effect.setOpacity(1.0)
            if hasattr(self, "_results_view_anim"):
                self._results_view_anim.stop()
                self._results_view_opacity.setOpacity(1.0)
        elif self.busy_indicator.isVisible():
            self._busy_animation.start()

    def _stop_busy_pulse(self) -> None:
        self._busy_animation.stop()
        self.busy_indicator.hide()

    def _refresh_overview(self) -> None:
        selected_run = self.config.runs_root / self._current_run_id()
        pilot_1k_current = self._pilot_validation_current(1_000)
        pilot_10k_current = self._pilot_validation_current(10_000)
        current_validation_ready = _formal_development_validation_complete(
            selected_run,
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        )
        stages = [
            (
                "CODE-15%",
                (
                    self.config.code15_root / "exams.csv"
                    if self.config.code15_root is not None
                    else self.config.work_root / "__code15_not_configured__"
                ),
                "Primärversorgungs-Referenz, ECG-Befundlabels und Mortalität",
            ),
            (
                "SaMi-Trop",
                (
                    self.config.sami_trop_root / "exams.csv"
                    if self.config.sami_trop_root is not None
                    else self.config.work_root / "__sami_trop_not_configured__"
                ),
                "Seropositive Chagas-Kohorte mit prospektivem Follow-up",
            ),
            (
                "GWAS-Kontext",
                self.config.work_root / "genetics" / "gwas_feasibility.json",
                (
                    "Genetische Architektur des räumlichen QRS-T-Winkels; "
                    "kein MR-, PRS- oder Kausalitätsanspruch"
                ),
            ),
            (
                "Dateninventar",
                self.config.work_root / "data_inventory.parquet",
                "Pfad-, Release- und Schema-Audit",
            ),
            (
                "Klinische Ereignisse",
                (
                    self.config.standardized_root
                    / "clinical_events"
                    / "clinical_events.manifest.json"
                ),
                "Labor, EMAR, Verordnungen, POE und ED-Medikation",
            ),
            (
                "ECG-Masterindex",
                self.config.master_root / "ecg_master_index.parquet",
                "Eine Zeile je ECG; klinische Evidenz normalisiert",
            ),
            (
                "Klinische Labels",
                self.config.labels_root / "endpoint_index_cohorts.parquet",
                "Vierstufige Labels und assessable controls",
            ),
            (
                "Event-Evidenz",
                (
                    self.config.work_root
                    / "provenance"
                    / "clinical_event_evidence_equivalence.json"
                ),
                (
                    "Direkt event-basiert oder exakt gegen Legacy-Evidenz "
                    "attestiert"
                ),
            ),
            (
                "Labelattestation",
                (
                    self.config.labels_root
                    / "development_label_attestation.json"
                ),
                "Semantischer Regelhash; Train/Validation exakt reproduziert",
            ),
            (
                "Biomarker-Audit",
                (
                    self.config.work_root
                    / "audits"
                    / "biomarkers"
                    / "biomarker_audit_summary.json"
                ),
                "Alle Formeln, Rollen, Panels und synthetischen Stressfälle",
            ),
            (
                "Wellenerkennung",
                (
                    self.config.resources_root
                    / "delineation"
                    / "benchmark_suite.json"
                ),
                "Externe Grenzen-, Rhythmus- und Rauschprüfung; Zusatzarme bleiben sichtbar",
            ),
            (
                "Features",
                self.config.features_root,
                "Core/Sentinel-Entwicklungsartefakte",
            ),
            (
                "Train / Validation",
                self.config.runs_root,
                "Locked Test bleibt versiegelt",
            ),
            (
                "Pathologie-Scan",
                selected_run / "phenome" / "phenome.manifest.json",
                "Train-Discovery und Validation-Replikation ohne Locked Test",
            ),
            (
                "Reports",
                selected_run / "reports" / "report.manifest.json",
                "Kanonische Tabellen und lokale druckbare Ausgaben",
            ),
            (
                "Abnahme-Audit",
                (
                    selected_run / "acceptance_audit.json"
                    if (selected_run / "freeze_manifest.json").is_file()
                    else selected_run / "acceptance_pre_freeze_audit.json"
                ),
                "Maschinenlesbare wissenschaftliche Abnahme",
            ),
            (
                "Freeze",
                selected_run / "freeze_manifest.json",
                "Locked Test versiegelt und noch ungelesen",
            ),
        ]
        rows = []
        for name, path, note in stages:
            if name == "Biomarker-Audit":
                ready = _biomarker_audit_current(
                    path,
                    expected_implementation_hash=(
                        self._current_biomarker_audit_hash
                    ),
                )
                if ready:
                    status = "vollständig und codeaktuell"
                elif path.is_file():
                    status = "veraltet: Audit erneut ausführen"
                else:
                    status = "fehlt"
            elif name == "GWAS-Kontext":
                presentation, ready = _gwas_context_presentation(
                    _read_json(path)
                )
                status = (
                    "vollständig; MR/PRS bewusst nicht schätzbar"
                    if ready
                    else presentation
                )
            elif name == "Wellenerkennung":
                ready = _delineation_benchmark_current(
                    path,
                    expected_implementation_hash=(
                        self._current_delineation_hash
                    ),
                )
                payload = _read_json(path)
                if ready:
                    completed = len(payload.get("datasets_completed", ()))
                    missing = len(payload.get("datasets_missing", ()))
                    status = (
                        f"Kernprüfung aktuell: {completed} Datensätze; "
                        f"{missing} Zusatzarme offen"
                    )
                elif path.is_file():
                    status = "veraltet oder Kernprüfung unvollständig"
                else:
                    status = "fehlt"
            elif (
                name
                in {
                    "Pathologie-Scan",
                    "Reports",
                    "Abnahme-Audit",
                    "Freeze",
                }
                and path.is_file()
                and not current_validation_ready
            ):
                ready = False
                status = "veraltet: neue Entwicklungsauswertung erforderlich"
            elif name == "Abnahme-Audit" and path.is_file():
                audit = _read_json(path)
                audit_status = str(audit.get("status", "unbekannt"))
                ready = audit_status in (
                    "ready_to_freeze",
                    "accepted_frozen_development",
                )
                status = audit_status.replace("_", " ")
            elif name == "Freeze" and path.is_file():
                freeze = _read_json(path)
                ready = freeze.get("locked_test_status") == "sealed_unread"
                status = (
                    "versiegelt, ungelesen"
                    if ready
                    else "Lock prüfen"
                )
            elif path.is_file():
                ready = True
                status = "vorhanden"
            elif name == "Features":
                manifests = list(path.glob("*/development/features.manifest.json"))
                selected_manifest = self._selected_feature_manifest_path()
                ready = _feature_manifest_complete(
                    selected_manifest,
                    expected_code_hash=self._current_feature_code_hash,
                )
                if ready:
                    status = "vollständig und codeaktuell"
                elif pilot_1k_current and pilot_10k_current:
                    status = (
                        "1.000er Methoden- und 10.000er Skalierungspilot "
                        "aktuell; Vollanalyse offen"
                    )
                elif pilot_1k_current:
                    status = "1.000er Methodenpilot aktuell; Vollanalyse offen"
                elif _manifest_complete(selected_manifest):
                    status = "veraltet: mit aktuellem Code neu berechnen"
                else:
                    status = "partiell" if manifests else "fehlt"
            elif name == "Event-Evidenz":
                ready = (
                    path.is_file()
                    or (
                        self.config.evidence_root
                        / "evidence_provenance.json"
                    ).is_file()
                )
                status = "verifiziert" if ready else "fehlt"
            elif name == "Train / Validation":
                runs = [
                    item.parent
                    for item in path.glob("*/screening_results.parquet")
                ]
                formal_runs = [
                    run
                    for run in runs
                    if _formal_development_validation_complete(
                        run,
                        expected_feature_code_hash=(
                            self._current_feature_code_hash
                        ),
                        expected_validation_code_hash=(
                            self._current_validation_code_hash
                        ),
                    )
                ]
                historical_formal_runs = [
                    run
                    for run in runs
                    if _formal_development_validation_complete(run)
                ]
                ready = bool(formal_runs)
                if formal_runs:
                    status = f"{len(formal_runs)} formaler Run"
                elif pilot_1k_current and pilot_10k_current:
                    status = "beide Entwicklungspiloten bestanden"
                elif pilot_1k_current:
                    status = "1.000er Methodenpilot bestanden"
                elif historical_formal_runs:
                    status = "veraltet: nach Featurelauf neu auswerten"
                elif runs:
                    status = f"{len(runs)} Vorschau-Run(s)"
                else:
                    status = "fehlt"
            else:
                ready = False
                status = "fehlt"
            rows.append((name, "bereit" if ready else "offen", status, note))
        self._fill_table(self.overview_metrics, rows)

        next_label, next_stage, target_tab = self._recommended_next_step()
        self._next_stage = next_stage
        self._next_target_tab = target_tab
        self.next_step_label.setText(next_label)
        self.next_step_button.setEnabled(bool(next_stage))

    def _recommended_next_step(self) -> tuple[str, str, int]:
        if (
            (self.config.work_root / "data_inventory.parquet").is_file()
            and not (
                self.config.work_root
                / "genetics"
                / "gwas_feasibility.json"
            ).is_file()
        ):
            return (
                "GWAS-Quellen inventarisieren und den erlaubten genetischen "
                "Claim-Umfang festschreiben.",
                "genetic-feasibility",
                1,
            )
        if not (self.config.work_root / "data_inventory.parquet").is_file():
            return "Dateninventar erstellen und Releases/Schemata prüfen.", "inventory", 1
        if not (
            self.config.standardized_root / "ingest.manifest.json"
        ).is_file():
            return "Rohdaten in versionierte Arbeitstabellen standardisieren.", "ingest", 1
        if not (
            self.config.standardized_root
            / "clinical_events"
            / "clinical_events.manifest.json"
        ).is_file():
            return (
                "Große klinische Ereignistabellen unveränderlich partitionieren.",
                "ingest-events",
                1,
            )
        if not (self.config.master_root / "ecg_master_index.parquet").is_file():
            return "ECG-zentrierten Masterindex und Subject-Splits aufbauen.", "build-index", 1
        if not (
            self.config.evidence_root / "machine_reports.manifest.json"
        ).is_file():
            return "Machine-Reports mit Negation und Unsicherheit parsen.", "parse-reports", 1
        if not (
            self.config.evidence_root / "notes.manifest.json"
        ).is_file():
            return (
                "Arztbriefe und Röntgen-Thorax-Berichte auf Risikofaktoren und Befunde durchsuchen.",
                "parse-notes",
                1,
            )
        if not (
            self.config.evidence_root / "evidence.manifest.json"
        ).is_file():
            return "Klinische Zeitfenster und Evidenztabellen erzeugen.", "build-evidence", 1
        if not (
            (
                self.config.evidence_root
                / "evidence_provenance.json"
            ).is_file()
            or (
                self.config.work_root
                / "provenance"
                / "clinical_event_evidence_equivalence.json"
            ).is_file()
        ):
            return (
                "Klinische Evidenz gegen die standardisierte Event-Schicht "
                "verifizieren.",
                "attest-event-evidence",
                1,
            )
        if not (
            self.config.labels_root / "endpoint_index_cohorts.parquet"
        ).is_file():
            return "Vierstufige klinische Labels und Kohorten erzeugen.", "build-labels", 1
        if not (
            self.config.labels_root
            / "development_label_attestation.json"
        ).is_file():
            return (
                "Train-/Validation-Labelregeln reproduzierbar attestieren.",
                "attest-labels",
                1,
            )
        delineation_benchmark_path = (
            self.config.resources_root
            / "delineation"
            / "benchmark_suite.json"
        )
        if not _delineation_benchmark_current(
            delineation_benchmark_path,
            expected_implementation_hash=self._current_delineation_hash,
        ):
            return (
                "Wellenerkennung extern an Grenzen, Rhythmus und Rauschen prüfen.",
                "validate-delineation",
                1,
            )
        biomarker_audit_path = (
            self.config.work_root
            / "audits"
            / "biomarkers"
            / "biomarker_audit_summary.json"
        )
        if not _biomarker_audit_current(
            biomarker_audit_path,
            expected_implementation_hash=self._current_biomarker_audit_hash,
        ):
            return (
                "Biomarker-Formeln, Einheiten, Invarianzen und Panelrollen prüfen.",
                "audit-biomarkers",
                1,
            )
        preset_id = self._preset_id()
        preset = get_analysis_preset(preset_id)
        pilot_1k_current = self._pilot_validation_current(1_000)
        pilot_10k_current = self._pilot_validation_current(10_000)
        if not pilot_1k_current:
            return (
                "Zuerst den endpunktabdeckenden 1.000er-Methodenpilot "
                f"für „{preset.display_name_de}“ starten.",
                "study-run",
                TAB_ANALYSIS,
            )
        if not pilot_10k_current:
            return (
                "Der 1.000er-Methodenpilot ist bestanden. Als Nächstes den "
                "10.000er-Skalierungspilot für den Monatslauf starten.",
                "study-run",
                TAB_ANALYSIS,
            )
        selected_feature_manifest = self._selected_feature_manifest_path()
        if not _feature_manifest_complete(
            selected_feature_manifest,
            expected_code_hash=self._current_feature_code_hash,
        ):
            pilot_summary = (
                "1.000er-Methoden- und 10.000er-Skalierungspilot sind "
                "bestanden. "
                if preset_id == "exhaustive_research"
                else "Der 1.000er-Methodenpilot ist bestanden. "
            )
            return (
                pilot_summary
                + f"Die vollständige Analyse „{preset.display_name_de}“ kann "
                "jetzt gestartet oder fortgesetzt werden.",
                "study-run",
                TAB_ANALYSIS,
            )
        selected_run = self.config.runs_root / self._current_run_id()
        if not _formal_development_validation_complete(
            selected_run,
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        ):
            return "Train-/Validation-Auswertung starten.", "validate", 1
        if not (
            selected_run / "phenome" / "phenome.manifest.json"
        ).is_file():
            return (
                "Breiten Pathologie-Scan in Train entdecken und in "
                "Validation replizieren.",
                "discover-phenome",
                1,
            )
        if not _required_group_vcg_complete(selected_run):
            return (
                "Die sechs vorab festgelegten Validation-Gruppen-VCGs "
                "als Abnahme-Set berechnen.",
                "results",
                3,
            )
        if not (
            selected_run / "reports" / "report.manifest.json"
        ).is_file():
            return (
                "Kanonische und druckbare Entwicklungsberichte erzeugen.",
                "report",
                1,
            )
        audit_path = (
            selected_run / "acceptance_audit.json"
            if (selected_run / "freeze_manifest.json").is_file()
            else selected_run / "acceptance_pre_freeze_audit.json"
        )
        audit = _read_json(audit_path)
        if not audit:
            return (
                "Wissenschaftlichen Abnahme-Audit ausführen.",
                "audit",
                1,
            )
        if audit.get("status") == "failed":
            return (
                "Offene Abnahmepunkte beheben und Audit erneut ausführen.",
                "audit",
                1,
            )
        return (
            "Ergebnisse, Stabilität und Leitlinienvergleich prüfen.",
            "results",
            TAB_RESULTS,
        )

    def _selected_feature_manifest_path(self) -> Path:
        preset = get_analysis_preset(self._preset_id())
        panel_key = "_".join(preset.panel_ids)
        return (
            self.config.features_root
            / panel_key
            / "development"
            / "features.manifest.json"
        )

    def _pilot_validation_current(self, size: int) -> bool:
        run_id = self._current_run_id()
        return _pilot_validation_compatible(
            self.config.runs_root / f"{run_id}-pilot-{int(size)}",
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        )

    def _open_next_step(self) -> None:
        self.workspace_tabs.setCurrentIndex(self._next_target_tab)
        if self._next_stage and self._next_stage != "results":
            index = self.stage_combo.findData(self._next_stage)
            if index >= 0:
                self.stage_combo.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Job actions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Analysis identity
    # ------------------------------------------------------------------

    def _current_analysis(self, *, refresh: bool = False):
        """The analysis this configuration starts or continues.

        Cached because deriving it hashes the marker tree and the panels,
        which costs a few hundred milliseconds -- fine once, not on every
        repaint.
        """

        if refresh or getattr(self, "_analysis_record", None) is None:
            try:
                record, _ = resolve_current_analysis(self.config)
            except Exception as exc:
                # A project that cannot be fingerprinted yet (paths missing,
                # panels unbuildable) must still open the window.
                LOGGER.warning("Analysis lineage resolution failed: %s", exc)
                self._analysis_record = None
                return None
            self._analysis_record = record
        return self._analysis_record

    def _current_run_id(self) -> str:
        record = self._current_analysis()
        return record.run_id if record is not None else DEFAULT_DEVELOPMENT_RUN_ID

    def _selected_current_run_id(self, combo: QComboBox) -> str:
        """Return a GUI selection only when it is the current lineage.

        Combo boxes are refreshed asynchronously while jobs write artifacts.
        A previously selected historical value can therefore survive for a
        short interval after the scientific fingerprint changes.  Every
        loader/action rechecks the identity at use time instead of trusting
        the visible list alone.
        """

        selected = str(combo.currentData() or "")
        return selected if selected == self._current_run_id() else ""

    def _refresh_analysis_identity(self, *, refresh: bool = False) -> None:
        record = self._current_analysis(refresh=refresh)
        if record is None:
            self.analysis_name_label.setText("noch nicht bestimmbar")
            self.analysis_lineage_label.setText(
                "Die Analyse wird benannt, sobald Projektpfade und Panels "
                "gelesen werden können."
            )
            return
        frozen = analysis_is_frozen(self.config.runs_root, record.run_id)
        self.analysis_name_label.setText(
            f"{record.run_id}{' — eingefroren' if frozen else ''}"
        )
        self.analysis_lineage_label.setText(record.describe_change_de())
        if self.analysis_notes.text() != record.notes:
            self.analysis_notes.setText(record.notes)

    def _save_analysis_notes(self) -> None:
        record = self._current_analysis()
        if record is None:
            return
        text = self.analysis_notes.text().strip()
        if text == record.notes:
            return
        AnalysisLineage(self.config.runs_root).set_notes(record.run_id, text)
        record.notes = text

    def _preset_id(self) -> str:
        """The study's one configuration.

        Kept as a method rather than inlined so the call sites read the same
        as before and a future second arm, if one is ever justified, has one
        place to appear.
        """

        return STUDY_PRESET_ID

    def _preset_changed(self) -> None:
        preset = get_analysis_preset(self._preset_id())
        self.preset_description.setText(
            f"{preset.description_de}\nDie Rechenzeit ist keine Einstellung: "
            "der Lauf verfeinert weiter, bis Sie ihn beenden, und zeigt "
            "dabei an, wie viel der letzte Zuwachs noch gebracht hat."
        )
        self.bootstrap_iterations.setValue(
            preset.policy.bootstrap_iterations
        )
        self._set_job_action_state(self._matching_study_job())

    def _update_technical_stage_explanation(self) -> None:
        stage = str(self.stage_combo.currentData() or "")
        if not stage:
            self.technical_stage_explanation.clear()
            return
        descriptor = stage_descriptor(stage)
        prerequisites = (
            ", ".join(
                stage_descriptor(item).display_name_de
                for item in descriptor.prerequisite_stage_ids
            )
            or "keine"
        )
        self.technical_stage_explanation.setText(
            f"Wozu: {descriptor.purpose_de}\n"
            f"Voraussetzung: {prerequisites}\n"
            f"Ergebnis: {descriptor.result_de}"
        )

    def _matching_study_job(self):
        run_id = self._current_run_id()
        preset_id = self._preset_id()
        try:
            records = self.job_store.list(limit=500)
        except Exception as exc:
            LOGGER.warning("Job store read failed: %s", exc)
            return None
        matches = [
            record
            for record in records
            if record.spec.stage == "study-run"
            and str(record.spec.arguments.get("run_id", "")) == run_id
            and str(record.spec.arguments.get("preset_id", "")) == preset_id
        ]
        if not matches:
            return None
        # Pilot and separately authorised full continuation share one
        # scientific run id.  Always select the newest actionable generation;
        # otherwise an older completed pilot can mask the running full job.
        return max(matches, key=lambda value: str(value.spec.created_at))

    @staticmethod
    def _select_run_in_combo(combo: QComboBox, run_id: str) -> None:
        index = combo.findData(run_id)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_job_action_state(self, record) -> None:
        if not hasattr(self, "study_action_button"):
            return
        if record is None:
            approved = _full_run_authorized(self.config.runs_root / self._current_run_id())
            self.study_action_button.setText("Voll-Lauf starten" if approved else "Analyse starten")
            self.study_action_button.setEnabled(True)
            self.analysis_status_label.setText(
                "Development-Voll-Lauf freigegeben. Start erfolgt erst per Klick."
                if approved else "Noch kein geführtes Analysevorhaben mit dieser Auswahl."
            )
            return
        state = record.state
        if state == JobState.PAUSED:
            label = "Analyse fortsetzen"
            enabled = True
            status = (
                "Sicher pausiert. Fertige Mikro-Batches bleiben erhalten; "
                "Fortsetzen beginnt am letzten Checkpoint."
            )
        elif state in {JobState.FAILED, JobState.BLOCKED, JobState.CANCELLED}:
            label = "Analyse erneut starten"
            enabled = True
            status = record.error_message or (
                "Die Analyse benötigt Aufmerksamkeit und kann mit denselben "
                "Einstellungen erneut gestartet werden."
            )
        elif state == JobState.COMPLETED:
            if _job_is_pilot_only(record) and _full_run_authorized(
                self.config.runs_root / self._current_run_id()
            ):
                label = "Freigegebenen Voll-Lauf starten"
                enabled = True
                status = (
                    "Beide Entwicklungspiloten und alle Go/No-Go-Gates sind "
                    "bestanden. Der Locked-Test bleibt im Voll-Lauf versiegelt."
                )
            else:
                label = "Ergebnisse öffnen"
                enabled = True
                status = "Die Analyse ist abgeschlossen; Ergebnisse sind verfügbar."
        else:
            label = "Laufende Analyse anzeigen"
            enabled = True
            status = record.progress_message or (
                "Die Analyse läuft im Hintergrund und darf parallel ausgewertet werden."
            )
        self.study_action_button.setText(label)
        self.study_action_button.setEnabled(enabled)
        self.analysis_status_label.setText(status)

    def _set_selected_job_controls(self, record) -> None:
        if record is not None and not _job_visible_for_current_run(record, self._current_run_id()):
            record = None  # Frozen foreign-fingerprint jobs are observable, not restartable.
        if record is None:
            for button in (
                self.pause_button,
                self.resume_button,
                self.retry_button,
                self.cancel_button,
                self.priority_button,
            ):
                button.setEnabled(False)
                if button is not self.priority_button:
                    button.setVisible(False)
            return
        action_states = (
            (
                self.pause_button,
                record.state in {JobState.RUNNING, JobState.QUEUED},
            ),
            (self.resume_button, record.state == JobState.PAUSED),
            (
                self.retry_button,
                record.state
                in {JobState.FAILED, JobState.CANCELLED, JobState.BLOCKED},
            ),
            (
                self.cancel_button,
                record.state
                not in {
                    JobState.CANCELLED,
                    JobState.COMPLETED,
                    JobState.FAILED,
                },
            ),
        )
        for button, available in action_states:
            button.setEnabled(available)
            button.setVisible(available)
        self.priority_button.setEnabled(
            record.state
            not in {
                JobState.CANCELLED,
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.BLOCKED,
            }
        )

    def start_or_resume_analysis(self) -> None:
        try:
            self._save_project_fields()
            record = self._matching_study_job()
            if record is not None:
                self._selected_job_id = record.spec.job_id
                if record.state == JobState.PAUSED:
                    self.job_store.request_resume(record.spec.job_id)
                    ensure_supervisor(self.config)
                elif record.state in {
                    JobState.FAILED,
                    JobState.BLOCKED,
                    JobState.CANCELLED,
                }:
                    self.job_store.retry(record.spec.job_id)
                    ensure_supervisor(self.config)
                elif record.state == JobState.COMPLETED:
                    run_id = self._current_run_id()
                    preset_id = self._preset_id()
                    if _job_is_pilot_only(record) and _full_run_authorized(
                        self.config.runs_root / run_id
                    ):
                        spec = guided_study_spec(
                            preset_id=preset_id,
                            run_id=run_id,
                            include_external_validation=False,
                            stop_after_pilots=False,
                        )
                        full_record = self.job_store.create(
                            stage="study-run",
                            project_file=self.config.save(),
                            arguments={
                                "study_spec": spec.to_dict(),
                                "preset_id": preset_id,
                                "run_id": run_id,
                                "include_locked_test": False,
                                "stop_after_pilots": False,
                            },
                            priority=60,
                            requested_by="gui_full_after_go_no_go",
                        )
                        ensure_supervisor(self.config)
                        self._selected_job_id = full_record.spec.job_id
                        self.refresh_jobs()
                        return
                    self.workspace_tabs.setCurrentIndex(TAB_RESULTS)
                    self._select_run_in_combo(
                        self.results_run_combo,
                        str(record.spec.arguments.get("run_id", "")),
                    )
                    self.refresh_results()
                    return
                self.refresh_jobs()
                return
            errors = self.config.validate_paths()
            if errors:
                raise RuntimeError(
                    "Das Projekt ist noch nicht startbereit:\n- "
                    + "\n- ".join(errors)
                )
            run_id = self._current_run_id()
            assert_development_run_mutable(
                self.config.runs_root / run_id,
                operation="Geführte Train-/Validation-Analyse",
            )
            preset_id = self._preset_id()
            priority = 70
            record = enqueue_guided_study(
                self.config,
                preset_id=preset_id,
                run_id=run_id,
                priority=priority,
                requested_by="gui",
                stop_after_pilots=not _full_run_authorized(self.config.runs_root / run_id),
            )
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Analyse konnte nicht gestartet werden",
                str(exc),
            )

    def _toggle_expert(self, visible: bool) -> None:
        self.expert_controls.setVisible(visible)

    def _update_stage_parameter_visibility(self) -> None:
        stage = str(self.stage_combo.currentData())
        self.future_horizons_combo.setEnabled(stage == "discover-phenome")
        self.future_phenotype_limit_combo.setEnabled(
            stage == "discover-phenome"
        )

    def enqueue_selected_stage(self) -> None:
        try:
            self._save_project_fields()
            stage = str(self.stage_combo.currentData())
            arguments = self._stage_arguments(stage)
            if stage in {"validate", "discover-phenome"}:
                assert_development_run_mutable(
                    self.config.runs_root / str(arguments["run_id"]),
                    operation=(
                        "Train/validation analysis"
                        if stage == "validate"
                        else "Phenome discovery"
                    ),
                )
            record = self.job_store.create(
                stage=stage,
                project_file=self.config.project_file,
                arguments=arguments,
                priority=(
                    70
                    if stage in (
                        "validate",
                        "discover-phenome",
                        "audit",
                    )
                    else 50
                ),
                requested_by="gui",
            )
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(self, "Job konnte nicht starten", str(exc))

    def _stage_arguments(self, stage: str) -> dict:
        preset = get_analysis_preset(self._preset_id())
        arguments: dict = {"force": False}
        if stage == "inventory":
            return {"count_rows": False}
        if stage == "extract-features":
            arguments.update(
                {
                    "panel_ids": list(preset.panel_ids),
                    "include_locked_test": False,
                    "max_records": (
                        self.max_records.value()
                        if self.max_records.value() > 0
                        else None
                    ),
                    "partition_size": self.partition_size.value(),
                    "micro_batch_size": self.micro_batch_size.value(),
                }
            )
        elif stage == "validate":
            arguments.update(
                {
                    "run_id": self._current_run_id(),
                    "panel_ids": list(preset.panel_ids),
                    "bootstrap_iterations": self.bootstrap_iterations.value(),
                    "include_predefined_scores": True,
                    "include_ptb_frozen_scores": True,
                    "include_exploratory_scores": True,
                    "include_all_ecg_secondary": True,
                    "grouped_score_folds": preset.policy.grouped_folds,
                    "grouped_score_repeats": (
                        preset.policy.grouped_cv_repeats
                    ),
                    "maximum_component_abs_correlation": (
                        preset.policy.maximum_component_abs_correlation
                    ),
                    "minimum_score_incremental_auc": (
                        preset.policy.minimum_score_incremental_auc
                    ),
                }
            )
        elif stage == "discover-phenome":
            exhaustive = preset.policy.include_broad_phenome_scan
            arguments.update(
                {
                    "run_id": self._current_run_id(),
                    "panel_ids": list(
                        preset.panel_ids
                        if exhaustive
                        else preset.panel_ids[:2]
                    ),
                    "bootstrap_iterations": self.bootstrap_iterations.value(),
                    "train_fdr_alpha": 0.01,
                    "validation_fdr_alpha": 0.05,
                    "min_class_inference": 30,
                    "min_class_formal": 100,
                    "future_horizons_days": list(
                        self.future_horizons_combo.currentData()
                    ),
                    "max_future_phenotypes": (
                        None
                        if exhaustive
                        else self.future_phenotype_limit_combo.currentData()
                    ),
                    "max_phenotypes": None,
                    "include_promoted_candidates": (
                        self.include_promoted_candidates_checkbox.isChecked()
                    ),
                }
            )
        elif stage == "report":
            arguments.update(
                {
                    "run_id": self._current_run_id(),
                }
            )
        elif stage == "audit":
            run_id = self._current_run_id()
            run_root = self.config.runs_root / run_id
            arguments.update(
                {
                    "run_id": run_id,
                    "ptb_run_id": DEFAULT_PTB_TRANSFER_RUN_ID,
                    "expected_ecg_count": 800_035,
                    "expected_subject_count": 161_352,
                    "expected_echo_study_count": 206_488,
                    "expected_label_definition_count": 63,
                    "minimum_bootstrap_iterations": 1_000,
                    "group_vcg_endpoints": [
                        "acute_mi_high_specificity",
                        "echo_lvh_strict",
                        "reduced_lvef",
                        "right_heart_strain_proxy",
                        "echo_lae",
                        "hyperkalemia_near_ecg",
                    ],
                    "require_freeze": (
                        run_root / "freeze_manifest.json"
                    ).is_file(),
                    "strict": True,
                    "deep_feature_verification": True,
                }
            )
        return arguments

    def enqueue_benchmark(self) -> None:
        try:
            self._save_project_fields()
            record = self.job_store.create(
                stage="benchmark",
                project_file=self.config.project_file,
                arguments={
                    "candidates": [1, 2, 4, 6, 7],
                    "tasks_per_worker": 3,
                    "apply_recommendation": True,
                },
                priority=80,
                requested_by="gui",
            )
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(self, "Benchmark", str(exc))

    def refresh_jobs(self) -> None:
        try:
            records = self.job_store.list(limit=250)
            background_records = self.job_store.list(
                states=[JobState.RUNNING, JobState.QUEUED, JobState.PAUSING, JobState.CANCELLING,
                        JobState.RESUMING, JobState.PAUSED], limit=1000)
        except Exception as exc:
            self.supervisor_status.setText(f"Jobdatenbank: {exc}")
            self.background_analysis_status.setText('Hintergrundstatus unbekannt: Jobdatenbank nicht lesbar.')
            return
        from ..background_job_view import background_job_notice
        self.background_analysis_status.setText(background_job_notice(background_records))
        current_run_id = self._current_run_id()
        records = _jobs_for_monitor(records, background_records, current_run_id)
        if self._selected_job_id and not any(
            record.spec.job_id == self._selected_job_id for record in records
        ):
            self._selected_job_id = None
            self._set_selected_job_controls(None)
        selected = self._selected_job_id
        grouped = {key: [] for key in self.job_tables}
        completed_by_key: dict[tuple[str, str], str] = {}
        for record in records:
            if record.state != JobState.COMPLETED:
                continue
            key = _job_supersession_key(record)
            completed_by_key[key] = max(
                completed_by_key.get(key, ""),
                record.spec.created_at,
            )
        for record in records:
            group_id = _job_display_group(record, completed_by_key)
            grouped[group_id].append(record)
        selected_location: Optional[tuple[str, int]] = None
        group_titles = {
            "active": "Läuft",
            "attention": "Benötigt Aufmerksamkeit",
            "history": "Verlauf",
        }
        for group_id, table in self.job_tables.items():
            table.blockSignals(True)
            rows = grouped[group_id]
            tab_index = self.job_group_tabs.indexOf(table)
            if tab_index >= 0:
                self.job_group_tabs.setTabText(
                    tab_index,
                    f"{group_titles[group_id]} ({len(rows)})",
                )
            table.setRowCount(len(rows))
            for row, record in enumerate(rows):
                job_run_id = _job_run_id(record)
                calibrated = calibrated_job_progress(
                    record,
                    (
                        self.config.runs_root / job_run_id
                        if job_run_id
                        else None
                    ),
                    expected_development_records=development_record_count(self.config.master_root),
                )
                values = (
                    _job_task_label(record) + (
                        f"\n{job_run_id} · eingefrorener Hintergrundlauf (nur Anzeige)"
                        if not _job_visible_for_current_run(record, current_run_id) else ""),
                    _job_state_label(record.state),
                    _job_progress_text(record, calibrated),
                    _job_eta_text(record, calibrated),
                    _job_result_text(record),
                    _job_action_required_text(record),
                )
                resource_text = (
                    grant_status_text(
                        self.config.jobs_root / "resource_grants",
                        record.spec.job_id, record.worker_pid,
                    ) if record.state == JobState.RUNNING else ""
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    tooltip = _job_cell_tooltip(record, column, calibrated)
                    item.setToolTip(tooltip + ("\n" + resource_text if resource_text else ""))
                    if column == 2 and resource_text:
                        item.setText(str(value) + "\n" + resource_text)
                    if column == 0:
                        item.setData(
                            Qt.ItemDataRole.UserRole,
                            record.spec.job_id,
                        )
                    if column == 1:
                        item.setForeground(_state_color(record.state))
                    table.setItem(row, column, item)
                if record.spec.job_id == selected:
                    selected_location = (group_id, row)
            table.resizeRowsToContents()
            table.blockSignals(False)
        if selected_location is not None:
            group_id, row = selected_location
            table = self.job_tables[group_id]
            table.selectRow(row)
            self.job_group_tabs.setCurrentWidget(table)

        self._set_job_action_state(self._matching_study_job())

        status = self.job_store.supervisor_status()
        heartbeat = str(status.get("heartbeat_at") or "")
        runtime_notice = supervisor_runtime_notice(
            status.get("pid"), self.config.jobs_root / "runtime_snapshots"
        ) if status else ""
        self.supervisor_status.setText(
            (
                f"Supervisor PID {status.get('pid')} · {heartbeat[11:19]} UTC\n{runtime_notice}"
                if status
                else "Supervisor startet mit dem ersten Job"
            )
        )
        hardware = detect_hardware()
        budget = global_memory_budget_gb(
            hardware,
            reserve_gb=self.config.os_memory_reserve_gb,
        )
        memory = detect_memory_status()
        critical_gb = critical_available_memory_gb(
            reserve_gb=self.config.os_memory_reserve_gb,
        )
        available_text = (
            f"{memory.available_gb:.1f}/{hardware.total_memory_gb:.1f} GB frei"
            if memory is not None
            else f"{hardware.total_memory_gb:.1f} GB gesamt"
        )
        self.resource_status.setText(
            f"{hardware.physical_cores} Kerne · RAM {available_text} · "
            f"Jobbudget {budget:.1f} GB"
        )
        self.resource_status.setToolTip(
            "Der Supervisor startet Hintergrundjobs nur mit realem "
            "Speicherpolster. Konfigurierte OS-/GUI-Reserve: "
            f"{self.config.os_memory_reserve_gb:.1f} GB. "
            f"Automatische Pausenschwelle: {critical_gb:.1f} GB "
            "nach drei aufeinanderfolgenden Messungen."
        )
        if memory is not None and memory.available_gb < critical_gb:
            self.resource_status.setStyleSheet(
                theme.status_qss("danger", self._dark_mode)
            )
        elif (
            memory is not None
            and memory.available_gb < self.config.os_memory_reserve_gb
        ):
            self.resource_status.setStyleSheet(
                theme.status_qss("warning", self._dark_mode)
            )
        else:
            self.resource_status.setStyleSheet("")
        self._update_resource_gauges(hardware, memory, critical_gb)
        self._sync_benchmark_recommendation()

    def _update_resource_gauges(
        self,
        hardware,
        memory,
        critical_gb: float,
    ) -> None:
        cpu_percent = current_cpu_percent()
        if cpu_percent is not None:
            self.cpu_gauge.setValue(int(round(min(100.0, max(0.0, cpu_percent)))))
            self.cpu_gauge.setStyleSheet(
                theme.gauge_chunk_qss("critical", self._dark_mode)
                if cpu_percent >= 90.0
                else theme.gauge_chunk_qss("warning", self._dark_mode)
                if cpu_percent >= 70.0
                else theme.gauge_chunk_qss("normal", self._dark_mode)
            )
            self.cpu_gauge.setToolTip(
                f"Systemweite CPU-Auslastung: {cpu_percent:.0f} % "
                f"(von {hardware.logical_cores} logischen Kernen)."
            )
        if memory is not None:
            self.ram_gauge.setValue(
                int(round(min(100.0, max(0.0, memory.used_percent))))
            )
            self.ram_gauge.setStyleSheet(
                theme.gauge_chunk_qss("critical", self._dark_mode)
                if memory.available_gb < critical_gb
                else theme.gauge_chunk_qss("warning", self._dark_mode)
                if memory.available_gb < self.config.os_memory_reserve_gb
                else theme.gauge_chunk_qss("normal", self._dark_mode)
            )
            self.ram_gauge.setToolTip(
                f"RAM-Auslastung: {memory.used_percent:.0f} % "
                f"({memory.available_gb:.1f} GB frei von "
                f"{hardware.total_memory_gb:.1f} GB)."
            )

    def _sync_benchmark_recommendation(self) -> None:
        path = self.config.jobs_root / "hardware_benchmark.json"
        if not path.is_file():
            return
        mtime = path.stat().st_mtime_ns
        payload = _read_json(path)
        recommended = int(
            payload.get("recommended_workers", self.config.workers)
        )
        if mtime != self._benchmark_mtime_ns:
            self._benchmark_mtime_ns = mtime
            self.config.workers = recommended
            self.workers_spin.setValue(recommended)
        self.resource_status.setText(
            self.resource_status.text()
            + f" · Empfehlung {recommended} Worker"
        )

    def _job_selected(self, group_id: str = "active") -> None:
        table = self.job_tables.get(group_id, self.jobs_table)
        rows = table.selectionModel().selectedRows()
        if not rows:
            return
        for other_group, other_table in self.job_tables.items():
            if other_group == group_id:
                continue
            other_table.blockSignals(True)
            other_table.clearSelection()
            other_table.blockSignals(False)
        item = table.item(rows[0].row(), 0)
        self._selected_job_id = str(
            item.data(Qt.ItemDataRole.UserRole) or item.text()
        )
        try:
            record = self.job_store.get(self._selected_job_id)
            self.priority_spin.setValue(record.spec.priority)
            self._set_selected_job_controls(record)
            self.job_explanation.setText(_job_plain_language_explanation(record))
            events = self.job_store.events(self._selected_job_id)
            profile = record.spec.resource_profile
            technical_header = (
                f"Job-ID: {record.spec.job_id}\n"
                f"Interne Stufe: {record.spec.stage}\n"
                f"Ressourcen: {profile.cpu_tokens} CPU-Tokens / "
                f"{profile.memory_gb:g} GB RAM / I/O {profile.io_weight:g}\n"
                f"Artefakthash: {record.artifact_hash or 'noch nicht vorhanden'}\n\n"
            )
            self.job_events.setPlainText(
                technical_header
                + "\n".join(
                    f"{event['created_at'][11:19]} "
                    f"{event['event_type']}: {event['message']}"
                    for event in events[-80:]
                )
            )
        except Exception as exc:
            self.job_events.setPlainText(str(exc))
            self.job_explanation.setText(str(exc))

    def _show_selected_job_telemetry(self) -> None:
        from ..job_telemetry import telemetry_preview
        if not self._selected_job_id:
            return
        try:
            record = self.job_store.get(self._selected_job_id)
        except (KeyError, OSError, RuntimeError) as exc:
            logging.getLogger(__name__).exception("Cannot load selected job for telemetry")
            QMessageBox.warning(self, "Messaufnahme nicht verfuegbar", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("SQL-/Ressourcen-Messaufnahme")
        layout = QVBoxLayout(dialog)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setPlainText(telemetry_preview(self.config, record.spec))
        layout.addWidget(preview)
        dialog.resize(780, 440)
        dialog.exec()

    def pause_selected_job(self) -> None:
        self._apply_job_action(self.job_store.request_pause)

    def resume_selected_job(self) -> None:
        self._apply_job_action(self.job_store.request_resume, start_supervisor=True)

    def retry_selected_job(self) -> None:
        if not self._selected_job_id:
            return
        try:
            record = self.job_store.get(self._selected_job_id)
        except Exception as exc:
            QMessageBox.warning(self, "Analyse fortsetzen", str(exc))
            self.refresh_jobs()
            return
        if not _job_visible_for_current_run(record, self._current_run_id()):
            self._selected_job_id = None
            self.refresh_jobs()
            QMessageBox.warning(
                self,
                "Inkompatibler Lauf",
                "Dieser historische Lauf passt nicht zum aktuellen "
                "Analyse-Fingerprint und kann aus der GUI nicht erneut "
                "gestartet werden.",
            )
            return
        self._apply_job_action(self.job_store.retry, start_supervisor=True)

    def cancel_selected_job(self) -> None:
        self._apply_job_action(self.job_store.request_cancel)

    def update_selected_priority(self) -> None:
        if not self._selected_job_id:
            return
        try:
            record = self.job_store.get(self._selected_job_id)
            if not _job_visible_for_current_run(record, self._current_run_id()):
                return
            self.job_store.set_priority(
                self._selected_job_id,
                self.priority_spin.value(),
            )
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.warning(self, "Priorität", str(exc))

    def _apply_job_action(self, action, *, start_supervisor: bool = False) -> None:
        if not self._selected_job_id:
            return
        try:
            record = self.job_store.get(self._selected_job_id)
            if not _job_visible_for_current_run(record, self._current_run_id()):
                QMessageBox.information(self, 'Eingefrorener Hintergrundlauf',
                    'Dieser Lauf wird hier nur angezeigt. Sein unabhängiger Worker läuft unverändert weiter.')
                return
            action(self._selected_job_id)
            if start_supervisor:
                ensure_supervisor(self.config)
            self.refresh_jobs()
        except Exception as exc:
            # The job may have changed state between the last poll and the
            # click -- a stage finishing is enough. That is a stale view, not
            # a failure, so refresh and say so plainly.
            self.refresh_jobs()
            QMessageBox.information(
                self,
                "Jobsteuerung",
                f"{exc}\n\nDie Ansicht wurde soeben aktualisiert; der "
                "Zustand hatte sich zwischen Anzeige und Klick geändert.",
            )

    def _poll_background_state(self) -> None:
        self.refresh_jobs()
        self._refresh_deep_search_status()
        signature = self._ui_artifact_signature()
        if signature == self._last_ui_artifact_signature:
            return
        self._last_ui_artifact_signature = signature
        self._refresh_overview()
        self._refresh_run_choices()
        if (
            self.workspace_tabs.currentIndex() == TAB_PROJECT
            and self.project_tabs.currentIndex() == 1
        ):
            self.refresh_cohorts()
        elif self.workspace_tabs.currentIndex() == TAB_RESULTS:
            self.refresh_results()
        elif self.workspace_tabs.currentIndex() == TAB_REPORTS:
            self.refresh_review()

    def _ui_artifact_signature(self) -> tuple:
        results_run = self._selected_current_run_id(self.results_run_combo)
        report_run = self._selected_current_run_id(self.report_run_combo)
        feature_failure_path = self._feature_failure_path(results_run)
        endpoint_id = str(self.endpoint_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "validation")
        group_vcg_summary = (
            group_vcg_artifact_dir(
                self.config,
                results_run,
                endpoint_id,
                split,
            )
            / "group_vcg_summary.parquet"
            if results_run and endpoint_id
            else self.config.runs_root / ".no-group-vcg-selection"
        )
        paths = (
            self.config.work_root / "data_inventory.parquet",
            self.config.labels_root / "label_summary.parquet",
            self.config.labels_root / "split_summary.parquet",
            feature_failure_path,
            self.config.runs_root / results_run / "screening_results.parquet",
            self.config.runs_root / results_run / "validate.manifest.json",
            self.config.runs_root / results_run / "pathology_marker_explorer.parquet",
            self.config.runs_root / results_run / "phenome" / "phenome.manifest.json",
            self.config.runs_root / results_run / "broad_axes" / "broad_axes.manifest.json",
            self.config.external_validation_root / results_run / "population_cohorts" / "population_cohorts.manifest.json",
            self.config.runs_root
            / results_run
            / "vcg_vs_classical_comparison.parquet",
            self.config.runs_root
            / results_run
            / "overfitting_stability_report.parquet",
            self.config.runs_root
            / report_run
            / "reports"
            / "review_candidates.parquet",
            group_vcg_summary,
        )
        return tuple(
            (
                str(path),
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            if path.is_file()
            else (str(path), 0, 0)
            for path in paths
        ) + (self._live_feature_signature(),)

    # ------------------------------------------------------------------
    # Cohorts/results
    # ------------------------------------------------------------------

    def refresh_cohorts(self) -> None:
        inventory_path = self.config.work_root / "data_inventory.parquet"
        label_path = self.config.labels_root / "label_summary.parquet"
        split_path = self.config.labels_root / "split_summary.parquet"
        window_path = (
            self.config.labels_root / "label_window_sensitivity_summary.parquet"
        )
        run_id = str(
            self._selected_current_run_id(self.results_run_combo)
            or self._current_run_id()
            or ""
        )
        failure_path = self._feature_failure_path(run_id)
        self.inventory_view.set_source(
            inventory_path if inventory_path.is_file() else None
        )
        self.label_view.set_source(label_path if label_path.is_file() else None)
        self.split_view.set_source(split_path if split_path.is_file() else None)
        self.window_view.set_source(
            window_path if window_path.is_file() else None
        )
        self.feature_failure_view.set_source(
            failure_path if failure_path.is_file() else None
        )
        self.cohort_context.setText(
            "Fehlende Referenzdaten bleiben unavailable; negative Kontrollen "
            "erfordern dokumentierte Beurteilbarkeit. Alle Splits liegen auf "
            "subject_id-Ebene."
        )
        self._update_qc_plot(label_path)
        self._update_window_plot(window_path)
        self._update_feature_failure_plot(failure_path)
        self._refresh_live_features()
        self._update_qc_table(split_path, label_path)
        self._update_foundation_overview(
            label_path=label_path,
            window_path=window_path,
            failure_path=failure_path,
        )

    def _update_foundation_overview(
        self,
        *,
        label_path: Path,
        window_path: Path,
        failure_path: Path,
    ) -> None:
        master = _read_json(
            self.config.master_root / "ecg_master_index_summary.json"
        )
        counts = dict(master.get("counts", {}))
        ecg_count = int(counts.get("ecg_count", 0) or 0)
        subject_count = int(counts.get("subject_count", 0) or 0)
        split_subjects = sum(
            int(item.get("subject_count", 0) or 0)
            for item in master.get("splits", ())
        )
        split_ok = bool(subject_count) and split_subjects == subject_count

        try:
            labels = (
                pd.read_parquet(label_path)
                if label_path.is_file()
                else pd.DataFrame()
            )
        except (OSError, ValueError):
            labels = pd.DataFrame()
        formal_count = int(
            labels.get(
                "power_status",
                pd.Series(dtype=str),
            ).astype(str).eq("formal_analysis_eligible").sum()
        )
        unavailable_fraction = None
        if not labels.empty and {
            "unavailable_count",
            "ecg_count",
        }.issubset(labels):
            denominator = pd.to_numeric(
                labels["ecg_count"], errors="coerce"
            ).sum()
            unavailable = pd.to_numeric(
                labels["unavailable_count"], errors="coerce"
            ).sum()
            if denominator > 0:
                unavailable_fraction = float(unavailable / denominator)

        try:
            windows = (
                pd.read_parquet(window_path)
                if window_path.is_file()
                else pd.DataFrame()
            )
        except (OSError, ValueError):
            windows = pd.DataFrame()
        median_delta = None
        if "median_absolute_delta_hours" in windows:
            values = pd.to_numeric(
                windows["median_absolute_delta_hours"],
                errors="coerce",
            ).dropna()
            if not values.empty:
                median_delta = float(values.median())

        try:
            failures = (
                pd.read_parquet(failure_path)
                if failure_path.is_file()
                else pd.DataFrame()
            )
        except (OSError, ValueError):
            failures = pd.DataFrame()
        failed_ecgs = int(
            pd.to_numeric(
                failures.get("record_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )

        blockers = []
        if not ecg_count or not subject_count:
            blockers.append("ECG-Masterindex fehlt")
        if labels.empty:
            blockers.append("klinische Referenzgruppen fehlen")
        elif formal_count == 0:
            blockers.append("noch kein Ziel hat genug Fälle und Kontrollen")
        if subject_count and not split_ok:
            blockers.append("Patientenzahlen der Splits sind inkonsistent")
        if not window_path.is_file():
            blockers.append("zeitliche Zuordnung ist noch nicht zusammengefasst")

        rows = [
            (
                "Wie viele Patienten und ECGs sind indexiert?",
                (
                    f"{subject_count:,} Patienten · {ecg_count:,} ECGs"
                    if subject_count and ecg_count
                    else "Noch kein vollständiger ECG-Masterindex"
                ),
                "bereit" if subject_count and ecg_count else "noch offen",
                "Alle ECGs eines Patienten bleiben in demselben Entwicklungssplit.",
            ),
            (
                "Wie viele klinische Ziele sind belastbar prüfbar?",
                (
                    f"{formal_count} von {len(labels)} Zielen formal auswertbar"
                    if not labels.empty
                    else "Klinische Referenzen noch nicht erstellt"
                ),
                "bereit" if formal_count else "Achtung",
                (
                    "Formale Primärtests benötigen mindestens 100 positive und "
                    "100 negative Patienten; kleinere Gruppen bleiben deskriptiv."
                ),
            ),
            (
                "Wie viele Referenzangaben fehlen?",
                (
                    f"{unavailable_fraction:.1%} der Endpunkt-ECG-Zuordnungen"
                    if unavailable_fraction is not None
                    else "Noch nicht bestimmbar"
                ),
                (
                    "transparent ausgewiesen"
                    if unavailable_fraction is not None
                    else "noch offen"
                ),
                "Fehlende Untersuchung wird niemals automatisch als gesund gewertet.",
            ),
            (
                "Wie ist die ECG- und Markerqualität?",
                (
                    f"{failed_ecgs:,} aggregierte fehlgeschlagene ECG-Verarbeitungen"
                    if failure_path.is_file()
                    else "Feature-Extraktion noch nicht vollständig ausgewertet"
                ),
                "prüfen" if failed_ecgs else "kein aktueller Blocker",
                "Details zeigen Rauschen, Clipping, fehlende Wellen und Rechenfehler getrennt.",
            ),
            (
                "Wie nah liegen klinische Referenzen am ECG?",
                (
                    f"Medianer absoluter Abstand {median_delta:.1f} Stunden"
                    if median_delta is not None
                    else "Zeitabstände noch nicht verfügbar"
                ),
                "sichtbar" if median_delta is not None else "noch offen",
                "Jedes Ziel behält sein vorab definiertes Zeitfenster und die zeitliche Richtung.",
            ),
            (
                "Sind Train und Validation patientengetrennt?",
                (
                    "Ja, Split-Summen entsprechen allen Patienten"
                    if split_ok
                    else "Noch nicht nachweisbar oder inkonsistent"
                ),
                "bestanden" if split_ok else "Blocker",
                "Kein Patient darf gleichzeitig in Train und Validation vorkommen.",
            ),
            (
                "Gibt es wissenschaftliche Blocker?",
                "; ".join(blockers) if blockers else "Keine in dieser Zusammenfassung",
                "Blocker" if blockers else "bereit zur nächsten Stufe",
                "Ein Klick auf die Detailtabs zeigt die zugrunde liegenden Zahlen und Ursachen.",
            ),
        ]
        self._fill_table(self.foundation_overview, rows)
        for row in range(self.foundation_overview.rowCount()):
            detail = self.foundation_overview.item(row, 3)
            if detail is not None:
                for column in range(self.foundation_overview.columnCount()):
                    item = self.foundation_overview.item(row, column)
                    if item is not None:
                        item.setToolTip(detail.text())

    def _active_development_feature_root(self) -> Optional[Path]:
        active_states = (
            JobState.RUNNING,
            JobState.PAUSING,
            JobState.PAUSED,
            JobState.RESUMING,
        )
        try:
            records = self.job_store.list(states=active_states, limit=100)
        except Exception as exc:
            LOGGER.warning("Active feature-job lookup failed: %s", exc)
            records = ()
        for record in records:
            if record.spec.stage != "extract-features":
                continue
            arguments = dict(record.spec.arguments)
            if arguments.get("locked_test_only") or arguments.get(
                "include_locked_test"
            ):
                continue
            panel_key = "_".join(
                str(item)
                for item in arguments.get(
                    "panel_ids",
                    ("primary_core", "sentinel"),
                )
            )
            maximum = arguments.get("max_records")
            scope = (
                f"smoke-development-{int(maximum)}"
                if maximum not in (None, "")
                else "development"
            )
            root = self.config.features_root / panel_key / scope
            if root.is_dir():
                return root
        development_roots = sorted(
            (
                path.parent
                for path in self.config.features_root.glob(
                    "*/development/features.manifest.json"
                )
                if path.parent.is_dir()
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        if development_roots:
            return development_roots[0]
        smoke_roots = sorted(
            (
                path
                for path in self.config.features_root.glob(
                    "*/smoke-development-*"
                )
                if path.is_dir()
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        return smoke_roots[0] if smoke_roots else None

    def _live_feature_signature(self) -> tuple:
        root = self._active_development_feature_root()
        if root is None:
            return ("live-features", "", 0, 0, 0)
        parts = tuple(root.glob("split=*/part-*.parquet"))
        if not parts:
            return ("live-features", str(root), 0, 0, 0)
        return (
            "live-features",
            str(root),
            len(parts),
            max(path.stat().st_mtime_ns for path in parts),
            sum(path.stat().st_size for path in parts),
        )

    def _refresh_live_features(self) -> None:
        self.live_feature_plot.clear()
        root = self._active_development_feature_root()
        parts = (
            tuple(sorted(root.glob("split=*/part-*.parquet")))
            if root is not None
            else ()
        )
        # part-NNNNNN.parquet is zero-padded and sequential, so the tail
        # of the already-sorted tuple is the most recently written
        # partitions -- see LIVE_FEATURE_PREVIEW_PART_LIMIT above.
        parts = parts[-LIVE_FEATURE_PREVIEW_PART_LIMIT:]
        if not parts:
            self.live_feature_context.setText(
                "Snapshot: partial | Noch keine atomar abgeschlossene "
                "Development-Partition verfügbar."
            )
            self.live_feature_table.setRowCount(0)
            return
        marker = str(self.live_marker_combo.currentData() or "")
        quoted_marker = '"' + marker.replace('"', '""') + '"'
        try:
            import duckdb

            with duckdb.connect() as connection:
                status = connection.execute(
                    """
                    SELECT
                        split,
                        extraction_status,
                        COUNT(*) AS record_count
                    FROM read_parquet(?, union_by_name=true)
                    GROUP BY split, extraction_status
                    ORDER BY split, extraction_status
                    """,
                    [[str(path) for path in parts]],
                ).fetchdf()
                histogram = connection.execute(
                    f"""
                    WITH selected AS (
                        SELECT
                            split,
                            TRY_CAST({quoted_marker} AS DOUBLE) AS value
                        FROM read_parquet(?, union_by_name=true)
                        WHERE extraction_status = 'ok'
                    ),
                    raw AS (
                        SELECT * FROM selected WHERE ISFINITE(value)
                    ),
                    bounds AS (
                        SELECT
                            QUANTILE_CONT(value, 0.01) AS low,
                            QUANTILE_CONT(value, 0.99) AS high
                        FROM raw
                    ),
                    binned AS (
                        SELECT
                            split,
                            CASE
                                WHEN high <= low THEN 0
                                ELSE LEAST(
                                    29,
                                    GREATEST(
                                        0,
                                        FLOOR(
                                            (value - low) * 30.0
                                            / NULLIF(high - low, 0)
                                        )::INTEGER
                                    )
                                )
                            END AS bin_id,
                            value
                        FROM raw
                        CROSS JOIN bounds
                        WHERE value BETWEEN low AND high
                    )
                    SELECT
                        split,
                        bin_id,
                        AVG(value) AS bin_center,
                        COUNT(*) AS record_count
                    FROM binned
                    GROUP BY split, bin_id
                    ORDER BY split, bin_id
                    """,
                    [[str(path) for path in parts]],
                ).fetchdf()
        except Exception as exc:
            self.live_feature_context.setText(
                "Snapshot: partial | Live-Abfrage nicht verfügbar: "
                f"{type(exc).__name__}: {exc}"
            )
            self.live_feature_table.setRowCount(0)
            return

        split_totals = status.groupby("split")["record_count"].transform("sum")
        table_rows = [
            (
                row.split,
                row.extraction_status,
                int(row.record_count),
                (
                    float(row.record_count) / float(split_totals.iloc[index])
                    if split_totals.iloc[index] > 0
                    else np.nan
                ),
            )
            for index, row in enumerate(status.itertuples(index=False))
        ]
        self._fill_table(self.live_feature_table, table_rows)
        colors = {"train": "#087e8b", "validation": "#d1495b"}
        for split, frame in histogram.groupby("split", sort=True):
            frame = frame.sort_values("bin_id")
            counts = pd.to_numeric(
                frame["record_count"],
                errors="coerce",
            ).to_numpy(dtype=float)
            total = float(np.nansum(counts))
            if total <= 0:
                continue
            self.live_feature_plot.plot(
                pd.to_numeric(
                    frame["bin_center"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                counts / total,
                pen=pg.mkPen(colors.get(str(split), "#5f6b6d"), width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=colors.get(str(split), "#5f6b6d"),
                name=str(split),
            )
        self.live_feature_context.setText(
            f"Snapshot: partial | {len(parts):,} fertige Partitionen | "
            f"{int(status['record_count'].sum()):,} ECGs | Marker: {marker} | "
            "rein deskriptiv, kein Freeze und keine formale Endpunktmetrik."
        )
        self.live_feature_plot.setLabel("bottom", marker)

    def _feature_failure_path(self, run_id: str = "") -> Path:
        """Return only the failure report bound to ``run_id`` provenance.

        Feature roots are shared infrastructure and can contain snapshots from
        older scientific runs.  Falling back to the newest such snapshot made
        an empty current result page silently display historical QC.  Missing
        or incomplete provenance therefore means unavailable, never "latest".
        """

        if run_id:
            provenance = _read_json(
                self.config.runs_root
                / run_id
                / "development_provenance.json"
            )
            root_text = str(
                provenance.get("feature_integrity", {}).get("root", "")
            )
            if root_text:
                candidate = (
                    Path(root_text).resolve()
                    / "feature_failure_report.parquet"
                )
                if candidate.is_file():
                    return candidate
        return self.config.features_root / "__not_available__.parquet"

    def _update_qc_plot(self, label_path: Path) -> None:
        self.qc_plot.clear()
        self.assessability_plot.clear()
        if not label_path.is_file():
            return
        frame = pd.read_parquet(label_path)
        prevalence = _numeric_column(
            frame,
            ("prevalence", "label_prevalence"),
        )
        if prevalence is None:
            positive = _numeric_column(
                frame,
                ("positive_count", "n_positive"),
            )
            negative = _numeric_column(
                frame,
                ("negative_count", "n_negative"),
            )
            if positive is None or negative is None:
                return
            denominator = positive + negative
            prevalence = positive / denominator.replace(0, np.nan)
        label_column = (
            "label_id" if "label_id" in frame else "endpoint_id"
        )
        ranked = (
            frame.assign(_prevalence=prevalence)
            .sort_values("_prevalence", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        prevalence = ranked["_prevalence"]
        x = np.arange(len(prevalence))
        self.qc_plot.addItem(
            pg.BarGraphItem(
                x=x,
                height=prevalence.fillna(0).to_numpy(),
                width=0.75,
                brush=pg.mkBrush("#087e8b"),
                pen=pg.mkPen(None),
            )
        )
        _set_categorical_ticks(
            self.qc_plot,
            ranked[label_column].astype(str).tolist()
            if label_column in ranked
            else [str(index + 1) for index in x],
        )
        self._draw_assessability(ranked, label_column)

    def _draw_assessability(
        self,
        frame: pd.DataFrame,
        label_column: str,
    ) -> None:
        columns = (
            (
                ("positive_ecg_count", "negative_ecg_count"),
                "#087e8b",
                "beurteilbar",
            ),
            (("indeterminate_count",), "#f0a202", "indeterminate"),
            (("unavailable_count",), "#b8c4c8", "unavailable"),
            (("excluded_count",), "#d1495b", "ausgeschlossen"),
        )
        values = []
        for names, _color, _label in columns:
            total = np.zeros(len(frame), dtype=float)
            for name in names:
                if name in frame:
                    total += pd.to_numeric(
                        frame[name],
                        errors="coerce",
                    ).fillna(0).to_numpy()
            values.append(total)
        denominator = np.sum(values, axis=0)
        denominator[denominator <= 0] = np.nan
        bottom = np.zeros(len(frame), dtype=float)
        x = np.arange(len(frame))
        for raw, (_names, color, label) in zip(values, columns):
            fraction = np.nan_to_num(raw / denominator, nan=0.0)
            self.assessability_plot.addItem(
                pg.BarGraphItem(
                    x=x,
                    y0=bottom,
                    height=fraction,
                    width=0.75,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(None),
                    name=label,
                )
            )
            bottom += fraction
        self.assessability_plot.setYRange(0, 1, padding=0.01)
        _set_categorical_ticks(
            self.assessability_plot,
            frame[label_column].astype(str).tolist()
            if label_column in frame
            else [str(index + 1) for index in x],
        )

    def _update_window_plot(self, path: Path) -> None:
        self.window_plot.clear()
        if not path.is_file():
            return
        frame = pd.read_parquet(path)
        required = {"median_absolute_delta_hours", "label_id"}
        if frame.empty or not required.issubset(frame):
            return
        values = pd.to_numeric(
            frame["median_absolute_delta_hours"],
            errors="coerce",
        )
        frame = frame.loc[values.notna()].copy().head(120).reset_index(drop=True)
        values = pd.to_numeric(
            frame["median_absolute_delta_hours"],
            errors="coerce",
        ).to_numpy()
        if frame.empty:
            return
        split = _string_column(frame, "split", "unknown")
        colors = {
            "train": "#087e8b",
            "validation": "#d1495b",
        }
        for split_name in sorted(set(split)):
            indices = np.flatnonzero(split.to_numpy() == split_name)
            self.window_plot.addItem(
                pg.ScatterPlotItem(
                    x=indices,
                    y=values[indices],
                    size=7,
                    brush=pg.mkBrush(colors.get(split_name, "#5f6b6d")),
                    pen=pg.mkPen(None),
                    name=split_name,
                )
            )
        profile = _string_column(frame, "window_profile", "")
        labels = [
            f"{label} | {window}"
            for label, window in zip(frame["label_id"].astype(str), profile)
        ]
        _set_categorical_ticks(self.window_plot, labels)

    def _update_feature_failure_plot(self, path: Path) -> None:
        self.feature_failure_plot.clear()
        if not path.is_file():
            return
        frame = pd.read_parquet(path)
        if frame.empty or "record_count" not in frame:
            return
        frame = (
            frame.assign(
                _count=pd.to_numeric(
                    frame["record_count"],
                    errors="coerce",
                )
            )
            .dropna(subset=["_count"])
            .nlargest(20, "_count")
            .reset_index(drop=True)
        )
        x = np.arange(len(frame))
        status = _string_column(frame, "extraction_status", "error")
        brushes = [
            pg.mkBrush(
                "#f0a202"
                if item in ("signal_qc_failed", "lead_placement_excluded")
                else "#d1495b"
            )
            for item in status
        ]
        self.feature_failure_plot.addItem(
            pg.BarGraphItem(
                x=x,
                height=frame["_count"].to_numpy(),
                width=0.75,
                brushes=brushes,
                pen=pg.mkPen(None),
            )
        )
        error_type = _string_column(frame, "error_type", "unknown")
        _set_categorical_ticks(self.feature_failure_plot, error_type.tolist())

    def _update_qc_table(self, split_path: Path, label_path: Path) -> None:
        checks = []
        if split_path.is_file():
            split = pd.read_parquet(split_path)
            checks.append(
                (
                    "Patientensplit vorhanden",
                    "OK" if not split.empty else "WARNUNG",
                    f"{len(split):,} Zusammenfassungszeilen",
                )
            )
        else:
            checks.append(("Patientensplit vorhanden", "OFFEN", "Keine Splitübersicht"))
        if label_path.is_file():
            labels = pd.read_parquet(label_path)
            checks.append(
                (
                    "Labelübersicht",
                    "OK" if not labels.empty else "WARNUNG",
                    f"{len(labels):,} Endpunkt-/Split-Zeilen",
                )
            )
        else:
            checks.append(("Labelübersicht", "OFFEN", "Noch nicht erzeugt"))
        locked_reports = list(
            self.config.runs_root.glob("*/final_test_report.parquet")
        )
        checks.append(
            (
                "Locked Test",
                "VERSIEGELT" if not locked_reports else "PRÜFEN",
                (
                    "Keine Testauswertung vorhanden"
                    if not locked_reports
                    else "Final-Test-Artefakt vorhanden"
                ),
            )
        )
        self._fill_table(self.qc_table, checks)

    def _refresh_run_choices(self) -> None:
        current_results = self.results_run_combo.currentData()
        current_reports = self.report_run_combo.currentData()
        current_run_id = self._current_run_id()
        artifact_runs = sorted(
            {
                path.parent.name
                for path in self.config.runs_root.glob(
                    "*/screening_results.parquet"
                )
            }
            | {
                path.parent.parent.name
                for path in self.config.runs_root.glob(
                    "*/reports/review_candidates.parquet"
                )
            }
        )
        descriptors = tuple(
            descriptor
            for descriptor in self.result_repository.available_runs(
                on_run_processed=QApplication.processEvents
            )
            if _result_run_visible_for_current(descriptor, current_run_id)
        )
        artifact_runs = [
            run_id
            for run_id in artifact_runs
            if run_id == current_run_id
            and any(item.run_id == run_id for item in descriptors)
        ]
        descriptor_ids = {item.run_id for item in descriptors}
        self.results_run_combo.blockSignals(True)
        self.results_run_combo.clear()
        for descriptor in descriptors:
            if descriptor.complete_compatible:
                prefix = "Vollständig"
            else:
                prefix = "Aktueller Pilot / unvollständig"
            self.results_run_combo.addItem(
                f"{prefix} · {descriptor.run_id}",
                descriptor.run_id,
            )
            self.results_run_combo.setItemData(
                self.results_run_combo.count() - 1,
                (
                    f"Validation: {descriptor.validation_status}; "
                    f"Vollständigkeit: {descriptor.analysis_completeness}; "
                    f"Version: {descriptor.validation_stage_version}"
                ),
                Qt.ItemDataRole.ToolTipRole,
            )
        result_index = self.results_run_combo.findData(current_results)
        if result_index < 0:
            default = self.result_repository.default_run(descriptors=descriptors)
            result_index = (
                self.results_run_combo.findData(default.run_id)
                if default is not None
                else 0
            )
        if self.results_run_combo.count():
            self.results_run_combo.setCurrentIndex(max(0, result_index))
        self.results_run_combo.blockSignals(False)
        self.atlas_run_combo.blockSignals(True)
        self.atlas_run_combo.clear()
        for index in range(self.results_run_combo.count()):
            self.atlas_run_combo.addItem(
                self.results_run_combo.itemText(index),
                self.results_run_combo.itemData(index),
            )
            self.atlas_run_combo.setItemData(
                index,
                self.results_run_combo.itemData(
                    index, Qt.ItemDataRole.ToolTipRole
                ),
                Qt.ItemDataRole.ToolTipRole,
            )
        if self.results_run_combo.currentIndex() >= 0:
            self.atlas_run_combo.setCurrentIndex(
                self.results_run_combo.currentIndex()
            )
        self.atlas_run_combo.blockSignals(False)

        report_runs = sorted(set(artifact_runs) | descriptor_ids)
        descriptor_by_id = {item.run_id: item for item in descriptors}
        self.report_run_combo.blockSignals(True)
        self.report_run_combo.clear()
        for run_id in report_runs:
            descriptor = descriptor_by_id.get(run_id)
            if descriptor is not None and descriptor.complete_compatible:
                prefix = "Vollständig"
            else:
                prefix = "Aktueller Pilot / unvollständig"
            self.report_run_combo.addItem(
                f"{prefix} · {run_id}",
                run_id,
            )
        report_index = self.report_run_combo.findData(current_reports)
        if report_index < 0:
            report_index = self.report_run_combo.findData(
                self.results_run_combo.currentData()
            )
        if report_index >= 0:
            self.report_run_combo.setCurrentIndex(report_index)
        self.report_run_combo.blockSignals(False)

    def _current_result_perspective(self) -> str:
        perspectives = tuple(self.result_perspective_tables)
        index = self.result_perspective_tabs.currentIndex()
        return perspectives[max(0, min(index, len(perspectives) - 1))]

    def _focus_discovery_perspective(self) -> None:
        widget = getattr(self, "_discovery_perspective_tab_widget", None)
        if widget is not None:
            self.result_perspective_tabs.setCurrentWidget(widget)

    def _refresh_discovery_summary_banner(self, run_root: Path) -> None:
        path = run_root / "phenome" / "new_associations_explorer.parquet"
        if not path.is_file():
            self.discovery_summary_banner.setText(
                "Noch keine Entdeckungs-Zusammenfassung verfügbar "
                "(breiter Scan für diesen Lauf noch nicht abgeschlossen)."
            )
            return
        try:
            import duckdb

            with duckdb.connect() as connection:
                counts = connection.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE status_id IN (
                                'replicated', 'replicated_association'
                            )
                        ) AS total,
                        COUNT(*) FILTER (
                            WHERE status_id IN (
                                'replicated', 'replicated_association'
                            ) AND cardiac_scope_id = 'cardiovascular'
                        ) AS cardiac,
                        COUNT(*) FILTER (
                            WHERE status_id IN (
                                'replicated', 'replicated_association'
                            ) AND cardiac_scope_id = 'non_cardiovascular'
                        ) AS non_cardiac,
                        COUNT(*) FILTER (
                            WHERE status_id IN (
                                'replicated', 'replicated_association'
                            )
                            AND discovery_class = 'future_outcome_hypothesis'
                        ) AS prognostic
                    FROM read_parquet(?)
                    """,
                    [str(path)],
                ).fetchdf()
        except Exception as exc:
            self.discovery_summary_banner.setText(
                f"Entdeckungs-Zusammenfassung nicht lesbar: {exc}"
            )
            return
        row = counts.iloc[0]
        total = int(row["total"])
        if total == 0:
            self.discovery_summary_banner.setText(
                "Bisher keine replizierten neuen Zusammenhänge in diesem "
                "Ergebnisstand."
            )
            return
        self.discovery_summary_banner.setText(
            f"{total:,} replizierte neue Zusammenhänge im aktuellen "
            f"Ergebnisstand — {int(row['cardiac']):,} kardial, "
            f"{int(row['non_cardiac']):,} nicht-kardial, "
            f"{int(row['prognostic']):,} prognostisch. Klicken für Details."
        )

    def _broad_axis_where(self) -> tuple[str, tuple]:
        clauses: list[str] = []
        parameters: list[object] = []
        axis = str(self.broad_axis_combo.currentData() or "")
        if axis:
            clauses.append("axis = ?")
            parameters.append(axis)
        relation = str(self.broad_axis_relation_combo.currentData() or "")
        if relation:
            clauses.append("temporal_relation = ?")
            parameters.append(relation)
        return " AND ".join(clauses), tuple(parameters)

    def _draw_broad_axis_plots(self, path: Path) -> None:
        self.broad_continuous_plot.clear()
        self.broad_binary_plot.clear()
        if not path.is_file():
            return
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - visible table remains usable
            LOGGER.warning("Broad-axis plot artifact is unreadable: %s", exc)
            return
        axis = str(self.broad_axis_combo.currentData() or "")
        relation = str(self.broad_axis_relation_combo.currentData() or "")
        if axis and "axis" in frame:
            frame = frame.loc[frame["axis"].astype(str).eq(axis)]
        if relation and "temporal_relation" in frame:
            frame = frame.loc[
                frame["temporal_relation"].astype(str).eq(relation)
            ]
        if frame.empty or not {"target_kind", "effect", "q_value"} <= set(frame):
            return
        frame = frame.copy()
        frame["effect"] = pd.to_numeric(frame["effect"], errors="coerce")
        frame["plot_q"] = pd.to_numeric(frame["q_value"], errors="coerce")
        frame = frame.loc[
            np.isfinite(frame["effect"])
            & np.isfinite(frame["plot_q"])
            & frame["plot_q"].gt(0)
        ]
        if frame.empty:
            return
        frame["plot_y"] = -np.log10(frame["plot_q"].clip(lower=1e-300))
        confirmed = frame.get(
            "adjusted_confirmed", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        frame["plot_confirmed"] = confirmed
        for kind, plot, neutral in (
            ("continuous", self.broad_continuous_plot, 0.0),
            ("binary", self.broad_binary_plot, 0.5),
        ):
            scoped = frame.loc[frame["target_kind"].astype(str).eq(kind)].copy()
            if scoped.empty:
                continue
            scoped["plot_x"] = scoped["effect"] - neutral
            if len(scoped) > self._SCATTER_POINT_LIMIT:
                scoped = scoped.nlargest(self._SCATTER_POINT_LIMIT, "plot_y")
            for flag, colour, label in (
                (False, "#d58b2a99", "repliziert, Adjustierung offen"),
                (True, "#1b9e7799", "adjustiert bestätigt"),
            ):
                block = scoped.loc[scoped["plot_confirmed"].eq(flag)]
                if block.empty:
                    continue
                plot.plot(
                    block["plot_x"].to_numpy(),
                    block["plot_y"].to_numpy(),
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(colour),
                    symbolPen=None,
                    name=f"{label} (n={len(block):,})",
                )
            plot.addItem(
                pg.InfiniteLine(
                    pos=0.0,
                    angle=90,
                    pen=pg.mkPen("#909090", width=1, style=Qt.PenStyle.DashLine),
                )
            )

    def _refresh_broad_axes(self, run_root: Path) -> None:
        """Bind the broad-axis tab to the newest results of this run."""

        directory = run_root / "broad_axes"
        hits = directory / "broad_axis_replicated_hits.parquet"
        where, parameters = self._broad_axis_where()
        self.broad_axis_table.set_source(
            hits if hits.is_file() else None,
            where_sql=where,
            parameters=parameters,
        )
        self._draw_broad_axis_plots(hits)
        coverage = directory / "broad_axis_coverage.parquet"
        self.broad_axis_coverage_table.set_source(
            coverage if coverage.is_file() else None
        )
        summary = _read_json(directory / "broad_axis_summary.json")
        if not summary:
            self.broad_axis_summary.setText(
                "Noch keine Breitachsen-Ergebnisse. Die Stufe läuft am Ende "
                "des Laufs."
            )
            return
        eligible = int(summary.get("n_targets_eligible", 0) or 0)
        total = int(summary.get("n_targets_total", 0) or 0)
        tests = int(summary.get("n_tests", 0) or 0)
        replicated = int(summary.get("n_replicated_hits", 0) or 0)
        adjusted = int(summary.get("n_adjusted_confirmed_hits", 0) or 0)
        skipped = summary.get("axes_skipped") or []
        lines = [
            "{} von {} Zielen über der Häufigkeitsschwelle, "
            "{:,} Tests, {:,} replizierte und {:,} adjustiert bestätigte "
            "Treffer.".format(
                format(eligible, ",d"),
                format(total, ",d"),
                tests,
                replicated,
                adjusted,
            ),
            "Achsen: {} | FDR-Familie: {}".format(
                ", ".join(summary.get("axes_screened", [])) or "keine",
                summary.get("fdr_family_definition", "?"),
            ),
        ]
        if skipped:
            lines.append(
                "Uebersprungen: "
                + ", ".join(
                    "{} ({}: {})".format(
                        item.get("axis"),
                        item.get("status", "unbekannt"),
                        item.get("message", ""),
                    )
                    for item in skipped
                )
            )
        self.broad_axis_summary.setText("\n".join(lines))

    def _refresh_promotion_candidates(self, run_root: Path) -> None:
        """Surfaces phenome_promotions.py's curated, deduplicated,
        candidate-biomarker-only hypothesis list. Deliberately a separate
        panel/table from the raw 'Breiter Scan' table above (which shows
        every FDR-replicated hit, including QC/classical-baseline rows) --
        this list is the specific 'publishable candidate' output the
        promotion pipeline exists to produce, and it was previously never
        surfaced anywhere in the GUI."""
        candidates_path = (
            run_root / "phenome" / "phenome_promotion_candidates.parquet"
        )
        summary_path = run_root / "phenome" / "promotion_candidates.json"
        payload = _read_json(summary_path)
        candidate_count = len(payload.get("candidates", []) or [])
        refresh_manifest = _read_json(
            run_root / "phenome" / "promotion_refresh.manifest.json"
        )
        refreshed_at = str(refresh_manifest.get("created_at", "") or "")
        if not candidates_path.is_file():
            self.promotion_candidates_summary.setText(
                "Noch keine kuratierte Hypothesenliste für diesen Lauf "
                "(refresh-phenome-promotions noch nicht ausgeführt)."
            )
            self.promotion_candidates_table.set_source(None)
            return
        self.promotion_candidates_summary.setText(
            f"{candidate_count:,} kuratierte Kandidaten"
            + (f" · zuletzt aktualisiert {refreshed_at[:19]} UTC" if refreshed_at else "")
            + " · status=draft_hypotheses_only, keine automatische Übernahme."
        )
        self.promotion_candidates_table.set_source(candidates_path)

    def _result_perspective_changed(self, _index: int = 0) -> None:
        perspective = self._current_result_perspective()
        self._primary_result_offsets[perspective] = 0
        self._configure_result_sort_options(perspective)
        self._refresh_result_repository()

    def _configure_result_sort_options(self, perspective: str) -> None:
        current = str(self.result_sort_combo.currentData() or "")
        if perspective == "discovery":
            options = (
                ("Beste Nachverfolgungspriorität", "evidence_rank"),
                ("Stärkster Zusammenhang", "association_strength"),
                ("Höchste statistische Sicherheit", "statistical_strength"),
                ("Kleinster Validation-q-Wert", "validation_q_value"),
                ("Name des möglichen Zusammenhangs", "target_name"),
            )
        else:
            options = (
                ("Beste Evidenz", "evidence_rank"),
                ("Höchste ROC-AUC", "roc_auc"),
                ("Höchste Sensitivität", "sensitivity_at_95_specificity"),
                ("Höchste Spezifität", "operating_specificity"),
                ("Name", "endpoint_name_de"),
            )
        allowed = {value for _label, value in options}
        selected = current if current in allowed else options[0][1]
        self.result_sort_combo.blockSignals(True)
        self.result_sort_combo.clear()
        for label, value in options:
            self.result_sort_combo.addItem(label, value)
        index = self.result_sort_combo.findData(selected)
        self.result_sort_combo.setCurrentIndex(max(0, index))
        self.result_sort_combo.blockSignals(False)

    def _discovery_filter_changed(self, *_args) -> None:
        if not hasattr(self, "_primary_result_offsets"):
            return
        self._primary_result_offsets["discovery"] = 0
        if self._current_result_perspective() == "discovery":
            self._refresh_result_repository()

    def _primary_result_query(self) -> ResultQuery:
        perspective = self._current_result_perspective()
        endpoint_id = (
            str(self.endpoint_filter.currentData() or "")
            if perspective != "discovery"
            and self.endpoint_filter.currentIndex() >= 0
            else ""
        )
        typed_pathology = (
            self.endpoint_filter.currentText().strip()
            if perspective != "discovery"
            and self.endpoint_filter.currentIndex() < 0
            else ""
        )
        search_parts = [
            self.result_search.text().strip(),
            typed_pathology,
        ]
        search = " ".join(item for item in search_parts if item)
        evidence = (
            str(self.discovery_status_combo.currentData() or "")
            if perspective == "discovery"
            else str(self.result_evidence_filter.currentData() or "")
        )
        discovery_scope = (
            str(self.discovery_scope_combo.currentData() or "")
            if perspective == "discovery"
            else ""
        )
        cardiac_scope = (
            str(self.discovery_cardiac_combo.currentData() or "")
            if perspective == "discovery"
            else ""
        )
        panel = str(self.panel_filter.currentData() or "")
        sort_by = str(self.result_sort_combo.currentData() or "evidence_rank")
        if sort_by == "endpoint_name_de" and perspective == "biomarkers":
            sort_by = "feature_name_de"
        elif sort_by == "endpoint_name_de" and perspective == "discovery":
            sort_by = "target_name"
        run_id = self._selected_current_run_id(self.results_run_combo)
        descriptor = next(
            (
                item
                for item in self.result_repository.available_runs(
                    on_run_processed=QApplication.processEvents
                )
                if item.run_id == run_id
            ),
            None,
        )
        explicitly_selected_preview = bool(
            descriptor is not None and not descriptor.complete_compatible
        )
        return ResultQuery(
            perspective=perspective,
            run_id=run_id,
            endpoint_ids=(endpoint_id,) if endpoint_id else (),
            search_text=search,
            panel_ids=(panel,) if panel else (),
            feature_families=(
                ("classical_ecg_rule",)
                if self.classical_only_filter.isChecked()
                else ()
            ),
            candidate_types=(discovery_scope,) if discovery_scope else (),
            baseline_ids=(cardiac_scope,) if cardiac_scope else (),
            evidence_grades=(evidence,) if evidence else (),
            minimum_sensitivity=(
                self.minimum_sensitivity_filter.value() / 100.0
                if perspective != "discovery"
                and self.minimum_sensitivity_filter.value() > 0
                else None
            ),
            minimum_specificity=(
                self.minimum_specificity_filter.value() / 100.0
                if perspective != "discovery"
                and self.minimum_specificity_filter.value() > 0
                else None
            ),
            validation_only=True,
            # Historical result views are intentionally unavailable in the
            # operational GUI. Provenance remains on disk, but can never be
            # adopted by an empty/current selection.
            include_historical=False,
            complete_runs_only=not explicitly_selected_preview,
            sort_by=sort_by,
            sort_descending=sort_by not in {
                "endpoint_name_de",
                "feature_name_de",
                "target_name",
                "validation_q_value",
            },
            offset=self._primary_result_offsets[perspective],
            limit=200,
        )

    def _refresh_result_repository(self, *_args) -> None:
        if not hasattr(self, "result_perspective_tables"):
            return
        query = self._primary_result_query()
        perspective = query.perspective
        if not query.run_id:
            self._set_primary_result_frame(
                self.result_perspective_tables[perspective],
                pd.DataFrame(),
                perspective,
            )
            self._primary_result_totals[perspective] = 0
            self.result_page_label.setText("0-0 von 0")
            self.result_previous_button.setEnabled(False)
            self.result_next_button.setEnabled(False)
            self.result_repository_status.setText(
                "Kein aktueller, kompatibler Ergebnislauf ausgewählt. "
                "Historische Runs werden nicht als Ersatz geladen."
            )
            self._update_active_result_filters(query)
            return
        try:
            page = self.result_repository.query(query)
        except Exception as exc:
            self._set_primary_result_frame(
                self.result_perspective_tables[perspective],
                pd.DataFrame(),
                perspective,
            )
            self.result_repository_status.setText(
                f"Ergebnisabfrage fehlgeschlagen: {type(exc).__name__}: {exc}"
            )
            return
        self._primary_result_totals[perspective] = page.total_count
        self._set_primary_result_frame(
            self.result_perspective_tables[perspective],
            page.rows,
            perspective,
        )
        first = min(page.total_count, page.offset + 1) if page.total_count else 0
        last = min(page.total_count, page.offset + len(page.rows))
        self.result_page_label.setText(
            f"{first:,}–{last:,} von {page.total_count:,}"
        )
        self.result_previous_button.setEnabled(page.offset > 0)
        self.result_next_button.setEnabled(
            page.offset + page.limit < page.total_count
        )
        if page.status in {"ok", "empty"}:
            unsupported = (
                " Nicht anwendbare Filter: "
                + ", ".join(page.unsupported_filters)
                if page.unsupported_filters
                else ""
            )
            self.result_repository_status.setText(
                (
                    f"Run {page.run_id} · {page.analysis_completeness} · "
                    f"Validation · Snapshot {page.snapshot_hash[:12]}"
                    if page.run_id
                    else page.message_de
                )
                + unsupported
            )
        else:
            self.result_repository_status.setText(page.message_de)
        self._update_active_result_filters(query)

    def _set_primary_result_frame(
        self,
        table: QTableWidget,
        frame: pd.DataFrame,
        perspective: str,
    ) -> None:
        definitions = {
            "pathologies": (
                ("endpoint_name_de", "Pathologie / klinisches Ziel"),
                ("pathology_group", "Verwandte Gruppe"),
                ("feature_name_de", "Biomarker"),
                ("candidate_type", "Typ"),
                ("roc_auc", "ROC-AUC"),
                ("sensitivity_at_95_specificity_reported", "Sensitivität am Trainingsschwellenwert (95-%-Ziel)"),
                ("operating_specificity", "Beobachtete Validierungsspezifität"),
                ("operating_point_validation_status", "Validierungsziel / unbekannt"),
                ("operating_point_training_status", "Trainingsziel / Schwellenprovenienz"),
                ("selection_eligible", "Zur Auswahl zugelassen"),
                ("q_value", "FDR q"),
                ("projection_plane", "2D-Ebene"),
                ("projection_hypothesis_q_value", "2D-Hypothese FDR q"),
                ("projection_hypothesis_restriction_operationalized", "2D-Restriktionsstatus"),
                ("evidence_grade_de", "Evidenz"),
                ("delta_auc", "Zusatznutzen ΔAUC"),
            ),
            "biomarkers": (
                ("feature_name_de", "Biomarker"),
                ("conceptual_family", "Markerfamilie"),
                ("endpoint_name_de", "Pathologie / klinisches Ziel"),
                ("roc_auc", "ROC-AUC"),
                ("sensitivity_at_95_specificity_reported", "Sensitivität am Trainingsschwellenwert (95-%-Ziel)"),
                ("operating_specificity", "Beobachtete Validierungsspezifität"),
                ("operating_point_validation_status", "Validierungsziel / unbekannt"),
                ("operating_point_training_status", "Trainingsziel / Schwellenprovenienz"),
                ("selection_eligible", "Zur Auswahl zugelassen"),
                ("q_value", "FDR q"),
                ("projection_plane", "2D-Ebene"),
                ("projection_hypothesis_q_value", "2D-Hypothese FDR q"),
                ("projection_hypothesis_restriction_operationalized", "2D-Restriktionsstatus"),
                ("evidence_grade_de", "Evidenz"),
                ("clinical_interpretation", "Einordnung"),
            ),
            "criteria": (
                ("display_name_de", "Kriterienkandidat"),
                ("stage_de", "Stand"),
                ("formula", "Exakte Formel"),
                ("specificity_95_threshold", "Grenzwert (95 % Spezifität)"),
                ("specificity_95_gray_zone", "Grauzone"),
                ("sensitivity", "Sensitivität"),
                ("specificity", "Spezifität"),
                ("roc_auc", "ROC-AUC"),
                ("evidence_grade_de", "Evidenz"),
                ("unit", "Einheit"),
            ),
            "discovery": (
                ("target_name", "Möglicher Zusammenhang"),
                ("scope_display_de", "Bereich"),
                ("discovery_domain", "Organsystem / Domäne"),
                ("cardiac_scope_de", "Pathologieart"),
                ("feature_name_de", "Biomarker"),
                ("status_de", "Prüfstatus"),
                ("association_strength", "Zusammenhangsstärke (0–1)"),
                ("validation_q_value", "Validation FDR q"),
                ("n_positive", "Fälle"),
                ("n_negative", "Kontrollen"),
                ("followup_priority_de", "Nachverfolgung"),
                ("likely_confounded", "Confounding-Hinweis"),
                ("adjusted_or_per_sd", "Adjustierte OR je SD"),
                ("adjusted_q_value", "Adjustierte FDR q"),
                ("interpretation_de", "Einordnung"),
            ),
        }[perspective]
        available = [
            item for item in definitions
            if item[0] in frame.columns
            and (
                not item[0].startswith("projection_")
                or frame[item[0]].notna().any()
            )
        ]
        if not available:
            available = [("message", "Ergebnis")]
            display = pd.DataFrame(
                {"message": ["Keine Ergebnisse für diese Auswahl."]}
            ) if frame.empty else frame.iloc[:, :1].copy()
        else:
            display = frame[[column for column, _ in available]].copy()
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(available))
        table.setHorizontalHeaderLabels([label for _, label in available])
        table.setRowCount(len(display))
        records = frame.to_dict(orient="records") if not frame.empty else []
        for display_row, (_source_index, row) in enumerate(display.iterrows()):
            source_record = (
                records[display_row]
                if display_row < len(records)
                else {"message": row.iloc[0]}
            )
            for column_index, (column, _label) in enumerate(available):
                value = row.get(column)
                item = _SortableTableItem(value)
                if column_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, source_record)
                if column == "feature_name_de":
                    item.setToolTip(
                        str(source_record.get("feature_tooltip_de", ""))
                    )
                elif column == "formula":
                    item.setToolTip(
                        "Train-fixierte, reproduzierbare Formel; fehlende "
                        "Komponenten machen Kombinationen nicht auswertbar."
                    )
                elif column == "projection_hypothesis_q_value":
                    item.setToolTip(
                        "BH-FDR nur innerhalb der vorab eingefrorenen 2D-Markerfamilie "
                        "dieses Endpunkts und Splits; ersetzt weder globales FDR noch Replikation."
                    )
                elif column == "projection_hypothesis_restriction_operationalized":
                    excluded = _nonnegative_integer_count(source_record.get(
                        "n_analysis_restriction_excluded"
                    ))
                    raw_observed = _nonnegative_integer_count(source_record.get("n_marker_raw_observed"))
                    count_note = (
                        f" Ausgeschlossen: {int(excluded):,} von "
                        f"{int(raw_observed):,} zunächst messbaren Fällen."
                        if excluded is not None and raw_observed is not None and excluded <= raw_observed
                        else " Ausschlusszahlen fehlen oder sind ungültig; nicht als null interpretieren."
                    )
                    item.setToolTip(
                        "Ein formaler 2D-Hypothesenwert ist nur zulässig, wenn die vorab "
                        "definierte Qualitäts-/Rhythmusrestriktion technisch umgesetzt ist."
                        + count_note
                    )
                table.setItem(display_row, column_index, item)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        if table.columnCount():
            table.horizontalHeader().setSectionResizeMode(
                0,
                QHeaderView.ResizeMode.Stretch,
            )
        table.setSortingEnabled(True)

    def _update_active_result_filters(self, query: ResultQuery) -> None:
        labels = []
        if query.endpoint_ids:
            labels.append(self.endpoint_filter.currentText())
        if query.search_text:
            labels.append(f"Suche: {query.search_text}")
        if query.panel_ids:
            labels.append(self.panel_filter.currentText())
        if query.feature_families:
            labels.append("Nur Leitlinienkriterien")
        if query.evidence_grades:
            labels.append(
                self.discovery_status_combo.currentText()
                if query.perspective == "discovery"
                else self.result_evidence_filter.currentText()
            )
        if query.candidate_types:
            labels.append(self.discovery_scope_combo.currentText())
        if query.baseline_ids and query.perspective == "discovery":
            labels.append(self.discovery_cardiac_combo.currentText())
        if query.minimum_sensitivity is not None:
            labels.append(f"Sensitivität ≥ {query.minimum_sensitivity:.0%}")
        if query.minimum_specificity is not None:
            labels.append(f"Spezifität ≥ {query.minimum_specificity:.0%}")
        self.active_result_filters.setText(
            "Aktive Filter: " + " · ".join(labels)
            if labels
            else "Keine zusätzlichen Filter aktiv"
        )

    def _reset_primary_result_filters(self) -> None:
        self.result_search.clear()
        self.endpoint_filter.setCurrentIndex(0)
        self.panel_filter.setCurrentIndex(0)
        self.classical_only_filter.setChecked(False)
        self.result_evidence_filter.setCurrentIndex(0)
        self.minimum_sensitivity_filter.setValue(0)
        self.minimum_specificity_filter.setValue(0)
        self.discovery_scope_combo.setCurrentIndex(0)
        self.discovery_cardiac_combo.setCurrentIndex(0)
        self.discovery_status_combo.setCurrentIndex(0)
        self.result_sort_combo.setCurrentIndex(0)
        for perspective in self._primary_result_offsets:
            self._primary_result_offsets[perspective] = 0
        self._refresh_result_repository()

    def _previous_primary_result_page(self) -> None:
        perspective = self._current_result_perspective()
        self._primary_result_offsets[perspective] = max(
            0,
            self._primary_result_offsets[perspective] - 200,
        )
        self._refresh_result_repository()

    def _next_primary_result_page(self) -> None:
        perspective = self._current_result_perspective()
        next_offset = self._primary_result_offsets[perspective] + 200
        if next_offset < self._primary_result_totals[perspective]:
            self._primary_result_offsets[perspective] = next_offset
        self._refresh_result_repository()

    def _open_primary_result_detail(self, perspective: str) -> None:
        table = self.result_perspective_tables[perspective]
        row = table.currentRow()
        if row < 0 or table.item(row, 0) is None:
            return
        payload = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        endpoint_id = str(payload.get("endpoint_id", ""))
        if endpoint_id:
            index = self.endpoint_filter.findData(endpoint_id)
            if index >= 0:
                self.endpoint_filter.setCurrentIndex(index)
        self.advanced_results_box.setChecked(True)
        self.result_view_combo.setCurrentIndex(0)

    def refresh_results(self) -> None:
        run_id = self._selected_current_run_id(self.results_run_combo)
        run_root = self.config.runs_root / run_id if run_id else None
        score_manifests = tuple(sorted(
            (run_root / "score_discovery").glob("*/score_discovery.manifest.json")
        )) if run_root is not None else ()
        revision = _artifact_revision((
            run_root / "validate.manifest.json",
            run_root / "survival" / "time_to_event.manifest.json",
            run_root / "therapy_response" / "therapy_response.manifest.json",
            run_root / "availability" / "segment_availability.manifest.json",
            run_root / "phenome" / "phenome.manifest.json",
            run_root / "broad_axes" / "broad_axes.manifest.json",
            self.config.external_validation_root / run_id / "population_cohorts" / "population_cohorts.manifest.json",
            *score_manifests,
        )) if run_root is not None else 0
        self.results_atlas.set_repository(self.result_repository)
        self.results_atlas.refresh(
            run_id,
            str(self.endpoint_filter.currentData() or ""),
            revision=revision,
        )
        if (self.results_mode_combo.currentData() != "detailed"
                or self.results_atlas_tabs.currentIndex() != 1):
            return
        self._refresh_result_repository()
        if not run_id:
            self.research_results_view.set_run_root(None)
            self.discovery_summary_banner.setText(
                "Noch keine Entdeckungs-Zusammenfassung verfügbar."
            )
            self.navigator_summary.setText(
                "Noch keine auswertbaren Validation-Ergebnisse."
            )
            self.result_context.setText("Noch kein Train-/Validation-Run verfügbar.")
            for table in (
                self.screening_table,
                self.curve_table,
                self.distribution_table,
                self.operating_table,
                self.baseline_table,
                self.stability_table,
                self.sensitivity_table,
                self.adjusted_model_table,
                self.decision_curve_table,
                self.phenome_table,
                self.promotion_candidates_table,
                self.transfer_table,
                self.group_vcg_audit_table,
            ):
                table.set_source(None)
            self.promotion_candidates_summary.setText("")
            self.group_vcg_build_button.setEnabled(False)
            self.group_vcg_required_button.setEnabled(False)
            self.group_vcg_context.setText(
                "Noch kein vollständiger Train-/Validation-Run verfügbar."
            )
            self._clear_group_vcg_view()
            self._clear_group_vcg_projections()
            return
        run_root = self.config.runs_root / run_id
        self._refresh_discovery_summary_banner(run_root)
        self.research_results_view.set_run_root(run_root)
        screening_path = run_root / "screening_results.parquet"
        where, parameters = self._result_filter()
        curve_path = run_root / "performance_curve_points.parquet"
        self.curve_table.set_source(
            curve_path if curve_path.is_file() else None,
            where_sql=where,
            parameters=parameters,
        )
        distribution_path = run_root / "score_distribution_bins.parquet"
        self.distribution_table.set_source(
            distribution_path if distribution_path.is_file() else None,
            where_sql=where,
            parameters=parameters,
        )
        operating_path = run_root / "operating_point_results.parquet"
        self.operating_table.set_source(
            operating_path if operating_path.is_file() else None,
            where_sql=where,
            parameters=parameters,
        )
        comparison_path = run_root / "vcg_vs_classical_comparison.parquet"
        comparison_where, comparison_parameters = self._endpoint_where()
        baseline_where, baseline_parameters = self._baseline_comparison_where()
        self.baseline_table.set_source(
            comparison_path if comparison_path.is_file() else None,
            where_sql=baseline_where,
            parameters=baseline_parameters,
        )
        stability_path = run_root / "overfitting_stability_report.parquet"
        self.stability_table.set_source(
            stability_path if stability_path.is_file() else None,
            where_sql=comparison_where,
            parameters=comparison_parameters,
        )
        sensitivity_where = "analysis_type <> 'single_marker'"
        sensitivity_parameters: tuple = ()
        if comparison_where:
            sensitivity_where = (
                f"{comparison_where} AND {sensitivity_where}"
            )
            sensitivity_parameters = comparison_parameters
        self.sensitivity_table.set_source(
            stability_path if stability_path.is_file() else None,
            where_sql=sensitivity_where,
            parameters=sensitivity_parameters,
        )
        model_where, model_parameters = self._endpoint_split_where()
        adjusted_path = run_root / "adjusted_model_results.parquet"
        self.adjusted_model_table.set_source(
            adjusted_path if adjusted_path.is_file() else None,
            where_sql=model_where,
            parameters=model_parameters,
        )
        decision_path = run_root / "decision_curve_results.parquet"
        self.decision_curve_table.set_source(
            decision_path if decision_path.is_file() else None,
            where_sql=comparison_where,
            parameters=comparison_parameters,
        )
        self._refresh_targeted_designs(run_root)
        phenome_path = _run_local_phenome_path(run_root)
        phenome_where, phenome_parameters = (
            self._phenome_where()
            if phenome_path.name
            in {
                "broad_discovery_replicated_hits.parquet",
                "broad_discovery_ranked.parquet",
            }
            else ("", ())
        )
        self.phenome_table.set_source(
            phenome_path if phenome_path.is_file() else None,
            where_sql=phenome_where,
            parameters=phenome_parameters,
        )
        self._refresh_promotion_candidates(run_root)
        self._refresh_broad_axes(run_root)
        transfer_path = _run_local_transfer_path(run_root)
        self.transfer_table.set_source(
            transfer_path if transfer_path.is_file() else None
        )

        current_formal_validation = _formal_development_validation_complete(
            run_root,
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        )
        completeness_text, freeze_eligible = _validation_presentation(
            run_root,
            expected_feature_code_hash=self._current_feature_code_hash,
            expected_validation_code_hash=self._current_validation_code_hash,
        )
        freeze = _read_json(run_root / "freeze_manifest.json")
        acceptance = _read_json(
            run_root
            / (
                "acceptance_audit.json"
                if freeze
                else "acceptance_pre_freeze_audit.json"
            )
        )
        acceptance_status = str(
            acceptance.get("status", "Audit offen")
        ).replace("_", " ")
        acceptance_ready = (
            current_formal_validation
            and acceptance.get("status") == "ready_to_freeze"
        )
        freeze_text = (
            f"Freeze {str(freeze.get('freeze_hash'))[:12]}"
            if freeze and current_formal_validation
            else "historischer Freeze, durch Codeänderung invalidiert"
            if freeze
            else "nicht eingefroren"
        )
        self.result_context.setText(
            f"Snapshot: {run_id} · Split: "
            f"{self.split_filter.currentData()} · "
            f"Grafik-Power: {self.inference_filter.currentText()} · "
            f"{completeness_text} · "
            f"{freeze_text} · explorative Scores bleiben getrennt markiert."
        )
        self.result_context.setToolTip(
            f"Abnahme-Audit: {acceptance_status}. Ein Freeze ist nur nach "
            "ready to freeze möglich."
        )
        self.freeze_button.setEnabled(
            freeze_eligible
            and acceptance_ready
            and not freeze
        )
        self.freeze_button.setToolTip(
            "Erstellt den unveränderlichen Freeze erst nach einem "
            "vollständig bestandenen Pre-Freeze-Audit. Locked-Test-"
            "Ergebnisse werden dabei nicht ausgewertet."
        )
        self._refresh_endpoint_filter(screening_path)
        self.results_atlas.refresh(
            run_id,
            str(self.endpoint_filter.currentData() or ""),
            revision=revision,
        )
        self._refresh_group_vcg(run_id)
        self._update_result_plots(
            screening_path,
            comparison_path,
            stability_path,
            phenome_path,
            curve_path,
            distribution_path,
            operating_path,
            adjusted_path,
            decision_path,
            transfer_path,
        )
        self._refresh_result_navigator(run_root)
        self.group_vcg_required_button.setEnabled(
            current_formal_validation
            and not immutable_development_run_markers(run_root)
        )

    def _refresh_result_navigator(self, run_root: Path) -> None:
        try:
            paths = self._ensure_result_explorer_artifacts(run_root)
        except Exception as exc:
            self.screening_table.set_source(None)
            self.performance_plot.clear()
            self.navigator_summary.setText(
                f"Ergebnisnavigator nicht verfügbar: {type(exc).__name__}: {exc}"
            )
            return
        mode = str(
            self.navigator_mode_combo.currentData() or "pathology_marker"
        )
        path = paths.get(mode)
        where, parameters = self._navigator_where(mode)
        self.screening_table.set_source(
            path if path is not None and path.is_file() else None,
            where_sql=where,
            parameters=parameters,
        )
        self._configure_navigator_columns(mode)
        self._draw_result_navigator(path, mode)
        self._update_navigator_detail()

    def _ensure_result_explorer_artifacts(
        self,
        run_root: Path,
    ) -> dict[str, Path]:
        # Do not rebuild large tables in the GUI or write into a production
        # run.  Legacy views may read only finished, declared stage outputs.
        result: dict[str, Path] = {}
        for manifest_path, names in (
            (run_root / "validate.manifest.json", (
                ("marker_pathology", "marker_pathology_explorer.parquet"),
                ("pathology_marker", "pathology_marker_explorer.parquet"),
            )),
            (run_root / "phenome" / "phenome.manifest.json", (
                ("discovery_ranked", "broad_discovery_ranked.parquet"),
            )),
        ):
            try:
                manifest = StageManifest.load(manifest_path)
            except (OSError, TypeError, ValueError):
                continue
            if manifest.status != "complete":
                continue
            declared = {str(Path(item).resolve()) for item in manifest.output_paths}
            for key, name in names:
                path = manifest_path.parent / name
                if path.is_file() and str(path.resolve()) in declared:
                    result[key] = path
        return result

    def _navigator_where(self, mode: str) -> tuple[str, tuple]:
        clauses: list[str] = []
        parameters: list[object] = []
        search = self.navigator_search.text().strip()
        if mode == "discovery_ranked":
            domain_where, domain_parameters = self._phenome_where()
            if domain_where:
                clauses.append(domain_where)
                parameters.extend(domain_parameters)
            if search:
                clauses.append(
                    "(CAST(target_name AS VARCHAR) ILIKE ? OR "
                    "CAST(target_id AS VARCHAR) ILIKE ? OR "
                    "CAST(feature_id AS VARCHAR) ILIKE ? OR "
                    "CAST(conceptual_family AS VARCHAR) ILIKE ?)"
                )
                parameters.extend([f"%{search}%"] * 4)
            return " AND ".join(clauses), tuple(parameters)
        endpoint = str(self.endpoint_filter.currentData() or "")
        panel = str(self.panel_filter.currentData() or "")
        inference_status = str(
            self.inference_filter.currentData() or ""
        )
        if endpoint:
            clauses.append("endpoint_id = ?")
            parameters.append(endpoint)
        if panel:
            clauses.append("panel_id = ?")
            parameters.append(panel)
        if inference_status:
            clauses.append("inference_status = ?")
            parameters.append(inference_status)
        if search:
            clauses.append(
                "(CAST(feature_id AS VARCHAR) ILIKE ? OR "
                "CAST(endpoint_id AS VARCHAR) ILIKE ? OR "
                "CAST(pathology_group AS VARCHAR) ILIKE ? OR "
                "CAST(conceptual_family AS VARCHAR) ILIKE ?)"
            )
            parameters.extend([f"%{search}%"] * 4)
        return " AND ".join(clauses), tuple(parameters)

    def _configure_navigator_columns(self, mode: str) -> None:
        if mode == "discovery_ranked":
            visible = {
                "evidence_tier",
                "discovery_domain",
                "target_name",
                "target_kind",
                "feature_id",
                "conceptual_family",
                "n_positive",
                "n_negative",
                "validation_effect",
                "validation_q_value",
                "validation_auc",
                "validation_ci_low",
                "validation_ci_high",
                "validation_stable",
                "replicated",
                "historical_implementation",
                "confounding_control",
                "followup_priority",
                "novelty_class",
                "interpretation",
            }
        else:
            visible = {
                "pathology_group",
                "endpoint_id",
                "feature_id",
                "candidate_type",
                "conceptual_family",
                "panel_id",
                "n_positive",
                "n_negative",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "average_precision",
                "sensitivity_at_90_specificity",
                "sensitivity_at_95_specificity",
                "specificity_at_90_sensitivity",
                "baseline_display_name",
                "delta_auc",
                "evidence_grade",
                "clinical_interpretation",
            }
        for index, column in enumerate(
            self.screening_table.model.frame.columns
        ):
            self.screening_table.view.setColumnHidden(
                index,
                str(column) not in visible,
            )
        self.screening_table.view.resizeColumnsToContents()

    def _draw_result_navigator(
        self,
        path: Optional[Path],
        mode: str,
    ) -> None:
        self.performance_plot.clear()
        if self.performance_plot.plotItem.legend is not None:
            self.performance_plot.plotItem.legend.clear()
        if path is None or not path.is_file():
            self.navigator_summary.setText("Keine Navigator-Daten verfügbar.")
            return
        import duckdb

        where, parameters = self._navigator_where(mode)
        where_sql = f" WHERE {where}" if where else ""
        with duckdb.connect(":memory:") as connection:
            connection.execute("SET memory_limit='256MB'")
            connection.execute("SET threads=1")
            frame = connection.execute(
                f"SELECT * FROM read_parquet(?) {where_sql} LIMIT 3001",
                [str(path), *parameters],
            ).fetchdf()
        truncated = len(frame) > 3000
        frame = frame.head(3000)
        sample_note = "Anzeige auf 3.000 gefilterte Zeilen begrenzt. " if truncated else ""
        if mode == "discovery_ranked":
            self.performance_plot.setLabel("bottom", "Validation-Effekt")
            self.performance_plot.setLabel("left", "-log10(q)")
            effect = pd.to_numeric(
                frame.get("validation_effect"),
                errors="coerce",
            )
            q_value = pd.to_numeric(
                frame.get("validation_q_value"),
                errors="coerce",
            )
            y_value = -np.log10(q_value.clip(lower=1e-300))
            valid = effect.notna() & y_value.notna()
            replicated = frame.get(
                "replicated",
                pd.Series(False, index=frame.index),
            ).fillna(False)
            historical = frame.get(
                "historical_implementation",
                pd.Series(False, index=frame.index),
            ).fillna(False)
            effect_valid = effect.loc[valid]
            y_valid = y_value.loc[valid]
            category = pd.Series(
                np.where(
                    historical.loc[valid],
                    "historical",
                    np.where(replicated.loc[valid], "replicated", "other"),
                ),
                index=effect_valid.index,
            )
            discovery_category_styles = (
                ("replicated", "#087e8b", "Repliziert"),
                ("historical", "#a2aaad", "Historisch (neu zu rechnen)"),
                ("other", "#8b969b", "Nicht repliziert"),
            )
            for key, color, label in discovery_category_styles:
                mask = (category == key).to_numpy()
                if not mask.any():
                    continue
                item = pg.ScatterPlotItem(
                    x=effect_valid.to_numpy()[mask],
                    y=y_valid.to_numpy()[mask],
                    size=7,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(None),
                )
                self.performance_plot.addItem(item)
                self.performance_plot.plotItem.legend.addItem(item, label)
            self.navigator_summary.setText(
                sample_note + f"{len(frame):,} priorisierte Zusammenhänge · "
                f"{int(replicated.sum()):,} repliziert · "
                f"{int(historical.sum()):,} historisch neu zu rechnen · "
                "Locked Test nicht gescannt · keine Diagnosen."
            )
            return
        self.performance_plot.setLabel("bottom", "ROC-AUC")
        self.performance_plot.setLabel(
            "left",
            "Sensitivität am Trainingsschwellenwert (Ziel: 95 % Spezifität)",
        )
        auc = pd.to_numeric(frame.get("roc_auc"), errors="coerce")
        sensitivity = pd.to_numeric(
            frame.get("sensitivity_at_95_specificity"),
            errors="coerce",
        )
        valid = auc.notna() & sensitivity.notna()
        candidate_type_styles = (
            ("vcg_single", "#087e8b", "VCG-Einzelmarker"),
            ("small_score", "#d1495b", "Kleiner Score"),
            ("classical_rule", "#b77900", "Klassische Regel"),
            ("ecg_single", "#557a95", "ECG-Einzelmarker"),
        )
        candidate_types = frame.get(
            "candidate_type",
            pd.Series("", index=frame.index),
        ).astype(str)
        auc_valid = auc.loc[valid]
        sensitivity_valid = sensitivity.loc[valid]
        types_valid = candidate_types.loc[valid]
        known_types = {key for key, _color, _label in candidate_type_styles}
        for key, color, label in candidate_type_styles + (
            ("__other__", "#7f8c92", "Sonstige"),
        ):
            mask = (
                (~types_valid.isin(known_types)).to_numpy()
                if key == "__other__"
                else (types_valid == key).to_numpy()
            )
            if not mask.any():
                continue
            item = pg.ScatterPlotItem(
                x=auc_valid.to_numpy()[mask],
                y=sensitivity_valid.to_numpy()[mask],
                size=7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen("#ffffff", width=0.4),
            )
            self.performance_plot.addItem(item)
            self.performance_plot.plotItem.legend.addItem(item, label)
        self.performance_plot.addLine(
            x=0.5,
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        historical = frame.get(
            "historical_pre_audit_result",
            pd.Series(False, index=frame.index),
        ).fillna(False)
        endpoint_count = (
            frame["endpoint_id"].nunique()
            if "endpoint_id" in frame
            else 0
        )
        self.navigator_summary.setText(
            sample_note + f"{len(frame):,} auswertbare Marker-Endpunkt-Paare · "
            f"{endpoint_count:,} Endpunkte · Validation · "
            f"{int(historical.sum()):,} historische Resultate müssen mit "
            "dem auditierten Code neu gerechnet werden."
        )

    def _filter_navigator_frame(
        self,
        frame: pd.DataFrame,
        mode: str,
    ) -> pd.DataFrame:
        result = frame
        search = self.navigator_search.text().strip().casefold()
        if mode == "discovery_ranked":
            domain = str(self.phenome_domain_combo.currentData() or "")
            if domain:
                column = (
                    "discovery_class"
                    if domain
                    in {"pathology_hypothesis", "future_outcome_hypothesis"}
                    else "discovery_domain"
                )
                result = result[
                    result[column].astype(str).eq(domain)
                ]
            search_columns = (
                "target_name",
                "target_id",
                "feature_id",
                "conceptual_family",
            )
        else:
            endpoint = str(self.endpoint_filter.currentData() or "")
            panel = str(self.panel_filter.currentData() or "")
            inference_status = str(
                self.inference_filter.currentData() or ""
            )
            if endpoint:
                result = result[
                    result["endpoint_id"].astype(str).eq(endpoint)
                ]
            if panel:
                result = result[
                    result["panel_id"].astype(str).eq(panel)
                ]
            if inference_status:
                result = result[
                    result["inference_status"]
                    .astype(str)
                    .eq(inference_status)
                ]
            search_columns = (
                "feature_id",
                "endpoint_id",
                "pathology_group",
                "conceptual_family",
            )
        if search:
            mask = pd.Series(False, index=result.index)
            for column in search_columns:
                if column in result:
                    mask |= (
                        result[column]
                        .astype(str)
                        .str.casefold()
                        .str.contains(search, regex=False)
                    )
            result = result[mask]
        return result

    def _update_navigator_detail(self) -> None:
        row = self.screening_table.selected_row()
        if not row:
            return
        if "target_name" in row:
            self.navigator_summary.setText(
                f"{row.get('target_name', '')} · {row.get('feature_id', '')} · "
                f"Evidenz {row.get('evidence_tier', '')} · "
                f"Validation-Effekt {_format_number(row.get('validation_effect'))} · "
                f"q {_format_number(row.get('validation_q_value'))}. "
                f"{row.get('interpretation', '')}"
            )
            return
        self.navigator_summary.setText(
            f"{row.get('endpoint_id', '')} · {row.get('feature_id', '')} · "
            f"AUC {_format_number(row.get('roc_auc'))} "
            f"[{_format_number(row.get('roc_auc_ci_low'))}; "
            f"{_format_number(row.get('roc_auc_ci_high'))}] · "
            f"Sensitivität am Trainingsschwellenwert (95-%-Spezifitätsziel) "
            f"{_format_number(row.get('sensitivity_at_95_specificity_reported'))} · "
            f"beobachtete Spezifität {_format_number(row.get('operating_specificity'))} · "
            f"Status {row.get('operating_point_validation_status', 'unknown')} · "
            f"{row.get('clinical_interpretation', '')}"
        )

    def enqueue_group_vcg(self) -> None:
        run_id = self._selected_current_run_id(self.results_run_combo)
        endpoint_id = str(self.endpoint_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "")
        if not run_id or not endpoint_id:
            QMessageBox.information(
                self,
                "Endpunkt auswählen",
                "Wählen Sie zuerst genau einen klinischen Endpunkt.",
            )
            return
        if split not in {"train", "validation"}:
            QMessageBox.warning(
                self,
                "Development-Split erforderlich",
                "Gruppen-VCG ist ausschließlich für Train oder Validation "
                "verfügbar. Der Locked Test bleibt versiegelt.",
            )
            return
        active_states = {
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.PAUSING,
            JobState.PAUSED,
            JobState.RESUMING,
        }
        for record in self.job_store.list(limit=500):
            arguments = dict(record.spec.arguments)
            if (
                record.state in active_states
                and record.spec.stage == "group-vcg"
                and str(arguments.get("run_id", "")) == run_id
                and str(arguments.get("endpoint_id", "")) == endpoint_id
                and str(arguments.get("split", "")) == split
            ):
                self.status_changed.emit(
                    f"Gruppen-VCG-Job {record.spec.job_id} ist bereits "
                    f"{record.state.value}."
                )
                self.workspace_tabs.setCurrentIndex(TAB_ANALYSIS)
                self.refresh_jobs()
                return
        record = self.job_store.create(
            stage="group-vcg",
            project_file=self.config.project_file,
            arguments={
                "run_id": run_id,
                "endpoint_id": endpoint_id,
                "split": split,
                "panel_ids": ("primary_core", "sentinel"),
                "max_per_class": 40,
                "minimum_per_class": 10,
                "candidate_multiplier": 4,
                "quality_minimum": 0.75,
                "bootstrap_iterations": 500,
            },
            priority=70,
        )
        ensure_supervisor(self.config)
        self.status_changed.emit(
            f"Gruppen-VCG-Job {record.spec.job_id} wurde gestartet."
        )
        self.workspace_tabs.setCurrentIndex(TAB_ANALYSIS)
        self.refresh_jobs()

    def enqueue_required_group_vcg(self) -> None:
        run_id = self._selected_current_run_id(self.results_run_combo)
        if not run_id:
            QMessageBox.information(
                self,
                "Run auswählen",
                "Wählen Sie zuerst einen vollständigen Train-/Validation-Run.",
            )
            return
        run_root = self.config.runs_root / run_id
        if not _formal_development_validation_complete(run_root):
            QMessageBox.warning(
                self,
                "Vollständiger Development-Run erforderlich",
                "Das Abnahme-Set benötigt einen vollständigen, nicht als "
                "Vorschau markierten Train-/Validation-Run.",
            )
            return
        active_states = {
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.PAUSING,
            JobState.PAUSED,
            JobState.RESUMING,
        }
        active_endpoints = {
            str(record.spec.arguments.get("endpoint_id", ""))
            for record in self.job_store.list(limit=10_000)
            if record.state in active_states
            and record.spec.stage == "group-vcg"
            and str(record.spec.arguments.get("run_id", "")) == run_id
            and str(record.spec.arguments.get("split", "")) == "validation"
        }
        queued: list[str] = []
        skipped: list[str] = []
        for endpoint_id in DEFAULT_GROUP_VCG_ENDPOINTS:
            manifest_path = (
                run_root
                / "group_vcg"
                / f"endpoint={endpoint_id}"
                / "split=validation"
                / "group_vcg.manifest.json"
            )
            if _manifest_complete(manifest_path) or endpoint_id in active_endpoints:
                skipped.append(endpoint_id)
                continue
            self.job_store.create(
                stage="group-vcg",
                project_file=self.config.project_file,
                arguments={
                    "run_id": run_id,
                    "endpoint_id": endpoint_id,
                    "split": "validation",
                    "panel_ids": ("primary_core", "sentinel"),
                    "max_per_class": 40,
                    "minimum_per_class": 10,
                    "candidate_multiplier": 4,
                    "quality_minimum": 0.75,
                    "bootstrap_iterations": 500,
                },
                priority=70,
                requested_by="gui",
            )
            queued.append(endpoint_id)
        if queued:
            ensure_supervisor(self.config)
        self.status_changed.emit(
            f"Gruppen-VCG-Abnahme-Set: {len(queued)} neu eingereiht, "
            f"{len(skipped)} bereits fertig oder aktiv."
        )
        self.workspace_tabs.setCurrentIndex(TAB_ANALYSIS)
        self.refresh_jobs()

    def _refresh_group_vcg(self, run_id: str) -> None:
        endpoint_id = str(self.endpoint_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "")
        selection_available = bool(
            endpoint_id and split in {"train", "validation"}
        )
        run_is_immutable = bool(
            immutable_development_run_markers(
                self.config.runs_root / run_id
            )
        )
        self.group_vcg_build_button.setEnabled(
            selection_available and not run_is_immutable
        )
        if not selection_available:
            self.group_vcg_context.setText(
                "Wählen Sie genau einen klinischen Endpunkt; die "
                "Gruppenansicht bleibt development-only."
            )
            self.group_vcg_audit_table.set_source(None)
            self._clear_group_vcg_view()
            self._clear_group_vcg_projections()
            return
        if run_is_immutable:
            self.group_vcg_build_button.setToolTip(
                "Der ausgewählte Run ist eingefroren. Vorhandene "
                "Gruppen-VCGs bleiben lesbar; neue Berechnungen benötigen "
                "eine neue Run-ID."
            )
        else:
            self.group_vcg_build_button.setToolTip(
                "Berechnet die train-/validation-basierte Gruppenansicht "
                "für den gewählten Endpunkt."
            )
        root = group_vcg_artifact_dir(
            self.config,
            run_id,
            endpoint_id,
            split,
        )
        summary_path = root / "group_vcg_summary.parquet"
        difference_path = root / "group_vcg_difference.parquet"
        audit_path = root / "group_vcg_case_audit.parquet"
        provenance = _read_json(root / "group_vcg_provenance.json")
        manifest_path = root / "group_vcg.manifest.json"
        try:
            manifest = StageManifest.load(manifest_path)
        except (OSError, TypeError, ValueError):
            manifest = None
        required_paths = {
            str(path.resolve()) for path in (
                summary_path, difference_path, audit_path,
                root / "group_vcg_provenance.json",
            )
        }
        verified_scope = bool(
            manifest is not None
            and manifest.status == "complete"
            and manifest.stage == "group-vcg"
            and manifest.stage_version == GROUP_VCG_STAGE_VERSION
            and required_paths.issubset(set(manifest.output_paths))
            and provenance.get("run_id") == run_id
            and provenance.get("endpoint_id") == endpoint_id
            and provenance.get("split") == split
            and provenance.get("locked_test_read") is False
            and provenance.get("input_hash") == manifest.input_hash
        )
        if not verified_scope or not all(Path(path).is_file() for path in required_paths):
            self.group_vcg_audit_table.set_source(None)
            self.group_vcg_context.setText(
                f"{endpoint_id} · {split}: keine vollständige und scope-geprüfte "
                "Gruppenansicht vorhanden. Alte oder unvollständige Artefakte "
                "werden nicht als aktuelles Resultat angezeigt."
            )
            self._clear_group_vcg_view()
            self._clear_group_vcg_projections()
            return
        self.group_vcg_audit_table.set_source(audit_path)
        summary = pd.read_parquet(summary_path)
        difference = pd.read_parquet(difference_path)
        class_counts = provenance.get("class_counts", {})
        snapshot_hash = str(
            summary["feature_snapshot_hash"].iloc[0]
            if "feature_snapshot_hash" in summary and not summary.empty
            else ""
        )
        self.group_vcg_context.setText(
            f"{endpoint_id} · {split} · n positiv="
            f"{class_counts.get('positive', '?')}, negativ="
            f"{class_counts.get('negative', '?')} · Snapshot "
            f"{snapshot_hash[:12]} · punktweise 95-%-Subject-Bootstrap-CI · "
            "deskriptive Development-Visualisierung."
        )
        self._render_group_vcg(summary, difference)

    def _refresh_endpoint_filter(self, screening_path: Path) -> None:
        current = self.endpoint_filter.currentData()
        endpoints = []
        if screening_path.is_file():
            frame = pd.read_parquet(
                screening_path,
                columns=["endpoint_id"],
            )
            endpoints = sorted(frame["endpoint_id"].dropna().astype(str).unique())
        expected_items = [
            ("", "Übersicht aller klinischen Ziele"),
            *[
                (
                    item,
                    f"{clinical_display_term(item).category_de} · "
                    f"{clinical_display_term(item).display_name_de}",
                )
                for item in endpoints
            ],
        ]
        if self.endpoint_filter.count() == len(expected_items) and all(
            self.endpoint_filter.itemData(index) == value
            for index, (value, _) in enumerate(expected_items)
        ):
            return
        self.endpoint_filter.blockSignals(True)
        self.endpoint_filter.clear()
        for value, label in expected_items:
            self.endpoint_filter.addItem(label, value)
            if value:
                term = clinical_display_term(value)
                self.endpoint_filter.setItemData(
                    self.endpoint_filter.count() - 1,
                    (
                        f"{term.short_definition_de}\n"
                        f"Quelle: {term.source_summary_de or 'siehe Labeldefinition'}"
                    ),
                    Qt.ItemDataRole.ToolTipRole,
                )
        index = self.endpoint_filter.findData(current)
        self.endpoint_filter.setCurrentIndex(max(0, index))
        self.endpoint_filter.blockSignals(False)

    # Colour-blind safe and ordered light to dark, so the tertile order is
    # readable without relying on hue alone.
    _SURVIVAL_GROUP_PENS = ("#66a0d1", "#c98b23", "#b3382c")

    def _refresh_reference_range_summary(self, reference_root) -> None:
        """Say how many strata could be estimated, and for how many people."""

        path = reference_root / "marker_reference_ranges.parquet"
        policy = _read_json(reference_root / "cohort_policy.json")
        if not path.is_file():
            self.reference_range_summary.setText(
                "Noch keine Referenzbereiche. Die Stufe 'reference-ranges' "
                "erzeugt sie."
            )
            return
        try:
            ranges = pd.read_parquet(
                path, columns=["marker_id", "estimable", "subject_count"]
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Reference-range artifact is unreadable: %s", exc)
            self.reference_range_summary.setText(
                "Referenzbereiche vorhanden, aber nicht lesbar."
            )
            return
        if not len(ranges):
            self.reference_range_summary.setText(
                "Referenzbereichstabelle ist leer."
            )
            return
        estimable = ranges["estimable"].astype(bool)
        markers = int(ranges["marker_id"].nunique())
        with_any = int(ranges.loc[estimable, "marker_id"].nunique())
        largest = int(
            pd.to_numeric(ranges["subject_count"], errors="coerce").max()
        )
        population = str(policy.get("population", "unbekannt"))
        transform = str(policy.get("vcg_transform", "unbekannt"))
        self.reference_range_summary.setText(
            f"{with_any:,} von {markers:,} Markern haben in mindestens einer "
            f"Schicht einen Referenzbereich; {int(estimable.sum()):,} von "
            f"{len(ranges):,} Schichtzeilen sind schätzbar. Größte Schicht: "
            f"{largest:,} Personen.\n"
            f"Population: {population} -- keine Normwerte. "
            f"Transformation: {transform}."
        )

    def _redraw_segment_availability(self) -> None:
        """Two bars per segment: cases and controls, side by side.

        The gap between them is the whole point, so they are drawn adjacent
        rather than stacked or as a single difference -- a reader has to see
        both levels to know whether a 20-point gap is 80 against 60 or 25
        against 5.
        """

        self.availability_plot.clear()
        table = getattr(self, "_segment_availability", None)
        if table is None or len(table) == 0:
            return
        endpoint = self.availability_endpoint_combo.currentData()
        if not endpoint:
            return
        subset = table.loc[
            table["endpoint_id"].astype(str).eq(str(endpoint))
        ].sort_values("segment")
        if subset.empty:
            return
        segments = list(subset["segment"].astype(str))
        positions = np.arange(len(segments), dtype=float)
        for offset, column, colour, label in (
            (-0.18, "available_positive_pct", "#b3382c", "Fälle"),
            (0.18, "available_negative_pct", "#66a0d1", "Kontrollen"),
        ):
            self.availability_plot.addItem(
                pg.BarGraphItem(
                    x=positions + offset,
                    height=subset[column].to_numpy(),
                    width=0.32,
                    brush=colour,
                    pen=None,
                    name=label,
                )
            )
        axis = self.availability_plot.getAxis("bottom")
        axis.setTicks(
            [[(position, name.upper()) for position, name in zip(positions, segments)]]
        )

    def _redraw_survival_curves(self) -> None:
        """Draw the cumulative incidence curves for the selected marker."""

        self.survival_plot.clear()
        curves = getattr(self, "_survival_curves", None)
        if curves is None or len(curves) == 0:
            return
        marker = self.survival_marker_combo.currentData()
        if not marker:
            return
        subset = curves.loc[curves["feature_id"].astype(str).eq(str(marker))]
        if subset.empty:
            return
        for index, group in enumerate(sorted(subset["group"].unique())):
            block = subset.loc[subset["group"].eq(group)].sort_values("time_days")
            colour = self._SURVIVAL_GROUP_PENS[
                index % len(self._SURVIVAL_GROUP_PENS)
            ]
            label = str(block["group_label"].iloc[0])
            n = int(block["n"].iloc[0])
            events = int(block["n_events"].iloc[0])
            self.survival_plot.plot(
                block["time_days"].to_numpy(),
                block["cumulative_incidence"].to_numpy() * 100.0,
                pen=pg.mkPen(colour, width=2),
                name=f"{label} (n={n:,}, {events:,} Ereignisse)",
                stepMode=False,
            )

    def _redraw_hazard_forest(self, fits) -> None:
        """One row per marker, both populations, with the interval."""

        self.hazard_forest_plot.clear()
        if fits is None or len(fits) == 0:
            return
        needed = {"feature_id", "population", "hazard_ratio_per_sd",
                  "hr_lower", "hr_upper"}
        if not needed <= set(fits.columns):
            return
        ordered = fits.sort_values(["feature_id", "population"])
        labels: list[tuple[float, str]] = []
        row = 0.0
        for marker in sorted(ordered["feature_id"].unique()):
            block = ordered.loc[ordered["feature_id"].eq(marker)]
            for _, entry in block.iterrows():
                first = str(entry["population"]).startswith("first")
                centre = float(entry["hazard_ratio_per_sd"])
                low = float(entry["hr_lower"])
                high = float(entry["hr_upper"])
                colour = "#1f4e79" if first else "#8c8c8c"
                self.hazard_forest_plot.addItem(
                    pg.ErrorBarItem(
                        x=np.array([centre]),
                        y=np.array([row]),
                        left=np.array([max(centre - low, 0.0)]),
                        right=np.array([max(high - centre, 0.0)]),
                        beam=0.12,
                        pen=pg.mkPen(colour, width=2),
                    )
                )
                self.hazard_forest_plot.plot(
                    [centre], [row],
                    pen=None, symbol="o", symbolSize=8,
                    symbolBrush=colour, symbolPen=colour,
                )
                short = "erstes ECG" if first else "alle, robust"
                labels.append((row, f"{marker} — {short}"))
                row -= 1.0
            row -= 0.4
        # No effect. Every interval that crosses it is a result compatible
        # with nothing happening.
        self.hazard_forest_plot.addItem(
            pg.InfiniteLine(
                pos=1.0, angle=90,
                pen=pg.mkPen("#a0a0a0", width=1, style=Qt.PenStyle.DashLine),
            )
        )
        axis = self.hazard_forest_plot.getAxis("left")
        axis.setTicks([[(position, text) for position, text in labels]])
        axis.setWidth(320)

    # A scatter of every ECG-echo pair would be 173.374 points, which pyqtgraph
    # draws slowly and a reader cannot see through. A systematic sample keeps
    # the shape of the cloud and the drawing responsive.
    _SCATTER_POINT_LIMIT = 6000

    @staticmethod
    def _thinned(frame, limit: int = 6000):
        """Every k-th row, so the sample is not a run of adjacent patients."""

        if frame is None or len(frame) <= limit:
            return frame
        step = int(np.ceil(len(frame) / float(limit)))
        return frame.iloc[::step]

    def _redraw_voltage_mass_scatter(self, pairs) -> None:
        self.voltage_mass_plot.clear()
        if pairs is None or len(pairs) == 0 or "lv_mass_g" not in pairs.columns:
            return
        marker = next(
            (
                name
                for name in (
                    "sokolow_lyon_voltage_surrogate",
                    "qrs_vector_magnitude",
                    "svg_magnitude",
                )
                if name in pairs.columns
            ),
            None,
        )
        if marker is None:
            return
        frame = pairs[["lv_mass_g", marker]].copy()
        if "hypertrophied" in pairs.columns:
            frame["hypertrophied"] = pairs["hypertrophied"].astype(bool)
        else:
            frame["hypertrophied"] = False
        frame = frame.dropna(subset=["lv_mass_g", marker])
        if frame.empty:
            return
        self.voltage_mass_plot.setLabel("left", biomarker_display_name(marker))
        for flag, colour, label in (
            (False, "#9fb6c9", "Wand normal"),
            (True, "#b3382c", "Wand verdickt (>= 1,2 cm)"),
        ):
            block = self._thinned(
                frame.loc[frame["hypertrophied"].eq(flag)],
                self._SCATTER_POINT_LIMIT // 2,
            )
            if block is None or block.empty:
                continue
            self.voltage_mass_plot.plot(
                block["lv_mass_g"].to_numpy(),
                block[marker].to_numpy(),
                pen=None, symbol="o", symbolSize=3,
                symbolBrush=pg.mkBrush(colour + "80"),
                symbolPen=None,
                name=f"{label} (n={len(frame.loc[frame['hypertrophied'].eq(flag)]):,})",
            )

    def _redraw_within_subject_scatter(self) -> None:
        from ..paired_scatter import prepare_paired_scatter
        self.within_subject_plot.clear()
        self.within_subject_plot.setTitle("Keine auswertbaren Paare ausgewählt.")
        self.within_subject_plot.setLabel("bottom", "Änderung der Bezugsgröße")
        self.within_subject_plot.setLabel("left", "Änderung des Markers")
        self._within_scatter_counts = {}
        changes = getattr(self, "_within_subject_changes", None)
        if changes is None or len(changes) == 0:
            return
        marker = self.within_marker_combo.currentData()
        if not marker or f"delta_{marker}" not in changes.columns:
            return
        analyte = self.within_analyte_combo.currentData()
        block, info = prepare_paired_scatter(changes, marker, analyte, limit=self._SCATTER_POINT_LIMIT)
        self._within_scatter_counts = info
        if info["status"] == "ambiguous_analyte_or_unit":
            self.within_subject_plot.setTitle("Mehrdeutige Laborgröße/Einheit: keine gemeinsame Grafik.")
            return
        title = f"{info['n_shown']:,} dargestellt / {info['n_valid']:,} auswertbar / {info['n_total']:,} Paare"
        title += f"<br>{info['n_known_patients']:,} bekannte Patienten"
        if info["n_missing_patient_rows"]:
            title += f" · {info['n_missing_patient_rows']:,} ohne Patientenkennung"
        if info["sampling"] != "none":
            title += " · Darstellungsstichprobe"
        if info["legacy_reference"]:
            title += "<br>Legacy: Referenzstatus fehlt"
        self.within_subject_plot.setTitle(title)
        if block.empty:
            return
        lab = "delta_lab"
        self.within_subject_plot.setLabel(
            "left", f"Änderung {biomarker_display_name(marker)}"
        )
        unit = info["unit"]
        label = str(analyte or "Bezugsgröße").replace("_", " ")
        self.within_subject_plot.setLabel(
            "bottom", f"Änderung {label}" + (f" [{unit}]" if unit else "")
        )
        self.within_subject_plot.plot(
            block[lab].to_numpy(),
            block[f"delta_{marker}"].to_numpy(),
            pen=None, symbol="o", symbolSize=5,
            symbolBrush=pg.mkBrush("#56B4E999" if self._dark_mode else "#1f4e7999"), symbolPen=None,
        )
        # Both axes are differences, so zero is the reference in each.
        for angle, position in ((90, 0.0), (0, 0.0)):
            self.within_subject_plot.addItem(
                pg.InfiniteLine(
                    pos=position, angle=angle,
                    pen=pg.mkPen("#a0a0a0", width=1, style=Qt.PenStyle.DashLine),
                )
            )

    def _refresh_targeted_designs(self, run_root: Path) -> None:
        """Fill the control panel and the five targeted designs.

        Each is optional: a run that predates a stage simply shows nothing for
        it rather than an error, which is what happens whenever an older run is
        opened after the pipeline has grown.
        """

        control_table = run_root / "controls" / "control_panel.parquet"
        self.control_panel_table.set_source(
            control_table if control_table.is_file() else None
        )
        verdict = _read_json(run_root / "controls" / "control_panel_verdict.json")
        if verdict:
            floor = verdict.get("confounding_noise_floor")
            wrong = int(verdict.get("n_positive_wrong_direction", 0) or 0)
            parts = [
                f"{verdict.get('n_positive_recovered', 0)} von "
                f"{verdict.get('n_positive_tested', 0)} prüfbaren "
                "Positivkontrollen wiedergefunden",
                f"{wrong} mit falscher Richtung",
                f"{verdict.get('n_negative_false_positive', 0)} von "
                f"{verdict.get('n_negative_tested', 0)} Negativkontrollen "
                "auffällig",
            ]
            if floor is not None:
                parts.append(f"gemessener Rauschboden {float(floor):.3f}")
            summary = "; ".join(parts) + "."
            if wrong:
                summary += (
                    " Eine Kontrolle zeigt in die falsche Richtung -- das ist "
                    "der schwerwiegende Fall und vor jeder weiteren "
                    "Auswertung zu klären."
                )
            self.control_panel_summary.setText(summary)
        else:
            self.control_panel_summary.setText(
                "Noch keine Kontrolltafel für diesen Lauf. Die Stufe "
                "'Kontrolltafel gegen bekannte Physiologie' lässt sich "
                "einzeln nachtragen; sie rechnet auf den fertigen "
                "Screening-Ergebnissen und dauert Sekunden."
            )

        sources = (
            (
                self.recovery_episode_table,
                run_root / "within_subject" / "recovery_episode_results.parquet",
            ),
            (
                self.voltage_mass_models_table,
                run_root / "voltage_mass_model_results.parquet",
            ),
            (
                self.within_subject_table,
                run_root / "within_subject" / "within_subject_change.parquet",
            ),
            (
                self.voltage_mass_table,
                run_root / "voltage_mass" / "voltage_mass_discordance.parquet",
            ),
            (
                self.time_to_event_table,
                run_root / "survival" / "time_to_event.parquet",
            ),
            (
                self.occlusion_yield_table,
                run_root / "occlusion" / "criterion_yield.parquet",
            ),
            (
                self.two_stage_table,
                run_root / "two_stage" / "confirmation_plan.parquet",
            ),
        )
        present = 0
        for table, path in sources:
            table.set_source(path if path.is_file() else None)
            present += int(path.is_file())

        shrinkage_path = run_root / "controls" / "control_panel_shrinkage.parquet"
        self.control_panel_shrinkage_table.set_source(
            shrinkage_path if shrinkage_path.is_file() else None
        )

        self._survival_curves = None
        curves_path = run_root / "survival" / "time_to_event_curves.parquet"
        previous_marker = self.survival_marker_combo.currentData()
        self.survival_marker_combo.blockSignals(True)
        self.survival_marker_combo.clear()
        if curves_path.is_file():
            try:
                curves = pd.read_parquet(curves_path)
            except Exception as exc:  # noqa: BLE001 - display must survive
                LOGGER.warning("Survival-curve artifact is unreadable: %s", exc)
                curves = None    # take the whole results view down
            if curves is not None and len(curves):
                self._survival_curves = curves
                for marker in sorted(curves["feature_id"].astype(str).unique()):
                    self.survival_marker_combo.addItem(
                        biomarker_display_name(marker), marker
                    )
                if previous_marker is not None:
                    index = self.survival_marker_combo.findData(previous_marker)
                    if index >= 0:
                        self.survival_marker_combo.setCurrentIndex(index)
        self.survival_marker_combo.blockSignals(False)
        self._redraw_survival_curves()

        self._segment_availability = None
        previous_endpoint = self.availability_endpoint_combo.currentData()
        self.availability_endpoint_combo.blockSignals(True)
        self.availability_endpoint_combo.clear()
        availability_path = (
            run_root / "availability" / "segment_availability.parquet"
        )
        self.availability_table.set_source(
            availability_path if availability_path.is_file() else None
        )
        if availability_path.is_file():
            try:
                availability = pd.read_parquet(availability_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Segment-availability artifact is unreadable: %s", exc)
                availability = None
            if availability is not None and len(availability):
                self._segment_availability = availability
                # Worst first: the endpoint whose markers depend most on the
                # diagnosis is the one a reader should see without hunting.
                order = (
                    availability.groupby("endpoint_id")["gap_points"]
                    .apply(lambda values: values.abs().max())
                    .sort_values(ascending=False)
                )
                for endpoint in order.index:
                    self.availability_endpoint_combo.addItem(
                        f"{clinical_display_term(endpoint).display_name_de} "
                        f"({order[endpoint]:+.0f} Punkte)",
                        endpoint,
                    )
                if previous_endpoint is not None:
                    index = self.availability_endpoint_combo.findData(
                        previous_endpoint
                    )
                    if index >= 0:
                        self.availability_endpoint_combo.setCurrentIndex(index)
        self.availability_endpoint_combo.blockSignals(False)
        self._redraw_segment_availability()

        reference_root = run_root / "reference_ranges"
        guideline_root = run_root / "guideline"
        for table, path in (
            (
                self.candidate_evidence_table,
                guideline_root / "candidate_evidence.parquet",
            ),
            (
                self.reference_range_table,
                reference_root / "marker_reference_ranges.parquet",
            ),
            (
                self.reference_cohort_table,
                reference_root / "cohort_summary.parquet",
            ),
            (
                self.transform_agreement_table,
                guideline_root / "transform_agreement.parquet",
            ),
            (
                self.reporting_set_table,
                guideline_root / "minimum_reporting_set.parquet",
            ),
            (
                self.reporting_algorithm_table,
                guideline_root / "reporting_algorithm.parquet",
            ),
            (
                self.mandatory_context_table,
                guideline_root / "mandatory_context_fields.parquet",
            ),
        ):
            table.set_source(path if path.is_file() else None)
        self._refresh_reference_range_summary(reference_root)
        self.lead_comparison_view.set_source(run_root / "lead_comparison_results.parquet")
        self._draw_evidence_ladder(
            guideline_root / "candidate_evidence.parquet"
        )
        self._draw_transform_agreement(
            guideline_root / "transform_agreement.parquet"
        )
        population_root = (
            self.config.external_validation_root
            / run_root.name
            / "population_cohorts"
        )
        for table, path in (
            (self.population_coverage_table, population_root / "coverage.parquet"),
            (self.population_value_table, population_root / "cohort_added_value.parquet"),
            (self.external_value_table, population_root.parent / "cohort_added_value.parquet"),
            (self.external_annotation_table, population_root.parent / "external_agreement.parquet"),
            (
                self.population_loader_acceptance_table,
                population_root / "loader_acceptance.parquet",
            ),
            (
                self.code15_reference_table,
                population_root / "code15_reference_ranges.parquet",
            ),
            (
                self.population_diagnostic_table,
                population_root / "code15_diagnostic_associations.parquet",
            ),
            (
                self.population_agreement_table,
                population_root / "direct_transform_agreement.parquet",
            ),
            (
                self.population_mortality_table,
                population_root / "mortality_associations.parquet",
            ),
            (
                self.population_chagas_table,
                population_root / "sami_trop_chagas_contrast.parquet",
            ),
        ):
            table.set_source(path if path.is_file() else None)
        self._draw_population_mortality(
            population_root / "mortality_associations.parquet"
        )
        self._refresh_gwas_context()

        redundancy_path = (
            run_root / "redundancy" / "marker_redundancy.parquet"
        )
        self.redundancy_table.set_source(
            redundancy_path if redundancy_path.is_file() else None
        )
        redundancy_summary = _read_json(
            run_root / "redundancy" / "marker_redundancy_summary.json"
        )
        if redundancy_summary:
            self.redundancy_summary.setText(
                f"{redundancy_summary.get('n_clusters', 0):,} Gruppen; "
                f"{redundancy_summary.get('n_redundant', 0):,} Marker tragen "
                "dieselbe Information wie ein anderer. Sie bleiben im Katalog "
                "-- die Gruppe sagt nur, dass sie zusammen einen Befund "
                "ergeben."
            )
        else:
            self.redundancy_summary.setText(
                "Noch keine Redundanzgruppen für diesen Lauf. Die Stufe "
                "'Markergruppen bestimmen' lässt sich einzeln nachtragen."
            )

        self._within_subject_changes = None
        self.dose_pair_view.set_source(run_root / "within_subject" / "dose_paired_changes.parquet")
        previous_within = self.within_marker_combo.currentData()
        previous_analyte = self.within_analyte_combo.currentData()
        self.within_marker_combo.blockSignals(True)
        self.within_analyte_combo.blockSignals(True)
        self.within_marker_combo.clear()
        self.within_analyte_combo.clear()
        changes_path = (
            run_root / "within_subject" / "dialysis_paired_changes.parquet"
        )
        if changes_path.is_file():
            try:
                changes = pd.read_parquet(changes_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Within-subject artifact is unreadable: %s", exc)
                changes = None
            if changes is not None and len(changes):
                self._within_subject_changes = changes
                if "analyte_id" in changes.columns:
                    analyte_rows = changes[
                        [
                            name
                            for name in ("analyte_id", "analyte_unit")
                            if name in changes.columns
                        ]
                    ].drop_duplicates("analyte_id")
                    for row in analyte_rows.itertuples(index=False):
                        analyte_id = str(row.analyte_id)
                        unit = str(getattr(row, "analyte_unit", ""))
                        label = analyte_id.replace("_", " ").title()
                        self.within_analyte_combo.addItem(
                            f"{label} ({unit})" if unit else label,
                            analyte_id,
                        )
                    if previous_analyte is not None:
                        index = self.within_analyte_combo.findData(
                            previous_analyte
                        )
                        if index >= 0:
                            self.within_analyte_combo.setCurrentIndex(index)
                for column in sorted(changes.columns):
                    if column.startswith("delta_") and column != "delta_lab":
                        marker = column[len("delta_"):]
                        self.within_marker_combo.addItem(
                            biomarker_display_name(marker), marker
                        )
                if previous_within is not None:
                    index = self.within_marker_combo.findData(previous_within)
                    if index >= 0:
                        self.within_marker_combo.setCurrentIndex(index)
        self.within_marker_combo.blockSignals(False)
        self.within_analyte_combo.blockSignals(False)
        self._redraw_within_subject_scatter()

        pairs_path = run_root / "voltage_mass" / "voltage_mass_pairs.parquet"
        pairs = None
        if pairs_path.is_file():
            try:
                pairs = pd.read_parquet(pairs_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Voltage/mass artifact is unreadable: %s", exc)
                pairs = None
        self._redraw_voltage_mass_scatter(pairs)

        fits_path = run_root / "survival" / "time_to_event.parquet"
        fits = None
        if fits_path.is_file():
            try:
                fits = pd.read_parquet(fits_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Survival-fit artifact is unreadable: %s", exc)
                fits = None
        self._redraw_hazard_forest(fits)

        notes: list[str] = []
        occlusion = _read_json(
            run_root / "occlusion" / "occlusion_yield_summary.json"
        )
        if occlusion.get("n_events"):
            notes.append(
                f"{occlusion['n_missed']:,} von {occlusion['n_events']:,} "
                "Ereignissen ohne ST-Hebungskriterium "
                f"({occlusion.get('missed_share_pct', 0.0):.1f} %); "
                f"{occlusion.get('n_recovered_by_equivalents', 0):,} davon "
                "von de Winter, Wellens oder posteriorem Muster erfasst. "
                "Der Endpunkt ist ein Revaskularisations-Surrogat ohne "
                "Angiographie."
            )
        within = _read_json(
            run_root / "within_subject" / "within_subject_summary.json"
        )
        self.joint_lab_coverage_view.set_summary(within.get("joint_lab_coverage"))
        if within.get("n_dialysis_pairs_with_lab"):
            notes.append(
                f"{within['n_dialysis_pairs_with_lab']:,} Dialysepaare mit "
                "beidseitigem Kaliumwert, Median "
                f"{within.get('median_delta_potassium', float('nan')):+.2f} "
                "mmol/l."
            )
        survival = _read_json(
            run_root / "survival" / "time_to_event_summary.json"
        )
        if survival.get("n_events"):
            notes.append(
                f"{survival['n_events']:,} Todesfälle in "
                f"{int(survival.get('horizon_days', 0)):d} Tagen über "
                f"{survival.get('n_ecgs', 0):,} ECGs."
            )
        voltage = _read_json(
            run_root / "voltage_mass" / "voltage_mass_summary.json"
        )
        if voltage.get("n_pairs"):
            notes.append(
                f"{voltage['n_pairs']:,} ECG-Echo-Paare, davon "
                f"{voltage.get('n_hypertrophied', 0):,} mit verdickter Wand; "
                "mediane LV-Masse "
                f"{voltage.get('median_lv_mass_g') or float('nan'):.0f} g."
            )
        # One correction across every arm that reports a p-value. Correcting
        # inside each arm would leave the choice *between* arms uncorrected,
        # which is where the selection actually happens.
        try:
            from ...mimic.targeted_multiplicity import (
                collect_arm_results,
                correct_across_arms,
                multiplicity_summary,
            )

            corrected = correct_across_arms(collect_arm_results(run_root))
            multiplicity = multiplicity_summary(
                corrected,
                noise_floor=verdict.get("confounding_noise_floor"),
            )
        except Exception as exc:  # noqa: BLE001 - display must survive
            LOGGER.warning("Multiplicity display summary failed: %s", exc)
            multiplicity = {}
        if multiplicity.get("n_tests"):
            floor = multiplicity.get("confounding_noise_floor")
            notes.append(
                f"Ueber alle Arme: {multiplicity['n_significant']} von "
                f"{multiplicity['n_tests']} Tests halten der gemeinsamen "
                "Mehrfachtestkorrektur stand"
                + (
                    f"; gemessener Rauschboden {float(floor):.3f} AUC -- was "
                    "darunter liegt, lässt sich in dieser Kohorte nicht von "
                    "Konfundierung trennen."
                    if floor is not None and np.isfinite(float(floor))
                    else "."
                )
            )

        if notes:
            self.targeted_design_summary.setText(" ".join(notes))
        elif present:
            self.targeted_design_summary.setText(
                f"{present} von 5 zielgerichteten Auswertungen vorhanden."
            )
        else:
            self.targeted_design_summary.setText(
                "Noch keine zielgerichteten Auswertungen für diesen Lauf. "
                "Alle fünf lassen sich einzeln nachtragen und rechnen auf "
                "den bereits extrahierten Merkmalen."
            )

    def _result_filter(self) -> tuple[str, tuple]:
        clauses = []
        parameters = []
        endpoint = str(self.endpoint_filter.currentData() or "")
        panel = str(self.panel_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "")
        if endpoint:
            clauses.append("endpoint_id = ?")
            parameters.append(endpoint)
        if panel:
            clauses.append("panel_id = ?")
            parameters.append(panel)
        if split:
            clauses.append("evaluation_split = ?")
            parameters.append(split)
        return " AND ".join(clauses), tuple(parameters)

    def _screening_result_filter(self) -> tuple[str, tuple]:
        where, parameters = self._result_filter()
        inference_status = str(
            self.inference_filter.currentData() or ""
        )
        clauses = [where] if where else []
        values = list(parameters)
        if inference_status:
            clauses.append("inference_status = ?")
            values.append(inference_status)
        controls = sorted(METHODOLOGICAL_CONTROL_FEATURE_IDS)
        clauses.append(
            "feature_id NOT IN ("
            + ", ".join("?" for _ in controls)
            + ")"
        )
        values.extend(controls)
        return " AND ".join(clauses), tuple(values)

    def _phenome_where(self) -> tuple[str, tuple]:
        value = str(self.phenome_domain_combo.currentData() or "")
        if not value:
            return "", ()
        if value in {
            "pathology_hypothesis",
            "future_outcome_hypothesis",
        }:
            return "discovery_class = ?", (value,)
        return "discovery_domain = ?", (value,)

    def _endpoint_where(self) -> tuple[str, tuple]:
        endpoint = str(self.endpoint_filter.currentData() or "")
        return (
            ("endpoint_id = ?", (endpoint,))
            if endpoint
            else ("", ())
        )

    def _baseline_comparison_where(self) -> tuple[str, tuple]:
        where, parameters = self._endpoint_where()
        verdict = str(self.baseline_verdict_filter.currentData() or "")
        if not verdict:
            return where, parameters
        clauses = [where] if where else []
        clauses.append("comparison_verdict = ?")
        return " AND ".join(clauses), (*parameters, verdict)

    def _endpoint_split_where(self) -> tuple[str, tuple]:
        clauses = []
        parameters = []
        endpoint = str(self.endpoint_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "")
        if endpoint:
            clauses.append("endpoint_id = ?")
            parameters.append(endpoint)
        if split:
            clauses.append("evaluation_split = ?")
            parameters.append(split)
        return " AND ".join(clauses), tuple(parameters)

    def _update_result_plots(
        self,
        screening_path: Path,
        comparison_path: Path,
        stability_path: Path,
        phenome_path: Path,
        curve_path: Path,
        distribution_path: Path,
        operating_path: Path,
        adjusted_path: Path,
        decision_path: Path,
        transfer_path: Path,
    ) -> None:
        self.matrix_plot.clear()
        if screening_path.is_file():
            frame = pd.read_parquet(screening_path)
            frame = self._filter_frame(frame)
            self._draw_marker_endpoint_matrix(frame)
        self._draw_baseline_forest(comparison_path)
        self._draw_stability(stability_path)
        self._draw_performance_curves(curve_path)
        self._draw_score_distributions(distribution_path)
        self._draw_operating_points(operating_path)
        self._draw_sensitivity_analysis(stability_path)
        self._draw_adjusted_models(adjusted_path, decision_path)
        self._draw_phenome(phenome_path)
        self._draw_transfer(transfer_path)

    def _filter_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        endpoint = str(self.endpoint_filter.currentData() or "")
        panel = str(self.panel_filter.currentData() or "")
        split = str(self.split_filter.currentData() or "")
        result = frame
        if endpoint and "endpoint_id" in result:
            result = result.loc[result["endpoint_id"].astype(str).eq(endpoint)]
        if panel and "panel_id" in result:
            result = result.loc[result["panel_id"].astype(str).eq(panel)]
        if split and "evaluation_split" in result:
            result = result.loc[
                result["evaluation_split"].astype(str).eq(split)
            ]
        inference_status = str(
            self.inference_filter.currentData() or ""
        )
        if inference_status and "inference_status" in result:
            result = result.loc[
                result["inference_status"]
                .astype(str)
                .eq(inference_status)
            ]
        if "feature_id" in result:
            result = result.loc[
                result["feature_id"].map(is_candidate_biomarker)
            ]
        return result

    def _draw_marker_endpoint_matrix(self, frame: pd.DataFrame) -> None:
        if not {"feature_id", "endpoint_id", "roc_auc"}.issubset(frame):
            return
        pivot = frame.pivot_table(
            index="feature_id",
            columns="endpoint_id",
            values="roc_auc",
            aggfunc="max",
        )
        if pivot.empty:
            return
        pivot = pivot.loc[
            pivot.max(axis=1).sort_values(ascending=False).head(40).index
        ]
        image = pg.ImageItem(auc_matrix_rgba(pivot.to_numpy(dtype=float).T))
        self.matrix_plot.addItem(image)
        self.matrix_plot.setToolTip(
            "Farbig: gemessene ROC-AUC (0,5 = keine Trennung). "
            "Transparente Zellen: nicht berechnet; kein Neutralwert."
        )
        self.matrix_plot.getAxis("bottom").setTicks(
            [[(index + 0.5, name) for index, name in enumerate(pivot.index)]]
        )
        self.matrix_plot.getAxis("left").setTicks(
            [[(index + 0.5, name) for index, name in enumerate(pivot.columns)]]
        )
        self.matrix_plot.setLabel("bottom", "Marker")
        self.matrix_plot.setLabel("left", "Endpunkt")

    def _draw_baseline_forest(self, path: Path) -> None:
        self.baseline_plot.clear()
        self.baseline_plot.addLine(
            x=0,
            pen=pg.mkPen("#555555", style=Qt.PenStyle.DashLine),
        )
        if not path.is_file():
            return
        frame = comparison_effect_frame(
            self._filter_frame(pd.read_parquet(path))
        )
        if frame.empty:
            return
        frame = frame.reindex(
            frame["effect_value"].abs().sort_values(ascending=False).index
        ).head(400)
        self.baseline_plot.setMinimumHeight(max(360, 80 + 16 * len(frame)))
        values = frame["effect_value"].to_numpy(dtype=float)
        y = np.arange(len(frame))
        low = frame["effect_ci_low"].to_numpy(dtype=float)
        high = frame["effect_ci_high"].to_numpy(dtype=float)
        finite_ci = np.isfinite(low) & np.isfinite(high)
        if finite_ci.any():
            self.baseline_plot.addItem(
                pg.ErrorBarItem(
                    x=values[finite_ci],
                    y=y[finite_ci],
                    left=np.maximum(
                        0.0,
                        values[finite_ci] - low[finite_ci],
                    ),
                    right=np.maximum(
                        0.0,
                        high[finite_ci] - values[finite_ci],
                    ),
                    beam=0.25,
                    pen=pg.mkPen("#5f6b6d"),
                )
            )
        self.baseline_plot.addItem(
            pg.ScatterPlotItem(
                x=values,
                y=y,
                size=8,
                brush=pg.mkBrush("#087e8b"),
                pen=pg.mkPen(None),
            )
        )
        self.baseline_plot.getAxis("left").setTicks(
            [[
                (index, str(value))
                for index, value in enumerate(comparison_effect_labels(frame))
            ]]
        )
        self.baseline_plot.setLabel(
            "bottom",
            "Gepaarte Differenz; positiv spricht für VCG",
        )

    def _draw_stability(self, path: Path) -> None:
        self.stability_plot.clear()
        self.stability_plot.plot(
            [0.5, 1.0],
            [0.5, 1.0],
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        if not path.is_file():
            return
        frame = self._filter_frame(pd.read_parquet(path))
        x_name = next(
            (
                name
                for name in ("train_roc_auc", "train_auc", "development_auc")
                if name in frame
            ),
            None,
        )
        y_name = next(
            (
                name
                for name in (
                    "validation_roc_auc",
                    "validation_auc",
                    "evaluation_auc",
                )
                if name in frame
            ),
            None,
        )
        if not x_name or not y_name:
            return
        x = pd.to_numeric(frame[x_name], errors="coerce")
        y = pd.to_numeric(frame[y_name], errors="coerce")
        valid = x.notna() & y.notna()
        self.stability_plot.addItem(
            pg.ScatterPlotItem(
                x=x[valid].to_numpy(),
                y=y[valid].to_numpy(),
                size=7,
                brush=pg.mkBrush("#d1495b"),
                pen=pg.mkPen(None),
            )
        )

    def _draw_performance_curves(self, path: Path) -> None:
        self.roc_plot.clear()
        self.pr_plot.clear()
        self.roc_plot.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        self.roc_plot.setXRange(0.0, 1.0, padding=0.01)
        self.roc_plot.setYRange(0.0, 1.0, padding=0.01)
        self.pr_plot.setXRange(0.0, 1.0, padding=0.01)
        self.pr_plot.setYRange(0.0, 1.0, padding=0.01)
        if not path.is_file():
            self.curve_context.setText(
                "Für diesen älteren Run wurden noch keine Kurvenartefakte erzeugt."
            )
            return
        frame = self._filter_frame(pd.read_parquet(path))
        if frame.empty or not {"curve_type", "x", "y"}.issubset(frame):
            return
        point_kind = (
            frame["point_kind"].astype(str)
            if "point_kind" in frame
            else pd.Series("", index=frame.index, dtype=str)
        )
        continuous = frame.loc[
            point_kind.eq("continuous_marker_curve")
        ].copy()
        group_columns = [
            column
            for column in (
                "endpoint_id",
                "feature_id",
                "evaluation_split",
            )
            if column in continuous
        ]
        curve_groups = (
            list(continuous.groupby(group_columns, sort=True))
            if group_columns and not continuous.empty
            else []
        )
        endpoint_selected = bool(self.endpoint_filter.currentData())
        curve_groups = curve_groups[:12 if endpoint_selected else 6]
        context_lines = []
        for index, (group_key, group) in enumerate(curve_groups):
            key_values = (
                group_key if isinstance(group_key, tuple) else (group_key,)
            )
            key = dict(zip(group_columns, key_values))
            label = (
                f"{key.get('endpoint_id', '')} | "
                f"{key.get('feature_id', '')}"
            )
            legend_label = (
                str(key.get("feature_id", ""))
                if endpoint_selected
                else f"{key.get('endpoint_id', '')} | {key.get('feature_id', '')}"
            )
            color = _indexed_color(index)
            for curve_type, plot in (
                ("roc", self.roc_plot),
                ("precision_recall", self.pr_plot),
            ):
                points = group.loc[
                    group["curve_type"].astype(str).eq(curve_type)
                ]
                if "point_order" in points:
                    points = points.sort_values("point_order")
                if points.empty:
                    continue
                plot.plot(
                    pd.to_numeric(points["x"], errors="coerce").to_numpy(),
                    pd.to_numeric(points["y"], errors="coerce").to_numpy(),
                    pen=pg.mkPen(color, width=2),
                    name=legend_label,
                )
            first = group.iloc[0]
            context_lines.append(
                f"{label}: AUC {_format_ci(first, 'roc_auc')}; "
                f"AP {_format_ci(first, 'average_precision')}; "
                f"{first.get('selection_role', '')}"
            )
        references = frame.loc[point_kind.eq("fixed_reference_point")]
        for curve_type, plot in (
            ("roc", self.roc_plot),
            ("precision_recall", self.pr_plot),
        ):
            points = references.loc[
                references["curve_type"].astype(str).eq(curve_type)
            ]
            if points.empty:
                continue
            plot.addItem(
                pg.ScatterPlotItem(
                    x=pd.to_numeric(points["x"], errors="coerce").to_numpy(),
                    y=pd.to_numeric(points["y"], errors="coerce").to_numpy(),
                    symbol="x",
                    size=11,
                    pen=pg.mkPen("#20272b", width=2),
                    brush=pg.mkBrush(None),
                )
            )
        self.curve_context.setText(
            "\n".join(context_lines)
            if context_lines
            else "Keine auswertbare Kurve für die aktuelle Filterung."
        )

    def _draw_score_distributions(self, path: Path) -> None:
        self.distribution_plot.clear()
        if not path.is_file():
            self.distribution_context.setText(
                "Für diesen Run sind keine Scoreverteilungen vorhanden."
            )
            return
        frame = self._filter_frame(pd.read_parquet(path))
        required = {
            "endpoint_id",
            "feature_id",
            "label",
            "bin_center",
            "fraction_within_class",
        }
        if frame.empty or not required.issubset(frame):
            return
        selection_role = (
            frame["selection_role"].astype(str)
            if "selection_role" in frame
            else pd.Series("", index=frame.index, dtype=str)
        )
        selection = (
            frame.assign(
                _role_rank=selection_role.map(
                    {
                        "prespecified_primary_marker": 0,
                        "train_primary_core_representative": 1,
                        "train_sentinel_representative": 2,
                    }
                )
                .fillna(9)
            )
            .sort_values(["_role_rank", "endpoint_id", "feature_id"])
            .iloc[0]
        )
        selected = frame.loc[
            frame["endpoint_id"].astype(str).eq(str(selection["endpoint_id"]))
            & frame["feature_id"].astype(str).eq(str(selection["feature_id"]))
        ]
        for label, color, name in (
            (0, "#4c78a8", "Negativ"),
            (1, "#d1495b", "Positiv"),
        ):
            values = selected.loc[selected["label"].eq(label)]
            values = values.sort_values(
                "bin_index" if "bin_index" in values else "bin_center"
            )
            if values.empty:
                continue
            self.distribution_plot.plot(
                pd.to_numeric(
                    values["bin_center"],
                    errors="coerce",
                ).to_numpy(),
                pd.to_numeric(
                    values["fraction_within_class"],
                    errors="coerce",
                ).to_numpy(),
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=pg.mkBrush(color),
                name=name,
            )
        self.distribution_context.setText(
            f"{selection['endpoint_id']} | {selection['feature_id']} | "
            f"{selection.get('evaluation_split', '')} | "
            f"{selection.get('selection_role', '')}. Werte außerhalb ±5 SD "
            "werden nur für die Darstellung in die Randklasse gelegt."
        )

    def _draw_operating_points(self, path: Path) -> None:
        self.operating_plot.clear()
        self.operating_plot.setXRange(0.0, 1.0, padding=0.01)
        self.operating_plot.setYRange(0.0, 1.0, padding=0.01)
        if not path.is_file():
            return
        frame = self._filter_frame(pd.read_parquet(path))
        required = {"operating_point", "specificity", "sensitivity"}
        if frame.empty or not required.issubset(frame):
            return
        specificity = pd.to_numeric(frame["specificity"], errors="coerce")
        sensitivity = pd.to_numeric(frame["sensitivity"], errors="coerce")
        valid = specificity.notna() & sensitivity.notna()
        frame = frame.loc[valid].head(500).reset_index(drop=True)
        specificity = pd.to_numeric(
            frame["specificity"], errors="coerce"
        ).to_numpy()
        sensitivity = pd.to_numeric(
            frame["sensitivity"], errors="coerce"
        ).to_numpy()
        for index, (operating_point, group) in enumerate(
            frame.groupby("operating_point", sort=True)
        ):
            indices = group.index.to_numpy(dtype=int)
            symbol = (
                "x"
                if str(operating_point) == "fixed_reference_point"
                else "o"
            )
            color = _indexed_color(index)
            self.operating_plot.addItem(
                pg.ScatterPlotItem(
                    x=specificity[indices],
                    y=sensitivity[indices],
                    symbol=symbol,
                    size=10,
                    pen=pg.mkPen(color, width=2),
                    brush=(
                        pg.mkBrush(None)
                        if symbol == "x"
                        else pg.mkBrush(color)
                    ),
                )
            )
        low_x = _numeric_array_or_nan(frame, "specificity_ci_low")
        high_x = _numeric_array_or_nan(frame, "specificity_ci_high")
        low_y = _numeric_array_or_nan(frame, "sensitivity_ci_low")
        high_y = _numeric_array_or_nan(frame, "sensitivity_ci_high")
        finite = (
            np.isfinite(low_x)
            & np.isfinite(high_x)
            & np.isfinite(low_y)
            & np.isfinite(high_y)
        )
        if finite.any():
            self.operating_plot.addItem(
                pg.ErrorBarItem(
                    x=specificity[finite],
                    y=sensitivity[finite],
                    left=np.maximum(0.0, specificity[finite] - low_x[finite]),
                    right=np.maximum(0.0, high_x[finite] - specificity[finite]),
                    bottom=np.maximum(0.0, sensitivity[finite] - low_y[finite]),
                    top=np.maximum(0.0, high_y[finite] - sensitivity[finite]),
                    beam=0.01,
                    pen=pg.mkPen("#7a8589"),
                )
            )

    def _draw_sensitivity_analysis(self, path: Path) -> None:
        self.sensitivity_plot.clear()
        self.sensitivity_plot.addLine(
            x=0.5,
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        if not path.is_file():
            return
        frame = pd.read_parquet(path)
        endpoint = str(self.endpoint_filter.currentData() or "")
        if endpoint and "endpoint_id" in frame:
            frame = frame.loc[frame["endpoint_id"].astype(str).eq(endpoint)]
        if "analysis_type" not in frame:
            return
        frame = frame.loc[
            ~frame["analysis_type"].astype(str).eq("single_marker")
        ].copy()
        values = pd.Series(np.nan, index=frame.index, dtype=float)
        for column in ("validation_auc", "roc_auc"):
            if column in frame:
                values = values.fillna(
                    pd.to_numeric(frame[column], errors="coerce")
                )
        frame = frame.loc[values.notna()].copy().head(400)
        values = values.loc[frame.index].to_numpy()
        if frame.empty:
            return
        self.sensitivity_plot.setMinimumHeight(max(360, 80 + 16 * len(frame)))
        labels = frame.apply(_sensitivity_row_label, axis=1)
        y = np.arange(len(frame))
        self.sensitivity_plot.addItem(
            pg.ScatterPlotItem(
                x=values,
                y=y,
                size=8,
                brush=pg.mkBrush("#d1495b"),
                pen=pg.mkPen(None),
            )
        )
        ci_low = _numeric_array_or_nan(frame, "roc_auc_ci_low")
        ci_high = _numeric_array_or_nan(frame, "roc_auc_ci_high")
        finite_ci = (
            np.isfinite(ci_low)
            & np.isfinite(ci_high)
            & np.isfinite(values)
        )
        if finite_ci.any():
            self.sensitivity_plot.addItem(
                pg.ErrorBarItem(
                    x=values[finite_ci],
                    y=y[finite_ci],
                    left=np.maximum(0.0, values[finite_ci] - ci_low[finite_ci]),
                    right=np.maximum(0.0, ci_high[finite_ci] - values[finite_ci]),
                    beam=0.12,
                    pen=pg.mkPen("#7a8589"),
                )
            )
        self.sensitivity_plot.getAxis("left").setTicks(
            [[(index, str(label)) for index, label in enumerate(labels)]]
        )

    def _draw_evidence_ladder(self, path: Path) -> None:
        self.evidence_ladder_plot.clear()
        levels = (
            "exploratory",
            "replicated_association",
            "guideline_candidate",
            "clinically_validated",
        )
        labels = (
            "Explorativ",
            "Repliziert",
            "Kriterienkandidat",
            "Klinisch validiert",
        )
        counts = np.zeros(len(levels), dtype=float)
        frame = pd.DataFrame()
        if path.is_file():
            frame = pd.read_parquet(path)
            if "evidence_tier" in frame:
                observed = frame["evidence_tier"].astype(str).value_counts()
                counts = np.asarray(
                    [float(observed.get(level, 0)) for level in levels]
                )
        colors = ("#8b949e", "#168aad", "#f4a261", "#2a9d8f")
        self.evidence_ladder_plot.addItem(
            pg.BarGraphItem(
                x=np.arange(len(levels)),
                height=counts,
                width=0.68,
                brushes=[pg.mkBrush(color) for color in colors],
                pens=[pg.mkPen(color) for color in colors],
            )
        )
        self.evidence_ladder_plot.getAxis("bottom").setTicks(
            [[(index, label) for index, label in enumerate(labels)]]
        )
        failed = 0
        if not frame.empty and "failed_gates" in frame:
            failed = int(frame["failed_gates"].astype(str).ne("").sum())
        self.evidence_ladder_summary.setText(
            f"{len(frame):,} Kriterien bewertet; "
            f"{int(counts[2]):,} Kriterienkandidaten; "
            f"{int(counts[3]):,} extern klinisch validiert. "
            f"Bei {failed:,} Kriterien sind fehlende Gates explizit ausgewiesen."
        )

    def _draw_transform_agreement(self, path: Path, *, reload: bool = True) -> None:
        previous_source = getattr(self, "_transform_agreement_source", None)
        self._transform_agreement_source = path
        self.transform_agreement_plot.clear()
        self.transform_agreement_plot.getAxis("left").setTicks([])
        self.transform_agreement_plot.setMinimumHeight(300)
        self.transform_agreement_plot.setTitle("Keine auswertbaren Methodenpaare")
        self.transform_agreement_summary.setText("Keine Transformationsdaten geladen.")
        if reload or previous_source != path or not hasattr(self, "_transform_agreement_frame"):
            # Only an explicit refresh reloads the result artifact. Marker
            # navigation uses this immutable display snapshot, never a stale
            # cross-run cache or repeated Parquet I/O.
            self._transform_agreement_frame = pd.DataFrame()
            if path.is_file():
                self._transform_agreement_frame = pd.read_parquet(path)
        frame = self._transform_agreement_frame
        required = {
            "marker_id",
            "bias",
            "limit_of_agreement_low",
            "limit_of_agreement_high",
        }
        if frame.empty or not required.issubset(frame):
            self.transform_marker_combo.blockSignals(True)
            self.transform_marker_combo.clear()
            self.transform_marker_combo.blockSignals(False)
            return
        previous = self.transform_marker_combo.currentText()
        self.transform_marker_combo.blockSignals(True)
        self.transform_marker_combo.clear()
        self.transform_marker_combo.addItems(sorted(frame.marker_id.dropna().astype(str).unique()))
        if self.transform_marker_combo.findText(previous) >= 0:
            self.transform_marker_combo.setCurrentText(previous)
        self.transform_marker_combo.blockSignals(False)
        frame = frame.loc[frame.marker_id.astype(str).eq(self.transform_marker_combo.currentText())].copy()
        paired = pd.to_numeric(frame.get("paired_count", pd.Series(dtype=float)), errors="coerce")
        index_n = pd.to_numeric(frame.get("index_record_count", pd.Series(dtype=float)), errors="coerce")
        availability = pd.to_numeric(frame.get("paired_availability_fraction", pd.Series(dtype=float)), errors="coerce")
        bias = _numeric_array_or_nan(frame, "bias")
        low = _numeric_array_or_nan(frame, "limit_of_agreement_low")
        high = _numeric_array_or_nan(frame, "limit_of_agreement_high")
        valid = np.isfinite(bias) & np.isfinite(low) & np.isfinite(high) & (low <= bias) & (bias <= high)
        if not valid.any():
            statuses = ", ".join(sorted(frame.get("status", pd.Series(dtype=str)).dropna().astype(str).unique())) or "nicht schätzbar"
            n_text = f"; größte Paarzahl n={int(paired.max())}" if paired.notna().any() else ""
            denominator = f" von {int(index_n.max())} Indexaufnahmen" if index_n.notna().any() else ""
            availability_text = f" ({availability.max():.1%} gepaart)" if availability.notna().any() else ""
            self.transform_agreement_summary.setText(
                f"Keine gültigen Übereinstimmungsgrenzen: {statuses}{n_text}{denominator}{availability_text}. "
                "Fehlende Grenzen sind keine Übereinstimmung und kein Nulleffekt."
            )
            return
        frame = frame.loc[valid]
        bias, low, high = bias[valid], low[valid], high[valid]
        selected_availability = availability.loc[frame.index] if not availability.empty else availability
        availability_text = (
            f"; Messpaar-Verfügbarkeit {selected_availability.min():.1%}–{selected_availability.max():.1%}"
            if selected_availability.notna().any() else ""
        )
        self.transform_agreement_summary.setText(
            f"{len(frame)} Methodenpaar(e) mit gültigen Grenzen{availability_text}. "
            "Deskriptive Differenzen; keine klinische Austauschbarkeitsfreigabe."
        )
        self.transform_agreement_plot.setTitle("Bias und 95-%-Übereinstimmungsgrenzen — keine Austauschbarkeitsfreigabe")
        self.transform_agreement_plot.setLabel("bottom", "Vergleich minus Primärmethode · Einheit des ausgewählten Markers")
        self.transform_agreement_plot.addLine(x=0.0, pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine))
        y = np.arange(len(frame))
        self.transform_agreement_plot.setYRange(-0.5, len(frame) - 0.5, padding=0)
        self.transform_agreement_plot.addItem(
            pg.ErrorBarItem(
                x=bias,
                y=y,
                left=np.maximum(0.0, bias - low),
                right=np.maximum(0.0, high - bias),
                beam=0.15,
                pen=pg.mkPen("#168aad"),
            )
        )
        self.transform_agreement_plot.addItem(
            pg.ScatterPlotItem(
                x=bias,
                y=y,
                size=8,
                brush=pg.mkBrush("#f4a261"),
                pen=pg.mkPen(None),
            )
        )
        self.transform_agreement_plot.getAxis("left").setTicks(
            [[
                (index, f"{row.get('comparison_transform', '?')} − {row.get('primary_transform', '?')} · n={row.get('paired_count', '?')}")
                for index, (_, row) in enumerate(frame.iterrows())
            ]]
        )
        self.transform_agreement_plot.setMinimumHeight(
            max(300, 70 + 18 * len(frame))
        )

    def _refresh_gwas_context(self) -> None:
        root = self.config.work_root / "genetics"
        summary_path = root / "gwas_feasibility.json"
        inventory_path = root / "gwas_source_inventory.parquet"
        presentation, _ = _gwas_context_presentation(
            _read_json(summary_path)
        )
        self.gwas_context_summary.setText(presentation)
        self.gwas_inventory_table.set_source(
            inventory_path if inventory_path.is_file() else None
        )

    def _draw_population_mortality(self, path: Path) -> None:
        self.population_mortality_plot.clear()
        self.population_mortality_plot.addLine(
            x=0.0,
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        forest_ticks = np.asarray(
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            dtype=float,
        )
        self.population_mortality_plot.getAxis("bottom").setTicks(
            [[
                (float(np.log2(value)), f"{value:g}")
                for value in forest_ticks
            ]]
        )
        self.population_mortality_plot.setXRange(-3.2, 3.2, padding=0.0)
        self.population_mortality_plot.setLabel(
            "bottom",
            "Hazard Ratio je SD (logarithmisch; Orange = Intervall ausserhalb 0,125-8)",
        )
        if not path.is_file():
            self.population_cohort_summary.setText(
                "CODE-15 und SaMi-Trop sind konfiguriert; die gemeinsame "
                "externe Merkmalsextraktion wurde für diesen Run noch nicht ausgeführt."
            )
            return
        frame = pd.read_parquet(path)
        estimable = frame.loc[
            frame.get("status", pd.Series(index=frame.index, dtype=str))
            .astype(str)
            .eq("estimable")
        ].copy()
        if estimable.empty:
            self.population_cohort_summary.setText(
                "Externe Kohorten wurden gelesen, aber noch kein "
                "Mortalitätsmodell war schätzbar."
            )
            return
        estimable["q_value_cohort"] = pd.to_numeric(
            estimable.get("q_value_cohort"), errors="coerce"
        )
        estimable = estimable.sort_values(
            ["q_value_cohort", "cohort", "marker_id"],
            na_position="last",
        ).head(50)
        hazard = _numeric_array_or_nan(estimable, "hazard_ratio_per_sd")
        low = _numeric_array_or_nan(estimable, "ci_low")
        high = _numeric_array_or_nan(estimable, "ci_high")
        valid, x, left, right, clipped_low, clipped_high = (
            _population_forest_coordinates(hazard, low, high)
        )
        estimable = estimable.loc[valid]
        hazard, low, high = hazard[valid], low[valid], high[valid]
        y = np.arange(len(estimable))
        self.population_mortality_plot.addItem(
            pg.ErrorBarItem(
                x=x,
                y=y,
                left=left,
                right=right,
                beam=0.15,
                pen=pg.mkPen("#168aad"),
            )
        )
        clipped = clipped_low | clipped_high
        for mask, brush in (
            (~clipped, "#d1495b"),
            (clipped, "#f4a261"),
        ):
            if np.any(mask):
                self.population_mortality_plot.addItem(
                    pg.ScatterPlotItem(
                        x=x[mask],
                        y=y[mask],
                        size=8,
                        brush=pg.mkBrush(brush),
                        pen=pg.mkPen(None),
                    )
                )
        labels = []
        for index, row in enumerate(estimable.itertuples(index=False)):
            clipping_note = ""
            if clipped_low[index] and clipped_high[index]:
                clipping_note = " · CI ↔"
            elif clipped_low[index]:
                clipping_note = " · CI ←"
            elif clipped_high[index]:
                clipping_note = " · CI →"
            if hazard[index] < POPULATION_FOREST_HR_MIN:
                clipping_note += f" HR {hazard[index]:.3g}"
            elif hazard[index] > POPULATION_FOREST_HR_MAX:
                clipping_note += f" HR {hazard[index]:.3g}"
            labels.append(
                f"{row.cohort} · {biomarker_display_name(str(row.marker_id))}"
                f"{clipping_note}"
            )
        self.population_mortality_plot.getAxis("left").setTicks(
            [[(index, label) for index, label in enumerate(labels)]]
        )
        self.population_mortality_plot.setMinimumHeight(
            max(320, 80 + 18 * len(estimable))
        )
        cohorts = ", ".join(sorted(estimable["cohort"].astype(str).unique()))
        significant = int(
            pd.to_numeric(estimable["q_value_cohort"], errors="coerce")
            .le(0.05)
            .sum()
        )
        self.population_cohort_summary.setText(
            f"{len(estimable):,} schätzbare Marker-Mortalitätsmodelle "
            f"({cohorts}); {significant:,} nach kohortenspezifischer FDR. "
            f"{int(clipped.sum()):,} Konfidenzintervalle werden an der "
            "robusten Darstellungsgrenze markiert. Alle Modelle sind für "
            "Alter und Geschlecht adjustiert."
        )

    def _open_pilot_workspace(self) -> None:
        from .pilot_workspace_widget import PilotWorkspaceDialog

        parent = self.config.work_root / "pilot_workspaces"
        candidates = sorted(parent.glob("*/workspace.json"))
        if not candidates:
            QMessageBox.information(self, "Pilot-Testarbeitsbereich", "Noch keine verifizierte Pilotkopie vorhanden.")
            return
        try:
            dialog = PilotWorkspaceDialog(candidates[-1].parent, self)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.show()
            self._pilot_workspace_dialog = dialog
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.exception("Pilot workspace could not be opened")
            QMessageBox.warning(self, "Pilot-Testarbeitsbereich", str(exc))

    def _draw_adjusted_models(
        self,
        adjusted_path: Path,
        decision_path: Path,
    ) -> None:
        self.calibration_plot.clear()
        self.decision_plot.clear()
        self.calibration_plot.setTitle("Kalibration: nicht berechnet / im Pilot nicht zulässig")
        self.decision_plot.setTitle("Entscheidungsnutzen: nicht berechnet / keine klinische Freigabe")
        self.calibration_plot.setXRange(0.0, 1.0, padding=0.02)
        self.calibration_plot.setYRange(0.0, 1.0, padding=0.02)
        calibration_path = adjusted_path.with_name(
            "calibration_curve_points.parquet"
        )
        if calibration_path.is_file():
            calibration = self._filter_frame(pd.read_parquet(calibration_path))
            required = {"model_id", "mean_predicted", "observed_rate"}
            if required.issubset(calibration):
                if not calibration.empty:
                    self.calibration_plot.setTitle("Beobachtet gegen vorhergesagt · mit Unsicherheit")
                    self.calibration_plot.plot(np.array([0., 1.]), np.array([0., 1.]), pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine))
                for index, (_, group) in enumerate(
                    calibration.groupby("model_id", sort=True)
                ):
                    group = group.sort_values("mean_predicted")
                    predicted = _numeric_array_or_nan(group, "mean_predicted")
                    observed = _numeric_array_or_nan(group, "observed_rate")
                    valid = np.isfinite(predicted) & np.isfinite(observed)
                    color = _indexed_color(index)
                    self.calibration_plot.plot(
                        predicted[valid],
                        observed[valid],
                        pen=pg.mkPen(color, width=2),
                        symbol="o",
                        symbolSize=7,
                        symbolBrush=pg.mkBrush(color),
                    )
                    ci_low = _numeric_array_or_nan(group, "observed_ci_low")
                    ci_high = _numeric_array_or_nan(group, "observed_ci_high")
                    finite_ci = valid & np.isfinite(ci_low) & np.isfinite(ci_high)
                    if finite_ci.any():
                        self.calibration_plot.addItem(
                            pg.ErrorBarItem(
                                x=predicted[finite_ci],
                                y=observed[finite_ci],
                                bottom=np.maximum(
                                    0.0, observed[finite_ci] - ci_low[finite_ci]
                                ),
                                top=np.maximum(
                                    0.0, ci_high[finite_ci] - observed[finite_ci]
                                ),
                                beam=0.012,
                                pen=pg.mkPen(color),
                            )
                        )
        if not decision_path.is_file():
            return
        decision = self._filter_frame(pd.read_parquet(decision_path))
        required = {"model_id", "threshold_probability", "net_benefit"}
        if decision.empty or not required.issubset(decision):
            return
        self.decision_plot.setTitle("Entscheidungsnutzen · Forschungsmodell")
        for index, (model_id, group) in enumerate(
            decision.groupby("model_id", sort=True)
        ):
            group = group.sort_values("threshold_probability")
            self.decision_plot.plot(
                pd.to_numeric(
                    group["threshold_probability"],
                    errors="coerce",
                ).to_numpy(),
                pd.to_numeric(
                    group["net_benefit"],
                    errors="coerce",
                ).to_numpy(),
                pen=pg.mkPen(_indexed_color(index), width=2),
            )
        reference = decision.sort_values("threshold_probability")
        if not reference.empty:
            self.decision_plot.plot(
                pd.to_numeric(
                    reference["threshold_probability"],
                    errors="coerce",
                ).to_numpy(),
                pd.to_numeric(
                    reference["treat_all_net_benefit"],
                    errors="coerce",
                ).to_numpy(),
                pen=pg.mkPen(
                    "#5f6b6d",
                    style=Qt.PenStyle.DashLine,
                ),
            )
            self.decision_plot.addLine(y=0.0, pen=pg.mkPen("#20272b"))

    def _draw_phenome(self, path: Path) -> None:
        self.phenome_plot.clear()
        if not path.is_file():
            return
        frame = pd.read_parquet(path)
        selected_domain = str(
            self.phenome_domain_combo.currentData() or ""
        )
        if selected_domain and "discovery_domain" in frame:
            if (
                selected_domain
                in {
                    "pathology_hypothesis",
                    "future_outcome_hypothesis",
                }
                and "discovery_class" in frame
            ):
                frame = frame.loc[
                    frame["discovery_class"].astype(str)
                    == selected_domain
                ]
            else:
                frame = frame.loc[
                    frame["discovery_domain"].astype(str)
                    == selected_domain
                ]
        if frame.empty:
            return
        effect_name = next(
            (
                name
                for name in (
                    "validation_effect",
                    "effect_size",
                    "rank_biserial",
                )
                if name in frame
            ),
            None,
        )
        q_name = next(
            (
                name
                for name in ("validation_q_value", "q_value")
                if name in frame
            ),
            None,
        )
        if not effect_name or not q_name:
            return
        effect = pd.to_numeric(frame[effect_name], errors="coerce")
        q_value = pd.to_numeric(frame[q_name], errors="coerce")
        valid = effect.notna() & q_value.gt(0)
        self.phenome_plot.addItem(
            pg.ScatterPlotItem(
                x=effect[valid].to_numpy(),
                y=-np.log10(q_value[valid].to_numpy()),
                size=7,
                brush=pg.mkBrush("#f0a202"),
                pen=pg.mkPen(None),
            )
        )

    def _draw_transfer(self, path: Path) -> None:
        self.transfer_plot.clear()
        self.transfer_plot.addLine(
            x=0.0,
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        self.transfer_plot.addLine(
            y=0.0,
            pen=pg.mkPen("#778388", style=Qt.PenStyle.DashLine),
        )
        if not path.is_file():
            return
        frame = pd.read_parquet(path)
        required = {
            "ptb_validation_rank_biserial",
            "mimic_validation_rank_biserial",
        }
        if frame.empty or not required.issubset(frame):
            return
        x_values = pd.to_numeric(
            frame["ptb_validation_rank_biserial"],
            errors="coerce",
        )
        y_values = pd.to_numeric(
            frame["mimic_validation_rank_biserial"],
            errors="coerce",
        )
        valid = x_values.notna() & y_values.notna()
        if not valid.any():
            return
        transfer_type = frame.get(
            "transfer_type",
            pd.Series("shared_single_marker", index=frame.index),
        ).astype(str)
        agreement = frame.get(
            "train_direction_agreement",
            pd.Series(pd.NA, index=frame.index),
        )
        for score_rows, symbol in ((False, "o"), (True, "s")):
            selected = (
                valid
                & transfer_type.eq("ptb_frozen_score").eq(score_rows)
            )
            if not selected.any():
                continue
            colors = []
            for value in agreement.loc[selected]:
                if pd.isna(value):
                    colors.append(pg.mkBrush("#778388"))
                elif bool(value):
                    colors.append(pg.mkBrush("#16803a"))
                else:
                    colors.append(pg.mkBrush("#d97706"))
            self.transfer_plot.addItem(
                pg.ScatterPlotItem(
                    x=x_values.loc[selected].to_numpy(),
                    y=y_values.loc[selected].to_numpy(),
                    size=9 if score_rows else 7,
                    symbol=symbol,
                    brush=colors,
                    pen=pg.mkPen("#ffffff", width=0.7),
                )
            )

    def freeze_selected_run(self) -> None:
        run_id = self._selected_current_run_id(self.results_run_combo)
        if not run_id:
            return
        answer = QMessageBox.question(
            self,
            "Entwicklungsstand einfrieren",
            (
                f"Run {run_id!r} unveränderlich einfrieren? "
                "Dies liest oder bewertet den Locked Test nicht."
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            record = self.job_store.create(
                stage="freeze",
                project_file=self.config.project_file,
                arguments={"run_id": run_id},
                priority=90,
                requested_by="gui",
            )
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.workspace_tabs.setCurrentIndex(TAB_ANALYSIS)
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(self, "Freeze", str(exc))

    # ------------------------------------------------------------------
    # ECG lab
    # ------------------------------------------------------------------

    def refresh_cases(self, preferred_case_id: Optional[str] = None) -> None:
        current = preferred_case_id if preferred_case_id is not None else self.case_combo.currentData()
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        for case in self.case_library.list_cases():
            self.case_combo.addItem(
                f"{case.pseudonym} · {case.source_format}",
                case.case_id,
            )
        index = self.case_combo.findData(current)
        if index >= 0:
            self.case_combo.setCurrentIndex(index)
        self.case_combo.blockSignals(False)
        if self.case_combo.count():
            self._case_changed()

    def _simulation_target(self):
        lab = self.expert_lab
        if lab.signal is None or lab.case is None:
            raise ValueError("Zuerst ein EKG im Labor laden")
        return (lab.signal, lab.reference.sampling_rate_hz, tuple(lab.case.lead_names),
                {"case_id": lab.case.case_id, "source_sha256": lab.case.source_sha256})

    def _import_simulated_ecg(self, path):
        try:
            case = self.case_library.import_case(path, pseudonym="SYNTHETISCH · Gewebesimulation")
            self.refresh_cases(preferred_case_id=case.case_id)
            self.workspace_tabs.setCurrentIndex(TAB_ECG_LAB)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Synthetisches EKG", str(exc))

    def import_ecg_case(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Lokales 12-Kanal-ECG importieren",
            "",
            (
                "ECG (*.hea *.dat *.csv *.tsv *.txt *.edf *.dcm *.dicom "
                "*.npy *.npz);;Alle Dateien (*)"
            ),
        )
        if not source:
            return
        metadata_dialog = ExternalECGDialog(parent=self)
        if metadata_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        metadata = metadata_dialog.get_meta()
        try:
            case = self.case_library.import_case(
                source,
                metadata=metadata,
            )
        except ECGImportMappingRequired as exc:
            mapping_dialog = LeadMappingDialog(
                exc.available_channels,
                base_meta=metadata,
                parent=self,
            )
            if mapping_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            try:
                case = self.case_library.import_case(
                    source,
                    metadata=mapping_dialog.get_meta(),
                )
            except Exception as second_error:
                QMessageBox.critical(
                    self,
                    "ECG-Import blockiert",
                    str(second_error),
                )
                return
        except Exception as exc:
            QMessageBox.critical(self, "ECG-Import blockiert", str(exc))
            return
        self.refresh_cases()
        index = self.case_combo.findData(case.case_id)
        if index >= 0:
            self.case_combo.setCurrentIndex(index)
        self.analyze_selected_case()

    def _open_dose_pair_ecg(self, pair: dict, side: str) -> None:
        from ..lab_pair_source import resolve_lab_pair_source
        try:
            source = resolve_lab_pair_source(self.config, pair, side)
            case = self.case_library.import_case(source, retain_source_path=False, copy_original=False)
            self.refresh_cases(preferred_case_id=case.case_id)
            if self.case_combo.currentData() != case.case_id:
                raise ValueError("Importierter Fall ist nicht in der Laborbibliothek verfuegbar")
            self.workspace_tabs.setCurrentIndex(3)
            displayed_status = self.case_status.text()
            self.case_status.setText(displayed_status + "\nAus ausgewaehltem Development-Dosis-Paar lokal importiert. "
                "Keine automatische Neuberechnung; bei Bedarf 'ECG analysieren' waehlen. Produktionsresultate unveraendert.")
        except Exception as exc:
            LOGGER.exception("Dose-pair ECG laboratory import rejected")
            QMessageBox.warning(self, "Dosis-Paar: Laborimport blockiert", str(exc))

    def analyze_selected_case(self) -> None:
        case_id = str(self.case_combo.currentData() or "")
        if not case_id:
            return
        try:
            context = self.expert_lab.next_ischemia_context()
            if context is None:
                self.lab_controller.start(self.config, case_id, self.expert_lab.powerline.currentData())
            else:
                self.lab_controller.start(self.config, case_id, self.expert_lab.powerline.currentData(), ischemia_context=context)
        except Exception as exc:
            LOGGER.exception("Local laboratory launch failed")
            QMessageBox.warning(self, "Laboranalyse", str(exc))

    def _lab_busy_changed(self, busy: bool) -> None:
        for control in (self.analyze_case_button, self.import_case_button, self.case_combo, self.edit_fiducials_button):
            control.setEnabled(not busy)
        self.cancel_lab_button.setEnabled(busy)
        self.lab_analysis_combo.setEnabled(not busy)

    def open_ecg_practice(self) -> None:
        """Open isolated exercise UI without touching the production run."""
        from .ecg_practice_dialog import ECGPracticeDialog

        dialog = getattr(self, "practice_dialog", None)
        if dialog is None:
            dialog = ECGPracticeDialog(self.config, self)
            self.practice_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _lab_analysis_selected(self, *_args) -> None:
        case_id = str(self.case_combo.currentData() or "")
        self._lab_analysis_choices[case_id] = self.lab_analysis_combo.currentData()
        self._case_changed()

    def _lab_mode_changed(self, research: bool) -> None:
        for index in self.expert_lab.extra_research_tabs:
            self.expert_lab.tabs.setTabVisible(index, research)
        self.case_pattern_summary.setVisible(research)
        self.case_delineation_status.setVisible(not self.expert_lab.blind.isChecked())

    def _show_results_from_lab(self) -> None:
        """Navigate to population results without equating them to case evidence."""
        for index in range(self.workspace_tabs.count()):
            if self.workspace_tabs.tabText(index) == "Ergebnisse":
                self.workspace_tabs.setCurrentIndex(index)
                break
        self.refresh_results()

    def _apply_expert_fiducials(self, fiducials: dict, native_fs: float, author: str) -> None:
        case_id = str(self.case_combo.currentData() or "")
        if not case_id or self.lab_controller.busy:
            return
        mapped = {key: round(sample * 500.0 / native_fs) for key, sample in fiducials.items()}
        count = round(self.expert_lab.reference.sample_count * 500.0 / native_fs)
        if any(sample < 0 or sample >= count for sample in mapped.values()):
            QMessageBox.warning(self, "Grenzen nicht übertragbar", "Eine native Grenze liegt außerhalb des 500-Hz-Analyserasters. Die lokale Originalmessung bleibt erhalten; keine stille Begrenzung.")
            return
        revisions = self.case_library.annotations(case_id)
        self.case_library.add_annotation(case_id,
            fiducials=mapped,
            notes=f"Expert native-sample revision mapped from {native_fs:g} Hz to legacy 500-Hz analysis coordinates",
            author=author, parent_revision_id=revisions[-1].revision_id if revisions else "")
        self.analyze_selected_case()

    def _display_lab_analysis(self, payload: dict) -> None:
        case_id = str(self.case_combo.currentData() or "")
        if payload.get("case_id") != case_id:
            return
        if self.lab_controller.request and Path(self.lab_controller.request["config"]["work_root"]).resolve() != self.config.work_root.resolve():
            return
        identifier = str(payload.get("analysis_id", ""))
        self._lab_analysis_choices[case_id] = identifier
        self.lab_analysis_combo.blockSignals(True)
        if self.lab_analysis_combo.findData(identifier) < 0:
            self.lab_analysis_combo.addItem(f"{identifier[:12]} · neue lokale Analyse", identifier)
        self.lab_analysis_combo.setCurrentIndex(self.lab_analysis_combo.findData(identifier))
        self.lab_analysis_combo.blockSignals(False)
        try:
            analysis_root = (
                self.config.cases_root
                / case_id
                / "analyses"
                / str(payload["analysis_id"])[:16]
            )
            self._render_case_arrays(
                analysis_root / "visualization_arrays.npz"
            )
            self._render_case_results(payload)
            if payload.get("analysis_status") == "blocked_signal_quality":
                self.case_status.setText(
                    "Analyse durch Signal-QC blockiert: "
                    + str(payload.get("block_reason", "unzureichende Qualität"))
                )
                self.case_delineation_status.setText(
                    "Keine Wellenmessung: Das Signal hat die vorgeschaltete "
                    "Qualitätsprüfung nicht bestanden."
                )
                self.edit_fiducials_button.setEnabled(False)
            else:
                annotation_text = (
                    " · manuelle Revision angewandt"
                    if payload.get("delineation", {}).get("annotation_applied")
                    else ""
                )
                self.case_status.setText(
                    f"{payload['delineation']['accepted_beat_count']} akzeptierte "
                    f"Beats · Forschungsanalyse {payload['analysis_id'][:12]}"
                    f"{annotation_text} · keine Diagnose."
                )
                inventory = payload.get("score_release_inventory", {})
                stale_count = int(
                    inventory.get("stale_release_count") or 0
                )
                compatible_count = int(
                    inventory.get("compatible_release_count") or 0
                )
                if stale_count and not compatible_count:
                    self.case_status.setText(
                        self.case_status.text()
                        + f" {stale_count} historische Scores wegen "
                        "geändertem Biomarker-Code blockiert."
                    )
                delineation_payload = payload.get("delineation", {})
                rhythm = delineation_payload.get("rhythm_context", {})
                morphology_template = delineation_payload.get(
                    "morphology_template",
                    {},
                )
                review_reasons = delineation_payload.get(
                    "review_reasons",
                    (),
                )
                review_text = (
                    "Manuelle Kontrolle empfohlen: "
                    + ", ".join(str(item) for item in review_reasons)
                    if delineation_payload.get("review_required")
                    else "Automatische Messung ohne zusätzlichen Review-Blocker."
                )
                self.case_delineation_status.setText(
                    "Rhythmuskontext: "
                    f"{rhythm.get('class', 'unknown')} "
                    f"({float(rhythm.get('confidence') or 0.0):.0%}) · "
                    "Fiduzialkonfidenz "
                    f"{float(delineation_payload.get('fiducial_confidence') or 0.0):.0%}. "
                    "Robuste Morphologie: "
                    + (
                        f"Median aus {int(morphology_template.get('used_beat_count') or 0)} "
                        "Schlägen. "
                        if morphology_template.get("status") == "used"
                        else "nicht verfügbar. "
                    )
                    + f"{review_text}"
                )
                rhythm_details = [
                    "Technisches Muster, keine Rhythmusdiagnose.",
                    (
                        "Methode: "
                        f"{delineation_payload.get('method', 'unknown')}"
                    ),
                    (
                        "Messschwelle: "
                        f"{float(delineation_payload.get('minimum_measurement_confidence') or 0.35):.0%}"
                    ),
                    (
                        "Morphologievorlage: "
                        f"{morphology_template.get('status', 'unknown')}; "
                        f"{int(morphology_template.get('used_beat_count') or 0)} "
                        "verwendet, "
                        f"{int(morphology_template.get('excluded_morphology_outlier_count') or 0)} "
                        "Morphologie-Ausreißer ausgeschlossen"
                    ),
                    (
                        "P-Abdeckung (messbar / Kandidaten): "
                        f"{float(rhythm.get('p_wave_coverage') or 0.0):.0%} / "
                        f"{float(rhythm.get('p_candidate_coverage') or 0.0):.0%}"
                    ),
                ]
                if rhythm.get("p_candidate_timing_mad_ms") is not None:
                    rhythm_details.append(
                        "P-R-Kopplung MAD: "
                        f"{float(rhythm['p_candidate_timing_mad_ms']):.1f} ms"
                    )
                if rhythm.get("p_candidate_timing_p10_p90_ms") is not None:
                    rhythm_details.append(
                        "P-R-Kopplung P10-P90: "
                        f"{float(rhythm['p_candidate_timing_p10_p90_ms']):.1f} ms"
                    )
                if rhythm.get("atrial_frequency_hz") is not None:
                    rhythm_details.append(
                        "Atriales Spektralmuster: "
                        f"{float(rhythm['atrial_frequency_hz']):.2f} Hz"
                    )
                if rhythm.get("atrial_spectral_concentration") is not None:
                    rhythm_details.append(
                        "Spektrale Konzentration: "
                        f"{float(rhythm['atrial_spectral_concentration']):.1%}"
                    )
                if rhythm.get("atrial_bandpower_ratio") is not None:
                    rhythm_details.append(
                        "Atriale Bandpower-Last (3,5-9 Hz / 0,5-40 Hz): "
                        f"{float(rhythm['atrial_bandpower_ratio']):.1%}"
                    )
                self.case_delineation_status.setToolTip(
                    "\n".join(rhythm_details)
                )
        except Exception as exc:
            QMessageBox.critical(self, "Forschungsanalyse", str(exc))

    def _case_changed(self) -> None:
        case_id = str(self.case_combo.currentData() or "")
        freshly_loaded_signal = None
        if case_id and hasattr(self, "expert_lab"):
            try:
                different_case = self.expert_lab.case is None or self.expert_lab.case.case_id != case_id
                self.expert_lab.load_case(self.case_library, case_id)
                if different_case:
                    freshly_loaded_signal = (self.expert_lab.signal, self.expert_lab.reference.sampling_rate_hz)
            except (ValueError, OSError) as exc:
                LOGGER.exception("Expert laboratory case could not be opened")
                QMessageBox.warning(self, "Laborfall", str(exc))
                return
        self.case_status.clear()
        self.ecg_plot.clear()
        self.ecg_plot.setTitle("")
        self.ecg_plot.getAxis("left").setTicks([])
        self._clear_vcg_view()
        self.rr_tachogram_plot.clear()
        self.poincare_plot.clear()
        self.beat_overlay_plot.clear()
        self.qt_rr_plot.clear()
        self.rr_tachogram_note.setText("Noch keine Forschungsanalyse verfügbar.")
        self.beat_overlay_note.setText("Noch keine Forschungsanalyse verfügbar.")
        self.qt_rr_note.setText("Noch keine Forschungsanalyse verfügbar.")
        self.show_vcg_cloud.setEnabled(False)
        self._current_case_ecg = None
        self._current_case_r_peaks = None
        self._current_case_sampling_rate = 0.0
        self._current_case_record_vcg = None
        self._current_case_vcg_normalized = None
        self._case_vcg_cursor_item = None
        self._case_ecg_cursor_line = None
        self._current_case_qt_rr = (None, None, None)
        self.case_findings.setRowCount(0)
        self.case_guideline_comparison.setRowCount(0)
        self.case_scores.setRowCount(0)
        self.case_single_markers.setRowCount(0)
        self.case_classical.setRowCount(0)
        self.case_features.setRowCount(0)
        self.case_delineation.setRowCount(0)
        self._current_case_fiducials = {}
        self._current_case_fiducial_details = {}
        self._current_case_segments = []
        self._current_case_array_path = None
        self._current_case_sample_count = 0
        self.edit_fiducials_button.setEnabled(False)
        self.case_delineation_status.setText(
            "Automatische Wellen- und Intervallmessung noch nicht ausgeführt."
        )
        self.case_pattern_summary.setText(
            "Noch keine Forschungsanalyse verfügbar."
        )
        case_id = str(self.case_combo.currentData() or "")
        if not case_id:
            return
        analysis_files = sorted(
            (
                path
                for path in (
                    self.config.cases_root / case_id / "analyses"
                ).glob("*/analysis.json")
                if (path.parent / "visualization_arrays.npz").is_file()
            ),
        )
        self.lab_analysis_combo.blockSignals(True)
        self.lab_analysis_combo.clear()
        self.lab_analysis_combo.addItem("Originalsignal – keine Analyse ausgewählt", "")
        valid_files = {}
        for path in analysis_files:
            item = _read_json(path)
            if not item or item.get("case_id") != case_id:
                continue
            identifier = str(item.get("analysis_id", ""))
            if not identifier or path.parent.name != identifier[:16]:
                continue
            valid_files[identifier] = path
            self.lab_analysis_combo.addItem(f"{identifier[:12]} · {item.get('created_at', 'Zeit unbekannt')} · {item.get('vcg_method', 'Methode unbekannt')} · gespeicherter Forschungsstand", identifier)
        selected_id = self._lab_analysis_choices.get(case_id)
        if selected_id is None and len(valid_files) == 1:
            selected_id = next(iter(valid_files))
        index = self.lab_analysis_combo.findData(selected_id or "")
        self.lab_analysis_combo.setCurrentIndex(max(0, index))
        self.lab_analysis_combo.blockSignals(False)
        if not analysis_files:
            self._render_unanalyzed_case(case_id, freshly_loaded_signal)
            return
        analysis_path = valid_files.get(selected_id)
        if analysis_path is None:
            self.expert_lab.clear_analysis()
            self._render_unanalyzed_case(case_id, freshly_loaded_signal)
            return
        try:
            payload = _read_json(analysis_path)
            if not payload:
                return
            self._render_case_arrays(
                analysis_path.parent / "visualization_arrays.npz"
            )
            self._render_case_results(payload)
            if payload.get("analysis_status") == "blocked_signal_quality":
                self.case_status.setText(
                    "Gespeicherte Analyse durch Signal-QC blockiert: "
                    + str(payload.get("block_reason", "unzureichende Qualität"))
                )
                self.case_delineation_status.setText(
                    "Keine verlässliche Wellenmessung; Signal nicht beurteilbar."
                )
                return
            delineation = payload.get("delineation", {})
            accepted = int(delineation.get("accepted_beat_count") or 0)
            confidence = float(
                delineation.get("fiducial_confidence") or 0.0
            )
            rhythm = delineation.get("rhythm_context", {})
            self.case_status.setText(
                f"{accepted} akzeptierte Beats · gespeicherte Forschungsanalyse "
                f"{str(payload.get('analysis_id', ''))[:12]} · keine Diagnose."
            )
            self.case_delineation_status.setText(
                "Automatische Wellenmessung geladen · Rhythmuskontext: "
                f"{rhythm.get('class', 'unknown')} · "
                f"Fiduzialkonfidenz {confidence:.0%}."
            )
            self.edit_fiducials_button.setEnabled(
                bool(self._current_case_fiducials)
            )
        except Exception as exc:
            self.case_status.setText(
                "Gespeicherte Forschungsanalyse konnte nicht dargestellt "
                f"werden: {type(exc).__name__}: {exc}"
            )

    def _render_unanalyzed_case(self, case_id: str, freshly_loaded_signal=None) -> None:
        """Display imported samples only: no filtering, delineation or markers."""
        try:
            case = self.case_library.load_case(case_id)
            ecg, sampling_rate = freshly_loaded_signal if freshly_loaded_signal is not None else self.case_library.load_signal(case_id)
            if (ecg.ndim != 2 or ecg.shape[1] != 12 or len(ecg) < 2
                    or not np.isfinite(sampling_rate) or sampling_rate <= 0
                    or not np.isfinite(ecg).all() or len(case.lead_names) != 12):
                raise ValueError("Rohsignal hat keine vollstaendige endliche 12-Kanal-Struktur")
            self._current_case_ecg = ecg
            self._current_case_sampling_rate = sampling_rate
            self._current_case_sample_count = len(ecg)
            self._current_case_r_peaks = np.empty(0, dtype=int)
            # At most 60 s in the initial preview; no downsampling that could
            # conceal brief morphology. Full signal stays in the case library.
            shown = max(2, int(min(len(ecg), 60*sampling_rate)))
            axis = np.arange(shown)/sampling_rate
            offsets = np.arange(12)[::-1]*3.
            color = "#56B4E9" if self._dark_mode else "#087e8b"
            for index in range(12):
                self.ecg_plot.plot(axis, ecg[:shown, index]+offsets[index], pen=pg.mkPen(color, width=1))
            self.ecg_plot.getAxis("left").setTicks([list(zip(offsets, case.lead_names))])
            self.ecg_plot.setLabel("bottom", "Zeit [s]")
            self.ecg_plot.setLabel("left", "Ableitungen: 3 mV Anzeigeversatz")
            self.ecg_plot.setTitle(f"Rohsignalvorschau: {shown:,}/{len(ecg):,} Samples; keine Wellen-/Markeranalyse")
            self.case_status.setText("Importiertes Roh-EKG sichtbar. Signalqualitaet noch nicht bewertet; keine diagnostische Freigabe.")
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.exception("Unanalyzed ECG preview failed")
            self.ecg_plot.clear()
            self.ecg_plot.setTitle("Rohsignalvorschau nicht verfuegbar")
            self.case_status.setText(f"Rohsignal nicht darstellbar: {exc}")

    def _render_case_arrays(self, path: Path) -> None:
        self.ecg_plot.setTitle("")
        self._current_case_array_path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            ecg = np.asarray(archive["diagnostic_ecg"], dtype=float)
            vcg = np.asarray(archive["vcg_xyz"], dtype=float)
            r_peaks = np.asarray(
                archive["r_peaks"]
                if "r_peaks" in archive
                else np.empty(0),
                dtype=int,
            )
            fiducial_names = np.asarray(
                archive["fiducial_names"]
                if "fiducial_names" in archive
                else np.empty(0, dtype="U24"),
            ).astype(str)
            fiducial_samples = np.asarray(
                archive["fiducial_samples"]
                if "fiducial_samples" in archive
                else np.empty(0),
                dtype=int,
            )
            fiducial_confidences = np.asarray(
                archive["fiducial_confidences"]
                if "fiducial_confidences" in archive
                else np.ones(len(fiducial_samples)),
                dtype=float,
            )
            fiducial_uncertainty = np.asarray(
                archive["fiducial_uncertainty_ms"]
                if "fiducial_uncertainty_ms" in archive
                else np.zeros(len(fiducial_samples)),
                dtype=float,
            )
            fiducial_sources = np.asarray(
                archive["fiducial_sources"]
                if "fiducial_sources" in archive
                else np.full(len(fiducial_samples), "legacy"),
            ).astype(str)
            fiducial_wave_statuses = np.asarray(
                archive["fiducial_wave_statuses"]
                if "fiducial_wave_statuses" in archive
                else np.full(len(fiducial_samples), "unknown"),
            ).astype(str)
            segment_names = np.asarray(
                archive["segment_names"]
                if "segment_names" in archive
                else np.empty(0, dtype="U24"),
            ).astype(str)
            segment_starts = np.asarray(
                archive["segment_start_samples"]
                if "segment_start_samples" in archive
                else np.empty(0),
                dtype=int,
            )
            segment_ends = np.asarray(
                archive["segment_end_samples"]
                if "segment_end_samples" in archive
                else np.empty(0),
                dtype=int,
            )
            segment_confidences = np.asarray(
                archive["segment_confidences"]
                if "segment_confidences" in archive
                else np.ones(len(segment_starts)),
                dtype=float,
            )
            segment_beat_indices = np.asarray(
                archive["segment_beat_indices"]
                if "segment_beat_indices" in archive
                else np.full(len(segment_starts), -1),
                dtype=int,
            )
            segment_r_peaks = np.asarray(
                archive["segment_r_peak_samples"]
                if "segment_r_peak_samples" in archive
                else np.full(len(segment_starts), -1),
                dtype=int,
            )
            sampling_rate = float(
                np.asarray(archive["sampling_rate_hz"]).item()
            )
            record_vcg = (
                np.asarray(archive["record_vcg_xyz"], dtype=float)
                if "record_vcg_xyz" in archive
                else None
            )
            qt_rr_beat_rr_ms = (
                np.asarray(archive["qt_rr_beat_rr_ms"], dtype=float)
                if "qt_rr_beat_rr_ms" in archive
                else None
            )
            qt_rr_beat_qt_ms = (
                np.asarray(archive["qt_rr_beat_qt_ms"], dtype=float)
                if "qt_rr_beat_qt_ms" in archive
                else None
            )
            qt_rr_beat_confidence = (
                np.asarray(archive["qt_rr_beat_confidence"], dtype=float)
                if "qt_rr_beat_confidence" in archive
                else None
            )
        self._current_case_sample_count = len(ecg)
        self._current_case_ecg = ecg
        self._current_case_sampling_rate = sampling_rate
        self._current_case_record_vcg = record_vcg
        self._current_case_qt_rr = (
            qt_rr_beat_rr_ms,
            qt_rr_beat_qt_ms,
            qt_rr_beat_confidence,
        )
        self.show_vcg_cloud.setEnabled(
            self._vcg_gl is not None and record_vcg is not None
        )
        self._current_case_fiducials = {
            str(name): int(sample)
            for name, sample in zip(fiducial_names, fiducial_samples)
        }
        self._current_case_fiducial_details = {
            str(name): {
                "sample": int(sample),
                "confidence": float(confidence),
                "uncertainty_ms": float(uncertainty),
                "source": str(source),
                "wave_status": str(wave_status),
            }
            for (
                name,
                sample,
                confidence,
                uncertainty,
                source,
                wave_status,
            ) in zip(
                fiducial_names,
                fiducial_samples,
                fiducial_confidences,
                fiducial_uncertainty,
                fiducial_sources,
                fiducial_wave_statuses,
            )
        }
        self._current_case_segments = [
            {
                "segment": str(name),
                "start_sample": int(start),
                "end_sample": int(end),
                "confidence": float(confidence),
                "beat_index": int(beat_index),
                "r_peak_sample": int(r_peak),
            }
            for (
                name,
                start,
                end,
                confidence,
                beat_index,
                r_peak,
            ) in zip(
                segment_names,
                segment_starts,
                segment_ends,
                segment_confidences,
                segment_beat_indices,
                segment_r_peaks,
            )
        ]
        self.edit_fiducials_button.setEnabled(
            bool(self._current_case_fiducials)
        )
        self.ecg_plot.clear()
        time_axis = np.arange(len(ecg)) / sampling_rate
        offsets = np.arange(ecg.shape[1])[::-1] * 3.0
        lead_names = (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )
        colors = ("#087e8b", "#d1495b", "#2f5d62", "#9a6d00")
        region_colors = {
            "p_wave": (8, 126, 139, 36),
            "pr_segment": (85, 122, 149, 28),
            "qrs_complex": (209, 73, 91, 38),
            "st_segment": (42, 157, 143, 30),
            "t_wave": (183, 121, 0, 34),
            "u_wave": (109, 76, 155, 34),
        }
        if self.show_case_wave_regions.isChecked():
            for segment in self._current_case_segments:
                start = segment["start_sample"] / sampling_rate
                end = segment["end_sample"] / sampling_rate
                if end <= start:
                    continue
                color = region_colors.get(
                    segment["segment"],
                    (80, 80, 80, 24),
                )
                region = pg.LinearRegionItem(
                    values=(start, end),
                    movable=False,
                    brush=pg.mkBrush(*color),
                    pen=pg.mkPen(None),
                )
                region.setZValue(-20)
                region.setToolTip(
                    (
                        f"Schlag {segment['beat_index'] + 1} · "
                        if segment.get("beat_index", -1) >= 0
                        else ""
                    )
                    + f"{segment['segment']}: "
                    f"{(end - start) * 1000.0:.1f} ms · "
                    f"Konfidenz {segment['confidence']:.0%}"
                )
                self.ecg_plot.addItem(region)
        for index in range(min(12, ecg.shape[1])):
            self.ecg_plot.plot(
                time_axis,
                ecg[:, index] + offsets[index],
                pen=pg.mkPen(colors[index % len(colors)], width=1),
            )
        self._case_ecg_cursor_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#111111", width=1.5),
        )
        self._case_ecg_cursor_line.setVisible(False)
        self.ecg_plot.addItem(self._case_ecg_cursor_line)
        valid_r = r_peaks[(r_peaks >= 0) & (r_peaks < len(ecg))]
        self._current_case_r_peaks = valid_r
        if len(valid_r) and ecg.shape[1] > 1:
            self.ecg_plot.plot(
                valid_r / sampling_rate,
                ecg[valid_r, 1] + offsets[1],
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush("#222222"),
                symbolPen=pg.mkPen(None),
            )
        overlay_colors = {
            "p_onset": "#087e8b",
            "p_peak": "#087e8b",
            "p_offset": "#087e8b",
            "q_peak": "#555555",
            "qrs_onset": "#d1495b",
            "r_peak": "#d1495b",
            "qrs_offset": "#d1495b",
            "t_onset": "#9a6d00",
            "t_peak": "#9a6d00",
            "t_offset": "#9a6d00",
        }
        for name, sample in self._current_case_fiducials.items():
            detail = self._current_case_fiducial_details.get(name, {})
            confidence = float(detail.get("confidence", 0.0))
            line = pg.InfiniteLine(
                pos=sample / sampling_rate,
                angle=90,
                movable=False,
                pen=pg.mkPen(
                    overlay_colors.get(name, "#555555"),
                    width=1.0 + confidence,
                    style=Qt.PenStyle.DashLine,
                ),
            )
            line.setToolTip(
                f"{name}: Sample {sample} · "
                f"Konfidenz {confidence:.0%} · "
                "Unsicherheit ±"
                f"{float(detail.get('uncertainty_ms', 0.0)):.1f} ms · "
                f"Status {detail.get('wave_status', 'unknown')} | "
                f"{detail.get('source', '')}"
            )
            self.ecg_plot.addItem(line)
        self.ecg_plot.getAxis("left").setTicks(
            [[
                (offsets[index], lead_names[index])
                for index in range(min(12, ecg.shape[1]))
            ]]
        )
        self._render_vcg(
            vcg,
            self._current_case_segments,
            record_vcg=(
                self._current_case_record_vcg
                if self.show_vcg_cloud.isChecked()
                else None
            ),
        )
        self._draw_rr_tachogram_and_poincare()
        self._draw_beat_overlay()
        self._draw_qt_rr_regression()

    def _draw_rr_tachogram_and_poincare(self) -> None:
        r_peaks = self._current_case_r_peaks
        fs = self._current_case_sampling_rate
        self.rr_tachogram_plot.clear()
        self.poincare_plot.clear()
        if r_peaks is None or len(r_peaks) < 3 or fs <= 0:
            self.rr_tachogram_note.setText(
                "Zu wenige erkannte Schläge für ein RR-Tachogramm."
            )
            return
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        self.rr_tachogram_note.setText(f"{len(rr_ms)} RR-Intervalle.")
        beat_times = r_peaks[1:] / fs
        self.rr_tachogram_plot.plot(
            beat_times,
            rr_ms,
            pen=pg.mkPen("#178292", width=1.5),
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush("#178292"),
            symbolPen=pg.mkPen(None),
        )
        rr_prev = rr_ms[:-1]
        rr_next = rr_ms[1:]
        if len(rr_prev):
            self.poincare_plot.plot(
                rr_prev,
                rr_next,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush(23, 130, 146, 140),
                symbolPen=pg.mkPen(None),
            )
            low = float(min(rr_prev.min(), rr_next.min()))
            high = float(max(rr_prev.max(), rr_next.max()))
            self.poincare_plot.plot(
                [low, high],
                [low, high],
                pen=pg.mkPen("#8a3b47", width=1, style=Qt.PenStyle.DashLine),
            )

    def _draw_beat_overlay(self, *_args) -> None:
        self.beat_overlay_plot.clear()
        ecg = self._current_case_ecg
        r_peaks = self._current_case_r_peaks
        fs = self._current_case_sampling_rate
        if ecg is None or r_peaks is None or len(r_peaks) < 2 or fs <= 0:
            self.beat_overlay_note.setText(
                "Zu wenige erkannte Schläge für eine Beat-Überlagerung."
            )
            return
        lead_index = self.beat_overlay_lead_combo.currentData()
        lead_index = 1 if lead_index is None else int(lead_index)
        if lead_index >= ecg.shape[1]:
            self.beat_overlay_note.setText("Ableitung nicht verfügbar.")
            return
        half = int(round(0.28 * fs))
        windows = [
            ecg[r - half : r + half + 1, lead_index]
            for r in r_peaks
            if r - half >= 0 and r + half + 1 <= len(ecg)
        ]
        if len(windows) < 2:
            self.beat_overlay_note.setText(
                "Zu wenige vollständige Beat-Fenster für diese Ableitung."
            )
            return
        self.beat_overlay_note.setText(
            f"{len(windows)} Beats überlagert (R-ausgerichtet, ±280 ms)."
        )
        stack = np.vstack(windows)
        t_ms = (np.arange(stack.shape[1]) - half) / fs * 1000.0
        for beat in stack:
            self.beat_overlay_plot.plot(
                t_ms, beat, pen=pg.mkPen(23, 130, 146, 60, width=1)
            )
        median_beat = np.median(stack, axis=0)
        self.beat_overlay_plot.plot(
            t_ms, median_beat, pen=pg.mkPen("#b42318", width=2.5)
        )

    def _draw_qt_rr_regression(self) -> None:
        self.qt_rr_plot.clear()
        rr_ms, qt_ms, _confidence = self._current_case_qt_rr
        if rr_ms is None or qt_ms is None or len(qt_ms) < 3:
            self.qt_rr_note.setText(
                "Für diesen Fall noch nicht verfügbar — erneut "
                "analysieren, um dieses Diagramm zu erhalten."
            )
            return
        self.qt_rr_note.setText(f"{len(qt_ms)} Beat-Paare.")
        self.qt_rr_plot.plot(
            rr_ms,
            qt_ms,
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(23, 130, 146, 150),
            symbolPen=pg.mkPen(None),
        )
        rr_s = rr_ms / 1000.0
        qtc_bazett = float(np.median(qt_ms / np.sqrt(rr_s)))
        qtc_fridericia = float(np.median(qt_ms / np.cbrt(rr_s)))
        rr_grid_s = np.linspace(
            max(0.25, float(rr_s.min())), min(3.0, float(rr_s.max())), 60
        )
        self.qt_rr_plot.plot(
            rr_grid_s * 1000.0,
            qtc_bazett * np.sqrt(rr_grid_s),
            pen=pg.mkPen("#b42318", width=2),
            name="Bazett",
        )
        self.qt_rr_plot.plot(
            rr_grid_s * 1000.0,
            qtc_fridericia * np.cbrt(rr_grid_s),
            pen=pg.mkPen("#9a6700", width=2, style=Qt.PenStyle.DashLine),
            name="Fridericia",
        )

    def _redraw_current_case_arrays(self, *_args) -> None:
        path = self._current_case_array_path
        if path is not None and path.is_file():
            self._render_case_arrays(path)

    def _case_mouse_moved(self, position) -> None:
        point = self.ecg_plot.plotItem.vb.mapSceneToView(position)
        self._set_case_cursor_time(float(point.x()))

    def _set_case_cursor_time(self, time_seconds: float) -> None:
        ecg = self._current_case_ecg
        fs = self._current_case_sampling_rate
        vcg = getattr(self, "_current_case_vcg_normalized", None)
        if ecg is None or fs <= 0 or vcg is None or len(vcg) == 0:
            return
        sample = int(round(time_seconds * fs))
        if sample < 0 or sample >= min(len(ecg), len(vcg)):
            if self._case_ecg_cursor_line is not None:
                self._case_ecg_cursor_line.setVisible(False)
            return
        if self._case_ecg_cursor_line is not None:
            self._case_ecg_cursor_line.setValue(sample / fs)
            self._case_ecg_cursor_line.setVisible(True)
        if self._vcg_gl is None or not np.all(np.isfinite(vcg[sample])):
            return
        point_3d = np.asarray(vcg[sample : sample + 1], dtype=np.float32)
        if self._case_vcg_cursor_item is None:
            self._case_vcg_cursor_item = self._vcg_gl.GLScatterPlotItem(
                pos=point_3d,
                color=(1.0, 0.84, 0.0, 1.0),
                size=12,
                pxMode=True,
            )
            self.vcg_view.addItem(self._case_vcg_cursor_item)
            self._vcg_items.append(self._case_vcg_cursor_item)
        else:
            self._case_vcg_cursor_item.setData(pos=point_3d)

    def edit_case_fiducials(self) -> None:
        case_id = str(self.case_combo.currentData() or "")
        if (
            not case_id
            or not self._current_case_fiducials
            or self._current_case_sample_count < 1
        ):
            QMessageBox.information(
                self,
                "Fiduzialpunkte",
                "Zuerst muss eine auswertbare Forschungsanalyse vorliegen.",
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Fiduzialpunkte prüfen")
        dialog.setMinimumWidth(430)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "Samplepositionen des repräsentativen Beats. Die Revision wird "
            "lokal versioniert und verändert das Rohsignal nicht."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        fields: dict[str, QSpinBox] = {}
        names = (
            "p_onset",
            "p_peak",
            "p_offset",
            "q_peak",
            "qrs_onset",
            "r_peak",
            "qrs_offset",
            "t_onset",
            "t_peak",
            "t_offset",
        )
        for name in names:
            field = QSpinBox()
            field.setRange(-1, self._current_case_sample_count - 1)
            field.setSpecialValueText("nicht gesetzt")
            field.setValue(self._current_case_fiducials.get(name, -1))
            field.setToolTip(
                "Absolute Sampleposition; -1 entfernt den manuellen Override"
            )
            fields[name] = field
            form.addRow(name, field)
        layout.addLayout(form)
        notes = QTextEdit()
        notes.setPlaceholderText("Begründung der lokalen Korrektur")
        notes.setMaximumHeight(90)
        layout.addWidget(notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        fiducials = {
            name: field.value()
            for name, field in fields.items()
            if field.value() >= 0
        }
        if not fiducials:
            QMessageBox.warning(
                self,
                "Fiduzialpunkte",
                "Mindestens ein Fiduzialpunkt muss gesetzt bleiben.",
            )
            return
        revisions = self.case_library.annotations(case_id)
        try:
            self.case_library.add_annotation(
                case_id,
                fiducials=fiducials,
                notes=notes.toPlainText().strip(),
                parent_revision_id=(
                    revisions[-1].revision_id if revisions else ""
                ),
            )
            self.analyze_selected_case()
        except Exception as exc:
            QMessageBox.critical(self, "Fiduzialpunkte", str(exc))

    def _load_paired_comparison_lookup(
        self,
        payload: dict,
    ) -> dict[tuple[str, str], dict]:
        """DB-weite Verdikte (Work Package A) für den Leitkennwert je
        Endpunkt/Kriterium, zum Anreichern der Pro-Fall-Ansicht. Bewusst
        über den prespezifizierten Leitmarker indiziert, nicht über den
        in dieser Zeile gezeigten Score — siehe Tooltip in der Tabelle."""
        run_id = str(
            payload.get("score_release_inventory", {}).get(
                "source_run_id",
                "",
            )
        )
        current_run_id = self._current_run_id()
        if not run_id or run_id != current_run_id:
            if run_id:
                LOGGER.warning(
                    "Ignoring case comparison from non-current run %s "
                    "(current: %s)",
                    run_id,
                    current_run_id,
                )
            return {}
        path = (
            self.config.runs_root
            / run_id
            / "vcg_vs_classical_comparison.parquet"
        )
        if not path.is_file():
            return {}
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "endpoint_id",
                    "baseline_feature_id",
                    "comparison_verdict_de",
                    "comparison_type",
                    "sensitivity_delta_vs_fixed_rule",
                    "sensitivity_delta_ci_low",
                    "sensitivity_delta_ci_high",
                    "delta_auc",
                    "delta_auc_ci_low",
                    "delta_auc_ci_high",
                ],
            )
        except Exception as exc:
            LOGGER.warning("Classical baseline lookup failed: %s", exc)
            return {}
        lookup: dict[tuple[str, str], dict] = {}
        for row in frame.itertuples(index=False):
            lookup[(str(row.endpoint_id), str(row.baseline_feature_id))] = (
                row._asdict()
            )
        return lookup

    def _case_payload_for_current_run(self, payload: dict) -> dict:
        """Hide run-derived case results when their lineage is historical.

        Signal measurements and classical rules are case-local and remain
        useful. Trained scores, criteria and diagnosis summaries are artifacts
        of a scientific run and must not survive a fingerprint change in the
        GUI merely because a stored ``analysis.json`` is newer.
        """

        inventory = dict(payload.get("score_release_inventory", {}) or {})
        source_run_id = str(inventory.get("source_run_id", "") or "")
        current_run_id = self._current_run_id()
        if not source_run_id or source_run_id == current_run_id:
            return payload
        safe = dict(payload)
        for key in (
            "research_findings",
            "score_releases",
            "single_marker_patterns",
        ):
            safe[key] = []
        safe["research_diagnosis_summary"] = {
            "message_de": (
                "Gespeicherte run-abhängige Bewertung ausgeblendet: "
                f"Quelle {source_run_id}, aktueller Run {current_run_id}."
            ),
            "probable_research_diagnoses": [],
        }
        safe["criterion_inventory"] = {
            "status": "historical_run_hidden",
            "source_run_id": source_run_id,
            "current_run_id": current_run_id,
            "candidate_count": 0,
        }
        safe["score_release_inventory"] = {
            **inventory,
            "compatible_release_count": 0,
            "endpoint_count": 0,
            "status": "historical_run_hidden",
        }
        return safe

    def _render_case_results(self, payload: dict) -> None:
        if hasattr(self, "expert_lab"):
            if self.expert_lab.set_analysis(payload) is False:
                return  # Late result after case navigation: do not load its arrays or views.
            arrays = self.config.cases_root / str(payload.get("case_id", "")) / "analyses" / str(payload.get("analysis_id", ""))[:16] / "visualization_arrays.npz"
            if arrays.is_file():
                self.expert_lab.set_analysis_arrays(arrays)
        payload = self._case_payload_for_current_run(payload)
        self.case_research_status.set_case_payload(payload)
        feature_values = dict(payload.get("feature_values", {}) or {})
        area = feature_values.get("qrs_t_area_angle_direct_8lead_deg")
        integral = feature_values.get("qrs_t_integral_angle_direct_8lead_deg")
        self.case_direct_angle_label.setText(
            "Direct-8-Lead: Area {} · Integral {}".format(
                _value_with_unit(area, "°") if area is not None else "nicht messbar",
                _value_with_unit(integral, "°")
                if integral is not None
                else "nicht messbar",
            )
        )
        finding_rows = []
        for finding in payload.get("research_findings", ()):
            primary = dict(finding.get("primary_operating_point", {}))
            validated_metrics = finding_validation_metrics(finding)
            comparators = list(finding.get("classical_comparators", ()))

            def _baseline_label(item: dict) -> str:
                # Sollwert alongside the name: does this case's verdict
                # land where the guideline formula says it should -- a
                # direct, in-place plausibility check, not just a
                # positive/negative flag.
                name = str(item.get("display_name", item.get("criterion_id", "")))
                sollwert = str(item.get("sollwert", "")).strip()
                return f"{name} (Sollwert: {sollwert})" if sollwert else name

            positive_baselines = [
                _baseline_label(item)
                for item in comparators
                if item.get("evaluable") is True
                and item.get("positive") is True
            ]
            negative_baselines = [
                _baseline_label(item)
                for item in comparators
                if item.get("evaluable") is True
                and item.get("positive") is False
            ]
            if positive_baselines:
                baseline_text = "positiv: " + ", ".join(positive_baselines)
            elif negative_baselines:
                baseline_text = "negativ: " + ", ".join(negative_baselines)
            elif comparators:
                baseline_text = "nicht auswertbar / keine anerkannte Regel"
            else:
                baseline_text = "kein anerkannter ECG-Diagnosebaseline"
            gray_low = primary.get("gray_zone_low")
            gray_high = primary.get("gray_zone_high")
            unit = str(finding.get("unit", ""))
            finding_rows.append(
                (
                    finding.get("endpoint_name_de", finding.get("endpoint_id", "")),
                    finding.get("classification_de", finding.get("classification", "")),
                    (
                        f"{finding.get('stage_de', '')} · "
                        f"{finding.get('evidence_grade_de', '')}"
                    ).strip(" ·"),
                    finding.get("feature_name_de", finding.get("feature_id", "")),
                    _value_with_unit(finding.get("value"), unit),
                    _value_with_unit(primary.get("threshold"), unit),
                    (
                        f"{_format_number(gray_low)} bis {_format_number(gray_high)} {unit}".strip()
                        if gray_low is not None and gray_high is not None
                        else "-"
                    ),
                    _value_with_unit(
                        finding.get("distance_to_gray_zone"),
                        unit,
                    ),
                    _format_percentage(validated_metrics["sensitivity"]),
                    _format_percentage(validated_metrics["specificity"]),
                    baseline_text,
                    finding.get("interpretation_de", ""),
                )
            )
        findings_sorting = self.case_findings.isSortingEnabled()
        self.case_findings.setSortingEnabled(False)
        self._fill_table(self.case_findings, finding_rows)
        for row_index, finding in enumerate(
            payload.get("research_findings", ())
        ):
            reasons = finding.get("applicability_reasons_de", ())
            tooltip = (
                "Anwendbarkeit: " + "; ".join(str(item) for item in reasons)
                if reasons
                else str(finding.get("research_warning_de", ""))
            )
            for column in range(self.case_findings.columnCount()):
                item = self.case_findings.item(row_index, column)
                if item is not None:
                    item.setToolTip(tooltip)

            metrics = finding_validation_metrics(finding)
            for column, metric in ((8, "sensitivity"), (9, "specificity")):
                item = self.case_findings.item(row_index, column)
                if item is not None:
                    item.setToolTip(tooltip + "\n" + metrics[metric + "_note"])

        # Attach evidence while the rows still have their input ordering.
        # Re-enabling sorting then moves the values and their tooltips together.
        self.case_findings.setSortingEnabled(findings_sorting)
        score_rows = []
        for score in payload.get("score_releases", ()):
            primary = score.get("primary_operating_point", {})
            metrics = score.get("validation_metrics", {})
            auc = _format_number(metrics.get("roc_auc"))
            auc_low = _format_number(metrics.get("roc_auc_ci_low"))
            auc_high = _format_number(metrics.get("roc_auc_ci_high"))
            score_rows.append(
                (
                    score.get("endpoint_id", ""),
                    _case_pattern_status_text(score),
                    score.get("evidence_status", ""),
                    _format_number(score.get("research_score")),
                    _format_number(primary.get("threshold_aligned")),
                    _format_number(primary.get("distance_to_threshold")),
                    (
                        f"{auc} [{auc_low}; {auc_high}]"
                        if auc and auc_low and auc_high
                        else auc
                    ),
                    _format_number(
                        metrics.get("sensitivity_at_95_specificity")
                    ),
                    score.get("classical_positive_count", 0),
                    (
                        f"{score.get('release_id', '')} · "
                        f"{score.get('stage', '')}"
                    ),
                )
            )
        self._fill_table(self.case_scores, score_rows)
        for row_index, score in enumerate(payload.get("score_releases", ())):
            tooltip = _analysis_restriction_tooltip(score)
            if not tooltip:
                continue
            for column in range(self.case_scores.columnCount()):
                item = self.case_scores.item(row_index, column)
                if item is not None:
                    item.setToolTip(tooltip)

        self._fill_table(
            self.case_guideline_comparison,
            _build_guideline_comparison_rows(
                payload,
                comparison_lookup=self._load_paired_comparison_lookup(
                    payload
                ),
            ),
        )
        single_marker_patterns = payload.get(
            "single_marker_patterns",
            (),
        )
        single_marker_rows = []
        for marker in single_marker_patterns:
            primary = marker.get("primary_operating_point", {})
            metrics = marker.get("validation_metrics", {})
            single_marker_rows.append(
                (
                    marker.get("endpoint_id", ""),
                    marker.get("feature_id", ""),
                    marker.get("feature_modality", ""),
                    _format_number(marker.get("raw_value")),
                    marker.get("direction", ""),
                    _format_number(primary.get("threshold_aligned")),
                    _format_number(
                        primary.get("distance_to_threshold")
                    ),
                    _case_pattern_status_text(marker),
                    _format_number(metrics.get("roc_auc")),
                    _format_number(
                        metrics.get("sensitivity_at_95_specificity")
                    ),
                    marker.get("evidence_grade", ""),
                    marker.get("conceptual_family", ""),
                )
            )
        self._fill_table(
            self.case_single_markers,
            single_marker_rows,
        )
        for row_index, marker in enumerate(single_marker_patterns):
            tooltip = _analysis_restriction_tooltip(marker)
            if not tooltip:
                continue
            for column in range(self.case_single_markers.columnCount()):
                item = self.case_single_markers.item(row_index, column)
                if item is not None:
                    item.setToolTip(tooltip)
        diagnosis_summary = dict(
            payload.get("research_diagnosis_summary", {})
        )
        probable = list(
            diagnosis_summary.get("probable_research_diagnoses", ())
        )
        if probable:
            labels = ", ".join(
                str(item.get("endpoint_name_de", item.get("endpoint_id", "")))
                for item in probable[:6]
            )
            self.case_pattern_summary.setText(
                "Wahrscheinliche Forschungsdiagnose: "
                + labels
                + ". "
                + str(payload.get("research_warning_de", ""))
            )
            self.case_pattern_summary.setStyleSheet(
                theme.status_box_qss("danger", self._dark_mode)
            )
        elif payload.get("analysis_status") == "blocked_signal_quality":
            self.case_pattern_summary.setText(
                diagnosis_summary.get(
                    "message_de",
                    "Signal nicht beurteilbar; keine Forschungsbewertung.",
                )
            )
            self.case_pattern_summary.setStyleSheet(
                theme.status_box_qss("warning", self._dark_mode)
            )
        elif payload.get("research_findings"):
            positive_exploratory = sum(
                item.get("classification") == "positive"
                for item in payload.get("research_findings", ())
            )
            not_evaluable = sum(
                item.get("classification") == "not_evaluable"
                for item in payload.get("research_findings", ())
            )
            self.case_pattern_summary.setText(
                str(diagnosis_summary.get("message_de", ""))
                + f" Explorativ positiv: {positive_exploratory}; "
                f"nicht beurteilbar: {not_evaluable}. Negative Kriterien "
                "schließen eine Pathologie nicht aus."
            )
            self.case_pattern_summary.setStyleSheet(
                theme.status_box_qss("neutral", self._dark_mode)
            )
        else:
            inventory = dict(payload.get("criterion_inventory", {}))
            self.case_pattern_summary.setText(
                str(diagnosis_summary.get("message_de", ""))
                + " Status des Kriterienregisters: "
                + str(inventory.get("status", "nicht verfügbar"))
                + ". Klassische Regeln und Rohmarker bleiben als technische "
                "Vergleiche sichtbar."
            )
            self.case_pattern_summary.setStyleSheet(
                theme.status_box_qss("neutral", self._dark_mode)
            )
        classical_rows = []
        relevant_endpoints: dict[str, set[str]] = {}
        for score in payload.get("score_releases", ()):
            endpoint_id = str(score.get("endpoint_id", ""))
            for comparator in score.get("classical_comparators", ()):
                criterion_id = str(comparator.get("criterion_id", ""))
                if criterion_id:
                    relevant_endpoints.setdefault(
                        criterion_id,
                        set(),
                    ).add(endpoint_id)
        for finding in payload.get("research_findings", ()):
            endpoint_name = str(
                finding.get("endpoint_name_de", finding.get("endpoint_id", ""))
            )
            for comparator in finding.get("classical_comparators", ()):
                criterion_id = str(comparator.get("criterion_id", ""))
                if criterion_id:
                    relevant_endpoints.setdefault(
                        criterion_id,
                        set(),
                    ).add(endpoint_name)
        for criterion, value in payload.get("classical_criteria", {}).items():
            unit = str(value.get("unit", "")).strip()
            sollwert = str(value.get("definition", "")).strip()
            classical_rows.append(
                (
                    criterion,
                    ", ".join(
                        sorted(relevant_endpoints.get(criterion, ()))
                    ),
                    value.get("evaluable"),
                    value.get("positive"),
                    _value_with_unit(value.get("value"), unit),
                    sollwert,
                    value.get("exclusion_reason", ""),
                )
            )
        self._fill_table(self.case_classical, classical_rows)
        feature_rows = []
        for score in payload.get("score_releases", ()):
            for component in score.get("component_details", ()):
                feature_rows.append(
                    (
                        score.get("endpoint_id", ""),
                        component.get("feature_id", ""),
                        _format_number(component.get("raw_value")),
                        component.get("direction", ""),
                        _format_number(component.get("train_mean")),
                        _format_number(
                            component.get("train_standard_deviation")
                        ),
                        _format_number(
                            component.get("aligned_z_contribution")
                        ),
                        component.get("contribution_direction", ""),
                    )
                )
        if not feature_rows:
            feature_rows = [
                (
                    "",
                    feature,
                    _format_number(value),
                    "",
                    "",
                    "",
                    "",
                    "Einzelmarker; kein kompatibler Score",
                )
                for feature, value in payload.get(
                    "feature_values",
                    {},
                ).items()
            ]
        self._fill_table(self.case_features, feature_rows)
        delineation_payload = payload.get("delineation", {})
        delineation_rows = []
        for segment in delineation_payload.get("wave_segments", ()):
            start = segment.get("start_seconds")
            end = segment.get("end_seconds")
            duration_ms = segment.get("duration_ms")
            status = str(segment.get("status", ""))
            reason = str(segment.get("reason", "")).strip()
            delineation_rows.append(
                (
                    segment.get("segment", ""),
                    f"{float(start):.3f} s" if start is not None else "-",
                    f"{float(end):.3f} s" if end is not None else "-",
                    (
                        f"{float(duration_ms):.1f} ms"
                        if duration_ms is not None
                        else "-"
                    ),
                    f"{float(segment.get('confidence') or 0.0):.0%}",
                    f"{status}: {reason}" if reason else status,
                )
            )
        for interval in delineation_payload.get(
            "interval_measurements",
            (),
        ):
            value = interval.get("value_ms")
            uncertainty = interval.get("uncertainty_ms")
            duration = (
                f"{float(value):.1f} ± {float(uncertainty):.1f} ms"
                if value is not None and uncertainty is not None
                else f"{float(value):.1f} ms"
                if value is not None
                else "-"
            )
            delineation_rows.append(
                (
                    interval.get("interval", ""),
                    "-",
                    "-",
                    duration,
                    f"{float(interval.get('confidence') or 0.0):.0%}",
                    (
                        f"{interval.get('status', '')}: "
                        f"{interval.get('reason', '')}"
                        if interval.get("reason")
                        else interval.get("status", "")
                    ),
                )
            )
        self._fill_table(self.case_delineation, delineation_rows)

    def _create_vcg_view(self):
        try:
            import pyqtgraph.opengl as gl

            view = gl.GLViewWidget()
            view.setCameraPosition(distance=8)
            grid = gl.GLGridItem()
            grid.setSize(8, 8)
            grid.setSpacing(1, 1)
            view.addItem(grid)
            self._vcg_gl = gl
            self._vcg_grid = grid
            self._vcg_items = []
            return view
        except Exception as exc:
            LOGGER.info("OpenGL VCG view unavailable: %s", exc)
            self._vcg_gl = None
            self._vcg_items = []
            placeholder = QLabel(
                "3D-VCG ist ohne funktionierendes OpenGL nicht verfügbar."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return placeholder

    def _create_group_vcg_view(self):
        try:
            import pyqtgraph.opengl as gl

            view = gl.GLViewWidget()
            view.setCameraPosition(distance=5)
            grid = gl.GLGridItem()
            grid.setSize(4, 4)
            grid.setSpacing(0.5, 0.5)
            view.addItem(grid)
            self._group_vcg_gl = gl
            self._group_vcg_grid = grid
            self._group_vcg_items = []
            return view
        except Exception as exc:
            LOGGER.info("OpenGL group VCG view unavailable: %s", exc)
            self._group_vcg_gl = None
            self._group_vcg_items = []
            placeholder = QLabel(
                "3D-Gruppen-VCG ist ohne funktionierendes OpenGL nicht "
                "verfügbar. Die orthogonalen Projektionen bleiben nutzbar."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setWordWrap(True)
            return placeholder

    def _render_group_vcg(
        self,
        summary: pd.DataFrame,
        difference: pd.DataFrame,
    ) -> None:
        self._clear_group_vcg_view()
        self._clear_group_vcg_projections()
        groups = {}
        for state in ("negative", "positive"):
            group = (
                summary.loc[summary["label_state"].eq(state)]
                .sort_values("sample_index")
                .reset_index(drop=True)
            )
            if group.empty:
                return
            groups[state] = group

        colors = {
            "positive": (0.02, 0.55, 0.58, 0.96),
            "negative": (0.78, 0.20, 0.28, 0.96),
        }
        axis_columns = {
            "x": "median_x_mV",
            "y": "median_y_mV",
            "z": "median_z_mV",
        }
        points_by_state = {
            state: group[
                ["median_x_mV", "median_y_mV", "median_z_mV"]
            ].to_numpy(dtype=float)
            for state, group in groups.items()
        }

        if self._group_vcg_gl is not None:
            all_points = np.vstack(tuple(points_by_state.values()))
            finite_scale = np.nanpercentile(np.abs(all_points), 99)
            scale = (
                float(finite_scale)
                if np.isfinite(finite_scale) and finite_scale > 0
                else 1.0
            )
            grid_size = max(0.5, scale * 5.0)
            grid_spacing = max(0.05, grid_size / 8.0)
            self._group_vcg_grid.setSize(grid_size, grid_size)
            self._group_vcg_grid.setSpacing(grid_spacing, grid_spacing)
            self.group_vcg_view.setCameraPosition(
                distance=max(2.0, scale * 9.0)
            )
            for state in ("negative", "positive"):
                points = points_by_state[state]
                progress = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
                base = np.asarray(colors[state][:3], dtype=np.float32)
                time_colors = np.empty((len(points), 4), dtype=np.float32)
                time_colors[:, :3] = (
                    (0.45 * (1.0 - progress[:, None]))
                    + base[None, :] * (0.55 + 0.45 * progress[:, None])
                )
                time_colors[:, 3] = 0.96
                line = self._group_vcg_gl.GLLinePlotItem(
                    pos=points.astype(np.float32),
                    color=time_colors,
                    width=3,
                    antialias=True,
                    mode="line_strip",
                )
                self.group_vcg_view.addItem(line)
                self._group_vcg_items.append(line)
                endpoints = self._group_vcg_gl.GLScatterPlotItem(
                    pos=points[[0, -1]].astype(np.float32),
                    color=np.asarray((time_colors[0], time_colors[-1])),
                    size=np.asarray((7.0, 11.0), dtype=np.float32),
                    pxMode=True,
                )
                self.group_vcg_view.addItem(endpoints)
                self._group_vcg_items.append(endpoints)
            negative = points_by_state["negative"]
            positive = points_by_state["positive"]
            vector_indices = np.arange(
                0,
                min(len(negative), len(positive)),
                25,
                dtype=int,
            )
            segments = np.empty((len(vector_indices) * 2, 3), dtype=float)
            segments[0::2] = negative[vector_indices]
            segments[1::2] = positive[vector_indices]
            vectors = self._group_vcg_gl.GLLinePlotItem(
                pos=segments.astype(np.float32),
                color=(0.93, 0.68, 0.12, 0.72),
                width=2,
                antialias=True,
                mode="lines",
            )
            self.group_vcg_view.addItem(vectors)
            self._group_vcg_items.append(vectors)

        for _, (plot, horizontal, vertical) in (
            self.group_vcg_projection_plots.items()
        ):
            if plot.plotItem.legend is None:
                plot.addLegend(offset=(8, 8))
            for state in ("negative", "positive"):
                group = groups[state]
                color = colors[state]
                pen = pg.mkPen(
                    tuple(int(channel * 255) for channel in color[:3]),
                    width=2.5,
                )
                plot.plot(
                    group[axis_columns[horizontal]].to_numpy(dtype=float),
                    group[axis_columns[vertical]].to_numpy(dtype=float),
                    pen=pen,
                    name="positiv" if state == "positive" else "negativ",
                )
                for marker_index, symbol, size in ((0, "s", 7), (-1, "t", 10)):
                    plot.plot(
                        [float(group[axis_columns[horizontal]].iloc[marker_index])],
                        [float(group[axis_columns[vertical]].iloc[marker_index])],
                        pen=None, symbol=symbol, symbolSize=size,
                        symbolBrush=pen.color(),
                    )
                zero_index = int(
                    np.argmin(np.abs(group["time_ms"].to_numpy(dtype=float)))
                )
                plot.plot(
                    [
                        float(
                            group[axis_columns[horizontal]].iloc[zero_index]
                        )
                    ],
                    [
                        float(
                            group[axis_columns[vertical]].iloc[zero_index]
                        )
                    ],
                    pen=None,
                    symbol="o",
                    symbolSize=7,
                    symbolBrush=pen.color(),
                )
            negative = groups["negative"]
            positive = groups["positive"]
            vector_indices = np.arange(
                0,
                min(len(negative), len(positive)),
                25,
                dtype=int,
            )
            horizontal_segments = np.empty(len(vector_indices) * 2)
            vertical_segments = np.empty(len(vector_indices) * 2)
            horizontal_segments[0::2] = negative[
                axis_columns[horizontal]
            ].to_numpy(dtype=float)[vector_indices]
            horizontal_segments[1::2] = positive[
                axis_columns[horizontal]
            ].to_numpy(dtype=float)[vector_indices]
            vertical_segments[0::2] = negative[
                axis_columns[vertical]
            ].to_numpy(dtype=float)[vector_indices]
            vertical_segments[1::2] = positive[
                axis_columns[vertical]
            ].to_numpy(dtype=float)[vector_indices]
            plot.plot(
                horizontal_segments,
                vertical_segments,
                pen=pg.mkPen("#c89419", width=1.2),
                connect="pairs",
            )
        self.group_vcg_delta_plot.clear()
        if {"time_ms", "delta_magnitude_mV", "delta_magnitude_ci_low_mV",
                "delta_magnitude_ci_high_mV"}.issubset(difference):
            time = pd.to_numeric(difference["time_ms"], errors="coerce").to_numpy(dtype=float)
            magnitude = pd.to_numeric(
                difference["delta_magnitude_mV"], errors="coerce",
            ).to_numpy(dtype=float)
            low = pd.to_numeric(
                difference["delta_magnitude_ci_low_mV"], errors="coerce",
            ).to_numpy(dtype=float)
            high = pd.to_numeric(
                difference["delta_magnitude_ci_high_mV"], errors="coerce",
            ).to_numpy(dtype=float)
            valid = np.isfinite(time) & np.isfinite(magnitude) & np.isfinite(low) & np.isfinite(high)
            valid &= (low >= 0) & (low <= high)
            if valid.any():
                time, magnitude, low, high = time[valid], magnitude[valid], low[valid], high[valid]
                lower = self.group_vcg_delta_plot.plot(time, low, pen=pg.mkPen("#a9c9d0", width=1))
                upper = self.group_vcg_delta_plot.plot(time, high, pen=pg.mkPen("#a9c9d0", width=1))
                self.group_vcg_delta_plot.addItem(pg.FillBetweenItem(
                    lower, upper, brush=pg.mkBrush(40, 109, 135, 45),
                ))
                self.group_vcg_delta_plot.plot(time, magnitude, pen=pg.mkPen("#286d87", width=2.5))
                self.group_vcg_delta_plot.addItem(pg.InfiniteLine(
                    pos=0, angle=90, pen=pg.mkPen("#777777", width=1, style=Qt.PenStyle.DashLine),
                ))

    def _clear_group_vcg_view(self) -> None:
        if getattr(self, "_group_vcg_gl", None) is None:
            return
        for item in getattr(self, "_group_vcg_items", ()):
            self.group_vcg_view.removeItem(item)
        self._group_vcg_items = []

    def _clear_group_vcg_projections(self) -> None:
        if hasattr(self, "group_vcg_delta_plot"):
            self.group_vcg_delta_plot.clear()
        for plot, _, _ in getattr(
            self,
            "group_vcg_projection_plots",
            {},
        ).values():
            plot.clear()

    def _render_vcg(
        self,
        vcg: np.ndarray,
        segments: Optional[list[dict]] = None,
        *,
        record_vcg: Optional[np.ndarray] = None,
    ) -> None:
        self._clear_vcg_view()
        if self._vcg_gl is None or vcg.ndim != 2 or vcg.shape[1] != 3:
            return
        finite = np.all(np.isfinite(vcg), axis=1)
        finite_points = vcg[finite]
        if len(finite_points) < 2:
            return
        scale = np.nanpercentile(np.abs(finite_points), 99)
        normalized = np.asarray(vcg, dtype=float).copy()
        if np.isfinite(scale) and scale > 0:
            normalized = normalized / scale * 3.0
        points = normalized[finite]
        self._current_case_vcg_normalized = normalized
        camera_points = points
        if (
            record_vcg is not None
            and record_vcg.ndim == 2
            and record_vcg.shape[1] == 3
        ):
            record_finite = np.all(np.isfinite(record_vcg), axis=1)
            record_points = record_vcg[record_finite]
            if len(record_points):
                record_normalized = record_points.astype(float)
                if np.isfinite(scale) and scale > 0:
                    record_normalized = record_normalized / scale * 3.0
                cloud = self._vcg_gl.GLScatterPlotItem(
                    pos=record_normalized.astype(np.float32),
                    color=(0.09, 0.51, 0.57, 0.18),
                    size=2,
                    pxMode=True,
                )
                self.vcg_view.addItem(cloud)
                self._vcg_items.append(cloud)
                camera_points = np.concatenate((points, record_normalized), axis=0)
        # Frame every displayed point, including excursions beyond the robust
        # display normalization. Camera translation does not rotate the axes
        # or alter the signal/marker values.
        from PyQt6.QtGui import QVector3D
        center, distance = _vcg_camera_bounds(
            camera_points, self.vcg_view.width(), self.vcg_view.height(),
        )
        self.vcg_view.setCameraPosition(pos=QVector3D(*map(float, center)), distance=distance)
        time_colors = pg.colormap.get("viridis").map(
            np.linspace(0.0, 1.0, len(points)),
            mode="float",
        )
        line = self._vcg_gl.GLLinePlotItem(
            pos=points.astype(np.float32),
            color=np.asarray(time_colors, dtype=np.float32),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        self.vcg_view.addItem(line)
        self._vcg_items.append(line)
        segment_colors = {
            "p_wave": (0.03, 0.49, 0.55, 0.98),
            "pr_segment": (0.33, 0.48, 0.58, 0.80),
            "qrs_complex": (0.82, 0.29, 0.36, 0.98),
            "st_segment": (0.16, 0.62, 0.56, 0.92),
            "t_wave": (0.72, 0.47, 0.00, 0.98),
            "u_wave": (0.43, 0.30, 0.61, 0.96),
        }
        selected_segment = str(self.case_vcg_segment_combo.currentData() or "")
        for segment in segments or ():
            if selected_segment and str(segment.get("segment")) != selected_segment:
                continue
            start = max(0, int(segment.get("start_sample", 0)))
            end = min(
                len(normalized) - 1,
                int(segment.get("end_sample", -1)),
            )
            if end - start < 1:
                continue
            segment_points = normalized[start : end + 1]
            segment_points = segment_points[
                np.all(np.isfinite(segment_points), axis=1)
            ]
            if len(segment_points) < 2:
                continue
            segment_line = self._vcg_gl.GLLinePlotItem(
                pos=segment_points.astype(np.float32),
                color=segment_colors.get(
                    str(segment.get("segment")),
                    (0.25, 0.25, 0.25, 0.85),
                ),
                width=4,
                antialias=True,
                mode="line_strip",
            )
            self.vcg_view.addItem(segment_line)
            self._vcg_items.append(segment_line)

    def _clear_vcg_view(self) -> None:
        if self._vcg_gl is None:
            return
        for item in self._vcg_items:
            self.vcg_view.removeItem(item)
        self._vcg_items = []
        self._case_vcg_cursor_item = None

    # ------------------------------------------------------------------
    # Reports/review
    # ------------------------------------------------------------------

    def enqueue_report(self) -> None:
        run_id = self._selected_current_run_id(self.report_run_combo)
        if not run_id:
            QMessageBox.information(
                self,
                "Bericht",
                "Noch kein berichtsfähiger Run verfügbar.",
            )
            return
        formats = [
            name
            for name, check in self.format_checks.items()
            if check.isChecked()
        ]
        if not formats:
            QMessageBox.warning(
                self,
                "Bericht",
                "Mindestens ein Ausgabeformat wählen.",
            )
            return
        report_type = str(self.report_type_combo.currentData())
        endpoint_id = str(self.report_endpoint_combo.currentData() or "")
        feature_id = str(self.report_marker_combo.currentData() or "")
        if report_type == "endpoint" and not endpoint_id:
            QMessageBox.information(
                self,
                "Klinisches Ziel auswählen",
                "Bitte wählen Sie die Pathologie oder das klinische Ziel für diesen Bericht.",
            )
            return
        if report_type == "marker" and not feature_id:
            QMessageBox.information(
                self,
                "Biomarker auswählen",
                "Bitte wählen Sie den Biomarker für diesen Bericht.",
            )
            return
        report_id = (
            f"{run_id}-{report_type}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        try:
            record = self.job_store.create(
                stage="report-build",
                project_file=self.config.project_file,
                arguments={
                    "run_id": run_id,
                    "report_id": report_id,
                    "report_type": report_type,
                    "language": self.report_language.currentData(),
                    "formats": formats,
                    "panel_ids": ["primary_core", "sentinel"],
                    "endpoint_ids": [endpoint_id] if endpoint_id else [],
                    "feature_ids": [feature_id] if feature_id else [],
                    "sections": [
                        "summary",
                        "methods",
                        "cohort",
                        "performance",
                        "baselines",
                        "stability",
                        "discovery",
                        "review",
                        "limitations",
                        "provenance",
                    ],
                    "include_case_ids": self.include_case_ids.isChecked(),
                    "include_case_appendix": (
                        self.include_case_appendix.isChecked()
                    ),
                },
                priority=75,
                requested_by="gui",
            )
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.workspace_tabs.setCurrentIndex(TAB_ANALYSIS)
            self.refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(self, "Bericht", str(exc))

    def _update_report_scope_controls(self) -> None:
        report_type = str(self.report_type_combo.currentData() or "study")
        endpoint_visible = report_type == "endpoint"
        marker_visible = report_type == "marker"
        self.report_endpoint_label.setVisible(endpoint_visible)
        self.report_endpoint_combo.setVisible(endpoint_visible)
        self.report_marker_label.setVisible(marker_visible)
        self.report_marker_combo.setVisible(marker_visible)
        messages = {
            "study": "Gesamtübersicht des gewählten Ergebnisstands.",
            "endpoint": "Wählen Sie genau ein klinisches Ziel; alle passenden Marker und Baselines werden verglichen.",
            "marker": "Wählen Sie genau einen Biomarker; alle zugehörigen klinischen Ziele werden dargestellt.",
            "job_qc": "Technischer Verlauf, Qualitätsprüfungen und reproduzierbare Herkunft des Laufs.",
        }
        self.report_scope_hint.setText(messages.get(report_type, ""))

    def refresh_review(self) -> None:
        run_id = self._selected_current_run_id(self.report_run_combo)
        path = (
            self.config.runs_root
            / run_id
            / "reports"
            / "review_candidates.parquet"
        )
        self._active_review_path = path if path.is_file() else None
        self.review_table.set_source(self._active_review_path)
        if self._active_review_path is None:
            self.review_details.setPlainText(
                "Für diesen Ergebnisstand wurde noch keine lokale "
                "Diskordanz-Reviewliste erzeugt. Es werden keine Fälle aus "
                "einem anderen Run eingeblendet."
            )
        else:
            self.review_details.clear()
        self._refresh_report_files(run_id)

    def _show_review_details(self) -> None:
        row = self.review_table.selected_row()
        if not row:
            self.review_details.clear()
            return
        preferred_order = (
            "subject_id",
            "study_id",
            "split",
            "ecg_time",
            "endpoint_id",
            "clinical_label_state",
            "label_confidence",
            "label_time_precision",
            "label_evidence_ids",
            "label_evidence",
            "lab_context",
            "echo_context",
            "procedure_context",
            "medication_context",
            "machine_report",
            "classical_criterion",
            "classical_positive",
            "vcg_feature_id",
            "vcg_value",
            "threshold_distance",
            "signal_quality_score",
            "fiducial_confidence",
            "waveform_path",
            "discordance_category",
            "review_status",
        )
        keys = [key for key in preferred_order if key in row]
        keys.extend(key for key in row if key not in keys)
        self.review_details.setPlainText(
            "\n".join(f"{key}: {_display_value(row[key])}" for key in keys)
        )

    def _open_review_ecg(self) -> None:
        from ..lab_pair_source import resolve_review_source

        run_id = self._selected_current_run_id(self.report_run_combo)
        row = self.review_table.selected_row()
        expected = self.config.runs_root / run_id / "reports" / "review_candidates.parquet"
        if not run_id or self._active_review_path != expected or not row:
            QMessageBox.warning(self, "Fall nicht geöffnet", "Keine aktuelle, zu diesem Run gehörende Reviewzeile ausgewählt.")
            return
        try:
            source = resolve_review_source(self.config, row)
            case = self.case_library.import_case(
                source, retain_source_path=False, copy_original=False,
            )
            self.refresh_cases(preferred_case_id=case.case_id)
            if self.case_combo.currentData() != case.case_id:
                raise ValueError("Der geprüfte Fall ist in der lokalen Bibliothek nicht verfügbar")
            self.workspace_tabs.setCurrentIndex(TAB_ECG_LAB)
            self.case_status.setText(
                self.case_status.text() + "\nAus einer lokalen Development-Reviewzeile "
                "geöffnet. Reviewbefund und Signalanalyse bleiben getrennt; "
                "keine automatische Neuberechnung oder Diagnose."
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("Review ECG laboratory link rejected: %s", exc)
            QMessageBox.warning(self, "Fall nicht geöffnet", str(exc))

    def _refresh_report_files(self, run_id: str) -> None:
        root = self.config.runs_root / run_id / "reports"
        files = []
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in (
                    ".pdf",
                    ".html",
                    ".xlsx",
                    ".csv",
                    ".parquet",
                ):
                    files.append(
                        (
                            str(path.relative_to(root)),
                            path.suffix.lower().lstrip("."),
                            datetime.fromtimestamp(
                                path.stat().st_mtime
                            ).isoformat(timespec="seconds"),
                        )
                    )
        self._fill_table(self.report_files, files[-500:])

    def open_report_folder(self) -> None:
        run_id = self._selected_current_run_id(self.report_run_combo)
        if not run_id:
            QMessageBox.information(
                self,
                "Kein aktueller Berichtslauf",
                "Historische oder inkompatible Berichtspfade werden nicht "
                "geöffnet.",
            )
            return
        path = self.config.runs_root / run_id / "reports"
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    # ------------------------------------------------------------------
    # Shared widgets
    # ------------------------------------------------------------------

    def _path_row(self, field: QLineEdit) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        layout.addWidget(
            self._button(
                "",
                "Lokalen Ordner wählen",
                lambda: self._browse_directory(field),
                QStyle.StandardPixmap.SP_DirOpenIcon,
            )
        )
        return container

    def _browse_directory(self, field: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Lokalen Ordner wählen",
            field.text(),
        )
        if selected:
            field.setText(selected)

    # ---------------------------------------------------------------
    # Deep combination search: live depth and a safe way to end it
    # ---------------------------------------------------------------

    def _build_deep_search_box(self) -> QGroupBox:
        """Live state and controls for every stage that has no natural end.

        Two stages are unbounded by design: the combination search can always
        go a level deeper, and the validation can always draw more resamples.
        From the user's seat that is one activity -- keep refining this run --
        so it gets one panel rather than one per stage.

        The three stop buttons differ in what they promise to finish first,
        which is what decides whether the result set stays comparable across
        endpoints. "Fortsetzen" withdraws the request again, because running a
        month now and adding weeks later is the intended way to use this.
        """

        box = QGroupBox("Verfeinerung und separater Kombinations-Nachlauf")
        box_layout = QVBoxLayout(box)
        self.deep_search_status = QLabel("Keine offene Verfeinerung aktiv.")
        self.deep_search_status.setWordWrap(True)
        self.deep_search_status.setObjectName("mimicSurfaceBox")
        box_layout.addWidget(self.deep_search_status)
        start_afterrun = self._button("Kombinations-Nachlauf separat starten",
            "Nur nach abgeschlossenem Hauptlauf. Eigener Job; Berichte des Hauptlaufs werden nicht blockiert.",
            self.enqueue_deep_afterrun)
        box_layout.addWidget(start_afterrun)

        buttons = QHBoxLayout()
        self.deep_stop_sweep_button = self._button(
            "Nach Durchlauf beenden",
            "Empfohlen: die restlichen Endpunkte werden noch bis zur bisher "
            "flachsten erreichten Tiefe ausgewertet. Danach hat jeder Endpunkt "
            "Kombinationen - keiner bleibt leer, nur weil die Suche ihn noch "
            "nicht erreicht hatte.",
            self.stop_deep_search_after_sweep,
        )
        buttons.addWidget(self.deep_stop_sweep_button)
        self.deep_stop_endpoint_button = self._button(
            "Nach aktuellem Endpunkt beenden",
            "Der gerade laufende Endpunkt wird fertig ausgewertet, spätere "
            "werden nicht mehr begonnen. Schneller, aber die späteren "
            "Endpunkte bleiben ohne Kombinationen.",
            self.stop_deep_search_after_endpoint,
        )
        buttons.addWidget(self.deep_stop_endpoint_button)
        self.deep_stop_now_button = self._button(
            "Sofort beenden",
            "Beendet nach der laufenden Tiefenstufe. Alles bereits "
            "Geschriebene bleibt gültig, aber die Auswertung ist "
            "unvollständig und ungleichmäßig.",
            self.stop_deep_search_immediately,
        )
        buttons.addWidget(self.deep_stop_now_button)
        self.deep_resume_button = self._button(
            "Fortsetzen",
            "Nimmt eine angeforderte Beendigung zurück. Bereits Gerechnetes "
            "bleibt erhalten und wird nicht wiederholt - die Verfeinerung "
            "setzt dort an, wo sie aufgehört hat.",
            self.resume_refinement,
        )
        buttons.addWidget(self.deep_resume_button)
        buttons.addStretch(1)
        box_layout.addLayout(buttons)
        for button in (
            self.deep_stop_sweep_button,
            self.deep_stop_endpoint_button,
            self.deep_stop_now_button,
            self.deep_resume_button,
        ):
            button.setEnabled(False)
        return box

    def enqueue_deep_afterrun(self) -> None:
        try:
            from ...mimic.workflow import _stage_arguments
            run_id = self._current_run_id()
            root = self.config.runs_root/run_id
            if _read_json(root/"guided_workflow.manifest.json").get("status") != "complete":
                raise ValueError("Der aktuelle Hauptlauf muss vor dem separaten Nachlauf abgeschlossen sein.")
            provenance = _read_json(root/"development_provenance.json")
            if provenance.get("analysis_completeness") != "complete_snapshot":
                raise ValueError("Pilot-/Teilresultate erlauben keinen Produktions-Nachlauf.")
            arguments = _stage_arguments(self.config, "deep-combination-search", preset_id=self._preset_id(), run_id=run_id)
            record = self.job_store.create(stage="deep-combination-search", project_file=self.config.project_file,
                arguments=arguments, priority=20, requested_by="gui_separate_afterrun")
            ensure_supervisor(self.config)
            self._selected_job_id = record.spec.job_id
            self.refresh_jobs()
        except Exception as exc:
            LOGGER.exception("Separate combination afterrun could not start")
            QMessageBox.warning(self, "Nachlauf nicht gestartet", str(exc))

    def _deep_search_live(self):
        """Return live state only for the fingerprint-derived current run."""

        from ...mimic.combination_stage import LIVE_FILE_NAME, read_live_state

        run_id = self._current_run_id()
        if not run_id:
            return None, None
        root = self.config.runs_root / run_id / "deep_combinations"
        path = root / LIVE_FILE_NAME
        if not path.is_file():
            return None, None
        state = read_live_state(root)
        if state is None:
            return None, None
        state_run_id = str(state.get("run_id", "") or "")
        if state_run_id and state_run_id != run_id:
            LOGGER.warning(
                "Ignoring deep-search state with mismatched run id: %s != %s",
                state_run_id,
                run_id,
            )
            return None, None
        return root, state

    def _refresh_deep_search_status(self) -> None:
        from ...mimic.refinement import (
            describe_activity,
            refinement_has_saturated,
            requested_stop_mode,
        )

        root, state = self._deep_search_live()
        self._deep_search_root = root
        active = state is not None
        for button in (
            self.deep_stop_sweep_button,
            self.deep_stop_endpoint_button,
            self.deep_stop_now_button,
        ):
            button.setEnabled(active)
        pending_stop = requested_stop_mode(root) if root is not None else None
        if hasattr(self, "deep_resume_button"):
            self.deep_resume_button.setEnabled(pending_stop is not None)
        if not active:
            self.deep_search_status.setText(
                "Keine offene Verfeinerung aktiv. Tiefensuche und Validierung "
                "melden sich hier, sobald sie laufen, und lassen sich dann "
                "jederzeit beenden und später fortsetzen."
            )
            return
        lines = [describe_activity(state)]
        completed = int(state.get("endpoints_completed", 0) or 0)
        total = int(state.get("endpoints_total", 0) or state.get("total", 0) or 0)
        if completed and total:
            lines.append(
                "Fertig ausgewertete Endpunkte: {}/{}".format(completed, total)
            )
        cap = state.get("sweep_depth_cap")
        if cap:
            lines.append(
                "Abbruch läuft: restliche Endpunkte werden bis Tiefe {} "
                "ausgewertet, damit die Ergebnisse vergleichbar bleiben.".format(cap)
            )
        # The one number that answers "should I keep going". A gain that has
        # stopped shrinking means the remaining uncertainty is in the data,
        # not in the amount of computation still available.
        saturated = refinement_has_saturated(state.get("gain"), threshold=0.0005)
        if saturated is True:
            lines.append(
                "Der letzte Zuwachs hat das Ergebnis kaum noch bewegt - "
                "länger rechnen bringt hier voraussichtlich nichts mehr."
            )
        elif saturated is False:
            lines.append("Das Ergebnis bewegt sich noch; länger rechnen lohnt.")
        if pending_stop is not None:
            lines.append(
                "Beendigung angefordert ({}). Mit 'Fortsetzen' wird die "
                "Anforderung zurückgenommen.".format(pending_stop)
            )
        self.deep_search_status.setText(chr(10).join(lines))

    def resume_refinement(self) -> None:
        """Withdraw a stop request so the run can be continued.

        Continuing is the normal case for these stages -- run a month, add
        weeks later -- so a stop request must not outlive the stop it caused.
        """

        from ...mimic.refinement import clear_stop

        # Re-resolve at click time. The analysis fingerprint can change after
        # the last polling tick; a cached historical root must never receive a
        # control file.
        root, _state = self._deep_search_live()
        self._deep_search_root = root
        if root is None:
            QMessageBox.information(
                self,
                "Verfeinerung",
                "Zurzeit läuft keine Verfeinerung.",
            )
            return
        clear_stop(root)
        self._refresh_deep_search_status()


    def _request_deep_search_stop(self, mode: str, explanation: str) -> None:
        from ...mimic.combination_stage import request_stop

        # Re-resolve at click time for the same reason as resume_refinement().
        root, _state = self._deep_search_live()
        self._deep_search_root = root
        if root is None:
            QMessageBox.information(
                self,
                "Tiefensuche",
                "Zurzeit läuft keine Tiefensuche.",
            )
            return
        confirmed = QMessageBox.question(
            self,
            "Tiefensuche beenden",
            explanation,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        try:
            request_stop(root, mode=mode, requested_by="gui")
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Tiefensuche", str(error))
            return
        self.deep_search_status.setText(
            "Abbruch angefordert ({}). Die Suche beendet sich, sobald sie den "
            "zugesagten Punkt erreicht hat; alles bereits Geschriebene bleibt "
            "gültig.".format(mode)
        )

    def stop_deep_search_after_sweep(self) -> None:
        self._request_deep_search_stop(
            "after_sweep",
            "Die restlichen Endpunkte werden noch bis zur bisher flachsten "
            "erreichten Tiefe ausgewertet.\n\nDamit hat am Ende jeder "
            "Endpunkt Kombinationen - keiner bleibt leer, nur weil die Suche "
            "ihn noch nicht erreicht hatte.\n\nJetzt so beenden?",
        )

    def stop_deep_search_after_endpoint(self) -> None:
        self._request_deep_search_stop(
            "after_endpoint",
            "Der gerade laufende Endpunkt wird fertig ausgewertet, spätere "
            "werden nicht mehr begonnen.\n\nDie späteren Endpunkte "
            "bleiben dann ohne Kombinationen; im Bericht ist vermerkt, welche "
            "das sind.\n\nJetzt so beenden?",
        )

    def stop_deep_search_immediately(self) -> None:
        self._request_deep_search_stop(
            "immediate",
            "Die Suche endet nach der laufenden Tiefenstufe.\n\nDer "
            "aktuelle Endpunkt bleibt dann halb ausgewertet und alle späteren "
            "ohne Kombinationen. Alles bereits Geschriebene bleibt gültig.\n\nWirklich sofort beenden?",
        )

    def _button(
        self,
        text: str,
        tooltip: str,
        slot,
        icon: Optional[QStyle.StandardPixmap] = None,
    ) -> QPushButton:
        button = QPushButton(text)
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        return button

    @staticmethod
    def _chart_guidance_label(text: str) -> QLabel:
        """Always-visible 'how to read this chart' block -- same info-box
        style already used for discovery_note/promotion_note, so a new
        explanatory paragraph reads as the same kind of content rather
        than inventing another visual language."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("mimicInfoBoxAccent")
        return label

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        return table

    @staticmethod
    def _fill_table(table: QTableWidget, rows) -> None:
        sorting_enabled = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                table.setItem(
                    row_index,
                    column_index,
                    _SortableTableItem(value),
                )
        table.setSortingEnabled(sorting_enabled)


def _vcg_camera_bounds(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    finite = np.asarray(points, dtype=float)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if not len(finite):
        return np.zeros(3), 8.0
    center = (finite.min(axis=0) + finite.max(axis=0)) / 2
    radius = float(np.max(np.linalg.norm(finite - center, axis=1)))
    half_angle = np.arctan(np.tan(np.deg2rad(30.0)) * min(1.0, max(1, height) / max(1, width)))
    return center, max(8.0, 1.15 * radius / np.sin(half_angle))


def _reserve_wrapped_label_height(widget: QWidget, lines: int) -> None:
    """Fix `widget`'s height to `lines` lines of its current font.

    Word-wrapped QLabels with dynamically-set text (status banners,
    counts) otherwise report a `sizeHint()` that changes with text
    length -- a zero-stretch QLabel in a QVBoxLayout always gets its
    full sizeHint() height, so every time the text changes (e.g. after
    a results refresh) the space left for sibling widgets shrinks or
    grows, visibly shifting the whole layout below it. Reserving a
    fixed height up front means text length no longer affects layout;
    text that would need more than `lines` lines is simply clipped
    rather than growing the box (or, for a QScrollArea wrapping a
    label with a legitimately variable line count, scrollable instead
    of clipped).
    """
    metrics = QFontMetrics(widget.font())
    line_height = metrics.lineSpacing()
    padding = widget.contentsMargins().top() + widget.contentsMargins().bottom()
    widget.setFixedHeight(line_height * lines + padding)


def _run_local_phenome_path(run_root: Path) -> Path:
    """Resolve Phenome output variants without crossing the run boundary."""

    candidates = (
        run_root / "phenome" / "broad_discovery_ranked.parquet",
        run_root / "phenome" / "broad_discovery_replicated_hits.parquet",
        run_root / "phenome" / "phenome_replicated_hits.parquet",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _run_local_transfer_path(run_root: Path) -> Path:
    """Resolve current and legacy filenames, always within one run."""

    candidates = (
        run_root / "ptb_transfer_matrix.parquet",
        run_root / "transfer_matrix.parquet",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _manifest_complete(path: Path) -> bool:
    payload = _read_json(path)
    return payload.get("status") == "complete"


def _biomarker_audit_current(
    path: Path,
    *,
    expected_implementation_hash: str,
) -> bool:
    payload = _read_json(path)
    return (
        payload.get("audit_version") == BIOMARKER_AUDIT_VERSION
        and payload.get("implementation_hash")
        == expected_implementation_hash
        and int(payload.get("calculator_errors", -1)) == 0
        and int(payload.get("default_panel_blockers", -1)) == 0
    )


def _delineation_benchmark_current(
    path: Path,
    *,
    expected_implementation_hash: str,
) -> bool:
    payload = _read_json(path)
    completed = {
        str(item) for item in payload.get("datasets_completed", ())
    }
    return (
        payload.get("suite_version") == DELINEATION_SUITE_VERSION
        and payload.get("implementation_hash") == expected_implementation_hash
        and set(DELINEATION_RELEASE_DATASETS).issubset(completed)
        and bool(payload.get("release_core_complete", False))
        and payload.get("locked_test_read") is False
    )


def _feature_manifest_complete(
    path: Path,
    *,
    expected_code_hash: Optional[str] = None,
) -> bool:
    payload = _read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("stage_version") != FEATURE_STAGE_VERSION
    ):
        return False
    if expected_code_hash is None:
        return True
    partition_index = _read_json(path.parent / "partition_index.json")
    return (
        partition_index.get("source_code_hash") == expected_code_hash
        and bool(partition_index.get("artifact_hash"))
        and int(partition_index.get("row_count", 0)) > 0
    )


def _validation_complete(run_root: Path) -> bool:
    required = (
        "screening_results.parquet",
        "classical_baseline_results.parquet",
        "machine_report_baseline_results.parquet",
        "vcg_vs_classical_comparison.parquet",
        "combination_results.parquet",
        "overfitting_stability_report.parquet",
        "performance_curve_points.parquet",
        "score_distribution_bins.parquet",
        "operating_point_results.parquet",
        "adjusted_model_results.parquet",
        "decision_curve_results.parquet",
        "development_provenance.json",
        "score_releases.json",
        "trained_marker_models.json",
        "validation_spec.json",
    )
    return all((run_root / name).is_file() for name in required)


def _pilot_validation_compatible(
    run_root: Path,
    *,
    expected_feature_code_hash: str,
    expected_validation_code_hash: str,
) -> bool:
    if not _validation_complete(run_root):
        return False
    provenance = _read_json(run_root / "development_provenance.json")
    manifest = _read_json(run_root / "validate.manifest.json")
    feature_integrity = provenance.get("feature_integrity", {})
    notes = {str(item) for item in manifest.get("notes", ())}
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("stage_version") == VALIDATION_STAGE_VERSION
        and provenance.get("feature_scope") == "development_preview"
        and provenance.get("preview") is True
        and provenance.get("analysis_completeness") == "partial_preview"
        and provenance.get("sampling_design") == "endpoint_enriched_pilot"
        and provenance.get("locked_test_labels_read") is False
        and provenance.get("prevalence_metrics_interpretable") is False
        and provenance.get("code_hash") == expected_validation_code_hash
        and isinstance(feature_integrity, dict)
        and feature_integrity.get("source_code_hash")
        == expected_feature_code_hash
        and "freeze_eligible=False" in notes
        and "prevalence_dependent_metrics=disabled_for_preview" in notes
    )


def _formal_development_validation_complete(
    run_root: Path,
    *,
    expected_feature_code_hash: Optional[str] = None,
    expected_validation_code_hash: Optional[str] = None,
) -> bool:
    if not _validation_complete(run_root):
        return False
    provenance = _read_json(run_root / "development_provenance.json")
    manifest = _read_json(run_root / "validate.manifest.json")
    scientifically_complete = (
        provenance.get("feature_scope") == "development_only"
        and provenance.get("preview") is False
        and provenance.get("analysis_completeness")
        == "complete_snapshot"
        and provenance.get("locked_test_labels_read") is False
        and manifest.get("status") == "complete"
    )
    if not scientifically_complete:
        return False
    feature_integrity = provenance.get("feature_integrity", {})
    if (
        expected_feature_code_hash is not None
        and feature_integrity.get("source_code_hash")
        != expected_feature_code_hash
    ):
        return False
    if (
        expected_validation_code_hash is not None
        and provenance.get("code_hash") != expected_validation_code_hash
    ):
        return False
    return True


def _required_group_vcg_complete(run_root: Path) -> bool:
    for endpoint_id in DEFAULT_GROUP_VCG_ENDPOINTS:
        manifest = _read_json(
            run_root
            / "group_vcg"
            / f"endpoint={endpoint_id}"
            / "split=validation"
            / "group_vcg.manifest.json"
        )
        notes = {
            str(note).strip().lower()
            for note in manifest.get("notes", ())
        }
        output_paths = tuple(manifest.get("output_paths", ()))
        if (
            manifest.get("status") != "complete"
            or "locked_test_read=false" not in notes
            or not output_paths
            or any(
                not Path(str(output)).exists()
                for output in output_paths
            )
        ):
            return False
    return True


def _validation_presentation(
    run_root: Path,
    *,
    expected_feature_code_hash: Optional[str] = None,
    expected_validation_code_hash: Optional[str] = None,
) -> tuple[str, bool]:
    if _formal_development_validation_complete(
        run_root,
        expected_feature_code_hash=expected_feature_code_hash,
        expected_validation_code_hash=expected_validation_code_hash,
    ):
        return "vollständig", True

    if (
        expected_feature_code_hash is not None
        or expected_validation_code_hash is not None
    ) and _formal_development_validation_complete(run_root):
        return (
            "historischer Entwicklungslauf "
            "(Code oder Feature-Snapshot veraltet)",
            False,
        )

    provenance = _read_json(run_root / "development_provenance.json")
    sampling_design = str(provenance.get("sampling_design", ""))
    is_preview = (
        provenance.get("preview") is True
        or provenance.get("feature_scope") == "development_preview"
        or provenance.get("analysis_completeness") == "partial_preview"
    )
    if is_preview:
        if sampling_design == "endpoint_enriched_pilot":
            return (
                "endpunktangereicherter Pilot "
                "(keine Prävalenzschätzung, nicht einfrierbar)",
                False,
            )
        return "partielle Vorschau (nicht einfrierbar)", False
    if _validation_complete(run_root):
        return "Entwicklungslauf (nicht einfrierbar)", False
    return "partiell (nicht einfrierbar)", False


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _numeric_column(
    frame: pd.DataFrame,
    names: tuple[str, ...],
) -> Optional[pd.Series]:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def _numeric_array_or_nan(
    frame: pd.DataFrame,
    name: str,
) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _string_column(
    frame: pd.DataFrame,
    name: str,
    default: str,
) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[name].fillna(default).astype(str)


def _set_categorical_ticks(
    plot: pg.PlotWidget,
    labels: list[str],
    *,
    maximum: int = 15,
) -> None:
    if not labels:
        return
    step = max(1, int(np.ceil(len(labels) / maximum)))
    ticks = [
        (index, label[:32])
        for index, label in enumerate(labels)
        if index % step == 0
    ]
    plot.getAxis("bottom").setTicks([ticks])


def _job_group_id(state: JobState) -> str:
    if state in {
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.PAUSING,
        JobState.RESUMING,
        JobState.CANCELLING,
    }:
        return "active"
    if state in {JobState.PAUSED, JobState.BLOCKED, JobState.FAILED}:
        return "attention"
    return "history"


def _gui_backend_contract_status() -> tuple[bool, str]:
    required = set(TECHNICAL_STAGE_IDS)
    required.update(guided_service_stage_ids())
    required.add("study-run")
    missing = sorted(required - set(MANAGED_SERVICE_STAGE_IDS))
    version_matches = (
        GUI_BACKEND_CONTRACT_VERSION == MIMIC_BACKEND_CONTRACT_VERSION
    )
    if missing or not version_matches:
        reasons = []
        if not version_matches:
            reasons.append(
                "GUI-Vertrag "
                f"{GUI_BACKEND_CONTRACT_VERSION} / Backend "
                f"{MIMIC_BACKEND_CONTRACT_VERSION}"
            )
        if missing:
            reasons.append("fehlende Stufen: " + ", ".join(missing))
        return False, "Oberfläche und Backend nicht synchron: " + "; ".join(reasons)
    return (
        True,
        "Oberfläche und Backend synchron · "
        f"Workflow {GUIDED_WORKFLOW_VERSION} · "
        f"Features {FEATURE_STAGE_VERSION} · "
        f"Validation {VALIDATION_STAGE_VERSION} · "
        f"PheWAS {PHENOME_STAGE_VERSION} · "
        f"Ergebnisabfrage {RESULT_QUERY_VERSION}",
    )


def _job_supersession_key(record) -> tuple[str, str]:
    arguments = dict(record.spec.arguments or {})
    study_spec = arguments.get("study_spec")
    study_run_id = (
        str(study_spec.get("run_id", ""))
        if isinstance(study_spec, dict)
        else ""
    )
    run_id = str(arguments.get("run_id", "") or study_run_id)
    if not run_id:
        run_id = str(
            record.spec.resource_profile.writer_namespace
            or record.spec.project_file
        )
    return str(record.spec.stage), run_id


def _job_run_id(record) -> str:
    """Return the scientific run identity carried by a job, if any."""

    arguments = dict(record.spec.arguments or {})
    study_spec = arguments.get("study_spec")
    study_run_id = (
        str(study_spec.get("run_id", ""))
        if isinstance(study_spec, dict)
        else ""
    )
    return str(arguments.get("run_id", "") or study_run_id)


def _job_is_pilot_only(record) -> bool:
    arguments = dict(record.spec.arguments or {})
    return bool(arguments.get("stop_after_pilots", False))


def _full_run_authorized(run_root: Path) -> bool:
    from ...mimic.scientific_release import scientific_release_valid
    if not scientific_release_valid(Path(run_root)):
        return False
    from ...mimic.scientific_release import reviewed_delta_acceptance_valid
    if reviewed_delta_acceptance_valid(Path(run_root)):
        return True
    payload = _read_json(Path(run_root) / "go_no_go" / "go_no_go.json")
    gates = payload.get("gates")
    details = payload.get("details", {})
    # A017 was complete despite three numerically invalid discovery markers.
    # Completion alone must not unlock the full-run button.
    for size in (1000, 10000):
        counts = (
            details.get(f"pilot_{size}", {}).get("features", {}).get("row_counts", {})
            if isinstance(details, dict) else {}
        )
        if counts.get("implausible_markers") != 0:
            return False
    return (
        payload.get("decision") in {"GO_FOR_FROZEN_FULL_RUN", "TECHNICAL_GO_SCIENTIFIC_REVIEW_REQUIRED"}
        and payload.get("run_id") == Path(run_root).name
        and isinstance(gates, dict)
        and bool(gates)
        and all(isinstance(value, bool) and value for value in gates.values())
        and not bool(payload.get("full_run_started", False))
    )


def _jobs_for_monitor(records, background_records, current_run_id):
    """Merge old active snapshot jobs without exposing incompatible histories/actions."""
    merged = {r.spec.job_id: r for r in records}
    merged.update({r.spec.job_id: r for r in background_records})
    active = {JobState.RUNNING, JobState.QUEUED, JobState.PAUSING,
              JobState.CANCELLING, JobState.RESUMING, JobState.PAUSED}
    return [r for r in merged.values() if _job_visible_for_current_run(r, current_run_id)
            or (r.spec.stage == 'study-run' and r.state in active)]


def _job_visible_for_current_run(record, current_run_id: str) -> bool:
    """Hide incompatible scientific histories without deleting provenance.

    Jobs without a run identity (inventory, benchmark and ingestion tools)
    remain useful for the project. A job that names a scientific run is
    actionable only when it exactly matches the fingerprint-derived current
    run id.
    """

    run_id = _job_run_id(record)
    return not run_id or run_id == str(current_run_id)


def _result_run_visible_for_current(descriptor, current_run_id: str) -> bool:
    """Expose only current, validation-compatible results in the GUI."""

    return (
        str(descriptor.run_id) == str(current_run_id)
        and bool(descriptor.validation_compatible)
    )


def _job_display_group(
    record,
    completed_by_key: dict[tuple[str, str], str],
) -> str:
    group_id = _job_group_id(record.state)
    if (
        record.state == JobState.FAILED
        and completed_by_key.get(_job_supersession_key(record), "")
        > record.spec.created_at
    ):
        return "history"
    return group_id


def _job_task_label(record) -> str:
    if record.spec.stage == "study-run":
        preset_id = str(record.spec.arguments.get("preset_id", ""))
        try:
            preset = get_analysis_preset(preset_id)
            return f"{preset.display_name_de}: klinische Kriterien finden"
        except KeyError:
            return "Geführte klinische Analyse"
    try:
        return stage_descriptor(record.spec.stage).display_name_de
    except KeyError:
        return {
            "report-build": "Druckbaren Bericht erstellen",
            "group-vcg": "Gruppen-VCG berechnen",
            "phenome-promote": "Discovery-Treffer prüfen",
        }.get(record.spec.stage, "Technische Forschungsaufgabe")


def _job_state_label(state: JobState) -> str:
    return {
        JobState.QUEUED: "Wartet auf Ressourcen",
        JobState.RUNNING: "Läuft",
        JobState.PAUSING: "Wird sicher pausiert",
        JobState.PAUSED: "Sicher pausiert",
        JobState.RESUMING: "Wird fortgesetzt",
        JobState.CANCELLING: "Wird beendet",
        JobState.CANCELLED: "Beendet",
        JobState.FAILED: "Fehlgeschlagen",
        JobState.COMPLETED: "Abgeschlossen",
        JobState.BLOCKED: "Blockiert",
    }.get(state, state.value)


def _job_progress_text(
    record,
    calibrated: Optional[CalibratedJobProgress] = None,
) -> str:
    if calibrated is not None:
        if calibrated.current_stage_fraction is not None:
            message_parts = calibrated.current_stage_message.split(";")
            stage_message = message_parts[0].strip()
            if any("letzter Feature-Checkpoint:" in part for part in message_parts[1:]):
                stage_message += "; " + next(
                    part.strip() for part in message_parts[1:]
                    if "letzter Feature-Checkpoint:" in part
                )
            stage_detail = f" · {stage_message}" if stage_message else ""
            return (
                f"Stufe {calibrated.stages_completed + 1}/{calibrated.stages_total}: "
                f"{100.0 * calibrated.current_stage_fraction:.2f} %{stage_detail}\n"
                f"Gesamt: {100.0 * calibrated.fraction:.2f} % (zeitgewichtete Schätzung)"
            )
        if calibrated.current_stage and not calibrated.current_stage_fraction_known:
            return (
                f"{calibrated.stages_completed}/{calibrated.stages_total} Stufen fertig"
                " · aktueller Anteil unbekannt"
            )
        percent = 100.0 * calibrated.fraction
        stage_text = (
            f" · Stufe {calibrated.stages_completed + 1}/{calibrated.stages_total}"
            if calibrated.current_stage
            and calibrated.stages_completed < calibrated.stages_total
            else ""
        )
        qualifier = "geschätzt " if record.state != JobState.COMPLETED else ""
        return f"{qualifier}{percent:.1f} %{stage_text}"
    if record.state == JobState.QUEUED:
        return "Noch nicht begonnen"
    if record.state != JobState.COMPLETED and not _job_has_uniform_progress(record):
        if record.progress_total > 0:
            return f"Teilphasenstand {record.progress_current:,}/{record.progress_total:,}; Gesamtanteil unbekannt"
        return record.progress_message or "Gesamtfortschritt unbekannt"
    if record.progress_total > 0:
        if record.state != JobState.COMPLETED and record.progress_current >= record.progress_total:
            return f"Records {record.progress_current:,}/{record.progress_total:,}; Abschluss ausstehend"
        current = (
            record.progress_total
            if record.state == JobState.COMPLETED
            else record.progress_current
        )
        percent = 100.0 * min(1.0, current / record.progress_total)
        return f"{percent:.1f} % ({current:,} von {record.progress_total:,})"
    if record.state == JobState.COMPLETED:
        return "100 %"
    if record.state == JobState.QUEUED:
        return "Noch nicht begonnen"
    return record.progress_message or "Fortschritt wird ermittelt"


def _job_eta_text(
    record,
    calibrated: Optional[CalibratedJobProgress] = None,
) -> str:
    if record.state == JobState.COMPLETED:
        return "fertig"
    if record.state == JobState.PAUSED:
        return "pausiert"
    if record.state in {JobState.FAILED, JobState.BLOCKED, JobState.CANCELLED}:
        return "angehalten"
    if record.state == JobState.QUEUED:
        return "wartet auf Start"
    if calibrated is not None:
        stage_eta = (
            f"Aktuelle Stufe: {_human_duration(calibrated.current_stage_remaining_seconds)}\n"
            if calibrated.current_stage_remaining_seconds is not None else ""
        )
        if calibrated.remaining_seconds is None:
            return stage_eta + "Gesamtrestarbeit unbekannt; noch nicht belastbar"
        estimate = _human_duration(calibrated.remaining_seconds)
        suffix = " + unbekannte Restarbeit" if calibrated.estimate_has_unmodelled_work else " (unsicher)"
        return f"{stage_eta}Gesamt: Planungswert {estimate}{suffix}"
    if not _job_has_uniform_progress(record):
        return "Restarbeit unbekannt; Teilphasenzaehler erlaubt keine ETA"
    if record.progress_total > 0 and record.progress_current >= record.progress_total:
        return "Zaehler abgearbeitet; Abschlussdauer noch unbekannt"
    if (
        record.progress_total <= 0
        or record.progress_current <= 0
        or not record.started_at
    ):
        return "noch nicht belastbar"
    try:
        started = datetime.fromisoformat(str(record.started_at))
        now = datetime.now(started.tzinfo or timezone.utc)
        elapsed = max(1.0, (now - started).total_seconds())
        remaining_units = max(
            0,
            record.progress_total - record.progress_current,
        )
        seconds = elapsed / record.progress_current * remaining_units
    except (TypeError, ValueError, OverflowError):
        return "noch nicht belastbar"
    return _human_duration(seconds)


def _job_has_uniform_progress(record) -> bool:
    from ...mimic.live_progress import UNIFORM_PROGRESS_STAGES
    return getattr(getattr(record, "spec", None), "stage", None) in UNIFORM_PROGRESS_STAGES


def _nonnegative_integer_count(value: object) -> Optional[int]:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _human_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "noch nicht belastbar"
    if seconds < 90:
        return "unter 2 Minuten"
    if seconds < 3600:
        return f"ca. {seconds / 60:.0f} Minuten"
    if seconds < 86_400:
        return f"ca. {seconds / 3600:.1f} Stunden"
    return f"ca. {seconds / 86_400:.1f} Tage"


def _job_result_text(record) -> str:
    if record.state == JobState.COMPLETED:
        return "Ergebnis verfügbar"
    if record.state in {
        JobState.RUNNING,
        JobState.PAUSING,
        JobState.PAUSED,
        JobState.RESUMING,
    } and record.progress_current > 0:
        return "Teilstand beschreibend sichtbar"
    if record.state in {JobState.FAILED, JobState.BLOCKED}:
        return "Kein finales Ergebnis"
    if record.state == JobState.CANCELLED:
        return "Nur fertige Teilstände"
    return "Noch kein Ergebnis"


def _job_action_required_text(record) -> str:
    if record.state == JobState.PAUSED:
        return "Fortsetzen oder beenden"
    if record.state == JobState.FAILED:
        return "Fehlerdetails prüfen, dann erneut starten"
    if record.state == JobState.BLOCKED:
        return record.error_message or "Blocker beheben"
    if record.state == JobState.CANCELLED:
        return "Bei Bedarf erneut starten"
    return "Keine"


def _job_plain_language_explanation(record) -> str:
    if record.spec.stage == "study-run":
        preset_id = str(record.spec.arguments.get("preset_id", ""))
        try:
            preset = get_analysis_preset(preset_id)
            purpose = preset.description_de
        except KeyError:
            purpose = "Führt den vollständigen klinischen Entwicklungsablauf aus."
        result = (
            "Ergebnis: klinische Referenzvergleiche, Markerleistung, "
            "Train-Grenzwerte, Stabilität, klassische Vergleiche und Berichte."
        )
        if record.state == JobState.COMPLETED and preset_id != "fast_preview":
            usability = (
                "Wissenschaftliche Verwendung: vollständige Train-/Validation-"
                "Artefakte dürfen nach bestandenem Audit zum Freeze geprüft werden."
            )
        else:
            usability = (
                "Wissenschaftliche Verwendung: laufende oder partielle Ergebnisse "
                "sind nur beschreibend und dürfen nicht eingefroren werden."
            )
    else:
        try:
            descriptor = stage_descriptor(record.spec.stage)
            purpose = descriptor.purpose_de
            result = f"Ergebnis: {descriptor.result_de}"
        except KeyError:
            purpose = "Technische Teilaufgabe der lokalen Forschungsplattform."
            result = "Ergebnis: siehe Artefakte und technisches Protokoll."
        usability = (
            "Wissenschaftliche Verwendung: Einzelergebnisse gelten nur zusammen "
            "mit dem vollständigen, kompatiblen Run-Manifest."
        )
    return (
        f"{_job_task_label(record)}\n\nWarum: {purpose}\n{result}\n"
        f"{usability}"
    )


def _job_cell_tooltip(
    record, column: int, calibrated: Optional[CalibratedJobProgress] = None,
) -> str:
    if calibrated is not None and column in {2, 3}:
        details = [calibrated.basis]
        if calibrated.current_stage:
            details.append(f"Aktuelle Stufe: {calibrated.current_stage}")
            details.append(
                "Stufenlaufzeit bis zum letzten erfassten Worker-Heartbeat: "
                + _human_duration(calibrated.current_stage_elapsed_seconds)
            )
        details.append(
            "Stufen sind unterschiedlich aufwendig; ihre Anzahl ist kein Zeitanteil. "
            "Ein Heartbeat belegt Erreichbarkeit, nicht erfolgreichen Analysefortschritt."
        )
        return "\n".join(details)
    if column == 0:
        return _job_plain_language_explanation(record)
    return (
        "Job-ID, interne Stufe, Hashes und Ressourcen stehen unter "
        "'Technische Details der gewählten Aufgabe'."
    )


def _state_color(state: JobState):
    from PyQt6.QtGui import QBrush, QColor

    colors = {
        JobState.QUEUED: "#52616b",
        JobState.RUNNING: "#087e8b",
        JobState.PAUSING: "#9a6d00",
        JobState.PAUSED: "#9a6d00",
        JobState.RESUMING: "#087e8b",
        JobState.CANCELLING: "#9a6d00",
        JobState.CANCELLED: "#68777d",
        JobState.FAILED: "#b23a48",
        JobState.COMPLETED: "#287a45",
        JobState.BLOCKED: "#b23a48",
    }
    return QBrush(QColor(colors[state]))


def _panel_color(panel_id: object):
    colors = {
        "primary_core": "#087e8b",
        "sentinel": "#d1495b",
        "discovery_reserve": "#f0a202",
        "classical": "#5f6b6d",
    }
    return pg.mkBrush(colors.get(str(panel_id), "#6c7a89"))


def _indexed_color(index: int) -> str:
    colors = (
        "#087e8b",
        "#d1495b",
        "#4c78a8",
        "#f0a202",
        "#287a45",
        "#8f5b9f",
        "#5f6b6d",
        "#c05a27",
        "#2d6a8a",
        "#a23e48",
        "#4f772d",
        "#6d597a",
    )
    return colors[int(index) % len(colors)]


def _format_ci(row: pd.Series, metric: str) -> str:
    value = _format_number(row.get(metric))
    low = _format_number(row.get(f"{metric}_ci_low"))
    high = _format_number(row.get(f"{metric}_ci_high"))
    if low and high:
        return f"{value} [{low}; {high}]"
    return value or "n/a"


def _build_guideline_comparison_rows(
    payload: dict,
    *,
    comparison_lookup: Optional[dict[tuple[str, str], dict]] = None,
) -> list[tuple]:
    """Pair each classical guideline criterion with the new research
    marker/score for the same endpoint, for exactly the currently loaded
    ECG-Labor case.

    Reuses fields already computed by ``analyze_case`` in
    ``vcg_biomarker/mimic/ecg_lab.py`` (``score_releases`` with their
    per-case ``primary_operating_point``, database-wide
    ``validation_metrics`` and per-endpoint ``classical_comparators``) —
    no new calculation, only a display-time join. ``comparison_lookup``
    (optional) adds the Work-Package-A significance verdict for the
    endpoint's prespecified lead marker against this exact classical
    criterion, keyed ``(endpoint_id, baseline_feature_id)`` — see
    ``_load_paired_comparison_lookup``. That lead marker may differ from
    the score shown in this row, so the verdict is always labeled as
    "DB-weit (Leitmarker)", never claimed to be about this row's score.
    """
    lookup = comparison_lookup or {}
    endpoint_names = {
        str(finding.get("endpoint_id", "")): str(
            finding.get(
                "endpoint_name_de",
                finding.get("endpoint_id", ""),
            )
        )
        for finding in payload.get("research_findings", ())
    }
    rows: list[tuple] = []
    for score in payload.get("score_releases", ()):
        endpoint_id = str(score.get("endpoint_id", ""))
        primary = score.get("primary_operating_point", {})
        metrics = score.get("validation_metrics", {})
        auc = _format_number(metrics.get("roc_auc"))
        auc_low = _format_number(metrics.get("roc_auc_ci_low"))
        auc_high = _format_number(metrics.get("roc_auc_ci_high"))
        db_auc_text = (
            f"{auc} [{auc_low}; {auc_high}]"
            if auc and auc_low and auc_high
            else (auc or "-")
        )
        new_marker_label = (
            f"{score.get('release_id', '')} · {score.get('stage', '')}"
        )
        comparators = list(score.get("classical_comparators", ())) or [
            {
                "display_name": "kein anerkanntes klassisches Kriterium",
                "evaluable": False,
                "positive": None,
                "value": None,
            }
        ]
        for comparator in comparators:
            if comparator.get("evaluable") is not True:
                verdict = "nicht auswertbar"
            elif comparator.get("positive") is True:
                verdict = "positiv"
            elif comparator.get("positive") is False:
                verdict = "negativ"
            else:
                verdict = "nicht auswertbar"
            criterion_id = str(comparator.get("criterion_id", ""))
            db_verdict_text = "-"
            if criterion_id:
                comparison_record = lookup.get(
                    (endpoint_id, f"classical_{criterion_id}")
                )
                if comparison_record is not None:
                    verdict_de = str(
                        comparison_record.get("comparison_verdict_de", "")
                    )
                    if (
                        comparison_record.get("comparison_type")
                        == "paired_continuous_marker_discrimination"
                    ):
                        delta_text = (
                            f"ΔAUC {_format_number(comparison_record.get('delta_auc'))}"
                            f" [{_format_number(comparison_record.get('delta_auc_ci_low'))}; "
                            f"{_format_number(comparison_record.get('delta_auc_ci_high'))}]"
                        )
                    else:
                        delta_text = (
                            "ΔSens. "
                            f"{_format_number(comparison_record.get('sensitivity_delta_vs_fixed_rule'))}"
                            f" [{_format_number(comparison_record.get('sensitivity_delta_ci_low'))}; "
                            f"{_format_number(comparison_record.get('sensitivity_delta_ci_high'))}]"
                        )
                    db_verdict_text = (
                        f"{verdict_de} ({delta_text})" if verdict_de else "-"
                    )
            rows.append(
                (
                    endpoint_names.get(endpoint_id, endpoint_id),
                    comparator.get(
                        "display_name",
                        comparator.get("criterion_id", ""),
                    ),
                    verdict,
                    _format_number(comparator.get("value")),
                    new_marker_label,
                    _format_number(score.get("research_score")),
                    _format_number(primary.get("distance_to_threshold")),
                    _case_pattern_status_text(score),
                    db_auc_text,
                    _format_number(
                        metrics.get("sensitivity_at_95_specificity")
                    ),
                    score.get("evidence_status", ""),
                    db_verdict_text,
                )
            )
    return rows


def _sensitivity_row_label(row: pd.Series) -> str:
    analysis = str(row.get("analysis_type", "sensitivity"))
    if analysis == "vcg_transform_sensitivity":
        detail = str(row.get("transform_method", "transform"))
    elif analysis == "label_window_sensitivity":
        detail = str(row.get("window_profile", "window"))
    elif analysis == "subgroup_stability":
        detail = (
            f"{row.get('subgroup_variable', 'subgroup')}="
            f"{row.get('subgroup_level', '')}"
        )
    else:
        detail = str(row.get("feature_id", ""))
    return f"{analysis} | {detail}"


def _analysis_restriction_tooltip(item: dict) -> str:
    restriction = dict(item.get("analysis_restriction", {}) or {})
    reason = str(
        item.get("not_evaluable_reason", "")
        or restriction.get("abstention_reason", "")
    ).strip()
    if not reason:
        return ""
    status = str(restriction.get("status", "")).strip()
    version = str(restriction.get("version", "")).strip()
    contract = " / ".join(value for value in (status, version) if value)
    return (
        f"Nicht beurteilbar: {reason}"
        + (f" Restriktionsvertrag: {contract}." if contract else "")
    )


def _case_pattern_status_text(item: dict) -> str:
    status = str(item.get("research_pattern_status", ""))
    restriction = dict(item.get("analysis_restriction", {}) or {})
    if restriction.get("abstention_reason"):
        return "nicht beurteilbar – P-Loop-Restriktion"
    return status


def _format_percentage(value: object) -> str:
    numeric = _numeric_sort_value(value)
    return f"{numeric:.1%}" if numeric is not None else "-"


def _value_with_unit(value: object, unit: str) -> str:
    formatted = _format_number(value)
    if not formatted:
        return "-"
    return f"{formatted} {unit}".strip()


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6g}" if np.isfinite(number) else ""


def _disable_auto_si_prefix(*plots: pg.PlotWidget) -> None:
    for plot in plots:
        for axis_name in ("bottom", "left"):
            plot.getAxis(axis_name).enableAutoSIPrefix(False)


def _numeric_sort_value(value: object) -> Optional[float]:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _display_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)
