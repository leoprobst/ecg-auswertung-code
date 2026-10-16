from __future__ import annotations

import math
import os
import platform
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import AppConfig
from ...core.analysis.threshold_sweep import threshold_sweep
from ...core.biomarkers.base import BiomarkerRegistry
from ...core.delineation.base import BeatDelineation, MarkerPoint, RecordDelineation
from ...core.pathologies.registry import PathologyRegistry
from ...core.pipeline import Pipeline, _format_eta


def _fmt_bytes(num: float) -> str:
    value = float(max(0.0, num))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def _limited_dir_size(path: Optional[Path], max_files: int = 5000) -> tuple[int, int, bool]:
    if path is None or not Path(path).exists():
        return 0, 0, False

    total = 0
    files = 0
    truncated = False
    stack = [Path(path)]

    while stack and files < max_files:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if files >= max_files:
                        truncated = True
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue

    if stack:
        truncated = True
    return total, files, truncated


class DeveloperDiagnosticsWidget(QWidget):
    """Small reproducible smoke test and capacity estimator for developers."""

    def __init__(
        self,
        config: AppConfig,
        pipeline: Pipeline,
        records_provider: Optional[Callable[[], list[int]]] = None,
        analysis_settings_provider: Optional[Callable[[], dict]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.pipeline = pipeline
        self.records_provider = records_provider
        self.analysis_settings_provider = analysis_settings_provider
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        controls = QGroupBox("Developer Diagnostics")
        form = QFormLayout(controls)

        self.synthetic_records_spin = QSpinBox()
        self.synthetic_records_spin.setRange(1, 100)
        self.synthetic_records_spin.setValue(3)
        self.synthetic_records_spin.setToolTip("Number of synthetic records used for the biomarker smoke benchmark.")
        form.addRow("Smoke records:", self.synthetic_records_spin)

        self.project_records_spin = QSpinBox()
        self.project_records_spin.setRange(1, 10_000_000)
        self.project_records_spin.setValue(100_000)
        self.project_records_spin.setToolTip("Used when no dataset selection is available.")
        form.addRow("Projection records:", self.project_records_spin)

        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(0, 10_000)
        self.bootstrap_spin.setValue(int(self.config.bootstrap_iterations))
        form.addRow("Bootstrap estimate:", self.bootstrap_spin)

        self.combo_spin = QSpinBox()
        self.combo_spin.setRange(1, 3)
        self.combo_spin.setValue(1)
        form.addRow("Max combination:", self.combo_spin)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["all registered biomarkers", "implemented only"])
        form.addRow("Biomarker scope:", self.scope_combo)

        button_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Diagnostics")
        self.run_btn.clicked.connect(self.run_diagnostics)
        button_row.addWidget(self.run_btn)

        self.copy_btn = QPushButton("Copy Output")
        self.copy_btn.clicked.connect(self._copy_output)
        button_row.addWidget(self.copy_btn)
        button_row.addStretch(1)
        form.addRow(button_row)

        layout.addWidget(controls)

        self.summary_label = QLabel("Ready")
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.output, stretch=1)

    def run_diagnostics(self):
        self.run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = self._build_report()
            self.output.setPlainText(report)
            self.summary_label.setText("Diagnostics complete")
        finally:
            QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)

    def _copy_output(self):
        QApplication.clipboard().setText(self.output.toPlainText())

    def _build_report(self) -> str:
        started = time.perf_counter()
        settings = self._current_analysis_settings()
        biomarker_names = self._biomarker_names()
        pathologies = PathologyRegistry.available()
        selected_records = self._selected_record_count()
        loaded_records = self._loaded_record_count()
        cached_records = self._cached_record_count()
        target_records = selected_records or loaded_records or self.project_records_spin.value()

        benchmark = self._run_biomarker_benchmark(self.synthetic_records_spin.value())
        sweep_benchmark = self._run_sweep_benchmark(settings["bootstrap_iterations"])

        dataset_size, dataset_files, dataset_truncated = _limited_dir_size(
            Path(self.config.ptbxl_path) if self.config.ptbxl_path else None,
            max_files=2000,
        )
        cache_dir = getattr(getattr(self.pipeline, "cache", None), "cache_dir", None)
        cache_size, cache_files, cache_truncated = _limited_dir_size(cache_dir, max_files=20_000)

        rows = []
        rows.append("VCG Biomarker Developer Diagnostics")
        rows.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        rows.append("")
        rows.extend(self._system_lines())
        rows.append("")
        rows.append("Project state")
        rows.append(f"- VCG method: {self.config.vcg_method}")
        rows.append(f"- Sampling rate: {self.config.sampling_rate} Hz ({'HR' if self.config.use_hr else 'LR'})")
        rows.append(f"- Dataset path: {self.config.ptbxl_path or 'not configured'}")
        rows.append(
            f"- Dataset sampled size: {_fmt_bytes(dataset_size)} across {dataset_files:,} files"
            f"{' (scan capped)' if dataset_truncated else ''}"
        )
        rows.append(f"- Cache path: {cache_dir or 'not configured'}")
        rows.append(
            f"- Cache size: {_fmt_bytes(cache_size)} across {cache_files:,} files"
            f"{' (scan capped)' if cache_truncated else ''}"
        )
        rows.append(f"- Loaded records: {loaded_records:,}")
        rows.append(f"- Selected records: {selected_records:,}")
        rows.append(f"- Cached records: {cached_records:,}")
        rows.append(f"- Projection records: {target_records:,}")
        rows.append("")
        rows.append("Registry")
        rows.append(f"- Biomarkers in scope: {len(biomarker_names):,}")
        rows.append(f"- Biomarker definitions total: {len(BiomarkerRegistry.definitions()):,}")
        rows.append(f"- Pathologies: {len(pathologies):,}")
        rows.append(f"- Top categories: {self._category_summary()}")
        rows.append("")
        rows.extend(self._benchmark_lines(benchmark, sweep_benchmark))
        rows.append("")
        rows.extend(self._prediction_lines(
            target_records=target_records,
            biomarker_count=len(biomarker_names),
            pathology_count=len(pathologies),
            benchmark=benchmark,
            sweep_benchmark=sweep_benchmark,
            settings=settings,
        ))
        rows.append("")
        rows.append("Pipeline/analysis decision")
        rows.append(
            "- Keep them separate: processing builds the expensive ECG/VCG/delineation/biomarker cache; "
            "analysis can then be rerun with new thresholds, bootstrap counts or correction methods."
        )
        rows.append("")
        rows.append(f"Report build time: {time.perf_counter() - started:.2f}s")
        return "\n".join(rows)

    def _current_analysis_settings(self) -> dict:
        settings = {
            "bootstrap_iterations": self.bootstrap_spin.value(),
            "max_combination": self.combo_spin.value(),
            "correction_method": self.config.multiple_testing_correction,
        }
        if self.analysis_settings_provider is not None:
            try:
                provided = self.analysis_settings_provider()
                if provided:
                    settings.update({k: v for k, v in provided.items() if v is not None})
            except Exception:
                pass
        settings["bootstrap_iterations"] = int(settings.get("bootstrap_iterations", self.bootstrap_spin.value()))
        settings["max_combination"] = int(settings.get("max_combination", self.combo_spin.value()))
        return settings

    def _biomarker_names(self) -> list[str]:
        if self.scope_combo.currentText() == "implemented only":
            return BiomarkerRegistry.available()
        return BiomarkerRegistry.available(include_metadata_only=True)

    def _selected_record_count(self) -> int:
        if self.records_provider is None:
            return 0
        try:
            return len(self.records_provider())
        except Exception:
            return 0

    def _loaded_record_count(self) -> int:
        loader = getattr(self.pipeline, "data_loader", None)
        db = getattr(loader, "_db", None)
        try:
            return int(len(db)) if db is not None else 0
        except Exception:
            return 0

    def _cached_record_count(self) -> int:
        cache = getattr(self.pipeline, "cache", None)
        if cache is None:
            return 0
        try:
            return int(cache.get_n_cached())
        except Exception:
            return 0

    def _category_summary(self) -> str:
        counts = Counter(d.category for d in BiomarkerRegistry.definitions())
        top = counts.most_common(6)
        return ", ".join(f"{name}={count}" for name, count in top) if top else "none"

    def _system_lines(self) -> list[str]:
        workers = getattr(self.pipeline, "n_workers", os.cpu_count() or 1)
        return [
            "System",
            f"- Platform: {platform.platform()}",
            f"- Python: {platform.python_version()}",
            f"- CPU cores visible: {os.cpu_count() or 1}",
            f"- Pipeline workers: {workers}",
            f"- NumPy: {np.__version__}",
        ]

    def _run_biomarker_benchmark(self, n_records: int) -> dict:
        n_records = max(1, int(n_records))
        values = []
        warnings_seen = []
        started = time.perf_counter()
        last_results = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for idx in range(n_records):
                delineation, ecg, vcg = self._synthetic_record(idx)
                last_results = BiomarkerRegistry.compute_all(
                    delineation=delineation,
                    ecg_12lead=ecg,
                    vcg_xyz=vcg,
                    fs=delineation.fs,
                )
                finite = sum(
                    1 for result in last_results
                    if result.value is not None and np.isfinite(float(result.value))
                )
                values.append(finite)
            warnings_seen = [str(w.message) for w in caught]

        elapsed = time.perf_counter() - started
        errors = sum(1 for result in last_results if str(result.computation_notes).startswith("Error:"))
        missing = sum(1 for result in last_results if result.value is None)
        return {
            "records": n_records,
            "elapsed": elapsed,
            "seconds_per_record": elapsed / n_records,
            "total_results": len(last_results),
            "finite_last_record": values[-1] if values else 0,
            "missing_last_record": missing,
            "errors_last_record": errors,
            "warnings": warnings_seen,
        }

    def _run_sweep_benchmark(self, bootstrap_iterations: int) -> dict:
        rng = np.random.default_rng(20260518)
        n = 600
        labels = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
        values = rng.normal(0.0, 1.0, n)
        values[labels == 1] += 0.65
        measured_bootstrap = min(max(int(bootstrap_iterations), 0), 25)
        started = time.perf_counter()
        threshold_sweep(
            values,
            labels,
            "synthetic_biomarker",
            "synthetic_pathology",
            bootstrap_iterations=measured_bootstrap,
        )
        elapsed = time.perf_counter() - started
        scale = max(1.0, float(bootstrap_iterations) / max(float(measured_bootstrap), 1.0))
        return {
            "n": n,
            "measured_bootstrap": measured_bootstrap,
            "target_bootstrap": bootstrap_iterations,
            "elapsed": elapsed,
            "estimated_seconds_per_sweep": elapsed * scale,
        }

    def _benchmark_lines(self, benchmark: dict, sweep_benchmark: dict) -> list[str]:
        rows = [
            "Smoke benchmark",
            f"- Synthetic records: {benchmark['records']}",
            f"- Biomarker compute time: {benchmark['elapsed']:.2f}s total, {benchmark['seconds_per_record']:.3f}s/record",
            f"- Results last record: {benchmark['total_results']:,}",
            f"- Finite values last record: {benchmark['finite_last_record']:,}",
            f"- Missing values last record: {benchmark['missing_last_record']:,}",
            f"- Calculator errors last record: {benchmark['errors_last_record']:,}",
            f"- Warnings captured: {len(benchmark['warnings'])}",
            (
                f"- Threshold sweep benchmark: {sweep_benchmark['elapsed']:.3f}s "
                f"for n={sweep_benchmark['n']}, bootstrap={sweep_benchmark['measured_bootstrap']}"
            ),
        ]
        for message, count in Counter(benchmark["warnings"]).most_common(5):
            rows.append(f"  warning x{count}: {message}")
        return rows

    def _prediction_lines(
        self,
        target_records: int,
        biomarker_count: int,
        pathology_count: int,
        benchmark: dict,
        sweep_benchmark: dict,
        settings: dict,
    ) -> list[str]:
        workers = max(1, int(getattr(self.pipeline, "n_workers", os.cpu_count() or 1)))
        per_record = max(float(benchmark["seconds_per_record"]), 1e-6)
        serial_process_s = per_record * target_records
        parallel_process_s = serial_process_s / workers * 1.25

        single_sweeps = biomarker_count * pathology_count
        requested_combo_sweeps, effective_combo_sweeps, skipped_combo_sweeps = (
            self._combination_sweep_counts(
                biomarker_count=biomarker_count,
                pathology_count=pathology_count,
                max_combination=int(settings["max_combination"]),
            )
        )
        total_sweeps = single_sweeps + effective_combo_sweeps
        sweep_seconds = self._estimated_sweep_seconds_for_records(
            sweep_benchmark,
            target_records,
        )
        analysis_s = total_sweeps * sweep_seconds / workers * 1.15

        cache_columns = 3 + biomarker_count + pathology_count
        csv_cache_size = target_records * cache_columns * 14.0
        parquet_cache_size = target_records * cache_columns * 6.0
        result_csv_size = max(1, total_sweeps) * 360.0
        evidence_json_size = max(1, total_sweeps) * 1600.0
        macro_batch_size = int(getattr(self.pipeline, "macro_batch_size", 500) or 500)
        batch_records = min(max(1, target_records), max(1, macro_batch_size))
        signal_memory = self.config.sampling_rate * 10 * 12 * 4 * batch_records

        rows = [
            "Predictions",
            f"- Processing lower-bound serial: {_format_eta(serial_process_s)}",
            f"- Processing lower-bound with {workers} workers: {_format_eta(parallel_process_s)}",
            f"- Single biomarker/pathology sweeps: {single_sweeps:,}",
            f"- Combination sweeps requested/effective: {requested_combo_sweeps:,}/{effective_combo_sweeps:,}",
            f"- Combination sweeps skipped by max_combination_tasks: {skipped_combo_sweeps:,}",
            f"- Analysis estimate: {_format_eta(analysis_s)}",
            f"- Cache CSV estimate: {_fmt_bytes(csv_cache_size)}",
            f"- Cache parquet estimate: {_fmt_bytes(parquet_cache_size)}",
            f"- Result CSV estimate: {_fmt_bytes(result_csv_size)}",
            f"- Evidence JSON estimate: {_fmt_bytes(evidence_json_size)}",
            f"- Signal RAM per streaming macro-batch estimate: {_fmt_bytes(signal_memory)}",
            f"- Correction: {settings['correction_method']}, bootstrap: {settings['bootstrap_iterations']}",
        ]
        if skipped_combo_sweeps:
            rows.append(
                f"- Note: the real pipeline skips each combination size that exceeds "
                f"max_combination_tasks={self.config.max_combination_tasks:,}; "
                "increase the limit only after a smaller screening run."
            )
        return rows

    def _combination_sweep_counts(
        self,
        biomarker_count: int,
        pathology_count: int,
        max_combination: int,
    ) -> tuple[int, int, int]:
        requested = 0
        effective = 0
        max_tasks = int(self.config.max_combination_tasks)

        for size in (2, 3):
            if max_combination < size or biomarker_count < size:
                continue
            sweeps = math.comb(biomarker_count, size) * pathology_count
            requested += sweeps
            if sweeps <= max_tasks:
                effective += sweeps

        return requested, effective, requested - effective

    def _estimated_sweep_seconds_for_records(self, sweep_benchmark: dict, target_records: int) -> float:
        benchmark_n = max(1, int(sweep_benchmark.get("n", 1)))
        target_n = max(1, int(target_records))
        per_sweep = max(float(sweep_benchmark["estimated_seconds_per_sweep"]), 1e-9)
        return per_sweep * max(1.0, target_n / benchmark_n)

    def _synthetic_record(self, seed: int) -> tuple[RecordDelineation, np.ndarray, np.ndarray]:
        fs = int(self.config.sampling_rate or 500)
        n = fs * 4
        t = np.arange(n, dtype=float) / fs
        beat_centers = np.array([0.8, 1.6, 2.4, 3.2]) * fs
        rng = np.random.default_rng(seed)

        vcg = np.zeros((n, 3), dtype=float)
        for center_sample in beat_centers:
            center = center_sample / fs
            tau = t - center
            qrs = np.exp(-0.5 * (tau / 0.035) ** 2)
            tw = np.exp(-0.5 * ((tau - 0.28) / 0.09) ** 2)
            pw = np.exp(-0.5 * ((tau + 0.18) / 0.055) ** 2)
            vcg[:, 0] += 900.0 * qrs * (tau / 0.035) + 210.0 * tw + 60.0 * pw
            vcg[:, 1] += 550.0 * qrs * (1.0 - (tau / 0.045) ** 2) + 150.0 * tw - 45.0 * pw
            vcg[:, 2] += 280.0 * qrs * np.sin(40.0 * tau) + 110.0 * tw * np.cos(9.0 * tau)

        vcg += rng.normal(0.0, 0.5, vcg.shape)
        weights = np.array([
            [0.90, 0.10, -0.15],
            [0.45, 0.85, -0.05],
            [-0.45, 0.75, 0.10],
            [-0.80, -0.35, 0.05],
            [0.70, -0.25, -0.10],
            [-0.10, 0.95, 0.15],
            [-0.30, 0.10, 0.95],
            [-0.05, 0.25, 1.00],
            [0.25, 0.35, 0.85],
            [0.55, 0.30, 0.55],
            [0.75, 0.20, 0.30],
            [0.90, 0.10, 0.10],
        ])
        ecg = vcg @ weights.T

        beats = []
        for center in beat_centers.astype(int):
            beats.append(BeatDelineation(
                r_peak=MarkerPoint(center, 0.98, "developer_synthetic"),
                p_onset=MarkerPoint(center - int(0.22 * fs), 0.90, "developer_synthetic"),
                p_peak=MarkerPoint(center - int(0.18 * fs), 0.90, "developer_synthetic"),
                p_offset=MarkerPoint(center - int(0.12 * fs), 0.90, "developer_synthetic"),
                q_peak=MarkerPoint(center - int(0.03 * fs), 0.96, "developer_synthetic"),
                r_onset=MarkerPoint(center - int(0.05 * fs), 0.96, "developer_synthetic"),
                r_offset=MarkerPoint(center + int(0.05 * fs), 0.96, "developer_synthetic"),
                s_peak=MarkerPoint(center + int(0.03 * fs), 0.96, "developer_synthetic"),
                j_point=MarkerPoint(center + int(0.06 * fs), 0.92, "developer_synthetic"),
                t_onset=MarkerPoint(center + int(0.16 * fs), 0.92, "developer_synthetic"),
                t_peak=MarkerPoint(center + int(0.28 * fs), 0.94, "developer_synthetic"),
                t_offset=MarkerPoint(center + int(0.42 * fs), 0.92, "developer_synthetic"),
            ))

        delineation = RecordDelineation(
            beats=beats,
            fs=fs,
            n_samples=n,
            ecg_id=900000 + seed,
            quality_score=0.95,
            source_method="developer_synthetic",
        )
        return delineation, ecg.astype(float), vcg.astype(float)
