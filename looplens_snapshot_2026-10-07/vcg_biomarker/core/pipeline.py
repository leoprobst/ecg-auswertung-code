from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from math import comb
from typing import Callable, Optional

import numpy as np
import pandas as pd


def _format_eta(seconds: float) -> str:
    if seconds < 0 or np.isnan(seconds) or np.isinf(seconds):
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"

from ..config import AppConfig
from .data_loader import PTBXLDataLoader, ECGRecord
from .preprocessing import (
    construct_robust_morphology_record,
    preprocess_record,
)
from .vcg_transform import VCGTransformRegistry, VCGResult
from .delineation.fusion_delineator import FusionDelineator
from .delineation.base import RecordDelineation
from .biomarkers.base import BiomarkerRegistry, BiomarkerResult
from .pathologies.registry import PathologyRegistry
from .analysis.threshold_sweep import (
    compute_auc,
    compute_roc_curve,
    quick_auc,
    threshold_sweep,
    SweepResult,
)
from .analysis.statistics import apply_multiple_testing_correction
from .analysis.results import AnalysisRun
from .analysis.normative_ranges import NormativeRangeStore


class PipelineState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    TRANSFORMING = "transforming"
    DELINEATING = "delineating"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    ERROR = "error"


class PipelineCancelled(RuntimeError):
    """Raised when a user-requested cancel stops a long-running job."""


@dataclass
class PipelineProgress:
    state: PipelineState = PipelineState.IDLE
    current_step: int = 0
    total_steps: int = 0
    current_record: int = 0
    total_records: int = 0
    message: str = ""
    elapsed_seconds: float = 0.0
    percent: int = 0


@dataclass
class ProcessedRecord:
    ecg_record: ECGRecord
    preprocessed: dict
    vcg_result: Optional[VCGResult] = None
    delineation: Optional[RecordDelineation] = None
    biomarker_results: list[BiomarkerResult] = field(default_factory=list)
    pathology_labels: dict[str, bool] = field(default_factory=dict)
    biomarker_dict: dict[str, Optional[float]] = field(default_factory=dict)
    error: Optional[str] = None


class Pipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self._state = PipelineState.IDLE
        self._progress = PipelineProgress()
        self._processed_records: list[ProcessedRecord] = []
        self._data_loader: Optional[PTBXLDataLoader] = None
        self._delineator = FusionDelineator(
            weights=config.delineation_weights if config else None,
            conflict_threshold=config.delineation_conflict_threshold if config else 20.0,
        )
        self._cancel_requested = False
        self._on_progress: Optional[Callable[[PipelineProgress], None]] = None
        self._on_log: Optional[Callable[[str], None]] = None
        self._on_analysis_update: Optional[Callable[[object], None]] = None
        self._metadata_arrays: dict[str, np.ndarray] = {}  # age/sex, populated by _collect_data()

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def progress(self) -> PipelineProgress:
        return self._progress

    @property
    def processed_records(self) -> list[ProcessedRecord]:
        return self._processed_records

    @property
    def data_loader(self) -> Optional[PTBXLDataLoader]:
        return self._data_loader

    @data_loader.setter
    def data_loader(self, loader: Optional[PTBXLDataLoader]):
        self._data_loader = loader

    def set_progress_callback(self, callback: Callable[[PipelineProgress], None]):
        self._on_progress = callback

    def set_log_callback(self, callback: Callable[[str], None]):
        self._on_log = callback

    def set_analysis_update_callback(self, callback: Optional[Callable[[object], None]]):
        self._on_analysis_update = callback

    def _emit_progress(self):
        if self._on_progress:
            self._on_progress(self._progress)

    def _log(self, message: str):
        if self._on_log:
            self._on_log(message)

    def _emit_analysis_update(self, run):
        if self._on_analysis_update:
            self._on_analysis_update(run)

    def request_cancel(self):
        self._cancel_requested = True

    def load_data(self, ptbxl_path: str, superclasses: Optional[list[str]] = None,
                  folds: Optional[list[int]] = None, validated_only: bool = False,
                  max_records: Optional[int] = None) -> list[int]:
        self._state = PipelineState.LOADING
        self._progress.state = PipelineState.LOADING
        self._progress.message = "Loading PTB-XL database..."
        self._progress.percent = 0
        self._emit_progress()

        try:
            self._data_loader = PTBXLDataLoader(ptbxl_path, use_hr=self.config.use_hr)
            self._data_loader.load_database()

            ecg_ids = self._data_loader.get_record_ids(
                superclasses=superclasses,
                folds=folds,
                validated_only=validated_only,
            )

            if max_records is not None and len(ecg_ids) > max_records:
                ecg_ids = ecg_ids[:max_records]

            self._progress.total_records = len(ecg_ids)
            self._progress.message = f"Found {len(ecg_ids)} records"
            self._progress.percent = 100
            self._emit_progress()

            self._log(f"Loaded database: {len(ecg_ids)} records selected")
            return ecg_ids

        except Exception as e:
            self._state = PipelineState.ERROR
            self._progress.message = f"Error: {str(e)}"
            self._emit_progress()
            raise

    def process_records(self, ecg_ids: list[int], vcg_method: str = "kors",
                        biomarker_names: Optional[list[str]] = None,
                        pathology_names: Optional[list[str]] = None) -> list[ProcessedRecord]:
        if self._data_loader is None:
            raise RuntimeError("Data loader not initialized. Call load_data() first or set data_loader.")

        self._cancel_requested = False
        self._processed_records = []
        total = len(ecg_ids)
        start_time = time.time()

        self._progress.total_records = total
        self._progress.total_steps = total
        self._progress.state = PipelineState.PREPROCESSING
        self._state = PipelineState.PREPROCESSING

        n_success = 0
        n_error = 0

        for i, eid in enumerate(ecg_ids):
            if self._cancel_requested:
                self._state = PipelineState.IDLE
                self._progress.message = "Cancelled"
                self._emit_progress()
                break

            self._progress.current_record = i + 1
            self._progress.current_step = i + 1
            self._progress.percent = int(100 * (i + 1) / total)

            try:
                processed = self._process_single_record(
                    eid, vcg_method, biomarker_names, pathology_names
                )
                self._processed_records.append(processed)
                n_success += 1

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0

                self._progress.elapsed_seconds = elapsed
                self._progress.message = (
                    f"Processing {i+1}/{total} "
                    f"(OK: {n_success}, Err: {n_error}) "
                    f"ETA: {_format_eta(eta)}"
                )
                self._emit_progress()

            except Exception as e:
                n_error += 1
                err_msg = f"Error processing record {eid}: {str(e)}"
                self._log(err_msg)

                error_record = ProcessedRecord(
                    ecg_record=ECGRecord(
                        ecg_id=eid, patient_id=0, age=None, sex=None,
                        height=None, weight=None, recording_date=None,
                        scp_codes={}, report="", heart_axis=None,
                        diagnostic_superclass=[], diagnostic_subclass=[],
                        validated_by_human=False, strat_fold=0,
                    ),
                    preprocessed={},
                    error=str(e),
                )
                self._processed_records.append(error_record)

        self._state = PipelineState.COMPLETE
        self._progress.state = PipelineState.COMPLETE
        self._progress.percent = 100
        self._progress.message = f"Complete: {n_success}/{total} OK, {n_error} errors"
        self._emit_progress()

        return self._processed_records

    def _process_single_record(self, ecg_id: int, vcg_method: str,
                                biomarker_names: Optional[list[str]],
                                pathology_names: Optional[list[str]]) -> ProcessedRecord:
        record = self._data_loader.get_record(ecg_id, load_signal=True)

        if not record.has_signal:
            raise ValueError(f"No signal data for ECG ID {ecg_id}")

        self._progress.state = PipelineState.PREPROCESSING
        preprocessed = preprocess_record(record.ecg_signal, record.sampling_rate, record.r_peaks)

        self._progress.state = PipelineState.TRANSFORMING
        vcg_delineation = VCGTransformRegistry.transform(
            preprocessed["delineation"],
            vcg_method,
        )

        self._progress.state = PipelineState.DELINEATING
        delineation = self._delineator.delineate_record(
            ecg_12lead=preprocessed["cleaned"],
            r_peaks=preprocessed["r_peaks"],
            fs=record.sampling_rate,
            ecg_id=ecg_id,
            vcg_xyz=vcg_delineation.xyz,
        )
        robust_morphology, template_diagnostics = (
            construct_robust_morphology_record(
                preprocessed["morphology"],
                delineation,
            )
        )
        vcg_result = VCGTransformRegistry.transform(
            robust_morphology,
            vcg_method,
        )
        record_vcg = VCGTransformRegistry.transform(
            preprocessed["morphology"],
            vcg_method,
        )
        preprocessed["morphology_template"] = robust_morphology
        preprocessed["morphology_template_diagnostics"] = (
            template_diagnostics
        )

        self._progress.state = PipelineState.EXTRACTING
        biomarker_results = BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=robust_morphology,
            vcg_xyz=vcg_result.xyz,
            fs=record.sampling_rate,
            names=biomarker_names,
            record_ecg_12lead=preprocessed["morphology"],
            record_raw_ecg_12lead=record.ecg_signal,
            record_acquisition={"original_sampling_rate_hz": float(record.sampling_rate)},
            record_vcg_xyz=record_vcg.xyz,
        )

        biomarker_dict = {br.name: br.value for br in biomarker_results}

        pathology_labels = PathologyRegistry.check_all(record.scp_codes)
        if pathology_names:
            pathology_labels = {k: v for k, v in pathology_labels.items() if k in pathology_names}

        return ProcessedRecord(
            ecg_record=record,
            preprocessed=preprocessed,
            vcg_result=vcg_result,
            delineation=delineation,
            biomarker_results=biomarker_results,
            pathology_labels=pathology_labels,
            biomarker_dict=biomarker_dict,
        )

    def _collect_data(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        biomarker_data = {}
        pathology_data = {}
        ages: list[float] = []
        sexes: list[str] = []

        for pr in self._processed_records:
            if pr.error is not None:
                continue
            for bname, bval in pr.biomarker_dict.items():
                if bname not in biomarker_data:
                    biomarker_data[bname] = []
                biomarker_data[bname].append(bval)
            for pname, present in pr.pathology_labels.items():
                if pname not in pathology_data:
                    pathology_data[pname] = []
                pathology_data[pname].append(1 if present else 0)
            ages.append(float(pr.ecg_record.age) if pr.ecg_record.age is not None else np.nan)
            sexes.append(pr.ecg_record.sex if pr.ecg_record.sex is not None else "")

        for bname in biomarker_data:
            biomarker_data[bname] = np.array(biomarker_data[bname], dtype=float)
        for pname in pathology_data:
            pathology_data[pname] = np.array(pathology_data[pname], dtype=int)

        self._metadata_arrays = {
            "age": np.array(ages, dtype=np.float32),
            "sex": np.array(sexes, dtype=object),
        }
        return biomarker_data, pathology_data

    def run_analysis(self, correction_method: str = "fdr_bh",
                     bootstrap_iterations: int = 1000,
                     combination_strategy: str = "none",
                     greedy_seeds: int = 3,
                     greedy_max_depth: int = 2,
                     max_biomarker_combination: int = 1,
                     compute_subgroups: bool = False) -> AnalysisRun:
        self._state = PipelineState.ANALYZING
        self._progress.state = PipelineState.ANALYZING
        self._progress.message = "Running threshold sweep analysis..."
        self._progress.percent = 0
        self._emit_progress()

        biomarker_data, pathology_data = self._collect_data()
        biomarker_names = list(biomarker_data.keys())
        pathology_names = list(pathology_data.keys())
        total_expected_tasks = self._count_single_analysis_tasks(biomarker_data, pathology_data)
        total_expected_tasks += self._count_effective_combined_analysis_tasks(
            biomarker_data,
            pathology_data,
            max_biomarker_combination=max_biomarker_combination,
        )

        results: list[SweepResult] = []

        for bname in biomarker_names:
            bv = biomarker_data[bname]
            n_valid = np.sum(np.isfinite(bv))

            if n_valid < 10:
                continue

            for pname in pathology_names:
                lb = pathology_data[pname]
                if np.sum(lb) < 5 or np.sum(lb == 0) < 5:
                    continue

                valid = np.isfinite(bv)
                lb_valid = lb[valid]
                if np.sum(lb_valid) < 5 or np.sum(lb_valid == 0) < 5:
                    continue

                result = threshold_sweep(
                    biomarker_values=bv,
                    labels=lb,
                    biomarker_name=bname,
                    pathology_name=pname,
                    bootstrap_iterations=bootstrap_iterations,
                )
                results.append(result)

        if combination_strategy == "greedy":
            self._progress.message = (
                f"Greedy-Kombinationsanalyse ({greedy_seeds} Seeds/Pathologie, "
                f"max. Tiefe {greedy_max_depth})..."
            )
            self._emit_progress()
            greedy_results = self._run_greedy_combination_analysis(
                biomarker_data, pathology_data,
                single_results=results,
                greedy_seeds=greedy_seeds,
                greedy_max_depth=greedy_max_depth,
                bootstrap_iterations=bootstrap_iterations,
            )
            results.extend(greedy_results)
        elif combination_strategy == "brute_force" or max_biomarker_combination >= 2:
            self._progress.message = "Running combined biomarker analysis (pairs)..."
            self._emit_progress()
            combo_results = self._run_combined_analysis(
                biomarker_data, pathology_data,
                combination_size=2,
                bootstrap_iterations=bootstrap_iterations,
            )
            results.extend(combo_results)

            if combination_strategy == "brute_force" or max_biomarker_combination >= 3:
                self._progress.message = "Running combined biomarker analysis (triplets)..."
                self._emit_progress()
                combo_results = self._run_combined_analysis(
                    biomarker_data, pathology_data,
                    combination_size=3,
                    bootstrap_iterations=bootstrap_iterations,
                )
                results.extend(combo_results)

        # Normative ranges from NORM cohort
        norm_store = self._compute_normative_ranges(biomarker_data, pathology_data)
        self._attach_normative_ranges(results, norm_store)

        # Stratified sweeps (optional)
        if compute_subgroups:
            self._progress.message = "Subgruppenanalyse (Geschlecht + Alter)..."
            self._emit_progress()
            self._compute_subgroup_results(
                results, biomarker_data, pathology_data,
                bootstrap_iterations=min(bootstrap_iterations, 50),
            )

        run = AnalysisRun(
            run_id=f"run_{int(time.time())}",
            biomarker_names=biomarker_names,
            pathology_names=pathology_names,
            vcg_method=self.config.vcg_method,
            results=results,
            correction_method=correction_method,
            cohort_filter={
                "source": "selected_records",
                "n_processed": len(self._processed_records),
                "n_expected_pairs": int(total_expected_tasks),
                "n_current_pairs": len(results),
                "status": "complete",
            },
            split_strategy="development_only",
            comparison_baseline={"name": "standard_ecg", "status": "not_configured"},
            analysis_mode="exploratory_threshold_sweep",
        )

        run.apply_corrections()

        self._state = PipelineState.COMPLETE
        self._progress.state = PipelineState.COMPLETE
        self._progress.percent = 100
        n_single = sum(1 for r in results if "+" not in r.biomarker_name)
        n_combo = sum(1 for r in results if "+" in r.biomarker_name)
        self._progress.message = (
            f"Analysis complete: {n_single} single + {n_combo} combined = {len(results)} total"
        )
        self._emit_progress()

        return run

    # ------------------------------------------------------------------
    # Normative ranges + subgroup analysis
    # ------------------------------------------------------------------

    def _compute_normative_ranges(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
    ) -> "Optional[NormativeRangeStore]":
        """Compute normative reference ranges from the NORM subpopulation."""
        norm_arr = pathology_data.get("NORM")
        if norm_arr is None:
            return None
        norm_mask = norm_arr.astype(bool)
        if norm_mask.sum() < 30:
            return None
        age_arr = self._metadata_arrays.get("age", np.array([], dtype=np.float32))
        sex_arr = self._metadata_arrays.get("sex", np.array([], dtype=object))
        if len(age_arr) != len(norm_mask) or len(sex_arr) != len(norm_mask):
            # Metadata not aligned with data — skip stratification
            age_arr = np.full(len(norm_mask), np.nan, dtype=np.float32)
            sex_arr = np.full(len(norm_mask), "", dtype=object)
        store = NormativeRangeStore()
        store.compute(biomarker_data, age_arr, sex_arr, norm_mask)
        return store

    def _attach_normative_ranges(
        self,
        results: list[SweepResult],
        norm_store: "Optional[NormativeRangeStore]",
    ) -> None:
        """Attach P5/P95 from the NORM cohort (all strata) to each SweepResult."""
        if norm_store is None:
            return
        for result in results:
            if "+" in result.biomarker_name:
                continue
            stratum = norm_store.get_overall(result.biomarker_name)
            if stratum is not None:
                result.normative_p5 = stratum.p5
                result.normative_p95 = stratum.p95

    def _compute_subgroup_results(
        self,
        results: list[SweepResult],
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
        bootstrap_iterations: int = 50,
    ) -> None:
        """
        Run stratified threshold sweeps (sex, age group) for high-AUC results.

        Results are stored in-place in SweepResult.subgroup_metrics.
        Only single-biomarker results with AUC ≥ config.subgroup_auc_threshold are processed.
        """
        from .analysis.normative_ranges import _AGE_BINS

        min_n = int(getattr(self.config, "subgroup_min_n", 30))
        auc_threshold = float(getattr(self.config, "subgroup_auc_threshold", 0.65))

        age_arr = self._metadata_arrays.get("age", np.array([], dtype=np.float32))
        sex_arr = self._metadata_arrays.get("sex", np.array([], dtype=object))

        n_records = next((len(v) for v in biomarker_data.values()), 0)
        if len(age_arr) != n_records or len(sex_arr) != n_records:
            return  # metadata not aligned

        # Build strata masks (on the full record set)
        strata: list[tuple[str, np.ndarray]] = []
        sex_M = sex_arr == "M"
        sex_F = sex_arr == "F"
        if sex_M.sum() >= min_n:
            strata.append(("sex_M", sex_M))
        if sex_F.sum() >= min_n:
            strata.append(("sex_F", sex_F))
        for lo, hi, label in _AGE_BINS:
            age_mask = np.isfinite(age_arr) & (age_arr >= lo) & (age_arr <= hi)
            if age_mask.sum() >= min_n:
                strata.append((label, age_mask))

        if not strata:
            return

        for result in results:
            if "+" in result.biomarker_name or result.auc < auc_threshold:
                continue
            bv = biomarker_data.get(result.biomarker_name)
            lb_all = pathology_data.get(result.pathology_name)
            if bv is None or lb_all is None:
                continue

            for stratum_label, mask in strata:
                bv_sub = bv[mask]
                lb_sub = lb_all[mask]
                n_total = len(bv_sub)
                n_pos = int(np.sum(lb_sub))
                n_neg = n_total - n_pos
                if n_total < min_n or n_pos < 5 or n_neg < 5:
                    continue
                try:
                    sub_result = threshold_sweep(
                        biomarker_values=bv_sub,
                        labels=lb_sub,
                        biomarker_name=result.biomarker_name,
                        pathology_name=result.pathology_name,
                        bootstrap_iterations=bootstrap_iterations,
                    )
                    result.subgroup_metrics[stratum_label] = {
                        "n": n_total,
                        "n_pos": n_pos,
                        "auc": round(sub_result.auc, 4),
                        "threshold": round(sub_result.optimal_threshold, 4),
                        "threshold_direction": sub_result.threshold_direction,
                        "sensitivity": round(sub_result.optimal_sensitivity, 4),
                        "specificity": round(sub_result.optimal_specificity, 4),
                        "ci_auc": [round(sub_result.ci_auc[0], 4), round(sub_result.ci_auc[1], 4)]
                        if sub_result.ci_auc else None,
                    }
                except Exception:
                    pass

    def _run_combined_analysis(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
        combination_size: int = 2,
        bootstrap_iterations: int = 100,
        on_result=None,
        existing_keys: Optional[set[tuple[str, str]]] = None,
    ) -> list[SweepResult]:
        results = []
        existing_keys = existing_keys or set()
        valid_biomarkers = [
            name for name, vals in biomarker_data.items()
            if np.sum(np.isfinite(vals)) >= 10
        ]

        if len(valid_biomarkers) < combination_size:
            return results

        n_combos = comb(len(valid_biomarkers), combination_size)
        requested_total = n_combos * len(pathology_data)

        if requested_total == 0:
            return results

        max_tasks = getattr(self.config, "max_combination_tasks", 200_000)
        if requested_total > max_tasks:
            self._log(
                f"Skipping combined biomarker analysis (size {combination_size}): "
                f"{requested_total:,} sweeps exceed max_combination_tasks={max_tasks:,}. "
                "Filter biomarkers or raise the limit after pre-screening."
            )
            return results

        total_combos = self._count_effective_combined_analysis_tasks_for_size(
            biomarker_data,
            pathology_data,
            combination_size=combination_size,
        )
        if total_combos == 0:
            return results

        self._progress.message = (
            f"Combined analysis (size {combination_size}): "
            f"{total_combos} effective sweeps from {n_combos} combinations"
        )
        self._progress.current_record = 0
        self._progress.current_step = 0
        self._progress.total_records = total_combos
        self._progress.total_steps = total_combos
        self._progress.percent = 0
        self._emit_progress()

        count = 0
        progress_interval = max(
            0.5,
            float(getattr(self.config, "analysis_progress_update_interval_seconds", 1.0)),
        )
        last_progress_update_at = 0.0

        def emit_combined_progress(force: bool = False):
            nonlocal last_progress_update_at
            now = time.time()
            if not force and (now - last_progress_update_at) < progress_interval:
                return
            pct = int(100 * count / max(total_combos, 1))
            self._progress.percent = pct
            self._progress.current_record = count
            self._progress.current_step = count
            self._progress.total_records = total_combos
            self._progress.total_steps = total_combos
            self._progress.message = (
                f"Combined (size {combination_size}): {count}/{total_combos} ({pct}%)"
            )
            self._emit_progress()
            last_progress_update_at = now

        for combo in combinations(valid_biomarkers, combination_size):
            if self._cancel_requested:
                raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")
            combo_name = " + ".join(combo)

            arrays = [biomarker_data[b] for b in combo]
            valid_mask = np.ones(len(arrays[0]), dtype=bool)
            for arr in arrays:
                valid_mask &= np.isfinite(arr)

            if np.sum(valid_mask) < 10:
                continue

            valid_arrays = [arr[valid_mask] for arr in arrays]

            for pname, lb in pathology_data.items():
                key = (combo_name, pname)
                if key in existing_keys:
                    count += 1
                    emit_combined_progress(force=(count % 100 == 0))
                    continue

                lb_valid = lb[valid_mask]
                if np.sum(lb_valid) < 5 or np.sum(lb_valid == 0) < 5:
                    continue

                combined_score, component_directions = self._compute_combined_score(
                    valid_arrays,
                    method="zscore_sum",
                    labels=lb_valid,
                    align_directions=True,
                    return_directions=True,
                )

                count += 1
                result = threshold_sweep(
                    biomarker_values=combined_score,
                    labels=lb_valid,
                    biomarker_name=combo_name,
                    pathology_name=pname,
                    bootstrap_iterations=bootstrap_iterations,
                )
                result.threshold_status = "exploratory_score"
                result.validation_split = "development_in_sample"
                result.quality_summary.update({
                    "score_method": "direction_aligned_zscore_sum",
                    "direction_aligned": True,
                    "component_directions": {
                        name: direction for name, direction in zip(combo, component_directions)
                    },
                    "normalization": "complete_case_zscore",
                    "validation": "in_sample_exploratory",
                })
                results.append(result)
                existing_keys.add(key)
                if on_result is not None:
                    on_result(result)

                emit_combined_progress(force=(count % 100 == 0))

        emit_combined_progress(force=True)
        return results

    def _run_greedy_combination_analysis(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
        single_results: list[SweepResult],
        greedy_seeds: int = 3,
        greedy_max_depth: int = 2,
        bootstrap_iterations: int = 1000,
        on_result=None,
    ) -> list[SweepResult]:
        """
        Greedy forward selection of biomarker combinations.

        For each pathology, takes the top-``greedy_seeds`` single biomarkers as
        starting seeds and greedily adds the candidate that maximises AUC (using
        a fast no-bootstrap sweep).  The final combinations receive a full sweep
        with bootstrap CIs.
        """
        results: list[SweepResult] = []

        results_by_path: dict[str, list[SweepResult]] = {}
        for r in single_results:
            if "+" not in r.biomarker_name:
                results_by_path.setdefault(r.pathology_name, []).append(r)

        valid_biomarkers = [
            name for name, vals in biomarker_data.items()
            if np.sum(np.isfinite(vals)) >= 10
        ]
        valid_set = set(valid_biomarkers)

        seen_combos: set[tuple[str, ...]] = set()

        for pname, lb in pathology_data.items():
            path_single = sorted(
                results_by_path.get(pname, []),
                key=lambda r: r.auc,
                reverse=True,
            )
            seeds = [
                r.biomarker_name for r in path_single[:greedy_seeds]
                if r.biomarker_name in valid_set
            ]
            if not seeds:
                continue

            lb = np.asarray(lb, dtype=np.int8)

            for seed in seeds:
                combo: list[str] = [seed]
                base_auc = next(
                    (r.auc for r in path_single if r.biomarker_name == seed), 0.5
                )

                for _depth in range(greedy_max_depth):
                    best_candidate: Optional[str] = None
                    best_auc = base_auc

                    candidates = [b for b in valid_biomarkers if b not in combo]
                    for candidate in candidates:
                        if self._cancel_requested:
                            break
                        arrays = [biomarker_data[b] for b in combo + [candidate]]
                        valid_mask = np.ones(len(arrays[0]), dtype=bool)
                        for arr in arrays:
                            valid_mask &= np.isfinite(arr)

                        if np.sum(valid_mask) < 10:
                            continue
                        lb_valid = lb[valid_mask]
                        if np.sum(lb_valid) < 5 or np.sum(lb_valid == 0) < 5:
                            continue

                        valid_arrays = [arr[valid_mask] for arr in arrays]
                        combined, _ = self._compute_combined_score(
                            valid_arrays,
                            labels=lb_valid,
                            align_directions=True,
                            return_directions=True,
                        )
                        auc = quick_auc(combined, lb_valid)
                        if auc > best_auc:
                            best_auc = auc
                            best_candidate = candidate

                    if best_candidate is None:
                        break
                    combo.append(best_candidate)
                    base_auc = best_auc

                if len(combo) < 2:
                    continue

                combo_key = tuple(sorted(combo))
                pair_key = (combo_key, pname)
                if pair_key in seen_combos:
                    continue
                seen_combos.add(pair_key)

                combo_name = " + ".join(combo)
                arrays = [biomarker_data[b] for b in combo]
                valid_mask = np.ones(len(arrays[0]), dtype=bool)
                for arr in arrays:
                    valid_mask &= np.isfinite(arr)
                if np.sum(valid_mask) < 10:
                    continue

                lb_valid = lb[valid_mask]
                if np.sum(lb_valid) < 5 or np.sum(lb_valid == 0) < 5:
                    continue

                valid_arrays = [arr[valid_mask] for arr in arrays]
                combined_score, component_directions = self._compute_combined_score(
                    valid_arrays,
                    labels=lb_valid,
                    align_directions=True,
                    return_directions=True,
                )

                result = threshold_sweep(
                    biomarker_values=combined_score,
                    labels=lb_valid,
                    biomarker_name=combo_name,
                    pathology_name=pname,
                    bootstrap_iterations=bootstrap_iterations,
                )
                result.threshold_status = "exploratory_score"
                result.validation_split = "development_in_sample"
                result.quality_summary.update({
                    "score_method": "direction_aligned_zscore_sum",
                    "direction_aligned": True,
                    "component_directions": {
                        name: direction
                        for name, direction in zip(combo, component_directions)
                    },
                    "combination_strategy": "greedy_forward",
                    "greedy_seeds": greedy_seeds,
                    "greedy_depth": len(combo) - 1,
                    "seed_biomarker": seed,
                    "normalization": "complete_case_zscore",
                    "validation": "in_sample_exploratory",
                })
                results.append(result)
                if on_result is not None:
                    on_result(result)

        return results

    def _count_single_analysis_tasks(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
    ) -> int:
        count = 0
        for bv in biomarker_data.values():
            bv = np.asarray(bv, dtype=float)
            valid = np.isfinite(bv)
            if np.sum(valid) < 10:
                continue
            for lb in pathology_data.values():
                lb_valid = np.asarray(lb)[valid]
                if np.sum(lb_valid == 1) >= 5 and np.sum(lb_valid == 0) >= 5:
                    count += 1
        return count

    def _count_effective_combined_analysis_tasks(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
        max_biomarker_combination: int,
    ) -> int:
        total = 0
        for size in (2, 3):
            if max_biomarker_combination < size:
                continue
            total += self._count_effective_combined_analysis_tasks_for_size(
                biomarker_data,
                pathology_data,
                combination_size=size,
            )
        return total

    def _count_effective_combined_analysis_tasks_for_size(
        self,
        biomarker_data: dict[str, np.ndarray],
        pathology_data: dict[str, np.ndarray],
        combination_size: int,
    ) -> int:
        valid_biomarkers = [
            name for name, vals in biomarker_data.items()
            if np.sum(np.isfinite(vals)) >= 10
        ]
        if len(valid_biomarkers) < combination_size:
            return 0

        requested_total = comb(len(valid_biomarkers), combination_size) * len(pathology_data)
        max_tasks = getattr(self.config, "max_combination_tasks", 200_000)
        if requested_total > max_tasks:
            return 0

        count = 0
        for combo in combinations(valid_biomarkers, combination_size):
            arrays = [np.asarray(biomarker_data[b], dtype=float) for b in combo]
            valid_mask = np.ones(len(arrays[0]), dtype=bool)
            for arr in arrays:
                valid_mask &= np.isfinite(arr)
            if np.sum(valid_mask) < 10:
                continue
            for lb in pathology_data.values():
                lb_valid = np.asarray(lb)[valid_mask]
                if np.sum(lb_valid == 1) >= 5 and np.sum(lb_valid == 0) >= 5:
                    count += 1
        return count

    def _component_direction_for_labels(self, values: np.ndarray, labels: np.ndarray) -> float:
        valid = np.isfinite(values) & np.isfinite(labels.astype(float))
        bv = values[valid]
        lb = labels[valid]
        if np.sum(lb == 1) < 5 or np.sum(lb == 0) < 5:
            return 1.0
        fpr, tpr, _ = compute_roc_curve(bv, lb)
        auc = compute_auc(fpr, tpr)
        return -1.0 if auc < 0.5 else 1.0

    def _compute_combined_score(
        self,
        biomarker_arrays: list[np.ndarray],
        method: str = "zscore_sum",
        labels: Optional[np.ndarray] = None,
        align_directions: bool = False,
        return_directions: bool = False,
    ):
        n = len(biomarker_arrays[0])
        score = np.zeros(n)
        labels_arr = np.asarray(labels) if labels is not None else None
        directions: list[float] = []
        for arr in biomarker_arrays:
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                directions.append(1.0)
                continue
            direction = 1.0
            if align_directions and labels_arr is not None and len(labels_arr) == len(arr):
                direction = self._component_direction_for_labels(arr, labels_arr)
            directions.append(direction)
            std = np.std(arr)
            mean = np.mean(arr) if len(arr) > 0 else 0.0
            if std > 0:
                score += direction * ((arr - mean) / std)
            else:
                score += direction * (arr - mean)
        if return_directions:
            return score, directions
        return score

    def get_biomarker_dataframe(self) -> pd.DataFrame:
        rows = []
        for pr in self._processed_records:
            if pr.error is not None:
                continue
            row = {"ecg_id": pr.ecg_record.ecg_id}
            for br in pr.biomarker_results:
                row[br.name] = br.value
            for pname, present in pr.pathology_labels.items():
                row[f"path_{pname}"] = 1 if present else 0
            row["age"] = pr.ecg_record.age
            row["sex"] = pr.ecg_record.sex
            row["quality_score"] = pr.delineation.quality_score if pr.delineation else 0
            rows.append(row)

        return pd.DataFrame(rows)
