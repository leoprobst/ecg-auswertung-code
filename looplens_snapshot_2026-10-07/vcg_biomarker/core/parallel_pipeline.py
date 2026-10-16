from __future__ import annotations

import gc
import os
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from typing import Optional

import numpy as np

from ..config import AppConfig
from .data_loader import PTBXLDataLoader, ECGRecord
from .cache import ResultCache
from .pipeline import (
    Pipeline,
    PipelineCancelled,
    PipelineState,
    PipelineProgress,
    ProcessedRecord,
    _format_eta,
)
from .analysis.threshold_sweep import SweepResult


_NATIVE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_WORKER_THREADPOOL_CONTROLLER = None


def _limit_worker_native_threads() -> None:
    """Prevent nested BLAS/OpenMP pools inside a process worker."""
    global _WORKER_THREADPOOL_CONTROLLER
    for name in _NATIVE_THREAD_ENV:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREADPOOL_CONTROLLER = threadpool_limits(limits=1)
    except ImportError:
        _WORKER_THREADPOOL_CONTROLLER = None


def _available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _recommended_worker_count(
    *,
    estimated_worker_memory_mb: int = 512,
) -> int:
    """Choose physical-core concurrency subject to an available-RAM cap."""
    logical = max(1, int(os.cpu_count() or 1))
    try:
        import psutil

        physical = int(psutil.cpu_count(logical=False) or logical)
    except (ImportError, AttributeError, OSError, ValueError):
        physical = max(1, (logical + 1) // 2)
    available = _available_memory_bytes()
    if available is None:
        return max(1, min(logical, physical))
    reserve = max(2 * 1024**3, int(0.20 * available))
    worker_bytes = int(estimated_worker_memory_mb * 1024**2)
    usable = max(worker_bytes, available - reserve)
    ram_cap = max(1, usable // worker_bytes)
    return max(1, min(logical, physical, int(ram_cap)))


def _serialize_sweep_result(r: SweepResult) -> dict:
    return r.to_json_dict(include_curves=True)


def _deserialize_sweep_result(d: dict) -> SweepResult:
    return SweepResult.from_json_dict(d)


# ---------------------------------------------------------------------------
# Worker functions (run in subprocess — no shared state)
# ---------------------------------------------------------------------------

def _sweep_batch_worker(batch: list[tuple]) -> list[dict]:
    from vcg_biomarker.core.analysis.threshold_sweep import threshold_sweep
    results = []
    for (bv, lb, bname, pname, bootstrap_iterations) in batch:
        try:
            result = threshold_sweep(
                biomarker_values=bv,
                labels=lb,
                biomarker_name=bname,
                pathology_name=pname,
                bootstrap_iterations=bootstrap_iterations,
            )
            results.append(_serialize_sweep_result(result))
        except Exception:
            pass
    return results


_ANALYSIS_PATHOLOGY_ITEMS: list[tuple[str, np.ndarray]] = []
_ANALYSIS_BOOTSTRAP_ITERATIONS = 1000


def _init_analysis_worker(pathology_items: list[tuple[str, np.ndarray]], bootstrap_iterations: int):
    global _ANALYSIS_PATHOLOGY_ITEMS, _ANALYSIS_BOOTSTRAP_ITERATIONS
    _limit_worker_native_threads()
    _ANALYSIS_PATHOLOGY_ITEMS = pathology_items
    _ANALYSIS_BOOTSTRAP_ITERATIONS = bootstrap_iterations


def _init_record_worker() -> None:
    _limit_worker_native_threads()


def _sweep_biomarker_batch_worker(batch: list[tuple[str, np.ndarray]]) -> dict:
    from vcg_biomarker.core.analysis.threshold_sweep import threshold_sweep

    results = []
    errors = []
    attempted = 0

    for bname, bv in batch:
        try:
            valid = np.isfinite(bv)
        except Exception:
            errors.append({"biomarker": bname, "pathology": "", "error": "Biomarker array is not numeric"})
            continue

        if int(np.sum(valid)) < 10:
            continue

        for pname, lb in _ANALYSIS_PATHOLOGY_ITEMS:
            lb_valid = lb[valid]
            if int(np.sum(lb_valid)) < 5 or int(np.sum(lb_valid == 0)) < 5:
                continue

            attempted += 1
            try:
                result = threshold_sweep(
                    biomarker_values=bv,
                    labels=lb,
                    biomarker_name=bname,
                    pathology_name=pname,
                    bootstrap_iterations=_ANALYSIS_BOOTSTRAP_ITERATIONS,
                )
                results.append(_serialize_sweep_result(result))
            except Exception as exc:
                errors.append({"biomarker": bname, "pathology": pname, "error": str(exc)})

    return {"results": results, "errors": errors, "attempted": attempted}


def _process_record_batch(batch: list[tuple]) -> list[dict]:
    from vcg_biomarker.core.preprocessing import (
        construct_robust_morphology_record,
        preprocess_record,
    )
    from vcg_biomarker.core.vcg_transform import VCGTransformRegistry
    from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
    from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
    from vcg_biomarker.core.pathologies.registry import PathologyRegistry

    results = []
    for (ecg_id, signal_data, fs, scp_codes, vcg_method,
         biomarker_names, pathology_names, delineation_weights, conflict_threshold) in batch:
        try:
            preprocessed = preprocess_record(signal_data, fs)
            vcg_delineation = VCGTransformRegistry.transform(
                preprocessed["delineation"],
                vcg_method,
            )
            delineator = FusionDelineator(
                weights=delineation_weights,
                conflict_threshold=conflict_threshold,
            )
            delineation = delineator.delineate_record(
                ecg_12lead=preprocessed["cleaned"],
                r_peaks=preprocessed["r_peaks"],
                fs=fs,
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
            biomarker_results = BiomarkerRegistry.compute_all(
                delineation=delineation,
                ecg_12lead=robust_morphology,
                vcg_xyz=vcg_result.xyz,
                fs=fs,
                names=biomarker_names,
                record_ecg_12lead=preprocessed["morphology"],
                record_raw_ecg_12lead=signal_data,
                record_acquisition={"original_sampling_rate_hz": float(fs)},
                record_vcg_xyz=record_vcg.xyz,
            )
            pathology_labels = PathologyRegistry.check_all(scp_codes)
            if pathology_names:
                pathology_labels = {k: v for k, v in pathology_labels.items()
                                    if k in pathology_names}
            results.append({
                "ecg_id": ecg_id,
                "error": None,
                "quality_score": float(delineation.quality_score),
                "morphology_template_status": template_diagnostics["status"],
                "morphology_template_used_beat_count": (
                    template_diagnostics["used_beat_count"]
                ),
                "biomarker_dict": {br.name: br.value for br in biomarker_results},
                "pathology_labels": pathology_labels,
            })
        except Exception as e:
            results.append({"ecg_id": ecg_id, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# ParallelPipeline
# ---------------------------------------------------------------------------

class ParallelPipeline(Pipeline):
    """
    Memory-efficient parallel pipeline that processes records in streaming
    macro-batches.

    Memory design:
      - Only one macro-batch of raw signals is in RAM at a time
        (~120 MB for batch_size=500 at 500 Hz / 10 s / 12 leads).
      - Results are written to individual chunk files immediately after
        each batch and released; cyclic GC is amortized across batches.
      - self._processed_records is NOT populated during batch processing;
        only the last successfully processed record is kept for the ECG viewer.
      - Analysis reads directly from the cache parquet file via
        get_analysis_arrays(), not from in-memory records.
    """

    # Default: process 500 records per macro-batch.
    # Lower to reduce peak RAM; raise for fewer I/O round-trips.
    DEFAULT_MACRO_BATCH_SIZE = 500

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__(config)
        self._cache = ResultCache()
        self._n_workers = _recommended_worker_count()
        self._macro_batch_size = self.DEFAULT_MACRO_BATCH_SIZE
        # In-memory accumulators used when no cache_dir is set
        self._bm_accumulator: dict[str, list[float]] = {}
        self._path_accumulator: dict[str, list[int]] = {}
        self._active_process_executor: Optional[ProcessPoolExecutor] = None
        self._active_thread_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_error_log_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache(self) -> ResultCache:
        return self._cache

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @n_workers.setter
    def n_workers(self, value: int):
        self._n_workers = max(
            1,
            min(int(value), _recommended_worker_count()),
        )

    @property
    def macro_batch_size(self) -> int:
        """Records loaded and processed in one streaming batch."""
        return self._macro_batch_size

    @macro_batch_size.setter
    def macro_batch_size(self, value: int):
        self._macro_batch_size = max(50, int(value))

    def request_cancel(self):
        super().request_cancel()
        self._shutdown_active_executors(wait=False, terminate=True)

    def _shutdown_active_executors(self, wait: bool = False, terminate: bool = False):
        self._shutdown_executor(self._active_process_executor, wait=wait, terminate=terminate)
        self._shutdown_executor(self._active_thread_executor, wait=wait, terminate=False)

    def _shutdown_executor(self, executor, wait: bool = False, terminate: bool = False):
        if executor is None:
            return
        if terminate:
            terminate_workers = getattr(executor, "terminate_workers", None)
            if callable(terminate_workers):
                try:
                    terminate_workers()
                    return
                except Exception:
                    pass
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            try:
                executor.shutdown(wait=wait)
            except Exception:
                pass
        except Exception:
            pass

    def has_processed_data(self) -> bool:
        """True when biomarker/pathology data is available for analysis."""
        if self._cache.cache_dir and self._cache.get_n_cached() > 0:
            return True
        if self._bm_accumulator:
            return True
        return bool(self._processed_records)

    # ------------------------------------------------------------------
    # process_records — streaming, memory-bounded
    # ------------------------------------------------------------------

    def process_records(
        self,
        ecg_ids: list[int],
        vcg_method: str = "kors",
        biomarker_names: Optional[list[str]] = None,
        pathology_names: Optional[list[str]] = None,
    ) -> list[ProcessedRecord]:
        if self._data_loader is None:
            raise RuntimeError("Data loader not initialized. Call load_data() first.")

        # --- Config hash ---
        self._cache.set_config_hash({
            "ptbxl_path": str(self.config.ptbxl_path) if self.config.ptbxl_path else "",
            "vcg_method": vcg_method,
            "biomarker_names": sorted(biomarker_names) if biomarker_names else "all",
            "pathology_names": sorted(pathology_names) if pathology_names else "all",
            "delineation_weights": self.config.delineation_weights,
            "delineation_conflict_threshold": self.config.delineation_conflict_threshold,
        })

        if self._cache.cache_dir and not self._cache.check_config_valid():
            self._log("Konfiguration geaendert — Cache wird geleert")
            self._cache.clear()
            self._cache.save_config_hash()

        # --- Checkpoint manifest (written at start so crashes are trackable) ---
        from .biomarkers.base import BiomarkerRegistry as _BM
        from .pathologies.registry import PathologyRegistry as _PR
        actual_bm_names = biomarker_names if biomarker_names is not None else _BM.available()
        actual_path_names = pathology_names if pathology_names is not None else _PR.available()

        if self._cache.cache_dir:
            self._cache.save_checkpoint_manifest(
                actual_bm_names,
                actual_path_names,
                vcg_method,
                n_records=self._cache.get_n_cached(),
            )

        # --- Init ---
        self._cancel_requested = False
        self._processed_records = []
        self._bm_accumulator = {}
        self._path_accumulator = {}
        total = len(ecg_ids)
        start_time = time.time()

        self._progress.total_records = total
        self._progress.total_steps = total
        self._progress.state = PipelineState.PREPROCESSING
        self._state = PipelineState.PREPROCESSING

        # --- Find uncached IDs ---
        uncached_ids = list(ecg_ids)
        n_already_cached = 0
        if self._cache.cache_dir:
            uncached_ids = self._cache.get_uncached_ids(ecg_ids)
            n_already_cached = len(ecg_ids) - len(uncached_ids)
            if n_already_cached > 0:
                self._log(
                    f"Cache: {n_already_cached:,} bereits gecacht, "
                    f"{len(uncached_ids):,} werden neu verarbeitet"
                )

        n_success = n_already_cached
        n_error = 0
        last_ok_ecg_id: Optional[int] = None
        chunk_id_counter = self._cache.get_next_chunk_id()

        # I/O loader — hoisted outside loop, captures data_loader once
        data_loader = self._data_loader

        def _load_signal(eid: int):
            record = data_loader.get_record(eid, load_signal=True)
            return eid, record

        if not uncached_ids:
            self._finish(n_success, n_error, total)
            return self._processed_records

        total_batches = (len(uncached_ids) + self._macro_batch_size - 1) // self._macro_batch_size

        # Keep both pools alive across macro-batches; pool creation is costly
        # on Windows and repeatedly imports NumPy/SciPy in every child.
        io_workers = max(1, min(32, self._n_workers * 2, len(uncached_ids)))
        with ProcessPoolExecutor(
            max_workers=self._n_workers,
            initializer=_init_record_worker,
        ) as executor, ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="vcg-signal-io",
        ) as io_exec:
            self._active_process_executor = executor
            self._active_thread_executor = io_exec

            for batch_idx, macro_start in enumerate(
                range(0, len(uncached_ids), self._macro_batch_size)
            ):
                if self._cancel_requested:
                    self._log("Pipeline abgebrochen")
                    break

                macro_ids = uncached_ids[macro_start: macro_start + self._macro_batch_size]
                batch_num = batch_idx + 1

                self._progress.message = (
                    f"Batch {batch_num}/{total_batches}: "
                    f"Lade {len(macro_ids)} Signale ({n_success:,}/{total:,} gesamt)..."
                )
                self._progress.percent = int(100 * n_success / max(total, 1))
                self._emit_progress()

                # ── 1. Parallel I/O: load signals for this macro-batch ──────
                signal_data: list[tuple[int, np.ndarray, int, dict]] = []
                eid_to_meta: dict[int, tuple[object, object]] = {}
                io_futures = {
                    io_exec.submit(_load_signal, eid): eid for eid in macro_ids
                }
                for fut in as_completed(io_futures):
                    eid = io_futures.pop(fut)
                    try:
                        _, record = fut.result()
                        if record.has_signal:
                            signal = np.asarray(
                                record.ecg_signal,
                                dtype=np.float32,
                            )
                            signal_data.append((
                                eid,
                                signal,
                                record.sampling_rate,
                                record.scp_codes,
                            ))
                            eid_to_meta[eid] = (record.age, record.sex)
                        else:
                            n_error += 1
                    except Exception:
                        n_error += 1
                io_futures.clear()
                record = None
                signal = None

                if self._cancel_requested:
                    self._log("Pipeline abgebrochen")
                    break

                if not signal_data:
                    continue

                # Load metadata (no signal) for this batch — fast
                # ── 2. Split into worker chunks ──────────────────────────────
                n_chunks = self._n_workers * 2
                chunk_size = max(1, len(signal_data) // n_chunks)
                worker_chunks: list[list[tuple]] = []
                for i in range(0, len(signal_data), chunk_size):
                    chunk = [
                        (eid, sig, fs, scp, vcg_method,
                         biomarker_names, pathology_names,
                         dict(self.config.delineation_weights),
                         self.config.delineation_conflict_threshold)
                        for eid, sig, fs, scp in signal_data[i: i + chunk_size]
                    ]
                    worker_chunks.append(chunk)

                self._progress.message = (
                    f"Batch {batch_num}/{total_batches}: "
                    f"Verarbeite {len(signal_data)} Records auf {self._n_workers} Kernen..."
                )
                self._emit_progress()

                # ── 3. Submit chunks, collect results ────────────────────────
                batch_rows: list[dict] = []
                futures_map = {
                    executor.submit(_process_record_batch, chunk): len(chunk)
                    for chunk in worker_chunks
                }
                # ProcessPoolExecutor owns the submitted arguments now; drop
                # the batch-side containers while workers are running.
                del worker_chunks, signal_data

                for fut in as_completed(futures_map):
                    if self._cancel_requested:
                        break
                    submitted_record_count = futures_map.pop(fut)
                    try:
                        chunk_results = fut.result()
                    except Exception as exc:
                        self._log(f"Chunk-Fehler (Batch {batch_num}): {exc}")
                        n_error += submitted_record_count
                        continue

                    for r in chunk_results:
                        eid = r["ecg_id"]
                        if r.get("error") is not None:
                            n_error += 1
                            continue

                        metadata = eid_to_meta.get(eid)
                        row: dict = {
                            "ecg_id": eid,
                            "quality_score": float(
                                r.get("quality_score", 0.0)
                            ),
                        }
                        if metadata is not None:
                            age, sex = metadata
                            if age is not None:
                                row["age"] = age
                            if sex is not None:
                                row["sex"] = sex
                        for k, v in r["biomarker_dict"].items():
                            row[f"bm_{k}"] = v
                        for k, v in r["pathology_labels"].items():
                            row[f"path_{k}"] = 1 if v else 0

                        batch_rows.append(row)
                        n_success += 1
                        last_ok_ecg_id = int(eid)

                    del chunk_results

                    # Incremental progress within batch
                    self._update_progress(
                        n_success - n_already_cached,
                        len(uncached_ids),
                        n_success, n_error, start_time,
                    )

                # ── 4a. Accumulate in-memory when no cache dir ───────────────
                if batch_rows and not self._cache.cache_dir:
                    for row in batch_rows:
                        for k, v in row.items():
                            if k.startswith("bm_"):
                                self._bm_accumulator.setdefault(k[3:], []).append(v)
                            elif k.startswith("path_"):
                                self._path_accumulator.setdefault(k[5:], []).append(v)

                # ── 4b. Write batch to cache chunk file ──────────────────────
                if batch_rows and self._cache.cache_dir:
                    self._cache.store_chunk(batch_rows, chunk_id_counter)
                    chunk_id_counter += 1
                    self._cache.save_checkpoint_manifest(
                        actual_bm_names,
                        actual_path_names,
                        vcg_method,
                        n_records=self._cache.get_n_cached(),
                    )

                # ── 5. Explicitly free batch memory ──────────────────────────
                del eid_to_meta, batch_rows
                if batch_idx % 8 == 7:
                    gc.collect()

            self._active_thread_executor = None
            gc.collect()

        # --- Consolidate chunks into single batch_results file ---
        self._active_process_executor = None
        if self._cancel_requested:
            if self._cache.cache_dir:
                self._cache.save_checkpoint_manifest(
                    actual_bm_names,
                    actual_path_names,
                    vcg_method,
                    n_records=self._cache.get_n_cached(),
                )
            self._state = PipelineState.IDLE
            self._progress.state = PipelineState.IDLE
            self._progress.message = "Pipeline abgebrochen - Fortschritt gespeichert"
            self._emit_progress()
            raise PipelineCancelled("Pipeline abgebrochen. Verarbeitete Records wurden im Cache gespeichert.")

        if self._cache.cache_dir:
            self._progress.message = "Konsolidiere Cache-Dateien..."
            self._emit_progress()
            self._cache.consolidate()
            self._cache.save_config_hash()
            n_total_cached = self._cache.get_n_cached()
            self._cache.save_checkpoint_manifest(
                actual_bm_names, actual_path_names, vcg_method, n_records=n_total_cached
            )

        # Worker IPC deliberately contains scalars only. Recompute exactly one
        # detailed record for the viewer instead of transporting full signals
        # and delineations for every processed ECG.
        if last_ok_ecg_id is not None:
            try:
                detailed = super()._process_single_record(
                    last_ok_ecg_id,
                    vcg_method,
                    biomarker_names,
                    pathology_names,
                )
                if not self.config.keep_signal_data:
                    detailed.preprocessed = {
                        "fs": detailed.ecg_record.sampling_rate,
                        "r_peaks": detailed.preprocessed.get("r_peaks"),
                    }
                    detailed.vcg_result = None
                self._processed_records = [detailed]
            except Exception as exc:
                self._log(
                    "Viewer-Datensatz konnte nicht nachberechnet werden: "
                    f"{exc}"
                )

        self._finish(n_success, n_error, total)
        return self._processed_records

    # ------------------------------------------------------------------
    # _collect_data — reads from cache, not from in-memory records
    # ------------------------------------------------------------------

    def _collect_data(self) -> tuple[dict, dict]:
        """
        Override: load biomarker/pathology arrays from cache (disk) or the
        in-memory accumulator.  Falls back to the parent only when neither
        source has data (legacy / single-record mode).
        """
        if self._cache.cache_dir:
            bm_data, path_data, meta = self._cache.get_analysis_arrays_full()
            if bm_data:
                n_records = next(iter(bm_data.values())).shape[0]
                self._log(
                    f"Analyse-Daten aus Cache: {n_records:,} Records, "
                    f"{len(bm_data)} Biomarker, {len(path_data)} Pathologien"
                )
                self._metadata_arrays = meta
                return bm_data, path_data

        if self._bm_accumulator:
            bm_data = {k: np.array(v, dtype=np.float32) for k, v in self._bm_accumulator.items()}
            path_data = {k: np.array(v, dtype=np.int8) for k, v in self._path_accumulator.items()}
            n_records = next(iter(bm_data.values())).shape[0] if bm_data else 0
            self._log(
                f"Analyse-Daten aus Speicher: {n_records:,} Records, "
                f"{len(bm_data)} Biomarker, {len(path_data)} Pathologien"
            )
            return bm_data, path_data

        return super()._collect_data()

    # ------------------------------------------------------------------
    # run_analysis — parallel sweep + cache save
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        correction_method: str = "fdr_bh",
        bootstrap_iterations: int = 1000,
        combination_strategy: str = "none",
        greedy_seeds: int = 3,
        greedy_max_depth: int = 2,
        max_biomarker_combination: int = 1,
        compute_subgroups: bool = False,
    ):
        from .analysis.results import AnalysisRun

        self._cancel_requested = False
        self._analysis_error_log_count = 0
        self._state = PipelineState.ANALYZING
        self._progress.state = PipelineState.ANALYZING
        self._progress.message = "Starte parallele Threshold-Sweep-Analyse..."
        self._progress.percent = 0
        self._progress.current_record = 0
        self._progress.current_step = 0
        self._progress.total_records = 0
        self._progress.total_steps = 0
        self._progress.elapsed_seconds = 0.0
        self._emit_progress()

        biomarker_data, pathology_data = self._collect_data()

        if not biomarker_data:
            if self._cache.cache_dir is not None:
                cache_parquet = self._cache.cache_dir / "batch_results.parquet"
                cache_csv = self._cache.cache_dir / "batch_results.csv"
                if cache_parquet.exists() and not cache_csv.exists():
                    raise RuntimeError(
                        "Analyse-Daten konnten nicht geladen werden: die Datei "
                        f"'{cache_parquet}' existiert, aber kein parquet-Reader ist installiert. "
                        "Bitte 'pip install pyarrow' ausführen und die App neu starten."
                    )
                raise RuntimeError(
                    "Analyse-Daten konnten nicht geladen werden. "
                    f"Cache-Verzeichnis '{self._cache.cache_dir}' ist leer oder beschädigt. "
                    "Bitte den Batch-Lauf (Pipeline → Verarbeitung) erneut ausführen."
                )
            raise RuntimeError(
                "Keine verarbeiteten Daten vorhanden. "
                "Bitte zuerst die Pipeline-Verarbeitung starten."
            )

        biomarker_names = list(biomarker_data.keys())
        pathology_names = list(pathology_data.keys())

        run_id = f"run_{int(time.time())}"
        checkpoint_interval = max(1.0, float(getattr(self.config, "checkpoint_interval_seconds", 30.0)))
        progress_update_interval = max(
            0.5,
            float(getattr(self.config, "analysis_progress_update_interval_seconds", 1.0)),
        )
        live_update_interval = max(
            1.0,
            float(getattr(self.config, "analysis_live_update_interval_seconds", 10.0)),
        )
        last_checkpoint_at = 0.0
        last_checkpoint_result_count = -1
        last_live_update_at = 0.0
        last_live_result_count = 0
        checkpoint_merged_before_save = False
        result_by_key: dict[tuple[str, str], SweepResult] = {}

        checkpoint = self._cache.load_analysis_checkpoint() if self._cache.cache_dir else None
        if checkpoint:
            compatible, reason = self._analysis_checkpoint_compatible(
                checkpoint,
                biomarker_names,
                pathology_names,
            )
            if compatible:
                run_id = checkpoint.get("run_id", run_id)
                loaded = self._merge_analysis_checkpoint_results(
                    checkpoint,
                    result_by_key,
                    biomarker_names,
                    pathology_names,
                    max_biomarker_combination,
                )
                if loaded:
                    self._log(f"Analyse-Checkpoint geladen: {loaded:,} Ergebnisse")
                    notes = []
                    if checkpoint.get("correction_method") != correction_method:
                        notes.append("Korrektur wird neu berechnet")
                    if int(checkpoint.get("bootstrap_iterations", bootstrap_iterations)) != int(bootstrap_iterations):
                        notes.append("neue Sweeps nutzen aktuelle Bootstrap-Zahl")
                    if int(checkpoint.get("max_biomarker_combination", max_biomarker_combination)) != int(max_biomarker_combination):
                        notes.append("Kombinationsumfang wird auf aktuelle Einstellung gefiltert")
                    if notes:
                        self._log("Analyse-Checkpoint mit geaenderten Einstellungen genutzt: " + "; ".join(notes))
                else:
                    self._log("Analyse-Checkpoint gefunden, aber keine Ergebnisse passen zur aktuellen Analyse.")
            else:
                self._log(f"Analyse-Checkpoint ignoriert: {reason}")
            # It has already been read and either merged or deliberately
            # rejected; do not deserialize the full pickle again on the first
            # periodic save.
            checkpoint_merged_before_save = True

        def save_checkpoint(force: bool = False):
            nonlocal last_checkpoint_at, last_checkpoint_result_count
            nonlocal checkpoint_merged_before_save
            if not self._cache.cache_dir:
                return
            now = time.time()
            current_count = len(result_by_key)
            if current_count == last_checkpoint_result_count:
                return
            if not force:
                checkpoint_min_delta = max(
                    1,
                    int(getattr(
                        self.config,
                        "analysis_checkpoint_min_new_results",
                        128,
                    )),
                )
                if (
                    (now - last_checkpoint_at) < checkpoint_interval
                    or current_count - max(0, last_checkpoint_result_count)
                    < checkpoint_min_delta
                ):
                    return
            if not checkpoint_merged_before_save:
                existing = self._cache.load_analysis_checkpoint()
                if existing:
                    compatible, _ = self._analysis_checkpoint_compatible(
                        existing,
                        biomarker_names,
                        pathology_names,
                    )
                    if compatible:
                        restored = self._merge_analysis_checkpoint_results(
                            existing,
                            result_by_key,
                            biomarker_names,
                            pathology_names,
                            max_biomarker_combination,
                        )
                        if restored:
                            self._log(
                                f"Bestehender Analyse-Checkpoint vor dem Speichern zusammengefuehrt: "
                                f"{restored:,} zusaetzliche Ergebnisse"
                            )
                checkpoint_merged_before_save = True
            payload = self._analysis_checkpoint_payload(
                run_id=run_id,
                biomarker_names=biomarker_names,
                pathology_names=pathology_names,
                correction_method=correction_method,
                bootstrap_iterations=bootstrap_iterations,
                max_biomarker_combination=max_biomarker_combination,
                combination_strategy=combination_strategy,
                greedy_seeds=greedy_seeds,
                greedy_max_depth=greedy_max_depth,
                results=list(result_by_key.values()),
            )
            if self._cache.save_analysis_checkpoint(payload):
                last_checkpoint_at = now
                last_checkpoint_result_count = current_count
                self._log(f"Analyse-Checkpoint gespeichert: {len(result_by_key):,} Ergebnisse")

        pathology_items: list[tuple[str, np.ndarray]] = []
        for pname in pathology_names:
            lb = np.asarray(pathology_data[pname])
            if int(np.sum(lb == 1)) < 5 or int(np.sum(lb == 0)) < 5:
                continue
            pathology_items.append((pname, lb.astype(np.int8, copy=False)))

        total_expected_tasks = 0
        expected_single_keys: set[tuple[str, str]] = set()
        biomarker_items: list[tuple[str, np.ndarray]] = []
        skipped_biomarkers = 0
        for bname in biomarker_names:
            bv = np.asarray(biomarker_data[bname], dtype=np.float32)
            valid = np.isfinite(bv)
            if int(np.sum(valid)) < 10:
                continue

            eligible_path_names = []
            for pname, lb in pathology_items:
                lb_valid = lb[valid]
                if int(np.sum(lb_valid == 1)) >= 5 and int(np.sum(lb_valid == 0)) >= 5:
                    eligible_path_names.append(pname)

            if not eligible_path_names:
                continue

            total_expected_tasks += len(eligible_path_names)
            for pname in eligible_path_names:
                expected_single_keys.add((bname, pname))
            if all((bname, pname) in result_by_key for pname in eligible_path_names):
                skipped_biomarkers += 1
                continue
            biomarker_items.append((bname, bv))

        total_expected_tasks += self._count_effective_combined_analysis_tasks(
            biomarker_data,
            pathology_data,
            max_biomarker_combination=max_biomarker_combination,
        )

        completed_tasks = len(expected_single_keys.intersection(result_by_key.keys()))
        self._progress.total_records = total_expected_tasks
        self._progress.total_steps = total_expected_tasks
        self._progress.current_record = completed_tasks
        self._progress.current_step = completed_tasks
        self._progress.percent = int(100 * completed_tasks / max(total_expected_tasks, 1))

        def emit_live_update(force: bool = False):
            nonlocal last_live_update_at, last_live_result_count
            if not result_by_key or self._on_analysis_update is None:
                return
            now = time.time()
            current_count = len(result_by_key)
            if current_count == last_live_result_count:
                return
            if not force:
                live_min_delta = max(
                    1,
                    int(getattr(
                        self.config,
                        "analysis_live_update_min_new_results",
                        64,
                    )),
                )
                if (
                    (now - last_live_update_at) < live_update_interval
                    or current_count - last_live_result_count < live_min_delta
                ):
                    return
            run = self._build_analysis_run(
                run_id=run_id,
                biomarker_names=biomarker_names,
                pathology_names=pathology_names,
                results=list(result_by_key.values()),
                correction_method=correction_method,
                total_expected_tasks=total_expected_tasks,
                partial=True,
            )
            self._emit_analysis_update(run)
            last_live_update_at = now
            last_live_result_count = current_count

        if result_by_key:
            emit_live_update(force=True)

        def handle_combo_result(result: SweepResult):
            result_by_key[(result.biomarker_name, result.pathology_name)] = result
            emit_live_update()
            save_checkpoint()

        if not pathology_items or total_expected_tasks == 0:
            self._log(
                "Keine auswertbaren Analyse-Paare gefunden "
                f"({len(biomarker_items)} Biomarker, {len(pathology_items)} Pathologien nach Filter)."
            )

        if biomarker_items:
            batch_size = max(1, int(getattr(self.config, "analysis_biomarker_chunk_size", 4)))
            batches = [
                biomarker_items[i: i + batch_size]
                for i in range(0, len(biomarker_items), batch_size)
            ]
            total_batches = len(batches)
            submitted_batches = 0
            max_pending = max(1, self._n_workers * 2)
            sweep_start = time.time()
            last_progress_update_at = 0.0

            def update_analysis_progress(force: bool = False):
                nonlocal completed_tasks, last_progress_update_at
                now = time.time()
                if not force and (now - last_progress_update_at) < progress_update_interval:
                    return
                completed_tasks = len(expected_single_keys.intersection(result_by_key.keys()))
                elapsed = now - sweep_start
                rate = max(0, completed_tasks) / elapsed if elapsed > 0 else 0
                remaining = max(0, total_expected_tasks - completed_tasks)
                eta = remaining / rate if rate > 0 else 0
                pct = int(100 * completed_tasks / max(total_expected_tasks, 1))
                self._progress.percent = pct
                self._progress.current_record = completed_tasks
                self._progress.current_step = completed_tasks
                self._progress.total_records = total_expected_tasks
                self._progress.total_steps = total_expected_tasks
                self._progress.elapsed_seconds = elapsed
                self._progress.message = (
                    f"Analyse: {completed_tasks:,}/{total_expected_tasks:,} ({pct}%) "
                    f"Rate: {rate:.1f}/s ETA: {_format_eta(eta)}"
                )
                self._emit_progress()
                last_progress_update_at = now

            self._progress.message = (
                f"Parallele Analyse: {total_expected_tasks:,} Sweeps "
                f"({len(biomarker_items):,} Biomarker, {skipped_biomarkers:,} aus Checkpoint) "
                f"auf {self._n_workers} Kernen..."
            )
            self._emit_progress()
            update_analysis_progress(force=True)

            executor = ProcessPoolExecutor(
                max_workers=self._n_workers,
                initializer=_init_analysis_worker,
                initargs=(pathology_items, bootstrap_iterations),
            )
            self._active_process_executor = executor

            def submit_more(pending: dict):
                nonlocal submitted_batches
                while submitted_batches < total_batches and len(pending) < max_pending:
                    batch = batches[submitted_batches]
                    fut = executor.submit(_sweep_biomarker_batch_worker, batch)
                    pending[fut] = len(batch)
                    submitted_batches += 1

            pending = {}
            try:
                submit_more(pending)
                while pending:
                    if self._cancel_requested:
                        raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")

                    done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    if not done:
                        update_analysis_progress()
                        emit_live_update()
                        save_checkpoint()
                        continue

                    for fut in done:
                        pending.pop(fut, None)
                        try:
                            payload = fut.result()
                        except Exception as exc:
                            self._analysis_error_log_count += 1
                            if self._analysis_error_log_count <= 10:
                                self._log(f"Analyse-Batch fehlgeschlagen: {exc}")
                            continue

                        batch_results = payload.get("results", []) if isinstance(payload, dict) else []
                        batch_errors = payload.get("errors", []) if isinstance(payload, dict) else []
                        for item in batch_results:
                            try:
                                result = _deserialize_sweep_result(item)
                                result_by_key[(result.biomarker_name, result.pathology_name)] = result
                            except Exception as exc:
                                self._analysis_error_log_count += 1
                                if self._analysis_error_log_count <= 10:
                                    self._log(f"Analyse-Ergebnis konnte nicht geladen werden: {exc}")
                        for err in batch_errors[: max(0, 10 - self._analysis_error_log_count)]:
                            self._analysis_error_log_count += 1
                            self._log(
                                "Analyse-Sweep fehlgeschlagen "
                                f"{err.get('biomarker', '?')} vs {err.get('pathology', '?')}: "
                                f"{err.get('error', '?')}"
                            )

                        update_analysis_progress(force=True)
                        emit_live_update()
                        save_checkpoint()

                    if self._cancel_requested:
                        raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")
                    submit_more(pending)
            except PipelineCancelled:
                save_checkpoint(force=True)
                self._state = PipelineState.IDLE
                self._progress.state = PipelineState.IDLE
                self._progress.message = "Analyse abgebrochen - Checkpoint gespeichert"
                self._emit_progress()
                raise
            finally:
                self._shutdown_executor(
                    executor,
                    wait=not self._cancel_requested,
                    terminate=self._cancel_requested,
                )
                if self._active_process_executor is executor:
                    self._active_process_executor = None

        if total_expected_tasks > 0 and not result_by_key:
            save_checkpoint(force=True)
            raise RuntimeError(
                "Analyse hat keine Ergebnisse erzeugt, obwohl gueltige Biomarker/Pathologien vorhanden sind. "
                "Der leere Lauf wurde nicht als finales Ergebnis gespeichert."
            )

        if combination_strategy == "greedy":
            if self._cancel_requested:
                save_checkpoint(force=True)
                raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")
            self._progress.message = (
                f"Greedy-Kombinationsanalyse ({greedy_seeds} Seeds/Pathologie, "
                f"max. Tiefe {greedy_max_depth})..."
            )
            self._emit_progress()
            single_results = list(result_by_key.values())
            for result in self._run_greedy_combination_analysis(
                biomarker_data, pathology_data,
                single_results=single_results,
                greedy_seeds=greedy_seeds,
                greedy_max_depth=greedy_max_depth,
                bootstrap_iterations=bootstrap_iterations,
                on_result=handle_combo_result,
            ):
                result_by_key[(result.biomarker_name, result.pathology_name)] = result
            save_checkpoint(force=True)
            emit_live_update(force=True)

        elif combination_strategy == "brute_force" or max_biomarker_combination >= 2:
            if self._cancel_requested:
                save_checkpoint(force=True)
                raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")
            self._progress.message = "Kombinierte Biomarker-Analyse (Paare)..."
            self._emit_progress()
            for result in self._run_combined_analysis(
                biomarker_data, pathology_data,
                combination_size=2, bootstrap_iterations=bootstrap_iterations,
                on_result=handle_combo_result,
                existing_keys=set(result_by_key.keys()),
            ):
                result_by_key[(result.biomarker_name, result.pathology_name)] = result
            save_checkpoint(force=True)
            emit_live_update(force=True)

            if combination_strategy == "brute_force" or max_biomarker_combination >= 3:
                if self._cancel_requested:
                    save_checkpoint(force=True)
                    raise PipelineCancelled("Analyse abgebrochen. Fortschritt wurde als Checkpoint gespeichert.")
                self._progress.message = "Kombinierte Biomarker-Analyse (Tripel)..."
                self._emit_progress()
                for result in self._run_combined_analysis(
                    biomarker_data, pathology_data,
                    combination_size=3, bootstrap_iterations=bootstrap_iterations,
                    on_result=handle_combo_result,
                    existing_keys=set(result_by_key.keys()),
                ):
                    result_by_key[(result.biomarker_name, result.pathology_name)] = result
                save_checkpoint(force=True)
                emit_live_update(force=True)

        results = list(result_by_key.values())

        # Normative ranges from NORM cohort — computed once, attached to all results
        norm_store = self._compute_normative_ranges(biomarker_data, pathology_data)
        self._attach_normative_ranges(results, norm_store)
        if norm_store is not None and self._cache.cache_dir:
            self._cache.save_normative_ranges(norm_store)
            self._log(f"Normwerte gespeichert: {len(norm_store)} Biomarker aus NORM-Kohorte")

        # Stratified sweeps (optional, single-threaded post-processing)
        if compute_subgroups and results:
            self._progress.message = "Subgruppenanalyse (Geschlecht + Altersgruppe)..."
            self._emit_progress()
            self._compute_subgroup_results(
                results, biomarker_data, pathology_data,
                bootstrap_iterations=min(bootstrap_iterations, 50),
            )
            n_with_sub = sum(1 for r in results if r.subgroup_metrics)
            self._log(f"Subgruppenanalyse: {n_with_sub} Biomarker-Pathologie-Paare stratifiziert")

        run = self._build_analysis_run(
            run_id=run_id,
            biomarker_names=biomarker_names,
            pathology_names=pathology_names,
            results=results,
            correction_method=correction_method,
            total_expected_tasks=total_expected_tasks,
            partial=False,
        )
        self._attach_clinical_baseline(run)

        if self._cache.cache_dir:
            self._cache.save_analysis_run(run)
            self._log("Analyse-Ergebnisse im Cache gespeichert")

        self._state = PipelineState.COMPLETE
        self._progress.state = PipelineState.COMPLETE
        self._progress.percent = 100
        self._progress.current_record = total_expected_tasks
        self._progress.current_step = total_expected_tasks
        self._progress.total_records = total_expected_tasks
        self._progress.total_steps = total_expected_tasks
        n_single = sum(1 for r in results if "+" not in r.biomarker_name)
        n_combo = sum(1 for r in results if "+" in r.biomarker_name)
        self._progress.message = (
            f"Analyse abgeschlossen: {n_single} einzel + {n_combo} kombiniert "
            f"= {len(results)} gesamt"
        )
        self._emit_progress()
        return run

    def _build_analysis_run(
        self,
        run_id: str,
        biomarker_names: list[str],
        pathology_names: list[str],
        results: list[SweepResult],
        correction_method: str,
        total_expected_tasks: int,
        partial: bool = False,
    ):
        from .analysis.results import AnalysisRun

        run = AnalysisRun(
            run_id=run_id,
            biomarker_names=biomarker_names,
            pathology_names=pathology_names,
            vcg_method=self.config.vcg_method,
            results=list(results),
            correction_method=correction_method,
            cohort_filter={
                "source": "cache" if self._cache.cache_dir else "memory",
                "n_biomarkers": len(biomarker_names),
                "n_pathologies": len(pathology_names),
                "n_expected_pairs": int(total_expected_tasks),
                "n_current_pairs": len(results),
                "status": "partial" if partial else "complete",
            },
            split_strategy="development_only",
            comparison_baseline={"name": "standard_ecg", "status": "not_configured"},
            analysis_mode="exploratory_threshold_sweep_live" if partial else "exploratory_threshold_sweep",
        )
        run.apply_corrections()
        return run

    def _attach_clinical_baseline(self, run):
        from .analysis.clinical_baseline import (
            evaluate_clinical_rules,
            required_clinical_baseline_columns,
        )

        if not self._cache.cache_dir:
            run.comparison_baseline = {
                "name": "clinical_ecg_rules",
                "status": "unavailable",
                "reason": "no_cache",
            }
            return

        columns = ["ecg_id"] + required_clinical_baseline_columns()
        frame = self._cache.load_columns(columns)
        if frame is None or frame.empty:
            run.comparison_baseline = {
                "name": "clinical_ecg_rules",
                "status": "unavailable",
                "reason": "missing_cache_columns",
            }
            return

        baseline_results = evaluate_clinical_rules(frame)
        run.clinical_baseline_results = baseline_results
        n_ok = sum(1 for item in baseline_results if item.status == "ok")
        run.comparison_baseline = {
            "name": "clinical_ecg_rules",
            "status": "computed" if n_ok else "unavailable",
            "n_rules": len(baseline_results),
            "n_available": n_ok,
            "reference_standard": "PTB-XL SCP-derived pathology labels",
            "note": "Fixed clinical ECG rules are contextual benchmarks, not final diagnostic decisions.",
        }
        self._log(f"Klinische Referenzregeln ausgewertet: {n_ok}/{len(baseline_results)} auswertbar")

    def _analysis_checkpoint_compatible(
        self,
        checkpoint: dict,
        biomarker_names: list[str],
        pathology_names: list[str],
    ) -> tuple[bool, str]:
        if checkpoint.get("schema") != "vcg_biomarker.analysis_checkpoint.v1":
            return False, "unbekanntes Checkpoint-Format"
        if checkpoint.get("vcg_method") != self.config.vcg_method:
            return False, (
                f"VCG-Methode passt nicht "
                f"({checkpoint.get('vcg_method', '?')} != {self.config.vcg_method})"
            )

        checkpoint_bm = set(checkpoint.get("biomarker_names", []))
        checkpoint_path = set(checkpoint.get("pathology_names", []))
        current_bm = set(biomarker_names)
        current_path = set(pathology_names)
        if checkpoint_bm and current_bm and not checkpoint_bm.intersection(current_bm):
            return False, "keine gemeinsamen Biomarker"
        if checkpoint_path and current_path and not checkpoint_path.intersection(current_path):
            return False, "keine gemeinsamen Pathologien"
        return True, ""

    def _analysis_checkpoint_matches(
        self,
        checkpoint: dict,
        biomarker_names: list[str],
        pathology_names: list[str],
        correction_method: str,
        bootstrap_iterations: int,
        max_biomarker_combination: int,
    ) -> bool:
        compatible, _ = self._analysis_checkpoint_compatible(
            checkpoint,
            biomarker_names,
            pathology_names,
        )
        return compatible

    def _merge_analysis_checkpoint_results(
        self,
        checkpoint: dict,
        result_by_key: dict[tuple[str, str], SweepResult],
        biomarker_names: list[str],
        pathology_names: list[str],
        max_biomarker_combination: int,
    ) -> int:
        biomarker_set = set(biomarker_names)
        pathology_set = set(pathology_names)
        max_combo = max(1, int(max_biomarker_combination))
        restored = 0

        for item in checkpoint.get("results", []):
            try:
                result = _deserialize_sweep_result(item)
            except Exception:
                continue

            if result.pathology_name not in pathology_set:
                continue
            parts = [part.strip() for part in result.biomarker_name.split("+")]
            if len(parts) > max_combo:
                continue
            if not all(part in biomarker_set for part in parts):
                continue

            key = (result.biomarker_name, result.pathology_name)
            if key not in result_by_key:
                result_by_key[key] = result
                restored += 1

        return restored

    def _analysis_checkpoint_payload(
        self,
        run_id: str,
        biomarker_names: list[str],
        pathology_names: list[str],
        correction_method: str,
        bootstrap_iterations: int,
        max_biomarker_combination: int,
        results: list[SweepResult],
        combination_strategy: str = "none",
        greedy_seeds: int = 3,
        greedy_max_depth: int = 2,
    ) -> dict:
        return {
            "run_id": run_id,
            "biomarker_names": biomarker_names,
            "pathology_names": pathology_names,
            "vcg_method": self.config.vcg_method,
            "correction_method": correction_method,
            "bootstrap_iterations": int(bootstrap_iterations),
            "max_biomarker_combination": int(max_biomarker_combination),
            "combination_strategy": combination_strategy,
            "greedy_seeds": int(greedy_seeds),
            "greedy_max_depth": int(greedy_max_depth),
            "results": [r.to_json_dict(include_curves=True) for r in results],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_progress(
        self, completed: int, total: int, n_success: int, n_error: int, start_time: float
    ):
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        self._progress.percent = int(100 * (n_success) / max(self._progress.total_records, 1))
        self._progress.current_record = completed
        self._progress.elapsed_seconds = elapsed
        self._progress.message = (
            f"Verarbeitet: {n_success:,} OK / {n_error} Fehler "
            f"| Rate: {rate:.1f}/s | ETA: {_format_eta(eta)}"
        )
        self._emit_progress()

    def _make_error_record(self, eid: int, error: str) -> ProcessedRecord:
        return ProcessedRecord(
            ecg_record=ECGRecord(
                ecg_id=eid, patient_id=0, age=None, sex=None,
                height=None, weight=None, recording_date=None,
                scp_codes={}, report="", heart_axis=None,
                diagnostic_superclass=[], diagnostic_subclass=[],
                validated_by_human=False, strat_fold=0,
            ),
            preprocessed={},
            error=error,
        )

    def _finish(self, n_success: int, n_error: int, total: int):
        self._state = PipelineState.COMPLETE
        self._progress.state = PipelineState.COMPLETE
        self._progress.percent = 100
        self._progress.message = (
            f"Abgeschlossen: {n_success:,}/{total:,} OK, {n_error} Fehler"
        )
        self._emit_progress()
