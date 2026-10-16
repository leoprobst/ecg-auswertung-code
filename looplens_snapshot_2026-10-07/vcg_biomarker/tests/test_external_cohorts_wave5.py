"""The outpatient cohorts, and the surrogate that has to prove itself.

CODE-15 and SaMi-Trop are now locally integrated.  These tests pin the format
contract without making the multi-gigabyte releases a unit-test dependency:
the properties of the HDF5 format that silently produce wrong numbers if they
are assumed instead of checked, and the rule that a ten-second stand-in for a
five-minute marker is not usable until calibrated in the same patients.

The four traps, in the order they would bite:

    400 Hz and not 500 -- a rate difference enters every marker that counts
    samples or measures a slope.

    Zero padding to a fixed 4.096 samples -- a delineator handed the padding
    sees a flat lead, or a step at the padding edge.

    A fixed lead order that is not this project's -- twelve markers computed
    on shuffled leads produce numbers rather than errors.

    An exam identifier that is not a patient identifier -- the same
    pseudo-replication the MIMIC reference cohort had to be protected from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.code15_cohort import (
    CODE15_LEAD_ORDER,
    CODE15_SAMPLES_PER_RECORD,
    CODE15_SAMPLING_RATE_HZ,
    LOADER_ACCEPTANCE_CHECKS,
    ArrayTracingSource,
    Hdf5TracingSource,
    audit_cohort_release,
    assert_lead_order,
    cohort_status,
    read_exam_metadata,
    read_records,
    resample_to,
    trim_zero_padding,
)
from vcg_biomarker.mimic.signals import assess_signal_quality
from vcg_biomarker.mimic.long_recording_calibration import (
    MINIMUM_GOLD_STANDARD_SECONDS,
    CalibrationPolicy,
    calibrate_surrogate,
    calibration_declaration,
    pair_recordings,
    waveform_database_status,
)
from vcg_biomarker.mimic.population_cohorts import (
    _diagnostic_associations,
    _diagnostic_label_scope,
    _extract_cohort,
    _loader_acceptance,
    _reference_ranges,
    _source_inventory,
    _stable_fraction,
    _survival_associations,
    _transportable_validation_markers,
    _transform_agreement,
)


def _spawn_thread_limit_state(
    _: int,
) -> tuple[dict[str, str | None], tuple[int, ...]]:
    """Report native-library limits from a fresh ProcessPool child."""

    import os

    from threadpoolctl import threadpool_info

    variable_names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    environment = {name: os.environ.get(name) for name in variable_names}
    native_threads = tuple(
        int(pool["num_threads"])
        for pool in threadpool_info()
        if pool.get("num_threads") is not None
    )
    return environment, native_threads


def _padded_tracing(
    signal_samples: int = 4000,
    total: int = CODE15_SAMPLES_PER_RECORD,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tracing = np.zeros((total, len(CODE15_LEAD_ORDER)), dtype=float)
    lead = rng.normal(0.0, 0.3, (signal_samples, len(CODE15_LEAD_ORDER)))
    # Never exactly zero, which is what distinguishes signal from padding.
    lead[np.abs(lead) < 1e-6] = 0.5
    start = (total - signal_samples) // 2
    tracing[start : start + signal_samples] = lead
    return tracing


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def test_only_mimic_replicated_scalar_markers_are_transported_externally(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runs_root / "R001"
    run_root.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "feature_id": [
                "qrs_t_integral_angle_direct_8lead_deg",
                "qrs_t_area_angle_direct_8lead_deg",
                "unregistered_composite_score",
            ],
            "stage": [
                "validation_replicated",
                "development_candidate",
                "validation_replicated",
            ],
            "locked_test_eligible": [True, False, True],
        }
    )
    candidates.to_parquet(run_root / "criterion_candidates.parquet", index=False)

    selected, provenance = _transportable_validation_markers(config, "R001")
    assert selected == ("qrs_t_integral_angle_direct_8lead_deg",)
    assert provenance["status"] == "selected"
    assert provenance["eligible_criterion_rows"] == 2
    assert provenance["transported_scalar_marker_count"] == 1


def test_code15_loader_acceptance_checks_are_executed() -> None:
    n = 120
    rate = np.linspace(45.0, 110.0, n)
    standardized_rate = (rate - rate.mean()) / rate.std()
    rng = np.random.default_rng(0)
    qt = 400.0 + 30.0 * (
        -0.45 * standardized_rate
        + np.sqrt(1.0 - 0.45**2) * rng.normal(size=n)
    )
    frame = pd.DataFrame(
        {
            "cohort": ["code15"] * n,
            "extraction_status": ["ok"] * n,
            "patient_id": np.arange(n),
            "exam_id": np.arange(n),
            "spatial_qrs_t_angle_deg": np.linspace(30.0, 60.0, n),
            "qt_interval_ms": qt,
            "heart_rate_bpm": rate,
        }
    )
    accepted = _loader_acceptance(frame)
    assert set(accepted["status"]) == {"passed"}
    assert accepted["passed"].all()

    frame["spatial_qrs_t_angle_deg"] = 100.0
    rejected = _loader_acceptance(frame)
    angle = rejected.loc[
        rejected["check_id"].eq("spatial_qrs_t_angle_lower_than_hospital")
    ].iloc[0]
    assert angle["status"] == "failed"
    assert not bool(angle["passed"])


def test_external_survival_reports_increment_over_classical_ecg() -> None:
    rng = np.random.default_rng(44)
    n = 320
    marker = rng.normal(size=n)
    age = rng.normal(58.0, 10.0, size=n)
    sex = np.where(np.arange(n) % 2, "M", "F")
    normal = rng.random(n) > 0.35
    linear = 0.45 * marker + 0.025 * (age - 58.0) - 0.35 * normal
    time_days = rng.exponential(scale=500.0 / np.exp(linear)) + 1.0
    frame = pd.DataFrame(
        {
            "cohort": "sami_trop",
            "patient_id": np.arange(n),
            "extraction_status": "ok",
            "death": True,
            "follow_up_years": time_days / 365.25,
            "age": age,
            "sex": sex,
            "reported_normal_ecg": normal,
            "synthetic_marker": marker,
        }
    )
    result = _survival_associations(
        frame,
        ("synthetic_marker",),
        incremental_marker_ids=("synthetic_marker",),
    )
    row = result.loc[result["marker_id"].eq("synthetic_marker")].iloc[0]
    assert row["status"] == "estimable"
    assert row["incremental_comparison_status"] == "estimable"
    assert np.isfinite(row["incremental_concordance"])
    assert np.isfinite(row["incremental_p_value"])
    assert np.isfinite(row["hazard_ratio_per_sd_plus_classical"])


def test_the_padding_is_trimmed_and_counted():
    tracing = _padded_tracing(signal_samples=4000)
    trimmed, kept, padded = trim_zero_padding(tracing)
    assert kept == 4000
    assert padded == CODE15_SAMPLES_PER_RECORD - 4000
    assert trimmed.shape == (4000, 12)
    # No zero row survives at either end.
    assert np.any(np.abs(trimmed[0]) > 0)
    assert np.any(np.abs(trimmed[-1]) > 0)


def test_a_recording_that_is_only_padding_returns_nothing_rather_than_raising():
    """A loader that raises on one file cannot read the other 345.778."""

    trimmed, kept, padded = trim_zero_padding(
        np.zeros((CODE15_SAMPLES_PER_RECORD, 12))
    )
    assert kept == 0
    assert padded == CODE15_SAMPLES_PER_RECORD
    assert trimmed.shape[0] == 0


def test_an_interior_zero_sample_is_not_treated_as_padding():
    tracing = _padded_tracing(signal_samples=4000)
    interior = 2048
    tracing[interior, :] = 0.0
    _, kept, _ = trim_zero_padding(tracing)
    assert kept == 4000


def test_resampling_preserves_duration_and_states_the_new_rate():
    source = np.column_stack(
        [np.sin(np.linspace(0.0, 8 * np.pi, 4000)) for _ in range(12)]
    )
    resampled = resample_to(source, CODE15_SAMPLING_RATE_HZ, 500.0)
    assert resampled.shape[0] == 5000
    assert resampled.shape[1] == 12
    source_seconds = source.shape[0] / CODE15_SAMPLING_RATE_HZ
    target_seconds = resampled.shape[0] / 500.0
    assert target_seconds == pytest.approx(source_seconds, rel=1e-6)
    # The waveform is preserved, not merely the length.
    assert resampled.max() == pytest.approx(source.max(), abs=0.01)


def test_resampling_to_the_same_rate_changes_nothing():
    values = np.random.default_rng(0).normal(size=(400, 12))
    assert resample_to(values, 500.0, 500.0) is values


def test_polyphase_resampling_preserves_upper_ecg_band_amplitude():
    time = np.arange(4000, dtype=float) / 400.0
    waveform = np.sin(2.0 * np.pi * 100.0 * time)
    source = np.repeat(waveform[:, None], 12, axis=1)
    resampled = resample_to(source, 400.0, 500.0)
    source_rms = float(np.sqrt(np.mean(source[100:-100, 0] ** 2)))
    target_rms = float(np.sqrt(np.mean(resampled[125:-125, 0] ** 2)))
    assert target_rms / source_rms > 0.95


def test_the_lead_order_is_checked_and_not_assumed():
    assert_lead_order(CODE15_LEAD_ORDER)
    wrong = list(CODE15_LEAD_ORDER)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError, match="lead order"):
        assert_lead_order(wrong)


def test_reading_trims_resamples_and_reports_what_it_dropped():
    ids = np.array([11, 22, 33])
    tracings = np.stack(
        [
            _padded_tracing(signal_samples=4000, seed=1),
            _padded_tracing(signal_samples=400, seed=2),  # one second: too short
            np.zeros((CODE15_SAMPLES_PER_RECORD, 12)),  # entirely padding
        ]
    )
    records, rejected = read_records(
        ArrayTracingSource(ids, tracings),
        patient_ids={11: 7, 22: 8, 33: 9},
    )
    assert len(records) == 1
    record = records[0]
    assert record.exam_id == 11
    assert record.patient_id == 7
    assert record.sampling_rate_hz == 500.0
    assert record.original_sampling_rate_hz == CODE15_SAMPLING_RATE_HZ
    assert record.lead_names == CODE15_LEAD_ORDER
    assert record.duration_seconds() == pytest.approx(10.0, abs=0.05)
    assert set(rejected["reason"]) == {
        "shorter_than_eight_seconds_after_trimming",
        "entirely_padding",
    }


def test_the_exam_identifier_is_not_the_patient_identifier():
    """CODE-15 carries both, and a percentile per recording repeats the
    pseudo-replication the MIMIC cohort had to be protected from."""

    ids = np.array([1, 2])
    tracings = np.stack([_padded_tracing(seed=4), _padded_tracing(seed=5)])
    records, _ = read_records(
        ArrayTracingSource(ids, tracings), patient_ids={1: 100, 2: 100}
    )
    assert [record.patient_id for record in records] == [100, 100]
    assert len({record.exam_id for record in records}) == 2


def test_a_shape_that_is_not_the_published_one_is_refused():
    with pytest.raises(ValueError, match="leads"):
        read_records(
            ArrayTracingSource(np.array([1]), np.zeros((1, 4096, 8)))
        )
    with pytest.raises(ValueError, match="match the exam"):
        read_records(
            ArrayTracingSource(np.array([1, 2]), np.zeros((1, 4096, 12)))
        )


def test_absent_data_is_reported_as_absent_rather_than_assumed(tmp_path):
    missing = cohort_status(tmp_path / "not-here", "code15")
    assert not missing.present
    assert "does not exist" in missing.note
    assert "normal value" in missing.purpose

    empty = tmp_path / "code15"
    empty.mkdir()
    status = cohort_status(empty, "code15")
    assert not status.present
    assert status.tracing_files == 0

    (empty / "exams.csv").write_text("exam_id\n1\n", encoding="utf-8")
    (empty / "exams_part0.hdf5").write_bytes(b"")
    filled = cohort_status(empty, "code15")
    assert filled.present
    assert filled.tracing_files == 1
    assert filled.metadata_present

    unconfigured = cohort_status(None, "sami_trop")
    assert not unconfigured.present
    assert "Chagas" in unconfigured.purpose


def test_the_loader_carries_the_check_that_decides_whether_it_works():
    """An outpatient QRS-T angle must come out below the hospital one."""

    checks = {check["check_id"]: check for check in LOADER_ACCEPTANCE_CHECKS}
    angle = checks["spatial_qrs_t_angle_lower_than_hospital"]
    assert angle["comparison"] == "less_than"
    assert angle["reference_value"] == 74.4
    assert "44-65" in angle["literature_expectation"]
    rate = checks["rate_qt_relationship_reproduced"]
    assert rate["reference_value"] == pytest.approx(-0.45)


def test_sami_positional_hdf5_join_is_length_checked(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "exams.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tracings", data=np.zeros((3, 32, 12)))
    source = Hdf5TracingSource(path, fallback_exam_ids=[91, 92, 93])
    batches = list(source.iter_batches(batch_size=2))
    assert np.concatenate([batch[0] for batch in batches]).tolist() == [91, 92, 93]
    assert source.exam_ids().tolist() == [91, 92, 93]
    with pytest.raises(ValueError, match="must match"):
        list(Hdf5TracingSource(path, fallback_exam_ids=[1, 2]).iter_batches())


def test_hdf5_selected_batches_read_only_requested_tracings(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "selected.hdf5"
    values = np.arange(20 * 8 * 12, dtype=np.float32).reshape(20, 8, 12)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exam_id", data=np.arange(100, 120))
        handle.create_dataset("tracings", data=values, chunks=(1, 8, 12))

    source = Hdf5TracingSource(path)
    batches = list(source.iter_selected_batches({119, 103, 111}, batch_size=2))

    assert [len(ids) for ids, _ in batches] == [2, 1]
    selected_ids = np.concatenate([ids for ids, _ in batches])
    selected_values = np.concatenate([signals for _, signals in batches])
    assert selected_ids.tolist() == [103, 111, 119]
    np.testing.assert_array_equal(selected_values, values[[3, 11, 19]])


def test_hdf5_selected_batches_support_positional_sami_ids(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sami-selected.hdf5"
    values = np.arange(5 * 4 * 12, dtype=np.float32).reshape(5, 4, 12)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tracings", data=values, chunks=(1, 4, 12))

    source = Hdf5TracingSource(path, fallback_exam_ids=[91, 92, 93, 94, 95])
    batches = list(source.iter_selected_batches({92, 95}, batch_size=8))

    assert len(batches) == 1
    assert batches[0][0].tolist() == [92, 95]
    np.testing.assert_array_equal(batches[0][1], values[[1, 4]])


def test_hdf5_selected_batches_never_issue_a_full_tracing_read(monkeypatch):
    class _Dataset:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.requests = []

        def __len__(self):
            return len(self.values)

        def __getitem__(self, key):
            self.requests.append(key)
            return self.values[key]

    class _Handle:
        def __init__(self, exam_ids, tracings):
            self.datasets = {
                "exam_id": _Dataset(exam_ids),
                "tracings": _Dataset(tracings),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __contains__(self, key):
            return key in self.datasets

        def __getitem__(self, key):
            return self.datasets[key]

    handle = _Handle(
        np.arange(100, 110),
        np.arange(10 * 4 * 12, dtype=np.float32).reshape(10, 4, 12),
    )
    source = Hdf5TracingSource("unused.hdf5")
    monkeypatch.setattr(source, "_open", lambda: handle)

    batches = list(source.iter_selected_batches({101, 107, 109}, batch_size=2))

    assert [ids.tolist() for ids, _ in batches] == [[101, 107], [109]]
    tracing_requests = handle.datasets["tracings"].requests
    assert tracing_requests == [[1, 7], [9]]
    assert not any(isinstance(request, slice) for request in tracing_requests)


def test_external_process_worker_reuses_immutable_panel_settings(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from vcg_biomarker.mimic import population_cohorts

    seen = []

    def fake_analyze(
        record,
        metadata,
        marker_ids,
        requested_biomarkers,
        *,
        powerline_hz,
        minimum_signal_quality,
    ):
        seen.append(
            (
                record.exam_id,
                metadata,
                marker_ids,
                requested_biomarkers,
                powerline_hz,
                minimum_signal_quality,
            )
        )
        return {"exam_id": record.exam_id}

    monkeypatch.setattr(population_cohorts, "_analyze_record", fake_analyze)
    population_cohorts._initialize_external_worker(
        ("marker_a", "marker_b"),
        ("marker_a",),
        50.0,
        0.61,
    )
    result = population_cohorts._analyze_external_worker_item(
        (SimpleNamespace(exam_id=17), {"age": 64})
    )

    assert result == {"exam_id": 17}
    assert seen == [
        (
            17,
            {"age": 64},
            ("marker_a", "marker_b"),
            ("marker_a",),
            50.0,
            0.61,
        )
    ]


def test_external_process_worker_is_windows_spawn_picklable() -> None:
    """The production entry point must survive Windows' fresh-interpreter spawn."""

    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from vcg_biomarker.mimic.code15_cohort import Code15Record
    from vcg_biomarker.mimic.population_cohorts import (
        _analyze_external_worker_item,
        _initialize_external_worker,
    )

    record = Code15Record(
        exam_id=701,
        patient_id=91,
        signal_millivolts=np.zeros((4_000, 12), dtype=np.float32),
        sampling_rate_hz=400.0,
        original_sampling_rate_hz=400.0,
        lead_names=CODE15_LEAD_ORDER,
        trimmed_samples=0,
        padded_samples=96,
        source="code15",
    )
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
        initializer=_initialize_external_worker,
        initargs=((), (), 60.0, 0.0),
    ) as executor:
        rows = list(
            executor.map(
                _analyze_external_worker_item,
                [(record, {"age": 60, "sex": "male"})] * 2,
                chunksize=1,
            )
        )
        thread_states = list(executor.map(_spawn_thread_limit_state, range(2)))

    assert [row["exam_id"] for row in rows] == [701, 701]
    assert all(row["extraction_status"] == "abstained_signal_qc" for row in rows)
    for environment, native_threads in thread_states:
        assert set(environment.values()) == {"1"}
        assert all(count == 1 for count in native_threads)


def test_missing_external_cohort_sex_stays_missing_instead_of_becoming_female(
    tmp_path,
):
    path = tmp_path / "exams.csv"
    pd.DataFrame(
        {
            "exam_id": [1, 2, 3],
            "patient_id": [10, 20, 30],
            "age": [50, 60, 70],
            "is_male": [1, 0, np.nan],
        }
    ).to_csv(path, index=False)

    metadata = read_exam_metadata(path).set_index("exam_id")

    assert metadata.loc[1, "sex"] == "M"
    assert metadata.loc[2, "sex"] == "F"
    assert pd.isna(metadata.loc[3, "sex"])


def test_release_audit_requires_every_metadata_referenced_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    code = tmp_path / "code15"
    code.mkdir()
    pd.DataFrame(
        {
            "exam_id": [11, 12],
            "patient_id": [1, 2],
            "age": [40, 50],
            "is_male": [True, False],
            "death": [False, False],
            "timey": [2.0, 3.0],
            "trace_file": ["part0.hdf5", "part1.hdf5"],
        }
    ).to_csv(code / "exams.csv", index=False)
    with h5py.File(code / "part0.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([0, 11]))
        handle.create_dataset("tracings", data=np.zeros((2, 4096, 12)))

    incomplete = audit_cohort_release(code, "code15")
    assert incomplete["status"] == "error"
    assert "part1.hdf5" in incomplete["diagnostic"]

    with h5py.File(code / "part1.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([0, 12]))
        handle.create_dataset("tracings", data=np.zeros((2, 4096, 12)))
    complete = audit_cohort_release(code, "code15")
    assert complete["status"] == "ok"
    assert complete["metadata_rows"] == 2
    assert complete["tracing_rows"] == 2


def test_sami_release_audit_pins_positional_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "sami"
    root.mkdir()
    pd.DataFrame(
        {
            "exam_id": [1, 2],
            "age": [60, 61],
            "is_male": [True, False],
            "death": [False, True],
            "timey": [2.0, 1.0],
        }
    ).to_csv(root / "exams.csv", index=False)
    with h5py.File(root / "exams.hdf5", "w") as handle:
        handle.create_dataset("tracings", data=np.zeros((1, 4096, 12)))
    assert audit_cohort_release(root, "sami_trop")["status"] == "error"
    with h5py.File(root / "exams.hdf5", "w") as handle:
        handle.create_dataset("tracings", data=np.zeros((2, 4096, 12)))
    assert audit_cohort_release(root, "sami_trop")["status"] == "ok"


def test_large_source_hashes_are_reused_only_for_exact_file_metadata(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "code15"
    root.mkdir()
    (root / "exams.csv").write_text("exam_id\n1\n", encoding="ascii")
    (root / "part.hdf5").write_bytes(b"stable")
    calls = []

    def digest(path):
        calls.append(str(path))
        return "a" * 64

    monkeypatch.setattr(
        "vcg_biomarker.mimic.population_cohorts.file_sha256",
        digest,
    )
    cache = {}
    first = _source_inventory(
        root,
        "code15",
        verify_sha256=True,
        hash_cache=cache,
    )
    second = _source_inventory(
        root,
        "code15",
        verify_sha256=True,
        hash_cache=cache,
    )
    assert len(calls) == 2
    assert not first["sha256_cache_hit"].any()
    assert second["sha256_cache_hit"].all()

    (root / "part.hdf5").write_bytes(b"changed size")
    _source_inventory(root, "code15", verify_sha256=True, hash_cache=cache)
    assert len(calls) == 3


def test_external_extraction_resumes_only_strongly_verified_partitions(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    h5py = pytest.importorskip("h5py")
    root = tmp_path / "code15"
    root.mkdir()
    pd.DataFrame(
        {
            "exam_id": [11, 12, 13],
            "patient_id": [1, 2, 3],
            "age": [40, 50, 60],
            "is_male": [True, False, True],
            "death": [False, False, True],
            "timey": [2.0, 3.0, 1.0],
            "trace_file": ["part.hdf5"] * 3,
        }
    ).to_csv(root / "exams.csv", index=False)
    with h5py.File(root / "part.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([11, 12, 13]))
        handle.create_dataset("tracings", data=np.zeros((3, 1, 1)))

    def fake_records(source, **_kwargs):
        records = [
            SimpleNamespace(
                exam_id=int(exam_id),
                patient_id=int(exam_id),
                source="code15",
                original_sampling_rate_hz=400.0,
                sampling_rate_hz=500.0,
                trimmed_samples=4000,
                padded_samples=96,
                signal_millivolts=np.zeros((10, 12)),
            )
            for exam_id in source.ids
        ]
        return records, pd.DataFrame()

    calls = []

    def fake_analyze(record, metadata, *_args, **_kwargs):
        calls.append(record.exam_id)
        return {
            **metadata,
            "exam_id": record.exam_id,
            "cohort": "code15",
            "extraction_status": "ok",
            "marker": float(record.exam_id),
        }

    monkeypatch.setattr(
        "vcg_biomarker.mimic.population_cohorts.read_records", fake_records
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.population_cohorts._analyze_record", fake_analyze
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.population_cohorts.EXTERNAL_FEATURE_PARTITION_SIZE",
        2,
    )
    output = tmp_path / "output"
    first_paths, _, _ = _extract_cohort(
        root,
        "code15",
        ("marker",),
        output,
        extraction_input_hash="a" * 64,
        max_records=None,
        workers=1,
        minimum_signal_quality=0.0,
        force=False,
        progress=None,
    )
    assert calls == [11, 12, 13]
    assert len(first_paths) == 2

    calls.clear()
    second_paths, _, _ = _extract_cohort(
        root,
        "code15",
        ("marker",),
        output,
        extraction_input_hash="a" * 64,
        max_records=None,
        workers=1,
        minimum_signal_quality=0.0,
        force=False,
        progress=None,
    )
    assert calls == []
    assert second_paths == first_paths


def test_reference_eligibility_uses_failed_abnormal_ecgs_too():
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "exam_id": [10, 11, 20],
            "extraction_status": ["ok", "failed", "ok"],
            "reported_normal_ecg": [True, False, True],
            "age": [50, 50, 55],
            "sex": ["F", "F", "M"],
            "marker": [1.0, np.nan, 2.0],
        }
    )
    result = _reference_ranges(frame, ("marker",))
    overall = result.loc[result["stratum"].eq("all")].iloc[0]
    assert overall["subject_count"] == 1
    assert overall["measurable_count"] == 1


def test_validation_never_learns_direction_when_development_is_too_small():
    development_ids = [i for i in range(1, 10_000) if _stable_fraction(i) < 0.70]
    validation_ids = [i for i in range(1, 10_000) if _stable_fraction(i) >= 0.70]
    ids = development_ids[:20] + validation_ids[:80]
    labels = [False] * 10 + [True] * 10 + [False] * 40 + [True] * 40
    frame = pd.DataFrame(
        {
            "patient_id": ids,
            "exam_id": ids,
            "extraction_status": "ok",
            "atrial_fibrillation": labels,
            "marker": np.arange(len(ids), dtype=float),
        }
    )
    result = _diagnostic_associations(frame, ("marker",))
    assert result.empty


def test_published_seven_second_external_record_has_explicit_qc_policy():
    signal = np.random.default_rng(7).normal(0.0, 0.2, (7 * 500, 12))
    strict = assess_signal_quality(signal, 500)
    external = assess_signal_quality(
        signal,
        500,
        minimum_duration_seconds=6.5,
    )
    assert "duration_below_8_seconds" in strict.reasons
    assert "duration_below_8_seconds" not in external.reasons


def test_direct_angle_agreement_is_method_labelled_and_abstains_when_small():
    frame = pd.DataFrame(
        {
            "cohort": ["code15"] * 4,
            "extraction_status": ["ok"] * 4,
            "qrs_t_area_angle_direct_8lead_deg": [10.0, 20.0, 30.0, 40.0],
            "qrs_t_integral_angle_direct_8lead_deg": [12.0, 22.0, 32.0, 42.0],
            "transform_kors__spatial_qrs_t_angle_deg": [8.0, 18.0, 28.0, 38.0],
            "transform_inverse_dower__spatial_qrs_t_angle_deg": [9.0, 19.0, 29.0, 39.0],
            "transform_pca_kors_aligned__spatial_qrs_t_angle_deg": [11.0, 21.0, 31.0, 41.0],
        }
    )
    result = _transform_agreement(frame)
    assert len(result) == 6
    assert set(result["method"]) == {
        "kors",
        "inverse_dower",
        "pca_kors_aligned",
    }
    kors = result.loc[
        result["direct_marker_id"].eq("qrs_t_area_angle_direct_8lead_deg")
        & result["method"].eq("kors")
    ].iloc[0]
    assert kors["bias_deg"] == pytest.approx(2.0)
    assert kors["pearson_r"] == pytest.approx(1.0)
    assert not bool(kors["estimable"])
    assert kors["status"] == "too_few_pairs"
    assert kors["paired_availability_fraction"] == 1
    assert np.isnan(kors["loa_low_deg"])
    assert kors["uncertainty_basis"] == "withheld_below_30_pairs"
    assert kors["claim_scope"] == "measurement_agreement_not_interchangeability"


def test_code15_normal_label_is_not_promoted_to_a_clinical_gold_standard():
    assert _diagnostic_label_scope("reported_normal_ecg") == (
        "automatic_same_ecg_annotation_not_independent_reference"
    )


def test_direct_angle_agreement_retains_index_denominator_and_labels_pca_scope():
    n = 40
    base = np.arange(n, dtype=float)
    frame = pd.DataFrame({
        "cohort": ["code15"] * n, "extraction_status": ["ok"] * 35 + ["failed"] * 5,
        "qrs_t_area_angle_direct_8lead_deg": base,
        "qrs_t_integral_angle_direct_8lead_deg": base + 1,
        "transform_kors__spatial_qrs_t_angle_deg": base - 2,
        "transform_inverse_dower__spatial_qrs_t_angle_deg": base - 3,
        "transform_pca_kors_aligned__spatial_qrs_t_angle_deg": base - 1,
    })
    result = _transform_agreement(frame)
    assert result.index_record_count.eq(40).all()
    assert result.extraction_ok_count.eq(35).all()
    assert result.n.eq(35).all()
    assert result.paired_availability_fraction.eq(.875).all()
    assert result.estimable.all() and result.loa_low_deg.notna().all()
    pca = result.loc[result.method.eq("pca_kors_aligned")]
    assert pca.pca_interpretation.eq(
        "record_local_pca_aligned_to_kors_sensitivity_not_anatomical_reference"
    ).all()
    assert _diagnostic_label_scope("atrial_fibrillation") == (
        "same_ecg_annotation_not_independent_clinical_reference"
    )


# --------------------------------------------------------------------------
# the ten-second surrogate
# --------------------------------------------------------------------------


def _long(subject: int, record: str, when: str, seconds: float) -> dict:
    return {
        "subject_id": subject,
        "record_id": record,
        "record_time": when,
        "duration_seconds": seconds,
    }


def test_only_recordings_long_enough_for_the_gold_standard_are_paired():
    long_frame = pd.DataFrame(
        [
            _long(1, "a", "2150-01-01 10:00", 600.0),
            _long(2, "b", "2150-01-01 10:00", 120.0),  # two minutes: too short
        ]
    )
    strips = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "ecg_time": ["2150-01-01 11:00", "2150-01-01 11:00"],
        }
    )
    pairs = pair_recordings(long_frame, strips)
    assert list(pairs["subject_id"]) == [1]
    assert pairs.iloc[0]["long_duration_seconds"] >= MINIMUM_GOLD_STANDARD_SECONDS


def test_a_strip_taken_days_later_is_not_a_second_measurement_of_one_state():
    long_frame = pd.DataFrame([_long(1, "a", "2150-01-01 10:00", 600.0)])
    strips = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "ecg_time": ["2150-01-04 10:00"],
        }
    )
    assert pair_recordings(long_frame, strips).empty


def test_the_nearest_strip_wins_and_each_subject_contributes_once():
    long_frame = pd.DataFrame(
        [
            _long(1, "a", "2150-01-01 10:00", 600.0),
            _long(1, "b", "2150-01-01 20:00", 600.0),
        ]
    )
    strips = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "study_id": [10, 11],
            "ecg_time": ["2150-01-01 09:30", "2150-01-01 14:00"],
        }
    )
    pairs = pair_recordings(long_frame, strips)
    assert len(pairs) == 1
    assert pairs.iloc[0]["study_id"] == 10
    assert pairs.iloc[0]["delta_hours"] == pytest.approx(0.5)


def test_a_surrogate_that_tracks_the_gold_standard_is_calibrated():
    rng = np.random.default_rng(4)
    surrogate = rng.normal(0.0, 1.0, 200)
    gold = 2.5 * surrogate + 4.0 + rng.normal(0.0, 0.6, 200)
    result = calibrate_surrogate(gold, surrogate, marker_id="prd_10s")
    assert result["carries"]
    assert result["verdict"] == "surrogate_calibrated"
    assert result["slope"] == pytest.approx(2.5, rel=0.1)
    assert result["intercept"] == pytest.approx(4.0, abs=0.3)
    assert result["pearson_ci_low"] < result["pearson_r"] < result["pearson_ci_high"]


def test_a_surrogate_that_does_not_track_is_declared_unreachable():
    """The answer may be no, and no is a result."""

    rng = np.random.default_rng(9)
    surrogate = rng.normal(size=200)
    gold = rng.normal(size=200)
    result = calibrate_surrogate(gold, surrogate, marker_id="prd_10s")
    assert not result["carries"]
    assert result["verdict"] == "surrogate_does_not_track_the_gold_standard"
    declaration = calibration_declaration([result]).iloc[0]
    assert "may not stand in" in declaration["permitted_use"]
    assert "not reachable from ten seconds" in declaration["permitted_use"]


def test_too_few_pairs_refuses_to_draw_a_calibration_line():
    result = calibrate_surrogate(
        np.arange(10.0), np.arange(10.0), marker_id="prd_10s"
    )
    assert not result["carries"]
    assert result["verdict"] == "too_few_pairs"
    assert np.isnan(result["slope"])


def test_the_agreement_floor_is_declared_before_the_data_is_seen():
    policy = CalibrationPolicy()
    assert policy.minimum_correlation == 0.70
    assert policy.minimum_pairs == 30
    assert policy.maximum_pairing_hours == 6.0


def test_the_waveform_database_is_reported_with_what_it_is_for(tmp_path):
    absent = waveform_database_status(None)
    assert not absent["present"]
    assert "200 recordings" in absent["note"]
    assert "calibrate" in absent["note"]

    root = tmp_path / "mimic4wdb"
    root.mkdir()
    empty = waveform_database_status(root)
    assert not empty["present"]
    assert empty["record_count"] == 0

    (root / "p100").mkdir()
    (root / "p100" / "83404654.hea").write_text("x\n", encoding="utf-8")
    present = waveform_database_status(root)
    assert present["present"]
    assert present["record_count"] == 1
