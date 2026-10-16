from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.group_vcg import (
    GroupVCGOptions,
    _atomic_write_group_checkpoint,
    _load_group_checkpoint,
    aggregate_group_vcg,
)
from vcg_biomarker.mimic.jobs import resource_profile_for_config
from vcg_biomarker.mimic.models import ReportSpec, ReportType


def _config(tmp_path: Path) -> MimicProjectConfig:
    database = tmp_path / "database"
    mimic_iv = database / "mimic-iv-3.1"
    mimic_ed = database / "mimic-iv-ed-2.2"
    mimic_ecg = database / "mimic-iv-ecg"
    for path in (mimic_iv, mimic_ed, mimic_ecg):
        path.mkdir(parents=True)
    echo_measurements = database / "structured-measurement.csv"
    echo_studies = database / "echo-study-list.csv"
    echo_records = database / "echo-record-list.csv"
    for path in (echo_measurements, echo_studies, echo_records):
        path.write_text("", encoding="ascii")
    config = MimicProjectConfig(
        mimic_iv_root=mimic_iv,
        mimic_ed_root=mimic_ed,
        mimic_ecg_root=mimic_ecg,
        echo_structured_path=echo_measurements,
        echo_study_list_path=echo_studies,
        echo_record_list_path=echo_records,
        work_root=tmp_path / "work",
        workers=7,
    )
    config.ensure_directories()
    return config


def test_group_vcg_options_never_allow_locked_test() -> None:
    with pytest.raises(PermissionError, match="development-only"):
        GroupVCGOptions(
            endpoint_id="reduced_lvef",
            split="locked_test",
        )


def test_group_vcg_preserves_opposite_group_signal_and_bootstrap_ci() -> None:
    rng = np.random.default_rng(17)
    samples = 100
    phase = np.linspace(0.0, 2.0 * np.pi, samples)
    base = np.column_stack(
        (
            np.sin(phase),
            0.5 * np.cos(phase),
            0.25 * np.sin(2.0 * phase),
        )
    )
    negative = [
        base + rng.normal(0.0, 0.01, size=base.shape)
        for _ in range(24)
    ]
    positive = [
        base
        + np.array([0.30, -0.10, 0.20])
        + rng.normal(0.0, 0.01, size=base.shape)
        for _ in range(24)
    ]

    summary, difference = aggregate_group_vcg(
        {"positive": positive, "negative": negative},
        sampling_rate_hz=500,
        pre_r_seconds=0.10,
        bootstrap_iterations=200,
        random_seed=41,
    )

    assert len(summary) == samples * 2
    assert len(difference) == samples
    assert difference["delta_x_mV"].median() == pytest.approx(
        0.30,
        abs=0.01,
    )
    assert difference["delta_y_mV"].median() == pytest.approx(
        -0.10,
        abs=0.01,
    )
    assert difference["delta_z_mV"].median() == pytest.approx(
        0.20,
        abs=0.01,
    )
    assert (difference["delta_x_ci_low_mV"] > 0.0).all()
    assert (difference["delta_y_ci_high_mV"] < 0.0).all()
    assert difference.loc[50, "time_ms"] == pytest.approx(0.0)
    assert set(summary["n_subjects"]) == {24}


def test_group_vcg_job_is_bounded_interactive_and_endpoint_scoped(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    profile = resource_profile_for_config(
        "group-vcg",
        config,
        {
            "run_id": "mimic-development",
            "endpoint_id": "reduced_lvef",
            "split": "validation",
        },
    )
    assert profile.interactive
    assert profile.cpu_tokens <= 2
    assert profile.writer_namespace == (
        "group-vcg:mimic-development:reduced_lvef:validation"
    )


def test_group_vcg_checkpoint_resumes_only_same_execution(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".group-vcg.resume.npz"
    loop = np.arange(1200, dtype=float).reshape(400, 3) / 1000.0
    audit = [
        {
            "subject_id": 1,
            "study_id": 11,
            "label_state": "positive",
            "status": "included",
        }
    ]
    _atomic_write_group_checkpoint(
        path,
        input_hash="input-a",
        execution_id="job-a",
        loops_by_state={"positive": [loop], "negative": []},
        audit_rows=audit,
        completed_study_ids={11},
        expected_samples=400,
    )

    loops, loaded_audit, completed = _load_group_checkpoint(
        path,
        input_hash="input-a",
        execution_id="job-a",
        force=True,
        expected_samples=400,
    )
    assert completed == {11}
    assert loaded_audit == audit
    np.testing.assert_allclose(loops["positive"][0], loop, atol=1e-6)

    loops, loaded_audit, completed = _load_group_checkpoint(
        path,
        input_hash="input-a",
        execution_id="job-b",
        force=False,
        expected_samples=400,
    )
    assert not path.exists()
    assert loops == {"positive": [], "negative": []}
    assert loaded_audit == []
    assert completed == set()


def test_group_vcg_service_uses_shared_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import services

    config = _config(tmp_path)
    captured = {}

    def fake_run(project, options, *, progress=None):
        captured["project"] = project
        captured["options"] = options
        captured["progress"] = progress
        return "group-vcg-result"

    monkeypatch.setattr(services, "run_group_vcg_analysis", fake_run)
    result = services.run_service_stage(
        config,
        "group-vcg",
        {
            "run_id": "development-a",
            "endpoint_id": "echo_lvh_strict",
            "split": "train",
            "max_per_class": 30,
            "bootstrap_iterations": 300,
            "_execution_id": "job-shared",
        },
    )

    assert result == "group-vcg-result"
    assert captured["project"] is config
    assert captured["options"].endpoint_id == "echo_lvh_strict"
    assert captured["options"].split == "train"
    assert captured["options"].max_per_class == 30
    assert captured["options"].bootstrap_iterations == 300
    assert captured["options"].execution_id == "job-shared"


def test_group_vcg_has_printable_orthogonal_report_figure() -> None:
    from vcg_biomarker.mimic.report_builder import _group_vcg_figures

    phase = np.linspace(0.0, 2.0 * np.pi, 80)
    base = np.column_stack(
        (np.sin(phase), np.cos(phase), 0.3 * np.sin(2 * phase))
    )
    summary, difference = aggregate_group_vcg(
        {
            "negative": [base for _ in range(12)],
            "positive": [
                base + np.array([0.1, 0.0, 0.05])
                for _ in range(12)
            ],
        },
        sampling_rate_hz=500,
        pre_r_seconds=0.08,
        bootstrap_iterations=100,
        random_seed=5,
    )
    for frame in (summary, difference):
        frame["endpoint_id"] = "reduced_lvef"
        frame["split"] = "validation"
    prefix = "group_vcg_reduced_lvef_validation"
    spec = ReportSpec(
        report_id="group-vcg-report",
        report_type=ReportType.STUDY,
        run_id="mimic-development",
        formats=("pdf",),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = _group_vcg_figures(
        {
            f"{prefix}_group_vcg_summary": summary,
            f"{prefix}_group_vcg_difference": difference,
        },
        spec,
        plt,
    )
    try:
        assert [identifier for identifier, _ in figures] == [
            "group_vcg_reduced_lvef_validation"
        ]
        assert len(figures[0][1].axes) == 4
    finally:
        for _, figure in figures:
            plt.close(figure)
