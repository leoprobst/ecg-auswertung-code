from __future__ import annotations

import gzip
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.data_loader import LEAD_MAP
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)
from vcg_biomarker.core.delineation.multilead_delineator import (
    MultileadDelineator,
)
from vcg_biomarker.core.preprocessing import prepare_signal_paths, resample_signal
from vcg_biomarker.mimic.classical import (
    ClassicalECGCriteria,
    _has_contiguous_pair,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.feature_extraction import (
    _write_feature_artifact_index,
    feature_implementation_hash,
    verify_feature_artifacts,
)
from vcg_biomarker.mimic.feature_record import (
    _acquisition_upper_bandwidth_hz,
    _wideband_acquisition_support,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.005-150 Hz", 150.0),
        ("0,05 – 150 Hz", 150.0),
        ("150Hz", 150.0),
        (None, None),
        ("unknown", None),
    ],
)
def test_acquisition_bandwidth_parser_is_conservative(raw, expected) -> None:
    assert _acquisition_upper_bandwidth_hz(raw) == expected


def test_hf_qrs_requires_the_complete_80_150_hz_acquisition_band() -> None:
    assert _wideband_acquisition_support("0.05-150 Hz") == (
        True,
        150.0,
        "supported_80_150_hz",
    )
    supported, upper, reason = _wideband_acquisition_support("0.05-100 Hz")
    assert supported is False
    assert upper == 100.0
    assert reason == "acquisition_bandwidth_below_150_hz"
from vcg_biomarker.mimic.evidence import (
    _AttestationResourceConfig,
    _reuse_bound_evidence_after_exact_attestation,
)
from vcg_biomarker.mimic.label_registry import (
    LABEL_RULE_VERSION,
    OUTCOME_BLIND_INDEX_POLICY,
    default_label_definitions,
)
from vcg_biomarker.mimic.labels import (
    _composite_rule,
    _exact_query_comparison,
    _index_cohort_query,
    _label_rules,
    _write_development_label_attestation,
    development_label_attestation_path,
    label_implementation_hash,
    verify_development_label_attestation,
    verify_clinical_label_artifacts,
)
from vcg_biomarker.mimic.models import (
    EligibilityState,
    FeaturePanelSpec,
    LabelState,
    PanelTier,
    TestLock as MimicTestLock,
)
from vcg_biomarker.mimic.panels import (
    CORE_REQUIRED_VCG,
    default_feature_panels,
)
from vcg_biomarker.mimic.report_parser import (
    MachineReportParser,
    ReportMentionState,
)
from vcg_biomarker.mimic.run_integrity import (
    assert_development_run_mutable,
)
from vcg_biomarker.mimic.schema import read_csv_header, validate_columns
from vcg_biomarker.mimic.signals import (
    WaveformValidationError,
    _convert_to_millivolts,
    _strict_lead_indices,
    assess_signal_quality,
)
from vcg_biomarker.mimic.splits import (
    assert_subject_split_integrity,
    assign_subject_split,
)
from vcg_biomarker.mimic.statistics import (
    apply_fdr_by_family,
    apply_frozen_score,
    enforce_inference_policy,
    evaluate_marker,
    fit_frozen_score,
    subject_cluster_bootstrap,
)
from vcg_biomarker.mimic.validation import (
    _is_clinical_combination_candidate,
    _resolved_feature_scope,
)
from vcg_biomarker.mimic import storage as mimic_storage
from vcg_biomarker.mimic.storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_fingerprint,
)


def test_bound_evidence_reuse_requires_exact_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import event_equivalence
    from vcg_biomarker.mimic.jobs import JobPauseRequested

    config = _config(tmp_path)
    constrained = _AttestationResourceConfig(config)
    # Single worker and low-memory mode keep the shadow rebuild from crowding
    # out the run. Admission control has already reserved machine headroom;
    # the wrapper must retain the evidence stage's deterministic ceiling.
    assert 2 <= constrained.duckdb_memory_limit_gb <= config.duckdb_memory_limit_gb
    assert constrained.workers == 1
    assert constrained.evidence_low_memory_mode
    output = tmp_path / "evidence.parquet"
    output.write_bytes(b"immutable-evidence")
    manifest = StageManifest.complete(
        stage="build-evidence",
        stage_version="1.9.0",
        input_hash="a" * 64,
        config_hash="b" * 64,
        output_paths=(str(output.resolve()),),
    )
    monkeypatch.setattr(
        event_equivalence,
        "attest_clinical_event_evidence_equivalence",
        lambda *_args, **_kwargs: {
            "status": "verified_exact_equivalence"
        },
    )
    reused = _reuse_bound_evidence_after_exact_attestation(
        config,
        manifest,
        progress=None,
    )
    assert reused is manifest
    assert output.read_bytes() == b"immutable-evidence"

    def fail_attestation(*_args, **_kwargs):
        raise RuntimeError("tables differ")

    monkeypatch.setattr(
        event_equivalence,
        "attest_clinical_event_evidence_equivalence",
        fail_attestation,
    )
    with pytest.raises(RuntimeError) as failure:
        _reuse_bound_evidence_after_exact_attestation(
            config,
            manifest,
            progress=None,
        )
    # The message must name the underlying cause. "Equivalence could not be
    # proven" alone covered a TypeError in a config wrapper and an
    # out-of-memory in the shadow rebuild, and neither says anything about
    # whether the evidence changed -- twice that wording sent the search to
    # the data instead of the code.
    assert "tables differ" in str(failure.value)
    assert "still unknown" in str(failure.value)
    assert output.read_bytes() == b"immutable-evidence"

    def request_pause(*_args, **_kwargs):
        raise JobPauseRequested("pause at checkpoint")

    monkeypatch.setattr(
        event_equivalence,
        "attest_clinical_event_evidence_equivalence",
        request_pause,
    )
    with pytest.raises(JobPauseRequested, match="pause at checkpoint"):
        _reuse_bound_evidence_after_exact_attestation(
            config,
            manifest,
            progress=None,
        )


def _config(tmp_path: Path) -> MimicProjectConfig:
    return MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )


def test_csv_and_gzip_headers_are_supported(tmp_path: Path) -> None:
    plain = tmp_path / "table.csv"
    compressed = tmp_path / "table.csv.gz"
    plain.write_text("subject_id,value\n1,2\n", encoding="ascii")
    with gzip.open(compressed, "wt", encoding="ascii", newline="") as handle:
        handle.write("subject_id,value\n1,2\n")

    assert read_csv_header(plain) == ("subject_id", "value")
    assert read_csv_header(compressed) == ("subject_id", "value")
    _, missing = validate_columns(compressed, ("subject_id", "charttime"))
    assert missing == ("charttime",)


def test_atomic_replace_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("complete", encoding="ascii")
    real_replace = mimic_storage.os.replace
    attempts = 0

    def transient_replace(source_path, target_path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient file lock")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(mimic_storage.os, "replace", transient_replace)
    mimic_storage.atomic_replace(
        source,
        target,
        attempts=3,
        initial_delay_seconds=0,
    )

    assert attempts == 3
    assert target.read_text(encoding="ascii") == "complete"
    assert not source.exists()


def test_label_states_and_missing_reference_are_distinct() -> None:
    assert {state.value for state in LabelState} == {
        "positive",
        "negative",
        "indeterminate",
        "unavailable",
    }
    assert EligibilityState.INSUFFICIENT_EVIDENCE != EligibilityState.ELIGIBLE
    definitions = {item.label_id: item for item in default_label_definitions()}
    troponin = definitions["troponin_positive_any"]
    assert "measured" in troponin.negative_rule
    assert "myocardial" not in troponin.display_name.casefold()
    assert definitions["right_heart_strain_proxy"].window_before_hours == -168
    assert definitions["right_heart_strain_proxy"].window_after_hours == 168
    assert definitions["mortality_30d"].window_after_hours == 720
    assert definitions["heart_failure_icd"].window_before_hours is None
    assert "partial" in definitions[
        "endocarditis_high_confidence_proxy"
    ].negative_rule
    assert {
        definition.index_policy for definition in definitions.values()
    } == {OUTCOME_BLIND_INDEX_POLICY}


def test_composite_rule_separates_controls_partial_and_unavailable() -> None:
    rule = _composite_rule(
        positive="complete_case",
        assessed="reference_assessed",
        negative="clean_control",
        indeterminate="partial_evidence",
        summary="'synthetic composite'",
    )
    frame = pd.DataFrame(
        {
            "complete_case": [True, False, False, False, False],
            "reference_assessed": [True, True, True, True, False],
            "clean_control": [False, True, False, False, False],
            "partial_evidence": [False, False, True, False, False],
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        result = connection.execute(
            "SELECT "
            f"({rule.state}) AS state, "
            f"({rule.eligibility}) AS eligibility, "
            f"({rule.confidence}) AS confidence "
            "FROM ctx"
        ).fetchdf()
    finally:
        connection.close()

    assert result["state"].tolist() == [
        "positive",
        "negative",
        "indeterminate",
        "indeterminate",
        "unavailable",
    ]
    assert result["eligibility"].tolist() == [
        "eligible",
        "eligible",
        "insufficient_evidence",
        "insufficient_evidence",
        "insufficient_evidence",
    ]
    assert result["confidence"].tolist() == [
        "high",
        "medium",
        "low",
        "low",
        "none",
    ]


def test_acute_mi_composite_does_not_treat_partial_evidence_as_control() -> None:
    rule = _label_rules()["acute_mi_high_specificity"]
    frame = pd.DataFrame(
        {
            "acute_troponin_change_proxy_value": [True, True, False, True],
            "troponin_serial_assessed": [True, True, True, True],
            "troponin_count_standard": [2, 2, 2, 2],
            "dx_acute_mi": [True, True, False, False],
            "proc_pci": [True, False, False, False],
            "pci_day_delta": [0, np.nan, np.nan, np.nan],
            "proc_coronary_angiography": [False, False, False, False],
            "dx_sepsis": [False, False, False, True],
            "dx_shock": [False] * 4,
            "dx_hypoxia_respiratory_failure": [False] * 4,
            "dx_atrial_fibrillation_flutter": [False] * 4,
            "dx_ventricular_tachycardia": [False] * 4,
            "dx_anemia": [False] * 4,
            "dx_acute_renal_failure": [False] * 4,
            "dx_chronic_kidney_disease": [False] * 4,
            "hemoglobin_nearest": [np.nan] * 4,
            "lactate_nearest": [np.nan] * 4,
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        result = connection.execute(
            f"SELECT ({rule.state}) AS state, "
            f"({rule.eligibility}) AS eligibility FROM ctx"
        ).fetchdf()
    finally:
        connection.close()

    assert result["state"].tolist() == [
        "positive",
        "indeterminate",
        "negative",
        "negative",
    ]
    assert result["eligibility"].tolist() == [
        "eligible",
        "insufficient_evidence",
        "eligible",
        "eligible",
    ]


def test_right_heart_proxy_requires_size_and_function_for_controls() -> None:
    rule = _label_rules()["right_heart_strain_proxy"]
    frame = pd.DataFrame(
        {
            "dx_pulmonary_embolism": [True] * 4,
            "dx_pulmonary_hypertension": [False] * 4,
            "has_tte_7d": [True] * 4,
            "rv_basal_diameter_cm": [4.5, 3.8, 3.8, np.nan],
            "rv_size_category": [None] * 4,
            "tapse_cm": [2.0, 2.0, np.nan, np.nan],
            "rv_function_category": [None] * 4,
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        states = connection.execute(
            f"SELECT ({rule.state}) AS state FROM ctx"
        ).fetchnumpy()["state"].tolist()
    finally:
        connection.close()

    assert states == [
        "positive",
        "negative",
        "indeterminate",
        "unavailable",
    ]


def test_label_artifact_integrity_rejects_duplicate_patient_endpoints(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    root = config.labels_root
    sealed = root / "sealed"
    sealed.mkdir(parents=True)
    config.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.connect(str(config.catalog_path)).close()

    dev_long = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [11],
            "label_id": ["heart_failure_icd"],
            "label_state": ["negative"],
            "eligibility_state": ["eligible"],
            "split": ["train"],
            "rule_version": [LABEL_RULE_VERSION],
            "evidence_ids_json": ["[]"],
            "time_delta_hours": [np.nan],
            "window_before_hours": [np.nan],
            "window_after_hours": [np.nan],
        }
    )
    locked_long = dev_long.copy()
    locked_long["subject_id"] = 2
    locked_long["study_id"] = 22
    locked_long["split"] = "locked_test"
    dev_cohort = dev_long[
        ["subject_id", "study_id", "label_id", "label_state",
         "eligibility_state", "split"]
    ].copy()
    locked_cohort = locked_long[
        ["subject_id", "study_id", "label_id", "label_state",
         "eligibility_state", "split"]
    ].copy()

    atomic_write_parquet(dev_long, root / "clinical_label_evidence.parquet")
    atomic_write_parquet(
        locked_long,
        sealed / "locked_clinical_label_evidence.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame({"subject_id": [1], "split": ["train"]}),
        root / "clinical_labels_wide.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame({"subject_id": [2], "split": ["locked_test"]}),
        sealed / "locked_clinical_labels_wide.parquet",
    )
    atomic_write_parquet(dev_cohort, root / "endpoint_index_cohorts.parquet")
    atomic_write_parquet(
        locked_cohort,
        sealed / "locked_endpoint_index_cohorts.parquet",
    )
    atomic_write_json(
        root / "label_definitions.json",
        {
            "rule_version": LABEL_RULE_VERSION,
            "definitions": [
                definition.to_dict()
                for definition in default_label_definitions()
            ],
        },
    )

    assert verify_clinical_label_artifacts(config)["status"] == "verified"

    atomic_write_parquet(
        pd.concat([dev_cohort, dev_cohort], ignore_index=True),
        root / "endpoint_index_cohorts.parquet",
    )
    with pytest.raises(RuntimeError, match="duplicate_subject_label"):
        verify_clinical_label_artifacts(config)


def test_index_cohort_is_reference_blind_and_retains_unverified(
    tmp_path: Path,
) -> None:
    rows = [
        (1, 101, "2024-01-01 08:00:00", "negative", "eligible", -1.0),
        (1, 102, "2024-01-01 09:00:00", "positive", "eligible", 2.0),
        (1, 103, "2024-01-01 10:00:00", "positive", "eligible", -0.5),
        (2, 201, "2024-01-02 08:00:00", "negative", "eligible", 4.0),
        (2, 202, "2024-01-02 09:00:00", "negative", "eligible", 1.0),
        (
            3,
            301,
            "2024-01-03 08:00:00",
            "unavailable",
            "insufficient_evidence",
            np.nan,
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=(
            "subject_id",
            "study_id",
            "ecg_time",
            "label_state",
            "eligibility_state",
            "time_delta_hours",
        ),
    )
    frame["ecg_time"] = pd.to_datetime(frame["ecg_time"])
    frame["split"] = "train"
    frame["label_id"] = "troponin_positive_any"
    frame["target_kind"] = "diagnostic"
    path = tmp_path / "labels.parquet"
    frame.to_parquet(path, index=False)

    selected = duckdb.connect().execute(
        _index_cohort_query(path)
    ).fetchdf()
    assert selected.set_index("subject_id")["study_id"].to_dict() == {
        1: 101,
        2: 201,
        3: 301,
    }
    assert set(selected["label_state"]) == {"negative", "unavailable"}
    assert selected.set_index("subject_id")["index_selection_strategy"].to_dict() == {
        1: "first_verified_ecg_reference_result_blind",
        2: "first_verified_ecg_reference_result_blind",
        3: "first_nonexcluded_ecg_never_verified",
    }


def test_x1_first_verified_ecg_is_the_diagnostic_index_and_time_zero_stays(
    tmp_path: Path,
) -> None:
    """Audit X1: an unverified first ECG no longer drops the patient."""
    rows = [
        # diagnostic: first ECG without reference, later one verified
        ("troponin_positive_any", "diagnostic", 4, 401, "2024-01-01 08:00", "indeterminate", "insufficient_evidence"),
        ("troponin_positive_any", "diagnostic", 4, 402, "2024-01-02 08:00", "positive", "eligible"),
        ("troponin_positive_any", "diagnostic", 4, 403, "2024-01-03 08:00", "negative", "eligible"),
        # prognostic: time zero is the first ECG whatever its follow-up
        ("mortality_30d", "predictive", 5, 501, "2024-01-01 08:00", "indeterminate", "insufficient_evidence"),
        ("mortality_30d", "predictive", 5, 502, "2024-01-05 08:00", "negative", "eligible"),
    ]
    frame = pd.DataFrame(rows, columns=(
        "label_id", "target_kind", "subject_id", "study_id", "ecg_time",
        "label_state", "eligibility_state"))
    frame["ecg_time"] = pd.to_datetime(frame["ecg_time"])
    frame["split"] = "train"
    path = tmp_path / "labels.parquet"
    frame.to_parquet(path, index=False)
    selected = duckdb.connect().execute(_index_cohort_query(path)).fetchdf()
    chosen = selected.set_index("label_id")
    assert chosen.loc["troponin_positive_any", "study_id"] == 402
    assert chosen.loc["mortality_30d", "study_id"] == 501
    assert chosen.loc["mortality_30d", "index_selection_strategy"] == "first_nonexcluded_ecg_time_zero"
    assert "reference_verified" not in selected.columns


def test_development_label_attestation_detects_artifact_changes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.ensure_directories()
    context_path = (
        config.evidence_root / "clinical_context_features.parquet"
    )
    pd.DataFrame({"split": ["train"]}).to_parquet(
        context_path,
        index=False,
    )
    artifact_paths = {
        "clinical_label_evidence": (
            config.labels_root / "clinical_label_evidence.parquet"
        ),
        "clinical_labels_wide": (
            config.labels_root / "clinical_labels_wide.parquet"
        ),
        "endpoint_index_cohorts": (
            config.labels_root / "endpoint_index_cohorts.parquet"
        ),
        "endpoint_window_sensitivity_cohorts": (
            config.labels_root
            / "endpoint_window_sensitivity_cohorts.parquet"
        ),
    }
    for path in artifact_paths.values():
        pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    (config.labels_root / "label_definitions.json").write_text(
        "{}",
        encoding="utf-8",
    )
    payload = _write_development_label_attestation(
        config,
        mode="generated_by_current_implementation",
        implementation_hash=label_implementation_hash(),
        comparisons={"all_exact": True, "status": "generated_current"},
    )

    verified = verify_development_label_attestation(config)
    assert verified["attestation_hash"] == payload["attestation_hash"]
    assert verified["locked_test_read_for_attestation"] is False
    assert len(verified["label_implementation_hash"]) == 64

    pd.DataFrame({"value": [2]}).to_parquet(
        artifact_paths["endpoint_index_cohorts"],
        index=False,
    )
    # Anyone about to rely on the labels must still be stopped: the
    # attestation no longer covers what they would read.
    with pytest.raises(RuntimeError, match="superseded"):
        verify_development_label_attestation(config)

    # The two stages that can fix it are told so instead, because raising
    # here would deadlock them -- the labels cannot be rebuilt while the
    # stale attestation forbids it, and cannot be re-attested before they
    # are rebuilt.
    assert (
        verify_development_label_attestation(config, require_current=False)
        is None
    )

    # A tampered attestation stays fatal in both modes: that is not a
    # superseded statement, it is an untrustworthy one.
    attestation_path = development_label_attestation_path(config)
    tampered = json.loads(attestation_path.read_text(encoding="utf-8"))
    tampered["scope"] = "everything_including_locked_test"
    attestation_path.write_text(json.dumps(tampered), encoding="utf-8")
    for require_current in (True, False):
        with pytest.raises(RuntimeError, match="hash verification failed"):
            verify_development_label_attestation(
                config, require_current=require_current
            )


def test_exact_label_comparison_is_duplicate_and_null_aware() -> None:
    connection = duckdb.connect()
    try:
        exact = _exact_query_comparison(
            connection,
            current_query="SELECT * FROM (VALUES (1), (1), (NULL)) t(value)",
            canonical_query="SELECT * FROM (VALUES (NULL), (1), (1)) t(value)",
        )
        different = _exact_query_comparison(
            connection,
            current_query="SELECT * FROM (VALUES (1), (NULL)) t(value)",
            canonical_query="SELECT * FROM (VALUES (1), (1)) t(value)",
        )
    finally:
        connection.close()

    assert exact["exact"]
    assert exact["current_minus_canonical"] == 0
    assert exact["canonical_minus_current"] == 0
    assert not different["exact"]
    assert different["current_minus_canonical"] == 1
    assert different["canonical_minus_current"] == 1


def test_subject_split_is_deterministic_and_detects_overlap(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = assign_subject_split(12345, config)
    second = assign_subject_split(12345, config)
    assert first == second
    assert first.split in {"train", "validation", "locked_test"}

    leaking = pd.DataFrame(
        {"subject_id": [1, 1], "split": ["train", "validation"]}
    )
    with pytest.raises(RuntimeError, match="leakage"):
        assert_subject_split_integrity(leaking)


def test_opposite_directions_form_an_informative_complete_case_score() -> None:
    labels = np.repeat([0, 1], 60)
    upward = np.r_[np.linspace(-3, -1, 60), np.linspace(1, 3, 60)]
    downward = -upward
    frame = pd.DataFrame({"up": upward, "down": downward})

    specification = fit_frozen_score(
        frame,
        labels,
        score_id="opposite-directions",
        endpoint_id="synthetic",
        components=("up", "down"),
        origin="unit_test",
        version="1",
    )
    score = apply_frozen_score(frame, specification)
    assert specification.directions == (1, -1)
    assert roc_auc_score(labels, score) > 0.99

    incomplete = frame.copy()
    incomplete.loc[0, "down"] = np.nan
    assert np.isnan(apply_frozen_score(incomplete, specification)[0])


def test_fdr_excludes_nonformal_results() -> None:
    frame = pd.DataFrame(
        {
            "endpoint_id": ["x", "x", "x"],
            "panel_id": ["primary_core"] * 3,
            "p_value": [0.001, 0.001, 0.9],
            "inference_status": [
                "formal_inference",
                "underpowered",
                "formal_inference",
            ],
        }
    )
    result = apply_fdr_by_family(frame)
    assert result.loc[0, "fdr_significant"]
    assert np.isnan(result.loc[1, "q_value"])
    assert not result.loc[1, "fdr_significant"]


def test_fdr_reports_global_and_hierarchical_sensitivity_separately() -> None:
    frame = pd.DataFrame(
        {
            "endpoint_id": ["focused"] + ["broad"] * 99,
            "panel_id": ["primary_core"] * 100,
            "evaluation_split": ["validation"] * 100,
            "p_value": [0.01] + [1.0] * 99,
            "inference_status": ["formal_inference"] * 100,
        }
    )

    result = apply_fdr_by_family(
        frame,
        family_columns=("endpoint_id", "panel_id", "evaluation_split"),
    )

    assert result.loc[0, "fdr_significant"]
    assert not result.loc[0, "global_fdr_significant"]
    assert result.loc[0, "hierarchical_family_selected"]
    assert result.loc[0, "hierarchical_fdr_significant"]
    assert result.loc[0, "global_q_value"] > result.loc[0, "q_value"]


def test_marker_assessability_exposes_differential_test_yield_gate() -> None:
    labels = np.repeat([0, 1], 50)
    values = np.arange(100, dtype=float)
    values[:30] = np.nan

    metrics = evaluate_marker(values, labels)

    assert metrics["n_input"] == 100
    assert metrics["n_evaluable"] == 70
    assert metrics["test_yield"] == pytest.approx(0.70)
    assert metrics["negative_test_yield"] == pytest.approx(0.40)
    assert metrics["positive_test_yield"] == pytest.approx(1.0)
    assert metrics["test_yield_gap"] == pytest.approx(0.60)
    assert not metrics["assessability_gate_passed"]
    assert metrics["missingness_sensitivity_status"] == (
        "worst_case_auc_bounds_computed"
    )
    assert metrics["missingness_auc_lower_bound"] == pytest.approx(0.40)
    assert metrics["missingness_auc_upper_bound"] == pytest.approx(1.0)
    assert not metrics["missingness_sensitivity_gate_passed"]
    assert not metrics["confirmatory_assessability_gate_passed"]
    assert "differential_test_yield" in metrics["assessability_status"]


def test_descriptive_only_metrics_do_not_expose_a_p_value() -> None:
    labels = np.repeat([0, 1], 8)
    values = np.r_[np.linspace(0, 1, 8), np.linspace(2, 3, 8)]
    metrics = evaluate_marker(values, labels)
    descriptive = enforce_inference_policy(metrics, "descriptive_only")
    underpowered = enforce_inference_policy(metrics, "underpowered")

    assert descriptive["roc_auc"] == pytest.approx(1.0)
    assert descriptive["rank_biserial"] == pytest.approx(1.0)
    assert descriptive["p_value"] is None
    assert underpowered["p_value"] is not None


def test_cluster_bootstrap_is_reproducible() -> None:
    subjects = np.repeat(np.arange(40), 2)
    labels = np.repeat(np.arange(40) >= 20, 2).astype(int)
    values = labels + np.tile([0.0, 0.1], 40)
    metric = lambda score, truth: roc_auc_score(truth, score)
    first = subject_cluster_bootstrap(
        subjects,
        values,
        labels,
        metric,
        iterations=100,
        seed=7,
    )
    second = subject_cluster_bootstrap(
        subjects,
        values,
        labels,
        metric,
        iterations=100,
        seed=7,
    )
    assert first == second


def test_test_lock_can_only_be_consumed_once() -> None:
    lock = MimicTestLock(run_id="run", freeze_hash="abc")
    lock.assert_evaluable("abc")
    lock.begin_evaluation()
    with pytest.raises(RuntimeError, match="already started"):
        lock.assert_evaluable("abc")
    lock.consume("artifact")
    with pytest.raises(RuntimeError, match="only start once"):
        lock.begin_evaluation()


def test_frozen_development_run_cannot_be_reused(tmp_path: Path) -> None:
    run_root = tmp_path / "mimic-development"
    run_root.mkdir()
    assert_development_run_mutable(
        run_root,
        operation="Development validation",
    )

    (run_root / "freeze_manifest.json").write_text(
        "{}",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="must use a new run ID"):
        assert_development_run_mutable(
            run_root,
            operation="Development validation",
        )

    (run_root / "test_lock.json").write_text("{}", encoding="ascii")
    with pytest.raises(
        RuntimeError,
        match="freeze_manifest.json, test_lock.json",
    ):
        assert_development_run_mutable(
            run_root,
            operation="Phenome discovery",
        )


def test_signal_leads_and_units_are_strict() -> None:
    names = [
        "V6",
        "II",
        "I",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
    ]
    assert _strict_lead_indices(names) == [2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    with pytest.raises(WaveformValidationError, match="Missing required"):
        _strict_lead_indices(names[:-1])

    signal = np.ones((2, 3))
    converted = _convert_to_millivolts(
        signal,
        ("V", "mV", "uV"),
    )
    np.testing.assert_allclose(converted[0], [1000.0, 1.0, 0.001])
    with pytest.raises(WaveformValidationError, match="Unsupported"):
        _convert_to_millivolts(signal[:, :1], ("counts",))


def test_representative_beat_excludes_implausible_and_fiducial_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def beat(r_peak: int, qrs_offset: int) -> BeatDelineation:
        return BeatDelineation(
            r_peak=MarkerPoint(r_peak, 0.9, "test"),
            r_onset=MarkerPoint(r_peak - 20, 0.8, "test"),
            r_offset=MarkerPoint(qrs_offset, 0.8, "test"),
            t_onset=MarkerPoint(r_peak + 70, 0.8, "test"),
            t_peak=MarkerPoint(r_peak + 120, 0.8, "test"),
            t_offset=MarkerPoint(r_peak + 190, 0.8, "test"),
        )

    normal = [beat(500 + index * 500, 530 + index * 500) for index in range(4)]
    fiducial_outlier = beat(2_500, 2_620)
    impossible = beat(3_000, 2_990)
    delineation = RecordDelineation(
        beats=[*normal, fiducial_outlier, impossible],
        fs=500,
        n_samples=4_000,
        ecg_id=1,
    )
    original_selector = RecordDelineation._select_accepted_beats
    selector_calls = 0

    def counted_selector(plausible):
        nonlocal selector_calls
        selector_calls += 1
        return original_selector(plausible)

    monkeypatch.setattr(
        RecordDelineation,
        "_select_accepted_beats",
        staticmethod(counted_selector),
    )

    assert delineation.n_beats == 6
    assert delineation.n_plausible_beats == 5
    assert delineation.n_accepted_beats == 4
    assert delineation.median_beat in normal

    returned = delineation.accepted_beats
    returned.clear()
    assert delineation.n_accepted_beats == 4
    assert selector_calls == 1

    normal[0].r_peak.confidence = 0.1
    assert delineation.n_plausible_beats == 4
    assert delineation.median_beat is not normal[0]
    assert selector_calls == 2


def test_multilead_delineation_handles_single_sample_edge_windows() -> None:
    delineator = MultileadDelineator()
    ecg = np.zeros((1_000, 12), dtype=float)

    assert delineator._find_p_offset_multilead(ecg, 11, 500) is None
    assert delineator._find_j_point(ecg, 979, 500) is None
    assert delineator._find_t_offset_multilead(ecg, 750, 500) is None


def test_any_nonfinite_signal_sample_is_not_usable() -> None:
    time = np.arange(5000) / 500.0
    lead_i = 0.5 * np.sin(2 * np.pi * time)
    signal = np.column_stack(
        [
            lead_i,
            lead_i * 1.5,
            lead_i * 0.5,
            *[lead_i * (1 + index / 20) for index in range(9)],
        ]
    )
    signal[100, 7] = np.nan
    quality = assess_signal_quality(signal, 500)
    assert not quality.usable
    assert "non_finite_samples" in quality.reasons


def test_contiguous_st_leads_require_an_actual_adjacent_pair() -> None:
    assert not _has_contiguous_pair({"V1": True, "V4": True})
    assert _has_contiguous_pair({"V2": True, "V3": True})


@pytest.mark.parametrize("powerline_hz,other_hz", [(50.0, 60.0), (60.0, 50.0)])
def test_signal_paths_notch_the_configured_powerline_frequency(
    powerline_hz: float,
    other_hz: float,
) -> None:
    fs = 500
    time = np.arange(fs * 10) / fs
    interference = np.sin(2.0 * np.pi * powerline_hz * time)
    baseline = 0.2 * np.sin(2.0 * np.pi * 1.0 * time)
    signal = np.tile((baseline + interference)[:, None], (1, 12))

    correctly_notched = prepare_signal_paths(
        signal,
        fs,
        powerline=powerline_hz,
    )["diagnostic"][:, 0]
    wrong_notch = prepare_signal_paths(
        signal,
        fs,
        powerline=other_hz,
    )["diagnostic"][:, 0]
    interior = slice(fs, -fs)
    basis = interference[interior]

    def projected_amplitude(values: np.ndarray) -> float:
        return float(
            2.0
            * abs(np.dot(values[interior], basis))
            / len(basis)
        )

    assert projected_amplitude(correctly_notched) < (
        0.25 * projected_amplitude(wrong_notch)
    )


def test_morphology_uses_the_full_150hz_diagnostic_path_without_refiltering() -> None:
    fs = 500
    time = np.arange(fs * 10) / fs
    signal = np.tile(np.sin(2 * np.pi * 120.0 * time)[:, None], (1, 12))
    paths = prepare_signal_paths(signal, fs, powerline=50.0)
    assert paths["morphology"] is paths["diagnostic"]
    interior = paths["morphology"][fs:-fs, 0]
    assert float(np.sqrt(np.mean(interior**2))) > 0.25


def test_polyphase_resampling_preserves_noninteger_timebase_and_edges() -> None:
    fs_orig = 499.5
    fs_target = 500.0
    values = np.zeros((4_995, 2), dtype=float)
    values[:, 1] = np.linspace(-1.0, 1.0, len(values))
    values[round(fs_orig), 0] = 1.0
    output = resample_signal(values, fs_orig, fs_target)
    assert output.shape == (5_000, 2)
    assert abs(int(np.argmax(output[:, 0])) - 500) <= 1
    # Line padding avoids the large periodic wrap-around discontinuity of
    # Fourier resampling at a ten-second record boundary.
    assert output[0, 1] == pytest.approx(-1.0, abs=0.01)
    assert output[-1, 1] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("frequency", [50.0, 60.0])
def test_signal_qc_detects_both_mains_frequencies(frequency: float) -> None:
    fs = 500
    time = np.arange(fs * 10) / fs
    base = 0.2 * np.sin(2 * np.pi * 1.3 * time)
    interference = 0.8 * np.sin(2 * np.pi * frequency * time)
    signal = np.tile((base + interference)[:, None], (1, 12))
    quality = assess_signal_quality(
        signal,
        fs,
        expected_powerline_hz=frequency,
    )
    assert quality.detected_powerline_hz == frequency
    assert quality.powerline_ratio > 0.08
    assert any("powerline_interference" in reason for reason in quality.reasons)


def test_signal_qc_flags_classic_limb_reversal_pattern_without_calling_it_disease() -> None:
    fs = 500
    time = np.arange(fs * 10) / fs
    qrs_like = np.sin(2 * np.pi * 12.0 * time)
    signal = np.tile(qrs_like[:, None], (1, 12))
    signal[:, 0] = -2.0 * np.maximum(qrs_like, 0.0) + 0.2 * np.minimum(qrs_like, 0.0)
    signal[:, 3] = 2.0 * np.maximum(qrs_like, 0.0) + 0.2 * np.minimum(qrs_like, 0.0)
    quality = assess_signal_quality(signal, fs)
    assert quality.limb_lead_reversal_suspected
    assert "limb_lead_reversal_or_dextrocardia_suspected" in quality.reasons


def test_v2_v3_st_thresholds_are_sex_age_specific_and_excludable() -> None:
    fs = 500
    ecg = np.zeros((1_000, 12), dtype=float)
    ecg[450, LEAD_MAP["V2"]] = 0.20
    ecg[450, LEAD_MAP["V3"]] = 0.20
    beat = BeatDelineation(
        r_peak=MarkerPoint(425, 1.0, "test"),
        r_onset=MarkerPoint(400, 1.0, "test"),
        r_offset=MarkerPoint(450, 1.0, "test"),
        j_point=MarkerPoint(450, 1.0, "test"),
        t_onset=MarkerPoint(500, 1.0, "test"),
        t_peak=MarkerPoint(600, 1.0, "test"),
        t_offset=MarkerPoint(700, 1.0, "test"),
    )
    delineation = RecordDelineation(
        beats=[beat],
        fs=fs,
        n_samples=len(ecg),
        ecg_id=1,
    )
    criteria = ClassicalECGCriteria()

    female = criteria.compute(ecg, delineation, age=45, sex="F")
    young_male = criteria.compute(ecg, delineation, age=35, sex="M")
    unknown_demographics = criteria.compute(
        ecg,
        delineation,
        age=None,
        sex=None,
    )
    excluded = criteria.compute(
        ecg,
        delineation,
        age=45,
        sex="F",
        paced_or_lbbb=True,
    )

    assert female["st_elevation_contiguous"].positive is True
    assert young_male["st_elevation_contiguous"].positive is False
    assert unknown_demographics["st_elevation_contiguous"].positive is None
    assert not unknown_demographics["st_elevation_contiguous"].evaluable
    assert not excluded["st_elevation_contiguous"].evaluable


def _synthetic_classical_delineation() -> RecordDelineation:
    beat = BeatDelineation(
        r_peak=MarkerPoint(425, 1.0, "test"),
        p_onset=MarkerPoint(300, 1.0, "test"),
        p_offset=MarkerPoint(360, 1.0, "test"),
        r_onset=MarkerPoint(400, 1.0, "test"),
        r_offset=MarkerPoint(450, 1.0, "test"),
        j_point=MarkerPoint(450, 1.0, "test"),
        t_onset=MarkerPoint(500, 1.0, "test"),
        t_peak=MarkerPoint(600, 1.0, "test"),
        t_offset=MarkerPoint(700, 1.0, "test"),
    )
    return RecordDelineation(
        beats=[beat],
        fs=500,
        n_samples=1_000,
        ecg_id=1,
    )


def test_named_ischemia_patterns_are_detected_but_remain_exploratory() -> None:
    criteria = ClassicalECGCriteria()
    delineation = _synthetic_classical_delineation()

    wellens_ecg = np.zeros((1_000, 12), dtype=float)
    for lead in ("V2", "V3"):
        wellens_ecg[560:640, LEAD_MAP[lead]] = -0.25
    wellens = criteria.compute(
        wellens_ecg,
        delineation,
        age=50,
        sex="F",
    )["wellens_ecg_pattern"]
    assert wellens.positive is True
    assert wellens.exploratory is True

    de_winter_ecg = np.zeros((1_000, 12), dtype=float)
    for lead in ("V2", "V3"):
        de_winter_ecg[450, LEAD_MAP[lead]] = -0.12
        de_winter_ecg[490, LEAD_MAP[lead]] = -0.04
        de_winter_ecg[560:640, LEAD_MAP[lead]] = 0.60
    de_winter = criteria.compute(
        de_winter_ecg,
        delineation,
        age=50,
        sex="F",
    )["de_winter_ecg_pattern"]
    assert de_winter.positive is True
    assert de_winter.exploratory is True


def test_s1q3t3_is_nonspecific_expert_pattern_and_qrs_rules_are_excludable() -> None:
    ecg = np.zeros((1_000, 12), dtype=float)
    ecg[430, LEAD_MAP["I"]] = -0.20
    ecg[400:415, LEAD_MAP["III"]] = -0.15
    ecg[425, LEAD_MAP["III"]] = 0.40
    ecg[560:640, LEAD_MAP["III"]] = -0.20
    criteria = ClassicalECGCriteria()
    delineation = _synthetic_classical_delineation()

    result = criteria.compute(ecg, delineation, age=50, sex="F")
    pattern = result["s1q3t3_pattern"]
    assert pattern.positive is True
    assert pattern.exploratory is True
    assert "nonspecific" in pattern.definition

    excluded = criteria.compute(
        ecg,
        delineation,
        age=50,
        sex="F",
        paced_or_lbbb=True,
    )
    for criterion_id in (
        "sokolow_lyon",
        "right_axis_deviation",
        "dominant_r_v1",
        "s1q3t3_pattern",
    ):
        assert not excluded[criterion_id].evaluable
        assert excluded[criterion_id].positive is None
    assert excluded["p_pulmonale"].evaluable
    # A recorder LBBB statement must not blank the bundle-branch patterns
    # themselves -- otherwise the LBBB pattern is never assessed on LBBB.
    lbbb_only = criteria.compute(ecg, delineation, age=50, sex="F", paced_or_lbbb=True, paced=False)
    assert not lbbb_only["sokolow_lyon"].evaluable
    for criterion_id in ("lbbb_pattern", "rbbb_pattern"):
        assert "paced" not in lbbb_only[criterion_id].exclusion_reason
        assert "LBBB" not in lbbb_only[criterion_id].exclusion_reason
    paced = criteria.compute(ecg, delineation, age=50, sex="F", paced_or_lbbb=True, paced=True)
    for criterion_id in ("lbbb_pattern", "rbbb_pattern"):
        assert not paced[criterion_id].evaluable
        assert paced[criterion_id].exclusion_reason == (
            "paced rhythm confounds QRS morphology classification"
        )


def test_electrolyte_pattern_components_are_reported_separately() -> None:
    fs = 500
    beat = BeatDelineation(
        r_peak=MarkerPoint(500, 1.0, "test"),
        p_onset=MarkerPoint(250, 1.0, "test"),
        p_peak=MarkerPoint(275, 1.0, "test"),
        p_offset=MarkerPoint(300, 1.0, "test"),
        r_onset=MarkerPoint(450, 1.0, "test"),
        r_offset=MarkerPoint(530, 1.0, "test"),
        j_point=MarkerPoint(530, 1.0, "test"),
        t_onset=MarkerPoint(560, 1.0, "test"),
        t_peak=MarkerPoint(640, 1.0, "test"),
        t_offset=MarkerPoint(720, 1.0, "test"),
    )
    delineation = RecordDelineation(
        beats=[beat],
        fs=fs,
        n_samples=1_000,
        ecg_id=1,
    )
    ecg = np.zeros((1_000, 12), dtype=float)
    ecg[275, LEAD_MAP["II"]] = 0.02
    ecg[275, LEAD_MAP["V1"]] = 0.02

    results = ClassicalECGCriteria().compute(ecg, delineation)

    assert results["qrs_widening_ge_120"].positive is True
    assert results["pr_prolongation_gt_200"].positive is True
    assert results["p_wave_attenuation"].positive is True
    assert all(
        results[name].exploratory
        for name in (
            "qrs_widening_ge_120",
            "pr_prolongation_gt_200",
            "p_wave_attenuation",
        )
    )


def test_panels_are_concrete_and_have_frozen_sizes() -> None:
    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    core = panels["primary_core"]
    sentinel = panels["sentinel"]
    discovery = panels["discovery_reserve"]
    criteria = panels["classical_criteria"]
    assert len(core.feature_ids) == 80
    assert len(sentinel.feature_ids) == 240
    assert discovery.feature_ids
    # The criteria panel holds the verdict of every classical rule, one for
    # one with the continuous quantity the rule is cut from, plus the
    # published reference scores (audit 5.1) and their verdicts where the
    # source gives a cut-off.
    from vcg_biomarker.mimic.panels import (
        REFERENCE_SCORE_FEATURE_IDS,
        REFERENCE_SCORE_VERDICT_IDS,
    )

    assert len(criteria.feature_ids) == 39 + 10 + 4
    core_verdicts = [
        feature_id for feature_id in criteria.feature_ids
        if feature_id.endswith("_positive")
        and feature_id not in REFERENCE_SCORE_VERDICT_IDS
    ]
    assert {
        feature_id.removesuffix("_positive") for feature_id in core_verdicts
    } == {
        feature_id
        for feature_id in core.feature_ids
        if feature_id.startswith("classical_")
    }
    assert set(REFERENCE_SCORE_FEATURE_IDS) <= set(criteria.feature_ids)
    assert not set(REFERENCE_SCORE_FEATURE_IDS) & set(core.feature_ids)
    assert {
        feature_id.removesuffix("_positive")
        for feature_id in REFERENCE_SCORE_VERDICT_IDS
    } <= set(REFERENCE_SCORE_FEATURE_IDS)
    assert set(CORE_REQUIRED_VCG) <= set(core.feature_ids)
    assert "classical_hyperkalemia_ecg_heuristic" in core.feature_ids
    assert len(
        [
            feature_id
            for feature_id in core.feature_ids
            if feature_id.startswith("classical_")
        ]
    ) == 39
    assert {
        "classical_qrs_widening_ge_120",
        "classical_pr_prolongation_gt_200",
        "classical_p_wave_attenuation",
    } <= set(core.feature_ids)
    assert {
        "classical_wellens_ecg_pattern",
        "classical_de_winter_ecg_pattern",
        "classical_s1q3t3_pattern",
    } <= set(core.feature_ids)
    vcg_features = {
        feature_id
        for panel in (core, sentinel, discovery)
        for feature_id in panel.feature_ids
        if not feature_id.startswith("classical_")
    }
    assert all(
        BiomarkerRegistry.definition(feature_id).unit.casefold()
        not in {"categorical", "text", "vector", "array"}
        for feature_id in vcg_features
    )
    assert "loop_quality_stratified_delta" not in vcg_features
    assert "sex_age_adjusted_residual_candidate" not in vcg_features
    with pytest.raises(ValueError, match="template"):
        FeaturePanelSpec(
            panel_id="invalid",
            version="1",
            tier=PanelTier.SENTINEL,
            feature_ids=("qrs_*",),
        )


def test_machine_report_parser_separates_negation_uncertainty_and_absence() -> None:
    parser = MachineReportParser()
    assert (
        parser.parse("No ST elevation.")["st_elevation"].state
        == ReportMentionState.NEGATIVE
    )
    assert (
        parser.parse("Cannot rule out old infarct.")["infarct"].state
        == ReportMentionState.UNCERTAIN
    )
    assert (
        parser.parse("Normal sinus rhythm.")["af"].state
        == ReportMentionState.NOT_MENTIONED
    )
    assert (
        parser.parse("Atrial fibrillation.")["af"].state
        == ReportMentionState.POSITIVE
    )


def test_feature_artifact_corruption_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "features"
    partition = root / "split=train" / "part-000000.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [11],
            "split": ["train"],
            "value": [0.5],
        }
    ).to_parquet(partition, index=False)
    fingerprint = file_fingerprint(partition, include_sha256=True)
    atomic_write_json(
        partition.with_suffix(".json"),
        {
            "part_hash": "part",
            "study_ids": [11],
            "row_count": 1,
            "parquet_size": fingerprint["size"],
            "parquet_sha256": fingerprint["sha256"],
        },
    )
    atomic_write_json(root / "features.manifest.json", {"status": "complete"})
    stage_input_payload = {"test": "input"}
    _write_feature_artifact_index(
        root,
        stage_input_hash=content_hash(stage_input_payload),
        stage_input_payload=stage_input_payload,
        source_code_hash=feature_implementation_hash(),
        panel_hash="panel",
        feature_ids=("value",),
        parquet_paths=(partition,),
        destination=root / "partition_index.json",
    )
    assert verify_feature_artifacts((partition,))["status"] == "verified"

    index_path = root / "partition_index.json"
    artifact_index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_index["source_code_hash"] = "stale-code"
    atomic_write_json(index_path, artifact_index)
    with pytest.raises(RuntimeError, match="different extraction code"):
        verify_feature_artifacts((partition,))
    artifact_index["source_code_hash"] = feature_implementation_hash()
    atomic_write_json(index_path, artifact_index)

    with partition.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_feature_artifacts((partition,))


def test_feature_artifact_rejects_duplicate_studies_across_partitions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    partitions = (
        root / "split=train" / "part-000000.parquet",
        root / "split=validation" / "part-000001.parquet",
    )
    for subject_id, split, partition in (
        (1, "train", partitions[0]),
        (2, "validation", partitions[1]),
    ):
        partition.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": [subject_id],
                "study_id": [11],
                "split": [split],
                "value": [0.5],
            }
        ).to_parquet(partition, index=False)
        fingerprint = file_fingerprint(partition, include_sha256=True)
        atomic_write_json(
            partition.with_suffix(".json"),
            {
                "part_hash": f"part-{split}",
                "study_ids": [11],
                "row_count": 1,
                "parquet_size": fingerprint["size"],
                "parquet_sha256": fingerprint["sha256"],
            },
        )
    atomic_write_json(root / "features.manifest.json", {"status": "complete"})
    stage_input_payload = {"test": "input"}
    _write_feature_artifact_index(
        root,
        stage_input_hash=content_hash(stage_input_payload),
        stage_input_payload=stage_input_payload,
        source_code_hash=feature_implementation_hash(),
        panel_hash="panel",
        feature_ids=("value",),
        parquet_paths=partitions,
        destination=root / "partition_index.json",
    )

    with pytest.raises(RuntimeError, match="across partitions"):
        verify_feature_artifacts(partitions)


def test_development_validation_rejects_every_locked_test_artifact(
    tmp_path: Path,
) -> None:
    all_root = tmp_path / "all"
    all_root.mkdir()
    manifest_path = all_root / "features.manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "status": "complete",
            "input_hash": "feature-input",
            "notes": [
                "locked_test_features_included=True",
                "locked_test_selection=blind_all_master_ecgs",
                "feature_artifact_hash=artifact-hash",
            ]
        },
    )
    integrity = {
        "root": str(all_root),
        "manifest_path": str(manifest_path),
        "complete_index": True,
        "stage_input_hash": "feature-input",
        "artifact_hash": "artifact-hash",
    }
    with pytest.raises(RuntimeError, match="locked_test"):
        _resolved_feature_scope("explicit", integrity)

    development_root = tmp_path / "development"
    development_root.mkdir()
    development_manifest = development_root / "features.manifest.json"
    atomic_write_json(
        development_manifest,
        {
            "status": "complete",
            "input_hash": "feature-input",
            "notes": [
                "locked_test_features_included=False",
                "locked_test_only=False",
                "locked_test_selection=not_loaded",
                "feature_artifact_hash=artifact-hash",
            ]
        },
    )
    integrity.update(
        {
            "root": str(development_root),
            "manifest_path": str(development_manifest),
        }
    )
    assert _resolved_feature_scope("explicit", integrity) == "development_only"

    smoke_root = tmp_path / "smoke-development-1000"
    smoke_root.mkdir()
    smoke_manifest = smoke_root / "features.manifest.json"
    atomic_write_json(
        smoke_manifest,
        {
            "status": "complete",
            "input_hash": "feature-input",
            "notes": [
                "locked_test_features_included=False",
                "locked_test_only=False",
                "locked_test_selection=not_loaded",
                "feature_artifact_hash=artifact-hash",
            ]
        },
    )
    integrity.update(
        {
            "root": str(smoke_root),
            "manifest_path": str(smoke_manifest),
        }
    )
    assert _resolved_feature_scope("explicit", integrity) == (
        "development_preview"
    )

    integrity["complete_index"] = False
    with pytest.raises(RuntimeError, match="complete indexed"):
        _resolved_feature_scope("explicit", integrity)


def test_quality_metadata_cannot_enter_exploratory_clinical_scores() -> None:
    assert not _is_clinical_combination_candidate(
        "kors_vs_pca_axis_difference_deg"
    )
    assert not _is_clinical_combination_candidate(
        "lead_to_vcg_consistency_score"
    )
    assert _is_clinical_combination_candidate("qrs_loop_elongation")


def test_echo_text_categories_preserve_assessable_positive_and_negative_states(
) -> None:
    rules = _label_rules()
    connection = duckdb.connect()
    try:
        connection.register(
            "ctx",
            pd.DataFrame(
                {
                    "has_tte_30d": [True],
                    "la_volume_index": [np.nan],
                    "la_size_category": ["LA Elongated (4Ch>5.2cm)"],
                    "septal_thickness_cm": [np.nan],
                    "posterior_wall_thickness_cm": [np.nan],
                    "lv_wall_category": ["Normal (<1.2cm)"],
                    "rv_basal_diameter_cm": [np.nan],
                    "rv_size_category": ["NL for BSA"],
                    "tapse_cm": [np.nan],
                    "rv_function_category": ["Hyperdynamic"],
                }
            ),
        )
        select_list = ", ".join(
            f"({rules[label_id].state}) AS {label_id}"
            for label_id in (
                "echo_lae",
                "lv_wall_thickening",
                "echo_rv_dilation",
                "echo_rv_systolic_dysfunction",
            )
        )
        row = connection.execute(
            f"SELECT {select_list} FROM ctx"
        ).fetchone()
    finally:
        connection.close()

    assert row == ("positive", "negative", "negative", "negative")


def test_evidence_and_labels_do_not_record_the_same_output_file():
    """Two stages writing one path under their own manifests broke each
    other's integrity record on every run (A091-A093 restarted from the
    evidence build each time). The evidence stage keeps its definitions copy
    under its own root."""
    from pathlib import Path
    import vcg_biomarker.mimic.evidence as evidence_module

    source = Path(evidence_module.__file__).read_text(encoding="utf-8")
    assert 'config.evidence_root / "label_definitions.json"' in source
    assert 'config.labels_root / "label_definitions.json"' not in source
