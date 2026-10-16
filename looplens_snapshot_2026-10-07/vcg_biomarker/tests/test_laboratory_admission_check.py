"""A laboratory group that vanishes must stop the run, not the analysis.

Every window aggregate in the evidence build filters on
``value_status = 'usable'``.  A group the unit rule does not recognise
therefore disappears from the analysis with no error anywhere -- which is how
28.6 million rows went missing while the screen went on reporting 727
laboratory targets.  Sodium, glucose, urea nitrogen, bicarbonate, leukocytes,
albumin, TSH, free T4, HbA1c, CRP, D-dimer, INR and the coagulation times were
all absent, and a group at 100 % quarantine looked exactly like a group nobody
had ordered.

Run against the evidence as it stood before the unit rule was rebuilt, this
check fails and names all fifteen groups.  Rows with no numeric result are kept
out of the denominator on purpose: a troponin ordered and never resulted is a
property of the hospital, not of this pipeline, and 58 % of troponin T rows are
exactly that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.acceptance import (
    AcceptanceExpectations,
    _check_laboratory_admission,
)
from vcg_biomarker.mimic.config import MimicProjectConfig


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def _write_labs(config: MimicProjectConfig, rows) -> None:
    frame = pd.DataFrame(rows, columns=["lab_group", "value_numeric", "value_status"])
    frame.to_parquet(config.evidence_root / "labs.parquet")


def test_a_fully_admitted_laboratory_set_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_labs(
        config,
        [("sodium", 139.0, "usable")] * 100
        + [("potassium", 4.1, "usable")] * 100,
    )
    observed, expected, evidence = _check_laboratory_admission(
        config, AcceptanceExpectations()
    )
    assert observed["groups"] == 2
    assert observed["refused_rows"] == 0
    assert expected["groups_above_limit"] == 0
    assert evidence.endswith("labs.parquet")


def test_a_group_the_unit_rule_does_not_know_stops_the_audit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_labs(
        config,
        [("potassium", 4.1, "usable")] * 100
        + [("sodium", 139.0, "unit_unrecognized")] * 100,
    )
    with pytest.raises(RuntimeError) as failure:
        _check_laboratory_admission(config, AcceptanceExpectations())
    assert "sodium" in str(failure.value)
    assert "potassium" not in str(failure.value)


def test_a_minority_of_refused_rows_is_tolerated(tmp_path: Path) -> None:
    # CK-MB legitimately loses its per-cent rows: the relative index is a
    # different quantity from the mass concentration, 12 % of the group.
    config = _config(tmp_path)
    _write_labs(
        config,
        [("ck_mb", 4.0, "usable")] * 88
        + [("ck_mb", 4.8, "unit_scale_ambiguous")] * 12,
    )
    observed, _, _ = _check_laboratory_admission(config, AcceptanceExpectations())
    assert observed["refused_rows"] == 12
    assert observed["worst_groups"]["ck_mb"] == pytest.approx(0.12)


def test_results_that_were_never_returned_do_not_count_against_the_rule(
    tmp_path: Path,
) -> None:
    # 268,313 of 459,872 troponin T rows carry no result at all. That is the
    # hospital, not the unit rule, and must not fail the audit.
    config = _config(tmp_path)
    _write_labs(
        config,
        [("troponin_t", 0.09, "usable")] * 40
        + [("troponin_t", None, "non_numeric")] * 60,
    )
    observed, _, _ = _check_laboratory_admission(config, AcceptanceExpectations())
    assert observed["numeric_rows"] == 40
    assert observed["refused_rows"] == 0


def test_the_tolerated_share_is_configurable_and_enforced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_labs(
        config,
        [("sodium", 139.0, "usable")] * 70
        + [("sodium", 139.0, "unit_unrecognized")] * 30,
    )
    # 30 % refused: allowed by the default, refused by a stricter expectation.
    _check_laboratory_admission(config, AcceptanceExpectations())
    with pytest.raises(RuntimeError):
        _check_laboratory_admission(
            config,
            AcceptanceExpectations(maximum_laboratory_quarantine_share=0.2),
        )


def test_missing_laboratory_evidence_is_an_error_not_a_pass(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        _check_laboratory_admission(config, AcceptanceExpectations())
