"""End-to-end mathematical QC coverage, not clinical reference limits."""
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import (
    SPECS, operator_pair,
)
from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
from vcg_biomarker.mimic.marker_plausibility import check_frame, declared_range


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_every_mechanism_has_qc_and_detects_corrupted_values(spec):
    limits = declared_range(spec.name, "dimensionless")
    assert limits is not None
    if spec.name == "t_energy_temporal_concentration":
        assert limits.lower == 1.0 and limits.upper is None
        report = check_frame(pd.DataFrame({spec.name: [1., 20., .99, -1., np.nan]}), {spec.name: limits})
        assert report.iloc[0].checked_values == 4
        assert report.iloc[0].violating_values == 2
        assert not limits.violations(np.array([1. - 2e-16]))[0]
        return
    assert limits.lower == 0.0 and 0 < limits.upper <= 1.0
    values = [0., limits.upper, -.01, limits.upper + .01, np.nan]
    report = check_frame(pd.DataFrame({spec.name: values}), {spec.name: limits})
    assert report.iloc[0].checked_values == 4
    assert report.iloc[0].violating_values == 2
    assert not limits.violations(np.array([limits.upper + 2e-16]))[0]


def test_extremal_operators_and_random_rotations_pass_qc():
    rng = np.random.default_rng(53)
    for _ in range(100):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qrs = np.tile([1., 0., 0.], (20, 1)) @ rotation
        for direction in ([1., 0., 0.], [0., 1., 0.], [1., 1., 0.]):
            result = operator_pair(qrs, np.tile(direction, (20, 1)) @ rotation)
            for suffix, value in result.items():
                limits = declared_range(f"qrs_st_directional_operator_{suffix}", "dimensionless")
                assert not limits.violations(np.array([value]))[0]


def test_freeze_preserves_actual_qc_contract_for_all_mechanism_markers():
    payload = marker_freeze_payload()
    rows = {row["marker_id"]: row for row in payload["markers"]}
    for spec in SPECS:
        limits = declared_range(spec.name, "dimensionless")
        assert rows[spec.name]["mathematical_plausibility_range"] == {
            "lower": limits.lower, "upper": limits.upper,
            "reason": limits.reason, "numerical_tolerance": limits.numerical_tolerance,
        }
        assert rows[spec.name]["initial_evidence_tier"] == "exploratory"


def test_extraction_partition_report_reaches_all_new_mechanism_columns(tmp_path):
    from vcg_biomarker.mimic.feature_extraction import _plausibility_report

    partition = tmp_path / "split=train"
    partition.mkdir()
    names = [spec.name for spec in SPECS]
    pd.DataFrame({name: [0.1, 2.0, np.nan] for name in names}).to_parquet(
        partition / "part-00000.parquet", index=False,
    )
    report = _plausibility_report(tmp_path, names)
    assert set(report.marker_id) == set(names)
    assert report.checked_values.eq(2).all()
    assert report.violating_values.eq(1).all()
