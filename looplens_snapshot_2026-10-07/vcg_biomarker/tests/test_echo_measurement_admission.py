"""Five declared echo columns were empty for the life of the project.

The echocardiogram is this project's best reference standard: quantitative,
timed, and the only Tier A source for structural endpoints.  The source carries
302 distinct measurement codes and the wide table drew on 69.

Worse than the ones never asked for were the five that were asked for by a name
that does not exist in MIMIC's vocabulary.  Measured over the 206,488 studies:

    tissue_e_prime_septal      septal_e_prime  ->  sept_e_prime         63.9 %
    tissue_e_prime_lateral     lateral_e_prime ->  lat_e_prime          64.5 %
    aortic_mean_gradient_mmhg  av_mn_grad      ->  av_mean_grad         16.5 %
    aortic_valve_area_cm2      av_area         ->  av_area_continuity   12.0 %
    ascending_aorta_diameter   asc_ao_diam     ->  ascending_diam       70.7 %

Nothing failed, because an empty column and an unmeasured quantity are
indistinguishable downstream.  Both halves of E/e-prime were missing from every
diastolic endpoint.

Rebuilt against the real long table, all 78 generated columns fill and none
stays empty; the derived stroke volume lands on 113,111 studies with a median
of 69.3 mL.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import vcg_biomarker.mimic.evidence as evidence
from vcg_biomarker.mimic.acceptance import _check_echo_admission
from vcg_biomarker.mimic.config import MimicProjectConfig


def test_no_declared_source_name_is_one_that_was_never_there():
    # The five wrong names, so a revert cannot pass unnoticed.
    retired = {
        "septal_e_prime",
        "lateral_e_prime",
        "e_prime_septal",
        "e_prime_lateral",
        "av_mn_grad",
        "av_mean_gradient",
        "av_area",
        "ava",
        "asc_ao_diam",
        "ascending_aorta",
    }
    declared = {
        name
        for _, sources in evidence._ECHO_NUMERIC_MEASUREMENTS
        for name in sources
    }
    assert declared.isdisjoint(retired), sorted(declared & retired)


def test_the_tissue_doppler_columns_name_the_unit_they_arrive_in():
    # A median of 0.07 and 0.09 is metres per second. The columns used to
    # promise centimetres, which is a lie a downstream reader cannot see.
    columns = {column for column, _ in evidence._ECHO_NUMERIC_MEASUREMENTS}
    assert "tissue_e_prime_septal_m_per_s" in columns
    assert "tissue_e_prime_lateral_m_per_s" in columns
    assert "tissue_e_prime_septal_cm_per_s" not in columns


def test_every_registry_entry_is_unique_and_named_once():
    numeric = [column for column, _ in evidence._ECHO_NUMERIC_MEASUREMENTS]
    text = [column for column, _ in evidence._ECHO_TEXT_MEASUREMENTS]
    ordinal = [column for column, _ in evidence._ECHO_ORDINAL_MEASUREMENTS]
    segments = list(evidence._ECHO_WALL_MOTION_SEGMENTS)
    for group in (numeric, text, ordinal, segments):
        assert len(group) == len(set(group))
    assert set(numeric).isdisjoint(text)
    assert set(numeric).isdisjoint(segments)
    assert set(text).isdisjoint(segments)


def test_the_wall_motion_segments_are_carried_individually():
    # Only the counts were kept before, which answers "is anything abnormal"
    # and never "where" -- the one question a vectorcardiogram claims to have
    # an opinion about.
    assert len(evidence._ECHO_WALL_MOTION_SEGMENTS) == 17
    declared = set(evidence.echo_derived_diastolic_columns())
    for segment in evidence._ECHO_WALL_MOTION_SEGMENTS:
        assert segment in declared


def test_the_stroke_volume_is_declared_as_a_wide_column():
    assert "lvot_stroke_volume_ml" in evidence.echo_derived_diastolic_columns()


def _write_wide(tmp_path: Path, frame: pd.DataFrame) -> MimicProjectConfig:
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
    frame.to_parquet(config.evidence_root / "echo_measurements_wide.parquet")
    return config


def _full_frame() -> pd.DataFrame:
    columns = evidence.echo_derived_diastolic_columns()
    return pd.DataFrame({name: [1.0, 2.0] for name in columns})


def test_a_fully_populated_echo_table_passes(tmp_path: Path):
    config = _write_wide(tmp_path, _full_frame())
    observed, expected, evidence_path = _check_echo_admission(config)
    assert observed["empty_columns"] == 0
    assert observed["declared_columns"] == len(
        evidence.echo_derived_diastolic_columns()
    )
    assert expected["empty_columns"] == 0
    assert evidence_path.endswith("echo_measurements_wide.parquet")


def test_a_column_that_is_empty_on_every_study_stops_the_audit(tmp_path: Path):
    frame = _full_frame()
    frame["aortic_valve_area_cm2"] = [None, None]
    config = _write_wide(tmp_path, frame)
    with pytest.raises(RuntimeError, match="no study at all"):
        _check_echo_admission(config)


def test_a_column_declared_but_never_written_stops_the_audit(tmp_path: Path):
    frame = _full_frame().drop(columns=["ivc_diameter_cm"])
    config = _write_wide(tmp_path, frame)
    with pytest.raises(RuntimeError, match="never written"):
        _check_echo_admission(config)


def test_a_rare_measurement_is_allowed_to_be_rare(tmp_path: Path):
    # Mitral valve area reaches 0.9 % of studies. That is the data, not a bug,
    # and the check must not confuse the two.
    frame = _full_frame()
    frame["mitral_valve_area_cm2"] = [2.2, None]
    config = _write_wide(tmp_path, frame)
    observed, _, _ = _check_echo_admission(config)
    assert observed["empty_columns"] == 0


def test_missing_echo_evidence_is_an_error_not_a_pass(tmp_path: Path):
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
    with pytest.raises(RuntimeError, match="missing"):
        _check_echo_admission(config)
