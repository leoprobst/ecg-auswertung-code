from __future__ import annotations

from vcg_biomarker.mimic.known_ecg_associations import (
    KNOWN_ECG_ASSOCIATIONS,
    classify_novelty,
    known_association_registry_summary,
)


def test_registry_has_unique_association_ids() -> None:
    ids = [entry.association_id for entry in KNOWN_ECG_ASSOCIATIONS]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 5


def test_registry_entries_have_no_empty_tokens_or_sources() -> None:
    for entry in KNOWN_ECG_ASSOCIATIONS:
        assert entry.marker_family_tokens
        assert entry.target_family_tokens
        assert all(token == token.casefold() for token in entry.marker_family_tokens)
        assert all(token == token.casefold() for token in entry.target_family_tokens)
        assert entry.summary_de.strip()
        assert entry.source.strip()


def test_qrs_t_angle_mortality_is_recognized_as_known() -> None:
    status = classify_novelty(
        feature_id="spatial_qrs_t_angle_deg",
        target_id="outcome_1y_mortality",
        target_name="1-year mortality",
    )
    assert status == "known_literature_association"


def test_p_terminal_force_atrial_fibrillation_is_recognized_as_known() -> None:
    status = classify_novelty(
        feature_id="p_terminal_force_v1",
        target_id="atrial_fibrillation_flutter",
        target_name="Atrial fibrillation",
    )
    assert status == "known_literature_association"


def test_age_gap_mortality_is_recognized_as_known() -> None:
    status = classify_novelty(
        feature_id="age_gap",
        target_id="outcome_90d_mortality",
        target_name="90-day mortality",
    )
    assert status == "known_literature_association"


def test_unrelated_marker_target_pair_is_not_yet_described() -> None:
    status = classify_novelty(
        feature_id="qrs_loop_planarity",
        target_id="socioeconomic_marital_widowed",
        target_name="Familienstand: Widowed",
    )
    assert status == "not_yet_described"


def test_marker_family_alone_without_matching_target_is_not_known() -> None:
    # spatial_qrs_t_angle is a known mortality marker, but paired with an
    # unrelated target family it must not be flagged as "known".
    status = classify_novelty(
        feature_id="spatial_qrs_t_angle_deg",
        target_id="socioeconomic_insurance_medicare",
        target_name="Versicherung: Medicare",
    )
    assert status == "not_yet_described"


def test_registry_summary_is_display_ready() -> None:
    summary = known_association_registry_summary()
    assert len(summary) == len(KNOWN_ECG_ASSOCIATIONS)
    for row in summary:
        assert set(row) == {"association_id", "summary_de", "source"}
