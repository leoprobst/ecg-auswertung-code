from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
from vcg_biomarker.mimic.research_coverage import research_coverage_rows


@pytest.fixture(scope="module")
def payload():
    return marker_freeze_payload()


def test_every_existing_numbered_chapter_is_accounted_for(payload):
    root = Path(__file__).resolve().parents[2]
    docs = {p.name for p in (root / "research_pathology_round_20260913").glob("[0-9]*.md")}
    rows = payload["research_coverage"]["chapters"]
    assert {Path(row["origin"]).name for row in rows} == docs
    assert len(rows) == len(docs) == 23
    for row in rows:
        assert row["unimplemented_or_unverified_requirements"]
        assert not row["clinical_claim_allowed"] and not row["analysis_complete"]


def test_each_declared_endpoint_pair_enters_the_model_inventory(payload):
    pairs = {(h["marker_id"], h["endpoint_id"]) for h in payload["mechanism_hypotheses"]}
    for chapter in payload["research_coverage"]["chapters"]:
        for marker in chapter["marker_ids"]:
            for endpoint in chapter["endpoint_ids"]:
                assert (marker, endpoint) in pairs


def test_missing_signal_implementation_fails_closed(payload):
    markers = [m for m in payload["markers"] if m["marker_id"] != "qtc_fridericia_ms"]
    with pytest.raises(ValueError, match="Unfrozen research measurements"):
        research_coverage_rows(markers)


def test_no_substitute_reference_for_rare_storage_diseases(payload):
    for row in payload["research_coverage"]["chapters"]:
        if Path(row["origin"]).name.startswith(("05B_", "05C_", "05D_", "09C_")):
            assert row["marker_ids"] and not row["endpoint_ids"]


def test_new_bindings_keep_reference_limits_and_all_chapter_origins(payload):
    from vcg_biomarker.mimic.mechanism_endpoint import analyze_mechanism_endpoint
    from vcg_biomarker.tests.test_mechanism_endpoint import universe

    hypotheses = [h for h in payload["mechanism_hypotheses"] if h.get("research_origins")]
    assert len(hypotheses) == 24
    # Use a genuine newly bound administrative endpoint, not a dummy label.
    h = next(h for h in hypotheses if h["endpoint_id"] == "pulmonary_embolism_icd")
    frame = universe().rename(columns={"marker": h["marker_id"]})
    frame["label_id"] = h["endpoint_id"]
    rows = analyze_mechanism_endpoint(frame, hypotheses=[h], preview=False)
    assert len(rows) == 4
    assert all(row["status"] == "estimable" for row in rows)
    assert all("not_CTPA" in row["reference_target"] for row in rows)
    assert all("CTPA_confirmation" in row["remaining_requirements_json"] for row in rows)
    assert all("13_PULMONARY_EMBOLISM.md" in row["research_origins_json"] for row in rows)
    assert all(not row["mechanism_requirements_fulfilled"] for row in rows)


def test_fixed_family_counts_all_uncomputed_new_questions(payload):
    from vcg_biomarker.mimic.mechanism_endpoint import adjust_mechanism_family
    result = adjust_mechanism_family(pd.DataFrame([{
        "feature_id": "m", "endpoint_id": "e", "split": "train", "model_id": "primary",
        "status": "estimable", "p_value": .001}]), planned_hypotheses=len(payload["mechanism_hypotheses"]))
    assert result.q_value.iloc[0] == pytest.approx(.308)


def test_research_source_is_part_of_validation_provenance():
    from vcg_biomarker.mimic.validation_provenance import validation_implementation_manifest
    assert "mimic/research_coverage.py" in validation_implementation_manifest()["files"]
