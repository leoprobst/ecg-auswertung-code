import json
import numpy as np
import pytest

from vcg_biomarker.mimic.reviewer_export import export_reviewer_packets
from vcg_biomarker.mimic.reviewer_packet import ReviewerPacket
from vcg_biomarker.mimic.reviewer_responses import save_review
from vcg_biomarker.mimic.reviewer_import import import_review_responses, validate_export_packets
from vcg_biomarker.tests.test_reviewer_export import selection


def issued(tmp_path):
    export_reviewer_packets(selection(),tmp_path/"issued",signal_loader=lambda _: (np.ones((100,12)),250.))
    return [ReviewerPacket(tmp_path/"issued"/"reviewer_packets"/f"slot-{i}") for i in (1,2)]


def answer(packet,root,reviewer="reader-one"):
    return save_review(packet,case_id=packet.cases.review_case_id.iloc[0],verdict="uncertain",
                       reviewer_id=reviewer,response_root=root)


def test_roundtrip_and_missing_response_denominator(tmp_path):
    packets = issued(tmp_path)
    path = answer(packets[0],tmp_path/"responses")
    before = path.read_bytes()
    rows, coverage = import_review_responses(packets,[path])
    assert len(rows) == 1
    assert len(coverage) == 4
    assert coverage.status.eq("received").sum() == 1
    assert coverage.status.eq("missing_response").sum() == 3
    assert not rows.clinical_claim_allowed.any()
    assert path.read_bytes() == before
    empty,missing = import_review_responses(packets,[])
    assert empty.empty and missing.status.eq("missing_response").all()


@pytest.mark.parametrize("field,value",[("signal_sha256","0"*64),("packet_sha256","0"*64),
    ("endpoint_id","wrong"),("reviewer_slot",2),("reviewer_verdict","invented"),
    ("clinical_claim_allowed",True),("recorded_at","2026-01-01"),("reviewer_id","")])
def test_mismatched_or_invalid_response_rejected(tmp_path,field,value):
    packets = issued(tmp_path)
    path = answer(packets[0],tmp_path/"responses")
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        import_review_responses(packets,[path])


def test_duplicate_answers_and_same_person_in_two_slots_rejected(tmp_path):
    packets = issued(tmp_path)
    first = answer(packets[0],tmp_path/"responses")
    with pytest.raises(ValueError,match="Duplicate response"):
        import_review_responses(packets,[first,first])
    second = answer(packets[1],tmp_path/"responses",reviewer="READER-ONE")
    with pytest.raises(ValueError,match="Same reviewer"):
        import_review_responses(packets,[first,second])


def test_same_size_packets_from_different_exports_are_rejected(tmp_path):
    packets = issued(tmp_path)
    other = issued(tmp_path/"other")
    manifest = json.loads((tmp_path/"issued"/"export_complete.json").read_text())
    validate_export_packets(manifest, packets)
    with pytest.raises(ValueError, match="original export"):
        validate_export_packets(manifest, [packets[0], other[1]])
    with pytest.raises(ValueError, match="disagree on cases"):
        import_review_responses([packets[0], other[1]], [])


@pytest.mark.parametrize("change", ["endpoint", "signal"])
def test_same_case_id_cannot_mask_different_question_or_signal(tmp_path, change):
    packets = issued(tmp_path)
    case = packets[1].cases.review_case_id.iloc[0]
    if change == "endpoint":
        packets[1].cases.loc[packets[1].cases.review_case_id.eq(case), "endpoint_id"] = "different_question"
    else:
        packets[1].hashes[f"{case}.npz"] = "0"*64
    with pytest.raises(ValueError, match="disagree|different signals"):
        import_review_responses(packets, [])


@pytest.mark.parametrize("field,value", [("n_cases", True), ("n_reviewers", 1),
    ("signal_sha256", {}), ("clinical_claim_allowed", True)])
def test_export_manifest_cannot_assert_wrong_inventory(tmp_path, field, value):
    packets = issued(tmp_path)
    manifest = json.loads((tmp_path/"issued"/"export_complete.json").read_text())
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_export_packets(manifest, packets)
