import json
import pytest
from vcg_biomarker.mimic.reviewer_packet import ReviewerPacket
from vcg_biomarker.mimic.reviewer_responses import save_review
from vcg_biomarker.tests.test_reviewer_packet import packet


def test_response_bound_to_signal_and_never_overwritten(tmp_path):
    model = ReviewerPacket(packet(tmp_path))
    case = model.cases.review_case_id.iloc[0]
    path = save_review(model,case_id=case,verdict="uncertain",reviewer_id="reader-a",response_root=tmp_path/"answers")
    original = path.read_bytes()
    record = json.loads(original)
    assert record["signal_sha256"] == model.hashes[f"{case}.npz"]
    assert record["reviewer_verdict"] == "uncertain"
    assert not record["clinical_claim_allowed"]
    with pytest.raises(FileExistsError):
        save_review(model,case_id=case,verdict="positive",reviewer_id="reader-a",response_root=tmp_path/"answers")
    assert path.read_bytes() == original


def test_invalid_verdict_and_in_packet_output_rejected(tmp_path):
    model = ReviewerPacket(packet(tmp_path))
    case = model.cases.review_case_id.iloc[0]
    with pytest.raises(ValueError,match="Explicit"):
        save_review(model,case_id=case,verdict=None,reviewer_id="a",response_root=tmp_path/"answers")
    with pytest.raises(ValueError,match="separate"):
        save_review(model,case_id=case,verdict="negative",reviewer_id="a",response_root=model.folder/"answers")
    assert not (model.folder/"answers").exists()
