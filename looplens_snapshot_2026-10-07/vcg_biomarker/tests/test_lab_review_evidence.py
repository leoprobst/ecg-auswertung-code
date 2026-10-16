import pytest
from vcg_biomarker.mimic.lab_workspace import LabReviewStore, ReportRevision
from vcg_biomarker.mimic.storage import content_hash


@pytest.mark.parametrize('change', ['hash', 'signal', 'criterion', 'analysis'])
def test_report_rejects_inconsistent_review_evidence(tmp_path, change):
    evidence = dict(criterion_id='qrs', analysis_id='a', signal_reference={'case_id': 'case-test', 'source_sha256': 'hash'})
    if change == 'signal':
        evidence['signal_reference']['source_sha256'] = 'other'
    if change == 'criterion':
        evidence['criterion_id'] = 'other'
    if change == 'analysis':
        evidence['analysis_id'] = 'other'
    review = dict(analysis_id='a', evidence_snapshot=evidence,
                  evidence_sha256='bad' if change == 'hash' else content_hash(evidence))
    store = LabReviewStore(tmp_path, 'case-test')
    with pytest.raises(ValueError, match='Review evidence'):
        store.save_report(ReportRevision('case-test', 'hash', 'reviewer', suggestion_review={'qrs': review}))
    assert store.reports() == []


def test_valid_and_legacy_evidence_remain_writable(tmp_path):
    evidence = dict(criterion_id='qrs', analysis_id='a', signal_reference={'case_id': 'case-test', 'source_sha256': 'hash'})
    store = LabReviewStore(tmp_path, 'case-test')
    store.save_report(ReportRevision('case-test', 'hash', 'reviewer', suggestion_review={
        'qrs': dict(analysis_id='a', evidence_snapshot=evidence, evidence_sha256=content_hash(evidence)),
        'legacy': dict(decision='rejected')}))
    assert len(store.reports()) == 1
