from vcg_biomarker.gui.lab_endpoint_review import endpoint_reviews


def test_endpoint_card_links_only_recorded_case_evidence():
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.widgets.endpoint_comparison_card import EndpointComparisonCard

    app = QApplication.instance() or QApplication([])
    card = EndpointComparisonCard(dark=True)
    review = endpoint_reviews({'research_findings': [finding(feature_id='vcg_angle')]})[0]
    rows = [finding(feature_id='vcg_angle')]
    card.set_review(review, rows, available_rules=set(), available_markers=set())
    assert not card.rule_button.isEnabled()
    assert not card.marker_button.isEnabled()
    assert 'nicht vorhanden' in card.reference.text()
    card.set_review(review, rows, available_rules={'qrs'}, available_markers={'vcg_angle'})
    assert card.rule_button.isEnabled() and card.marker_button.isEnabled()
    assert 'Wahrscheinlichkeit' not in card.classic.text()
    card.close()


def finding(state='positive', **changes):
    return dict(endpoint_id='ef', endpoint_name_de='LVEF', criterion_id='angle',
                classification=state, applicable=True,
                classical_comparators=[dict(criterion_id='qrs', evaluable=True, positive=False)], **changes)


def test_empty_and_malformed_are_not_negative():
    for payload in ({}, {'research_findings': None}, {'research_findings': [None, {}]}):
        assert endpoint_reviews(payload) == []


def test_grouped_discordance_is_not_diagnosis_or_probability():
    row = finding()
    result = endpoint_reviews({'research_findings': [row, row]})
    assert len(result) == 1
    assert result[0]['comparison'] == 'Kriteriendiskordanz'
    assert result[0]['probability'] is None
    assert result[0]['clinical_claim_allowed'] is False


def test_mixed_not_majority_voted():
    rows = [finding()] * 20 + [finding('negative')]
    result = endpoint_reviews({'research_findings': rows})[0]
    assert result['research_status'] == 'mixed'
    assert result['comparison'] == 'Keine eindeutige Gegenüberstellung'


def test_missing_applicability_and_classical_evidence_stay_unknown():
    row = finding()
    del row['applicable']
    row['classical_comparators'][0]['evaluable'] = None
    result = endpoint_reviews({'research_findings': [row]})[0]
    assert result['research_status'] == result['classical_status'] == 'unknown'


def test_partial_and_negative_are_distinct():
    missing = finding()
    missing['classical_comparators'] = []
    result = endpoint_reviews({'research_findings': [finding('negative'), missing]})[0]
    assert result['classical_status'] == 'partial'


def test_gray_zone_and_endpoint_isolation():
    other = finding('negative')
    other['endpoint_id'] = 'other'
    result = endpoint_reviews({'research_findings': [finding('gray_zone'), other]})
    assert result[0]['research_status'] == 'partial'
    assert result[1]['research_status'] == 'negative'
