from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from vcg_biomarker.gui.practice_cases import choose_practice_case, practice_solution
from vcg_biomarker.gui.wpw_localization import (
    EASYWPWInputs, easy_wpw_localize, propose_easy_wpw_inputs,
)


def test_practice_uses_annotated_independent_training_source_not_ptbxl(tmp_path):
    locked = tmp_path / 'ptb-xl' / 'fold10'
    locked.mkdir(parents=True)
    (locked / 'locked.hea').write_text(
        'locked 12 500 5000\n# Dx: 164889003\n', encoding='utf-8'
    )
    group = tmp_path / 'georgia' / 'g1'
    group.mkdir(parents=True)
    header = group / 'demo.hea'
    header.write_text('demo 12 500 5000\n# Age: 42\n# Sex: F\n# Dx: 164889003,999999\n', encoding='utf-8')
    case = choose_practice_case(tmp_path, rng=random.Random(3))
    assert case.header == header.resolve()
    answer = practice_solution(case)
    assert answer['ecg_annotation_labels'] == ['atrial_fibrillation']
    assert answer['unmapped_snomed_codes'] == ['999999']
    assert answer['reference_kind'] == 'expert_ecg_annotation_not_independent_disease_reference'
    with pytest.raises(ValueError, match='Keine geeignete'):
        choose_practice_case(tmp_path, excluded_headers=frozenset({header.resolve()}), attempts=3)


@pytest.mark.parametrize('v1,transition,delta,expected', [
    ('positive', 'unknown', 'aVL', 'Mitralring posteroseptal'),
    ('positive', 'unknown', 'II', 'Mitralring posterolateral'),
    ('positive', 'unknown', 'III', 'Mitralring anterolateral'),
    ('negative', 'at_or_before_v3', 'II', 'Trikuspidalring anteroseptal'),
    ('isoelectric', 'at_or_before_v3', 'aVR', 'Trikuspidalring posteroseptal'),
    ('negative', 'after_v3', 'aVL', 'Trikuspidalring posterolateral'),
    ('negative', 'after_v3', 'II', 'Trikuspidalring anterolateral'),
])
def test_easy_wpw_published_decision_tree(v1, transition, delta, expected):
    inputs = EASYWPWInputs(True, True, True, True, v1, delta, transition)
    result = easy_wpw_localize(inputs)
    assert result.status == 'candidate' and result.region == expected
    assert result.probability is None
    assert result.source_url.startswith('https://')


def test_easy_wpw_abstains_without_required_population_and_inputs():
    inputs = EASYWPWInputs(False, True, True, True, 'positive', 'aVL')
    assert easy_wpw_localize(inputs).status == 'abstained'
    assert easy_wpw_localize(replace(inputs, manifest_preexcitation_confirmed=True,
                                     structural_disease_excluded=False)).status == 'abstained'
    assert easy_wpw_localize(replace(inputs, manifest_preexcitation_confirmed=True,
                                     v1_polarity='negative')).status == 'abstained'


def test_delta_measurement_preview_rejects_unreliable_boundaries_and_ties():
    signal = np.zeros((500, 12))
    with pytest.raises(ValueError, match='QRS-Grenzen'):
        propose_easy_wpw_inputs(signal, 500, 10, 70)
    signal[110:120, 1] = .25
    signal[110:120, 2] = .04
    signal[110:120, 3] = .03
    signal[110:120, 4] = .02
    signal[100:180, 6] = .5
    signal[100:180, 7] = .4
    signal[100:180, 8] = .3
    preview = propose_easy_wpw_inputs(signal, 500, 100, 180)
    assert preview['most_positive_delta_lead_proposal'] == 'II'
    assert preview['status'].startswith('unvalidated_')
    signal[110:120, 2] = .245
    assert propose_easy_wpw_inputs(signal, 500, 100, 180)['most_positive_delta_lead_proposal'] is None
