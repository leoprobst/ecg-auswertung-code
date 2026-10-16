from types import SimpleNamespace
import numpy as np
from vcg_biomarker.mimic import classical
from vcg_biomarker.core.data_loader import LEAD_NAMES_12


def test_rbbb_requires_a_wide_lateral_s(monkeypatch):
    """Audit S5 (AHA/ACCF/HRS 2009): the lateral S must be wide, not only deep."""
    monkeypatch.setattr(classical, '_terminal_exceeds_initial_r', lambda *args: True)
    beat = SimpleNamespace(qrs_duration_ms=lambda fs: 130,
                           r_onset=SimpleNamespace(sample=0), r_offset=SimpleNamespace(sample=99))
    qrs = {lead: dict(peak_to_peak=1.2, r_height=1., s_depth=.2) for lead in LEAD_NAMES_12}
    q = {lead: dict(duration_ms=0, depth_mv=0) for lead in LEAD_NAMES_12}
    outcomes = []
    for s_samples in (10, 30):          # 20 ms against 60 ms at 500 Hz
        signal = np.zeros((100, 12))
        for lead in (0, 11):            # I and V6: R of 20 ms, then S
            signal[5:15, lead] = 1.0
            signal[15:15 + s_samples, lead] = -.2
        result = classical.ClassicalECGCriteria()._conduction_criteria(
            signal, qrs, q, beat, np.zeros(12), 500,
            paced=False, qrs_evaluable=True, p_evaluable=False)['rbbb_pattern']
        assert 'duration >40 ms or longer than R' in result.definition
        assert result.exploratory is True
        outcomes.append(result.positive)
    assert outcomes == [False, True]


def test_historical_rbbb_scope_explained_in_gui():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_rule_details
    text = classical_rule_details('rbbb_pattern', {'definition': 'wide terminal S'}, 'old')
    assert 'nicht die terminale S-Dauer' in text
    assert 'Keine vollständige RSB-Diagnose' in text
