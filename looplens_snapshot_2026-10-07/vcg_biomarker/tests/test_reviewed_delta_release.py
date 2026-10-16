import json
import pytest
from vcg_biomarker.mimic.scientific_release import (
    REQUIRED_GATE_CHECKS, DELTA_ACCEPTANCE_CHECKS, SCIENTIFIC_RELEASE_VERSION,
    reviewed_delta_acceptance_valid,
)
from vcg_biomarker.mimic.storage import atomic_write_json, file_sha256


def fixture(root):
    release = dict(version=SCIENTIFIC_RELEASE_VERSION, run_id=root.name,
                   run_fingerprint='source', status='approved_for_development_full_run',
                   clinical_approval=False, locked_test_read=False, gates={}, gate_evidence={})
    def bind(name, payload):
        path = root / f'{name}.json'
        atomic_write_json(path, payload)
        return dict(relative_path=path.name, sha256=file_sha256(path))
    for gate, checks in REQUIRED_GATE_CHECKS.items():
        release['gates'][gate] = True
        release['gate_evidence'][gate] = bind(gate, dict(gate=gate, status='PASS',
            run_fingerprint='source', checks={key: True for key in checks}))
    acceptance = dict(version='reviewed-delta-1.0', run_id=root.name, run_fingerprint='source',
        status='PASS', full_run_started=False, checks={key: True for key in DELTA_ACCEPTANCE_CHECKS},
        evidence={key: bind(key, {'test_fixture': True}) for key in
                  ('historical_1000', 'historical_10000', 'change_review', 'current_delta', 'runtime')})
    release['technical_acceptance'] = bind('technical_acceptance', acceptance)
    atomic_write_json(root / 'scientific_release.json', release)
    return release, acceptance


def test_delta_requires_all_scientific_gates_and_bound_proofs(tmp_path):
    fixture(tmp_path)
    assert reviewed_delta_acceptance_valid(tmp_path, 'source')
    assert not reviewed_delta_acceptance_valid(tmp_path, 'other')
    atomic_write_json(tmp_path / 'current_delta.json', {'changed': True})
    assert not reviewed_delta_acceptance_valid(tmp_path)


@pytest.mark.parametrize('key', DELTA_ACCEPTANCE_CHECKS)
def test_missing_review_check_cannot_unlock(tmp_path, key):
    release, acceptance = fixture(tmp_path)
    acceptance['checks'].pop(key)
    path = tmp_path / 'technical_acceptance.json'
    atomic_write_json(path, acceptance)
    release['technical_acceptance']['sha256'] = file_sha256(path)
    atomic_write_json(tmp_path / 'scientific_release.json', release)
    assert not reviewed_delta_acceptance_valid(tmp_path)


def test_gui_accepts_explicit_delta_but_not_invalidated_science(tmp_path):
    from vcg_biomarker.gui.widgets.mimic_project_widget import _full_run_authorized
    fixture(tmp_path)
    assert _full_run_authorized(tmp_path)
    atomic_write_json(tmp_path / 'software_suite.json', {})
    assert not _full_run_authorized(tmp_path)


def test_reviewed_delta_omits_only_pilot_stages_on_launch_and_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vcg_biomarker.mimic import workflow, scientific_release, capacity_planning, services
    from vcg_biomarker.mimic.config import MimicProjectConfig
    config = MimicProjectConfig.discover(database_root=tmp_path/'db', work_root=tmp_path/'work')
    config.ensure_directories()
    from vcg_biomarker.mimic.marker_freeze import freeze_marker_inventory
    freeze_marker_inventory(config, run_id='test-current', finalize_inputs=True)
    monkeypatch.setattr(workflow, '_assert_generated_run_is_current', lambda *a: 'source')
    monkeypatch.setattr(workflow, '_assert_run_preflight_gates', lambda *a: None)
    monkeypatch.setattr(scientific_release, 'scientific_release_valid', lambda *a: True)
    monkeypatch.setattr(scientific_release, 'reviewed_delta_acceptance_valid', lambda *a: True)
    monkeypatch.setattr(capacity_planning, 'require_capacity_at_start', lambda *a: None)
    calls = []
    def stage(config, name, arguments, **kwargs):
        calls.append((name, arguments.get('max_records')))
        return SimpleNamespace(status='complete', row_counts={'rows': 1})
    monkeypatch.setattr(services, 'run_service_stage', stage)
    stages = ('inventory', *workflow.GUIDED_PILOT_1K_STAGES,
              *workflow.GUIDED_PILOT_10K_STAGES, 'pilot-go-no-go', 'extract-features')
    args = dict(preset_id='exhaustive_research', run_id='test-current',
                stop_after_pilots=False, include_locked_test=False,
                study_spec={'stage_ids': stages})
    result = workflow.run_guided_study(config, args)
    assert result.completed_stages == ('inventory', 'extract-features')
    assert calls == [('inventory', None), ('extract-features', None)]
    calls.clear()
    assert workflow.run_guided_study(config, args).completed_stages == result.completed_stages
    assert not calls
