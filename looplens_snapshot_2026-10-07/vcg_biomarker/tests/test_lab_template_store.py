import json
import pytest
from vcg_biomarker.gui.lab_template_store import TemplateMeasurementStore
from vcg_biomarker.gui.lab_representative_beat import cluster_median, manual_template_interval
from vcg_biomarker.tests.test_lab_representative_beat import median_inputs


def setup_store(tmp_path, analysis='a'):
    arrays, inventory = median_inputs()
    template = cluster_median(arrays, inventory, 0)
    reference = {'source_sha256': 'abc'}
    store = TemplateMeasurementStore(tmp_path, template, reference, analysis)
    measurement = manual_template_interval(template, reference, analysis, 0., 40., 'QRS', 'Reviewer', 'II')
    return store, measurement


def test_saved_measurement_reopens_only_for_exact_template(tmp_path):
    store, item = setup_store(tmp_path)
    active = store.add(item)
    reopened, _ = setup_store(tmp_path)
    assert reopened.active() == active
    other, _ = setup_store(tmp_path, 'other-analysis')
    assert other.active() == []


def test_withdrawal_preserves_original_file_and_history(tmp_path):
    store, item = setup_store(tmp_path)
    active = store.add(item)
    original_path = next(store.root.glob('*.json'))
    original_bytes = original_path.read_bytes()
    assert store.withdraw(active[0]['local_event_id'], 'Reviewer', 'Grenze falsch') == []
    assert original_path.read_bytes() == original_bytes
    assert len(store.events()) == 2
    assert setup_store(tmp_path)[0].active() == []


def test_corrupted_history_blocks_load_and_append(tmp_path):
    store, item = setup_store(tmp_path)
    store.add(item)
    path = next(store.root.glob('*.json'))
    content = json.loads(path.read_text(encoding='utf-8'))
    content['event']['measurement']['duration_ms'] = 999
    path.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ValueError, match='Integritätsprüfung'):
        store.active()
    with pytest.raises(ValueError):
        store.add(item)
    assert len(list(store.root.glob('*.json'))) == 1


def test_inconsistent_coordinates_and_missing_withdrawal_reason_rejected(tmp_path):
    store, item = setup_store(tmp_path)
    with pytest.raises(ValueError):
        store.add(dict(item, duration_ms=300))
    active = store.add(item)
    with pytest.raises(ValueError):
        store.withdraw(active[0]['local_event_id'], 'Reviewer', '')
    assert len(store.active()) == 1


def test_history_hashes_full_template_only_once_per_context(tmp_path, monkeypatch):
    from vcg_biomarker.gui import lab_template_store as module
    real_export = module.median_export
    calls = []
    def counted(*args):
        calls.append(1)
        return real_export(*args)
    monkeypatch.setattr(module, 'median_export', counted)
    store, item = setup_store(tmp_path)
    for _ in range(8):
        store.add(item)
    assert len(store.active()) == 8
    assert len(store.events()) == 8
    assert len(calls) == 1


def test_opened_history_owns_immutable_time_grid(tmp_path):
    arrays, inventory = median_inputs()
    template = cluster_median(arrays, inventory, 0)
    reference = {'source_sha256': 'abc'}
    item = manual_template_interval(template, reference, 'a', 0., 40., 'QRS', 'Reviewer', 'II')
    store = TemplateMeasurementStore(tmp_path, template, reference, 'a')
    store.add(item)
    template['time_ms'] += 100
    template['sampling_rate_hz'] = 1000
    assert store.active()[0]['duration_ms'] == 40
    assert not store._times.flags.writeable
