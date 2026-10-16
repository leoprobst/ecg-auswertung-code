import hashlib
import importlib.util
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import broad_axis_stage as current
from vcg_biomarker.mimic import broad_axis_adjustment as adjustment


def previous(name, filename):
    spec = importlib.util.spec_from_file_location('vcg_biomarker.mimic.' + name, Path(__file__).parent / 'fixtures' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['binary', 'continuous'])
def test_prepared_fits_exactly_match_pinned_source(kind):
    old = previous('_old_adjustment', 'broad_adjustment_before_preparation.py')
    from vcg_biomarker.tests.test_broad_axis_adjustment import _frame
    frame = _frame(n=400).sample(frac=1, random_state=4)
    frame['rhythm_context'] = np.where(frame.subject_id % 2, 'sinus', 'other')
    frame['qrs_duration_ms'] = 90 + frame.subject_id % 30
    frame.loc[frame.subject_id % 17 == 0, 'marker_value'] = np.nan
    frame.loc[frame.subject_id % 19 == 0, 'age'] = np.nan
    frame = pd.concat([frame, frame.iloc[:10].assign(ecg_time=pd.Timestamp('2030-01-01'))], ignore_index=True)
    if kind == 'continuous':
        frame['target_value'] = np.random.default_rng(6).normal(size=len(frame))
    prepared = frame.sort_values(['subject_id', 'ecg_time', 'study_id'], kind='mergesort').drop_duplicates('subject_id', keep='first')
    for covariates in [(), adjustment.PRIMARY_COVARIATES, adjustment.SENSITIVITY_COVARIATES]:
        expected = old.fit_adjusted_association(frame, target_kind=kind, covariates=covariates)
        actual = adjustment.fit_adjusted_association(prepared, target_kind=kind, covariates=covariates, _prepared=True)
        pd.testing.assert_frame_equal(pd.DataFrame([actual]), pd.DataFrame([expected]), check_exact=True)


@pytest.mark.parametrize('kind', ['binary', 'continuous'])
def test_actual_packet_resume_preserves_results_and_skips_saved_queries(tmp_path, monkeypatch, kind):
    old = previous('_old_stage', 'broad_axis_before_adjustment_packets.py')
    rng = np.random.default_rng(22)
    n = 320
    master = pd.DataFrame(dict(study_id=np.arange(n), subject_id=np.arange(n),
        ecg_time=pd.date_range('2020-01-01', periods=n, freq='min'),
        split=np.where(np.arange(n)<160, 'train', 'validation'), age=rng.normal(60,10,n),
        sex=np.where(np.arange(n)%2,'F','M'), machine_heart_rate_bpm=rng.normal(70,5,n),
        triage_heart_rate=np.nan, icustay_id=np.nan, ed_stay_id=np.nan,
        hadm_id=np.arange(n), machine_qrs_duration_ms=100.0))
    markers = ['m'+str(i) for i in range(17)] + ['qrs_duration_ms']
    features = pd.DataFrame(dict(study_id=np.arange(n), signal_quality_score=rng.uniform(.7,1,n),
        rhythm_context=np.where(np.arange(n)%3,'sinus','other'), qrs_duration_ms=rng.normal(100,10,n)))
    for marker in markers:
        features[marker] = rng.normal(size=n)
    features.loc[::11, 'm0'] = np.nan
    features.loc[::13, 'qrs_duration_ms'] = np.nan
    presence = pd.DataFrame(dict(study_id=np.arange(n)[np.arange(n)%2==0], target_id='a', temporal_relation='undated'))
    axis = 'procedure_code' if kind=='binary' else 'laboratory_item'
    if kind=='continuous':
        presence = pd.DataFrame(dict(study_id=np.arange(n), subject_id=np.arange(n),
            target_id='a', temporal_relation='undated', value=rng.normal(size=n)))
    replicated = pd.DataFrame([dict(axis=axis, target_id='a', temporal_relation='undated', target_label='A', feature_id=m) for m in markers])
    queries = []
    class Observed:
        def __init__(self, connection): self.connection=connection
        def execute(self, query):
            if '__vcg_adjustment_marker_' in query: queries.append(query)
            return self.connection.execute(query)
    @contextmanager
    def isolated(config):
        connection = duckdb.connect()
        connection.register('features', features)
        connection.register('master', master)
        connection.register('presence', presence)
        yield Observed(connection)
        connection.close()
    from vcg_biomarker.mimic import storage
    monkeypatch.setattr(storage, 'isolated_analysis_connection', isolated)
    for module in (old, current):
        monkeypatch.setattr(module, '_parquet', lambda path: "read_parquet('"+str(path)+"')" if str(path).endswith('development_basis.parquet') else 'master')
        monkeypatch.setattr(module, '_feature_parquet', lambda value: 'features')
        monkeypatch.setattr(module, 'available_markers', lambda connection, glob, requested: requested)
        monkeypatch.setattr(module, 'binary_axis_values_sql', lambda *a, **k: 'SELECT * FROM presence')
        monkeypatch.setattr(module, 'continuous_axis_values_sql', lambda *a, **k: 'SELECT * FROM presence')
    kwargs = dict(config=SimpleNamespace(master_root=tmp_path), replicated=replicated,
        feature_glob='features', sources={axis:dict(available=True, relation='presence', entity_column='target_id', counts_sql=lambda: None)})
    expected = old._adjust_replicated_hits(**kwargs)
    queries.clear()
    def pause(message):
        if 'Marker 8/18 gesichert' in message: raise RuntimeError('checkpoint pause')
    with pytest.raises(RuntimeError, match='checkpoint pause'):
        current._adjust_replicated_hits(**kwargs, checkpoint_root=tmp_path/'packets', checkpoint_input_hash='same', progress=pause)
    saved = next((tmp_path/'packets').rglob('part-000000.parquet'))
    before = (saved.stat().st_mtime_ns, hashlib.sha256(saved.read_bytes()).hexdigest())
    assert len(queries)==2  # one streamed projection plus one unsaved packet
    queries.clear()
    actual = current._adjust_replicated_hits(**kwargs, checkpoint_root=tmp_path/'packets', checkpoint_input_hash='same')
    assert len(queries)==2
    assert before == (saved.stat().st_mtime_ns, hashlib.sha256(saved.read_bytes()).hexdigest())
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    queries.clear()
    actual = current._adjust_replicated_hits(**kwargs, checkpoint_root=tmp_path/'packets', checkpoint_input_hash='same')
    assert not queries
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    queries.clear()
    helper = current._adjust_replicated_hits(**kwargs, checkpoint_root=tmp_path/'packets', checkpoint_input_hash='same', helper_shard=(0, 1))
    assert helper.empty and not queries
    assert before == (saved.stat().st_mtime_ns, hashlib.sha256(saved.read_bytes()).hexdigest())
