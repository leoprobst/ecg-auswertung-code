import json
from vcg_biomarker.gui.lab_beat_annotations import read_beat_labels


def test_versions_and_corrupt_file_are_isolated(tmp_path):
    reference = dict(sampling_rate_hz=250, sample_count=500, source_sha256='a')
    base = dict(signal=reference, analysis_id='one', label='Sinusschlag', created_at='2026-09-22',
                schema_version='1.0.0', r_time_seconds=.1)
    (tmp_path/'old.json').write_text(json.dumps(base), encoding='utf-8')
    modern = dict(base, schema_version='1.1.0', native_r_sample=25, author='Reviewer',
                  label='sonstige', created_at='2026-09-23')
    (tmp_path/'new.json').write_text(json.dumps(modern), encoding='utf-8')
    (tmp_path/'bad.json').write_text('{', encoding='utf-8')
    labels, errors = read_beat_labels(tmp_path, reference, 'one')
    assert labels == {25: 'sonstige'} and len(errors) == 1
    assert read_beat_labels(tmp_path, reference, 'other')[0] == {}


def test_modern_author_and_sample_are_required(tmp_path):
    reference = dict(sampling_rate_hz=250, sample_count=500)
    base = dict(signal=reference, analysis_id='one', label='Artefakt', created_at='2026-09-22',
                schema_version='1.1.0', native_r_sample=True, author='Reviewer')
    (tmp_path/'bool.json').write_text(json.dumps(base), encoding='utf-8')
    (tmp_path/'no_author.json').write_text(json.dumps(dict(base, native_r_sample=25, author='')), encoding='utf-8')
    labels, errors = read_beat_labels(tmp_path, reference, 'one')
    assert not labels and len(errors) == 2
