import csv
import pytest
from vcg_biomarker.gui.lab_csv_export import write_csv_atomic


def test_csv_keeps_precision_and_unicode(tmp_path):
    path = tmp_path / 'export.csv'
    write_csv_atomic(path, [{'name': 'Gerät', 'value': 1.123456789}], ['name', 'value'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        assert list(csv.DictReader(stream)) == [{'name': 'Gerät', 'value': '1.123456789'}]
    assert not list(tmp_path.glob('*.tmp'))


def test_partial_generation_does_not_replace_existing_file(tmp_path):
    path = tmp_path / 'export.csv'
    path.write_bytes(b'previous export')
    def rows():
        yield {'value': 1}
        raise OSError('simulated source failure')
    with pytest.raises(OSError):
        write_csv_atomic(path, rows(), ['value'])
    assert path.read_bytes() == b'previous export'
    assert not list(tmp_path.glob('*.tmp'))


def test_text_formula_guard_preserves_numeric_negative_values(tmp_path):
    path = tmp_path / 'export.csv'
    write_csv_atomic(path, [{'author': '=1+1', 'value': -0.15}], ['author', 'value'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        assert list(csv.DictReader(stream)) == [{'author': "'=1+1", 'value': '-0.15'}]
