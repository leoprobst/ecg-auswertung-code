import pytest
from vcg_biomarker.gui.lab_qt_alert import qt_alert_items


@pytest.mark.parametrize('value,expected', [(500,False),(500.01,True),(None,False),(True,False),(float('nan'),False),('501',False)])
def test_strict_threshold_and_missingness(value, expected):
    row = dict(interval='QTc Fridericia', value_ms=value, status='measurable')
    assert bool(qt_alert_items({'delineation': {'interval_measurements': [row]}})) is expected


def test_duplicates_and_unmeasurable_do_not_create_flags():
    row = dict(interval='QTc Bazett', value_ms=520, status='measurable')
    assert not qt_alert_items({'delineation': {'interval_measurements': [row,row]}})
    row['status'] = 'not_evaluable'
    assert not qt_alert_items({'delineation': {'interval_measurements': [row]}})
