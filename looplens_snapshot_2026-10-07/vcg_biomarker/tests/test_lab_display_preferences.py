import json
import pytest
from vcg_biomarker.gui.lab_display_preferences import decode_display_defaults, encode_display_defaults


def defaults():
    return dict(version='1.0.0', layout='6x2', lead='aVR', order='cabrera',
                standard_avr='native', speed='25', gain='10', fit=True)


def test_round_trip_and_missing_settings():
    assert decode_display_defaults(None) is None
    assert decode_display_defaults(encode_display_defaults(defaults())) == defaults()


@pytest.mark.parametrize('key,value', [('version', '2.0'), ('fit', 'false'), ('fit', 1),
    ('speed', '100'), ('lead', '-aVR'), ('gain', 10), ('layout', 'unknown'), ('patient_id', 'secret')])
def test_invalid_or_non_display_data_is_rejected(key, value):
    settings = defaults()
    settings[key] = value
    with pytest.raises(ValueError):
        decode_display_defaults(json.dumps(settings))


@pytest.mark.parametrize('raw', ['[]', 'null', '{', 'false', '{}'])
def test_bad_documents_rejected(raw):
    with pytest.raises(ValueError):
        decode_display_defaults(raw)


@pytest.fixture
def isolated_widget_settings(tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.widgets import expert_ecg_lab as module
    app = QApplication.instance() or QApplication([])
    class LocalSettings(QSettings):
        def __init__(self, *args):
            super().__init__(str(tmp_path/'display.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(module, 'QSettings', LocalSettings)
    yield module, LocalSettings, tmp_path/'display.ini'
    app.processEvents()


def test_widget_reopens_display_defaults_without_clinical_state(isolated_widget_settings):
    module, settings_type, path = isolated_widget_settings
    first = module.ExpertECGLab()
    second = None
    try:
        initial_calibration = first.pixels_per_mm.value()
        first.layout_choice.setCurrentIndex(1)
        first.lead_order.setCurrentIndex(1)
        first.lead_choice.setCurrentText('aVR')
        first.speed.setCurrentText('50')
        first.gain.setCurrentText('20')
        first.fit.setChecked(False)
        first.pixels_per_mm.setValue(7)
        first.blind.setChecked(True)
        first.mode.setCurrentIndex(1)
        first._save_display_defaults()
        second = module.ExpertECGLab()
        assert second.layout_choice.currentIndex() == 1
        assert second.lead_order.currentIndex() == 1
        assert second.avr_polarity.currentIndex() == 1
        assert second.lead_choice.currentText() == 'aVR'
        assert second.speed.currentText() == '50'
        assert second.gain.currentText() == '20'
        assert not second.fit.isChecked()
        assert second.pixels_per_mm.value() == initial_calibration
        assert second.mode.currentIndex() == 0
        assert not second.blind.isChecked()
        assert second.case is None and second.signal is None
        second.lead_order.setCurrentIndex(0)
        assert second.avr_polarity.currentIndex() == 0
    finally:
        first.close()
        if second is not None:
            second.close()


def test_corrupt_defaults_never_overwritten(isolated_widget_settings):
    from vcg_biomarker.gui.lab_display_preferences import KEY
    module, settings_type, path = isolated_widget_settings
    settings = settings_type()
    settings.setValue(KEY, '{invalid')
    settings.sync()
    before = path.read_bytes()
    widget = module.ExpertECGLab()
    try:
        assert not widget.save_display_defaults.isEnabled()
        widget._save_display_defaults()
        assert path.read_bytes() == before
        assert widget.layout_choice.currentIndex() == 0
    finally:
        widget.close()
