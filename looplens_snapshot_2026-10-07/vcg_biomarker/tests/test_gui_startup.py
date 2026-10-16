"""Desktop startup requirements that must precede native context creation."""
import pytest


def test_shared_gl_contexts_are_enabled_before_application_creation(monkeypatch):
    from PyQt6.QtCore import Qt
    from vcg_biomarker import main as desktop

    calls = []

    class StopBeforeWindowCreation(Exception):
        pass

    class ApplicationProbe:
        @staticmethod
        def setAttribute(attribute):
            calls.append(attribute)

        def __init__(self, arguments):
            assert calls == [Qt.ApplicationAttribute.AA_ShareOpenGLContexts]
            raise StopBeforeWindowCreation

    monkeypatch.setattr(desktop, 'QApplication', ApplicationProbe)
    with pytest.raises(StopBeforeWindowCreation):
        desktop.main()
