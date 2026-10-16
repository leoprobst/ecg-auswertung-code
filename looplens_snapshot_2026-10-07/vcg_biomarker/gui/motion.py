"""Shared, user-controlled motion preference for presentation-only effects.

This preference never changes signal processing, measurements, or exports.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6 import sip


class MotionPolicy(QObject):
    changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("VCG", "VCGBiomarker")
        self._reduced = self._settings.value("display/reduced_motion", False, type=bool)

    @property
    def reduced(self) -> bool:
        return self._reduced

    def set_reduced(self, value: bool) -> None:
        value = bool(value)
        if value == self._reduced:
            return
        self._reduced = value
        self._settings.setValue("display/reduced_motion", value)
        self.changed.emit(value)


motion_policy = MotionPolicy()


def current_motion_policy() -> MotionPolicy:
    """Return a live policy after a test/application QApplication was replaced."""
    global motion_policy
    if sip.isdeleted(motion_policy):
        motion_policy = MotionPolicy()
    return motion_policy
