from __future__ import annotations

import math
from typing import Callable, Optional

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QWidget


_BACKGROUND = QColor("#0f2124")
_TRACE = QColor("#178292")
_TRACE_GLOW = QColor(23, 130, 146, 70)
_TRACK = QColor(23, 130, 146, 45)
_TEXT = QColor("#eaf4f5")
_SUBTEXT = QColor("#8fb9bd")

_CYCLE_WIDTH = 150.0
_SCROLL_SPEED = 90.0
_TICK_MS = 33


def _ecg_shape(t: float) -> float:
    """Stylized single PQRST complex over t in [0, 1); flat baseline
    everywhere else, matching a classic monitor sweep trace."""
    if 0.08 <= t < 0.14:
        return 0.10 * math.sin((t - 0.08) / 0.06 * math.pi)
    if 0.20 <= t < 0.235:
        return -0.15 * (t - 0.20) / 0.035
    if 0.235 <= t < 0.285:
        return -0.15 + 1.45 * (t - 0.235) / 0.05
    if 0.285 <= t < 0.335:
        return 1.30 - 1.65 * (t - 0.285) / 0.05
    if 0.335 <= t < 0.375:
        return -0.35 + 0.35 * (t - 0.335) / 0.04
    if 0.46 <= t < 0.64:
        return 0.26 * math.sin((t - 0.46) / 0.18 * math.pi)
    return 0.0


class AnimatedSplashScreen(QWidget):
    """Themed startup splash -- a scrolling ECG pulse trace with a real
    progress readout, shown while MainWindow/MimicProjectWidget build.

    Deliberately MIMIC-agnostic: callers drive it purely through
    ``set_progress(message, fraction)``, so it stays reusable and never
    needs to know anything about the widgets it is covering for.
    """

    def __init__(self, *, width: int = 560, height: int = 320) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SplashScreen,
        )
        self.setFixedSize(width, height)
        self._phase = 0.0
        self._message = "Starte..."
        self._fraction = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            geometry = screen.availableGeometry()
            self.move(
                geometry.center().x() - width // 2,
                geometry.center().y() - height // 2,
            )

    def _tick(self) -> None:
        self._phase += _SCROLL_SPEED * (_TICK_MS / 1000.0)
        self.update()

    def set_progress(self, message: str, fraction: float) -> None:
        """Report real startup progress. `fraction` in [0, 1]."""
        self._message = message
        self._fraction = max(0.0, min(1.0, fraction))
        self.update()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        painter.fillRect(rect, _BACKGROUND)

        painter.setPen(_TEXT)
        painter.setFont(QFont("Segoe UI", 19, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(0, 34, rect.width(), 36),
            Qt.AlignmentFlag.AlignHCenter,
            "VCG Feature Scanner",
        )
        painter.setPen(_SUBTEXT)
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRectF(0, 66, rect.width(), 20),
            Qt.AlignmentFlag.AlignHCenter,
            "MIMIC-IV / PTB-XL Forschungsplattform",
        )

        self._draw_pulse(painter, QRectF(36, 108, rect.width() - 72, 92))

        bar_rect = QRectF(60, rect.height() - 68, rect.width() - 120, 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_TRACK)
        painter.drawRoundedRect(bar_rect, 3, 3)
        filled = QRectF(bar_rect)
        filled.setWidth(bar_rect.width() * self._fraction)
        painter.setBrush(_TRACE)
        painter.drawRoundedRect(filled, 3, 3)

        painter.setPen(_SUBTEXT)
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRectF(0, rect.height() - 46, rect.width(), 22),
            Qt.AlignmentFlag.AlignHCenter,
            self._message,
        )
        painter.end()

    def _draw_pulse(self, painter: QPainter, rect: QRectF) -> None:
        path = QPainterPath()
        width = rect.width()
        mid_y = rect.top() + rect.height() / 2
        samples = 220
        path.moveTo(rect.left(), mid_y)
        for index in range(samples + 1):
            x = rect.left() + width * index / samples
            local = ((x + self._phase) % _CYCLE_WIDTH) / _CYCLE_WIDTH
            y = mid_y - _ecg_shape(local) * (rect.height() * 0.46)
            path.lineTo(x, y)
        painter.setPen(QPen(_TRACE_GLOW, 6))
        painter.drawPath(path)
        painter.setPen(QPen(_TRACE, 2.2))
        painter.drawPath(path)


ProgressCallback = Callable[[str, float], None]


def null_progress_callback(message: str, fraction: float) -> None:
    return None


def resolve_progress_callback(
    callback: Optional[ProgressCallback],
) -> ProgressCallback:
    return callback if callback is not None else null_progress_callback
