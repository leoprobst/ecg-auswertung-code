"""Central light/dark theme for the MIMIC workspace GUI.

Cosmetic-only design system: semantic color tokens (shared vocabulary
for the central QSS stylesheet, pyqtgraph chart theming, and the
scattered per-widget status colors), plus small QSS-generation helpers.

The dark palette deliberately reuses the startup splash screen's exact
colors (`splash_screen.py`'s `_BACKGROUND`/`_TRACE`/`_TEXT`/`_SUBTEXT`)
so the running app visually continues the loading screen.

QSS-safety note: `generate_stylesheet()` is built from many small
`_rule(selector, body)` calls rather than one big `.format()`/f-string
over a brace-heavy multi-line block -- `_rule()` is the only place that
escapes literal `{`/`}`, so a stray brace deep in a large composed QSS
string can't silently corrupt the sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
import pyqtgraph as pg


@dataclass(frozen=True)
class ThemeTokens:
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_strong: str
    success: str
    warning: str
    warning_surface: str
    danger: str
    danger_surface: str
    danger_muted: str
    bg: str
    surface: str
    surface_alt: str
    text: str
    text_strong: str
    text_muted: str
    border: str
    border_strong: str


LIGHT = ThemeTokens(
    accent="#178292",
    accent_hover="#e8f2f3",
    accent_pressed="#d7e8ea",
    accent_strong="#0f6b78",
    success="#176b42",
    warning="#9a6700",
    warning_surface="#fdf3e7",
    danger="#b42318",
    danger_surface="#fff3f4",
    danger_muted="#8a3b47",
    bg="#f4f7f8",
    surface="#ffffff",
    surface_alt="#eef5f6",
    text="#1d2a30",
    text_strong="#182226",
    text_muted="#52616b",
    border="#cfd8dc",
    border_strong="#b9c6cb",
)

# Reuses splash_screen.py's exact _BACKGROUND (#0f2124), _TEXT (#eaf4f5)
# and _SUBTEXT (#8fb9bd) so the running app continues the loading screen.
DARK = ThemeTokens(
    accent="#178292",
    accent_hover="#1f97a4",
    accent_pressed="#13707c",
    accent_strong="#1f97a4",
    success="#2fa365",
    warning="#e0a53d",
    warning_surface="#3a2f12",
    danger="#e0605a",
    danger_surface="#3a1f1e",
    danger_muted="#c98891",
    bg="#0f2124",
    surface="#16292c",
    surface_alt="#1b3236",
    text="#eaf4f5",
    text_strong="#ffffff",
    text_muted="#8fb9bd",
    border="#25454a",
    border_strong="#2f5a60",
)

RADIUS = 8
RADIUS_SMALL = 6
RADIUS_CARD = 10


def tokens_for(dark: bool) -> ThemeTokens:
    return DARK if dark else LIGHT


def _rule(selector: str, body: str) -> str:
    return f"{selector} {{\n    {body}\n}}\n"


def generate_stylesheet(dark: bool) -> str:
    t = tokens_for(dark)
    parts = [
        _rule(
            "QWidget",
            f"font-size: 12px; color: {t.text}; background-color: {t.bg};",
        ),
        _rule("QLabel", "background-color: transparent;"),
        _rule(
            "QGroupBox",
            f"border: 1px solid {t.border}; border-top: 2px solid {t.accent}; "
            f"border-radius: {RADIUS_CARD}px; "
            f"margin-top: 10px; padding-top: 10px; font-weight: 600; "
            f"background-color: {t.surface}; color: {t.text};",
        ),
        _rule(
            "QGroupBox::title",
            f"left: 10px; padding: 0 5px; background-color: {t.surface}; "
            f"color: {t.text};",
        ),
        _rule(
            "QGroupBox::indicator",
            f"width: 15px; height: 15px; border: 1px solid {t.border_strong}; "
            f"border-radius: {RADIUS_SMALL}px; background-color: {t.surface};",
        ),
        _rule(
            "QGroupBox::indicator:checked",
            f"background-color: {t.accent}; border-color: {t.accent};",
        ),
        _rule(
            "QPushButton",
            f"min-height: 28px; padding: 4px 12px; "
            f"border: 1px solid {t.border_strong}; border-radius: {RADIUS}px; "
            f"background-color: {t.surface}; color: {t.text};",
        ),
        _rule(
            "QPushButton:hover",
            f"background-color: {t.accent_hover}; border-color: {t.accent};",
        ),
        _rule("QPushButton:pressed", f"background-color: {t.accent_pressed};"),
        _rule(
            "QPushButton:disabled",
            f"color: {t.text_muted}; background-color: {t.surface_alt};",
        ),
        _rule(
            "QLineEdit, QComboBox, QSpinBox, QTextEdit",
            f"color: {t.text}; background-color: {t.surface}; "
            f"border: 1px solid {t.border_strong}; border-radius: {RADIUS}px; "
            f"min-height: 24px; padding: 2px 6px;",
        ),
        _rule(
            "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus",
            f"border: 1px solid {t.accent};",
        ),
        _rule("QComboBox::drop-down", "border: none; width: 22px;"),
        _rule(
            "QComboBox QAbstractItemView",
            f"color: {t.text}; background-color: {t.surface}; "
            f"selection-color: #ffffff; selection-background-color: "
            f"{t.accent_strong}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "QTableView, QTableWidget",
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px; "
            f"gridline-color: {t.border}; color: {t.text}; "
            f"background-color: {t.surface}; "
            f"alternate-background-color: {t.surface_alt}; "
            f"selection-color: #ffffff; "
            f"selection-background-color: {t.accent_strong};",
        ),
        _rule(
            "QHeaderView::section, QTableCornerButton::section",
            f"color: {t.text}; background-color: {t.surface_alt}; border: 0; "
            f"border-right: 1px solid {t.border}; "
            f"border-bottom: 1px solid {t.border}; padding: 5px;",
        ),
        _rule(
            "QTabWidget::pane",
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px; "
            f"background-color: {t.surface};",
        ),
        _rule(
            "QTabBar::tab",
            f"color: {t.text_muted}; background-color: {t.surface_alt}; "
            f"border: 1px solid {t.border}; border-top: 2px solid transparent; "
            f"border-top-left-radius: {RADIUS_SMALL}px; "
            f"border-top-right-radius: {RADIUS_SMALL}px; "
            f"padding: 7px 14px; margin-right: 2px;",
        ),
        _rule(
            "QTabBar::tab:selected",
            f"color: {t.accent_strong}; background-color: {t.surface}; "
            f"border-top: 2px solid {t.accent}; "
            f"border-bottom-color: {t.surface}; font-weight: 600;",
        ),
        _rule(
            "QTabBar::tab:hover:!selected",
            f"color: {t.text}; background-color: {t.accent_hover};",
        ),
        _rule(
            "QProgressBar",
            f"min-height: 16px; color: {t.text}; "
            f"background-color: {t.surface_alt}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_SMALL}px; text-align: center;",
        ),
        _rule(
            "QProgressBar::chunk",
            f"background-color: {t.accent}; border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "QToolTip",
            # text_strong/surface_alt is the existing high-contrast "text
            # on card" token pair used elsewhere in this stylesheet --
            # previously this rule paired text_strong with a hard-coded
            # #ffffff text color, which collided in dark mode (text_strong
            # is #ffffff there too), rendering tooltips as invisible white
            # text on a white bubble.
            f"color: {t.text_strong}; background-color: {t.surface_alt}; "
            f"border: 1px solid {t.border_strong}; border-radius: 4px; "
            f"padding: 4px 6px;",
        ),
        _rule("QCheckBox", f"color: {t.text}; spacing: 6px;"),
        _rule(
            "QCheckBox::indicator",
            f"width: 16px; height: 16px; "
            f"border: 1px solid {t.border_strong}; "
            f"border-radius: {RADIUS_SMALL}px; background-color: {t.surface};",
        ),
        _rule(
            "QCheckBox::indicator:checked",
            f"background-color: {t.accent}; border-color: {t.accent};",
        ),
        _rule(
            "QScrollBar:vertical",
            "background: transparent; width: 12px; margin: 2px;",
        ),
        _rule(
            "QScrollBar::handle:vertical",
            f"background: {t.border_strong}; border-radius: {RADIUS_SMALL}px; "
            f"min-height: 24px;",
        ),
        _rule("QScrollBar::handle:vertical:hover", f"background: {t.accent};"),
        _rule(
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical",
            "height: 0px;",
        ),
        _rule(
            "QScrollBar:horizontal",
            "background: transparent; height: 12px; margin: 2px;",
        ),
        _rule(
            "QScrollBar::handle:horizontal",
            f"background: {t.border_strong}; border-radius: {RADIUS_SMALL}px; "
            f"min-width: 24px;",
        ),
        _rule("QScrollBar::handle:horizontal:hover", f"background: {t.accent};"),
        _rule(
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal",
            "width: 0px;",
        ),
        _rule("QSplitter::handle", f"background-color: {t.border}; border-radius: 2px;"),
        _rule("QSplitter::handle:hover", f"background-color: {t.accent};"),
        _rule(
            "QMessageBox, QDialog",
            f"background-color: {t.surface}; color: {t.text};",
        ),
        _rule(
            "QMenuBar",
            f"background-color: {t.surface}; color: {t.text}; "
            f"border-bottom: 1px solid {t.border};",
        ),
        _rule(
            "QMenuBar::item:selected",
            f"background-color: {t.accent_hover}; border-radius: 4px;",
        ),
        _rule(
            "QMenu",
            f"background-color: {t.surface}; color: {t.text}; "
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px;",
        ),
        _rule("QMenu::item:selected", f"background-color: {t.accent_hover};"),
        _rule("QStatusBar", f"background-color: {t.surface}; color: {t.text_muted};"),
        # Object-named rules for the info-box / status-label consolidation
        # (Teil 5) -- widgets opt in via setObjectName(...).
        _rule(
            "QFrame#mimicHeaderSurface",
            # Reuses the splash screen's dark-teal-to-background sweep (see
            # module docstring) so the header continues that identity
            # instead of blending flat into the page.
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {t.surface_alt}, stop:1 {t.bg}); "
            f"border-radius: {RADIUS_CARD}px; "
            f"border-bottom: 3px solid {t.accent}; "
            f"padding: 6px 12px;",
        ),
        _rule(
            "#mimicHeaderTitle",
            f"font-size: 19px; font-weight: 650; color: {t.accent_strong}; "
            f"background-color: transparent;",
        ),
        _rule(
            "#mimicBusyDot",
            f"background-color: {t.accent}; border-radius: 4px;",
        ),
        _rule(
            "#mimicInfoBoxAccent",
            f"background: {t.surface_alt}; border-left: 4px solid {t.accent}; "
            f"border-radius: 4px; padding: 7px 9px; color: {t.text}; "
            f"font-weight: 600;",
        ),
        _rule(
            "#mimicInfoBoxWarning",
            f"background: {t.warning_surface}; "
            f"border-left: 4px solid {t.warning}; border-radius: 4px; "
            f"padding: 6px 9px; color: {t.text};",
        ),
        _rule(
            "#mimicInfoBoxDanger",
            f"background: {t.danger_surface}; border-left: 4px solid {t.danger}; "
            f"border-radius: 4px; padding: 7px 9px; color: {t.text};",
        ),
        _rule(
            "#mimicInfoBoxNeutral",
            f"background: {t.surface_alt}; border-radius: 4px; "
            f"padding: 7px 9px; color: {t.text_muted};",
        ),
        _rule(
            "#mimicSurfaceBox",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-radius: 4px; padding: 7px; color: {t.text};",
        ),
        # Results atlas: an answer-first hero, evidence-coded cards, and a
        # quiet research navigation. These are presentation states, not new
        # clinical evidence levels or animated measurements.
        _rule(
            "QFrame#resultsAtlasHero",
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 {t.surface_alt}, stop:1 {t.surface}); "
            f"border: 1px solid {t.border}; border-left: 5px solid {t.accent}; "
            f"border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "#resultsAtlasKicker",
            f"font-size: 10px; font-weight: 700; color: {t.accent_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "#resultsAtlasHeadline",
            f"font-size: 20px; font-weight: 700; color: {t.text_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "#resultsAtlasState",
            f"color: {t.text_muted}; background: {t.surface}; "
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px; "
            "padding: 6px 9px;",
        ),
        _rule(
            '#resultsAtlasState[tone="research"]',
            f"color: {t.text}; background: {t.surface_alt}; border-color: {t.accent};",
        ),
        _rule(
            '#resultsAtlasState[tone="validated"]',
            f"color: {t.success}; background: {t.surface}; border-color: {t.success};",
        ),
        _rule(
            "QFrame#resultsAtlasCard",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-left: 4px solid {t.accent}; border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "QFrame#resultsAtlasEmpty",
            f"background: {t.surface}; border: 1px dashed {t.border_strong}; "
            f"border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "#resultsAtlasEmptyTitle",
            f"font-size: 16px; font-weight: 700; color: {t.text_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "QFrame#resultsAtlasEmptyStep",
            f"background: {t.surface_alt}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "#resultsAtlasEmptyStepTitle",
            f"font-weight: 700; color: {t.accent_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            'QFrame#resultsAtlasCard[tier="candidate"]',
            f"border-left-color: {t.warning};",
        ),
        _rule(
            'QFrame#resultsAtlasCard[tier="validated"]',
            f"border-left-color: {t.success};",
        ),
        _rule(
            "#resultsAtlasOrdinal",
            f"font-size: 10px; font-weight: 700; color: {t.text_muted}; "
            "background-color: transparent;",
        ),
        _rule(
            "#resultsAtlasEvidenceTag",
            f"color: {t.accent_strong}; background: {t.surface_alt}; "
            f"border: 1px solid {t.accent}; border-radius: {RADIUS_SMALL}px; "
            "padding: 3px 7px; font-weight: 650;",
        ),
        _rule(
            '#resultsAtlasEvidenceTag[tier="candidate"]',
            f"color: {t.warning}; background: {t.warning_surface}; "
            f"border-color: {t.warning};",
        ),
        _rule(
            '#resultsAtlasEvidenceTag[tier="validated"]',
            f"color: {t.success}; background: {t.surface}; "
            f"border-color: {t.success};",
        ),
        _rule(
            "#resultsAtlasCardTitle",
            f"font-size: 16px; font-weight: 700; color: {t.text_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "QPushButton#resultsAtlasCardAction",
            f"background: {t.accent}; border-color: {t.accent}; "
            "color: #ffffff; font-weight: 650;",
        ),
        _rule(
            "QPushButton#resultsAtlasCardAction:hover",
            f"background: {t.accent_strong}; border-color: {t.accent_strong};",
        ),
        _rule(
            "QTreeWidget#resultsAtlasNavigation",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_CARD}px; padding: 6px;",
        ),
        _rule(
            "QTreeWidget#resultsAtlasNavigation::item",
            f"padding: 6px 5px; border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "QTreeWidget#resultsAtlasNavigation::item:selected",
            f"background: {t.surface_alt}; color: {t.accent_strong}; "
            "font-weight: 650;",
        ),
        # ECG laboratory: the same visual language distinguishes source
        # signal, display controls, local measurements and research output.
        _rule(
            "QFrame#ecgLabHero",
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {t.surface_alt}, stop:1 {t.surface}); "
            f"border: 1px solid {t.border}; border-left: 5px solid {t.accent}; "
            f"border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "#ecgLabKicker, #ecgLabSectionLabel",
            f"font-size: 10px; font-weight: 700; color: {t.accent_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "#ecgLabTitle",
            f"font-size: 19px; font-weight: 700; color: {t.text_strong}; "
            "background-color: transparent;",
        ),
        _rule(
            "#ecgLabCaseStatus",
            f"color: {t.text_muted}; background: {t.surface}; "
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px; "
            "padding: 5px 8px;",
        ),
        _rule(
            "QToolButton#ecgLabSetupToggle",
            f"color: {t.accent_strong}; background: transparent; "
            "border: none; font-weight: 650; padding: 2px 4px;",
        ),
        _rule(
            "QToolButton#ecgLabSetupToggle:hover",
            f"color: {t.text_strong}; background: {t.surface_alt};",
        ),
        _rule(
            "QFrame#ecgLabSetupPanel, QFrame#ecgLabControlDeck, "
            "QFrame#ecgLabMeasurementDock",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "QFrame#ecgLabSignalStage",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-top: 3px solid {t.accent}; border-radius: {RADIUS_CARD}px;",
        ),
        _rule(
            "QWidget#ecgLabResearchSignalPanel, QWidget#ecgLabResearchSpatialPanel",
            f"background: {t.surface}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "QFrame#labQuickSuggestions",
            f"background: {t.surface_alt}; border: 1px solid {t.border}; "
            f"border-left: 4px solid {t.accent}; border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "QFrame#labViewerStrip",
            f"background: {t.surface_alt}; border: 1px solid {t.border}; "
            f"border-radius: {RADIUS_SMALL}px;",
        ),
        _rule(
            "#ecgLabMicrocopy",
            f"font-size: 10px; color: {t.text_muted}; background-color: transparent;",
        ),
        _rule(
            "QPushButton#ecgLabPrimaryAction",
            f"background: {t.accent}; color: #ffffff; border-color: {t.accent}; "
            "font-weight: 650;",
        ),
        _rule(
            "QPushButton#ecgLabPrimaryAction:hover",
            f"background: {t.accent_strong}; border-color: {t.accent_strong};",
        ),
        _rule(
            "QTabWidget#ecgLabWorkbenchTabs::pane",
            f"border: 1px solid {t.border}; border-radius: {RADIUS_SMALL}px; "
            f"background: {t.surface};",
        ),
        _rule("#mimicTextMuted", f"color: {t.text_muted}; background-color: transparent;"),
        _rule(
            "#mimicTextDangerMuted",
            f"color: {t.danger_muted}; background-color: transparent;",
        ),
    ]
    return "\n".join(parts)


_STATUS_COLOR = {
    "success": lambda t: t.success,
    "warning": lambda t: t.warning,
    "danger": lambda t: t.danger,
    "muted": lambda t: t.text_muted,
    "accent": lambda t: t.accent_strong,
}

_STATUS_BOX = {
    "success": lambda t: (t.success, t.surface_alt),
    "warning": lambda t: (t.warning, t.warning_surface),
    "danger": lambda t: (t.danger, t.danger_surface),
    "accent": lambda t: (t.accent, t.surface_alt),
    "neutral": lambda t: (None, t.surface_alt),
}


def status_qss(level: str, dark: bool, *, bold: bool = True) -> str:
    """Inline `color: ...;` rule for a dynamic, state-driven text label."""
    t = tokens_for(dark)
    color = _STATUS_COLOR[level](t)
    weight = "font-weight: 600;" if bold else ""
    return f"color: {color}; {weight}"


def status_box_qss(level: str, dark: bool) -> str:
    """Inline background/border/padding rule for a dynamic info box."""
    t = tokens_for(dark)
    border, bg = _STATUS_BOX[level](t)
    border_rule = f"border-left: 4px solid {border}; " if border else ""
    return (
        f"background: {bg}; {border_rule}border-radius: 4px; "
        f"padding: 7px 9px; color: {t.text};"
    )


def gauge_chunk_qss(level: str, dark: bool) -> str:
    """QProgressBar::chunk override for the CPU/RAM gauges."""
    t = tokens_for(dark)
    color = {"normal": t.accent, "warning": t.warning, "critical": t.danger}[level]
    return f"QProgressBar::chunk {{ background-color: {color}; border-radius: {RADIUS_SMALL}px; }}"


def apply_plot_theme(plot: "pg.PlotWidget", dark: bool) -> None:
    """Re-theme a pyqtgraph PlotWidget's background/axes -- QSS does not
    reach pyqtgraph at all, so this must be called explicitly for every
    tracked plot whenever the theme changes."""
    t = tokens_for(dark)
    plot.setBackground(t.surface if dark else "w")
    axis_color = t.text if dark else "#1d2a30"
    for axis_name in ("left", "bottom", "right", "top"):
        axis = plot.getAxis(axis_name)
        if axis is None:
            continue
        axis.setPen(axis_color)
        axis.setTextPen(axis_color)
