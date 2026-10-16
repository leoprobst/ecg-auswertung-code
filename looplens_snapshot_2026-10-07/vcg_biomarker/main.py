import sys
from multiprocessing import freeze_support
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.config import AppConfig
from vcg_biomarker.gui.main_window import MainWindow
from vcg_biomarker.gui.widgets.splash_screen import AnimatedSplashScreen


def main():
    # pyqtgraph caches shader programs across GL widgets. Independent native
    # contexts can otherwise reuse invalid program IDs when a lab is reopened.
    # Qt requires this attribute before the application/context is created.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    splash = AnimatedSplashScreen()
    splash.show()
    app.processEvents()

    config = AppConfig()

    if len(sys.argv) > 1:
        config.ptbxl_path = Path(sys.argv[1])

    window = MainWindow(config, progress_callback=splash.set_progress)
    window.showMaximized()

    fade = QPropertyAnimation(splash, b"windowOpacity", splash)
    fade.setDuration(280)
    fade.setStartValue(1.0)
    fade.setEndValue(0.0)
    fade.setEasingCurve(QEasingCurve.Type.OutCubic)
    fade.finished.connect(splash.close)
    splash._fade_anim = fade
    fade.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    freeze_support()
    main()
