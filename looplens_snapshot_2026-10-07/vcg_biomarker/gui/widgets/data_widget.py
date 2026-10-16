from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QComboBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ...config import AppConfig
from ...core.data_loader import PTBXLDataLoader, PTBXL_DIAGNOSTIC_SUPERCLASSES
from ...core.pipeline import Pipeline


class DataWidget(QWidget):
    dataset_loaded = pyqtSignal(dict)
    view_record_requested = pyqtSignal(int)

    def __init__(self, config: AppConfig, pipeline: Pipeline, parent=None):
        super().__init__(parent)
        self.config = config
        self.pipeline = pipeline
        self._data_loader: PTBXLDataLoader | None = None
        self._selected_ids: list[int] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        path_group = QGroupBox("Dataset Path")
        path_layout = QHBoxLayout(path_group)
        self.path_label = QLabel("No dataset selected")
        self.path_label.setMinimumWidth(400)
        path_layout.addWidget(self.path_label)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(self.browse_btn)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_dataset)
        path_layout.addWidget(self.load_btn)
        layout.addWidget(path_group)

        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)

        sc_layout = QHBoxLayout()
        sc_layout.addWidget(QLabel("Diagnostic Superclasses:"))
        self.sc_checkboxes = {}
        for sc in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            cb = QCheckBox(sc)
            cb.setChecked(True)
            self.sc_checkboxes[sc] = cb
            sc_layout.addWidget(cb)
        filter_layout.addLayout(sc_layout)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Folds:"))
        self.fold_combo = QComboBox()
        self.fold_combo.addItems(["All", "1-8 (Train)", "9 (Validation)", "10 (Test)"])
        options_layout.addWidget(self.fold_combo)

        options_layout.addWidget(QLabel("Max Records:"))
        self.max_records_spin = QSpinBox()
        self.max_records_spin.setRange(0, 100000)
        self.max_records_spin.setValue(0)
        self.max_records_spin.setSpecialValueText("All")
        options_layout.addWidget(self.max_records_spin)

        self.validated_cb = QCheckBox("Validated by human only")
        options_layout.addWidget(self.validated_cb)
        filter_layout.addLayout(options_layout)

        self.apply_filter_btn = QPushButton("Apply Filters")
        self.apply_filter_btn.clicked.connect(self._apply_filters)
        filter_layout.addWidget(self.apply_filter_btn)

        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View ECG ID:"))
        self.ecg_id_spin = QSpinBox()
        self.ecg_id_spin.setRange(1, 99999)
        self.ecg_id_spin.setValue(1)
        view_layout.addWidget(self.ecg_id_spin)
        self.view_btn = QPushButton("Load into Viewers")
        self.view_btn.clicked.connect(self._on_view_record)
        view_layout.addWidget(self.view_btn)
        filter_layout.addLayout(view_layout)
        layout.addWidget(filter_group)

        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Category", "Count"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stats_layout.addWidget(self.stats_table)
        layout.addWidget(stats_group)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def set_dataset_path(self, path: str):
        self.path_label.setText(path)
        self.config.ptbxl_path = path

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select PTB-XL Directory")
        if path:
            self.set_dataset_path(path)

    def _load_dataset(self):
        path = self.path_label.text()
        if path == "No dataset selected":
            return

        try:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.status_label.setText("Loading database...")

            self._data_loader = PTBXLDataLoader(path, use_hr=self.config.use_hr)
            db = self._data_loader.load_database()

            self.pipeline.data_loader = self._data_loader

            self._update_stats()
            self.progress.setVisible(False)
            self.status_label.setText(f"Loaded {len(db)} records. Pipeline data_loader shared.")
            self.dataset_loaded.emit({"n_records": len(db)})

        except Exception as e:
            self.progress.setVisible(False)
            self.status_label.setText(f"Error: {str(e)}")

    def _update_stats(self):
        if not self._data_loader:
            return
        dist = self._data_loader.get_diagnostic_distribution()
        self.stats_table.setRowCount(len(dist))
        for i, (sc, row) in enumerate(dist.iterrows()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(sc))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(int(row["count"]))))

    def _apply_filters(self):
        if not self._data_loader:
            self.status_label.setText("Load dataset first")
            return

        superclasses = [sc for sc, cb in self.sc_checkboxes.items() if cb.isChecked()]
        if not superclasses:
            superclasses = None

        fold_text = self.fold_combo.currentText()
        folds = None
        if fold_text == "1-8 (Train)":
            folds = list(range(1, 9))
        elif fold_text == "9 (Validation)":
            folds = [9]
        elif fold_text == "10 (Test)":
            folds = [10]

        max_records = self.max_records_spin.value()
        if max_records == 0:
            max_records = None

        self._selected_ids = self._data_loader.get_record_ids(
            superclasses=superclasses,
            folds=folds,
            validated_only=self.validated_cb.isChecked(),
        )

        if max_records and len(self._selected_ids) > max_records:
            self._selected_ids = self._selected_ids[:max_records]

        self.status_label.setText(f"Selected {len(self._selected_ids)} records")

    def get_selected_ids(self) -> list[int]:
        return self._selected_ids

    def _on_view_record(self):
        ecg_id = self.ecg_id_spin.value()
        self.view_record_requested.emit(ecg_id)
