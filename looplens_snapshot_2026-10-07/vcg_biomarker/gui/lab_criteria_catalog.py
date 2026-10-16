"""Current source registry browser, deliberately separate from case evidence."""
import json
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit, QPushButton, QAbstractItemView


def catalog_entries(registry):
    """Use the source-locked registry, not another copy of clinical formulas."""
    return sorted((definition.to_dict() for definition in registry.all()),
                  key=lambda item: (item['display_name'].casefold(), item['baseline_id']))


class CriteriaCatalogWidget(QWidget):
    criterionRequested = pyqtSignal(str)

    def __init__(self, parent=None, registry=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        note = QLabel('Aktuelles Projektregister – Nachschlagewerk, kein Ergebnis dieses Falls. '
                      'Projekt-IDs sind keine AHA-Statement-Codes. Formeln und Quellenstand können '
                      'von historischen Analysen abweichen; keine automatische Befundübernahme.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Name, Endpunkt, Formel oder Quelle suchen …')
        layout.addWidget(self.search)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Kriterium', 'Projekt-ID', 'Quellenstand'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.table)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.open_case = QPushButton('Gespeichertes Fallkriterium zur Prüfung öffnen')
        self.open_case.setEnabled(False)
        self.open_case.clicked.connect(self.request_case)
        layout.addWidget(self.open_case)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.entries = []
        try:
            if registry is None:
                from ..mimic.baseline_registry import BaselineRegistry
                registry = BaselineRegistry()
            self.entries = catalog_entries(registry)
            self.status.setText(f'{len(self.entries)} Definitionen · Registerhash: {registry.registry_hash}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText('Quellenregister nicht verfügbar: ' + str(exc))
        self.table.setRowCount(len(self.entries))
        for row, item in enumerate(self.entries):
            for col, value in enumerate((item['display_name'], item['baseline_id'], item['source_version'])):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.table.setColumnWidth(0, 350)
        self.table.setColumnWidth(1, 270)
        self.search.textChanged.connect(self.filter)
        self.table.itemSelectionChanged.connect(self.select)

    def filter(self, text):
        terms = text.casefold().split()
        for row, item in enumerate(self.entries):
            hidden = not all(term in json.dumps(item, ensure_ascii=False, default=str).casefold() for term in terms)
            self.table.setRowHidden(row, hidden)
            if hidden and self.table.currentRow() == row:
                self.table.clearSelection()
                self.table.setCurrentCell(-1, -1)
        self.select()

    def select(self):
        row = self.table.currentRow()
        self.open_case.setEnabled(0 <= row < len(self.entries) and bool(self.entries[row].get('criterion_id')))
        if row < 0:
            self.details.clear()
            return
        item = self.entries[row]
        self.details.setPlainText('\n\n'.join(f'{label}: {item.get(key) or "nicht hinterlegt"}' for key, label in (
            ('display_name', 'Bezeichnung'), ('formula', 'Definition'), ('required_inputs', 'Erforderliche Eingaben'),
            ('exclusions', 'Ausschlüsse'), ('strata', 'Stratifizierung'), ('endpoint_ids', 'Forschungsendpunkte'),
            ('source_title', 'Quelle'), ('source_version', 'Quellenstand'), ('source_url', 'Literaturadresse'),
            ('notes', 'Aussagegrenzen'))))

    def request_case(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.entries) and not self.table.isRowHidden(row):
            criterion = self.entries[row].get('criterion_id')
            if criterion:
                self.criterionRequested.emit(criterion)
