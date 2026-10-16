"""Bounded, read-only tree of recorded evidence; never infer missing edges."""
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem


class MarkerProvenanceTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(['Herkunft / Eingabe', 'Gespeicherte Angabe'])
        self.setMaximumHeight(260)
        self.setColumnWidth(0, 310)
        self.setToolTip('Aufklappbare Messprovenienz, kein Kausalgraph. Unbekannte Eingaben werden nicht ergänzt.')

    def set_evidence(self, detail):
        self.clear()
        budget = [400]

        def add(parent, key, value, depth=0):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            structured = isinstance(value, (dict, list, tuple))
            label = f'{len(value)} Einträge' if structured else 'nicht dokumentiert' if value is None else str(value)
            item = QTreeWidgetItem(parent, [str(key), label[:500]])
            item.setToolTip(1, label[:4000])
            if structured and depth < 7:
                entries = value.items() if isinstance(value, dict) else enumerate(value)
                for child_key, child in entries:
                    if budget[0] <= 1:
                        QTreeWidgetItem(item, ['Weitere Angaben', 'Vollständige gespeicherte Daten im JSON-Inspektor'])
                        budget[0] = 0
                        break
                    add(item, child_key, child, depth + 1)
            elif structured:
                item.setToolTip(1, 'Weitere verschachtelte Angaben im vollständigen JSON-Inspektor.')
            return item

        for title, value, limit in (
            ('Analysestand', detail.get('analysis_id'), 20),
            ('Gespeicherte Messung / Fenster', detail.get('measurement_provenance'), 280),
            ('Aktuelles Register – kein historischer Nachweis', detail.get('current_registry_reference_not_historical_proof'), 100),
        ):
            budget[0] = limit  # One long beat list must not hide the other provenance roots.
            root = add(self, title, value)
            if root:
                root.setExpanded(True)
