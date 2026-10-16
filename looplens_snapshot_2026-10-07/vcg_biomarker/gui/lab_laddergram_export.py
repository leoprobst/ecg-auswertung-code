"""Vector/raster export of an explicitly manual laddergram, not ECG paper scale."""
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPdfWriter, QPageLayout, QPageSize, QFont
from PyQt6.QtSvg import QSvgGenerator
import numpy as np


def export_laddergram(path, document, *, signal=None, lead=None):
    from .lab_laddergram import validate_diagram, LEVELS
    validate_diagram(document, document['signal_reference'])
    strip = None
    if signal is not None:
        from ..core.data_loader import LEAD_NAMES_12
        values = np.asarray(signal)
        if values.shape != (document['signal_reference']['sample_count'], 12) or lead not in LEAD_NAMES_12:
            raise ValueError('EKG-Export: Signalform oder Ableitung ungültig')
        strip = values[:, list(LEAD_NAMES_12).index(lead)]
        finite = strip[np.isfinite(strip)]
        if not finite.size:
            raise ValueError('EKG-Export: keine endlichen Signalwerte')
        low, high = float(finite.min()), float(finite.max())
    metadata = {'included': strip is not None}
    if strip is not None:
        metadata.update(lead=lead, unit='mV', minimum=low, maximum=high,
                        amplitude_scale='automatic_not_calibrated_ecg_paper',
                        processing='native_signal_no_filter_no_inversion')
    suffix = path.suffix.lower()
    if suffix == '.png':
        device = QImage(1400, 600, QImage.Format.Format_ARGB32)
        device.fill(QColor('white'))
    elif suffix == '.svg':
        device = QSvgGenerator()
        device.setFileName(str(path))
        device.setSize(QSize(1400, 600))
        device.setViewBox(QRectF(0, 0, 1400, 600))
        device.setTitle('Manuelles Laddergram – Forschung, keine Diagnosefreigabe')
    elif suffix == '.pdf':
        device = QPdfWriter(str(path))
        device.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        device.setPageOrientation(QPageLayout.Orientation.Landscape)
        device.setTitle('Manuelles Laddergram – Forschung')
    else:
        raise ValueError('Exportformat nicht unterstützt')
    painter = QPainter(device)
    if not painter.isActive():
        raise OSError('Grafikexport konnte nicht gestartet werden')
    try:
        painter.setWindow(0, 0, 1400, 600)
        painter.fillRect(QRectF(0, 0, 1400, 600), QColor('white'))
        font = QFont('Arial')
        font.setPixelSize(16)
        painter.setFont(font)
        painter.setPen(QColor('#172333'))
        reference = document['signal_reference']
        caption = ('Manuelles Laddergram · mechanistische Hypothese, keine Diagnosefreigabe\n'
                   f'Fall: {reference["case_id"]} · Bearbeiter: {document.get("author", "unbekannt")}\n'
                   f'Signalhash: {reference["source_sha256"]}\n'
                   f'Zeitpunkt: {document.get("created_at", "unbekannt")} · schematische Ebenen, keine anatomische Entfernung')
        painter.drawText(QRectF(25, 10, 1350, 115), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, caption)
        fs = reference['sampling_rate_hz']
        duration = max(1/fs, (reference['sample_count']-1)/fs)
        x = lambda sample: 80 + 1240 * (sample/fs) / duration
        y = lambda level: 430 - LEVELS[level]*(75 if strip is not None else 130)
        if strip is not None:
            from PyQt6.QtGui import QPainterPath
            painter.drawText(80, 140, f'{lead} · Originalsignal · {low:.4g} bis {high:.4g} mV · automatische Skalierung, kein EKG-Papiermaßstab')
            trace = QPainterPath()
            connected = False
            span = high-low
            for sample, value in enumerate(strip):
                if not np.isfinite(value):
                    connected = False
                    continue
                ordinate = 205 if span == 0 else 245 - 80 * (float(value)-low)/span
                if connected:
                    trace.lineTo(x(sample), ordinate)
                else:
                    trace.moveTo(x(sample), ordinate)
                connected = True
            painter.setPen(QPen(QColor('#172333'), 1))
            painter.drawPath(trace)
        for level in LEVELS:
            painter.setPen(QPen(QColor('#999999'), 1, Qt.PenStyle.DotLine))
            painter.drawLine(80, y(level), 1320, y(level))
            painter.setPen(QColor('#172333'))
            painter.drawText(25, y(level)+5, level)
        from PyQt6.QtCore import QLineF
        events = {item['id']: item for item in document['events']}
        for item in events.values():
            painter.setPen(QPen(QColor('#0072b2'), 3))
            painter.drawLine(QLineF(x(item['sample']), y(item['level'])-18, x(item['sample']), y(item['level'])+18))
        for edge in document['edges']:
            a, b = events[edge['source']], events[edge['target']]
            x0, y0, x1, y1 = x(a['sample']), y(a['level']), x(b['sample']), y(b['level'])
            painter.setPen(QPen(QColor('#d55e00'), 2))
            if edge['kind'] == 'blocked_hypothesis':
                x1, y1 = (x0+x1)/2, (y0+y1)/2
                painter.drawLine(QLineF(x1-6, y1-6, x1+6, y1+6))
                painter.drawLine(QLineF(x1-6, y1+6, x1+6, y1-6))
            painter.drawLine(QLineF(x0, y0, x1, y1))
        painter.setPen(QColor('#172333'))
        for index in range(11):
            seconds = index * duration / 10
            painter.drawText(QRectF(55+index*124, 470, 80, 30), Qt.AlignmentFlag.AlignCenter, f'{seconds:.3g}')
        painter.drawText(80, 525, 'Zeit [s] im Originalsignal · Kreuz = manuell postulierte Blockierung')
    finally:
        painter.end()
    if suffix == '.png' and not device.save(str(path)):
        raise OSError('PNG konnte nicht gespeichert werden')
    return metadata
