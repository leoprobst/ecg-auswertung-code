"""Local calibrated vector ECG PDF. Physical units independent of viewport/DPI."""
from __future__ import annotations

import math
import html
from numbers import Real
import numpy as np
from PyQt6.QtCore import QRectF, QSizeF, QLineF
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPageSize, QPageLayout, QTextDocument, QPdfWriter, QImage

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def paper_coordinates(seconds, mv, speed=25., gain=10.):
    return np.asarray(seconds) * speed, -np.asarray(mv) * gain


def research_appendix_html(payload, marker_names):
    """Only explicitly selected research measurements, no diagnostic upgrading."""
    esc = lambda value: html.escape(str(value))
    values = payload.get('feature_values')
    values = values if isinstance(values, dict) else {}
    measurements = payload.get('feature_measurements')
    measurements = measurements if isinstance(measurements, dict) else {}
    rows = []
    for name in dict.fromkeys(marker_names):
        if name not in values:
            continue
        value = values[name]
        finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
        detail = measurements.get(name)
        detail = detail if isinstance(detail, dict) else {}
        status = detail.get('status') or 'historischer Stand ohne Messstatus'
        if not finite:
            status = f'nicht berechenbar / ungültig; gespeicherter Status: {status}'
        rows.append('<tr>' + ''.join(f'<td>{esc(item)}</td>' for item in (
            name, value if finite else 'nicht berechenbar', detail.get('unit') or 'unbekannt',
            status, detail.get('formula') or 'nicht dokumentiert')) + '</tr>')
    return ('<h1>Forschungsanhang – keine klinische Diagnose</h1>'
            '<p>Explizit ausgewählte automatische Messungen. Keine Erkrankungswahrscheinlichkeiten, '
            'keine unabhängige Bestätigung und keine klinische Freigabe. Fehlend bedeutet nicht normal.</p>'
            f'<p>Analyse: {esc(payload.get("analysis_id", "unbekannt"))}<br>'
            f'Transformation: {esc(payload.get("vcg_method", "unbekannt"))}</p>'
            '<table border="1" cellpadding="4"><tr><th>Marker</th><th>Wert</th><th>Einheit</th><th>Messstatus</th><th>Formel</th></tr>'
            + ''.join(rows) + '</table>')


def _lead_display_label(lead: int, display_sign: int) -> str:
    return '−aVR (invertiert)' if lead == 3 and display_sign == -1 else LEADS[lead]


def export_ecg_pdf(path, report_html, signal, fs, speed=25., gain=10., *, research_html=None, cabrera=False, invert_avr=False):
    """Print all native samples, segmented 3x4 plus lead II, vector paths."""
    printer = QPdfWriter(str(path))
    printer.setResolution(300)
    printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    printer.setPageOrientation(QPageLayout.Orientation.Landscape)
    doc = QTextDocument()
    layout_device = QImage(1, 1, QImage.Format.Format_ARGB32)
    layout_device.setDotsPerMeterX(round(96/.0254))
    layout_device.setDotsPerMeterY(round(96/.0254))
    doc.documentLayout().setPaintDevice(layout_device)
    rect = printer.pageLayout().paintRect(QPageLayout.Unit.Millimeter)
    width, height = rect.width(), rect.height()
    css_mm = 25.4/96
    doc.setPageSize(QSizeF(width/css_mm, height/css_mm))
    doc.setHtml(report_html)
    painter = QPainter(printer)
    if not painter.isActive():
        raise OSError("PDF painter could not start")
    try:
        mm = printer.resolution()/25.4
        for page in range(doc.pageCount()):
            if page:
                printer.newPage()
            painter.save()
            painter.fillRect(QRectF(0, 0, width*mm, height*mm), QColor("white"))
            painter.scale(mm*css_mm, mm*css_mm)
            painter.setClipRect(QRectF(0, 0, width/css_mm, height/css_mm))
            painter.translate(0, -page*height/css_mm)
            doc.drawContents(painter)
            painter.restore()
        duration = len(signal)/fs
        usable_width = width-14
        page_duration = usable_width/speed
        order = [4, 1, 6, 9, 0, 5, 7, 10, 3, 2, 8, 11] if cabrera else [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        avr_sign = -1 if cabrera or invert_avr else 1
        for page in range(max(1, math.ceil(duration/page_duration))):
            printer.newPage()
            painter.save()
            painter.scale(mm, mm)
            painter.fillRect(QRectF(0, 0, width, height), QColor("white"))
            font = QFont("Arial")
            font.setPixelSize(3)  # painter coordinates are millimetres here
            painter.setFont(font)
            painter.setPen(QColor("#172333"))
            begin = page*page_duration
            display = 'Cabrera / −aVR' if cabrera else 'Standard / −aVR' if invert_avr else 'Standard / aVR'
            painter.drawText(QRectF(0, 0, width, 8), 0, f"Originalsignal · {fs:g} Hz · {speed:g} mm/s · {gain:g} mm/mV · {display} · 3×4 zeitversetzt + II · ab {begin:.3f} s · Druck: 100 %")
            row_height = (height-18)/4
            for pos, lead in enumerate(order):
                row, col = divmod(pos, 4)
                start = begin+col*page_duration/4
                box = QRectF(12+col*usable_width/4, 12+row*row_height, usable_width/4, row_height)
                _draw_strip(painter, signal, fs, lead, start, page_duration/4, box, speed, gain, display_sign=avr_sign if lead == 3 else 1)
            _draw_strip(painter, signal, fs, 1, begin, page_duration, QRectF(12, 12+3*row_height, usable_width, row_height), speed, gain)
            painter.restore()
        if research_html:
            doc.setHtml(research_html)
            for page in range(doc.pageCount()):
                printer.newPage()
                painter.save()
                painter.fillRect(QRectF(0, 0, width*mm, height*mm), QColor("white"))
                painter.scale(mm*css_mm, mm*css_mm)
                painter.setClipRect(QRectF(0, 0, width/css_mm, height/css_mm))
                painter.translate(0, -page*height/css_mm)
                doc.drawContents(painter)
                painter.restore()
    finally:
        painter.end()


def _draw_strip(painter, signal, fs, lead, start, duration, box, speed, gain, *, display_sign=1):
    painter.save()
    painter.setClipRect(box)
    for mm in range(math.ceil(box.width())+1):
        painter.setPen(QPen(QColor("#ddbbbb" if mm % 5 == 0 else "#f1dddd"), .12 if mm % 5 == 0 else .06))
        painter.drawLine(QLineF(box.left()+mm, box.top(), box.left()+mm, box.bottom()))
    for mm in range(math.ceil(box.height())+1):
        painter.setPen(QPen(QColor("#ddbbbb" if mm % 5 == 0 else "#f1dddd"), .12 if mm % 5 == 0 else .06))
        painter.drawLine(QLineF(box.left(), box.top()+mm, box.right(), box.top()+mm))
    a, b = max(0, round(start*fs)), min(len(signal), round((start+duration)*fs))
    if b > a:
        times = np.arange(a, b)/fs-start
        x, y = paper_coordinates(times, display_sign*signal[a:b, lead], speed, gain)
        path = QPainterPath()
        pen_down = False
        for px, py in zip(x, y):
            if not np.isfinite(py):
                pen_down = False
            elif pen_down:
                path.lineTo(box.left()+float(px), box.center().y()+float(py))
            else:
                path.moveTo(box.left()+float(px), box.center().y()+float(py))
                pen_down = True
        painter.setPen(QPen(QColor("#111111"), .16))
        painter.drawPath(path)
        clipped = np.nanmax(np.abs(y)) >= box.height()/2
    else:
        clipped = False
    painter.setPen(QColor("#a00000" if clipped else "#152f40"))
    lead_label = _lead_display_label(lead, display_sign)
    painter.drawText(QRectF(box.left()+1, box.top()+1, box.width()-2, 5), 0, f"{lead_label} · {start:.2f} s" + (" · Amplitude angeschnitten" if clipped else ""))
    painter.restore()
    # 1-mV calibration pulse outside data rectangle, 200-ms width in paper units.
    if box.left() == 12:
        painter.setPen(QPen(QColor("#111111"), .16))
        path = QPainterPath()
        path.moveTo(1, box.center().y())
        path.lineTo(2, box.center().y())
        path.lineTo(2, box.center().y()-gain)
        path.lineTo(2+.2*speed, box.center().y()-gain)
        path.lineTo(2+.2*speed, box.center().y())
        painter.drawPath(path)
