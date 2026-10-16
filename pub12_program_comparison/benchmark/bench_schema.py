"""Output schema of the benchmark: one row per dataset/recording/tool/measurement.

Long format (tidy). Failures are their own status values, not silent row losses.
The measurement keys follow section 4 of the benchmark specification.
The mapping to native export names is a planning state (bindung = "zu_binden"),
until code line and version are frozen for each tool.
"""
from __future__ import annotations

import csv
from pathlib import Path

SPALTEN = [
    # (name, description)
    ("lauf_id", "Identifier of the computation run (assignment, date), for tracing back to script and configuration"),
    ("datensatz", "ludb | ptbxl | ptbxlplus | qtdb"),
    ("aufnahme", "Recording identifier of the dataset (no clinic person identifier)"),
    ("cluster_id", "pseudonymous person key for clustered bootstrap CI; mapping table only under <local path>"),
    ("ableitung", "I II III aVR aVL aVF V1-V6 | X Y Z | global"),
    ("bezug", "aufnahme | medianschlag | schlag"),
    ("schlag_nr", "running number for bezug=schlag, otherwise empty"),
    ("werkzeug", "braveheart | looplens | ecgdeli | neurokit2 | ge12sl | unig | referenz_experte"),
    ("werkzeug_version", "Release or commit"),
    ("konfig_hash", "SHA-256 of the frozen configuration (Annoparams, batch_settings, filter)"),
    ("arm", "nativ | gemeinsam | referenz"),
    ("vorverarbeitung", "Identifier from the frozen preprocessing plan (e.g. VV0 = no additional preprocessing)"),
    ("messgroesse", "Key from messgroessen.csv"),
    ("native_bezeichnung", "Column/field name in the tool export"),
    ("einheit", "ms | mV | mV*ms | grad | 1"),
    ("wert", "Number, empty if status != ok"),
    ("status", "ok | nicht_definiert | nicht_ladbar | werkzeugfehler | fehlend | ausgeschlossen_qc | nicht_zuordenbar"),
    ("statusgrund", "short text, empty if ok"),
]

SPALTENNAMEN = [s[0] for s in SPALTEN]

STATUS = {
    "ok", "nicht_definiert", "nicht_ladbar", "werkzeugfehler",
    "fehlend", "ausgeschlossen_qc", "nicht_zuordenbar",
}

# Measurements of the primary families (specification section 4).
# BRAVEHEART names: fields of the classes VCG_Calc / Lead_Morphology in release v1.8.2,
# as far as read in the source code (VCG_Calc.m lines 160-162, 223-224; Beats.m: only Q, QRS, S, T, Tend).
# Everything with "zu_binden" is a hypothesis until bound to code lines.
MESSGROESSEN = [
    # key, description, unit, braveheart_native, binding, remark
    ("qrs_dauer_ableitung", "QRS end minus QRS onset per lead", "ms", "", "n/a",
     "BRAVEHEART annotates fiducials only on the VCG median beat, no lead-wise boundaries"),
    ("qrs_dauer_global", "global QRS duration", "ms", "VCG_Calc.qrs_int", "zu_binden",
     "qrs_int = (qend-1)*1000/hz from the median beat; bind the definition of qend before the start"),
    ("p_dauer", "P end minus P onset", "ms", "", "n/a",
     "Beats.m has no P fiducials (Q, QRS, S, T, Tend)"),
    ("pr_intervall", "QRS onset minus P onset", "ms", "", "n/a", "no P mark in BRAVEHEART"),
    ("qt_global", "QRS onset to T end", "ms", "VCG_Calc.qt_int", "zu_binden",
     "qt_int = (Tend - Q)*1000/hz on the median beat; check the Q convention (onset of the Q wave)"),
    ("qtcf", "QT(s)/RR(s)^(1/3)", "ms", "median_intervals.qtc_f", "zu_binden",
     "bind the RR contract (median RR vs. median beat) before the start"),
    ("qrst_winkel_peak", "Angle of the maximal vector norms of QRS and T, 0-180", "grad",
     "VCG_Calc.qrst_angle_peak", "zu_binden", "Record the matrix (Kors/Dower), origin and baseline"),
    ("qrst_winkel_integral", "Angle of the signed XYZ integrals of QRS and T, 0-180", "grad",
     "VCG_Calc.qrst_angle_area", "zu_binden", "Additionally bind the integration and baseline contract"),
    ("vektorbetrag_max_qrs", "max sqrt(X^2+Y^2+Z^2) in the QRS window", "mV",
     "VCG_Calc.q_peak_mag", "zu_binden", ""),
    ("integralbetrag_qrs", "Norm of the signed XYZ integral vector in the QRS window", "mV*ms",
     "VCG_Calc.q_area_mag", "zu_binden", "Magnitude of the integral, not integral of the magnitude (that is VMQ_area)"),
    ("sai_qrst", "Sum of absolute XYZ integrals of QRST", "mV*ms", "VCG_Calc.sai_qrst", "zu_binden", ""),
    ("schleifenlaenge_qrs", "Sum of Euclidean XYZ step lengths in the QRS window", "mV",
     "VCG_Calc.vcg_length_qrs", "zu_binden", "Check quadrature and time step"),
]


def schreibe_schema(zielordner: Path) -> list[Path]:
    """Writes ausgabeschema.csv and messgroessen.csv into the target folder."""
    zielordner.mkdir(parents=True, exist_ok=True)
    p1 = zielordner / "ausgabeschema.csv"
    with p1.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spalte", "beschreibung"])
        w.writerows(SPALTEN)
    p2 = zielordner / "messgroessen.csv"
    with p2.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schluessel", "beschreibung", "einheit", "braveheart_nativ", "bindung", "bemerkung"])
        w.writerows(MESSGROESSEN)
    return [p1, p2]


def pruefe_zeile(z: dict) -> list[str]:
    """Returns error texts for an output row (empty list = valid)."""
    fehler = []
    fehlt = [s for s in SPALTENNAMEN if s not in z]
    if fehlt:
        fehler.append(f"Columns missing: {fehlt}")
    if z.get("status") not in STATUS:
        fehler.append(f"Unknown status: {z.get('status')!r}")
    if z.get("status") == "ok" and z.get("wert") in (None, ""):
        fehler.append("status=ok without value")
    if z.get("status") != "ok" and z.get("wert") not in (None, ""):
        fehler.append("Value set although status != ok")
    schluessel = {m[0] for m in MESSGROESSEN}
    if z.get("messgroesse") not in schluessel:
        fehler.append(f"Unknown measurement: {z.get('messgroesse')!r}")
    return fehler


def oeffne_ausgabe(pfad: Path):
    """Opens an output file with a header row, returns (file, csv.DictWriter)."""
    f = pfad.open("w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=SPALTENNAMEN)
    w.writeheader()
    return f, w
