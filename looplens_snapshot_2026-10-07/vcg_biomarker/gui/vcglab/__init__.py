"""VCG-Analyselabor: Schleifen, Vektoren, Maße und Musterlesung des Vektorkardiogramms.

Eigenständiges Fenster (``py -3.12 -m vcg_biomarker.gui.vcglab``) mit kleinem Vorschaufenster;
die Einbindung ins ECG-Labor erfolgt erst nach Freigabe. Die Mathematik liegt in ``measures.py``
(reine NumPy-Funktionen), die Musterlesung in ``reading.py``; ``analysis.py`` koppelt lesend an
``core`` (Delineation, Kors/inverse Dower, physiologischer Ursprung, Markerregister).
"""
