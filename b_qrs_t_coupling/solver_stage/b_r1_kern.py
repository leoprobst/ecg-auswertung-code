"""AW-238 B-R1: korrigierte Bootstrap-Gueltigkeitsregel (Kern, keine Datenzugriffe).

Fehler in den gebundenen Originalskripten (nicht geaendert):
  werkzeuge/b_test/statistik.py::fit    Zeile 93:  if min(counts.values())<100: raise ...  (counts = verschiedene Personen je Klasse)
  werkzeuge/b_test/aw171_kernels.py::gates Zeile 22: require(min(count_nonzero(yy==c))>=100, 'kleine_Strata')
Beide Pruefungen wurden in JEDER Bootstrap-Ziehung auf die Zahl der Personen mit Multiplizitaet > 0 angewandt. Bei 145 Faellen
und einer Ziehung von G Personen aus G ist der Erwartungswert verschiedener Personen etwa 63,2 % von 145, also etwa 92. Die
Stichprobeneignung (mindestens 100 Personen je Klasse) ist eine Eigenschaft der Originalstichprobe und wird dort einmal
geprueft. Fuer die Ziehungen gelten nur noch die Bedingungen der numerischen Schaetzbarkeit.

Diese Datei patcht die beiden Pruefungen NUR zur Laufzeit und NUR fuer Replikatfits (Gewichtsvektor uebergeben). Punktschaetzer
auf der Originalstichprobe laufen mit den unveraenderten Originalregeln.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import contextlib
import sys

import numpy as np

ROOT = str(data_path("b.auswertung"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from werkzeuge.b_test import aw171_kernels as k  # noqa: E402
from werkzeuge.b_test import statistik as st  # noqa: E402


def gates_corrected(x, y, w=None, ess=False):
    """Kopie von aw171_kernels.gates ohne die Personenzahlpruefung in Replikaten (w uebergeben)."""
    sample = w is None
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    k.require(w.shape == y.shape and np.isfinite(w).all() and (w >= 0).all(), "Gewichte")
    a = w > 0
    xx, yy, ww = x[a], y[a], w[a]
    k.require(np.isfinite(xx).all() and set(np.unique(yy)) == {0, 1}, "Klasse/finite")
    if sample:
        k.require(min(np.count_nonzero(yy == c) for c in [0, 1]) >= 100, "kleine_Strata")
    if ess:
        k.require(min(ww[yy == c].sum() ** 2 / (ww[yy == c] @ ww[yy == c]) for c in [0, 1]) >= 100, "ESS")
    k.require(len(yy) > xx.shape[1] and np.linalg.matrix_rank(xx) == xx.shape[1], "Rang")
    k.require(k.separation(xx, yy) == "keine", "Separation")
    return xx, yy, ww


@contextlib.contextmanager
def replicate_rule():
    """Schaltet die Personenzahlpruefung in Replikaten ab (nur innerhalb des Blocks)."""
    old_counts, old_gates = st.patient_counts, k.gates
    st.patient_counts = lambda subjects, y: {"0": 10 ** 9, "1": 10 ** 9}
    k.gates = gates_corrected
    try:
        yield
    finally:
        st.patient_counts, k.gates = old_counts, old_gates


def numeric(v):
    return {a: float(b) for a, b in v.items() if isinstance(b, (int, float, np.integer, np.floating)) and np.isfinite(b)}


def multiplicities(union_n, r):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026, int(r)])))
    return np.bincount(rng.integers(0, union_n, union_n), minlength=union_n)


def percentile_ci(values, minimum=950):
    v = np.asarray(values, float)
    if len(v) < minimum:
        return None
    return np.quantile(v, [0.025, 0.975], method="linear").tolist()
