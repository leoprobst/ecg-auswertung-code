"""AW-243 Teil 4: strengere Null fuer S5-R2 (zirkulaer verschobene Chartreihe je Person und Aufenthalt).

Verfahren der Schaetzung (unveraendert gegenueber AW-110/AW-242): Gitter -720..+720 min (5 min), naechster Chartwert im
selben Aufenthalt, Verlust = Median der absoluten Differenzen je Offset, Identifizierbarkeit: >= 3 vollstaendige EKG,
naheoptimale Zone (<= Minimum + 1/min) zusammenhaengend und <= 120 min breit, Minimum <= 5/min, Zone nicht am Rand.
Hier eigene Formulierung der Schaetzung und der Verschiebung.
"""
import numpy as np

GRID = np.arange(-720, 721, 5)
SHIFT_S = GRID * 60.0


def pick_matrix(t_ecg, tt, start=None, end=None):
    """Index des naechsten Chartwerts (Gleichstand: frueherer) je EKG und Offset; Form (n, K)."""
    tg = t_ecg[:, None] + SHIFT_S[None, :]
    k = np.searchsorted(tt, tg)
    left = np.clip(k - 1, 0, len(tt) - 1)
    right = np.clip(k, 0, len(tt) - 1)
    use_left = np.abs(tt[left] - tg) <= np.abs(tt[right] - tg)
    return np.where(use_left, left, right)


def circ_shift_values(tt, vv, shift_s):
    """Werte auf denselben Zeitpunkten tt: neuer Wert j = Wert der Originalreihe zur (zirkulaer) um shift_s verschobenen Zeit.

    Die Reihe wird als Kreis der Laenge T = Spanne + mittlerer Abstand aufgefasst; der naechste Originalpunkt zaehlt."""
    n = len(tt)
    u = tt - tt[0]
    dt = float(np.median(np.diff(u))) if n > 1 else 1.0
    T = float(u[-1] + dt)
    w = np.mod(u + shift_s, T)
    ext_u = np.r_[u, T]
    ext_i = np.r_[np.arange(n), 0]
    k = np.searchsorted(ext_u, w)
    left = np.clip(k - 1, 0, n)
    right = np.clip(k, 0, n)
    use_left = np.abs(ext_u[left] - w) <= np.abs(ext_u[right] - w)
    j = np.where(use_left, left, right)
    return vv[ext_i[j]], T


def estimate(V, h):
    """V (n,K), h (n,) -> dict(ok, offset, width, minimum)."""
    loss = np.median(np.abs(V - h[:, None]), axis=0)
    lo = float(loss.min())
    near = GRID[loss <= lo + 1]
    opts = GRID[np.isclose(loss, lo, rtol=0, atol=1e-10)]
    off = int(sorted(opts, key=lambda x: (abs(x), x))[0])
    width = int(near[-1] - near[0])
    connected = bool(np.all(np.diff(near) == 5))
    edge_free = bool(near[0] > -720 and near[-1] < 720)
    ok = bool(len(h) >= 3 and width <= 120 and lo <= 5 and edge_free and connected)
    return ok, off, width, lo


def shift_range(span_s, mode):
    """Zulaessiger Verschiebungsbereich [lo, hi] in Sekunden. mode: 'n1' (>= 24 h), 'n2' (>= 72 h), jeweils hoechstens Spanne/4."""
    base = {'n1': 24 * 3600.0, 'n2': 72 * 3600.0}[mode]
    lo = min(base, span_s / 4.0)
    return lo, span_s - lo


def draw_shift(rng, tt, mode):
    """Verschiebung in Sekunden. 'n3' = Vielfache von 24 h (erhaelt die Tagesphase, Diagnosevariante)."""
    span = float(tt[-1] - tt[0])
    if mode == 'n3':
        kmax = int(max(span // 86400.0 - 1, 2))
        return float(rng.integers(2, kmax + 1) * 86400.0)
    lo, hi = shift_range(span, mode)
    return float(rng.uniform(lo, hi))
