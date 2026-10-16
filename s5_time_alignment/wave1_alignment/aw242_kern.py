"""AW-242 Kern: Herzfrequenz-Zeitausrichtung (S5), vektorisiert, Parity zu AW-110.

Nachtraegliche (post hoc) Nachrechnung. Keine Testpartition. Eingaben: audit.duckdb (AW-110, nur lesend).
Das Auswerteverfahren ist gegenueber AW-110 unveraendert (Gitter -720..+720 min, Schritt 5; naechster Referenzwert
innerhalb des Suchfensters, Gleichstand frueher; Referenz und Ziel strikt im selben Aufenthalt; Vollabdeckung an
allen Offsets; Identifizierbarkeit: >=3 vollstaendige EKG, naheoptimale Zone (<= Minimum+1/min) zusammenhaengend
und <=120 min breit, Minimum <=5/min, kein naheoptimaler Punkt am Rand).
"""
import numpy as np

D = np.arange(-720, 721, 5)
K = len(D)
SHIFT = D * 60.0
I0 = int(np.where(D == 0)[0][0])


def chart_slices(subj, stay):
    """Index-Bereiche je (Person, Aufenthalt) in nach (subj, stay, t) sortierten Arrays."""
    key_change = np.r_[True, (subj[1:] != subj[:-1]) | (stay[1:] != stay[:-1])]
    starts = np.flatnonzero(key_change)
    ends = np.r_[starts[1:], len(subj)]
    return {(int(subj[s]), int(stay[s])): (int(s), int(e)) for s, e in zip(starts, ends)}


def ecg_matrix(t, start, end, times, values, window=1800.0):
    """Referenzwerte V[K] und Abstaende dt[K] fuer ein EKG; times/values im Aufenthalt, sortiert.

    Rueckgabe: (V, dt, ok). V enthaelt den Referenzwert des naechsten Chartwerts je Offset (NaN wenn keiner
    im Aufenthalt). ok = Vollabdeckung (alle Offsets: Abstand <= window und Ziel im Aufenthalt).
    """
    targets = t + SHIFT
    inside = (times >= start) & (times <= end)
    tt = times[inside]
    vv = values[inside]
    if not len(tt):
        return None, None, False
    k = np.searchsorted(tt, targets)
    left = np.clip(k - 1, 0, len(tt) - 1)
    right = np.clip(k, 0, len(tt) - 1)
    pick = np.where(np.abs(tt[left] - targets) <= np.abs(tt[right] - targets), left, right)
    dt = np.abs(tt[pick] - targets)
    ok = bool(np.all((dt <= window) & (targets >= start) & (targets <= end)))
    return vv[pick], dt, ok


def estimate_from_loss(loss, n):
    """Schaetzung aus der Personenkurve; gibt Dict mit allen Teilkriterien zurueck."""
    lo = float(np.min(loss))
    near = D[loss <= lo + 1]
    opts = D[np.isclose(loss, lo, rtol=0, atol=1e-10)]
    offset = int(sorted(opts, key=lambda x: (abs(x), x))[0])
    width = int(near[-1] - near[0])
    connected = bool(np.all(np.diff(near) == 5))
    edge_free = bool(near[0] > -720 and near[-1] < 720)
    c_n = n >= 3
    c_w = width <= 120
    c_m = lo <= 5
    ok = bool(c_n and c_w and c_m and edge_free and connected)
    return dict(n=int(n), offset=offset, width=width, minimum=lo, ok=ok, c_n=bool(c_n), c_w=bool(c_w), c_m=bool(c_m),
                c_edge=edge_free, c_conn=connected, argmin_n=int(len(opts)), near_low=int(near[0]), near_high=int(near[-1]),
                tie_mid=float((opts[0] + opts[-1]) / 2.0))


def loss_curve(V, h):
    """V: (n,K) Referenzwerte, h: (n,) ECG-Frequenzen -> Median der absoluten Differenzen je Offset."""
    return np.median(np.abs(V - h[:, None]), axis=0)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def smd_cont(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / s) if s > 0 else float('nan')


def smd_bin(p1, p0):
    s = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0)
    return float((p1 - p0) / s) if s > 0 else float('nan')
