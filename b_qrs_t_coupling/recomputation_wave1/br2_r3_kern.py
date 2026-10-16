"""AW-241 B-R2/B-R3: Rechenkern (keine Datenzugriffe, keine Seiteneffekte beim Import).

Bivariate Regressionskalibrierung (Winkel A, Fréchet-Marker F) mit angenommener Fehlerkovarianz E, Szenarioraster (B-R2),
differenzielle und persistente Fehlermodelle (B-R3). Alle Groessen in Entwicklungs-SD-Einheiten der Spalten des B3-Designs.
Kern uebernommen und erweitert aus werkzeuge/b_test/aw198_run.py und aw211_common.py (Originale unveraendert).
"""
import numpy as np
from scipy.special import expit

TARGETS = (1.25, 1.0)
RA_SCAN = np.round(np.arange(1000, 49, -1) / 1000, 3)           # 1,000 ... 0,050
KAPPA = np.round(np.arange(100, 2001) / 100, 2)                 # 1,00 ... 20,00
PI_SCAN = np.round(np.arange(0, 100) / 100, 3)                  # 0,00 ... 0,99
B_COARSE = np.round(np.arange(0, 21) / 10, 2)                   # 0,0 ... 2,0 (Kontroll-SD)
STATUS = {1: "Kreuzung", 2: "Grenze vor Kreuzung", 3: "unzulaessig am Start", 4: "am Start unter Ziel", 5: "keine Kreuzung im Suchbereich"}


def wcov(x, w):
    c = x - np.average(x, axis=0, weights=w)
    return c.T @ (w[:, None] * c) / w.sum()


def pair_error(delta, w):
    c = delta - np.average(delta, axis=0, weights=w)
    return c.T @ (w[:, None] * c) / (w.sum() - 1) / 2


def components(z, y, w, logistic_fit, b=None, initial=None):
    """Logit-Fit und Kovarianzbausteine; wirft ValueError bei Nichtkonvergenz (Score-Kriterium wie AW-198)."""
    if b is None:
        b = logistic_fit(z, y, w, initial=initial) if initial is not None else logistic_fit(z, y, w)
    pr = expit(z @ b)
    score = np.max(np.abs(z.T @ (w * (pr - y)) + np.r_[0, np.full(z.shape[1] - 1, 1e-6)] * b)) / w.sum()
    if not np.isfinite(b).all() or score >= 1e-7:
        raise ValueError("nicht konvergiert")
    other = z[:, :-2]
    W = z[:, -2:]
    coef = np.linalg.solve(other.T @ (w[:, None] * other), other.T @ (w[:, None] * W))
    S = wcov(W - other @ coef, w)
    V = wcov(W, w)
    return {"b": b, "S": S, "V": V, "coef": coef, "u": S @ b[-2:]}


def corrected(c, E):
    """Korrigierte Marker- und Winkel-log-OR, kleinster Eigenwert von T=S-E, partielle latente Korrelation."""
    T = c["S"] - E
    lam = np.linalg.eigvalsh(T).min()
    if lam <= 1e-10:
        return np.nan, np.nan, lam, np.nan
    cw = np.linalg.solve(T, c["u"])
    return cw[1], cw[0], lam, T[0, 1] / np.sqrt(T[0, 0] * T[1, 1])


def rel_of(c, E):
    V = c["V"]
    return {"R_A": 1 - E[0, 0] / V[0, 0], "R_F": 1 - E[1, 1] / V[1, 1], "rho": E[0, 1] / np.sqrt(E[0, 0] * E[1, 1])}


def grid(c, RA, RF, RHO):
    """Korrigierter Marker-log-OR ueber Achsen (RF, rho, RA) mit Zulaessigkeit; Reliabilitaeten bezogen auf V (beobachtete Varianz)."""
    S, V, u = c["S"], c["V"], c["u"]
    RA = np.asarray(RA, float)[None, None, :]
    RF = np.asarray(RF, float)[:, None, None]
    RHO = np.asarray(RHO, float)[None, :, None]
    eA = (1 - RA) * V[0, 0]
    eF = (1 - RF) * V[1, 1]
    eAF = RHO * np.sqrt(eA * eF)
    T11, T22, T12 = np.broadcast_arrays(S[0, 0] - eA, S[1, 1] - eF, S[0, 1] - eAF)
    det = T11 * T22 - T12 ** 2
    lam = (T11 + T22 - np.sqrt((T11 - T22) ** 2 + 4 * T12 ** 2)) / 2
    ok = lam > 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        cF = (-T12 * u[0] + T11 * u[1]) / det
        cA = (T22 * u[0] - T12 * u[1]) / det
        pcorr = T12 / np.sqrt(T11 * T22)
    return {"logor": np.where(ok, cF, np.nan), "angle_logor": np.where(ok, cA, np.nan), "ok": ok, "pcorr": np.where(ok, pcorr, np.nan)}


def first_bad_scan(values_ok, values_lo, lt):
    """Index der ersten Abwaertskreuzung/Unzulaessigkeit entlang einer Achse; Status wie STATUS."""
    bad = (~values_ok) | (values_lo < lt)
    if not bad.any():
        return 5, None
    k = int(np.argmax(bad))
    if not values_ok[k]:
        return (3 if k == 0 else 2), k
    if k == 0:
        return 4, 0
    return 1, k


def kipp(c, RFv, RHOv, target):
    """Hoechste Winkelreliabilitaet R_A*, bei der die korrigierte OR beim Senken von R_A erstmals unter target faellt (je RF x rho-Zelle)."""
    g = grid(c, RA_SCAN, RFv, RHOv)
    L, ok = g["logor"], g["ok"]
    lt = np.log(target)
    nF, nR, _ = L.shape
    ra = np.full((nF, nR), np.nan)
    status = np.zeros((nF, nR), int)
    bound = np.full((nF, nR), np.nan)
    for i in range(nF):
        for j in range(nR):
            st, k = first_bad_scan(ok[i, j], np.where(ok[i, j], L[i, j], 0.0), lt)
            status[i, j] = st
            if st == 5:
                continue
            if st in (2, 3):
                bound[i, j] = RA_SCAN[k - 1] if k > 0 else np.nan
                continue
            if st == 4:
                ra[i, j] = 1.0
                continue
            x0, x1 = RA_SCAN[k - 1], RA_SCAN[k]
            y0, y1 = L[i, j, k - 1], L[i, j, k]
            ra[i, j] = x0 + (lt - y0) * (x1 - x0) / (y1 - y0)
            rest = ~ok[i, j, k:]
            bound[i, j] = RA_SCAN[k + int(np.argmax(rest)) - 1] if rest.any() else np.nan
    return {"ra": ra, "status": status, "bound": bound}


# ------------------------------------------------------------------ B-R3 (c): persistente Fehler, (b): Fehlervarianz bei Faellen

def kappa_paths(c, Ee, Ec, pc):
    """Korrigierter log-OR entlang kappa (alle KAPPA-Werte) fuer drei Pfade: nur Winkel, nur Marker, beide gleich.
    E(kA,kF) = (1-pc) Ee + pc D Ec D, D = diag(sqrt(kA), sqrt(kF)); Rueckgabe je Pfad (logor, ok)."""
    S, u = c["S"], c["u"]
    k = KAPPA
    out = {}
    for name, (ka, kf) in {"A": (k, np.ones_like(k)), "F": (np.ones_like(k), k), "AF": (k, k)}.items():
        dA, dF = np.sqrt(ka), np.sqrt(kf)
        E11 = (1 - pc) * Ee[0, 0] + pc * ka * Ec[0, 0]
        E22 = (1 - pc) * Ee[1, 1] + pc * kf * Ec[1, 1]
        E12 = (1 - pc) * Ee[0, 1] + pc * dA * dF * Ec[0, 1]
        T11, T22, T12 = S[0, 0] - E11, S[1, 1] - E22, S[0, 1] - E12
        det = T11 * T22 - T12 ** 2
        lam = (T11 + T22 - np.sqrt((T11 - T22) ** 2 + 4 * T12 ** 2)) / 2
        ok = lam > 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            cF = (-T12 * u[0] + T11 * u[1]) / det
        out[name] = (np.where(ok, cF, np.nan), ok)
    return out


def persist_paths(c, E, rho_p):
    """Zusatzfehler persistenter Art: e_p,v = pi_v * R_v * V_vv (R_v aus E); E_tot = E + E_p.
    Pfade: A (nur Winkel), F (nur Marker), AF (beide gleiches pi). Rueckgabe je Pfad (logor, ok) ueber PI_SCAN."""
    S, V, u = c["S"], c["V"], c["u"]
    RA = max(1 - E[0, 0] / V[0, 0], 0.0)
    RF = max(1 - E[1, 1] / V[1, 1], 0.0)
    p = PI_SCAN
    out = {}
    for name, (pa, pf) in {"A": (p, 0 * p), "F": (0 * p, p), "AF": (p, p)}.items():
        epA, epF = pa * RA * V[0, 0], pf * RF * V[1, 1]
        E11, E22 = E[0, 0] + epA, E[1, 1] + epF
        E12 = E[0, 1] + rho_p * np.sqrt(epA * epF)
        T11, T22, T12 = S[0, 0] - E11, S[1, 1] - E22, S[0, 1] - E12
        det = T11 * T22 - T12 ** 2
        lam = (T11 + T22 - np.sqrt((T11 - T22) ** 2 + 4 * T12 ** 2)) / 2
        ok = lam > 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            cF = (-T12 * u[0] + T11 * u[1]) / det
        out[name] = (np.where(ok, cF, np.nan), ok)
    return out


def tip_on_axis(axis, logor, ok, target):
    """Erste Abwaertskreuzung entlang einer geordneten Achse (Start = Achsenanfang, Wert dort >= Ziel erwartet).
    Rueckgabe (status, wert, grenze): wert = interpolierte Kreuzung, grenze = letzter zulaessiger Achsenwert vor Unzulaessigkeit."""
    lt = np.log(target)
    st, k = first_bad_scan(ok, np.where(ok, logor, 0.0), lt)
    if st == 5:
        return 5, np.nan, np.nan
    if st in (2, 3):
        return st, np.nan, float(axis[k - 1]) if k > 0 else np.nan
    if st == 4:
        return 4, float(axis[0]), np.nan
    x0, x1, y0, y1 = axis[k - 1], axis[k], logor[k - 1], logor[k]
    val = float(x0 + (lt - y0) * (x1 - x0) / (y1 - y0))
    rest = ~ok[k:]
    bound = float(axis[k + int(np.argmax(rest)) - 1]) if rest.any() else np.nan
    return 1, val, bound


def tip_shift(shift_eval, axis_coarse, target, iters=8):
    """Kippen entlang eines Verschiebungspfads b >= 0 mit Neuanpassung je Punkt.
    shift_eval(b) -> (logor, ok). Grobsuche auf axis_coarse, dann Bisektion im Klammerintervall.
    Rueckgabe (status, wert, grenze) wie tip_on_axis."""
    lt = np.log(target)
    prev_b = None
    for b in axis_coarse:
        lo, ok = shift_eval(float(b))
        bad = (not ok) or (lo < lt)
        if not bad:
            prev_b = float(b)
            continue
        if prev_b is None:
            return (3 if not ok else 4), (np.nan if not ok else float(b)), np.nan
        a, z = prev_b, float(b)
        for _ in range(iters):
            m = 0.5 * (a + z)
            lm, okm = shift_eval(m)
            if (not okm) or (lm < lt):
                z = m
            else:
                a = m
        lz, okz = shift_eval(z)
        if okz:
            # Kreuzung: lineare Interpolation im letzten Klammerintervall
            la, _ = shift_eval(a)
            val = a + (lt - la) * (z - a) / (lz - la) if lz != la else z
            return 1, float(val), np.nan
        return 2, np.nan, float(a)
    return 5, np.nan, np.nan
