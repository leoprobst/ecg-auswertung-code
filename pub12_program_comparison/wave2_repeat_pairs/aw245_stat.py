# -*- coding: utf-8 -*-
"""aw245_stat.py (version 1.0): person-cluster bootstrap with weight matrices and summary measures for measurement pairs (numpy only).

All measures are computed for the point estimate (row 0, all weights 1) and B draws (rows 1..B) at the same time; common draws for all measures.
The unit of the draw is the person; a person with several pairs/recordings carries the weights over to all of their units.
"""
from __future__ import annotations

import numpy as np

QRS_SCHW = (120, 130, 150)
QTC_SCHW = (450, 470, 500)


def person_gewichte(pid, B, seed):
    """Weight matrix (B+1, n_units): row 0 = ones, rows 1..B = number of times the person of the unit was drawn."""
    pid = np.asarray(pid)
    up, inv = np.unique(pid, return_inverse=True)
    n = len(up)
    rng = np.random.default_rng(seed)
    W = np.empty((B + 1, len(pid)), dtype=np.float64)
    W[0] = 1.0
    for b in range(B):
        s = rng.integers(0, n, n)
        c = np.bincount(s, minlength=n).astype(np.float64)
        W[b + 1] = c[inv]
    return W


def wquantil(x, W, q):
    """Weighted lower quantile per row of W (value at which the cumulative weight sum is >= q * N). x: (n,) without NaN."""
    o = np.argsort(x, kind="stable")
    cs = np.cumsum(W[:, o], axis=1)
    N = cs[:, -1]
    j = (cs >= q * N[:, None] - 1e-9).argmax(axis=1)
    return np.asarray(x)[o][j]


def klasse(x, schw):
    x = np.asarray(x, float)
    return sum((x >= t).astype(int) for t in schw)


def paar_kennzahlen(a, b, W, schw, voll=True):
    """Summary measures for value pairs (a, b) (e.g. first and second recording or two programs). Returns dict name -> array (B+1,).

    D<t>: share on different sides of the threshold t; wechsel: different class; hoeher: class of b higher than that of a;
    bias = mean(b - a); sd, rc (= 1.96 sd of the difference), mittel_abs, median_abs, p95_abs, icc (ICC(2,1)), ccc, n.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    v = np.isfinite(a) & np.isfinite(b)
    a0, b0 = a[v], b[v]
    Wv = W[:, v]
    N = Wv.sum(axis=1)
    out = {"n": N}
    d = b0 - a0
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = (Wv @ d) / N
        out["bias"] = bias
        out["sd"] = np.sqrt(np.maximum(((Wv @ (d * d)) - N * bias ** 2) / (N - 1), 0.0))
        out["rc"] = 1.96 * out["sd"]
        out["mittel_abs"] = (Wv @ np.abs(d)) / N
        for t in schw:
            out["D%d" % t] = (Wv @ ((a0 >= t) != (b0 >= t)).astype(float)) / N
        ca, cb = klasse(a0, schw), klasse(b0, schw)
        out["wechsel"] = (Wv @ (ca != cb).astype(float)) / N
        out["hoeher"] = (Wv @ (cb > ca).astype(float)) / N
        if voll:
            out["median_abs"] = wquantil(np.abs(d), Wv, 0.5)
            out["p95_abs"] = wquantil(np.abs(d), Wv, 0.95)
            ma, mb = (Wv @ a0) / N, (Wv @ b0) / N
            saa, sbb, sab = (Wv @ (a0 * a0)) / N, (Wv @ (b0 * b0)) / N, (Wv @ (a0 * b0)) / N
            dbar = ma - mb
            d2 = saa - 2 * sab + sbb
            mbar = (ma + mb) / 2
            m2 = (saa + 2 * sab + sbb) / 4
            msr = 2.0 * N * (m2 - mbar ** 2) / (N - 1.0)
            mse = N * (d2 - dbar ** 2) / 2.0 / (N - 1.0)
            msc = N * dbar ** 2 / 2.0
            out["icc"] = (msr - mse) / (msr + mse + 2.0 * (msc - mse) / N)
            va, vb, cab = saa - ma ** 2, sbb - mb ** 2, sab - ma * mb
            out["ccc"] = 2 * cab / (va + vb + (ma - mb) ** 2)
            out["sd_a"] = np.sqrt(np.maximum(va * N / (N - 1), 0.0))
            out["sd_b"] = np.sqrt(np.maximum(vb * N / (N - 1), 0.0))
    return out


def ci(v):
    v = np.asarray(v, float)
    x = v[1:]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(v[0]), float("nan"), float("nan")
    return float(v[0]), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
