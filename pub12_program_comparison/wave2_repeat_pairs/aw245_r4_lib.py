# -*- coding: utf-8 -*-
"""aw245_r4_lib.py (version 1.0): building blocks for the signal test of the aVL/aVF labelling in PTB-XL+ (numpy only).

Idea: the ECGDeli annotation file of a lead contains, per beat, the R, Q and S peak of that lead. If an annotation file belongs to a signal, the R peak lies
on a local maximum and the Q and S peaks on local minima of that signal (within +-w samples). For each annotation lead l and each signal lead y the
share of peaks that meet this condition is determined (12 x 12 matrix per recording). The ten unsuspected leads serve as calibration (annotation l against signal l is known
to be correct); the null distribution "no information" arises by combining the annotations of one recording with the signals of another recording.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
I_AVL, I_AVF = 4, 5
N_SAMPLES = 5000


def lies_dat(pfad_ohne_endung):
    """PTB-XL records500: format 16, 12 channels, little endian, interleaved; returns int16 (5000, 12)."""
    b = np.fromfile(str(pfad_ohne_endung) + ".dat", dtype="<i2")
    if b.size != N_SAMPLES * 12:
        raise ValueError("unexpected length %d" % b.size)
    return b.reshape(N_SAMPLES, 12)


def lokale_extrema(x, w):
    """x (n, 12): matrices (ismax, ismin): value >= resp. <= all values in the window +-w (edges False)."""
    n = x.shape[0]
    x = x.astype(np.int32)
    mx = np.ones_like(x, dtype=bool)
    mn = np.ones_like(x, dtype=bool)
    for k in range(1, w + 1):
        for s in (k, -k):
            sh = np.roll(x, s, axis=0)
            mx &= x >= sh
            mn &= x <= sh
    mx[:w] = False
    mx[-w:] = False
    mn[:w] = False
    mn[-w:] = False
    return mx, mn


def passung(ismax, ismin, R, Q, S, w):
    """Share of matching peaks: matrices (12, 12) for R, Q, S (annotation lead x signal lead) and number of valid peaks (12, 3).

    R, Q, S: (beats, 12) sample indices, -1 = missing. A peak counts only if w <= index < n - w.
    """
    n = ismax.shape[0]
    out = np.full((3, 12, 12), np.nan)
    cnt = np.zeros((12, 3), dtype=np.int32)
    for t, (A, tab) in enumerate(((R, ismax), (Q, ismin), (S, ismin))):
        for l in range(12):
            idx = A[:, l]
            ok = (idx >= w) & (idx < n - w)
            cnt[l, t] = int(ok.sum())
            if ok.any():
                out[t, l, :] = tab[idx[ok], :].mean(axis=0)
    return out, cnt


def dominante_passung(x, R, H=15, tol=1):
    """Share of beats in which the dominant deflection (absolute deviation from the window median, window +-H around the annotated R peak) of signal y
    lies at most tol samples from the annotated R peak of the annotation lead l. Returns M (12 annotation x 12 signal) and number of valid beats (12,).
    R: (beats, 12) indices, -1 = missing. x: (n, 12)."""
    n = x.shape[0]
    xf = x.astype(np.float64)
    M = np.full((12, 12), np.nan)
    cnt = np.zeros(12, dtype=np.int32)
    ar = np.arange(-H, H + 1)
    for l in range(12):
        idx = R[:, l]
        ok = (idx >= H) & (idx < n - H)
        cnt[l] = int(ok.sum())
        if not ok.any():
            continue
        idx = idx[ok]
        win = xf[idx[:, None] + ar[None, :], :]  # (nb, 2H+1, 12)
        base = np.median(win, axis=1, keepdims=True)
        dom = np.argmax(np.abs(win - base), axis=1) - H  # (nb, 12)
        M[l, :] = np.mean(np.abs(dom) <= tol, axis=0)
    return M, cnt


def gesamt(m3):
    """Mean of the three peak types (nanmean) -> (12, 12)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(m3, axis=0)


def d_swap(M, a=I_AVL, b=I_AVF):
    """Own name minus foreign name: (M[a,a] + M[b,b]) - (M[a,b] + M[b,a]); positive: the annotation fits the signal of the same name."""
    return (M[a, a] + M[b, b]) - (M[a, b] + M[b, a])


def klassifiziere(D, tau):
    """0 = as named (D > tau), 1 = exchanged (D < -tau), otherwise 2 = unclear; NaN stays 3 = no data."""
    D = np.asarray(D, float)
    return np.where(np.isnan(D), 3, np.where(D > tau, 0, np.where(D < -tau, 1, 2)))
