"""AW-203 / N3-V: Messkette v2 (Pipeline aw203-v2) fuer hohe Herzfrequenz.

Gebunden durch Nachtrag 2 zur Spezifikation N3 v1 (SPEZIFIKATION_N3_NACHTRAG2_ENTWURF.md). aw197_messung.py bleibt unveraendert
und wird fuer Einlesen, R-Zacken, Schlagauswahl und Fensterpruefung wiederverwendet; geaendert ist nur die Endpunktbestimmung.
Keine Fall- oder Personenkennungen in Ausgaben. Aufruf nur mit python -I.

Aenderungen gegenueber aw197-v1 (alle technisch, siehe Nachtrag 2 Abschnitt 2):
  1. T-Amplitude, T-Gipfel-Zeit, T-Steilheit und R-Amplitude aus einem eigenen R-ausgerichteten Medianschlag (200 ms vor bis
     0,75 RR nach der R-Zacke) mit direkter Gipfelsuche in einem mit sqrt(RR) skalierten Fenster. Keine Delineation je Schlag,
     kein wavelet-Delineator fuer diese Groessen.
  2. Basislinie = Median der 32 ms, die 8 ms vor dem QRS-Beginn des Medianschlags enden (PR-Isoelektrik); QRS-Beginn aus der
     Steigungsschwelle (10 % der maximalen Steigung). Ohne QRS-Beginn: 60 bis 30 ms vor der R-Zacke (wie v1 Modus 2).
  3. Fuer QRS-Dauer, PR, P-Amplitude, QT, JT, Tpeak-Tend bleibt der dwt-Delineator von v1 auf dem v1-Medianzyklus, mit mehr
     Wiederholungen im aneinandergelegten Zyklus (Mindestlaenge 2500 Abtastwerte statt fest 7 Wiederholungen), damit die
     Ausnahme "data length too small" (unter 2000 Abtastwerte) nicht mehr auftritt. Rest der v1-Definitionen unveraendert.
"""
import numpy as np
import scipy.signal as ss
import neurokit2 as nk
import aw197_messung as M
from aw197_messung import CFG, FS, _fill, detect_r, hf_beats, _endpoint_nan, pd  # noqa: F401

PIPELINE_VERSION = "aw203-v2"

V2 = dict(
    pre_s=0.200,            # Medianschlag: 200 ms vor der R-Zacke
    post_rr=0.75,           # Medianschlag: bis 0,75 RR nach der R-Zacke
    ts_k=0.18, ts_rr=0.25,  # T-Suchfenster Beginn: min(0,18*sqrt(RR), 0,25*RR) nach der R-Zacke
    te_k=0.50, te_rr=0.60,  # T-Suchfenster Ende:   min(0,50*sqrt(RR), 0,60*RR) nach der R-Zacke
    slope_k=0.12,           # T-Steilheit: Intervall T-Gipfel bis T-Gipfel + 0,12*sqrt(RR)
    thr_onset=0.10,         # QRS-Beginn: 10 % der maximalen Steigung
    min_r=0.10,             # mV, R-Amplitude nach Basislinie (wie v1)
    min_tile=2500,          # dwt: Mindestlaenge des aneinandergelegten Zyklus
)


def _tile_count(L):
    n = max(7, int(np.ceil(V2["min_tile"] / L)))
    return n if n % 2 == 1 else n + 1


def delineate_dwt(med, pre):
    """v1-Delineation (dwt) mit angepasster Wiederholungszahl. Gibt Fiducials als dict zurueck oder None."""
    L = len(med)
    ramp = np.linspace(med[:10].mean(), med[-10:].mean(), L)
    m0 = med - ramp
    n = _tile_count(L)
    m = n // 2
    tile = np.tile(m0, n)
    rp = np.array([i * L + pre for i in range(n)])
    try:
        _, w = nk.ecg_delineate(tile, rpeaks=rp, sampling_rate=FS, method="dwt", show=False)
    except Exception:
        return None, m0

    def g(key):
        if key not in w:
            return None
        v = w[key][m]
        if v is None or v != v:
            return None
        return int(v) - m * L
    return {k: g(k) for k in ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_R_Onsets", "ECG_R_Offsets",
                              "ECG_T_Peaks", "ECG_T_Offsets")}, m0


def dwt_endpoints(med, pre):
    """QRS, PR, P-Amplitude, QT, JT, Tpeak-Tend wie in v1 (Basislinie fuer die P-Amplitude wie v1 Modus 1/2)."""
    ep = {k: np.nan for k in ("qrs_ms", "pr_ms", "p_amp", "tpe_ms", "qt_ms", "jt_ms")}
    fid, m0 = delineate_dwt(med, pre)
    if fid is None:
        ep["dwt_fail"] = 1
        return ep
    L = len(med)
    ms = 1000.0 / FS
    Pon, Ppk, Poff = fid["ECG_P_Onsets"], fid["ECG_P_Peaks"], fid["ECG_P_Offsets"]
    Ron, Roff = fid["ECG_R_Onsets"], fid["ECG_R_Offsets"]
    Tpk, Toff = fid["ECG_T_Peaks"], fid["ECG_T_Offsets"]
    rpk = int(np.argmax(m0[max(0, pre - 20):pre + 21])) + max(0, pre - 20)
    if Ron is None or not (0 < Ron < rpk):
        Ron = None
    if Roff is not None and not (rpk < Roff < L):
        Roff = None
    if Roff is not None and Ron is not None and 40 <= (Roff - Ron) * ms <= 200:
        ep["qrs_ms"] = (Roff - Ron) * ms
    if Poff is not None and Ron is not None and Poff >= Ron:
        Poff = None
    if Pon is not None and Ppk is not None and Poff is not None and Ron is not None and 0 <= Pon < Ppk < Poff < Ron:
        pr = (Ron - Pon) * ms
        if 80 <= pr <= 400:
            ep["pr_ms"] = pr
    if Ron is not None:
        a = max(0, Ron - int(0.040 * FS))
        base = float(np.median(m0[a:max(a + 2, Ron - 4)]))
    else:
        a = max(0, rpk - int(0.060 * FS))
        base = float(np.median(m0[a:max(a + 2, rpk - int(0.030 * FS))]))
    if Ppk is not None:
        ep["p_amp"] = float(m0[Ppk] - base)
    if Tpk is not None and rpk + int(0.06 * FS) < Tpk < L and Toff is not None and Toff > Tpk:
        ep["tpe_ms"] = (Toff - Tpk) * ms
        if Ron is not None:
            ep["qt_ms"] = (Toff - Ron) * ms
        if Roff is not None:
            ep["jt_ms"] = (Toff - Roff) * ms
    return ep


def qrs_onset(m, pre):
    """QRS-Beginn im Medianschlag: letzter Abtastwert vor der groessten Steigung, an dem |Steigung| unter 10 % des Maximums liegt."""
    d = ss.savgol_filter(m, 11, 3, deriv=1, delta=1.0 / FS)
    a, b = max(0, pre - int(0.060 * FS)), pre + int(0.010 * FS)
    if b <= a + 3:
        return None, d
    ad = np.abs(d[a:b])
    i0 = a + int(np.argmax(ad))
    thr = V2["thr_onset"] * ad.max()
    j = i0
    lim = max(1, pre - int(0.140 * FS))
    while j > lim and abs(d[j]) >= thr:
        j -= 1
    if j <= lim:
        return None, d
    return j, d


def t_features(m, pre, rr):
    """T-Amplitude, T-Gipfel-Zeit (ms nach R), T-Steilheit, R-Amplitude, Basislinienmodus aus dem Medianschlag m (R bei Index pre)."""
    out = dict(t_amp=np.nan, t_pk_ms=np.nan, t_slope=np.nan, r_amp=np.nan, baseline_mode=np.nan, t_over_r=np.nan, fail=0)
    L = len(m)
    on, d = qrs_onset(m, pre)
    if on is not None and on - int(0.048 * FS) >= 0:
        a = on - int(0.040 * FS)
        base = float(np.median(m[max(0, a):max(a + 2, on - 4)]))
        out["baseline_mode"] = 1
    else:
        a = max(0, pre - int(0.060 * FS))
        base = float(np.median(m[a:max(a + 2, pre - int(0.030 * FS))]))
        out["baseline_mode"] = 2
    rpk = int(np.argmax(m[max(0, pre - 20):pre + 21])) + max(0, pre - 20)
    out["r_amp"] = float(m[rpk] - base)
    if out["r_amp"] < V2["min_r"]:
        out["fail"] = 1
        return out
    sq = np.sqrt(rr)
    ts = int(round(min(V2["ts_k"] * sq, V2["ts_rr"] * rr) * FS))
    te = int(round(min(V2["te_k"] * sq, V2["te_rr"] * rr) * FS))
    i_a, i_b = rpk + ts, min(rpk + te, L - 1)
    if i_b - i_a < 10:
        out["fail"] = 1
        return out
    y = ss.savgol_filter(m, 11, 3) - base      # 22 ms Glaettung (Savitzky-Golay, 11 Punkte, Grad 3)
    seg = y[i_a:i_b + 1]
    cand = []
    for sign in (1.0, -1.0):
        pk, _ = ss.find_peaks(sign * seg)        # innere lokale Extrema (Randpunkte zaehlen nicht)
        for p in pk:
            cand.append((abs(seg[p]), i_a + p))
    if not cand:
        out["fail"] = 1
        return out
    _, ip = max(cand)
    out["t_amp"] = float(y[ip])
    out["t_pk_ms"] = (ip - rpk) * 1000.0 / FS
    out["t_over_r"] = out["t_amp"] / out["r_amp"]
    j_b = min(L - 1, ip + int(round(V2["slope_k"] * sq * FS)))
    if j_b - ip >= 3:
        out["t_slope"] = float(d[ip:j_b + 1].min())
    return out


def measure_window_v2(x, off, t0, t1, hf_thr, splithalf=False, do_dwt=True):
    """Wie aw197_messung.measure_window (Fenster-, Rhythmus-, Rauschpruefung unveraendert), Endpunkte nach v2."""
    res = dict(valid=0, reason="", n_beats=0, n_rr=0, n_ecto=0, n_noise=0, n_acc=0, cover_s=0.0, hr=np.nan, rr_med=np.nan)
    i0 = int(round((t0 - off) * FS))
    i1 = int(round((t1 - off) * FS))
    pad = int(CFG["pad_s"] * FS)
    a, b = max(0, i0 - pad), min(len(x), i1 + pad)
    if i1 <= 0 or i0 >= len(x) or b - a < 20 * FS:
        res["reason"] = "kein_signal"
        return res
    raw = np.asarray(x[a:b], dtype=float)
    seg, nanm = _fill(raw)
    if seg is None or (~nanm[max(0, i0 - a):max(0, i1 - a)]).sum() < CFG["min_cover_s"] * FS:
        res["reason"] = "kein_signal"
        return res
    if np.ptp(seg) < 0.1:
        res["reason"] = "flach"
        return res
    try:
        R = detect_r(seg)
    except Exception:
        res["reason"] = "peaks_fail"
        return res
    lo, hi = i0 - a, i1 - a
    if len(R) < 5:
        res["reason"] = "zu_wenig_schlaege"
        return res
    rr = np.diff(R) / FS
    loc = pd.Series(rr).rolling(CFG["med_n"], center=True, min_periods=3).median().to_numpy()
    nb = len(R)
    prev_ok = np.zeros(nb, bool)
    next_ok = np.zeros(nb, bool)
    in_range_prev = np.zeros(nb, bool)
    in_range_next = np.zeros(nb, bool)
    rr_in = (rr >= CFG["rr_min"]) & (rr <= CFG["rr_max"])
    reg = np.abs(rr - loc) <= CFG["ecto"] * loc
    in_range_next[:-1] = rr_in
    in_range_prev[1:] = rr_in
    next_ok[:-1] = reg
    prev_ok[1:] = reg
    inwin = (R >= lo) & (R <= hi)
    res["n_beats"] = int(inwin.sum())
    if res["n_beats"] < 5:
        res["reason"] = "zu_wenig_schlaege"
        return res
    rr_ok = in_range_prev & in_range_next
    ecto_ok = prev_ok & next_ok
    ecto_bad = inwin & rr_ok & ~ecto_ok
    res["n_rr"] = int((inwin & ~rr_ok).sum())
    res["n_ecto"] = int(ecto_bad.sum())
    hfv, hf = hf_beats(seg, R)
    a_s, b_s = int(CFG["hf_pre"] * FS), int(CFG["hf_post"] * FS)
    noise = np.zeros(nb, bool)
    flat_n = int(CFG["flat_ms"] / 1000 * FS)
    for j in range(nb):
        if not inwin[j]:
            continue
        s = seg[max(0, R[j] - a_s):R[j] + b_s]
        nm = nanm[max(0, R[j] - a_s):R[j] + b_s]
        if nm.any() or hfv[j] > hf_thr or np.abs(s).max() > CFG["sat_abs"]:
            noise[j] = True
            continue
        dz = (np.diff(s) == 0)
        if dz.any():
            c = np.convolve(dz.astype(int), np.ones(flat_n, int), "valid")
            if (c >= flat_n).any():
                noise[j] = True
    res["n_noise"] = int((inwin & rr_ok & ecto_ok & noise).sum())
    acc = inwin & rr_ok & ecto_ok & ~noise
    res["n_acc"] = int(acc.sum())
    irr_frac = (res["n_ecto"] + res["n_rr"]) / max(res["n_beats"], 1)
    res["irr_frac"] = float(irr_frac)
    if res["n_acc"] < 10:
        res["reason"] = "zu_wenig_auswertbar"
        return res
    rr_prev = np.empty(nb)
    rr_prev[0] = np.nan
    rr_prev[1:] = rr
    rr_med = float(np.median(rr_prev[acc]))
    res["rr_med"] = rr_med
    res["hr"] = 60.0 / rr_med
    res["cover_s"] = float(res["n_acc"] * rr_med)
    if irr_frac > CFG["irr_frac"]:
        res["reason"] = "unregelmaessig"
        return res
    if res["cover_s"] < CFG["min_cover_s"]:
        res["reason"] = "zu_wenig_signal"
        return res
    d = np.diff(seg) * FS
    spike = 0
    idxa = np.where(acc)[0]
    for j in idxa:
        s = d[max(0, R[j] - 50):R[j] + 5]
        if len(s) > 6:
            up = np.where(s > CFG["pace_slew"])[0]
            dn = np.where(s < -CFG["pace_slew"])[0]
            if len(up) and len(dn) and (np.abs(up[:, None] - dn[None, :]) <= 3).any() and \
                    (np.abs(s).max() > 2 * CFG["pace_slew"]):
                spike += 1
    res["pace_frac"] = float(spike / len(idxa))
    if res["pace_frac"] > CFG["pace_frac"]:
        res["reason"] = "schrittmacher"
        return res
    # Medianzyklus v1 (Gueltigkeitspruefung QRS-Polaritaet und -Amplitude, dwt-Endpunkte)
    L = int(round(rr_med * FS))
    pre = int(round(CFG["beat_pre"] * L))
    sel = [r for r in R[acc] if r - pre >= 0 and r - pre + L <= len(seg)]
    if len(sel) < 10:
        res["reason"] = "zu_wenig_auswertbar"
        return res
    beats = np.array([seg[r - pre:r - pre + L] for r in sel])
    med = np.median(beats, axis=0)
    seg_q = med[max(0, pre - 30):pre + 31]
    b0 = np.median(med[max(0, pre - 100):max(1, pre - 40)]) if pre > 100 else 0.0
    if abs(seg_q.min() - b0) > abs(seg_q.max() - b0):
        res["reason"] = "qrs_negativ"
        return res
    if seg_q.max() - b0 < CFG["min_qrs_amp"]:
        res["reason"] = "qrs_klein"
        return res
    res["valid"] = 1
    # Medianschlag v2
    pre2 = int(round(V2["pre_s"] * FS))
    post2 = int(round(V2["post_rr"] * rr_med * FS))
    sel2 = [r for r in R[acc] if r - pre2 >= 0 and r + post2 + 1 <= len(seg)]
    if len(sel2) < 10:
        res["fail"] = 1
        res["t_reason"] = "zu_wenig_schlaege_v2"
    else:
        med2 = np.median(np.array([seg[r - pre2:r + post2 + 1] for r in sel2]), axis=0)
        tf = t_features(med2, pre2, rr_med)
        res.update(tf)
        if splithalf:
            for h, nm_ in ((0, "a"), (1, "b")):
                bh = np.array([seg[r - pre2:r + post2 + 1] for r in sel2])[h::2]
                if len(bh) >= 8:
                    e2 = t_features(np.median(bh, axis=0), pre2, rr_med)
                    for k in ("t_amp", "t_slope"):
                        res[f"{k}_{nm_}"] = e2.get(k, np.nan)
    if do_dwt:
        ep = dwt_endpoints(med, pre)
        res.update(ep)
        if splithalf:
            for h, nm_ in ((0, "a"), (1, "b")):
                bh = beats[h::2]
                if len(bh) >= 8:
                    e2 = dwt_endpoints(np.median(bh, axis=0), pre)
                    for k in ("qrs_ms", "pr_ms", "p_amp"):
                        res[f"{k}_{nm_}"] = e2.get(k, np.nan)
    return res
