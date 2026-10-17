"""AW-197 / N3-V: Messbibliothek (Ableitung II, Medianschlag, neurokit2 dwt).

Gebunden durch Spezifikation N3 v1 (SHA-256 5324ea55...), Abschnitte 4 und 15.
Keine Fall- oder Personenkennungen in Ausgaben dieser Datei; Fallnummern nur in E:-Dateien der Treiber.
Aufruf nur mit python -I (Signaldateien sind unvertrauenswuerdig).

Pipeline-Version aw197-v1. Aenderungen am Delineator nach der Messung von Expositionsfaellen sind verboten
(Spezifikation Abschnitt 4); jede Aenderung vorher im Schritt-1-Bericht festhalten.
"""
import warnings
import numpy as np
import pandas as pd
import scipy.signal as ss
import neurokit2 as nk
import vitaldb

warnings.filterwarnings("ignore")

PIPELINE_VERSION = "aw197-v1"
FS = 500
ZERO = 4102444800.0  # 2100-01-01 00:00:00 UTC = Zeitnull der VitalDB-Fallzeit (EVENT 'Case started')

CFG = dict(
    rr_min=0.3, rr_max=2.0,            # s, Spezifikation Abschnitt 4
    ecto=0.20,                         # RR-Abweichung vom lokalen Median
    med_n=11,                          # lokaler RR-Median ueber 11 Intervalle
    min_cover_s=60.0,                  # auswertbares Signal je Fenster
    irr_frac=0.30,                     # Fenster unregelmaessig, wenn mehr als 30 % der Schlaege abweichen
    beat_pre=0.45, beat_post=0.55,     # Medianzyklus in Anteilen des medianen RR
    hf_hp=50.0,                        # Hz, Hochpass fuer Rauschenergie (Kauter)
    hf_pre=0.25, hf_post=0.45,         # s um die R-Zacke
    sat_abs=4.5,                       # mV, Saettigung
    flat_ms=40.0,                      # Flachlinie: mehr als 40 ms ohne Aenderung
    pace_slew=300.0,                   # mV/s, Schrittmacherspike (Schwelle je Abtastwert)
    pace_frac=0.30,
    min_qrs_amp=0.15,                  # mV, Medianschlag-Amplitude (R minus Basis)
    pad_s=3.0,
)


def ecg_filter_desc(seg):
    """Spektrale Beschreibung zur Dokumentation des Monitorfilters (Spezifikation 15.1)."""
    seg = np.nan_to_num(seg - np.nanmean(seg))
    f, P = ss.welch(seg, FS, nperseg=16384)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(P[m].sum()) if m.any() else np.nan

    lf = band(0.02, 0.25)
    card = band(0.7, 3.0)
    mid = band(18.0, 35.0)
    hf = band(45.0, 57.0)
    m = (f >= 0.06) & (f <= 0.40) & (P > 0)
    slope = float(np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0]) if m.sum() >= 6 else np.nan
    return dict(r_lf=np.log10(lf / card + 1e-12), r_hf=np.log10(hf / mid + 1e-12), sl_lf=slope)


def load_case(path, with_numeric=True, extra_numeric=()):
    """Liest EKG II und optional Begleitspuren aus einer .vital-Datei. Gibt dict zurueck."""
    want = ["SNUADC/ECG_II"]
    num = ["Solar8000/BT", "Solar8000/HR"] + list(extra_numeric) if with_numeric else []
    vf = vitaldb.VitalFile(path, want + num + ["SNUADC/ECG_V5", "EVENT"])
    out = dict(off=float(vf.dtstart - ZERO), dur=float(vf.dtend - vf.dtstart))
    trk = vf.trks.get("SNUADC/ECG_II")
    if trk is None:
        out["x"] = None
        out["meta"] = dict(has_ecg_ii=False)
        return out
    x = vf.get_track_samples("SNUADC/ECG_II", 1 / FS).astype(np.float32)
    out["x"] = x
    out["x5"] = (vf.get_track_samples("SNUADC/ECG_V5", 1 / FS).astype(np.float32)
                 if "SNUADC/ECG_V5" in vf.trks and vf.trks["SNUADC/ECG_V5"].recs else None)
    devtype = vf.devs["SNUADC"].type if "SNUADC" in vf.devs else None
    sol = vf.devs["Solar8000"].type if "Solar8000" in vf.devs else None
    out["meta"] = dict(
        has_ecg_ii=True, has_v5=("SNUADC/ECG_V5" in vf.trks), srate=float(trk.srate), unit=trk.unit,
        gain=float(trk.gain), mindisp=float(trk.mindisp), maxdisp=float(trk.maxdisp),
        adc=devtype, monitor=sol, has_st_ii=False, n_recs=len(trk.recs),
        nan_frac=float(np.isnan(x).mean()),
    )
    ev = []
    if "EVENT" in vf.trks:
        ev = [(float(r["dt"] - ZERO), str(r["val"])) for r in vf.trks["EVENT"].recs]
    out["events"] = ev
    numerics = {}
    for n in num:
        t = vf.trks.get(n)
        if t is None or not t.recs:
            continue
        dt = np.array([r["dt"] for r in t.recs], float) - ZERO
        val = np.array([r["val"] for r in t.recs], float)
        numerics[n] = (dt, val)
    out["numerics"] = numerics
    return out


def numeric_window(numerics, name, t0, t1, lo=None, hi=None):
    if name not in numerics:
        return np.nan
    dt, v = numerics[name]
    m = (dt >= t0) & (dt <= t1) & np.isfinite(v)
    if lo is not None:
        m &= v >= lo
    if hi is not None:
        m &= v <= hi
    return float(v[m].mean()) if m.sum() >= 3 else np.nan


def _fill(seg):
    nanm = ~np.isfinite(seg)
    if nanm.all():
        return None, nanm
    if nanm.any():
        idx = np.arange(len(seg))
        seg = np.interp(idx, idx[~nanm], seg[~nanm])
    return seg.astype(float), nanm


def detect_r(seg):
    cl = nk.ecg_clean(seg, sampling_rate=FS, method="neurokit", powerline=60)
    _, info = nk.ecg_peaks(cl, sampling_rate=FS, method="kalidas2017")
    R = np.asarray(info["ECG_R_Peaks"], dtype=int)
    # Verfeinerung auf das lokale Maximum des Rohsignals (+-40 ms)
    out = []
    for r in R:
        a, b = max(0, r - 20), min(len(seg), r + 21)
        out.append(a + int(np.argmax(seg[a:b])))
    return np.unique(np.array(out, dtype=int))


_SOS_HF = ss.butter(4, CFG["hf_hp"], "hp", fs=FS, output="sos")


def hf_beats(seg, R):
    """HF-RMS je Schlag (>50 Hz) um die R-Zacke."""
    hf = ss.sosfiltfilt(_SOS_HF, seg)
    a = int(CFG["hf_pre"] * FS)
    b = int(CFG["hf_post"] * FS)
    out = np.full(len(R), np.nan)
    for i, r in enumerate(R):
        lo, hi = max(0, r - a), min(len(hf), r + b)
        out[i] = np.sqrt(np.mean(hf[lo:hi] ** 2))
    return out, hf


def _endpoint_nan():
    return {k: np.nan for k in ("t_amp", "t_slope", "qrs_ms", "pr_ms", "p_amp", "tpe_ms", "qt_ms", "jt_ms",
                                "t_over_r", "r_amp", "t_pk_ms", "baseline_mode")}


def delineate_median(med, pre):
    """Delineation des Medianzyklus (Laenge L, R bei Index pre). Gibt Endpunkt-dict zurueck."""
    ep = _endpoint_nan()
    L = len(med)
    ramp = np.linspace(med[:10].mean(), med[-10:].mean(), L)
    m0 = med - ramp
    n, m = 7, 3
    tile = np.tile(m0, n)
    rp = np.array([i * L + pre for i in range(n)])
    try:
        _, w = nk.ecg_delineate(tile, rpeaks=rp, sampling_rate=FS, method="dwt", show=False)
    except Exception:
        ep["fail"] = 1
        return ep

    def g(key):
        if key not in w:
            return None
        v = w[key][m]
        if v is None or v != v:
            return None
        return int(v) - m * L

    Pon, Ppk, Poff = g("ECG_P_Onsets"), g("ECG_P_Peaks"), g("ECG_P_Offsets")
    Ron, Roff = g("ECG_R_Onsets"), g("ECG_R_Offsets")
    Tpk, Toff = g("ECG_T_Peaks"), g("ECG_T_Offsets")
    ms = 1000.0 / FS
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
    # Basislinie (isoelektrische Linie im PR-Segment): letzte 32 ms der PR-Strecke direkt vor dem QRS-Beginn
    # (Modus 1, immer wenn QRS-Beginn bestimmt ist, damit die Regel zwischen Fenstern nicht wechselt);
    # Modus 2: 60 bis 30 ms vor der R-Zacke, wenn der QRS-Beginn fehlt.
    if Ron is not None:
        a = max(0, Ron - int(0.040 * FS))
        base = float(np.median(m0[a:max(a + 2, Ron - 4)]))
        ep["baseline_mode"] = 1
    else:
        a = max(0, rpk - int(0.060 * FS))
        base = float(np.median(m0[a:max(a + 2, rpk - int(0.030 * FS))]))
        ep["baseline_mode"] = 2
    ep["r_amp"] = float(m0[rpk] - base)
    if ep["r_amp"] < 0.1:
        ep["fail"] = 1
        return ep
    if Ppk is not None:  # P-Amplitude am P-Gipfel des Delineators (Fehlerbehebung: vorher Absturz bei fehlendem P-Gipfel)
        ep["p_amp"] = float(m0[Ppk] - base)
    if Tpk is not None and rpk + int(0.06 * FS) < Tpk < L:
        ep["t_amp"] = float(m0[Tpk] - base)
        ep["t_pk_ms"] = (Tpk - pre) * ms
        ep["t_over_r"] = ep["t_amp"] / ep["r_amp"]
        d = ss.savgol_filter(m0, 11, 3, deriv=1, delta=1.0 / FS)
        if Toff is not None and Toff > Tpk:
            ep["tpe_ms"] = (Toff - Tpk) * ms
            if Ron is not None:
                ep["qt_ms"] = (Toff - Ron) * ms
            if Roff is not None:
                ep["jt_ms"] = (Toff - Roff) * ms
            seg = d[Tpk:Toff + 1]
            if len(seg) >= 3:
                ep["t_slope"] = float(seg.min())
    else:
        ep["fail"] = 1
    return ep


def measure_window(x, off, t0, t1, hf_thr, splithalf=False):
    """Misst ein Fenster [t0, t1] (Sekunden relativ zu Zeitnull). x: ECG-Array, Index 0 = Dateibeginn (off)."""
    res = dict(valid=0, reason="", n_beats=0, n_rr=0, n_ecto=0, n_noise=0, n_acc=0, cover_s=0.0, hr=np.nan,
               rr_med=np.nan)
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
    # Schlaege im Fenster (Zeitpunkt relativ zum Segment)
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
    # Rauschen
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
    # Nicht-Sinusrhythmus: grosser Anteil unregelmaessiger Intervalle
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
    # Schrittmacher: schnelle Doppelflanke vor der R-Zacke
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
    # Medianzyklus
    L = int(round(rr_med * FS))
    pre = int(round(CFG["beat_pre"] * L))
    sel = [r for r in R[acc] if r - pre >= 0 and r - pre + L <= len(seg)]
    if len(sel) < 10:
        res["reason"] = "zu_wenig_auswertbar"
        return res
    beats = np.array([seg[r - pre:r - pre + L] for r in sel])
    med = np.median(beats, axis=0)
    # QRS-Polaritaet (Ableitung II: positive R erwartet)
    seg_q = med[max(0, pre - 30):pre + 31]
    b0 = np.median(med[max(0, pre - 100):max(1, pre - 40)]) if pre > 100 else 0.0
    if abs(seg_q.min() - b0) > abs(seg_q.max() - b0):
        res["reason"] = "qrs_negativ"
        return res
    if seg_q.max() - b0 < CFG["min_qrs_amp"]:
        res["reason"] = "qrs_klein"
        return res
    res["valid"] = 1
    ep = delineate_median(med, pre)
    for k, v in ep.items():
        res[k] = v
    if splithalf:
        sel = np.array(sel)
        for h, nm_ in ((0, "a"), (1, "b")):
            bh = beats[h::2]
            if len(bh) >= 8:
                e2 = delineate_median(np.median(bh, axis=0), pre)
                for k in ("t_amp", "t_slope", "qrs_ms", "pr_ms", "p_amp"):
                    res[f"{k}_{nm_}"] = e2.get(k, np.nan)
    return res
