# -*- coding: utf-8 -*-
"""aw245_r10.py (version 1.0): EchoNext bound D >= ln(2/sqrt(3)) = 0.144, numerical counter-check (no data, no EchoNext waveforms).

Derivation (in the report): assumptions (V1) the Einthoven/Goldberger relations hold exactly (III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2), (V2) each of the six
leads is brought to equal variance with its own constant scale (scale = 1 / standard deviation of the data set), (V3) D = largest absolute value of the logarithm of the ratio
(fitted coefficient / expected coefficient) of the eight coefficients of III, aVR, aVL, aVF on I and II. Then the ratio for lead x on lead y equals sigma_y / sigma_x.
Proof: aVL^2 + 2 aVR^2 = 1.5 sI^2 + 0.75 sII^2 and aVF^2 + 2 aVR^2 = 1.5 sII^2 + 0.75 sI^2 (variances, the correlation cancels), sum: aVL^2 + aVF^2 + 4 aVR^2 = 2.25 (sI^2 + sII^2).
If all ratios sigma_y/sigma_x lie in [e^-D, e^D], then x >= M e^-D (M = max(sI, sII)) holds for each of the three derived leads, hence 2.25 (sI^2 + sII^2) >= 6 M^2 e^(-2D), and
sI^2 + sII^2 <= 2 M^2 gives e^(-2D) <= 0.75, i.e. D >= 0.5 ln(4/3) = ln(2/sqrt(3)) = 0.1438. Equality only for sI = sII, rho = 0.5.
This script checks: (1) minimum of D numerically over (sII/sI, rho), (2) the identities, (3) counter-examples without variance standardisation, (4) prediction of the eight observed ratios
from sI/sII = 1.034 and rho = 0.54, (5) effect of median filter and clipping on synthetic signals (Einthoven-consistent, standardisation with data set constants).
Usage: python -I -B aw245_r10.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
OUT = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw245")))
OUT.mkdir(parents=True, exist_ok=True)
BOUND = np.log(2 / np.sqrt(3))
ERG = {}


def sds(p, q, rho):
    """Standard deviations of I, II, III, aVR, aVL, aVF for sI = p, sII = q, correlation rho."""
    v3 = p * p + q * q - 2 * rho * p * q
    vr = (p * p + q * q + 2 * rho * p * q) / 4
    vl = p * p + q * q / 4 - rho * p * q
    vf = q * q + p * p / 4 - rho * p * q
    return np.array([p, q, np.sqrt(v3), np.sqrt(vr), np.sqrt(vl), np.sqrt(vf)])


def ratios(s):
    """eight ratios sigma_y / sigma_x: (III, aVR, aVL, aVF) x (I, II) in the order of the supplementary table (III/I, III/II, aVR/I, aVR/II, aVL/I, aVL/II, aVF/I, aVF/II)."""
    out = []
    for x in (2, 3, 4, 5):
        for y in (0, 1):
            out.append(s[y] / s[x])
    return np.array(out)


def d_von(p, q, rho):
    return np.max(np.abs(np.log(ratios(sds(p, q, rho)))))


def main():
    # (1) numerical
    best = (9, None)
    for r in np.exp(np.linspace(-1.0, 1.0, 401)):
        for rho in np.linspace(-0.95, 0.95, 381):
            d = d_von(1.0, r, rho)
            if d < best[0]:
                best = (d, (r, rho))
    ERG["minimum_D_numerisch"] = best[0]
    ERG["minimum_bei_sII_durch_sI"], ERG["minimum_bei_rho"] = float(best[1][0]), float(best[1][1])
    ERG["schranke_ln_2_durch_wurzel3"] = float(BOUND)
    ERG["minimum_gleich_schranke"] = bool(abs(best[0] - BOUND) < 1e-3)
    # (2) random check of the inequality and of the identity
    rng = np.random.default_rng(245)
    verletzt = 0
    ident_max = 0.0
    kleinste_reserve = 9.0
    for _ in range(200000):
        p, q = np.exp(rng.normal(0, 0.7, 2))
        rho = rng.uniform(-0.99, 0.99)
        s = sds(p, q, rho)
        d = np.max(np.abs(np.log(ratios(s))))
        verletzt += d < BOUND - 1e-12
        kleinste_reserve = min(kleinste_reserve, d - BOUND)
        ident_max = max(ident_max, abs(s[4] ** 2 + s[5] ** 2 + 4 * s[3] ** 2 - 2.25 * (p * p + q * q)))
    ERG["zufall_n"] = 200000
    ERG["zufall_verletzungen"] = int(verletzt)
    ERG["zufall_kleinste_reserve_D_minus_schranke"] = float(kleinste_reserve)
    ERG["identitaet_max_abweichung"] = float(ident_max)
    # (3) counter-examples
    ERG["gegenbeispiel_gemeinsamer_faktor_D"] = 0.0
    ERG["gegenbeispiel_beliebige_skalen"] = "Scales a=b=c=d=e=f=1 (no scale or one common scale) give all eight ratios 1 and D = 0 < 0.144; the bound holds only if the scales enforce equal variance (other normalisations not checked)"
    # (4) prediction of the observed ratios
    beob_tr = np.array([1.0600, 1.0245, 1.1521, 1.1162, 1.1870, 1.1405, 1.2232, 1.1897])
    beob_va = np.array([1.0607, 1.0255, 1.1521, 1.1162, 1.1871, 1.1413, 1.2231, 1.1898])
    s = sds(1.034, 1.0, 0.54)
    pred = ratios(s)
    ERG["vorhersage_verhaeltnisse"] = [round(float(x), 4) for x in pred]
    ERG["beobachtet_training"] = beob_tr.tolist()
    ERG["max_abweichung_log_training"] = float(np.max(np.abs(np.log(pred) - np.log(beob_tr))))
    ERG["max_abweichung_log_validierung"] = float(np.max(np.abs(np.log(pred) - np.log(beob_va))))
    ERG["D_vorhergesagt"] = float(np.max(np.abs(np.log(pred))))
    ERG["D_beobachtet_training"] = float(np.max(np.abs(np.log(beob_tr))))
    # (5) synthetic signals
    ERG.update(synthetisch(rng))
    (OUT / "r10_ergebnis.json").write_text(json.dumps(ERG, indent=1), encoding="utf-8")
    print(json.dumps(ERG, indent=1))


def synthetisch(rng, n_ecg=400, fs=500, dauer=10):
    from scipy.signal import medfilt
    n = fs * dauer
    t = np.arange(n) / fs
    sigs = np.zeros((n_ecg, 6, n))
    for e in range(n_ecg):
        hr = rng.uniform(55, 100) / 60
        axis = np.deg2rad(rng.normal(30, 35))
        amp = np.exp(rng.normal(0, 0.5))
        x = np.zeros(n)
        y = np.zeros(n)
        ph = rng.uniform(0, 1 / hr)
        for k in range(int(dauer * hr) + 2):
            tc = ph + k / hr
            qrs = np.exp(-0.5 * ((t - tc) / 0.012) ** 2)
            tw = 0.35 * np.exp(-0.5 * ((t - tc - 0.25) / 0.05) ** 2)
            ang = axis + rng.normal(0, 0.08)
            x += amp * (qrs * np.cos(ang) + tw * np.cos(ang + 0.3))
            y += amp * (qrs * np.sin(ang) + tw * np.sin(ang + 0.3)) * 0.9
        # baseline wander and noise (common in the vector so that Einthoven holds exactly; noise separate per vector component)
        x += 0.15 * amp * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6))
        y += 0.15 * amp * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6))
        x += rng.normal(0, 0.02 * amp, n)
        y += rng.normal(0, 0.02 * amp, n)
        lead_i = x
        lead_ii = 0.5 * x + np.sqrt(3) / 2 * y
        sigs[e] = np.vstack([lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2])
    out = {}

    def auswertung(S, name):
        # data set constants per lead: mean and SD over all samples, then standardisation (V2)
        mu = S.mean(axis=(0, 2), keepdims=True)
        sd = S.std(axis=(0, 2), keepdims=True)
        Z = (S - mu) / sd
        ratios_e = []
        for e in range(S.shape[0]):
            A = np.column_stack([Z[e, 0], Z[e, 1], np.ones(Z.shape[2])])
            row = []
            for x in (2, 3, 4, 5):
                coef = np.linalg.lstsq(A, Z[e, x], rcond=None)[0]
                exp_ = {2: (-1.0, 1.0), 3: (-0.5, -0.5), 4: (1.0, -0.5), 5: (-0.5, 1.0)}[x]
                row += [coef[0] / exp_[0], coef[1] / exp_[1]]
            ratios_e.append(row)
        # order as in the table: III/I, III/II, aVR/I, aVR/II, ... : coef[0] = on I, coef[1] = on II
        r = np.median(np.array(ratios_e), axis=0)
        return r, float(np.max(np.abs(np.log(r)))), sd.squeeze().tolist()

    r0, d0, sd0 = auswertung(sigs, "linear")
    out["synthetisch_sd_je_ableitung"] = [round(x, 4) for x in sd0]
    out["synthetisch_rho_I_II"] = float(np.corrcoef(sigs[:, 0].ravel(), sigs[:, 1].ravel())[0, 1])
    out["synthetisch_linear_D"] = d0
    out["synthetisch_linear_Verhaeltnisse"] = [round(float(x), 4) for x in r0]
    s_pred = ratios(sds(sd0[0], sd0[1], out["synthetisch_rho_I_II"]))
    out["synthetisch_linear_vorhersage_aus_sd_und_rho"] = [round(float(x), 4) for x in s_pred]
    # median filter per lead (kernel 5 samples, noise suppression) and baseline median filter (kernel 1.0 s subtracted)
    S1 = np.empty_like(sigs)
    S2 = np.empty_like(sigs)
    for e in range(sigs.shape[0]):
        for l in range(6):
            S1[e, l] = medfilt(sigs[e, l], 5)
            base = medfilt(sigs[e, l], 2 * (fs // 2) + 1)
            S2[e, l] = sigs[e, l] - base
    r1, d1, _ = auswertung(S1, "medfilt5")
    r2, d2, _ = auswertung(S2, "basislinie")
    # clipping at the 0.1th and 99.9th percentile per lead (data set)
    S3 = sigs.copy()
    for l in range(6):
        lo, hi = np.percentile(sigs[:, l], [0.1, 99.9])
        S3[:, l] = np.clip(sigs[:, l], lo, hi)
    r3, d3, _ = auswertung(S3, "clip")
    S4 = S2.copy()
    for l in range(6):
        lo, hi = np.percentile(S4[:, l], [0.1, 99.9])
        S4[:, l] = np.clip(S4[:, l], lo, hi)
    r4, d4, _ = auswertung(S4, "alles")
    for nm, r, d in (("medianfilter5", r1, d1), ("basislinie_medianfilter", r2, d2), ("clipping", r3, d3), ("basislinie_und_clipping", r4, d4)):
        out["synthetisch_%s_D" % nm] = d
        out["synthetisch_%s_max_abw_log_gegen_linear" % nm] = float(np.max(np.abs(np.log(r) - np.log(r0))))
    return out


if __name__ == "__main__":
    main()
