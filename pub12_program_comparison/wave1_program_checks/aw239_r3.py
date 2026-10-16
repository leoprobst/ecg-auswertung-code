# -*- coding: utf-8 -*-
"""aw239_r3.py (AW-239, P12-R3, Version 1.0): ECGDeli in PTB-XL+, kontrollierte Zerlegung des Versatzes an allen Aufnahmen (explorativ, post hoc).

Eingabe: ecgdeli_punkte.npz (aw239_r3_einlesen.py; alle veroeffentlichten ECGDeli-Fiducialdateien von PTB-XL+ 1.0.1), Archivspalten QRS_Dur_Global und QT_Int_Global von 12SL,
Uni-G und ECGDeli (PTB-XL+ 1.0.1 features, Hashpruefung gegen die Hashes der Datensperre/Projektpruefung). Einheit: Aufnahme (alle Aufnahmen von PTB-XL+); Intervalle:
Perzentil-Bootstrap ueber Personen (patient_id), 2.000 Ziehungen, fester Seed. Es wird kein Code des Herstellers ausgefuehrt, die Regeln sind eigene Nachbauten nach
ExtractIntervalFeaturesFromFPT.m (Zeilen 48 bis 87, N1/AW-176).

(1) Rekonstruktion: Dauer = Median ueber Schlaege von (zweitspaetestes QRS-Ende minus zweitfruehester QRS-Beginn ueber die 12 Ableitungen) mal 2 ms; QT = Median ueber Schlaege von
    (zweitspaetestes T-Ende minus zweitfruehester QRS-Beginn); Trefferquote gegen die Archivspalten (exakt / innerhalb 1 und 2 ms), Restfaelle nach Schlagzahl.
(2) Randpruefung: Anteil der Ableitungsschlaege, in denen QRS-Beginn = Q-Gipfel minus 13 Abtastwerte und QRS-Ende = S-Gipfel plus 13 Abtastwerte (26 ms bei 500 Hz) gilt.
(3) Regelvarianten (Ränder 0, 20, 26 ms; zweitextrem / extrem / Median ueber Ableitungen / zuerst je Ableitung ueber Schlaege; Q-/S-Gipfel als Grenzen): Median je Variante.
(4) Sequentielle Zerlegung des Versatzes ECGDeli minus 12SL (Mittelwert und Median der Aufnahmewerte) in zwei Reihenfolgen: Rand, Aggregation, Rest; fuer QT zusaetzlich getrennt
    nach Aggregation des QRS-Beginns und des T-Endes.
Aufruf: <local path> -I -B aw239_r3.py --modus lauf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

VERSION = "1.0"
SEED = 15102393
B_ZIEHUNGEN = 2000
CHUNK = 250
NPZ = Path(str(data_path("pub12.benchmark_pub12_aw239_r3_ecgdeli_punkte_npz")))
OUT = Path(str(data_path("pub12.benchmark_pub12_aw239_r3")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw239")))
META = Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv")))
FEAT = Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features")))
FEAT_HASH = {"12sl_features.csv": "cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29",
             "unig_features.csv": "f95ab01df5c9ed329485fd811578f0c39eabb0667a36ec37c7dda427d9d8f5a5",
             "ecgdeli_features.csv": "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556"}
META_HASH_QUELLE = Path(str(data_path("pub12.samples_ptb_x_sha256sums_txt")))


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Zeilenweise (nan-bewusste) Aggregation ueber die 12 Ableitungen
# ---------------------------------------------------------------------------------------------------------------
def zu_float(M):
    M = np.asarray(M, float).copy()
    M[M < 5] = np.nan  # -1 (fehlt) und Zeiten unter 5 Abtastwerten gelten als fehlend
    return M


def kleinster(M, k):
    """k-kleinster nicht fehlender Wert je Zeile (k = 1 extrem, 2 zweitextrem); NaN, wenn weniger als k Werte."""
    s = np.sort(M, axis=1)
    cnt = (~np.isnan(M)).sum(axis=1)
    out = s[:, k - 1].copy()
    out[cnt < k] = np.nan
    return out


def groesster(M, k):
    cnt = (~np.isnan(M)).sum(axis=1)
    s = np.sort(M, axis=1)
    idx = np.clip(cnt - k, 0, M.shape[1] - 1)
    out = s[np.arange(len(M)), idx].copy()
    out[cnt < k] = np.nan
    return out


def nanmed(M):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(M, axis=1)


AGG_KLEIN = {"zweit": lambda M: kleinster(M, 2), "extrem": lambda M: kleinster(M, 1), "median": nanmed}
AGG_GROSS = {"zweit": lambda M: groesster(M, 2), "extrem": lambda M: groesster(M, 1), "median": nanmed}


def je_aufnahme_median(werte_schlag, offsets):
    """Median ueber die Schlaege je Aufnahme (nan-bewusst). werte_schlag: Array (Schlaege,), offsets: Grenzen."""
    out = np.full(len(offsets) - 1, np.nan)
    for i in range(len(out)):
        x = werte_schlag[offsets[i]:offsets[i + 1]]
        x = x[~np.isnan(x)]
        if len(x):
            out[i] = np.median(x)
    return out


class Punkte:
    """Geladene Punktematrizen in Float (NaN = fehlt)."""

    def __init__(self, z):
        self.ids = z["ids"]
        self.off = z["offsets"]
        self.on = zu_float(z["on"])
        self.offs = zu_float(z["off"])
        self.Q = zu_float(z["Q"])
        self.S = zu_float(z["S"])
        self.T = zu_float(z["Toff"])
        self.n_beats = np.diff(self.off)

    # Grenzen mit Rand m (Abtastwerte): Beginn = Q - m, Ende = S + m (Gipfelbasis). Dieselben Ableitungsschlaege wie in der Archivregel: ist der annotierte
    # Beginn (Ende) fehlend (Zeit unter 5 Abtastwerten, ECGDeli setzt NaN), bleibt auch der Gipfelbasis-Wert fehlend.
    def beginn(self, m):
        return np.where(np.isnan(self.on), np.nan, self.Q - m)

    def ende(self, m):
        return np.where(np.isnan(self.offs), np.nan, self.S + m)


def qrs_dauer(P, beg, end, agg_b="zweit", agg_e="zweit"):
    """Dauer je Aufnahme in ms: Median ueber Schlaege von (Aggregat(Ende) - Aggregat(Beginn)) mal 2."""
    d = (AGG_GROSS[agg_e](end) - AGG_KLEIN[agg_b](beg)) * 2.0
    return je_aufnahme_median(d, P.off)


def qrs_dauer_zuerst_ableitung(P, beg, end, art):
    """Zuerst je Ableitung der Median ueber Schlaege (Beginn, Ende), dann Aggregation ueber die Ableitungen."""
    out = np.full(len(P.ids), np.nan)
    for i in range(len(P.ids)):
        a, b = P.off[i], P.off[i + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mb, me = np.nanmedian(beg[a:b], axis=0), np.nanmedian(end[a:b], axis=0)
        if art == "extrem":
            if np.isfinite(mb).any() and np.isfinite(me).any():
                out[i] = (np.nanmax(me) - np.nanmin(mb)) * 2.0
        elif art == "maxbreite":
            w = me - mb
            if np.isfinite(w).any():
                out[i] = np.nanmax(w) * 2.0
        elif art == "medianbreite":
            w = me - mb
            if np.isfinite(w).any():
                out[i] = np.nanmedian(w) * 2.0
    return out


def qt_dauer(P, beg, agg_b="zweit", agg_t="zweit"):
    d = (AGG_GROSS[agg_t](P.T) - AGG_KLEIN[agg_b](beg)) * 2.0
    return je_aufnahme_median(d, P.off)


def qt_zuerst_ableitung(P, beg, art):
    out = np.full(len(P.ids), np.nan)
    for i in range(len(P.ids)):
        a, b = P.off[i], P.off[i + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mb, mt = np.nanmedian(beg[a:b], axis=0), np.nanmedian(P.T[a:b], axis=0)
        if art == "extrem" and np.isfinite(mb).any() and np.isfinite(mt).any():
            out[i] = (np.nanmax(mt) - np.nanmin(mb)) * 2.0
        elif art == "medianQT":
            w = mt - mb
            if np.isfinite(w).any():
                out[i] = np.nanmedian(w) * 2.0
    return out


# ---------------------------------------------------------------------------------------------------------------
# Bootstrap ueber Personen (Cluster) mit Aufnahmewerten
# ---------------------------------------------------------------------------------------------------------------
class ClusterZieh:
    """Ziehungen ueber Personen; chunk_c liefert die Zaehlmatrix je Aufnahme (Ziehungen x Aufnahmen)."""

    def __init__(self, patient_idx, n_pat, seed, B=B_ZIEHUNGEN, chunk=CHUNK):
        self.pi, self.npat, self.seed, self.B, self.chunk = patient_idx, n_pat, seed, B, chunk
        self.n_chunks = int(np.ceil(B / chunk))

    def chunk_c(self, c):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, c]))
        m = min(self.chunk, self.B - c * self.chunk)
        Cp = rng.multinomial(self.npat, np.full(self.npat, 1.0 / self.npat), size=m).astype(np.float64)
        return Cp[:, self.pi]


def boot_mittel(Z, werte_liste):
    """Gewichtete Mittel mehrerer Wertevektoren (NaN-bewusst, je Vektor eigener Nenner) je Ziehung; Zeile 0 = Punktschaetzer. Rueckgabe Array (B+1, len)."""
    X = np.column_stack([np.where(np.isfinite(v), v, 0.0) for v in werte_liste])
    V = np.column_stack([np.isfinite(v).astype(float) for v in werte_liste])
    n = X.shape[0]
    parts = [(np.ones((1, n)))] + [Z.chunk_c(c) for c in range(Z.n_chunks)]
    out = []
    for C in parts:
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((C @ X) / (C @ V))
    return np.vstack(out)


def boot_median(Z, v):
    """Gewichteter unterer Median je Ziehung; Zeile 0 = Punktschaetzer."""
    ok = np.isfinite(v)
    idx = np.flatnonzero(ok)
    o = np.argsort(v[idx], kind="stable")
    idx = idx[o]
    xs = v[idx]
    res = []
    for C in [np.ones((1, len(v)))] + [Z.chunk_c(c) for c in range(Z.n_chunks)]:
        W = C[:, idx]
        cs = np.cumsum(W, axis=1)
        N = cs[:, -1]
        j = (cs >= 0.5 * N[:, None] - 1e-9).argmax(axis=1)
        res.append(xs[j])
    return np.concatenate(res)


def ci(v):
    v = np.asarray(v, float)
    lo, hi = np.percentile(v[1:][np.isfinite(v[1:])], [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


# ---------------------------------------------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------------------------------------------
def trefferquote(name, eigen, archiv):
    ok = np.isfinite(eigen) & np.isfinite(archiv)
    d = (eigen - archiv)[ok]
    return dict(regel=name, n=int(ok.sum()), exakt_n=int((d == 0).sum()), exakt_pct=100 * float(np.mean(d == 0)), innerhalb_1ms_pct=100 * float(np.mean(np.abs(d) <= 1)),
                innerhalb_2ms_pct=100 * float(np.mean(np.abs(d) <= 2)), median_diff_ms=float(np.median(d)), max_betrag_ms=float(np.abs(d).max()))


def lade_archiv(ids):
    out = {}
    for f, h in FEAT_HASH.items():
        ist = sha256_datei(FEAT / f)
        if ist != h:
            raise SystemExit("Hash von %s stimmt nicht" % f)
    for tool, f in (("12SL", "12sl_features.csv"), ("Uni-G", "unig_features.csv"), ("ECGDeli", "ecgdeli_features.csv")):
        d = pd.read_csv(FEAT / f, usecols=["ecg_id", "QRS_Dur_Global", "QT_Int_Global"])
        d = d[~d.ecg_id.duplicated(keep="first")].set_index("ecg_id").reindex(ids)
        out[tool] = dict(qrs=d["QRS_Dur_Global"].to_numpy(float), qt=d["QT_Int_Global"].to_numpy(float))
    return out


def rand_pruefung(P):
    on, Q, off, S = P.on, P.Q, P.offs, P.S
    both = np.isfinite(on) & np.isfinite(Q)
    r = dict(n_ableitungsschlaege=int(on.size), n_beginn_und_Q=int(both.sum()), beginn_gleich_Q_minus_13=int(((Q - on)[both] == 13).sum()),
             beginn_anderer_abstand=int(((Q - on)[both] != 13).sum()), n_beginn_ohne_Q=int((np.isfinite(on) & ~np.isfinite(Q)).sum()), n_Q_ohne_beginn=int((~np.isfinite(on) & np.isfinite(Q)).sum()))
    both = np.isfinite(off) & np.isfinite(S)
    r.update(n_ende_und_S=int(both.sum()), ende_gleich_S_plus_13=int(((off - S)[both] == 13).sum()), ende_anderer_abstand=int(((off - S)[both] != 13).sum()),
             n_ende_ohne_S=int((np.isfinite(off) & ~np.isfinite(S)).sum()), n_S_ohne_ende=int((~np.isfinite(off) & np.isfinite(S)).sum()))
    r["anteil_beginn_13_pct"] = 100 * r["beginn_gleich_Q_minus_13"] / max(r["n_beginn_und_Q"], 1)
    r["anteil_ende_13_pct"] = 100 * r["ende_gleich_S_plus_13"] / max(r["n_ende_und_S"], 1)
    return pd.DataFrame([r])


def lauf():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "aw239_r3.log"

    def log(m):
        z = time.strftime("%H:%M:%S") + " " + str(m)
        print(z, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(z + "\n")

    me = Path(__file__)
    log("Start aw239_r3 Version %s, Skript-SHA-256 %s, npz-SHA-256 %s" % (VERSION, sha256_datei(me), sha256_datei(NPZ)))
    z = np.load(NPZ)
    P = Punkte(z)
    ids = P.ids
    arch = lade_archiv(ids)
    meta = pd.read_csv(META, usecols=["ecg_id", "patient_id"]).drop_duplicates("ecg_id").set_index("ecg_id").reindex(ids)
    pat, pat_inv = np.unique(meta.patient_id.to_numpy(), return_inverse=True)
    Z = ClusterZieh(pat_inv, len(pat), SEED)
    log("Aufnahmen %d, Personen %d, Schlaege %d" % (len(ids), len(pat), len(P.on)))
    dirs = [OUT, KOPIE]
    OUT.mkdir(parents=True, exist_ok=True)
    KOPIE.mkdir(parents=True, exist_ok=True)

    def schreibe(df, name):
        for d in dirs:
            df.to_csv(Path(d) / name, index=False, encoding="utf-8", float_format="%.6g")

    schreibe(rand_pruefung(P), "r3_randpruefung.csv")
    # --- annotierte Grenzen (Archivregel)
    V0 = qrs_dauer(P, P.on, P.offs, "zweit", "zweit")
    QT0 = qt_dauer(P, P.on, "zweit", "zweit")
    rows = [dict(groesse="QRS", **trefferquote("zweitspaetestes Ende minus zweitfruehester Beginn, Median ueber Schlaege (Archivregel)", V0, arch["ECGDeli"]["qrs"])),
            dict(groesse="QT", **trefferquote("zweitspaetestes T-Ende minus zweitfruehester QRS-Beginn, Median ueber Schlaege (Archivregel)", QT0, arch["ECGDeli"]["qt"]))]
    cand = {"Extrem (frueheste Beginn bis spaeteste Ende), Median ueber Schlaege": qrs_dauer(P, P.on, P.offs, "extrem", "extrem"),
            "Maximum der Ableitungsmediane der Breite": qrs_dauer_zuerst_ableitung(P, P.on, P.offs, "maxbreite"),
            "Median der Ableitungsbreiten je Schlag": je_aufnahme_median(nanmed(P.offs - P.on) * 2.0, P.off)}
    for k, v in cand.items():
        rows.append(dict(groesse="QRS", **trefferquote(k, v, arch["ECGDeli"]["qrs"])))
    schreibe(pd.DataFrame(rows), "r3_trefferquote.csv")
    # Restfaelle nach Schlagzahl
    d = V0 - arch["ECGDeli"]["qrs"]
    ok = np.isfinite(d)
    nb = P.n_beats
    rf = []
    for lo, hi in ((0, 5), (5, 8), (8, 11), (11, 14), (14, 100)):
        m = ok & (nb >= lo) & (nb < hi)
        if m.sum():
            rf.append(dict(schlaege_von=lo, schlaege_bis=hi - 1, n=int(m.sum()), exakt_pct=100 * float(np.mean(d[m] == 0)), innerhalb_2ms_pct=100 * float(np.mean(np.abs(d[m]) <= 2)),
                           n_nicht_exakt=int((d[m] != 0).sum())))
    schreibe(pd.DataFrame(rf), "r3_restfaelle_nach_schlagzahl.csv")
    # --- Regelvarianten QRS (Gipfelbasis mit Rand m in Abtastwerten: 0, 10 = 20 ms, 13 = 26 ms)
    var = {}
    for m in (0, 10, 13):
        for agg in ("zweit", "extrem", "median"):
            var["qrs|rand%d|%s" % (2 * m, agg)] = qrs_dauer(P, P.beginn(m), P.ende(m), agg, agg)
        var["qrs|rand%d|zuerst_ableitung_extrem" % (2 * m)] = qrs_dauer_zuerst_ableitung(P, P.beginn(m), P.ende(m), "extrem")
        var["qrs|rand%d|zuerst_ableitung_maxbreite" % (2 * m)] = qrs_dauer_zuerst_ableitung(P, P.beginn(m), P.ende(m), "maxbreite")
        var["qrs|rand%d|zuerst_ableitung_medianbreite" % (2 * m)] = qrs_dauer_zuerst_ableitung(P, P.beginn(m), P.ende(m), "medianbreite")
    for m in (0, 13):
        for agg in ("zweit", "extrem", "median"):
            var["qt|rand%d|%s" % (2 * m, agg)] = qt_dauer(P, P.beginn(m), agg, agg)
        var["qt|rand%d|zuerst_ableitung_extrem" % (2 * m)] = qt_zuerst_ableitung(P, P.beginn(m), "extrem")
        var["qt|rand%d|zuerst_ableitung_medianQT" % (2 * m)] = qt_zuerst_ableitung(P, P.beginn(m), "medianQT")
        # getrennte Aggregation Beginn / T-Ende
        for ab in ("zweit", "median"):
            for at in ("zweit", "median"):
                var["qt|rand%d|beginn_%s_tende_%s" % (2 * m, ab, at)] = qt_dauer(P, P.beginn(m), ab, at)
    rows = []
    for k, v in sorted(var.items()):
        g = "qrs" if k.startswith("qrs") else "qt"
        r = dict(groesse=g.upper(), variante=k.split("|", 1)[1], n=int(np.isfinite(v).sum()))
        mm = boot_median(Z, v)
        r["median_ms"], r["median_ki_u"], r["median_ki_o"] = ci(mm)
        for tool in ("12SL", "Uni-G"):
            dd = v - arch[tool][g]
            r["median_diff_zu_%s_ms" % tool], r["diff_%s_ki_u" % tool], r["diff_%s_ki_o" % tool] = ci(boot_median(Z, dd))
        dd = v - arch["ECGDeli"][g]
        r["median_diff_zu_Archiv_ECGDeli_ms"] = float(np.nanmedian(dd))
        rows.append(r)
    schreibe(pd.DataFrame(rows), "r3_regelvarianten.csv")
    # --- Zerlegung QRS: ECGDeli (Archivregel V0) minus 12SL
    def zerlege(ziel, schritte, ref, name):
        """schritte: Liste von (Bezeichnung, Vektor nach dem Schritt) beginnend mit dem Ausgang. Beitrag = vorher minus nachher; Rest = letzter Vektor minus Referenz."""
        out = []
        pref = [s[1] for s in schritte]
        for i in range(len(schritte) - 1):
            out.append((schritte[i + 1][0], pref[i] - pref[i + 1]))
        out.append(("Rest: Endvariante minus " + ref[0], pref[-1] - ref[1]))
        total = pref[0] - ref[1]
        rows = []
        rows.append(dict(zerlegung=name, schritt="Gesamt: Ausgang minus " + ref[0], n=int(np.isfinite(total).sum()), **stat(total)))
        for b, v in out:
            rows.append(dict(zerlegung=name, schritt=b, n=int(np.isfinite(v).sum()), **stat(v)))
        return rows

    def stat(v):
        r = {}
        mm = boot_mittel(Z, [v])[:, 0]
        r["mittel_ms"], r["mittel_ki_u"], r["mittel_ki_o"] = ci(mm)
        r["median_ms"], r["median_ki_u"], r["median_ki_o"] = ci(boot_median(Z, v))
        return r

    rowsz = []
    a0 = var["qrs|rand26|zweit"]  # Gipfelbasis mit 26 ms Rand und Zweitextrem entspricht V0 (bei lueckenlosen Rand-Annahmen); Pruefung unten
    rowsz.append(dict(zerlegung="Kontrolle", schritt="Gipfelbasis 26 ms/zweit minus Archivregel V0 (Median ueber Aufnahmen, ms)", n=int(np.isfinite(a0 - V0).sum()),
                      mittel_ms=float(np.nanmean(a0 - V0)), median_ms=float(np.nanmedian(a0 - V0))))
    ref12 = ("12SL", arch["12SL"]["qrs"])
    ends = {"extrem": var["qrs|rand0|extrem"], "median": var["qrs|rand0|median"], "zuerst_ableitung_extrem": var["qrs|rand0|zuerst_ableitung_extrem"]}
    ends13 = {"extrem": var["qrs|rand26|extrem"], "median": var["qrs|rand26|median"], "zuerst_ableitung_extrem": var["qrs|rand26|zuerst_ableitung_extrem"]}
    for e in ends:
        # Reihenfolge 1: Rand, dann Aggregation
        s1 = [("Ausgang (Archivregel, 26 ms Rand, zweitextrem)", V0), ("Rand 26 ms -> 0 ms", var["qrs|rand0|zweit"]), ("Aggregation zweitextrem -> " + e, ends[e])]
        rowsz += zerlege(None, s1, ref12, "QRS Reihenfolge 1 (Rand, dann Aggregation), Endvariante " + e)
        s2 = [("Ausgang (Archivregel, 26 ms Rand, zweitextrem)", V0), ("Aggregation zweitextrem -> " + e, ends13[e]), ("Rand 26 ms -> 0 ms", ends[e])]
        rowsz += zerlege(None, s2, ref12, "QRS Reihenfolge 2 (Aggregation, dann Rand), Endvariante " + e)
    schreibe(pd.DataFrame(rowsz), "r3_zerlegung_qrs.csv")
    # --- Zerlegung QT: ECGDeli minus 12SL; Rand, Beginn-Aggregation, T-Ende-Aggregation
    ref12q = ("12SL", arch["12SL"]["qt"])
    q = lambda m, ab, at: var["qt|rand%d|beginn_%s_tende_%s" % (2 * m, ab, at)]  # noqa: E731
    rowsq = []
    # Ausgang: Archivregel (Rand 26, Beginn zweit, T-Ende zweit) = QT0
    for ab_end, at_end in (("median", "median"), ("zweit", "median")):
        # Reihenfolge 1: Rand, Beginn, T-Ende
        s1 = [("Ausgang (Archivregel)", QT0), ("Rand 26 ms -> 0 ms", q(0, "zweit", "zweit")), ("Beginn zweitfruehester -> " + ab_end, q(0, ab_end, "zweit")),
              ("T-Ende zweitspaetestes -> " + at_end, q(0, ab_end, at_end))]
        rowsq += zerlege(None, s1, ref12q, "QT Reihenfolge 1 (Rand, Beginn, T-Ende), Endvariante Beginn %s / T-Ende %s" % (ab_end, at_end))
        # Reihenfolge 2: T-Ende, Beginn, Rand
        s2 = [("Ausgang (Archivregel)", QT0), ("T-Ende zweitspaetestes -> " + at_end, q(13, "zweit", at_end)), ("Beginn zweitfruehester -> " + ab_end, q(13, ab_end, at_end)),
              ("Rand 26 ms -> 0 ms", q(0, ab_end, at_end))]
        rowsq += zerlege(None, s2, ref12q, "QT Reihenfolge 2 (T-Ende, Beginn, Rand), Endvariante Beginn %s / T-Ende %s" % (ab_end, at_end))
    schreibe(pd.DataFrame(rowsq), "r3_zerlegung_qt.csv")
    # --- Verteilungen der Archivwerte zum Vergleich
    vt = []
    for t in ("12SL", "Uni-G", "ECGDeli"):
        for g in ("qrs", "qt"):
            x = arch[t][g][np.isfinite(arch[t][g])]
            vt.append(dict(programm=t, groesse=g.upper(), n=len(x), median=float(np.median(x)), p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95))))
    schreibe(pd.DataFrame(vt), "r3_archivwerte.csv")
    info = dict(version=VERSION, skript_sha256=sha256_datei(me), npz_sha256=sha256_datei(NPZ), seed=SEED, ziehungen=B_ZIEHUNGEN, aufnahmen=int(len(ids)), personen=int(len(pat)),
                schlaege=int(len(P.on)), start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                numpy=np.__version__, pandas=pd.__version__, archiv_hashes=FEAT_HASH)
    for d in dirs:
        (Path(d) / "r3_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.parse_args(argv)
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
