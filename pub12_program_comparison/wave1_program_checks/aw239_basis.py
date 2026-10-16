# -*- coding: utf-8 -*-
"""aw239_basis.py (AW-239, Version 1.0): gemeinsame Bausteine der Zusatzrechnungen P12-R1, R2, R5 fuer PUB-12 (alle explorativ, post hoc).

Rechenbasis: aw234_auswertung.py Version 1.0 (und darueber schwellen_aw228.py Version 1.3); beide werden mit festem SHA-256 geprueft, bevor sie importiert
werden. Die gesperrten Tabellen werden nur gelesen, vorher stimmt die Hashpruefung der Datensperre vom 15.10.2026 (15 Eintraege wie AW-234).

Inhalt: Laden von P0 (18.869 Personen), Metadaten (Alter, Geschlecht, SCP-Codes, Schrittmacherhinweise), Spaltenbaukasten mit Personenbootstrap
(Ziehungen als Zaehlmatrizen, gemeinsame Ziehungen fuer alle Kennzahlen), Kennzahlen aus Summen (Bias, SD, Grenzen, ICC(2,1), Lin-CCC, Kappa, PABAK, PPA, NPA,
Steigung der Differenz auf den Mittelwert), gewichtete Quantile je Ziehung.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

A234_PFAD = Path(str(data_path("pub12.werkzeuge_pub12_aw234_auswertung_py")))
A234_HASH = "356c5d1fdbb6576b248f7be23d19d10d3a3009bf917466c3ff576909a4d4a6c7"
B_ZIEHUNGEN = 2000
CHUNK = 250
WERKZEUGE = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]
QRS_SCHWELLEN = (120, 130, 150)


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _lade_a234():
    h = sha256_datei(A234_PFAD)
    if h != A234_HASH:
        raise SystemExit("aw234_auswertung.py weicht von Version 1.0 ab: %s" % h)
    spec = importlib.util.spec_from_file_location("aw234_auswertung", str(A234_PFAD))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aw234_auswertung"] = mod
    spec.loader.exec_module(mod)
    return mod


A = _lade_a234()
S = A.S
log = S.log
ki95 = S.ki95
orientiere = S.orientiere
fridericia = S.fridericia


# ---------------------------------------------------------------------------------------------------------------
# QTc-Formeln (QT und RR in ms, Ergebnis ms)
# ---------------------------------------------------------------------------------------------------------------
def qtc_formel(name, qt, rr_ms):
    qt = np.asarray(qt, float)
    rr = np.asarray(rr_ms, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = rr / 1000.0
        if name == "Bazett":
            out = qt / np.sqrt(rs)
        elif name == "Fridericia":
            out = qt / np.cbrt(rs)
        elif name == "Framingham":
            out = qt + 154.0 * (1.0 - rs)
        elif name == "Hodge":
            hr = 60000.0 / rr
            out = qt + 1.75 * (hr - 60.0)
        else:
            raise ValueError(name)
    return np.where(np.isfinite(rr) & (rr > 0) & np.isfinite(qt), out, np.nan)


# ---------------------------------------------------------------------------------------------------------------
# Spaltenbaukasten und Bootstrap
# ---------------------------------------------------------------------------------------------------------------
class Baukasten:
    """Sammelt Personenspalten (Laenge K). Indikatorspalten (exakt in float32 summierbar) und Momentenspalten (float64) getrennt."""

    def __init__(self, K):
        self.K = K
        self.ind, self.mom = {}, {}

    def indikator(self, name, v):
        assert name not in self.ind and name not in self.mom, name
        self.ind[name] = np.asarray(v, float).astype(np.float32)

    def moment(self, name, v):
        assert name not in self.ind and name not in self.mom, name
        v = np.asarray(v, float)
        self.mom[name] = np.where(np.isfinite(v), v, 0.0)

    def matrizen(self):
        """Baut die beiden Matrizen einmal (Speicher sparend: Spalten werden beim Fuellen aus den Woerterbuechern entfernt) und gibt danach denselben Inhalt zurueck."""
        if getattr(self, "_M", None) is None:
            ni, nm = list(self.ind), list(self.mom)
            Xi = np.empty((self.K, len(ni)), np.float32)
            for j, n in enumerate(ni):
                Xi[:, j] = self.ind.pop(n)
            Xm = np.empty((self.K, len(nm)), np.float64)
            for j, n in enumerate(nm):
                Xm[:, j] = self.mom.pop(n)
            self._M = (ni, Xi, nm, Xm)
        return self._M


def ziehungen(K, seed):
    return S.Ziehungen(K, seed, B=B_ZIEHUNGEN, chunk=CHUNK)


def boot_summen(bk, zieh, W=None):
    """Spaltensummen: Zeile 0 der Rueckgabe ist der Punktschaetzer (alle Gewichte 1), danach B Ziehungen. Rueckgabe dict Name -> Vektor (B+1,)."""
    ni, Xi, nm, Xm = bk.matrizen()
    out_i, out_m = [Xi.sum(axis=0, dtype=np.float64, keepdims=True)], [Xm.sum(axis=0, keepdims=True)]
    for c in range(zieh.n_chunks):
        C = zieh.chunk_c(c)
        out_i.append((C.astype(np.float32) @ Xi).astype(np.float64))
        out_m.append(C @ Xm)
    Si, Sm = np.vstack(out_i), np.vstack(out_m)
    res = {n: Si[:, i] for i, n in enumerate(ni)}
    res.update({n: Sm[:, i] for i, n in enumerate(nm)})
    return res


def ci(v):
    """Punktschaetzer (Index 0) und Perzentilintervall aus Ziehungen 1..B."""
    v = np.asarray(v, float)
    lo, hi = ki95(v[1:])
    return float(v[0]), lo, hi


# ---------------------------------------------------------------------------------------------------------------
# Kennzahlen aus Summen
# ---------------------------------------------------------------------------------------------------------------
MOM_NAMEN = ("V", "a", "b", "aa", "bb", "ab", "ad", "adm")


def spalten_paar(bk, pref, a, b, mask, schwellen=QRS_SCHWELLEN, klassen=True):
    """Momenten- und Indikatorspalten fuer ein Paar (a Kandidat, b Komparator) auf der Teilmenge mask."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.asarray(mask, bool)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    m = (a0 + b0) / 2.0
    ad = np.abs(a0 - b0)
    bk.moment(pref + "|V", ok.astype(float))
    for nm, v in (("a", a0), ("b", b0), ("aa", a0 * a0), ("bb", b0 * b0), ("ab", a0 * b0), ("ad", ad), ("adm", ad * m)):
        bk.moment(pref + "|" + nm, v)
    for t in schwellen:
        pa, pb = ok & (a0 >= t), ok & (b0 >= t)
        bk.indikator("%s|n11|%d" % (pref, t), pa & pb)
        bk.indikator("%s|n10|%d" % (pref, t), pa & ~pb)
        bk.indikator("%s|n01|%d" % (pref, t), ~pa & pb)
        bk.indikator("%s|n00|%d" % (pref, t), ok & ~pa & ~pb)
    if klassen:
        ca = (a0 >= 120).astype(int) + (a0 >= 130) + (a0 >= 150)
        cb = (b0 >= 120).astype(int) + (b0 >= 130) + (b0 >= 150)
        bk.indikator(pref + "|wechsel", ok & (ca != cb))
        bk.indikator(pref + "|hoeher", ok & (ca > cb))
        bk.indikator(pref + "|niedriger", ok & (ca < cb))
    return ok


def stats_moment(Sx, pref):
    """Bias, SD, Grenzen, ICC(2,1), Lin-CCC, Steigungen aus den Momentensummen (Arrays ueber Ziehungen)."""
    g = lambda k: Sx[pref + "|" + k]  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        N = g("V")
        ma, mb = g("a") / N, g("b") / N
        saa, sbb, sab = g("aa") / N, g("bb") / N, g("ab") / N
        dbar = ma - mb
        d2 = saa - 2 * sab + sbb
        mbar = (ma + mb) / 2
        m2 = (saa + 2 * sab + sbb) / 4
        sd = np.sqrt(np.maximum((d2 - dbar ** 2) * N / (N - 1), 0))
        msr = 2.0 * N * (m2 - mbar ** 2) / (N - 1.0)
        mse = N * (d2 - dbar ** 2) / 2.0 / (N - 1.0)
        msc = N * dbar ** 2 / 2.0
        icc = (msr - mse) / (msr + mse + 2.0 * (msc - mse) / N)
        va, vb, cab = saa - ma ** 2, sbb - mb ** 2, sab - ma * mb
        ccc = 2 * cab / (va + vb + (ma - mb) ** 2)
        dm = (saa - sbb) / 2.0
        vm = m2 - mbar ** 2
        slope_d = (dm - dbar * mbar) / vm
        ad, adm = g("ad") / N, g("adm") / N
        slope_ad = (adm - ad * mbar) / vm
    return dict(n=N, bias=dbar, sd=sd, loa_u=dbar - 1.96 * sd, loa_o=dbar + 1.96 * sd, icc=icc, ccc=ccc, slope_d_on_m=slope_d,
                slope_absd_on_m=slope_ad, mean_absd=ad, mean_m=mbar)


def stats_2x2(Sx, pref, t):
    n11, n10, n01, n00 = (Sx["%s|%s|%d" % (pref, k, t)] for k in ("n11", "n10", "n01", "n00"))
    N = n11 + n10 + n01 + n00
    with np.errstate(invalid="ignore", divide="ignore"):
        po = (n11 + n00) / N
        pa, pb = (n11 + n10) / N, (n11 + n01) / N
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = np.where(pe < 1, (po - pe) / (1 - pe), np.nan)
        ppa = 2 * n11 / (2 * n11 + n10 + n01)
        npa = 2 * n00 / (2 * n00 + n10 + n01)
    return dict(n=N, disc=(n10 + n01) / N, nur_a=n10 / N, nur_b=n01 / N, prev_a=pa, prev_b=pb, po=po, kappa=kappa, pabak=2 * po - 1, ppa=ppa, npa=npa,
                n_disc=n10 + n01, n11=n11, n10=n10, n01=n01, n00=n00)


# ---------------------------------------------------------------------------------------------------------------
# Gewichtete Quantile je Ziehung
# ---------------------------------------------------------------------------------------------------------------
def quantile_boot(x, zieh, qs, valid=None):
    """Gewichtete untere Quantile (kleinster Wert, bei dem die kumulierte Gewichtssumme >= q * N ist) je Ziehung; Zeile 0 = Punktschaetzer.
    x: Array Laenge K (NaN = fehlt). Rueckgabe Array (B+1, len(qs))."""
    x = np.asarray(x, float)
    ok = np.isfinite(x) if valid is None else (np.isfinite(x) & np.asarray(valid, bool))
    idx = np.flatnonzero(ok)
    xs_order = np.argsort(x[idx], kind="stable")
    idx_s = idx[xs_order]
    xs = x[idx_s]
    out = []
    W0 = np.ones((1, len(idx_s)))
    blocks = [W0] + [zieh.chunk_c(c)[:, idx_s] for c in range(zieh.n_chunks)]
    for W in blocks:
        cs = np.cumsum(W, axis=1)
        N = cs[:, -1]
        row = []
        for q in qs:
            j = (cs >= q * N[:, None] - 1e-9).argmax(axis=1)
            row.append(xs[j])
        out.append(np.column_stack(row))
    return np.vstack(out)


# ---------------------------------------------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------------------------------------------
class Basis:
    pass


def lade_basis(mit_hash=True):
    """Hashpruefung der gesperrten Tabellen, dann Laden. Rueckgabe Basis mit ids, D (nativ), Metadaten, Archivspalten."""
    if mit_hash:
        ok = S.hashpruefung(S.gelesene_dateien())
        log("Hashpruefung: %d Eintraege, alle stimmen: %s" % (len(S.STATE["pruef"]), ok))
        if not ok:
            raise SystemExit("ABBRUCH: Hashpruefung nicht bestanden")
    B = Basis()
    B.pruef = list(S.STATE["pruef"])
    ids, D = S.lade_alles()
    B.ids, B.D = ids, D
    meta = pd.read_csv(S.PFADE["meta"], usecols=["ecg_id", "sex", "age", "scp_codes", "report", "pacemaker", "patient_id"]).set_index("ecg_id").reindex(ids)
    B.sex = meta["sex"].to_numpy(float)
    B.age = meta["age"].to_numpy(float)
    B.report = meta["report"].fillna("").astype(str).str.lower().to_numpy()
    B.pacemaker_feld = meta["pacemaker"]
    codes = meta["scp_codes"].map(ast.literal_eval)
    B.scp = list(codes)
    has = lambda c: codes.map(lambda x: c in x).to_numpy(bool)  # noqa: E731
    B.code = {c: has(c) for c in ("CLBBB", "CRBBB", "IVCD", "PACE", "AFIB", "AFLT", "ILBBB", "IRBBB")}
    B.clbbb_p = codes.map(lambda x: x.get("CLBBB", 0.0)).to_numpy(float)
    # Archivspalten
    P = S.PFADE
    arch = {}
    for tool, f, cols in (("12SL", "12sl_features.csv", ["HR__Global", "QT_Int_Global", "QT_IntBazett_Global", "QT_IntFridericia_Global", "QT_IntFramingham_Global",
                                                        "QT_IntCorr_Global", "QRS_Dur_Global"]),
                          ("Uni-G", "unig_features.csv", ["HR__Global", "QT_Int_Global", "QT_IntBazett_Global", "QT_IntFridericia_Global", "QT_IntFramingham_Global",
                                                         "QT_IntHodge_Global", "QT_IntCorr_Global", "QRS_Dur_Global"]),
                          ("ECGDeli", "ecgdeli_features.csv", ["RR_Mean_Global", "QT_Int_Global", "QT_IntFramingham_Global", "QRS_Dur_Global"])):
        df = pd.read_csv(P["plus"] / f, usecols=["ecg_id"] + cols)
        df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id").reindex(ids)
        arch[tool] = {c: df[c].to_numpy(float) for c in cols}
    B.arch = arch
    return B


def strata_masken(B):
    """Teilmengen (nicht ueberschneidungsfrei): alle, CLBBB, ohne_CLBBB, CRBBB, IVCD, PACE, AF (AFIB oder AFLT), Alter, Geschlecht."""
    K = len(B.ids)
    c = B.code
    age = B.age
    return {"alle": np.ones(K, bool), "CLBBB": c["CLBBB"], "ohne_CLBBB": ~c["CLBBB"], "CRBBB": c["CRBBB"], "IVCD": c["IVCD"], "PACE": c["PACE"],
            "AF": c["AFIB"] | c["AFLT"], "alter_unter60": age < 60, "alter_60bis79": (age >= 60) & (age < 80), "alter_ab80": age >= 80,
            "maennlich": B.sex == 0, "weiblich": B.sex == 1}


def schreibe(df, name, out_dirs):
    for d in out_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(d) / name, index=False, encoding="utf-8", float_format="%.6g")
