# -*- coding: utf-8 -*-
"""aw239_r2.py (AW-239, P12-R2, Version 1.0): CLBBB-Untergruppen ohne Schrittmacherhinweis, nur Wahrscheinlichkeit 100, Morphologie unabhaengig vom Code (PUB-12, explorativ, post hoc).

Population: P0 (18.869 Personen), SCP-Code CLBBB (418; Definition wie AW-234: Schluessel vorhanden). Paare der fuenf Programme wie in R1 (Kandidat minus Komparator).
(a) Schrittmacherhinweise: SCP-Code PACE, Feld "pacemaker" der Datenbank, oder Berichtstext (klein geschrieben) mit einem der Muster
    schrittmach, pacemak, \\bpaced?\\b, \\bpacing\\b, stimulier, \\bvvi\\b, \\bddd\\b, \\bvdd\\b, \\baai\\b; die Treffer werden einzeln ausgezählt.
(b) Nur Wahrscheinlichkeit 100 (Likelihood des SCP-Codes CLBBB = 100).
(c) Morphologiekriterium (VOR dem Lauf festgelegt, aus dem Median-Schlag von 12SL, PTB-XL+ 1.0.1, 500 Hz, 12 Ableitungen, Werte in Mikrovolt; QRS-Grenzen aus
    QRS_On_Global und QRS_Off_Global des 12SL-Archivs, in ms; Abtastwert = ms / 2):
    Ableitungen V1, V2, V5, V6, I, aVL. Je Ableitung: Segment [Beginn, Ende], 3-Punkt-Glaettung (1,2,1)/4, Hauptwelle = Abtastwert mit dem groessten Betrag der Abweichung vom
    Segmentanfang, Vorzeichen so gedreht, dass die Hauptwelle positiv ist (y). Mittleres Drittel = Abtastwerte n/3 bis 2n/3 des Segments.
    KERBUNG: mindestens zwei Gipfel von y mit Prominenz >= 0,05 mV, deren dazwischenliegendes Tal im mittleren Drittel liegt, mit Taltiefe >= 0,05 mV unter dem niedrigeren
    der beiden Gipfel und y(Tal) >= 0,25 mal y(Hauptgipfel) (das Tal faellt nicht zur Nulllinie zurueck).
    VERSCHLEIFUNG: im mittleren Drittel mindestens 6 aufeinanderfolgende Abtastwerte (12 ms) mit |Steigung| < 0,15 mal maximale |Steigung| des Segments, bei gleichem Vorzeichen
    der Steigung 3 Abtastwerte vor und nach der Flaeche und dort jeweils |Steigung| >= 0,4 mal Maximum (Schulter auf einer Flanke, kein Gipfel).
    Ableitung positiv = Kerbung oder Verschleifung. Aufnahme morphologie-positiv = mindestens 2 der 6 Ableitungen positiv. Dauerkriterium der Strauss-Kriterien (>= 140 ms Maenner,
    >= 130 ms Frauen) wird NICHT zur Auswahl benutzt (es wuerde nach dem Messwert auswaehlen, dessen Uebereinstimmung geprueft wird); als Empfindlichkeitsfassung wird die
    12SL-Dauer zusaetzlich verlangt und als bedingt gekennzeichnet.
    Gueltigkeitspruefung des Kriteriums (vorab festgelegt): positive Quote in CLBBB (418) gegen CRBBB (alle P0) und gegen 500 schmale Personen (12SL-QRS unter 100 ms, ohne
    Schenkelblock-/IVCD-/PACE-Code, Zufallsziehung). Das Kriterium gilt nur dann als belastbar, wenn die Quote bei CLBBB mindestens doppelt so hoch ist wie bei den schmalen Personen
    und bei diesen hoechstens 20 % betraegt; sonst wird Schritt (c) als nicht belastbar berichtet.
Statistik: D120, D130, D150, Klassenwechsel und Anteil der Personen mit Mittelwert beider Programme innerhalb 10 ms von 150 je Untergruppe, Perzentil-Bootstrap ueber Personen
(2.000 Ziehungen, fester Seed, gemeinsame Ziehungen).
Aufruf: <local path> -I -B aw239_r2.py --modus lauf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.dont_write_bytecode = True

VERSION = "1.0"
SEED = 15102392
BASIS_PFAD = Path(str(data_path("pub12.werkzeuge_pub12_aw239_aw239_basis_py")))
BASIS_HASH = "50fcf31d788931c671bd0021e112d2f9272795daecd2c24bbd215af0b0d4c7bd"
OUT = Path(str(data_path("pub12.benchmark_pub12_aw239_r2")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw239")))
MED = Path(str(data_path("pub12.benchmark_pub12_aw239_medians")))
GRUPPEN = Path(str(data_path("pub12.benchmark_pub12_aw239_download_r2_gruppen_csv")))
SUMS = Path(str(data_path("pub12.benchmark_pub12_aw239_download_sha256sums_txt")))


def _lade_basis():
    h = hashlib.sha256(BASIS_PFAD.read_bytes()).hexdigest()
    if BASIS_HASH != "UNGESETZT" and h != BASIS_HASH:
        raise SystemExit("aw239_basis.py weicht ab: %s" % h)
    spec = importlib.util.spec_from_file_location("aw239_basis", str(BASIS_PFAD))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aw239_basis"] = mod
    spec.loader.exec_module(mod)
    return mod, h


BA, BASIS_IST = _lade_basis()
log = BA.log

SM_MUSTER = {"schrittmach": r"schrittmach", "pacemak": r"pacemak", "paced": r"\bpaced?\b", "pacing": r"\bpacing\b", "stimulier": r"stimulier", "vvi": r"\bvvi\b",
             "ddd": r"\bddd\b", "vdd": r"\bvdd\b", "aai": r"\baai\b"}
MORPH_LEADS = ["V1", "V2", "V5", "V6", "I", "aVL"]
PROM_MV = 0.05
TAL_MIN_REL = 0.25
FLACH_REL, STEIL_REL, FLACH_N, SCHULTER_ABST = 0.15, 0.4, 6, 3
ALLE_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ---------------------------------------------------------------------------------------------------------------
# Median-Schlag lesen und Morphologie
# ---------------------------------------------------------------------------------------------------------------
def lese_median(stamm):
    """12SL-Median-Schlag (WFDB-Kopf, 32 Bit). Rueckgabe (Werte in Mikrovolt (n, 12), Ableitungsnamen, fs)."""
    lines = Path(str(stamm) + ".hea").read_text().splitlines()
    first = lines[0].split()
    n, fs, L = int(first[1]), float(first[2]), int(first[3])
    fields = [x.split() for x in lines[1:n + 1]]
    gb = [re.match(r"([^(/]+)(?:\(([^)]+)\))?", x[2]) for x in fields]
    gain = np.array([float(g.group(1)) for g in gb])
    base = np.array([int(g.group(2) or 0) for g in gb])
    names = [x[-1] for x in fields]
    fmt = fields[0][1]
    raw = np.fromfile(str(stamm) + ".dat", dtype={"16": "<i2", "32": "<i4"}[fmt]).reshape(L, n)
    return (raw - base) / gain, names, fs


def glaette(x):
    p = np.concatenate([[x[0]], x, [x[-1]]])
    return (p[:-2] + 2 * p[1:-1] + p[2:]) / 4.0


def lead_morph(x_mv, i_on, i_off):
    """(kerbung, verschleifung) fuer eine Ableitung; x_mv in mV, Indizes in Abtastwerten."""
    seg = np.asarray(x_mv[i_on:i_off + 1], float)
    n = len(seg)
    if n < 20:
        return False, False
    xs = glaette(seg)
    dev = xs - xs[0]
    im = int(np.argmax(np.abs(dev)))
    sg = 1.0 if dev[im] >= 0 else -1.0
    y = sg * dev
    lo, hi = n // 3, (2 * n) // 3
    kerbung = False
    pk, _ = find_peaks(y, prominence=PROM_MV)
    for a, b in zip(pk[:-1], pk[1:]):
        v = a + int(np.argmin(y[a:b + 1]))
        if lo <= v <= hi and min(y[a], y[b]) - y[v] >= PROM_MV and y[v] >= TAL_MIN_REL * y[im]:
            kerbung = True
            break
    dx = np.gradient(xs)
    smax = np.abs(dx).max()
    versch = False
    if smax > 0:
        flach = np.abs(dx) < FLACH_REL * smax
        i = lo
        while i <= hi:
            if flach[i]:
                j = i
                while j + 1 <= hi and flach[j + 1]:
                    j += 1
                if j - i + 1 >= FLACH_N and i - SCHULTER_ABST >= 0 and j + SCHULTER_ABST < n:
                    d1, d2 = dx[i - SCHULTER_ABST], dx[j + SCHULTER_ABST]
                    if np.sign(d1) == np.sign(d2) and abs(d1) >= STEIL_REL * smax and abs(d2) >= STEIL_REL * smax:
                        versch = True
                        break
                i = j + 1
            else:
                i += 1
    return kerbung, versch


def aufnahme_morph(x_uv, names, on_ms, off_ms):
    """Rueckgabe dict: Ableitung -> (kerbung, verschleifung); positiv = mindestens 2 Ableitungen mit Kerbung oder Verschleifung."""
    i_on, i_off = int(round(on_ms / 2.0)), int(round(off_ms / 2.0))
    res = {}
    for ld in MORPH_LEADS:
        res[ld] = lead_morph(x_uv[:, names.index(ld)] / 1000.0, i_on, i_off)
    npos = sum(1 for k, v in res.items() if v[0] or v[1])
    return res, npos


# ---------------------------------------------------------------------------------------------------------------
# Schrittmacherhinweise
# ---------------------------------------------------------------------------------------------------------------
def schrittmacher_hinweise(B):
    """Je Person: Code PACE, Feld pacemaker, Berichtsmuster -> dict Name -> bool-Array."""
    out = {"scp_PACE": B.code["PACE"].copy(), "feld_pacemaker": B.pacemaker_feld.notna().to_numpy()}
    rep = B.report
    for k, rx in SM_MUSTER.items():
        r = re.compile(rx)
        out["bericht_" + k] = np.array([bool(r.search(s)) for s in rep])
    anyhit = np.zeros(len(rep), bool)
    for v in out.values():
        anyhit |= v
    out["irgendein_hinweis"] = anyhit
    return out


# ---------------------------------------------------------------------------------------------------------------
def baue(B, qrs, subsets):
    bk = BA.Baukasten(len(B.ids))
    for ta, tb in [BA.orientiere(BA.WERKZEUGE[i], BA.WERKZEUGE[j]) for i in range(5) for j in range(i + 1, 5)]:
        a, b = np.asarray(qrs[ta], float), np.asarray(qrs[tb], float)
        for sub, mask in subsets.items():
            pref = "p|%s|%s|%s" % (ta, tb, sub)
            ok = BA.spalten_paar(bk, pref, a, b, mask)
            m = np.where(ok, (np.where(ok, a, 0) + np.where(ok, b, 0)) / 2.0, 0.0)
            bk.indikator(pref + "|nah150", ok & (np.abs(m - 150) < 10))
    return bk


def tabelle(Sx, subsets):
    rows = []
    for ta, tb in [BA.orientiere(BA.WERKZEUGE[i], BA.WERKZEUGE[j]) for i in range(5) for j in range(i + 1, 5)]:
        for sub in subsets:
            pref = "p|%s|%s|%s" % (ta, tb, sub)
            V = Sx[pref + "|V"]
            r = dict(kandidat=ta, komparator=tb, untergruppe=sub, n=int(V[0]))
            if V[0] == 0:
                rows.append(r)
                continue
            for t in BA.QRS_SCHWELLEN:
                s2 = BA.stats_2x2(Sx, pref, t)
                r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t] = BA.ci(100 * s2["disc"])
                r["n_disc_%d" % t] = int(s2["n_disc"][0])
            r["klassenwechsel_pct"], r["klassenwechsel_ki_u"], r["klassenwechsel_ki_o"] = BA.ci(100 * Sx[pref + "|wechsel"] / V)
            r["anteil_innerhalb10ms_von_150_pct"], r["nah_ki_u"], r["nah_ki_o"] = BA.ci(100 * Sx[pref + "|nah150"] / V)
            sm = BA.stats_moment(Sx, pref)
            r["bias_ms"], r["bias_ki_u"], r["bias_ki_o"] = BA.ci(sm["bias"])
            rows.append(r)
    return pd.DataFrame(rows)


def lauf():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    BA.S.STATE["log"] = str(OUT / "aw239_r2.log")
    me = Path(__file__)
    log("Start aw239_r2 Version %s, Skript-SHA-256 %s, Basis %s" % (VERSION, BA.sha256_datei(me), BASIS_IST))
    B = BA.lade_basis(mit_hash=True)
    pruef = list(B.pruef)
    pd.DataFrame(pruef).to_csv(OUT / "00_hashpruefung.csv", index=False, encoding="utf-8")
    qrs = {t: B.D[t]["qrs"] for t in BA.WERKZEUGE}
    ids = B.ids
    pos = pd.Series(np.arange(len(ids)), index=ids)
    K = len(ids)
    cl = B.code["CLBBB"]
    hinw = schrittmacher_hinweise(B)
    sm_tab = pd.DataFrame([dict(hinweis=k, n_CLBBB=int((v & cl).sum()), n_gesamt_P0=int(v.sum())) for k, v in hinw.items()])
    log("Schrittmacherhinweise in CLBBB:\n" + sm_tab.to_string())
    ohne_sm = cl & ~hinw["irgendein_hinweis"]
    p100 = cl & (B.clbbb_p >= 100)
    # --- Morphologie
    sums = {}
    for z in SUMS.read_text(encoding="utf-8").splitlines():
        p = z.split()
        if len(p) == 2 and p[1].startswith("median_beats/12sl/"):
            sums[p[1]] = p[0]
    f12 = pd.read_csv(BA.S.PFADE["plus"] / "12sl_features.csv", usecols=["ecg_id", "QRS_On_Global", "QRS_Off_Global", "QRS_Dur_Global"]).drop_duplicates("ecg_id").set_index("ecg_id")
    gr = pd.read_csv(GRUPPEN)
    lead_rows, rec_rows = [], []
    morph = {}
    fehlend = 0
    nicht_im_spiegel = 0
    for _, r in gr.iterrows():
        e = int(r.ecg_id)
        rel = "median_beats/12sl/%05d/%05d_medians" % (e // 1000 * 1000, e)
        pth = MED / rel
        okfile = pth.with_suffix(".hea").exists() and pth.with_suffix(".dat").exists()
        if not okfile:
            if (rel + ".hea") in sums:
                fehlend += 1
            else:
                nicht_im_spiegel += 1
            continue
        h1 = hashlib.sha256(pth.with_suffix(".hea").read_bytes()).hexdigest() == sums[rel + ".hea"]
        h2 = hashlib.sha256(pth.with_suffix(".dat").read_bytes()).hexdigest() == sums[rel + ".dat"]
        if not (h1 and h2):
            raise SystemExit("Hash der Median-Schlag-Datei stimmt nicht: %s" % rel)
        x, names, fs = lese_median(pth)
        res, npos = aufnahme_morph(x, names, float(f12.loc[e, "QRS_On_Global"]), float(f12.loc[e, "QRS_Off_Global"]))
        morph[(r.gruppe, e)] = npos
        i_on, i_off = int(round(float(f12.loc[e, "QRS_On_Global"]) / 2.0)), int(round(float(f12.loc[e, "QRS_Off_Global"]) / 2.0))
        amp = {ld: float(np.ptp(x[i_on:i_off + 1, names.index(ld)]) / 1000.0) for ld in ("V1", "V5")}
        rec_rows.append(dict(gruppe=r.gruppe, ecg_id=e, n_leads_positiv=npos, morph_positiv=npos >= 2, amp_V1_mV=amp["V1"], amp_V5_mV=amp["V5"]))
        for ld, (k, v) in res.items():
            lead_rows.append(dict(gruppe=r.gruppe, ableitung=ld, kerbung=k, verschleifung=v, positiv=k or v))
    rec = pd.DataFrame(rec_rows)
    lr = pd.DataFrame(lead_rows)
    det = []
    for g, d in rec.groupby("gruppe"):
        det.append(dict(gruppe=g, n=len(d), morph_positiv_n=int(d.morph_positiv.sum()), morph_positiv_pct=100 * float(d.morph_positiv.mean()),
                        mittlere_zahl_positiver_ableitungen=float(d.n_leads_positiv.mean()), median_ptp_V1_mV_Einheitenkontrolle=float(d.amp_V1_mV.median()),
                        median_ptp_V5_mV_Einheitenkontrolle=float(d.amp_V5_mV.median())))
    det = pd.DataFrame(det)
    lw = lr.groupby(["gruppe", "ableitung"]).agg(n=("positiv", "size"), kerbung_pct=("kerbung", lambda s: 100 * s.mean()), verschleifung_pct=("verschleifung", lambda s: 100 * s.mean()),
                                              positiv_pct=("positiv", lambda s: 100 * s.mean())).reset_index()
    log("Morphologie-Detektor:\n" + det.to_string())
    qc = det.set_index("gruppe").morph_positiv_pct
    belastbar = bool(qc.get("CLBBB", 0) >= 2 * qc.get("schmal", 100) and qc.get("schmal", 100) <= 20.0)
    log("Detektor belastbar nach vorab festgelegtem Kriterium: %s (CLBBB %.1f %%, schmal %.1f %%)" % (belastbar, qc.get("CLBBB", np.nan), qc.get("schmal", np.nan)))
    # Masken auf P0
    m_pos = np.zeros(K, bool)
    m_neg = np.zeros(K, bool)
    for (g, e), npos in morph.items():
        if g != "CLBBB":
            continue
        i = pos[e]
        if npos >= 2:
            m_pos[i] = True
        else:
            m_neg[i] = True
    dur12 = B.arch["12SL"]["QRS_Dur_Global"]
    strauss_dauer = np.where(B.sex == 0, dur12 >= 140, dur12 >= 130)
    subsets = {"CLBBB_alle": cl, "CLBBB_ohne_Schrittmacherhinweis": ohne_sm, "CLBBB_nur_Wahrscheinlichkeit_100": p100, "CLBBB_ohne_SM_und_Wahrscheinlichkeit_100": ohne_sm & p100}
    if belastbar:
        subsets.update({"CLBBB_morphologie_positiv": cl & m_pos, "CLBBB_morphologie_negativ": cl & m_neg, "CLBBB_morphologie_positiv_und_12SL_Dauer_Strauss_BEDINGT": cl & m_pos & strauss_dauer})
    bk = baue(B, qrs, subsets)
    Sx = BA.boot_summen(bk, BA.ziehungen(K, SEED))
    tab = tabelle(Sx, subsets)
    # Kontrolle: CLBBB_alle gegen R1/AW-234 (D150 BRAVEHEART-12SL 14,1)
    ta, tb = "BRAVEHEART", "12SL"
    d150 = 100 * BA.stats_2x2(Sx, "p|%s|%s|CLBBB_alle" % (ta, tb), 150)["disc"][0]
    kon = pd.DataFrame([dict(kontrolle="CLBBB_alle BRAVEHEART-12SL D150", soll="14.1", ist="%.2f" % d150, bestanden=abs(d150 - 14.1) < 0.06),
                        dict(kontrolle="CLBBB-Menge", soll="418", ist=str(int(cl.sum())), bestanden=int(cl.sum()) == 418),
                        dict(kontrolle="Median-Schlag-Dateien: im Spiegel verzeichnet, lokal aber fehlend", soll="0", ist=str(fehlend), bestanden=fehlend == 0),
                        dict(kontrolle="Median-Schlag-Dateien: im Spiegel nicht vorhanden (nur Information; Person morphologisch unbestimmt)", soll="Information", ist=str(nicht_im_spiegel), bestanden=True)])
    log("Kontrollen:\n" + kon.to_string())
    if not kon.bestanden.all():
        log("ABBRUCH: Kontrolle nicht bestanden")
        kon.to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
        return 3
    # Gruppengroessen
    gs = pd.DataFrame([dict(untergruppe=k, n_personen=int(v.sum())) for k, v in subsets.items()] +
                      [dict(untergruppe="CLBBB_morphologie_unbestimmt_ohne_Median_Schlag", n_personen=int((cl & ~m_pos & ~m_neg).sum()))])
    dirs = [OUT, KOPIE]
    BA.schreibe(tab, "r2_untergruppen.csv", dirs)
    BA.schreibe(sm_tab, "r2_schrittmacherhinweise.csv", dirs)
    BA.schreibe(det, "r2_morphologie_detektor.csv", dirs)
    BA.schreibe(lw, "r2_morphologie_je_ableitung.csv", dirs)
    BA.schreibe(gs, "r2_gruppengroessen.csv", dirs)
    BA.schreibe(kon, "r2_kontrollen.csv", dirs)
    BA.schreibe(pd.DataFrame(pruef), "r2_hashpruefung.csv", dirs)
    info = dict(version=VERSION, skript_sha256=BA.sha256_datei(me), basis_sha256=BASIS_IST, seed=SEED, ziehungen=BA.B_ZIEHUNGEN, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                ende=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(pruef),
                morphologie_belastbar=belastbar, detektor_parameter=dict(prominenz_mV=PROM_MV, tal_min_rel=TAL_MIN_REL, flach_rel=FLACH_REL, steil_rel=STEIL_REL, flach_n=FLACH_N,
                                                                         schulter_abstand=SCHULTER_ABST, ableitungen=MORPH_LEADS))
    for d in dirs:
        (Path(d) / "r2_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.parse_args(argv)
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
