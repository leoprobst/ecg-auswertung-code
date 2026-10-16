# -*- coding: utf-8 -*-
"""bench_auswertung.py (AW-221): Auswertungsskript fuer den Benchmark I-61 nach AUSWERTUNGSVORSCHRIFT_V2.md
und AUSWERTUNGSVORSCHRIFT_V2_NACHTRAG0.md (P2 entfaellt, Holm-Familie m = 1).

Aufruf (echter Lauf):  <local path> -I bench_auswertung.py
Pruefung der Datensperre ohne Auswertung:  ... bench_auswertung.py --nur-hash
Test auf einer synthetischen Spiegelung der Dateiformate: bench_auswertung_test.py (setzt die Pfade um).

Aufbau: Abschnitt 1 Hilfsfunktionen (Wilson, Baender, Holm), Abschnitt 2 Bootstrap-Maschine (Personencluster,
Seed 14102027, 10.000 Ziehungen, gemeinsame Ziehungen fuer alle Einheiten einer Menge), Abschnitt 3 Hashpruefung,
Abschnitt 4 Laden, Abschnitt 5 Einheiten fuer Paare, Abschnitt 6 Stufen (P1, Verstaerkung, Z1, Z2, Z3, Filter, Sens.).
Alles Ergebnisrelevante steht in diesem Skript; es werden keine Schwellen oder Regeln nach Sicht verstellt.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference

import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0"
SEED_BOOT = 14102027
B_ZIEHUNGEN = 10000
CHUNK = 250
Z95 = 1.959963984540054
MS_PRO_SAMPLE = 2.0  # 500 Hz
MARGE_H_P1 = 0.01  # untere Bandgrenze "kaum diskordant" (V2 1 und 6.4)
ALPHA = 0.05
MIN_NENNER_P0 = 100  # Stratumaussagen P0 (V2 4.2)
MIN_NENNER_SG = 50
MIN_NENNER_LUDB = 20
WINKEL_MINNORM_MV = 0.05

PFADE = {
    "erg": Path(str(data_path("pub12.benchmark_ergebnisse"))),
    "listen": Path(str(data_path("pub12.aw_219_listen"))),
    "audit": Path(str(data_path("pub12.aw_217_kalibrierungsaudit_stichprobe_rekonstruiert_csv"))),
    "meta": Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv"))),
    "ludb": Path(str(data_path("pub12.lobachevsky_university_electrocardiography_database_1_0_1_data"))),
    "plus": Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features"))),
    "wz": Path(str(data_path("pub12.werkzeuge_benchmark"))),
    "sperre": Path(str(data_path("pub12.benchmark_vorbereitung_20261015_datensperre_hashes_entwurf_json"))),
    "out": Path(str(data_path("pub12.benchmark_auswertung_aw221"))),
    "kopie": Path(str(data_path("pub12.publikationen_pub_10_benchmark_ergebnisse_aw221"))),
    "zwischen": Path(str(data_path("pub12.koordination_abgaben_aw_221_zwischenstand_md"))),
}

# Hashes der drei Hashlisten (DATENSPERRE_2026-10-15.md)
SPERRE_LISTEN = {
    "sperre": "daab9d46395857e729f7c725d4e5fff2f1ab4e35a0b2e90e0333c20cd3f58c07",
    "manifest": "5b1511cc19de86b28cf84849c1007d1f54cf30362144f5df27854f1653372d07",
    "aw220": "986f17a154eba3904dfe83967564fc85f3ee1d8c7778f32edf0c176d61839230",
}
# Listen AW-219 (LOOPLENS_LAUF_AW219.md Abschnitt 1)
LISTEN_HASH = {
    "S_G_ecg_id.txt": "2f6fbc6d9fabd71d5a24bc81a16d45aaf577dd7315d41f633722d80b4b95a4c8",
    "S_G_stratum.csv": "1d6ee7894033151a0dc1c0a2805111d9054ba28b3f35132cfb6e86800e9e674b",
    "PT_ecg_id.txt": "ef7f6c018091686cb17c5a714a613a6e0594c67093872367f284f8c026156224",
    "P0_ecg_id.txt": "d50ddf704b3fb6ef13c649c27baa0c470c9ba959faf9ed92b6eaf12e240a9e0a",
    "nicht_erst_ecg_id.txt": "0cc5e7de7733c7fd8c98cf459670fafd784d5437ac88c76fd5794e2c7a55d2c3",
    "T0_ecg_id.txt": "47e3cefee0eb67efb8b9bf4e6b3b8f11963b3cc875b93be2f3dca74dd55688ad",
}
AUDIT_HASH = "5af97a7d52ce117d5c62082c06e0c4828ecb68acc32bad444f20a0dde263ed14"
BH_LUDB_HASH = {  # BRAVEHEART LUDB nativ (AW-208), am 15.10.2026 vor der Auswertung berechnet (nicht in DATENSPERRE_2026-10-15.md)
    "ludb/ausgabe_langformat.csv": "74b46a99635b748697403919f4fa2838fa6133b284358646e73a5f047098baa0",
    "ludb/processed_ecgs_roh.csv.gz": "c060b97b72a41eaa0848d598d595185e68e6f1674f2eaa3a7d6f724bd3ee0665",
    "ludb/annotations_ludb.tar.gz": "3977e2ea0660576f170b41772244643caecf58f5ce7bbb2fc4be55c8dc228cc7",
}
ARCHIV_HASH = {  # V2 Abschnitt 3
    "12sl_features.csv": "cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29",
    "unig_features.csv": "f95ab01df5c9ed329485fd811578f0c39eabb0667a36ec37c7dda427d9d8f5a5",
    "ecgdeli_features.csv": "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556",
}
STRATUM_N_P0 = {"S1": 225, "S2": 1529, "S3": 1011, "S4": 8565, "S5": 7539}  # V2 4.2
STRATUM_N_GEZOGEN = {"S1": 150, "S2": 200, "S3": 100, "S4": 275, "S5": 275}

STATE = {"log": None, "pruef": [], "stufen": [], "dateien": [], "test": False}


# ---------------------------------------------------------------------------------------------------------------
# 1. Hilfsfunktionen
# ---------------------------------------------------------------------------------------------------------------
def log(msg):
    z = time.strftime("%H:%M:%S") + " " + str(msg)
    print(z, flush=True)
    if STATE["log"] is not None:
        with open(STATE["log"], "a", encoding="utf-8") as f:
            f.write(z + "\n")


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def wilson(p, n, z=Z95):
    """Wilson-Intervall fuer einen Anteil p bei (effektiver) Stichprobengroesse n."""
    if n is None or not np.isfinite(n) or n <= 0 or not np.isfinite(p):
        return (float("nan"), float("nan"))
    z2 = z * z
    zentrum = (p + z2 / (2 * n)) / (1 + z2 / n)
    halb = z * math.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0.0)) / (1 + z2 / n)
    return (max(zentrum - halb, 0.0), min(zentrum + halb, 1.0))


def gew_anteil(x, w):
    """Gewichteter Anteil, Kish-Effektivgroesse und Wilson-Intervall (V2 4.2)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    sw = w.sum()
    if sw <= 0 or len(x) == 0:
        return dict(p=float("nan"), n=len(x), n_eff=float("nan"), lo=float("nan"), hi=float("nan"))
    p = float((w * x).sum() / sw)
    n_eff = float(sw * sw / (w * w).sum())
    lo, hi = wilson(p, n_eff)
    return dict(p=p, n=len(x), n_eff=n_eff, lo=lo, hi=hi)


def holm(pwerte, alpha=ALPHA):
    """Holm: aufsteigend sortieren, k-ter kleinster p gegen alpha/(m-k+1), nach der ersten Nichtablehnung stoppen."""
    m = len(pwerte)
    ordnung = sorted(range(m), key=lambda i: pwerte[i])
    abgelehnt = [False] * m
    for k, i in enumerate(ordnung, 1):
        if pwerte[i] <= alpha / (m - k + 1):
            abgelehnt[i] = True
        else:
            break
    return abgelehnt


BAND_TEXT = {
    "diskordanz": ["kaum diskordant", "gering", "deutlich", "überwiegend verschieden zugeordnet"],
    "zeit": ["innerhalb der Abtastauflösung", "klein", "mittel", "groß"],
    "winkel": ["klein", "mittel", "deutlich", "groß"],
    "verstaerkung": ["praktisch unverändert", "gelegentlich verändert", "häufig verändert", "überwiegend verändert"],
    "ausfall": ["selten", "gelegentlich", "häufig", "Selektion wahrscheinlich"],
}


def band_rang(tabelle, wert):
    """Rang 1..4 nach V2 6.4. Anteile als Bruch (0..1), Zeit in ms, Winkel in Grad. 'bis x' schliesst x ein, 'unter x' schliesst x aus."""
    if wert is None or not np.isfinite(wert):
        return None
    if tabelle in ("diskordanz", "verstaerkung"):
        pr = 100.0 * wert
        return 1 if pr < 1 else 2 if pr < 5 else 3 if pr < 20 else 4
    if tabelle == "zeit":
        return 1 if wert <= 4 else 2 if wert <= 10 else 3 if wert <= 20 else 4
    if tabelle == "winkel":
        return 1 if wert <= 5 else 2 if wert <= 10 else 3 if wert <= 20 else 4
    if tabelle == "ausfall":
        pr = 100.0 * wert
        return 1 if pr <= 1 else 2 if pr <= 5 else 3 if pr <= 10 else 4
    raise ValueError(tabelle)


def band_text(tabelle, wert):
    r = band_rang(tabelle, wert)
    return None if r is None else BAND_TEXT[tabelle][r - 1]


def band_mit_ki(tabelle, punkt, lo, hi):
    """Regel V2 6.4: Band am Punktschaetzer; liegt eine KI-Grenze in einem anderen Band, 'Band X bis Y'."""
    rp, rl, rh = band_rang(tabelle, punkt), band_rang(tabelle, lo), band_rang(tabelle, hi)
    if rp is None:
        return None
    raenge = [r for r in (rp, rl, rh) if r is not None]
    if len(set(raenge)) == 1:
        return BAND_TEXT[tabelle][rp - 1]
    return "Band %s bis %s" % (BAND_TEXT[tabelle][min(raenge) - 1], BAND_TEXT[tabelle][max(raenge) - 1])


def fz(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return ("%." + str(nd) + "f") % x


def proz(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return ("%." + str(nd) + "f") % (100.0 * x)


# ---------------------------------------------------------------------------------------------------------------
# 2. Bootstrap-Maschine
# ---------------------------------------------------------------------------------------------------------------
class Einheit:
    """Eine Auswertungseinheit: Beobachtungen mit Clusterindex (Person bzw. Aufnahme), Differenz d, 0/1-Indikatoren,
    Verhaeltnisse, Kreuztabellen und Quantile. Gewichte der Ziehung = Zaehlung des Clusters (mal Designgewicht gew)."""

    def __init__(self, name, cluster, d, ind=None, ratios=None, cross=None, quant=None, gew=None, behalte=None, meta=None):
        self.name = name
        self.cluster = np.asarray(cluster, dtype=np.int64)
        self.d = np.asarray(d, dtype=float)
        self.ind = {k: np.asarray(v, float) for k, v in (ind or {}).items()}
        self.ratios = {k: (np.asarray(a, float), np.asarray(b, float)) for k, (a, b) in (ratios or {}).items()}
        self.cross = {k: tuple(np.asarray(x, float) for x in v) for k, v in (cross or {}).items()}
        self.quant = {k: (np.asarray(v, float), tuple(p)) for k, (v, p) in (quant or {}).items()}
        self.gew = None if gew is None else np.asarray(gew, float)
        if self.gew is not None and self.quant:
            raise ValueError("Quantile nur ohne Designgewichte")
        self.behalte = behalte
        self.meta = meta or {}
        self.n = len(self.d)


def stat_W(e, W):
    """Statistiken einer Einheit fuer eine Gewichtsmatrix W (m, n). Rueckgabe: dict Name -> Array (m,)."""
    out = {}
    if e.n == 0:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        N = W.sum(axis=1)
        out["n_gewicht"] = N
        out["bias"] = (W @ e.d) / N
        out["mae"] = (W @ np.abs(e.d)) / N
        for k, x in e.ind.items():
            out[k] = (W @ x) / N
        for k, (num, den) in e.ratios.items():
            out[k] = (W @ num) / (W @ den)
        for k, (A, B_, C_, D_) in e.cross.items():
            a, b, c, d = W @ A, W @ B_, W @ C_, W @ D_
            n = a + b + c + d
            out[k + "_disk"] = (b + c) / n
            out[k + "_ppa"] = 2 * a / (2 * a + b + c)
            out[k + "_npa"] = 2 * d / (2 * d + b + c)
            po = (a + d) / n
            pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
            out[k + "_kappa"] = (po - pe) / (1 - pe)
        for k, (v, probs) in e.quant.items():
            order = np.argsort(v, kind="stable")
            vs = v[order]
            cum = np.cumsum(W[:, order], axis=1)
            Nq = cum[:, -1]
            n_obs = vs.shape[0]
            for p in probs:
                h = (Nq - 1.0) * p
                j = np.floor(h)
                fr = h - j
                k1 = np.minimum((cum <= j[:, None]).sum(axis=1), n_obs - 1)
                k2 = np.minimum((cum <= (j + 1)[:, None]).sum(axis=1), n_obs - 1)
                qv = vs[k1] + fr * (vs[k2] - vs[k1])
                qv[Nq <= 0] = np.nan  # Ziehung ohne Treffer dieser Einheit: kein Quantil
                out["%s_q%g" % (k, p)] = qv
    if e.behalte is not None:
        out = {k: v for k, v in out.items() if k in e.behalte or k == "n_gewicht"}
    return out


def punktschaetzung(e):
    """Punktschaetzer mit denselben Formeln (Gewicht 1 bzw. Designgewicht je Beobachtung)."""
    W = np.ones((1, e.n)) if e.gew is None else e.gew[None, :].astype(float)
    return {k: float(v[0]) for k, v in stat_W(e, W).items()}


def zieh_zaehlungen(rng, K, m, strata=None):
    """m Ziehungen mit Zuruecklegen von K Clustern: Zaehlungsmatrix (m, K). Mit strata (Label je Cluster): innerhalb der Strata."""
    if strata is None:
        return rng.multinomial(K, np.full(K, 1.0 / K), size=m)
    C = np.zeros((m, K), dtype=np.int64)
    strata = np.asarray(strata)
    for lab in np.unique(strata):
        idx = np.flatnonzero(strata == lab)
        C[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=m)
    return C


def boot_lauf(einheiten, K, strata=None, B=None, chunk=None, seed=None):
    """Gemeinsame Ziehungen fuer alle Einheiten. Chunk c nutzt SeedSequence([seed, c]); unabhaengig von der Reihenfolge."""
    B = B_ZIEHUNGEN if B is None else B
    chunk = CHUNK if chunk is None else chunk
    seed = SEED_BOOT if seed is None else seed
    res = {e.name: {} for e in einheiten}
    for ci, start in enumerate(range(0, B, chunk)):
        m = min(chunk, B - start)
        rng = np.random.default_rng(np.random.SeedSequence([seed, ci]))
        C = zieh_zaehlungen(rng, K, m, strata)
        for e in einheiten:
            W = C[:, e.cluster].astype(np.float64)
            if e.gew is not None:
                W = W * e.gew[None, :]
            st = stat_W(e, W)
            for k, v in st.items():
                res[e.name].setdefault(k, []).append(v)
    return {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in res.items()}


def ki95(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return (float(lo), float(hi))


def boot_p_einseitig_kleiner(arr, marge, minimum=1e-4):
    """Einseitiger Bootstrap-p-Wert fuer H: Groesse > marge: Anteil der Schaetzer, die hoechstens an der Marge liegen; Minimum 1e-4."""
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(max(np.mean(a <= marge), minimum))


# ---------------------------------------------------------------------------------------------------------------
# 3. Hashpruefung (V2 11.2 Punkt 3; DATENSPERRE_2026-10-15.md)
# ---------------------------------------------------------------------------------------------------------------
def _zeile(datei, soll, ist, ok, hinweis=""):
    STATE["pruef"].append(dict(datei=str(datei), soll=soll, ist=ist, ok=bool(ok), hinweis=hinweis))


def hashpruefung():
    """Prueft die drei Hashlisten, dann jede darin genannte Datei, die AW-219-Listen, die Audit-Liste und die Archivdateien.
    Rueckgabe: True, wenn alle gebundenen Hashes stimmen. Dateien ausserhalb der Datensperre werden nur gemeldet."""
    P = PFADE
    STATE["pruef"] = []
    listen = {"sperre": P["sperre"], "manifest": P["erg"] / "looplens/langformat/MANIFEST.json", "aw220": P["erg"] / "AW220_HASHES.txt"}
    for k, p in listen.items():
        ist = sha256_datei(p) if Path(p).exists() else "FEHLT"
        _zeile(p, SPERRE_LISTEN[k], ist, ist == SPERRE_LISTEN[k], "Hashliste")
    # Hashliste 1: JSON mit absoluten Pfaden
    try:
        j = json.loads(Path(listen["sperre"]).read_text(encoding="utf-8"))
        for pfad, v in j["dateien"].items():
            ist = sha256_datei(pfad) if Path(pfad).exists() else "FEHLT"
            _zeile(pfad, v["sha256"], ist, ist == v["sha256"], "datensperre_hashes_entwurf.json")
    except Exception as ex:  # noqa: BLE001
        _zeile(listen["sperre"], "lesbar", "Fehler: %s" % ex, False)
    # Hashliste 2: MANIFEST.json (Tabellen des LoopLens-Langformats)
    try:
        m = json.loads(Path(listen["manifest"]).read_text(encoding="utf-8"))
        for n, v in m["tabellen"].items():
            pfad = Path(listen["manifest"]).parent / n
            ist = sha256_datei(pfad) if pfad.exists() else "FEHLT"
            _zeile(pfad, v["sha256"], ist, ist == v["sha256"], "MANIFEST.json looplens")
    except Exception as ex:  # noqa: BLE001
        _zeile(listen["manifest"], "lesbar", "Fehler: %s" % ex, False)
    # Hashliste 3: AW220_HASHES.txt (Ergebnisse relativ zu ergebnisse, Skripte relativ zu werkzeuge/benchmark)
    skripte = False
    try:
        for z in Path(listen["aw220"]).read_text(encoding="utf-8").splitlines():
            z = z.strip()
            if not z:
                continue
            if z.startswith("#"):
                skripte = "Skripte" in z
                continue
            mt = re.match(r"^([0-9a-f]{64}) \*(.+)$", z)
            if not mt:
                _zeile(z, "Zeilenformat", "unlesbar", False, "AW220_HASHES.txt")
                continue
            pfad = (P["wz"] if skripte else P["erg"]) / mt.group(2)
            ist = sha256_datei(pfad) if pfad.exists() else "FEHLT"
            _zeile(pfad, mt.group(1), ist, ist == mt.group(1), "AW220_HASHES.txt " + ("Skript" if skripte else "Ergebnis"))
    except Exception as ex:  # noqa: BLE001
        _zeile(listen["aw220"], "lesbar", "Fehler: %s" % ex, False)
    # Mengen AW-219 (Hashes aus LOOPLENS_LAUF_AW219.md), Audit-Liste, Archivdateien (V2 Abschnitt 3)
    for n, soll in LISTEN_HASH.items():
        p = P["listen"] / n
        ist = sha256_datei(p) if p.exists() else "FEHLT"
        _zeile(p, soll, ist, ist == soll, "Liste AW-219")
    ist = sha256_datei(P["audit"]) if Path(P["audit"]).exists() else "FEHLT"
    _zeile(P["audit"], AUDIT_HASH, ist, ist == AUDIT_HASH, "Audit-Stichprobe rekonstruiert (Nachtrag 0 Abschnitt 3)")
    for n, soll in ARCHIV_HASH.items():
        p = P["plus"] / n
        ist = sha256_datei(p) if p.exists() else "FEHLT"
        _zeile(p, soll, ist, ist == soll, "Archivprodukt PTB-XL+ (V2 Abschnitt 3)")
    # BRAVEHEART LUDB nativ (AW-208) steht nicht in der Datensperre; die Hashes wurden vor der Auswertung am 15.10.2026 festgehalten und hier gebunden
    for rel, soll in BH_LUDB_HASH.items():
        p = P["erg"] / rel
        ist = sha256_datei(p) if p.exists() else "FEHLT"
        _zeile(p, soll, ist, ist == soll, "BRAVEHEART LUDB nativ AW-208, nicht in Datensperre, Hash vor der Auswertung festgehalten")
    # offene Datensaetze: Pruefsummen der Veroeffentlichung (Metadatentabelle PTB-XL, LUDB-Expertenannotationen)
    try:
        for z in (Path(P["meta"]).parent / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            mt = re.match(r"^([0-9a-f]{64}) \*?(ptbxl_database\.csv)$", z.strip())
            if mt:
                ist = sha256_datei(P["meta"])
                _zeile(P["meta"], mt.group(1), ist, ist == mt.group(1), "PTB-XL SHA256SUMS.txt")
        n_ann = 0
        for z in (Path(P["ludb"]).parent / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            mt = re.match(r"^([0-9a-f]{64}) \*?(data/\d+\.(i|ii|iii|avr|avl|avf|v1|v2|v3|v4|v5|v6))$", z.strip())
            if mt:
                p = Path(P["ludb"]).parent / mt.group(2)
                ist = sha256_datei(p) if p.exists() else "FEHLT"
                _zeile(p, mt.group(1), ist, ist == mt.group(1), "LUDB SHA256SUMS.txt (Expertenannotation)")
                n_ann += 1
        if n_ann != 2400:
            _zeile("LUDB-Annotationen", "2400 Dateien", "%d" % n_ann, False, "SHA256SUMS.txt")
    except Exception as ex:  # noqa: BLE001
        _zeile("Pruefsummen offene Datensaetze", "lesbar", "Fehler: %s" % ex, False)
    alle_ok = all(z["ok"] for z in STATE["pruef"])
    return alle_ok


def schreibe_hashpruefung(outdir):
    df = pd.DataFrame(STATE["pruef"])
    df.to_csv(Path(outdir) / "00_hashpruefung.csv", index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------------------------------------------
# 4. Laden
# ---------------------------------------------------------------------------------------------------------------
ORDNUNG = ["12SL", "BRAVEHEART", "ECGDeli", "LoopLens", "NeuroKit2", "Uni-G"]  # ASCII-Reihenfolge (V2 3, Regel 2)
KONTROLLEN = []


def kontrolle(name, soll, ist, ok, hinweis=""):
    KONTROLLEN.append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok), hinweis=hinweis))
    log("KONTROLLE %s: soll %s ist %s -> %s" % (name, soll, ist, "ok" if ok else "ABWEICHUNG"))


def lese_ids(p):
    return np.array(sorted(int(x) for x in Path(p).read_text(encoding="ascii").split()), dtype=np.int64)


def lade_meta():
    P = PFADE
    spalten = ["ecg_id", "patient_id", "age", "sex", "device", "scp_codes", "baseline_drift", "static_noise", "burst_noise",
               "electrodes_problems", "extra_beats", "pacemaker", "strat_fold"]
    d = pd.read_csv(P["meta"], usecols=spalten)
    d = d.sort_values("ecg_id").reset_index(drop=True)
    d["codes"] = d.scp_codes.map(lambda s: set(ast.literal_eval(s).keys()))

    def nz(c):
        return (d[c].fillna("").astype(str).str.strip() != "").to_numpy()

    d["marker_rausch"] = nz("baseline_drift") | nz("static_noise") | nz("burst_noise") | nz("electrodes_problems")
    d["marker_extra"] = nz("extra_beats")
    d["marker_pace"] = nz("pacemaker")
    d["pace"] = d.marker_pace.to_numpy() | d.codes.map(lambda c: "PACE" in c).to_numpy()
    d["breit"] = d.codes.map(lambda c: bool(c & {"CLBBB", "CRBBB", "IVCD"}))
    d["af"] = d.codes.map(lambda c: bool(c & {"AFIB", "AFLT"}))
    d["sr"] = d.codes.map(lambda c: "SR" in c)
    d["rhythmus"] = np.where(d.af, "AFIB_AFLT", np.where(d.sr, "SR", "sonstig"))
    dev = d.device.fillna("").astype(str).str.strip()
    d["geraet"] = np.select([dev.str.startswith("AT-60"), dev.str.startswith("AT-6"), dev.str.startswith("CS-12"), dev.str.startswith("CS100")],
                            ["AT-60", "AT-6", "CS-12", "CS100"], default="andere")
    alter = d.age.astype(float)
    d["altersgruppe"] = np.select([alter < 18, alter < 40, alter < 60, alter < 80], ["unter_18", "18_39", "40_59", "60_79"],
                                  default="ab_80_inkl_ueber_89")
    d["geschlecht"] = np.where(d.sex == 0, "sex_0", "sex_1")

    def stratum(codes, pace, breit, af):
        if pace:
            return "S1"
        if breit:
            return "S2"
        if af:
            return "S3"
        if "NORM" in codes:
            return "S4"
        return "S5"

    d["stratum"] = [stratum(c, p, b, a) for c, p, b, a in zip(d.codes, d.pace, d.breit, d.af)]
    d["erst"] = ~d.duplicated("patient_id", keep="first")
    return d


class Tab:
    """Breite Tabelle aus einem Langformat: Werte und Status je (Datensatz, Vorverarbeitung, Aufnahme) x Messgroesse."""

    def __init__(self, v, s):
        self.v, self.s = v, s

    @staticmethod
    def aus_langformat(df):
        df = df.copy()
        df["aufnahme"] = df["aufnahme"].astype(str).astype(int)
        df["wert"] = pd.to_numeric(df["wert"], errors="coerce")
        idx = ["datensatz", "vorverarbeitung", "aufnahme"]
        dup = int(df.duplicated(idx + ["messgroesse"]).sum())
        if dup:
            raise ValueError("Doppelte Zeilen im Langformat: %d" % dup)
        v = df.pivot(index=idx, columns="messgroesse", values="wert").sort_index()
        s = df.pivot(index=idx, columns="messgroesse", values="status").sort_index()
        return Tab(v, s)

    @staticmethod
    def aus_liste(tabs):
        return Tab(pd.concat([t.v for t in tabs]).sort_index(), pd.concat([t.s for t in tabs]).sort_index())

    def hole(self, ds, vv, key, ids):
        n = len(ids)
        leer = (np.full(n, np.nan), np.full(n, "nicht_im_lauf", dtype=object))
        try:
            sv = self.v.loc[(ds, vv)]
            ss = self.s.loc[(ds, vv)]
        except KeyError:
            return leer
        if key not in sv.columns:
            return leer
        val = sv[key].reindex(ids).to_numpy(float)
        st = ss[key].reindex(ids).to_numpy(object)
        st = np.where(pd.isna(st), "nicht_im_lauf", st).astype(object)
        bad = (st == "ok") & ~np.isfinite(val)
        st[bad] = "ok_nicht_endlich"
        val = np.where(st == "ok", val, np.nan)
        return val, st

    def hat(self, ds, vv):
        try:
            self.v.loc[(ds, vv)]
            return True
        except KeyError:
            return False


LANG_SPALTEN = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status"]


def lese_lang(pfad):
    return pd.read_csv(pfad, usecols=LANG_SPALTEN, dtype={"datensatz": str, "aufnahme": str, "vorverarbeitung": str,
                                                         "messgroesse": str, "wert": str, "status": str})


def id_aus_name(name):
    m = re.search(r"(?:ptbxl|ludb)_(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else None


def lese_bh_roh(pfad, spalten):
    """processed_ecgs_roh.csv.gz: nur die genannten Spalten; Aufnahme-ID aus filename."""
    df = pd.read_csv(pfad, usecols=["filename"] + list(spalten))
    df["ecg"] = df.filename.map(id_aus_name)
    return df.set_index("ecg")


def lese_check_liste(pfad):
    if not Path(pfad).exists():
        return set()
    df = pd.read_csv(pfad, usecols=["filename"])
    return set(int(x) for x in df.filename.map(id_aus_name).dropna())


def lese_anno_tar(pfad):
    """BRAVEHEART-Schlagmarken: dict id -> Array (n,4) [Q, QRS, S, Tend] in Samples."""
    out = {}
    if not Path(pfad).exists():
        return out
    zahl = r"(-?\d+|NaN|nan)"
    muster = re.compile(r"^\s*" + zahl + "," + zahl + "," + zahl + "," + zahl + r"\s*$")
    with tarfile.open(pfad, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile() or not m.name.endswith(".anno"):
                continue
            i = id_aus_name(m.name)
            if i is None:
                continue
            txt = t.extractfile(m).read().decode("latin1")
            zeilen = []
            for z in txt.splitlines():
                mt = muster.match(z)
                if mt:
                    # MATLAB-Indizes sind 1-basiert (Beats.m); alle anderen Werkzeuge und LUDB zaehlen ab 0: 1 abziehen
                    zeilen.append([(float(mt.group(k)) - 1.0) if mt.group(k).lower() != "nan" else np.nan for k in (1, 2, 3, 4)])
                else:
                    break
            out[i] = np.array(zeilen, dtype=float).reshape(-1, 4)
    return out


ARCHIV_SPALTEN = {
    "12SL": {"qrs_dauer_global": "QRS_Dur_Global", "qt_global": "QT_Int_Global", "qtcf": "QT_IntFridericia_Global",
             "pr_intervall": "PR_Int_Global", "p_dauer": "P_Dur_Global"},
    "Uni-G": {"qrs_dauer_global": "QRS_Dur_Global", "qt_global": "QT_Int_Global", "qtcf": "QT_IntFridericia_Global",
              "pr_intervall": "PR_Int_Global", "p_dauer": "P_Dur_Global"},
    "ECGDeli": {"qrs_dauer_global": "QRS_Dur_Global", "qt_global": "QT_Int_Global", "pr_intervall": "PQ_Int_Global",
                "p_dauer": "P_Dur_Global"},
}
ARCHIV_DATEI = {"12SL": "12sl_features.csv", "Uni-G": "unig_features.csv", "ECGDeli": "ecgdeli_features.csv"}


def lade_archiv(tool, ids_all):
    """Archivwerte (ms) ausgerichtet auf ids_all. Status: ok | fehlend | nicht_definiert (keine P-Welle laut P_Found_Global) | nicht_im_lauf."""
    pfad = PFADE["plus"] / ARCHIV_DATEI[tool]
    mapping = ARCHIV_SPALTEN[tool]
    kopf = list(pd.read_csv(pfad, nrows=0).columns)
    cols = ["ecg_id"] + sorted(set(mapping.values()))
    hat_pfound = "P_Found_Global" in kopf
    if hat_pfound:
        cols.append("P_Found_Global")
    df = pd.read_csv(pfad, usecols=cols)
    df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id")
    imlauf = np.isin(ids_all, df.index.to_numpy())
    df = df.reindex(ids_all)
    out = {}
    for g, c in mapping.items():
        v = df[c].to_numpy(float)
        st = np.where(~imlauf, "nicht_im_lauf", np.where(np.isfinite(v), "ok", "fehlend")).astype(object)
        if hat_pfound and g in ("pr_intervall", "p_dauer"):
            pf = df["P_Found_Global"].to_numpy(float)
            st = np.where((st == "fehlend") & (pf == 0), "nicht_definiert", st).astype(object)
        out[g] = (np.where(st == "ok", v, np.nan), st)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 5. Daten, Quellen, Einheiten
# ---------------------------------------------------------------------------------------------------------------
ZEIT_KEYS = {"qrs_dauer_global", "qt_global", "qtcf", "pr_intervall", "p_dauer"}
ALLE_TOOLS_ZEIT = ["LoopLens", "BRAVEHEART", "12SL", "ECGDeli", "Uni-G"]
GROESSEN = {
    "qrs_dauer_global": dict(art="zeit", schwellen=(110, 120, 130), tools=["LoopLens", "BRAVEHEART", "12SL", "ECGDeli", "Uni-G"]),
    "qt_global": dict(art="zeit", schwellen=(), tools=["LoopLens", "BRAVEHEART", "12SL", "ECGDeli", "Uni-G"]),
    "qtcf": dict(art="zeit", schwellen=(480, 500), tools=["LoopLens", "BRAVEHEART", "12SL", "Uni-G"]),
    "pr_intervall": dict(art="zeit", schwellen=(), tools=["LoopLens", "12SL", "ECGDeli", "Uni-G"]),
    "p_dauer": dict(art="zeit", schwellen=(), tools=["LoopLens", "12SL", "ECGDeli", "Uni-G"]),
    "qrst_winkel_peak": dict(art="winkel", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
    "qrst_winkel_integral": dict(art="winkel", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
    "vektorbetrag_max_qrs": dict(art="betrag", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
    "integralbetrag_qrs": dict(art="betrag", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
    "sai_qrst": dict(art="betrag", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
    "schleifenlaenge_qrs": dict(art="betrag", schwellen=(), tools=["LoopLens", "BRAVEHEART"]),
}
UNDEFINIERT_PAAR = {"LoopLens/BRAVEHEART verschieden definiert": {"qrst_winkel_peak", "qrst_winkel_integral", "vektorbetrag_max_qrs",
                                                                  "integralbetrag_qrs", "sai_qrst", "schleifenlaenge_qrs"}}
AUSFALL_STATUS = ("werkzeugfehler", "fehlend", "ausgeschlossen_qc", "nicht_zuordenbar", "ok_nicht_endlich", "nicht_im_lauf", "nicht_ladbar")


def ll_key(g):
    return g if g in ZEIT_KEYS else g + "_ll"


def schluessel_von(g, tool):
    """Namen mit Operatorsuffix (Schema-Nachtrag AW-221): nur bei verschieden definierten Groessen _ll bzw. _bh."""
    if g in UNDEFINIERT_PAAR["LoopLens/BRAVEHEART verschieden definiert"]:
        return g + {"LoopLens": "_ll", "BRAVEHEART": "_bh"}.get(tool, "")
    return g


class Daten:
    pass


def orientiere(t1, t2):
    """Differenz = Kandidat minus Komparator; LoopLens ist Kandidat, sonst ist das in ORDNUNG spaetere Werkzeug der Kandidat (V2 3, Regel 3)."""
    if "LoopLens" in (t1, t2):
        return ("LoopLens", t2 if t1 == "LoopLens" else t1)
    a, b = sorted((t1, t2), key=ORDNUNG.index)
    return (b, a)


def kontrolle_k2():
    """K2 (V2 11.1): Referenz um +20 ms verschoben, Harness: Bias gleich +20 ms mit Abweichung hoechstens 1e-9 (synthetische Differenzen)."""
    rng = np.random.default_rng(14102027)
    ref = np.round(rng.normal(100, 15, 500) / 2) * 2
    e = baue_einheit("k2", ref + 20.0, ref, np.arange(500), "zeit", (120,))
    pk = punktschaetzung(e)
    ok = abs(pk["bias"] - 20.0) <= 1e-9 and abs(pk["mae"] - 20.0) <= 1e-9 and abs(pk["absd_q0.5"] - 20.0) <= 1e-9 and abs(pk["d_q0.5"] - 20.0) <= 1e-9
    kontrolle("K2 Harness: Verschiebung um +20 ms ergibt Bias +20 ms", "20 (1e-9)", "%.12g" % pk["bias"], ok)


def lade_alles():
    P = PFADE
    d = Daten()
    meta = lade_meta()
    d.meta = meta
    d.ids_all = meta.ecg_id.to_numpy(np.int64)
    d.pos = pd.Series(np.arange(len(d.ids_all)), index=d.ids_all)
    d.p0 = lese_ids(P["listen"] / "P0_ecg_id.txt")
    d.pt = lese_ids(P["listen"] / "PT_ecg_id.txt")
    d.sg = lese_ids(P["listen"] / "S_G_ecg_id.txt")
    d.nicht_erst = lese_ids(P["listen"] / "nicht_erst_ecg_id.txt")
    sgs = pd.read_csv(P["listen"] / "S_G_stratum.csv")
    d.sg_stratum = sgs.set_index("ecg_id").stratum.reindex(d.sg).to_numpy()
    erst_ids = d.ids_all[meta.erst.to_numpy()]
    kontrolle("L1 P0 gleich niedrigste ecg_id je Person (Metadaten)", len(erst_ids), len(d.p0), bool(np.array_equal(erst_ids, d.p0)))
    kontrolle("L2 nicht-Erstaufnahmen gleich Liste", len(d.nicht_erst), int((~meta.erst).sum()),
              bool(np.array_equal(d.ids_all[~meta.erst.to_numpy()], d.nicht_erst)))
    kontrolle("L3 S_G Teilmenge von P0", len(d.sg), int(np.isin(d.sg, d.p0).sum()), bool(np.isin(d.sg, d.p0).all()))
    kontrolle("L4 PT disjunkt zu P0", 0, int(np.isin(d.pt, d.p0).sum()), not bool(np.isin(d.pt, d.p0).any()))
    d.idx_p0 = d.pos[d.p0].to_numpy()
    pat = meta.patient_id.to_numpy()
    pat2person = {p: k for k, p in enumerate(pat[d.idx_p0])}
    d.person_all = np.array([pat2person.get(p, -1) for p in pat], dtype=np.int64)
    kontrolle("L5 jede Aufnahme gehoert zu einer P0-Person", 0, int((d.person_all < 0).sum()), bool((d.person_all >= 0).all()))
    d.K = len(d.p0)
    # Strata S_G: Zaehlung und Gewichte
    p0meta = meta.iloc[d.idx_p0]
    zaehl = p0meta.stratum.value_counts().to_dict()
    kontrolle("L6 Stratumgroessen in P0 gleich V2 4.2", STRATUM_N_P0, {k: int(zaehl.get(k, 0)) for k in STRATUM_N_P0},
              all(int(zaehl.get(k, 0)) == v for k, v in STRATUM_N_P0.items()), "im Test nicht erwartet" if STATE["test"] else "")
    sg_meta = meta.iloc[d.pos[d.sg].to_numpy()]
    kontrolle("L7 S_G-Strata aus Metadaten gleich Stratumliste", "gleich", "gleich" if np.array_equal(sg_meta.stratum.to_numpy(), d.sg_stratum) else "verschieden",
              bool(np.array_equal(sg_meta.stratum.to_numpy(), d.sg_stratum)))
    nh = {k: (int(zaehl.get(k, 0)) if STATE["test"] else STRATUM_N_P0[k]) for k in STRATUM_N_P0}
    nsg = pd.Series(d.sg_stratum).value_counts().to_dict()
    d.sg_gew = np.array([nh[s] / nsg[s] for s in d.sg_stratum], dtype=float)
    d.sg_person = d.pos[d.sg].to_numpy()
    # V2 4.1 Zaehlproben
    if not STATE["test"]:
        kontrolle("M1 PTB-XL Aufnahmen / Personen", "21799 / 18869", "%d / %d" % (len(meta), len(d.p0)), len(meta) == 21799 and len(d.p0) == 18869)
        REF_M2 = reference("bench_auswertung.m2_rauschmarker")
        if REF_M2 is None:
            print("NOTE: no local reference value 'bench_auswertung.m2_rauschmarker'; check skipped.")
        else:
            kontrolle("M2 mindestens ein Rauschmarker", REF_M2, int(meta.marker_rausch.sum()), int(meta.marker_rausch.sum()) == REF_M2)
        REF_M3 = reference("bench_auswertung.m3_extra_beats_pacemaker")
        if REF_M3 is None:
            print("NOTE: no local reference value 'bench_auswertung.m3_extra_beats_pacemaker'; check skipped.")
        else:
            kontrolle("M3 extra_beats / pacemaker", "%d / %d" % tuple(REF_M3), "%d / %d" % (meta.marker_extra.sum(), meta.marker_pace.sum()),
                      meta.marker_extra.sum() == REF_M3[0] and meta.marker_pace.sum() == REF_M3[1])
        g = meta.geraet.value_counts().to_dict()
        REF_M4 = reference("bench_auswertung.m4_geraetegruppen")
        if REF_M4 is None:
            print("NOTE: no local reference value 'bench_auswertung.m4_geraetegruppen'; check skipped.")
        else:
            kontrolle("M4 Geraetegruppen", ", ".join("%s %d" % kv for kv in REF_M4.items()), str(g),
                      all(g.get(k_) == v_ for k_, v_ in REF_M4.items()))
        REF_M5 = reference("bench_auswertung.m5_fold10")
        if REF_M5 is None:
            print("NOTE: no local reference value 'bench_auswertung.m5_fold10'; check skipped.")
        else:
            kontrolle("M5 Fold 10 gesamt / in P0", "%d / %d" % tuple(REF_M5), "%d / %d" % ((meta.strat_fold == 10).sum(), (p0meta.strat_fold == 10).sum()),
                      (meta.strat_fold == 10).sum() == REF_M5[0] and (p0meta.strat_fold == 10).sum() == REF_M5[1])
        REF_M6 = reference("bench_auswertung.m6_unter18_p0")
        if REF_M6 is None:
            print("NOTE: no local reference value 'bench_auswertung.m6_unter18_p0'; check skipped.")
        else:
            kontrolle("M6 unter 18 Jahre in P0", REF_M6, int((p0meta.altersgruppe == "unter_18").sum()), int((p0meta.altersgruppe == "unter_18").sum()) == REF_M6)
    # Audit-Liste
    au = pd.read_csv(P["audit"])
    d.audit_ids = au.ecg_id.to_numpy(np.int64)
    d.audit_p0 = au[au.in_P0 == 1].ecg_id.to_numpy(np.int64)
    REF_M7 = reference("bench_auswertung.m7_audit_in_p0")
    if REF_M7 is None:
        print("NOTE: no local reference value 'bench_auswertung.m7_audit_in_p0'; check skipped.")
    else:
        kontrolle("M7 Audit-Stichprobe: Aufnahmen / in P0", "2000 / %d" % REF_M7, "%d / %d" % (len(au), len(d.audit_p0)),
                  True if STATE["test"] else (len(au) == 2000 and len(d.audit_p0) == REF_M7))
    # LoopLens
    erg = P["erg"]
    log("lade LoopLens")
    d.ll = Tab.aus_langformat(lese_lang(erg / "looplens/langformat/looplens_langformat.csv.gz"))
    d.llqc = Tab.aus_langformat(lese_lang(erg / "looplens/langformat/looplens_qc_ausgeschlossen_werte.csv.gz"))
    d.ll_beats = pd.read_csv(erg / "looplens/langformat/looplens_schlaege.csv.gz",
                             dtype={"datensatz": str, "vorverarbeitung": str})
    # ll_lauf.py kodiert fehlende Marken als -1 (Befund der Vorabpruefung): als fehlend lesen
    for c_ in ("p_beginn", "p_ende", "qrs_beginn", "qrs_ende", "t_ende"):
        d.ll_beats[c_] = pd.to_numeric(d.ll_beats[c_], errors="coerce")
        d.ll_beats.loc[d.ll_beats[c_] < 0, c_] = np.nan
    # BRAVEHEART nativ PTB-XL, LUDB, Arme
    log("lade BRAVEHEART")
    d.bh = Tab.aus_langformat(lese_lang(erg / "ptbxl/ausgabe_langformat.csv"))
    d.bh_roh = lese_bh_roh(erg / "ptbxl/processed_ecgs_roh.csv.gz", ["num_beats", "pacing_detected", "hr", "q_peak_mag", "t_peak_mag"])
    d.bh_check = lese_check_liste(erg / "ptbxl/check_ecg_list_alle.csv")
    d.lade_fehler = []
    d.bh_anno = {}
    d.bh_arm_beats = {}
    try:
        d.bh_ludb = Tab.aus_langformat(lese_lang(erg / "ludb/ausgabe_langformat.csv"))
        d.bh_ludb_roh = lese_bh_roh(erg / "ludb/processed_ecgs_roh.csv.gz", ["num_beats"])
        d.bh_anno[("ludb", "VV0")] = lese_anno_tar(erg / "ludb/annotations_ludb.tar.gz")
    except Exception as ex:  # noqa: BLE001
        d.lade_fehler.append("BRAVEHEART LUDB nativ: %s" % ex)
        log("LADEFEHLER BRAVEHEART LUDB nativ: %s" % ex)
    try:
        d.bh_anno[("ptbxl", "VV0")] = lese_anno_tar(erg / "ptbxl/annotations_ptbxl.tar.gz")
    except Exception as ex:  # noqa: BLE001
        d.lade_fehler.append("BRAVEHEART PTB-XL Schlagmarken: %s" % ex)
        log("LADEFEHLER BRAVEHEART PTB-XL Schlagmarken: %s" % ex)
    arme = {"g05": "VV0_G0.5", "g1": "VV0_G1", "g2": "VV0_G2", "vv1": "VV1", "vv2": "VV2"}
    tabs = []
    try:
        for ordner, ds in (("braveheart_sg", "ptbxl"), ("braveheart_ludb_arme", "ludb")):
            for a, vv in arme.items():
                pfad = erg / ordner / a
                t = Tab.aus_langformat(lese_lang(pfad / "ausgabe_langformat.csv"))
                tabs.append(t)
                roh = lese_bh_roh(pfad / "processed_ecgs_roh.csv.gz", ["num_beats"])
                d.bh_arm_beats[(ds, vv)] = roh["num_beats"]
                tars = sorted(pfad.glob("annotations_*.tar.gz"))
                d.bh_anno[(ds, vv)] = lese_anno_tar(tars[0]) if tars else {}
        d.bh_arme = Tab.aus_liste(tabs)
    except Exception as ex:  # noqa: BLE001
        d.lade_fehler.append("BRAVEHEART Arme: %s" % ex)
        log("LADEFEHLER BRAVEHEART Arme: %s" % ex)
    # NeuroKit2
    log("lade NeuroKit2")
    d.nk = None
    try:
        d.nk = pd.read_csv(erg / "neurokit2/langformat/neurokit2_schlaege.csv.gz",
                           dtype={"datensatz": str, "vorverarbeitung": str, "ableitung": str})
    except Exception as ex:  # noqa: BLE001
        d.lade_fehler.append("NeuroKit2: %s" % ex)
        log("LADEFEHLER NeuroKit2: %s" % ex)
    # Archiv
    log("lade Archivprodukte")
    d.archiv = {t: lade_archiv(t, d.ids_all) for t in ARCHIV_DATEI}
    # Quellen P0-Universum (alle PTB-XL-Aufnahmen)
    Q = {}
    ids = d.ids_all
    for g, cfg in GROESSEN.items():
        for t in cfg["tools"]:
            if t == "LoopLens":
                Q[(t, g)] = d.ll.hole("ptbxl", "VV0", ll_key(g), ids)
            elif t == "BRAVEHEART":
                Q[(t, g)] = d.bh.hole("ptbxl", "VV0", g, ids)
            else:
                Q[(t, g)] = d.archiv[t][g]
    # LoopLens: PR und P-Dauer fehlend, wenn in allen exportierten Schlaegen keine P-Welle vorhanden ist -> nicht_definiert (strukturell, V2 6.4 und 9)
    lb0 = d.ll_beats[(d.ll_beats.datensatz == "ptbxl") & (d.ll_beats.vorverarbeitung == "VV0")]
    if len(lb0):
        pvmax = pd.to_numeric(lb0.p_vorhanden, errors="coerce").groupby(lb0.aufnahme.to_numpy()).max()
        kein_p = np.isin(ids, pvmax[pvmax == 0].index.to_numpy())
        for g_ in ("pr_intervall", "p_dauer"):
            v_, s_ = Q[("LoopLens", g_)]
            Q[("LoopLens", g_)] = (v_, np.where((s_ == "fehlend") & kein_p, "nicht_definiert", s_).astype(object))
    d.Q = Q
    # Zusatzgroessen
    d.ll_rr = d.ll.hole("ptbxl", "VV0", "rr_intervall_ll", ids)
    d.ll_qmag = d.ll.hole("ptbxl", "VV0", "vektorbetrag_max_qrs_ll", ids)
    d.ll_tmag = d.ll.hole("ptbxl", "VV0", "vektorbetrag_max_t_ll", ids)
    d.bh_qmag = d.bh.hole("ptbxl", "VV0", "vektorbetrag_max_qrs", ids)
    tm = d.bh_roh["t_peak_mag"]
    tm = tm[~tm.index.duplicated(keep="first")].reindex(ids).to_numpy(float)
    d.bh_tmag = (np.where(np.isfinite(tm), tm, np.nan), np.where(np.isfinite(tm), "ok", "fehlend").astype(object))
    d.bh_check_mask = np.isin(ids, np.array(sorted(d.bh_check), dtype=np.int64))
    if not STATE["test"]:
        try:
            zaehl_kontrollen(d)
        except Exception as ex:  # noqa: BLE001
            kontrolle("Z0 Zaehlkontrollen ausfuehrbar", "ohne Ausnahme", "Ausnahme: %s" % ex, False)
    return d


def zaehl_kontrollen(d):
    """Vollstaendigkeit gegen die Zaehlungen der Laufberichte (AW-219 MANIFEST, AW-220 Abschnitt 4); nur Zaehlungen, keine Werte."""
    ll = d.ll.s["qrs_dauer_global"]
    soll_ll = {(r_[0], r_[1]): (r_[2], r_[3], r_[4]) for r_ in reference("bench_auswertung.soll_ll", [])}
    if not soll_ll:
        print("NOTE: no local reference value 'bench_auswertung.soll_ll'; check skipped.")
    n_aufg = int(len(ll))
    REF_Z1 = reference("bench_auswertung.z1_aufgaben")
    if REF_Z1 is None:
        print("NOTE: no local reference value 'bench_auswertung.z1_aufgaben'; check skipped.")
    else:
        kontrolle("Z1 LoopLens Aufgaben (Datensatz x Vorverarbeitung x Aufnahme)", REF_Z1, n_aufg, n_aufg == REF_Z1)
    for k, (st_, n_qc, n_ok) in soll_ll.items():
        # Status 'ok' der Aufnahme: kein Statuszeilen-Unterschied zwischen Groessen bei QC; qrs_dauer_global kann zusaetzlich fehlend sein
        nq = int((ll.loc[k] == "ausgeschlossen_qc").sum())
        kontrolle("Z2 LoopLens ausgeschlossen_qc %s %s" % k, n_qc, nq, nq == n_qc)
    nfe = int((d.bh.s["qrs_dauer_global"].loc["ptbxl"] == "werkzeugfehler").sum())
    REF_Z3 = reference("bench_auswertung.z3_werkzeugfehler_ptbxl")
    if REF_Z3 is None:
        print("NOTE: no local reference value 'bench_auswertung.z3_werkzeugfehler_ptbxl'; check skipped.")
    else:
        kontrolle("Z3 BRAVEHEART PTB-XL nativ: Werkzeugfehler-Aufnahmen", REF_Z3, nfe, nfe == REF_Z3, "AW-220 Abschnitt 6 (aufstellung.json)")
    soll_arme = {(r_[0], r_[1]): r_[2] for r_ in reference("bench_auswertung.soll_arme", [])}
    if not soll_arme:
        print("NOTE: no local reference value 'bench_auswertung.soll_arme'; check skipped.")
    for (ds, vv), soll in soll_arme.items():
        n = int((d.bh_arme.s["qrs_dauer_global"].loc[(ds, vv)] == "werkzeugfehler").sum())
        kontrolle("Z4 BRAVEHEART Arm %s %s: Werkzeugfehler" % (ds, vv), soll, n, n == soll, "AW-220 Abschnitt 4")
    nl = int((d.bh_ludb.s["qrs_dauer_global"].loc["ludb"] == "werkzeugfehler").sum())
    REF_Z5 = reference("bench_auswertung.z5_werkzeugfehler_ludb")
    if REF_Z5 is None:
        print("NOTE: no local reference value 'bench_auswertung.z5_werkzeugfehler_ludb'; check skipped.")
    else:
        kontrolle("Z5 BRAVEHEART LUDB nativ: Werkzeugfehler", REF_Z5, nl, nl == REF_Z5, "AW-208")


# ---------------------------------------------------------------------------------------------------------------
# Paare, Einheiten, Zusatzkennzahlen
# ---------------------------------------------------------------------------------------------------------------
def ok_maske(*paare):
    m = None
    for v, s in paare:
        x = (s == "ok") & np.isfinite(v)
        m = x if m is None else (m & x)
    return m


def baue_einheit(name, a, b, cluster, art, schwellen=(), nah=False, behalte=None, meta=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dd = a - b
    absd = np.abs(dd)
    ind = {}
    if art in ("zeit", "winkel"):
        for L in (2, 5, 10, 20, 40):
            ind["inn_%d" % L] = absd <= L
    else:
        mw = np.abs((a + b) / 2.0)
        for L in (2, 5, 10, 20, 40):
            ind["inn_%d" % L] = absd <= (L / 100.0) * mw
    cross, ratios = {}, {}
    for t in schwellen:
        pa, pb = a >= t, b >= t
        cross["t%d" % t] = (pa & pb, pa & ~pb, ~pa & pb, ~pa & ~pb)
    if nah and 120 in schwellen:
        mm = (a + b) / 2.0
        nahm = (mm >= 100) & (mm < 140)
        disc = (a >= 120) ^ (b >= 120)
        ratios["d120_nah"] = (disc & nahm, nahm)
    quant = {"d": (dd, (0.025, 0.5, 0.975)), "absd": (absd, (0.5,))}
    return Einheit(name, cluster, dd, ind=ind, ratios=ratios, cross=cross, quant=quant, behalte=behalte, meta=meta)


def icc_a1(a, b):
    x = np.column_stack([a, b])
    n, k = x.shape
    gm = x.mean()
    rm = x.mean(axis=1)
    cm = x.mean(axis=0)
    ssr = k * ((rm - gm) ** 2).sum()
    ssc = n * ((cm - gm) ** 2).sum()
    sst = ((x - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    nen = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / nen) if nen != 0 else float("nan")


def extras(a, b, art):
    """Punktkennzahlen ohne Intervall (Supplement V2 6.1)."""
    import warnings
    from scipy import stats as sst
    warnings.simplefilter("ignore")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    r = {}
    if n < 3:
        return r
    dd = a - b
    mw = (a + b) / 2.0
    r["sd"] = float(dd.std(ddof=1))
    r["rmse"] = float(np.sqrt((dd ** 2).mean()))
    r["loa_normal_lo"] = float(dd.mean() - 1.96 * dd.std(ddof=1))
    r["loa_normal_hi"] = float(dd.mean() + 1.96 * dd.std(ddof=1))
    va, vb = a.var(), b.var()
    r["ccc"] = float(2 * ((a - a.mean()) * (b - b.mean())).mean() / (va + vb + (a.mean() - b.mean()) ** 2)) if (va + vb) > 0 else float("nan")
    r["icc_abs"] = icc_a1(a, b)
    if mw.std() > 0:
        sl, ic = np.polyfit(mw, dd, 1)
        r["regr_diff_auf_mittel_steigung"] = float(sl)
        r["regr_diff_auf_mittel_achsenabschnitt"] = float(ic)
        r["pearson_r_diff_mittel"] = float(np.corrcoef(mw, dd)[0, 1])
        sp = sst.spearmanr(np.abs(dd), mw)
        r["spearman_absd_mittel"] = float(sp[0])
    if art == "betrag":
        pos = (a > 0) & (b > 0)
        if pos.sum() > 0:
            lr = np.log(a[pos] / b[pos])
            r["log_verhaeltnis_mittel"] = float(lr.mean())
            r["geometrisches_verhaeltnis"] = float(np.exp(lr.mean()))
    return r


def status_zaehlung(st):
    z = pd.Series(st).value_counts().to_dict()
    return {k: int(z.get(k, 0)) for k in ("ok", "werkzeugfehler", "fehlend", "ausgeschlossen_qc", "nicht_definiert", "nicht_zuordenbar",
                                            "ok_nicht_endlich", "nicht_im_lauf", "nicht_ladbar")}


def ausfallanteil(st):
    z = status_zaehlung(st)
    n = len(st) - z["nicht_definiert"]
    if n <= 0:
        return float("nan")
    return sum(z[k] for k in AUSFALL_STATUS) / n


# ---------------------------------------------------------------------------------------------------------------
# 6. Stufen
# ---------------------------------------------------------------------------------------------------------------
RES = {}


def schreibe_csv(df, name, kopie=True):
    p = PFADE["out"] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")
    if kopie and name not in STATE["dateien"]:
        STATE["dateien"].append(name)
    return p


def schreibe_text(text, name, kopie=True):
    p = PFADE["out"] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    if kopie and name not in STATE["dateien"]:
        STATE["dateien"].append(name)
    return p


def flach(name, pk, bs, schluessel=None):
    """Punktwert und 95-%-Perzentilintervall je Statistik als flache Zeile."""
    row = {}
    for k, v in pk.items():
        if schluessel is not None and k not in schluessel:
            continue
        row[k] = v
        if bs is not None and k in bs and k != "n_gewicht":
            lo, hi = ki95(bs[k])
            row[k + "_lo"] = lo
            row[k + "_hi"] = hi
    return row


def speicher_gb():
    """Arbeitsspeicher (aktuell, Spitze) des eigenen Prozesses in GB; nur zur Betriebsueberwachung."""
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        return mi.rss / 1e9, getattr(mi, "peak_wset", mi.rss) / 1e9
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def zwischenstand(text):
    try:
        p = PFADE["zwischen"]
        p.parent.mkdir(parents=True, exist_ok=True)
        kopf = "# AW-221 Zwischenstand\n\nStand: %s. Skript bench_auswertung.py Version %s.\n\n" % (time.strftime("%d.%m.%Y %H:%M:%S"), VERSION)
        p.write_text(kopf + text + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def stufe(name, fn, pflicht=False):
    t0 = time.time()
    log("=== Stufe %s: Start" % name)
    zwischenstand("Stufe %s laeuft seit %s.\n\nBisherige Stufen:\n%s" % (name, time.strftime("%H:%M:%S"),
                                                                         "\n".join("- %s: %s (%.0f s)" % (s["stufe"], s["status"], s["sekunden"]) for s in STATE["stufen"])))
    try:
        fn()
        STATE["stufen"].append(dict(stufe=name, status="ok", sekunden=time.time() - t0, fehler=""))
        log("=== Stufe %s: fertig (%.0f s; Speicher %.2f GB, Spitze %.2f GB)" % ((name, time.time() - t0) + speicher_gb()))
        return True
    except Exception as ex:  # noqa: BLE001
        tb = traceback.format_exc()
        log("FEHLER in Stufe %s: %s\n%s" % (name, ex, tb))
        STATE["stufen"].append(dict(stufe=name, status="FEHLER", sekunden=time.time() - t0, fehler=tb[-1500:]))
        if pflicht:
            raise
        return False


def pnenner(d, g, cand, comp, idx, extra=None):
    """Gemeinsamer Nenner eines Paares auf den Positionen idx; Rueckgabe: sel (Indizes in idx), a, b."""
    va, sa = d.Q[(cand, g)]
    vb, sb = d.Q[(comp, g)]
    m = ok_maske((va[idx], sa[idx]), (vb[idx], sb[idx]))
    if extra is not None:
        m = m & extra[idx]
    sel = np.flatnonzero(m)
    return sel, va[idx][sel], vb[idx][sel]


def winkel_norm_maske(d, minnorm=True):
    """Winkel-Mindestnorm (V2 5.3): Betragsmaxima von QRS- und T-Vektor in beiden Werkzeugen mindestens 0,05 mV."""
    if not minnorm:
        return np.ones(len(d.ids_all), dtype=bool)
    m = np.ones(len(d.ids_all), dtype=bool)
    for v, s in (d.ll_qmag, d.ll_tmag, d.bh_qmag, d.bh_tmag):
        m &= (s == "ok") & np.isfinite(v) & (np.where(np.isfinite(v), v, 0.0) >= WINKEL_MINNORM_MV)
    return m


BEHALTE_STRATA = {"bias", "mae", "absd_q0.5", "t120_disk", "t110_disk", "t130_disk", "t120_ppa", "t120_npa", "t120_kappa", "d120_nah",
                  "inn_5", "inn_10"}


def stufe_p0(d):
    """P1, sekundaere Paare auf P0, Strata, Sensitivitaeten: eine gemeinsame Bootstrap-Menge (gleiche Ziehungen)."""
    idx0 = d.idx_p0
    pers0 = d.person_all[idx0]
    units, info = [], {}
    out_pairs = []
    norm = winkel_norm_maske(d, True)
    # (a) alle Paare auf P0
    for g, cfg in GROESSEN.items():
        tools = cfg["tools"]
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                cand, comp = orientiere(tools[i], tools[j])
                sel, a, b = pnenner(d, g, cand, comp, idx0, extra=norm if cfg["art"] == "winkel" else None)
                name = "Z1|%s|%s|%s" % (g, cand, comp)
                if len(sel) < 3:
                    log("Paar %s: weniger als 3 gemeinsame Werte" % name)
                    continue
                e = baue_einheit(name, a, b, pers0[sel], cfg["art"], cfg["schwellen"],
                                 nah=(g == "qrs_dauer_global"), meta=dict(g=g, cand=cand, comp=comp))
                units.append(e)
                info[name] = dict(g=g, cand=cand, comp=comp, art=cfg["art"], n_p0=len(idx0), n=len(sel), a=a, b=b, sel=sel,
                                  st_cand=status_zaehlung(d.Q[(cand, g)][1][idx0]), st_comp=status_zaehlung(d.Q[(comp, g)][1][idx0]))
    P1N = "Z1|qrs_dauer_global|LoopLens|BRAVEHEART"
    p1 = info[P1N]
    RES["P1_name"] = P1N
    # (b) Schranken fuer fehlende Werte (P1)
    va, sa = d.Q[("LoopLens", "qrs_dauer_global")]
    vb, sb = d.Q[("BRAVEHEART", "qrs_dauer_global")]
    okc = ok_maske((va[idx0], sa[idx0]), (vb[idx0], sb[idx0]))
    a_all = np.where(okc, va[idx0], 0.0)
    b_all = np.where(okc, vb[idx0], 0.0)
    ind_sch = {}
    for t in (110, 120, 130):
        disc = okc & ((a_all >= t) ^ (b_all >= t))
        ind_sch["lb%d" % t] = disc
        ind_sch["ub%d" % t] = disc | ~okc
    units.append(Einheit("P1|schranken", pers0, np.zeros(len(idx0)), ind=ind_sch))
    # (c) Strata fuer P1
    meta = d.meta
    sel1 = p1["sel"]
    hr = 60000.0 / np.where(d.ll_rr[1][idx0] == "ok", d.ll_rr[0][idx0], np.nan)
    hr1 = hr[sel1]
    q1, q2 = np.nanquantile(hr1, [1 / 3, 2 / 3]) if np.isfinite(hr1).any() else (np.nan, np.nan)
    hf = np.where(~np.isfinite(hr), None, np.where(hr <= q1, "HF_niedrig", np.where(hr <= q2, "HF_mittel", "HF_hoch")))
    m0 = meta.iloc[idx0]
    schemata = {
        "QRS_PTBXL_Code": np.where(m0.breit.to_numpy(), "breit", "schmal"),
        "Schrittmacher": np.where(m0.pace.to_numpy(), "ja", "nein"),
        "Rhythmus": m0.rhythmus.to_numpy(),
        "Signalqualitaet": np.where(m0.marker_rausch.to_numpy(), "Rauschmarker_ja", "Rauschmarker_nein"),
        "Geraetegruppe": m0.geraet.to_numpy(),
        "Herzfrequenzterzil": hf,
        "Geschlecht": m0.geschlecht.to_numpy(),
        "Alter": m0.altersgruppe.to_numpy(),
    }
    RES["hf_grenzen"] = (float(q1) if np.isfinite(q1) else None, float(q2) if np.isfinite(q2) else None)
    strata_zeilen, mnar_zeilen = [], []
    in_common = np.zeros(len(idx0), dtype=bool)
    in_common[sel1] = True
    fehl_ll = ~((sa[idx0] == "ok") & np.isfinite(va[idx0]))
    fehl_bh = ~((sb[idx0] == "ok") & np.isfinite(vb[idx0]))
    for sch, lab in schemata.items():
        diffs = []
        for lev in sorted(set(x for x in lab if x is not None)):
            mask = np.array([x == lev for x in lab])
            n_lev = int(mask.sum())
            n_com = int((mask & in_common).sum())
            zeile = dict(schema=sch, stufe=lev, n_p0=n_lev, n_gemeinsam=n_com, ausfall_ll=float(fehl_ll[mask].mean()) if n_lev else np.nan,
                         ausfall_bh=float(fehl_bh[mask].mean()) if n_lev else np.nan,
                         ausfall_gemeinsam=float(1 - n_com / n_lev) if n_lev else np.nan)
            nur_zaehlung = (sch == "Alter" and lev == "unter_18")
            zeile["nur_zaehlung"] = bool(nur_zaehlung or n_com < MIN_NENNER_P0)
            mnar_zeilen.append(zeile)
            if n_lev >= MIN_NENNER_P0 and not nur_zaehlung:
                diffs.append(zeile["ausfall_gemeinsam"])
            if n_com >= MIN_NENNER_P0 and not nur_zaehlung:
                pos = np.flatnonzero(mask[sel1])
                e = baue_einheit("P1strat|%s|%s" % (sch, lev), p1["a"][pos], p1["b"][pos], pers0[sel1][pos], "zeit", (110, 120, 130),
                                 nah=True, behalte=BEHALTE_STRATA)
                units.append(e)
        RES.setdefault("mnar_schema", {})[sch] = (max(diffs) - min(diffs)) if len(diffs) >= 2 else float("nan")
    schreibe_csv(pd.DataFrame(mnar_zeilen), "P1_ausfall_je_stratum.csv")
    # (d) Sensitivitaeten fuer P1
    sens_defs = {}
    p0 = np.ones(len(idx0), dtype=bool)
    fold = meta.strat_fold.to_numpy()
    audit_mask_all = np.isin(d.ids_all, d.audit_p0)
    idx_all_ohne_pt = d.pos[np.setdiff1d(d.ids_all, d.pt)].to_numpy()
    # S1 alle Aufnahmen ausser PT, Personencluster
    sel_s1, a_s1, b_s1 = pnenner(d, "qrs_dauer_global", "LoopLens", "BRAVEHEART", idx_all_ohne_pt)
    sens_defs["S1_alle_Aufnahmen_ohne_PT"] = (a_s1, b_s1, d.person_all[idx_all_ohne_pt][sel_s1])

    def teilmenge(maskall):
        mm = maskall[idx0]
        pos = np.flatnonzero(mm[sel1])
        return (p1["a"][pos], p1["b"][pos], pers0[sel1][pos])

    sens_defs["S3_ohne_Rauschmarker"] = teilmenge(~meta.marker_rausch.to_numpy())
    # S4: werkzeugeigene Qualitaetsausschluesse (LoopLens-QC) nicht anwenden: Werte aus der Zusatztabelle
    vq, sq = d.llqc.hole("ptbxl", "VV0", "qrs_dauer_global", d.ids_all)
    va4 = np.where((va_status_all := d.Q[("LoopLens", "qrs_dauer_global")][1]) == "ausgeschlossen_qc", vq, d.Q[("LoopLens", "qrs_dauer_global")][0])
    sa4 = np.where((va_status_all == "ausgeschlossen_qc") & (sq == "ok"), "ok", va_status_all).astype(object)
    m4 = ok_maske((va4[idx0], sa4[idx0]), (vb[idx0], sb[idx0]))
    sel4 = np.flatnonzero(m4)
    sens_defs["S4_ohne_werkzeugeigene_Qualitaetsausschluesse"] = (va4[idx0][sel4], vb[idx0][sel4], pers0[sel4])
    # S5: ohne Vereinigung aller Listen und Marker
    bh_fehler = np.zeros(len(d.ids_all), dtype=bool)
    for g_ in ("qrs_dauer_global", "qt_global", "qrst_winkel_peak"):
        bh_fehler |= d.Q[("BRAVEHEART", g_)][1] == "werkzeugfehler"
    ll_qc = np.zeros(len(d.ids_all), dtype=bool)
    for g_ in ("qrs_dauer_global",):
        ll_qc |= d.Q[("LoopLens", g_)][1] == "ausgeschlossen_qc"
    union = d.bh_check_mask | bh_fehler | ll_qc | meta.marker_rausch.to_numpy() | meta.pace.to_numpy() | meta.marker_extra.to_numpy()
    RES["s5_union_anteil_p0"] = float(union[idx0].mean())
    sens_defs["S5_ohne_Vereinigung_Listen_Marker"] = teilmenge(~union)
    sens_defs["S13_nur_Fold10"] = teilmenge(fold == 10)
    sens_defs["S13_Fold10_ohne_Audit"] = teilmenge((fold == 10) & ~audit_mask_all)
    sens_defs["ohne_Audit_Personen_P0"] = teilmenge(~audit_mask_all)
    for nm, (a, b, cl) in sens_defs.items():
        if len(a) < 3:
            log("Sensitivitaet %s: weniger als 3 gemeinsame Personen, uebersprungen" % nm)
            continue
        units.append(baue_einheit("P1sens|" + nm, a, b, cl, "zeit", (110, 120, 130), nah=True, behalte=BEHALTE_STRATA, meta=dict(n=len(a))))
    # S7: Winkel ohne Mindestnorm
    sel7, a7, b7 = pnenner(d, "qrst_winkel_peak", "LoopLens", "BRAVEHEART", idx0, extra=None)
    if len(sel7) >= 3:
        units.append(baue_einheit("Z1sens|S7_Winkel_ohne_Mindestnorm|qrst_winkel_peak|LoopLens|BRAVEHEART", a7, b7, pers0[sel7], "winkel", (),
                                  meta=dict(n=len(sel7))))
    log("Bootstrap P0: %d Einheiten, K=%d, B=%d" % (len(units), d.K, B_ZIEHUNGEN))
    t0 = time.time()
    bs = boot_lauf(units, d.K)
    log("Bootstrap P0 fertig (%.0f s)" % (time.time() - t0))
    pk = {e.name: punktschaetzung(e) for e in units}
    RES["p0_pk"] = pk
    # --------------- Ausgaben: P1 zuerst ---------------
    RES["P1"] = p1_detail(d, info, pk, bs, P1N, okc, idx0)
    einz = PFADE["out"] / "einzel"
    einz.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ecg_id": d.p0[p1["sel"]], "looplens_ms": p1["a"], "braveheart_ms": p1["b"]}).to_csv(einz / "P1_paare.csv.gz", index=False)
    log("P1 geschrieben: D120 = %s, p = %s" % (RES["P1"]["D120"], RES["P1"]["h_p1_p_einseitig"]))
    ausfall_rows, paar_rows, ex_rows = [], [], []
    for name, inf in info.items():
        if name not in pk:
            continue
        row = dict(paar=name, rolle=("bestaetigend (P1)" if name == P1N else "sekundaer, deskriptiv, punktweise Intervalle ohne Mehrfachkorrektur"),
                   groesse=inf["g"], schluessel_kandidat=schluessel_von(inf["g"], inf["cand"]), schluessel_komparator=schluessel_von(inf["g"], inf["comp"]),
                   kandidat=inf["cand"], komparator=inf["comp"], art=inf["art"],
                   hinweis=("Differenz zweier verschieden definierter Groessen (Nachtrag 0)" if inf["g"] in UNDEFINIERT_PAAR["LoopLens/BRAVEHEART verschieden definiert"] and
                            {inf["cand"], inf["comp"]} == {"LoopLens", "BRAVEHEART"} else
                            ("ECGDeli PQ_Int_Global als PR gelesen (V2 5.3)" if (inf["g"] == "pr_intervall" and "ECGDeli" in (inf["cand"], inf["comp"])) else "")),
                   n_p0=inf["n_p0"], n_gemeinsam=inf["n"], anteil_gemeinsam=inf["n"] / inf["n_p0"])
        row.update(flach(name, pk[name], bs[name]))
        try:
            row.update(extras(inf["a"], inf["b"], inf["art"]))
            qq = np.quantile(inf["a"] - inf["b"], [0.05, 0.25, 0.75, 0.95])
            row.update(dict(d_q0_05=qq[0], d_q0_25=qq[1], d_q0_75=qq[2], d_q0_95=qq[3], iqr_d=qq[2] - qq[1]))
        except Exception as ex:  # noqa: BLE001
            log("Zusatzkennzahlen fuer %s nicht berechenbar: %s" % (name, ex))
        paar_rows.append(row)
    df_pairs = pd.DataFrame(paar_rows)
    schreibe_csv(df_pairs, "Z1_paare_P0_sekundaer.csv")
    done = set()
    for name, inf in info.items():
        for rolle, tool in (("kandidat", inf["cand"]), ("komparator", inf["comp"])):
            key = (inf["g"], tool)
            if key in done:
                continue
            done.add(key)
            st = inf["st_cand"] if rolle == "kandidat" else inf["st_comp"]
            af = ausfallanteil(d.Q[(tool, inf["g"])][1][idx0])
            ausfall_rows.append(dict(groesse=inf["g"], werkzeug=tool, n_p0=inf["n_p0"], ausfallanteil=af, band=band_text("ausfall", af), **st))
    schreibe_csv(pd.DataFrame(ausfall_rows), "Z1_ausfall_je_werkzeug_P0.csv")
    # S7 und Winkel-Ausschluss
    n_winkel_ok = int(len(sel7))
    n_winkel_norm = int(info["Z1|qrst_winkel_peak|LoopLens|BRAVEHEART"]["n"]) if "Z1|qrst_winkel_peak|LoopLens|BRAVEHEART" in info else 0
    RES["winkel_ausschluss"] = dict(n_ok_beide=n_winkel_ok, n_nach_minnorm=n_winkel_norm,
                                    anteil_ausgeschlossen=(1 - n_winkel_norm / n_winkel_ok) if n_winkel_ok else float("nan"))
    sw = "Z1sens|S7_Winkel_ohne_Mindestnorm|qrst_winkel_peak|LoopLens|BRAVEHEART"
    if sw in pk:
        r7 = dict(sensitivitaet="S7_Winkel_ohne_Mindestnorm", n=len(sel7))
        r7.update(flach(sw, pk[sw], bs[sw]))
        schreibe_csv(pd.DataFrame([r7]), "Z1_S7_winkel_ohne_mindestnorm.csv")
    schreibe_csv(pd.DataFrame([RES["winkel_ausschluss"]]), "Z1_winkel_mindestnorm_ausschluss.csv")
    # Strata
    srows = []
    for e in units:
        if e.name.startswith("P1strat|"):
            _, sch, lev = e.name.split("|", 2)
            row = dict(schema=sch, stufe=lev, n=e.n)
            row.update(flach(e.name, pk[e.name], bs[e.name]))
            dz = pk[e.name]["t120_disk"]
            row["d120_wilson_lo"], row["d120_wilson_hi"] = wilson(dz, e.n)
            row["d120_band"] = band_text("diskordanz", dz)
            row["median_absd_band"] = band_text("zeit", pk[e.name]["absd_q0.5"])
            srows.append(row)
    schreibe_csv(pd.DataFrame(srows), "P1_strata.csv")
    # Sensitivitaeten
    rows = []
    base = dict(sensitivitaet="Primaer P0", n=p1["n"])
    base.update(flach(P1N, pk[P1N], bs[P1N], BEHALTE_STRATA))
    rows.append(base)
    for e in units:
        if e.name.startswith("P1sens|"):
            row = dict(sensitivitaet=e.name.split("|", 1)[1], n=e.n)
            row.update(flach(e.name, pk[e.name], bs[e.name]))
            row["d120_band"] = band_text("diskordanz", pk[e.name]["t120_disk"])
            row["explorativ_anteil_bootstrap_unter_marge_kein_test"] = boot_p_einseitig_kleiner(bs[e.name]["t120_disk"], MARGE_H_P1)
            rows.append(row)
    schreibe_csv(pd.DataFrame(rows), "P1_sensitivitaeten.csv")
    # Schranken und MNAR
    sch = "P1|schranken"
    srow = dict(n_p0=len(idx0), n_gemeinsam=p1["n"], n_fehlend=len(idx0) - p1["n"])
    for t in (110, 120, 130):
        srow["D%d_gemeinsamer_nenner" % t] = pk[P1N]["t%d_disk" % t]
        srow["D%d_untere_schranke_alle_fehlenden_konkordant" % t] = pk[sch]["lb%d" % t]
        srow["D%d_obere_schranke_alle_fehlenden_diskordant" % t] = pk[sch]["ub%d" % t]
        for s_, k_ in (("lb", "untere"), ("ub", "obere")):
            lo, hi = ki95(bs[sch]["%s%d" % (s_, t)])
            srow["D%d_%s_schranke_ki_lo" % (t, k_)] = lo
            srow["D%d_%s_schranke_ki_hi" % (t, k_)] = hi
    srow["explorativ_anteil_bootstrap_untere_schranke_unter_marge_kein_test"] = boot_p_einseitig_kleiner(bs[sch]["lb120"], MARGE_H_P1)
    schreibe_csv(pd.DataFrame([srow]), "P1_schranken_fehlende.csv")
    RES["P1_schranken"] = srow
    # Bland-Altman-Haufigkeiten (nur Aggregat)
    ba_rows = []
    for nm, lim_m, lim_d, bm, bd in (("P1", (0, 400), (-200, 200), 10, 5),):
        a, b = p1["a"], p1["b"]
        H, xe, ye = np.histogram2d((a + b) / 2, a - b, bins=[np.arange(lim_m[0], lim_m[1] + bm, bm), np.arange(lim_d[0], lim_d[1] + bd, bd)])
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j] > 0:
                    ba_rows.append(dict(abbildung=nm, mittel_von=xe[i], mittel_bis=xe[i + 1], diff_von=ye[j], diff_bis=ye[j + 1], anzahl=int(H[i, j])))
    schreibe_csv(pd.DataFrame(ba_rows), "Z1_bland_altman_P1_haeufigkeiten.csv")
    RES["info_p0"] = {k: {kk: vv for kk, vv in v.items() if kk in ("g", "cand", "comp", "art", "n")} for k, v in info.items()}


def p1_detail(d, info, pk, bs, P1N, okc, idx0):
    p1 = info[P1N]
    k = pk[P1N]
    b_ = bs[P1N]
    D = k["t120_disk"]
    lo, hi = ki95(b_["t120_disk"])
    pw = boot_p_einseitig_kleiner(b_["t120_disk"], MARGE_H_P1)
    abg = holm([pw], ALPHA)[0]
    med = k["absd_q0.5"]
    mlo, mhi = ki95(b_["absd_q0.5"])
    dn = k["d120_nah"]
    dnlo, dnhi = ki95(b_["d120_nah"])
    a, b = p1["a"], p1["b"]
    mm = (a + b) / 2.0
    nah = (mm >= 100) & (mm < 140)
    r = dict(
        n_p0=p1["n_p0"], n_gemeinsam=p1["n"], status_looplens=p1["st_cand"], status_braveheart=p1["st_comp"],
        D120=D, D120_lo=lo, D120_hi=hi, D120_band=band_text("diskordanz", D), D120_band_mit_ki=band_mit_ki("diskordanz", D, lo, hi),
        h_p1_p_einseitig=pw, holm_m=1, holm_schwelle=ALPHA, h_p1_abgelehnt=bool(abg),
        D120_nah=dn, D120_nah_lo=dnlo, D120_nah_hi=dnhi, D120_nah_n=int(nah.sum()),
        D120_nah_band_mit_ki=band_mit_ki("diskordanz", dn, dnlo, dnhi),
        median_abs_diff=med, median_abs_lo=mlo, median_abs_hi=mhi, median_band_mit_ki=band_mit_ki("zeit", med, mlo, mhi),
        bias=k["bias"], mae=k["mae"], ppa=k["t120_ppa"], npa=k["t120_npa"], kappa=k["t120_kappa"],
        loa_lo=k["d_q0.025"], loa_hi=k["d_q0.975"], innerhalb_5=k["inn_5"], innerhalb_10=k["inn_10"],
    )
    for key in ("bias", "mae", "t120_ppa", "t120_npa", "t120_kappa", "d_q0.025", "d_q0.975", "inn_5", "inn_10", "t110_disk", "t130_disk"):
        l_, h_ = ki95(b_[key])
        r[key + "_lo"], r[key + "_hi"] = l_, h_
    # Kreuztabelle
    kt = []
    for t in (110, 120, 130):
        pa, pb = a >= t, b >= t
        kt.append(dict(schwelle=t, A_beide_ab_schwelle=int((pa & pb).sum()), B_nur_looplens=int((pa & ~pb).sum()), C_nur_braveheart=int((~pa & pb).sum()),
                       D_keines=int((~pa & ~pb).sum()), n=len(a), diskordanz=k["t%d_disk" % t], ppa=k["t%d_ppa" % t], npa=k["t%d_npa" % t],
                       kappa=k["t%d_kappa" % t], diskordanz_lo=ki95(b_["t%d_disk" % t])[0], diskordanz_hi=ki95(b_["t%d_disk" % t])[1],
                       band=band_text("diskordanz", k["t%d_disk" % t])))
    schreibe_csv(pd.DataFrame(kt), "P1_kreuztabelle.csv")
    # Nenner
    nen = []
    for tool, st in (("LoopLens", p1["st_cand"]), ("BRAVEHEART", p1["st_comp"])):
        row = dict(werkzeug=tool, n_p0=p1["n_p0"])
        row.update(st)
        af = (p1["n_p0"] - st["ok"] - st["nicht_definiert"]) / max(p1["n_p0"] - st["nicht_definiert"], 1)
        row["ausfallanteil"] = af
        row["ausfall_band"] = band_text("ausfall", af)
        nen.append(row)
    nen.append(dict(werkzeug="gemeinsamer Nenner", n_p0=p1["n_p0"], ok=p1["n"], ausfallanteil=1 - p1["n"] / p1["n_p0"],
                    ausfall_band=band_text("ausfall", 1 - p1["n"] / p1["n_p0"])))
    schreibe_csv(pd.DataFrame(nen), "P1_nenner.csv")
    # Kennzahlen (eine Zeile) und Supplement
    flatr = {kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in r.items()}
    schreibe_csv(pd.DataFrame([flatr]), "P1_kennzahlen.csv")
    sup = {}
    sup.update(extras(a, b, "zeit"))
    qq = np.quantile(a - b, [0.05, 0.25, 0.5, 0.75, 0.95])
    sup.update(dict(median_d=qq[2], q05=qq[0], q25=qq[1], q75=qq[3], q95=qq[4], iqr=qq[3] - qq[1], mittel_a=float(a.mean()), mittel_b=float(b.mean())))
    for L in (2, 5, 10, 20, 40):
        sup["innerhalb_%d" % L] = k["inn_%d" % L]
    schreibe_csv(pd.DataFrame([sup]), "P1_supplement.csv")
    return r


# ---------------------------------------------------------------------------------------------------------------
# NeuroKit2: Schlagcluster ueber Ableitungen, Aggregationsvarianten g1 bis g3 (V2 5.4)
# ---------------------------------------------------------------------------------------------------------------
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NK_SPALTEN = ["qrs_gipfel_schlag_sample", "p_beginn_schlag_sample", "p_gipfel_schlag_sample", "p_ende_schlag_sample",
              "q_gipfel_schlag_sample", "qrs_beginn_schlag_sample", "qrs_ende_schlag_sample", "s_gipfel_schlag_sample",
              "t_beginn_schlag_sample", "t_gipfel_schlag_sample", "t_ende_schlag_sample"]
# Indizes in einer Schlagzeile
I_RPK, I_PON, I_PPK, I_POFF, I_QPK, I_RON, I_ROFF, I_SPK, I_TON, I_TPK, I_TOFF = range(11)
FENSTER_SAMPLES = 40  # 80 ms bei 500 Hz


def cluster_peaks(items, window=FENSTER_SAMPLES, min_leads=3):
    """Cluster je Wellentyp: verankert am ersten (fruehesten) Gipfel, Abstand hoechstens window Samples vom Anker,
    jede Ableitung hoechstens einmal, Unterstuetzung durch mindestens min_leads Ableitungen (V2 5.5, Regel aus AW-140 Abschnitt 3).
    items: Liste (gipfel_sample, ableitungsindex, nutzlast)."""
    items = sorted(items, key=lambda x: (x[0], x[1]))
    n = len(items)
    used = [False] * n
    clusters = []
    for i in range(n):
        if used[i]:
            continue
        anchor = items[i][0]
        members = {}
        for j in range(i, n):
            if used[j]:
                continue
            if items[j][0] - anchor > window:
                break
            if items[j][1] not in members:
                members[items[j][1]] = j
        for j in members.values():
            used[j] = True
        if len(members) >= min_leads:
            clusters.append([items[j] for j in sorted(members.values())])
    return clusters


def cluster_anker(c):
    return float(np.median([it[0] for it in c]))


def _fin(*x):
    return all(np.isfinite(v) for v in x)


def nk_record(leadbeats):
    """leadbeats: dict Ableitungsindex -> Array (nb, 11) in Samples (NaN = fehlt). Rueckgabe: Cluster fuer QRS, P und T."""
    qrs_items, p_items, t_items = [], [], []
    for L, B in leadbeats.items():
        for row in B:
            ron, rpk, roff = row[I_RON], row[I_RPK], row[I_ROFF]
            if _fin(ron, rpk, roff) and ron < rpk < roff and 40 <= (roff - ron) * MS_PRO_SAMPLE <= 300:
                qrs_items.append((float(rpk), L, row))
            pon, ppk, poff = row[I_PON], row[I_PPK], row[I_POFF]
            if _fin(pon, ppk, poff) and pon < ppk < poff:
                p_items.append((float(ppk), L, row))
            ton, tpk, toff = row[I_TON], row[I_TPK], row[I_TOFF]
            if _fin(tpk, toff) and tpk < toff and (not np.isfinite(ton) or ton < tpk):
                t_items.append((float(tpk), L, row))
    return dict(qrs=cluster_peaks(qrs_items), p=cluster_peaks(p_items), t=cluster_peaks(t_items))


def _cluster_werte(c, art):
    """(g1, g2, {Ableitung: Dauer}) eines Clusters oder None bei weniger als drei gueltigen Ableitungen. Dauern in ms."""
    a_l, b_l, leads = [], [], []
    for pk, L, row in c:
        ron, roff, pon, ppk, poff, toff = row[I_RON], row[I_ROFF], row[I_PON], row[I_PPK], row[I_POFF], row[I_TOFF]
        if art == "qrs":
            a_, b_ = ron, roff
        elif art == "qt":
            if not (np.isfinite(toff) and toff > ron and 200 <= (toff - ron) * MS_PRO_SAMPLE <= 700):
                continue
            a_, b_ = ron, toff
        else:  # pr: Start = P-Beginn, Ende = QRS-Beginn
            if not (_fin(pon, ppk, poff) and pon < ppk < poff and ron > pon):
                continue
            a_, b_ = pon, ron
        a_l.append(a_)
        b_l.append(b_)
        leads.append(L)
    if len(a_l) < 3:
        return None
    a_s = np.sort(np.array(a_l))
    b_s = np.sort(np.array(b_l))
    if art == "pr":
        g1 = (b_s[0] - a_s[0]) * MS_PRO_SAMPLE
        g2 = (b_s[1] - a_s[1]) * MS_PRO_SAMPLE
    else:
        g1 = (b_s[-1] - a_s[0]) * MS_PRO_SAMPLE
        g2 = (b_s[-2] - a_s[1]) * MS_PRO_SAMPLE
    dur = {L: (b_ - a_) * MS_PRO_SAMPLE for L, a_, b_ in zip(leads, a_l, b_l)}
    return g1, g2, dur


def nk_agg(qrs_clusters, art, subset=None):
    cl = qrs_clusters if subset is None else [qrs_clusters[i] for i in subset]
    g1s, g2s, lead = [], [], {}
    for c in cl:
        r = _cluster_werte(c, art)
        if r is None:
            continue
        g1s.append(r[0])
        g2s.append(r[1])
        for L, v in r[2].items():
            lead.setdefault(L, []).append(v)
    if not g1s:
        return (float("nan"), float("nan"), float("nan"))
    g3 = max(np.median(v) for v in lead.values()) if len(lead) >= 3 else float("nan")
    return (float(np.median(g1s)), float(np.median(g2s)), float(g3))


def nk_parse(nk):
    """Breite NeuroKit2-Schlagtabelle -> dict (datensatz, vorverarbeitung, aufnahme) -> Record mit Clustern und g-Werten."""
    df = nk.copy()
    df["ableitung"] = df["ableitung"].astype(str).str.upper()
    df["li"] = df["ableitung"].map({l: i for i, l in enumerate(LEADS)})
    df = df.dropna(subset=["li"])
    df["aufnahme"] = df["aufnahme"].astype(int)
    df = df.sort_values(["datensatz", "vorverarbeitung", "aufnahme", "li", "schlag_nr"], kind="stable")
    arr = df[NK_SPALTEN].to_numpy(float)
    ds = df.datensatz.to_numpy()
    vv = df.vorverarbeitung.to_numpy()
    aid = df.aufnahme.to_numpy(int)
    li = df.li.to_numpy(int)
    n = len(df)
    out = {}
    if n == 0:
        return out
    chg = np.ones(n, bool)
    chg[1:] = (ds[1:] != ds[:-1]) | (vv[1:] != vv[:-1]) | (aid[1:] != aid[:-1])
    starts = np.flatnonzero(chg)
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        l = li[s:e]
        c2 = np.flatnonzero(np.r_[True, l[1:] != l[:-1]])
        c2e = np.r_[c2[1:], len(l)]
        lb = {int(l[a_]): arr[s + a_:s + b_] for a_, b_ in zip(c2, c2e)}
        rec = nk_record(lb)
        v = {}
        for art, nm in (("qrs", "QRS"), ("qt", "QT"), ("pr", "PR")):
            g1, g2, g3 = nk_agg(rec["qrs"], art)
            v[nm + "_g1"], v[nm + "_g2"], v[nm + "_g3"] = g1, g2, g3
        v["n_beats"] = float(len(rec["qrs"]))
        v["n_leads_mit_schlaegen"] = float(len(lb))
        rec["werte"] = v
        out[(str(ds[s]), str(vv[s]), int(aid[s]))] = rec
    return out


# ---------------------------------------------------------------------------------------------------------------
# Verstaerkungsarm (V2 7) und Filterarm (V2 8)
# ---------------------------------------------------------------------------------------------------------------
QUANT = {
    "LoopLens": {"zeit": ["qrs_dauer_global", "qt_global", "qtcf", "pr_intervall", "p_dauer"],
                 "winkel": ["qrst_winkel_peak", "qrst_winkel_integral"],
                 "betrag": ["vektorbetrag_max_qrs", "vektorbetrag_max_t", "integralbetrag_qrs", "sai_qrst", "schleifenlaenge_qrs", "svg_betrag"]},
    "BRAVEHEART": {"zeit": ["qrs_dauer_global", "qt_global", "qtcf"],
                   "winkel": ["qrst_winkel_peak", "qrst_winkel_integral"],
                   "betrag": ["vektorbetrag_max_qrs", "integralbetrag_qrs", "sai_qrst", "schleifenlaenge_qrs"]},
    "NeuroKit2": {"zeit": ["QRS_g1", "QRS_g2", "QRS_g3", "QT_g1", "QT_g2", "QT_g3", "PR_g1", "PR_g2", "PR_g3"], "winkel": [], "betrag": []},
}
SCHWELLEN_ARM = {"qrs_dauer_global": (120,), "QRS_g1": (120,), "qtcf": (480, 500)}
FAKTOR_VV = {0.5: "VV0_G0.5", 1.0: "VV0_G1", 2.0: "VV0_G2"}


def werte_tool(d, tool, ds, vv, ids):
    """Alle Groessen eines Werkzeugs in einer Vorverarbeitung fuer die Aufnahmen ids: dict Groesse -> (Werte, Status)."""
    out = {}
    arten = QUANT[tool]
    for art, qs in arten.items():
        for q in qs:
            if tool == "LoopLens":
                out[q] = d.ll.hole(ds, vv, ll_key(q), ids)
            elif tool == "BRAVEHEART":
                if vv == "VV0":
                    out[q] = (d.bh if ds == "ptbxl" else d.bh_ludb).hole(ds, "VV0", q, ids)
                else:
                    out[q] = d.bh_arme.hole(ds, vv, q, ids)
            else:
                v = np.full(len(ids), np.nan)
                st = np.full(len(ids), "fehlend", dtype=object)
                for i, aid in enumerate(ids):
                    rec = d.nk_rec.get((ds, vv, int(aid)))
                    if rec is None:
                        continue
                    x = rec["werte"].get(q, np.nan)
                    if np.isfinite(x):
                        v[i] = x
                        st[i] = "ok"
                out[q] = (v, st)
    return out


def art_von(tool, q):
    for art, qs in QUANT[tool].items():
        if q in qs:
            return art
    raise KeyError(q)


def veraendert(art, x1, xf, f):
    """Veraenderungskriterium V2 7 Zielgroesse (1)."""
    if art == "zeit":
        return np.abs(xf - x1) > 1e-9
    if art == "winkel":
        return np.abs(xf - x1) > 0.001
    ref = f * x1
    return np.abs(xf - ref) > 1e-6 * np.abs(ref) + 1e-15 * (ref == 0)


def aenderung(art, x1, xf, f):
    if art == "betrag":
        ref = f * x1
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ref != 0, (xf - ref) / ref, xf - ref)
    return xf - x1


def verteilung(a):
    a = np.abs(np.asarray(a, float))
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(rms=np.nan, median_abs=np.nan, p95_abs=np.nan, p99_abs=np.nan, max_abs=np.nan)
    return dict(rms=float(np.sqrt((a ** 2).mean())), median_abs=float(np.median(a)), p95_abs=float(np.percentile(a, 95)),
                p99_abs=float(np.percentile(a, 99)), max_abs=float(a.max()))


def zelle_verstaerkung(tool, q, art, f, x1, s1, xf, sf, w, ds):
    ok = (s1 == "ok") & (sf == "ok") & np.isfinite(x1) & np.isfinite(xf)
    n_ids = len(x1)
    row = dict(werkzeug=tool, datensatz=ds, groesse=q, art=art, faktor=f, n_aufnahmen=n_ids, n_nenner=int(ok.sum()))
    if ok.sum() == 0:
        return row
    ch = veraendert(art, x1[ok], xf[ok], f)
    wg = gew_anteil(ch, w[ok])
    row.update(anteil_gewichtet=wg["p"], wilson_lo=wg["lo"], wilson_hi=wg["hi"], n_eff=wg["n_eff"],
               band_punkt=band_text("verstaerkung", wg["p"]), band_mit_ki=band_mit_ki("verstaerkung", wg["p"], wg["lo"], wg["hi"]))
    ug = gew_anteil(ch, np.ones(int(ok.sum())))
    row.update(anteil_ungewichtet=ug["p"], ung_wilson_lo=ug["lo"], ung_wilson_hi=ug["hi"], n_veraendert=int(ch.sum()))
    row.update({"aend_ungewichtet_" + k: v for k, v in verteilung(aenderung(art, x1[ok], xf[ok], f)).items()})
    row["aenderung_einheit"] = "relativ zu Faktor mal Wert" if art == "betrag" else ("Grad" if art == "winkel" else "ms")
    return row


def stufe_verstaerkung(d, ds="ptbxl"):
    """V2 7: Zellen je Werkzeug x Groesse x Faktor, Statuswechsel, Schlagzahlwechsel, Schwellenwechsel, Pruefsaetze, Kontrollpruefsatz, K1."""
    if ds == "ptbxl":
        ids = d.sg
        w = d.sg_gew
    else:
        ids = np.array(sorted(set(d.bh_ludb.v.loc[("ludb", "VV0")].index.astype(int))), dtype=np.int64)
        w = np.ones(len(ids))
    zellen, status_rows, schwellen_rows, k1_rows, kon_rows, strat_rows = [], [], [], [], [], []
    wt = {}
    for tool in ("LoopLens", "BRAVEHEART", "NeuroKit2"):
        for vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2"):
            wt[(tool, vv)] = werte_tool(d, tool, ds, vv, ids)
    # Beats je Aufnahme und Vorverarbeitung
    llb = d.ll_beats[d.ll_beats.datensatz == ds].groupby(["vorverarbeitung", "aufnahme"]).size()

    def schlagzahl(tool, vv):
        if tool == "LoopLens":
            return np.array([llb.get((vv, int(i)), np.nan) if (vv, int(i)) in llb.index else np.nan for i in ids], float)
        if tool == "BRAVEHEART":
            sr = d.bh_arm_beats.get((ds, vv)) if vv != "VV0" else (d.bh_roh["num_beats"] if ds == "ptbxl" else d.bh_ludb_roh["num_beats"])
            if sr is None:
                return np.full(len(ids), np.nan)
            sr = sr[~sr.index.duplicated(keep="first")]
            return sr.reindex(ids).to_numpy(float)
        return np.array([d.nk_rec[(ds, vv, int(i))]["werte"]["n_beats"] if (ds, vv, int(i)) in d.nk_rec else np.nan for i in ids], float)

    for tool in ("LoopLens", "BRAVEHEART", "NeuroKit2"):
        w1 = wt[(tool, "VV0_G1")]
        # K1 (Faktor 1 gegen Lauf ohne Skalierung)
        w0 = wt[(tool, "VV0")]
        for art, qs in QUANT[tool].items():
            for q in qs:
                x0, s0 = w0[q]
                x1, s1 = w1[q]
                both = (s0 == "ok") & (s1 == "ok")
                if art == "zeit" or tool in ("BRAVEHEART", "NeuroKit2"):
                    diff = np.abs(x0[both] - x1[both]) > 0
                    schwelle = "gleicher Wert (bitgleich)"
                else:
                    diff = np.abs(x0[both] - x1[both]) > 1e-9
                    schwelle = "Abweichung hoechstens 1e-9"
                k1_rows.append(dict(werkzeug=tool, datensatz=ds, groesse=q, kriterium=schwelle, n_verglichen=int(both.sum()),
                                    n_abweichend=int(diff.sum()), status_gleich=bool(np.array_equal(s0 == "ok", s1 == "ok")),
                                    bestanden=bool(diff.sum() == 0 and np.array_equal(s0 == "ok", s1 == "ok"))))
        for f in (0.5, 2.0):
            wf = wt[(tool, FAKTOR_VV[f])]
            for art, qs in QUANT[tool].items():
                for q in qs:
                    x1, s1 = w1[q]
                    xf, sf = wf[q]
                    zellen.append(zelle_verstaerkung(tool, q, art, f, x1, s1, xf, sf, w, ds))
                    if ds == "ptbxl":
                        okq = (s1 == "ok") & (sf == "ok") & np.isfinite(x1) & np.isfinite(xf)
                        for stv in sorted(set(d.sg_stratum)):
                            mk = okq & (d.sg_stratum == stv)
                            if mk.sum() < MIN_NENNER_SG:
                                strat_rows.append(dict(werkzeug=tool, groesse=q, art=art, faktor=f, stratum=stv, n_nenner=int(mk.sum()), nur_zaehlung=True))
                                continue
                            chs = veraendert(art, x1[mk], xf[mk], f)
                            gs = gew_anteil(chs, np.ones(int(mk.sum())))
                            strat_rows.append(dict(werkzeug=tool, groesse=q, art=art, faktor=f, stratum=stv, n_nenner=int(mk.sum()), nur_zaehlung=False,
                                                   anteil_ungewichtet=gs["p"], wilson_lo=gs["lo"], wilson_hi=gs["hi"], n_veraendert=int(chs.sum()),
                                                   band_punkt=band_text("verstaerkung", gs["p"]), band_mit_ki=band_mit_ki("verstaerkung", gs["p"], gs["lo"], gs["hi"])))
                    # Statuswechsel
                    okw = gew_anteil((s1 == "ok") != (sf == "ok"), w)
                    status_rows.append(dict(werkzeug=tool, datensatz=ds, groesse=q, faktor=f, art="statuswechsel_ok", anteil=okw["p"],
                                            wilson_lo=okw["lo"], wilson_hi=okw["hi"], n=len(ids)))
                    # Schwellenwechsel
                    if q in SCHWELLEN_ARM:
                        both = (s1 == "ok") & (sf == "ok")
                        for t in SCHWELLEN_ARM[q]:
                            wsch = gew_anteil((x1[both] >= t) != (xf[both] >= t), w[both])
                            schwellen_rows.append(dict(werkzeug=tool, datensatz=ds, groesse=q, faktor=f, schwelle=t, n_nenner=int(both.sum()),
                                                       anteil_wechsel=wsch["p"], wilson_lo=wsch["lo"], wilson_hi=wsch["hi"],
                                                       n_wechsel=int(((x1[both] >= t) != (xf[both] >= t)).sum())))
            n1, nf = schlagzahl(tool, "VV0_G1"), schlagzahl(tool, FAKTOR_VV[f])
            bz = np.isfinite(n1) & np.isfinite(nf)
            if bz.sum():
                sw = gew_anteil(n1[bz] != nf[bz], w[bz])
                status_rows.append(dict(werkzeug=tool, datensatz=ds, groesse="schlagzahl", faktor=f, art="schlagzahlwechsel", anteil=sw["p"],
                                        wilson_lo=sw["lo"], wilson_hi=sw["hi"], n=int(bz.sum())))
        # Kontrollpruefsatz Betraege
        if tool in ("LoopLens", "BRAVEHEART"):
            for q in QUANT[tool]["betrag"]:
                for f in (0.5, 2.0):
                    x1, s1 = w1[q]
                    xf, sf = wt[(tool, FAKTOR_VV[f])][q]
                    ok = (s1 == "ok") & (sf == "ok")
                    ref = f * x1[ok]
                    relabw = np.abs(xf[ok] - ref) <= 1e-6 * np.abs(ref)
                    anteil = float(relabw.mean()) if ok.sum() else float("nan")
                    kon_rows.append(dict(werkzeug=tool, datensatz=ds, groesse=q, faktor=f, n=int(ok.sum()), anteil_innerhalb_1e6=anteil,
                                         bestanden=bool(anteil >= 0.99)))
    k1ok = {}
    for tool in ("LoopLens", "BRAVEHEART", "NeuroKit2"):
        rows_t = [r for r in k1_rows if r["werkzeug"] == tool]
        k1ok[tool] = bool(rows_t) and all(r["bestanden"] for r in rows_t)
    RES["K1_bestanden" + ("" if ds == "ptbxl" else "_LUDB")] = k1ok
    zel = pd.DataFrame(zellen)
    zel["k1_bestanden"] = zel.werkzeug.map(k1ok)
    zel["rolle"] = "Robustheitsarm (V1 bis V3 replizierend, V4 offen beschreibend); LUDB ungewichtet, sekundaer"
    sfx = "" if ds == "ptbxl" else "_LUDB"
    schreibe_csv(zel, "V_verstaerkung_zellen%s.csv" % sfx)
    schreibe_csv(pd.DataFrame(status_rows), "V_status_schlagzahl%s.csv" % sfx)
    schreibe_csv(pd.DataFrame(schwellen_rows), "V_schwellenwechsel%s.csv" % sfx)
    schreibe_csv(pd.DataFrame(k1_rows), "V_K1_faktor1_gegen_ohne_skalierung%s.csv" % sfx)
    schreibe_csv(pd.DataFrame(kon_rows), "V_kontrollpruefsatz_betraege%s.csv" % sfx)
    if strat_rows:
        schreibe_csv(pd.DataFrame(strat_rows), "V_verstaerkung_je_S_G_stratum.csv")
    RES["V_zellen" + sfx] = zel
    if ds != "ptbxl":
        return
    # Pruefsaetze V1 bis V4 (ein Werkzeug mit nicht bestandener K1 ist fuer den Arm ungueltig, V2 7 und 11.1)
    ps = []

    def zeile(tool, q, f):
        z = zel[(zel.werkzeug == tool) & (zel.groesse == q) & (zel.faktor == f)]
        return None if z.empty or "anteil_gewichtet" not in z or not np.isfinite(z.iloc[0].get("anteil_gewichtet", np.nan)) else z.iloc[0]

    def urteil_v1(tool, qs, bez):
        if not k1ok.get(tool, False):
            return "ungueltig (K1 nicht bestanden)", ""
        zellenp = [(q, f, zeile(tool, q, f)) for q in qs for f in (0.5, 2.0)]
        zellenp = [(q, f, z) for q, f, z in zellenp if z is not None]
        if not zellenp:
            return "nicht auswertbar", ""
        best = [(q, f) for q, f, z in zellenp if z.anteil_gewichtet >= 0.05 and z.wilson_lo > 0.01]
        wid = all(z.wilson_hi < 0.05 for _, _, z in zellenp)
        detail = "; ".join("%s f=%g: %s %% (Wilson %s bis %s)" % (q, f, proz(z.anteil_gewichtet), proz(z.wilson_lo), proz(z.wilson_hi)) for q, f, z in zellenp)
        if best:
            return "bestaetigt", detail
        if wid:
            return "widerlegt", detail
        return "unentschieden", detail

    u, det = urteil_v1("LoopLens", ["pr_intervall", "qt_global"], "V1")
    ps.append(dict(pruefsatz="V1", werkzeug="LoopLens", urteil=u, regel="bestaetigt: mindestens eine Zelle (PR oder QT, Faktor 0,5 oder 2) mit gewichtetem Anteil >= 5 % und untere Wilson-Grenze > 1 %; widerlegt: obere Grenze aller vier Zellen < 5 %; sonst unentschieden (Auslegung des Wortes oder als Existenz ueber die vier Zellen)", zellen=det))
    # V2: QRS-Beginn relativ zum Gipfel (LoopLens Schlagexport); Schlaege zwischen Faktor 1 und Faktor f ueber identische R-Gipfel gepaart
    lb = d.ll_beats[d.ll_beats.datensatz == ds].copy()
    lb["rel"] = (pd.to_numeric(lb.qrs_beginn, errors="coerce") - pd.to_numeric(lb.r_gipfel, errors="coerce")) * MS_PRO_SAMPLE
    wser = pd.Series(w, index=ids)
    zv2 = []
    base_b = lb[lb.vorverarbeitung == "VV0_G1"][["aufnahme", "r_gipfel", "rel"]]
    for f in (0.5, 2.0):
        fb = lb[lb.vorverarbeitung == FAKTOR_VV[f]][["aufnahme", "r_gipfel", "rel"]]
        m = base_b.merge(fb, on=["aufnahme", "r_gipfel"], suffixes=("_1", "_f"))
        m = m[np.isfinite(m.rel_1) & np.isfinite(m.rel_f)]
        if len(m) == 0:
            continue
        chg = (np.abs(m.rel_1 - m.rel_f) > 1e-9).groupby(m.aufnahme).any()
        chg = chg[chg.index.isin(ids)]
        if len(chg) == 0:
            continue
        g = gew_anteil(chg.to_numpy(float), wser.reindex(chg.index).to_numpy(float))
        zv2.append((f, g))
    if zv2 and k1ok.get("LoopLens", False):
        if all(g["hi"] <= 0.02 for _, g in zv2):
            u2 = "bestaetigt"
        elif any(g["lo"] > 0.01 for _, g in zv2):
            u2 = "widerlegt"
        else:
            u2 = "unentschieden"
    elif zv2:
        u2 = "ungueltig (K1 nicht bestanden)"
    else:
        u2 = "nicht auswertbar"
    if zv2:
        ps.append(dict(pruefsatz="V2", werkzeug="LoopLens", urteil=u2, regel="bestaetigt: obere 95-%-Grenze des Anteils veraenderter Aufnahmen hoechstens 2 % (beide Faktoren); widerlegt: untere Grenze ueber 1 % (ein Faktor); sonst unentschieden. Aufnahme veraendert, wenn bei mindestens einem ueber den R-Gipfel gepaarten Schlag (QRS-Beginn minus R-Gipfel) der ms-Wert abweicht",
                       zellen="; ".join("f=%g: %s %% (Wilson %s bis %s), n=%d" % (f, proz(g["p"]), proz(g["lo"]), proz(g["hi"]), g["n"]) for f, g in zv2)))
        for f, g in zv2:
            zellen.append(dict(werkzeug="LoopLens", datensatz=ds, groesse="qrs_beginn_relativ_zum_gipfel", art="zeit", faktor=f, n_nenner=g["n"],
                               anteil_gewichtet=g["p"], wilson_lo=g["lo"], wilson_hi=g["hi"], n_eff=g["n_eff"],
                               band_punkt=band_text("verstaerkung", g["p"]), band_mit_ki=band_mit_ki("verstaerkung", g["p"], g["lo"], g["hi"])))
        zel = pd.DataFrame(zellen)
        zel["k1_bestanden"] = zel.werkzeug.map(k1ok)
        zel["rolle"] = "Robustheitsarm (V1 bis V3 replizierend, V4 offen beschreibend)"
        schreibe_csv(zel, "V_verstaerkung_zellen.csv")
        RES["V_zellen"] = zel
    else:
        ps.append(dict(pruefsatz="V2", werkzeug="LoopLens", urteil="nicht auswertbar", regel="", zellen="kein Schlagexport fuer S_G"))
    u, det = urteil_v1("NeuroKit2", ["PR_g1", "QT_g1"], "V3")
    ps.append(dict(pruefsatz="V3", werkzeug="NeuroKit2", urteil=u, regel="wie V1 fuer PR_g1 und QT_g1 aus den NeuroKit2-Schlagmarken", zellen=det))
    zb = zel[zel.werkzeug == "BRAVEHEART"]
    ps.append(dict(pruefsatz="V4", werkzeug="BRAVEHEART",
                   urteil=("offene Messung ohne gerichtete Erwartung (Zellen siehe V_verstaerkung_zellen.csv)" if k1ok.get("BRAVEHEART", False) else "ungueltig (K1 nicht bestanden)"),
                   regel="Baender nach 6.4; Nullbefund mit oberer Wilson-Grenze, nicht 'robust'",
                   zellen="; ".join("%s f=%g: %s %% (Wilson %s bis %s; %s)" % (r.groesse, r.faktor, proz(r.anteil_gewichtet), proz(r.wilson_lo), proz(r.wilson_hi), r.band_mit_ki)
                                    for r in zb.itertuples() if hasattr(r, "anteil_gewichtet") and np.isfinite(r.anteil_gewichtet))))
    schreibe_csv(pd.DataFrame(ps), "V_pruefsaetze.csv")
    RES["V_pruefsaetze"] = ps


def stufe_filterarm(d, ds="ptbxl"):
    """V2 8: gepaarte Differenzen VV1 minus VV0 und VV2 minus VV0 je Werkzeug (BRAVEHEART, NeuroKit2); keine Vergleiche zwischen Werkzeugen.
    LoopLens: Filterschalter nicht gebunden, Arm nicht gerechnet."""
    if ds == "ptbxl":
        ids = d.sg
        w = d.sg_gew
    else:
        ids = np.array(sorted(set(d.bh_ludb.v.loc[("ludb", "VV0")].index.astype(int))), dtype=np.int64)
        w = np.ones(len(ids))
    rows, sw_rows = [], []
    for tool in ("BRAVEHEART", "NeuroKit2"):
        w0 = werte_tool(d, tool, ds, "VV0", ids)
        for vv in ("VV1", "VV2"):
            wx = werte_tool(d, tool, ds, vv, ids)
            for art, qs in QUANT[tool].items():
                for q in qs:
                    x0, s0 = w0[q]
                    xx, sx = wx[q]
                    ok = (s0 == "ok") & (sx == "ok")
                    row = dict(werkzeug=tool, datensatz=ds, vergleich="%s minus VV0" % vv, groesse=q, art=art, n_aufnahmen=len(ids), n_gemeinsam=int(ok.sum()))
                    wo = gew_anteil((s0 == "ok") & (sx != "ok"), w)
                    wn = gew_anteil((s0 != "ok") & (sx == "ok"), w)
                    row.update(anteil_ok_zu_nicht_ok=wo["p"], anteil_nicht_ok_zu_ok=wn["p"])
                    if ok.sum() >= 3:
                        dd = xx[ok] - x0[ok]
                        ww = w[ok]
                        row.update(bias_gew=float((ww * dd).sum() / ww.sum()), mae_gew=float((ww * np.abs(dd)).sum() / ww.sum()),
                                   median_d_ungewichtet=float(np.median(dd)), median_abs_ungewichtet=float(np.median(np.abs(dd))))
                        if art == "betrag":
                            mw = np.abs((xx[ok] + x0[ok]) / 2.0)
                            row["innerhalb_5"] = gew_anteil(np.abs(dd) <= 0.05 * mw, ww)["p"]
                            row["innerhalb_10"] = gew_anteil(np.abs(dd) <= 0.10 * mw, ww)["p"]
                        else:
                            row["innerhalb_5"] = gew_anteil(np.abs(dd) <= 5, ww)["p"]
                            row["innerhalb_10"] = gew_anteil(np.abs(dd) <= 10, ww)["p"]
                        ch = gew_anteil(np.abs(dd) > 1e-9, ww)
                        row.update(anteil_veraendert=ch["p"], veraendert_wilson_lo=ch["lo"], veraendert_wilson_hi=ch["hi"], n_eff=ch["n_eff"])
                        if q in SCHWELLEN_ARM:
                            for t in SCHWELLEN_ARM[q]:
                                sw = gew_anteil((x0[ok] >= t) != (xx[ok] >= t), ww)
                                sw_rows.append(dict(werkzeug=tool, datensatz=ds, vergleich="%s minus VV0" % vv, groesse=q, schwelle=t, n=int(ok.sum()),
                                                    anteil_wechsel=sw["p"], wilson_lo=sw["lo"], wilson_hi=sw["hi"]))
                    rows.append(row)
    sfx = "" if ds == "ptbxl" else "_LUDB"
    for r_ in rows:
        r_["rolle"] = "explorativ-beschreibend (Filterkonfiguration, keine Pruefsaetze, kein Vergleich der Aenderungsgroessen zwischen Werkzeugen)"
    schreibe_csv(pd.DataFrame(rows), "F_filterarm_differenzen%s.csv" % sfx)
    schreibe_csv(pd.DataFrame(sw_rows), "F_filterarm_schwellenwechsel%s.csv" % sfx)


# ---------------------------------------------------------------------------------------------------------------
# Z2: Aggregationseffekt (V2 5.4) auf S_G, Designgewichte, gemeinsame Ziehungen
# ---------------------------------------------------------------------------------------------------------------
def bh_anno_duration_matched(nk_rec, anno, art):
    """Schlagmarken BRAVEHEART (.anno) gegen NeuroKit2-Cluster: QRS-Gipfel innerhalb 80 ms, eins zu eins (minimale Gesamtabweichung).
    Rueckgabe: (NK2 g1, g2, g3, BRAVEHEART-Median in ms, Zahl gematchter Schlaege); art 'qrs' oder 'qt'."""
    from scipy.optimize import linear_sum_assignment
    qc = nk_rec["qrs"]
    if len(qc) == 0 or anno is None or len(anno) == 0:
        return (np.nan,) * 4 + (0,)
    r_nk = np.array([cluster_anker(c) for c in qc])
    r_bh = anno[:, 1]
    cost = np.abs(r_nk[:, None] - r_bh[None, :])
    big = 1e6
    ri, ci = linear_sum_assignment(np.where(cost <= FENSTER_SAMPLES, cost, big))
    paare = [(i, j) for i, j in zip(ri, ci) if cost[i, j] <= FENSTER_SAMPLES]
    # nur Schlaege, deren NeuroKit2-Cluster fuer diese Groesse gueltig ist (gleiche Schlagmenge fuer beide Werkzeuge, V2 5.4)
    paare = [(i, j) for i, j in paare if _cluster_werte(qc[i], art) is not None]
    if not paare:
        return (np.nan,) * 4 + (0,)
    sub = [i for i, _ in paare]
    g1, g2, g3 = nk_agg(qc, art, subset=sub)
    if art == "qrs":
        dur = (anno[[j for _, j in paare], 2] - anno[[j for _, j in paare], 0]) * MS_PRO_SAMPLE
    else:
        dur = (anno[[j for _, j in paare], 3] - anno[[j for _, j in paare], 0]) * MS_PRO_SAMPLE
    return (g1, g2, g3, float(np.median(dur)), len(paare))


def stufe_z2(d):
    ids = d.sg
    w = d.sg_gew
    pos_sg = d.pos[ids].to_numpy()
    units, info = [], {}
    for art, g in (("qrs", "qrs_dauer_global"), ("qt", "qt_global")):
        nkv = {k: np.array([d.nk_rec[("ptbxl", "VV0", int(i))]["werte"].get("%s_%s" % (art.upper(), k), np.nan) if ("ptbxl", "VV0", int(i)) in d.nk_rec else np.nan
                            for i in ids]) for k in ("g1", "g2", "g3")}
        partner = {}
        for t in ("BRAVEHEART", "12SL", "ECGDeli", "Uni-G"):
            v, s = d.Q[(t, g)]
            partner[t] = np.where(s[pos_sg] == "ok", v[pos_sg], np.nan)
        # Schlagabgleich mit BRAVEHEART-Schlagmarken
        bm = np.full(len(ids), np.nan)
        gm = {k: np.full(len(ids), np.nan) for k in ("g1", "g2", "g3")}
        nmatch = np.zeros(len(ids))
        for n_, i in enumerate(ids):
            rec = d.nk_rec.get(("ptbxl", "VV0", int(i)))
            if rec is None:
                continue
            g1, g2, g3, bmed, nm = bh_anno_duration_matched(rec, d.bh_anno[("ptbxl", "VV0")].get(int(i)), art)
            gm["g1"][n_], gm["g2"][n_], gm["g3"][n_], bm[n_], nmatch[n_] = g1, g2, g3, bmed, nm
        partner_alle = dict(partner)
        # Verfahrenseffekt innerhalb BRAVEHEART (V2 5.4): Dauer aus Schlagmarken (.anno, Median ueber alle Schlaege) gegen Medianschlag-Wert
        va_ = np.full(len(ids), np.nan)
        for n_, i in enumerate(ids):
            z_ = d.bh_anno[("ptbxl", "VV0")].get(int(i))
            if z_ is not None and len(z_):
                dd_ = (z_[:, 2] - z_[:, 0]) if art == "qrs" else (z_[:, 3] - z_[:, 0])
                dd_ = dd_[np.isfinite(dd_)]
                if len(dd_):
                    va_[n_] = float(np.median(dd_)) * MS_PRO_SAMPLE
        mv_ = partner["BRAVEHEART"]
        cv_ = np.flatnonzero(np.isfinite(va_) & np.isfinite(mv_))
        if len(cv_) >= 10:
            dv_ = va_[cv_] - mv_[cv_]
            nmv = "Z2v|%s|BRAVEHEART|Schlagmarken-Median-minus-Medianschlag" % art
            units.append(Einheit(nmv, cv_, dv_, ind={"inn_5": np.abs(dv_) <= 5, "inn_10": np.abs(dv_) <= 10}, gew=w[cv_]))
            info[nmv] = dict(art=art, fam="verfahrenseffekt_BRAVEHEART", partner="BRAVEHEART", kand="BRAVEHEART_Schlagmarken", komp="BRAVEHEART_Medianschlag",
                             g="-", n=len(cv_), n_gematcht_mittel=np.nan)
        for fam, (vals, partner_map) in {"unabgeglichen": (nkv, partner_alle), "schlagabgeglichen": (gm, {"BRAVEHEART_Schlagmarken": bm})}.items():
            for pt, pv in partner_map.items():
                common = np.isfinite(pv) & np.isfinite(vals["g1"]) & np.isfinite(vals["g2"]) & np.isfinite(vals["g3"])
                if common.sum() < 10:
                    continue
                tool_p = pt.split("_")[0]
                cand, comp = orientiere("NeuroKit2", tool_p) if tool_p in ORDNUNG else ("NeuroKit2", pt)
                for gk in ("g1", "g2", "g3"):
                    nk_v = vals[gk][common]
                    pv_c = pv[common]
                    a, b = (nk_v, pv_c) if cand == "NeuroKit2" else (pv_c, nk_v)
                    nm = "Z2|%s|%s|%s|%s-%s|%s" % (art, fam, pt, cand, comp, gk)
                    sel = np.flatnonzero(common)
                    dd = a - b
                    units.append(Einheit(nm, sel, dd, ind={"inn_5": np.abs(dd) <= 5, "inn_10": np.abs(dd) <= 10}, gew=w[sel]))
                    info[nm] = dict(art=art, fam=fam, partner=pt, kand=cand, komp=comp, g=gk, n=int(common.sum()),
                                    n_gematcht_mittel=float(np.mean(nmatch[common])) if fam == "schlagabgeglichen" else np.nan)
    bs = boot_lauf(units, len(ids), strata=d.sg_stratum)
    pk = {e.name: punktschaetzung(e) for e in units}
    rows, drows = [], []
    for nm, inf in info.items():
        row = dict(paar=nm, **inf)
        row.update(flach(nm, pk[nm], bs[nm], {"bias", "mae", "inn_5", "inn_10"}))
        row["bias_einheit"] = "ms, gewichtet (Designgewichte), Differenz = %s minus %s" % (inf["kand"], inf["komp"])
        rows.append(row)
    for r_ in rows:
        r_["rolle"] = ("sekundaer, beschreibend (Aggregationseffekt Z2, Basis fuer KA4), punktweise Intervalle" if (r_["fam"] == "schlagabgeglichen" and r_["art"] == "qrs")
                       else "explorativ (kein Schlagabgleich bzw. kein g-Vergleich; nicht Basis fuer KA4), punktweise Intervalle")
    schreibe_csv(pd.DataFrame(rows), "Z2_aggregation_paare.csv")
    # Differenzen zwischen Aggregationsvarianten (gemeinsame Ziehungen)
    gruppen = {}
    for nm, inf in info.items():
        gruppen.setdefault((inf["art"], inf["fam"], inf["partner"]), {})[inf["g"]] = nm
    for (art, fam, pt), gm_ in sorted(gruppen.items()):
        if len(gm_) < 3:
            continue
        for a_, b_ in (("g1", "g2"), ("g1", "g3"), ("g2", "g3")):
            na, nb = gm_[a_], gm_[b_]
            for stat in ("mae", "bias"):
                arr = bs[na][stat] - bs[nb][stat]
                lo, hi = ki95(arr)
                pkt = pk[na][stat] - pk[nb][stat]
                drows.append(dict(groesse=art, familie=fam, partner=pt, statistik=stat, differenz="%s minus %s" % (a_, b_), punkt=pkt, ki_lo=lo, ki_hi=hi,
                                  band_betrag_punkt=band_text("zeit", abs(pkt)) if stat == "mae" else "", n=info[na]["n"]))
    dtab = pd.DataFrame(drows)
    schreibe_csv(dtab, "Z2_aggregationseffekt_differenzen.csv")
    # KA4: mindestens zwei unabhaengige Paare rechenbar?
    # Rechenbar im Sinn von V2 5.4 und der Matrix (Abschnitt 3) sind Paare mit Schlagmarken auf beiden Seiten und Schlagabgleich: hier nur
    # NeuroKit2 gegen BRAVEHEART-Schlagmarken (QRS-Dauer). Alle anderen Partner (12SL, Uni-G, ECGDeli global, BRAVEHEART Medianschlag) sind explorativ.
    unabh = sorted(set(inf["partner"] for inf in info.values() if inf["fam"] == "schlagabgeglichen" and inf["art"] == "qrs"))
    expl = sorted(set(inf["partner"] for inf in info.values() if inf["fam"] in ("unabgeglichen",)))
    ka4 = dict(rechenbare_unabhaengige_partner=unabh, anzahl=len(unabh), ka4_gilt=len(unabh) >= 2, explorative_partner_ohne_schlagabgleich=expl,
               hinweis="Weniger als zwei unabhaengige Paare: KA4 entfaellt (V2 5.4 Fallback)" if len(unabh) < 2 else "")
    if not dtab.empty:
        sub = dtab[(dtab.statistik == "mae") & (dtab.groesse == "qrs")]
        ka4["mae_differenz_qrs_spanne_min"] = float(sub.punkt.abs().min()) if len(sub) else None
        ka4["mae_differenz_qrs_spanne_max"] = float(sub.punkt.abs().max()) if len(sub) else None
    RES["KA4"] = ka4
    schreibe_text(json.dumps(ka4, indent=1, ensure_ascii=False), "Z2_KA4_bedingung.json")


# ---------------------------------------------------------------------------------------------------------------
# Z3: LUDB-Grenzfehler gegen Expertenreferenz (V2 5.5)
# ---------------------------------------------------------------------------------------------------------------
LUDB_EXT = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]


def lese_ludb_ableitung(pfad, ext):
    import wfdb
    a = wfdb.rdann(str(pfad), ext)
    out = {"p": [], "N": [], "t": []}
    on = pk = None
    pksym = None
    for s, sym in zip(a.sample, a.symbol):
        if sym == "(":
            on, pk, pksym = int(s), None, None
        elif sym in out and on is not None:
            pk, pksym = int(s), sym
        elif sym == ")":
            if on is not None and pk is not None and pksym is not None:
                out[pksym].append((pk, on, int(s)))
            on = pk = pksym = None
    return out


def lese_ludb_referenz(aid):
    items = {"p": [], "N": [], "t": []}
    alle = []
    for li, ext in enumerate(LUDB_EXT):
        res = lese_ludb_ableitung(PFADE["ludb"] / str(aid), ext)
        for kind, lst in res.items():
            for pk, on, off in lst:
                items[kind].append((float(pk), li, (float(on), float(off))))
                alle += [on, off]
    qrs = cluster_peaks(items["N"])
    pc = cluster_peaks(items["p"])
    tc = cluster_peaks(items["t"])
    return dict(qrs=qrs, p=pc, t=tc, dom=(min(alle), max(alle)) if alle else (np.nan, np.nan))


def wave_info(c, kind, nk):
    ons, offs = [], []
    for pk, L, pl in c:
        if nk:
            on, off = {"qrs": (pl[I_RON], pl[I_ROFF]), "p": (pl[I_PON], pl[I_POFF]), "t": (pl[I_TON], pl[I_TOFF])}[kind]
        else:
            on, off = pl
        ons.append(on)
        offs.append(off)
    ons = np.array(ons, float)
    offs = np.array(offs, float)

    def agg(f, a):
        a = a[np.isfinite(a)]
        return float(f(a)) if len(a) else float("nan")

    return dict(peak=cluster_anker(c), n=len(c), on_ext=agg(np.min, ons), off_ext=agg(np.max, offs), on_med=agg(np.median, ons), off_med=agg(np.median, offs))


def beats_aus(qrs_cl, p_cl, t_cl, nk):
    P_ = sorted((wave_info(c, "p", nk) for c in p_cl), key=lambda x: x["peak"])
    T_ = sorted((wave_info(c, "t", nk) for c in t_cl), key=lambda x: x["peak"])
    beats = []
    for c in qrs_cl:
        q = wave_info(c, "qrs", nk)
        pcand = [x for x in P_ if 0 < q["peak"] - x["peak"] <= 200]  # letzter P-Gipfel vor dem Anker innerhalb 400 ms
        tcand = [x for x in T_ if 0 < x["peak"] - q["peak"] <= 350]  # erster T-Gipfel danach innerhalb 700 ms
        beats.append(dict(r=q["peak"], q=q, p=(pcand[-1] if pcand else None), t=(tcand[0] if tcand else None)))
    beats.sort(key=lambda b: b["r"])
    return beats


GRENZEN_ALLE = ["p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"]


def ref_grenze(b, name, rv):
    wave = {"p": "p", "qrs": "q", "t": "t"}[name.split("_")[0]]
    w_ = b[wave]
    if w_ is None:
        return float("nan"), None
    return w_[("on_" if name.endswith("_on") else "off_") + rv], w_["peak"]


def werkzeug_beats_ll(d, ds, vv, aid, tabelle):
    t = tabelle.get((ds, vv, aid))
    if t is None:
        return []
    out = []
    for r in t.itertuples():
        pv = getattr(r, "p_vorhanden", 1)
        pv = 1 if (pd.isna(pv) or pv == "") else int(pv)
        b = {"qrs_on": r.qrs_beginn, "qrs_off": r.qrs_ende, "t_off": r.t_ende}
        if pv:
            b["p_on"], b["p_off"] = r.p_beginn, r.p_ende
        out.append(dict(r=float(r.r_gipfel), b={k: float(v) for k, v in b.items() if pd.notna(v)}, pk={}))
    return out


def werkzeug_beats_bh(anno):
    if anno is None:
        return []
    out = []
    for z in anno:
        if not np.isfinite(z[1]):
            continue
        b = {k: float(v) for k, v in (("qrs_on", z[0]), ("qrs_off", z[2]), ("t_off", z[3])) if np.isfinite(v)}
        out.append(dict(r=float(z[1]), b=b, pk={}))
    return out


def werkzeug_beats_nk(rec, variante):
    bl = beats_aus(rec["qrs"], rec["p"], rec["t"], True)
    out = []
    for b in bl:
        bb = {}
        pk = {}
        for nm in GRENZEN_ALLE:
            v, peak = ref_grenze(b, nm, variante)
            if np.isfinite(v):
                bb[nm] = v
            if nm.startswith("p") and peak is not None:
                pk["p"] = peak
            if nm.startswith("t") and peak is not None:
                pk["t"] = peak
        out.append(dict(r=b["r"], b=bb, pk=pk))
    return out


def fehler_aufnahme(ref_beats, tool_beats, fenster, nur_varianten=("ext", "med")):
    """Eins-zu-eins-Zuordnung ueber den QRS-Gipfel; Rueckgabe: Liste (ref_idx, ref_variante, grenze, fehler_ms) und Zaehlung nicht_zuordenbar."""
    from scipy.optimize import linear_sum_assignment
    if not ref_beats or not tool_beats:
        return [], 0, 0
    rr = np.array([b["r"] for b in ref_beats])
    tr = np.array([b["r"] for b in tool_beats])
    cost = np.abs(rr[:, None] - tr[None, :])
    ri, ci = linear_sum_assignment(np.where(cost <= fenster, cost, 1e6))
    rows = []
    nz_set = set()
    nm = 0
    for i, j in zip(ri, ci):
        if cost[i, j] > fenster:
            continue
        nm += 1
        tb = tool_beats[j]
        for nme, tv in tb["b"].items():
            for rv in nur_varianten:
                rg, rpk = ref_grenze(ref_beats[i], nme, rv)
                if not np.isfinite(rg):
                    continue
                wave = nme.split("_")[0]
                if wave in ("p", "t") and wave in tb["pk"] and rpk is not None and abs(tb["pk"][wave] - rpk) > FENSTER_SAMPLES:
                    nz_set.add((i, wave, rv))
                    continue
                rows.append((i, rv, nme, (tv - rg) * MS_PRO_SAMPLE))
    return rows, len(nz_set), nm


def referenz_dauer(ref_beats, rv, art):
    v = []
    for b in ref_beats:
        on, off = b["q"]["on_" + rv], b["q"]["off_" + rv]
        if art == "qrs":
            if np.isfinite(on) and np.isfinite(off):
                v.append((off - on) * MS_PRO_SAMPLE)
        elif b["t"] is not None and np.isfinite(b["t"]["off_" + rv]) and np.isfinite(on):
            v.append((b["t"]["off_" + rv] - on) * MS_PRO_SAMPLE)
    return float(np.median(v)) if v else float("nan")


def stufe_z3(d):
    ids = np.array(sorted(set(d.bh_ludb.v.loc[("ludb", "VV0")].index.astype(int))), dtype=np.int64)
    log("Z3: lese LUDB-Expertenreferenz (%d Aufnahmen)" % len(ids))
    ref = {}
    for aid in ids:
        r = lese_ludb_referenz(int(aid))
        r["beats"] = beats_aus(r["qrs"], r["p"], r["t"], False)
        ref[int(aid)] = r
    rec_idx = {int(a): k for k, a in enumerate(ids)}
    lbt = {k: g for k, g in d.ll_beats[d.ll_beats.datensatz == "ludb"].groupby(["datensatz", "vorverarbeitung", "aufnahme"])}
    ebenen = []  # (ebene, vv, beats_fn)
    for vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2"):
        ebenen.append(("LoopLens_global", vv, lambda aid, vv=vv: werkzeug_beats_ll(d, "ludb", vv, aid, lbt)))
    for vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2", "VV1", "VV2"):
        ebenen.append(("BRAVEHEART_Schlagmarken", vv, lambda aid, vv=vv: werkzeug_beats_bh(d.bh_anno.get(("ludb", vv), {}).get(aid))))
        for var, nm in (("ext", "NeuroKit2_g1"), ("med", "NeuroKit2_median")):
            ebenen.append((nm, vv, lambda aid, vv=vv, var=var: (werkzeug_beats_nk(d.nk_rec[("ludb", vv, aid)], var) if ("ludb", vv, aid) in d.nk_rec else [])))
    long_rows, zaehl_rows = [], []
    for fenster, tag in ((FENSTER_SAMPLES, "80ms"), (75, "150ms")):
        for ebene, vv, fn in ebenen:
            if tag == "150ms" and vv != "VV0":
                continue
            n_ref = n_tool = n_match = n_nz = n_dom = 0
            for aid in ids:
                aid = int(aid)
                rb = ref[aid]["beats"]
                tb = fn(aid)
                rows, nz, nm = fehler_aufnahme(rb, tb, fenster)
                n_ref += len(rb)
                n_tool += len(tb)
                lo_d, hi_d = ref[aid]["dom"]
                n_dom += sum(1 for b_ in tb if np.isfinite(lo_d) and lo_d <= b_["r"] <= hi_d)
                n_match += nm
                n_nz += nz
                for (ri, rv, nme, err) in rows:
                    long_rows.append((tag, ebene, vv, rv, nme, aid, ri, err))
            zaehl_rows.append(dict(fenster=tag, ebene=ebene, vorverarbeitung=vv, n_aufnahmen=len(ids), n_referenzschlaege=n_ref, n_werkzeugschlaege=n_tool,
                                   n_werkzeugschlaege_in_expertendomaene=n_dom, n_gematcht=n_match, anteil_referenz_gematcht=(n_match / n_ref if n_ref else np.nan), n_nicht_zuordenbar_p_t=n_nz))
    L = pd.DataFrame(long_rows, columns=["fenster", "ebene", "vv", "ref", "grenze", "aufnahme", "ref_schlag", "fehler_ms"])
    schreibe_csv(pd.DataFrame(zaehl_rows), "Z3_ludb_zuordnung.csv")
    # Einheiten
    units, meta_u = [], {}
    for (tag, ebene, vv, rv, gr), g in L.groupby(["fenster", "ebene", "vv", "ref", "grenze"]):
        cl = np.array([rec_idx[a] for a in g.aufnahme])
        e = g.fehler_ms.to_numpy(float)
        nm = "Z3|%s|%s|%s|%s|%s" % (tag, ebene, vv, rv, gr)
        units.append(Einheit(nm, cl, e, quant={"d": (e, (0.5,)), "absd": (np.abs(e), (0.5, 0.95))}))
        meta_u[nm] = dict(fenster=tag, ebene=ebene, vv=vv, ref=rv, grenze=gr, n_schlaege=len(g), n_aufnahmen=int(g.aufnahme.nunique()))
    # gepaarte Aenderungen gegen Basisarm
    paare = []
    for ebene in L.ebene.unique():
        if ebene == "LoopLens_global":
            vergl = [("VV0_G0.5", "VV0_G1"), ("VV0_G2", "VV0_G1")]
        else:
            vergl = [("VV0_G0.5", "VV0_G1"), ("VV0_G2", "VV0_G1"), ("VV1", "VV0"), ("VV2", "VV0")]
        for arm, bas in vergl:
            A = L[(L.fenster == "80ms") & (L.ebene == ebene) & (L.vv == arm)]
            Bq = L[(L.fenster == "80ms") & (L.ebene == ebene) & (L.vv == bas)]
            if A.empty or Bq.empty:
                continue
            m = A.merge(Bq, on=["ref", "grenze", "aufnahme", "ref_schlag"], suffixes=("_arm", "_bas"))
            for (rv, gr), g in m.groupby(["ref", "grenze"]):
                nm = "Z3p|%s|%s|%s|%s|%s" % (ebene, arm, bas, rv, gr)
                cl = np.array([rec_idx[a] for a in g.aufnahme])
                ea, eb = np.abs(g.fehler_ms_arm.to_numpy(float)), np.abs(g.fehler_ms_bas.to_numpy(float))
                dd = ea - eb
                units.append(Einheit(nm, cl, dd, ind={"fehler_kleiner": ea < eb, "fehler_groesser": ea > eb},
                                     quant={"abs_arm": (ea, (0.5,)), "abs_bas": (eb, (0.5,))}))
                paare.append((nm, dict(ebene=ebene, arm=arm, basis=bas, ref=rv, grenze=gr, n_schlaege=len(g), n_aufnahmen=int(g.aufnahme.nunique()))))
    # Dauerfehler je Aufnahme (Werkzeugwert minus Median der Referenzdauer der Aufnahme)
    RD = {(art, rv): np.array([referenz_dauer(ref[int(a)]["beats"], rv, art) for a in ids]) for art in ("qrs", "qt") for rv in ("ext", "med")}
    for vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2", "VV1", "VV2"):
        llv = {q: d.ll.hole("ludb", vv, q, ids) for q in ("qrs_dauer_global", "qt_global")} if vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2") else {}
        bhv = {q: (d.bh_ludb if vv == "VV0" else d.bh_arme).hole("ludb", vv, q, ids) for q in ("qrs_dauer_global", "qt_global")}
        nkv = werte_tool(d, "NeuroKit2", "ludb", vv, ids)
        bh_anno = d.bh_anno.get(("ludb", vv), {})
        for art, key in (("qrs", "qrs_dauer_global"), ("qt", "qt_global")):
            quellen = []
            if key in llv:
                quellen.append(("LoopLens_global", llv[key]))
            quellen.append(("BRAVEHEART_Medianschlag", bhv[key]))
            vb = np.full(len(ids), np.nan)
            sb = np.full(len(ids), "fehlend", dtype=object)
            for k, aid in enumerate(ids):
                z = bh_anno.get(int(aid))
                if z is not None and len(z):
                    vb[k] = float(np.median((z[:, 2] - z[:, 0]) * MS_PRO_SAMPLE)) if art == "qrs" else float(np.median((z[:, 3] - z[:, 0]) * MS_PRO_SAMPLE))
                    sb[k] = "ok"
            quellen.append(("BRAVEHEART_Schlagmarken_Median", (vb, sb)))
            for gk in ("g1", "g2", "g3"):
                quellen.append(("NeuroKit2_" + gk, nkv["%s_%s" % (art.upper(), gk)]))
            for nme, (v, s_) in quellen:
                for rv in ("ext", "med"):
                    rd = RD[(art, rv)]
                    ok = (s_ == "ok") & np.isfinite(v) & np.isfinite(rd)
                    if ok.sum() < 3:
                        continue
                    e = v[ok] - rd[ok]
                    nm = "Z3d|%s|%s|%s|%s|%s" % (art, nme, vv, rv, "dauer")
                    units.append(Einheit(nm, np.flatnonzero(ok), e, quant={"d": (e, (0.5,)), "absd": (np.abs(e), (0.5, 0.95))}))
                    meta_u[nm] = dict(art=art, werkzeug_ebene=nme, vv=vv, ref=rv, n_aufnahmen=int(ok.sum()))
    log("Z3 Bootstrap: %d Einheiten" % len(units))
    bs = boot_lauf(units, len(ids))
    pk = {e.name: punktschaetzung(e) for e in units}
    rows, prow, drow = [], [], []
    for e in units:
        if e.name.startswith("Z3|"):
            r = dict(meta_u[e.name])
            r.update(flach(e.name, pk[e.name], bs[e.name]))
            r["band_median_abs_punkt"] = band_text("zeit", pk[e.name]["absd_q0.5"])
            r["band_median_abs_mit_ki"] = band_mit_ki("zeit", pk[e.name]["absd_q0.5"], *ki95(bs[e.name]["absd_q0.5"]))
            rows.append(r)
        elif e.name.startswith("Z3d|"):
            r = dict(meta_u[e.name])
            r.update(flach(e.name, pk[e.name], bs[e.name]))
            r["band_median_abs_punkt"] = band_text("zeit", pk[e.name]["absd_q0.5"])
            drow.append(r)
    dpair = dict(paare)
    for e in units:
        if e.name.startswith("Z3p|"):
            r = dict(dpair[e.name])
            r.update(flach(e.name, pk[e.name], bs[e.name]))
            r["median_abs_arm_minus_basis"] = pk[e.name]["abs_arm_q0.5"] - pk[e.name]["abs_bas_q0.5"]
            lo, hi = ki95(bs[e.name]["abs_arm_q0.5"] - bs[e.name]["abs_bas_q0.5"])
            r["median_abs_arm_minus_basis_lo"], r["median_abs_arm_minus_basis_hi"] = lo, hi
            prow.append(r)
    T = pd.DataFrame(rows)
    T["rolle"] = "sekundaer, beschreibend (LUDB-Referenzabweichung bei je Ableitung normierten Amplituden), punktweise Intervalle"
    schreibe_csv(T, "Z3_ludb_grenzfehler.csv")
    schreibe_csv(pd.DataFrame(drow), "Z3_ludb_dauerfehler.csv")
    schreibe_csv(pd.DataFrame(prow), "Z3_ludb_arme_gepaart.csv")
    # KA5: Vorzeichen und Band unter R_ext und R_med (nativ, 80 ms)
    ka5 = []
    if not T.empty:
        sub = T[(T.fenster == "80ms") & (T.vv == "VV0")]
        for (ebene, gr), g in sub.groupby(["ebene", "grenze"]):
            ge, gm = g[g.ref == "ext"], g[g.ref == "med"]
            if ge.empty or gm.empty:
                continue
            e_, m_ = ge.iloc[0], gm.iloc[0]
            sg_e, sg_m = np.sign(e_["d_q0.5"]), np.sign(m_["d_q0.5"])
            b_e, b_m = band_text("zeit", e_["absd_q0.5"]), band_text("zeit", m_["absd_q0.5"])
            gleich = bool(sg_e == sg_m and b_e == b_m)
            ka5.append(dict(ebene=ebene, grenze=gr, median_fehler_ext=e_["d_q0.5"], median_fehler_med=m_["d_q0.5"], median_abs_ext=e_["absd_q0.5"],
                            median_abs_med=m_["absd_q0.5"], band_ext=b_e, band_med=b_m, vorzeichen_gleich=bool(sg_e == sg_m),
                            aussage=("Band " + b_e + " (unter beiden Referenzen)" if gleich else "referenzabhaengig"),
                            zusatz="bei je Ableitung normierten Amplituden"))
    if not T.empty:
        sub = T[(T.fenster == "80ms") & (T.vv == "VV0")]
        e_ = sub[(sub.ebene == "NeuroKit2_g1") & (sub.ref == "ext")].set_index("grenze")
        m_ = sub[(sub.ebene == "NeuroKit2_median") & (sub.ref == "med")].set_index("grenze")
        for gr in sorted(set(e_.index) & set(m_.index)):
            sg_e, sg_m = np.sign(e_.loc[gr, "d_q0.5"]), np.sign(m_.loc[gr, "d_q0.5"])
            b_e, b_m = band_text("zeit", e_.loc[gr, "absd_q0.5"]), band_text("zeit", m_.loc[gr, "absd_q0.5"])
            gleich = bool(sg_e == sg_m and b_e == b_m)
            ka5.append(dict(ebene="NeuroKit2 (g1 gegen R_ext; Median gegen R_med, V2 5.5)", grenze=gr, median_fehler_ext=e_.loc[gr, "d_q0.5"],
                            median_fehler_med=m_.loc[gr, "d_q0.5"], median_abs_ext=e_.loc[gr, "absd_q0.5"], median_abs_med=m_.loc[gr, "absd_q0.5"],
                            band_ext=b_e, band_med=b_m, vorzeichen_gleich=bool(sg_e == sg_m),
                            aussage=("Band " + b_e + " (unter beiden Referenzen)" if gleich else "referenzabhaengig"),
                            zusatz="bei je Ableitung normierten Amplituden"))
    schreibe_csv(pd.DataFrame(ka5), "Z3_KA5_referenzabhaengigkeit.csv")
    RES["Z3_n_units"] = len(units)


# ---------------------------------------------------------------------------------------------------------------
# P1-Urteil, Saetze, nicht gerechnete Teile, Hauptprogramm
# ---------------------------------------------------------------------------------------------------------------
def stufe_p1_urteil(d):
    r = RES["P1"]
    sch = RES["P1_schranken"]
    n, N = r["n_gemeinsam"], r["n_p0"]
    m = N - n
    bc = r["D120"] * n
    # MNAR: Bedingung (V2 9) und Kipp-Punkt
    abw = RES.get("mnar_schema", {})
    pflicht = [k for k, v in abw.items() if np.isfinite(v) and v > 0.05]
    mn = dict(n_fehlend=m, bedingung_unterschied_ueber_5_prozentpunkte_in_schemata=pflicht,
              D120_untere_schranke=sch["D120_untere_schranke_alle_fehlenden_konkordant"], D120_obere_schranke=sch["D120_obere_schranke_alle_fehlenden_diskordant"])
    if m > 0:
        qstar = (MARGE_H_P1 * N - bc) / m
        mn["anteil_fehlender_diskordant_noetig_damit_D_gleich_1_prozent"] = qstar
        mn["kippt_h_p1_bei_allen_fehlenden_konkordant"] = bool(sch["D120_untere_schranke_alle_fehlenden_konkordant"] <= MARGE_H_P1)
        mn["explorativ_anteil_bootstrap_untere_schranke_unter_marge_kein_test"] = sch["explorativ_anteil_bootstrap_untere_schranke_unter_marge_kein_test"]
        bias, lo, hi = r["bias"], r["bias_lo"], r["bias_hi"]
        mn["bias_kipp_delta_punkt_ms"] = -bias * N / m
        grenze = lo if lo > 0 else (hi if hi < 0 else 0.0)
        mn["bias_kipp_delta_ki_grenze_ms"] = (-(n * grenze) / m - bias) if grenze != 0 else 0.0
        mn["hinweis"] = ("Kipp-Punkt als Verschiebung der fehlenden Differenzen gegenueber dem beobachteten Mittel; Naeherung fuer die KI-Grenze "
                         "(Gewichtung n/N, m/N).")
    schreibe_csv(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in mn.items()}]), "P1_mnar_kipppunkt.csv")
    RES["P1_mnar"] = mn
    # Urteil nach KA1 (V2 13)
    D, lo, hi = r["D120"], r["D120_lo"], r["D120_hi"]
    t = []
    t.append("# P1: QRS-Dauer und 120-ms-Zuordnung, LoopLens gegen BRAVEHEART (bestaetigend)\n")
    t.append("Nenner: %d von %d Personen in P0 (gemeinsamer Nenner, beide Werkzeuge Status ok). Differenz = LoopLens minus BRAVEHEART.\n" % (n, N))
    t.append("Entscheidungsstatistik D120 = %s %% (95-%%-Perzentil-KI %s bis %s %%), Band am Punktschaetzer: %s; Band mit KI: %s.\n" % (
        proz(D), proz(lo), proz(hi), r["D120_band"], r["D120_band_mit_ki"]))
    t.append("H_P1 (D120 groesser als 1 %%): einseitiger Bootstrap-p-Wert = %s (10.000 Ziehungen, Seed %d, Minimum 1e-4); Holm mit m = 1 (Nachtrag 0), Schwelle %.2f: **%s**.\n" % (
        ("%.4g" % r["h_p1_p_einseitig"]), SEED_BOOT, ALPHA, "H_P1 abgelehnt" if r["h_p1_abgelehnt"] else "H_P1 nicht abgelehnt"))
    if not r["h_p1_abgelehnt"]:
        t.append("\nSatzwahl KA1 (nicht abgelehnt): \"Eine Diskordanz ueber 1 %% ist nicht nachweisbar (%s %%, KI %s bis %s).\"\n" % (proz(D), proz(lo), proz(hi)))
    else:
        rd = band_rang("diskordanz", D)
        rm = band_rang("zeit", r["median_abs_diff"])
        gleich = rd == rm
        t.append("\nSatzwahl KA1 (abgelehnt): D120 = %s %% (Band: %s); D120-nah (Personen mit Mittel beider Werkzeuge von 100 bis unter 140 ms, n = %d) = %s %% (KI %s bis %s; Band: %s); "
                 "Median des Betrags der Differenz %s ms (KI %s bis %s; Band: %s). " % (
                     proz(D), r["D120_band_mit_ki"], r["D120_nah_n"], proz(r["D120_nah"]), proz(r["D120_nah_lo"]), proz(r["D120_nah_hi"]),
                     r["D120_nah_band_mit_ki"], fz(r["median_abs_diff"]), fz(r["median_abs_lo"]), fz(r["median_abs_hi"]), r["median_band_mit_ki"]))
        if gleich:
            t.append("Die Baender von D120 und Median liegen auf derselben Stufe (Stufe %d von 4).\n" % rd)
        else:
            t.append("Die Baender von D120 (Stufe %d von 4) und Median (Stufe %d von 4) liegen auseinander: beide werden genannt, es wird kein Gesamtadjektiv gebildet.\n" % (rd, rm))
    t.append("\n## Mitberichte (nicht entscheidend)\n")
    t.append("- Bias (LoopLens minus BRAVEHEART): %s ms (KI %s bis %s); MAE %s ms; Median des Betrags %s ms.\n" % (fz(r["bias"]), fz(r["bias_lo"]), fz(r["bias_hi"]), fz(r["mae"]), fz(r["median_abs_diff"])))
    t.append("- Empirische Limits of Agreement (2,5 und 97,5 %%): %s bis %s ms (KI untere Grenze %s bis %s, obere Grenze %s bis %s).\n" % (
        fz(r["loa_lo"]), fz(r["loa_hi"]), fz(r["d_q0.025_lo"]), fz(r["d_q0.025_hi"]), fz(r["d_q0.975_lo"]), fz(r["d_q0.975_hi"])))
    t.append("- Anteil innerhalb 5 ms: %s %% (KI %s bis %s), innerhalb 10 ms: %s %% (KI %s bis %s).\n" % (
        proz(r["innerhalb_5"]), proz(r["inn_5_lo"]), proz(r["inn_5_hi"]), proz(r["innerhalb_10"]), proz(r["inn_10_lo"]), proz(r["inn_10_hi"])))
    t.append("- PPA %s %% (KI %s bis %s), NPA %s %% (KI %s bis %s), Cohens Kappa %s (KI %s bis %s) an 120 ms.\n" % (
        proz(r["ppa"]), proz(r["t120_ppa_lo"]), proz(r["t120_ppa_hi"]), proz(r["npa"]), proz(r["t120_npa_lo"]), proz(r["t120_npa_hi"]),
        fz(r["kappa"], 3), fz(r["t120_kappa_lo"], 3), fz(r["t120_kappa_hi"], 3)))
    t.append("- Sekundaere Schwellen: D110 = %s %% (KI %s bis %s), D130 = %s %% (KI %s bis %s).\n" % (
        proz(RES["p0_pk"][RES["P1_name"]]["t110_disk"]), proz(r["t110_disk_lo"]), proz(r["t110_disk_hi"]),
        proz(RES["p0_pk"][RES["P1_name"]]["t130_disk"]), proz(r["t130_disk_lo"]), proz(r["t130_disk_hi"])))
    t.append("- Status LoopLens in P0: %s; Status BRAVEHEART in P0: %s.\n" % (json.dumps(r["status_looplens"]), json.dumps(r["status_braveheart"])))
    t.append("\n## Fehlende Werte (V2 6.2 und 9)\n")
    t.append("- Schranken fuer D120 bei %d fehlenden Personen: untere Schranke (alle Fehlenden konkordant) %s %%, obere Schranke (alle diskordant) %s %%; Wert auf dem gemeinsamen Nenner %s %%.\n" % (
        m, proz(sch["D120_untere_schranke_alle_fehlenden_konkordant"]), proz(sch["D120_obere_schranke_alle_fehlenden_diskordant"]), proz(D)))
    t.append("- Untere Schranke: Anteil der Bootstrap-Schaetzer der unteren Schranke bis zur Marge 1 %% (explorativ, kein Test, V2 6.3) = %s (Bootstrap ueber alle Personen in P0).\n" % ("%.4g" % sch["explorativ_anteil_bootstrap_untere_schranke_unter_marge_kein_test"]))
    t.append("- MNAR-Pflichtbedingung (Unterschied der gemeinsamen Ausfallanteile ueber 5 Prozentpunkte zwischen Strata mit mindestens 100 Personen): erfuellt in %s.\n" % (
        ", ".join(pflicht) if pflicht else "keinem Schema"))
    if m > 0:
        q_ = mn["anteil_fehlender_diskordant_noetig_damit_D_gleich_1_prozent"]
        if q_ < 0:
            t.append("- Kipp-Punkt: D120 bleibt auch bei lauter konkordanten Fehlenden ueber 1 %% (rechnerischer Anteil %s %%, negativ); die Entscheidung kippt unter keiner Annahme ueber die Fehlenden.\n" % proz(q_))
        elif q_ > 1:
            t.append("- Kipp-Punkt: D120 bleibt auch bei lauter diskordanten Fehlenden unter 1 %% (rechnerischer Anteil %s %%).\n" % proz(q_))
        else:
            t.append("- Kipp-Punkt: damit D120 auf 1 %% faellt, muessten %s %% der fehlenden Personen diskordant sein.\n" % proz(q_))
    t.append("\nKeine Deutung ueber die vorab festgelegte Deutungsregel (V2 12) hinaus. Definitionsunterschiede der Operatoren siehe Nachtrag 0 und Tabelle 1.\n")
    schreibe_text("".join(t), "P1_URTEIL.md")
    # Bedingte Klinikersaetze (V2 13 mit Nachtrag 0 Folge 4)
    s = ["# Bedingte Saetze (Bedingungen ausgewertet, Text nicht ueber die festgeschriebene Fassung hinaus)\n\n"]
    if r["h_p1_abgelehnt"]:
        s.append("**Satz A (Bedingung H_P1 abgelehnt: erfuellt):** \"Wer QRS-Dauer aus LoopLens und BRAVEHEART mischt, sollte die Definition pruefen: An der 120-ms-Grenze ordnen die Werkzeuge %s %% der Personen unterschiedlich zu.\"\n\n" % proz(D))
        s.append("**Satz B:** Bedingung nicht erfuellt, Satz entfaellt.\n\n")
    else:
        s.append("**Satz A:** Bedingung nicht erfuellt, Satz entfaellt.\n\n")
        s.append("**Satz B (Bedingung H_P1 nicht abgelehnt: erfuellt):** \"Fuer die QRS-Dauer an der 120-ms-Grenze liegen LoopLens und BRAVEHEART auf PTB-XL innerhalb der berichteten Baender; fuer andere Geraete oder Datensaetze folgt daraus nichts.\"\n\n")
    zel = RES.get("V_zellen")
    sc = []
    if zel is not None and len(zel) and "anteil_gewichtet" in zel:
        z2 = zel[(zel.art.isin(["zeit", "winkel"])) & (zel.anteil_gewichtet >= 0.05)]
        sc = [(x.werkzeug, x.groesse, x.faktor, x.anteil_gewichtet) for x in z2.itertuples()]
    if sc:
        werkz = sorted(set(x[0] for x in sc))
        s.append("**Satz C (Bedingung mindestens eine Zeit- oder Winkelzelle im Band haeufig veraendert oder hoeher: erfuellt fuer %s):** \"Zeit- und Winkelgroessen von <Werkzeug> haengen bei gleichmaessiger Verstaerkungsaenderung von der Amplitude ab; absolute mV-Schwellen koennen bei unterschiedlichem Verstaerkungsniveau zu unterschiedlichen Ergebnissen fuehren.\" Zellen: %s\n\n" % (
            ", ".join(werkz), "; ".join("%s %s Faktor %g: %s %%" % (a, b, c, proz(e)) for a, b, c, e in sc)))
    else:
        s.append("**Satz C:** Bedingung nicht erfuellt (keine Zeit- oder Winkelzelle im Band haeufig veraendert oder hoeher), Satz entfaellt.\n\n")
    ka4 = RES.get("KA4", {})
    if ka4.get("ka4_gilt"):
        s.append("**Satz D (Bedingung KA4 gilt: erfuellt, %d unabhaengige Partner rechenbar):** \"Die Wahl der Aggregation allein verschiebt die QRS-Dauer um <a> bis <b> ms.\" Spanne der Betraege der MAE-Differenzen (QRS): %s bis %s ms (siehe Z2_aggregationseffekt_differenzen.csv; Werte <a> und <b> vom Koordinator festzulegen).\n" % (
            ka4.get("anzahl", 0), fz(ka4.get("mae_differenz_qrs_spanne_min")), fz(ka4.get("mae_differenz_qrs_spanne_max"))))
    else:
        s.append("**Satz D:** Bedingung nicht erfuellt, Satz entfaellt.\n")
    schreibe_text("".join(s), "SAETZE_Bedingungen.md")


NICHT_GERECHNET = [
    ("P2 Peak-QRS-T-Winkel als bestaetigende Groesse", "entfallen nach Nachtrag 0 Abschnitt 1 (Operatoren verschieden); Winkel nur deskriptiv mit Suffix _ll und _bh (Z1_paare_P0_sekundaer.csv)"),
    ("Aggregationsvarianten g1 bis g3 fuer LoopLens", "LoopLens liefert nur fusionierte globale Grenzen, keinen ableitungsweisen Export (LOOPLENS_LAUF_AW219 Abschnitt 5 Punkt 2)"),
    ("Filterarm LoopLens (VV1, VV2 bzw. VV2n)", "im Auftrag AW-219 nicht gerechnet (kein Lauf der Kette mit externem Bandpass); es liegen keine Werte vor"),
    ("P- und T-Gipfelpruefung (nicht_zuordenbar) fuer LoopLens und BRAVEHEART", "diese Exporte enthalten keine P- und T-Gipfel (BRAVEHEART gar keine P-Marken); die P-/T-Zuordnung geschieht dort ueber den Schlag; die Pruefung gilt nur fuer NeuroKit2"),
    ("LUDB-Pilotaufnahmen 1 bis 5 kennzeichnen", "nicht gekennzeichnet; sie bleiben in der Analyse (V2 4.2)"),
    ("Medianschlag-Dauer BRAVEHEART gegen NeuroKit2 als Aggregationseffekt", "gehoert nach V2 5.4 nicht zum Aggregationseffekt; als explorativer Vergleich ohne Schlagabgleich ausgewiesen"),
    ("S8 ohne Randschlaege", "R_rand (Kontrolle K6, Sprungantwort) wurde nicht gemessen; die Kennzeichnung ist nicht moeglich"),
    ("S9 ECGDeli ableitungsweise mit aVL/aVF-Kreuzzuordnung", "ableitungsweise ECGDeli-Spalten nicht eingelesen; ECGDeli geht nur mit globalen Spalten ein"),
    ("S10 QTcF nativ gegen QTcF_g", "RR_ref nach V2 5.3 braucht das Signal (nk.ecg_peaks Ableitung II); die Auswertung liest keine Signale"),
    ("QTcF_g", "wie S10"),
    ("Rastercheck der Archivwerte (Rest modulo 2 und 4 ms)", "nur an PT vorgesehen (V2 5.3); Zeitbaender wurden nicht nach groeberem Raster angepasst"),
    ("Uni-G VM_Ang_QRS-T gegen Peak-Winkel", "Peak-Operator nicht belegt (Nachtrag 0 Abschnitt 1 Folge 6); nativ-only ohne Namensgleichheit"),
    ("SVG gegen BRAVEHEART svg_peak_mag und svg_area_mag", "Gleichheit der Groessen nicht belegt; nur Skalierungsarm"),
    ("Rueckskalierung O2 (LUDB)", "Entscheidung K steht aus (V2 5.5)"),
    ("Adjudikation O3", "entfaellt (AE-270)"),
    ("Strata fuer LUDB", "nicht gerechnet; LUDB nur ueber alle 200 Aufnahmen"),
    ("Strata fuer andere Paare als P1", "Strata nur fuer P1 (V2 4.2 Strata vorab fuer die Primaergroesse)"),
    ("Schlagzahl BRAVEHEART-Medianschlag", "Zahl der Schlaege aus processed_ecgs_roh (num_beats), nicht aus Schlagmarken"),
]


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nur-hash", action="store_true")
    ap.add_argument("--test", action="store_true", help="nur fuer den Test: Pfade werden von ausserhalb gesetzt, Hashpruefung uebersprungen")
    ap.add_argument("--b", type=int, default=None, help="nur Test: Zahl der Ziehungen")
    ap.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args(argv)
    if (args.b or args.chunk) and not args.test:
        print("--b und --chunk sind nur im Testlauf zulaessig (10.000 Ziehungen, Chunk 250 sind Teil der Festlegung)")
        sys.exit(2)
    global B_ZIEHUNGEN, CHUNK
    if args.b:
        B_ZIEHUNGEN = args.b
    if args.chunk:
        CHUNK = args.chunk
    STATE["test"] = bool(args.test)
    PFADE["out"].mkdir(parents=True, exist_ok=True)
    STATE["log"] = PFADE["out"] / "auswertung_aw221.log"
    skript = Path(__file__).resolve()
    sk_hash = sha256_datei(skript)
    log("bench_auswertung.py Version %s, SHA-256 %s, Python %s, numpy %s, pandas %s" % (VERSION, sk_hash, sys.version.split()[0], np.__version__, pd.__version__))
    if not args.test:
        ok = hashpruefung()
        df = schreibe_hashpruefung(PFADE["out"])
        log("Hashpruefung: %d Eintraege, %d bestanden" % (len(df), int(df.ok.sum())))
        if not ok:
            log("ABBRUCH: Hashpruefung nicht bestanden (V2 11.2 Punkt 3)")
            print(df[~df.ok].to_string())
            sys.exit(3)
    else:
        log("TESTLAUF: Hashpruefung uebersprungen; Ergebnisse sind keine Befunde")
    if args.nur_hash:
        return 0
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    kontrolle_k2()
    d = lade_alles()
    log("lade NeuroKit2-Schlagcluster")
    d.nk_rec = {}
    try:
        d.nk_rec = nk_parse(d.nk)
        log("NeuroKit2: %d Records geclustert" % len(d.nk_rec))
    except Exception as ex:  # noqa: BLE001
        d.lade_fehler.append("NeuroKit2 Cluster: %s" % ex)
        log("LADEFEHLER NeuroKit2-Cluster: %s | %s" % (ex, traceback.format_exc()))
    stufe("P0 (P1, Paare, Strata, Sensitivitaeten)", lambda: stufe_p0(d))
    stufe("Verstaerkungsarm PTB-XL (V1 bis V4)", lambda: stufe_verstaerkung(d, "ptbxl"))
    stufe("P1-Urteil, Saetze, MNAR", lambda: stufe_p1_urteil(d))
    stufe("Z2 Aggregationseffekt", lambda: stufe_z2(d))
    stufe("Verstaerkungsarm LUDB", lambda: stufe_verstaerkung(d, "ludb"))
    stufe("Z3 LUDB-Grenzfehler", lambda: stufe_z3(d))
    stufe("Filterarm PTB-XL", lambda: stufe_filterarm(d, "ptbxl"))
    stufe("Filterarm LUDB", lambda: stufe_filterarm(d, "ludb"))
    if STATE["stufen"] and STATE["stufen"][-1]["status"] == "ok":
        pass
    # Saetze erneut, falls Verstaerkung/Z2 nach dem Urteil fertig wurden
    stufe("Saetze (Wiederholung mit allen Stufen)", lambda: stufe_p1_urteil(d))
    schreibe_csv(pd.DataFrame(NICHT_GERECHNET, columns=["teil", "grund"]), "nicht_gerechnet.csv")
    schreibe_csv(pd.DataFrame(KONTROLLEN), "01_kontrollen.csv")
    schreibe_csv(pd.DataFrame(STATE["stufen"]), "02_stufen.csv")
    lauf = dict(skript=skript.name, skript_sha256=sk_hash, version=VERSION, start=t_start, ende=time.strftime("%Y-%m-%d %H:%M:%S"), seed_bootstrap=SEED_BOOT,
                ziehungen=B_ZIEHUNGEN, chunk=CHUNK, python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, test=bool(args.test),
                stufen=STATE["stufen"], kontrollen_nicht_bestanden=[k["kontrolle"] for k in KONTROLLEN if not k["bestanden"]],
                ladefehler=list(getattr(d, "lade_fehler", [])), speicher_spitze_gb=speicher_gb()[1])
    schreibe_text(json.dumps(lauf, indent=1, ensure_ascii=False), "LAUF.json")
    # Kopie der Aggregate (keine Einzelaufnahmen)
    if not args.test:
        ziel = PFADE["kopie"]
        ziel.mkdir(parents=True, exist_ok=True)
        for n in STATE["dateien"] + ["00_hashpruefung.csv", "auswertung_aw221.log"]:
            q = PFADE["out"] / n
            if q.exists():
                shutil.copy2(q, ziel / n)
        log("Aggregate nach %s kopiert" % ziel)
    fehler = [s for s in STATE["stufen"] if s["status"] != "ok"]
    zwischenstand("Lauf beendet %s. Stufen: %s. Fehlerstufen: %d." % (time.strftime("%H:%M:%S"), ", ".join("%s=%s" % (s["stufe"], s["status"]) for s in STATE["stufen"]), len(fehler)))
    log("FERTIG; Fehlerstufen: %d" % len(fehler))
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(haupt())
