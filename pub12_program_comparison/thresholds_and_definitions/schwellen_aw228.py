# -*- coding: utf-8 -*-
"""schwellen_aw228.py (AW-228, Teil C, Version 1.3): explorative Schwellenanalysen fuer PUB-12 (Etablierungsstueck).

Fragestellung (beschreibend, keine Hypothesentests): Wie oft ordnen zwei Messwerkzeuge dieselbe Person an klinisch benutzten
Schwellen verschieden zu? QRS-Dauer ab 120, 130, 150 ms; QTc ab 450, 470, 500 ms. Population P0 (je Person die niedrigste ecg_id,
18.869 Personen), native Ausgaben, gemeinsamer Nenner je Paar (beide Werkzeuge mit gueltigem Wert). Werkzeuge: LoopLens (LL),
BRAVEHEART (BH), 12SL und Uni-G sowie ECGDeli aus den PTB-XL+-Archivtabellen. NeuroKit2 nur auf S_G (1.000 Personen, Designgewichte),
nur QRS-Dauer (g1, g2, g3), so ausgewiesen.

QTc-Festlegung (vor dem Lauf): (1) "QTcF einheitlich": Fridericia, QT / RR(s)^(1/3), aus dem globalen QT des Werkzeugs und dessen
eigener RR-Quelle neu berechnet (LL: rr_intervall_ll; BH: 60000/hr; 12SL, Uni-G: 60000/HR__Global; ECGDeli: RR_Mean_Global).
(2) "QTcF nativ": das gelieferte Fridericia-QTc (LL qtcf, BH qtcf, 12SL und Uni-G QT_IntFridericia_Global; ECGDeli liefert keines).
Die Standard-QTc der Werkzeuge (12SL Bazett, Uni-G Hodge, ECGDeli Framingham) gehen nicht ein, weil ihre Formeln verschieden sind.

Statistik: Diskordanz = Anteil der Personen im gemeinsamen Nenner, bei denen genau ein Werkzeug die Schwelle erreicht (Wert >= Schwelle).
Bias = Mittel der Differenz Kandidat minus Komparator (LoopLens ist Kandidat, sonst das in ORDNUNG spaetere Werkzeug), Median des
Betrags der Differenz. Intervalle: Perzentil-Bootstrap (2,5 und 97,5 %) mit Personencluster, 2.000 Ziehungen, fester Seed, gemeinsame
Ziehungen fuer alle Einheiten; zufaelliger Nenner je Ziehung. S_G: Ziehung innerhalb der Strata, Designgewichte N_h / n_h.
Kleinzellenregel: Anzahlen von 1 bis 9 stehen in der veroeffentlichbaren Kopie als "<10"; wird eine der beiden Einseitig-Zellen
(nur A, nur B) maskiert, wird auch die andere maskiert (Komplement). Null bleibt Null.

Version 1.1 (15.10.2026) gegenueber 1.0 (SHA-256 e2c1622fdaac75235e462583810a639327a53921db5dc717833262368524b0d4): einzige Aenderung ist die
Toleranz des Bias in der Nachbaukontrolle K1 (0,0051 auf 0,0101 ms). Grund: Version 1.0 brach nach der Hashpruefung bei K1 ab, weil der Bericht
AUSWERTUNG_AW221_v2.md fuer ECGDeli gegen 12SL "+64,91" nennt, die Ergebnisdatei Z1_paare_P0_sekundaer.csv aber 64,9046 enthaelt (richtig gerundet 64,90);
mein Nachbau ergab 64,9046. Es waren zu diesem Zeitpunkt keine Bootstrap-Ergebnisse berechnet. Schaetzer, Ziehungen und Seeds sind unveraendert.

Version 1.2 (15.10.2026) gegenueber 1.1 (SHA-256 53c60e46cbc0f3eaacc22ba0d39e8a4ce02d3d3620377035db5343144c252eea): nur die Maskierung der
veroeffentlichbaren Kopie. Ist eine Einseitig-Zelle (nur A, nur B) maskiert, wird zusaetzlich n_disc nicht ausgewiesen und der Diskordanzanteil mit
Intervall auf eine Dezimalstelle gerundet, weil n_disc zusammen mit n_ge_a und n_ge_b die maskierte Zelle exakt zurueckrechnen liesse (Komplement).
Schaetzer, Ziehungen, Seeds und die unmaskierten Tabellen sind unveraendert (Bitvergleich im Bericht).

Version 1.3 (15.10.2026) gegenueber 1.2 (SHA-256 aa8a59aa412370670277cac8a83750e2635753a15451d150a176d88a411d8ce9): wieder nur die
veroeffentlichbare Kopie. Anlass: Die Kleinzellenpruefung (werkzeuge/kleinzellen/kleinzellen_scan.py) der Kopien aus Version 1.2 meldete Komplemente:
n_gemeinsam und Zaehlungen, die um 1 bis 9 vom Nenner oder von der Mengengroesse (18.869 bzw. 1.000) abweichen, liessen kleine Zellen (Personen ohne
Wert, Personen unter der Schwelle) exakt zurueckrechnen. Jetzt werden solche Zellen als "nicht ausgewiesen" (Komplement) bzw. "<10" geschrieben,
S3 und S4 gibt es nur noch maskiert in der Kopie, 01_kontrollen.csv wird nicht kopiert. Schaetzer, Ziehungen, Seeds und die unmaskierten Tabellen
sind unveraendert (Bitvergleich im Bericht). Die Aenderung erfolgte NACH Sicht der Ausgaben, ergebnisneutral und nur zur Maskierung.

Aufruf (Python der Auswertung, ohne Bytecode):
  <local path> -I -B schwellen_aw228.py --modus hash|lauf
Test (Kunstdaten):  ... -I -B schwellen_aw228_test.py
Das Skript liest gesperrte Tabellen nur und prueft vorher alle Hashes (Datensperre 15.10.2026, V2 11.2 Punkt 3).
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
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

VERSION = "1.3"
SEED = 15102228
SEED_SG = 15102229
B_ZIEHUNGEN = 2000
CHUNK = 250
SCHWELLEN = {"qrs": (120, 130, 150), "qtc": (450, 470, 500)}
ORDNUNG = ["12SL", "BRAVEHEART", "ECGDeli", "LoopLens", "NeuroKit2", "Uni-G"]
STRATUM_N_P0 = {"S1": 225, "S2": 1529, "S3": 1011, "S4": 8565, "S5": 7539}
KLEIN = 10

PFADE = {
    "erg": Path(str(data_path("pub12.benchmark_ergebnisse"))),
    "listen": Path(str(data_path("pub12.aw_219_listen"))),
    "meta": Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv"))),
    "plus": Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features"))),
    "wz": Path(str(data_path("pub12.werkzeuge_benchmark"))),
    "sperre": Path(str(data_path("pub12.benchmark_vorbereitung_20261015_datensperre_hashes_entwurf_json"))),
    "out": Path(str(data_path("pub12.benchmark_pub12_schwellen_aw228"))),
    "kopie": Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse"))),
    "v2skript": Path(str(data_path("pub12.werkzeuge_benchmark_bench_auswertung_v11_py"))),
}
SPERRE_LISTEN = {
    "sperre": "daab9d46395857e729f7c725d4e5fff2f1ab4e35a0b2e90e0333c20cd3f58c07",
    "manifest": "5b1511cc19de86b28cf84849c1007d1f54cf30362144f5df27854f1653372d07",
    "aw220": "986f17a154eba3904dfe83967564fc85f3ee1d8c7778f32edf0c176d61839230",
}
LISTEN_HASH = {
    "S_G_ecg_id.txt": "2f6fbc6d9fabd71d5a24bc81a16d45aaf577dd7315d41f633722d80b4b95a4c8",
    "S_G_stratum.csv": "1d6ee7894033151a0dc1c0a2805111d9054ba28b3f35132cfb6e86800e9e674b",
    "P0_ecg_id.txt": "d50ddf704b3fb6ef13c649c27baa0c470c9ba959faf9ed92b6eaf12e240a9e0a",
}
ARCHIV_HASH = {
    "12sl_features.csv": "cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29",
    "unig_features.csv": "f95ab01df5c9ed329485fd811578f0c39eabb0667a36ec37c7dda427d9d8f5a5",
    "ecgdeli_features.csv": "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556",
}
V2_SKRIPT_HASH = "53a50658c4a881f9eea17bf1bb0a03431a03d7c9e1bdf9cc46783a3fbea854b3"

STATE = {"log": None, "pruef": [], "kontrollen": []}


def log(msg):
    z = time.strftime("%H:%M:%S") + " " + str(msg)
    print(z, flush=True)
    if STATE["log"]:
        with open(STATE["log"], "a", encoding="utf-8") as f:
            f.write(z + "\n")


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# 1. Hilfsfunktionen
# ---------------------------------------------------------------------------------------------------------------
def orientiere(t1, t2):
    """Differenz = Kandidat minus Komparator; LoopLens ist Kandidat, sonst das in ORDNUNG spaetere Werkzeug (V2 3, Regel 3)."""
    if "LoopLens" in (t1, t2):
        return ("LoopLens", t2 if t1 == "LoopLens" else t1)
    a, b = sorted((t1, t2), key=ORDNUNG.index)
    return (b, a)


def fridericia(qt_ms, rr_ms):
    """QTcF in ms aus QT (ms) und RR (ms): QT / RR(s)^(1/3). Ungueltige RR (<= 0 oder nicht endlich) ergeben NaN."""
    qt = np.asarray(qt_ms, float)
    rr = np.asarray(rr_ms, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = qt / np.cbrt(rr / 1000.0)
    out = np.where(np.isfinite(rr) & (rr > 0) & np.isfinite(qt), out, np.nan)
    return out


def klein(n):
    return 1 <= int(n) <= KLEIN - 1


def maskiere_paar(n_a, n_b):
    """Kleinzellenregel fuer die beiden Einseitig-Zellen: ist eine in 1..9, werden beide nicht ausgewiesen. Rueckgabe (a, b) als Text."""
    if klein(n_a) or klein(n_b):
        return ("<10" if klein(n_a) else "nicht ausgewiesen", "<10" if klein(n_b) else "nicht ausgewiesen")
    return (str(int(n_a)), str(int(n_b)))


def maske_zaehl(n):
    return "<10" if klein(n) else str(int(n))


def gew_median_unten(werte_sortiert, gew_sortiert, gesamt):
    """Unterer gewichteter Median: kleinster Wert, bei dem die kumulierte Gewichtssumme mindestens die Haelfte der Gesamtsumme erreicht."""
    cs = np.cumsum(gew_sortiert)
    i = int(np.argmax(cs >= 0.5 * gesamt))
    return werte_sortiert[i]


# ---------------------------------------------------------------------------------------------------------------
# 2. Bootstrap-Maschine
# ---------------------------------------------------------------------------------------------------------------
class Ziehungen:
    """Zaehlmatrizen C (Ziehungen x Personen) je Chunk; Chunk c nutzt SeedSequence([seed, c]).
    strata=None: Personencluster ueber alle K Personen, multinomial(K, 1/K) (zufaelliger Nenner je Ziehung).
    strata=Array: Ziehung innerhalb der Strata (n_h aus n_h)."""

    def __init__(self, K, seed, B=B_ZIEHUNGEN, chunk=CHUNK, strata=None):
        self.K, self.seed, self.B, self.chunk, self.strata = K, seed, B, chunk, strata
        self.n_chunks = int(math.ceil(B / chunk))

    def chunk_c(self, c):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, c]))
        m = min(self.chunk, self.B - c * self.chunk)
        if self.strata is None:
            return rng.multinomial(self.K, np.full(self.K, 1.0 / self.K), size=m).astype(np.float64)
        C = np.zeros((m, self.K))
        for s in np.unique(self.strata):
            idx = np.flatnonzero(self.strata == s)
            C[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=m)
        return C


class Einheit:
    """Ein Paar (Kandidat a, Komparator b) auf einer Personenmenge. a, b, w, valid haben die Laenge der Menge (K bzw. n_SG)."""

    def __init__(self, name, a, b, w, art, schwellen):
        self.name = name
        a, b = np.asarray(a, float), np.asarray(b, float)
        self.valid = np.isfinite(a) & np.isfinite(b)
        self.v = np.flatnonzero(self.valid)
        self.a, self.b = a[self.v], b[self.v]
        self.w = np.asarray(w, float)[self.v]
        self.d = self.a - self.b
        self.absd = np.abs(self.d)
        self.order = np.argsort(self.absd, kind="stable")
        self.schwellen = tuple(schwellen)
        cols, self.namen = [], []
        for t in self.schwellen:
            pa, pb = self.a >= t, self.b >= t
            for nm, x in (("prev_a", pa), ("prev_b", pb), ("disc", pa ^ pb), ("nur_a", pa & ~pb), ("nur_b", ~pa & pb)):
                cols.append(x.astype(float))
                self.namen.append("%s_%d" % (nm, t))
        cols.append(self.d)
        self.namen.append("bias")
        self.X = np.column_stack(cols)

    def stat_W(self, W):
        """W: (Ziehungen x n_valid) Gewichte inkl. Zaehlmatrix. Rueckgabe dict Name -> Vektor der Laenge Ziehungen."""
        N = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            M = (W @ self.X) / N[:, None]
        out = {nm: M[:, i] for i, nm in enumerate(self.namen)}
        if len(self.v) == 0:
            out["medabs"] = np.full(W.shape[0], np.nan)
            return out
        Wo = W[:, self.order]
        cs = np.cumsum(Wo, axis=1)
        idx = (cs >= 0.5 * N[:, None]).argmax(axis=1)
        med = self.absd[self.order][idx]
        med = np.where(N > 0, med, np.nan)
        out["medabs"] = med
        return out

    def punkt(self):
        return {k: float(v[0]) for k, v in self.stat_W(self.w[None, :]).items()}

    def zaehlungen(self):
        z = {"n": int(len(self.v))}
        for t in self.schwellen:
            pa, pb = self.a >= t, self.b >= t
            z[t] = dict(ge_a=int(pa.sum()), ge_b=int(pb.sum()), nur_a=int((pa & ~pb).sum()), nur_b=int((~pa & pb).sum()),
                        disc=int((pa ^ pb).sum()))
        return z


def boot_lauf(einheiten, zieh, valid_zeilen_aus_C=lambda C, e: C[:, e.v]):
    """Alle Einheiten mit denselben Ziehungen. Rueckgabe: dict Einheitsname -> dict Statistik -> Array (B,)."""
    res = {e.name: {} for e in einheiten}
    for c in range(zieh.n_chunks):
        C = zieh.chunk_c(c)
        for e in einheiten:
            W = valid_zeilen_aus_C(C, e) * e.w[None, :]
            st = e.stat_W(W)
            for k, v in st.items():
                res[e.name].setdefault(k, []).append(v)
    return {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in res.items()}


def ki95(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# ---------------------------------------------------------------------------------------------------------------
# 3. Tabellen aus Einheiten
# ---------------------------------------------------------------------------------------------------------------
def tabelle_zeilen(e, pk, bs, zl, ta, tb, scope, messung, gewichtet):
    """Zeilen (eine je Schwelle) mit Punktschaetzer, KI und Zaehlungen; unmaskiert (intern)."""
    rows = []
    for t in e.schwellen:
        r = dict(scope=scope, messung=messung, kandidat=ta, komparator=tb, gewichtet=gewichtet, schwelle=t, n_gemeinsam=zl["n"])
        z = zl[t]
        for k in ("ge_a", "ge_b", "nur_a", "nur_b", "disc"):
            r["n_" + k] = z[k]
        for nm, key in (("anteil_ge_kandidat", "prev_a"), ("anteil_ge_komparator", "prev_b"), ("diskordanz", "disc"),
                        ("anteil_nur_kandidat", "nur_a"), ("anteil_nur_komparator", "nur_b")):
            k = "%s_%d" % (key, t)
            lo, hi = ki95(bs[k])
            r[nm] = 100.0 * pk[k]
            r[nm + "_ki_u"], r[nm + "_ki_o"] = 100.0 * lo, 100.0 * hi
        lo, hi = ki95(bs["bias"])
        r["bias_ms"], r["bias_ki_u"], r["bias_ki_o"] = pk["bias"], lo, hi
        lo, hi = ki95(bs["medabs"])
        r["median_betrag_ms"], r["median_betrag_ki_u"], r["median_betrag_ki_o"] = pk["medabs"], lo, hi
        rows.append(r)
    return rows


def klein_oder_komplement(v, n):
    """Zelle v (Anzahl) im Nenner n: klein, wenn v oder n - v in 1..9 liegt (Komplement exakt zurueckrechenbar)."""
    return klein(v) or klein(n - v)


def maskiere_tabelle(df, K_ref=None):
    """Veroeffentlichbare Kopie: Zaehlungen 1..9 als "<10", ebenso Zaehlungen, deren Komplement zum Nenner 1..9 betraegt;
    abhaengige Anteile und Intervalle "nicht ausgewiesen". K_ref: Groesse der Personenmenge (P0 18.869, S_G 1.000); weicht n_gemeinsam um 1..9
    davon ab, steht n_gemeinsam als "nicht ausgewiesen" (die Zahl der Personen ohne Wert waere sonst exakt rueckrechenbar)."""
    df = df.copy().astype(object)
    gruppen = {
        "n_ge_a": ["anteil_ge_kandidat"], "n_ge_b": ["anteil_ge_komparator"], "n_disc": ["diskordanz"],
        "n_nur_a": ["anteil_nur_kandidat"], "n_nur_b": ["anteil_nur_komparator"],
    }
    for i in df.index:
        n = int(df.at[i, "n_gemeinsam"])
        na, nb = int(df.at[i, "n_nur_a"]), int(df.at[i, "n_nur_b"])
        ma, mb = maskiere_paar(na, nb)
        if klein(n - na) or klein(n - nb):
            ma, mb = "<10" if klein(na) else "nicht ausgewiesen", "<10" if klein(nb) else "nicht ausgewiesen"
        df.at[i, "n_nur_a"], df.at[i, "n_nur_b"] = ma, mb
        if not ma.isdigit():
            for s in ("", "_ki_u", "_ki_o"):
                df.at[i, "anteil_nur_kandidat" + s] = "nicht ausgewiesen"
        if not mb.isdigit():
            for s in ("", "_ki_u", "_ki_o"):
                df.at[i, "anteil_nur_komparator" + s] = "nicht ausgewiesen"
        pair_masked = (not ma.isdigit()) or (not mb.isdigit())
        for cnt in ("n_ge_a", "n_ge_b", "n_disc"):
            v = int(df.at[i, cnt])
            if cnt == "n_disc" and pair_masked and not klein_oder_komplement(v, n):
                # Komplement: n_disc zusammen mit n_ge_a und n_ge_b liesse die maskierte Einseitig-Zelle exakt zurueckrechnen
                df.at[i, cnt] = "nicht ausgewiesen"
                for s in ("", "_ki_u", "_ki_o"):
                    df.at[i, "diskordanz" + s] = round(float(df.at[i, "diskordanz" + s]), 1)
                continue
            if klein_oder_komplement(v, n):
                df.at[i, cnt] = "<10" if klein(v) else "nicht ausgewiesen"
                for g in gruppen[cnt]:
                    for s in ("", "_ki_u", "_ki_o"):
                        df.at[i, g + s] = "nicht ausgewiesen"
            else:
                df.at[i, cnt] = str(v)
        df.at[i, "n_gemeinsam"] = ("nicht ausgewiesen" if (K_ref is not None and klein(K_ref - n)) else maske_zaehl(n))
    for c in df.columns:
        df[c] = [round(x, 3) if isinstance(x, (float, np.floating)) else x for x in df[c]]
    return df


def maskiere_verfuegbarkeit(df, K, spalten):
    """Veroeffentlichbare Kopie von Zaehltabellen: Werte 1..9 und Werte mit K - Wert in 1..9 stehen als "nicht ausgewiesen"."""
    df = df.copy().astype(object)
    for c in spalten:
        df[c] = [("nicht ausgewiesen" if (klein(int(v)) or klein(K - int(v))) else int(v)) for v in df[c]]
    return df


def matrix_md(df, messung, schwelle, scope, werkzeuge):
    """Kurzform der Diskordanzmatrix (Prozent, symmetrisch) fuer eine Messung und Schwelle."""
    s = df[(df.scope == scope) & (df.messung == messung) & (df.schwelle == schwelle)]
    M = {}
    for _, r in s.iterrows():
        M[(r.kandidat, r.komparator)] = r.diskordanz
        M[(r.komparator, r.kandidat)] = r.diskordanz
    kopf = "| | " + " | ".join(werkzeuge) + " |\n|---|" + "---:|" * len(werkzeuge) + "\n"
    zeilen = []
    for a in werkzeuge:
        zs = []
        for b in werkzeuge:
            if a == b:
                zs.append("-")
            else:
                v = M.get((a, b))
                zs.append("n/a" if v is None else (v if isinstance(v, str) else "%.1f" % v))
        zeilen.append("| %s | %s |" % (a, " | ".join(zs)))
    return kopf + "\n".join(zeilen) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# 4. Hashpruefung (Datensperre)
# ---------------------------------------------------------------------------------------------------------------
def _zeile(datei, soll, ist, ok, hinweis=""):
    STATE["pruef"].append(dict(datei=str(datei), soll=soll, ist=ist, ok=bool(ok), hinweis=hinweis))


def hashpruefung(gelesen):
    """Prueft die drei Hashlisten, danach jede Datei aus `gelesen` (absolute Pfade) gegen die Listen; Zusatzlisten, Archive, Metadaten.
    Jede gelesene Datei muss in einer Hashliste stehen oder einen eigenen Sollhash tragen. Rueckgabe True, wenn alles stimmt."""
    P = PFADE
    STATE["pruef"] = []
    listen = {"sperre": P["sperre"], "manifest": P["erg"] / "looplens/langformat/MANIFEST.json", "aw220": P["erg"] / "AW220_HASHES.txt"}
    soll = {}
    for k, p in listen.items():
        ist = sha256_datei(p) if Path(p).exists() else "FEHLT"
        _zeile(p, SPERRE_LISTEN[k], ist, ist == SPERRE_LISTEN[k], "Hashliste")
    j = json.loads(Path(listen["sperre"]).read_text(encoding="utf-8"))
    for pfad, v in j["dateien"].items():
        soll[os.path.normcase(os.path.abspath(pfad))] = v["sha256"]
    m = json.loads(Path(listen["manifest"]).read_text(encoding="utf-8"))
    for n, v in m["tabellen"].items():
        soll[os.path.normcase(os.path.abspath(Path(listen["manifest"]).parent / n))] = v["sha256"]
    for z in Path(listen["aw220"]).read_text(encoding="utf-8").splitlines():
        z = z.strip()
        if not z or z.startswith("#"):
            continue
        mt = re.match(r"^([0-9a-f]{64}) \*(.+)$", z)
        if mt and "." in mt.group(2) and not mt.group(2).endswith(".py") and not mt.group(2).endswith(".ps1"):
            soll[os.path.normcase(os.path.abspath(P["erg"] / mt.group(2)))] = mt.group(1)
    for n, s in LISTEN_HASH.items():
        soll[os.path.normcase(os.path.abspath(P["listen"] / n))] = s
    for n, s in ARCHIV_HASH.items():
        soll[os.path.normcase(os.path.abspath(P["plus"] / n))] = s
    try:
        for z in (Path(P["meta"]).parent / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            mt = re.match(r"^([0-9a-f]{64}) \*?(ptbxl_database\.csv)$", z.strip())
            if mt:
                soll[os.path.normcase(os.path.abspath(P["meta"]))] = mt.group(1)
    except Exception as ex:  # noqa: BLE001
        _zeile("PTB-XL SHA256SUMS", "lesbar", "Fehler: %s" % ex, False)
    for g in gelesen:
        key = os.path.normcase(os.path.abspath(g))
        if key not in soll:
            _zeile(g, "in einer Hashliste", "nicht gelistet", False, "gelesene Datei ohne Sollhash")
            continue
        ist = sha256_datei(g) if Path(g).exists() else "FEHLT"
        _zeile(g, soll[key], ist, ist == soll[key], "gelesene Datei")
    ist = sha256_datei(P["v2skript"]) if Path(P["v2skript"]).exists() else "FEHLT"
    _zeile(P["v2skript"], V2_SKRIPT_HASH, ist, ist == V2_SKRIPT_HASH, "V2-Skript 1.1 (nk_parse wird daraus genutzt)")
    return all(z["ok"] for z in STATE["pruef"])


def gelesene_dateien():
    P = PFADE
    L = P["erg"] / "looplens/langformat/looplens_langformat.csv.gz"
    return [L, P["erg"] / "ptbxl/ausgabe_langformat.csv", P["erg"] / "ptbxl/processed_ecgs_roh.csv.gz",
            P["erg"] / "neurokit2/langformat/neurokit2_schlaege.csv.gz",
            P["plus"] / "12sl_features.csv", P["plus"] / "unig_features.csv", P["plus"] / "ecgdeli_features.csv",
            P["listen"] / "P0_ecg_id.txt", P["listen"] / "S_G_ecg_id.txt", P["listen"] / "S_G_stratum.csv", P["meta"]]


# ---------------------------------------------------------------------------------------------------------------
# 5. Laden
# ---------------------------------------------------------------------------------------------------------------
def lese_ids(p):
    return np.array(sorted(int(x) for x in Path(p).read_text(encoding="ascii").split()), dtype=np.int64)


def lang_ok(df, key, ids):
    """Wert (nur Status ok und endlich) fuer die ids."""
    d = df[df.messgroesse == key]
    d = d[d.status == "ok"]
    v = pd.to_numeric(d["wert"], errors="coerce")
    s = pd.Series(v.to_numpy(), index=d["aufnahme"].astype(int).to_numpy())
    s = s[~s.index.duplicated(keep="first")]
    out = s.reindex(ids).to_numpy(float)
    return np.where(np.isfinite(out), out, np.nan)


def lade_alles():
    P = PFADE
    meta = pd.read_csv(P["meta"], usecols=["ecg_id", "patient_id"]).sort_values("ecg_id").reset_index(drop=True)
    erst = ~meta.duplicated("patient_id", keep="first")
    ids_p0 = meta.ecg_id.to_numpy(np.int64)[erst.to_numpy()]
    p0_liste = lese_ids(P["listen"] / "P0_ecg_id.txt")
    kontrolle("L1 P0 gleich niedrigste ecg_id je Person", "18869 gleich Liste", "%d %s" % (len(ids_p0), "gleich" if np.array_equal(ids_p0, p0_liste) else "verschieden"),
              bool(np.array_equal(ids_p0, p0_liste)) and len(ids_p0) == 18869)
    ids = ids_p0
    D = {}
    log("lade LoopLens")
    ll = pd.read_csv(P["erg"] / "looplens/langformat/looplens_langformat.csv.gz",
                     usecols=["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status"], dtype=str)
    ll = ll[(ll.datensatz == "ptbxl") & (ll.vorverarbeitung == "VV0") & ll.messgroesse.isin(["qrs_dauer_global", "qt_global", "qtcf", "rr_intervall_ll"])]
    D["LoopLens"] = dict(qrs=lang_ok(ll, "qrs_dauer_global", ids), qt=lang_ok(ll, "qt_global", ids), rr=lang_ok(ll, "rr_intervall_ll", ids),
                         qtc_nativ=lang_ok(ll, "qtcf", ids))
    del ll
    log("lade BRAVEHEART")
    bh = pd.read_csv(P["erg"] / "ptbxl/ausgabe_langformat.csv", usecols=["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status"], dtype=str)
    bh = bh[(bh.datensatz == "ptbxl") & (bh.vorverarbeitung == "VV0") & bh.messgroesse.isin(["qrs_dauer_global", "qt_global", "qtcf"])]
    roh = pd.read_csv(P["erg"] / "ptbxl/processed_ecgs_roh.csv.gz", usecols=["filename", "hr"])
    roh["ecg"] = roh.filename.map(lambda s: int(re.search(r"ptbxl_(\d+)", os.path.basename(str(s))).group(1)))
    roh = roh[~roh.ecg.duplicated(keep="first")].set_index("ecg")
    hr = roh["hr"].reindex(ids).to_numpy(float)
    rr_bh = np.where(np.isfinite(hr) & (hr > 0), 60000.0 / hr, np.nan)
    D["BRAVEHEART"] = dict(qrs=lang_ok(bh, "qrs_dauer_global", ids), qt=lang_ok(bh, "qt_global", ids), rr=rr_bh, qtc_nativ=lang_ok(bh, "qtcf", ids))
    del bh
    log("lade Archivprodukte")
    for tool, f, rr_art in (("12SL", "12sl_features.csv", "hr"), ("Uni-G", "unig_features.csv", "hr"), ("ECGDeli", "ecgdeli_features.csv", "rr")):
        cols = ["ecg_id", "QRS_Dur_Global", "QT_Int_Global"]
        if rr_art == "hr":
            cols += ["HR__Global", "QT_IntFridericia_Global"]
        else:
            cols += ["RR_Mean_Global"]
        df = pd.read_csv(P["plus"] / f, usecols=cols)
        df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id").reindex(ids)
        if rr_art == "hr":
            h = df["HR__Global"].to_numpy(float)
            rr = np.where(np.isfinite(h) & (h > 0), 60000.0 / h, np.nan)
            nat = df["QT_IntFridericia_Global"].to_numpy(float)
        else:
            rr = df["RR_Mean_Global"].to_numpy(float)
            nat = np.full(len(ids), np.nan)
        D[tool] = dict(qrs=df["QRS_Dur_Global"].to_numpy(float), qt=df["QT_Int_Global"].to_numpy(float), rr=rr, qtc_nativ=nat)
    for t, d in D.items():
        d["qtc_einh"] = fridericia(d["qt"], d["rr"])
    return ids, D


def lade_sg_nk(ids_p0):
    """S_G (1.000 Personen aus P0), Strata, Designgewichte und NeuroKit2 QRS g1 bis g3 (VV0) ueber nk_parse aus dem V2-Skript 1.1."""
    P = PFADE
    sg = lese_ids(P["listen"] / "S_G_ecg_id.txt")
    st = pd.read_csv(P["listen"] / "S_G_stratum.csv").set_index("ecg_id").stratum.reindex(sg).to_numpy()
    kontrolle("L2 S_G Teilmenge von P0", "alle", "%d von %d" % (int(np.isin(sg, ids_p0).sum()), len(sg)), bool(np.isin(sg, ids_p0).all()))
    n_sg = pd.Series(st).value_counts().to_dict()
    w = np.array([STRATUM_N_P0[s] / n_sg[s] for s in st], float)
    spec = importlib.util.spec_from_file_location("bench_v11", str(P["v2skript"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log("lade NeuroKit2")
    nk = pd.read_csv(P["erg"] / "neurokit2/langformat/neurokit2_schlaege.csv.gz", dtype={"datensatz": str, "vorverarbeitung": str, "ableitung": str})
    nk = nk[(nk.datensatz == "ptbxl") & (nk.vorverarbeitung == "VV0") & nk.aufnahme.isin(sg)]
    rec = mod.nk_parse(nk)
    del nk
    out = {}
    for g in ("QRS_g1", "QRS_g2", "QRS_g3"):
        v = np.full(len(sg), np.nan)
        for i, aid in enumerate(sg):
            r = rec.get(("ptbxl", "VV0", int(aid)))
            if r is not None:
                x = r["werte"].get(g, np.nan)
                if np.isfinite(x):
                    v[i] = x
        out[g] = v
    return sg, st, w, out


def kontrolle(name, soll, ist, ok, hinweis=""):
    STATE["kontrollen"].append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok), hinweis=hinweis))
    log("KONTROLLE %s: soll %s ist %s -> %s" % (name, soll, ist, "ok" if ok else "ABWEICHUNG"))


# ---------------------------------------------------------------------------------------------------------------
# 6. Lauf
# ---------------------------------------------------------------------------------------------------------------
def baue_einheiten_p0(ids, D):
    einheiten, meta = [], {}
    w = np.ones(len(ids))
    tools_qrs = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]
    spezi = [("qrs", "qrs", tools_qrs, SCHWELLEN["qrs"]), ("qtc_einh", "qtc_einh", tools_qrs, SCHWELLEN["qtc"]),
             ("qtc_nativ", "qtc_nativ", ["LoopLens", "BRAVEHEART", "12SL", "Uni-G"], SCHWELLEN["qtc"])]
    for messung, key, tools, schw in spezi:
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                ta, tb = orientiere(tools[i], tools[j])
                name = "P0|%s|%s|%s" % (messung, ta, tb)
                einheiten.append(Einheit(name, D[ta][key], D[tb][key], w, "zeit", schw))
                meta[name] = ("P0", messung, ta, tb, False)
    return einheiten, meta


def baue_einheiten_sg(ids, D, sg, w, nkw):
    pos = pd.Series(np.arange(len(ids)), index=ids)
    ix = pos[sg].to_numpy()
    einheiten, meta = [], {}
    for g, v in nkw.items():
        for t in ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]:
            ta, tb = orientiere("NeuroKit2", t)
            a = v if ta == "NeuroKit2" else D[ta]["qrs"][ix]
            b = v if tb == "NeuroKit2" else D[tb]["qrs"][ix]
            name = "SG|%s|%s|%s" % (g, ta, tb)
            einheiten.append(Einheit(name, a, b, w, "zeit", SCHWELLEN["qrs"]))
            meta[name] = ("S_G", g, ta, tb, True)
    return einheiten, meta


# Erwartete Werte aus AUSWERTUNG_AW221_v2.md Abschnitt 4 und 6 (P1 und Z1, Lauf mit Skriptversion 1.1) als Kontrolle des Nachbaus
Z1_ERWARTET = {
    ("LoopLens", "12SL"): (18780, 18.25, 19.6), ("LoopLens", "Uni-G"): (18778, 16.26, 19.0), ("LoopLens", "ECGDeli"): (18780, -46.69, 72.3),
    ("BRAVEHEART", "12SL"): (18663, -0.78, 2.1), ("Uni-G", "BRAVEHEART"): (18661, 2.74, 2.5), ("ECGDeli", "BRAVEHEART"): (18663, 65.65, 91.0),
    ("ECGDeli", "12SL"): (18869, 64.91, 90.6), ("Uni-G", "12SL"): (18867, 1.95, 2.5),
}


def kontrollen_nachbau(einheiten):
    byname = {e.name: e for e in einheiten}
    for (ta, tb), (n_soll, bias_soll, d120_soll) in Z1_ERWARTET.items():
        e = byname["P0|qrs|%s|%s" % (ta, tb)]
        pk = e.punkt()
        n = len(e.v)
        kontrolle("K1 Nachbau Z1 %s gegen %s: n, Bias, D120" % (ta, tb), "%d, %.2f, %.1f %%" % (n_soll, bias_soll, d120_soll),
                  "%d, %.2f, %.2f %%" % (n, pk["bias"], 100 * pk["disc_120"]),
                  n == n_soll and abs(pk["bias"] - bias_soll) <= 0.0101 and abs(100 * pk["disc_120"] - d120_soll) <= 0.051)
    e = byname["P0|qrs|LoopLens|BRAVEHEART"]
    pk = e.punkt()
    kontrolle("K2 Nachbau P1: n gemeinsam, Bias, D120 (LoopLens minus BRAVEHEART)", "18580, 19.00, 20.04 %",
              "%d, %.2f, %.2f %%" % (len(e.v), pk["bias"], 100 * pk["disc_120"]),
              len(e.v) == 18580 and abs(pk["bias"] - 19.00) <= 0.0051 and abs(100 * pk["disc_120"] - 20.04) <= 0.0051)


def schreibe_tab(df, name, mask=False, K_ref=None):
    P = PFADE
    P["out"].mkdir(parents=True, exist_ok=True)
    df.to_csv(P["out"] / name, index=False, encoding="utf-8", float_format="%.6g")
    if mask:
        m = maskiere_tabelle(df, K_ref)
        m.to_csv(P["out"] / name.replace(".csv", "_maskiert.csv"), index=False, encoding="utf-8", float_format="%.6g")
        P["kopie"].mkdir(parents=True, exist_ok=True)
        m.to_csv(P["kopie"] / name.replace(".csv", "_maskiert.csv"), index=False, encoding="utf-8", float_format="%.4g")
        return m
    return None


def lauf():
    P = PFADE
    P["out"].mkdir(parents=True, exist_ok=True)
    STATE["log"] = str(P["out"] / "schwellen_aw228.log")
    t0 = time.time()
    log("Start schwellen_aw228 Version %s, Skript-SHA-256 %s" % (VERSION, sha256_datei(__file__)))
    ok = hashpruefung(gelesene_dateien())
    pd.DataFrame(STATE["pruef"]).to_csv(P["out"] / "00_hashpruefung.csv", index=False, encoding="utf-8")
    log("Hashpruefung: %d Eintraege, alle stimmen: %s" % (len(STATE["pruef"]), ok))
    if not ok:
        log("ABBRUCH: Hashpruefung nicht bestanden (gesperrte Tabellen duerfen nicht gelesen werden)")
        return 2
    ids, D = lade_alles()
    einheiten, meta = baue_einheiten_p0(ids, D)
    kontrollen_nachbau(einheiten)
    sg, st, w_sg, nkw = lade_sg_nk(ids)
    eh_sg, meta_sg = baue_einheiten_sg(ids, D, sg, w_sg, nkw)
    meta.update(meta_sg)
    if not all(k["bestanden"] for k in STATE["kontrollen"]):
        pd.DataFrame(STATE["kontrollen"]).to_csv(P["out"] / "01_kontrollen.csv", index=False, encoding="utf-8")
        log("ABBRUCH: eine Kontrolle des Nachbaus ist nicht bestanden (vor der Bootstrap-Rechnung)")
        return 3
    # Rekonstruktionspruefung QTcF (Neuberechnung gegen gelieferte Werte)
    rk = []
    for t in ("LoopLens", "BRAVEHEART", "12SL", "Uni-G"):
        a, b = D[t]["qtc_einh"], D[t]["qtc_nativ"]
        m = np.isfinite(a) & np.isfinite(b)
        dd = (a - b)[m]
        rk.append(dict(werkzeug=t, n=int(m.sum()), median_diff_ms=float(np.median(dd)), p95_betrag_ms=float(np.percentile(np.abs(dd), 95)),
                       max_betrag_ms=float(np.max(np.abs(dd))), anteil_innerhalb_1ms=float(np.mean(np.abs(dd) <= 1.0)),
                       anteil_innerhalb_5ms=float(np.mean(np.abs(dd) <= 5.0))))
    pd.DataFrame(rk).to_csv(P["out"] / "S3_qtcf_rekonstruktion.csv", index=False, encoding="utf-8", float_format="%.6g")
    # Verfuegbarkeit je Werkzeug
    av = []
    for t in ("LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"):
        av.append(dict(werkzeug=t, n_qrs=int(np.isfinite(D[t]["qrs"]).sum()), n_qt=int(np.isfinite(D[t]["qt"]).sum()),
                       n_rr=int(np.isfinite(D[t]["rr"]).sum()), n_qtcf_einheitlich=int(np.isfinite(D[t]["qtc_einh"]).sum()),
                       n_qtcf_nativ=int(np.isfinite(D[t]["qtc_nativ"]).sum()), von=len(ids)))
    pd.DataFrame(av).to_csv(P["out"] / "S4_verfuegbarkeit.csv", index=False, encoding="utf-8")
    # Bootstrap
    log("Bootstrap P0: %d Einheiten, %d Ziehungen" % (len(einheiten), B_ZIEHUNGEN))
    zieh = Ziehungen(len(ids), SEED)
    bs = boot_lauf(einheiten, zieh)
    log("Bootstrap S_G: %d Einheiten" % len(eh_sg))
    zieh_sg = Ziehungen(len(sg), SEED_SG, strata=st)
    bs_sg = boot_lauf(eh_sg, zieh_sg)
    rows = []
    for e in einheiten + eh_sg:
        sc, messung, ta, tb, gew = meta[e.name]
        b = bs.get(e.name) or bs_sg[e.name]
        rows += tabelle_zeilen(e, e.punkt(), b, e.zaehlungen(), ta, tb, sc, messung, gew)
    df = pd.DataFrame(rows)
    p0 = df[df.scope == "P0"].reset_index(drop=True)
    s_g = df[df.scope == "S_G"].reset_index(drop=True)
    m0 = schreibe_tab(p0, "S1_paare_P0.csv", mask=True, K_ref=len(ids))
    ms = schreibe_tab(s_g, "S2_paare_SG_neurokit2.csv", mask=True, K_ref=len(sg))
    # Diskordanzmatrizen (maskierte Fassung)
    wz = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]
    txt = ["# Diskordanzmatrizen (Prozent der Personen im gemeinsamen Nenner; genau ein Werkzeug erreicht die Schwelle), AW-228, explorativ\n"]
    for messung, schw in (("qrs", SCHWELLEN["qrs"]), ("qtc_einh", SCHWELLEN["qtc"]), ("qtc_nativ", SCHWELLEN["qtc"])):
        wzm = wz if messung != "qtc_nativ" else wz[:4]
        for t in schw:
            txt.append("\n## %s ab %d ms, P0\n\n" % ({"qrs": "QRS-Dauer", "qtc_einh": "QTcF einheitlich (Fridericia neu berechnet)", "qtc_nativ": "QTcF nativ (geliefertes Fridericia)"}[messung], t))
            txt.append(matrix_md(m0, messung, t, "P0", wzm))
    (P["out"] / "S5_diskordanzmatrizen_P0.md").write_text("".join(txt), encoding="utf-8")
    P["kopie"].mkdir(parents=True, exist_ok=True)
    (P["kopie"] / "S5_diskordanzmatrizen_P0.md").write_text("".join(txt), encoding="utf-8")
    pd.DataFrame(STATE["kontrollen"]).to_csv(P["out"] / "01_kontrollen.csv", index=False, encoding="utf-8")
    # Veroeffentlichbare Kopien von S3 und S4 (Zaehlspalten mit Komplement unter 10 maskiert); 01_kontrollen.csv bleibt unter E: (enthaelt Nenner)
    s3 = pd.DataFrame(rk)
    maskiere_verfuegbarkeit(s3, len(ids), ["n"]).to_csv(P["kopie"] / "S3_qtcf_rekonstruktion_maskiert.csv", index=False, encoding="utf-8", float_format="%.6g")
    s4 = pd.DataFrame(av)
    maskiere_verfuegbarkeit(s4, len(ids), ["n_qrs", "n_qt", "n_rr", "n_qtcf_einheitlich", "n_qtcf_nativ"]).to_csv(
        P["kopie"] / "S4_verfuegbarkeit_maskiert.csv", index=False, encoding="utf-8")
    (P["kopie"] / "00_hashpruefung.csv").write_bytes((P["out"] / "00_hashpruefung.csv").read_bytes())
    lauf_info = dict(version=VERSION, skript_sha256=sha256_datei(__file__), seed=SEED, seed_sg=SEED_SG, ziehungen=B_ZIEHUNGEN, chunk=CHUNK,
                     start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"),
                     python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(STATE["pruef"]),
                     kontrollen=len(STATE["kontrollen"]), kontrollen_nicht_bestanden=sum(1 for k in STATE["kontrollen"] if not k["bestanden"]))
    (P["out"] / "LAUF.json").write_text(json.dumps(lauf_info, indent=1), encoding="utf-8")
    (P["kopie"] / "LAUF.json").write_text(json.dumps(lauf_info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["hash", "lauf"], required=True)
    a = ap.parse_args(argv)
    if a.modus == "hash":
        ok = hashpruefung(gelesene_dateien())
        for z in STATE["pruef"]:
            print(("OK   " if z["ok"] else "FEHL ") + str(z["datei"]))
        print("alle stimmen:", ok, "Eintraege:", len(STATE["pruef"]))
        return 0 if ok else 2
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
