"""AW-223: Externe Wiederholbarkeit fuer Paper A an PTB-XL-Paaren, Auswertungsskript (Version 1.1).

Version 1.1 (15.10.2026) gegenueber 1.0 (SHA-256 2fe6b9a108ed663708724f5ba14e164c4207fc366904a40b6730a5d87b8f617d,
Kopie aw223_auswertung_v1.0.py): nur Darstellung der Kleinzellenregel. maskiere_zellen beschriftete eine zusaetzlich
wegen eines Komplements zurueckgehaltene Zelle faelschlich mit "<10", obwohl sie 10 oder mehr betragen kann; sie heisst
jetzt "nicht ausgewiesen". Dazu klarere Bezeichnungen der Strata. Keine Aenderung an Schaetzern, Ziehungen, Seeds,
Populationen, Intervallen oder Klassen (Bitvergleich der Ergebnisdateien beider Laeufe im Bericht).

Genau nach der gebundenen Spezifikation SPEZIFIKATION_A_EXTERN_PTBXL_AW223_ENTWURF_v2.md
(SHA-256 421afdf307df3a91f9421a45a53fe28654a489ee47a315514843032afb7ed6b4, AE-280).

Modi
  --modus zaehlprobe : Paarbildung, Ausschluesse, Populationen, Statuszaehlung. Liest die Spalte wert NICHT.
  --modus lauf       : voller Lauf (liest wert aus den beiden gesperrten Langformaten).

Aufruf (Python der Auswertung, ohne Bytecode):
  <local path> -B aw223_auswertung.py --modus lauf

Die Schaetzer estimate und moments stammen unveraendert aus werkzeuge/rv_a1/wiederholbarkeit.py
(SHA-256 77485bcd...9bf49): das Skript prueft den Hash, entnimmt die beiden Funktionen per AST und
ruft sie auf; das Modul selbst wird nicht importiert (sein Import legt Dateien im rv_a1-Ausgabeordner an).
Kein Messwert wird vor der Hashpruefung der Eingaben und des Schritt-0-Ergebnisses gelesen.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ast
import datetime as dt
import gzip
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

VERSION = "1.1"
NBOOT = 2000
MIN_GUELTIG = 1900
MIN_PERSONEN_SCHAETZBAR = 30
MIN_PERSONEN_KLASSE = 100
BAND_LO, BAND_HI = 0.80, 1.25
Z_SEED_BASIS = 223910
KLEIN = 10  # Zellen 1 bis 9 -> "<10"

AUSW = Path(str(data_path("pub12.auswertung")))
SPEC = (AUSW / "nachrechnung" / "SPEZIFIKATION_A_EXTERN_PTBXL_AW223_ENTWURF_v2.md",
        "421afdf307df3a91f9421a45a53fe28654a489ee47a315514843032afb7ed6b4")
STEP0_MD = (AUSW / "nachrechnung" / "A_EXTERN_PTBXL_SCHRITT0_AW223.md",
            "34908ec202ea0f4251edad25afc917f6c81c5a39e023dc6788afb2570781fb6f")
STEP0_JSON = (AUSW / "nachrechnung" / "a_extern_ptbxl_aw223" / "schritt0_ergebnis.json",
              "1dbed012b9d5d096c35c5f01d5eb479e95b0bd4b7903dedaaa1de24aea1beafb")
RV = (AUSW / "werkzeuge" / "rv_a1" / "wiederholbarkeit.py",
      "77485bcdfdcb926e908a48b59f0b00f5ae308275ed6733260c138392e9c7bf49")
SPERRLISTE = (Path(str(data_path("pub12.benchmark_vorbereitung_20261015_datensperre_hashes_entwurf_json"))),
              "daab9d46395857e729f7c725d4e5fff2f1ab4e35a0b2e90e0333c20cd3f58c07")
EINGABEN = {
    "langformat_looplens": (Path(str(data_path("pub12.benchmark_ergebnisse_looplens_langformat_looplens_langformat_csv_gz"))),
                            "fab08057b3e411e1ae3969143bfc732f8854986c50eb13b684f67af65b6bc680"),
    "langformat_braveheart": (Path(str(data_path("pub12.benchmark_ergebnisse_ptbxl_ausgabe_langformat_csv"))),
                              "0bf1a8474a74f5447da67faba52a551d59a4dc2b1c7a311d4da01ed9fbddc9ed"),
    "metadaten_ptbxl": (Path(str(data_path("pub12.research_diagnostic_identifiability_20261013_ptb_xl_1_0_3_ptbxl_database_csv"))),
                        "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"),
    "paarliste_stunden": (Path(str(data_path("pub12.aw_223_aw223_paarliste_stunden_csv"))),
                          "b8ed58b509f6d72d35c983ad9b303234d947389b80d1a20e54f08e6d8bf13e0b"),
    "skript_paarliste": (AUSW / "werkzeuge" / "benchmark" / "aw223_paarliste.py",
                         "68d7a79dbaae0deca4dc1abb04806f4eb4bc24ac6d9bb52d7a50a58b8d16e888"),
    "paarliste_ae271": (Path(str(data_path("pub12.benchmark_vorbereitung_20261015_ae271_ptbxl_paare_csv"))),
                        "94c251a91eaa9aad7d1f5a1ab0914c159f5a97b46fc8e705953274d1620a2727"),
    "pilotliste_pt": (Path(str(data_path("pub12.aw_219_listen_pt_ecg_id_txt"))),
                      "ef7f6c018091686cb17c5a714a613a6e0594c67093872367f284f8c026156224"),
    "zielwerte_a": (AUSW / "publikationen" / "PUB-01_messvaliditaet" / "preprint" / "medrxiv_upload_2026-10-14" /
                    "supplementary_data_S2_marker_reliability_public_v3.csv",
                    "fef2842d1ed8c33f3f8f55013a0d51848c85858179abc433478c45df6ab8ef8f"),
}
PTBXL_ROOT = Path(str(data_path("pub12.samples_ptb_x")))
OUT_E = Path(str(data_path("pub12.aw_223_ergebnis")))
OUT_INTERN = Path(str(data_path("pub12.aw_223_intern")))
OUT_PUB = AUSW / "nachrechnung" / "a_extern_ptbxl_aw223"

# (k, LoopLens-Schluessel, A-Kennung, Teiler fuer A-Werte, Rolle, erwartete Einheit im Langformat)
MARKER = [
    (0, "qrs_dauer_global", "qrs_duration_ms", 1.0, "primaer", "ms"),
    (1, "qtcf", "qtc_fridericia_ms", 1.0, "primaer", "ms"),
    (2, "qrst_winkel_integral_ll", "spatial_qrs_t_angle_deg", 1.0, "primaer_bedingt", "grad"),
    (3, "p_dauer", "p_duration_ms", 1.0, "sekundaer", "ms"),
    (4, "pr_intervall", "pr_interval_ms", 1.0, "sekundaer", "ms"),
    (5, "qt_global", "qt_interval_ms", 1.0, "sekundaer", "ms"),
    (6, "qrst_winkel_peak_ll", "spatial_peak_qrs_t_angle_deg", 1.0, "sekundaer", "grad"),
    (7, "vektorbetrag_max_qrs_ll", "qrs_vector_magnitude", 1000.0, "beschreibend", "mv"),
    (8, "vektorbetrag_max_t_ll", "t_vector_magnitude", 1000.0, "beschreibend", "mv"),
    (9, "integralbetrag_qrs_ll", "qrs_area_vector_magnitude_mv_ms", 1.0, "beschreibend", "mv*ms"),
    (10, "sai_qrst_ll", "sum_absolute_qrst_integral_mv_ms", 1.0, "beschreibend", "mv*ms"),
    (11, "schleifenlaenge_qrs_ll", "qrs_loop_path_length", 1.0, "beschreibend", "mv"),
    (12, "svg_betrag_ll", "svg_magnitude", 1000.0, "beschreibend", "mv*ms"),
]
MARKER_NAMEN = {
    "qrs_dauer_global": "QRS-Dauer", "qtcf": "QTcF", "qrst_winkel_integral_ll": "Flächenwinkel QRS-T",
    "p_dauer": "P-Dauer", "pr_intervall": "PR-Intervall", "qt_global": "QT-Intervall",
    "qrst_winkel_peak_ll": "Spitzenwinkel QRS-T", "vektorbetrag_max_qrs_ll": "Vektorbetrag QRS",
    "vektorbetrag_max_t_ll": "Vektorbetrag T", "integralbetrag_qrs_ll": "Betrag der QRS-Fläche",
    "sai_qrst_ll": "SAI QRS-T", "schleifenlaenge_qrs_ll": "QRS-Schleifenlänge", "svg_betrag_ll": "SVG-Betrag",
}
RR_KEY = "rr_intervall_ll"
LL_KEYS = [m[1] for m in MARKER]
LL_ALLE = LL_KEYS + [RR_KEY]
EINHEITEN_ERWARTET = {m[1]: m[5] for m in MARKER}
EINHEITEN_ERWARTET[RR_KEY] = "ms"
# BRAVEHEART-Schluessel (Langformat) -> LoopLens-Schluessel (Spalte der Wertematrix)
BH_ZU_LL = {
    "qrs_dauer_global": "qrs_dauer_global", "qtcf": "qtcf", "qrst_winkel_integral": "qrst_winkel_integral_ll",
    "qt_global": "qt_global", "qrst_winkel_peak": "qrst_winkel_peak_ll",
    "vektorbetrag_max_qrs": "vektorbetrag_max_qrs_ll", "integralbetrag_qrs": "integralbetrag_qrs_ll",
    "sai_qrst": "sai_qrst_ll", "schleifenlaenge_qrs": "schleifenlaenge_qrs_ll",
}
BH_MARKER = list(BH_ZU_LL.values())
# Zielwerte aus Spezifikation 7.2 (Kontrolle gegen die Datei, nicht Rechengrundlage)
SPEC_RC_A = [27.27, 61.78, 82.41, 47.84, 66.71, 76.85, 91.39, 0.5420, 0.4146, 44.85, 178.54, 1.5548, reference("aw223_auswertung.spec_rc_marker13")]
SPEC_SIGMA_A = [0.0273, 0.0186, 0.0095, 0.0334, 0.0217, 0.0130, 0.0102, 0.0388, 0.0725, 0.0596, 0.1044, 0.0929, 0.1056]
SCHWELLEN = {"qrst_winkel_integral_ll": [(">", 90.0), (">", 105.0), (">", 120.0)], "p_dauer": [(">=", 120.0)]}

# Analyse: Nummer a (Seed der A-Ziehung), Seed der Personenziehung, Klasse erlaubt
ANALYSEN = {
    "PRIM": (0, 223010, True), "S1": (1, 223011, True), "S2": (2, 223012, True), "S3": (3, 223013, True),
    "S4": (4, 223014, True), "S5": (5, 223015, False), "S6": (6, 223016, True), "S7": (7, 223017, True),
    "S8": (8, 223018, True), "S9": (9, 223019, False), "S10": (10, 223020, False), "S11": (11, 223021, False),
    "S12": (12, 223022, False), "S13": (13, 223023, False),
    "Z1": (21, 223030, False), "Z2": (22, 223031, False), "Z3": (23, 223032, False),
}
# Planungszaehlungen aus Spezifikation 3.2, 3.3, 9: (Paare, Personen), None = nicht angegeben
PLANUNG = reference("aw223_auswertung_v1.1.planung", {})
PLANUNG_MERKMALE = reference("aw223_auswertung_v1.1.planung_merkmale",
                              {k: ["k. A.", "k. A."] for k in ('rauschvermerk_mit', 'rauschvermerk_ohne', 'schrittmacher_mit', 'schrittmacher_ohne', 'geraet_ort_gleich', 'geraet_ort_verschieden', 'proxy_analogon')})
PLANUNG_SCHRITTE = {int(k): v for k, v in reference("aw223_auswertung_v1.1.planung_schritte",
                                                   {"24": [["k. A.", "k. A."]] * 3, "168": [["k. A.", "k. A."]] * 3}).items()}

VERBOTENE_STAEMME = ["bestätig", "replizier", "validier", "widerleg", "überschätz", "unterschätz", "falsch",
                     "stabiler", "genauer"]
LANGE_STRICHE = ("\u2014", "\u2013")

PFLICHTZUSATZ = ("Die Bandlage nennt keine Ursache: Abstand bis 24 h (biologische Variation über bis zu einem Tag, "
                 "in A mit 0 bis 2 h kaum enthalten), Paarkriterium, fehlende Setting- und Prozedurbereinigung, "
                 "Population (in A selbst QRS-RC 27,27 ms für alle Paare gegen 15,99 ms im Proxy), Geräte und "
                 "Technik sind hier nicht getrennt; beide Quellen sind mit LoopLens gemessen.")


# ------------------------------------------------------------------ Hilfen
def sha_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pruefe_sprache(text, quelle=""):
    """Wirft, wenn ein verbotenes Wort oder ein langer Gedankenstrich im Text steht."""
    low = text.lower()
    for s in VERBOTENE_STAEMME:
        if s in low:
            i = low.index(s)
            raise ValueError(f"Verbotenes Wort '{s}' in {quelle}: ...{text[max(0, i - 40):i + 40]}...")
    for c in LANGE_STRICHE:
        if c in text:
            raise ValueError(f"Langer Gedankenstrich in {quelle}")
    return True


def zelle(n):
    """Kleinzellenregel: 1 bis 9 -> '<10'."""
    n = int(n)
    return "<10" if 1 <= n <= 9 else str(n)


def maskiere_zellen(zahlen):
    """Zellen einer Aufteilung mit bekannter Summe: 1 bis 9 -> '<10'; ist genau eine Zelle so verdeckt, wird
    zusaetzlich die kleinste uebrige positive Zelle zurueckgehalten und als 'nicht ausgewiesen' beschriftet (nicht als
    '<10', denn sie kann 10 oder mehr betragen); sonst waere die kleine Zelle per Differenz zur Summe ableitbar."""
    z = [int(v) for v in zahlen]
    klein = [1 <= v <= 9 for v in z]
    zusatz = [False] * len(z)
    if sum(klein) == 1:
        rest = [(v, i) for i, v in enumerate(z) if not klein[i] and v > 0]
        if rest:
            zusatz[min(rest)[1]] = True
    return ["<10" if k else ("nicht ausgewiesen" if zs else str(v)) for v, k, zs in zip(z, klein, zusatz)]


def fz(x, d=2):
    """Zahl mit Dezimalkomma."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "k. A."
    return f"{x:.{d}f}".replace(".", ",")


def fzi(lo, hi, d=2):
    return f"[{fz(lo, d)}; {fz(hi, d)}]"


# ------------------------------------------------------------------ Hash- und Bindungspruefung
def pruefe_bindung(nur_sperre_und_papiere=False):
    """Prueft Spezifikation, Schritt 0, Rechenvertrag, Sperrliste und alle Eingaben. Gibt Hash-Protokoll zurueck."""
    prot = {}
    for name, (pfad, soll) in (("spezifikation", SPEC), ("schritt0_md", STEP0_MD), ("schritt0_json", STEP0_JSON),
                               ("rechenvertrag_rv_a1", RV), ("sperrliste", SPERRLISTE)):
        ist = sha_datei(pfad)
        if ist != soll:
            raise RuntimeError(f"Hash weicht ab: {name} {pfad} soll {soll} ist {ist}")
        prot[name] = {"pfad": str(pfad), "sha256": ist, "gleich": True}
    sperre = json.loads(Path(SPERRLISTE[0]).read_text(encoding="utf8"))["dateien"]
    for name, (pfad, soll) in EINGABEN.items():
        ist = sha_datei(pfad)
        if ist != soll:
            raise RuntimeError(f"Hash weicht ab: {name} {pfad} soll {soll} ist {ist}")
        eintrag = {"pfad": str(pfad), "sha256": ist, "gleich": True}
        key = str(pfad).replace("\\", "/")
        if key in sperre:
            if sperre[key]["sha256"] != ist:
                raise RuntimeError(f"Datensperre weicht ab: {key}")
            eintrag["in_sperrliste_gleich"] = True
        prot[name] = eintrag
    return prot


def lade_schritt0():
    d = json.loads(Path(STEP0_JSON[0]).read_text(encoding="utf8"))
    return {k: v["urteil"] for k, v in d["marker"].items()}


# ------------------------------------------------------------------ Rechenvertrag rv_a1
def lade_rv(nboot=NBOOT):
    """estimate und moments unveraendert aus wiederholbarkeit.py (Hash geprueft), per AST ohne Modulimport."""
    pfad, soll = RV
    raw = Path(pfad).read_bytes()
    if hashlib.sha256(raw).hexdigest() != soll:
        raise RuntimeError("rv_a1/wiederholbarkeit.py weicht vom gebundenen Hash ab")
    tree = ast.parse(raw.decode("utf8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("estimate", "moments")]
    if sorted(f.name for f in fns) != ["estimate", "moments"]:
        raise RuntimeError("estimate/moments nicht gefunden")
    ns = {"np": np, "B": nboot}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(pfad), "exec"), ns)
    return ns["estimate"], ns["moments"]


def icc_unabhaengig(x, y):
    """ICC(A,1) (McGraw und Wong 1996) direkt aus der Zwei-Faktoren-Varianzanalyse der Rohwerte, gleichgewichtete
    Paare, ohne normalisierte Momente. Unabhaengiger Weg fuer die Gegenprobe V4."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    k = 2
    data = np.stack([x, y], axis=1)
    gm = data.mean()
    rm = data.mean(1)
    cm = data.mean(0)
    msr = k * np.sum((rm - gm) ** 2) / (n - 1)
    msc = n * np.sum((cm - gm) ** 2) / (k - 1)
    sse = np.sum((data - rm[:, None] - cm[None, :] + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def selbsttest_rv(moments_fn, log=print):
    """Gegenprobe V4 an einem synthetischen Paarsatz mit bekanntem ICC: moments (rv_a1) gegen die unabhaengige
    Formel. Bricht bei Abweichung ab."""
    rng = np.random.default_rng(20261015)
    n, sb, sw = 1500, 10.0, 4.0
    t = rng.normal(100.0, sb, n)
    x = t + rng.normal(0.0, sw, n)
    y = t + rng.normal(0.0, sw, n) + 1.5
    ids = np.arange(n, dtype=np.int64)
    with np.errstate(all="ignore"):
        point, _boot, npair, nper, bsd, den, scale = moments_fn(x, y, ids, ids, None)
    icc_ref = icc_unabhaengig(x, y)
    d = y - x
    rc_ref = 1.96 * np.std(d, ddof=1)
    bias_ref = float(d.mean())
    ok = (abs(point[0, 0] - icc_ref) < 1e-9 and abs(point[0, 3] - rc_ref) < 1e-9 * max(1.0, rc_ref)
          and abs(point[0, 1] - bias_ref) < 1e-9)
    log(f"Selbsttest rv_a1: ICC {point[0, 0]:.9f} gegen unabhaengig {icc_ref:.9f}; RC {point[0, 3]:.9f} gegen {rc_ref:.9f}; "
        f"Bias {point[0, 1]:.9f} gegen {bias_ref:.9f}; wahres ICC {sb ** 2 / (sb ** 2 + sw ** 2):.4f}; {'ok' if ok else 'ABWEICHUNG'}")
    if not ok:
        raise RuntimeError("Gegenprobe V4 fehlgeschlagen")
    return True


def ci_perzentil(v, mind=MIN_GUELTIG):
    """95-%-Perzentilintervall (2,5 / 97,5, lineare Interpolation); verlangt mindestens mind endliche Ziehungen."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < mind:
        return float("nan"), float("nan"), len(v)
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(lo), float(hi), len(v)


def gewichte(n_personen, seed, nboot=NBOOT):
    """Personengewichte Multinomial(n; 1/n) aus default_rng(seed); Personen aufsteigend nach patient_id."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_personen, np.full(n_personen, 1.0 / n_personen), size=nboot)


def z_ziehung(a, k, nboot=NBOOT):
    return np.random.default_rng(np.random.SeedSequence([Z_SEED_BASIS, int(a), int(k)])).standard_normal(nboot)


def kombiniertes_R(rc_boot, rc_pt, rc_a, lo_a, hi_a, a, k):
    """R = RC_PTB / RC_A; kombiniertes Intervall (PTB-Bootstrap und lognormale A-Unsicherheit) und nur-PTB-Intervall."""
    sigma = (math.log(hi_a) - math.log(lo_a)) / (2 * 1.96)
    z = z_ziehung(a, k, len(rc_boot))
    rb = rc_boot / (rc_a * np.exp(sigma * z))
    lo, hi, n = ci_perzentil(rb)
    lo_p, hi_p, _ = ci_perzentil(rc_boot / rc_a)
    # Monte-Carlo-Fehler: Grenzen aus Ziehung 1-1000 gegen 1001-2000
    h = len(rb) // 2
    a1 = ci_perzentil(rb[:h], mind=int(0.95 * h))
    a2 = ci_perzentil(rb[h:], mind=int(0.95 * h))
    return {"R": rc_pt / rc_a, "R_lo": lo, "R_hi": hi, "R_n_gueltig": n, "R_lo_ptb": lo_p, "R_hi_ptb": hi_p,
            "sigma_a": sigma, "mc_R_lo": abs(a1[0] - a2[0]), "mc_R_hi": abs(a1[1] - a2[1])}


def klassifiziere(lo, hi, n_personen, n_gueltig, rolle_eff, klasse_erlaubt=True):
    """Klasse nach 7.3. rolle_eff in primaer, sekundaer, beschreibend."""
    if rolle_eff not in ("primaer", "sekundaer") or not klasse_erlaubt:
        return "beschreibend (keine Klasse)"
    if (not (n_personen >= MIN_PERSONEN_KLASSE)) or n_gueltig < MIN_GUELTIG or not (np.isfinite(lo) and np.isfinite(hi)):
        return "nicht entscheidbar (Nenner)"
    if lo >= BAND_LO and hi <= BAND_HI:
        return "übereinstimmend"
    if lo > BAND_HI:
        return "höher"
    if hi < BAND_LO:
        return "niedriger"
    return "unentschieden"


def erlaubter_satz(klasse, name, R, lo, hi, n_personen=None, sensitivitaet=False, fenster="Paare bis 24 h"):
    pre = "Sensitivität: " if sensitivitaet else ""
    if klasse == "übereinstimmend":
        s = (f"Der RC von {name} liegt in PTB-XL ({fenster}) im Band 0,80 bis 1,25 des in A berichteten RC "
             f"(R = {fz(R)}, 95 % {fzi(lo, hi)}).")
    elif klasse == "höher":
        s = f"Der RC liegt in PTB-XL oberhalb dieses Bandes (R = {fz(R)} {fzi(lo, hi)})."
    elif klasse == "niedriger":
        s = f"Der RC liegt in PTB-XL unterhalb dieses Bandes (R = {fz(R)} {fzi(lo, hi)})."
    elif klasse == "unentschieden":
        s = "Das Intervall von R überlappt die Bandgrenze; Übereinstimmung ist weder belegt noch ausgeschlossen."
    elif klasse.startswith("nicht entscheidbar"):
        s = f"Nicht entscheidbar bei n = {n_personen} Personen." if n_personen is not None else "Nicht entscheidbar."
    else:
        s = f"Beschreibend: R = {fz(R)} {fzi(lo, hi)}, ohne Bandklasse."
    s = pre + s
    pruefe_sprache(s, "erlaubter_satz")
    return s


def rolle_effektiv(rolle, key, step0):
    if rolle == "primaer":
        return "primaer"
    if rolle == "primaer_bedingt":
        return "primaer" if step0.get(key) == "gleich" else "beschreibend"
    if rolle == "sekundaer":
        return "sekundaer" if step0.get(key) == "gleich" else "beschreibend"
    return "beschreibend"


# ------------------------------------------------------------------ Daten laden
def lade_zielwerte(pfad=None):
    pfad = pfad or EINGABEN["zielwerte_a"][0]
    d = pd.read_csv(pfad).set_index("marker_identifier")
    z = {}
    for k, key, aid, div, rolle, unit in MARKER:
        r = d.loc[aid]
        z[key] = {
            "k": k, "a_id": aid, "div": div, "unit_a": str(r["unit"]),
            "rc": float(r["all_pairs_repeatability_coefficient"]) / div,
            "rc_lo": float(r["all_pairs_repeatability_coefficient_confidence_limit_lower"]) / div,
            "rc_hi": float(r["all_pairs_repeatability_coefficient_confidence_limit_upper"]) / div,
            "icc": float(r["all_pairs_intraclass_correlation"]),
            "icc_lo": float(r["all_pairs_intraclass_correlation_confidence_limit_lower"]),
            "icc_hi": float(r["all_pairs_intraclass_correlation_confidence_limit_upper"]),
            "bias": float(r["all_pairs_bias"]) / div,
            "paare": int(float(r["all_pairs_finite_pairs"])), "personen": int(float(r["all_pairs_finite_persons"])),
            "rc_proxy": float(r["stability_proxy_repeatability_coefficient"]) / div,
            "rc_proxy_lo": float(r["stability_proxy_repeatability_coefficient_confidence_limit_lower"]) / div,
            "rc_proxy_hi": float(r["stability_proxy_repeatability_coefficient_confidence_limit_upper"]) / div,
            "icc_proxy": float(r["stability_proxy_intraclass_correlation"]),
            "bias_proxy": float(r["stability_proxy_bias"]) / div,
        }
    return z


def kontrolle_zielwerte(z):
    """Vergleich mit der Tabelle 7.2 der Spezifikation: Abweichung ueber 1 % beim RC bricht ab, sigma wird gemeldet."""
    meldungen = []
    for (k, key, *_), rc_spec, s_spec in zip(MARKER, SPEC_RC_A, SPEC_SIGMA_A):
        t = z[key]
        rc_reference = rc_spec
        if rc_reference is None:
            print("NOTE: no local reference value for the target RC of " + key + "; that check is skipped.")
        if rc_reference is not None and abs(t["rc"] - rc_spec) / rc_spec > 0.01:
            raise RuntimeError(f"Zielwert RC weicht von Spezifikation 7.2 ab: {key} {t['rc']} gegen {rc_spec}")
        sigma = (math.log(t["rc_hi"]) - math.log(t["rc_lo"])) / (2 * 1.96)
        if abs(sigma - s_spec) > 6e-5:
            meldungen.append(f"sigma_A {key}: Datei {sigma:.5f} gegen Spezifikation {s_spec}")
    return meldungen


def lade_metadaten(pfad=None):
    pfad = pfad or EINGABEN["metadaten_ptbxl"][0]
    cols = ["ecg_id", "patient_id", "age", "sex", "site", "device", "recording_date", "scp_codes",
            "baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker"]
    m = pd.read_csv(pfad, usecols=cols)
    m["ecg_id"] = m.ecg_id.astype(int)
    m = m.set_index("ecg_id").sort_index()

    def leer(c):
        return m[c].fillna("").astype(str).str.strip() == ""

    keys = m.scp_codes.map(lambda s: set(ast.literal_eval(s).keys()) if isinstance(s, str) else set())
    m["rausch"] = ~(leer("baseline_drift") & leer("static_noise") & leer("burst_noise") & leer("electrodes_problems"))
    m["schrittmacher"] = (~leer("pacemaker")) | keys.map(lambda s: "PACE" in s)
    m["sr"] = keys.map(lambda s: "SR" in s)
    m["norm"] = keys.map(lambda s: "NORM" in s)
    m["block"] = keys.map(lambda s: bool(s & {"CLBBB", "CRBBB", "ILBBB", "IRBBB"}))
    m["device"] = m.device.fillna("").astype(str)
    m["site"] = m.site.fillna(-1)
    m["devgrp"] = m.device.map(lambda s: s.split()[0] if s.split() else "")
    m["alter_flag300"] = m.age == 300
    m["alter"] = m.age.where(m.age != 300, 90)
    return m


def lade_paarliste(pfad=None):
    pfad = pfad or EINGABEN["paarliste_stunden"][0]
    p = pd.read_csv(pfad)
    p["patient_id"] = p.patient_id.round().astype(int)
    p["ecg_id_1"] = p.ecg_id_1.astype(int)
    p["ecg_id_2"] = p.ecg_id_2.astype(int)
    for c in ("qrs_ok_ll", "qrs_ok_bh", "pt_paar"):
        p[c] = p[c].astype(str).str.lower().eq("true")
    return p


def signalhashes(ids, cache, log=print):
    """SHA-256 der dekodierten Signale (float64 der physikalischen Werte) aus den Originaldateien (records500)."""
    import wfdb
    cache = Path(cache)
    have = {}
    if cache.exists():
        d = pd.read_csv(cache, dtype={"sha256": str})
        have = dict(zip(d.ecg_id.astype(int), d.sha256))
    sums = {}
    for line in (PTBXL_ROOT / "SHA256SUMS.txt").read_text().splitlines():
        t = line.split(maxsplit=1)
        if len(t) == 2:
            sums[t[1].strip().lstrip("*").replace("\\", "/")] = t[0]
    neu = [i for i in sorted(set(int(i) for i in ids)) if i not in have]
    t0 = time.time()
    for n, i in enumerate(neu):
        rel = f"records500/{i // 1000 * 1000:05d}/{i:05d}_hr"
        base = PTBXL_ROOT / rel
        for suf in (".hea", ".dat"):
            soll = sums.get(rel + suf)
            if soll is None or sha_datei(str(base) + suf) != soll:
                raise RuntimeError(f"Datei weicht von SHA256SUMS ab oder fehlt: {rel}{suf}")
        r = wfdb.rdrecord(str(base))
        sig = np.ascontiguousarray(np.asarray(r.p_signal, dtype=np.float64))
        have[i] = hashlib.sha256(sig.tobytes()).hexdigest()
        if n % 500 == 0:
            log(f"Signalhash {n}/{len(neu)} ({time.time() - t0:.0f} s)")
    if neu:
        pd.DataFrame({"ecg_id": sorted(have), "sha256": [have[i] for i in sorted(have)]}).to_csv(cache, index=False)
    return have


def lade_langformat(pfad, werkzeug, mit_wert, schluessel_map=None, bezug=None):
    """Wertematrix ecg_id x Schluessel (float, NaN = fehlend). Ohne mit_wert: 1.0 bei status ok, sonst NaN
    (die Spalte wert wird dann nicht gelesen)."""
    use = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "einheit", "status", "bezug", "ableitung"]
    if mit_wert:
        use.append("wert")
    d = pd.read_csv(pfad, usecols=use, dtype={"aufnahme": str, "status": str, "messgroesse": str,
                                              "vorverarbeitung": str, "datensatz": str, "einheit": str})
    d = d[(d.datensatz == "ptbxl") & (d.vorverarbeitung == "VV0")]
    if bezug:
        d = d[d.bezug == bezug]
    d = d[d.ableitung == "global"]
    d["aufnahme"] = d.aufnahme.astype(int)
    if schluessel_map:
        d = d[d.messgroesse.isin(schluessel_map)]
        d["messgroesse"] = d.messgroesse.map(schluessel_map)
    if d.duplicated(["aufnahme", "messgroesse"]).any():
        raise RuntimeError(f"Doppelte Zeilen (aufnahme, messgroesse) in {werkzeug}")
    einh = d.groupby("messgroesse").einheit.agg(lambda s: sorted(set(s.dropna().str.lower())))
    if mit_wert:
        w = pd.to_numeric(d.wert, errors="coerce")
        gueltig = d.status.eq("ok") & np.isfinite(w)
        v = w.where(gueltig)
        diff_ok_nicht_endlich = int((d.status.eq("ok") & ~np.isfinite(w)).sum())
    else:
        v = pd.Series(np.where(d.status.eq("ok"), 1.0, np.nan), index=d.index)
        diff_ok_nicht_endlich = None
    wide = pd.DataFrame({"ecg_id": d.aufnahme.values, "k": d.messgroesse.values, "v": v.values}).pivot(
        index="ecg_id", columns="k", values="v")
    stat = pd.DataFrame({"ecg_id": d.aufnahme.values, "k": d.messgroesse.values, "s": d.status.values}).pivot(
        index="ecg_id", columns="k", values="s")
    return wide.sort_index(), stat.sort_index(), einh.to_dict(), diff_ok_nicht_endlich


# ------------------------------------------------------------------ Paartabelle
def paar_merkmale(P, meta, ids_identisch=None):
    P = P.copy()
    m1 = meta.reindex(P.ecg_id_1.to_numpy())
    m2 = meta.reindex(P.ecg_id_2.to_numpy())
    if (m1.patient_id.to_numpy() != P.patient_id.to_numpy()).any() or (m2.patient_id.to_numpy() != P.patient_id.to_numpy()).any():
        raise RuntimeError("patient_id der Paarliste weicht von den Metadaten ab")
    P["rausch_any"] = m1.rausch.to_numpy() | m2.rausch.to_numpy()
    P["pm_any"] = m1.schrittmacher.to_numpy() | m2.schrittmacher.to_numpy()
    P["gleich_geraet_ort"] = (m1.device.to_numpy() == m2.device.to_numpy()) & (m1.site.to_numpy() == m2.site.to_numpy())
    P["sr_beide"] = m1.sr.to_numpy() & m2.sr.to_numpy()
    P["norm_beide"] = m1.norm.to_numpy() & m2.norm.to_numpy()
    P["block_gleich"] = m1.block.to_numpy() == m2.block.to_numpy()
    P["proxy"] = P.sr_beide & ~P.pm_any & P.block_gleich
    P["sex1"] = m1.sex.to_numpy()
    P["alter1"] = m1.alter.to_numpy()
    P["alter1_flag300"] = m1.alter_flag300.to_numpy()
    P["devgrp"] = m1.devgrp.to_numpy()
    P["norm1"] = m1.norm.to_numpy()
    P["sr1"] = m1.sr.to_numpy()
    return P


def schritte(P, h, qrs_col="qrs_ok_ll"):
    """Schritte der Pair-Ebene 3.2 fuer das Fenster h: (Paare, Personen) nach Schritt 0, 1, 2, 3 und Masken."""
    m0 = P.abstand_h <= h
    m1 = m0 & ~P.pt_paar
    m2 = m1 & P[qrs_col]
    m3 = m2 & ~P.identisch
    out = []
    for m in (m0, m1, m2, m3):
        out.append((int(m.sum()), int(P.patient_id[m].nunique())))
    return out, (m0, m1, m2, m3)


def populationen(P, V_ll):
    """Alle Analysepopulationen als Liste von Dictionaries (name, analyse, df, tool, stratum, beschreibung)."""
    liste = []

    def add(name, analyse, df, tool="looplens", beschr="", stratum="", plan_key=None, besch_only=False):
        liste.append({"name": name, "analyse": analyse, "df": df, "tool": tool, "beschreibung": beschr,
                      "stratum": stratum, "plan_key": plan_key or name})

    def basis(h, pt_aus=True, qrs_col="qrs_ok_ll"):
        m = (P.abstand_h <= h) & P[qrs_col] & ~P.identisch
        if pt_aus:
            m &= ~P.pt_paar
        return m

    prim = basis(24)
    add("PRIM", "PRIM", P[prim], beschr="Primär: Paare bis 24 h, ohne PT-Paare, QRS-Extraktion auf beiden ECG, Signale verschieden")
    add("S1", "S1", P[basis(168)], beschr="Paare bis 168 h")
    add("S2", "S2", P[prim & P.gleich_geraet_ort], beschr="gleiches Gerät und gleicher Ort")
    add("S3", "S3", P[prim & ~P.rausch_any], beschr="ohne Aufnahmen mit Rauschvermerk")
    add("S4", "S4", P[prim & ~P.pm_any], beschr="ohne Schrittmacher")
    add("S5", "S5", P[prim & P.proxy], beschr="Proxy-Analogon (SR, kein Schrittmacher, gleicher Blockstatus), beschreibend")
    s6 = P[prim].sort_values(["ecg_id_1", "ecg_id_2"]).drop_duplicates("patient_id", keep="first")
    add("S6", "S6", s6, beschr="eine Paarung je Person (erstes Paar nach ecg_id_1)")
    add("S7", "S7", P[basis(24, pt_aus=False)], beschr="Paare mit PT-Aufnahme wieder aufgenommen")
    add("S8", "S8", P[prim & ~P.rausch_any & ~P.pm_any & P.gleich_geraet_ort],
        beschr="kombiniert: kein Rauschvermerk, kein Schrittmacher, gleiches Gerät")
    add("S9", "S9", P[basis(24, qrs_col="qrs_ok_bh")], tool="braveheart",
        beschr="BRAVEHEART, eigenes Primärfenster (qrs_ok_bh)")
    add("S9b", "S9", P[prim & P.qrs_ok_bh], tool="braveheart",
        beschr="BRAVEHEART, gleiche Paare wie LoopLens-Primär", plan_key="S9b")
    for lab, mask in (("maennlich", P.sex1 == 0), ("weiblich", P.sex1 == 1)):
        add("S10_" + lab, "S10", P[prim & mask], beschr="Geschlecht " + {"maennlich": "männlich", "weiblich": "weiblich"}[lab] + " (0 = männlich)", stratum=lab)
    add("S10_unter65", "S10", P[prim & (P.alter1 < 65)], beschr="Alter < 65 am ersten ECG", stratum="unter65")
    add("S10_ab65", "S10", P[prim & (P.alter1 >= 65)], beschr="Alter >= 65 am ersten ECG", stratum="ab65")
    for g in ("CS100", "AT-6", "CS-12", "AT-60"):
        add("S11_" + g, "S11", P[prim & P.gleich_geraet_ort & (P.devgrp == g)],
            beschr="Gerätegruppe " + g + " (gleiches Gerät und Ort)", stratum=g)
    if V_ll is not None and RR_KEY in V_ll.columns:
        hf1 = 60000.0 / V_ll[RR_KEY].reindex(P.ecg_id_1.to_numpy()).to_numpy()
        hf2 = 60000.0 / V_ll[RR_KEY].reindex(P.ecg_id_2.to_numpy()).to_numpy()
        dh = np.abs(hf2 - hf1)
        fin = np.isfinite(dh)
        for lab, mk in (("unter5", fin & (dh < 5)), ("5bis15", fin & (dh >= 5) & (dh <= 15)), ("ueber15", fin & (dh > 15))):
            add("S12_" + lab, "S12", P[prim & mk], stratum=lab, plan_key="S12",
                beschr="Herzfrequenzänderung " + {"unter5": "unter 5", "5bis15": "5 bis 15", "ueber15": "über 15"}[lab] + " /min")
    add("S13", "S13", P[prim & P.norm_beide], beschr="Normalbefund (beide ECG mit SCP-Code NORM)")
    add("Z1", "Z1", P[basis(2)], beschr="Abstand bis 2 h")
    add("Z2", "Z2", P[basis(24) & (P.abstand_h > 2)], beschr="Abstand über 2 h bis 24 h")
    add("Z3", "Z3", P[basis(168) & (P.abstand_h > 24)], beschr="Abstand über 24 h bis 168 h")
    return liste


# ------------------------------------------------------------------ Schaetzung
def schaetze(df, V, keys, seed, moments_fn, nboot=NBOOT):
    """Aufruf von moments (rv_a1) fuer die Paare df; liefert Punktwerte, Bootstrap, Roh-Differenzen je Marker."""
    x = V.reindex(df.ecg_id_1.to_numpy())[keys].to_numpy(float)
    y = V.reindex(df.ecg_id_2.to_numpy())[keys].to_numpy(float)
    subjects = df.patient_id.to_numpy().astype(np.int64)
    allids = np.unique(subjects)
    w = gewichte(len(allids), seed, nboot)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point, boot, npair, nper, bsd, den, scale = moments_fn(x, y, subjects, allids, w)
    est = (nper >= MIN_PERSONEN_SCHAETZBAR) & (npair > 1) & (np.asarray(den) > 0)
    point = point.copy()
    point[~est] = np.nan
    boot = boot.copy()
    boot[:, ~est, :] = np.nan
    return {"x": x, "y": y, "subjects": subjects, "allids": allids, "w": w, "point": point, "boot": boot,
            "npair": npair, "nper": nper, "bsd": bsd, "est": est, "scale": scale}


def robust_kennzahlen(x, y, rc_klass):
    ok = np.isfinite(x) & np.isfinite(y)
    d = (y - x)[ok]
    if len(d) < 2:
        return {k: float("nan") for k in ("rc_robust", "rho", "median_diff", "p95_abs", "q025", "q975")}
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    rcr = 1.96 * 1.4826 * mad
    return {"rc_robust": rcr, "rho": rcr / rc_klass if np.isfinite(rc_klass) and rc_klass > 0 else float("nan"),
            "median_diff": med, "p95_abs": float(np.percentile(np.abs(d), 95)),
            "q025": float(np.percentile(d, 2.5)), "q975": float(np.percentile(d, 97.5))}


def zeilen_fuer_population(pop, V, keys, ziel, step0, moments_fn, mit_R=True, proxy=False):
    """Kennzahlen je Marker fuer eine Population."""
    analyse = pop["analyse"]
    a, seed, klasse_erlaubt = ANALYSEN[analyse]
    df = pop["df"]
    n_pers_gesamt = int(df.patient_id.nunique())
    if len(df) == 0 or n_pers_gesamt < 2:
        return [], None
    S = schaetze(df, V, keys, seed, moments_fn)
    zeilen = []
    for j, key in enumerate(keys):
        cfg = [m for m in MARKER if m[1] == key][0]
        k, _, aid, div, rolle, _u = cfg
        pt, bt = S["point"][j], S["boot"][:, j, :]
        icc_lo, icc_hi, icc_n = ci_perzentil(bt[:, 0])
        b_lo, b_hi, _ = ci_perzentil(bt[:, 1])
        sd_lo, sd_hi, _ = ci_perzentil(bt[:, 2])
        rc_lo, rc_hi, rc_n = ci_perzentil(bt[:, 3])
        h = len(bt) // 2
        mc1 = ci_perzentil(bt[:h, 3], mind=int(0.95 * h))
        mc2 = ci_perzentil(bt[h:, 3], mind=int(0.95 * h))
        rc = float(pt[3])
        rob = robust_kennzahlen(S["x"][:, j], S["y"][:, j], rc)
        zeile = {
            "population": pop["name"], "analyse": analyse, "werkzeug": pop["tool"], "marker": key,
            "marker_name": MARKER_NAMEN[key], "seed": seed, "a": a, "k": k,
            "n_paare": int(S["npair"][j]), "n_personen": int(S["nper"][j]), "schaetzbar": bool(S["est"][j]),
            "icc": float(pt[0]), "icc_lo": icc_lo, "icc_hi": icc_hi,
            "bias": float(pt[1]), "bias_lo": b_lo, "bias_hi": b_hi,
            "sd_diff": float(pt[2]), "sd_diff_lo": sd_lo, "sd_diff_hi": sd_hi,
            "rc": rc, "rc_lo": rc_lo, "rc_hi": rc_hi, "rc_n_gueltig": rc_n,
            "sem": float(pt[2]) / math.sqrt(2) if np.isfinite(pt[2]) else float("nan"),
            "zwischen_sd": float(S["bsd"][j]) if S["est"][j] else float("nan"),
            "mc_rc_lo": abs(mc1[0] - mc2[0]), "mc_rc_hi": abs(mc1[1] - mc2[1]),
        }
        zeile["rc_ueber_zwischen_sd"] = (zeile["rc"] / zeile["zwischen_sd"]
                                         if np.isfinite(zeile["zwischen_sd"]) and zeile["zwischen_sd"] > 0 else float("nan"))
        zeile.update(rob)
        t = ziel[key]
        zeile["ausreissergetrieben_lesehilfe"] = bool(np.isfinite(zeile["rho"]) and zeile["rho"] < 0.6)
        if pop["tool"] == "looplens":
            zeile.update({"rc_a": t["rc"], "rc_a_lo": t["rc_lo"], "rc_a_hi": t["rc_hi"], "icc_a": t["icc"],
                          "bias_a": t["bias"], "icc_diff": zeile["icc"] - t["icc"],
                          "bias_diff_ueber_zwischen_sd": (abs(zeile["bias"] - t["bias"]) / zeile["zwischen_sd"]
                                                          if np.isfinite(zeile["zwischen_sd"]) and zeile["zwischen_sd"] > 0 else float("nan"))})
        rolle_eff = rolle_effektiv(rolle, key, step0) if pop["tool"] == "looplens" else "beschreibend"
        zeile["rolle"] = rolle_eff
        if mit_R and pop["tool"] == "looplens" and S["est"][j]:
            r = kombiniertes_R(bt[:, 3], rc, t["rc"], t["rc_lo"], t["rc_hi"], a, k)
            zeile.update(r)
            zeile["klasse"] = klassifiziere(r["R_lo"], r["R_hi"], zeile["n_personen"], r["R_n_gueltig"], rolle_eff,
                                            klasse_erlaubt)
            if rolle in ("primaer_bedingt", "sekundaer") and rolle_eff == "beschreibend":
                zeile["klasse_grund"] = "Schritt 0: " + step0.get(key, "?")
            if proxy or analyse == "S5":
                zeile["R_proxy_punkt"] = rc / t["rc_proxy"]
                zeile["rc_a_proxy"] = t["rc_proxy"]
        else:
            if pop["tool"] == "looplens":
                zeile["klasse"] = "nicht schätzbar" if not S["est"][j] else "beschreibend (keine Klasse)"
            else:
                zeile["klasse"] = "keine Klasse (anderes Werkzeug)"
        zeilen.append(zeile)
    return zeilen, S


# ------------------------------------------------------------------ Schwellenkreuzungen
def zelle_kategorie(a_ueber, b_ueber):
    return np.where(a_ueber & b_ueber, 0, np.where(~a_ueber & ~b_ueber, 3, np.where(a_ueber & ~b_ueber, 1, 2)))


def kappa_aus_zellen(c):
    """c: (..., 4) = beide darueber, erstes darueber/zweites darunter, erstes darunter/zweites darueber, beide darunter."""
    c = np.asarray(c, dtype=float)
    n = c.sum(-1)
    with np.errstate(all="ignore"):
        po = (c[..., 0] + c[..., 3]) / n
        p1 = (c[..., 0] + c[..., 1]) / n
        p2 = (c[..., 0] + c[..., 2]) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        return (po - pe) / (1 - pe)


def schwellen_tabelle(S, keys, seed_info=None):
    """Schwellenkreuzungen fuer die Primaerpopulation mit den gemeinsamen Ziehungen."""
    zeilen = []
    allids = S["allids"]
    iv = np.searchsorted(allids, S["subjects"])
    for key, liste in SCHWELLEN.items():
        j = keys.index(key)
        x, y = S["x"][:, j], S["y"][:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        for op, thr in liste:
            if op == ">":
                ax, ay = x > thr, y > thr
            else:
                ax, ay = x >= thr, y >= thr
            cat = zelle_kategorie(ax, ay)
            agg = np.zeros((len(allids), 4))
            np.add.at(agg, (iv[ok], cat[ok]), 1.0)
            tot = agg.sum(0)
            boot = S["w"] @ agg
            n = tot.sum()
            mind1 = tot[0] + tot[1] + tot[2]
            with np.errstate(all="ignore"):
                wa = (tot[1] + tot[2]) / n if n > 0 else float("nan")
                wm = (tot[1] + tot[2]) / mind1 if mind1 > 0 else float("nan")
                kap = float(kappa_aus_zellen(tot))
                bwa = (boot[:, 1] + boot[:, 2]) / boot.sum(1)
                bm1 = boot[:, 0] + boot[:, 1] + boot[:, 2]
                bwm = (boot[:, 1] + boot[:, 2]) / bm1
                bk = kappa_aus_zellen(boot)
            masked = maskiere_zellen(tot.astype(int))
            frei = all(s.isdigit() for s in masked)
            wa_lo, wa_hi, _ = ci_perzentil(bwa)
            wm_lo, wm_hi, _ = ci_perzentil(bwm)
            k_lo, k_hi, _ = ci_perzentil(bk)
            zeilen.append({
                "marker": key, "schwelle": f"{op} {thr:g}", "n_paare": int(n),
                "beide_darueber": masked[0], "erstes_darueber_zweites_darunter": masked[1],
                "erstes_darunter_zweites_darueber": masked[2], "beide_darunter": masked[3],
                "wechsel_anteil_alle": wa if frei else float("nan"), "wechsel_anteil_alle_lo": wa_lo if frei else float("nan"),
                "wechsel_anteil_alle_hi": wa_hi if frei else float("nan"),
                "wechsel_anteil_min1": wm if frei else float("nan"), "wechsel_anteil_min1_lo": wm_lo if frei else float("nan"),
                "wechsel_anteil_min1_hi": wm_hi if frei else float("nan"),
                "kappa": kap if frei else float("nan"), "kappa_lo": k_lo if frei else float("nan"),
                "kappa_hi": k_hi if frei else float("nan"),
                "hinweis": "" if frei else "Kleinzellenregel: Anteile und kappa nicht ausgewiesen",
                "_intern_zellen": [int(v) for v in tot],
            })
    return zeilen


def verfuegbarkeit(df, V, keys):
    """Verfuegbarkeit im Paar: beide / nur erstes / nur zweites / keines je Marker."""
    zeilen = []
    for key in keys:
        a = np.isfinite(V.reindex(df.ecg_id_1.to_numpy())[key].to_numpy(float))
        b = np.isfinite(V.reindex(df.ecg_id_2.to_numpy())[key].to_numpy(float))
        c = [int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())]
        m = maskiere_zellen(c)
        zeilen.append({"marker": key, "n_paare_gesamt": len(df), "beide": m[0], "nur_erstes": m[1], "nur_zweites": m[2],
                       "keines": m[3], "_intern": c})
    return zeilen


def p_verfuegbarkeit(df, V, meta):
    ids = pd.unique(np.r_[df.ecg_id_1.to_numpy(), df.ecg_id_2.to_numpy()])
    p = np.isfinite(V.reindex(ids)["p_dauer"].to_numpy(float))
    sub = meta.reindex(ids)
    ns = (sub.norm & sub.sr).to_numpy()
    return {"ecg_gesamt": int(len(ids)), "ecg_mit_p": int(p.sum()), "anteil_mit_p": float(p.mean()),
            "ecg_norm_sr": int(ns.sum()), "ecg_norm_sr_mit_p": int((p & ns).sum()),
            "anteil_norm_sr_ohne_p": float(1 - (p & ns).sum() / ns.sum()) if ns.sum() else float("nan")}


# ------------------------------------------------------------------ Gesamtrechnung
def rechne(P, V_ll, V_bh, meta, ziel, step0, moments_fn, mit_wert=True, log=print):
    """Gesamte Auswertung auf den Paar- und Wertetabellen. Gibt Ergebnis-Dictionary zurueck."""
    res = {}
    # Pair-Ebene
    zeil = []
    for h in (24, 168):
        sc, _ = schritte(P, h)
        txt = ["Kandidaten (Schritt 0)", "ohne PT-Paare (Schritt 1)", "QRS-Extraktion auf beiden ECG (Schritt 2)",
               "Signale verschieden (Schritt 3)"]
        last, last_p = sc[0]
        for i, ((np_, nper), t) in enumerate(zip(sc, txt)):
            if i == 0 or i == 3:
                sp, sq = str(np_), str(nper)
                last, last_p = np_, nper
            else:
                sp = str(np_) if not (1 <= last - np_ <= 9) else "nicht ausgewiesen (Kleinzellenregel)"
                sq = str(nper) if not (1 <= last_p - nper <= 9) else "nicht ausgewiesen (Kleinzellenregel)"
                if not sp.startswith("nicht"):
                    last = np_
                if not sq.startswith("nicht"):
                    last_p = nper
            zeil.append({"fenster_h": h, "schritt": t, "paare": sp, "personen": sq,
                         "planung": ("{} / {}".format(*PLANUNG_SCHRITTE[h][i]) if i < 3 else "Zählung im Lauf"),
                         "_intern": [np_, nper]})
    res["schritte"] = zeil
    # Populationen
    pops = populationen(P, V_ll if mit_wert else None)
    zp = []
    for p in pops:
        df = p["df"]
        pl = PLANUNG.get(p["plan_key"], PLANUNG.get(p["name"]))
        nper = int(df.patient_id.nunique())
        zp.append({"population": p["name"], "analyse": p["analyse"], "werkzeug": p["tool"], "beschreibung": p["beschreibung"],
                   "paare": zelle(len(df)), "personen": zelle(nper),
                   "planung_paare": "" if not pl or pl[0] is None else pl[0],
                   "planung_personen": "" if not pl else pl[1],
                   "_intern": [len(df), nper]})
    res["populationen"] = zp
    # Zaehlungen der Merkmale
    prim = (P.abstand_h <= 24) & ~P.pt_paar & P.qrs_ok_ll
    zm = []
    for name, mk in (("rauschvermerk_mit", P.rausch_any), ("rauschvermerk_ohne", ~P.rausch_any),
                     ("schrittmacher_mit", P.pm_any), ("schrittmacher_ohne", ~P.pm_any),
                     ("geraet_ort_gleich", P.gleich_geraet_ort), ("geraet_ort_verschieden", ~P.gleich_geraet_ort),
                     ("proxy_analogon", P.proxy)):
        d = P[prim & mk]
        pl = PLANUNG_MERKMALE[name]
        zm.append({"merkmal": name, "paare": zelle(len(d)), "personen": zelle(d.patient_id.nunique()),
                   "planung": f"{pl[0]} / {pl[1]}", "_intern": [len(d), int(d.patient_id.nunique())]})
    res["merkmale"] = zm
    # Hilfsgroessen der Pair-Ebene
    res["identische_signale"] = {
        "paare_identisch_bis_168h": int(((P.abstand_h <= 168) & P.identisch).sum()),
        "paare_identisch_bis_24h": int(((P.abstand_h <= 24) & P.identisch).sum()),
        "abstand0_verschiedenes_signal_bis_24h": int(((P.abstand_h <= 24) & (P.abstand_h == 0) & ~P.identisch).sum()),
        "alter_300_paare_primaer": int(P[prim & ~P.identisch].alter1_flag300.sum()),
    }
    if not mit_wert:
        res["pops_objekte"] = pops
        return res
    # Schaetzungen
    alle_zeilen = []
    schwellen, verf, pver, prim_S = [], [], None, None
    for pop in pops:
        analyse = pop["analyse"]
        # Strata unter 30 Personen: nicht berichtet
        nper = int(pop["df"].patient_id.nunique())
        if nper < MIN_PERSONEN_SCHAETZBAR:
            alle_zeilen.append({"population": pop["name"], "analyse": analyse, "werkzeug": pop["tool"],
                                "marker": "ALLE", "klasse": "nicht berichtet (weniger als 30 Personen)",
                                "n_personen": nper})
            continue
        if pop["tool"] == "braveheart":
            zeilen, S = zeilen_fuer_population(pop, V_bh, BH_MARKER, ziel, step0, moments_fn, mit_R=False)
        else:
            zeilen, S = zeilen_fuer_population(pop, V_ll, LL_KEYS, ziel, step0, moments_fn, mit_R=True)
        alle_zeilen.extend(zeilen)
        log(f"{pop['name']}: {len(pop['df'])} Paare, {nper} Personen, {len(zeilen)} Zeilen")
        if pop["name"] == "PRIM":
            prim_S = S
            schwellen = schwellen_tabelle(S, LL_KEYS)
            verf = verfuegbarkeit(pop["df"], V_ll, LL_KEYS)
            pver = p_verfuegbarkeit(pop["df"], V_ll, meta)
    res["kennzahlen"] = alle_zeilen
    res["schwellen"] = schwellen
    res["verfuegbarkeit"] = verf
    res["p_verfuegbarkeit"] = pver
    res["pops_objekte"] = pops
    res["prim_S"] = prim_S
    return res


# ------------------------------------------------------------------ Ausgabe
def zeilen_oeffentlich(zeilen):
    """Kleinzellenregel auf die Zaehlspalten der Kennzahlzeilen."""
    out = []
    for z in zeilen:
        z = dict(z)
        for c in ("n_paare", "n_personen"):
            if c in z and isinstance(z[c], (int, np.integer)):
                z[c] = zelle(z[c])
        out.append(z)
    return out


def schreibe_csv(zeilen, pfad):
    if not zeilen:
        return
    df = pd.DataFrame([{k: v for k, v in z.items() if not str(k).startswith("_")} for z in zeilen])
    df.to_csv(pfad, index=False, float_format="%.10g")


def satz_tabelle(res, step0):
    """Erlaubte Saetze je Klasse fuer die Primaer- und Sekundaermarker der Primaeranalyse (und Sensitivitaeten)."""
    zeilen = []
    for z in res.get("kennzahlen", []):
        if z.get("marker") in (None, "ALLE") or "R" not in z:
            continue
        sens = z["analyse"] != "PRIM"
        fenster = "Paare bis 24 h" if z["analyse"] not in ("S1",) else "Paare bis 168 h"
        name = MARKER_NAMEN[z["marker"]]
        satz = erlaubter_satz(z["klasse"], name, z["R"], z["R_lo"], z["R_hi"], z["n_personen"], sens, fenster)
        zeilen.append({"population": z["population"], "analyse": z["analyse"], "marker": z["marker"], "rolle": z["rolle"],
                       "klasse": z["klasse"], "R": z["R"], "R_lo": z["R_lo"], "R_hi": z["R_hi"],
                       "n_personen": z["n_personen"], "satz": satz, "pflichtzusatz": PFLICHTZUSATZ})
    return zeilen


def md_tabelle(kopf, zeilen):
    z = ["| " + " | ".join(kopf) + " |", "|" + "|".join(["---"] * len(kopf)) + "|"]
    for r in zeilen:
        z.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(z)


def auto_markdown(res, step0):
    """Kompakte Tabellen (deutsch, Dezimalkomma) fuer den Berichtsentwurf. Zaehlzellen nach Kleinzellenregel."""
    kz = [z for z in res["kennzahlen"] if z.get("marker") not in (None, "ALLE")]
    teile = ["# AW-223 Ergebnistabellen (vom Skript erzeugt)", ""]
    teile += ["## Populationen", "", md_tabelle(
        ["Population", "Beschreibung", "Paare", "Personen", "Planung Paare", "Planung Personen"],
        [[p["population"], p["beschreibung"], p["paare"], p["personen"], p["planung_paare"], p["planung_personen"]]
         for p in res["populationen"]]), ""]

    def zeile(z):
        n = zelle(z["n_personen"]) if isinstance(z["n_personen"], (int, np.integer)) else z["n_personen"]
        if "R" in z:
            return [z["marker_name"], n, fz(z["icc"], 3) + " " + fzi(z["icc_lo"], z["icc_hi"], 3),
                    fz(z["bias"], 2) + " " + fzi(z["bias_lo"], z["bias_hi"], 2),
                    fz(z["rc"], 3) + " " + fzi(z["rc_lo"], z["rc_hi"], 3), fz(z["rc_a"], 3),
                    fz(z["R"], 3) + " " + fzi(z["R_lo"], z["R_hi"], 3), fzi(z["R_lo_ptb"], z["R_hi_ptb"], 3),
                    z["klasse"] + (" (" + z["klasse_grund"] + ")" if z.get("klasse_grund") else "")]
        return [z["marker_name"], n, fz(z["icc"], 3), fz(z["bias"], 2), fz(z["rc"], 3), "", "", "", z["klasse"]]

    kopf = ["Marker", "Personen", "ICC", "Bias", "RC", "RC A", "R (kombiniert)", "R (nur PTB-XL)", "Klasse"]
    for ana, titel in (("PRIM", "Primäranalyse (Paare bis 24 h)"), ("S1", "Sensitivität S1 (Paare bis 168 h)")):
        sel = [z for z in kz if z["population"] == ana and z["werkzeug"] == "looplens"]
        if sel:
            teile += ["## " + titel, "", md_tabelle(kopf, [zeile(z) for z in sel]), ""]
    sel = [z for z in kz if z["analyse"] != "PRIM" and z["werkzeug"] == "looplens" and z["marker"] in
           ("qrs_dauer_global", "qtcf", "qrst_winkel_integral_ll", "qt_global", "qrst_winkel_peak_ll")]
    teile += ["## Sensitivitäten und Zeitzerlegung (R mit kombiniertem Intervall)", "", md_tabelle(
        ["Population", "Marker", "Personen", "R", "Klasse"],
        [[z["population"], z["marker_name"], zelle(z["n_personen"]) if isinstance(z["n_personen"], (int, np.integer)) else z["n_personen"],
          (fz(z["R"], 3) + " " + fzi(z["R_lo"], z["R_hi"], 3)) if "R" in z else "k. A.", z["klasse"]] for z in sel]), ""]
    sw = res.get("schwellen") or []
    if sw:
        teile += ["## Schwellenkreuzungen (Primärpopulation)", "", md_tabelle(
            ["Marker", "Schwelle", "Paare", "beide darüber", "erstes darüber", "zweites darüber", "beide darunter",
             "Wechselanteil alle", "kappa"],
            [[w["marker"], w["schwelle"], w["n_paare"], w["beide_darueber"], w["erstes_darueber_zweites_darunter"],
              w["erstes_darunter_zweites_darueber"], w["beide_darunter"],
              fz(100 * w["wechsel_anteil_alle"], 1) + " %" if np.isfinite(w["wechsel_anteil_alle"]) else "k. A.",
              fz(w["kappa"], 3) if np.isfinite(w["kappa"]) else "k. A."] for w in sw]), ""]
    text = "\n".join(teile) + "\n"
    pruefe_sprache(text, "auto_markdown")
    return text


def schreibe_lauf(res, ziel, step0, V_ll, out_e, out_intern):
    """Schreibt alle Ergebnisdateien des Laufs (Aggregate nach out_e, Einzelpaare nur nach out_intern)."""
    out_e, out_intern = Path(out_e), Path(out_intern)
    kz = res["kennzahlen"]
    schreibe_csv(zeilen_oeffentlich(kz), out_e / "aw223_kennzahlen.csv")
    schreibe_csv(kz, out_intern / "aw223_kennzahlen_unmaskiert.csv")
    sat = satz_tabelle(res, step0)
    schreibe_csv(sat, out_e / "aw223_klassen.csv")
    schreibe_csv(res["schwellen"], out_e / "aw223_schwellen.csv")
    schreibe_csv(res["verfuegbarkeit"], out_e / "aw223_verfuegbarkeit_paar.csv")
    (out_e / "aw223_p_verfuegbarkeit.json").write_text(json.dumps(res["p_verfuegbarkeit"], indent=1), encoding="utf8")
    zt = [{"marker": k, **{c: v for c, v in t.items()}} for k, t in ziel.items()]
    schreibe_csv(zt, out_e / "aw223_zielwerte_a_verwendet.csv")
    (out_e / "aw223_ergebnistabellen.md").write_text(auto_markdown(res, step0), encoding="utf8")
    pops = {p["name"]: p for p in res["pops_objekte"]}
    d = pops["S1"]["df"][["patient_id", "ecg_id_1", "ecg_id_2", "abstand_h"]].copy()
    for key in LL_ALLE:
        d[key + "_1"] = V_ll.reindex(d.ecg_id_1.to_numpy())[key].to_numpy()
        d[key + "_2"] = V_ll.reindex(d.ecg_id_2.to_numpy())[key].to_numpy()
    d.to_csv(out_intern / "einzelpaare_bis168h_intern.csv.gz", index=False, float_format="%.10g")


def hauptprogramm(args):
    t0 = time.time()
    lauf_id = dt.datetime.now().strftime("AW223_%Y%m%d_%H%M%S")
    mit_wert = args.modus == "lauf"
    out_e = Path(args.out_e)
    out_pub = Path(args.out_pub)
    out_intern = Path(args.out_intern)
    for p in (out_e, out_pub, out_intern):
        p.mkdir(parents=True, exist_ok=True)
    logf = open(out_e / f"aw223_{args.modus}.log", "w", encoding="utf8")

    def log(msg):
        s = f"{dt.datetime.now():%H:%M:%S} {msg}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    skript = Path(sys.argv[0]).resolve()
    skript_hash = sha_datei(skript)
    log(f"AW-223 Auswertung Version {VERSION}, Modus {args.modus}, Skript {skript.name} SHA-256 {skript_hash}")
    prot = pruefe_bindung()
    log(f"Bindung geprueft: {len(prot)} Hashes gleich")
    step0 = lade_schritt0()
    log("Schritt 0: " + ", ".join(f"{k}={v}" for k, v in step0.items()))
    _est, moments_fn = lade_rv()
    selbsttest_rv(moments_fn, log)
    ziel = lade_zielwerte()
    for m in kontrolle_zielwerte(ziel):
        log("Hinweis Zielwerte: " + m)
    meta = lade_metadaten()
    P = lade_paarliste()
    log(f"Paarliste {len(P)} Paare, {P.patient_id.nunique()} Personen")
    # Rauchtest ohne Messwerte: Abstand >= 0, Personenzuordnung, Gegenprobe gegen die AE-271-Liste (Tage)
    if not (P.abstand_h >= 0).all():
        raise RuntimeError("abstand_h < 0 in der Paarliste")
    a271 = pd.read_csv(EINGABEN["paarliste_ae271"][0])
    a271["patient_id"] = a271.patient_id.round().astype(int)
    j = P.merge(a271, on=["patient_id", "ecg_id_1", "ecg_id_2"], how="outer", suffixes=("", "_271"), indicator=True)
    gleich_paare = bool((j["_merge"] == "both").all())
    max_diff = float((j.abstand_h - 24.0 * j.abstand_tage).abs().max())
    flags = bool(((j.qrs_ok_ll == j.qrs_ok_ll_271) & (j.qrs_ok_bh == j.qrs_ok_bh_271)).all()) if gleich_paare else False
    log(f"Gegenprobe AE-271: gleiche Paare {gleich_paare}, groesste Abweichung Stunden {max_diff:.2e}, Statusmerker gleich {flags}")
    if not (gleich_paare and max_diff < 1e-5 and flags):
        raise RuntimeError("Gegenprobe gegen die AE-271-Liste fehlgeschlagen")
    P = paar_merkmale(P, meta)
    # Signalhashes fuer alle Paare bis 168 h
    ids = pd.unique(np.r_[P[P.abstand_h <= 168].ecg_id_1.to_numpy(), P[P.abstand_h <= 168].ecg_id_2.to_numpy()])
    cache = out_intern / "signalhashes_ptbxl_aw223.csv"
    sh = signalhashes(ids, cache, log)
    P["identisch"] = [bool(a in sh and b in sh and sh[a] == sh[b]) for a, b in zip(P.ecg_id_1, P.ecg_id_2)]
    log(f"Signalhashes {len(sh)}; identische Paare bis 168 h: {int(((P.abstand_h <= 168) & P.identisch).sum())}")
    # Wertematrizen
    ll_map = None
    V_ll, st_ll, einh_ll, diff_ll = lade_langformat(EINGABEN["langformat_looplens"][0], "looplens", mit_wert, ll_map, "aufnahme")
    for key in LL_ALLE:
        if key not in V_ll.columns:
            raise RuntimeError(f"Schluessel fehlt im LoopLens-Langformat: {key}")
        if einh_ll.get(key) != [EINHEITEN_ERWARTET[key]]:
            raise RuntimeError(f"Einheit weicht ab: {key} {einh_ll.get(key)} gegen {EINHEITEN_ERWARTET[key]}")
    V_ll = V_ll[LL_ALLE]
    V_bh, st_bh, einh_bh, diff_bh = lade_langformat(EINGABEN["langformat_braveheart"][0], "braveheart", mit_wert, BH_ZU_LL, "medianschlag")
    V_bh = V_bh.reindex(columns=LL_ALLE)
    log(f"Wertematrizen: LoopLens {V_ll.shape}, BRAVEHEART {V_bh.shape}; status ok aber nicht endlich: LL {diff_ll}, BH {diff_bh}")
    res = rechne(P, V_ll, V_bh, meta, ziel, step0, moments_fn, mit_wert=mit_wert, log=log)
    # --- Ausgabe
    sub = "zaehlprobe_" if not mit_wert else ""
    for name in ("schritte", "populationen", "merkmale"):
        schreibe_csv(res[name], out_e / f"aw223_{sub}{name}.csv")
    (out_e / f"aw223_{sub}identische_signale.json").write_text(json.dumps(res["identische_signale"], indent=1), encoding="utf8")
    verf_status = []
    if not mit_wert:
        # Statusverfuegbarkeit je Marker in der Primaerpopulation (aus status ok)
        pops = {p["name"]: p for p in res["pops_objekte"]}
        verf_status = verfuegbarkeit(pops["PRIM"]["df"], V_ll, LL_KEYS)
        schreibe_csv(verf_status, out_e / "aw223_zaehlprobe_verfuegbarkeit_status.csv")
        planung_verf = []
        for key in LL_KEYS:
            a = np.isfinite(V_ll.reindex(P.ecg_id_1.to_numpy())[key].to_numpy(float))
            b = np.isfinite(V_ll.reindex(P.ecg_id_2.to_numpy())[key].to_numpy(float))
            m24 = (P.abstand_h <= 24) & a & b
            planung_verf.append({"marker": key, "paare_bis24h_vor_pt_abzug": zelle(int(m24.sum())),
                                 "personen": zelle(int(P.patient_id[m24].nunique()))})
        schreibe_csv(planung_verf, out_e / "aw223_zaehlprobe_markerpaare.csv")
    else:
        schreibe_lauf(res, ziel, step0, V_ll, out_e, out_intern)
    # Sprachpruefung der erzeugten Textspalten
    for f in list(out_e.glob("aw223_*.csv")) + list(out_e.glob("aw223_*.md")):
        pruefe_sprache(f.read_text(encoding="utf8"), f.name)
    peak = None
    try:
        import psutil
        peak = psutil.Process().memory_info().rss
    except Exception:
        pass
    manifest = {"lauf_id": lauf_id, "version": VERSION, "modus": args.modus, "skript": str(skript), "skript_sha256": skript_hash,
                "start": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                "ende": dt.datetime.now().isoformat(timespec="seconds"), "dauer_s": round(time.time() - t0, 1),
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "nboot": NBOOT, "seeds": {k: v[1] for k, v in ANALYSEN.items()}, "z_seed_basis": Z_SEED_BASIS,
                "bindung": prot, "schritt0": step0, "einheiten_looplens": {k: v for k, v in einh_ll.items()},
                "status_ok_nicht_endlich": {"looplens": diff_ll, "braveheart": diff_bh}, "rss_ende": peak,
                "wert_gelesen": mit_wert}
    (out_e / f"aw223_manifest_{args.modus}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    # Kopie der (kleinzellengeprueften) Aggregate in den Berichtsordner, ohne Einzelpaare
    import shutil
    for f in (list(out_e.glob("aw223_*.csv")) + list(out_e.glob("aw223_*.json")) + list(out_e.glob("aw223_*.log"))
              + list(out_e.glob("aw223_*.md"))):
        shutil.copy2(f, out_pub / f.name)
    log(f"fertig in {time.time() - t0:.0f} s")
    logf.close()
    return res, manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["zaehlprobe", "lauf"], required=True)
    ap.add_argument("--out-e", default=str(OUT_E))
    ap.add_argument("--out-pub", default=str(OUT_PUB))
    ap.add_argument("--out-intern", default=str(OUT_INTERN))
    args = ap.parse_args(argv)
    hauptprogramm(args)


if __name__ == "__main__":
    main()
