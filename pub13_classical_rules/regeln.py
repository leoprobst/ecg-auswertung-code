"""PUB-13 (K1): Regeln als reine Funktionen nach dem Wortlaut der Primaerquelle (AW-232, Stand v3).

Zweck: treue, einzeln pruefbare Umsetzung der Faustregeln; ersetzt `classical.py` nicht und
aendert es nicht. Die Funktionen rechnen nur auf einem Messdatensatz `m` (ein EKG), nie auf
Referenzen. Drei Zustaende: True, False, None (nicht auswertbar, keine Stille als "negativ").

Messdatensatz `m` (alle Spannungen in mV, Zeiten in ms, Werte positiv gezaehlt):
  sex            "M", "F" oder None
  amp[lead]      {"r": hoechste positive Auslenkung des QRS inkl. R', "s": tiefste negative Auslenkung
                  (Betrag) inkl. S' bzw. QS}     lead in I II III aVR aVL aVF V1..V6
  q[lead]        {"dur_ms": Q-Dauer, "depth_mv": Q-Tiefe, "qs": True wenn reiner QS-Komplex}
  wave[lead]     {"r_dur_ms", "s_dur_ms", "rp_mv"} fuer die RSB-Funktionen
  qrs_ms, pr_ms, p_ms, qt_ms, rr_ms
  ptf_v1, p_notch_ms   nur fuer die gestrichene Regel R17 (Entwurf, ungenutzt)
  rhythm         {"sinus": bool|None, "af": bool|None, "paced": bool|None, "lbbb": bool|None}

GEBUNDENE WORTLAUTE (v3, am 15.10.2026 aus den Volltexten in E:\\VCG\\auswertung\\volltexte\\pub13 gelesen,
Seite/Tabelle je Regel in SPEZIFIKATION_K1_ENTWURF_v3.md Abschnitt 6.0). `REGELN` fuehrt den Status je Regel.
"""
from __future__ import annotations

import math
import re
from typing import Optional

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
PREC = ("V1", "V2", "V3", "V4", "V5", "V6")
ALL12 = LIMB + PREC

REGELN = {
    "R02": dict(name="Sokolow-Lyon", pmid="18107386; 19228820; 2949887", status="GEBUNDEN: SV1 + RV5/V6 > 35 mm = 3,5 mV (Hancock 2009 Tab. 1 und Casale: >; Original 1949: Tabelle '35 mm or more', Schluss 'exceeds 35 mm'); primaer >, Variante >="),
    "R03": dict(name="Cornell Spannung / Produkt", pmid="2949887; 19228820; 1401620; 7829796", status="GEBUNDEN Spannung: RaVL + SV3 > 28 mm Maenner, > 20 mm Frauen (Hancock Tab. 1, Casale). GEBUNDEN Produkt (AE-309 Nachtrag, Molloy 1992 Volltext): (RaVL + SV3 + 0,8 mV bei Frauen) x QRS > 2436 mm*ms = 243,6 mV*ms; Okin 1995 nutzt dieselbe Frauenkorrektur (800 uV) und die kohortenspezifische Partition 286 mV*ms; die AHA-Tabelle (Hancock Tab. 1) fuehrt das Produkt ohne Frauenkorrektur (Abweichung der Sekundaerquelle); A155 nutzt +0,6 mV und 244 (abweichende Umsetzung)"),
    "R04": dict(name="Peguero-Lo Presti", pmid="28359515", status="GEBUNDEN (Methoden): SD + SV4 >= 2,8 mV Maenner, >= 2,3 mV Frauen; SD = tiefstes S oder QS aller Ableitungen, Basislinie PR-Segment; liegt SD in V4, wird V4 verdoppelt; Ausschluss kompletter LSB/RSB und Schrittmacher"),
    "R06": dict(name="Niedervoltage (Erguss)", pmid="8697827", status="GEBUNDEN: Eisenberg: QRS-Amplitude < 5 mm in allen Extremitaetenableitungen; < 10 mm in allen Brustwandableitungen; 'alle' = beides; 'irgendeine' = eine der drei; primaer Extremitaeten (AE-295), strikt <"),
    "R07": dict(name="Niedervoltage (Amyloid)", pmid="25212550", status="GEBUNDEN: Cyrille: alle Extremitaeten <= 0,5 mV; alle Brustwand <= 1,0 mV; Sokolow-Index <= 1,5 mV (Abstract nennt einmal <= 0,5, Tippfehler); primaer Extremitaeten (AE-295)"),
    "R08": dict(name="QTc > 500 ms", pmid="20142454; 19228821", status="GEBUNDEN: > 500 ms (Drew 2010 'highly abnormal'; Rautaharju 2009/FDA 'longer than 500 ms'); Bazett primaer (AE-293); Rautaharju empfiehlt lineare Korrektur und Anpassung bei breitem QRS (JT), keine Korrektur bei grosser RR-Variabilitaet (VHF); verlaengert: Frauen >= 460 ms, Maenner > 450 ms"),
    "R09": dict(name="Kompletter Rechtsschenkelblock", pmid="19228822", status="GEBUNDEN (Surawicz 2009 Teil III): QRS >= 120 ms; rsr', rsR' oder rSR' in V1 oder V2; S-Zacke laenger als R oder > 40 ms in I UND V6; R-Gipfelzeit V5/V6 normal, V1 > 50 ms (nur Pflicht bei reinem R in V1); die ersten drei Kriterien muessen erfuellt sein. Inkompletter RSB: QRS 110 bis < 120 ms. In MIMIC primaer die Maschinenaussage"),
    "R14": dict(name="Pathologische Q (UDMI 2018)", pmid="30571511", status="GEBUNDEN (Tabelle 3, Circulation-Fassung): 'Any Q wave in V2-V3 > 0.02 s or QS complex in V2-V3'; 'Q wave >= 0.03 s and >= 1 mm deep or QS complex in I, II, aVL, aVF or V4-V6 in any two leads of a contiguous lead grouping (I, aVL; V1-V6; II, III, aVF)'; 'In the Absence of LVH and LBBB'; dritter Arm (R > 0,04 s in V1-V2, R/S > 1, positives T) nicht Teil von K1"),
    "R18": dict(name="PR > 200 ms", pmid="30586772; 19549974", status="GEBUNDEN: Kusumoto 2018: P-Wellen mit 1:1-Ueberleitung und PR > 200 ms; Cheng: Ableitung II, Ausschluss VHF, Antiarrhythmika/Glykoside, Schrittmacher; K1: Maschinen-PR primaer, Sinus aus Maschinenaussage (AE-295)"),
    "R19": dict(name="QRS-Schwellen nach Geschlecht", pmid="34455430; 19228822; 21376930; 25090172", status="GEBUNDEN: ESC 2021: LBBB-Morphologie: QRS >= 150 ms Klasse I, 130 bis 149 ms IIa; Nicht-LBBB: >= 150 ms IIa, 130 bis 149 ms IIb; < 130 ms nicht indiziert; Text: Simulationen sagen 9 bis 13 ms niedrigere Schwellen fuer Frauen voraus; AHA Teil III: kompletter Block >= 120 ms, QRS > 110 ms abnormal; Strauss: 140 ms M / 130 ms F plus Kerbung"),
    "R20": dict(name="Normales EKG bei Brustschmerz", pmid="37622654; 11667934", status="GEBUNDEN: ESC 2023: EKG bei NSTE-ACS in mehr als einem Drittel normal; Rule-out-Algorithmen mit Sensitivitaet und NPV >= 99 % (Fehlerrate <= 1 %); K1-Grenze 1 % von den Autoren vorab gesetzt, ESC 2023 als Orientierung"),
    "R13": dict(name="Hyperkaliaemie-Zeichen", pmid="31151388; 18235147", status="GESTRICHEN (AE-293/E4); hyperk_varga bleibt als Entwurf ungenutzt"),
    "R17": dict(name="Linksatriale Veraenderung", pmid="19228820; 17196473", status="GESTRICHEN (AE-293/E9): Morris-PTF aus dem Cache nicht exportierbar; Funktionen p_dauer_ge_120, ptf_v1_*, p_gekerbt bleiben als Entwurf ungenutzt"),
}


# ----------------------------------------------------------------- Hilfen
def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _lead(m: dict, lead: str, key: str) -> Optional[float]:
    try:
        v = m["amp"][lead][key]
    except (KeyError, TypeError):
        return None
    return float(v) if _ok(v) else None


def _sex(m: dict) -> Optional[str]:
    s = str(m.get("sex") or "").strip().upper()[:1]
    return s if s in ("M", "F") else None


def _sum(*v) -> Optional[float]:
    return None if any(x is None for x in v) else float(sum(v))


def _max(*v) -> Optional[float]:
    return None if any(x is None for x in v) else float(max(v))


def _cmp(value, thr, strict: bool) -> Optional[bool]:
    if value is None:
        return None
    return bool(value > thr) if strict else bool(value >= thr)


def _rhythm(m: dict, key: str) -> Optional[bool]:
    r = m.get("rhythm") or {}
    return r.get(key)


# ----------------------------------------------------------- LVH (R02 bis R04)
def sokolow_lyon_mv(m: dict) -> Optional[float]:
    """SV1 + max(RV5, RV6); fehlt eine Ableitung, ist die Summe unbekannt (nicht klein)."""
    return _sum(_lead(m, "V1", "s"), _max(_lead(m, "V5", "r"), _lead(m, "V6", "r")))


def sokolow_lyon(m: dict, strict: bool = True) -> Optional[bool]:
    """> 3,5 mV (Hancock Tab. 1, Casale); strict=False: >= 3,5 mV (Original 1949 Tabelle).
    Fehlt V5 oder V6, ist die Regel nur dann entschieden, wenn schon die vorhandene Ableitung
    die Schwelle erreicht (untere Schranke); sonst None."""
    full = sokolow_lyon_mv(m)
    if full is not None:
        return _cmp(full, 3.5, strict)
    s1 = _lead(m, "V1", "s")
    known = [x for x in (_lead(m, "V5", "r"), _lead(m, "V6", "r")) if x is not None]
    if s1 is not None and known and _cmp(s1 + max(known), 3.5, strict):
        return True
    return None


def cornell_mv(m: dict) -> Optional[float]:
    return _sum(_lead(m, "aVL", "r"), _lead(m, "V3", "s"))


def cornell_voltage(m: dict) -> Optional[bool]:
    """Casale 1987 / Hancock Tab. 1: RaVL + SV3 > 2,8 mV Maenner, > 2,0 mV Frauen (Geschlecht Pflicht)."""
    s, v = _sex(m), cornell_mv(m)
    if s is None or v is None:
        return None
    return bool(v > (2.8 if s == "M" else 2.0))


def cornell_product_mvms(m: dict, female_offset: float = 0.8) -> Optional[float]:
    """(RaVL + SV3 [+ female_offset bei Frauen]) x QRS-Dauer in mV*ms. Vorgabe: Molloy 1992 (PMID 1401620,
    Volltext): geschlechtsspezifische Cornell-Spannung 'adjusted by the addition of 8 mm [0.8 mV] for female
    gender', Produkt = diese Spannung x QRS-Dauer. female_offset=0.0: AHA-Tabelle (Hancock 2009 Tab. 1) ohne
    Frauenkorrektur (Abweichung der Sekundaerquelle); female_offset=0.6: A155-Umsetzung (abweichend)."""
    v, q = cornell_mv(m), m.get("qrs_ms")
    if v is None or not _ok(q):
        return None
    if female_offset:
        s = _sex(m)
        if s is None:
            return None
        v = v + (female_offset if s == "F" else 0.0)
    return v * float(q)


def cornell_product(m: dict, thr_mvms: float = 243.6, female_offset: float = 0.8) -> Optional[bool]:
    """Vorgabe (AE-309 Nachtrag, K bindet): (RaVL + SV3 + 0,8 mV bei Frauen) x QRS > 2436 mm*ms = 243,6 mV*ms
    (Molloy 1992, Partition bei auf 95 % abgestimmter Spezifitaet, Operator im Text nicht ausdruecklich: '>' wie
    in der AHA-Tabelle). Dokumentierte Varianten: AHA-Tabelle ohne Frauenkorrektur (female_offset=0.0);
    Okin 1995 mit gleicher Korrektur und kohortenspezifischer Partition 286 mV*ms (thr_mvms=286); A155
    (thr_mvms=244, female_offset=0.6). Geschlecht ist Pflicht, sobald female_offset > 0."""
    p = cornell_product_mvms(m, female_offset)
    return None if p is None else bool(p > thr_mvms)


def peguero_mv(m: dict) -> Optional[float]:
    """SD + SV4 mit SD = tiefstes S (oder QS) aller zwoelf Ableitungen; liegt SD in V4, wird V4
    verdoppelt (Peguero 2017, Methoden)."""
    s_all = [_lead(m, l, "s") for l in ALL12]
    if any(x is None for x in s_all):
        return None
    return _sum(max(s_all), _lead(m, "V4", "s"))


def peguero_lo_presti(m: dict, strict: bool = False) -> Optional[bool]:
    """>= 2,8 mV Maenner, >= 2,3 mV Frauen (Peguero 2017, Methoden; Abbildungstext nennt einmal >)."""
    s, v = _sex(m), peguero_mv(m)
    if s is None or v is None:
        return None
    return _cmp(v, 2.8 if s == "M" else 2.3, strict)


# ------------------------------------------------------ Niedervoltage (R06/R07)
def _ptp(m: dict, lead: str) -> Optional[float]:
    return _sum(_lead(m, lead, "r"), _lead(m, lead, "s"))


def _all_below(values, thr, strict) -> Optional[bool]:
    if any(x is None for x in values):
        return None
    return bool(all((x < thr) if strict else (x <= thr) for x in values))


def low_voltage_limb(m: dict, strict: bool = False) -> Optional[bool]:
    """Alle sechs Extremitaetenableitungen: Gesamtamplitude <= 0,5 mV (Cyrille, R07) bzw. < 5 mm
    (Eisenberg, R06; strict=True)."""
    return _all_below([_ptp(m, l) for l in LIMB], 0.5, strict)


def low_voltage_precordial(m: dict, strict: bool = False) -> Optional[bool]:
    """Alle sechs Brustwandableitungen <= 1,0 mV (Cyrille) bzw. < 10 mm (Eisenberg, strict=True)."""
    return _all_below([_ptp(m, l) for l in PREC], 1.0, strict)


def low_voltage_sokolow(m: dict) -> Optional[bool]:
    """Sokolow-Index (SV1 + RV5/V6) <= 1,5 mV (Cyrille)."""
    v = sokolow_lyon_mv(m)
    return None if v is None else bool(v <= 1.5)


def low_voltage_ab(m: dict, strict: bool = False) -> Optional[bool]:
    """Eisenberg 'irgendeine Definition': Extremitaeten ODER Brustwand (Sekundaervariante); eine fehlende
    Ableitung macht die Aussage nur dann unbekannt, wenn die andere Haelfte sie nicht entscheidet."""
    a, b = low_voltage_limb(m, strict), low_voltage_precordial(m, strict)
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


# ------------------------------------------------------------------- QTc (R08)
def qtc_ms(qt_ms, rr_ms, formula: str = "bazett") -> Optional[float]:
    if not (_ok(qt_ms) and _ok(rr_ms)) or rr_ms <= 0:
        return None
    rr = rr_ms / 1000.0
    f = formula.lower()
    if f == "bazett":
        return qt_ms / math.sqrt(rr)
    if f == "fridericia":
        return qt_ms / rr ** (1.0 / 3.0)
    if f == "framingham":
        return qt_ms + 154.0 * (1.0 - rr)
    if f == "hodges":
        hr = 60.0 / rr
        return qt_ms + 1.75 * (hr - 60.0)
    raise ValueError(f)


def qtc_gt(m: dict, thr: float = 500.0, formula: str = "bazett") -> Optional[bool]:
    v = qtc_ms(m.get("qt_ms"), m.get("rr_ms"), formula)
    if v is None:
        return None
    if _rhythm(m, "af") is True:
        return None  # Drew/Rautaharju: bei VHF keine Rate-Korrektur ohne Mittelung
    return bool(v > thr)


def qtc_category(qtc: Optional[float], sex: Optional[str]) -> Optional[int]:
    """Stufen fuer die Dosis-Wirkung (Rautaharju 2009 Empfehlungskasten, normativ: 'adjusted QT of 460 ms or
    longer in women and 450 ms or longer in men' = verlaengert; der Fliesstext sagt fuer Maenner 'longer than
    450 ms', die Empfehlung gewinnt, AE-309 C5): 0 = nicht verlaengert (Maenner < 450 ms, Frauen < 460 ms);
    1 = verlaengert bis 480 ms; 2 = > 480 bis 500 ms; 3 = > 500 ms."""
    s = str(sex or "").strip().upper()[:1]
    if not _ok(qtc) or s not in ("M", "F"):
        return None
    if qtc > 500:
        return 3
    if qtc > 480:
        return 2
    if (s == "M" and qtc >= 450) or (s == "F" and qtc >= 460):
        return 1
    return 0


# ------------------------------------------------------------- QRS (R19, R09)
def qrs_flags(m: dict) -> dict:
    q = m.get("qrs_ms")
    keys = ("ge110", "ge120", "ge130", "ge140", "ge150", "sex_specific_140_130")
    if not _ok(q):
        return {k: None for k in keys}
    s = _sex(m)
    return {
        "ge110": q > 110, "ge120": q >= 120, "ge130": q >= 130, "ge140": q >= 140, "ge150": q >= 150,
        # Strauss 2011: Dauerteil des strengen LSB (Kerbung nicht Teil dieser Funktion)
        "sex_specific_140_130": None if s is None else bool(q >= (140 if s == "M" else 130)),
    }


def crt_qrs_class(m: dict, lbbb: Optional[bool]) -> Optional[str]:
    """ESC 2021 (Abb./Empfehlungen CRT): Stufe nach QRS-Dauer und Morphologie ohne LVEF/Klinik:
    'I' LBBB und >= 150; 'IIa' LBBB 130 bis 149 oder Nicht-LBBB >= 150; 'IIb' Nicht-LBBB 130 bis 149;
    'keine' < 130 ms."""
    q = m.get("qrs_ms")
    if not _ok(q) or lbbb is None:
        return None
    if q < 130:
        return "keine"
    if lbbb:
        return "I" if q >= 150 else "IIa"
    return "IIa" if q >= 150 else "IIb"


def _wide_s(w: dict, lead: str) -> Optional[bool]:
    d = w.get(lead) or {}
    s, r = d.get("s_dur_ms"), d.get("r_dur_ms")
    if not (_ok(s) and _ok(r)):
        return None
    return bool(s > r or s > 40.0)


def _rsb_pattern(m: dict) -> Optional[bool]:
    """Kriterien 2 und 3 nach Surawicz 2009: rsr'/rsR'/rSR' in V1 oder V2 (R'-Amplitude >= 0,05 mV als
    technische Untergrenze) UND S-Zacke breiter als R oder > 40 ms in I UND V6."""
    w = m.get("wave") or {}

    def rp(lead):
        v = (w.get(lead) or {}).get("rp_mv")
        return float(v) if _ok(v) else None

    rps = [rp("V1"), rp("V2")]
    if all(x is None for x in rps):
        return None
    rsr = any(x is not None and x >= 0.05 for x in rps)
    ws = [_wide_s(w, "I"), _wide_s(w, "V6")]
    if any(x is None for x in ws):
        return None if rsr else False
    return bool(rsr and ws[0] and ws[1])


def rsb_komplett(m: dict) -> Optional[bool]:
    """Kompletter RSB: QRS >= 120 ms plus Muster (Surawicz 2009; Kriterium 4 nur bei reinem R in V1,
    hier nicht gerechnet). Schrittmacher ausgeschlossen."""
    if _rhythm(m, "paced") is True:
        return None
    q = m.get("qrs_ms")
    if not _ok(q):
        return None
    if q < 120:
        return False
    return _rsb_pattern(m)


def rsb_inkomplett(m: dict) -> Optional[bool]:
    """Inkompletter RSB: QRS 110 bis < 120 ms, sonst gleiche Kriterien (Surawicz 2009)."""
    if _rhythm(m, "paced") is True:
        return None
    q = m.get("qrs_ms")
    if not _ok(q):
        return None
    if not (110.0 <= q < 120.0):
        return False
    return _rsb_pattern(m)


# ------------------------------------------------------------- PR (R18)
def pr_gt_200(m: dict) -> Optional[bool]:
    """Kusumoto 2018: P-Wellen mit 1:1-Ueberleitung und PR > 200 ms; Cheng 2009: Ableitung II, ohne VHF und
    Schrittmacher. Nur im Sinusrhythmus definiert; ist der Rhythmus unbekannt, wird der Wert nicht gewertet."""
    pr = m.get("pr_ms")
    if not _ok(pr) or pr <= 0:
        return None
    if _rhythm(m, "af") is True or _rhythm(m, "paced") is True:
        return None
    if _rhythm(m, "sinus") is not True:
        return None
    return bool(pr > 200.0)


# ------------------------------------------------------------- Q-Zacken (R14)
def lvh_voltage_flag(m: dict) -> Optional[bool]:
    """LVH-Ausschluss fuer R14 ueber Spannung (AE-295): Sokolow-Lyon (>) ODER Cornell Spannung positiv.
    True, wenn eine Komponente positiv ist; False nur, wenn beide Komponenten bekannt und negativ sind;
    sonst None (unbekannter LVH-Status, AE-309 M11)."""
    a, b = sokolow_lyon(m), cornell_voltage(m)
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


def q_udmi2018(m: dict, qs_both_v2v3: bool = True, lvh_by_voltage: bool = True) -> Optional[bool]:
    """UDMI 2018 Tabelle 3, Q-Arme (Circulation-Fassung): V2/V3: Q > 20 ms in einer der beiden Ableitungen oder QS in V2 UND V3
    (AE-306 D4, konservative Lesart; qs_both_v2v3=False: QS in einer der beiden Ableitungen); I, II, aVL, aVF, V4 bis V6: Q >= 30 ms und >= 0,10 mV
    (1 mm) oder QS in mindestens zwei Ableitungen einer zusammenhaengenden Gruppe (I, aVL; II, aVF [III ist in
    der Ableitungsliste nicht aufgefuehrt]; V4-V5, V5-V6). Ausschluss: LVH (ueber Spannung), LSB, Schrittmacher.
    Der dritte Arm (R > 40 ms in V1/V2, R/S > 1, positives T) ist nicht Teil von K1."""
    if _rhythm(m, "paced") is True or _rhythm(m, "lbbb") is True or m.get("lvh_exclusion") is True:
        return None
    if lvh_by_voltage and lvh_voltage_flag(m) is not False:
        return None  # LVH positiv oder unbekannt: nicht auswertbar (M11), als Abdeckung berichten
    q = m.get("q")
    if not q:
        return None
    need = ("I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6")
    for l in need:
        d = q.get(l)
        if d is None or not (_ok(d.get("dur_ms")) and _ok(d.get("depth_mv"))) or d.get("qs") is None:
            return None

    def path(l):
        d = q[l]
        return bool(d["qs"] or (d["dur_ms"] >= 30.0 and d["depth_mv"] >= 0.10))

    def qwave(l):
        return bool((not q[l]["qs"]) and q[l]["dur_ms"] > 20.0)

    qs2, qs3 = bool(q["V2"]["qs"]), bool(q["V3"]["qs"])
    qs_arm = (qs2 and qs3) if qs_both_v2v3 else (qs2 or qs3)
    arm1 = bool(qwave("V2") or qwave("V3") or qs_arm)
    pairs = (("I", "aVL"), ("II", "aVF"), ("V4", "V5"), ("V5", "V6"))
    arm2 = any(path(a) and path(b) for a, b in pairs)
    return bool(arm1 or arm2)


# ------------------------------------------------------------- gestrichene Regeln (Entwurf, ungenutzt)
def p_dauer_ge_120(m: dict) -> Optional[bool]:
    p = m.get("p_ms")
    return None if not _ok(p) or p <= 0 else bool(p >= 120.0)


def ptf_v1_mm_s(m: dict) -> Optional[float]:
    t = m.get("ptf_v1")
    if not t or not (_ok(t.get("depth_mv")) and _ok(t.get("dur_ms"))):
        return None
    return abs(t["depth_mv"]) * 10.0 * (t["dur_ms"] / 1000.0)


def ptf_v1_pathologisch(m: dict, thr_mm_s: float = 0.04) -> Optional[bool]:
    v = ptf_v1_mm_s(m)
    return None if v is None else bool(v >= thr_mm_s)


def p_gekerbt(m: dict, min_interpeak_ms: float = 40.0) -> Optional[bool]:
    d = m.get("p_notch_ms")
    if d is None:
        return None
    return bool(_ok(d) and d >= min_interpeak_ms)


def hyperk_varga(m: dict) -> dict:
    q, pr = m.get("qrs_ms"), m.get("pr_ms")
    return {"qrs_gt_110": None if not _ok(q) else bool(q > 110.0), "pr_gt_200": pr_gt_200(m)}


# ------------------------------------------------------------- Maschinentext (R20, R18)
_NORMAL_BAD = re.compile(r"otherwise|except|borderline|abnormal|possible|probable|consider", re.I)


def maschine_normal(text: Optional[str]) -> Optional[bool]:
    """'Normal ECG' ohne Zusatz. None bei leerem Text. Die Variantenfrage (mit Zusatz) ist
    eine Sensitivitaet und wird vor Referenzsicht festgelegt."""
    if text is None or not str(text).strip():
        return None
    t = str(text)
    if not re.search(r"normal ecg", t, re.I):
        return False
    return not bool(_NORMAL_BAD.search(t))


_SINUS = re.compile(r"\bsinus (rhythm|bradycardia|tachycardia|arrhythmia)\b", re.I)
_NOT_SINUS = re.compile(
    r"fibrillation|afib|flutter|junctional (rhythm|tachycardia|bradycardia|escape)|ectopic|atrial rhythm|wandering|"
    r"supraventricular (tachycardia|rhythm)|atrial tachycardia|idioventricular|ventricular tachycardia|"
    r"paced|pacing|pacemaker|\bpacer\b|a-?v dissociation|2nd degree|second degree|3rd degree|third degree|"
    r"mobitz|wenckebach|high[- ]degree|complete a-?v block|\b[234]:1\b|undetermined rhythm", re.I)
_NEUTRAL = re.compile(r"pacer detection suspended[^|\n]*|st junctional", re.I)


def maschine_sinus(text: Optional[str]) -> Optional[bool]:
    """Sinusrhythmus aus dem Maschinenbefund (AE-295, an 300 Befundtexten aus train geprueft, AE-306 D6):
    ein Sinus-Satz (Rhythmus, Bradykardie, Tachykardie, Arrhythmie; 'probable' zaehlt) UND keine Aussage zu
    Flimmern, Flattern, Knotenrhythmus, ektopem/wanderndem Vorhofrhythmus, supraventrikulaerer oder
    ventrikulaerer Tachykardie, Schrittmacher oder Stimulation (auch 'demand pacing'), AV-Dissoziation,
    AV-Block 2./3. Grades oder 2:1/3:1/4:1-Ueberleitung. Neutral (vor der Pruefung entfernt): 'Pacer detection
    suspended due to external noise' und 'ST junctional depression'. AV-Block 1. Grades und Extrasystolen
    schliessen Sinus nicht aus. None bei leerem Text."""
    if text is None or not str(text).strip():
        return None
    t = _NEUTRAL.sub(" ", str(text))
    if _NOT_SINUS.search(t):
        return False
    return bool(_SINUS.search(t))
