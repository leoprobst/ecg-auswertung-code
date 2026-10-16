"""Einheitentests fuer regeln.py (synthetische Messdatensaetze, keine Patientendaten). Stand v3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regeln as R  # noqa: E402


def base(**kw):
    amp = {l: {"r": 0.5, "s": 0.3} for l in R.ALL12}
    m = {"sex": "M", "amp": amp, "qrs_ms": 90.0, "pr_ms": 160.0, "p_ms": 100.0, "qt_ms": 400.0, "rr_ms": 1000.0,
         "rhythm": {"sinus": True, "af": False, "paced": False, "lbbb": False}}
    m.update(kw)
    return m


def test_sokolow_missing_lead_is_unknown_not_negative():
    m = base()
    del m["amp"]["V6"]
    assert R.sokolow_lyon(m) is None  # Maximum von V5/V6 unbekannt, untere Schranke 0,8 mV
    m["amp"]["V1"]["s"], m["amp"]["V5"]["r"] = 1.5, 2.1
    assert R.sokolow_lyon(m) is True  # schon V5 genuegt
    del m["amp"]["V5"]
    assert R.sokolow_lyon(m) is None


def test_sokolow_threshold_and_operator():
    m = base()
    m["amp"]["V1"]["s"], m["amp"]["V5"]["r"] = 1.5, 2.0
    assert R.sokolow_lyon(m) is False  # Vorgabe AHA Tab. 1: > 35 mm
    assert R.sokolow_lyon(m, strict=False) is True  # Original 1949: 35 mm or more


def test_cornell_sex_specific_and_requires_sex():
    m = base()
    m["amp"]["aVL"]["r"], m["amp"]["V3"]["s"] = 1.2, 1.0  # 2,2 mV
    assert R.cornell_voltage(m) is False
    m["sex"] = "F"
    assert R.cornell_voltage(m) is True
    m["sex"] = None
    assert R.cornell_voltage(m) is None


def test_cornell_product_molloy_and_variants():
    # Vorgabe (Molloy 1992): (RaVL + SV3 + 0,8 mV bei Frauen) x QRS > 243,6 mV*ms
    m = base(sex="F", qrs_ms=100.0)
    m["amp"]["aVL"]["r"], m["amp"]["V3"]["s"] = 1.0, 0.8  # 1,8 mV; Frau: 2,6 mV x 100 = 260
    assert R.cornell_product(m) is True
    assert R.cornell_product(m, female_offset=0.0) is False  # AHA-Tabelle ohne Frauenkorrektur: 180
    assert R.cornell_product(m, thr_mvms=286.0) is False  # Okin-Partition
    m["amp"]["V3"]["s"] = 0.5  # 1,5 + 0,8 = 2,3 x 100 = 230
    assert R.cornell_product(m) is False
    assert R.cornell_product(m, thr_mvms=244.0, female_offset=0.6) is False  # A155: 2,1 x 100 = 210
    m["qrs_ms"] = 106.0  # 2,3 x 106 = 243,8 > 243,6
    assert R.cornell_product(m) is True
    # Maenner: keine Korrektur
    mm = base(sex="M", qrs_ms=100.0)
    mm["amp"]["aVL"]["r"], mm["amp"]["V3"]["s"] = 1.3, 1.2  # 2,5 x 100 = 250
    assert R.cornell_product(mm) is True and R.cornell_product(mm, female_offset=0.0) is True
    mm["sex"] = None  # Geschlecht ist bei Frauenkorrektur Pflicht
    assert R.cornell_product(mm) is None and R.cornell_product(mm, female_offset=0.0) is True


def test_peguero_operator_and_doubling():
    m = base()
    m["amp"]["V2"]["s"], m["amp"]["V4"]["s"] = 2.0, 0.9  # 2,9 >= 2,8
    assert R.peguero_lo_presti(m) is True
    m["sex"] = "F"
    m["amp"]["V2"]["s"], m["amp"]["V4"]["s"] = 1.5, 0.8  # 2,3
    assert R.peguero_lo_presti(m) is True and R.peguero_lo_presti(m, strict=True) is False
    m["amp"]["V2"]["s"], m["amp"]["V4"]["s"] = 0.3, 1.2  # SD liegt in V4: V4 wird verdoppelt (1,2 + 1,2)
    assert abs(R.peguero_mv(m) - 2.4) < 1e-9


def test_low_voltage_components():
    m = base()
    for l in R.PREC:
        m["amp"][l] = {"r": 0.9, "s": 0.4}
    for l in R.LIMB:
        m["amp"][l] = {"r": 0.2, "s": 0.2}
    assert R.low_voltage_limb(m) is True and R.low_voltage_precordial(m) is False and R.low_voltage_ab(m) is True
    m["amp"]["I"] = {"r": 0.25, "s": 0.25}  # genau 0,5 mV
    assert R.low_voltage_limb(m) is True and R.low_voltage_limb(m, strict=True) is False
    m["amp"]["I"] = {"r": 0.2, "s": 0.2}
    del m["amp"]["V3"]
    assert R.low_voltage_precordial(m) is None and R.low_voltage_ab(m) is True
    m["amp"]["I"] = {"r": 0.4, "s": 0.4}
    assert R.low_voltage_ab(m) is None


def test_low_voltage_sokolow():
    m = base()
    m["amp"]["V1"]["s"], m["amp"]["V5"]["r"] = 0.5, 1.0
    assert R.low_voltage_sokolow(m) is True
    m["amp"]["V5"]["r"] = 1.1
    assert R.low_voltage_sokolow(m) is False


def test_qtc_formulas_and_af():
    assert abs(R.qtc_ms(400, 1000, "bazett") - 400) < 1e-9
    assert abs(R.qtc_ms(400, 640, "bazett") - 400 / 0.8) < 1e-6
    assert abs(R.qtc_ms(400, 640, "fridericia") - 400 / (0.64 ** (1 / 3))) < 1e-6
    m = base(qt_ms=420.0, rr_ms=640.0)
    assert R.qtc_gt(m, 500, "bazett") is True and R.qtc_gt(m, 500, "fridericia") is False
    m["rhythm"]["af"] = True
    assert R.qtc_gt(m) is None


def test_qtc_category():
    assert R.qtc_category(449, "M") == 0 and R.qtc_category(450, "M") == 1  # Empfehlungskasten: 450 ms or longer
    assert R.qtc_category(459, "F") == 0 and R.qtc_category(460, "F") == 1
    assert R.qtc_category(481, "M") == 2 and R.qtc_category(500, "F") == 2 and R.qtc_category(501, "F") == 3
    assert R.qtc_category(None, "M") is None and R.qtc_category(470, None) is None


def test_pr_requires_sinus():
    m = base(pr_ms=210.0)
    assert R.pr_gt_200(m) is True
    m["rhythm"]["sinus"] = None
    assert R.pr_gt_200(m) is None
    m["rhythm"] = {"sinus": True, "af": False, "paced": True, "lbbb": False}
    assert R.pr_gt_200(m) is None
    assert R.pr_gt_200(base(pr_ms=200.0)) is False


def test_qrs_flags_and_crt_class():
    f = R.qrs_flags(base(qrs_ms=135.0, sex="F"))
    assert f["ge120"] and f["ge130"] and not f["ge140"] and f["sex_specific_140_130"]
    assert R.qrs_flags(base(qrs_ms=135.0, sex="M"))["sex_specific_140_130"] is False
    assert R.qrs_flags(base(qrs_ms=110.0))["ge110"] is False and R.qrs_flags(base(qrs_ms=111.0))["ge110"] is True
    assert R.crt_qrs_class(base(qrs_ms=155.0), True) == "I"
    assert R.crt_qrs_class(base(qrs_ms=140.0), True) == "IIa"
    assert R.crt_qrs_class(base(qrs_ms=155.0), False) == "IIa"
    assert R.crt_qrs_class(base(qrs_ms=140.0), False) == "IIb"
    assert R.crt_qrs_class(base(qrs_ms=125.0), False) == "keine"
    assert R.crt_qrs_class(base(qrs_ms=140.0), None) is None


def _q(dur=0.0, depth=0.0, qs=False):
    return {"dur_ms": dur, "depth_mv": depth, "qs": qs}


def _qall():
    return {l: _q() for l in ("I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6")}


def test_q_udmi_v2v3_arm():
    q = _qall()
    m = base(q=q)
    assert R.q_udmi2018(m) is False
    q["V2"] = _q(21.0, 0.02)
    assert R.q_udmi2018(m) is True  # Q-Zacke > 20 ms in einer Ableitung
    q["V2"] = _q(20.0, 0.30)
    assert R.q_udmi2018(m) is False  # genau 20 ms zaehlt nicht (> 0,02 s)
    q["V3"] = _q(60.0, 0.5, True)  # QS in V3 allein
    assert R.q_udmi2018(m) is False  # D4 (AE-306): QS in V2 UND V3 verlangt
    assert R.q_udmi2018(m, qs_both_v2v3=False) is True
    q["V2"] = _q(60.0, 0.5, True)
    assert R.q_udmi2018(m) is True  # QS in beiden Ableitungen


def test_q_udmi_groups_and_contiguity():
    q = _qall()
    m = base(q=q)
    q["II"], q["aVF"] = _q(32, 0.12), _q(34, 0.11)
    assert R.q_udmi2018(m) is True
    q["aVF"] = _q(34, 0.05)  # zu flach
    assert R.q_udmi2018(m) is False
    q["aVF"] = _q(34, 0.11)
    q["II"] = _q(29, 0.2)  # zu kurz
    assert R.q_udmi2018(m) is False
    q2 = _qall()
    q2["V4"], q2["V6"] = _q(31, 0.2), _q(31, 0.2)  # V4 und V6 nicht benachbart
    assert R.q_udmi2018(base(q=q2)) is False
    q2["V5"] = _q(31, 0.2)
    assert R.q_udmi2018(base(q=q2)) is True
    q3 = _qall()
    q3["I"], q3["aVL"] = _q(30, 0.10), _q(0, 0.0, True)
    assert R.q_udmi2018(base(q=q3)) is True


def test_q_udmi_exclusions():
    q = _qall()
    q["V2"] = _q(30.0, 0.2)
    m = base(q=q)
    assert R.q_udmi2018(m) is True
    m["rhythm"]["lbbb"] = True
    assert R.q_udmi2018(m) is None
    m2 = base(q=q)
    m2["amp"]["V1"]["s"], m2["amp"]["V5"]["r"] = 2.0, 2.0  # Sokolow-Lyon positiv
    assert R.lvh_voltage_flag(m2) is True and R.q_udmi2018(m2) is None
    assert R.q_udmi2018(m2, lvh_by_voltage=False) is True


def test_rsb_complete_and_incomplete():
    w = {"V1": {"rp_mv": 0.4}, "V2": {"rp_mv": 0.2},
         "I": {"s_dur_ms": 50.0, "r_dur_ms": 60.0}, "V6": {"s_dur_ms": 45.0, "r_dur_ms": 50.0}}
    assert R.rsb_komplett(base(qrs_ms=130.0, wave=w)) is True
    assert R.rsb_komplett(base(qrs_ms=120.0, wave=w)) is True
    assert R.rsb_komplett(base(qrs_ms=119.0, wave=w)) is False
    assert R.rsb_inkomplett(base(qrs_ms=115.0, wave=w)) is True
    assert R.rsb_inkomplett(base(qrs_ms=125.0, wave=w)) is False
    w["V6"]["s_dur_ms"] = 30.0  # S nicht breiter als R und nicht > 40 ms: "I UND V6" verletzt
    assert R.rsb_komplett(base(qrs_ms=130.0, wave=w)) is False
    w["V6"]["s_dur_ms"] = 55.0
    w["I"]["s_dur_ms"] = 30.0
    assert R.rsb_komplett(base(qrs_ms=130.0, wave=w)) is False
    w["I"] = {"s_dur_ms": 30.0, "r_dur_ms": 20.0}  # S laenger als R genuegt
    assert R.rsb_komplett(base(qrs_ms=130.0, wave=w)) is True
    m = base(qrs_ms=130.0, wave=w)
    m["rhythm"]["paced"] = True
    assert R.rsb_komplett(m) is None


def test_maschine_normal():
    assert R.maschine_normal("Sinus rhythm Normal ECG") is True
    assert R.maschine_normal("Sinus rhythm Otherwise normal ECG") is False
    assert R.maschine_normal("Normal ECG except for rate") is False
    assert R.maschine_normal("Abnormal ECG") is False
    assert R.maschine_normal("") is None


def test_maschine_sinus():
    assert R.maschine_sinus("Sinus rhythm Normal ECG") is True
    assert R.maschine_sinus("Sinus bradycardia with 1st degree AV block") is True
    assert R.maschine_sinus("Sinus bradycardia with borderline 1st degree A-V block.") is True
    assert R.maschine_sinus("Sinus rhythm with PAC(s) | Sinus rhythm with PVCs") is True
    assert R.maschine_sinus("Probable sinus tachycardia") is True
    assert R.maschine_sinus("Atrial fibrillation with rapid ventricular response") is False
    assert R.maschine_sinus("Sinus rhythm with ventricular pacing paced rhythm") is False
    assert R.maschine_sinus("Sinus bradycardia | Demand pacing") is False  # D6: Stichprobenbefund
    assert R.maschine_sinus("Sinus rhythm | Pacemaker rhythm - no further analysis") is False
    assert R.maschine_sinus("Atrial pacing") is False
    assert R.maschine_sinus("Sinus rhythm | ST junctional depression is nonspecific | Borderline ECG") is True  # D6
    assert R.maschine_sinus("Pacer detection suspended due to external noise-REVIEW ADVISED | Sinus rhythm") is True
    assert R.maschine_sinus("Sinus rhythm with 2nd degree A-V block, Mobitz I") is False
    assert R.maschine_sinus("Sinus rhythm with A-V dissociation") is False
    assert R.maschine_sinus("Atrial flutter with rapid ventricular response with 2:1 A-V block") is False
    assert R.maschine_sinus("Junctional rhythm") is False
    assert R.maschine_sinus("Ectopic atrial rhythm") is False
    assert R.maschine_sinus("Sinus or ectopic atrial tachycardia") is False
    assert R.maschine_sinus("Regular supraventricular rhythm") is False
    assert R.maschine_sinus("Undetermined rhythm") is False
    assert R.maschine_sinus("Prolonged QT interval") is False
    assert R.maschine_sinus(None) is None


def test_hyperk_varga_has_no_peaked_t():
    r = R.hyperk_varga(base(qrs_ms=115.0, pr_ms=210.0))
    assert r == {"qrs_gt_110": True, "pr_gt_200": True}


def test_lvh_voltage_flag_unknown_is_none():
    m = base()
    assert R.lvh_voltage_flag(m) is False  # beide bekannt und negativ
    m2 = base()
    del m2["amp"]["V3"]  # Cornell unbekannt, Sokolow-Lyon bekannt und negativ
    assert R.lvh_voltage_flag(m2) is None
    m3 = base()
    m3["amp"] = {}
    assert R.lvh_voltage_flag(m3) is None
    m4 = base()
    m4["amp"]["V1"]["s"], m4["amp"]["V5"]["r"] = 2.0, 2.0
    del m4["amp"]["V3"]
    assert R.lvh_voltage_flag(m4) is True  # eine positive Komponente genuegt


def test_q_udmi_unknown_lvh_is_not_evaluable():
    q = {l: {"dur_ms": 0.0, "depth_mv": 0.0, "qs": False} for l in ("I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6")}
    q["V2"] = {"dur_ms": 30.0, "depth_mv": 0.2, "qs": False}
    m = base(q=q)
    assert R.q_udmi2018(m) is True
    m["amp"] = {}  # Spannung unbekannt: keine stille Auswertung (M11)
    assert R.q_udmi2018(m) is None
    assert R.q_udmi2018(m, lvh_by_voltage=False) is True
