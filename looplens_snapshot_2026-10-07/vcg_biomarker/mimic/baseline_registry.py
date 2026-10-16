from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .models import BaselineDefinition, BaselineKind
from .storage import content_hash, file_sha256


AHA_ECG_SOURCE = (
    "AHA/ACCF/HRS Recommendations for the Standardization and "
    "Interpretation of the Electrocardiogram"
)
AHA_ECG_URL = "https://www.ahajournals.org/toc/circ/119/10"
AHA_REPOLARIZATION_URL = (
    "https://www.ahajournals.org/doi/10.1161/"
    "CIRCULATIONAHA.108.191096"
)
AHA_HYPERTROPHY_URL = (
    "https://www.ahajournals.org/doi/10.1161/"
    "CIRCULATIONAHA.108.191097"
)
AHA_ISCHEMIA_URL = (
    "https://www.ahajournals.org/doi/10.1161/"
    "CIRCULATIONAHA.108.191098"
)
ACC_ACS_URL = (
    "https://www.acc.org/Guidelines/Guidelines/2025/02/27/17/21/"
    "Acute-Coronary-Syndromes-2025"
)
ESC_AF_URL = (
    "https://www.escardio.org/guidelines/clinical-practice-guidelines/"
    "all-esc-practice-guidelines/atrial-fibrillation/"
)
WELLENS_ORIGINAL_URL = (
    "https://www.sciencedirect.com/science/article/pii/000287038290480X"
)
DE_WINTER_ORIGINAL_URL = (
    "https://www.nejm.org/doi/abs/10.1056/NEJMc0804737"
)
PE_ECG_REVIEW_URL = "https://pubmed.ncbi.nlm.nih.gov/11593473/"
HYPOKALEMIA_COHORT_URL = "https://pubmed.ncbi.nlm.nih.gov/38098171/"


def _rule(
    baseline_id: str,
    endpoint_ids: tuple[str, ...],
    display_name: str,
    criterion_id: str,
    formula: str,
    required_inputs: tuple[str, ...],
    *,
    source_title: str = AHA_ECG_SOURCE,
    source_version: str = "2009 Parts V-VI",
    source_url: str = AHA_ECG_URL,
    exclusions: tuple[str, ...] = (),
    strata: tuple[str, ...] = (),
    kind: BaselineKind = BaselineKind.FIXED_RULE,
    notes: str = "",
    literature_reference_metrics: Optional[dict[str, Any]] = None,
) -> BaselineDefinition:
    return BaselineDefinition(
        baseline_id=baseline_id,
        endpoint_ids=endpoint_ids,
        display_name=display_name,
        kind=kind,
        source_title=source_title,
        source_version=source_version,
        source_url=source_url,
        formula=formula,
        required_inputs=required_inputs,
        exclusions=exclusions,
        strata=strata,
        criterion_id=criterion_id,
        notes=notes,
        literature_reference_metrics=literature_reference_metrics,
    )


_ACS_ENDPOINTS = (
    "acute_mi_high_specificity",
    "acute_mi_probable",
    "acute_troponin_change_proxy",
    "omi_revascularization_proxy",
    "troponin_positive_any",
)
_LVH_ENDPOINTS = (
    "echo_lvh_strict",
    "echo_lvh_sensitive",
    "lv_wall_thickening",
    "lv_dilatation_proxy",
    "concentric_remodeling_proxy",
)
_SCAR_ENDPOINTS = (
    "old_mi_icd",
    "mi_or_revascularization_history",
    "regional_wall_motion_abnormality",
)
_RIGHT_ENDPOINTS = (
    "echo_rv_dilation",
    "echo_rv_systolic_dysfunction",
    "elevated_rvsp_or_pasp",
    "echo_pulmonary_hypertension",
    "pulmonary_hypertension_icd",
    "pulmonary_embolism_icd",
    "right_heart_strain_proxy",
)
_ATRIAL_ENDPOINTS = (
    "echo_lae",
    "echo_rae",
    "new_onset_af_30d",
    "new_onset_af_90d",
    "new_onset_af_1y",
)
_QT_ENDPOINTS = (
    "qtc_gt_480",
    "qtc_gt_500",
    "qt_prolonging_drug_exposure",
    "vt_vf_icd_or_event",
    "cardiac_arrest",
)
_HYPERKALEMIA_ENDPOINTS = (
    "hyperkalemia_near_ecg",
    "severe_hyperkalemia_near_ecg",
)
_HYPOKALEMIA_ENDPOINTS = (
    "hypokalemia_near_ecg",
    "severe_hypokalemia_near_ecg",
)


BASELINE_DEFINITIONS: tuple[BaselineDefinition, ...] = (
    _rule(
        "sgarbossa_smith_modified_pacing", _ACS_ENDPOINTS,
        "Modifizierte Sgarbossa-Morphologie bei ventrikulärer Stimulation",
        "sgarbossa_smith_modified_pacing",
        "Modified Sgarbossa components; anterior concordant STD restricted to V1-V3",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "j_point", "documented_ventricular_pacing"),
        source_title="Dodd et al.: Modified Sgarbossa criteria in ventricular paced rhythm",
        source_version="2021; standard modified criteria, no V4-V6 extension",
        source_url="https://pubmed.ncbi.nlm.nih.gov/34172301/",
        exclusions=("missing_ventricular_paced_morphology", "poor_signal_quality"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Not applicable based on device presence or atrial pacing alone; morphology screen, not an automatic clinical diagnosis.",
    ),
    _rule(
        "sgarbossa_smith_modified_lbbb", _ACS_ENDPOINTS,
        "Modifizierte Sgarbossa-Morphologie bei belegtem LSB",
        "sgarbossa_smith_modified_lbbb",
        "Any concordant STE >=0.1 mV, concordant STD >=0.1 mV in V1-V3, or discordant STE >=0.1 mV and >=25% of S depth",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "j_point", "documented_lbbb"),
        source_title="Smith et al.: Modified Sgarbossa rule in LBBB",
        source_version="2012; LBBB population only",
        source_url="https://pubmed.ncbi.nlm.nih.gov/22939607/",
        exclusions=("missing_documented_lbbb", "poor_signal_quality", "unreliable_baseline"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Morphology only; no automatic MI diagnosis or reperfusion recommendation. Generic pacing flags do not establish applicability.",
    ),
    _rule(
        "guideline_st_elevation",
        _ACS_ENDPOINTS,
        "ST-Hebung in benachbarten Ableitungen",
        "st_elevation_contiguous",
        "J-point elevation in at least two contiguous leads with age/sex V2-V3 thresholds",
        ("diagnostic_ecg_mv", "qrs_offset", "age", "sex"),
        source_title=f"{AHA_ECG_SOURCE}, Part VI: Ischemia/Infarction",
        source_version="2009; clinical context checked against 2025 ACS guideline",
        source_url=AHA_ISCHEMIA_URL,
        exclusions=("paced_rhythm", "lbbb_separate_framework", "poor_signal_quality"),
        strata=("age", "sex", "lead_group"),
        notes=(
            "ECG comparator only; clinical ACS/MI reference labels remain "
            f"separate. Current management context: {ACC_ACS_URL}"
        ),
    ),
    _rule(
        "guideline_st_depression",
        _ACS_ENDPOINTS,
        "Horizontale/abfallende ST-Senkung",
        "st_depression_contiguous",
        "J+60 ms depression <= -0.05 mV in at least two contiguous leads",
        ("diagnostic_ecg_mv", "qrs_offset"),
        source_title=f"{AHA_ECG_SOURCE}, Part VI: Ischemia/Infarction",
        source_version="2009; clinical context checked against 2025 ACS guideline",
        source_url=AHA_ISCHEMIA_URL,
        exclusions=("paced_rhythm", "lbbb_separate_framework", "poor_signal_quality"),
        notes=f"Current management context: {ACC_ACS_URL}",
    ),
    _rule(
        "guideline_t_inversion",
        _ACS_ENDPOINTS,
        "T-Inversion in benachbarten Ableitungen",
        "t_inversion_contiguous",
        "UDMI-2018 quantitative R/S arm: T-wave amplitude < -0.10 mV AND R/S >1 in two contiguous leads",
        ("diagnostic_ecg_mv", "t_peak", "qrs_onset", "qrs_offset"),
        source_title="Fourth Universal Definition of Myocardial Infarction, Table 2",
        source_version="2018; quantitative R/S morphology component only",
        source_url="https://www.jacc.org/doi/10.1016/j.jacc.2018.08.1038",
        notes="Unquantified prominent-R alternative not implemented. Newness and clinical ischemia are not inferred from one ECG; a negative component does not exclude ischemia.",
    ),
    _rule(
        "posterior_mi_pattern",
        _ACS_ENDPOINTS + _SCAR_ENDPOINTS,
        "Posteriores Infarktmuster",
        "posterior_mi_pattern",
        "ST depression V1-V3 with dominant anterior R and upright T-wave",
        ("diagnostic_ecg_mv", "qrs_offset", "t_peak"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Pattern comparator; not proof of coronary occlusion.",
    ),
    _rule(
        "wellens_ecg_pattern",
        _ACS_ENDPOINTS,
        "Wellens-Morphologiemuster",
        "wellens_ecg_pattern",
        (
            "minimal J-point deviation with deep negative or "
            "positive-then-negative T morphology in at least two of V2-V4"
        ),
        ("diagnostic_ecg_mv", "qrs_offset", "t_fiducials"),
        source_title=(
            "de Zwaan, Bar and Wellens: Characteristic ECG pattern "
            "indicating critical proximal LAD stenosis"
        ),
        source_version="1982 original report; morphology screen only",
        source_url=WELLENS_ORIGINAL_URL,
        exclusions=("paced_rhythm", "lbbb", "poor_signal_quality"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Named morphology screen, not a universal diagnostic gold "
            "standard. Symptoms, timing, biomarkers and competing causes "
            "must be interpreted separately."
        ),
    ),
    _rule(
        "de_winter_ecg_pattern",
        _ACS_ENDPOINTS,
        "de-Winter-Morphologiemuster",
        "de_winter_ecg_pattern",
        (
            "upsloping precordial J-point depression continuing into "
            "prominent positive T waves in at least two of V2-V6"
        ),
        ("diagnostic_ecg_mv", "qrs_offset", "t_fiducials"),
        source_title=(
            "de Winter et al.: A New ECG Sign of Proximal LAD Occlusion"
        ),
        source_version="2008 original report; morphology screen only",
        source_url=DE_WINTER_ORIGINAL_URL,
        exclusions=("paced_rhythm", "lbbb", "poor_signal_quality"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Named expert pattern and comparator only; it does not establish "
            "coronary occlusion without the clinical reference evidence."
        ),
    ),
    _rule(
        "sokolow_lyon",
        _LVH_ENDPOINTS,
        "Sokolow-Lyon",
        "sokolow_lyon",
        "S(V1) + max(R(V5), R(V6)) >= 3.5 mV",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        source_title=f"{AHA_ECG_SOURCE}, Part V: Chamber Hypertrophy",
        source_url=AHA_HYPERTROPHY_URL,
    ),
    _rule(
        "cornell_voltage",
        _LVH_ENDPOINTS,
        "Cornell Voltage",
        "cornell_voltage",
        "R(aVL) + S(V3) > 2.8 mV male or > 2.0 mV female",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "sex"),
        strata=("sex",),
        source_title=f"{AHA_ECG_SOURCE}, Part V: Chamber Hypertrophy",
        source_url=AHA_HYPERTROPHY_URL,
    ),
    _rule(
        "cornell_product",
        _LVH_ENDPOINTS,
        "Cornell Product",
        "cornell_product",
        (
            "sex-adjusted Cornell voltage multiplied by QRS duration "
            "> 244 mV*ms (equivalent to 2440 mm*ms)"
        ),
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "sex"),
        strata=("sex",),
        source_title=f"{AHA_ECG_SOURCE}, Part V: Chamber Hypertrophy",
        source_url=AHA_HYPERTROPHY_URL,
    ),
    _rule(
        "peguero_lo_presti",
        _LVH_ENDPOINTS,
        "Peguero-Lo Presti",
        "peguero_lo_presti",
        "deepest S in any lead + S(V4), sex-specific threshold",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "sex"),
        strata=("sex",),
        source_title=f"{AHA_ECG_SOURCE}, Part V: Chamber Hypertrophy",
        source_url=AHA_HYPERTROPHY_URL,
    ),
    _rule(
        "pathological_q_waves",
        _SCAR_ENDPOINTS,
        "Pathologische Q-Wellen",
        "pathological_q_waves",
        "UDMI-2018 Q-wave component: V2/V3 >20 ms; other eligible leads >=30 ms AND >=0.10 mV in one lead group",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        exclusions=("lbbb", "paced_rhythm"),
        source_title="Fourth Universal Definition of Myocardial Infarction, Table 3 (Q-wave component only)",
        source_version="2018; Q-wave arm; QS and posterior R equivalents separately assessed",
        source_url="https://www.jacc.org/doi/10.1016/j.jacc.2018.08.1038",
        notes="Not a complete prior-MI diagnosis. V2/V3 use strict >20 ms; isolated III and aVR excluded. Clinical QRS confounders require review.",
    ),
    _rule(
        "qs_complexes",
        _SCAR_ENDPOINTS,
        "QS-Morphologie (strenger technischer Screen)",
        "qs_complexes",
        "Numerically absent positive QRS deflection (<=1e-12 mV), negative depth >=0.10 mV in >=2 contiguous precordial leads",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        exclusions=("lbbb", "paced_rhythm"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Technical amplitude floor and numerical tolerance, not clinical QS cutoffs. Noise/baseline sensitive; not a prior-MI diagnosis or validated QS detector.",
        source_title=f"{AHA_ECG_SOURCE}, Part VI: Ischemia/Infarction",
        source_url=AHA_ISCHEMIA_URL,
    ),
    _rule(
        "poor_r_wave_progression",
        _SCAR_ENDPOINTS,
        "Schlechte R-Progression",
        "poor_r_wave_progression",
        "insufficient precordial R-wave progression",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
    ),
    _rule(
        "right_axis_deviation",
        _RIGHT_ENDPOINTS,
        "Rechtsachsenabweichung",
        "right_axis_deviation",
        "frontal QRS axis > 90 degrees",
        ("machine_qrs_axis_deg_or_lead_axis",),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Supportive pattern; no accepted stand-alone ECG diagnostic rule for PE/PH.",
    ),
    _rule(
        "dominant_r_v1",
        _RIGHT_ENDPOINTS,
        "Dominantes R in V1",
        "dominant_r_v1",
        "R/S ratio in V1 > 1",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
    ),
    _rule(
        "dominant_s_v5_v6",
        _RIGHT_ENDPOINTS,
        "Dominantes S in V5/V6",
        "dominant_s_v5_v6",
        "S-wave amplitude greater than R-wave amplitude in V5 or V6",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Supportive rightward-force pattern; not a stand-alone "
            "diagnostic rule for PE, pulmonary hypertension or RV disease."
        ),
    ),
    _rule(
        "s1q3t3_pattern",
        _RIGHT_ENDPOINTS,
        "S1Q3T3-Muster",
        "s1q3t3_pattern",
        (
            "S in lead I, initial Q in lead III and inverted T in lead III"
        ),
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset", "t_fiducials"),
        source_title="Electrocardiographic manifestations of pulmonary embolism",
        source_version="2001 review; nonspecific supportive pattern",
        source_url=PE_ECG_REVIEW_URL,
        exclusions=("paced_rhythm", "lbbb", "poor_signal_quality"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Nonspecific right-heart-strain pattern. It is neither sensitive "
            "nor specific enough to diagnose or exclude pulmonary embolism."
        ),
    ),
    _rule(
        "p_terminal_force_v1",
        _ATRIAL_ENDPOINTS,
        "P terminal force V1",
        "p_terminal_force_v1",
        "terminal negative P area in V1 beyond the prespecified threshold",
        ("diagnostic_ecg_mv", "p_onset", "p_offset"),
    ),
    _rule(
        "p_duration_ge_120",
        _ATRIAL_ENDPOINTS,
        "P-Dauer >=120 ms",
        "p_duration_ge_120",
        "median P-wave duration >= 120 ms",
        ("p_onset", "p_offset", "sampling_rate_hz"),
    ),
    _rule(
        "bifid_p_wave",
        ("echo_lae",),
        "Bifide P-Welle",
        "bifid_p_wave",
        "notched or bifid P-wave morphology in lead II",
        ("diagnostic_ecg_mv", "p_onset", "p_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Supportive atrial morphology; not an independent LA-size reference.",
    ),
    _rule(
        "p_pulmonale",
        ("echo_rae",) + _RIGHT_ENDPOINTS,
        "P pulmonale",
        "p_pulmonale",
        "inferior-lead P-wave amplitude beyond the prespecified threshold",
        ("diagnostic_ecg_mv", "p_onset", "p_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Supportive right-atrial/right-heart pattern; not a stand-alone "
            "structural diagnosis."
        ),
    ),
    _rule(
        "qtc_bazett",
        _QT_ENDPOINTS,
        "QTc Bazett",
        "qtc_bazett",
        "QT / square root of RR",
        ("qrs_onset", "t_offset", "rr_interval"),
        source_title="AHA/ACCF/HRS ECG Standardization Part IV",
        source_version="2009",
        source_url=AHA_REPOLARIZATION_URL,
        kind=BaselineKind.CONTINUOUS_MARKER,
    ),
    _rule(
        "qtc_fridericia",
        _QT_ENDPOINTS,
        "QTc Fridericia",
        "qtc_fridericia",
        "QT / RR^(1/3)",
        ("qrs_onset", "t_offset", "rr_interval"),
        source_title="AHA/ACCF/HRS ECG Standardization Part IV",
        source_version="2009",
        source_url=AHA_REPOLARIZATION_URL,
        kind=BaselineKind.CONTINUOUS_MARKER,
    ),
    _rule(
        "qtc_framingham",
        _QT_ENDPOINTS,
        "QTc Framingham",
        "qtc_framingham",
        "QT + 154 * (1 - RR), with QT and RR in seconds",
        ("qrs_onset", "t_offset", "rr_interval"),
        source_title="AHA/ACCF/HRS ECG Standardization Part IV",
        source_version="2009",
        source_url=AHA_REPOLARIZATION_URL,
        kind=BaselineKind.CONTINUOUS_MARKER,
    ),
    _rule(
        "jtc_fridericia",
        _QT_ENDPOINTS,
        "JTc Fridericia",
        "jtc_fridericia",
        "QTc Fridericia minus QRS duration",
        ("qtc_fridericia", "qrs_duration"),
        source_title="AHA/ACCF/HRS ECG Standardization Part IV",
        source_version="2009",
        source_url=AHA_REPOLARIZATION_URL,
        kind=BaselineKind.CONTINUOUS_MARKER,
    ),
    _rule(
        "tpeak_tend",
        _QT_ENDPOINTS,
        "Tpeak-Tend",
        "tpeak_tend",
        "interval from T-wave peak to T-wave end",
        ("t_peak", "t_offset", "sampling_rate_hz"),
        source_title="AHA/ACCF/HRS ECG Standardization Part IV",
        source_version="2009; research comparator",
        source_url=AHA_REPOLARIZATION_URL,
        kind=BaselineKind.CONTINUOUS_MARKER,
        notes="Research repolarization comparator, not a universal binary rule.",
    ),
    _rule(
        "qtc_gt_480",
        _QT_ENDPOINTS,
        "QTc >480 ms",
        "qtc_gt_480",
        "Fridericia-corrected QT interval > 480 ms",
        ("qtc_fridericia",),
    ),
    _rule(
        "qtc_gt_500",
        _QT_ENDPOINTS,
        "QTc >500 ms",
        "qtc_gt_500",
        "Fridericia-corrected QT interval > 500 ms",
        ("qtc_fridericia",),
    ),
    _rule(
        "hyperkalemia_ecg_heuristic",
        _HYPERKALEMIA_ENDPOINTS,
        "Hyperkaliämie-ECG-Muster",
        "hyperkalemia_ecg_heuristic",
        "peaked T-wave pattern with QRS/PR/P-wave context",
        ("diagnostic_ecg_mv", "p_fiducials", "qrs_fiducials", "t_fiducials"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes="Supportive ECG pattern only; laboratory potassium is the reference.",
    ),
    _rule(
        "peaked_t_wave_heuristic",
        _HYPERKALEMIA_ENDPOINTS,
        "Spitze T-Wellen",
        "peaked_t_wave_heuristic",
        "prespecified narrow and high-amplitude T-wave morphology",
        ("diagnostic_ecg_mv", "t_fiducials"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Supportive and nonspecific morphology only; laboratory "
            "potassium is the reference."
        ),
    ),
    _rule(
        "qrs_widening_ge_120",
        _HYPERKALEMIA_ENDPOINTS,
        "QRS-Verbreiterung ab 120 ms",
        "qrs_widening_ge_120",
        "QRS duration >=120 ms",
        ("qrs_onset", "qrs_offset", "sampling_rate_hz"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Nonspecific supportive component only; conduction disease and "
            "pacing are important competing explanations."
        ),
    ),
    _rule(
        "pr_prolongation_gt_200",
        _HYPERKALEMIA_ENDPOINTS,
        "PR-Verlängerung über 200 ms",
        "pr_prolongation_gt_200",
        "PR interval >200 ms",
        ("p_onset", "qrs_onset", "sampling_rate_hz"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Nonspecific supportive component only; AV conduction disease "
            "and medication effects are competing explanations."
        ),
    ),
    _rule(
        "p_wave_attenuation",
        _HYPERKALEMIA_ENDPOINTS,
        "Abgeschwächte P-Welle",
        "p_wave_attenuation",
        "maximum P amplitude in leads II/V1 <0.05 mV",
        ("diagnostic_ecg_mv", "p_fiducials"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Nonspecific supportive component only; low voltage, lead "
            "placement and noise are competing explanations."
        ),
    ),
    BaselineDefinition(
        baseline_id="no_established_ecg_rule_hypokalemia",
        endpoint_ids=_HYPOKALEMIA_ENDPOINTS,
        display_name="Kein anerkannter ECG-Diagnosebaseline",
        kind=BaselineKind.NONE_ESTABLISHED,
        source_title=(
            "Prevalence and prognostic value of electrocardiographic "
            "abnormalities in hypokalemia: A multicenter cohort study"
        ),
        source_version="2024 multicenter cohort",
        source_url=HYPOKALEMIA_COHORT_URL,
        formula=(
            "No sufficiently sensitive stand-alone binary ECG rule; "
            "T-wave flattening, ST depression, QT/QU prolongation and U waves "
            "are supportive patterns while laboratory potassium remains the "
            "reference."
        ),
        required_inputs=(),
        notes=(
            "The supportive patterns are analyzed separately where robustly "
            "measurable; they are not combined into a diagnostic gold standard."
        ),
    ),
    BaselineDefinition(
        baseline_id="no_established_ecg_rule_reduced_ef",
        endpoint_ids=("reduced_lvef", "heart_failure_icd"),
        display_name="Kein anerkannter ECG-Diagnosebaseline",
        kind=BaselineKind.NONE_ESTABLISHED,
        source_title=AHA_ECG_SOURCE,
        source_version="2009",
        source_url=AHA_ECG_URL,
        formula="No stand-alone ECG rule establishes reduced ejection fraction.",
        required_inputs=(),
    ),
    BaselineDefinition(
        baseline_id="no_established_ecg_rule_future_af",
        endpoint_ids=("new_onset_af_30d", "new_onset_af_90d", "new_onset_af_1y"),
        display_name="Kein anerkannter ECG-Diagnosebaseline",
        kind=BaselineKind.NONE_ESTABLISHED,
        source_title="2024 ESC Guidelines for Atrial Fibrillation",
        source_version="2024",
        source_url=ESC_AF_URL,
        formula="No stand-alone ECG rule validates future incident AF risk.",
        required_inputs=(),
    ),
    _rule(
        "left_axis_deviation",
        _LVH_ENDPOINTS,
        "Linksachsenabweichung",
        "left_axis_deviation",
        "frontal QRS axis -30 to -90 degrees",
        ("machine_qrs_axis_deg_or_lead_axis",),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Supportive pattern; frequently a normal variant or reflects "
            "conduction-system disease (e.g. left anterior fascicular "
            "block) rather than structural LV disease alone."
        ),
    ),
    _rule(
        "low_voltage_qrs",
        _LVH_ENDPOINTS,
        "Niedrige QRS-Amplitude",
        "low_voltage_qrs",
        "peak-to-peak QRS <0.5 mV in all limb leads, or <1.0 mV in all "
        "precordial leads",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        notes=(
            "Low voltage alongside echo wall thickening (voltage-mass "
            "discordance) is the classic infiltrative-cardiomyopathy "
            "(amyloid) suspicion pattern; low voltage alone is "
            "nonspecific and also occurs with pericardial effusion, "
            "COPD or obesity."
        ),
    ),
    _rule(
        "rbbb_pattern",
        _RIGHT_ENDPOINTS,
        "Rechtsschenkelblock-Muster",
        "rbbb_pattern",
        "QRS >=120 ms with an rSR'-type terminal notch in V1 and a wide "
        "terminal S in I or V6",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        exclusions=("paced_rhythm", "lbbb"),
        notes=(
            "Morphology surrogate, not a full vector-loop BBB "
            "classification. RBBB is commonly a normal variant; the "
            "right-heart-strain endpoint pairing mirrors the existing "
            "dominant_r_v1/s1q3t3_pattern precedent and is supportive "
            "only, not diagnostic of PE or pulmonary hypertension."
        ),
    ),
    _rule(
        "lbbb_pattern",
        _LVH_ENDPOINTS,
        "Linksschenkelblock-Muster",
        "lbbb_pattern",
        "QRS >=120 ms with no initial septal q in I/V6, a predominantly "
        "negative QRS in V1, and no distinct terminal S in V6",
        ("diagnostic_ecg_mv", "qrs_onset", "qrs_offset"),
        kind=BaselineKind.EXPERT_PATTERN,
        exclusions=("paced_rhythm",),
        notes=(
            "Morphology surrogate, not a full vector-loop BBB "
            "classification. New LBBB is associated with LV systolic "
            "dysfunction/dilatation (lv_dilatation_proxy); it can also "
            "reflect isolated conduction-system disease without "
            "structural dysfunction."
        ),
    ),
)


class BaselineRegistry:
    def __init__(
        self,
        definitions: Iterable[BaselineDefinition] = BASELINE_DEFINITIONS,
    ) -> None:
        self._definitions = tuple(definitions)
        identifiers = [item.baseline_id for item in self._definitions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Baseline registry contains duplicate IDs")
        self._source_lock = _load_source_lock()
        self._source_lock_hash = file_sha256(_source_lock_path())
        _validate_definition_sources(self._definitions, self._source_lock)

    @property
    def registry_hash(self) -> str:
        return content_hash(
            {
                "definitions": [
                    item.to_dict() for item in self._definitions
                ],
                "source_lock_sha256": self._source_lock_hash,
            }
        )

    @property
    def source_lock_hash(self) -> str:
        return self._source_lock_hash

    @property
    def source_lock_path(self) -> Path:
        return _source_lock_path()

    @property
    def source_lock(self) -> dict:
        return dict(self._source_lock)

    def all(self) -> tuple[BaselineDefinition, ...]:
        return self._definitions

    def get(self, baseline_id: str) -> BaselineDefinition:
        for item in self._definitions:
            if item.baseline_id == baseline_id:
                return item
        raise KeyError(baseline_id)

    def for_endpoint(self, endpoint_id: str) -> tuple[BaselineDefinition, ...]:
        matches = tuple(
            item for item in self._definitions if endpoint_id in item.endpoint_ids
        )
        if matches:
            return matches
        return (
            BaselineDefinition(
                baseline_id=f"none_established__{endpoint_id}",
                endpoint_ids=(endpoint_id,),
                display_name="Kein anerkannter ECG-Diagnosebaseline",
                kind=BaselineKind.NONE_ESTABLISHED,
                source_title="Endpoint-specific evidence review required",
                source_version="registry-1.0.0",
                source_url="",
                formula="No accepted stand-alone ECG diagnostic criterion is registered.",
                required_inputs=(),
            ),
        )

    def frame(self, endpoint_id: Optional[str] = None) -> pd.DataFrame:
        definitions = (
            self.for_endpoint(endpoint_id)
            if endpoint_id is not None
            else self._definitions
        )
        frame = pd.DataFrame([asdict(item) for item in definitions])
        frame["source_lock_sha256"] = self._source_lock_hash
        frame["source_lock_version"] = self._source_lock.get("version", "")
        return frame


def _source_lock_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "baselines" / "SOURCE.json"


def _load_source_lock() -> dict:
    path = _source_lock_path()
    if not path.is_file():
        raise FileNotFoundError(f"Baseline source lock is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("version") or not isinstance(payload.get("sources"), list):
        raise RuntimeError(f"Invalid baseline source lock: {path}")
    source_ids = [
        str(item.get("source_id", ""))
        for item in payload["sources"]
        if isinstance(item, dict)
    ]
    if (
        not source_ids
        or any(not item for item in source_ids)
        or len(source_ids) != len(set(source_ids))
    ):
        raise RuntimeError(f"Invalid or duplicate baseline source IDs: {path}")
    return payload


def _validate_definition_sources(
    definitions: tuple[BaselineDefinition, ...],
    source_lock: dict,
) -> None:
    pinned_urls = {
        str(url).rstrip("/")
        for source in source_lock["sources"]
        for url in source.get("urls", ())
    }
    missing = sorted(
        {
            item.source_url
            for item in definitions
            if item.source_url
            and item.source_url.rstrip("/") not in pinned_urls
        }
    )
    if missing:
        raise RuntimeError(
            "Baseline definitions reference sources outside the local source "
            f"lock: {missing}"
        )
