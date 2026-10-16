from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


MAPPING_VERSION = "1.5.0"


@dataclass(frozen=True)
class ICDGroup:
    group_id: str
    icd9_prefixes: tuple[str, ...] = ()
    icd10_prefixes: tuple[str, ...] = ()
    description: str = ""

    def matches(self, code: str, version: int) -> bool:
        normalized = code.upper().replace(".", "").strip()
        prefixes = self.icd9_prefixes if int(version) == 9 else self.icd10_prefixes
        return any(normalized.startswith(prefix) for prefix in prefixes)


ICD_GROUPS: tuple[ICDGroup, ...] = (
    # Audit X9: needed to keep hypertensive and diabetic patients out of the
    # reference-range cohort, where they move the voltage markers.
    ICDGroup("essential_hypertension", ("401",), ("I10",), "Essential hypertension"),
    ICDGroup(
        "diabetes_mellitus",
        ("250",),
        ("E08", "E09", "E10", "E11", "E13"),
        "Diabetes mellitus",
    ),
    ICDGroup("acute_mi", ("410",), ("I21", "I22"), "Acute myocardial infarction"),
    ICDGroup("old_mi", ("412",), ("I252",), "Old myocardial infarction"),
    ICDGroup("ischemic_heart_disease", ("410", "411", "412", "413", "414"), ("I20", "I21", "I22", "I23", "I24", "I25")),
    ICDGroup("heart_failure", ("428",), ("I50",), "Heart failure"),
    ICDGroup("atrial_fibrillation_flutter", ("42731", "42732"), ("I48",)),
    ICDGroup("ventricular_tachycardia", ("4271",), ("I472", "I47A", "I47B")),
    ICDGroup("ventricular_fibrillation", ("42741", "42742"), ("I490", "I491")),
    ICDGroup("cardiac_arrest", ("4275",), ("I46",)),
    ICDGroup("pulmonary_embolism", ("4151",), ("I26",)),
    ICDGroup("pulmonary_hypertension", ("416",), ("I27",)),
    ICDGroup("cardiomyopathy", ("425",), ("I42", "I43")),
    ICDGroup("hypertrophic_cardiomyopathy", ("4251",), ("I421", "I422")),
    ICDGroup("dilated_cardiomyopathy", ("4254",), ("I420",)),
    # Systemic/organ-limited amyloidosis, deliberately excluding cerebral
    # amyloid angiopathy (I68.0).  These codes document the underlying
    # disease; they do not prove cardiac involvement and are labelled as such
    # by every downstream analysis.
    ICDGroup("systemic_amyloidosis", ("2773",), ("E85",)),
    # Competing causes of a thick LV wall.  I11/I13 and 402/404 identify
    # hypertensive heart (and heart/kidney) disease rather than uncomplicated
    # hypertension, which would be far too non-specific for this contrast.
    ICDGroup(
        "hypertensive_heart_disease",
        ("402", "404"),
        ("I11", "I13"),
    ),
    ICDGroup(
        "aortic_valve_disease",
        ("395", "396", "4241"),
        ("I06", "I08", "I35"),
    ),
    ICDGroup("mitral_valve_disease", ("394", "396", "4240"), ("I05", "I08", "I34")),
    ICDGroup("tricuspid_valve_disease", ("3970", "4242"), ("I07", "I08", "I36")),
    ICDGroup("pericarditis", ("420", "423"), ("I30", "I31")),
    # Effusion separately from inflammation. ``pericarditis`` pools I30
    # (acute pericarditis) with I31 (other pericardial disease), and the two
    # are different ECG questions: low voltage and electrical alternans follow
    # the fluid, not the inflammation. Until now the only reference for
    # ``qrs_alternans_vector`` was ``pericarditis_icd``, which is the wrong
    # one. 1.144 subjects with an ECG carry these codes, over 15.621
    # recordings.
    # 423.9 "unspecified disease of pericardium" is deliberately left out. It
    # would add 886 subjects, but the endpoint exists to name fluid, and an
    # unspecified pericardial disease is not evidence of any. 1.144 subjects
    # with an ECG and 15.621 recordings carry the specific codes.
    ICDGroup(
        "pericardial_effusion",
        ("4233",),
        ("I313",),
        "Pericardial effusion (non-inflammatory)",
    ),
    # Chronic lung disease and cor pulmonale, the reference the P-pulmonale
    # and right-axis criteria have never had. I27.0 is already inside
    # ``pulmonary_hypertension``; this group is about the airway disease that
    # produces the right-heart ECG pattern.
    ICDGroup(
        "copd",
        ("491", "492", "496"),
        ("J41", "J42", "J43", "J44"),
        "Chronic obstructive pulmonary disease",
    ),
    # Device *in situ*, which the implantation procedure does not capture: a
    # pacemaker put in years ago leaves no procedure code in this encounter.
    # Measured over the cohort, the status codes reach 21.712 recordings and
    # the procedure 37.763; together 57.983, and they overlap only partly.
    ICDGroup(
        "cardiac_device_status",
        ("V450", "V4501", "V4502"),
        ("Z950", "Z95810"),
        "Cardiac pacemaker or defibrillator in situ",
    ),
    # Audit 5.2.5: conduction progression. Codes resolved against the local
    # d_icd_diagnoses (MIMIC-IV 3.1): 426.0 complete, 426.12 Mobitz II,
    # 426.13 other second-degree AV block; I44.1 second-degree, I44.2
    # complete. First-degree block is not a progression target.
    ICDGroup(
        "advanced_av_block",
        ("4260", "42612", "42613"),
        ("I441", "I442"),
        "Second-degree or complete AV block",
    ),
    ICDGroup(
        "left_bundle_branch_block",
        ("4263",),
        ("I447",),
        "Left bundle branch block",
    ),
    ICDGroup("myocarditis", ("422",), ("I40", "I41")),
    ICDGroup("endocarditis", ("421",), ("I33", "I38", "I39")),
    ICDGroup("sepsis", ("038", "99591", "99592"), ("A40", "A41", "R652")),
    ICDGroup("shock", ("7855",), ("R57",)),
    ICDGroup("acute_renal_failure", ("584",), ("N17",)),
    ICDGroup("chronic_kidney_disease", ("585",), ("N18",)),
    ICDGroup("anemia", ("280", "281", "282", "283", "284", "285"), ("D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59", "D60", "D61", "D62", "D63", "D64")),
    ICDGroup("hypoxia_respiratory_failure", ("51881", "51882", "79902"), ("J96", "R0902")),
    ICDGroup("electrolyte_disorder", ("276",), ("E87",)),
)


PROCEDURE_GROUPS: tuple[ICDGroup, ...] = (
    ICDGroup(
        "pci",
        ("0066", "3601", "3602", "3605", "3606", "3607", "3609"),
        ("027",),
        "Percutaneous coronary intervention/revascularization",
    ),
    ICDGroup(
        "coronary_angiography",
        ("8855", "8856", "8857"),
        ("B210", "B211", "B212", "B213", "B214", "B215", "4A023N"),
        "Coronary angiography/cardiac catheterization",
    ),
    ICDGroup("cabg", ("3610", "3611", "3612", "3613", "3614", "3615", "3616", "3617", "3619"), ("021",), "Coronary artery bypass graft"),
    ICDGroup("pacemaker_or_icd", ("0050", "0051", "0052", "3770", "3771", "3772", "3773", "378"), ("0JH", "02H"), "Pacemaker or ICD procedure"),
    ICDGroup("cardioversion_or_defibrillation", ("9960", "9961", "9962"), ("5A2204Z",), "Cardioversion or defibrillation"),
    # Audit 5.2.5: permanent pacemaker (incl. CRT-P) implantation only, from
    # the local d_icd_procedures. ICD-9: 37.80-37.83 initial insertion, 00.50
    # CRT-P total system; replacements (37.85-37.87, 00.53) and temporary
    # pacing (37.78) are left out. ICD-10-PCS: generator insertion into chest
    # (0JH6) or abdominal (0JH8) subcutaneous tissue, open or percutaneous,
    # device 4-7 (single, single rate-responsive, dual chamber, CRT-P).
    # ``pacemaker_or_icd`` above is broader (02H and 0JH also code central
    # lines and ports) and is left as it was.
    ICDGroup(
        "permanent_pacemaker_implant",
        ("3780", "3781", "3782", "3783", "0050"),
        tuple(
            f"0JH{site}{approach}{device}Z"
            for site in ("6", "8")
            for approach in ("0", "3")
            for device in ("4", "5", "6", "7")
        ),
        "Permanent pacemaker or CRT-P implantation",
    ),
    ICDGroup("valve_procedure", ("351", "352"), ("02Q", "02R", "02U"), "Valve repair or replacement"),
)


LAB_ITEM_GROUPS: dict[str, tuple[int, ...]] = {
    "troponin_i": (51002, 52642),
    "troponin_t": (51003,),
    "ck_mb": (50908, 50911, 51580, 51595),
    "ntprobnp": (50963,),
    "potassium": (50822, 50971, 52452, 52610),
    "magnesium": (50960,),
    "total_calcium": (50893, 52034, 52035),
    "ionized_calcium": (50808, 51624),
    "creatinine": (50912, 52024, 52546),
    "hemoglobin": (50811, 51222, 51640, 51645),
    "lactate": (50813, 52442, 53154),
    "blood_gas_ph": (50820,),
    # Second wave: every itemid below was resolved against this project's own
    # d_labitems table by exact label match, not guessed. Blood fluid only --
    # the urine glucose itemids (51478, 51981) are deliberately excluded, as
    # they measure something else entirely under the same label.
    "sodium": (50983, 52623),
    "bicarbonate": (50882,),
    "phosphate": (50970,),
    "glucose": (50809, 50931, 52569),
    "hba1c": (50852,),
    "urea_nitrogen": (51006, 52647),
    "c_reactive_protein": (50889,),
    "white_blood_cells": (51301, 51755, 51756),
    "tsh": (50993,),
    "free_t4": (50995,),
    "inr": (51237, 51675),
    "prothrombin_time": (51274, 52921),
    "partial_thromboplastin_time": (51275, 52923),
    "d_dimer": (50915, 51196, 52551),
    "albumin": (50862, 53085),
}


# ICU chartevents itemids for the curated vitals/ventilation/GCS signals --
# must match vcg_biomarker/mimic/event_ingest.py's CHARTEVENTS_CURATED_ITEMIDS
# filter, verified against the actual local d_items.csv.gz label text.
ICU_CHARTEVENTS_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (220045,),
    "systolic_bp": (220050, 220179),
    "spo2": (220277,),
    "gcs_eye": (220739,),
    "gcs_verbal": (223900,),
    "gcs_motor": (223901,),
}

# Invasive haemodynamics (audit 5.2.9). Every itemid was resolved against the
# local d_items.csv.gz (MIMIC-IV 3.1, linksto=chartevents, param_type
# Numeric, category "Hemodynamics"); alarm limits (220063/220066/220072/
# 220073) are deliberately excluded. The chartevents ingest already keeps
# every numeric chart item, so these values are in the event registry.
ICU_HEMODYNAMIC_ITEMIDS: dict[str, tuple[int, ...]] = {
    "cvp": (220074,),  # Central Venous Pressure, mmHg
    "pap_systolic": (220059,),  # Pulmonary Artery Pressure systolic, mmHg
    "pap_diastolic": (220060,),  # Pulmonary Artery Pressure diastolic, mmHg
    "pap_mean": (220061,),  # Pulmonary Artery Pressure mean, mmHg
    "pcwp": (223771,),  # PCWP, mmHg
    "cardiac_output": (220088, 224842),  # CO thermodilution / CCO, L/min
    "svo2": (223772,),  # SvO2, %
}

# Physically possible ranges, used only to drop charting artefacts (999,
# negative outputs, alarm values typed into the value field). They are not
# clinical thresholds and decide no label.
ICU_HEMODYNAMIC_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "cvp": (-10.0, 50.0),
    "pap_systolic": (5.0, 150.0),
    "pap_diastolic": (0.0, 90.0),
    "pap_mean": (3.0, 110.0),
    "pcwp": (0.0, 60.0),
    "cardiac_output": (0.5, 25.0),
    "svo2": (5.0, 100.0),
}

# ICU intubation/extubation procedureevents itemids.
ICU_VENTILATION_ITEMIDS = {"intubation": 224385, "extubation": 227194}

# ICU inputevents itemids for the 5 vasopressors already covered by
# MEDICATION_PATTERNS["vasopressor"] above -- same drug concept, sharpened
# with itemid-based dose-rate/duration precision instead of a name-string
# "was this class ordered anywhere" flag.
ICU_VASOPRESSOR_ITEMIDS: tuple[int, ...] = (
    221906,  # Norepinephrine
    221289,  # Epinephrine
    221749,  # Phenylephrine
    222315,  # Vasopressin
    221662,  # Dopamine
)


MEDICATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "qt_prolonging_drug": (
        "amiodarone", "sotalol", "dofetilide", "ibutilide", "procainamide",
        "quinidine", "azithromycin", "erythromycin", "levofloxacin",
        "moxifloxacin", "haloperidol", "ziprasidone", "methadone",
        "ondansetron", "fluconazole",
    ),
    "antiarrhythmic_class_i_iii": (
        "amiodarone", "sotalol", "dofetilide", "ibutilide", "procainamide",
        "quinidine", "flecainide", "propafenone", "disopyramide",
        "lidocaine", "mexiletine",
    ),
    "beta_blocker": (
        "metoprolol", "atenolol", "carvedilol", "propranolol",
        "labetalol", "bisoprolol", "esmolol", "nadolol",
    ),
    "calcium_channel_blocker": (
        "diltiazem", "verapamil", "amlodipine", "nifedipine", "nicardipine",
    ),
    "digoxin": ("digoxin",),
    "diuretic": (
        "furosemide", "bumetanide", "torsemide", "hydrochlorothiazide",
        "chlorthalidone", "metolazone", "spironolactone",
    ),
    "vasopressor": (
        "norepinephrine", "noradrenaline", "epinephrine", "phenylephrine",
        "vasopressin", "dopamine",
    ),
    "anticoagulant": (
        "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
        "enoxaparin", "heparin",
    ),
    "antiplatelet": (
        "aspirin", "clopidogrel", "prasugrel", "ticagrelor",
        "eptifibatide", "tirofiban",
    ),
    "heparin": ("heparin", "enoxaparin"),
    "nitrate": ("nitroglycerin", "isosorbide"),
    "electrolyte_replacement": (
        "potassium chloride", "potassium phosphate", "magnesium sulfate",
        "magnesium oxide", "calcium gluconate", "calcium chloride",
    ),
    "endocarditis_antibiotic_context": (
        "vancomycin", "nafcillin", "oxacillin", "cefazolin", "ceftriaxone",
        "ampicillin", "gentamicin", "daptomycin",
    ),
    "adenosine": ("adenosine",),
}


# AV-nodal-blocking / rhythm-terminating agents relevant to acute
# tachycardia therapy response (verapamil/diltiazem already live under
# calcium_channel_blocker, so they are covered without duplication).
AV_NODAL_BLOCKING_MEDICATION_CLASSES: frozenset[str] = frozenset(
    {
        "adenosine",
        "antiarrhythmic_class_i_iii",
        "calcium_channel_blocker",
        "beta_blocker",
        "digoxin",
    }
)


TYPICAL_ENDOCARDITIS_ORGANISMS: tuple[str, ...] = (
    "staphylococcus aureus",
    "streptococcus viridans",
    "streptococcus bovis",
    "streptococcus gallolyticus",
    "enterococcus faecalis",
    "enterococcus faecium",
    "haemophilus",
    "aggregatibacter",
    "cardiobacterium",
    "eikenella",
    "kingella",
)


ECHO_MEASUREMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "lvef_3d": ("lvef_3d",),
    "lvef_biplane": ("biplane_lvef",),
    "lvef_generic": ("lvef",),
    "lv_wall_category": ("lv_wall_thickness",),
    "septal_thickness_cm": ("septal_thickness",),
    "posterior_wall_thickness_cm": ("inf_lat_thickness",),
    "lv_internal_diameter_diastole_cm": (
        "lvedd",
        "lvid_d",
        "lv_dimen_diastole",
    ),
    "lv_internal_diameter_systole_cm": (
        "lvesd",
        "lvid_s",
        "lv_dimen_systole",
    ),
    "lv_mass_index": ("lv_mass_index", "lvmi"),
    "la_volume_index": ("la_vol_index", "lavi"),
    "la_volume_ml": ("la_vol",),
    "la_size_category": ("la_size",),
    "ra_size_category": ("ra_size",),
    "rv_basal_diameter_cm": ("rv_diam",),
    "rv_size_category": ("rv_chamber_size",),
    "rv_function_category": ("rv_function",),
    "tapse_cm": ("tapse",),
    "direct_estimated_pa_pressure_mmhg": (
        "tv_est_pa_press_min",
        "rvsp",
        "pasp",
        "rv_systolic_pressure",
    ),
    "tr_pressure_gradient_mmhg": ("tr_mmhg",),
    "tr_velocity_m_per_s": ("tr_velocity", "tr_peak_velocity"),
    "ivc_ra_pressure_category": ("ivc",),
    "pulmonary_hypertension_category": ("phtn_severity",),
    "aortic_stenosis": ("aortic_stenosis",),
    "aortic_regurgitation": ("aortic_regurg",),
    "mitral_stenosis": ("mitral_stenosis", "mitral_stenosis2"),
    "mitral_regurgitation": ("mitral_regurg",),
    "tricuspid_regurgitation": ("tricuspid_regurg",),
    "pulmonic_stenosis": ("pulm_stenosis",),
    "pulmonic_regurgitation": ("pulm_regurg",),
    "pericardial_effusion": ("pericardial_effusion",),
    "mean_e_to_eprime": ("mean_e_to_eprime",),
    "septal_e_to_eprime": ("septal_e_to_eprime",),
    "lateral_e_to_eprime": ("lateral_e_to_eprime",),
}


def classify_icd(code: str, version: int, groups: Iterable[ICDGroup] = ICD_GROUPS) -> Optional[str]:
    for group in groups:
        if group.matches(code, version):
            return group.group_id
    return None


def medication_classes(name: str) -> tuple[str, ...]:
    normalized = name.casefold()
    return tuple(
        group
        for group, patterns in MEDICATION_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    )


def icd_case_sql(
    code_expression: str,
    version_expression: str,
    groups: Iterable[ICDGroup] = ICD_GROUPS,
) -> str:
    clauses: list[str] = []
    for group in groups:
        for version, prefixes in ((9, group.icd9_prefixes), (10, group.icd10_prefixes)):
            for prefix in prefixes:
                escaped = prefix.replace("'", "''")
                clauses.append(
                    f"WHEN {version_expression} = {version} "
                    f"AND {code_expression} LIKE '{escaped}%' "
                    f"THEN '{group.group_id}'"
                )
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def lab_group_case_sql(item_expression: str = "itemid") -> str:
    clauses = []
    for group, itemids in LAB_ITEM_GROUPS.items():
        values = ", ".join(str(itemid) for itemid in itemids)
        clauses.append(f"WHEN {item_expression} IN ({values}) THEN '{group}'")
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def medication_case_sql(name_expression: str) -> str:
    clauses: list[str] = []
    for group, patterns in MEDICATION_PATTERNS.items():
        conditions = []
        for pattern in patterns:
            escaped = pattern.replace("'", "''")
            conditions.append(
                f"LOWER({name_expression}) LIKE '%{escaped}%'"
            )
        condition = " OR ".join(conditions)
        clauses.append(f"WHEN ({condition}) THEN '{group}'")
    return "CASE " + " ".join(clauses) + " ELSE NULL END"
