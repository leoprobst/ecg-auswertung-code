from __future__ import annotations

from .models import LabelDefinition, TargetKind


LABEL_RULE_VERSION = "2.9.0"
OUTCOME_BLIND_INDEX_POLICY = (
    "first eligible ECG per subject, selected by ECG time and study_id "
    "without using label state or outcome timing"
)
MI_REFERENCE = "https://academic.oup.com/eurheartj/article/40/3/237/5079081"
ECHO_REFERENCE = (
    "https://www.asecho.org/wp-content/uploads/2015/01/"
    "ChamberQuantification2015.pdf"
)


def default_label_definitions() -> tuple[LabelDefinition, ...]:
    definitions = [
        _label(
            "troponin_positive_any",
            "Troponin above assay reference limit",
            TargetKind.DIAGNOSTIC,
            ("labevents", "d_labitems"),
            "closest eligible ECG to first positive troponin; prefer ECG before draw",
            -6,
            12,
            "same-assay troponin value > row-specific upper reference limit",
            "troponin measured in-window and all results <= upper reference limit",
            panel="primary",
            references=(MI_REFERENCE,),
        ),
        _label(
            "acute_troponin_change_proxy",
            "Acute troponin change proxy",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents",),
            "last ECG before first qualifying serial change",
            -6,
            12,
            "two same-assay values with URL crossing or >=20% relative change and one value > URL",
            "serial same-assay troponins available without qualifying change or elevation",
            references=(MI_REFERENCE,),
            panel="primary",
        ),
        _label(
            "troponin_rise_from_undetectable",
            "Troponin rise from undetectable",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents",),
            "last ECG before first qualifying serial change",
            -6,
            12,
            (
                "first same-assay result reported below a detection limit "
                "at or below the URL, later result certainly above the URL"
            ),
            (
                "first result below the detection limit and no later result "
                "that could exceed the URL"
            ),
            exclusions=(
                "series starting with a detectable value are outside the "
                "phenotype, not negatives",
            ),
            references=(MI_REFERENCE,),
        ),
        _label(
            "acute_mi_high_specificity",
            "Acute MI high-specificity retrospective composite",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents", "diagnoses_icd", "procedures_icd"),
            "last ECG before qualifying troponin/PCI event",
            -6,
            24,
            (
                "acute troponin change + acute-MI ICD in encounter + "
                "PCI recorded on the ECG calendar date"
            ),
            (
                "serial troponin without acute-MI/coronary-procedure evidence, "
                "including explicit non-coronary demand-context controls"
            ),
            exclusions=(
                "isolated myocardial injury without ischemic evidence",
                (
                    "procedures_icd has date precision; same-date PCI is not "
                    "claimed as a minute-resolved 24-hour interval"
                ),
            ),
            references=(MI_REFERENCE,),
            panel="primary",
        ),
        _label(
            "acute_mi_probable",
            "Probable acute MI retrospective composite",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents", "diagnoses_icd", "procedures_icd", "emar"),
            "last ECG before qualifying event",
            -6,
            72,
            (
                "acute injury + acute-MI ICD + cath/PCI on calendar dates "
                "0-3 days after ECG or ACS treatment context"
            ),
            (
                "serial troponin without acute-MI/coronary-procedure evidence; "
                "partial composites are indeterminate"
            ),
            references=(MI_REFERENCE,),
        ),
        _label(
            "omi_revascularization_proxy",
            "Acute coronary revascularization proxy",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents", "procedures_icd", "edstays", "triage"),
            "last pre-PCI ECG in chest-pain/ACS context",
            -6,
            24,
            (
                "acute injury + chest-pain/ACS context + PCI on the same or "
                "next calendar date"
            ),
            (
                "serial troponin without acute-MI/coronary-procedure evidence; "
                "partial composites are indeterminate"
            ),
            exclusions=(
                "no angiographic occlusion information is available",
                "PCI timing is day-resolved in procedures_icd",
            ),
            references=(MI_REFERENCE,),
            panel="primary",
        ),
        _label(
            "myocardial_injury_demand_context",
            "Myocardial injury with demand context",
            TargetKind.RETROSPECTIVE_PROXY,
            ("labevents", "diagnoses_icd", "emar"),
            "closest ECG to positive troponin",
            -6,
            12,
            "troponin positive + sepsis/shock/hypoxia/tachyarrhythmia/anemia/renal context + no PCI",
            "troponin assessed without this context",
        ),
    ]

    definitions.extend(_echo_definitions())
    definitions.extend(_electrolyte_definitions())
    definitions.extend(_predictive_and_secondary_definitions())
    return tuple(definitions)


# Audit D7: endpoints that repeat another one. They keep their labels for
# description and old artefacts, but get no test of their own: the RWT pair
# counts the same studies (14,812 / 7,633), and preserved LVEF is the exact
# complement of reduced LVEF, so its AUC is 1 - AUC. Each would double a
# hypothesis in the multiplicity family and the run time.
ENDPOINT_ALIASES: dict[str, tuple[str, str]] = {
    "concentric_remodeling_rwt": ("concentric_remodeling_proxy", "identical"),
    "preserved_lvef": ("reduced_lvef", "complement"),
}


def definitions_by_id() -> dict[str, LabelDefinition]:
    return {definition.label_id: definition for definition in default_label_definitions()}


def _echo_definitions() -> list[LabelDefinition]:
    rows = [
        ("echo_lvh_strict", "Echo LVH by LV mass index (ASE)", "LVMI >115 g/m2 in men or >95 g/m2 in women; reported LVMI, else Devereux linear mass over echo BSA", "LVMI at or below the sex-specific threshold"),
        ("echo_lvh_sensitive", "Sensitive echo LVH", "LVMI above threshold or explicit mild-or-worse LVH", "explicit normal LVMI/wall category"),
        ("lvh_concentric_ase", "Concentric LVH (ASE geometry)", "LVMI above the sex-specific threshold and RWT >0.42", "normal geometry: LVMI at or below threshold and RWT <=0.42"),
        ("lvh_eccentric_ase", "Eccentric LVH (ASE geometry)", "LVMI above the sex-specific threshold and RWT <=0.42", "normal geometry: LVMI at or below threshold and RWT <=0.42"),
        ("lv_wall_thickening", "LV wall thickening proxy", "septal or posterior wall >=1.2 cm or explicit thickening", "both observed walls below threshold or explicit normal"),
        ("lv_dilatation_proxy", "LV dilatation proxy", "LVIDd >5.8 cm in men or >5.2 cm in women", "sex-specific LVIDd at or below threshold"),
        ("concentric_remodeling_proxy", "Elevated relative wall thickness (geometry proxy)", "relative wall thickness >0.42; LV mass not constrained, does not distinguish concentric remodeling from concentric LVH", "complete geometry with RWT <=0.42; does not exclude eccentric LVH"),
        ("reduced_lvef_strict", "LVEF below 40%", "preferred LVEF <40%", "preferred LVEF >=40%"),
        ("reduced_lvef", "LVEF below 50%", "preferred LVEF <50%", "preferred LVEF >=50%"),
        ("preserved_lvef", "Preserved LVEF", "preferred LVEF >=50%", "preferred LVEF <50%"),
        ("echo_lae", "Echo left atrial enlargement", "LAVI >34 ml/m2 or explicit LAE category", "LAVI <=34 or explicit normal category"),
        ("echo_rae", "Echo right atrial enlargement", "explicit mild-or-worse RAE category", "explicit normal RA category"),
        ("echo_rv_dilation", "Echo RV dilation", "RV basal diameter >=4.2 cm or explicit RVE", "diameter <4.2 cm or explicit normal"),
        ("echo_rv_systolic_dysfunction", "Echo RV systolic dysfunction", "TAPSE <1.7 cm or explicit global dysfunction", "TAPSE >=1.7 cm or explicit normal function"),
        ("elevated_rvsp_or_pasp", "Elevated RVSP/PASP", "direct RVSP/PASP or validated TRV+RAP estimate above configured threshold", "available pressure estimate below threshold"),
        ("echo_pulmonary_hypertension", "Structured echo pulmonary hypertension", "explicit mild-or-worse pulmonary-hypertension category", "explicit normal/high-normal or PH-unlikely category"),
        ("regional_wall_motion_abnormality", "Regional wall motion abnormality with complete negative coverage", "resting wall-motion section names a hypokinetic, akinetic or dyskinetic segment without conflicting normal evidence", "all 16 named core resting segments explicitly normal or hyperdynamic; optional apical cap must be normal if present; duplicates do not add coverage; stress, conflicting, blank and unvisualized evidence cannot certify a negative"),
        ("regional_wall_motion_documentation_proxy", "Wall motion documentation phenotype (exploratory)", "an abnormal segment is documented", "a section exists without a recorded abnormality; this is not exclusion of disease"),
        ("regional_wall_motion_abnormality_explicit", "Regional wall motion abnormality, explicit normals only", "visible segment hypokinetic, akinetic or dyskinetic", "at least one segment explicitly normal and none abnormal; strict sensitivity arm, underpowered because normals are rarely written down"),
        # Audit D8: readers fill the AS field mostly when the valve is
        # abnormal, so the controls are graded mild/sclerotic valves
        # (1,267 against 1,366), not healthy ones. The Vmax endpoint below
        # (1,417 against 20,343) is the primary AS contrast.
        ("aortic_stenosis", "Moderate-or-worse aortic stenosis (graded-valve controls)", "normalized severity moderate or severe", "explicit none/minimal/mild in the AS field; controls are mostly mild or sclerotic valves, not healthy valves"),
        ("aortic_regurgitation", "Moderate-or-worse aortic regurgitation", "normalized severity moderate or severe", "explicit physiologic/trace/mild"),
        ("mitral_stenosis", "Moderate-or-worse mitral stenosis", "normalized severity moderate or severe", "explicit none/minimal/mild"),
        ("mitral_regurgitation", "Moderate-or-worse mitral regurgitation", "normalized severity moderate or severe", "explicit physiologic/trace/mild"),
        ("tricuspid_regurgitation", "Moderate-or-worse tricuspid regurgitation", "normalized severity moderate or severe", "explicit physiologic/trace/mild"),
        ("significant_valvular_disease", "Significant valvular disease", "any valve moderate-or-worse", "all assessed valves below moderate"),
        ("aortic_stenosis_severe", "Severe aortic stenosis", "normalized severity severe", "explicit severity below severe"),
        ("aortic_regurgitation_severe", "Severe aortic regurgitation", "normalized severity severe", "explicit severity below severe"),
        ("mitral_stenosis_severe", "Severe mitral stenosis", "normalized severity severe", "explicit severity below severe"),
        ("mitral_regurgitation_severe", "Severe mitral regurgitation", "normalized severity severe", "explicit severity below severe"),
        ("tricuspid_regurgitation_severe", "Severe tricuspid regurgitation", "normalized severity severe", "explicit severity below severe"),
        ("significant_valvular_disease_severe", "Any severe valvular disease", "any valve severe", "all assessed valves below severe"),
        # Quantitative diastolic, aortic-haemodynamic and linear-geometry
        # endpoints. These are the reference standards the P-wave and P-loop
        # markers were always missing: an atrial marker screened only against
        # valve categories and LVEF has no anatomically matched target.
        ("diastolic_relaxation_abnormality", "Impaired relaxation filling pattern", "E/A <0.8 with deceleration time >200 ms", "E/A 0.8-2.0 with deceleration time 160-200 ms"),
        ("restrictive_filling_pattern", "Restrictive filling pattern", "E/A >2.0 with deceleration time <160 ms", "E/A <=2.0 or deceleration time >=160 ms with both measured"),
        ("elevated_filling_pressure_proxy", "Elevated LV filling pressure proxy", "E/e-prime >14", "E/e-prime <=14"),
        ("severe_aortic_stenosis_velocity", "Severe aortic stenosis by peak velocity", "aortic peak velocity >=4.0 m/s", "aortic peak velocity <4.0 m/s"),
        ("moderate_aortic_stenosis_velocity", "Moderate-or-worse aortic stenosis by peak velocity", "aortic peak velocity >=3.0 m/s", "aortic peak velocity <3.0 m/s"),
        ("la_dilatation_linear", "Left atrial dilatation by linear dimension", "LA anteroposterior dimension >4.0 cm", "LA anteroposterior dimension <=4.0 cm"),
        ("ra_dilatation_linear", "Right atrial dilatation by length", "RA length >5.3 cm", "RA length <=5.3 cm"),
        ("concentric_remodeling_rwt", "Elevated echo relative wall thickness (geometry proxy)", "echo relative wall thickness >0.42; LV mass not constrained, not confirmed concentric remodeling", "echo relative wall thickness <=0.42; does not exclude eccentric LVH"),
        ("aortic_root_dilatation", "Aortic root dilatation", "aortic sinus diameter >4.0 cm", "aortic sinus diameter <=4.0 cm"),
    ]
    return [
        _label(
            label_id,
            display,
            TargetKind.RETROSPECTIVE_PROXY if label_id in {
                "regional_wall_motion_documentation_proxy",
                "regional_wall_motion_abnormality_explicit",
            } else TargetKind.DIAGNOSTIC,
            ("structured-measurement",),
            "nearest non-conflicting resting TTE; prefer earlier study on equal distance",
            -24 * 30,
            24 * 30,
            positive,
            negative,
            exclusions=("TEE and stress echo are sensitivity arms",),
            references=(ECHO_REFERENCE,),
            panel="primary" if label_id in {
                "echo_lvh_strict", "reduced_lvef", "echo_lae",
                "echo_rv_dilation", "echo_rv_systolic_dysfunction",
            } else "secondary",
        )
        for label_id, display, positive, negative in rows
    ]


def _electrolyte_definitions() -> list[LabelDefinition]:
    rows = [
        ("qtc_gt_480", "Machine-derived QTc above 480 ms", "machine QTc Fridericia >480 ms", "machine QTc Fridericia <=480 ms"),
        ("qtc_gt_500", "Machine-derived QTc above 500 ms", "machine QTc Fridericia >500 ms", "machine QTc Fridericia <=500 ms"),
        ("hypokalemia_near_ecg", "Hypokalemia near ECG", "potassium <3.5 mmol/L", "potassium 3.5-5.5 mmol/L"),
        ("severe_hypokalemia_near_ecg", "Severe hypokalemia near ECG", "potassium <3.0 mmol/L", "potassium >=3.0 mmol/L"),
        ("hyperkalemia_near_ecg", "Hyperkalemia near ECG", "potassium >5.5 mmol/L", "potassium 3.5-5.5 mmol/L"),
        ("severe_hyperkalemia_near_ecg", "Severe hyperkalemia near ECG", "potassium >=6.5 mmol/L", "potassium <6.5 mmol/L"),
        ("hypomagnesemia_near_ecg", "Hypomagnesemia near ECG", "unit-validated magnesium below row lower reference", "measured magnesium within reference"),
        ("hypocalcemia_near_ecg", "Hypocalcemia near ECG", "unit-validated total or ionized calcium below row lower reference", "measured calcium within reference"),
    ]
    return [
        _label(
            label_id,
            display,
            # The qtc_* rows read the recorder's own QT measurement of the
            # same ECG the marker is computed from -- same signal, same
            # T-offset problem, no independent reference. They stay in the
            # catalog as an analytical-concordance check and are barred from
            # diagnostic claims.
            TargetKind.ANALYTICAL_CONCORDANCE
            if label_id.startswith("qtc_")
            else TargetKind.DIAGNOSTIC,
            (
                ("machine_measurements",)
                if label_id.startswith("qtc_")
                else ("labevents", "d_labitems")
            ),
            "closest measurement to ECG",
            -6,
            6,
            positive,
            negative,
            panel="primary" if "kalemia" in label_id else "secondary",
        )
        for label_id, display, positive, negative in rows
    ]


def _predictive_and_secondary_definitions() -> list[LabelDefinition]:
    definitions: list[LabelDefinition] = []
    for days in (30, 90, 365):
        definitions.append(
            _label(
                f"new_onset_af_{days}d" if days < 365 else "new_onset_af_1y",
                f"New-onset AF within {days} days",
                TargetKind.PREDICTIVE,
                ("diagnoses_icd", "machine_measurements", "admissions"),
                "first eligible sinus ECG without current or prior AF/AFlutter",
                0,
                24 * days,
                "future AF/AFlutter ICD or machine-reported ECG",
                "no AF plus documented observation through horizon",
                exclusions=("current AF/AFlutter", "known prior AF/AFlutter", "death before horizon"),
                panel="secondary",
                estimand=(
                    "complete_followup_binary_risk_among_subjects_observed_"
                    "through_the_horizon"
                ),
                selection_mechanism=(
                    "controls_require_documented_contact_through_the_horizon "
                    "and_deaths_inside_the_horizon_are_dropped_as_"
                    "indeterminate; both condition the cohort on post-index "
                    "observation, which is associated with health status"
                ),
                primary_analysis_arm="time_to_event",
            )
        )

    # Audit 5.2.5: conduction progression, built like new-onset AF.
    definitions.append(
        _label(
            "conduction_progression_1y",
            "Conduction progression within 1 year",
            TargetKind.PREDICTIVE,
            ("diagnoses_icd", "procedures_icd", "admissions", "machine_measurements"),
            "first eligible ECG without known LBBB, high-grade AV block or pacemaker",
            0,
            24 * 365,
            (
                "later admission coded with second-degree/complete AV block or "
                "LBBB, or a first permanent pacemaker/CRT-P implanted on a "
                "later date"
            ),
            "none of these plus documented observation through one year",
            exclusions=(
                "LBBB or pacing in the index ECG report",
                "prior LBBB/high-grade AV block, device in situ or earlier pacemaker",
                "death before the horizon",
            ),
            panel="secondary",
            estimand=(
                "complete_followup_binary_risk_among_subjects_observed_"
                "through_the_horizon"
            ),
            selection_mechanism=(
                "controls_require_documented_contact_through_the_horizon "
                "and_deaths_inside_the_horizon_are_dropped_as_"
                "indeterminate; both condition the cohort on post-index "
                "observation, which is associated with health status"
            ),
            primary_analysis_arm="time_to_event",
        )
    )

    rows = [
        ("old_mi_icd", "Prior myocardial infarction", ("diagnoses_icd",), "old MI ICD before or during index encounter"),
        ("mi_or_revascularization_history", "MI or revascularization history", ("diagnoses_icd", "procedures_icd"), "prior MI, PCI or CABG"),
        ("heart_failure_icd", "Heart failure ICD context", ("diagnoses_icd",), "heart-failure ICD in encounter"),
        ("elevated_ntprobnp", "Elevated NT-proBNP", ("labevents",), "NT-proBNP above row upper reference"),
        ("acute_decompensated_heart_failure_proxy", "ADHF proxy", ("diagnoses_icd", "labevents", "emar"), "HF ICD + elevated NT-proBNP + dyspnea/diuretic context; controls require measured non-elevated NT-proBNP and no HF ICD"),
        ("pulmonary_embolism_icd", "Pulmonary embolism ICD context", ("diagnoses_icd",), "PE ICD in encounter"),
        ("pulmonary_hypertension_icd", "Pulmonary hypertension ICD context", ("diagnoses_icd",), "PH ICD in encounter"),
        ("right_heart_strain_proxy", "Right-heart strain proxy", ("structured-measurement", "diagnoses_icd"), "RV dilation/dysfunction near PE/PH encounter; controls require assessed normal RV size and function"),
        ("cardiomyopathy_icd", "Cardiomyopathy ICD context", ("diagnoses_icd",), "cardiomyopathy ICD in encounter"),
        ("pericarditis_icd", "Pericarditis ICD context", ("diagnoses_icd",), "pericarditis ICD in encounter"),
        # Effusion apart from inflammation: low voltage and electrical
        # alternans follow the fluid. Until this existed the only reference
        # for qrs_alternans_vector was pericarditis_icd, which is a different
        # condition.
        ("pericardial_effusion", "Pericardial effusion ICD context", ("diagnoses_icd",), "pericardial-effusion ICD in encounter"),
        # The classically ECG-diagnosable cardiomyopathy: low voltage with a
        # pseudo-infarct pattern. classical_low_voltage_qrs had no reference.
        # The codes document systemic amyloidosis and do not prove cardiac
        # involvement, which is why the display name says context.
        ("cardiac_amyloidosis", "Systemic amyloidosis ICD context", ("diagnoses_icd",), "systemic-amyloidosis ICD in encounter; cardiac involvement is not proven by the code"),
        ("hypertrophic_cardiomyopathy_icd", "Hypertrophic cardiomyopathy ICD context", ("diagnoses_icd",), "HCM ICD in encounter"),
        # Gives classical_p_pulmonale and the right-axis rules a reference.
        ("copd_icd", "COPD ICD context", ("diagnoses_icd",), "COPD ICD in encounter"),
        # A paced ventricle changes every QRS-derived marker: measured over
        # 597 recordings, QRS duration separates paced from unpaced at AUC
        # 0,884 and the spatial QRS-T angle at 0,814. Without this endpoint
        # pacing is neither a target nor a covariate anywhere in the analysis.
        ("pacemaker_carrier", "Cardiac device in situ", ("diagnoses_icd", "procedures_icd"), "device status code or pacemaker/ICD procedure; recordings the machine report calls paced without either are withheld as indeterminate"),
        # The list carried only the three prospective new-onset AF endpoints,
        # so "this patient already has AF" was an exclusion and never a target.
        # Read from the ICD history alone, never from the machine report: the
        # report reads the same tracing the markers are computed from.
        ("prevalent_atrial_fibrillation", "Documented atrial fibrillation before the ECG", ("diagnoses_icd",), "AF/AFlutter ICD dated at or before the index ECG; the machine report is deliberately not used"),
        ("myocarditis_icd", "Myocarditis ICD context", ("diagnoses_icd", "labevents"), "myocarditis ICD with myocardial injury context; ICD-only cases are indeterminate"),
        ("endocarditis_high_confidence_proxy", "High-confidence endocarditis proxy", ("diagnoses_icd", "microbiologyevents", "procedures_icd"), "endocarditis ICD + typical organism + therapy/valve procedure context; partial evidence is indeterminate"),
        ("vt_vf_icd_or_event", "VT/VF event context", ("diagnoses_icd", "procedures_icd"), "VT/VF ICD or cardioversion/defibrillation"),
        ("cardiac_arrest", "Cardiac arrest", ("diagnoses_icd",), "cardiac-arrest ICD in encounter"),
        ("in_hospital_mortality", "In-hospital mortality", ("admissions",), "hospital_expire_flag=1"),
        ("mortality_30d", "30-day mortality", ("patients",), "date of death within 30 days after index ECG"),
        ("qt_prolonging_drug_exposure", "QT-prolonging drug exposure", ("emar", "prescriptions", "pharmacy", "pyxis", "medrecon"), "actual administration within -24/+2h or active order spanning ECG"),
        ("antiarrhythmic_exposure_near_ecg", "Class I/III antiarrhythmic exposure", ("emar", "prescriptions", "pharmacy", "pyxis", "medrecon"), "actual administration within -48/+2h or active order"),
        ("digoxin_exposure_near_ecg", "Digoxin exposure", ("emar", "prescriptions", "pharmacy", "pyxis", "medrecon"), "actual administration within -48/+2h or active order"),
    ]
    windows: dict[str, tuple[float | None, float | None]] = {
        "old_mi_icd": (None, 0),
        "mi_or_revascularization_history": (None, 0),
        "prevalent_atrial_fibrillation": (None, 0),
        "elevated_ntprobnp": (-6, 12),
        "acute_decompensated_heart_failure_proxy": (-6, 12),
        "right_heart_strain_proxy": (-24 * 7, 24 * 7),
        "in_hospital_mortality": (0, None),
        "mortality_30d": (0, 24 * 30),
        "qt_prolonging_drug_exposure": (-24, 2),
        "antiarrhythmic_exposure_near_ecg": (-48, 2),
        "digoxin_exposure_near_ecg": (-48, 2),
    }
    index_policies = {
        "old_mi_icd": "earliest eligible ECG after available prior-history assessment",
        "mi_or_revascularization_history": (
            "earliest eligible ECG after available prior-history assessment"
        ),
        "elevated_ntprobnp": (
            "ECG nearest the qualifying NT-proBNP result, positive cases first"
        ),
        "acute_decompensated_heart_failure_proxy": (
            "ECG nearest qualifying NT-proBNP in the linked encounter"
        ),
        "right_heart_strain_proxy": (
            "ECG nearest the qualifying TTE in a linked PE/PH encounter"
        ),
        "in_hospital_mortality": "first eligible ECG in the linked admission",
        "mortality_30d": "first eligible ECG with 30-day outcome ascertainment",
        "qt_prolonging_drug_exposure": (
            "first ECG with complete source-specific exposure assessment"
        ),
        "antiarrhythmic_exposure_near_ecg": (
            "first ECG with complete source-specific exposure assessment"
        ),
        "digoxin_exposure_near_ecg": (
            "first ECG with complete source-specific exposure assessment"
        ),
    }
    negative_rules = {
        "old_mi_icd": (
            "available prior encounter history without old-MI evidence"
        ),
        "mi_or_revascularization_history": (
            "available prior encounter history without MI, PCI or CABG evidence"
        ),
        "elevated_ntprobnp": (
            "NT-proBNP measured in -6/+12 h and at or below row-specific URL"
        ),
        "acute_decompensated_heart_failure_proxy": (
            "NT-proBNP measured in -6/+12 h at or below URL and no HF ICD; "
            "partial composites are indeterminate"
        ),
        "right_heart_strain_proxy": (
            "PE/PH context with TTE within +/-7 days and both RV size and "
            "function explicitly assessed as normal"
        ),
        "myocarditis_icd": (
            "linked encounter without myocarditis ICD; ICD-only cases are "
            "indeterminate"
        ),
        "endocarditis_high_confidence_proxy": (
            "linked encounter without endocarditis ICD; partial composites "
            "are indeterminate"
        ),
        "in_hospital_mortality": (
            "linked admission with hospital_expire_flag=0"
        ),
        "mortality_30d": (
            "documented survival beyond 30 days; incomplete follow-up is "
            "unavailable"
        ),
        "qt_prolonging_drug_exposure": (
            "medication sources assessed without qualifying -24/+2 h exposure"
        ),
        "antiarrhythmic_exposure_near_ecg": (
            "medication sources assessed without qualifying -48/+2 h exposure"
        ),
        "digoxin_exposure_near_ecg": (
            "medication sources assessed without qualifying -48/+2 h exposure"
        ),
    }
    for label_id, display, sources, positive in rows:
        target_kind = (
            TargetKind.PREDICTIVE
            if label_id in {"in_hospital_mortality", "mortality_30d"}
            else TargetKind.RETROSPECTIVE_PROXY
        )
        definitions.append(
            _label(
                label_id,
                display,
                target_kind,
                sources,
                index_policies.get(
                    label_id,
                    "earliest eligible ECG in the deterministically linked encounter",
                ),
                windows.get(label_id, (None, None))[0],
                windows.get(label_id, (None, None))[1],
                positive,
                negative_rules.get(
                    label_id,
                    "linked encounter assessed without positive evidence",
                ),
            )
        )
    return definitions


def _label(
    label_id: str,
    display_name: str,
    target_kind: TargetKind,
    source_tables: tuple[str, ...],
    index_policy: str,
    before_hours: float | None,
    after_hours: float | None,
    positive_rule: str,
    negative_rule: str,
    *,
    exclusions: tuple[str, ...] = (),
    references: tuple[str, ...] = (),
    panel: str = "secondary",
    estimand: str = "",
    selection_mechanism: str = "",
    primary_analysis_arm: str = "binary",
) -> LabelDefinition:
    return LabelDefinition(
        label_id=label_id,
        display_name=display_name,
        target_kind=target_kind,
        source_tables=source_tables,
        # Source evidence may still be linked to each ECG according to the
        # caller's rule, but the patient-level cohort index is always chosen
        # without looking at that evidence or its outcome state.
        index_policy=OUTCOME_BLIND_INDEX_POLICY,
        window_before_hours=before_hours,
        window_after_hours=after_hours,
        positive_rule=positive_rule,
        negative_rule=negative_rule,
        exclusions=exclusions,
        estimand=estimand,
        selection_mechanism=selection_mechanism,
        primary_analysis_arm=primary_analysis_arm,
        confidence_rule="high=direct complete evidence; medium=structured proxy; low=single retrospective code",
        rule_version=LABEL_RULE_VERSION,
        references=references,
        panel=panel,
    )
