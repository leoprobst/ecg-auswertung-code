from __future__ import annotations

from typing import Iterable, Optional

from .base import BiomarkerDefinition, BiomarkerRegistry


DATA_DRIVEN_CUTOFF_NOTE = (
    "Exploratory VCG candidate: no fixed clinical cutoff is assigned here. "
    "A pathologic range is pathology-specific and must be estimated in the "
    "threshold sweep, with direction taken from SweepResult.direction and "
    "validation status reported before diagnostic use."
)


def _category_for(family: str) -> str:
    if family == "Beat-Variabilitaet":
        return "rhythm"
    if family.startswith("Klassische ECG"):
        return "ecg"
    if family == "Experimentell/Topologisch":
        return "exploratory"
    if family == "Literaturbasierte Ergaenzung":
        return "literature"
    if family.startswith("Loop-"):
        return "loop"
    if family.startswith("Mathematische Loop-Morphologie"):
        return "loop"
    if family in ("P-Loop Morphologie extra", "T-Loop Morphologie extra", "P-QRS-T Loop-Beziehungen"):
        return "loop"
    if family in ("Qualitaet/Elektroden", "Signalbasis"):
        return "quality"
    if family in ("Zeitintervalle", "ST-Zeitpunkte", "R-Wellen-Progression"):
        return "ecg"
    if family in ("Manuelle Review",):
        return "review"
    if family in ("Morphologische Signalverarbeitung", "PCA/SVD", "Template/Distanz"):
        return "signal"
    if family in ("Pathophysiologische Surrogate",):
        return "pathophysiology"
    if family in ("QRS-T/Repolarisation", "Ventricular Gradient"):
        return "repolarization"
    if family in ("Zusaetzliche VCG-Idee",):
        return "exploratory"
    return "vcg"


def _aliases_for(name: str) -> list[str]:
    aliases: list[str] = []
    if name.endswith("_max/min"):
        stem = name[: -len("_max/min")]
        aliases.extend([f"{stem}_max", f"{stem}_min"])
    if name.endswith("_x/y/z/magnitude/axis"):
        stem = name[: -len("_x/y/z/magnitude/axis")]
        aliases.extend([f"{stem}_{part}" for part in ["x", "y", "z", "magnitude", "axis"]])
    if name == "tpeak_vector / tend_vector / j_point_vector":
        aliases.extend(["tpeak_vector", "tend_vector", "j_point_vector"])
    if name == "rr_entropy/irregularity_index":
        aliases.extend(["rr_entropy", "rr_irregularity_index"])
    if name == "rr_mean_ms/sd_ms/rmssd_ms/pnn50":
        aliases.extend(["rr_mean_ms", "rr_sd_ms", "rr_rmssd_ms", "rr_pnn50"])
    if " / " in name:
        aliases.extend(part.strip().replace(" ", "_") for part in name.split(" / "))
    if name.endswith("_positive/negative/abs_peak"):
        stem = name[: -len("_positive/negative/abs_peak")]
        aliases.extend([f"{stem}_positive", f"{stem}_negative", f"{stem}_abs_peak"])
    if name.endswith("_lambda1/lambda2/lambda3"):
        stem = name[: -len("_lambda1/lambda2/lambda3")]
        aliases.extend([f"{stem}_lambda1", f"{stem}_lambda2", f"{stem}_lambda3"])
    if name.endswith("_plane_normal_x/y/z/azimuth/elevation"):
        stem = name[: -len("_plane_normal_x/y/z/azimuth/elevation")]
        aliases.extend([
            f"{stem}_plane_normal_x",
            f"{stem}_plane_normal_y",
            f"{stem}_plane_normal_z",
            f"{stem}_plane_normal_azimuth",
            f"{stem}_plane_normal_elevation",
        ])
    if name.endswith("_radial_skewness/kurtosis"):
        stem = name[: -len("_radial_skewness/kurtosis")]
        aliases.extend([f"{stem}_radial_skewness", f"{stem}_radial_kurtosis"])
    if "/" in name and name.startswith("<seg>_"):
        left, right = name.split("/", 1)
        prefix = left.rsplit("_", 1)[0]
        aliases.extend([left, f"{prefix}_{right}"])
    return sorted(set(a for a in aliases if a != name))


def _unit_for_name(name: str, formula: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    lower_name = name.lower()
    lower_formula = formula.lower()

    if lower_name.endswith("_max/min") or "peak_to_peak" in lower_name:
        return "mV"
    if "positive_area" in lower_name or "negative_area" in lower_name:
        return "mV*s"
    if any(token in lower_name for token in ["ratio", "fraction", "index", "entropy", "surrogate"]):
        return "dimensionless"
    if "peak_count" in lower_name or "self_crossings" in lower_name:
        return "count"
    if "angle" in lower_name or lower_name.endswith("_axis") or "azimuth" in lower_name or "elevation" in lower_name:
        return "degrees"
    if "rr_" in lower_name and ("_ms" in lower_name or "rmssd" in lower_name):
        return "mixed"
    if lower_name.endswith("_vector") or "_vector_" in lower_name:
        return "mixed" if "/" in lower_name else "mV"
    if "wasserstein" in lower_name or "distance" in lower_name:
        return "mV"
    if "work" in lower_name or "energy_transfer" in lower_name:
        return "dimensionless" if "transfer" in lower_name else "mV^2"
    if "integral" in lower_formula:
        return "mV*s"
    return "dimensionless"


def _definition(
    index: int,
    family: str,
    name: str,
    description: str,
    formula: str,
    focus: str,
    unit: Optional[str] = None,
) -> BiomarkerDefinition:
    notes = focus
    if "/" in name or "<seg>" in name or "*" in name:
        notes = (
            f"{focus} Composite or template-style PDF entry; concrete emitted "
            "feature names should use the aliases/placeholders consistently."
        )
    return BiomarkerDefinition(
        name=name,
        family=family,
        description=description,
        formula=formula,
        unit=_unit_for_name(name, formula, unit),
        category=_category_for(family),
        source="biomarker.pdf",
        source_index=index,
        pathological_note=DATA_DRIVEN_CUTOFF_NOTE,
        implementation_status="candidate",
        aliases=_aliases_for(name),
        notes=notes,
    )


def _manual_definitions() -> list[BiomarkerDefinition]:
    rows = [
        (1, "Achsenamplituden", "vcg_x_max/min", "Maximum and minimum X-axis VCG amplitude.", "max(x), min(x) over the signal or selected segment.", "Axis-specific electrical deflection.", "mV"),
        (2, "Achsenamplituden", "vcg_x_peak_to_peak", "Peak-to-peak X-axis VCG amplitude.", "max(x) - min(x).", "Axis dominance and low/high-voltage context.", "mV"),
        (3, "Achsenamplituden", "vcg_y_max/min", "Maximum and minimum Y-axis VCG amplitude.", "max(y), min(y) over the signal or selected segment.", "Axis-specific electrical deflection.", "mV"),
        (4, "Achsenamplituden", "vcg_y_peak_to_peak", "Peak-to-peak Y-axis VCG amplitude.", "max(y) - min(y).", "Axis dominance and low/high-voltage context.", "mV"),
        (5, "Achsenamplituden", "vcg_z_max/min", "Maximum and minimum Z-axis VCG amplitude.", "max(z), min(z) over the signal or selected segment.", "Axis-specific electrical deflection.", "mV"),
        (6, "Achsenamplituden", "vcg_z_peak_to_peak", "Peak-to-peak Z-axis VCG amplitude.", "max(z) - min(z).", "Axis dominance and low/high-voltage context.", "mV"),
        (7, "Achsenflaechen", "vcg_x_negative_area", "Negative signed area of the X-axis signal.", "Integral min(x(t), 0) dt using trapezoidal integration.", "Direction-specific counter-component.", "mV*s"),
        (8, "Achsenflaechen", "vcg_x_positive_area", "Positive signed area of the X-axis signal.", "Integral max(x(t), 0) dt using trapezoidal integration.", "Direction-specific activity load.", "mV*s"),
        (9, "Achsenflaechen", "vcg_y_negative_area", "Negative signed area of the Y-axis signal.", "Integral min(y(t), 0) dt using trapezoidal integration.", "Direction-specific counter-component.", "mV*s"),
        (10, "Achsenflaechen", "vcg_y_positive_area", "Positive signed area of the Y-axis signal.", "Integral max(y(t), 0) dt using trapezoidal integration.", "Direction-specific activity load.", "mV*s"),
        (11, "Achsenflaechen", "vcg_z_negative_area", "Negative signed area of the Z-axis signal.", "Integral min(z(t), 0) dt using trapezoidal integration.", "Direction-specific counter-component.", "mV*s"),
        (12, "Achsenflaechen", "vcg_z_positive_area", "Positive signed area of the Z-axis signal.", "Integral max(z(t), 0) dt using trapezoidal integration.", "Direction-specific activity load.", "mV*s"),
        (13, "Achsenverhaeltnisse", "vcg_x_y_ptp_ratio", "Ratio of X to Y peak-to-peak amplitude.", "ptp(x) / (ptp(y) + eps).", "Spatial axis dominance or compression.", "dimensionless"),
        (14, "Achsenverhaeltnisse", "vcg_x_z_ptp_ratio", "Ratio of X to Z peak-to-peak amplitude.", "ptp(x) / (ptp(z) + eps).", "Spatial axis dominance or compression.", "dimensionless"),
        (15, "Achsenverhaeltnisse", "vcg_y_z_ptp_ratio", "Ratio of Y to Z peak-to-peak amplitude.", "ptp(y) / (ptp(z) + eps).", "Spatial axis dominance or compression.", "dimensionless"),
        (16, "Anisotropie/Kompression", "axis_anisotropy_std_max_min_ratio", "Ratio of largest to smallest axis standard deviation.", "max(std(x,y,z)) / (min(std(x,y,z)) + eps).", "Volume-conductor, electrode and transform context.", "dimensionless"),
        (17, "Anisotropie/Kompression", "axis_compression_index_std_min_max", "Axis compression index.", "min(std(x,y,z)) / (max(std(x,y,z)) + eps).", "Volume-conductor, electrode and transform context.", "dimensionless"),
        (18, "Anisotropie/Kompression", "frontal_plane_dominance_xy", "Variance dominance of the frontal X-Y plane.", "(var(x) + var(y)) / (var(x) + var(y) + var(z) + eps).", "Plane dominance and transform context.", "dimensionless"),
        (19, "Anisotropie/Kompression", "horizontal_plane_dominance_xz", "Variance dominance of the horizontal X-Z plane.", "(var(x) + var(z)) / (var(x) + var(y) + var(z) + eps).", "Plane dominance and transform context.", "dimensionless"),
        (20, "Anisotropie/Kompression", "sagittal_plane_dominance_yz", "Variance dominance of the sagittal Y-Z plane.", "(var(y) + var(z)) / (var(x) + var(y) + var(z) + eps).", "Plane dominance and transform context.", "dimensionless"),
        (21, "Anisotropie/Kompression", "x/y/z_dominance_variance_fraction", "Per-axis variance fraction for X, Y and Z.", "var(axis) / (var(x) + var(y) + var(z) + eps).", "Per-axis dominance and transform context.", "mixed"),
        (22, "Beat-Variabilitaet", "<seg>_axis_beat_to_beat_variability_deg", "Beat-to-beat variability of segmental vector axis.", "Angle between consecutive beat peak vectors.", "Axis instability.", "degrees"),
        (23, "Beat-Variabilitaet", "<seg>_beat_to_beat_*_distance", "Beat-to-beat template distance metrics.", "Procrustes, DTW, Hausdorff or Euclidean distance to the median beat.", "Extrasystoles, artifacts and morphology shifts.", "mixed"),
        (24, "Beat-Variabilitaet", "<seg>_loop_area_variability", "Beat-to-beat variability of loop area.", "Standard deviation of segmental loop areas across beats.", "Beat-to-beat morphology variability.", "mV^2"),
        (25, "Beat-Variabilitaet", "<seg>_vector_variability", "Beat-to-beat variability of vector magnitude.", "Standard deviation of peak vector magnitudes across beats.", "Morphology variability.", "mV"),
        (26, "Beat-Variabilitaet", "atrial_fibrillation_rr_surrogate", "RR/P-consistency surrogate for atrial fibrillation context.", "Weighted RR coefficient-of-variation, successive irregularity and evaluable P-wave coverage.", "Technical rhythm context only; not an AF diagnosis. Low-quality or short records are not evaluable.", "dimensionless"),
        (27, "Beat-Variabilitaet", "qrs_alternans_vector", "Odd-even difference of peak QRS vector magnitudes.", "abs(mean(even-beat peak norm(QRS)) - mean(odd-beat peak norm(QRS))); time-normalized segments.", "Exploratory amplitude alternation; blind to pure directional alternation, not validated clinical TWA.", "mV"),
        (28, "Beat-Variabilitaet", "qrs_t_angle_variability", "Beat-to-beat variability of the spatial QRS-T angle.", "Standard deviation of beat-level spatial QRS-T angle.", "Repolarization lability.", "degrees"),
        (29, "Beat-Variabilitaet", "rr_entropy/irregularity_index", "RR entropy and irregularity index.", "Histogram entropy of RR intervals and sd(RR) / mean(RR).", "AF or ectopy surrogate.", "mixed"),
        (30, "Beat-Variabilitaet", "rr_mean_ms/sd_ms/rmssd_ms/pnn50", "Basic RR and short-record HRV features.", "RR mean, standard deviation, RMSSD and pNN50.", "Rhythm irregularity; limited in 10-second ECGs.", "mixed"),
        (31, "Beat-Variabilitaet", "t_wave_alternans_vector", "Odd-even difference of peak T vector magnitudes.", "abs(mean(even-beat peak norm(T)) - mean(odd-beat peak norm(T))); time-normalized segments.", "Exploratory amplitude alternation; blind to pure directional alternation, not validated clinical TWA.", "mV"),
        (32, "Experimentell/Topologisch", "<seg>_cov_*", "Covariance and information-geometry descriptors.", "Trace, determinant, condition number, entropy and log-eigenvalue norms of the segmental point cloud.", "Point-cloud morphology.", "mixed"),
        (33, "Experimentell/Topologisch", "<seg>_directional_hysteresis_angle_deg", "Directional hysteresis angle.", "Angle between vectors from the first and second segment halves.", "Loop hysteresis.", "degrees"),
        (34, "Experimentell/Topologisch", "<seg>_electrical_work", "Electrical work surrogate.", "Integral ||r(t)|| * ||dr/dt|| dt.", "Mechanics-inspired trajectory measure.", "mV^2"),
        (35, "Experimentell/Topologisch", "<seg>_loop_angular_momentum", "Loop angular momentum surrogate.", "Integral ||r(t) x dr/dt|| dt.", "Rotation and loop dynamics.", "mV^2"),
        (36, "Experimentell/Topologisch", "<seg>_radial_peak_count/persistence_proxy", "Radial topology proxies.", "Peak count and prominence of ||r(t) - centroid||.", "Loop folding or bulging.", "mixed"),
        (37, "Experimentell/Topologisch", "<seg>_self_crossings_*", "Projected loop self-crossings.", "Count 2D segment intersections in frontal, horizontal and sagittal projections.", "Topological loop complexity.", "count"),
        (38, "Experimentell/Topologisch", "<seg>_spherical_direction_entropy", "Spherical direction entropy.", "Entropy of sign or direction classes of normalized vectors.", "Directional diversity.", "dimensionless"),
        (39, "Experimentell/Topologisch", "<seg>_symbolic_*", "Symbolic VCG trajectory descriptors.", "Dominant direction letters plus n-gram and transition entropy.", "Spatial movement sequence.", "mixed"),
        (40, "Experimentell/Topologisch", "qrs_t_energy_transfer_proxy", "QRS-to-T energy transfer proxy.", "Integral T magnitude / (Integral QRS magnitude + eps).", "Depolarization/repolarization balance.", "dimensionless"),
        (41, "Experimentell/Topologisch", "qrs_t_wasserstein_radial_proxy", "Radial Wasserstein distance between QRS and T.", "Wasserstein distance between radial QRS and T distributions.", "Spatial QRS-T distribution difference.", "mV"),
        (42, "Hauptvektoren", "qrs_initial_20ms_vector", "Initial QRS vector during the first 20 ms.", "Mean r(t) in the first 20 ms after QRS onset.", "Septal or initial activation.", "mV"),
        (43, "Hauptvektoren", "qrs_terminal_20ms_vector", "Terminal QRS vector during the final 20 ms.", "Mean r(t) in the final 20 ms before QRS offset.", "Bundle-branch-block or late activation context.", "mV"),
        (44, "Hauptvektoren", "tpeak_vector / tend_vector / j_point_vector", "Vectors at T peak, T end and J point.", "r(t) at fiducial points, optionally median or mean across beats.", "Repolarization and ST context.", "mV"),
    ]
    return [_definition(*row) for row in rows]


def _vector_integral_definitions() -> list[BiomarkerDefinition]:
    definitions = []
    row = 45
    segments = [
        ("p", "P"),
        ("qrs", "QRS"),
        ("qrst", "QRST"),
        ("st", "ST"),
        ("st_t", "ST-T"),
        ("t", "T"),
    ]
    kinds = [
        ("area", "Area/integral vector", "Integral r(t) dt over the segment.", "mV*s"),
        ("max", "Maximum vector", "Vector at maximum segmental spatial magnitude.", "mixed"),
        ("mean", "Mean vector", "Mean r(t) over the segment.", "mV"),
    ]
    for segment, label in segments:
        for kind, description_prefix, formula, unit in kinds:
            name = f"{segment}_{kind}_vector_x/y/z/magnitude/axis"
            definitions.append(_definition(
                row,
                "Hauptvektoren/Integrale",
                name,
                f"{description_prefix} in the {label} segment, reported as components, magnitude and orientation.",
                formula + " Orientation includes frontal axis and can be extended to azimuth/elevation.",
                "Direction and strength of segmental electrical activity.",
                "mixed" if unit == "mixed" or "axis" in name else unit,
            ))
            row += 1
    return definitions


KINEMATIC_FEATURES = [
    ("acceleration_energy", "Acceleration energy", "Integral ||a(t)|| dt, with a=d2r/dt2.", "mV/s"),
    ("acceleration_max", "Maximum acceleration", "max(||a(t)||), with a=d2r/dt2.", "mV/s^2"),
    ("acceleration_mean", "Mean acceleration", "mean(||a(t)||), with a=d2r/dt2.", "mV/s^2"),
    ("angular_velocity_max", "Maximum angular velocity", "max(||r(t) x v(t)|| / (||r(t)||^2 + eps)).", "rad/s"),
    ("angular_velocity_mean", "Mean angular velocity", "mean(||r(t) x v(t)|| / (||r(t)||^2 + eps)).", "rad/s"),
    ("curvature_integral", "Integrated curvature", "Integral kappa(t) dt, kappa=||v x a||/(||v||^3 + eps).", "s/mV"),
    ("curvature_max", "Maximum curvature", "max(kappa(t)), kappa=||v x a||/(||v||^3 + eps).", "1/mV"),
    ("curvature_mean", "Mean curvature", "mean(kappa(t)), kappa=||v x a||/(||v||^3 + eps).", "1/mV"),
    ("jerk_energy", "Jerk energy", "Integral ||j(t)|| dt, with j=d3r/dt3.", "mV/s^2"),
    ("jerk_max", "Maximum jerk", "max(||j(t)||), with j=d3r/dt3.", "mV/s^3"),
    ("jerk_mean", "Mean jerk", "mean(||j(t)||), with j=d3r/dt3.", "mV/s^3"),
    ("path_length_per_time", "Path length per time", "Integral ||v(t)|| dt divided by segment duration.", "mV/s"),
    ("rotation_entropy", "Rotation entropy", "Histogram entropy of angular velocity.", "dimensionless"),
    ("sharpness_index", "Sharpness index", "max(||a(t)||) / (mean(||v(t)||) + eps).", "1/s"),
    ("smoothness_index", "Smoothness index", "1 / (1 + normalized jerk energy).", "dimensionless"),
    ("torsion_integral_abs", "Integrated absolute torsion", "Integral |tau(t)| dt, tau=((v x a) dot j)/(||v x a||^2 + eps).", "s/mV"),
    ("torsion_max_abs", "Maximum absolute torsion", "max(|tau(t)|), tau=((v x a) dot j)/(||v x a||^2 + eps).", "1/mV"),
    ("velocity_energy", "Velocity energy", "Integral ||v(t)||^2 dt, with v=dr/dt.", "mV^2/s"),
    ("velocity_entropy", "Velocity entropy", "Histogram entropy of ||v(t)||.", "dimensionless"),
    ("velocity_integral", "Velocity integral", "Integral ||v(t)|| dt.", "mV"),
    ("velocity_max", "Maximum vector velocity", "max(||v(t)||), with v=dr/dt.", "mV/s"),
    ("velocity_mean", "Mean vector velocity", "mean(||v(t)||), with v=dr/dt.", "mV/s"),
    ("velocity_median", "Median vector velocity", "median(||v(t)||), with v=dr/dt.", "mV/s"),
    ("velocity_peak_count", "Number of velocity peaks", "Peak count in ||v(t)|| after pre-specified smoothing.", "count"),
    ("velocity_rms", "RMS vector velocity", "sqrt(mean(||v(t)||^2)), with v=dr/dt.", "mV/s"),
]


def _kinematic_definitions() -> list[BiomarkerDefinition]:
    definitions = []
    row = 63
    plan = [
        ("beat", 0, 25),
        ("global_vcg", 0, 25),
        ("p", 0, 25),
        ("pr", 0, 25),
        ("qrs", 0, 8),
        ("qrs_early", 0, 25),
        ("qrs", 8, 11),
        ("qrs_late", 0, 2),
        ("qrs_late", 2, 25),
        ("qrs_mid", 0, 25),
        ("qrs", 11, 25),
        ("qrst", 0, 25),
        ("st", 0, 15),
        ("st_t", 0, 25),
        ("st", 15, 25),
        ("t", 0, 8),
        ("t_downstroke", 0, 25),
        ("t_early", 0, 25),
        ("t", 8, 11),
        ("t_late", 0, 2),
        ("t_late", 2, 25),
        ("t_mid", 0, 25),
        ("t", 11, 17),
        ("t_upstroke", 0, 25),
        ("t", 17, 25),
    ]
    for segment, start, stop in plan:
        for suffix, label, formula, unit in KINEMATIC_FEATURES[start:stop]:
            definitions.append(_definition(
                row,
                "Kinematik",
                f"{segment}_{suffix}",
                f"{label} in the {segment} segment.",
                formula,
                "Trajectory dynamics, conduction delay, fragmentation and noise sensitivity.",
                unit,
            ))
            row += 1
    if row != 488:
        raise RuntimeError(f"Internal catalog error: expected first 487 rows, got {row - 1}")
    return definitions


def _kinematic_landmark_definitions() -> list[BiomarkerDefinition]:
    rows = [
        (488, "late_qrs_slowing_index", "Late QRS slowing index.", "qrs_late_velocity_mean / (qrs_early_velocity_mean + eps).", "dimensionless"),
        (489, "qrs_velocity_t_velocity_ratio", "Ratio of mean QRS to mean T vector velocity.", "qrs_velocity_mean / (t_velocity_mean + eps).", "dimensionless"),
        (490, "t_wave_deceleration_index", "T-wave deceleration index.", "t_downstroke_velocity_mean / (t_upstroke_velocity_mean + eps).", "dimensionless"),
        (491, "velocity_at_j_point", "Spatial velocity at the J point.", "||v(t)|| at QRS offset/J point.", "mV/s"),
        (492, "velocity_at_p_onset", "Spatial velocity at P-wave onset.", "||v(t)|| at P onset.", "mV/s"),
        (493, "velocity_at_qrs_onset", "Spatial velocity at QRS onset.", "||v(t)|| at QRS onset.", "mV/s"),
        (494, "velocity_at_r_peak", "Spatial velocity at the R peak.", "||v(t)|| at R peak.", "mV/s"),
        (495, "velocity_at_t_end", "Spatial velocity at T-wave end.", "||v(t)|| at T end.", "mV/s"),
        (496, "velocity_at_t_peak", "Spatial velocity at T-wave peak.", "||v(t)|| at T peak.", "mV/s"),
    ]
    return [
        _definition(
            index,
            "Kinematische Landmarken",
            name,
            description,
            formula,
            "Dynamic fiducial-point and segment consistency check.",
            unit,
        )
        for index, name, description, formula, unit in rows
    ]


def _classic_ecg_definitions() -> list[BiomarkerDefinition]:
    definitions = []
    row = 497
    leads = ["II", "I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"]
    for lead in leads:
        definitions.append(_definition(
            row,
            "Klassische ECG-Projektionen",
            f"lead_{lead}_qrs_peak_positive/negative/abs_peak",
            f"Positive, negative and absolute QRS peak amplitude in lead {lead}.",
            "max(lead signal), min(lead signal) and max(abs(lead signal)) inside the QRS window; report in uV when ECG input is in mV.",
            "Classic lead context and virtual ECG criteria.",
            "uV",
        ))
        row += 1

    definitions.extend([
        _definition(
            506,
            "Klassische ECG-Surrogate",
            "cornell_voltage_surrogate",
            "Cornell-like voltage marker.",
            "R_aVL + abs(S_V3), reported in uV when ECG input is in mV.",
            "LVH surrogate; sex-specific and population-specific rule thresholds require separate validation.",
            "uV",
        ),
        _definition(
            507,
            "Klassische ECG-Surrogate",
            "sokolow_lyon_voltage_surrogate",
            "Sokolow-Lyon-like voltage marker.",
            "abs(S_V1) + max(R_V5, R_V6), reported in uV when ECG input is in mV.",
            "LVH surrogate; not a complete clinical decision rule by itself.",
            "uV",
        ),
        BiomarkerDefinition(
            name="peguero_lo_presti_voltage_surrogate",
            family="Klassische ECG-Surrogate",
            description="Peguero-Lo Presti-like LVH voltage marker.",
            formula="deepest S wave in any 12-lead ECG lead + abs(S_V4), reported in uV when ECG input is in mV.",
            unit="uV",
            category="ecg",
            source="clinical_reference",
            source_index=525,
            pathological_note=(
                "Clinical LVH voltage surrogate; fixed thresholds are evaluated in "
                "the performance dashboard as contextual benchmarks."
            ),
            implementation_status="candidate",
            notes="LVH surrogate; sex-specific and population-specific thresholds require separate validation.",
        ),
    ])
    return definitions


def _literature_extension_definitions() -> list[BiomarkerDefinition]:
    rows = [
        (508, "hf_qrs_vector_notching_score", "VCG notching score of QRS velocity.", "Peak count and prominence in a high-frequency M(t) or ||v|| component.", "Non-homogeneous depolarization.", "dimensionless"),
        (509, "microvolt_t_wave_alternans_vector", "3D T-wave alternans in vector-space signal.", "Even-odd difference of resampled T loops, reported as RMS or maximum difference.", "Plausible malignant-arrhythmia and repolarization-lability marker.", "uV"),
        (510, "p_wave_mvp_vector_score", "Morphology-voltage-P-duration score in VCG context.", "Composite of P duration, P voltage or magnitude and P morphology complexity.", "AF or atrial-remodeling hypothesis.", "dimensionless"),
        (511, "p_wave_terminal_force_vector_vcg", "VCG P-wave terminal force.", "Terminal P-vector area or magnitude in the final half of the P wave; optional V1 projection.", "Atrial cardiomyopathy or AF-risk hypothesis.", "uV*ms"),
        (512, "qrs_area_t_area_crt_response_surrogate", "QRS-area/T-area combination marker.", "Combined threshold or score model from QRS area and T area.", "CRT, conduction and repolarization hypothesis; exploratory.", "dimensionless"),
        (513, "qrs_fragmentation_high_frequency_energy", "High-frequency QRS fragmentation energy.", "Axiswise 80-150 Hz bandpass followed by spatial energy inside QRS; requires documented analogue bandwidth through 150 Hz.", "Fragmentation, scar and conduction inhomogeneity.", "uV^2"),
        (514, "qrs_loop_dihedral_angle_deg", "Dihedral angle of the QRS loop.", "Angle between best-fit planes of early and late QRS subloops.", "Loop folding and segment-plane change.", "degrees"),
        (515, "qrs_loop_folding_index", "Figure-of-eight or folding index of the QRS loop.", "Self-crossings plus radial peak persistence and rotation changes.", "Local conduction delay or fiber-orientation hypothesis.", "dimensionless"),
        (516, "qrs_loop_roundness_planarity_composite", "Composite QRS-loop morphology score.", "Robust z-scaling of roundness, planarity, thickness and rotational-angle features.", "Quantitative VCG-loop morphology.", "dimensionless"),
        (517, "qrs_t_angle_lability_index", "Beat-to-beat lability of the spatial QRS-T angle.", "Standard deviation or median absolute deviation of beat-level QRS-T angles.", "Repolarization instability.", "degrees"),
        (518, "qt_rr_hysteresis_surrogate", "QT/RR hysteresis surrogate.", "Cross-lag or correlation of QT_i against RR_i and RR_{i-1}.", "Autonomic and restitution coupling.", "dimensionless"),
        (519, "qt_variability_index_vector", "QT variability index in vector space.", "log10((var(QT)/mean(QT)^2)/(var(RR)/mean(RR)^2)).", "Repolarization instability; document 10-second ECG limitation.", "dimensionless"),
        (520, "spatial_mean_qrs_t_angle_deg", "Mean-vector based spatial QRS-T angle.", "Angle between time-mean QRS and T vectors.", "Complement to peak and area spatial QRS-T angle.", "degrees"),
        (521, "spatial_peak_qrs_t_angle_deg", "Peak-vector based spatial QRS-T angle.", "Angle between maximum QRS vector and maximum T vector, beat-wise and median.", "Complements mean and area QRS-T angle; prognostic plausibility.", "degrees"),
        (522, "spatial_ventricular_gradient_dispersion", "Beat-to-beat spatial ventricular gradient dispersion.", "Standard deviation or MAD of beat-level SVG vectors.", "Repolarization and gradient lability.", "uV*ms"),
        (523, "t_wave_area_dispersion_12lead", "T-wave area dispersion across 12 leads.", "Standard deviation or range of T-wave areas across all ECG leads.", "Repolarization heterogeneity; plausible SCD-risk marker.", "uV*ms"),
        (524, "tpeak_tend_vector_dispersion", "Tpeak-Tend vector dispersion.", "Beat-to-beat variation of the vector from T peak to T end.", "Terminal repolarization heterogeneity.", "uV"),
        (525, "vcg_t_area_dispersion_xyz", "T-area dispersion in X/Y/Z.", "Standard deviation of T-area components and magnitude fractions.", "VCG analogue of T-area dispersion.", "uV*ms"),
    ]
    return [
        _definition(index, "Literaturbasierte Ergaenzung", name, description, formula, focus, unit)
        for index, name, description, formula, focus, unit in rows
    ]


LOOP_GEOMETRY_FEATURES = [
    ("actual_area_to_convex_hull_ratio", "compactness", "actual_area / (convex_hull_area + eps).", "dimensionless"),
    ("area_best_fit_plane", "area in the best-fit plane", "Shoelace area after projection onto PC1-PC2.", "mV^2"),
    ("bounding_box_volume", "axis-aligned bounding-box volume", "prod(maxXYZ - minXYZ).", "mV^3"),
    ("bounding_ellipsoid_volume_approx", "bounding ellipsoid volume approximation", "4/3*pi*prod(2*sqrt(lambda_i)).", "mV^3"),
    ("centroid_shift_magnitude", "centroid shift magnitude", "||centroid||.", "mV"),
    ("convex_hull_area_best_fit_plane", "2D convex-hull area in the best-fit plane", "ConvexHull area in PC1-PC2.", "mV^2"),
    ("convex_hull_volume_3d", "3D convex-hull volume", "scipy ConvexHull volume.", "mV^3"),
    ("diameter", "loop diameter", "Maximum pairwise point distance.", "mV"),
    ("eccentricity", "loop eccentricity", "sqrt(1 - lambda2/(lambda1 + eps)).", "dimensionless"),
    ("elongation", "loop elongation", "major_width / (minor_width + eps).", "dimensionless"),
    ("flatness", "loop flatness", "lambda3 / (lambda2 + eps).", "dimensionless"),
    ("lambda1/lambda2/lambda3", "point-cloud covariance eigenvalues", "Eigenvalues of the covariance matrix.", "mV^2"),
    ("linearity", "loop linearity", "lambda1 / (sum(lambda) + eps).", "dimensionless"),
    ("max_point_distance", "maximum point distance", "Maximum pairwise point distance.", "mV"),
    ("mean_distance_centroid", "mean centroid distance", "mean(||r - centroid||).", "mV"),
    ("nonplanarity", "loop nonplanarity", "lambda3 / (sum(lambda) + eps).", "dimensionless"),
    ("path_length", "loop path length", "sum(||diff(r)||).", "mV"),
    ("path_length_per_ms", "loop path length per millisecond", "path_length / segment_duration_ms.", "mV/ms"),
    ("perimeter_best_fit_plane", "perimeter in the best-fit plane", "2D projected path perimeter.", "mV"),
    ("planarity", "loop planarity", "(lambda1 + lambda2) / (sum(lambda) + eps).", "dimensionless"),
    ("plane_normal_x/y/z/azimuth/elevation", "loop-plane normal vector and orientation", "PC3 normal vector with azimuth and elevation.", "mixed"),
    ("radial_skewness/kurtosis", "radial skewness and kurtosis", "Moments of ||r - centroid||.", "dimensionless"),
    ("return_to_baseline_distance", "return-to-baseline distance", "||r_end - r_start|| as return proxy.", "mV"),
    ("rms_distance_centroid", "RMS centroid distance", "sqrt(mean(||r - centroid||^2)).", "mV"),
    ("roundness_best_fit_plane", "roundness in the best-fit plane", "4*pi*area / closed_projected_perimeter^2; perimeter includes the straight endpoint-to-start closure; abstain for zero perimeter.", "dimensionless"),
    ("sphericity", "loop sphericity", "lambda3 / (lambda1 + eps).", "dimensionless"),
    ("start_end_distance", "start-end distance", "||r_end - r_start||.", "mV"),
    ("thickness", "loop thickness", "Range of PCA score on PC3.", "mV"),
    ("tortuosity", "closed-loop tortuosity", "(open path + closure distance) / convex-hull perimeter in the best-fit plane.", "dimensionless"),
    ("width_major", "major width", "Range of PCA score on PC1.", "mV"),
    ("width_minor", "minor width", "Range of PCA score on PC2.", "mV"),
]


def _loop_geometry_definitions() -> list[BiomarkerDefinition]:
    definitions = []
    row = 526
    for segment, label in [("p", "P"), ("qrs", "QRS"), ("qrst", "QRST"), ("st", "ST"), ("t", "T")]:
        for suffix, label_suffix, formula, unit in LOOP_GEOMETRY_FEATURES:
            definitions.append(_definition(
                row,
                "Loop-Geometrie",
                f"{segment}_loop_{suffix}",
                f"{label_suffix.capitalize()} of the {label} loop.",
                formula,
                "Visually inspectable VCG morphology.",
                unit,
            ))
            row += 1
    return definitions


LOOP_PROJECTION_FEATURES = [
    ("area", "projected area", "Shoelace area in the selected 2D projection.", "mV^2"),
    ("convex_hull_area", "projected convex-hull area", "ConvexHull area in the selected 2D projection.", "mV^2"),
    ("eccentricity", "projected eccentricity", "PCA ellipse eccentricity in the selected 2D projection.", "dimensionless"),
    ("max_width", "maximum projected width", "PCA score range on projected PC1.", "mV"),
    ("min_width", "minimum projected width", "PCA score range on projected PC2.", "mV"),
    ("perimeter", "projected perimeter", "Sum of 2D segment lengths.", "mV"),
    ("rotation_change_count", "rotation-change count", "Number of sign changes in local projected curvature.", "count"),
    ("rotation_direction", "rotation direction", "sign(signed_area).", "dimensionless"),
    ("roundness", "projected roundness", "4*pi*area / closed_projected_perimeter^2; perimeter includes the straight endpoint-to-start closure; abstain for zero perimeter.", "dimensionless"),
    ("self_intersection_count", "projected self-intersection count", "2D segment intersection count.", "count"),
    ("signed_area", "signed projected area", "Signed Shoelace area.", "mV^2"),
    ("winding_number", "projected winding number", "unwrap(angle around centroid) / (2*pi).", "dimensionless"),
]


def _loop_projection_definitions() -> list[BiomarkerDefinition]:
    definitions = []
    row = 681
    planes = [
        ("frontal_xy", "frontal X-Y"),
        ("horizontal_xz", "horizontal X-Z"),
        ("sagittal_yz", "sagittal Y-Z"),
    ]
    segments = [
        ("p", "P"),
        ("qrs", "QRS"),
        ("qrst", "QRST"),
        ("st", "ST"),
        ("t", "T"),
    ]
    for segment, segment_label in segments:
        for plane, plane_label in planes:
            for suffix, label_suffix, formula, unit in LOOP_PROJECTION_FEATURES:
                definitions.append(_definition(
                    row,
                    "Loop-Projektionen",
                    f"{segment}_loop_{plane}_{suffix}",
                    f"{label_suffix.capitalize()} of the {segment_label} loop in the {plane_label} projection.",
                    formula,
                    "Rotation, orientation and topological complexity in 2D loop projections.",
                    unit,
                ))
                row += 1
    return definitions


def _extra_tail_definitions() -> list[BiomarkerDefinition]:
    rows = [
        (861, "Loop-Robustheit und Reliabilitaet", "loop_confidence_weighted_value", "Confidence-weighted morphology value.", "feature_value * loop_confidence.", "Robust variant of fiducial-dependent markers.", "mixed"),
        (862, "Loop-Robustheit und Reliabilitaet", "loop_fiducial_margin", "Fiducial margin.", "Distance from segment borders to peak or landmarks, normalized by segment duration.", "Checks whether the segment window is plausibly placed.", "dimensionless"),
        (863, "Loop-Robustheit und Reliabilitaet", "loop_interbeat_reproducibility", "Interbeat reproducibility.", "Median distance of single-beat loops to the median loop.", "Stability of loop markers.", "mV"),
        (864, "Loop-Robustheit und Reliabilitaet", "loop_missingness_risk", "Loop missingness risk.", "Fraction of missing or invalid P/T/QRS fiducials.", "Analysis weighting and warning signal.", "fraction"),
        (865, "Loop-Robustheit und Reliabilitaet", "loop_outlier_beat_fraction", "Fraction of outlier beat loops.", "Fraction with distance to median loop greater than 2*IQR.", "Extrasystoles and artifacts.", "fraction"),
        (866, "Loop-Robustheit und Reliabilitaet", "loop_quality_stratified_delta", "Quality-stratified delta value.", "Feature at high signal quality minus feature at low signal quality.", "Artifact dependence of a marker.", "mixed"),
        (867, "Loop-Robustheit und Reliabilitaet", "loop_snr_proxy", "Loop signal-to-noise proxy.", "Loop amplitude divided by noise estimate before or outside the segment.", "Trustworthiness of small P and T loops.", "dimensionless"),
        (868, "Magnitude-Verhaeltnis", "qrs_t_peak_magnitude_ratio", "Ratio of QRS to T peak vector magnitude.", "QRS_peak / (T_peak + eps).", "Depolarization/repolarization amplitude comparison.", "dimensionless"),
        (869, "Manuelle Review", "landmark_*_index_median", "Review landmark median indices.", "Median sample indices of QRS/T/fiducial landmarks.", "GUI and review prioritization.", "sample"),
        (870, "Manuelle Review", "manual_review_required", "Manual-review flag.", "Threshold over uncertainty and loop scores.", "Prioritizes cases for review.", "binary"),
        (871, "Manuelle Review", "qrs_loop_abnormal_score", "QRS-loop abnormality score.", "Aggregation of nonplanarity and variability.", "Visual QRS-loop review.", "dimensionless"),
        (872, "Manuelle Review", "t_loop_abnormal_score", "T-loop abnormality score.", "Aggregation of nonplanarity and variability.", "Repolarization review.", "dimensionless"),
        (873, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_bending_energy", "Loop bending energy.", "Integral kappa(t)^2 ds.", "Captures strong local direction changes without ML.", "1/mV"),
        (874, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_curvature_entropy", "Entropy of the curvature distribution.", "Shannon entropy of a histogram of kappa(t).", "Complex versus simple loop shape.", "dimensionless"),
        (875, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_curvature_peak_count", "Number of local curvature maxima.", "Peak count of kappa(t) after minimum prominence filtering.", "Fragmentation, notching and multi-peak morphology.", "count"),
        (876, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_curvature_torsion_ratio", "Ratio of curvature to torsion.", "Integral |kappa| / (Integral |tau| + eps).", "Planar bending versus spatial twisting.", "dimensionless"),
        (877, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_mean_curvature", "Mean loop curvature.", "mean(kappa(t)) over the resampled loop.", "Simple versus strongly curved loop shape.", "1/mV"),
        (878, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_torsion_sign_changes", "Sign changes of torsion.", "Count sign(tau(t)) changes after smoothing.", "Changing excursion out of the main plane.", "count"),
        (879, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_total_curvature", "Total 3D loop curvature.", "Integral |kappa(t)| ds with kappa=||r' x r''||/||r'||^3.", "QRS/P/T folding, local conduction delay and repolarization complexity; noise sensitive.", "dimensionless"),
        (880, "Mathematische Loop-Morphologie - Differentialgeometrie", "loop_total_torsion", "Total spatial torsion.", "Integral |tau(t)| ds with tau=((r' x r'') dot r''')/||r' x r''||^2.", "Nonplanar QRS/T/P-loop complexity.", "dimensionless"),
        (881, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_actual_to_ellipse_area_ratio", "Actual-to-ellipse area ratio.", "projected_loop_area / (ellipse_area + eps).", "Concavity and deviation from an ellipse.", "dimensionless"),
        (882, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_best_fit_ellipse_error", "Distance to the best-fit ellipse.", "Projection into the main plane; RMS distance to the fitted ellipse.", "Irregularity relative to simple elliptical loop form.", "mV"),
        (883, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_ellipse_area", "Area of the best-fit ellipse.", "pi*a*b.", "Idealized loop area.", "mV^2"),
        (884, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_ellipse_axis_ratio", "Ellipse axis ratio.", "major_axis / (minor_axis + eps).", "Elongated versus round.", "dimensionless"),
        (885, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_ellipse_major_axis", "Major axis of the best-fit ellipse.", "Ellipse/PCA main-plane major axis length.", "Loop extent.", "mV"),
        (886, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_ellipse_minor_axis", "Minor axis of the best-fit ellipse.", "Minor axis length of the ellipse.", "Loop width.", "mV"),
        (887, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_ellipse_orientation_deg", "Orientation of the ellipse.", "Angle of the major axis in the loop main plane or projection.", "Direction of the basic loop form.", "degrees"),
        (888, "Mathematische Loop-Morphologie - Ellipse und Superellipse", "loop_superellipse_exponent", "Superellipse exponent.", "Fit |x/a|^n + |y/b|^n = 1.", "Rectangular versus oval loop shape.", "dimensionless"),
        (889, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_backtracking_fraction", "Fraction of backward trajectory motion.", "Fraction with dot(tangent_i, mean_direction) < 0.", "Retrograde movement or reversal components of the loop.", "fraction"),
        (890, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_fold_density", "Normalized folding density.", "total_self_intersections / (path_length + eps).", "Scale-independent folding marker.", "1/mV"),
        (891, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_min_nonlocal_distance", "Minimum distance of non-adjacent loop segments.", "Minimum distance(segment_i, segment_j) with temporal exclusion window.", "How closely loop parts approach each other.", "mV"),
        (892, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_near_self_approach_count", "Near self-approaches of the loop.", "Count non-adjacent point or segment pairs with distance < epsilon.", "Hidden folding without true projected crossing.", "count"),
        (893, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_self_intersection_xy", "Self-intersections in XY projection.", "Non-adjacent line segment intersection count.", "Folding in frontal plane.", "count"),
        (894, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_self_intersection_xz", "Self-intersections in XZ projection.", "Segment intersection count in horizontal projection.", "Anterior/posterior folding.", "count"),
        (895, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_self_intersection_yz", "Self-intersections in YZ projection.", "Segment intersection count in sagittal projection.", "Vertical folding.", "count"),
        (896, "Mathematische Loop-Morphologie - Folding und Selbstannaeherung", "loop_total_self_intersections", "Total crossing index.", "Sum of self-intersections from XY, XZ and YZ.", "Overall projected folding.", "count"),
        (897, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_complexity_ratio", "Harmonic complexity ratio.", "high_order_energy / (total_fourier_energy + eps).", "Deviation from simple ellipse.", "dimensionless"),
        (898, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_ellipticity", "Ellipticity of the fundamental harmonic.", "Axis ratio of the k=1 component in the main plane.", "Elongated versus round basic form.", "dimensionless"),
        (899, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_harmonic1_energy", "Energy of the first Fourier harmonic.", "FFT of the resampled XYZ curve; energy k=1.", "Elliptical fundamental part of the loop.", "mV^2"),
        (900, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_harmonic2_energy", "Energy of the second Fourier harmonic.", "FFT energy k=2 of the centered/resampled loop.", "Bimodality and deformation.", "mV^2"),
        (901, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_harmonic3_energy", "Energy of the third Fourier harmonic.", "FFT energy k=3.", "More complex loop shape.", "mV^2"),
        (902, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_high_order_energy", "Energy of higher harmonics.", "Sum of FFT energy k>=4.", "Roughness or fragmentation without peak detection.", "mV^2"),
        (903, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_phase_shift_xyz", "Phase shift of XYZ components.", "Compare phase(k=1) of X, Y and Z.", "Temporal axis offset in the spatial vector.", "degrees"),
        (904, "Mathematische Loop-Morphologie - Fourier-Deskriptoren", "loop_fourier_reconstruction_error_k3", "Reconstruction error with three harmonics.", "RMS(loop - inverse FFT using only k<=3).", "How well simple form explains the loop.", "mV"),
        (905, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_best_plane_residual_p95", "95th percentile distance to global main plane.", "P95 of signed or absolute PCA-plane residuals.", "Strong outliers from the plane.", "mV"),
        (906, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_plane_drift_start_end", "Start-end plane drift.", "Angle from first local plane to last local plane.", "Rotation of the loop plane over time.", "degrees"),
        (907, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_plane_outlier_fraction", "Fraction of local plane outliers.", "Fraction of sliding windows with angle to global plane above threshold.", "Local excursions out of the main plane.", "fraction"),
        (908, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_plane_twist_integral", "Integral of local plane change.", "Sum angle(n_i, n_{i+1}) along the sliding-plane sequence.", "Spatial twist complexity.", "degrees"),
        (909, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_signed_distance_to_plane_skewness", "Skewness of signed plane residuals.", "skew(signed distance to best plane).", "One-sided protrusion of the loop.", "dimensionless"),
        (910, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_sliding_plane_angle_mean", "Mean local plane angle.", "PCA plane in sliding window; mean angle to global plane.", "Local plane deviation.", "degrees"),
        (911, "Mathematische Loop-Morphologie - Gleitende Ebenen", "loop_sliding_plane_angle_std", "Standard deviation of local plane angle.", "std of angles of sliding normal vectors.", "Loop-plane instability.", "degrees"),
        (912, "Mathematische Loop-Morphologie - Graphfeatures", "loop_geodesic_euclidean_ratio", "Geodesic-to-Euclidean ratio.", "Path length along loop / direct distance.", "Trajectory detour.", "dimensionless"),
        (913, "Mathematische Loop-Morphologie - Graphfeatures", "loop_graph_bottleneck_distance", "Graph bottleneck distance.", "Minimum non-adjacent segment or point distance.", "Bottlenecks and self-approach.", "mV"),
        (914, "Mathematische Loop-Morphologie - Graphfeatures", "loop_graph_branchiness", "Graph branchiness index.", "Number of nodes with degree > 2 in epsilon graph.", "Folding or overlap in the point graph.", "count"),
        (915, "Mathematische Loop-Morphologie - Graphfeatures", "loop_graph_cycle_count_proxy", "Cycle proxy in the loop graph.", "edges - nodes + connected_components in epsilon graph.", "Loop complexity.", "count"),
        (916, "Mathematische Loop-Morphologie - Graphfeatures", "loop_knn_graph_length", "Total KNN graph length.", "Sum of k-nearest-neighbor edge lengths of the loop point cloud.", "Spatial compactness.", "mV"),
        (917, "Mathematische Loop-Morphologie - Graphfeatures", "loop_local_density_variance", "Variance of local neighbor density.", "Variance of point count inside radius epsilon.", "Clustering and uneven point density.", "dimensionless"),
        (918, "Mathematische Loop-Morphologie - Graphfeatures", "loop_revisit_index", "Revisit index.", "Fraction of points near earlier non-adjacent positions.", "Return or folding of trajectory.", "fraction"),
        (919, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_angular_coverage_deg", "Covered angular span.", "360 deg - missing_sector_angle, or range of unwrapped angles.", "Directional coverage of the loop.", "degrees"),
        (920, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_concavity_count", "Number of concave segments.", "Count segments with inward hull deviation above threshold.", "Multiple deformations.", "count"),
        (921, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_concavity_depth_max", "Maximum concavity depth.", "Maximum inward distance from loop to convex hull.", "Strongest concave defect.", "mV"),
        (922, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_convexity_ratio", "Convexity ratio.", "projected_loop_area / (convex_hull_area + eps).", "Concavity and defect sectors.", "dimensionless"),
        (923, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_hull_fill_ratio_3d", "3D hull fill ratio.", "Point-cloud or alpha-shape volume, or ellipsoid volume, divided by ConvexHull volume.", "Spatial filling versus thin curve.", "dimensionless"),
        (924, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_missing_sector_angle", "Largest missing angular sector.", "Maximum gap of polar angles around centroid.", "Missing direction sector, possible scar or conduction block hypothesis.", "degrees"),
        (925, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_radial_uniformity", "Radial uniformity.", "std(radius) / (mean(radius) + eps).", "Uneven bulging.", "dimensionless"),
        (926, "Mathematische Loop-Morphologie - Konvexitaet und Sektoren", "loop_sector_entropy", "Entropy of angular distribution.", "Histogram entropy of polar angles in main plane.", "Uniform versus sector-limited morphology.", "dimensionless"),
        (927, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_dihedral_angle_entropy", "Entropy of dihedral angles.", "Histogram entropy of local dihedral angles.", "Spatial complexity.", "dimensionless"),
        (928, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_dihedral_angle_mean", "Mean angle between local three-point planes.", "Local normals n_i=cross(r_i-r_{i-1}, r_{i+1}-r_i); mean(angle(n_i,n_{i+1})).", "Plane changes and spatial folding.", "degrees"),
        (929, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_dihedral_angle_std", "Standard deviation of local dihedral angles.", "std(angle(n_i,n_{i+1})).", "Loop-plane instability.", "degrees"),
        (930, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_plane_flip_count", "Number of large local plane jumps.", "count(dihedral_angle > 60 deg), threshold must be documented.", "Loop folding or local excursions from main plane.", "count"),
        (931, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_rotation_consistency", "Consistency of rotation.", "1 - std(rot_angle)/(mean(rot_angle)+eps).", "Uniform versus discontinuous rotation.", "dimensionless"),
        (932, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_rotational_angle_mean", "Mean local loop rotation.", "mean(angle(r_i, r_{i+1})) or tangent angle.", "Uniformity of loop rotation.", "degrees"),
        (933, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_rotational_angle_rmssd", "Short-term rotational-angle variability.", "RMSSD of local rotational angles.", "Abrupt direction changes.", "degrees"),
        (934, "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel", "loop_rotational_angle_std", "Standard deviation of local rotational angle.", "std(angle(r_i, r_{i+1})).", "Uneven rotation or fragmented activation.", "degrees"),
        (935, "Mathematische Loop-Morphologie - Topologie light", "loop_alpha_shape_area_proxy", "Alpha-shape area proxy.", "Hull area at multiple epsilon/alpha scales, approximated without heavy TDA.", "Nonconvex hull filling.", "mV^2"),
        (936, "Mathematische Loop-Morphologie - Topologie light", "loop_hole_radius_proxy", "Loop-hole radius proxy.", "median(radius)-p10(radius) around centroid.", "Size of central loop hole.", "mV"),
        (937, "Mathematische Loop-Morphologie - Topologie light", "loop_multiscale_area_stability", "Multiscale area stability.", "Coefficient of variation or slope of loop area across smoothing scales.", "Robust versus noise-driven form.", "dimensionless"),
        (938, "Mathematische Loop-Morphologie - Topologie light", "loop_multiscale_curvature_slope", "Multiscale curvature slope.", "Regression of curvature_energy against smoothing scale.", "Roughness or fractal-like loop shape.", "dimensionless"),
        (939, "Mathematische Loop-Morphologie - Topologie light", "loop_multiscale_intersection_count", "Multiscale self-intersections.", "Self-intersections after multiple smoothing/downsampling scales.", "Persistent folding versus noise.", "count"),
        (940, "Mathematische Loop-Morphologie - Topologie light", "loop_radial_peak_persistence", "Persistence of radial peaks.", "Mean prominence of radial peaks across smoothing scales.", "Stable bulges.", "mV"),
        (941, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_acceleration_burst_count", "Acceleration bursts.", "Peak count of ||acceleration||.", "Abrupt activation.", "count"),
        (942, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_curvature_speed_correlation", "Correlation of curvature and speed.", "corr(kappa(t), speed(t)).", "Braking in curves or fast direction changes.", "dimensionless"),
        (943, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_late_slowing_fraction", "Fraction of late slow movement.", "Fraction in the final third with speed < p25.", "Late conduction delay or terminal repolarization.", "fraction"),
        (944, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_shape_length_normalized", "Time-normalized loop shape length.", "Resample loop to N points and normalize length by magnitude/area.", "Pure shape independent of duration.", "dimensionless"),
        (945, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_speed_shape_correlation", "Correlation of radius and speed.", "corr(||r-centroid||, speed).", "Whether outer loop parts are traversed faster or slower.", "dimensionless"),
        (946, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_stop_go_index", "Stop-and-go index.", "fraction(speed < p10) + fraction(speed > p90).", "Discontinuous trajectory progression.", "fraction"),
        (947, "Mathematische Loop-Morphologie - Zeitnormalisierte Form", "loop_time_warping_energy", "Time-warping energy.", "DTW cost between actual parametrization and linear reference.", "Uneven activation or repolarization.", "dimensionless"),
        (948, "Morphologische Signalverarbeitung", "<segment>_fragmentation_index", "Fragmentation index.", "Peak and trough count per second.", "Notching, noise and conduction inhomogeneity.", "count/s"),
        (949, "Morphologische Signalverarbeitung", "<segment>_hilbert_amplitude/phase/frequency", "Hilbert features.", "Analytic signal: amplitude, phase and instantaneous frequency.", "Time-frequency morphology.", "mixed"),
        (950, "Morphologische Signalverarbeitung", "<segment>_hjorth_activity/mobility/complexity", "Hjorth parameters.", "Variance and derivative ratios.", "Shape activity and complexity.", "mixed"),
        (951, "Morphologische Signalverarbeitung", "<segment>_katz_fractal_dimension", "Katz fractal dimension.", "log(n)/(log(d/L)+log(n)).", "Roughness and complexity.", "dimensionless"),
        (952, "Morphologische Signalverarbeitung", "<segment>_magnitude_mean/std/rms", "Morphological magnitude statistics.", "Statistics over M(t) inside the segment.", "Signal shape and spread.", "mixed"),
        (953, "Morphologische Signalverarbeitung", "<segment>_notching_index", "Notching index.", "Peak count of the second derivative per unit time.", "High-frequency QRS and notching context.", "count/s"),
        (954, "Morphologische Signalverarbeitung", "<segment>_permutation_entropy_order3", "Permutation entropy.", "Ordinal-pattern entropy.", "Nonlinear shape complexity.", "dimensionless"),
        (955, "Morphologische Signalverarbeitung", "<segment>_total_power/centroid/bandwidth", "Spectral features.", "Welch PSD and derived total power, spectral centroid and bandwidth.", "Frequency content.", "mixed"),
        (956, "Morphologische Signalverarbeitung", "<segment>_wavelet_scale*_energy", "Wavelet energy.", "Ricker-wavelet convolution per scale.", "Multiscale shape; check noise.", "mixed"),
        (957, "P-Loop Morphologie extra", "p_loop_anterior_posterior_balance", "Anterior-posterior balance of the P loop.", "Area or time fraction with Z positive versus Z negative.", "Atrial propagation direction.", "dimensionless"),
        (958, "P-Loop Morphologie extra", "p_loop_biphasic_index", "Biphasic P-loop index.", "Twice the smaller positive/negative projection mass along the main P-loop axis, divided by total projection mass.", "Atrial activation with substantial counter-direction.", "dimensionless"),
        (959, "P-Loop Morphologie extra", "p_loop_double_rotation_index", "Double-rotation direction index in P.", "Sign changes of signed area rate inside P loop.", "Complex atrial activation.", "count"),
        (960, "P-Loop Morphologie extra", "p_loop_fractionated_motion_index", "P-loop fragmentation index.", "Peak count in P speed or P curvature divided by P duration.", "Atrial fragmentation; P-SNR is critical.", "count/s"),
        (961, "P-Loop Morphologie extra", "p_loop_interbeat_axis_variability", "Beat-to-beat P-axis variability.", "Standard deviation or angular spread of beat-wise P main vectors.", "Variable atrial conduction.", "degrees"),
        (962, "P-Loop Morphologie extra", "p_loop_left_right_balance", "Left-right balance of the P loop.", "Area or time fraction with X positive versus X negative.", "Right-/left-atrial dominance hypothesis.", "dimensionless"),
        (963, "P-Loop Morphologie extra", "p_loop_plane_instability", "Instability of the P-loop plane.", "std of local dihedral/sliding-plane angles inside P.", "Changing atrial propagation plane.", "degrees"),
        (964, "P-Loop Morphologie extra", "p_loop_template_instability", "P-loop template instability.", "Median distance of individual P loops to record-internal median P loop.", "P morphology variability.", "mV"),
        (965, "P-Loop Morphologie extra", "p_loop_terminal_force_vector", "Terminal P force vector.", "Mean or area vector of the last 20 ms of the P loop.", "Terminal atrial activation.", "uV*ms"),
        (966, "P-Loop Morphologie extra", "p_loop_terminal_to_initial_angle", "Initial-terminal P-loop angle.", "Angle between first 20-ms P vector and final 20-ms P vector.", "Intra-atrial conduction change.", "degrees"),
        (967, "P-QRS-T Loop-Beziehungen", "p_loop_to_qrs_loop_area_ratio", "P-to-QRS loop area ratio.", "P_area / (QRS_area + eps).", "Relative atrial activity size.", "dimensionless"),
        (968, "P-QRS-T Loop-Beziehungen", "p_qrs_loop_plane_angle", "Angle of P plane to QRS plane.", "angle(normal_P, normal_QRS).", "Atrial-ventricular orientation relationship.", "degrees"),
        (969, "P-QRS-T Loop-Beziehungen", "p_qrs_t_axis_consistency", "Consistency of P-QRS-T axes.", "Directional dispersion of P, QRS and T main vectors.", "Global electrical organization.", "dimensionless"),
        (970, "P-QRS-T Loop-Beziehungen", "p_qrs_t_plane_triad_volume", "Volume of the three loop normal vectors.", "abs(det([n_P, n_QRS, n_T])).", "Three-dimensional organization of the three loop planes.", "dimensionless"),
        (971, "P-QRS-T Loop-Beziehungen", "p_t_loop_plane_angle", "Angle of P plane to T plane.", "angle(normal_P, normal_T).", "Atrial versus repolarization direction.", "degrees"),
        (972, "P-QRS-T Loop-Beziehungen", "qrs_t_loop_shape_distance", "Shape distance from QRS to T loop.", "Procrustes or DTW after resampling and normalization.", "Depolarization/repolarization shape coupling.", "dimensionless"),
        (973, "P-QRS-T Loop-Beziehungen", "t_loop_to_qrs_loop_complexity_ratio", "T-to-QRS complexity ratio.", "T_complexity / (QRS_complexity + eps), e.g. curvature entropy or high-order Fourier energy.", "Repolarization versus depolarization complexity.", "dimensionless"),
        (974, "PCA/SVD", "<segment>_lead8_effective_rank", "Effective rank of eight independent leads.", "exp(Shannon entropy of eigenvalue shares).", "Signal complexity degree.", "dimensionless"),
        (975, "PCA/SVD", "<segment>_lead8_lambda1..lambda8", "SVD eigenvalues of eight independent leads.", "SVD/PCA of the lead matrix per segment.", "Dipolarity and nondipolarity.", "uV^2"),
        (976, "PCA/SVD", "<segment>_lead8_nondipolar_fraction", "Nondipolar fraction.", "sum(lambda4..)/sum(lambda).", "Residual or complexity marker.", "fraction"),
        (977, "PCA/SVD", "jackknife_influence_<lead>", "Lead jackknife influence.", "Change in residual after removing one lead.", "Lead robustness.", "mixed"),
        (978, "PCA/SVD", "residual_rms_<lead>", "Lead-specific residual RMS.", "RMS after rank-3 reconstruction.", "Lead outlier or nondipolarity.", "uV"),
        (979, "Pathophysiologische Surrogate", "activation_inefficiency", "Activation inefficiency.", "QRS path length / (max displacement + eps).", "Detour and conduction inhomogeneity.", "dimensionless"),
        (980, "Pathophysiologische Surrogate", "acute_ischemia_vector_score", "Acute ischemia vector score.", "bounded(ST magnitude / QRS + ST-T angle/180).", "Exploratory ST-T score.", "dimensionless"),
        (981, "Pathophysiologische Surrogate", "conduction_detour_index", "Conduction detour index.", "max(0, (closed QRS path - 2*diameter)/(2*diameter)).", "Detoured activation beyond the minimum out-and-back trajectory.", "dimensionless"),
        (982, "Pathophysiologische Surrogate", "depolarization_repolarization_mismatch", "Depolarization/repolarization mismatch.", "Angle between QRS and T vectors.", "Repolarization heterogeneity.", "degrees"),
        (983, "Pathophysiologische Surrogate", "initial_qrs_loss_index", "Initial QRS loss index.", "1 - ||early_QRS||/(QRS_peak + eps).", "Infarction or scar surrogate.", "dimensionless"),
        (984, "Pathophysiologische Surrogate", "late_qrs_vector_magnitude", "Late QRS magnitude.", "||late_QRS_vector||.", "Late activation and conduction delay.", "mV"),
        (985, "Pathophysiologische Surrogate", "lbbb_like_vector_score", "LBBB-like vector score.", "bounded(qrs_duration/160 + late-component/qrs_peak).", "Exploratory score, not a diagnosis.", "dimensionless"),
        (986, "Pathophysiologische Surrogate", "lvh_vector_score / rvh_vector_score", "Hypertrophy vector surrogates.", "bounded(QRS magnitude/area or axis component).", "Not classic diagnostic criteria; VCG screening only.", "dimensionless"),
        (987, "Pathophysiologische Surrogate", "qrs_loop_notch_burden", "QRS notch burden.", "Peak count of QRS dynamics.", "Fragmentation and scar context.", "count"),
        (988, "Pathophysiologische Surrogate", "qt_prolongation/shortening_surrogate", "QT prolongation/shortening proxy.", "qt_duration/450 or 320/qt_duration.", "QT context; inspect T end.", "dimensionless"),
        (989, "Pathophysiologische Surrogate", "rbbb_like_terminal_vector_score", "RBBB-like terminal score.", "bounded(qrs_duration/140 + terminal anterior/posterior component).", "Exploratory score.", "dimensionless"),
        (990, "Pathophysiologische Surrogate", "scar_vector_index", "Scar vector index.", "Aggregation of initial loss, deformation and notch burden.", "Hypothesis marker for scar.", "dimensionless"),
        (991, "Pathophysiologische Surrogate", "st_qrs_discordance_angle_deg", "ST-QRS discordance.", "Angle from ST to QRS vector.", "Ischemia and reciprocity context.", "degrees"),
        (992, "Pathophysiologische Surrogate", "st_vector_magnitude", "ST vector magnitude.", "||ST_vector||.", "Spatial ST burden.", "mV"),
        (993, "Pathophysiologische Surrogate", "terminal_qrs_rightward/leftward/posterior/anterior_index", "Terminal direction indices.", "Positive and negative components of late_QRS_vector.", "Bundle-branch or fascicular-pattern surrogates.", "mixed"),
        (994, "QRS-T/Repolarisation", "qrs_area_to_t_area_angle_deg", "Published area-based spatial QRS-T angle.", "Angle between origin-corrected QRS-onset:QRS-offset area and complete JT (QRS-offset:T-offset) area vectors.", "Repolarization heterogeneity.", "degrees"),
        (995, "QRS-T/Repolarisation", "qrs_loop_area_t_loop_area_ratio", "QRS/T loop area ratio.", "QRS_area_best_fit_plane / (T_area_best_fit_plane + eps).", "Depolarization/repolarization geometry.", "dimensionless"),
        (996, "QRS-T/Repolarisation", "qrs_loop_volume_t_loop_volume_ratio", "QRS/T loop volume ratio.", "QRS ellipsoid volume / (T ellipsoid volume + eps).", "Spatial loop size.", "dimensionless"),
        (997, "QRS-T/Repolarisation", "qrs_t_cosine_distance", "QRS-T cosine distance.", "1 - cos(flattened curves).", "Shape and directional similarity.", "dimensionless"),
        (998, "QRS-T/Repolarisation", "qrs_t_dtw_distance", "QRS-T DTW distance.", "Dynamic time warping between resampled loops.", "Time-warping robust shape distance.", "dimensionless"),
        (999, "QRS-T/Repolarisation", "qrs_t_hausdorff_distance", "QRS-T Hausdorff distance.", "Bidirectional Hausdorff distance.", "Extreme-point and shape distance.", "mV"),
        (1000, "QRS-T/Repolarisation", "qrs_t_morphological_discordance", "QRS-T morphological discordance.", "Mean of bounded cosine, centered/RMS-normalized DTW and Procrustes shape distances.", "Exploratory repolarization mismatch without mixing physical units.", "dimensionless"),
        (1001, "QRS-T/Repolarisation", "qrs_t_procrustes_distance", "QRS-T Procrustes distance.", "Scaled/rotated curve deviation.", "Shape comparison independent of rotation/scaling.", "dimensionless"),
        (1002, "QRS-T/Repolarisation", "qrs_t_vector_correlation_distance", "QRS-T correlation distance.", "1 - corr(resampled QRS curve, T curve).", "Morphological similarity.", "dimensionless"),
        (1003, "QRS-T/Repolarisation", "spatial_qrs_t_angle_deg", "Spatial area QRS-T angle.", "Angle between QRS-onset:QRS-offset area and JT (QRS-offset:T-offset) area vectors after physiological-origin correction.", "Depolarization/repolarization discordance.", "degrees"),
        (1004, "QRS-T/Repolarisation", "svg_magnitude", "Wilson spatial ventricular gradient magnitude.", "Norm of the origin-corrected QRS-onset:T-offset area vector.", "Ventricular-gradient burden.", "uV*ms"),
        (1005, "QRS-T/Repolarisation", "terminal_repolarization_burden", "Terminal repolarization burden.", "||late_T_vector||.", "T-end instability.", "mV"),
        (1006, "Qualitaet/Elektroden", "chest_lead_misplacement_probability_proxy", "Chest lead misplacement probability proxy.", "Error in R-wave progression.", "Lead-placement hint, not proof.", "dimensionless"),
        (1007, "Qualitaet/Elektroden", "einthoven_relation_error_rms", "Einthoven relation error.", "RMS(III - (II - I)).", "Limb-lead quality control.", "uV"),
        (1008, "Qualitaet/Elektroden", "kors_vs_pca_axis_difference_deg", "Kors-vs-PCA axis difference.", "Angle between mean Kors XYZ and PCA3 axis.", "Transformation robustness.", "degrees"),
        (1009, "Qualitaet/Elektroden", "residual_per_lead_<lead>", "Residual per lead.", "RMS after rank-3 reconstruction.", "Lead outlier or electrode error.", "uV"),
        (1010, "Qualitaet/Elektroden", "vcg_trust_score", "VCG trust score.", "1 - aggregation of fiducial, transformation and lead errors.", "Weighting all markers.", "dimensionless"),
        (1011, "R-Wellen-Progression", "poor_r_wave_progression_surrogate", "Negative progression fraction.", "max(0, -slope).", "MI, lead placement and hypertrophy context; not specific.", "dimensionless"),
        (1012, "R-Wellen-Progression", "r_wave_progression_slope_v1_v6", "Slope of R amplitudes from V1 to V6.", "Linear regression of R amplitude against lead index 1..6.", "Poor R-wave progression context.", "uV/lead"),
        (1013, "Raeumliche Magnitude", "spatial_low_voltage_qrs_peak_magnitude", "Peak QRS magnitude as low-voltage context.", "median max(||r||) in QRS.", "Check low voltage, hypertrophy and transformation influence.", "mV"),
        (1014, "Raeumliche Magnitude", "spatial_magnitude_integral", "Integral descriptor of vector magnitude.", "Integral M(t) dt.", "Global strength of the electrical spatial vector.", "mV*s"),
        (1015, "Raeumliche Magnitude", "spatial_magnitude_max", "Maximum descriptor of vector magnitude.", "max(M).", "Global strength of the electrical spatial vector.", "mV"),
        (1016, "Raeumliche Magnitude", "spatial_magnitude_mean", "Mean descriptor of vector magnitude.", "mean(M).", "Global strength of the electrical spatial vector.", "mV"),
        (1017, "Raeumliche Magnitude", "spatial_magnitude_median", "Median descriptor of vector magnitude.", "median(M).", "Global strength of the electrical spatial vector.", "mV"),
        (1018, "Raeumliche Magnitude", "spatial_magnitude_p95", "95th percentile descriptor of vector magnitude.", "95th percentile of M.", "Global strength of the electrical spatial vector.", "mV"),
        (1019, "Raeumliche Magnitude", "spatial_magnitude_rms", "RMS descriptor of vector magnitude.", "sqrt(mean(M^2)).", "Global strength of the electrical spatial vector.", "mV"),
        (1020, "ST-Zeitpunkte", "st20_deviation_mean_abs", "Mean absolute ST deviation 20 ms after J point.", "Median over beats: mean(abs(lead value - baseline)) at QRS_offset + 20 ms.", "Ischemia/artifact context; QRS offset is critical.", "uV"),
        (1021, "ST-Zeitpunkte", "st20_elevation_max_lead / depression_min_lead", "Maximum elevation/depression 20 ms after J point.", "max/min over leads after baseline correction.", "Inspect ST elevation/depression vectors.", "uV"),
        (1022, "ST-Zeitpunkte", "st40_deviation_mean_abs", "Mean absolute ST deviation 40 ms after J point.", "Median over beats: mean(abs(lead value - baseline)) at QRS_offset + 40 ms.", "Ischemia/artifact context; QRS offset is critical.", "uV"),
        (1023, "ST-Zeitpunkte", "st40_elevation_max_lead / depression_min_lead", "Maximum elevation/depression 40 ms after J point.", "max/min over leads after baseline correction.", "Inspect ST elevation/depression vectors.", "uV"),
        (1024, "ST-Zeitpunkte", "st60_deviation_mean_abs", "Mean absolute ST deviation 60 ms after J point.", "Median over beats: mean(abs(lead value - baseline)) at QRS_offset + 60 ms.", "Ischemia/artifact context; QRS offset is critical.", "uV"),
        (1025, "ST-Zeitpunkte", "st60_elevation_max_lead / depression_min_lead", "Maximum elevation/depression 60 ms after J point.", "max/min over leads after baseline correction.", "Inspect ST elevation/depression vectors.", "uV"),
        (1026, "ST-Zeitpunkte", "st80_deviation_mean_abs", "Mean absolute ST deviation 80 ms after J point.", "Median over beats: mean(abs(lead value - baseline)) at QRS_offset + 80 ms.", "Ischemia/artifact context; QRS offset is critical.", "uV"),
        (1027, "ST-Zeitpunkte", "st80_elevation_max_lead / depression_min_lead", "Maximum elevation/depression 80 ms after J point.", "max/min over leads after baseline correction.", "Inspect ST elevation/depression vectors.", "uV"),
        (1028, "Segment-Magnitudenflaechen", "magnitude_area_qrs", "Integral of vector magnitude in QRS segment.", "Integral ||r(t)|| dt in the segment.", "Segmental activity load.", "mV*s"),
        (1029, "Segment-Magnitudenflaechen", "magnitude_area_qrst", "Integral of vector magnitude in QRST segment.", "Integral ||r(t)|| dt in the segment.", "Segmental activity load.", "mV*s"),
        (1030, "Segment-Magnitudenflaechen", "magnitude_area_st_t", "Integral of vector magnitude in ST-T segment.", "Integral ||r(t)|| dt in the segment.", "Segmental activity load.", "mV*s"),
        (1031, "Segment-spezifisch", "initial_5/10/20/40ms_qrs_vector", "Initial QRS vectors.", "mean(r) in first 5/10/20/40 ms.", "Septal and initial activation.", "mV"),
        (1032, "Segment-spezifisch", "j_point_vector", "J-point vector.", "r(QRS offset).", "ST start point.", "mV"),
        (1033, "Segment-spezifisch", "p_area_vector", "P area vector.", "Integral r(t) dt in P segment.", "Atrial area and direction marker.", "mV*s"),
        (1034, "Segment-spezifisch", "p_low_voltage_index", "P low-voltage index.", "1/(||P_mean|| + eps).", "Small P signal and noise vulnerability.", "1/mV"),
        (1035, "Segment-spezifisch", "p_mean_vector", "P mean vector.", "mean(r) in P segment.", "Atrial activity; P detection is critical.", "mV"),
        (1036, "Segment-spezifisch", "p_vector_magnitude", "P vector magnitude.", "||P_mean_vector||.", "P low voltage and atrial conduction.", "mV"),
        (1037, "Segment-spezifisch", "pr_baseline_instability", "PR baseline instability.", "std of PR points over beats.", "Artifact or atrial repolarization.", "mV"),
        (1038, "Segment-spezifisch", "pr_drift_magnitude", "PR drift.", "||r_end - r_start|| in PR.", "Baseline and PR shift.", "mV"),
        (1039, "Segment-spezifisch", "pr_qrs_angle_deg", "PR-QRS angle.", "Angle from PR vector to QRS vector.", "Segment relation.", "degrees"),
        (1040, "Segment-spezifisch", "pr_segment_slope_magnitude", "PR slope.", "||r_end - r_start|| / duration.", "PR deviation dynamics.", "mV/s"),
        (1041, "Segment-spezifisch", "pr_st_angle_deg", "PR-ST angle.", "Angle from PR vector to ST vector.", "ST/PR coupling.", "degrees"),
        (1042, "Segment-spezifisch", "pr_t_angle_deg", "PR-T angle.", "Angle from PR vector to T vector.", "Repolarization relation.", "degrees"),
        (1043, "Segment-spezifisch", "pr_vector", "PR vector.", "mean(r) in PR segment.", "AV or atrial repolarization surrogate.", "mV"),
        (1044, "Segment-spezifisch", "qrs_early/mid/late_mean_vector", "QRS third vectors.", "mean(r) in QRS thirds.", "Activation propagation.", "mV"),
        (1045, "Segment-spezifisch", "st20/st40/st60/st80_vector", "ST timepoint vectors.", "r(QRS offset + 20/40/60/80 ms).", "Spatial ST elevation/depression.", "mV"),
        (1046, "Segment-spezifisch", "st_area_vector", "ST area vector.", "Integral r(t) dt in ST segment.", "ST burden.", "mV*s"),
        (1047, "Segment-spezifisch", "st_loop_drift", "ST loop drift.", "||ST_end - ST_start||.", "Ischemia and baseline context.", "mV"),
        (1048, "Segment-spezifisch", "st_qrs_discordance_deg", "ST-QRS discordance.", "Angle from ST vector to QRS vector.", "ST/QRS relation.", "degrees"),
        (1049, "Segment-spezifisch", "st_slope_vector", "ST slope vector.", "(ST80 - ST20) / 60 ms.", "ST dynamics.", "mV/s"),
        (1050, "Segment-spezifisch", "st_t_continuity_index", "ST-T continuity.", "1 - ||ST - T||/(||ST|| + ||T|| + eps).", "Transition from ST to T.", "dimensionless"),
        (1051, "Segment-spezifisch", "t_area_vector", "Published GEH T-area (JT) vector.", "Integral (r(t)-origin) dt from QRS offset through T offset.", "Repolarization including the ST/JT interval.", "mV*s"),
        (1052, "Segment-spezifisch", "t_downstroke_vector", "T downstroke vector.", "mean(r) from T peak to T end.", "T falling phase.", "mV"),
        (1053, "Segment-spezifisch", "t_upstroke_downstroke_angle_deg", "T upstroke/downstroke angle.", "Angle between the two T sub-vectors.", "T asymmetry.", "degrees"),
        (1054, "Segment-spezifisch", "t_upstroke_vector", "T upstroke vector.", "mean(r) from T onset to T peak.", "T rising phase.", "mV"),
        (1055, "Segment-spezifisch", "terminal_5/10/20/40ms_qrs_vector", "Terminal QRS vectors.", "mean(r) in final 5/10/20/40 ms.", "Bundle-branch block and late activation.", "mV"),
        (1056, "Segment-spezifisch", "terminal_qrs_angle_relative_to_main_qrs_deg", "Terminal QRS to main QRS angle.", "Angle from terminal vector to QRS mean vector.", "Terminal conduction delay.", "degrees"),
        (1057, "Segment-spezifisch", "tpeak_to_tend_vector", "Tpeak-to-Tend vector.", "r(Tend) - r(Tpeak), median.", "Terminal repolarization.", "mV"),
        (1058, "Segment-spezifisch", "u_wave_candidate_magnitude", "U-wave candidate magnitude.", "max M(t) in window after T end before next QRS.", "Uncertain heuristic; review hint only.", "mV"),
        (1059, "Segment-spezifisch", "u_wave_detected_flag/confidence", "U-wave flag and confidence.", "Flag and low confidence in plausible post-T window.", "Do not interpret as hard measurement.", "mixed"),
        (1060, "Signalbasis", "delineation_lead", "Lead used for P/QRS/T delineation.", "Lead II preferred; fallback lead by finite fraction and amplitude.", "Explains faulty segmentation visually.", "categorical"),
        (1061, "Signalbasis", "delineation_method", "Documented wave-detection method.", "NeuroKit2 DWT preferred, then WFDB XQRS or SciPy fallback.", "Traceability of P/QRS/T points.", "categorical"),
        (1062, "Signalbasis", "fiducial_confidence", "Confidence of automatic P/QRS/T delineation.", "Combination of RR plausibility, available P/T points and beat quality.", "Methodological check axis for all segment-dependent biomarkers.", "dimensionless"),
        (1063, "Signalbasis", "quality_score", "Global signal-quality context.", "Aggregated from finite fraction, noise, baseline wander and flat leads.", "Not a pathology marker; filters review candidates and artifacts.", "dimensionless"),
        (1064, "Signalbasis", "spatial_magnitude_m_t", "Spatial electrical vector magnitude.", "M(t)=sqrt(X(t)^2+Y(t)^2+Z(t)^2).", "Scaling, electrode position and noise affect all magnitude features.", "mV"),
        (1065, "Signalbasis", "vcg_axes_xyz", "Kors-like reconstructed XYZ VCG axes from 12-lead ECG.", "Matrix transform of standard leads to r(t)=[X,Y,Z].", "Foundation of all VCG features; check transform and lead quality.", "mV"),
        (1066, "T-Loop Morphologie extra", "t_loop_apex_sharpness", "Sharpness at T apex.", "Menger curvature of the VCG trajectory around T peak using +/-8 ms support.", "Pointed versus round T loop.", "1/mV"),
        (1067, "T-Loop Morphologie extra", "t_loop_bulging_index", "T-loop bulging index.", "Max distance of T loop from the line connecting T_on and T_end.", "Bulging of the repolarization loop.", "mV"),
        (1068, "T-Loop Morphologie extra", "t_loop_late_area_fraction", "Late T areal-sweep fraction.", "Absolute areal sweep in final T third / total absolute T-loop areal sweep in the same best-fit plane.", "Terminal repolarization burden.", "fraction"),
        (1069, "T-Loop Morphologie extra", "t_loop_qrs_plane_escape_angle", "T-loop escape from QRS plane.", "Angle of T-loop normal vector to QRS-loop normal vector.", "Repolarization outside depolarization plane.", "degrees"),
        (1070, "T-Loop Morphologie extra", "t_loop_repolarization_tail_index", "Terminal repolarization tail.", "Path length in the final temporal third of T divided by total T-loop path.", "Disproportionate terminal repolarization trajectory.", "dimensionless"),
        (1071, "T-Loop Morphologie extra", "t_loop_roundness_vs_planarity_ratio", "Roundness-to-planarity ratio.", "roundness / (planarity + eps).", "Round-but-nonplanar or flat-elliptical shape.", "dimensionless"),
        (1072, "T-Loop Morphologie extra", "t_loop_terminal_dispersion", "Terminal T dispersion.", "Spread of final T vectors or plane residuals at T end.", "Terminal repolarization heterogeneity.", "mV"),
        (1073, "T-Loop Morphologie extra", "t_loop_upstroke_downstroke_angle", "Angle from T upstroke to T downstroke.", "Angle of mean or area vectors of both T parts.", "Direction change in T.", "degrees"),
        (1074, "T-Loop Morphologie extra", "t_loop_upstroke_downstroke_area_ratio", "T upstroke/downstroke area ratio.", "Area(T_on-T_peak) / (Area(T_peak-T_end) + eps).", "Asymmetric repolarization.", "dimensionless"),
        (1075, "T-Loop Morphologie extra", "t_loop_vector_reversal_index", "Backward fraction of the T loop.", "Fraction of tangents with dot(tangent, mean_T_direction) < 0.", "Direction reversal in repolarization.", "fraction"),
        (1076, "Template/Distanz", "<segment>_record_template_*_distance", "Record-internal template distances.", "Beat-to-median template with Euclidean, cosine, correlation, Procrustes, DTW or Hausdorff distance.", "ECG morphology stability.", "mixed"),
        (1077, "Template/Distanz", "external_norm/pathology_template_bank_available_flag", "Template-bank availability flag.", "0 while no external template bank is loaded.", "Prevents invented pathology templates.", "binary"),
        (1078, "Template/Distanz", "qrs_internal_cluster_distance_median", "Internal QRS cluster distance.", "Distance of resampled QRS beats to the centroid.", "Rare or variable morphology.", "mV"),
        (1079, "Ventricular Gradient", "spatial_ventricular_gradient_vector", "Wilson spatial ventricular gradient.", "Origin-corrected area vector integrated continuously from QRS onset through T offset (equivalent to QRS + JT area).", "Coupling of depolarization and repolarization.", "uV*ms"),
        (1080, "Ventricular Gradient", "svg_angle_to_qrs_area_deg", "Angle of SVG to QRS area vector.", "Angle between SVG and QRS area vector.", "Repolarization heterogeneity and vector deviation.", "degrees"),
        (1081, "Ventricular Gradient", "svg_angle_to_t_area_deg", "Angle of SVG to T area vector.", "Angle between SVG and T area vector.", "Repolarization heterogeneity and vector deviation.", "degrees"),
        (1082, "Ventricular Gradient", "svg_qrs_area_ratio", "Magnitude ratio of SVG to QRS area vector.", "||SVG|| / (||QRS_area|| + eps).", "Repolarization heterogeneity and vector deviation.", "dimensionless"),
        (1083, "Ventricular Gradient", "svg_t_area_ratio", "Magnitude ratio of SVG to T area vector.", "||SVG|| / (||T_area|| + eps).", "Repolarization heterogeneity and vector deviation.", "dimensionless"),
    ]

    angle_names = [
        (1084, "early_late_qrs_angle_deg"),
        (1085, "early_late_t_angle_deg"),
        (1086, "frontal_xy_qrs_t_angle_deg"),
        (1087, "horizontal_xz_qrs_t_angle_deg"),
        (1088, "initial_terminal_qrs_angle_deg"),
        (1089, "j_vector_to_qrs_axis_angle_deg"),
        (1090, "max_qrs_to_max_t_angle_deg"),
        (1091, "p_qrs_angle_deg"),
        (1092, "p_qrs_loop_plane_angle_deg"),
        (1093, "p_t_angle_deg"),
        (1094, "p_t_loop_plane_angle_deg"),
        (1095, "qrs_area_to_t_area_angle_deg_axis_relation"),
        (1096, "qrs_axis_to_t_axis_angle_deg"),
        (1097, "qrs_st_angle_deg"),
        (1098, "qrs_t_loop_plane_angle_deg"),
        (1099, "sagittal_yz_qrs_t_angle_deg"),
        (1100, "spatial_qrs_t_angle_deg_catalog"),
        (1101, "st_t_angle_deg"),
        (1102, "st_vector_to_tpeak_angle_deg"),
    ]
    rows.extend([
        (
            index,
            "Winkel/Achsenbeziehungen",
            name,
            "Angle between two physiologically defined vectors or loop planes.",
            "arccos(dot(a,b)/(||a|| ||b||)) in degrees; for planes use normal vectors.",
            "Discordance, conduction delay and repolarization heterogeneity.",
            "degrees",
        )
        for index, name in angle_names
    ])

    rows.extend([
        (1103, "Zeitintervalle", "electric_diastole_ms", "Electrical diastole.", "RR - QT.", "Heart-rate and repolarization context.", "ms"),
        (1104, "Zeitintervalle", "electric_systole_ms", "Electrical systole.", "Alias/proxy for QT or QRST duration.", "Comparison with diastole/RR.", "ms"),
        (1105, "Zeitintervalle", "heart_rate_bpm", "Median heart rate.", "60 / median(RR in seconds).", "Rhythm status, tachy/brady context.", "bpm"),
        (1106, "Zeitintervalle", "jt_interval_ms", "JT interval.", "QRS offset to T end.", "Repolarization without QRS duration component.", "ms"),
        (1107, "Zeitintervalle", "jtc_bazett/fridericia/framingham_ms", "JTc variants.", "QTc - QRS duration.", "Useful adjunct when QRS is wide.", "ms"),
        (1108, "Zeitintervalle", "p_duration_ms", "P duration.", "P onset to P offset.", "Atrial conduction time.", "ms"),
        (1109, "Zeitintervalle", "p_onset_to_t_end_ms", "P onset to T end.", "P onset to T end.", "Total electrical activity duration of the beat.", "ms"),
        (1110, "Zeitintervalle", "pr_interval_ms", "PR interval.", "P onset to QRS onset, median over beats.", "P fiducials critical; check AF/artifact.", "ms"),
        (1111, "Zeitintervalle", "qrs_duration_ms", "QRS duration.", "QRS onset to QRS offset.", "Conduction delay; inspect QRS end.", "ms"),
        (1112, "Zeitintervalle", "qrs_early/mid/late_duration_ms", "QRS third durations.", "QRS split into three equal time intervals.", "Activation spread inside QRS.", "ms"),
        (1113, "Zeitintervalle", "qt_interval_ms", "QT interval.", "QRS onset to T end.", "T end and U wave are critical.", "ms"),
        (1114, "Zeitintervalle", "qtc_bazett_ms", "QTc Bazett.", "QT / sqrt(RR[s]).", "Heart-rate correction; limited at high/low HR.", "ms"),
        (1115, "Zeitintervalle", "qtc_framingham_ms", "QTc Framingham.", "QT + 154*(1 - RR[s]).", "Linear QTc correction.", "ms"),
        (1116, "Zeitintervalle", "qtc_fridericia_ms", "QTc Fridericia.", "QT / RR[s]^(1/3).", "More robust QTc variant.", "ms"),
        (1117, "Zeitintervalle", "rr_interval_ms", "RR interval.", "median(diff(R peaks))/fs*1000.", "Basis for QTc and HRV.", "ms"),
        (1118, "Zeitintervalle", "rr_variance_ms2", "RR variance.", "var(diff(R peaks) in ms).", "Rhythm irregularity.", "ms^2"),
        (1119, "Zeitintervalle", "st_duration_ms", "ST duration.", "ST window length after QRS end.", "QRS offset determines the window.", "ms"),
        (1120, "Zeitintervalle", "t_duration_ms", "T duration.", "T onset to T end.", "Inspect T morphology and U waves.", "ms"),
        (1121, "Zeitintervalle", "t_upstroke/downstroke_duration_ms", "T upstroke/downstroke duration.", "T onset to T peak and T peak to T end.", "T asymmetry and repolarization shape.", "ms"),
        (1122, "Zeitintervalle", "tpeak_tend_ms", "Tpeak-Tend interval.", "T peak to T end.", "Repolarization dispersion; T fiducials critical.", "ms"),
        (1123, "Zusaetzliche VCG-Idee", "activation_front_spread_velocity_proxy", "Activation-front spread velocity proxy.", "QRS path length / QRS duration, additionally split early/late.", "Conduction spread versus duration.", "mV/s"),
        (1124, "Zusaetzliche VCG-Idee", "axis_crossing_sequence_entropy", "Axis-crossing sequence entropy.", "Sequence of X/Y/Z zero crossings and entropy/pattern counts.", "Symbolic morphology.", "dimensionless"),
        (1125, "Zusaetzliche VCG-Idee", "beatwise_loop_barycenter_motion", "Motion of loop barycenter over beats.", "Path length of barycenters across beat sequence.", "Morphological instability.", "mV"),
        (1126, "Zusaetzliche VCG-Idee", "breathing_baseline_vector_swing", "Respiratory/baseline vector swing.", "Low-frequency XYZ centroid motion.", "Volume conductor, electrode and baseline context.", "mV"),
        (1127, "Zusaetzliche VCG-Idee", "curvature_torsion_phase_coupling", "Curvature-torsion coupling.", "Mutual information or correlation between curvature(t) and torsion(t).", "3D dynamics complexity.", "dimensionless"),
        (1128, "Zusaetzliche VCG-Idee", "curvature_velocity_coupling", "Curvature-velocity coupling.", "corr(curvature(t), speed(t)) per segment.", "Whether strong direction changes occur at high or low speed.", "dimensionless"),
        (1129, "Zusaetzliche VCG-Idee", "directional_concentration_kappa_proxy", "Directional concentration proxy.", "||mean(unit(r))||; low value indicates broad directional spread.", "Directional stability of loops.", "dimensionless"),
        (1130, "Zusaetzliche VCG-Idee", "ectopic_beat_vector_deviation", "Ectopic beat vector deviation.", "Distance of single-beat loop to median sinus-beat loop.", "Analyze extrasystoles separately; do not mix them as pathology.", "mV"),
        (1131, "Zusaetzliche VCG-Idee", "electrical_action_density_map", "Electrical action density.", "Local density of M*speed in 3D bins.", "Spatial activity clusters.", "mixed"),
        (1132, "Zusaetzliche VCG-Idee", "geodesic_direction_path_length", "Geodesic direction path length.", "Sum angular distances between normalized r(t).", "Direction changes without amplitude influence.", "degrees"),
        (1133, "Zusaetzliche VCG-Idee", "lead_to_vcg_consistency_score", "Lead-to-VCG consistency.", "Regress 12 leads to XYZ and back; reconstruction error per lead.", "Transformation and lead quality.", "dimensionless"),
        (1134, "Zusaetzliche VCG-Idee", "local_plane_drift_index", "Local plane drift.", "Sliding PCA normal vectors; std of angles.", "Nonplanar dynamic loop motion.", "degrees"),
        (1135, "Zusaetzliche VCG-Idee", "loop_area_density", "Loop area density.", "area_best_fit_plane / path_length or / duration.", "Compact versus extended loop form.", "mV"),
        (1136, "Zusaetzliche VCG-Idee", "loop_chirality_index_3d", "3D chirality index.", "Sign/magnitude of mean(dot(r, v x a)).", "Handedness and rotation of the 3D loop.", "mV^3/s^2"),
        (1137, "Zusaetzliche VCG-Idee", "loop_energy_density", "Loop energy density.", "Integral M(t)^2 dt / path_length.", "Whether high voltage occurs on short or long path.", "mV*s"),
        (1138, "Zusaetzliche VCG-Idee", "medication_sensitive_repolarization_score", "Medication/electrolyte repolarization score.", "Combination of QTc, JTc, T-loop sharpening, Tpeak-Tend and SVG deviation.", "Hypothesis for QT/electrolyte/medication effects; do not infer cause.", "dimensionless"),
        (1139, "Zusaetzliche VCG-Idee", "multi_resolution_qrs_shape_signature", "Multiscale QRS shape signature.", "Loop metrics on resampled/smoothed QRS curves at multiple scales.", "Robustness against noise versus true form.", "mixed"),
        (1140, "Zusaetzliche VCG-Idee", "pacemaker_capture_vector_signature", "Pacemaker capture vector signature.", "Initial QRS steepness, spike proxy, terminal axis and broad QRS loop dynamics.", "Subgroup/quality analysis only when structured pacemaker flag exists.", "dimensionless"),
        (1141, "Zusaetzliche VCG-Idee", "persistent_homology_loop_lifetime", "Retired persistent-homology candidate.", "No scalar is emitted until a validated persistence-diagram implementation is available.", "Historical name retained for provenance; not evaluable.", "mV"),
        (1142, "Zusaetzliche VCG-Idee", "qrs_t_directional_wasserstein", "Wasserstein distance of QRS/T directions.", "Optimal assignment of amplitude-weighted unit-direction quantiles with angular geodesic cost divided by pi.", "Depolarization/repolarization direction-distribution difference.", "dimensionless"),
        (1143, "Zusaetzliche VCG-Idee", "qrs_t_loop_mirror_symmetry", "QRS-T mirror symmetry.", "Bounded RMS distance between centered, RMS-normalized QRS and sign-inverted T trajectories in fixed VCG axes.", "Check plausible repolarization counter-direction.", "dimensionless"),
        (1144, "Zusaetzliche VCG-Idee", "repolarization_front_spread_velocity_proxy", "Repolarization-front spread velocity proxy.", "T path length / T duration.", "T-loop spread and repolarization duration.", "mV/s"),
        (1145, "Zusaetzliche VCG-Idee", "repolarization_return_efficiency", "Efficiency of return to baseline after T.", "Direct T-peak-to-end distance divided by terminal T path length.", "Efficient versus detoured terminal repolarization return.", "dimensionless"),
        (1146, "Zusaetzliche VCG-Idee", "segmental_directional_memory", "Directional memory between segments.", "Cosine similarity between end direction of QRS and initial direction of ST/T.", "Continuity from activation to repolarization.", "dimensionless"),
        (1147, "Zusaetzliche VCG-Idee", "sex_age_adjusted_residual_candidate", "Age/sex-adjusted residual marker.", "Not evaluable per record until a train-only normative age/sex model has been fitted.", "Only with normative reference; no zero placeholder is emitted.", "z-score"),
        (1148, "Zusaetzliche VCG-Idee", "spherical_harmonic_vcg_signature", "Spherical signature of VCG direction.", "Projection of normalized r(t) onto coarse spherical-harmonic/bin basis.", "Direction distribution independent of magnitude.", "mixed"),
        (1149, "Zusaetzliche VCG-Idee", "st_t_continuity_breakpoint_score", "ST-T transition breakpoint.", "Change-point score in ST-T curve of M(t), speed or direction.", "Abrupt ST-T morphology changes.", "dimensionless"),
        (1150, "Zusaetzliche VCG-Idee", "topological_signature_distance_qrs_t", "Retired topological QRS-T candidate.", "No scalar is emitted; the historical implementation was only a Hausdorff alias.", "Historical name retained for provenance; not evaluable.", "mV"),
        (1151, "Zusaetzliche VCG-Idee", "trajectory_reversal_fraction", "Fraction of backward movement.", "Fraction with dot(v_t, mean_direction) < 0 in segment.", "Retrograde activation or loop folding.", "fraction"),
        (1152, "Zusaetzliche VCG-Idee", "vcg_phase_space_determinism", "Recurrence determinism.", "Fraction of fixed-scale recurrent points in diagonal runs of at least three samples, excluding a Theiler window.", "Repeated local trajectory structure.", "dimensionless"),
        (1153, "Zusaetzliche VCG-Idee", "vcg_phase_space_recurrence_rate", "Recurrence rate in XYZ phase space.", "Fraction of nonlocal point pairs within 10% of QRS-loop diameter; adjacent samples are excluded.", "Fixed-scale trajectory recurrence.", "fraction"),
        (1154, "Zusaetzliche VCG-Idee", "vcg_vector_field_curl_proxy", "Curl/vortex proxy.", "Local rotation from cross products of neighboring movement vectors.", "Rotation dominance in the loop.", "mixed"),
        (1155, "Zusaetzliche VCG-Idee", "vcg_vector_field_divergence_proxy", "Trajectory divergence proxy.", "Change in local point density over time; expanding versus contracting.", "Loop expansion and return dynamics.", "mixed"),
    ])

    return [_definition(*row) for row in rows]


def candidate_definitions(limit: Optional[int] = 1155) -> list[BiomarkerDefinition]:
    definitions = (
        _manual_definitions()
        + _vector_integral_definitions()
        + _kinematic_definitions()
        + _kinematic_landmark_definitions()
        + _classic_ecg_definitions()
        + _literature_extension_definitions()
        + _loop_geometry_definitions()
        + _loop_projection_definitions()
        + _extra_tail_definitions()
    )
    definitions.sort(key=lambda item: item.source_index or 0)
    if limit is not None:
        definitions = [definition for definition in definitions if (definition.source_index or 0) <= limit]
    return definitions


def register_candidate_biomarkers(limit: Optional[int] = 1155) -> None:
    for definition in candidate_definitions(limit=limit):
        BiomarkerRegistry.register_definition(definition)


__all__ = ["candidate_definitions", "register_candidate_biomarkers", "DATA_DRIVEN_CUTOFF_NOTE"]
