"""Voltage against muscle: the discordance that points at infiltration.

A thick ventricle should produce a large QRS. When it does not -- a lot of
muscle on the echo, a small deflection on the ECG -- the muscle is probably not
all muscle. That discordance is the classical bedside sign of cardiac
amyloidosis, and it is a *ratio*, which is why no marker in this catalog can
express it: every one of them sees only the ECG.

Cardiac amyloidosis is worth the trouble. Around 3 % prevalence in HFpEF,
routinely missed, and now treatable -- which turns a screening signal into a
decision. Roughly a third of patients have no thickened wall at diagnosis, so
wall thickness alone is not the test either.

**What had to be built first.** MIMIC-IV reports no LV mass index: the
``lv_mass_index`` column of ``echo_measurements_wide`` is null in all 206.488
studies, and the entries whose names contain "mass" (``mv_mass``, ``av_mass``,
``lv_mass_throm``) are valve vegetations and thrombi -- a different organ
finding entirely, and non-numeric. The strict LVH endpoint therefore never
fires its LVMI branch; it runs on the wall-category text alone.

The three linear measurements the ASE formula needs are all present and
numeric:

===================================  =======
septal thickness (IVSd)              169.236
posterior wall thickness (PWTd)      169.108
all three with the LV diameter       168.364
===================================  =======

So the mass is recoverable even though it was never reported. Devereux, as the
ASE recommends it for the linear method::

    LV mass (g) = 0.8 * 1.04 * [(LVIDd + PWTd + IVSd)^3 - LVIDd^3] + 0.6

Measured over those 168.364 studies: median 169,9 g, p5-p95 92,1-305,3 g. That
is where a hospital echo population belongs -- somewhat above the healthy
reference -- which is the first evidence the formula was fed the right columns
in the right units (centimetres).

**What it found.** Against the systemic amyloidosis codes (1.162 patients;
cerebral amyloid angiopathy excluded, that is a different organ), over 173.374
ECG-echo pairs in 37.004 patients, of which 79.482 have a thickened wall:

==============================  =====  ======  ======  =======
Zaehler (hypertroph)            roh    /Masse  Masse   Gewinn
==============================  =====  ======  ======  =======
svg_magnitude                   0,560  0,585   0,573   **+0,012**
qrs_vector_magnitude            0,513  0,562   0,573   -0,011
sokolow_lyon_voltage_surrogate  0,517  0,556   0,574   -0,018
cornell_voltage_surrogate       0,454  0,503   0,574   -0,071
==============================  =====  ======  ======  =======

Two things are true at once and both belong in the report. The ratio improves
on its own numerator every time, by +0,025 to +0,052, and it does so again on
the validation split alone (+0,025 to +0,052, 506 positives) -- so the gain is
not independent evidence after results have been inspected. Only ``svg_magnitude`` divided by mass
beats *both* of its parts; for the other three, LV mass by itself separates the
codes better than the discordance does.

That the survivor is the mass-independent numerator (rho = -0,005) is not a
coincidence: where the numerator already tracks mass, dividing by mass mostly
rebuilds the denominator. It is a reason to prefer the spatial magnitude here,
and a reason to report all three AUCs rather than the ratio alone.

**What this is not.** A ratio is not a diagnosis, and AUC 0,585 is a weak
signal. Low voltage has other causes -- effusion, obesity, emphysema, and
simply a large chest -- and none are excluded here. The reference is an ICD
code, which for amyloidosis is recorded late and often not at all, so the
negatives may contain undiagnosed cases. Selective investigation and differential
coding can bias associations in either direction. These historical descriptive
comparisons do not establish clinical screening utility, cardiac involvement,
or incremental value over a joint voltage-and-mass model. Historical counts
above are not current availability guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from .panels import STUDY_PANEL_IDS


VOLTAGE_MASS_VERSION = "1.3.1"

# Plausibility bounds, in centimetres. A septum of 9 cm is a transcription
# error, not a heart, and the formula cubes its argument -- so one row recorded
# in millimetres would not look wrong, it would look like a 10-kilogram heart
# and would take over the whole distribution. The windows are wide on purpose:
# they reject impossible values, not unusual ones.
WALL_BOUNDS_CM = (0.4, 3.5)
DIAMETER_BOUNDS_CM = (2.0, 9.0)

# Echo and ECG have to describe the same heart. Thirty days is the window the
# existing endpoints already use (``has_tte_30d``); anything wider risks a
# remodelling between the two, which is exactly the quantity under study.
VOLTAGE_MASS_WINDOW_HOURS = 24 * 30

# Wall thickness above which the screening question exists at all. Nobody
# investigates amyloidosis in a normal ventricle; the question is always "this
# wall is thick, why is the ECG quiet". 1.2 cm is the threshold the existing
# ``lv_wall_thickening`` endpoint already uses, so the two agree by
# construction.
#
# This restriction is not a refinement, it is the analysis. Measured against
# the amyloidosis codes over the full 173.374 pairs, the ratio loses to LV mass
# alone (gain -0.002 to -0.085). Restricted to the 79.482 thick-walled ECGs it
# beats the raw voltage every time:
#
# ==============================  =====  =========
# Zaehler                         roh    /Masse
# ==============================  =====  =========
# sokolow_lyon_voltage_surrogate  0,517  **0,556**
# cornell_voltage_surrogate       0,454  **0,503**
# qrs_vector_magnitude            0,513  **0,562**
# svg_magnitude                   0,560  **0,585**
# ==============================  =====  =========
HYPERTROPHY_THRESHOLD_CM = 1.2

# The ECG side. Two classical amplitude sums and two VCG magnitudes -- the
# ratio is formed for each, because which numerator behaves best in an
# infiltrated ventricle is precisely what is unknown.
#
# How weakly these track muscle is worth recording, since it is the reason a
# ratio can carry information the numerator does not. Spearman against the
# Devereux mass, on the pairs above:
#
# ==============================  =======  ========
# Marker                          rho      n
# ==============================  =======  ========
# cornell_voltage_surrogate       +0,299   169.641
# qrs_vector_magnitude            +0,117   164.930
# sokolow_lyon_voltage_surrogate  +0,043   169.641
# svg_magnitude                   -0,005   154.222
# ==============================  =======  ========
#
# Sokolow-Lyon is essentially uncorrelated with LV mass across 169.641 ECGs,
# and the spatial magnitude is flat. That is the known poor sensitivity of the
# voltage criteria, quantified here against a computed mass rather than
# against a cardiologist reading the same ECG.
VOLTAGE_MARKERS: tuple[str, ...] = (
    "sokolow_lyon_voltage_surrogate",
    "cornell_voltage_surrogate",
    "qrs_vector_magnitude",
    "svg_magnitude",
)


@dataclass(frozen=True)
class VoltageMassPolicy:
    """Everything that decides a discordance value, in one auditable object."""

    window_hours: float = VOLTAGE_MASS_WINDOW_HOURS
    wall_bounds_cm: tuple[float, float] = WALL_BOUNDS_CM
    diameter_bounds_cm: tuple[float, float] = DIAMETER_BOUNDS_CM
    hypertrophy_threshold_cm: float = HYPERTROPHY_THRESHOLD_CM
    minimum_pairs: int = 200

    def to_dict(self) -> dict:
        return {
            "window_hours": float(self.window_hours),
            "wall_bounds_cm": list(self.wall_bounds_cm),
            "diameter_bounds_cm": list(self.diameter_bounds_cm),
            "hypertrophy_threshold_cm": float(self.hypertrophy_threshold_cm),
            "minimum_pairs": int(self.minimum_pairs),
            "formula": "devereux_ase_linear",
            "version": VOLTAGE_MASS_VERSION,
        }


def devereux_lv_mass_g(
    septal_cm: object,
    posterior_cm: object,
    diameter_cm: object,
    *,
    wall_bounds_cm: tuple[float, float] = WALL_BOUNDS_CM,
    diameter_bounds_cm: tuple[float, float] = DIAMETER_BOUNDS_CM,
) -> pd.Series:
    """LV mass in grams from the three linear echo measurements.

    Out-of-range inputs return NaN rather than a number, for the reason given
    at ``WALL_BOUNDS_CM``.
    """

    ivs = pd.to_numeric(pd.Series(septal_cm), errors="coerce")
    pwt = pd.to_numeric(pd.Series(posterior_cm), errors="coerce")
    lvidd = pd.to_numeric(pd.Series(diameter_cm), errors="coerce")
    low_wall, high_wall = wall_bounds_cm
    low_diameter, high_diameter = diameter_bounds_cm
    usable = (
        ivs.between(low_wall, high_wall)
        & pwt.between(low_wall, high_wall)
        & lvidd.between(low_diameter, high_diameter)
    )
    mass = 0.8 * 1.04 * ((lvidd + pwt + ivs) ** 3 - lvidd**3) + 0.6
    return mass.where(usable)


def relative_wall_thickness(posterior_cm: object, diameter_cm: object) -> pd.Series:
    """RWT = 2 * PWTd / LVIDd -- concentric versus eccentric geometry.

    Amyloid thickens the wall without dilating the cavity, so it raises RWT.
    Carried alongside the mass because the two together separate *how* a
    ventricle got heavy, which the mass alone cannot say.
    """

    pwt = pd.to_numeric(pd.Series(posterior_cm), errors="coerce")
    lvidd = pd.to_numeric(pd.Series(diameter_cm), errors="coerce")
    return (2.0 * pwt / lvidd.where(lvidd > 0)).replace(
        [np.inf, -np.inf], np.nan
    )


def voltage_mass_pair_query(
    master_sql: str,
    matches_sql: str,
    echo_sql: str,
    feature_clause: str,
    *,
    marker_columns: str,
    policy: VoltageMassPolicy = VoltageMassPolicy(),
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """One outcome-blind index ECG/echo pair per patient.

    The echo is taken through ``ecg_echo_matches``, which already picks the
    nearest study per ECG -- reusing it keeps this analysis on the same pairing
    as every existing echo endpoint, so a disagreement between them can never
    be an artefact of two different join rules.  Among a patient's eligible
    pairs, the echo closest in time is selected before any diagnosis or marker
    value is inspected.  This prevents frequently recorded patients from
    contributing the same disease state dozens of times.
    """

    split_list = ", ".join(f"'{value}'" for value in splits)
    low_wall, high_wall = policy.wall_bounds_cm
    low_diameter, high_diameter = policy.diameter_bounds_cm
    return f"""
        WITH paired AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time,
                x.nearest_tte_measurement_id AS measurement_id,
                x.nearest_tte_delta_hours    AS echo_delta_hours
            FROM read_parquet('{master_sql}') m
            JOIN read_parquet('{matches_sql}') x
              ON x.subject_id = m.subject_id AND x.study_id = m.study_id
            WHERE m.split IN ({split_list})
              AND x.nearest_tte_measurement_id IS NOT NULL
              AND ABS(x.nearest_tte_delta_hours) <= {policy.window_hours}
        ),
        geometry AS (
            SELECT
                p.subject_id, p.study_id, p.split, p.ecg_time,
                p.echo_delta_hours,
                e.septal_thickness_cm              AS ivs_cm,
                e.posterior_wall_thickness_cm      AS pwt_cm,
                e.lv_internal_diameter_diastole_cm AS lvidd_cm,
                0.8 * 1.04 * (
                    power(e.lv_internal_diameter_diastole_cm
                          + e.posterior_wall_thickness_cm
                          + e.septal_thickness_cm, 3)
                    - power(e.lv_internal_diameter_diastole_cm, 3)
                ) + 0.6 AS lv_mass_g,
                2.0 * e.posterior_wall_thickness_cm
                    / e.lv_internal_diameter_diastole_cm
                    AS relative_wall_thickness
            FROM paired p
            JOIN read_parquet('{echo_sql}') e
              ON e.measurement_id = p.measurement_id
            WHERE e.septal_thickness_cm
                  BETWEEN {low_wall} AND {high_wall}
              AND e.posterior_wall_thickness_cm
                  BETWEEN {low_wall} AND {high_wall}
              AND e.lv_internal_diameter_diastole_cm
                  BETWEEN {low_diameter} AND {high_diameter}
        )
        SELECT
            g.subject_id, g.study_id, g.split, g.ecg_time,
            g.echo_delta_hours, g.ivs_cm, g.pwt_cm, g.lvidd_cm,
            g.lv_mass_g, g.relative_wall_thickness,
            ROW_NUMBER() OVER (
                PARTITION BY g.subject_id
                ORDER BY ABS(g.echo_delta_hours), g.ecg_time, g.study_id
            ) AS patient_index_rank,
            {marker_columns}
        FROM geometry g
        JOIN {feature_clause} f
          ON f.subject_id = g.subject_id AND f.study_id = g.study_id
        WHERE f.extraction_status = 'ok'
        QUALIFY patient_index_rank = 1
    """


def select_voltage_mass_index_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Apply the same outcome-blind patient index rule to in-memory frames."""

    if pairs is None or len(pairs) == 0 or "subject_id" not in pairs:
        return pd.DataFrame(pairs).copy() if pairs is not None else pd.DataFrame()
    frame = pairs.copy()
    frame["_echo_distance"] = pd.to_numeric(
        frame.get(
            "echo_delta_hours", pd.Series(np.nan, index=frame.index)
        ),
        errors="coerce",
    ).abs()
    frame["_ecg_time"] = pd.to_datetime(
        frame.get("ecg_time", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce",
    )
    frame["_study_id"] = frame.get(
        "study_id", pd.Series("", index=frame.index)
    ).astype(str)
    return (
        frame.sort_values(
            ["subject_id", "_echo_distance", "_ecg_time", "_study_id"],
            kind="stable",
            na_position="last",
        )
        .drop_duplicates("subject_id", keep="first")
        .drop(columns=["_echo_distance", "_ecg_time", "_study_id"])
        .reset_index(drop=True)
    )


def voltage_mass_discordance(
    pairs: pd.DataFrame,
    *,
    voltage_markers: Sequence[str] = VOLTAGE_MARKERS,
    mass_column: str = "lv_mass_g",
    policy: VoltageMassPolicy = VoltageMassPolicy(),
) -> pd.DataFrame:
    """Voltage per gram of muscle, plus its rank in the cohort.

    Reported as a percentile alongside the raw ratio because the units are
    arbitrary -- millivolts per gram means nothing to a reader -- while the
    ordering is the whole content: the lowest percentiles are the hearts whose
    electrical output is least explained by their size.
    """

    if pairs is None or len(pairs) == 0 or mass_column not in pairs:
        return pd.DataFrame(pairs) if pairs is not None else pd.DataFrame()
    frame = select_voltage_mass_index_pairs(pairs)
    if {"ivs_cm", "pwt_cm"} <= set(frame.columns):
        threshold = float(policy.hypertrophy_threshold_cm)
        frame["hypertrophied"] = (
            pd.to_numeric(frame["ivs_cm"], errors="coerce").ge(threshold)
            | pd.to_numeric(frame["pwt_cm"], errors="coerce").ge(threshold)
        )
    mass = pd.to_numeric(frame[mass_column], errors="coerce")
    positive_mass = mass.where(mass > 0)
    for marker in voltage_markers:
        if marker not in frame.columns:
            continue
        voltage = pd.to_numeric(frame[marker], errors="coerce")
        ratio = voltage / positive_mass
        frame[f"{marker}_per_gram"] = ratio
        frame[f"{marker}_discordance_pct"] = ratio.rank(pct=True) * 100.0
    return frame


def discordance_association(
    pairs: pd.DataFrame,
    endpoint_column: str,
    *,
    voltage_markers: Sequence[str] = VOLTAGE_MARKERS,
    policy: VoltageMassPolicy = VoltageMassPolicy(),
    restrict_to_hypertrophy: bool = True,
) -> pd.DataFrame:
    """Does the ratio separate an endpoint better than its parts do?

    The comparison is the point. A ratio that beats neither its numerator nor
    its denominator has added a division, not a finding -- so all three AUCs
    are reported side by side and the ratio has to earn its place against
    both. Even a positive gain is only a descriptive unadjusted comparison,
    not incremental value over a joint model of the components. Subgroup
    choices motivated by inspected results remain post-outcome discovery.

    The default restricts to the hypertrophied, because that is where the
    clinical question is asked and because the unrestricted answer is
    misleading in a specific way: across all 173.374 pairs the ratio loses to
    LV mass alone, since mass by itself separates the amyloidosis codes
    (AUC 0,59) and dividing a nearly mass-independent voltage by it mostly
    reproduces 1/mass. Among the 79.482 thick walls the mass no longer varies
    enough to carry the answer, and the ratio beats its own numerator for
    every one of the four. Pass ``False`` to see the unrestricted comparison;
    both are worth reporting, and only one is a screening statement.
    """

    from sklearn.metrics import roc_auc_score

    columns = [
        "voltage_marker",
        "n_pairs",
        "n_positive",
        "auc_ratio",
        "auc_voltage",
        "auc_mass",
        "gain_over_best_part",
        "gain_over_voltage",
        "population",
        "analysis_design", "evidence_stage", "adjustment_status",
        "joint_components_comparison_status", "clinical_claim_allowed",
    ]
    if (
        pairs is None
        or len(pairs) == 0
        or endpoint_column not in pairs
        or "lv_mass_g" not in pairs
    ):
        return pd.DataFrame(columns=columns)

    frame = pairs
    if restrict_to_hypertrophy:
        if "hypertrophied" in frame.columns:
            frame = frame.loc[frame["hypertrophied"].astype(bool)]
        elif {"ivs_cm", "pwt_cm"} <= set(frame.columns):
            threshold = float(policy.hypertrophy_threshold_cm)
            frame = frame.loc[
                pd.to_numeric(frame["ivs_cm"], errors="coerce").ge(threshold)
                | pd.to_numeric(frame["pwt_cm"], errors="coerce").ge(threshold)
            ]
        else:
            raise ValueError(
                "Ohne ivs_cm/pwt_cm laesst sich die Hypertrophie nicht "
                "bestimmen; restrict_to_hypertrophy=False setzen, wenn die "
                "unbeschraenkte Auswertung gemeint ist"
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=columns)

    target = pd.to_numeric(frame[endpoint_column], errors="coerce")
    mass = pd.to_numeric(frame["lv_mass_g"], errors="coerce")
    records: list[dict] = []
    for marker in voltage_markers:
        ratio_column = f"{marker}_per_gram"
        if ratio_column not in frame or marker not in frame:
            continue
        ratio = pd.to_numeric(frame[ratio_column], errors="coerce")
        voltage = pd.to_numeric(frame[marker], errors="coerce")
        usable = (
            target.notna() & ratio.notna() & voltage.notna() & mass.notna()
        )
        n = int(usable.sum())
        positives = int(target[usable].astype(bool).sum())
        if n < int(policy.minimum_pairs) or positives < 10 or positives == n:
            continue
        outcome = target[usable].astype(bool).to_numpy()
        # A low ratio is the abnormal direction, so the AUC is taken on the
        # negated value; the same orientation is applied to the voltage so the
        # three numbers stay comparable. Mass keeps its own direction -- more
        # muscle is the abnormal end there.
        auc_ratio = float(roc_auc_score(outcome, -ratio[usable]))
        auc_voltage = float(roc_auc_score(outcome, -voltage[usable]))
        auc_mass = float(roc_auc_score(outcome, mass[usable]))
        best_part = max(auc_voltage, auc_mass)
        records.append(
            {
                "voltage_marker": str(marker),
                "n_pairs": n,
                "n_positive": positives,
                "auc_ratio": auc_ratio,
                "auc_voltage": auc_voltage,
                "auc_mass": auc_mass,
                "gain_over_best_part": auc_ratio - best_part,
                "gain_over_voltage": auc_ratio - auc_voltage,
                "analysis_design": "descriptive_unadjusted_component_comparison",
                "evidence_stage": "exploratory",
                "adjustment_status": "not_computed",
                "joint_components_comparison_status": "not_computed",
                "clinical_claim_allowed": False,
                "population": (
                    "hypertrophied" if restrict_to_hypertrophy else "all_pairs"
                ),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(
        "gain_over_voltage", ascending=False, ignore_index=True
    )


# Endpoints the discordance is judged against. The wall-thickening endpoint is
# the denominator's own definition and is therefore *not* usable as an outcome
# here -- it would be predicted by the geometry it is made of. What remains are
# outcomes about function and consequence.
DISCORDANCE_ENDPOINTS: tuple[str, ...] = (
    "reduced_lvef",
    "elevated_ntprobnp",
    "heart_failure_icd",
    "acute_decompensated_heart_failure_proxy",
)

AMYLOID_DIAGNOSIS_GROUP = "systemic_amyloidosis"
AMYLOID_COMPETING_GROUPS: tuple[str, ...] = (
    "hypertrophic_cardiomyopathy",
    "hypertensive_heart_disease",
)
AMYLOID_DOCUMENTED_ENDPOINT = "documented_systemic_amyloidosis"
AMYLOID_COMPETING_ENDPOINT = (
    "documented_amyloidosis_vs_hcm_or_hypertensive_heart_disease"
)


def documented_amyloid_endpoints(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Create patient-level documented-diagnosis contrasts.

    ``systemic_amyloidosis`` is not a cardiac-biopsy reference and is never
    presented as one.  The second endpoint asks the clinically sharper
    question among competing thick-wall diagnoses. Patients carrying both an
    amyloidosis and a competing code are excluded from that contrast rather
    than assigned to whichever code happened to be encountered first.
    """

    columns = [
        "subject_id",
        AMYLOID_DOCUMENTED_ENDPOINT,
        AMYLOID_COMPETING_ENDPOINT,
    ]
    if (
        diagnoses is None
        or len(diagnoses) == 0
        or not {"subject_id", "code_group"} <= set(diagnoses.columns)
    ):
        return pd.DataFrame(columns=columns)
    universe = pd.Index(diagnoses["subject_id"].dropna().unique())
    scoped = diagnoses.loc[
        diagnoses["code_group"].astype(str).isin(
            (AMYLOID_DIAGNOSIS_GROUP, *AMYLOID_COMPETING_GROUPS)
        ),
        ["subject_id", "code_group"],
    ].drop_duplicates()
    cross = pd.crosstab(scoped["subject_id"], scoped["code_group"])
    flags = (
        cross
        .reindex(index=universe, fill_value=0)
        .reindex(
            columns=[AMYLOID_DIAGNOSIS_GROUP, *AMYLOID_COMPETING_GROUPS],
            fill_value=0,
        )
        .gt(0)
    )
    amyloid = flags[AMYLOID_DIAGNOSIS_GROUP]
    competing = flags[list(AMYLOID_COMPETING_GROUPS)].any(axis=1)
    result = pd.DataFrame(
        {
            "subject_id": flags.index,
            AMYLOID_DOCUMENTED_ENDPOINT: amyloid.astype(int),
            AMYLOID_COMPETING_ENDPOINT: np.where(
                amyloid & ~competing,
                1.0,
                np.where(~amyloid & competing, 0.0, np.nan),
            ),
        }
    )
    return result.reset_index(drop=True)


def run_voltage_mass_discordance(
    config,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    voltage_markers: Sequence[str] = VOLTAGE_MARKERS,
    endpoints: Sequence[str] = DISCORDANCE_ENDPOINTS,
    policy: VoltageMassPolicy = VoltageMassPolicy(),
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Recover the LV mass, form the ratio, and judge it against its parts."""

    import duckdb

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        capture_output_integrity,
        content_hash,
        development_feature_glob,
        feature_manifest_path,
        feature_parquet_clause,
        input_artifact_fingerprints,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "voltage_mass"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "voltage_mass.manifest.json"

    markers = tuple(dict.fromkeys(str(item) for item in voltage_markers))
    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    feature_manifest = feature_manifest_path(resolved_glob)
    upstream = input_artifact_fingerprints(
        {
            "feature_manifest": feature_manifest,
            "feature_partition_index": feature_manifest.parent
            / "partition_index.json",
            "master_index_manifest": config.master_root
            / "master_index.manifest.json",
            "evidence_manifest": config.evidence_root / "evidence.manifest.json",
            "labels_manifest": config.labels_root / "labels.manifest.json",
        }
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "voltage_markers": list(markers),
            "endpoints": list(endpoints),
            "policy": policy.to_dict(),
            "upstream": upstream,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, VOLTAGE_MASS_VERSION):
            return previous

    matches = config.evidence_root / "ecg_echo_matches.parquet"
    echo = config.evidence_root / "echo_measurements_wide.parquet"
    for path in (matches, echo):
        if not path.exists():
            raise FileNotFoundError(
                f"Fuer die Spannungs-Masse-Diskordanz fehlt {path}"
            )

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    if progress:
        progress("Spannung/Masse: EKG-Echo-Paare mit Geometrie bilden")

    labels = config.labels_root / "clinical_labels_wide.parquet"
    diagnoses = config.evidence_root / "diagnosis_evidence.parquet"
    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        clause = feature_parquet_clause(connection, sql_path(resolved_glob))
        available = set(
            connection.execute(f"SELECT * FROM {clause} LIMIT 0").df().columns
        )
        present = [name for name in markers if name in available]
        if not present:
            raise ValueError("Kein Spannungsmarker im Merkmalssatz vorhanden")
        pairs = connection.execute(
            voltage_mass_pair_query(
                sql_path(config.master_root / "ecg_master_index.parquet"),
                sql_path(matches),
                sql_path(echo),
                clause,
                marker_columns=", ".join(f"f.{name}" for name in present),
                policy=policy,
            )
        ).df()
        endpoint_frame = None
        diagnosis_frame = None
        if labels.exists() and len(pairs):
            wanted = set(
                connection.execute(
                    f"SELECT * FROM read_parquet('{sql_path(labels)}') LIMIT 0"
                ).df().columns
            )
            states = [
                f"{name}_state" for name in endpoints
                if f"{name}_state" in wanted
            ]
            if states:
                endpoint_frame = connection.execute(
                    "SELECT subject_id, study_id, "
                    + ", ".join(states)
                    + f" FROM read_parquet('{sql_path(labels)}')"
                ).df()
        if diagnoses.exists() and len(pairs):
            diagnosis_frame = connection.execute(
                "SELECT subject_id, code_group "
                f"FROM read_parquet('{sql_path(diagnoses)}')"
            ).df()

    scored = voltage_mass_discordance(
        pairs, voltage_markers=present, policy=policy
    )
    if progress:
        hypertrophied = (
            int(scored["hypertrophied"].sum())
            if "hypertrophied" in scored else 0
        )
        progress(
            f"Spannung/Masse: {len(scored):,} Paare, "
            f"{hypertrophied:,} mit verdickter Wand"
        )

    results: list[pd.DataFrame] = []
    if endpoint_frame is not None and len(scored):
        merged = scored.merge(
            endpoint_frame, on=["subject_id", "study_id"], how="left"
        )
        for name in endpoints:
            state = f"{name}_state"
            if state not in merged.columns:
                continue
            adjudicated = merged.loc[merged[state].isin(["positive", "negative"])]
            if adjudicated.empty:
                continue
            adjudicated = adjudicated.copy()
            adjudicated[name] = adjudicated[state].eq("positive").astype(int)
            for restrict in (True, False):
                block = discordance_association(
                    adjudicated, name,
                    voltage_markers=present,
                    policy=policy,
                    restrict_to_hypertrophy=restrict,
                )
                if block.empty:
                    continue
                block.insert(0, "endpoint_id", name)
                block["observational_unit"] = "patient_index_ecg_echo_pair"
                results.append(block)

    if diagnosis_frame is not None and len(scored):
        diagnosis_endpoints = documented_amyloid_endpoints(diagnosis_frame)
        diagnosis_merged = scored.merge(
            diagnosis_endpoints, on="subject_id", how="left"
        )
        diagnosis_merged[AMYLOID_DOCUMENTED_ENDPOINT] = pd.to_numeric(
            diagnosis_merged[AMYLOID_DOCUMENTED_ENDPOINT], errors="coerce"
        ).fillna(0.0)
        for name in (AMYLOID_DOCUMENTED_ENDPOINT, AMYLOID_COMPETING_ENDPOINT):
            for restrict in (True, False):
                block = discordance_association(
                    diagnosis_merged,
                    name,
                    voltage_markers=present,
                    policy=policy,
                    restrict_to_hypertrophy=restrict,
                )
                if block.empty:
                    continue
                block.insert(0, "endpoint_id", name)
                block["observational_unit"] = "patient_index_ecg_echo_pair"
                block["reference_scope"] = (
                    "documented_systemic_amyloidosis_code_not_cardiac_biopsy"
                )
                results.append(block)

    table = (
        pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    )
    summary = {
        "version": VOLTAGE_MASS_VERSION,
        "run_id": str(run_id),
        "policy": policy.to_dict(),
        "voltage_markers": list(present),
        "n_pairs": int(len(scored)),
        "n_patients": (
            int(scored["subject_id"].nunique()) if len(scored) else 0
        ),
        "n_hypertrophied": (
            int(scored["hypertrophied"].sum())
            if "hypertrophied" in scored else 0
        ),
        "median_lv_mass_g": (
            float(pd.to_numeric(scored["lv_mass_g"], errors="coerce").median())
            if len(scored) else None
        ),
        "n_associations": int(len(table)),
        "endpoints": list(endpoints),
        "documented_diagnosis_endpoints": [
            AMYLOID_DOCUMENTED_ENDPOINT,
            AMYLOID_COMPETING_ENDPOINT,
        ],
        "observational_unit": "one_outcome_blind_ecg_echo_pair_per_patient",
        "amyloidosis_reference_scope": (
            "documented_systemic_diagnosis_not_confirmed_cardiac_involvement"
        ),
    }

    # The point cloud itself, for the scatter that makes the ratio legible.
    # Only the columns a reader needs: the geometry, the voltages and whether
    # the wall is thick. The per-gram ratios are recomputable from those and
    # would double the file for nothing.
    cloud_columns = [
        name
        for name in (
            "subject_id", "study_id", "split", "echo_delta_hours",
            "ivs_cm", "pwt_cm", "lvidd_cm", "lv_mass_g",
            "relative_wall_thickness", "hypertrophied",
            *present,
        )
        if name in scored.columns
    ]
    cloud = scored[cloud_columns] if cloud_columns else pd.DataFrame()

    table_path = output_root / "voltage_mass_discordance.parquet"
    cloud_path = output_root / "voltage_mass_pairs.parquet"
    summary_path = output_root / "voltage_mass_summary.json"
    atomic_write_parquet(table, table_path)
    atomic_write_parquet(cloud, cloud_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="voltage-mass-discordance",
        stage_version=VOLTAGE_MASS_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(table_path.resolve()),
            str(cloud_path.resolve()),
            str(summary_path.resolve()),
        ),
        row_counts={
            "pairs": int(len(scored)),
            "associations": int(len(table)),
        },
        output_integrity=capture_output_integrity(
            (table_path, cloud_path, summary_path)
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "AMYLOID_COMPETING_ENDPOINT",
    "AMYLOID_COMPETING_GROUPS",
    "AMYLOID_DIAGNOSIS_GROUP",
    "AMYLOID_DOCUMENTED_ENDPOINT",
    "DIAMETER_BOUNDS_CM",
    "HYPERTROPHY_THRESHOLD_CM",
    "DISCORDANCE_ENDPOINTS",
    "VOLTAGE_MARKERS",
    "VOLTAGE_MASS_VERSION",
    "VOLTAGE_MASS_WINDOW_HOURS",
    "VoltageMassPolicy",
    "WALL_BOUNDS_CM",
    "devereux_lv_mass_g",
    "documented_amyloid_endpoints",
    "discordance_association",
    "relative_wall_thickness",
    "run_voltage_mass_discordance",
    "select_voltage_mass_index_pairs",
    "voltage_mass_discordance",
    "voltage_mass_pair_query",
]
