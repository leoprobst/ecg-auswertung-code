"""A thick wall with a quiet ECG -- the one question no single marker can ask.

Every marker in this catalog sees the ECG alone, so none of them can express a
*discordance* between the ECG and the echo. That is the classical sign of
cardiac amyloidosis, and building it required recovering a quantity MIMIC never
reports: LV mass is null in all 206.488 echo studies, while the three linear
measurements the ASE formula needs are present in 168.364.

Two things found while building this, both pinned below:

* the ``lv_mass_index`` branch of ``echo_lvh_strict`` can never fire, and the
  echo entries containing "mass" are valve vegetations, not myocardium;
* the ratio has to be judged inside the hypertrophied. Across all pairs it
  loses to LV mass alone, because mass by itself separates the amyloidosis
  codes and a nearly mass-independent voltage divided by mass mostly rebuilds
  1/mass. Among thick walls the mass stops varying and the ratio beats its own
  numerator every time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.voltage_mass import (
    AMYLOID_COMPETING_ENDPOINT,
    AMYLOID_DOCUMENTED_ENDPOINT,
    DIAMETER_BOUNDS_CM,
    HYPERTROPHY_THRESHOLD_CM,
    VOLTAGE_MARKERS,
    VoltageMassPolicy,
    WALL_BOUNDS_CM,
    devereux_lv_mass_g,
    documented_amyloid_endpoints,
    discordance_association,
    relative_wall_thickness,
    select_voltage_mass_index_pairs,
    voltage_mass_discordance,
    voltage_mass_pair_query,
)
from vcg_biomarker.mimic.mappings import ICD_GROUPS


# --- The formula -------------------------------------------------------------


def test_a_normal_ventricle_lands_in_the_normal_mass_range():
    """IVS 1,0 / PW 1,0 / LVIDd 4,8 cm is a textbook normal heart."""

    mass = float(devereux_lv_mass_g(1.0, 1.0, 4.8).iloc[0])
    assert 150 < mass < 190, f"{mass:.1f} g liegt ausserhalb des Normbereichs"


def test_a_thickened_ventricle_weighs_more_than_a_normal_one():
    normal = float(devereux_lv_mass_g(1.0, 1.0, 4.8).iloc[0])
    thick = float(devereux_lv_mass_g(1.6, 1.6, 4.2).iloc[0])
    assert thick > normal


def test_millimetres_are_refused_rather_than_cubed():
    """The formula cubes its argument: 16 mm read as 16 cm is a 10-kg heart."""

    assert pd.isna(devereux_lv_mass_g(16, 16, 42).iloc[0])


@pytest.mark.parametrize(
    "ivs, pwt, lvidd",
    [
        (0.1, 1.0, 4.8),   # septum below the floor
        (1.0, 9.0, 4.8),   # posterior wall above the ceiling
        (1.0, 1.0, 0.5),   # cavity too small to be a ventricle
        (1.0, 1.0, 15.0),  # cavity too large
    ],
)
def test_implausible_geometry_yields_no_mass(ivs, pwt, lvidd):
    assert pd.isna(devereux_lv_mass_g(ivs, pwt, lvidd).iloc[0])


def test_the_bounds_are_wide_enough_for_real_disease():
    """They must reject impossible values, not unusual ones."""

    # Severe hypertrophic cardiomyopathy: a 2,8 cm septum is real.
    assert not pd.isna(devereux_lv_mass_g(2.8, 1.4, 3.8).iloc[0])
    # A severely dilated ventricle at 7,5 cm is real.
    assert not pd.isna(devereux_lv_mass_g(0.9, 0.9, 7.5).iloc[0])


def test_mass_is_computed_elementwise_over_a_column():
    mass = devereux_lv_mass_g(
        pd.Series([1.0, 1.6, 16.0]),
        pd.Series([1.0, 1.6, 16.0]),
        pd.Series([4.8, 4.2, 42.0]),
    )
    assert len(mass) == 3
    assert mass.notna().tolist() == [True, True, False]


# --- Geometry ----------------------------------------------------------------


def test_relative_wall_thickness_separates_concentric_from_eccentric():
    """0,42 is the accepted cut; amyloid raises RWT, dilatation lowers it."""

    concentric = float(relative_wall_thickness(1.6, 4.2).iloc[0])
    eccentric = float(relative_wall_thickness(0.9, 6.5).iloc[0])
    assert concentric > 0.42 > eccentric


def test_a_zero_diameter_gives_no_thickness_ratio():
    assert pd.isna(relative_wall_thickness(1.0, 0.0).iloc[0])


# --- The ratio ---------------------------------------------------------------


def _pairs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "study_id": np.arange(n),
            "split": "train",
            "ivs_cm": rng.uniform(0.8, 1.8, n),
            "pwt_cm": rng.uniform(0.8, 1.8, n),
            "lvidd_cm": rng.uniform(3.5, 5.5, n),
        }
    )
    frame["lv_mass_g"] = devereux_lv_mass_g(
        frame["ivs_cm"], frame["pwt_cm"], frame["lvidd_cm"]
    ).to_numpy()
    for marker in VOLTAGE_MARKERS:
        frame[marker] = rng.uniform(1.0, 3.0, n)
    return frame


def test_every_voltage_marker_gets_a_ratio_and_a_percentile():
    scored = voltage_mass_discordance(_pairs())
    for marker in VOLTAGE_MARKERS:
        assert f"{marker}_per_gram" in scored
        assert f"{marker}_discordance_pct" in scored


def test_the_percentile_spans_the_whole_range():
    """It is a rank in the cohort, so it has to use the cohort."""

    scored = voltage_mass_discordance(_pairs())
    pct = scored[f"{VOLTAGE_MARKERS[0]}_discordance_pct"]
    assert pct.min() < 5 and pct.max() > 95


def test_the_lowest_percentile_is_the_quietest_heart_for_its_size():
    scored = voltage_mass_discordance(_pairs())
    marker = VOLTAGE_MARKERS[0]
    lowest = scored.loc[scored[f"{marker}_discordance_pct"].idxmin()]
    highest = scored.loc[scored[f"{marker}_discordance_pct"].idxmax()]
    assert lowest[f"{marker}_per_gram"] < highest[f"{marker}_per_gram"]


def test_a_zero_mass_never_becomes_an_infinite_ratio():
    frame = _pairs(n=50)
    frame.loc[0, "lv_mass_g"] = 0.0
    scored = voltage_mass_discordance(frame)
    ratio = scored[f"{VOLTAGE_MARKERS[0]}_per_gram"]
    assert pd.isna(ratio.iloc[0])
    assert np.isfinite(ratio.dropna()).all()


def test_missing_voltage_markers_are_skipped_not_fabricated():
    frame = _pairs().drop(columns=[VOLTAGE_MARKERS[0]])
    scored = voltage_mass_discordance(frame)
    assert f"{VOLTAGE_MARKERS[0]}_per_gram" not in scored
    assert f"{VOLTAGE_MARKERS[1]}_per_gram" in scored


def test_an_empty_frame_survives_scoring():
    assert len(voltage_mass_discordance(pd.DataFrame())) == 0


def test_repeated_ecgs_are_reduced_to_the_closest_outcome_blind_echo_pair():
    frame = _pairs(n=4)
    repeated = pd.concat([frame, frame.iloc[[0]].copy()], ignore_index=True)
    repeated.loc[0, ["echo_delta_hours", "ecg_time", "study_id"]] = [
        12.0,
        pd.Timestamp("2026-01-01"),
        100,
    ]
    repeated.loc[4, ["echo_delta_hours", "ecg_time", "study_id"]] = [
        -1.0,
        pd.Timestamp("2026-02-01"),
        101,
    ]

    selected = select_voltage_mass_index_pairs(repeated)

    assert selected["subject_id"].is_unique
    chosen = selected.loc[selected["subject_id"].eq(0)].iloc[0]
    assert int(chosen["study_id"]) == 101
    assert len(voltage_mass_discordance(repeated)) == frame["subject_id"].nunique()


# --- The population the question is asked in ---------------------------------


def test_hypertrophy_is_flagged_from_either_wall():
    frame = _pairs(n=6)
    frame.loc[:, ["ivs_cm", "pwt_cm"]] = 1.0
    frame.loc[0, "ivs_cm"] = 1.5   # septum alone
    frame.loc[1, "pwt_cm"] = 1.5   # posterior wall alone
    scored = voltage_mass_discordance(frame)
    assert bool(scored.loc[0, "hypertrophied"])
    assert bool(scored.loc[1, "hypertrophied"])
    assert not bool(scored.loc[2, "hypertrophied"])


def test_the_threshold_matches_the_existing_wall_thickening_endpoint():
    """Two definitions of a thick wall in one project would be one too many."""

    assert HYPERTROPHY_THRESHOLD_CM == 1.2


def test_the_association_restricts_to_the_hypertrophied_by_default():
    frame = _pairs(n=600, seed=1)
    frame["amyloid"] = 0
    frame.loc[frame.index[:60], "amyloid"] = 1
    scored = voltage_mass_discordance(frame)
    restricted = discordance_association(scored, "amyloid")
    unrestricted = discordance_association(
        scored, "amyloid", restrict_to_hypertrophy=False
    )
    assert set(restricted["population"]) == {"hypertrophied"}
    assert set(unrestricted["population"]) == {"all_pairs"}
    assert restricted["n_pairs"].max() < unrestricted["n_pairs"].max()


def test_restricting_without_the_geometry_is_an_error_not_a_silent_pass():
    """Quietly analysing everyone would answer a different question."""

    frame = voltage_mass_discordance(_pairs()).drop(
        columns=["ivs_cm", "pwt_cm", "hypertrophied"]
    )
    frame["amyloid"] = 0
    frame.loc[frame.index[:40], "amyloid"] = 1
    with pytest.raises(ValueError, match="Hypertrophie"):
        discordance_association(frame, "amyloid")


# --- The comparison the ratio has to survive ---------------------------------


def test_all_three_aucs_are_reported_side_by_side():
    """A ratio beating neither part has added a division, not a finding."""

    frame = _pairs(n=800, seed=2)
    frame["amyloid"] = (frame["lv_mass_g"] > frame["lv_mass_g"].median()).astype(int)
    scored = voltage_mass_discordance(frame)
    result = discordance_association(scored, "amyloid")
    for column in ("auc_ratio", "auc_voltage", "auc_mass", "gain_over_voltage"):
        assert column in result
    assert result.evidence_stage.eq("exploratory").all()
    assert result.adjustment_status.eq("not_computed").all()
    assert result.joint_components_comparison_status.eq("not_computed").all()
    assert not result.clinical_claim_allowed.any()


def test_a_ratio_that_only_reproduces_the_denominator_shows_no_gain():
    """Constant voltage makes the ratio exactly 1/mass -- and nothing new."""

    frame = _pairs(n=800, seed=3)
    for marker in VOLTAGE_MARKERS:
        frame[marker] = 2.0
    frame["amyloid"] = (frame["lv_mass_g"] > frame["lv_mass_g"].median()).astype(int)
    result = discordance_association(voltage_mass_discordance(frame), "amyloid")
    assert (result["gain_over_best_part"] <= 1e-9).all()


def test_a_true_discordance_is_detected():
    """Voltage low *for the mass* marks the cases; the ratio must find it.

    Getting this scenario right took two attempts, and both failures were the
    test being wrong rather than the code. The mass has to span a range far
    wider than the effect on the voltage, or the raw voltage separates the
    groups on its own and the ratio has nothing left to add: at a 1,6-fold
    mass range the raw voltage scored AUC 1,00 by itself, at 3,5-fold 0,89, at
    13,3-fold 0,79. Only in the last does the construction actually test a
    discordance.

    So: walls 1,2-3,0 cm and a cavity of 3,0-7,0 cm, giving 114-1517 g. Every
    heart makes voltage in proportion to its muscle; the affected fifth makes
    half as much for the same muscle. Mass alone then says nothing (both groups
    are drawn from one distribution), voltage alone says something but not
    much, and the ratio is decisive.
    """

    rng = np.random.default_rng(4)
    n = 900
    walls = rng.uniform(1.2, 3.0, n)
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "study_id": np.arange(n),
            "split": "train",
            "ivs_cm": walls,
            "pwt_cm": walls,
            "lvidd_cm": rng.uniform(3.0, 7.0, n),
        }
    )
    frame["lv_mass_g"] = devereux_lv_mass_g(
        frame["ivs_cm"], frame["pwt_cm"], frame["lvidd_cm"]
    ).to_numpy()
    marker = VOLTAGE_MARKERS[0]
    affected = np.zeros(n, dtype=bool)
    affected[: n // 5] = True
    frame[marker] = frame["lv_mass_g"] * np.where(affected, 1.0, 2.0) / 100.0
    frame[marker] += rng.normal(0, 0.02, n)
    frame["amyloid"] = affected.astype(int)

    result = discordance_association(
        voltage_mass_discordance(frame), "amyloid", voltage_markers=[marker]
    )
    row = result.iloc[0]
    assert row["auc_ratio"] > 0.99, "die Diskordanz muss den Fall finden"
    assert row["auc_mass"] < 0.60, (
        "die Masse allein darf hier nichts sehen -- beide Gruppen stammen aus "
        "derselben Verteilung"
    )
    assert row["auc_voltage"] < 0.85, (
        "sieht die rohe Spannung den Fall schon, prueft der Test nicht die "
        "Diskordanz sondern die Spannung"
    )
    assert row["gain_over_best_part"] > 0.15, "und deutlich besser als beide Teile"


def test_too_few_positives_produce_no_row_rather_than_a_number():
    frame = _pairs(n=600, seed=5)
    frame["amyloid"] = 0
    frame.loc[frame.index[:3], "amyloid"] = 1
    result = discordance_association(voltage_mass_discordance(frame), "amyloid")
    assert result.empty


def test_a_missing_endpoint_yields_an_empty_table():
    scored = voltage_mass_discordance(_pairs())
    assert discordance_association(scored, "nicht_vorhanden").empty


# --- The query ---------------------------------------------------------------


def test_the_query_reuses_the_existing_echo_pairing():
    """A second join rule would make disagreements uninterpretable."""

    sql = voltage_mass_pair_query(
        "master.parquet", "matches.parquet", "echo.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "nearest_tte_measurement_id" in sql
    assert "matches.parquet" in sql


def test_the_query_selects_one_outcome_blind_pair_per_patient():
    sql = voltage_mass_pair_query(
        "master.parquet", "matches.parquet", "echo.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "PARTITION BY g.subject_id" in sql
    assert "ORDER BY ABS(g.echo_delta_hours), g.ecg_time, g.study_id" in sql
    assert "QUALIFY patient_index_rank = 1" in sql


def test_documented_amyloid_is_compared_with_specific_thick_wall_diagnoses():
    diagnoses = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 4, 5],
            "code_group": [
                "systemic_amyloidosis",
                "hypertrophic_cardiomyopathy",
                "hypertensive_heart_disease",
                "systemic_amyloidosis",
                "hypertrophic_cardiomyopathy",
                "heart_failure",
            ],
        }
    )

    endpoints = documented_amyloid_endpoints(diagnoses).set_index("subject_id")

    assert endpoints.loc[1, AMYLOID_DOCUMENTED_ENDPOINT] == 1
    assert endpoints.loc[5, AMYLOID_DOCUMENTED_ENDPOINT] == 0
    assert endpoints.loc[1, AMYLOID_COMPETING_ENDPOINT] == 1
    assert endpoints.loc[2, AMYLOID_COMPETING_ENDPOINT] == 0
    assert endpoints.loc[3, AMYLOID_COMPETING_ENDPOINT] == 0
    assert pd.isna(endpoints.loc[4, AMYLOID_COMPETING_ENDPOINT])
    assert pd.isna(endpoints.loc[5, AMYLOID_COMPETING_ENDPOINT])


def test_amyloid_and_hypertensive_heart_disease_icd_groups_are_specific():
    groups = {group.group_id: group for group in ICD_GROUPS}
    assert groups["systemic_amyloidosis"].matches("E85.82", 10)
    assert groups["systemic_amyloidosis"].matches("277.3", 9)
    assert not groups["systemic_amyloidosis"].matches("I68.0", 10)
    assert groups["hypertensive_heart_disease"].matches("I11.0", 10)
    assert groups["hypertensive_heart_disease"].matches("402.91", 9)
    assert not groups["hypertensive_heart_disease"].matches("I10", 10)


def test_the_query_filters_implausible_geometry_before_cubing():
    sql = voltage_mass_pair_query(
        "master.parquet", "matches.parquet", "echo.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    for bound in (*WALL_BOUNDS_CM, *DIAMETER_BOUNDS_CM):
        assert str(bound) in sql


def test_the_query_only_reads_successfully_extracted_ecgs():
    sql = voltage_mass_pair_query(
        "master.parquet", "matches.parquet", "echo.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "extraction_status = 'ok'" in sql


def test_the_window_keeps_echo_and_ecg_on_the_same_heart():
    policy = VoltageMassPolicy()
    assert policy.window_hours == 24 * 30
    sql = voltage_mass_pair_query(
        "master.parquet", "matches.parquet", "echo.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
        policy=policy,
    )
    assert "ABS(x.nearest_tte_delta_hours) <= 720" in sql


def test_the_policy_records_the_formula_it_used():
    payload = VoltageMassPolicy().to_dict()
    assert payload["formula"] == "devereux_ase_linear"
    for key in ("window_hours", "hypertrophy_threshold_cm", "version"):
        assert key in payload
