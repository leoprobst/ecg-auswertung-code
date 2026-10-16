"""Comparing a patient with themselves removes what cannot be modelled away.

Measured on the real cohort, 604 dialysis pairs with serum potassium on both
sides (median shift -0.30 mmol/l):

=================================  =======  =========  =====
Marker                             rho      p          n
=================================  =======  =========  =====
tpeak_tend_ms                      -0.210   5e-06        463
qrs_t_peak_magnitude_ratio         -0.168   4e-04        435
classical_peaked_t_wave_heuristic  +0.149   0.0018       438
t_loop_width_major                 +0.142   0.0027       446
qrs_duration_ms                    +0.132   0.0016       569
qtc_fridericia_ms                  -0.101   0.045        396
st_vector_magnitude                -0.034   0.47         465
=================================  =======  =========  =====

Every sign is the textbook one: potassium falls, Tpeak-Tend lengthens, the
peaked T recedes, the QRS narrows, the QTc lengthens. Two markers with no
repolarisation mechanism show nothing, which is what they should show.

For comparison, the cross-sectional analysis of the same marker
``t_loop_width_major`` against severe hyperkalaemia gave an incremental AUC of
+0.0006 with the confidence interval spanning zero. The effect was never
absent; it was buried under between-person variation.

The design's whole value rests on the pairing being correct, so that is what
these tests hold: the intervention must fall *between* the two recordings, both
must belong to one patient, and a lab difference must never be built from an
implausible value.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import (
    DIALYSIS_ITEMIDS,
    DIALYSIS_LAB_ANALYTES,
    DIALYSIS_WINDOW,
    DOSE_CONTEXT_COLUMNS,
    DRUG_WINDOW,
    POTASSIUM_BOUNDS,
    SERUM_POTASSIUM_ITEMIDS,
    WITHIN_SUBJECT_MARKERS,
    InterventionWindow,
    add_within_subject_fdr,
    dose_pair_query,
    dose_pairs_all_classes_query,
    intervention_pair_query,
    nearest_lab_query,
    nearest_labs_query,
    paired_change_table,
    within_subject_association,
)


MARKERS = ("marker_a", "marker_b")


def _pairs(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "first_study_id",
            "second_study_id",
            "marker_a_first",
            "marker_a_second",
            "marker_b_first",
            "marker_b_second",
        ],
    )


def _labs(rows):
    return pd.DataFrame(rows, columns=["study_id", "value"]).assign(
        subject_id=1, itemid=50971, unit="mmol/l", lab_reference_status="eligible")


# --- What the pairing must guarantee ----------------------------------------


def test_the_intervention_must_fall_between_the_two_recordings():
    """A session that started before the first ECG puts both on one side."""

    sql = intervention_pair_query(
        "m.parquet", "p.parquet", "read_parquet('f')",
        window=DIALYSIS_WINDOW, marker_columns="a.x",
    )
    assert "p.starttime > c.first_ecg_time" in sql
    assert "p.starttime < c.second_ecg_time" in sql


def test_the_pair_stays_within_one_patient():
    sql = intervention_pair_query(
        "m.parquet", "p.parquet", "read_parquet('f')",
        window=DIALYSIS_WINDOW, marker_columns="a.x",
    )
    assert "PARTITION BY m.subject_id" in sql
    assert "p.subject_id = c.subject_id" in sql


def test_the_admission_filter_reads_the_pair_not_the_ecg():
    """``a`` and ``b`` are feature rows and carry no admission number."""

    sql = intervention_pair_query(
        "m.parquet", "p.parquet", "read_parquet('f')",
        window=DIALYSIS_WINDOW, marker_columns="a.x",
    )
    assert "t.first_hadm_id = t.second_hadm_id" in sql
    assert "a.hadm_id" not in sql


def test_the_admission_filter_can_be_switched_off():
    window = InterventionWindow(
        "x", "X", DIALYSIS_ITEMIDS, require_same_admission=False
    )
    sql = intervention_pair_query(
        "m.parquet", "p.parquet", "read_parquet('f')",
        window=window, marker_columns="a.x",
    )
    assert "first_hadm_id" not in sql.split("FROM treated")[1]


def test_the_locked_test_split_is_never_read():
    sql = intervention_pair_query(
        "m.parquet", "p.parquet", "read_parquet('f')",
        window=DIALYSIS_WINDOW, marker_columns="a.x",
    )
    assert "'train'" in sql and "'validation'" in sql
    assert "locked_test" not in sql


def test_only_dialysis_treatments_count_as_the_intervention():
    """Catheter insertion and filter changes move no electrolytes."""

    assert 224270 not in DIALYSIS_ITEMIDS, "Dialysis Catheter ist kein Verfahren"
    assert 225436 not in DIALYSIS_ITEMIDS, "CRRT Filter Change ist kein Verfahren"
    assert 225441 in DIALYSIS_ITEMIDS, "Hemodialysis fehlt"
    assert len(DIALYSIS_ITEMIDS) >= 5


def test_only_blood_potassium_counts():
    """Urine, ascites, pleural fluid and CSF all carry the word."""

    assert set(SERUM_POTASSIUM_ITEMIDS) == {50822, 50971, 52452, 52610}
    for urine_or_worse in (51097, 50847, 51800, 51064):
        assert urine_or_worse not in SERUM_POTASSIUM_ITEMIDS


def test_the_nearest_lab_is_one_value_per_ecg():
    sql = nearest_lab_query("m.parquet", "l.parquet", itemids=(50971,))
    assert "rank_by_distance = 1" in sql
    assert "PARTITION BY m.study_id" in sql
    assert "abs(DATE_DIFF" in sql
    assert "locked_test" not in sql


def test_all_dialysis_labs_share_one_scan_but_not_units():
    identifiers = {item.analyte_id for item in DIALYSIS_LAB_ANALYTES}
    assert identifiers == {
        "potassium",
        "calcium_total",
        "calcium_ionized",
        "magnesium",
        "ph",
    }
    total = next(item for item in DIALYSIS_LAB_ANALYTES if item.analyte_id == "calcium_total")
    ionized = next(
        item for item in DIALYSIS_LAB_ANALYTES if item.analyte_id == "calcium_ionized"
    )
    assert total.unit != ionized.unit
    assert set(total.itemids).isdisjoint(ionized.itemids)
    sql = nearest_labs_query("m.parquet", "l.parquet")
    assert sql.count("read_parquet('l.parquet')") == 1
    assert "PARTITION BY m.study_id, a.analyte_id" in sql
    assert "locked_test" not in sql


def test_within_subject_fdr_has_design_and_global_families():
    frame = pd.DataFrame(
        {
            "design": ["dialysis_k", "dialysis_k", "dialysis_mg"],
            "p_value": [0.01, 0.04, 0.03],
        }
    )
    result = add_within_subject_fdr(frame)
    assert result.loc[:1, "design_q_value"].tolist() == pytest.approx([0.02, 0.04])
    assert result["global_q_value"].notna().all()


# --- The paired differences --------------------------------------------------


def test_the_difference_is_second_minus_first():
    pairs = _pairs([(1, 10, 11, 100.0, 130.0, 2.0, 1.0)])
    labs = _labs([(10, 6.0), (11, 4.0)])
    changes = paired_change_table(pairs, labs, MARKERS)
    row = changes.iloc[0]
    assert row["delta_lab"] == pytest.approx(-2.0)
    assert row["delta_marker_a"] == pytest.approx(30.0)
    assert row["delta_marker_b"] == pytest.approx(-1.0)


def test_an_implausible_lab_value_removes_the_difference_not_the_pair():
    """A unit error at one end would destroy the whole distribution."""

    pairs = _pairs([(1, 10, 11, 100.0, 130.0, 2.0, 1.0)])
    labs = _labs([(10, 6.0), (11, 640.0)])
    changes = paired_change_table(pairs, labs, MARKERS)
    row = changes.iloc[0]
    assert not np.isfinite(row["delta_lab"])
    assert bool(row["lab_within_bounds"]) is False
    # The marker difference survives: it is not what was implausible.
    assert row["delta_marker_a"] == pytest.approx(30.0)


def test_a_missing_lab_on_one_side_yields_no_difference():
    pairs = _pairs([(1, 10, 11, 100.0, 130.0, 2.0, 1.0)])
    labs = _labs([(10, 4.5)])
    changes = paired_change_table(pairs, labs, MARKERS)
    assert not np.isfinite(changes.iloc[0]["delta_lab"])


def test_the_bounds_are_declared_and_physiological():
    lower, upper = POTASSIUM_BOUNDS
    assert lower < 3.0 < 5.5 < upper
    assert upper <= 12.0, "ueber 12 mmol/l ist kein Messwert, sondern ein Fehler"


# --- The association ---------------------------------------------------------


def _linked(rng, n=300, slope=-1.0, noise=0.3):
    delta_lab = rng.normal(-0.4, 1.0, size=n)
    frame = pd.DataFrame({"delta_lab": delta_lab, "subject_id": np.arange(n) + 1})
    frame["delta_marker_a"] = slope * delta_lab + rng.normal(0, noise, size=n)
    frame["delta_marker_b"] = rng.normal(0, 1.0, size=n)
    return frame


def test_a_real_within_subject_link_is_found_with_the_right_sign():
    rng = np.random.default_rng(5)
    result = within_subject_association(_linked(rng), MARKERS)
    row = result.loc[result["feature_id"].eq("marker_a")].iloc[0]
    assert row["spearman_rho"] < -0.7
    assert row["p_value"] < 1e-10
    assert row["n_pairs"] == 300


def test_an_unlinked_marker_shows_nothing():
    rng = np.random.default_rng(6)
    result = within_subject_association(_linked(rng), MARKERS)
    row = result.loc[result["feature_id"].eq("marker_b")].iloc[0]
    assert abs(row["spearman_rho"]) < 0.15
    assert row["p_value"] > 0.01


def test_the_strongest_link_is_listed_first():
    rng = np.random.default_rng(7)
    result = within_subject_association(_linked(rng), MARKERS)
    assert result.iloc[0]["feature_id"] == "marker_a"


def test_too_few_pairs_are_not_reported_at_all():
    rng = np.random.default_rng(8)
    result = within_subject_association(
        _linked(rng, n=25), MARKERS, minimum_pairs=30
    )
    assert result.empty


def test_a_marker_absent_from_the_frame_is_skipped_not_invented():
    rng = np.random.default_rng(9)
    result = within_subject_association(
        _linked(rng), ("marker_a", "never_computed")
    )
    assert set(result["feature_id"]) == {"marker_a"}


def test_an_empty_frame_returns_the_empty_shape():
    result = within_subject_association(pd.DataFrame(), MARKERS)
    assert result.empty
    assert "spearman_rho" in result.columns


def test_the_window_records_its_own_policy():
    payload = DIALYSIS_WINDOW.to_dict()
    for key in ("procedure_itemids", "minimum_hours_between", "version"):
        assert key in payload
    assert payload["maximum_hours_between"] <= 48.0, (
        "laenger und das Paar umschliesst mehr als eine Sitzung"
    )


# --- Dose response -----------------------------------------------------------


def test_the_dose_analysis_stays_inside_the_treated_group():
    """Comparing treated with untreated puts confounding by indication back."""

    from vcg_biomarker.mimic.within_subject import (
        DRUG_WINDOW, dose_pair_query, drug_class_pattern,
    )

    sql = dose_pair_query(
        "m.parquet", "i.parquet", "d.parquet", "read_parquet('f')",
        window=DRUG_WINDOW, marker_columns="a.x",
        label_pattern=drug_class_pattern("beta_blocker"),
    )
    assert "JOIN dose_identity x" in sql, "ein LEFT JOIN wuerde Unbehandelte einlassen"
    assert "x.starttime > c.first_ecg_time" in sql
    assert "x.starttime < c.second_ecg_time" in sql


def test_the_drug_window_is_short_enough_to_measure_a_drug():
    """Measured: rho +0.151 at four hours, +0.036 at twenty-four."""

    from vcg_biomarker.mimic.within_subject import DRUG_WINDOW

    assert DRUG_WINDOW.maximum_hours_between <= 6.0


def test_doses_are_ranked_inside_each_drug_not_pooled_raw():
    """Milligrams of metoprolol and micrograms of esmolol are not comparable."""

    from vcg_biomarker.mimic.within_subject import normalise_dose_within_drug

    frame = pd.DataFrame(
        {
            "dose_itemid": [1] * 60 + [2] * 60,
            "dose_amount": list(range(1,61)) + [x * 1000 for x in range(1,61)],
            "dose_unit": ["mg"]*60 + ["mcg"]*60,
        }
    )
    ranked = normalise_dose_within_drug(frame, minimum_administrations=50)
    for item in (1, 2):
        part = ranked.loc[ranked["dose_itemid"].eq(item), "dose_rank"]
        assert part.min() == pytest.approx(1 / 60, abs=1e-6)
        assert part.max() == pytest.approx(1.0)


def test_a_drug_given_too_rarely_gets_no_dose_rank():
    from vcg_biomarker.mimic.within_subject import normalise_dose_within_drug

    frame = pd.DataFrame({"dose_itemid": [1] * 10, "dose_amount": range(10)})
    ranked = normalise_dose_within_drug(frame, minimum_administrations=50)
    assert ranked["dose_rank"].isna().all()


def test_a_dose_response_is_found_when_it_exists():
    from vcg_biomarker.mimic.within_subject import dose_response_association

    rng = np.random.default_rng(11)
    dose = rng.uniform(0, 1, size=400)
    frame = pd.DataFrame(
        {
            "dose_rank": dose,
            "subject_id": np.arange(400) + 1,
            "delta_marker_a": -8.0 * dose + rng.normal(0, 1.0, size=400),
            "delta_marker_b": rng.normal(0, 1.0, size=400),
        }
    )
    result = dose_response_association(frame, MARKERS)
    row = result.loc[result["feature_id"].eq("marker_a")].iloc[0]
    assert row["spearman_rho"] < -0.8
    assert row["p_value"] < 1e-20
    other = result.loc[result["feature_id"].eq("marker_b")].iloc[0]
    assert abs(other["spearman_rho"]) < 0.15


def test_an_unknown_drug_class_is_refused_not_guessed():
    from vcg_biomarker.mimic.within_subject import drug_class_pattern

    with pytest.raises(ValueError, match="Unbekannte"):
        drug_class_pattern("kein_medikament")


def test_the_known_classes_resolve_to_real_drug_names():
    from vcg_biomarker.mimic.within_subject import drug_class_pattern

    assert "metoprolol" in drug_class_pattern("beta_blocker")
    assert "diltiazem" in drug_class_pattern("calcium_channel_blocker")
    assert drug_class_pattern("adenosine") == "adenosine"


# --- The positive control the dose design needs -------------------------------


def test_the_dose_query_pairs_a_column_from_the_master_index():
    """``machine_heart_rate_bpm`` is the one direction known in advance.

    It carries the role ``ecg_context_comparator``, so it is deliberately in
    no feature panel -- which meant the beta-blocker arm ran without its
    control until the query learned to pair a master column too.
    """

    sql = dose_pair_query(
        "master.parquet", "inputevents/*.parquet", "d_items.parquet",
        "read_parquet('f/*.parquet')",
        window=DRUG_WINDOW,
        marker_columns="b.tpeak_tend_ms - a.tpeak_tend_ms AS delta_tpeak_tend_ms",
        label_pattern="metoprolol",
        context_columns=("machine_heart_rate_bpm",),
    )
    assert "delta_machine_heart_rate_bpm" in sql
    assert "ma.study_id = t.first_study_id" in sql
    assert "mb.study_id = t.second_study_id" in sql


def test_without_a_context_column_the_query_gains_no_extra_join():
    sql = dose_pair_query(
        "master.parquet", "inputevents/*.parquet", "d_items.parquet",
        "read_parquet('f/*.parquet')",
        window=DRUG_WINDOW,
        marker_columns="b.tpeak_tend_ms - a.tpeak_tend_ms AS delta_tpeak_tend_ms",
        label_pattern="metoprolol",
    )
    assert " ma." not in sql and " mb." not in sql


def test_all_drug_classes_share_one_physical_scan_plan():
    sql = dose_pairs_all_classes_query(
        "master.parquet",
        "inputevents/*.parquet",
        "d_items.parquet",
        "read_parquet('features/*.parquet')",
        window=DRUG_WINDOW,
        marker_ids=("marker_a", "marker_b"),
        class_patterns={
            "beta_blocker": "metoprolol|esmolol",
            "adenosine": "adenosine",
        },
        context_columns=("machine_heart_rate_bpm",),
    )
    assert sql.count("read_parquet('inputevents/*.parquet')") == 1
    assert sql.count("read_parquet('d_items.parquet')") == 1
    assert sql.count("read_parquet('master.parquet')") == 1
    assert sql.count("read_parquet('features/*.parquet')") == 1
    assert "feature_rows AS MATERIALIZED" in sql
    assert "d.drug_class" in sql
    assert "item_dictionary AS MATERIALIZED" in sql
    assert "delta_machine_heart_rate_bpm" in sql


def test_combined_dose_query_matches_separate_class_queries(tmp_path):
    times = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 02:00", "2026-01-01 05:00"]
    )
    master = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "study_id": [10, 11, 12],
            "split": ["train", "train", "train"],
            "ecg_time": times,
            "hadm_id": [100, 100, 100],
            "machine_heart_rate_bpm": [100.0, 90.0, 80.0],
        }
    )
    features = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "study_id": [10, 11, 12],
            "extraction_status": ["ok", "ok", "ok"],
            "signal_quality_score": [.8, .9, .95],
            "marker_a": [1.0, 3.0, 8.0],
            "marker_b": [10.0, 9.0, 7.0],
        }
    )
    inputs = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "starttime": pd.to_datetime(
                ["2026-01-01 01:00", "2026-01-01 03:00"]
            ),
            "itemid": [1, 2],
            "amount": [5.0, 10.0],
            "amountuom": ["mg", "mg"],
            "statusdescription": ["FinishedRunning", "Stopped"],
            "orderid": [123,124],
            "linkorderid": [123,124],
            "endtime": pd.to_datetime(["2026-01-01 01:30", "2026-01-01 03:30"]),
        }
    )
    items = pd.DataFrame(
        {"itemid": [1, 2], "label": ["Metoprolol", "Diltiazem"]}
    )
    paths = {
        "master": tmp_path / "master.parquet",
        "features": tmp_path / "features.parquet",
        "inputs": tmp_path / "inputs.parquet",
        "items": tmp_path / "items.parquet",
    }
    master.to_parquet(paths["master"], index=False)
    features.to_parquet(paths["features"], index=False)
    inputs.to_parquet(paths["inputs"], index=False)
    items.to_parquet(paths["items"], index=False)
    sql_paths = {
        key: str(path).replace("\\", "/") for key, path in paths.items()
    }
    feature_clause = f"read_parquet('{sql_paths['features']}')"
    patterns = {"beta": "metoprolol", "calcium": "diltiazem"}
    connection = duckdb.connect(":memory:")
    try:
        combined = connection.execute(
            dose_pairs_all_classes_query(
                sql_paths["master"],
                sql_paths["inputs"],
                sql_paths["items"],
                feature_clause,
                window=DRUG_WINDOW,
                marker_ids=MARKERS,
                class_patterns=patterns,
                context_columns=DOSE_CONTEXT_COLUMNS,
            )
        ).fetchdf()
        separate = []
        for drug_class, pattern in patterns.items():
            frame = connection.execute(
                dose_pair_query(
                    sql_paths["master"],
                    sql_paths["inputs"],
                    sql_paths["items"],
                    feature_clause,
                    window=DRUG_WINDOW,
                    marker_columns=(
                        "b.marker_a - a.marker_a AS delta_marker_a, "
                        "b.marker_b - a.marker_b AS delta_marker_b"
                    ),
                    label_pattern=pattern,
                    context_columns=DOSE_CONTEXT_COLUMNS,
                )
            ).fetchdf()
            frame.insert(0, "drug_class", drug_class)
            separate.append(frame)
    finally:
        connection.close()

    expected = pd.concat(separate, ignore_index=True)
    columns = [
        "drug_class",
        "first_study_id",
        "second_study_id",
        "dose_amount",
        "dose_itemid",
        "delta_marker_a",
        "delta_marker_b",
        "delta_machine_heart_rate_bpm",
    ]
    expected = expected[columns].sort_values(columns[:3]).reset_index(drop=True)
    observed = combined[columns].sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected)


def test_the_context_column_is_a_paired_difference_not_a_level():
    """A heart rate *after* the dose says nothing; the change does."""

    sql = dose_pair_query(
        "master.parquet", "inputevents/*.parquet", "d_items.parquet",
        "read_parquet('f/*.parquet')",
        window=DRUG_WINDOW,
        marker_columns="b.x - a.x AS delta_x",
        label_pattern="metoprolol",
        context_columns=("machine_heart_rate_bpm",),
    )
    assert (
        "mb.machine_heart_rate_bpm - ma.machine_heart_rate_bpm"
        in sql
    )


def test_the_control_column_is_named_once_and_used_everywhere():
    """Two names for the control would let one arm run without it."""

    assert DOSE_CONTEXT_COLUMNS == ("machine_heart_rate_bpm",)
    # It must not also sit in the marker list: no feature panel contains it,
    # so it would silently drop out of every result.
    assert not set(DOSE_CONTEXT_COLUMNS) & set(WITHIN_SUBJECT_MARKERS)
