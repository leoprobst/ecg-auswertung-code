"""A laboratory group without a declared unit was silently deleted.

``_lab_unit_status_case_sql`` listed the expected units for twelve laboratory
groups.  Every other group fell through to ``ELSE 'unit_unrecognized'`` -- and
every window aggregate in the evidence build filters on
``value_status = 'usable'``.  Measured on the 52.1 million ingested rows, that
removed **28.6 million** of them: sodium, glucose, urea nitrogen, bicarbonate,
leukocytes, albumin, TSH, free T4, HbA1c, CRP, D-dimer and the coagulation
times.  INR was lost twice over, because it is unitless and the generic
``unit_missing`` rule fired first.

Re-running the corrected rule over the same rows recovers 30,294,846 of them
and leaves 421,127 quarantined (0.8 %), almost all of which are laboratory
orders that never returned a result.

The two units that are *not* recovered are the point of the second half of these
tests: CK-MB in per cent is the relative index rather than the mass
concentration, and a bare "ng/ml" D-dimer does not say whether it is a
fibrinogen-equivalent or a D-dimer unit -- the two differ by about a factor of
two.  Accepting either would put two distributions in one column.
"""

from __future__ import annotations

import duckdb
import pytest

from vcg_biomarker.mimic.evidence import (
    LAB_AMBIGUOUS_UNITS,
    LAB_EXPECTED_UNITS,
    LAB_UNITLESS_GROUPS,
    _lab_unit_status_case_sql,
)
from vcg_biomarker.mimic.mappings import LAB_ITEM_GROUPS


def _classify(rows):
    """Run the production CASE expression over a handful of literal rows."""

    case = _lab_unit_status_case_sql()
    values = ", ".join(
        "({itemid}, {valuenum}, {unit})".format(
            itemid=itemid,
            valuenum="NULL" if valuenum is None else repr(float(valuenum)),
            unit="NULL" if unit is None else "'" + unit.replace("'", "''") + "'",
        )
        for itemid, valuenum, unit in rows
    )
    query = f"""
    WITH source(itemid, valuenum, valueuom) AS (VALUES {values})
    SELECT
        CASE
            WHEN TRY_CAST(valuenum AS DOUBLE) IS NULL THEN 'non_numeric'
            {case}
            WHEN NULLIF(TRIM(valueuom), '') IS NULL THEN 'unit_missing'
            ELSE 'unit_unrecognized'
        END AS value_status
    FROM source
    """
    with duckdb.connect() as connection:
        return [row[0] for row in connection.execute(query).fetchall()]


def _first_item(group: str) -> int:
    return int(sorted(LAB_ITEM_GROUPS[group])[0])


def test_every_ingested_group_has_a_declared_unit_or_is_unitless():
    # The defect was a group having no entry at all, so this is the guard that
    # keeps a newly added group from disappearing the same way.
    undeclared = sorted(
        group
        for group in LAB_ITEM_GROUPS
        if group not in LAB_EXPECTED_UNITS and group not in LAB_UNITLESS_GROUPS
    )
    assert undeclared == [], undeclared


@pytest.mark.parametrize(
    "group,unit",
    [
        ("sodium", "mEq/L"),
        ("glucose", "mg/dL"),
        ("urea_nitrogen", "mg/dL"),
        ("bicarbonate", "mEq/L"),
        ("white_blood_cells", "K/uL"),
        ("albumin", "g/dL"),
        ("tsh", "uIU/mL"),
        ("free_t4", "ng/dL"),
        ("hba1c", "%"),
        ("c_reactive_protein", "mg/L"),
        ("prothrombin_time", "sec"),
        ("partial_thromboplastin_time", "sec"),
        ("phosphate", "mg/dL"),
        ("potassium", "mEq/L"),
        ("creatinine", "mg/dL"),
    ],
)
def test_a_routine_result_in_its_usual_unit_is_usable(group: str, unit: str):
    assert _classify([(_first_item(group), 1.0, unit)]) == ["usable"]


def test_a_unitless_ratio_is_not_discarded_for_having_no_unit():
    # 1,767,424 INR values were lost to the generic missing-unit rule.
    assert _classify([(_first_item("inr"), 1.3, None)]) == ["usable"]
    assert _classify([(_first_item("inr"), 1.3, "")]) == ["usable"]


def test_a_missing_unit_still_disqualifies_a_group_that_should_have_one():
    assert _classify([(_first_item("sodium"), 139.0, None)]) == ["unit_missing"]


def test_an_unexpected_unit_is_still_quarantined():
    assert _classify([(_first_item("sodium"), 139.0, "mg/dL")]) == [
        "unit_unrecognized"
    ]


def test_a_result_without_a_number_is_quarantined_before_any_unit_rule():
    assert _classify([(_first_item("troponin_t"), None, "ng/mL")]) == [
        "non_numeric"
    ]


def test_a_different_quantity_in_the_same_group_is_named_not_absorbed():
    # CK-MB as a percentage of total CK is the relative index, not the mass
    # concentration the group is defined as.
    assert _classify([(_first_item("ck_mb"), 4.8, "%")]) == [
        "unit_scale_ambiguous"
    ]
    assert _classify([(_first_item("ck_mb"), 4.0, "ng/mL")]) == ["usable"]


def test_an_ambiguous_assay_scale_is_named_not_absorbed():
    # FEU runs about twice DDU; a bare ng/mL does not say which.
    assert _classify([(_first_item("d_dimer"), 856.0, "ng/mL FEU")]) == ["usable"]
    assert _classify([(_first_item("d_dimer"), 510.0, "ng/mL")]) == [
        "unit_scale_ambiguous"
    ]


def test_ambiguous_units_are_never_also_declared_as_expected():
    for group, ambiguous in LAB_AMBIGUOUS_UNITS.items():
        expected = set(LAB_EXPECTED_UNITS.get(group, ()))
        assert expected.isdisjoint(set(ambiguous)), group
