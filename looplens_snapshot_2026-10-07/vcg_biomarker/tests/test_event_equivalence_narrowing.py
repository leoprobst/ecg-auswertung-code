"""Two narrowings in the canonical/event-backed equivalence comparison.

Both exist because an exact comparison can fail for reasons that have nothing
to do with the evidence, and both are written so the comparison gets *stricter*
over time rather than accumulating permanent exemptions.

The first is the additive-only declaration. It records that the canonical
evidence on disk predates a column -- a statement about a moment in time that
stops being true once canonical is rebuilt. Left unnarrowed it would exclude
the column from comparison forever; narrowed, the column rejoins the strict
comparison the moment both sides have it.

The second is floating-point reductions. ``stddev`` is a parallel reduction and
IEEE-754 addition is not associative, so the result follows the thread schedule
in its last few bits. Comparing those bits tests the scheduler, not the data,
so the declared columns are compared at single precision instead -- roughly
seven significant digits, far below anything a real change could hide in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")

from vcg_biomarker.mimic.event_equivalence import (  # noqa: E402
    FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE,
    _compare_or_resume_table,
    _comparison_projection,
    _effective_additive_only_columns,
)


def _write(path: Path, frame) -> Path:
    frame.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("right,expected", [([1, 1, None], 0), ([1, 2, None], 1), ([2, 2, None], 2)])
def test_equal_cardinality_difference_counts_keep_duplicates_and_nulls(tmp_path, right, expected):
    left_path = _write(tmp_path / "left.parquet", pd.DataFrame({"value": [1., 1., None]}))
    right_path = _write(tmp_path / "right.parquet", pd.DataFrame({"value": right}, dtype=float))
    with duckdb.connect() as connection:
        result, reused = _compare_or_resume_table(
            connection, checkpoint_path=tmp_path / "checkpoint.json", checkpoint={},
            table_name="test", canonical_path=left_path, event_backed_path=right_path,
        )
    assert not reused
    assert result["canonical_only_row_count"] == expected
    assert result["event_backed_only_row_count"] == expected
    assert result["equivalent"] is (expected == 0)


def test_unequal_cardinality_is_rejected_without_derived_zero(tmp_path):
    left = _write(tmp_path / "left.parquet", pd.DataFrame({"v": [1, 1]}))
    right = _write(tmp_path / "right.parquet", pd.DataFrame({"v": [1]}))
    with duckdb.connect() as connection:
        result, _ = _compare_or_resume_table(
            connection, checkpoint_path=tmp_path / "checkpoint.json", checkpoint={},
            table_name="test", canonical_path=left, event_backed_path=right,
        )
    assert not result["equivalent"]
    assert result["event_backed_only_row_count"] == -1
    assert result["difference_count_method"] == "not_computed_schema_or_cardinality_mismatch"


def test_old_comparison_policy_is_recomputed_then_current_checkpoint_reused(tmp_path):
    path = _write(tmp_path / "data.parquet", pd.DataFrame({"v": [1, 1]}))
    checkpoint = {}
    kwargs = dict(checkpoint_path=tmp_path / "checkpoint.json", checkpoint=checkpoint,
                  table_name="test", canonical_path=path, event_backed_path=path)
    with duckdb.connect() as connection:
        first, reused = _compare_or_resume_table(connection, **kwargs)
        assert not reused
        first["comparison_policy"].pop("difference_count_method")
        _, reused = _compare_or_resume_table(connection, **kwargs)
        assert not reused
        _, reused = _compare_or_resume_table(connection, **kwargs)
        assert reused


class TestEffectiveAdditiveColumns:
    def test_column_absent_from_canonical_stays_additive(self):
        assert _effective_additive_only_columns(
            ("grown",),
            table_name="t.parquet",
            canonical_schema_names={"subject_id"},
            event_schema_names={"subject_id", "grown"},
        ) == ("grown",)

    def test_column_present_on_both_sides_is_no_longer_additive(self):
        """A rebuilt canonical must not leave the column permanently excluded."""

        assert _effective_additive_only_columns(
            ("grown",),
            table_name="t.parquet",
            canonical_schema_names={"subject_id", "grown"},
            event_schema_names={"subject_id", "grown"},
        ) == ()

    def test_partial_catch_up_narrows_only_what_caught_up(self):
        assert _effective_additive_only_columns(
            ("early", "late"),
            table_name="t.parquet",
            canonical_schema_names={"subject_id", "early"},
            event_schema_names={"subject_id", "early", "late"},
        ) == ("late",)

    def test_column_missing_from_the_rebuild_is_still_an_error(self):
        """The other half of the guarantee is unconditional: a declared column
        gone from the event-backed side means the rebuild lost it."""

        with pytest.raises(RuntimeError, match="missing from the event-backed"):
            _effective_additive_only_columns(
                ("grown",),
                table_name="t.parquet",
                canonical_schema_names={"subject_id"},
                event_schema_names={"subject_id"},
            )

    def test_empty_declaration_needs_no_schema(self):
        assert _effective_additive_only_columns(
            (),
            table_name="t.parquet",
            canonical_schema_names=set(),
            event_schema_names=set(),
        ) == ()


class TestComparisonProjection:
    def test_plain_columns_are_quoted_and_untouched(self):
        assert _comparison_projection(["a", "b"]) == '"a", "b"'

    def test_declared_reductions_are_cast_to_single_precision(self):
        projection = _comparison_projection(
            ["a", "sd"], single_precision_columns=("sd",)
        )
        assert projection == '"a", CAST("sd" AS REAL) AS "sd"'

    def test_excluded_columns_are_dropped_entirely(self):
        projection = _comparison_projection(
            ["a", "tie", "sd"],
            single_precision_columns=("sd",),
            exclude=("tie",),
        )
        assert projection == '"a", CAST("sd" AS REAL) AS "sd"'


class TestTableComparison:
    def test_rebuilt_canonical_compares_the_formerly_additive_column(
        self, tmp_path
    ):
        """Once both sides carry it, a difference in it must be caught."""

        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1, 2], "grown": [10.0, 20.0]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1, 2], "grown": [10.0, 99.0]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                additive_only_columns=("grown",),
            )
        assert not comparison["equivalent"], (
            "a stale additive declaration must not hide a real difference"
        )

    def test_still_additive_column_is_excluded_from_the_comparison(
        self, tmp_path
    ):
        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1, 2]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1, 2], "grown": [10.0, 99.0]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                additive_only_columns=("grown",),
            )
        assert comparison["equivalent"]
        assert not comparison["full_table_equivalent"]

    def test_last_bit_disagreement_in_a_declared_reduction_is_tolerated(
        self, tmp_path
    ):
        """The measured signature: three ULP apart, nothing else different."""

        exact = 8.491847585174629
        one_scheduler_later = 8.49184758517462
        assert exact != one_scheduler_later
        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [exact]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [one_scheduler_later]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                floating_point_reduction_columns=("sd",),
            )
        assert comparison["equivalent"]
        assert comparison["canonical_only_row_count"] == 0
        # The narrowing must stay visible so the attestation audit sees it.
        assert not comparison["full_table_equivalent"]
        assert comparison["comparison_policy"]["mode"] == (
            "exact_full_table_single_precision_reductions"
        )

    def test_a_real_difference_in_a_reduction_column_still_fails(
        self, tmp_path
    ):
        """Single precision still holds ~7 digits; a real change is far larger."""

        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [8.491847]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [8.492847]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                floating_point_reduction_columns=("sd",),
            )
        assert not comparison["equivalent"]

    def test_null_against_a_value_is_never_tolerated(self, tmp_path):
        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [8.491847585174629]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [None]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                floating_point_reduction_columns=("sd",),
            )
        assert not comparison["equivalent"]

    def test_undeclared_reduction_column_is_compared_bit_for_bit(
        self, tmp_path
    ):
        """The narrowing applies only where it is declared."""

        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [8.491847585174629]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1], "sd": [8.49184758517462]}),
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="t.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
            )
        assert not comparison["equivalent"]

    def test_declared_reduction_column_must_exist(self, tmp_path):
        canonical = _write(
            tmp_path / "canonical.parquet",
            pd.DataFrame({"subject_id": [1]}),
        )
        event_backed = _write(
            tmp_path / "event.parquet",
            pd.DataFrame({"subject_id": [1]}),
        )
        with duckdb.connect(":memory:") as connection:
            with pytest.raises(RuntimeError, match="reduction columns"):
                _compare_or_resume_table(
                    connection,
                    checkpoint_path=tmp_path / "checkpoint.json",
                    checkpoint={},
                    table_name="t.parquet",
                    canonical_path=canonical,
                    event_backed_path=event_backed,
                    floating_point_reduction_columns=("absent",),
                )


def test_reduction_declaration_names_only_standard_deviations():
    """A guard on the declaration itself.

    The justification -- a parallel sum whose order follows the scheduler --
    holds for a standard deviation. It does not hold for a count, an id or a
    timestamp, and this narrowing must not drift onto one.
    """

    for table, columns in FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE.items():
        assert columns, f"{table} declares an empty reduction set"
        for column in columns:
            assert column.endswith("_stddev"), (
                f"{table}: {column} is not a standard deviation; single "
                "precision is not justified for it"
            )


class TestNewMedicationClassRetiresItself:
    """The third narrowing of the same family.

    A medication class recognised by the newer code but not by the evidence on
    disk makes the event-backed side carry rows canonical cannot have. That
    tolerance, too, describes a moment rather than a rule, and rebuilding the
    canonical evidence ends it.
    """

    @staticmethod
    def _table(path: Path, classes) -> Path:
        return _write(
            path,
            pd.DataFrame(
                {
                    "subject_id": list(range(1, len(classes) + 1)),
                    "medication_class": list(classes),
                }
            ),
        )

    def test_class_absent_from_canonical_is_still_new(self, tmp_path):
        from vcg_biomarker.mimic.event_equivalence import (
            _new_medication_class_is_still_new,
        )

        canonical = self._table(tmp_path / "c.parquet", ["beta_blocker"])
        with duckdb.connect(":memory:") as connection:
            assert _new_medication_class_is_still_new(
                connection,
                canonical_path=canonical,
                class_column="medication_class",
                new_medication_class="adenosine",
            )

    def test_class_present_in_canonical_retires_the_tolerance(self, tmp_path):
        from vcg_biomarker.mimic.event_equivalence import (
            _new_medication_class_is_still_new,
        )

        canonical = self._table(
            tmp_path / "c.parquet", ["beta_blocker", "adenosine"]
        )
        with duckdb.connect(":memory:") as connection:
            assert not _new_medication_class_is_still_new(
                connection,
                canonical_path=canonical,
                class_column="medication_class",
                new_medication_class="adenosine",
            )

    def test_caught_up_canonical_gets_the_strict_comparison(self, tmp_path):
        """Not an error, and not an exemption -- the plain comparison."""

        from vcg_biomarker.mimic.event_equivalence import (
            _compare_or_resume_table_excluding_new_class_rows,
        )

        canonical = self._table(
            tmp_path / "c.parquet", ["beta_blocker", "adenosine"]
        )
        event_backed = self._table(
            tmp_path / "e.parquet", ["beta_blocker", "adenosine"]
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table_excluding_new_class_rows(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="medication_evidence.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                new_medication_class="adenosine",
            )
        assert comparison["equivalent"]
        assert comparison["full_table_equivalent"]
        assert comparison["comparison_policy"]["mode"] == "exact_full_table"

    def test_a_difference_in_the_caught_up_class_is_caught(self, tmp_path):
        """The rows the tolerance used to set aside are now compared."""

        from vcg_biomarker.mimic.event_equivalence import (
            _compare_or_resume_table_excluding_new_class_rows,
        )

        canonical = self._table(
            tmp_path / "c.parquet", ["beta_blocker", "adenosine"]
        )
        event_backed = self._table(
            tmp_path / "e.parquet", ["beta_blocker", "amiodarone"]
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table_excluding_new_class_rows(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="medication_evidence.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                new_medication_class="adenosine",
            )
        assert not comparison["equivalent"]

    def test_a_class_still_new_keeps_the_row_count_tolerance(self, tmp_path):
        from vcg_biomarker.mimic.event_equivalence import (
            _compare_or_resume_table_excluding_new_class_rows,
        )

        canonical = self._table(tmp_path / "c.parquet", ["beta_blocker"])
        event_backed = self._table(
            tmp_path / "e.parquet", ["beta_blocker", "adenosine"]
        )
        with duckdb.connect(":memory:") as connection:
            comparison, _ = _compare_or_resume_table_excluding_new_class_rows(
                connection,
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint={},
                table_name="medication_evidence.parquet",
                canonical_path=canonical,
                event_backed_path=event_backed,
                new_medication_class="adenosine",
            )
        assert comparison["equivalent"]
        assert comparison["excluded_new_medication_class"] == "adenosine"
        assert comparison["excluded_new_medication_class_row_count"] == 1

    def test_unexplained_row_growth_still_fails_while_the_class_is_new(
        self, tmp_path
    ):
        """The tolerance covers the new class only, never row growth at large."""

        from vcg_biomarker.mimic.event_equivalence import (
            _compare_or_resume_table_excluding_new_class_rows,
        )

        canonical = self._table(tmp_path / "c.parquet", ["beta_blocker"])
        event_backed = self._table(
            tmp_path / "e.parquet",
            ["beta_blocker", "adenosine", "digoxin"],
        )
        with duckdb.connect(":memory:") as connection:
            with pytest.raises(RuntimeError, match="not fully explained"):
                _compare_or_resume_table_excluding_new_class_rows(
                    connection,
                    checkpoint_path=tmp_path / "checkpoint.json",
                    checkpoint={},
                    table_name="medication_evidence.parquet",
                    canonical_path=canonical,
                    event_backed_path=event_backed,
                    new_medication_class="adenosine",
                )
