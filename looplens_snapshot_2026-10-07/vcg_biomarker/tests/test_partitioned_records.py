import pandas as pd
import pytest

from vcg_biomarker.mimic.partitioned_records import PartitionedRecords, records_frame


def test_endpoint_buffers_release_dicts_preserve_order_and_schema(tmp_path):
    first = [{"endpoint": "a", "value": 1., "p": .02}, {"endpoint": "a", "value": 2., "p": .5}]
    second = [{"endpoint": "b", "value": 3., "p": .3}]
    rows = PartitionedRecords(tmp_path)
    rows.extend(first)
    assert rows[0:] == first
    rows.flush()
    assert not rows.pending and len(rows) == 2
    offset = len(rows)
    rows.extend(second)
    assert rows[offset:] == second
    rows.flush()
    pd.testing.assert_frame_equal(records_frame(rows), pd.DataFrame(first + second))
    with pytest.raises(ValueError, match="Historical"):
        rows[0:]


def test_null_schema_promotion_and_empty(tmp_path):
    rows = PartitionedRecords(tmp_path)
    assert records_frame(rows).empty
    rows.append({"x": None})
    rows.flush()
    rows.append({"x": 3., "optional": "present"})
    result = records_frame(rows)
    assert pd.isna(result.iloc[0].x)
    assert result.iloc[1].x == 3
    assert pd.isna(result.iloc[0].optional)


def test_buffer_roundtrip_through_existing_checkpoint(tmp_path):
    from vcg_biomarker.mimic.validation import (
        _write_endpoint_validation_checkpoint, _load_endpoint_validation_checkpoint)
    rows = PartitionedRecords(tmp_path / "parts")
    rows.extend([{"endpoint": "a", "p": .1}])
    rows.flush()
    rows.extend([{"endpoint": "b", "p": .2}])
    path = tmp_path / "endpoint.json"
    _write_endpoint_validation_checkpoint(path, input_hash="test", endpoint_id="b",
        result_groups={"records": rows}, group_offsets={"records": 1})
    restored = _load_endpoint_validation_checkpoint(path, input_hash="test", endpoint_id="b")
    assert restored["result_groups"]["records"] == [{"endpoint": "b", "p": .2}]


def test_resume_adopts_same_verified_parquet_without_dict_materialization(tmp_path):
    from vcg_biomarker.mimic.validation import (
        _write_endpoint_validation_checkpoint, _load_endpoint_validation_checkpoint,
        _extend_validation_result_groups)
    rows = PartitionedRecords(tmp_path / "parts")
    rows.extend([{"endpoint": "a", "p": .1}])
    path = tmp_path / "endpoint.json"
    _write_endpoint_validation_checkpoint(path, input_hash="test", endpoint_id="a",
        result_groups={"records": rows}, group_offsets={"records": 0})
    assert rows.pending == []
    restored = _load_endpoint_validation_checkpoint(path, input_hash="test", endpoint_id="a",
                                                    materialize_partitions=False)
    assert restored["result_groups"]["records"] == []
    resumed = PartitionedRecords(tmp_path / "resume")
    _extend_validation_result_groups({"records": resumed}, restored["result_groups"],
                                    partition_paths=restored["result_group_paths"])
    assert resumed.parts == rows.parts
    assert not (tmp_path / "resume").exists()
    pd.testing.assert_frame_equal(records_frame(resumed), records_frame(rows))
    resumed.parts[0].write_bytes(b"corrupted test fixture")
    with pytest.raises(RuntimeError, match="changed after commit"):
        records_frame(resumed)
