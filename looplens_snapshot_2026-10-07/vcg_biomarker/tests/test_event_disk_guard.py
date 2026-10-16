from types import SimpleNamespace

import pyarrow as pa
import pytest

from vcg_biomarker.mimic import event_ingest as ingest


def test_guard_checks_existing_volume_without_creating_or_removing_paths(tmp_path, monkeypatch):
    target = tmp_path/"new"/"nested"
    checked = []
    def usage(path):
        checked.append(path)
        return SimpleNamespace(free=ingest.EVENT_DISK_RESERVE_BYTES+99)
    monkeypatch.setattr(ingest.shutil, "disk_usage", usage)
    with pytest.raises(OSError, match="checkpoints retained") as error:
        ingest._require_event_disk_space(target, 100)
    assert error.value.errno == 28
    assert checked == [tmp_path.resolve()]
    assert not target.parent.exists()
    ingest._require_event_disk_space(target, 99)


def test_source_batch_does_not_write_when_reserve_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    checkpoint = tmp_path/"existing_checkpoint.json"
    checkpoint.write_text("unchanged")
    with pytest.raises(OSError):
        ingest._write_staged_event_batch(tmp_path, "labevents", 0, pa.record_batch({"value": [1.]}))
    assert checkpoint.read_text() == "unchanged"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing_checkpoint.json"]


@pytest.mark.parametrize("allowance", [-1, True, 1.5])
def test_invalid_allowance_rejected(tmp_path, allowance):
    with pytest.raises(ValueError):
        ingest._require_event_disk_space(tmp_path, allowance)
