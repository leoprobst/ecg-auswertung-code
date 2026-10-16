"""An unchanged inventory is reused byte for byte; a changed source rewrites it."""
import os

from vcg_biomarker.mimic.inventory import build_inventory
from vcg_biomarker.mimic.storage import file_sha256
from vcg_biomarker.tests.test_provenance_integrity import _config


def _state(config):
    paths = [config.work_root / name for name in ("data_inventory.json", "inventory.manifest.json")]
    return [(file_sha256(p), os.stat(p).st_mtime_ns) for p in paths]


def test_identical_inventory_is_reused_and_a_changed_source_rewrites(tmp_path):
    config = _config(tmp_path)
    first = build_inventory(config)
    before = _state(config)
    second = build_inventory(config)
    # The marker freeze fingerprints these files; a rewrite with a new
    # timestamp used to invalidate the freeze of a run's own pilots.
    assert second.created_at == first.created_at
    assert _state(config) == before
    source = config.mimic_iv_root / "hosp" / "patients.csv.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"not a real table")
    third = build_inventory(config)
    assert third.input_hash != first.input_hash
    assert _state(config) != before
