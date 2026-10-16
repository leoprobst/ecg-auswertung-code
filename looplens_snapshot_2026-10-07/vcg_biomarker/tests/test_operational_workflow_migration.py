from pathlib import Path
import pytest
from audit_tools.apply_operational_fix import migrated_workflow_checkpoint
from vcg_biomarker.mimic.storage import content_hash, source_tree_hash


def test_operational_migration_retains_completed_stages_and_rejects_unbound_hash(tmp_path):
    roots = [tmp_path / "old", tmp_path / "new"]
    for i, root in enumerate(roots):
        (root / "vcg_biomarker/mimic").mkdir(parents=True)
        (root / "vcg_biomarker/core").mkdir()
        (root / "vcg_biomarker/mimic/workflow.py").write_text('GUIDED_WORKFLOW_VERSION = "1"\n')
        (root / "vcg_biomarker/mimic/jobs.py").write_text(f'OPERATIONAL = {i}\n')
    payload = {"study_spec": {"stage_ids": ["extract-features", "validate"]},
               "project_config_hash": "config", "completed_stages": ["extract-features"],
               "remaining_stages": ["validate"], "status": "cancelled"}
    backend = content_hash({"mimic": source_tree_hash(roots[0] / "vcg_biomarker/mimic"),
                            "core": source_tree_hash(roots[0] / "vcg_biomarker/core")})
    payload["workflow_hash"] = content_hash({"workflow_version": "1", "study_spec": payload["study_spec"],
        "scientific_project_config_hash": "config", "implementation_hash": backend})
    changed = migrated_workflow_checkpoint(payload, *roots)
    assert changed["workflow_hash"] != payload["workflow_hash"]
    assert {k:v for k,v in changed.items() if k != "workflow_hash"} == {k:v for k,v in payload.items() if k != "workflow_hash"}
    with pytest.raises(RuntimeError, match="does not match"):
        migrated_workflow_checkpoint(dict(payload, workflow_hash="tampered"), *roots)
