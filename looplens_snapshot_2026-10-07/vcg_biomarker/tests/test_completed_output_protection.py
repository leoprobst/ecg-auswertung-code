import json
from pathlib import Path
import pytest
from audit_tools.protect_completed_outputs import prepare, verify


def test_seal_excludes_live_state_and_detects_output_changes(tmp_path):
    root=tmp_path/'runs'/'run';root.mkdir(parents=True)
    (root/'guided_workflow.manifest.json').write_text(json.dumps({'completed_stages':['inventory']}))
    result=tmp_path/'result.txt';result.write_text('done')
    manifest=tmp_path/'inventory.manifest.json'
    manifest.write_text(json.dumps({'stage':'inventory','status':'complete','output_paths':[str(result)]}))
    seal=prepare(tmp_path,'run')
    assert {Path(x['path']).name for x in seal['files']}=={'inventory.manifest.json','result.txt'}
    assert verify(seal,full=True,require_readonly=False)['files_verified']==2
    result.write_text('modified')
    with pytest.raises(RuntimeError,match='changed'):
        verify(seal,full=True,require_readonly=False)


def test_seal_refuses_project_escape(tmp_path):
    work=tmp_path/'work';root=work/'runs'/'run';root.mkdir(parents=True)
    (root/'guided_workflow.manifest.json').write_text(json.dumps({'completed_stages':['inventory']}))
    outside=tmp_path/'other.txt';outside.write_text('outside')
    (work/'inventory.manifest.json').write_text(json.dumps({'stage':'inventory','status':'complete','output_paths':[str(outside)]}))
    with pytest.raises(RuntimeError,match='escaped'):
        prepare(work,'run')
