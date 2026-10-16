import ast
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic import phenome
from vcg_biomarker.mimic.storage import content_hash


def declaration(root, base="input"):
    value = dict(schema_version="1.0", run_id=root.name, checkpoint_input_hash=base,
                 status="deferred_by_user", arm="future_phenome",
                 authorization="explicit_user_instruction", reason="User deferred prognosis arm")
    value["amendment_hash"] = content_hash(value)
    return value


def test_default_dispatch_and_retained_checkpoint_science_unchanged(monkeypatch):
    fixture=Path(__file__).parent/'fixtures/phenome_before_future_defer.py'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='4e76040be5bfd063ac83fafd1bd6efed9f8e9cb8767ff25b3a2a0998cc6daa3f'
    old=ast.parse(fixture.read_text(encoding='utf-8'))
    new=ast.parse(Path(phenome.__file__).read_text(encoding='utf-8'))
    funcs=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    before,after=funcs(old),funcs(new)
    for name,node in before.items():
        if name!='run_phenome_discovery':
            assert ast.dump(node)==ast.dump(after[name]),name
    main_old,main_new=before['run_phenome_discovery'],after['run_phenome_discovery']
    for name in ('input_hash','checkpoint_root'):
        assignments=lambda node:[n for n in ast.walk(node) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert [ast.dump(n) for n in assignments(main_old)]==[ast.dump(n) for n in assignments(main_new)]
    old_call=next(n for n in ast.walk(main_old) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='run_future_phenome_discovery')
    new_call=next(n for n in ast.walk(main_new) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='_run_future_arm')
    assert [ast.dump(k) for k in old_call.keywords]==[ast.dump(k) for k in new_call.keywords]
    assert [ast.dump(a) for a in old_call.args]==[ast.dump(a) for a in new_call.args[1:]]
    calls=[];sentinel=object();config=object();kwargs=dict(a=1,b=(2,3))
    monkeypatch.setattr(phenome,'run_future_phenome_discovery',lambda c,**k:(calls.append((c,k)),sentinel)[1])
    assert phenome._run_future_arm(None,config,**kwargs) is sentinel
    assert calls==[(config,kwargs)]


def test_deferred_arm_preserves_original_files_and_never_runs_models(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();output=root/'phenome';output.mkdir()
    files=[output/'future_phenome_train_screening.parquet',output/'future_phenome_replicated_hits.parquet',
           output/'.checkpoints/future/part-000000.parquet']
    for path in files:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'preserved original result')
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    request=declaration(root)
    monkeypatch.setattr(phenome,'run_future_phenome_discovery',lambda *a,**k:pytest.fail('Future executed'))
    progress=[]
    paths=phenome._run_future_arm(request,object(),output_root=output,progress=progress.append)
    for key in ('train','validation','replicated'):
        assert pd.read_parquet(paths[key]).empty
        assert 'future_deferred' in paths[key].parts
    status=pd.read_parquet(paths['scope_status']).iloc[0]
    assert status.execution_status=='deferred_by_user' and not status.screen_completed and not status.inference_performed
    assert json.loads(paths['scope_declaration'].read_text())==request
    assert all((p.read_bytes(),p.stat().st_mtime_ns)==v for p,v in before.items())
    saved={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths.values()}
    phenome._run_future_arm(request,object(),output_root=output)
    assert all((p.read_bytes(),p.stat().st_mtime_ns)==v for p,v in saved.items())
    pd.DataFrame({'fake':[1]}).to_parquet(paths['train'])
    with pytest.raises(RuntimeError,match='compatibility artifact'):
        phenome._run_future_arm(request,object(),output_root=output)


@pytest.mark.parametrize('field,value',[('run_id','other'),('checkpoint_input_hash','stale'),
    ('authorization','unconfirmed'),('status','completed'),('arm','pathology')])
def test_stale_or_unapproved_scope_fails_closed(tmp_path,field,value):
    request=declaration(tmp_path);request[field]=value
    request['amendment_hash']=content_hash({k:v for k,v in request.items() if k!='amendment_hash'})
    (tmp_path/'future_arm_deferred.json').write_text(json.dumps(request),encoding='utf-8')
    with pytest.raises(RuntimeError,match='Invalid or stale'):
        phenome._future_defer_request(tmp_path,'input')


def test_full_scope_and_deferred_manifest_cannot_share_cache_identity(tmp_path):
    assert phenome._future_defer_request(tmp_path,'input') is None
    assert phenome._phenome_manifest_input_hash('input',None)=='input'
    request=declaration(tmp_path)
    path=tmp_path/'future_arm_deferred.json';path.write_text(json.dumps(request),encoding='utf-8')
    valid=phenome._future_defer_request(tmp_path,'input')
    assert valid==request and phenome._phenome_manifest_input_hash('input',valid)!='input'
    request['reason']='changed without hash update';path.write_text(json.dumps(request),encoding='utf-8')
    with pytest.raises(RuntimeError,match='Invalid or stale'):
        phenome._future_defer_request(tmp_path,'input')
