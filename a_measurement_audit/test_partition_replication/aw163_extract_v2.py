"""Technical continuation: JSON tuple canonicalization, fixed-coefficient sensitivity."""
import sys,json
from pathlib import Path
from . import aw163_extract as e
from .aw163_preflight import ROOT,OUT,sha
read_original=e.readsmall
save_original=e.a.save
adapter_original=e.a.adapter
def read(p):
    x=read_original(p)
    if isinstance(x,dict) and 'options' in x and 'sensitivity_vcg_methods' in x['options']:
        x['options']['sensitivity_vcg_methods']=tuple(x['options']['sensitivity_vcg_methods'])
    return x
def save(n,x):
    if isinstance(x,dict) and 'freeze_sha256' in x:
        x['continuation_binding_sha256']=sha(OUT/'continuation_binding_v2.json')
    save_original(n,x)
def adapter():
    import numpy as np
    fr,env=adapter_original();env['RUN']=OUT
    original=env['compute']
    def compute(package,row,names,method='kors',historical=False):
        result=original(package,row,names,method,historical)
        if method=='inverse_dower':
            scope={'np':np,'PLANES':{'frontal':(0,1,1,1),'transversal':(0,2,1,-1),'rechts_sagittal':(2,1,-1,1)},'release_import':lambda:fr,'prepare_record':lambda unused:package}
            e.a.ast_functions(ROOT/'werkzeuge/varianten/adapter.py',['context'],scope)
            e.a.ast_functions(ROOT/'werkzeuge/rv_a1/vorzeichen.py',['angle','turn','measure','variant'],scope)
            target=OUT/'M_sensitivity_private'/f'{int(row["study_id"])}.json'
            if target.exists():
                prior=read_original(target)
                assert prior['continuation_binding_sha256']==sha(OUT/'continuation_binding_v2.json')
            else:
                try:
                    segments,audit=scope['variant'](row)
                    value={'status':'ok','methods':{m:scope['measure'](v) for m,v in segments.items()},'audit':audit}
                except Exception as ex:value={'status':'technical_missing','error_private':str(ex),'exception_type':type(ex).__name__}
                value.update(continuation_binding_sha256=sha(OUT/'continuation_binding_v2.json'),freeze_sha256=sha(OUT/'extraction_freeze.json'))
                save_original(str(target.relative_to(OUT)),e.clean(value))
        return result
    env['compute']=compute
    return fr,env
def prepare():
    assert not (OUT/'W_records_private').exists() and not (OUT/'M_records_private').exists()
    for phase in ('W0','W1','W2'):
        save_original(f'original_status_{phase}.json',read_original(OUT/f'status_{phase}.json'))
    save_original('continuation_binding_v2.json',{'utc':e.a.utc(),'old_freeze_sha256':sha(OUT/'extraction_freeze.json'),'code':{str(p):sha(p) for p in [Path(__file__),ROOT/'werkzeuge/rv_a1/vorzeichen.py',ROOT/'werkzeuge/varianten/adapter.py']},'reason':'JSON empty list versus Python empty tuple; canonicalization only, no value access before failure','no_measurement_parameter_change':True,'sensitivity':'Original variant/measure/turn AST; same prepared package and fiducials, fixed original inverse-Dower origin/bounds'})
    e.public('Technical continuation before W signal access',f'Binding v2 `{sha(OUT/"continuation_binding_v2.json")}`; JSON option comparison canonicalized, no change to measurement or selection')
if __name__=='__main__':
    if sys.argv[1]=='prepare_v2':prepare()
    else:
        for p,h in read_original(OUT/'continuation_binding_v2.json')['code'].items():assert sha(p)==h
        e.readsmall=read;e.a.save=save;e.a.adapter=adapter;e.main()
