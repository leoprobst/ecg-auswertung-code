"""Final check: output hashes, full universes and no real identifiers in the project."""
import os,ast,re,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import pandas as pd,pyarrow.parquet as pq
from . import aw163_extract as e
from .aw163_preflight import ROOT,OUT,sha
def main():
    e.PHASE='delivery_verification';e.a.guard()
    assert e.readsmall(OUT/'status_report.json')['state']=='complete'
    hashes=e.readsmall(OUT/'output_hashes.json');assert all(sha(p)==h for p,h in hashes.items())
    for p in (ROOT/'werkzeuge/a_testreplikation').glob('aw163_*.py'):ast.parse(p.read_text(encoding='utf8'))
    e.public('Final check',f'Output binding `{sha(OUT/"output_hashes.json")}`; check full universes and absence of identifiers in own project deliverables')
    ids=e.a.readp(e.OLD/'identity_test_private.parquet',['subject_id','study_id'],test=True)
    forbidden={str(int(v)) for k in ids.columns for v in ids[k].unique()}
    p=OUT/'W_pairs_after_gates_private.parquet';cols=[k for k in pq.ParquetFile(p).schema_arrow.names if k.startswith(('hadm_id','ed_stay_id'))]
    if cols:
        f=e.a.readp(p,cols,test=True)
        forbidden|={str(int(v)) for k in cols for v in f[k].dropna().unique()}
    public=[ROOT/'nachrechnung/A_TESTREPLIKATION_AW163.md',ROOT/'nachrechnung/a_testreplikation.csv',ROOT/'koordination/abgaben/AW-163.md',*sorted((ROOT/'werkzeuge/a_testreplikation').glob('aw163_*'))]
    matches=[]
    for p in public:
        if not p.is_file():continue
        text=p.read_text(encoding='utf8')
        overlap=set(re.findall(r'(?<![\w.])\d{8}(?![\w.])',text))&forbidden
        if overlap:matches.append({'path':str(p),'count':len(overlap)})
    register=(ROOT/'koordination/TESTZUGRIFFE.md').read_text(encoding='utf8')
    own='\n'.join(line for line in register.splitlines() if '| AW-163 |' in line)
    if set(re.findall(r'(?<![\w.])\d{8}(?![\w.])',own))&forbidden:matches.append({'path':'own AW163 register lines','count':1})
    r=pd.read_csv(OUT/'repeat_marker.csv');m=pd.read_csv(OUT/'matrix_marker.csv');h=pd.read_csv(OUT/'matrix_half_marker.csv')
    assert r.groupby('population').marker_id.nunique().eq(2005).all()
    assert len(m)==2005 and m.marker_id.nunique()==2005
    assert h.groupby('half').marker_id.nunique().eq(2005).all()
    assert not r.duplicated(['population','marker_id']).any() and not h.duplicated(['half','marker_id']).any()
    result={'passed':not matches,'output_hashes_passed':True,'all_marker_tables_cover_2005_including_technical_exclusions':True,'own_project_files_checked':len(public),'own_register_lines_checked':len(own.splitlines()),'actual_identifier_matches':matches,'no_identifiers_emitted_to_project':not matches}
    e.a.save('delivery_verification.json',result)
    assert result['passed']
    e.a.save('status_verify.json',{'state':'complete'})
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
