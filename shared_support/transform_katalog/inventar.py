import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
from werkzeuge.varianten.audit import memory_guard,ROOT,DEV
from werkzeuge.varianten.lauf import sha
from werkzeuge.varianten.adapter import release_import
from werkzeuge.varianten.vertrag import development_file
PROJECT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("a.transform_katalog")))
SAMPLE=Path(str(data_path("a.varianten_aw023b_stichprobe_liste_parquet")))
CONTEXT=Path(str(data_path("a.varianten_aw023c_stichprobe_kontext_parquet")))
FREEZE=Path(str(data_path("a.varianten_aw023b_stichprobenmanifest_json")))
def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def main():
    mem=memory_guard()
    import pandas as pd,pyarrow.parquet as pq
    fr=release_import()
    bind=json.loads(Path(str(data_path("a.varianten_aw023c_kontextbericht_json"))).read_text(encoding='utf8'))
    assert sha(SAMPLE)==bind['sample_sha256'] and sha(CONTEXT)==bind['context_sha256']
    def read(p,cols):
        pq.ParquetFile(p);return pq.read_table(p,columns=cols).to_pandas()
    OUT.mkdir(parents=True,exist_ok=True)
    sample=read(SAMPLE,['subject_id','study_id','split'])
    ctx=read(CONTEXT,['subject_id','study_id','split','waveform_path'])
    assert len(sample)==5648 and not sample.subject_id.duplicated().any() and sample.split.isin(['train','validation']).all()
    from werkzeuge.varianten.aw023b import resolve_waveform,MAPPING,MANIFEST
    assert sha(MAPPING)==json.loads(MANIFEST.read_text(encoding='utf8'))['sha256']
    import pyarrow.dataset as ds
    pq.ParquetFile(MAPPING)
    mapping=ds.dataset(str(MAPPING),format='parquet').to_table(columns=['subject_id','study_id','split','relative_waveform_path'],filter=ds.field('study_id').isin(sample.study_id.tolist())).to_pandas()
    mapping=mapping.drop_duplicates()
    joined=sample.merge(ctx,on=['subject_id','study_id','split'],validate='one_to_one').merge(mapping,on=['subject_id','study_id','split'],validate='one_to_one');assert len(joined)==5648
    assert all(Path(a).resolve()==Path(resolve_waveform(b)).resolve() for a,b in zip(joined.waveform_path,joined.relative_waveform_path))
    p=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e_redundancy_marker_redundancy_parquet")))
    names=sorted(read(p,['feature_id']).feature_id.unique())
    schemas={};sources=[]
    freeze=json.loads(FREEZE.read_text(encoding='utf8'))
    for item in freeze['feature_sources']:
        p=Path(item['path']);fold=p.parent.name.split('=')[1];development_file(p,fold)
        sc=pq.ParquetFile(p).schema_arrow
        digest=hashlib.sha256(str(sc).encode()).hexdigest();schemas[digest]=sc.names
        sources.append({'path':str(p),'schema_sha256':digest,'size':p.stat().st_size})
    available=set(fr.BiomarkerRegistry.available());definitions={d.name:d for d in fr.BiomarkerRegistry.definitions()}
    families=pd.read_csv(PROJECT/'schema/markerfamilien.csv',sep=';').set_index('marker_id').familie.to_dict()
    twins=pd.read_csv(str(data_path("a.aw056_transform_final_csv"))).marker_id.tolist()
    records=[]
    for n in names:
        d=definitions.get(n)
        records.append({'marker_id':n,'familie':families.get(n,'nicht_zugeordnet'),'registry_family':d.family if d else '', 'unit':d.unit if d else '', 'implemented':n in available,'in_definition':d is not None,'in_all_reference_schemas':all(n in x for x in schemas.values()),'aw056_twin':n in twins,'hauptarbeit_B':any(x in n for x in ('frechet','mirror','dtw','qrs_t_angle')),'status':'tor_ausstehend'})
    pd.DataFrame(records).to_csv(OUT/'inventar.csv',index=False)
    report={'memory_start':mem,'sample_n':len(sample),'folds':sample.split.value_counts().to_dict(),'markers':len(names),'implemented':sum(r['implemented'] for r in records),'nonregistry':[r['marker_id'] for r in records if not r['implemented']], 'missing_reference':[r['marker_id'] for r in records if not r['in_all_reference_schemas']], 'twins_n':len(twins),'sources':sources,'schemas':schemas,'binding':{'sample':sha(SAMPLE),'context':sha(CONTEXT),'freeze':sha(FREEZE),'specification':sha(PROJECT/'nachrechnung/SPEZIFIKATION_AW067.md'),'release':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'vcg_biomarker').rglob('*.py')}},'test_access':False}
    save('inventar.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','schemas','binding')},ensure_ascii=False))
if __name__=='__main__':main()
