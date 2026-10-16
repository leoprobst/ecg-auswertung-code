"""AW-023: ausschließlich Rauchtest (maximal 20), keine Vollrechnung.

K liefert ein JSON mit mapping_path/mapping_sha256, endpoint_id,
index_policy, reference_time_attestation und freigegebenen Verarbeitungshashes.
Personenbezogene Eingaben/Checkpoints ausschließlich im E:-Auswertungsbereich.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse,json,hashlib,time,threading
from pathlib import Path
from .audit import OUT,DEV,memory_guard,ROOT
from .vertrag import draw,validate_rows,development_file,ACTIVE

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def guarded_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise ValueError('Ausgabe außerhalb E:-Variantenbereich')
    return path

def checkpoint(path,data):
    """Atomarer Checkpoint; bestehende Datei nicht löschen."""
    path=guarded_output(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)

class BlockCheckpoint:
    """200er-Fortsetzungsvertrag; ein unvollständiger Block wird erneut gerechnet."""
    def __init__(self,path,freeze_hash):
        self.path=guarded_output(path);self.freeze_hash=freeze_hash
        self.completed=[];self.pending=[]
        if self.path.exists():
            state=json.loads(self.path.read_text(encoding='utf-8'))
            if state['freeze_hash']!=freeze_hash:raise ValueError('Blockcheckpoint-Freeze abweichend')
            self.completed=state['completed']
    def mark(self,identifier):
        if identifier in self.completed or identifier in self.pending:raise ValueError('Doppelter Blockeintrag')
        self.pending.append(identifier)
        if len(self.pending)==200:self.flush()
    def flush(self):
        self.completed.extend(self.pending);self.pending=[]
        checkpoint(self.path,{'freeze_hash':self.freeze_hash,'completed':self.completed,'block_size':200})

class Budget:
    def __init__(self):self.peak=0;self.over=False;self.done=threading.Event()
    def __enter__(self):
        import psutil
        self.process=psutil.Process()
        def monitor():
            while not self.done.is_set():
                info=self.process.memory_info();self.peak=max(self.peak,info.rss,getattr(info,'peak_wset',0))
                self.over=self.peak>1.5*1024**3
                if self.over:
                    # Nur den eigenen Laborprozess beenden, niemals Hauptlauf/Jobs.
                    (OUT/'AW023_SPEICHERSTOPP.json').write_text(json.dumps({'reason':'peak_rss_over_1_5_GiB','peak_rss_bytes':self.peak}),encoding='utf-8')
                    os._exit(75)
                self.done.wait(.05)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start();return self
    def check(self):
        memory_guard()
        if self.over:raise RuntimeError('RSS-Budget überschritten')
    def __exit__(self,*args):self.done.set();self.thread.join()

def inputs(attestation):
    mapping=guarded_output(attestation['mapping_path'])
    if attestation.get('endpoint_id')!='reduced_lvef_strict' or not attestation.get('index_policy') or not attestation.get('reference_time_attestation'):
        raise ValueError('Endpunkt-/Index-/Zeitattestation fehlt')
    if sha(mapping)!=attestation['mapping_sha256']:raise ValueError('Mappinghash abweichend')
    import pyarrow.parquet as pq
    schema=pq.ParquetFile(mapping).schema_arrow
    from .vertrag import REQUIRED
    optional=['machine_filtering','machine_bandwidth','machine_report']
    columns=list(REQUIRED)+[x for x in optional if x in schema.names]
    if set(REQUIRED)-set(schema.names):raise ValueError('Mapping unvollständig')
    frame=pq.read_table(mapping,columns=columns).to_pandas()
    validate_rows(frame)
    # Der Loaderpfad muss aus dem attestierten Export kommen, nie ID-Heuristik.
    waveform_root=Path(str(data_path("a.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_files"))).resolve()
    if not all(Path(p).resolve().is_relative_to(waveform_root) for p in frame.waveform_path):raise ValueError('Rohpfad außerhalb attestiertem WFDB-Root')
    return draw(frame)

def expected_values(sample,attestation):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from .adapter import ORIGINALS
    selected=[]
    for fold in ('train','validation'):
        subset=sample[sample.split==fold]
        wanted=list(map(int,subset.study_id))
        for file in sorted((DEV/f'split={fold}').glob('part-*.parquet')):
            file=development_file(file,fold);schema=pq.ParquetFile(file).schema_arrow
            cols=['subject_id','study_id','split','extraction_status',*ORIGINALS,'feature_confidence_json','feature_unevaluable_reason_json']
            if set(cols)-set(schema.names):continue
            table=ds.dataset(str(file),format='parquet').to_table(columns=cols,filter=ds.field('study_id').isin(wanted))
            if table.num_rows:
                if str(file) not in attestation.get('feature_file_sha256',{}):raise ValueError('Featuredateihash nicht attestiert')
                if sha(file)!=attestation['feature_file_sha256'][str(file)]:raise ValueError('Featurehash abweichend')
                selected.extend(table.to_pylist())
    if len(selected)!=len(sample):raise ValueError('Originalreferenzen fehlen/mehrdeutig')
    by_id={int(r['study_id']):r for r in selected}
    if len(by_id)!=len(sample):raise ValueError('Doppelte ECG-ID')
    for row in sample.to_dict('records'):
        ref=by_id[int(row['study_id'])]
        if ref['split']!=row['split'] or int(ref['subject_id'])!=int(row['subject_id']):raise ValueError('Split-/Personenidentität abweichend')
    return by_id

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attestation',required=True);args=parser.parse_args()
    memory_guard();attestation_path=guarded_output(args.attestation)
    attestation=json.loads(attestation_path.read_text(encoding='utf-8'))
    from .adapter import original_record,equivalent,prepare_record,prepared_originals,variants,OPTIONS,release_import,ORIGINALS
    # Erzeuger-/Optionenidentität ist ein Eingangstor, kein Resultat des Outputhashs.
    fr=release_import()
    if attestation.get('feature_record_sha256')!=sha(fr.__file__):raise ValueError('Verarbeitungscode nicht attestiert')
    if attestation.get('options')!=json.loads(json.dumps(OPTIONS)):raise ValueError('Optionsidentität nicht attestiert')
    sample=inputs(attestation)
    directory=OUT/'stichprobe';directory.mkdir(parents=True,exist_ok=True)
    import pyarrow as pa,pyarrow.parquet as pq
    target=directory/'AW023_liste.parquet'
    pq.write_table(pa.Table.from_pandas(sample[['subject_id','study_id','split','label']],preserve_index=False),target)
    sample_hash=sha(target)
    # 10 pro Fold, vorab aus identitätsbasiertem Ranking, ohne Variantenwahl.
    smoke=sample.groupby(['split','label'],sort=True,group_keys=False).head(5)
    if len(smoke)!=20:raise ValueError('Weniger als 20 geplante Entwicklungseingaben')
    expected=expected_values(smoke,attestation)
    state_path=OUT/'werte'/'AW023_checkpoint.json'
    state={'sample_hash':sample_hash,'attestation_hash':sha(attestation_path),'equivalent':[],'completed':[],'seconds':[],'block_size':200,'limit':20}
    if state_path.exists():
        prior=json.loads(state_path.read_text(encoding='utf-8'))
        if any(prior[k]!=state[k] for k in ('sample_hash','attestation_hash')):raise ValueError('Checkpoint gehört zu anderem Freeze')
        state=prior
    start=time.perf_counter()
    with Budget() as budget:
        # Alle Originale zuerst. Jede Abweichung verhindert sämtliche Varianten.
        for row in smoke.to_dict('records'):
            budget.check();identifier=int(row['study_id'])
            if identifier in state['equivalent']:continue
            actual=original_record(row)
            if actual['extraction_status']!=expected[identifier]['extraction_status']:raise RuntimeError('Extraktionsstatus abweichend')
            equivalent(actual,expected[identifier])
            if actual['extraction_status']!='ok':raise RuntimeError('Rauchtest-Eingabe nicht vollständig extrahierbar')
            package=prepare_record(row);local=prepared_originals(package)
            equivalent(local,expected[identifier]);del package
            original_rows=[{'subject_id':int(row['subject_id']),'study_id':identifier,'split':row['split'],'marker':m,'value':local.get(m)} for m in ORIGINALS]
            path=OUT/'werte'/'originale'/f'AW023_{identifier}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist(original_rows),path)
            state['equivalent'].append(identifier);checkpoint(state_path,state)
        for row in smoke.to_dict('records'):
            budget.check();identifier=int(row['study_id'])
            if identifier in state['completed']:continue
            started=time.perf_counter();package=prepare_record(row);values,qc=variants(package)
            if tuple(values)!=ACTIVE:raise RuntimeError('Variantenvertrag abweichend')
            rows=[{'subject_id':int(row['subject_id']),'study_id':identifier,'split':row['split'],'variant':key,**value,'fs':package['fs'],'qc_json':json.dumps(qc)} for key,value in values.items()]
            for result in rows:
                path=OUT/'werte'/result['variant']/f'AW023_{identifier}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.Table.from_pylist([result]),path)
            state['completed'].append(identifier);state['seconds'].append(time.perf_counter()-started)
            del package;checkpoint(state_path,state)  # Rauchtest strenger als 200er-Block.
        import numpy as np
        report={'n':len(state['completed']),'equivalent':len(state['equivalent']),'seconds_this_session':time.perf_counter()-start,'variant_seconds_mean':float(np.mean(state['seconds'])),'peak_rss_bytes':budget.peak,'full_run_authorized':False,'projection_4800_variant_seconds':float(np.mean(state['seconds'])*4800),'projection_5648_variant_seconds':float(np.mean(state['seconds'])*5648)}
        checkpoint(OUT/'AW023_RAUCHTEST.json',report)
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
