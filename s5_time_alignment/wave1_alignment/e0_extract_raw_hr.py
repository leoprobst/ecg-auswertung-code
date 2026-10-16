"""AW-242 E0: Herzfrequenz-Rohzeilen (itemid 220045) mit storetime aus der MIMIC-IV-3.1-Roh-CSV.

Nur Entwicklungspersonen (train/validation der A155-Featurecachepartitionen) mit mindestens einem ICU-Aufenthalt.
Private Ausgabe nur nach <local path> (Personenzeilen). Nachtraeglich (post hoc), keine Testpartition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, time, hashlib, threading
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
OUT = Path(str(data_path("s5.aw242")))
B = Path(str(data_path("s5.mimic_validation")))
CSV = Path(str(data_path("s5.mimic_iv_3_1_icu_chartevents_csv_gz")))
PEAK = 0

def guard():
    import psutil
    global PEAK
    vm = psutil.virtual_memory()
    PEAK = max(PEAK, psutil.Process().memory_info().rss)
    if vm.available < 1.5 * 2**30 or PEAK > 1.4 * 2**30:
        (OUT / 'e0_stop.json').write_text(json.dumps({'reason': 'Ressourcentor', 'available': vm.available, 'rss': PEAK}))
        os._exit(75)

def watch():
    while True:
        guard(); time.sleep(1)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def status(msg):
    (OUT / 'e0_status.json').write_text(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'msg': msg, 'peak_rss': PEAK}))

def extract(c, csv, out, people_table):
    """Filter itemid 220045 und Personenliste; strict_mode=false wegen nicht normgerechter Anfuehrungszeichen in der Roh-CSV."""
    q = f"""COPY (SELECT c.subject_id, c.hadm_id, c.stay_id, c.caregiver_id, c.charttime, c.storetime, c.valuenum, c.warning
      FROM read_csv('{Path(csv).as_posix()}', header=true, compression='gzip', strict_mode=false,
        columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT','caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP','itemid':'BIGINT','value':'VARCHAR','valuenum':'DOUBLE','valueuom':'VARCHAR','warning':'BIGINT'}}) c
      WHERE c.itemid = 220045 AND c.subject_id IN (SELECT subject_id FROM {people_table})) TO '{Path(out).as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    c.execute(q)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    guard(); threading.Thread(target=watch, daemon=True).start()
    import duckdb, pyarrow.parquet as pq
    t0 = time.time()
    c = duckdb.connect(str(OUT / 'e0_work.duckdb'))
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.execute(f"SET temp_directory='{(OUT / 'spill').as_posix()}'")
    c.execute('CREATE OR REPLACE TABLE whitelist(subject_id BIGINT, fold VARCHAR)')
    for fold in ['train', 'validation']:
        p = B / f'runs/A155-2026-10-04-d6e70e/analysis_feature_cache/cache-89d6aaf0a6b5957e/split={fold}/features.parquet'
        for batch in pq.ParquetFile(p).iter_batches(columns=['subject_id'], batch_size=16384, use_threads=False):
            c.register('batch', batch)
            c.execute(f"INSERT INTO whitelist SELECT DISTINCT subject_id, '{fold}' FROM batch")
    c.execute('CREATE OR REPLACE TABLE wl AS SELECT DISTINCT subject_id, fold FROM whitelist')
    assert c.execute('SELECT count(*) FROM (SELECT subject_id FROM wl GROUP BY 1 HAVING count(DISTINCT fold)>1)').fetchone()[0] == 0
    icu = (B / 'standardized/icustays.parquet').as_posix()
    c.execute(f"CREATE OR REPLACE TABLE icu_people AS SELECT DISTINCT w.subject_id FROM wl w JOIN read_parquet('{icu}') i USING(subject_id)")
    n_people = c.execute('SELECT count(*) FROM icu_people').fetchone()[0]
    status(f'whitelist fertig, ICU-Personen {n_people}')
    out = OUT / 'raw_hr.parquet'
    status('Lese Roh-CSV (einfaedig, gzip)')
    extract(c, CSV, out, 'icu_people')
    n = c.execute(f"SELECT count(*), count(DISTINCT subject_id) FROM read_parquet('{out.as_posix()}')").fetchone()
    c.close()
    res = {'rows': n[0], 'people': n[1], 'icu_people_whitelist': n_people, 'seconds': time.time() - t0, 'peak_rss': PEAK,
           'script_sha256': sha(__file__), 'csv_bytes': CSV.stat().st_size, 'out_sha256': sha(out), 'out_bytes': out.stat().st_size,
           'test_access': False}
    (OUT / 'e0_result.json').write_text(json.dumps(res, indent=1))
    status('fertig')
    print(json.dumps(res))
