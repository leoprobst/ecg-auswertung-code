"""AW-255 S3 Machbarkeit: Laden von Labor und Besuchen (nur Train). Privat nach <local path>."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, time
sys.dont_write_bytecode = True
import duckdb
from pathlib import Path
DATA = Path(str(data_path("s3.mcmed_data_dir")))
W = Path(str(data_path("s3.work_dir"))); W.mkdir(exist_ok=True)
c = duckdb.connect(str(W / 's3.duckdb'))
c.execute("SET memory_limit='700MB'; SET threads=2")
c.execute(f"SET temp_directory='{(W / 'spill').as_posix()}'")
p = DATA / 'split_random_train.csv'
assert 'test' not in p.name
t0 = time.time()
c.execute(f"CREATE OR REPLACE TABLE sp AS SELECT column0 AS CSN FROM read_csv('{p.as_posix()}',header=false,columns={{'column0':'BIGINT'}})")
print('train', c.execute('select count(*) from sp').fetchone())
c.execute(f"""CREATE OR REPLACE TABLE visit AS SELECT v.CSN, v.MRN, v.Age, v.Gender, v.ED_dispo, v.ED_LOS,
  epoch_us(v.Arrival_time)/1e6 a, epoch_us(v.Roomed_time)/1e6 r, epoch_us(v.Departure_time)/1e6 d, v.Dx_ICD10, v.Triage_acuity
  FROM read_csv('{(DATA/'visits.csv').as_posix()}',header=true,columns={{'MRN':'BIGINT','CSN':'BIGINT','Visit_no':'INT','Visits':'INT','Age':'DOUBLE','Gender':'VARCHAR','Race':'VARCHAR','Ethnicity':'VARCHAR','Means_of_arrival':'VARCHAR','Triage_Temp':'DOUBLE','Triage_HR':'DOUBLE','Triage_RR':'DOUBLE','Triage_SpO2':'DOUBLE','Triage_SBP':'DOUBLE','Triage_DBP':'DOUBLE','Triage_acuity':'VARCHAR','CC':'VARCHAR','ED_dispo':'VARCHAR','Hours_to_next_visit':'DOUBLE','Dispo_class_next_visit':'VARCHAR','ED_LOS':'DOUBLE','Hosp_LOS':'DOUBLE','DC_dispo':'VARCHAR','Payor_class':'VARCHAR','Admit_service':'VARCHAR','Dx_ICD9':'VARCHAR','Dx_ICD10':'VARCHAR','Dx_name':'VARCHAR','Arrival_time':'TIMESTAMP','Roomed_time':'TIMESTAMP','Dispo_time':'TIMESTAMP','Admit_time':'TIMESTAMP','Departure_time':'TIMESTAMP'}}) v
  JOIN sp USING(CSN)""")
print('visit', c.execute('select count(*), count(distinct MRN) from visit').fetchone(), round(time.time()-t0))
c.execute(f"""CREATE OR REPLACE TABLE labs AS SELECT l.CSN, epoch_us(l.Order_time)/1e6 t_order, epoch_us(l.Result_time)/1e6 t_result,
  l.Display_name, l.Component_name cname, l.Component_value cval, l.Component_result cres, l.Component_units unit, l.Component_abnormal abn
  FROM read_csv('{(DATA/'labs.csv').as_posix()}',header=true,all_varchar=true,strict_mode=false,null_padding=true,ignore_errors=true) l
  JOIN sp USING(CSN)""".replace("l.Order_time","try_cast(l.Order_time as timestamp)").replace("l.Result_time","try_cast(l.Result_time as timestamp)"))
print('labs', c.execute('select count(*), count(distinct CSN) from labs').fetchone(), round(time.time()-t0))
c.execute(f"""CREATE OR REPLACE TABLE orders AS SELECT o.CSN, epoch_us(try_cast(o.Order_time as timestamp))/1e6 t_order, o.Order_type, o.Procedure_name pname,
  epoch_us(try_cast(o.First_admin_time as timestamp))/1e6 t_admin, epoch_us(try_cast(o.Result_time as timestamp))/1e6 t_result
  FROM read_csv('{(DATA/'orders.csv').as_posix()}',header=true,all_varchar=true,strict_mode=false,null_padding=true,ignore_errors=true) o
  JOIN sp USING(CSN)""")
print('orders', c.execute('select count(*), count(distinct CSN) from orders').fetchone(), round(time.time()-t0))
