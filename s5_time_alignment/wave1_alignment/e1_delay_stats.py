"""AW-242 E1: Paritaet der Roh-Extraktion gegen die standardisierte Referenz (AW-110) und Verzoegerungsverteilung storetime-charttime.

Ausgabe: Aggregate (JSON) und eine Stichprobe der Verzoegerungen (private .npy). Keine Kennungen in den Aggregaten.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import aw242_guard as G
G.start('e1')
import duckdb, numpy as np
OUT = G.OUT
c = duckdb.connect()
c.execute("SET memory_limit='700MB'; SET threads=2")
c.execute(f"ATTACH '{data_path('s5.uhrzeit_aw110_audit_duckdb')}' AS a (READ_ONLY)")
raw = (OUT / 'raw_hr.parquet').as_posix()
res = {}
# Paritaet: Roh-Zeilen der Primaerpersonen mit 30 <= HF <= 220 und Zeit
res['raw_rows_primary_30_220'] = c.execute(f"SELECT count(*) FROM read_parquet('{raw}') r JOIN a.primary_people p USING(subject_id) WHERE r.valuenum BETWEEN 30 AND 220 AND r.charttime IS NOT NULL").fetchone()[0]
res['standardized_hf_raw_rows'] = c.execute('SELECT count(*) FROM a.hf_raw').fetchone()[0]
res['raw_distinct_time_primary'] = c.execute(f"SELECT count(*) FROM (SELECT DISTINCT r.subject_id, r.stay_id, r.charttime FROM read_parquet('{raw}') r JOIN a.primary_people p USING(subject_id) WHERE r.valuenum BETWEEN 30 AND 220 AND r.charttime IS NOT NULL)").fetchone()[0]
res['standardized_hf_rows'] = c.execute('SELECT count(*) FROM a.hf').fetchone()[0]
# Wertegleichheit der Medianreferenz
res['value_mismatch_rows'] = c.execute(f"""SELECT count(*) FROM (SELECT r.subject_id, r.stay_id, r.charttime, median(r.valuenum) v FROM read_parquet('{raw}') r JOIN a.primary_people p USING(subject_id)
   WHERE r.valuenum BETWEEN 30 AND 220 AND r.charttime IS NOT NULL GROUP BY ALL) x
   FULL OUTER JOIN a.hf h ON h.subject_id=x.subject_id AND h.icustay_id=x.stay_id AND h.t=x.charttime WHERE x.v IS NULL OR h.hf IS NULL OR abs(x.v-h.hf)>1e-9""").fetchone()[0]
# Verzoegerung (Sekunden) fuer Primaerpersonen
q = f"""SELECT epoch(r.storetime)-epoch(r.charttime) d, minute(r.charttime) mi, second(r.charttime) se, r.warning w, r.valuenum v
        FROM read_parquet('{raw}') r JOIN a.primary_people p USING(subject_id) WHERE r.valuenum BETWEEN 30 AND 220 AND r.charttime IS NOT NULL"""
dl = c.execute(q).fetchnumpy()
d = dl['d'].astype(float)
res['delay'] = dict(n=int(len(d)), missing_store=int(np.isnan(d).sum()), negative=int((d < 0).sum()), zero=int((d == 0).sum()),
                    quantiles_min={str(p): float(np.nanquantile(d, p) / 60) for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
                    le5=float(np.nanmean(d <= 300)), le15=float(np.nanmean(d <= 900)), le60=float(np.nanmean(d <= 3600)), gt6h=float(np.nanmean(d > 6 * 3600)))
mi = dl['mi']
res['minute'] = dict(minute0=float(np.mean(mi == 0)), minute_nonzero=float(np.mean(mi != 0)), second0=float(np.mean(dl['se'] == 0)))
res['warning_share'] = float(np.mean(dl['w'] == 1))
# Verzoegerung getrennt nach Minute 0 / nicht 0
m0 = (mi == 0)
res['delay_by_minute0'] = {'minute0_median_min': float(np.nanmedian(d[m0]) / 60), 'minute_nonzero_median_min': float(np.nanmedian(d[~m0]) / 60),
                           'minute0_le15': float(np.nanmean(d[m0] <= 900)), 'minute_nonzero_le15': float(np.nanmean(d[~m0] <= 900))}
rng = np.random.default_rng(242)
samp = d[np.isfinite(d) & (d >= 0)]
samp = rng.choice(samp, size=min(500000, len(samp)), replace=False)
np.save(OUT / 'delay_sample_s.npy', samp)
(OUT / 'e1_delay_stats.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
