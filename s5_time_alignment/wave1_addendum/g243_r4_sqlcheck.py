"""AW-243: zweiter Rechenweg fuer die Risikopersonen und das Kontaktende (reines SQL, DuckDB), gegen g243_r4_lauf.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
import duckdb

PP = str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_documented_af_landmark_20261012_v2_first_eligible_landmarks_parquet"))
TL = str(data_path("s5.mimic_validation_evidence_patient_clinical_timeline_parquet"))
c = duckdb.connect(); c.execute("SET memory_limit='600MB'; SET threads=2")
res = {}
for mode in ('archiv', 'exakt'):
    fu = "(l.followup_hours * 3600.0)" if mode == 'archiv' else "(epoch(t.last_observed_time) - epoch(l.ecg_time))"
    q = f"""
    WITH b AS (
      SELECT l.split, round(l.next_af_event_delta_hours * 3600.0) AS fut,
             {fu} AS fu,
             epoch(l.dischtime) - epoch(l.ecg_time) AS lm,
             CASE WHEN l.date_of_death IS NULL THEN NULL ELSE epoch(l.date_of_death) - epoch(l.ecg_time) END AS dth
      FROM read_parquet('{PP}') l JOIN read_parquet('{TL}') t USING (subject_id)),
    g AS (
      SELECT *, LEAST(fu - lm, COALESCE(dth - lm, 1e18)) AS cens,
             (fut - lm) AS evt
      FROM b
      WHERE (fut IS NULL OR (fut <= fu AND (dth IS NULL OR fut < dth))) AND fu > lm AND (dth IS NULL OR dth > lm)),
    h AS (
      SELECT *, COALESCE(evt > 0 AND evt <= cens AND evt <= 365*86400, FALSE) AS ev FROM g)
    SELECT split, count(*) AS n, count(*) FILTER (WHERE ev) AS af,
           count(*) FILTER (WHERE NOT ev AND (fu - lm) < COALESCE(dth - lm, 1e18) AND (fu - lm) < 365*86400) AS contact_end,
           count(*) FILTER (WHERE NOT ev AND COALESCE(dth - lm, 1e18) < (fu - lm) AND COALESCE(dth - lm, 1e18) < 365*86400) AS death_end,
           count(*) FILTER (WHERE NOT ev AND 365*86400 < (fu - lm) AND 365*86400 < COALESCE(dth - lm, 1e18)) AS horizon_end
    FROM h GROUP BY split ORDER BY split"""
    res[mode] = c.execute(q).df().to_dict(orient='records')
print(json.dumps(res, indent=1))
