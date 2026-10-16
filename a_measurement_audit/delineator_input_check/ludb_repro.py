"""Reproduction check of the Kors variant on LUDB against the stored fused_global errors of the earlier beat-boundary run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
from . import auswerten as A

STORED = Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_beat_boundaries_20261012_v1")))
RECORDS = json.loads((A.OUT / 'praereg' / 'liste_ludb.json').read_text())


def main():
    recs = A.load('ludb_ergebnisse.jsonl', 'ludb')
    byrec = {r['record']: r for r in recs}
    tot = same = missing_mine = missing_stored = diff = 0
    pairs_equal = pairs_total = 0
    marker_rows = []
    for k, rec_id in enumerate(RECORDS, 1):
        r = byrec.get(rec_id)
        if r is None or r['status'] != 'ok':
            continue
        st = pd.read_parquet(STORED / f'errors-{k:03d}.parquet')
        st = st[(st.scope == 'fused_global') & (st.source == 'fused') & st.landmark.isin(A.POINTS_LUDB)] if hasattr(A, 'POINTS_LUDB') else st[(st.scope == 'fused_global') & (st.source == 'fused') & st.landmark.isin(['p_onset', 'p_offset', 'r_onset', 'r_offset', 't_offset'])]
        mine, matched, n_pairs = A.ludb_errors(r, 'kors')
        stored_pairs = set(st.reference_beat.unique())
        pairs_total += len(stored_pairs); pairs_equal += len(stored_pairs & set(matched))
        name_map = {'r_onset': 'qrs_onset', 'r_offset': 'qrs_offset'}
        for _, row in st.iterrows():
            nm = name_map.get(row.landmark, row.landmark)
            t = mine.get((int(row.reference_beat), nm))
            tot += 1
            if t is None or t[2] is None:
                if pd.isna(row.prediction_sample):
                    same += 1
                else:
                    missing_mine += 1
                continue
            if pd.isna(row.prediction_sample):
                missing_stored += 1
            elif int(row.prediction_sample) == int(t[2]):
                same += 1
            else:
                diff += 1
    out = {'zeilen_gespeichert': tot, 'identisch': same, 'nur_im_gespeicherten_vorhanden': missing_mine, 'nur_in_neuer_vorhanden': missing_stored, 'wert_abweichend': diff,
           'anteil_identisch_pct': 100. * same / tot, 'zuordnung_beats_gespeichert': pairs_total, 'zuordnung_beats_gleich': pairs_equal}
    (A.RES / 'ludb_repro.json').write_text(json.dumps(out, indent=2), encoding='utf8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
