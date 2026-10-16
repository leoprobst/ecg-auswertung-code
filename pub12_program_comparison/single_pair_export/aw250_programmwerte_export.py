# -*- coding: utf-8 -*-
"""aw250_programmwerte_export.py (program comparison study, version 1.0): table of the program values per recording for PTB-XL and LUDB (open data), only ecg_id and program values.

Task: supplementary data as CSV, only PTB-XL and LUDB recordings, no EchoNext and no MIMIC rows, no metadata (no age, sex, diagnosis).
PTB-XL: population P0 (first recording per person, 18,869 recordings); per program (LoopLens, BRAVEHEART, 12SL, Uni-G, ECGDeli) QRS duration, QT, RR (BRAVEHEART: derived from the heart rate),
  delivered QTc and Fridericia QTc from QT and RR. Values come from the locked benchmark tables via the released loader aw245_basis.lade_werte (read only, hash check of the data lock beforehand).
LUDB: per recording (200) the median of the error of the NeuroKit2 boundary (dwt, V0) against the expert mark, QRS onset and QRS offset, in the four global definitions (aggregation over the leads:
  earliest/earliest, median/median, earliest/median, median/earliest), plus the number of beats. Functions from aw239_r6_auswertung (hash checked), nothing redefined.
Checks (assert): PTB-XL pair counts and the share 20.04 % (LoopLens against BRAVEHEART at 120 ms), LUDB pooled medians against r6_globale_zellen.csv.
Usage: python -I -B aw250_programmwerte_export.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
WZ = Path(str(data_path("pub12.werkzeuge_pub12")))
sys.path.insert(0, str(WZ / "aw245"))
sys.path.insert(0, str(WZ / "aw239"))
import aw245_basis as AB  # noqa: E402
import aw239_r6_auswertung as R6  # noqa: E402

HIER = Path(str(data_path("pub12.publikationen_pub_12_etablierung")))
OUT = HIER / "supplementary_data"
S = AB.S
PROGS = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]
SLUG = {"LoopLens": "looplens", "BRAVEHEART": "braveheart", "12SL": "ge12sl", "Uni-G": "unig", "ECGDeli": "ecgdeli"}


def ptbxl():
    pruef = AB.pruefe_sperre()
    ids = S.lese_ids(S.PFADE["listen"] / "P0_ecg_id.txt")
    assert len(ids) == 18869
    D = AB.lade_werte(ids)
    df = pd.DataFrame({"ecg_id": ids})
    for p in PROGS:
        d = D[p]
        s = SLUG[p]
        df[f"{s}_qrs_ms"] = d["qrs"]
        df[f"{s}_qt_ms"] = d["qt"]
        df[f"{s}_rr_ms"] = np.round(d["rr"], 1)
        df[f"{s}_qtc_delivered_ms"] = d["qtc_geliefert"]
        df[f"{s}_qtcf_from_qt_rr_ms"] = np.round(d["qtc_einh"], 1)
    # Checks
    nb = df[["braveheart_qrs_ms", "ge12sl_qrs_ms"]].notna().all(axis=1).sum()
    assert int(nb) == 18663, nb
    ll_bh = df[["looplens_qrs_ms", "braveheart_qrs_ms"]].dropna()
    assert len(ll_bh) == 18580, len(ll_bh)
    disc = ((ll_bh.looplens_qrs_ms >= 120) != (ll_bh.braveheart_qrs_ms >= 120)).mean() * 100
    assert round(disc, 2) == 20.04, disc
    return df


def ludb():
    V2 = R6.lade_v2()
    ids = sorted(int(x.split("/")[-1]) for x in (R6.LAUF.LUDB / "RECORDS").read_text().split())
    assert len(ids) == 200
    st = pickle.load(open(R6.IN / "nk_V0.pkl", "rb"))
    ref = {}
    for aid in ids:
        r = V2.lese_ludb_referenz(aid)
        r["beats"] = V2.beats_aus(r["qrs"], r["p"], r["t"], False)
        ref[aid] = r
    g = R6.globale_fehler(V2, st, ref, ids)
    # Check against the published pooled medians (V0)
    z = pd.read_csv(HIER / "ergebnisse" / "aw239" / "r6_globale_zellen.csv")
    z = z[z.variante == "V0"]
    ex = {}
    for (ta, ra, gr), d in g.groupby(["tool_agg", "ref_agg", "grenze"]):
        soll = z[(z.neurokit_aggregation == ta) & (z.referenz_aggregation == ra) & (z.grenze == gr)]
        assert len(soll) == 1, (ta, ra, gr)
        assert abs(float(np.median(d.fehler_ms)) - float(soll.median_ms.iloc[0])) < 0.51, (ta, ra, gr, float(np.median(d.fehler_ms)), float(soll.median_ms.iloc[0]))
        ex[(ta, ra, gr)] = len(d)
    rows = []
    for ai, aid in enumerate(ids):
        r = {"ludb_record": aid, "nk2_beats_all_leads": int(sum(len(a) for a in st.get(aid, {}).values()))}
        for (ta, ra, gr), name in ((("ext", "ext", "qrs_on"), "onset_error_earliest_vs_earliest_ms"), (("med", "med", "qrs_on"), "onset_error_median_vs_median_ms"),
                                   (("ext", "med", "qrs_on"), "onset_error_earliest_vs_median_ms"), (("med", "ext", "qrs_on"), "onset_error_median_vs_earliest_ms"),
                                   (("ext", "ext", "qrs_off"), "offset_error_earliest_vs_earliest_ms"), (("med", "med", "qrs_off"), "offset_error_median_vs_median_ms")):
            d = g[(g.ai == ai) & (g.tool_agg == ta) & (g.ref_agg == ra) & (g.grenze == gr)].fehler_ms
            r[name] = float(np.median(d)) if len(d) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = ptbxl()
    p1 = OUT / "SupplementaryData1_PTBXL_program_values.csv"
    a.to_csv(p1, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6g")
    b = ludb()
    p2 = OUT / "SupplementaryData2_LUDB_NeuroKit2_boundary_errors.csv"
    b.to_csv(p2, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6g")
    print("written:", p1, a.shape, "|", p2, b.shape)
    print(a.notna().sum().to_string())


if __name__ == "__main__":
    main()
