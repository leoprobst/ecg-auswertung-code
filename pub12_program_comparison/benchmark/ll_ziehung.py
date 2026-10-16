"""Draw of PT (seed 14102028) and S_G (seed 14102026) according to section 4.2 of the pre-registered analysis protocol.

Metadata columns only (ecg_id, patient_id, scp_codes, pacemaker, strat_fold). No results read.
Output: <local path> (lists with hash). Call: python -I ll_ziehung.py
Rules (protocol 4.2): P0 = per person the recording with the lowest ecg_id. Strata by priority S1..S5
(count check against the protocol: 225 / 1529 / 1011 / 8565 / 7539). Draw per stratum from the ascending sorted
ecg_id with one default_rng(14102026), in sequence S1..S5, rng.choice(replace=False), sorted ascending at the end.
PT: 40 from the non-first recordings outside fold 10 (pool size checked against a local reference value), sorted ascending, default_rng(14102028).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import check_reference
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

META = Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv")))
OUT = Path(str(data_path("pub12.aw_219_listen")))
SEED_SG, SEED_PT = 14102026, 14102028
SOLL_STRATA = {"S1": 225, "S2": 1529, "S3": 1011, "S4": 8565, "S5": 7539}
SOLL_N = {"S1": 150, "S2": 200, "S3": 100, "S4": 275, "S5": 275}


def stratum(codes, pacemaker):
    pm = ("PACE" in codes) or (isinstance(pacemaker, str) and pacemaker.strip() != "")
    if pm:
        return "S1"
    if codes & {"CLBBB", "CRBBB", "IVCD"}:
        return "S2"
    if codes & {"AFIB", "AFLT"}:
        return "S3"
    if "NORM" in codes:
        return "S4"
    return "S5"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(META, usecols=["ecg_id", "patient_id", "scp_codes", "pacemaker", "strat_fold"]).sort_values("ecg_id")
    d["codes"] = d.scp_codes.map(lambda s: set(ast.literal_eval(s).keys()))
    erst = ~d.duplicated("patient_id", keep="first")
    p0 = d[erst].copy()
    p0["stratum"] = [stratum(c, p) for c, p in zip(p0.codes, p0.pacemaker)]
    zaehl = p0.stratum.value_counts().sort_index().to_dict()
    assert zaehl == SOLL_STRATA, zaehl
    rng = np.random.default_rng(SEED_SG)
    gezogen, je = [], {}
    for s in ("S1", "S2", "S3", "S4", "S5"):
        ids = np.sort(p0.loc[p0.stratum == s, "ecg_id"].to_numpy())
        k = min(SOLL_N[s], len(ids))
        pick = rng.choice(ids, size=k, replace=False)
        je[s] = int(k)
        gezogen.extend(int(x) for x in pick)
    sg = sorted(gezogen)
    assert len(sg) == len(set(sg)) == 1000
    pt_pool = d[(~erst) & (d.strat_fold != 10)]
    check_reference("ll_ziehung.pt_pool_n", len(pt_pool), what="size of the PT pool")
    rng_pt = np.random.default_rng(SEED_PT)
    pt = sorted(int(x) for x in rng_pt.choice(np.sort(pt_pool.ecg_id.to_numpy()), size=40, replace=False))
    paths = {}
    for name, lst in (("S_G", sg), ("PT", pt), ("P0", sorted(int(x) for x in p0.ecg_id)),
                      ("nicht_erst", sorted(int(x) for x in d[~erst].ecg_id))):
        p = OUT / f"{name}_ecg_id.txt"
        p.write_text("\n".join(str(x) for x in lst) + "\n", encoding="ascii")
        paths[name] = {"datei": str(p), "n": len(lst), "sha256": sha(p)}
    # stratum assignment of S_G (only ecg_id and stratum) for the weighting later
    zuord = OUT / "S_G_stratum.csv"
    s_map = p0.set_index("ecg_id").stratum
    zuord.write_text("ecg_id,stratum\n" + "\n".join(f"{i},{s_map[i]}" for i in sg) + "\n", encoding="ascii")
    paths["S_G_stratum"] = {"datei": str(zuord), "n": len(sg), "sha256": sha(zuord)}
    man = {"auftrag": "AW-219", "seed_S_G": SEED_SG, "seed_PT": SEED_PT, "stratum_groesse_P0": zaehl, "gezogen_je_stratum": je,
           "pt_pool": len(pt_pool), "dateien": paths, "skript": Path(__file__).name,
           "skript_sha256": sha(__file__), "hinweis": "Draw via rng.choice(replace=False), per stratum ascending sorted ecg_id, one generator for S1..S5"}
    (OUT / "ziehung_manifest.json").write_text(json.dumps(man, indent=2), encoding="ascii")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
