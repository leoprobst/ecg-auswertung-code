# -*- coding: utf-8 -*-
"""aw245_r9.py (version 1.0): repeatability per program on PTB-XL pairs and classification of the discordance between programs (program comparison study, exploratory, post hoc).

Population: pairs of the same person from the bound pair list of the preceding pair-list stage (aw223_paarliste_stunden.csv, 2,930 pairs, hash checked against the bound full value). Primary: interval up to 24 h without PT pairs
(recordings from the LoopLens pilot set); sensitivities: all pairs up to 24 h, up to 2 h, up to 168 h, one pairing per person.
Values: locked benchmark long formats (LoopLens, BRAVEHEART; read only, hash check of the data lock beforehand) and PTB-XL+ archive columns (12SL, Uni-G, ECGDeli), per recording.
Per program: discordance between the two recordings of the same person at QRS 120/130/150 and QTc 450/470/500 (uniform Fridericia from own QT and RR, and delivered
standard QTc), class change, bias, SD and RC of the difference, ICC(2,1), median and P95 of the absolute difference. Comparison on the same recordings (common pair set, all five
programs valid at both recordings): discordance between each two programs (recording level, both recordings stacked) against the repeat discordance of each program.
Repeat discordance contains true change of state (upper bound of the pure measurement scatter). Intervals: person-cluster bootstrap, 2,000 draws, fixed seed, common draws.
Usage: python -I -B aw245_r9.py --modus lauf [--ohne-paritaet]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aw245_basis as AB  # noqa: E402
import aw245_stat as T  # noqa: E402

VERSION = "1.0"
SEED = 15102451
B = 2000
MASSE = {"qrs": ("QRS duration", T.QRS_SCHW), "qtc_einh": ("QTc uniform Fridericia", T.QTC_SCHW), "qtc_geliefert": ("QTc delivered (standard formula of the program)", T.QTC_SCHW)}
WZ = AB.WERKZEUGE
KONTROLLEN = []


def kontrolle(name, soll, ist, ok):
    KONTROLLEN.append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok)))
    AB.log("CHECK %s: expected %s actual %s -> %s" % (name, soll, ist, "ok" if ok else "DEVIATION"))


def lauf(paritaet=True):
    t0 = time.time()
    AB.OUT_ROH.mkdir(parents=True, exist_ok=True)
    AB.OUT_PUB.mkdir(parents=True, exist_ok=True)
    AB.S.STATE["log"] = str(AB.OUT_ROH / "r9_lauf.log")
    AB.log("aw245_r9 version %s" % VERSION)
    pruef = AB.pruefe_sperre()
    pl_hash = AB.pruefe_paarliste()
    pairs = pd.read_csv(AB.PAARLISTE)
    pairs["pt_paar"] = pairs["pt_paar"].astype(str).str.lower().isin(["true", "1"])
    kontrolle("Pairs in total", 2930, len(pairs), len(pairs) == 2930)
    kontrolle("Pairs within 24 h / persons", "327 / 268", "%d / %d" % ((pairs.abstand_h <= 24).sum(), pairs[pairs.abstand_h <= 24].patient_id.nunique()),
              (pairs.abstand_h <= 24).sum() == 327 and pairs[pairs.abstand_h <= 24].patient_id.nunique() == 268)
    kontrolle("Pairs within 168 h / persons", "1217 / 891", "%d / %d" % ((pairs.abstand_h <= 168).sum(), pairs[pairs.abstand_h <= 168].patient_id.nunique()),
              (pairs.abstand_h <= 168).sum() == 1217 and pairs[pairs.abstand_h <= 168].patient_id.nunique() == 891)
    kontrolle("Pairs within 2 h / persons", "109 / 80", "%d / %d" % ((pairs.abstand_h <= 2).sum(), pairs[pairs.abstand_h <= 2].patient_id.nunique()),
              (pairs.abstand_h <= 2).sum() == 109 and pairs[pairs.abstand_h <= 2].patient_id.nunique() == 80)
    prim = (pairs.abstand_h <= 24) & ~pairs.pt_paar
    reference_prim = reference("aw245_r9.prim_pairs_qrs_ok_ll")
    if reference_prim is None:
        print("NOTE: no local reference value 'aw245_r9.prim_pairs_qrs_ok_ll'; check skipped.")
    kontrolle("Primary population without PT (pairs within 24 h)", "local reference value: primary pairs with LoopLens QRS at both recordings", "%d without this condition; with: %d" % (prim.sum(), (prim & pairs.qrs_ok_ll).sum()),
              reference_prim is None or (prim & pairs.qrs_ok_ll).sum() == reference_prim)
    # values
    ids = np.array(sorted(set(pairs.ecg_id_1.astype(int)) | set(pairs.ecg_id_2.astype(int))), dtype=np.int64)
    pos = {int(e): i for i, e in enumerate(ids)}
    Dv = AB.lade_werte(ids)
    i1 = np.array([pos[int(e)] for e in pairs.ecg_id_1])
    i2 = np.array([pos[int(e)] for e in pairs.ecg_id_2])
    pid = pairs.patient_id.to_numpy()
    # Population 1P: first pairing per person (smallest ecg_id_1) within the primary population
    p1 = np.zeros(len(pairs), bool)
    sub = pairs[prim].sort_values(["patient_id", "ecg_id_1", "ecg_id_2"])
    p1[sub.groupby("patient_id").head(1).index.to_numpy()] = True
    POP = {
        "PRIM_24h_ohnePT": prim.to_numpy(),
        "ALLE_24h": (pairs.abstand_h <= 24).to_numpy(),
        "PRIM_2h_ohnePT": ((pairs.abstand_h <= 2) & ~pairs.pt_paar).to_numpy(),
        "PRIM_168h_ohnePT": ((pairs.abstand_h <= 168) & ~pairs.pt_paar).to_numpy(),
        "PRIM_24h_eine_Paarung_je_Person": p1,
    }
    for k, v in POP.items():
        AB.log("Population %s: %d pairs, %d persons" % (k, v.sum(), pd.Series(pid[v]).nunique()))

    rows_re, rows_zw, rows_vgl = [], [], []

    def zeilen(d, base, keys):
        for k in keys:
            if k in d:
                est, lo, hi = T.ci(d[k])
                rows_base = dict(base)
                rows_base.update(kennzahl=k, schaetzer=est, ki_unten=lo, ki_oben=hi)
                yield rows_base

    for pname, sel in POP.items():
        idx = np.flatnonzero(sel)
        W = T.person_gewichte(pid[idx], B, SEED + list(POP).index(pname))
        n_pers = pd.Series(pid[idx]).nunique()
        for m, (mname, schw) in MASSE.items():
            x1 = {t: Dv[t][m][i1[idx]] for t in WZ}
            x2 = {t: Dv[t][m][i2[idx]] for t in WZ}
            gemeinsam = np.all([np.isfinite(x1[t]) & np.isfinite(x2[t]) for t in WZ], axis=0)
            kontrolle("%s %s common pair set" % (pname, m), "count", "%d of %d pairs" % (gemeinsam.sum(), len(idx)), True)
            # repeatability per program (all available pairs) and on the common pair set
            for variante, mask in (("verfuegbar", np.ones(len(idx), bool)), ("gemeinsame_Paare", gemeinsam)):
                if variante == "gemeinsame_Paare" and pname not in ("PRIM_24h_ohnePT", "PRIM_168h_ohnePT", "ALLE_24h", "PRIM_2h_ohnePT"):
                    continue
                for t in WZ:
                    a, b = x1[t].copy(), x2[t].copy()
                    a[~mask] = np.nan
                    kz = T.paar_kennzahlen(a, b, W, schw)
                    n_pairs = int(kz["n"][0])
                    ok = np.isfinite(a) & np.isfinite(b)
                    base = dict(population=pname, variante=variante, mass=m, programm=t, paare=n_pairs, personen=int(pd.Series(pid[idx][ok]).nunique()))
                    rows_re.extend(zeilen(kz, base, ["bias", "sd", "rc", "icc", "ccc", "mittel_abs", "median_abs", "p95_abs"] + ["D%d" % s for s in schw] + ["wechsel", "hoeher"]))
            # between programs at recording level
            for variante in ("gemeinsame_Paare", "verfuegbar"):
                if variante == "verfuegbar" and pname != "PRIM_24h_ohnePT":
                    continue
                mask = gemeinsam if variante == "gemeinsame_Paare" else np.ones(len(idx), bool)
                W2 = np.hstack([W, W])
                for ia in range(len(WZ)):
                    for ib in range(ia + 1, len(WZ)):
                        ta, tb = AB.S.orientiere(WZ[ia], WZ[ib])  # ta candidate, tb comparator
                        a = np.concatenate([x1[tb], x2[tb]])
                        b = np.concatenate([x1[ta], x2[ta]])
                        mm = np.concatenate([mask, mask])
                        a = np.where(mm, a, np.nan)
                        b = np.where(mm, b, np.nan)
                        kz = T.paar_kennzahlen(a, b, W2, schw, voll=True)
                        ok = np.isfinite(a) & np.isfinite(b)
                        pids2 = np.concatenate([pid[idx], pid[idx]])
                        base = dict(population=pname, variante=variante, mass=m, kandidat=ta, komparator=tb, aufnahmen=int(kz["n"][0]), personen=int(pd.Series(pids2[ok]).nunique()))
                        rows_zw.extend(zeilen(kz, base, ["bias", "sd", "rc", "icc", "ccc", "mittel_abs", "median_abs", "p95_abs"] + ["D%d" % s for s in schw] + ["wechsel", "hoeher"]))
                        if variante == "gemeinsame_Paare":
                            zw_cache = kz
                            # comparison of repeat (program ta and tb) against between-program discordance, common draws
                            for tw in (ta, tb):
                                a1, b1 = x1[tw].copy(), x2[tw].copy()
                                a1[~mask] = np.nan
                                b1[~mask] = np.nan
                                kre = T.paar_kennzahlen(a1, b1, W, schw)
                                for s in schw:
                                    dre, dzw = kre["D%d" % s], kz["D%d" % s]
                                    diff = dre - dzw
                                    est, lo, hi = T.ci(diff)
                                    rows_vgl.append(dict(population=pname, mass=m, schwelle=s, programm_wiederholung=tw, paar_zwischen="%s gegen %s" % (ta, tb), paare=int(kre["n"][0]),
                                                         D_wiederholung=float(dre[0]), D_zwischen=float(dzw[0]), differenz_wiederholung_minus_zwischen=est, ki_unten=lo, ki_oben=hi,
                                                         sd_wiederholung=float(kre["sd"][0]), sd_zwischen=float(kz["sd"][0]),
                                                         verhaeltnis_sd=float(kre["sd"][0] / kz["sd"][0]) if kz["sd"][0] > 0 else np.nan,
                                                         wechsel_wiederholung=float(kre["wechsel"][0]), wechsel_zwischen=float(kz["wechsel"][0])))
    df_re = pd.DataFrame(rows_re)
    df_zw = pd.DataFrame(rows_zw)
    df_vgl = pd.DataFrame(rows_vgl)

    # Context: between-program discordance in P0 (one recording per person), without clustering (independent)
    meta = AB.lade_meta()
    erst = meta.reset_index().sort_values("ecg_id").drop_duplicates("patient_id", keep="first")
    ids_p0 = erst.ecg_id.to_numpy(np.int64)
    D0 = AB.lade_werte(ids_p0)
    rows_p0 = []
    for m, (mname, schw) in MASSE.items():
        for ia in range(len(WZ)):
            for ib in range(ia + 1, len(WZ)):
                ta, tb = AB.S.orientiere(WZ[ia], WZ[ib])
                a, b = D0[tb][m], D0[ta][m]
                ok = np.isfinite(a) & np.isfinite(b)
                for s in schw:
                    k = int(((a[ok] >= s) != (b[ok] >= s)).sum())
                    lo, hi = _wilson(k, int(ok.sum()))
                    rows_p0.append(dict(mass=m, schwelle=s, kandidat=ta, komparator=tb, n=int(ok.sum()), diskordant=k, D=k / ok.sum(), ki_unten=lo, ki_oben=hi))
    df_p0 = pd.DataFrame(rows_p0)

    # Parity of the loader against the locked computation base (schwellen_aw228.lade_alles)
    par = {}
    if paritaet:
        AB.log("Parity: lade_alles() of the computation base")
        ids_ref, Dref = AB.S.lade_alles()
        gleich_ids = bool(np.array_equal(ids_ref, ids_p0))
        kontrolle("P0 identifiers equal to the computation base", "equal", "equal" if gleich_ids else "different", gleich_ids)
        for t in WZ:
            for key in ("qrs", "qt", "rr", "qtc_nativ", "qtc_einh"):
                x, y = D0[t][key], Dref[t][key]
                g = bool(np.array_equal(np.isfinite(x), np.isfinite(y)) and np.allclose(x[np.isfinite(x)], y[np.isfinite(y)], rtol=0, atol=1e-9))
                par["%s.%s" % (t, key)] = g
        kontrolle("Loader equal to the computation base (all programs and quantities)", "all equal", "%d of %d" % (sum(par.values()), len(par)), all(par.values()))
        del ids_ref, Dref

    # writing
    def schreibe(df, name):
        df.to_csv(AB.OUT_PUB / name, index=False, encoding="utf-8", float_format="%.6g")
        df.to_csv(AB.OUT_ROH / name, index=False, encoding="utf-8", float_format="%.6g")
    schreibe(df_re, "r9_wiederholbarkeit.csv")
    schreibe(df_zw, "r9_zwischen_programmen.csv")
    schreibe(df_vgl, "r9_vergleich_wiederholung_zwischen.csv")
    schreibe(df_p0, "r9_kontext_p0_zwischen_programmen.csv")
    schreibe(pd.DataFrame(KONTROLLEN), "r9_kontrollen.csv")
    schreibe(pd.DataFrame(pruef), "r9_hashpruefung.csv")
    info = dict(version=VERSION, seed_basis=SEED, B=B, paarliste_sha256=pl_hash, populationen={k: int(v.sum()) for k, v in POP.items()}, laufzeit_s=round(time.time() - t0, 1),
                paritaet=par, kontrollen_bestanden=int(sum(k["bestanden"] for k in KONTROLLEN)), kontrollen_gesamt=len(KONTROLLEN))
    (AB.OUT_PUB / "r9_LAUF.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    AB.log("done in %.1f s; checks passed %d of %d" % (time.time() - t0, info["kontrollen_bestanden"], info["kontrollen_gesamt"]))
    if info["kontrollen_bestanden"] != info["kontrollen_gesamt"]:
        raise SystemExit("NOT ALL CHECKS PASSED")


def _wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", default="lauf")
    ap.add_argument("--ohne-paritaet", action="store_true")
    a = ap.parse_args()
    lauf(paritaet=not a.ohne_paritaet)
