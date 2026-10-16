# -*- coding: utf-8 -*-
"""aw239_r6_auswertung.py (AW-239, P12-R6, Version 1.0): LUDB und NeuroKit2, Zeitmessung, Skalierung, Gegenprobe (PUB-12, explorativ, post hoc).

Eingabe: nk_<Variante>.pkl aus aw239_r6_lauf.py (NeuroKit2 0.2.13, 200 LUDB-Aufnahmen), LUDB-Expertenmarken. Die Referenz- und Zuordnungsfunktionen stammen unveraendert aus
bench_auswertung_v11.py (V2-Skript 1.1, Hash geprueft): Cluster der Expertenmarken ueber die Ableitungen (mindestens 3 Ableitungen, 80 ms), eins-zu-eins-Zuordnung ueber den QRS-Gipfel (80 ms).

Zellen (identische Definition je Zelle):
  global/ext-ext    NeuroKit2-Beginn = frueheste Ableitungsgrenze des Clusters (g1-Definition), Referenz = fruehester Expertenbeginn
  global/med-med    NeuroKit2-Beginn = Median ueber die Ableitungen, Referenz = Median der Expertenbeginne
  global/ext-med    NeuroKit2 frueheste, Referenz Median   (gemischt, wie die Benchmark-Dauer g1 gegen Median)
  global/med-ext    NeuroKit2 Median, Referenz fruehester
  ableitungsweise   je Ableitung und Schlag: NeuroKit2-Grenze der Ableitung minus Expertengrenze derselben Ableitung (Zuordnung ueber den R-Gipfel der Ableitung, 80 ms, eins-zu-eins)
Kennzahlen: Median des Fehlers (NeuroKit2 minus Experte, ms), Anteil mit Fehler unter -50 ms, Anteil innerhalb +-20 ms; fuer QRS-Beginn und QRS-Ende. Intervalle: Perzentil-Bootstrap
ueber Aufnahmen (200, 2.000 Ziehungen, fester Seed). Ableitungsweise zusaetzlich: Anteil der NeuroKit2-Beginne innerhalb der Expertenmarke der P-Welle, Fehler nach Lage des
NeuroKit2-Q-Gipfels (0 bis 60 ms nach dem Expertenbeginn oder anderswo), Ableitungsgruppen (Extremitaeten, V1 bis V3, V4 bis V6).
Gepaart gegen V0: Anteil identischer Marken (Skaleninvarianz) und Median der Aenderung.
Kontrollen: (K1) 10 Aufnahmen (20, 40, ..., 200) gegen die gespeicherten AW-220-Marken (VV0), (K2) V0 global ext-ext und med-med gegen Z3 (-118 / -55 ms, Toleranz 1 ms).
Aufruf: <local path> -I -B aw239_r6_auswertung.py --modus lauf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

VERSION = "1.0"
SEED = 15102397
B_ZIEHUNGEN = 2000
CHUNK = 250
HIER = Path(__file__).resolve().parent
BASIS_PFAD = HIER / "aw239_basis.py"
LAUF_PFAD = HIER / "aw239_r6_lauf.py"
BASIS_HASH = "50fcf31d788931c671bd0021e112d2f9272795daecd2c24bbd215af0b0d4c7bd"
IN = Path(str(data_path("pub12.benchmark_pub12_aw239_r6")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw239")))
VARIANTEN = ["V0", "G0.1", "G0.5", "G2", "G10", "Glognorm", "Gtypisch", "Rexpert"]
LEADGRUPPEN = {"alle": range(12), "extremitaeten": range(0, 6), "V1bis3": range(6, 9), "V4bis6": range(9, 12)}
GRENZEN = ["qrs_on", "qrs_off"]


def _lade(name, pfad, soll=None):
    h = hashlib.sha256(Path(pfad).read_bytes()).hexdigest()
    if soll not in (None, "UNGESETZT") and h != soll:
        raise SystemExit("%s weicht ab: %s" % (name, h))
    spec = importlib.util.spec_from_file_location(name, str(pfad))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m, h


BA, BASIS_IST = _lade("aw239_basis", BASIS_PFAD, BASIS_HASH)
LAUF, LAUF_HASH = _lade("aw239_r6_lauf", LAUF_PFAD)
S = BA.S
log = BA.log


def lade_v2():
    p = S.PFADE["v2skript"]
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if h != S.V2_SKRIPT_HASH:
        raise SystemExit("V2-Skript weicht ab")
    spec = importlib.util.spec_from_file_location("bench_v11_r6", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------------------------
# Bootstrap ueber Aufnahmen fuer gepoolte Fehler
# ---------------------------------------------------------------------------------------------------------------
class Boot:
    def __init__(self, n_rec, seed=SEED):
        self.z = S.Ziehungen(n_rec, seed, B=B_ZIEHUNGEN, chunk=CHUNK)

    def _bloecke(self, rec_idx):
        yield np.ones((1, self.z.K))[:, rec_idx]
        for c in range(self.z.n_chunks):
            yield self.z.chunk_c(c)[:, rec_idx]

    def median(self, e, rec_idx):
        e = np.asarray(e, float)
        ok = np.isfinite(e)
        e, rec_idx = e[ok], np.asarray(rec_idx)[ok]
        if len(e) == 0:
            return np.full(B_ZIEHUNGEN + 1, np.nan)
        o = np.argsort(e, kind="stable")
        xs, ri = e[o], rec_idx[o]
        res = []
        for W in self._bloecke(ri):
            cs = np.cumsum(W, axis=1)
            N = cs[:, -1]
            j = (cs >= 0.5 * N[:, None] - 1e-9).argmax(axis=1)
            res.append(xs[j])
        return np.concatenate(res)

    def anteil(self, ind, rec_idx, nenner_ok=None):
        ind = np.asarray(ind, float)
        ok = np.ones(len(ind), bool) if nenner_ok is None else np.asarray(nenner_ok, bool)
        ind, rec_idx = ind[ok], np.asarray(rec_idx)[ok]
        if len(ind) == 0:
            return np.full(B_ZIEHUNGEN + 1, np.nan)
        res = []
        for W in self._bloecke(rec_idx):
            with np.errstate(invalid="ignore", divide="ignore"):
                res.append((W @ ind) / W.sum(axis=1))
        return np.concatenate(res)


def ci(v):
    v = np.asarray(v, float)
    w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return float(v[0]), np.nan, np.nan
    lo, hi = np.percentile(w, [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


def kennzahlen(boot, e, rec_idx, prefix=""):
    e = np.asarray(e, float)
    r = {prefix + "n": int(np.isfinite(e).sum())}
    if r[prefix + "n"] == 0:
        return r
    ok = np.isfinite(e)
    for nm, v in (("median_ms", boot.median(e, rec_idx)), ("anteil_unter_minus50_pct", 100 * boot.anteil(e < -50, rec_idx, ok)),
                  ("anteil_innerhalb20_pct", 100 * boot.anteil(np.abs(e) <= 20, rec_idx, ok))):
        r[prefix + nm], r[prefix + nm + "_ki_u"], r[prefix + nm + "_ki_o"] = ci(v)
    return r


# ---------------------------------------------------------------------------------------------------------------
# Marken
# ---------------------------------------------------------------------------------------------------------------
def globale_fehler(V2, store, ref, ids):
    """Zeilen (aufnahme_index, tool_agg, ref_agg, grenze, fehler_ms) fuer die vier globalen Zellen."""
    rows = []
    for ai, aid in enumerate(ids):
        lb = store.get(aid)
        if not lb:
            continue
        rec = V2.nk_record({j: np.asarray(a, float) for j, a in lb.items()})
        for tool_agg in ("ext", "med"):
            tb = V2.werkzeug_beats_nk(rec, tool_agg)
            r_, nz, nm = V2.fehler_aufnahme(ref[aid]["beats"], tb, V2.FENSTER_SAMPLES)
            for (ri, rv, nme, err) in r_:
                if nme in GRENZEN:
                    rows.append((ai, tool_agg, rv, nme, ri, err))
    return pd.DataFrame(rows, columns=["ai", "tool_agg", "ref_agg", "grenze", "ref_schlag", "fehler_ms"])


def ableitungsweise(store, expert, ids):
    """Zeilen je Ableitung und Schlag: Fehler Beginn/Ende, Lage des Q-Gipfels, P-Welle."""
    from scipy.optimize import linear_sum_assignment
    rows = []
    for ai, aid in enumerate(ids):
        lb = store.get(aid)
        if not lb:
            continue
        for j, arr in lb.items():
            arr = np.asarray(arr, float)
            ex = expert[aid][j]
            N = ex["N"]
            if len(arr) == 0 or not N:
                continue
            rpk = arr[:, 0]
            epk = np.array([n[0] for n in N], float)
            cost = np.abs(rpk[:, None] - epk[None, :])
            ri, ci_ = linear_sum_assignment(np.where(cost <= 40, cost, 1e6))
            for a, b in zip(ri, ci_):
                if cost[a, b] > 40:
                    continue
                pk, on, off = N[b]
                t_on, t_off, t_q = arr[a, 5], arr[a, 6], arr[a, 4]
                p_in = np.nan
                pc = [p for p in ex["p"] if 0 < pk - p[0] <= 200]
                if pc and np.isfinite(t_on):
                    p = pc[-1]
                    p_in = float(p[1] <= t_on <= p[2])
                q_rel = (t_q - on) * 2.0 if np.isfinite(t_q) else np.nan
                rows.append((ai, j, (t_on - on) * 2.0 if np.isfinite(t_on) else np.nan, (t_off - off) * 2.0 if np.isfinite(t_off) else np.nan,
                             p_in, q_rel, aid, int(b)))
    return pd.DataFrame(rows, columns=["ai", "lead", "err_on", "err_off", "onset_in_P", "q_rel_ms", "aid", "exp_beat"])


def lese_alle_expert(ids):
    return {aid: LAUF.lese_expert(aid) for aid in ids}


# ---------------------------------------------------------------------------------------------------------------
def kontrollen(V2, ids, store0, ref, glob0):
    """K1: frischer Lauf gegen gespeicherte AW-220-Marken (10 Aufnahmen), K2: Z3-Werte."""
    out = []
    P = S.PFADE
    nkpfad = P["erg"] / "neurokit2/langformat/neurokit2_schlaege.csv.gz"
    nk = pd.read_csv(nkpfad, dtype={"datensatz": str, "vorverarbeitung": str, "ableitung": str})
    nk = nk[(nk.datensatz == "ludb") & (nk.vorverarbeitung == "VV0")]
    sel = [a for a in range(20, 201, 20) if a in ids]
    gleich = tot = 0
    for aid in sel:
        d = nk[nk.aufnahme == aid]
        for j, ld in enumerate(LAUF.LEADS):
            dd = d[d.ableitung.str.upper() == ld].sort_values("schlag_nr")
            ref_arr = dd[V2.NK_SPALTEN].to_numpy(float)
            neu = np.asarray(store0[aid].get(j, np.zeros((0, 11))), float)
            tot += 1
            if ref_arr.shape == neu.shape and np.array_equal(np.nan_to_num(ref_arr, nan=-9e9), np.nan_to_num(neu, nan=-9e9)):
                gleich += 1
    out.append(dict(kontrolle="K1 frischer Lauf (V0) gleich gespeicherte AW-220-Marken, 10 Aufnahmen x 12 Ableitungen", soll="120 von 120 (AW-231)", ist="%d von %d" % (gleich, tot),
                    bestanden=gleich == tot))
    for tool_agg, ref_agg, soll in (("ext", "ext", -118), ("med", "med", -55)):
        g = glob0[(glob0.tool_agg == tool_agg) & (glob0.ref_agg == ref_agg) & (glob0.grenze == "qrs_on")].fehler_ms
        m = float(np.median(g))
        out.append(dict(kontrolle="K2 V0 global %s-%s QRS-Beginn Median (Z3 AW-221 v2)" % (tool_agg, ref_agg), soll="%d ms (+-1)" % soll, ist="%.1f" % m, bestanden=abs(m - soll) <= 1.0))
    return pd.DataFrame(out)


def lauf():
    t0 = time.time()
    IN.mkdir(parents=True, exist_ok=True)
    S.STATE["log"] = str(IN / "aw239_r6_auswertung.log")
    me = Path(__file__)
    log("Start aw239_r6_auswertung Version %s, Skript-SHA-256 %s, Basis %s, Lauf-Skript %s" % (VERSION, BA.sha256_datei(me), BASIS_IST, LAUF_HASH))
    ok = S.hashpruefung(S.gelesene_dateien())
    log("Hashpruefung (Datensperre): %d Eintraege, alle stimmen: %s" % (len(S.STATE["pruef"]), ok))
    if not ok:
        raise SystemExit("ABBRUCH: Hashpruefung nicht bestanden")
    pruef = list(S.STATE["pruef"])
    V2 = lade_v2()
    ids = sorted(int(x.split("/")[-1]) for x in (LAUF.LUDB / "RECORDS").read_text().split())
    log("Aufnahmen: %d" % len(ids))
    stores = {v: pickle.load(open(IN / ("nk_%s.pkl" % v), "rb")) for v in VARIANTEN if (IN / ("nk_%s.pkl" % v)).exists()}
    log("Varianten geladen: %s" % list(stores))
    ref = {}
    for aid in ids:
        r = V2.lese_ludb_referenz(aid)
        r["beats"] = V2.beats_aus(r["qrs"], r["p"], r["t"], False)
        ref[aid] = r
    expert = lese_alle_expert(ids)
    boot = Boot(len(ids))
    glob, abl = {}, {}
    for v, st in stores.items():
        glob[v] = globale_fehler(V2, st, ref, ids)
        abl[v] = ableitungsweise(st, expert, ids)
        log("%s: globale Fehlerzeilen %d, ableitungsweise %d" % (v, len(glob[v]), len(abl[v])))
    kon = kontrollen(V2, ids, stores["V0"], ref, glob["V0"])
    log("Kontrollen:\n" + kon.to_string())
    kon.to_csv(IN / "r6_kontrollen.csv", index=False, encoding="utf-8")
    if not kon.bestanden.all():
        log("ABBRUCH: Kontrolle nicht bestanden")
        return 3
    dirs = [IN, KOPIE]
    KOPIE.mkdir(parents=True, exist_ok=True)

    def schreibe(df, name):
        for d in dirs:
            df.to_csv(Path(d) / name, index=False, encoding="utf-8", float_format="%.6g")

    # globale Zellen
    rows = []
    for v, g in glob.items():
        for (ta, ra, gr), d in g.groupby(["tool_agg", "ref_agg", "grenze"]):
            r = dict(variante=v, neurokit_aggregation=ta, referenz_aggregation=ra, grenze=gr, zelle="identisch definiert" if ta == ra else "gemischt")
            r.update(kennzahlen(boot, d.fehler_ms.to_numpy(), d.ai.to_numpy()))
            r["n_aufnahmen"] = int(d.ai.nunique())
            rows.append(r)
    schreibe(pd.DataFrame(rows), "r6_globale_zellen.csv")
    # ableitungsweise Zellen
    rows = []
    for v, a in abl.items():
        for gname, leads in LEADGRUPPEN.items():
            d = a[a.lead.isin(list(leads))]
            for gr, col in (("qrs_on", "err_on"), ("qrs_off", "err_off")):
                r = dict(variante=v, ableitungsgruppe=gname, grenze=gr)
                r.update(kennzahlen(boot, d[col].to_numpy(), d.ai.to_numpy()))
                if gr == "qrs_on":
                    ok_ = d.onset_in_P.notna()
                    r["anteil_beginn_innerhalb_expertenP_pct"], r["p_ki_u"], r["p_ki_o"] = ci(100 * boot.anteil(d.onset_in_P.fillna(0).to_numpy(), d.ai.to_numpy(), ok_.to_numpy()))
                    r["n_mit_expertenP"] = int(ok_.sum())
                rows.append(r)
    schreibe(pd.DataFrame(rows), "r6_ableitungsweise.csv")
    # Mechanismus: Fehler nach Lage des NeuroKit2-Q-Gipfels (V0 und alle Varianten, alle Ableitungen)
    rows = []
    for v, a in abl.items():
        d = a[a.q_rel_ms.notna() & a.err_on.notna()]
        nah = (d.q_rel_ms >= 0) & (d.q_rel_ms <= 60)
        for nm, m in (("Q-Gipfel 0 bis 60 ms nach Expertenbeginn", nah), ("Q-Gipfel anderswo", ~nah)):
            r = dict(variante=v, gruppe=nm)
            r.update(kennzahlen(boot, d[m].err_on.to_numpy(), d[m].ai.to_numpy()))
            rows.append(r)
    schreibe(pd.DataFrame(rows), "r6_mechanismus_q_lage.csv")
    # gepaart gegen V0 (ableitungsweise, gleiche Aufnahme/Ableitung/Expertenschlag)
    rows = []
    a0 = abl["V0"].set_index(["aid", "lead", "exp_beat"])
    for v, a in abl.items():
        if v == "V0":
            continue
        m = a.set_index(["aid", "lead", "exp_beat"]).join(a0[["err_on", "err_off"]], rsuffix="_v0", how="inner")
        for gr, c1, c0 in (("qrs_on", "err_on", "err_on_v0"), ("qrs_off", "err_off", "err_off_v0")):
            x, y = m[c1].to_numpy(float), m[c0].to_numpy(float)
            okp = np.isfinite(x) & np.isfinite(y)
            r = dict(variante=v, grenze=gr, n_gepaart=int(okp.sum()), n_beide_nicht_gesetzt=int((~np.isfinite(x) & ~np.isfinite(y)).sum()),
                     n_nur_variante_gesetzt=int((np.isfinite(x) & ~np.isfinite(y)).sum()), n_nur_v0_gesetzt=int((~np.isfinite(x) & np.isfinite(y)).sum()))
            if okp.sum():
                dd = (x - y)[okp]
                r["anteil_identisch_pct"] = 100 * float(np.mean(dd == 0))
                r["anteil_innerhalb_2ms_pct"] = 100 * float(np.mean(np.abs(dd) <= 2))
                r["median_aenderung_ms"] = float(np.median(dd))
                r["median_betrag_aenderung_ms"] = float(np.median(np.abs(dd)))
                r["p95_betrag_aenderung_ms"] = float(np.percentile(np.abs(dd), 95))
            rows.append(r)
    schreibe(pd.DataFrame(rows), "r6_gepaart_gegen_v0.csv")
    # Zuordnungsraten
    rows = []
    for v, st in stores.items():
        n_beats = sum(len(a) for lb in st.values() for a in lb.values())
        rows.append(dict(variante=v, aufnahmen=len(st), ableitungen=sum(len(lb) for lb in st.values()), schlaege_gesamt=n_beats,
                         ableitungsweise_zugeordnet=len(abl[v]), globale_zeilen_qrs_on=int(((glob[v].grenze == "qrs_on")).sum())))
    schreibe(pd.DataFrame(rows), "r6_zuordnung.csv")
    info_l = json.loads((IN / "r6_lauf_info.json").read_text(encoding="utf-8")) if (IN / "r6_lauf_info.json").exists() else {}
    info = dict(version=VERSION, skript_sha256=BA.sha256_datei(me), lauf_skript_sha256=LAUF_HASH, basis_sha256=BASIS_IST, v2_skript_sha256=S.V2_SKRIPT_HASH, seed=SEED,
                ziehungen=B_ZIEHUNGEN, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(pruef), lauf_info=info_l)
    for d in dirs:
        (Path(d) / "r6_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    pd.DataFrame(pruef).to_csv(KOPIE / "r6_hashpruefung.csv", index=False, encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.parse_args(argv)
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
