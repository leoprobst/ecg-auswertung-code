"""Converts the raw checkpoints of the NeuroKit2 run (nk_lauf.py) into the long format of ausgabeschema.csv.

No evaluation: only rows are written and statuses are counted (dataset x preprocessing x status).
Up to eleven rows per beat and lead (bezug = schlag, schlag_nr = running number of the R wave in this lead, 1-based).
Values in ms (sample index times 1000/sampling rate). Measure keys (schema addendum, schema/messgroessen_nachtrag_aw220.csv):
  qrs_gipfel_schlag (ECG_R_Peaks)   qrs_beginn_schlag (ECG_R_Onsets)   qrs_ende_schlag (ECG_R_Offsets)
  p_beginn_schlag (ECG_P_Onsets)    p_gipfel_schlag (ECG_P_Peaks)     p_ende_schlag (ECG_P_Offsets)
  q_gipfel_schlag (ECG_Q_Peaks)     s_gipfel_schlag (ECG_S_Peaks)
  t_beginn_schlag (ECG_T_Onsets)    t_gipfel_schlag (ECG_T_Peaks)     t_ende_schlag (ECG_T_Offsets)
Status: ok | fehlend (marker not set for this beat, or lead without R wave) | werkzeugfehler (exception in the lead
or record) | nicht_ladbar. arm = nativ; vorverarbeitung = VV0 | VV0_G0.5 | VV0_G1 | VV0_G2 | VV1 | VV2.
Additionally a wide beat table (neurokit2_schlaege.csv.gz, sample indices) and MANIFEST.json with hashes, counts, environment.

Usage:  python -I -B nk_langformat.py [--roh <jsonl>] [--out <folder>] [--lauf-id <id>]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
sys.dont_write_bytecode = True

META = Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv")))

# Order as in nk_lauf.py: [R_Peak] + WELLEN
SPALTEN_SCHLAG = ("ECG_R_Peaks", "ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets",
                  "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")
SCHLUESSEL = {
    "ECG_R_Peaks": "qrs_gipfel_schlag", "ECG_R_Onsets": "qrs_beginn_schlag", "ECG_R_Offsets": "qrs_ende_schlag",
    "ECG_P_Onsets": "p_beginn_schlag", "ECG_P_Peaks": "p_gipfel_schlag", "ECG_P_Offsets": "p_ende_schlag",
    "ECG_Q_Peaks": "q_gipfel_schlag", "ECG_S_Peaks": "s_gipfel_schlag",
    "ECG_T_Onsets": "t_beginn_schlag", "ECG_T_Peaks": "t_gipfel_schlag", "ECG_T_Offsets": "t_ende_schlag",
}
BESCHREIBUNG = {
    "qrs_gipfel_schlag": "R wave per beat and lead", "qrs_beginn_schlag": "QRS onset (R onset) per beat and lead",
    "qrs_ende_schlag": "QRS end (R offset) per beat and lead", "p_beginn_schlag": "P onset per beat and lead",
    "p_gipfel_schlag": "P peak per beat and lead", "p_ende_schlag": "P end per beat and lead",
    "q_gipfel_schlag": "Q peak per beat and lead", "s_gipfel_schlag": "S peak per beat and lead",
    "t_beginn_schlag": "T onset per beat and lead", "t_gipfel_schlag": "T peak per beat and lead",
    "t_ende_schlag": "T end per beat and lead",
}
SPALTEN = ["lauf_id", "datensatz", "aufnahme", "cluster_id", "ableitung", "bezug", "schlag_nr", "werkzeug", "werkzeug_version",
           "konfig_hash", "arm", "vorverarbeitung", "messgroesse", "native_bezeichnung", "einheit", "wert", "status", "statusgrund"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def sha_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def cluster_karte():
    pid = {}
    with META.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            pid[int(r["ecg_id"])] = float(r["patient_id"])
    rang = {p: i + 1 for i, p in enumerate(sorted(set(pid.values())))}
    return {e: f"p{rang[p]:06d}" for e, p in pid.items()}


def ms(idx, fs):
    v = idx * 1000.0 / fs
    return str(int(v)) if v == int(v) else repr(v)


def schreibe_nachtrag():
    p = HIER / "schema" / "messgroessen_nachtrag_aw220.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schluessel", "beschreibung", "einheit", "neurokit2_nativ", "bindung", "bemerkung"])
        for nat, k in SCHLUESSEL.items():
            w.writerow([k, BESCHREIBUNG[k], "ms", nat, "gebunden AW-220",
                        "nk.ecg_delineate(method='dwt'); sample index times 1000/fs; -1 in the raw checkpoint = marker missing"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roh", default=str(data_path("pub12.benchmark_ergebnisse_neurokit2_roh_aw220_nk_haupt_jsonl")))
    ap.add_argument("--out", default=str(data_path("pub12.benchmark_ergebnisse_neurokit2_langformat")))
    ap.add_argument("--lauf-id", default="AW220_nk_haupt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import nk_lauf as N  # only for AUFRUF and hash; imports no tool library
    import numpy
    import scipy
    import neurokit2
    konfig_hash = N.AUFRUF_HASH
    cmap = cluster_karte()
    seen = {}
    for line in Path(args.roh).read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["key"]] = r
    zaehl = {}
    lead_fehler_n = {}
    zeilen_n = {}
    handles = {}
    nachtrag = schreibe_nachtrag()

    def oeffne(ds, vv):
        key = (ds, vv)
        if key not in handles:
            tag = vv.replace(".", "")
            f = gzip.open(out / f"neurokit2_{ds}_{tag}_langformat.csv.gz", "wt", encoding="utf8", newline="")
            w = csv.writer(f)
            w.writerow(SPALTEN)
            handles[key] = (f, w)
            zeilen_n[key] = 0
        return handles[key]

    sch = gzip.open(out / "neurokit2_schlaege.csv.gz", "wt", encoding="utf8", newline="")
    ws = csv.writer(sch)
    ws.writerow(["lauf_id", "datensatz", "aufnahme", "cluster_id", "vorverarbeitung", "ableitung", "schlag_nr"] + [SCHLUESSEL[c] + "_sample" for c in SPALTEN_SCHLAG])
    for key in sorted(seen, key=lambda k: (seen[k]["datensatz"], seen[k]["vv"], int(seen[k]["aufnahme"]))):
        r = seen[key]
        ds, rec, vv = r["datensatz"], r["aufnahme"], r["vv"]
        cl = cmap[int(rec)] if ds == "ptbxl" else f"l{int(rec):03d}"
        f, w = oeffne(ds, vv)
        fs = r.get("fs", 500)
        basis = [args.lauf_id, ds, rec, cl]
        post = ["neurokit2", "0.2.13", konfig_hash, "nativ", vv]
        st = r["status"]
        zaehl[(ds, vv, st)] = zaehl.get((ds, vv, st), 0) + 1

        def kopf(ableitung, bezug, nr):
            return basis[:4] + [ableitung, bezug, nr] + post

        if st != "ok":
            for nat, k in SCHLUESSEL.items():
                w.writerow(kopf("global", "aufnahme", "") + [k, nat, "ms", "", st, r.get("grund", "")])
                zeilen_n[(ds, vv)] += 1
            continue
        leads = r["leads"]
        for lead in LEADS:
            if lead in r.get("lead_fehler", {}):
                lead_fehler_n[(ds, vv)] = lead_fehler_n.get((ds, vv), 0) + 1
                for nat, k in SCHLUESSEL.items():
                    w.writerow(kopf(lead, "aufnahme", "") + [k, nat, "ms", "", "werkzeugfehler", r["lead_fehler"][lead]])
                    zeilen_n[(ds, vv)] += 1
                continue
            beats = leads.get(lead, [])
            if not beats:
                w.writerow(kopf(lead, "aufnahme", "") + ["qrs_gipfel_schlag", "ECG_R_Peaks", "ms", "", "fehlend", "no R wave detected"])
                zeilen_n[(ds, vv)] += 1
                continue
            for i, b in enumerate(beats, 1):
                ws.writerow(basis + [vv, lead, i] + [("" if v < 0 else v) for v in b])
                for c, v in zip(SPALTEN_SCHLAG, b):
                    k = SCHLUESSEL[c]
                    if v < 0:
                        w.writerow(kopf(lead, "schlag", i) + [k, c, "ms", "", "fehlend", "no marker (NeuroKit2 dwt)"])
                    else:
                        w.writerow(kopf(lead, "schlag", i) + [k, c, "ms", ms(v, fs), "ok", ""])
                    zeilen_n[(ds, vv)] += 1
    for f, _ in handles.values():
        f.close()
    sch.close()
    tabellen = {}
    for p in sorted(out.glob("neurokit2_*.csv.gz")):
        tabellen[p.name] = {"sha256": sha_datei(p), "bytes": p.stat().st_size}
    man = {"auftrag": "AW-220", "lauf_id": args.lauf_id, "roh": args.roh, "roh_sha256": sha_datei(Path(args.roh)),
           "konfig_hash": konfig_hash, "aufruf": N.AUFRUF, "n_aufgaben": len(seen),
           "umgebung": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__, "neurokit2": neurokit2.__version__},
           "skripte_sha256": {n: sha_datei(HIER / n) for n in ("nk_lauf.py", "bench_arme.py", "nk_langformat.py")},
           "status_zaehlung_aufnahmen": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(zaehl.items())},
           "ableitungsfehler_zeilengruppen": {f"{a}|{b}": n for (a, b), n in sorted(lead_fehler_n.items())},
           "zeilen_je_datei": {f"{a}|{b}": n for (a, b), n in sorted(zeilen_n.items())},
           "tabellen": tabellen, "schema_nachtrag": str(nachtrag),
           "hinweis": "Counting only. No distributions, comparisons or differences computed."}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="ascii")
    print(json.dumps({"n_aufgaben": len(seen), "status": man["status_zaehlung_aufnahmen"], "ableitungsfehler": man["ableitungsfehler_zeilengruppen"]}, indent=1))


if __name__ == "__main__":
    main()
