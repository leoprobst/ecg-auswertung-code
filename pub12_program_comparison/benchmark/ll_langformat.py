"""Converts the raw checkpoints of the LoopLens run (ll_lauf.py) into the long format of ausgabeschema.csv.

No analysis: only rows are written and statuses are counted (dataset x arm x status). The data lock (hash of the tables)
is computed and entered in langformat/MANIFEST.json; the substantive lock is applied by the maintainers.

Call:  python -I -B ll_langformat.py [--roh <jsonl>] [--out <folder>]
Coding:  vorverarbeitung = VV0 (no scaling) | VV0_G0.5 | VV0_G1 | VV0_G2 (gain arm, factor after reading); arm = nativ;
            bezug = aufnahme (beats: separate table looplens_schlaege.csv.gz).
Status:     nicht_ladbar | werkzeugfehler (exception of the chain, also time-out) | ausgeschlossen_qc (signal QC of the
            chain unusable; the values are then only in looplens_qc_ausgeschlossen_werte.csv.gz for the sensitivity analysis) | fehlend (marker
            without value, reason from computation_notes) | ok.
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

ROOT_HASH = "0cd40d41d773434fd41d5ac7e91507aed85e7786e9a33ed86810fbe2765341d5"
WERKZEUG_VERSION = "A155-survival-nullable-20261007 (Quellbaum 0cd40d41...41d5)"
META = Path(str(data_path("pub12.samples_ptb_x_ptbxl_database_csv")))
SKRIPT_DIR = Path(__file__).resolve().parent

# Register name -> (measurement, unit, factor); extensions are candidates for a schema addendum
ABBILDUNG = {
    "qrs_duration_ms": ("qrs_dauer_global", "ms", 1.0),
    "qt_interval_ms": ("qt_global", "ms", 1.0),
    "qtc_fridericia_ms": ("qtcf", "ms", 1.0),
    "pr_interval_ms": ("pr_intervall", "ms", 1.0),
    "p_duration_ms": ("p_dauer", "ms", 1.0),
    "rr_interval_ms": ("rr_intervall_ll", "ms", 1.0),
    "spatial_qrs_t_angle_peak": ("qrst_winkel_peak_ll", "grad", 1.0),
    "spatial_qrs_t_angle_mean": ("qrst_winkel_integral_ll", "grad", 1.0),
    "qrs_vector_magnitude": ("vektorbetrag_max_qrs_ll", "mV", 0.001),
    "t_vector_magnitude": ("vektorbetrag_max_t_ll", "mV", 0.001),
    "qrs_area_vector_magnitude_mv_ms": ("integralbetrag_qrs_ll", "mV*ms", 1.0),
    "sum_absolute_qrst_integral_mv_ms": ("sai_qrst_ll", "mV*ms", 1.0),
    "qrs_loop_path_length": ("schleifenlaenge_qrs_ll", "mV", 1.0),
    "svg_magnitude": ("svg_betrag_ll", "mV*ms", 0.001),
}
ERWARTETE_ROH_EINHEIT = {
    "qrs_duration_ms": "ms", "qt_interval_ms": "ms", "qtc_fridericia_ms": "ms", "pr_interval_ms": "ms",
    "p_duration_ms": "ms", "rr_interval_ms": "ms", "spatial_qrs_t_angle_peak": "degrees",
    "spatial_qrs_t_angle_mean": "degrees", "qrs_vector_magnitude": "uV", "t_vector_magnitude": "uV",
    "qrs_area_vector_magnitude_mv_ms": "mV*ms", "sum_absolute_qrst_integral_mv_ms": "mV*ms",
    "qrs_loop_path_length": "mV", "svg_magnitude": "uV*ms",
}
SPALTEN = ["lauf_id", "datensatz", "aufnahme", "cluster_id", "ableitung", "bezug", "schlag_nr", "werkzeug", "werkzeug_version",
           "konfig_hash", "arm", "vorverarbeitung", "messgroesse", "native_bezeichnung", "einheit", "wert", "status", "statusgrund"]


def sha_file_plain(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def cluster_karte():
    import csv as _csv
    pid = {}
    with META.open(encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            pid[int(r["ecg_id"])] = float(r["patient_id"])
    rang = {p: i + 1 for i, p in enumerate(sorted(set(pid.values())))}
    return {e: f"p{rang[p]:06d}" for e, p in pid.items()}


def vv(row):
    if row["arm"] == "verstaerkung":
        f = float(row["faktor"])
        return "VV0_G" + (str(int(f)) if f == int(f) else str(f))
    return "VV0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roh", default=str(data_path("pub12.benchmark_ergebnisse_looplens_roh_aw219_haupt_jsonl")))
    ap.add_argument("--out", default=str(data_path("pub12.benchmark_ergebnisse_looplens_langformat")))
    ap.add_argument("--lauf-id", default="AW219_haupt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"quellbaum": ROOT_HASH, "ll_lauf_sha256": sha_file_plain(SKRIPT_DIR / "ll_lauf.py"), "powerline_hz": 50, "min_dauer_s": 6.5,
           "vcg": "kors", "delineator": "FusionDelineator() Standardgewichte 0.4/0.35/0.25, Konflikt 20", "marker": list(ABBILDUNG)}
    konfig_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cmap = cluster_karte()
    (out / "cluster_zuordnung_ecg_id.csv").write_text("ecg_id,cluster_id\n" + "\n".join(f"{e},{c}" for e, c in sorted(cmap.items())) + "\n", encoding="ascii")
    seen = {}
    for line in Path(args.roh).read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["key"]] = r  # later line wins (resumption)
    zaehl = {}
    einheiten_abw = 0
    paths = {n: out / n for n in ("looplens_langformat.csv.gz", "looplens_schlaege.csv.gz", "looplens_qc_ausgeschlossen_werte.csv.gz")}
    with gzip.open(paths["looplens_langformat.csv.gz"], "wt", encoding="utf8", newline="") as fl, \
            gzip.open(paths["looplens_schlaege.csv.gz"], "wt", encoding="utf8", newline="") as fs_, \
            gzip.open(paths["looplens_qc_ausgeschlossen_werte.csv.gz"], "wt", encoding="utf8", newline="") as fq:
        wl = csv.DictWriter(fl, fieldnames=SPALTEN); wl.writeheader()
        wq = csv.DictWriter(fq, fieldnames=SPALTEN); wq.writeheader()
        ws = csv.writer(fs_)
        ws.writerow(["lauf_id", "datensatz", "aufnahme", "cluster_id", "arm", "vorverarbeitung", "schlag_nr", "r_gipfel", "p_beginn",
                     "p_ende", "qrs_beginn", "qrs_ende", "t_ende", "p_vorhanden", "akzeptiert", "repraesentativ"])
        for key in sorted(seen, key=lambda k: (seen[k]["datensatz"], int(seen[k]["aufnahme"]), seen[k]["arm"], float(seen[k].get("faktor") or 0))):
            r = seen[key]
            ds, rec = r["datensatz"], r["aufnahme"]
            cl = cmap[int(rec)] if ds == "ptbxl" else f"l{int(rec):03d}"
            base = {"lauf_id": args.lauf_id, "datensatz": ds, "aufnahme": rec, "cluster_id": cl, "ableitung": "global", "bezug": "aufnahme",
                    "schlag_nr": "", "werkzeug": "looplens", "werkzeug_version": WERKZEUG_VERSION, "konfig_hash": konfig_hash,
                    "arm": "nativ", "vorverarbeitung": vv(r)}
            st = r["status"]
            qc_aus = st == "ok" and not r.get("qc_brauchbar", True)
            zeile_status = "ausgeschlossen_qc" if qc_aus else st
            z = zaehl.setdefault((ds, vv(r), zeile_status), 0)
            zaehl[(ds, vv(r), zeile_status)] = z + 1
            for reg, (mg, einh, fak) in ABBILDUNG.items():
                row = dict(base, messgroesse=mg, native_bezeichnung=reg, einheit=einh, wert="", status="", statusgrund="")
                if st != "ok":
                    row.update(status=st, statusgrund=r.get("grund", ""))
                    wl.writerow(row)
                    continue
                m = r["marker"].get(reg)
                if m is None:
                    # statusgrund texts are written into the output table and stay as they are
                    row.update(status="fehlend", statusgrund="Marker nicht im Registerstand")
                    wl.writerow(row)
                    continue
                v, un, note = m
                if un != ERWARTETE_ROH_EINHEIT[reg]:
                    einheiten_abw += 1
                if v is None:
                    row.update(status=("ausgeschlossen_qc" if qc_aus else "fehlend"),
                               statusgrund=(r.get("qc_gruende", "") if qc_aus else (note or "")))
                    wl.writerow(row)
                    continue
                wert = repr(float(v) * fak)
                if qc_aus:
                    wl.writerow(dict(row, status="ausgeschlossen_qc", statusgrund=r.get("qc_gruende", "")))
                    wq.writerow(dict(row, wert=wert, status="ok", statusgrund="Signal-QC der Kette unbrauchbar: " + r.get("qc_gruende", "")))
                else:
                    wl.writerow(dict(row, wert=wert, status="ok"))
            if st == "ok":
                akz = set(r.get("akzeptiert_r") or [])
                for i, b in enumerate(r["schlaege"], 1):
                    ws.writerow([args.lauf_id, ds, rec, cl, "nativ", vv(r), i, b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                                 (1 if b[0] in akz else 0) if r.get("akzeptiert_r") is not None else "", 1 if b[0] == r.get("rep_r") else 0])
    man = {"auftrag": "AW-219", "lauf_id": args.lauf_id, "roh": args.roh, "roh_sha256": sha_file_plain(Path(args.roh)),
           "konfig_hash": konfig_hash, "konfig": cfg, "n_aufgaben": len(seen), "einheiten_abweichungen": einheiten_abw,
           "status_zaehlung": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(zaehl.items())},
           "tabellen": {n: {"sha256": sha_file_plain(p), "bytes": p.stat().st_size} for n, p in paths.items()},
           "hinweis": "Counts only. No distributions, comparisons or differences computed."}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="ascii")
    print(json.dumps({"n_aufgaben": len(seen), "einheiten_abweichungen": einheiten_abw, "status": man["status_zaehlung"]}, indent=1))


if __name__ == "__main__":
    main()
