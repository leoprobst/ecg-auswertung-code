"""PUB-13 Testergaenzung (Option B): Treiber fuer den zweiten, nachtraeglich festgelegten Testdurchgang.

Die gebundene Datei k1_auswertung.py (Skript v3.2) bleibt bitgleich. Dieser Treiber importiert sie unveraendert und ruft
deren Funktionen (load_base, choose_index, load_text_and_rr, machine_flags, evaluate_rules, guard_phase, ...) auf. Er ersetzt
nur den Teil von `process_cohort`, der im Modus `test` den Merkmalscache fest abschaltet (load_cache liefert dort nur den
Schluessel, `has_cache = (mode != "test")`).

Teilbefehle
  selftest                 Selbsttest an Kunstdaten (kein Datenzugriff ausser temporaerer Dateien).
  aequivalenz-dev          Entwicklungsdaten: Regeltabelle mit dem Entwicklungscache ueber den Treiberpfad bauen und mit der vorhandenen
                           Regeltabelle der Entwicklung vergleichen (alle Spalten, exakt); schreibt ein Arbeitsverzeichnis fuer den
                           anschliessenden Einheitenlauf (Skript, Modus dev) in einem neuen Ausgabeordner.
  vergleich-einheiten      Einheitenergebnisse eines Ausgabeordners (dev) mit den vorhandenen Entwicklungsergebnissen vergleichen.
  extraktion-dev-probe     Entwicklungsdaten: Extraktionspfad (gebundener Arbeiter) an N EKG gegen den Cache pruefen.
  manifest                 Manifest M' bauen (M3 plus Treiber, Extraktionscode des Arbeiters).
  extraktion               Testmerkmale ergebnisblind extrahieren (nur nach Eintrag 'Beginn' im Register; Wachter des Skripts).
  aufbereiten-test         Regeltabelle der Testpartition aus den extrahierten Merkmalen bauen; schreibt das Arbeitsverzeichnis.
  einheiten                Startet das unveraenderte Skript (run) fuer genau R02, R03, R04, R06, R07, R14 in drei Prozessen (Test: erzwungen).
  abschluss                assemble des Skripts (schreibt 'Ende'), nur wenn genau die sechs Einheiten vorliegen.

Keine Kennungen in Ausgaben. MIMIC-Daten bleiben lokal.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import k1_geruest as G  # noqa: E402
import k1_auswertung as S  # noqa: E402  (unveraendert importiert; ihre Konstanten und Funktionen werden benutzt)

FEATURE_SRC = str(data_path("pub13.release_sources_a155_survival_nullable_20261007"))
FEATURE_CODE_HASH = "c166bd00403dcde9f154d6c50f15ae16791029d9b507c4a4094e6f96215789b9"
ORIG = Path(str(data_path("pub13.aw236_v3")))  # Ausgabeordner des ersten Durchgangs (nur lesen)
DEFAULT_ERG = Path(str(data_path("pub13.aw236_v3_erg")))  # Ausgabeordner der Testergaenzung
PUB_ERG = Path(str(data_path("pub13.publikationen_pub_13_kardiologie_ergebnisse_aw236_ergaenzung")))
DEFAULT_SPEC = str(data_path("pub13.publikationen_pub_13_kardiologie_spezifikation_k1_entwurf_v3_2_md"))
SIX = ("R02", "R03", "R04", "R06", "R07", "R14")
KEEP_COLS = ["study_id", "extraction_status", "error_type"] + list(dict.fromkeys(S.CACHE_COLS))
IGNORE_KEYS = {"dauer_s"}  # reine Laufzeit, kein Ergebnis
WORK_FILES = ["main_keys", "main_rules", "main_cov", "main_refs", "ed_keys", "ed_rules", "ed_cov", "ed_refs"]

# Einheiten des zweiten Durchgangs: genau diese sechs, in drei festen Gruppen (je ein Prozess); nichts anderes wird im Test gerechnet.
GRUPPEN_TEST = (("R02", "R07"), ("R03", "R06"), ("R04", "R14"))
# Cache-abhaengige Regelspalten der sechs Einheiten (im ersten Durchgang vollstaendig leer): nur diese duerfen sich aendern.
AENDERBAR = ("R02_gt", "R02_ge", "R03_gt", "R03_ge", "R03_prod", "R03_fixM", "R03_fixF", "R04_ge", "R04_gt", "R04_fixM", "R04_fixF",
             "R06_a", "R06_b", "R06_c", "R07_a", "R07_b", "R07_c", "R14_a155", "lvh_volt")
# Cache-abhaengige Spalten der NICHT neu gerechneten Einheiten (R09, R18, R19): evaluate_rules(..., True) fuellt sie; der Treiber setzt sie
# auf die Werte des ersten Durchgangs (leer) zurueck, damit diese Einheiten in diesem Durchgang weder gerechnet noch beeinflusst werden.
ZURUECKSETZEN = ("R09_a155", "R18_own", "qrs_own")
OK_STATUS = ("ok", "signal_qc_failed", "lead_placement_excluded")  # Status des Arbeiters, die im Entwicklungscache vorkommen
FEHLER_GRENZE = 0.01  # Anteil Extraktionsfehler ("error" oder unbekannter Status) an allen Index-EKG; darueber wird gesperrt
SPERRDATEI = "SPERRE_FEHLERANTEIL.json"


# ====================================================================== Hilfen
def sha256_file(p) -> str:
    return G.sha256_file(p)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _ist_na(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def werte_gleich(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Typgerechter exakter Vergleich ohne Umwandlung nach float: gleiche Ganzzahl-, Bool-, Float- und Datumstypen direkt; sonst Element fuer
    Element mit gleichem Python-Typ und gleichem Wert (Zeichenketten als Zeichenketten, grosse Ganzzahlen ohne Praezisionsverlust).
    Fehlende Werte (NaN, NaT, None, NA) sind nur an denselben Positionen gleich."""
    if x.shape != y.shape:
        raise ValueError("Formen verschieden")
    if x.dtype == y.dtype and x.dtype.kind in "iub":
        return x == y
    if x.dtype == y.dtype and x.dtype.kind == "f":
        return (x == y) | (np.isnan(x) & np.isnan(y))
    if x.dtype == y.dtype and x.dtype.kind == "M":
        return (x == y) | (np.isnat(x) & np.isnat(y))
    out = np.empty(len(x), dtype=bool)
    for i, (p, q) in enumerate(zip(x, y)):
        if isinstance(p, np.generic):
            p = p.item()
        if isinstance(q, np.generic):
            q = q.item()
        na_p, na_q = _ist_na(p), _ist_na(q)
        out[i] = (na_p and na_q) if (na_p or na_q) else (type(p) is type(q) and bool(p == q))
    return out


def frames_exact_equal(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Liste der Spalten, die nicht exakt gleich sind (NaN gleich NaN, kein Toleranzbereich); leer = identisch."""
    problems = []
    if list(a.columns) != list(b.columns):
        problems.append(f"Spaltenliste verschieden: {set(a.columns) ^ set(b.columns)}")
    if len(a) != len(b):
        problems.append(f"Zeilenzahl {len(a)} != {len(b)}")
        return problems
    for c in a.columns:
        if c not in b.columns:
            continue
        same = werte_gleich(a[c].to_numpy(), b[c].to_numpy())
        if not bool(np.all(same)):
            problems.append(f"{c}: {int((~same).sum())} Zeilen verschieden")
    return problems


def all_nan(col: pd.Series) -> bool:
    return bool(col.isna().all())


def compare_json(a, b, path="", ignore=IGNORE_KEYS, out=None) -> list[str]:
    """Rekursiver exakter Vergleich zweier JSON-Strukturen; ignorierte Schluessel nur Laufzeit."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: nur in einer Seite")
            else:
                compare_json(a[k], b[k], f"{path}/{k}", ignore, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: Listenlaenge {len(a)} != {len(b)}")
        else:
            for i, (p, q) in enumerate(zip(a, b)):
                compare_json(p, q, f"{path}[{i}]", ignore, out)
    else:
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
    return out


def args_guard(out: Path, spec: str = DEFAULT_SPEC):
    """Wachter-Argumente des Skripts (Spezifikation, Skript, Manifest M', Entwicklungsergebnisse D) fuer den Ausgabeordner."""
    return S.guard_args_for("test", SimpleNamespace(spec=spec), Path(out))


def pruefe_manifest(out: Path) -> None:
    """Manifest M' vollstaendig verifizieren (alle Hashes aller Eintraege). Vor jedem Testzugriff aufzurufen."""
    mf = Path(out) / "manifest.json"
    if not mf.exists():
        raise SystemExit("Manifest M' fehlt")
    problems = G.verify_manifest(mf)
    if problems:
        raise SystemExit("Manifest M' verletzt: " + "; ".join(problems[:5]))


def beginne_schritt(out: Path, ga: dict, was: str) -> None:
    """Wie die Schritte von Skript v3.2: Waechter (nur nach Eintrag 'Beginn'), bei technischer Fortsetzung Zeile 'Fortsetzung' im Register
    (continue_entry), danach Fortschrittsprotokoll."""
    S.guard_phase("test", "locked_test", **ga)
    S.continue_entry(Path(out), ga, was)
    S.append_progress(Path(out), was)


# ---------------------------------------------------------------- Fehleranteil der Extraktion
def statusverteilung_dateien(files) -> dict:
    """Zaehlt extraction_status ueber ALLE angegebenen Teil- oder Merkmalsdateien (auch aus frueheren, fortgesetzten Laeufen)."""
    cnt: dict = {}
    for p in files:
        s = pd.read_parquet(p, columns=["extraction_status"])["extraction_status"]
        for k, v in s.value_counts(dropna=False).items():
            key = None if pd.isna(k) else str(k)
            cnt[key] = cnt.get(key, 0) + int(v)
    return cnt


def fehleranteil(cnt: dict) -> tuple[int, int, float]:
    n = int(sum(cnt.values()))
    err = int(sum(v for k, v in cnt.items() if k not in OK_STATUS))  # "error", unbekannter oder fehlender Status
    return err, n, (err / n if n else 1.0)


def pruefe_fehleranteil(cnt: dict, sperrdatei: Path | None = None, grenze: float = FEHLER_GRENZE) -> float:
    """Liegt der Anteil Fehler ueber der Grenze (strikt groesser), wird eine Sperrdatei geschrieben und abgebrochen."""
    err, n, share = fehleranteil(cnt)
    if n == 0 or share > grenze:
        if sperrdatei is not None:
            Path(sperrdatei).parent.mkdir(parents=True, exist_ok=True)
            Path(sperrdatei).write_text(json.dumps({"fehler": err, "gesamt": n, "anteil": share, "grenze": grenze, "status": cnt}, indent=1, default=str),
                                        encoding="utf-8")
        raise SystemExit(f"Extraktionsfehler {err} von {n} ({100.0 * share:.2f} %) ueber der Grenze {100.0 * grenze:.1f} %: Weiterverarbeitung gesperrt")
    return share


def pruefe_keine_sperre(merkmale_dir: Path) -> None:
    if (Path(merkmale_dir) / SPERRDATEI).exists():
        raise SystemExit("Sperrdatei Fehleranteil vorhanden: Weiterverarbeitung gesperrt (K fragen)")


# ---------------------------------------------------------------- Vergleich der Regeltabelle des Tests mit dem ersten Durchgang
def vergleiche_regeltabelle_test(new: pd.DataFrame, old: pd.DataFrame) -> list[str]:
    """Gibt Probleme zurueck (leer = bestanden). Schema (Spaltenliste, Reihenfolge, dtype, Zeilen, Index) muss gleich sein. Nur die in AENDERBAR
    genannten Spalten duerfen sich aendern; alle anderen Spalten, auch vollstaendig leere, sind exakt gleich dem ersten Durchgang."""
    probs: list[str] = []
    if list(new.columns) != list(old.columns):
        probs.append(f"Schema: Spaltenliste oder Reihenfolge verschieden ({set(new.columns) ^ set(old.columns) or 'Reihenfolge'})")
        return probs
    if len(new) != len(old) or not new.index.equals(old.index):
        probs.append("Schema: Zeilenzahl oder Index verschieden")
        return probs
    for c in old.columns:
        if new[c].dtype != old[c].dtype:
            probs.append(f"Schema: dtype {c}: {new[c].dtype} != {old[c].dtype}")
    for c in AENDERBAR + ZURUECKSETZEN:
        if c not in old.columns:
            probs.append(f"Liste enthaelt unbekannte Spalte {c}")
        elif not all_nan(old[c]):
            probs.append(f"{c}: im ersten Durchgang nicht leer (Annahme der Liste verletzt)")
    if probs:
        return probs
    rest = [c for c in old.columns if c not in AENDERBAR]
    probs += frames_exact_equal(new[rest], old[rest])
    for c in AENDERBAR:
        v = new[c].to_numpy(dtype=float)
        if not bool(np.all(np.isnan(v) | (v == 0.0) | (v == 1.0))):
            probs.append(f"{c}: Werte ausserhalb von 0, 1, leer")
        if bool(np.all(np.isnan(v))):
            probs.append(f"{c}: nach der Extraktion vollstaendig leer")
    return probs


# ---------------------------------------------------------------- Umfang des Einheitenlaufs
def erzwinge_einheiten(units: str | None, modus: str) -> tuple[str, ...]:
    """Im Test genau die sechs festgelegten Einheiten; im Dev (nur fuer Proben des Treibers) eine Teilmenge davon. Alles andere bricht ab."""
    req = tuple(u.strip() for u in (units or ",".join(SIX)).split(",") if u.strip())
    if modus == "test":
        if sorted(req) != sorted(SIX):
            raise SystemExit(f"Im Test werden genau {', '.join(SIX)} gerechnet (verlangt: {', '.join(req)})")
        return SIX
    extra = [u for u in req if u not in SIX]
    if extra or not req:
        raise SystemExit(f"Unzulaessige Einheiten: {extra}")
    return req


def pruefe_ergebnisordner(units_dir: Path, erlaubt: tuple[str, ...], vollstaendig: bool) -> None:
    names = sorted(p.stem for p in Path(units_dir).glob("*.json")) if Path(units_dir).exists() else []
    fremd = [n for n in names if n not in erlaubt]
    if fremd:
        raise SystemExit(f"Ergebnisordner enthaelt nicht erlaubte Einheiten: {fremd}")
    if vollstaendig and sorted(names) != sorted(erlaubt):
        raise SystemExit(f"Ergebnisordner unvollstaendig: vorhanden {names}")


# ====================================================================== Regeltabelle (der ersetzte Teil von process_cohort)
def lade_merkmalsdatei(path, idx: pd.DataFrame) -> pd.DataFrame:
    """Liest die extrahierten Merkmale (Schluessel study_id plus CACHE_COLS) und prueft sie gegen den Index: genau eine Zeile je
    Index-EKG, keine fremden EKG. Fehlende Spalten sind ein Fehler (kein stilles NaN)."""
    f = pd.read_parquet(path)
    miss = [c for c in S.CACHE_COLS if c not in f.columns]
    if miss:
        raise SystemExit(f"Merkmalsdatei ohne Spalten: {miss[:5]}")
    if f["study_id"].duplicated().any():
        raise SystemExit("Merkmalsdatei: doppelte study_id")
    ids = set(idx["study_id"].tolist())
    fids = set(f["study_id"].tolist())
    if fids - ids:
        raise SystemExit(f"Merkmalsdatei enthaelt {len(fids - ids)} EKG ausserhalb des Index")
    if ids - fids:
        raise SystemExit(f"Merkmalsdatei fehlen {len(ids - fids)} Index-EKG")
    f = f[["study_id"] + S.CACHE_COLS].copy()
    # Im Entwicklungscache fehlen Zeilen mit Extraktionsfehler ganz; die Verbindung auf study_id ergibt dort NaN. Hier stehen solche Zeilen
    # mit leerem Wert: gleichgestellt, damit evaluate_rules sie wie fehlende Cache-Zeilen behandelt (Signalqualitaet unbekannt, nicht nutzbar).
    f["signal_quality_usable"] = f["signal_quality_usable"].astype(object).where(f["signal_quality_usable"].notna(), np.nan)
    return f


def regeltabelle(con, cfg, idx_main: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Entspricht den Schritten von S.process_cohort("main", ..., light=False) vom Befundtext bis zur Regeltabelle, mit Cache."""
    txt = S.load_text_and_rr(con, cfg, idx_main)
    idx = idx_main.merge(txt, on="study_id", how="left")
    mf = S.machine_flags(idx)
    idx = pd.concat([idx, mf], axis=1)
    idx = idx.merge(cache, on="study_id", how="left")
    checks = S.plausibility_checks(idx)
    rules = S.evaluate_rules(idx, True)
    rules = pd.concat([rules, mf], axis=1)
    return rules, checks


def hauptindex(con, cfg, mode: str, ga: dict) -> pd.DataFrame:
    base = S.load_base(con, cfg, mode, ga)
    base = base[base["sex"].isin(["M", "F"])]
    return S.choose_index(base)


def schreibe_arbeitsordner(src_work: Path, dst_work: Path, rules: pd.DataFrame, extra_files=()) -> None:
    dst_work.mkdir(parents=True, exist_ok=True)
    for nm in WORK_FILES:
        if nm == "main_rules":
            rules.reset_index(drop=True).to_parquet(dst_work / "main_rules.parquet")
        else:
            shutil.copyfile(src_work / f"{nm}.parquet", dst_work / f"{nm}.parquet")
    for nm in extra_files:
        if (src_work / nm).exists():
            shutil.copyfile(src_work / nm, dst_work / nm)


# ====================================================================== Aequivalenz auf Entwicklungsdaten
def cmd_aequivalenz_dev(a):
    out = Path(a.out)
    src_work = ORIG / "dev" / "work"
    cfg = S.CFG
    con = S.connect(cfg)
    t0 = time.time()
    idx = hauptindex(con, cfg, "dev", {})
    keys_old = pd.read_parquet(src_work / "main_keys.parquet", columns=["study_id"])
    if not np.array_equal(idx["study_id"].to_numpy(), keys_old["study_id"].to_numpy()):
        raise SystemExit("Aequivalenz verletzt: Index-EKG (Reihenfolge) weicht von den Schluesseln der Entwicklung ab")
    log(f"Index {len(idx)} Zeilen gleich den Schluesseln der Entwicklung ({time.time() - t0:.0f} s)")
    cache = S.load_cache(con, cfg, idx.merge(S.load_text_and_rr(con, cfg, idx), on="study_id", how="left"), "dev", {})
    log(f"Entwicklungscache gelesen: {len(cache)} Zeilen")
    rules, checks = regeltabelle(con, cfg, idx, cache)
    old = pd.read_parquet(src_work / "main_rules.parquet")
    probs = frames_exact_equal(rules.reset_index(drop=True), old)
    res = {"aequivalenz_regeltabelle": "bestanden" if not probs else "verletzt", "probleme": probs[:20], "zeilen": int(len(rules)),
           "spalten": int(rules.shape[1]), "pruefungen": checks}
    out.mkdir(parents=True, exist_ok=True)
    (out / "aequivalenz_regeltabelle.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log(json.dumps(res, default=str)[:600])
    if probs:
        raise SystemExit("Aequivalenz der Regeltabelle verletzt")
    schreibe_arbeitsordner(src_work, out / "dev" / "work", rules)
    shutil.copyfile(ORIG / "dev" / "design_dev.json", out / "dev" / "design_dev.json")
    log(f"Arbeitsordner geschrieben: {out / 'dev' / 'work'}")


def cmd_vergleich_einheiten(a):
    out = Path(a.out)
    ok = True
    rep = {}
    for u in a.units.split(","):
        pa, pb = ORIG / "dev" / "units" / f"{u}.json", out / "dev" / "units" / f"{u}.json"
        if not pb.exists():
            rep[u] = "Ergebnis fehlt"
            ok = False
            continue
        d = compare_json(json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8")))
        rep[u] = {"gleich": not d, "abweichungen": d[:10], "datei_gleich_bis_auf_laufzeit": not d}
        ok = ok and not d
    (out / "aequivalenz_einheiten.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log(json.dumps(rep, indent=1)[:2000])
    if not ok:
        raise SystemExit("Aequivalenz der Einheiten verletzt")
    log("Aequivalenz der Einheiten: alle exakt gleich")


# ====================================================================== Extraktion (gebundener Arbeiter)
def _lade_fe():
    if FEATURE_SRC not in sys.path:
        sys.path.insert(0, FEATURE_SRC)
    from vcg_biomarker.mimic import feature_extraction as fe
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.panels import default_feature_panels
    from vcg_biomarker.mimic.storage import optional_column_select, relation_columns
    return fe, MimicProjectConfig, default_feature_panels, optional_column_select, relation_columns


def task_config():
    fe, Cfg, dfp, _, _ = _lade_fe()
    cfg = Cfg.load(str(data_path("pub13.mimic_validation_project_json")))
    if fe.feature_implementation_hash() != FEATURE_CODE_HASH:
        raise SystemExit("Arbeiter-Quellstand weicht vom gebundenen feature_code_hash ab")
    panels = {p.panel_id: p for p in dfp()}
    ids = []
    for pid in ("primary_core", "sentinel", "discovery_reserve", "classical_criteria"):
        ids.extend(panels[pid].feature_ids)
    ids = list(dict.fromkeys(ids))
    vcg = [n for n in ids if not n.startswith("classical_")]
    cls = [n for n in ids if n.startswith("classical_") and not n.endswith("_positive")]
    core = set(panels["primary_core"].feature_ids)
    sens = [n for n in vcg if n in core and n not in fe.TRANSFORM_INDEPENDENT_FEATURE_IDS]
    return fe.build_feature_task_config(cfg, vcg_feature_ids=vcg, sensitivity_feature_ids=sens, classical_feature_ids=cls)


def master_zeilen(study_ids, splits) -> pd.DataFrame:
    """Zeilen im Format der Kohortenabfrage der Stufe aus dem Masterindex (keine Labels, keine Endpunkte)."""
    import duckdb
    fe, _, _, ocs, rc = _lade_fe()
    master = str(data_path("pub13.mimic_validation_master_ecg_master_index_parquet"))
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=2")
    mcols = rc(con, f"read_parquet('{master}')")
    bw, flt = ocs(mcols, "machine_bandwidth", "VARCHAR"), ocs(mcols, "machine_filtering", "VARCHAR")
    mi = fe._machine_interval_select(mcols)
    con.register("k", pd.DataFrame({"study_id": sorted(int(x) for x in study_ids)}))
    sp = ", ".join(f"'{s}'" for s in splits)
    df = con.execute(
        f"""SELECT subject_id, study_id, split, ecg_time, waveform_path, age, sex,
            CASE WHEN machine_metadata_usable_for_primary THEN machine_qrs_axis_deg ELSE NULL END AS machine_qrs_axis_deg,
            {mi},
            CASE WHEN machine_metadata_usable_for_primary THEN machine_report ELSE NULL END AS machine_report,
            machine_cart_id, {bw}, {flt}, machine_time_mismatch, machine_time_mismatch_seconds,
            machine_metadata_temporal_status, machine_metadata_usable_for_primary
            FROM read_parquet('{master}') m JOIN k USING (study_id) WHERE split IN ({sp}) ORDER BY study_id"""
    ).df()
    con.close()
    return df


def _batch_worker_name():
    return "_extract_record_batch_worker"


def extrahiere(rows: pd.DataFrame, parts_dir: Path, workers: int, batch: int = 25, flush_every: int = 500) -> dict:
    """Rechnet die Zeilen mit dem gebundenen Arbeiter (Spawn-Pool), schreibt Teildateien mit den benoetigten Spalten und kann
    nach Abbruch fortsetzen (bereits geschriebene study_id werden uebersprungen)."""
    import multiprocessing as mp
    fe, *_ = _lade_fe()
    cfg_t = task_config()
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in sorted(parts_dir.glob("part-*.parquet")):
        done.update(pd.read_parquet(p, columns=["study_id"])["study_id"].tolist())
    todo = rows[~rows["study_id"].isin(done)]
    n_part = len(list(parts_dir.glob("part-*.parquet")))
    log(f"Extraktion: {len(rows)} EKG, bereits fertig {len(done)}, offen {len(todo)}, Arbeiter {workers}")
    recs = todo.to_dict("records")
    batches = [recs[i:i + batch] for i in range(0, len(recs), batch)]
    ctx = mp.get_context("spawn")
    buf, t0, n_done, stat = [], time.time(), 0, {}
    with ctx.Pool(workers, initializer=fe._initialize_feature_worker, initargs=(cfg_t,)) as pool:
        for res in pool.imap_unordered(fe._extract_record_batch_worker, batches):
            for r in res:
                row = {k: r.get(k) for k in KEEP_COLS}
                buf.append(row)
                stat[r.get("extraction_status")] = stat.get(r.get("extraction_status"), 0) + 1
            n_done += len(res)
            if len(buf) >= flush_every:
                n_part += 1
                _schreibe_teil(parts_dir / f"part-{n_part:05d}.parquet", buf)
                buf = []
                rate = n_done / max(time.time() - t0, 1e-9)
                log(f"{n_done}/{len(recs)} EKG, {rate:.2f} EKG/s, Status {stat}")
        if buf:
            n_part += 1
            _schreibe_teil(parts_dir / f"part-{n_part:05d}.parquet", buf)
    return {"offen_gerechnet": len(recs), "status": stat, "sekunden": round(time.time() - t0)}


def _schreibe_teil(path: Path, buf: list[dict]) -> None:
    df = pd.DataFrame(buf, columns=KEEP_COLS)
    for c in KEEP_COLS:
        if c in ("study_id",):
            df[c] = df[c].astype("int64")
        elif c in ("extraction_status", "error_type"):
            df[c] = df[c].astype(object).where(df[c].notna(), None).map(lambda v: None if v is None else str(v))
        elif c in ("signal_quality_usable", "classical_pathological_q_waves_positive", "classical_rbbb_pattern_positive"):
            df[c] = pd.array([None if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v) for v in df[c].tolist()], dtype="boolean")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def fuege_teile_zusammen(parts_dir: Path, ziel: Path, erwartet_ids=None) -> dict:
    frames = [pd.read_parquet(p) for p in sorted(parts_dir.glob("part-*.parquet"))]
    f = pd.concat(frames, ignore_index=True).sort_values("study_id").reset_index(drop=True)
    if f["study_id"].duplicated().any():
        raise SystemExit("Teile enthalten doppelte study_id")
    if erwartet_ids is not None:
        e = set(int(x) for x in erwartet_ids)
        if set(f["study_id"].tolist()) != e:
            raise SystemExit("Teile decken die erwartete Menge nicht genau ab")
    tmp = ziel.with_suffix(".tmp")
    f.to_parquet(tmp)
    os.replace(tmp, ziel)
    return {"zeilen": int(len(f)), "sha256": sha256_file(ziel), "status": f["extraction_status"].value_counts(dropna=False).to_dict()}


def cmd_extraktion(a):
    out = Path(a.out)
    pruefe_manifest(out)  # vor dem ersten Testzugriff: alle Hashes von M' (Treiber, Extraktionscode, Skript, Eingaben)
    ga = args_guard(out)
    beginne_schritt(out, ga, "extraktion")  # Waechter (Beginn), bei Fortsetzung Registerzeile, Fortschrittsprotokoll
    mdir = out / "test" / "merkmale"
    pruefe_keine_sperre(mdir)
    keys = pd.read_parquet(ORIG / "test" / "work" / "main_keys.parquet", columns=["study_id"])
    rows = master_zeilen(keys["study_id"].tolist(), ["locked_test"])
    if len(rows) != len(keys):
        raise SystemExit(f"Masterindex liefert {len(rows)} statt {len(keys)} Zeilen")
    stat = extrahiere(rows, mdir / "teile", a.workers)
    info = fuege_teile_zusammen(mdir / "teile", mdir / "merkmale_test.parquet", keys["study_id"].tolist())
    info["lauf"] = stat
    # Fehleranteil ueber ALLE Teile (auch aus fortgesetzten Laeufen), nicht nur ueber neu gerechnete Zeilen
    cnt = statusverteilung_dateien(sorted((mdir / "teile").glob("part-*.parquet")))
    info["status_alle_teile"] = cnt
    err, n, share = fehleranteil(cnt)
    info["fehleranteil"] = {"fehler": err, "gesamt": n, "anteil": share, "grenze": FEHLER_GRENZE}
    (mdir / "merkmale_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    S.append_progress(out, f"extraktion beendet: Merkmalshash {info['sha256']}, Fehler {err} von {n}")
    pruefe_fehleranteil(cnt, mdir / SPERRDATEI)
    log(json.dumps(info, default=str))


def cmd_extraktion_dev_probe(a):
    """Entwicklungsdaten: Extraktionspfad des Treibers (Zeilenabfrage, Spawn-Pool, Teildateien, Zusammenfuehren) gegen den Cache."""
    import pyarrow.parquet as pq
    import duckdb
    cache_path = S.CFG["cache"].format(s="validation")
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=2")
    ids = con.execute(f"SELECT study_id FROM read_parquet('{cache_path}') ORDER BY hash(study_id * 104729 + 7) LIMIT {int(a.n)}").df()["study_id"].tolist()
    con.close()
    rows = master_zeilen(ids, ["validation"])
    work = Path(a.out) / "probe"
    if work.exists():
        raise SystemExit("probe-Ordner existiert schon (nichts wird ueberschrieben)")
    st = extrahiere(rows, work / "teile", a.workers, batch=5, flush_every=20)
    info = fuege_teile_zusammen(work / "teile", work / "merkmale.parquet", ids)
    fresh = pd.read_parquet(work / "merkmale.parquet").set_index("study_id").sort_index()
    pf = pq.ParquetFile(cache_path)
    sid = pf.read(columns=["study_id"]).column("study_id").to_numpy()
    want = set(int(x) for x in ids)
    rg, off = [], 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if any(int(v) in want for v in sid[off:off + n]):
            rg.append(g)
        off += n
    res, n_cells, n_same = {}, 0, 0
    for c in S.CACHE_COLS:
        tb = pf.read_row_groups(rg, columns=["study_id", c]).to_pandas()
        tb = tb[tb["study_id"].isin(want)].set_index("study_id").sort_index()
        x, y = tb[c], fresh[c].reindex(tb.index)
        try:
            xf, yf = x.astype(float).to_numpy(), y.astype(float).to_numpy()
            same = (xf == yf) | (np.isnan(xf) & np.isnan(yf))
        except (TypeError, ValueError):
            same = np.array([(pd.isna(p) and pd.isna(q)) or (p == q) for p, q in zip(x.tolist(), y.tolist())])
        n_cells += len(same)
        n_same += int(same.sum())
        if not bool(np.all(same)):
            res[c] = int((~same).sum())
    summary = {"ekg": len(ids), "spalten": len(S.CACHE_COLS), "zellen": n_cells, "identisch": n_same, "abweichende_spalten": res,
               "extraktion": st, "zusammenfuegen": {k: v for k, v in info.items() if k != "sha256"}}
    (Path(a.out) / "extraktion_dev_probe.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log(json.dumps(summary, default=str))
    if res:
        raise SystemExit("Extraktionspfad weicht vom Entwicklungscache ab")


# ====================================================================== Manifest M'
def cmd_manifest(a):
    out = Path(a.out)
    man = json.loads((ORIG / "manifest.json").read_text(encoding="utf-8"))
    extra = [Path(__file__).resolve()]
    mimic = Path(FEATURE_SRC) / "vcg_biomarker" / "mimic"
    for nm in ("classical.py", "lead_measurements.py", "feature_record.py", "panels.py", "report_parser.py", "signals.py", "feature_extraction.py"):
        extra.append(mimic / nm)
    extra += [p for p in sorted((Path(FEATURE_SRC) / "vcg_biomarker" / "core").rglob("*.py")) if "__pycache__" not in p.parts]
    for p in extra:
        man[str(p)] = sha256_file(p)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    log(f"Manifest M' mit {len(man)} Eintraegen: {sha256_file(out / 'manifest.json')}")
    print("manifest", sha256_file(out / "manifest.json"))


def cmd_vorbereiten(a):
    """Ausgabeordner der Testergaenzung: Entwicklungsergebnisse kopieren (D bleibt gleich), Manifest M' bauen."""
    out = Path(a.out)
    for sub in ("units",):
        (out / "dev" / sub).mkdir(parents=True, exist_ok=True)
        for p in sorted((ORIG / "dev" / sub).glob("*.json")):
            shutil.copyfile(p, out / "dev" / sub / p.name)
    shutil.copyfile(ORIG / "dev" / "design_dev.json", out / "dev" / "design_dev.json")
    shutil.copyfile(ORIG / "dev" / "results.json", out / "dev" / "results.json")
    d_new, d_old = S.dev_results_hash(out), S.dev_results_hash(ORIG)
    if d_new != d_old:
        raise SystemExit("Entwicklungsergebnis-Hash D weicht nach dem Kopieren ab")
    log(f"D = {d_new}")
    cmd_manifest(a)


# ====================================================================== Testpartition: Regeltabelle
def cmd_aufbereiten_test(a):
    out = Path(a.out)
    pruefe_manifest(out)
    ga = args_guard(out)
    beginne_schritt(out, ga, "aufbereiten-test")
    mdir = out / "test" / "merkmale"
    pruefe_keine_sperre(mdir)
    # Fehleranteil erneut und unabhaengig aus allen Teilen und aus der zusammengefuehrten Datei pruefen
    cnt_parts = statusverteilung_dateien(sorted((mdir / "teile").glob("part-*.parquet")))
    cnt_file = statusverteilung_dateien([mdir / "merkmale_test.parquet"])
    if cnt_parts != cnt_file:
        raise SystemExit("Statusverteilung der Teile weicht von der zusammengefuehrten Merkmalsdatei ab")
    pruefe_fehleranteil(cnt_file, mdir / SPERRDATEI)
    cfg = S.CFG
    con = S.connect(cfg)
    idx = hauptindex(con, cfg, "test", ga)
    src = ORIG / "test" / "work"
    keys_old = pd.read_parquet(src / "main_keys.parquet", columns=["study_id"])
    if not np.array_equal(idx["study_id"].to_numpy(), keys_old["study_id"].to_numpy()):
        raise SystemExit("Index-EKG der Testpartition weichen von den Schluesseln des ersten Durchgangs ab")
    feat = lade_merkmalsdatei(mdir / "merkmale_test.parquet", idx)
    rules, checks = regeltabelle(con, cfg, idx, feat)
    old = pd.read_parquet(src / "main_rules.parquet")
    rules = rules.reset_index(drop=True)
    # R09, R18, R19 werden in diesem Durchgang nicht gerechnet: ihre cache-abhaengigen Spalten bleiben auf dem Stand des ersten Durchgangs
    verworfen = {c: int(rules[c].notna().sum()) for c in ZURUECKSETZEN}
    for c in ZURUECKSETZEN:
        rules[c] = old[c].to_numpy()
    probs = vergleiche_regeltabelle_test(rules, old)
    if probs:
        raise SystemExit("Regeltabelle der Testpartition verletzt den Vergleich mit dem ersten Durchgang: " + "; ".join(probs[:8]))
    schreibe_arbeitsordner(src, out / "test" / "work", rules, extra_files=("eingaben_test.json", "prepare_summary.json"))
    info = {"zeilen": int(len(rules)), "spalten_neu_befuellt": list(AENDERBAR), "zurueckgesetzt_nicht_gerechnete_einheiten": verworfen,
            "pruefungen": checks, "fehleranteil": dict(zip(("fehler", "gesamt", "anteil"), fehleranteil(cnt_file))),
            "merkmalshash": sha256_file(mdir / "merkmale_test.parquet")}
    (out / "test" / "work" / "aufbereitung_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    S.append_progress(out, f"aufbereiten-test beendet: Merkmalshash {info['merkmalshash']}")
    log(json.dumps(info, default=str)[:800])


# ====================================================================== Einheitenlauf und Abschluss (Treiber startet das Skript selbst)
def pruefe_vor_einheiten(out: Path) -> None:
    """Voraussetzungen des Einheitenlaufs im Test: Aufbereitung beendet, keine Sperre, nicht gerechnete Einheiten unberuehrt."""
    out = Path(out)
    wdir = out / "test" / "work"
    pruefe_keine_sperre(out / "test" / "merkmale")
    if not (wdir / "aufbereitung_info.json").exists():
        raise SystemExit("Aufbereitung nicht abgeschlossen (aufbereitung_info.json fehlt)")
    cnt = statusverteilung_dateien([out / "test" / "merkmale" / "merkmale_test.parquet"])
    pruefe_fehleranteil(cnt, out / "test" / "merkmale" / SPERRDATEI)
    new = pd.read_parquet(wdir / "main_rules.parquet", columns=list(ZURUECKSETZEN))
    old = pd.read_parquet(ORIG / "test" / "work" / "main_rules.parquet", columns=list(ZURUECKSETZEN))
    if frames_exact_equal(new, old):
        raise SystemExit("Cache-Spalten der nicht gerechneten Einheiten (R09, R18, R19) weichen vom ersten Durchgang ab")
    pruefe_ergebnisordner(out / "test" / "units", SIX, vollstaendig=False)


def cmd_einheiten(a):
    """Startet das unveraenderte Skript v3.2 (run) als drei Prozesse mit festen Gruppen; erzwingt genau die sechs Einheiten."""
    import subprocess
    modus = a.modus
    out = Path(a.out)
    units = erzwinge_einheiten(a.units, modus)
    if modus == "test":
        pruefe_manifest(out)
        if Path(sys.executable).parent.parent.name != "auswertung":
            raise SystemExit("Einheitenlauf nur mit der Umgebung auswertung (wie im ersten Durchgang)")
        ga = args_guard(out)
        S.guard_phase("test", "locked_test", **ga)
        pruefe_vor_einheiten(out)
        gruppen = GRUPPEN_TEST
    else:
        gruppen = tuple(g for g in (tuple(u for u in gr if u in units) for gr in GRUPPEN_TEST) if g)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for k, g in enumerate(gruppen, 1):
        cmd = [sys.executable, "-B", str(Path(S.__file__)), "run", "--mode", modus, "--out", str(out), "--units", ",".join(g)]
        lf = open(out / "logs" / f"einheiten_{modus}_g{k}.log", "w", encoding="utf-8")
        procs.append((g, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf))
        log(f"Gruppe {k} gestartet: {','.join(g)}")
    codes = []
    for g, pr, lf in procs:
        codes.append(pr.wait())
        lf.close()
    if any(c != 0 for c in codes):
        raise SystemExit(f"Einheitenlauf fehlgeschlagen, Rueckgabecodes {codes} (Logs in {out / 'logs'})")
    pruefe_ergebnisordner(out / modus / "units", units, vollstaendig=True)
    log(f"Einheiten gerechnet: {', '.join(units)}")
    if modus == "test":
        S.append_progress(out, "einheiten beendet: " + ",".join(units))


def cmd_abschluss(a):
    """assemble des Skripts (schreibt im Test die Zeile 'Ende'); nur wenn genau die sechs Einheiten vorliegen."""
    out = Path(a.out)
    modus = a.modus
    units = erzwinge_einheiten(a.units, modus)
    if modus == "test":
        pruefe_manifest(out)
        pruefe_vor_einheiten(out)  # Sperrdatei, Fehleranteil aus der Merkmalsdatei, Aufbereitung, nicht gerechnete Einheiten unberuehrt
    pruefe_ergebnisordner(out / modus / "units", units, vollstaendig=True)
    S.main(["assemble", "--mode", modus, "--out", str(out), "--pub", str(a.pub)])


# ====================================================================== Selbsttest an Kunstdaten
def selftest() -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="k1_erg_selftest_"))
    ok = []

    def check(name, cond):
        ok.append((name, bool(cond)))
        print(("ok   " if cond else "FEHLER"), name)

    # 1. frames_exact_equal / compare_json
    a = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [0, 1, 1]})
    check("frames gleich", frames_exact_equal(a, a.copy()) == [])
    b = a.copy()
    b.loc[2, "x"] = 3.0000000001
    check("frames: kleinste Abweichung erkannt", frames_exact_equal(a, b) != [])
    check("json gleich bis auf Laufzeit", compare_json({"a": 1, "dauer_s": 5, "b": [1, 2]}, {"a": 1, "dauer_s": 9, "b": [1, 2]}) == [])
    check("json Abweichung erkannt", compare_json({"a": 1.0}, {"a": 1.0000001}) != [])

    # 2. Merkmalsdatei laden: Vertragspruefungen
    ids = [11, 12, 13]
    feat = pd.DataFrame({"study_id": ids})
    for c in S.CACHE_COLS:
        feat[c] = np.nan
    p = tmp / "f.parquet"
    feat.to_parquet(p)
    idx = pd.DataFrame({"study_id": ids})
    check("Merkmalsdatei: vollstaendig akzeptiert", len(lade_merkmalsdatei(p, idx)) == 3)
    for nm, f2 in (("fehlende EKG", feat.iloc[:2]), ("fremde EKG", pd.concat([feat, feat.iloc[[0]].assign(study_id=99)])),
                   ("doppelte", pd.concat([feat, feat.iloc[[0]]])), ("fehlende Spalte", feat.drop(columns=[S.CACHE_COLS[0]]))):
        q = tmp / f"{nm}.parquet"
        f2.to_parquet(q)
        try:
            lade_merkmalsdatei(q, idx)
            check(f"Merkmalsdatei: {nm} abgelehnt", False)
        except SystemExit:
            check(f"Merkmalsdatei: {nm} abgelehnt", True)

    # 3. Regelpfad mit Kunstdaten: unveraendertes evaluate_rules ueber den Treiberpfad (kein Cache-Fest-Aus)
    n = 4
    cols = {c: np.full(n, np.nan) for c in S.CACHE_COLS}
    for L in S.AMP_LEADS:
        cols[f"lead_measurement_{L}_qrs_r_height_mv"] = np.array([0.5, 2.0, 0.5, 0.5])
        cols[f"lead_measurement_{L}_qrs_s_depth_mv"] = np.array([0.5, 0.5, 0.5, 0.5])
    cols["lead_measurement_V1_qrs_s_depth_mv"] = np.array([0.5, 2.5, 0.5, 0.5])  # Zeile 2: S V1 gross
    cols["lead_measurement_V5_qrs_r_height_mv"] = np.array([0.5, 2.0, 0.5, 0.5])  # Zeile 2: R V5 gross -> Sokolow-Lyon positiv
    cols["signal_quality_usable"] = np.array([True, True, False, True], dtype=object)
    cols["classical_pathological_q_waves_positive"] = np.array([False, False, False, True], dtype=object)
    cols["classical_rbbb_pattern_positive"] = np.array([False, False, False, False], dtype=object)
    cols["classical_qrs_widening_ge_120"] = np.array([100.0, 100, 100, 100])
    cols["pr_interval_ms"] = np.array([160.0, 160, 160, 160])
    cols["context_machine_qrs_duration_ms"] = np.array([100.0] * 4)
    cols["context_machine_pr_interval_ms"] = np.array([160.0] * 4)
    cols["context_machine_qt_interval_ms"] = np.array([400.0] * 4)
    idxs = pd.DataFrame({"study_id": np.arange(n), "sex": ["M", "M", "F", "M"], "report_pacemaker_state": ["not_mentioned"] * n,
                         "report_lbbb_state": ["not_mentioned"] * n, "report_rbbb_state": ["not_mentioned"] * n,
                         "report_af_state": ["not_mentioned"] * n, "sinus": [1.0] * n, "rr_ms": [800.0] * n,
                         "qrs_ms": [100.0] * n, "pr_ms": [160.0] * n, "qt_ms": [400.0] * n, "report_infarct_state": ["not_mentioned"] * n})
    cache = pd.concat([pd.DataFrame({"study_id": np.arange(n)}), pd.DataFrame(cols)], axis=1)
    merged = idxs.merge(cache, on="study_id", how="left")
    with_cache = S.evaluate_rules(merged, True)
    without = S.evaluate_rules(idxs, False)
    check("Kunstdaten: ohne Cache R02 leer", without["R02_gt"].isna().all())
    check("Kunstdaten: mit Cache Zeile 2 Sokolow-Lyon positiv, Zeile 1 negativ", with_cache["R02_gt"].iloc[1] == 1.0 and with_cache["R02_gt"].iloc[0] == 0.0)
    check("Kunstdaten: schlechte Signalqualitaet (Zeile 3) bleibt leer", np.isnan(with_cache["R02_gt"].iloc[2]))
    check("Kunstdaten: Q-Zacken-Flag Zeile 4 positiv (R14_a155)", with_cache["R14_a155"].iloc[3] == 1.0)
    check("Kunstdaten: Rhythmusregel R08 ohne Cache gleich mit Cache", frames_exact_equal(with_cache[["R08_baz"]], without[["R08_baz"]]) == [])

    feat_na = feat.copy()
    feat_na["signal_quality_usable"] = pd.array([True, None, False], dtype="boolean")
    pq_na = tmp / "na.parquet"
    feat_na.to_parquet(pq_na)
    ln = lade_merkmalsdatei(pq_na, idx)
    check("Merkmalsdatei: leere Signalqualitaet wird zu NaN (wie fehlende Cache-Zeile)", ln["signal_quality_usable"].isna().iloc[1] and ln["signal_quality_usable"].iloc[0] == True)  # noqa: E712
    mrg = idxs.iloc[:3].merge(pd.DataFrame({"study_id": [0, 1, 2], "signal_quality_usable": ln["signal_quality_usable"].to_numpy()}), on="study_id", how="left")
    try:
        S.evaluate_rules(mrg.assign(**{c: np.nan for c in S.CACHE_COLS if c != "signal_quality_usable"}), True)
        check("evaluate_rules verarbeitet NaN in Signalqualitaet", True)
    except Exception:
        check("evaluate_rules verarbeitet NaN in Signalqualitaet", False)

    # 4. Teildateien: Schreiben, Fortsetzen, Zusammenfuegen
    parts = tmp / "teile"
    parts.mkdir()
    buf = [{k: (i if k == "study_id" else (None if k in ("error_type",) else ("ok" if k == "extraction_status" else (True if k.endswith("_positive") or k == "signal_quality_usable" else 1.5)))) for k in KEEP_COLS} for i in range(5)]
    _schreibe_teil(parts / "part-00001.parquet", buf[:3])
    _schreibe_teil(parts / "part-00002.parquet", buf[3:])
    info = fuege_teile_zusammen(parts, tmp / "z.parquet", list(range(5)))
    check("Teile zusammengefuegt (5 Zeilen)", info["zeilen"] == 5)
    try:
        fuege_teile_zusammen(parts, tmp / "z2.parquet", list(range(6)))
        check("Teile: fehlende Menge abgelehnt", False)
    except SystemExit:
        check("Teile: fehlende Menge abgelehnt", True)
    _schreibe_teil(parts / "part-00003.parquet", buf[:1])
    try:
        fuege_teile_zusammen(parts, tmp / "z3.parquet")
        check("Teile: Doppelte abgelehnt", False)
    except SystemExit:
        check("Teile: Doppelte abgelehnt", True)

    # 5. Manifest: Verifikation erkennt Aenderung
    f1 = tmp / "a.txt"
    f1.write_text("x", encoding="utf-8")
    m = tmp / "m.json"
    m.write_text(json.dumps({str(f1): sha256_file(f1)}), encoding="utf-8")
    check("Manifest gueltig", G.verify_manifest(m) == [])
    f1.write_text("y", encoding="utf-8")
    check("Manifest: Aenderung erkannt", G.verify_manifest(m) != [])

    # 6. Fehleranteil: negative Tests (Grenze, Status, Fortsetzung ueber mehrere Teile)
    check("Fehleranteil genau 1,0 % wird nicht gesperrt", abs(pruefe_fehleranteil({"ok": 990, "error": 10}) - 0.01) < 1e-12)
    for nm, cnt_ in (("ueber 1,0 % (1,01)", {"ok": 9899, "error": 101}), ("unbekannter Status zaehlt als Fehler", {"ok": 980, "weird": 20}),
                     ("fehlender Status zaehlt als Fehler", {"ok": 980, None: 20}), ("leere Menge", {})):
        sp = tmp / f"sperre_{abs(hash(nm))}.json"
        try:
            pruefe_fehleranteil(cnt_, sp)
            check(f"Fehleranteil {nm}: gesperrt", False)
        except SystemExit:
            check(f"Fehleranteil {nm}: gesperrt, Sperrdatei geschrieben", sp.exists())

    def zeile(i, status):
        return {k: (i if k == "study_id" else (status if k == "extraction_status" else (None if k == "error_type" else (
            True if k.endswith("_positive") or k == "signal_quality_usable" else 1.5)))) for k in KEEP_COLS}

    fp = tmp / "fort"
    fp.mkdir()
    _schreibe_teil(fp / "part-00001.parquet", [zeile(i, "error" if i < 10 else "ok") for i in range(200)])  # frueherer Lauf: 5 % Fehler
    _schreibe_teil(fp / "part-00002.parquet", [zeile(1000 + i, "ok") for i in range(600)])  # Fortsetzung: sauber
    cnt_new_only = statusverteilung_dateien([fp / "part-00002.parquet"])
    cnt_all = statusverteilung_dateien(sorted(fp.glob("part-*.parquet")))
    check("Fortsetzung: nur neu gerechneter Teil wuerde den Fehler verdecken", fehleranteil(cnt_new_only)[2] == 0.0)
    try:
        pruefe_fehleranteil(cnt_all, fp / SPERRDATEI)
        check("Fortsetzung: Fehleranteil ueber alle Teile (1,25 %) gesperrt", False)
    except SystemExit:
        check("Fortsetzung: Fehleranteil ueber alle Teile (1,25 %) gesperrt", (fp / SPERRDATEI).exists())
    try:
        pruefe_keine_sperre(fp)
        check("Sperrdatei blockiert jede weitere Aufbereitung", False)
    except SystemExit:
        check("Sperrdatei blockiert jede weitere Aufbereitung", True)

    # 7. Spaltenvergleich: ausdrueckliche Liste, alles andere exakt, auch leere Spalten und Schema
    cols_all = list(AENDERBAR) + list(ZURUECKSETZEN) + ["R08_baz", "qtc_baz", "R14_o2", "leer_unabh"]
    old = pd.DataFrame({c: np.full(5, np.nan) for c in cols_all})
    old["R08_baz"] = [0, 1, 0, np.nan, 1.0]
    old["qtc_baz"] = [430.0, 510, 440, np.nan, 520]
    old["R14_o2"] = [0.0, 0, 1, 0, 0]
    new = old.copy()
    for c in AENDERBAR:
        new[c] = [0.0, 1, 0, np.nan, 1]
    check("Vergleich: korrekt befuellte Tabelle akzeptiert", vergleiche_regeltabelle_test(new, old) == [])
    neg = {}
    n2 = new.copy(); n2["leer_unabh"] = [1.0, 0, 0, 0, 0]; neg["zuvor vollstaendig leere, nicht gelistete Spalte erhaelt Werte"] = n2
    n3 = new.copy(); n3.loc[0, "qtc_baz"] = 431.0; neg["gefuellte Spalte veraendert"] = n3
    n4 = new.copy(); n4["R09_a155"] = [1.0, 0, 0, 0, 0]; neg["Spalte einer nicht gerechneten Einheit (R09) veraendert"] = n4
    neg["Spalte umbenannt (Schema)"] = new.rename(columns={"leer_unabh": "x"})
    neg["Spaltenreihenfolge (Schema)"] = new[list(reversed(list(new.columns)))]
    n7 = new.copy(); n7["R14_o2"] = n7["R14_o2"].astype("float32"); neg["Datentyp (Schema)"] = n7
    n8 = new.copy(); n8["R02_gt"] = np.nan; neg["aenderbare Spalte nach Extraktion vollstaendig leer"] = n8
    n9 = new.copy(); n9.loc[1, "R02_gt"] = 2.0; neg["Wert ausserhalb 0 und 1"] = n9
    neg["Zeilenzahl"] = new.iloc[:4]
    for nm, df_ in neg.items():
        check(f"Vergleich erkennt: {nm}", vergleiche_regeltabelle_test(df_, old) != [])
    check("Liste der aenderbaren Spalten: 19 Eintraege ohne Doppelte, keine Ueberschneidung mit R09/R18/R19-Spalten",
          len(set(AENDERBAR)) == 19 and not set(AENDERBAR) & set(ZURUECKSETZEN))

    # 8. Einheitenlauf: Umfang erzwungen; Ergebnisordner
    check("Einheiten Test: genau die sechs akzeptiert", erzwinge_einheiten(",".join(SIX), "test") == SIX)
    for nm, u_ in (("mit R09", ",".join(SIX) + ",R09"), ("mit R19", ",".join(SIX) + ",R19"), ("mit R18", ",".join(SIX) + ",R18"),
                   ("Teilmenge", "R02,R03"), ("R09 allein", "R09"), ("R08 und R20", "R08,R20")):
        try:
            erzwinge_einheiten(u_, "test")
            check(f"Einheiten Test {nm}: abgelehnt", False)
        except SystemExit:
            check(f"Einheiten Test {nm}: abgelehnt", True)
    try:
        erzwinge_einheiten("R07,R09", "dev")
        check("Einheiten Dev mit R09: abgelehnt", False)
    except SystemExit:
        check("Einheiten Dev mit R09: abgelehnt", True)
    ud = tmp / "units"
    ud.mkdir()
    for u in SIX:
        (ud / f"{u}.json").write_text("{}", encoding="utf-8")
    pruefe_ergebnisordner(ud, SIX, vollstaendig=True)
    check("Ergebnisordner mit genau den sechs Einheiten akzeptiert", True)
    (ud / "R09.json").write_text("{}", encoding="utf-8")
    try:
        pruefe_ergebnisordner(ud, SIX, vollstaendig=False)
        check("Ergebnisordner mit R09: abgelehnt", False)
    except SystemExit:
        check("Ergebnisordner mit R09: abgelehnt", True)
    (ud / "R09.json").unlink()
    (ud / "R14.json").unlink()
    try:
        pruefe_ergebnisordner(ud, SIX, vollstaendig=True)
        check("Ergebnisordner unvollstaendig: abgelehnt", False)
    except SystemExit:
        check("Ergebnisordner unvollstaendig: abgelehnt", True)

    # 9. Manifest M' wird vor dem ersten Testzugriff vollstaendig geprueft (Befehle brechen ab, bevor Waechter, Abfrage oder Start erreicht sind)
    calls: list = []
    g = globals()
    saved = {k: g[k] for k in ("master_zeilen", "args_guard", "beginne_schritt")}
    g["master_zeilen"] = lambda *a_, **k_: calls.append("master_zeilen")
    g["args_guard"] = lambda *a_, **k_: calls.append("args_guard") or {}
    g["beginne_schritt"] = lambda *a_, **k_: calls.append("beginne_schritt")
    saved_guard, saved_popen = S.guard_phase, None
    S.guard_phase = lambda *a_, **k_: calls.append("guard_phase")
    try:
        import subprocess
        saved_popen = subprocess.Popen
        subprocess.Popen = lambda *a_, **k_: calls.append("Popen")
        od = tmp / "mout"
        od.mkdir()
        fx = tmp / "fx.txt"
        fx.write_text("a", encoding="utf-8")
        (od / "manifest.json").write_text(json.dumps({str(fx): sha256_file(fx)}), encoding="utf-8")
        fx.write_text("b", encoding="utf-8")  # Hash veraendert
        for nm, fn, ns in (("extraktion", cmd_extraktion, SimpleNamespace(out=str(od), workers=1)),
                           ("aufbereiten-test", cmd_aufbereiten_test, SimpleNamespace(out=str(od))),
                           ("einheiten", cmd_einheiten, SimpleNamespace(out=str(od), modus="test", units=",".join(SIX))),
                           ("abschluss", cmd_abschluss, SimpleNamespace(out=str(od), modus="test", units=",".join(SIX), pub=str(tmp)))):
            calls.clear()
            try:
                fn(ns)
                check(f"{nm}: veraendertes Manifest bricht ab", False)
            except SystemExit:
                check(f"{nm}: veraendertes Manifest bricht ab, vor Waechter, Abfrage und Prozessstart", calls == [])
        (od / "manifest.json").unlink()
        calls.clear()
        try:
            cmd_extraktion(SimpleNamespace(out=str(od), workers=1))
            check("extraktion: fehlendes Manifest bricht ab", False)
        except SystemExit:
            check("extraktion: fehlendes Manifest bricht ab", calls == [])
    finally:
        for k, v in saved.items():
            g[k] = v
        S.guard_phase = saved_guard
        if saved_popen is not None:
            subprocess.Popen = saved_popen

    # 10. Waechter und technische Fortsetzung (continue_entry) wie in Skript v3.2
    reg = tmp / "reg.md"
    reg.write_text("", encoding="utf-8")
    hs = dict(spec_hash="s" * 64, script_hash="c" * 64, manifest_hash="m" * 64, dev_results_hash="d" * 64)
    ga_t = dict(register_path=str(reg), **hs)
    o2 = tmp / "o2"
    o2.mkdir()
    try:
        beginne_schritt(o2, ga_t, "x")
        check("Schritt ohne Eintrag Beginn: gesperrt", False)
    except PermissionError:
        check("Schritt ohne Eintrag Beginn: gesperrt", True)
    G.append_register(reg, G.register_line("Beginn", "T", hs["spec_hash"], hs["script_hash"], hs["manifest_hash"], hs["dev_results_hash"], "Test"))
    beginne_schritt(o2, ga_t, "schritt1")
    n_fort1 = sum(1 for ln in reg.read_text(encoding="utf-8").splitlines() if "| Fortsetzung:" in ln)
    beginne_schritt(o2, ga_t, "schritt2")
    n_fort2 = sum(1 for ln in reg.read_text(encoding="utf-8").splitlines() if "| Fortsetzung:" in ln)
    check("erster Schritt eines Durchgangs: keine Fortsetzungszeile", n_fort1 == 0)
    check("technische Fortsetzung: genau eine Zeile 'Fortsetzung' im Register", n_fort2 == 1)
    G.append_register(reg, G.register_line("Ende", "T", hs["spec_hash"], hs["script_hash"], hs["manifest_hash"], hs["dev_results_hash"], "Test", result_hash="e" * 64))
    try:
        beginne_schritt(o2, ga_t, "schritt3")
        check("nach Ende: weiterer Schritt gesperrt", False)
    except PermissionError:
        check("nach Ende: weiterer Schritt gesperrt", True)

    # 11. Typgerechter Wertevergleich (kein float): Zeichenketten, grosse Ganzzahlen, NaN-Positionen
    def eq(x, y):
        return bool(np.all(werte_gleich(np.asarray(x), np.asarray(y))))

    check("Wertevergleich: gleiche Zeichenketten gleich", eq(np.array(["1", "a"], dtype=object), np.array(["1", "a"], dtype=object)))
    check("Wertevergleich: 01 gegen 1 (Zeichenketten) verschieden", not eq(np.array(["01"], dtype=object), np.array(["1"], dtype=object)))
    check("Wertevergleich: Zeichenkette 1 gegen Zahl 1 verschieden", not eq(np.array(["1"], dtype=object), np.array([1], dtype=object)))
    check("Wertevergleich: 1 gegen 1.0 (verschiedener Typ) verschieden", not eq(np.array([1], dtype=object), np.array([1.0], dtype=object)))
    check("Wertevergleich: int64 2**53+1 gegen 2**53 verschieden (float wuerde gleich liefern)",
          float(2 ** 53 + 1) == float(2 ** 53) and not eq(np.array([2 ** 53 + 1], dtype="int64"), np.array([2 ** 53], dtype="int64")))
    check("Wertevergleich: int64 gleiche grosse Zahl gleich", eq(np.array([2 ** 62 + 7], dtype="int64"), np.array([2 ** 62 + 7], dtype="int64")))
    check("Wertevergleich: Python-Ganzzahlen ueber 64 Bit verschieden", not eq(np.array([2 ** 70 + 1], dtype=object), np.array([2 ** 70 + 2], dtype=object)))
    check("Wertevergleich: NaN-Positionen muessen gleich sein", eq(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
          and not eq(np.array([1.0, np.nan]), np.array([np.nan, 1.0])))
    check("Wertevergleich: fehlend gegen vorhanden verschieden (object)", not eq(np.array([None, "a"], dtype=object), np.array(["a", None], dtype=object)))
    check("Wertevergleich: None gleich NaN an gleicher Position (beide fehlend)", eq(np.array([None, "a"], dtype=object), np.array([np.nan, "a"], dtype=object)))
    check("Wertevergleich: float 0.1+0.2 gegen 0.3 verschieden (keine Toleranz)", not eq(np.array([0.1 + 0.2]), np.array([0.3])))
    dfa = pd.DataFrame({"s": pd.Series(["01", "x"], dtype=object), "n": np.array([2 ** 53 + 1, 5], dtype="int64")})
    dfb = pd.DataFrame({"s": pd.Series(["1", "x"], dtype=object), "n": np.array([2 ** 53, 5], dtype="int64")})
    pr_ = frames_exact_equal(dfa, dfb)
    check("frames_exact_equal erkennt 01 gegen 1 und grosse Ganzzahlen", len(pr_) == 2 and frames_exact_equal(dfa, dfa.copy()) == [])

    # 12. Abschluss: Sperrdatei und Fehleranteil werden vor assemble geprueft
    orig_saved = g["ORIG"]
    s_main_saved = S.main
    aufr: list = []
    try:
        g["ORIG"] = tmp / "orig"
        (tmp / "orig" / "test" / "work").mkdir(parents=True)
        leer = pd.DataFrame({c: np.full(3, np.nan) for c in ZURUECKSETZEN})
        leer.to_parquet(tmp / "orig" / "test" / "work" / "main_rules.parquet")
        S.main = lambda argv=None: aufr.append("assemble")

        def baue_out(name, status_liste, sperre):
            o = tmp / name
            (o / "test" / "work").mkdir(parents=True)
            (o / "test" / "merkmale").mkdir(parents=True)
            (o / "test" / "units").mkdir(parents=True)
            leer.to_parquet(o / "test" / "work" / "main_rules.parquet")
            (o / "test" / "work" / "aufbereitung_info.json").write_text("{}", encoding="utf-8")
            pd.DataFrame({"extraction_status": status_liste}).to_parquet(o / "test" / "merkmale" / "merkmale_test.parquet")
            for u in SIX:
                (o / "test" / "units" / f"{u}.json").write_text("{}", encoding="utf-8")
            fm = tmp / f"{name}_f.txt"
            fm.write_text("x", encoding="utf-8")
            (o / "manifest.json").write_text(json.dumps({str(fm): sha256_file(fm)}), encoding="utf-8")
            if sperre:
                (o / "test" / "merkmale" / SPERRDATEI).write_text("{}", encoding="utf-8")
            return o

        def ns_(o):
            return SimpleNamespace(out=str(o), modus="test", units=",".join(SIX), pub=str(tmp))

        for nm, o in (("Sperrdatei vorhanden", baue_out("abs1", ["ok"] * 100, True)),
                      ("Fehleranteil 2 % ohne Sperrdatei", baue_out("abs2", ["ok"] * 98 + ["error"] * 2, False))):
            aufr.clear()
            try:
                cmd_abschluss(ns_(o))
                check(f"Abschluss: {nm} gesperrt", False)
            except SystemExit:
                check(f"Abschluss: {nm} gesperrt, assemble nicht aufgerufen", aufr == [])
        check("Abschluss: bei Fehleranteil ueber 1 % wird die Sperrdatei geschrieben", (tmp / "abs2" / "test" / "merkmale" / SPERRDATEI).exists())
        aufr.clear()
        cmd_abschluss(ns_(baue_out("abs3", ["ok"] * 99 + ["error"], False)))
        check("Abschluss: sauberer Fall (1,0 % Fehler) ruft assemble auf", aufr == ["assemble"])
    finally:
        g["ORIG"] = orig_saved
        S.main = s_main_saved

    # 13. Skript v3.2 ist unveraendert
    check("Skript v3.2 bitgleich", sha256_file(Path(S.__file__)) == "1d40d782c65ceeffd4be9c4380e4eef822cbc98aaa710a4d7bf340699bf773e8")
    bad = [n for n, c in ok if not c]
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(ok) - len(bad)} von {len(ok)} Pruefungen bestanden")
    if bad:
        raise SystemExit("Selbsttest fehlgeschlagen: " + ", ".join(bad))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "aequivalenz-dev", "vergleich-einheiten", "extraktion-dev-probe", "manifest", "vorbereiten",
                                    "extraktion", "aufbereiten-test", "einheiten", "abschluss"])
    ap.add_argument("--out", default=str(DEFAULT_ERG))
    ap.add_argument("--units", default=",".join(SIX))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--modus", choices=["dev", "test"], default="test")
    ap.add_argument("--pub", default=str(PUB_ERG))
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "aequivalenz-dev":
        cmd_aequivalenz_dev(a)
    elif a.cmd == "vergleich-einheiten":
        cmd_vergleich_einheiten(a)
    elif a.cmd == "extraktion-dev-probe":
        cmd_extraktion_dev_probe(a)
    elif a.cmd == "manifest":
        cmd_manifest(a)
    elif a.cmd == "vorbereiten":
        cmd_vorbereiten(a)
    elif a.cmd == "extraktion":
        cmd_extraktion(a)
    elif a.cmd == "aufbereiten-test":
        cmd_aufbereiten_test(a)
    elif a.cmd == "einheiten":
        cmd_einheiten(a)
    elif a.cmd == "abschluss":
        cmd_abschluss(a)


if __name__ == "__main__":
    main()
