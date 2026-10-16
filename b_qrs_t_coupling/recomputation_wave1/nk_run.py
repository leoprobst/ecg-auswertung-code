"""AW-241 Abschnitt 17 (Schwachstellenpruefung B v15, Eintrag 4): Negativkontroll-Diagnosen im B3-Modell.

Nachtraeglich (post hoc), nur Entwicklungsdaten (train, validation). Die Diagnoseliste und die Codes sind in NK_REGELN.md mit Zeitstempel und
Skripthash festgelegt, bevor irgendeine Diagnosezeile gelesen wurde. Das Skript prueft beim Start, dass Regeldatei und Skript zusammenpassen.

Codelogik wie H4 ("ICD in index encounter", labels.py:2704-2725): Praefixvergleich auf dem normalisierten Code (Grossbuchstaben, ohne Punkt) je ICD-Version;
Index-Aufenthalt = hadm_id des Index-EKG (ecg_master_index, COALESCE(Aufnahme-Zeitfenster, ED-hadm_id)); hadm_id NULL -> nicht verfuegbar (ausgeschlossen);
Code vorhanden -> positiv, sonst negativ.

Modell: B3-Design mit Marker (feste Entwicklungsskalierung, Ridge 1e-6, CR1-Wald) wie das Hauptmodell, nur anderer Endpunkt.

Aufruf:  python nk_run.py test   (Kunstdaten)   |   python nk_run.py run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import hashlib
import json
import os
import sys
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AGG = ROOT / "nachrechnung/AW241"
REGELN = AGG / "NK_REGELN.md"
MASTER = str(data_path("b.mimic_validation_master_ecg_master_index_parquet"))
DIAG = str(data_path("b.mimic_validation_standardized_diagnoses_icd_parquet"))
FOLDS = ["train", "validation"]
OR_PUB = {"train": 2.480, "validation": 2.392}
MIN_CASES = 100

# Reihenfolge wie von K festgelegt; Retinopathie und die LVEF-Haupt-OR werden daneben gestellt.
# Je Eintrag: v9/v10 = Fallpraefixe; family9/family10 = Familienpraefixe (nur Divertikulose: Personen mit anderen Familiencodes werden ausgeschlossen);
# men_only = nur Maenner.
RULES = {
    "cataract": dict(label="Cataract", v9=["366"], v10=["H25", "H26"]),
    "glaucoma": dict(label="Glaucoma", v9=["365"], v10=["H40", "H42"]),
    "hearing_loss": dict(label="Hearing loss", v9=["389"], v10=["H90", "H91"]),
    "inguinal_hernia": dict(label="Inguinal hernia", v9=["550"], v10=["K40"]),
    "diverticulosis_no_bleeding": dict(label="Diverticulosis without bleeding", v9=["56200", "56210"], v10=["K5710", "K5730", "K5750", "K5790"],
                                       family9=["562"], family10=["K57"]),
    "hemorrhoids": dict(label="Hemorrhoids", v9=["455"], v10=["K64"]),
    "knee_hip_osteoarthritis": dict(label="Knee or hip osteoarthritis",
                                    v9=["71515", "71516", "71525", "71526", "71535", "71536", "71595", "71596"], v10=["M16", "M17"]),
    "bph": dict(label="Benign prostatic hyperplasia (men only)", v9=["6000", "6001", "6002", "6009"], v10=["N40"], men_only=True),
}
BESIDE = {
    "retinopathy": dict(label="Retinopathy", v9=["3620", "3621"],
                        v10=["H350", "H360"] + [f"E{d}3{k}" for d in ("08", "09", "10", "11", "13") for k in "12345"]),
}


def rules_json():
    return json.dumps({"RULES": RULES, "BESIDE": BESIDE}, sort_keys=True, ensure_ascii=True)


def rules_sha():
    return hashlib.sha256(rules_json().encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(code):
    return str(code).upper().replace(".", "").strip()


def matches(code, version, p9, p10):
    c = norm(code)
    pre = p9 if int(version) == 9 else p10
    return any(c.startswith(p) for p in pre)


def flags_from_codes(pairs, diag, rules):
    """pairs: DataFrame(key, has_hadm bool, male bool); diag: DataFrame(key, icd_code, icd_version).
    Rueckgabe: je Regel Series mit 1.0 / 0.0 / NaN (nicht verfuegbar) in Reihenfolge von pairs."""
    out = {}
    d = diag.copy()
    d["_c"] = d.icd_code.map(norm)
    for name, r in rules.items():
        v9, v10 = r["v9"], r["v10"]
        d["_case"] = [matches(c, v, v9, v10) for c, v in zip(d.icd_code, d.icd_version)]
        case = set(d.loc[d._case, "key"])
        excl = set()
        if "family9" in r:
            d["_fam"] = [matches(c, v, r["family9"], r["family10"]) for c, v in zip(d.icd_code, d.icd_version)]
            other = d._fam & ~d._case
            excl = set(d.loc[other, "key"])            # Divertikulose-Familie mit Blutung/Divertikulitis/Perforation: weder Fall noch Kontrolle
            case = case - excl
        y = pairs.key.map(lambda k: 1.0 if k in case else 0.0).to_numpy(float)
        y[pairs.key.isin(excl).to_numpy()] = np.nan
        y[~pairs.has_hadm.to_numpy(bool)] = np.nan
        if r.get("men_only"):
            y[~pairs.male.to_numpy(bool)] = np.nan
        out[name] = y
    return out


def selftest():
    pairs = pd.DataFrame({"key": ["a", "b", "c", "d", "e", "f", "g"], "has_hadm": [1, 1, 1, 1, 1, 0, 1], "male": [1, 1, 0, 1, 1, 1, 0]}).astype({"has_hadm": bool, "male": bool})
    diag = pd.DataFrame([
        ("a", "H25.11", 10), ("a", "I10", 10),          # Katarakt (ICD-10)
        ("b", "36610", 9), ("b", "56210", 9),           # Katarakt (ICD-9), Divertikulose ohne Blutung
        ("c", "K5731", 10), ("c", "K5730", 10),         # Divertikulose mit Blutung und ohne: ausgeschlossen
        ("d", "N400", 10), ("d", "56211", 9),           # BPH; Divertikulitis ICD-9 -> ausgeschlossen fuer Divertikulose
        ("e", "M170", 10), ("e", "E1131", 10), ("e", "5504", 9),   # Gonarthrose, diabetische Retinopathie, ICD-9-Code in ICD-10-Aufenthalt zaehlt nicht fuer 550 (Version 9 -> 550 ja)
        ("f", "H25", 10),                                 # hadm_id fehlt -> NaN
        ("g", "N40", 10), ("g", "71516", 9), ("g", "40100", 9)])
    diag.columns = ["key", "icd_code", "icd_version"]
    fl = flags_from_codes(pairs, diag, {**RULES, **BESIDE})
    exp = {
        "cataract": [1, 1, 0, 0, 0, np.nan, 0], "diverticulosis_no_bleeding": [0, 1, np.nan, np.nan, 0, np.nan, 0],
        "bph": [0, 0, np.nan, 1, 0, np.nan, np.nan], "knee_hip_osteoarthritis": [0, 0, 0, 0, 1, np.nan, 1],
        "retinopathy": [0, 0, 0, 0, 1, np.nan, 0], "inguinal_hernia": [0, 0, 0, 0, 1, np.nan, 0],
    }
    for k, v in exp.items():
        got = fl[k]
        assert np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(np.array(v, float), nan=-1)), (k, got, v)
    # Prefixlogik: 'H250' darf nicht auf H26 passen, ICD-9 '366' nicht auf Version 10
    assert not matches("366", 10, ["366"], ["H25"]) and matches("H26.9", 10, ["366"], ["H25", "H26"])
    print("Selbsttest bestanden")


def check_rules_file():
    txt = REGELN.read_text(encoding="utf8")
    me = sha_file(__file__)
    assert f"Skript-SHA256: {me}" in txt, "Regeldatei gehoert nicht zu diesem Skript (Hash; Nachtrag-Zeilen zaehlen als Skript-SHA256)"
    assert f"Regelwerk-SHA256: {rules_sha()}" in txt, "Regelwerk in Regeldatei weicht ab"
    stamp = [ln for ln in txt.splitlines() if ln.startswith("Zeitstempel (UTC):")]
    assert stamp, "Zeitstempel fehlt"
    return me, stamp[0]


def main():
    from werkzeuge.b_test import aw211_common as C
    from werkzeuge.b_test.daten import classify_lvef
    from werkzeuge.b_test.statistik import raw_design, fixed_design, fit
    import duckdb
    me, stamp = check_rules_file()
    C.guard()
    f, info, scale = C.outcome_frame()
    assert len(f) == 129198 and set(f.split) == set(FOLDS)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='1GB'")
    con.execute("PRAGMA threads=1")
    ids = f[["study_id"]].drop_duplicates()
    con.register("ids", ids)
    m = con.execute(f"SELECT study_id, hadm_id FROM read_parquet('{MASTER}') WHERE split IN ('train','validation') AND study_id IN (SELECT study_id FROM ids)").df()
    assert m.study_id.is_unique and len(m) == len(ids) == len(f), "Index-EKG nicht eindeutig im Masterindex"
    f = f.merge(m, on="study_id", validate="one_to_one")
    f["has_hadm"] = f.hadm_id.notna()
    f["key"] = f.subject_id.astype("int64").astype(str) + "|" + f.hadm_id.fillna(-1).astype("int64").astype(str)
    pairs = f.loc[f.has_hadm, ["subject_id", "hadm_id"]].drop_duplicates().astype("int64")
    con.register("pairs", pairs)
    diag = con.execute(f"SELECT d.subject_id, d.hadm_id, d.icd_code, d.icd_version FROM read_parquet('{DIAG}') d JOIN pairs p ON d.subject_id=p.subject_id AND d.hadm_id=p.hadm_id").df()
    diag["key"] = diag.subject_id.astype("int64").astype(str) + "|" + diag.hadm_id.astype("int64").astype(str)
    sx = con.execute(f"SELECT study_id, sex FROM read_parquet('{MASTER}') WHERE split IN ('train','validation') AND study_id IN (SELECT study_id FROM ids)").df()
    f = f.merge(sx, on="study_id", validate="one_to_one")
    assert set(f.sex.unique()) <= {"M", "F"}
    fp = pd.DataFrame({"key": f.key.to_numpy(), "has_hadm": f.has_hadm.to_numpy(), "male": (f.sex == "M").to_numpy()})
    assert (pd.crosstab(f.sex, f.sex_numeric).to_numpy() > 0).sum() == 2, "sex_numeric nicht eindeutig an M/F gekoppelt"
    flags = flags_from_codes(fp, diag, {**RULES, **BESIDE})

    raw, names = raw_design(f, C.F)
    complete = np.isfinite(raw).all(1)
    elig_b3 = f.eligible_B3.to_numpy(bool)
    assert ((f.eligible_reference.to_numpy(bool) & complete) == elig_b3).all()
    z_all = np.full((len(f), len(names) + 1), np.nan)
    z_all[complete] = fixed_design(raw[complete], scale)
    y_lvef = np.full(len(f), np.nan)
    er = f.eligible_reference.to_numpy(bool)
    y_lvef[er] = classify_lvef(f.preferred_lvef.to_numpy(float)[er])
    subj = f.subject_id.to_numpy()

    def run_one(y, mask, fold):
        fm = mask & (f.split.isin(FOLDS) if fold == "pooled" else f.split.eq(fold)).to_numpy() & np.isfinite(y)
        n = int(fm.sum())
        cases = int((y[fm] == 1).sum())
        ctr = n - cases
        rec = dict(n=n, cases=cases, controls=ctr, prevalence=cases / n if n else None)
        if cases < MIN_CASES or ctr < MIN_CASES:
            rec["status"] = "descriptive_only"
            return rec
        try:
            zz = z_all[fm]
            const = [j for j in range(1, zz.shape[1] - 1) if not np.ptp(zz[:, j]) > 1e-9]    # Nachtrag: konstante Nuisance-Spalten (z. B. Geschlecht bei nur Maennern) entfernen
            assert zz[:, -1].std() > 0
            if const:
                zz = np.delete(zz, const, axis=1)
                rec["dropped_constant_columns"] = [names[j - 1] for j in const]
            r = fit(zz, y[fm], subj[fm])
            rec.update(status="ok", OR=r["or"], CI=r["or_ci95_wald"], N=r["N"], p=r["p_raw"], delta_auc=r["delta_auc"],
                       class_persons=r["class_persons"])
        except ValueError as ex:
            rec.update(status="fit_failed", reason=str(ex)[:80])
        return rec

    res = {"script_sha256": me, "regeln": stamp, "regelwerk_sha256": rules_sha(), "population_primary": "eligible_B3 (Hauptanalysemenge)",
           "population_secondary": "alle Entwicklungspersonen mit vollstaendigem B3-Design (Index-EKG), verifiziert oder nicht", "main": {}, "primary": {}, "secondary": {}, "flow": {}}
    for fold in FOLDS:
        mk = elig_b3 & f.split.eq(fold).to_numpy()
        r = run_one(y_lvef, elig_b3, fold)
        assert abs(r["OR"] - OR_PUB[fold]) < 0.002, f"Haupt-OR weicht ab: {r['OR']}"
        res["main"][fold] = r
        res.setdefault("main_hadm", {})[fold] = run_one(y_lvef, elig_b3 & f.has_hadm.to_numpy(), fold)
        res["flow"][fold] = dict(index_persons=int(f.split.eq(fold).sum()), complete=int((complete & f.split.eq(fold).to_numpy()).sum()), eligible_B3=int(mk.sum()),
                                 eligible_B3_with_hadm=int((mk & f.has_hadm.to_numpy()).sum()), complete_with_hadm=int((complete & f.has_hadm.to_numpy() & f.split.eq(fold).to_numpy()).sum()))
    res["main_hadm"]["pooled"] = run_one(y_lvef, elig_b3 & f.has_hadm.to_numpy(), "pooled")
    res["main"]["pooled"] = run_one(y_lvef, elig_b3, "pooled")
    for name in list(RULES) + list(BESIDE):
        y = flags[name]
        res["primary"][name] = {fold: run_one(y, elig_b3, fold) for fold in FOLDS + ["pooled"]}
        res["secondary"][name] = {fold: run_one(y, complete, fold) for fold in FOLDS + ["pooled"]}
        # Ausschluesse je Regel (nicht verfuegbar) in der Primaerpopulation
        res["primary"][name]["unavailable_excluded"] = {fold: int((elig_b3 & f.split.eq(fold).to_numpy() & ~np.isfinite(y)).sum()) for fold in FOLDS}
    C.dump(AGG / "NK_ergebnis.json", res)
    lines = ["AW-241 Abschnitt 17 Negativkontrollen (Entwicklung, nur Aggregate)", "Skripthash nk_run.py: " + me, stamp, ""]
    for pop in ("primary", "secondary"):
        lines.append(f"[{pop}]")
        for name in list(RULES) + list(BESIDE):
            for fold in FOLDS + ["pooled"]:
                r = res[pop][name][fold]
                s = f"{name:28s} {fold:10s} n={r['n']} cases={r['cases']} status={r['status']}" + (f" dropped={r['dropped_constant_columns']}" if r.get('dropped_constant_columns') else "")
                if r["status"] == "ok":
                    s += f" OR={r['OR']:.3f} [{r['CI'][0]:.3f};{r['CI'][1]:.3f}]"
                lines.append(s)
        lines.append("")
    for key in ("main", "main_hadm"):
        for fold in FOLDS + ["pooled"]:
            r = res[key][fold]
            lines.append(f"LVEF-Haupt [{key}] {fold}: OR={r['OR']:.3f} [{r['CI'][0]:.3f};{r['CI'][1]:.3f}] n={r['n']} cases={r['cases']}")
    (AGG / "NK_ausgabe_roh.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        main()
    else:
        raise SystemExit("Aufruf: nk_run.py test|run")
