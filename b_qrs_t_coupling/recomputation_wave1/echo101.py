"""AW-241 Pruefauftrag MIMIC-IV-ECHO 1.0.1: Gehen 'entfernte Studien mit Identifikatorkollision' in die LVEF-Labels oder die Verifikation von B ein?
Die Versionshinweise nennen die 15 Studien nicht (siehe Bericht Abschnitt 15). Lokal pruefbar sind:
 (1) welche Echo-Identifikatoren B verwendet (measurement_id der strukturierten Messungen) und ob es dort Kollisionen gibt;
 (2) Zahl der B-Personen, deren Referenz-Echo ein DICOM-Studienpendant in der lokalen Studienliste hat, und deren Zahl mit mehrfach belegter measurement_id;
 (3) Obergrenze der Wirkung auf die Haupt-OR, wenn bis zu 15 Personen entfernt wuerden (Einschritt-Naeherung der Einflusswerte, Worst Case).
Aufruf: python echo101.py dev | test.   Ausgabe nur Aggregate, Zellen unter 10 als '<10'. Keine Kennungen ausgeben.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
sys.path.insert(0, str(data_path("b.auswertung")))
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

AGG = Path(str(data_path("b.nachrechnung_aw241")))
SL = ("read_csv('" + str(data_path("b.echo_study_list_csv")) + "', header=true, all_varchar=true)")


def sup(n):
    return int(n) if n >= 10 else "<10"


def id_sets():
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'; SET threads=2")
    linked = {int(r[0]) for r in c.execute(f"SELECT DISTINCT measurement_id FROM {SL} WHERE measurement_id IS NOT NULL AND measurement_id<>''").fetchall()}
    shared = {int(r[0]) for r in c.execute(f"SELECT measurement_id FROM {SL} WHERE measurement_id IS NOT NULL AND measurement_id<>'' GROUP BY 1 HAVING count(DISTINCT study_id)>1").fetchall()}
    return linked, shared


def worst_case(z, y, k=15):
    """Einschritt-Naeherung: beta_(-i) = beta - H^-1 x_i (y_i - p_i) / (1 - h_i); Worst Case: die k Personen mit dem groessten Beitrag entfernen."""
    from werkzeuge.nachrechnung.core import logistic_fit
    w = np.ones(len(y))
    b = logistic_fit(z, y, w)
    p = expit(z @ b)
    pen = np.r_[0, np.full(z.shape[1] - 1, 1e-6)]
    H = z.T @ (z * (p * (1 - p))[:, None]) + np.diag(pen)
    Hi = np.linalg.inv(H)
    h = p * (1 - p) * np.einsum("ij,jk,ik->i", z, Hi, z)
    g = (z @ Hi[:, -1]) * (y - p) / (1 - h)         # Beitrag zur Markerkomponente
    top_dec = np.sort(g)[::-1][:k].sum()             # beta sinkt um diese Summe
    top_inc = np.sort(g)[:k].sum()                   # beta steigt um -diese Summe
    return dict(beta=float(b[-1]), or_=float(np.exp(b[-1])), or_min=float(np.exp(b[-1] - top_dec)), or_max=float(np.exp(b[-1] - top_inc)), k=k, n=len(y), max_leverage=float(h.max()))


def dev():
    from werkzeuge.b_test import aw211_common as C
    from werkzeuge.b_test.daten import classify_lvef
    from werkzeuge.b_test.statistik import raw_design, fixed_design
    linked, shared = id_sets()
    f, finfo, scale = C.outcome_frame()
    e, _ = C.read(C.POP, ["study_id", "nearest_tte_measurement_id", "eligible_reference", "eligible_B3", "split"])
    g = f.loc[f.eligible_B3.eq(True)].merge(e[["study_id", "nearest_tte_measurement_id"]], on="study_id", validate="one_to_one")
    out = {"linked_ids": len(linked), "shared_ids": len(shared)}
    mid = g.nearest_tte_measurement_id.to_numpy(float)
    assert np.isfinite(mid).all()
    mid = mid.astype(np.int64)
    for fold in ("train", "validation"):
        m = g.split.eq(fold).to_numpy()
        sub = g.loc[m]
        yy = classify_lvef(sub.preferred_lvef.to_numpy(float))
        raw, _ = raw_design(sub, C.F)
        z = fixed_design(raw, scale)
        mm = mid[m]
        ln = np.isin(mm, list(linked))
        sh = np.isin(mm, list(shared))
        out[fold] = dict(personen_B3=int(len(sub)), referenz_echo_mit_DICOM_studienpendant=sup(int(ln.sum())), referenz_echo_mit_mehrfach_belegter_measurement_id=sup(int(sh.sum())),
                         worst_case_15=worst_case(z, yy))
        # Gegenprobe: tatsaechliche Entfernung der 'verlinkten' Personen aendert die OR (nur zur Groessenordnung, kein Befund ueber die 15 Studien)
    (AGG / "ECHO101_dev_ausgabe.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out, indent=1))


def test():
    import loeser_s3b as L
    from werkzeuge.b_test import adapter, daten
    linked, shared = id_sets()
    frames, package = L.load_frames()
    f63 = frames["BEF-0063"]
    h1 = package["hypotheses"]["BEF-0063"]
    cap = {}
    orig = adapter.fit

    def cap_fit(z, y, subjects, direction=1, weights=None, wald=True):
        cap.update(z=np.asarray(z, float), y=np.asarray(y, float))
        return orig(z, y, subjects, direction, weights=weights, wald=wald)

    adapter.fit = cap_fit
    union = np.unique(f63.subject_id.to_numpy())
    res = adapter.estimate(f63, h1, union, bootstrap=False)
    adapter.fit = orig
    old = json.loads(L.AW142.read_text(encoding="utf8"))["results"]["BEF-0063"]
    assert np.isclose(res["or"], old["or"], rtol=1e-9) and res["N"] == old["N"]
    zz, good, _ = daten.masks(f63, h1["endpoint"], h1["marker"], h1["comparator"])
    mid = zz.loc[good, "nearest_tte_measurement_id"].to_numpy(float)
    assert len(mid) == len(cap["y"]) and np.isfinite(mid).all()
    mid = mid.astype(np.int64)
    ln, sh = np.isin(mid, list(linked)), np.isin(mid, list(shared))
    out = {"N": int(res["N"]), "or": float(res["or"]), "referenz_echo_mit_DICOM_studienpendant": sup(int(ln.sum())), "referenz_echo_mit_mehrfach_belegter_measurement_id": sup(int(sh.sum())),
           "worst_case_15": worst_case(cap["z"], cap["y"])}
    (AGG / "ECHO101_test_ausgabe.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
