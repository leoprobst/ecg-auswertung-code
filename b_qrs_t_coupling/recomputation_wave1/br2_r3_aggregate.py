"""AW-241 B-R2/B-R3: Aggregation der Bootstrapergebnisse (nur Aggregate, keine Personenwerte).
Eingaben: <local path>{point.json, boot_<fold>.npz}; AW-204-Aggregat fuer Anker (d).
Ausgaben: nachrechnung/AW241/r2r3_*.csv und B_R2R3_ausgabe.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
sys.path.insert(0, str(data_path("b.auswertung")))
import numpy as np  # noqa: E402
import br2_r3_kern as K  # noqa: E402
import br2_r3_run as R  # noqa: E402

OUT = R.OUT
AGG = Path(str(data_path("b.nachrechnung_aw241")))
MIN = 950
FOLDS = R.FOLDS


def q(v, p=(0.025, 0.975)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, p, method="linear").tolist() if len(v) else [np.nan, np.nan]


def r(v, n=4):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def write(name, rows):
    with (AGG / name).open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    pt = json.loads((OUT / "point.json").read_text(encoding="utf8"))
    lines = ["AW-241 B-R2/B-R3 Aggregation (post hoc, Entwicklung). Skripthash br2_r3_aggregate.py: " + R.C.sha(__file__), ""]
    rows_anchor, rows_grid, rows_kipp, rows_box, rows_tip = [], [], [], [], []
    for fold in FOLDS:
        z = np.load(OUT / f"boot_{fold}.npz", allow_pickle=True)
        n = len(z["r"])
        meta = json.loads((OUT / f"boot_{fold}_meta.json").read_text(encoding="utf8"))
        P = pt["folds"][fold]
        lines.append(f"{fold}: gueltige Ziehungen {n} von {meta['attempted']}, Ausfaelle {meta['failures']}")
        # ---------------- Anker
        for ai, a in enumerate(R.ANCH):
            rel = z["a_rel"][:, ai, :]
            lo = z["a_logor"][:, ai]
            adm = np.isfinite(lo)
            orr = np.exp(lo[adm])
            pa = P["anchors"][a]
            row = dict(fold=fold, anker=a, gueltige_ziehungen=n,
                       R_A=r(pa["rel"][0]), R_A_lo=r(q(rel[:, 0])[0]), R_A_hi=r(q(rel[:, 0])[1]),
                       R_F=r(pa["rel"][1]), R_F_lo=r(q(rel[:, 1])[0]), R_F_hi=r(q(rel[:, 1])[1]),
                       rho=r(pa["rel"][2]), rho_lo=r(q(rel[:, 2])[0]), rho_hi=r(q(rel[:, 2])[1]),
                       OR_punkt=r(np.exp(pa["logor"]), 3) if pa["logor"] is not None else "n/a",
                       anteil_unzulaessig=r(1 - adm.mean(), 3),
                       OR_q025_zulaessig=r(np.quantile(orr, .025), 3) if adm.sum() else "n/a", OR_median_zulaessig=r(np.median(orr), 3) if adm.sum() else "n/a",
                       OR_q975_zulaessig=r(np.quantile(orr, .975), 3) if adm.sum() else "n/a",
                       anteil_OR_unter_1_25=r(np.mean(orr < 1.25) * adm.mean(), 4) if adm.sum() else "n/a", anteil_OR_unter_1_0=r(np.mean(orr < 1.0) * adm.mean(), 4) if adm.sum() else "n/a")
            for ti, t in enumerate(K.TARGETS):
                st = z["a_kst"][:, ai, ti]
                kra = z["a_kra"][:, ai, ti]
                bd = z["a_kbd"][:, ai, ti]
                cross = (st == 1) & np.isfinite(kra)
                row[f"kipp{t}_kreuzungen"] = int(cross.sum())
                row[f"kipp{t}_grenze_vor_kreuzung"] = int((st == 2).sum())
                row[f"kipp{t}_Punkt"] = f"{K.STATUS[pa['kipp'][str(t)][1]]}" + ("" if pa["kipp"][str(t)][0] is None else f" {pa['kipp'][str(t)][0]:.3f}")
                row[f"zulaessigkeitsgrenze_R_A_q025"] = r(q(bd)[0], 3)
                row[f"zulaessigkeitsgrenze_R_A_q975"] = r(q(bd)[1], 3)
                if cross.sum() >= MIN:
                    ci = q(kra[cross])
                    row[f"kipp{t}_R_A_intervall"] = f"{ci[0]:.3f}; {ci[1]:.3f}"
                else:
                    row[f"kipp{t}_R_A_intervall"] = "n/a (unter 950 Kreuzungen)"
            rows_anchor.append(row)
        # ---------------- Raster
        G = z["grid"].astype(float)          # [n, RF, RHO, RA]
        Lp = np.array(P["grid_logor"], float)
        for i, rf in enumerate(R.RF_R2):
            for j, rho in enumerate(R.RHO_R2):
                for k, ra in enumerate(R.RA_R2):
                    g = G[:, i, j, k]
                    adm = np.isfinite(g)
                    orr = np.exp(g[adm])
                    rows_grid.append(dict(fold=fold, R_A=ra, R_F=rf, rho=rho, OR_punkt=r(np.exp(Lp[i, j, k]), 3), zulaessig_anteil=r(adm.mean(), 3),
                                          OR_q025=r(np.quantile(orr, .025), 3) if adm.sum() >= 20 else "n/a", OR_q975=r(np.quantile(orr, .975), 3) if adm.sum() >= 20 else "n/a",
                                          anteil_unter_1_25=r(np.mean(orr < 1.25) * adm.mean(), 3), anteil_unter_1_0=r(np.mean(orr < 1.0) * adm.mean(), 3)))
        # ---------------- Kippkurve
        kpt = {t: P["kipp"][str(t)] for t in K.TARGETS}
        for ti, t in enumerate(K.TARGETS):
            for i, rf in enumerate(R.RF_R2):
                for j, rho in enumerate(R.RHO_R2):
                    st = z["kipp_st"][:, ti, i, j]
                    ra = z["kipp_ra"][:, ti, i, j].astype(float)
                    cross = (st == 1) & np.isfinite(ra)
                    ci = q(ra[cross]) if cross.sum() >= MIN else None
                    rows_kipp.append(dict(fold=fold, ziel=t, R_F=rf, rho=rho, R_A_kipp_punkt=r(kpt[t]["ra"][i][j], 3),
                                          status_punkt=K.STATUS[kpt[t]["status"][i][j]], zulaessigkeitsgrenze_punkt=r(kpt[t]["bound"][i][j], 3),
                                          kreuzungen=int(cross.sum()), grenze_vor_kreuzung=int((st == 2).sum()), unzulaessig_am_start=int((st == 3).sum()),
                                          R_A_intervall=f"{ci[0]:.3f}; {ci[1]:.3f}" if ci else "n/a"))
        # ---------------- Quader
        S = z["S"]; V = z["V"]; u = z["u"]
        for a in ("P_all", "EF_mix"):
            ai = R.ANCH.index(a)
            rel = z["a_rel"][:, ai, :]
            lo_ = [q(rel[:, c])[0] for c in range(3)]
            hi_ = [q(rel[:, c])[1] for c in range(3)]
            axes = [np.linspace(lo_[0], hi_[0], 5), np.linspace(lo_[1], hi_[1], 5), np.linspace(lo_[2], hi_[2], 5)]
            mins, inad = [], []
            for d in range(n):
                c = {"S": np.array([[S[d, 0], S[d, 1]], [S[d, 1], S[d, 2]]]), "V": np.array([[V[d, 0], V[d, 1]], [V[d, 1], V[d, 2]]]), "u": u[d]}
                g = K.grid(c, axes[0], axes[1], axes[2])
                ok = g["ok"]
                inad.append(1 - ok.mean())
                mins.append(np.exp(np.nanmin(g["logor"])) if ok.any() else np.nan)
            mins = np.array(mins)
            fin = np.isfinite(mins)
            c0 = {"S": np.array(P["S"]), "V": np.array(P["V"]), "u": np.array(P["u"])}
            g0 = K.grid(c0, axes[0], axes[1], axes[2])
            rows_box.append(dict(fold=fold, anker=a, R_A_box=f"{lo_[0]:.3f}; {hi_[0]:.3f}", R_F_box=f"{lo_[1]:.3f}; {hi_[1]:.3f}", rho_box=f"{lo_[2]:.3f}; {hi_[2]:.3f}",
                                 min_OR_punkt=r(np.exp(np.nanmin(g0["logor"])), 3) if g0["ok"].any() else "n/a", unzulaessige_gitterpunkte_punkt=r(1 - g0["ok"].mean(), 3),
                                 min_OR_q025=r(np.nanquantile(mins, .025), 3), min_OR_median=r(np.nanmedian(mins), 3), min_OR_q975=r(np.nanquantile(mins, .975), 3),
                                 anteil_ziehungen_ohne_zulaessigen_punkt=r(1 - fin.mean(), 3), anteil_ziehungen_min_unter_1_25=r(np.mean(mins[fin] < 1.25) * fin.mean(), 3),
                                 mittlerer_anteil_unzulaessiger_gitterpunkte=r(np.mean(inad), 3)))
        # ---------------- Kipppunkte B-R3
        T = z["tips"]                        # [n, nkeys, 3]
        gap = z["gap"]                       # [n, 4]
        for ki, key in enumerate(R.TIP_KEYS):
            kind, anc, path, extra, t = key
            st = T[:, ki, 0].astype(int)
            val = T[:, ki, 1]
            bd = T[:, ki, 2]
            cross = (st == 1) & np.isfinite(val)
            ptip = P["tips"]["|".join(map(str, key))]
            row = dict(fold=fold, modell=kind, anker=anc, pfad=path, zusatz=extra, ziel=t,
                       punkt_status=K.STATUS[ptip[0]], punkt_wert=r(ptip[1], 3), punkt_grenze=r(ptip[2], 3),
                       kreuzungen=int(cross.sum()), grenze_vor_kreuzung=int((st == 2).sum()), keine_kreuzung=int((st == 5).sum()), sonstige=int(np.isin(st, [3, 4]).sum()), gueltige_ziehungen=n)
            if cross.sum() >= MIN:
                ci = q(val[cross])
                row["wert_intervall"] = f"{ci[0]:.3f}; {ci[1]:.3f}"
            else:
                row["wert_intervall"] = "n/a (unter 950 Kreuzungen)"
            bnd = np.isfinite(bd)
            row["grenze_q025"] = r(np.quantile(bd[bnd], .025), 3) if bnd.sum() >= 20 else "n/a"
            row["grenze_q975"] = r(np.quantile(bd[bnd], .975), 3) if bnd.sum() >= 20 else "n/a"
            if kind == "shift" and path in ("A", "F"):
                dg = gap[:, 0] if path == "A" else gap[:, 1]
                frac = val / dg
                row["anteil_am_beobachteten_abstand_punkt"] = r(ptip[1] / P["gap"][0 if path == "A" else 1], 3) if ptip[1] is not None else "n/a"
                row["anteil_am_abstand_intervall"] = (f"{q(frac[cross])[0]:.3f}; {q(frac[cross])[1]:.3f}" if cross.sum() >= MIN else "n/a")
            rows_tip.append(row)
        lines.append(f"  beobachteter Abstand Fall-Kontrolle (Kontroll-SD): Winkel {P['gap'][0]:.3f}, Marker {P['gap'][1]:.3f}")
    write("r2r3_anker.csv", rows_anchor)
    write("r2r3_raster.csv", rows_grid)
    write("r2r3_kippkurve.csv", rows_kipp)
    write("r2r3_quader.csv", rows_box)
    write("r2r3_kipppunkte_B_R3.csv", rows_tip)
    # Punktraster 2D
    rows2 = []
    for fold in FOLDS:
        P = pt["folds"][fold]
        for a in R.SHIFT_ANCH:
            M = np.array(P["shift2d_" + a], float)
            for i, bA in enumerate(R.B2D):
                for j, bF in enumerate(R.B2D):
                    rows2.append(dict(fold=fold, modell="shift2d", anker=a, b_A=bA, b_F=bF, OR_korrigiert=r(np.exp(M[i, j]), 3)))
        Km = np.array(P["kappa2d"], float)
        for i, ka in enumerate(R.K2D):
            for j, kf in enumerate(R.K2D):
                rows2.append(dict(fold=fold, modell="kappa2d", anker="EF_mix", b_A=ka, b_F=kf, OR_korrigiert=r(np.exp(Km[i, j]), 3)))
    write("r2r3_punktraster_B_R3.csv", rows2)
    # Anker (d): Amplitudenmechanismus
    t = json.loads(Path(str(data_path("b.aw204_toleranz_aggregate_json"))).read_text(encoding="utf8"))
    sd_dev = 46.8418
    lines.append("")
    lines.append("Anker (d), Amplitudenmechanismus (AW-204, ALL12, Winkel, vorzeichenbehafteter Mittelwert der Aenderung; SE = SD der Differenzen / sqrt(n)):")
    for key, pct in (("ALL12_f1010", 1), ("ALL12_f1030", 3), ("ALL12_f1050", 5)):
        a = t["scenarios"][key]["spatial_qrs_t_angle_deg"]
        se = a["sd_delta"] / np.sqrt(a["n_pairs"])
        lines.append(f"  +{pct} %: mittlere Aenderung {a['mean']:.3f} Grad (SE {se:.3f}; 95 % {a['mean'] - 1.96 * se:.2f} bis {a['mean'] + 1.96 * se:.2f}), RMS {a['rms']:.2f} Grad, n {a['n_pairs']}; "
                     f"in Entwicklungs-SD des Winkels: {a['mean'] / sd_dev:.4f} (95 % {(a['mean'] - 1.96 * se) / sd_dev:.4f} bis {(a['mean'] + 1.96 * se) / sd_dev:.4f})")
    (AGG / "B_R2R3_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
