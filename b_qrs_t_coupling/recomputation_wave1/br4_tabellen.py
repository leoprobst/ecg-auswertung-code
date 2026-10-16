"""Tipping-point tables from the per-fold result files <fold>.json of the multiple-imputation analysis (aggregates only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import json
import sys
from pathlib import Path

OUT = Path(str(data_path("b.aw241_r4")))
AGG = Path(str(data_path("b.nachrechnung_aw241")))
sys.path.insert(0, str(data_path("b.auswertung")))


def fd(t, dcc):
    if t is None:
        return None, "n/a"
    if t["status"] == "Kreuzung":
        return t["d"], f"{t['d']:.2f}"
    if t["status"].startswith("bei delta=0"):
        return 0.0, "0 (schon bei delta=0 darunter)"
    return None, f"nicht erreicht bis 8 (OR bei 8: {t['or_at_8']:.2f})"


def main():
    rows_tip, rows_or = [], []
    lines = ["Output of the tipping-point tables (post hoc, development partition, aggregates only)."]
    for fold in ("train", "validation"):
        d = json.loads((OUT / f"r4_results_{fold}.json").read_text(encoding="utf8"))["folds"][fold]
        dcc = d["d_CC"]
        lines.append(f"{fold}: valid imputations {d['n_valid_imputations']}, rows {d['n_rows']}, imputed Frechet cells {d['imputed_F_cells']}, angle cells {d['imputed_A_cells']}, class both {d['class_both']}, Frechet only {d['class_fonly']}, d_CC {dcc:.3f}")
        for name, m in d["models"].items():
            o0 = m["or_delta0"]
            rows_or.append(dict(fold=fold, modell=name, pfad="delta=0", d=0, OR=round(o0["or"], 3), lo=round(o0["lo"], 3), hi=round(o0["hi"], 3)))
            for k, v in m["or_table"].items():
                path, dd = k.split("|")
                if dd == "0":
                    continue
                rows_or.append(dict(fold=fold, modell=name, pfad=path, d=dd, OR=round(v["or"], 3), lo=round(v["lo"], 3), hi=round(v["hi"], 3), anteil_fréchet_an_untergrenze=round(v["floor_share"], 3)))
            for t in m["tips"]:
                pv, ps = fd(t["point"], dcc)
                hv_, hs_ = fd(t["delta_low_at_upper_limit"], dcc)  # delta at which the UPPER Rubin limit reaches the target (larger delta)
                lv, ls = fd(t["delta_high_at_lower_limit"], dcc)  # delta at which the LOWER Rubin limit reaches the target (smaller delta)
                h1v, h1s = fd(t["half1"], dcc)
                h2v, h2s = fd(t["half2"], dcc)
                rows_tip.append(dict(fold=fold, modell=name, pfad=t["path"], ziel=t["target"], d_stern=ps, d_stern_in_vielfachem_dCC=(f"{pv / dcc:.2f}" if pv is not None else "n/a"),
                                     delta_stern_intervall_klein=ls, delta_stern_intervall_gross=hs_, hälfte_1=h1s, hälfte_2=h2s, OR_bei_delta0=round(o0["or"], 3)))
    for name, rows in (("b_r4_kipppunkte.csv", rows_tip), ("b_r4_or_gitter.csv", rows_or)):
        with (AGG / name).open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list({k: 1 for r in rows for k in r}.keys()))
            w.writeheader()
            w.writerows(rows)
    lines += [str(r) for r in rows_tip]
    (AGG / "B_R4_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
