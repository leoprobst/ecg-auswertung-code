"""Supplementary Figure S4 of the program comparison study (plain matplotlib style).

Usage: python -I -B abbildung_S4_pub12_v10.py

Content: difference repeat minus program change (percentage points) with common bootstrap intervals, exactly one pair per person (246 pairs/persons for QRS, 231 for QTc),
six combinations of 12SL, Uni-G and BRAVEHEART, QRS at 120, 130 and 150 ms and QTc 450 ms (Fridericia for all).
Data: the existing aggregate table of the single-pair differences (nothing recomputed). Every plotted number is checked by assert against the difference rows of the supplementary table
on repeat and between-program discordance, and so are the findings stated in the manuscript (-2.6 [-4.7; -0.8] at 120 ms, +1.4 [0.2; 2.8] at 130 ms, all 150 ms intervals contain 0 or end at 0.0).
Output: PDF, SVG and PNG (600 dpi) in the figure folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # noqa: E402

HIER = Path(str(data_path("pub12.publikationen_pub_12_etablierung")))
OUT = HIER / "abbildungen" / "v9"

# Plain matplotlib style (white background, DejaVu Sans, readable sizes)
PETROL = "#1F5E73"
ROST = "#B5532A"
ANTHRAZIT = "#222222"
BREITE_ZWEISPALTIG = 175 / 25.4   # two-column width in inches
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.frameon": False, "axes.edgecolor": ANTHRAZIT, "axes.labelcolor": ANTHRAZIT,
    "xtick.color": ANTHRAZIT, "ytick.color": ANTHRAZIT, "text.color": ANTHRAZIT,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "errorbar.capsize": 2,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
plt.rcParams["legend.fontsize"] = 7
P, R = PETROL, ROST


def panel(ax, buchstabe):
    ax.text(-0.02, 1.02, buchstabe, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=ANTHRAZIT)


def speichern(fig, stamm):
    stamm = Path(stamm)
    stamm.parent.mkdir(parents=True, exist_ok=True)
    for endung in ("pdf", "svg", "png"):
        fig.savefig(stamm.with_suffix("." + endung))
    plt.close(fig)


PAARE = [("BRAVEHEART", "12SL"), ("Uni-G", "12SL"), ("Uni-G", "BRAVEHEART")]   # (candidate, comparator) as in the table "X against Y"
PANELS = [("qrs", 120, "QRS at 120 ms"), ("qrs", 130, "QRS at 130 ms"), ("qrs", 150, "QRS at 150 ms"), ("qtc_einh", 450, "QTc at 450 ms")]
XLIM = (-6.0, 14.0)


def r1(x):
    return round(float(x) * 100 + 0.0, 1) + 0.0


def lese_tabelle_s28e():
    """Difference rows of the supplementary table: (measure, threshold, repeat_of, between) -> (diff, lo, hi) as numbers."""
    t = (HIER / "SUPPLEMENT_PUB12_v9.md").read_text(encoding="utf-8")
    kopf = "#### Difference of repeat and between-program discordance on the same pairs"
    assert t.count(kopf) == 1
    ab = t.split(kopf)[1].split("Result.")[0]
    out = {}
    for z in ab.splitlines():
        m = re.match(r"\| (QRS|QTc, Fridericia for all) \| (\d+) \| Repeat of (\S+) \| (.+?) \| [\d.]+ \| [\d.]+ \| (-?[\d.]+) \((-?[\d.]+) to (-?[\d.]+)\) \|", z)
        if m:
            out[("qrs" if m.group(1) == "QRS" else "qtc_einh", int(m.group(2)), m.group(3), m.group(4))] = (float(m.group(5)), float(m.group(6)), float(m.group(7)))
    assert len(out) == 24, len(out)
    return out


def main():
    d = pd.read_csv(HIER / "ergebnisse" / "aw247" / "einpaar_differenz.csv")
    s28e = lese_tabelle_s28e()
    # y positions: per program pair one header row and two rows (repeat of the two programs)
    zeilen = []
    y = 0
    for a, b in PAARE:
        zeilen.append(("kopf", f"{a} against {b}", y))
        y += 1
        for rp in (a, b):
            zeilen.append(("zeile", rp, y, (a, b)))
            y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 4, figsize=(BREITE_ZWEISPALTIG, 3.3), sharey=False, gridspec_kw=dict(wspace=0.10))
    gedruckt = {}
    for k, (mass, schw, titel) in enumerate(PANELS):
        ax = axes[k]
        for z in zeilen:
            if z[0] == "kopf":
                continue
            _, rp, yy, (a, b) = z
            sub = d[(d.mass == mass) & (d.schwelle == schw) & (d.programm_wiederholung == rp) & (d.paar_zwischen == f"{a} gegen {b}")]
            assert len(sub) == 1, (mass, schw, rp, a, b, len(sub))
            r = sub.iloc[0]
            est, lo, hi = 100 * r.differenz_wiederholung_minus_zwischen, 100 * r.ki_unten, 100 * r.ki_oben
            assert abs(100 * (r.D_wiederholung - r.D_zwischen) - est) < 1e-3
            gedruckt[(mass, schw, rp, f"{a} against {b}")] = (r1(r.differenz_wiederholung_minus_zwischen), r1(r.ki_unten), r1(r.ki_oben))
            ax.plot([lo, hi], [yy, yy], color=P, lw=1.0)
            ax.plot(est, yy, "o", color=P, ms=4.2, zorder=3)
        ax.axvline(0, color=R, lw=0.9, ls="--", zorder=1)
        ax.set_xlim(*XLIM)
        ax.set_ylim(y - 0.6, -0.8)
        ax.set_xticks([-5, 0, 5, 10])
        ax.text(0.0, 1.02, titel, transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom", fontweight="bold")
        panel(ax, "ABCD"[k])
        if k == 0:
            ax.set_yticks([z[2] for z in zeilen])
            ax.set_yticklabels([z[1] if z[0] == "kopf" else f"   Repeat of {z[1]}" for z in zeilen])
            for lab, z in zip(ax.get_yticklabels(), zeilen):
                if z[0] == "kopf":
                    lab.set_fontweight("bold")
            ax.tick_params(axis="y", length=0)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        if k == 3:
            ax.text(XLIM[1], y - 0.35, "", fontsize=6)
    fig.text(0.63, 0.075, "Repeat minus program change (percentage points)", fontsize=8, ha="center", va="bottom")
    fig.text(0.63, 0.005, "Right of the dashed line: the repeat crosses the threshold more often than the program change.", fontsize=7, ha="center", va="bottom", color="#333333")
    fig.subplots_adjust(left=0.27, right=0.99, top=0.90, bottom=0.20)
    # ---- Checks against the supplementary table
    for (mass, schw, rp, paar), (e, lo, hi) in gedruckt.items():
        ex = s28e[(mass, schw, rp, paar)]
        assert (e, lo, hi) == ex, ((mass, schw, rp, paar), (e, lo, hi), ex)
    assert gedruckt[("qrs", 120, "BRAVEHEART", "Uni-G against BRAVEHEART")] == (-2.6, -4.7, -0.8)
    assert gedruckt[("qrs", 130, "BRAVEHEART", "BRAVEHEART against 12SL")] == (1.4, 0.2, 2.8)
    # QRS: except for these two, every interval contains 0 or ends at 0.0 (Results text, supplementary table)
    for (mass, schw, rp, paar), (e, lo, hi) in gedruckt.items():
        if mass != "qrs":
            continue
        if (schw, rp, paar) in ((120, "BRAVEHEART", "Uni-G against BRAVEHEART"), (130, "BRAVEHEART", "BRAVEHEART against 12SL")):
            assert lo > 0 or hi < 0
        else:
            assert lo <= 0.0 <= hi, ((mass, schw, rp, paar), lo, hi)
    for (mass, schw, rp, paar), (e, lo, hi) in gedruckt.items():
        if mass == "qrs" and schw == 150:
            assert lo <= 0.0 <= hi
    # QTc: the intervals that exclude 0 (repeat larger), for the legend
    ausschluss = sorted((rp, paar) for (mass, schw, rp, paar), (e, lo, hi) in gedruckt.items() if mass == "qtc_einh" and (lo > 0 or hi < 0))
    assert ausschluss == [("BRAVEHEART", "Uni-G against BRAVEHEART"), ("Uni-G", "Uni-G against 12SL"), ("Uni-G", "Uni-G against BRAVEHEART")], ausschluss
    assert set(d[d.mass == "qrs"].paare) == {246} and set(d[d.mass == "qtc_einh"].paare) == {231}
    assert len(gedruckt) == 24
    speichern(fig, OUT / "FigS4_differenz_wiederholung_programmwechsel")
    from PIL import Image
    w = Image.open(str(OUT / "FigS4_differenz_wiederholung_programmwechsel") + ".png").size[0] / 600 * 25.4
    assert w <= 175.5, w
    print(f"FigS4: {w:.1f} mm wide; 24 cells checked; QTc intervals without 0:", ausschluss)


if __name__ == "__main__":
    main()
