"""Figure 1 of the time-alignment note: time offset, heart-rate error, identifiability.

Sources (aggregates only):
  offset results table (columns zeilenschluessel, wert)
  offset report table (section 2, used only to cross-check the denominators)
Usage: fig1_s5_zeitversatz.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PETROL = "#1F5E73"
ROST = "#B5532A"
OCKER = "#D9A441"
ANTHRAZIT = "#222222"
HELLGRAU = "#D9D9D9"

# Plain matplotlib style: white background, readable font sizes, no custom fonts.
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.edgecolor": ANTHRAZIT,
    "axes.labelcolor": ANTHRAZIT,
    "xtick.color": ANTHRAZIT,
    "ytick.color": ANTHRAZIT,
    "text.color": ANTHRAZIT,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "errorbar.capsize": 2,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

ROOT = Path(str(data_path("s5.auswertung")))
CSV = ROOT / "nachrechnung" / "uhrzeit_aw110.csv"
MD = ROOT / "nachrechnung" / "UHRZEIT_AW110.md"
OUT = Path(__file__).resolve().parent / "fig1_s5_zeitversatz"

FOLDS = ("train", "validation")
NAME = {"train": "Development", "validation": "Internal assessment"}
COL = {"train": PETROL, "validation": OCKER}

# ---- Reading -------------------------------------------------------------
roh = {}
with open(CSV, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        roh[r["zeilenschluessel"]] = r["wert"]


def g(fold, key):
    return float(roh[f"icu_results.folds.{fold}.{key}"])


curve = {f: np.array([g(f, f"population_curve[{i}]") for i in range(289)]) for f in FOLDS}
offset_h = (np.arange(289) - 144) * 5 / 60.0
n = {f: {k: int(g(f, k)) for k in ("primary_people", "complete_people", "at_least_3_people", "identifiable_people")}
     for f in FOLDS}
inside = {f: {w: int(g(f, f"within.{w}.count")) for w in (15, 60, 360)} for f in FOLDS}
ci60 = {f: (g(f, "bootstrap_95ci.conditional_within60[0]") * 100, g(f, "bootstrap_95ci.conditional_within60[1]") * 100)
        for f in FOLDS}

# ---- Checks against the manuscript and the report table ------------------
for f in FOLDS:
    assert len(curve[f]) == 289 and offset_h[0] == -12 and offset_h[-1] == 12
    assert int(g(f, "population_optimum_minutes")) == 0
    assert np.argmin(curve[f]) == 144 and round(float(curve[f].min()), 1) == 3.0
MS = {  # manuscript, abstract and results section
    "train": dict(primary_people=17067, complete_people=6089, identifiable_people=392, w60=263, pct60=67.1, ci=(62.2, 71.6)),
    "validation": dict(primary_people=5846, complete_people=2062, identifiable_people=145, w60=102, pct60=70.3, ci=(62.4, 77.3)),
}
for f in FOLDS:
    for k in ("primary_people", "complete_people", "identifiable_people"):
        assert n[f][k] == MS[f][k], (f, k)
    assert inside[f][60] == MS[f]["w60"]
    assert round(100 * inside[f][60] / n[f]["identifiable_people"], 1) == MS[f]["pct60"]
    assert (round(ci60[f][0], 1), round(ci60[f][1], 1)) == MS[f]["ci"]
    # denominators of the chain: monotonically decreasing
    ketten = [n[f]["primary_people"], n[f]["complete_people"], n[f]["at_least_3_people"],
              n[f]["identifiable_people"], inside[f][60]]
    assert ketten == sorted(ketten, reverse=True)
# Cross-check of the "at least 3 complete ECGs" counts against the report table
md = MD.read_text(encoding="utf-8")
zeile = re.search(r"^\| at_least_3_people \| (\d+) \| (\d+) \|", md, re.M)
assert zeile and (int(zeile[1]), int(zeile[2])) == (n["train"]["at_least_3_people"], n["validation"]["at_least_3_people"]) == (1482, 497)
for f, kol in (("train", 1), ("validation", 2)):
    for k in ("primary_people", "complete_people", "identifiable_people"):
        z = re.search(rf"^\| {k} \| (\d+) \| (\d+) \|", md, re.M)
        assert z and int(z[kol]) == n[f][k]
# +-15 and +-360 min against the report table (shares in percent, decimal comma)
for f, kol in (("train", 2), ("validation", 3)):
    for w in (15, 360):
        z = re.search(rf"^\| ±{w} min \| (\d+)/(\d+) \(([\d,]+) %\) \| (\d+)/(\d+) \(([\d,]+) %\)", md, re.M)
        assert z
        c, d = (int(z[1]), int(z[2])) if f == "train" else (int(z[4]), int(z[5]))
        assert c == inside[f][w] and d == n[f]["identifiable_people"]

# ---- Drawing ---------------------------------------------------------------
plt.rcParams.update(STYLE)
BR, HO = 150.0, 112.0  # mm
fig = plt.figure(figsize=(BR / 25.4, HO / 25.4))
axA = fig.add_axes([0.095, 0.63, 0.885, 0.33])
axB = fig.add_axes([0.375, 0.115, 0.285, 0.36])   # attrition of persons
axC = fig.add_axes([0.805, 0.115, 0.185, 0.36])   # share inside the window


def buchstabe(x, y, b):
    fig.text(x, y, b, fontsize=10, fontweight="bold", color=ANTHRAZIT, ha="left", va="top")


# A: population curve
for f in FOLDS:
    axA.plot(offset_h, curve[f], color=COL[f], lw=1.2, label=NAME[f])
axA.set_xlim(-12, 12)
axA.set_ylim(0, 12)
axA.set_xticks(range(-12, 13, 2))
axA.set_xlabel("Offset applied to ECG time (h)")
axA.set_ylabel("Median absolute heart-rate\ndifference (beats per minute)")
axA.axvline(0, color=HELLGRAU, lw=0.7, zorder=0)
axA.annotate("Minimum 3.0 at 0 min\nin both partitions", xy=(0, 3.0), xytext=(1.6, 1.2), fontsize=7,
             color=ANTHRAZIT, va="bottom", ha="left",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ANTHRAZIT))
axA.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, handlelength=1.6)
buchstabe(0.005, 0.995, "A")

# B: attrition of persons (dots, logarithmic axis; no bars, since bar length means nothing on a log axis)
stufen = [
    ("People with an ECG nominally\ninside an intensive-care stay", "primary_people"),
    ("Complete error curve\n(reference at every offset)", "complete_people"),
    ("At least 3 complete ECGs", "at_least_3_people"),
    ("Identifiable individual offset", "identifiable_people"),
    ("Estimated offset within ±60 min", None),
]
dy = {"train": -0.15, "validation": 0.15}
for i, (lab, key) in enumerate(stufen):
    for f in FOLDS:
        wert = n[f][key] if key else inside[f][60]
        axB.plot(wert, i + dy[f], "o", color=COL[f], ms=4.5, label=NAME[f] if i == 0 else None, zorder=3)
        axB.text(wert * 1.25, i + dy[f], f"{wert:,}", va="center", ha="left", fontsize=6.5, color=ANTHRAZIT)
axB.set_xscale("log")
axB.set_xlim(60, 90000)
axB.set_ylim(len(stufen) - 0.5, -0.5)
axB.set_yticks(range(len(stufen)))
axB.set_yticklabels([s[0] for s in stufen], fontsize=7)
axB.set_xlabel("People (n, logarithmic axis)")
axB.tick_params(axis="y", length=0)
for yy in range(len(stufen)):
    axB.axhline(yy, color=HELLGRAU, lw=0.4, zorder=0)
buchstabe(0.005, 0.50, "B")

# C: share inside the window
fenster = (15, 60, 360)
xs = np.arange(3)
dx = {"train": -0.14, "validation": 0.14}
for f in FOLDS:
    anteil = [100 * inside[f][w] / n[f]["identifiable_people"] for w in fenster]
    axC.plot(xs + dx[f], anteil, "o", color=COL[f], ms=4, ls="none", label=NAME[f], zorder=3)
    lo, hi = ci60[f]
    a60 = anteil[1]
    axC.errorbar(1 + dx[f], a60, yerr=[[a60 - lo], [hi - a60]], color=COL[f], lw=1.0, capsize=2, ls="none", zorder=2)
axC.hlines(90, 1 - 0.42, 1 + 0.42, color=ROST, ls="--", lw=0.9)
axC.text(1.0, 92, "Specified criterion\nat ±60 min: 90%", color=ROST, fontsize=7, ha="center", va="bottom")
axC.set_xticks(xs)
axC.set_xticklabels(["±15", "±60", "±360"])
axC.set_xlim(-0.55, 2.55)
axC.set_ylim(0, 115)
axC.set_yticks(range(0, 101, 20))
axC.set_xlabel("Window (±min)")
axC.set_ylabel("Identifiable people with offset\ninside window (%)")
axC.legend(loc="lower right", handlelength=0.8, borderaxespad=0.1)
buchstabe(0.665, 0.50, "C")

OUT.parent.mkdir(parents=True, exist_ok=True)
for endung in ("pdf", "svg", "png"):
    fig.savefig(OUT.with_suffix("." + endung))
plt.close(fig)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
w, h = Image.open(str(OUT) + ".png").size
mm = w / 600 * 25.4
print("PNG", w, h, "px;", round(mm, 1), "mm wide")
assert 146 <= mm <= 152, mm
print("done", OUT)
