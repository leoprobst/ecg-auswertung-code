"""Abbildung S1 des Kurzberichts PUB-13 (v2, nur im Preprint-Supplement): Risikoverhaeltnisse fuer QTc > 500 ms.

Quelle (nur Aggregate, keine Patientendaten):
  ergebnisse_aw236/results_dev_aggregate.json  (Einheiten R08_VTVF, R08_Tod)
  ergebnisse_aw236/results_test_aggregate.json (Einheiten R08_VTVF, R08_Tod)
Gezeigt werden die unadjustierten 95-%-Intervalle (Spalten lo95, hi95). Jede gezeichnete Zahl wird per
assert gegen die im Manuskript gedruckte Rundung (2 Stellen) geprueft.

Aufruf: python -I abbildung1_pub13_v1.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402

PETROL = "#1F5E73"
ROST = "#B5532A"
OCKER = "#D9A441"
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


def referenzlinie(ax, x):
    ax.axvline(x, color=ROST, ls="--", lw=0.9)


def panel(ax, buchstabe):
    ax.text(-0.02, 1.02, buchstabe, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=ANTHRAZIT)


def speichern(fig, stamm):
    stamm = Path(stamm)
    stamm.parent.mkdir(parents=True, exist_ok=True)
    for endung in ("pdf", "svg", "png"):
        fig.savefig(stamm.with_suffix("." + endung))
    plt.close(fig)

QUELLE = Path(str(data_path("pub13.publikationen_pub_13_kardiologie_ergebnisse_aw236")))
DEV = json.loads((QUELLE / "results_dev_aggregate.json").read_text(encoding="utf-8"))
TEST = json.loads((QUELLE / "results_test_aggregate.json").read_text(encoding="utf-8"))


def trio(d):
    return (d["rr"], d["lo95"], d["hi95"]) if "lo95" in d else (d["rr"], d["lo95"], d["hi95"])


def zeilen(einheit, ergebnis, quelle):
    r = quelle[einheit]
    return [
        trio(r["primaer"][ergebnis]),
        trio(r["konkordanz"][ergebnis]),
        trio(r["sekundaer"][ergebnis]["vollstaendig_adjustiert"]),
    ]


NAMEN = ["Bazett, unadjusted (primary)", "Fridericia, unadjusted", "Bazett, fully adjusted"]
TITEL = {
    "R08_Tod": "Death within 30 days",
    "R08_VTVF": "VT, VF or cardiac arrest (same admission)",
}
ERG = {"R08_Tod": "Tod_30_Tage", "R08_VTVF": "VTVF_Herzstillstand"}

# Im Manuskript gedruckte Zahlen (Text, Tabelle S3a); Reihenfolge wie NAMEN
GEDRUCKT = {
    "R08_Tod": [("5.07 (4.52 to 5.68)", "4.19 (3.28 to 5.37)"),
                ("4.31 (3.54 to 5.25)", "3.64 (2.38 to 5.55)"),
                ("1.79 (1.59 to 2.02)", "1.52 (1.18 to 1.96)")],
    "R08_VTVF": [("3.69 (3.12 to 4.35)", "3.18 (2.20 to 4.58)"),
                 ("2.07 (1.48 to 2.89)", "3.46 (2.03 to 5.89)"),
                 ("2.52 (2.12 to 3.00)", "2.22 (1.53 to 3.22)")],
}


def fmt(t):
    return f"{t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f})"


fig, axes = plt.subplots(1, 2, figsize=(BREITE_ZWEISPALTIG, 3.1), sharex=True)
for ax, einheit, buchstabe in zip(axes, ("R08_Tod", "R08_VTVF"), "AB"):
    dev = zeilen(einheit, ERG[einheit], DEV)
    test = zeilen(einheit, ERG[einheit], TEST)
    n = len(NAMEN)
    for k in range(n):
        assert fmt(dev[k]) == GEDRUCKT[einheit][k][0], (einheit, k, fmt(dev[k]))
        assert fmt(test[k]) == GEDRUCKT[einheit][k][1], (einheit, k, fmt(test[k]))
        y = n - 1 - k
        for off, werte, farbe, marker in ((0.16, dev[k], PETROL, "o"),
                                          (-0.16, test[k], OCKER, "s")):
            ax.errorbar(werte[0], y + off, xerr=[[werte[0] - werte[1]], [werte[2] - werte[0]]],
                        fmt=marker, color=farbe, ms=4.2, lw=1.2, capsize=2.0, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(NAMEN)))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.9, 9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Risk ratio, QTc > 500 ms vs 500 ms or less (log scale)")
    ax.set_title(TITEL[einheit], fontsize=8, loc="left")
    referenzlinie(ax, x=1.0)
    panel(ax, buchstabe)
h1, = axes[0].plot([], [], "o", color=PETROL, label="Development (n = 128,400)")
h2, = axes[0].plot([], [], "s", color=OCKER, label="Test partition (n = 31,800)")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=7)
ziel = Path(__file__).resolve().parent / "FigS1_pub13_v2"
speichern(fig, str(ziel))
print("geschrieben:", ziel)
