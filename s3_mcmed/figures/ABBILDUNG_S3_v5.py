"""Abbildung 1 fuer MANUSKRIPT_S3_v5 (PUB-07), Hausstil the author.

Inhalt: A erwartete mittlere Aenderung der T-Amplitude (mm) je Kaliumaenderung gegen die Wiederholungsgrenze;
        B beobachtete Mediane und Quartile je Stufe der Kaliumaenderung (nur Stufen mit mindestens 10 Paaren).

Quelle (nur Aggregate, nur oeffentliche Werte): <local path>
Gelesen werden ausschliesslich diese Felder: klinisch.mm_je_mmol, klinisch.mm_je_mmol_ci,
klinisch.wiederholungsgrenze_mm, praezision.sd_dk und je Stufe n, dT_median_mm, dT_iqr_mv fuer die drei
Stufen mit mehr als 10 Paaren. Die Stufen 0,5 bis 1 und ab 1 werden NICHT gelesen und nicht gezeichnet
(Kleinzellenregel: ihre Zaehlungen sind aus der Differenz zur Gesamtzahl ableitbar).
Jede gezeichnete Zahl wird gegen den Manuskripttext geprueft (Assertions, Protokoll am Ende).

Zusaetzlich zu v1: die robuste Wiederholungsgrenze (0,0526 mV = 0,53 mm, nachtraeglich berechnet, S3_POSTHOC_H2.md Abschnitt 5)
wird gleichrangig als zweite Grenze gezeichnet.

Aufruf: <local path> -I ABBILDUNG_S3_v5.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import types

PETROL = "#1F5E73"
ROST = "#B5532A"
ANTHRAZIT = "#222222"
MM = 1 / 25.4   # one millimetre in inches


def _anwenden():
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


def _panel(ax, buchstabe):
    ax.text(-0.02, 1.02, buchstabe, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=ANTHRAZIT)


def _speichern(fig, stamm):
    stamm = Path(stamm)
    stamm.parent.mkdir(parents=True, exist_ok=True)
    for endung in ("pdf", "svg", "png"):
        fig.savefig(stamm.with_suffix("." + endung))
    plt.close(fig)


# plain matplotlib style in place of the private house-style module of the original script
hs = types.SimpleNamespace(PETROL=PETROL, ROST=ROST, MM=MM, BREITE_ZWEISPALTIG=175 * MM, anwenden=_anwenden, panel=_panel, speichern=_speichern)

# the figure checks every drawn number against the text of the manuscript; give a local copy of the text
OUT = Path(str(data_path("s3.figure_out_dir"))) / "ABBILDUNG_S3_v5"
MS = Path(str(data_path("s3.manuscript_md")))
JSONF = Path(str(data_path("s3.masked_result_json")))
GRAU = "#6B6B6B"
LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


raw = json.loads(JSONF.read_text(encoding="utf-8"))["h1"]
kl = raw["klinisch"]
stufen_roh = kl["individuell_je_kaliumstufe"]

# nur oeffentliche, nicht ableitbare Felder
beta = kl["mm_je_mmol"]
lo, hi = kl["mm_je_mmol_ci"]
rc = kl["wiederholungsgrenze_mm"]
sd_dk = raw["praezision"]["sd_dk"]
rc_rob = 0.53  # mm, robuste Wiederholungsgrenze (0,0526 mV), nachtraeglich berechnet (Post-hoc-Bericht, Tabelle Abschnitt 5)
STUFEN = [
    ("dk_ab_-1_abwaerts", "\u2264 -1.0"),
    ("dk_-1_bis_-0.5", "-1.0 to -0.5"),
    ("dk_betrag_unter_0.5", "|\u0394K| < 0.5"),
]
stufen = []
for key, lab in STUFEN:
    s = stufen_roh[key]
    assert s["n"] >= 10, (key, "Stufe zu klein fuer Einzelanzeige")
    q1, q3 = (v * 10 for v in s["dT_iqr_mv"])
    stufen.append((lab, s["n"], s["dT_median_mm"], q1, q3))

# ---------------------------------------------------------------- Gegenpruefung gegen den Manuskripttext
txt = MS.read_text(encoding="utf-8")


def fmt(x, nd=2):
    s = f"{x:.{nd}f}"
    return s


def must(s, label):
    assert s in txt, f"Manuskript enthaelt nicht: {label} -> '{s}'"
    log(f"ok  {label}: {s}")


must(f"{fmt(beta)} mm per mmol/L", "Steigung")
must(f"95% CI {fmt(lo)} to {fmt(hi)} mm", "Konfidenzintervall")
must(f"{fmt(rc)} mm", "Wiederholungsgrenze")
must(f"{fmt(rc_rob)} mm", "robuste Wiederholungsgrenze")
must(f"{fmt(sd_dk)} mmol/L", "SD der Kaliumaenderung")
for lab, n, med, q1, q3 in stufen:
    must(f"({n} pairs)", f"n Stufe {lab}")
must("(23 pairs)", "n Stufe ab 0,5 (nicht gezeichnet)")
must("-0.08 mm (interquartile range -0.70 to 0.27)", "Stufe 1")
must("-0.01 mm (-0.96 to 0.41)", "Stufe 2")
must("-0.09 mm (-0.67 to 0.41", "Stufe 3")
assert [round(s[2], 2) for s in stufen] == [-0.08, -0.01, -0.09]
assert [(round(s[3], 2), round(s[4], 2)) for s in stufen] == [(-0.70, 0.27), (-0.96, 0.41), (-0.67, 0.41)]

# ---------------------------------------------------------------- Zeichnung
hs.anwenden()
plt.rcParams["axes.unicode_minus"] = True
PETROL, ROST = hs.PETROL, hs.ROST
fig, (axa, axb) = plt.subplots(1, 2, figsize=(hs.BREITE_ZWEISPALTIG, 70 * hs.MM), gridspec_kw={"width_ratios": [1.15, 1]})

# Panel A
x = np.linspace(-2, 2, 161)
inner = np.abs(x) <= 1.0
axa.axhspan(-rc, rc, color=ROST, alpha=0.08, lw=0)
axa.axhline(rc, color=ROST, ls="--", lw=0.9)
axa.axhline(-rc, color=ROST, ls="--", lw=0.9)
for v in (-rc_rob, rc_rob):
    axa.axhline(v, color=ROST, ls=":", lw=1.0)
axa.text(-1.97, rc_rob + 0.06, "Robust limit \u00b10.53 mm\n(post hoc)", fontsize=6.5, color=ROST, ha="left", va="bottom")
band_lo = np.where(x >= 0, lo * x, hi * x)
band_hi = np.where(x >= 0, hi * x, lo * x)
axa.fill_between(x, band_lo, band_hi, color=PETROL, alpha=0.10, lw=0)
axa.fill_between(x, band_lo, band_hi, where=inner, color=PETROL, alpha=0.22, lw=0)
axa.plot(x[inner], (beta * x)[inner], color=PETROL, lw=1.4)
axa.plot(x[~inner & (x < 0)], (beta * x)[~inner & (x < 0)], color=PETROL, lw=1.0, ls="--")
axa.plot(x[~inner & (x > 0)], (beta * x)[~inner & (x > 0)], color=PETROL, lw=1.0, ls="--")
for v in (-sd_dk, sd_dk):
    axa.axvline(v, color=GRAU, ls=":", lw=0.8)
axa.text(sd_dk, 2.42, " \u00b11 SD of observed\n \u0394K (0.70)", fontsize=6.5, color=GRAU, va="top", ha="left")
axa.text(-1.97, rc + 0.07, "Pre-specified limit,\nsingle measurements \u00b11.73 mm", fontsize=7, color=ROST, ha="left", va="bottom")
axa.axhline(0, color="#222222", lw=0.5)
axa.set_xlim(-2, 2)
axa.set_ylim(-2.5, 2.5)
axa.set_xlabel("Potassium change \u0394K (mmol/L)")
axa.set_ylabel("Expected mean change in T amplitude (mm)")
hs.panel(axa, "A")

# Panel B
pos = np.arange(len(stufen))
axb.axhspan(-rc, rc, color=ROST, alpha=0.08, lw=0)
axb.axhline(rc, color=ROST, ls="--", lw=0.9)
axb.axhline(-rc, color=ROST, ls="--", lw=0.9)
for v in (-rc_rob, rc_rob):
    axb.axhline(v, color=ROST, ls=":", lw=1.0)
axb.text(-0.55, rc_rob + 0.06, "Robust limit \u00b10.53 mm (post hoc)", fontsize=6.5, color=ROST, ha="left", va="bottom")
axb.axhline(0, color="#222222", lw=0.5)
for p, (lab, n, med, q1, q3) in zip(pos, stufen):
    axb.plot([p, p], [q1, q3], color=PETROL, lw=2.2, solid_capstyle="butt")
    axb.plot([p], [med], "o", color="white", mec=PETROL, mew=1.2, ms=5, zorder=3)
    axb.text(p, 2.38, f"n = {n}", ha="center", va="top", fontsize=7)
axb.set_xticks(pos)
axb.set_xticklabels([s[0] for s in stufen])
axb.set_xlim(-0.6, len(stufen) - 0.4)
axb.set_ylim(-2.5, 2.5)
axb.set_xlabel("Category of potassium change \u0394K (mmol/L)")
axb.set_ylabel("Observed change in T amplitude (mm)")
axb.text(len(stufen) - 0.45, -rc - 0.08, "\u0394K \u2265 0.5: 23 pairs, not shown", fontsize=6.5, color=GRAU, ha="right", va="top")
hs.panel(axb, "B")

fig.tight_layout(w_pad=2.0)
hs.speichern(fig, OUT)
log(f"gespeichert: {OUT}.png/.pdf/.svg")
Path(str(OUT) + "_protokoll.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
