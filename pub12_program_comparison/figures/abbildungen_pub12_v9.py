"""Figures of the program comparison study (main text and supplement), plain matplotlib style.

Usage: python -I -B abbildungen_pub12_v9.py

Output: PDF, SVG and PNG (600 dpi) files in the figure output folder.
Data: only aggregate result tables of the preceding analysis stages and one report table with fitted coefficients; no single recordings.

Figure 1  Distribution of the QRS duration of one program by CLBBB code, distance of the paired mean from the 150 ms threshold,
          share classified differently.
Figure 2  Repeat recordings of one person against program changes (QRS at 120 and 150 ms, QTc at 450 ms).
Figure 3  QTc at 450 ms: pairwise discordance with the delivered formula and with Fridericia for all programs, heart rate strata,
          formula change within a program.
Figure 4  Threshold equivalence between programs, arithmetic split of the ECGDeli offset, NeuroKit2 QRS onset error by definition.
Supplementary Figure S3  Fitted coefficients of the derived limb leads, reproduction of the feature columns,
          signal test of the aVL/aVF labelling.
Every drawn number is checked by assert against the rounding printed in the manuscript.  Aggregates only, no single recordings.
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
from matplotlib.lines import Line2D  # noqa: E402

HIER = Path(str(data_path("pub12.publikationen_pub_12_etablierung")))
AUSWERTUNG = HIER.parents[1]
W1 = HIER / "ergebnisse" / "aw239"
W2 = HIER / "ergebnisse" / "aw245"
W234 = HIER / "ergebnisse" / "aw234"
OUT = HIER / "abbildungen" / "v9"
OUT.mkdir(parents=True, exist_ok=True)
# Plain matplotlib style (white background, DejaVu Sans, readable sizes, no logo)
PETROL = "#1F5E73"
ROST = "#B5532A"
OCKER = "#D9A441"
ANTHRAZIT = "#222222"
HELLGRAU = "#D9D9D9"
KASTEN = "#F1F6F8"
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


def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=ANTHRAZIT)


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext))
    plt.close(fig)
plt.rcParams["legend.fontsize"] = 7

TOOL = {"12SL": "#0072B2", "Uni-G": "#56B4E9", "BRAVEHEART": "#009E73",
        "ECGDeli": "#E69F00", "LoopLens": "#CC79A7", "NeuroKit2": "#000000"}
GRAU = "#7A7A7A"
MAXBREITE_MM = 175.5
P, R, OCK = PETROL, ROST, OCKER


def pruefe_breite(stamm, max_mm=MAXBREITE_MM):
    from PIL import Image
    w = Image.open(str(stamm) + ".png").size[0] / 600 * 25.4
    assert w <= max_mm, (stamm, round(w, 1))
    print(f"{Path(stamm).name}: {w:.1f} mm wide")


def zeile(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    r = df[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def r1(x, nd=1):
    return round(float(x), nd)


# ------------------------------------------------------------------ Figure 1
def abb1():
    h = pd.read_csv(W1 / "r1_histogramm_qrs_5ms.csv")
    a = pd.read_csv(W1 / "r1_abstand_zur_schwelle.csv")
    hs = h[h.programm == "12SL"]
    cl = hs[hs.teilmenge == "CLBBB"].set_index("klasse_von_ms").anzahl
    al = hs[hs.teilmenge == "alle"].set_index("klasse_von_ms").anzahl
    assert int(cl.sum()) == 418 and int(al.sum()) == 18869
    nocl = (al - cl.reindex(al.index).fillna(0)).clip(lower=0)
    assert int(nocl.sum()) == 18869 - 418
    bins = np.arange(40, 300, 5)
    pc = 100 * cl.reindex(bins).fillna(0) / cl.sum()
    pn = 100 * nocl.reindex(bins).fillna(0) / nocl.sum()

    fig = plt.figure(figsize=(BREITE_ZWEISPALTIG, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.85, 1.0], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(140, 160, color=KASTEN, zorder=0)
    ax.step(bins + 2.5, pn, where="mid", color=GRAU, lw=1.1, label="No CLBBB code\n(n = 18,451)")
    ax.step(bins + 2.5, pc, where="mid", color=P, lw=1.4, label="CLBBB code\n(n = 418)")
    ax.axvline(120, color=R, ls=":", lw=0.9)
    ax.axvline(150, color=R, ls="--", lw=0.9)
    ymax = float(max(pn.max(), pc.max()))
    ax.text(119, ymax * 1.2, "120 ms", color=R, fontsize=7, ha="right", va="top")
    ax.text(151, ymax * 1.0, "150 ms", color=R, fontsize=7, ha="left", va="top")
    ax.set_xlim(60, 285)
    ax.set_xticks([100, 150, 200, 250])
    ax.set_ylim(0, ymax * 1.42)
    ax.set_xlabel("QRS duration, 12SL (ms; 5-ms classes)")
    ax.set_ylabel("Share of persons in group per class (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), handlelength=1.4, labelspacing=0.5)
    panel(ax, "A")
    assert round(100 * cl.reindex(range(140, 160, 5)).sum() / 418) == 45      # shaded band 140 to 159 ms

    d = a[(a.kandidat == "BRAVEHEART") & (a.komparator == "12SL") & (a.schwelle == 150)]
    cls = ["0-5", "5-10", "10-20", "ge20"]
    lab = ["0 to 5", "5 to 10", "10 to 20", "20 or more"]
    sub = {"CLBBB": d[d.teilmenge == "CLBBB"].set_index("abstand_ms").loc[cls],
           "ohne": d[d.teilmenge == "ohne_CLBBB"].set_index("abstand_ms").loc[cls],
           "alle": d[d.teilmenge == "alle"].set_index("abstand_ms").loc[cls]}
    xlab = "Distance from 150 ms (ms);\nmean of BRAVEHEART\nand 12SL"
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(4)
    ax2.bar(x - 0.2, sub["CLBBB"].anteil_personen_pct, width=0.38, color=P, label="CLBBB code")
    ax2.bar(x + 0.2, sub["ohne"].anteil_personen_pct, width=0.38, color="white", ec=GRAU, lw=0.9, label="No CLBBB code")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lab, rotation=45, ha="right")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel("Persons in group (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(-0.6, 3.6)
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.0, handletextpad=0.4, fontsize=6.5)
    panel(ax2, "B")
    nah_c = r1(sub["CLBBB"].anteil_personen_pct.iloc[:2].sum())
    nah_o = r1(sub["ohne"].anteil_personen_pct.iloc[:2].sum())
    assert (nah_c, nah_o) == (47.0, 2.0), (nah_c, nah_o)       # text: 47.0 % against 2.0 %

    ax3 = fig.add_subplot(gs[2])
    pr = {}
    for key, col, mk, filled, dy in (("CLBBB", P, "o", True, -0.2), ("alle", OCK, "D", True, 0.0), ("ohne", GRAU, "o", False, 0.2)):
        s = sub[key]
        for i, c in enumerate(cls):
            lo, hi, v = float(s.D_ki_u.loc[c]), float(s.D_ki_o.loc[c]), float(s.D_pct.loc[c])
            ax3.plot([i + dy, i + dy], [lo, hi], color=col, lw=1.0)
            ax3.plot(i + dy, v, mk, color=col, ms=4.2 if mk == "o" else 3.6, mfc=col if filled else "white", mec=col, mew=0.9)
            pr[(key, c)] = (r1(v), r1(lo), r1(hi))
    exp = {("CLBBB", "0-5"): (48.0, 37.9, 57.7), ("CLBBB", "5-10"): (10.2, 4.9, 16.3),
           ("ohne", "0-5"): (58.3, 51.1, 65.9), ("ohne", "5-10"): (22.9, 17.0, 28.7),
           ("ohne", "ge20"): (0.1, 0.0, 0.1),
           ("alle", "0-5"): (54.5, 48.3, 60.5), ("alle", "5-10"): (18.7, 14.3, 22.9), ("alle", "10-20"): (7.1, 4.7, 9.4)}
    for k, v in exp.items():
        assert pr[k] == v, (k, pr[k], v)
    alle = sub["alle"]
    assert round(float(alle.D_pct.iloc[3]), 2) == 0.09 and round(float(alle.D_ki_u.iloc[3]), 2) == 0.05 and round(float(alle.D_ki_o.iloc[3]), 2) == 0.13
    ax3.set_xticks(x)
    ax3.set_xticklabels(lab, rotation=45, ha="right")
    ax3.set_xlabel(xlab)
    ax3.set_ylabel("Classified differently at 150 ms (%)")
    ax3.set_ylim(0, 70)
    ax3.set_xlim(-0.6, 3.6)
    leg = [Line2D([0], [0], marker="o", color=P, lw=0, ms=4.2, label="CLBBB code"),
           Line2D([0], [0], marker="D", color=OCK, lw=0, ms=3.6, label="All persons"),
           Line2D([0], [0], marker="o", color=GRAU, mfc="white", lw=0, ms=4.2, label="No CLBBB code")]
    ax3.legend(handles=leg, loc="upper right", handletextpad=0.2, fontsize=6.5)
    panel(ax3, "C")
    fig.tight_layout()
    save_figure(fig, OUT / "Fig1_abstand_zur_schwelle")
    pruefe_breite(OUT / "Fig1_abstand_zur_schwelle")


# ------------------------------------------------------------------ Figure 2
def abb2():
    w = pd.read_csv(W2 / "r9_wiederholbarkeit.csv")
    z = pd.read_csv(W2 / "r9_zwischen_programmen.csv")
    w = w[(w.population == "PRIM_24h_ohnePT") & (w.variante == "gemeinsame_Paare")]
    z = z[(z.population == "PRIM_24h_ohnePT") & (z.variante == "gemeinsame_Paare")]
    rep_prog = ["12SL", "BRAVEHEART", "Uni-G", "LoopLens", "ECGDeli"]
    btw = [("BRAVEHEART", "12SL"), ("Uni-G", "12SL"), ("Uni-G", "BRAVEHEART"), ("LoopLens", "BRAVEHEART"), ("ECGDeli", "12SL")]
    panels = [("qrs", "D120", "QRS at 120 ms", 30), ("qrs", "D150", "QRS at 150 ms", 30), ("qtc_einh", "D450", "QTc at 450 ms (Fridericia for all)", 30)]
    fig = plt.figure(figsize=(BREITE_ZWEISPALTIG, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.12)
    rows = [("rep", p) for p in rep_prog] + [("gap", None)] + [("btw", b) for b in btw]
    ypos = {}
    y = 0
    for kind, v in rows:
        ypos[(kind, v)] = y
        y += 1
    printed = {}
    for k, (mass, kz, titel, xmax) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        for kind, v in rows:
            if kind == "gap":
                continue
            yy = ypos[(kind, v)]
            if kind == "rep":
                r = zeile(w[(w.mass == mass) & (w.kennzahl == kz)], programm=v)
                col = P if v in ("12SL", "BRAVEHEART", "Uni-G") else GRAU
                filled = True
                key = ("rep", v)
            else:
                r = zeile(z[(z.mass == mass) & (z.kennzahl == kz)], kandidat=v[0], komparator=v[1])
                col = R if v[0] in ("BRAVEHEART", "Uni-G") else GRAU
                filled = False
                key = ("btw", v)
            est, lo, hi = 100 * float(r.schaetzer), 100 * float(r.ki_unten), 100 * float(r.ki_oben)
            printed[(mass, kz) + key] = (r1(est), r1(lo), r1(hi))
            if est > xmax:
                ax.annotate("", xy=(xmax, yy), xytext=(xmax - 3.2, yy), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
                ax.text(xmax - 0.4, yy - 0.45, f"{est:.0f}", ha="right", va="bottom", fontsize=7, color=col)
                continue
            ax.plot([lo, min(hi, xmax)], [yy, yy], color=col, lw=1.0)
            ax.plot(est, yy, "o", color=col, ms=4.4, mfc=col if filled else "white", mec=col, mew=1.0, clip_on=False, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(y - 0.4, -0.8)
        ax.set_xlabel("Classified differently (%)")
        ax.text(0.0, 1.02, titel, transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom", fontweight="bold")
        panel(ax, "ABC"[k])
        if k == 0:
            names = {}
            for kind, v in rows:
                if kind == "rep":
                    names[ypos[(kind, v)]] = v
                elif kind == "btw":
                    names[ypos[(kind, v)]] = f"{v[0]} against {v[1]}"
            ax.set_yticks(list(names))
            ax.set_yticklabels(list(names.values()))
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
    le = "\u2264"
    leg = [Line2D([0], [0], marker="o", color=P, lw=1, ms=4.4, label=f"Same program, two recordings of one person ({le} 24 h; 296 pairs; 280 for QTc)"),
           Line2D([0], [0], marker="o", color=R, mfc="white", lw=1, ms=4.4, label="Same recording, two of 12SL, Uni-G, BRAVEHEART (592 pair positions; 560 for QTc)"),
           Line2D([0], [0], marker="o", color=GRAU, lw=1, ms=4.4, label="ECGDeli and LoopLens (grey; filled: same program, open: two programs)")]
    fig.legend(handles=leg, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.17)
    exp = {("qrs", "D120", "rep", "12SL"): (1.7, 0.3, 3.3), ("qrs", "D120", "rep", "BRAVEHEART"): (1.4, 0.3, 2.8),
           ("qrs", "D120", "rep", "Uni-G"): (5.4, 2.4, 8.8), ("qrs", "D120", "rep", "LoopLens"): (16.2, 11.8, 20.9),
           ("qrs", "D120", "rep", "ECGDeli"): (0.0, 0.0, 0.0),
           ("qrs", "D120", "btw", ("BRAVEHEART", "12SL")): (1.9, 0.7, 3.3), ("qrs", "D120", "btw", ("Uni-G", "12SL")): (3.5, 2.0, 5.4),
           ("qrs", "D120", "btw", ("Uni-G", "BRAVEHEART")): (4.4, 2.5, 6.6), ("qrs", "D120", "btw", ("LoopLens", "BRAVEHEART")): (25.3, 20.4, 30.7),
           ("qrs", "D150", "rep", "12SL"): (2.7, 0.7, 5.1), ("qrs", "D150", "rep", "BRAVEHEART"): (2.0, 0.3, 4.2),
           ("qrs", "D150", "rep", "Uni-G"): (2.4, 0.7, 4.6), ("qrs", "D150", "rep", "ECGDeli"): (8.8, 5.4, 12.5),
           ("qrs", "D150", "rep", "LoopLens"): (3.0, 1.0, 5.7),
           ("qrs", "D150", "btw", ("BRAVEHEART", "12SL")): (2.7, 0.4, 5.9), ("qrs", "D150", "btw", ("Uni-G", "12SL")): (1.2, 0.3, 2.2),
           ("qrs", "D150", "btw", ("Uni-G", "BRAVEHEART")): (3.2, 0.8, 6.5), ("qrs", "D150", "btw", ("LoopLens", "BRAVEHEART")): (4.6, 1.9, 8.0),
           ("qtc_einh", "D450", "rep", "12SL"): (14.3, 10.1, 18.8), ("qtc_einh", "D450", "rep", "BRAVEHEART"): (15.0, 10.9, 19.4),
           ("qtc_einh", "D450", "rep", "Uni-G"): (17.1, 12.6, 22.0), ("qtc_einh", "D450", "rep", "LoopLens"): (14.3, 10.2, 18.9),
           ("qtc_einh", "D450", "rep", "ECGDeli"): (9.6, 6.0, 13.5),
           ("qtc_einh", "D450", "btw", ("BRAVEHEART", "12SL")): (11.8, 8.6, 15.2), ("qtc_einh", "D450", "btw", ("Uni-G", "12SL")): (9.6, 7.0, 12.4),
           ("qtc_einh", "D450", "btw", ("Uni-G", "BRAVEHEART")): (10.4, 7.6, 13.4), ("qtc_einh", "D450", "btw", ("LoopLens", "BRAVEHEART")): (12.5, 8.8, 16.7)}
    for k, v in exp.items():
        assert printed[k] == v, (k, printed[k], v)
    rep120 = [printed[("qrs", "D120", "rep", p)][0] for p in ("12SL", "BRAVEHEART", "Uni-G")]
    btw120 = [printed[("qrs", "D120", "btw", b)][0] for b in btw[:3]]
    rep150 = [printed[("qrs", "D150", "rep", p)][0] for p in ("12SL", "BRAVEHEART", "Uni-G")]
    btw150 = [printed[("qrs", "D150", "btw", b)][0] for b in btw[:3]]
    assert (min(rep120), max(rep120), min(btw120), max(btw120)) == (1.4, 5.4, 1.9, 4.4)
    assert (min(rep150), max(rep150), min(btw150), max(btw150)) == (2.0, 2.7, 1.2, 3.2)
    repq = [printed[("qtc_einh", "D450", "rep", p)][0] for p in ("12SL", "BRAVEHEART", "Uni-G", "LoopLens")]
    btwq = [printed[("qtc_einh", "D450", "btw", b)][0] for b in btw[:4]]
    assert (min(repq), max(repq)) == (14.3, 17.1) and (min(btwq), max(btwq)) == (9.6, 12.5)
    assert printed[("qrs", "D120", "btw", ("ECGDeli", "12SL"))][0] == 89.0 and printed[("qrs", "D150", "btw", ("ECGDeli", "12SL"))][0] == 77.0
    assert printed[("qtc_einh", "D450", "btw", ("ECGDeli", "12SL"))][0] == 78.0
    assert set(w[w.mass == "qtc_einh"].paare) == {280} and set(w[w.mass == "qrs"].paare) == {296} and int(z[z.mass == "qrs"].aufnahmen.iloc[0]) == 592
    assert set(z[z.mass == "qtc_einh"].aufnahmen) == {560}
    save_figure(fig, OUT / "Fig2_wiederholung_gegen_programmwechsel")
    pruefe_breite(OUT / "Fig2_wiederholung_gegen_programmwechsel")


# ------------------------------------------------------------------ Figure 3
def abb3():
    zw = pd.read_csv(W1 / "r5_zwischen_programmen.csv")
    inn = pd.read_csv(W1 / "r5_innerhalb_programm.csv")
    t4 = pd.read_csv(W234 / "T4_qtc_schwellen.csv")
    t4 = t4[(t4.scope == "P0") & (t4.teilmenge == "alle") & (t4.schema == "geschlechtsneutral") & (t4.stufe == "n450")]
    pairs = [("BRAVEHEART", "12SL"), ("Uni-G", "12SL"), ("LoopLens", "12SL"), ("Uni-G", "BRAVEHEART"), ("LoopLens", "BRAVEHEART"), ("LoopLens", "Uni-G")]
    fig = plt.figure(figsize=(BREITE_ZWEISPALTIG, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 0.9], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    pa = {}
    npairs = []
    for i, (a, b) in enumerate(pairs):
        d = {}
        for ver, mess in (("geliefert", "qtc_std"), ("fridericia", "qtc_einh")):
            r = zeile(t4[t4.messung == mess], kandidat=a, komparator=b)
            d[ver] = (float(r.diskordanz_pct), float(r.diskordanz_ki_u), float(r.diskordanz_ki_o))
            pa[(ver, a, b)] = tuple(r1(v) for v in d[ver])
            npairs.append(int(r.n_gemeinsam))
        ax.plot([d["geliefert"][0], d["fridericia"][0]], [i, i], color=HELLGRAU, lw=1.5, zorder=0)
        ax.plot(d["geliefert"][1:], [i, i], color=R, lw=0.9)
        ax.plot(d["geliefert"][0], i, "o", color=R, mfc="white", ms=4.4, mew=1.0)
        ax.plot(d["fridericia"][1:], [i, i], color=P, lw=0.9)
        ax.plot(d["fridericia"][0], i, "o", color=P, ms=4.4, mec="white", mew=0.4)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{a} against {b}" for a, b in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 32)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Classified differently at QTc 450 ms (%)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=R, mfc="white", lw=0.9, ms=4.4, label="Formula as delivered"),
                       Line2D([0], [0], marker="o", color=P, lw=0.9, ms=4.4, label="Fridericia for all")],
              loc="lower right", fontsize=7, handlelength=1.2, handletextpad=0.4)
    panel(ax, "A")
    # Supplementary Tables S3a and S3b (pairwise sets)
    exp = {("geliefert", "BRAVEHEART", "12SL"): (19.5, 18.9, 20.0), ("fridericia", "BRAVEHEART", "12SL"): (7.9, 7.5, 8.3),
           ("geliefert", "Uni-G", "12SL"): (19.9, 19.3, 20.5), ("fridericia", "Uni-G", "12SL"): (5.9, 5.6, 6.3),
           ("geliefert", "LoopLens", "12SL"): (18.3, 17.7, 18.9), ("fridericia", "LoopLens", "12SL"): (8.2, 7.8, 8.6),
           ("geliefert", "Uni-G", "BRAVEHEART"): (7.6, 7.2, 7.9), ("fridericia", "Uni-G", "BRAVEHEART"): (7.2, 6.8, 7.6),
           ("geliefert", "LoopLens", "BRAVEHEART"): (8.5, 8.1, 9.0), ("fridericia", "LoopLens", "BRAVEHEART"): (8.3, 7.9, 8.7),
           ("geliefert", "LoopLens", "Uni-G"): (9.0, 8.6, 9.4), ("fridericia", "LoopLens", "Uni-G"): (8.0, 7.6, 8.4)}
    for k, v in exp.items():
        assert pa[k] == v, (k, pa[k], v)
    # text: "7.6 to 19.9 %" with standard formulas, "5.9 to 7.9 %" with Fridericia for all (12SL, Uni-G, BRAVEHEART)
    tri = [("BRAVEHEART", "12SL"), ("Uni-G", "12SL"), ("Uni-G", "BRAVEHEART")]
    assert (min(pa[("geliefert",) + p][0] for p in tri), max(pa[("geliefert",) + p][0] for p in tri)) == (7.6, 19.9)
    assert (min(pa[("fridericia",) + p][0] for p in tri), max(pa[("fridericia",) + p][0] for p in tri)) == (5.9, 7.9)
    assert min(npairs) == 17924 and max(npairs) == 18867

    # B: heart rate strata, BRAVEHEART against 12SL (common set of persons with all values)
    ax2 = fig.add_subplot(gs[1])
    strata = [("HR_unter60", "< 60"), ("HR_60bis100", "60 to 100"), ("HR_ueber100", "> 100")]
    pb = {}
    for i, (s, lab) in enumerate(strata):
        for ver, col, filled, dy in (("geliefert_gesamt", R, False, -0.1), ("fridericia_gesamt", P, True, 0.1)):
            r = zeile(zw[(zw.schwellenstufe == "n450") & (zw.stratum == s) & (zw.vergleich == ver)], kandidat="BRAVEHEART", komparator="12SL")
            ax2.plot([i + dy, i + dy], [r.D_ki_u, r.D_ki_o], color=col, lw=0.9)
            ax2.plot(i + dy, r.D_pct, "o", color=col, ms=4.4, mfc=col if filled else "white", mew=1.0)
            pb[(s, ver)] = (r1(r.D_pct), r1(r.D_ki_u), r1(r.D_ki_o), int(r.n))
    assert pb[("HR_ueber100", "geliefert_gesamt")] == (53.6, 50.5, 56.6, 1044), pb
    assert pb[("HR_ueber100", "fridericia_gesamt")][:3] == (10.6, 8.8, 12.5), pb
    assert pb[("HR_unter60", "geliefert_gesamt")][0] == 7.5 and pb[("HR_unter60", "fridericia_gesamt")][0] == 6.9
    assert pb[("HR_60bis100", "geliefert_gesamt")][0] == 18.9 and pb[("HR_60bis100", "fridericia_gesamt")][0] == 7.2
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([s[1] for s in strata])
    ax2.set_xlabel("Heart rate of 12SL (beats/min);\nBRAVEHEART against 12SL")
    ax2.set_ylabel("Classified differently at 450 ms (%)")
    ax2.set_ylim(0, 62)
    ax2.set_xlim(-0.5, 2.5)
    panel(ax2, "B")

    # C: formula change within a program (Bazett against Fridericia)
    ax3 = fig.add_subplot(gs[2])
    progs = ["12SL", "Uni-G", "LoopLens", "BRAVEHEART", "ECGDeli"]
    pc = {}
    for i, p in enumerate(progs):
        r = zeile(inn[(inn.schwellenstufe == "n450") & (inn.formel_1 == "Bazett") & (inn.formel_2 == "Fridericia")], programm=p)
        ax3.plot([r.D_ki_u, r.D_ki_o], [i, i], color=P, lw=1.0)
        ax3.plot(r.D_pct, i, "o", color=P, ms=4.4)
        pc[p] = (r1(r.D_pct), r1(r.D_ki_u), r1(r.D_ki_o))
    exp3 = {"12SL": (14.7, 14.2, 15.2), "Uni-G": (13.8, 13.3, 14.3), "LoopLens": (13.9, 13.4, 14.4), "BRAVEHEART": (10.7, 10.2, 11.1), "ECGDeli": (5.3, 5.0, 5.6)}
    assert pc == exp3, pc
    assert (min(pc[p][0] for p in ("12SL", "Uni-G", "LoopLens", "BRAVEHEART")), max(pc[p][0] for p in progs)) == (10.7, 14.7)   # text: 10.7 to 14.7 without ECGDeli
    ax3.set_yticks(range(len(progs)))
    ax3.set_yticklabels(progs)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 18)
    ax3.set_xticks([0, 5, 10, 15])
    ax3.set_xlabel("Bazett against Fridericia, same\nQT and heart rate: classified\ndifferently at 450 ms (%)")
    panel(ax3, "C")
    fig.tight_layout()
    save_figure(fig, OUT / "Fig3_qtc_formel_herzfrequenz")
    pruefe_breite(OUT / "Fig3_qtc_formel_herzfrequenz")


# ------------------------------------------------------------------ Figure 4
def abb4():
    e = pd.read_csv(W1 / "r1_entsprechungsschwellen.csv")
    e = e[e.teilmenge == "alle"]
    zq = pd.read_csv(W1 / "r3_zerlegung_qrs.csv")
    zt = pd.read_csv(W1 / "r3_zerlegung_qt.csv")
    gz = pd.read_csv(W1 / "r6_globale_zellen.csv")
    ab = pd.read_csv(W1 / "r6_ableitungsweise.csv")
    fig = plt.figure(figsize=(BREITE_ZWEISPALTIG, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.95)
    # A: equivalence thresholds
    ax = fig.add_subplot(gs[0])
    refs = [120, 130, 150]
    pe = {}
    ax.plot([110, 215], [110, 215], color=R, ls="--", lw=0.9, zorder=0)
    ax.annotate("Equal thresholds", xy=(158, 158), xytext=(126, 176), color=R, fontsize=7, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=R, lw=0.6))
    for p, dx in (("BRAVEHEART", -1.8), ("Uni-G", 0.0), ("LoopLens", 1.8), ("ECGDeli", 3.6)):
        xs, ys, lo, hi = [], [], [], []
        for rf in refs:
            r = zeile(e, programm=p, referenzschwelle_ms=rf)
            xs.append(rf + dx)
            ys.append(float(r.aequivalente_schwelle_ms))
            lo.append(float(r.aequivalente_schwelle_ms) - float(r.aequivalente_schwelle_ms_ki_u))
            hi.append(float(r.aequivalente_schwelle_ms_ki_o) - float(r.aequivalente_schwelle_ms))
            pe[(p, rf)] = (int(r.aequivalente_schwelle_ms), int(r.aequivalente_schwelle_ms_ki_u), int(r.aequivalente_schwelle_ms_ki_o))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=TOOL[p], ms=4.2, lw=1.0, label=p)
    exp = {("BRAVEHEART", 120): (118, 116, 118), ("BRAVEHEART", 130): (128, 128, 130), ("BRAVEHEART", 150): (150, 148, 150),
           ("Uni-G", 120): (122, 120, 122), ("Uni-G", 130): (130, 130, 132), ("Uni-G", 150): (150, 148, 150),
           ("LoopLens", 120): (136, 136, 136), ("LoopLens", 130): (146, 144, 146), ("LoopLens", 150): (166, 166, 168),
           ("ECGDeli", 120): (186, 186, 187), ("ECGDeli", 130): (192, 192, 194), ("ECGDeli", 150): (208, 206, 209)}
    for k, v in exp.items():
        assert pe[k] == v, (k, pe[k], v)
    # Discussion: "shifted by 16 to 66 ms" (LoopLens 150: 16; ECGDeli 120: 66)
    assert pe[("LoopLens", 150)][0] - 150 == 16 and pe[("ECGDeli", 120)][0] - 120 == 66
    ax.set_xticks(refs)
    ax.set_xlim(112, 175)
    ax.set_ylim(110, 215)
    ax.set_xlabel("Threshold applied to 12SL (ms)")
    ax.set_ylabel("Threshold in the program with the same\nshare of persons at or above (ms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=7, ncol=2, columnspacing=0.8, handletextpad=0.2)
    panel(ax, "A")

    # B: arithmetic split of the ECGDeli offset against 12SL (mean, ms)
    ax2 = fig.add_subplot(gs[1])
    q = zq[zq.zerlegung.str.contains("Reihenfolge 1") & zq.zerlegung.str.contains("Endvariante extrem")].set_index("schritt")
    total_q = float(q.loc["Gesamt: Ausgang minus 12SL"].mittel_ms)
    rand_q = float(q.loc["Rand 26 ms -> 0 ms"].mittel_ms)
    assert round(total_q, 1) == 65.1 and round(rand_q, 1) == 52.0
    rest_q = total_q - rand_q
    assert round(rest_q, 1) == 13.1
    t = zt[zt.zerlegung.str.startswith("QT Reihenfolge 1") & zt.zerlegung.str.contains("Beginn median / T-Ende median")].set_index("schritt")
    total_t = float(t.loc["Gesamt: Ausgang minus 12SL"].mittel_ms)
    rand_t = float(t.loc["Rand 26 ms -> 0 ms"].mittel_ms)
    beg_t = float(t.loc["Beginn zweitfruehester -> median"].mittel_ms)
    tend_t = float(t.loc["T-Ende zweitspaetestes -> median"].mittel_ms)
    rest_t = float(t.loc["Rest: Endvariante minus 12SL"].mittel_ms)
    assert (round(total_t, 1), round(rand_t, 1), round(beg_t, 1), round(tend_t, 1), round(rest_t, 1)) == (113.8, 26.0, 10.1, 75.5, 2.3)
    assert abs(rand_t + beg_t + tend_t + rest_t - total_t) < 0.01
    parts_q = [(rand_q, P), (rest_q, HELLGRAU)]
    parts_t = [(rand_t, P), (tend_t, R), (beg_t, OCK), (rest_t, HELLGRAU)]
    for yy, parts in ((1, parts_q), (0, parts_t)):
        left = 0.0
        for v, col in parts:
            ax2.barh(yy, v, left=left, color=col, height=0.55, edgecolor="white", lw=0.6)
            if v >= 20:
                ax2.text(left + v / 2, yy, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")
            elif v >= 8:
                ax2.text(left + v / 2, yy + 0.33, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=ANTHRAZIT)
            else:
                ax2.text(left + v / 2, yy - 0.33, f"{v:.1f}", ha="center", va="top", fontsize=7, color=ANTHRAZIT)
            left += v
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(["QRS duration", "QT interval"])
    ax2.set_xlim(0, 130)
    ax2.set_ylim(-0.8, 3.1)
    ax2.set_xlabel("Arithmetic split of the mean difference\nECGDeli minus 12SL (ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P, R, OCK, HELLGRAU)]
    ax2.legend(handles, ["Margins", "T end vs median", "QRS onset vs median", "Remainder"], loc="upper right", fontsize=7, bbox_to_anchor=(1.02, 1.0))
    panel(ax2, "B")

    # C: NeuroKit2 onset error by definition (all four global cells of the text)
    ax3 = fig.add_subplot(gs[2])
    gcell = lambda v, n, r_: zeile(gz, variante=v, neurokit_aggregation=n, referenz_aggregation=r_, grenze="qrs_on")  # noqa: E731
    lead = lambda v, g: ab[(ab.variante == v) & (ab.ableitungsgruppe == g) & (ab.grenze == "qrs_on")].iloc[0]  # noqa: E731
    cells = [("Lead-wise: all leads", lambda v: lead(v, "alle")),
             ("Lead-wise: limb leads", lambda v: lead(v, "extremitaeten")),
             ("Lead-wise: V1 to V3", lambda v: lead(v, "V1bis3")),
             ("Lead-wise: V4 to V6", lambda v: lead(v, "V4bis6")),
             ("Global: earliest / earliest", lambda v: gcell(v, "ext", "ext")),
             ("Global: median / median", lambda v: gcell(v, "med", "med")),
             ("Global: median / earliest", lambda v: gcell(v, "med", "ext")),
             ("Global: earliest / median", lambda v: gcell(v, "ext", "med"))]
    pn = {}
    for i, (nm, f) in enumerate(cells):
        r = f("V0")
        ax3.plot([r.median_ms_ki_u, r.median_ms_ki_o], [i, i], color=P, lw=1.0)
        ax3.plot(r.median_ms, i, "o", color=P, ms=4.4)
        pn[nm] = (int(r.median_ms), int(r.median_ms_ki_u), int(r.median_ms_ki_o))
    for gv, mk in (("G0.1", "x"), ("G10", "+")):
        xs, ys = [], []
        for i, (nm, f) in enumerate(cells):
            rr = f(gv)
            assert abs(float(rr.median_ms) - float(f("V0").median_ms)) <= 2, (gv, nm)
            xs.append(float(rr.median_ms))
            ys.append(i)
        ax3.plot(xs, ys, mk, color=R, ms=5, mew=1.0, label=f"Signal x{gv[1:]}")
    exp = {"Lead-wise: all leads": (-52, -58, -48), "Lead-wise: limb leads": (-22, -28, -18), "Lead-wise: V1 to V3": (-88, -92, -86),
           "Lead-wise: V4 to V6": (-48, -58, -26), "Global: earliest / earliest": (-118, -122, -114), "Global: median / median": (-55, -60, -50)}
    for k, v in exp.items():
        assert pn[k] == v, (k, pn[k], v)
    assert pn["Global: median / earliest"][0] == -44 and pn["Global: earliest / median"][0] == -130      # text: -118, -55, -44, -130
    ax3.axvline(0, color=R, ls="--", lw=0.9)
    ax3.text(2, 0.0, "Expert\nmark", color=R, fontsize=7, ha="left", va="center")
    ax3.set_yticks(range(len(cells)))
    ax3.set_yticklabels([c[0] for c in cells])
    ax3.invert_yaxis()
    ax3.set_xlim(-140, 20)
    ax3.set_xlabel("NeuroKit2 dwt QRS onset minus expert mark,\nmedian (ms), 200 LUDB recordings")
    hh, ll = ax3.get_legend_handles_labels()
    hh = [Line2D([0], [0], marker="o", color=P, lw=1, ms=4.4)] + hh
    ax3.legend(hh, ["Median, 95% interval"] + ll, loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.3), ncol=1, handletextpad=0.4)
    panel(ax3, "C")
    fig.tight_layout()
    save_figure(fig, OUT / "Fig4_forschungswerkzeuge")
    pruefe_breite(OUT / "Fig4_forschungswerkzeuge")


# ------------------------------------------------------------------ Supplementary Figure S3 (formerly Fig 5)
def abbS3():
    md = (AUSWERTUNG / "nachrechnung/ECHONEXT_SB05_AW216.md").read_text(encoding="utf-8")
    g = {}
    for split in ("train", "val"):
        sec = md.split(f"### Kandidaten {split}")[1].split("###")[0]
        line = [l for l in sec.splitlines() if l.startswith("| G (acht Mediane")][0]
        o1 = line.split("|")[2]
        g[split] = [float(x.strip().replace(",", ".")) for x in re.split(r",\s(?=\d)", o1.strip())]
        assert len(g[split]) == 8, (split, o1)
    labels = ["III/I", "III/II", "aVR/I", "aVR/II", "aVL/I", "aVL/II", "aVF/I", "aVF/II"]
    assert [round(v, 4) for v in g["train"]] == [1.06, 1.0245, 1.1521, 1.1162, 1.187, 1.1405, 1.2232, 1.1897]
    assert [round(v, 4) for v in g["val"]] == [1.0607, 1.0255, 1.1521, 1.1162, 1.1871, 1.1413, 1.2231, 1.1898]
    ta = pd.read_csv(W2 / "r4_teil_a_dauer_gegen_features.csv")
    tb = pd.read_csv(W2 / "r4_teil_b_klassen.csv")
    fig = plt.figure(figsize=(BREITE_ZWEISPALTIG, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 0.8], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    x = np.arange(8)
    ax.axhline(1.0, color=R, ls="--", lw=0.9)
    ax.text(7.45, 1.005, "single common scale factor", color=R, fontsize=7, ha="right", va="bottom")
    ax.plot(x - 0.14, g["train"], "o", color=P, ms=4.2, label="Training split (n = 26,218)")
    ax.plot(x + 0.14, g["val"], "s", color=P, mfc="white", mew=1.0, ms=4.2, label="Validation split (n = 4,626)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.98, 1.31)
    ax.set_ylabel("Fitted / expected coefficient")
    ax.set_xlabel("Derived limb lead / reference lead")
    ax.legend(loc="upper left")
    panel(ax, "A")
    assert round(max(g["train"]), 2) == 1.22 and round(min(g["train"]), 2) == 1.02

    ax2 = fig.add_subplot(gs[1])
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    ta = ta.set_index("ableitung")
    own = [100 * ta.loc[l, "eigen_exakt"] / ta.loc[l, "n"] for l in leads]
    cross = [100 * ta.loc[l, "vertauscht_exakt"] / ta.loc[l, "n"] if l in ("aVL", "aVF") else np.nan for l in leads]
    xx = np.arange(12)
    ax2.bar(xx - 0.2, own, width=0.4, color=P, label="Annotation file: own lead")
    ax2.bar(xx + 0.2, cross, width=0.4, color=R, label="Annotation file: other lead")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(leads, rotation=90)
    ax2.set_ylim(0, 122)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Feature column reproduced exactly (%)")
    ax2.set_xlabel("Lead of the feature column")
    ax2.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.02, 1.03), fontsize=7)
    own_r = {l: round(v, 1) for l, v in zip(leads, own)}
    assert own_r["aVL"] == 3.8 and own_r["aVF"] == 3.8 and 96.5 <= min(v for l, v in own_r.items() if l not in ("aVL", "aVF")) and max(v for l, v in own_r.items() if l not in ("aVL", "aVF")) <= 98.4
    assert round(cross[4], 1) == 96.4 and round(cross[5], 1) == 96.7
    panel(ax2, "B")

    ax3 = fig.add_subplot(gs[2])
    tb = tb.set_index("klasse")
    k_n = tb.loc["wie benannt (Annotation aVL passt zu Signal aVL)"]
    k_v = tb.loc["vertauscht (Annotation aVL passt zu Signal aVF)"]
    k_u = tb.loc["unklar (|D| <= Nullschwelle)"]
    vals = [float(k_v.anteil_pct), float(k_n.anteil_pct), float(k_u.anteil_pct)]
    assert [round(v, 1) for v in vals[:2]] == [66.6, 3.9] and round(100 * int(k_u.n) / 21794, 1) == 29.5 and abs(vals[2] - 100 * 6419 / 21794) < 0.01
    assert int(k_v.n) == 14520 and int(k_n.n) == 855 and int(k_u.n) == 6419
    assert int(k_v.n) + int(k_n.n) == 15375 and round(100 * int(k_v.n) / 15375, 1) == 94.4      # text: 94.4 % of the 15,375 decidable recordings
    ax3.bar(range(3), vals, color=[R, P, HELLGRAU], width=0.65)
    for i, v in enumerate(vals):
        ax3.text(i, v + 1.5, f"{[100*int(k_v.n)/21794, 100*int(k_n.n)/21794, 100*int(k_u.n)/21794][i]:.1f}", ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["Crossed", "As named", "Undecided"], rotation=45, ha="right")
    ax3.set_ylim(0, 80)
    ax3.set_ylabel("Recordings, signal test (%)")
    panel(ax3, "C")
    fig.tight_layout()
    save_figure(fig, OUT / "FigS3_datensaetze")
    pruefe_breite(OUT / "FigS3_datensaetze")


ERWARTET_IM_TEXT = [
    "1.4 to 5.4", "1.9 to 4.4", "2.0 to 2.7", "1.2 to 3.2", "14.3 to 17.1", "9.6 to 12.5",
    "14.1 to 16.3", "0.9 to 1.2", "47.0%", "53.6%", "10.6%", "166", "208", "65.1", "52.0",
    "20.04", "19.47 to 20.62", "94.4%", "1.02 to 1.22", "7.6 to 19.9", "5.9 to 7.9", "10.7 to 14.7",
    "54.5%", "18.7%", "7.1%", "-118, -55, -44 or -130", "592 pair positions", "Supplementary Figure S3",
]


def pruefe_manuskript():
    p = HIER / "MANUSKRIPT_PUB12_v9.md"
    txt = p.read_text(encoding="utf-8")
    fehlt = [s for s in ERWARTET_IM_TEXT if s not in txt]
    assert not fehlt, f"not found in the manuscript: {fehlt}"
    print("Text check ok:", len(ERWARTET_IM_TEXT), "text passages")


if __name__ == "__main__":
    abb1()
    abb2()
    abb3()
    abb4()
    abbS3()
    pruefe_manuskript()
    print("done:", sorted(p.name for p in OUT.glob("*.png")))
