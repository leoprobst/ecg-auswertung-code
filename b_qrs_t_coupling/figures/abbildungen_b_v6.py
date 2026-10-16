"""Figures for the QRS-T coupling manuscript (manuscript version 6, supplement version 5), plain matplotlib style.

Version 6 is derived from the preceding figure script (version 3.1): Fig. 1B combines the completeness of the basic model
and the availability of the marker (difference below 10 persons), and the flow table of the supplement is coarsened accordingly.

Replaces the raw SVG figures embedded in the supplement (title inside the figure, generic font, no axis labels,
with spurious precision such as "2.340 (2.090-2.620)") and builds the missing flow diagram.

Aggregates only:
  results table of the exploratory test analysis (aggregate CSV: flow, bootstrap OR, measurement error)
  supplement text, version 3.1 (S2, S3, S8-S10, S13-S15; cross-check by assert)
  manuscript text, version 3.1 (Tables 2-3 and text; cross-check by assert)
Call: python -I <this script>
Every drawn number is checked by assert against the rounding printed in the manuscript or supplement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

AUSW = Path(str(data_path("b.auswertung")))

PUB = AUSW / "publikationen" / "PUB-02_qrs_t_kopplung_lvf"
OUT = PUB / "abbildungen"
MM = 1 / 25.4  # millimetre in inch
PET, ROST, OCKER, ANTH, GRAU, KASTEN = "#1F5E73", "#B5532A", "#D9A441", "#222222", "#D9D9D9", "#F1F6F8"
MITTELGRAU = "#6B6B6B"
LOG = []


def log(m):
    LOG.append(m)
    print(m)


# Plain matplotlib style (white background, DejaVu Sans, readable sizes, no logo, no custom fonts)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans", "axes.titlesize": 9, "axes.edgecolor": ANTH, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.dpi": 600,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "font.size": 8,
                     "axes.labelsize": 8, "savefig.bbox": "standard", "savefig.pad_inches": 0.0})


def save_figure(fig, stem):
    """Write the figure as PDF, SVG and PNG; `stem` is the path without suffix."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + suffix))
    plt.close(fig)



# Number check against version 3.1 of the texts; version 6 changed only the style
MS = (PUB / "MANUSKRIPT_B_v3_1.md").read_text(encoding="utf-8").replace("\r\n", "\n")
SUP = (PUB / "SUPPLEMENT_B_v5.md").read_text(encoding="utf-8").replace("\r\n", "\n")
CSV = pd.read_csv(AUSW / "nachrechnung" / "b_test_explorativ_v2.csv", low_memory=False)
S3 = SUP[SUP.index("## S3. "):SUP.index("### S3a.")]
ALLTXT = MS + "\n" + SUP


def g4(x):
    """Format as printed in the supplement: 4 significant digits, exponent as in '4.239e-05'."""
    return f"{x:.4g}"


def has(s, where=ALLTXT):
    assert s in where, f"not printed: {s!r}"
    return True


def cval(aid, metric):
    r = CSV[(CSV.analysis_id == aid) & (CSV.metric == metric)]
    assert len(r) == 1, (aid, metric, len(r))
    r = r.iloc[0]
    return float(r.value), r.ci_low, r.ci_high, r.valid, r.attempted


def s3_wald(nnn):
    """Wald OR (CI) from S3, row 'Association row nnn'; returns (or, lo, hi, N, cases, controls)."""
    m = re.search(rf"^\| Association row {nnn} \| .*? \| (\d[\d,]*)/(\d[\d,]*)/(\d[\d,]*) \| (\d+\.\d+) \((\d+\.\d+)-(\d+\.\d+)\) \|",
                  S3, flags=re.M)
    assert m, nnn
    n, c, k = (int(x.replace(",", "")) for x in m.groups()[:3])
    return float(m.group(4)), float(m.group(5)), float(m.group(6)), n, c, k


def fmt_or(o, lo, hi):
    return f"{o:.2f} ({lo:.2f}-{hi:.2f})"


# =====================================================================================================
# Figure 1: flow diagram
# =====================================================================================================
def flow_stage(aid, stage, field):
    """Value from the CSV (H1_flow / H4_flow): stage name, field = recordings|persons|excluded_recordings|excluded_persons."""
    r = CSV[(CSV.analysis_id == aid) & (CSV.stratum == '{"stage": "%s"}' % stage)]
    st = r.metric.str.extract(r"stages\.(\d+)\.")[0].iloc[0]
    q = r[r.metric == f"stages.{st}.{field}"]
    assert len(q) == 1
    v = q.iloc[0]
    return None if pd.isna(v.value) else int(float(v.value))


def parse_s15a_flow():
    sec = SUP[SUP.index("### S15a."):SUP.index("### S15b.")]
    out = {}
    for m in re.finditer(r"^\| (H[14]) \| (\w+) \| (\d+|suppressed) \| (\d+|suppressed) \| (\d+|suppressed) \| (\d+|suppressed) \|$", sec, flags=re.M):
        e, stage, ecg, per, xe, xp = m.groups()
        out[(e, stage)] = dict(recordings=None if ecg == "suppressed" else int(ecg), persons=None if per == "suppressed" else int(per),
                               excluded_recordings=None if xe == "suppressed" else int(xe),
                               excluded_persons=None if xp == "suppressed" else int(xp))
    return out


FLOW = parse_s15a_flow()
STAGES = ["candidate", "adult", "technical", "reference_available", "index", "reference_valid", "B3_complete", "C_e"]
assert len(FLOW) == 16
for aid, e in (("H1_flow", "H1"), ("H4_flow", "H4")):
    for st in STAGES:
        for f in ("recordings", "persons", "excluded_recordings", "excluded_persons"):
            if (e, st) == ("H4", "B3_complete") or ((e, st) == ("H4", "C_e") and f.startswith("excluded")):
                continue  # coarsened in the supplement table (difference below 10 persons)
            assert FLOW[(e, st)][f] == flow_stage(aid, st, f), (e, st, f)
log("Fig. 1: flow counts S15a == CSV H1_flow/H4_flow (all 2 x 8 stages x 4 fields)")
OV_D, OV_C = int(cval("overlap", "D_overlap_n")[0]), int(cval("overlap", "C_overlap_n")[0])
assert (OV_D, OV_C) == (8306, 7052)
has("Reference-index overlap was 8306 persons and complete-case overlap was 7052")

# Case/control counts from the S8 table and S3
CC = {"H1": dict(valid=(969, 8405), c=(800, 7571), b4=(7840, 636, 7204)),
      "H4": dict(valid=(515, 19302), c=(433, 17737), b4=(17345, 334, 17011))}
has("| reduced LVEF | positive | 969 | 800 | 17.44% (169) | 636 | 20.50% (164) |")
has("| reduced LVEF | negative | 8405 | 7571 | 9.92% (834) | 7204 | 4.85% (367) |")
has("| cardiomyopathy codes | positive | 515 | 433 | 15.92% (82) | 334 | 22.86% (99) |")
has("| cardiomyopathy codes | negative | 19302 | 17737 | 8.11% (1565) | 17011 | 4.09% (726) |")
assert s3_wald("007")[3:] == (7840, 636, 7204) and s3_wald("011")[3:] == (17345, 334, 17011)
assert s3_wald("001")[3:] == (8371, 800, 7571) and s3_wald("004")[3:] == (18170, 433, 17737)
assert FLOW[("H1", "C_e")]["persons"] == 8371 and FLOW[("H4", "C_e")]["persons"] == 18170
assert FLOW[("H1", "reference_valid")]["persons"] == 969 + 8405 and FLOW[("H4", "reference_valid")]["persons"] == 515 + 19302
assert 800 + 7571 == 8371 and 433 + 17737 == 18170 and 636 + 7204 == 7840 and 334 + 17011 == 17345


def n(x):
    return "n/a" if x is None else f"{x:,}"


def fig1():
    W, H = 175.0, 178.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    cols = {"H1": dict(x0=1.5, title="Reduced LVEF (<40%)", ref="Echo LVEF reference\n(within \u00b130 days)"),
            "H4": dict(x0=90.0, title="Coded cardiomyopathy", ref="Encounter assigned\n(code status known)")}
    BW, EW, GAP = 45.0, 33.0, 6.0   # main box, exclusion box, gap
    bh = 13.0
    ypos0 = 10.0
    pitch = 21.0

    def box(x, y, w, h, lines, fill=KASTEN, edge=PET, tc=ANTH, bold_first=True, lw=0.8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw))
        k = len(lines)
        step = 3.5
        ytext = y + h / 2 - (k - 1) * step / 2
        for i, s in enumerate(lines):
            ax.text(x + w / 2, ytext + i * step, s, ha="center", va="center", fontsize=8, color=tc,
                    fontweight="semibold" if (bold_first and i == 0) else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ANTH, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))

    for e, panel in (("H1", "A"), ("H4", "B")):
        c = cols[e]
        x0 = c["x0"]
        ax.text(x0, 3.6, panel, fontsize=10, fontweight="bold", color=ANTH, ha="left", va="center")
        ax.text(x0 + 6.0, 3.6, c["title"], fontsize=8, fontweight="semibold", color=PET, ha="left", va="center")
        F = lambda st, f: FLOW[(e, st)][f]  # noqa: E731
        spec = [
            (["Candidate ECGs", f"{n(F('candidate','recordings'))} ECGs", f"{n(F('candidate','persons'))} persons"], None),
            (["Adults (\u226518 years)", f"{n(F('adult','recordings'))} ECGs", f"{n(F('adult','persons'))} persons"],
             ["Not adult", f"{n(F('adult','excluded_recordings'))} ECGs", f"{n(F('adult','excluded_persons'))} persons"]),
            (["Technically successful ECG", f"{n(F('technical','recordings'))} ECGs", f"{n(F('technical','persons'))} persons"],
             ["Technical failure", f"{n(F('technical','excluded_recordings'))} ECGs", f"{n(F('technical','excluded_persons'))} persons"]),
            ([c["ref"].split("\n")[0], c["ref"].split("\n")[1], f"{n(F('reference_available','recordings'))} ECGs, {n(F('reference_available','persons'))} persons"],
             ["No eligible reference", f"{n(F('reference_available','excluded_recordings'))} ECGs", f"{n(F('reference_available','excluded_persons'))} persons"]),
            (["Index ECG, valid reference", f"{n(F('index','recordings'))} ECGs, {n(F('index','persons'))} persons",
              f"{n(CC[e]['valid'][0])} cases, {n(CC[e]['valid'][1])} controls"],
             ["Later ECGs of the same", "persons", f"{n(F('index','excluded_recordings'))} ECGs"]),
            (["Complete basic-model data", f"{n(F('B3_complete','persons'))} persons"],
             ["Incomplete basic-model", "data", f"{n(F('B3_complete','excluded_persons'))} persons"]),
        ]
        if F('B3_complete', 'persons') is None:  # stages combined
            spec.pop()
        # last stage C_e
        ex_ce = F("C_e", "excluded_persons")
        if F('B3_complete', 'persons') is None:
            assert F('reference_valid', 'persons') - F('C_e', 'persons') == ex_ce == 1647
            ce_ex = ['Incomplete basic-model', 'data or marker missing', f"{n(ex_ce)} persons"]
        else:
            ce_ex = (["Marker missing", f"{n(ex_ce)} persons"] if ex_ce is not None else ["Marker missing", "<10 persons", "(count suppressed)"])
        spec.append((["Analysis set (marker available)", f"{n(F('C_e','persons'))} persons",
                      f"{n(CC[e]['c'][0])} cases, {n(CC[e]['c'][1])} controls"], ce_ex))
        y = ypos0
        centers = []
        for i, (main, excl) in enumerate(spec):
            last = i == len(spec) - 1
            box(x0, y, BW, bh, main, fill=PET if last else KASTEN, edge=PET, tc="white" if last else ANTH)
            centers.append((x0 + BW / 2, y, y + bh))
            if excl is not None:
                ex = x0 + BW + GAP
                box(ex, y - 6.8 - (0 if len(excl) == 3 else -1.0), EW, 11.0 if len(excl) < 4 else 13.0, excl,
                    fill="white", edge=MITTELGRAU, bold_first=False, lw=0.6)
            y += pitch
        # arrows along the main column and sideways exclusion arrows
        for i in range(len(spec) - 1):
            cx, _, yb = centers[i]
            arrow(cx, yb, cx, centers[i + 1][1])
        for i, (main, excl) in enumerate(spec):
            if excl is not None:
                yy = centers[i][1] - 4.6 - (0 if len(excl) == 3 else -1.0) + (0 if len(excl) == 3 else 0.0)
                yy = centers[i][1] - 4.0
                ax.annotate("", xy=(x0 + BW + GAP, yy), xytext=(x0 + BW / 2, yy),
                            arrowprops=dict(arrowstyle="-|>", color=MITTELGRAU, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
        # branch: model with additional adjustment (B4)
        yl = centers[-1][2]
        b4 = CC[e]["b4"]
        yb4 = yl + 6.0
        box(x0, yb4, BW, bh, ["Stronger-model set", f"{n(b4[0])} persons", f"{n(b4[1])} cases, {n(b4[2])} controls"],
            fill="white", edge=PET)
        arrow(x0 + BW / 2, yl, x0 + BW / 2, yb4)
    # note on the overlap
    ax.text(W / 2, H - 4.0,
            f"The two endpoints share persons ({n(OV_D)} at the reference stage, {n(OV_C)} in the analysis sets); counts cannot be added.",
            ha="center", va="center", fontsize=8, color=ANTH)
    save_figure(fig, OUT / "Fig1_participant_flow")
    log("Fig. 1 written")


# =====================================================================================================
# Forest plot helper functions
# =====================================================================================================
def draw_forest(fig_w, groups, panel_titles, xlim, xticks, xlabel, fname, numfmt=fmt_or, label_w=49.0, ax_w=34.0,
                num_w=24.0, row_h=4.1, band=None, legend_items=None, ref_line=1.0, ref_text="OR = 1", panel_letters=("A", "B"),
                extra_lines=None, foot=None, xscale="log"):
    """groups: list of (group title, [row, ...]); row = dict(label, cells=[c1, c2], tick=[t1, t2] optional).
    Cell: dict(kind in wald|boot|point|ne|na, o, lo, hi, text optional, arrow_left/arrow_right optional)."""
    rows = []
    for gt, items in groups:
        rows.append(("group", gt, None))
        for it in items:
            rows.append(("row", it["label"], it))
    N = len(rows)
    top = 11.0
    legend_h = 15.0
    axis_h = 11.0
    H = top + N * row_h + axis_h + legend_h
    fig = plt.figure(figsize=(fig_w * MM, H * MM))
    npan = len(panel_titles)
    x_lab = 0.0
    # axes
    x_axes = []
    x = label_w + 3.0
    for i in range(npan):
        x_axes.append(x)
        x += ax_w + (num_w if numfmt else 0.0) + 4.0
    axes = []
    for i in range(npan):
        a = fig.add_axes([x_axes[i] / fig_w, (axis_h + legend_h) / H, ax_w / fig_w, N * row_h / H])
        axes.append(a)
    lab = fig.add_axes([0, (axis_h + legend_h) / H, label_w / fig_w, N * row_h / H])
    lab.set_xlim(0, label_w)
    lab.set_ylim(N - 0.5, -0.5)
    lab.axis("off")
    ymax = N - 0.5
    for gi, a in enumerate(axes):
        a.set_xscale(xscale)
        a.set_xlim(*xlim)
        a.set_ylim(ymax, -0.5)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_xticks(xticks)
        a.set_xticklabels([("%g" % t) for t in xticks])
        a.xaxis.set_minor_locator(plt.NullLocator())
        a.set_xlabel(xlabel, fontsize=8, labelpad=2)
        a.tick_params(axis="x", length=3, width=0.7, pad=2)
        # reference line OR = 1
        a.axvline(ref_line, color=ROST, ls="--", lw=0.9, zorder=1)
        if band and band[gi] is not None:
            lo, hi, mid = band[gi]
            a.axvspan(lo, hi, color=GRAU, alpha=0.45, lw=0, zorder=0)
            a.axvline(mid, color=MITTELGRAU, lw=0.6, zorder=1)
        if extra_lines and extra_lines[gi]:
            for xv, lt in extra_lines[gi]:
                a.axvline(xv, color=ROST, ls="--", lw=0.9, zorder=1)
        # group background separators
    # panel labels
    for gi, a in enumerate(axes):
        xl = x_axes[gi]
        fig.text((xl - 3.0) / fig_w, 1 - 3.5 / H, panel_letters[gi], fontsize=10, fontweight="bold", color=ANTH, ha="right", va="center")
        fig.text((xl - 1.0) / fig_w, 1 - 3.5 / H, panel_titles[gi], fontsize=8, fontweight="semibold", color=PET, ha="left", va="center")
        if numfmt:
            fig.text((xl + ax_w + 1.0) / fig_w, 1 - 9.0 / H, "OR (95% CI)", fontsize=8, color=ANTH, ha="left", va="center")
    # rows
    for i, (typ, label, it) in enumerate(rows):
        if typ == "group":
            lab.text(1.0, i, label, fontsize=8, fontweight="semibold", color=PET, ha="left", va="center")
            continue
        lab.text(2.5, i, label, fontsize=8, color=ANTH, ha="left", va="center")
        for gi, a in enumerate(axes):
            c = it["cells"][gi]
            xl = x_axes[gi]
            numx = xl + ax_w + 1.0
            kind = c["kind"]
            if it.get("tick") and it["tick"][gi]:
                a.plot([it["tick"][gi]] * 2, [i - 0.33, i + 0.33], color=MITTELGRAU, lw=0.8, zorder=2)
            if kind in ("wald", "boot"):
                o, lo, hi = c["o"], c["lo"], c["hi"]
                col = PET if kind == "wald" else OCKER
                a.plot([max(lo, xlim[0]), min(hi, xlim[1])], [i, i], color=col, lw=1.3, solid_capstyle="butt", zorder=3)
                if lo < xlim[0]:
                    a.plot([xlim[0]], [i], marker="<", color=col, ms=3.5, mec=col, zorder=4, clip_on=False)
                if hi > xlim[1]:
                    a.plot([xlim[1]], [i], marker=">", color=col, ms=3.5, mec=col, zorder=4, clip_on=False)
                if kind == "wald":
                    a.plot([o], [i], marker="o", ms=4.0, mfc=PET, mec=PET, zorder=5)
                else:
                    a.plot([o], [i], marker="D", ms=3.8, mfc=OCKER, mec=ANTH, mew=0.5, zorder=5)
                txt = c.get("text") or numfmt(o, lo, hi)
                if numfmt:
                    fig.text(numx / fig_w, 1 - (top + (i + 0.5) * row_h) / H, txt, fontsize=8, color=ANTH, ha="left", va="center")
            elif kind == "point":
                a.plot([c["o"]], [i], marker="o", ms=4.0, mfc="white", mec=PET, mew=1.0, zorder=5)
                if c.get("hollow_col"):
                    pass
                if numfmt:
                    fig.text(numx / fig_w, 1 - (top + (i + 0.5) * row_h) / H, c.get("text") or f"{c['o']:.2f} (no CI)", fontsize=8,
                             color=ANTH, ha="left", va="center")
            elif kind in ("ne", "na"):
                if numfmt:
                    fig.text(numx / fig_w, 1 - (top + (i + 0.5) * row_h) / H, "not estimable" if kind == "ne" else "not applicable",
                             fontsize=8, color=MITTELGRAU, ha="left", va="center", style="italic")
    # legend
    handles = legend_items or []
    if handles:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(2.0 / fig_w, 1.0 / H), ncol=2, frameon=False,
                   fontsize=8, handlelength=1.8, columnspacing=1.2, labelspacing=0.3, borderaxespad=0)
    ref_x = [axes[g] for g in range(npan)]
    for a in ref_x:
        a.text(ref_line, -0.7, ref_text, color=ROST, fontsize=8, ha="left", va="bottom", transform=a.transData)
    save_figure(fig, OUT / fname)
    return H


# =====================================================================================================
# Figure 2: forest plot of the associations
# =====================================================================================================
def W(nnn):
    o, lo, hi, *_ = s3_wald(nnn)
    return dict(kind="wald", o=o, lo=lo, hi=hi)


def B(aid, metric="or"):
    o, lo, hi, v, a = cval(aid, metric)
    return dict(kind="boot", o=o, lo=float(lo), hi=float(hi))


def P(aid, metric="or"):
    o, *_ = cval(aid, metric)
    return dict(kind="point", o=o)


NE = dict(kind="ne")
NA = dict(kind="na")


def fig2():
    # --- cross-check of the drawn numbers against the printed text
    chk = {
        "001": "2.34 (2.09-2.62)", "004": "2.03 (1.79-2.29)", "007": "2.51 (2.20-2.87)", "008": "2.49 (2.21-2.82)",
        "011": "1.98 (1.69-2.33)", "012": "2.10 (1.83-2.41)", "016": "2.34 (2.09-2.61)", "017": "2.42 (2.14-2.74)",
        "018": "2.29 (2.06-2.53)", "025": "2.39 (2.04-2.81)", "026": "2.50 (2.19-2.86)", "027": "2.67 (2.20-3.25)",
        "028": "2.20 (1.92-2.52)", "029": "2.37 (1.97-2.86)", "030": "2.34 (2.03-2.70)", "045": "2.15 (1.79-2.58)",
        "046": "2.11 (1.79-2.48)", "047": "2.16 (1.79-2.60)", "048": "1.87 (1.60-2.18)", "049": "2.23 (1.81-2.76)",
        "050": "1.91 (1.64-2.21)"}
    for k, s in chk.items():
        o, lo, hi, *_ = s3_wald(k)
        assert fmt_or(o, lo, hi) == s, (k, s)
    # main text and tables
    for s in ("| LVEF <40%; normalized Fr\u00e9chet | 8,371 / 800 / 7,571 | 2.34 (2.09-2.62) |",
              "| Coded cardiomyopathy; normalized beat-median Fr\u00e9chet | 18,170 / 433 / 17,737 | 2.03 (1.79-2.29) |",
              "| LVEF <40%, basic on stronger-model set | 7,840 / 636 / 7,204 | 2.49 (2.21-2.82) |",
              "| LVEF <40%, stronger model | 7,840 / 636 / 7,204 | 2.51 (2.20-2.87) |",
              "| Coded cardiomyopathy, basic on stronger-model set | 17,345 / 334 / 17,011 | 2.10 (1.83-2.41) |",
              "| Coded cardiomyopathy, stronger model | 17,345 / 334 / 17,011 | 1.98 (1.69-2.33) |",
              "Inclusive LVEF \u226440% gave OR 2.29 (2.06-2.53)",
              "Unweighted OR was 2.65 (conditional Wald CI 2.28-3.07), versus IPW OR 2.97 (2.52-3.50)"):
        has(s, MS)
    # verification Wald (S8)
    has("| unweighted, same first technical index | 4844 | 4413 | 431 | 2.6457223 (2.2798401; 3.0703236) |", SUP)
    has("| IPW, same first technical index | 4844 | 4413 | 431 | 2.9678222 (2.519857; 3.4954239) |", SUP)
    unw = dict(kind="wald", o=2.6457223, lo=2.2798401, hi=3.0703236)
    ipw_old = dict(kind="wald", o=2.9678222, lo=2.519857, hi=3.4954239)
    assert fmt_or(2.6457223, 2.2798401, 3.0703236) == "2.65 (2.28-3.07)" and fmt_or(2.9678222, 2.519857, 3.4954239) == "2.97 (2.52-3.50)"
    # bootstrap rows (CSV) against printed strings
    def chk_boot(aid, metric, printed):
        o, lo, hi, v, a = cval(aid, metric)
        assert f"{g4(o)} [{g4(float(lo))}; {g4(float(hi))}]" == printed, (aid, printed)
        has(printed)
    chk_boot("H1_MI", "or", "2.317 [2.112; 2.592]")
    chk_boot("H4_MI", "or", "2.042 [1.808; 2.305]")
    chk_boot("H1_W24_qrs_ge120", "or", "2.344 [1.988; 2.834]")
    chk_boot("H4_W24_qrs_ge120", "or", "1.883 [1.6; 2.227]")
    chk_boot("H1_W19", "p95_or", "3.011 [2.587; 3.529]")
    chk_boot("H1_W19", "p99_or", "3.019 [2.596; 3.539]")
    for aid, printed in (("H1_no_qtc_C", "2.281"), ("H1_no_qtc_extended", "2.267"), ("H1_missing_indicator_extended", "2.307"),
                         ("H4_no_qtc_C", "1.986"), ("H4_no_qtc_extended", "2.029"), ("H4_missing_indicator_extended", "2.027"),
                         ("H1_W24_rbbb", "2.437"), ("H1_cross_M", "2.331"), ("H4_cross_F", "2.053")):
        o = cval(aid, "or")[0]
        assert g4(o) == printed, (aid, printed)
        has(printed + " (CI unavailable)") if "rbbb" in aid or "qtc" in aid or "missing" in aid else None
    has("| H1/rbbb | estimable | 791 / 146 / 645 | 2.437 (CI unavailable) | 96/1000 |", SUP)
    has("| H1/M | 8369 / 799 / 7570 | 2.331 |", SUP)
    has("| H4/F | 18041 / 429 / 17612 | 2.053 |", SUP)
    log("Fig. 2: all Wald, bootstrap and point rows checked against manuscript/supplement (assert)")

    # primary-model band
    mo, ml, mh, *_ = s3_wald("001")
    co, cl, ch, *_ = s3_wald("004")
    groups = [
        ("Primary model", [
            dict(label="Basic model plus marker", cells=[W("001"), W("004")])]),
        ("Adjustment and missing data", [
            dict(label="Basic model, stronger-model set", cells=[W("008"), W("012")]),
            dict(label="Stronger model (voltage, axis)", cells=[W("007"), W("011")]),
            dict(label="QTc omitted, same persons", cells=[P("H1_no_qtc_C"), P("H4_no_qtc_C")]),
            dict(label="QTc omitted, extended set", cells=[P("H1_no_qtc_extended"), P("H4_no_qtc_extended")]),
            dict(label="Missing-value indicators, extended set", cells=[P("H1_missing_indicator_extended"), P("H4_missing_indicator_extended")]),
            dict(label="Multiple imputation (20 imputations)", cells=[B("H1_MI"), B("H4_MI")])]),
        ("Reference definition", [
            dict(label="LVEF \u226440% (inclusive threshold)", cells=[W("018"), NA]),
            dict(label="Echo within 29 days", cells=[W("016"), NA]),
            dict(label="Echo within 7 days", cells=[W("017"), NA])]),
        ("Subgroups", [
            dict(label="Age <65 years", cells=[W("027"), W("047")]),
            dict(label="Age \u226565 years", cells=[W("028"), W("048")]),
            dict(label="Recorded sex: female", cells=[W("029"), W("049")]),
            dict(label="Recorded sex: male", cells=[W("030"), W("050")]),
            dict(label="Sinus rhythm (machine-report selector)", cells=[W("026"), W("046")]),
            dict(label="QRS <120 ms, no block pattern", cells=[W("025"), W("045")]),
            dict(label="QRS \u2265120 ms", cells=[B("H1_W24_qrs_ge120"), B("H4_W24_qrs_ge120")]),
            dict(label="Right bundle-branch block pattern", cells=[P("H1_W24_rbbb"), NE]),
            dict(label="Left bundle-branch block pattern", cells=[NE, NE])]),
        ("Verification weighting", [
            dict(label="Unweighted", cells=[unw, NA]),
            dict(label="IPW, marker-free weights", cells=[ipw_old, NA]),
            dict(label="IPW, marker-containing, P95 cap", cells=[B("H1_W19", "p95_or"), NA]),
            dict(label="IPW, marker-containing, P99 cap", cells=[B("H1_W19", "p99_or"), NA])]),
        ("Other marker form", [
            dict(label="Other Fr\u00e9chet form, common set", cells=[P("H1_cross_M"), P("H4_cross_F")])]),
    ]
    handles = [
        Line2D([0], [0], marker="o", color=PET, lw=1.3, ms=4.0, label="Wald interval (person-cluster)"),
        Line2D([0], [0], marker="D", color=OCKER, mec=ANTH, mew=0.5, lw=1.3, ms=3.8, label="Bootstrap percentile interval"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec=PET, mew=1.0, ms=4.0, label="Point estimate only (no interval)"),
        Rectangle((0, 0), 1, 1, fc=GRAU, alpha=0.45, ec="none", label="Primary model 95% CI"),
    ]
    H = draw_forest(175.0, groups, ["Reduced LVEF (<40%)", "Coded cardiomyopathy"], (0.95, 4.2), [1, 1.5, 2, 3, 4],
                    "Odds ratio per development SD (log scale)", "Fig2_association_sensitivities",
                    band=[(ml, mh, mo), (cl, ch, co)], legend_items=handles)
    log(f"Fig. 2 written (H={H:.1f} mm)")


# =====================================================================================================
# Figure 3: measurement-error correction in the assessment partition
# =====================================================================================================
def fig3():
    # cross-check against S15c (printed rows)
    S15C = SUP[SUP.index("### S15c."):SUP.index("### S15d.")]

    def csv_pair(aid, var):
        """linear_or/quadratic_or with bootstrap CI; cross-check against the S15c row (printed 4 digits)."""
        out = []
        for m in ("linear_or", "quadratic_or"):
            o, lo, hi, v, a = cval(aid, m)
            out.append(dict(kind="boot", o=o, lo=float(lo), hi=float(hi)))
            assert v == a == 1000
            has(f"{g4(o)} [{g4(float(lo))}; {g4(float(hi))}]", S15C)
        # reference (uncorrected under the same angle model): exp(uncorrected_beta)
        ub = cval(aid, "uncorrected_beta")[0]
        return out, float(np.exp(ub)), ub

    steps = {}
    for var, aid1, aid4 in (("raw", "H1_error_raw", "H4_error_raw"), ("clipped", "H1_error_clipped", "H4_error_clipped"),
                            ("logit", "H1_error_logit", "H4_error_logit"), ("spline", "H1_error_spline", "H4_error_spline"),
                            ("joint", "H1_error_joint", None)):
        steps[(var, "H1")] = csv_pair(aid1, var)
        if aid4:
            steps[(var, "H4")] = csv_pair(aid4, var)
    # reference ticks against the printed difference: log(OR) - paired delta beta = uncorrected beta
    printed_delta = {("raw", "H1"): (2.339, -0.001531), ("raw", "H4"): (2.013, -0.007613), ("clipped", "H1"): (2.338, -0.001815),
                     ("logit", "H1"): (2.4, 0.003061), ("spline", "H1"): (2.401, 0.002343), ("joint", "H1"): (2.629, 0.1153)}
    for (var, e), (o, d) in printed_delta.items():
        ub = steps[(var, e)][2]
        assert abs((np.log(o) - d) - ub) < 0.0025, (var, e, np.log(o) - d, ub)
    # RC (angle only) and joint RC
    rc1 = cval("H1_error_rc", "or")
    rc4 = cval("H4_error_rc", "or")
    assert (g4(rc1[0]), g4(float(rc1[1])), g4(float(rc1[2]))) == ("2.087", "0.4186", "9.548")
    assert (g4(rc4[0]), g4(float(rc4[1])), g4(float(rc4[2]))) == ("0.5628", "4.239e-05", "28.99")
    has("Assessment angle-only RC gave marker OR 2.087 [0.4186; 9.548] for reduced LVEF and 0.5628 [4.239e-05; 28.99] for cardiomyopathy", MS)
    jr = cval("H1_error_joint_rc", "or")
    assert (g4(jr[0]), g4(float(jr[1])), g4(float(jr[2]))) == ("8.887", "3.294", "48.49")
    has("Joint RC for reduced LVEF gave 8.887 [3.294; 48.49]", MS)
    assert (rc1[3], rc1[4]) == (1000, 1000) and (int(rc4[3]), int(rc4[4])) == (989, 1000)
    unc1 = dict(kind="wald", **dict(zip(("o", "lo", "hi"), s3_wald("001")[:3])))
    unc4 = dict(kind="wald", **dict(zip(("o", "lo", "hi"), s3_wald("004")[:3])))
    ref1 = float(np.exp(cval("H1_error_raw", "uncorrected_beta")[0]))
    ref4 = float(np.exp(cval("H4_error_raw", "uncorrected_beta")[0]))
    assert f"{ref1:.3f}" == "2.343" and f"{ref4:.3f}" == "2.029"
    has("compared with 2.343 in assessment", MS)

    def c(aid_rc):
        o, lo, hi, v, a = cval(aid_rc, "or")
        return dict(kind="boot", o=o, lo=float(lo), hi=float(hi))

    def two(var, e):
        cl, ref, _ = steps[(var, e)]
        return cl, ref

    groups = []
    groups.append(("Reference", [
        dict(label="Uncorrected (Wald)", cells=[unc1, unc4])]))
    groups.append(("Regression calibration", [
        dict(label="Angle error only", cells=[c("H1_error_rc"), c("H4_error_rc")]),
        dict(label="Joint angle and marker error", cells=[c("H1_error_joint_rc"), NE])]))
    names = {"raw": "SIMEX, normal angle errors", "clipped": "SIMEX, angle bounded to 0\u2013180\u00b0".replace("\u2013", " to "),
             "logit": "SIMEX, logit-transformed angle", "spline": "SIMEX, flexible (spline) angle",
             "joint": "SIMEX, joint angle and marker error"}
    for var in ("raw", "clipped", "logit", "spline", "joint"):
        rows = []
        for k, mode in enumerate(("Linear extrapolation", "Quadratic extrapolation")):
            c1 = steps[(var, "H1")][0][k]
            c4 = steps[(var, "H4")][0][k] if (var, "H4") in steps else NE
            t1 = steps[(var, "H1")][1]
            t4 = steps[(var, "H4")][1] if (var, "H4") in steps else None
            rows.append(dict(label=mode, cells=[c1, c4], tick=[t1, t4]))
        groups.append((names[var], rows))

    def fmt3(o, lo, hi):
        lo_s = "<0.01" if lo < 0.005 else f"{lo:.2f}"
        return f"{o:.2f} ({lo_s}-{hi:.2f})"

    handles = [
        Line2D([0], [0], marker="o", color=PET, lw=1.3, ms=4.0, label="Wald interval (person-cluster)"),
        Line2D([0], [0], marker="D", color=OCKER, mec=ANTH, mew=0.5, lw=1.3, ms=3.8, label="Bootstrap percentile interval"),
        Line2D([0], [0], color=MITTELGRAU, lw=0.8, marker="None", label="Uncorrected estimate under the same angle model"),
    ]
    H = draw_forest(175.0, groups, ["Reduced LVEF (<40%)", "Coded cardiomyopathy"], (0.3, 64.0), [0.5, 1, 2, 5, 10, 30],
                    "Odds ratio per development SD (log scale)", "Fig3_measurement_error_assessment",
                    numfmt=fmt3, ax_w=36.0, num_w=24.0, label_w=46.0, legend_items=handles, ref_text="OR = 1")
    log(f"Fig. 3 written (H={H:.1f} mm)")


# =====================================================================================================
# Figure 4: measurement error, development partitions (training / internal validation)
# =====================================================================================================
def tri(s):
    """'2.289 [0.989; 4.502]' -> (o, lo, hi) and check against the supplement."""
    has(s, SUP)
    m = re.fullmatch(r"(\S+) \[(\S+); (\S+)\]", s)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def fig4():
    def boot(s):
        o, lo, hi = tri(s)
        return dict(kind="boot", o=o, lo=lo, hi=hi)

    def wald(s):
        o, lo, hi = tri(s)
        return dict(kind="wald", o=o, lo=lo, hi=hi)

    def pt(o_s):
        return dict(kind="point", o=float(o_s))
    # point estimates without interval: printed rows
    has("| training | simultaneous | rc | 2.335 [not estimable; not estimable] | 93.3% |".replace("93.3%", "93.3 %"), SUP)
    has("| internal validation | simultaneous | rc | 2.726 [not estimable; not estimable] | 115.0 % |", SUP)
    has("| Corrected training marker OR | 2.155 | CI not estimable, 17/500 valid draws; 25,388 persons / 2,672 cases |", SUP)
    has("| Corrected internal-validation marker OR | 2.184 | CI not estimable, 100/500 valid draws; 8,495 persons / 849 cases |", SUP)

    def block(part):
        """Rows per partition."""
        t = part == "training"
        D = {
            "unc": wald("2.480 [2.334; 2.625]") if t else wald("2.392 [2.143; 2.717]"),
            "tend": pt("2.155") if t else pt("2.184"),
            "rc_all": boot("2.289 [0.989; 4.502]") if t else boot("2.839 [0.772; 13.625]"),
            "rc_sim": pt("2.335") if t else pt("2.726"),
            "rc_icc70": boot("2.447 [2.183; 2.848]") if t else boot("2.468 [1.841; 3.089]"),
            "rc_icc617": boot("2.369 [1.706; 3.384]") if t else boot("2.656 [1.179; 5.134]"),
            "rc_icc55": NE, "rc_icc45": NE,
            "jrc_all": boot("8.737 [5.248; 21.525]") if t else boot("9.259 [3.795; 60.030]"),
            "jrc_sim": boot("3.589 [2.475; 30.006]") if t else boot("3.719 [1.866; 41.708]"),
            "sx_quad": boot("2.471 [2.278; 2.670]") if t else boot("2.413 [2.079; 2.882]"),
            "sx_lin": boot("2.478 [2.321; 2.641]") if t else boot("2.398 [2.120; 2.760]"),
            "sx_sim": boot("2.471 [2.277; 2.666]") if t else boot("2.412 [2.079; 2.873]"),
            "sx_icc70": boot("2.473 [2.327; 2.685]") if t else boot("2.409 [2.094; 2.724]"),
            "sx_icc55": boot("2.470 [2.319; 2.689]") if t else boot("2.414 [2.069; 2.751]"),
            "sx_icc45": boot("2.469 [2.313; 2.701]") if t else boot("2.417 [2.057; 2.765]"),
            "sx_clip": boot("2.480 [2.294; 2.707]") if t else boot("2.432 [2.134; 2.892]"),
            "sx_logit": boot("2.545 [2.360; 2.783]") if t else boot("2.571 [2.252; 2.991]"),
            "sx_spline": boot("2.539 [2.354; 2.773]") if t else boot("2.516 [2.154; 2.892]"),
            "sx_joint_lin": boot("2.806 [2.617; 3.031]") if t else boot("2.699 [2.369; 3.074]"),
            "sx_joint_quad": boot("3.074 [2.814; 3.374]") if t else boot("2.974 [2.571; 3.524]"),
        }
        return D
    Tr, Va = block("training"), block("internal validation")
    # RC at scenario ICC 0.55/0.45: not estimable (S10)
    has("| training | icc_0.45 | rc | not estimable [not estimable; not estimable] | not estimable | 0/200 |", SUP)
    has("| training | icc_0.55 | rc | not estimable [not estimable; not estimable] | not estimable | 0/200 |", SUP)
    has("| internal validation | icc_0.45 | rc | not estimable [not estimable; not estimable] | not estimable | 0/200 |", SUP)
    has("| internal validation | icc_0.55 | rc | not estimable [not estimable; not estimable] | not estimable | 2/200 |", SUP)

    def row(label, key, tick=None):
        return dict(label=label, cells=[Tr[key], Va[key]])
    groups = [
        ("Reference", [row("Uncorrected (Wald)", "unc")]),
        ("Boundary correction", [row("Individual T-end correction", "tend")]),
        ("Regression calibration", [
            row("Repeat-pair error variance", "rc_all"),
            row("Simultaneous repeat pairs", "rc_sim"),
            row("Assumed angle ICC 0.70", "rc_icc70"),
            row("Assumed angle ICC 0.617", "rc_icc617"),
            row("Assumed angle ICC 0.55", "rc_icc55"),
            row("Assumed angle ICC 0.45", "rc_icc45"),
            row("Joint error, repeat pairs", "jrc_all"),
            row("Joint error, simultaneous pairs", "jrc_sim")]),
        ("Simulation-extrapolation", [
            row("Repeat pairs, quadratic", "sx_quad"),
            row("Repeat pairs, linear", "sx_lin"),
            row("Simultaneous repeat pairs", "sx_sim"),
            row("Assumed angle ICC 0.70", "sx_icc70"),
            row("Assumed angle ICC 0.55", "sx_icc55"),
            row("Assumed angle ICC 0.45", "sx_icc45"),
            row("Angle bounded to 0 to 180\u00b0", "sx_clip"),
            row("Logit-transformed angle", "sx_logit"),
            row("Flexible (spline) angle", "sx_spline"),
            row("Joint error, linear", "sx_joint_lin"),
            row("Joint error, quadratic", "sx_joint_quad")]),
    ]
    handles = [
        Line2D([0], [0], marker="o", color=PET, lw=1.3, ms=4.0, label="Wald interval (person-cluster)"),
        Line2D([0], [0], marker="D", color=OCKER, mec=ANTH, mew=0.5, lw=1.3, ms=3.8, label="Bootstrap percentile interval"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec=PET, mew=1.0, ms=4.0, label="Point estimate only (interval not estimable)"),
    ]
    uo = Tr["unc"]
    vo = Va["unc"]
    H = draw_forest(175.0, groups, ["Training partition", "Internal validation partition"], (0.6, 64.0), [1, 2, 5, 10, 30],
                    "Odds ratio per development SD (log scale)", "Fig4_measurement_error_development",
                    numfmt=lambda o, lo, hi: f"{o:.2f} ({'<0.01' if lo < 0.005 else format(lo, '.2f')}-{hi:.2f})", ax_w=36.0, num_w=24.0,
                    label_w=46.0, legend_items=handles, band=[(uo["lo"], uo["hi"], uo["o"]), (vo["lo"], vo["hi"], vo["o"])])
    log(f"Fig. 4 written (H={H:.1f} mm)")


# =====================================================================================================
# Figure 5: known-truth simulation
# =====================================================================================================
def fig5():
    sim = {
        "null": [("No measurement error (oracle)", "0.994 [0.859; 1.145]"), ("Uncorrected", "1.247 [1.096; 1.395]"),
                 ("SIMEX", "1.118 [0.973; 1.271]"), ("Regression calibration", "0.993 [0.848; 1.171]"),
                 ("Joint regression calibration", "0.991 [0.800; 1.238]")],
        "true": [("No measurement error (oracle)", "1.951 [1.692; 2.296]"), ("Uncorrected", "2.070 [1.851; 2.354]"),
                 ("SIMEX", "1.856 [1.647; 2.140]"), ("Regression calibration", "1.617 [1.400; 1.895]"),
                 ("Joint regression calibration", "1.913 [1.579; 2.348]")],
    }
    for scen, lst in sim.items():
        for _, s in lst:
            has(s, SUP)
    has("| null | oracle | 200 | 1.000 | 0.994 [0.859; 1.145] |", SUP)
    has("| true | oracle | 200 | 1.934 | 1.951 [1.692; 2.296] |", SUP)
    obs = (2.480, 2.392)
    has("| training | all pairs | uncorrected | 2.480 [2.334; 2.625] |", SUP)
    has("| internal validation | all pairs | uncorrected | 2.392 [2.143; 2.717] |", SUP)
    has("uncorrected geometric-mean marker OR was 1.247 and SIMEX OR 1.118", MS)
    has("None of 200 null runs attained either observed effect", MS)
    # null rows: share >= observed = 0.000
    for r in ("| null | uncorrected | 200 | 1.000 | 1.247 [1.096; 1.395] | 0.221 | 1.000 | 0.000/0.000 |",
              "| null | simex | 200 | 1.000 | 1.118 [0.973; 1.271] | 0.112 | 0.521 | 0.000/0.000 |"):
        has(r, SUP)
    figH = 49.0
    fig = plt.figure(figsize=(175 * MM, figH * MM))
    label_w, ax_w = 46.0, 52.0
    n = 5
    row_h = 6.0
    top, bottom = 11.0, 11.0
    axs = []
    xs = [label_w + 3.0, label_w + 3.0 + ax_w + 14.0]
    # axis widths: two axes
    xs = [label_w + 2.0, label_w + 2.0 + ax_w + 8.0]
    ax_w = 58.0
    xs = [label_w + 2.0, label_w + 2.0 + ax_w + 8.0]
    ax_w = min(ax_w, (175 - 2.0 - xs[0] - 8.0) / 2.0)
    xs = [label_w + 2.0, label_w + 2.0 + ax_w + 8.0]
    lab = fig.add_axes([0, bottom / figH, label_w / 175.0, n * row_h / figH])
    lab.set_xlim(0, label_w)
    lab.set_ylim(n - 0.5, -1.4)
    lab.axis("off")
    for i, (name, _) in enumerate(sim["null"]):
        lab.text(1.5, i, name, fontsize=8, color=ANTH, ha="left", va="center")
    titles = {"null": ("A", "Null scenario (true OR = 1)"), "true": ("B", "Genuine-effect scenario (true OR = 1.934)")}
    for k, scen in enumerate(("null", "true")):
        a = fig.add_axes([xs[k] / 175.0, bottom / figH, ax_w / 175.0, n * row_h / figH])
        a.set_xscale("log")
        a.set_xlim(0.7, 3.0)
        a.set_ylim(n - 0.5, -1.4)
        a.set_yticks([])
        for sp in ("left", "top", "right"):
            a.spines[sp].set_visible(False)
        a.set_xticks([0.7, 1, 1.5, 2, 3])
        a.set_xticklabels(["0.7", "1", "1.5", "2", "3"])
        a.xaxis.set_minor_locator(plt.NullLocator())
        a.set_xlabel("Geometric-mean odds ratio per SD (log scale)", fontsize=8, labelpad=2)
        a.tick_params(axis="x", length=3, width=0.7, pad=2)
        fig.text((xs[k] - 2.0) / 175.0, 1 - 3.5 / figH, titles[scen][0], fontsize=10, fontweight="bold", color=ANTH, ha="right", va="center")
        fig.text((xs[k] - 0.5) / 175.0, 1 - 3.5 / figH, titles[scen][1], fontsize=8, fontweight="semibold", color=PET, ha="left", va="center")
        if scen == "null":
            a.axvline(1.0, color=MITTELGRAU, lw=0.8, zorder=1)
            for v, lab_t, dy in ((obs[0], "Observed, training", 0), (obs[1], "Observed, validation", 1)):
                a.axvline(v, color=ROST, ls="--", lw=0.9, zorder=1)
            a.text(obs[1] / 1.03, -0.95, "Observed OR 2.39 / 2.48", color=ROST, fontsize=8, ha="right", va="center")
        else:
            a.axvline(1.934, color=ROST, ls="--", lw=0.9, zorder=1)
            a.text(1.934 * 1.02, -0.95, "True OR 1.934", color=ROST, fontsize=8, ha="left", va="center")
        for i, (name, s) in enumerate(sim[scen]):
            o, lo, hi = tri(s)
            a.plot([lo, hi], [i, i], color=PET, lw=1.3, solid_capstyle="butt", zorder=3)
            a.plot([o], [i], marker="o", ms=4.0, mfc=PET, mec=PET, zorder=5)
    save_figure(fig, OUT / "Fig5_known_truth_simulation")
    log("Fig. 5 written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    (OUT / "abbildungen_b_v6_pruefprotokoll.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
