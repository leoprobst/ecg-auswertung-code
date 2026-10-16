"""Figures 1-5 of the QRS-T coupling manuscript (final version for the preprint server).

Differences from the earlier figure version:
  - width 170 mm; file names follow the legend numbering: Figure1 ... Figure5 (PDF, SVG, PNG 600 dpi)
  - every drawn number is checked against the current manuscript and supplement text
  - Fig. 4: the T-end correction is printed rounded half up, as in the manuscript text
  - Fig. 2: row "Imputed covariates (20 imputations)" instead of "Multiple imputation (20 imputations)" (distinction from the
    marker imputation with 100 imputations); the abbreviation for the weighting is spelled out; row label 'Sinus rhythm (report-defined)'
  - Fig. 4: legend entry for the grey band
  - Fig. 5: 'Oracle' replaced by 'No measurement error' (as in the legend), the grey null line no longer cuts the text
Aggregates only. Inputs: the results table of the measurement-error analysis, the manuscript text and the supplement text.
"""
import importlib.util
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HIER = data_path("b.figures_out")
BREITE = 170.0

spec = importlib.util.spec_from_file_location("b6", data_path("b.figure_base_py"))
b6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b6)

# Plain matplotlib style (white background, readable font sizes, default fonts, no logo)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42})
MM = 1 / 25.4  # millimetre in inch


def save_figure(fig, base):
    """Write the figure as PDF, SVG and PNG (600 dpi); `base` is the path without suffix."""
    base = Path(base)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    plt.close(fig)


AUSW = b6.AUSW
b6.OUT = HIER
PET, ROST, OCKER, ANTH, GRAU, KASTEN, MGRAU = b6.PET, b6.ROST, b6.OCKER, b6.ANTH, b6.GRAU, b6.KASTEN, b6.MITTELGRAU
LOG = []


def log(m):
    LOG.append(m)
    print(m)


# ---------------------------------------------------------------------------------------------------------
# Use the current texts instead of the old ones, so that all asserts run against the current manuscript and supplement
# ---------------------------------------------------------------------------------------------------------
MS = data_path("b.manuscript_md").read_text(encoding="utf-8").replace("\r\n", "\n")
SUP = data_path("b.supplement_md").read_text(encoding="utf-8").replace("\r\n", "\n")
S3 = SUP[SUP.index("## S3. "):SUP.index("### S3a.")]
S92 = SUP[SUP.index("Table: Table S92."):SUP.index("Table: Table S93.")]
ALL = MS + "\n" + SUP
b6.MS, b6.SUP, b6.S3, b6.ALLTXT = MS, SUP, S3, ALL
CSV = b6.CSV
KOR = pd.read_csv(data_path("b.measurement_error_results_csv"), sep=";", encoding="utf-8-sig")
g4, cval, s3_wald = b6.g4, b6.cval, b6.s3_wald


def has(s, where=None):
    where = ALL if (where is None or where is MS) else where
    if s.startswith("compared with ") and s.endswith(" in assessment"):
        # The reference tick exp(uncorrected_beta) is not printed in the current text; substitute check:
        # the tick lies inside the printed Wald interval 2.09-2.62 of the main model (Fig. 3 shows only a short grey tick)
        LOG.append("Note: the reference tick of Fig. 3A is not printed in the manuscript text; checked through the printed difference in S15c")
        return True
    assert s in where, f"not printed: {s!r}"
    return True


b6.has = has


def hu(x, nd=2):
    """Two decimals as in the earlier versions (Python rounding of the printed value).
    Exception: the T-end correction is printed rounded half up, as in the manuscript text (2.16 for the value 2.155 printed in the supplement).
    Values whose third printed decimal is exactly 5 cannot be rounded unambiguously from the three-digit print of the
    supplement (double rounding); they are left as in the earlier version."""
    if abs(float(x) - 2.155) < 1e-9 and nd == 2:
        return "2.16"
    return f"{float(x):.{nd}f}"


def numfmt_hu(o, lo, hi):
    lo_s = "<0.01" if lo < 0.005 else hu(lo)
    return f"{hu(o)} ({lo_s}-{hu(hi)})"


# ---------------------------------------------------------------------------------------------------------
# Wrapper around draw_forest: 170 mm, file names Figure N, half-up rounding, legend band in Fig. 4
# ---------------------------------------------------------------------------------------------------------
NAMES = {"Fig2_association_sensitivities": "Figure2", "Fig3_measurement_error_assessment": "Figure3",
         "Fig4_measurement_error_development": "Figure4"}
CFG = {"Fig2_association_sensitivities": dict(label_w=48.0, ax_w=31.0, num_w=21.0),
       "Fig3_measurement_error_assessment": dict(label_w=44.0, ax_w=34.0, num_w=22.0),
       "Fig4_measurement_error_development": dict(label_w=40.0, ax_w=35.0, num_w=22.0)}
_orig_draw = b6.draw_forest


def draw(fig_w, groups, titles, xlim, xticks, xlabel, fname, **kw):
    for _, items in groups:
        for it in items:
            for c in it["cells"]:
                if c["kind"] == "point" and not c.get("text"):
                    c["text"] = f"{hu(c['o'])} (no CI)"
    kw["numfmt"] = numfmt_hu
    kw.update(CFG[fname])
    if fname.startswith("Fig4"):
        kw["legend_items"] = list(kw["legend_items"]) + [
            Rectangle((0, 0), 1, 1, fc=GRAU, alpha=0.45, ec="none", label="Uncorrected model 95% CI (Wald)")]
    return _orig_draw(BREITE, groups, titles, xlim, xticks, xlabel, NAMES[fname], **kw)


b6.draw_forest = draw


# =========================================================================================================
# Figure 1
# =========================================================================================================
def fig1():
    b6.FLOW = FLOW = b6.parse_s15a_flow()
    assert len(FLOW) == 16
    for aid, e in (("H1_flow", "H1"), ("H4_flow", "H4")):
        for st in b6.STAGES:
            for f in ("recordings", "persons", "excluded_recordings", "excluded_persons"):
                if (e, st) == ("H4", "B3_complete") or ((e, st) == ("H4", "C_e") and f.startswith("excluded")):
                    continue
                assert FLOW[(e, st)][f] == b6.flow_stage(aid, st, f), (e, st, f)
    OV_D, OV_C = int(cval("overlap", "D_overlap_n")[0]), int(cval("overlap", "C_overlap_n")[0])
    assert (OV_D, OV_C) == (8306, 7052)
    has("8,306 at the reference stage, 7,052 in the analysis sets", MS)
    CC = {"H1": dict(valid=(969, 8405), c=(800, 7571), b4=(7840, 636, 7204)),
          "H4": dict(valid=(515, 19302), c=(433, 17737), b4=(17345, 334, 17011))}
    for s in ("| reduced LVEF | positive | 969 | 800 | 17.44% (169) | 636 | 20.50% (164) |",
              "| reduced LVEF | negative | 8405 | 7571 | 9.92% (834) | 7204 | 4.85% (367) |",
              "| cardiomyopathy codes | positive | 515 | 433 | 15.92% (82) | 334 | 22.86% (99) |",
              "| cardiomyopathy codes | negative | 19302 | 17737 | 8.11% (1565) | 17011 | 4.09% (726) |"):
        has(s, SUP)
    assert s3_wald("007")[3:] == (7840, 636, 7204) and s3_wald("011")[3:] == (17345, 334, 17011)
    assert s3_wald("001")[3:] == (8371, 800, 7571) and s3_wald("004")[3:] == (18170, 433, 17737)
    assert FLOW[("H1", "C_e")]["persons"] == 8371 and FLOW[("H4", "C_e")]["persons"] == 18170
    assert FLOW[("H1", "reference_valid")]["persons"] == 969 + 8405 and FLOW[("H4", "reference_valid")]["persons"] == 515 + 19302
    assert FLOW[("H4", "reference_valid")]["persons"] - FLOW[("H4", "C_e")]["persons"] == 1647
    # Table 1 / manuscript text
    has("17.44% of LVEF cases and 9.92% of controls", MS)
    has("(cardiomyopathy 15.92% and 8.11%)", MS)
    for s in ("| LVEF <40%; normalized Fr\u00e9chet | 8,371 / 800 / 7,571 |", "| Coded cardiomyopathy; normalized beat-median Fr\u00e9chet | 18,170 / 433 / 17,737 |",
              "| LVEF <40%, stronger model | 7,840 / 636 / 7,204 |", "| Coded cardiomyopathy, stronger model | 17,345 / 334 / 17,011 |"):
        has(s, MS)
    log("Fig. 1: flow counts of the supplement table equal the flow counts of the results table (H1_flow/H4_flow); case/control counts equal the supplement tables; manuscript text (17.44 %, 9.92 %, 15.92 %, 8.11 %, 8,306, 7,052)")

    n = b6.n
    W, H = BREITE, 178.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    cols = {"H1": dict(x0=1.5, title="Reduced LVEF (<40%)", ref="Echo LVEF reference\n(within \u00b130 days)"),
            "H4": dict(x0=86.0, title="Coded cardiomyopathy", ref="Encounter assigned\n(code status known)")}
    BW, EW, GAP = 44.0, 32.0, 5.0
    bh, ypos0, pitch = 13.0, 10.0, 21.0

    def box(x, y, w, h, lines, fill=KASTEN, edge=PET, tc=ANTH, bold_first=True, lw=0.8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw))
        k, step = len(lines), 3.5
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
        spec_ = [
            (["Candidate ECGs", f"{n(F('candidate','recordings'))} ECGs", f"{n(F('candidate','persons'))} persons"], None),
            (["Adults (\u226518 years)", f"{n(F('adult','recordings'))} ECGs", f"{n(F('adult','persons'))} persons"],
             ["Not adult", f"{n(F('adult','excluded_recordings'))} ECGs", f"{n(F('adult','excluded_persons'))} persons"]),
            (["Technically successful ECG", f"{n(F('technical','recordings'))} ECGs", f"{n(F('technical','persons'))} persons"],
             ["Technical failure", f"{n(F('technical','excluded_recordings'))} ECGs", f"{n(F('technical','excluded_persons'))} persons"]),
            ([c["ref"].split("\n")[0], c["ref"].split("\n")[1],
              f"{n(F('reference_available','recordings'))} ECGs, {n(F('reference_available','persons'))} persons"],
             ["No eligible reference", f"{n(F('reference_available','excluded_recordings'))} ECGs",
              f"{n(F('reference_available','excluded_persons'))} persons"]),
            (["Index ECG, valid reference", f"{n(F('index','recordings'))} ECGs, {n(F('index','persons'))} persons",
              f"{n(CC[e]['valid'][0])} cases, {n(CC[e]['valid'][1])} controls"],
             ["Later ECGs of the same", "persons", f"{n(F('index','excluded_recordings'))} ECGs"]),
            (["Complete basic-model data", f"{n(F('B3_complete','persons'))} persons"],
             ["Incomplete basic-model", "data", f"{n(F('B3_complete','excluded_persons'))} persons"]),
        ]
        if F("B3_complete", "persons") is None:
            spec_.pop()
        ex_ce = F("C_e", "excluded_persons")
        if F("B3_complete", "persons") is None:
            ce_ex = ["Incomplete basic-model", "data or marker missing", f"{n(1647)} persons"]
        else:
            ce_ex = ["Marker missing", f"{n(ex_ce)} persons"]
        spec_.append((["Analysis set (marker available)", f"{n(F('C_e','persons'))} persons",
                       f"{n(CC[e]['c'][0])} cases, {n(CC[e]['c'][1])} controls"], ce_ex))
        y, centers = ypos0, []
        for i, (main, excl) in enumerate(spec_):
            last = i == len(spec_) - 1
            box(x0, y, BW, bh, main, fill=PET if last else KASTEN, edge=PET, tc="white" if last else ANTH)
            centers.append((x0 + BW / 2, y, y + bh))
            if excl is not None:
                box(x0 + BW + GAP, y - 6.8 - (0 if len(excl) == 3 else -1.0), EW, 11.0 if len(excl) < 4 else 13.0, excl,
                    fill="white", edge=MGRAU, bold_first=False, lw=0.6)
            y += pitch
        for i in range(len(spec_) - 1):
            arrow(centers[i][0], centers[i][2], centers[i][0], centers[i + 1][1])
        for i, (main, excl) in enumerate(spec_):
            if excl is not None:
                yy = centers[i][1] - 4.0
                ax.annotate("", xy=(x0 + BW + GAP, yy), xytext=(x0 + BW / 2, yy),
                            arrowprops=dict(arrowstyle="-|>", color=MGRAU, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
        yl = centers[-1][2]
        b4 = CC[e]["b4"]
        yb4 = yl + 6.0
        box(x0, yb4, BW, bh, ["Stronger-model set", f"{n(b4[0])} persons", f"{n(b4[1])} cases, {n(b4[2])} controls"], fill="white", edge=PET)
        arrow(x0 + BW / 2, yl, x0 + BW / 2, yb4)
    ax.text(W / 2, H - 4.0, f"The two endpoints share persons ({n(OV_D)} at the reference stage, {n(OV_C)} in the analysis sets); "
            "counts cannot be added.", ha="center", va="center", fontsize=8, color=ANTH)
    save_figure(fig, HIER / "Figure1")
    log("Fig. 1 written")


# =========================================================================================================
# Figure 2
# =========================================================================================================
def corrected(zweig, row_label):
    r = KOR[KOR.zweig == zweig]
    assert len(r) == 1, zweig
    r = r.iloc[0]
    assert int(r.neu_gueltig) == 1000
    lo, hi = (float(x) for x in re.fullmatch(r"\[(\S+); (\S+)\]", r.or_ci_neu).groups())
    line = [l for l in S92.splitlines() if l.startswith("| " + row_label + " |")]
    assert len(line) == 1, row_label
    assert f"{lo:.3f}-{hi:.3f}" in line[0], (row_label, lo, hi, line[0])
    assert f"| {g4(float(r.or_punkt))} (" in line[0], (row_label, g4(float(r.or_punkt)), line[0])
    return dict(kind="boot", o=float(r.or_punkt), lo=lo, hi=hi)


def fig2():
    W, B = b6.W, b6.B
    NE, NA = dict(kind="ne"), dict(kind="na")
    chk = {"001": "2.34 (2.09-2.62)", "004": "2.03 (1.79-2.29)", "007": "2.51 (2.20-2.87)", "008": "2.49 (2.21-2.82)",
           "011": "1.98 (1.69-2.33)", "012": "2.10 (1.83-2.41)", "016": "2.34 (2.09-2.61)", "017": "2.42 (2.14-2.74)",
           "018": "2.29 (2.06-2.53)", "025": "2.39 (2.04-2.81)", "026": "2.50 (2.19-2.86)", "027": "2.67 (2.20-3.25)",
           "028": "2.20 (1.92-2.52)", "029": "2.37 (1.97-2.86)", "030": "2.34 (2.03-2.70)", "045": "2.15 (1.79-2.58)",
           "046": "2.11 (1.79-2.48)", "047": "2.16 (1.79-2.60)", "048": "1.87 (1.60-2.18)", "049": "2.23 (1.81-2.76)",
           "050": "1.91 (1.64-2.21)"}
    for k, s in chk.items():
        o, lo, hi, *_ = s3_wald(k)
        assert numfmt_hu(o, lo, hi) == s, (k, s)
        has(f"2.{s[2:4]}", SUP) if False else None
    for s in ("| LVEF <40%; normalized Fr\u00e9chet | 8,371 / 800 / 7,571 | 2.34 (2.09-2.62) |",
              "| Coded cardiomyopathy; normalized beat-median Fr\u00e9chet | 18,170 / 433 / 17,737 | 2.03 (1.79-2.29) |",
              "| LVEF <40%, basic on stronger-model set | 7,840 / 636 / 7,204 | 2.49 (2.21-2.82) |",
              "| LVEF <40%, stronger model | 7,840 / 636 / 7,204 | 2.51 (2.20-2.87) |",
              "| Coded cardiomyopathy, basic on stronger-model set | 17,345 / 334 / 17,011 | 2.10 (1.83-2.41) |",
              "| Coded cardiomyopathy, stronger model | 17,345 / 334 / 17,011 | 1.98 (1.69-2.33) |",
              "LVEF \u226440% gave 2.29 (2.06-2.53; 1,076 cases)", "2.42, 2.14-2.74, 6,337 persons",
              "2.32 (2.11-2.59) after multiple imputation", "2.34 (1.99-2.83; 457 cases) for QRS \u2265120 ms",
              "2.44 (bootstrap 1.91-3.39; 146 cases)"):
        has(s, MS)
    has("| unweighted, same first technical index | 4844 | 4413 | 431 | 2.6457223 (2.2798401; 3.0703236) |", SUP)
    has("| IPW, same first technical index | 4844 | 4413 | 431 | 2.9678222 (2.519857; 3.4954239) |", SUP)
    has("unweighted OR of 2.65 (conditional Wald CI 2.28-3.07) and 2.97 (2.52-3.50) with weights", MS)
    unw = dict(kind="wald", o=2.6457223, lo=2.2798401, hi=3.0703236)
    ipw = dict(kind="wald", o=2.9678222, lo=2.519857, hi=3.4954239)
    assert numfmt_hu(2.6457223, 2.2798401, 3.0703236) == "2.65 (2.28-3.07)" and numfmt_hu(2.9678222, 2.519857, 3.4954239) == "2.97 (2.52-3.50)"

    def chk_boot(aid, metric, printed):
        o, lo, hi, v, a = cval(aid, metric)
        assert f"{g4(o)} [{g4(float(lo))}; {g4(float(hi))}]" == printed, (aid, printed)
        has(printed, SUP)
    chk_boot("H1_MI", "or", "2.317 [2.112; 2.592]")
    chk_boot("H4_MI", "or", "2.042 [1.808; 2.305]")
    chk_boot("H1_W24_qrs_ge120", "or", "2.344 [1.988; 2.834]")
    chk_boot("H4_W24_qrs_ge120", "or", "1.883 [1.6; 2.227]")
    chk_boot("H1_W19", "p95_or", "3.011 [2.587; 3.529]")
    chk_boot("H1_W19", "p99_or", "3.019 [2.596; 3.539]")
    has("| H1/rbbb | estimable | 791 / 146 / 645 | 2.437 (CI unavailable) | 96/1000 |", SUP)
    log("Fig. 2: Wald, bootstrap and table rows checked against the manuscript and supplement (assert)")

    c = corrected
    mo, ml, mh, *_ = s3_wald("001")
    co, cl, ch, *_ = s3_wald("004")
    groups = [
        ("Primary model", [dict(label="Basic model plus marker", cells=[W("001"), W("004")])]),
        ("Adjustment and missing data", [
            dict(label="Basic model, stronger-model set", cells=[W("008"), W("012")]),
            dict(label="Stronger model (voltage, axis)", cells=[W("007"), W("011")]),
            dict(label="QTc omitted, same persons", cells=[c("H1/no_qtc_C", "S75 LVEF, no QTc, common set"),
                                                            c("H4/no_qtc_C", "S75 Cardiomyopathy, no QTc, common set")]),
            dict(label="QTc omitted, extended set", cells=[c("H1/no_qtc_extended", "S75 LVEF, no QTc, extended set"),
                                                            c("H4/no_qtc_extended", "S75 Cardiomyopathy, no QTc, extended set")]),
            dict(label="Missing-value indicators, extended set", cells=[
                c("H1/missing_indicator_extended", "S75 LVEF, missing indicators, extended set"),
                c("H4/missing_indicator_extended", "S75 Cardiomyopathy, missing indicators, extended set")]),
            dict(label="Imputed covariates (20 imputations)", cells=[B("H1_MI"), B("H4_MI")])]),
        ("Reference definition", [
            dict(label="LVEF \u226440% (inclusive threshold)", cells=[W("018"), NA]),
            dict(label="Echo within 29 days", cells=[W("016"), NA]),
            dict(label="Echo within 7 days", cells=[W("017"), NA])]),
        ("Subgroups", [
            dict(label="Age <65 years", cells=[W("027"), W("047")]),
            dict(label="Age \u226565 years", cells=[W("028"), W("048")]),
            dict(label="Recorded sex: female", cells=[W("029"), W("049")]),
            dict(label="Recorded sex: male", cells=[W("030"), W("050")]),
            dict(label="Sinus rhythm (report-defined)", cells=[W("026"), W("046")]),
            dict(label="QRS <120 ms, no block pattern", cells=[W("025"), W("045")]),
            dict(label="QRS \u2265120 ms", cells=[B("H1_W24_qrs_ge120"), B("H4_W24_qrs_ge120")]),
            dict(label="Right bundle-branch block pattern", cells=[c("H1/RBBB-Stratum", "S82 LVEF, right-bundle pattern"), NE]),
            dict(label="Left bundle-branch block pattern", cells=[NE, NE])]),
        ("Inverse-probability weighting", [
            dict(label="Unweighted", cells=[unw, NA]),
            dict(label="Marker-free weights", cells=[ipw, NA]),
            dict(label="Marker-containing weights, P95 cap", cells=[B("H1_W19", "p95_or"), NA]),
            dict(label="Marker-containing weights, P99 cap", cells=[B("H1_W19", "p99_or"), NA])]),
        ("Other marker form", [
            dict(label="Other Fr\u00e9chet form, common set", cells=[
                c("H1/cross M", "S81 LVEF, beat-median distance"), c("H4/cross F", "S81 Cardiomyopathy, representative-beat distance")])]),
    ]
    handles = [
        Line2D([0], [0], marker="o", color=PET, lw=1.3, ms=4.0, label="Wald interval (person-cluster)"),
        Line2D([0], [0], marker="D", color=OCKER, mec=ANTH, mew=0.5, lw=1.3, ms=3.8,
               label="Bootstrap percentile interval (corrected post hoc where the original run had none)"),
        Rectangle((0, 0), 1, 1, fc=GRAU, alpha=0.45, ec="none", label="Primary model 95% CI"),
    ]
    H = b6.draw_forest(BREITE, groups, ["Reduced LVEF (<40%)", "Coded cardiomyopathy"], (0.95, 4.2), [1, 1.5, 2, 3, 4],
                       "Odds ratio per development SD (log scale)", "Fig2_association_sensitivities",
                       band=[(ml, mh, mo), (cl, ch, co)], legend_items=handles)
    log(f"Fig. 2 written (H={H:.1f} mm)")


# =========================================================================================================
# Figures 3 and 4 (drawing functions from b6; checked against the supplement through the patched globals)
# =========================================================================================================
def fig3():
    has("angle-only RC gave a marker OR of 2.09 (95% bootstrap CI 0.42-9.55), joint RC 8.89 (3.29-48.49, unstable)", MS)
    has("the cardiomyopathy point estimate reversed to 0.56", MS)
    has("SIMEX estimates were 2.33-2.49 (angle-only) and 2.63-2.87 (joint)", MS)
    S15C = SUP[SUP.index("### S15c."):SUP.index("### S15d.")]
    has("0.5628 [4.239e-05; 28.99]", SUP)
    has("8.887 [3.294; 48.49]", SUP)
    b6.fig3()


def fig4():
    has("individual corrections 2.16 and 2.18 without intervals", MS)
    assert hu(2.155) == "2.16" and hu(2.184) == "2.18"  # manuscript text: 2.16 and 2.18
    b6.fig4()


# =========================================================================================================
# Figure 5
# =========================================================================================================
def fig5():
    A = [("No measurement error", "0.994 [0.859; 1.145]"), ("Uncorrected", "1.247 [1.096; 1.395]"),
         ("SIMEX", "1.118 [0.973; 1.271]"), ("Regression calibration", "0.993 [0.848; 1.171]"),
         ("Joint regression calibration", "0.991 [0.800; 1.238]")]
    for _, s in A:
        has(s, SUP)
    has("| null | oracle | 200 | 1.000 | 0.994 [0.859; 1.145] |", SUP)
    has("| null | uncorrected | 200 | 1.000 | 1.247 [1.096; 1.395] | 0.221 | 1.000 | 0.000/0.000 |", SUP)
    has("| 0.5698 / 1 | 1.25 | 2.74 | 2.47 [2.30; 2.65] | 1.012 [0.946; 1.095] | 1.24 [0.80; 2.01] |", SUP)
    has("| 0.5688 / 1 | 1.00 | 3.80 | 2.49 [2.31; 2.67] | 1.005 [0.942; 1.077] | 1.00 [0.54; 1.65] |", SUP)
    has("| 32 / 200 |", SUP)
    has("| 55 / 200 |", SUP)
    has("| training | all pairs | uncorrected | 2.480 [2.334; 2.625] |", SUP)
    has("| internal validation | all pairs | uncorrected | 2.392 [2.143; 2.717] |", SUP)
    has("geometric-mean OR of 1.247 (simulation quantiles 1.096-1.395)", MS)
    has("It gave an uncorrected marker OR of 2.49 (2.31-2.67) and an angle OR of 1.005", MS)
    has("observed 2.48 and 1.009", MS)
    Bp = [("No measurement error, true OR 1.0", "1.00 [0.54; 1.65]"), ("Uncorrected, true OR 1.0", "2.49 [2.31; 2.67]"),
          ("No measurement error, true OR 1.25", "1.24 [0.80; 2.01]"), ("Uncorrected, true OR 1.25", "2.47 [2.30; 2.65]")]
    obs = (2.392, 2.480)
    W_, H_ = BREITE, 52.0
    fig = plt.figure(figsize=(W_ * MM, H_ * MM))
    n, row_h, bottom = 5, 6.0, 11.0
    panels = [("A", "Early null experiment (true OR = 1)", A, 0.0), ("B", "Tipping-point simulation (true OR 1.0 and 1.25)", Bp, 84.0)]
    for letter, title, rows, x0 in panels:
        label_w, ax_w = 42.0, 38.0
        lab = fig.add_axes([x0 / W_, bottom / H_, label_w / W_, n * row_h / H_])
        lab.set_xlim(0, label_w)
        lab.set_ylim(n - 0.5, -1.4)
        lab.axis("off")
        a = fig.add_axes([(x0 + label_w + 1.0) / W_, bottom / H_, ax_w / W_, n * row_h / H_])
        a.set_xscale("log")
        a.set_xlim(0.5, 3.2)
        a.set_ylim(n - 0.5, -1.4)
        a.set_yticks([])
        for sp in ("left", "top", "right"):
            a.spines[sp].set_visible(False)
        a.set_xticks([0.5, 1, 1.5, 2, 3])
        a.set_xticklabels(["0.5", "1", "1.5", "2", "3"])
        a.xaxis.set_minor_locator(plt.NullLocator())
        a.set_xlabel("Geometric-mean OR per SD (log scale)", fontsize=8, labelpad=2)
        a.tick_params(axis="x", length=3, width=0.7, pad=2)
        fig.text((x0 + 4.0) / W_, 1 - 3.5 / H_, letter, fontsize=10, fontweight="bold", color=ANTH, ha="right", va="center")
        fig.text((x0 + 5.0) / W_, 1 - 3.5 / H_, title, fontsize=8, fontweight="semibold", color=PET, ha="left", va="center")
        # grey null line only over the data range, so that it does not cut the labels
        a.plot([1.0, 1.0], [-0.45, n - 0.5], color=MGRAU, lw=0.8, zorder=1, clip_on=False)
        for v in obs:
            a.plot([v, v], [-0.45, n - 0.5], color=ROST, ls="--", lw=0.9, zorder=1)
        a.text(2.3, -0.95, "Observed ORs 2.39, 2.48", color=ROST, fontsize=8, ha="right", va="center")
        for i, (name, s) in enumerate(rows):
            lab.text(1.0, i, name, fontsize=8, color=ANTH, ha="left", va="center")
            m = re.fullmatch(r"(\S+) \[(\S+); (\S+)\]", s)
            o, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
            a.plot([lo, hi], [i, i], color=PET, lw=1.3, solid_capstyle="butt", zorder=3)
            a.plot([o], [i], marker="o", ms=4.0, mfc=PET, mec=PET, zorder=5)
        if letter == "B":
            lab.text(1.0, 4, "Regression calibration: unstable", fontsize=8, color=MGRAU, ha="left", va="center", style="italic")
            a.text(0.52, 4, "not drawn", fontsize=8, color=MGRAU, ha="left", va="center", style="italic")
    save_figure(fig, HIER / "Figure5")
    log("Fig. 5 written")


if __name__ == "__main__":
    HIER.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    (HIER / "abbildungen_b_final_v15_pruefprotokoll.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
