"""Final figures for preprint A (aggregate data only).

Output: PDF, SVG and PNG (600 dpi) per figure; width at most 170 mm.
Naming follows the figure legend numbers:
  Fig1_repeatability_key_measures        (main text, four measures from Table 2)
  Fig2_icc_distribution                  (main text)
  FigS1_catalog_matrix_classes
  FigS2_matrix_context
  FigS3_LUDB_boundary_errors             (copied unchanged from an earlier figure set, checked against Table 5)
  FigS4_repairs_delta_auc                (reference line "Fixed technical gain 0.02")

Sources (aggregates only):
  supplementary_data_S2_marker_reliability_public_v4.csv   (Fig 1, Fig 2)
  supplementary_data_S2_marker_matrix_public_v4.csv        (Fig S2)
  rv_a1_zusatzanalysen.csv                                 (Fig S1: bootstrap intervals of the class counts)
  T4_varianten.csv                                         (Fig S4)
  LUDB_AW140_boundary_summary.csv                          (check of Table 5 for Fig S3)
Cross-check by assert against the manuscript and supplement text files read below.
Usage: python -I abbildungen_a_final_v27.py   (under 1 GB RAM, about 30 s)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import hashlib
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
AUSW = Path(str(data_path("a.auswertung")))

PUB = AUSW / "publikationen" / "PUB-01_messvaliditaet"
OUT = PUB / "abbildungen" / "final_v27"
REL = PUB / "supplementary_data_S2_marker_reliability_public_v4.csv"
MAT = PUB / "supplementary_data_S2_marker_matrix_public_v4.csv"
RV = AUSW / "nachrechnung" / "rv_a1_zusatzanalysen.csv"
T4 = PUB / "T4_varianten.csv"
LUDB = PUB / "abbildungen" / "v10" / "LUDB_AW140_boundary_summary.csv"
LUDB_PNG_STEM = PUB / "abbildungen" / "v12" / "FigS1_LUDB_boundary_errors"
MS = (PUB / "MANUSKRIPT_PREPRINT_A_v25.md").read_text(encoding="utf-8")
SUP = (PUB / "SUPPLEMENT_PREPRINT_v22.md").read_text(encoding="utf-8")
MM = 1 / 25.4
B = 168 * MM  # design width; with a tight bounding box every file stays below 170 mm
MAXBREITE_MM = 170.0
GRAU = "#6B6B6B"
PETROL = "#1F5E73"
RUST = "#B5532A"
OCHRE = "#D9A441"
CHARCOAL = "#222222"
LIGHTGRAY = "#D9D9D9"
LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


def apply_style():
    """Plain matplotlib style: white background, DejaVu Sans, readable sizes."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": CHARCOAL, "axes.labelcolor": CHARCOAL, "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL, "text.color": CHARCOAL,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": LIGHTGRAY, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 2,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=CHARCOAL)


def reference_line(ax, x=None, y=None, text=None, **kw):
    if x is not None:
        ax.axvline(x, color=RUST, ls="--", lw=0.9, **kw)
        if text:
            ax.text(x, 1.0, " " + text, transform=ax.get_xaxis_transform(), color=RUST,
                    fontsize=7, ha="left", va="top")
    if y is not None:
        ax.axhline(y, color=RUST, ls="--", lw=0.9, **kw)
        if text:
            ax.text(1.0, y, text + " ", transform=ax.get_yaxis_transform(), color=RUST,
                    fontsize=7, ha="right", va="bottom")


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext))
    plt.close(fig)


def nums(s):
    return [float(x.replace("\u2212", "-").replace(",", "")) for x in re.findall(r"[+\u2212-]?\d[\d,]*\.?\d*", s)]


def table_rows(text, caption_start):
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Table: " + caption_start))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("|"))
    rows = []
    for l in lines[j + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return rows


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol


def rnd(x, k):
    """Round half up, as in the table display."""
    return float(np.floor(abs(float(x)) * 10 ** k + 0.5 + 1e-9) / 10 ** k * np.sign(x))


def need(text, needle, where):
    assert needle in text, f"{where}: '{needle}' is missing"


apply_style()
plt.rcParams["axes.unicode_minus"] = True
rel = pd.read_csv(REL, low_memory=False).set_index("marker_identifier")
assert len(rel) == 2005

# =============================================================== Figure 1 (four measures from Table 2)
MEAS = [  # (label, marker_identifier, row name in Table 2)
    ("QRS duration", "qrs_duration_ms", "QRS duration (ms)"),
    ("QTcF", "qtc_fridericia_ms", "QTcF (ms)"),
    ("Spatial area QRS-T angle", "spatial_qrs_t_angle_deg", "Spatial area QRS-T angle (\u00b0)"),
    ("P duration, multilead fusion", "p_duration_ms", "P duration, multilead fusion (ms)"),
]
t2 = table_rows(MS, "Table 2.")
assert len(t2) == 4
n_ok = 0
for lab, mid, pref in MEAS:
    row = next(r for r in t2 if r[0] == pref)
    d = rel.loc[mid]
    icc, lo, hi = nums(row[2])
    rc, rlo, rhi = nums(row[4])
    pairs, pers = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", row[1])]
    assert near(rnd(d.all_pairs_intraclass_correlation, 4), icc) and near(rnd(d.all_pairs_intraclass_correlation_confidence_limit_lower, 4), lo) \
        and near(rnd(d.all_pairs_intraclass_correlation_confidence_limit_upper, 4), hi), (mid, "ICC")
    assert near(rnd(d.all_pairs_bias, 2), nums(row[3])[0]), (mid, "bias")
    assert near(rnd(d.all_pairs_repeatability_coefficient, 2), rc) and near(rnd(d.all_pairs_repeatability_coefficient_confidence_limit_lower, 2), rlo) \
        and near(rnd(d.all_pairs_repeatability_coefficient_confidence_limit_upper, 2), rhi), (mid, "RC")
    assert int(d.all_pairs_finite_pairs) == pairs and int(d.all_pairs_finite_persons) == pers, (mid, "n")
    assert near(rnd(d.stability_proxy_intraclass_correlation, 4), nums(row[5])[0]), (mid, "Proxy-ICC")
    assert near(rnd(d.stability_proxy_repeatability_coefficient, 2), nums(row[6])[0]), (mid, "Proxy-RC")
    n_ok += 1
log(f"Fig 1: {n_ok}/4 rows of Table 2 (ICC, 95% interval, bias, RC, 95% interval, pairs, persons, proxy ICC, proxy RC) match the public file marker_reliability_public_v4.")
for lab, mid, _ in MEAS:  # proxy intervals: compare with the supplement where they are printed
    d = rel.loc[mid]
    s1 = f"{d.stability_proxy_intraclass_correlation:.4f} [{d.stability_proxy_intraclass_correlation_confidence_limit_lower:.4f}, {d.stability_proxy_intraclass_correlation_confidence_limit_upper:.4f}]"
    s2 = f"{d.stability_proxy_repeatability_coefficient:.2f} [{d.stability_proxy_repeatability_coefficient_confidence_limit_lower:.2f}, {d.stability_proxy_repeatability_coefficient_confidence_limit_upper:.2f}]"
    log(f"  Proxy {mid}: ICC {s1} in the supplement: {s1 in SUP}; RC {s2} in the supplement: {s2 in SUP} (otherwise the source is only the public CSV)")
need(MS, "ICC \u22650.75 line is an author-chosen reporting convention", "Methods 2.3")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(B, 52 * MM), sharey=True, gridspec_kw={"wspace": 0.06})
ypos = np.arange(len(MEAS))
OFFS = {"all": -0.17, "proxy": 0.17}
STY = {"all": dict(color=PETROL, marker="o"), "proxy": dict(color=OCHRE, marker="s")}
LAB = {"all": "All clinical pairs", "proxy": "Report-defined stability proxy"}
PRE = {"all": "all_pairs_", "proxy": "stability_proxy_"}
for pop in ("all", "proxy"):
    for ax, stem in ((axa, "intraclass_correlation"), (axb, "repeatability_coefficient")):
        vals = np.array([rel.loc[m[1]][PRE[pop] + stem] for m in MEAS], dtype=float)
        los = np.array([rel.loc[m[1]][PRE[pop] + stem + "_confidence_limit_lower"] for m in MEAS], dtype=float)
        his = np.array([rel.loc[m[1]][PRE[pop] + stem + "_confidence_limit_upper"] for m in MEAS], dtype=float)
        ax.errorbar(vals, ypos + OFFS[pop], xerr=[vals - los, his - vals], fmt=STY[pop]["marker"], color=STY[pop]["color"],
                    ms=4, elinewidth=1.0, capsize=1.8, capthick=0.8, label=LAB[pop], zorder=3, ecolor=STY[pop]["color"])
for ax in (axa, axb):
    for y in ypos[:-1]:
        ax.axhline(y + 0.5, color=LIGHTGRAY, lw=0.5, zorder=0)
axa.set_yticks(ypos)
axa.set_yticklabels([m[0] for m in MEAS])
axa.set_ylim(len(MEAS) - 0.5, -0.5)
axa.set_xlim(0, 1)
axa.set_xlabel("ICC(A,1), absolute agreement")
axa.axvline(0.75, color=RUST, ls="--", lw=0.9, zorder=1)
axa.text(0.735, 0.985, "Reporting convention\nICC = 0.75", transform=axa.get_xaxis_transform(), color=RUST, fontsize=7,
         ha="right", va="top", linespacing=1.05)
axb.set_xlim(0, 100)
axb.set_xlabel("Repeatability coefficient (ms; degrees for the angle)")
axb.tick_params(axis="y", length=0)
panel(axa, "A")
axb.text(-0.02, 1.02, "B", transform=axb.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold", color=CHARCOAL)
h, l = axa.get_legend_handles_labels()
axb.legend(h, l, loc="upper right", bbox_to_anchor=(1.0, 0.99), handletextpad=0.4)
save_figure(fig, OUT / "Fig1_repeatability_key_measures")

# =============================================================== Figure 2 (ICC distribution)
icc = pd.to_numeric(rel.all_pairs_intraclass_correlation, errors="coerce")
circ = rel.circular_marker.astype(str).str.lower() == "yes"
assert circ.sum() == 29 and icc.notna().sum() == 1987
sel = icc.notna() & ~circ
x = icc[sel].astype(float)
assert len(x) == 1958
med = float(np.median(x))
n75 = int((x >= 0.75).sum())
share = n75 / len(x) * 100
assert round(med, 3) == 0.468 and round(med, 4) == 0.4678
assert n75 == 311 and round(share, 2) == 15.88
need(MS, "the median ICC over the 1,958 estimable non-circular markers was 0.468 (311 reached 0.75; 15.88%)", "Results 3.2")
need(SUP, "| Median ICC, all pairs | 0.4678 |", "S2i-3")
allf = icc[icc.notna()].astype(float)
assert round(float(np.median(allf)), 4) == 0.4665 and int((allf >= 0.75).sum()) == 313
need(MS, "0.4665 [0.4591, 0.4778] (313 reached 0.75; 15.75%", "Results 3.2 historical set")
FAM = [("durations", "Durations", "#0072B2"), ("amplitudes_areas", "Amplitudes and areas", "#E69F00"),
       ("shape", "Shape measures", "#009E73"), ("ratios", "Ratios", "#CC79A7"),
       ("angles", "Angles", "#D55E00"), ("counters_flags", "Counts and flags", "#56B4E9")]
S3D = {"amplitudes_areas": (716, 0.3770), "durations": (58, 0.6777), "shape": (418, 0.4884), "ratios": (360, 0.4975),
       "angles": (261, 0.4370), "counters_flags": (174, 0.5177)}
for fam, (n_est, m_est) in S3D.items():  # family values of the full set (including circular markers) against Table S3d-1
    v = allf[rel.loc[allf.index, "family"] == fam]
    assert len(v) == n_est and near(round(float(np.median(v)), 4), m_est, 1e-9), fam
need(SUP, "| all | amplitudes_areas | 719 | 716 | 0.377 |", "S3d-1")
fam_n = {}
fam_med = {}
for fam, _, _ in FAM:
    v = x[rel.loc[x.index, "family"] == fam]
    fam_n[fam] = len(v)
    fam_med[fam] = float(np.median(v))
assert sum(fam_n.values()) == 1958
log(f"Fig 2: n = {len(x)} (non-circular, estimable markers), median {med:.4f}, {n75} with ICC >= 0.75 = {share:.2f}%; "
    f"full set with 29 circular markers: 1987, median {np.median(allf):.4f}, 313; family values (medians and numbers of estimable markers) of the full set match Table S3d-1.")
log("  Families (non-circular): " + "; ".join(f"{lab} n={fam_n[f]} median {fam_med[f]:.3f}" for f, lab, _ in FAM))
n_neg = int((x < 0).sum())
log(f"  ICC < 0 for {n_neg} markers (minimum {x.min():.4f}); maximum {x.max():.4f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(B, 70 * MM), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.25})
edges = np.round(np.arange(-0.05, 1.0001, 0.05), 2)
cnt, _ = np.histogram(x, bins=edges)
assert cnt.sum() == 1958 and x.min() >= edges[0] and x.max() < edges[-1]
axa.bar(edges[:-1], cnt, width=0.05, align="edge", color=PETROL, edgecolor="white", linewidth=0.4, zorder=2)
ymax = cnt.max() * 2.0
axa.set_ylim(0, ymax)
axa.set_xlim(-0.05, 1.0)
axa.axvline(med, ymax=0.62, color=CHARCOAL, lw=0.9, zorder=3)
axa.axvline(0.75, color=RUST, ls="--", lw=0.9, zorder=3)
axa.text(med, ymax * 0.635, f"Median {med:.3f}", ha="center", va="bottom", fontsize=7, color=CHARCOAL)
axa.text(0.735, ymax * 0.985, f"Reporting convention\nICC = 0.75: {n75} of {len(x):,}\nmarkers ({share:.2f}%)\nat or above", ha="right",
         va="top", fontsize=7, color=RUST, linespacing=1.1)
axa.set_xlabel("ICC(A,1) between recordings 0-2 h apart")
axa.set_ylabel("Markers (n)")
panel(axa, "A")
xs = np.sort(x.values)
yy = np.arange(1, len(xs) + 1) / len(xs) * 100
axb.step(xs, yy, where="post", color=CHARCOAL, lw=1.8, zorder=2, label=f"All markers ({len(x):,})")
for fam, lab, col in FAM:
    v = np.sort(x[rel.loc[x.index, "family"] == fam].values)
    axb.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=col, lw=1.0, zorder=3, label=f"{lab} ({len(v):,})")
axb.axvline(0.75, color=RUST, ls="--", lw=0.9, zorder=1)
axb.axhline(50, color=LIGHTGRAY, lw=0.5, zorder=0)
axb.set_xlim(-0.05, 1.0)
axb.set_ylim(0, 100)
axb.set_xlabel("ICC(A,1) between recordings 0-2 h apart")
axb.set_ylabel("Cumulative share of markers (%)")
axb.text(0.76, 3, "0.75", color=RUST, fontsize=7, ha="left", va="bottom")
hpt, lpt = axb.get_legend_handles_labels()
fig.legend(hpt, lpt, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17), handlelength=1.6, columnspacing=1.4, labelspacing=0.3)
pb = axb.get_position()
fig.text(pb.x0 - 0.075, pb.y1 + 0.012, "B", ha="right", va="bottom", fontsize=10, fontweight="bold", color=CHARCOAL)
save_figure(fig, OUT / "Fig2_icc_distribution")

# =============================================================== Figure S1 (former S4): class composition
rv = pd.read_csv(RV, low_memory=False)
tc = rv[(rv.block == "transform_class_counts") & (rv.universe == "katalog")].set_index(["normalization", "class"])
CLS = [("Dependent", "transformabhaengig"), ("Conditional", "bedingt"), ("Stable", None),
       ("Identical at shared fiducials,\nnonconstant", "stabil"), ("Identical at shared fiducials,\nconstant", "konstant_identisch"),
       ("Not estimable", "nicht_schaetzbar"), ("Not testable", "nicht_pruefbar")]
exp = {"transformabhaengig": 1071, "bedingt": 684, "stabil": 217, "konstant_identisch": 1, "nicht_schaetzbar": 31, "nicht_pruefbar": 1}
for k, v in exp.items():
    assert int(tc.loc[("Kors_SD", k)].point) == v, k
N_CAT = 2005
assert sum(exp.values()) == N_CAT
tcl = rel  # public file: same counts
mt = pd.read_csv(MAT, usecols=["marker_identifier", "technical_class", "primary_matrix_class", "boundary_shift_4_ms_worst_class",
                               "affine_conversion_summary_class", "matrix_halfwidth_to_all_pairs_repeatability_coefficient_ratio"], low_memory=False)
cc = mt.technical_class.value_counts().to_dict()
assert (cc["dependent"], cc["conditional"], cc["matrix_independent_identical"], cc["constant_identical"], cc["not_estimable"], cc["not_testable"]) == (1071, 684, 217, 1, 31, 1)
for needle in ("1,071 dependent", "684 conditional", "217 identical nonconstant outputs", "(214 without them)", "31 not estimable", "1 not testable"):
    need(MS, needle, "Table 4 / note")
need(MS, "| Dependent | 1,045 (60.4%) |", "Table 4")
assert round(float(tc.loc[("Kors_SD", "transformabhaengig")].fraction_universe_point) * 100, 1) == 53.4
need(SUP, "Percentages refer to all 2,005 identifiers", "legend of former S4")
log("Fig S1: class counts 1071/684/0/217/1/31/1 (sum 2005) = public file matrix_public_v4 = Table 4 and note (historical set; 1045/684/214 without the 29 circular markers); intervals from the class-count interval table (transform_class_counts, Kors_SD and pooled_SD).")


def cls_val(norm, key):
    if key is None:
        return 0.0, 0.0, 0.0
    r = tc.loc[(norm, key)]
    return float(r.point), float(r.ci_low), float(r.ci_high)


fig, ax = plt.subplots(figsize=(B, 66 * MM))
n = len(CLS)
for i, (lab, key) in enumerate(CLS):
    p, lo, hi = cls_val("Kors_SD", key)
    ax.barh(i, p, height=0.42, color=PETROL, zorder=2)
    ax.plot([lo, hi], [i, i], color=CHARCOAL, lw=0.9, zorder=3, solid_capstyle="butt")
    for xx in (lo, hi):
        ax.plot([xx, xx], [i - 0.1, i + 0.1], color=CHARCOAL, lw=0.9, zorder=3)
    pp, plo, phi = cls_val("pooled_SD", key)
    ax.plot([plo, phi], [i + 0.33, i + 0.33], color=OCHRE, lw=0.9, zorder=3)
    ax.plot(pp, i + 0.33, marker="D", ms=3.0, color=OCHRE, mec=OCHRE, zorder=4, ls="none")
    frac = p / N_CAT * 100
    ftxt = "<0.1%" if 0 < frac < 0.1 else f"{frac:.1f}%"
    ax.text(max(hi, phi) + 18, i, f"{int(round(p)):,} ({ftxt})", va="center", ha="left", fontsize=7)
ax.set_yticks(range(n))
ax.set_yticklabels([c[0] for c in CLS])
ax.set_ylim(n - 0.5, -0.55)
ax.set_xlim(0, 1400)
ax.set_xlabel("Catalog identifiers (n; total = 2,005)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
hand = [Patch(color=PETROL, label="Primary rule (Kors SD normalization), 95% interval"),
        Line2D([0], [0], marker="D", ms=3.0, color=OCHRE, label="Sensitivity (pooled SD normalization), 95% interval")]
ax.legend(handles=hand, loc="lower right", bbox_to_anchor=(1.0, 0.22), handlelength=1.4)
save_figure(fig, OUT / "FigS1_catalog_matrix_classes")

# =============================================================== Figure S2 (former S3): matrix context
dep = mt[mt.primary_matrix_class.isin(["dependent", "conditional"])]
assert len(dep) == 1755
b4 = dep.boundary_shift_4_ms_worst_class.value_counts().to_dict()
cal = dep.affine_conversion_summary_class.value_counts().to_dict()
assert (b4["stable"], b4["conditional"], b4["dependent"], b4["not_estimable"]) == (295, 933, 496, 31)
assert (cal["conditional"], cal["dependent"]) == (683, 1072) and cal.get("stable", 0) == 0 and cal.get("not_estimable", 0) == 0
ratio = pd.to_numeric(dep.matrix_halfwidth_to_all_pairs_repeatability_coefficient_ratio, errors="coerce").dropna().astype(float)
assert len(ratio) == 1755
q1, medr, q3 = np.percentile(ratio, [25, 50, 75])
assert (round(medr, 3), round(q1, 3), round(q3, 3)) == (0.544, 0.342, 0.776)
need(SUP, "295 stable, 933 conditional, 496 dependent, 0 constant-identical and 31 not estimable", "S2g")
need(SUP, "Final classes were 1,072 dependent and 683 conditional", "S2g")
need(SUP, "median ratio of matrix LoA half-width to repeat RC was 0.544 (IQR 0.342-0.776)", "S2g")
need(SUP, "changed 20 of 1,071 dependent outputs to conditional", "S2g")
n_low = int((ratio < 0.01).sum())
log(f"Fig S2: Kors/Kors +-4 ms {b4}; calibrated {cal}; ratio n = {len(ratio)}, median {medr:.4f}, IQR {q1:.4f}-{q3:.4f}; {n_low} outputs below 0.01 in the first bar; "
    f"values from matrix_public_v4, texts in S2g cross-checked.")
assert n_low == 3
fig, (a, bb) = plt.subplots(1, 2, figsize=(B, 82 * MM), gridspec_kw={"width_ratios": [1.05, 1]})
klassen = [("stable", "Stable"), ("conditional", "Conditional"), ("dependent", "Dependent"), ("not_estimable", "Not estimable")]
y = np.arange(len(klassen))[::-1]
h = 0.36
v1 = [b4.get(k, 0) for k, _ in klassen]
v2 = [cal.get(k, 0) for k, _ in klassen]
a.barh(y + h / 2, v1, h, color=PETROL, label="Kors versus Kors, boundaries shifted by \u00b14 ms (500 ECGs)")
a.barh(y - h / 2, v2, h, color=OCHRE, label="Inverse Dower versus Kors after cross-fitted\naffine calibration (5,648 ECGs)")
for yy_, v in zip(y + h / 2, v1):
    a.text(v + 15, yy_, f"{v:,}", va="center", fontsize=7, color=CHARCOAL)
for yy_, v in zip(y - h / 2, v2):
    a.text(v + 15, yy_, f"{v:,}", va="center", fontsize=7, color=CHARCOAL)
a.set_yticks(y)
a.set_yticklabels([nm for _, nm in klassen])
a.set_xlim(0, 1400)
a.set_xlabel("Non-identical estimable outputs (n; of 1,755)")
a.legend(loc="upper center", fontsize=7, frameon=False, handlelength=1.2, bbox_to_anchor=(0.45, -0.2))
panel(a, "A")
lg = np.log10(ratio.clip(lower=1e-3))
edges2 = np.arange(-2.0, 1.01, 0.1)
cnt2, _ = np.histogram(lg.clip(-2.0 + 1e-9, 1.0 - 1e-9), bins=edges2)
assert cnt2.sum() == 1755
bb.bar(edges2[:-1], cnt2, width=0.1, align="edge", color=PETROL, edgecolor="white", linewidth=0.3)
reference_line(bb, x=0.0, text="Equal to repeat RC")
bb.axvline(np.log10(medr), color=CHARCOAL, lw=0.9)
bb.text(-1.95, max(cnt2) * 0.98, f"Black line: median {medr:.3f}\n(IQR {q1:.3f}-{q3:.3f})", ha="left", va="top", fontsize=7, color=CHARCOAL)
bb.text(-1.95, max(cnt2) * 0.80, f"{n_low} outputs below 0.01\nare in the first bar", ha="left", va="top", fontsize=6.5, color=GRAU)
ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
bb.set_xticks(np.log10(ticks))
bb.set_xticklabels([f"{t:g}" for t in ticks])
bb.set_xlim(-2.05, 1.05)
bb.set_xlabel("Matrix limits-of-agreement half-width \u00f7 repeat RC (log scale)")
bb.set_ylabel("Outputs (n)")
panel(bb, "B")
fig.tight_layout(w_pad=2.0)
save_figure(fig, OUT / "FigS2_matrix_context")

# =============================================================== Figure S3 (former S1): LUDB, copied unchanged + check against Table 5
lu = pd.read_csv(LUDB)
lu = lu[(lu.scope == "fused_global") & (lu.lead == "global")].set_index("landmark")
t5 = table_rows(MS, "Table 5.")
MAPB = {"P onset": "p_onset", "P offset": "p_offset", "QRS onset": "r_onset", "QRS offset": "r_offset", "T onset": "t_onset", "T offset": "t_offset"}
assert len(t5) == 6
for r in t5:
    d = lu.loc[MAPB[r[0]]]
    npair, nref = [int(v) for v in nums(r[1])]
    assert (int(d.n_paired), int(d.n_reference_rows)) == (npair, nref), r[0]
    m, sd = nums(r[2])
    assert near(round(d.mean_error_ms, 2), m) and near(round(d.sd_error_ms, 2), sd), (r[0], "mean/SD")
    md, mlo, mhi = nums(r[3])
    assert (d.median_error_ms, d.median_error_ci_low_ms, d.median_error_ci_high_ms) == (md, mlo, mhi), (r[0], "Median")
    ma, alo, ahi = nums(r[4])
    assert near(round(d.mae_ms, 2), ma) and near(round(d.mae_ci_low_ms, 2), alo) and near(round(d.mae_ci_high_ms, 2), ahi), (r[0], "MAE")
log("Fig S3: 6/6 rows of Table 5 (n, mean, SD, median, MAE, intervals) match the LUDB boundary summary table; the panels shown (five boundaries, without T onset) come from the earlier LUDB boundary-error figure (histogram medians and n as in Table 5), copied unchanged.")
for ext in (".pdf", ".svg", ".png"):
    src = LUDB_PNG_STEM.with_suffix(ext)
    dst = OUT / ("FigS3_LUDB_boundary_errors" + ext)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert hashlib.md5(src.read_bytes()).hexdigest() == hashlib.md5(dst.read_bytes()).hexdigest()

# =============================================================== Figure S4 (former S2): repairs
va = pd.read_csv(T4).reset_index(drop=True)
assert len(va) == 16
va["repair"] = va.index + 1
t6 = [l for l in SUP[SUP.index("| T6. Repair"):].split("\n") if l.startswith("| Repair ")][:16]
assert len(t6) == 16
for (_, r), l in zip(va.iterrows(), t6):
    c = [v.strip() for v in l.strip().strip("|").split("|")]
    assert int(c[0].split()[1]) == r.repair
    dv = nums(c[4])
    assert near(dv[0], round(r.delta_auc, 4), 1e-9) and near(dv[1], round(r.ci_low, 4), 1e-9) and near(dv[2], round(r.ci_high, 4), 1e-9), c[0]
    oa, va_ = nums(c[3])
    assert near(oa, round(r.original_auc, 4), 1.1e-4) and near(va_, round(r.variant_auc, 4), 1e-9), c[0]
assert (va.ci_high < 0.02).all()
need(SUP, "No variant achieved the fixed technical 0.02 point gain", "S4a")
log("Fig S4: 16 repairs (original AUC, repair AUC, delta AUC, interval) match Table T6 of the supplement and the repair variants table; upper interval limit < 0.02 for all; reference line as 'Fixed technical gain 0.02' (S4a).")
fig, axs = plt.subplots(1, 2, figsize=(B, 60 * MM), sharex=True, gridspec_kw={"wspace": 0.25})
for ax, (lo_i, hi_i), let in zip(axs, ((0, 8), (8, 16)), "AB"):
    d = va.iloc[lo_i:hi_i]
    yv = np.arange(len(d))
    ax.errorbar(d.delta_auc, yv, xerr=[d.delta_auc - d.ci_low, d.ci_high - d.delta_auc], fmt="o", color=PETROL, ms=4,
                elinewidth=1.0, capsize=1.8, capthick=0.8, zorder=3)
    ax.axvline(0, color=GRAU, lw=0.7, zorder=1)
    ax.axvline(0.02, color=RUST, ls="--", lw=0.9, zorder=1)
    ax.set_yticks(yv)
    ax.set_yticklabels([f"Repair {int(k)}" for k in d.repair])
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlim(-0.32, 0.05)
    ax.set_xlabel("Change in AUC (repair minus original)")
    ax.text(0.02, 1.0, "Fixed technical\ngain 0.02", transform=ax.get_xaxis_transform(), color=RUST, fontsize=7,
            ha="left", va="bottom", linespacing=1.0)
    panel(ax, let)
    ax.yaxis.grid(True, color=LIGHTGRAY, lw=0.5)
    ax.set_axisbelow(True)
save_figure(fig, OUT / "FigS4_repairs_delta_auc")

# =============================================================== Width and file check
NAMEN = ["Fig1_repeatability_key_measures", "Fig2_icc_distribution", "FigS1_catalog_matrix_classes", "FigS2_matrix_context",
         "FigS3_LUDB_boundary_errors", "FigS4_repairs_delta_auc"]
for nm in NAMEN:
    for ext in (".pdf", ".svg", ".png"):
        assert (OUT / (nm + ext)).exists() and (OUT / (nm + ext)).stat().st_size > 1000, nm + ext
    im = Image.open(OUT / (nm + ".png"))
    dpi = im.info.get("dpi", (0, 0))[0]
    wmm = im.size[0] / 600 * 25.4
    hmm = im.size[1] / 600 * 25.4
    assert abs(dpi - 600) < 0.1 and wmm <= MAXBREITE_MM, (nm, dpi, wmm)
    log(f"File {nm}: PDF, SVG, PNG present; PNG {im.size[0]} x {im.size[1]} px at {dpi:.0f} dpi = {wmm:.1f} x {hmm:.1f} mm (limit {MAXBREITE_MM:.0f} mm)")
(OUT / "abbildungen_a_final_v27_pruefprotokoll.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("done")
