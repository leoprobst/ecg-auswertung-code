"""Additional aggregation for the T-end covariate analysis (cases only, QRS duration changes in detail). Reads the records JSON of the T-end analysis (without identifiers)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
from pathlib import Path

import numpy as np

rows = json.loads(Path(str(data_path("b.aw241_tende_records_json"))).read_text(encoding="utf8"))
ok = [r for r in rows if "error" not in r]
FE = ["qrs_duration_ms", "qt_interval_ms", "qtc_fridericia_ms", "heart_rate_bpm", "qrs_t_normalized_discrete_frechet", "spatial_qrs_t_angle_deg"]
lines = [f"T-end covariance, addendum: recordings {len(rows)}, evaluated {len(ok)}, errors {len(rows) - len(ok)}; cases {sum(r['y'] == 1 for r in ok)}, controls {sum(r['y'] == 0 for r in ok)}"]
for grp, sel in (("Cases", [r for r in ok if r["y"] == 1]), ("Controls", [r for r in ok if r["y"] == 0])):
    for ms in (-32, -16, 16, 32):
        base = np.array([np.nan if r["0"][0] is None else r["0"][0] for r in sel], float)
        new = np.array([np.nan if r[str(ms)][0] is None else r[str(ms)][0] for r in sel], float)
        b = np.isfinite(base) & np.isfinite(new)
        d = (new - base)[b]
        ch = d[np.abs(d) > 1e-9]
        lines.append(f"{grp} {ms:+d} ms: QRS duration changed in {len(ch)} of {b.sum()} ({len(ch) / max(b.sum(), 1):.3f}); mean over all {d.mean():.3f} ms; among the changed: median |delta| {np.median(np.abs(ch)) if len(ch) else float('nan'):.1f} ms, 95th percentile {np.quantile(np.abs(ch), .95) if len(ch) else float('nan'):.1f} ms, maximum {np.abs(ch).max() if len(ch) else float('nan'):.1f} ms; sign positive {int((ch > 0).sum())}, negative {int((ch < 0).sum())}")
# QT: deviation of the change from the applied shift
for ms in (-32, 32):
    base = np.array([np.nan if r["0"][1] is None else r["0"][1] for r in ok if r["y"] == 1], float)
    new = np.array([np.nan if r[str(ms)][1] is None else r[str(ms)][1] for r in ok if r["y"] == 1], float)
    b = np.isfinite(base) & np.isfinite(new)
    d = (new - base)[b] - ms
    lines.append(f"Cases {ms:+d} ms: QT change deviates from the shift in {int((np.abs(d) > 1e-9).sum())} of {b.sum()} (median |deviation| of the deviating {np.median(np.abs(d[np.abs(d) > 1e-9])) if (np.abs(d) > 1e-9).any() else float('nan'):.1f} ms, maximum {np.abs(d).max():.1f} ms)")
    qb = np.array([np.nan if r["0"][2] is None else r["0"][2] for r in ok if r["y"] == 1], float)
    qn = np.array([np.nan if r[str(ms)][2] is None else r[str(ms)][2] for r in ok if r["y"] == 1], float)
    qt0 = base
    expected = qb * (qt0 + ms) / qt0 - qb
    bb = np.isfinite(qb) & np.isfinite(qn) & np.isfinite(expected)
    dd = (qn - qb)[bb] - expected[bb]
    lines.append(f"Cases {ms:+d} ms: QTc change versus the arithmetic recomputation QTc*(QT+shift)/QT: mean difference {dd.mean():.2f} ms, 95th percentile |difference| {np.quantile(np.abs(dd), .95):.2f} ms, maximum {np.abs(dd).max():.1f} ms, n {int(bb.sum())}")
Path(str(data_path("b.nachrechnung_aw241_tende_kovariaten_zusatz_ausgabe_txt"))).write_text("\n".join(lines), encoding="utf8")
print("\n".join(lines))
