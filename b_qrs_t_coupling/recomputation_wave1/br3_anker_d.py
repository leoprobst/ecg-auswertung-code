"""Recomputation R3 (d): shift of the angle and the Frechet marker under an overall gain change of +1/+3/+5 % (aggregates of the gain-tolerance analysis, ALL12 scenarios). Aggregates only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import math
from pathlib import Path

t = json.loads(Path(str(data_path("b.aw204_toleranz_aggregate_json"))).read_text(encoding="utf8"))
SD = {"spatial_qrs_t_angle_deg": 46.8418, "qrs_t_normalized_discrete_frechet": 0.765}
lines = ["Recomputation R3 (d): mean signed change of the measured values under an overall gain change (gain-tolerance analysis, ALL12 scenarios, 384 to 385 ECGs), in development SD"]
for key, pct in (("ALL12_f1010", 1), ("ALL12_f1030", 3), ("ALL12_f1050", 5)):
    for var, lab in (("spatial_qrs_t_angle_deg", "angle"), ("qrs_t_normalized_discrete_frechet", "Frechet")):
        a = t["scenarios"][key][var]
        se = a["sd_delta"] / math.sqrt(a["n_pairs"])
        lo, hi = a["mean"] - 1.96 * se, a["mean"] + 1.96 * se
        lines.append(f"+{pct} % {lab}: mean {a['mean'] / SD[var]:.4f} SD (95 % {lo / SD[var]:.4f} to {hi / SD[var]:.4f}), RMS {a['rms'] / SD[var]:.3f} SD, n {a['n_pairs']}")
Path(str(data_path("b.nachrechnung_aw241_b_r3_anker_d_ausgabe_txt"))).write_text("\n".join(lines), encoding="utf8")
print("\n".join(lines))
