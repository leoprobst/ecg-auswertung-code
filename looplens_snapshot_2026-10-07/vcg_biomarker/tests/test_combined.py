import numpy as np
from vcg_biomarker.core.analysis.threshold_sweep import threshold_sweep
from vcg_biomarker.core.analysis.statistics import fdr_bh_correction
from vcg_biomarker.core.analysis.results import AnalysisRun
from itertools import combinations

np.random.seed(42)
n = 300

bm_names = ["qrs_duration", "spatial_qrs_t_angle_mean", "t_vector_magnitude"]
path_names = ["IMI", "LVH", "NORM"]

biomarker_data = {}
for name in bm_names:
    biomarker_data[name] = np.random.randn(n) * 10 + 50

pathology_data = {}
for name in path_names:
    pathology_data[name] = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

biomarker_data["qrs_duration"][pathology_data["IMI"] == 1] += 20
biomarker_data["spatial_qrs_t_angle_mean"][pathology_data["IMI"] == 1] += 15

results = []

for bname in bm_names:
    for pname in path_names:
        bv = biomarker_data[bname]
        lb = pathology_data[pname]
        valid = ~np.isnan(bv)
        r = threshold_sweep(bv[valid], lb[valid], bname, pname, bootstrap_iterations=10)
        results.append(r)

for combo in combinations(bm_names, 2):
    combo_name = " + ".join(combo)
    arrays = [biomarker_data[b] for b in combo]
    valid = np.ones(n, dtype=bool)
    for a in arrays:
        valid &= ~np.isnan(a)

    combined = np.zeros(n)
    for a in arrays:
        arr = a[valid]
        if np.std(arr) > 0:
            combined[valid] += (arr - np.mean(arr)) / np.std(arr)
        else:
            combined[valid] += arr

    for pname in path_names:
        lb = pathology_data[pname][valid]
        r = threshold_sweep(combined, lb, combo_name, pname, bootstrap_iterations=10)
        results.append(r)

run = AnalysisRun(
    run_id="test",
    biomarker_names=bm_names,
    pathology_names=path_names,
    vcg_method="kors",
    results=results,
)
run.apply_corrections()

df = run.to_dataframe()
print("Single biomarker results:")
single = df[~df["biomarker"].str.contains("\\+")]
print(single[["biomarker", "pathology", "auc", "threshold", "sensitivity", "specificity", "p_corrected", "significant"]].to_string())

print()
print("Combined biomarker results:")
combined = df[df["biomarker"].str.contains("\\+")]
print(combined[["biomarker", "pathology", "auc", "threshold", "sensitivity", "specificity", "p_corrected", "significant"]].to_string())

print()
print("=== Combined analysis test PASSED ===")
