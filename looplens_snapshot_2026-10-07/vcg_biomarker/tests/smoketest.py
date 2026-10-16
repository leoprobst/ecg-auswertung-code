import numpy as np
from vcg_biomarker.config import AppConfig
from vcg_biomarker.core.pipeline import Pipeline
from vcg_biomarker.core.vcg_transform import VCGTransformRegistry
from vcg_biomarker.core.preprocessing import preprocess_record
from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
from vcg_biomarker.core.pathologies.registry import PathologyRegistry

fs = 500
duration = 3
n_samples = fs * duration
ecg = np.random.randn(n_samples, 12) * 0.1

for beat_n in range(3):
    center = int((beat_n + 0.5) * fs)
    width = int(0.08 * fs)
    if center + width < n_samples and center - width > 0:
        ecg[center-width:center+width, 1] += 1.0
        ecg[center-width:center, 1] -= 0.1

preprocessed = preprocess_record(ecg, fs)
print("Preprocessing OK:", preprocessed["cleaned"].shape)

vcg = VCGTransformRegistry.transform(preprocessed["cleaned"], "kors")
print("VCG transform OK:", vcg.vx.shape, vcg.method_name)

delineator = FusionDelineator()
delineation = delineator.delineate_record(
    ecg_12lead=preprocessed["cleaned"],
    r_peaks=preprocessed["r_peaks"],
    fs=fs,
    ecg_id=0,
    vcg_xyz=vcg.xyz,
)
print("Delineation OK:", delineation.n_beats, "beats, quality=", round(delineation.quality_score, 2))

biomarkers = BiomarkerRegistry.compute_all(
    delineation=delineation,
    ecg_12lead=preprocessed["cleaned"],
    vcg_xyz=vcg.xyz,
    fs=fs,
)
n_ok = sum(1 for b in biomarkers if b.value is not None)
print("Biomarkers OK:", n_ok, "/", len(biomarkers), "computed")

path_labels = PathologyRegistry.check_all({"NORM": 100})
print("Pathology check OK:", len(path_labels), "pathologies checked")

print()
print("=== End-to-end pipeline test PASSED ===")
