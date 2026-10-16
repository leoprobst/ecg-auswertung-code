from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from .delineation.base import measurement_confidence_threshold


@dataclass
class ReferenceLoop:
    name: str
    age_min: int
    age_max: int
    sex: str
    n_subjects: int
    qrs_mean: np.ndarray
    t_mean: np.ndarray
    p_mean: Optional[np.ndarray] = None
    qrs_cov: Optional[np.ndarray] = None
    t_cov: Optional[np.ndarray] = None
    fs: int = 500
    qrs_start_sample: int = 0
    t_start_sample: int = 0


class ReferenceLoopStore:
    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir
        self._references: dict[str, ReferenceLoop] = {}

    def _key(self, age: Optional[int], sex: Optional[str]) -> str:
        age_group = "all"
        if age is not None:
            if age < 30:
                age_group = "18-29"
            elif age < 50:
                age_group = "30-49"
            elif age < 70:
                age_group = "50-69"
            else:
                age_group = "70+"
        sex_key = sex if sex else "all"
        return f"{age_group}_{sex_key}"

    def compute_from_records(self, records: list, fs: int = 500) -> dict[str, ReferenceLoop]:
        from .pipeline import ProcessedRecord

        groups: dict[str, list[ProcessedRecord]] = {}

        for pr in records:
            if pr.error is not None or pr.vcg_result is None or pr.delineation is None:
                continue
            key = self._key(pr.ecg_record.age, pr.ecg_record.sex)
            if key not in groups:
                groups[key] = []
            groups[key].append(pr)

        for key, group in groups.items():
            if len(group) < 5:
                continue

            try:
                ref = self._compute_reference(key, group, fs)
                if ref is not None:
                    self._references[key] = ref
            except Exception:
                continue

        if self._cache_dir:
            self.save_to_disk(self._cache_dir)

        return self._references

    def _compute_reference(self, key: str, group: list, fs: int) -> Optional[ReferenceLoop]:
        from .pipeline import ProcessedRecord

        qrs_loops = []
        t_loops = []

        for pr in group:
            if pr.vcg_result is None or pr.delineation is None:
                continue
            vcg = pr.vcg_result
            beat = pr.delineation.representative_beat
            if beat is None:
                continue

            if beat.segment_confidence("qrs") >= measurement_confidence_threshold("qrs"):
                qrs_loops.append(vcg.xyz[beat.r_onset.sample:beat.r_offset.sample+1])
            if beat.segment_confidence("t") >= measurement_confidence_threshold("t"):
                t_loops.append(vcg.xyz[beat.t_onset.sample:beat.t_offset.sample+1])

        if len(qrs_loops) < 3 or len(t_loops) < 3:
            return None

        qrs_resampled = self._resample_loops(qrs_loops, target_len=50)
        t_resampled = self._resample_loops(t_loops, target_len=50)

        if len(qrs_resampled) == 0 or len(t_resampled) == 0:
            return None

        qrs_mean = np.mean(qrs_resampled, axis=0)
        t_mean = np.mean(t_resampled, axis=0)

        qrs_flat = qrs_resampled.reshape(len(qrs_resampled), -1)
        t_flat = t_resampled.reshape(len(t_resampled), -1)

        qrs_cov = np.cov(qrs_flat.T) if qrs_flat.shape[0] > 2 else None
        t_cov = np.cov(t_flat.T) if t_flat.shape[0] > 2 else None

        age_parts = key.split("_")[0]
        age_min, age_max = {
            "18-29": (18, 29), "30-49": (30, 49),
            "50-69": (50, 69), "70+": (70, 100), "all": (0, 120),
        }.get(age_parts, (0, 120))

        sex = key.split("_")[1] if "_" in key else "all"

        return ReferenceLoop(
            name=f"ref_{key}",
            age_min=age_min,
            age_max=age_max,
            sex=sex,
            n_subjects=len(group),
            qrs_mean=qrs_mean,
            t_mean=t_mean,
            qrs_cov=qrs_cov,
            t_cov=t_cov,
            fs=fs,
        )

    def _resample_loops(self, loops: list[np.ndarray], target_len: int = 50) -> np.ndarray:
        resampled = []
        for loop in loops:
            if len(loop) < 3:
                continue
            n = len(loop)
            indices = np.linspace(0, n - 1, target_len).astype(int)
            indices = np.clip(indices, 0, n - 1)
            resampled.append(loop[indices])
        if not resampled:
            return np.array([])
        return np.array(resampled)

    def get_reference(self, age: Optional[int] = None, sex: Optional[str] = None) -> Optional[ReferenceLoop]:
        key = self._key(age, sex)
        if key in self._references:
            return self._references[key]

        broader_key = self._key(None, sex)
        if broader_key in self._references:
            return self._references[broader_key]

        broadest_key = self._key(None, None)
        if broadest_key in self._references:
            return self._references[broadest_key]

        return None

    def compute_confidence_ellipse(self, ref: ReferenceLoop, loop_type: str = "qrs",
                                    n_std: float = 2.0, plane: str = "frontal",
                                    n_points: int = 100) -> Optional[np.ndarray]:
        cov = ref.qrs_cov if loop_type == "qrs" else ref.t_cov
        mean = ref.qrs_mean if loop_type == "qrs" else ref.t_mean

        if cov is None or mean is None:
            return None

        if plane == "frontal":
            idx = [0, 1]
        elif plane == "horizontal":
            idx = [0, 2]
        elif plane == "sagittal":
            idx = [1, 2]
        else:
            return None

        mean_2d = mean[:, idx].mean(axis=0)
        all_pts = mean[:, idx]
        cov_2d = np.cov(all_pts.T)

        if cov_2d.shape != (2, 2):
            return None

        eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = n_std * 2 * np.sqrt(eigenvalues[0])
        height = n_std * 2 * np.sqrt(eigenvalues[1])

        theta = np.linspace(0, 2 * np.pi, n_points)
        ellipse = np.column_stack([width / 2 * np.cos(theta), height / 2 * np.sin(theta)])

        rotation = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
        ellipse = ellipse @ rotation.T
        ellipse[:, 0] += mean_2d[0]
        ellipse[:, 1] += mean_2d[1]

        return ellipse

    def save_to_disk(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, ref in self._references.items():
            rows.append({
                "key": key,
                "name": ref.name,
                "age_min": ref.age_min,
                "age_max": ref.age_max,
                "sex": ref.sex,
                "n_subjects": ref.n_subjects,
                "fs": ref.fs,
            })
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(directory / "reference_index.csv", index=False)

            np.savez(
                directory / "reference_data.npz",
                **{f"qrs_mean_{key}": ref.qrs_mean for key, ref in self._references.items()},
                **{f"t_mean_{key}": ref.t_mean for key, ref in self._references.items()},
            )

    def load_from_disk(self, directory: Path):
        index_path = directory / "reference_index.csv"
        data_path = directory / "reference_data.npz"

        if not index_path.exists() or not data_path.exists():
            return

        df = pd.read_csv(index_path)
        data = np.load(data_path)

        for _, row in df.iterrows():
            key = row["key"]
            qrs_mean = data.get(f"qrs_mean_{key}")
            t_mean = data.get(f"t_mean_{key}")

            if qrs_mean is not None and t_mean is not None:
                self._references[key] = ReferenceLoop(
                    name=row["name"],
                    age_min=int(row["age_min"]),
                    age_max=int(row["age_max"]),
                    sex=row["sex"],
                    n_subjects=int(row["n_subjects"]),
                    qrs_mean=qrs_mean,
                    t_mean=t_mean,
                    fs=int(row.get("fs", 500)),
                )

    @property
    def available_references(self) -> list[str]:
        return list(self._references.keys())
