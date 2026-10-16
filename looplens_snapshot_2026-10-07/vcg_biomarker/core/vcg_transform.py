from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vcg_geometry import (
    VCG_AXIS_CONVENTION,
    VCG_AXIS_CONVENTION_VERSION,
)


KORS_MATRIX = np.array([
    [-0.130,  0.050, -0.010,  0.140,  0.060,  0.540,  0.380, -0.070],
    [ 0.060, -0.020, -0.050,  0.060, -0.170,  0.130, -0.070,  0.930],
    [-0.430, -0.060, -0.140, -0.200, -0.110,  0.310,  0.110, -0.230],
])

KORS_CHANNELS = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II"]

KORS_REFERENCES = [
    "Kors JA, van Herpen G, Sittig AC, van Bemmel JH. Reconstruction of "
    "the Frank vectorcardiogram from standard electrocardiographic leads: "
    "diagnostic comparison of different methods. Eur Heart J. "
    "1990;11(12):1083-1092. PMID: 2292255.",
    "Kors JA, van Herpen G. Methodology of QRS vectorcardiography: the Frank lead system vs. correction systems based on 12-lead ECGs. J Electrocardiol. 2001;34 Suppl:11-7.",
]

INVERSE_DOWER_MATRIX = np.array([
    [-0.172, -0.074,  0.122,  0.231,  0.239,  0.194,  0.156, -0.010],
    [ 0.057, -0.019, -0.106, -0.022,  0.041,  0.048, -0.227,  0.887],
    [-0.229, -0.310, -0.246, -0.063,  0.055,  0.108,  0.022,  0.102],
])

INVERSE_DOWER_CHANNELS = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II"]

INVERSE_DOWER_REFERENCES = [
    "Dower GE, Machado HB, Osborne JA. On deriving the electrocardiogram from vectorcardiographic leads. Clin Cardiol. 1980;3(2):87-95.",
    "Edenbrandt L, Pahlm O. Vectorcardiogram synthesized from a 12-lead ECG: superiority of the inverse Dower matrix. J Electrocardiol. 1988;21(4):361-7.",
]


@dataclass
class VCGResult:
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    method_name: str
    channels_used: list[str]
    references: list[str]

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([self.vx, self.vy, self.vz])

    @property
    def axis_convention(self) -> str:
        """Coordinate and angular convention shared by all VCG consumers."""

        return VCG_AXIS_CONVENTION

    @property
    def axis_convention_version(self) -> str:
        return VCG_AXIS_CONVENTION_VERSION

    @property
    def spatial_magnitude(self) -> np.ndarray:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def spatial_velocity(self) -> np.ndarray:
        dx = np.diff(self.vx)
        dy = np.diff(self.vy)
        dz = np.diff(self.vz)
        return np.sqrt(dx**2 + dy**2 + dz**2)


class VCGTransformBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def channels(self) -> list[str]: ...

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def references(self) -> list[str]: ...

    def transform(self, ecg_12lead: np.ndarray) -> VCGResult:
        ecg_12lead = np.asarray(ecg_12lead, dtype=float)
        if ecg_12lead.ndim != 2 or ecg_12lead.shape[1] != 12:
            raise ValueError(
                "VCG transformation requires a samples-by-12 ECG in "
                f"standard lead order, got shape {ecg_12lead.shape}"
            )
        if not np.all(np.isfinite(ecg_12lead)):
            raise ValueError("VCG transformation requires finite ECG samples")
        channel_indices = self._get_channel_indices()
        ecg_subset = ecg_12lead[:, channel_indices]
        vcg_xyz = ecg_subset @ self.matrix.T

        return VCGResult(
            vx=vcg_xyz[:, 0],
            vy=vcg_xyz[:, 1],
            vz=vcg_xyz[:, 2],
            method_name=self.name,
            channels_used=self.channels,
            references=self.references,
        )

    def _get_channel_indices(self) -> list[int]:
        from .data_loader import LEAD_MAP
        indices = []
        for ch in self.channels:
            if ch not in LEAD_MAP:
                raise ValueError(f"Channel {ch} not found in standard 12-lead ECG")
            indices.append(LEAD_MAP[ch])
        return indices


class KorsTransform(VCGTransformBase):
    @property
    def name(self) -> str:
        return "kors"

    @property
    def channels(self) -> list[str]:
        return KORS_CHANNELS

    @property
    def matrix(self) -> np.ndarray:
        return KORS_MATRIX

    @property
    def references(self) -> list[str]:
        return KORS_REFERENCES


class InverseDowerTransform(VCGTransformBase):
    @property
    def name(self) -> str:
        return "inverse_dower"

    @property
    def channels(self) -> list[str]:
        return INVERSE_DOWER_CHANNELS

    @property
    def matrix(self) -> np.ndarray:
        return INVERSE_DOWER_MATRIX

    @property
    def references(self) -> list[str]:
        return INVERSE_DOWER_REFERENCES


class VCGTransformRegistry:
    _transforms: dict[str, VCGTransformBase] = {}

    @classmethod
    def register(cls, transform: VCGTransformBase) -> None:
        cls._transforms[transform.name] = transform

    @classmethod
    def get(cls, name: str) -> VCGTransformBase:
        if name not in cls._transforms:
            raise ValueError(f"VCG transform '{name}' not registered. Available: {list(cls._transforms.keys())}")
        return cls._transforms[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._transforms.keys())

    @classmethod
    def transform(cls, ecg_12lead: np.ndarray, method: str) -> VCGResult:
        return cls.get(method).transform(ecg_12lead)

    @classmethod
    def transform_all(cls, ecg_12lead: np.ndarray) -> dict[str, VCGResult]:
        results = {}
        for name, transform in cls._transforms.items():
            results[name] = transform.transform(ecg_12lead)
        return results


VCGTransformRegistry.register(KorsTransform())
VCGTransformRegistry.register(InverseDowerTransform())


def loop_dominant_axis(
    vcg_xyz: Optional[np.ndarray],
    start: int,
    end: int,
) -> Optional[np.ndarray]:
    """Dominant spatial axis of the VCG loop trajectory over [start, end],
    as a unit vector -- the direction of greatest positional spread of
    the (mean-centered) loop points, via SVD. This is an AXIS, not a
    signed direction: SVD's leading singular vector has an arbitrary
    sign, so callers must compare two axes via abs(dot product), never a
    raw dot product or a signed angle. Shared between the delineation
    core (used to disambiguate P/T candidates and retrograde-P search by
    spatial direction rather than amplitude alone) and VCG biomarkers
    (used to characterize inter-segment axis relationships), so both
    consumers get identical geometry from a single implementation.
    """
    if vcg_xyz is None:
        return None
    if end <= start or start < 0 or end >= len(vcg_xyz):
        return None
    window = np.asarray(vcg_xyz[start : end + 1], dtype=float)
    if window.shape[0] < 3 or window.shape[1] < 3:
        return None
    centered = window - np.mean(window, axis=0)
    if not np.any(np.linalg.norm(centered, axis=1) > 1e-9):
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    norm = np.linalg.norm(axis)
    if norm <= 1e-9:
        return None
    return axis / norm


def aggregate_loop_axes(axes: list[np.ndarray]) -> Optional[np.ndarray]:
    """Robust mean spatial axis from >=3 unit axes of arbitrary sign.
    Each axis is aligned to the running reference's sign before
    averaging, since an axis and its negation describe the same
    spatial line."""
    if len(axes) < 3:
        return None
    reference = axes[0]
    aligned = [reference]
    for axis in axes[1:]:
        aligned.append(-axis if np.dot(axis, reference) < 0 else axis)
    mean_axis = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean_axis)
    if norm <= 1e-9:
        return None
    return mean_axis / norm


def axis_angle_deg(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> Optional[float]:
    """Smallest angle (0-90 deg) between two undirected spatial axes."""
    if first is None or second is None:
        return None
    cos_similarity = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_similarity)))


def orthogonal_residual_magnitude(
    vcg_xyz: Optional[np.ndarray],
    axis: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Per-sample magnitude of the VCG trajectory after projecting out
    the component along `axis` -- a spatial (not temporal) SNR
    enhancement. Energy pointing along `axis` (e.g. a beat's own
    dominant QRS-loop direction, from loop_dominant_axis over its QRS
    window) is suppressed; energy pointing elsewhere (e.g. a P wave
    whose loop axis differs from the QRS axis) is relatively preserved.
    Returns a 1-D array the same length as `vcg_xyz`; `axis` need not be
    a unit vector, it is re-normalized here. Deliberately does not
    accept a window -- callers project the whole record once per axis
    and slice the result themselves, matching how a real ECG lead
    column is passed around in the delineator's candidate search.
    """
    if vcg_xyz is None or axis is None:
        return None
    data = np.asarray(vcg_xyz, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] == 0:
        return None
    unit_axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(unit_axis)
    if axis_norm <= 1e-9:
        return None
    unit_axis = unit_axis / axis_norm
    projection_length = data[:, :3] @ unit_axis
    residual = data[:, :3] - np.outer(projection_length, unit_axis)
    return np.linalg.norm(residual, axis=1)
