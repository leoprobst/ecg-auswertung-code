"""Thorax-Silhouette, Elektrodenpositionen und Dipol-Näherung für virtuelle Ableitungen.

Der Thorax ist parametrisch (Superellipsen-Querschnitte, gelofted), in Modell-
koordinaten (mm, +x links, −y anterior, +z kranial) um das Herz gelegt; das Herz
liegt links der Mittellinie und retrosternal. Elektrodenorte folgen den Standard-
positionen (V1/V2 4. ICS parasternal, V4 5. ICS medioklavikular, V6 mittlere
Axillarlinie; V7–V9 posterior, V3R/V4R rechts) und werden auf die Hautoberfläche
projiziert.

Virtuelle Ableitung: Ein einzelner Stromdipol im unendlichen homogenen Leiter,
φ(p, t) = k · D(t)·(p − c) / |p − c|³, mit D(t) aus der Kors-Transformation der
gemessenen Ableitungen. Der Maßstab k wird je Aufzeichnung an V1–V6 angepasst,
die erreichte Übereinstimmung wird angezeigt. Das ist eine Hypothesenebene (H):
Nahfeldeffekte, Inhomogenität und Lungen fehlen; posteriore und rechtsseitige
Ableitungen sind daher grobe Näherungen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .timeline import KORS, BeatTimeline

# Thoraxlage relativ zum Herzzentrum (mm)
MIDLINE_X = -38.0  # Herzzentrum liegt links der Mittellinie
CENTER_Y = 28.0  # Mittelpunkt des Querschnitts liegt posterior des Herzzentrums
Z_MIN, Z_MAX = -340.0, 235.0
# Körpermaße: Skalierung der Silhouette (lateral, anterior-posterior, Höhe) aus Größe/Gewicht (Referenz 175 cm, BMI 24)
TORSO_SCALE = {"lateral": 1.0, "ap": 1.0, "height": 1.0, "note": "Referenzthorax (175 cm, BMI 24)"}


def configure_torso(height_cm: Optional[float] = None, weight_kg: Optional[float] = None, sex: Optional[str] = None) -> str:
    """Silhouette an Körpermaße anpassen: Breite ~ BMI^0,5, Tiefe ~ BMI^0,7, Höhe ~ Körpergröße; Frauen etwas schmaler."""
    lateral = ap = height = 1.0
    parts = []
    if height_cm:
        height = float(np.clip(height_cm / 175.0, 0.8, 1.2))
        parts.append(f"{height_cm:.0f} cm")
    if height_cm and weight_kg:
        bmi = float(weight_kg) / (float(height_cm) / 100.0) ** 2
        lateral = float(np.clip((bmi / 24.0) ** 0.5, 0.85, 1.35))
        ap = float(np.clip((bmi / 24.0) ** 0.7, 0.8, 1.5))
        parts.append(f"BMI {bmi:.1f}")
    if sex and str(sex).lower().startswith(("f", "w")):
        lateral *= 0.95
        parts.append("weiblich")
    TORSO_SCALE.update(lateral=lateral, ap=ap, height=height, note=", ".join(parts) or "Referenzthorax (175 cm, BMI 24)")
    return TORSO_SCALE["note"]


def _profile(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halbbreite a (lateral), Halbtiefe b (AP), Mittelpunkt-y je Höhe."""
    z = z / TORSO_SCALE["height"]
    a = np.full_like(z, 172.0 * TORSO_SCALE["lateral"])
    b = np.full_like(z, 108.0 * TORSO_SCALE["ap"])
    # Schultern/Hals
    top = np.clip((z - 150.0) / (Z_MAX - 150.0), 0.0, 1.0)
    a = a * (1.0 - 0.42 * top ** 1.6)
    b = b * (1.0 - 0.25 * top ** 1.6)
    # Taille
    waist = np.exp(-0.5 * ((z + 170.0) / 70.0) ** 2)
    a = a * (1.0 - 0.10 * waist)
    b = b * (1.0 - 0.06 * waist)
    # Becken
    hip = np.clip((-220.0 - z) / 120.0, 0.0, 1.0)
    a = a * (1.0 + 0.06 * hip)
    return a, b, np.full_like(z, CENTER_Y)


def _superellipse(phi: np.ndarray, a: float, b: float, n: float = 2.6) -> tuple[np.ndarray, np.ndarray]:
    """(dx, dy) vom Querschnittsmittelpunkt; φ = 0 anterior (−y), φ = +90° links (+x)."""
    s, c = np.sin(phi), np.cos(phi)
    x = a * np.sign(s) * np.abs(s) ** (2.0 / n)
    y = -b * np.sign(c) * np.abs(c) ** (2.0 / n)
    return x, y


def torso_mesh(n_rings: int = 44, n_around: int = 56) -> tuple[np.ndarray, np.ndarray]:
    """(Vertices (N,3), Faces (M,3)) der Thoraxhülle; φ = 0 anterior, positiv nach links."""
    z = np.linspace(Z_MIN * TORSO_SCALE["height"], Z_MAX * TORSO_SCALE["height"], n_rings)
    a, b, cy = _profile(z)
    phi = np.linspace(-np.pi, np.pi, n_around, endpoint=False)
    verts = []
    for k in range(n_rings):
        x, y = _superellipse(phi, a[k], b[k])
        verts.append(np.column_stack([MIDLINE_X + x, cy[k] + y, np.full(n_around, z[k])]))
    v = np.vstack(verts).astype(np.float32)
    faces = []
    for k in range(n_rings - 1):
        for j in range(n_around):
            j2 = (j + 1) % n_around
            p0, p1, p2, p3 = k * n_around + j, k * n_around + j2, (k + 1) * n_around + j, (k + 1) * n_around + j2
            faces.append((p0, p1, p2))
            faces.append((p1, p3, p2))
    # Deckel oben/unten (Fächer)
    top_c = len(v)
    bot_c = len(v) + 1
    v = np.vstack([v, [[MIDLINE_X, CENTER_Y, z[-1]], [MIDLINE_X, CENTER_Y, z[0]]]]).astype(np.float32)
    for j in range(n_around):
        j2 = (j + 1) % n_around
        faces.append(((n_rings - 1) * n_around + j, (n_rings - 1) * n_around + j2, top_c))
        faces.append((j2, j, bot_c))
    return v, np.asarray(faces, dtype=np.int32)


def surface_point(phi_deg: float, z: float, offset_mm: float = 0.0) -> np.ndarray:
    """Punkt auf der Haut (φ = 0 anterior, + links, ±180 posterior) mit Abstand nach außen."""
    z = float(z) * TORSO_SCALE["height"]
    zz = np.array([float(np.clip(z, Z_MIN * TORSO_SCALE["height"], Z_MAX * TORSO_SCALE["height"]))])
    a, b, cy = _profile(zz)
    a, b = float(a[0]), float(b[0])
    phi = np.radians(phi_deg)
    # Polarwinkel vom Querschnittsmittelpunkt: Strahl (sin φ, −cos φ) bis zur Superellipse |x/a|^n + |y/b|^n = 1
    dx, dy = np.sin(phi), -np.cos(phi)
    n = 2.6
    t = 1.0 / ((abs(dx) / a) ** n + (abs(dy) / b) ** n) ** (1.0 / n)
    x, y = dx * t, dy * t
    p = np.array([MIDLINE_X + x, float(cy[0]) + y, float(zz[0])])
    if offset_mm:
        outward = np.array([x / a ** 2, y / b ** 2, 0.0])  # Gradient der Ellipse als Näherung der Normalen
        norm = np.linalg.norm(outward)
        if norm > 1e-6:
            p = p + outward / norm * offset_mm
    return p


@dataclass(frozen=True)
class Electrode:
    name: str
    phi_deg: float
    z: float
    leads: tuple[str, ...]  # Ableitungen, deren Kurve neben dieser Elektrode schwebt

    @property
    def position(self) -> np.ndarray:
        return surface_point(self.phi_deg, self.z)


# Standardpositionen (Winkel um die Thoraxachse, Höhe relativ zum Herzzentrum)
ELECTRODES: tuple[Electrode, ...] = (
    Electrode("V1", -11.0, 22.0, ("V1",)),
    Electrode("V2", 11.0, 22.0, ("V2",)),
    Electrode("V3", 25.0, 6.0, ("V3",)),
    Electrode("V4", 41.0, -10.0, ("V4",)),
    Electrode("V5", 60.0, -10.0, ("V5",)),
    Electrode("V6", 84.0, -10.0, ("V6",)),
    # Extremitätenelektroden am Rumpf nach Mason-Likar: infraklavikulär bzw. an den Darmbeinkämmen
    Electrode("RA", -62.0, 150.0, ("aVR",)),
    Electrode("LA", 62.0, 150.0, ("aVL",)),
    Electrode("LL", 32.0, -270.0, ("aVF",)),
    Electrode("RL", -32.0, -270.0, ()),
)
# Bipolare Ableitungen liegen zwischen den Elektroden (Einthoven-Dreieck): Kurvenort = Seitenmitte, leicht nach außen versetzt
BIPOLAR_TRACE_SITES = {"I": ("RA", "LA"), "II": ("RA", "LL"), "III": ("LA", "LL")}
ELECTRODE_BY_NAME = {e.name: e for e in ELECTRODES}
# Voreinstellungen für die virtuelle Elektrode (nicht Teil der Standard-12-Ableitung)
VIRTUAL_PRESETS = {
    "V7 (hintere Axillarlinie)": (108.0, -10.0),
    "V8 (Skapulaspitze)": (135.0, -10.0),
    "V9 (paravertebral links)": (162.0, -10.0),
    "V3R": (-25.0, 6.0),
    "V4R": (-41.0, -10.0),
    "V5R": (-60.0, -10.0),
    "Lewis-Position (2. ICS rechts)": (-11.0, 70.0),
    "Epigastrisch": (5.0, -110.0),
}


# ------------------------------------------------------------------ Dipol
def dipole_lead_field(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """(P, 3): Potentialbeitrag je Dipolkomponente an jedem Punkt (ohne Maßstab)."""
    rel = np.atleast_2d(points).astype(np.float64) - centre
    r = np.linalg.norm(rel, axis=1) + 1e-6
    return rel / (r ** 3)[:, None]


class DipoleSimulator:
    """Simuliert Ableitungen an beliebigen Orten aus dem Kors-Dipol des ECG."""

    def __init__(self, centre: np.ndarray):
        self.centre = np.asarray(centre, dtype=np.float64)
        self._base_centre = self.centre.copy()
        self.scale = 1.0
        self.fit_r2: dict[str, float] = {}
        self._xyz: Optional[np.ndarray] = None  # (n, 3) Modellkoordinaten
        self._fs = 500.0
        self._wct = np.zeros(3)
        wct_points = np.vstack([ELECTRODE_BY_NAME[n].position for n in ("RA", "LA", "LL")])
        self._wct = dipole_lead_field(wct_points, self.centre).mean(0)

    def set_timeline(self, tl: BeatTimeline) -> None:
        self._xyz = None
        self.fit_r2 = {}
        if tl.signal is None:
            return
        names = list(tl.lead_names)
        try:
            cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
        except ValueError:
            return
        xyz = (KORS @ tl.signal[:, cols].T).T
        self._xyz = np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])  # vcg_to_model vektorisiert
        self._fs = tl.fs
        # Elektrisches Zentrum und Maßstab an V1–V6 anpassen (Rastersuche ±30 mm, kleinste Quadrate)
        meas = {n: tl.signal[:, names.index(n)] - np.nanmedian(tl.signal[:, names.index(n)])
                for n in ("V1", "V2", "V3", "V4", "V5", "V6") if n in names}
        if not meas:
            return
        best = None
        base_centre = self._base_centre
        wct_points = np.vstack([ELECTRODE_BY_NAME[n].position for n in ("RA", "LA", "LL")])
        for dx in (-30.0, -15.0, 0.0, 15.0, 30.0):
            for dy in (-30.0, -15.0, 0.0, 15.0, 30.0):
                for dz in (-30.0, 0.0, 30.0):
                    centre = base_centre + np.array([dx, dy, dz])
                    wct = dipole_lead_field(wct_points, centre).mean(0)
                    sims = {n: self._xyz @ (dipole_lead_field(ELECTRODE_BY_NAME[n].position[None, :], centre)[0] - wct) for n in meas}
                    num = sum(float(np.dot(sims[n], meas[n])) for n in meas)
                    den = sum(float(np.dot(sims[n], sims[n])) for n in meas) + 1e-12
                    k = num / den
                    err = sum(float(np.sum((meas[n] - k * sims[n]) ** 2)) for n in meas)
                    if best is None or err < best[0]:
                        best = (err, centre, wct, k)
        _, self.centre, self._wct, self.scale = best
        for n in meas:
            sim = self._unipolar_raw(ELECTRODE_BY_NAME[n].position) * self.scale
            ss_res = float(np.sum((meas[n] - sim) ** 2))
            ss_tot = float(np.sum((meas[n] - meas[n].mean()) ** 2)) + 1e-9
            self.fit_r2[n] = max(0.0, 1.0 - ss_res / ss_tot)

    @property
    def available(self) -> bool:
        return self._xyz is not None

    def _unipolar_raw(self, point: np.ndarray) -> np.ndarray:
        lf = dipole_lead_field(point[None, :], self.centre)[0] - self._wct  # Wilson-Referenz
        return self._xyz @ lf

    def unipolar(self, point: np.ndarray) -> np.ndarray:
        """Simulierte unipolare Ableitung (gegen Wilson-Zentralterminal) über die ganze Aufzeichnung, mV."""
        if self._xyz is None:
            return np.zeros(0)
        return self._unipolar_raw(np.asarray(point, dtype=np.float64)) * self.scale

    def surface_potentials(self, points: np.ndarray, k: int) -> np.ndarray:
        """Potentiale an vielen Punkten für den Zeitindex k (Körperoberflächenkarte)."""
        if self._xyz is None or k < 0 or k >= len(self._xyz):
            return np.zeros(len(points))
        lf = dipole_lead_field(points, self.centre) - self._wct
        return (lf @ self._xyz[k]) * self.scale

    @property
    def fit_summary(self) -> str:
        if not self.fit_r2:
            return "keine Kalibrierung"
        return " ".join(f"{n} {r2:.2f}" for n, r2 in self.fit_r2.items())
