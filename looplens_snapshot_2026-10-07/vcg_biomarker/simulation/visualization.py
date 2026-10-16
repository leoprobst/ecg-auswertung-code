"""Geometry and state summaries for the research viewer, without Qt or new physics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraphSlice:
    xyz: np.ndarray
    start: np.ndarray
    end: np.ndarray
    weight_end: np.ndarray

    def interpolate(self, values):
        values = np.asarray(values)
        if values.ndim != 1 or (len(self.end) and len(values) <= int(self.end.max())):
            raise ValueError("Slice values do not match graph nodes")
        a, b = values[self.start], values[self.end]
        return a * (1 - self.weight_end) + b * self.weight_end


def graph_slice(xyz, edges, axis, coordinate):
    """Intersections with ACTUAL graph edges; no fictitious filled volume."""
    xyz, edges = np.asarray(xyz, float), np.asarray(edges, int)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges < 0)
            or np.any(edges >= len(xyz)) or axis not in (0, 1, 2)
            or not np.isfinite(coordinate)):
        raise ValueError("Invalid graph or cut plane")
    d0, d1 = xyz[edges[:, 0], axis] - coordinate, xyz[edges[:, 1], axis] - coordinate
    # In-plane edges are represented by their endpoints; duplicates are harmless
    # and the view describes graph intersections, not a continuous tissue volume.
    crosses = ((d0 <= 0) & (d1 >= 0)) | ((d1 <= 0) & (d0 >= 0))
    selected = edges[crosses]
    denominator = d0[crosses] - d1[crosses]
    weights = np.zeros(len(selected))
    np.divide(d0[crosses], denominator, out=weights, where=denominator != 0)
    weights = np.clip(weights, 0, 1)
    position = xyz[selected[:, 0]] * (1 - weights[:, None]) + xyz[selected[:, 1]] * weights[:, None]
    return GraphSlice(position, selected[:, 0], selected[:, 1], weights)


def surface_segments(vertices, faces, axis, coordinate):
    """Exact triangle/plane intersection endpoints for an anatomical outline."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or np.any(faces < 0) or np.any(faces >= len(vertices))
            or axis not in (0, 1, 2) or not np.isfinite(coordinate)):
        raise ValueError("Invalid triangle surface")
    triangles = vertices[faces]
    d = triangles[:, :, axis] - coordinate
    hits = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        crosses = (da < 0) & (db > 0) | (da > 0) & (db < 0)
        fraction = np.zeros(len(d))
        np.divide(da, da-db, out=fraction, where=crosses)
        points = triangles[:, a] + fraction[:, None] * (triangles[:, b] - triangles[:, a])
        hits.append((crosses, points))
    count = sum(mask.astype(int) for mask, _ in hits)
    selected = count == 2
    result = np.empty((int(selected.sum()), 2, 3), float)
    if not selected.any():
        return result
    cursor = np.zeros(int(selected.sum()), int)
    rows = np.arange(len(cursor))
    for mask, points in hits:
        use = mask[selected]
        result[rows[use], cursor[use]] = points[selected][use]
        cursor[use] += 1
    return result


def process_summary(times_ms, voltage, activation_ms, cells):
    times = np.asarray(times_ms, float)
    potential = np.asarray(voltage, float)
    activation = np.asarray(activation_ms, float)
    cell = np.asarray(cells, int)
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or potential.shape != (len(times), len(cell)) or activation.shape != (len(cell),)
            or not np.isfinite(potential).all()):
        raise ValueError("Invalid simulation result for process summary")
    groups = {'ventricle': np.isin(cell, (0, 1, 2)), 'atrium': cell == 3,
              'av_junction': cell == 4, 'conduction_cable': cell == 5}
    out = {}
    for label, mask in groups.items():
        if not mask.any():
            continue
        active = (potential[:, mask] >= .5).mean(axis=1)
        ever_activated = np.isfinite(activation[mask])[None, :] & (times[:, None] >= activation[mask][None, :])
        recovered = (ever_activated & (potential[:, mask] < .13)).mean(axis=1)
        first_activation = np.searchsorted(times, activation[mask][np.isfinite(activation[mask])], side='left')
        first_activation = first_activation[first_activation < len(times)]
        new_activation = np.bincount(first_activation, minlength=len(times)) / mask.sum()
        out[label] = {'n_nodes': int(mask.sum()), 'active_fraction': active,
                      'recovered_fraction': recovered, 'new_activation_fraction': new_activation}
    return out
