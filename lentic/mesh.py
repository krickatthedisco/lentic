"""Ridge solids for a lenticular plate.

Polygons live in the XZ plane (X to the right, Z up) and are extruded along Y.
A counter-clockwise polygon produces outward faces.
"""

from __future__ import annotations

import numpy as np

# (x0, z0, x1, z1) in fractions of one ridge. Z is 0 at the foot and 1 at the peak.
Facet = tuple[float, float, float, float]


def facet_profiles(n_images: int) -> list[Facet]:
    """Slopes for 2 images, or slopes plus a flat top for 3."""
    if n_images == 2:
        return [
            (0.0, 0.0, 0.5, 1.0),
            (0.5, 1.0, 1.0, 0.0),
        ]
    if n_images == 3:
        return [
            (0.0, 0.0, 0.25, 1.0),
            (0.25, 1.0, 0.75, 1.0),
            (0.75, 1.0, 1.0, 0.0),
        ]
    raise ValueError("use 2 images (left and right) or 3 (left, front, and right)")


def triangle_normals(triangles: np.ndarray) -> np.ndarray:
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-12)


def mesh_volume(triangles: np.ndarray) -> float:
    if len(triangles) == 0:
        return 0.0
    return float(np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6.0)


def extrude_xz(poly: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """Extrude an XZ polygon into a closed triangle mesh."""
    points = _dedup(np.asarray(poly, dtype=np.float64))
    if len(points) < 3 or abs(y1 - y0) < 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    if _signed_area(points) < 0:
        points = points[::-1].copy()
    if y1 < y0:
        y0, y1 = y1, y0

    tris: list[list[tuple[float, float, float]]] = []
    for index in range(1, len(points) - 1):
        a, b, c = points[0], points[index], points[index + 1]
        # A counter-clockwise XZ loop points toward -Y, so the high cap is reversed.
        tris.append(_at_y(a, c, b, y1))
        tris.append(_at_y(a, b, c, y0))
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        p0 = (float(start[0]), y0, float(start[1]))
        p1 = (float(start[0]), y1, float(start[1]))
        q1 = (float(end[0]), y1, float(end[1]))
        q0 = (float(end[0]), y0, float(end[1]))
        tris.append([p0, p1, q1])
        tris.append([p0, q1, q0])
    return _drop_degenerate(np.asarray(tris, dtype=np.float64))


def _dedup(points: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2)
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > tol:
            kept.append(point)
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= tol:
        kept.pop()
    return np.asarray(kept, dtype=np.float64)


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    z = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def _at_y(a: np.ndarray, b: np.ndarray, c: np.ndarray, y: float) -> list[tuple[float, float, float]]:
    return [
        (float(a[0]), y, float(a[1])),
        (float(b[0]), y, float(b[1])),
        (float(c[0]), y, float(c[1])),
    ]


def _drop_degenerate(triangles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(triangles) == 0:
        return triangles
    spans = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles[np.linalg.norm(spans, axis=1) > eps]
