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


def extrude_trapezoids(
    x0: np.ndarray,
    z0: np.ndarray,
    x1: np.ndarray,
    z1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    foot: float,
    x0_foot: np.ndarray | None = None,
    x1_foot: np.ndarray | None = None,
) -> np.ndarray:
    """Extrude many copies of the slope profile used by one ridge.

    Each item is the quad (x0, z0), (x1, z1), (x1_foot, foot), (x0_foot, foot),
    running from y0 to y1. The foot corners default to x0 and x1, which matches
    extrude_xz on that same quad. A wider foot lets neighboring filaments tuck
    together without moving the slope or the peak.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1)
    z1 = np.asarray(z1, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    y1 = np.asarray(y1, dtype=np.float64).reshape(-1)
    x0_foot = x0 if x0_foot is None else np.asarray(x0_foot, dtype=np.float64).reshape(-1)
    x1_foot = x1 if x1_foot is None else np.asarray(x1_foot, dtype=np.float64).reshape(-1)
    count = int(x0.shape[0])
    if count == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    foot_z = np.broadcast_to(np.asarray(foot, dtype=np.float64), (count,))
    low = np.minimum(y0, y1)
    high = np.maximum(y0, y1)
    area = (
        x0 * z1
        - z0 * x1
        + x1 * foot_z
        - z1 * x1_foot
        + x1_foot * foot_z
        - foot_z * x0_foot
        + x0_foot * z0
        - foot_z * x0
    )
    positive = area >= 0.0
    ax, az = x0, z0
    bx = np.where(positive, x1, x0_foot)
    bz = np.where(positive, z1, foot_z)
    cx = x1_foot
    cz = foot_z
    dx = np.where(positive, x0_foot, x1)
    dz = np.where(positive, foot_z, z1)
    tris = np.empty((count, 12, 3, 3), dtype=np.float64)
    _put(tris, 0, (ax, high, az), (cx, high, cz), (bx, high, bz))
    _put(tris, 1, (ax, low, az), (bx, low, bz), (cx, low, cz))
    _put(tris, 2, (ax, high, az), (dx, high, dz), (cx, high, cz))
    _put(tris, 3, (ax, low, az), (cx, low, cz), (dx, low, dz))
    _put_wall(tris, 4, ax, az, bx, bz, low, high)
    _put_wall(tris, 6, bx, bz, cx, cz, low, high)
    _put_wall(tris, 8, cx, cz, dx, dz, low, high)
    _put_wall(tris, 10, dx, dz, ax, az, low, high)
    return _drop_degenerate(tris.reshape(-1, 3, 3))


def extrude_ring(
    xs: np.ndarray,
    zs: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Extrude one XZ ring per row. ``xs`` and ``zs`` are shaped (corners, count)."""
    xs = np.asarray(xs, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    y1 = np.asarray(y1, dtype=np.float64).reshape(-1)
    corners, count = xs.shape
    if count == 0 or corners < 3:
        return np.zeros((0, 3, 3), dtype=np.float64)
    nxt_x = np.roll(xs, -1, axis=0)
    nxt_z = np.roll(zs, -1, axis=0)
    area = np.sum(xs * nxt_z - nxt_x * zs, axis=0)
    reverse = area < 0
    if np.any(reverse):
        xs = xs.copy()
        zs = zs.copy()
        xs[:, reverse] = xs[::-1, reverse]
        zs[:, reverse] = zs[::-1, reverse]
    low = np.minimum(y0, y1)
    high = np.maximum(y0, y1)
    cap_count = corners - 2
    tris = np.empty((count, cap_count * 2 + corners * 2, 3, 3), dtype=np.float64)
    slot = 0
    for index in range(1, corners - 1):
        _put(tris, slot, (xs[0], low, zs[0]), (xs[index], low, zs[index]), (xs[index + 1], low, zs[index + 1]))
        _put(tris, slot + 1, (xs[0], high, zs[0]), (xs[index + 1], high, zs[index + 1]), (xs[index], high, zs[index]))
        slot += 2
    for index in range(corners):
        nxt = (index + 1) % corners
        _put_wall(tris, slot, xs[index], zs[index], xs[nxt], zs[nxt], low, high)
        slot += 2
    return _drop_degenerate(tris.reshape(-1, 3, 3))


def _put(tris: np.ndarray, index: int, a, b, c) -> None:
    for corner, (xs, ys, zs) in enumerate((a, b, c)):
        tris[:, index, corner, 0] = xs
        tris[:, index, corner, 1] = ys
        tris[:, index, corner, 2] = zs


def _put_wall(tris: np.ndarray, index: int, sx, sz, ex, ez, low, high) -> None:
    _put(tris, index, (sx, low, sz), (sx, high, sz), (ex, high, ez))
    _put(tris, index + 1, (sx, low, sz), (ex, high, ez), (ex, low, ez))


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


def extrude_xy(poly: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Extrude an XY polygon into a closed mesh. Counter-clockwise points up."""
    points = _dedup(np.asarray(poly, dtype=np.float64))
    if len(points) < 3 or abs(z1 - z0) < 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    if _signed_area(points) < 0:
        points = points[::-1].copy()
    if z1 < z0:
        z0, z1 = z1, z0
    tris: list[list[tuple[float, float, float]]] = []
    for index in range(1, len(points) - 1):
        a, b, c = points[0], points[index], points[index + 1]
        tris.append(_at_z(a, b, c, z1))
        tris.append(_at_z(a, c, b, z0))
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        p0 = (float(start[0]), float(start[1]), z0)
        p1 = (float(end[0]), float(end[1]), z0)
        q1 = (float(end[0]), float(end[1]), z1)
        q0 = (float(start[0]), float(start[1]), z1)
        tris.append([p0, q1, q0])
        tris.append([p0, p1, q1])
    return _drop_degenerate(np.asarray(tris, dtype=np.float64))


def _at_z(a: np.ndarray, b: np.ndarray, c: np.ndarray, z: float) -> list[tuple[float, float, float]]:
    return [
        (float(a[0]), float(a[1]), z),
        (float(b[0]), float(b[1]), z),
        (float(c[0]), float(c[1]), z),
    ]


def wall_xy(poly: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Side wall of an XY loop. A counter-clockwise loop faces away from its interior."""
    points = _dedup(np.asarray(poly, dtype=np.float64))
    if len(points) < 2 or abs(z1 - z0) < 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    if z1 < z0:
        z0, z1 = z1, z0
    tris: list[list[tuple[float, float, float]]] = []
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        p0 = (float(start[0]), float(start[1]), z0)
        p1 = (float(end[0]), float(end[1]), z0)
        q1 = (float(end[0]), float(end[1]), z1)
        q0 = (float(start[0]), float(start[1]), z1)
        tris.append([p0, q1, q0])
        tris.append([p0, p1, q1])
    return _drop_degenerate(np.asarray(tris, dtype=np.float64))


def border_mesh(outer: np.ndarray, inner: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """A frame: the outer loop minus the inner loop, extruded from z0 to z1.

    Both loops are counter-clockwise. The inner loop is a hole.
    """
    if z1 < z0:
        z0, z1 = z1, z0
    if z1 - z0 < 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    floor = _triangulate_with_holes(np.asarray(outer, dtype=np.float64), [np.asarray(inner, dtype=np.float64)])
    top = floor.copy()
    top[:, :, 2] = z1
    bottom = floor[:, ::-1, :].copy()
    bottom[:, :, 2] = z0
    outside = wall_xy(outer, z0, z1)
    inside = wall_xy(np.asarray(inner, dtype=np.float64)[::-1], z0, z1)
    return np.ascontiguousarray(np.concatenate([bottom, top, outside, inside]))


def box_mesh(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> np.ndarray:
    """A closed box. Normals point outward and the volume is positive."""
    if x1 - x0 < 1e-9 or y1 - y0 < 1e-9 or z1 - z0 < 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    points = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return points[faces]


def circle_polygon(cx: float, cy: float, radius: float, segments: int = 32) -> np.ndarray:
    """A regular polygon whose flats sit on the requested radius, counter-clockwise."""
    if radius <= 0 or segments < 3:
        return np.zeros((0, 2), dtype=np.float64)
    vertex_radius = radius / np.cos(np.pi / segments)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return np.column_stack((cx + vertex_radius * np.cos(angles), cy + vertex_radius * np.sin(angles)))


def cylinder_cavity(
    cx: float,
    cy: float,
    radius: float,
    z0: float,
    z1: float,
    segments: int = 32,
) -> np.ndarray:
    """A round hole. Normals point into the hole, so its volume is negative.

    The flats of the polygon sit on the requested radius, so a round magnet
    of that size can drop in.
    """
    polygon = circle_polygon(cx, cy, radius, segments)
    if len(polygon) < 3 or z1 - z0 <= 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    triangles = np.empty((segments * 4, 3, 3), dtype=np.float64)
    for index in range(segments):
        nxt = (index + 1) % segments
        bottom_0 = (xs[index], ys[index], z0)
        bottom_1 = (xs[nxt], ys[nxt], z0)
        top_0 = (xs[index], ys[index], z1)
        top_1 = (xs[nxt], ys[nxt], z1)
        slot = index * 4
        triangles[slot] = ((cx, cy, z0), bottom_0, bottom_1)
        triangles[slot + 1] = ((cx, cy, z1), top_1, top_0)
        triangles[slot + 2] = (bottom_0, top_1, bottom_1)
        triangles[slot + 3] = (bottom_0, top_0, top_1)
    return triangles


def cylinder_tunnel(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    segments: int = 32,
) -> np.ndarray:
    """A round hole from ``start`` to ``end``. Normals point into the hole.

    The flat caps sit on the ends. A hole that meets a face at an angle has to
    be extended past that face by more than its radius, or the cap covers
    part of the opening.
    """
    begin = np.asarray(start, dtype=np.float64)
    finish = np.asarray(end, dtype=np.float64)
    axis = finish - begin
    length = float(np.linalg.norm(axis))
    if length < 1e-9 or radius <= 0 or segments < 3:
        return np.zeros((0, 3, 3), dtype=np.float64)
    direction = axis / length
    helper = np.array([0.0, 0.0, 1.0] if abs(direction[2]) < 0.9 else [1.0, 0.0, 0.0])
    across = np.cross(direction, helper)
    across = across / np.linalg.norm(across)
    upward = np.cross(direction, across)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    offsets = [radius * (np.cos(angle) * across + np.sin(angle) * upward) for angle in angles]
    ring0 = [begin + offset for offset in offsets]
    ring1 = [finish + offset for offset in offsets]
    triangles = np.empty((segments * 4, 3, 3), dtype=np.float64)
    for index in range(segments):
        nxt = (index + 1) % segments
        slot = index * 4
        triangles[slot] = (ring0[index], ring0[nxt], ring1[nxt])
        triangles[slot + 1] = (ring0[index], ring1[nxt], ring1[index])
        triangles[slot + 2] = (begin, ring0[nxt], ring0[index])
        triangles[slot + 3] = (finish, ring1[index], ring1[nxt])
    if mesh_volume(triangles) > 0:
        triangles = triangles[:, ::-1, :]
    return triangles


def cylinder_solid(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    segments: int = 32,
) -> np.ndarray:
    """A solid round bar from ``start`` to ``end``. Its volume is positive."""
    hole = cylinder_tunnel(start, end, radius, segments)
    if len(hole) == 0:
        return hole
    return np.ascontiguousarray(hole[:, ::-1, :])


def difference(solid: np.ndarray, cutters: list[np.ndarray]) -> np.ndarray:
    """Remove each solid cutter from a closed mesh."""
    import manifold3d

    result = _as_manifold(solid)
    for cutter in cutters:
        if len(cutter) == 0:
            continue
        result = result - _as_manifold(cutter)
    if result.is_empty() or result.volume() <= 1e-6:
        raise ValueError("the hanger hole cut away the frame")
    return _triangles_from_mesh(result.to_mesh())


def keyhole_solid(
    x: float,
    y_head: float,
    y_rest: float,
    shaft_radius: float,
    head_radius: float,
    lip: float,
    pocket: float,
) -> np.ndarray:
    """A keyhole pocket. The back opening is narrow at the top and wide below.

    Past the lip, the cavity is as wide as the screw head all the way up, so
    the head can slide up and catch.
    """
    import manifold3d

    if pocket - lip <= 1e-6 or shaft_radius <= 0 or head_radius <= shaft_radius:
        return np.zeros((0, 3, 3), dtype=np.float64)
    narrow = _keyhole_section(x, y_head, y_rest, shaft_radius, head_radius, shaft_radius)
    wide = _keyhole_section(x, y_head, y_rest, head_radius, head_radius, head_radius)
    opening = manifold3d.Manifold.extrude(narrow, float(pocket))
    cavity = manifold3d.Manifold.extrude(wide, float(pocket - lip)).translate((0.0, 0.0, float(lip)))
    return _triangles_from_mesh((opening + cavity).to_mesh())


def _keyhole_section(x: float, y_head: float, y_rest: float, slot: float, head_radius: float, top_radius: float):
    import manifold3d

    def circle(cx: float, cy: float, radius: float):
        return manifold3d.CrossSection.circle(radius, 32).translate((cx, cy))

    rect = manifold3d.CrossSection(
        [[(x - slot, y_head), (x + slot, y_head), (x + slot, y_rest), (x - slot, y_rest)]]
    )
    return circle(x, y_head, head_radius) + rect + circle(x, y_rest, top_radius)


def _as_manifold(triangles: np.ndarray):
    import manifold3d

    flat = np.ascontiguousarray(triangles, dtype=np.float64).reshape(-1, 3)
    rounded = np.round(flat, 5)
    vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.uint32)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    mesh = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces[keep]),
    )
    return manifold3d.Manifold(mesh)


def _triangles_from_mesh(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]
    return np.ascontiguousarray(vertices[faces])


def open_pocket_base(
    width: float,
    height: float,
    top_z: float,
    pocket_z: float,
    holes: list[np.ndarray],
) -> np.ndarray:
    """A plate with pockets that are open on the bed and closed at pocket_z.

    Each hole is a counter-clockwise polygon in XY. The magnet can be glued
    in from the back after the print, so there is no floor under it.
    """
    if top_z - pocket_z <= 1e-9 or pocket_z <= 1e-9:
        return np.zeros((0, 3, 3), dtype=np.float64)
    rings = [_inside_ring(np.asarray(hole, dtype=np.float64), width, height) for hole in holes]
    rings = [ring for ring in rings if len(ring) >= 3]
    outer = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)
    floor = _triangulate_with_holes(outer, rings)
    floor[:, :, 2] = 0.0
    floor = floor[:, ::-1, :]
    cap = box_mesh(0.0, 0.0, pocket_z, width, height, top_z)
    cap = _drop_downward_face(cap, pocket_z)
    walls = _vertical_sides(box_mesh(0.0, 0.0, 0.0, width, height, pocket_z))
    pieces = [floor, cap, walls]
    for ring in rings:
        pieces.append(_hole_walls(ring, 0.0, pocket_z))
        pieces.append(_hole_ceiling(ring, pocket_z))
    return _drop_degenerate(np.ascontiguousarray(np.concatenate(pieces)))


def _inside_ring(hole: np.ndarray, width: float, height: float) -> np.ndarray:
    """Pull a hole just inside the plate so its opening stays on the bottom face."""
    margin = 1e-4
    ring = np.array(hole, dtype=np.float64, copy=True)
    ring[:, 0] = np.clip(ring[:, 0], margin, width - margin)
    ring[:, 1] = np.clip(ring[:, 1], margin, height - margin)
    keep = np.ones(len(ring), dtype=bool)
    nxt = np.roll(ring, -1, axis=0)
    keep &= np.linalg.norm(nxt - ring, axis=1) > 1e-8
    return ring[keep]


def _drop_downward_face(triangles: np.ndarray, z: float) -> np.ndarray:
    if len(triangles) == 0:
        return triangles
    flat = np.all(np.abs(triangles[:, :, 2] - z) < 1e-8, axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles[~(flat & (normals[:, 2] < 0))]


def _vertical_sides(triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return triangles
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles[np.abs(normals[:, 2]) <= 1e-8]


def _hole_walls(ring: np.ndarray, z0: float, z1: float) -> np.ndarray:
    count = len(ring)
    triangles = np.empty((count * 2, 3, 3), dtype=np.float64)
    for index in range(count):
        nxt = (index + 1) % count
        bottom_0 = (ring[index, 0], ring[index, 1], z0)
        bottom_1 = (ring[nxt, 0], ring[nxt, 1], z0)
        top_0 = (ring[index, 0], ring[index, 1], z1)
        top_1 = (ring[nxt, 0], ring[nxt, 1], z1)
        slot = index * 2
        triangles[slot] = (bottom_0, top_0, top_1)
        triangles[slot + 1] = (bottom_0, top_1, bottom_1)
    return triangles


def _hole_ceiling(ring: np.ndarray, z: float) -> np.ndarray:
    center = ring.mean(axis=0)
    count = len(ring)
    triangles = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        nxt = (index + 1) % count
        triangles[index] = (
            (center[0], center[1], z),
            (ring[nxt, 0], ring[nxt, 1], z),
            (ring[index, 0], ring[index, 1], z),
        )
    return triangles


class _Ring:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.prev = self
        self.next = self


def _link(points: np.ndarray, clockwise: bool) -> _Ring:
    area = float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))
    ordered = points[::-1] if (area > 0) == clockwise else points
    nodes = [_Ring(float(x), float(y)) for x, y in ordered]
    for index, node in enumerate(nodes):
        node.prev = nodes[index - 1]
        node.next = nodes[(index + 1) % len(nodes)]
    return nodes[0]


def _turn(a: _Ring, b: _Ring, c: _Ring) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _turn_xy(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _inside_triangle(px: float, py: float, a: _Ring, b: _Ring, c: _Ring) -> bool:
    first = _turn_xy(a.x, a.y, b.x, b.y, px, py)
    second = _turn_xy(b.x, b.y, c.x, c.y, px, py)
    third = _turn_xy(c.x, c.y, a.x, a.y, px, py)
    return (first > 1e-10 and second > 1e-10 and third > 1e-10) or (
        first < -1e-10 and second < -1e-10 and third < -1e-10
    )


def _locally_inside(node: _Ring, x: float, y: float) -> bool:
    left_of_in = _turn_xy(node.prev.x, node.prev.y, node.x, node.y, x, y) >= -1e-10
    left_of_out = _turn_xy(node.x, node.y, node.next.x, node.next.y, x, y) >= -1e-10
    if _turn(node.prev, node, node.next) > 0:
        return left_of_in and left_of_out
    return left_of_in or left_of_out


def _segments_cross(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float
) -> bool:
    if max(ax, bx) < min(cx, dx) - 1e-9 or max(cx, dx) < min(ax, bx) - 1e-9:
        return False
    if max(ay, by) < min(cy, dy) - 1e-9 or max(cy, dy) < min(ay, by) - 1e-9:
        return False
    first = _turn_xy(ax, ay, bx, by, cx, cy)
    second = _turn_xy(ax, ay, bx, by, dx, dy)
    third = _turn_xy(cx, cy, dx, dy, ax, ay)
    fourth = _turn_xy(cx, cy, dx, dy, bx, by)
    return first * second < -1e-12 and third * fourth < -1e-12


def _segment_clear(ring: _Ring, ax: float, ay: float, bx: float, by: float) -> bool:
    node = ring
    while True:
        if _segments_cross(ax, ay, bx, by, node.x, node.y, node.next.x, node.next.y):
            return False
        node = node.next
        if node is ring:
            return True


def _find_bridge(outer: _Ring, hx: float, hy: float) -> _Ring | None:
    """A vertex of the plate that can see the leftmost point of a hole."""
    node = outer
    closest = float("-inf")
    chosen: _Ring | None = None
    while True:
        y0, y1 = node.y, node.next.y
        crosses = (hy <= y0 and hy >= y1) or (hy >= y0 and hy <= y1)
        if crosses and abs(y1 - y0) > 1e-15:
            x = node.x + (hy - y0) * (node.next.x - node.x) / (y1 - y0)
            if x <= hx + 1e-9 and x > closest:
                closest = x
                chosen = node if node.x < node.next.x else node.next
                if abs(x - hx) <= 1e-8:
                    return chosen
        node = node.next
        if node is outer:
            break
    if chosen is None:
        return None
    stop = chosen
    best = chosen
    best_tan = float("inf")
    node = chosen
    while True:
        on_span = hx >= node.x >= chosen.x and abs(hx - node.x) > 1e-9
        in_sector = _inside_triangle(node.x, node.y, _Ring(hx, hy), chosen, _Ring(closest, hy))
        if on_span and in_sector:
            tan = abs(hy - node.y) / (hx - node.x)
            if tan < best_tan and _locally_inside(node, hx, hy) and _segment_clear(outer, node.x, node.y, hx, hy):
                best = node
                best_tan = tan
        node = node.next
        if node is stop:
            break
    if not _segment_clear(outer, best.x, best.y, hx, hy):
        return None
    return best


def _splice(outer: _Ring, hole: _Ring) -> _Ring:
    """Join a clockwise hole into a counter-clockwise outline along one bridge."""
    copy_outer = _Ring(outer.x, outer.y)
    copy_hole = _Ring(hole.x, hole.y)
    outer_next = outer.next
    hole_prev = hole.prev
    outer.next = hole
    hole.prev = outer
    copy_outer.next = outer_next
    outer_next.prev = copy_outer
    copy_hole.next = copy_outer
    copy_outer.prev = copy_hole
    hole_prev.next = copy_hole
    copy_hole.prev = hole_prev
    return outer


def _triangulate_with_holes(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """Counter-clockwise triangles of the plate bottom, holes excluded."""
    ring = _link(outer, clockwise=False)
    ordered = sorted(holes, key=lambda hole: float(np.max(hole[:, 0])), reverse=True)
    for hole in ordered:
        leftmost = min(range(len(hole)), key=lambda index: (float(hole[index, 0]), float(hole[index, 1])))
        hole_ring = _link(hole, clockwise=True)
        bridge_on_hole = hole_ring
        for _ in range(leftmost):
            bridge_on_hole = bridge_on_hole.next
        # _link may have reversed the hole, so find the point again.
        start = bridge_on_hole
        bridge_on_hole = min(
            _walk(hole_ring),
            key=lambda node: (node.x, node.y),
        )
        del start
        bridge = _find_bridge(ring, bridge_on_hole.x, bridge_on_hole.y)
        if bridge is None:
            raise ValueError("those magnet holes do not sit inside the plate")
        ring = _splice(bridge, bridge_on_hole)
    return _ear_clip(ring)


def _walk(start: _Ring) -> list[_Ring]:
    nodes = []
    node = start
    while True:
        nodes.append(node)
        node = node.next
        if node is start:
            return nodes


def _ear_clip(start: _Ring) -> np.ndarray:
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    node = start
    guard = 0
    while node.next is not node.prev:
        guard += 1
        if guard > 200000:
            raise ValueError("could not open the magnet pockets")
        if _is_ear(node):
            prev_node, next_node = node.prev, node.next
            triangles.append(
                (
                    (prev_node.x, prev_node.y, 0.0),
                    (node.x, node.y, 0.0),
                    (next_node.x, next_node.y, 0.0),
                )
            )
            prev_node.next = next_node
            next_node.prev = prev_node
            node = next_node
            start = next_node
            continue
        node = node.next
        if node is start:
            raise ValueError("could not open the magnet pockets")
    return np.array(triangles, dtype=np.float64)


def _is_ear(node: _Ring) -> bool:
    prev_node, next_node = node.prev, node.next
    if _turn(prev_node, node, next_node) <= 1e-10:
        return False
    other = next_node.next
    while other is not prev_node:
        if _inside_triangle(other.x, other.y, prev_node, node, next_node):
            return False
        other = other.next
    return True


def _drop_degenerate(triangles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(triangles) == 0:
        return triangles
    spans = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles[np.linalg.norm(spans, axis=1) > eps]
