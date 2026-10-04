"""Build a color lenticular plate from two or three pictures.

Each ridge is a solid wedge. The first picture is on the slope that faces
left, the second on the slope that faces right, and a third picture on the
flat top. Filament colors are separate meshes so a multi-material slicer
can print them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from lentic.color import as_rgb_array, choose_palette, floyd_steinberg, nearest_indices
from lentic.mesh import (
    box_mesh,
    circle_polygon,
    cylinder_cavity,
    extrude_ring,
    extrude_trapezoids,
    extrude_xz,
    facet_profiles,
    open_pocket_base,
)

# A 0.4 mm nozzle lays a line about this wide. Smaller and larger nozzles
# scale from here, so the crest is cut off one line in from the tip.
_LINE_MM = 0.5
_NOZZLE_MM = 0.4

ANGLE_NAMES = {
    2: ("left", "right"),
    3: ("left", "front", "right"),
}


def angle_names_for(n_images: int, orientation: str = "vertical") -> tuple[str, ...]:
    """Names of the pictures, in the order stored on the model."""
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    if n_images == 2:
        return ("left", "right") if orientation == "vertical" else ("bottom", "top")
    if n_images == 3:
        return ("left", "front", "right") if orientation == "vertical" else ("bottom", "front", "top")
    raise ValueError("use 2 images or 3")


@dataclass(frozen=True)
class Grid:
    width_mm: float
    height_mm: float
    pitch_mm: float
    row_mm: float
    ridge_height_mm: float
    n_ridges: int
    n_rows: int


@dataclass
class MeshPart:
    role: str
    rgb: tuple[int, int, int]
    triangles: np.ndarray


@dataclass(frozen=True)
class MagnetSpec:
    """Pockets in the base, open during the print so a magnet can be dropped in."""

    count: int
    shape: str
    thickness_mm: float
    below_mm: float
    above_mm: float
    diameter_mm: float = 0.0
    width_mm: float = 0.0
    length_mm: float = 0.0
    clearance_mm: float = 0.4
    arrangement: str = "grid"
    turned: bool = False
    edge_mm: float = 2.5

    def required_base(self) -> float:
        return self.below_mm + self.thickness_mm + self.above_mm

    def hole_span(self) -> tuple[float, float]:
        extra = self.clearance_mm
        if self.shape == "round":
            size = self.diameter_mm + extra
            return size, size
        wide = self.width_mm + extra
        tall = self.length_mm + extra
        if self.turned:
            return tall, wide
        return wide, tall


@dataclass
class LenticModel:
    grid: Grid
    base_mm: float
    embed_mm: float
    palette: np.ndarray
    base_rgb: tuple[int, int, int]
    angle_names: tuple[str, ...]
    orientation: str = "vertical"
    parts: list[MeshPart] = field(default_factory=list)
    previews: list[np.ndarray] = field(default_factory=list)
    fitted: list[np.ndarray] = field(default_factory=list)
    top_preview: np.ndarray | None = None
    magnets: MagnetSpec | None = None
    magnet_places: list[tuple[float, float]] = field(default_factory=list)
    crest_width_mm: float = 0.0
    crest_height_mm: float = 0.0
    crest_rgb: tuple[int, int, int] | None = None


def resolve_grid(
    width_mm: float,
    height_mm: float,
    *,
    n_images: int,
    pitch_mm: float | None = None,
    row_mm: float = 0.4,
    ridge_height_mm: float | None = None,
    nozzle_mm: float = 0.4,
    lines_per_facet: int = 1,
    orientation: str = "vertical",
) -> Grid:
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width and height must be positive")
    if n_images not in ANGLE_NAMES:
        raise ValueError("use 2 images or 3")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    if pitch_mm is None:
        if nozzle_mm <= 0 or lines_per_facet < 1:
            raise ValueError("nozzle and lines per facet must be positive")
        pitch_mm = nozzle_mm * lines_per_facet * n_images
    if pitch_mm <= 0 or row_mm <= 0:
        raise ValueError("pitch and row size must be positive")
    if orientation == "horizontal":
        n_ridges = max(1, int(round(height_mm / pitch_mm)))
        n_rows = max(1, int(round(width_mm / row_mm)))
        pitch = height_mm / n_ridges
        row = width_mm / n_rows
    else:
        n_ridges = max(1, int(round(width_mm / pitch_mm)))
        n_rows = max(1, int(round(height_mm / row_mm)))
        pitch = width_mm / n_ridges
        row = height_mm / n_rows
    if ridge_height_mm is None:
        ridge_height_mm = pitch * 0.8
    if ridge_height_mm <= 0:
        raise ValueError("ridge height must be positive")
    return Grid(width_mm, height_mm, pitch, row, ridge_height_mm, n_ridges, n_rows)


def build_from_indices(
    indices: list[np.ndarray],
    palette: np.ndarray,
    *,
    width_mm: float,
    height_mm: float,
    base_mm: float = 0.8,
    ridge_height_mm: float | None = None,
    base_rgb: tuple[int, int, int] = (244, 241, 234),
    embed_mm: float = 0.05,
    seam_mm: float = 0.02,
    line_mm: float = _LINE_MM,
    fitted: list[np.ndarray] | None = None,
    orientation: str = "vertical",
    magnets: MagnetSpec | None = None,
    crest_line: bool = False,
    crest_rgb: tuple[int, int, int] = (0, 0, 0),
    crest_width_mm: float = 0.4,
    crest_height_mm: float = 0.2,
) -> LenticModel:
    if len(indices) not in ANGLE_NAMES:
        raise ValueError("use 2 images or 3")
    if magnets is not None:
        _check_magnets(spec=magnets)
        if magnets.above_mm <= embed_mm:
            raise ValueError("thickness above the magnets has to leave plastic between the pockets and the ridges")
        base_mm = magnets.required_base()
    if orientation == "horizontal":
        model = _build_horizontal(
            indices,
            palette,
            width_mm=width_mm,
            height_mm=height_mm,
            base_mm=base_mm,
            ridge_height_mm=ridge_height_mm,
            base_rgb=base_rgb,
            embed_mm=embed_mm,
            seam_mm=seam_mm,
            line_mm=line_mm,
            fitted=fitted,
            crest_line=crest_line,
            crest_rgb=crest_rgb,
            crest_width_mm=crest_width_mm,
            crest_height_mm=crest_height_mm,
        )
        if magnets is not None:
            _install_magnets(model, magnets)
        return model
    palette = as_rgb_array(palette)
    shapes = {idx.shape for idx in indices}
    if len(shapes) != 1:
        raise ValueError("every view must use the same pixel grid")
    n_rows, n_ridges = indices[0].shape
    if n_rows < 1 or n_ridges < 1:
        raise ValueError("pixel grid is empty")
    for view in indices:
        if view.min() < 0 or view.max() >= len(palette):
            raise ValueError("a pixel points outside the palette")
    if base_mm <= 0:
        raise ValueError("base thickness must be positive")
    if embed_mm < 0 or embed_mm >= base_mm:
        raise ValueError("embed must be smaller than the base thickness")
    grid = resolve_grid(
        width_mm,
        height_mm,
        n_images=len(indices),
        pitch_mm=width_mm / n_ridges,
        row_mm=height_mm / n_rows,
        ridge_height_mm=ridge_height_mm if ridge_height_mm is not None else width_mm / n_ridges * 0.8,
        lines_per_facet=1,
    )
    # resolve_grid recomputes counts from pitch; force the caller's grid.
    grid = Grid(
        width_mm,
        height_mm,
        width_mm / n_ridges,
        height_mm / n_rows,
        grid.ridge_height_mm,
        n_ridges,
        n_rows,
    )
    profiles = facet_profiles(len(indices))
    foot = base_mm - embed_mm
    peak = base_mm + grid.ridge_height_mm
    span = peak - foot
    seam = min(seam_mm, grid.pitch_mm * 0.2)
    blunt = len(indices) == 2 and line_mm > 0
    buckets: dict[int, dict[str, list[np.ndarray]]] = {
        color: {key: [] for key in ("x0", "z0", "x1", "z1", "y0", "y1", "x0_foot", "x1_foot", "ring_x", "ring_z")}
        for color in range(len(palette))
    }

    for ridge in range(n_ridges):
        ridge_x0 = ridge * grid.pitch_mm
        ridge_x1 = (ridge + 1) * grid.pitch_mm
        for angle, (fx0, fz0, fx1, fz1) in enumerate(profiles):
            starts, ends, colors = _color_runs(indices[angle][:, ridge])
            x0 = ridge_x0 + fx0 * grid.pitch_mm
            x1 = ridge_x0 + fx1 * grid.pitch_mm
            # Tuck only the buried foot past a shared edge. Moving the slope
            # itself puts both pictures on the peak, and the tip looks mixed.
            x0_foot = max(ridge_x0, x0 - seam) if fx0 > 0 else x0
            x1_foot = min(ridge_x1, x1 + seam) if fx1 < 1 else x1
            if x1 - x0 < 1e-6:
                continue
            z0 = foot + fz0 * span
            z1 = foot + fz1 * span
            ring = _blunt_crest(x0, z0, x1, z1, x0_foot, x1_foot, foot, line_mm) if blunt else None
            y1 = height_mm - starts * grid.row_mm
            y0 = height_mm - ends * grid.row_mm
            for color in np.unique(colors):
                mask = colors == color
                taken = int(mask.sum())
                slot = buckets[int(color)]
                slot["x0"].append(np.full(taken, x0))
                slot["z0"].append(np.full(taken, z0))
                slot["x1"].append(np.full(taken, x1))
                slot["z1"].append(np.full(taken, z1))
                slot["x0_foot"].append(np.full(taken, x0_foot))
                slot["x1_foot"].append(np.full(taken, x1_foot))
                slot["y0"].append(y0[mask])
                slot["y1"].append(y1[mask])
                if ring is not None:
                    slot["ring_x"].append(np.repeat(ring[0][:, None], taken, axis=1))
                    slot["ring_z"].append(np.repeat(ring[1][:, None], taken, axis=1))

    base_poly = np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, base_mm], [0.0, base_mm]],
        dtype=np.float64,
    )
    parts = [
        MeshPart("base", tuple(int(channel) for channel in base_rgb), extrude_xz(base_poly, 0.0, height_mm))
    ]
    for color, slot in buckets.items():
        if not slot["x0"]:
            continue
        rgb = tuple(int(channel) for channel in palette[color])
        if slot["ring_x"]:
            solid = extrude_ring(
                np.concatenate(slot["ring_x"], axis=1),
                np.concatenate(slot["ring_z"], axis=1),
                np.concatenate(slot["y0"]),
                np.concatenate(slot["y1"]),
            )
        else:
            solid = extrude_trapezoids(
                np.concatenate(slot["x0"]),
                np.concatenate(slot["z0"]),
                np.concatenate(slot["x1"]),
                np.concatenate(slot["z1"]),
                np.concatenate(slot["y0"]),
                np.concatenate(slot["y1"]),
                foot,
                np.concatenate(slot["x0_foot"]),
                np.concatenate(slot["x1_foot"]),
            )
        parts.append(MeshPart("filament", rgb, solid))
    crest_color = None
    if crest_line:
        _check_crest(grid.pitch_mm, crest_width_mm, crest_height_mm)
        crest_color = tuple(int(channel) for channel in crest_rgb)
        _add_filament(
            parts,
            crest_color,
            _crest_boxes(grid, base_mm, embed_mm, len(indices), line_mm, crest_width_mm, crest_height_mm),
        )

    previews = [palette[view] for view in indices]
    top = np.zeros((n_rows, n_ridges * len(indices), 3), dtype=np.uint8)
    for angle, preview in enumerate(previews):
        top[:, angle :: len(indices), :] = preview
    model = LenticModel(
        grid=grid,
        base_mm=base_mm,
        embed_mm=embed_mm,
        palette=palette,
        base_rgb=tuple(int(channel) for channel in base_rgb),
        angle_names=angle_names_for(len(indices), "vertical"),
        orientation="vertical",
        parts=parts,
        previews=previews,
        fitted=list(fitted) if fitted is not None else previews,
        top_preview=top,
        crest_width_mm=crest_width_mm if crest_line else 0.0,
        crest_height_mm=crest_height_mm if crest_line else 0.0,
        crest_rgb=crest_color,
    )
    if magnets is not None:
        _install_magnets(model, magnets)
    return model


def build_from_images(
    paths: list[str | Path],
    *,
    width_mm: float,
    height_mm: float | None = None,
    pitch_mm: float | None = None,
    row_mm: float = 0.4,
    ridge_height_mm: float | None = None,
    base_mm: float = 0.8,
    nozzle_mm: float = 0.4,
    lines_per_facet: int = 1,
    palette: np.ndarray | None = None,
    max_colors: int = 4,
    dither: bool = True,
    base_rgb: tuple[int, int, int] = (244, 241, 234),
    embed_mm: float = 0.05,
    seam_mm: float = 0.02,
    orientation: str = "vertical",
    crops: list[tuple[float, float, float, float] | None] | None = None,
    flips: list[tuple[bool, bool]] | None = None,
    magnets: MagnetSpec | None = None,
    crest_line: bool = False,
    crest_rgb: tuple[int, int, int] = (0, 0, 0),
    crest_width_mm: float = 0.4,
    crest_height_mm: float = 0.2,
) -> LenticModel:
    if len(paths) not in ANGLE_NAMES:
        raise ValueError("use 2 images or 3")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    first = _open_rgb(paths[0])
    if height_mm is None:
        height_mm = width_mm * first.size[1] / first.size[0]
    grid = resolve_grid(
        width_mm,
        height_mm,
        n_images=len(paths),
        pitch_mm=pitch_mm,
        row_mm=row_mm,
        ridge_height_mm=ridge_height_mm,
        nozzle_mm=nozzle_mm,
        lines_per_facet=lines_per_facet,
        orientation=orientation,
    )
    if crops is None:
        crops = [None] * len(paths)
    if len(crops) != len(paths):
        raise ValueError("each picture needs its own crop")
    if flips is None:
        flips = [(False, False)] * len(paths)
    if len(flips) != len(paths):
        raise ValueError("each picture needs its own flip")
    plate_aspect = grid.width_mm / grid.height_mm
    if orientation == "horizontal":
        pixel_size = (grid.n_rows, grid.n_ridges)
    else:
        pixel_size = (grid.n_ridges, grid.n_rows)
    fitted = [
        _fit_image(path, pixel_size, plate_aspect=plate_aspect, crop=crop, flip=flip)
        for path, crop, flip in zip(paths, crops, flips)
    ]
    if palette is None:
        samples = np.concatenate([image.reshape(-1, 3) for image in fitted], axis=0)
        palette = choose_palette(samples, max_colors)
    else:
        palette = as_rgb_array(palette)
    if dither:
        indices = [floyd_steinberg(image, palette) for image in fitted]
    else:
        indices = [nearest_indices(image, palette) for image in fitted]
    return build_from_indices(
        indices,
        palette,
        width_mm=grid.width_mm,
        height_mm=grid.height_mm,
        base_mm=base_mm,
        ridge_height_mm=grid.ridge_height_mm,
        base_rgb=base_rgb,
        embed_mm=embed_mm,
        seam_mm=seam_mm,
        line_mm=nozzle_mm * (_LINE_MM / _NOZZLE_MM),
        fitted=fitted,
        orientation=orientation,
        magnets=magnets,
        crest_line=crest_line,
        crest_rgb=crest_rgb,
        crest_width_mm=crest_width_mm,
        crest_height_mm=crest_height_mm,
    )


def _build_horizontal(
    indices: list[np.ndarray],
    palette: np.ndarray,
    *,
    width_mm: float,
    height_mm: float,
    base_mm: float,
    ridge_height_mm: float | None,
    base_rgb: tuple[int, int, int],
    embed_mm: float,
    seam_mm: float,
    line_mm: float,
    fitted: list[np.ndarray] | None,
    crest_line: bool = False,
    crest_rgb: tuple[int, int, int] = (0, 0, 0),
    crest_width_mm: float = 0.4,
    crest_height_mm: float = 0.2,
) -> LenticModel:
    """Ridges run across the width. The first picture faces down, the last faces up.

    The vertical builder is reused in a swapped frame, then X and Y are exchanged
    and the winding is reversed so the faces still point outward.
    """
    local = [np.ascontiguousarray(np.flip(np.asarray(view), axis=(0, 1)).T) for view in indices]
    model = build_from_indices(
        local,
        palette,
        width_mm=height_mm,
        height_mm=width_mm,
        base_mm=base_mm,
        ridge_height_mm=ridge_height_mm,
        base_rgb=base_rgb,
        embed_mm=embed_mm,
        seam_mm=seam_mm,
        line_mm=line_mm,
        orientation="vertical",
        crest_line=crest_line,
        crest_rgb=crest_rgb,
        crest_width_mm=crest_width_mm,
        crest_height_mm=crest_height_mm,
    )
    for part in model.parts:
        part.triangles = _swap_xy(part.triangles)
    n_ridges, n_rows = np.asarray(indices[0]).shape
    model.grid = Grid(
        width_mm,
        height_mm,
        height_mm / n_ridges,
        width_mm / n_rows,
        model.grid.ridge_height_mm,
        int(n_ridges),
        int(n_rows),
    )
    model.orientation = "horizontal"
    model.angle_names = angle_names_for(len(indices), "horizontal")
    model.previews = [model.palette[np.asarray(view)] for view in indices]
    if fitted is not None:
        model.fitted = list(fitted)
    top = np.zeros((n_ridges * len(indices), n_rows, 3), dtype=np.uint8)
    for angle, preview in enumerate(model.previews):
        top[angle :: len(indices), :, :] = preview
    model.top_preview = top
    return model


def _blunt_crest(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    x0_foot: float,
    x1_foot: float,
    foot: float,
    line: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Flatten the crest so each slope stops while it is still one line wide.

    The old crest x stays in the profile, at the lower height, so the two
    pictures meet on a short flat instead of a knife edge.
    """
    dx = x1 - x0
    dz = z1 - z0
    if abs(dz) < 1e-9 or abs(dx) <= line + 1e-6:
        return None
    if z1 >= z0:
        xs = x1 - math.copysign(line, dx)
        zs = z0 + (xs - x0) / dx * dz
        ring_x = np.array([x0, xs, x1, x1_foot, x0_foot], dtype=np.float64)
        ring_z = np.array([z0, zs, zs, foot, foot], dtype=np.float64)
    else:
        xs = x0 + math.copysign(line, dx)
        zs = z0 + (xs - x0) / dx * dz
        ring_x = np.array([x0, xs, x1, x1_foot, x0_foot], dtype=np.float64)
        ring_z = np.array([zs, zs, z1, foot, foot], dtype=np.float64)
    return ring_x, ring_z


def _check_crest(pitch: float, width: float, height: float) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("the crest line width and height must be greater than 0")
    if width >= pitch:
        raise ValueError("the crest line has to be narrower than the ridge pitch")


def _crest_top(pitch: float, foot: float, peak: float, n_images: int, line_mm: float) -> float:
    """Height of the flat where a line can sit between the two pictures."""
    if n_images == 2 and line_mm > 0:
        half = pitch / 2.0
        if half > line_mm:
            return foot + (half - line_mm) / half * (peak - foot)
    return peak


def _crest_boxes(
    grid: Grid,
    base_mm: float,
    embed_mm: float,
    n_images: int,
    line_mm: float,
    width: float,
    height: float,
) -> np.ndarray:
    foot = base_mm - embed_mm
    peak = base_mm + grid.ridge_height_mm
    top = _crest_top(grid.pitch_mm, foot, peak, n_images, line_mm)
    half = width / 2.0
    boxes = [
        box_mesh(
            (ridge + 0.5) * grid.pitch_mm - half,
            0.0,
            top,
            (ridge + 0.5) * grid.pitch_mm + half,
            grid.height_mm,
            top + height,
        )
        for ridge in range(grid.n_ridges)
    ]
    return np.ascontiguousarray(np.concatenate(boxes))


def _add_filament(parts: list[MeshPart], rgb: tuple[int, int, int], triangles: np.ndarray) -> None:
    """Keep one mesh per filament, so a black crest joins a black picture."""
    color = tuple(int(channel) for channel in rgb)
    for part in parts:
        if part.role == "filament" and tuple(int(channel) for channel in part.rgb) == color:
            part.triangles = np.ascontiguousarray(np.concatenate([part.triangles, triangles]))
            part.rgb = color
            return
    parts.append(MeshPart("filament", color, triangles))


def _check_magnets(spec: MagnetSpec) -> None:
    if spec.shape not in {"round", "rect"}:
        raise ValueError("magnet shape must be round or rectangular")
    if spec.arrangement not in {"across", "down", "grid", "center"}:
        raise ValueError("magnet arrangement must be along the width, along the height, a grid, or centered")
    if spec.edge_mm < 0:
        raise ValueError("distance from the edge must be zero or more")
    if spec.count < 1 or spec.count > 24:
        raise ValueError("magnet count must be from 1 to 24")
    if spec.thickness_mm <= 0 or spec.above_mm <= 0:
        raise ValueError("magnet thickness, and the plastic above the magnets, must be greater than 0")
    if spec.below_mm < 0:
        raise ValueError("thickness below the magnets must be zero or more")
    if spec.shape == "round" and spec.diameter_mm <= 0:
        raise ValueError("magnet diameter must be greater than 0")
    if spec.shape == "rect" and (spec.width_mm <= 0 or spec.length_mm <= 0):
        raise ValueError("magnet width and length must be greater than 0")


def _install_magnets(model: LenticModel, spec: MagnetSpec) -> None:
    """Replace the solid base with one that has pockets for pause-and-insert magnets."""
    width = model.grid.width_mm
    height = model.grid.height_mm
    hole_w, hole_h = spec.hole_span()
    centers = _magnet_centers(
        spec.count,
        width,
        height,
        hole_w,
        hole_h,
        spec.arrangement,
        spec.edge_mm,
    )
    z0 = spec.below_mm
    z1 = spec.below_mm + spec.thickness_mm
    base = next(part for part in model.parts if part.role == "base")
    if spec.below_mm == 0:
        base.triangles = open_pocket_base(width, height, model.base_mm, z1, _hole_outlines(spec, centers, hole_w, hole_h))
    else:
        if spec.shape == "round":
            radius = hole_w / 2.0
            holes = [cylinder_cavity(cx, cy, radius, z0, z1) for cx, cy in centers]
        else:
            holes = [
                np.ascontiguousarray(
                    box_mesh(
                        cx - hole_w / 2.0,
                        cy - hole_h / 2.0,
                        z0,
                        cx + hole_w / 2.0,
                        cy + hole_h / 2.0,
                        z1,
                    )[:, ::-1, :]
                )
                for cx, cy in centers
            ]
        outer = box_mesh(0.0, 0.0, 0.0, width, height, model.base_mm)
        base.triangles = np.ascontiguousarray(np.concatenate([outer, *holes]))
    model.magnets = spec
    model.magnet_places = centers


def _hole_outlines(
    spec: MagnetSpec,
    centers: list[tuple[float, float]],
    hole_w: float,
    hole_h: float,
) -> list[np.ndarray]:
    """Counter-clockwise outlines of the pockets, in the plate's XY plane."""
    rings = []
    for cx, cy in centers:
        if spec.shape == "round":
            rings.append(circle_polygon(cx, cy, hole_w / 2.0))
        else:
            x0 = cx - hole_w / 2.0
            x1 = cx + hole_w / 2.0
            y0 = cy - hole_h / 2.0
            y1 = cy + hole_h / 2.0
            rings.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64))
    return rings


def _magnet_centers(
    count: int,
    width: float,
    height: float,
    hole_w: float,
    hole_h: float,
    arrangement: str = "grid",
    edge: float = 2.5,
) -> list[tuple[float, float]]:
    """Even centers, with plastic around each hole and at the edge of the plate."""
    if count < 1:
        raise ValueError("magnet count must be at least 1")
    if edge < 0:
        raise ValueError("distance from the edge must be zero or more")
    gap = 2.0
    if arrangement == "across":
        layouts = [(count, 1)]
    elif arrangement == "down":
        layouts = [(1, count)]
    else:
        layouts = [(cols, math.ceil(count / cols)) for cols in range(1, count + 1)]
    best: tuple[tuple[float, float, int], int, int] | None = None
    for cols, rows in layouts:
        need_w = cols * hole_w + (cols - 1) * gap + 2 * edge
        need_h = rows * hole_h + (rows - 1) * gap + 2 * edge
        if need_w > width + 1e-6 or need_h > height + 1e-6:
            continue
        empty = cols * rows - count
        aspect = abs(math.log((cols / rows) / max(width / height, 1e-6)))
        score = (float(empty), aspect, rows)
        if best is None or score < best[0]:
            best = (score, cols, rows)
    if best is None:
        if arrangement == "across":
            raise ValueError("those magnets do not fit in one row across the plate. Use fewer, or a smaller magnet.")
        if arrangement == "down":
            raise ValueError("those magnets do not fit in one column down the plate. Use fewer, or a smaller magnet.")
        if arrangement == "center":
            raise ValueError("those magnets do not fit in the center with that distance from the edge. Use fewer, or a smaller magnet.")
        raise ValueError("those magnets do not fit on this plate. Use fewer, or a smaller magnet.")
    cols, rows = best[1], best[2]
    if arrangement == "center":
        xs = _packed(cols, width, hole_w, gap)
        ys = _packed(rows, height, hole_h, gap)
    else:
        xs = _spread(cols, width, hole_w, edge, gap)
        ys = _spread(rows, height, hole_h, edge, gap)
    if xs is None or ys is None:
        raise ValueError("those magnets do not fit on this plate. Use fewer, or a smaller magnet.")
    pitch = xs[1] - xs[0] if len(xs) > 1 else 0.0
    points: list[tuple[float, float]] = []
    placed = 0
    for row in range(rows):
        across = min(cols, count - placed)
        if across == len(xs):
            row_xs = xs
        else:
            row_span = (across - 1) * pitch
            start = (width - row_span) / 2.0
            row_xs = [start + col * pitch for col in range(across)]
        for x in row_xs:
            points.append((x, ys[row]))
        placed += across
    return points


def _packed(count: int, span: float, hole: float, gap: float) -> list[float]:
    """Centers packed together in the middle of the plate."""
    if count == 1:
        return [span / 2.0]
    pitch = hole + gap
    used = count * hole + (count - 1) * gap
    start = (span - used) / 2.0 + hole / 2.0
    return [start + index * pitch for index in range(count)]


def _spread(count: int, span: float, hole: float, edge: float, gap: float) -> list[float] | None:
    """Centers spread across the plate, never closer than the hole plus a gap."""
    inset = edge + hole / 2.0
    if count == 1:
        if span < hole + 2 * edge:
            return None
        return [span / 2.0]
    room = span - 2 * inset
    if room < (count - 1) * (hole + gap) - 1e-6:
        return None
    step = room / (count - 1)
    return [inset + index * step for index in range(count)]


def _color_runs(column: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start row, end row, and color for each solid run down one ridge."""
    column = np.asarray(column)
    if len(column) == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, empty
    change = np.flatnonzero(column[1:] != column[:-1]) + 1
    starts = np.empty(len(change) + 1, dtype=np.int32)
    ends = np.empty_like(starts)
    starts[0] = 0
    starts[1:] = change
    ends[:-1] = change
    ends[-1] = len(column)
    return starts, ends, column[starts]


def _swap_xy(triangles: np.ndarray) -> np.ndarray:
    swapped = np.ascontiguousarray(triangles[:, :, [1, 0, 2]])
    return np.ascontiguousarray(swapped[:, ::-1, :])


def _open_rgb(path: str | Path) -> Image.Image:
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"image not found: {file}")
    with Image.open(file) as image:
        # Camera photos store rotation in a tag. Honor it so the plate matches the picture.
        return ImageOps.exif_transpose(image).convert("RGB")


def _fit_image(
    path: str | Path,
    pixel_size: tuple[int, int],
    *,
    plate_aspect: float,
    crop: tuple[float, float, float, float] | None,
    flip: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """Sample a picture onto the plate.

    Ridges are wider than rows, so the pixel grid is not the same shape as the
    plate. Crop in the plate's shape first, then resample. A landscape photo
    on a landscape plate keeps its sides.
    """
    image = _open_rgb(path)
    flip_horizontal, flip_vertical = flip
    if flip_horizontal:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    width_px, height_px = pixel_size
    if crop is None and image.size == (width_px, height_px):
        return np.asarray(image, dtype=np.uint8)
    if crop is None:
        crop = _max_crop(image.width / image.height, plate_aspect)
    image = _crop_fraction(image, crop)
    if image.size != (width_px, height_px):
        image = image.resize((max(1, width_px), max(1, height_px)), Image.Resampling.BOX)
    return np.asarray(image, dtype=np.uint8)


def _max_crop(image_aspect: float, plate_aspect: float) -> tuple[float, float, float, float]:
    """Largest centered region of the plate's shape, as fractions of the picture."""
    if image_aspect <= 0 or plate_aspect <= 0:
        raise ValueError("aspect ratios must be positive")
    if image_aspect > plate_aspect:
        width = plate_aspect / image_aspect
        return ((1.0 - width) / 2.0, 0.0, width, 1.0)
    height = image_aspect / plate_aspect
    return (0.0, (1.0 - height) / 2.0, 1.0, min(height, 1.0))


def _crop_fraction(image: Image.Image, crop: tuple[float, float, float, float]) -> Image.Image:
    x, y, width, height = (float(value) for value in crop)
    x = min(max(x, 0.0), 0.99)
    y = min(max(y, 0.0), 0.99)
    width = min(max(width, 0.01), 1.0 - x)
    height = min(max(height, 0.01), 1.0 - y)
    left = int(round(x * image.width))
    top = int(round(y * image.height))
    right = min(image.width, max(left + 1, int(round((x + width) * image.width))))
    bottom = min(image.height, max(top + 1, int(round((y + height) * image.height))))
    return image.crop((left, top, right, bottom))
