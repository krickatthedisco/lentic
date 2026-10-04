"""Build a color lenticular plate from two or three pictures.

Each ridge is a solid wedge. The first picture is on the slope that faces
left, the second on the slope that faces right, and a third picture on the
flat top. Filament colors are separate meshes so a multi-material slicer
can print them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from lentic.color import as_rgb_array, choose_palette, floyd_steinberg, nearest_indices
from lentic.mesh import extrude_xz, facet_profiles

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
    fitted: list[np.ndarray] | None = None,
    orientation: str = "vertical",
) -> LenticModel:
    if len(indices) not in ANGLE_NAMES:
        raise ValueError("use 2 images or 3")
    if orientation == "horizontal":
        return _build_horizontal(
            indices,
            palette,
            width_mm=width_mm,
            height_mm=height_mm,
            base_mm=base_mm,
            ridge_height_mm=ridge_height_mm,
            base_rgb=base_rgb,
            embed_mm=embed_mm,
            seam_mm=seam_mm,
            fitted=fitted,
        )
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
    buckets: dict[int, list[np.ndarray]] = {color: [] for color in range(len(palette))}

    for ridge in range(n_ridges):
        ridge_x0 = ridge * grid.pitch_mm
        ridge_x1 = (ridge + 1) * grid.pitch_mm
        for angle, (fx0, fz0, fx1, fz1) in enumerate(profiles):
            view = indices[angle]
            row = 0
            while row < n_rows:
                color = int(view[row, ridge])
                end = row + 1
                while end < n_rows and int(view[end, ridge]) == color:
                    end += 1
                y1 = height_mm - row * grid.row_mm
                y0 = height_mm - end * grid.row_mm
                x0 = ridge_x0 + fx0 * grid.pitch_mm
                x1 = ridge_x0 + fx1 * grid.pitch_mm
                if fx0 > 0:
                    x0 = max(ridge_x0, x0 - seam)
                if fx1 < 1:
                    x1 = min(ridge_x1, x1 + seam)
                if x1 - x0 < 1e-6:
                    row = end
                    continue
                z0 = foot + fz0 * span
                z1 = foot + fz1 * span
                solid = extrude_xz(
                    np.array([[x0, z0], [x1, z1], [x1, foot], [x0, foot]], dtype=np.float64),
                    y0,
                    y1,
                )
                if len(solid):
                    buckets[color].append(solid)
                row = end

    base_poly = np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, base_mm], [0.0, base_mm]],
        dtype=np.float64,
    )
    parts = [
        MeshPart("base", tuple(int(channel) for channel in base_rgb), extrude_xz(base_poly, 0.0, height_mm))
    ]
    for color, solids in buckets.items():
        if not solids:
            continue
        rgb = tuple(int(channel) for channel in palette[color])
        parts.append(MeshPart("filament", rgb, np.concatenate(solids, axis=0)))

    previews = [palette[view] for view in indices]
    top = np.zeros((n_rows, n_ridges * len(indices), 3), dtype=np.uint8)
    for angle, preview in enumerate(previews):
        top[:, angle :: len(indices), :] = preview
    return LenticModel(
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
    )


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
    plate_aspect = grid.width_mm / grid.height_mm
    if orientation == "horizontal":
        pixel_size = (grid.n_rows, grid.n_ridges)
    else:
        pixel_size = (grid.n_ridges, grid.n_rows)
    fitted = [
        _fit_image(path, pixel_size, plate_aspect=plate_aspect, crop=crop)
        for path, crop in zip(paths, crops)
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
        fitted=fitted,
        orientation=orientation,
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
    fitted: list[np.ndarray] | None,
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
        orientation="vertical",
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


def _swap_xy(triangles: np.ndarray) -> np.ndarray:
    swapped = np.ascontiguousarray(triangles[:, :, [1, 0, 2]])
    return np.ascontiguousarray(swapped[:, ::-1, :])


def _open_rgb(path: str | Path) -> Image.Image:
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"image not found: {file}")
    with Image.open(file) as image:
        return image.convert("RGB")


def _fit_image(
    path: str | Path,
    pixel_size: tuple[int, int],
    *,
    plate_aspect: float,
    crop: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Sample a picture onto the plate.

    Ridges are wider than rows, so the pixel grid is not the same shape as the
    plate. Crop in the plate's shape first, then resample. A landscape photo
    on a landscape plate keeps its sides.
    """
    image = _open_rgb(path)
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
