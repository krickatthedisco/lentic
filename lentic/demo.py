"""Draw a two-picture sample and export a plate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lentic.build import build_from_images, resolve_grid
from lentic.color import parse_hex
from lentic.export import export_model

PALETTE = ("#ffd60a", "#e63946", "#0d1b2a", "#4cc9f0")


def run_demo(out_dir: str | Path, width_mm: float = 64.0) -> Path:
    height_mm = width_mm * 9.0 / 16.0
    grid = resolve_grid(width_mm, height_mm, n_images=2, row_mm=0.8, nozzle_mm=0.4, lines_per_facet=2)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    left, right = _pictures((grid.n_ridges, grid.n_rows))
    left_path = out / "input_left.png"
    right_path = out / "input_right.png"
    left.save(left_path)
    right.save(right_path)
    model = build_from_images(
        [left_path, right_path],
        width_mm=width_mm,
        height_mm=height_mm,
        row_mm=0.8,
        nozzle_mm=0.4,
        lines_per_facet=2,
        palette=np.array([parse_hex(color) for color in PALETTE], dtype=np.uint8),
        dither=False,
        base_rgb=parse_hex("#f4f1ea"),
    )
    return export_model(model, out)


def _pictures(size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    width, height = size
    left = Image.new("RGB", size, "#ffd60a")
    right = Image.new("RGB", size, "#0d1b2a")
    draw_left = ImageDraw.Draw(left)
    draw_right = ImageDraw.Draw(right)
    radius = int(min(width, height) * 0.32)
    cx, cy = width // 2, height // 2
    draw_left.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#e63946")
    draw_right.polygon(
        [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)],
        fill="#4cc9f0",
    )
    return left, right
