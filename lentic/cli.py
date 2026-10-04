"""Command line for building a color lenticular plate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lentic.build import build_from_images
from lentic.color import parse_hex, rgb_to_hex
from lentic.export import export_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lentic",
        description="Build a color lenticular plate from two or three images.",
        epilog=(
            "examples:\n"
            "  lentic left.png right.png --width 120 --out plate\n"
            "  lentic left.png right.png --width 80 --palette #111111,#f5f5f5 --no-dither\n"
            "  lentic --demo --out demo_out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="*", help="left, right, and optional front image")
    parser.add_argument("--demo", action="store_true", help="export a red-circle / cyan-diamond sample")
    parser.add_argument("--gui", action="store_true", help="open the design window in a browser")
    parser.add_argument("--port", type=int, default=8765, help="port for the design window")
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="ridges run sideways; pass the top image, then the bottom image",
    )
    parser.add_argument("--out", default="lentic-out", help="output folder")
    parser.add_argument("--width", type=float, help="plate width in millimeters")
    parser.add_argument("--height", type=float, help="plate height in millimeters; default keeps the first image's aspect")
    parser.add_argument("--pitch", type=float, help="ridge width in millimeters; default is nozzle x lines x image count")
    parser.add_argument("--row", type=float, default=0.4, help="picture row height in millimeters")
    parser.add_argument("--ridge-height", type=float, help="slope height above the base; default is 0.8 x pitch")
    parser.add_argument("--base", type=float, default=0.8, help="backing thickness in millimeters")
    parser.add_argument("--base-color", default="#f4f1ea", help="backing filament color")
    parser.add_argument("--nozzle", type=float, default=0.4, help="nozzle diameter used when pitch is omitted")
    parser.add_argument("--lines", type=int, default=5, help="extrusion lines per slope when pitch is omitted")
    parser.add_argument("--palette", help="comma-separated filament colors, for example #111111,#f5f5f5,#e63946")
    parser.add_argument("--max-colors", type=int, default=4, help="palette size when --palette is omitted")
    parser.add_argument("--no-dither", action="store_true", help="snap each pixel to the nearest filament with no dither")
    parser.add_argument("--crest-line", action="store_true", help="add one filament line along the top of each ridge")
    parser.add_argument("--crest-color", default="#000000", help="crest line color")
    parser.add_argument("--crest-width", type=float, default=0.4, help="crest line width in millimeters")
    parser.add_argument("--crest-height", type=float, default=0.2, help="crest line height in millimeters")
    if argv is None and getattr(sys, "frozen", False) and len(sys.argv) == 1:
        argv = ["--gui"]
    args = parser.parse_args(argv)
    if args.gui:
        from lentic.gui import launch

        launch(port=args.port)
        return 0

    try:
        if args.demo:
            from lentic.demo import run_demo

            html = run_demo(args.out, width_mm=args.width or 64.0)
            print(f"wrote {html}")
            return 0
        if len(args.images) not in (2, 3):
            print("error: pass two or three images, or use --demo", file=sys.stderr)
            return 2
        images = list(args.images)
        if args.horizontal and len(images) == 2:
            images = [images[1], images[0]]
        elif args.horizontal and len(images) == 3:
            top, bottom, front = images
            images = [bottom, front, top]
        if args.width is None:
            print("error: --width is required (millimeters)", file=sys.stderr)
            return 2
        palette = None
        if args.palette:
            palette = np.array([parse_hex(item) for item in args.palette.split(",")], dtype=np.uint8)
        model = build_from_images(
            images,
            width_mm=args.width,
            height_mm=args.height,
            pitch_mm=args.pitch,
            row_mm=args.row,
            ridge_height_mm=args.ridge_height,
            base_mm=args.base,
            nozzle_mm=args.nozzle,
            lines_per_facet=args.lines,
            palette=palette,
            max_colors=args.max_colors,
            dither=not args.no_dither,
            base_rgb=parse_hex(args.base_color),
            orientation="horizontal" if args.horizontal else "vertical",
            crest_line=args.crest_line,
            crest_rgb=parse_hex(args.crest_color),
            crest_width_mm=args.crest_width,
            crest_height_mm=args.crest_height,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    html = export_model(model, args.out)
    grid = model.grid
    if grid.pitch_mm < args.nozzle:
        print("warning: ridge pitch is smaller than the nozzle, so the slopes may blend", file=sys.stderr)
    print(
        f"plate {grid.width_mm:.1f} x {grid.height_mm:.1f} x {model.base_mm + grid.ridge_height_mm:.1f} mm, "
        f"{grid.n_ridges} ridges at {grid.pitch_mm:.2f} mm"
    )
    print("angles: " + ", ".join(model.angle_names))
    for part in model.parts:
        print(f"  {part.role} #{rgb_to_hex(part.rgb)}  {len(part.triangles)} triangles")
    print(f"wrote {html}")
    return 0
