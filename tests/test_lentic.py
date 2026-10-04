"""Geometry, color matching, and file export for lentic plates."""

from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from lentic.build import MagnetSpec, _open_rgb, build_from_images, build_from_indices
from lentic.cli import main
from lentic.color import choose_palette, floyd_steinberg, median_cut, nearest_indices, parse_hex
from lentic.export import export_model
from lentic.mesh import box_mesh, cylinder_cavity, extrude_trapezoids, extrude_xz, mesh_volume, triangle_normals


RED = np.array([255, 0, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 255], dtype=np.uint8)
GREEN = np.array([0, 180, 0], dtype=np.uint8)
PALETTE = np.stack([RED, BLUE, GREEN])


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise AssertionError(f"bad stl size for {count} triangles")
    payload = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(count, 50)
    copied = np.ascontiguousarray(payload[:, 12:48])
    return copied.view("<f4").reshape(count, 3, 3).copy()


def slope_normals(triangles: np.ndarray) -> np.ndarray:
    normals = triangle_normals(triangles)
    xspan = triangles[:, :, 0].max(axis=1) - triangles[:, :, 0].min(axis=1)
    zspan = triangles[:, :, 2].max(axis=1) - triangles[:, :, 2].min(axis=1)
    mask = (np.abs(normals[:, 1]) < 0.3) & (xspan > 0.15) & (zspan > 0.15)
    return normals[mask]


class TestColor(unittest.TestCase):
    def test_hex_and_nearest(self):
        self.assertEqual(parse_hex("#f00"), (255, 0, 0))
        self.assertEqual(parse_hex("abc"), (170, 187, 204))
        with self.assertRaises(ValueError):
            parse_hex("nope")
        pixels = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        self.assertEqual(nearest_indices(pixels, PALETTE[:2]).tolist(), [[0, 1]])

    def test_median_cut_finds_red_and_blue(self):
        cloud = np.concatenate(
            [np.tile(RED, (40, 1)), np.tile(BLUE, (40, 1))],
            axis=0,
        )
        palette = median_cut(cloud, 2)
        found = {tuple(int(channel) for channel in color) for color in palette}
        self.assertIn((255, 0, 0), found)
        self.assertIn((0, 0, 255), found)

    def test_dither_uses_both_filaments(self):
        gray = np.full((8, 8, 3), 128, dtype=np.uint8)
        palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        chosen = floyd_steinberg(gray, palette)
        counts = np.bincount(chosen.ravel(), minlength=2)
        self.assertGreater(counts[0], 8)
        self.assertGreater(counts[1], 8)

    def test_small_blue_region_gets_its_own_filament(self):
        cream = np.array([253, 248, 232], dtype=np.uint8)
        red = np.array([214, 50, 46], dtype=np.uint8)
        ink = np.array([40, 38, 36], dtype=np.uint8)
        blue = np.array([42, 81, 132], dtype=np.uint8)
        cloud = np.concatenate(
            [
                np.tile(cream, (5000, 1)),
                np.tile(red, (400, 1)),
                np.tile(ink, (300, 1)),
                np.tile(blue, (70, 1)),
            ]
        )
        palette = choose_palette(cloud, 8)
        nearest = palette[int(nearest_indices(blue.reshape(1, 1, 3), palette)[0, 0])]
        self.assertGreater(int(nearest[2]), int(nearest[0]) + 30)
        self.assertGreater(int(nearest[2]), int(nearest[1]) + 15)

    def test_dither_keeps_a_flat_patch(self):
        image = np.full((12, 12, 3), 250, dtype=np.uint8)
        image[4:8, 4:8] = (42, 81, 132)
        palette = np.array([[250, 250, 250], [42, 81, 132]], dtype=np.uint8)
        chosen = floyd_steinberg(image, palette)
        self.assertTrue(np.all(chosen[5:7, 5:7] == 1))


class TestBuild(unittest.TestCase):
    def test_batched_slopes_match_a_single_extrusion(self):
        for poly, y0, y1, foot in (
            (np.array([[0.0, 0.0], [0.8, 1.4], [0.8, -0.05], [0.0, -0.05]]), 0.0, 1.6, -0.05),
            (np.array([[0.8, 1.4], [1.6, 0.0], [1.6, -0.05], [0.8, -0.05]]), 1.6, 0.2, -0.05),
        ):
            single = extrude_xz(poly, y0, y1)
            batch = extrude_trapezoids(
                [poly[0, 0]],
                [poly[0, 1]],
                [poly[1, 0]],
                [poly[1, 1]],
                [y0],
                [y1],
                foot,
            )
            self.assertEqual(len(single), len(batch))
            self.assertAlmostEqual(mesh_volume(single), mesh_volume(batch), places=5)
        tucked = np.array([[0.0, 0.0], [0.8, 1.4], [0.9, -0.05], [-0.02, -0.05]], dtype=np.float64)
        single = extrude_xz(tucked, 0.0, 1.6)
        batch = extrude_trapezoids(
            [tucked[0, 0]],
            [tucked[0, 1]],
            [tucked[1, 0]],
            [tucked[1, 1]],
            [0.0],
            [1.6],
            -0.05,
            [tucked[3, 0]],
            [tucked[2, 0]],
        )
        self.assertAlmostEqual(mesh_volume(single), mesh_volume(batch), places=5)

    def test_left_and_right_slopes_face_opposite_ways(self):
        left = np.zeros((1, 2), dtype=np.int32)
        right = np.ones((1, 2), dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE[:2],
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=1.6,
            embed_mm=0,
            seam_mm=0,
            line_mm=0,
        )
        red = next(part for part in model.parts if part.rgb == (255, 0, 0))
        blue = next(part for part in model.parts if part.rgb == (0, 0, 255))
        self.assertEqual(len(red.triangles), 16)
        self.assertTrue(np.all(slope_normals(red.triangles)[:, 0] < -0.2))
        self.assertTrue(np.all(slope_normals(blue.triangles)[:, 0] > 0.2))
        self.assertAlmostEqual(mesh_volume(red.triangles), 2 * 0.5 * 1.0 * 1.6 * 2.0, places=5)
        base = next(part for part in model.parts if part.role == "base")
        self.assertAlmostEqual(mesh_volume(base.triangles), 4 * 2 * 0.8, places=5)
        for part in model.parts:
            self.assertGreater(mesh_volume(part.triangles), 0)

    def test_the_seam_does_not_cross_at_the_peak(self):
        left = np.zeros((1, 2), dtype=np.int32)
        right = np.ones((1, 2), dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE[:2],
            width_mm=3.2,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=1.6,
            embed_mm=0.05,
            seam_mm=0.08,
            line_mm=0,
        )
        peak = model.base_mm + model.grid.ridge_height_mm
        foot = model.base_mm - model.embed_mm
        red = next(part for part in model.parts if part.rgb == (255, 0, 0))
        blue = next(part for part in model.parts if part.rgb == (0, 0, 255))
        for ridge in range(2):
            center = (ridge + 0.5) * model.grid.pitch_mm
            red_points = red.triangles.reshape(-1, 3)
            blue_points = blue.triangles.reshape(-1, 3)
            red_tip = red_points[np.isclose(red_points[:, 2], peak)]
            blue_tip = blue_points[np.isclose(blue_points[:, 2], peak)]
            red_tip = red_tip[np.abs(red_tip[:, 0] - center) < 0.5]
            blue_tip = blue_tip[np.abs(blue_tip[:, 0] - center) < 0.5]
            self.assertGreater(len(red_tip), 0)
            self.assertGreater(len(blue_tip), 0)
            self.assertAlmostEqual(red_tip[:, 0].max(), center, places=5)
            self.assertAlmostEqual(blue_tip[:, 0].min(), center, places=5)
            red_foot = red_points[np.isclose(red_points[:, 2], foot)]
            red_foot = red_foot[(red_foot[:, 0] > center - 0.2) & (red_foot[:, 0] < center + 0.5)]
            self.assertGreater(red_foot[:, 0].max(), center + 0.05)

    def test_the_crest_stops_one_line_in_from_the_slope(self):
        left = np.zeros((1, 1), dtype=np.int32)
        right = np.ones((1, 1), dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE[:2],
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=3.2,
            embed_mm=0.05,
            seam_mm=0,
            line_mm=0.5,
        )
        center = 2.0
        nominal = model.base_mm + 3.2
        red = next(part for part in model.parts if part.rgb == (255, 0, 0)).triangles.reshape(-1, 3)
        blue = next(part for part in model.parts if part.rgb == (0, 0, 255)).triangles.reshape(-1, 3)
        self.assertLess(red[:, 2].max(), nominal - 0.4)
        self.assertLess(blue[:, 2].max(), nominal - 0.4)
        shelf = red[:, 2].max()
        red_top = red[np.isclose(red[:, 2], shelf)]
        blue_top = blue[np.isclose(blue[:, 2], shelf)]
        self.assertAlmostEqual(red_top[:, 0].max(), center, places=4)
        self.assertAlmostEqual(red_top[:, 0].min(), center - 0.5, places=4)
        self.assertAlmostEqual(blue_top[:, 0].min(), center, places=4)
        self.assertAlmostEqual(blue_top[:, 0].max(), center + 0.5, places=4)

    def test_a_crest_line_sits_on_the_flat(self):
        left = np.zeros((1, 1), dtype=np.int32)
        right = np.ones((1, 1), dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE[:2],
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=3.2,
            embed_mm=0.05,
            seam_mm=0,
            line_mm=0.5,
            crest_line=True,
            crest_rgb=(0, 0, 0),
            crest_width_mm=0.4,
            crest_height_mm=0.2,
        )
        black = next(part for part in model.parts if part.role == "crest")
        self.assertEqual(black.rgb, (0, 0, 0))
        self.assertAlmostEqual(mesh_volume(black.triangles), 0.4 * 2.0 * 0.2, places=4)
        points = black.triangles.reshape(-1, 3)
        self.assertAlmostEqual(points[:, 0].min(), 1.8, places=4)
        self.assertAlmostEqual(points[:, 0].max(), 2.2, places=4)
        foot = 0.75
        shelf = foot + (2.0 - 0.5) / 2.0 * (model.base_mm + 3.2 - foot)
        self.assertGreater(points[:, 2].min(), shelf)
        self.assertAlmostEqual(points[:, 2].min(), 3.22, places=4)
        self.assertAlmostEqual(points[:, 2].max(), 3.42, places=4)
        self.assertLess(3.2, points[:, 2].min())
        self.assertLess(points[:, 2].min(), 3.4)
        self.assertGreater(points[:, 2].max(), 3.4)
        colors = np.concatenate(
            [part.triangles.reshape(-1, 3) for part in model.parts if part.role == "filament"]
        )
        self.assertGreater(points[:, 2].min(), colors[:, 2].max())
        from lentic.export import _print_notes

        self.assertIn("0.40 mm wide and 0.20 mm tall", _print_notes(model, []))

    def test_a_crest_line_follows_sideways_ridges(self):
        left = np.zeros((1, 1), dtype=np.int32)
        right = np.ones((1, 1), dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE[:2],
            width_mm=8,
            height_mm=4,
            base_mm=0.8,
            ridge_height_mm=3.2,
            embed_mm=0.05,
            seam_mm=0,
            line_mm=0.5,
            orientation="horizontal",
            crest_line=True,
            crest_width_mm=0.4,
            crest_height_mm=0.2,
        )
        black = next(part for part in model.parts if part.role == "crest")
        self.assertAlmostEqual(mesh_volume(black.triangles), 0.4 * 8.0 * 0.2, places=4)
        points = black.triangles.reshape(-1, 3)
        self.assertAlmostEqual(points[:, 0].min(), 0.0, places=3)
        self.assertAlmostEqual(points[:, 0].max(), 8.0, places=3)
        self.assertAlmostEqual(points[:, 1].min(), 1.8, places=3)
        self.assertAlmostEqual(points[:, 1].max(), 2.2, places=3)

    def test_a_black_crest_stays_its_own_part_above_black_plastic(self):
        black = np.zeros((2, 1), dtype=np.int32)
        red = np.ones((2, 1), dtype=np.int32)
        model = build_from_indices(
            [black, red],
            np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8),
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=3.2,
            embed_mm=0.05,
            seam_mm=0,
            line_mm=0.5,
            crest_line=True,
            crest_rgb=(0, 0, 0),
            crest_width_mm=0.4,
            crest_height_mm=0.2,
            layer_height_mm=0.2,
            initial_layer_mm=0.2,
        )
        crest = [part for part in model.parts if part.role == "crest"]
        self.assertEqual(len(crest), 1)
        picture = next(part for part in model.parts if part.role == "filament" and part.rgb == (0, 0, 0))
        self.assertGreater(crest[0].triangles[:, :, 2].min(), picture.triangles[:, :, 2].max())

    def test_a_crest_line_wider_than_the_pitch_is_rejected(self):
        left = np.zeros((1, 1), dtype=np.int32)
        right = np.ones((1, 1), dtype=np.int32)
        with self.assertRaises(ValueError):
            build_from_indices(
                [left, right],
                PALETTE[:2],
                width_mm=4,
                height_mm=2,
                crest_line=True,
                crest_width_mm=4,
            )

    def test_image_top_is_the_high_y_side(self):
        left = np.array([[0, 0], [1, 1]], dtype=np.int32)
        right = np.full((2, 2), 2, dtype=np.int32)
        model = build_from_indices(
            [left, right],
            PALETTE,
            width_mm=4,
            height_mm=4,
            ridge_height_mm=1.0,
            embed_mm=0,
            seam_mm=0,
        )
        red = next(part for part in model.parts if part.rgb == (255, 0, 0))
        blue = next(part for part in model.parts if part.rgb == (0, 0, 255))
        self.assertGreaterEqual(red.triangles[:, :, 1].min(), 2 - 1e-6)
        self.assertLessEqual(blue.triangles[:, :, 1].max(), 2 + 1e-6)
        self.assertEqual(tuple(int(channel) for channel in model.previews[0][0, 0]), (255, 0, 0))

    def test_front_face_points_up(self):
        views = [np.full((1, 1), index, dtype=np.int32) for index in range(3)]
        model = build_from_indices(
            views,
            PALETTE,
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=2,
            embed_mm=0,
            seam_mm=0,
        )
        front = next(part for part in model.parts if part.rgb == (0, 0, 255))
        triangles = front.triangles
        normals = triangle_normals(triangles)
        mean_z = triangles[:, :, 2].mean(axis=1)
        zspan = triangles[:, :, 2].max(axis=1) - triangles[:, :, 2].min(axis=1)
        top = (zspan < 1e-6) & (mean_z > 2.5)
        self.assertGreater(top.sum(), 0)
        self.assertTrue(np.all(normals[top, 2] > 0.9))

    def test_one_image_is_rejected(self):
        with self.assertRaises(ValueError):
            build_from_indices([np.zeros((1, 1), dtype=np.int32)], PALETTE[:1], width_mm=10, height_mm=10)

    def test_images_round_trip_colors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(root / "a.png")
            Image.new("RGB", (2, 2), (0, 0, 255)).save(root / "b.png")
            model = build_from_images(
                [root / "a.png", root / "b.png"],
                width_mm=4,
                height_mm=4,
                pitch_mm=2,
                row_mm=2,
                palette=PALETTE[:2],
                dither=False,
                ridge_height_mm=1,
            )
            self.assertTrue(np.all(model.previews[0] == RED))
            self.assertTrue(np.all(model.previews[1] == BLUE))

    def test_landscape_photo_keeps_both_sides_on_a_landscape_plate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture = Image.new("RGB", (100, 50), (0, 0, 255))
            picture.paste(Image.new("RGB", (15, 50), (255, 0, 0)), (0, 0))
            picture.save(root / "wide.png")
            Image.new("RGB", (40, 20), (0, 0, 255)).save(root / "other.png")
            model = build_from_images(
                [root / "wide.png", root / "other.png"],
                width_mm=8,
                height_mm=4,
                pitch_mm=0.8,
                row_mm=0.4,
                palette=PALETTE[:2],
                dither=False,
                ridge_height_mm=0.6,
            )
            view = model.previews[0]
            self.assertEqual(tuple(int(channel) for channel in view[0, 0]), (255, 0, 0))
            self.assertEqual(tuple(int(channel) for channel in view[0, -1]), (0, 0, 255))

    def test_vertical_flip_puts_the_bottom_row_on_top(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture = Image.new("RGB", (4, 2), (0, 0, 255))
            picture.paste(Image.new("RGB", (4, 1), (255, 0, 0)), (0, 0))
            picture.save(root / "stack.png")
            Image.new("RGB", (4, 2), (0, 0, 255)).save(root / "other.png")
            model = build_from_images(
                [root / "stack.png", root / "other.png"],
                width_mm=4,
                height_mm=2,
                pitch_mm=1,
                row_mm=1,
                palette=PALETTE[:2],
                dither=False,
                ridge_height_mm=0.6,
                flips=[(False, True), (False, False)],
            )
            view = model.previews[0]
            self.assertEqual(tuple(int(channel) for channel in view[0, 0]), (0, 0, 255))
            self.assertEqual(tuple(int(channel) for channel in view[-1, 0]), (255, 0, 0))

    def test_upside_down_photo_tag_is_turned_upright(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "turned.jpg"
            picture = Image.new("RGB", (32, 16), (0, 0, 255))
            picture.paste(Image.new("RGB", (32, 8), (255, 0, 0)), (0, 0))
            exif = Image.Exif()
            exif[274] = 3
            picture.save(path, exif=exif, quality=95)
            opened = np.asarray(_open_rgb(path))
            self.assertGreater(int(opened[0, 0, 2]), 200)
            self.assertGreater(int(opened[-1, 0, 0]), 200)

    def test_crop_can_keep_only_the_left_half(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture = Image.new("RGB", (40, 20), (0, 0, 255))
            picture.paste(Image.new("RGB", (20, 20), (255, 0, 0)), (0, 0))
            picture.save(root / "wide.png")
            Image.new("RGB", (40, 20), (255, 0, 0)).save(root / "other.png")
            model = build_from_images(
                [root / "wide.png", root / "other.png"],
                width_mm=4,
                height_mm=4,
                pitch_mm=1,
                row_mm=1,
                palette=PALETTE[:2],
                dither=False,
                ridge_height_mm=0.6,
                crops=[(0.0, 0.0, 0.5, 0.5), None],
            )
            self.assertTrue(np.all(model.previews[0] == RED))


class TestExport(unittest.TestCase):
    def test_files_and_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(root / "a.png")
            Image.new("RGB", (2, 2), (0, 0, 255)).save(root / "b.png")
            out = root / "plate"
            code = main(
                [
                    str(root / "a.png"),
                    str(root / "b.png"),
                    "--width",
                    "4",
                    "--height",
                    "4",
                    "--pitch",
                    "2",
                    "--row",
                    "2",
                    "--ridge-height",
                    "1",
                    "--palette",
                    "#ff0000,#0000ff",
                    "--no-dither",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            red = read_stl_vertices(out / "filament_ff0000.stl")
            self.assertEqual(len(red), 24)
            self.assertGreater(mesh_volume(red.astype(np.float64)), 0)
            with zipfile.ZipFile(out / "model.3mf") as package:
                xml = package.read("3D/3dmodel.model").decode("utf-8")
                settings = package.read("Metadata/model_settings.config").decode("utf-8")
            self.assertEqual(xml.count("<object "), 4)
            self.assertEqual(xml.count("<item "), 1)
            self.assertEqual(settings.count('subtype="normal_part"'), 3)
            self.assertIn("red #ff0000", settings)
            self.assertIn("blue #0000ff", settings)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([angle["name"] for angle in manifest["angles"]], ["left", "right"])
            html = (out / "preview.html").read_text(encoding="utf-8")
            self.assertIn('id="angle-slider"', html)
            self.assertIn("view_left.png", html)
            self.assertIn("view_right.png", html)
            self.assertEqual(main(["--demo", "--width", "16", "--out", str(root / "demo")]), 0)
            self.assertTrue((root / "demo" / "preview.html").is_file())


class TestMagnets(unittest.TestCase):
    def _plate(self, **kwargs):
        left = np.zeros((2, 2), dtype=np.int32)
        right = np.ones((2, 2), dtype=np.int32)
        settings = {
            "width_mm": 40,
            "height_mm": 20,
            "base_mm": 0.8,
            "ridge_height_mm": 1.0,
            "embed_mm": 0.05,
            "seam_mm": 0,
        }
        settings.update(kwargs)
        return build_from_indices([left, right], PALETTE[:2], **settings)

    def test_box_and_round_hole_have_the_right_volume(self):
        self.assertAlmostEqual(mesh_volume(box_mesh(0, 0, 0, 10, 4, 2)), 80, places=5)
        hole = cylinder_cavity(0, 0, 3, 0.4, 2.4)
        self.assertLess(mesh_volume(hole), 0)
        self.assertAlmostEqual(mesh_volume(hole), -math.pi * 9 * 2, delta=0.8)

    def test_above_and_below_set_the_base_and_the_pause(self):
        spec = MagnetSpec(
            count=2,
            shape="round",
            thickness_mm=2,
            below_mm=1.2,
            above_mm=0.8,
            diameter_mm=6,
        )
        model = self._plate(magnets=spec)
        self.assertAlmostEqual(model.base_mm, 4.0)
        base = next(part for part in model.parts if part.role == "base")
        radius = (6 + spec.clearance_mm) / 2
        expected = 40 * 20 * 4.0 - 2 * math.pi * radius * radius * 2
        self.assertAlmostEqual(mesh_volume(base.triangles), expected, delta=abs(expected) * 0.03)
        heights = base.triangles.reshape(-1, 3)[:, 2]
        self.assertTrue(np.any(np.isclose(heights, 1.2)))
        self.assertTrue(np.any(np.isclose(heights, 3.2)))
        filament = next(part for part in model.parts if part.role == "filament")
        self.assertGreater(filament.triangles[:, :, 2].min(), 3.2)

    def test_rectangular_pockets_remove_their_exact_volume(self):
        spec = MagnetSpec(
            count=1,
            shape="rect",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.5,
            width_mm=10,
            length_mm=5,
        )
        model = self._plate(magnets=spec)
        self.assertAlmostEqual(model.base_mm, 2.9)
        base = next(part for part in model.parts if part.role == "base")
        hole_w = 10 + spec.clearance_mm
        hole_h = 5 + spec.clearance_mm
        expected = 40 * 20 * 2.9 - hole_w * hole_h * 2
        self.assertAlmostEqual(mesh_volume(base.triangles), expected, places=4)

    def test_magnets_that_do_not_fit_are_rejected(self):
        spec = MagnetSpec(count=8, shape="round", thickness_mm=2, below_mm=0.4, above_mm=0.4, diameter_mm=6)
        with self.assertRaises(ValueError):
            self._plate(width_mm=20, height_mm=10, magnets=spec)

    def test_arrangement_and_rotation_move_the_pockets(self):
        across = MagnetSpec(
            count=3,
            shape="round",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.4,
            diameter_mm=6,
            arrangement="across",
        )
        model = self._plate(width_mm=80, height_mm=30, magnets=across)
        self.assertEqual(len(model.magnet_places), 3)
        ys = [place[1] for place in model.magnet_places]
        xs = [place[0] for place in model.magnet_places]
        self.assertAlmostEqual(max(ys) - min(ys), 0, places=5)
        self.assertGreater(max(xs) - min(xs), 12)

        down = MagnetSpec(
            count=3,
            shape="round",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.4,
            diameter_mm=6,
            arrangement="down",
        )
        model = self._plate(width_mm=30, height_mm=80, magnets=down)
        xs = [place[0] for place in model.magnet_places]
        ys = [place[1] for place in model.magnet_places]
        self.assertAlmostEqual(max(xs) - min(xs), 0, places=5)
        self.assertGreater(max(ys) - min(ys), 12)

        centered = MagnetSpec(
            count=3,
            shape="round",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.4,
            diameter_mm=6,
            arrangement="center",
        )
        model = self._plate(width_mm=80, height_mm=30, magnets=centered)
        xs = sorted(place[0] for place in model.magnet_places)
        self.assertLess(xs[-1] - xs[0], 20)
        self.assertGreater(xs[0], 25)

        inset = MagnetSpec(
            count=3,
            shape="round",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.4,
            diameter_mm=6,
            arrangement="across",
            edge_mm=8,
        )
        model = self._plate(width_mm=80, height_mm=30, magnets=inset)
        xs = sorted(place[0] for place in model.magnet_places)
        self.assertAlmostEqual(xs[0], 8 + (6 + inset.clearance_mm) / 2, places=3)
        self.assertAlmostEqual(xs[-1], 80 - xs[0], places=3)

        turned = MagnetSpec(
            count=1,
            shape="rect",
            thickness_mm=2,
            below_mm=0.4,
            above_mm=0.4,
            width_mm=12,
            length_mm=4,
            turned=True,
        )
        model = self._plate(width_mm=40, height_mm=40, magnets=turned)
        base = next(part for part in model.parts if part.role == "base")
        floor = base.triangles.reshape(-1, 3)
        floor = floor[np.isclose(floor[:, 2], 0.4)]
        self.assertAlmostEqual(floor[:, 0].max() - floor[:, 0].min(), 4.4, places=3)
        self.assertAlmostEqual(floor[:, 1].max() - floor[:, 1].min(), 12.4, places=3)

    def test_zero_below_opens_the_pockets_through_the_bed(self):
        spec = MagnetSpec(
            count=2,
            shape="round",
            thickness_mm=2,
            below_mm=0,
            above_mm=0.8,
            diameter_mm=6,
            arrangement="across",
        )
        model = self._plate(width_mm=80, height_mm=40, magnets=spec)
        self.assertAlmostEqual(model.base_mm, 2.8)
        base = next(part for part in model.parts if part.role == "base")
        radius = (6 + spec.clearance_mm) / 2
        area = 32 * radius * radius * math.tan(math.pi / 32)
        expected = 80 * 40 * 2.8 - 2 * area * 2
        self.assertAlmostEqual(mesh_volume(base.triangles), expected, delta=0.05)
        for place in model.magnet_places:
            self.assertFalse(_covers(base.triangles, place[0], place[1], 0.0))
            self.assertTrue(_covers(base.triangles, place[0], place[1], spec.thickness_mm))
        from lentic.export import _print_notes

        self.assertIn("Glue the magnets in after the print", _print_notes(model, []))

    def test_zero_below_rectangular_pocket_matches_the_magnet(self):
        spec = MagnetSpec(
            count=1,
            shape="rect",
            thickness_mm=2,
            below_mm=0,
            above_mm=0.6,
            width_mm=10,
            length_mm=5,
        )
        model = self._plate(magnets=spec)
        base = next(part for part in model.parts if part.role == "base")
        hole_w = 10 + spec.clearance_mm
        hole_h = 5 + spec.clearance_mm
        expected = 40 * 20 * 2.6 - hole_w * hole_h * 2
        self.assertAlmostEqual(mesh_volume(base.triangles), expected, places=3)
        place = model.magnet_places[0]
        self.assertFalse(_covers(base.triangles, place[0], place[1], 0.0))

    def test_negative_plastic_below_the_magnets_is_rejected(self):
        spec = MagnetSpec(count=1, shape="round", thickness_mm=2, below_mm=-0.2, above_mm=0.4, diameter_mm=6)
        with self.assertRaises(ValueError):
            self._plate(magnets=spec)

    def test_horizontal_plate_keeps_the_pockets(self):
        spec = MagnetSpec(count=2, shape="rect", thickness_mm=1, below_mm=0.6, above_mm=0.7, width_mm=8, length_mm=4)
        model = self._plate(orientation="horizontal", magnets=spec)
        self.assertEqual(model.orientation, "horizontal")
        self.assertAlmostEqual(model.base_mm, 2.3)
        base = next(part for part in model.parts if part.role == "base")
        hole = (8 + spec.clearance_mm) * (4 + spec.clearance_mm) * 1
        self.assertAlmostEqual(mesh_volume(base.triangles), 40 * 20 * 2.3 - 2 * hole, places=3)


def _covers(triangles: np.ndarray, x: float, y: float, z: float) -> bool:
    layer = triangles[np.all(np.abs(triangles[:, :, 2] - z) < 1e-6, axis=1)]
    for corner_a, corner_b, corner_c in layer:
        if _inside_triangle(x, y, corner_a, corner_b, corner_c):
            return True
    return False


def _inside_triangle(px: float, py: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0x, v0y = c[0] - a[0], c[1] - a[1]
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = px - a[0], py - a[1]
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    along_c = (dot11 * dot02 - dot01 * dot12) / denom
    along_b = (dot00 * dot12 - dot01 * dot02) / denom
    return along_c >= -1e-8 and along_b >= -1e-8 and along_c + along_b <= 1 + 1e-8


if __name__ == "__main__":
    unittest.main()
