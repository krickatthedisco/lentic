"""Geometry, color matching, and file export for lentic plates."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from lentic.build import _open_rgb, build_from_images, build_from_indices
from lentic.cli import main
from lentic.color import choose_palette, floyd_steinberg, median_cut, nearest_indices, parse_hex
from lentic.export import export_model
from lentic.mesh import mesh_volume, triangle_normals


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
            self.assertEqual(len(red), 16)
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


if __name__ == "__main__":
    unittest.main()
