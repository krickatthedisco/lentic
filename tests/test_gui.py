"""Design window requests and the multi-part STL export."""

from __future__ import annotations

import io
import json
import struct
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from lentic.export import multipart_stl_bytes
from lentic.gui import make_server, parse_form


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[str, bytes]:
    boundary = "----lenticTestBoundary7f3a"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    for key, (filename, data) in files.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\n"
                "Content-Type: image/png\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(chunks)


def _plate_fields() -> dict[str, str]:
    return {
        "width": "8",
        "height": "8",
        "base": "0.8",
        "ridge_height": "1.2",
        "pitch": "4",
        "row": "4",
        "max_colors": "4",
        "dither": "0",
        "base_color": "#f4f1ea",
    }


class TestMultipartStl(unittest.TestCase):
    def test_each_filament_is_its_own_solid(self):
        from lentic.build import build_from_indices
        import numpy as np

        left = np.zeros((1, 2), dtype=np.int32)
        right = np.ones((1, 2), dtype=np.int32)
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        model = build_from_indices(
            [left, right],
            palette,
            width_mm=4,
            height_mm=2,
            base_mm=0.8,
            ridge_height_mm=1.2,
            embed_mm=0,
            seam_mm=0,
        )
        text = multipart_stl_bytes(model).decode("ascii")
        solids = [line.split()[1] for line in text.splitlines() if line.startswith("solid ")]
        self.assertEqual(solids, ["base_000000", "filament_ff0000", "filament_0000ff"])
        self.assertEqual(text.count("endsolid "), 3)
        self.assertEqual(text.count("endfacet"), sum(len(part.triangles) for part in model.parts))
        self.assertTrue(text.startswith("solid base_000000\n"))


class TestGuiServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = make_server("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_page_has_upload_and_millimeter_controls(self):
        html = urllib.request.urlopen(self.base + "/").read().decode("utf-8")
        self.assertIn("Upload left picture", html)
        self.assertIn("Upload right picture", html)
        self.assertIn("Upload front picture (Optional)", html)
        self.assertIn('aria-label="Remove left picture"', html)
        self.assertIn('aria-label="Remove front picture"', html)
        self.assertIn("> mm<", html)
        self.assertIn("Export for slicer", html)
        self.assertIn("Maintain aspect ratio", html)
        self.assertIn("Drag each frame", html)
        self.assertIn("Flip horizontal", html)
        self.assertIn("Flip vertical", html)
        self.assertIn("Use this text", html)
        self.assertIn("Use my filament colors", html)
        self.assertIn("Add magnet pockets", html)
        self.assertIn("Add a crest line", html)
        self.assertIn("introducing vertical banding across the image", html)
        self.assertIn('id="crest-width" type="number" min="0.1" step="0.1" value="0.4"', html)
        self.assertIn('id="crest-height" type="number" min="0.04" step="0.01" value="0.2"', html)
        self.assertIn('id="layer-height" type="number" min="0.04" step="0.01" value="0.2"', html)
        self.assertIn('id="initial-layer" type="number" min="0.04" step="0.01" value="0.2"', html)
        self.assertIn("Use the same layer height and first layer height as the slicer.", html)
        self.assertIn('id="crest-color" type="color" value="#000000"', html)
        self.assertIn("View from Left", html)
        self.assertIn("View from Right", html)
        self.assertIn("Thickness below magnets", html)
        self.assertIn("glue the magnets in from the back after the print is complete", html)
        self.assertIn("Thickness above magnets", html)
        self.assertIn("Show magnet locations", html)
        self.assertIn("Along the width", html)
        self.assertIn("Centered", html)
        self.assertIn("Distance from edge", html)
        self.assertIn("Rotate rectangular magnets", html)
        self.assertIn("Building the plate", html)
        self.assertIn("Top and bottom", html)
        self.assertIn('id="nozzle"', html)
        self.assertIn('value="0.2"', html)
        self.assertIn('value="0.6"', html)
        self.assertIn('value="0.4" selected', html)
        self.assertIn(
            "Selecting your nozzle size automatically sets the default values to match, further adjustments may still improve image quality so play around with the values",
            html,
        )
        self.assertIn('id="base-color" type="color" value="#000000"', html)
        self.assertIn("This plate uses a 4 mm pitch, a 3.2 mm ridge, and 0.4 mm rows.", html)
        self.assertIn("Each slope is 5 lines of the 0.4 mm nozzle.", html)
        self.assertIn("The tip is cut flat 0.5 mm in from each side.", html)
        self.assertIn("Layers are 0.2 mm after a 0.2 mm first layer.", html)
        self.assertIn("The crest line, if you turn it on, is 0.4 mm wide and 0.2 mm tall.", html)
        self.assertIn('id="pitch" type="number" min="0.2" step="0.1" value="4"', html)
        self.assertIn('id="ridge" type="number" min="0.2" step="0.1" value="3.2"', html)
        self.assertIn('id="width" type="number" min="5" step="0.1" value="200"', html)
        self.assertIn('id="row" type="number" min="0.2" step="0.1" value="0.4"', html)
        self.assertIn('id="view"', html)
        font = urllib.request.urlopen(self.base + "/fonts/Oswald.ttf")
        self.assertEqual(font.status, 200)
        self.assertIn("font", font.headers.get("Content-Type", ""))
        self.assertGreater(len(font.read(4)), 0)

    def test_preview_and_export(self):
        boundary, body = _multipart(
            _plate_fields(),
            {"left": ("left.png", _png((255, 0, 0))), "right": ("right.png", _png((0, 0, 255)))},
        )
        fields, files = parse_form(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(fields["width"], "8")
        self.assertEqual(set(files), {"left", "right"})

        preview = self._post("/api/preview", boundary, body)
        self.assertEqual(preview.status, 200)
        payload = preview.read()
        meta_length = struct.unpack_from("<I", payload, 0)[0]
        meta = json.loads(payload[4 : 4 + meta_length])
        colors = {part["hex"] for part in meta["parts"]}
        self.assertIn("#ff0000", colors)
        self.assertIn("#0000ff", colors)
        self.assertGreater(meta["parts"][0]["count"], 0)

        exported = self._post("/api/export", boundary, body)
        self.assertEqual(exported.status, 200)
        self.assertIn("lentic-plate.3mf", exported.headers.get("Content-Disposition", ""))
        package = zipfile.ZipFile(io.BytesIO(exported.read()))
        model_xml = package.read("3D/3dmodel.model").decode("utf-8")
        settings = package.read("Metadata/model_settings.config").decode("utf-8")
        self.assertEqual(model_xml.count("<item "), 1)
        self.assertEqual(settings.count('subtype="normal_part"'), 3)
        self.assertIn("red #ff0000", settings)
        self.assertIn("blue #0000ff", settings)

    def _post(self, path: str, boundary: str, body: bytes):
        request = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return urllib.request.urlopen(request)


if __name__ == "__main__":
    unittest.main()
