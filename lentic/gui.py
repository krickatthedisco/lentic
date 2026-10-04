"""Local design window for a color lenticular plate."""

from __future__ import annotations

import json
import struct
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from lentic.build import LenticModel, build_from_images
from lentic.color import color_name, parse_hex, rgb_to_hex
from lentic.export import plate_3mf_bytes

WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_BODY = 32 * 1024 * 1024
_SAMPLES: tuple[bytes, bytes] | None = None


def launch(port: int = 8765, open_browser: bool = True) -> None:
    try:
        server = make_server("127.0.0.1", port)
    except OSError:
        server = make_server("127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Lentic is open at {url}")
    print("Close this window with Ctrl+C.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def model_from_form(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> LenticModel:
    orientation = _orientation(fields)
    if orientation == "horizontal":
        if "top" not in files or "bottom" not in files:
            raise ValueError("upload a top picture and a bottom picture")
        order = ["bottom", "front", "top"] if "front" in files else ["bottom", "top"]
    else:
        if "left" not in files or "right" not in files:
            raise ValueError("upload a left picture and a right picture")
        order = ["left", "front", "right"] if "front" in files else ["left", "right"]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        paths = []
        for key in order:
            filename, data = files[key]
            suffix = Path(filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                suffix = ".png"
            path = root / f"{key}{suffix}"
            path.write_bytes(data)
            paths.append(path)
        try:
            return build_from_images(
                paths,
                width_mm=_millimeters(fields, "width"),
                height_mm=_millimeters(fields, "height"),
                pitch_mm=_millimeters(fields, "pitch"),
                row_mm=_millimeters(fields, "row"),
                ridge_height_mm=_millimeters(fields, "ridge_height"),
                base_mm=_millimeters(fields, "base"),
                palette=_palette(fields),
                max_colors=_max_colors(fields),
                dither=_flag(fields, "dither", default=True),
                base_rgb=parse_hex(fields.get("base_color") or "#f4f1ea"),
                orientation=orientation,
                crops=_crops(fields, order),
                flips=_flips(fields, order),
            )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("one of the pictures could not be read") from exc


def pack_preview(model: LenticModel) -> bytes:
    """JSON header, padded to 4 bytes, then float32 xyz in a Y-up frame."""
    width = model.grid.width_mm
    height = model.grid.height_mm
    blobs: list[bytes] = []
    parts: list[dict] = []
    cursor = 0
    for part in model.parts:
        if len(part.triangles) == 0:
            continue
        points = np.asarray(part.triangles, dtype=np.float64).reshape(-1, 3)
        placed = np.empty_like(points)
        placed[:, 0] = points[:, 0] - width / 2.0
        placed[:, 1] = points[:, 2]
        placed[:, 2] = -(points[:, 1] - height / 2.0)
        flat = np.ascontiguousarray(placed, dtype="<f4").reshape(-1)
        parts.append(
            {
                "role": part.role,
                "name": color_name(part.rgb),
                "hex": f"#{rgb_to_hex(part.rgb)}",
                "offset": cursor,
                "count": int(flat.shape[0]),
            }
        )
        blobs.append(flat.tobytes())
        cursor += int(flat.shape[0])
    meta = json.dumps(
        {
            "width_mm": round(width, 4),
            "height_mm": round(height, 4),
            "depth_mm": round(model.base_mm + model.grid.ridge_height_mm, 4),
            "pitch_mm": round(model.grid.pitch_mm, 4),
            "row_mm": round(model.grid.row_mm, 4),
            "ridge_height_mm": round(model.grid.ridge_height_mm, 4),
            "ridges": model.grid.n_ridges,
            "rows": model.grid.n_rows,
            "angles": list(model.angle_names),
            "orientation": model.orientation,
            "parts": parts,
        }
    ).encode("utf-8")
    pad = (4 - (len(meta) % 4)) % 4
    return struct.pack("<I", len(meta)) + meta + (b" " * pad) + b"".join(blobs)


def parse_form(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    if "boundary=" not in content_type:
        raise ValueError("expected a multipart form")
    raw_boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    marker = b"--" + raw_boundary.encode("ascii", "replace")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for raw in body.split(marker):
        if not raw or raw.startswith(b"--"):
            continue
        if raw.startswith(b"\r\n"):
            raw = raw[2:]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        header_blob, separator, data = raw.partition(b"\r\n\r\n")
        if not separator:
            continue
        disposition = ""
        for line in header_blob.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition"):
                disposition = line
        name = _disposition(disposition, "name")
        filename = _disposition(disposition, "filename")
        if not name:
            continue
        if filename:
            files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return fields, files


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/sample/left.png", "/sample/right.png"):
            left, right = _sample_pngs()
            payload = left if path.endswith("left.png") else right
            self._send(200, payload, "image/png")
            return
        file_path = _safe_file(path)
        if file_path is None:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        kind = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".png": "image/png",
        }.get(file_path.suffix.lower(), "application/octet-stream")
        self._send(200, file_path.read_bytes(), kind)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b'{"error":"bad length"}', "application/json")
            return
        if length < 0 or length > MAX_BODY:
            self._send(400, b'{"error":"upload is too large"}', "application/json")
            return
        body = self.rfile.read(length)
        try:
            fields, files = parse_form(self.headers.get("Content-Type", ""), body)
            model = model_from_form(fields, files)
            if path == "/api/preview":
                self._send(200, pack_preview(model), "application/octet-stream")
                return
            if path == "/api/export":
                self._send(
                    200,
                    plate_3mf_bytes(model),
                    "model/3mf",
                    filename="lentic-plate.3mf",
                )
                return
            self._send(404, b'{"error":"unknown request"}', "application/json")
        except ValueError as exc:
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, payload, "application/json")

    def _send(self, status: int, body: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)


def _millimeters(fields: dict[str, str], key: str) -> float:
    try:
        value = float(fields[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{key.replace('_', ' ')} must be a number of millimeters") from None
    if value <= 0:
        raise ValueError(f"{key.replace('_', ' ')} must be greater than 0")
    return value


def _crops(fields: dict[str, str], order: list[str]) -> list[tuple[float, float, float, float] | None]:
    crops = []
    for key in order:
        text = (fields.get(f"crop_{key}") or "").strip()
        if not text:
            crops.append(None)
            continue
        try:
            parts = tuple(float(piece) for piece in text.split(","))
        except ValueError:
            raise ValueError(f"the {key} crop must be four numbers") from None
        if len(parts) != 4:
            raise ValueError(f"the {key} crop must be four numbers")
        crops.append(parts)
    return crops


def _flips(fields: dict[str, str], order: list[str]) -> list[tuple[bool, bool]]:
    return [(_flag(fields, f"flip_h_{key}", default=False), _flag(fields, f"flip_v_{key}", default=False)) for key in order]


def _orientation(fields: dict[str, str]) -> str:
    value = (fields.get("orientation") or "vertical").strip().lower()
    if value in {"horizontal", "top", "sideways"}:
        return "horizontal"
    if value in {"vertical", "side", "left", ""}:
        return "vertical"
    raise ValueError("orientation must be vertical or horizontal")


def _palette(fields: dict[str, str]):
    text = (fields.get("palette") or "").strip()
    if not text:
        return None
    colors = [parse_hex(item) for item in text.split(",") if item.strip()]
    if not colors:
        return None
    return np.array(colors, dtype=np.uint8)


def _max_colors(fields: dict[str, str]) -> int:
    try:
        value = int(fields.get("max_colors") or "4")
    except ValueError:
        raise ValueError("filament count must be a whole number") from None
    if value < 1 or value > 8:
        raise ValueError("filament count must be from 1 to 8")
    return value


def _flag(fields: dict[str, str], key: str, default: bool) -> bool:
    if key not in fields:
        return default
    return fields[key].strip().lower() not in {"0", "false", "off", "no"}


def _disposition(header: str, key: str) -> str:
    token = key + '="'
    start = header.find(token)
    if start >= 0:
        start += len(token)
        end = header.find('"', start)
        return header[start:end]
    token = key + "="
    start = header.find(token)
    if start < 0:
        return ""
    start += len(token)
    end = header.find(";", start)
    if end < 0:
        end = len(header)
    return header[start:end].strip().strip('"')


def _safe_file(url_path: str) -> Path | None:
    if url_path in ("/", "/index.html"):
        return WEB_ROOT / "index.html"
    relative = url_path.lstrip("/")
    if not relative or ".." in relative.split("/"):
        return None
    candidate = (WEB_ROOT / relative).resolve()
    if WEB_ROOT.resolve() not in candidate.parents:
        return None
    if candidate.is_file():
        return candidate
    return None


def _sample_pngs() -> tuple[bytes, bytes]:
    global _SAMPLES
    if _SAMPLES is None:
        from io import BytesIO

        from lentic.demo import _pictures

        left, right = _pictures((320, 180))
        _SAMPLES = (_png(left), _png(right))
    return _SAMPLES


def _png(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
