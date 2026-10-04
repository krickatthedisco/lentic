"""Write STL parts, a 3MF, previews, and a small angle viewer."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from lentic.build import LenticModel, MeshPart
from lentic.color import rgb_to_hex
from lentic.mesh import triangle_normals


def export_model(model: LenticModel, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[tuple[MeshPart, str]] = []
    for part in model.parts:
        if len(part.triangles) == 0:
            continue
        name = _filename(part, {item[1] for item in written})
        write_stl(out / name, part.triangles)
        written.append((part, name))
    if not written:
        raise ValueError("nothing to export")

    view_files = []
    for name, preview in zip(model.angle_names, model.previews):
        filename = f"view_{name}.png"
        _save_preview(out / filename, preview)
        view_files.append(filename)
    for name, fitted in zip(model.angle_names, model.fitted):
        _save_preview(out / f"fitted_{name}.png", fitted, min_width=360)
    top_name = "view_top.png"
    if model.top_preview is not None:
        _save_preview(out / top_name, model.top_preview)

    write_3mf(out / "model.3mf", written)
    manifest = _manifest(model, written, view_files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "PRINT.txt").write_text(_print_notes(model, written), encoding="utf-8")
    html_path = out / "preview.html"
    html_path.write_text(_html(model, written, view_files, top_name), encoding="utf-8")
    return html_path


def multipart_stl_bytes(model: LenticModel) -> bytes:
    """One ASCII STL, with a named solid for each filament and the base."""
    used: set[str] = set()
    chunks: list[str] = []
    for part in model.parts:
        if len(part.triangles) == 0:
            continue
        stem = "base" if part.role == "base" else "filament"
        name = f"{stem}_{rgb_to_hex(part.rgb)}"
        if name in used:
            suffix = 2
            while f"{name}_{suffix}" in used:
                suffix += 1
            name = f"{name}_{suffix}"
        used.add(name)
        chunks.append(_ascii_solid(name, part.triangles))
    if not chunks:
        raise ValueError("nothing to export")
    return "".join(chunks).encode("ascii")


def _ascii_solid(name: str, triangles: np.ndarray) -> str:
    triangles = np.asarray(triangles, dtype=np.float64)
    normals = triangle_normals(triangles)
    lines = [f"solid {name}\n"]
    for index in range(len(triangles)):
        normal = normals[index]
        v0, v1, v2 = triangles[index]
        lines.append(
            f" facet normal {normal[0]:.5f} {normal[1]:.5f} {normal[2]:.5f}\n"
            "  outer loop\n"
            f"   vertex {v0[0]:.5f} {v0[1]:.5f} {v0[2]:.5f}\n"
            f"   vertex {v1[0]:.5f} {v1[1]:.5f} {v1[2]:.5f}\n"
            f"   vertex {v2[0]:.5f} {v2[1]:.5f} {v2[2]:.5f}\n"
            "  endloop\n"
            " endfacet\n"
        )
    lines.append(f"endsolid {name}\n")
    return "".join(lines)


def write_stl(path: Path, triangles: np.ndarray) -> None:
    triangles = np.ascontiguousarray(triangles, dtype="<f4")
    count = int(triangles.shape[0])
    normals = np.ascontiguousarray(triangle_normals(triangles), dtype="<f4")
    raw = np.zeros((count, 50), dtype=np.uint8)
    raw[:, 0:12] = normals.view(np.uint8).reshape(count, 12)
    raw[:, 12:48] = triangles.reshape(count, 9).view(np.uint8).reshape(count, 36)
    with Path(path).open("wb") as handle:
        handle.write(b"lentic color lenticular".ljust(80, b"\0"))
        handle.write(struct.pack("<I", count))
        handle.write(raw.tobytes())


def write_3mf(path: Path, parts: list[tuple[MeshPart, str]]) -> None:
    objects = []
    items = []
    for object_id, (part, filename) in enumerate(parts, start=1):
        vertices, faces = _index_triangles(part.triangles)
        if len(faces) == 0:
            continue
        objects.append(_mesh_xml(object_id, f"{filename} #{rgb_to_hex(part.rgb)}", vertices, faces))
        items.append(f'    <item objectid="{object_id}"/>')
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        '  <metadata name="Application">lentic</metadata>\n'
        '  <metadata name="Title">Color lenticular plate</metadata>\n'
        "  <resources>\n"
        + "\n".join(objects)
        + "\n  </resources>\n  <build>\n"
        + "\n".join(items)
        + "\n  </build>\n</model>\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
            "</Types>\n",
        )
        package.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            "</Relationships>\n",
        )
        package.writestr("3D/3dmodel.model", model)


def _index_triangles(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(triangles, dtype=np.float64).reshape(-1, 3)
    unique, inverse = np.unique(np.round(flat, 5), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return unique, faces[valid]


def _mesh_xml(object_id: int, name: str, vertices: np.ndarray, faces: np.ndarray) -> str:
    verts = "\n".join(
        f'          <vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in vertices
    )
    tris = "\n".join(
        f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces
    )
    return (
        f'    <object id="{object_id}" type="model" name="{_xml(name)}">\n'
        "      <mesh>\n        <vertices>\n"
        f"{verts}\n        </vertices>\n        <triangles>\n"
        f"{tris}\n        </triangles>\n      </mesh>\n    </object>"
    )


def _filename(part: MeshPart, used: set[str]) -> str:
    stem = "base" if part.role == "base" else "filament"
    name = f"{stem}_{rgb_to_hex(part.rgb)}.stl"
    if name not in used:
        return name
    suffix = 2
    while f"{stem}_{rgb_to_hex(part.rgb)}_{suffix}.stl" in used:
        suffix += 1
    return f"{stem}_{rgb_to_hex(part.rgb)}_{suffix}.stl"


def _save_preview(path: Path, rgb: np.ndarray, min_width: int = 480) -> None:
    height, width = rgb.shape[:2]
    scale = max(1, int(np.ceil(min_width / max(width, 1))))
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    if scale > 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    image.save(path)


def _manifest(model: LenticModel, parts: list[tuple[MeshPart, str]], view_files: list[str]) -> dict:
    grid = model.grid
    return {
        "width_mm": round(grid.width_mm, 4),
        "height_mm": round(grid.height_mm, 4),
        "depth_mm": round(model.base_mm + grid.ridge_height_mm, 4),
        "base_mm": round(model.base_mm, 4),
        "ridge_height_mm": round(grid.ridge_height_mm, 4),
        "pitch_mm": round(grid.pitch_mm, 4),
        "row_mm": round(grid.row_mm, 4),
        "ridges": grid.n_ridges,
        "rows": grid.n_rows,
        "angles": [
            {"name": name, "preview": filename}
            for name, filename in zip(model.angle_names, view_files)
        ],
        "parts": [
            {
                "file": filename,
                "role": part.role,
                "hex": f"#{rgb_to_hex(part.rgb)}",
                "triangles": int(len(part.triangles)),
            }
            for part, filename in parts
        ],
    }


def _print_notes(model: LenticModel, parts: list[tuple[MeshPart, str]]) -> str:
    grid = model.grid
    lines = [
        "Lentic color plate",
        f"Size: {grid.width_mm:.2f} x {grid.height_mm:.2f} x {model.base_mm + grid.ridge_height_mm:.2f} mm",
        f"Ridges: {grid.n_ridges} at {grid.pitch_mm:.2f} mm, {grid.ridge_height_mm:.2f} mm tall",
        "",
        "Import model.3mf, or import every STL together as one object with multiple parts.",
        "Assign each file the filament closest to its color:",
        "",
    ]
    for part, filename in parts:
        lines.append(f"  {filename}  #{rgb_to_hex(part.rgb)}")
    lines.extend(
        [
            "",
            "Lay the plate flat with the ridges facing up. Do not stand it on its side.",
            _viewing_note(model),
            "Tilt the finished print to switch pictures.",
            "",
        ]
    )
    return "\n".join(lines)


def _viewing_note(model: LenticModel) -> str:
    if model.orientation == "horizontal":
        return "The bottom picture is on the slopes that face down. The top picture faces up."
    return "The first image is on the slopes that face left. The second faces right."


def _html(
    model: LenticModel,
    parts: list[tuple[MeshPart, str]],
    view_files: list[str],
    top_name: str,
) -> str:
    grid = model.grid
    images = []
    for index, (name, filename) in enumerate(zip(model.angle_names, view_files)):
        klass = "angle" if index == 0 else "angle overlay"
        images.append(
            f'<img class="{klass}" id="angle-{index}" data-name="{_xml(name)}" '
            f'alt="View from the { _xml(name) }" src="{_xml(filename)}">'
        )
    swatches = []
    for part, filename in parts:
        hex_color = rgb_to_hex(part.rgb)
        swatches.append(
            f'<li><span class="swatch" style="background:#{hex_color}"></span>'
            f"<code>{_xml(filename)}</code> #{hex_color}</li>"
        )
    names = ",".join(json.dumps(name) for name in model.angle_names)
    depth = model.base_mm + grid.ridge_height_mm
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lentic preview</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; font: 16px/1.45 "Segoe UI", sans-serif; background: #f4f0e8; color: #1c1917; }}
  main {{ max-width: 760px; margin: 0 auto; padding: 28px 20px 48px; }}
  h1 {{ font-size: 28px; margin: 0 0 8px; }}
  p {{ margin: 0 0 12px; }}
  .stage {{ position: relative; max-width: 640px; background: #fff; border: 1px solid #d6d3d1; }}
  .stage img {{ width: 100%; height: auto; display: block; image-rendering: pixelated; }}
  .overlay {{ position: absolute; left: 0; top: 0; opacity: 0; }}
  label {{ display: block; margin: 16px 0 6px; font-weight: 600; }}
  input[type="range"] {{ width: min(640px, 100%); accent-color: #9a3412; }}
  #angle-label {{ font-size: 18px; text-transform: capitalize; }}
  ul {{ padding-left: 0; list-style: none; }}
  li {{ display: flex; gap: 8px; align-items: center; margin: 6px 0; }}
  .swatch {{ width: 18px; height: 18px; border: 1px solid #44403c; display: inline-block; }}
  figure {{ margin: 18px 0; }}
  figcaption {{ font-size: 14px; color: #44403c; margin-bottom: 6px; }}
  img.wide {{ max-width: 640px; width: 100%; height: auto; image-rendering: pixelated; border: 1px solid #d6d3d1; }}
</style>
</head>
<body>
<main>
  <h1>Lentic preview</h1>
  <p>{grid.width_mm:.1f} x {grid.height_mm:.1f} x {depth:.1f} mm.
  Drag the slider to switch viewing angle. This is the ideal view of each slope, before the plastic softens the edges.</p>
  <div class="stage" id="stage">
    {''.join(images)}
  </div>
  <label for="angle-slider">Viewing angle</label>
  <input id="angle-slider" type="range" min="0" max="100" value="0">
  <p id="angle-label">{_xml(model.angle_names[0])}</p>
  <figure>
    <figcaption>Straight down, where both slopes are visible as stripes.</figcaption>
    <img class="wide" src="{top_name}" alt="Top view with both pictures striped together">
  </figure>
  <h2>Filaments</h2>
  <ul>
    {''.join(swatches)}
  </ul>
  <p>Import <code>model.3mf</code> and match each part to the swatch. Keep the ridges facing up.</p>
</main>
<script>
const names = [{names}];
const images = [...document.querySelectorAll(".angle")];
const slider = document.getElementById("angle-slider");
const label = document.getElementById("angle-label");
function show(value) {{
  const pos = (Number(value) / 100) * (images.length - 1);
  images.forEach((img, index) => {{
    img.style.opacity = String(Math.max(0, 1 - Math.abs(pos - index)));
  }});
  label.textContent = names[Math.round(pos)];
}}
slider.addEventListener("input", () => show(slider.value));
show(0);
</script>
</body>
</html>
"""


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
