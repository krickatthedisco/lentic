# Lentic

Lentic turns two or three pictures into a 3D-printable plate that flips between them as you tilt it. Each picture is matched to filament colors, then built as its own solid so a multi-material slicer (Bambu AMS, Prusa MMU, OrcaSlicer, PrusaSlicer) can print the color.

The design window runs in your browser, on your computer. Pictures are not uploaded anywhere.

This is an independent project. It is not LuBan and it does not include LuBan's code.

```
cross section of one ridge

image A          image B
(left slope)     (right slope)
      peak
     /    \
    /      \
   /        \
  +----------+  base
```

From the left you see the first image. From the right you see the second. **Top and bottom** turns the ridges sideways, so tipping the plate up or down switches the pictures (a Clean / Dirty dishwasher magnet is one use case). With three images, the flat top shows the middle one when you look straight on. Straight down, both slopes show up as stripes.

## Install

You need Python 3.10 or newer.

```powershell
cd lentic
python -m pip install -e .
```

Or, without installing, run `python -m lentic` from this folder.

## Design window

```powershell
lentic --gui
```

That opens a page on `127.0.0.1`. Upload a picture for each view. Every size is in millimeters.

- **Left and right** or **Top and bottom** chooses which way the pictures switch.
- **Maintain aspect ratio** keeps the plate the same shape as the first picture. Editing width or height updates the other.
- Drag the frame on a picture to crop it. The frame matches the plate, so the rest of the photo is not silently cut off. Corner handles resize it. **Reset crop** uses the whole picture again.
- Ridge pitch and row size are the resolution. A larger plate keeps the same pitch, so it holds more of the picture. A 0.4 mm nozzle starts at a 0.8 mm pitch and 0.4 mm rows.
- The view beside the controls is the actual plate. Drag to rotate, scroll to zoom. Each swatch is a filament, named from its color and labeled with the hex code.

**Export multi-part STL** downloads one ASCII STL. Each filament, and the base, is its own `solid` named with its hex color, already assembled in place. In a Prusa-family slicer, use Split to parts so the colors stay aligned.

## Make a plate from the command line

```powershell
lentic left.png right.png --width 120 --out plate
```

`--width` is millimeters. Height follows the first image unless you pass `--height`. The first file is the left view, the second is the right view, and an optional third file is the front view. Add `--horizontal` to switch top and bottom instead. In that mode the first file is the top view and the second is the bottom view.

Colors are chosen automatically (four filaments) and dithered. Small distinct areas, such as a patch of blue, are kept when the filament count allows. For flat graphics, name the filaments yourself and turn dithering off:

```powershell
lentic left.png right.png --width 80 --palette #111111,#f5f5f5,#e63946 --no-dither --out plate
```

A small sample (red circle on yellow, cyan diamond on navy) is built with:

```powershell
lentic --demo --out demo_out
```

Open `preview.html` in that folder and drag the slider. That is the ideal view of each slope, which is sharper than the plastic will be.

## What gets written

| File | Role |
| --- | --- |
| `model.3mf` | All parts in one file, millimeters |
| `base_*.stl` | Backing plate |
| `filament_*.stl` | One mesh per filament color. The name is the hex color |
| `view_left.png`, `view_right.png` | Picture you should see from that side, after filament matching |
| `view_top.png` | Both pictures striped, which is the straight-down view |
| `PRINT.txt` | Slicer checklist |
| `preview.html` | Angle slider |

## Print

1. Import `model.3mf`, or import every STL at once and load them as a single object with multiple parts.
2. Assign each part the filament closest to the color in its filename.
3. Leave the plate flat, ridges up. Standing it on its side ruins the flip.
4. Slice. A 0.4 mm nozzle can print one line on each slope, so two pictures default to a 0.8 mm ridge and 0.4 mm rows. If a slope comes out too thin, raise `--pitch` or `--lines` (1.6 mm is two lines on each slope). If the flip is weak, raise `--ridge-height`.

Photos are posterized down to the filament count. Simple graphics with a few flat colors stay much cleaner. `--max-colors` can go up to the number of filaments you actually have.

Ridge pitch defaults to `nozzle × lines × number of images` (0.4 × 1 × 2 = 0.8 mm). Rows default to the nozzle width, 0.4 mm. Pass `--pitch` or `--row` to override them.

## License

MIT. See [LICENSE](LICENSE).
