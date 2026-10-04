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

## Run it

### Windows, without installing Python

[![Download Lentic.exe](https://img.shields.io/badge/Download-Lentic.exe-9a3412?style=for-the-badge)](https://github.com/krickatthedisco/lentic/releases/latest/download/Lentic.exe)

Double-click `Lentic.exe`. Leave that window open while you design, and close it when you are done. The page stays on your computer.

The same release includes [lentic-source.zip](https://github.com/krickatthedisco/lentic/releases/latest/download/lentic-source.zip) if you want the source.

macOS and Linux, and anyone who wants to change the program, use the Python steps below. It is the same program either way.

Install [Python 3.10 or newer](https://www.python.org/downloads/). On Windows, check **Add python.exe to PATH** on the first installer screen.

Download this repository with the green **Code** button, then **Download ZIP**, and unzip it. Or, if you use Git:

```bat
git clone https://github.com/krickatthedisco/lentic.git
cd lentic
```

On Windows, double-click `start.bat` in that folder. It installs what Lentic needs and opens the design window. Leave the black window open while you work. Close it with Ctrl+C.

On macOS or Linux, from that same folder:

```sh
chmod +x start.sh
./start.sh
```

The page stays on your computer, at `127.0.0.1`. Upload a picture for each view. Every size is in millimeters.

- **Left and right** or **Top and bottom** chooses which way the pictures switch.
- Each picture can be words instead of a photo. Type the text, pick an open-source font, choose the letter and background colors, then **Use this text**.
- **Use my filament colors** prints with the spools you pick. Add or remove colors, up to eight. Leave it off and Lentic chooses the colors from the pictures.
- **Add magnet pockets** cuts holes in the base. Pick how many, then a round magnet (diameter and thickness) or a rectangular one (width, length, and thickness). **Thickness below magnets** and **Thickness above magnets** set the plastic on each side, and the base thickness becomes the sum. **Arrangement** lines them up across the width, down the height, in a grid, or centered in a cluster. **Distance from edge** is the plastic between the outer holes and the rim. **Rotate rectangular magnets** turns each one. **Show magnet locations** draws the pockets through the plate. Pause when the pocket is full, drop the magnets in, and let the print cover them. Set **Thickness below magnets** to 0 to leave the pockets open, then glue the magnets in from the back after the print.
- **Maintain aspect ratio** keeps the plate the same shape as the first picture. Editing width or height updates the other.
- Drag the frame on a picture to crop it. The frame matches the plate, so the rest of the photo is not silently cut off. Corner handles resize it. **Reset crop** uses the whole picture again.
- The base starts black. Change **Base color** if the backing should be another filament.
- **Nozzle** is 0.2, 0.3, 0.4, 0.5, or 0.6 mm. Picking one fills in the pitch, row size, ridge height, and crest line for that nozzle. Five lines stay on each slope, each row is one nozzle wide, and the ridge is 0.8 times the pitch. A 0.4 mm nozzle starts at a 4 mm pitch, a 0.4 mm row, and a 3.2 mm ridge. The tip is cut flat 0.5 mm in from each side, and that cut is 1.25 times the nozzle width. Change any of those afterward if you want.
- **Add a crest line** lays one filament along the top of each ridge. It is off until you turn it on. It starts black, 0.4 mm wide and 0.2 mm tall, and both the size and the color can change. It prints as its own filament and hides the other picture when you tilt the plate.
- The view beside the controls is the actual plate. Drag to rotate, scroll to zoom. **View from Left** and **View from Right** (or top and bottom) turn it 45 degrees so you can check one picture. Each swatch is a filament, named from its color and labeled with the hex code.

**Export for slicer** downloads `lentic-plate.3mf`. It opens as one plate with one part per color, already assembled. Match each part to the filament named on it, then slice. Do not split the plate into shells. An STL cannot keep those colors as separate parts: a slicer breaks it into one piece per ridge.

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
| `model.3mf` | One assembled plate, one part per color |
| `base_*.stl` | Backing plate |
| `filament_*.stl` | One mesh per filament color. The name is the hex color |
| `view_left.png`, `view_right.png` | Picture you should see from that side, after filament matching |
| `view_top.png` | Both pictures striped, which is the straight-down view |
| `PRINT.txt` | Slicer checklist |
| `preview.html` | Angle slider |

## Print

1. Open `model.3mf`. It is one plate, already split into one part per color.
2. Match each part to the filament named on it. Do not split the plate into shells.
3. Leave the plate flat, ridges up. Standing it on its side ruins the flip.
4. Slice with the nozzle you picked. A 0.4 mm nozzle starts at 0.2 mm layers, a 4 mm pitch, five lines on each slope, and a 3.2 mm ridge, so a side view shows one picture. Rows are one nozzle wide. A smaller nozzle uses the finer pitch and row from the nozzle menu. If the flip is weak, raise the ridge height.

Photos are posterized down to the filament count. Simple graphics with a few flat colors stay much cleaner. `--max-colors` can go up to the number of filaments you actually have.

Ridge pitch defaults to `nozzle × lines × number of images` (0.4 × 5 × 2 = 4.0 mm). Rows default to the nozzle width, 0.4 mm. Pass `--pitch`, `--lines`, or `--row` to override them.

## License

MIT. See [LICENSE](LICENSE).
