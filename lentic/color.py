"""Match pictures to a small set of filament colors."""

from __future__ import annotations

import numpy as np

_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def parse_hex(text: str) -> tuple[int, int, int]:
    raw = text.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raise ValueError(f"not a hex color: {text!r}")
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int] | np.ndarray) -> str:
    return "{:02x}{:02x}{:02x}".format(*(int(channel) for channel in rgb))


def color_name(rgb: tuple[int, int, int] | np.ndarray) -> str:
    """Plain name such as 'dark blue' or 'light red' for a filament swatch."""
    red, green, blue = (int(channel) / 255.0 for channel in rgb)
    peak = max(red, green, blue)
    floor = min(red, green, blue)
    lightness = (peak + floor) / 2.0
    chroma = peak - floor
    if chroma < 1e-8:
        return _neutral_name(lightness)
    saturation = chroma / (1.0 - abs(2.0 * lightness - 1.0))
    if peak == red:
        hue = ((green - blue) / chroma) % 6.0
    elif peak == green:
        hue = (blue - red) / chroma + 2.0
    else:
        hue = (red - green) / chroma + 4.0
    hue *= 60.0
    if saturation < 0.18:
        return _neutral_name(lightness)
    if 20.0 <= hue < 50.0 and lightness < 0.55 and saturation < 0.65:
        family = "brown"
    else:
        family = "red"
        for limit, name in (
            (18.0, "red"),
            (46.0, "orange"),
            (70.0, "yellow"),
            (165.0, "green"),
            (200.0, "cyan"),
            (255.0, "blue"),
            (290.0, "purple"),
            (345.0, "pink"),
            (360.0, "red"),
        ):
            if hue < limit:
                family = name
                break
    if lightness >= 0.78 and 25.0 <= hue < 70.0 and saturation < 0.55:
        return "beige"
    if lightness >= 0.72:
        return f"light {family}"
    if lightness <= 0.40:
        return f"dark {family}"
    return family


def _neutral_name(lightness: float) -> str:
    if lightness >= 0.90:
        return "white"
    if lightness <= 0.12:
        return "black"
    if lightness >= 0.68:
        return "light gray"
    if lightness <= 0.35:
        return "dark gray"
    return "gray"


def as_rgb_array(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors, dtype=np.float64)
    if values.size == 0:
        raise ValueError("palette is empty")
    if values.ndim == 1:
        values = values.reshape(1, 3)
    if values.shape[-1] != 3:
        raise ValueError("colors must be RGB")
    if values.max(initial=0) <= 1.0:
        values = values * 255.0
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 to CIE Lab. Accepts an array whose last axis is RGB."""
    unit = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(unit <= 0.04045, unit / 12.92, ((unit + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _XYZ.T
    t = xyz / _D65
    f = np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4.0 / 29.0)
    lab = np.empty(f.shape, dtype=np.float64)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def nearest_indices(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    palette = as_rgb_array(palette)
    pixels = rgb_to_lab(np.asarray(rgb, dtype=np.float64))
    swatches = rgb_to_lab(palette.reshape(-1, 1, 3)).reshape(-1, 3)
    delta = pixels[..., None, :] - swatches
    return np.argmin(np.sum(delta * delta, axis=-1), axis=-1).astype(np.int32)


def _lab_pixel(red: float, green: float, blue: float) -> tuple[float, float, float]:
    """sRGB 0-255 to CIE Lab for one pixel. Scalar so dither stays usable on a fine grid."""

    def linear(channel: float) -> float:
        unit = channel / 255.0
        if unit <= 0.04045:
            return unit / 12.92
        return ((unit + 0.055) / 1.055) ** 2.4

    r_lin = linear(red)
    g_lin = linear(green)
    b_lin = linear(blue)
    x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) / _D65[0]
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) / _D65[2]

    def cube(channel: float) -> float:
        if channel > _DELTA**3:
            return channel ** (1.0 / 3.0)
        return channel / (3 * _DELTA**2) + 4.0 / 29.0

    fx, fy, fz = cube(x), cube(y), cube(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _nearest_filament(lab: tuple[float, float, float], swatches: list[tuple[float, float, float]]) -> tuple[int, float]:
    best = 0
    best_distance = None
    for index, swatch in enumerate(swatches):
        distance = (swatch[0] - lab[0]) ** 2 + (swatch[1] - lab[1]) ** 2 + (swatch[2] - lab[2]) ** 2
        if best_distance is None or distance < best_distance:
            best = index
            best_distance = distance
    return best, 0.0 if best_distance is None else best_distance


def floyd_steinberg(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Serpentine Floyd-Steinberg dither onto the filament palette.

    A picture pixel that is already close to a filament stays that filament.
    Neighbor error cannot knock it into another color, so a flat area does not
    start dithering only once some other object is reached. Pixels between
    filaments dither from the first row.
    """
    palette = as_rgb_array(palette)
    source = np.asarray(rgb, dtype=np.float64)
    image = source.copy()
    height, width = image.shape[:2]
    chosen = np.zeros((height, width), dtype=np.int32)
    swatches = [tuple(float(channel) for channel in lab) for lab in rgb_to_lab(palette)]
    filaments = [tuple(float(channel) for channel in color) for color in palette]
    # Wide of "close" leaves a between-filament background solid until a
    # contrasting shape, such as hair, pushes error into it. Error only
    # travels downward, so the dither then starts at that shape.
    snap = 8.0**2
    for y in range(height):
        forward = y % 2 == 0
        columns = range(width) if forward else range(width - 1, -1, -1)
        step = 1 if forward else -1
        for x in columns:
            original = _lab_pixel(float(source[y, x, 0]), float(source[y, x, 1]), float(source[y, x, 2]))
            source_best, source_distance = _nearest_filament(original, swatches)
            red = min(255.0, max(0.0, float(image[y, x, 0])))
            green = min(255.0, max(0.0, float(image[y, x, 1])))
            blue = min(255.0, max(0.0, float(image[y, x, 2])))
            best, _best_distance = _nearest_filament(_lab_pixel(red, green, blue), swatches)
            if source_distance <= snap:
                chosen[y, x] = source_best
                continue
            chosen[y, x] = best
            filament = filaments[best]
            error_r = red - filament[0]
            error_g = green - filament[1]
            error_b = blue - filament[2]
            ahead = x + step
            if 0 <= ahead < width:
                image[y, ahead, 0] += error_r * (7.0 / 16.0)
                image[y, ahead, 1] += error_g * (7.0 / 16.0)
                image[y, ahead, 2] += error_b * (7.0 / 16.0)
            if y + 1 >= height:
                continue
            image[y + 1, x, 0] += error_r * (5.0 / 16.0)
            image[y + 1, x, 1] += error_g * (5.0 / 16.0)
            image[y + 1, x, 2] += error_b * (5.0 / 16.0)
            back = x - step
            if 0 <= back < width:
                image[y + 1, back, 0] += error_r * (3.0 / 16.0)
                image[y + 1, back, 1] += error_g * (3.0 / 16.0)
                image[y + 1, back, 2] += error_b * (3.0 / 16.0)
            if 0 <= ahead < width:
                image[y + 1, ahead, 0] += error_r * (1.0 / 16.0)
                image[y + 1, ahead, 1] += error_g * (1.0 / 16.0)
                image[y + 1, ahead, 2] += error_b * (1.0 / 16.0)
    return chosen


def choose_palette(pixels: np.ndarray, n_colors: int) -> np.ndarray:
    """Filament colors for a picture, including small but distinct hues.

    Median cut spends every split on the largest population, so a patch of
    blue on a cream, red, and gray picture never becomes its own filament.
    Candidates are Lab bins. Each new color is the one farthest from the
    colors already chosen, with population softened so a saturated minority
    can beat another shade of the background.
    """
    samples = as_rgb_array(pixels).reshape(-1, 3)
    if len(samples) == 0:
        raise ValueError("no pixels to build a palette from")
    n_colors = max(1, int(n_colors))
    if len(samples) > 80000:
        picked = np.linspace(0, len(samples) - 1, 80000).astype(np.int64)
        samples = samples[picked]
    lab = rgb_to_lab(samples)
    step = np.array([6.0, 8.0, 8.0], dtype=np.float64)
    keys = np.round(lab / step).astype(np.int32)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    samples = samples[order]
    lab = lab[order]
    change = np.any(np.diff(keys, axis=0) != 0, axis=1)
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    ends = np.concatenate((starts[1:], [len(keys)]))
    counts = (ends - starts).astype(np.float64)
    means_lab = np.vstack([lab[start:end].mean(axis=0) for start, end in zip(starts, ends)])
    means_rgb = np.vstack([samples[start:end].mean(axis=0) for start, end in zip(starts, ends)])
    chroma = np.sqrt(means_lab[:, 1] ** 2 + means_lab[:, 2] ** 2)
    total = float(counts.sum())
    keep = (counts >= max(4.0, total * 0.004)) | ((chroma >= 22.0) & (counts >= max(3.0, total * 0.0015)))
    if not np.any(keep):
        keep = counts == counts.max()
    counts = counts[keep]
    means_lab = means_lab[keep]
    means_rgb = means_rgb[keep]
    chroma = chroma[keep]

    chosen = [int(np.argmax(counts))]
    while len(chosen) < min(n_colors, len(counts)):
        delta = means_lab[:, None, :] - means_lab[chosen][None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)
        weight = np.power(counts, 0.35) * (1.0 + np.clip((chroma - 18.0) / 25.0, 0.0, 2.5))
        score = dist * weight
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] <= 0.0 or dist[nxt] < 8.0:
            break
        chosen.append(nxt)

    seeds = means_lab[chosen]
    delta = lab[:, None, :] - seeds[None, :, :]
    owner = np.argmin(np.sum(delta * delta, axis=-1), axis=1)
    colors: list[tuple[int, int, int]] = []
    for index in range(len(chosen)):
        group = samples[owner == index]
        if len(group) == 0:
            continue
        color = tuple(int(channel) for channel in group.mean(axis=0).round())
        if color not in colors:
            colors.append(color)
    if not colors:
        colors.append(tuple(int(channel) for channel in samples.mean(axis=0).round()))
    ordered = sorted(colors, key=lambda rgb: float(rgb_to_lab(np.array(rgb))[0]), reverse=True)
    return np.asarray(ordered, dtype=np.uint8)


def median_cut(pixels: np.ndarray, n_colors: int) -> np.ndarray:
    """Palette of up to n_colors from an (N, 3) cloud of sRGB pixels."""
    samples = as_rgb_array(pixels).reshape(-1, 3)
    if len(samples) == 0:
        raise ValueError("no pixels to build a palette from")
    n_colors = max(1, int(n_colors))
    buckets = [samples]
    while len(buckets) < n_colors:
        scores = []
        for bucket in buckets:
            if len(bucket) < 2:
                scores.append((-1, 0))
                continue
            span = bucket.max(axis=0).astype(np.int16) - bucket.min(axis=0).astype(np.int16)
            scores.append((int(span.max()), int(span.argmax())))
        choice = max(range(len(buckets)), key=lambda i: scores[i][0])
        if scores[choice][0] <= 0:
            break
        channel = scores[choice][1]
        bucket = buckets.pop(choice)
        bucket = bucket[np.argsort(bucket[:, channel], kind="stable")]
        mid = len(bucket) // 2
        if mid == 0:
            buckets.append(bucket)
            break
        buckets.append(bucket[:mid])
        buckets.append(bucket[mid:])
    colors = [tuple(int(channel) for channel in bucket.mean(axis=0).round()) for bucket in buckets if len(bucket)]
    unique: list[tuple[int, int, int]] = []
    for color in colors:
        if color not in unique:
            unique.append(color)
    ordered = sorted(unique, key=lambda rgb: float(rgb_to_lab(np.array(rgb))[0]), reverse=True)
    return np.asarray(ordered, dtype=np.uint8)
