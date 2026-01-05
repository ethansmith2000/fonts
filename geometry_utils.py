import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont


def discover_font_files(font_root: str,
                        extensions: Sequence[str] = (".ttf", ".otf")) -> List[str]:
    font_paths: List[str] = []
    for root, _, files in os.walk(font_root):
        for name in files:
            if name.lower().endswith(tuple(ext.lower() for ext in extensions)):
                font_paths.append(os.path.join(root, name))
    if not font_paths:
        raise FileNotFoundError(
            f"No font files with extensions {extensions} found under {font_root}")
    font_paths.sort()
    return font_paths


def _sample_line(p0, p1, steps: int) -> np.ndarray:
    steps = max(2, steps)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    return (1 - t) * p0 + t * p1


def _sample_quadratic(p0, p1, p2, steps: int) -> np.ndarray:
    steps = max(2, steps)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


def _sample_cubic(p0, p1, p2, p3, steps: int) -> np.ndarray:
    steps = max(2, steps)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    p3 = np.array(p3, dtype=np.float32)
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (
        t ** 3) * p3


def commands_to_polylines(commands,
                          samples_per_curve: int = 32) -> List[List[Tuple[float, float]]]:
    polylines: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    current_start: Optional[Tuple[float, float]] = None
    pen_pos: Optional[Tuple[float, float]] = None

    def append_point(pt):
        nonlocal current
        if not current or (current[-1][0] != pt[0] or current[-1][1] != pt[1]):
            current.append((float(pt[0]), float(pt[1])))

    for cmd_name, pts in commands:
        pts = [tuple(p) for p in pts]
        if cmd_name == "moveTo":
            if current:
                polylines.append(current)
                current = []
            if not pts:
                continue
            pen_pos = pts[0]
            current_start = pts[0]
            append_point(pen_pos)
        elif cmd_name == "lineTo" and pen_pos is not None and pts:
            for sampled in _sample_line(pen_pos, pts[0], max(2, samples_per_curve // 8))[1:]:
                append_point(tuple(sampled))
            pen_pos = pts[0]
        elif cmd_name == "curveTo" and pen_pos is not None and len(pts) == 3:
            curve = _sample_cubic(pen_pos, pts[0], pts[1], pts[2], samples_per_curve)
            for sampled in curve[1:]:
                append_point(tuple(sampled))
            pen_pos = pts[2]
        elif cmd_name == "qCurveTo" and pen_pos is not None and pts:
            controls = [np.array(p, dtype=np.float32) for p in pts]
            start = np.array(pen_pos, dtype=np.float32)
            if len(controls) == 1:
                end = controls[0]
                curve = _sample_quadratic(start, (start + end) / 2.0, end, samples_per_curve)
                for sampled in curve[1:]:
                    append_point(tuple(sampled))
                pen_pos = tuple(end.tolist())
            else:
                for idx in range(len(controls) - 1):
                    ctrl = controls[idx]
                    if idx == len(controls) - 2:
                        end = controls[idx + 1]
                    else:
                        end = (ctrl + controls[idx + 1]) / 2.0
                    segment = _sample_quadratic(start, ctrl, end, samples_per_curve)
                    for sampled in segment[1:]:
                        append_point(tuple(sampled))
                    start = end
                pen_pos = tuple((controls[-1]).tolist())
        elif cmd_name == "closePath" and pen_pos is not None and current_start is not None:
            segment = _sample_line(pen_pos, current_start, max(2, samples_per_curve // 8))
            for sampled in segment[1:]:
                append_point(tuple(sampled))
            polylines.append(current)
            current = []
            pen_pos = None
            current_start = None

    if current:
        polylines.append(current)

    return polylines


def load_font_glyph_polylines(font_path: str,
                              chars: Sequence[str],
                              samples_per_curve: int = 32) -> Tuple[Dict[str, List[List[
                                  Tuple[float, float]]]], int]:
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    results: Dict[str, List[List[Tuple[float, float]]]] = {}
    for char in chars:
        glyph_name = None if cmap is None else cmap.get(ord(char))
        if glyph_name is None or glyph_name not in glyph_set:
            continue
        pen = DecomposingRecordingPen(glyph_set)
        glyph_set[glyph_name].draw(pen)
        commands = pen.value
        polylines = commands_to_polylines(commands, samples_per_curve=samples_per_curve)
        if polylines:
            results[char] = polylines
    units_per_em = font["head"].unitsPerEm if "head" in font else 1000
    font.close()
    return results, units_per_em


def stack_and_normalize(polylines: Sequence[Sequence[Tuple[float, float]]],
                        units_per_em: float,
                        eps: float = 1e-6) -> np.ndarray:
    sequences: List[np.ndarray] = []
    for contour in polylines:
        if not contour:
            continue
        sequences.append(np.asarray(contour, dtype=np.float32))
    if not sequences:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.concatenate(sequences, axis=0)
    scale = units_per_em if units_per_em > 0 else 1000.0
    points = points / float(scale)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    max_extent = np.abs(points).max()
    max_extent = max(max_extent, eps)
    points = points / max_extent
    return points.astype(np.float32)


def render_glyph_bitmap(font_path: str,
                        char: str,
                        image_size: int = 256,
                        padding: float = 0.08) -> Optional[np.ndarray]:
    try:
        font_size = max(1, int(image_size * (1.0 - padding * 2)))
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return None

    canvas = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
    except ValueError:
        return None
    if bbox is None:
        return None

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (image_size - text_w) / 2 - bbox[0]
    y = (image_size - text_h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    bitmap = np.array(canvas, dtype=np.uint8)
    if bitmap.max() == 0:
        return None
    return bitmap


def bitmap_to_sdf(bitmap: np.ndarray) -> Optional[np.ndarray]:
    if bitmap is None or bitmap.size == 0:
        return None
    fg = (bitmap > 127).astype(np.uint8)
    if fg.max() == 0:
        return None
    inside = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
    sdf = (outside - inside) / max(bitmap.shape[0], bitmap.shape[1], 1)
    return sdf.astype(np.float32)


def _bilinear_sample_numpy(grid: np.ndarray, xs: np.ndarray,
                           ys: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)

    Ia = grid[y0, x0]
    Ib = grid[y0, x1]
    Ic = grid[y1, x0]
    Id = grid[y1, x1]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def sample_sdf_points(sdf_map: np.ndarray,
                      num_samples: int,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    if sdf_map is None:
        raise ValueError("sdf_map must not be None")
    rng = rng or np.random.default_rng()
    h, w = sdf_map.shape
    xs = rng.random(num_samples) * max(w - 1, 1)
    ys = rng.random(num_samples) * max(h - 1, 1)
    values = _bilinear_sample_numpy(sdf_map, xs, ys)
    coords = np.stack([
        (xs / max(w - 1, 1)) * 2.0 - 1.0,
        (ys / max(h - 1, 1)) * 2.0 - 1.0,
    ], axis=-1)
    return coords.astype(np.float32), values.astype(np.float32)

