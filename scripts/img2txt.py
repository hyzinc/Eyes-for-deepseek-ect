#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""img2txt.py - deterministic image-to-text transcript for text-only LLMs.

No multimodal API is used. The output is a plain-text transcript (ASCII art,
color palette, color grid, edges, tiles) that a text-only model such as
DeepSeek can reason over.
"""

import argparse
import math
import subprocess
import sys
import tempfile
import os
from collections import Counter

from PIL import Image, ImageFilter, ImageOps


RAMP = "@%#*+=-:. "  # dark -> light, 10 levels


def clamp255(v):
    return max(0, min(255, int(round(v))))


def hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    h = 0.0
    if d > 0:
        if mx == r:
            h = 60.0 * (((g - b) / d) % 6)
        elif mx == g:
            h = 60.0 * ((b - r) / d + 2)
        else:
            h = 60.0 * ((r - g) / d + 4)
    s = 0.0 if mx == 0 else d / mx
    v = mx
    return h, s, v


def color_name(r, g, b):
    h, s, v = hsv(r, g, b)
    if v < 0.12:
        return "black"
    if s < 0.12:
        if v > 0.85:
            return "white"
        if v > 0.45:
            return "light-gray"
        return "dark-gray"
    if 10 <= h < 50 and s > 0.35 and v < 0.55:
        return "brown"
    if v < 0.30:
        return "dark-" + _hue_name(h, s)
    if v > 0.88 and s < 0.35:
        return "pale-" + _hue_name(h, s)
    if s < 0.45 and v > 0.72:
        return "light-" + _hue_name(h, s)
    return _hue_name(h, s)


def _hue_name(h, s):
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange" if s > 0.55 else "yellow"
    if h < 65:
        return "yellow"
    if h < 150:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 285:
        return "purple"
    if h < 330:
        return "pink"
    return "red"


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(clamp255(x) for x in rgb)


def ascii_grid(img, cols, rows=None, invert=False, contrast=1.0):
    w, h = img.size
    if rows is None:
        rows = max(1, int(round(cols * (h / float(w)) * 0.5)))
    small = img.resize((cols, rows), Image.BILINEAR)
    gray = small.convert("L")
    if invert:
        gray = ImageOps.invert(gray)
    px = gray.load()
    lines = []
    for y in range(rows):
        line = []
        for x in range(cols):
            v = px[x, y]
            v = clamp255((v - 127.5) * contrast + 127.5)
            idx = v * (len(RAMP) - 1) // 255
            line.append(RAMP[idx])
        lines.append("".join(line))
    return lines, rows, cols


def cell_averages(img, cols, rows):
    small = img.resize((cols, rows), Image.BILINEAR)
    px = small.load()
    out = []
    for y in range(rows):
        row = []
        for x in range(cols):
            rgb = px[x, y][:3]
            row.append((hex_of(rgb), color_name(*rgb)))
        out.append(row)
    return out


def palette(img, n=8, sample=96):
    small = img.copy()
    small.thumbnail((sample, sample), Image.BILINEAR)
    px = small.convert("RGB").load()
    counter = Counter()
    w, h = small.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y][:3]
            bucket = (r // 16 * 16, g // 16 * 16, b // 16 * 16)
            counter[bucket] += 1
    total = float(sum(counter.values())) or 1.0
    items = []
    for (rgb, cnt) in counter.most_common(n):
        items.append((hex_of(rgb), color_name(*rgb), cnt / total))
    return items


def brightness_contrast(img):
    small = img.convert("L").resize((64, 64), Image.BILINEAR)
    px = small.load()
    vals = [px[x, y] for y in range(64) for x in range(64)]
    mean = sum(vals) / float(len(vals))
    var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
    return mean / 255.0, math.sqrt(var) / 255.0


def tiles(img, rows=3, cols=3, ascii_cols=16):
    w, h = img.size
    out = []
    for ty in range(rows):
        for tx in range(cols):
            x0 = int(w * tx / cols)
            x1 = int(w * (tx + 1) / cols)
            y0 = int(h * ty / rows)
            y1 = int(h * (ty + 1) / rows)
            crop = img.crop((x0, y0, x1, y1))
            avg = crop.resize((1, 1), Image.BILINEAR).load()[0, 0][:3]
            lines, r_, c_ = ascii_grid(crop, ascii_cols)
            out.append(
                {
                    "id": "t%d%d" % (ty + 1, tx + 1),
                    "box": (x0, y0, x1, y1),
                    "hex": hex_of(avg),
                    "name": color_name(*avg),
                    "brightness": brightness_contrast(crop)[0],
                    "ascii": lines,
                }
            )
    return out


def run_ocr(img):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        img.convert("RGB").save(tmp, format="PNG")
        proc = subprocess.run(
            ["tesseract", tmp, "stdout"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = proc.stdout.strip()
        if not text:
            return "[OCR] no text detected"
        return "[OCR] detected text:\n" + text
    except FileNotFoundError:
        return "[OCR] tesseract not installed (optional; core modes do not need it)"
    except Exception as exc:
        return "[OCR] failed: %s" % exc
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def render_transcript(img, args):
    w, h = img.size
    bright, contrast = brightness_contrast(img)
    lines = []

    lines.append("## META")
    lines.append("- size: %dx%d" % (w, h))
    lines.append("- aspect: %.2f" % (w / float(max(1, h))))
    lines.append("- brightness: %.2f (0=black, 1=white)" % bright)
    lines.append("- contrast: %.2f (0=flat, 1=high)" % contrast)
    small = img.convert("RGB").resize((1, 1), Image.BILINEAR).load()[0, 0]
    lines.append("- overall_color: %s (%s)" % (hex_of(small), color_name(*small)))
    lines.append("")

    if args.palette:
        lines.append("## PALETTE top %d (by area ratio)" % args.palette)
        for hexv, name, ratio in palette(img, args.palette):
            lines.append("- %s %s %.1f%%" % (hexv, name, ratio * 100))
        lines.append("")

    if args.grid:
        gcols, grows = args.grid
        lines.append("## COLOR GRID %dx%d (left-to-right, top-to-bottom)" % (gcols, grows))
        for row in cell_averages(img, gcols, grows):
            lines.append(" | ".join("%s %s" % cell for cell in row))
        lines.append("")

    if args.ascii:
        art, rows, cols = ascii_grid(img, args.ascii, contrast=args.contrast)
        lines.append("## ASCII %dx%d (darker char = darker pixel; aspect corrected)" % (cols, rows))
        lines.extend(art)
        lines.append("")

    if args.edge:
        edge = img.convert("L").filter(ImageFilter.FIND_EDGES)
        art, rows, cols = ascii_grid(edge, args.edge, invert=True, contrast=args.contrast)
        lines.append("## EDGE MAP %dx%d (outlines only)" % (cols, rows))
        lines.extend(art)
        lines.append("")

    if args.tiles:
        trows, tcols = args.tiles
        lines.append("## TILES %dx%d (zoomed crops; coordinates are pixel boxes)" % (trows, tcols))
        for t in tiles(img, trows, tcols, args.tile_ascii):
            lines.append("### %s box=%s avg=%s (%s) brightness=%.2f" % (
                t["id"], t["box"], t["hex"], t["name"], t["brightness"]))
            lines.extend(t["ascii"])
            lines.append("")

    if args.ocr:
        lines.append(run_ocr(img))
        lines.append("")

    return "\n".join(lines)


def parse_dims(s, default=8):
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    return int(s), default


def main():
    ap = argparse.ArgumentParser(description="Convert an image into a text transcript for a text-only LLM.")
    ap.add_argument("image")
    ap.add_argument("--ascii", type=int, default=48, help="ASCII art width in chars (0 to disable)")
    ap.add_argument("--edge", type=int, default=0, help="edge map width in chars (0 to disable)")
    ap.add_argument("--palette", type=int, default=6, help="palette size (0 to disable)")
    ap.add_argument("--grid", default="6x4", help="color grid WxH (0 to disable)")
    ap.add_argument("--tiles", default="0x0", help="tile crops RxC, e.g. 3x3 (0 to disable)")
    ap.add_argument("--tile-ascii", type=int, default=16)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--ocr", action="store_true", help="optional local tesseract OCR")
    ap.add_argument("--crop", default="", help="crop box x0,y0,x1,y1 before transcribing")
    args = ap.parse_args()

    if args.grid == "0":
        args.grid = None
    else:
        args.grid = parse_dims(args.grid)
    if args.tiles == "0x0":
        args.tiles = None
    else:
        args.tiles = parse_dims(args.tiles)

    img = Image.open(args.image).convert("RGB")
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        img = img.crop((x0, y0, x1, y1))
    print(render_transcript(img, args))


if __name__ == "__main__":
    main()
