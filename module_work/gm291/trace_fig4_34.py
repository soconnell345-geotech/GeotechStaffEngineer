"""Trace GEC-13 Fig. 4-34 (p. 4-77) off the embedded image (GM-291).

The figure is a clean chart image (674 x 591 px) with no vector layer.
Grid: x 1.0 to 3.0 m at five vertical lines, y 50 to 100 % at eleven
horizontal lines. The curve is read on every pixel row: its centre column
among the dark pixels that are not a grid line. Prints the calibration, the
read at each 5 % grid row and the dense trace, and writes an overlay image
with the reads drawn back over the figure.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
src = Image.open(sys.argv[1]).convert("RGB")
img = np.asarray(src.convert("L")).astype(float)
H, W = img.shape
dark = img < 128


def runs(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


# the plot box: rows/cols where a dark run spans most of the plot
band_rows = slice(40, 500)
band_cols = slice(140, 640)
vfrac = dark[band_rows, :].mean(axis=0)
hfrac = dark[:, band_cols].mean(axis=1)
# the image's own border frame (first and last pixel) is not a grid line
vlines = [(a + b) / 2 for a, b in runs(vfrac > 0.8) if 100 < a < 660]
hlines = [(a + b) / 2 for a, b in runs(hfrac > 0.8) if 20 < a < 520]
print("vertical grid lines (px):", vlines)
print("horizontal grid lines (px):", hlines)
assert len(vlines) == 5, vlines
assert len(hlines) == 11, hlines
xs_m = [1.0, 1.5, 2.0, 2.5, 3.0]
ys_pc = [100 - 5 * i for i in range(11)]
ax, bx = np.polyfit(vlines, xs_m, 1)
ay, by = np.polyfit(hlines, ys_pc, 1)
print("x residuals (m):", np.round(np.array(xs_m) - (ax * np.array(vlines) + bx), 4))
print("y residuals (%):", np.round(np.array(ys_pc) - (ay * np.array(hlines) + by), 3))

grid_cols = set()
for v in vlines:
    for c in range(int(v) - 3, int(v) + 4):
        grid_cols.add(c)
grid_rows = set()
for h in hlines:
    for r in range(int(h) - 3, int(h) + 4):
        grid_rows.add(r)

trace = []
for row in range(int(hlines[0]) + 1, int(hlines[-1])):
    if row in grid_rows:
        continue
    cols = [c for c in range(int(vlines[0]) + 4, int(vlines[-1]) - 3)
            if dark[row, c] and c not in grid_cols]
    if not cols:
        continue
    # the curve: the longest run of dark columns on the row
    rr = runs([c in set(cols) for c in range(W)])
    rr = [r for r in rr if 2 <= r[1] - r[0] <= 15]
    if not rr:
        continue
    a, b = max(rr, key=lambda r: r[1] - r[0])
    trace.append((row, (a + b) / 2, b - a + 1))

pts = [(ax * c + bx, ay * r + by, w) for r, c, w in trace]
print(f"{len(pts)} rows traced; curve from Dr {pts[0][1]:.1f} % at "
      f"{pts[0][0]:.3f} m to Dr {pts[-1][1]:.1f} % at {pts[-1][0]:.3f} m")

# read at each 5 % grid row: rows on a grid line are hidden by it, so fit a
# line through the trace within 1.5 % either side of the target
print("\nDr (%)  spacing (m)  [rows used, local slope]")
reads = {}
for target in range(55, 91, 5):
    near = [(x, y) for x, y, w in pts if abs(y - target) <= 1.5]
    if len(near) < 3:
        print(target, "too few rows", len(near))
        continue
    yy = np.array([p[1] for p in near])
    xx = np.array([p[0] for p in near])
    k, c0 = np.polyfit(yy, xx, 1)
    reads[target] = k * target + c0
    print(f"{target:5d}  {reads[target]:.3f}  [{len(near)}, {k:.4f}]")

# draw the reads back over the figure
ov = src.copy()
d = ImageDraw.Draw(ov)
for target, x in reads.items():
    px = (x - bx) / ax
    py = (target - by) / ay
    d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 0, 0), width=2)
for x, y, w in pts[::6]:
    px = (x - bx) / ax
    py = (y - by) / ay
    d.point((px, py), fill=(0, 160, 255))
ov.save(sys.argv[2])
print("\noverlay:", sys.argv[2])
