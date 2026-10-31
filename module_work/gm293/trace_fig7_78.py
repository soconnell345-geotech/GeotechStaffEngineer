"""Read the two curves of GEC-12 Vol I Fig. 7-78 (p. 383; Idriss and
Boulanger 2007): s_r / sigma'_vo against (N1)60-e, off the figure's
embedded image (826 x 556 px), for seismic_geotech.residual_strength
(GM-293).

Run: python -I trace_fig7_78.py "<path>/GEC 12 vol 1.pdf"

Axes, from the image's own grid lines: (N1)60-e = 0 at column 135 and 30
at 781; s_r / sigma'_vo = 0 at row 462.5 and 0.4 at row 35 (1 px = 0.046
blows and 0.0009 in the ratio). A dark pixel is one whose brightest
channel is under 90 (the curves are black; the grid is grey).

- "Significant" curve (void redistribution effects could be
  significant): solid from N = 0, dashed from about N 15. Followed column
  by column, each column taking the dark run nearest a straight line
  through the last 20 points found (the markers drawn over the line leave
  gaps); tabulated at each blow by a straight line fitted to the points
  within 0.6 blows. Its label hides it from N 18.9 to 24.4; above the
  label each dash is read row by row (its middle).
- "Negligible" curve: leaves the solid line at N 9 (a spur on the line
  marks it) and leaves the chart at 0.4 near N 17. Read row by row, each
  row taking the dark run nearest a straight line through the last 20
  points (within 4 px: the gaps between its dashes are long); tabulated
  at each 0.01 of the ratio. Its label hides it from
  0.265 to 0.342; above the label each dash is read as for the other
  curve.

Prints the tables written into residual_strength.py.
"""
import sys

import fitz
import numpy as np

doc = fitz.open(sys.argv[1])
pix = fitz.Pixmap(doc, doc[428].get_images()[0][0])
assert (pix.width, pix.height) == (826, 556), (pix.width, pix.height)
a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
    pix.height, pix.width, pix.n)[:, :, :3].astype(int)
dark = a.max(axis=2) < 90
X0, X1, Y0, Y1 = 135.0, 781.0, 462.5, 35.0
col = lambda N: X0 + N * (X1 - X0) / 30.0
row = lambda r: Y0 - r * (Y0 - Y1) / 0.4
N_of = lambda c: (c - X0) * 30.0 / (X1 - X0)
r_of = lambda y: (Y0 - y) * 0.4 / (Y0 - Y1)


def runs(vals, widest=7):
    """Middles of the runs of consecutive pixels, at most ``widest`` long."""
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= 1:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out if len(g) <= widest]


def follow(steps, find, start, within=3.0):
    """Follow a curve: at each step take the candidate nearest a straight
    line through the last 20 points found (within ``within`` px)."""
    pts = []
    for s in steps:
        cand = find(s)
        if len(pts) >= 5:
            p = np.array(pts[-20:])
            k, b = np.polyfit(p[:, 0], p[:, 1], 1)
            pred = k * s + b
        else:
            pred = start if not pts else pts[-1][1]
        if cand:
            best = min(cand, key=lambda v: abs(v - pred))
            if abs(best - pred) <= within:
                pts.append((s, best))
    return pts


def dashes(n_lo, n_hi, r_lo, r_hi):
    """(N, ratio) at the middle of each dash in the window, read row by
    row."""
    pts = []
    for y in range(int(row(r_hi)), int(row(r_lo)) + 1):
        xs = runs([c for c in range(int(col(n_lo)), int(col(n_hi)) + 1)
                   if dark[y, c]], widest=5)
        if len(xs) == 1:
            pts.append((N_of(xs[0]), r_of(y)))
    out, cur = [], []
    for p in sorted(pts, key=lambda t: t[1]):
        if cur and p[1] - cur[-1][1] > 0.0015:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return [(float(np.mean([q[0] for q in d])),
             float(np.mean([q[1] for q in d]))) for d in out if len(d) >= 3]


# significant: column by column, N 0.1 to 18.9 (the label from 18.96)
sig = [(N_of(c), r_of(y)) for c, y in follow(
    range(int(round(col(0.1))), int(col(18.9)) + 1),
    lambda c: runs([r for r in range(38, 460) if dark[r, c]]),
    row(0.032))]


def sig_at(N, half=0.6):
    """The straight line fitted to the points within ``half`` blows."""
    p = np.array([q for q in sig if abs(q[0] - N) <= half])
    k, b = np.polyfit(p[:, 0], p[:, 1], 1)
    return k * N + b


print("SIGNIFICANT = (")
table = [(float(N), sig_at(N)) for N in range(0, 19)] + [(18.9, sig_at(18.9))]
upper = dashes(23.9, 29.6, 0.250, 0.399)
table += upper
# where the curve leaves the chart at 0.4: the last two dashes extended
(n1, v1), (n2, v2) = upper[-2:]
table.append((n2 + (0.4 - v2) * (n2 - n1) / (v2 - v1), 0.4))
for N, v in table:
    print(f"    ({N:.2f}, {v:.4f}),")
print(")")

# negligible: row by row, ratio 0.105 to 0.264 (the label from 0.265)
neg = [(r_of(y), N_of(x)) for y, x in follow(
    range(int(row(0.105)), int(row(0.264)) - 1, -1),
    lambda y: runs([c for c in range(int(col(8)), int(col(20)))
                    if dark[y, c]]),
    col(11.0), within=4.0)]
rs, ns = zip(*sorted(neg))
# below 0.105: the spur where it leaves the solid line, and the one dash
# between (N 9.9; the column's upper dark run)
c = int(round(col(9.9)))
ys = runs([r for r in range(38, 460) if dark[r, c - 1:c + 2].any()], 6)
low_dash = min((r_of(y) for y in ys if r_of(y) > sig_at(9.9) + 0.003))
print("NEGLIGIBLE = (")
print(f"    (9.00, {sig_at(9.0):.4f}),   # leaves the solid line")
print(f"    (9.90, {low_dash:.4f}),")
for r in np.arange(0.11, 0.265, 0.01):
    print(f"    ({np.interp(r, rs, ns):.2f}, {r:.4f}),")
upper = dashes(15.6, 17.3, 0.340, 0.399)
for N, v in upper:
    print(f"    ({N:.2f}, {v:.4f}),")
(n1, v1), (n2, v2) = upper[-2:]
print(f"    ({n2 + (0.4 - v2) * (n2 - n1) / (v2 - v1):.2f}, 0.4000),")
print(")")
