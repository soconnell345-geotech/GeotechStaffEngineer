"""UFC 3-250-01 Figure F-1, the 18-kip single-axle plain concrete design
chart, read off the print.

Figure F-1 (pdf_page 243, printed 224) is worked as para 13-3 (pdf_page
56, printed 37) says: from the 28-day flexural strength across to the k
curve, down (or up) to the pass-level line, across to the thickness
scale. The reference layer's ``figure_f1_rigid_thickness`` is exact at its
two Appendix G anchors but barely moves with flexural strength (k 100 pci,
1,000,000 passes: 6.6 in at 400 psi, 6.2 in at 900 psi, where the chart
gives 9.9 and 6.0), and at k 50 pci it gives 6.5 in where Appendix G-8.12
reads 8. This module reads the chart itself instead.

How it was read (GM-298, 2026-10-31): the chart is a raster embedded in
the PDF (560 x 696 px; flexural strength 1,000 and 100 psi at y = 51 and
660, thickness 12 and 4 in at the same heights; frame x = 58 to 506).
Each pass-level line is straight: fitted by least squares to its ink
(residual 0.5-0.9 px). Each k curve was tracked row by row from the top
edge, the curve labels masked, and fitted with one smooth curve (residual
0.3 px); its x is tabled below every 25 psi from 1,000 psi down to where
it leaves the chart. Between k curves x is interpolated in log(k); between
pass-level lines the height in log(passes).

Checked against every Figure F-1 read printed in Appendix G: G-7
(pdf_page 283, printed 264) 650 psi, k 100, 20,000,000 passes 8.1 in
(read 8.1); G-8.11 (pdf_page 286, printed 267) 650 psi, k 325, 1,200,000
passes 6.3 in (6.3); G-8.12 (pdf_pages 286-287) 650 psi, k 50, 1,200,000
passes "a pavement slab 8 in thick" (7.8, 8 in once rounded up).

UNITS: psi, pci (psi/in), inches, passes of the 18-kip single axle.
"""

import math

FIGF1_REF = "UFC 3-250-01, Figure F-1 (pdf_page 243, printed 224)"
PARA_13_3_REF = "UFC 3-250-01, para 13-3 (pdf_page 56, printed 37)"
TABLE_10_1_NOTE_REF = "UFC 3-250-01, Table 10-1 notes (pdf_page 50, printed 31)"
K_MAX_DESIGN = 500.0   # Table 10-1 note 2: "a k of 500 psi/in will be the
#                        maximum used for design"

_Y1000, _Y100 = 51.0, 660.0
_PX_R = (_Y100 - _Y1000) / 900.0       # px per psi
_PX_T = (_Y100 - _Y1000) / 8.0         # px per inch (12 in at the top)
_R_STEP = 25.0

# k (pci) -> x (chart px) of the k curve at R = 1000, 975, ... psi, down to
# where the curve leaves the chart at its right edge.
_K_X = {
    500: (58.1, 61.8, 65.9, 70.5, 75.4, 80.6, 86.1, 91.8, 97.7, 103.9, 110.4,
          117.3, 124.5, 132.1, 140.2, 148.8, 158.1, 167.9, 178.6, 190.0,
          202.3, 215.6, 230.1, 245.9, 263.3, 282.4, 303.7, 327.6, 354.6,
          385.4, 420.8, 461.8),
    400: (75.5, 80.4, 85.4, 90.5, 95.7, 101.2, 106.9, 112.8, 119.0, 125.5,
          132.3, 139.5, 147.0, 155.1, 163.6, 172.7, 182.4, 192.8, 204.0,
          216.0, 229.0, 243.0, 258.3, 274.9, 293.1, 313.1, 335.4, 360.2,
          388.2, 420.1, 456.7),
    300: (97.4, 102.5, 107.4, 112.5, 117.7, 123.3, 129.2, 135.4, 142.0, 148.9,
          156.2, 163.9, 171.9, 180.5, 189.5, 199.1, 209.3, 220.2, 231.8,
          244.4, 258.0, 272.6, 288.5, 305.8, 324.8, 345.6, 368.7, 394.5,
          423.4, 456.4, 494.2),
    200: (122.1, 127.2, 132.6, 138.1, 144.0, 150.0, 156.4, 163.0, 169.9,
          177.2, 184.8, 192.8, 201.2, 210.1, 219.6, 229.7, 240.4, 251.9,
          264.1, 277.2, 291.3, 306.5, 322.9, 340.8, 360.3, 381.9, 405.9,
          433.0, 463.7),
    100: (146.0, 152.0, 157.5, 163.0, 168.7, 174.7, 181.2, 188.0, 195.1,
          202.5, 210.3, 218.4, 226.9, 235.9, 245.4, 255.5, 266.4, 278.0,
          290.6, 304.1, 318.7, 334.4, 351.3, 369.7, 389.8, 412.0, 436.9,
          465.4),
    50: (167.1, 172.1, 177.5, 183.3, 189.4, 195.8, 202.5, 209.5, 216.7, 224.3,
         232.2, 240.6, 249.3, 258.6, 268.4, 278.9, 290.2, 302.3, 315.5, 329.7,
         345.2, 362.0, 380.3, 400.3, 422.0, 445.6, 471.3),
    25: (184.7, 191.0, 196.9, 202.8, 208.9, 215.2, 222.0, 229.2, 236.9, 245.1,
         253.8, 263.0, 272.7, 282.8, 293.5, 304.7, 316.5, 329.0, 342.2, 356.3,
         371.4, 387.6, 405.1, 423.9, 444.4, 466.5, 490.4),
}
# passes -> (slope, intercept) of the pass-level line, y = slope * x + b.
_LINES = {
    50_000_000: (-1.5215, 700.3),
    10_000_000: (-1.4552, 715.7),
    1_000_000: (-1.3564, 738.0),
    100_000: (-1.2542, 762.9),
    30_000: (-1.1972, 775.8),
    10_000: (-1.1430, 787.8),
    3_000: (-1.0844, 803.4),
    1_000: (-1.0249, 816.2),
}
_KS = sorted(_K_X)
_NS = sorted(_LINES)


def _x_on_k(k, flexural_psi):
    xs = _K_X[k]
    i = (1000.0 - flexural_psi) / _R_STEP
    if i > len(xs) - 1 + 1e-9:
        return None
    i0 = min(int(math.floor(i)), len(xs) - 2)
    f = i - i0
    return xs[i0] + f * (xs[i0 + 1] - xs[i0])


def _bracket(values, v):
    hi = next(p for p in values if p >= v)
    lo = max(p for p in values if p <= v)
    return lo, hi


def figure_f1_thickness(flexural_psi, k_pci, passes) -> dict:
    """Plain concrete thickness (in) off Figure F-1 for this 28-day
    flexural strength, k and passes of the 18-kip single axle.

    k above 500 pci is taken as 500 (Table 10-1 note 2); ValueError for k
    below the chart's 25 pci curve, flexural strength outside 100-1,000
    psi, passes outside 1,000-50,000,000, or a reading off the chart (a k
    curve that has left it, or more than 12 in). A reading under the
    chart's 4 in is returned as 4.0 with ``below_chart`` set.
    """
    notes = []
    if k_pci is None or k_pci < 25.0:
        raise ValueError(
            f"k {k_pci} pci is below Figure F-1's lowest curve (25 pci; "
            "Table 10-1 note 1 makes 25 the minimum used for design)."
        )
    if k_pci > K_MAX_DESIGN:
        notes.append(
            f"k {k_pci:g} pci taken as {K_MAX_DESIGN:g} pci, the maximum "
            "used for design (Table 10-1 note 2)."
        )
        k_pci = K_MAX_DESIGN
    if flexural_psi is None or not 100.0 <= flexural_psi <= 1000.0:
        raise ValueError(
            f"Flexural strength {flexural_psi} psi is outside Figure F-1's "
            "scale (100-1,000 psi)."
        )
    if passes is None or not 1000 * (1 - 1e-9) <= passes <= 5e7 * (1 + 1e-9):
        raise ValueError(
            f"{passes} passes is outside Figure F-1's pass-level lines "
            "(1,000 to 50,000,000)."
        )
    passes = min(max(passes, 1000), 50_000_000)

    k_lo, k_hi = _bracket(_KS, k_pci)
    x_lo, x_hi = _x_on_k(k_lo, flexural_psi), _x_on_k(k_hi, flexural_psi)
    if x_lo is None or x_hi is None:
        raise ValueError(
            f"At {flexural_psi:g} psi the k {k_pci:g} curve has left Figure "
            "F-1 (beyond its right edge); the chart gives no thickness."
        )
    if k_hi == k_lo:
        x = x_lo
    else:
        f = ((math.log10(k_pci) - math.log10(k_lo))
             / (math.log10(k_hi) - math.log10(k_lo)))
        x = x_lo + f * (x_hi - x_lo)

    n_lo, n_hi = _bracket(_NS, passes)
    y_lo = _LINES[n_lo][0] * x + _LINES[n_lo][1]
    y_hi = _LINES[n_hi][0] * x + _LINES[n_hi][1]
    if n_hi == n_lo:
        y = y_lo
    else:
        f = ((math.log10(passes) - math.log10(n_lo))
             / (math.log10(n_hi) - math.log10(n_lo)))
        y = y_lo + f * (y_hi - y_lo)
    t = 12.0 - (y - _Y1000) / _PX_T
    if t > 12.0 + 1e-6:
        raise ValueError(
            f"Figure F-1 reads {t:.1f} in, above its 12 in scale, for "
            f"{flexural_psi:g} psi, k {k_pci:g} pci, {passes:,.0f} passes."
        )
    below = t < 4.0
    if below:
        notes.append(
            f"Figure F-1 reads {t:.1f} in, under its 4 in scale; 4.0 is "
            "returned (the 6 in minimum of para 13-3 governs a slab)."
        )
        t = 4.0
    return {
        "flexural_psi": flexural_psi,
        "k_pci": k_pci,
        "passes": passes,
        "thickness_in": round(t, 1),
        "below_chart": below,
        "chart_read": True,
        "tolerance": (
            "read off the printed Figure F-1 (about 0.02 in a pixel); "
            "every Appendix G read of it (G-7, G-8.11, G-8.12) reproduced; "
            "interpolated in log(k) and log(passes) between the drawn "
            "curves and lines"),
        "notes": notes,
        "reference": FIGF1_REF,
    }
