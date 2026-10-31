"""UFC 3-250-01 Figure E-1, the 18-kip single-axle flexible design curve,
read off the printed chart.

Figure E-1 (pdf_page 211, printed 192) gives the thickness of cover above
a layer of a given CBR for 10 to 10,000,000 passes, as eight curves on a
log CBR axis. The reference layer's ``figure_e1_flexible_thickness`` is
exact at its Appendix G anchors but strays between them: at CBR 10 it
gives 5.7 in for 1,000,000 passes where the curve is at 7.4, and 8.7 in
for 10,000,000 where the curve is at 8.0. This module reads the chart
itself instead.

How it was read (GM-298, 2026-10-31): the chart is a raster embedded in
the PDF (520 x 664 px; CBR 1, 10 and 100 at x = 48, 276 and 504; 0 and
30 in at y = 598 and 28, 19 px per inch). Each curve was followed along
its length from the left frame, the curve-number labels bridged, then
smoothed over three samples and thinned to the points below (0.5 px
tolerance). Each curve runs from CBR 1 to the CBR where it meets the axis.
Between curves the thickness is interpolated in log(passes).

Checked against every Figure E-1 read printed in Appendix G: G-1 (pdf_page
275, printed 256) CBR 3 at 1,000,000 passes 16.4 in (read 16.4) and CBR 4
at 1,395,400 passes 14 in (13.9); G-3 (pdf_pages 277-279, printed 258-260)
at 5,000,000 passes CBR 10 7.8 in (7.8), CBR 7 9.8 (9.8), CBR 4 14.5
(14.5), CBR 25 3.4 (3.4). G-8.10 (pdf_page 286, printed 267) says CBR 3.5
at 1,200,000 passes gives 16 in; the curves give 15.1 there (the reference
layer also 15.1).

UNITS: CBR %, inches, passes of the 18-kip single axle, dual tires.
"""

import math

FIGE1_REF = "UFC 3-250-01, Figure E-1 (pdf_page 211, printed 192)"

# passes -> ((CBR, thickness in), ...) along the printed curve, CBR 1 to
# the axis.
_E1 = {
    10_000_000: ((1, 30.00), (1.1, 29.03), (1.28, 27.01), (1.69, 23.73),
                 (2.66, 18.79), (3.85, 15.22), (5.54, 11.90), (6.41, 10.68),
                 (7.34, 9.72), (9.41, 8.31), (13.9, 6.33), (24.9, 3.63),
                 (29, 2.75), (31.4, 2.16), (33.3, 1.51), (34.5, 0.80),
                 (34.7, 0.00)),
    1_000_000: ((1, 28.59), (1.24, 25.80), (1.42, 24.26), (1.65, 22.53),
                (2.19, 19.52), (2.72, 17.44), (3.15, 15.94), (3.94, 13.86),
                (5.63, 10.79), (6.41, 9.82), (8.48, 8.23), (10.6, 7.06),
                (20.3, 4.06), (24.7, 3.06), (27.7, 2.32), (29, 1.91),
                (30.1, 1.48), (31.2, 0.78), (31.5, 0.00)),
    100_000: ((1, 26.55), (1.1, 25.36), (1.49, 21.89), (2.07, 18.48),
              (3.6, 13.26), (4.47, 11.42), (5.5, 9.83), (6.8, 8.58),
              (9.76, 6.75), (20.7, 3.26), (22.6, 2.71), (25.2, 1.95),
              (26.7, 1.29), (27.3, 0.82), (27.6, 0.00)),
    10_000: ((1, 23.92), (1.1, 22.79), (1.24, 21.50), (1.55, 19.16),
             (1.81, 17.69), (2.88, 13.32), (3.96, 10.68), (4.81, 9.34),
             (6.4, 7.78), (8.94, 6.17), (15.1, 3.83), (17.8, 3.01),
             (18.9, 2.65), (20.5, 2.08), (21.9, 1.46), (22.7, 0.76),
             (23, 0.00)),
    1_000: ((1, 20.51), (1.29, 18.03), (1.92, 14.29), (2.31, 12.65),
            (3.12, 10.24), (3.6, 9.30), (4.19, 8.42), (4.93, 7.58),
            (7.16, 5.84), (11.7, 3.69), (14.2, 2.70), (15.6, 2.16),
            (17.1, 1.33), (17.5, 0.88), (17.9, 0.00)),
    100: ((1, 15.89), (1.31, 13.49), (1.42, 12.65), (1.89, 10.49),
          (2.17, 9.53), (2.78, 8.12), (3.28, 7.30), (4.17, 6.17),
          (7.61, 3.60), (9.62, 2.46), (10.5, 1.92), (11.3, 1.31),
          (11.7, 0.86), (11.9, 0.00)),
    50: ((1, 14.16), (1.41, 11.22), (1.76, 9.68), (1.98, 8.94), (2.56, 7.59),
         (3.02, 6.77), (6.85, 3.27), (7.58, 2.79), (8.6, 2.10), (9.32, 1.52),
         (9.75, 1.10), (10, 0.64), (10.1, 0.00)),
    10: ((1, 9.25), (1.42, 7.40), (2.24, 5.41), (4, 3.08), (4.73, 2.27),
         (5.36, 1.56), (5.64, 1.16), (5.84, 0.72), (6.01, 0.00)),
}
_PASSES = sorted(_E1)


def _on_curve(passes, cbr):
    pts = _E1[passes]
    if cbr >= pts[-1][0]:
        return 0.0
    lc = math.log10(cbr)
    for (c0, t0), (c1, t1) in zip(pts, pts[1:]):
        if cbr <= c1:
            f = (lc - math.log10(c0)) / (math.log10(c1) - math.log10(c0))
            return t0 + f * (t1 - t0)
    return 0.0  # pragma: no cover


def figure_e1_thickness(cbr, passes) -> dict:
    """Thickness of cover (in) above a layer of this CBR for this many
    passes of the 18-kip single axle, read off Figure E-1.

    0.0 where the CBR lies past the end of the curves (no cover needed
    over that layer). ValueError below CBR 1 or outside 10-10,000,000
    passes, the range the chart prints.
    """
    if cbr is None or cbr < 1.0:
        raise ValueError(
            f"CBR {cbr} is below Figure E-1's CBR axis (1-100)."
        )
    if passes is None or not 10 * (1 - 1e-9) <= passes <= 1e7 * (1 + 1e-9):
        raise ValueError(
            f"{passes} passes is outside Figure E-1's curves (10 to "
            "10,000,000 passes of the 18-kip single axle)."
        )
    passes = min(max(passes, 10), 10_000_000)
    hi = next(p for p in _PASSES if p >= passes)
    lo = max([p for p in _PASSES if p <= passes])
    t_hi = _on_curve(hi, cbr)
    if hi == lo:
        t = t_hi
    else:
        t_lo = _on_curve(lo, cbr)
        f = ((math.log10(passes) - math.log10(lo))
             / (math.log10(hi) - math.log10(lo)))
        t = t_lo + f * (t_hi - t_lo)
    return {
        "cbr": cbr,
        "passes": passes,
        "thickness_in": round(t, 1),
        "chart_read": True,
        "tolerance": (
            "read along each printed curve (about 0.05 in a pixel); every "
            "Appendix G read of it (G-1, G-3) reproduced within 0.1 in; "
            "interpolated in log(passes) between curves"),
        "reference": FIGE1_REF,
    }
