"""AASHTO 1993 Section 3.2.1 effective modulus of subgrade reaction, with
Figure 3.3 read off the printed chart.

Figure 3.3 (pdf_page 128, printed II-39) gives the composite k_inf of a
subbase of thickness Dsb and modulus Esb over a roadbed of modulus MR. The
reference layer's ``composite_k.composite_k_subbase`` fits a power law
through the four Table 3.3 rows, all at Dsb = 6 in, and scales it by a
three-point guess for Dsb; away from those rows it departs from the chart
(at Dsb 6 in, Esb 400,000 psi, MR 5,000 psi it gives 3,026 pci where the
chart gives about 490). This module reads the chart itself instead.

How it was read (GM-298, 2026-10-31): the scanned page rendered at its scan
resolution (about 305 dpi). Every printed curve's height was measured at
each even Dsb gridline (6-18 in) by fitting the curve's ink locally; the
eleven k_inf diagonals (50-2000 pci) and the turning line by where they
cross the gridline rows. k_inf then follows the chart's own construction:

  y_E = height of the Esb curve at Dsb    (upper quadrant)
  y_M = height of the MR curve at Dsb     (lower quadrant)
  x_t = y_M + c_t(y_M)                    (turning line, x - y = c_t)
  c   = x_t - y_E                         (the diagonal through (x_t, y_E))
  k_inf from c, log-interpolated between the drawn diagonals

Esb and MR are interpolated between curves in log(Esb), log(MR); Dsb
linearly between gridlines. Checked against the five printed anchors, all
at Dsb = 6 in: the figure's own example (Esb 20,000, MR 7,000 -> 400; read
391) and the four Table 3.3 rows (pdf_page 132, printed II-43: 1,100 ->
1,085; 160 -> 154; 230 -> 230; 410 -> 391), within 5 %, the spread of the
guide's own two reads of one point (400 and 410).

Figure 3.4 (pdf_page 129, printed II-40), k on a rigid foundation within
10 ft, is read here too (GM-303): the reference layer scales one
multiplier per MR, exact at the four Table 3.3 rows (all Dsg 5 ft), by a
three-point depth guess, but the chart's k depends on k_inf as well; away
from those rows it gave 12-36 % below the chart at Dsg 2-3 ft or at twice
the MR/19.4 k_inf, and up to 26 % above it at Dsg 7-9 ft. See
``k_on_rigid_foundation``.

The worksheet (Table 3.2, pdf_page 127, printed II-38) then follows the
reference layer step by step, with Figure 3.5 (relative damage) and
Figure 3.6 (loss of support) from
``geotech_references.aashto_1993.composite_k``.

UNITS: US customary (psi, pci, inches, feet).
"""

import math

import numpy as np

from geotech_references.aashto_1993 import equations as _eq

try:  # Figures 3.4-3.6 (refs follow-on module)
    from geotech_references.aashto_1993 import composite_k as _ck
except ImportError:  # pragma: no cover - depends on refs version
    _ck = None

FIG33_REF = "AASHTO 1993 Guide, Figure 3.3 (pdf_page 128, printed II-39)"

# Upper quadrant: heights (scan pixels, downward) of the nine Esb curves at
# each even Dsb gridline, in _ESB order.
_ESB = (1_000_000, 600_000, 400_000, 200_000, 100_000, 75_000, 50_000,
        30_000, 15_000)
_UPPER = {
    18: (945.5, 974.2, 999.2, 1034.7, 1074.7, 1090.8, 1112.1, 1141.7, 1180.3),
    16: (990.7, 1018.2, 1040.3, 1072.9, 1112.6, 1127.1, 1146.9, 1176.5, 1212.3),
    14: (1037.1, 1062.4, 1083.1, 1114.5, 1153.7, 1166.1, 1185.5, 1211.9, 1246.0),
    12: (1092.4, 1115.3, 1134.5, 1164.3, 1200.4, 1212.1, 1230.5, 1254.0, 1286.4),
    10: (1155.9, 1177.8, 1194.6, 1222.9, 1254.7, 1265.1, 1281.4, 1304.6, 1333.1),
    8: (1233.2, 1251.9, 1266.8, 1293.1, 1319.0, 1330.3, 1344.5, 1364.5, 1390.3),
    6: (1328.5, 1342.8, 1355.8, 1377.1, 1397.1, 1407.6, 1420.4, 1434.5, 1459.5),
}
# Lower quadrant: heights of the nine MR curves, in _MR order. (Dsb 8 in,
# MR 10,000: the curve is not resolved at that gridline; 2203.5 is set from
# its neighbours' steps between Dsb 10, 8 and 6.)
_MR = (1000, 2000, 3000, 5000, 7000, 10_000, 12_000, 16_000, 20_000)
_LOWER = {
    18: (1747.7, 1832.0, 1882.5, 1942.3, 1986.8, 2030.3, 2052.6, 2087.3, 2114.2),
    16: (1765.5, 1855.1, 1906.1, 1968.8, 2012.2, 2055.2, 2079.2, 2114.0, 2141.8),
    14: (1785.8, 1878.9, 1931.1, 1993.3, 2039.8, 2081.2, 2108.6, 2141.8, 2173.0),
    12: (1811.5, 1907.2, 1959.3, 2025.2, 2071.8, 2115.3, 2141.5, 2178.6, 2207.7),
    10: (1843.0, 1939.3, 1992.9, 2062.7, 2107.5, 2156.6, 2183.0, 2218.0, 2251.0),
    8: (1879.6, 1981.2, 2034.1, 2110.0, 2152.0, 2203.5, 2232.0, 2270.9, 2302.6),
    6: (1926.8, 2032.5, 2087.1, 2167.4, 2214.7, 2269.3, 2296.6, 2335.9, 2370.7),
}
# The k_inf diagonals x - y = c, as drawn, and the turning line
# x - y = -339.5 at y = 1770 rising to -334.5 at y = 2480.
_K_LINES = (50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000)
_K_C = (69.5, 190.5, 312.0, 380.5, 433.5, 480.5, 515.5, 557.0, 600.5, 673.0,
        724.5)
DSB_RANGE = (6.0, 18.0)
ESB_RANGE = (15_000.0, 1_000_000.0)
MR_RANGE = (1000.0, 20_000.0)


def _turning_c(y):
    return -339.5 + (y - 1770.0) * (5.0 / 710.0)


def _curve_height(table, labels, dsb, value):
    logs = np.log10(np.array(labels, dtype=float))
    order = np.argsort(logs)

    def at(d):
        return float(np.interp(math.log10(value), logs[order],
                               np.array(table[d])[order]))
    ds = sorted(table)
    return float(np.interp(dsb, ds, [at(d) for d in ds]))


def composite_k_inf(mr_psi, esb_psi, dsb_in) -> dict:
    """Composite k_inf from Figure 3.3 as printed (semi-infinite roadbed).

    Inside the printed curves the chart's own construction is followed (see
    the module docstring). Outside them:

    - Dsb below 6 in, Esb below 15,000 psi or MR below 1,000 psi raise
      ValueError (reading off the lowest curve would overstate k);
    - Dsb above 18 in, Esb above 1,000,000 psi or MR above 20,000 psi use
      the nearest printed curve, with a note (understates k);
    - a k_inf above the highest drawn diagonal is held at 2,000 pci, with a
      note.

    Returns {'mr_psi', 'esb_psi', 'dsb_in', 'k_inf_pci', 'chart_read',
    'method', 'reference', 'note'?}.
    """
    for name, val in (("mr_psi", mr_psi), ("esb_psi", esb_psi),
                      ("dsb_in", dsb_in)):
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    if dsb_in < DSB_RANGE[0]:
        raise ValueError(
            f"Subbase thickness {dsb_in} in is below Figure 3.3's printed "
            f"Dsb scale ({DSB_RANGE[0]:g}-{DSB_RANGE[1]:g} in); read the "
            "chart and give k_pci directly, or design the slab on the "
            "roadbed (mr_psi)."
        )
    if esb_psi < ESB_RANGE[0]:
        raise ValueError(
            f"Subbase modulus {esb_psi} psi is below Figure 3.3's lowest "
            f"printed curve ({ESB_RANGE[0]:,.0f} psi); give k_pci directly "
            "or design the slab on the roadbed (mr_psi)."
        )
    if mr_psi < MR_RANGE[0]:
        raise ValueError(
            f"Roadbed modulus {mr_psi} psi is below Figure 3.3's lowest "
            f"printed curve ({MR_RANGE[0]:,.0f} psi); give k_pci directly."
        )
    notes = []
    dsb = dsb_in
    if dsb > DSB_RANGE[1]:
        dsb = DSB_RANGE[1]
        notes.append(f"Dsb {dsb_in} in is above the printed scale; the "
                     f"{DSB_RANGE[1]:g}-in gridline was used (understates k).")
    esb = esb_psi
    if esb > ESB_RANGE[1]:
        esb = ESB_RANGE[1]
        notes.append(f"Esb {esb_psi:,.0f} psi is above the highest printed "
                     "curve; the 1,000,000-psi curve was used (understates k).")
    mr = mr_psi
    if mr > MR_RANGE[1]:
        mr = MR_RANGE[1]
        notes.append(f"MR {mr_psi:,.0f} psi is above the highest printed "
                     "curve; the 20,000-psi curve was used (understates k).")

    y_e = _curve_height(_UPPER, _ESB, dsb, esb)
    y_m = _curve_height(_LOWER, _MR, dsb, mr)
    c = y_m + _turning_c(y_m) - y_e
    if c > _K_C[-1]:
        k_inf = float(_K_LINES[-1])
        notes.append("The construction lands beyond the highest drawn "
                     "diagonal; k_inf held at 2,000 pci.")
    else:
        log_k = float(np.interp(c, _K_C, [math.log10(k) for k in _K_LINES]))
        k_inf = 10 ** log_k
    out = {
        "mr_psi": mr_psi, "esb_psi": esb_psi, "dsb_in": dsb_in,
        "k_inf_pci": round(k_inf, 1), "chart_read": True,
        "method": ("Figure 3.3 read off the printed chart (curve heights at "
                   "the Dsb gridlines, the turning line and the k diagonals; "
                   "the chart's own construction)"),
        "reference": FIG33_REF,
    }
    if notes:
        out["note"] = " ".join(notes)
    return out


FIG34_REF = "AASHTO 1993 Guide, Figure 3.4 (pdf_page 129, printed II-40)"

# Figure 3.4, left half: the three Dsg curves as (MR psi, height in scan
# pixels, downward), thinned to within 0.5 px of the traced ink. The frame
# runs from 1910.5 (bottom) to 464.5 (top).
_DSG_CURVES = {
    2: ((20007, 1782.2), (19142, 1773.9), (17570, 1755.8), (17354, 1754.3),
        (16450, 1742.1), (15723, 1730.8), (15566, 1729.5), (15035, 1721.3),
        (14466, 1711.5), (13110, 1685.3), (13012, 1684.2), (12265, 1667.9),
        (10968, 1637.9), (9868, 1607.5), (9200, 1587.4), (8433, 1561.9),
        (7883, 1540.7), (7137, 1508.9), (6567, 1482.2), (6210, 1462.1),
        (5661, 1427.0), (4979, 1376.0), (4696, 1352.0), (4377, 1322.0),
        (3850, 1265.0), (3796, 1261.0), (3609, 1239.0), (3340, 1201.0),
        (3208, 1178.0), (3077, 1149.0), (3011, 1139.0), (2649, 1038.0),
        (2590, 1028.0), (2519, 1009.0), (2247, 925.0), (2176, 909.0),
        (1852, 795.0), (1698, 728.0), (1634, 693.0), (1557, 671.0),
        (1553, 663.0), (1391, 581.8), (1326, 560.0), (1169, 471.0)),
    5: ((20007, 1861.5), (18828, 1852.2), (17590, 1841.2), (17492, 1841.4),
        (16548, 1833.0), (15664, 1823.6), (15566, 1823.9), (15428, 1822.5),
        (14643, 1813.4), (13110, 1792.9), (12992, 1792.3), (12245, 1780.9),
        (11184, 1762.8), (10968, 1760.0), (8964, 1721.2), (8650, 1713.8),
        (8473, 1710.5), (7785, 1692.9), (7431, 1682.4), (6410, 1649.4),
        (5899, 1629.6), (5408, 1608.5), (4877, 1583.7), (4531, 1563.9),
        (3932, 1524.0), (3506, 1491.0), (3345, 1477.0), (3026, 1444.0),
        (2533, 1382.0), (2438, 1372.0), (2168, 1332.0), (1840, 1268.0),
        (1785, 1261.0), (1708, 1245.0), (1474, 1183.0), (1377, 1150.0),
        (1340, 1145.0), (1239, 1107.0), (1181, 1074.0), (1146, 1064.0),
        (1143, 1057.0), (1104, 1043.0), (1099, 1034.0), (1073, 1028.0)),
    10: ((20007, 1906.1), (19083, 1904.6), (18238, 1901.1), (17727, 1898.1),
         (17590, 1897.6), (17472, 1898.0), (16725, 1893.9), (15664, 1886.8),
         (15487, 1887.0), (14760, 1880.9), (13601, 1869.9), (13110, 1864.2),
         (12953, 1863.7), (10575, 1835.1), (9180, 1813.3), (8198, 1796.0),
         (7490, 1781.2), (6724, 1763.6), (6488, 1759.1), (5702, 1738.6),
         (5290, 1725.9), (4897, 1711.7), (4366, 1689.3), (3855, 1663.5),
         (3437, 1638.0), (3040, 1610.0), (2762, 1587.0), (2534, 1566.0),
         (2301, 1542.0), (1809, 1482.0), (1582, 1445.0), (1455, 1421.0),
         (1281, 1383.0), (1206, 1371.0), (1083, 1338.0)),
}
# Right half: each drawn k_inf line by the k it reads at the frame's bottom
# and at its top (pci on the k scale; the lines that leave through the
# right side extended to the top).
_FIG34_K_INF = (50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400)
_FIG34_K_BOTTOM = (17.6, 40.4, 108.6, 199.0, 296.7, 407.9, 514.6, 636.4,
                   772.0, 1053.0, 1355.7, 1666.2)
_FIG34_K_TOP = (76.1, 228.1, 620.9, 1115.2, 1629.0, 2215.4, 2838.8, 3512.0,
                4173.0, 5710.2, 6959.2, 8321.8)
_FIG34_Y = (1910.5, 464.5)
FIG34_MR_RANGE = (1169.0, 20_000.0)  # where all three curves are drawn
FIG34_K_MAX = 2000.0                 # the k scale's right end


def k_on_rigid_foundation(mr_psi, dsg_ft, k_inf_pci) -> dict:
    """k on a rigid foundation within 10 ft, from Figure 3.4 as printed.

    Read at the scan resolution (GM-303): the three Dsg curves (2, 5, 10
    ft) traced on the left half, the twelve k_inf lines (50-1,400 pci)
    fitted on the right. The chart's construction: up from MR to the Dsg
    curve, across to the k_inf line, down to k. Dsg between curves is
    linear in Dsg; k_inf between lines is linear in k_inf. Checked against
    the figure's example (MR 4,000, Dsg 5, k_inf 230 -> 300; read 302) and
    Table 3.3 (pdf_page 132, printed II-43; all Dsg 5: 160 -> 230 read 223;
    410 -> 540 read 537; 1,100 -> 1,350 read 1,377), within 3 %.

    Dsg of 10 ft or more: no correction (II-37, step (4): bedrock "within
    10 feet of the surface of the subgrade"). Outside the drawn curves, the
    nearest curve or none, with a note; each choice understates k:

    - Dsg below 2 ft uses the 2-ft curve;
    - MR above 20,000 psi or below 1,169 psi (where the 2-ft curve leaves
      the top) uses the curve's end;
    - k_inf outside the drawn lines (50-1,400 pci) applies no correction;
    - a k past the scale's 2,000 pci end is held at 2,000.

    Returns {'mr_psi', 'dsg_ft', 'k_inf_pci', 'k_pci', 'chart_read',
    'method', 'reference', 'note'?}.
    """
    for name, val in (("mr_psi", mr_psi), ("dsg_ft", dsg_ft),
                      ("k_inf_pci", k_inf_pci)):
        if val is None or val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    out = {"mr_psi": mr_psi, "dsg_ft": dsg_ft, "k_inf_pci": k_inf_pci,
           "chart_read": True, "reference": FIG34_REF,
           "method": ("Figure 3.4 read off the printed chart (the Dsg curves "
                      "traced, the k_inf lines fitted; the chart's own "
                      "construction)")}
    if dsg_ft >= 10:
        out["k_pci"] = round(k_inf_pci, 1)
        out["note"] = ("Depth to the rigid foundation is 10 ft or more; no "
                       "correction (II-37, step (4)).")
        return out
    if not _FIG34_K_INF[0] <= k_inf_pci <= _FIG34_K_INF[-1]:
        out["k_pci"] = round(k_inf_pci, 1)
        out["note"] = (f"k_inf {k_inf_pci:,.0f} pci is outside Figure 3.4's "
                       "drawn lines (50-1,400 pci); no correction applied "
                       "(understates k).")
        return out
    notes = []
    dsg = dsg_ft
    if dsg < 2:
        dsg = 2.0
        notes.append(f"Dsg {dsg_ft} ft is shallower than the 2-ft curve, "
                     "the shallowest drawn; the 2-ft curve was used "
                     "(understates k).")
    mr = mr_psi
    if not FIG34_MR_RANGE[0] <= mr <= FIG34_MR_RANGE[1]:
        mr = min(max(mr, FIG34_MR_RANGE[0]), FIG34_MR_RANGE[1])
        notes.append(f"MR {mr_psi:,.0f} psi is past the drawn curves; read "
                     f"at {mr:,.0f} psi (understates k).")

    def height(d):
        pts = _DSG_CURVES[d][::-1]          # MR ascending
        return float(np.interp(mr, [p[0] for p in pts], [p[1] for p in pts]))
    lo, hi = (2, 5) if dsg <= 5 else (5, 10)
    y = height(lo) + (height(hi) - height(lo)) * (dsg - lo) / (hi - lo)
    frac = (_FIG34_Y[0] - y) / (_FIG34_Y[0] - _FIG34_Y[1])
    k_lines = [b + (t - b) * frac for b, t in zip(_FIG34_K_BOTTOM,
                                                  _FIG34_K_TOP)]
    k = float(np.interp(k_inf_pci, _FIG34_K_INF, k_lines))
    if k > FIG34_K_MAX:
        k = FIG34_K_MAX
        notes.append("The construction lands past the k scale's 2,000 pci "
                     "end; k held at 2,000 pci.")
    out["k_pci"] = round(k, 1)
    if notes:
        out["note"] = " ".join(notes)
    return out


def effective_k(seasonal, slab_d_in, dsb_in=None, esb_psi=None, dsg_ft=None,
                ls=0.0) -> dict:
    """Table 3.2 worksheet (pdf_page 127, printed II-38) with Figure 3.3 as
    printed: seasonal composite k -> rigid-foundation correction (Figure
    3.4, Dsg < 10 ft) -> relative damage u_r (Figure 3.5) -> average u_r ->
    effective k -> loss-of-support correction (Figure 3.6).

    Same arguments and result keys as the reference layer's
    ``composite_k.effective_modulus_subgrade_reaction``; the Figure 3.3 and
    3.4 steps differ (``composite_k_inf``, ``k_on_rigid_foundation``). A
    season with no subbase uses k = MR/19.4 (Section 3.2.1, printed II-44).
    """
    if _ck is None:  # pragma: no cover - depends on refs version
        raise NotImplementedError(
            "The composite-k worksheet needs "
            "geotech_references.aashto_1993.composite_k (Figures 3.4-3.6)."
        )
    if not seasonal:
        raise ValueError("seasonal must be a non-empty list of per-period dicts")
    if slab_d_in <= 0:
        raise ValueError(f"slab_d_in must be > 0, got {slab_d_in}")

    has_subbase = dsb_in is not None and dsb_in > 0
    rows = []
    notes = []
    for i, period in enumerate(seasonal):
        if "mr_psi" not in period:
            raise ValueError(f"seasonal[{i}] is missing the required key 'mr_psi'")
        mr = period["mr_psi"]
        if mr <= 0:
            raise ValueError(f"seasonal[{i}]['mr_psi'] must be > 0, got {mr}")
        period_esb = period.get("esb_psi", esb_psi)

        row = {"period": i, "mr_psi": mr}
        if has_subbase and period_esb is not None and period_esb > 0:
            comp = composite_k_inf(mr, period_esb, dsb_in)
            k_composite = comp["k_inf_pci"]
            row["esb_psi"] = period_esb
            row["composite_k_pci"] = k_composite
            if "note" in comp:
                row["note"] = comp["note"]
                if comp["note"] not in notes:
                    notes.append(comp["note"])
        else:
            simple = _eq.modulus_subgrade_reaction_simple(mr)
            k_composite = simple["k_pci"]
            row["composite_k_pci"] = k_composite
            row["note"] = "no subbase (or esb_psi not given): k = MR/19.4"

        if dsg_ft is not None and dsg_ft < 10:
            rf = k_on_rigid_foundation(mr, dsg_ft, k_composite)
            k_for_ur = rf["k_pci"]
            row["k_rigid_foundation_pci"] = k_for_ur
            if "note" in rf:
                row["note"] = (row["note"] + " " + rf["note"]
                               if "note" in row else rf["note"])
                if rf["note"] not in notes:
                    notes.append(rf["note"])
        else:
            k_for_ur = k_composite

        row["u_r"] = _ck.relative_damage_rigid(slab_d_in, k_for_ur)["u_r"]
        rows.append(row)

    n = len(rows)
    ur_sum = sum(r["u_r"] for r in rows)
    ur_avg = ur_sum / n
    # Inverse of the Figure 3.5 relation used for u_r.
    base = slab_d_in ** 0.75 - (100 * ur_avg) ** (1 / 3.42)
    if base <= 0:
        raise ValueError(
            f"Average relative damage (u_r={ur_avg:.4f}) is too large for "
            f"slab_d_in={slab_d_in} -- the back-solved effective k would be "
            "non-physical. Increase slab_d_in or check the seasonal inputs."
        )
    k_effective = (base / 0.39) ** 4
    ls_result = _ck.k_loss_of_support(k_effective, ls)

    out = {
        "rows": rows,
        "n_periods": n,
        "ur_sum": round(ur_sum, 4),
        "ur_avg": round(ur_avg, 4),
        "effective_k_pci": round(k_effective, 1),
        "ls": ls,
        "k_corrected_for_loss_of_support_pci": ls_result["k_corrected_pci"],
        "slab_d_in": slab_d_in, "dsb_in": dsb_in, "esb_psi": esb_psi,
        "dsg_ft": dsg_ft,
        "equation": ("Table 3.2 worksheet: per-period composite k (Fig. 3.3 "
                     "read off the printed chart, or MR/19.4 with no subbase) "
                     "-> rigid-foundation correction (Fig. 3.4 read off the "
                     "printed chart, if dsg_ft<10) "
                     "-> relative damage u_r (Fig. 3.5) -> average u_r -> "
                     "back-solved effective k -> loss-of-support correction "
                     "(Fig. 3.6)"),
        "reference": "AASHTO 1993 Guide, Table 3.2 (pdf_page 127, printed II-38)",
        "figure_3_3_reference": FIG33_REF,
    }
    if dsg_ft is not None and dsg_ft < 10:
        out["figure_3_4_reference"] = FIG34_REF
    if notes:
        out["notes"] = notes
    return out
