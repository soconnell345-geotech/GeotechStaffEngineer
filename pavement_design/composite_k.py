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

The worksheet (Table 3.2, pdf_page 127, printed II-38) then follows the
reference layer step by step, with Figure 3.4 (rigid foundation), Figure
3.5 (relative damage) and Figure 3.6 (loss of support) from
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


def effective_k(seasonal, slab_d_in, dsb_in=None, esb_psi=None, dsg_ft=None,
                ls=0.0) -> dict:
    """Table 3.2 worksheet (pdf_page 127, printed II-38) with Figure 3.3 as
    printed: seasonal composite k -> rigid-foundation correction (Figure
    3.4, Dsg < 10 ft) -> relative damage u_r (Figure 3.5) -> average u_r ->
    effective k -> loss-of-support correction (Figure 3.6).

    Same arguments and result keys as the reference layer's
    ``composite_k.effective_modulus_subgrade_reaction``; only the Figure
    3.3 step differs (``composite_k_inf``). A season with no subbase uses
    k = MR/19.4 (Section 3.2.1, printed II-44).
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
            rf = _ck.k_rigid_foundation_correction(mr, dsg_ft, k_composite)
            k_for_ur = rf["k_pci"]
            row["k_rigid_foundation_pci"] = k_for_ur
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
                     "-> rigid-foundation correction (Fig. 3.4, if dsg_ft<10) "
                     "-> relative damage u_r (Fig. 3.5) -> average u_r -> "
                     "back-solved effective k -> loss-of-support correction "
                     "(Fig. 3.6)"),
        "reference": "AASHTO 1993 Guide, Table 3.2 (pdf_page 127, printed II-38)",
        "figure_3_3_reference": FIG33_REF,
    }
    if notes:
        out["notes"] = notes
    return out
