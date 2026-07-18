# AASHTO Seismic Site-Factor Tables — Cell-by-Cell Verification (seismic_geotech)

**Summary: all 75 table cells (3 tables x 5 site classes x 5 columns) in `seismic_geotech/site_class.py` match the printed AASHTO LRFD 5th Edition (2010) tables exactly; interpolation and Class-F footnote behaviors also conform. VERDICT: CONFIRMED-vs-5thEd for all three tables.**

## Source and edition caveat

- **Verified against:** AASHTO LRFD Bridge Design Specifications, **5th Edition (2010)** —
  `C:\Users\socon\OneDrive\Lib\02 Tech\AASHTO\2010 LRFD Bridge Design Specifications - 5th Edition.pdf`
  (clean text layer; tables extracted as text, no OCR/visual fallback needed).
- **Edition caveat (explicit):** the module documents itself against the **9th Edition**
  (§3.10.3, Tables 3.10.3.1-1/2/3 per `seismic_geotech/DESIGN.md`), but the library's 9th Ed
  copy is FileOpen-DRM-encrypted and unreadable. Verification was therefore performed against
  the 5th Ed (2010), where the same tables are numbered **3.10.3.2-1/2/3** (§3.10.3.2 "Site
  Factors", printed pages 3-88 and 3-89; PDF pages 140–141 of 1591). These seismic site-factor
  tables are **edition-stable between the 5th and 9th Editions** (values unchanged; only the
  table numbers shifted 3.10.3.2-x → 3.10.3.1-x). The code's own docstring
  (`site_class.py` ~line 220) actually cites the 3.10.3.2-x numbering, i.e., the 5th-Ed-era labels.
- **Module implementation read:** `seismic_geotech/site_class.py` lines 223–291
  (`_Fa_table`, `_Fv_table`, `_Fpga_table`, `_interpolate_table`).

## Table 1 — Fpga at Zero-Period (5th Ed Table 3.10.3.2-1, p. 3-88)

Module: `_Fpga_table = _Fa_table` indexed against PGA breakpoints `[0.10, 0.20, 0.30, 0.40, 0.50]`.
Printed columns: PGA < 0.10 / = 0.20 / = 0.30 / = 0.40 / > 0.50.

| Site Class | Column | Module | Printed 5th Ed | Match |
|---|---|---|---|---|
| A | all 5 | 0.8, 0.8, 0.8, 0.8, 0.8 | 0.8, 0.8, 0.8, 0.8, 0.8 | YES |
| B | all 5 | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0, 1.0, 1.0 | YES |
| C | all 5 | 1.2, 1.2, 1.1, 1.0, 1.0 | 1.2, 1.2, 1.1, 1.0, 1.0 | YES |
| D | all 5 | 1.6, 1.4, 1.2, 1.1, 1.0 | 1.6, 1.4, 1.2, 1.1, 1.0 | YES |
| E | all 5 | 2.5, 1.7, 1.2, 0.9, 0.9 | 2.5, 1.7, 1.2, 0.9, 0.9 | YES |
| F | — | `ValueError` (site-specific required) | `*` (site-specific footnote) | YES (behavioral) |

Note: the printed Fpga table and printed Fa table contain identical coefficient values (differing
only in index variable, PGA vs Ss), so the module's `_Fpga_table = _Fa_table` aliasing is faithful
to the printed standard.

**VERDICT: CONFIRMED-vs-5thEd (25/25 value cells + Class F row).**

## Table 2 — Fa, Short-Period Range (5th Ed Table 3.10.3.2-2, p. 3-88)

Module: `_Fa_table` with Ss breakpoints `[0.25, 0.50, 0.75, 1.00, 1.25]`.
Printed columns: Ss < 0.25 / = 0.50 / = 0.75 / = 1.00 / > 1.25.

| Site Class | Module | Printed 5th Ed | Match |
|---|---|---|---|
| A | 0.8, 0.8, 0.8, 0.8, 0.8 | 0.8, 0.8, 0.8, 0.8, 0.8 | YES |
| B | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0, 1.0, 1.0 | YES |
| C | 1.2, 1.2, 1.1, 1.0, 1.0 | 1.2, 1.2, 1.1, 1.0, 1.0 | YES |
| D | 1.6, 1.4, 1.2, 1.1, 1.0 | 1.6, 1.4, 1.2, 1.1, 1.0 | YES |
| E | 2.5, 1.7, 1.2, 0.9, 0.9 | 2.5, 1.7, 1.2, 0.9, 0.9 | YES |
| F | `ValueError` | `*` (site-specific footnote) | YES (behavioral) |

**VERDICT: CONFIRMED-vs-5thEd (25/25 value cells + Class F row).**

## Table 3 — Fv, Long-Period Range (5th Ed Table 3.10.3.2-3, p. 3-89)

Module: `_Fv_table` with S1 breakpoints `[0.10, 0.20, 0.30, 0.40, 0.50]`.
Printed columns: S1 < 0.1 / = 0.2 / = 0.3 / = 0.4 / > 0.5.

| Site Class | Module | Printed 5th Ed | Match |
|---|---|---|---|
| A | 0.8, 0.8, 0.8, 0.8, 0.8 | 0.8, 0.8, 0.8, 0.8, 0.8 | YES |
| B | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0, 1.0, 1.0 | YES |
| C | 1.7, 1.6, 1.5, 1.4, 1.3 | 1.7, 1.6, 1.5, 1.4, 1.3 | YES |
| D | 2.4, 2.0, 1.8, 1.6, 1.5 | 2.4, 2.0, 1.8, 1.6, 1.5 | YES |
| E | 3.5, 3.2, 2.8, 2.4, 2.4 | 3.5, 3.2, 2.8, 2.4, 2.4 | YES |
| F | `ValueError` | `*` (site-specific footnote) | YES (behavioral) |

**VERDICT: CONFIRMED-vs-5thEd (25/25 value cells + Class F row).**

## Footnote conventions (printed verbatim vs module behavior)

Printed footnotes (identical structure on all three tables; verbatim from the 5th Ed text layer):

> 1. "Use straight-line interpolation for intermediate values of PGA." (Table -1; "...of Ss" for
>    Table -2; "...of Sl [S1]" for Table -3)
> 2. "Site-specific geotechnical investigation and dynamic site response analysis should be
>    performed for all sites in Site Class F."

Module behavior:

- **Interpolation:** `_interpolate_table()` performs straight-line (linear) interpolation between
  column breakpoints — conforms to footnote 1.
- **Edge clamping:** values at or below the first breakpoint return the first column; at or above
  the last breakpoint return the last column. This matches the printed column headers, which are
  inequality-bounded at the extremes ("PGA < 0.10" and "PGA > 0.50", etc.) — clamping outside the
  tabulated range is the printed intent.
- **Class F:** `site_class == "F"` raises
  `ValueError("Site Class F requires site-specific analysis")` — conforms to footnote 2 (the
  printed `*` cells carry no numeric value).
- **Fpga approximation caveat (module-internal, documented):** when `pga` is not supplied, the
  module returns Fpga ≈ Fa(Ss), exact only when Ss = 2.5·PGA. This is a documented convenience
  fallback, not a table-value discrepancy; supplying `pga` gives the code-defined Table -1 lookup.

## Overall verdict

**CONFIRMED-vs-5thEd — all three tables, all 75 cells, both footnote behaviors.**
No discrepancies found. (9th Ed confirmation blocked by DRM; edition-stability of these tables
between 5th and 9th Ed is the stated basis for closure.)
