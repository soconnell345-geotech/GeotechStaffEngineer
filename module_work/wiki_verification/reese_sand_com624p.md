# Reese Sand p-y Verification vs COM624P Manual (FHWA-SA-91-048)

**SUMMARY (final, 2026-07-18): B_static and the k tables check out; A_static is plausibly
within chart read-off at shallow depth but runs HIGH at mid-depth; the module's
_REESE_A_CYCLIC and _REESE_B_CYCLIC tables are a documented reconstruction that does NOT
match the manual's Figs 3.12/3.13 (manual A_c is a low near-vertical curve ~0.9-1.1 with
deep value 0.88 shared with A_s; manual B_c starts ~0.5, humps to ~0.85, ends 0.55) — and
the audited "A_c -> 0.55 at depth" asymptote CONTRADICTS the manual, which prints
"x/b > 5.0, A = 0.88" for BOTH loadings on Fig 3.12; 0.55 is the *B_c* asymptote.**

---

## Source and page map

- Source PDF: `C:\Users\socon\OneDrive\Lib\02 Tech\FhWA\Lateral Loading and COM624P\COM624P Users Manual.pdf`
  (COM624P User's Manual, Wang & Reese, FHWA-SA-91-048, 1993). Scanned; NO text layer on
  these pages — all values read visually from 200-450 dpi renders.
- Printed-page <-> PDF-page offset: printed 340 = PDF 362 (offset +22, confirmed on-page).
- Section: Part III "RECOMMENDATIONS FOR p-y CURVES FOR SAND" (Reese, Cox & Koop 1974),
  printed pp. 340-347 = PDF 362-369.

| Item | Printed page | PDF page | Note |
|---|---|---|---|
| Section start, Mustang Island background | 340-341 | 362-363 | phi=39 deg, gamma'=66 pcf |
| Eqs 3.30-3.32 (Ko=0.4, pst, psd) | 341 | 363 | |
| Fig 3.11 characteristic p-y curve family (yk, ym=b/60, yu=3b/80) | 342 | 364 | |
| Steps 4-8; Eq 3.33 pu=A*ps (A from Fig 3.12); Eq 3.34 pm=B*ps (B from Fig 3.13); Eq 3.35 p=(kx)y, k from Tables 3.4/3.5 | 343 | 365 | |
| **Figure 3.12** — coefficients A_c and A_s vs x/b | 344 | 366 | Curves only, no printed table |
| **Figure 3.13** — coefficient B (B_c, B_s) vs x/b | 345 | 367 | Curves only, no printed table |
| **Tables 3.4/3.5** — representative k for submerged / above-WT sand | 346 | 368 | Printed numbers |
| Eqs 3.36-3.39 parabola fit | 346 | 368 | |
| Eq 3.40 yk; sand above WT (Parker & Reese 1971) | 347 | 369 | |

**Figure-numbering note:** the task brief (and module comments/DESIGN.md) cite
"COM624P Figs 2.19/2.20" — in this manual the A/B coefficient charts are actually
**Figures 3.12 and 3.13** (printed pp. 344-345), and the k tables are **Tables 3.4 and
3.5** (printed p. 346), not "Tables 2.1/2.2". The module docstring for
`_sand_k_recommendation` cites "Table 2.2" — stale/incorrect citation.

## What the manual actually provides

- NO printed numeric table exists for A or B — Figs 3.12/3.13 are hand-drawn curves with
  discrete circle markers (open = static, filled = cyclic) at x/b intervals of ~0.5.
  Comparison is therefore curve read-off (tolerance taken as +/-0.05-0.10 in the
  coefficient, per the scan quality and line thickness).
- Printed asymptote annotations (exact text):
  - Fig 3.12: "x/b > 5.0, A = 0.88" (single annotation — applies to the merged
    static+cyclic curve at depth; both curves visibly merge by x/b ~ 3.5-4).
  - Fig 3.13: "x/b > 5.0, B_c = 0.55, B_s = 0.5".
- Tables 3.4/3.5 print k by relative density only (Loose/Medium/Dense), static and
  cyclic alike, in lb/in^3. No friction-angle parameterization appears in the manual.

## Method for curve read-off

Figures 3.12/3.13 rendered at 200 dpi; axes calibrated from the frame + printed tick/label
positions (OpenCV); data-point markers located automatically (open circles = static, via
white-interior contour detection; filled dots = cyclic, via erosion blob detection); curve
ordinates cross-checked by scanning dark-pixel runs at each grid row. Calibration
self-checks: A-axis ticks read back as 0.97/2.00; B-axis ticks as 1.00/2.00; deep
asymptotes read 0.87-0.88 (A) and 0.49/0.54 (B_s/B_c) vs printed 0.88 and 0.5/0.55.
Adopted read-off tolerance: **+/-0.05 nominal, +/-0.10 outer bound** (hand-drawn 1993
scan, line width ~0.03 in coefficient units).

## 1. A coefficients — module `_REESE_A_STATIC` / `_REESE_A_CYCLIC` vs Figure 3.12 (printed p. 344, PDF 366)

Module grid `_REESE_AB_ZB = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]`
(`lateral_pile/py_curves.py` lines 703-707).

### A_s (static)

| x/b | Module | Manual Fig 3.12 read | Diff | Verdict |
|---|---|---|---|---|
| 0 | 2.90 | ~2.85 (curve exits top frame) | +0.05 | CONFIRMED-WITHIN-READOFF |
| 0.5 | 2.60 | 2.50-2.58 | +0.02..+0.10 | CONFIRMED-WITHIN-READOFF |
| 1.0 | 2.25 | 2.11-2.15 (marker) | +0.12 | DISCREPANCY (marginal) |
| 1.5 | 1.90 | 1.77 (marker ring center) | +0.13 | DISCREPANCY (marginal) |
| 2.0 | 1.62 | 1.48 (marker) | +0.14 | DISCREPANCY |
| 2.5 | 1.42 | 1.24 (marker ring center) | +0.18 | DISCREPANCY |
| 3.0 | 1.28 | 1.04 (marker) | +0.24 | DISCREPANCY |
| 4.0 | 1.02 | 0.88 (already at asymptote) | +0.14 | DISCREPANCY |
| 5.0 | 0.88 | 0.88 (printed annotation) | 0.00 | CONFIRMED-PRIMARY |

Pattern: the module's A_s endpoints are right, but the descent is too slow — the manual
curve reaches the 0.88 asymptote by x/b ~ 3.7, while the module holds 1.02 at 4.0. The
mid-depth ordinates are HIGH by 0.12-0.24, i.e. beyond the outer read-off bound for
x/b = 2.0-4.0. (Consequence is conservative-side pu inflation of ~10-25% at those depths
in the `construction="reese1974"` branch only; the default "simplified" branch does not
use these tables.)

### A_c (cyclic) — MAJOR DISCREPANCY

Manual Fig 3.12 A_c (dashed, filled dots), as measured:

| x/b | Manual A_c read | Module `_REESE_A_CYCLIC` |
|---|---|---|
| 0 | ~0.73-0.76 (curve exits top frame) | 2.90 |
| 0.5 | 0.91 (dot) | 2.55 |
| 1.0 | 1.06 (dot) | 2.20 |
| 1.5 | 1.08 (dot) | 1.85 |
| 2.0 | 1.01 (dot) | 1.55 |
| 2.5 | 0.97 (dot) | 1.30 |
| 3.0 | ~0.93 | 1.05 |
| 4.0 | ~0.88 (merged with static curve) | 0.80 |
| >= 5.0 | **0.88 (printed: "x/b > 5.0, A = 0.88")** | **0.55** |

The module's cyclic A table bears no resemblance to the manual chart at ANY depth. The
manual A_c is a low, near-vertical curve: ~0.73 at the surface, a gentle hump to ~1.08
at x/b ~ 1.5, settling to the SAME deep value as static, A = 0.88 (Fig 3.12 prints a
single asymptote annotation for the chart). Verdict: **DISCREPANCY (major — reconstructed,
not digitized)**. DESIGN.md's own Source-basis note admits this table was "corrected" at
v5.3 to a fabricated 2.90 -> 0.55 descent to enforce A >= B; the premise was wrong (see
Section 3).

## 2. B coefficients — module `_REESE_B_STATIC` / `_REESE_B_CYCLIC` vs Figure 3.13 (printed p. 345, PDF 367)

### B_s (static)

| x/b | Module | Manual Fig 3.13 read | Diff | Verdict |
|---|---|---|---|---|
| 0 | 2.20 | ~2.2 (2.12 at x/b=0.15, extrapolated) | ~0.00 | CONFIRMED-WITHIN-READOFF |
| 0.5 | 1.85 | 1.85 (marker) | 0.00 | CONFIRMED-WITHIN-READOFF |
| 1.0 | 1.55 | 1.56 (marker ring center) | -0.01 | CONFIRMED-WITHIN-READOFF |
| 1.5 | 1.28 | 1.25-1.26 (marker) | +0.02 | CONFIRMED-WITHIN-READOFF |
| 2.0 | 1.05 | 1.04-1.05 (marker ring center) | 0.00 | CONFIRMED-WITHIN-READOFF |
| 2.5 | 0.90 | 0.88 (marker ring center) | +0.02 | CONFIRMED-WITHIN-READOFF |
| 3.0 | 0.80 | ~0.70 | +0.10 | DISCREPANCY (marginal, at outer bound) |
| 4.0 | 0.62 | ~0.54 | +0.08 | CONFIRMED-WITHIN-READOFF (outer bound) |
| 5.0 | 0.50 | 0.50 (printed: "B_s = 0.5") | 0.00 | CONFIRMED-PRIMARY |

B_STATIC is clearly a genuine digitization of this chart — five marker points match to
+/-0.02. Same "too-slow descent" pattern as A_s appears mildly at x/b 3-4.

### B_c (cyclic) — MAJOR DISCREPANCY

| x/b | Manual B_c read | Module `_REESE_B_CYCLIC` |
|---|---|---|
| 0 | ~0.50 (curve exits top frame) | 2.20 |
| 0.5 | 0.71 (dot) | 1.90 |
| 1.0 | 0.84 (dot) | 1.62 |
| 1.5 | 0.86 (dot) | 1.35 |
| 2.0 | 0.82 (dot) | 1.12 |
| 2.5 | 0.78 (dot) | 0.95 |
| 3.0 | ~0.69 | 0.80 |
| 4.0 | ~0.56 | 0.65 |
| >= 5.0 | **0.55 (printed: "B_c = 0.55")** | **0.55** |

Manual B_c: ~0.5 at surface, humps to ~0.86 at x/b ~ 1.5, decays to 0.55. The module
descends monotonically from 2.20. Only the deep endpoint (0.55) matches. Verdict:
**DISCREPANCY (major — reconstructed, not digitized)**.

## 3. A_c deep-asymptote check (explicit, per audit)

- Audit/module claim: A_c approaches **0.55** at depth (`py_curves.py` line 698 comment
  "A_c=0.55", `_REESE_A_CYCLIC` endpoint 0.55, DESIGN.md line 67).
- Manual: Figure 3.12 prints **"x/b > 5.0, A = 0.88"** — one annotation covering the
  chart; the static and cyclic curves visibly merge by x/b ~ 3.5-4 and share the 0.88
  deep value. **0.55 is the deep value of B_c** (Figure 3.13: "x/b > 5.0, B_c = 0.55,
  B_s = 0.5").
- Verdict: **DISCREPANCY**. The module's A_c asymptote conflates the A-chart with the
  B-chart cyclic asymptote.
- Corollary: the v5.3 "A >= B" correction rationale collapses. In the manual, A_c >= B_c
  holds everywhere ALREADY (0.73 vs 0.50 at surface; 1.08 vs 0.86 mid; 0.88 vs 0.55
  deep) — with correctly digitized charts no fabricated descent is needed, and the
  cyclic ultimate/m-point do NOT coincide at depth (pu = 0.88*ps > pm = 0.55*ps).

## 4. Soil-modulus k for sand — module `_sand_k_recommendation` vs Tables 3.4/3.5 (printed p. 346, PDF 368)

Printed values (both tables are headed "Static and Cyclic Loading"; keyed by RELATIVE
DENSITY only — the manual has NO friction-angle parameterization):

- Table 3.4, submerged sand: Loose **20**, Medium **60**, Dense **125** lb/in^3
  (= 5,429 / 16,287 / 33,931 kN/m^3 at 1 lb/in^3 = 271.45 kN/m^3).
- Table 3.5, above water table: Loose **25**, Medium **90**, Dense **225** lb/in^3
  (= 6,786 / 24,430 / 61,076 kN/m^3).

Module (`py_curves.py` lines 676-681, phi-keyed at [25, 30, 35, 40] deg):

| Branch | Module kN/m^3 | As lb/in^3 | Manual anchor | Verdict |
|---|---|---|---|---|
| above WT, phi=25 | 6,800 | 25.1 | Loose 25 | CONFIRMED-WITHIN-READOFF (conversion exact to 0.2%) |
| above WT, phi=30 | 24,000 | 88.4 | Medium 90 | CONFIRMED-WITHIN-READOFF (-1.8%) |
| above WT, phi=35 | 61,000 | 224.7 | Dense 225 | CONFIRMED-WITHIN-READOFF (-0.1%) |
| above WT, phi=40 | 170,000 | 626 | (none) | NOT-FOUND in manual (extrapolation from another source) |
| below WT, phi=25 | 5,400 | 19.9 | Loose 20 | CONFIRMED-WITHIN-READOFF (-0.5%) |
| below WT, phi=30 | 11,000 | 40.5 | Medium 60 = 16,287 | DISCREPANCY (-32%) under the above-WT phi mapping |
| below WT, phi=35 | 22,000 | 81.0 | Dense 125 = 33,931 | DISCREPANCY (-35%) under the above-WT phi mapping |
| below WT, phi=40 | 45,000 | 165.8 | (none) | NOT-FOUND in manual |

Notes: (a) the above-WT branch implies the density->phi mapping Loose=25 / Medium=30 /
Dense=35; the below-WT branch does NOT reproduce the printed submerged anchors under
that same mapping (its interpolant passes through 60 lb/in^3 at phi ~ 32.4 and
125 lb/in^3 at phi ~ 37.4 instead) — the two branches embed inconsistent phi mappings.
The below-WT numbers resemble the continuous k-vs-phi chart used by API RP 2A / LPILE
rather than this manual's Table 3.4. (b) The docstring cites "COM624P Manual Table 2.2"
— the actual sources are **Tables 3.4 and 3.5**; citation is stale/incorrect. (c) phi
keying itself is an interpretation layered on the manual (acceptable engineering
convenience, but not primary-source).

## Verdict summary

| Item | Verdict |
|---|---|
| A_s surface (2.90) + deep asymptote (0.88 at z/b>=5) | CONFIRMED-PRIMARY (asymptote printed) |
| A_s ordinates z/b 0-0.5 | CONFIRMED-WITHIN-READOFF |
| A_s ordinates z/b 1.0-4.0 | DISCREPANCY (high by +0.12..+0.24; too-slow descent) |
| A_c entire table incl. 0.55 deep asymptote | DISCREPANCY (major; manual A_c ~ 0.73 -> 1.08 -> 0.88; 0.55 is B_c's asymptote) |
| B_s ordinates z/b 0-2.5 and 5.0 | CONFIRMED-WITHIN-READOFF / CONFIRMED-PRIMARY (0.5 printed) |
| B_s ordinates z/b 3.0-4.0 | marginal (+0.08..+0.10, at outer bound) |
| B_c entire table except deep endpoint | DISCREPANCY (major; manual B_c ~ 0.50 -> 0.86 -> 0.55) |
| B_c deep endpoint 0.55 | CONFIRMED-PRIMARY (printed) |
| k above-WT anchors (three densities) | CONFIRMED-WITHIN-READOFF (unit-converted printed values) |
| k below-WT medium/dense | DISCREPANCY (inconsistent phi mapping vs above-WT branch; -32%/-35% vs printed anchors) |
| k phi=40 values (both branches) | NOT-FOUND (beyond the manual's tables) |
| Module/docstring figure+table citations ("Figs 2.19/2.20", "Table 2.2") | DISCREPANCY (actual: Figs 3.12/3.13, Tables 3.4/3.5) |

Scope note: these tables feed only the opt-in `construction="reese1974"` branch of
`SandReese` (default remains "simplified", which is chart-free) and
`_sand_k_recommendation`. No code was modified; verification only.

