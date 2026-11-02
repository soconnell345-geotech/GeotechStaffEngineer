# pavement_design: the charts and tables GM-298 left, as printed

BACKLOG GM-303, 2026-10-31 to 2026-11-02. Read from the PDFs in
`geotech-references/docs/`: UFC 3-250-01 (14 November 2016; PDF page =
printed page + 19) and the AASHTO Guide for Design of Pavement Structures
(1993; a scan, read as page images; page II-n is PDF page 89 + n).
Page numbers are the printed ones.

Scope, as the backlog line lists it: the UFC Appendix E vehicle curves E-2
to E-31 (used by `ufc_mixed_traffic`), AASHTO Figures 2.3/2.4, the Table
3.1 iteration, Figure 3.4's depth scaling, the Appendix D LEF tables other
than D.4, and the Chapter 19 frost charts (Figures 19-5 and 19-6, which the
app does not call: report only). Charts were read off the page image at the
scan's resolution and checked against every worked example printed for
them.

## UFC Figures E-2 to E-31, the vehicle curves (fixed)

- **Print.** Thirty charts (Figure E-n on printed page 191 + n). Each gives
  thickness above the subgrade against CBR, one curve per number of
  passes, for one vehicle and its gross weight. Table G-1 (printed 256-257)
  works a mixed-traffic example on them at CBR 3.
- **Module before.** `ufc_mixed_traffic` used the reference layer's grids.
  Against a trace of the print, those grids are within about 0.05 in
  (median) over CBR 3-30 on most charts. They are off elsewhere:
  - by 2-6 in where a curve has already reached the axis (the knees and
    tails);
  - up to 9.5 in thin near CBR 1 (E-4, E-5; E-8 5.6, E-9 4.2, E-28 3.2);
  - on E-16 throughout (up to 6.8 in thin, 9.8 thick) and on E-30
    throughout (up to 10.9 in thin);
  - E-15's top is 15 in, not 14.

  The library's allowable passes bisect on the thickness rounded to 0.1 in,
  so E-28 at 16.4 in returns exactly 10^5.5 = 316,228.
- **How it was read.** Every curve of every chart was traced on the ink,
  smoothed over 5 samples, its ends followed to where the ink stops or to
  the frame, and thinned to within 0.5 px.
  - A column-by-column check of the result against the ink gives a median
    offset of 0.12-0.19 px and a 95th percentile of 0.51-0.74 px on every
    chart. Every offset over 2 px is at a label gap (checked by eye).
  - No two curves cross.
  - Gross weights and pass labels match each chart's legend.
- **Now.** `pavement_design/ufc_vehicles.py` holds the thirty charts as read
  (`_FIGURES`: refs key, printed caption, gross weight, top of scale,
  curves).
  - `figure_e_vehicle_thickness`: above the top curve it extends in log
    passes, with a note; below the lowest curve it reads that curve, with
    a note.
  - `figure_e_vehicle_allowable_passes`: says when a thickness is above the
    highest curve or below the lowest; refuses a thickness above the scale
    where the curves leave the chart.
  - `ufc_mixed_traffic` uses them and carries their notes.
- **Anchors** (Table G-1, CBR 3):

  | Read | Print | Module |
  |---|---|---|
  | E-28 thickness at 100,000 passes | 15.8 in | 15.62 |
  | E-26 thickness at 500,000 passes | 12.8 in | 12.73 |
  | E-2 thickness at 20,000,000 passes | 6.1 in | 6.115 (extended; the 10,000,000 curve reads 6.00) |
  | E-28 allowable passes at 16.4 in | 252,915 | 337,788 |
  | Mixed-traffic total, 18-kip equivalent | 1,395,400 | 1,296,044 (was 1,316,228) |

  - The E-28 allowable is 312,351-365,297 for 16.35-16.45 in. A direct
    pixel read gives about 342,000. The curves at CBR 3 read 11.80 /
    13.85 / 15.62 / 17.09 / 18.23 in.
  - The total is 7.1 % under G-1. The print says why (p. 256): "Solutions of
    the problems using PCASE software differ slightly from the solutions
    determined through the use of the pavement design charts."
  - PAV-7 is now 1,296,044 (rtol 0.05, 1,395,400 accepted as an
    alternative), and the worked example's `mixed_total` is the same.
- Noted, not fixed:
  - Table G-2 (printed 258) takes the 5-axle truck as the controlling
    vehicle: 362,788 printed for the 18-kip allowable at 15.8 in, where
    Figure E-1 gives 337,249. The module always takes the vehicle needing
    the most thickness.
  - Figure E-10's caption prints M113A1; its legend prints M113A3. The
    module keeps the caption.
  - Figure E-1 refuses more than 10,000,000 passes, while the vehicle charts
    extend, as Table G-1 itself does for E-2.

## AASHTO Figures 2.3 and 2.4, the effective roadbed modulus (no change)

- **Print.** Figure 2.3 (II-14) prints uf = 1.18 x 10^8 x MR^-2.32, and its
  scale agrees: MR 30,000 -> 0.005, MR 1,000 -> 13.0. Figure 2.4's example
  (II-15) reads uf off that scale for each month (0.01 / 1.51 / 0.51 /
  0.13 ...): sum 3.72, average 0.31, MR 5,000 psi.
- **Module.** Follows the printed equation: 0.0124 / 1.544 / 0.519 / 0.142
  ..., sum 3.846, average 0.3205, MR 4,923 psi. That is 1.5 % below the
  example, which comes from the example's rounded chart reads. PAV-8's key
  is 4,923. The reference layer's docstring claims the equation gives 3.72
  (sent to ref04-review).

## AASHTO Table 3.1, the performance-period iteration (fixed)

- **Print.** Step 7 (II-35): if the trial and calculated periods differ by
  more than 1 year, take their average as the next trial. If they differ by
  less than 1 year, "convergence is reached and the average is said to be
  the predicted performance period" (the example: 8.5 and 8.2, "about 8
  years").
  Table H.2 (H-7): one iteration, 13 -> 13.2, "Convergence achieved after
  only one iteration", "about 13 years".
- **Module before.** The reference layer's iteration with its default
  tolerance of 0.1 year, reporting the last calculated period.
- **Now.** `performance.estimate_performance_period` passes `tol=1.0` and
  reports the average of the last row's trial and calculated periods when
  converged, with a note citing Step 7. When not converged it says so and
  reports the last calculated period.

  | Case | Before | Now |
  |---|---|---|
  | V-076, Table H.2 | 13.21 yr | 13.12 (13 and 13.23 in one iteration) |
  | Made-up flexible, swelling and frost | 8.05 | 8.18 |
  | Made-up rigid, swelling | 12.80 | 13.03 |

  The V-076 test now asserts one iteration, the first calculated period
  13.2 and the period 13.1.

## AASHTO Figure 3.4, k on a rigid foundation within 10 ft (fixed)

- **Print.** Figure 3.4 (II-40): up from MR to the Dsg curve (2, 5 and 10
  ft are drawn), across to the k_inf line (50 to 1,400 pci are drawn), down
  to k. Example: MR 4,000, Dsg 5, k_inf 230 -> 300. Table 3.3 (II-43), all
  at Dsg 5 ft: k_inf 1,100 / 160 / 230 / 410 -> 1,350 / 230 / 300 / 540.
  II-37, step (4): the correction applies when bedrock lies "within 10 feet
  of the surface of the subgrade".
- **Module before.** The reference layer's `k_rigid_foundation_correction`
  is exact at the four Table 3.3 rows. Away from them it scales one
  multiplier per MR by a three-point depth guess. On the chart k also
  depends on k_inf: the lines fan out. Against the chart read below, it
  was:
  - 12-36 % low at Dsg 2-3 ft or at twice the MR/19.4 k_inf (MR 4,000,
    Dsg 2, k_inf 412: 589 where the chart gives 914);
  - up to 26 % high at Dsg 7-9 ft (MR 2,500, Dsg 9, k_inf 129: 171 where
    the chart gives 136);
  - 1.21x at Dsg just under 10 ft (MR 4,000), then 1.0 at 10 ft.
- **How it was read.** The page at the scan's resolution (about 305 dpi):
  - The three Dsg curves traced on the ink, thinned to within 0.5 px at the
    95th percentile (1.8 px at worst).
  - The twelve k_inf lines fitted as straight lines (median offset 0.5-1.4
    px).
  - The axes from their gridlines.
  - The example's own arrows (MR 4,000 at x 1199.5) confirm the MR scale.
- **Now.** `composite_k.k_on_rigid_foundation(mr_psi, dsg_ft, k_inf_pci)`
  follows the chart's construction: linear in Dsg between curves, linear in
  k_inf between lines. `effective_k` uses it; the reference layer's
  Figures 3.5 and 3.6 stay. Outside the drawn curves each choice
  understates k, with a note:
  - Dsg under 2 ft reads the 2-ft curve;
  - MR outside 1,169-20,000 psi reads the curve's end;
  - k_inf outside 50-1,400 pci takes no correction;
  - a k past the scale's 2,000 pci is held there.

  Dsg of 10 ft or more is no correction, as before.

  | Read | Print | Module |
  |---|---|---|
  | Example, MR 4,000 / Dsg 5 / k_inf 230 | 300 | 303 |
  | Table 3.3, k_inf 1,100 / 160 / 230 / 410 | 1,350 / 230 / 300 / 540 | 1,378 / 222 / 303 / 539 |
  | Table 3.3 worksheet (the module's own Fig 3.3 reads feed Fig 3.4) | 1,350 / 230 / 300 / 540 | 1,353 / 210 / 304 / 504 |
  | Sum u_r / average / effective k / with LS 1.0 | 7.25 / 0.60 / 540 / 170 | 7.38 / 0.615 / 519 / 164.5 (was 7.36 / 0.613 / 522 / 165) |

  The guide reads one Figure 3.3 point as both 400 and 410, and its reads
  of Figure 3.4 sit within -3.5 to +2.1 % of the chart read. The worksheet
  rows stack two chart reads: March 210 for 230 and June 504 for 540,
  coming from composite k of 154 and 391 against the printed 160 and 410.
  The V-074 tests now check Figure 3.4 alone within 4 %, and the worksheet
  rows within 10 % (u_r 0.03, average 0.02, effective k 5 %).
- **Changes saved runs:** any rigid design given `dsg_ft` under 10.

## AASHTO Appendix D, the LEF tables (no change)

Tables D.1 (flexible single, pt 2.0), D.5 (flexible tandem, 2.5), D.6
(flexible triple, 2.5), D.11 (rigid tandem, 2.0) and D.14 (rigid tandem,
2.5) were compared cell by cell with the scans: 1,500 cells, no
discrepancies. D.4 was checked in GM-298. The app reads them through the
reference layer's `lef.load_equivalency_factor`.

## UFC Figures 19-5 and 19-6, the frost charts (report only)

The app does not call either. Sent to ref04-review.

- **Figure 19-5** (printed 79):
  - All ten r lines (0.6 to 3.0) are straight through the origin, and s =
    b/4 by the two printed axes.
  - Slopes b/c read at c = 40, 50, 70 and 76: r 0.6 0.876, 0.8 0.84, 1.0
    0.80, 1.2 0.77, 1.4 0.74, 1.6 0.716, 1.8 0.692, 2.0 0.668, 2.5 0.613,
    3.0 0.571.
  - The reference layer has only r 2.0 (0.65625, from the example's 21/32)
    and 3.0 (0.595, the average of G-8.9's 26/42 and G-8.11's 22/38.5).
    G-8.11 sits on the r 3.0 line; G-8.9 does not.
- **Figure 19-6** is on printed page 82 (the reference layer says 83). Its
  three lines are straight.
  - At 8 in: F1&S1 92, F2&S2 66, F3&F4 49 psi/in.
  - At 20 in: 221 / 162 / 120.
  - The reference layer has only F3&F4 at 8 in (50).

## Gates (2026-11-02, branch wip/gm303 on master 538dd74, refs a0f2cf4; each run in the foreground, judged by pytest's exit code)

- `pavement_design/tests`: 94 passed, exit 0.
- `validation_examples` V-055 and V-074 to V-076: 22 passed, exit 0.
- funhouse_agent: the pavement adapter, worked examples, eval suite, calc
  package parity, citations, reference adapters, adapter ergonomics,
  stated defaults, unused inputs and top-level params: 802 passed and 2
  skipped, exit 0.
- `webapp/tests/test_public_text.py` and `test_page_fingerprints.py`: 19
  passed, exit 0.
- The adapter's text is unchanged, so `test_defaults_used` was not run. All
  nine pavement method probes in `method_probes.json` run through the
  dispatcher without an error.
- The user manual was not rebuilt.
