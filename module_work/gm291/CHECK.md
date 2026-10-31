# GM-291: ground_improvement against GEC-13 as printed

Checked 2026-10-31 against FHWA GEC-13, *Ground Modification Methods
Reference Manual* (FHWA-NHI-16-027), Vols. I and II, read from the PDFs page
by page, and UFC 3-220-10, *Soil Mechanics* (2026), Sec. 5-7.4 (pp. 266
to 268) where GEC-13 prints no formula. Takes in GM-279 (the
settlement ratio called n') and GM-233's stone-column items. Every value in
the module now has its page below, or is listed as not printed.

## What changed

### Wick drains and surcharge (GEC-13 Vol. I, Ch. 2, Sec. 4.2)

| Item | Print | Before | Now |
|---|---|---|---|
| Radial consolidation | Eq. 2-1 (Rixner et al. 1986a), p. 2-32: t = (d_c^2/8c_h)[F(n) + F_s + F_r] ln[1/(1 - U_h)] | same form, cited to Barron/Hansbo and NHI-06-019 | unchanged form; cited to Eq. 2-1 |
| F(n) | Eq. 2-2, "simplified": ln(d_c/d_w) - 0.75 | same | same |
| Smear term | GEC-13 prints no formula: smear "often ignored for typical projects"; "a sample disturbance factor F_s ≈ 2 may be considered" in highly plastic, sensitive soils (p. 2-35). UFC Eq. 5-25: F_s ≈ (k_h/k_s) ln(s); "If smear and well resistance are ignored, Fs and Fr are set equal to zero" (p. 267) | Hansbo's ln(n/s) + (k_h/k_s) ln(s) - 0.75, from memory, printed in neither | F(n) + (k_h/k_s) ln(s), UFC Eq. 5-25 as printed |
| Smear default | ignored (both manuals) | smear_ratio 2.0, k_h/k_s 2.0 | smear_ratio 1.0 (no smear); the adapter declares a left-out smear_ratio as "smear ignored (F_s = 0)", and k_h/k_s applies only with a smear ratio given |
| Well resistance | "rarely significant, and, therefore, ignored except for extremely deep drains or where a combination of high loads and very permeable soils is present" (p. 2-36) | not modelled; the manual said "with smear and well resistance" | not modelled, said so in the tool, steps and manual |
| Drain diameter | Eq. 2-3, d_w = 2(a + b)/π; "the most common is to use 2.5 inch" (p. 2-35) | default 0.066 m (100 x 4 mm band, unprinted) | default 0.0635 m (2.5 in) |
| d_c | 1.05 s triangular, 1.13 s square (p. 2-33) | same | same, cited |
| Combined U | Carrillo (1942), UFC Eq. 5-27, p. 268. GEC-13: vertical drainage "should be considered for major/complex projects" (p. 2-33); "for a conservative approach, no beneficial effects may be considered for vertical drainage in preliminary feasibility evaluations" (p. 2-34) | combined, cited "Carrillo (1942)" | combined, cited UFC Eq. 5-27; U_r reported beside it with p. 2-34's words |
| Surcharge | no equation printed; "There are three basic variables ... time, PVD spacing, and surcharge" (p. 2-36) | same code | drain defaults as above; references fixed |

Printed example pinned (`ground_improvement/tests/test_gec13_print.py`):
GEC-13 Sec. 4.4 (p. 2-38), 20 ft of clay, c_v 0.1 ft²/day, c_h = 2 c_v, a
2.5 in drain, triangular, 90 percent: "on the order of 300 days at an
approximate spacing of 8 feet or 500 days at an approximate spacing of 10
feet". The tool gives 299.2 and 503.0 days (n 40.32, F 2.947; n 50.4, F
3.170), and on its new defaults U_r = 90.0 % at 299.2 days. V-020 and
WE-GI-1 pinned the same example with explicit inputs and are unchanged.

### Vibro-compaction (GEC-13 Vol. I, Ch. 4)

| Item | Print | Before | Now |
|---|---|---|---|
| Fines | "proper densification generally cannot be achieved when the granular soil contains more than 12 to 15 percent silt OR more than 2 percent clay" (p. 4-65); zone B "ideally suited for vibro-compaction with fines content below 10 percent" (p. 4-67); Table 4-5 (p. 4-66): SM inhibited above 8 % silt, SC above 2 % clay | ≤10 favourable, 10-15 reduced, 15-20 marginal, >20 not feasible (unprinted) | ≤10 ideally suited, 10-12 feasible, 12-15 marginal (the printed range, read as marginal: the module's reading), >15 not feasible. The clay fraction is not an input; every result says so |
| D50 | no D50 limit printed (gradation by Fig. 4-27 zones) | <0.1 mm not feasible, <0.2 marginal | removed; the adapter refuses D50 with the reason |
| SPT N | Table 4-8 (p. 4-74), N to the relative density of a normally consolidated sand; "It is unlikely, even for heavy loading, that it will be necessary to achieve a relative density above 85 percent" (p. 4-75) | N >25 not feasible, 20-25 moderate (unprinted) | N read by Table 4-8; very dense (N >50, D_r 85-100 %) not feasible |
| Probe spacing | Fig. 4-34 (p. 4-77), "Relative density versus probe spacing for silty sands", the lower-bound gradation (p. 4-76); criteria 60 % (slabs, tanks, embankments), 70-75 % (column and bridge footings), 80 % (machinery, mats) (p. 4-77) | improvement ratio target N / initial N, 1.5 → 3.0 m to 4.0 → 1.5 m, fines factor (a rule of thumb the module called "NOT tied to a specific figure/table and NOT benchmarked"); target_N_spt default 25 | spacing off Fig. 4-34 by `target_relative_density`, 55 to 90 %, refused outside; no target, no spacing (the result says to give one). target_N_spt refused with the reason |
| Pattern | a square pattern "requires 5 to 8 percent more points" (p. 4-77) | not said | said with a square pattern; Fig. 4-34 names no pattern |

Fig. 4-34 read: the figure is an embedded chart image (674 x 591 px, no
vector layer). The grid lines were found and fitted (x within 0.0015 m, y
within 0.05 %), the curve's centre column read on every pixel row clear of a
grid line (269 rows, 89.6 % at 1.411 m to 55.5 % at 2.475 m), and a line
fitted through the rows within 1.5 % of each 5 % grid line. Reads drawn back
over the figure sit on the curve. Nodes (D_r % → m): 55 → 2.50, 60 → 2.30,
65 → 2.13, 70 → 1.95, 75 → 1.80, 80 → 1.67, 85 → 1.52, 90 → 1.40
(`vibro.FIG_4_34`; the script is `trace_fig4_34.py` here).

### Aggregate columns (GEC-13 Vol. I, Ch. 5)

- GM-279: the settlement ratio of Fig. 5-21 is printed as "n" in quotes,
  "defined as the ratio of settlement without stone columns to that with
  stone columns" (p. 5-36); it is not Eq. 5-2's stress ratio n. The module,
  results, calc steps, figure, adapter, manual, review checklist, reviewer
  agent, DESIGN.md, validation notes and tests now say "settlement ratio"
  where they said n'. The result key `settlement_ratio` is unchanged.
- Stress ratio n: default 5.0 kept. The print: measured "between 2.0 and
  5.0" for stone columns (p. 5-31); "For preliminary design, a ratio of 2.5
  is often conservatively used for stability and bearing resistance
  calculations" (p. 5-32); rammed aggregate piers 5 to 10 under flexible
  embankment loading, 10 under rigid footings (p. 5-43); "stress
  concentration ratios range from 4 to 45 for rigid footings" and 5 to 10
  "for settlement control of embankments" (p. 5-45). The tool text now
  quotes all of it (it left out p. 5-32's preliminary-design ratio).
  Which value is the default is an owner question (below).
- Settlement methods (stress reduction, Fig. 5-21 curves): unchanged (GM-220,
  GM-276). 171 recorded results identical before and after.
- GM-233, stone columns: WE-GI-2's phi_c 39 was resolved by GM-276 (WE-GI-2
  calls `priebe_improvement_factor` with as_ratio 0.277 only: 2.548 against
  the printed 2.7; no phi_c remains except in the refused-parameter lists).
  The A/A_c scan was settled by GM-276 on both printed copies: PRIEBE_CURVE_4
  and BALAAM_FLEXIBLE_10 agree from A/A_c 5.0 and are within 0.04 at 4.0 to
  4.5, where the printed (4) and (5a) run on one line.
- Example 1 (rammed aggregate piers, pp. 5-46, 5-47; q_g 6,383 psf, s_uz
  about 0.7 in, as V-018 records them): the two-layer stiffness-modulus method (Eqs. 5-11 to 5-13) is
  not in the module, as V-018 and the manual say. Example 2 (pp. 5-48,
  5-49): pinned since GM-276.

### Feasibility screening

| Item | Print | Before | Now |
|---|---|---|---|
| Aggregate columns, strength | "most significant improvement ... 300 to 1000 psf" (p. 5-9, item 2); "Special care must be taken ... organics and peat ... less than 300 psf" (item 3); "a practical upper limit is in the range of an undrained strength of 1000 to 2000 psf for stone columns" (item 5) | applicable for c_u 10-100 kPa; organic c_u <10 kPa not applicable (unprinted); a clay under 10 kPa fell to "too stiff/dense" | not applicable above 2000 psf (96 kPa); noted within 1000-2000 psf (pre-boring) and below 300 psf; organic below 300 psf applicable with p. 5-9's special-care sentence |
| Typical sizes | stone columns 1.5-4 ft, typically 3.0-3.6 ft dry; spacing 5-12 ft (p. 5-31); n as above; PVD spacing 3-5 ft homogeneous clays, 4-6 ft silty clays, 5-6.5 ft coarser soils (p. 2-33); d_w 1.5-5.5 in, most commonly 2.5 in (p. 2-35) | piers 0.45-0.9 m, 1.5-3.0 m, n 3-8; drains 1.0-2.5 m, 0.05-0.07 m; surcharge 1-5 m of fill, 18-20 kN/m³ (all unprinted) | the printed values; surcharge: "GEC-13 prints no typical surcharge height" with p. 2-36's sentence |
| Vibro | as Ch. 4 above | fines >20 not applicable, >15 marginal; N >25 not applicable; spacing 1.5-3.5 m, target N 20-30 | fines >15 not applicable, 12-15 marginal; N very dense not applicable; Fig. 4-34's 1.40-2.50 m and p. 4-77's criteria |

## Results before and after

Recorded on every ground_improvement call the targeted tests make (267
records before, 284 after; `scratchpad gm291/rec`). Unchanged: every
aggregate pier result (39), settlement ratio (102), Priebe curve (30),
area ratio, composite modulus, bearing. Moved:

| Case (made-up inputs) | Before | After |
|---|---|---|
| wick_drains, s 1.5 m, c_h 3, c_v 1 m²/yr, H_dr 5 m, t 0.5 yr, defaults | F 3.116, U_r 78.8 %, U 82.2 % | F 2.461, U_r 86.0 %, U 88.2 % |
| design_drain_spacing, 90 % in 0.5 yr, same soil | 1.30 m | 1.46 m |
| design_drain_spacing, 90 % in 0.25 yr | 1.00 m (the range's bound, 88.1 %) | 1.08 m (90.1 %) |
| surcharge_preloading with drains at 1.5 m, 90 % | 0.675 yr | 0.540 yr |
| vibro_compaction, FC 8 %, N 12, no target | spacing 2.65 m (target N 25 default) | no spacing; asks for target_relative_density (70 % gives 1.95 m) |
| feasibility, organic, c_u 8 kPa | aggregate piers not applicable | applicable, with p. 5-9's special-care sentence |
| feasibility sizing text | unprinted ranges | printed ranges (every run) |

## Noted, not fixed (no result changes, or no printed basis to change to)

- `design_drain_spacing` returns the end of `spacing_range` when the target
  is not met, its comment says "with warning" and none is given; the result
  shows the U reached.
- Surcharge: σ'_p = σ'_v0 + q is the module's (GEC-13 p. 2-36 says only
  that the surcharge overconsolidates the soil); the "tight schedule" note
  under 6 months is unprinted text.
- Well resistance (UFC Eq. 5-26) and a direct F_s input (GEC-13's F_s ≈ 2)
  are not offered; GEC-13 ignores the first for typical drains.
- Feasibility: the aggregate-pier screen N_spt < 15 for a granular soil is
  not printed in GEC-13 Ch. 5 and is kept (no printed screen to put there).
- Vibro: the 2 % clay limit and Table 4-5's SM 8 % silt limit need inputs
  the tool does not take; each result says so.

## Owner question

- Aggregate piers' stress ratio n defaults to 5.0: the top of the measured
  range for stone columns and the bottom of 5 to 10 for rammed aggregate
  piers under embankments. GEC-13 p. 5-32: "For preliminary design, a
  ratio of 2.5 is often conservatively used for stability and bearing
  resistance calculations" (stone columns). Keep 5.0, use 2.5, or ask for
  the column type? At a_s 0.2, 2.5 gives SRF 0.77 where 5.0 gives 0.56.

## Gates (targeted; at ≥1000 MB free, one at a time)

- ground_improvement/tests + validation_examples/test_published_v018_v020.py:
  138 passed, 1 skipped.
- calc_package/tests + the funhouse adapter, parity, defaults, GM-179/220/221,
  aliases and top-level files, `-k` ground improvement: 375 passed.
- funhouse_agent/tests `-k` improvement, vibro, wick, priebe: 73 passed.
- test_defaults_used, test_gm179_unused_inputs, test_top_level_params,
  test_render_tables, test_calc_package_figure_layout (the method probes),
  `-k` ground improvement and every-default: 546 passed, 14 skipped.
- validation_examples `-k` improvement/GI/ground: 3 passed.
- foundry_test_harness `-k` ground improvement: 10 passed.
- test_worked_examples: passed; test_reviewers: 93 passed.
- test_citations: 1 failure, not this change: the main checkout's
  references (newer than this branch's base) carry GEC-10 Eq. 9-13, which
  master took off the gap list in 506aba1.
- webapp/tests/test_page_fingerprints.py: 15 passed (no page carries the
  changed tool text; no re-record).
- test_public_text on every changed file: clean.
- Not run: the full suites.
