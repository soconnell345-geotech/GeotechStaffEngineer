# GM-114: the MSE wall calc package (CHECK)

Branch `claude/gm-114-mse-calc-package`, from master `d45380a`.

## What was built

1. **The MSE calc package, on `mse_wall` itself (GM-141's way).**
   `calc_package.retaining_wall_package` with `wall_type="mse"` (or
   `"mse_wall"`) runs `run_mse_wall`, the tool's own code path, and renders
   the module's existing MSE calc steps (`retaining_walls/calc_steps.py`,
   `_calc_steps_mse`), like every other package: given and assumed inputs
   apart, Values Used, Choices, steps, figures, tables, the defaults the
   tool left in force. The adapter's `not_packaged` refusal ("run the
   mse_wall tool") is gone. The package's parameters are now the union of
   `cantilever_wall` and `mse_wall` (`_union_spec`), as for the other
   two-tool packages.
   - Response: `analysis_type` "MSE Wall", the verdict, `checks_failed`,
     `checks_not_run`, the three FOS, plus `sigma_v_kPa`, `n_levels`,
     `FOS_pullout_min`, `FOS_rupture_min`, `passes_internal` and
     `failed_level_depths_m` (`_mse_numbers`), each the module's number.
2. **Scope: level backfill, vertical face.**
   - A non-zero `backfill_slope` is refused before the run
     (`_mse_package_scope`), saying the package covers a level backfill
     only and that the `mse_wall` tool covers a sloping or broken backslope
     (Coulomb Kab, Eqs. 4-2, 4-3, 4-7). `backfill_slope=0` is taken.
   - `mse_wall` has no face-batter input (it takes a vertical face). The
     names `face_batter`, `wall_batter`, `batter`, `batter_angle`,
     `face_batter_deg` are refused by name (the package's `dropped`), with
     the reason: Ka by Eq. 4-1 (p. 4-15) for a vertical face; GEC-11 Vol I
     p. 4-36 takes a face battered 10 deg or more by Eq. 4-26's simplified
     Coulomb Ka, which is not built. A value of 0 is taken as "not used".
3. **LRFD Strength I min, apart.** `check_external_stability_lrfd` (the
   `mse_lrfd_external_stability` tool) gave the Strength I min combination
   (EV min, EH min) for sliding only. It now also gives, level and sloping
   backfill:
   - `eccentricity.eL_strength_min_m` and `strength_min_basis`;
   - `bearing.sigma_v_strength_min_kPa`, `eL_strength_min_m`,
     `B_eff_strength_min_m`, `CDR_strength_min` (against the given
     `bearing_resistance_strength`, or with `gamma_foundation` a qn by Eq.
     4-22 over its own L', `qn_strength_min_kPa`, `qR_strength_min_kPa`).
   - Factors: `gamma_EV_min` (1.00) and `gamma_EH_min` (0.90), GEC-11 Vol I
     Table 4-2 (p. 4-4); the live load at `gamma_LL` (1.75) on its thrust
     and, in bearing, over the reinforced zone, as GEC-11 Vol II Table
     E4-5.1 (p. E4-6) lists Strength I (minimum) (EV 1.00, EH 0.90, LL
     1.75) and Table E4-6.3 (p. E4-9) works it: ΣV 65.57 = 57.69 + 1.75 ×
     4.50 k/ft, σv 4.92 ksf. Basis for checking it: p. 4-23, "Maximum
     permanent loads, minimum permanent loads, and total extremes should be
     checked for complex ... walls" and "for each strength limit load
     group, e must be less than emax"; for bearing p. 4-28, "a critical
     value must be determined by comparisons of applicable load
     combinations".
   - Pinned to the printed Str I (min) columns: E4 (V-009) eL 2.65 ft
     (Table E4-6.2, p. E4-8); bearing eL 2.34 ft, B' 13.33 ft, σv 4.92 ksf,
     CDR 2.13 (Table E4-6.3, p. E4-9); the module 2.657, 2.338, 13.324,
     4.921, 2.134. E3 (V-056) eL 2.35 ft (Table E3-6.2, p. E3-9); bearing
     B' 19.31 ft, σv 6.83 ksf, CDR 1.54 (Table E3-6.3, p. E3-11); the
     module 2.344, 19.311, 6.826, 1.538.
   - It is counted in `eL_governing_m` and `CDR_governing`. On a level
     backfill it never governs: its vertical loads are the critical
     combination's (review fix 3 below) under a smaller overturning moment,
     so smaller e, wider L', lower σv and qR no lower; no level answer
     moves by it. On a sloping backfill EH also scales the thrust's
     vertical component, so it could govern; in the GEC-11 Example E1/E3
     validation cases (V-056/057) it does not.
   - The ASD `mse_wall` path is untouched (it has no load factors).

## The steps and their pages (GEC-11, FHWA-NHI-10-024 Vol I unless said)

Pages read off the printed page footers in the library's `GEC 11 Vol 1.pdf`.

| Step | Equation | Page cited |
|---|---|---|
| Rankine Ka (retained fill) | Ka = tan²(45 - φb/2) | Eq. 4-1, p. 4-15 |
| Total active force | Pa = ½Ka γb H² + Ka q H | Eqs. 4-5, 4-6, p. 4-20; Fig. 4-2 |
| Sliding resistance (drained) | R = V tan δb, δb = min(φr, φf) | Eq. 4-12, p. 4-21; μ p. 4-22 |
| Sliding resistance (undrained) | R = min(cu L, V tan φr) | p. 4-21 Step 3; UFC 3-220-20 p. 223; GEC-6 Eq. 5-39 (EN 1997-1 Eq. 6.5 with the gap) |
| Sliding along the lowest sheet | FOS_s = Vs tan ρ / Pa,s | p. 4-21 Step 3(iii), Eq. 3-9 (p. 3-21); Vol II p. E1-7 |
| FOS sliding | R / Pa (or least surface) | Eq. 4-12 (p. 4-21) over Eqs. 4-5, 4-6 (p. 4-20); required 1.5, the module's ASD value (Vol II App. F.1, p. F-2) |
| Overturning moment | Σ P z | Eqs. 4-5, 4-6 (p. 4-20), arms H/3, H/2 as in Eq. 4-19 (p. 4-27) |
| Stabilizing moment | Σ W x | Fig. 4-7 (p. 4-24); live load over the zone left out, p. 4-22 |
| FOS overturning | M_stab / M_ot | required 2.0, the module's ASD value (App. F.1, p. F-2); p. 4-18 "Limiting eccentricity (formerly known as overturning)", checked on pp. 4-22, 4-23 |
| Bearing eccentricity, σv | eB, σv = Vb / (L - 2eB) | Eq. 4-18 (p. 4-26), Eqs. 4-19, 4-20 (p. 4-27), unfactored |
| FOS bearing | q_allowable / σv | Eq. 4-17 (p. 4-26), allowable in place of qR |
| Local shear | γr H ≤ 3 cu | Eq. 4-24, p. 4-30 |
| Kr/Ka | Fig. 4-10 curves | Fig. 4-10, p. 4-37 |
| Tmax | σH Sv | Eq. 4-32a, pp. 4-46, 4-47 (Fig. 4-14) |
| F* | empirical or tested | Eqs. 3-4, 3-5 (p. 3-20); Table 3-6 note (i), p. 3-17 |
| Pullout | Pr = F* α σv Le C Rc | Eq. 3-2 (p. 3-16), Eq. 4-35 (p. 4-49) |
| Tensile resistance | Tal × Rc | Eq. 3-11a (p. 3-22), p. 4-47, Eqs. 4-33, 4-34 (pp. 4-48, 4-49) |

Before this branch, Pa ("GEC-11, Ch. 4"), the two moments, the FOS
sliding / overturning / bearing, the eccentricity and Kr/Ka had no page;
now each has one. The ASD required FOS (1.5, 2.0) have no printed source:
the step says they are the module's own values and cites where GEC-11 says
it does not update ASD.

## Tests (made-up values only)

- `funhouse_agent/tests/test_gm114_mse_calc_package.py` (12): the package
  builds (HTML); its numbers equal `analyze_mse_wall` called directly; it
  renders the tool's own result; every CalcStep cites a page (geogrid and
  steel-strip walls); the HTML carries the pages; no `q_allowable` →
  bearing NOT CHECKED; sloping backfill and face batter refused; 0 taken;
  `wall_type="mse_wall"` works; the cantilever package unchanged.
- `retaining_walls/tests/test_gm114_lrfd_strength_min.py` (7): eL and
  bearing for Strength I min against a hand calculation; never governs on
  a level backfill; qn over its own width; no CDR without a resistance;
  counted in governing with a sloping backfill; the factors given are used.
- `funhouse_agent/tests/test_calc_package_tool_parity.py`: two MSE cases
  in CASES (tool numbers, tool defaults), one in MODULE_CASES (module
  numbers); the "dropped" test now checks the batter refusal.

Run (this environment, Python 3.11): `retaining_walls`, `calc_package`, the
calc-package adapter tests, the MSE adapter tests, `validation_examples`,
`webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
**1626 passed, 3 skipped, 1 xfailed, 1 failed**. The failure,
`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`,
fails the same way with this branch's changes stashed (an environment
matter, pile-group figure lettering; not touched here). Two
`test_gm179_unused_inputs.py` liquepy cases fail because liquepy does not
install here (eqsig wheel build). The page fingerprints did not move.

## Review fixes (REVIEW_GM-114)

1. **Strength I (min) bearing: the live load over the zone at 1.75.** It
   took `gamma_EV_min` (1.00) on Vs; GEC-11 Vol II Table E4-5.1 (p. E4-6)
   lists Strength I (minimum) at LL 1.75 and Table E4-6.3 (p. E4-9) works
   it so. Level (`_bearing(gamma_EV_min, gamma_LL, MOA_min)`) and sloping
   (`vs=gamma_LL`) fixed; `STRENGTH_MIN_BASIS` says it and cites p. 4-28
   for the bearing comparison. E4 moves from σv 4.758 ksf, CDR 2.207 to
   4.921 ksf, 2.134 (printed 4.92, 2.13); E3 has no live load and does not
   move. The hand test takes SV = 1.00 V1 + 1.75 Vs; E4's and E3's Str I
   (min) columns are pinned (section 3), replacing V-056's "given as the
   maximum factors" workaround.
2. **"Formerly overturning" page.** The words are on p. 4-18, "Limiting
   eccentricity (formerly known as overturning)"; the FOS overturning step
   now cites p. 4-18 with that quote, "checked on pp. 4-22, 4-23" (the
   table above too).
3. **The critical-combination bearing had the same slip (pre-existing).**
   It put the live load over the zone at `gamma_EV_min` (1.00). The print
   is unambiguous that it is 1.75:
   - GEC-11 Vol II Table E4-6.3 (p. E4-9), critical values: ΣV_C 65.57*
     and M_RA-C 590.09*, Str I (min)'s column (65.57 = 57.69 + 1.75 ×
     4.50 k/ft; 590.09 = 519.21 + 1.75 × 40.5), with Str I (max)'s M_OA-C
     223.30: eL 3.41 ft, B' 11.19 ft, σv-c 5.86 ksf, CDR 1.79.
   - Table E5-7.3 (p. E5-20) does the same: V_Bb-C 133.35 is Str I
     (min)'s "V = V4+V5+VS" (116.52 + 1.75 × 9.62, Table E5-4.3, p. E5-9),
     with Str I (max)'s M_OB 571.95.
   - E3 (Table E3-6.3) and E1 have no live load over the zone (E1 Vs = 0),
     so they cannot tell the two apart.

   Fixed: level `_bearing(gamma_EV_min, gamma_LL)`, sloping `vs=gamma_LL`;
   the docstring, DESIGN.md and the comments say it. V-009 now pins the
   critical σv 5.86 ksf (abs 0.01, was rel 2.5 %) and CDR 1.79, and the
   hand calculation ΣV_C, M_RA-C, eL 3.41 ft; the "source rounding" note is
   gone from the test, WE-MSE-1 and RESULTS.md.

4. **`test_defaults_used` on the two-tool package.** With `mse_wall` in
   the package, `_union_spec` drops a case the tools word differently
   (`q_allowable`: cantilever "the bearing check is not run", MSE "the
   external bearing check is not run", both true; `phi_foundation`:
   cantilever takes `phi_backfill`, MSE "30 deg, assumed"), leaving it to
   the run, which lists the case of the tool it ran (`_package_declared`).
   The harness read cases from the union only, so it gave the case string
   back as a value (TypeError) and flagged `q_allowable` as stating no
   case. Fixed in the harness, no test weakened: a package's inputs are
   read as the package reads them, its own and then those of the tool the
   call selects (`_run_tool_parameters`, with `calc_package._selector_value`
   taken out of `_run_tool` so both read the selector the same way). Two
   MSE probes added (`retaining_wall_package#mse`, `#mse_bearing`), so the
   harness runs the MSE package too.

## Release note for review fix 3 (a saved answer moves)

`mse_lrfd_external_stability` (and `analyze_mse_wall(lrfd_external=True)`):
the critical (EV min, EH max) Strength I bearing now carries the live load
over the reinforced zone at LL 1.75, as GEC-11 Vol II Table E4-6.3 does; it
took 1.00 before. Only walls with a live load over the zone move (level
backfill with a surcharge, or a broken backslope whose level ground reaches
over the zone); sloping or broken backslopes with no traffic over the zone
(E1, E3) do not move. What moves, re-run:

| Case | Key | Before | After | Printed |
|---|---|---|---|---|
| GEC-11 E4 / WE-MSE-1 (qR given 10.50 ksf) | `sigma_v_critical_kPa` | 275.41 (5.75 ksf) | 280.75 (5.86 ksf), +1.9 % | 5.86 ksf |
| | `CDR_critical` | 1.825 | 1.791 | 1.79 |
| | `CDR_governing` | 1.566 (Str I max) | 1.566, unchanged | 1.57 |
| E4 wall with `gamma_foundation` (qR by Eq. 4-22) | `sigma_v_critical_kPa` | 275.41 | 280.75 | n/a |
| | `B_eff_critical_m`, `qR_critical_kPa` | 3.2954, 471.13 | 3.4082, 487.25 | n/a |
| | `CDR_critical` | 1.711 | 1.736 | n/a |
| | `CDR_governing` | 1.711 (critical) | 1.735 (Str I max now governs) | n/a |
| E1 (both traffic settings, and with `gamma_foundation`), E3 / WE-MSE-2 | every key | — | unchanged | — |

The bearing demand σv-c rises wherever a live load is over the zone. With
a given qR the critical CDR falls by the same ratio. With qR worked out
from L' (`gamma_foundation`) the wider L' raises qR more than σv rises, so
that CDR rises; on E4 the governing bearing CDR goes from 1.711 to 1.735,
and Strength I max governs in place of the critical combination. No
sliding or eccentricity value moves, and no pass or fail changes in these
cases.

## What is left

- A sloping or broken backslope in the package: the module's calc steps
  print a level backfill only (GM-112); printing the GEC-11 Sec. 4.4.4
  backslope terms is the work.
- A battered face (10 deg or more): Eq. 4-26 (p. 4-36) and the failure
  surface for a battered face (p. 4-33, AASHTO Fig. 11.10.6.3.1-1) are not
  built in `mse_wall` at all; nor is the external Ka for it, Eqs. 4-2 and
  4-3 with θ (p. 4-15), which `_MSE_BATTER_REASON` does not name yet.
- The package is ASD (`mse_wall`); there is no LRFD package. The
  `mse_lrfd_external_stability` result, Strength I min included, is not
  printed in any package.
- The user manual (`docs/user_manual/manual.html`) is regenerated at
  merge. (The gallery's Exhibit 9 figure, which labels the E4 critical
  bearing CDR, is rebuilt: 1.82 before review fix 3, 1.79 now.)

## Release note (four lines)

- New: the MSE wall calc package — `retaining_wall_package` with `wall_type="mse"` runs `mse_wall` and prints each step with its GEC-11 page; level backfill and vertical face only (a sloping backfill or a face batter is refused).
- `mse_lrfd_external_stability` now reports Strength I min (EV 1.00, EH 0.90, LL 1.75; GEC-11 Vol I Table 4-2, Vol II Table E4-5.1) eccentricity and bearing, counted in the governing values; on a level backfill no answer changes.
- The MSE calc steps now cite printed pages for the active force, the moments, the factors of safety, the bearing eccentricity and Kr/Ka.
- Fix: `mse_lrfd_external_stability`'s critical-combination bearing now carries the live load over the reinforced zone at LL 1.75, as GEC-11 Vol II Table E4-6.3 does (it took 1.00). Where a live load is over the zone the critical bearing stress rises: GEC-11 Example E4 goes from 5.75 to 5.86 ksf (printed 5.86), CDR 1.825 to 1.79 (printed 1.79); with `gamma_foundation` its CDR rises (wider L'), 1.711 to 1.736, and Strength I max governs at 1.735. Walls with no live load over the zone do not change.
