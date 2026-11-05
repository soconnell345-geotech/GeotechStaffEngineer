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
     Table 4-2 (p. 4-4); the live load as in the module's critical
     combination. Basis for checking it: p. 4-23, "Maximum permanent loads,
     minimum permanent loads, and total extremes should be checked for
     complex ... walls" and "for each strength limit load group, e must be
     less than emax".
   - It is counted in `eL_governing_m` and `CDR_governing`. On a level
     backfill it never governs (the critical combination's vertical loads
     under a smaller overturning moment: smaller e, wider L', lower
     sigma_v, qR no lower), so no level answer moves. On a sloping backfill
     EH also scales the thrust's vertical component, so it could govern; in
     the GEC-11 Example E1/E3 validation cases (V-056/057) it does not, and
     their pins pass unchanged.
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
| FOS overturning | M_stab / M_ot | required 2.0, the module's ASD value (App. F.1, p. F-2); p. 4-22 "(formerly overturning)" |
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

## What is left

- A sloping or broken backslope in the package: the module's calc steps
  print a level backfill only (GM-112); printing the GEC-11 Sec. 4.4.4
  backslope terms is the work.
- A battered face (10 deg or more): Eq. 4-26 (p. 4-36) and the failure
  surface for a battered face (p. 4-33, AASHTO Fig. 11.10.6.3.1-1) are not
  built in `mse_wall` at all.
- The package is ASD (`mse_wall`); there is no LRFD package. The
  `mse_lrfd_external_stability` result, Strength I min included, is not
  printed in any package.

## Release note (three lines)

- New: the MSE wall calc package — `retaining_wall_package` with `wall_type="mse"` runs `mse_wall` and prints each step with its GEC-11 page; level backfill and vertical face only (a sloping backfill or a face batter is refused).
- `mse_lrfd_external_stability` now reports Strength I min (EV 1.00, EH 0.90; GEC-11 Table 4-2) eccentricity and bearing, counted in the governing values; on a level backfill no answer changes.
- The MSE calc steps now cite printed pages for the active force, the moments, the factors of safety, the bearing eccentricity and Kr/Ka.
