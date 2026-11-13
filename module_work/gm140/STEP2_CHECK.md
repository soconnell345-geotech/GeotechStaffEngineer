# GM-140 step 2: a method's own solution, or no number

Branch `claude/gm-140-step2-own-solution-or-no-number`, from master 2cc2c45
(step 1 merged; `STEP1_CHECK.md` is on master). 2026-11-13.

**The rule.** A factor of safety reported under a method's name is that
method's own converged solution on that surface. Where the method has no
solution there is no number: `FOS` is None, `method_solved` is False and the
result says why. It holds for one surface, every search family, the yield
acceleration, FOSM, Monte Carlo and the reliability wrapper.

## 1. What the print says

- UFC 3-220-10 p. 362: "Critical failure surfaces can be missed as a result
  of non-convergence. The engineer should be able to identify when
  convergence issues are present." Read from the reference library's text of
  the page.
- EM 1110-2-1902 C-10d, pp. C-37 to C-38: ways to avoid non-convergence
  (reject the unrealistic one of multiple solutions, a sensible starting F,
  a realistic initial surface). Not in a repository: taken from the account's
  and the brief's summary; nothing was coded from it beyond the citation.
- The methods' own equations, as the account cites them and as the module's
  ledger checked them against the papers: Bishop (1955) Eq. 13, p. 10;
  Spencer (1967) Eqs. 7 and 8, p. 15 (the moment and force factors of safety
  agree at the solution); Morgenstern and Price (1965) Eq. 18, p. 85.
- No printed rule was found for a probabilistic run that meets a realization
  with no solution, or for a yield coefficient whose bisection meets one. The
  module's rule for each is stated in the result as its own (section 6).

No formula, default, coefficient or tolerance was added. The solver's
settings the account could not find in print (D29: the lambda range of 1.5,
the 0.001 agreement, the 0.05 m_alpha clamp, the fitted moment axis) are as
they were.

## 2. The departures, reproduced

`slope_stability/tests/test_own_solution_or_no_number.py` (22 tests, made-up
geometries, each docstring with the rule and its page). On master the file
does not import (the rule's solver module does not exist), so the master
behaviour of each departure was measured by script
(`step2_section11.py` and scratch probes) and is given here:

| Departure | Master | Branch |
|---|---|---|
| D23/D30, GM-139. A circle with its centre below the crest of a made-up phi = 0 slope, Spencer / M-P / GLE | 17.68 from the old approximate solver, labelled (about 0.05 on GM-139's circle) | no number; Bishop's simplified method solves it, 17.68 under its own name |
| D23. A Spencer circular search (Slide2 #85 without its support, 6 x 6) | 267 circles scored by the old solver, no warning | none; 329 trials counted without a solution, warned |
| D27. Bishop's iteration stopped at its limit | the last iterate, with a Python warning | `NoSolution`; `FOS` None in a result; not scored in a search |
| D49. No driving moment (a stabilizing-pile row whose force exceeds it) | `FOS` 999.9 | no number, "no net driving moment" |
| D49. FOSM, mean plus one standard deviation has no solution | the approximate solver's value (or 999.9) taken as F | no result; the error names `c_prime = 50` and why |
| D49. Monte Carlo, some realizations without a solution | their approximate values in the sample | counted, left out, pf bounded both ways |
| D49. `method="bishops"` | ran Bishop's simplified method | refused with the six names, in nine entry points |
| D45. `f_interslice="sine"` in a comparison or the report package | accepted | refused with the four names |
| D30. Yield acceleration by Spencer | the approximate solver at any trial kh without a rigorous solution | Spencer's own solution at every kh, or no ky |
| Item 9. A polyline scored in a search | could be a state about a second moment axis | `analyze_slope`'s FOS on it to 1e-9, or not scored |

## 3. What changed, by the brief's items

1. **The old approximate solvers are deleted** (`spencer_fos_legacy`,
   `morgenstern_price_fos_legacy`, about 425 lines) and gone from the package's
   exports. Consumers outside the module: none (grep of the whole repository:
   only `slope_stability/__init__.py`, `methods.py`, `DESIGN.md`, this
   folder's notes and the audit, whose row 6 this closes). `solve_rigorous`
   and `RigorousSolution` went with them; `solution.solve_method` and
   `MethodSolution` replace them.
2. **Circles and polylines alike.** `search._compute_fos` scores every trial
   with `solve_method`; a trial without a solution is not scored, with or
   without supports. `_reinforced_rigorous_fos` and
   `_rigorous_noncircular_fos` are gone.
3. **The lambda search** (`gle.gle_fos`): GM-60's robust search is the one
   search for every run. At a fixed lambda, an iteration that ran its 80
   iterations without settling is solved by the direct march, and where that
   finds nothing there is no state (before: the last iterate was returned as
   a state, flagged by nobody). Measured, section 5.
4. **Bishop and Janbu.** `bishop_fos` raises `NoSolution` when not within its
   tolerance after 50 iterations; `janbu_fos` when its force-equilibrium
   iteration has no state. In a result `FOS` is None; in a search the trial is
   not scored.
5. **999.9.** `fellenius_fos` / `bishop_fos` / `janbu_fos` raise `NoSolution`
   where they returned it; `solve_method` also refuses a ratio at or above it
   (the driving sum zero to rounding). FOSM and Monte Carlo no longer go
   through the search scorer, so they cannot meet it.
6. **Probabilistic runs.** FOSM: a ValueError naming the variable, the value
   and the reason. Monte Carlo: `n_no_solution`, `pf_lower`, `pf_upper`,
   `no_solution_note` (in `to_dict()`: `n_no_solution`, `pf_bounds`,
   `no_solution_note`); the statistics are over the realizations with a
   solution.
7. **Unknown names.** `solution.check_method` / `check_interslice`, called by
   `analyze_slope`, `compare_methods_table` (through it),
   `search_critical_surface`, `_compute_fos` (so every search function),
   `solve_method`, `yield_acceleration`, `fosm_fos`, `monte_carlo_fos` and the
   report package (before its search). The adapter already refused both.
8. **Searches count.** `SearchResult.no_solution`: `{family, method, n,
   not_converged, no_driving}` per family and method; `n_no_solution`; in
   `to_dict()`, `summary()` and the calc package's search table.
   `search_supplements.no_solution_warning` warns when the governing family
   lost trials to non-convergence.
9. **One solver.** `_compute_fos` and `analyze_slope` call the same function
   with the same arguments: the same moment axis (the GLE's fitted one), the
   same tolerance, the same test of convergence. ACADS 4's weak-layer search:
   section 4.

Also: `calc_steps.py` and `results.py` no longer speak of an approximate
solver; a method without a solution has a "no solution" row in the calc
package's comparison and no row in `compare_methods_table`; the slope calc
packages return `FOS` None with `no_solution` instead of failing or printing
another solver's value; `gle`'s "m_alpha clamped" warning is said once (it
was appended at every call).

## 4. The validation set (account section 11): before, after, printed

"Before" is master 2cc2c45 in a separate worktree, "after" the branch; both by
`module_work/gm140/step2_section11.py` (new; none existed) and by the pinned
tests. `slope_stability/tests` and the 21 `validation_examples` files that
import the module: master 784 passed, 17 skipped, 2 xfailed. On the branch the
same tests gave 773 passed and 11 failed before any test was edited; the 11
are listed under the table. No test pinned to a printed answer failed, and no
tolerance was widened.

| Ref | Problem | Before | After | Printed | What moved it |
|---|---|---|---|---|---|
| B1 | F&K (1977) homogeneous, dry and ru | pins hold | same | F&K 1.928, 2.080, 2.041, 2.073, 2.076 | nothing |
| B2 | F&K composite surface | pins hold | same | F&K 1.288, 1.373, 1.370 | nothing |
| B3 | ACADS 1(a), search: Bishop / Spencer / GLE / Janbu | 0.9851 / 0.9840 / 0.9840 / 0.9830 | same | 1.00 (Slide2 0.987, 0.986, 0.990) | nothing: no circle of these searches went to the old solver (0 trials without a solution; Janbu 1 toe circle) |
| B4 | Duncan (2000) reliability | pins hold | same | beta 2.32, about 1 % | nothing |
| B5 | Duncan, Wright & Brandon suite | all pass | same | book not in hand | nothing |
| B6 | ACADS 1(a), LE against fem2d SRM | pins hold | same | 1.00 | nothing |
| B7 | #95, #96, #98 rapid drawdown (V-037, V-038, V-041) | pins hold | same | 1.347; 1.443; 0.931, 1.039 | nothing (`rapid_drawdown.py` untouched; the lambda search change moved no pinned stage) |
| B8 | Pore-pressure grid (V-029) | pins hold | same | not pinned to print | nothing |
| B9/B16 | ACADS 1(b), crack (V-026) | pins hold | same | Slide2 1.596 / 1.592 / 1.592 | nothing |
| B10 | #54 micro-pile row (V-040) | pins hold | same | 1.193 | nothing |
| **B11** | **ACADS 4 (V-028), the declared family, loads on: Spencer / M-P / Janbu corrected** | 0.7532 / 0.7225 / 0.7284 | same (0 of 81 without a solution, each method) | 0.760 / 0.720 / 0.734 | nothing |
| **B11** | **ACADS 4, weak-layer search, Spencer (seed 3, 700 trials, loads on)** | **0.515** on x = 42.97 to 64.25; `analyze_slope` there: no Spencer solution (approximate 1.162), Janbu 1.222 | **0.830** on x = 41.59 to 77.19; `analyze_slope` there: Spencer 0.830 (theta 7.97 deg), Janbu corrected 0.830, M-P 0.816 | 0.760 (Table 9.4) | items 3 and 9: the 0.515 was a state about the second moment axis on an iteration that had not settled |
| **B11** | **ACADS 4, the calc-package report of that search (no loads: the package takes no load zones)** | prints **1.161** (the old solver on the search's 0.515 surface) | prints **0.879**, the search's own value on its surface | none | items 1 and 9 |
| B12 | V-054 steep phi = 0 slope, toe circle (Bishop) | pins hold | same | 2.16 | nothing; on its circle Spencer and GLE now have no number (the "anisotropic equals isotropic" check compared two approximate values) |
| B12 | #86 without supports, Spencer search | 0.9420 | same | 0.942 | nothing |
| B12 | #86, five tiebacks: Bishop / Spencer / GLE | 1.6131 / 1.6065 / 1.6070 | same | 1.629 / 1.620 / 1.622 | nothing: 6 planar trials have no Janbu state (held by the tiebacks), counted |
| B12/B16 | #85, one support, active: Bishop / Spencer / GLE | 1.3715 / 1.5505 / 1.5726 | same | Bishop 1.531, Spencer 1.884, GLE 1.575 | nothing governing: supported circles already had to converge. Counted now: Spencer 357 circles of the main search without a solution; Bishop 298 trials with no driving moment left |
| B12/B16 | #85, passive, GLE | 1.3594 | same | 1.378 | nothing |
| (brief) | **#85 without its support, Spencer search, 6 x 6** | **0.9095** (toe circle; 267 circles to the old solver) | **1.0468** (toe circle, Spencer's own; 329 trials without a solution) | none (Bishop 0.9095 before and after) | item 1: section 5 |
| B12 | V-032 (Slide2 #61) | pins hold | same | 1.366 | nothing |
| B13 | Duncan Ex. 1; Pilarcitos (drawdown) | pins hold | same | Taylor 0.706, 0.679; 0.823 | nothing |
| B14 | Weak layer over rock, 10 m cut | pins hold | same | none | nothing |
| B15 | V-030 Duncan (2000) surface and FOSM / Monte Carlo (5 tests) | pins hold | same | Slide2 1.157; Duncan 1.17 | nothing: every evaluation solves |
| B15 | V-043 sand fill, DE (strict xfail) | 1.234 | xfail holds | Slide2 1.188 | not measured again here: the test still fails against print, as marked |
| B16/B17 | V-043 sand fill, Spencer: pinned circle / critical circle / default search | 1.2096 / 1.1858 / 1.1859 | same | Slide2 1.209 (circle), 1.188 | nothing (0 trials without a solution) |
| B16 | V-031: Bishop / Spencer / GLE on the pinned circle; Spencer search | 1.4205 / 1.4247 / 1.4091; 1.4174 | same | 1.417 / 1.422 | nothing |
| B16 | Vertical cut, Bishop search | pins hold | same | Taylor 0.851 | nothing |
| B17 | V-042 (Hassan & Wolff): Bishop search; FOSM; Monte Carlo | 1.3335; cov 0.1228, beta 2.312; pf 0.0165 | same (0 realizations without a solution) | 1.334; beta 2.336 | nothing |
| B18 | Surface touching the base | pins hold | same | (consistency) | nothing |
| V-035/036 | Slide2 #79, #81 | pins hold | same | 1.44, 1.15 | nothing |
| V-039 | Newmark: ky of the #104-structure demonstration, Spencer / Bishop / M-P / Janbu | 0.56491 / 0.53519 / 0.55899 / 0.46921 | same | Ky 0.139 is Slide2 #104's, a different section | nothing: the same ky to five decimals by every method |
| V-044 | Slide2 #70 submerged: Bishop / Spencer / GLE | 1.5974 / 1.5962 / 1.5933 | same | 1.603 / 1.599 / 1.599 | nothing (the account's "degenerate free Spencer solution" is the same state before and after: theta 1.70 deg) |
| V-045 | Slide2 #34, FOSM | pins hold | same | none pinned | nothing |
| V-048, V-052 | Lowe-Karafiath; Pilarcitos | pins hold | same | 0.823, 1.047, 1.043 | nothing |

**Reading it.** The brief expected movement on #85, #86, ACADS 1(a), V-043,
V-031, V-044, V-039 and the probabilistic cases. Only #85 without its support
and ACADS 4's search move. On the others the old solver, an unsettled
iteration or the second moment axis never supplied the governing value, which
the script shows by counting: their searches report no trials without a
solution in the governing family. A wider check on made-up circles (three
slopes, 1,432 GLE solves, constant and half-sine): no converged value moved by
more than 0.002; 3 circles gained a solution (the lambda search) and 2 lost
one (an iteration that had not settled).

**ACADS 4's weak-layer search (item 9's case).** It returns Spencer 0.830,
9.2 % above the printed 0.760, so `test_v028_weak_layer_search_against_print`
stays a strict xfail and its reason is rewritten. The search and the single
surface now agree (0.830 both). What is left is the search's generator: 277 of
its 700 trials have no Spencer solution and its best surface is not the
declared family's best (0.753). That is step 7 (D12). Nothing was tuned.

**The 11 tests that failed, and what each asserted:**

| Test | It asserted | Now |
|---|---|---|
| `test_m_alpha::test_w11_surface_has_no_spencer_solution` | an m_alpha warning counted at the approximate solver's FOS | no number, so no F to state m_alpha at |
| `test_reinforcement::...test_ito_matsui_increases_fos_and_drops_with_spacing` | `FOS` with the pile row above the FOS without: it was 999.9 | no number at that spacing; the ordering asserted at two spacings that have one (3.25, 9.40 against 2.22) |
| `test_reinforcement::TestLegacyFallbackWarnsOnReinforcement` (2) | the fallback's Python warning | `NoSolution`, no warning, with supports and without |
| `test_search_reinforcement::test_reinforced_rigorous_nonconvergence_is_rejected` | `spencer_fos` falling back to 0.938 | `NoSolution` |
| `...test_no_solution_with_supports_reports_no_fos` (2) | the sentence "ignores the 1 support(s)" | the new sentence |
| `...test_no_supports_labels_the_legacy_value` | the labelled approximate value | no number |
| `...test_search_labels_a_legacy_critical_circle` | a governing circle scored by the old solver, labelled | the governing circle is Spencer's own; the lost trials are counted and warned of |
| `test_search_robustness::test_reject_stats_counts_degenerate_jagged` | a zig-zag "converging" about the second axis, then refused for m_alpha | no Spencer solution: counted as not converged |
| `test_published_v054_anisotropic_su::test_v054_anisotropic_reduces_to_isotropic_exactly` | equal Spencer and GLE values of two models: both were the approximate solver's | Bishop's values equal; Spencer and GLE have no number in either model |

Outside the module: `funhouse_agent/tests/test_slope_stability_adapter.py::
test_stabilizing_pile_ito_matsui_accepted` asserted `FOS > 0` on a 999.9;
it now uses a spacing with a factor of safety, and a second test pins the
no-number case.

## 5. Slide2 #85 without its support, Spencer, 6 x 6 (the count asked for)

| | Master | Branch |
|---|---|---|
| Surfaces evaluated | 1,960 | 2,143 |
| Circles handed to the old approximate solver | 267 | none: removed |
| Trials without a Spencer solution, as the result counts them | not counted | 329: main search 100, toe circles 87, circles above the toe 142 |
| Reported minimum | 0.9095, a toe circle (the old solver's value) | 1.0468, a toe circle (Spencer's own; Bishop on it 1.0468) |
| Bishop search, same grid | 0.9095 | 0.9095 |
| Run time | 18.7 s | 16 to 19 s (two runs) |

The same 267 circles, solved again with the branch's lambda search: 255 have
no Spencer solution, 2 gain one, 10 could not be rebuilt by the probe from
their rounded centres. So the lambda search of item 3 rescues few of them: on
this slope the circles have no solution to find. The 329 are more than the 267
because the families no longer stop early on approximate values (each
re-scores up to 12 short-listed circles until 3 solve).

**Where they lie** (the 329; toe (15, 10), crest (25, 30), H = 20, a rigid
base at the toe's level; the script's `where85` part):

- Lower end: 116 at the toe, 213 on the face above it (median 0.10 H above
  the toe, at most 0.49 H).
- Upper end: from 0.35 H in front of the crest (on the face) to 1.6 H behind
  it; median 0.83 H behind the crest.
- Centre: from 0.72 H below the crest's level to 3.24 H above it; median
  0.86 H above.
- Steepest slice base: 37 to 89 degrees, median 59.
- Why: 191 have no lambda from -1.5 to 1.5 at which the moment and force
  factors of safety agree; 97 have no equilibrium state at any lambda in
  that range; 41 end on a bracket whose bisection does not close (the two
  factors still differ at the closest lambda found).

They are not a fringe of odd circles: they are spread over the whole range of
circles a search tries on this slope, the governing toe circles included.

**Why Spencer has none there.** On a phi = 0 circle the moment factor of
safety does not depend on lambda (1.8349 on the circle centred (15.0, 39.6),
R = 20.36). The force factor of safety falls toward it as lambda grows (1.9365
at 0, 1.8728 at 0.5, 1.8569 at 1.0, 1.8534 at 1.5) and does not meet it; a
range of 5 or 60 finds no crossing either, so the lambda range (D29) is not
what stops it. The top of these circles is in tension (the solver reports
base tension). EM 1110-2-1902 C-10c (p. C-36) prints the remedy, a tension
crack; that is step 4.

**What this means for a user.** A Spencer (or Morgenstern-Price, or GLE)
search of a steep cohesive slope can now report a higher minimum than before
(1.047 against 0.910 here), because the circles that govern have no solution
by that method and are no longer scored by another. The result says so: the
trials are counted, and the warning names the family, the count, UFC 3-220-10
p. 362 and what to check (Bishop's simplified method on circles; a tension
crack). The earlier 0.910 was the right number for the slope but was Bishop's
value under Spencer's name.

## 6. Calls made (for the owner to revisit)

1. **The direct march as a fallback: only where the iteration ran and did
   not settle.** With supports the march also takes over where the
   fixed-point step cannot be taken (GM-60). Doing the same without supports
   was measured and not adopted: a 300-trial noncircular Spencer search of
   ACADS 1(a) took 111 s against 8 s (39 trials without a solution against
   67), ACADS 4's weak-layer search 297 s against 18 s (145 against 277), with
   the same minimum in both (1.114; 0.830). Circular searches cost the same
   either way. So a difference remains between runs with and without
   supports in how a fixed-lambda state is solved, not in the lambda search.
2. **An iteration that does not settle is no state.** No print for the 80
   iterations (D29). It removed 2 of 1,432 converged values on made-up
   circles, and the 0.515 above.
3. **One moment axis: the GLE's fitted one.** The search's second, fixed
   axis is dropped from scoring. Measured over four searches (about 2,100 trial
   solves) with the new lambda search it rescued no surface. The alternative was to
   give `analyze_slope` the second axis too.
4. **The solver functions raise.** `fellenius_fos`, `bishop_fos`,
   `janbu_fos`, `spencer_fos` and `morgenstern_price_fos` raise `NoSolution`
   (a ValueError, with `reason` and `kind`) instead of returning a marker or
   another solver's value; `solve_method` returns a `MethodSolution` and
   never raises for it. A caller that used a solver function's return on a
   surface without a solution now gets an exception, not a wrong number.
5. **A ratio at or above 999.9 is no factor of safety.** Where the driving
   sum is zero to rounding the ratio comes to 1e10 or more; `solve_method`
   gives no number and calls it "no driving". The threshold is the module's
   old marker, not a printed value.
6. **Two kinds, one warning.** `not_converged` and `no_driving` are counted
   apart. Only the first raises the warning: a surface with no driving
   moment left (a support's force exceeds it) is held, not a critical surface
   missed. Janbu's "no state" is counted as `not_converged` (the solver does
   not say which it was).
7. **FOSM gives no result; Monte Carlo counts and bounds.** As the brief
   set. Both say in the result or the error that the rule is the module's
   own. FOSM raises (it has no partial result to return).
8. **Probabilistic runs solve the given surface as `analyze_slope` does.**
   They went through the search scorer, so a fixed noncircular surface could
   be refused in some realizations by the search's shape screens (jagged
   below 1.5, span, slice count) and dropped without notice. Those screens
   are for trial surfaces; a given surface is now solved or has no solution.
   No pinned probabilistic value moved.
9. **Yield acceleration.** `ky` None with the reason, the kh and the bounds
   reached. The upper end of the kh bracket (2.0 by default, a numerical
   device) is halved until the method solves there; without that a rigorous
   method with no solution at kh = 2 would give no ky for an ordinary slope.
   No printed rule for either.
10. **A value below 0.05 in a search** is counted with the trials without a
    solution (`not_converged`). The 0.05 floor is the existing screen.
11. **`compare_methods_table`** has no row for a method without a solution
    and lists it in `methods_without_solution`, which now names any method
    (Fellenius, Bishop, Janbu, Spencer, Morgenstern-Price).

## 7. Where the rule now holds

- `solution.solve_method` (new): every method, both kinds of surface.
- `methods.fellenius_fos`, `bishop_fos`, `spencer_fos`,
  `morgenstern_price_fos`; `gle.janbu_fos`; `gle.gle_fos` (lambda search;
  `_GLESystem.solve_for_lambda`).
- `analysis.analyze_slope` (headline and every compare value),
  `compare_methods_table`, `search_critical_surface`.
- `search._compute_fos`, so every family and search function: `grid_search`,
  `optimize_radius`, the refinement walk, `search_entry_exit`,
  `search_noncircular`, `search_de`, `search_pso`,
  `search_weak_layer_biased`; `search_supplements.search_toe_circles` (toe
  circles and circles above the toe, their Bishop scan and their re-scoring),
  `check_infinite_slope` (planes), and the checks made on a governing surface
  (`_support_arm_warning`, `_profile_end_warning`, `m_alpha_confirmation`).
- `newmark.yield_acceleration` (every trial kh).
- `probabilistic.fosm_fos`, `monte_carlo_fos` (fixed surface and
  re-searched), so `reliability.wrappers.slope_reliability`.
- `geo_project.builders` (unchanged code): its searches and probabilistic
  runs call the functions above; a FOSM without a solution raises there.
- `funhouse_agent/adapters/slope_stability.py` (result descriptions; it
  already refused unknown names) and `adapters/calc_package.py`
  (`slope_stability_package`, `slope_report_package`: `FOS` None with
  `no_solution`; names checked before the search); `slope_stability/
  calc_steps.py` and `results.py` (tables, summaries).
- Rapid drawdown: `rapid_drawdown.py` is untouched (the brief). It calls
  `gle_fos` itself, so it gets the lambda search and nothing else; see
  section 10.

## 8. Tests

Judged by pytest's exit code.

| Run | Result |
|---|---|
| Master, `slope_stability/tests` and the 21 `validation_examples/test_published_*` files that import the module | 784 passed, 17 skipped, 2 xfailed (exit 0; 884 s) |
| Branch, the same files (with the new test file and the rewritten tests) | 806 passed, 17 skipped, 2 xfailed (exit 0; 882 s) |
| Branch, `slope_stability/tests/test_own_solution_or_no_number.py` alone | 22 passed |
| Branch, every file under `calc_package/tests`, `funhouse_agent/tests`, `funhouse_agent/deep/tests`, `reliability/tests`, `fem2d/tests`, `geotech_common/tests`, `report_ingest/tests`, `dxf_import/tests` and `geo_project/tests` that names `slope_stability` (43 files, `test_defaults_used.py` whole), with `webapp/tests/test_page_fingerprints.py` and `test_public_text.py` | 2,956 passed, 26 skipped, 1 failed (exit 1) |
| The 1 failure | `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`: fails the same way on master in this environment; not a slope test |
| Branch, `webapp/tests/test_public_text.py`, `test_backlog.py`, `test_docs_currency.py`, with the new files staged | 22 passed |

The two xfails are strict and unchanged in kind: ACADS 4's weak-layer search
(reason rewritten, section 4) and V-043's DE search. The 17 skips are the
suite's own, unchanged from master.

`python -m webapp.tests.page_builds record` was run and wrote an identical
`page_fingerprints.json`: the fingerprints do not cover these result
descriptions, so there is no fingerprint commit.

User manual: two parameter rows of `docs/user_manual/manual.html` hand-edited
(`yield_acceleration` `kh_max`; `slope_report_package` `f_interslice`, now
with its allowed values). No rebuild. No new input was added, so no new probe
in `funhouse_agent/tests/method_probes.py`; `test_defaults_used.py -k slope`
passes unchanged.

## 9. Release note

slope_stability: a factor of safety is now always the named method's own
converged solution on the surface, or there is none. The old approximate
Spencer and Morgenstern-Price solvers are removed. Where Spencer,
Morgenstern-Price or GLE do not converge, where Bishop's or Janbu's iteration
does not converge, or where a surface has no driving moment left (the old
"999.9"), `FOS` is null, `method_solved` is false and the warning says why; a
calc package prints no number. A search scores no surface by another solver:
it counts the trials without a solution by family and method (`no_solution`)
and warns when the governing family lost trials to non-convergence. So a
Spencer, Morgenstern-Price or GLE search of a steep cohesive (phi = 0) slope
can report a higher minimum than before, with that warning: on such slopes
those methods have no solution on the circles that govern (Slide2 #85 without
its support: 1.047 where the earlier 0.910 was Bishop's value under Spencer's
name); use Bishop's simplified method there, or model the tension crack. A
noncircular search and `analyze_slope` now give the same value for the same
surface (ACADS 4's weak-layer search: 0.830, was 0.515; printed 0.760). The
yield coefficient is null, with the reason, where the method has no solution
at a kh the bisection needs. FOSM gives no result, naming the variable, where
an evaluation has no solution; Monte Carlo counts such realizations, leaves
them out and bounds the probability of failure both ways. An unknown method
or interslice function is refused. No benchmark pinned to a printed answer
moved.

## 10. Stopped short of, and noted, not fixed

**Stopped short of.**

- `rapid_drawdown.py` is not edited (the brief: only calls the removed
  solvers break, and none does). So its search does not count the per-surface
  solves it refuses for a stage that did not converge, and
  `RapidDrawdownSearchResult.FOS` still holds 999.9 when no surface is found
  (`no_surface_found` says so). Step 9.
- `geo_project/` is not edited (the brief). Its JSON summary of a Monte Carlo
  run does not carry `n_no_solution` or the pf bounds; the result object it
  keeps (`_result`) does. A FOSM without a solution raises through
  `run_analyses`.
- `gle.GLEResult` still holds `fos = 999.9` beside `converged = False` in two
  early returns. It is the engine's state, read by `rapid_drawdown.py`; no
  caller of the method functions or of `solve_method` can receive it.

**Noted, not fixed.**

- D39, D24 (step 4): Spencer, Morgenstern-Price and GLE have no solution on
  the governing circles of a steep phi = 0 slope without a tension crack.
  This step makes that visible; the printed remedy is step 4's.
- D42, D13 (step 6): the planar family is scored by Janbu, the toe families
  scanned by Bishop, a polyline named for Bishop solved by Spencer, and M-P /
  GLE take the half-sine function in a search. Unchanged. One consequence
  seen: where the planar family governs a Spencer report, the report's
  Spencer re-analysis may now have no number; the package then returns `FOS`
  None with `no_solution` and the search's own value in `candidates`.
- D22 (step 3): `_solution_fails` still confirms a would-be critical polyline
  about three moment axes and two other slicings; the jagged and span screens
  stand.
- D29: the GLE's settings stand. The toe families' re-scoring tries up to 12
  short-listed circles when none solves (run time only).
- D12 (step 7): ACADS 4's weak-layer search, 0.830 against 0.760; V-043's DE
  search, a strict xfail as before.
- `funhouse_agent/tests/test_defaults_used.py::...[profile_figure|plot_data]`
  fails on master too in this environment (`logx` / `logy` change nothing in
  its result). Not a slope test.
- Janbu's "no state" is not split into no driving and not converged (call 6).
