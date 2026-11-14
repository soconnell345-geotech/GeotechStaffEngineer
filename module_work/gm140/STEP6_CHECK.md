# GM-140 step 6: one method across the search

Branch `claude/gm-140-step6-one-method`, from master 2d3c1a9 (steps 1, 2, 3,
4, 5a and 9, "Where a circle ends" and the GLE step are merged; their CHECK
files are on master). 2026-11-14.

**The rule.** Every family of surfaces in a search is scored by the method
named, with the interslice function named, so the reported minimum compares
like with like. A family the named method cannot score is reported under the
method that did score it, apart from the others.

## 0. Read this first

1. **One headline moves to another method's value, as the brief's items 4
   and 5 rule.** Slide2 #85 without its support, Spencer named, 6 x 6:
   Spencer's method solves none of the circles passing above the toe (142
   without a solution), so that family is scored by Janbu's simplified
   method, and its 1.033 is below Spencer's own lowest, 1.047 (toe circle).
   The result reports 1.033, labelled Janbu, with a warning first,
   `method_used`, Spencer's 1.047 in `named_method_lowest`, and
   `screen_unsolved`: 8 circles Bishop's simplified method scored down to
   0.9095 on which Spencer's method has no solution. Janbu's simplified
   method is known to read oddly on this face's steep entries (GM-130). If
   the lead wants the named method's value as the headline with the lower
   one beside it, that is the other form of the call in section 6, item 2,
   and a small change in `search_supplements.apply_supplements`.
2. **A noncircular search now runs a circular search first** (section 3,
   "the circle"). It costs the circular search's time on every noncircular
   search through `search_critical_surface` (ACADS 1(a), 200 random
   polylines by Spencer's method: 1.7 s to 4.6 s). `circle_check=False`
   turns it off and the result then says it was not checked.
3. **`candidates` keeps its form** (family: FOS). The method and function
   that scored each family are in a new list beside it,
   `candidate_methods`, because 20 tests, the calc package and the agent's
   adapter read `candidates` as numbers (section 6, item 4).
4. **Commit trailer.** The common brief asks for a `Co-Authored-By: Claude
   Opus 5.5` line; this session ran as another model, so its commits carry
   the trailer of the model that wrote them, as the slope brief asks ("the
   Co-Authored-By line of the model you run as").
5. **A test that fails on master too, not this step's.**
   `funhouse_agent/tests/test_defaults_used.py::
   test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
   (logx and logy change nothing in the result) fails on master 2d3c1a9 in
   this session's environment as well.

## 1. What the print says

- **EM 1110-2-1902 4-3b(9)(a), p. 4-8**: state the method used (the
  account's section 8).
- **EM C-2b, p. C-16**: the Ordinary Method of Slices "can only be used
  with circular slip surfaces".
- **EM C-5a, p. C-26**: Spencer's method extended to noncircular surfaces.
- **FHWA NHI-06-088 Table 6-1, p. 6-21**: Bishop's the minimum method,
  Spencer's where available.
- **UFC 3-220-10 Table 7-7 step 5, p. 374**: select Spencer's or the
  Morgenstern-Price method for the failure surfaces of an MSE slope.
- **UFC 3-220-10 p. 362**: "Critical failure surfaces can be missed as a
  result of non-convergence. The engineer should be able to identify when
  convergence issues are present."
- **Bishop (1955), p. 10**: Eq. 13 takes moments about the centre of a
  circle.
- **Morgenstern and Price (1965), p. 85**: the interslice function f(x).
- **EM 4-2e(2), p. 4-7** (the account's section 3): for a noncircular
  search the initial surface should correspond to the estimated critical
  surface. **Duncan (1996, p. 581)**: unless geology constrains it, the
  critical surface is essentially circular.
- **Not found in print.** A rule on comparing surfaces solved by different
  methods (the account's D42): none. The order in which another method
  stands in: none. Janbu's correction factor f0 (D28): not in hand.
- **Searched for and not found in the reference library** (as the brief
  asks for the circle check): the library's UFC 3-220-10 (the 2026 UFS
  text, chapter 7) and FHWA NHI-06-088 (chapters 6 and 10) say only that
  the common programs "have very robust search routines for circular and
  noncircular failure surfaces" (UFC, Sec. 7-1 area) and that "Computer
  solutions are also available for failure modes defined by planar and
  non-circular surfaces" (FHWA). Neither prints a rule for starting a
  noncircular search from the critical circle. EM 1110-2-1902 is not in
  the library; the sentence above is the account's citation of 4-2e(2).
  It says "estimated critical surface", not "critical circle", so taking
  the critical circle is a call (section 6, item 5).

## 2. The departures, reproduced

`slope_stability/tests/test_one_method.py` (30 tests, made-up geometries).
All 30 fail on master 2d3c1a9 and pass on the branch. Many fail on master
for want of the new result fields; these fail on master on the behaviour
itself:

| Departure | Test | Master |
|---|---|---|
| D42: planes always by Janbu | `test_spencer_named_scores_the_planes_by_spencer` | the planes of a Spencer search are solved by Janbu's simplified method |
| D13: half-sine always | `test_gle_with_the_constant_function_in_every_family`, `test_gle_constant_search_is_the_spencer_search` | `search_critical_surface` has no `f_interslice`; a GLE search takes half-sine |
| D13: PSO and weak layer always Spencer | `test_pso_and_weak_layer_use_morgenstern_price_when_named[pso, weak_layer]` | every polyline solved by Spencer's method, labelled 'Spencer' |
| D13: every noncircular kind | `test_every_noncircular_search_takes_the_function_named[4 kinds]` | half-sine whatever is named |
| D26: OMS on polylines | `test_the_ordinary_method_of_slices_named_on_planes_and_polylines` | the Ordinary Method of Slices scores the polylines |
| D49: labels | `test_a_governing_toe_circle_is_labelled_with_its_method[4]`, `test_a_grid_circle_by_morgenstern_price_is_labelled_so`, `test_a_noncircular_search_named_for_bishop_is_labelled_spencer` | 'Morgenstern_price', 'Gle'; a Spencer result labelled 'Bishop' |
| D42: headline | `test_bishop_named_on_a_cohesionless_face` | planes by Janbu, labelled Janbu, no statement of Bishop's own lowest |
| circle check | `test_a_noncircular_minimum_is_checked_against_the_circle` | 30 random polylines report their own minimum alone |

## 3. What changed, by the brief's items

**Item 1, the function and the method in every search.**
`search_critical_surface(f_interslice='half_sine')`; an unknown name is
refused (`solution.check_interslice`). It is put in force for every solve
made inside the search (`search._applies_interslice`,
`search.interslice_in_force`), and `search._method_solution`, the one place
a search solves a surface, reads it. `grid_search`, `search_entry_exit`,
`search_noncircular`, `search_de`, `search_pso` and
`search_weak_layer_biased` take `f_interslice` (None: the function in
force). `search_pso` and `search_weak_layer_biased` take `method`.
`search_rapid_drawdown` already handed its `f_interslice` to every family's
staged solve; it now refuses an unknown name, and its surface is labelled
Morgenstern-Price when the function is not the constant one.

**Item 2, the planes.** `check_infinite_slope` scores each face's planes by
the method named (`PLANAR_METHOD`, Janbu's simplified method, is now only
the default of a direct call with no method).

**Item 3, screens.** The Bishop screen of the main search
(`bishop_screen`), the Bishop scan of the toe-circle families under
Spencer / Morgenstern-Price / GLE (and of the circles above the toe under
Janbu) and the refinement walk's radius by Bishop stay as run-time devices.
Checked in the code: every value a family reports is the scoring method's
own on the surface reported (the toe families re-optimise and score the
shortlist by the method; the walk scores Bishop's circle by the method and
re-optimises the radius by the method at its end; the screen's own values
in `grid_fos` carry `method: bishop` and cannot be reported). `choices`
names them ("screens by another method (run-time devices)"), and
`candidate_methods[...]["screen"]` names the screen of a family.

**Item 4, a family the method cannot score.** `search.scoring_order`: the
method named where it applies, then Spencer's method, then Janbu's
simplified method (`search.STAND_INS`). Bishop's simplified method and the
Ordinary Method of Slices do not apply to a plane or a polyline
(`search.CIRCLES_ONLY`). A family "solves none" when it scored no surface
and lost trials to non-convergence (the `no_solution` count,
`not_converged`); a family with no surface for another reason (windows,
geometry, a support holding it) is not handed on. It is applied to the
main search (`analysis._main_search`), the toe-circle families
(`search_supplements._toe_families`) and the planes, face by face
(`check_infinite_slope`). The family is reported apart: its own entry in
`candidate_methods` with `named: False` and the `reason`.

**Item 5, the headline.** The lowest over all families is reported. Where
its family was scored by another method: `SearchResult.method_used`, a
warning placed first, and `named_method_lowest` (the named method's own
lowest value and its family, or FOS None with a note when it scored no
family, as in a noncircular search named for Bishop).

**Item 6, truthful entries.** `candidate_methods`: `{family, method,
f_interslice, FOS, named, reason}` (and `screen`, `flags`). The governing
surface's label is the method that scored it (`search.method_label`).

**Surfaces the method could not solve.** `search._note_screen_unsolved`
records the screen value of a surface the screen scored and the method had
no solution on (the Bishop screen's shortlist, the refinement walk, the
entry and exit screen, the toe-circle shortlists).
`SearchResult.screen_unsolved` counts those below the reported minimum,
with the lowest screen value, by family; a warning says so.

**From the GLE step.** Each `infinite_slope_checks` entry and the planar
family's `candidate_methods` entry carry `flags`: what the printed
acceptability checks find on the lowest plane (Spencer's `low_m_alpha` on
a steep end among them), with `n_planes` and `n_solved_by_named`.

**The circle.** Through `search_critical_surface` a noncircular search
first runs the circular search (centre grid and, with `toe_circles`, the
toe-circle families; the caller's windows; default grid) by the method
that scores the polylines, in a context of its own so that its trials are
not counted as this search's. Its critical circle, as the polyline of its
slice-base points (`search.circle_as_polyline`), is scored as one more
trial (`search.add_circle_trial`) and reported where it is the lowest.
`circle_check` gives the circle, its value as a circle and as a polyline,
the search's own lowest polyline, and which governs; a warning says so
when the search's own polylines stayed above the circle. With
`circle_check=False`, in a search function called on its own, and in a
rapid-drawdown search (two or three staged solves a circle), no circular
search runs and the result says the minimum was not checked against a
circle. Under a time budget the circular search may use a quarter of it.

**D26 on one given surface.** `analyze_slope` still solves the Ordinary
Method of Slices on a polyline (Fredlund and Krahn 1977 report such a value
on their composite surface; VALIDATION.md B2) and now warns with EM C-2b's
words. A search does not score a polyline by it.

**The calc package.** `slope_report_package` passes `f_interslice` to the
search (before: only to the re-analysis) and re-analyses the governing
surface by the method that scored it; the response carries
`candidate_methods`, and `method_used`, `method_named` and
`named_method_lowest` where they differ. The search section of the report
prints the method beside each family's minimum, the circle check and the
screen values not solved.

## 4. The validation set (account section 11): before, after, printed

Measured on master 2d3c1a9 and on the branch by
`module_work/gm140/step2_section11.py`, `step3_section11.py` and
`circle_ends_section11.py` (unchanged) and by
`module_work/gm140/step6_section11.py` (added for this step's rows).

### Values that move

| Benchmark | Module before | Module after | Printed | Which part of the rule moved it |
|---|---|---|---|---|
| B12, #86 without supports, c' = 0: planar family (Bishop named) | 0.9430 (Janbu) | 0.9442 (Spencer, apart) | tan 37 / tan 38.66 = 0.9419 | item 2 and 4: the planes by Spencer's method, Bishop's being for circles. +0.24 % on the closed form, +0.11 % before |
| B12, the same, Bishop's headline | 0.942 (circular) | 0.942 | 0.942 | not moved |
| B12, #86 without supports, circles kept off the face, Bishop | 0.9430, labelled Janbu | 0.9442, Spencer, `method_used`; Bishop's lowest 0.9515 beside it | 0.9419 | items 4 and 5 |
| B12, planar family against closed forms (#86 dry, ru 0.2, kh 0.1; Duncan Ex. 3 dry, kh 0.15; Duncan Ex. 4) | +0.11 to +0.15 % (Janbu, whatever was named) | by the method named: Janbu +0.11 to +0.15 %, Spencer +0.24 to +0.57 %, Morgenstern-Price and GLE half-sine +0.08 to +0.11 % | closed form | item 2 |
| V-043 sand, default Spencer search: planar family | 1.2575 | 1.2591 | (the search's headline 1.1859 is not moved; Slide2 circle 1.209) | item 2 |
| V-043 sand, DE, seeds 1 to 3, Spencer | 1.237 / 1.234 / 1.233 | 1.194 each; the evolution's own lowest polylines unchanged, in `circle_check` | Slide2 1.188, Tandjiria 1.192 (+0.5 %, +0.2 %) | the circle check: the critical circle (1.186) as the polyline of its 30 slice bases |
| V-043 sand, random noncircular, 2000 trials, seeds 1 and 2 | 1.34 and 1.46 (the test file's record) | 1.194 each | 1.188 | the same |
| B3, ACADS 1(a), random noncircular, 200 trials, Spencer | 1.0916 | 0.9844 | 1.00 (the circle; Spencer 0.9838 on it, Slide2 0.986) | the circle check; 0.9844 is the circle's 0.9838 plus 0.06 % of discretisation |
| the same, Morgenstern-Price / Janbu named | 1.0794 / 1.0357 | 0.9844 / 0.9832 | 1.00 | the same |
| ACADS 1(a), DE seed 1 / particle swarm seed 1, Spencer | 0.9907 / 0.9931 | 0.9844 / 0.9844 | 1.00 | the same |
| ACADS 1(a), particle swarm, Morgenstern-Price named | 0.9931 by Spencer | 0.9844; own polylines 0.9917 by Morgenstern-Price | 1.00 | item 1 (the swarm by the method named) and the circle check |
| #85 without its support, Spencer, 6 x 6 (step 2's measured case; no printed value) | 1.0468 (toe circle) | 1.0330 by Janbu (circle above the toe); Spencer's own 1.0468 stated | none (Bishop 0.9095 on this section) | items 4 and 5 (section 0, item 1) |
| #86 with five tiebacks: `no_solution` | the face's 6 held planes counted under Janbu | counted under Spencer or GLE | (FOS 1.6131 / 1.6065 / 1.6070 not moved; printed 1.629 / 1.620 / 1.622) | item 2 |

### Values that do not move (to the digits the scripts print)

#85 with one support, active and passive, Bishop, Spencer and GLE, and on
Slide2's circle; #85 without support, Bishop 6 x 6; #86 with five tiebacks,
Bishop 1.6131, Spencer 1.6065, GLE 1.6070; #86 without supports, Spencer
0.942; ACADS 1(a) circular searches (Spencer 0.9838, toe circle); V-043
Spencer circular 1.1859; V-031; V-044; V-039 (Newmark, Ky); the FOSM and
Monte Carlo cases (B4, V-030, V-042, V-045); ACADS 4 on its declared family
and its weak-layer search (0.8301; printed 0.760; the strict xfail at 3 %
stays, step 7: its critical circle, 0.9765, is above the search's own
surface, so the circle does not govern); V-054 Bishop 2.346 (printed 2.16);
V-054 Spencer and GLE-constant 2.4448; V-032; V-042 entry and exit; Duncan
Ex. 1, D = 2 and 3; the vertical cut and the other rows of "Where a circle
ends"; Duncan Ex. 4 Janbu; B14 (Bishop and Spencer 0.8822); GM-130's clay
face; the single-surface rows. V-035 and V-036 (Slide2 #79 and #81) are
closed-form tests (`infinite_slope_fos`) and run no search. V-041 and the
other drawdown searches pass their tests unchanged (one staged solve
scored every family before and after).

### New reports on unmoved values

- V-054, Spencer or GLE named (2.4448; printed 2.16; Bishop 2.346): the
  result now carries `screen_unsolved`: 12 circles the Bishop screens
  scored below 2.4448, down to 2.3464, on which the method has no solution
  (7 in the refinement walk, 3 toe circles, 2 above the toe). That is the
  gap between this slope's Spencer value and Bishop's, now said.
- ACADS 1(a) noncircular: the headline's flags are now those of the
  circle's polyline (`side_tension`); the lowest unflagged surface (1.096)
  is as before.

### GM-56: how many planes the named method solves

Every method that applies to a plane solves all six planes of every planar
family measured (six families; the table is VALIDATION.md B25): Janbu's
simplified method, Spencer's method, Morgenstern-Price and GLE, 6 of 6
each. So no rigorous method loses the infinite-slope mechanism on these
faces since the GLE step. That item 4's order still reports the family
when a method does lose it is shown by
`test_a_method_that_solves_none_of_the_planes_gives_way_in_order`
(Morgenstern-Price made not to converge on planes: the family is scored
by Spencer's method; Spencer's too: by Janbu's simplified method, within
0.5 % of the closed form, with the reason and `n_solved_by_named` 0 of 6).

### Run time (this machine, other jobs running; seconds)

| Search | Before | After |
|---|---|---|
| ACADS 1(a), circular, Spencer, 25 slices | 7.0 | 4.9 |
| #86 without supports, Spencer, default | 8.0 | 5.6 |
| V-043 sand, circular, Spencer | 5.5 | 5.6 |
| ACADS 1(a), random noncircular, 200 trials, Spencer | 1.7 | 4.6 |
| ACADS 1(a), DE, Spencer | 5.4 | 6.9 |
| V-043 sand, DE, Spencer | 4.5 | 8.4 |
| V-028 weak-layer search, 700 trials | 9.9 | 13.8 |

Circular searches do the same work as before (the differences are the
machine's load). A noncircular search adds the circular search of the
check.

## 5. Where the rule now holds

- `search_critical_surface`: every `surface_type` (circular, entry_exit,
  noncircular, noncircular_de, pso, weak_layer) and every supplementary
  family (toe circles, circles above the toe, planes).
- `grid_search`, `search_entry_exit`, `search_noncircular`, `search_de`,
  `search_pso`, `search_weak_layer_biased`, `search_toe_circles`,
  `check_infinite_slope`: the function in force; the method named.
- `search_rapid_drawdown`: one staged solve for every family, named for
  each in `search.candidate_methods`; an unknown `f_interslice` refused;
  the label follows the function.
- `monte_carlo_fos(research_surface=True)`: its searches go through
  `search_critical_surface` and take the rule (its `search_kwargs` may
  carry `f_interslice`).
- `geo_project`'s limit-equilibrium builder: calls
  `search_critical_surface` with the analysis' method, so its search takes
  the rule (not edited; see section 9).
- The agent's adapter (`funhouse_agent/adapters/slope_stability.py`):
  `f_interslice` and `circle_check` on `search_critical_surface`, their
  descriptions, `applies_when`, the returns keys; two probes in
  `method_probes.json` (`#gle_constant`, `#noncircular`).
- The calc package (`slope_report_package`): section 3.
- Not searches, so not touched by this rule: `analyze_slope`,
  `compare_methods_table` (they solve one surface by the method and
  function given; `analyze_slope` gains the D26 warning),
  `yield_acceleration`, Newmark, FOSM and Monte Carlo on a fixed surface,
  `reliability/wrappers.py`.

## 6. Calls made (for the owner to revisit)

1. **The order of the stand-ins: Spencer's method, then Janbu's simplified
   method** (the lead's, in the brief). Basis: a complete equilibrium
   method first (EM C-5a, p. C-26; FHWA Table 6-1, p. 6-21). Other form:
   Janbu's simplified method first for the planes, as before step 6 (it
   sits closer to the closed form: +0.11 % against Spencer's +0.24 %).
2. **The lower value governs and both are stated** (the lead's). Other
   form: the named method's own lowest value is the headline and the lower
   value by another method is stated beside it. No printed rule was found.
3. **"Solves none" is decided face by face for the planes**, and for a
   family only when it lost trials to non-convergence (builder's call).
   Other forms: the whole planar family handed on only when no face is
   solved; a family handed on whenever it scored nothing.
4. **`candidates` keeps its form and `candidate_methods` carries the
   methods** (builder's call; the brief says each entry of `candidates`
   carries the method). Other form: `candidates[family]` becomes a record,
   which changes what every consumer reads.
5. **The circle check takes the critical circle of the module's own
   circular search, run by default before a noncircular search**
   (builder's call on the orchestrator's item). EM 4-2e(2) speaks of the
   "estimated critical surface"; the library prints no rule (section 1).
   Other forms: only where the caller hands in a circular result; or
   report the circle's value beside the noncircular minimum without
   adopting the circle's polyline as the surface reported.
6. **The circular search of the check uses the default 10 x 10 grid and
   the caller's windows**, not `nx` / `ny` (which mean other things in a
   particle-swarm search). Builder's call.
7. **The Ordinary Method of Slices on one given polyline stays available,
   with a warning** (builder's call on D26). Other form: refused, or
   solved by Spencer's method as Bishop's is.
8. **The drawdown search's noncircular kinds are not checked against a
   circle** and say so (builder's call: a circular drawdown search costs
   two or three staged solves a circle).

## 7. Tests

By pytest's exit code, in the venv of the common brief (Python 3.12).

| Run | Result |
|---|---|
| `slope_stability/tests/test_one_method.py` on the branch | 30 passed |
| the same file on master 2d3c1a9 | 30 failed (failing first) |
| One run of: `slope_stability/tests` (all of it); the 21 `validation_examples/test_published_*` files that import `slope_stability`; the 42 files of `calc_package/tests`, `funhouse_agent/tests`, `reliability/tests`, `fem2d/tests`, `geotech_common/tests`, `report_ingest/tests`, `dxf_import/tests` and `geo_project/tests` that name `slope_stability`; `funhouse_agent/tests/test_defaults_used.py` (all of it, so `-k slope` too); `webapp/tests/test_public_text.py` | 4,014 passed, 49 skipped, 4 xfailed, 1 failed; exit 1 for the one failure, `test_defaults_used.py::...[profile_figure|plot_data]`, which fails on master 2d3c1a9 too (section 0, item 5) |
| `webapp/tests/test_page_fingerprints.py` | 13 passed, before and after `python -m webapp.tests.page_builds record`; the record did not change, so there is no fingerprint commit |
| The user manual, with `geotech-references` (e1bbc5b, v1.7.0, the commit master pins) first on PYTHONPATH: `check_manual.py --rebuild --write`, then `check_manual.py`, then `pytest webapp/tests/test_user_manual_current.py` | exit 0, exit 0, 1 passed (before the rebuild: 1 miss, `search_critical_surface.circle_check`) |

Existing tests changed, each with its reason in the test:

- `test_gle_root.py::test_a_search_makes_no_scan_below_0`: 1.0916 ->
  0.9844 (printed 1.00), the pin the brief names.
- `validation_examples/test_published_v043_slope_reinforced.py`: the DE
  minima 1.234 -> 1.194 (printed 1.188); the strict xfail of GM-344 on the
  three seeds and the strict xfail of GM-117
  (`test_v043_sand_noncircular_reaches_the_published`) both pass and
  their marks are removed. Nothing was tuned: the polylines the evolution
  generates itself are unchanged and the test asserts they still stand
  above 1.23.
- `test_search_supplements.py`: the planar family on #86 by Spencer's
  method (0.9430 -> 0.9442; closed form 0.9419), and
  `test_planar_family_alone_on_86_is_janbu` renamed and rewritten (the
  family reported under Spencer's method with Bishop's own lowest beside
  it; Janbu named still gives 0.9430).
- `test_choices.py` (two tests): the method choice names the method asked
  for; the family scored by another method is a choice of its own.
- `test_m_alpha.py`: the Ordinary Method of Slices no longer scores a
  polyline in a search (D26); the DE search's count is 2,326 (the circle's
  polyline is one more trial).
- `test_search_robustness.py`: the jagged screen's case is Janbu's alone
  (the Ordinary Method of Slices' polylines are scored by Spencer's
  method).
- `test_own_solution_or_no_number.py`: a search takes `f_interslice` and
  refuses an unknown one.
- `funhouse_agent/tests/test_slope_stability_adapter.py`: the flagged
  headline of ACADS 1(a)'s noncircular search is the circle's polyline.

## 8. Release note

A critical-surface search now scores every family of trial surfaces by the
method asked for, with the interslice function asked for, and says which
method scored each family. Results that change: (1) the shallow planes on
a face of cohesionless soil are scored by the method named instead of
always by Janbu's simplified method, so that family's value moves by up to
about 0.5 % (0.9430 to 0.9442 on a 1.25H:1V sand face with Spencer's or
Bishop's method named); (2) with Bishop's simplified method or the
Ordinary Method of Slices named, planes and noncircular surfaces are
scored by Spencer's method and reported under it, and a family the method
solves none of is scored by Spencer's method, then Janbu's simplified
method: where such a family gives the lowest value the result now reports
it under that method, with the named method's own lowest value beside it;
(3) Morgenstern-Price and GLE searches take the interslice function given
(`f_interslice`, half-sine by default as before), and particle-swarm and
weak-layer searches use the method named instead of always Spencer's;
(4) a noncircular search first runs the circular search by the same method
and includes its critical circle among its trials, so it no longer reports
a minimum above the critical circle's value (a 200-trial random search on
a simple slope: 1.092 to 0.984, printed 1.00), at the cost of the circular
search's run time; (5) labels: 'Morgenstern-Price' and 'GLE' in place of
'Morgenstern_price' and 'Gle'. A result also reports the surfaces a Bishop
screen scored below the reported minimum that the method could not solve.

## 9. Stopped short of, and noted, not fixed

- **D28, Janbu's correction factor f0**: not found in print; left as it
  is (out of scope by the brief). It now scores a family only when named
  or as the last stand-in.
- **`geo_project/builders.py`** (not to be edited): its entry reports
  `"method": a.method`, the method named. The search's own result, which
  it also carries, has `method_used` and the warning. A label, not a
  value.
- **`yield_acceleration`, FOSM, Monte Carlo on a fixed surface** take no
  `f_interslice`; their Morgenstern-Price and GLE solves use the half-sine
  function, as their documents say. Not a search; no saved value changes.
- **D12, the noncircular generators**: V-043's evolution still ends at
  1.233 to 1.237 on its own polylines and ACADS 4's weak-layer search at
  0.830 (printed 0.760). The circle check reports the first beside the
  circle; the generators are step 7.
- **The circle check with supports** was not measured on a printed case
  (no noncircular benchmark of section 11 has supports).
- **Timing figures in the adapter's `method` description** for the
  noncircular searches were measured before this step; the text now says
  the circle check adds the circular search's time. They were not
  re-measured on the development laptop.
- **The user manual** is in a commit of its own, the last on the branch.
  **The page fingerprints** did not change (re-recorded: no difference),
  so there is no fingerprint commit.
