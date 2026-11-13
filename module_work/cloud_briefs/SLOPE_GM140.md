# Cloud helper brief: the slope_stability rebuild (BACKLOG GM-140)

Read `COMMON_MODULES.md` (beside this file) first: its rules bind every task here. This file adds what is common to the rebuild's steps, then has one section per cloud session. Your routine's prompt names your section; do that section only.

The rebuild has ten steps (the account's section 12). Step 1 (one material per point) is merged (6cb96ec, 2026-11-13); its record, with the fix round after review, is `module_work/gm140/STEP1_CHECK.md`. Sections here: steps 2, 3, 4, 5a, 6, 8, 9 and 10. Steps 5b and 7 are a short note at the end.

## Common to every step

### Read, in this order

1. `module_work/slope_integral_review/ACCOUNT.md`, all of it. It is the specification. Sections 2 to 9 give each departure from print (D1 to D50) with its page cites; section 11 is the published validation set; section 12 the ten steps; Appendix A the code locations at commit 60a2975 (line numbers have moved; find the code by what it does).
2. `module_work/owner_calls_2026-11-12/CALLS.md`, section A: OWN-101 to OWN-104. They are decided; build them as written.
3. `slope_stability/DESIGN.md` and `slope_stability/VALIDATION.md`.
4. `module_work/gm140/`: the CHECK file of every step already merged (`STEP1_CHECK.md` and so on). Your section names the steps it requires. If a required step's CHECK file is not on master, stop and say so: your branch would start from the wrong code.

### The method (why this rebuild exists)

The owner stopped the earlier slope fixes because each was fitted to one case. So:

- **One rule, everywhere.** Your step changes one rule. Apply it to every method (Ordinary Method of Slices, Bishop's simplified, Janbu's simplified, Spencer, Morgenstern-Price and GLE), every family of surfaces (grid circles, entry and exit circles, circles through the toe and above it, planes, random, differential-evolution, particle-swarm and weak-layer polylines) and every caller: `analyze_slope`, `compare_methods_table`, `search_critical_surface`, rapid drawdown and its search, the yield acceleration and Newmark, FOSM and Monte Carlo, `reliability/wrappers.py`, `geo_project`'s limit-equilibrium builder, the agent's adapter (`funhouse_agent/adapters/slope_stability.py`) and the calc packages. Grep for every caller; list in the CHECK file each place the rule now holds.
- **Reproduce the departure first** as failing tests with made-up values. Each test's docstring states the printed rule and its page.
- **Re-run the whole validation set of the account's section 11, before and after.** The CHECK file carries one table: benchmark, module before, module after, printed answer, and for each value that moves, which part of your rule moved it. If `module_work/gm140/` already holds a script that runs section 11, use it unchanged and add rows only for benchmarks you add. If you must write one, name it `module_work/gm140/step<N>_section11.py` (other sessions run in parallel; never write a path another session might also create).
- **Re-pin against the printed answers, never against the module's previous numbers.** A test that pins the module's own value may take the new value only with the printed answer and the percentage difference in the same assertion's comment (`# GM-140 step N: 1.372 -> 1.49; printed 1.531 (source, page)`). If your rule moves a value away from print, do not re-pin and do not widen a tolerance: explain the likely cause in the CHECK file and mark the test a strict xfail with that reason.
- **Nothing fitted.** No constant, tolerance, grid, seed or slice count chosen because it makes one case pass. If a benchmark's inputs look wrong, compare them with the printed problem (`module_work/slope_v53/slide2_selected_problems.txt` holds the Slide2 verification text for some problems) and say what differs; do not tune.
- **No edge-case chasing.** Other departures you meet are "noted, not fixed" lines in the CHECK file, with their D number.

### Print

- Every formula, default and check cites a printed page. Where the references differ, the result's `choices` lists the cited options and `defaults_used` the default.
- FHWA NHI-06-088, UFC 3-220-10, GEC 11 and GEC 7 are in the reference library (COMMON_MODULES says how to reach its text): read the page's own text before you code from it.
- EM 1110-2-1902 (2003), Whitman and Bailey (1967), Duncan (1996), Duncan, Buchignani and De Wet (1987), Duncan, Wright and Brandon (2014) and the Slide tutorials are the owner's own copies and are in no repository. Work from the account's quotations and from the equations this file gives, each with its page. Cite them in code by document, page and equation number. Do not write a formula from memory as if it were the print: if you need a page that neither the account nor this file quotes, stop short of that part and say so in the CHECK file.
- A new option gets all of: the library parameter, the adapter's parameter and its description, an entry in `choices` with the printed basis and the alternatives, `defaults_used` when left at its default, the matching row of `docs/user_manual/manual.html`, and a probe in `funhouse_agent/tests/method_probes.py`.
- **Calls.** Where judgement is needed between printed options and this file has not decided it, choose, state both forms in the result and the tool description, and list the call in the CHECK file under "Calls made" for the owner to revisit.

### Working beside other sessions

- Other sessions change `slope_stability/` at the same time; the lead merges them one by one. Keep your diff to the functions your rule needs: no reformatting, no renames, no moved code. Put new logic in new functions.
- Do not edit `BACKLOG.md`, `HANDOFF.md`, `CLAUDE.md`, `geotech-references/`, `packages/aec-document-review/`, or `dxf_import/` and `geo_project/` unless your section says so. `slope_stability/DESIGN.md` and `VALIDATION.md` are the module's documents: update the parts your rule changes.
- Re-record the page fingerprints in a commit of its own, the last one on the branch, so the lead can redo it at merge.
- Branch `claude/gm-140-step<N>-<slug>` (named in your section). CHECK file `module_work/gm140/STEP<N>_CHECK.md`. End each commit message with the Co-Authored-By line of the model you run as.

### Tests to run (pytest's exit code; `-n auto` is fine)

`slope_stability/tests` (all of it); every `validation_examples/test_published_*` file that imports `slope_stability`; the `calc_package/tests`, `funhouse_agent/tests`, `reliability/tests`, `fem2d/tests`, `geotech_common/tests`, `report_ingest/tests`, `dxf_import/tests` and `geo_project/tests` files that name `slope_stability`; `funhouse_agent/tests/test_defaults_used.py -k slope`; `webapp/tests/test_page_fingerprints.py`; `webapp/tests/test_public_text.py`.

### The CHECK file also carries

The section 11 table; the list of places the rule now holds; "Calls made"; a one-paragraph release note (answers move in every step); and what you stopped short of.

---

## Step 2: a method's own solution, or no number

Branch `claude/gm-140-step2-own-solution-or-no-number`. Requires on master: step 1. First of the series 2, 3, 6 (all three edit the search and the surface screening).

**The rule.** A factor of safety reported under a method's name is that method's own converged solution on that surface. Where the method has no solution there is no number: `FOS` is None, `method_solved` is False and the result says why. This holds for one surface, every search family, the yield acceleration, FOSM, Monte Carlo and the reliability wrapper.

**Departures and print.** D23, D27, D30, and D49's parts on input and sentinel values; GM-81's rest, GM-139; the audit's row 6 (`module_work/module_audit_2026-11-12/AUDIT.md` section 4, slope_stability). UFC 3-220-10 p. 362: "Critical failure surfaces can be missed as a result of non-convergence. The engineer should be able to identify when convergence issues are present." EM 1110-2-1902 C-10d, pp. C-37 to C-38 (ways to avoid non-convergence: reject the unrealistic one of multiple solutions, a sensible starting F, a realistic initial surface). Spencer (1967) Eqs. 7 and 8, p. 15; Morgenstern and Price (1965) Eq. 18, p. 85; Bishop (1955) Eq. 13, p. 10.

**What changes.**

1. The old approximate Spencer and Morgenstern-Price solvers (`spencer_fos_legacy`, `morgenstern_price_fos_legacy`) supply no value anywhere: not to one surface, a comparison table, a circular or entry and exit search, a supplementary family, the yield acceleration, a probabilistic run or a calc package. Remove them from the package's exports; delete them unless a consumer outside the module needs the code (grep the whole repo and say what you found).
2. Circles and polylines are treated alike: a Spencer, Morgenstern-Price or GLE trial that does not converge is not scored, with or without supports.
3. The rigorous solver's more robust search for lambda runs today only where supports are present (GM-59, GM-60, GM-81). Make it the one search for every run, so fewer surfaces are left without a solution. Measure on Slide2 #85 without its support, 6 x 6 Spencer search: the account counts 267 circles handed to the old solver today. Report how many circles have no rigorous solution before and after, and where they lie.
4. Bishop's iteration that does not converge (50 iterations) gives no number, in a result and in a search (D27). Janbu's simplified method likewise.
5. The solver's 999.9 marker (zero or negative driving) never leaves a solver as a factor of safety: a result carries None with the reason, a search skips and counts the surface, and FOSM, Monte Carlo and `reliability/wrappers.py` never take it as a sample.
6. A probabilistic run that meets a realisation with no solution: FOSM gives no result and names the variable and the perturbed value; Monte Carlo counts such realisations, leaves them out of the statistics and reports the count with the probability of failure bounded both ways (all of them counted as failures, none of them). No printed rule was found for this; state it in the result as the module's own. List it under "Calls made".
7. An unknown method name is refused with the list of names, in every entry point and the adapter (D49). An unknown `f_interslice` likewise.
8. Every search result counts the trials left without a solution, by family and by method, and warns when the governing family lost trials this way (UFC p. 362).
9. **One solver for a method, in a search and on one surface.** A search's own scorer and the single-surface solver must return the same solution for the same surface: the same rule for the moment axis, the same tolerance, the same test of convergence. Today a noncircular search can accept a solution about a second, fixed moment axis that the single-surface solver does not confirm. The case found in step 1: ACADS 4's weak-layer search (`validation_examples/test_published_v028_e8.py::test_v028_weak_layer_search_against_print`, a strict xfail) returns Spencer 0.515 on a surface where `gle_fos` does not converge and Janbu's corrected value is 1.22, with the interslice force at -43.5 degrees; a plain family of surfaces along the seam reads 0.753 against the printed 0.760. Report what that search returns after your change. If it comes within the test's 3 % of print, remove the mark; do not tune the search to get there.

**Out of scope.** The m_alpha rule and the acceptability checks (step 3). Which method scores which family, and a polyline named for Bishop running Spencer (step 6). Cracks and reverse curvature (step 4). Inside `rapid_drawdown.py`, change only the calls that the removed solvers break; the stages themselves, and the fall-back to Ordinary Method of Slices estimates when stage 1 does not converge, belong to step 9. The settings of the GLE solver that the account could not find in print (D29) stay as they are, apart from item 3.

**Benchmarks.** All of section 11. Expect movement, and explain each, on: #85 with and without its support (Spencer 1.551, GLE 1.573; printed GLE 1.575, Bishop 1.531); #86 with tiebacks (1.612 / 1.606 / 1.606; printed 1.629 / 1.620 / 1.622); ACADS 1(a) (0.985; printed 1.00, Slide2 0.986); V-043 Spencer (1.1859; Slide2 1.209 circle); V-031; V-044 (the account notes a degenerate free Spencer solution on this ponded slope; printed 1.603 / 1.599 / 1.599); the Newmark case V-039 (Ky 0.139); the FOSM and Monte Carlo cases (B4, V-030, V-042, V-045). ACADS 4 since step 1 (V-028, B11): on its declared family of surfaces along the seam, with the printed loads, Spencer 0.753, Morgenstern-Price 0.722, Janbu corrected 0.728 against the printed 0.760, 0.720, 0.734; its weak-layer search is item 9's case; its calc-package report prints 1.161 from the old approximate solver on the search's surface, which must become "no number".

**Tests (failing first).** A circle the rigorous methods cannot solve (GM-139's odd circle: a centre below the ground on a phi = 0 slope) returns no number where today it returns about 0.05. A Spencer circular search scores no circle by another solver and counts those without a solution. A Bishop solution stopped at its iteration limit gives no number. A Monte Carlo run and a FOSM run each meeting a surface with no solution. An unknown method name. The yield acceleration by Spencer uses Spencer's own solution at every trial kh.

---

## Step 3: the printed acceptability checks

Branch `claude/gm-140-step3-acceptability-checks`. Requires on master: steps 1 and 2. Second of the series.

**The rule.** Once a method has solved a surface (step 2) and the surface is one sliding mass (GM-122), every further test is a printed acceptability check whose finding is reported with the surface: which check, which slice, what value. A search keeps a flagged surface by default and says so; refusing on a check is a stated option. Nothing is left out silently, and circles and polylines meet the same checks.

**Owner call.** OWN-102, as written in CALLS.md, with one limit added on 2026-11-13: Duncan, Wright and Brandon (2014, p. 93) print no Ordinary Method of Slices equation with known forces, so the comparison of Bishop's factor of safety with the Ordinary Method of Slices value applies only to circles with no supports and no seismic coefficient, and the result says when it was not made and why.

**Departures and print.** D20, D21, D22, D50, D19; GM-130, GM-56. Whitman and Bailey (1967, p. 487, Eq. 5): the user is warned if, for any slice, the numerator becomes negative or the denominator becomes negative or small, "for example, if cos(theta) + tan(phi) sin(theta) / F < 0.2 ... the user must examine the result in detail". Duncan (1996, p. 580): calculate the Ordinary Method of Slices value for each circle when Bishop's method is used; where Bishop's F is below it, Bishop's solution has numerical problems and the Ordinary Method of Slices value is the better answer. EM 1110-2-1902 C-10a (pp. C-33 to C-34): calculated normal and interslice forces should be examined; tensile forces should not exist on the sides or bottom of slices; the line of thrust should be within the sliding mass. EM C-10b (pp. C-34 to C-36) prints four remedies for a steep exit, in this order: flatten the toe of the surface (Eq. C-32: an exit at alpha = -(45 deg - phi'/2) for a horizontal ground surface and interslice force); change the side-force inclination; use the Ordinary Method of Slices; give the exit zone a passive strength as cohesion with phi = 0 (the only one usable with Bishop's method). UFC 3-220-10 pp. 361 to 362. Bishop (1955) Eq. 13, p. 10 writes the frictional term as W - ub with no lower limit.

**What changes.**

1. **m_alpha below 0.2 (OWN-102).** A search keeps the surface, flags it, and reports where the low m_alpha was (the surface and the slice). For every Bishop circle without supports or a seismic coefficient the Ordinary Method of Slices value is computed for the same circle and reported beside Bishop's; where Bishop's is the lower, the Ordinary Method of Slices value is taken for that circle and the result says so. Today's refusal stays as a stated option. The exemption of phi' = 0 slices stays, stated in the result as the module's own reasoning (no printed source); "every slice" stays an option. The warning names EM C-10b's remedies; do not build the remedies as automatic changes to the surface.
2. **Bishop's numerator (D21).** The default follows Bishop's Eq. 13 as printed: W - ub is not set to zero when negative. The result names each slice where ub exceeds W, as Whitman and Bailey ask. Today's clamp stays as a stated option. Do the same for the Ordinary Method of Slices' N' = W cos(alpha) - ul (Bishop 1955 Eq. 9, p. 10). In the GLE solver report the slices whose effective normal was held at zero; do not change its clamps (D29). The lead made this call: list it under "Calls made".
3. **Compression and the line of thrust (D22, D50).** For Spencer, Morgenstern-Price and GLE the result reports, per surface: the slices with a negative effective normal force on the base; the slice sides with a tensile interslice force; and the slices where the line of thrust, not clipped, lies outside the sliding mass. The figure may still clip the line for display, but says when it left the mass.
4. **In a search**, the result gives the flags the critical surface carries, how many trials carried each flag by family, and, when the critical surface carries a flag, the lowest factor of safety among the surfaces that carry none, beside the headline. The headline stays the lowest solved surface (with item 1's substitution). The lead made this call: list it under "Calls made".
5. **The noncircular screens (D22).** Span, slice count, the refusal of a factor of safety below 1.5 on a jagged surface, agreement between moment axes and agreement between slicings are numerical stand-ins. Measure first, on every noncircular case of section 11: per screen, how many trials it refuses, and how many of those the printed checks of item 3 flag. A screen that refuses only surfaces the printed checks flag is removed. A screen that refuses surfaces the printed checks pass stays, counted in the result and named in `choices` as the module's own, not found in print; the CHECK file says what it catches. Add no new screen.
6. **One below-base test (D19).** The toe-circle family uses the same test as the other families (GM-128's rule).
7. **The direct march where the lagged iteration cannot step (from step 2).** Step 2's review proposed solving Spencer's lambda by the direct march on every run without supports where the lagged iteration's denominator vanishes. It was built, measured and kept out of step 2. On the reviewer's phi = 0 circle (ground (0,0), (15,0), (30,10), (50,10); cu 25 kPa, gamma 18; circle (32,7) R 9) it gives Spencer 2.9458 against an independent 2.9435; but five tests then fail, because states that stand only on the solver's clamps become "converged": `test_m_alpha::test_w11_surface_has_no_spencer_solution` (1.56 at every slicing, with the m_alpha clamp and base tension), `test_search_robustness::test_spurious_rigorous_states_were_slicing_artefacts[sandstone]` (1.192 at 30 and 41 slices only), `test_search_robustness::test_a_refused_lowest_surface_is_reported` (0.710 where no surface held), `test_m_alpha::test_w11_de_search_runs_its_whole_budget` (465 surfaces instead of 2,325) and `test_slope_stability_adapter::TestNewmark::test_yield_acceleration` (no ky at a trial kh of 1.0); ACADS 4's search took 84.5 s instead of 23 s. The strict xfail `test_the_march_is_tried_where_the_lagged_iteration_cannot_step` (within 1 % of 2.9435) holds the case. Decide it here. The lead's view: a state that stands only on the m_alpha clamp or on tension on a slice base is not the method's own solution; with that rule (such a state refused, and named as item 2 asks) the march could go in. The patch is held outside the repository.

**Out of scope.** Which method scores which family (step 6). New generators of noncircular surfaces (step 7). Tension cracks as the remedy for tension at the crest (step 4): report the tension, do not add a crack. The search's ranges.

**Benchmarks.** All of section 11. Expect movement on every search that refused surfaces for m_alpha: #85 (Bishop with its support 1.372, discard 1.508 against 1.511 with every slice checked; printed 1.531, Duncan and Wright 1.51); #85 passive (1.254; printed 1.324); #86; ACADS 1(a) (printed 1.00); V-054 toe circle (2.346; printed 2.16); V-032 (1.315; printed 1.366); V-042 (1.3334; printed 1.334); Duncan's Example 1 (0.703 and 0.676; Taylor 0.706 and 0.679); the vertical cut (0.842; Taylor 0.851); and the noncircular cases V-028 (after step 1), V-043 (1.234; Slide2 1.188) and the weak layer over rock (B14). ACADS 4's weak-layer search is the test case for items 3 and 4: after step 2 it reads 0.830, a solved value, against the printed 0.760 (the 0.515 that step 1 left had tensile interslice forces across the back wedge and a negative effective normal force on 11 of 30 slices); the declared family of surfaces along the seam reads 0.753 / 0.722 / 0.728 against the printed 0.760 / 0.720 / 0.734. The result must flag what the search's surface carries and show the lowest surface without a flag beside it. The strict xfail at 3 % stays: flag and report, never tune the search to the print. The generator of such surfaces is step 7, cut this week, so the xfail outlives the week. GM-130 records the circle-above-toe family's Janbu value moving from 1.033 to 1.138 under the old screen: re-measure it.

**Tests (failing first).** A Bishop search on a slope with a steep-exit circle keeps and flags it, with the slice named and both values reported; the same search with the refusal option reproduces today's count. A circle with supports says the comparison was not made. A slice with ub above W is named, and the as-printed and clamped values differ as arithmetic says. A Spencer surface with tension at the crest reports the slices and sides. A surface whose line of thrust leaves the mass says so.

---

## Step 6: one method across the search

Branch `claude/gm-140-step6-one-method`. Requires on master: steps 1, 2 and 3. Third of the series.

**The rule.** Every family of surfaces in a search is scored by the method named, with the interslice function named, so the reported minimum compares like with like. A family the named method cannot score is reported under the method that did score it, apart from the others.

**Departures and print.** D13, D42, D26, and D49's label part. EM 1110-2-1902 4-3b(9)(a), p. 4-8 (state the method used). EM C-2b, p. C-16: the Ordinary Method of Slices "can only be used with circular slip surfaces". EM C-5a, p. C-26: Spencer's method extended to noncircular surfaces. FHWA NHI-06-088 Table 6-1, p. 6-21 (Bishop the minimum method; Spencer where available). UFC 3-220-10 Table 7-7 step 5, p. 374. No printed rule on comparing surfaces solved by different methods was found (the account's D42); the case is that a search minimises the factor of safety of the method it reports (GM-138).

**What changes.**

1. `search_critical_surface` and the drawdown search take the interslice function (`f_interslice`) and every family and every kind of search honours it: grid, entry and exit, toe circles, circles above the toe, planes, random, differential evolution, particle swarm and weak layer. Particle swarm and the weak-layer search no longer always use Spencer; Morgenstern-Price and GLE no longer always use the half-sine.
2. Planes are scored by the method named, no longer always by Janbu's simplified method.
3. A screen by another method (the Bishop screen of the toe circles and of the main search, the refinement walk's choice of radius) may stay as a stated run-time device only if every value the family reports, and every comparison between families, is the named method's own factor of safety on the surface reported. `choices` names the screen.
4. Where the named method does not apply to a family's surfaces (Bishop's method and the Ordinary Method of Slices on a plane or a polyline), or solves none of them, that family is scored by Spencer's method; if Spencer's method solves none of them, by Janbu's simplified method. The family is reported apart, under the method that scored it, with the reason. The order is the lead's call (a complete method first, as the EM and FHWA prefer): list it under "Calls made".
5. The headline. The reported factor of safety is the lowest over all families. When the governing family was scored by a method other than the one named, `method_used` and a warning say so, and the result also gives the named method's own lowest value. The lead made this call (the lower value governs; both are stated): list it under "Calls made".
6. Each entry of `candidates` carries the method and the interslice function that scored it, truthfully (today toe-circle and planar results are labelled "Gle" or "Morgenstern_price").

**Out of scope.** New families or generators (step 7). The correction factor of Janbu's simplified method (D28: not found in print; leave it, say so). Cracks, water, supports.

**Benchmarks.** All of section 11. Expect movement on: #86 without supports, c' = 0 (planes 0.942; printed tan 37 deg / tan 38.66 deg = 0.942); the planar family against its closed forms (#86, Duncan and Wright's Examples 3 and 4: +0.11 to +0.15 %); V-035 and V-036 (Slide2 #79 and #81: referee 1.44 and 1.15); V-054 (2.346; printed 2.16); #85 and #86 with supports; V-043; V-028 (after step 1); B14. GM-56 records that Spencer and GLE refuse thin planes: measure, for each plane family in section 11, how many of its surfaces the named method solves, and report it. If a rigorous method named loses the infinite-slope mechanism on a cohesionless face, item 4's order must still report the family: show that it does.

**Tests (failing first).** Spencer named: the planes are scored by Spencer. GLE with the constant function named: every family's entry says GLE, constant. Bishop named on a cohesionless face: the planar family is reported apart with its method, the headline and the named method's own lowest value are both in the result. A particle-swarm and a weak-layer search with Morgenstern-Price named use it.

---

## Step 4: tension cracks and reverse curvature as printed

Branch `claude/gm-140-step4-cracks-reverse-curvature`. Requires on master: step 1. Runs in parallel with the series and with steps 5a, 8, 9 and 10.

**The rule.** A tension crack ends the slip surface. The soil upslope of the crack is left out of the computation, the slices end at a vertical boundary at the crack, and water in the crack acts on that boundary. The crack's depth is the one the print gives. The same rule closes a reverse-curvature circle.

**Owner call.** OWN-103, as written in CALLS.md.

**Departures and print.** D39, D24; GM-18's crack item. EM 1110-2-1902 2-2f(3), p. 2-6: the soil upslope from the crack "is then ignored in the stability computations ... by terminating the slices near the crest of the slope with a slice having a vertical boundary"; water in the crack is a force on the boundary of the slice next to it. 2-2f(4), p. 2-6: the depth is the one producing the minimum factor of safety, where tensile but not compressive stresses are eliminated. 2-2f(5), pp. 2-6 to 2-9, total stresses:

    z_t = (2 c_D / gamma) tan(45 deg + phi_D / 2)        (Eq. 2-4)
    c_D = c / F                                           (Eq. 2-5)
    phi_D = arctan(tan(phi) / F)                          (Eq. 2-6)

2-2f(6), p. 2-9: for effective-stress analyses the depth can also be estimated from Rankine's active earth pressure theory with c' and phi' and the appropriate pore water pressures (no equation is printed). C-10c, p. C-36: tensile stresses should be eliminated before an analysis is considered acceptable; the slip surface is terminated where it reaches the bottom of the crack; the depth is found (1) by assuming a range of depths and taking the one of minimum factor of safety, or (2) as the depth over which Rankine's active pressures are negative, "for total stresses and homogeneous soil"

    d_crack = 2 c_D / (gamma tan(45 deg - phi_D / 2))     (Eq. C-36)

with the same developed strengths; where Eq. C-36 gives a depth greater than the height of an embankment on a weak foundation, the crack is as deep as the embankment and does not extend into the foundation. Slide v6 Tutorial 16, p. 16-1: "A tension crack essentially terminates the slip surface."

**What changes.**

1. **The default model** ends the surface at the crack (today's "truncation"). Today's default, which keeps the cracked wedge as driving soil without strength, stays as a stated option.
2. **The side.** The crack is at the upper end of the slip surface, the end where the ground is higher (EM 2-2f(3): near the crest, the soil upslope ignored), whichever way the slope faces. An explicit side stays as an input.
3. **The depth** is one of: given by the user; Rankine's depth with the developed strengths (Eqs. 2-4 to 2-6 and C-36), F being the factor of safety of the surface itself, found by repeating the solution until the depth stops changing; or the depth of minimum factor of safety from a scan (2-2f(4), C-10c(1)). Eq. C-36 is printed for total stresses and one soil. Use it where the cracked depth lies in one soil and, for an effective-stress layer, above the water table (2-2f(6), the lead's reading: list it under "Calls made"); elsewhere use the scan and say so. A Rankine depth that passes the bottom of the soil it was computed for stops at that bottom (C-10c's embankment rule), and the result says so.
4. **Reverse-curvature circles (OWN-103).** Such a circle is closed by a vertical crack of the printed depth (item 3, Rankine's by default, the scan as the option). Where the overhanging part of the arc runs deeper than that crack, the circle is discarded, counted, and located in the result. "Discard all" stays a stated option. The crack from the centre's level to the ground, which no printed source gives, is removed.
5. Crack water acts on the vertical boundary as today (0.5 gamma_w z_w^2 at a third of its depth), on the side the crack is on.
6. The rule holds for a given surface and in every search family, every method, the drawdown stages and the probabilistic runs.

**Out of scope.** The m_alpha rule and the other acceptability checks (step 3): do not change which surfaces a search refuses for any other reason. Supports (step 8), including how much of a support Bishop's method counts on #85.

**Benchmarks.** All of section 11. ACADS 1(b), Slide2 #2 (V-026): today 1.485 / 1.486 / 1.482 by the old default and 1.603 / 1.599 / 1.600 with the surface ended at the crack; printed Bishop 1.596, Spencer 1.592, GLE 1.592, referee 1.65. The problem text is in `slide2_selected_problems.txt`: it takes the crack depth from Craig's Rankine expression with the full strengths, not the developed ones, and fills the crack with water. Check the test's inputs against it and enter that depth as the given depth; report beside it what Eq. C-36 with the developed strengths would give, but do not pin the benchmark on it. V-031, Slide2 #57 (a dry crack, text in the same file; 1.421 / 1.425 / 1.409; printed 1.417 / 1.422). #85 with its support (Bishop 1.372 today on a reverse-curvature circle; 1.508 with all such circles discarded; printed 1.531, Duncan and Wright 1.51) and #85 passive (1.254, discard 1.316; printed 1.324): measure the value under OWN-103 and report all three readings. V-043's clay crest.

**Tests (failing first).** A crack on a slope rising towards high x and one rising towards low x both land at the crest end. The default leaves the wedge out (weights and the surface's end as arithmetic gives). Rankine's depth reproduces Eq. C-36 by hand for a phi = 0 clay and for a c, phi soil at the converged F. A reverse-curvature circle whose overhang is shallower than the crack is closed at the printed depth; a deeper one is discarded and counted. The scan returns the depth of minimum factor of safety on a case where a hand table shows the minimum.

---

## Step 5a: pore pressure as printed

Branch `claude/gm-140-step5a-pore-pressure`. Requires on master: step 1. Runs in parallel with the series and with steps 4, 8, 9 and 10.

**The rule.** The pore pressure on a slice base is the printed one for the kind of water input and the kind of strength the layer has, from one function that every caller uses.

**Owner call.** OWN-104, as written in CALLS.md: the cos-squared correction is built, off by default, and every result with a sloping water line names which reading it used.

**Departures and print.** D2, D3, D4, and D16's slicing half; GM-18's seepage item. FHWA NHI-06-088 Eq. 6-16, p. 6-14: u = h_w gamma_w cos^2(alpha_w); p. 6-14 also puts slice boundaries at the water surface's intersections. UFC 3-220-10 p. 355 (7-5.1): the water table "corrected to an approximate piezometric surface based on the slope". EM 1110-2-1902 2-2c, p. 2-4 and Table 2-1, p. 2-2: pore pressure is zero in total-stress analyses; note 2 of the table: total-stress envelopes with phi > 0 apply only to partly saturated soils. EM C-7, p. C-29: r_u = u / (gamma z) for a uniform slope. Read the FHWA and UFC pages themselves in the reference library.

**What changes.**

1. **The water line's kind (D2, OWN-104).** The caller says what the line is. A piezometric line (measured heads; the default): the full vertical head below it, as today, never corrected. A water table with seepage parallel to it: u = h_w gamma_w cos^2(alpha_w), with alpha_w the slope of the line above the base's mid-point. `choices` states the premise (seepage parallel to a straight water table) and that the uncorrected head is the lower factor of safety. Where the line lies above the ground (ponded water) no correction applies.
2. **r_u in a layered column (D3).** u = r_u times the total overburden stress at the base, summed over the soils in the column by step 1's lookup. EM C-7 defines r_u for a uniform slope only; state in the result that the layered form is the module's reading of that definition. Say in the CHECK file whether a surcharge or ponded water counts in the overburden and why.
3. **Total-stress layers (D4).** No pore pressure is subtracted on the base of a slice whose layer is analysed in total stresses (EM 2-2c). A total-stress layer given phi > 0 below the water line draws a warning that cites Table 2-1's note 2.
4. **The pond's edge (D16).** Where the water line meets the ground is a slice boundary, for circles and polylines.
5. The rule holds for every method, every family, a pore-pressure grid given with a line (say which governs where; unchanged), the drawdown stages' use of the same function, and the probabilistic runs.

**Out of scope.** The stresses that SHANSEP and Hoek-Brown strengths are taken at (D40: noted, not fixed). How the drawdown stages match their slices (step 9). Line, inclined and varying loads, the vertical seismic coefficient, the statement on surcharge inertia and the warning for Bishop's method with a seismic coefficient (step 5b, later).

**Benchmarks.** All of section 11. With the default (no correction) only D3, D4 and the pond-edge boundary can move a value: say which rows they move. Rows with water or r_u: the Fredlund and Krahn slope with r_u = 0.25 (1.606 / 1.758 / 1.763 / 1.764; printed 1.607 / 1.766 / 1.761 / 1.764) and the composite surface with r_u (1.138 / 1.132; printed 1.118 / 1.118); V-031 (printed 1.417 / 1.422); V-044 submerged (printed 1.603 / 1.599 / 1.599); V-029 (the pore-pressure grid); V-028 (a piezometric surface, so not corrected); the drawdown rows. For the correction itself, pin a printed case: look in the reference library for a worked use of Eq. 6-16 or for the infinite slope with seepage parallel to the face, cite the page, and show that a long plane with the water table at the surface reproduces the printed closed form with the correction on. If the library prints neither, say so and test the equation by hand arithmetic only.

**Tests (failing first).** A sloping water table read both ways gives u at a base as arithmetic says. A two-soil column with r_u. A total-stress layer below the water line carries no u; with phi > 0 it warns. A partly ponded face has a slice boundary at the water's edge. The result of a run with a sloping line names the reading; with a level line it does not need to.

---

## Step 8: supports as printed

Branch `claude/gm-140-step8-supports`. Requires on master: step 1. Runs in parallel with the series and with steps 4, 5a, 9 and 10. The pages of Duncan, Wright and Brandon (2014) this step needs are in hand since 2026-11-13 (the owner's copy, in no repository): everything from that book below is given by page and equation number.

**The rule.** A support enters a solution as a known force, the way the print gives it: by Method A or Method B, along the reinforcement's own direction, and no larger than the reinforcement can carry where the slip surface crosses it.

**Departures and print.** D34, D35, D36, D37; GM-82; OWN-59; GM-140's note on `reinforcement._line_slip_intersection`.

- Duncan, Wright and Brandon (2014), section 8.2, pp. 159 to 160. Method A: the forces are allowable forces and are not divided by the computed factor of safety, F = soil resisting moment / (overturning moment - reinforcement moment) (Eq. 8.1; Eq. 8.2 in terms of stresses). Method B: the forces are ultimate (long-term capacity) forces and are divided by F as the soil strength is, F = (soil resisting moment + reinforcement moment) / overturning moment (Eq. 8.3; Eq. 8.4). Method A is preferable (p. 160). A program must say which it uses, so that the right measure of force is entered.
- Same book, section 6.5.3, pp. 91 to 93: Bishop's simplified method with known forces,

      F = SUM[(c' dl cos(a) + (W - F_v - u dl cos(a)) tan(phi')) / (cos(a) + sin(a) tan(phi') / F)] / (SUM(W sin(a)) - M_n / r)      (Eq. 6.78)
      F_v = -P cos(beta) + T sin(psi), positive upward                                                                                  (Eq. 6.75)

  M_n is the net moment about the centre of all known forces except the weight (seismic kW, external loads P, reinforcement T), counterclockwise positive; T acts at psi from the horizontal where the reinforcement crosses the slip surface (Eq. 6.70 gives its moment as T times its arm about the centre). Horizontal reinforcement enters moment equilibrium only; inclined reinforcement enters both, and the vertical part is always included (p. 93). Because only vertical equilibrium is used, the method neglects what a horizontal force adds to the normal stress on the slip surface; care is needed where large horizontal forces contribute, though results are comparable to complete-equilibrium procedures (p. 93). No Ordinary Method of Slices equation with known forces is printed (p. 93).
- Same book, section 8.6, pp. 163 to 164: the force acts along the reinforcement's original orientation (recommended, the more conservative); a force parallel to the slip surface is the other extreme.
- Same book, section 8.4, pp. 161 to 163: the long-term capacity along a layer is the lesser of T_lim and the pullout resistance measured from each end (Fig. 8.2); pullout from the end, T_po = 2 gamma z alpha F* L_e under a constant depth of fill z (Eq. 8.8), and T_po = gamma tan(beta) alpha F* L_e^2 beneath the slope face (Eq. 8.9); Table 8.3 (after FHWA 2009): alpha 0.6 geotextiles, 0.8 geogrids, 1.0 steel strips and grids; F* 0.67 tan(phi), 0.8 tan(phi), 1.0 tan(phi); T_all = T_LTDS / F_R (Eq. 8.10), F_R 1.3, 1.5 or 2.0 by consequence and uncertainty (Table 8.4).
- GEC 11 Vol. II section 9.1, p. 9-1, section 9.2.6, p. 9-14, and p. 9-24; GEC 7 Eq. 5.2, p. 105 (a nail's load is the least of its pullout behind the surface, the bar, and the facing with pullout in front) and Eqs. 5.6 to 5.9, pp. 119 to 120. Both are in the reference library: read them.
- Duncan (1996), p. 583, as the account quotes it.

**What changes.**

1. **The method stated.** Every result with supports names Method A or Method B and the measure of force the input is taken to be (allowable or ultimate), citing Eqs. 8.1 to 8.4. Method A stays the default.
2. **Bishop's method.** Check the module's support terms against Eq. 6.78 with Eqs. 6.70 and 6.75: the arm, the sign and the vertical part. Where they differ, follow the print. A nearly horizontal support crossing near the centre's level has a small arm and so a small effect: that is the method as printed, with the book's caution. Keep the short-arm warning, cite p. 93, and do not adjust how much of the support counts.
3. **The Ordinary Method of Slices with supports** has no printed form in the references held. Look once more in the reference library; if none is found, the method is refused where supports cross the surface, with that reason, and the comparison table leaves it out and says why. The lead made this call: list it under "Calls made".
4. **Direction.** The force acts along the reinforcement's own orientation by default; parallel to the slip surface is a stated option. Fix the geosynthetic's first crossing and direction, and settle a stabilizing pile's "normal" direction, against this rule (GM-82).
5. **What a layer can carry at the crossing (D34).** A geosynthetic's force at the slip surface is the lesser of its long-term strength and its pullout resistance on each side of the crossing. The reference library's GEC 11 and Eqs. 8.8 to 8.10 above both print the pullout; offer both, state the default (the one the module's inputs already follow) and why. An input the print treats as a finding about the fill or the product (F*, alpha, F_R) is asked for, not filled in silently, unless the page prints a default and the result names it.
6. **A nail (D35).** Add the front term of GEC 7 Eq. 5.2.
7. **A support whose pull assists sliding (OWN-59).** Both readings offered; slack (the support counts nothing) is the stated default.
8. **The legacy single force** without an elevation is refused (D37).
9. **The crossing** of a support and a slip surface is found exactly (a line with a circle, a line with each segment of a polyline), not by sampling 200 points.

**Out of scope.** Stabilizing piles' Ito-Matsui force as its own method (OWN-13). Reverse-curvature circles (step 4). New support types.

**Benchmarks.** All of section 11, and two printed ones to add:

- **The check problem of Fig. 8.1 (pp. 160 to 161).** A slope 20 ft high at 2 vertical on 1 horizontal; gamma = 100 pcf, phi = 0, c = 500 psf; a firm layer whose top is at the toe's level; one reinforcing force of 10,000 lb/ft acting horizontally 10 ft above the toe. Bishop's simplified method, circles tangent to the top of the firm layer; the critical circles exit slightly above the toe. Printed: F = 2.19 by Method A, 1.72 by Method B, within 1 or 2 %. Convert to SI in the test and say so. A hand check of the moment equation on tangent circles gives 1.72 for Method B and 2.21 for Method A, the latter on a circle whose centre is at the crest's level, the limit before reverse curvature: report the critical circle of each method.
- **The slope of Fig. 8.4 (pp. 164 to 165), which the validation set carries as Slide2 #86.** A fill on rock 24 ft high at 1.25 horizontal to 1 vertical; gamma = 130 pcf, c' = 0, phi' = 37 deg; six layers of reinforcement starting at the bottom of the slope, 4 ft apart vertically, each 20 ft long, each with a tensile force of 800 lb per foot that decreases linearly to zero over the last 4 ft of embedded length (Fig. 8.4a: the full 800 lb from the face to 16 ft along the layer, zero at 20 ft). Spencer's procedure, circles: minimum F = 1.62 by Method A and 1.46 by Method B. The account lists this case with five tiebacks at 1.612 / 1.606 / 1.606 against Slide2's 1.629 / 1.620 / 1.622. Compare the test's inputs with the figure's as given here, say what differs, and pin both methods.

Also expect movement on #85 (one support: active GLE 1.573, Bishop 1.372; passive Bishop 1.254, GLE 1.379; printed 1.575, 1.531, 1.324, 1.378), V-040 (#54, a row of micro-piles: active 1.223, passive 1.203; printed 1.193), V-043 and V-056 / V-057.

**Tests (failing first).** The two printed benchmarks. A geosynthetic crossed near its end carries its pullout, not its full strength. A nail limited by its front term. A horizontal support adds nothing to Bishop's vertical equilibrium; an inclined one adds T sin(psi). A support pulling with the slide counts nothing by default and is named. A crossing next to a circle's vertical end is found. The legacy force without an elevation is refused.

---

## Step 9: rapid drawdown as EM 1110-2-1902 Appendix G prints it

Branch `claude/gm-140-step9-rapid-drawdown`. Requires on master: step 1. Runs in parallel with the series and with steps 4, 5a, 8 and 10. Your edits belong in `slope_stability/rapid_drawdown.py` and its result class; step 2 changes the same file only where a removed solver is called.

**The rule.** Each stage is computed as Appendix G prints it, each stage's factor of safety is reported, and the answer is the factor of safety of the last stage computed.

**Departures and print.** D38; OWN-17; GM-15; GM-130's drawdown item; GM-135; D16's drawdown half. EM 1110-2-1902 Appendix G, pp. G-1 to G-23, and Table 2-1, p. 2-2. The manual is not in the repository; the equations and the worked example's values below were read from the page and checked by hand, and are all you need.

- First stage (G-3a, p. G-2): pore pressures before drawdown; on each base sigma'_c = N / dl (Eq. G-1) and tau_c = S / dl (Eq. G-2).
- The strength of a soil consolidated isotropically (K_c = 1), from the R envelope's c_R and phi_R (p. G-6):

      d = c_R cos(phi_R) cos(phi') / (1 - sin(phi_R))                 (Eq. G-5)
      psi = arctan[ sin(phi_R) cos(phi') / (1 - sin(phi_R)) ]         (Eq. G-6)
      tau_ff(K_c = 1) = d + sigma'_c tan(psi)                         (Eq. G-9, p. G-7)

  or d and psi measured directly from consolidated-undrained tests (G-3b(1)(a) and (b), pp. G-4 to G-5).
- The strength of a soil consolidated at failure (K_c = K_f): tau_ff(K_c = K_f) = c' + sigma'_c tan(phi') (Eq. G-10, p. G-7); K_f = (1 + sin(phi')) / (1 - sin(phi')) where c' = 0 (Eq. G-7), and Eq. G-8 where it is not (already in the module and checked).
- The consolidation stress ratio of the slice (Eq. G-11, p. G-7; repeated as G-24, p. G-17), held between 1 and K_f (G-3b(2)(c), p. G-7):

      K_c = [ sigma'_c + tau_c (sin(phi') + 1) / cos(phi') ] / [ sigma'_c + tau_c (sin(phi') - 1) / cos(phi') ]

- The second-stage strength (Eq. G-12, p. G-8):

      tau_ff = [ (K_f - K_c) tau_ff(K_c = 1) + (K_c - 1) tau_ff(K_c = K_f) ] / (K_f - 1)

- Second stage (G-3c, p. G-8): solved with tau_ff as cohesion, phi = 0 and zero pore pressure on the slices that do not drain freely; c', phi' and the pore pressures after drawdown on free-draining ones; water left against the slope is an external load.
- The drained check (G-3d, p. G-8): sigma = N / dl with N from the second-stage computation (Eq. G-13); sigma'_d = sigma - u with u after drawdown (Eq. G-14); s_d = c' + sigma'_d tan(phi') (Eq. G-15). Where s_d is not below tau_ff on any slice the second stage's factor of safety is the answer.
- Third stage (G-3e, pp. G-8 to G-9): "the same conditions as for the second-stage computations" except that the slices whose drained strength is lower take c', phi' and the pore pressure after drawdown. Its factor of safety is the answer.
- The Corps of Engineers' 1970 procedure (G-2, pp. G-1 to G-2) uses the R envelope's c_R and phi_R as entered, in a composite envelope with the effective-stress one; it has no third stage.
- The example's water (G-4b, p. G-9): a horizontal piezometric line at the maximum pool, and another at the lowered pool, "for simplicity"; seepage analyses would often be appropriate.

**What the worked example shows.** A local check on 2026-11-13 ran the module slice by slice against the manual's example (Figs. G-10 to G-13). Stage 1 is sound (Spencer 2.200, Bishop 2.195; printed 2.20). Stage 2 reads 1.375 against the printed 1.52, for two reasons, both in how tau_ff is built:

1. The K_c = 1 strength uses c_R and phi_R as entered. The manual converts them by Eqs. G-5 and G-6 (1,200 psf and 16 deg become 1,379 psf and 18.2 deg; Fig. G-5 prints both pairs). The module's strength is 13 % low on every slice.
2. K_c is not Eq. G-11. The manual has 1.70 or 1.71 on all twelve slices; the module's own rule (with the base inclination and two special branches) gives 1.68 to 3.00.

With both printed rules put in, the module's own solver gives 2.200 / 1.513 / 1.434 (Spencer, 50 slices) against the printed 2.20 / 1.52 / 1.44, with tau_ff within 0.02 ksf of Fig. G-12b on every slice and the same three slices drained. Either rule alone: 1.444 (the envelope), 1.375 (K_c). GM-15's steep case is cause 2.

**What changes.**

1. **The K_c = 1 strength** of the Lowe and Karafiath and the three-stage procedures follows Eqs. G-5, G-6 and G-9. A layer may give d and psi directly instead. The Corps two-stage procedure keeps c_R and phi_R as entered.
2. **K_c** follows Eq. G-11, held between 1 and K_f. The base inclination and the special branches go.
3. **The second stage is solved and kept**: its factor of safety and its N on every slice. For Lowe and Karafiath it is the answer.
4. **The drained check uses the second stage's N** (Eqs. G-13 to G-15). A negative sigma'_d gives s_d = c' (the manual prints s_d = 0.00 for slice 1, where sigma'_d = -0.41 and c' = 0; for c' above zero this is a reading: list it under "Calls made"). The two present bases (an Ordinary Method of Slices normal, and a drained GLE solution) are not the EM's and go; a caller who names an old value is told the rule. This settles OWN-17.
5. **The third stage is solved only if a slice needs it**, with c', phi' and the after-drawdown pore pressure on those slices (not the drained strength entered as a cohesion) and the second stage's strengths on the rest.
6. **Water, stated.** Before drawdown: the seepage line when one is given, otherwise the level pool of the manual's own example, named in the result as that assumption. After drawdown: add the matching input for a line; without it, the level lowered pool, named.
7. **The result carries each stage**: the factor of safety of stages 1, 2 and 3 (where computed), which stage the answer is from, and per slice sigma'_c, tau_c, K_c, K_f, the two envelope strengths, tau_ff, the second stage's N, the after-drawdown u, sigma'_d, s_d and which strength was used (the columns of Figs. G-10b and G-11b).
8. **One set of slices for all stages**: pair slices by index, and refuse a surface where the stages' slice counts differ (GM-135).
9. **A stage with no solution gives no number**: no estimates from the Ordinary Method of Slices in place of a first stage that did not converge.
10. The same procedure scores every trial of the drawdown search.

**The manual's example, to pin (EM G-4, pp. G-9 to G-11; Figs. G-5, G-6).** Convert to SI in the test and say so.

- Section: toe at El. 0; 3 horizontal to 1 vertical up to El. 74; 2.5 to 1 up to the crest at El. 110; one soil, 135 pcf above and below water. c' = 0, phi' = 30 deg; c_R = 1,200 psf, phi_R = 16 deg (d = 1,379 psf, psi = 18.2 deg). Pool from El. 103 to El. 24, level piezometric lines.
- Slip surface: a circle of radius 210 ft, centre 169.5 ft from the toe and 210 ft above El. 0 (tangent to El. 0; it meets the face at (72.0, 24.0) and the crest at x = 354.2 ft). Twelve slices, numbered from the crest; their edges, recovered from the printed widths, in feet from the toe: 354.2, 350.2, 330, 312, 294.5, 267.0, 245, 222, 200, 169.5, 135, 105, 72.0.
- Printed factors of safety. Improved procedure by the Simplified Bishop Method: 2.20, 1.52, 1.44 (pp. G-15, G-18, G-20); by the Modified Swedish Method at Spencer's side-force inclinations (6.0, 12.2, 13.7 deg): 2.23, 1.52, 1.44 (pp. G-20 to G-22). Corps 1970 procedure by the Modified Swedish Method with side forces at 19.4 deg: 3.49, then 1.35 (pp. G-9 to G-15).
- Fig. G-10b (p. G-16), the Bishop example's strengths for the second stage, and Fig. G-11b (p. G-19), its drained check (ksf; N in kips):

| Slice | sigma'_c | tau_c | K_c | tau_ff (K_c = 1) | tau_ff (K_c = K_f) | tau_ff | N, stage 2 | u after | sigma'_d | s_d | Used in stage 3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.32 | 0.08 | 1.71 | 1.49 | 0.19 | 1.02 | -6 | 0 | -0.72 | 0 | drained |
| 2 | 1.44 | 0.38 | 1.71 | 1.85 | 0.83 | 1.49 | 52 | 0 | 1.48 | 0.86 | drained |
| 3 | 2.90 | 0.76 | 1.71 | 2.33 | 1.67 | 2.10 | 120 | 0 | 4.61 | 2.66 | undrained |
| 4 | 3.64 | 0.95 | 1.71 | 2.58 | 2.10 | 2.41 | 148 | 0 | 6.52 | 3.76 | undrained |
| 5 | 4.05 | 1.06 | 1.71 | 2.71 | 2.34 | 2.58 | 250 | 0 | 7.69 | 4.44 | undrained |
| 6 | 4.45 | 1.17 | 1.71 | 2.84 | 2.57 | 2.75 | 204 | 0.31 | 8.13 | 4.69 | undrained |
| 7 | 4.57 | 1.20 | 1.71 | 2.88 | 2.64 | 2.80 | 208 | 0.85 | 7.77 | 4.49 | undrained |
| 8 | 4.54 | 1.19 | 1.71 | 2.87 | 2.62 | 2.78 | 191 | 1.22 | 7.30 | 4.22 | undrained |
| 9 | 4.31 | 1.13 | 1.71 | 2.80 | 2.49 | 2.69 | 246 | 1.43 | 6.61 | 3.81 | undrained |
| 10 | 3.66 | 0.96 | 1.71 | 2.58 | 2.11 | 2.42 | 235 | 1.41 | 5.38 | 3.11 | undrained |
| 11 | 2.60 | 0.68 | 1.71 | 2.23 | 1.50 | 1.97 | 149 | 1.09 | 3.75 | 2.16 | undrained |
| 12 | 1.01 | 0.27 | 1.71 | 1.71 | 0.59 | 1.31 | 73 | 0.43 | 1.61 | 0.93 | drained |

  K_f = 3.00 on every slice. The printed areas are those of slices with straight bases, 0.9 % less than the area down to the arc, so pin strengths and stages, not the printed weights. Pin on the manual's twelve slices: tau_ff of every slice within the table's rounding, the three drained slices, and the three stages by Bishop within 1 %; and the stages by Spencer on the module's usual slicing within 1 % of 2.20 (the check explains the manual's 2.23 by its whole-degree base angles), 1.52 and 1.44.

**The Corps 1970 procedure and Slide2 #95: the lead's call (list it under "Calls made").** The manual's 1.35 (Slide2 prints 1.347) was computed by the Modified Swedish Method with the side forces at a fixed 19.4 deg, "for consistency with the original procedure ... although Spencer's Method is currently recommended" (G-5, p. G-9). The local check reproduces 3.49 and 1.34 that way; Spencer's method on the same stages gives 1.24. So:

- The module's two-stage procedure keeps Spencer's method and its answer (1.24 here). The result names the method and says that the manual's example of this procedure used the Modified Swedish Method at a fixed side-force inclination.
- The printed 3.49 and 1.35 are pinned through a force-equilibrium solution at the printed inclination of 19.4 deg (the GLE engine's force-equilibrium factor of safety at a fixed, constant side-force inclination: its march already does this arithmetic). Make it a library function used by the test and described in `DESIGN.md` as the manual's check method (EM C-4); do not add it to the agent's tools or to `method` names this week.
- Remove the pins that rest on a seepage line no page prints (1.34 for #95; 1.273, 1.370 and 1.450 for #96) and the conclusions drawn from them.

**Benchmarks.** All of section 11's drawdown rows, each stage beside print: the manual's example as above; Slide2 #96 at its 110 ft pool (printed 1.443; the check's scratch run under the printed rules gives 1.427, and cannot tell a 103 ft from a 110 ft pool by the final value); #98 (search: 0.837 and 0.938 today; printed 0.931 and 1.039, on a recovered section; the scratch run stays about 10 % low in every method, so report, do not tune); V-048; V-052 and its search (Pilarcitos: Corps 0.823, Lowe and Karafiath 1.047, three-stage 1.043; the scratch run moves Lowe and Karafiath from 1.279 to 0.944 under Eq. G-11, so GM-15's steep case is cause 2: re-measure V-052b); V-041's planar case (0.783; infinite slope 0.772).

**Not settled, to state in the CHECK file, not to guess.** Which method of slices Slide2 used for #95 and #96. A base in tension at the first stage (sigma'_c below zero): the manual prints no rule; state the one you use. Soils with c' above zero exercise Eq. G-8 with Eq. G-11 only on #98's recovered section. The after-drawdown line has no printed benchmark.

**Out of scope.** Seepage analysis. The pore-pressure function itself (step 5a). A Modified Swedish method for the agent's tools.

**Tests (failing first).** K_c by Eq. G-11 and the K_c = 1 strength by Eqs. G-5, G-6 and G-9 against the table's slices. The second stage reported on its own. The drained check from the second stage's N. No third stage when no slice needs it. A first stage without a solution gives no number. The manual's example, stage by stage, by Bishop and by Spencer. The 1970 procedure's 3.49 and 1.35 at 19.4 deg.

---

## Step 10: reporting as printed

Branch `claude/gm-140-step10-reporting`. Requires on master: step 1. Runs in parallel with the series and with steps 4, 5a, 8 and 9. One part waits for step 6 (named below).

**The rule.** A result and its report show what the print asks an engineer to be shown, and an input that changes nothing is not accepted.

**Departures and print.** D45, D46, D47, D49's label part; GM-132's slope part; the audit's rows 1 and 9 (`module_work/module_audit_2026-11-12/AUDIT.md` section 4, slope_stability). FHWA NHI-06-088 Fig. 6-13, p. 6-23 (contours of the minimum factor of safety over the trial centres) and p. 6-49 ("ordered output and plotting capability for the failure surface of 10 minimum safety factors"). GEC 11 Vol. II p. 9-24 ("a contour plot of lowest safety factors about trial failure circle centroids is recommended"). Duncan, Buchignani and De Wet (1987), Fig. 28: the contours should enclose the minimum. EM 1110-2-1902 4-3b(9)(b), p. 4-8: show the critical surface and the other significant surfaces with their factors of safety, and for circles the centre and radius; 4-2d(2), p. 4-2: the output designates the most critical surface and what other surfaces were analysed, and the engineer checks slice weights, strengths and pore pressures at the bottoms of slices.

**What changes.**

1. **A contour plot** of the minimum factor of safety over the grid of centres (D46), from the per-centre minima the result already stores, with the critical centre marked. The result says whether the lowest contour closes inside the grid.
2. **An ordered list** of the ten lowest surfaces (D47), each with its factor of safety, its family and its geometry (centre and radius, or the polyline), in the result and the report; the figure draws them.
3. **The slice table comes first** in the report (weights, strengths and pore pressures at the bases: EM 4-2d(2)).
4. **Inputs that do nothing (D45).** The calc packages' `time_budget_s`, `tol` and `FOS_required` are wired or removed; an unknown `f_interslice` is refused. Update the probes marked FROZEN for them.
5. **Figures at print size (GM-186).** The interslice-force figure's lettering (4.0 pt today); the note over the title of the cross-section; the key over the drawing in the slice-forces figure.
6. **`choices` and `defaults_used`, the slope part of GM-132:** "set" in place of `[True]` where a truthy value that is not a boolean runs the code; a name the outer method does not declare (`m_alpha_rule` in rapid drawdown) appears in `defaults_used`.
7. **A circle scored on its own in cohesionless soil** (the audit's row 9): the result of `analyze_slope` says that a single circle through soil with c' = 0 at the face overestimates against the infinite-slope mechanism, which the search scores with planes. A statement only.
8. **The method each family was scored with**, in the report: this part waits for step 6. If step 6's CHECK file is not on master when you start, leave the table out and say so.

**Out of scope.** Calc packages for rapid drawdown, the yield acceleration, FOSM and Monte Carlo (the audit's row 1): say what each would need; build none. DXF export of the slip surface. Any change to a computed value: if a value moves, something is wrong; stop and say so.

**Benchmarks.** All of section 11, to show no value moves.

**Tests.** The contour data and the closed-contour statement on a grid whose minimum is inside, and on one whose minimum is on the edge. The ordered list is sorted and complete. The report's first table is the slice table. Each formerly inert input either changes the result or is refused. The calc-package tests and the figure layout checks.

---

## Later, not this week: steps 5b and 7

- **Step 5b, loads as printed (D5, D6, D7).** Line, inclined and varying loads; an optional vertical seismic coefficient; a stated choice on whether a surcharge carries inertia; a warning for Bishop's method with a seismic coefficient (EM C-3b, p. C-18). Duncan, Wright and Brandon (2014) Eqs. 6.74 to 6.78, pp. 92 to 93, give Bishop's method with a load P normal to the top of a slice (F_v = -P cos(beta) + T sin(psi)) and say a vertical seismic force would enter the vertical equilibrium. After step 5a.
- **Step 7, search coverage as printed (D9, D12, D14, D43; GM-117, GM-89).** Face exits at all heights in the main search; noncircular surfaces generated from the critical circle and along weak layers; impenetrable layers and composite surfaces; several independent starts. After step 6; it needs its own section here before it starts.
