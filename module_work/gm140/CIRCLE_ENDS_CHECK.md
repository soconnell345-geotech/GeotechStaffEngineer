# GM-140 "Where a circle ends" (GM-343): check file

Branch `claude/gm-140-circle-ends`, from master c5ed3c5 (steps 1, 2, 3 and
4 merged; their CHECK files are on master). Brief:
`module_work/cloud_briefs/SLOPE_GM140.md`, section "Where a circle ends
(after step 3)". Step 6 was not on master when this was built; "merge
master before you push" found nothing new to merge (see "Tests").

## The rule

A circle's trial surface is an arc below the ground between two points
where the circle meets the ground. A circle that meets the ground at more
than two points is split at those points. Each arc with ground above it is
a trial surface of its own, solved and screened like any other (steps 2 to
4 apply to it unchanged), and a search scores each one. A mass with no
driving moment is refused, as any such surface is. Nothing is refused
merely for being one of two masses, and no arc is carried under the ground
in front of the toe. The result says which arc it is.

## What the print says

- **Circles with their lower end on the face.** FHWA NHI-06-088 p. 6-38
  (the account's GM-122 entry). **A search that covers face exits at every
  height:** GEC 11 p. 9-6. Both from the account's citations; neither page
  was re-read in the reference library for this step, because no formula
  or value is taken from them (they name the family, which already
  existed).
- **Taylor's stability number for a vertical face, phi = 0: 3.83** (the
  account's section 11, B16). For the test's cut (5 m, cu 20 kPa, 18
  kN/m3): F = 3.83 x 20 / (18 x 5) = 0.8511.
- **No held reference prints a rule for a circle that meets the ground
  more than twice.** D25: GM-122's refusal followed Slide2's documented
  behaviour (its error -105), not print. The split is the orchestrator's
  ruling of 2026-11-13 (the brief). The code, DESIGN.md, the tool
  description and the search's `choices` entry all say it is the module's
  rule and not a printed one.
- **The touch allowance** (`_TOUCH_FRACTION`, 1 % of the circle's greatest
  depth below the ground) has no printed source either. Its value is
  unchanged (out of scope); its role is described under "Calls made".

## What was wrong (measured on master c5ed3c5)

All as the brief measured it. Taylor's circle, centre (13.02, 10.96),
through the toe: 0.8514 by hand; the module read it from x 6.04, under the
level ground in front of the toe: 2.676 on 30 slices, 2.682 on 120. Circles
leaving the face 2 % and 5 % of H above the toe: refused as two masses. The
default Bishop search: 0.9353, on a circle dipping 0.048 m under the floor
(inside the touch allowance). The toe-circle search shrank each radius by a
millionth; the 200-point sampling of `find_entry_exit` stepped over the gap.

## What changed, and why

1. **`slip_surface.py`.**
   - `circle_ground_meetings()`: where a circle meets the ground, from the
     ground's segments and corners (a quadratic on each segment), not by
     sampling. Between neighbouring meeting points the arc is wholly below
     the ground or wholly above it (its depth is concave along a segment),
     so a toe or any other corner divides the arc exactly.
   - `CircularSlipSurface` carries the two ends of the arc it stands for
     (`x_start`, `x_end`; both optional). `find_entry_exit` returns them,
     so the slices, supports, crack and checks are cut between them.
     `arcs(geom)` gives one circle per arc. A circle given without ends
     that bounds one arc is that arc, as before; one that bounds several
     raises `SeveralArcsError` from `find_entry_exit` only (it is not a
     refusal: every analysis below scores each arc).
   - `given_arc()`: ends given by a caller are set on the meeting points,
     or refused with the circle's arcs listed. `circle_of(result)`: the
     circle a result reports, with its arc's ends.
   - The two-mass refusal of a circle is gone. `GroundCrossingError` stays
     for polylines (see "Stopped short of").
   - A stretch that runs out of the profile below the ground is not
     between two meeting points and is no arc (before: the whole circle
     was refused where it also closed another stretch; now its closed
     arcs are surfaces). A circle's side with the ground above the
     centre's level ends an arc, as before (step 4's reverse-curvature
     rule applies to it unchanged).
2. **The searches** (`search.py`, `search_supplements.py`).
   - `_compute_fos`, every family's scorer: a split circle's score is the
     lowest of its arcs' scores; each arc goes through the same solve,
     counts and flags. `_check_entry_exit`: with windows, the arcs inside
     the windows are the ones scored.
   - The search remembers the arc that scored lowest on each split circle
     (`_governing_circle`, `_name_arc`), so the reported surface and every
     later analysis keep its two ends.
   - Toe circles and circles above the toe: `toe_arc()` takes, on the
     circle through each chord's two ends, the arc whose downslope end is
     the chord's start. The radius is the circle's own (no `1 - 1e-6`).
3. **`analysis.py` and the results.**
   - `lowest_arc()`: `analyze_slope`, `rapid_drawdown_fos` and
     `yield_acceleration` on a circle that bounds several arcs analyse each
     and return the lowest; FOSM and Monte Carlo take the arc with the
     lowest factor of safety at the values given and hold it.
   - `SlopeStabilityResult.arc` (also on the drawdown and yield results):
     the arc's two ends, its number among the circle's arcs, every arc of
     the circle, how it was chosen, each other arc with its value or the
     reason it has none, and the count of those by reason
     (`no_driving`, `not_converged`, `refused`). The first warning says the
     same in a sentence.
4. **The count.** `SearchResult.circles_split` = `{arcs: circles}`; in
   `to_dict()` and `summary()`. `n_rejected_ground_crossings` now counts
   polylines only. A `choices` entry ("circles that meet the ground at more
   than two points") appears when any circle was split.
5. **The agent's adapter** (`funhouse_agent/adapters/slope_stability.py`):
   `arc_x_start`, `arc_x_end` on every tool that takes a circle; `arc` and
   `circles_split` in the returns; the `radius` description says what
   happens to a circle with several arcs. The calc package's re-analysis
   and probabilistic annex keep the governing arc
   (`funhouse_agent/adapters/calc_package.py`). `docs/user_manual/manual.html`:
   the six parameter tables and the search's worked example, by hand. A
   probe: `slope_stability|analyze_slope#arc`.

## Places the rule now holds

| Caller | How |
|---|---|
| `analyze_slope` (so `compare_methods_table`, which calls it) | `lowest_arc` over the circle's arcs; ends given are checked (`given_arc`) |
| `search_critical_surface` and each search it runs: `grid_search`, `search_entry_exit`, `optimize_radius`, the refinement walk and its Bishop screen | `_compute_fos` / `_check_entry_exit`; the reported circle by `_governing_circle` |
| Toe circles, circles above the toe | `toe_arc` |
| The profile-end check (`search_supplements`, circles on the extended profile) | `_governing_circle` |
| The governing surface's supports, crack statement, support-arm warning, acceptability report, figure geometry | `circle_of(crit)` |
| `rapid_drawdown_fos`; `search_rapid_drawdown` (its per-surface solve receives each arc; its stage detail uses `circle_of`) | `lowest_arc`; `_compute_fos` with `fos_fn` |
| `yield_acceleration` (so Newmark, which takes its ky) | `lowest_arc` on ky |
| `fosm_fos`, `monte_carlo_fos` (so `reliability/wrappers.py`, which calls them) | `_resolve_surface`: the lowest arc at the values given, held |
| The calc packages (`slope_report_package` re-analysis and annex) | `circle_of(crit)` |
| Trial circles in the figures (`plotting._circle_arc`, `calc_package/interactive.py`) | every arc drawn |
| The agent's adapter | `arc_x_start` / `arc_x_end` build the circle with its ends |
| Planes, random, DE, PSO and weak-layer polylines | not circles: unchanged (see "Stopped short of") |
| `geo_project`'s limit-equilibrium builder | not edited (the brief forbids it): it re-analyses by xc, yc, radius, so `analyze_slope` / `fosm_fos` take the lowest arc again, the same arc unless the methods differ; noted |

## Calls made (for the owner; OWN-143)

1. **The ruling itself** (the orchestrator's): every arc is a trial
   surface. The other form: keep only the arc that holds the crest.
2. **A corner the arc only passes under does not split a circle.** The
   rule's words are "crossing it, or touching it within the touch
   allowance". Built first with a corner passed under by less than the
   allowance (at a local minimum of the depth) as a split point. Measured:
   an arc may then pass up to the allowance under a toe and be ended
   there, which leaves a small free face, so each search's minimum sat
   below its toe circles and moved with the constant, the defect the brief
   names. The vertical cut read 0.8440 (its toe circles 0.8514, Taylor
   0.8511); V-042's entry / exit search read 1.3269 for 1.3335 (printed
   1.334; Slide 1.340), a benchmark whose circles cross the ground twice;
   a windowed search (entry 0 to 4, toe at 5) lost its main-family minimum,
   1.3198 -> 1.3348, because the circle entering at 3.98 was split at the
   toe and neither part lay in the windows. As built: a circle is split
   where it crosses the ground or passes through a corner (zero depth to
   1e-9 R). A touch is (a) a stretch dipping under the ground by less than
   the allowance: no mass, no arc, as before; (b) a corner an arc passes
   under by less than the allowance: a point where an arc *given by its
   two ends* may end ("a touch within the allowance counts as a meeting
   point"), which is how Taylor's circle to two decimals is ended at the
   toe. A circle given without ends that passes under such a corner is one
   surface, as before this step, and its result names the corner and says
   how to end the arc there. The other form is one constant:
   `slip_surface._SPLIT_AT_TOUCHED_CORNERS = True`. Confirmed by the
   orchestrator after the review: an allowance must not change the answer
   ("Fix round after review", ruling 3).
3. **One circle, several arcs, no arc named: the lowest governs**, in
   `analyze_slope`, rapid drawdown (lowest drawdown factor of safety) and
   the yield coefficient (lowest ky). The other form: refuse and ask for
   the arc. The result always lists the others.
4. **FOSM and Monte Carlo on such a circle** use the arc with the lowest
   factor of safety at the values given, for every evaluation. The other
   form: re-choose the arc in each evaluation (then the mass changes
   between evaluations).
5. **The toe-circle families score the arc that ends at the chord's start
   only**, not the circle's other arcs (those are scored when the main
   search meets the same circle). The brief's item 2 has both sentences
   ("every family scores each arc"; "set the arc's ends at the toe");
   this is the reading where the family is "circles ending at the toe".
6. **With windows, only arcs inside the windows are scored**, and a split
   circle with no arc inside is counted in `circles_split`. Since the fix
   round the arcs left out are counted too (`arcs_outside_windows`, by
   family) and the statements say "each arc inside the windows".
7. **Ends given by a caller** must be within a thousandth of the radius of
   meeting points (a result's ends are often rounded) and are set on them.
8. **Where no arc has a value**, the result returned is the first arc's
   "no solution" result, with every arc listed.

## Section 11, before and after

Scripts, each run on master c5ed3c5 (a worktree) and on the branch:
`step2_section11.py`, `step3_section11.py` and `step4_section11.py`
unchanged, and `circle_ends_section11.py` (new, this step's rows).
"Part of the rule": **A** a circle split where it meets the ground more
than twice; **B** a toe circle ended at the toe (its own radius, no arc in
front of the toe); **C** the ends found from the ground's segments, exact,
where bisection stopped within 1e-6 m.

### Values that move

| Benchmark | Before | After | Printed | Part of the rule |
|---|---|---|---|---|
| B16 vertical cut, default search, Bishop / Ordinary Method of Slices | 0.9353 (main search; toe circles 1.0156, above the toe 0.9387; 402 circles refused as two masses) | **0.8514** (a toe circle; main 0.8523, above the toe 0.8527; 553 circles split) | Taylor Ns 3.83: 0.8511 (+0.03 %) | A and B |
| the same, `m_alpha_rule='every_slice'`, flag / refuse | 0.9353 / 0.9348 | 0.8514 / 0.8514 | 0.8511 | A and B |
| the same, Janbu | 1.1250 | 0.9137 (toe circle) | no printed Janbu value (Taylor's is a moment solution) | A and B |
| the same, Spencer / Morgenstern-Price | 0.9626 / 0.9626 | 0.8632 / 0.8632 | 0.8511 (+1.4 %); see check (c) | A |
| Taylor's circle (13.02, 10.96) through the toe, Bishop, 30 / 120 slices | 2.6755 / 2.6825 (from x 6.04) | 0.8514 / 0.8515 (arc 2 of 2, from the toe) | 0.8514 by hand | A |
| the same, R 13.00 (7 mm under the toe), ends given at the toe | no such input | 0.8501 / 0.8503 | 0.8514 (-0.13 %) | A (a touch as an end) |
| the same, R 13.00, no ends | 2.6705 / 2.6775 | 2.6705 / 2.6775, with the corner named | - | not moved |
| Circles leaving the face 2 % / 5 % of H above the toe | refused | 0.8690 / 0.8973 | the brief's own 0.869 / 0.896 | A |
| w10 circle (10, 39.5) R 40, Bishop / Spencer | refused | 7.3546 / 7.3871 (the slope's arc) | none (GM-122 measured Spencer 7.40 on that mass) | A |
| w10 circle (0, 50) R 50.92, Bishop | refused | 1.1496 (its arc on the face; the stretch at the profile's end runs out) | none | A |
| Windows given, entry 0 to 4 with the toe at 5 (`test_choices`) | 1.3164 ("toe circle" entering at x 3.99) | 1.3198 (main search; no toe circle in the window) | none | B |
| V-042 entry / exit search, Bishop | 1.3335 (main; toe circles 1.3383) | 1.3333 (toe circle; main 1.3334) | Hassan and Wolff 1.334, Slide 1.340 (-0.05 %; before -0.04 %) | B |
| ACADS 1(a) search, Bishop / Spencer / GLE | 0.9851 / 0.9840 / 0.9840 (toe circle) | 0.9849 / 0.9838 / 0.9838 (toe circle) | 1.00 (the referee's; Slide2 0.987): -0.2 % from Slide2 by Bishop, 0.02 % further than before | B |
| #85 one support, active, GLE search | 1.5732 (toe circle) | 1.5726 (toe circle) | Slide2 1.575 (-0.11 % -> -0.15 %) | B |
| #86 five tiebacks, GLE search | 1.6070 (toe circle) | 1.6069 | Slide2 1.622 (D&W 1.61) | B |
| V-043 force to raise the pinned sand circle to 1.35, Spencer | 40.9 kN/m | 45.2 kN/m | 44-45 kN/m (+0.5 to +2.8 %; before -7 %) | C: see below |
| Duncan Ex. 4, Janbu search, 30 slices (flagged `low_m_alpha` on both trees) | 0.8426 | 0.9303 | none for Janbu; the lowest unflagged surface, 1.2154, is unchanged | C: see below |

Non-governing candidates that move (the headline does not): #85 active
toe circle 1.5579 -> 1.5589 and above the toe 1.8190 -> 1.8195 (Bishop),
2.0649 -> 2.0520 (Spencer), 2.0774 -> 2.0562 (GLE); #85 passive toe circle
1.3285 -> 1.3286; V-043 sand toe circle 1.2561 -> 1.2592; B14 toe circle
5.3287 -> 5.3230 (all B). Counts that move with no value: #85 above the
toe "no driving" 92 -> 207 (a circle's arcs are each counted), V-042 main
"no driving" 4 -> 579 (the arcs under the level ground in front of the
toe), surfaces evaluated on #85 without its support 2150 -> 2104.

**V-043's force (C).** The geosynthetic (z = 3) crosses the pinned circle
at x 8.388, which is also a slice boundary (the same level is a layer
top), and the circle leaves the ground at x = 22 on that level. Which of
the two slices at the boundary takes the force is decided in the ninth
figure. On master the same circle with its radius 1e-8 m larger reads
Spencer 1.3353 at T = 40.9 where the pinned radius reads 1.3501, and the
force 45.2 for 40.9 (measured: `r (1 + 1e-9)`). This step finds the exit
at x = 22 exactly, where bisection gave 22.0000003, which lands on the
other side. The value moved towards print, so the test is re-pinned with
the printed answer in the comment; the tie itself is not this step's and
is noted for step 8 (supports).

**Duncan Ex. 4, Janbu (C).** The circle master reports ((9.779, 7.027), R
9.495) reads 0.8426 on the branch too, with the same two ends to seven
figures and no circle of this search is split. The search's walk ends on
another flagged circle because end points differ in the seventh figure.
Both headlines carry `low_m_alpha` (step 3 keeps and flags them); the
lowest unflagged surface is 1.2154 on both trees. No test pins the
flagged value.

### Values that do not move (check (b))

`step2_section11.py`: #85 without its support (Spencer 1.0468), #85 with
one support by Bishop 1.5080 and Spencer 1.5505, #85 passive by Bishop
1.3157 and GLE 1.3594, #86 by Bishop and Spencer, ACADS 1(a) by Janbu
0.9830, V-043 on its pinned circle 1.2096 and its critical circle 1.1858
and its default Spencer search 1.1859, V-031 on its pinned circle (1.4205 /
1.4247 / 1.4091) and its search 1.4174, V-044 on its pinned circle (1.5974
/ 1.5962 / 1.5933), V-039's yield coefficients (0.5649 / 0.5352 / 0.5590 /
0.4692), every FOSM and Monte Carlo row (B4, V-030, V-042, V-045) and
ACADS 4 (noncircular). `step3_section11.py`: #85 active and passive in
every m_alpha option (headlines 1.5080, 1.5109, 1.3157), V-032 (1.3149),
Duncan Ex. 1 in each D, B14 (0.8856). `circle_ends_section11.py`: the w10
default search 0.9379, #85's clay face 0.9095, V-054 as given 2.3464, the
search's old vertical-cut circle (18.48, 6.23) R 6.27 analysed alone
0.9368 (its dip under the floor was a touch before and is one now).

`step4_section11.py` (every pinned test of the validation set): master
385 passed, 1 skipped, 4 xfailed; the branch 386 passed, 1 skipped, 3
xfailed after the V-043 re-pin (as first run: 385 passed and that one
failed). The one mark removed is the vertical cut's. By file, all passing
with their pins untouched: validation_examples V-015/V-017 (13), V-026 (13),
V-026 E4 (5), V-028 (3), V-028 E8 (8), V-029 (6), V-030 (5), V-031 (10),
V-035 (3), V-037 (5), V-039 (3), V-040 (4), V-040 E6 (5), V-041 (7), V-042
(3), V-043 (7 of 8; the eighth is the force above), V-044 (2), V-045 (2),
V-048 (4), V-052 (4), V-054 (4); slope_stability tests
test_slope_stability (93), test_duncan_verification (33),
test_search_supplements (18), test_search_reinforcement (33),
test_circle_slicing (9), test_rapid_drawdown (16),
test_usability_features (49), test_m_alpha (18, two of them re-pinned for
the vertical cut).

Pins changed in tests: `test_m_alpha.py` (the vertical cut: 0.935 ->
0.851, printed 0.851; the xfail removed), `test_published_v043_slope_reinforced.py`
(40.9 -> 45.2, printed 44-45), `test_choices.py` (no toe circle in a
window that leaves the toe out), `test_ground_crossings.py` (four tests
that asserted the refusal of a circle now assert the split).

## The check

- **(a)** Default Bishop search on the vertical cut: **0.8514** against
  Taylor's 0.8511 (+0.03 %; the limit is 2 %). The strict xfail
  `test_vertical_cut_in_undrained_clay_against_taylor` passes and its mark
  is removed. The single surface ended at the toe: Taylor's circle through
  the toe **0.8515** on 120 slices (0.8514 on 30) against 0.8514 by hand
  (+0.01 %); to two decimals (R 13.00) with its ends given, 0.8503
  (-0.13 %; the limit is 0.5 %).
- **(b)** Benchmarks with two crossings: unchanged. See the table.
- **(c)** Spencer on the arc ended at the toe, solved on the circle (60
  slices): **no solution** ("no lambda from -1.5 to 1.5 gives an
  equilibrium state"), reported with step 2's reason and no number.
  Morgenstern-Price the same. So a Spencer or GLE search of such a cut
  leaves Taylor's surface unscored (it is counted under `no_solution`,
  `not_converged`): the Spencer and Morgenstern-Price searches read 0.8632
  on another circle from the toe (to x 25.37), 1.4 % above Taylor.

## Tests

All with `/tmp/v` (Python 3.12, `pip install -e packages/aec-agent-core -e .`),
judged by pytest's exit code.

| Run | Result |
|---|---|
| `slope_stability/tests` (all) | 784 passed, 17 skipped, 1 xfailed (the xfail is an older one, not this step's) |
| `slope_stability/tests/test_circle_ends.py` (new) | 14 passed |
| `validation_examples/test_published_*` that import slope_stability, with the main slope test files (`step4_section11.py`) | 386 passed, 1 skipped, 3 xfailed (after the V-043 re-pin) |
| `calc_package/tests`, `funhouse_agent/tests`, `reliability/tests`, `fem2d/tests`, `geotech_common/tests`, `report_ingest/tests`, `dxf_import/tests`, `geo_project/tests` files that name slope_stability (39 files), with `webapp/tests/test_public_text.py` | 2940 passed, 29 skipped, **1 failed** |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | 49 passed, 18 skipped |
| `webapp/tests/test_page_fingerprints.py` | passed with no re-record: the fingerprints did not change (no fingerprint commit on this branch) |
| `webapp/tests/test_public_text.py` | 5 passed, before each commit |

The one failure is
`test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
(`logx` and `logy` "changed nothing in the result"). It fails the same way
on master c5ed3c5 in this environment and touches nothing of this step.

Master moved during the build by notes only (BACKLOG.md and the brief:
GM-344); merged before the push, no code.

The new tests (`slope_stability/tests/test_circle_ends.py`, 14) cannot run
on master: the names they import do not exist there. The behaviours they
pin were measured on master with `circle_ends_section11.py` (the "before"
column above): Taylor's circle 2.68, the face circles and the four-point
circle refused, the default search 0.935.

## Release note

A circular slip surface now ends where the circle meets the ground. A
circle that crosses the ground more than twice is no longer refused: it is
split at those points and each part below the ground is analysed as a slip
surface of its own, with the lowest reported and named (`arc`). A toe
circle ends at the toe instead of running on under the level ground in
front of it. On a vertical or very steep face with ground in front of it
the factor of safety falls: a 5 m vertical cut in clay reads 0.851 instead
of 0.935 (Taylor 0.851). A search given entry and exit windows that leave
the toe out no longer reports a "toe circle" that entered in front of the
toe. Results on ordinary slopes are unchanged. A search's result counts the
circles it split (`circles_split`) and the arcs of those circles its
entry / exit windows left out, not scored (`arcs_outside_windows`);
`n_rejected.ground_crossings` now counts noncircular surfaces only. New inputs `arc_x_start` and `arc_x_end`
name one arc of a circle.

## Stopped short of, and noted, not fixed

- **Polylines.** A polyline that leaves the ground and comes back into it
  is still refused (`GroundCrossingError`). The brief's rule and its four
  items are about circles; splitting a polyline needs its own rule for
  where the parts end. Noted.
- **Print pages not re-read.** FHWA NHI-06-088 p. 6-38 and GEC 11 p. 9-6
  are cited from the account; no value is taken from them.
- **`geo_project/builders.py`** re-analyses the governing circle by xc, yc
  and radius (not edited: outside this section). It gets the lowest arc
  again, not the remembered ends.
- **`slope_stability_package`** (the calc package given one circle) takes
  no `arc_x_start` / `arc_x_end`: a circle with several arcs gets the
  lowest. Noted.
- **`docs/user_manual/worked_examples.json`** still holds the search
  example's old output (it is rebuilt at a release); only the manual's
  page was edited by hand. That stored output predates steps 2 to 4 too.
- **Spencer / GLE on steep phi = 0 faces** (check (c)): no solution on
  Taylor's arc. Which method scores a family is step 6's.
- **D25** is closed for circles by this step; for polylines it stands.
- **Janbu on the vertical cut** reads 0.914 (7 % above Taylor's moment
  solution; before 1.125). Not chased.
- **A support crossing on a slice boundary** is given to either neighbour
  slice, decided at 1e-9 (V-043's force, above). For step 8.
- **A flagged Janbu minimum** (Duncan Ex. 4, 30 slices) moves with the
  seventh figure of the end points (above). The unflagged minimum does
  not.
- **The Bishop screen's rescoring** (`grid_search`, `screened`) now applies
  the windows to a split circle's arcs; without windows nothing changes.
- **`test_defaults_used[profile_figure|plot_data]`** fails on master in
  this environment; not this step's.

## Fix round after review (2026-11-14)

The review (`CIRCLE_ENDS_REVIEW.md`: pass with defects) and the
orchestrator's three rulings. Branch `fix/gm140-circle-ends-review`.

### The merge

`origin/fix/gm140-step5a-review` (6dbd5d5: step 5a, already merged with
master 3d553a8 = steps 3 and 9) merged first (a4ecf4c); `origin/master` was
already inside it. Four files conflicted; each keeps both steps' rules:

- `rapid_drawdown.py`: the result keeps step 9's stages, `choices` and
  `notes` and this step's `arc`. The split comes after step 9's check of
  the layers for the 1970 procedure (so that refusal is not turned into
  "no arc can be analysed"), and each arc goes through the stages with
  step 9's inputs (`drawdown_phreatic_points`, `stage_solver`,
  `extra_boundaries`). Every stage is cut between the arc's two ends
  (`find_entry_exit` of the bound arc), and 5a's pond edge
  (`slice_breaks`) stays a slice boundary inside them.
  `search_rapid_drawdown` re-analyses `circle_of(crit)` (merged clean).
- The adapter's rapid-drawdown `returns`: step 9's keys plus `arc`.
- `VALIDATION.md`: B-5a, then B23.
- `manual.html`: rebuilt by `check_manual.py --rebuild --write` (the
  cloud builder's hand edit is gone); `check_manual.py` and
  `test_user_manual_current.py` pass. `geotech-references` at e1bbc5b.

The merge moved no value of this step: on every row of the table below
the branch reads what it read before the merge, and the drawdown rows
(two crossings) read what step 5a reads.

### Ruling 1: V-043's force

No code change. The pin stays 45.2 kN/m (printed 44 to 45). Its note, in
the test and in B23, now says it is one of the two outcomes of a tie: the
geosynthetic meets the slip surface on a slice boundary (x 8.388) and
changes slice within 1e-8 m of the arc's ends; the base read 40.88 kN/m at
R and 45.22 at R + 1e-8 m. The tie is step 8's (a stated rule for a
crossing on a slice boundary, or a cut at the crossing), then a re-pin;
the lead files the backlog item.

### Ruling 2: what the engineer is told about scoring

- `SearchResult.arcs_outside_windows` (`{family: arcs}`, beside
  `circles_split`, the families named as `no_solution` names them;
  in `to_dict` and the summary when not empty): the arcs of split circles
  that lie outside the entry / exit windows, given or the main family's
  automatic ones, and so are not scored. Counted per trial evaluation, as
  `circles_split` (`search._count_outside`, from `_check_entry_exit` and
  `_compute_fos`; the drawdown search names its families too).
- The statements now say "each arc inside the windows scored as a trial
  surface of its own": the summary (`results.py`), the search's choice
  (`search_supplements.search_choices`, with the count), the adapter's
  `circles_split` description, the docstrings.
- DESIGN.md "Where a circle ends": the toe-circle and circle-above-the-toe
  families score only the arc that ends at the toe or face point
  (`search_toe_circles.toe_arc`); the circle's other arcs are not scored
  by them and not counted. The choice's basis says the same. No new
  scoring.
- On the vertical cut the default Bishop search splits 553 circles and
  leaves 586 arcs outside the main family's automatic windows (some
  circles have both arcs outside); before this round those were counted
  nowhere.
- Test: `test_circle_ends.py::test_arcs_outside_the_windows_are_counted_not_scored`
  (the default search of the vertical cut; and one centre with windows
  given, where each split circle's lens is outside the entry window, is
  counted and is not scored: no `no_driving`). The one-centre test
  without windows now also asserts that nothing is counted.

### Ruling 3: a corner passed under within the touch allowance

`_SPLIT_AT_TOUCHED_CORNERS = False` stands, because an allowance must not
change the answer: split there, the vertical cut's default search reads
0.8440, the allowance acting as about 1 % more height (0.8511 x 5 / 5.05 =
0.843); not split, it reads 0.85135 whatever the allowance (0.005, 0.01,
0.02; the review's measurement). DESIGN.md says so; the warning of a
result that names the touched corner now says the circle is not split
there and why; "Calls made" (b) below lists both forms.

### The check on the merged tree

- Default Bishop search of Taylor's vertical cut: **0.85135**, against
  0.8511 (Taylor, Ns 3.83): +0.03 % (limit 2 %).
- The arc ended at the toe (centre (13.02, 10.96), R through the toe, to
  x 24.5664, 30 slices): **0.85135**, against 0.8514 by hand: -0.01 %
  (limit 0.5 %).

### Table (`review/circleix_table.py`; baseline = origin/fix/gm140-step5a-review's slope_stability)

| Case (the tests' own inputs) | Baseline (master with step 5a) | Branch | Change | Printed (page) | Cause |
|---|---|---|---|---|---|
| Vertical 5 m cut, cu 20, default Bishop search | 0.93529 (x 20.0 to 24.63) | 0.85135 (a toe circle, x 20.0 to 24.575) | -8.97 % | Taylor Ns 3.83: 0.8511 (FHWA NHI-06-088 Fig. 6-14, p. 6-27; the number from the spec) | the rule: split, and the toe circle ends at the toe |
| The same, Taylor's circle ended at the toe (ends given), 30 slices | no such input (a circle took no ends) | 0.85135 | - | 0.8514 by hand | the rule |
| Taylor 45 deg, D 2.25 (`test_taylor_undrained_nc`), default search | 1.9419 | 1.9419 | 0.00 % | 1.945 from the chart (FHWA NHI-06-088 Fig. 6-15, p. 6-28) | none (two crossings) |
| ACADS 1(a), default Bishop search | 0.98514 | 0.98493 | -0.02 % | 1.00 (Giam and Donald 1989); Slide2 #1 0.987 | a toe circle ends at the toe (its own radius, no millionth shrink) |
| ACADS 1(a), B3 entry / exit search 12 x 12, Bishop | 0.98514 | 0.98493 | -0.02 % | the same | the same |
| ACADS 1(b), 3.81 m crack full of water, Slide2's grid, Bishop | 1.58962 | 1.58961 | -0.00 % | 1.596 (Slide2 Verification #2, Table 2.2, p. 25) | ends found exactly (7th figure) |
| the same, Spencer | 1.58490 | 1.58490 | 0.00 % | 1.592 (the same) | none |
| ACADS 4, weak-layer search (V-028 E8's inputs), Spencer | 0.83007 | 0.83007 | 0.00 % | 0.760 (Slide2 #9, Table 9.4, p. 52) | none (polylines) |
| Slide2 #85 with its support, active, Bishop search | 1.50803 | 1.50803 | 0.00 % | D&W 1.51 (Duncan and Wright 2005, Fig. 6.34); Slide2 1.531 (Table 85.2) | none |
| the same, GLE search | 1.57321 | 1.57257 (toe circle) | -0.04 % | Slide2 GLE 1.575 (Table 85.2) | a toe circle ends at the toe |
| V-043 force to raise the sand circle to 1.35 (kN/m) | 40.881 | 45.223 | +10.6 % | 44 to 45 (Tandjiria et al. 2002; Slide2 verification) | the tie of ruling 1 (support crossing on a slice boundary) |
| Step 9, EM example by Bishop, the manual's 12 slices, stages 1 / 2 / 3 | 2.19249 / 1.51375 / 1.43844 | 2.19249 / 1.51375 / 1.43844 | 0.00 % | 2.20 / 1.52 / 1.44 (EM 1110-2-1902 pp. G-15, G-18, G-20) | none |
| #95 Corps 1970, force at 19.4 deg, 103 ft pool | 1.33793 | 1.33793 | 0.00 % | 1.35 (EM 1110-2-1902 p. G-15) | none |

Other pins moved, from `git diff origin/fix/gm140-step5a-review...HEAD` on
the test files and VALIDATION.md (not by running suites):

| Pin | Before | After | Printed | Cause |
|---|---|---|---|---|
| `test_m_alpha.py`, the vertical cut, 4 pins (the strict xfail removed; default, `every_slice` flag, `refuse`) | 0.935 | 0.851 | 0.851 (Taylor, Ns 3.83) | the rule |
| `test_published_v043_slope_reinforced.py`, the force | 40.9 | 45.2 | 44 to 45 | the tie (ruling 1) |
| `test_choices.py`, windows that leave the toe out (entry 0 to 4, toe at 5): the governing family | "circular, toe circle" (1.3164) | "circular" (1.3198, not pinned) | none | no toe circle lies in the window: a toe circle ends at the toe |
| `test_ground_crossings.py`, four tests that asserted a circle's refusal | refused | split; arc ends pinned (3.695, 16.305), (21.059, 37.014), (26.549, 31.509) | none | the rule |
| B23 (VALIDATION.md): the vertical cut by Janbu / Spencer and M-P | 1.1250 / 0.9626 | 0.9137 / 0.8632 | 0.8511 (a moment solution) | the rule; Janbu's gap is step 6's |
| B23: circles leaving the face 2 % / 5 % of H above the toe; the w10 circle by Bishop / Spencer | refused | 0.869 / 0.897; 7.355 / 7.387 | none | the rule |
| `test_circle_ends.py` (new file), `method_probes.json` (a new probe, `analyze_slope#arc`) | - | new pins only | as in the file | - |

### Calls made (confirmed after the review; for the owner, OWN-143)

- **(a) GM-122's ruling replaced** (the orchestrator's, 2026-11-13): each
  arc with ground above it is a trial surface of its own (built). The
  other form: keep only the arc that holds the crest.
- **(b) A circle passing under a corner within the touch allowance is not
  split there** (built; `_SPLIT_AT_TOUCHED_CORNERS = False`), so the
  answer does not depend on the allowance: 0.85135 on the vertical cut
  whatever the allowance. The other form: split there (0.8440 on the
  vertical cut; one constant, `True`).

### Tests run in this round (each file alone, by exit code)

Python `C:\dev\GeotechStaffEngineer\.venv` (3.14), PYTHONPATH the
worktree's references, the worktree and `packages/aec-agent-core`
(`slope_stability.__file__` printed from the worktree); one file at a time,
each started with at least 3 GB free.

| File | Exit | Result |
|---|---|---|
| slope_stability/tests/test_circle_ends.py | 0 | 15 passed |
| slope_stability/tests/test_m_alpha.py | 0 | 18 passed |
| slope_stability/tests/test_ground_crossings.py | 0 | 21 passed |
| slope_stability/tests/test_choices.py | 0 | 7 passed |
| slope_stability/tests/test_pore_pressure_as_printed.py | 0 | 50 passed |
| slope_stability/tests/test_rapid_drawdown.py | 0 | 15 passed |
| slope_stability/tests/test_rapid_drawdown_em_appendix_g.py | 0 | 23 passed |
| slope_stability/tests/test_search_reinforcement.py | 0 | 33 passed |
| funhouse_agent/tests/test_slope_stability_adapter.py | 0 | 94 passed |
| validation_examples/test_published_v043_slope_reinforced.py | 0 | 8 passed, 1 xfailed |
| validation_examples/test_published_v052_drawdown_pilarcitos.py | 0 | 6 passed |
| validation_examples/test_published_v037_slope.py | 0 | 5 passed, 1 xfailed |
| validation_examples/test_published_v041.py | 0 | 6 passed |
| validation_examples/test_published_v048_drawdown_methods.py | 0 | 4 passed |
| funhouse_agent/tests/test_adapter_ergonomics.py | 0 | 170 passed |
| funhouse_agent/tests/test_calc_package_adapter.py | 0 | 87 passed |
| geotech_common/tests/test_adapters.py | 0 | 101 passed |
| funhouse_agent/tests/test_eval_suite_v54.py | 0 | 10 passed |
| report_ingest/tests/test_recompute.py | 0 | 57 passed |
| funhouse_agent/deep/tests/test_wave1_geotech.py | 1 | 19 passed, 1 failed |
| webapp/tests/test_public_text.py | 0 | 5 passed |
| module_work/doc10/check_manual.py --rebuild --write, then check_manual.py | 0, 0 | OK |
| webapp/tests/test_user_manual_current.py | 0 | 1 passed |

The one failure, `test_the_rapid_drawdown_tail_parameter_is_visible`,
asserts that the deep agent's `describe_method` returns
`rapid_drawdown_fos`'s doc whole. It fails on the baseline too: with step
5a's `funhouse_agent` and `slope_stability` the doc is 16,569 characters
before fitting, over the 16,000 cap, and comes back shortened (measured
with the test's own call). The merge adds 1,368 characters (17,937), the
`arc` key among them. Not this round's change; for the lead.
