# GM-140 step 4: tension cracks and reverse curvature as printed

2026-11-13. Branch `claude/gm-140-step4-cracks-reverse-curvature`, from
master at 2cc2c45 (step 1 merged). Brief:
`module_work/cloud_briefs/SLOPE_GM140.md`, "Step 4". Departures D39 and D24
of `module_work/slope_integral_review/ACCOUNT.md`, GM-18's crack item, and
the owner's call OWN-103.

**Read this first.** The rule is built as the brief and OWN-103 write it,
and three printed benchmarks move towards print with it (ACADS 1(b),
V-043's clay crest, V-054). But closing reverse-curvature circles by the
Rankine crack moves the phi = 0 chart cases and Slide2 #85's Bishop search
AWAY from print (section 4). Those tests are strict xfails with the reason,
not re-pinned. The cause is one discontinuity in the call itself, described
in section 4, and it is the owner's to settle.

## 1. What the print says

USACE EM 1110-2-1902 (31 Oct 2003), from the account's quotations and the
brief's equations (the manual is the owner's copy and is in no repository;
no formula here is written from memory):

* 2-2f(3), p. 2-6: the soil upslope from the crack "is then ignored in the
  stability computations ... by terminating the slices near the crest of
  the slope with a slice having a vertical boundary"; water in the crack is
  a force on the boundary of the slice next to it.
* 2-2f(4), p. 2-6: the depth is the one producing the minimum factor of
  safety, where tensile but not compressive stresses are eliminated.
* 2-2f(5), pp. 2-6 to 2-9, total stresses: z_t = (2 c_D / gamma) tan(45 +
  phi_D / 2) (Eq. 2-4), c_D = c / F (Eq. 2-5), phi_D = arctan(tan(phi) / F)
  (Eq. 2-6).
* 2-2f(6), p. 2-9: for effective stresses the depth can also be estimated
  from Rankine's theory with c', phi' and the pore pressures; no equation
  is printed.
* C-10c, p. C-36: tensile stresses should be eliminated; the slip surface
  is terminated where it reaches the bottom of the crack; the depth is (1)
  the one of minimum factor of safety over a range of depths, or (2) the
  depth over which Rankine's active pressures are negative, "for total
  stresses and homogeneous soil", d_crack = 2 c_D / (gamma tan(45 - phi_D /
  2)) (Eq. C-36); where that is more than the height of an embankment on a
  weak foundation, the crack is as deep as the embankment and does not
  extend into the foundation.

Slide v6 Tutorial 16, p. 16-1: "A tension crack essentially terminates the
slip surface."

Slide2 Verification #2 (`module_work/slope_v53/slide2_selected_problems.txt`):
the crack depth is from Craig's Rankine expression with the full strengths
(2 c' / (gamma sqrt(Ka)) = 3.81 m), and the crack is full of water. The
test's inputs match it (c' 32 kPa, phi' 10 deg, gamma 20 kN/m3, the ACADS 1
slope); that depth is entered as the given depth.

## 2. What changed and why

New module `slope_stability/cracks.py` holds the rule; the other files are
changed only where they call it.

1. **Default model.** `tension_crack_model` defaults to `'truncation'`: the
   surface ends at the crack and the soil upslope is left out. `'strength'`
   (the cracked soil kept as weight without strength, the old default, no
   printed source) stays as a stated option. The mass that is kept is now
   cut into the `n_slices` asked for (`slices._kept_span`); before, the
   whole surface was cut and the cracked slices dropped, so a result said
   30 slices and had 28.
2. **Side.** `tension_crack_side` defaults to `'crest'`: the end of the
   surface where the ground is higher (`cracks.crest_side`). `'entry'` and
   `'exit'` stay.
3. **Depth.** New `tension_crack_depth_basis`: `'given'` (default),
   `'rankine'` (Eq. C-36 with Eqs. 2-5 and 2-6, F the surface's own by the
   method in use, the solution repeated until the depth changes by under
   1 mm) or `'min_fos'` (the scan). The Rankine depth stops at the bottom
   of the soil at the crack and says so.
4. **Reverse-curvature circles (OWN-103).** Closed by a crack of the
   printed depth (new `reverse_curvature_crack_basis`: `'rankine'` default,
   `'min_fos'`; a given crack, when there is one, is that crack). A circle
   whose overhang is deeper than its crack is discarded, counted and
   located (`SearchResult.n_discarded_reverse_curvature`,
   `reverse_curvature_discarded`); given to a single-surface method it
   raises `cracks.ReverseCurvatureDiscarded` with both depths.
   `reverse_curvature='discard'` discards every one. The crack from the
   centre's level to the ground is gone.
5. **Crack water** as before (0.5 gamma_w z_w^2 at z_w / 3), on the
   vertical boundary on the crack's side; a water depth above the crack's
   own fills it.
6. **Everywhere.** `build_slices` raises `cracks.CrackDepthNeeded` for a
   geometry whose crack still depends on the factor of safety, so a caller
   that was missed fails loudly instead of slicing on a silent basis.

The result carries the crack it was solved with
(`SlopeStabilityResult.tension_crack`, the depth used, one warning line); a
search's `choices` gain "tension crack", "tension crack side", "tension
crack depth" (when a crack is asked for) and "reverse-curvature circles".
The adapter has the two new inputs, the new defaults, new descriptions and
`applies_when` entries; the manual's matching rows and two prose lines are
hand-edited; three probes are added to `method_probes.json`.

### Places the rule now holds

| Caller | How |
|---|---|
| `analyze_slope`, `compare_methods` | `solve_with_crack` around the whole solve; every method (OMS, Bishop, Janbu, Spencer, Morgenstern-Price, GLE: `test_rankine_depth_is_at_the_surfaces_own_fos_phi_zero`) |
| `compare_methods_table` | through `analyze_slope` |
| `search._compute_fos` | the scorer of every family: grid circles, entry/exit circles, circles through and above the toe (`search_supplements`), planes, random, DE, PSO and weak-layer polylines; discards counted |
| the result builders of `grid_search`, `search_noncircular`, `search_pso`, `search_weak_layer_biased`, `search_entry_exit`, `search_de` | `search._report_geometry`: the reported surface is sliced with its own crack |
| `search_supplements` | `_make_result`, the support-arm and legacy-value warnings, the m_alpha confirmation (`cracks.geometry_at_fos`); the governing surface's crack statement (`_crack_statement`) |
| `search_critical_surface`, `search_rapid_drawdown` | discards counted and located; `choices` |
| `rapid_drawdown_fos` | one crack for every stage, at the drawdown F; the search's per-surface solve gets the fixed geometry from `_compute_fos` |
| `yield_acceleration` (so Newmark) | Rankine's depth at F = 1; a scanned depth from the surface without kh |
| `fosm_fos`, `monte_carlo_fos` | through `_compute_fos`: each run its own crack |
| `reliability/wrappers.py`, `geo_project`'s builder, the calc packages | through the above; none names a crack input |
| `funhouse_agent/adapters/slope_stability.py` | inputs, defaults, descriptions |

## 3. Worked example (made-up values)

A 10 m, 2H:1V slope in clay, cu 40 kPa, gamma 18 kN/m3, circle (22, 18),
R = 19, Bishop (`test_cracks_as_printed.py`).

| Crack | Before (default model) | After (default model) |
|---|---|---|
| none | 1.663 | 1.663 |
| 1 m given | 1.599 | 1.609 |
| 2 m given | 1.533 | 1.573 |
| 3 m given | 1.463 | 1.560 |
| 4 m given | 1.391 | 1.576 |
| 5 m given | 1.315 | 1.631 |
| Rankine, developed strengths | not offered | 1.560 at 2.85 m = 2 (40 / 1.560) / 18 |
| scan for minimum F | not offered | 1.560 at 3.10 m |

With the surface ended at the crack the factor of safety has a minimum in
depth, close to the Rankine depth with the developed strengths: the
manual's two ways agree. With the old model it falls without limit.

## 4. The validation set (account section 11), before and after

Method: `module_work/gm140/step4_section11.py` runs every
`validation_examples` file that uses slope_stability and the benchmark
files of `slope_stability/tests`. On master (2cc2c45, a detached worktree) 378
passed, 1 skipped, 2 xfailed (V-043's DE search, GM-117, and V-028's
weak-layer search): every pin holds. On the branch 377 passed, 1 skipped,
7 xfailed: three existing tests became strict xfails, two strict xfails
and two passing tests are new, and every other pin either holds unchanged
or is re-pinned as listed below. The "before" column was measured on that
worktree, not read from the tests' docstrings (two of which were stale:
the Taylor cases). A given surface with no crack and no
reverse curvature runs the same code as before
(`cracks.may_need_solution` is False), so B1, B2, B4, B7, B8, B10, B11,
B15, B18, V-030, V-035/036, V-039, V-044, V-045, V-048, V-052 read exactly
as before; the searches not listed (B3, B6, B13, B14, B17, V-031, V-032,
V-042, #86, the drawdown searches) pass their pins unchanged.

| Benchmark | Module before | Module after | Printed | Which part of the rule moved it |
|---|---|---|---|---|
| B9/B16 ACADS 1(b), Slide2 #2 (V-026), Bishop / Spencer / GLE, 3.81 m given crack, full of water | 1.485 / 1.486 / 1.482 | 1.603 / 1.599 / 1.600 | 1.596 / 1.592 / 1.592; referee 1.65 | the default model (+0.4 / +0.4 / +0.5 %; was -7 %) |
| same, `'strength'` option | the same | 1.485 / 1.486 / 1.482 | | not moved |
| same, Eq. C-36 developed strengths, full of water | not offered | 1.631 / 1.627 / 1.628 at 2.19 m | (the problem uses full strengths) | reported beside, not pinned |
| same, Eq. C-36, dry / scan, dry | not offered | 1.649 / 1.646 / 1.646 at 2.16 m; 1.649 at 2.41 m | | reported |
| V-043 clay embankment, water-filled crest crack (GM-18), Spencer, recovered circle | 1.041 (no crack entered; the depth is not printed) | 0.964 with Eq. C-36, 2.14 m, full of water | 0.975 | Rankine depth (-1.1 %; was +6.8 %). New test |
| V-031, Slide2 #57 (dry crack), Bishop / Spencer / GLE | 1.421 / 1.425 / 1.409 | the same | 1.417 / 1.422 | not moved: the test enters no crack (the depth is not in the recovered text) |
| B12 V-054 steep phi = 0 slope, Bishop search | 2.346 | 2.113 | 2.16 (FLAC 2.18) | reverse curvature: toe circle closed by Eq. C-36's 1.89 m crack (-2.2 %; was +8.6 %) |
| V-054 toe circle (2.5, 4.0) | 2.379 | 2.218 | 2.16 | the same (+2.7 %; was +10.1 %) |
| V-054 Bakklandet ratios on that circle | 1.942, -18.4 % | 1.693, -23.7 % | -24 % | the same, at the active strength |
| V-054 Bakklandet ratios on circle (3, 3) | 1.886, -20.7 % | 1.848, -16.7 % | -24 % | the same. **Away: strict xfail** |
| B16 vertical cut in clay, Bishop search | 0.842 | 0.752 | Taylor 0.851 | reverse curvature: crack 2.95 m where it was 2.30 m. **Away: strict xfail** |
| Taylor 45 deg, D = 2.25 (test_taylor_undrained_nc) | 1.942 | 1.835 | Taylor 1.947 | reverse curvature: a toe circle 0.01 m below the crest gets a 3.03 m crack. **Away: strict xfail** |
| Taylor 45 deg, short profile | 2.062 | 1.833 | Taylor 1.947 | the same. **Strict xfail** |
| B12/B16 #85, one support, active, Bishop search | 1.372 | 1.361 | 1.531 (D&W 1.51) | reverse curvature (below). **Away: strict xfail** |
| #85 passive, Bishop search | 1.254 | 1.206 | 1.324 (D&W 1.32) | the same |
| #85 active Spencer / GLE search | 1.551 / 1.573 | 1.551 / 1.573 | 1.884 / 1.575 | not moved |
| #85 passive Spencer / GLE search | 1.326 / 1.359 | 1.326 / 1.359 | 1.378 (GLE) | not moved |
| #85 toe family, Spencer | 1.564 | 1.627 | spread 1.51 to 1.884 | toe circles closed or discarded by the rule |
| Caltrans T&S Ex. 10-3 / 10-4 (worked example WE-SLOPE-1; V-015 itself builds its own slices and is not moved) | OMS 0.839, Bishop 0.977 | no number by default; the same with a 6.096 m crack given | 0.87; 0.96 | the printed circle's centre is 20 ft below the crest, deeper than Eq. C-36's crack (about 1.2 m): discarded unless a crack is given |

### Slide2 #85: the three readings (OWN-103 asked for the measurement)

| Reading | Active, Bishop | Passive, Bishop | Printed |
|---|---|---|---|
| Before: closed at the centre's level | 1.372 | 1.254 | 1.531 / 1.324 |
| OWN-103: closed by Eq. C-36's crack, deeper overhangs discarded | 1.361 | 1.206 | (D&W 1.51 / 1.32) |
| Every such circle discarded (`reverse_curvature='discard'`) | 1.508 | 1.316 | |

Under OWN-103 the old governing circle (centre 5.89 ft below the crest)
is discarded: Rankine's crack at its F is 5.21 ft. The search then governs
on a circle whose centre is 0.89 ft below the crest, closed by a 5.25 ft
crack. Ten circles are discarded in the active search, 23 in the passive.
How much of the support Bishop's method counts on such a circle is step
8's and is not touched.

### Why the chart cases move away: one discontinuity

A circle whose centre is a hair below the crest takes the whole Rankine
crack (about 3 m in the Taylor case); a circle whose centre is a hair above
it takes none. So on steep phi = 0 slopes the search settles on circles
that are barely reverse-curved, and reads lower than a search in which no
surface has a crack. Taylor's chart and Duncan and Wright's #85 have no
crack. Three consistent readings, measured:

| Case | OWN-103 (default) | Every such circle discarded | Rankine crack on every surface | Printed |
|---|---|---|---|---|
| Vertical 5 m cut in clay, Bishop | 0.752 | 0.935 | 0.605 | Taylor 0.851 |
| Taylor 45 deg, D = 2.25 | 1.835 | 1.942 | 1.811 | Taylor 1.947 |
| V-054, 45 deg from the toe | 2.113 | 2.346 | not measured | 2.16 (FLAC 2.18) |

For the owner: the call as written is built and is the default. If the
chart cases are to stay on print, "discard all" as the default does it for
Taylor (-0.3 %) and #85 (1.508 / 1.316 against 1.531 / 1.324), at the cost
of V-054 (+8.6 %) and the vertical cut (+9.9 %, the unsafe side).

## 5. Calls made

1. **Rankine's depth for an effective-stress soil** (the brief's reading of
   2-2f(6)): used where the soil at the crack is drained with no pore
   pressure inside the cracked depth (no water table there, no ru, no
   pressure grid); otherwise the scan, and the result says why.
2. **Soils without one c and phi.** SHANSEP and Hoek-Brown soils are
   scanned. An `'anisotropic'` soil takes its active strength with phi = 0
   in Eq. C-36 (Rankine's active pressures); no print for anisotropy. This
   keeps the model's isotropic identity (V-054).
3. **The scan.** 21 depths in equal steps from 0 (or the overhang's depth)
   to the lowest point of the surface's upper half, shallowest first, the
   first minimum taken; no minimum by the last depth means no number (a
   search leaves the surface out; a reverse-curvature circle is discarded).
   Without the "first minimum" rule a deep, degenerate depth won on a
   submerged slope (Duncan's Ex. 6 read 0.19). The step count is not from
   print.
4. **A reverse-curvature circle with the scan** is discarded where F rises
   from the overhang's own depth down (the depth of minimum F is then not
   below the overhang).
5. **Reverse curvature at the lower end** of a surface is discarded: a
   crack is at the upper end (2-2f(3)) and no print closes the other.
6. **A given crack on a reverse-curvature circle** is the crack that
   closes it; shallower than the overhang, the circle is discarded.
7. **Ends at one level** (no crest): the crack is at the low-x end, as
   before.
8. **The yield acceleration** takes Rankine's depth at F = 1, the state it
   solves for; a scanned depth comes from the surface without kh.
9. **Rapid drawdown** takes one crack for every stage, at the drawdown
   factor of safety (the stages are step 9's; only the entry is wrapped).
10. **A discarded circle given to a single-surface method raises** an
    error with both depths, where before it was evaluated either way: "no
    number" instead of a number on a surface the rule does not admit.
11. **The Rankine iteration's tolerance** is 1 mm of depth, 25 solutions
    at most; not from print.

## 6. Tests run (pytest's exit code)

| Run | Result |
|---|---|
| the 62 consumer files naming slope_stability (calc_package, funhouse_agent, reliability, fem2d, geotech_common, report_ingest, dxf_import, geo_project, every `validation_examples/test_published_*`) with `slope_stability/tests` and `webapp/tests/test_public_text.py` | 3691 passed, 45 skipped, 7 xfailed, 1 failed (the plot_data probe below, not this step's) |
| `slope_stability/tests` alone | 702 passed, 17 skipped, 4 xfailed before the last additions; inside the run above after them |
| `module_work/doc10/check_manual.py` (the manual's name check, no rebuild) | OK |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | passed (47 passed, 17 skipped with the two guard files beside it); the three new probes run |
| `webapp/tests/test_page_fingerprints.py` | 13 passed with no re-record: the pages' fingerprints do not hold the slope tool's parameter text, so there is no fingerprint commit |
| `module_work/gm140/step4_section11.py` on master / on the branch | 378 passed, 1 skipped, 2 xfailed / 377 passed, 1 skipped, 7 xfailed |

Strict xfails added, each with the printed answer and the reason: the
vertical cut; Taylor 45 deg (two tests); #85's Bishop search against
1.531; the Bakklandet reduction on circle (3, 3).

One failure is not this step's: `test_defaults_used.py::
test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
fails the same way on master (2cc2c45) in this environment (`logx`, `logy`).

## 7. Release note

Slope stability: a tension crack now ends the slip surface at the crack by
default, as EM 1110-2-1902 and Slide print it, and sits at the end of the
surface where the ground is higher. A crack given without a model reads
higher than before (ACADS 1(b): 1.485 to 1.603 against the printed 1.596);
`tension_crack_model='strength'` gives the earlier value, and a crack
given with `tension_crack_side='entry'` on a slope whose crest is at high
x is still at the low-x end. The depth can now be Rankine's with the
developed strengths (EM Eq. C-36) or the depth of minimum factor of
safety. A circle whose centre is below the ground at its upper end is
closed by a Rankine crack, where it was closed at its centre's level, and
is left out where it overhangs deeper than that crack: searches on steep
clay slopes read differently (lower on the cases measured: a vertical cut
0.842 to 0.752, Slide2 #85 with its support 1.372 to 1.361), and such a
circle given on its own may now get an error instead of a number; give a
tension crack depth to close it. The result names the crack it used and
counts the circles a search left out.

## 8. Noted, not fixed

* The crack's bottom is the ground at the surface's end less the depth, as
  before; on a sloping crest that is not the depth at the crack itself.
* The Rankine check for water uses the geometry's own water table; in a
  rapid drawdown the stages' pools are not looked at.
* A noncircular search has no reverse curvature, so the rule does not
  count anything there; a polyline that turns back is already refused.
* The worked example WE-SLOPE-1's stored `computed_result` (0.8328 /
  0.9709) is stale against master (0.8385 / 0.9769, measured at 2cc2c45)
  before this step; not touched. The example now gives its printed circle
  a 6.096 m crack so that it still runs.
* V-031 (#57) and the V-043 sand case enter no crack: the recovered
  problem text gives no depth.
* D36 (how much of a support Bishop counts on a circle crossing near the
  centre's level) is step 8's; #85's Bishop values depend on it.
* The supports crossing a surface are found on the whole circle, not on
  the part a crack leaves (step 8).
* The m_alpha rule is unchanged (step 3): which surfaces a search refuses
  for any other reason is as before.

## 9. Stopped short of

Nothing in the section. The manual's rows are hand-edited, not rebuilt.
`BACKLOG.md`, `HANDOFF.md` and `CLAUDE.md` are not edited.

## 10. Fix round after review

2026-11-13. Branch `fix/gm140-step4-review`, cut from the step's branch at
5d63dec, with master merged first (decec33, step 2; merge 5c65f6c). It acts
on an independent review of the step (outside the repository) and on the
orchestrator's rulings on its two calls (F1, D1). Sections 2 to 9 above are
the record as first built; where they disagree with this section, this one
holds.

**The merge with step 2.** Both rules kept: a trial with no solution of the
method is no number and is counted as such (step 2), and the crack rule
(step 4); the counts stay apart by reason. Step 2's
`probabilistic._eval_fos` sliced its surface directly, which raises
`CrackDepthNeeded` under a Rankine or scanned crack: it now solves through
`cracks.solve_with_crack`, each run with its own crack.

**F1 (the orchestrator's ruling, option 4; refines OWN-103).** The tension
crack is a condition of the analysis, set by the engineer, and the same on
every trial surface; a circle's shape never switches it on. As first built,
a reverse-curvature circle was closed by Rankine's crack whenever no crack
was given (`reverse_curvature_crack_basis`), so the factor of safety jumped
as a circle's centre crossed the crest's level, and the searches on steep
phi = 0 slopes settled on barely reverse-curved circles (section 4's
"discontinuity").

* No crack asked for: a reverse-curvature circle is not an admissible
  surface. It is discarded, counted under `reverse_curvature_no_crack` and
  located; the search's `choices` say how many and that naming a crack
  (given depth, Rankine, scan) admits those within its depth. No crack
  appears in such a result.
* A crack asked for: every surface ends at the crack's bottom; a
  reverse-curvature circle whose overhang lies within the crack's depth is
  closed by that crack, nothing deeper; one overhanging deeper is discarded
  (`reverse_curvature_deeper_than_crack`, OWN-103's limit).
* `reverse_curvature='discard'` stays; the centre-level crack stays removed;
  `reverse_curvature_crack_basis` is gone (geometry, adapter, cracks).
* Files: `cracks.py` (`plan_crack`, the refusal reasons and their words,
  `crack_choices`), `geometry.py`, the adapter's `reverse_curvature`
  description.
* Continuity, on the review's made-up clay (ground (0, 0), (10, 0), (14, 8),
  (40, 8); cu 30 kPa, gamma 18 kN/m3), two circles through the toe with
  their centres 2 cm above and below the crest's level, Bishop: a given 3 m
  crack 0.9489 / 0.9491; Rankine's 0.9328 / 0.9327 (3.57 m); the scan
  0.9272 / 0.9269; no crack: the upper one 1.1312 with no crack, the lower
  one refused (`reverse_curvature_no_crack`). The default search there
  discards 27 circles and reads 1.0625 with no crack in the result.
  (`test_cracks_as_printed.py`.)
* The benchmarks with no crack read as before step 4 (VALIDATION.md B21):
  Taylor 45 deg 1.942 and the short profile 2.065 (the two strict xfails
  removed); V-054's toe search 2.346; #85 toe family 1.564; Slide2 #85 with
  its support, Bishop, 1.508 with 28 circles discarded (printed 1.531, D&W
  1.51) and passive 1.316 with 41 discarded (printed 1.324, D&W 1.32),
  reported and not pinned (the step's brief sets no tolerance on #85; the
  strict xfail against 1.531 removed). With Rankine's crack asked for on
  every surface #85 reads 1.331 / 1.188. Those that used the centre-level
  crack do not come back: the vertical cut reads 0.935 against Taylor's
  0.851 (+9.9 %, the unsafe side; before step 4 0.842 on a circle closed at
  its centre's level; 0.605 with Rankine's crack asked for), a strict xfail
  with the reason; V-054's toe circle (2.5, 4) and circle (3, 3) are no
  surfaces without a crack, so those tests ask for Rankine's crack and read
  as first built (2.218; Bakklandet 1.693 and 1.848, the latter still a
  strict xfail against 24 %).
* The 5.89 ft crack that four tests gave only to keep #85's old 1.372
  (`test_search_reinforcement.py` three times, `test_calc_steps_modern.py`
  once) is gone. Where a test needs the toe circle for something else
  (Spencer's lack of solution, the calc package's no-solution sections, the
  short-arm warning) it gives a 6 ft crack, deeper than that circle's
  5.89 ft overhang, with the reason stated (Bishop 1.372; 29 % of the
  support counts); the default search and the single circle with no crack
  assert the discard.

**D1 (agreed by the orchestrator).** A surface that nowhere passes below
the crack's depth, measured below the ground above it, lies wholly within
the cracked depth and is no slip surface (EM 2-2f(3), p. 2-6; C-10c,
p. C-36: the slices end at a vertical boundary at the crack's bottom, which
such a surface never reaches). The rule is geometry only: no threshold on F
or on the mass's size. `cracks.depth_below_ground` gives the surface's
greatest depth below the ground between its ends (exact at the ground's
vertices and the circle's lowest point); a given crack at least that deep
refuses the surface (`within_cracked_depth`); a Rankine iteration that
reaches it starts again from F with no crack and refuses the surface if it
reaches it again; the scan stops short of it. A search counts the refusals;
`analyze_slope` raises `cracks.CrackRuleRefused`.

* **It predates step 4.** On master before step 4 (the tree at decec33),
  ACADS 1(b) searched with the given 3.81 m water-filled crack on Slide2's
  centre grid (x 31 to 47, z 34 to 49, 16 by 16, 40 slices, no entry or
  exit windows), Bishop, reads 0.0503 with master's default
  (`'strength'`: circle (37.267, 34.938), R 5.652, from x 39.44 to 40.14)
  and 0.0520 with the surface ended at the crack (circle (35.133, 40.375),
  R 11.461, x 39.95 to 40.56): slivers on the face. As first built, 0.052.
* After: Bishop 1.5896 (circle (37.560, 43.165), R 19.676, through the toe)
  and Spencer 1.5849 (circle (37.533, 42.875), R 19.399), against the
  printed 1.596 / 1.592 (Slide2 #2, Table 2.2, p. 25): -0.4 % each, inside
  the step's 1 %; 456 surfaces refused as wholly within the cracked depth.
  (`validation_examples/test_published_v026_slope.py`; the Spencer search is
  marked slow, 4 minutes.) A unit test refuses a 2 m crack on a circle
  shallower than 2 m in both crack models and keeps a crack half as deep.

**D3.** The crack is reported and drawn where it is: where the surface
reaches the crack's bottom and the slices end (`tension_crack['x_m']`,
with `ground_z_m` and `base_z_m` there, and the warning's "at x = ..."),
not at the surface's ground end. ACADS 1(b)'s toe circle: x 53.169 (the
surface's end 55.461). Files: `cracks.py` (`_crack_x`, `to_dict`,
`statement`), `plotting.py`. Tests: V-026, `test_cracks_as_printed.py`.

**D4.** V-031 (Slide2 #57) has a dry crack in the problem (p. 203) and the
text prints no depth: the test now enters Rankine's with the developed
strengths (Eq. C-36, p. C-36), the module's printed basis, dry, 50 slices,
and reads Bishop 1.4170 / Spencer 1.4211 against the printed 1.417 /
1.422 (Table 57.3, p. 205), pinned to the print at the file's tolerance
(rel 0.005). The test without a crack keeps its values, labelled the
module's own.

**D5.** A search counts the crack rule's refusals by reason
(`SearchResult.crack_refused`, in `to_dict` too: for each reason its count,
its words, where the surfaces end and the first few):
`reverse_curvature_no_crack`, `reverse_curvature_deeper_than_crack`,
`reverse_curvature_no_crack_at_end`, `reverse_curvature_discard_asked`,
`within_cracked_depth`, `scan_no_minimum`. `n_discarded_reverse_curvature`
sums the four reverse-curvature reasons; the search's `choices` give the
reverse-curvature count and, apart, "surfaces the crack rule refused" for
the others; the adapter's returns text names `crack_refused`. The scorer
catches `cracks.CrackRuleRefused` (whose subclass is
`ReverseCurvatureDiscarded`) and counts it under its reason, where before
every refusal counted as a reverse-curvature discard. Files: `cracks.py`,
`search.py`, `results.py`, the adapter; `analysis.py`, `rapid_drawdown.py`
and `newmark.py` raise through `cracks.refusal_error`.

**Pins changed** (every one a printed value with its page, or labelled the
module's own):

| File | Pin | Before this round | Now |
|---|---|---|---|
| `validation_examples/test_published_v026_slope.py` | grid search with the 3.81 m wet crack, Bishop / Spencer (new) | none | the printed 1.596 / 1.592 at rel 0.01 (Table 2.2, p. 25) |
| same | the crack's place (new) | none | x 53.169, the slices' end (module's own geometry) |
| `validation_examples/test_published_v031_slope.py` | no crack, Bishop / Spencer / GLE 1.421 / 1.425 / 1.409 | beside the printed values | labelled the module's own |
| same | with its dry crack, Rankine's (new) | none | the printed 1.417 / 1.422 at rel 0.005 (Table 57.3, p. 205) |
| `validation_examples/test_published_v043_slope_reinforced.py` | clay with the Rankine crack, Spencer 0.964 | unlabelled | labelled the module's own; printed 0.975 with its table and pages |
| `validation_examples/test_published_v054_anisotropic_su.py` | toe circle (2.5, 4) 2.218 | Rankine on reverse circles by default | Rankine's crack asked for, labelled the module's own; refused with none |
| same | Bakklandet 1.693; circle (3, 3) strict xfail | the same default | the same values, Rankine's crack asked for |
| `slope_stability/tests/test_search_supplements.py` | anchor (2.5, 4) 2.218 | by default | Rankine's crack asked for (module's own); refused with none |
| same | toe search 2.113, exit 7.6; mirrored 2.113 | first built | 2.346, exit 9.3 (the pre-step-4 pins); discards counted |
| `slope_stability/tests/test_slope_stability.py` | Taylor 1.947 rel 0.01; short profile > 1.947 | strict xfails | passing, xfails removed |
| `slope_stability/tests/test_m_alpha.py` | vertical cut 0.842 | strict xfail | strict xfail against Taylor's 0.851 rel 0.02; new test pins 0.935 (module's own); the phi = 0 m_alpha warning on circle (21, 2.7), R 3.3, with a given 2.3 m crack |
| `slope_stability/tests/test_search_reinforcement.py` | #85 Bishop search against 1.531 | strict xfail | removed: reported in this file, not pinned; the default asserted equal to 'discard' |
| same | first-built 1.361 / Rankine checks | pinned | replaced by the discard assertions (no crack) |
| same | toe family Spencer 1.627 | first built | 1.564 (module's own, the pre-step-4 pin) |
| same | 1.372 on the toe circle (two tests) with a 5.89 ft crack | pinned | a 6 ft crack, F > 1 (the tests are about Spencer's lack of solution and the short arm) |
| `slope_stability/tests/test_calc_steps_modern.py` | the toe circle's 5.89 crack | 5.89 | 6.0, reason stated |
| `funhouse_agent/tests/test_slope_stability_adapter.py` | `reverse_curvature_crack_basis` default `'rankine'` | asserted | asserted absent; `reverse_curvature` default `'tension_crack'` |
| `funhouse_agent/tests/method_probes.json` | the reverse-curvature probe | no crack | Rankine's crack asked for |

**Noted, not fixed.** A surface that passes below the crack's depth
somewhere, but whose crest half stays above the crack's bottom, still gets
the crack at mid-surface as first built (`slices.py`, the stub at the
upper half's lowest point); D1 refuses only the surfaces wholly within the
cracked depth. With Rankine's crack asked for, V-054's toe search reads
2.087 (every surface now has the crack; first built 2.113 with it on the
reverse-curvature circles only); not a test.

**Tests run** (pytest's exit code, one file at a time):

| File | Exit | Count |
|---|---|---|
| `slope_stability/tests/test_cracks_as_printed.py` | 0 | 40 passed |
| `slope_stability/tests/test_search_reinforcement.py` | 0 | 33 passed |
| `slope_stability/tests/test_calc_steps_modern.py` | 0 | 23 passed |
| `slope_stability/tests/test_slope_stability.py` | 0 | 92 passed |
| `slope_stability/tests/test_search_supplements.py` | 0 | 18 passed |
| `slope_stability/tests/test_m_alpha.py` | 0 | 17 passed, 1 xfailed |
| `slope_stability/tests/test_circle_slicing.py` | 0 | 9 passed |
| `slope_stability/tests/test_usability_features.py` | 0 | 49 passed |
| `slope_stability/tests/test_own_solution_or_no_number.py` | 0 | 26 passed, 1 xfailed |
| `validation_examples/test_published_v026_slope.py` | 0 | 13 passed (the Spencer search, 4 minutes, in the first run; 12 rerun after the last code change) |
| `validation_examples/test_published_v031_slope.py` | 0 | passed |
| `validation_examples/test_published_v043_slope_reinforced.py` | 0 | 8 passed, 1 xfailed |
| `validation_examples/test_published_v054_anisotropic_su.py` | 0 | 4 passed, 1 xfailed |
| `funhouse_agent/tests/test_slope_stability_adapter.py` | 0 | 92 passed |
| `funhouse_agent/tests/test_worked_examples.py` | 0 | 38 passed, 2 skipped |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | 0 | 47 passed, 17 skipped |
| `funhouse_agent/tests/test_allowed_values.py -k slope` | 0 | 20 passed |
| `funhouse_agent/tests/test_top_level_params.py -k slope` | 0 | 3 passed |
| `module_work/doc10/check_manual.py --rebuild --write`, then the check | 0 | OK (manual.html rebuilt: the slope rows and the parameter counts) |

**Release note** (replaces section 7's): Slope stability: a tension crack
now ends the slip surface at the crack by default, as EM 1110-2-1902 and
Slide print it, and sits at the end of the surface where the ground is
higher; the result reports and the plot draws it where the surface reaches
its bottom. A crack given without a model reads higher than before (ACADS
1(b): 1.485 to 1.603 against the printed 1.596);
`tension_crack_model='strength'` gives the earlier value. The depth can now
be Rankine's with the developed strengths (EM Eq. C-36) or the depth of
minimum factor of safety. A circle whose centre is below the ground at its
upper end (reverse curvature) is no longer closed by a crack from its
centre's level: with no crack asked for it is left out of a search and gets
no number on its own; with a crack asked for it is closed by that crack
where its overhang lies within the crack's depth. Searches on steep clay
slopes can read higher than before where they settled on such a circle (a
vertical cut 0.842 to 0.935); give a tension crack to admit those circles.
A surface that lies wholly within a given crack's depth is no longer a slip
surface (ACADS 1(b) searched with its crack: 0.05 on a sliver before, 1.590
now against 1.596). A search counts the surfaces the crack rule refused, by
reason (`crack_refused`).
