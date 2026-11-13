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
