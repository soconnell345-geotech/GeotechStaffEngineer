# slope_stability — Limit Equilibrium Slope Stability

## Purpose
Circular and noncircular slip surface analysis by the methods of slices
(Fellenius / OMS, Bishop's simplified, Janbu, and the rigorous Spencer,
Morgenstern-Price and GLE in `gle.py`) with critical surface search,
entry/exit range constraints, and nails, anchors, geosynthetics and
stabilizing piles in every method (`reinforcement.py`).

## References
- Fellenius (1927) — Ordinary Method of Slices
- Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
- Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
- Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
- Wright & Duncan (1991) — Limit equilibrium stability analyses for reinforced
  slopes, Transportation Research Record 1330, 40-46 (support forces in each method)
- Zhu, Lee, Qian & Chen (2005) — A concise algorithm for computing the factor of
  safety using the Morgenstern-Price method, Can. Geotech. J. 42, 272-278
- Rocscience Slide2 help — Active/Passive Force Application; Grid Search
  (reverse curvature)
- FHWA GEC-7, Lazarte et al. (2003) — Soil nail walls (nails.py)

## Files
- `geometry.py` — SlopeSoilLayer (elevation-based), SlopeGeometry (polyline surface/GWT)
- `slip_surface.py` — CircularSlipSurface (center + radius), PolylineSlipSurface (polyline points); a surface below the ground over more than one stretch is refused (`GroundCrossingError`, GM-122)
- `slices.py` — Slice dataclass, build_slices() with multi-layer weight/pore pressure
- `methods.py` — fellenius_fos(), bishop_fos(), spencer_fos() / morgenstern_price_fos() (wrappers of the rigorous `gle.gle_fos`; the `*_legacy` approximate forms are the fallback), solve_rigorous()
- `gle.py` — the rigorous general limit equilibrium (Fredlund & Krahn 1977; supports by the slice-by-slice march of Zhu et al. 2005, GM-60) and janbu_fos()
- `reinforcement.py` — nails, anchors, geosynthetics and stabilizing piles as forces at the base of the slice each crosses, in every method (GM-53, GM-59)
- `search.py` — optimize_radius(), grid_search() (with the local refinement of its lowest centres, GM-87), search_noncircular(), entry/exit filtering; a rigorous noncircular solution must hold about a second axis and slicing (`_solution_fails`, GM-57 review)
- `search_supplements.py` — toe-circle, circle-passing-above-the-toe and infinite-slope (shallow planar) families scored after every `search_critical_surface` (GM-13, GM-14, GM-122)
- `search_ranges.py` — the automatic entry / exit windows and centre ranges, set from the slope's toe, crest, height and depth to the base (GM-57)
- `analysis.py` — analyze_slope(), search_critical_surface() orchestrators
- `results.py` — SliceData, SlopeStabilityResult, SearchResult
- `nails.py` — SoilNail, NailContribution (nails reach the FOS through `reinforcement.py`)
- `tests/` — 598 tests in 29 files (collected 2026-10-17), among them:
- `tests/test_ground_crossings.py` — 21 tests (one sliding mass per trial surface, the touch tolerance at its threshold, the refusals counted, the circles passing above the toe, GM-122; entry / exit windows given the wrong way round, GM-124)
- `tests/test_slope_stability.py` — 92 tests (core + entry/exit limits + noncircular search; Taylor's phi = 0 chart, GM-57)
- `tests/test_search_ranges.py` — 23 tests (the automatic ranges; a directed search and what the result says it covered; the same slope drawn narrow and wide, GM-57)
- `tests/test_duncan_verification.py` — 28 tests (Duncan §7.6-7.7 verification examples; Ex 1 against Taylor's chart, GM-63, and the default search on long profiles, GM-87)
- `tests/test_nails.py` — 38 tests + 16 skipped (nail intersection/pullout/capacity; the skipped ones test the old nail path, replaced by `reinforcement.py`)
- `tests/test_reinforcement.py`, `tests/test_search_reinforcement.py` — 22 + 33 tests (supports in each method; supports in searches, GM-53, GM-54, GM-59, GM-60)

## Public API
```python
# Circular analysis
analyze_slope(geom, xc, yc, radius, method="bishop", tol=1e-4, ...) -> SlopeStabilityResult

# Noncircular analysis (polyline slip surface)
analyze_slope(geom, slip_surface=PolylineSlipSurface(points), method="spencer", tol=1e-4, ...)

# Critical surface search — circular (grid search)
search_critical_surface(geom, x_range, y_range, nx, ny, method="bishop",
                        surface_type="circular", x_entry_range=None, x_exit_range=None,
                        tol=1e-4, ...) -> SearchResult

# Critical surface search — noncircular (random polyline)
search_critical_surface(geom, surface_type="noncircular",
                        x_entry_range=None, x_exit_range=None,
                        n_trials=500, n_points=5, seed=None,
                        tol=1e-4, ...) -> SearchResult
```

## Slip Surface Types

### Circular (CircularSlipSurface)
- Defined by center (xc, yc) and radius
- Entry/exit found by bisection against ground surface
- Works with all three methods (Fellenius, Bishop, Spencer)
- Spencer reduces to Bishop for circular (theta=0)

### Noncircular (PolylineSlipSurface)
- Defined by polyline points [(x1,z1), (x2,z2), ...]
- x-coordinates must be monotonically increasing
- Entry = first point, exit = last point (or intersection with ground)
- Spencer, Morgenstern-Price / GLE and Janbu (`gle.py`); Bishop asked for on a
  noncircular surface is analysed with Spencer, and the result says so (GM-81)
- `slip_elevation_at(x)` — linear interpolation between points
- `tangent_angle_at(x)` — angle of segment containing x
- Both classes share `is_circular` property for dispatch

## Search Algorithms

### Circular Grid Search
- `grid_search(geom, x_range, y_range, nx, ny, method, ...)` — evaluates nx*ny circle centers
- `optimize_radius(geom, xc, yc, method, ...)` — golden-section radius optimization at each center
  (each step reuses the FOS of the interior radius it keeps, since 2026-10-15: same result,
  about a quarter fewer solves)
- Optional `x_entry_range` / `x_exit_range` — reject surfaces whose entry/exit fall outside bounds
- FOS guard: surfaces with FOS < 0.05 are rejected as numerical artifacts
- `search_critical_surface` and `search_rapid_drawdown` then refine the lowest grid centres
  (`grid_search(refine=True)`, GM-87; next section). `grid_search` called directly does not.

### Local refinement of the centre grid (GM-87, 2026-10-15)
A grid of centres finds the critical circle only to within its spacing, and the
automatic grid spans the whole profile, so on a long profile it is coarse.
Duncan Ex 1 (2:1 slope, su/(gamma H) = 0.120) drawn with 4 H of level ground
each side has its 10 x 10 centres 1.1 H apart in x; Taylor's critical circle
(FHWA NHI-06-088 Fig. 6-15, after Taylor 1948) has its centre 1.0 H in front of
the crest, between two of them, and the grid read 0.739 against 0.705 at D = 2
(0.755 with 5 H each side). The radius was never the problem: at every centre
`optimize_radius` already found the circle tangent to the firm base.

- **What.** After the grid, its local minima (no lower point among the eight
  around them; at most three, those within 10 % of the lowest grid FOS) are each
  refined by a compass (pattern) search on the centre (`search._refine_centre`):
  the four centres half a grid step away in x and in y are scored, each with its
  own optimised radius exactly as on the grid; the walk moves to any lower one and
  halves the steps when none is lower, down to a sixteenth of the grid spacing
  (at most 40 centres a start). It stays inside the grid's own centre ranges, so
  caller-given ranges are honoured and the grid-edge warning still means what it
  says.
- **Why a refinement, not a family of circles tangent to the base.** Slide2 and
  others offer circles tangent to each layer boundary ("tangent" surfaces) as a way
  to choose radii. Here the radius was already tangent to the firm base at every
  centre (the base at -1.00 H in every run); what was missing was the centre, and a
  tangent family still needs a centre grid. A local search on the centre is
  general: it serves any slope, not only one with a firm base.
- **Rigorous methods.** Spencer / Morgenstern-Price / GLE walk with Bishop (the same
  critical circles to about 1 % and some 30 times cheaper, as the toe-circle family
  does) and re-optimise the radius at each walk's end with the method itself; the
  method's own FOS decides. Measured against a walk with Spencer itself on nine
  models: the same FOS to four decimals on eight, at +2 to +23 % run time instead of
  +40 to +225 %. Not where supports are modelled: Bishop counts a support only
  through its moment about the centre (GM-59) and the two methods' critical circles
  part (Slide2 #85, anchored: the Bishop walk found nothing below Spencer's grid
  minimum 1.581, Spencer's own walk 1.574), so a reinforced model walks with the
  method itself.
- **On the base.** `build_slices` drops up to max(2, 3 % of n_slices) below-base END
  slices as tangential slivers (SS-5). Where the toe sits on a rigid base (Slide2 #85
  without its support) a circle whose lowest point is a few millimetres under the
  base just beyond the toe sheds its two toe slices and scores low: 0.867 on a circle
  refused at 90 slices, against the base-tangent toe circle's 0.883. The walk found
  such circles, so it keeps to circles that at most touch the base
  (`search._dips_below_base`, the same test as `build_slices` with no tolerance), as
  the toe-circle family does. The grid itself is unchanged.
- **Rapid drawdown** (`search_rapid_drawdown`, its own per-surface solve): the lowest
  grid centre only, three step halvings, at most 12 centres.
- **Off switch.** `search_critical_surface(refine=False)` /
  `search_rapid_drawdown(refine=False)` give the grid alone.
- **Taylor on long profiles** (Bishop, 30 slices, default search): D = 2, 4 H and 5 H
  each side: 0.739 / 0.755 -> 0.703 / 0.703 (Taylor 0.706); D = 3: 4 H / 5 H / 6 H:
  0.690 / 0.696 / 0.703 -> 0.676 (Taylor 0.679); the centre 1.00 H in front of the
  crest, the circle tangent to the firm base, no warning. Spencer gives the same.
- **What moved** (every move is down; none of the published benchmarks' tests needed
  a change): a coarse 4 x 4 re-search in the Monte Carlo test 1.372 -> 1.253 (the
  default grid and a 30 x 30 grid give 1.252 / 1.254); Duncan Ex 4 sand over clay
  1.319 -> 1.249 (both circles cut by the left end of the profile, the GM-63 warning
  on both); a 3 x 3 grid on the calc-package slope 1.471 -> 1.454 (30 x 30: 1.453;
  the refined circle exits 1.9 m from the automatic exit window's start, so that
  edge warning now fires); the window-limited toe-geometry circle 2.784 -> 2.770;
  ACADS 1(a) default Bishop 0.990 -> 0.988 on the grid (the toe circle 0.985 still
  governs). Rapid drawdown: #98 Corps 0.8370 -> 0.8341, DWW 0.9352 -> 0.9349, LK
  0.9632 -> 0.9619 (published 0.931 / 1.039 / 1.075; ours were already about 10 % low
  on the approximate recovered geometry, and move up to 0.3 % further); Pilarcitos
  true section Corps 0.7258 -> 0.7236 (published 0.823), and on the 2:1 / 2.5:1 / 3:1
  sections LK and DWW 0.4-0.6 % lower, toward the published ~1.05.
- **Runtime** (one process, shared laptop, noisy): default circular Bishop +0.2 to
  +0.4 s (ACADS 1(a) 0.93 -> 1.16 s, Duncan Ex 1 4 H 1.0 -> 1.3 s); default Spencer
  ACADS 1(a) 14.0 -> 13.0 s and Duncan Ex 1 4 H 9.5 -> 7.6 s (the golden-section
  reuse in `optimize_radius` more than pays for the walk); rapid-drawdown circular
  searches +4 to +6 s (#98 Corps 8.5 -> 12.2 s, Pilarcitos 18.5 -> 19.4 s).

### Automatic search ranges (GM-57, 2026-10-16)
Ranges the caller does not give are set from the slope, not from the width of
the model (`search_ranges.py`; `search_critical_surface` and
`search_rapid_drawdown`). Before, they were fractions of the profile (entry in
the first 40 %, exit in the last 40 %, centres over the whole profile from
1 unit above the highest ground to 2 H above it), so the answer depended on how
the model was drawn: an off-centre slope had its critical surface cut off by a
window, and a model drawn wide put the split between the windows out on the
level ground (only large circles admissible) and spread the grid thin.

- **The slope.** The faces of the ground (`slope_faces`) grouped by the way
  they rise; each group's lowest toe, highest crest and height H. A group less
  than a third as tall as the tallest is left out; an embankment, a dam or a
  cut gives two groups and their windows are joined.
- **Depth to the base** d, from the toe to the bottom of the deepest layer:
  Taylor's firm base (FHWA NHI-06-088 Fig. 6-15, after Taylor 1948; depth
  factor D = 1 + d/H) and Janbu's d = D/H (Fig. 6-16). The model base, not a
  firm layer above it: a firm layer only shortens the circle, so the base
  gives the wider range.
- **Centre heights** from 0.1 H to max(2 H, d) above the highest ground over
  the centre range. Janbu's charts (FHWA NHI-06-088 Figs. 6-16 and 6-19) put
  the critical centre 1.35-3 H above the toe for depth factors up to 3;
  midpoint circles rise with d.
- **Reach**, the deepest circle the grid holds (tangent to the base from a
  centre at the grid's top, Y above the toe): sqrt(d (2Y + d)) beyond the toe
  (a 2 % of H slack when the base is at the toe: nothing comes out beyond it)
  and sqrt((Y + d)^2 - (Y - H)^2) behind the crest. Taylor's critical circles,
  from independent phi = 0 optimisations, stay inside with room to spare:

  | Case (Taylor) | Out beyond the toe | In behind the crest |
  |---|---|---|
  | 2:1, D = 2 | 1.19 H (reach 2.65 H) | 1.76 H (3.46 H) |
  | 2:1, D = 3 | 2.69 H (4.0 H) | 3.2 H (4.58 H) |
  | 1:1, D = 2.25 | 1.96 H (3.01 H) | 2.45 H (3.75 H) |

- **Windows.** The end on a group's toe side lies from the toe's reach to the
  face's mid-length; the end on the crest side from the mid-length to the
  crest's reach. The centre grid spans the windows (a circle's centre lies
  between its ends), so its spacing follows the slope. Caller-given windows
  widen an automatic centre range to cover them. A split nearer the toe and
  the crest (the middle third) was tried and rejected: it hid the circle
  above the geosynthetic in reinforced ACADS 1(a) (1.105 became 1.267, the
  unsafe side).
- **No least length.** A tiny circle straddling the mid-length of a c' = 0
  face is admitted, and it IS the answer there: shallower slides tend to the
  infinite slope, tan(phi')/tan(beta) (Slide2 #86 without supports: a circle
  0.4 ft long at 0.9420, the closed form 0.9419, against 0.9443 for the
  planar surfaces with Janbu's f0). It governs, and the result says it is the
  infinite-slope mechanism (the planar-tie note, with no window-edge
  warnings, as shallower slides only tend to the same value). A least length
  of H/2 was tried in review and dropped: it was a rule the search added,
  and it also hid a 4.6 m circle in a 4 m weak layer over rock.
- **Not a judgment about the ground.** The automatic ranges serve the slope as
  a whole. A failure confined to part of the face -- a weak layer above the
  toe, a slide above the water outside the slope (FHWA NHI-06-088 p. 6-38) --
  is the engineer's or the agent's call to search for, by giving the windows;
  the search does not detect such layers itself (the owner's ruling,
  2026-10-16). On a 10 m 1:1 cut with 4 m of weak soil over rock the default
  search reads 4.82 (5.25 before GM-57) where the circle in the weak layer
  gives 0.885 (a 20 x 20 search with both windows over the whole profile);
  directed (entry 25.5-28, exit 29-40) it finds 0.895 on the default grid,
  0.889 on 20 x 20, 0.888 by entry / exit.
- **The model's ends.** A range the end of the profile cuts is recorded
  (`SearchResult.search_ranges["clipped"]`: where the slope called for it);
  when the governing surface ends at such a cut and the GM-63 check finds no
  lower circle on level ground past the end, a warning says the model is
  shorter than the slope's reach.
- **Reported, so the coverage can be judged.** `SearchResult.search_ranges`:
  the ranges used, which were automatic, the slopes they came from with toe,
  crest, H, d and both reaches, and the cuts. `SearchResult.governing_location`:
  the governing surface's ends, lowest point, greatest depth below the
  ground (what tells a sliver from a slide), the layers along its base in
  order, and its ends and depth against the slope's toe and crest in
  multiples of H. `SearchResult.refused_lowest` (below): the lowest surface
  the search refused. All in `to_dict()` and `summary()`. Level ground (no
  face) keeps the old fractions.
- **What moved** (VALIDATION.md B14): every published benchmark still passes;
  the critical FOS fell wherever a window had held the search off a surface
  (soft foundation drawn off-centre 0.782 -> 0.676, Taylor 0.679; Slide2 #85
  discard 1.503 -> 1.485, a circle that converges to D&W's 1.51 where the old
  toe circle converges to 1.557), and on c' = 0 faces the circle search now
  reaches the infinite slope itself (Duncan Ex 2 / Ex 4 1.6816 / 2.3474 ->
  1.6782 / 2.3442, the closed forms 1.678 / 2.344). Rose: an entry/exit
  search's 10 x 10 points over the wider windows (V-042 1.325 -> 1.332, toward
  the published 1.334 / 1.340) and random noncircular searches with few
  trials, whose draws now fall on the face too (seed-dependent both ways).
  The Pilarcitos rapid-drawdown tests give the deep slide's windows: the
  automatic ones find the rapid-drawdown infinite slope, a sliver of a
  circle about an inch deep on the face above the remaining pool at
  (gamma'/gamma) tan(phi')/tan(beta) = 1.075 (Lowe-Karafiath / DWW, true
  section), reported as that mechanism and pinned in its own test;
  whether the published values belong to a deep slide cannot be checked from
  the reference library.
- **Runtime**: unchanged for models drawn about 1-2 H each side; a wide model's
  grid is now denser over the slope, at the same number of centres.

### Rigorous noncircular solutions must hold (GM-57 review, GM-116)
A noncircular surface scored by Spencer / Morgenstern-Price / GLE is kept as a
search's new best only if its solution holds about a second moment axis (two of
gle's least-squares axis, the pinned axis and the pinned axis raised by the
span agree within 2 %) and with one slice more or twice as many (a converged
solution within 50 %): `search._solution_fails`, run through
`_compute_fos(confirm_below=best so far)` by the random, weak-layer, PSO and
DE searches. A refused surface is not only counted
(`n_rejected_nonconverged`): the lowest one is kept in
`SearchResult.refused_lowest` (FOS, where, why: "solved about one moment axis
only" / "at 30 slices only, not at 31 or 60"), and when it is below the
reported FOS, or nothing held, a warning says so ("The lowest surface found,
FOS X (x = a to b), did not hold: ..."), so the user can judge what the search
covered. Spencer and GLE satisfy force and moment equilibrium, so their
solution does not depend on the moment axis; a state that "converges" about
one axis or at one slice count is a solver artefact, and the searches had been
reporting such states as critical:

| Surface | Reported | What held |
|---|---:|---|
| 10 m 1:1 cut, 3.5 m weak clay over sandstone, random search: a surface 13 m deep in the sandstone | 0.90 (HEAD) / 1.06 | about gle's least-squares axis only; analyze_slope's Spencer: no solution; Fellenius 21.7 |
| V-043 sand fill, DE seed 1 | 0.717 (HEAD) | about the pinned axis only, at lambda -0.35; 29 / 31 slices give 1.90 / 1.55 at lambda +0.2 |
| 4 m weak layer over rock, directed DE: a surface diving at 75 deg into the rock | 0.731 | about two axes at 30 slices; none at 29, 31 or 60 (Janbu 0.42, Fellenius 8.5) |

The check runs only when a surface would become the best (a few times a
search). A surface that fails is counted with the non-converged rejections.

### Noncircular Random Search
- `search_noncircular(geom, n_trials, n_points, x_entry_range, x_exit_range, ...)` — random polylines
- Generates random polyline surfaces between entry and exit x-ranges
- Intermediate points at random depths below ground surface
- Uses Spencer's method for each trial
- Returns best (minimum FOS) surface found

### Supplementary families: toe circles and the infinite slope (GM-13, GM-14)
`search_critical_surface` scores two more surface families after the main
search, with the same method, and reports the lowest surface
(`search_supplements.apply_supplements`). Both close cases where the search
reported a factor of safety that was too HIGH.

- **Why.** Until GM-57 the automatic windows came from the MODEL extent,
  not the slope: entry in the first 40 % of the width, exit in the last
  40 %, circle centres from 1 unit above the crest. A slope that does not
  sit in the middle of its model was under-sampled. Steep phi = 0 slope, H = 5 m at
  45 deg, su = 40, model starting at the toe (V-054): the critical toe
  circle exits the crest at x = 7.1, inside the automatic exit window's
  lower bound (15), and its centre (2.5, 4.0) is below the crest, under the
  grid. Grid 2.78; true 2.287 (with the ranges set from the slope, GM-57,
  the grid reads 2.344: the toe circle's centre is still below it). And
  with c' = 0 the critical failure is not a circle at all but the shallow
  planar slide parallel to the face, which no circle of finite depth
  reaches: Slide2 #86 without its supports, grid 1.57 (0.974 since GM-57)
  against tan 37 / tan 38.66 = 0.942. Where a circle wins (the main search
  can now reach the shallow slide), lies on the cohesionless face, is
  shallow (its deepest point within a twentieth of its length) and comes
  within 2 % of what the infinite slope gives there -- the planar surfaces'
  FOS, tan(phi')/tan(beta) for a dry face, or (gamma'/gamma) tan(phi') /
  tan(beta) for a face saturated to the surface with seepage parallel to it
  (Duncan, Wright & Brandon 2014, Ch. 14; `infinite_slope_fos`), the
  saturated face a rapid drawdown leaves, where the planar family's own
  drawdown solve can fail -- the result says it is the infinite-slope
  mechanism and gives no window-edge warnings; a deep circle scoring about
  the same is not called that. `infinite_slope_checks` carry both closed
  forms (`FOS_dry_closed_form`, `FOS_seepage_closed_form`).
- **Toe circles** (Taylor 1937; Duncan, Wright & Brandon 2014): for each
  slope face (`slope_faces`: maximal runs of segments steeper than 5 deg
  rising the same way; a flat stretch ends a face, so each bench has its own
  toe; the three tallest faces), circles through the toe and a second ground
  point upslope. The second point is scanned at 0.15-4 times the face size,
  densest near the toe; the centre along the chord's perpendicular bisector
  from a near-semicircle to a flat arc; both refined by golden section.
  Centres must lie above the ground at their own x. Spencer / M-P / GLE scan
  with Bishop and re-optimize the best three circles with the method itself,
  so the reported FOS is the method's own (Bishop and the rigorous methods
  pick the same circles to about 1 %; this keeps the Spencer overhead near
  zero). Circular and entry_exit searches only.
- **Infinite slope (shallow planar, translational)** (Duncan, Wright &
  Brandon 2014, Ch. 14; Skempton & DeLory 1957): wherever a drained
  Mohr-Coulomb layer with c' = 0 and phi' > 0 is the material at the surface
  of a face steeper than 5 deg (`cohesionless_face_runs`), planar polylines
  parallel to the face (over the whole outcrop, each half and each third)
  are scored by `_compute_fos`. Each is a tenth of a slice width deep, with
  end ramps 0.15 of a slice width long, so the ramps end before the first
  and last slices' midpoints and EVERY slice base is parallel to the face
  (deeper surfaces whose ramps span a slice midpoint were mis-sliced: a
  slice on a 45 deg ramp swung the FOS 5 % either side of the closed form).
  They are REAL surfaces, so pore pressure, ponding, surcharge and kh act
  on them as on any other. They are scored by **Janbu's simplified method
  whatever method the search uses**: the infinite slope is a
  force-equilibrium result (the interslice forces on a slice of an infinite
  slope are equal and opposite), Janbu's force equilibrium with no
  interslice shear reproduces it, and moment equilibrium says nothing on a
  thin planar slide -- the rigorous GLE often finds no F_m = F_f crossing
  there (the surface is rejected) and Spencer at 1 % depth ran 6 % above
  the closed form on a 2:1 sand face. Measured against the closed form
  F = tan phi' (cos b - kh sin b - ru / cos b) / (sin b + kh cos b), dry,
  with ru 0.2, with kh 0.1-0.15, both face directions: +0.1 % to +0.4 %
  (the Janbu f0 factor; 30 slices +0.2 %). The closed-form dry value is
  reported beside each outcrop. Not evaluated where a tension crack is
  modelled: the crack model strips the strength of the crest half of any
  surface, which on a shallow slide is an artifact; the closed form is
  still reported and compared. All search types.
- **What the result says.** `SearchResult.governing_surface`
  ('circular' / 'noncircular' / 'toe_circle' / 'infinite_slope'),
  `candidates` (each family's minimum), `infinite_slope_checks` (one entry
  per cohesionless outcrop: x range, beta, layer, phi', closed form, FOS or
  the reason it was not evaluated), and `warnings`: a family undercut the
  main search; the governing surface ends on the edge of a window or the
  centre grid (a bound on the model edge is not a choice and is not
  flagged); a cohesionless face where the dry infinite slope is below the
  reported FOS but no planar surface could be scored (windows, crack). All
  in `to_dict()` and `summary()`.
- **Constraints.** Caller-given `x_entry_range` / `x_exit_range` /
  `x_range` / `y_range` are honoured by both families (a planar slide's
  left end must lie in the entry window and its right end in the exit
  window). The automatic windows are a heuristic and do not restrict them.
  `toe_circles=False` / `infinite_slope=False` switch either family off.
- **Runtime** (ACADS 1(a), one process, best of two, measured twice on a
  shared laptop): default circular Bishop 0.83 -> 0.94 s and 0.89 -> 1.00 s
  (+12-13 %), default circular Spencer 20.9 -> 20.7 s and 20.9 -> 22.7 s
  (0 to +8 %), entry_exit 12x12 Bishop +8-11 %, noncircular Spencer 300
  trials unchanged (no cohesionless face). The toe family adds
  about 450 Bishop evaluations per face (plus about 30 with the rigorous
  method when it is Spencer / M-P / GLE); the planar family six Janbu
  evaluations per outcrop.
- **Toe circles on a rigid base.** A toe circle may at most TOUCH the
  bottom of the deepest layer. `build_slices` tolerates a below-base
  sliver or two (SS-5); a toe sitting on a rigid base would let every toe
  circle through it shed its toe slices that way and score low (Slide2 #85
  without support: 0.83, against 0.88 for the base-tangent toe circle).
- **Rapid drawdown (GM-54).** `search_rapid_drawdown` runs both families
  with its own per-surface drawdown FOS (`fos_fn`), honouring the caller's
  ranges, and its `search` carries `governing_surface` / `candidates` /
  `warnings`. With a per-surface solve this costly there is no Bishop
  screen; the toe scan is coarser instead (7 centre offsets, 6 golden steps
  on the far end). The thin planar surfaces converge in the drawdown GLE: on
  a free-draining c' = 0 shell drawn down from el 20 to 5 the planar FOS is
  0.942 = tan 37 / 0.8, the closed form.
- **Not covered.** Planar sliding ON a weak layer (not parallel to the
  face) is the noncircular / weak-layer searches' job. SHANSEP clay with
  su_min = 0 also has a depth-free infinite-slope FOS but is not treated
  as cohesionless here.

### Reinforcement in every search surface (GM-53)
Until 2026-10-14 `search._compute_fos` scored every trial surface WITHOUT
its reinforcement (only `analyze_slope` called
`compute_reinforcement_forces`), so every search of a reinforced slope found
the unreinforced critical surface and FOS (ACADS 1(a) with one 200 kN/m
geosynthetic: search 0.990, `analyze_slope` on that circle 1.713).
`_compute_fos` now resolves the crossings on each trial surface exactly as
`analyze_slope` does (`search.trial_reinforcement`: `compute_reinforcement_
forces` between the surface's entry and exit) and passes them to every method,
circular and noncircular, so every search kind -- grid, entry/exit, random
and DE noncircular, PSO, weak-layer, the toe-circle and infinite-slope
families, and the fixed-surface FOSM / Monte Carlo through `_compute_fos` --
scores the reinforced surface. `rapid_drawdown_fos` applies the crossings at
every stage solve, so `search_rapid_drawdown` does too. The final critical
result lists the crossings (`reinforcements`, as `analyze_slope` gives them).
Unreinforced geometry: nothing is resolved and every solve is unchanged.

- **A reinforced surface whose rigorous solve fails is rejected.**
  `spencer_fos` / `morgenstern_price_fos` fall back to the legacy solver when
  the GLE does not converge, and that solver IGNORES reinforcement (it warns).
  In a search that let a heavily anchored surface score unreinforced and win
  (Slide2 #85, Spencer: 0.86 against Bishop 1.26 on the same circle), so for a
  reinforced circle `_compute_fos` requires the GLE to converge
  (`_reinforced_rigorous_fos`) and otherwise rejects the surface. Single-surface
  `analyze_slope` now reports no FOS there (GM-81, below).
- **Measured** (Slide2 verification manual; Duncan & Wright 2005):
  - #86 (five 800 lb/ft tiebacks, "global failure" windows entry 0-4 ft, exit
    30-75 ft): Bishop 1.612 / Spencer 1.606 / GLE 1.606 against Slide2 1.629 /
    1.620 / 1.622 and D&W 1.61. Before: the unreinforced 0.942 (infinite
    slope) or the deep circle.
  - #85 (one 9,000 lb/ft support at mid-height, active): see "Support forces
    in each method" below (GM-59, GM-60) for the current values and why the
    Bishop search gives 1.304 by default.
  - V-043 (Tandjiria geosynthetic at the fill base): the published force
    (44-56 kN/m to raise the UNREINFORCED critical surface to 1.35) is a
    single-surface quantity; a reinforced search cannot reach 1.35 with any
    basal force, because the shallow sand-fill surfaces above the
    reinforcement (infinite slope 1.259) govern.
- **Runtime**: the crossing is resolved on every trial surface of a
  reinforced slope -- Bishop searches +28 % to +41 %, Spencer +11 % (ACADS
  1(a) + geosynthetic; #86 with tiebacks). Unreinforced searches unchanged.

### Support forces in each method; active and passive (GM-59, 2026-10-14)
**Where a support force acts.** Wright & Duncan (1991, TRR 1330 -- read in
full) represent each reinforcement "as concentrated forces on the base of
single slices" and include it in every equilibrium equation the procedure
satisfies; Slide2 likewise applies "a force ... at the point of intersection
of the slip surface with the support". So each nail / anchor / geosynthetic /
pile force acts at the base of the slice it crosses:

| Method | Moment about the centre | Vertical equilibrium of that slice | Horizontal equilibrium |
|--------|------------------------|------------------------------------|------------------------|
| Bishop's simplified | yes (signed) | yes: W -> W - Fz in the numerator, so the base normal and, where phi > 0, the strength change | not satisfied by the method |
| Ordinary Method of Slices | yes (signed) | the component normal to the base adds to N | not satisfied |
| Spencer / M-P / GLE / Janbu (`gle.py`) | exact | yes (base-normal equation) | yes (E march) |

Wright & Duncan's conclusion 6: "In Bishop's simplified procedure, which
satisfies vertical force equilibrium, the vertical components of the
reinforcement forces should be included in the equations of vertical
equilibrium" (they trace Low & Duncan's different results to leaving them
out). Until GM-59 Bishop and the OMS counted only the moment, and its
magnitude: an inclined support whose line of action passes on the far side
of the centre counted as resisting. Now the moment is signed (the horizontal
component taken as resisting, into the slope, as the GLE takes it) and the
vertical / normal component enters the slice. On a circle only the force's
component along the slip surface has a moment about the centre (the normal
component passes through it), so a support crossing a steep part of the
circle near the centre's level counts for little in Bishop's method -- that
is the statics, and every method that satisfies moment equilibrium gives the
same F on a phi = 0 circle (F_m does not depend on the interslice forces
there).

**Active or passive** (`support_convention` on `SoilNail`, `Anchor`,
`Geosynthetic`, `StabilizingPile`; Duncan & Wright 2005 Ch. 8 Methods A/B;
Slide2 "Active/Passive Force Application"):
- `'active'` (**default**, Method A): F = soil resisting / (driving -
  support). "The support force is NOT divided by the factor of safety ...
  Only the soil strength is divided by the factor of safety. Hence the
  support capacity input ... should be the ALLOWABLE support force" (Slide2).
  "According to Duncan and Wright (2005), Method A is preferable, because the
  soil strength and the reinforcement forces have different sources of
  uncertainty, factoring them separately makes it possible to reflect these
  differences." Tensioned tiebacks act on the mass before it moves.
- `'passive'` (Method B): F = (soil resisting + support) / driving; "both
  soil strength and support forces are divided by the factor of safety", so
  the input is the ULTIMATE force. Here the force acts as T / F in every
  equation above (its moment, its vertical and horizontal components).
  Passive gives the lower F for the same force.
Every method honours the choice (before GM-59 only Fellenius / Bishop, and
only for piles; the GLE applied every support unfactored).

**Reverse-curvature circles** (`SlopeGeometry.reverse_curvature`). A circle
whose centre lies below the ground where it comes out rises past the
centre's level and would turn back over itself. Slide2 calls these "reverse
curvature" surfaces, which "cannot actually exist", and by default closes
them with "a vertical tension crack where the slip surface begins to reverse
direction (i.e. when the slip surface y-coordinate is equal to the slip
center y-coordinate)", always dry; otherwise it discards them. The module has
always done the former (`find_entry_exit` ends the circle where the arc is
vertical, at the centre's level). Now: `'tension_crack'` (default, as
Slide2 and as before) or `'discard'` (the searches drop such circles; a
circle given to `analyze_slope` is still evaluated). A search whose
governing circle is one says so in `warnings`, with the crack depth.

**Bishop / OMS critical circle with supports at a short arm.** When the
supports crossing a Bishop or OMS critical circle act with an average arm
under half the radius (under half their force counted through the moment),
the search warns, gives the share and what Spencer finds on the same circle
(`search_supplements._support_arm_warning`).

**Slide2 #85** (D&W 2005 Fig 6.34; c = 350 psf, phi = 0, 98 pcf, 20 ft face
at 2V:1H on a rigid base, one 9,000 lb/ft support at mid-height):

| | Module | Slide2 (circular) | D&W |
|---|---:|---:|---:|
| Active, on Slide2's printed GLE circle (15.446, 37.624, R 27.594): Bishop = Spencer = GLE | 1.564 | GLE 1.575 | 1.51 |
| Active, Bishop search, default (reverse-curvature toe circle, 5.9 ft crack, arm 4.1 ft on R 14.1 ft) | 1.304 + warnings | 1.531 | 1.51 |
| Active, Bishop search, `reverse_curvature='discard'` | 1.503 | 1.531 | 1.51 |
| Active, Spencer search | 1.581 | 1.884 | |
| Active, GLE search | 1.609 | 1.575 | |
| Passive, Bishop on Slide2's active circle | 1.329 | | 1.32 |
| Passive, Bishop search, default / `'discard'` | 1.208 / 1.305 | 1.324 | 1.32 |
| Passive, Spencer search | 1.346 | 1.872 | |
| Passive, GLE search | 1.379 | 1.378 | |

The default Bishop search finds a lower F than Slide2 because it searches
reverse-curvature toe circles (centre 6 ft below the crest), which Slide2
would close the same way but its grid, placed above the slope, never
generates; on that circle the support passes 4.1 ft below the centre and
Spencer finds no lambda in -1.5 to 1.5 at which moment and force equilibrium
give the same F. With such circles discarded the module reproduces D&W and
Slide2. Slide2's Spencer (1.884 /
1.872) is far above its own GLE: on a phi = 0 circle every converged
moment-equilibrium method gives the same F, so its Spencer search must have
found no solution on the lower circles; the module's Spencer converges there
(GM-60) and gives values within the published spread.

### GLE / Spencer with large point loads (GM-60, 2026-10-14)
With a large support force the rigorous solver lost surfaces (Slide2 #85's
own circle: Spencer and GLE fell back to the unreinforced legacy value 0.913;
a search rejected it). Three causes, all in `gle.py`:
1. **Starting value.** The fixed-point step F = resisting / driving was
   abandoned when the driving term was <= 0 at the current estimate. With an
   anchor the horizontal driving term is negative at a small trial F and
   positive at the true one (the base normals grow with F through the
   c*l*sin(alpha)/F term), so the step from the start F = 1.3 could not be
   taken although the equation has a solution. With support forces the
   state is now solved by the direct march below, F by a bracketed root
   search (`_nearest_root`).
2. **The lagged iteration.** For a fixed lambda the solver iterated
   N -> F -> E -> X = lambda f E with X under-relaxed; with a concentrated
   load at a slice base it swings between two states and diverges, and the
   last state (returned after 80 steps) is not an equilibrium state: old
   "converged" #85 answers had lambda 0.57 / 1.00 where the true crossing is
   at 1.28 / beyond 1.5. With support forces present (lambda != 0) the state
   is now found by the direct march (`_GLESystem._march`): for each slice,
   vertical equilibrium (the base-normal equation) and horizontal
   equilibrium with X_{i+1} = lambda f_{i+1} E_{i+1} are two linear equations
   in N_i and E_{i+1}, solved slice by slice -- the Morgenstern-Price
   recursion in the closed form of Zhu, Lee, Qian & Chen (2005, Can. Geotech.
   J. 42: 272-278) -- and F_f is the F that closes the march (E_n = 0).
3. **The lambda search.** A lambda with no equilibrium state ended the scan;
   a bracket whose bisection did not close (F_f jumping between roots) was
   reported as the answer; the bound +/-1.5 itself was never tried (15 steps
   of 0.1 overshoot 1.5 by a rounding error); and no state at lambda = 0
   meant no answer at all. With support forces: such lambdas are stepped
   over, an unclosed bracket is passed over and the scan goes on, the bound
   is tried, and the scan starts from the nearest lambda with a state.
Without support forces nothing changes (the unreinforced path is
byte-identical). The toe-circle family's rigorous re-scoring
(`search_toe_circles`) now passes over a Bishop-shortlisted circle the
rigorous method cannot solve and tries the next (up to 12), so anchored #85
with Spencer finds toe circles again (1.584; before, none).

Measured on #85's printed circle: Spencer 1.5641 at lambda 1.005 (theta 45
deg), GLE half-sine 1.5641 at lambda 1.465, both equal to Bishop as they must
be for phi = 0 (Slide2 GLE 1.575). Each slice's horizontal and vertical
equilibrium closes to rounding (`test_search_reinforcement.py`); the old
lagged state on the circle (15, 39.667, R 29.661), reported as converged,
missed a slice's horizontal equilibrium by up to 199 lb/ft. Runtime of the
#85 default searches (once each, shared laptop): Spencer 40 -> 37 s, GLE
62 -> 32 s; #86 unchanged (Bishop 1.612, Spencer 1.606, GLE 1.606).

### A single surface: warnings, and no silent fallback (GM-81, 2026-10-15)
`analyze_slope` results (`SlopeStabilityResult`) now carry what the searches
said only about their governing surface, and say when the method asked for
has no solution:

- **`warnings`** (in `to_dict()` when there are any, `summary()`, the calc
  package's "Warnings" section): a reverse-curvature circle closed by a
  crack, with its depth (the text the search gives its critical circle,
  `search_supplements._reverse_curvature_warning`; with
  `reverse_curvature='discard'` the given circle is still evaluated and
  the note says so); supports crossing a Bishop / OMS circle with a short
  arm, with Spencer's FOS on the circle or that it has none
  (`_support_arm_warning`); a rigorous method with no solution; Bishop
  asked for on a noncircular surface (analysed with Spencer, as before).
- **`method_solved`** False when Spencer / Morgenstern-Price / GLE finds no
  equilibrium solution on the surface (the GLE does not converge; the
  reason, e.g. "no lambda from -1.5 to 1.5 gives moment and force factors
  of safety that agree", is in the warning). **`method_used`** names what
  produced `FOS`: the method itself, the "legacy approximate ... solver",
  or None. **`methods_without_solution`** lists the rigorous FOS fields
  (headline or `compare_methods`) that hold no converged solution.
- **What `FOS` holds when there is no solution** (decided here):
  - *Supports cross the surface: `FOS` is None* (to_dict null,
    `is_stable` None; the calc package prints no stability check). The
    legacy approximate solver ignores supports, so its value is that of
    the slope WITHOUT them: on #85's anchored toe circle 0.891 for a
    surface Bishop puts at 1.304. Before GM-81 that number came back as
    the Spencer FOS with only a Python warning. It is not returned at all.
  - *No support crosses it: `FOS` is the legacy approximate solver's
    value, labelled* (`method_solved` False, `method_used`, the warning
    giving theta or lambda). That solver answers the same slope; on a
    phi = 0 circle with theta = 0 it gives Bishop's value, which every
    method satisfying moment equilibrium gives there. Values are unchanged
    from before: `methods.solve_rigorous` makes the same solves as
    `spencer_fos` / `morgenstern_price_fos`.
  The same rule applies to the `compare_methods` Spencer / M-P values.
- **Searches** never consume `analyze_slope`; they score with
  `search._compute_fos`, which rejects a reinforced circle and any
  noncircular surface with no GLE solution (GM-53, SS-6) and passes over
  such toe circles in the rigorous re-scoring (GM-60). A circle no support
  crosses is still scored by the legacy solver when the GLE fails (#85
  without its support: every circle, see below), so when the governing
  circle is one of those the search labels it
  (`_legacy_critical_warning`: `critical.method_solved` False, a warning).
- **Found on the way.** Without supports the GLE keeps the old lambda
  search (GM-60 made only the supported one robust), and on #85's clay
  without its support it finds no solution on any circle tried (a 6 x 6
  Spencer search called the legacy solver 415 times); the legacy value is
  Bishop's there. Not changed here.

### One sliding mass per trial surface (GM-122, GM-124; 2026-10-16)
**The rule.** Limit equilibrium takes the soil above the slip surface as one
free body, divided into slices (Duncan, Wright & Brandon 2014, Ch. 6). A
trial surface that is below the ground over more than one stretch leaves
the ground and comes back into it: it bounds more than one mass, with air
between them that belongs to neither. `find_entry_exit` refuses it with
`GroundCrossingError` (a `ValueError`), circle or polyline. Slide2 does the
same: "Each valid slip surface analyzed by Slide2 always has exactly two
intersection points with the slope (ground surface)", and "More than two
slip surface / slope intersections with no valid slip surface" is its
error -105, a surface not analysed.

**Before**, a circle was taken from its first crossing to its last. The
slices in the air had no height and were dropped one by one, so the slice
count, the reported entry and the layers along the surface were wrong. On
the w10 geometry (F61: ground (0, 0), (20, 0), (30, 10), (60, 10); 3.5 m of
residual clay over sandstone): the circle (10, 39.5) R 40 crosses the
ground at x 3.70, 16.30, 21.06 and 37.01 and read Bishop 13.55 on 34 of 40
slices (the face mass alone: Spencer 7.40); the default search's critical
circle (0, 50) R 50.92 read 1.143 on 9 of 40 slices, with its entry named
at x 9.64 in the level ground in front of the toe and sandstone along its
base, while its slices lay on the face from x 26.55 to 31.51.

**Why refuse rather than score one of the masses.** No reference says
which of two masses a circle bounds is the surface to report, and picking
one (say the one the model's end does not cut) makes the answer depend on
the model's width. The circles and polylines that bound each mass alone are
scored by the search anyway. This is the SS-5 precedent: a surface below
the deepest layer once had its middle slices "silently dropped" and gave a
"meaningless (often absurdly low) FOS"; it is refused instead.

**Tolerances.**
- A point within 1e-6 m of the ground neither starts nor ends a stretch: a
  circle the toe-circle family builds through the toe, or an entry / exit
  circle through its entry point, is on the ground there, whatever the
  rounding.
- A stretch whose greatest depth is under 1 % of the deepest stretch's only
  touches the ground and is left out (`_TOUCH_FRACTION`). V-042's pinned
  circle passes 1.3 cm over the toe and dips 0.2 mm under the ground in
  front of it: it is now the face mass, x 5.01 to 18.22, every slice kept
  (Bishop 1.336). Taken from the touch at x 4.30, the slice at the toe fell
  in the air and was dropped (1.325). Published: Hassan & Wolff 1.334,
  Slide2 1.340.
- A circle below the ground at an end of the profile runs out of the model
  and is refused, as it was before (`ValueError`, "does not intersect the
  ground surface at 2 points").

**Counted, not warned.** Every search counts its refusals:
`SearchResult.n_rejected_ground_crossings`,
`to_dict()["n_rejected"]["ground_crossings"]`, a line in `summary()`. A
search run inside another adds its count to the outer one (a contextvar
tally, `search._counts_ground_refusals`), so `search_critical_surface` and
`search_rapid_drawdown` count their supplementary families' too. There is
no warning: the default search on w10 refuses 35 of its circles, routinely.

**Callers.** Everything builds slices through `find_entry_exit` and already
refused a `ValueError`: `search._compute_fos` and `_check_entry_exit` (which
count), `analyze_slope` (raises to its caller), rapid drawdown (through
`_compute_fos`'s `fos_fn`), the supplements, the figures
(`plotting._circle_arc`), the interactive calc package and Newmark.

**What moved** (slope "not slow" suite and the slope validation examples,
before and after): every published value the tests pin stayed put except
V-042, which moved toward the published one: the pinned circle 1.325 ->
1.336 (FOSM beta_LN 2.244 -> 2.308 against Hassan & Wolff's 2.336; Monte
Carlo Pf 1.75 % -> 1.65 %), its entry / exit search 1.3317 -> 1.3324
(before: a circle entering at x 3.33 in front of the toe, 26 of 30 slices).
Two coarse test searches with no published value moved: a 4 x 4 grid
1.0999 -> 1.0986 (before: 21 of 30 slices kept) and a 4 x 4 entry / exit
search 1.2562 -> 1.2622 (before: a circle through the toe that runs out of
the model's left end under the level ground, scored as the face mass
because rounding put it a hair above the toe; now refused, as Slide2
refuses "Two slip surface / slope intersections, but one or more slip
surface / nonslope External Boundary", its error -103). Main-search minima
before the supplements and the refinement moved by up to 0.002 (ACADS 1(a)
entry / exit 0.9858 -> 0.9879; reported 0.9848 unchanged, published 1.00,
Slide2 0.987). On w10 the default search then reported Bishop 6.346 (a
toe circle) / Spencer 6.349, both through the toe, where it had reported
1.142 / 1.142 on the circle above: the refusal uncovered a gap in what the
search covers, below.

**What the refusal uncovered: circles passing above the toe.** With the
full defaults on w10 (F61: nx = ny = 10, 40 slices) the main search tries
centres over x 0-60, z 11-50 (10 x 10) and at each 20 radii
(`optimize_radius`: from 0.5 m below the highest ground to the distance to
the profile's first point at the model base), then golden-section refines
the best. Its automatic windows (GM-57) are entry x 0-25, exit x 25-60:
they meet at the face's mid-length (x 25, z 5), so every circle they admit
enters below z 5, in the sandstone; the clay comes out on the face above
z 6.5. Of the 2,000 coarse circles, 139 lie in the windows (lowest Bishop
6.92), 1,728 do not cross the ground twice, 12 are refused and 121 bound
one mass but fall outside the windows, 117 of them with their lower end on
the face above the middle; the lowest of those, 0.980 (centre (26.67,
11.0), R 4.45, x 26.55-31.00), runs through the clay. The radii reach the
face circles; the windows refuse them. Before GM-122 they passed only as
circles that crossed the ground four times, taken from a first crossing in
front of the toe.

FHWA NHI-06-088 p. 6-38: "If there is a weak layer above the toe of the
slope, a circle passing above the toe of the slope may be more critical.
Similarly, if there is water outside the toe of the slope, a circle passing
above the water may be more critical." The search now scores them as a
supplementary family, `circle_above_toe`, run with the toe circles
(`search_supplements.search_toe_circles`, `toe_circles=True`):
- from points on each face at 0.1 to 0.9 of its height, each chord to the
  ground upslope (the toe circles' far-end offsets) and each centre along
  its bisector (u = ln(t/c), 0.02 to 4) is scanned with Bishop; the best
  start's height is refined by golden section with the far end held, then
  the far end; Spencer / M-P / GLE, and in this family Janbu too, re-score
  the best three with the method itself. Janbu is screened here because
  these circles can enter the face steeply, where Janbu's solve falls with
  the slice count: on Slide2 #85's clay face without its support, scanned
  with Janbu itself, the family reported 0.697 on a circle (entry about
  67 deg) that Bishop and Spencer read 1.22 and Janbu 0.86 on 20 slices,
  0.70 on 30 and none on 40 or more. Screened, it gives 1.033 (1.036 on
  120 slices) and the toe circle's 1.024 governs, as before the family.
  The main search and the toe circles keep scanning with Janbu itself;
- a circle must come out at its start (one that comes out lower is a toe
  or base circle), keep its centre above the ground at both ends (no
  reverse-curvature circle closed by a crack) and keep m_alpha =
  cos(a) (1 + tan(a) tan(phi) / F) >= 0.2 on every slice under Bishop or
  Janbu, which Slide2 checks by default (error -112). The family leaves
  out the circles that fail either check, and the "surface families
  tried" entry of `choices` says so. Without the last two, Duncan Ex 4 read
  0.17 on a reverse-curvature circle with negative m_alpha on its steep
  slices (1.93 on 80 slices, Spencer 1.53), and Slide2 #85 0.865 on a
  circle closed by a crack 8.45 deep, 42 % of the slope's height;
- caller-given windows and centre ranges bound it; the automatic ones do
  not;
- with a per-surface solve (rapid drawdown) it starts at 0.35 and 0.7 of
  the height and does not refine the height.

A family rather than more radii or wider default windows: the radii
already reach these circles and the windows refuse them, and widening the
GM-57 windows changes every main search the tests pin. The family is
bounded (about 2,000 Bishop solves a face), says when it governs (a
warning naming the start, its height above the toe and what kept the main
search off it) and reports what it covered: `candidates`,
`governing_surface`, `search_ranges['faces_covered']` (each face's toe and
crest, the heights its circles started from, whether the height was
refined) and the "surface families tried" entry of `choices`. Nothing in it
looks at the layers.

On w10 (F61 settings) the default search now reports Bishop 0.936 /
Spencer 0.935 / Janbu 0.938 on a circle from x 26.50 on the face (z 6.50, where the
clay's base comes out) to 30.81, every slice base in the clay; main search
6.349, toe circle 6.346. On the weak layer over rock of
`test_search_ranges` (4 m over rock meeting the face 6 m up): 4.82 -> 0.884
against 0.885 from a 20 x 20 search with both windows over the whole
profile. Run time of a default search: Bishop about 1.5 -> 2.5 s on w10
and on the pinned cases (ACADS 1(a), V-042, Duncan Ex 1 drawn long, Slide2
#86); Spencer (10 to 21 s) about 1 s more, inside the run-to-run scatter;
a rapid-drawdown search (V-048, #98, 6 x 5 grid) 14 -> 19 s. What moved with it (slope "not slow" suite and slope
validation examples): Slide2 #85 with reverse-curvature circles discarded,
1.485 -> 1.467 active and 1.302 -> 1.296 passive at 30 slices, a circle
coming out 0.24 / 0.12 above the toe that reads 1.508 / 1.318 at 240
slices against D&W 1.51 / 1.32 (Slide2 1.531 / 1.324); V-052b Corps
0.724 -> 0.722 (published 0.823); two coarse test searches with no
published value, 1.0986 -> 1.0789 and 1.2622 -> 1.2606. Every other pinned
value stayed put.

**GM-124: entry and exit windows.** The entry is the low-x end of a surface
and the exit its high-x end, whichever way the slope faces
(`find_entry_exit` returns them in that order). `search._ordered_windows`
reads a window given high bound first low bound first, and swaps an entry
window lying wholly at or right of the exit window, with a note at the head
of `warnings`; `grid_search`, `search_entry_exit`, `search_critical_surface`
and `search_rapid_drawdown` apply it. Given the other way round, no circle
could start in one window and end in the other, and the entry / exit
search returned nothing and said nothing (F61 [13], [14]); it now also
says when it scored no circle at all.

## Critical Sign Conventions
- **Driving moment**: uses `W*(x_mid - xc)/R` + `abs(driving)`, NOT `W*sin(alpha)`
  directly, because `sin(alpha)` can be negative for L-to-R slopes
- **Seismic moment**: computed separately from gravity driving, then
  `driving = abs(gravity) + abs(seismic)` so seismic always increases
  driving regardless of slope direction (L-to-R vs R-to-L)
- **Spencer m_alpha**: `cos(alpha - theta) + sin(alpha - theta)*tan(phi')/F`
  (reduces to Bishop when theta=0)

## Key Technical Notes
- **Spencer = Bishop for circular surfaces (the legacy approximate solver
  only)**: in `spencer_fos_legacy` the sin(alpha) = (x-xc)/R identity makes
  moment and force driving identical, forcing theta=0. The rigorous Spencer
  (`gle.py`, the default) does not; on a circle it differs from Bishop where
  phi > 0 (e.g. Slide2 #86 with tiebacks: Bishop 1.612, Spencer 1.606) and
  equals it for phi = 0
- **Spencer noncircular (the legacy solver)**: uses interslice force marching
  (no moment arm). FOS is the value that makes the last interslice force = 0.
  The rigorous GLE (the default) satisfies moment equilibrium as well, about
  its own moment axis.
- **For c'=0 (cohesionless)**: circular analysis overestimates FOS vs infinite
  slope solution tan(phi)/tan(beta) — this is a well-known limitation. Since
  GM-14 `search_critical_surface` scores shallow planar surfaces on every
  cohesionless face and reports the infinite slope when it governs (see
  "Supplementary families" above); a circle scored on its own
  (`analyze_slope`) still carries the limitation.
- **Coordinates**: elevation-based (x, z where z=elevation), NOT depth-based
- **SlopeSoilLayer**: separate from geotech_common/SoilLayer;
  `SoilProfile.to_slope_stability_input()` maps a SoilProfile to
  SlopeSoilLayer dicts
- **Undrained**: phi=0, cu used. Bishop = Fellenius when phi=0 (m_alpha = cos(alpha))
- **FOS convergence guard**: in `_compute_fos` (search.py), FOS < 0.05 is treated
  as non-convergence and returns FOS_MAX (999). Prevents negative/absurd FOS from
  polluting search results.
- **Below-the-base rejection, generalized (SS-5, v5.4 F2c)**: a trial slip
  surface whose base drops below the bottom of the DEEPEST soil layer passes
  through material that does not exist, so it is geometrically inadmissible.
  `build_slices` skips such below-base slices and, after the loop, REJECTS the
  surface (raises `ValueError`, which `_compute_fos` maps to `_FOS_MAX` so a
  search discards it) when EITHER (1) a below-base gap is bracketed by surviving
  slices on both sides (the original interior-hole check), OR (2) the below-base
  span exceeds a small tangential-sliver tolerance `max(2, round(0.03·n_slices))`
  (NEW). Trigger (2) closes a hole in trigger (1): a circle that plunges below a
  shallow rigid base right after entry and re-emerges only near the exit leaves a
  ONE-SIDED surviving fragment (all below-base x fall OUTSIDE
  `[slices[0], slices[-1]]`, so no bracketing pair exists), and the tiny
  exit-side fragment returned a spurious low FOS that won searches (Slide2 #85
  steep clay on a rigid base at el 10: a circle dipping ~4 ft below the base
  scored ~0.72 on 12 of 40 slices). **Why this cannot exclude a legitimate
  critical surface:** the true critical surface lies entirely within the soil; it
  can at most graze the deepest layer bottom TANGENTIALLY, which from finite
  slicing touches at most a slice or two at the tangent point — well inside the
  sliver tolerance. Any surface with more below-base slices genuinely leaves the
  soil and is impossible, independent of the strength parameters, so the rule is
  default-ON. Proven no-shift: the full `slope_stability/` (432) and
  `validation_examples/` (179) suites pass byte-identically with the guard (no
  pinned FOS moved — the guard only ever rejects surfaces that previously
  produced a spurious value). See `tests/test_search_robustness.py`. (The
  separate c'=0 case — a circular search over-estimating the critical for a
  c'=0 slope whose true minimum is the planar infinite slope — is NOT part of
  this rule; it is handled by the infinite-slope family, see "Supplementary
  families" under Search Algorithms.)
- **Noncircular degenerate-surface guard (SS-6, v5.3 B2)**: the random / weak-layer
  polyline generators can emit jagged zig-zag surfaces on which the rigorous GLE
  either fails to converge (the legacy Spencer fallback then returns a spurious
  low FOS ~0.05-0.2) or converges to a spurious low value — a search would wrongly
  report that as the critical surface (seen on ACADS 4). `_compute_fos` guards
  noncircular trials three ways: (1) a cheap geometric admissibility check rejects
  sliver / too-short-span / too-few-slice surfaces (`_noncircular_admissible`);
  (2) the rigorous FOS REQUIRES GLE convergence (`_rigorous_noncircular_fos`
  returns None otherwise, instead of falling back to the legacy value); and
  (3) a **low-FOS jaggedness gate** rejects any surface whose FOS is below
  `_LOW_FOS_JAGGED_GATE` (1.5) AND is non-concave / near-vertical (`_is_jagged`) —
  a low FOS on a jagged surface is a solver artifact, not a real critical surface.
  The gate is deliberately scoped to LOW-FOS surfaces so the generators keep
  exploring the many high-FOS jagged trials (which harmlessly lose the search).
  Circular surfaces are untouched. See `tests/test_search_robustness.py`.
  - **Span floor is problem-anchored, not model-wide** (v5.3 review fix): the
    admissibility span floor is anchored to the caller's entry/exit WINDOW extent
    (`_window_span`, half the window centre-to-centre distance) when a search
    supplies one, else to the local slope HEIGHT (0.5·H). The previous floor of
    15% of the FULL model width silently rejected legitimate LOCALIZED failures in
    a wide model (a long tailwater bench or wide right-of-way inflated the floor
    past a real slide length). The min-slice check is unchanged.
  - **Silent rejections are counted** (v5.3 review fix): each noncircular
    generator passes a `reject_stats` tally into `_compute_fos`; the per-reason
    counts (geometry / non-converged / jagged) land on
    `SearchResult.n_rejected_*` (and `to_dict()['n_rejected']`), and a search that
    rejects a MAJORITY of its trials emits a `warnings.warn` so an
    under-resolved / mis-windowed search is not silently reported as a clean
    result.
  - **Convergence-gate moment axis has a robust fallback** (v5.3 review fix):
    `_rigorous_noncircular_fos` first tries gle's internal least-squares axis
    (byte-identical to before) and, only if that fails to converge, retries with
    an explicit pinned `axis_point` (`_noncircular_axis_point`: span midpoint,
    half a span above the crest). The least-squares fit can land absurdly far
    away on a near-straight surface and spuriously fail; the second pass rescues
    such a valid surface without ever discarding a convergence the first pass
    already found (so no existing search result moves). A converged GLE FOS is
    axis-independent.
- **`infinite_slope_fos` (v5.3 B2c)**: closed-form planar/translational FOS for
  the shallow surface-parallel mechanism, `analysis.infinite_slope_fos(slope_angle,
  phi, gamma, c=0, depth=1, water_condition='dry'|'seepage_parallel'|'ru', ...)`
  → `InfiniteSlopeResult`. FOS = [c' + (γz cos²β − u)tanφ']/(γz sinβ cosβ);
  cohesionless dry reduces to tan φ'/tan β (depth-free), seepage-parallel at the
  surface to (γ'/γ)·that. Validated vs Slide2 #79 (1.44) / #81 (1.15). This is
  the exact answer for the c'=0 circular-overestimate limitation noted above;
  the search reaches it through the infinite-slope family (GM-14).
- **Rapid drawdown (v5.3 B2a)** — `rapid_drawdown.rapid_drawdown_fos(geom, from_el,
  to_el, xc/yc/radius, method='corps_2stage'|'lowe_karafiath'|'duncan_3stage')`
  → `RapidDrawdownResult`
  (also `analysis.rapid_drawdown_fos`). Low-permeability layers carry the total-stress
  **R-envelope** (`R_c`, `R_phi`) alongside the effective `c_prime`/`phi`; `R_phi is
  None` = free-draining. The method is wired to the GLE engine by OVERRIDING each
  undrained slice's base strength to a fixed `c = τ_ff, φ = 0` and re-solving. Theory:
  * **Stage 1** — effective-stress ("S") analysis at the FULL pool gives the
    consolidation stresses on each base: σ'_fc = N'/l and τ_fc = S_mob/l (from the
    GLE per-slice effective normal / mobilized shear).
  * **Stage 2** — undrained strength τ_ff. Corps 2-stage: τ_ff = min(R-envelope,
    drained) at σ'_fc (the "combined" envelope, capping R-envelope over-strength at
    low σ'). Duncan 3-stage AND Lowe-Karafiath 2-stage: linear interpolation with the
    consolidation stress ratio Kc = σ'_1c/σ'_3c between the Kc=1 (R / IC-U, lower) and
    Kc=Kf (drained, upper) envelopes: τ_ff = τ_R + (Kc−1)/(Kf−1)·(τ_drained − τ_R),
    capped at the drained (Kf) bound. Kc per slice is back-figured from σ'_fc, τ_fc
    and the base angle α assuming a vertical major principal consolidation stress;
    Kf = (1+sinφ')/(1−sinφ'). **Lowe-Karafiath (1960)** uses this identical stage-2
    strength but stops here (no stage 3); Duncan-Wright-Wong (1990) is Lowe-Karafiath
    plus the stage-3 drained check, so `lowe_karafiath` FOS ≥ `duncan_3stage` FOS
    always (stage 3 only ever *substitutes a lower* strength). On the exact #95/#96
    dam `lowe_karafiath` lands at 1.36 flat / 1.45 seepage — the seepage value sits
    on the published Duncan-Wright-Wong 1.443, corroborating that the Kc stage-2 is
    sound and the default 3-stage's deficit is entirely the Fellenius stage-3 normal
    (see the E2 note below), not the interpolation.
  * **R-envelope input convention — the Kc=1 line (EM G-12 reconciliation).** The
    Kc=1 (R / IC-U) envelope is taken DIRECTLY as `τ_R = R_c + σ'_fc·tan(R_phi)`,
    i.e. `(R_c, R_phi)` are the strength envelope ALREADY plotted as τ_ff vs the
    effective consolidation stress σ'_fc — the "Total Stress R Envelope (linear)"
    Slide2/GeoStudio expose and that the EM 1110-2-1902 App. G verification
    problems (#95–#99) tabulate. This is NOT the raw isotropically-consolidated CU
    triaxial envelope: EM Eq. G-12 maps that raw envelope onto the τ_ff-vs-σ'_fc
    plane via d = R_c·cos(R_phi)·cos(φ')/(1−sin R_phi) and ψ = atan[sin(R_phi)·
    cos(φ')/(1−sin R_phi)]. Feeding the raw parameters through *and* letting the
    module treat them as the plotted line would double-apply that transform and
    OVER-predict: for #96 (R_c=1200 psf, R_phi=16°, φ'=30°) the G-12 image is
    (d=1379 psf, ψ=18.24°), which drives `lowe_karafiath` to 1.53 seepage —
    overshooting the published 1.443 that the raw (as-plotted) inputs reproduce at
    1.45. So supply `(R_c, R_phi)` as the τ_ff-vs-σ'_fc envelope (Slide2
    convention, validated by V-037/V-048); a user starting from raw CU triaxial
    data applies EM G-12 first. Confirmed empirically (raw 1.45 vs G-12 1.53 on
    #95/#96; published 1.443).
  * **Stage 3** (3-stage only, `duncan_3stage`) — where the post-drawdown DRAINED
    strength (low-pool effective stress) is less than the stage-2 undrained strength,
    it is substituted
    (Duncan, Wright & Brandon 2014, Ch. 9: third-stage drained strength on the
    drawn-down effective stresses).
  * The final FOS is a GLE/Spencer solve at the DRAWN-DOWN pool (external water load
    removed to the low level; undrained slices carry τ_ff as φ=0 total-stress
    strength). Validated vs Slide2 #95 (2-stage) / #96 (3-stage) — see RESULTS V-037/V-038.
  * **Stage-1 seepage option (`stage1_phreatic_points`)** — the DEFAULT stage-1 uses a
    flat full-pool phreatic (hydrostatic to the reservoir), the conservative
    no-through-seepage bound. A low-permeability embankment under an established full
    pool actually has a steady-state seepage field whose phreatic DECLINES through the
    dam, so the true σ'_fc (and mobilized τ_ff) are higher. Passing the flow-net /
    Casagrande phreatic line as `stage1_phreatic_points` (a `[(x,z),…]` surface that
    starts at the pool level on the upstream face, preserving the external reservoir
    load there) reproduces the steady-seepage condition Slide2/EM 1110-2-1902 use.
    Default `None` = unchanged. For Slide2 #95 this recovers the Corps 2-stage FOS from
    the conservative 1.21 to 1.34 vs the published 1.347 (~0.6%).
  * **Duncan 3-stage for c'=0 soils — stage-3 normal (v5.4 E2).** For #95/#96 the
    default 3-stage lands ~12% below the published 1.443. The V5.3 note blamed the Kc
    interpolation; the E2 re-investigation found the Kc interpolation is sound (it yields
    ~1.45 on its own) and the deficit is in the STAGE-3 drained substitution: the
    drawn-down effective normal σ'_post was estimated with the Fellenius `W·cosα/l − u`
    term, which under-predicts N' (it neglects interslice forces) and so substitutes a
    too-low drained strength on 17/50 slices. The optional
    `stage3_effective_normal='gle'` uses the RIGOROUS GLE effective normal from a
    drawn-down drained solve (the same `base_normal_eff` basis stage 1 uses) — 9
    physically-genuine substitutions instead of 17 — lifting #96 to 1.306 (flat) /
    1.370 (seepage), closing most of the residual (~5% left, within the
    representative-flow-net + LE-N'-at-FOS sensitivity that shows as the +0.6% Corps
    residual). **Default is `'fellenius'` (byte-identical to before);** the refinement
    is opt-in. Only the 3-stage has a stage 3, so the 2-stage is untouched.
    The default is the Ordinary Method of Slices estimate of the normal, as
    FHWA NHI-06-088 sets the method out (pp. 6-12 to 6-13: the side forces
    ignored, the uplift at the base taken off the total weight);
    'gle' takes the normal from a rigorous equilibrium solution, the basis
    stage 1 uses, and comes closer to Duncan, Wright & Wong (1990)'s published
    3-stage values (#96: 1.31 / 1.37 against 1.443; 'fellenius' 1.235 /
    1.273). The default stays 'fellenius' by the owner's ruling.
  * **The Fellenius normal under the remaining pool (GM-57 review).** Water
    ponded above a slice raises its total normal stress and its pore pressure
    equally (FHWA NHI-06-088 p. 6-13: total weights with the boundary water
    forces, or submerged weights, give the same result), but the per-slice
    `W·cosα/l − u` counted the pond's pressure on the base in full and its
    weight only at cos²α. Under a deep pool on an inclined base σ'_post went
    negative, was clamped to zero, and the drained substitution left no
    strength: the true Pilarcitos section searched fully automatically gave
    DWW 0.052 on a toe circle wholly under the remaining pool (2.048 with
    'gle'). `_fellenius_effective_normal` takes the pond out of both W and u:
    that circle now gives 1.57; slices with no pond are byte-identical. (The
    stage-1 fallback, used when the full-pool GLE fails, uses the same
    estimate.) The default stays `'fellenius'` (the owner's ruling).
- **Rapid-drawdown critical-surface search (v5.4 E1)** — `rapid_drawdown.
  search_rapid_drawdown(geom, from_el, to_el, method=…, surface_type='circular'|
  'entry_exit'|'noncircular'|'noncircular_de', …)` → `RapidDrawdownSearchResult`
  (also `analysis.search_rapid_drawdown`; exported at the package level). It finds the
  MINIMUM-FOS surface with the drawdown strength substituted per trial, for both stage
  methods and both circular/noncircular surfaces. It is a THIN composition: `search.py`
  gained an optional `fos_fn(geom, slip)->FOS` hook on `_compute_fos` and the search
  entry points (`grid_search`/`optimize_radius`/`search_entry_exit`/`search_noncircular`/
  `search_de`), default `None` = byte-identical; the wrapper passes a drawdown-FOS
  closure so the grid/entry-exit/random/DE loops, radius optimisation, entry/exit
  filtering, and noncircular guards are all SHARED — no search internals are duplicated.
  The result carries the ordinary rich `SearchResult` plus the stage-level
  `RapidDrawdownResult` recomputed on the winning surface. **Robustness gate:** the
  drawdown closure REJECTS a trial surface whose stage-1 or stage-3 GLE did not converge,
  whose drawn-down solve retained too few slices, or whose stage-1 (full-pool) FOS is
  implausibly low (<0.5) — a large near-flat circle self-clips to a handful of slices
  whose GLE "converges" to a spurious low FOS (~0.18) that would otherwise win the
  search. Validated vs Slide2 #98 Walter Bouldin search minima (Corps 0.931 / DWW 1.039)
  on the RECOVERED geometry (~10% low, geometry-limited) and by exact wrapper mechanics
  on #95/#96 — see RESULTS V-041. Note: each trial is several LE solves + two geometry
  deep-copies, so a circular grid search is heavy; prefer a modest `nx·ny` or
  `entry_exit` and bounded entry/exit windows (an O(n_surfaces) hotspot, not optimised).
- **Newmark seismic sliding block (v5.3 B2b)** — `newmark.py`, three functions
  (exported + adapter `yield_acceleration` / `newmark_displacement` /
  `newmark_jibson2007`):
  * **`yield_acceleration`** — the yield seismic coefficient ky (Newmark critical
    acceleration ay = ky·g) for a SPECIFIED surface, by bisection on the module's
    pseudo-static FOS (`analyze_slope` with a trial `kh`; the seismic_force = kh·W path
    validated by Loukidis #62/#63 = V-033). FOS(kh) is monotone decreasing, so bisection
    on FOS=1 gives ky directly. If the static FOS ≤ 1, ky=0. The caller's `geom.kh` is
    not mutated (shallow copy).
  * **`newmark_displacement(ky, accel, dt, polarity=…)`** — rigid-block double
    integration (Newmark 1965), no upslope rebound (relative velocity clamped ≥ 0), so
    displacement is monotonic. Two polarity conventions:
      - `polarity="downslope"` (**default**, standard Newmark 1965 / Jibson 2007): the
        SIGNED record is integrated and the block is driven only when the ground
        acceleration exceeds ay in the destabilizing (downslope-positive) direction; the
        reverse polarity decelerates it. Only the destabilizing polarity of a symmetric
        record contributes → about HALF the rectified displacement.
      - `polarity="rectified"`: the ABSOLUTE record is integrated so both polarities drive
        the single downslope block — a conservative, orientation-independent option (use
        when the record's sign relative to the slope's downslope direction is unknown).
    Trapezoidal on the piecewise-linear relative velocity — EXACT for a rectangular pulse
    (`D = ap(ap−ay)T²/(2ay)`), the integrator's validation (identical under both
    polarities for a one-sided pulse).
  * **`newmark_jibson2007(ky, amax)`** — Jibson (2007) Eq. 6 regression,
    `log10 D[cm] = 0.215 + log10[(1−ky/amax)^2.341·(ky/amax)^−1.438]`, σ=0.510 log10,
    a time-history-free cross-check (valid 0 < ky < amax).
  * Slide2 #104 (Tutorial-28 record) is the target problem; its acceleration record and
    geometry are not published, so the integrator is validated analytically and Jibson
    against its published equation — see RESULTS V-039.
- **Stabilizing piles (v5.3 B2d)** — `reinforcement.StabilizingPile` on
  `geom.stabilizing_piles`, resolved by `compute_reinforcement_forces` like the other
  reinforcement (active convention: the resisting force reduces the driving moment) and
  applied where the vertical pile crosses the slip surface. Two ways to set the row's
  resistance:
  * **Specified shear** — `shear_capacity` (kN per pile); the per-metre force is
    `shear_capacity/spacing` (Slide2 #54 / Yamagami 2000). Direction 'horizontal'
    (default) or 'normal' (perpendicular to the slip surface).
  * **Ito & Matsui (1975)** — `ito_matsui=True` with the pile `diameter` (clear spacing
    D2 = spacing − diameter). `ito_matsui_pressure(c,phi,gamma,z,D1,D2)` is the ORIGINAL
    1975 plastic-deformation lateral force per unit depth on one pile (their Eq. 13),
    `p(z) = c·D1·[(A−2√N_phi·tanphi−1)/(N_phi·tanphi) + Fc] − c·(D1·Fc − 2·D2/√N_phi)
    + (gamma·z/N_phi)·(D1·A − D2)` with N_phi=tan²(45+phi/2),
    `A = (D1/D2)^(√N_phi·tanphi+N_phi−1)·exp[(D1−D2)/D2·N_phi·tanphi·tan(pi/8+phi/4)]`,
    `Fc=(2tanphi+2√N_phi+1/√N_phi)/(√N_phi·tanphi+N_phi−1)`. Note the exp argument is
    `tan(pi/8+phi/4)` and the first-term coefficient is `1/(N_phi·tanphi)` (per the
    printed equation; the D1·Fc contributions cancel). The phi=0 cohesive limit
    (Eq. 23) is `p(z)=c·{D1·(3 ln(D1/D2)+(D1−D2)/D2·tan(pi/8)) − 2(D1−D2)}+gamma·z·(D1−D2)`.
    Hand-checked: c=10/phi=20/gamma=18/z=5/D1=2/D2=1.5 → 105.079; c=25/phi=0/gamma=18/z=4/
    D1=2/D2=1 → 146.683. `ito_matsui_lateral_force` integrates p(z) from the pile head to
    the slip surface (closed form, linear in gamma·z); divided by the spacing for the
    per-metre force.
  * Validation: #54 (specified shear) → CONVENTION, RESULTS V-040 (no-pile +1.1%,
    with-pile +2.5%; residual = active-vs-passive support convention + figure-read pile
    location). The Ito-Matsui FORMULA is unit-tested for the #106 spacing trend (its
    cross-section is not in the manual).
  * **Support convention — active/passive (v5.4 E6).** `StabilizingPile.
    support_convention` = `'active'` (**default**) applies the unfactored pile force as
    a REDUCTION of the driving moment (Slide2 "Method A"); `'passive'` adds it to the
    RESISTING side instead (Slide2 "Method B"). Active gives a larger FOS gain for the
    same capacity (it shrinks the denominator), which slightly over-predicts vs a
    passive pile; passive is the more conservative treatment. Until GM-59 only the
    circular moment methods (Fellenius/Bishop) honoured the convention; now every
    method does, and nails, anchors and geosynthetics take it too (see "Support
    forces in each method" above). On #54 the
    passive option narrows the with-pile Bishop residual from +2.5% (active) to +0.8%
    vs the published 1.193 (VALIDATION.md B10 / RESULTS V-040-E6). Default 'active' is
    byte-identical. Implemented via `reinforcement.moment_resistance` /
    `horizontal_resistance` (passive forces) alongside the existing `*_reduction`
    (active forces), split on `ReinforcementForce.passive`.
- **Tension crack — side + model (v5.4 E4)**: `geom.tension_crack_depth` opens a
  vertical crack at the CREST end of the slip surface; `tension_crack_water_depth`
  fills it (hydrostatic thrust `0.5·γw·z_w²` at `z_w/3` above the crack base,
  treated downstream as an always-driving magnitude). Two selectors, both
  DEFAULT-PRESERVING:
  * `tension_crack_side` = `'entry'` (default, low-x) or `'exit'` (high-x) — the
    crack forms on whichever side the crest is on. The exit-side option removes
    the old need to MIRROR the slope so the crest landed on the entry side (the
    exit-side crack on the un-mirrored slope is machine-precision identical to the
    entry-side crack on the mirror, by symmetry).
  * `tension_crack_model` = `'strength'` (default) keeps the cracked wedge as
    zero-shear-strength DRIVING soil; `'truncation'` REMOVES it from the sliding
    mass (the mass ends at the vertical crack face), matching Slide2/UTEXAS.
    Truncation is LESS conservative (removes the driving wedge). On Slide2 #2 /
    ACADS 1(b) the truncation model reproduces the published water-crack FOS
    (Bishop 1.596/Spencer 1.592/GLE 1.592) to <0.1%, resolving the V-026
    strength-model conservatism (documented there as a CONVENTION). See
    VALIDATION.md B9 / RESULTS V-026.
- **Empty-space slices**: slices where the slip surface is below all soil layers
  are skipped (zero weight, no contribution)
- **Nails**: `SlopeGeometry.nails` (`nails.SoilNail`) reach every method
  through `reinforcement.py`, with anchors, geosynthetics and stabilizing
  piles (see "Support forces in each method" above). The old nail path in
  `nails.py`'s own FOS integration is not used; its tests stay skipped.
- **The legacy Spencer / Morgenstern-Price are APPROXIMATE (SS-1)**:
  `spencer_fos_legacy` / `morgenstern_price_fos_legacy` use a shifted
  m_alpha = cos(alpha−theta) + sin(alpha−theta)·tanφ/F formulation (with
  theta = const for Spencer, atan(λ·f(x)) for M-P) and a force-equilibrium
  driving term W·(sinα + cosα·tanθ), iterated until FOS_moment = FOS_force.
  This is an engineering approximation, NOT the textbook-exact GLE
  interslice-force recursion. `spencer_fos` / `morgenstern_price_fos` (the
  defaults) wrap the rigorous `gle.gle_fos`; the legacy forms are only the
  fallback where the GLE finds no solution, and since GM-81 the result says
  so (`method_solved`, `method_used`), or gives no FOS where supports cross
  the surface (the legacy forms ignore supports).
- **Bishop numerator clamp (SS-2, v5.1)**: the frictional term uses
  max(W − u·b, 0)·tanφ so artesian/perched pore pressure (u·b > W) cannot
  contribute negative resistance; consistent with the Fellenius
  effective-normal clamp. Spencer/M-P share the (W − u·b) form but are left
  unclamped to preserve the Duncan-validated behavior (noted for follow-up).
- **Pore pressure model (SS-3)**: `_pore_pressure_at_base` uses the full
  hydrostatic head to the GWT surface (no cos²β seepage correction for
  inclined phreatic surfaces) — slightly conservative (overestimates u) on
  steep water tables. The layer `ru` coefficient is the alternative.
- **Pore-pressure GRID / TIN input (v5.4 E3)**: `SlopeGeometry.pore_pressure_points`
  is an optional list of scattered `(x, z, u)` triples (kPa) — a flow-net / TIN
  sampling of the pore-pressure field. When set, the base pore pressure at each
  slice is INTERPOLATED from it by `geometry.build_pore_pressure_interpolator`
  (scipy `LinearNDInterpolator` = linear on the Delaunay triangulation, with a
  `NearestNDInterpolator` fallback outside the convex hull and for degenerate
  point sets < 3 points / collinear; suction clamped to u ≥ 0). It OVERRIDES the
  piezometric-line (`gwt_points`) and per-layer `ru` base pore pressure and is
  wired through the search path (build_slices builds the interpolator once per
  call). The ponded-water buttress + submerged weight still come from
  `gwt_points`, so a reservoir over a flow-net field sets BOTH. Default `None` =
  unchanged. A grid encoding a hydrostatic field reproduces the `gwt_points`
  base pressure (hence FOS) to machine precision; TIN-linear is exact for a
  linear field. Enables Slide2 #10 / ACADS 5 (pore-pressure-grid, previously
  N/A-scope) — see VALIDATION.md B8 / RESULTS V-029. (scipy imported lazily, as
  in `search.search_de`.)
- **Multiple surcharge zones (v5.4 E8)**: `SlopeGeometry.surcharges` is an optional
  list of `(pressure_kPa, x_start, x_end)` zones for problems with several distinct
  loaded areas (a bench load + a crest load, etc.). `surcharge_at(x)` SUMS every zone
  covering x on top of the single `surcharge`/`surcharge_x_range` pair (which is
  unchanged), so slices and the search pick them up with no change (both route through
  `surcharge_at`). Default `None` = single-surcharge behaviour byte-identical. A
  linearly-varying (trapezoidal) load is represented as its mean uniform pressure or as
  several thin zones. Exercised on Slide2 #9 / ACADS 4 (bench + crest loads) — see
  VALIDATION.md B11 / RESULTS V-028.
- **Probabilistic FOS — correlated depth-varying su law (v5.4 F1)**: `probabilistic.py`
  normally varies each scalar parameter (phi/c_prime/cu/gamma/gamma_sat) INDEPENDENTLY
  per targeted layer. That over-counts a strength profile whose uncertainty is a single
  shared su-GRADIENT spread across many thin sub-layers. The additive `linear_su` law
  (a `variables` entry carrying a `'law'` key) treats `su(z) = a + b*(datum_z - z)` as
  ONE correlated `(a, b)` random-variable pair, applied COHERENTLY to every targeted
  layer's cu (each layer's mid/top/bottom elevation, `z_ref`; floored at `su_min`; layer
  forced undrained). FOSM adds the Taylor cross-term
  `Var = (dFa/2)^2 + (dFb/2)^2 + 2*rho_ab*(dFa/2)*(dFb/2)` (a std-0 component, e.g. a
  fixed intercept, drops out); Monte Carlo samples `(a, b)` from a bivariate normal via
  Cholesky (`b = b_m + b_s*zb`, `a = a_m + a_s*(rho*zb + sqrt(1-rho^2)*za)`). The scalar
  path is byte-identical when no law is present (variables split; scalar RNG stream drawn
  first). Also NEW: **`gamma_sat` is now a valid scalar variable** — for a submerged slope
  the operative unit weight is `gamma_sat` (below the water table), and its uncertainty is
  buoyancy-amplified (the buoyant weight `gamma_sat - gamma_w` carries a larger relative
  spread than `gamma_sat` itself), so varying the dry `gamma` alone is inert. Closes the
  Duncan (2000) LASH input-COV FOSM gap — see VALIDATION / RESULTS V-030 and
  `validation_examples/test_published_v030_fosm_slope.py`.
- **Probabilistic FOS — correlated SCALAR pairs (v5.4.1)**: the same correlation
  machinery generalizes from the su-law `(a,b)` pair to arbitrary scalar variable
  pairs via the `correlations` argument of `fosm_fos` / `monte_carlo_fos` — a list
  of `(key1, key2, rho)` triples over the scalar variable keys (e.g.
  `[("c_prime", "phi", -0.5)]`, keys optionally `:LayerName`-scoped). FOSM adds
  the Taylor cross-term `2*rho*(dF1/2)*(dF2/2)` as a signed `corr(k1,k2)` entry in
  `variable_variance_pct` (so the per-variable shares still total 100%); Monte
  Carlo draws the pair from a bivariate normal via Cholesky, mapped through each
  variable's own marginal (normal/lognormal) so the correlation does not change
  either marginal distribution. Default `None` = every variable independent
  (byte-identical to the classic Taylor series). For a c'-φ' pair both raising the
  FOS, a NEGATIVE correlation reduces the FOS variance. Adapter-exposed
  (`correlations`). Validated against Slide2 #34's published Table 34.1
  coefficients — see RESULTS/INVENTORY V-045 and
  `validation_examples/test_published_v045_slope_correlated.py`. (At very high COV,
  φ' COV≈1.2 in #34, FOSM's linear Taylor series diverges from MC by ~8% — a
  documented linearization limit, not a defect; the machinery matches MC to <1% at
  moderate COV.)
- **Anisotropic undrained strength su(α) (v5.4 F6)**: `SlopeSoilLayer.strength_model
  = 'anisotropic'` makes the undrained strength vary with the slice-base inclination
  α (the ADP active/DSS/passive anisotropy common in soft-clay practice; φ=0).
  Parameters `su_active` / `su_dss` / `su_passive`. The interpolation follows
  Casagrande & Carrillo (1944), `su(i) = su_H + (su_V − su_H)·sin²(i)` with `i` the
  MAJOR-PRINCIPAL-STRESS inclination from horizontal and the failure plane at 45° to
  σ1 (φ_u=0, so `i = α + 45°`); since `2·sin²(α+45°) − 1 = sin(2α)` this reduces to a
  clean sin(2α) form on the base angle:
    * `α ≥ +45°` → `su_active` (triaxial-compression / active); `α ≤ −45°` → `su_passive`
      (triaxial-extension / passive); `α = 0` → `su_dss` (direct simple shear);
    * `0 ≤ α < 45°`: `su_dss + (su_active − su_dss)·sin(2α)`;
    * `−45° < α < 0`: `su_dss + (su_passive − su_dss)·sin(2|α|)`.
  `_anisotropic_su(α)` (geometry.py) is called per slice from `build_slices` with the
  slice-base α threaded through `strength_at(…, alpha=α)`. **α SIGN CONVENTION (the #1
  error source — α is the failure-PLANE angle, not the σ1 inclination; they differ by
  ~45°):** positive α = base dipping toward +x = the ACTIVE (driving/crest) side for
  the module's standard slope (toe at low x, crest at high x — every validation
  geometry). A mirrored slope must swap su_active/su_passive. With equal su's the
  model reduces to `mohr_coulomb` cu EXACTLY (isotropic identity). Adapter-exposed
  (`strength_model='anisotropic'` + `su_active`/`su_dss`/`su_passive`). Validated —
  RESULTS/INVENTORY V-054 and `test_published_v054_anisotropic_su.py`: isotropic
  anchor FOS 2.29 vs published ~2.16–2.2 (Taylor/FLAC, H=5/45°/su=40), 2.237 with level
  ground beyond the toe, where the model starts (GM-63); the Bakklandet
  ADP ratios (DSS/A=0.63, P/A=0.35) drop the FOS ~21% (critical-vs-critical),
  bracketing the published FEM 24% (1.43→1.09) — CONVENTION (FEM vs LEM). Refs: Lo
  (1965), Su & Liao (1999), Al-Karni & Al-Shamrani (2000) — anisotropy matters mainly
  for slopes flatter than ~53°.

## Duncan Verification Examples (test_duncan_verification.py)

| Example | Description | Published FOS | Method | Tolerance |
|---------|-------------|--------------|--------|-----------|
| 1 | Saturated clay, undrained (cu=600psf, phi=0) | ~1.0 Fellenius, ~1.08 Bishop | Fellenius, Bishop | ±10% |
| 2 | Cohesionless slope (c=0, phi=40, 2H:1V) | ~1.17 | Bishop, Spencer | ±15% |
| 3 | Cohesionless with seismic (kh=0.15) | < Ex 2 | Spencer | directional |
| 4 | Two-layer (sand over clay) | ~1.4 Bishop, ~1.3 Fellenius | Bishop, Fellenius | ±20% |
| 6 | Submerged slope with GWT | varies | Bishop | ±15% |

All dimensions converted from imperial (ft/psf/pcf) to SI (m/kPa/kN/m³).
Example 5 skipped (curved Mohr-Coulomb envelope not supported).

**Ex 1 against Taylor's chart (GM-63, 2026-10-15).** For phi = 0 the
critical circle passes below the toe, tangent to the firm base (FHWA
NHI-06-088 Figs. 6-14 and 6-15, after Taylor 1948). su/(gamma H) = 0.1200
gives 0.970 with the base at the toe (D = 1), 0.883 with it 1.2 m below,
0.706 at D = 2 (this test's model) and 0.679 at D = 3; Taylor's D = 2
circle enters 1.76 H behind the crest and comes out 1.19 H beyond the toe
(nH), the D = 3 circle 3.18 H and 2.69 H. Bishop and Spencer score those
circles 0.706 / 0.678. The test's profile ends 1 H beyond the crest and
the toe, so no circle the search may try reaches them: 0.727 at D = 2 and
at D = 3 (0.719 on a 20x20 grid, 0.722 entry-exit), where fem2d's SRM,
whose rollers stood 1.6 H out, read 0.648 (fem2d VALIDATION.md §11; 0.660
since GM-88, whose default model carries the level ground to twice the
depth of clay below the toe each side). On a
profile with 4 H of level ground each side and the centre grid over the
slope, the search gives 0.705 (D = 2) and 0.676 (D = 3).

Since then a governing circle that ends within a tenth of the model width
of an end of the profile is checked against circles about nine nearby
centres on the ground carried on level past the ends
(`search_supplements._profile_end_warning`; Bishop scan, then the
method's own FOS; a rapid-drawdown search scores its own centre only, with
its own FOS); when one that crosses an end is at least 1 % lower the
search warns and gives its FOS (Duncan Ex 1's profile: 0.709 at D = 2,
0.681 at D = 3, against 0.727). The steep phi = 0 toe circle of GM-13 /
V-054 (H 5 m at 45 deg, clay to 10 m below the toe, the profile starting
at the toe) gets it too: 2.237 against 2.286, toward the published 2.16-2.2
(FLAC, Taylor).

**GM-87 (2026-10-15).** The default centre grid spans the whole profile,
so on a long profile it was coarse (D = 2, 4 H each side: 0.739 against
0.705). Its lowest centres are now refined (see "Local refinement of the
centre grid" above): the default search gives 0.703 at D = 2 (4 H and 5 H
each side) and 0.676 at D = 3 (4 H, 5 H, 6 H), Taylor's circle; on this
class's own short profile 0.726, still with the profile-end warning. A
noncircular governing surface that ends near an end of the profile is now
checked too: circles about the circle through its two ends and its lowest
point, scored with the method that scored it (Spencer when Bishop was asked
for). On the V-054 model a random noncircular search enters at the toe at
3.863 and the check gives 2.215 past the left end. A noncircular mechanism
that would itself carry on past the end (along a weak layer, say) is still
not checked.
