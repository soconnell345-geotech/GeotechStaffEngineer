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
- `geometry.py` — SlopeSoilLayer (flat or polyline top and bottom), SlopeGeometry (polyline surface/GWT); `column_at`, the one lookup of the layer at a point (GM-140 step 1)
- `slip_surface.py` — CircularSlipSurface (center + radius), PolylineSlipSurface (polyline points); a surface below the ground over more than one stretch is refused (`GroundCrossingError`, GM-122)
- `slices.py` — Slice dataclass, build_slices() with multi-layer weight/pore pressure (the weight summed over `SlopeGeometry.column_at`)
- `methods.py` — fellenius_fos(), bishop_fos(), spencer_fos() / morgenstern_price_fos() (wrappers of the rigorous `gle.gle_fos`); each returns the method's own solution or raises `NoSolution` (GM-140 step 2: the `*_legacy` approximate forms and `solve_rigorous()` are removed)
- `solution.py` — `solve_method()`, the one solver every caller uses for a method (a result, every search family, the yield acceleration, FOSM, Monte Carlo): a `MethodSolution` with the method's own FOS, or None and the reason; `NoSolution`; `check_method()` / `check_interslice()` (an unknown name is refused)
- `gle.py` — the rigorous general limit equilibrium (Fredlund & Krahn 1977; supports by the slice-by-slice march of Zhu et al. 2005, GM-60) and janbu_fos()
- `reinforcement.py` — nails, anchors, geosynthetics and stabilizing piles as forces at the base of the slice each crosses, in every method (GM-53, GM-59)
- `search.py` — optimize_radius(), grid_search() (with the local refinement of its lowest centres, GM-87), search_noncircular(), entry/exit filtering; a rigorous noncircular solution must hold about a second axis and slicing (`_solution_fails`, GM-57 review); the search's time budget and the Bishop screen (`search_budget`, `bishop_screen`, GM-123)
- `acceptability.py` — the printed acceptability checks on a solved surface (GM-140 step 3): `surface_findings()` gives each finding with its check, source, slices or slice sides and values (m_alpha below 0.2, Bishop against the Ordinary Method of Slices, a negative frictional term, tension on bases and sides, the line of thrust outside the mass); a flagged surface is kept, in a result and in a search
- `search_supplements.py` — toe-circle, circle-passing-above-the-toe and infinite-slope (shallow planar) families scored after every `search_critical_surface` (GM-13, GM-14, GM-122)
- `search_ranges.py` — the automatic entry / exit windows and centre ranges, set from the slope's toe, crest, height and depth to the base (GM-57)
- `analysis.py` — analyze_slope(), search_critical_surface() orchestrators
- `results.py` — SliceData, SlopeStabilityResult, SearchResult
- `nails.py` — SoilNail, NailContribution (nails reach the FOS through `reinforcement.py`)
- `tests/` — 614 tests in 30 files (collected 2026-10-17), among them:
- `tests/test_ground_crossings.py` — 21 tests (one sliding mass per trial surface, the touch tolerance at its threshold, the refusals counted, the circles passing above the toe, GM-122; entry / exit windows given the wrong way round, GM-124)
- `tests/test_slope_stability.py` — 92 tests (core + entry/exit limits + noncircular search; Taylor's phi = 0 chart, GM-57)
- `tests/test_search_budget.py` — 16 tests (a search stopped by its time budget reports the lowest surface in what it covered, what that was and the projected run time; a larger budget covers it all; a finished search is the unbudgeted one; the scan order finds a narrow valley early; differential evolution cut short; the Bishop screen finds the same surface, says when it was not applied or its cap left points out, GM-123)
- `tests/test_search_ranges.py` — 23 tests (the automatic ranges; a directed search and what the result says it covered; the same slope drawn narrow and wide, GM-57)
- `tests/test_duncan_verification.py` — 28 tests (Duncan §7.6-7.7 verification examples; Ex 1 against Taylor's chart, GM-63, and the default search on long profiles, GM-87)
- `tests/test_nails.py` — 38 tests + 16 skipped (nail intersection/pullout/capacity; the skipped ones test the old nail path, replaced by `reinforcement.py`)
- `tests/test_reinforcement.py`, `tests/test_search_reinforcement.py` — 22 + 33 tests (supports in each method; supports in searches, GM-53, GM-54, GM-59, GM-60)

## Layers: one material per point (GM-140 step 1, 2026-11-13)

Print: a slice that holds two soils is weighed in two portions, each
soil's part of the slice at its own unit weight (EM 1110-2-1902 Appendix F,
F-5d, p. F-19: slices containing both embankment and foundation soil "were
divided into two portions for calculating the slice weights"; worked in
Fig. F-11, p. F-21), with total unit weights for all soils (2-2d, p. 2-5);
and the engineer checks "the weights of slices, shear strength properties,
and pore water pressures at the bottoms of slices" (EM 4-2d(2), p. 4-2).
2-2d and 4-2d(2) are as quoted in
`module_work/slope_integral_review/ACCOUNT.md` section 2 (the EM is not in
the repository).

- **The rule.** At a point (x, z) below the ground, the layer is the first
  one listed in `soil_layers` whose bottom and top at x contain z.
  `SlopeGeometry.column_at(x, z_low, z_high)` gives a column as stretches,
  each with its one layer; `layer_at_point(x, z)` is the same rule at a
  point. The slice weight (`slices._compute_slice_weight`) is summed over
  the column; the strength, ru, the drawdown envelope and the supports'
  layer come from `layer_at_point` at the base; SHANSEP's sigma'v and the
  seismic force come from that weight. The section drawings
  (`plotting._draw_section_base`, `calc_package.interactive`) fill each
  layer where it holds the ground (`layer_bands`).
- **What an input means.** A layer's `top_elevation` / `bottom_elevation`,
  or `top_boundary_points` / `bottom_boundary_points` (polylines sorted by
  x, level beyond their ends), say where the layer MAY be. A polyline
  bottom on one layer and a flat top on the next, at the boundary's highest
  point, therefore means the lower layer starts at the boundary. Giving the
  lower layer the same polyline as `top_boundary_points` says the same
  with no overlap.
- **Overlaps.** Where two layers cover the same ground the first listed
  holds it. `SlopeGeometry.layer_overlaps()` lists each layer that loses
  ground (the layers that hold it, the x range, the tallest shared stretch
  below the ground). The results of `analyze_slope`,
  `search_critical_surface`, `compare_methods_table` and
  `rapid_drawdown_fos` carry it (`layer_overlaps`, in `to_dict()` when not
  empty; a LAYERS line in the summaries of the first three), and the calc
  package prints it as a table. The FOSM and Monte Carlo results, the
  yield acceleration, the result of a search function called directly
  (`grid_search`, `search_weak_layer_biased` and the others in `search.py`)
  and the top level of the drawdown search's result do not carry it: read
  it from the geometry there.
- **Lenses.** A layer listed before the layer it sits in holds its own
  ground and the host is weighed around it (two stretches in the column).
  No closed-region input was added.
- **Gaps.** Ground between the surface and the bottom of the deepest layer
  that no layer covers is refused when the `SlopeGeometry` is built, with
  the elevations and the x range. That includes ground above the first
  layer's top. Below the bottom of the deepest layer is the model's base,
  as before (GM-128).
- **How it is found.** `SlopeGeometry._scan_layers`: the ground and every
  top and bottom are straight between their vertices, so gaps and overlaps
  open and close only where two of those lines cross; the column is checked
  at the ends and middle of every stretch between such points. A stretch
  shorter than 1e-6 m is neither.
- **Before.** The weight summed every layer that overlapped the slice
  height, each from its own flat top (a top could not slope), while the
  strength used the rule above. With the usual layout the wedge between a
  sloping boundary and the lower layer's flat top was weighed twice: V-028
  weighed 8,768 kN/m against 3,003 from its area. Models whose layers do
  not overlap are unchanged, slice for slice (VALIDATION.md B19).
- **Not done here.** A layer boundary's vertices are not slice boundaries
  (only its crossings with the slip surface are), so a slice with a kink
  of the boundary inside it takes its weight from the mid-point column;
  the su-with-depth law still reads a layer's flat `top_elevation` /
  `bottom_elevation` (the weak-layer search aims at the ground a layer
  holds at each x since the fix round, `search._layer_band_at`; its lower
  limit is still the lowest flat `bottom_elevation`); `ru` still uses the base
  layer's unit weight over the column (the account's D3). See
  `module_work/gm140/STEP1_CHECK.md`.

## Tension cracks and reverse-curvature circles as printed (GM-140 step 4, 2026-11-13)

The rule lives in `cracks.py`, one for every method, surface family and
caller. Print: USACE EM 1110-2-1902 (31 Oct 2003) 2-2f, pp. 2-6 to 2-9, and
C-10c, p. C-36; Slide v6 Tutorial 16, p. 16-1.

**What a crack does** (`tension_crack_model`). `'truncation'` (default): the
slip surface ends where it reaches the bottom of the crack, the soil upslope
of the crack is left out, and the slices end at a vertical boundary there
(EM 2-2f(3): "terminating the slices near the crest of the slope with a
slice having a vertical boundary"; C-10c; Slide: "A tension crack
essentially terminates the slip surface"). The mass that is kept is cut into
the `n_slices` asked for. `'strength'` keeps the cracked soil as driving
weight without strength: the default before this step, with no printed
source, kept as a stated option. Either way the crack's water acts on the
vertical boundary: 0.5 gamma_w z_w^2 at z_w / 3 above the crack's bottom; a
`tension_crack_water_depth` above the crack's own depth fills it.

**Where it is** (`tension_crack_side`). `'crest'` (default): the end of the
slip surface where the ground is higher (EM 2-2f(3): near the crest),
whichever way the slope faces (`cracks.crest_side`; ends at one level:
low x). `'entry'` / `'exit'` give the end. The crack's bottom is the ground
at that end of the surface less the depth. The crack stands where the
surface reaches that bottom, which is where the slices end; the result
reports it there (`tension_crack['x_m']`, the ground `ground_z_m` and the
bottom `base_z_m` at that x) and the plot draws it there.

**A surface wholly within the cracked depth.** A surface that nowhere passes
below the crack's depth, measured below the ground above it
(`cracks.depth_below_ground`), has no vertical boundary at the crack's
bottom (EM 2-2f(3); C-10c): it is no slip surface. A search refuses and
counts it (`crack_refused['within_cracked_depth']`); given to
`analyze_slope` it raises `cracks.CrackRuleRefused` saying so. Without this,
a search with a deep given crack settled on slivers on the face held wholly
within the crack (ACADS 1(b) with its 3.81 m water-filled crack on Slide2's
own centre grid: F 0.052 on a 0.6 m sliver; so did the module before
step 4, 0.050 with the old default and 0.052 with the surface ended at the
crack). A Rankine depth that would reach below the deepest point of the
surface is solved again from F with no crack; the scan stops at that depth.

**How deep** (`tension_crack_depth_basis`):

| Basis | Depth | Print |
|---|---|---|
| `'given'` (default) | `tension_crack_depth`; 0 = no crack | the caller |
| `'rankine'` | d = 2 c_D / (gamma tan(45 - phi_D / 2)), c_D = c / F, phi_D = arctan(tan(phi) / F) | Eq. C-36, p. C-36 ("for total stresses and homogeneous soil"); Eqs. 2-5, 2-6, p. 2-9 (Eq. 2-4 is the same depth) |
| `'min_fos'` | the depth of minimum factor of safety from a scan | 2-2f(4), p. 2-6; C-10c(1), p. C-36 |

* **Rankine.** F is the factor of safety of the surface itself, by the method
  in use: the surface is solved, the depth recomputed, and so on from F = 1
  until the depth changes by less than 1 mm (`DEPTH_TOL`; at most 25
  solutions). c, phi and gamma are those of the soil at the ground at the
  crack. A depth past that soil's bottom stops there (C-10c: the crack is as
  deep as the embankment and does not extend into the foundation), and the
  result says so. c = 0 gives no crack. Module's readings, each stated in the
  result: a drained soil with pore water pressure inside the cracked depth
  (the water table there, ru, or a pressure grid) is scanned instead, because
  2-2f(6), p. 2-9, prints no equation for effective stresses; so is a SHANSEP
  or Hoek-Brown soil (no c and phi); an `'anisotropic'` soil takes its active
  strength with phi = 0 (Rankine's active pressures).
* **Scan.** 21 depths in equal steps from 0 to the lowest point of the
  surface's upper half, shallowest first; the first minimum met is taken
  (past the depth that removes the tension a deeper crack removes
  compression, and F rises: 2-2f(4), "tensile but not compressive stresses
  are eliminated"). A factor of safety still falling at the last depth has
  no minimum and the surface gets no number. On the made-up slope of
  `test_cracks_as_printed.py` the scan's depth is 3.10 m and Rankine's
  2.85 m, F within 0.001: the manual's two ways agree.

**Reverse-curvature circles** (`reverse_curvature`; the owner's call
OWN-103, as the orchestrator refined it on the step 4 review's F1). A circle
whose centre is below the ground at its upper end would turn back over
itself above its vertical tangent, at the centre's level. The tension crack
is a condition of the analysis, set by the engineer and the same on every
trial surface; a circle's shape never switches it on. `'tension_crack'`
(default): with a crack asked for (`tension_crack_depth`, or
`tension_crack_depth_basis` `'rankine'` or `'min_fos'`), such a circle is
closed by that crack, its surface ending at the crack's bottom like any
other, where its overhang (the centre's depth below the ground) lies within
the crack's depth; a circle overhanging deeper is discarded (a crack is
justified only to the depth the soil can be in tension, C-10c). With no
crack asked for, every such circle is discarded: it is not an admissible
surface, and naming a crack admits those within its depth. `'discard'`
discards every such circle. A circle overhanging at its lower end is always
discarded: a crack is at the upper end (EM 2-2f(3)) and no print closes the
other. A search counts the surfaces the crack rule refuses by reason and
locates them (`SearchResult.crack_refused`: `reverse_curvature_no_crack`,
`reverse_curvature_deeper_than_crack`, `reverse_curvature_no_crack_at_end`,
`reverse_curvature_discard_asked`, `within_cracked_depth`,
`scan_no_minimum`; `n_discarded_reverse_curvature` sums the four
reverse-curvature reasons), and its `choices` name each; a discarded circle
given to `analyze_slope`, `rapid_drawdown_fos` or `yield_acceleration`
raises `cracks.ReverseCurvatureDiscarded` (a `cracks.CrackRuleRefused`) with
the reason and the depths. The crack from the centre's level to the ground,
which no printed source gives, is removed. (As first built, step 4 closed a
reverse-curvature circle with Rankine's crack whenever no crack was given,
so a circle's shape switched the crack on: the search's minimum then jumped
where the centre crossed the crest's level, and Slide2 #85 read 1.361 on such
a circle. The review's F1 removed that; `reverse_curvature_crack_basis` is
gone.)

**How it is wired.** `cracks.plan_crack` says what a surface's crack needs;
`cracks.solve_with_crack` hands the caller's solver copies of the geometry
with the crack fixed (`fixed_geometry`), repeating as the basis requires.
`slices.build_slices` reads the crack from there (`crack_for_slices`) and
raises `cracks.CrackDepthNeeded` for a geometry whose crack still depends on
the factor of safety, so nothing slices such a surface on a silent basis.
The callers: `analyze_slope` (and `compare_methods`), the search scorer
`search._compute_fos` (every family: grid and entry/exit circles, circles
through and above the toe, planes, random, differential-evolution,
particle-swarm and weak-layer polylines; a `fos_fn` such as the drawdown
search's gets the fixed geometry), `rapid_drawdown_fos` (one crack for every
stage, at the drawdown F), `yield_acceleration` (Rankine's depth at F = 1,
the state it solves for; a scanned depth from the surface without kh), and
through `_compute_fos` the FOSM and Monte Carlo runs (each run its own
crack), so also `reliability/wrappers.py`, `geo_project`'s builder, the
agent's adapter and the calc packages. A result carries the crack it was
solved with (`tension_crack`, the depth used, one warning line); a search's
`choices` say the crack's model, side and depth basis and what was done with
reverse-curvature circles.

**Measured** (VALIDATION.md B21): ACADS 1(b) 1.603 / 1.599 / 1.600 on
Slide2's circle and, searched on Slide2's own grid, Bishop 1.590 / Spencer
1.585, against the printed 1.596 / 1.592 / 1.592; V-031 with its dry crack
1.417 / 1.422 printed; V-043's clay crest 0.964 against 0.975. With no crack
asked for, the searches that read a reverse-curvature circle as first built
read what they read before step 4, the overhanging circles discarded:
Taylor's 45 deg slope 1.942 against 1.947, V-054's steep clay slope 2.346
(printed 2.16 for the slope, whose profile ends at the toe), Slide2 #85 with
its support Bishop 1.508 / passive 1.316 against 1.531 / 1.324 (D&W 1.51 /
1.32). The vertical cut reads 0.935 against Taylor's 0.851, a strict xfail
with the reason (before step 4, 0.842 on a circle closed at its centre's
level; with Rankine's crack asked for, 0.605).

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
- Sliced at the ground's breaks, its layer, water-table and crack-base
  crossings and the surcharge edges, each slice's base its chord (GM-134;
  "Where a circle is sliced" below)
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
- Sliced at its corners, the ground's breaks and its layer and water-table
  crossings, each slice's base angle and length from its chord (GM-131; "Where
  a noncircular surface is sliced" below), and at surcharge and crack
  edges (GM-134). A circle gets the same boundaries less the corners.
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
- **Rigorous methods.** Spencer / Morgenstern-Price / GLE pick each centre's circle
  with Bishop (its radius optimisation some 30 times cheaper, as the toe-circle family
  does) and score that circle with the method itself, so the walk moves only where the
  method's own FOS falls (since GM-138; before, it moved on Bishop's FOS, which on
  V-043 led it away from Spencer's critical circle: "The refinement walk moves on the
  method's own FOS" below); at each walk's end the radius is optimised again with the
  method itself; the method's own FOS decides. Measured on the walk before GM-138
  (moving on Bishop's FOS) against a walk with Spencer itself on nine models: the
  same FOS to four decimals on eight, at +2 to +23 % run time instead of +40 to
  +225 %. Not where supports are modelled: Bishop counts a support only
  through its moment about the centre (GM-59) and the two methods' critical circles
  part (Slide2 #85, anchored: the Bishop walk found nothing below Spencer's grid
  minimum 1.581, Spencer's own walk 1.574), so a reinforced model walks with the
  method itself.
- **On the base.** Until GM-128 `build_slices` dropped up to max(2, 3 % of n_slices)
  below-base END slices as tangential slivers (SS-5). Where the toe sits on a rigid
  base (Slide2 #85 without its support) a circle whose lowest point is a few
  millimetres under the base just beyond the toe shed its two toe slices and scored
  low: 0.867 on a circle refused at 90 slices, against the base-tangent toe circle's
  0.883. The walk found such circles, so it keeps to circles that at most touch the
  base (`search._dips_below_base`), as every search family does since GM-128 ("A
  surface touching the model's base"). The grid itself is unchanged.
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
  0.4 ft long at 0.9420, the closed form 0.9419, against 0.9430 for the
  planar surfaces with Janbu's f0 and their end ramps). It governs, and the result says it is the
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
All three states came from even slices that took one segment's angle across a
corner: sliced at its corners (GM-131), none has such a state (two have no
solution, V-043's solves at 1.715 at any count), and the check stays as the
guard (`test_a_state_at_one_slicing_only_is_refused`).

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
reported a factor of safety that was too HIGH. `SearchResult.families_scored`
lists the families that gave an admissible surface and
`families_not_scored` why each other family run gave none (none held, the
time budget, the windows given, no c' = 0 face); the "surface families
scored" entry of `choices` says both, and points at `governing_location`
and `refused_lowest` only where the result has them (GM-125).

- **Why.** Until GM-57 the automatic windows came from the MODEL extent,
  not the slope: entry in the first 40 % of the width, exit in the last
  40 %, circle centres from 1 unit above the crest. A slope that does not
  sit in the middle of its model was under-sampled. Steep phi = 0 slope, H = 5 m at
  45 deg, su = 40, model starting at the toe (V-054): the critical toe
  circle exits the crest at x = 7.1, inside the automatic exit window's
  lower bound (15), and its centre (2.5, 4.0) is below the crest, under the
  grid. Grid 2.78; true 2.287 (with the ranges set from the slope, GM-57,
  the grid reads 2.344: the toe circle's centre is still below it; since
  GM-134 the toe circle the search finds reads 2.346 and the grid 2.348,
  the 2.287 having been a slicing error at the arc's vertical end). And
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
  are scored by `_compute_fos`. Each is a twentieth of a slice width deep,
  with end ramps 1.5 times the depth long; each ramp is a slice of its own
  (GM-131) and every other slice base is parallel to the face. The ramps
  move the FOS off the closed form by about their share of the slide's
  weight, 0.75 / (20 n) at each end. (Before GM-131 sliced a polyline at
  its corners, the surfaces were a tenth of a slice width deep with ramps
  ending before the first and last slices' midpoints, since an even slice
  across a ramp's corner swung the FOS 5 % either side of the closed form;
  at a fortieth, the stage-1 solve of a face under a full pool drifts up.)
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
  with ru 0.2, with kh 0.1-0.15, both face directions: +0.11 % to +0.15 %
  on 30 slices (Janbu's f0 and the end ramps; VALIDATION.md B12). The
  closed-form dry value is
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
  bottom of the deepest layer, as every search's circle (GM-128). Where the
  toe sits on a rigid base, a toe circle dipping under it pivots about the
  toe into the base: until GM-128 it shed its toe slices and scored low
  (Slide2 #85 without support: 0.83, against 0.88 for the base-tangent toe
  circle).
- **Rapid drawdown (GM-54).** `search_rapid_drawdown` runs both families
  with its own per-surface drawdown FOS (`fos_fn`), honouring the caller's
  ranges, and its `search` carries `governing_surface` / `candidates` /
  `warnings`. With a per-surface solve this costly there is no Bishop
  screen; the toe scan is coarser instead (7 centre offsets, 6 golden steps
  on the far end). The thin planar surfaces converge in the drawdown GLE: on
  a free-draining c' = 0 shell drawn down from el 20 to 5 the planar FOS is
  within 0.4 % of tan 37 / 0.8 = 0.942, the closed form. Under a full pool
  they solve stage 1 since GM-131 (none did before): on the Slide2 #95/#96
  dam the Corps 2-stage planar slide reads 0.783 against the infinite
  slope's (gamma' / gamma) tan(phi') / tan(beta) = 0.772. On a face only
  partly under the stage-1 pool the ramp slices can carry Spencer's stage 3
  off the force balance (an upper third of the Pilarcitos face: 1.21
  against 1.62 by force equilibrium alone); there it does not govern and
  errs low.
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
    Bishop search gives 1.372 by default (1.304 before GM-134).
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

**Reverse-curvature circles** (`SlopeGeometry.reverse_curvature`): see "Tension cracks and reverse-curvature circles as printed" above (GM-140 step 4). Before it such a circle was closed by a dry crack from its centre's level to the ground, as Slide2's default is quoted to do; the numbers in the table below marked "default" are from then.

**Bishop / OMS critical circle with supports at a short arm.** When the
supports crossing a Bishop or OMS critical circle act with an average arm
under half the radius (under half their force counted through the moment),
the search warns, gives the share and what Spencer finds on the same circle
(`search_supplements._support_arm_warning`).

**Slide2 #85** (D&W 2005 Fig 6.34; c = 350 psf, phi = 0, 98 pcf, 20 ft face
at 2V:1H on a rigid base, one 9,000 lb/ft support at mid-height):

| | Module | Slide2 (circular) | D&W |
|---|---:|---:|---:|
| Active, on Slide2's printed GLE circle (15.446, 37.624, R 27.594): Bishop / Spencer / GLE (1.564 each before GM-134) | 1.566 / 1.565 / 1.566 | GLE 1.575 | 1.51 |
| Active, Bishop search, default (reverse-curvature toe circle, 5.9 ft crack, arm 4.1 ft on R 14.1 ft; 1.304 before GM-134) | 1.372 + warnings | 1.531 | 1.51 |
| Active, Bishop search, `reverse_curvature='discard'` (1.503 before GM-122, 1.467 before GM-127, 1.432 before GM-134) | 1.508 | 1.531 | 1.51 |
| Active, Bishop search, `'discard'`, `m_alpha_rule='every_slice'` (1.467 before GM-134) | 1.511 | 1.531 | 1.51 |
| Active, Spencer search (1.542 before GM-134) | 1.551 | 1.884 | |
| Active, GLE search (1.584 before GM-134) | 1.573 | 1.575 | |
| Passive, Bishop on Slide2's active circle | 1.330 | | 1.32 |
| Passive, Bishop search, default / `'discard'` / `'discard'` with `'every_slice'` (1.208 / 1.274 / 1.296 before GM-134) | 1.254 / 1.316 / 1.316 | 1.324 | 1.32 |
| Passive, Spencer search | 1.346 | 1.872 | |
| Passive, GLE search | 1.379 | 1.378 | |

The default Bishop search finds a lower F than Slide2 because it searches
reverse-curvature toe circles (centre 6 ft below the crest), which Slide2
would close the same way but its grid, placed above the slope, never
generates; on that circle the support passes 4.1 ft below the centre and
Spencer finds no lambda in -1.5 to 1.5 at which moment and force equilibrium
give the same F. Its last slice has m_alpha = cos(a) 0.18 on 30 slices
(phi = 0), which Slide2 refuses with its 'Check malpha < 0.2' setting on;
the module does not check phi = 0 slices by default, since m_alpha does not
change the FOS there (GM-127). With such circles discarded and every slice
checked (`m_alpha_rule='every_slice'`), as Slide2 does, the module
reproduces D&W and Slide2: 1.467 / 1.296 at 30 slices, 1.508 / 1.318 at
240. With them discarded and the default rule, the circles passing above
the toe keep one centred at the crest's level whose last slice has m_alpha
0.195 / 0.191: 1.432 / 1.274 at 30 slices, 1.484 / 1.307 at 240 (-1.7 % /
-1.0 % on D&W). Slide2's Spencer (1.884 /
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

### A method's own solution, or no number (GM-140 step 2, 2026-11-13)
**The rule.** A factor of safety reported under a method's name is that
method's own converged solution on that surface. Where the method has no
solution there is no number: `FOS` is None, `method_solved` is False and the
result says why. One rule for every method (Ordinary Method of Slices,
Bishop, Janbu, Spencer, Morgenstern-Price, GLE), every family of surfaces
and every caller.

**Print.** UFC 3-220-10 p. 362: "Critical failure surfaces can be missed as
a result of non-convergence. The engineer should be able to identify when
convergence issues are present." EM 1110-2-1902 C-10d (pp. C-37 to C-38)
lists the ways to avoid non-convergence (reject the unrealistic one of
multiple solutions, a sensible starting F, a realistic initial surface).
The methods' own equations: Bishop (1955) Eq. 13, p. 10; Spencer (1967)
Eqs. 7 and 8, p. 15; Morgenstern and Price (1965) Eq. 18, p. 85.

**What it replaces** (the account's D23, D27, D30 and parts of D49; GM-81's
rest, GM-139). It supersedes the parts of the GM-81 section below that keep
the approximate solver's value.

- **One solver** (`solution.solve_method`). `analyze_slope`, the search
  scorer `search._compute_fos` (every family), `newmark.yield_acceleration`,
  `probabilistic.fosm_fos` / `monte_carlo_fos` (so
  `reliability.wrappers.slope_reliability`) and the calc packages all take a
  method's value from it. A search's score of a surface is the
  single-surface FOS of that surface, to the last digit.
- **No approximate solver.** `spencer_fos_legacy` and
  `morgenstern_price_fos_legacy` are deleted (no consumer outside the
  module). `spencer_fos`, `morgenstern_price_fos`, `bishop_fos`,
  `fellenius_fos` and `janbu_fos` raise `NoSolution` (a ValueError, with
  the reason) where they have no solution.
- **Bishop and Janbu.** Bishop's iteration not within its tolerance after
  50 iterations has no solution (it returned its last iterate with a Python
  warning). Janbu's force-equilibrium iteration with no state has none.
- **No driving.** A surface with no net driving moment or force has no
  factor of safety. The solvers' internal 999.9 never leaves them as one;
  `solve_method` also refuses a ratio at or above it (the driving sum zero
  to rounding). 999.9 remains only as the searches' internal "not scored"
  score (`search._FOS_MAX`) and inside `gle.GLEResult` beside
  `converged=False`.
- **The lambda search** (`gle.gle_fos`). GM-60's robust search (the scan
  started from the nearest lambda with a state when lambda = 0 has none; a
  lambda with no state stepped over; a bracket whose bisection does not
  close passed over) ran only with supports. It is the one search for every
  run. At a fixed lambda, an iteration on {N, E, X, F} that runs its 80
  iterations without settling is solved by the direct march
  (`_solve_marched`), and where that finds nothing there is no state: the
  last iterate was returned as a state before, flagged by nobody.
  *Not adopted:* the direct march as the fallback wherever the fixed-point
  step cannot be taken (as with supports). Measured, it solves about half
  of the otherwise unsolved random polylines of a noncircular search at 14
  to 16 times the run time (111 s against 8 s; 297 s against 18 s) with the
  same minimum in both cases tried. The step 2 review asked for it (D1);
  tried in the fix round and not applied: it finds the Spencer solution of
  a phi = 0 circle the module misses (2.9458; the "independent" 2.9435
  was wrong: the moment ratio is 3.08, GM-140 step 3's fix round), but it
  also gives "converged" states that rest on the m_alpha
  clamp, with base tension, on two surfaces GM-131 found to have no
  Spencer solution (1.56 at every slicing on w11; 1.19 at 30 and 41
  slices only on the deep sandstone polyline), takes away the yield
  coefficient of an ordinary slope (0.327 without it; with it Spencer has
  no solution at a trial kh of 1.0 and no ky is given) and makes ACADS 4's
  weak-layer search take 84.5 s against 22 s. A call for the owner; see
  `module_work/gm140/STEP2_CHECK.md`.
- **One moment axis.** On a polyline the GLE takes moments about its fitted
  axis (`gle._fit_axis_point`) in a search and on one surface alike. The
  search's second try about a fixed axis (`_noncircular_axis_point`) is
  gone from scoring: with the lambda search above it rescued no surface in
  four searches (about 2,100 trial solves), and it was how ACADS 4's weak-layer search
  returned Spencer 0.515 on a surface `analyze_slope` had no solution on.
  The fixed axes remain in the confirmation check of a would-be critical
  surface (`_solution_fails`), which is a screen, not a score.
- **Searches count.** `SearchResult.no_solution` lists the trials left
  without a solution by family and by method (`n_no_solution` the total),
  in `to_dict()` and `summary()`; `warnings` says so when the governing
  family lost trials this way. Measured on Slide2 #85 without its support,
  6 x 6 Spencer search: 267 circles went to the approximate solver before;
  329 trials have no Spencer solution now (the search no longer stops its
  families early on the approximate values) and the reported minimum is
  1.047 on a toe circle where Bishop's simplified method gives 0.910: on a
  steep phi = 0 slope Spencer's method has no solution on the circles that
  govern (tension at the top; the force factor of safety approaches the
  moment one from above and does not meet it for lambda up to 1.5), and the
  result says a lower surface cannot be ruled out.
- **Yield acceleration.** Every trial kh is the method's own solution. The
  upper end of the kh bracket is halved until the method solves there.
  Where a trial kh the bisection needs has no solution, `ky` is None with
  the reason, the kh, and the bounds reached.
- **FOSM and Monte Carlo.** FOSM with no solution at the mean values or at
  a value one standard deviation off gives no result (a ValueError naming
  the variable and the value). Monte Carlo counts realizations with no
  solution (`n_no_solution`), leaves them out of the statistics and bounds
  the probability of failure both ways (`pf_lower`, `pf_upper`). No printed
  rule was found for either; the result says the rule is the module's own.
  They solve the one given surface as `analyze_slope` does (before, through
  the search scorer and its screens).
- **Unknown names.** An unknown `method` is refused with the list of names
  in every entry point (it ran Bishop's simplified method); an unknown
  `f_interslice` likewise.
- **Not changed** (other steps): which method scores which family, and a
  polyline named for Bishop running Spencer (step 6); the m_alpha rule and
  the acceptability checks (step 3); the rapid-drawdown stages and their
  fall-back to Ordinary Method of Slices estimates (step 9;
  `rapid_drawdown.py` is untouched, and its search does not yet count its
  per-surface solves without a solution); the GLE's lambda range, tolerance
  and m_alpha clamp (D29).

### The printed acceptability checks (GM-140 step 3, 2026-11-13)

**The rule.** Once a method has solved a surface (step 2) and the surface is
one sliding mass (GM-122), every further test is a printed acceptability
check whose finding is reported with the surface: which check, which slice,
what value. A search keeps a flagged surface by default and says so;
refusing on a check is a stated option. Nothing is left out silently, and
circles and polylines meet the same checks. Code: `acceptability.py`
(`surface_findings`), called by `analyze_slope`, by every search's scorer
(`search._compute_fos`, `_trial_checks`) and by `search_critical_surface` on
its governing surface (`search_supplements.critical_acceptability`).

| Check (`flags`) | Printed source | What is reported |
|---|---|---|
| `low_m_alpha` | Whitman and Bailey (1967, p. 487, Eq. 5): warn if cos(theta) + tan(phi) sin(theta) / F < 0.2 on any slice; "the user must examine the result in detail" | the slices, their m_alpha, base angle and whether the base is inclined against the sliding; EM 1110-2-1902 C-10b's four remedies (pp. C-34 to C-36) where it is, named and not applied |
| `bishop_below_oms` | Duncan (1996, p. 580): compute the Ordinary Method of Slices value for each circle when Bishop's method is used; where Bishop's is the lower, the Ordinary Method of Slices value is the better answer | both values; the Ordinary Method of Slices value is taken (`FOS`, with `method_used`), Bishop's own stays in `FOS_bishop` |
| `negative_frictional_term` | Bishop (1955) Eqs. 9 and 13, p. 10, no lower limit; Whitman and Bailey (1967, p. 487) ask for a warning | the slices where u b exceeds W (Bishop) or u l exceeds W cos(alpha) (Ordinary Method of Slices), and the rule in force |
| `base_tension`, `side_tension`, `thrust_outside` | EM 1110-2-1902 C-10a (pp. C-33 to C-34): forces examined, no tension on the sides or bottom of slices, the line of thrust within the sliding mass; UFC 3-220-10 pp. 361 to 362 | Spencer, Morgenstern-Price and GLE: the slices with a negative effective normal force, the slice sides with a tensile interslice force, and the sides where the line of thrust as solved (not clipped) is outside the mass |

**m_alpha below 0.2 (OWN-102).** A search keeps such a surface, flags it and
counts it by family (`SearchResult.trial_flags`); `low_m_alpha='refuse'`
leaves it out and counts it, the rule before this step (GM-127; the section
below describes that rule and its history). The exemption of phi' = 0
slices stays the default and is stated in the result as the module's own
reasoning, with no printed source; `m_alpha_rule='every_slice'` is the
option.

**Bishop against the Ordinary Method of Slices.** Made in the one solver
(`solution.bishop_against_oms`), so it holds for a given circle, every
search family, FOSM and Monte Carlo. It is made only on circles with no
supports and no seismic coefficient (the limit the owner added to OWN-102 on
2026-11-13: Duncan, Wright and Brandon 2014, p. 93, print no Ordinary
Method of Slices equation with known forces); otherwise the result says it
was not made and why. "Below" means below by more than the tolerance
Bishop's iteration is solved to: on a phi = 0 circle the two are the same
sum (EM C-2b, p. C-16).

**The frictional term (D21).** `methods.NEGATIVE_TERM_RULES`: `as_printed`
(default) uses W - u b and W cos(alpha) - u l as printed; `zero` sets the
term to zero where negative (the rule before this step, no printed source).
Parameter `negative_friction_term` of `analyze_slope` and
`search_critical_surface`; the rule is held in a context variable so every
solve inside the call follows it. The GLE solver's own settings (no
friction on a base in tension, the 0.05 m_alpha clamp; D29) are not
changed: `base_tension` names the slices the first applies to.

**In a search.** `trial_flags` gives, by family and method, the trials
scored, those flagged and each flag; `critical_flags` and
`critical.acceptability` what the governing surface carries; and when it
carries a flag, `lowest_unflagged` gives the lowest factor of safety among
the scored surfaces of the same method that carry none (among the trials
the search happened to score: it is not a second search). The headline stays
the lowest solved surface. Every exported search that returns a
`SearchResult` gets this from one place, `search_supplements.
report_acceptability`: `search_critical_surface` calls it, and
`search._reports_acceptability` calls it on a search called on its own
(`search_noncircular`, `search_de`, `search_pso`, `search_weak_layer_biased`,
`search_entry_exit`, `grid_search`; the step 3 review's defect 2).
`search_rapid_drawdown` scores with its own staged solve, on which the checks
are not made.

Neither number alone is safe (ACADS 1(a) by Spencer: 0.633 flagged, 1.09
with none, printed 1.00; ACADS 4's weak layer: 0.830 flagged, 1.667 with
none, printed 0.760), so a headline that carries a flag is never given alone.
The agent's tool result (the adapter) carries, as fields and in its
one-sentence `summary` (`search_supplements.headline_report`): the headline,
its flags (`critical_flags`), the lowest surface with no flag
(`lowest_unflagged`, or that none was found) and the option that leaves
flagged surfaces out (`refuse_option`). Only m_alpha has such an option
(`low_m_alpha='refuse'`); no option leaves out surfaces with the other flags,
and `refuse_option` says so. The other form of this call (the headline is the
lowest surface with no flag) is on the owner's desk (`module_work/gm140/
STEP3_CHECK.md`, "Calls made").

**The noncircular screens (D22), measured** (`module_work/gm140/
step3_screens.py`; table in `module_work/gm140/STEP3_CHECK.md`). On the
noncircular cases of the validation set and the module's two rock-cut
cases:

- agreement between moment axes (38 refused), agreement between slicings (1)
  and a FOS below 1.5 on a jagged surface (88), all measured with Spencer's
  method: every surface refused carried a flag of the printed checks. The
  first two are removed for the method's own solves, the jagged screen for
  Spencer, Morgenstern-Price and GLE (`search.REMOVED_SCREENS`); such a
  surface is scored, kept and flagged. `SearchResult.refused_lowest` is no
  longer filled.
- the jagged screen stays, counted (`screen_refusals`) and named in
  `choices` as the module's own, for Janbu's simplified method and the
  Ordinary Method of Slices (`search._jagged_screen_applies`): no check of
  C-10a covers them, so the surfaces it refuses there carry no flag. On
  ACADS 1(a), 200 random trials, it refuses 29 Janbu and 42 Ordinary Method
  of Slices surfaces no printed check flags (the step 3 review's defect 1).
- span: of the surfaces it refused that have a Spencer solution, the printed
  checks pass 25 of 48. It stays, counted (`screen_refusals`) and named in
  `choices` as the module's own, not found in print.
- slice count: it refused no trial in any measured case; kept as it was.
- the jagged gate also applies to a caller's own per-surface solve (the
  rapid-drawdown search), on which the printed checks are not made here.

Consequence to know: a random noncircular search can now end on a zig-zag
surface with a low factor of safety, flagged, with the lowest unflagged
surface beside it (ACADS 1(a), 200 trials: 0.633 flagged, 1.096 unflagged;
1.092 before; printed 1.00).

**One below-base test (D19).** The toe-circle family asks
`search._dips_below_base`, as every family does.

**The direct march (step 2's open question): not applied.** Measured again
with the rule that a marched state resting on the m_alpha clamp or on a base
in tension is no state: the reviewer's phi = 0 circle solves (2.9458) and
the GM-131 surfaces no longer do, but a GLE solve loses the state
the lambda search had, an ordinary slope loses its yield coefficient (no
Spencer solution at kh = 1.0) and two searches change. The strict xfail in
`test_own_solution_or_no_number.py` holds the case. Its target is the moment
ratio, not the 2.9435 the step 2 review gave: with phi = 0 on a circle,
Spencer's F must equal cu R (arc length) / sum(W x), x the moment arm about
the centre; the test works it out on 600 slices of the surface closed by a
dry crack at x = 41 (3.0 m, given, since GM-140 step 4 discards the
reverse-curvature circle with no crack asked for): 3.08, against Bishop's
3.07 on 30 slices.

### A single surface: warnings, and no silent fallback (GM-81, 2026-10-15)
*(Its two "what `FOS` holds when there is no solution" cases, the labelled
approximate value and `_legacy_critical_warning`, are superseded by GM-140
step 2 above: there is no number in either case.)*
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
  asked for on a noncircular surface (analysed with Spencer, as before);
  m_alpha below 0.2 on a slice with phi' > 0, or on any slice with
  `m_alpha_rule='every_slice'` (GM-127, `analysis.m_alpha_statement`).
- **`method_solved`** False when Spencer / Morgenstern-Price / GLE finds no
  equilibrium solution on the surface (the GLE does not converge; the
  reason, e.g. "no lambda from -1.5 to 1.5 gives moment and force factors
  of safety that agree", is in the warning). **`method_used`** is ''
  when `FOS` is the method's own solution and None when the method has
  none (it could name the "legacy approximate ... solver" until GM-140
  step 2). **`methods_without_solution`** lists the FOS fields (headline
  or `compare_methods`) that hold no converged solution.
- **What `FOS` holds when there is no solution.** Since GM-140 step 2,
  nothing, with or without supports: `FOS` is None (to_dict null,
  `is_stable` None; the calc package prints no stability check), and the
  same for the `compare_methods` values. What GM-81 decided here, now
  superseded: with supports crossing the surface `FOS` was None, because
  the legacy approximate solver ignored supports (on #85's anchored toe
  circle 0.938 for a surface Bishop puts at 1.372; 0.891 and 1.304 before
  GM-134; before GM-81 that number came back as the Spencer FOS with only
  a Python warning); with no support crossing it `FOS` was that solver's
  value, labelled (`method_solved` False, `method_used`, the warning
  giving theta or lambda). That solver (`methods.solve_rigorous` and the
  `*_legacy` functions) has been removed.
- **Searches** never consume `analyze_slope`; since GM-140 step 2 they
  score every surface by the method's own solution (`solution.solve_method`
  through `search._compute_fos`, the solver `analyze_slope` uses too), and
  a surface with no solution is not scored and is counted by family and
  method (`no_solution`). Until then a circle no support crossed was scored
  by the legacy solver when the GLE failed (#85 without its support: every
  circle, see below), and the search labelled such a governing circle
  (`_legacy_critical_warning`, removed).
- **Found on the way.** Without supports the GLE kept the old lambda
  search (GM-60 made only the supported one robust), and on #85's clay
  without its support it found no solution on any circle tried (a 6 x 6
  Spencer search called the legacy solver 415 times); the legacy value was
  Bishop's there. Changed in GM-140 step 2: the robust lambda search
  serves every run (the direct march where the fixed-point step cannot be
  taken is still used with supports only; see "The lambda search").

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
  reverse-curvature circle closed by a crack; the circles through the toe
  follow `reverse_curvature`, see GM-127 below) and keep m_alpha =
  cos(a) (1 + tan(a) tan(phi) / F) >= 0.2 on the slices the m_alpha rule
  looks at (Slide2 checks every slice with its 'Check malpha < 0.2'
  setting on, error -112; since GM-127 every family checks, in
  `_compute_fos`, by default on the slices with phi' > 0). The family
  leaves out the circles that fail either check, and `choices` says so.
  Without the last two, Duncan Ex 4 read
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
refined) and the "surface families scored" entry of `choices`. Nothing in it
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

### Time budget of a search; the Bishop screen (GM-123, 2026-10-17)

A search had no bound on its run time. In a live run a 30 x 30 entry/exit
grid with 60 slices scored with Spencer took 957 s (Bishop on the same grid
45 s, the same FOS) and a second one ran past the agent's 1,800 s turn, so
the user got no answer.

**The budget.** `search_critical_surface` and `search_rapid_drawdown` take
`time_budget_s`. The library's default is None, no limit, so a library
search, and every pinned and validation test, never depends on the
machine's speed. The agent's tools give every search
`search.SEARCH_TIME_BUDGET_S` (180 s) unless the agent gives another, up
to 600 s (more is refused, not clipped): the slope_stability adapter's
searches, `monte_carlo_fos` with `research_surface` (per realization), the
calc_package slope report (its search and its Monte Carlo re-searches) and
the project run (`geo_project.builders.run_analyses`, per search); SALib
reaches the slope tools through the adapter. A Monte Carlo run counts its
re-searches the budget cut (`n_searches_cut`). `search.search_budget`
holds the budget in a context variable that every search reads
(`_time_up`). The main search may use 80 % of it. The centre grid and the
entry/exit grid score their points by `search._scan_grid`: coarse first
(every s-th point along each axis and the last, s the largest power of two
leaving at least three intervals), then, after each finished level, around
the three lowest points so far (the 3 x 3 neighbourhood at half the step,
moving to its lowest point until the centre is the lowest, the step halving
down to every point), then the next level, s halved, down to every point.
A search cut short has covered the whole range at a coarser spacing and
looked closely where the FOS is lowest. On the live-run slope's 30 x 30
Bishop entry/exit map, whose minimum lies in one entry row 0.31 m wide just
above a clay / sandstone contact, the order reaches the minimum after 44
points, against 713 for coarse first alone (a review measured 0.5 % of the
minimum on ACADS 102 -> 29, Slide2 #57 53 -> 30, Slide2 #61 37 -> 20).
A centre grid keeps a quarter of the main search's time for its refinement
walk (GM-87), which then stops where it is. The random noncircular and
weak-layer searches stop between trials, the particle swarm between
iterations and differential evolution between generations (cut short with
no confirmed surface of its own, it reports the random search's best that
seeded it). The toe-circle and circle-passing-above-the-toe families stop at
the full budget; the infinite slope, the profile-end and support checks
always run. A main search scores until it has one surface, whatever the
budget. A search that finishes inside its budget is exactly the search
without one: each point is scored once, whatever the order, and the record
is assembled in the grid's own order (proved against the previous code on
ACADS, Slide2 #57 and the Taylor long profile with Bishop and Spencer: the
same FOS, surface, grid_fos and surfaces scored).

A search cut short reports the lowest surface in what it covered. The
first warning (ahead of a swapped-windows note) opens "Cut short by its time
budget: the reported FOS may be too high (unconservative) ...", and it and
`search_ranges['time_budget']` say, for each stage
stopped, what it scored of how many, at what spacing it covered the range,
what it did not score, the time it took and the run time projected for the
whole stage (time per point times the points), and what did not run; the
warning names the levers: for Spencer, Morgenstern-Price or GLE first the
Bishop screen, then a larger `time_budget_s` (at most 600 s; when the
projected time is longer, it says that alone cannot cover the search), fewer
points or slices, narrower windows. `choices` names the budget, given or
default (the adapters mark the tool's 180 s as the default), with no run
time, so a search's choices do not change from run to run.

Why 180 s for the agent: the ordinary searches finish. Measured on the development laptop
(30 slices): the default centre grid 2-3 s with Bishop and 14-25 s with
Spencer or Morgenstern-Price; 10 x 10 and 12 x 12 entry/exit grids (Slide2
#57, the ACADS slope) 2 s with Bishop and 59-67 s with Spencer,
Morgenstern-Price or GLE. The host ran the 30 x 30 Spencer search about as
fast (957 s against 972 s) and the Bishop one at about half the speed, so
these fit in the main search's 144 s there too. Three minutes a search
leaves the turn room for several searches, the model's own calls and the
answer.

**Cost of each method.** Per trial surface, against Bishop's simplified
method, on the critical circles of four slopes (ACADS, Slide2 #57, the
Taylor long profile, the live-run slope): Fellenius about 1 times, Janbu
1-4 times, Spencer 31-37 times, Morgenstern-Price 31-39 times and GLE 32-40
times at 30 slices; at 60 slices Spencer 42-56, Morgenstern-Price 42-68 and
GLE 41-72 times. A whole search costs less than the ratio where it already
scans with Bishop (the refinement walk, GM-87; the toe-circle families,
GM-13): the default centre grid 5-13 times, an entry/exit grid 28-46 times.
The noncircular searches score with Spencer whatever method is named for
Bishop (PSO and weak_layer always). Measured on the same four slopes at 30
slices, at the tool's defaults (500 trials, 5 points), no budget, the
supplementary families included (GM-129, 2026-10-17): `noncircular` 6.0-7.2 s,
`weak_layer` 2.7-7.1 s, `noncircular_de` 9.9-11.8 s on three slopes and
41.4 s on the live-run slope, where differential evolution scored 2,325
surfaces (465 on the others). The live run of that slope on the host took
141 s at 40 slices, inside the 180 s budget.

**The Bishop screen** (`bishop_screen`, default off). For a circular or
entry/exit search with Spencer, Morgenstern-Price or GLE, every centre or
pair is scored with Bishop; the lowest are re-scored with the method itself
(at least 10, every one within 5 % of the screen's minimum, at most 30: the
centre's radius optimised again; the pair's centre offset by golden section
from the screen's offset); only those can be reported, and `grid_fos` tags
the others `method: bishop`. Not applied where supports are modelled (as the
refinement walk) or to a per-surface solve such as rapid drawdown; not to
noncircular searches, where a Bishop-like method and a rigorous one can
rank surfaces differently. Measured with Spencer, Morgenstern-Price and GLE
on ACADS, Slide2 #57, the Taylor long profile and the live-run slope
(default grid and 30 x 30 entry/exit at 60 slices): the same governing
surface and FOS in every case, in 35-60 % of the time on the default
centre grids, 15-30 % on 10 x 10 and 12 x 12 entry/exit grids and 3 % on
the 30 x 30 grid at 60 slices (33 s against 972 s). Asked for and not
applied (a noncircular search, supports modelled, or a method other than
Spencer, Morgenstern-Price or GLE), `choices` says so and why; when its cap
of 30 leaves out points within 5 % of the screen's lowest, it says how many. A first version that re-scored by golden
section about the screen's offset, without the screen's own circle, moved
Slide2 #57 Spencer 1.4005 -> 1.4031 to a neighbouring circle: the screen
depends on its short list and its re-score. It stays an option: no
reference prescribes screening a rigorous search with Bishop, and Bishop
and the rigorous methods can rank circles differently where interslice
forces matter.

### m_alpha below 0.2: Slide2's check on every scored surface (GM-127, 2026-10-17)

> **Since GM-140 step 3 (2026-11-13)** a surface with m_alpha below 0.2 is
> kept and flagged by default (OWN-102; see "The printed acceptability
> checks" above). What this section describes, leaving such a surface out
> and counting it, is the stated option `low_m_alpha='refuse'`. The values
> below were measured under that rule.

m_alpha = cos(a) + sin(a) tan(phi') / F, with the base angle a positive in
the sliding direction, divides the normal force on a slice base in Bishop's
simplified method (Bishop 1955), in Janbu's simplified method and in the GLE
(Fredlund & Krahn 1977; `gle._GLESystem._normal`). Where it is small or
negative the normal force grows without bound or turns negative, and the FOS
may be incorrect or misleading (Whitman & Bailey 1967), though not
necessarily wrong. Slide2 checks it on the final iteration and, with its
'Check malpha < 0.2' setting on, reports a surface with m_alpha < 0.2 on
any slice as invalid (its error -112); its documentation says such slices
are usually steep ones in the passive zone. GM-122 applied the check to the
circles passing above the toe; GM-127 applies it to every surface a search
scores.

**Which slices** (`m_alpha_rule`, `methods.m_alpha_checked`). By default
('frictional') the slices with phi' > 0. On a phi' = 0 slice m_alpha =
cos(a) does not depend on F, and in Bishop's sum the slice's resistance,
c b / m_alpha, is c l whatever the normal force, so leaving such a surface
out does not remove a spurious FOS: it only raises the minimum. A vertical
5 m cut in undrained clay (cu 20 kPa, 18 kN/m3; `test_m_alpha`): the
default search finds 0.800 on a toe circle closed by a crack whose last
slice has m_alpha 0.18; checking every slice leaves it out and reads 0.933,
against Taylor's 3.83 cu / (gamma H) = 0.851, 9.6 % on the unsafe side.
'every_slice' checks every slice, as Slide2 does; `search_critical_surface`,
`analyze_slope` and the agent's search tool take it as `m_alpha_rule`, the
`choices` entry names the rule used (given or default) and offers the other,
and inner searches (the drawdown search's stages) use the rule in force.

**The rule per method** (`methods.m_alpha_values`, `search._m_alpha_least`):

- Bishop: at Bishop's F, the base angle signed by the gravity moment about
  the centre, as `bishop_fos` signs it.
- Janbu (simplified): the same expression at Janbu's uncorrected F, the F
  its normal forces are solved with (the correction factor multiplies
  afterwards).
- Spencer, Morgenstern-Price, GLE: the module solves all three with the GLE,
  whose normal force has the same denominator at the method's own F, so the
  method's own m_alpha IS the Bishop-form expression at the converged F.
  The legacy Spencer solver's `cos(a - theta) + sin(a - theta) tan(phi')/F`
  is its own fallback form and is not the check.
- Ordinary Method of Slices: no m_alpha (the normal force is W cos a), so
  nothing is checked, and `choices` says so.
- Bishop asked for on a noncircular surface is solved with Spencer and
  checked as Spencer.

**Search against a single surface.** A search leaves the surface out (it
scores 999.9 in `_compute_fos`) and counts it: every family (the centre grid
and its refinement, entry/exit, the toe circles and the circles passing
above the toe, noncircular random, PSO, DE and the directed weak-layer
seeds, and the rapid-drawdown searches, on the drawn-down stage). The count
is `SearchResult.n_rejected_m_alpha` (inner searches add to the outer one),
`to_dict()["n_rejected"]["m_alpha"]`, a line in `summary()`, a row in the
calc package, and the `choices` entry "slip surfaces left out for m_alpha"
with its basis. When no surface holds, a warning gives each reason's count
and suggests a circular or entry_exit search. The differential evolution's
objective scores a refused surface 100 + 10 (0.2 - m_alpha) instead of a
flat 999.9, so the evolution is led back toward admissible surfaces rather
than stopping on a flat score (the w11 search below stopped after 600
evaluations on the flat score and runs all 2,325 graded); when the
evolution ends on no admissible surface of its own, the best surface of the
random search that seeded it is reported, and a warning says so. A single
surface given to `analyze_slope` or `rapid_drawdown_fos` is analysed and its
FOS reported, with a warning: the least m_alpha and its slice, the base
angle against or with the sliding, phi', how many slices, whether it is in
the passive zone, the references, that a search would leave it out, and
that finer slices lower m_alpha at a steep end rather than clear it: a
surface that comes out no steeper than the Rankine passive angle, 45 deg -
phi'/2 (Duncan, Wright & Brandon 2014), avoids such slices. FOSM / PEM /
Monte Carlo on a fixed surface do not refuse it (`probabilistic._eval_fos`):
they sample the surface the user chose, whose deterministic analysis carries
the warning. The Spencer value in the support-arm warning is that circle's
own.

**Twice the slices.** After a search the governing surface is solved again
at the search's slice count and at twice it (two solves,
`search_supplements.m_alpha_confirmation`); `choices` gives both FOS and
least m_alpha, or why there is none on twice the slices: the finer slicing
cannot be built (until GM-134 Taylor's deep circle, tangent to the base,
shed a slice below it at 60; since GM-128 a circle touching the base is kept
at every count) or the method has no solution there. A warning
(`finer_slices_warning`) says when the FOS moves more than 1 % (with
m_alpha too where it falls below 0.2 on the finer slicing), or when the
method has no solution on the finer slicing (Slide2 #85 with its support,
Spencer search: 1.542 on 30 slices, none on 60); a slicing that cannot be
built says nothing about the FOS and is not warned of. On a phi = 0 circle
closed by a crack at its vertical tangent the last slice's m_alpha falls
as the slicing gets finer (Slide2 #85's 1.304 circle: 0.182 on 30 slices,
0.129 on 60, 0.065 on 240). Moves above 1 % on the pinned cases: #85 Bishop
1.304 to 1.325, its clay face without the support 0.882 to 0.897, GM-13
2.286 to 2.316, Duncan Ex 4 Bishop 1.259 to 1.288 and Janbu 1.199 to 1.238,
the vertical cut 0.801 to 0.813. The rapid-drawdown search does not do this.
Those moves were mostly the circle's tangent slicing at its vertical end;
since GM-134 each slice takes its chord ("Where a circle is sliced"): #85
1.372 on 30 slices, 1.373 on 60; GM-13's toe circle 2.346 with no warning;
Duncan Ex 4 Janbu 1.203 to 1.212; the vertical cut 0.842.

**Reverse curvature in the toe families.** The circles through the toe
follow the geometry's `reverse_curvature` (default: closed by a crack, as
Slide2's default does), as the main search does; the circles passing above
the toe leave reverse-curvature circles out (GM-122). Made the same either
way, a pinned case moved away from its published value: both families
following `reverse_curvature` took Slide2 #85 with its support from 1.304
to 0.866 on a circle passing above the toe closed by a crack at the crest
(D&W 1.51, Slide2 1.531; measured with every slice checked); both leaving
them out took the short Taylor profile from 1.970 to 2.015 (Taylor 1.947)
by leaving out a toe circle closed by a 1.65 m crack whose least m_alpha is
0.23, a circle Slide2 keeps. On a phi = 0 circle the slice at the crack has
m_alpha = cos(a) near zero, which the default does not check;
'every_slice' leaves such circles out of both families.

**What moved against 184c3f3** (slope "not slow" suite, slope validation
examples, `funhouse_agent -k slope`; every other value stayed put):

| Case | Before | After | Published | Why |
|---|---:|---:|---:|---|
| Duncan Ex 4, Janbu search, 30 slices (`test_m_alpha`) | 1.076 | 1.199 | (Janbu 1.213 on 120 slices) | old circle's least m_alpha 0.007, in the sand: left out |
| Slide2 #85 + support, Bishop, `'discard'`, active | 1.467 | 1.432 | 1.531 (D&W 1.51) | the circles passing above the toe no longer check phi = 0 slices: a circle with m_alpha 0.195 on its last slice is kept (`'every_slice'`: 1.467) |
| same, passive | 1.296 | 1.274 | 1.324 (D&W 1.32) | the same (`'every_slice'`: 1.296) |
| #85 clay face, no support: circles passing above the toe (not the governing family) | 0.910 | 0.906 | | the same, m_alpha 0.19 on a phi = 0 slice (`'every_slice'`: 0.910) |
| same, Janbu search: circles passing above the toe (not the governing family; the toe circle's 1.024 governs, unchanged) | 1.033 | 1.138 | | the family's Bishop scan now ranks first circles centred at the crest's level with a near-vertical last slice, on which Janbu has no solution, and the re-scored shortlist ends on a different circle (1.142 on 120 slices; `'every_slice'`: 1.033) |

The governing values of Slide2 #85 (1.304 active, 1.208 passive, Spencer
1.542, GLE 1.584), its clay face without the support (0.882), GM-13 (2.286),
the thin-base model (0.2715) and the vertical cut (0.800) are those of
184c3f3: every slice below 0.2 on them is phi = 0. (Since GM-134: #85
1.372 / 1.254 / 1.551 / 1.573, its clay face 0.910, GM-13 2.346 and the
vertical cut 0.842; VALIDATION.md B16.)

The w11 geometry (ground (0,0), (20,0), (30,10), (60,10); 3.5 m of clay,
c' = 3, phi' = 22, over sandstone c' = 200, phi' = 40; no water): Bishop
0.9359, Spencer 0.9354 (screened or not) unchanged, on the circle in the
clay passing above the toe (least m_alpha 0.79; 49 trial circles left out).
The DE search given windows 20-30 / 28-60 (5 points, seed 3) reported 1.570
on a surface 32 m deep through the sandstone with m_alpha -0.37 on 10 of
its 40 slices; it now runs all 2,325 evaluations and reports 6.847 on a
surface through the sandstone (entry x = 20.97, exit x = 39.06) with
m_alpha >= 0.2, leaving 194 out for m_alpha (since GM-131 it finds no
surface that holds and says so; "Where a noncircular surface is sliced"
below). Its interior points sit at
0.05 to 0.95 of the depth to the model's base at -30 m, so surfaces in the
3.5 m of clay are a sliver of what it samples; the circular and entry_exit
searches find the clay's 0.935. Run time: the check is one pass over a
scored surface's slices; the ACADS Spencer entry/exit search took 45 s
before and after, and the slope suites ran no slower.

### Where a noncircular surface is sliced (GM-131, 2026-10-17)
`slices.slice_boundaries(geom, slip, n_slices)` gives the x of every slice
boundary, entry to exit, and `build_slices` cuts there.

- **A circle**: `n_slices` slices of equal width, each with the circle's
  angle at its midpoint (until GM-134, next section).
- **A polyline**: a boundary at each corner of the surface, each break in
  the ground surface, and each crossing of a layer's top or bottom or of the
  water table, strictly between entry and exit. Every corner is kept, however
  close to another boundary (one within 1e-6 m of an end, find_entry_exit's
  bisection tolerance, is that end): merged away, the slice across it took
  its chord across the corner (a 4 m step 1 mm wide at the entry read 2.139
  on 30 slices, 2.259 on 60, against 2.521 on both). A ground break or a
  crossing within 1e-4 of the span of another boundary is merged into it.
  The crossings are exact: the surface and each
  boundary are straight between the union of their vertices, so the
  difference changes sign once at most on each piece. The stretches between
  boundaries share out `max(n_slices, stretches)` slices in proportion to
  their width, at least one each (each further slice goes to the stretch
  whose slices are widest), and each stretch is cut evenly. Each slice's
  base angle and base length come from the chord between its two
  boundaries, which IS the surface there, so the base of every slice is one
  straight segment in one material under one straight piece of ground.
- **References.** Duncan, Wright & Brandon (2014) Ch. 6 (each slice base in
  one material, the surface straight across it); Slide2 help, "Slice at
  piezo-surface intersection", and its non-vertical slice method ("slice
  boundaries will be located at each vertex of a failure surface").
- **Why.** Cut evenly, a slice across a corner took one segment's angle for
  its whole weight. On Taylor's D = 2 profile one search's polyline read
  0.650 on 30 slices, 0.807 on 29, 1.182 on 31 and 0.862 on 45, against
  0.970-0.973 now on 29 to 45; a search's minimum picked out the low errors
  (Taylor DE 0.610, -13.6 % on the 0.706 circle; now 0.680, -3.7 %, the
  polyline's own freedom), and a fixed surface could err either way (V-030
  1.190, now 1.218; the old slicing reaches 1.218 only at 3,840 slices).
- **What a result says.** `n_slices` on a result is the number of slices
  cut, which on a polyline can exceed the number asked for (an 82-point
  polyline asked for 60: 85). Every consumer works from the slices
  themselves (each method, m_alpha (GM-127), the slice tables, the
  rapid-drawdown stage mapping by `x_mid`), so none assumes equal widths;
  a surface below the ground in more than one stretch (GM-122) is refused
  by `find_entry_exit` before any slicing; the section figure draws each
  polyline boundary from the surface at the boundary rather than at the
  slice's mid-base.
- **The planar family** (infinite slope) is a polyline too: its end ramps
  are slices of their own, a twentieth of a slice width deep (see
  "Supplementary families").
- **What moved**: VALIDATION.md B15 (every value with a published or
  recorded answer, before and after). The w11 DE search given windows 20-30
  / 28-60 (5 points, seed 3), which reported 6.847, finds no surface that
  holds and says so (2,325 evaluated: 398 refused for geometry, 1,774 not
  converged, 134 for m_alpha); the old surface reads 6.80 at any count.
- **Not done here; done by GM-134** (next section): circles cut at layer
  and water-table crossings, and surcharge-zone and tension-crack edges as
  boundaries.

### Where a circle is sliced; surcharge and crack edges (GM-134, 2026-10-17)
- **Where.** `slices.slice_breaks` gives, for a circle as for a polyline,
  each break in the ground surface, each crossing of a layer's top or
  bottom, of the water table and of a tension crack's base on the crest
  half of the surface (and the middle of the surface, where that half ends,
  when the crack reaches it), and each edge of a surcharge zone; a polyline
  adds its corners. A circle's crossings are the roots of the circle-line
  quadratic on each straight piece of the line, on the lower arc. The
  stretches share out the slices as GM-131's do. Before, a circle was cut
  evenly, and neither kind was cut at a surcharge or crack edge: a slice
  across a crossing took one material, pore pressure or load for its whole
  base, and a fixed circle's FOS jumped with the count (V-043's circle
  1.246 / 1.190 / 1.180 / 1.232 on 25 / 29 / 30 / 31 slices; 1.208-1.211 on
  25 to 45 now, 1.213 on 3,840 either way, Slide2 1.209).
- **The base of each slice is its chord**, on a circle as on a polyline:
  alpha = atan2(z_right - z_left, b), base length = hypot(b, dz); the base
  elevation stays on the arc at x_mid, so the weight is the area under the
  arc. Why the chord and not the circle's tangent at the slice midpoint
  (the rule before GM-134):
  - The methods assume a straight base (Duncan, Wright & Brandon, Ch. 6).
    On a circle the chord's perpendicular bisector passes through the
    centre, so its normal force acts through the centre, as moment
    equilibrium about the centre takes it. Bishop's driving moment,
    W (x_mid - xc) / R, does not use alpha, and its resisting term
    c' b / m_alpha is c' times b / cos(alpha), the chord's length. Slide2's
    help takes "the mid-point of a slice base" for the moment arm and says
    nothing more of the base angle.
  - The tangent segments do not join. Where the arc turns vertical (a
    circle closed at its vertical tangent, x = xc +/- R, by reverse
    curvature; a vertical cut; a steep exit) the last slice's base,
    b / cos(alpha), falls short of the arc by about a factor of the square
    root of two however thin the slice, and the FOS converges only as one
    over the square root of the count: one toe circle 4.7 % low at 30
    slices and still 0.4 % at 3,840. The chord's error falls as one over
    the count (within about 0.5 % at 30 slices; Janbu 0.9 %).
  - Elsewhere the two agree within 0.3 % (V-043's circle at 30 slices:
    tangent 0.11 % low, chord 0.26 % low).
- **A circle cut off at its vertical tangent** ends where
  `slip_elevation_at` says outside; `slices._boundary_elevation` gives the
  centre's level there, so its last chord exists.
- **Every consumer works from the slices**: each method, m_alpha (GM-127),
  the slice tables, the figures, the calc steps, FOSM and Monte Carlo (each
  sample calls `build_slices`). A surface below the ground in more than one
  stretch is refused by `find_entry_exit` before any slicing (GM-122).
  Rapid drawdown cuts both stages alike: each stage's water-table crossings
  are boundaries in the other (`build_slices(extra_boundaries=)`), so the
  stage-2 strengths map slice for slice. `search._dips_below_base` took the
  same boundaries; since a crossing of the deepest layer's bottom was a
  boundary, a circle dipping under it was refused at every slice count
  (before, one 8 mm under the base was scored on 30 slices and refused on
  90). GM-128 measures the depth under the base on the surface itself, and
  the base's crossings are no longer boundaries ("A surface touching the
  model's base").
- **Side effects.** On a phi = 0 circle Spencer and the GLE now equal
  Bishop to about 0.1 % rather than exactly, mainly because the shear on a
  chord base acts at R cos(dtheta / 2) from the centre, not R (`gle.py`
  takes the arms as they are); the normal's offset from the centre counts
  for little (moving z_base to the chord's midpoint changes Spencer by
  0.0006). Their searches on phi = 0 slopes take 20-25 % longer (more solver
  iterations per surface and more surfaces in the centre refinement); the
  slicing itself costs 0.03-0.05 ms more per surface, and Bishop searches
  run as before. The toe circles that won on steep phi = 0 slopes (GM-13's
  2.287) won partly by the tangent's error at their vertical end.
- **What moved**: VALIDATION.md B16, every value with its old code's own
  value on 3,840 slices.
- **Not done.** The edge of ponded water (where the water table crosses
  the ground) is not a slice boundary.

### The refinement walk moves on the method's own FOS (GM-138, 2026-10-18)
- **The defect.** On V-043 (Slide2 #39 sand fill, c' = 0 / phi' 37 over 3 m
  of soft clay, phi = 0, on a base at el 0) the default Spencer search
  reported 1.238 on a centre of its grid, (16.67, 10.87), where an
  independent brute force finds 1.186 on 30 slices (1.190 on 480): a circle
  tangent to the clay's base centred over the face's mid-length,
  (14.98, 9.76) R 9.76. The circle the test pins (1.210, from an old
  entry/exit search) is not the critical one either. Along the base-tangent
  circles Spencer's FOS is a bowl about x = 15, the face's mid-length:
  1.240 / 1.187 / 1.237 at centre x 13.33 / 15.0 / 16.67, the grid's columns
  falling either side of it.
- **Why the walk missed it.** The grid's lowest centres are refined by a
  compass walk (GM-87). For Spencer, Morgenstern-Price and GLE the walk
  scored each centre with Bishop and only its end with the method. Bishop and
  Spencer rank this slope's two families of circles differently: at the
  start, Bishop's best circle is the deep one, 1.270 (Spencer 1.237); one
  step on, at x 18.33, Bishop's best is a shallow circle on the sand face,
  1.259, near the infinite slope tan 37 / tan 31 = 1.256, which Spencer
  reads no lower. The walk took the first step that lowered Bishop's FOS,
  followed the shallow circles to 1.256 and ended there; Spencer at the end
  was no lower than at the start, so all three walks added nothing. Through
  the frictional fill Spencer reads the deep circles 2-3 % below Bishop; on
  the shallow circles the two agree (force equilibrium governs a thin slide).
- **Not the other suspects.** The centre heights (9.6-21) hold the critical
  centre and its bowl is flat in height (1.1858-1.1864 from 9.35 to 10.5);
  the radius scan finds it at its centre (`optimize_radius` at (15.0,
  10.15) gives 1.187); the search takes 13 s, far inside a 180 s budget;
  it leaves two circles out for m_alpha; and it is a circle of the grid's
  own family. With the Bishop screen on, the grid's minima are
  Bishop's and the old walk reached the bowl by chance (1.187).
- **The fix.** At each centre of the walk Bishop optimises the radius, as
  before, and the method scores that circle (one solve more a centre); the
  walk moves only where the method's FOS falls. At the walk's end the method
  optimises the radius again and the lower of the two is kept (`grid_fos`
  records the circle reported). Each move now lowers the method's own FOS;
  Bishop only chooses the radius, the cheap one-dimensional part. Bishop,
  Janbu and the ordinary method walk with themselves, unchanged, as do a
  model with supports and a per-surface solve (rapid drawdown).
- **What moved** (default searches on the published circular cases, before
  and after; best known: the brute force above, Spencer on phi = 0 cases
  taken on Bishop's best circle; VALIDATION.md B17): V-043 Spencer 1.2375 ->
  1.1859 (best 1.1858), Morgenstern-Price and GLE 1.2464 -> 1.1958; Slide2 #57
  (V-031) Spencer 1.4177 -> 1.4174 (best 1.4173, Slide2 1.422); V-042 Spencer
  1.3464 -> 1.3461 (best 1.3454); Duncan Ex 4 Spencer 1.2586 -> 1.2584 (best
  1.2584). Every other case, every Bishop search and every search the slope
  suite and the slope validation examples pin are unchanged. Every default
  search is now within 0.0013 of the lowest circle the brute force found
  among circles with no reverse curvature, or below it (Slide2 #85, whose
  searches report toe circles closed by a crack).
- **The families.** Re-measured (Bishop / Spencer): the toe circle governs
  ACADS 1(a) by 0.001 (0.9851 / 0.9840 against the grid's 0.9863 / 0.9850)
  and the steep phi = 0 slope from its toe (GM-13, V-054's face) by 0.001
  (2.3464 / 2.3466 against 2.3475 / 2.3477); with level ground past that toe
  the base circle governs (2.222). On V-043 the toe circle, the circle
  passing above the toe and the infinite slope read 1.254-1.298, 6-9 % above
  the circle found.
- **Run time** (best of two, laptop): V-043 Spencer 11.5 -> 13.0 s (its walks
  now go somewhere), V-031 13.3 -> 13.5 s, Slide2 #85 with its support 21.2 ->
  21.1 s, V-054 17.4 -> 18.5 s; Bishop 1.9-2.7 s before and after; all the
  searches of the slope suite 452 -> 456 s.

### A surface touching the model's base (GM-128, 2026-10-18)
- **The defect as filed.** Taylor's critical circle for D = 2 (FHWA
  NHI-06-088 Fig. 6-15), tangent to the firm base, was refused at 40, 60
  and 80 slices and accepted at 30 and 50: `build_slices` dropped up to
  max(2, 3 % of n_slices) below-base END slices as slivers and refused a
  surface with more (SS-5), and whether a slice midpoint fell in the
  circle's dip of millimetres under the base depended on the count. GM-138
  found V-043's critical circle tangent to the base too.
- **On master it no longer reproduced as filed.** Since GM-134 the base's
  crossings were slice boundaries, so the count no longer decided: Taylor's
  circle tangent to 1e-9 m was kept at 25-100 slices and 1 micron under
  refused at every count. The tolerance had become an accident of the
  slicing (crossings closer than 1e-4 of the span merge): under a micron.
  The entry/exit search had recovered V-043's clay circle 3.5 mm under the
  base, and its test cut the radius to 0.1 mm above it.
- **The rule.** The depth of a surface under the bottom of the deepest
  layer is measured on the surface, not on its slices: `slices.base_depth`
  gives the greatest depth exactly (between vertices the deepest bottom is
  the least of straight lines, a circle's lower arc is convex and a
  polyline straight, so the depth peaks at a piece's end, where two bottoms
  cross, or where the arc runs parallel to a bottom; at a vertical step in
  a bottom, which a DXF import can give, each piece's bottoms are taken
  from points inside it, since `bottom_at` reads the step's left value).
  A surface
  `build_slices` is given within a thousandth of the model's height of the
  base (`base_tolerance`; `model_height` is the highest ground less the
  lowest point of the base) touches it and keeps every slice at any count,
  a slice under the base taking the deepest layer's strength; one reaching
  further is refused at any count, with its depth and where (Taylor's
  circle 5 cm under): "Slip surface passes below the bottom of the deepest
  soil layer: 0.050 m below it at x = 48.77 (a surface may touch it, within
  0.024 m, a thousandth of the model's height)." The base's crossings are no longer slice boundaries (a
  crossing within the tolerance only touches it, and the slice under takes
  the same layer): a dip of microns would otherwise put a boundary there
  and move the FOS (Taylor on 25 slices 0.7045 -> 0.7036). A slice's
  weight counts the part of it below the base (at most a thousandth of the
  model's height) as empty; on Taylor's circle 1 cm under that is worth
  about 0.00005 of FOS, and it is left to the review of the slice integrals
  (GM-140).
- **Why a thousandth of the model's height.** A circle given as tangent
  with its centre and radius to the centimetre, as published examples give
  them, lies within it on a model 10 m or more high (Taylor's 24.4 m model:
  0.024 m; Slide2 #85: 0.020 m; V-043's 9 m: 0.009 m). Reaching that far
  under a firm base lowers the FOS by no more than the slice count moves it:
  Taylor's circle by 0.02 % (the slice count moves it 0.26 %: 0.7045 on 25
  slices, 0.7063 on 100), V-043's by 0.06 %, Slide2 #85's 'discard' circles
  by 0.14-0.26 %. A fraction rather than a length, because the FOS changes
  with a circle's depth over the model's own length scale.
- **A search keeps to the model.** `_compute_fos` leaves out of every
  search family a surface reaching more than a millionth of the model's
  height under the base (`search._dips_below_base`,
  `_SEARCH_BASE_TOUCH`); a caller scoring one given surface passes
  `inside_model=False`, as the probabilistic analyses of a fixed surface do.
  Within the thousandth the FOS still falls with depth, so a search granted
  it ends every base-tangent search that far into ground the model does not
  have, on a circle another program refuses as crossing the model's
  boundary. Measured with the search granted it: Taylor 0.7049 -> 0.7048
  (21 mm under), V-031 Spencer 1.4174 -> 1.4160, Slide2 #85 'discard'
  1.511 -> 1.506 (19 mm under); and on #85 with its support the toe family
  pivoted about the toe, which sits on the base, onto circles centred
  0.64 ft past it whose support went uncounted: 1.01 against 1.372. The
  radius search closes on the circle tangent to the base from above, since
  the check refuses anything under it. A cheap bound (`base_clearance`:
  the surface's lowest point less the base's highest) skips the exact
  depth for a surface wholly above the base.
- **Not modelled.** A firm layer inside the model, not at its base, is a
  layer like any other: a circle passes through it with its strength.
  Slide2's impenetrable material and composite surfaces (a circle that
  follows the top of a firm layer) are not implemented.
- **What moved** (VALIDATION.md B18). Given surfaces: Taylor's circle kept
  at 25, 30, 40, 50, 60, 80 and 100 slices (0.7045-0.7063), and to 1 cm
  under the base at the same FOS within 0.0002; 5 and 10 cm under refused
  at every count; V-043's clay circle as recovered, 3.5 mm under, 1.0408
  against the pinned 1.0409; Slide2 #85's toe circle (Bishop) 1.37156 ->
  1.37152 (the base's crossing at its toe is no longer a slice boundary).
  Searches: Taylor D = 3 Bishop 0.677377 -> 0.677376 (another centre in
  the same flat valley); the toe family's candidate in #85's 'discard'
  searches 1.5584 -> 1.5583 (not governing). Nothing else the slope suite,
  the slope validation examples, `funhouse_agent -k slope` or the worked
  examples record moved, and no search result rose.
- **Run time** (default searches, laptop, best of three, before / after):
  Taylor Bishop 2.1-2.3 / 2.1 s, Spencer 12.8-14.4 / 12.4 s; V-043 Spencer
  14.1-14.7 / 13.6 s; #85 with its support, Bishop 2.5 / 2.5 s.

## Critical Sign Conventions
- **Driving moment**: uses `W*(x_mid - xc)/R` + `abs(driving)`, NOT `W*sin(alpha)`
  directly, because `sin(alpha)` can be negative for L-to-R slopes
- **Seismic moment**: computed separately from gravity driving, then
  `driving = abs(gravity) + abs(seismic)` so seismic always increases
  driving regardless of slope direction (L-to-R vs R-to-L)
- **Spencer m_alpha**: `cos(alpha - theta) + sin(alpha - theta)*tan(phi')/F`
  (reduces to Bishop when theta=0); the legacy Spencer solver only
- **m_alpha check** (GM-127, `methods.sliding_sign` / `m_alpha_values`):
  `cos(a) + sin(a) tan(phi') / F` with a positive in the sliding direction:
  for Bishop on a circle, the sign of the gravity moment about the centre
  (pond forces included), as `bishop_fos` takes it; for Janbu and the GLE
  family, the sign of sum (W + surcharge) sin(a), as `gle._normalize` takes
  it; by default only slices with phi' > 0 are checked
  (`methods.m_alpha_checked`)

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
- **Below-the-base rejection, generalized (SS-5, v5.4 F2c; replaced by GM-128,
  "A surface touching the model's base", as the end of this note says)**: a trial slip
  surface whose base drops below the bottom of the DEEPEST soil layer passes
  through material that does not exist, so it is geometrically inadmissible.
  `build_slices` skips such below-base slices and, after the loop, REJECTS the
  surface (raises `ValueError`, which `_compute_fos` maps to `_FOS_MAX` so a
  search discards it) when EITHER (1) a below-base gap is bracketed by surviving
  slices on both sides (the original interior-hole check), OR (2) the below-base
  span exceeds a small tangential-sliver tolerance `max(2, round(0.03·n_slices))`
  (NEW; since GM-131, 3 % of the slices cut, which on a polyline may be more
  than asked for). Trigger (2) closes a hole in trigger (1): a circle that plunges below a
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
  families" under Search Algorithms.) **Since GM-128** no slice is dropped:
  the depth of a surface under the base is measured on the surface
  (`slices.base_depth`), a given surface within a thousandth of the model's
  height of the base touches it and keeps every slice, one deeper is refused
  at any count, and a search keeps to surfaces inside the model. The claim
  above that a tangent circle touches "at most a slice or two" did not hold:
  Taylor's circle lost more at 40, 60 and 80 slices.
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
  - **Convergence-gate moment axis has a robust fallback** (v5.3 review fix;
    removed from scoring by GM-140 step 2, see "A method's own solution, or
    no number"):
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
- **Rapid drawdown, as EM 1110-2-1902 Appendix G prints it (GM-140 step 9,
  2026-11-13)** — `rapid_drawdown.rapid_drawdown_fos(geom, from_el, to_el,
  xc/yc/radius, method='duncan_3stage'|'lowe_karafiath'|'corps_2stage')`
  → `RapidDrawdownResult` (also `analysis.rapid_drawdown_fos`). **The rule:
  each stage is computed as Appendix G prints it, each stage's factor of
  safety is reported, and the answer is the factor of safety of the last stage
  computed.** A layer that does not drain freely carries the R envelope
  (`R_c`, `R_phi`) beside its effective `c_prime`/`phi`, or the measured
  `R_d`, `R_psi`; a layer with neither is free-draining. The manual (31 Oct
  2003) is the owner's copy; the equations below were read from its pages.
  * **First stage (G-3a, p. G-2).** Effective-stress analysis with the pore
    pressures before drawdown: σ'_c = N'/dl (Eq. G-1), τ_c = S/dl (Eq. G-2).
  * **Strengths for the second stage, `lowe_karafiath` and `duncan_3stage`
    (G-3b, pp. G-4 to G-8).** The strength of soil consolidated at K_c = 1 is
    τ_ff = d + σ'_c tan ψ (Eq. G-9), with d = c_R cos φ_R cos φ' / (1 − sin φ_R)
    (Eq. G-5) and ψ = arctan[sin φ_R cos φ' / (1 − sin φ_R)] (Eq. G-6)
    (`r_envelope_d_psi`), or d and ψ as measured (`R_d`, `R_psi`; G-3b(1)(a)
    and (b)). The strength of soil consolidated at failure is τ_ff = c' + σ'_c
    tan φ' (Eq. G-10); K_f by Eq. G-7 where c' = 0 and Eq. G-8 where it is not
    (`_Kf`). The slice's K_c is Eq. G-11 (p. G-7), held between 1 and K_f
    (`consolidation_stress_ratio`), and τ_ff Eq. G-12 (p. G-8;
    `second_stage_strength`).
  * **Strength for the second stage, `corps_2stage` (G-2, pp. G-1 to G-2).**
    The lower of the R envelope, c_R + σ'_c tan φ_R with c_R and φ_R as
    entered, and the effective-stress envelope. No third stage.
  * **Second stage (G-3c, p. G-8).** Solved after drawdown with τ_ff as
    cohesion, φ = 0 and no pore pressure on the slices that do not drain
    freely; c', φ' and the pore pressure after drawdown on the free-draining
    ones; the water left against the slope is an external load. Its factor of
    safety and its N on every base are kept. It is the answer of
    `lowe_karafiath` and `corps_2stage`.
  * **Drained check (G-3d, p. G-8), `duncan_3stage`.** σ = N/dl with N from
    the second stage (Eq. G-13), σ'_d = σ − u with u after drawdown (G-14),
    s_d = c' + σ'_d tan φ' (G-15) (`drained_check`). A negative σ'_d gives
    s_d = c': the manual prints 0.00 for such a slice with c' = 0 (Fig. G-11b);
    for c' above zero that is the module's reading. Where s_d is not below
    τ_ff on any slice the second stage is the answer and no third stage is
    computed.
  * **Third stage (G-3e, pp. G-8 to G-9).** The second stage again with c', φ'
    and the after-drawdown pore pressure on the slices whose drained strength
    is lower (not that strength entered as a cohesion), τ_ff with φ = 0 on the
    rest.
  * **Water (Table 2-1, p. 2-2; G-4b, p. G-9).** Before drawdown: the seepage
    line when one is given (`stage1_phreatic_points`), otherwise a level pool,
    the assumption of the manual's own example ("for simplicity"; seepage
    analyses would often be appropriate), named in the result as assumed.
    After drawdown: the line given (`drawdown_phreatic_points`), otherwise the
    level lowered pool, named. No printed benchmark uses a line after
    drawdown.
  * **One set of slices (GM-135).** The slices are cut at the water crossings
    before and after drawdown alike and paired by index; a surface on which
    the two cuts differ is refused.
  * **A stage with no solution gives no number.** `FOS` is None and
    `no_solution` says which stage and why; no Ordinary Method of Slices
    estimate stands in for a first stage that did not converge.
  * **A base in tension before drawdown (σ'_c below zero).** The manual
    prints no rule. The module takes σ'_c as zero there, uses the lower of
    the two envelope strengths and names the slices in a warning.
  * **What the result carries.** `stage1_fos`, `stage2_fos`, `stage3_fos`
    (None where not computed), `answer_stage`, `stage_method`, `water_before`,
    `water_after`, `choices`, and per slice the columns of Figs. G-10b and
    G-11b: σ'_c, τ_c, K_c, K_f, the two envelope strengths, τ_ff, the second
    stage's N, the after-drawdown u, σ'_d, s_d and the strength used
    (`slice_table()`; `to_dict()["slices"]`).
  * **The method of slices.** Every stage is solved by Spencer's method (the
    GLE engine; `f_interslice` selects another side-force function), which the
    manual recommends (G-5, p. G-9). The manual's example of the improved
    procedure is by the Simplified Bishop Method (2.20 / 1.52 / 1.44, pp. G-15
    to G-20) and its example of the 1970 procedure by the Modified Swedish
    Method with the side forces at a fixed 19.4 degrees (3.49, then 1.35;
    pp. G-9 to G-15). **The manual's check method (EM C-4):**
    `rapid_drawdown.fixed_inclination_solution(slices, slip, theta_deg,
    equilibrium)` is the force-equilibrium factor of safety at a fixed,
    constant side-force inclination (the GLE engine's direct march at
    λ = tan θ), and, with `equilibrium='moment'` at θ = 0, the Simplified
    Bishop Method. It is a library function for reproducing the manual's
    numbers (`rapid_drawdown_fos(stage_solver=('force', 19.4))` or
    `'bishop'`, with `extra_boundaries` for the manual's own slice edges);
    it is not one of the agent's tools and not a `method` name. On the
    manual's example the 1970 procedure reads 3.486 / 1.341 by that method
    (printed 3.49 / 1.35) and 1.239 by Spencer's method, the module's answer;
    a Corps result says so in `notes`.
  * **The removed input.** `stage3_effective_normal` ('fellenius', an
    Ordinary Method of Slices normal, or 'gle', a drained solution of the
    drawn-down slope) is gone: neither basis is the manual's. A caller who
    names it is refused with the rule (`STAGE3_NORMAL_RULE`). This settles
    OWN-17.
  * **What this replaced.** Until step 9 the K_c = 1 strength took c_R and
    φ_R as entered (13 % low on the manual's example: 1,200 psf and 16° are
    1,379 psf and 18.2° by Eqs. G-5 and G-6, and Fig. G-5 prints both pairs),
    K_c came from the base inclination with two special branches (1.68 to
    3.00 on the example's slices, where Eq. G-11 gives the printed 1.71 on all
    twelve), the drained check ran before any second-stage solution on an
    Ordinary Method of Slices normal, and the second and third stages were one
    solve. An earlier note here argued from values on a seepage line no page
    prints that c_R and φ_R were already the τ_ff-against-σ'_c envelope; the
    manual's own example shows otherwise, and that note is withdrawn. Pinned
    stage by stage in `tests/test_rapid_drawdown_em_appendix_g.py`; record in
    `module_work/gm140/STEP9_CHECK.md`.
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
  `RapidDrawdownResult` recomputed on the winning surface. **Every trial is scored by
  `rapid_drawdown_fos` itself (GM-140 step 9), so a trial on which a stage has no
  solution, or whose slices before and after drawdown differ, is not scored.**
  The drawdown closure also REJECTS a trial surface
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
- **Tension crack**: see "Tension cracks and reverse-curvature circles as printed" above (GM-140 step 4). Since that step the surface ends at the crack by default (`tension_crack_model='truncation'`), the crack is at the crest end (`tension_crack_side='crest'`) and its depth can be Rankine's or scanned; `'strength'` is the earlier default.
- **Empty-space slices**: slices where the slip surface is below all soil layers
  are skipped (zero weight, no contribution)
- **Nails**: `SlopeGeometry.nails` (`nails.SoilNail`) reach every method
  through `reinforcement.py`, with anchors, geosynthetics and stabilizing
  piles (see "Support forces in each method" above). The old nail path in
  `nails.py`'s own FOS integration is not used; its tests stay skipped.
- **The legacy Spencer / Morgenstern-Price were APPROXIMATE (SS-1)**
  (removed by GM-140 step 2; the rest of this entry is history):
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
(FLAC, Taylor). (Since GM-134: the toe circle the search finds 2.346, and
with level ground 40 m past the toe a base circle 2.222.)

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
