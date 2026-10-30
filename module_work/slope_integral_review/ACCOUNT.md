# slope_stability: written account of the whole chain (GM-140)

2026-10-30. Asked for by BACKLOG GM-140 (owner, 2026-10-18: "please defer
future slope stability fixes until we can do it more integrally"). This is a
document only: no code was changed. It describes what the module does today,
step by step, sets each step against the printed references, lists every
place where the module departs from them or carries a fix made for one case,
lists the published benchmarks the test suite pins, and ends with a short
ordered plan for the owner to agree. Once agreed, each plan step gets a
BACKLOG ID under GM-140.

How to read it:

* Each step has three parts: what the module does, what the references
  print, and the departures (D1, D2, ...). A departure entry gives the
  module's behaviour, the printed practice with its page, and the
  consequence.
* Every formula and default is tied to a printed page. Where no printed
  source was found, the entry says "not found in print".
* Backlog items this review absorbs are listed under the step they belong
  to.
* Code locations are only in Appendix A, keyed by departure number.
* Departure numbers are labels, not a count; D8, D17, D31 and D48 are not
  used.
* Numbers marked "measured in this review" were computed on 2026-10-30 with
  the module as it stands. Where a corrected rule was tried, it was a
  substitution in a scratch run. Nothing in the repository was changed.

## 0. Summary

The module builds a slope from a ground line, soil layers, water and loads.
It generates trial slip surfaces: circles from a centre grid, circles
through and above the toe, shallow planes parallel to cohesionless faces,
and random or optimised polylines. It cuts each surface into slices and
screens it, then solves it by the Ordinary Method of Slices, Bishop's
simplified method, Janbu's simplified method, Spencer's method or
Morgenstern-Price. It reports the lowest factor of safety with warnings,
tables and figures. The core equations match the original papers: Bishop
(1955), Spencer (1967) and Morgenstern & Price (1965), each checked against
the printed paper (ledger `module_work/wiki_verification/le_method_originals.md`).
On simple published problems the suite agrees with the printed answers to
within about 1 %.

The problems are in what surrounds the equations: the model's weights, the
screens that decide which surfaces count, which method actually scores each
family of surfaces, and several defaults that differ from print. The eight
departures that matter most:

1. **D1. A sloping layer boundary makes the module count soil weight twice.**
   Each layer is weighed from its own flat top, so a lower layer is also
   weighed under the layer above it. On the ACADS 4 benchmark (V-028, B11)
   the slices weigh 2.9 times the soil. The pinned 0.808 (+3.6 % on the
   referee's 0.78) becomes 1.038 on the same surface when each point is
   weighed once (measured in this review). EM 1110-2-1902 p. 4-2 asks the
   engineer to check slice weights first. Every model with a sloping
   boundary is affected: the drawing importer and the project builder
   produce exactly this layout.
2. **D23/D30. Circular searches with Spencer or Morgenstern-Price still
   accept the old approximate solver's value** where the rigorous solution
   fails; noncircular searches refuse such a surface. EM C-10d (p. C-37) and
   UFC 3-220-10 p. 362 treat non-convergence as something the engineer
   must see.
3. **D20. Surfaces with m_alpha below 0.2 are dropped from every search
   without a trace on the result.** Whitman & Bailey (1967, p. 487, Eq. 5),
   the source of the 0.2, prints a warning and an instruction to examine
   the surface, not a rejection. EM C-10b (pp. C-34 to C-36) prints the
   remedies for the steep exit that causes it.
4. **D39. Tension cracks: the default keeps the cracked wedge as driving
   soil, and the crack side defaults to the low-x end.** EM 2-2f (p. 2-6),
   EM C-10c (p. C-36) and the Slide tutorial (p. 16-1) all end the slip
   surface at the crack. ACADS 1(b) reads 1.485 against Slide2's 1.596 (-7 %).
5. **D24. Reverse-curvature circles are closed with a dry crack and
   searched by default.** On #85 this gives Bishop 1.372 against the
   printed 1.531 (Duncan & Wright 1.51). The crack depth comes from the
   circle's geometry, not from the tension test EM C-10c prints.
6. **D38. The rapid-drawdown stages do not follow EM 1110-2-1902 App. G.**
   Stage 1 defaults to a flat pool with no seepage, and stages 2 and 3 are
   merged. The stage-3 drained check uses an OMS normal stress where Eq.
   G-13 (p. G-8) uses the stage-2 normal force. #96 reads 1.273 against the
   printed 1.443.
7. **D42. Families of surfaces are scored by different methods and compared
   as one.** Planes are scored by Janbu, toe circles are screened by Bishop,
   and a noncircular search scores with Spencer when Bishop is named.
   UFC 3-220-10 Table 7-7 (p. 374) and EM 4-3 (p. 4-8) expect one stated
   method.
8. **D2. No cos² correction for a sloping water table.** FHWA NHI-06-088
   Eq. 6-16 (p. 6-14) prints u = h_w γ_w cos²α_w; UFC 3-220-10 p. 355 says
   to correct the water table to a piezometric surface by its slope.

The nine single-case fixes of 2026-10-15 to 10-18 are assessed in §9. Most
of them are sound and in line with print. Two, the m_alpha rejection (GM-127)
and the dry crack on reverse-curvature circles (inherited, GM-59), are
departures to settle. The plan in §11 has ten ordered steps. Step 1 is the
weight error, because every number taken on a sloping boundary depends on
it.

## 1. Printed sources used

From the reference library (`geotech-references/docs`):

* FHWA NHI-06-088, *Soils and Foundations* Vol. I (2006), Ch. 6, pp. 6-1 to
  6-60.
* UFC 3-220-10, *Soil Mechanics* (16 Jan 2026), Ch. 7. Printed page numbers
  are used (the printed page is the PDF page less 28).
* FHWA GEC 11 Vol. II, *MSE Walls and Reinforced Soil Slopes* (Nov 2009),
  Ch. 9, pp. 9-1 to 9-24.
* FHWA GEC 7, *Soil Nail Walls Reference Manual*, Ch. 5, pp. 105 to 120.
* EN 1997-1 (Eurocode 7) §11.5.1.

From the owner's own library (published works, read for this account, not
copied into the repository):

* USACE EM 1110-2-1902, *Slope Stability* (31 Oct 2003): Ch. 2 to 4,
  App. C (methods and their limits) and App. G (rapid drawdown).
* Bishop (1955), Spencer (1967) and Morgenstern & Price (1965): the
  original papers, already checked equation by equation in the ledger
  above.
* Whitman & Bailey (1967), "Use of computers for slope stability
  analysis", JSMFD 93(SM4), pp. 475 to 498.
* Duncan (1996), "State of the art: limit equilibrium and finite-element
  analysis of slopes", J. Geotech. Eng. 122(7), pp. 577 to 596.
* Duncan, Buchignani & De Wet (1987), *An Engineering Manual for Slope
  Stability Studies*, Virginia Tech. The text layer carries no printed page
  numbers, so pages are cited as PDF pages.
* Rocscience, Slide v6 Tutorial Manual: Tutorials 03 (noncircular), 04
  (composite surfaces) and 16 (tension cracks), and the Developer's Tip
  "Active or Passive Support Modeling in Slide".
* In the repository: the Slide2 Verification Manual problem text
  (`module_work/slope_v53/slide2_selected_problems.txt`).

**Not in hand.** Duncan, Wright & Brandon (2014), *Soil Strength and Slope
Stability*, is not in hand. It is cited by the code's comments for the
supports' Methods A and B, Janbu's correction factor, the toe-circle rules
and several defaults; where only that book is cited, this account says "not
found in print". Also not in hand: the Slide2 help pages (m_alpha error
-112, reverse-curvature choices), any SLOPE/W documentation, and Abramson
et al. (2002) Fig. 6.18 (Janbu's f0). The code's statements about Slide2
behaviour that come from the help pages are second-hand here. So is
everything about SLOPE/W.

## 2. Step 1: the model the analysis is given

### What the module does

* **Ground.** A polyline of (x, z) points.
* **Layers.** Each layer has a flat top elevation and either a flat bottom
  or a polyline bottom. The strength at a point comes from the first listed
  layer that contains it. There are four strength models: Mohr-Coulomb
  (drained c′, φ′ or undrained su); SHANSEP, su = S · OCR^m · σ′v with
  defaults S = 0.22 and m = 0.8; Hoek-Brown; and undrained su that varies
  with the base inclination. A layer may also carry ru and the rapid
  drawdown R-envelope.
* **Water.** A piezometric line gives u = γ_w × the vertical depth below it
  at the slice base midpoint. Alternatives are ru on a layer, used where the
  line gives nothing, or a scattered grid of (x, z, u) points interpolated
  linearly. Where the line lies above the ground, the water is a pond: its
  weight is added to the slice and its horizontal thrust is applied.
* **Loads.** Uniform vertical surcharge zones only.
* **Seismic.** A horizontal coefficient kh times the soil weight of each
  slice, acting at the slice's mid-height.
* **Crack water.** ½ γ_w z_w² at z_w/3 on the face of the first retained
  slice.
* **Supports.** Soil nails, anchors, geosynthetic layers, stabilizing piles
  (Ito-Matsui), and a legacy single horizontal force. All are active
  (Method A) by default.

### What the references print

* Total unit weights for all soils, above and below water (EM 1110-2-1902
  2-2d, p. 2-5). One weight per point.
* For total-stress analyses, pore pressure is set to zero (EM 2-2c, p. 2-4;
  Table 2-1, p. 2-2).
* For a sloping water table, u = h_w γ_w cos²α_w (FHWA Eq. 6-16, p. 6-14).
  UFC 3-220-10 p. 355 (7-5.1): "the water table ... corrected to an
  approximate piezometric surface based on the slope".
* All external loads are to be represented, including structures, anchors
  and hawser forces (EM 2-2e, p. 2-6).
* Pseudo-static seismic is a horizontal and/or vertical force at the
  centroid of each slice equal to the coefficient times the total slice
  weight (GEC 11 Vol. II §9.2.8, p. 9-18). UFC 3-220-10 p. 360 applies a
  horizontal force to the free body. EM 2-6 (p. 2-12) defers earthquake
  analysis to another manual.

### Departures

**D1. Soil weighed twice under sloping layer boundaries.**

* *Module:* each layer's weight is taken from the slice base, or the
  layer's bottom, up to the lower of the layer's flat top and the ground.
  Every layer that overlaps the slice height counts. The strength lookup
  uses only the first listed layer that contains the point. A layer's top
  is always flat. The usual way to describe a sloping boundary is to give
  the upper layer a polyline bottom and the next layer a flat top at that
  boundary's highest point (the tests, the drawing importer and the project
  builder all do this). That double-weighs the wedge between the boundary
  and the lower layer's flat top.
* *Print:* one material and one unit weight per point (EM 2-2d, p. 2-5).
  The engineer must check "the weights of slices, shear strength properties,
  and pore water pressures at the bottoms of slices" (EM 4-2d(2), p. 4-2).
* *Consequence (measured in this review):*
  * V-028 (ACADS 4, B11): all three layers are given a top at el. 40, and
    every slice weighs 2.9 to 3.0 times γ·h·b. The pinned surface's sliding
    mass weighs 8,768 kN/m against 3,003 kN/m from its area.
  * On that surface Spencer is 0.808 as pinned and 1.038 with each point
    weighed once. The weak-layer search, same settings, then returns 1.43,
    with 387 of 700 trials not solved.
  * The agreement with the referee's 0.78 rests on the tripled weight, so
    B11 and V-028 do not validate the module.
  * The two-layer test geometry in `test_polyline_layers.py` gives Bishop
    2.488 as coded against 2.849.
  * The error can raise or lower the FOS. Benchmarks with flat layers are
    not affected.

**D2. No cos² correction for a sloping piezometric line.**

* *Module:* full vertical head below the line, everywhere. The code calls
  this conservative (its note SS-3).
* *Print:* FHWA Eq. 6-16, p. 6-14; UFC p. 355.
* *Consequence:* u is overstated under steep water tables, so the FOS is
  low. Not a safety issue, but it differs from print and from programs
  that apply the correction. Folds GM-18 ("no cos²β seepage").

**D3. ru pressure uses the base layer's unit weight over the whole column.**

* *Module:* u = ru × γ(base layer) × column height.
* *Print:* EM C-7 defines ru = u/γz for a uniform slope (p. C-29). No
  printed rule for layered columns was found.
* *Consequence:* in a layered column u differs from ru × the overburden
  stress. Small.

**D4. Pore pressure is subtracted on total-stress layers.**

* *Module:* u below the line acts on every base.
* *Print:* u = 0 for total-stress analyses (EM 2-2c, p. 2-4). Total-stress
  envelopes with φ > 0 apply only to partly saturated soils (Table 2-1
  note 2, p. 2-2).
* *Consequence:* none for φ = 0 layers. A total-stress layer given φ > 0
  would lose strength wrongly.

**D5. Only uniform vertical surcharges.**

* *Module:* no line loads, no inclined loads, no varying loads. Slide2 #9's
  crest load of 20 to 40 kPa is entered as a uniform 30 kPa (B11).
* *Print:* EM 2-2e, p. 2-6.
* *Consequence:* structures, footings and inclined loads cannot be
  modelled as printed.

**D6. Seismic force on soil weight only, horizontal only, at mid-height.**

* *Module:* kh × soil weight, with no inertia from surcharge or ponded water
  and no vertical coefficient.
* *Print:* coefficient × total slice weight, horizontal and/or vertical,
  at the slice centroid (GEC 11 Vol. II p. 9-18).
* *Consequence:* surcharge inertia is left out, which is unconservative
  under heavy surcharge. No vertical coefficient can be given.

**D7. Bishop's method is used with a seismic force and no warning.**

* *Print:* "the suitability of the Simplified Bishop Method for
  pseudo-static earthquake analyses where an additional horizontal force
  is applied is questionable" (EM C-3b, p. C-18).
* *Consequence:* the default search method (Bishop) runs pseudo-static
  cases silently.

**D40. Hoek-Brown and SHANSEP take their stress before the solve.**

* *Module:* Hoek-Brown strength is fixed from an OMS normal before the
  method solves, not from the method's own normal. The SHANSEP and
  Hoek-Brown stresses leave the pond's weight out but take its head into
  u, so σ′v is low under ponded water.
* *Print:* not found in print for use inside a limit equilibrium solve.
  The SHANSEP defaults (S 0.22, m 0.8) were not checked against print for
  this account; UFC 3-220-10 Ch. 8 (Eq. 8-20, Tables 8-7 to 8-9) is where
  to check them.
* *Consequence:* conservative under ponds; Hoek-Brown strength is not
  consistent with the method used.

**D41. Anisotropic su is interpolated with the base angle.**

* *Print:* EM 2-2b(6) (p. 2-4, Fig. 2-2) describes anisotropy. The module's
  interpolation rule was not found in print.

### Backlog items folded here

* GM-18: no cos² seepage (D2).
* GM-140 extra: a `reinforcement_force` given without an elevation is
  silently not applied (§7).
* GM-19: curved (nonlinear) strength envelopes not built. EM 2-2b(3)
  (pp. 2-1 to 2-3, Fig. 2-1) prints them as a standard option. Duncan's
  Ex. 5 is skipped for this reason.

## 3. Step 2: trial surfaces and how they are generated

### What the module does

**Circles from a centre grid.** These are the default: Bishop, a 10 × 10
grid, 30 slices.

* At each centre, 20 radii are scanned, from 0.5 m below the highest ground
  to the distance to the deepest layer's bottom. A golden-section search
  then refines the best radius. One radius is kept per centre.
* Up to three grid minima within 10 % of the lowest are refined by a compass
  search on the centre, stepping 1/2 down to 1/16 of the grid spacing
  (GM-87). The walk moves on the method's own FOS (GM-138). With Spencer,
  Morgenstern-Price or GLE, the walk picks each centre's radius by Bishop's
  method and scores that circle with the method itself; at the walk's end
  the radius is optimised again with the method.

**Automatic ranges (GM-57).**

* A slope face is a run of ground segments steeper than 5°. Face groups
  less than a third as tall as the tallest are dropped.
* Centre heights run from 0.1 H to max(2 H, d) above the highest ground,
  where d is the depth from the toe to the deepest layer's bottom.
* The "reach" beyond the toe and behind the crest is the deepest circle the
  grid can hold.
* The entry window runs from the toe's reach to the face's mid-length, and
  the exit window from the mid-length to the crest's reach. **Every surface
  of the main search must cross the middle of the face.**

**Entry/exit circles.** Arcs through pairs of points in the two windows.

**Supplementary families**, scored after the main search, outside its
windows:

* Circles through the toe of each of the three tallest faces, with the far
  end at 10 distances from the toe and 10 arc depths. With a rigorous
  method these are screened by Bishop, then the best three are re-optimised
  with the method itself.
* Circles passing above the toe, with their lower end on the face at 0.1 to
  0.9 of its height (GM-122, after FHWA p. 6-38).
* Shallow planes parallel to any face where cohesionless soil is at the
  surface (the infinite-slope mechanism), scored by Janbu's simplified
  method whatever method was named.

**Noncircular surfaces.**

* Random polylines (500 trials, 5 points). The inner vertices sit at evenly
  spaced x, each at an independent random depth between 0.5 m and 90 % of
  the depth to the model base.
* Differential evolution (DE), seeded by a short random search, which
  penalises bumps.
* A particle swarm (PSO).
* A weak-layer-biased random search.
* A noncircular search named with Bishop is scored by Spencer. PSO and the
  weak-layer search always use Spencer. Morgenstern-Price and GLE always
  use the half-sine function in a search.

**Bounds.** Surfaces stop at the model base (the deepest layer's bottom),
within a millionth of the model height in searches. They cannot pass the
ends of the ground profile; a check against circles past the end warns
when the governing surface ends there.

### What the references print

* Search with a grid of centres and a range of radii at each, and plot
  contours of the minimum F per centre (FHWA p. 6-22, Fig. 6-13 p. 6-23).
  Locate the critical circle by families (circles tangent to one elevation,
  or circles through the toe), contour F, and make sure the contours
  enclose the minimum (Duncan, Buchignani & De Wet 1987, PDF pp. 75 to 79).
* Several local minima can exist; check multiple centres (FHWA p. 6-22,
  point 1). Search all areas of the slope (p. 6-23, point 2). Thin weak
  zones need software that simulates those failures (point 3).
* The search must not jump from one face to the other. Run several
  independently started searches, or a suite of centres with varied radii
  (EM 4-2e(1), pp. 4-3 to 4-7).
* For noncircular searches, the initial surface should correspond to the
  estimated critical surface. Steep exits at the toe need a restriction.
  The final shifts should be no more than 10 to 25 % of the thinnest
  stratum's thickness (EM 4-2e(2), p. 4-7).
* Unreinforced stability for reinforced soil slopes: use circular-arc and
  sliding-wedge surfaces through the toe, through the face at several
  elevations, and deep below the toe (GEC 11 Vol. II §9.2.5, p. 9-6).
* Spencer and Morgenstern-Price with "very robust search routines" for
  circular and noncircular surfaces (UFC p. 352). Search face exits,
  circular and noncircular, then compound and global surfaces (Table 7-7,
  p. 374).
* Noncircular search approaches: random generators of kinematically
  admissible surfaces, dynamic programming, moving one point at a time, and
  simplex (Duncan 1996, p. 581). Unless geology constrains it, the critical
  surface is essentially circular (same page).
* Slide's noncircular search for a weak layer generates two points on a
  polyline drawn in the layer and projects the surface up to the ground at
  set angles (Slide v6 Tutorial 03, pp. 3-3 to 3-5). Its composite option
  lets a circle follow the top of impenetrable material (Tutorial 04,
  pp. 4-1 and 4-3).
* For homogeneous soil, circular surfaces; in layered soils, attention to
  weaker layers with noncircular surfaces (EC7 §11.5.1).

### Departures

**D9. The main search covers only surfaces that cross the middle of a face.**

* *Module:* surfaces confined to the upper or lower part of a face are
  reached only through the supplementary families.
* *Print:* GEC 11 p. 9-6 (face at several elevations); FHWA p. 6-23 point 2.
* *Consequence:* the weak-layer-over-rock cut reads 4.82 against about 0.9
  to 1.05 unless the windows are given (B14, B16). The module says where
  its ranges ran. Under the owner's rule (feedback "judgment not
  hard-coded") the agent directs the search; EV-25 measures that. Whether
  the default should still exclude these surfaces is an owner decision.

**D10. The face definition and centre heights are partly not in print.**

* *Module:* faces are steeper than 5°; groups under one third of the tallest
  are dropped; centre heights run from 0.1 H to max(2 H, d).
* *Print:* the upper centre height follows Janbu's charts (FHWA Figs. 6-16
  and 6-19). The 5°, the one-third and the 0.1 H were not found in print.

**D11. One radius per centre, from a 20-radius scan.**

* *Print:* a range of radii at each centre (FHWA p. 6-22).
* *Consequence:* a second minimum in radius at the same centre (a shallow
  and a deep mechanism) is not kept. Small for most slopes.

**D12. The random noncircular generator draws every inner vertex
independently.**

* *Module:* no concave-up shape is required, the generator is not seeded
  from the critical circle or from a weak layer, and the depth is measured
  to the model base.
* *Print:* EM 4-2e(2), p. 4-7; Duncan 1996, p. 581; Slide Tutorial 03,
  pp. 3-3 to 3-5.
* *Consequence:*
  * Most trials are refused: 427 of 700 on V-028.
  * Results drift with the seed and the number of trials (V-043, 2,000
    trials, seeds 1 to 5: 1.149 to 1.270).
  * A noncircular result can end above the best circle: V-043 DE 1.234
    against the circle's 1.186 to 1.213 (B15, B17). Duncan 1996 p. 581
    implies a free search should reach at least the circle, so a result
    above it is a search failure.
  * Folds GM-117 in full.

**D13. The scoring method differs from the one named.**

* *Module:* Bishop named means Spencer for noncircular surfaces; PSO and the
  weak-layer search always use Spencer; Morgenstern-Price and GLE in a
  search always use the half-sine function; the search has no choice of
  interslice function.
* *Consequence:* the result states the method used, but an engineer who
  asks for GLE with a constant function gets half-sine in a search.

**D14. No impenetrable material and no composite surfaces.**

* *Print:* Slide Tutorial 04, pp. 4-1 and 4-3. FHWA p. 6-40 (circle tangent
  to the top of the firm layer); Taylor's depth factor D, the firm layer
  "through which the failure circle does not pass" (FHWA pp. 6-26 to 6-29).
* *Consequence:* only the model base limits depth. A firm layer above the
  base must be represented by strength alone, and circles enter it.

**D15. Polylines must have increasing x, and no surface may pass the
profile end.**

* *Module:* a noncircular mechanism that would carry on past the profile
  end is checked only through a circle (GM-89).
* *Print:* EM C-10e(2) (p. C-38) warns about near-vertical boundaries.
* *Consequence:* small, and the end is warned.

### Backlog items folded here

* GM-117, all of it: few-trial drift; GLE seldom converges on a thin weak
  layer; the slicing-agreement check does little; the 2 % axis agreement
  is a rule of thumb; generators waste trials; DE depth is measured to the
  model base; seed the noncircular search from the critical circle; V-043
  DE ends above the circle; a noncircular result should never end above
  the best circle.
* GM-89: noncircular past the profile end judged through a circle proxy;
  Duncan Ex. 1 noncircular 0.603 / 0.683 against Taylor 0.706; no
  refinement of entry and exit.
* GM-126 (parked): score each sliding mass of a multi-stretch surface.
* GM-140 extras:
  * In one of the agent's live tests (a clay face over a weak layer), the
    weak-layer window missed the clay face, and DE found nothing.
  * Impenetrable material and composite surfaces are not modelled (D14).

## 4. Step 3: slicing

### What the module does

* Slice boundaries go at ground breaks, layer tops and bottoms (not the
  model base), water-table crossings, the crack base, surcharge edges and
  polyline corners. Circles are cut wherever they cross any of these
  (GM-131, GM-134).
* Boundaries closer than 1/10,000 of the span are merged. The rest of the
  slices are shared out by width.
* Each base is the chord between its ends, with α and length from the
  chord.
* Weight is taken from the column at the slice mid-point (D1 applies).
  Strength comes from the layer at the base mid-point; u from the base
  mid-point.
* A slice with no height is dropped.
* A surface below the model base by more than 1/1,000 of the model height
  is refused (one given surface) or left out (searches, at 1/1,000,000).
  The part below the base counts as empty in the weight (GM-128, an owner
  call).
* The circle's entry and exit points are found by sampling 200 points
  along it, then bisection to 1e-6.

### What the references print

* 10 to 15 slices for hand work.
* Slice sides placed so each base lies in one layer, and at the water-table
  and surface intersections.
* Tops at breaks in slope. Widths need not be equal (FHWA p. 6-14).
* "Take l as the secant to the arc" (FHWA p. 6-16).
* Slices split at the phreatic surface and at layer boundaries (UFC p. 353,
  slices 2A and 6A).
* Avoid surfaces that cross boundaries at very flat angles, for numerical
  reasons (EM C-10e(3), pp. C-38 to C-39).

The module's slicing is in line with this print.

### Departures

**D16. The pond edge is not a slice boundary.**

* *Module:* where the water table meets the ground there is no boundary.
  In rapid drawdown, stage-3 slices are matched to stage-1 slices by
  nearest mid-point (GM-135).
* *Print:* boundaries at the water surface intersections (FHWA p. 6-14).
* *Consequence:* GM-135 records a planar slide partly under the pool at
  Spencer 1.21 against 1.62.

**D18. The 200-point entry/exit sampling can miss a crossing.**

* *Module:* a crossing near the vertical end of a circle can be missed
  (GM-140 extra).
* *Print:* EM C-10e(3), pp. C-38 to C-39.
* *Consequence:* rare wrong entry or exit points.

**D19. The model-base limits differ between families and are not in
print.**

* *Module:* tolerances of 1/1,000 of the model height for a given surface
  and 1/1,000,000 in searches. The toe-circle family uses a different
  below-base test. The part below the base weighs nothing.
* *Print:* not found in print (the tolerance was an owner call).
* *Consequence:* the families admit slightly different surfaces.

### Backlog items folded here

* GM-135 (D16).
* GM-140 extras (D18, D19).
* GM-89: the sliver tolerance can let the grid return a circle that drops
  below-base end slices. GM-140 asks whether GM-128 supersedes this; it
  appears to for searches, since searches now refuse below-base surfaces.

## 5. Step 4: admitting or refusing a trial surface

### What the module does

**Every surface.**

* A surface that leaves the ground and comes back (more than one stretch
  below ground) is refused (GM-122).
* A FOS below 0.05, or not a number, is refused.
* In searches, a surface with m_alpha = cos α (1 + tan α tan φ′/F) below 0.2
  on any slice with φ′ > 0 is refused and counted (GM-127). Slices with
  φ′ = 0 are exempt by default; "every slice" is an option.
* A given surface is analysed, and the result says when m_alpha is low.

**Noncircular surfaces only.**

* At least max(5, n/3) slices.
* A span of at least half the window span, or half the slope height.
* The rigorous solution must converge.
* A FOS below 1.5 on a jagged surface is refused. Jagged means a segment
  steeper than 80°, a backward step, or an angle that drops more than 25°
  below the steepest so far.
* A surface that would become the critical one must also solve about two
  of three moment axes within 2 %, and at one slice more or twice the
  slices within 50 %.

**Circles.**

* With supports, the rigorous methods must converge.
* Without supports, Spencer and Morgenstern-Price fall back to the old
  approximate solver (D23).
* Reverse-curvature circles are closed with a dry crack. "Discard" is an
  option.

### What the references print

* Calculated normal forces and interslice forces should be examined. No
  tension on slice sides or bases. The line of thrust inside the sliding
  mass (EM C-10a, pp. C-33 to C-34; UFC pp. 361 to 362).
* A steep exit at the toe gives very large or negative forces. The exit's
  inclination for passive conditions is α = −(45° − φ′/2) (Eq. C-32).
* Remedies, in the order printed (EM C-10b, pp. C-34 to C-36):
  1. flatten the toe of the surface (the best, needs noncircular surfaces);
  2. change the side-force inclination;
  3. use the OMS;
  4. give the exit zone a passive strength as c with φ = 0 (the only remedy
     with Bishop's method).
* Bishop's method "must always be done in such a way that the user is
  warned if, for any slice, the numerator becomes negative or if the
  denominator becomes negative or small; for example, if cos θ + tan φ sin
  θ / F < 0.2 ... the user must examine the result in detail" (Whitman &
  Bailey 1967, p. 487, Eq. 5).
* If Bishop's F is below the OMS F for the same circle, Bishop's solution
  has numerical problems and the OMS value is the better answer. Calculate
  the OMS for each circle when Bishop's method is used (Duncan 1996,
  p. 580).
* Non-convergence: reject the unrealistic one of multiple solutions, use a
  sensible starting F, and use a realistic initial surface (EM C-10d,
  pp. C-37 to C-38). "Non-convergence ... can cause critical surfaces to
  be missed"; the engineer should identify convergence issues (UFC p. 362).

### Departures

**D20. m_alpha below 0.2 is a silent refusal in searches.**

* *Module:* the surface is left out and counted, but the result does not
  say where the low m_alpha was. φ′ = 0 slices are exempt by default. The
  check against the OMS is not made.
* *Print:* warn and examine, for any slice (Whitman & Bailey p. 487); fix
  the exit (EM C-10b); compare with the OMS (Duncan 1996 p. 580). The
  φ′ = 0 exemption is reasoned in the code (on such a slice m_alpha = cos α
  does not change F) but was not found in print. Slide2's rejection
  (error -112) is from its help pages, not in hand.
* *Consequence:* steep-exit circles are removed instead of corrected, so
  the critical surface can move. #85, Bishop, discard: 1.508 with the
  exemption against 1.511 checking every slice (B16). Folds GM-130 (the
  m_alpha switch).

**D21. Bishop's numerator is clamped without a warning on the result.**

* *Module:* where u·b exceeds W, the slice's frictional term is set to
  zero. The calculation steps state the rule; the result does not say
  which slices it applied to.
* *Print:* Bishop (1955) Eq. 13, p. 10 has no clamp. Whitman & Bailey p. 487
  ask for a warning when the numerator is negative.
* *Consequence:* artesian slices pass without notice.

**D22. The noncircular screens are numerical stand-ins for the printed
checks.**

* *Module:* span, slice count, jaggedness, moment-axis agreement and
  slicing agreement are checked. Compression on bases and sides, and the
  line of thrust, are not.
* *Print:* EM C-10a; UFC p. 362.
* *Consequence:* sound surfaces can be refused and unsound ones kept. The
  50 % slicing check does little (GM-117). Folds GM-56: Spencer and GLE
  refuse thin planar polylines; Janbu f0 1.0025 there.

**D23. Circular searches accept the old approximate solver's value.**

* *Module:* without supports, a Spencer or Morgenstern-Price circle whose
  rigorous solution fails is scored by the old approximate solver.
  Noncircular searches refuse such a surface.
* *Print:* EM C-10d, p. C-37; UFC p. 362.
* *Consequence:*
  * On #85 without its support, a 6 × 6 Spencer search called the old
    solver 415 times (GM-81).
  * A single odd circle can report a false Spencer FOS near 0.05 (GM-139).
  * The two kinds of surface are treated differently.

**D24. Reverse-curvature circles are searched, closed by a dry crack.**

* *Module:* the crack runs from the centre's level to the ground.
* *Print:* a crack's depth is set to remove tension, by the minimum-F depth
  or by Rankine's depth with developed strengths (EM C-10c, p. C-36;
  EM 2-2f(4) and (5), pp. 2-6 to 2-9). No printed source for a crack set by
  the circle's geometry was found. Slide2's default is from its help
  pages, not in hand.
* *Consequence:* #85 with its support: Bishop 1.372 on such a circle,
  against the printed 1.531 (D&W 1.51). With "discard": 1.508.

**D25 (not a departure). Multi-stretch surfaces are refused (GM-122).**
This matches Slide2's documented behaviour as quoted in the code; no
independent printed source is in hand. Scoring each mass separately is
GM-126, parked.

### Backlog items folded here

* GM-130: the m_alpha switch; drawdown stage 1 not checked; the
  circle-above-toe Bishop screen moved Janbu from 1.033 to 1.138.
* GM-139 (D23).
* GM-81, the old λ search on unsupported runs (D23).
* GM-56 (D22).
* GM-18: a degenerate free Spencer solution on ponded V-044.

## 6. Step 5: the methods

### What the module does

**Ordinary Method of Slices (OMS).**

* N′ = W cos α − u l, clamped at zero.
* F = Σ[c′l + N′ tan φ′] / driving.
* Driving on a circle is the moment arm form ΣW(x − x_c)/R plus the pond,
  seismic and crack-water moments; on a polyline, ΣW sin α plus those
  forces.
* Matches Bishop (1955) Eq. 9, p. 10 and Spencer (1967) Eq. 4, p. 14,
  except for the clamp. On a polyline the OMS is applied as a force sum;
  EM C-2b (p. C-16) says the OMS "can only be used with circular slip
  surfaces".

**Bishop's simplified method.**

* m_alpha = cos α + sin α tan φ′/F (Bishop 1955 Eq. 12, p. 10; UFC Fig. 7-4,
  p. 353).
* F = Σ[(c′b + (W − ub) tan φ′)/m_alpha] / ΣW sin α (Eq. 13 with
  X_n − X_{n+1} = 0, p. 11).
* The sign of α is taken positive in the sliding direction (the code's fix
  SS-4; UFC p. 353 prints the sign rule).
* The starting F is the OMS value. The tolerance is 1e-4 and the limit is
  50 iterations; if it does not converge, the last value is returned with a
  program warning only.
* Circles only: a polyline named for Bishop runs Spencer with a warning.

**Janbu's simplified method.**

* Force equilibrium at λ = 0 in the GLE engine.
* Corrected by f0 = 1 + b1 (d/L − 1.4 (d/L)²), with b1 = 0.69 (c only),
  0.31 (φ only) or 0.50 (both). The soil type is taken from whether any
  slice has c and any has φ.

**Spencer, Morgenstern-Price and GLE.** A Fredlund-Krahn discrete GLE:

* X = λ f(x) E (Morgenstern & Price 1965 Eq. 18, p. 85). Normals come from
  vertical slice equilibrium, E is marched across the slices, and F_m and
  F_f are found separately. λ is taken where they cross (Spencer 1967
  Eqs. 7 and 8, p. 15).
* f(x): constant (Spencer), half-sine, half-sine clipped at 0.5, or
  trapezoidal.
* λ is searched from −1.5 to 1.5 in steps of 0.1, then by bisection.
* "Converged" means |F_m − F_f| < 0.001. The reported F is their mean.
* m_alpha is clamped at 0.05, with a warning. The base effective normal is
  clamped at zero in both equations.
* On a polyline, the moment axis is a circle fitted through the base
  mid-points, falling back to a point above the chord.

**The old approximate Spencer and Morgenstern-Price solvers.**

* Kept as a fallback. They use Spencer's shifted m_alpha kernel (Eq. 5,
  p. 14) with an approximate split between force and moment, not the
  slice-by-slice recursion (the code's own note SS-1).

**Other calculations.**

* Infinite-slope closed form.
* Rapid drawdown (§7).
* Newmark sliding-block displacements, with the yield coefficient found
  by bisection on kh, and the Jibson (2007) and Bray & Travasarou (2007)
  regressions.
* FOSM and Monte Carlo reliability on a fixed surface (B4).

### What the references print

* Complete methods (Spencer, Morgenstern-Price) are more accurate and
  preferred. The 2003 EM's criteria are to be used with methods that
  satisfy all conditions of equilibrium (EM transmittal para. 1; C-1e,
  pp. C-5 to C-6; C-11, p. C-39).
* Bishop's method is acceptable for circles; check major designs with
  Spencer (EM C-3c, p. C-18). It is within about 5 % of rigorous methods
  (same page) and within about 7 % (Whitman & Bailey Table 2, p. 486).
* Complete methods differ by no more than about 12 % from each other and
  about 6 % from the correct answer (Duncan 1996, p. 580).
* Spencer's θ is solved; Wright (1969) extended the method to noncircular
  surfaces (EM C-5a, p. C-26).
* Check Spencer by the force-equilibrium procedure with Spencer's θ
  (EM C-5d, pp. C-26 to C-27).
* The OMS differs by up to 20 % and more with high pore pressures; it is
  equal to rigorous methods when φ = 0 (EM C-2b, p. C-16).
* Bishop is the minimum method; use Spencer where available (FHWA Table
  6-1, p. 6-21). Do not use φ to represent strength increasing with depth
  (same table).

### Departures

**D26. The OMS is offered on polylines.**

* *Print:* EM C-2b, p. C-16 (circles only).
* *Consequence:* small. The OMS is a check method, but on a polyline it is
  not the printed method.

**D27. A Bishop solution that does not converge is reported as a number.**

* *Module:* only a program warning is raised; the result does not carry it
  (GM-81).
* *Print:* EM C-10d.
* *Consequence:* an unconverged value can be reported or win a search.

**D28. Janbu's f0 is not found in print.**

* *Module:* the f0 formula and the b1 values come from Abramson et al.
  Fig. 6.18 (not in hand). The EM uses the Corps' Modified Swedish method
  instead (C-4).
* *Consequence:* the correction is unchecked. F&K print Janbu 2.041 against
  the module's 2.017 (B1). It also scores the planar family (D42).

**D29. The GLE settings are not found in print.**

* *Module:* the λ range of ±1.5, the 0.001 tolerance, the 0.05 m_alpha clamp
  and the fitted moment axis.
* *Print:* the method itself is in Spencer pp. 14 to 15 and M-P p. 85; the
  settings are not.
* *Consequence:*
  * On #85 the GLE critical sits at the edge of the λ range: 1.609 against
    Slide2's 1.575 (GM-82).
  * λ is poorly determined on φ = 0 circles, and the solve is slower
    (GM-137).
  * "Base tension" is warned but not acted on; EM C-10c's remedy is a
    tension crack.

**D30. The old approximate solvers are still used.**

* *Module:* their values reach single-surface results (labelled since
  GM-81), circular searches (D23) and the Newmark yield coefficient
  (GM-81).
* *Print:* EM C-10d.

**D32. Methods not built.**

* Corps Modified Swedish / force equilibrium with an assumed side-force
  inclination: EM C-4 is the Corps' printed check method, and C-5d uses it
  to check Spencer.
* The wedge method (EM C-6, pp. C-27 to C-28).
* Janbu's rigorous method, Sarma's method, Barton-Bandis joints,
  liquefied strength, curved strength envelopes, and Hassiotis or Kumar &
  Hall piles (GM-19).
* *Consequence:* the hand check EM 4-2a (pp. 4-1 to 4-2) asks for is not
  available inside the module.

### Backlog items folded here

* GM-19 (D32).
* GM-137 (D29).
* GM-82: the GLE λ bound (D29).
* GM-136: V-030 sits 5.3 % above Slide2 (1.218 against 1.157; Duncan 1.17).
  Spencer, GLE and Janbu are high together, so the input is the likely
  cause.
* GM-12 (parked): run time of rigorous searches.

## 7. Step 6: water, loads, seismic and supports inside the solve

### What the module does

**Water.**

* Ponded water acts as vertical weight on the slice and as a horizontal
  thrust, ½ γ_w (d_left² − d_right²), at its own height. This is the
  printed way to apply boundary water forces with total weights (FHWA
  p. 6-13; EM 2-2d and 2-2e, pp. 2-5 to 2-6).

**Driving side.** The magnitudes of the gravity, seismic and crack-water
terms are added, so the seismic term always drives.

**Supports.** Each support's force acts at the base of the slice it crosses
(Duncan 1996, p. 583: the point of application along the reinforcement has
no significant effect).

* Bishop: the moment, plus the vertical component in that slice's vertical
  equilibrium (Wright & Duncan 1991, as summarised by Duncan 1996, p. 583).
* OMS on circles: the moment, plus the component normal to the base.
* GLE methods: every component in every equation.
* Active, the default, is Method A: the allowable force, not divided by F.
  Passive is Method B: the ultimate force, divided by F.
* Nails: min(pullout behind the surface, bar tension). Anchors: the full
  allowable force when the bond zone lies beyond the surface. Geosynthetics:
  the full allowable force wherever a layer crosses the surface.
* Stabilizing piles: the Ito-Matsui force.

**Tension cracks.**

* The depth is given by the user and the crack sits on the low-x end unless
  "exit" is chosen.
* Model "strength" (the default) keeps the cracked wedge as zero-strength
  driving soil. "Truncation" removes it.

**Rapid drawdown** (Corps 2-stage, Lowe-Karafiath, Duncan-Wright-Wong
3-stage).

* Stage 1 is a GLE solve at full pool. By default the pool is flat and the
  pressures hydrostatic, with no seepage; a seepage line can be given.
* Stage 2 takes the undrained strength from Eq. G-12 with Kf from Eq. G-7
  or G-8.
* Stage 3 first compares the drained strength with the undrained one,
  slice by slice. The drained strength is taken at an OMS normal stress on
  the drawn-down slices (default) or at a drained GLE solve of the
  drawn-down slope. Then one solve is made with the lower strength on each
  slice.

### What the references print

**Supports.**

* Active supports decrease the driving force, passive ones increase the
  resisting force. The choice is "somewhat arbitrary". Passive gives a
  lower F for the same capacity. Active is Method A (allowable force) and
  passive is Method B (ultimate force). Rocscience recommends active, for
  better convergence (Developer's Tip, pp. 2, 4 and 5).
* GEC 11's Eq. 9-1 applies the safety factor to the reinforcement. For
  programs where reinforcement reduces the driving moment, divide Tal by
  FSR (p. 9-24).
* Duncan 1996 p. 583: the factor of safety can go on the reinforcement load
  or on the soil strength, and need not be the same for both.
* A reinforcement layer's capacity is the smaller of its allowable pullout
  behind the surface and Tal (GEC 11 §9.1, p. 9-1). Only reinforcement that
  extends 3 ft (1 m) beyond the surface counts (§9.2.6, p. 9-14).
* A nail's load is Tmax = min(R_T, R_F, R_PO), behind, in the bar, and in
  front (GEC 7 Eq. 5.2, p. 105). Wedge equilibrium with a nail: Eqs. 5.6 to
  5.9, pp. 119 to 120.

**Tension cracks.**

* "The soil upslope from the crack ... is then ignored in the stability
  computations ... by terminating the slices near the crest of the slope
  with a slice having a vertical boundary" (EM 2-2f(3), p. 2-6). Water in
  the crack acts on that boundary.
* The crack depth is the one giving the minimum F, or Rankine's depth with
  developed parameters c_D = c/F and tan φ_D = tan φ/F (EM 2-2f(4) and (5),
  pp. 2-6 to 2-9; C-10c, Eq. C-36, p. C-36).
* The crack is not deeper than the embankment on a weak foundation (C-10c).
* "A tension crack essentially terminates the slip surface" (Slide v6
  Tutorial 16, p. 16-1).

**Rapid drawdown** (EM App. G, G-3, pp. G-2 to G-9; Table 2-1, p. 2-2).

* First-stage pore pressures are steady-state seepage pressures.
* Stage 1 gives σ′_c = N/l and τ_c = S/l (Eqs. G-1, G-2).
* Stage 2 is solved with the Eq. G-12 strengths and φ = 0.
* The drained check uses σ = N/l **from the second-stage computation**
  (Eq. G-13), with u after drawdown (G-14) and s_d = c′ + σ′_d tan φ′
  (G-15).
* Stage 3 is solved only if some slice's drained strength is lower,
  "using the same conditions as for the second-stage computations" except
  on those slices (p. G-9).
* Kf follows Eq. G-7 for c′ = 0 and Eq. G-8 for c′ > 0 (the ledger; the
  module was fixed to G-8).

### Departures

**D33 (note, not a departure). Supports default to active.** This matches
the Rocscience recommendation and Duncan & Wright's preference as quoted
there. The engineer must enter the allowable force. The result should say
which convention was used (it does).

**D34. Geosynthetics take their full allowable force wherever they cross.**

* *Module:* no pullout check behind the surface and no minimum embedment.
* *Print:* GEC 11 pp. 9-1 and 9-14.
* *Consequence:* unconservative for surfaces near the ends of the layers.
  Folds GM-82 (the geosynthetic's first crossing and its direction).

**D35. A nail has no facing or front-pullout term.**

* *Print:* GEC 7 Eq. 5.2, p. 105.
* *Consequence:* the head's capacity never limits the nail. Small.

**D36. Support directions in some cases are not settled (GM-82).**

* *Module:* the "normal" direction of a pile is unresolved; the OMS on a
  polyline leaves the support's normal component out of N; Bishop counts
  little of a support crossing near the centre's level (warned, as printed
  in Duncan 1996 p. 583's terms).
* A support whose pull assists sliding: the owner decided on 2026-10-20
  (OWN-59) to offer both readings with slack as the stated default. This
  is not built yet; it belongs to this review.

**D37. A legacy horizontal force with no elevation is silently ignored.**

* *Consequence:* the input is lost without notice (GM-140 extra).

**D38. The rapid-drawdown stages are not as printed.**

* *Module:*
  * Stage 1 defaults to a flat pool with no seepage.
  * Stages 2 and 3 are merged: the substitution is decided before any
    solve.
  * The drained check uses an OMS normal (default) or a drained GLE normal,
    not the stage-2 N of Eq. G-13.
  * If stage 1 does not converge, it falls back to OMS estimates (warned).
* *Print:* EM Table 2-1 (p. 2-2); G-3d and G-3e (pp. G-8 to G-9).
* *Consequence:*
  * #96: 1.273 with the default, 1.370 with "gle", against the printed
    1.443.
  * #98 search: 0.837 against 0.931 (Corps) and 0.938 against 1.039 (DWW),
    on a recovered section that is about 10 % low.
  * OWN-17 (the stage-3 default) is still open.
  * GM-15: the steep-φ′ Kc overshoot of about 22 % at φ′ = 45° (V-052b).

**D39. Tension cracks are not as printed.**

* *Module:*
  1. the depth is given by the user only, with no Rankine estimate and no
     minimum-F scan;
  2. the default model keeps the wedge as driving soil;
  3. the side is a flag that defaults to the low-x end, not taken from
     where the crest is.
* *Print:* EM 2-2f (pp. 2-6 to 2-9); C-10c (p. C-36); Slide Tutorial 16
  (p. 16-1).
* *Consequence:*
  * ACADS 1(b): 1.485, 1.486 and 1.482 (Bishop, Spencer, GLE) with the
    default model, against Slide2's 1.596, 1.592 and 1.592 (-7 %).
    Truncation gives 1.603, 1.599 and 1.600 (B16).
  * A crack on a slope that rises towards high x lands on the toe side
    unless "exit" is set.
  * Folds GM-18 (the water-filled crack in V-043's clay crest).

### Backlog items folded here

* GM-82 (D34, D36), including the short-arm R/2 warning, which is a choice.
* OWN-59 (D36).
* GM-18: the single anchor in #85, the drawdown search hot spot, and
  Slide2 #33.
* GM-15 and OWN-17 (D38).
* GM-130: drawdown stage 1 not checked.
* OWN-13 / SPL-3: Ito-Matsui as its own method.

## 8. Step 7: choosing the critical surface

### What the module does

* The lowest F over the main search and the supplementary families is
  reported.
* Each family is scored as described in §3: circles with the method named;
  toe circles screened by Bishop when a rigorous method is named;
  shallow planes by Janbu's simplified method; noncircular surfaces by
  Spencer when Bishop is named.
* An optional Bishop screen is available for the main circular search;
  it is off by default (GM-123).
* The agent's tools set a 180 s time budget; the library sets none. A
  search that is cut short reports the lowest surface it found and says
  what it did not score.
* Warnings say when the critical surface lies on the edge of a window or of
  the centre grid, or at the end of the profile.

### What the references print

* State the method used (EM 4-3b(9)(a), p. 4-8). Use one method,
  Spencer or Morgenstern-Price, for all surfaces (UFC Table 7-7, p. 374).
* Run several independently started searches (EM 4-2e(1)(b), p. 4-3), or a
  suite of centres and radii examined by contours (4-2e(1)(c), p. 4-7;
  Duncan et al. 1987, PDF p. 75).

### Departures

**D42. Families are compared across different methods.**

* *Print:* UFC p. 374; EM p. 4-8.
* *Consequence:* a family can win because its method differs (Janbu planes
  against Spencer circles), not because its mechanism is more critical.
  The result names the governing family and the method of each, but the
  comparison is not like with like.

**D43. One search start.**

* *Module:* one grid; refinement only of the grid's own minima.
* *Print:* EM 4-2e(1)(b), p. 4-3; Duncan 1996 p. 581; FHWA p. 6-22.
* *Consequence:* a separate local minimum outside the grid's best region
  can be missed. The edge warnings cover only part of this.

**D44. The time budget is not in print.**

* *Consequence:* a result can depend on machine speed when a search is cut
  short. It is stated in the result (GM-123).

### Backlog items folded here

* GM-117: a noncircular result should never end above the best circle.
* GM-123: the time budget.
* GM-130: the Bishop screen.

## 9. Step 8: what is reported

### What the module does

**Numbers, for one surface:**

* the FOS;
* the method, whether it solved, and what was used instead;
* the comparison values (OMS, Bishop, Spencer, M-P, Janbu corrected and
  uncorrected);
* λ or θ;
* the entry and exit points, and the circle's centre and radius or the
  polyline;
* a slice table (weight, base, strength, u, N, interslice E and X);
* the line of thrust, for rigorous methods;
* warnings.

**Numbers, for a search:**

* the number of surfaces evaluated and refused, by reason;
* each family's minimum and the governing family;
* the ranges used, and the method each family was scored with;
* the trial surfaces.

**Figures:** a cross-section with slices, the line of thrust and supports;
a map of up to 200 trial surfaces coloured by F; slice forces; interslice
forces; the Monte Carlo histogram; and the FOSM tornado.

### What the references print

* Show the critical surface and the other significant surfaces with their
  F; for circles, the centre's coordinates and the radius (EM 4-3b(9)(b),
  p. 4-8).
* For automatic searches, the output should designate the most critical
  surface and what other surfaces were analysed (EM 4-2d(2), p. 4-2).
* Program features: "ordered output and plotting capability for the
  failure surface of 10 minimum safety factors" (FHWA p. 6-49).
* A contour plot of minimum F over the trial centres (FHWA Fig. 6-13,
  p. 6-23). For reinforced slopes, "a contour plot of lowest safety factors
  about trial failure circle centroids is recommended" (GEC 11 p. 9-24).
* Examine slice forces, compression, and the line of thrust (UFC p. 362;
  EM C-10a).

### Departures

**D46. No contour plot of minimum F over the centres.**

* *Module:* the per-centre minima are stored but not contoured.
* *Print:* FHWA p. 6-23; GEC 11 p. 9-24; Duncan et al. 1987, Fig. 28.
* *Consequence:* the engineer cannot see whether the minimum is enclosed.

**D47. No ordered list of the lowest surfaces.**

* *Module:* family minima and a list of trial surfaces only.
* *Print:* FHWA p. 6-49; EM 4-3b(9)(b), p. 4-8.

**D50. The line of thrust is drawn clipped to the sliding mass.**

* *Module:* the clip is for display. Nothing says when the line leaves the
  mass, and nothing checks the interslice forces for tension.
* *Print:* EM C-10a; UFC p. 362.
* *Consequence:* the printed acceptability check cannot be made from the
  module's figure.

**D49. Labels and input checks.**

* An unknown method name runs Bishop without saying so.
* A Bishop solution that does not converge is not in the result (D27).
* Toe-circle and planar results are labelled "Gle" or "Morgenstern_price"
  (GM-81).
* The probabilistic analyses take the 999.9 sentinel as a FOS in places
  (GM-81).

**D45. Calculation-package inputs.**

* `time_budget_s`, `tol` and `FOS_required` change nothing.
* `f_interslice` accepts any string (GM-140 extras).
* The interslice-force figure prints at 4.0 pt, and the figure note and
  key overlap.

### Backlog items folded here

* GM-81: the rest.
* GM-132: the slope part of `applies_when` and `defaults_used`.
* GM-18: DXF export of the slip surface.
* GM-140 extras (D45, D49). A polyline named for Bishop now runs Spencer
  with a warning; this was a GM-140 extra and is already done.

## 10. The nine single-case fixes (2026-10-15 to 10-18)

One entry each: what it does, its printed basis, and a verdict.

* **GM-87, compass refinement of the grid minima.** The grid spacing alone
  missed Taylor's circle (0.739 against 0.706). *Basis:* FHWA p. 6-22 and
  Duncan et al. 1987 (contours enclosing the minimum). *Verdict:* sound;
  keep. It is the local form of the printed contouring. Add the contour
  plot (D46).
* **GM-122, refuse multi-stretch surfaces, and the circles-above-toe
  family.** *Basis:* FHWA p. 6-38 for the family; the refusal follows
  Slide2, which is not in print here. *Verdict:* keep both. The family is
  a stop-gap for D9 and should become part of a search that covers face
  exits at all heights (GEC 11 p. 9-6).
* **GM-123, time budget, coarse-then-fine grid, optional Bishop screen.**
  *Basis:* none in print. *Verdict:* keep as run-time engineering, but the
  result must keep saying what was cut.
* **GM-124, windows given the wrong way round are swapped.** *Verdict:*
  input handling; keep.
* **GM-127, m_alpha < 0.2 refusal with φ′ = 0 slices exempt, a check on
  twice the slices, and a graded penalty in DE.** *Basis:* the 0.2 is
  Whitman & Bailey p. 487, but printed as a warning. *Verdict:* a departure
  (D20). The owner should decide: warn and correct, as printed, or reject,
  as Slide2 does.
* **GM-128, base depth measured on the surface, with a tolerance of 1/1,000
  of the model height.** *Basis:* the owner's call. *Verdict:* keep. Make
  the toe family use the same test (D19).
* **GM-131, polylines sliced at their corners with chord bases.** *Basis:*
  FHWA p. 6-14 and p. 6-16. *Verdict:* sound; keep. It removed the slicing
  artefacts behind several earlier fixes.
* **GM-134, circles cut at every crossing, with chord bases.** *Basis:*
  FHWA p. 6-14 and UFC p. 353. *Verdict:* sound; keep.
* **GM-138, the refinement walk scored by the method's own F.** V-043
  Spencer moved from 1.2375 to 1.1859. *Basis:* the search must minimise
  the method's F. *Verdict:* sound; keep. The same principle argues against
  D42.

Related earlier work: GM-57 (ranges from the slope; see D9 and D10) and
GM-59/60 (supports in Bishop's vertical equilibrium, the GLE march with
large supports, and the reverse-curvature option; see D24 and D36).

## 11. The published validation set

These are every benchmark the suite pins, from `slope_stability/VALIDATION.md`
(latest values after GM-131, 134 and 138) and the `validation_examples`
tests. "Printed" is the published answer.

| Ref | Problem (source) | What it checks | Module | Printed |
|---|---|---|---|---|
| B1 | F&K (1977) homogeneous slope, given circle (Slide2 #21), dry | Each method's equations | OMS 1.926, Bishop 2.075, Janbu 2.017, Spencer 2.074, M-P 2.076 | F&K 1.928, 2.080, 2.041, 2.073, 2.076 |
| B1 | Same, ru = 0.25 | ru | OMS 1.606, Bishop 1.758, Spencer 1.763, M-P 1.764 | 1.607, 1.766, 1.761, 1.764 (Slide2 OMS 1.687: formulation) |
| B2 | F&K composite surface on a weak layer (Slide2 #22) | Polyline surface, GLE | OMS 1.313, Spencer 1.386, M-P 1.381; ru: 1.138, 1.132 | F&K 1.288, 1.373, 1.370; 1.118, 1.118 (3 % gate) |
| B3 | ACADS 1(a) (Slide2 #1), search | Circle search | Bishop 0.986, Spencer/GLE 0.985, Janbu 0.989 | 1.00 (Slide2 0.987, 0.986, 0.990) |
| B4 | Duncan (2000) reliability | Lognormal β, pf | β 2.318, pf 1.02 % | 2.32, about 1 % |
| B5 | Duncan, Wright & Brandon textbook suite (19 tests) | Mixed cases | all pass; Ex. 6 assertion inverted (pond buttress); Ex. 5 skipped (curved envelope) | book not in hand |
| B6 | ACADS 1(a), LE against fem2d SRM | Cross-check | LE 0.985 to 0.989; SRM 1.053 | 1.00 |
| B7 | #95, EM 1110-2-1902 dam, Corps 2-stage (V-037) | Drawdown, given surface | 1.34 (seepage stage 1) | 1.347 |
| B7 | #96, same dam, DWW 3-stage (V-038) | Drawdown stage 3 | 1.273 default; 1.370 "gle" | 1.443 |
| B7 | #98, search (V-041), recovered section | Drawdown search | Corps 0.837, DWW 0.938 | 0.931, 1.039 |
| B8 | Pore-pressure grid (V-029) | Interpolation | identical to the line; 1.25 on the reconstruction | not pinned to print |
| B9/B16 | ACADS 1(b) (Slide2 #2), crack (V-026) | Crack models | strength 1.485/1.486/1.482; truncation 1.603/1.599/1.600 | Slide2 1.596/1.592/1.592 |
| B10 | #54 micro-pile row (V-040) | Pile convention | active 1.223; passive 1.203 | 1.193 |
| B11 | ACADS 4 (Slide2 #9), seam + 2 loads (V-028, V-028 E8) | Weak seam, loads | Spencer 0.808, GLE 0.802 (0.806, 0.799 with loads) | referee 0.78; Slide2 Spencer 0.760, GLE 0.720. **Invalid while D1 stands: 1.038 with each point weighed once** |
| B12 | V-054 steep φ = 0 slope, toe circle | Toe family | 2.346 (2.222 with level ground past the toe) | 2.16 (FLAC 2.18) |
| B12 | #86 without supports, c′ = 0 | Planar family | 0.942 | tan 37° / tan 38.66° = 0.942 |
| B12 | Planar family against closed forms (#86, D&W Ex. 3 and 4) | Planar family | +0.11 to +0.15 % | closed form |
| B12 | #86, five tiebacks | Supports in search | Bishop 1.612, Spencer 1.606, GLE 1.606 | 1.629, 1.620, 1.622 (D&W 1.61) |
| B12/B16 | #85, one support, active | Supports, reverse curvature | GLE 1.573, Spencer 1.551, Bishop 1.372 (discard 1.508) | GLE 1.575, Spencer 1.884, Bishop 1.531 (D&W 1.51) |
| B12/B16 | #85, passive | Method B | Bishop 1.254 (discard 1.316), GLE 1.379 | 1.324 (D&W 1.32), 1.378 |
| B12 | V-032 (Slide2 #61) | Entry/exit search | 1.315 | 1.366 (Baker 1.35) |
| B13 | Duncan Ex. 1, D = 2 and 3 | Grid refinement | 0.703, 0.676 | Taylor 0.706, 0.679 |
| B13 | Pilarcitos section, Corps (drawdown) | Drawdown search | 0.7236 | 0.823 |
| B14 | Weak layer over rock, 10 m cut | Default ranges | 4.82 default; about 1.05 directed (since GM-134) | no printed answer (EV-25) |
| B15 | V-028 seam surface, slicing | Corner slicing | 0.808 on 30 to 1,600 slices | 0.78 (see B11, D1) |
| B15 | V-030 Duncan (2000) surface | Polyline slicing | Spencer 1.218, GLE 1.220, Janbu 1.207 | Slide2 1.157, 1.160, 1.168; Duncan 1.17 (GM-136) |
| B15 | V-043 sand fill, DE | Noncircular search | 1.234 (known gap, strict xfail) | Slide2 1.188, Tandjiria 1.192 |
| B15 | V-041 #95/#96, Corps search | Drawdown planar | 0.783 | infinite slope 0.772 |
| B16 | V-043 circle, Spencer | Circle slicing | 1.210 | Slide2 1.209, Tandjiria 1.219 |
| B16 | V-031 (Slide2 #57), Bishop/Spencer/GLE | Layers + water | 1.421/1.425/1.409 | 1.417/1.422 |
| B16 | Vertical cut, Bishop search | φ = 0 | 0.842 | Taylor 0.851 |
| B17 | V-043 Spencer, refined | Refinement walk | 1.1859 | Slide2 1.209 (circle), 1.188 (noncircular) |
| B17 | V-042 (Hassan & Wolff) | Entry/exit | Bishop 1.3334 | 1.334 (Slide2 1.340) |
| B17 | Duncan Ex. 1, 4 H each side | Search | Bishop 0.7049 | Taylor 0.706 |
| B18 | Surface touching the base | Base tolerance | 0.7045 to 0.7063 on 25 to 100 slices | (consistency check) |
| V-035/036 | Slide2 #79 and #81 (Duncan & Wright Figs. 14.4, 14.7), cohesionless embankments | Infinite-slope mechanism | see the test file | referee 1.44, 1.15 |
| V-039 | Slide2 #104 | Newmark displacement | integrator and Jibson pass; #104 limited by the record | FS 1.359, 0.978; Ky 0.139; 5.042 cm |
| V-044 | Slide2 #70 (Duncan & Wright Fig. 6.27), submerged slope | Ponded water | see the test file | Bishop 1.603, Spencer 1.599, GLE 1.599 |
| V-045 | Slide2 #34, correlated strengths | FOSM | 3.482 (capability check) | none pinned |
| V-048, V-052 | Lowe-Karafiath method; Pilarcitos Dam (Slide2 #97) | Drawdown method ordering | see the test files | Corps 0.823, LK 1.047, DWW 1.043 |

Reading the set as a whole:

* The equations are well validated on simple cases (B1, B3, B16, B17).
* The searches agree where the critical surface is a circle on simple
  ground.
* The cases that do not agree are the ones this account names: cracks
  (D39), reverse curvature (D24), drawdown stage 3 (D38), weak seams on
  sloping boundaries (D1), noncircular searches (D12), and V-030
  (probably input).
* Only one benchmark, B11/V-028, uses a sloping layer boundary, and it is
  the one D1 invalidates. The suite has no valid check of sloping
  boundaries.

## 12. Proposed plan, in order

Each step is integral: it changes one rule for every method, family and
caller, then re-runs the whole validation set (§11) and re-pins against the
printed answers, not against the module's previous numbers. No step is
started until the owner agrees the plan. Owner decisions are marked
[owner].

1. **One material per point.** Weight, strength and pore pressure all come
   from the same layer at a point, with sloping tops as well as bottoms, or
   layers as closed regions. Fix the drawing importer and the project
   builder conventions to match. Re-pin V-028 and B11, and add a printed
   sloping-boundary benchmark (D1).
2. **A method's own solution, or no number.** Remove the old approximate
   solvers from searches, single surfaces, Newmark and the probabilistic
   analyses. Extend the robust λ search to unsupported runs. Put every
   non-convergence, Bishop's included, in the result (D23, D27, D30, D49;
   GM-81, GM-139).
3. **The printed acceptability checks.** Compression on bases and sides,
   the unclipped line of thrust inside the mass, Whitman & Bailey's
   m_alpha warning, and Bishop against the OMS on every circle. Steep exits
   handled by EM C-10b's remedies rather than silent refusal. [owner]:
   whether searches still refuse m_alpha < 0.2 and keep the φ′ = 0
   exemption (D20, D21, D22, D50).
4. **Tension cracks and reverse curvature as printed.** Truncation as the
   default; the crack side taken from the crest; depth by Rankine with
   developed strengths or by a minimum-F scan. [owner]: whether
   reverse-curvature circles are discarded or closed by a crack sized by
   the tension test (D24, D39).
5. **Water and loads as printed.** The cos² correction (default [owner]),
   ru on overburden stress, the pond edge as a slice boundary, u = 0 on
   total-stress layers. Line, inclined and varying loads. Seismic force on
   the total slice weight with an optional vertical coefficient, and a
   warning for Bishop with kh (D2 to D7, D16).
6. **One method across the search.** Every family is scored, or at least
   re-scored for the final comparison, by the method named, with the named
   interslice function. Report a family that only another method can solve
   as that (D13, D42).
7. **Search coverage as printed.**
   * Face exits at all heights in the main search (D9).
   * Noncircular surfaces generated from the critical circle and along weak
     layers, with concave-up shapes and final shifts of 10 to 25 % of the
     thinnest stratum (D12; GM-117).
   * Impenetrable layers and composite surfaces (D14).
   * Several independent starts (D43).
   * The module stays general; the agent directs it (EV-25).
8. **Supports as printed.** Geosynthetic pullout and minimum embedment,
   the nail's front term, both readings of a pull that assists sliding
   with slack the default (OWN-59), a legacy force without an elevation
   refused, and the GM-82 directions (D34 to D37).
9. **Rapid drawdown as EM App. G prints it.** Seepage stage 1 as the
   default when a seepage line is given or can be derived; stage 2 solved;
   the drained check from stage 2's N (Eq. G-13); stage 3 solved only if
   needed. This settles OWN-17. Re-measure #95, #96 and #98, and GM-15
   (D38).
10. **Reporting as printed.** A contour plot of minimum F over the centres,
    an ordered list of the lowest surfaces, the method actually used per
    family, and the slice table first in the report. Remove or wire the
    calculation-package inputs that do nothing (D45 to D47, D49; GM-132).

Methods not built (D32, GM-19) are not in the plan. The Modified Swedish /
force-equilibrium check (EM C-4, C-5d) is the one to consider first if the
owner wants a method added, because the EM uses it to verify Spencer.

## Appendix A: code locations

Paths are relative to the repository root, at commit 60a2975.

| D | Where |
|---|---|
| D1 | `slope_stability/slices.py:781` (weight sums every overlapping layer); `slope_stability/geometry.py:471` (strength: first layer containing the point); `geometry.py:289` (flat top); `dxf_import/converter.py:95-190` (next layer's top at the boundary's highest point); `geo_project/schema.py:519` (layer top = bottom of the layer above); `validation_examples/test_published_v028_slope.py:42-48` (all three tops at el. 40) |
| D2 | `slope_stability/slices.py:821` (u = γ_w × vertical depth; note SS-3) |
| D3 | `slope_stability/slices.py:681-684` |
| D5 | `slope_stability/geometry.py` (`surcharges` zones); `VALIDATION.md` B11 (ramp as mean) |
| D6 | `slope_stability/slices.py:726-728` (kh on soil weight only) |
| D7 | `slope_stability/analysis.py:587` (search default Bishop); `slices.py:726` (kh applied whatever the method) |
| D9, D10 | `slope_stability/search_ranges.py:1-60, 76, 202, 222`; `search_supplements.py:79` |
| D11 | `slope_stability/search.py:1019` (20 radii + golden section) |
| D12 | `slope_stability/search.py:1575, 1665` (independent vertex depths); `search.py:1732, 1967, 2426` |
| D13 | `slope_stability/analysis.py:734`; `search.py:902` (half-sine fixed); `analysis.py:581` (no interslice argument) |
| D16 | `slope_stability/slices.py:301` (break list); `rapid_drawdown.py:567` (nearest mid-point) |
| D18 | `slope_stability/slip_surface.py:233` (200 samples) |
| D19 | `slope_stability/slices.py:442`; `search.py:1226, 1229`; `search_supplements.py:358` (toe-family below-base test) |
| D20 | `slope_stability/methods.py:377, 387`; `search.py:160`; `search.py` refusal counters |
| D21 | `slope_stability/methods.py:352` |
| D22 | `slope_stability/search.py:555, 597, 742, 771` |
| D23 | `slope_stability/search.py:962, 968` (fallback-capable solvers in circular scoring) against `search.py:694` (noncircular requires convergence); `methods.py:877, 922` |
| D24 | `slope_stability/geometry.py:371`; `slip_surface.py:296`; `search.py` (discard branch in the trial scorer) |
| D26 | `slope_stability/methods.py:68` (OMS); `methods.py:136-145` (force sum on polylines) |
| D27 | `slope_stability/methods.py:360-364` |
| D28 | `slope_stability/gle.py:985, 1000` |
| D29 | `slope_stability/gle.py:212, 439, 717, 908, 460` |
| D30 | `slope_stability/methods.py:445, 650` (old solvers); `newmark.py:167` |
| D34 | `slope_stability/reinforcement.py:92` |
| D35 | `slope_stability/nails.py:136, 273` (min of pullout and tensile) |
| D36 | `slope_stability/reinforcement.py` (support directions); `search_supplements.py:851` |
| D37 | `slope_stability/search.py:651-660` (a legacy force counts only with an elevation) |
| D38 | `slope_stability/rapid_drawdown.py:374, 375, 282, 567` |
| D39 | `slope_stability/geometry.py:354-355`; `slices.py:602, 765` |
| D40 | `slope_stability/slices.py` (Hoek-Brown normal and SHANSEP stress); `geometry.py:96` |
| D42 | `slope_stability/search_supplements.py:103, 125, 1224` |
| D43 | `slope_stability/search.py:1162-1165, 1314` |
| D44 | `slope_stability/search.py:212` |
| D46, D47 | `slope_stability/plotting.py:335` (trial map, no contours); `results.py` (`grid_fos`, `candidates`) |
| D49 | `slope_stability/analysis.py:193` (unknown method runs Bishop); `probabilistic.py:493` |
| D50 | `slope_stability/analysis.py:279-290` (thrust line clipped for display) |

## Appendix B: backlog items absorbed

GM-12, GM-15, GM-18, GM-19, GM-56, GM-81, GM-82, GM-89, GM-117, GM-126,
GM-130, GM-132, GM-135, GM-136, GM-137, GM-139, OWN-13 / SPL-3, OWN-17,
OWN-59 and the GM-140 extras, each under the step named above. EV-25 (does
the agent direct the search) stays separate; it measures the agent, not the
module.
