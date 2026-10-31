# downdrag's toe and sand beta against UFC 3-220-20 and GEC-12, as printed

BACKLOG GM-283, 2026-10-31. Read from the PDFs in `geotech-references/docs/`:
UFC 3-220-20 (16 January 2025) chapter 6 and Appendix B-5.4; GEC-12
(FHWA-NHI-16-009, 2016) Vol 1 chapter 7. Page numbers are the printed ones
(UFC: pdf index = page + 37; GEC-12 Vol 1: page + 45).

The module's drag procedure is UFC 3-220-20 §6-7.4 (pp. 476-478) with
GEC-12 §7.3.6.1 (pp. 342-348). Both leave the static method to the
engineer: "Characterize the nominal side and base resistances using an
appropriate SCA method" (UFC step 2, p. 476); "Using an appropriate static
analysis method for the pile type and subsurface conditions" (GEC-12 Step
2, p. 342). So the toe and the sand's beta take the routes the two manuals
print for driven piles, with the other printed route offered.

## What the print gives

- **UFC Eq 6-20 and Table 6-17 (p. 421).** qb = Nq sigma'zD for drained
  conditions, Z/b of 5 or more. Table 6-17 prints two routes:
  - the Fellenius (2021) column: clay 25-30 deg, Nq 3-30; silt 28-34 deg,
    20-40; sand 32-40 deg, 30-150; gravel 35-45 deg, 60-300. These are
    GEC-12 Table 7-9's Nt (p. 256) to the digit;
  - the columns "based on Cheng (2004)" at Z/b = 10, 20 and 40, and the
    equation under the table: Nq ~= 1.2 (Z/b)^-0.437 exp[6.34 (Z/b)^0.0486
    tan phi'], at most 200 "unless supported by project-specific
    information". The equation reproduces all 18 printed Cheng cells to
    the whole number (`downdrag/tests/test_toe_and_beta.py`); e.g. Z/b 10:
    37 at 32 deg and 168 at 40 deg.
- **UFC Table 6-15 (p. 414), beta for driven piles,** printed as pairs:
  clay 0.15 at 25 deg / 0.35 at 30 deg; silt 0.25 / 28, 0.50 / 34; sand
  0.30 / 32, 0.90 / 40; gravel 0.35 / 35, 0.80 / 45. The same as GEC-12
  Table 7-9's beta. GEC-12 §7.2.1.3.4 Step 2b (p. 257): "use Table 7-9 to
  estimate beta coefficient from phi' angle"; Step 5b (p. 258) the same for
  Nt.
- **UFC Eq 6-6 (pp. 411-412):** beta = K tan delta, K/K0 off Fig 6-13
  (Salgado) or Table 6-13, K0 = 1 - sin phi', delta/phi' off Table 6-14.
  The module's old default, (1 - sin phi') tan phi', is this with K/K0 = 1
  and delta = phi', which UFC draws only for non-displacement columns at low
  relative density (Fig 6-13, p. 413: the dashed Dr = 20 % line about
  0.85-1.0; Table 6-14: delta/phi' = 1 for formed-in-place columns). For
  displacement piles Fig 6-13's solid lines give K/K0 from about 1.1 (Dr
  20 %, sigma'z/Pa = 3) to above 6 (Dr 80 % near the surface).
- **UFC p. 424:** N*c = 9 for Z/b of at least 2.5 (Brinch Hansen 1957), for
  Eq 6-27, qb = N*c su (p. 423). UFC's drag example takes Rb = 9 su B^2
  (App. B-5.4, p. 637); it has no sand toe.

## What changed

| Value | Before | Now | Source |
|---|---|---|---|
| beta, cohesionless layer, not given | (1 - sin phi') tan phi' (cited "Fellenius 1991"), 0.28-0.30 at any phi' | GEC-12 Table 7-9 by `soil_class` (sand unless given), linear in phi', held at the range's ends with a warning | GEC-12 Table 7-9, Step 2b; UFC Table 6-15 |
| Nt, cohesionless toe, not given | a piecewise curve in phi' printed nowhere (`_Nt_from_phi`, the curve GM-282 removed from axial_pile) | `Nt_method="fellenius"` (default): Table 6-17's Fellenius column = Table 7-9, by `soil_class`; `"cheng"`: the equation, at most 200 | UFC Table 6-17, Eq 6-20; GEC-12 Eq. 7-20, Step 5b |
| Nt, clay toe | 9 | 9, now cited | UFC p. 424; App. B-5.4 |
| beta = 0.3 fallbacks (GM-04's "uncited 0.3") | in the shaft loop and the soft-clay check | removed: dead code (a sand's beta is always set, a beta clay must give beta) | |

The result's `choices` now carry the toe's Nt (given, clay, or the route
with the other route's value) and the cohesionless layers' beta (given or
from the table). The adapter takes `Nt_method` and a layer's `soil_class`;
the calc package's toe step names the Nt and its source.

### beta and Nt by phi', sand (old -> new), hand-checked

| phi' (deg) | beta old | beta new | Nt old | Nt new (Fellenius) | Nt Cheng, Z/b 40 |
|---|---|---|---|---|---|
| 28 | 0.282 | 0.300 (held at 32) | 30 | 30 (held) | 13.5 |
| 30 | 0.289 | 0.300 (held) | 46 | 30 (held) | 19.1 |
| 32 | 0.294 | 0.300 | 62 | 30 | 27.4 |
| 34 | 0.297 | 0.450 | 86 | 60 | 39.9 |
| 35 | 0.299 | 0.525 | 102 | 75 | 48.5 |
| 36 | 0.299 | 0.600 | 118 | 90 | 59.2 |
| 38 | 0.300 | 0.750 | 150 | 120 | 89.7 |
| 40 | 0.300 | 0.900 | 190 | 150 | 139.0 |
| 42 | 0.298 | 0.900 (held at 40) | 230 | 150 (held) | 221.4 (200 used) |

By hand at 36 deg: old beta = (1 - 0.5878) x 0.7265 = 0.2995; new beta =
0.30 + (36 - 32)/(40 - 32) x 0.60 = 0.60, Nt = 30 + 0.5 x 120 = 90. Cheng:
40^-0.437 = 0.1995, 40^0.0486 = 1.1964, 6.34 x 1.1964 x tan 36 deg = 5.511,
Nq = 1.2 x 0.1995 x e^5.511 = 59.2.

The new beta is up to three times the old in dense sand: more drag where
dense sand lies above the neutral plane, more resistance below it. The new
Nt is 20-35 % lower over 32-40 deg.

### Results that move (old at c513235 -> now), each checked by hand

**The manual's worked example** (`docs/user_manual/gen_worked_examples.py`:
L 18 m, d 0.4 m, clay 0-6 m cu 30, sand phi' 34 deg below, 2 m fill at 19
kN/m3, water at 3 m, Q_dead 400 kN; beta and Nt left to the defaults):

| | Before | Now |
|---|---|---|
| toe resistance | 2,386.7 kN (Nt 86) | 1,665.2 kN (Nt 60) |
| sand beta | 0.297 | 0.450 |
| curves at 100 % toe | do not intersect | intersect |
| headline (auto, UFC §6-7.4) | settlement compatibility: NP 5.99 m, drag 226 kN | force equilibrium at 100 % toe: NP 17.66 m, drag 1,308 kN |

By hand (final stress, P = 1.2566 m, A = 0.12566 m2): sigma'v at the toe
= 38 + 17 x 3 + 7.19 x 3 + 9.19 x 12 = 220.85 kPa; toe = 60 x 220.85 x
0.12566 = 1,665.2 kN. Drag to 17.655 m: clay 1.0 x 30 x 1.2566 x 6 =
226.2, sand 0.45 x 1.2566 x 11.655 x (110.57 + 9.19 x 11.655/2) = 1,081.7,
total 1,307.9 kN. Resistance below: 1,665.2 + 0.45 x 1.2566 x 0.345 x
219.27 = 1,708.0 = 400 + 1,307.9. The module adds a warning that the
intersection lies below 5.99 m, where the soil settles no more than the
pile. The swing is the §6-7.4 procedure meeting the printed beta and Nt:
with the full toe the curves now meet, and the conservative full-mobilization
drag of step 3 is the headline; settlement compatibility and the 0 % and
50 % toe cases stay in `neutral_plane_comparison`.

**The class docstring's example** (fill phi' 30, soft clay cu 30, dense
sand phi' 35, L 18 m, d 0.3 m, Q_dead 500 kN, 3 m fill): toe 1,637.0 ->
1,203.6 kN (Nt 102 -> 75); betas 0.289 / 0.299 -> 0.300 (held) / 0.525;
headline NP 12.99 m (settlement compatibility) -> 17.33 m (100 % toe);
drag 350.9 -> 778.1 kN; maximum load 850.9 -> 1,278.1 kN. By hand: toe =
75 x 227.04 x 0.070686 = 1,203.6; drag = fill 0.30 x 0.9425 x 251.6 =
71.1, clay 282.7, sand 0.525 x 0.9425 x 857.3 = 424.2, total 778.1;
resistance below = 1,203.6 + 74.5 = 1,278.1 = 500 + 778.1.

**F67** (the live-test profile, `funhouse_agent/tests/test_downdrag_beta_in_clay.py`):
drag 543.7 -> 545.1 kN (top sand phi' 32: beta 0.2938 -> 0.300; by hand
+0.0062 x 1.6 x 136 kPa.m = +1.35 kN); with the clay by beta 0.25, 654.6
-> 656.0; beta 0.30, 772.6 -> 774.2. NP 13.994 -> 13.996 m (settlement
compatibility both). Toe 5,803.2 -> 4,642.6 kN (phi' 38: Nt 150 -> 120,
ratio 0.8). The live-smoke key for F67 (`module_work/live_smoke/flows.json`,
"13.99 m", range 6.5-22.5 m) still holds.

**Module tests re-pinned** (`downdrag/tests/test_downdrag.py`):
- standard case (fill phi' 28, clay, sand phi' 36, HP 0.01006 m2): toe
  257.9 -> 196.7 kN (Nt 118 -> 90; 257.9 x 90/118 = 196.7); the whole
  shaft plus toe 644 -> 1,055.5 kN, so the "overloaded pile" test's Q_dead
  goes 1,000 -> 1,200 kN to stay overloaded;
- deep clay case: NP 3.30 -> 3.2638 m, by hand 300 + D = 28.27 + 345.58 +
  (131.14 - D) with beta 0.30, D the drag above the plane.

V-004 (`validation_examples/`) gives beta and Nt, so it does not move.

### The compact post_liquefaction_downdrag result

The two new choices would have taken the size-tested results over their
budget (GM-207/240). The compact result keeps each as what, chosen and how;
basis and alternatives are in `detail='full'`, and so is the whole entry
where the call gave Nt or beta. With beta and Nt given (the size tests):
one zone 7,589 -> 7,634, the noisy CPT up to 10,428 -> 10,469 (budget
10,500). With both from the defaults: one zone 7,681 -> 7,856, the noisy
CPT 10,239-10,745 -> 10,593-11,096 (90 runs: seeds 1-30, E_s none, 10 and
30 MPa); 60 of the 90 were over the budget before, all 90 now.

## The other defaults on this path

- **Neutral plane:** `neutral_plane_method="auto"` is UFC §6-7.4 steps 4-5
  (p. 477): the intersection of the curves, and where they do not meet,
  settlement compatibility (UFC §6-5.8.4.3, p. 449: "an iterative process
  ... so that the mobilized toe resistance used to locate the neutral plane
  is compatible with the pile toe penetration"). Already cited; unchanged.
- **Toe mobilization:** 1.0 by default, "a conservative approach ... fully
  mobilize side friction and base resistance" (UFC §6-7.4 step 3, p. 477);
  0 %, 50 % and 100 % are always reported (UFC §6-5.8.4.2, p. 447, after
  Siegel et al. 2013; GEC-12 Steps 4-5, pp. 343-346). Already cited;
  unchanged.
- **beta in clay:** given, never derived (GM-249); unchanged.
- **GM-04, "beta = 0.3 default uncited":** the 0.3 fallbacks were dead
  code; removed.

## Noted, not fixed

- Table 6-15's beta rows are for driven piles. A drilled shaft's beta (UFC
  Table 6-15's FHWA rows by depth; GEC-10) is given on the layer. The
  module does not know the pile type.
- Table 6-17 prints no Cheng values for gravel; the equation runs on phi'
  alone, so `Nt_method="cheng"` takes a gravel toe like any other.
- Z/b of 5 or more (Table 6-17) and 2.5 or more (N*c = 9) are not checked.
- Vesic's N*q (UFC Eq 6-23, needs the rigidity index) and the SPT and CPT
  base methods (UFC Tables 6-20, 6-23) are not offered; give `Nt`.
- UFC Eq 6-6 with Fig 6-13's printed fit (K/K0 ~= A (sigma'z/Pa)^b by Dr,
  displacement or not) and Table 6-14's delta/phi' is a third printed route
  to beta. It needs Dr and the pile's material, so it is not offered; a beta
  worked out that way is given on the layer.
- The drained toe in a beta clay (GEC-12 Eq. 7-20 with Table 7-9's clay
  Nt 3-30) stays the owner's call (GM-249); a clay toe is 9 cu.
- The default-strength noisy-CPT post-liquefaction result is over the
  10,500-character budget (it was before GM-283 too); no number changes.
