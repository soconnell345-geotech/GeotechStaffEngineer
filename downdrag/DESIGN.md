# Downdrag Module — Design Notes

## Theory

Pile downdrag (negative skin friction) occurs when soil around a pile settles
more than the pile itself. The settling soil drags the pile downward, inducing
additional compressive load. Common triggers:

- **Fill placement**: New embankment causes consolidation of underlying soft soils.
- **Groundwater drawdown**: Lowering the water table increases effective stress,
  causing consolidation.

## The neutral plane (Fellenius 2004; UFC 3-220-20 §6-7.4)

The analysis finds the **neutral plane** — the depth where pile settlement equals
soil settlement. Above the NP, soil settles more than the pile, inducing downward
friction (negative skin friction; its sum is the drag load). Below the NP, the
pile settles more than the soil, and friction acts as resistance (positive skin
friction).

### Load and resistance curves

Two force curves are constructed along the pile (UFC 3-220-20 §6-7.4 step 4,
Fig 6-26; GEC-12 §7.3.6.1 step 4, Figs 7-57 to 7-59):

1. **Load curve** (UFC Curve B): `Q_top(z) = Q_dead + ∫₀ᶻ fs·P·dz`
   (dead load + negative skin friction from the head). The pile's own weight
   is left out, as UFC's and GEC-12's curves leave it out (next section);
   `include_pile_weight=True` adds `W_pile(0→z)`.

2. **Resistance curve** (UFC Curve A): `Q_bot(z) = Q_toe + ∫_z^L fs·P·dz`
   (mobilized toe resistance + positive skin friction from the toe upward)

The neutral plane is where `Q_top(z_np) = Q_bot(z_np)`. The maximum axial
load in the pile, Q_np, occurs at this depth; the drag force is
`DF = Q_np - Q_dead` (UFC Eq 6-80, GEC-12 Eq 7-70).

### The pile's own weight: left out by default (owner, 2026-10-15)

The owner's decision (OWN-58): "Leave it out by default, as the references
do." UFC 3-220-20 §6-7.4 step 4 (checked against the text): "The load curve
(Curve B in Figure 6-26) is created by plotting the unfactored permanent load
on the element at the top of the element and adding the cumulative negative
skin resistance (drag load) as a function of depth". GEC-12 §7.3.6.1 step 4
(p. 344, checked against the PDF): "Add the unfactored permanent load to the
accumulated shaft resistance versus depth" (Fig 7-58, "unfactored permanent
load plus cumulative shaft resistance vs depth"). Neither adds the pile's
weight, nor do the worked examples (UFC App. B-5.5, GEC-12 Vol III Block 17)
or CGPR #56 §3.2.3 (Hannigan et al. 1997). Before 2026-10-15 the module put
the pile's weight in the load curve.

- `include_pile_weight=False` (default): the load curve, Q_np and the drag
  force carry no pile weight; `pile_weight_to_np` is 0.
- `include_pile_weight=True`: the weight of the pile above each depth,
  `pile_unit_weight x pile_area x z`, joins the load curve, so it moves the
  neutral plane up a little and adds to Q_np, the drag force and the UFC Eq
  6-80 demand.

Every result says which: a sentence at the end of `neutral_plane_basis`
("The pile's own weight is left out of the load curve, as UFC 3-220-20 Fig
6-26 and FHWA GEC-12 §7.3.6.1 do (include_pile_weight=False)."),
`include_pile_weight` and `pile_weight_basis` in `to_dict()`, a "Pile's own
weight" line at the head of `summary()`, and in the calc package an input
row, the force-components table and the Q_np equation.

Numbers that move (the weight left out vs in, at the old 24 kN/m³):

| Case | Weight left out (default) | Weight in (before) |
|---|---|---|
| DD-1 headline (plane at 8.00 m) | Q_np = DF = 150.8 kN | 174.9 kN (+24.1) |
| DD-1, 0 % toe mobilization row | 10.43 m, Q_np 259.5 kN | 10.12 m, 274.7 kN |
| DD-1, plane at the pile toe | Q_np 518.9 kN | 564.2 kN |
| V-004 (published 54 ft / 486 / 285 kips) | 54.00 ft, 486.6, 285.6 kips | 53.61 ft, 490.1, 289.1 kips (20 kN/m³) |

The UFC App. B-5.5 test enters no pile weight (`pile_unit_weight = 0`) and
does not move.

**Unit weight default (GM-50).** The module has no pile-material input. Its
stiffness default `pile_E = 200e6` kPa is steel, the calc package defaulted
the unit weight to steel (78.5 kN/m³) and the tool to concrete (24). The
default is now steel everywhere (`DowndragAnalysis`, the
`downdrag_analysis` tool, `downdrag_package`), the same material as the
stiffness; the parameter descriptions say so and give concrete's ~24. It is
used only when the weight is included. It multiplies `pile_area`, which
defaults to the solid circle, so for an H or pipe pile give `pile_area` as
the steel area (the calc package takes no `pile_area`: give an equivalent
unit weight there).

## Neutral-plane methods

The owner (2026-10-09): "Downdrag has many methods and it should be open to
multiple assumptions. See DM 7.2 or the FHWA driven pile manual." (2026-10-14):
"I think presenting all of them is fine." Every method below is evaluated on
the same inputs in every run and reported side by side (next section). Each is
named as the reference words it; page numbers are the printed ones, checked
against the source PDFs in `geotech-references/docs/`. UFC 3-220-20 = 16 Jan
2025 (DM 7.2); GEC-12 = FHWA-NHI-16-009 Vol I (Vol III = its design example);
GEC-10 = FHWA-NHI-18-024; CGPR #56 = Greenfield & Filz (2009), cited by section
only (not public-domain, not in the repo).

| Method (row name; `neutral_plane_method`) | Source | Assumptions | When to use |
|---|---|---|---|
| **Neutral plane at 100% toe mobilization** (`force_equilibrium`, `toe_mobilization=1`; what `auto` uses when the curves intersect) | UFC §6-7.4 steps 3-5, Fig 6-26 (pp. 476-478); GEC-12 §7.3.5.7 (pp. 334-336, Fig 7-49) and §7.3.6.1 step 5, Fig 7-60 point A (p. 346); GEC-12 Vol III Fig D-33 (the name); GEC-10 §10.6.2, Fig 10-20 (pp. 10-47/48, defers to GEC-12 §7.3.6.1); CGPR #56 §3.2.3 | Intersection of the load and resistance curves; side and base fully mobilized; unfactored permanent load only, transient load excluded (UFC step 4) | Drag force for the STRUCTURAL strength limit state: UFC step 3 calls it "a conservative approach for evaluating the drag force"; GEC-12 Vol III Block 17: "the toe mobilization curve that should be used to check the pile section's structural strength". Not conservative for settlement (UFC §6-5.8.4.2) |
| **Neutral plane at 50% / 0% toe mobilization** (`force_equilibrium` with `toe_mobilization`; all three always reported) | UFC §6-5.8.4.2 "Neutral Plane Method" (p. 447) and App. B-5.5, Table B-28 (p. 640); GEC-12 §7.3.6.1 "Recommended Approach for Downdrag" steps 2, 4, 5, Figs 7-55 to 7-60 (pp. 342-346; Siegel et al. 2013); GEC-12 Vol III Fig D-32 (the name) | Side fully mobilized (~0.1 in of movement), toe at a fraction of its nominal resistance (full toe needs 4-10 % of the width). Less toe -> shallower plane, less drag, more settlement | SERVICE limit (settlement): analyse 0 / 50 / 100 %; if the conclusion does not change, stop; if it does, refine with t-z / q-z curves (UFC §6-5.8.4.2) |
| **Mobilized toe resistance compatible with the toe penetration** (`settlement_compatibility`; what `auto` uses when the curves do not intersect) | UFC §6-5.8.4.3, Fig 6-19 (pp. 449-450): pile and soil settle equally at the neutral plane; "the difference between the soil and pile settlement at the toe elevation is the pile penetration"; with a q-z curve, "an iterative process can be applied, as described by Fellenius (2021), so that the mobilized toe resistance used to locate the neutral plane is compatible with the pile toe penetration". CGPR #56 §3.2.4 (PILENEG, elastic toe) | As implemented: elastic-perfectly-plastic toe, the elastic penetration from the equivalent-footing compression of the bearing stratum (UFC Eqs 6-49 to 6-54) standing in for a q-z curve; side fully mobilized (partly when the load at the plane is less than the friction below). Never below the full-mobilization intersection; if the soil still out-settles the elastic pile there, the toe is pushed to its full resistance and the plane IS that intersection | When the ultimate toe makes an intersection impossible (end-bearing piles); whenever the toe force should follow the toe movement |
| **Neutral plane at the bearing layer (end-bearing pile)** (`bearing_layer_top`) | UFC §6-7.4 step 5 (p. 477): "the neutral plane will be located near the interface of the column and the bearing layer for an end-bearing column bearing in a stratum that is much stiffer than the compressible soil"; §6-5.7 (p. 438, Fellenius 2021: end-bearing -> large drag force, small downdrag). CGPR #56 §3.2.2 (Poulos, Eq 3.7) caps the plane at the bearing-layer top | End-bearing pile; the toe below all the settling soil, in a much stiffer stratum. Set by that statement, not computed (checked: the toe must be able to carry Q_np minus the friction below) | Estimate / check for end-bearing piles. Not for a floating pile (toe inside the settling soil) |
| **Neutral plane at the pile toe** (`pile_toe`) | GEC-12 §7.3.5.7 (p. 336): Goudreault & Fellenius (1994) recommend "locating the neutral plane at the pile toe" for "most cases" of GROUP settlement (piles below the plane reinforce the soil); CGPR #56 Table 3.1 "end bearing" = 1.0 L | The whole shaft in negative skin friction | Group-settlement simplification; for a single pile's drag it is the UPPER BOUND, never a silent default (G8) |
| **Endo et al. (1969)** (`endo` + `endo_bearing_condition`) | CGPR #56 §3.2.1, Table 3.1 (Endo et al. 1969; Little 1994) | Plane at 0.67 / 0.75 / 1.0 of the embedment for floating / stiff-flexible bearing / end bearing piles through consolidating clay; drag load = full friction above it; the method itself gives no settlement (the module computes one for the plane, as for every row) | Estimate / check level only |

Not found in the references, so not built:

- **"A fraction of the compressible depth"**: no reference in the repo states
  the neutral plane as a fraction of the compressible-layer thickness. The
  fraction rules found are Endo (a fraction of the EMBEDMENT, above) and the
  equivalent-footing depths (UFC Table 6-32, p. 447: 2/3 of the embedment in
  the bearing layer, etc.; GEC-12 §7.3.5.3, p. 320: an equivalent footing
  1/3 D above the toe, Terzaghi & Peck 1967), which locate a SETTLEMENT
  footing, not a neutral plane, and give no drag.
- Poulos and PILENEG (CGPR #56 §3.2.2, §3.2.4) stay as the profile-input
  functions in `cgpr56.py` (`downdrag_method_comparison`); they need a
  two-layer idealization or a bearing modulus the soil-parameter analysis
  does not carry.

Drag-load assumptions the references state:

| Assumption | Source | In the module |
|---|---|---|
| Unit negative friction = the nominal side resistance (alpha / beta); higher alpha, beta conservative for drag, lower for resistance | UFC §6-7.4 steps 2-3 (pp. 476-477); GEC-10 §10.6.2 (p. 10-48, unit drag f_DN by the side-resistance methods); GEC-12 §7.3.6.1 step 2 | Yes; the engineer picks alpha / beta per layer |
| Side and base resistance on the effective-stress profile that INCLUDES the settlement-causing change (Siegel et al. 2013) | UFC §6-7.4 step 2 (p. 476); GEC-12 §7.3.6.1 (p. 348) points the same way without a rule (next section) | Default since 2026-10-14: `skin_friction_stress="final"` (beta friction and the toe in sand); `"initial"` on request |
| Transient load excluded from the head load (it reverses the friction while it acts) | UFC §6-7.4 step 4 (p. 477) | Yes (Q_dead only) |
| Drag excluded from the geotechnical strength check, included in structural strength and in settlement | UFC Table 6-30 (p. 437); GEC-12 §7.3.6 (pp. 336-337); GEC-10 §10.6.1 (p. 10-46; notes AASHTO treats DD as a load in geotechnical strength) | Yes (UFC / FHWA) |
| Load factor 1.10 on the drag force (1.25 on Q_d) | UFC Eq 6-80 (p. 478); GEC-12 Eq 7-70 (pp. 346-347: MnDOT 1.1, no AASHTO factor yet) | Yes |
| Post-liquefaction drag = 50 % of the pre-earthquake beta side resistance in liquefied layers | GEC-10 §10.7 (p. 10-52) | Not implemented |

## Effective stress for skin friction: final (owner, 2026-10-14)

The owner: "For stress, see what FHWA GEC 12 says about initial vs final
stress and go with that; if it's silent, use final stress like UFC."

**FHWA GEC-12 (FHWA-NHI-16-009, Vol I).** §7.3.6.1 "Recommended Approach for
Downdrag" never says "initial" or "final" effective stress. Step 1 (p. 342):
"Assume soil consolidation and ground settlement will occur." Step 2: "Using an
appropriate static analysis method for the pile type and subsurface conditions,
determine the nominal shaft, mobilized toe, and total mobilized resistance as a
function of pile penetration depth." The one statement on stress comes after
step 7 (p. 348): "It should be noted that effective stress changes such as
approach fill placement after pile driving will alter the plot of the permanent
load plus cumulative shaft resistance. This will in turn alter the location of
the neutral plane, the maximum compression force in the pile, the magnitude of
the drag force, and calculated settlement due to downdrag." That says the
shaft resistance in the analysis follows the stress change, but it is not a
rule for which stress to use. The Vol III design example draws its neutral
planes (Blocks 15 and 17) on the Table D-6 resistance, a DrivenPiles Nordlund
analysis (§D.10.1), and does not say whether that analysis carries the
embankment's stress. So GEC-12 is silent in the owner's sense.

**UFC 3-220-20 (16 Jan 2025) §6-7.4 "Design for Drag Force", step 2
(p. 476)**, checked against the PDF: "Keep in mind that lower unit side
friction from lower α and β is conservative for resistance, while higher unit
side friction from higher α and β is conservative for negative skin friction.
When side friction is evaluated using an effective stress method, such as the
Beta method, Seigel et al. (2013) demonstrate analytically that unit side
friction increases due to the increases in vertical stress. Nominal side and
base resistances should be evaluated considering the vertical effective stress
profile that includes stress change(s) producing settlement." (UFC spells
Siegel "Seigel" here.)

**Default: `skin_friction_stress="final"`**: beta-method unit skin friction and
the toe resistance in sand use sigma'v0 plus the fill / drawdown change, for
every method in the table. Alpha (cu) friction and the toe in clay do not
depend on it. `"initial"` stays available. Every result states which stress it
used (`neutral_plane_basis`, the summary header, the calc package inputs).
Numbers that move: only cases with beta layers or a sand toe under a stress
change. V-004 (no fill) and UFC App. B-5.5 (all alpha, clay toe) do not move.
DD-1 (the pile weight then in the load curve): the 0 % toe mobilization plane
goes from 9.49 m to 10.12 m, drag load 192 to 244 kN; the pile-toe row's drag
load 413 to 519 kN; the toe resistance 2482 to 3236 kN; the headline plane
(8.00 m, 151 kN, set by the clay) does not move.

## Every method side by side; the headline

`DowndragAnalysis(neutral_plane_method=..., toe_mobilization=...,
endo_bearing_condition=..., skin_friction_stress=...)`. Every method in the
table above is evaluated on the same inputs in every run. The result carries
them as a first-class table:

- `neutral_plane_comparison` (and the summary's "Neutral Plane by Each
  Method" table, and the calc package's table of the same name): one row per
  method with `name` (as the reference words it), `method` (the
  `neutral_plane_method` value), `toe_mobilization`, `source`, `assumptions`,
  `applies`, `used` (the row that gives the headline), `reason` (why it does
  not apply, or a note), and, when it applies, `neutral_plane_depth_m`,
  `max_pile_load_kN` (Q_np), `dragload_kN` (negative skin friction above the
  plane, UFC's "drag load", §6-7.4 step 4), `drag_force_kN` (DF = Q_np -
  Q_dead, UFC Eq 6-80 / GEC-12 Eq 7-70; it includes the pile weight above the
  plane only when `include_pile_weight=True`), `toe_force_kN`,
  `soil_settlement_at_np_mm` and `pile_settlement_at_np_mm` (both from the
  original ground), `elastic_compression_above_np_mm` and
  `pile_settlement_mm` (pile head; next section).
- `methods_range`: one sentence, always, e.g. "The methods give neutral
  planes from 8.00 m to 15.00 m and drag forces from 151 kN to 519 kN (4 of 7
  methods apply)." No threshold decides whether it is shown.
- A method that does not apply says why (curves that do not intersect, a
  floating pile for the bearing-layer statement, a toe that could not carry
  what a set plane asks of it, a missing Endo condition, an overloaded pile).
  Asking for one as the headline raises a ValueError with that reason; never
  a silent fallback.

**Headline: `auto` = the UFC 3-220-20 §6-7.4 procedure** (the existing callers
and the calc package read the headline fields: `neutral_plane_depth_m`,
`dragload_kN`, `max_pile_load_kN`, ...). The neutral plane is the intersection
of the load and resistance curves with the toe at `toe_mobilization` (default
1.0: full mobilization, UFC step 3); when the curves do not intersect, the
plane where the pile and the soil settle equally with the toe force compatible
with the toe penetration (§6-5.8.4.3, Fig 6-19). `neutral_plane_basis` starts
"UFC 3-220-20 §6-7.4 procedure: ..." and `headline_method` names the row. A
caller who names a method gets that row as the headline, and
`neutral_plane_basis` starts with its name and source.

DD-1 (8 m of clay over dense sand, 0.4 m pile, 15 m, 40 kPa fill, Q_dead 0,
final stress, the pile's own weight left out), the summary table:

```
  Method                                                  NP   Q_np   Drag   Drag   Pile
                                                                      load  force  settl.
                                                          (m)  (kN)   (kN)   (kN)   (mm)
  Neutral plane at 100% toe mobilization               not applicable (3236 kN toe > 519 kN deliverable)
  Neutral plane at 50% toe mobilization                not applicable (1618 kN toe > 519 kN deliverable)
  Neutral plane at 0% toe mobilization                 10.43   259    259    259    0.1
* Mobilized toe resistance compatible with the toe
  penetration                                           8.00   151    151    151    0.0
  Neutral plane at the bearing layer (end-bearing pile) 8.00   151    151    151    0.0
  Neutral plane at the pile toe                        15.00   519    519    519    0.1
  Endo et al. (1969)                                   not applicable (needs endo_bearing_condition)
The methods give neutral planes from 8.00 m to 15.00 m and drag forces from 151 kN to 519 kN (4 of 7 methods apply).
```

With Q_dead = 0 and the weight left out, Q_np = drag force = drag load in
every row. With `include_pile_weight=True` (24 kN/m³) the rows were 10.12 m /
275 / 244 / 275, 8.00 m / 175 / 151 / 175 and 15.00 m / 564 / 519 / 564.

(The settlements are tiny because DD-1's sand has no modulus, so nothing is
computed below the toe.)

### Settlement at the neutral plane and of the pile head

UFC 3-220-20 Fig 6-19 and §6-5.8 item 6: the pile and the soil settle equally
at the neutral plane; the pile head settles that much plus the elastic
compression of the pile above the plane (Eq 6-45). Each row reports, from the
original ground:

- soil settlement at the plane = consolidation of the settling layers between
  the plane and the toe (Eqs 6-53, 6-54) + the settlement of the soil below the
  toe under the fill / drawdown alone (3 B' below the toe);
- pile settlement at the plane = settlement under the toe (bearing-stratum
  compression under the toe force and the fill / drawdown, Eqs 6-49 to 6-54)
  + elastic compression of the pile between the plane and the toe; set equal
  to the soil's when the toe is at its full resistance and the soil settles
  more (the toe is pushed down with it);
- pile head settlement = the larger of the two + the elastic compression above
  the plane. A method that sets the plane where the soil settles more than the
  pile can follow (a shallow plane) gets the soil's settlement there, so a
  higher plane gives more settlement, as UFC §6-5.8.4.2 says.

The headline `pile_settlement`, `soil_settlement_at_np` and
`elastic_shortening` are the headline row's. **Changed 2026-10-14**: before,
`pile_settlement` was the elastic shortening above the plane + the toe
settlement only, which left out the soil compressing below the plane (the
downdrag itself) and the pile compression below the plane, and
`soil_settlement_at_np` was measured from the soil at the toe level. The
floating pile of the tests (20 m in 25 m of settling clay, 300 kN): pile
settlement 34.6 mm -> 313.9 mm (the soil at the plane settles 313.8 mm); UFC
App. B-5.5 headline 1.98 mm -> 2.91 mm (adds 0.93 mm of pile compression below
the plane and toe settlement). The settlement check (`settlement_ok`) used
the larger of the old pile settlement and the toe-level soil settlement, so it
moves less.

## Validation

**UFC 3-220-20 App. B-5.5, Table B-28** (Example 5, 12 in PPC piles through
soft into stiff clay; `tests/test_downdrag.py::TestUFCAppendixB5NeutralPlane`).
Published NP 48 / 50 / 52 ft below grade and Pmax 120 / 126 / 133 kips for
0 / 50 / 100 % base mobilization; exact from the printed load-transfer rates
47.75 / 50.16 / 52.48 ft and 119.5 / 126.3 / 133.0 kips.

| Toe mobilization | Published | Exact from printed rates | Before 2026-10-14 | After |
|---|---|---|---|---|
| 0 % | 48 ft / 120 kips | 47.75 ft / 119.5 | 47.81 ft / 119.6 | 47.70 ft / 119.4 |
| 50 % | 50 ft / 126 kips | 50.16 ft / 126.3 | 50.16 ft / 126.1 | 50.14 ft / 126.2 |
| 100 % | 52 ft / 133 kips | 52.48 ft / 133.0 | 52.50 ft / 132.9 | 52.48 ft / 132.9 |

All still match the published values (to 1 ft and 1 kip; the test pins 0.1 ft
of the exact values). Before, the reported Pmax was 1.2-1.9 % above the toe +
positive friction at the same plane (GM-01); now they are equal. The test
enters the transition layer as alpha = 720/3000 (2.88 k/ft against the printed
2.9), which is the remaining 0.05 ft at 0 %. The neutral plane at the bearing
layer gives 48.0 ft, the base of the soft clay (B-5.7: "the position of the
neutral plane is below the bottom of the soft clay"), with Q_np = 120.0 kips
exactly. GEC-12 Fig 7-60 (p. 346) is graphical (curved shaft profile), so it
is cited, not reproduced.

**V-004, GEC-12 Vol III North Abutment, Block 17**
(`validation_examples/test_published_v001_v005.py`; the published Table D-6
shaft fed in through calibrated per-layer beta): published NP 54 ft below the
pile head, Qmax 486 kips, DF 285 kips at 100 % toe mobilization. Before
GM-01: NP 53.30 ft, Qmax 487.2, DF 286.2 kips (pile weight in the load
curve). Since GM-01 (2026-10-14): 54.00 ft, 486.6, 285.6 kips with the pile
weight left out, as the example does, and the default since 2026-10-15
(`test_v004_neutral_plane_and_drag_force`); 53.61 ft, 490.1, 289.1 kips with
`include_pile_weight=True` (`test_v004_with_pile_weight_included`). The
calibration helper integrates the shaft as the module does (its old node-sum
matched the old drag rule, not the old plane search), and the calibrated
shaft totals the published 344.1 kips.

The Endo ratios are as transcribed in `cgpr56.py`
(`ENDO_NEUTRAL_PLANE_RATIOS`); only the 0.75 ratio is exercised by a
published example (CGPR #56 §3.4.2 in `tests/test_cgpr56.py`). The Endo row
now equals `endo_method` on the module's own profile to rounding (2 % apart
before).

## GM-01: one integral of one profile (fixed 2026-10-14)

Before: the plane search (`_find_neutral_plane`) summed each segment's
friction with its DEEPER node's unit skin friction and interpolated the
crossing linearly; the reported drag load (`_compute_dragload`) summed each
segment with its SHALLOWER node's value; the positive friction
(`_compute_positive_resistance`) used the shallower node and skipped the part
segment between the plane and the next node; the rows' settlement and
imposed-plane checks interpolated the node curves linearly (a third variant).
With beta friction growing with depth, or a layer boundary inside a segment,
these differ by up to max(fs) x perimeter x segment: DD-1 at 0 % toe reported
Q_np 219.6 kN against 210.6 kN of positive friction below the same plane
(4.3 %); UFC B-5.5 1.2-1.9 %.

Now: every use goes through one integral, `_shaft_cumulative` / `_shaft_to`:
the exact integral of the reported unit skin friction profile read as
piecewise linear (the trapezoid rule on the nodes; a part segment integrated
exactly with the interpolated value). The plane search solves the resulting
quadratic inside the segment exactly, so at every intersection Q_np = mobilized
toe + positive friction to rounding. The grid carries every layer boundary
twice (the value above, then the value below: a repeated depth is a jump, as
in a CGPR #56 profile) and the water-table depths once (the kink in beta
friction), so the integral is exact for alpha layers (constant) and beta
layers (linear between kinks). Pinned by `TestQuadratureConsistency` (every
row's drag load against an independent integral of the reported profile; Q_np
= toe + positive friction at every intersection; the boundary as a jump), the
B-5.5 test, the Endo cross-check and the V-004 tests.

### How `auto` (the UFC 3-220-20 §6-7.4 procedure) locates the neutral plane

Every result states how (`neutral_plane_method`, `headline_method`,
`neutral_plane_basis`, `toe_force_mobilized_kN`, `warnings`). Source: UFC
3-220-20 Ch 6, §6-7.4 steps 3-5, §6-5.8.4.2 and Fig 6-19 (§6-5.8.4.3), checked
against the PDF.

1. **`force_equilibrium`** ("Neutral plane at 100% toe mobilization") — the
   load and resistance curves intersect. Side and base resistance are taken
   fully mobilized: UFC §6-7.4 step 3 calls this "a conservative approach for
   evaluating the drag force". When the intersection lies below the depth
   where the computed soil settlement falls to the pile's, a warning says the
   friction in between is counted as negative, and that full base
   mobilization is NOT conservative for settlement (§6-5.8.4.2: check 0 %,
   50 %, 100 %, all three in the table).
2. **`settlement_compatibility`** ("Mobilized toe resistance compatible with
   the toe penetration") — the curves do NOT intersect because the toe
   resistance exceeds everything the pile can carry to its toe
   (`Q_dead + the whole shaft in negative skin friction < R_toe`, + W when
   `include_pile_weight=True`). This is
   the common case with the default ULTIMATE toe (`Nt` from phi, 100-150 in
   dense sand): the base cannot be fully mobilized. The plane is where the
   soil and the pile settle equally (Fig 6-19): the soil settlement there =
   the settlement under the toe force the pile actually delivers + the
   elastic compression of the pile below the plane. The toe carries
   `Q_np - positive friction below` (floored at 0, in which case the friction
   below is only partly mobilized). For an end-bearing pile in a stratum
   stiffer than the compressible soil this is near the top of the bearing
   layer (§6-7.4 step 5); when the soil settles more than the pile all the way
   to the toe, the plane is AT the toe and the text says "end-bearing". With
   no settling soil at all there is no negative skin friction: plane at the
   head, zero drag. The warning gives the upper bound (plane at the toe, the
   whole shaft in negative skin friction) and how to intersect the curves
   instead (give `Nt` or `toe_mobilization` for the MOBILIZED toe resistance).
3. **`none`** — the dead load alone exceeds the total geotechnical resistance
   (toe + whole shaft). §6-7.4 step 5's limiting case: "there is no neutral
   plane". Reported at the head with zero drag, `geotechnical_ok = False`,
   and a `NO NEUTRAL PLANE` warning.

Before 2026-10-09 (live smoke G8) cases 2 and 3 silently reported the pile
TOE as the neutral plane and counted the whole shaft, non-settling bearing
sand included, as drag (DD-1: 409 kN where the 8 m of settling clay can give
151 kN).

Note on direction: more dead load moves the NP UP (UFC §6-7.4 step 5;
`Q_dead + F(0->z) = R_toe + F(z->L)`). A test used to assert the opposite
and passed only through the toe default.

### Limit States

1. **Structural (UFC Eq 6-80)**: LRFD factored demand must not exceed
   the pile's structural capacity:
   - `1.25·Q_dead + 1.10·(Q_np - Q_dead) ≤ P_r`
   - where `Q_np` = max pile load at neutral plane
   - The factor 1.25 applies to dead load, 1.10 to the drag force component

2. **Geotechnical**: The pile must have adequate bearing capacity below
   the NP. Per AASHTO/Fellenius/UFC, **dragload is NOT included** in the
   geotechnical check — it cancels at the neutral plane:
   - `Q_dead ≤ positive_skin + Q_toe`
   - **Any method that applies dragload to the geotechnical ULS is incorrect.**

3. **Settlement**: Pile head settlement (settlement at the neutral plane +
   elastic compression above it) ≤ allowable settlement (serviceability).

## Sign Conventions

- **Positive** = compression (dead load, dragload, pile weight all positive)
- **Skin friction** always computed as a positive magnitude `fs` (kPa);
  the code determines whether it acts as drag (above NP) or resistance (below NP)
  based on position relative to the neutral plane

## Skin Friction Methods

- **Cohesionless (beta method, UFC Eq 6-7)**: `fs = β · σ'v`
  where β = (1 - sin φ) · tan φ for NC soil (Fellenius 1991)
- **Cohesive (alpha method, UFC Eq 6-8)**: `fs = α · cu`
  where α defaults to 1.0 unless overridden

## UFC Equation Coverage

### Eq 6-45 — Elastic Compression of Pile
`δe = ΔQ · Z / (Ap · Ep)`
Implemented as `_compression_above()` (head to the neutral plane, under the
load curve) and inside `_pile_settlement_at()` (plane to toe). Integrates
Q(z)/(A*E) segment by segment.

### Eq 6-49/6-50 — Equivalent Footing Dimensions
`B' = B + z₂` and `L' = L + z₂`
For a single pile, B' = L' = pile diameter. Used in toe settlement.

### Eq 6-51 — 2V:1H Stress Distribution
`Δσz = Q / ((B' + z')(L' + z'))`
Used in `_compute_toe_settlement()` to distribute pile tip load into
the bearing stratum. The stress decreases with depth below the tip
as the loaded area spreads.

### Eq 6-52 — Stress Change at Neutral Plane
`Δσz = Q/((B'+z')(L'+z')) + Δσz,other`
Combined pile load stress + fill/GW stress at each depth.

### Eq 6-53 — Settlement of Clay (Modified Compression Indices)
Uses C_εc (modified compression index) and C_εr (modified recompression
index) instead of the traditional Cc/(1+e0) form:

- NC: `δs = C_εc · H₀ · log₁₀(σ_final/σ'v0)`
- OC stays OC: `δs = C_εr · H₀ · log₁₀(σ_final/σ'v0)`
- OC → NC: sum of recompression to σ'p + virgin compression beyond

The code accepts either:
- Traditional `Cc, Cr, e0` (auto-converts to C_ec = Cc/(1+e0))
- Direct `C_ec, C_er` (UFC notation)

### Eq 6-54 — Elastic Settlement of Coarse-Grained Soil
`δs = H₀ · (1+νs)(1-2νs) / ((1-νs)·Es) · Δσ'z`
For cohesionless settling layers. Requires `E_s` (Young's modulus)
and `nu_s` (Poisson's ratio) on the soil layer.

### Eq 6-80 — Structural Drag Force Check (LRFD)
`1.25·Q_d + 1.10·(Q_np - Q_d) ≤ P_r`
Factored demand vs factored structural resistance. The drag force
component `(Q_np - Q_d)` is the drag load, plus the pile self-weight to
the neutral plane only when `include_pile_weight=True`. The result
reports `structural_demand` (the left side of this equation) for
transparency.

## Consolidation Settlement (Traditional Form)

Also supports the standard e-log(p) method (Cc/Cr/e0/σ'v0/σ'p):

- NC: `Sc = Cc·H/(1+e0) · log₁₀((σ'v0 + Δσ)/σ'v0)`
- OC stays OC: `Sc = Cr·H/(1+e0) · log₁₀((σ'v0 + Δσ)/σ'v0)`
- OC → NC: `Sc = Cr·H/(1+e0) · log₁₀(σ'p/σ'v0) + Cc·H/(1+e0) · log₁₀((σ'v0+Δσ)/σ'p)`

These are internally converted to modified indices for computation.

## Toe Settlement — Equivalent Footing Approach

The bearing stratum settlement below the pile tip uses:

1. Equivalent footing: B' = L' = pile_diameter (single pile)
2. Influence zone: 3·B' below pile tip, discretized into sublayers
3. 2V:1H stress distribution (Eq 6-51) from pile tip load
4. Additional stress from fill/GW drawdown at each depth
5. Settlement via Eq 6-53 (clay) or Eq 6-54 (sand) for each sublayer

## Stress Changes from Loading

- **Fill placement**: `Δσ = γ_fill · H_fill` (uniform, 1-D assumption for extensive fill)
- **GW drawdown**: `Δσ = γ_w · drawdown` (for depths between old and new GWT)

## Edge Cases

- **No settling layers** (or settling layers with no fill / drawdown to load
  them): no consolidation settlement. If the curves still intersect (a
  mobilized `Nt`), that intersection is the NP as usual (V-004 runs this
  way); otherwise there is no negative skin friction: NP at the head, zero
  drag, stated in `neutral_plane_basis`.
- **Neutral plane at pile toe**: as the headline only when the soil settles
  more than the pile all the way down to the toe (end-bearing; the text says
  so), when the curves intersect exactly there, or when the caller asks for
  `pile_toe` (the upper bound). Never as a silent default; always in the
  table.
- **Pile overloaded** (`Q_dead` > toe + whole shaft): no neutral plane
  (method `none`), zero drag, geotechnical check fails, warning.
- **Soil settlement profile**: evaluated per sublayer at its MIDPOINT and
  accumulated from the toe up (zero at the toe level). Before 2026-10-09 each
  sublayer took the layer and stress at its top node, so the sublayer just
  below a boundary was given the upper layer's settlement and the top
  sublayer (sigma'v0 = 0) counted nothing.
- **Cohesionless settling layers**: Supported via elastic settlement (Eq 6-54).
  Requires E_s and nu_s parameters on the layer.

## Key Assumptions

- Pile is rigid relative to soil (valid for driven piles and short drilled shafts)
- Fill is extensive (1-D stress increase, no lateral stress distribution)
- Transition zone at NP is neglected (conservative for dragload)
- Settlement is ultimate (100% consolidation); use `settlement/time_rate.py`
  functions externally for time-dependent analysis
- Single pile (no group effects); for pile groups, use equivalent footing
  dimensions from `pile_group/` module

## References

1. Fellenius, B.H. (2004). "Unified design of piled foundations with emphasis
   on settlement analysis." ASCE GSP 125, pp. 253-275.
2. Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
3. AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 10.7.3.7.
4. UFC 3-220-20, 16 Jan 2025, Chapter 6, Eqs 6-45, 6-49–6-54, 6-80.
5. FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7 (Beta method); §7.3.5.7 and
   §7.3.6.1 (neutral plane, 0/50/100 % toe mobilization).
6. Briaud, J.-L. & Tucker, L.M. (1997). "Design and construction guidelines
   for downdrag on uncoated and bitumen-coated piles." NCHRP Report 393.
7. FHWA GEC-10 (FHWA-NHI-18-024), §10.6 (downdrag; defers to GEC-12
   §7.3.6.1) and §10.7 (post-liquefaction drag).
8. Siegel, T.C. et al. (2013), the neutral-plane downdrag procedure within
   LRFD, as cited by UFC 3-220-20 §6-5.8.4.2 and GEC-12 §7.3.6.1.
9. UFC 3-220-20, Appendix B-5 (Example 5), Table B-28: the validation case.

## CGPR #56 Method Family (`cgpr56.py`)

Added 2026-09-04 from "Downdrag and Drag Load on Piles" (Greenfield & Filz,
Virginia Tech CGPR #56, Feb 2009): `endo_method` (Section 3.2.1),
`poulos_method` (3.2.2), `fellenius_method_cgpr56` (3.2.3),
`pileneg_procedure` (3.2.4 — the PILENEG program's calculation procedure as
documented in the report, not the original code), `rigid_block_method`
(4.3.1), `drag_load_reduction_method` (4.3.2, Jeong & Briaud 1994 Table 4.1
with the Figure 4.2 conservative extensions: constant below s/d = 2.5, step
to A = 1.0 above s/d = 5), plus `downdrag_method_comparison` and
`consolidation_settlement_profile`. All take user-supplied piecewise-linear
skin-friction / settlement profiles and are dimensionally consistent (any
one unit system). Validation: the report's Section 3.4 worked example is
reproduced method-by-method in `tests/test_cgpr56.py` (Table 3.7 within
0.5% on forces, 0.1 ft on neutral planes with the report's discretization;
achieved-vs-published stated in each test docstring).

### `fellenius_method_cgpr56` vs the existing `DowndragAnalysis`

Both locate the neutral plane at the intersection of the load and resistance
curves. Documented differences:

1. **Inputs**: `DowndragAnalysis` builds unit skin friction from soil
   parameters (beta/alpha per layer) and effective stress; the CGPR #56
   function takes the fs-vs-depth profile directly (report convention).
   Both read the profile as piecewise linear, a repeated depth a jump.
2. **Pile self-weight**: UFC Fig 6-26, GEC-12 §7.3.6.1 and the CGPR #56 /
   Hannigan et al. (1997) formulation omit it, and so does
   `DowndragAnalysis` by default since 2026-10-15
   (`include_pile_weight=True` adds it). With matched fs/toe inputs the two
   implementations agree on the worked example within 1.5%
   (`tests/test_cgpr56.py::TestCrossCheckExistingFellenius`).
3. **Settlement**: both report the settlement at the neutral plane + the
   elastic compression above it. CGPR #56 spreads the pile / group load 2:1
   from an equivalent footing AT the neutral plane; `DowndragAnalysis` takes
   the free-field compression of the settling layers below the plane and
   spreads the toe force only below the toe. The CGPR #56 form matches the
   report's published example.
