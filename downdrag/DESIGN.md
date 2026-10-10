# Downdrag Module — Design Notes

## Theory

Pile downdrag (negative skin friction) occurs when soil around a pile settles
more than the pile itself. The settling soil drags the pile downward, inducing
additional compressive load. Common triggers:

- **Fill placement**: New embankment causes consolidation of underlying soft soils.
- **Groundwater drawdown**: Lowering the water table increases effective stress,
  causing consolidation.

## Fellenius Unified Method (2004/2006)

The analysis finds the **neutral plane** — the depth where pile settlement equals
soil settlement. Above the NP, soil settles more than the pile, inducing downward
friction (dragload). Below the NP, the pile settles more than the soil, and friction
acts as resistance (positive skin friction).

### Force Equilibrium

Two force curves are constructed along the pile:

1. **Load from top**: `Q_top(z) = Q_dead + W_pile(0→z) + ∫₀ᶻ fs·P·dz`
   (dead load + pile weight + accumulated friction from surface)

2. **Resistance from bottom**: `Q_bot(z) = Q_toe + ∫_L^z fs·P·dz`
   (toe resistance + accumulated friction from tip upward)

The neutral plane is where `Q_top(z_np) = Q_bot(z_np)`. The maximum axial
load in the pile occurs at this depth.

## Neutral-plane methods (survey, 2026-10-09)

The owner: "Downdrag has many methods and it should be open to multiple
assumptions. See DM 7.2 or the FHWA driven pile manual." This table lists what
each reference in the repo actually states (checked against the source PDFs
in `geotech-references/docs/`; page numbers are the printed ones). UFC
3-220-20 = 16 Jan 2025 (DM 7.2); GEC-12 = FHWA-NHI-16-009 Vol I; GEC-10 =
FHWA-NHI-18-024; CGPR #56 = Greenfield & Filz (2009), cited by section only
(not public-domain, not in the repo).

| Method (`neutral_plane_method`) | Source | Assumptions | When to use |
|---|---|---|---|
| **Force equilibrium, full mobilization** (`force_equilibrium`, `toe_mobilization=1`; first step of `auto`) | UFC §6-7.4 steps 3-5, Fig 6-26 (pp. 476-478); GEC-12 §7.3.5.7 (pp. 334-336, Fig 7-49) and §7.3.6.1 step 5, Fig 7-60 point A (p. 346); GEC-10 §10.6.2, Fig 10-20 (pp. 10-47/48, defers to GEC-12 §7.3.6.1); CGPR #56 §3.2.3 | Plane where Q_d + cumulative negative friction meets toe + cumulative positive friction; side and base fully mobilized; unfactored permanent load only, transient load excluded (UFC step 4) | Drag force for the STRUCTURAL strength limit state: UFC step 3 calls it "a conservative approach for evaluating the drag force". Not conservative for settlement (UFC §6-5.8.4.2) |
| **Force equilibrium, partial toe: 0 / 50 / 100 %** (`toe_mobilization`; all three always in the comparison) | UFC §6-5.8.4.2 (p. 447) and App. B-5.5, Table B-28 (p. 640); GEC-12 §7.3.6.1 steps 2, 4, 5, Figs 7-55 to 7-60 (pp. 342-346) — FHWA's recommended downdrag method (Siegel et al. 2013) | Side fully mobilized (~0.1 in of movement), base at a fraction of nominal (full base needs 4-10 % of the width). Less base -> shallower plane, less drag, more settlement | SERVICE limit (settlement): bracket 0 / 50 / 100 %; if the conclusion does not change, stop; if it does, refine with t-z / q-z curves (UFC §6-5.8.4.2) |
| **Settlement compatibility** (`settlement_compatibility`; second step of `auto`) | UFC §6-5.8.4.3, Fig 6-19 (pp. 449-450): pile and soil settle equally at the plane; the soil-minus-pile settlement at the toe is the toe penetration; a q-z curve makes the toe force compatible with that penetration (Fellenius 2021). GEC-12 §7.3.5.7 (p. 334). CGPR #56 §3.2.4 (PILENEG, elastic toe) | As implemented: elastic-perfectly-plastic toe, the elastic penetration from the equivalent-footing compression of the bearing stratum (UFC Eqs 6-49 to 6-54) standing in for a q-z curve; side fully mobilized (partly when the load at the plane is less than the friction below). Never below the full-mobilization crossing; if the soil still out-settles the elastic pile there, the toe yields and the plane IS that crossing | When the ultimate toe makes force equilibrium impossible (the curves do not cross: end-bearing piles); whenever the toe force should follow the toe movement |
| **Bearing-layer rule** (`bearing_layer_top`) | UFC §6-7.4 step 5 (p. 477): the plane is "near the interface of the column and the bearing layer for an end-bearing column bearing in a stratum that is much stiffer than the compressible soil"; §6-5.7 (p. 438, Fellenius 2021: end-bearing -> large drag, small downdrag). CGPR #56 §3.2.2 (Poulos, Eq 3.7) caps the plane at the bearing-layer top | End-bearing pile; the toe below all the settling soil, in a much stiffer stratum. A rule, not an equilibrium (checked: the toe must be able to carry Q_np minus the friction below) | Estimate / check for end-bearing piles. Not for a floating pile (toe inside the settling soil) |
| **Pile toe** (`pile_toe`; role "bound") | GEC-12 §7.3.5.7 (p. 336): Goudreault & Fellenius (1994) place the plane at the toe "for most cases" of GROUP settlement (piles below the plane reinforce the soil); CGPR #56 Table 3.1 "end bearing" = 1.0 L | The whole shaft as drag | Group-settlement simplification; for a single pile's drag it is the UPPER BOUND, never a silent default (G8) |
| **Endo ratio** (`endo` + `endo_bearing_condition`) | CGPR #56 §3.2.1, Table 3.1 (Endo et al. 1969; Little 1994) | Plane at 0.67 / 0.75 / 1.0 of the embedment for floating / stiff-flexible bearing / end bearing piles through consolidating clay; drag = full friction above it; no settlement | Estimate / check level only |

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
| Side resistance on the effective-stress profile that INCLUDES the settlement-causing change (Siegel et al. 2013) | UFC §6-7.4 step 2 (p. 476) | `skin_friction_stress="final"`; default `"initial"` (the module's basis before 2026-10-09) — owner's call |
| Transient load excluded from the head load (it reverses the friction while it acts) | UFC §6-7.4 step 4 (p. 477) | Yes (Q_dead only) |
| Drag excluded from the geotechnical strength check, included in structural strength and in settlement | UFC Table 6-30 (p. 437); GEC-12 §7.3.6 (pp. 336-337); GEC-10 §10.6.1 (p. 10-46; notes AASHTO treats DD as a load in geotechnical strength) | Yes (UFC / FHWA) |
| Load factor 1.10 on the drag force (1.25 on Q_d) | UFC Eq 6-80 (p. 478); GEC-12 Eq 7-70 (pp. 346-347: MnDOT 1.1, no AASHTO factor yet) | Yes |
| Post-liquefaction drag = 50 % of the pre-earthquake beta side resistance in liquefied layers | GEC-10 §10.7 (p. 10-52) | Not implemented |

### Options, comparison and the default

`DowndragAnalysis(neutral_plane_method=..., toe_mobilization=...,
endo_bearing_condition=..., skin_friction_stress=...)`. Whatever is chosen,
every basis above is evaluated on the same inputs and returned in
`neutral_plane_comparison` (one row each: name, method, role, source,
assumptions, applies, reason; and when it applies the plane depth, drag load,
load at the plane, toe force, and the soil and pile settlement AT the plane).
Both settlements at the plane are measured from the soil at the toe level
(soil: consolidation of the soil between the plane and the toe; pile: toe
penetration + pile compression below the plane), so equal values mean the
basis is settlement-compatible (UFC Fig 6-19). A basis that does not apply is
marked with the reason (curves that do not cross, a floating pile for the
bearing-layer rule, a toe that could not carry the drag a rule implies, a
missing Endo condition, an overloaded pile). Choosing a basis that does not
apply raises a ValueError with that reason — never a silent fallback.

**Default: `auto` = UFC 3-220-20's procedure, unchanged from G8.** UFC
3-220-20 is the governing reference for DoD work and its §6-7.4 procedure is
force equilibrium (full mobilization unless `toe_mobilization` is given),
with settlement compatibility when the curves do not cross. The numbers V-004,
the CGPR #56 cross-check and the live-smoke keys rest on do not move. The
alternatives are always in the result, and when they differ materially the
result also carries a `judgment` block and a warning saying so:

```
"judgment": {"question": "Which neutral-plane basis governs for this pile?",
             "options": [{"name", "source", "assumptions", "applies", "result"}],
             "used": "<row name>", "why": "<why, or that the caller chose it>"}
```

Material = among the applicable design bases plus the one used (the pile-toe
bound and a compatibility row with no computed soil settlement do not count),
neutral planes more than max(0.5 m, 5 % of L) apart, or drag loads more than
max(10 kN, 10 % of the largest) apart (`NP_SPREAD_*`, `DRAG_SPREAD_*` in
`analysis.py`). A clear-cut case (a friction pile floating in settling clay:
every applicable basis within tolerance) carries no `judgment`.

DD-1 (8 m of clay over dense sand, 0.4 m pile, 15 m, 40 kPa fill, Q_dead 0):
full and 50 % toe — not applicable (the 2482 kN ultimate toe cannot be
mobilized); 0 % toe 9.5 m / 191 kN; settlement compatibility 8.0 m / 151 kN
(used); bearing-layer rule 8.0 m / 151 kN; pile toe (bound) 15.0 m / 409 kN;
Endo needs a condition. Judgment emitted.

Validation: UFC 3-220-20 App. B-5.5 Table B-28 (Example 5, PPC piles through
soft into stiff clay; `tests/test_downdrag.py::TestUFCAppendixB5NeutralPlane`).
Published NP 48 / 50 / 52 ft below grade and Pmax 120 / 126 / 133 kips for 0 /
50 / 100 % base mobilization; achieved 47.81 / 50.16 / 52.50 ft and 119.6 /
126.1 / 132.9 kips (exact from the printed rates: 47.75 / 50.16 / 52.48 ft).
The bearing-layer rule gives 48.0 ft, the base of the soft clay (B-5.7: "the
position of the neutral plane is below the bottom of the soft clay"). GEC-12
Fig 7-60 (p. 346) is graphical (curved shaft profile), so it is cited, not
reproduced. The Endo ratios are as transcribed in `cgpr56.py`
(`ENDO_NEUTRAL_PLANE_RATIOS`); only the 0.75 ratio is exercised by a
published example. Endo: CGPR #56 §3.4.2 worked example in `tests/test_cgpr56.py`,
and the option cross-checked against `endo_method`.

Known quadrature mismatch (pre-existing, not changed): the crossing sums
friction with each segment's lower-node value, the reported drag and positive
friction with its upper-node value; they differ by at most about
max(fs) x perimeter x dz (~4 % on DD-1 at 0 % toe).

### How `auto` locates the neutral plane (`neutral_plane_method` in the result)

Every result states its basis (`neutral_plane_method`, `neutral_plane_basis`,
`toe_force_mobilized_kN`, `warnings`). Source: UFC 3-220-20 Vol 2 (DM 7.2)
Ch 6, §6-7.4 steps 3-5, §6-5.8.4.2 and Fig 6-19 (§6-5.8.4.3), as transcribed
in `geotech_references/dm7_2/text/chapter06.json`.

1. **`force_equilibrium`** — the curves cross. Side and base resistance are
   taken fully mobilized: UFC §6-7.4 step 3 calls this "a conservative
   approach for evaluating the drag force". Unchanged from the original
   implementation (V-004 and the CGPR #56 cross-check pin it). When the
   crossing lies below the depth where the computed soil settlement falls to
   the pile's, a warning says the friction in between is counted as drag,
   and that full base mobilization is NOT conservative for settlement
   (§6-5.8.4.2: check 0 %, 50 %, 100 % base mobilization via `Nt`).
2. **`settlement_compatibility`** — the curves do NOT cross because the toe
   resistance exceeds everything the pile can carry to its toe
   (`Q_dead + W + whole shaft as drag < R_toe`). This is the common case with
   the default ULTIMATE toe (`Nt` from phi, 100-150 in dense sand): the base
   cannot be fully mobilized, so force equilibrium alone does not place the
   NP. It is placed where the soil and pile settle equally (Fig 6-19): soil
   settlement (measured from the toe level) = toe penetration under the toe
   force the pile actually delivers + elastic compression of the pile below
   the NP. The toe carries `Q_np - positive friction below` (floored at 0,
   in which case the friction below is only partly mobilized). For an
   end-bearing pile in a stratum stiffer than the compressible soil this is
   near the top of the bearing layer (§6-7.4 step 5); when the soil settles
   more than the pile all the way to the toe, the NP is AT the toe and the
   basis says "end-bearing". With no settling soil at all, there is no
   negative skin friction: NP at the head, zero drag. The warning gives the
   upper bound (NP at the toe, whole shaft as drag) and how to use force
   equilibrium instead (give `Nt` for the MOBILIZED toe resistance).
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

### Settlement Compatibility

At the neutral plane:
- **Soil settlement** = cumulative consolidation settlement from the settling
  zone (accumulated from the bottom of the settling zone upward)
- **Pile settlement** = elastic shortening above NP + settlement of the
  bearing stratum below the pile tip

These should be equal (or close) at the NP for full compatibility.

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

3. **Settlement**: Pile settlement ≤ allowable settlement (serviceability).

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
Implemented as `_compute_elastic_shortening()`. Integrates Q(z)/(A*E)
from pile head to the neutral plane.

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
component `(Q_np - Q_d)` includes both dragload and pile self-weight
to the neutral plane. The result reports `structural_demand` (the left
side of this equation) for transparency.

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
  them): no consolidation settlement. If the curves still cross (a mobilized
  `Nt`), force equilibrium places the NP as usual (V-004 runs this way);
  otherwise there is no negative skin friction: NP at the head, zero drag,
  stated in the basis.
- **Neutral plane at pile toe**: only when the soil settles more than the
  pile all the way down to the toe (end-bearing; basis says so), when the
  curves cross exactly there, or when the caller asks for `pile_toe` (the
  upper bound). Never as a silent default; always in the comparison as the
  bound.
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

Both implement the Fellenius neutral-plane force equilibrium (load curve vs
resistance curve). Documented differences — the existing `DowndragAnalysis`
behavior is unchanged:

1. **Inputs**: `DowndragAnalysis` builds unit skin friction from soil
   parameters (beta/alpha per layer) and effective stress; the CGPR #56
   function takes the fs-vs-depth profile directly (report convention).
2. **Pile self-weight**: `DowndragAnalysis` includes it in the load curve
   (UFC-consistent); the CGPR #56 / Hannigan et al. (1997) formulation
   omits it. With pile weight zeroed and matched fs/toe inputs the two
   implementations agree on the worked example within 1.5%
   (`tests/test_cgpr56.py::TestCrossCheckExistingFellenius`).
3. **Settlement**: `DowndragAnalysis` reports settlement as elastic
   shortening + toe-zone consolidation below the tip; CGPR #56 reports
   free-field settlement at the neutral plane (with the pile/group load
   spread 2:1 from an equivalent footing AT the neutral plane) + elastic
   compression above it. These answer slightly different questions; the
   CGPR #56 form matches the report's published example.
