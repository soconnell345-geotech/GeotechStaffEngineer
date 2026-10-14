# sheet_pile — Sheet Pile Wall Design

## Purpose
Analyzes cantilever and anchored sheet pile walls using Rankine or Coulomb
earth pressure theory. Computes embedment depth, maximum moment, and
anchor force.

## References
- Rankine (1857) — active/passive pressure coefficients
- Coulomb (1776) — with wall friction
- USS Steel Sheet Piling Design Manual

## Files
- `earth_pressure.py` — Ka, Kp, active/passive pressure distributions
- `cantilever.py` — cantilever wall embedment + moment
- `anchored.py` — free-earth support method
- `results.py` — SheetPileResult with summary()/to_dict()

## Public API
```python
analyze_sheet_pile(wall_type, excavation_depth, layers, ...) -> SheetPileResult
```

## Key Notes
- earth_pressure module reused by retaining_walls/earth_pressure.py
- Layers are dicts with phi, c, gamma, thickness

## Cantilever safety basis (SP-3, changed in v5.1)
The classic simplified cantilever method uses ONE of two safety bases:
1. **FS on passive resistance** (~1.5-2.0) with NO embedment increase, or
2. **Unfactored passive (FS ≈ 1.0)** with a 20-40% embedment increase.

Prior to v5.1 `analyze_cantilever` applied both (FOS_passive = 1.5 AND a
hardcoded 1.2× depth increase) — a double safety margin. Since v5.1 the
default is basis 1: `FOS_passive = 1.5`, `embedment_increase = 1.0` (a new,
documented parameter). Pass `FOS_passive = 1.0, embedment_increase = 1.2-1.4`
for basis 2. Result records both `embedment_converged` and
`embedment_increase`; calc_steps renders whichever basis was used.
**Behavior change:** default design embedment is 1/1.2 of the pre-v5.1 value
(the old default combination was conservative double-counting); max moment is
computed over the (shorter) design embedment accordingly.
Anchored walls (free earth support) are unchanged: factored passive, no
depth increase.

## Log-spiral passive coefficient (SP-4, v5.3)
`earth_pressure.caquot_kerisel_Kp(phi, delta, Kp_initial=None)` adds the
Caquot-Kerisel (1948) log-spiral passive coefficient: Kp' = R·Kp0, where Kp0 is
the passive coefficient at δ=φ (digitized from the C-K chart; the φ=30 anchor is
the Caltrans Fig 4-20 value 6.30) and R = Kp(δ)/Kp(δ=φ) ≤ 1 is the wall-friction
reduction (Caltrans T&S Matrix 4-1 / NAVFAC DM-7.2, tabulated φ 30-35, δ/φ
0.40-0.50). Unlike Coulomb (planar wedge), the log spiral does NOT over-predict
Kp at high δ/φ. For φ=30, δ/φ=0.5, Kp' = 6.30·0.746 = 4.70 (Caltrans Ex 8-1).
Selectable via `pressure_method="log_spiral"` in `analyze_cantilever` /
`analyze_anchored` (uses Coulomb active + the log-spiral passive; set the layer
`wall_friction_deg` = δ). Default remains Rankine/Coulomb. Validated in
`validation_examples` (V-013).

## Anchored wall shear and moment (GM-69, fixed 2026-10-14)
`analyze_anchored` finds the embedment D by moments about the anchor and the
anchor force T by horizontal equilibrium (both unchanged), then integrates the
net pressure down the wall for the shear V(z) and moment M(z). Until
2026-10-14 that integration started the shear at -T at the TOP of the wall
instead of applying T at the anchor, so every moment below the anchor was too
large by T x anchor_depth; the pressures below the excavation level were also
left out, and the maximum was searched only above it. Now: the reaction T
enters at `anchor_depth` (within the step that contains it), the active,
factored passive and net water pressures below the excavation level are
integrated as in the equilibrium loops, and the maximum |M| is taken over the
whole wall. At the toe V = 0 and M ~ 0 (free earth support closes); the
points are kept on the result (`profile`, not in `to_dict`) and printed by
`get_tables()`.

**Pinned against** the USS Steel Sheet Piling Design Manual (United States
Steel, 1984 reprint by FHWA), Design Example No. 2, "Design of Anchor Sheet
Pile Wall - Cohesive Soil With Sand Backfill - Free Earth Support Method",
pp. 101-102: H = 20 ft, anchor 5 ft below the top, low water 6 ft (same both
sides, so effective unit weights and no net water), sand backfill γ = 115 /
γ' = 60 pcf with Ka = 0.33 to 8.5 ft, soft clay c = 400 psf, γ' = 65 pcf to
the dredge line, medium soft clay below with c = 500 psf (the factored
strength, FS = 1.70). Published: D² + 30D = 257.4, "D = 7.0 ft" (the root
is 6.96 ft); T = 6077 - 412(7) = 3193 lb/ft. Module (Ka = 0.33 entered as
φ = 30.25°, FOS_passive = 1.0): D = 6.92 ft, T = 3188 lb/ft. The example
prints no maximum moment; from its own pressure diagram and T = 3193 lb/ft
the shear is zero 15.51 ft below the top and M_max = 15,773 ft-lb/ft by hand
(the zero-shear procedure of the manual's Example No. 1, p. 98). Module:
15,649 ft-lb/ft at 15.42 ft (-0.8 %); the old loop gave 31,590 (+100 %).
The manual's Example No. 1 (granular soil, pp. 95-98) prints M_max but has a
line load the module does not take and leaves the surcharge out of its own
maximum-moment step, so it is not used as the pin. Cross-checked against the
independent `soe.free_earth.solve_about_support` (exact support load,
trapezoid integration): uniform sand φ = 30°, γ = 18, H = 6 m, anchor at
1.5 m, FOS 1.0: M_max 113.2 vs 113.8 kN-m/m. Hand check, same wall with
FOS_passive = 1.5: T = 75.6 kN/m, zero shear at
z0 = sqrt(2T/(Ka γ)) = 5.02 m, M = Ka γ z0³/6 - T (z0 - a) = 139.6 kN-m/m;
module 138.9 (old loop 252.3).

What moved (D and T are unchanged everywhere; M_max in kN-m/m):

| case | old | new |
|---|---|---|
| USS Ex. 2 (SI) | 140.5 at 4.70 m | 69.6 at 4.70 m |
| sand φ30 γ18, H 6, a 1.5, FOS 1.5 | 252.3 at 4.99 m | 138.9 at 4.99 m |
| sand φ32 γ18, H 6, a 1.5 (an eval question) | 216.8 | 117.0 |
| sand φ30, H 7, a 1.5 (test_basic_anchored) | 384.5 | 234.3 |
| sand φ32, H 5, a 1.5 (foundry tier-1) | 133.8 | 61.5 |
| φ32 γ19, H 6, a 1.0, water at 0 | 1041.2 at 5.98 m | 794.5 at 6.27 m (below the excavation level, now searched) |
| V-013 wall, classical Rankine, FOS 1.3 | 596.5 | 187.1 |

`soe` was checked for the same pattern: `soe.free_earth._shear_moment`
subtracts each support load at its own depth (`V - force * (z >= depth)`),
so it never had this error.
