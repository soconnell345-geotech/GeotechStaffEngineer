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

- **No settling layers**: If no layers have `settling=True`, there is no
  consolidation settlement and no downdrag. The neutral plane defaults to
  the pile tip.
- **Neutral plane at pile tip**: Occurs when dead load + dragload exceeds
  the total shaft + toe capacity. This means the pile is overloaded.
- **Neutral plane at pile head**: Occurs when there is minimal dragload
  (very little settlement). The pile behaves normally.
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
5. FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7 (Beta method).
6. Briaud, J.-L. & Tucker, L.M. (1997). "Design and construction guidelines
   for downdrag on uncoated and bitumen-coated piles." NCHRP Report 393.
