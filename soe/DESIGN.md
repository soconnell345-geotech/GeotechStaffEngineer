# SOE (Support of Excavation) Module — Design Notes

## Scope

Multi-level braced and cantilever excavation support walls including:
- Soldier pile walls (HP sections + lagging)
- Sheet pile walls (Z-type and U-type sections)
- Secant/tangent pile walls (structural demands only)
- Diaphragm (slurry) walls (structural demands only)

**Excluded:** Soil nail walls (separate `soil_nail/` module), full ACI/AISC
structural design (future work), beam-spring SSI analysis (future work).

## Theory

### Apparent Earth Pressure (Terzaghi-Peck)

For braced/anchored excavations, classical Rankine/Coulomb pressure
distributions do not apply because strut installation changes the
deformation pattern. Terzaghi & Peck (1967) and Peck (1969) developed
empirical apparent pressure envelopes from field measurements:

| Soil Type | Shape | Max Ordinate |
|-----------|-------|--------------|
| Sand | Uniform (rectangular) | p = 0.65 × Ka × γ × H |
| Soft-medium clay (N > 4) | Uniform | p = Ka × γ × H, Ka = 1 − m(4cu/γH) |
| Stiff clay (N ≤ 4) | Trapezoidal | p = (0.2 to 0.4) × γ × H |

Where N = γH/cu is the stability number and m = 1.0 for most cases.

### Tributary Area Method

From the California Trenching and Shoring Manual:
1. Divide the wall into spans between support levels
2. Load each span with the apparent pressure envelope
3. Treat each span as simply supported
4. Support reactions = tributary loads from adjacent spans
5. Max moment in critical span governs wall section

### Embedment (Below Lowest Support)

Below the lowest support level, the wall acts as a cantilever resisting
passive earth pressure. Embedment D is found by moment balance about
the lowest support until:

    M_passive ≥ M_active

Then D_design = 1.2 × D (20% increase per USACE EM 1110-2-2504).

### Cantilever Walls

For unbraced walls (typically H ≤ 4–5 m in sand, H ≤ 3 m in clay),
classical Rankine active/passive pressure is used with limit equilibrium
to find embedment. Moments about the wall base determine embedment
depth and maximum moment.

## Units

All SI:
- Lengths: meters (m)
- Pressures: kilopascals (kPa)
- Forces: kilonewtons per meter of wall (kN/m)
- Moments: kilonewton-meters per meter of wall (kN·m/m)
- Unit weights: kN/m³
- Angles: degrees
- Section modulus: cm³ (for steel section selection)
- Section properties in database: US customary (in, in², in³, in⁴, lb/ft)
  as published; conversion to SI where needed.

## Sign Conventions

- Depth z: positive downward from top of wall
- Earth pressure: positive toward excavation (active pushes wall in)
- Bending moment: positive = tension on excavation side
- Support reactions: positive = compressive load in strut

## Steel Section Selection

Section databases include manufacturer data:
- **HP sections**: AISC 16th Ed (HP8–HP14)
- **Sheet pile sections**: Nucor Skyline PZ series + ArcelorMittal AZ series
- **W sections**: AISC 16th Ed (common wale/strut sizes W14–W24)

Selection uses Allowable Stress Design (ASD):
- Fb = 0.66 × Fy (compact sections)
- Required Sx = M_max / Fb

## References

1. Terzaghi, K. & Peck, R.B. (1967). *Soil Mechanics in Engineering Practice*, 2nd Ed.
2. Peck, R.B. (1969). "Deep Excavation and Tunneling in Soft Ground." SOA Report, 7th ICSMFE.
3. FHWA-IF-99-015: *Ground Anchors and Anchored Systems* (GEC-4).
4. California Dept. of Transportation (2011). *Trenching and Shoring Manual*.
5. USACE EM 1110-2-2504: *Design of Sheet Pile Walls*.
6. AISC (2017). *Steel Construction Manual*, 16th Edition.
7. Nucor Skyline. *Steel Sheet Pile Catalog*.
8. ArcelorMittal. *Steel Sheet Piling Design Manual*.
9. PTI DC35.1: *Recommendations for Prestressed Rock and Soil Anchors*.

## Future Work

- Full ACI 318 concrete design for secant/diaphragm walls
- Full AISC connection design for wales, struts, bracing
- Beam-spring (FD) SSI analysis (adapt from lateral_pile/solver.py)
- Lagging design between soldier piles
- Raker and corner bracing (3D configurations)
- Seismic apparent pressure (M-O increments)
- Ground anchor design per GEC-4/PTI (Phase 3)
- Stability checks: basal heave, bottom blowout, piping (Phase 2)
