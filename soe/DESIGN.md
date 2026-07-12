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

## Basal heave — three methods (v5.3)
`stability.py` offers three basal-heave checks; all are additive/independent:
- `check_basal_heave_terzaghi` — Terzaghi (1943) inverted-footing bearing.
- `check_basal_heave_bjerrum_eide` — Bjerrum-Eide (1956) bearing ratio
  FOS = cu·Nc/(γH+q), Nc from the H/Be, Be/Le table. NO sidewall shear.
- `check_basal_heave_caltrans` (NEW) — the Caltrans T&S / Terzaghi limiting-
  equilibrium force balance that INCLUDES the sidewall-shear resistance
  S = cu·H on the vertical failure plane:
  resisting F_RS = cu·Nc·(0.7B); driving F_dr = 0.7B·H·γ + 0.7B·q − cu·H;
  FS = F_RS/F_dr. Nc from the Skempton/Bjerrum-Eide form
  5.14(1+0.2 B/L)(1+0.2 H/B) (H/B capped at 2.5; ~7.7 at H/B=2, L/B=3), or a
  chart-read `Nc`. The 0.7B block width and the side-shear term make it less
  conservative than the bearing-ratio method — it reproduces Caltrans Ex 10-2
  (FS=1.54). Validated in `validation_examples` (V-014).

## FHWA/GEC-4 apparent-pressure anchored wall (v5.3)
`earth_pressure.fhwa_apparent_pressure_anchored_wall(H, anchor_depths, γ, φ, …)`
builds the FHWA apparent earth-pressure envelope and distributes it to the
anchors by the tributary (hinge) method:
pe = 0.65·Ka·γ·H²/(H − H1/3 − Hn+1/3) (H1 = depth to top anchor, Hn+1 = lowest
anchor to base; total load = 1.3× the triangular Rankine total). Returns pe, the
surcharge term ps = Ka·q, per-anchor tributary loads TH_i (top/interior/bottom
formulas), subgrade reaction R, hinge moments, and anchor design loads
DL = TH·s/cos(incl). Reproduces GEC-4 Design Example 1 (V-016, 2-anchor) natively
and the Caltrans Ex 8-1 single-anchor envelope (pe = σ_a, PT, upper tributary).

## Log-spiral passive coefficient (v5.3)
`earth_pressure.caquot_kerisel_Kp(phi, delta)` — the Caquot-Kerisel (1948)
log-spiral passive coefficient Kp' = R·Kp0 (base Kp0 at δ=φ; reduction R for
δ/φ from Caltrans Matrix 4-1 / NAVFAC DM-7.2). Unlike Coulomb it does not
over-predict Kp at high δ/φ. φ=30, δ/φ=0.5 → 4.70 (V-013). Mirrored in
`sheet_pile.earth_pressure` (a byte-identical copy of the table) and selectable
there as `pressure_method="log_spiral"`.

**Source basis (Caquot-Kerisel Kp0 base values + R reduction grid).** These are
CHART/TABLE reads. Only the **φ=30 → Kp0=6.30 entry is a verified chart read**
(Caltrans T&S Manual Fig 4-20, anchored by V-013 Example 8-1). The neighbouring
Kp0 values (other φ) and the full R(δ/φ) grid are attributed to
Caquot-Kerisel 1948 / NAVFAC DM-7.2 / Caltrans Matrix 4-1 but were **not
confirmed per-value from a source in hand**. This is the **top wiki-wishlist
item** for this module: re-verify the Kp0 column and R grid against Caquot &
Kerisel (1948) tables, NAVFAC DM-7.2 passive-pressure charts, and Caltrans
Fig 4-20 / Matrix 4-1. (A DM7.2 Kp chart in the in-house `dm7` figure catalog
would allow a vision cross-read of the 5 unverified Kp0 values — a recommended
follow-up.)

## Ground anchor design (built — provenance)
`anchor_design.py` is fully implemented (grouted-anchor bond capacity, tendon
selection). **Source basis:** the presumptive ultimate grout-to-ground **bond
stress table** (`_BOND_STRESS` in `anchor_design.py`, sand/clay/rock) are
**nominal presumptive values per FHWA GEC-4 (Ground Anchors & Anchored Systems,
FHWA-IF-99-015) Table 4** (the code already carries the "nominal values" caveat);
tendon strand/bar data follow **PTI DC35.1** and **ASTM A722** (bar) / **A416**
(strand). **RED FLAG — UNBENCHMARKED:** the anchor-design tables have **no
worked-example numeric validation** (no `validation_examples` anchor), so they
rest on the transcribed nominal values alone. Wiki-wishlist: GEC-4 Table 4,
PTI DC35.1 strand/bar tables, ASTM A722/A416.

## Future Work

- Full ACI 318 concrete design for secant/diaphragm walls
- Full AISC connection design for wales, struts, bracing
- Beam-spring (FD) SSI analysis (adapt from lateral_pile/solver.py)
- Lagging design between soldier piles
- Raker and corner bracing (3D configurations)
- Seismic apparent pressure (M-O increments)
- Stability checks: basal heave, bottom blowout, piping (Phase 2)
