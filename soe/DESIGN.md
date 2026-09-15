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

### Pressure above the excavation (braced walls)

The apparent-pressure envelope carries no surcharge or water. FHWA GEC-4
Sec 5.2.4: "Water pressures and surcharge pressures should be added
explicitly to the diagram" — so the braced design pressure
(`free_earth.apparent_pressure_profile`) is:

    p(z) = envelope(z) + K·q + u(z)

- Sand envelope: effective stress, 0.65·Ka·γ'avg·H with γ' = γ − γw below the
  water table; K = envelope Ka; hydrostatic water u added below `gwt_depth`.
- Clay envelopes: total stress; K = 1.0 (undrained, φ = 0); no water added.

### Tributary Area Method

From the California Trenching and Shoring Manual and GEC-4 Sec 5.4:
1. Divide the wall into spans between support levels
2. Load each span with the design pressure p(z) above
3. The span above the first support is a **cantilever** (moment p·d1²/2 for a
   uniform envelope); the spans below are simply supported (hinge method)
4. Support reactions = tributary loads; the bottom span's load goes wholly to
   the lowest support (conservative for that support — the free-earth-support
   load from below is reported alongside in `free_earth`)
5. Max moment = the larger of the span moments and the embedded-portion moment
   from the free-earth-support body

### Embedment (free earth support about the lowest support)

`embedment.compute_embedment` / `free_earth.solve_about_support`, Caltrans T&S
Manual Sec 8-4.02 (hinge method):
- Free body: the whole wall for one support level; for two or more, the wall
  below the lowest support (moments at the supports above taken as zero).
- Pressure: p(z) above the excavation; below it, layered Rankine active
  pressure (overburden from the ground surface, surcharge included) plus net
  water, against Rankine passive pressure on the excavation side.
- D solves MR = FS·MD (FS = `FOS_passive`, default 1.5; the manual uses 1.3);
  D′ solves MR = MD, and horizontal equilibrium at D′ gives the support load.
- No depth increase by default (`embedment_increase` = 1.0), as in Example 8-1.
- **Pinned to Caltrans Example 8-1** (`tests/test_free_earth.py`): D = 6.09 ft,
  D′ = 4.89 ft, T = 14,254 lb/ft, M = 22,494 ft-lb/ft at the anchor, zero shear
  9.69 ft below it — all within 1 %.

### Cantilever Walls (Caltrans Simplified Method)

`beam_analysis.analyze_cantilever_excavation` / `free_earth.solve_cantilever`,
Caltrans T&S Manual Sec 7-5.02: layered Rankine active and passive pressures,
water on both sides, moments about the rotation point O at depth D0 below the
excavation; D = 1.2·D0 (AASHTO 3.11.5.6 — the increase accounts for rotation
below O and is **not** a factor of safety, so it sits alongside
`FOS_passive`; this differs from `sheet_pile.analyze_cantilever`, which since
v5.1 carries safety on FOS alone). The net horizontal force to O is reported;
a positive value means embedment must increase (step 4). Max moment and shear
come from the same diagram. Ka/Kp in the result are those of the embedment
layer; every depth uses its own layer. A note is added past H = 5.5 m (18 ft,
Caltrans 7-1). **Pinned** to the closed form for uniform dry sand and to
`sheet_pile.analyze_cantilever` (layered, wet, surcharge; D0 within 0.5 %).

### Water on the two sides

Retained side: hydrostatic from `gwt_depth`. Excavation side: from
`gwt_depth_excavation` (default = `gwt_depth`), never above the excavation
base. Below the base the net water pressure is a driving pressure.

### Defect history (2026-09-15, field feedback N2)

Found on a real review session (`module_work/field_feedback/2026-09-15_nairobi-soe-rerun_v5.15.0/FINDINGS.md`):
- `analyze_cantilever_excavation` took γ, φ, c from the **first layer only**
  and put the passive resultant at **H + D/3 about the wall base** (should be
  D/3); max moment was taken at the dredge line. A 6.3 m wall in uniform sand
  got D = 2.5 m (correct 6.9 m before the 1.2 increase); the session's profile,
  topped by a φ = 0 clay, got Ka = Kp = 1.0 and D = 0.60 m.
- `compute_embedment` used one layer, started the active triangle at zero at
  the support (ignoring overburden) and used arms d/3 and D/3.
- The braced top span was treated as simply supported (p·d1²/8 instead of
  p·d1²/2), and surcharge and water were not applied to braced loads.
The tests had only checked signs and monotonicity.

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
and Caltrans Ex 8-1 for a single anchor. With ONE anchor level the tributary
formulas do not apply: the wall is solved by free earth support about the anchor
(`free_earth.solve_about_support`; `Kp` override for a log-spiral passive,
`FOS_embedment` default 1.3), returning the total TH, D, D′ and the wall moment —
D = 6.09 ft, T1 = 14,254 lb/ft, M = 22,494 ft-lb/ft (V-013). Until 2026-09-15 the
single-anchor result stopped at the upper tributary load without saying so (N3).

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
