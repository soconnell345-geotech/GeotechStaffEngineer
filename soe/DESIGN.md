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

The envelope is the sand one where the layers marked `sand` are the
thicker over the excavation depth, else a clay one by the average N.
`SOEWallLayer.soil_type` defaults to `sand` in the library, so the agent
tools (`braced_excavation`, `apparent_pressure`) refuse a layer above the
excavation base that gives none (GM-182, 2026-10-19): the default picked
the method unstated. The cantilever analysis does not read it.

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

### No moment balance within 6H

Both solvers search 0 < D ≤ 6H for the first depth where the resisting moment
catches up. When none does — e.g. a φ = 0 clay where the net pressure below
the excavation, 4c − γH − q, is negative, so the passive side never catches up
at any depth — the result says so instead of reporting a depth:
- `solve_cantilever`: D0, D, the net force at O, moment and shear are `None`,
  `converged=False`, and the first note reads "No moment balance about the
  point of rotation O within 6H = … m of embedment: the wall cannot stand as a
  cantilever in this soil … Consider supports … or a deeper or stiffer
  stratum."
- `solve_about_support`: `D` is `None` when MR = FS·MD has no root; `D_prime`,
  the support load, moments and shear are `None` when MR = MD has none (they all
  come from the FS = 1 body). The note says which of the two failed.
- Callers carry the `None` through: `compute_embedment` returns `None`;
  `analyze_cantilever_excavation` and `analyze_braced_excavation` return
  `None` for the embedment, wall length and (with no FS = 1 body) the moment,
  shear and Sx, with `converged=False` in `to_dict()` and the warning as the
  first note; the braced tributary support loads stay (they do not depend on
  the embedment) and a note gives the span moment above the excavation. The
  single-anchor `fhwa_apparent_pressure_anchored_wall` returns `None` for the
  anchor load, design load, D, D′ and moment, and keeps `TH_upper` (the
  tributary load above the anchor).

Until 2026-10-14 (BACKLOG GM-20) the root finder handed back the 6H search
limit and the solvers reported D0 = 6H, D = 1.2 × 6H and a moment, shear and
support load computed at that arbitrary depth, with only a note.

### Water on the two sides

Retained side: hydrostatic from `gwt_depth`. Excavation side: from
`gwt_depth_excavation` (default = `gwt_depth`), never above the excavation
base. Below the base the net water pressure is a driving pressure.

### Defect history (2026-09-15, field feedback N2)

Found on a real review session (the 2026-09-15 SOE re-run; its `FINDINGS.md` is under `module_work/field_feedback/`):
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
`Kp` and `FOS_embedment` are read on the single-anchor route only: its result
carries them as `choices`, and with two or more anchors the tool refuses them
and `defaults_used` leaves them out (GM-182). The multi-anchor route works out
no embedment; `braced_excavation` does, about the lowest support.

## Log-spiral passive coefficient (v5.3)
`earth_pressure.caquot_kerisel_Kp(phi, delta)` — the Caquot-Kerisel (1948)
log-spiral passive coefficient Kp' = R·Kp0 (base Kp0 at δ=φ; reduction R for
δ/φ from Caltrans Matrix 4-1 / NAVFAC DM-7.2). Unlike Coulomb it does not
over-predict Kp at high δ/φ. φ=30, δ/φ=0.5 → 4.70 (V-013). Mirrored in
`sheet_pile.earth_pressure` (a byte-identical copy of the table) and selectable
there as `pressure_method="log_spiral"`.

**Source basis (Caquot-Kerisel Kp0 base values + R reduction grid), cross-read
2026-10-19 (GM-21).** Kp0 (δ = φ, vertical wall, level backfill: the β/φ = 0
curve) is read off the chart as printed in the Caltrans T&S Manual (July
2025) Figure 4-20, p. 4-30 (the same chart: FHWA NHI-06-089 Figure 10-10,
p. 10-22), on a 300 dpi render at φ ± 0.2°. R is the table printed on the
same figure (φ 10–45°, −δ/φ 0–0.7), identical in NHI-06-089; Matrix 4-1
(p. 4-29) quotes its φ 30/32/35 rows.

| φ (deg) | Kp0 before | Kp0 now | chart read | Kp_Rankine / R(0) |
|---|---|---|---|---|
| 10 | 2.20 (clamped) | 1.65 | 1.65 | 1.64 |
| 20 | 3.10 | 3.00 | 3.03 | 3.01 |
| 25 | 4.40 | 4.30 | 4.31 | 4.29 |
| 30 | 6.30 | 6.30 (kept: Example 8-1, p. 8-12, reads 6.3) | 6.40 | 6.42 |
| 35 | 9.20 | 10.2 | 10.17 | 10.19 |
| 40 | 14.0 | 17.7 | 17.77 | 17.55 |
| 45 | 14.0 (clamped) | 33.5 | ~33.6 | 33.5 |

Before, R was tabulated only for φ 30–35° and δ/φ 0.40–0.50 and clamped
outside that (φ = 40° used φ = 35°'s R). Now the full printed grid is used,
linear from δ/φ 0.7 to R = 1 at δ = φ; below δ/φ = 0.1 R still runs to the
Rankine anchor so δ = 0 returns exactly the Rankine Kp (the printed R(0)
times the chart Kp0 is Rankine within 2 %). DM 7.2 Figure 4-13 (p. 193) is
Kerisel & Absi (1990) at δ/φ = 0.66: within 5 % of this table up to φ 35°,
and 8–12 % lower at 40–45° (the 1948 chart runs higher there). Pinned in
`soe/tests/test_caquot_kerisel_printed.py`.

**Sources side by side (GM-183, 2026-10-19).** The tables moved to
`geotech_common/log_spiral.py`, which both `caquot_kerisel_Kp` copies call
(one table, no copy to drift), and gained what the references print beside
the 1948 chart: (1) the chart's sloping-ground curves, β/φ −0.9 to +1 at φ
10–45° (`caquot_kerisel_Kp(..., beta_deg=)`), pinned to the manual's p. 4-29
example (φ 32, β 27, δ 14: Kp ≈ 19, R 0.679, Kp′ = 13; the module gives
19.3 and 13.05); (2) UFC 3-220-20 Fig. 4-13 (Kerisel & Absi 1990, δ/φ′ =
0.66 only, β/φ′ −0.6 to +1, Kp ≤ 20); (3) UFC 3-220-20 Eq. 4-12 (level
ground, any δ). The default stays the 1948 chart. Both figures are printed
only as charts: every value was read by tracing each curve on the rendered
page, calibrated on the printed grid lines, and drawn back over the figure
to check. Kp0 between nodes is now interpolated with log Kp linear in φ
(the chart's log scale), on 2.5° nodes; linear Kp between the 5° nodes had
read up to about 5 % above the chart between nodes (φ 32: 7.86 → 7.57;
37.5: 13.95 → 13.3; 42.5: 25.6 → 24.3); the 5° nodes are unchanged.

## Calc package (GM-41)
`soe/calc_steps.py` renders the three wall tools through
`calc_package(soe_package, wall_type=cantilever|braced|fhwa_anchored)`. The
package runs the tool's own `run_<tool>` and prints the result: for that the
cantilever result carries `layers_used` (each layer's Ka, Kp), the
`pressure_points` at the break depths and the `diagram` (net pressure, shear,
moment to O); the braced result carries the `apparent_pressure` envelope, the
`spans` (tributary split) and both diagrams. These fields are not in
`to_dict()`, so the tools' outputs are unchanged.

## Ground anchor design (built — provenance)
`anchor_design.py` is fully implemented (grouted-anchor bond capacity, tendon
selection). Source: GEC-4, *Ground Anchors and Anchored Systems*
(FHWA-IF-99-015), Sec. 5.3; tendon strand/bar data follow **PTI DC35.1** and
**ASTM A722** (bar) / **A416** (strand); the bars and the 15 mm strand are
GEC-4 Table 9 (p. 77) and Table 10 (p. 78) as printed, pinned by
`test_tendon_limits.py` (GM-182/185 review, 2026-10-19: the 32 mm bar had
819 mm2 / 848 kN against Table 9's 806 / 835, and the 1-3/4 in bar 1548 /
1601 against 1716 / 1779; a 32 mm bar now holds a design load of 501 kN at
0.60 SMTS, was 509).

**Benchmarked against GEC-4's two design examples** (`test_anchor_gec4.py`):
App. A design example 1 (pp. A-7 to A-9, Figure A-3: H 10 m, phi' 33,
anchors at 15 deg from 2.5 m and 6.25 m deep, DL 445 kN, sand and silt
medium dense at 100 kN/m, FS 2.0) gives a bond length of 8.9 m and unbonded
lengths of 5.7 m and 3.9 m, "minimum unbonded length for strand anchor is
4.5 m" — the module gives 8.90 m, 5.74 m and 3.90 m (bar) / 4.50 m
(strand); design example 2 (pp. A-27, A-28: mudstone taken as hard shales
and slates, 360 kN/m, FS 3.0) gives 7.5 m for 900 kN and 5.3 m for 636 kN —
the module gives both.

### Bond: the ultimate load transfer (GM-191, 2026-10-19)
`soil_type` names a printed row, and the row's value is used as the
quantity it is printed in: the ultimate LOAD TRANSFER per metre of bond
length (kN/m). Bond length = FOS_bond x DL / load transfer, at least the
minimum bond length. The hole diameter does not enter it.

- GEC-4 Table 6 (p. 71): "Presumptive ultimate values of load transfer for
  preliminary design of small diameter straight shaft gravity-grouted ground
  anchors in soil"; "The maximum allowable anchor design load in soil may be
  determined by multiplying the bond length by the ultimate transfer load
  and dividing by a factor of safety of 2.0."
- GEC-4 Table 8 (p. 74): "Presumptive ultimate values of load transfer for
  preliminary design of ground anchors in rock"; "The maximum allowable
  anchor design load in competent rock may be determined by multiplying the
  bond length by the ultimate transfer load and dividing by a factor of
  safety of 3.0."
- GEC-4 Table 7 (p. 73, after PTI 1996) prints average ultimate bond
  STRESSES (MPa). One given as `bond_stress_kPa` is multiplied by the
  perimeter of `drill_diameter_mm` (default 150 mm): "capacity estimates
  should be based on the as-drilled hole diameter" (p. 72). A load transfer
  found by preproduction tests (p. 75) goes in `load_transfer_kN_per_m`.

The rows (kN/m): Table 6 — `sand_gravel_loose` 145, `sand_gravel_medium`
220, `sand_gravel_dense` 290, `sand_loose` 100, `sand_medium` 145,
`sand_dense` 190, `sand_silt_loose` 70, `sand_silt_medium` 100,
`sand_silt_dense` 130, `silt_clay_stiff` 30, `silt_clay_hard` 60 ("silt-clay
mixture with low plasticity or fine micaceous sand or silt mixtures");
Table 8 — `granite_basalt` 730, `dolomitic_limestone` 580, `soft_limestone`
440, `sandstone` 440, `slates_hard_shales` 360, `soft_shales` 150.

Until GM-191 the table was nine "bond stresses" (kPa) multiplied by the
perimeter of a 150 mm hole and cited to no table that printed them. The
trace, row by row:

| Old row (kPa) | What it was | Now |
|---|---|---|
| `sand_loose` 70 | Table 6's sand-and-silt loose, kN/m | Table 6 sand loose, 100 kN/m |
| `sand_medium` 145 | Table 6's sand medium dense, kN/m | the same 145, as kN/m |
| `sand_dense` 250 | no printed row | Table 6 sand dense, 190 kN/m |
| `gravel` 310 | no printed row | removed; `sand_gravel_*` 145/220/290 |
| `clay_stiff` 50 | the middle of Table 7's gravity-grouted cohesive range (30-70 kPa) | removed; `silt_clay_stiff` 30 kN/m, or `bond_stress_kPa` |
| `clay_hard` 100 | Table 6's sand-and-silt row (70-130), kN/m | removed; `silt_clay_hard` 60 kN/m |
| `rock_soft` 700 | no printed row | removed; Table 8 rows |
| `rock_medium` 1050 | no printed row | removed; Table 8 rows |
| `rock_hard` 1750 | the middle of Table 7's dolomitic limestone range, labelled granite | removed; `granite_basalt` 730 kN/m |

A removed name raises a ValueError that says what the row was and which
printed rows to use. A kN/m value read as kPa over a 150 mm hole (perimeter
0.471 m) gave 47 % of the load transfer and a bond length 2.1 times Table
6's: conservative for holes under 318 mm (where pi x d = 1 m), unconservative
above.

**Release note (GM-191):** the anchor's bond length now uses GEC-4's
printed load transfer per metre (Tables 6 and 8) and no longer depends on
the hole diameter. At DL 445 kN on strand, bond lengths before -> after:
`sand_loose` 26.98 -> 8.90 m; `sand_medium` 13.03 -> 6.14 m; `sand_dense`
7.55 -> 4.68 m; `gravel` 6.09 -> `sand_gravel_dense` 4.50 m (the minimum);
`clay_stiff` 37.77 -> `silt_clay_stiff` 29.67 m; `clay_hard` 18.89 ->
`silt_clay_hard` 14.83 m; `rock_soft` 3.00 -> `soft_shales` 8.90 m (longer:
the old row was unconservative); `rock_medium` 3.00 -> `soft_limestone` 4.50
m; `rock_hard` 3.00 -> `granite_basalt` 4.50 m (both now at the 4.5 m
minimum for strand in rock). The soil rows shorten (the old reading was
conservative for a 150 mm hole); the old rock rows lengthen. Six names are
gone (`gravel`, `clay_stiff`, `clay_hard`, `rock_soft`, `rock_medium`,
`rock_hard`) and say where they went.

### Factor of safety and minimum lengths (GM-192, 2026-10-19)
- **FOS_bond** defaults by ground: 2.0 for the Table 6 soil rows (p. 71) and
  3.0 for the Table 8 rock rows (competent rock, "to account for
  uncertainties associated with potential discontinuities in the rock
  mass", p. 74). GEC-4 also gives 2.0 for "weak rocks such as clay shales",
  intermediate geomaterials of unconfined compressive strength 0.5 to 5.0
  MPa (p. 74); give `FOS_bond` 2.0 for those. The result's `choices` states
  which was used, given or default, with the other as the alternative.
- **Distance behind the critical potential failure surface:** the greater of
  1.5 m and H/5. GEC-4 App. E Part 3 C (p. E-15): "The free stressing length
  (unbonded length) shall extend at least 1.5 m or 20 percent of the height
  of the wall, whichever is greater, behind the critical failure surface";
  Sec. 5.3.2 (p. 65) and 5.3.4 (p. 68): "a minimum distance of H/5 or 1.5
  m". The surface rises from the corner of the excavation at 45 + phi/2
  (p. 65, walls in cohesionless soils); the distance is measured normal to
  it and the unbonded length runs along the anchor to that line, as Figure
  A-3 draws it.
- **Minimum unbonded length:** GEC-4 Sec. 5.3.4 (p. 67): "The minimum
  unbonded length for rock and soil ground anchors is 4.5 m for strand
  tendons and 3 m for bar tendons."
- **Minimum bond length:** GEC-4 App. E Part 3 B (p. E-15): "The minimum
  bond length shall be 4.5 m for strand tendons in rock and 3 m for bar
  tendons in rock. The minimum bond length shall be 4.5 m for strand and bar
  tendons in soil." A note is added when a soil bond length passes 12 m
  (p. 71) or a rock one 10 m (p. 74).

**Release note (GM-192):** rock rows default to FOS_bond 3.0 (was 2.0); the
minimum bond length is 4.5 m in soil and for strand in rock (was 3 m
everywhere); a bar's minimum unbonded length is 3 m (was 4.5 m); the
unbonded length extends the greater of 1.5 m and H/5 behind the failure
surface, measured normal to it (was 1.5 m measured along the level of the
anchor head). At phi 33 and 15 deg: H 10 m, anchor 2.5 m deep 5.72 ->
5.74 m; H 12 m, 3 m deep 6.56 -> 6.88 m; H 10 m, 6.25 m deep, bar 4.50 ->
3.90 m (strand stays 4.50 m).

### The ground of a given load transfer (GM-199, 2026-10-20)
A given `bond_stress_kPa` or `load_transfer_kN_per_m` replaces the row's
load transfer, but the ground (soil or rock) still sets FOS_bond's default
and the minimum bond length, and the value does not say it: GEC-4 Table 7
(p. 73) prints bond stresses of 0.15 to 3.1 MPa in rock and 0.03 to 1.38
MPa in soil, and Tables 6 and 8 overlap at 150 to 290 kN/m. So beside a
given one `soil_type` is not defaulted: the call names a row, or the ground
alone, `soil_type="soil"` or `"rock"` (which have no load transfer of their
own and are refused without one). A given one with no `soil_type` is
refused with the question. `ground` on the result says which was used.

**Release note (GM-199):** a call that gives `bond_stress_kPa` or
`load_transfer_kN_per_m` must now also give `soil_type` (a row, or "soil" or
"rock"). Until now it took the default `sand_medium`'s soil: a 1,000 kPa
rock bond stress over a 150 mm hole (471 kN/m) at DL 1,000 kN got FS 2.0 and
4.50 m (the soil minimum; 2 x 1,000 / 471 = 4.24 m); with `soil_type="rock"`
it gets FS 3.0 and 6.37 m.

**GEC-4 design example 1 (App. A, pp. A-7 to A-9, Figure A-3; worked
example WE-SOE-2):** DL 445 kN, `sand_silt_medium` 100 kN/m, FS 2.0: bond
8.90 m (printed 8.9 m); unbonded 5.74 m at 2.5 m deep and 3.90 m at 6.25 m
deep for bar anchors (printed 5.7 and 3.9 m), 4.50 m for strand at 6.25 m
(the printed footnote: "Minimum unbonded length for strand anchor is
4.5 m"). The example's limit line is 2 m beyond the critical failure
surface; the module's is the greater of 1.5 m and H/5, also 2 m at H = 10 m.

Tendon limits (GM-160, 2026-10-19): GEC-4 Sec. 5.3.8 (p. 77) - design load
<= 0.60 SMTS (`max_load_pct`), lock-off <= 0.70 SMTS, maximum test load <=
0.80 SMTS; `tendon.governed_by` names the limit that sized it. Bars were
held to the design-load limit only until then.

Test load (GM-185, 2026-10-19): `test_load_factor` x DL, the maximum load
of the proof and the performance test, default 1.33. GEC-4: 120 to 150
percent, 133 commonly used for permanent and 120 for temporary
applications, 150 for potentially creeping soils or with no independent
reference for the dial gauge (Sec. 7.4.2.2, p. 143); the performance and
proof test schedules both end at 1.33 DL (Tables 21 and 22, pp. 144, 147),
and Table 22 adds that the test load for a temporary support of excavation
"may be set at 1.20 DL"; permanent anchors "typically loaded to 133
percent" (p. 76). The default is 1.33 although this module mostly serves
temporary excavation support: 1.33 is the schedules' test load and 1.20 a
permission, and the higher test load is the stricter check. At
`max_load_pct` 0.60 the two size the same tendon. At 1.33 the 0.80
test-load limit allows 0.60 SMTS, the design-load limit (GEC-4's own
example, p. 77); at 1.50 it allows 0.53 and governs.
Until GM-185 the 1.50 DL performance test was fixed and sized every tendon
(and `performance_test_kN` was 1.50 DL beside a 1.33 DL proof test): DL 300
kN on 15 mm strand was 3 strands, now 2; a 32 mm bar at DL 500 kN, which
GM-160 made a 36 mm bar, is a 32 mm bar again at 1.33. The result's
`choices` list the method, the test load, the SMTS limit, the load transfer
(and the hole diameter when a bond stress is given), FOS_bond, the distance
behind the failure surface and the minimum unbonded and bond lengths, each
given, default or the module's, with its basis and alternatives.

## Future Work

- Full ACI 318 concrete design for secant/diaphragm walls
- Full AISC connection design for wales, struts, bracing
- Beam-spring (FD) SSI analysis (adapt from lateral_pile/solver.py)
- Lagging design between soldier piles
- Raker and corner bracing (3D configurations)
- Seismic apparent pressure (M-O increments)
- Stability checks: basal heave, bottom blowout, piping (Phase 2)
