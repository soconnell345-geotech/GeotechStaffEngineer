# retaining_walls (cantilever wall): a water table behind the wall

BACKLOG GM-292, 2026-10-31, branch `wip/gm292` from master 474e5e3. GM-288
left the backfill drained ("Not modelled: water behind the wall"); this adds
an optional water table. Read from the PDFs in `geotech-references/docs/`,
each page rendered and read: UFC 3-220-20 (16 January 2025) Sec. 4-4.1,
Figs. 4-5, 4-16 and 4-27, Sec. 5-3.2 and Fig. 5-40; FHWA NHI-06-089 (2006)
Vol II Sec. 10.3, Fig. 10-12, Example 10-1, Sec. 10.5.3 and Fig. 10-17;
GEC-6 (FHWA-SA-02-054) Sec. 5.2.3.2; at review, EM 1110-2-2104 (1 November
2023) pp. 18 and 115-116 and EN 1997-1:2004 Sec. 6.5.2.1. Page numbers are
the printed ones.
gamma_w is `geotech_common.water.GAMMA_W`, 9.81 kN/m3.

The wall analysis covers cantilever walls and gravity walls alike ("a
cantilever (or gravity) wall"); the calc package runs the same code.

## The inputs

- `water_table_depth` (geometry): a level water table that depth below the
  top of the backfill at the stem, m, 0 or more. Left out: no water table,
  the backfill drained, as before. h_w = H - d_w is the water's height above
  the underside of the base; at or below the base (h_w = 0) nothing changes.
- `gamma_sat_backfill`: the backfill's unit weight below the water table;
  default `gamma_backfill`. More than gamma_w.
- `base_uplift`: "linear" (default) or "uniform".
- `uplift_in_bearing`: default False.
- The last three are refused without `water_table_depth`.

## The pressure on the wall (as printed)

- **UFC 3-220-20 Sec. 4-4.1 "Water Pressure Effects"** (p. 199): "Water
  applies a triangular pressure distribution equivalent to a soil with a
  unit weight of 62.4 pcf and an earth pressure coefficient of unity". P.
  200: "Instead of the total unit weight (γ), the buoyant or effective unit
  weight (γb) is used in the earth pressure equations for soils below the
  water table. The second effect is that an additional load, supplied by
  the water pressure, is applied to the wall. Figure 4-16(a) shows combined
  influence..." Fig. 4-16(a) (p. 202) prints σ_A = K_A[γh + γb(z − h)] and
  σ_W = γw(Z − h), and a chart of (P_A,wet + P_W)/P_A,dry against H_W/H for
  φ' 30°, γ 125 pcf, γb 62.5 pcf.
- **UFC Figs. 4-5 and 4-6** (p. 183): moist unit weight above the water
  table, buoyant below, the water pressure u = γw z2 added; Note A, no
  negative pressure.
- **NHI-06-089 Eq. 10-10** (p. 10-14): "For a c'-φ' soil with a groundwater
  table, the effective lateral earth pressure ... pa′ = Ka (γz - u) -
  2c′√Ka". P. 10-15: "In all cases where water is present in the soil,
  full hydrostatic pressure is added to the lateral earth pressure computed
  by Equations 10-8 to 10-11 to obtain the total lateral pressure".
- **NHI Fig. 10-12** (p. 10-27): pa = Ka[γ hw + γ' zw], pw = γw zw, "the
  lateral water pressure coefficient = 1.0. The lateral pressure
  computations should consider the greatest unbalanced water head
  anticipated to act on the wall, since this generally results in the
  largest total lateral load."
- **Module now.** Exactly these: the active pressure Ka(q + γ min(z, z_w) +
  γb max(z − z_w, 0)) − 2c√Ka, held at no less than zero (GM-288's tension
  zone), integrated exactly over the pressure plane through the heel
  (`earth_pressure.active_force_components_water`; a 20,000-slice sum
  agrees to 1e-6 in the tests), plus Pw = γw h_w² / 2 at h_w / 3,
  horizontal. Pa keeps its inclination (Coulomb's δ, sloped Rankine's β);
  Pw drives sliding and overturning with P_h.

## The drained backfill (the default, as printed)

- **NHI Sec. 10.5.3** (p. 10-41): "If adequate drainage measures are
  provided, the hydrostatic pressure due to groundwater behind the wall
  generally need not be considered. However, hydrostatic pressure must be
  considered for portions of the wall below the level of the weep holes
  unless a deeper drainage system is provided behind the base of the wall.
  The wall must be designed for the full hydrostatic pressure when it is
  necessary to maintain the groundwater level behind the wall."
- **Module now.** Left out, no water table, and `defaults_used` says so with
  this page. Given, the result's choices give the drained backfill's
  FOS_sliding and FOS_overturning beside the water table's.

## The unit weight below the water table

- **Print.** UFC Fig. 4-16(a) (p. 202) takes γ = 125 pcf above the water
  table and γb = 62.5 pcf below it: the same soil, its buoyant weight about
  γ − γw. NHI Example 10-1 (p. 10-28) gives γ 115 pcf above and γsat 120
  pcf below.
- **Module now.** `gamma_sat_backfill`, default `gamma_backfill` (the UFC
  figure's reading), stated in `defaults_used` and the choices. The soil on
  the heel and over a battered back below the water table adds its excess
  weight (γsat − γ) to the resisting weights: the total weight, with the
  uplift counted separately. EM 1110-2-2104 (2023) p. 18: vertical earth
  "is expected weight of moist or buoyant soil on a structure. Water in the
  soil is included under hydrostatic load, Hs", and Hs "includes water
  pressure above and below the ground line. It also includes water forces
  above or in a structure, including weight of water within soil on a
  structure, and uplift."
- **The default's direction (review).** Against γsat = γ + 2 the default
  errs on the safe side for sliding, overturning and the eccentricity where
  the heel carries soil below the water table (hand-check wall: FOS_sliding
  0.710 against 0.725, FOS_overturning 1.708 against 1.770, e 0.657 against
  0.617), and on the unsafe side for the bearing pressures (q_toe 157.2
  against 159.4 kPa, 1.4 % low) and for a wall with no heel (B 1.0 m:
  FOS_sliding 0.166 against 0.162). The choice says so.

## The uplift on the base (as printed; its moment the module's assumption)

- **Print.** UFC p. 203: "Water pressures can add an uplift force on the
  base of retaining walls which can be detrimental to the wall stability.
  These are discussed in ... Section 4-5"; Sec. 4-5 does not take it up,
  and UFC Fig. 4-27 (p. 220) and NHI Fig. 10-17 (p. 10-39) draw none. UFC
  Fig. 5-40 (p. 346, Sec. 5-5.2.4, a structure below the water table): "U =
  Total Uplift Force = γw·h·L ... The uplift force should be calculated
  based on the maximum elevation of the water table." Found at review:
  EM 1110-2-2104 (2023) para. E-12, spillway stilling basin walls, Load
  Cases 3A and 3B (p. 115): "Uplift across base varying uniformly from
  tailwater at heel to a value midway between tailwater and jump profile
  at toe", a straight line between the heads at the heel and the toe; Load
  Case 3C (p. 116), the same water both sides: "Uplift equal to tailwater".
  UFS 3-220-10 Sec. 6-2.6.3.2 (p. 294) integrates a flow net's pore
  pressures along the base for the general case.
- **Module now.** The distribution follows EM 1110-2-2104 para. E-12, with
  the other printed case in the choices:
  - "linear" (default): γw h_w at the heel falling to none at the toe, no
    water in front of the wall above the underside of the base; U = γw h_w
    B / 2 at 2B/3 from the toe.
  - "uniform": γw h_w under the whole base, the same head at both (EM
    1110-2-2104 p. 116; UFC Fig. 5-40); U = γw h_w B at B/2.
  The uplift comes off V in sliding (the friction on max(V − U, 0)); its
  moment about the toe is added to the overturning moment (the module's
  assumption: UFC Fig. 4-27's F_OT = W a / (P_H b − P_V f) has no uplift;
  the choice gives the FOS with it taken off the stabilizing moment). The
  resultant and the eccentricity are the same either way, and the
  eccentricity check counts it.

## The uplift in the bearing pressures (as printed)

- **UFC Sec. 5-3.2 "Groundwater Correction"** (pp. 291-292): "Uplift is
  present on the base of the foundation when the groundwater table is above
  the bearing elevation. Its effect on the bearing pressure can be
  conservatively ignored unless sliding or uplift is a concern."
- **GEC-6 Sec. 5.2.3.2 "Location of the Ground Water Table"** (p. 56): the
  effective stresses below the water table control the bearing resistance.
- **Module now.** `uplift_in_bearing` False by default (q_toe, q_heel, q_eq
  on the total weights); True counts it, as EN 1997-1 Sec. 6.5.2.1(3)P
  (p. 63) has it: "Water pressures not caused by the foundation load shall
  be included as actions". The tool text tells the agent to give
  q_allowable for the water table at that level (GEC-6 p. 56). The choice
  gives the other reading's q values.
- **The default's direction (review).** Leaving the uplift out is on the
  safe side while the resultant is in the middle third: the linear uplift
  counted takes nothing off q_toe and lowers q_eq (B 3.6 m, d_w 3 m: q_toe
  123.15 both ways, q_eq 100.75 left out against 96.11 counted). Past B/6
  counting it raises them (hand-check wall, e 0.617: q_toe 161.6 against
  159.4), but that wall fails the eccentricity check already.

## Printed examples reproduced

**NHI Example 10-1** (pp. 10-28, 10-29): an 18 ft smooth wall, β 10°, the
water table 6 ft down, γ 115 pcf above, γsat 120 pcf below, φ 30°, Coulomb
Ka 0.374 at δ 0. Run as a wall with no heel (the pressure plane is the
face), in kN/m and converted back:

The first three rows are printed; the last three are worked by hand from
the printed pressures (the diagrams' areas and centroid), not printed.

| | printed, or from the printed pressures | module |
|---|---|---|
| p'a at 6 ft (psf) | 258.1 | 257.8 |
| p'a at 18 ft (psf) | 516.6 | 515.9 |
| u at 18 ft (psf) | 748.8 | 749.4 |
| effective earth thrust, the diagram's area (lb/ft) | 5422.5 | 5416.0 |
| its height above the base (ft) | 6.570 | 6.571 |
| water thrust (lb/ft) | 4492.8 | 4496.3 |

The differences are the example's rounded Ka (0.374 against 0.3737) and
γw (62.4 pcf against 9.81 kN/m3 = 62.45 pcf).

**UFC Fig. 4-16(a)'s chart** (p. 202), φ' 30°, γ 125 pcf: the module gives
the ratio 1.999 at H_W/H = 1 (the curve reaches 2.0). Below that the curve
as drawn reads up to about 0.04 above the figure's own equations: at
H_W/H 0.6 the curve reads 1.40, the equations 1.36 (module 1.36); at 0.4
about 1.18 against 1.16; at 0.8 about 1.68 against 1.64. The module follows the printed
equations; the tests pin those, worked by hand in pcf.

## Hand check, a made-up wall

H 5 m, B 3.0, toe 0.5, stem 0.3 at the top to 0.5 at the base (back
battered), base 0.5; γ 18, γsat 20, φ 30 (Rankine Ka 1/3, δb = 2/3·30 =
20°), concrete 23.6; water table 2.0 m down, so h_w = 3.0 m, z_w = 2.0 m,
γb = 20 − 9.81 = 10.19 kN/m3.

- σ'v 36 kPa at the water table, 36 + 10.19 × 3 = 66.57 at the base.
  Pa = (18 × 2²/2 + 36 × 3 + 10.19 × 3²/2) / 3 = (36 + 108 + 45.855) / 3
  = **63.285** kN/m; its moment about the base (12 × 3.667 + 36 × 1.5 +
  15.285 × 1.0) = 113.285, z_Pa = 1.790 m.
- Pw = 9.81 × 9 / 2 = **44.145** kN/m at 1.0 m.
- Weights (kN/m, x from the toe): base 35.4 at 1.5; stem 31.86 at 0.65 and
  10.62 at 0.867; soil over the battered back 8.1 at 0.933; soil on the
  heel 162.0 at 2.0; below the water table the heel soil's excess 2 × 2.0 ×
  2.5 = 10.0 at 2.0 and the battered back's 2 × 0.5 × 0.111 × 2.5 = 0.278
  at 0.963. Sum 258.258; moments 434.840.
- Linear uplift U = 9.81 × 3 × 3 / 2 = **44.145** at 2.0 m, moment 88.29.
- Sliding: V = 258.258 − 44.145 = 214.113; R = 214.113 tan 20° = 77.931;
  FOS = 77.931 / (63.285 + 44.145) = **0.725** (module 0.7254). Drained:
  1.20.
- Overturning: M_ot = 113.285 + 44.145 + 88.29 = 245.72; FOS = 434.840 /
  245.72 = **1.770** (module 1.7697). Drained: 3.32. Taken off the
  stabilizing moment instead: (434.840 − 88.29) / 157.43 = 2.20.
- Eccentricity: e = 1.5 − (434.840 − 245.72) / 214.113 = **0.617** m > B/6
  = 0.5, fails (module 0.6167).
- Bearing, uplift left out: V 258.258, M_over 157.43, e = 0.426; q =
  86.086 (1 ± 0.8517) = **159.40 / 12.77** kPa; q_eq = 258.258 / 2.148 =
  **120.21**, FOS 200 / 120.21 = 1.66. Uplift counted: e 0.617, the
  triangle q_toe = 2 × 214.113 / (3 × 0.883) = **161.6**, q_eq 121.2.
- Uniform uplift: U = 88.29 at 1.5; FOS_sliding = 169.97 × 0.364 / 107.43
  = **0.576**.
- A separate smoke wall (H 5, B 3.2, γ 19, φ 32, d_w 2) showed the linear
  uplift counted in bearing takes exactly γw h_w = 29.4 kPa off q_heel and
  nothing off q_toe while the pressure stays a trapezoid, as it must (the
  uplift is itself a linear pressure, zero at the toe).

The numbers are pinned in `retaining_walls/tests/test_gm292_water_table.py`
and `funhouse_agent/tests/test_gm292_wall_water_adapter.py`.

## A wall with no water table is unchanged

- **Recorded calls.** A pytest plugin recorded every
  `analyze_cantilever_wall` call (inputs, `to_dict()`, summary and the
  terms) in ten wall test files, on master and on this branch: 122 calls
  each, 291 tests passed both times. Every call's output, summary and terms
  are identical; the only differences are new keys (the new inputs at their
  defaults, and the water terms Pw = 0, U = 0, h_w = 0 in the term dicts).
- **Tool and package.** Five made-up walls (level, Coulomb with slope and
  surcharges, cohesive with a key and passive, front-battered on rock,
  adhesion) through `cantilever_wall` and `retaining_wall_package`, master
  against the branch: every number and both figures identical. One change
  on every cantilever run: `defaults_used` now lists `water_table_depth`,
  "no water table: the backfill drained ... (FHWA NHI-06-089 Sec. 10.5.3,
  p. 10-41 ...)", and the package's table of assumed inputs prints the same
  row. This is wanted: the drained backfill is now a stated default.

## Tool text

- `cantilever_wall` brief: "...the backfill is taken as drained unless
  water_table_depth puts a water table behind the wall."
- New parameters `water_table_depth`, `gamma_sat_backfill`, `base_uplift`,
  `uplift_in_bearing`, each with its page; `applies_when` (the last three
  only with `water_table_depth`); `when_absent` for the first two.
- Returns: `water_table_depth_m`, `water_height_m`, `water_thrust_kN_per_m`,
  `uplift_kN_per_m` (with a water table), and the same in `sliding_basis`.
- The calc package prints "Active Thrust with a Water Table (effective
  stress)", "Water Pressure Thrust" and "Uplift on the Base", the input
  rows d_w, γ_sat, the uplift and whether bearing counts it, sliding FOS =
  R / (P_h + P_w) with V net of U, and M_ot with P_w z_Pw + U x_U. The
  figure draws the water table over the heel, the effective earth pressure,
  the water triangle and P_w.
- Method probes `retaining_walls|cantilever_wall#water` and
  `calc_package|retaining_wall_package#water`, so the defaults guard checks
  the new defaults (it found that a `gamma_sat_backfill` given equal to the
  default lost the default's basis; fixed).

## Checks run

Each in the foreground, by exit code, at least 3 GB free:

- `retaining_walls/tests` + V-061 + `calc_package/tests/test_steps_from_a_minimal_analysis.py`: 493 passed (before the last test edits; the new file again after: 27 passed).
- `funhouse_agent/tests/test_defaults_used.py`, `test_gm179_unused_inputs.py`, `test_top_level_params.py` `-k "retaining or wall or cantilever"`: 119 passed, one failed (fixed, above); the water probes then 8 passed.
- `test_choices_stated_defaults.py`, `test_adapter_ergonomics.py`, `test_gm288_wall_adapter.py`, `test_gm212_wall_sliding.py`, `test_gm292_wall_water_adapter.py`, `test_gm292_water_table.py`: 224 passed.
- `test_calc_package_tool_parity.py`, `..._figure_conventions.py`, `..._figure_layout.py`, `..._assumed_inputs.py`, `test_calc_package_adapter.py`, `test_render_tables.py` `-k "retaining or wall or cantilever"`: 22 passed.
- `test_citations.py`: 5 passed (it caught "NHI Fig. 10-12" read as a UFC figure; now "FHWA NHI-06-089 Fig. 10-12").
- `test_public_text` on every changed file: clean.

## Owner questions

1. The uplift's moment: added to the overturning moment (default, the
   smaller FOS: 1.77) or taken off the stabilizing moment (2.20)? No page
   held prints it; the resultant and the eccentricity are the same either
   way.

Settled by the print at review, so not asked: the uplift's distribution
(EM 1110-2-2104 para. E-12, p. 115), leaving it out of the bearing
pressures (UFC p. 292, on the safe side while e <= B/6) and γsat = γ when
none is given (UFC Fig. 4-16(a), γb = γ − γw; its direction stated in the
choice).

## Noted, not fixed

- Water in front of the wall (tailwater) and seepage under it (UFC Fig.
  4-16(c), flow-net corrections) are not modelled; the passive resistance
  in front stays on the moist unit weight.
- Water in a tension crack (NHI p. 10-15, "Unless positive drainage
  measures are provided, water infiltration into the tension crack may
  result in hydrostatic pressure") is not added.
- The water table is level; NHI Example 10-1 draws it parallel to a sloping
  backfill, which at the wall face is the same.
- The MSE wall's external checks have no water table.
- A wall the uplift floats (net V <= 0, e.g. uniform uplift under a light
  wall with the water at the top) reports e = 0 and passes the eccentricity
  check; sliding and overturning still fail, so the wall still fails.

## Review (2026-10-31)

Every page above re-read from the PDFs. The four owner questions checked
against the held manuals: three are settled by a printed page (above), the
uplift's moment is not. Fixed at review, wording only (no number changes):
the uplift's basis (EM 1110-2-2104 para. E-12 in place of "no reference
held prints"), the bearing default's direction and EN 1997-1 Sec.
6.5.2.1(3)P for the counted option, the γsat default's direction, the
saturated weight's basis (EM 1110-2-2104 p. 18; UFC Fig. 5-40's
buoyant-weight sentence is about soil under an anchored structure and did
not support it), and Example 10-1's worked thrusts labelled. A wall with no
water table replayed independently: 205 recorded `analyze_cantilever_wall`
calls and 23 recorded `cantilever_wall` / `retaining_wall_package` calls,
plus the method probes, on 474e5e3 and on c25b5f2: identical but for the
new keys, the `defaults_used` line and the package's input row; the figures
byte for byte. Trial merge on master 001526b clean (1,406 passed), and with
GM-297 on top clean (1,470 passed).
