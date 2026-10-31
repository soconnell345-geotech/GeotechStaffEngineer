# lateral_pile against GEC 9 and COM624P, as printed

BACKLOG GM-287, 2026-10-31. Read from the PDFs: FHWA GEC 9
(FHWA-HIF-18-031, 2018; `geotech-references/docs/GEC 9.pdf`) chapter 6,
chapter 11, Appendix A and Appendix B; the COM624P manual (FHWA-SA-91-048,
1993; the owner's scan), Part II (the differential equation) and Part III
(the p-y criteria); API RP 2A-WSD, 21st ed. with Supplement 3 (2007),
Secs. 6.8.6-6.8.7 (owner's library). Page numbers are the printed ones
(GEC 9: pdf index = page + 3; COM624P: pdf page = page + 22 from p. 299;
API: p. 70 at pdf index 81).

The module offers seven p-y criteria on the agent tool (Matlock soft clay,
Jeanjean soft clay, stiff clay below and above the water table, Reese sand,
API sand, weak rock) and one more in the library (liquefied sand), a
finite-difference beam-column solver with axial load, free, fixed and
partial heads and a stickup, and three pile sections (pipe or solid, H, a
reinforced-concrete circle with Branson's cracked EI, and composite
transformed sections). The clay criteria and weak rock were checked
against these same pages in GM-202, GM-208 and GM-159 and are not
repeated here; this item reads the sand criteria, the k tables, the
solver, the sections, the head conditions and the printed examples.

## Sand, Reese et al. (1974) (GEC 9 App. A.4, pp. 226-230; COM624P pp. 340-347)

- **p_s.** alpha = phi/2, beta = 45 + phi/2, K0 = 0.4, Ka = tan^2(45 -
  phi/2) (Eq. A-22, p. 227); p_s the smaller of A-23 (wedge) and A-24
  (flow). Collected as p_st = (C1 z + C2 D) gamma z and p_sd = C3 D gamma z,
  at 30 deg C1/C2/C3 = 1.912 / 2.667 / 28.745. The module took C1, C2, C3
  from a table cited "COM624P Table 2.1": no such table is in the COM624P
  manual, and its values (1.00 / 1.25 / 67 at 30 deg; 2.28 / 2.98 / 180 at
  35) are printed nowhere held. **Fixed**: `_reese_sand_coefficients(phi)`
  is A-22 to A-24 in closed form.
- **A-23's third term.** GEC 9 prints K0 tan(beta) (tan(phi) sin(beta) -
  tan(alpha)) with no z. COM624P Eq. 3.31 (p. 341) prints it with one (K_o
  x tan(beta) ...), every other term in the bracket is a length, and API
  Fig. 6.8.6-1, which plots the same C1, reads 1.914 at 30 deg: 1.912 with
  the z, 1.751 without. The module keeps the z.
- **The curve.** y_u = 3D/80, p_u = A p_s (Eq. A-25, A from Fig. A-10);
  y_m = D/60, p_m = B p_s (Eq. A-26, B from Fig. A-11); the initial line
  p = (k z) y (Eq. A-27, k from Table A-3); the parabola p = C y^(1/n)
  through the m-point with the m-segment's slope (Eqs. A-28 to A-31); y_k
  from Eq. A-32 (p. 228). The module's `construction='reese1974'` matches.
  Its A and B tables (read off COM624P Figs. 3.12, 3.13 in 2026-07) lie
  within about 0.05 of GEC 9's redraws, Figs. A-10 (p. 229) and A-11 (p.
  230); A_c at the surface reads 0.70 on A-10 against the table's 0.75.
- **The default.** The module's default was `construction='simplified'`,
  its own curve: the initial line, a 1/3-power parabola through the
  ultimate point and API's A = max(0.9, 3 - 0.8 z/D). No held page prints
  it. **Fixed**: the default is now 'reese1974'; 'simplified' stays as an
  option, and the result's `choices` says no page prints it.
- **Step 10 (p. 228).** "For y >= 3D/80, p is equal to the ultimate
  resistance per unit length ... calculated in Step 3": Step 3 is p_s, but
  Fig. A-9 and Step 6 put the plateau at p_u = A p_s, and COM624P's Fig.
  3.11 the same. The module follows Fig. A-9 (a slip in the step's text).
- **Where the line misses the parabola.** Step 10 also says that where
  A-27 and A-28 have no intersection, the line runs until it meets the
  Step 10 branch. The module takes the smaller of the line and the curve,
  which can meet the m-segment first. Noted, not fixed (rare).

## Sand k (GEC 9 Table A-3, p. 228; COM624P Tables 3.4, 3.5, p. 346; API Fig. 6.8.7-1, p. 70)

- **Table A-3** gives k by relative density only: loose / medium / dense
  20 / 60 / 125 lb/in^3 submerged, 25 / 90 / 225 above the water (5,429 /
  16,287 / 33,931 and 6,786 / 24,430 / 61,076 kN/m^3). The module's
  `_sand_k_recommendation(phi)` mapped phi 25 / 30 / 35 to the three
  columns and gave phi 40 values no page prints (45,000 and 170,000
  kN/m^3; GM-28's flag). **Fixed**: it now takes 'loose', 'medium' or
  'dense' (`SAND_K_TABLE_A3`), and every sand layer's result lists the
  table's values beside the k in force (`choices`).
- **API Fig. 6.8.7-1** gives k by phi' (relative density on the same
  gridlines: phi' 28 / 29 / 30 / 36 / 40 / 45 at Dr 0 / 20 / 40 / 60 / 80 /
  100), below and above the water table; Sec. 6.8.7 says to "determine
  [k] from Figure 6.8.7-1 as function of angle of internal friction". The
  curves were read off the figure's vector paths, each axis calibrated on
  its drawn ticks: below the water, 34.1 / 82.0 / 156.6 lb/in^3 at 30 / 35
  / 40 deg; above it, 46.6 / 136.2 / 277.3. `SoilProfile` built API sand
  layers with an unprinted step table cited "API RP2A Table 6.8-1" (no
  such table). **Fixed**: `_estimate_k_sand` reads Fig. 6.8.7-1 (below
  phi' 28.8, where the curves start, and above 40 it takes the end value),
  and the layer's `choices` says so.

## Sand, API (API RP 2A-WSD Secs. 6.8.6-6.8.7, pp. 70-71)

- **p_u.** p_us = (C1 H + C2 D) gamma H (Eq. 6.8.6-1), p_ud = C3 D gamma H
  (Eq. 6.8.6-2), the smaller; C1, C2, C3 "determined from Figure 6.8.6-1
  as function of phi'"; no equation is given. The module took them from
  the same unprinted table as Reese sand. **Fixed**: `_API_SAND_*` holds
  the figure read off its vector curves at each degree, 20 to 40 (1.914 /
  2.679 / 28.56 at 30 deg; 2.988 / 3.461 / 54.03 at 35); outside 20-40 the
  nearer end, with a warning. The chart and A-22 to A-24 agree within
  1.2 % for C1 and C2 and 4.1 % for C3 from 30 to 40 deg (2 % and 7 % at
  25).
- **The curve.** P = A p_u tanh(k H y / (A p_u)) (Eq. 6.8.7-1), A = 0.9
  cyclic, (3.0 - 0.8 H/D) >= 0.9 static. The module matches.
- **D.** API's D is the "average pile diameter from surface to depth"; the
  module takes the diameter at the depth. The same for a uniform pile.
  Noted, not fixed.

## Liquefied sand, Rollins et al. (2005) (GEC 9 App. A.6, pp. 232-233)

- p = A (B y)^C with p in kN/m, y in mm, z in m (Eq. A-43); A = 3e-7 (z +
  1)^6.05, B = 2.80 (z + 1)^0.11, C = 2.85 (z + 1)^-0.41; times P_d = 3.81
  ln D + 5.6 (Eq. A-44, D in m, for about 0.3 to 1 m). The module put y in
  metres, so p came out 1000^C times too small (C runs from 2.85 at the
  surface to 1.3 at 20 ft). **Fixed.**
- Printed limits: p up to about 85 lb/in (14.9 kN/m), y up to 6 in, depth
  up to 20 ft, Dr about 50 %, gamma' about 61.8 pcf. Beyond 150 mm the
  module holds p at its 150 mm value; no page prints the curve there
  (stated in the docstring).
- GEC 9 says A-43 holds for D = 324 mm and P_d corrects other diameters,
  yet P_d is 1.31 at 0.324 m (1 at 0.30 m). P_d is applied at every
  diameter, as printed. Noted, not fixed: an owner question below.
- The criterion is not on the agent tool, so no saved report used it.

## Soft clay, Jeanjean (2009)

Neither Jeanjean (2009, OTC 20158) nor API RP 2GEO (2014) is in the
reference library, and GEC 9 does not print the criterion (its soft clay
is Matlock's, App. A.1). The model is kept as it was; its docstring, the
result's basis and the tool text now say it is not checked against a
print. Owner question below.

## Solver and boundary conditions (GEC 9 Sec. 6.3, p. 69; COM624P pp. 297-300)

- **The equation.** EI y'''' + P_x y'' - p - W = 0 (GEC 9 Eq. 6-1;
  COM624P Eq. 2.7 without W). The module solves it without W, a
  distributed load: none of its tools take one. Noted.
- **Shear.** The module reports V = EI y''' + P_x y', the shear on a
  horizontal plane (COM624P's V_y, Eq. 2.2). COM624P Eq. 2.10 (p. 300) and
  GEC 9 Eq. 6-2 print V = EI y''', the shear normal to the deflected axis,
  V_n = V_y - P_x dy/dx (COM624P Eq. 2.9, p. 299; "V_n will mostly be used
  in computations"). They differ by P_x times the slope: 0.003 kip in
  Example B.1, nothing at a fixed head. Noted, not fixed. The head shear
  boundary condition is V_y = V_t, the horizontal load, which is right
  for either.
- **Tip.** M = 0 and V = 0. Matches COM624P's.
- **Partial head fixity.** GEC 9 Sec. 6.7.3 (p. 89): a pile cap's
  rotational restraint puts the head between free and fixed. The solver's
  row read M = -K_r x slope, which drives the rotation instead of resisting
  it: on validation.py's soft clay case (a 0.61 m pipe, su 25 kPa, 100
  kN) K_r 1e4 kN-m/rad gave 24.57 mm against 19.24 mm free, and 1e5 gave
  -4.79 mm. **Fixed**: M = +K_r x slope. Now 18.87 / 16.24 / 8.99 / 5.15
  mm at K_r 1e3 / 1e4 / 1e5 / 1e7, between free (19.24) and fixed (5.10). The agent tool offers free and fixed heads only; the
  partial head is reached from the library.

## Pile section (GEC 9 ch. 11)

- **Pipe.** I = pi (d0^4 - di^4)/64 (Eq. 11-51, p. 171). Matches.
- **Concrete.** GEC 9 Eq. 11-18 (p. 159) prints Ec = 4,730 sqrt(f'c)
  (MPa) as the stress-strain curve's initial slope; the module takes ACI's
  4,700, 0.6 % lower. f_r = 0.62 sqrt(f'c) (Eq. 11-20, p. 159). Matches.
- **Cracked EI.** GEC 9 Sec. 11.3.10 (p. 158) computes the nonlinear EI
  from moment and curvature; the module offers Branson's effective
  moment of inertia (ACI 318) and the uncracked transformed section. No
  moment-curvature analysis. Noted (OWN-19's subject).

## Printed examples

- **GEC 9 Example B.1 (pp. 235-242). Pinned.** The ITS pole: a 54 in
  drilled shaft 45 ft long, top 1.5 ft above the ground, free head; Table
  B-1 (p. 236): Reese sand, phi 30 deg, to 15 ft below the top (gamma'
  115 pcf, k 90 pci), below the water table to 30 ft (52.6 pcf, 60 pci),
  soft clay to 40 ft (su 300 psf, 47.6 pcf, eps50 0.02), Reese sand to the
  tip (57.6 pcf, 80 pci); f'c 4,000 psi, 20 #10 bars, 3 in cover to the
  bars. Printed (LPILE, p. 239): 0.03 in under the service loads (144
  kip-ft, 3 kip, 9 kip axial); 727 kip-ft and 35.5 kip under the strength
  loads (672 kip-ft, 14 kip, 12 kip). The module, with the shaft's
  uncracked transformed section: 0.0333 in, 735.6 kip-ft (+1.2 %), 36.8
  kip (+3.6 %). LPILE took the nonlinear EI; above the cracking moment
  (608 kip-ft) the uncracked section is a little stiff, so the moment is
  a little high. With Branson's EI (`Pile.from_rc_section`): 0.0355 in,
  691.4 kip-ft, 47.1 kip, further off.
  `validation.py::TestPublishedBenchmarks::test_gec9_example_b1_its_pole`.
- **GEC 9 Example B.2** is a pile group; out of scope (pile_group).
- **COM624P Example 1** (pp. 101-126, stiff clay above the water) was
  pinned in GM-202 and does not move.
- **COM624P Example 2** (a 36 in concrete pile in layered soil with sand)
  uses COM624P's equivalent-depth method for layered soil, which the
  module does not have; not a clean pin. Not pinned.
- **V-017** (FHWA Micropile Manual Sample Problem 2, LPILE 4; Reese sand)
  moves toward the print, below.

## Values that move (old -> new; hand-checked against the print)

Recorded before and after over every lateral_pile, pile_group and
calc_package test, the lateral and sand tests of validation_examples,
funhouse_agent and geotech_common, and the lateral probes of
test_defaults_used: 5,475 results before. 3,637 are unchanged, 1,506
moved, 332 have no counterpart (325 points of API sand curves, whose y
grid scales with p_u; 7 calls to the renamed coefficient helpers). Every
moved result holds a sand model or a sand k: no clay or rock result moved.

| Input | Old | New | Cause |
|---|---|---|---|
| Reese sand p_s, phi 30, gamma 9, D 0.6 m, z 3 m | 101.2 kN/m | 198.0 | GEC 9 A-22 to A-24 |
| Reese sand p_s, phi 35, same | 233.3 kN/m | 296.0 | the same |
| Reese sand p at y_m = D/60, phi 35, gamma 9, k 22,000, z 3 m (default curve) | 160.2 kN/m ('simplified') | 148.0 ('reese1974') | the same, and the default construction |
| API sand p_u, phi 35, gamma 9, D 0.6 m, z 3 m | 233.3 kN/m | 298.1 | API Fig. 6.8.6-1 |
| liquefied sand, z 1 m, y 50 mm, D 0.6 m | 1.3e-6 kN/m | 3.43 | y in mm (Eq. A-43) |
| `SoilProfile` API sand k, phi 30 above the water | 16,300 kN/m^3 | 12,649 | API Fig. 6.8.7-1 |
| `SoilProfile` API sand k, phi 32 below the water | 22,000 | 13,898 | the same |
| `SoilProfile` API sand k, phi 33 below the water | 33,200 | 16,477 | the same |
| `_sand_k_recommendation(30)` | 16,300 (phi 30 read as medium) | refused: takes 'loose', 'medium', 'dense' | GEC 9 Table A-3 |
| partial head, validation.py soft clay case, 100 kN, K_r 1e4 | 24.57 mm (free 19.24) | 16.24 mm (fixed 5.10) | the solver's sign |
| GEC 9 B.1, transformed section | 0.0333 in, 744.7 kip-ft, 35.2 kip | 0.0333 in, 735.6, 36.8 | Reese p_s, default construction |
| V-017, casing-only EI | 3.92 mm, -39.3 kN-m | 3.57 mm, -36.8 (print 3.3, -37.3) | the same |
| V-017, composite EI | 3.60 mm | 3.32 mm (print 3.3) | the same |
| V-017, 'reese1974' / 'simplified' | 5.02 / 3.92 mm | 3.57 / 2.83 mm | Reese p_s |
| API sand benchmark (validation.py), 89 kN | 3.62 mm, 97.7 kN-m | 3.68 mm, 98.9 | API Fig. 6.8.6-1 |
| the same, 178 kN | 8.61 mm, 217.5 kN-m | 9.06 mm, 225.1 | the same |
| lateral_pile_package probe (API sand, phi 32, 100 kN) | 2.507 mm, 267.1 kN-m | 2.502 mm, 266.8 | the same |
| calc package test, Reese sand, 100 kN | 2.64 mm, 151.0 kN-m | 2.46 mm, 142.2 | Reese p_s, default construction |

The other moved solves are API sand cases within 1 to 5 % (the
validation.py sand equilibrium, mesh and proportionality cases; the
adapter's stickup and pile-E cases; the package parity cases).

Test text that moved with them: `test_sand_reese_construction_default_is_reese1974`
(was `..._is_simplified`); the two API sand benchmarks; V-017's six tests
and RESULTS.md rows; `test_choices.py` (a sand layer now adds its k's
choice); two `SoilProfile` tests in `test_adapters.py` (the sand k's
choice comes first); `test_defaults_used`'s probe lists `tolerance` as not
reached (the probe's sand now converges in the same three iterations at
1e-5 and 1.5e-5). New tests: p_s against A-22 to A-24 and API's chart,
the chart read-off, Table A-3, Rollins' units, the partial head, B.1.

The agent tool's text changed: `construction` defaults to 'reese1974',
the sand k's sources and values, Jeanjean unchecked.

## Noted, not fixed

- GEC 9 prints A-23's third term without its z (a misprint; COM624P Eq.
  3.31 and API's chart have it).
- GEC 9 Step 10 names p_s for the plateau; Fig. A-9 puts it at A p_s
  (used).
- Where the initial line misses the parabola, GEC 9 runs it to the
  plateau; the module takes the smaller of line and curve (rare).
- API's D is the average diameter to the depth; the module takes it at
  the depth (the same for a uniform pile).
- Ec: GEC 9 Eq. 11-18 prints 4,730 sqrt(f'c); the module takes ACI's
  4,700 (0.6 %).
- Cracked EI: GEC 9 Sec. 11.3.10 computes moment-curvature; the module
  has Branson's and the uncracked transformed section (OWN-19).
- Shear: the module reports V_y = EI y''' + P_x y'; COM624P Eq. 2.10 and
  GEC 9 Eq. 6-2 print V_n = EI y''' (0.003 kip apart in B.1).
- No distributed load W in the equation (GEC 9 Eq. 6-1); no tool takes
  one.
- Liquefied sand: P_d is 1.31 at the 0.324 m pile A-43 was built on.
- Owner questions: get Jeanjean (2009) or API RP 2GEO, or remove
  SoftClayJeanjean; whether P_d should be 1 at 0.324 m.
