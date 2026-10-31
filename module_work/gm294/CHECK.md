# wave_equation against GEC-12 and UFC 3-220-20, as printed

BACKLOG GM-294, 2026-10-31. Read from the PDFs in `geotech-references/docs/`:
GEC-12 (FHWA-NHI-16-009/010, 2016) Vol 1 Ch. 7-8, Vol 2 Ch. 12 (pp. 153-246),
Ch. 15 (pp. 356-359) and Appendix C (pp. 519-541), Vol 3 Appendix D (pp.
37-43); UFC 3-220-20 (16 January 2025) Sec. 6-9.2 (pp. 486-488). Page numbers
are the printed ones.

Scope: everything in `wave_equation/` (hammer, cushion, pile, soil, time
integration, bearing graph, drivability, calc steps) and
`funhouse_agent/adapters/wave_equation.py`. Each equation, default and limit
was read against the print. What was fixed and what was only noted is below.
GEC-12's worked examples run GRLWEAP's diesel models, two cushions, triangular
skin and partial embedment, which this module does not model; none is pinned
as a validation example. Example 6 (Vulcan 014) is the nearest and is
compared under "What moved".

## Permanent set (fixed)

- **Print.** GEC-12 Vol 2 Sec. 12.3, p. 158: "The permanent set ... is
  calculated by subtracting a weighted average of the shaft and toe quakes
  from the maximum pile toe displacement." The weights are not printed.
- **Module before.** The toe spring's plastic offset, max(D_toe,max - Q_toe,
  0). The same with equal quakes; it ignored the shaft quake, and with no
  toe resistance (skin_fraction 1.0) the offset was zero, so a friction pile
  read refusal (blow count 1,000,000) whatever it penetrated.
- **Now.** s = D_toe,max - (f Q_side + (1 - f) Q_toe), f the share of R_ult on
  the shaft (`SoilSetup.weighted_quake`). The weights are the reading taken;
  on Example 6 with the 0.40 in toe quake it gives 92.6 blows/ft against
  GRLWEAP's 94 (Table 12-2 text, p. 183), where the old rule gave 305. The
  calc package prints the step, with the reading stated.

## Smith damping (fixed)

- **Print.** UFC 3-220-20 p. 487: Smith damping is "the product of the
  damping factor, pile velocity, and mobilized static resistance"; p. 488:
  Smith viscous "uses the fully-mobilized static resistance in place of the
  mobilized static resistance", proportional to velocity. GEC-12 Vol 2
  p. 156 states the Smith form; p. 242 sets Smith damping "in most
  instances" and Smith viscous for a residual stress analysis, a vibratory
  hammer, and sometimes a refined analysis.
- **Module before.** `smith` matched. `smith_viscous` damped only while the
  velocity and displacement had the same sign, which no reference prints.
- **Now.** `smith_viscous` is J R_ult v throughout the blow. A no-tension toe
  spring whose gap is open carries no damping either (the toe has left the
  soil; before, `smith_viscous` could pull on it). The calc package now
  prints the damping form the run used (it printed Smith damping for a
  `smith_viscous` package, whose analysis carries no soil object).

## Quakes and damping factors (defaults kept or corrected; descriptions)

- **Print.** GEC-12 Vol 2 Sec. 12.6.7, p. 211: shaft quake 0.10 in "usually
  satisfactory"; shaft damping 0.05 s/ft non-cohesive, 0.10 silty and clayey
  sands and sandy silts, 0.15 cohesive silts and sandy clays, 0.20 cohesive
  soils, perhaps twice as high for restrike; toe damping about 0.15 s/ft for
  most soils; toe quake D/120 to D/60 for displacement piles, more in
  saturated or sensitive soils. P. 171: 0.10 in toe quake for a
  non-displacement (H-) pile. P. 212: hard rock, toe quake 0.04 in, toe
  damping lower (0.05 s/ft). Examples: D/60 medium or dense sand, D/120 very
  dense soil (pp. 162, 170, 182). Vol 3 p. 40: H-piles D/60 and D/120. UFC
  p. 487: base quake 0.04 to 0.4 in, side damping 0.05 to 0.2 s/ft, base
  0.15 s/ft.
- **Module.** Shaft quake 2.5 mm, toe quake 2.5 mm, shaft damping 0.16 s/m
  match (the toe quake as the non-displacement pile's). Toe damping 0.50 s/m
  is now **0.49** (0.15 s/ft). The tool's descriptions now carry the printed
  options with pages, so the agent picks D/60 or D/120 for a displacement
  pile instead of the H-pile default.

## Hammers (fixed)

- **Print.** GEC-12 Vol 2 Appendix C, "Pile Hammer Information" (pp.
  519-541): rated energy, ram weight and stroke of each hammer GRLWEAP
  lists. Efficiency 0.67 for an air hammer, 0.80 for an open end diesel
  (Example 6, p. 185); about 0.95 for a hydraulic hammer with measured ram
  kinetic energy (p. 205). P. 206 counts the ICE I-Series among the diesels
  with fuel pump settings. Impact velocity: Eq. 15-1 (p. 358), with the
  stroke times the efficiency (p. 205).
- **Module before.** Vulcan 08 and 010 strokes 0.914 m (printed 3.25 ft,
  0.991 m). Delmag D19-32, D30-32, D46-32, D62-22: strokes 2.5 to 2.8 m and
  rated energies 53.2 to 153.8 kN-m (printed 3.23 to 3.67 m and 57.5 to
  223.2 kN-m); D19-32 ram 19.0 kN (printed 4.00 kips, 17.8 kN), D62-22 61.5
  (13.66 kips, 60.8). ICE I-30V2 and I-46V2 listed as hydraulic at 1.22 m
  and 0.95 (printed open end diesels, 11.50 and 11.81 ft).
- **Now.** All 14 hammers take Appendix C's ram weight, stroke and rated
  energy, to the SI rounding (a test pins each); the ICE I-series are
  diesels at 0.80. Appendix C lists the I-30 v2 twice (No. 1251, p. 534:
  6.62 kips, 10.84 ft; No. 1504, p. 536: 6.61 kips, 11.50 ft); the 1500
  series, which holds every v2 model, is taken.

## Numerical time step (fixed)

- **Print.** GEC-12 Vol 2 Sec. 12.6.8, p. 213: the critical time increment
  is divided by Smith's "safety factor against numerical instability",
  normally 1.6 in GRLWEAP (3 or 5 when warnings suggest it).
- **Module before.** 0.8 x the critical step (a factor of 1.25).
- **Now.** Divided by 1.6 (`NUMERICAL_SAFETY_FACTOR`). Results move under
  1 %.

## Drivability refusal (fixed)

- **Print.** GEC-12 Vol 2 Sec. 12.4, p. 161: hammer, energy and cushion
  "should always be chosen such that the maximum expected penetration
  resistance is less than 120 blows/ft"; the one exception, piles through
  soft soils seated on hard rock, 5 blows per 1/4 in (240 blows/ft). Vol 3
  p. 43: "The recommended blow count limit is 120 blows per foot" for
  practical refusal in the drivability study. The Vol 2 examples call
  refusal over 240 blows/ft (pp. 172, 177, 188).
- **Module before.** 3000 blows/m (914 blows/ft), no source.
- **Now.** 393.7 blows/m (120 blows/ft) by default; the description gives
  787.4 (240 blows/ft) for a pile seated on hard rock. The result, its
  summary and the calc package state the limit used.

## Pile (fixed)

- Steel unit weight 78.5 kN/m3 is now **77.3** (492 lb/ft3, GEC-12 Vol 2
  p. 462), in the module and the tool. Steel E 200e6 kPa = 29,000 ksi
  (Vol 1 p. 317), kept. Segments about 1 m (Vol 2 p. 155; GRLWEAP 3.3 ft,
  p. 213), kept. Concrete E "on the order of 4,000 ksi" varying with
  strength (Vol 1 p. 317), dynamic higher than static (Vol 2 p. 213): in the
  description; the module has no concrete default.

## Driving stresses in the calc package (fixed)

- **Print.** GEC-12 Vol 1 Eq. 8-33 (p. 445): steel sigma_dr = phi_da (0.9 Fy),
  phi_da = 1.0. Eqs. 8-62 and 8-63 (p. 459): concrete 0.85 f'c - fpe in
  compression, 0.095 sqrt(f'c) + fpe in tension (ksi). New H-piles: Fy 50
  ksi (Vol 3 p. 38).
- **Module before.** A "Compressive Stress Ratio" step: stress over E as a
  strain, and the stress over an assumed fy = 250 MPa.
- **Now.** The yield stress the maximum compression needs, Fy >= sigma / 0.9
  (Eq. 8-33), with the 50 ksi note and the concrete equations named. The
  module has no pile grade or f'c input, so it does not pass or fail.

## Tool text (fixed)

- `bearing_graph` "returns" listed `max_pile_force_kN`, which the method
  does not return; now its real keys. `drivability` returns
  `refusal_blow_count_per_m`. Cushion, helmet, efficiency, pile E and unit
  weight, segment length and damping model descriptions carry the printed
  values with pages, and say which defaults are not printed.

## What moved (made-up cases, before -> after)

Steel pipe 0.32 m x 10 mm, 18 m, cushion 500,000 kN/m COR 0.80, Vulcan 010
unless named; half the resistance on the shaft unless given.

| Case | Change |
|---|---|
| Defaults, 900 kN | 68.2 -> 62.5 blows/m; 204.5 -> 212.5 MPa (67.7 with the old stroke: 0.5 from the step, toe damping and unit weight, the rest the Vulcan 010's printed stroke) |
| Toe quake 5 mm, 30 % shaft, 900 kN | 82.4 -> 70.3 (77.0 with the old stroke: the weighted quake 4.25 mm) |
| All shaft, 600 kN | refusal -> 31.8 |
| Drivability 6/12/18 m at 900/1500/2000 kN | 64.0/148.0/1430.9, can drive -> 59.1/131.3/719.1, refusal at 18 m (393.7 limit) |
| ICE I-30V2, 900 kN | 4.77 -> 7.42 m/s; 61.7 -> 29.7 blows/m |
| Delmag D30-32, 1500 kN | 6.63 -> 7.39 m/s; 63.6 -> 50.2 blows/m; 375.9 -> 418.5 MPa |
| smith_viscous, 900 kN | 69.4 -> 63.5 (69.0 with the old stroke) |
| Bearing graph 300 to 1500 kN | 22.0/44.4/68.2/107.7/191.0 -> 20.4/41.0/62.5/95.6/163.5 |
| GEC-12 Ex 6, Vulcan 014, 400 kips, 30 % shaft, J_side 0.05 s/ft (GRLWEAP 56 and 94 blows/ft) | toe quake 0.12 in: 52.9 -> 51.7; 0.40 in: 305 -> 92.6 |

Ex 6 is run with the full 66 ft embedded and Table 12-2's Nycast hammer
cushion as the one cushion (the module has no stick-up or pile cushion), so
it is a comparison, not a validation.

## Noted, not fixed

- **Defaults not printed.** The 500,000 kN/m cushion, its COR 0.80 and the
  5 kN helmet are in no held reference. GEC-12 Vol 2 gives wood cushions
  COR 0.50 (0.25 badly worn, p. 156), its examples' hammer cushions 0.90 to
  0.92 (pp. 174, 183) and about 1.4 million kN/m (Ex 6, Vulcan 014), helmets
  1.0 to 4.07 kips. The descriptions say so. Backlog candidate: they change
  every run that leaves the cushion or helmet out.
- **One cushion, helmet lumped.** GEC-12 Fig. 12-2 (p. 157) has a hammer
  cushion, a helmet mass and a pile cushion; the module has one cushion and
  the helmet's weight on the first segment, so its head stress carries the
  helmet's inertia. Two cushions in series can be given as one (1/k = 1/k1 +
  1/k2, p. 209).
- **No embedment input.** The pile is taken as embedded over its full
  length, the shaft resistance uniform (Ex 1 is triangular; Ex 6 embeds 52.5
  of 66 ft). Drivability takes the pile as long as each depth.
- **The drivability tool takes no quake or damping inputs** (the module
  does); it runs the defaults. Backlog candidate if a displacement pile's
  D/60 toe quake should be reachable there.
- **Diesel hammers** are a rigid ram at rated energy x 0.80; GRLWEAP's
  combustion cycle and fuel settings (p. 206) are not modelled.
- **Cushion unloading** with energy return COR^2 is not printed in a held
  reference (p. 156 defines the COR only).
- **Steel wave speed** is 5,038 m/s from E and 77.3 kN/m3; UFC p. 487 says
  about 16,800 ft/s (5,121 m/s); GEC-12 p. 213 allows E up to 7 % higher for
  large pipes. Not changed.
- **The 30 blows/ft lower bound** (GEC-12 Vol 3 pp. 37-38: 30 to 120
  blows/ft) is not reported.
- `geotech_common.check_wave_equation` (outside this module) limits concrete
  tension to 0.7 sqrt(f'c) MPa, not Eq. 8-63; nothing in a run calls it.

## Owner questions

None. The weighted-quake weights are a reading (the page names none); the
R_ult shares reproduce Example 6's 94 blows/ft. 120 against 240 blows/ft is
the printed rule against its printed exception.
