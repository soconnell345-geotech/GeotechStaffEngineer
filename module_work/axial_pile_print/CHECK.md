# axial_pile against GEC-12 chapter 7, as printed

BACKLOG GM-282, 2026-10-30. Read from the PDFs in `geotech-references/docs/`:
GEC-12 (FHWA-NHI-16-009, 2016) Vol 1 chapter 7 and Vol 3 Appendix D; FHWA
NHI-06-089 (2006) Vol 2 ch. 9 for the factor of safety. Page numbers are the
printed ones (GEC-12 Vol 1: pdf index = page + 45; Vol 3: page + 33).

The driven-pile static methods the module offers: Nordlund (sand), the alpha
method (clay; Fig. 7-17, Fig. 7-18 as an option, API alpha past the charts),
the beta method (every layer, `method="beta"`), open-end pipes (Paikowsky and
Whitman, Tomlinson), uplift. It offers no CPT method (Nottingham and
Schmertmann, Eslami and Fellenius), no Brown (2001) and not the full API
method; those are not checked here.

Charts were checked on each figure's own image: the figures are 150 dpi
rasters inside the PDF (Figs. 7-9, 7-14, 7-15 pdf 280/283, Fig. 7-16 pdf
284, Figs. 7-17/7-18 pdf 293/294), extracted as they are and read column by
column (the darkness-weighted centre of the drawn line) against the
figure's own gridlines.

## Nordlund (Sec. 7.2.1.3.1, Steps 1-11, pp. 242-245)

- **Equations.** fs = K_delta C_F sigma'd sin delta per layer (Eq. 7-8,
  Steps 5-6); Rp = alpha_t N'q sigma'p Ap, at most qL Ap (Eq. 7-9, Steps
  7-9). The module matches.
- **Fig. 7-9 (p. 235), delta/phi by V.** Curves a, b, c and f at V = 0.4,
  0.5, 0.8, 1.0, 1.5, 1.8 ft3/ft: the module within 0.003 of the print.
- **Tables 7-6 and 7-7 (pp. 240-241), K_delta.** The V = 0.1, 1 and 10
  columns match the module as printed; the printed in-between columns are
  log-linear in V (phi 25, V 0.2: 0.75; phi 40, V 0.5 and 2: 2.61 and
  3.39), as Step 3d says and the module does.
- **Fig. 7-14 (p. 238), C_F.** delta/phi 0.2 to 1.4 at phi 20-45: within
  0.003.
- **Fig. 7-15 (p. 238), qL.** phi 31-43.5: within 1 % (e.g. 40 deg: print
  208.7 tsf, module 208.5).
- **Fig. 7-16(a) (p. 239), alpha_t.** Every degree 20-45, the three D/b
  curves: within 0.004 (the curves merge from about 35 deg).
- **Fig. 7-16(b) (p. 239), N'q. Fixed.** The module's nodes (measured
  2026-10-18) read 1.0-3.5 % low from 29 deg up: at 30 / 35 / 40 / 44 deg
  the print gives 30.1 / 63.8 / 160.4 / 374.0, the module had 29.6 / 62.8 /
  155.3 / 359.3. Read again at every degree on the figure's own image (the
  decade gridlines 174.5 px apart; a pixel is 1.3 % in N'q, the misses
  were 1-3 px at the top end). Below 29 deg the two agree within 0.7 %. GEC-12's
  own worked example agrees with the new nodes: Table D-6's toe below its
  plateau (sigma'p not limited) was 2.0-2.3 % (toe phi 33), 4.6-4.9 % (36)
  and 2.7 % (40) above the module; now 0.3-0.7 % and 1.9-2.3 % above it
  and 0.6 % below it.
  The AP-1 eval key's hand derivation (GM-238) read N'q = 40 at 32 deg off
  the figure; the old node was 39.3, the new 40.0.
- **Limits.** Step 5: no limit on sigma'd; Step 8 (p. 244): "The limiting
  value of sigma'p is 3 ksf"; Step 9: qL from Fig. 7-15. The module applies
  them (Step 8 by default, `toe_stress_limit`).
- **p. 232.** Piles over 18 in were outside the correlation database;
  Nordlund tends to over predict over 24 in. Warnings on the result.
- **Resistance factor.** Table 7-1: Nordlund (sand), shaft and toe, 0.45;
  in tension 0.35. Added (see below).

## alpha method (Sec. 7.2.1.3.2, Steps 1-6, pp. 245-250)

- **Equations.** fs = Ca = alpha su (Eq. 7-10); qp = Nc su, Nc usually 9
  (Eq. 7-11). The module matches.
- **Fig. 7-17 (p. 248).** Both families at D = 10b and 40b, su 0.3-3.4
  ksf: within 0.005 ksf. D from the ground surface to the bottom of the clay
  layer, or to the toe if less (p. 246): as the module takes it. H-piles on
  the corrugated curve with the box perimeter (p. 247): the default.
- **Fig. 7-18 (p. 249).** All three graphs at su 0.8-4.4 ksf: within
  0.007.
- **Past the charts (p. 247, Step 1 p. 250).** API alpha recommended in
  highly overconsolidated clays; the line is the engineer's (GM-229).
- **Shrink-swell zone (p. 247).** No adhesion there: `shrink_swell_depth`.
- **Resistance factor.** Table 7-1 "Shaft Resistance and Toe Resistance:
  Clay and Mixed Soils, alpha-method 0.35" (the 9 su toe with it); tension
  0.25. Added.

## API alpha (Sec. 7.2.1.3.3, pp. 251-255)

- **Equations.** alpha = 0.5 psi^-0.5 for psi <= 1.0 (Eq. 7-13), 0.5
  psi^-0.25 for psi > 1.0 (Eq. 7-14), psi = su / sigma'vo (Eq. 7-15). The
  module matches.
- **The 1.0 limit. Reworded.** GEC-12 prints no limit "alpha <= 1.0"; p. 252
  says "An alpha value of 1.0 is recommended for unconsolidated clays",
  where Eq. 7-13 gives more (psi below 0.25). The module keeps alpha at most
  1.0 and now cites p. 252 for it, not "API RP 2A's own limit". It applies
  only where API alpha is taken (su past Fig. 7-17's curves), where psi is
  large, so it does not bind in practice; no number moves.
- **Resistance factor.** p. 255: AASHTO gives none for the API method; use
  phi_dyn. The module takes API alpha only inside the alpha method, as p.
  247 recommends, so such a layer keeps the alpha method's 0.35 (stated in
  the choice).

## beta method (Sec. 7.2.1.3.4, pp. 255-258)

- **Equations.** fs = beta sigma'vo (Eq. 7-19), beta = K tan delta; qp =
  Nt sigma'p (Eq. 7-20). No limiting values (p. 256). Nt about 3 in clay
  (p. 256). The module matches.
- **beta and Nt. Fixed.** Steps 2 and 5 (pp. 257-258): "a. Use local
  experience to select beta coefficient", else "b. use Table 7-9 to
  estimate beta coefficient from phi' angle" (the same for Nt). Table 7-9
  (p. 256) matches the module. The module's DEFAULT was beta = (1 - sin
  phi') tan phi' with Nt from a piecewise curve in phi' (3-10 over 0-20
  deg, ..., +20 per degree above 38). Neither is printed in GEC-12 (nor in
  UFC 3-220-20, whose beta route, p. 412, takes K/K0 from Fig. 6-13 and
  delta/phi' tables). Removed: Table 7-9 by soil type (linear in phi'
  across the type's phi' range, the module's reading of a range) is now the
  default and only automatic route; "from_phi" is refused with the printed
  route named; a layer's own beta and Nt still win (Steps 2a, 5a).
- **Resistance factor.** Table 7-1, compression: "beta-method (1991):
  Differs in Manual" (note 1: AASHTO's beta is Skempton's 1951, the
  manual's Fellenius' 1991); p. 258: "AASHTO (2014) does not provide a
  recommended resistance factor for the nominal resistance determined by
  the beta-Method by Fellenius", use phi_dyn (Table 7-2). The module gives
  no factored compression resistance for the beta method and says why.
  Tension: Table 7-1 prints "beta-method (1991) 0.20"; used as printed,
  with a note that p. 258 gives no compression factor for it.

## Open-end pipes (Sec. 7.10.7, pp. 405-408)

- Eq. 7-88 (plugged) and Eq. 7-89 (unplugged, with -Wp), the lesser
  governs (Paikowsky and Whitman, p. 406); fsi 1/3 to 1/2 of fso (p. 407);
  Tomlinson's 105 ksf in cohesionless soil once Fig. 7-15's qL exceeds it
  (p. 407); Eq. 7-90, Rn = 0.8 Ca As + 4.5 su Ap, for pipes under 24 in in
  predominantly cohesive soil (p. 407); an H-pile's box toe area in soil
  (p. 408). The module matches; no change.
- **Resistance factor.** Table 7-1 has no row for Eq. 7-90: no factored
  resistance under it, said so. Under Eqs. 7-88/7-89 the parts take their
  methods' factors; the inside friction takes the outside's; Eq. 7-89 gives
  no factor for the plug weight, subtracted unfactored (the conservative
  reading).

## Uplift (Sec. 7.2.3.1, pp. 288-289)

- Eq. 7-38: Rr = phi_up Rs, the nominal shaft resistance "calculated from
  any of the AASHTO static analysis methods with a resistance factor for
  uplift, phi_up, from Table 7-1"; AASHTO C10.7.3.10 reduces it to 80 % of
  the compression factor, "Therefore, the shaft resistance does not need to
  be further reduced for the uplift case". Up to 0.60 with a tension load
  test, 0.50 with a dynamic test. The module's nominal uplift is the whole
  outside shaft by default (`uplift_skin_fraction` 1.0); no change.
- **Factored uplift. Added.** Eq. 7-38 with phi_up by layer: Nordlund
  0.35, alpha method 0.25, beta method 0.20 (Vol 3 p. 115 factors a layered
  profile so). On the nominal shaft, without `uplift_skin_fraction` or the
  pile weight.

## Resistance factors (Sec. 7.1.5, Table 7-1, pp. 223-225). Added

- Eq. 7-3 (p. 224): Rr = phi_stat Rs + phi_stat Rp, "the shaft resistance
  multiplied by the resistance factor associated with the shaft resistance
  computation method plus the toe resistance multiplied by the resistance
  factor for the toe resistance computation method" (p. 223).
- Layered profiles: Vol 3 p. 111 (Pier 2, Figs. D-50 and D-54): "phi_stat =
  0.45 was applied to the cohesionless soils as a result of using the
  Nordlund method while phi_stat = 0.35 was applied to the cohesive soils as
  a result of using the alpha method"; p. 115, tension, 0.35 and 0.25.
- The module had no factored resistance. It now returns `Q_factored`
  (Eq. 7-3, layer by layer), `Q_uplift_factored` (Eq. 7-38) and
  `resistance_factors` (each method's factor and the nominal part it
  multiplies), in `to_dict`, the summary, the calc steps and
  `capacity_vs_depth` (`Q_factored_kN`), with a choice citing Table 7-1.
  None, with the reason, where AASHTO gives none (the beta method, Eq.
  7-90).
- Stated, not applied: a small group (fewer than 3 piles per substructure
  unit) reduces the factors by 20 % (p. 225); the AASHTO calibration
  considered few large diameter pipe piles (p. 223); phi_dyn of the field
  method (Table 7-2) sets the number of piles (AASHTO C10.5.5.2.3, p. 225).

## Factor of safety

- GEC-12 (2016) designs by resistance factors and prints no factor of
  safety. FHWA NHI-06-089 (2006) Vol 2 p. 9-25, Table 9-5: static load test
  with wave equation analysis 2.00, dynamic testing with wave equation
  2.25, indicator piles with wave equation 2.50, wave equation 2.75, Gates
  dynamic formula 3.50. The module's default 2.5 is that indicator-pile
  row. Kept; now stated in a choice and the tool description.

## Worked examples (Vol 3 Appendix D)

- **V-001, North Abutment, Table D-6 (p. 28), Nordlund.** Pinned before
  (shaft +1.1 to +1.2 %). Added: the toe below its plateau (above). The
  Step 8 default toe moves 359.7 -> 371.5 kips (-16 % -> -13 % against the
  428.1 plateau the example reached without Step 8).
- **V-002, South Abutment, Table D-100, alpha method.** Pinned before;
  unchanged (no Fig. 7-16 or beta in it).
- **V-059, Pier 2, Table D-51 (p. 100), layered. New.** Nordlund in Layers
  2 and 4, the alpha method in six clay sublayers, toe phi 43 and 38, with
  the example's choices (smooth steel curves, Ca held at the chart's end,
  sigma'p not limited): shaft +1.1 to +1.8 % at every depth 0-60 ft, clay
  toe +0.5 %, Layer 2 toe -2.9 %, Layer 4 toe -2.1 % to -0.1 %. The
  layer-by-layer factoring is checked by its parts (the factored resistance
  is printed only as Fig. D-54).
- No Vol 3 example uses the beta method, an open pipe or Eq. 7-90, and none
  prints a factored value as a number.

## Values that move (old -> new, kN unless stated; checked by hand)

- N'q (Fig. 7-16(b)): every Nordlund toe with toe phi from 29 deg up that
  qL does not cap, +1.0 to +3.5 %. A sand H-pile (toe phi 39) toe 1,288.1
  -> 1,323.5 (x 130.9 / 127.4); a closed pipe in the same sand 1,747.1 ->
  1,795.1; V-001's Step 8 toe 1,600.0 -> 1,652.6 (359.7 -> 371.5 kips, x
  160.4 / 155.3). AP-1 (toe phi 32) does not move: qL governs its toe.
- beta method, Table 7-9 in place of (1 - sin phi') tan phi': sand beta
  0.30-0.90 over phi' 32-40 (was about 0.29-0.30), clay 0.15-0.35 over
  25-30 (was about 0.27); sand Nt 30-150 (was 70-170 at 33-39 deg). A closed
  pipe in sand (phi' 33/37, toe 39), 18 m: shaft 722.8 -> 1,524.8, toe
  4,254.8 -> 3,378.8 (Nt 170 -> 135), total 4,977.6 -> 4,903.6. A layered
  H-pile, 20 m: 3,101.3 -> 2,932.6 (toe Nt 118 -> 90 at 36 deg). A pile
  with its toe in clay, 9 m: 230.8 -> 170.4 (clay beta 0.27 -> 0.15 at the
  25 deg default; toe Nt 3 unchanged).
- New outputs only (no nominal value moves): `Q_factored`,
  `Q_uplift_factored`, `resistance_factors`, `Q_factored_kN` per depth.

## Noted, not fixed

- Eq. 7-38 takes no pile weight; the module adds `pile_weight` to the
  nominal uplift when given (an input, off by default; the factored uplift
  leaves it out).
- The 105 ksf open-pipe limit is also applied to a beta-method toe (Sec.
  7.10.7 reads it for the static resistance of an open pipe in
  cohesionless soil, whatever the method).
- Table 7-9's ranges are read linearly in phi'; GEC-12 gives no other
  reading.
- Table D-51's clay toe uses 1.025 ft2, the module's box 1.029 ft2 (0.4 %).
- No CPT methods, Brown (2001) or the full API method are offered.

## Proposed for the backlog (to the lead)

- `downdrag/analysis.py` `_Nt_from_phi` (line 2984, used at line 1950 for a
  toe in sand without a given Nt) is the same unprinted piecewise curve
  removed here: it changes what a live downdrag run returns.
- `geotech_references.gec_12.figure_7_16b_nq` (the reference library) has
  seven coarse nodes, 4-7 % under the printed figure at 30-40 deg (150 at 40
  deg against 160.4); the agent's reference tool returns it.
