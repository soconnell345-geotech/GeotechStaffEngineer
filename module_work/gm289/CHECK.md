# pile_group: group uplift as GEC-12 Sec. 7.2.3.2 and UFC 3-220-20 print it

BACKLOG GM-289, 2026-10-31. Read from the PDFs in `geotech-references/docs/`:
GEC-12 (FHWA-NHI-16-009) Vol I Tables 7-1 and 7-2 (pp. 224-226) and Sec.
7.2.3 (pp. 288-293), and Vol III Appendix D, Block 13 for the North Abutment (pp. 51-54)
and the South Abutment (pp. 203-209); UFC 3-220-20 (16 January 2025), the
uplift capacity of deep foundations (pp. 433-436). Page numbers are the
printed ones (PDF page = printed + 46 in GEC-12 Vol I, + 34 in Vol III, + 38
in UFC 3-220-20). Rendered and read on the page: GEC-12 Vol I pp. 224, 226,
289, 291, 292, 293; Vol III pp. 53, 54; UFC pp. 434, 435, 436. The Vol III
South Abutment pages (205-208) were read from their text layer.

Before: pile_group checked a group in tension pile by pile only (each
pile's load over `axial_capacity_tension`); DESIGN.md listed group uplift
as not offered.

## What the print says

**GEC-12 Vol I.**

- p. 291: "AASHTO specifications (2014) limit uplift resistance of a pile
  group to the lesser of: 1. The sum of individual pile resistance in axial
  tension. 2. The resistance in axial tension of the pile group considered
  as a block." Eq. 7-39, R_r = phi R_n = phi_ug R_ug, "phi_ug = resistance
  factor for axial tension, 0.50 per Table 7-1".
- Sec. 7.2.3.2.2, cohesionless soils (p. 291): the first resistance is Eq.
  7-40, R_r = phi R_ug = phi_up R_s n. "The second resistance is defined by
  the weight of soil contained with the pile group block depicted by Figure
  7-33. This approach outlined by Tomlinson (1994), uses the effective
  weight of the block of soil extending upward from the pile toe level at a
  slope of 1H:4V. For simplicity in performing the calculation, the weights
  of the piles within the soil block are considered equal to the weight of
  the soil. The group uplift resistance is the lesser resistance determined
  from Equation 7-40 or Figure 7-33." Fig. 7-33 (p. 292): the block from
  "(B)(Z)" at the toes, sides at 1 in 4, height D.
- Sec. 7.2.3.2.3, cohesive soils (p. 292): the lesser of Eq. 7-40 and Eq.
  7-41, R_n = R_ug = 2D(B + Z)s_u + W_g, "D = pile embedded length", "B =
  width of pile group", "Z = length of pile group", "s_u = weighted average
  of the undrained shear strength over the depth of pile embedment along
  the pile group perimeter (ksf)", "W_g = effective weight of the pile/soil
  block including pile cap weight (kips)". Fig. 7-34 (p. 293): vertical
  sides.
- Table 7-1 (p. 224): "Group Uplift Resistance, phi_ug | Sand and Clay |
  0.50"; single piles in tension, phi_up: Nordlund 0.35, alpha 0.25, beta
  0.20, lambda 0.30, SPT 0.25, CPT 0.40. Table 7-2 (p. 226): single pile in
  tension, static load test 0.60, dynamic testing with signal matching 0.50.
  Eq. 7-38 (p. 289): R_r = phi R_n = phi_up R_s.
- p. 225: "If the pile foundation design consists of a small group (i.e.
  less than 3 piles per substructure unit in this manual) AASHTO (2014)
  recommends the resistance factors in Tables 7-1 and 7-2 be reduced by 20%
  ... The definition of a small group ranges from 2 or 3 according to
  Isenhower and Long (1997) to 5 according to Paikowsky et al. (2004)."

**UFC 3-220-20** prints a group uplift method, the same block:

- p. 434: "in pile groups, uplift resistance is limited by the lesser of the
  combined uplift resistances of the individual elements and the uplift
  resistance of a block or wedge of soil surrounding the pile group."
- Table 6-28 (p. 434, AASHTO 2020), "group block uplift, all soils": driven
  piles 0.50, drilled shafts 0.45, micropiles 0.50. Note A: "resistance
  factors should be reduced by 20% for single drilled shafts and small pile
  groups (<5 piles) to account for lack of redundancy and micropiles in
  marginal ground conditions." A single element's effective weight is
  factored by "a minimum dead load factor equal to 0.9".
- Eq. 6-40 (p. 435): R_r,gu = min{ n(R_r,s + W_r,e) + W_r,e,cap ;
  R_r,ublock }. pp. 435-436: the block's weight uses effective unit weights;
  "Usually, the difference between the unit weight of the foundation
  material and the soil may be neglected by treating all materials as
  having the unit weight of soil. The effective weight of the pile cap,
  W_cap, also should account for any expected buoyancy and is included in
  the weight of the block. The effective weights of the soil block and cap
  should be left unfactored since the group block resistance factors
  provided in Table 6-28 are applied to the effective weight and the side
  resistance for undrained conditions."
- p. 436: coarse-grained soils, "the weight of the truncated prism having a
  rectangular base, shown in Figure 6-16(a), plus the weight of the cap";
  Eq. 6-41, V_block = B·L·Z + Z²/4·(B + L) + Z³/12 (its Z the depth, L the
  group length); for layers or a water table in the block, "The thickness
  of each layer should replace Z in Equation 6-41. The plan dimensions B
  and L should be replaced by the dimensions of the block at the bottom of
  each layer, calculated using the 4V:1H slope." Undrained: Eq. 6-42, W_e,g
  = B·L·(Z1·gamma_m + Z2·gamma_b) + W_cap, and Eq. 6-43, R_n,block =
  2Z·(B + L)·s_u,avg + W_e,g. Fig. 6-16 draws (B)(L) across the outer piles.

UFC differs from GEC-12 in three things: its piles' sum adds each pile's and
the cap's effective weight factored by 0.9; drilled shafts take 0.45; a small
group is below 5 piles.

## The worked examples (GEC-12 Vol III, Block 13)

Both check the back row in tension of an abutment group ("only a single row
of piles is providing the tension resistance rather than the entire pile
group", pp. 51, 203), B = 1 ft (the pile width), Z from "exterior pile edge
to exterior pile edge".

- **North Abutment** (pp. 51-54), cohesionless: 406 kips on the back row of
  9 piles; W_block >= 406/0.50 = 812 kips; Z = 41 ft; three layers, gamma'
  105, 42.6, 49.6 pcf (Fig. D-26); Eq. D-1, V = h/6[Z_b·B_t + Z_t·B_b +
  2(Z_t·B_t + Z_b·B_b)]; Table D-18: 6,260.8, 3,310.8 and 830.1 ft3, 657.4,
  141.0 and 41.2 kips, 839.6 kips at 27 ft; for Configuration 3 (Z = 37 ft),
  28 ft.
  **Module:** 839.60 kips, the layer volumes 6,260.8, 3,310.8 and 830.1 ft3;
  Z = 37 ft gives 773.5 kips at 27 ft and 825.6 at 28 ft, so 28 ft.
  Reproduced. Print slips: Table D-17 prints layer 2's B_top as 49.5 (9.5 by
  the 1H:4V geometry, and Table D-18's 3,310.8 ft3 uses 9.5) and layer 3's
  thickness as 20.0 (7 ft to the 27 ft toes, as Table D-18 takes it).
- **South Abutment** (pp. 203-209), cohesive: 276 kips on 13 piles; R_ug >=
  552 kips; D = 9 ft, B = 1 ft, Z = 37 ft, s_u = 0.67 ksf, gamma' = 47.6 pcf;
  W_piles = 0.074 kip/ft × 9 × 13 = 8.66 kips, W_soil = 15.85 kips, W_cap =
  3.25 × 38.5 × 4.2 × 150 pcf = 78.83 kips (half the cap, over the row in
  tension), W_g = 103.3; R_ug = 2(9)(38)(0.67) + 103.3 = 561.6 kips > 552.
  **Module:** 561.62 kips with the piles' 8.66 kips given in with the cap.
  The module takes the piles as soil, as GEC-12 p. 291 and UFC p. 435 say;
  the example adds the steel on top of the full soil prism (the soil the 13
  piles displace weighs about 0.8 kip). Reproduced.

Both examples then take the deeper of the two penetrations (18 ft for the
piles one by one against 9 ft for the block, South Abutment, p. 209): the
lesser resistance at a given depth, as the module returns it.

## Module now

- `group_uplift_resistance(B, Z, D, n_piles, soil_type, layers, su,
  cap_weight, pile_type, piles_resistance, group_piles)`
  (`group_efficiency.py`): the
  block by layers (prismoidal volume, exact for plan dimensions linear in
  depth; one layer is Eq. 6-41), its weight plus the cap's, plus 2D(B+Z)s_u
  in clay; phi_ug by pile type (`GROUP_UPLIFT_PHI`) with the small-group
  20 %; the lesser of phi_ug R_ug and the piles' sum, and which governs.
  Layers that stop short of the toes are refused; a layer past them is cut
  off; fewer than two piles are refused. `group_piles`, the piles under the
  cap, counts for the small-group rule when the block is part of the group
  (p. 225: "per substructure unit").
- Tool `group_uplift`: the layout (n_rows, n_cols, spacing, pile_diameter;
  B and Z to the outer faces), pile_length, soil_type, unit_weight or
  layers, cu (clay), cap_weight (0 when left out, the safe side), pile_type
  (driven), axial_capacity_tension (each pile's phi_up R_s) and uplift_load.
  It returns the block, phi_ug, the factored block, the piles' sum, the
  governing resistance (`group_uplift_resistance_kN`), which governs
  (`group_uplift_governs`) and the utilization. Without the piles'
  resistance it gives the block and says the group's is no more than that.
  n_rows 1 checks one row, as Vol III does; `piles_under_cap` gives the
  group's count for the small-group rule.
- `pile_group_simple` and `pile_group_6dof`: any pile in tension gets
  `group_uplift_note`: how many, their summed tension, and the group_uplift
  call to run over them (n_rows 1 for one row, piles_under_cap, uplift_load);
  a group in net tension, the whole group as well.
- `pile_group_package`, wherever a pile carries tension, a Group Uplift
  section:
  - over the piles in tension (fewer than all), as Vol III checks the back
    row: the block at the toes from their plan extent plus the diameter,
    their summed tension the load, their own `axial_capacity_tension` (each
    pile's factored resistance, as the utilization takes it), the
    small-group rule on all the piles under the cap, and the cap's weight in
    their share, k/n (Vol III p. 207: "Calculate weight of the pile cap,
    conservatively taken as the tributary are[a] over piles in tension",
    half the 6.5 ft cap over one of two rows; the share by count is the
    tributary area of evenly spaced piles);
  - over the whole group when it carries net tension (Vz < 0), the net
    uplift the load (Ch. 7);
  - piles in tension that do not fill a rectangle (a pile in compression
    within their extent) are not checked as a block, and the section says
    so; one pile in tension, its own check only.
  The result carries the tool's keys for the check with the higher
  utilization, `group_uplift_over` (which), and `group_uplift_checks` when
  there are two. Without pile_length, soil_type and unit_weight (cu in
  clay) it says what it needs. A group with no pile in tension gets no
  section.

`choices` state each reading: the lesser and which governs, with UFC Eq.
6-40's piles' sum (the weights added) as the alternative; the block as
printed for the soil, B and Z to the outer faces (Vol III), the piles as
soil; phi_ug with its small-group rule (below 3 driven piles, GEC-12 p. 225;
below 5 for the others, UFC note A), and for 3 or 4 driven piles UFC's
reading (0.40) as the alternative; the cap weight (0 unless given), over
the piles in tension its share. Over the piles in tension each choice's
`what` ends ", piles in tension".

## Worked by hand (made-up, the package probe)

3 × 3 driven piles, 0.4 m, at 1.2 m; D = 12 m; gamma' = 9 kN/m3; each pile's
factored tension resistance 200 kN; net uplift 900 kN. B = Z = 2(1.2) + 0.4
= 2.8 m. V = 2.8²(12) + 12²(2.8 + 2.8)/4 + 12³/12 = 94.08 + 201.60 + 144.00
= 439.68 m3; W = 3,957.1 kN = R_ug; phi_ug R_ug = 0.50 × 3,957.1 = 1,978.6
kN; the piles 9 × 200 = 1,800 kN, the lesser: the piles govern; 900/1,800 =
0.50. The package returns the same (`test_gm289_group_uplift.py`). In clay
(3 × 3 at 1.0 m, 0.3 m piles, D = 12 m, gamma' = 8, s_u = 30 kPa): B = Z =
2.3 m, W = 2.3²(12)(8) = 507.8 kN, shear 2(12)(4.6)(30) = 3,312.0 kN, R_ug =
3,819.8 kN, factored 1,909.9 kN.

Review fix, the piles in tension of a group in compression (made-up):

- 3 × 3 at 1.2 m, 0.4 m piles, Vz 900 kN, My 1,200 kN·m: P = 100 ± 1,200 ×
  1.2/8.64 = 100 ± 166.7, the column at x = −1.2 m in tension, 3 × 66.7 =
  200 kN. Its block B = 0.4, Z = 2(1.2) + 0.4 = 2.8 m, D = 12 m: V = 0.4 ×
  2.8 × 12 + 144 × 3.2/4 + 1,728/12 = 13.44 + 115.20 + 144.00 = 272.64 m3,
  W = 2,453.8 kN, with a 450 kN cap's 3/9 2,603.8 kN; the piles 3 × 200 =
  600 kN govern; 200.1/600 = 0.33 (the package: the same).
- The reviewer's group: 2 × 9, spacing 1.5 m along the rows and 1.2 m
  across, 0.3 m piles, Vz 11,000 kN, Mx 8,800 kN·m: the back row 611.1 −
  8,800 × 0.6/(18 × 0.36) = −203.7 kN a pile, each passing alone (204 of
  250 kN, 0.815); 1,833.3 kN on the row. Its block B = 0.3, Z = 12.3 m, 8 m
  in sand at 9 kN/m3: V = 29.52 + 201.60 + 42.67 = 273.79 m3, R_ug =
  2,464.1 kN, phi_ug R_ug = 1,232.0 kN < 9 × 250 = 2,250: the block
  governs, 1,833.3/1,232.0 = 1.49, the row fails (the package: the same).
- Vol III North Abutment, Group Configuration 1, through the package (Table
  D-12: 2 rows of 9 at Sbx 5 ft and Sby 4 ft; Table D-11: Q 2,505 kips, Muy
  6,625 kip-ft; 1 ft piles at 27 ft, Fig. D-26's layers; 46 kips a pile,
  Table D-14): the rigid cap puts 2,505/2 − 6,625/4 = −403.8 kips on the
  back row (Table D-13 prints −406 from its free body, Fig. D-25; its
  reactions sum to 2,502); the block over the row is Table D-18's 839.6
  kips, 419.8 factored, against the piles' 414 kips: the piles govern,
  0.98.

## Replay: groups not in net tension (as built)

- The pile group tests and guards recorded on master 001526b (61 tool calls:
  pile_group_simple 10, pile_group_6dof 12, group_efficiency 10, Meyerhof 6,
  pile_group_package 23; 94 calc-step digests), run again on the branch: 60
  calls identical, one differs only in a refusal's list of valid parameters
  (it now names cap_weight, layers, pile_length and unit_weight); all 94
  calc-step digests found. None of the recorded calls had Vz < 0.
- 157 made-up calls with no net tension (72 simple and 72 6-DOF, 74 of them
  with piles in tension under a moment; 8 packages; 4 group_efficiency; 1
  Meyerhof): byte-identical outputs on master and the branch.

## Review fix (gm287-review): the trigger

- **Finding.** GEC-12 Vol III applies the block to the back row in tension
  of a group in net compression (Table D-13: Q 2,505 kips, the back row
  −406 kips; "only a single row of piles is providing the tension
  resistance rather than the entire pile group", pp. 51, 203); in Table D-19
  the block governs 10 of the 15 North Abutment lengths. The build checked
  only Vz < 0, and the package wrote "does not apply" for such a group.
- **Now.** As above: the rigid-cap tools note any pile in tension; the
  package checks the block over the piles in tension (their extent, their
  tension, their capacities, the cap's share), says so where they do not
  fill a rectangle, keeps the whole-group check in net tension, and counts
  all the piles under the cap for the small-group rule.
- **Replay.** The recorded test-set calls on master 001526b against the
  fix: 52 of 61 tool calls identical; 8 differ only by an added
  `group_uplift_note`, each a group with piles in tension (Mx or My 0.5
  kN·m at Vz 0: 0.1 to 0.2 kN a pile); 1 the refusal's parameter list
  (above). Calc steps: 88 of 94 records identical, 6 identical but for an
  added Group Uplift section (the same groups). The 157 made-up calls: 81
  identical, none with a pile in tension; 76 differ only by added uplift
  keys, each with piles in tension (74 rigid-cap calls and 2 packages at Vz
  0, My 400). No group without a pile in tension moved.

## Noted, not fixed

- A pile in tension by a fraction of a kN (a moment on an unloaded group)
  gets the note and the section like any other.
- Battered piles: the package takes the block at the toes from the piles'
  positions at the cap.
- One soil type per block: GEC-12 and UFC give the sand block and the clay
  block, not a mixed profile.
- Not offered: UFC p. 434's resistance factor 0.8 for uplift at the extreme
  limit state, and the factors of safety for tension (UFC Table 6-29).
- Print slips: GEC-12 Table D-17 (above); GEC-12 p. 291 gives R_r of Eq.
  7-39 as "(Eq. 7-4)", the shaft resistance equation (7-3 is the factored
  resistance); Sec. 7.2.3.2.2 calls Eq. 7-40's phi_up R_s n a "nominal
  group resistance", where the equation makes it R_r.
