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
  cap_weight, pile_type, piles_resistance)` (`group_efficiency.py`): the
  block by layers (prismoidal volume, exact for plan dimensions linear in
  depth; one layer is Eq. 6-41), its weight plus the cap's, plus 2D(B+Z)s_u
  in clay; phi_ug by pile type (`GROUP_UPLIFT_PHI`) with the small-group
  20 %; the lesser of phi_ug R_ug and the piles' sum, and which governs.
  Layers that stop short of the toes are refused; a layer past them is cut
  off; fewer than two piles are refused.
- Tool `group_uplift`: the layout (n_rows, n_cols, spacing, pile_diameter;
  B and Z to the outer faces), pile_length, soil_type, unit_weight or
  layers, cu (clay), cap_weight (0 when left out, the safe side), pile_type
  (driven), axial_capacity_tension (each pile's phi_up R_s) and uplift_load.
  It returns the block, phi_ug, the factored block, the piles' sum, the
  governing resistance (`group_uplift_resistance_kN`), which governs
  (`group_uplift_governs`) and the utilization. Without the piles'
  resistance it gives the block and says the group's is no more than that.
  n_rows 1 checks one row, as Vol III does.
- `pile_group_simple` and `pile_group_6dof`: a group in net tension (Vz <
  0) gets `group_uplift_note`, the net uplift and the check to run.
- `pile_group_package`: a group in net tension gets a Group Uplift section:
  the block at the toes from the piles' plan extent plus the diameter, its
  weight, R_ug, phi_ug R_ug, the piles' sum from their
  `axial_capacity_tension` (as the utilization takes it, each pile's
  factored resistance), the lesser and the check against the net uplift;
  the result carries the same keys as the tool. Without pile_length,
  soil_type and unit_weight (cu in clay) it says what it needs. A group not
  in net tension gets no section.

`choices` state each reading: the lesser and which governs, with UFC Eq.
6-40's piles' sum (the weights added) as the alternative; the block as
printed for the soil, B and Z to the outer faces (Vol III), the piles as
soil; phi_ug with its small-group rule (below 3 driven piles, GEC-12 p. 225;
below 5 for the others, UFC note A), and for 3 or 4 driven piles UFC's
reading (0.40) as the alternative; the cap weight (0 unless given).

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

## Replay: groups not in net tension

- The pile group tests and guards recorded on master 001526b (61 tool calls:
  pile_group_simple 10, pile_group_6dof 12, group_efficiency 10, Meyerhof 6,
  pile_group_package 23; 94 calc-step digests), run again on the branch: 60
  calls identical, one differs only in a refusal's list of valid parameters
  (it now names cap_weight, layers, pile_length and unit_weight); all 94
  calc-step digests found. None of the recorded calls had Vz < 0.
- 157 made-up calls with no net tension (72 simple and 72 6-DOF, 74 of them
  with piles in tension under a moment; 8 packages; 4 group_efficiency; 1
  Meyerhof): byte-identical outputs on master and the branch.

## Noted, not fixed

- Vol III checks the back row in tension of a group in net compression; the
  tool does it with n_rows 1, but the rigid-cap tools and the package note
  and check only a group in net tension (proposed to the backlog).
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
