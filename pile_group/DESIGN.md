# pile_group — Rigid Cap Pile Group Analysis

## Purpose
Analyzes pile groups under combined loading (6-DOF) with rigid cap
assumption. Includes group efficiency and p-multiplier corrections.

## References
- FHWA GEC-12 (FHWA-NHI-16-009) Vol. I ch. 7: group efficiency (Sec.
  7.2.2, Eq. 7-35), block failure (Eqs. 7-36, 7-37, Fig. 7-31), group
  settlement (Meyerhof, Eqs. 7-50 to 7-52; the equivalent footing of Fig.
  7-44 and Eq. 7-55; a group in cohesive soil settles by consolidation,
  Sec. 7.3.5.3, not worked here), p-multipliers (Sec.
  7.3.7.6, AASHTO as printed pp. 367 and 370). Vol. III Appendix D works
  the North Abutment, Pier 2 and South Abutment (V-005, V-060).
- UFC 3-220-20 (2025) ch. 6: Eq. 6-37, Table 6-26 (group efficiency
  factors by pile type), Eqs. 6-38, 6-39 (block failure), Sec. 6-5.8.4
  (equivalent footing, Eq. 6-54 for coarse-grained soil, pp. 448-449,
  to 10 % of the bearing stress; App. B-5.7, as
  GEC-12 step 1c, to 10 % of the existing vertical effective stress), Table 6-38
  (AASHTO extended to 6b; Mokwa 1999). Appendix B-5.4 and Table B-32 (V-060).
- USACE EM 1110-2-2906 and CPGA (ITL-89-4): the rigid-cap 6-DOF method.
  Neither is held, and no held document prints a 6-DOF (battered) group
  benchmark: the 6-DOF method is pinned only on GEC-12 Vol. III's
  vertical-pile case with equal springs (V-060).
- Group uplift (GM-289): GEC-12 Sec. 7.2.3.2 (Eqs. 7-39 to 7-41, Figs.
  7-33 and 7-34, Table 7-1's phi_ug 0.50, p. 225's small group); UFC
  3-220-20 Eqs. 6-40 to 6-43 and Table 6-28. Vol. III Block 13 (North and
  South Abutments) reproduced (`module_work/gm289/CHECK.md`). Removed in
  GM-286: Converse-Labarre, printed in no held document.
- The small group in uplift (GM-310, the call for OWN-118 of 2026-11-12):
  the 20 % cut starts below 5 piles for every pile type (UFC 3-220-20 Table
  6-28 note A, AASHTO 2020). GEC-12 p. 225 (AASHTO 2014) reads it as fewer
  than 3 for driven piles, the driven default before GM-310; the two differ
  only for 3 or 4 driven piles (phi_ug 0.40 against 0.50), where the
  result's choices give GEC-12's value and the block at it.

## Files
- `pile_layout.py` — GroupPile dataclass, create_rectangular_layout() generator
- `group_efficiency.py` — group_efficiency_factor() (the printed eta_g
  rows), block_failure_resistance() / block_failure_capacity(),
  group_uplift_resistance(),
  p_multiplier() (AASHTO, Mokwa), group_settlement_equivalent_raft(),
  meyerhof_group_settlement()
- `rigid_cap.py` — simplified elastic + 6-DOF stiffness matrix
- (`GroupLoad` and `PileGroupResult` (summary()/to_dict()) live in `rigid_cap.py`; there is no `results.py`)

## Public API
```python
# exports of pile_group/__init__.py
GroupPile, create_rectangular_layout(n_rows, n_cols, spacing_x, spacing_y,
    axial_stiffness=None, lateral_stiffness=None) -> List[GroupPile]
GroupLoad, PileGroupResult
analyze_vertical_group_simple(piles: List[GroupPile], load: GroupLoad) -> PileGroupResult
analyze_group_6dof(piles: List[GroupPile], load: GroupLoad) -> PileGroupResult
group_efficiency_factor, block_failure_resistance, block_failure_capacity,
group_plan_dimensions, group_uplift_resistance, p_multiplier,
group_settlement_equivalent_raft, meyerhof_group_settlement,
BLOCK_FAILURE_PHI, GROUP_UPLIFT_PHI   # see group_efficiency.py for signatures
```

## Key Notes
- Piles defined by (x, y) positions in plan view
- 6-DOF: Vx, Vy, Vz (axial), Mx, My, Mz (torsion)
- SoilProfile adapter: to_pile_group_input() returns weighted avg phi/cu

## Sign Convention (PG-2, v5.1)
One explicit right-hand-rule convention end-to-end (simple method, 6-DOF
stiffness assembly, and pile-force back-calc):
- Right-handed global axes with **z UP**; x, y are plan coordinates.
- **Vz and dz are positive DOWNWARD** (compression / settlement) for
  engineering convenience; everything else is right-hand rule about
  +x, +y, +z(up).
- Consequences: **+My compresses the +x side; +Mx uplifts (tensions)
  the +y side; +Mz twists counterclockwise in plan (viewed from above).**
- Simple method: `P_i = Vz/n + My*x_i/Sx2 - Mx*y_i/Sy2` (note the minus on
  Mx — pre-v5.1 the code used `+ Mx*y_i/Sy2`, which is not realizable in
  any right-handed frame together with the +My term). GEC-12 Vol I
  Sec. 8.8 Step 3 prints it as Eq. 8-79 (p. 468), as the maximum pile
  load with the terms added (signs here by the module's convention), the
  vertical load including the cap's and the soil's weight (Wc, Ws: Vz
  here is all of it; the tool adds none), x and y measured there from the
  centre of the column, which carries the loads.
- The product term (GM-319 re-check, the lead's call, for the owner to
  revisit): Eq. 8-79 gives back Mx and My only where SUM dx dy = 0 about
  the centroid (a layout symmetric about one of its centroidal axes). A
  group that has lost a corner pile, or an L-shaped cap, is not: there the
  general rigid-cap form is used, P = Vz/n + a dx + b dy with
  a = (My Syy + Mx Sxy)/D, b = -(Mx Sxx + My Sxy)/D, D = Sxx Syy - Sxy^2
  (Sxx = SUM dx^2, Syy = SUM dy^2, Sxy = SUM dx dy; `simple_load_plane`).
  Eq. 8-79 is its case Sxy = 0, so every symmetric layout keeps its
  numbers; |Sxy| at or below 1e-9 of sqrt(Sxx Syy) is taken as zero
  (`PRODUCT_TOL`). Basis: statics, the equilibrium of a rigid cap on
  pinned piles of equal axial stiffness; it equals the 6-DOF method with
  equal ka. The result's notes and the calc package say when it is used.
  Before: a 2 x 3 group at 1.5 m less a corner, Vz 2000 kN, Mx and My
  300 kN-m, gave 409.5 / 481.0 / 552.4 / 242.9 / 314.3 kN (moments about
  the centroid My 450 and Mx 364); now 426.7 / 466.7 / 506.7 / 280.0 /
  320.0 kN. Piles on one line not along x or y carry the moment across
  the line only; one about the line is refused.
- Where the simple method takes the loads (GM-319 review): Vz, Mx and My
  are taken to act at the centroid of the piles (xc, yc), each pile's x
  and y measured from it; a load applied at another point (xa, ya) needs
  its moment about the centroid added, Vz x (xa - xc) to My and
  -Vz x (ya - yc) to Mx (+My compresses the +x side, +Mx uplifts the +y
  side). A centroid more than 1 mm from the origin of the coordinates is
  warned of, and the warning says `pile_group_6dof` takes its loads at the
  origin. Basis: statics of the rigid cap (with distances from the
  centroid SUM(x_i - xc) = 0, so the forces sum to Vz; with the product
  term they give back Mx and My); no held reference prints the distances from the piles' centroid,
  and Eq. 8-79's centre of the column is that point only when the column
  stands over it. Before, x and y were measured from the origin: piles at
  x = -1, 1, 3 m with Vz 900 kN and My 200 kN-m gave 281.8, 318.2 and
  354.5 kN (sum 954.5); now 250, 300 and 350 kN. A layout centred on the
  origin gives the same numbers as before. The result gives `centroid_m`,
  a note, and `warnings` when off the origin; with equal stiffness, dz is
  reported at the origin (the kinematics above), so the cap displacements
  equal the 6-DOF method's for the same loads moved to the origin; a note
  names the point dz is reported at and the settlement under the
  centroid.
- Where the 6-DOF method takes the loads (GM-319 re-check): Vx, Vy, Vz,
  Mx, My and Mz act at the origin of the coordinates (x = 0, y = 0), at
  the cap, the point the piles' x and y are measured from, not at the
  piles' centroid (its kinematics use x and y as given); a load applied
  at another point (xa, ya) needs its moment about the origin added,
  Vz x xa to My, -Vz x ya to Mx and xa x Vy - ya x Vx to Mz. The tool's
  description and loads say so, and its result's notes, naming the piles'
  centroid (before, nothing did: an off-centre layout with Vz 1800 kN,
  Mx 90 and My -120 kN-m gave a 341.7 kN maximum in the simple tool and
  1091.7 kN in the 6-DOF tool).
- 6-DOF: K = sum(B^T k B) with rigid-cap kinematics
  `ux = dx - rz*y`, `uy = dy + rz*x`, `s = dz - rx*y + ry*x`; the same B
  back-calculates pile forces, so they equilibrate the applied loads.
  Each pile is an axial spring ka along its axis and a lateral spring kl
  across it, pinned to the cap; in the global axes its head stiffness is
  k = kl*I + (ka - kl)*l*l^T, l the direction cosines (the pile's stiffness
  "transformed to global coordinates", EM 1110-2-2906 para. 4-5b(2)). Since
  GM-319 all of it is assembled, the kxz/kyz terms a batter gives included
  (before, they were left at zero and a battered pile's axial force did not
  balance the applied loads). With kl = 0 a statically determinate group
  returns the forces statics gives (UFC 3-220-20 Sec. 6-6.3: batter piles
  carrying axial load only). Still left out, and said in the result's
  `notes`: any moment or torsional stiffness of a pile's own head.
- Simple method (GM-319): equal vertical piles only. A battered pile or
  unequal `axial_stiffness` is refused. Cap dz, rx, ry are worked out only
  when every pile carries the same stiffness; otherwise
  `cap_displacements` is empty and `notes` says so (never zeros).
- Piles on one line (GM-319 review): a non-zero Mx with every pile on
  y = 0, or My with every pile on x = 0, is refused by both methods:
  piles pinned to the cap on one line carry no moment about it (before,
  the simple method skipped the term without a word). The simple method
  refuses it on any one line (a single row at y = 2 m cannot carry Mx
  either); the 6-DOF method, whose loads act at the origin, finds a single
  row off the origin singular and says so. A zero moment runs.
- Batter: the direction cosines are (sin batter_x, sin batter_y, the
  rest); sin²(batter_x) + sin²(batter_y) at or above 1 is refused (before,
  lz was set to 0 and the vector was not of unit length).

## Group efficiency (GM-286)
- eta_g is read from the printed rows, linear between the printed spacings;
  where an input that picks the row (cap contact, su against 2 ksf) is
  left out, the lower row is taken and the other given as the alternative.
  Spacing below the row's printed minimum is refused.
- GEC-12 Sec. 7.2.2.1 prints no eta_g for jetted or predrilled piles in
  cohesionless soil, and over a weak deposit takes the lesser of the sum
  and block failure; the module adds no number of its own for either.

## Group uplift (GM-289)
- The lesser of the piles' sum (GEC-12 Eq. 7-40, phi_up R_s n: each pile's
  factored tension resistance as given) and the block, phi_ug R_ug (Eq.
  7-39). The block: in cohesionless soil the effective weight of the soil
  within sides at 1H:4V from the toes (Fig. 7-33), by layers with the plan
  dimensions at each layer's bottom (UFC p. 436); in cohesive soil the
  vertical block's weight plus 2D(B+Z)su (Eq. 7-41). B and Z at the toes to
  the outer pile faces (Vol. III); the piles taken as soil (GEC-12 p. 291,
  UFC p. 435); the cap's effective weight added when given (0 otherwise).
- phi_ug: 0.50 driven (GEC-12 Table 7-1), 0.45 drilled shafts and 0.50
  micropiles (UFC Table 6-28), none printed for CFA or DDP (nominal only);
  times 0.8 below 3 driven piles (GEC-12 p. 225) or 5 of the others (UFC
  Table 6-28 note A). For 3 or 4 driven piles the choices give UFC's 0.40.
- UFC Eq. 6-40 adds each pile's and the cap's effective weight (x 0.9) to
  the piles' sum; offered in the choices, not taken.
- Where it applies (GM-289 review): wherever a pile carries tension. GEC-12
  Vol. III checks the back row in tension of an abutment group in
  compression as a block ("only a single row of piles is providing the
  tension resistance rather than the entire pile group", pp. 51, 203), and
  Ch. 7 the whole group in net tension. The rigid-cap tools note any pile
  in tension; the calc package checks the block over the piles in tension
  (their extent plus the diameter, their summed tension, their own
  capacities, the cap's weight in their share k/n, as Vol. III p. 207 takes
  the tributary area over them), over the outer row of them, the outer two
  rows and so on from the most-tensioned edge inward (review 2: the outer
  row's block can fail where the block over two rows passes; each set its
  own tension, capacities and share), and over the whole group in net
  tension (Vz < 0), given pile_length, soil_type and the unit weight; the
  highest utilization governs. Piles in
  tension that do not fill a rectangle are not checked as a block (said
  so). The small-group rule counts all the piles under the cap ("per
  substructure unit", p. 225; `group_piles`, the tool's `piles_under_cap`).
  The `group_uplift` tool runs it on a rectangular layout (n_rows 1 for a
  row in tension). A group with no pile in tension is unchanged.

## Readings where the print reads two ways (GM-286 review)
- Block failure: B and Z to the outer pile faces (GEC-12 Fig. 7-31). UFC
  App. B-5.4 takes them centre to centre of the outer piles (18 ft for its
  7x7 group of 1 ft piles at 3 ft); the tool gives that block and its
  resistance as the alternative in `choices`, and V-060 pins B-5.4 at its
  printed 18 ft.
- p-multiplier from 5b to 6b: a straight line to 1.0 at 6b, the module's
  reading of UFC p. 465's "AASHTO (2020) guidance, extrapolated to a pile
  spacing of 6·b". AASHTO's 3b-5b line carried on to 6b is the other
  reading (rows 3+ 0.9 at 6b); the tool's `choices` give its value between
  5b and 6b.
- Equivalent-footing settlement: summed to where the stress increase is
  below 10 % of the footing's bearing stress (UFC p. 449), the default, as
  the owner ruled for downdrag (OWN-62). GEC-12 step 1c (p. 324) and UFC
  App. B-5.7 (p. 641) take 10 % of the existing vertical effective stress;
  the function takes no effective-stress profile, so that depth is given
  as `influence_depth_m`, and the result's `choices` say so.
- Equivalent-footing settlement, the soil (GM-319 review): UFC Eq. 6-54
  is printed for the elastic compression of coarse-grained soil (pp.
  448-449). The tool says so; a group in cohesive soil is sent to
  consolidation (GEC-12 Sec. 7.3.5.3; `consolidation_settlement`). Poisson's
  ratio is accepted from 0 to below 0.5 (no printed basis for a lower
  limit); above 0.40, the top of GEC-5 Table 8-10's 0.2 to 0.4 for sands
  and gravels (p. 8-26), the result warns. With Es and nu given, the
  result's `choices` state M and M/Es. Spacing below the pile diameter
  and a `num_sublayers` that is not a whole number of 1 or more are
  refused.

## Failure modes / guards (PG-1, PG-3, v5.0-5.1)
- `analyze_group_6dof` raises `ValueError` if a loaded DOF has no stiffness
  (e.g. lateral load on vertical piles without `lateral_stiffness`, or a
  moment about the one line every pile is on), saying for each load why;
  unloaded unsupported DOFs are statically condensed.
- `analyze_vertical_group_simple` warns (`UserWarning`) if called with
  nonzero Vx/Vy/Mz, which it ignores by design (the agent's tool refuses
  them), and raises on a battered pile or unequal stiffness (GM-319) and
  on a moment about the one line every pile is on (GM-319 review).
