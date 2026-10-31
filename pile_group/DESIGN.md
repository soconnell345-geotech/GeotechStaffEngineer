# pile_group — Rigid Cap Pile Group Analysis

## Purpose
Analyzes pile groups under combined loading (6-DOF) with rigid cap
assumption. Includes group efficiency and p-multiplier corrections.

## References
- FHWA GEC-12 (FHWA-NHI-16-009) Vol. I ch. 7: group efficiency (Sec.
  7.2.2, Eq. 7-35), block failure (Eqs. 7-36, 7-37, Fig. 7-31), group
  settlement (Eqs. 7-50 to 7-52; Sec. 7.3.5.3), p-multipliers (Sec.
  7.3.7.6, AASHTO as printed pp. 367 and 370). Vol. III Appendix D works
  the North Abutment, Pier 2 and South Abutment (V-005, V-060).
- UFC 3-220-20 (2025) ch. 6: Eq. 6-37, Table 6-26 (group efficiency
  factors by pile type), Eqs. 6-38, 6-39 (block failure), Sec. 6-5.8.4
  (equivalent footing, to 10 % of the bearing stress; App. B-5.7, as
  GEC-12 step 1c, to 10 % of the existing vertical effective stress), Table 6-38
  (AASHTO extended to 6b; Mokwa 1999). Appendix B-5.4 and Table B-32 (V-060).
- USACE EM 1110-2-2906 and CPGA (ITL-89-4): the rigid-cap 6-DOF method.
  Neither is held, and no held document prints a 6-DOF (battered) group
  benchmark: the 6-DOF method is pinned only on GEC-12 Vol. III's
  vertical-pile case with equal springs (V-060).
- Not offered: a group uplift method (GEC-12 Sec. 7.2.3.2, Eqs. 7-39 to
  7-41). Removed in GM-286: Converse-Labarre, printed in no held document.

## Files
- `pile_layout.py` — GroupPile dataclass, rectangular_layout() generator
- `group_efficiency.py` — group_efficiency_factor() (the printed eta_g
  rows), block_failure_resistance() / block_failure_capacity(),
  p_multiplier() (AASHTO, Mokwa), group_settlement_equivalent_raft(),
  meyerhof_group_settlement()
- `rigid_cap.py` — simplified elastic + 6-DOF stiffness matrix
- `results.py` — PileGroupResult with summary()/to_dict()

## Public API
```python
analyze_pile_group(piles, loads, soil_params, ...) -> PileGroupResult
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
  any right-handed frame together with the +My term).
- 6-DOF: K = sum(B^T k B) with rigid-cap kinematics
  `ux = dx - rz*y`, `uy = dy + rz*x`, `s = dz - rx*y + ry*x`; the same B
  back-calculates pile forces, so they equilibrate the applied loads.
  In-plane kxy and lateral-torsion couplings are assembled; the battered
  kxz/kyz force<->rotation couplings remain out of scope (PG-1 limitation,
  documented in the docstring).

## Group efficiency (GM-286)
- eta_g is read from the printed rows, linear between the printed spacings;
  where an input that picks the row (cap contact, su against 2 ksf) is
  left out, the lower row is taken and the other given as the alternative.
  Spacing below the row's printed minimum is refused.
- GEC-12 Sec. 7.2.2.1 prints no eta_g for jetted or predrilled piles in
  cohesionless soil, and over a weak deposit takes the lesser of the sum
  and block failure; the module adds no number of its own for either.

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

## Failure modes / guards (PG-1, PG-3, v5.0-5.1)
- `analyze_group_6dof` raises `ValueError` if a loaded DOF has no stiffness
  (e.g. lateral load on vertical piles without `lateral_stiffness`);
  unloaded unsupported DOFs are statically condensed.
- `analyze_vertical_group_simple` warns (`UserWarning`) if called with
  nonzero Vx/Vy/Mz, which it ignores by design.
