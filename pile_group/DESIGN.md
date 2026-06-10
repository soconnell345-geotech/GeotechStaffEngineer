# pile_group — Rigid Cap Pile Group Analysis

## Purpose
Analyzes pile groups under combined loading (6-DOF) with rigid cap
assumption. Includes group efficiency and p-multiplier corrections.

## References
- Converse-Labarre formula for group efficiency
- FHWA GEC-12 (Driven Piles) — group effects
- Reese & Van Impe (2001) — p-multipliers

## Files
- `pile_layout.py` — GroupPile dataclass, rectangular_layout() generator
- `group_efficiency.py` — converse_labarre(), block_failure(), p_multiplier()
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

## Failure modes / guards (PG-1, PG-3, v5.0-5.1)
- `analyze_group_6dof` raises `ValueError` if a loaded DOF has no stiffness
  (e.g. lateral load on vertical piles without `lateral_stiffness`);
  unloaded unsupported DOFs are statically condensed.
- `analyze_vertical_group_simple` warns (`UserWarning`) if called with
  nonzero Vx/Vy/Mz, which it ignores by design.
