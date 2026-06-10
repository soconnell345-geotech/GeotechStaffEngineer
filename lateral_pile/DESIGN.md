# lateral_pile — Lateral Pile Analysis (COM624P)

## Purpose
Finite-difference beam-on-nonlinear-foundation solver for laterally loaded
piles. Implements 8 p-y curve models and matches COM624P published results.
Supports an above-ground free (stickup) length for pile-bent / column
applications.

## References
- Matlock (1970) — soft clay p-y curves
- Reese et al. (1974) — sand p-y curves
- API RP2A (2000) — sand simplified p-y
- Reese (1975) — stiff clay below/above water table
- Jeanjean (2009) — soft clay updated model
- Rollins et al. (2005) — liquefied sand p-y curves
- COM624P User Manual (Wang & Reese 1993)

## Files
- `pile.py` — Pile dataclass (solid/pipe, variable EI sections)
- `soil.py` — SoilLayer with p-y model interface, layer validation
- `py_curves.py` — 8 models: Matlock, Jeanjean, StiffClayBelowWT, StiffClayAboveWT, SandReese, SandAPI, WeakRock, SandLiquefied
- `solver.py` — FD beam-column solver (banded scipy.linalg.solve_banded)
- `analysis.py` — LateralPileAnalysis orchestrator class
- `results.py` — Results container with matplotlib plotting
- `validation.py` — benchmark pytest suite (Hetenyi, COM624P, equilibrium)
- `tests/` — regression tests (stickup feature, banded-vs-dense solve)

## Public API
```python
analysis = LateralPileAnalysis(pile, soil_layers)
results = analysis.solve(Vt=100, Mt=0, Q=0, n_elements=100, stickup=0.0)
results.y_top     # deflection at the loaded head (top of stickup)
results.y_ground  # deflection at the ground surface
results.summary()
```

## Critical Technical Notes
- **FD Solver**: MUST use full (n+5) system with explicit fictitious nodes.
  A substitution-based (n+1) approach caused sign errors in BCs.
- **Banded solve**: the (n+5) system is pentadiagonal except the head/tip
  shear BC rows, which widen the band to |i-j| <= 4. Solved with
  `scipy.linalg.solve_banded((4,4), ...)` — O(n), numerically identical to
  the previous dense `np.linalg.solve` (regression-tested to machine
  precision in `tests/test_stickup_and_banded.py`).
- **Stickup (above-ground free length)**: `solve(..., stickup=e)` extends
  the FD mesh above z=0 with zero soil resistance (p = 0) over the stickup;
  head load/BCs act at the top of the stickup. `pile.length` remains the
  EMBEDDED length; node depths above grade are negative. At the node that
  lands exactly on the grade discontinuity, Es is averaged (Es/2) to keep
  the scheme second-order; with stickup=0 behavior is byte-identical to the
  pre-stickup solver. Validated against the closed-form equivalence: for a
  free head with Q=0, stickup e == applying M = Mt + Vt*e at grade for the
  embedded response (matches to machine precision at equal h).
  No t-z/Q-z axial springs are modeled (out of scope); with Q != 0 the
  stickup mesh also captures the P-delta moment over the free length.
- **Sign convention**: M = EI*y'', V = EI*y''' + Q*y'. Positive Vt -> positive y.
- **Moment equilibrium**: integral(p*z*dz) = -Mt (soil reaction opposes applied moment)
- **Matlock cyclic**: Only engages when deflections exceed 3*y50
- **API sand**: tanh model is linear at small loads
- **numpy >=2.0**: np.trapz renamed to np.trapezoid — use try/except

## Validation
- COM624P Example 1 (Sabine River): 0.3% match to published 34.3mm
- Hetenyi closed-form: <2% error
- Force/moment equilibrium: <0.01% error
- validation.py (97 tests) + tests/ (12 regression tests)
