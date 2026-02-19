# lateral_pile — Lateral Pile Analysis (COM624P)

## Purpose
Finite-difference beam-on-nonlinear-foundation solver for laterally loaded
piles. Implements 8 p-y curve models and matches COM624P published results.

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
- `solver.py` — FD beam-column solver
- `analysis.py` — LateralPileAnalysis orchestrator class
- `results.py` — Results container with matplotlib plotting
- `validation.py` — 58 pytest tests across 8 test classes

## Public API
```python
analysis = LateralPileAnalysis(pile, soil_layers)
results = analysis.analyze(Vt=100, Mt=0, Qa=0, n=100)
results.y[0]  # top deflection
results.summary()
```

## Critical Technical Notes
- **FD Solver**: MUST use full (n+5) system with explicit fictitious nodes.
  A substitution-based (n+1) approach caused sign errors in BCs.
- **Sign convention**: M = EI*y'', V = EI*y''' + Q*y'. Positive Vt -> positive y.
- **Moment equilibrium**: integral(p*z*dz) = -Mt (soil reaction opposes applied moment)
- **Matlock cyclic**: Only engages when deflections exceed 3*y50
- **API sand**: tanh model is linear at small loads
- **numpy >=2.0**: np.trapz renamed to np.trapezoid — use try/except

## Validation
- COM624P Example 1 (Sabine River): 0.3% match to published 34.3mm
- Hetenyi closed-form: <2% error
- Force/moment equilibrium: <0.01% error
- 66 tests (in validation.py, not tests/ subfolder)
