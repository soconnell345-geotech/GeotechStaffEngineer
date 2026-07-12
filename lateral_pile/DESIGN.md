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
- **SandReese construction (v5.3)**: `SandReese(construction=...)` selects the
  sand p-y curve build. `"simplified"` (**default**, byte-identical to prior):
  initial linear (k*z*y) + a 1/3-power softening parabola anchored at the ultimate
  point (yu=3b/80) + plateau, with A = max(0.9, 3-0.8 z/b). `"reese1974"`: the FULL
  four-segment Reese (1974) curve — initial linear -> parabola (p=C*y^(1/n)) ->
  straight m-segment (pm..pu_curve) -> plateau — with the A/B chart coefficients
  (`_REESE_A_*`/`_REESE_B_*`, digitized from Reese & Van Impe 2001 Figs 3.30/3.31 /
  COM624P; asymptotes A_s=0.88, A_c=0.55, B_s=0.50, B_c=0.55), the m-point
  pm=B*pu at ym=b/60, and pu_curve=A*pu at yu=3b/80. NOTE: the full construction is
  SOFTER than the simplified at the working deflection (the m-point pm=B*pu with
  B->0.5 is genuinely soft), so it is not a drop-in "stiffer" replacement — see the
  V-017 note in `validation_examples/RESULTS.md` (the V-017 deflection gap is the
  composite/nonlinear section EI, not the p-y curve).
  - **Source basis (Reese A/B tables, `_REESE_A_*`/`_REESE_B_*` in `py_curves.py`):**
    transcription/digitization of the A and B coefficient charts. Authoring-time
    source-consultation status (Reese & Van Impe 2001 / COM624P in hand vs
    reconstructed) was not recorded. Values were re-digitized and corrected in the
    v5.3 review (the cyclic `_REESE_A_CYCLIC` had dipped below `_REESE_B_CYCLIC`,
    silently degenerating the four-segment curve to linear-plateau; corrected to
    descend 2.9 -> 0.55 staying >= B_c) and are constrained by the physical A>=B
    monotonicity and anchored by the p-y regression tests + V-017. RED-FLAG note:
    the anchors verify curve SHAPE/behavior, not the exact chart ordinates, so the
    ordinates remain a **candidate for verification against the owner's reference
    wiki** (Reese, Cox & Koop 1974 OTC 2080 originals; Reese & Van Impe 2001
    Figs 3.30/3.31; COM624P FHWA-SA-91-048 Figs 2.19/2.20).
- **numpy >=2.0**: np.trapz renamed to np.trapezoid — use try/except
- **Composite / transformed-section EI (`composite_section.py`, v5.4 E5)**:
  `composite_section_ei(section_type, ...)` returns a `CompositeSection` with the
  UNCRACKED transformed-section `EI` (kN·m²), `EA` (kN), transformed area/inertia
  and a `summary()`. Three cases: `filled_pipe` (concrete/grout-filled steel
  pipe), `cased_concrete` (steel casing + grout core + optional circular bar
  ring), `reinforced_concrete` (circular or rectangular RC). Method: EI =
  Σ E_i·I_i about the composite neutral axis; a steel pipe/casing and the
  concrete core it confines are summed directly (non-overlapping), while bars
  embedded in concrete are added at the NET modulus (E_bar − E_concrete) — the
  (n−1) transformed rule that avoids double-counting displaced concrete. Concrete
  modulus from `E_concrete` or `fc` via ACI 318 `Ec = 4700√f'c` (MPa). Feed the
  result into an analysis with `Pile.from_composite_section(length, section,
  diameter)`. **Basis is uncracked / gross** (upper-bound working stiffness) —
  cracked / moment-curvature (M-φ) EI is out of scope (for a moment-dependent
  cracked circular-RC EI use `ReinforcedConcreteSection` / `Pile.from_rc_section`,
  Branson's equation). Validates V-017: the composite EI upgrades the micropile
  head-deflection flag from CONVENTION to PASS (see validation_examples/RESULTS.md).

## Validation
- COM624P Example 1 (Sabine River): 0.3% match to published 34.3mm
- Hetenyi closed-form: <2% error
- Force/moment equilibrium: <0.01% error
- validation.py (97 tests) + tests/ (12 regression tests) + composite-section
  hand-check tests (tests/test_composite_section.py)
