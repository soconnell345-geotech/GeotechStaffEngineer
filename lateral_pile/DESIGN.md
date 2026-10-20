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
- Reese et al. (1975) — stiff clay below the water table
- Welch and Reese (1972) — stiff clay above the water table, static and
  cyclic, as printed in the COM624P manual (FHWA-SA-91-048) pp. 336-340 and
  GEC 9 App. A.3, pp. 224-226
- Jeanjean (2009) — soft clay updated model
- Rollins et al. (2005) — liquefied sand p-y curves
- Reese (1997) — weak rock p-y curves, as printed in FHWA GEC 9
  (FHWA-HIF-18-031, 2018) Appendix A.5, pp. 230-232, and GEC 10
  (FHWA-NHI-18-024, 2018) Section 9.3.3.4.4, pp. 9-31 to 9-33
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
- **Matlock cyclic**: Only engages when deflections exceed 3*y50. Not as
  printed above z_r (found in the GM-202 sweep, not fixed): past 3 y50 the
  code drops straight to 0.72 pu z/z_r, where COM624P Fig. 3.5b and GEC 9
  App. A.1 (Eq. A-6) fall linearly from 0.72 pu at 3 y50 to 0.72 pu z/z_r
  at 15 y50 (softer than printed between 3 and 15 y50).
- **Stiff clay below the water table**: the static and cyclic curves are a
  simplified construction (linear, a square-root parabola, a plateau at
  A pu), not the printed five- and four-segment curves of Reese et al.
  (1975) in COM624P pp. 328-335 and GEC 9 App. A.2 (Eqs. A-7 to A-18);
  found in the GM-202 sweep, not fixed. Cyclic is not ignored.
- **Stiff clay above the water table (`StiffClayAboveWT`, GM-202,
  2026-10-19)**: Welch & Reese (1972) as printed in COM624P pp. 336-340
  (Eqs. 3.27-3.29, Figs. 3.9-3.10) and GEC 9 App. A.3 (Eqs. A-19 to A-21).
  pu the smaller of Eqs. 3.8/3.9 (J = 0.5, total unit weight above the
  water table), y50 = 2.5 eps50 b (Eq. 3.10). Static: p/pu = 0.5
  (y/y50)^(1/4), p = pu beyond 16 y50. Cyclic, for N cycles of the design
  load: for each p/pu, C = 9.6 (p/pu)^4 (3.28) and y_c = y_s + y50 C log N
  (3.29), "log" read as the common logarithm. With y_s = 16 y50 (p/pu)^4
  from 3.27 this is y_c = y50 (p/pu)^4 (16 + 9.6 log N): the static curve
  with y50 times (1 + 0.6 log N), reaching pu at 16 y50 + 9.6 y50 log N as
  Fig. 3.10 labels it; N = 1 is the static curve. `n_cycles` (N) is
  required with loading='cyclic' and refused with 'static': neither
  manual gives a default N (COM624P step 2: "the number of times the
  design lateral load will be applied"), so the engineer states it; it is
  reported in `choices` with y50's factor. Before GM-202 'cyclic' gave the
  static curve (a cap at 16 y50 the static curve already has).
  - **Anchors**: the static curve reproduces COM624P Example 1 (printed
    pp. 101-126: a 14HP89, strong axis, in uniform stiff clay above the
    water table, c 14 psi, gamma 0.069 lb/in^3, eps50 0.007, static): the
    printed p-y curves at 20/60/100/150 in (pp. 121-122) to 0.002 %, and
    the head deflection and moment at 20 and 40 kips (p. 126: 0.145 in,
    661 in-kip; 0.623 in, 1,740 in-kip) within 0.2 %. Neither COM624P nor
    GEC 9 prints a cyclic example, so the cyclic curve is a HAND
    CALCULATION from the printed equations: Example 1's clay at 60 in,
    N = 100, pu = 1,098.258 lb/in, y50 = 0.25725 in; at p/pu = 0.5,
    y_s = 0.25725 in, C = 0.6, y_c = 0.25725 + 0.25725 x 0.6 x 2 =
    0.56595 in, where the cyclic curve carries 549.13 lb/in and the static
    668.77 lb/in; and the plateau starts at 16 y50 + 9.6 y50 log N (Fig.
    3.10). `tests/test_stiff_clay_above_wt.py`.
  - The validation test named `test_com624p_example1_sabine_river` is not
    the manual's Example 1 (which is the 14HP89 above, and static); where
    its 34.3 mm comes from is unverified.
- **API sand**: tanh model is linear at small loads
- **SandReese construction (v5.3)**: `SandReese(construction=...)` selects the
  sand p-y curve build. `"simplified"` (**default**, byte-identical to prior):
  initial linear (k*z*y) + a 1/3-power softening parabola anchored at the ultimate
  point (yu=3b/80) + plateau, with A = max(0.9, 3-0.8 z/b). `"reese1974"`: the FULL
  four-segment Reese (1974) curve — initial linear -> parabola (p=C*y^(1/n)) ->
  straight m-segment (pm..pu_curve) -> plateau — with the A/B chart coefficients
  (`_REESE_A_*`/`_REESE_B_*`, RE-DIGITIZED 2026-07-18 from COM624P FHWA-SA-91-048
  Figures 3.12/3.13, printed pp. 344-345, source PDF in the owner's library;
  printed asymptotes A=0.88 at x/b>5 — ONE annotation, static and cyclic merge at
  depth — and B_s=0.50 / B_c=0.55), the m-point pm=B*pu at ym=b/60, and
  pu_curve=A*pu at yu=3b/80. NOTE: the full construction is SOFTER than the
  simplified at the working deflection, so it is not a drop-in "stiffer"
  replacement — see the V-017 note in `validation_examples/RESULTS.md`.
  - **Source basis (Reese A/B tables) — VERIFIED-PRIMARY 2026-07-18** (wiki
    verification, `module_work/wiki_verification/reese_sand_com624p.md`): all
    four tables now carry page-calibrated marker read-offs of COM624P Figs
    3.12/3.13 (tolerance ±0.05 nominal / 0.10 outer; printed asymptote
    annotations exact). This RETIRED the pre-2026-07-18 cyclic tables, which
    were v5.3 reconstructions descending 2.9 -> 0.55 premised on "A_c = 0.55 at
    depth" — the manual shows 0.55 is **B_c's** asymptote; the real A_c is a low
    hump (~0.75 -> ~1.08 at x/b 1.5 -> 0.88 deep) and A >= B holds everywhere in
    the printed charts as-is. The old "Figs 2.19/2.20 / Table 2.2" citations
    were stale (this manual's charts are Figs 3.12/3.13; k tables are Tables
    3.4/3.5). k-table submerged medium/dense corrected to the printed anchors
    (60/125 lb/in^3); the phi=40 k points are extrapolations beyond the manual
    (flagged non-primary). The earlier audit RED-FLAG on chart ordinates is
    CLOSED.
- **Weak rock (`WeakRock`, GM-159, 2026-10-18)**: Reese (1997) as printed
  in GEC 9 App. A.5 (Eqs. A-33 to A-42). p_ur = alpha_r qu b (1 + 1.4 z_r/b)
  to z_r = 3b, 5.2 alpha_r qu b below; alpha_r = 1 - (2/3) RQD/100 (1/3 at
  RQD 100 %, 1.0 at RQD 0); K_ir = k_ir E_m with k_ir = 100 + 400 z_r/(3b)
  to 3b, 500 below; p = K_ir y to y_A, then (p_ur/2)(y/y_rm)^0.25 to p_ur,
  y_rm = k_rm b. z_r is the depth below the ROCK SURFACE: `rock_top`, which
  `LateralPileAnalysis` fills (on a copy) with the top of the run of
  weak-rock layers holding the layer. Defaults: RQD 100 (the printed lower
  bound alpha_r = 1/3), k_rm 0.0005 (the softer end of 0.00005-0.0005);
  both reported in `choices`. GEC 9 prints A-38 with a lower-case k_ir and
  A-41 as y_m; Fig. A-12, A-35 and A-42 make them K_ir and y_rm. `gamma_r`
  is accepted and not used (no unit weight in the criterion); the criterion
  is static only (GEC 9 p. 92), so 'cyclic' gives the static curve and says
  so. Before GM-159 alpha_r was fixed at 1.0 (p_ur three times the printed
  value at RQD 100), depth was measured from the ground, and the curve was
  0.25 E_m y to p_ur (an initial slope of 0.25 E_m against the printed
  100-500 E_m).
  Checked against GEC 10 Fig. 9-31 (p. 9-33) in `tests/test_weak_rock.py`.
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
- COM624P Example 1 (printed pp. 101-126, 14HP89 in stiff clay above the
  water table): p-y curves to 0.002 %, head deflection and moment within
  0.2 % (`tests/test_stiff_clay_above_wt.py`)
- "COM624P Example 1 (Sabine River)" in `validation.py`: 0.3% match to
  34.3mm, a source that is not the manual's Example 1 (see above)
- Hetenyi closed-form: <2% error
- Force/moment equilibrium: <0.01% error
- validation.py (97 tests) + tests/ (12 regression tests) + composite-section
  hand-check tests (tests/test_composite_section.py)
