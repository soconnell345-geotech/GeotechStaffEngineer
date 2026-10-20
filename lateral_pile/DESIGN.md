# lateral_pile — Lateral Pile Analysis (COM624P)

## Purpose
Finite-difference beam-on-nonlinear-foundation solver for laterally loaded
piles. Implements 8 p-y curve models and matches COM624P published results.
Supports an above-ground free (stickup) length for pile-bent / column
applications.

## References
- Matlock (1970) — soft clay p-y curves, static and cyclic, as printed in
  the COM624P manual (FHWA-SA-91-048) pp. 323-327 and GEC 9 App. A.1,
  pp. 216-219
- Reese et al. (1974) — sand p-y curves
- API RP2A (2000) — sand simplified p-y
- Reese, Cox & Koop (1975) — stiff clay below the water table, static and
  cyclic, as printed in the COM624P manual pp. 328-335 and GEC 9 App. A.2,
  pp. 219-224
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
- **No solution, or an iteration that stopped (GM-208, 2026-10-19)**: the
  secant (p/y) iteration runs to `max_iterations` (default 500; it was 100,
  and softening curves took 120-140 near their peak; a run that converged
  within 100 is unchanged). When it stops short, or "converges" only after
  running away (deflections beyond the pile's length, where every secant
  modulus is too small to change the solve: on a 0.9 m, 15 m shaft in
  cyclic stiff clay at 900 kN it reached 1.5e10 mm at iteration 643), the
  head is pushed (displacement control, `solver._head_load_capacity`): at
  head deflections from 0.001 b to the larger of 10 b and the length, the
  load factor that holds each is iterated, and the peak refined. Below 1,
  there is no solution and `SoilResistanceExceeded` (a ValueError) says
  so plainly, with the most the pile and soil carry and the deflection
  there; the tool returns it as its error. At 1 or more, the result comes
  back with `converged` False and a `message` saying the iteration
  stopped, not the soil. The push takes about 3 s, on these paths only.
  Load control cannot pass the peak of a softening system; the push can.
- **Sign convention**: M = EI*y'', V = EI*y''' + Q*y'. Positive Vt -> positive y.
- **Moment equilibrium**: integral(p*z*dz) = -Mt (soil reaction opposes applied moment)
- **Matlock cyclic (GM-208, 2026-10-19)**: as printed in COM624P
  pp. 326-327 (Eq. 3.13, Fig. 3.5b) and GEC 9 Eq. A-6: the static curve
  while p is below 0.72 pu (reached at 2.99 y50); past 3 y50, at x >= x_r
  p = 0.72 pu, and above x_r p falls on a straight line from 0.72 pu at
  3 y50 to 0.72 pu (x/x_r) at 15 y50 and stays there. Before GM-208 it
  dropped straight to 0.72 pu x/x_r at 3 y50 (a step, softer than printed
  between 3 and 15 y50). Neither manual prints a cyclic soft-clay example;
  the curve is pinned to a HAND CALCULATION (c 20 kPa, gamma' 8, eps50
  0.02, b 0.6 m, x 2 m: pu 65.6 kN/m, x_r 4.865 m; 47.23 kN/m at 3 y50,
  33.32 at 9 y50, 19.42 from 15 y50), `tests/test_matlock_cyclic.py`.
- **Stiff clay below the water table (`StiffClayBelowWT`, GM-208,
  2026-10-19)**: Reese, Cox & Koop (1975) as printed in COM624P pp. 328-335
  (Eqs. 3.14-3.26, Figs. 3.6-3.8, Tables 3.2-3.3) and GEC 9 App. A.2
  (Eqs. A-7 to A-18). pc the smaller of 2 ca b + gamma' b x + 2.83 ca x and
  11 c b (ca taken as the layer's c); y50 = eps50 b (Eq. 3.17; NOT the
  2.5 eps50 b of the other clay criteria); the initial line p = (k x) y.
  Static: 0.5 pc (y/y50)^0.5 to As y50; minus 0.055 pc ((y - As y50)/
  (As y50))^1.25 to 6 As y50; a line of slope -0.0625 pc/y50 to 18 As y50;
  the residual pc (0.5 (6 As)^0.5 - 0.411 - 0.75 As) beyond. Cyclic, yp =
  4.1 Ac y50: Ac pc [1 - |(y - 0.45 yp)/(0.45 yp)|^2.5] to 0.6 yp; a line
  of slope -0.085 pc/y50 to 1.8 yp; 0.936 Ac pc - 0.102 pc yp/y50 beyond.
  The cyclic method has no number of cycles. GEC 9 p. 219 limits this
  criterion to stiff clay "exposed to free water at or near the ground
  surface"; with the groundwater at depth, "the Welch and Reese criterion
  is more appropriate, even below groundwater", and stiff clay below a
  sand stratum "would normally not be subject to degradation due to free
  water (unless scour removed the overlying sand)". `SoilProfile` follows
  that rule and states it (GM-215); it takes a clay as stiff from
  `stiff_clay_cu`, 100 kPa by default, its own value: GEC 9 prints soft
  clays as 250-500 psf (p. 216) and stiff as 1,000-2,000 psf (p. 219) and
  no boundary (GM-217, `geotech_common/DESIGN.md`). The initial line holds until
  it meets the curve and the smaller p holds where they do not meet
  (COM624P notes, pp. 333, 335). Where As < 0.223 (x/b below about 0.08)
  the static residual is negative; p is held at zero.
  - **k**: `ks` for static loading, `kc` for cyclic (kN/m3), each required
    with its loading and refused with the other. Table 3.2 (A-1) by the
    average undrained shear strength to five diameters, 48-96 / 96-192 /
    192-383 kPa (0.5-1 / 1-2 / 2-4 ton/ft^2): ks 135,700 / 271,400 /
    542,900 and kc 54,300 / 108,600 / 217,200 kN/m3 (500 / 1,000 / 2,000
    and 200 / 400 / 800 lb/in^3, 1 lb/in^3 = 271.447 kN/m3). Table 3.3:
    eps50 0.007 / 0.005 / 0.004. The k in force is reported in `choices`.
  - **As and Ac** were read off COM624P Fig. 3.7 (p. 330) and its redraw,
    GEC 9 Fig. A-6 (p. 224): the COM624P scan rendered at 300 dpi and
    mapped through its four frame corners (the page is skewed 1.4 degrees;
    the A ticks read back within 0.003); the GEC 9 raster mapped through
    its grid lines; each line's centre taken across rows where it is steep
    and down columns where it is flat, away from the test-data marks. The
    prints agree within 0.013; the table (x/b 0 to 4.5) is their mean
    rounded to 0.005, with the deep As 0.60 as COM624P prints it (GEC 9
    redraw 0.593). Both start at 0.2; As reaches 0.60 near x/b 4.5, Ac
    0.305 near 1.75; both held constant below (dashed to x/b 12 on the
    figure). Before GM-208 the table ran 2.5 to 0.88 (As) and 0.2 to 0.55
    (Ac), neither from the figure.
  - **Where the manuals differ**: GEC 9 prints A-15 as yp = 4.1 As y50;
    COM624P Eq. 3.23 and Fig. 3.8 give 4.1 Ac y50 (used). GEC 9's static
    step 9 discards the initial line when it meets A-11 beyond As y50;
    COM624P's note keeps the smaller p (used). They differ only near the
    ground (for a 0.9 m shaft, c 100 kPa, ks 271,400: the top 0.2 m).
  - **Anchors**: no printed example in either manual uses this criterion
    (COM624P Examples 1-6, GEC 9 App. B), so the curves are pinned to a
    HAND CALCULATION: c 100 kPa, gamma' 9, eps50 0.005, b 0.6 m, x 1.5 m
    (As 0.565, Ac 0.305): pc 552.6 kN/m, y50 3 mm; static (ks 271,400)
    207.68 kN/m at As y50, 293.30 at 6 mm, 168.44 at 20 mm, 47.44 beyond
    18 As y50; cyclic (kc 108,600) 168.54 at 0.45 yp, 114.71 at 5 mm,
    87.27 beyond 1.8 yp. `tests/test_stiff_clay_below_wt.py`.
  - Before GM-208: a line, a square-root parabola and a plateau at A pu
    with y50 = 2.5 eps50 b, and 'cyclic' took ks. On a 0.9 m, 15 m shaft
    (E 25 GPa, free head) in c 100 kPa, the head deflection at 600 kN went
    from 13.7 to 12.1 mm static and 19.4 to 63.0 mm cyclic (kc 108,600).
    The softening curves need more than the old default of 100 secant
    iterations at larger loads (static 900 kN: 133; cyclic 600 kN: 126);
    the default is 500 since GM-208. Cyclic at 900 kN has no solution:
    the pile and soil carry about 890 kN (see "No solution" above).
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
  - The soft-clay case in `validation.py` labelled "COM624P Example 1"
    until GM-209 is not the manual's Example 1 (which is the 14HP89
    above, and static), and no printed source for its 34.3 mm was found
    (the COM624P manual, Matlock 1970 OTC 1204, GEC 9 and other
    lateral-pile literature searched): it is now
    `test_soft_clay_regression_case_source_unknown`, pinned to the
    module's own 34.4 mm, and `test_com624p_example_1` checks the real
    Example 1 (p. 126) beside it.
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
- Stiff clay below the water table and Matlock cyclic (GM-208): hand
  calculations from the printed equations, no printed example
  (`tests/test_stiff_clay_below_wt.py`, `tests/test_matlock_cyclic.py`)
- COM624P Example 1 (printed pp. 101-126, 14HP89 in stiff clay above the
  water table): p-y curves to 0.002 %, head deflection and moment within
  0.2 % (`tests/test_stiff_clay_above_wt.py`)
- `validation.py`'s soft-clay case is a regression case, source unknown
  (GM-209): not a published check (see above)
- Hetenyi closed-form: <2% error
- Force/moment equilibrium: <0.01% error
- validation.py (97 tests) + tests/ (12 regression tests) + composite-section
  hand-check tests (tests/test_composite_section.py)
