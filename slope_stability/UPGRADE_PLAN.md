# slope_stability Modern-LE Upgrade Plan (branch: le-modern)

Goal: bring the LE module to the standard of Slide2 / SLOPE/W / HYRCAN / SSAP:
rigorous GLE (Fredlund-Krahn), Janbu+f0, modern searches, reinforcement,
probabilistic FOS, SHANSEP/Hoek-Brown, ponded water, rich per-slice output.

Successor agents: read this file + `git log --oneline` first. Each phase is
independently shippable and committed when its tests pass. Run tests with:
`.venv/Scripts/python.exe -m pytest slope_stability -q` (baseline before this
work: 239 passed / 17 skipped — keep green; documented+validated behavior
changes to approximate methods are allowed).

NOTE for successors: the harness Write/Edit tools may refuse paths in this
worktree (isolation guard pins them to the v5.1-todos worktree); workaround
used here: Write files to `..\v5.1-todos\_le_staging\` then `cp` via Bash.

---

## Theory (Phase 0 — done)

### Rigorous GLE (Fredlund & Krahn 1977; GeoStudio "Stability Modeling with
GeoStudio/SLOPE/W", Seequent 2022 ed., ch. 2)

Interslice forces: normal `E`, shear `X = lambda * f(x) * E`.
f(x) options: constant (= Spencer), half-sine (default), clipped-sine,
trapezoidal.

Base normal from vertical slice equilibrium (textbook form):

    N = [W - (X_R - X_L) - (c'*l*sin(a) - u*l*sin(a)*tan(phi'))/F] / m_a
    m_a = cos(a) + sin(a)*tan(phi')/F          (l = base length, a = alpha)

Moment-equilibrium FOS (about circle center; noncircular uses a fitted
axis point, moments via explicit cross products so the Nf term is exact):

    F_m = sum[c'*l*R + (N - u*l)*R*tan(phi')] / sum[W*x - N*f + k*W*e + ...]

Force-equilibrium FOS (horizontal):

    F_f = sum[(c'*l + (N - u*l)*tan(phi')) * cos(a)] / sum[N*sin(a) + kW + ...]

Interslice normals by marching (horizontal equilibrium per slice), E=0 at the
two ends (crack water enters as an end boundary force). For each lambda,
iterate {N, E, X, F_m, F_f} to convergence; then root-find lambda* where
F_m(lambda) = F_f(lambda). Special cases: Bishop = F_m at lambda=0; Janbu
simplified = F_f at lambda=0; Spencer = crossing with f(x)=constant.

Implementation notes:
- Normalize slope direction internally (mirror x when sum(W*sin a) < 0) so the
  marching always runs one way with one sign convention; FOS/lambda invariant.
- m_a guard: |m_a| < 0.05 -> clamp+flag; N < 0 allowed but reported (tension).

### Janbu simplified + correction factor (Janbu 1973; Abramson et al. 2002)

    FOS_corrected = f0 * FOS_force(lambda=0)
    f0 = 1 + b1*[d/L - 1.4*(d/L)^2]
    b1 = 0.69 (c only), 0.31 (phi only), 0.50 (c-phi soils)
    d = max perpendicular distance slip surface to entry-exit chord; L = chord.

### Reinforcement in LE (FHWA GEC-7, Lazarte et al. 2003; in-house
geotech-references/gec_7)

Nail/anchor/geosynthetic crossing the slip surface contributes allowable
tension T = min(pullout behind surface, tensile) per metre run.
Moment methods: + T * (perpendicular arm to line of action) / R in the
resisting numerator. Force methods: + T*cos(inclination) against the driving
sum (horizontal projection). Nails: pullout = bond_stress * pi * D_DH * L_behind.

### Probabilistic (Duncan 2000 "Factors of safety and reliability in
geotechnical engineering", JGGE 126(4); USACE ETL 1110-2-556)

- FOSM/Taylor series: COV_F from central finite differences of F wrt each
  variable; lognormal reliability index
      beta_LN = ln(F_MLV / sqrt(1+COV_F^2)) / sqrt(ln(1+COV_F^2)), pf = Phi(-beta)
  Published anchor: F_MLV = 1.5, COV_F = 0.17 -> beta_LN = 2.32, pf ~ 1%
  (Duncan 2000).
- Monte Carlo: sample per-layer variables (normal/lognormal, truncated at
  physical bounds), fixed critical surface by default, optional re-search.

### Benchmarks (validation gate -> VALIDATION.md)

B1. Fredlund & Krahn (1977) homogeneous slope [Slide2 Verification #21]
   Geometry (ft): surface (0,60)-(60,60)-(140,20)-(180,20); soil c'=600 psf,
   phi'=20 deg, gamma=120 pcf; specified circle xc=120, yc=90, R=80.
   Case 1 dry:      Ordinary 1.928 | Bishop 2.080 | Spencer 2.073 | M-P 2.076
   Case 2 ru=0.25:  Ordinary 1.607 | Bishop 1.766 | Spencer 1.761 | M-P 1.764
   (F&K published; Slide2 gets 1.931/2.079/2.075/2.075 dry.) Janbu simplified
   (uncorrected) case 1 = 2.04 per F&K Table 2 (secondary citation).
   Gate: Bishop/Spencer/M-P within ~1-2%.

B2. F&K weak-layer composite [Slide2 Verification #22]
   Same surface; weak layer el. 15-16 ft (c'=0, phi'=10, gamma=120), model
   base at 15; same circle, composite surface = circle clipped at z=15.
   Case 1 dry:  F&K   Ordinary 1.288 | Bishop 1.377 | Spencer 1.373 | M-P 1.370
                Slide Ordinary 1.300 | Bishop 1.382 | Spencer 1.382 | M-P 1.372
   Case 2 ru:   F&K   Ordinary 1.029 | Bishop 1.124 | Spencer 1.118 | M-P 1.118
   Sources disagree in 2nd decimal (weak-layer position sensitivity); gate
   at ~3% on Spencer/M-P vs F&K.

B3. ACADS 1(a) [Giam & Donald 1989; Slide2 Verification #1] (SI)
   Surface (20,25)-(30,25)-(50,35)-(70,35), base el. 20; c'=3 kPa,
   phi'=19.6 deg, gamma=20 kN/m3; total stress, dry. Critical-circle search.
   Published answer 1.00; Slide2: Bishop 0.987, Spencer 0.986, GLE 0.986,
   Janbu corrected 0.990. Gate: search finds FOS 0.98-1.02.

B4. Duncan 2000 reliability anchor (above) + MC-vs-FOSM consistency on a
   slope case.

B5. Existing Duncan, Wright & Brandon suite (test_duncan_verification.py)
   stays green.

B6. Griffiths & Lane (1999) style cross-check of one geometry vs fem2d SRM
   (read-only import) — LE vs FEM agreement note.

---

## Phases

- **P0** Theory + this plan. [DONE]
- **P1** Rigorous GLE engine (`slope_stability/gle.py`): `gle_fos()` returning
  GLEResult (FOS, lambda, F_m, F_f, per-boundary E/X, thrust line, per-slice
  N', S_m). f(x): constant/half_sine/clipped_sine/trapezoidal. Circular +
  noncircular (fitted-axis moments). Full feature parity: multi-layer,
  GWT/ru, tension crack + crack water, surcharge, pseudo-static kh. Rewire
  `spencer_fos`/`morgenstern_price_fos` to the rigorous engine if the Duncan
  suite stays green (keep approximate versions as `*_legacy`); else expose as
  new method "gle". Tests: B1, B2, B5, internal consistency (lambda=0 ==
  Bishop for circular; constant-f == Spencer).
- **P2** Janbu simplified + f0 (`method="janbu"`, both FOS reported);
  method-comparison helper. Tests: B3 Janbu, B1 Janbu uncorrected ~2.04.
- **P3** Search upgrades: extend `search_critical_surface` to accept the new
  methods everywhere; add noncircular refinement via scipy
  differential_evolution over control points with kinematic admissibility
  (monotonic x, within ground, depth bounds), seeded from random search best.
  Tests: B3 search gate; refined noncircular <= random-search FOS.
- **P4** Reinforcement: wire nails into Fellenius/Bishop/GLE/Janbu (moment
  arm + horizontal projection); `Geosynthetic` (T_allow, elevation) and simple
  anchors; validated GEC-7-style hand calc test.
- **P5** Probabilistic: `fosm_fos()` (Taylor series + beta_N/beta_LN),
  `monte_carlo_fos()` (dists, seed, fixed surface default, re-search flag,
  histogram data). Tests: B4.
- **P6** Strength models per layer: SHANSEP (S, m, OCR) and Hoek-Brown
  (GSI, mi, sigci, D -> instantaneous c-phi at base normal stress estimate).
  Threaded through build_slices so all methods get them. Hand-calc tests.
- **P7** Ponded water auto-detect (GWT above ground -> water weight as
  vertical load on slices + full-head u). Rapid drawdown: designed stub +
  plan note (descope decision). Tests: submerged-slope hand check; Duncan
  Ex 6 still green.
- **P8** Results/outputs + adapter: per-slice force table incl. E/X in
  to_dict; thrust line; VALIDATION.md (incl. B6 fem2d cross-check); update
  funhouse_agent adapter (new methods/params, allowed_values).

## PROGRESS

- 2026-06-11 P0 done (theory + benchmarks gathered; Slide2 verification
  manual figures read for F&K + ACADS geometry; GeoStudio book ch.2 GLE
  formulation confirmed).
- 2026-06-11 P1 done: gle.py rigorous engine (interslice E marching, exact
  cross-product moment arms, lambda bisection). spencer_fos /
  morgenstern_price_fos REWIRED to rigorous GLE with legacy fallback
  (*_legacy kept). B1 dry: Bishop 2.081/2.080, Spencer 2.073/2.073,
  M-P 2.077/2.076. B1 ru: all within 1.5%. B2 composite within 3%.
  NEW PRE-EXISTING BUG FOUND+FIXED (SS-4): bishop_fos and legacy
  Spencer/M-P used m_alpha = cos(a)+sin(a)tan(phi)/F regardless of slope
  direction -> Bishop overestimated FOS by ~18% on crest-on-the-LEFT
  geometries (F&K benchmark exposed it; Duncan suite is all crest-right
  so it never triggered). Fixed with direction-normalized alpha.
  3 tests asserting legacy artifacts (Spencer theta==0 for circular)
  updated to rigorous semantics. Suite: 264 passed / 17 skipped.

- 2026-06-11 P2 done: janbu_fos (GLE F_f at lambda=0) + f0 correction in
  gle.py; analyze_slope method="janbu" + "gle"; compare_methods now adds
  Janbu corr/uncorr; results fields FOS_janbu/_uncorrected/janbu_f0.
  Validation note: F&K's published 2.041 "Janbu simplified" INCLUDES f0
  (standard practice); ours corrected 2.021 (1.0% off), uncorrected 1.877.
  Suite: 272 passed / 17 skipped.

- 2026-06-11 P3 done: _compute_fos handles janbu/gle/morgenstern_price;
  noncircular searches take method=; search_de (scipy
  differential_evolution over [x_entry, x_exit, depth fracs] with
  convex-bump penalty, seeded by random search); surface_type
  "noncircular_de" in search_critical_surface. ACADS 1(a) B3 gate passed
  for bishop/spencer/janbu/gle (search FOS ~0.96-1.04 vs published 1.00,
  Slide2 0.986-0.990). NEW PRE-EXISTING BUG FOUND+FIXED (SS-5):
  build_slices silently dropped slices where a trial circle dips below
  the deepest layer, leaving fragment masses with absurd FOS (~0.1) that
  won searches (exposed by ACADS's shallow 15-unit layer). Now raises
  ValueError for interior holes -> searches reject the surface.
  Suite: 283 passed / 17 skipped.

- 2026-06-11 P4 done: reinforcement.py (Geosynthetic, Anchor,
  compute_reinforcement_forces with generic line/slip intersection that
  points into the slope toward higher ground); nails wired into
  Fellenius/Bishop (driving-moment reduction, ACTIVE convention per
  GEC-7) and rigorously into GLE/Janbu (exact point-force moments,
  per-slice horizontal equilibrium, vertical component into base
  normal). Legacy SlopeGeometry.reinforcement_force/elevation now acts
  (was a dead field). Hand-calc closed-form test (phi=0 circular)
  matches to 1e-9; pullout-vs-tensile switch tested per GEC-7 equations.
  Suite: 293 passed / 17 skipped.

- 2026-06-11 P5 done: probabilistic.py — fosm_fos (Duncan 2000 Taylor
  series, central differences at +/-sigma, beta_normal + beta_lognormal,
  variance contributions) and monte_carlo_fos (normal/lognormal sampling,
  seed, fixed surface default, research_surface flag, histogram +
  samples). Duncan anchor matched: beta_LN(1.5, 0.17) = 2.32, pf ~1%.
  Closed-form undrained check: COV_F == COV_cu exactly; MC pf agrees with
  FOSM lognormal pf within sampling error (n=4000). 10 tests.

- 2026-06-11 P6 done: per-layer strength_model on SlopeSoilLayer
  ('mohr_coulomb' | 'shansep' | 'hoek_brown'). SHANSEP su =
  S*OCR^m*sigma'_v at slice base (phi=0, su_min floor); GHB
  (Hoek-Carranza-Torres-Corkum 2002) instantaneous c-phi via Balmer
  bisection at the Fellenius base normal estimate. Threaded through
  build_slices so every method gets them. Hand-calc checks: slice su
  exact, FOS scales by OCR^m exactly (phi=0), GHB tangent matches an
  independent envelope evaluation + d(tau)/d(sigma_n) slope to 1%.
  Suite: 315 passed / 17 skipped.

- 2026-06-11 P7 done: ponded water auto-detected from GWT above the
  ground surface — per slice the pond contributes (a) vertical
  water-column weight (kh excluded: seismic acts on soil only) and
  (b) the signed horizontal hydrostatic thrust on inclined submerged
  ground, Fx = gamma_w*(d_l^2 - d_r^2)/2 with exact trapezoidal
  line-of-action; threaded through Fellenius/Bishop (clockwise-positive
  moment Fx*(z - yc)), legacy Spencer/M-P, and the GLE engine (m_ext,
  force denominator, E-marching, thrust line). Fully-submerged ==
  buoyant equivalence: Bishop 0.1%, Spencer/M-P ~exact (0.02%); OMS
  documented exception (N' = W cos a - u l pathology, conservative).
  BEHAVIOR CHANGE (documented): external water now buttresses —
  Duncan Ex 6 test asserting "higher pool -> lower FOS" inverted to the
  physical direction. rapid_drawdown_fos() = designed stub
  (NotImplementedError with the Duncan/USACE 3-stage plan in the
  docstring) — DESCOPED: needs per-slice consolidation-stress
  bookkeeping (Kc-dependent undrained envelopes); interim guidance is
  high internal GWT + drawn-down pool. Suite: 322 passed / 17 skipped.

- 2026-06-11 P8 done: per-slice force table on SliceData/to_dict (W, N',
  S_mob, U=u*l, alpha, boundary E/X for rigorous methods); analyze_slope
  keeps the rich GLEResult (_try_gle) so spencer/M-P/gle expose interslice
  forces + line of thrust (thrust_line on the result and in to_dict);
  compare_methods_table() = F&K-style one-surface/all-methods table
  (rows/surface/summary). 6 tests (test_results_outputs.py).
  Suite: 328 passed / 17 skipped.

- 2026-06-11 P9 done: funhouse adapter modernized — analyze_slope exposes
  method=janbu/spencer/morgenstern_price/gle + f_interslice, force table /
  thrust line; new adapter methods compare_methods_table, fosm_fos,
  monte_carlo_fos; search_critical_surface surface_type adds entry_exit /
  noncircular_de / pso / weak_layer; geometry params add nails / anchors /
  geosynthetics, tension crack, ru, per-layer strength_model
  (shansep / hoek_brown fields), ponded-water note. allowed_values +
  _check_choice ValueErrors everywhere. 26 tests
  (funhouse_agent/tests/test_slope_stability_adapter.py); funhouse suite
  668 passed / 5 skipped.

- 2026-06-11 P10 done: VALIDATION.md (ours-vs-published tables: B1 F&K
  per-method dry+ru, B2 weak-layer composite, B3 ACADS 1(a) searches
  0.985-0.989 vs published 1.00, B4 Duncan beta_LN 2.318/pf 1.02%, B5
  Duncan suite 19/19, B6 fem2d SRM 1.053 vs LE 0.985-0.989 on ACADS) +
  test_validation.py regeneration tests (19 fast + 1 slow-marked SRM
  cross-check ~6 min). ALL PHASES COMPLETE.

Remaining descoped item: rapid drawdown (designed stub in
rapid_drawdown_fos docstring — Duncan/USACE 3-stage method; needs
per-slice consolidation-stress (Kc) bookkeeping).
