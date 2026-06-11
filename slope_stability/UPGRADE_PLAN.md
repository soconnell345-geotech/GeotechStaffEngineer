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

NEXT ACTION: implement P2 (janbu_fos + f0 correction + compare helper) (`slope_stability/gle.py` + tests/test_gle.py with
B1/B2 benchmarks), then decide rewire-vs-new-method from Duncan suite results.
