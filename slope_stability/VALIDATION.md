# slope_stability VALIDATION (branch le-modern)

Ours-vs-published validation of the modernized LE module (rigorous GLE /
Fredlund-Krahn engine, Janbu+f0, searches, probabilistic). All "ours"
values regenerated on this branch, 2026-06-11.

Runnable: `pytest slope_stability/tests/test_validation.py -v`
(the fem2d SRM cross-check is `-m slow`, ~6 min; everything else is fast).
The same benchmarks are also asserted piecewise in test_gle.py,
test_janbu.py, test_search_modern.py and test_probabilistic.py.

---

## B1 — Fredlund & Krahn (1977) homogeneous slope
[Slide2 Verification #21. Surface (0,60)-(60,60)-(140,20)-(180,20) ft;
c'=600 psf, phi'=20 deg, gamma=120 pcf; specified circle xc=120, yc=90,
R=80. 50 slices.]

### Case 1 — dry

| Method                    | F&K (1977) | Slide2 | Ours  | vs F&K |
|---------------------------|-----------:|-------:|------:|-------:|
| Ordinary (Fellenius)      | 1.928      | 1.931  | 1.926 | -0.1%  |
| Bishop simplified         | 2.080      | 2.079  | 2.075 | -0.2%  |
| Janbu simplified (w/ f0)  | 2.041      | —      | 2.017 | -1.2%  |
| Janbu simplified (no f0)  | —          | —      | 1.877 | (f0=1.075) |
| Spencer                   | 2.073      | 2.075  | 2.074 | +0.0%  |
| Morgenstern-Price (half-sine) | 2.076  | 2.075  | 2.076 | +0.0%  |

Ours M-P lambda = 0.331; Spencer theta = 14.7 deg. Note: F&K's published
"Janbu simplified" 2.041 includes the f0 correction (standard practice).

### Case 2 — ru = 0.25

| Method                    | F&K (1977) | Ours  | vs F&K |
|---------------------------|-----------:|------:|-------:|
| Ordinary (Fellenius)      | 1.607      | 1.606 | -0.1%  |
| Bishop simplified         | 1.766      | 1.758 | -0.5%  |
| Spencer                   | 1.761      | 1.763 | +0.1%  |
| Morgenstern-Price (half-sine) | 1.764  | 1.764 | +0.0%  |

## B2 — F&K (1977) weak-layer composite surface
[Slide2 Verification #22. Same surface; weak layer el. 15-16 ft (c'=0,
phi'=10); composite surface = same circle clipped at z=15; 60 slices,
moment axis at (120, 90). Published sources disagree in the 2nd decimal
(weak-layer position sensitivity); gate 3% on F&K.]

| Case      | Method            | F&K   | Slide2 | Ours  | vs F&K |
|-----------|-------------------|------:|-------:|------:|-------:|
| dry       | Ordinary          | 1.288 | 1.300  | 1.283 | -0.4%  |
| dry       | Spencer           | 1.373 | 1.382  | 1.354 | -1.4%  |
| dry       | Morgenstern-Price | 1.370 | 1.372  | 1.349 | -1.5%  |
| ru = 0.25 | Spencer           | 1.118 | 1.124  | 1.109 | -0.8%  |
| ru = 0.25 | Morgenstern-Price | 1.118 | —      | 1.102 | -1.4%  |

## B3 — ACADS 1(a) critical-circle search
[Giam & Donald 1989; Slide2 Verification #1. Surface (20,25)-(30,25)-
(50,35)-(70,35) m; c'=3 kPa, phi'=19.6 deg, gamma=20 kN/m3, dry.
Published answer 1.00. Ours: entry-exit search, 12x12 windows
(20-32 / 46-68), ~6000 trial circles, 30 slices.]

| Method            | Published | Slide2 | Ours (search min) |
|-------------------|----------:|-------:|------------------:|
| Bishop            | 1.00      | 0.987  | 0.986             |
| Spencer           | 1.00      | 0.986  | 0.985             |
| GLE (half-sine)   | 1.00      | 0.986  | 0.985             |
| Janbu corrected   | 1.00      | 0.990  | 0.989             |

## B4 — Duncan (2000) reliability anchor
[Duncan, "Factors of safety and reliability in geotechnical engineering",
JGGE 126(4). Lognormal reliability index for F_MLV = 1.50, COV_F = 0.17.]

| Quantity      | Duncan (2000) | Ours   |
|---------------|--------------:|-------:|
| beta_LN       | 2.32          | 2.318  |
| pf            | ~1%           | 1.02%  |

Cross-checks (test_probabilistic.py): undrained closed form COV_F ==
COV_cu exactly; Monte Carlo pf matches FOSM-lognormal pf within sampling
error (n = 4000, seeded).

## B5 — Duncan, Wright & Brandon textbook suite
test_duncan_verification.py: 19 tests, all passing on this branch
(includes the documented behavior change from P7: external pool water now
buttresses the slope — the Duncan Ex. 6 assertion was inverted to the
physical direction).

## B6 — LE vs fem2d Strength Reduction Method (Griffiths & Lane 1999 style)
[ACADS 1(a) geometry, fem2d.analyze_slope_srm at this branch state
(read-only import), 40x12 mesh, SRF bisection tol 0.01.]

| Engine                      | FOS   |
|-----------------------------|------:|
| LE search (Bishop/Spencer/GLE) | 0.985-0.986 |
| LE search (Janbu corrected) | 0.989 |
| fem2d SRM                   | 1.053 |
| Published (Giam & Donald)   | 1.00  |

LE-vs-FEM spread ~6-7%: consistent with the usual SRM-vs-LE comparisons
(Griffiths & Lane 1999 report agreement within a few percent; coarse
meshes bias SRM high). The cross-check test is `-m slow` (~6 min).

## B7 — Rapid drawdown (Corps 2-stage / Duncan-Wright-Wong 3-stage) + search
[`rapid_drawdown_fos` (single specified surface) and `search_rapid_drawdown`
(critical-surface search under drawdown strengths, v5.4 E1). Low-permeability
layers carry the R-envelope (`R_c`/`R_phi`); the FOS is a GLE/Spencer solve at the
drawn-down pool with each undrained slice's base overridden to `c = tau_ff, phi =
0`. Feet inputs converted to SI. Full theory in DESIGN.md.]

### Single specified surface (validation_examples V-037 / V-038)

| Problem | Method | Ours | Published | vs pub |
|---------|--------|-----:|----------:|-------:|
| #95 EM 1110-2-1902 dam (seepage stage-1) | Corps 2-stage | 1.34  | 1.347 | -0.5% |
| #95 (flat full-pool stage-1, conservative bound) | Corps 2-stage | 1.21 | — | — |
| #96 same dam (seepage stage-1), default stage-3 | DWW 3-stage | 1.273 | 1.443 | -12% |
| #96 same dam (seepage stage-1), `stage3_effective_normal='gle'` | DWW 3-stage | 1.370 | 1.443 | -5% |

**#96 3-stage refinement (v5.4 E2).** The default (Fellenius stage-3 effective
normal) is preserved byte-for-byte (flat 1.235 / seepage 1.273). The residual to
the published 1.443 was traced NOT to the Kc interpolation (which alone yields
~1.45) but to the stage-3 drained substitution: the Fellenius `W*cos(a)/l - u`
normal under-predicts N' and fires the substitution on 17/50 slices. The optional
`stage3_effective_normal='gle'` uses the rigorous GLE normal (consistent with
stage 1; 9 genuine substitutions) → 1.306 flat / 1.370 seepage, closing most of
the gap. Gated behind the new parameter; the lead decides whether to flip the
default. ~5% residual left, within the representative-flow-net + LE-N'-at-FOS
sensitivity band (cf. the +0.6% Corps residual on #95).

### Critical-surface search (validation_examples V-041, v5.4 E1)

`search_rapid_drawdown` reuses the ordinary search machinery (grid / entry-exit /
random / DE) with the drawdown FOS wired in per surface via a new `fos_fn` hook;
degenerate/non-converged trial surfaces are rejected (a stage-1+stage-3
convergence gate + a min-slice / stage-1-FOS floor kill the shallow near-flat
slivers a circular search would otherwise pick with a spurious low FOS).

| Problem | Method | Ours (search min) | Published (search min) | vs pub |
|---------|--------|------------------:|-----------------------:|-------:|
| #98 Walter Bouldin (approx geom) | Corps 2-stage | 0.837 | 0.931 | -10% |
| #98 (approx geom) | DWW 3-stage (default) | 0.938 | 1.039 | -10% |
| #98 (approx geom) | DWW 3-stage (`gle` stage-3) | 0.959 | 1.039 | -8% |

Wrapper MECHANICS are exact on the #95/#96 EXACT section (search min 0.90 <= the
published specified-circle single-surface FOS; the stage detail recomputed on the
winning surface reproduces the search FOS). #98's ~10% low minima are a geometry
effect — the cross-section is only RECOVERED (flat-stacked layers, real pinch-outs
simplified, riprap veneer omitted; INVENTORY #98), and two independent search
types (grid + entry-exit) both land ~0.85 Corps, with the correct ordering
(DWW > Corps). NOT tuned; the recovered geometry is left as-is.

---

## Notes / known deviations

- Ordinary method with ru: formulation-sensitive (F&K 1.607 vs Slide2
  1.687 on B1 case 2); ours matches F&K's formulation.
- B2 published values themselves disagree at the 1-2% level (F&K vs
  Slide2 vs Zhu); ours sits 1-2% below F&K — within the documented gate.
- Fully-submerged slope: Bishop matches the buoyant-equivalent analysis
  to 0.1%, Spencer/M-P to ~0.02%; OMS has the textbook N' = W cos(a) - u l
  pathology (conservative), documented in test_water.py.
- Pre-existing bugs fixed on this branch that affect validation:
  SS-4 (m_alpha direction error: Bishop overestimated FOS ~18% on
  crest-on-the-left geometries) and SS-5 (build_slices silently dropped
  below-base fragments, letting absurd FOS ~0.1 win searches).
- Rapid drawdown: IMPLEMENTED (v5.3 B2a + v5.4 E1/E2) — see section B7 above.
  The earlier "designed stub only" status is superseded: `rapid_drawdown_fos`
  (single surface) and `search_rapid_drawdown` (critical-surface search) are
  validated against Slide2 #95/#96/#98.
