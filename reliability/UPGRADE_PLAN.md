# reliability/ — Geotechnical Reliability Module: Build Plan

New top-level package: statistical variability of subsurface properties +
probabilistic geotech analyses (CONSOLIDATION_PLAN item #7 — the prime
long-term target). Hard deps: numpy + scipy only. pystra/SALib/gstools remain
optional cross-check / pointer territory.

## Architecture

```
reliability/
  __init__.py        public API
  variables.py       RandomVariable (normal/lognormal/uniform/triangular,
                     truncation), correlation-matrix assembly + validation
  stats.py           sample stats, COV from data/params, combined COV (Eq 7-5),
                     std-dev-from-range 6/N-sigma rule (Eq 7-6), beta<->pf,
                     rate of exceedance (Eqs 7-15/7-16)
  results.py         FOSMResult / PEMResult / MonteCarloResult / FORMResult
                     dataclasses with summary()/to_dict() (house pattern from
                     slope_stability/probabilistic.py + results.py)
  fosm.py            FOSM Taylor-series engine (central difference at +/-1 sigma,
                     per-variable variance contributions = Duncan's table,
                     correlated cross terms)
  pem.py             Rosenblueth point-estimate (2^n, correlation-adjusted
                     weights; 2n+1 multiplicative reduced option)
  monte_carlo.py     numpy Generator MC (seed, LHS via scipy.stats.qmc,
                     Cholesky/Nataf-lite correlation, convergence trace, pf+CI)
  form.py            native FORM: HL-RF iteration with Rackwitz-Fiessler
                     equivalent normals; design point, alpha vector
  cov_database.py    published COV ranges as DATA with provenance
                     (Phoon & Kulhawy 1999; Duncan 2000 Table 1 / UFC 3-220-20
                     ch. 7; scale-of-fluctuation table) + cov_guidance()
  spatial.py         Vanmarcke variance reduction Gamma^2(L/delta)
                     (exact exponential-ACF + simple delta/L approximations)
  wrappers.py        pre-canned g() builders: bearing capacity FOS, axial pile
                     FOS, slope delegate -> slope_stability.probabilistic
  tests/             DM7-validated resurrected suite + engine + validation tests
  DESIGN.md          theory & conventions (normal vs lognormal beta, correlation)
  VALIDATION.md      ours vs published (centerpiece)
  UPGRADE_PLAN.md    this file
```

All engines drive a user-supplied callable `g(values: dict[str, float]) -> float`.
`convention="fos"` (failure at g<1) or `"margin"` (failure at g<0).

## Theory anchors (citations + benchmark values)

- **Reliability indices** (Duncan 2000, Eqs.; UFC 3-220-20 Eqs. 7-7/7-12):
  - normal:    beta = (mu_F - 1)/sigma_F   (FOS convention)  /  mu_g/sigma_g (margin)
  - lognormal: beta_LN = ln(F_MLV / sqrt(1+COV_F^2)) / sqrt(ln(1+COV_F^2))
  - **Benchmark**: F_MLV=1.5, COV_F=0.17 -> beta_LN=2.32, pf~1% (Duncan 2000;
    already the validated anchor in slope_stability/VALIDATION.md).
- **FOSM** (UFC 3-220-20 Eqs. 7-8..7-11; Duncan 2000 Taylor series):
  sigma_g^2 = sum (Delta g_i / 2)^2 (+ 2 rho_ij dgi dgj sigma cross terms when
  correlated). Free validation: resurrected DM7Eqs tests
  (`git show 450320b~1:DM7Eqs/tests/test_dm7_2_chapter7.py`), e.g.
  g=x-y, mu=[10,4], sigma=[2,1] -> mu_g=6, sigma_g=sqrt(5), beta=2.683.
- **PEM** (Rosenblueth 1975; UFC 3-220-20 Eqs. 7-13/7-14): 2^n points at
  mu +/- sigma, weights 2^-n * (1 + sum s_i s_j rho_ij). Benchmark: g=x*y,
  mu=[2,3], sigma=[1,1] -> mu_g=6, var_g=14 (resurrected DM7 test).
- **Exact linear-margin anchors** (USACE ETL 1110-2-547 App. B; Ang & Tang):
  - R,S normal: beta = (muR-muS)/sqrt(sigR^2+sigS^2) — exact; all four engines
    must agree.
  - R,S lognormal: beta_LN = ln[(muR/muS) sqrt((1+VS^2)/(1+VR^2))] /
    sqrt(ln[(1+VR^2)(1+VS^2)]) — exact for g=R/S; FORM must match.
- **FORM** (Hasofer-Lind 1974; Rackwitz-Fiessler 1978): HL-RF in U space,
  equivalent normals sigma_eq = phi(PHI^-1(F(x)))/f(x),
  mu_eq = x - sigma_eq*PHI^-1(F(x)); beta=|u*|, alpha=-grad/|grad|.
  Cross-check vs pystra (installed in this venv -> live test, skip-guarded).
- **Combined COV** (UFC 3-220-20 Eq. 7-5; Phoon & Kulhawy 1999):
  COV_total^2 = COV_inherent^2 + COV_measure^2 + COV_transform^2
  (spatially averaged: Gamma^2*COV_w^2 + ...).
- **6-sigma rule** (Duncan 2000 "three-sigma"/range rule; UFC Eq. 7-6):
  sigma = (HCV - LCV)/N, N=6 default (UFC), 4 conservative.
- **Vanmarcke variance reduction** (Vanmarcke 1977, 1983): for exponential ACF
  rho(tau)=exp(-2|tau|/delta): Gamma^2(L) = 2(delta/2L)^2 [2L/delta - 1 +
  exp(-2L/delta)]; limits Gamma^2->1 (L<<delta), -> delta/L (L>>delta).
  Simple approximation Gamma^2 = min(1, delta/L).
- **COV knowledge base**: Duncan (2000) Table 1 (also reproduced in UFC
  3-220-20 ch. 7), Phoon & Kulhawy (1999) inherent/measurement/
  scale-of-fluctuation tables. Each entry stored with property, test,
  cov_range, source string.

## Phases

- **Phase 0 — harvest (DONE)**: recovered DM7Eqs chapter7.py + tests from git
  history into module_work/_harvest (not committed; regenerate with
  `git show 450320b~1:...`). Read slope_stability/probabilistic.py (read-only
  cross-validation target), funhouse adapter pattern, pystra_adapter.
- **Phase 1 — plan**: this file, committed.
- **Phase 2 — core**: variables.py, stats.py, results.py + tests (incl. the
  resurrected DM7-validated stats/index tests). Suite green, commit.
- **Phase 3 — engines**: fosm.py, pem.py, monte_carlo.py, form.py + tests
  (DM7 values, exact linear anchors, engine-agreement on shared problem,
  pystra cross-check test with skipif). Commit.
- **Phase 4 — knowledge base + spatial**: cov_database.py, spatial.py + tests.
  Commit.
- **Phase 5 — wrappers**: wrappers.py (bearing, axial pile, slope delegate) +
  cross-validation test (same slope through reliability.fosm and
  slope_stability.fosm_fos agrees). Commit.
- **Phase 6 — funhouse adapter**: funhouse_agent/adapters/reliability_adapter.py
  (methods: fosm, pem, monte_carlo, form, cov_guidance, combined_cov,
  variance_reduction), MODULE_REGISTRY entry, adapter tests; pyproject
  packages include + testpaths. funhouse tests green. Commit.
- **Phase 7 — docs**: DESIGN.md, VALIDATION.md (ours-vs-published table),
  CLAUDE.md module line, CONSOLIDATION_PLAN.md #7 -> DONE. Full suite green.
  Commit.

## Descoped (designed path documented in DESIGN.md)

- Random-field FEM / 2-D random fields -> gstools_agent pointer.
- SORM -> pystra_agent (optional dep) already exposes it.
- Full Nataf correlation-coefficient correction (we do Cholesky in normal
  space = "Nataf-lite"; exact for normal marginals, good approximation for
  moderate rho with lognormal marginals).
- System reliability, LRFD resistance-factor calibration -> next increment.

## PROGRESS

- [x] Phase 0: harvest
- [x] Phase 1: plan committed
- [x] Phase 2: core (variables/stats/results) — 65 tests
- [x] Phase 3: engines (FOSM/PEM/MC/FORM) — 117 tests, pystra cross-check live
- [ ] Phase 4: cov_database + spatial
- [ ] Phase 5: wrappers + slope cross-validation
- [ ] Phase 6: funhouse adapter + registration
- [ ] Phase 7: docs + full suite green

NEXT ACTION: Phase 4 — cov_database.py + spatial.py.
