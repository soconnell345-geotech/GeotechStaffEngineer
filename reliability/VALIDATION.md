# reliability/ — Validation: Ours vs Published

Every offline-computable row is asserted in `reliability/tests/test_validation.py`
(plus the engine test files), so this table cannot silently rot. Run:

```
python -m pytest reliability -q
```

## 1. Published anchors

| # | Source | Problem | Published | Ours | Test |
|---|--------|---------|-----------|------|------|
| 1 | Duncan (2000) Fig./text, retaining-wall sliding | F_MLV = 1.50, COV_F = 17% | pf ≈ 1% (reliability ≈ 99%); beta_LN = 2.32 | beta_LN = 2.320, pf = 1.02% | test_validation::test_retaining_wall_sliding |
| 2 | Duncan (2000), LASH terminal cut slope | F_MLV = 1.17, COV_F = 16% | pf = 18% (observed: ~22% of slope length failed) | pf = 18.0% | test_validation::test_lash_terminal_cut_slope |
| 3 | Duncan (2000), consolidation settlement | most-likely 1.07 ft, COV 21% | settlement ratio at 1% exceedance ≈ 1.6 | SR(1%) = 1.61 (lognormal model) | test_validation::test_consolidation_settlement_exceedance |
| 4 | UFC 3-220-20 ch. 7 (DM7-2), Eqs. 7-8/7-10/7-11/7-7 | FOSM, g = x−y, mu=[10,4], sigma=[2,1] | mu_g=6, sigma_g=√5, beta=2.683 | exact match | test_validation::test_fosm_linear (full resurrected DM7Eqs suite in test_stats/test_fosm/test_pem) |
| 5 | UFC 3-220-20 ch. 7, Eqs. 7-13/7-14 | Rosenblueth PEM, g = x·y, mu=[2,3], sigma=[1,1] | E[g]=6, Var[g]=14 | exact match | test_validation::test_pem_product |
| 6 | UFC 3-220-20 ch. 7, Eq. 7-16 | 10% exceedance in 50 yr | lambda = 0.002107 (≈475-yr return) | exact match | test_validation::test_seismic_rate_of_exceedance |
| 7 | USACE ETL 1110-2-547 App. B / Ang & Tang | exact normal margin R−S: R(15,2), S(10,1.5) | beta = 2.000 exact | FOSM 2.000, PEM 2.000, FORM 2.0000, MC pf = 0.0228 (n=200k, exact 0.02275) | test_validation::test_normal_margin_all_engines |
| 8 | USACE ETL 1110-2-547 App. B (lognormal R/S closed form) | R LN(20, 20%), S LN(10, 25%) | beta = 2.2275 exact | FORM beta = 2.2275 | test_validation::test_lognormal_quotient_exact |
| 9 | Duncan (2000) Table 3 COV values | gamma 3–7%, phi' 2–13%, su 13–40%, SPT 15–45%, ... | stored verbatim (all 15 rows verified digit-for-digit against the in-hand paper 2026-07-18, module_work/wiki_verification/duncan_2000_cov.md) | provenance-locked | test_cov_spatial::TestPublishedValues |
| 10 | ISSMGE-TC304 (2021) Tables 1.2/1.3, 3.1 | site-specific COVs (su clay mean 28.2%, phi sand mean 7.9%) + scales of fluctuation (clay delta_v avg 2.47 m) | stored verbatim from the report PDF | provenance-locked | test_cov_spatial |

## 2. FORM vs pystra cross-check (live, pystra installed)

Nonlinear 3-variable margin g = R·B − S, R ~ LN(200, 30), B ~ N(1.0, 0.05),
S ~ LN(100, 20):

| | beta | pf |
|---|------|----|
| reliability.form (native HL-RF + Rackwitz-Fiessler) | 2.7725 | 2.781e-3 |
| pystra FORM (via pystra_agent) | 2.7725 | 2.781e-3 |

Identical to 4 decimals. Also checked for the linear normal case (abs 2e-3).
Tests: `test_form::TestPystraCrossCheck` (skip-guarded if pystra absent).

## 3. Engine-agreement table (shared geotech problem)

Square footing B=2 m, Df=1.5 m on sand, q_applied = 700 kPa;
phi' ~ LN(32°, COV 8%), gamma ~ N(18 kN/m³, COV 5%) (Vesic q_ult).
FOS = q_ult/q. Test: `test_validation::TestEngineAgreementSharedProblem`.

| Engine | E[FOS] | COV_F | beta_LN / beta | pf | g() calls |
|--------|--------|-------|----------------|----|-----------|
| FOSM (Taylor) | 2.220 | 0.354 | 2.151 (LN) | 1.57e-2 | 5 |
| PEM (Rosenblueth 2^n) | 2.366 | 0.333 | 2.496 (LN) | 6.28e-3 | 4 |
| Monte Carlo (n=20k, seed 42) | 2.375 | 0.385 | 2.316 (LN fit) | empirical 3.80e-3 [CI 2.9e-3 – 4.7e-3] | 20,000 |
| FORM (HL-RF) | — | — | 2.627 | 4.31e-3 | 41 |

Reading the table (expected behavior, asserted in the test):

- The full-distribution methods agree: FORM pf (4.31e-3) lies inside the
  Monte Carlo 95% CI; FORM's design point is phi* = 25.97°, gamma* = 17.5.
- FOSM, being a linearization at the mean, underestimates E[FOS] for this
  convex g (exp-type Nq growth in phi) and is conservative on pf here.
- The "which variable matters" outputs agree: FOSM variance contributions
  phi' 98% / gamma 2%; FORM alpha² split phi' 96.1% / gamma 3.9%
  (alpha_phi = −0.980, alpha_gamma = −0.197, both resistance-like).

## 4. Slope cross-validation (same problem, both code paths)

Undrained clay slope (cu = 40 kPa, COV 20%), fixed circle (30, 32, R=26),
Fellenius, 40 slices:

| Path | F_MLV | beta_LN | pf_LN |
|------|-------|---------|-------|
| slope_stability.probabilistic.fosm_fos (validated module) | 1.437 | 1.7315 | 4.17e-2 |
| reliability.fosm driving the slope FOS as black-box g() | 1.437 | 1.7315 | 4.17e-2 |

Agreement to 1e-6 (identical ±1σ evaluations). Test:
`test_wrappers::TestSlopeCrossValidation`. The `slope_reliability` wrapper
delegates to the slope_stability implementation outright.

## 5. Internal exactness checks (selected)

- Correlated FOSM/PEM reproduce Var[x±y] = σ1²+σ2²±2ρσ1σ2 exactly
  (test_fosm, test_pem).
- FORM with a single lognormal FOS variable reproduces the Duncan lognormal
  index exactly (beta = 2.32 at F=1.5, COV 17%; design point F* = 1.000).
- FORM with one uniform variable reproduces the exact pf = 0.25 anchor.
- Rackwitz-Fiessler equivalent normals match the lognormal closed form
  (sigma_eq = zeta·x, mu_eq = x(1 − ln x + lambda)) to 1e-6.
- Vanmarcke Gamma²: limits Gamma²(0)=1 and Gamma²(L→∞)=delta/L verified;
  exact value at 2L/delta=2 equals 0.5(1+e⁻²).
