# reliability/ — Design Notes

Geotechnical reliability: statistical variability of subsurface properties +
probabilistic analyses. Hard deps: numpy + scipy only.

## Architecture

```
g(values: dict[str, float]) -> float        # user-supplied performance fn
            ▲
   ┌────────┴──────────┬─────────────┬──────────────┐
 fosm.py             pem.py      monte_carlo.py   form.py
 (Taylor ±1σ)      (Rosenblueth)  (numpy/LHS)    (HL-RF + R-F)
   └────────┬──────────┴─────────────┴──────────────┘
        variables.py (RandomVariable, correlation)  →  results.py
 cov_database.py (published COVs)   spatial.py (Vanmarcke Γ²)
 wrappers.py (bearing / pile / slope-delegate)
 funhouse_agent/adapters/reliability_adapter.py (agent surface)
```

All engines accept the same inputs: `g`, `variables` (dict spec or
`RandomVariable` list), optional `correlation`, and `convention`.

## Conventions — the part people get wrong

**convention="fos"** (default): g returns a FACTOR OF SAFETY; failure at
g < 1.

- `beta_normal = (E[F] − 1) / sigma_F` — assumes F normally distributed.
- `beta_lognormal = ln(F_MLV / sqrt(1+COV_F²)) / sqrt(ln(1+COV_F²))` —
  assumes F lognormal (Duncan 2000; UFC 3-220-20 Eq. 7-12).

**convention="margin"**: g is a margin (R − S); failure at g < 0.
`beta_normal = E[g]/sigma_g`. The lognormal index is **None** — a margin can
be negative, so a lognormal model of g is meaningless. Use a quotient
F = R/S with convention="fos" if you want the lognormal convention.

When to prefer which index (Duncan 2000, USACE practice):

- FOS values are bounded below by 0 and right-skewed → the lognormal index
  is the usual geotechnical reporting convention (and what the published
  anchors use: F=1.5/COV 17% → beta_LN=2.32, pf≈1%).
- The normal index is appropriate for additive margins and for comparison
  with structural targets (e.g. beta_T in LRFD work).
- FORM's beta is geometric (distance to the failure surface in U-space):
  it equals the normal index for linear-normal problems and the lognormal
  index for products/quotients of lognormals; in general it is the most
  defensible of the three because it uses the full marginal distributions.

## Engines

| Engine | Evaluations | Uses dists? | Correlation | Strengths |
|--------|-------------|-------------|-------------|-----------|
| `fosm` | 2n+1 | moments only | cross terms in Var | cheap; Duncan variance-contribution table |
| `pem` | 2^n (or 2n+1 reduced) | moments only | Rosenblueth weights | captures convexity of g; no derivatives |
| `monte_carlo` | n | yes | Cholesky in Z space | empirical pf + CI; any g; histogram |
| `form` | ~(2n+2)/iter | yes (R-F equivalent normals) | Cholesky in U space | design point + alphas; exact for linear/LN cases |

Implementation notes:

- **FOSM** uses ±1σ central differences (secant over the plausible range),
  not a small numerical step — intentional, per Duncan (2000)/USACE. The
  variance-contribution percentages are computed from the uncorrelated
  diagonal terms.
- **PEM full scheme** weights: `2^-n (1 + Σ s_i s_j ρ_ij)`; negative-weight
  combinations (strong correlations) raise rather than silently clamp.
  The `multiplicative` (2n+1) reduced scheme is Rosenblueth's alternative
  for many variables; uncorrelated and g≠0 only.
- **Monte Carlo**: numpy `Generator(seed)`; optional Latin Hypercube via
  `scipy.stats.qmc`; correlation imposed on standard normals via Cholesky,
  then marginal transform `x = F⁻¹(Φ(z))` ("Nataf-lite", see below);
  non-finite g realizations are dropped and reported via `n`. pf comes with
  a 95% binomial CI and a 10-point convergence trace — if the CI is wide,
  raise n (rule of thumb n ≥ 10/pf).
- **FORM**: improved HL-RF in U-space with merit-based step halving;
  non-normal marginals handled by Rackwitz-Fiessler equivalent normals
  (closed-form-exact for lognormal); gradient by central differences in
  x-space mapped through `∇_u g = Lᵀ(σ_eq ∘ ∇_x g)`. beta is signed: the
  mean point inside the failure domain gives beta < 0. alpha is the unit
  sensitivity vector; alpha² sums to 1 and is reported as a percentage
  split (negative alpha = resistance variable: failure when it drops).

## Random variables

`RandomVariable(name, mean, std|cov, dist, lower, upper, mode)` with
`normal | lognormal | uniform | triangular`. Lognormal moments are the
ARITHMETIC mean/std (underlying mu_ln/sigma_ln derived internally).
Uniform/triangular accept bounds (preferred) or symmetric mean/std.
`lower`/`upper` on normal/lognormal truncate (renormalized CDF/PPF) —
sampling and FORM honor truncation; moment engines (FOSM/PEM) use the
underlying moments.

## Correlation handling ("Nataf-lite") — limitation

The correlation matrix is applied to the underlying STANDARD NORMALS
(Cholesky), then each marginal is transformed. For normal marginals this is
exact. For non-normal marginals the realized product-moment correlation of
x differs slightly from the target (the full Nataf transform would inflate
the normal-space ρ to compensate). For the moderate |ρ| ≤ ~0.6 typical of
soil-property cross-correlations the discrepancy is small (a few percent of
ρ). If you need exact Nataf, use pystra (it implements the correction) —
or add the Nataf integral here as a future increment.

## Property-variability knowledge base

`cov_guidance(property, soil_type=, test=, category=)` returns published
rows (percent COV + provenance string). Categories:

- `inherent` — Duncan (2000) Table 1 (verified against the paper),
- `site_specific` — ISSMGE-TC304 (2021) Tables 1.2/1.3/1.4,
- `total_test` — in-situ test variability (Kulhawy & Trautmann ranges as
  reproduced in Duncan Table 1),
- `transformation` — Phoon & Kulhawy (1999b) correlation-model
  uncertainties as quoted in UFC 3-220-20 §7-3.1.3.

`combined_cov` implements UFC Eq. 7-5 (sum of squares), with an optional
`variance_reduction` factor applied to the inherent part only.
`std_from_range` is the UFC Eq. 7-6 N-sigma rule (N=6 default).

## Spatial averaging

`variance_reduction(L, delta)` = Vanmarcke Γ²(L/δ): exact exponential-ACF
variance function (default) or the δ/L approximation. Workflow:

```python
g2  = variance_reduction(L=10, delta=2.5)            # averaging length L
cov = combined_cov(cov_inherent, cov_meas, cov_trans, variance_reduction=g2)
rv  = RandomVariable("su", mean=50, cov=cov, dist="lognormal")
```

Published δ guidance: `scale_of_fluctuation_guidance(soil_type)` (Cami et
al. 2020 / TC304 Table 3.1; δ_h/δ_v typically 10–20). This is 1-D averaging
along a line (failure surface, pile shaft); full 2-D/3-D random fields,
kriging and conditional simulation live in `gstools_agent`.

## Wrappers

- `bearing_capacity_reliability` / `axial_pile_reliability`: build
  FOS-convention g() around the deterministic modules; variable means
  default to the deterministic inputs; per-layer scoping `'cohesion:2'`.
- `slope_reliability`: delegates outright to
  `slope_stability.probabilistic` (FOSM/MC), which is the Duncan-validated
  implementation; the cross-validation test pins both paths together.

## Limitations / descoped (designed path)

- **SORM**: use `pystra_agent` (optional dep) — curvature correction rarely
  changes geotech decisions at beta < 3.5.
- **Exact Nataf correlation correction**: see above; pystra or future
  increment.
- **Random-field FEM / 2-D spatial simulation**: `gstools_agent` (SRF,
  kriging) + `fem2d`; a coupled random-field-FEM driver is the natural next
  increment.
- **System reliability** (multiple failure modes, series/parallel bounds)
  and **LRFD resistance-factor calibration** (beta_T → phi factors): next
  increments; the engines and the COV knowledge base here are the
  prerequisites for both.

## References

Duncan (2000) JGGE 126(4); UFC 3-220-20 (2025) ch. 7; USACE ETL 1110-2-547
(1995); Phoon & Kulhawy (1999a,b) CGJ 36(4); Rosenblueth (1975) PNAS 72(10);
Hasofer & Lind (1974); Rackwitz & Fiessler (1978); Vanmarcke (1977, 1983);
Baecher & Christian (2003); Cami et al. (2020); ISSMGE-TC304 (2021).
