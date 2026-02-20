# pystra_agent Design Document

## Purpose

The pystra_agent module provides structural reliability analysis capabilities for geotechnical engineering applications. It wraps the pystra library to perform:

1. **FORM** (First Order Reliability Method): Linear approximation of failure surface
2. **SORM** (Second Order Reliability Method): Quadratic approximation including curvature
3. **Monte Carlo**: Direct simulation for exact probability estimates

## Theory

### Reliability Index and Probability of Failure

The reliability index β is the shortest distance from the origin to the failure surface in standard normal space (U-space). It relates to probability of failure by:

```
Pf = Φ(-β)
```

where Φ is the standard normal CDF.

Typical values:
- β = 2.0 → Pf ≈ 0.023 (2.3%)
- β = 3.0 → Pf ≈ 0.0013 (0.13%)
- β = 4.0 → Pf ≈ 0.00003 (0.003%)

Target reliability indices:
- Structures with moderate consequences: β ≥ 3.0
- Critical infrastructure: β ≥ 3.5 to 4.0
- Temporary structures: β ≥ 2.5

### Limit State Function

The limit state function g(X) separates safe and failure domains:
- g(X) > 0: Safe
- g(X) = 0: Limit state (failure surface)
- g(X) < 0: Failure

Common forms in geotechnical engineering:
- **Bearing capacity**: g = Qult - Q (resistance - load)
- **Slope stability**: g = FS - 1.0 (factor of safety - 1)
- **Settlement**: g = δallow - δ (allowable - actual settlement)

### FORM (First Order Reliability Method)

FORM finds the design point x* (most probable failure point) using gradient-based optimization. The design point is the point on the failure surface closest to the origin in U-space.

**Sensitivity factors (alpha)**: Indicate the relative importance of each variable's uncertainty:
- α_i = ∇g_u / |∇g_u| at design point
- Σ(α_i²) = 1.0
- |α_i| close to 1: Variable dominates reliability
- α_i > 0: Variable acts as resistance (increasing it increases safety)
- α_i < 0: Variable acts as load (increasing it decreases safety)

**Advantages**:
- Fast (typically converges in 5-10 iterations)
- Provides design point and sensitivity information
- Accurate for linear/mildly nonlinear limit states

**Limitations**:
- Less accurate for highly nonlinear limit states
- Assumes unimodal failure region

### SORM (Second Order Reliability Method)

SORM improves FORM by including curvature information via principal curvatures κ_i at the design point. Several approximations exist; pystra uses Breitung's formula:

```
Pf_SORM ≈ Φ(-β) ∏(1 + κ_i*β)^(-1/2)
```

**Advantages**:
- More accurate than FORM for nonlinear limit states
- Still computationally efficient

**Limitations**:
- Requires second derivatives (Hessian)
- Can be unstable for very nonlinear problems

**When to use SORM**:
- Limit state involves products, powers, or ratios of variables
- FORM and Monte Carlo estimates differ significantly
- Need to verify FORM approximation quality

### Monte Carlo Simulation

Monte Carlo directly samples the joint distribution and counts failures:

```
Pf_MC = (# failures) / (# samples)
```

**Advantages**:
- Exact (in the limit of infinite samples)
- No assumptions about linearity or distribution

**Limitations**:
- Computationally expensive for rare events
- COV of estimate ≈ sqrt((1-Pf)/(n*Pf))

**Sample size guidelines**:
- For Pf ≈ 10^-2: 10,000 samples (COV ≈ 10%)
- For Pf ≈ 10^-3: 100,000 samples (COV ≈ 10%)
- For Pf ≈ 10^-4: 1,000,000 samples (COV ≈ 10%)

Rule of thumb: Need at least 100 failures for COV < 10%.

## Distributions

### Common Distributions in Geotechnical Engineering

| Variable | Typical Distribution | Reasoning |
|----------|---------------------|-----------|
| Undrained shear strength (Su) | Lognormal | Cannot be negative, often positively skewed |
| Friction angle (φ) | Normal or Beta | Bounded (0-90°), symmetric near mean |
| Unit weight (γ) | Normal | Relatively small variability |
| SPT N-value | Lognormal | Count data, large variability |
| Loads (dead) | Normal | Central limit theorem |
| Loads (live) | Gumbel | Extreme value distribution |
| Earthquake magnitude | Gumbel | Extreme value |

### Distribution Parameters

**Normal**: Specified by mean μ and standard deviation σ.

**Lognormal**: pystra accepts mean and stdv of the **underlying normal** distribution, not the lognormal mean/stdv. To convert:
- Given lognormal mean m and COV v:
- λ = ln(m) - 0.5*ln(1 + v²)
- ζ = sqrt(ln(1 + v²))
- Use mean=exp(λ+ζ²/2), stdv as needed

For geotechnical work, typical COV values:
- Unit weight: 3-5%
- SPT N-value: 30-50%
- Undrained strength: 20-40%
- Friction angle: 5-15%
- Loads: 10-25%

**Constant**: Use for deterministic parameters (e.g., geometry).

## Correlation

Variables may be correlated (e.g., φ and c, or SPT N-values at different depths). Specify via correlation matrix:

```python
correlation = [
    [1.0, 0.3],  # Variable 1: perfect self-correlation, 0.3 with Variable 2
    [0.3, 1.0],  # Variable 2: 0.3 with Variable 1, perfect self-correlation
]
```

**Requirements**:
- Square matrix (n_variables × n_variables)
- Symmetric
- 1.0 on diagonal
- Positive definite (all eigenvalues > 0)

**Common correlations**:
- c and φ: -0.2 to -0.5 (negative: high φ → low c)
- Adjacent soil layers: 0.3 to 0.7
- Same property at different locations: 0.5 to 0.9 (decreases with distance)

## numpy 2.x Compatibility

pystra was developed for numpy 1.x and has bugs with numpy 2.x related to scalar/array handling. Three methods in `Transformation` class return arrays when scalars are expected:

1. **jacobian()**: Returns matrix instead of scalar for diagonal elements
2. **x_to_u()**: Array element extraction fails
3. **u_to_x()**: Array element extraction fails

**Solution**: Monkey-patch these methods to use `.flat[0]` for scalar extraction. The patches are applied in `_apply_numpy2_patches()` which is called at the start of each analysis function.

The patches are idempotent (safe to call multiple times) and only applied for numpy >= 2.0.

## Input Validation

### Security

Limit state expressions are compiled via `eval()` which is a security risk. Mitigations:
1. Restricted namespace (no __builtins__)
2. Whitelist of allowed identifiers (variable names + math functions)
3. Regex token extraction to detect unknown identifiers

**Allowed in expressions**:
- Variable names from the variables list
- Math functions: sqrt, log, exp, sin, cos, tan, pi, abs, min, max, etc.
- Operators: +, -, *, /, **, (), etc.

**Not allowed**:
- Import statements
- System calls
- File I/O
- Arbitrary function calls

### Engineering Checks

- At least one non-constant variable (else problem is deterministic)
- Correlation matrix validity (if provided)
- Limit state compiles successfully
- Variable names are valid Python identifiers

## Output Format

All result classes provide:
1. **summary()**: Human-readable text for reports
2. **to_dict()**: JSON-serializable dict for API/storage
3. **Plot methods**: Engineering-quality plots (FORM only has importance plot)

## Workflow Example

```python
from pystra_agent import analyze_form, analyze_sorm, analyze_monte_carlo

# Define problem: Bearing capacity check
# Ultimate capacity Qult = 5.14*c*Nc (Terzaghi for φ=0 clay)
# Factor of safety FS = Qult / Q
# Limit state: g = Qult - FS_target*Q

variables = [
    {"name": "c", "dist": "lognormal", "mean": 50, "stdv": 15},  # kPa
    {"name": "Nc", "dist": "constant", "value": 5.14},
    {"name": "Q", "dist": "normal", "mean": 200, "stdv": 40},  # kN
]

# g = 5.14*c*Nc - 1.5*Q (require FS ≥ 1.5)
limit_state = "c * Nc - 1.5 * Q"

# Run FORM
result_form = analyze_form(variables, limit_state)
print(result_form.summary())
print(f"Sensitivity to cohesion: {result_form.alpha['c']:.3f}")

# Run SORM for comparison (though this LSF is linear)
result_sorm = analyze_sorm(variables, limit_state)
print(f"FORM beta = {result_sorm.beta_form:.3f}")
print(f"SORM beta = {result_sorm.beta_breitung:.3f}")

# Verify with Monte Carlo
result_mc = analyze_monte_carlo(variables, limit_state, n_samples=100000)
print(f"MC beta = {result_mc.beta:.3f} (COV = {result_mc.cov_pf:.3f})")
```

## Testing Strategy

### Tier 1 Tests (No pystra required)
- Result dataclass instantiation, summary(), to_dict()
- Input validation (bad distributions, empty variables, invalid correlation)
- Foundry metadata (list_methods, describe_method)
- JSON serialization of empty results
- Plot smoke test (create axes, no show)

### Tier 2 Tests (Requires pystra)
- **Known analytical problems**:
  - R - S with R~N(200,20), S~N(100,30) → β ≈ 2.77, Pf ≈ 0.0028
  - Nonlinear: R² - S² with lognormals
- **Distribution tests**: Normal, lognormal, constant, correlation
- **Sensitivity factor validation**: Signs and magnitudes
- **SORM vs FORM**: Should agree for linear LSF, differ for nonlinear
- **Monte Carlo convergence**: Should approach FORM result with enough samples
- **Foundry integration**: Round-trip JSON through agent functions
- **3+ variable problems**: Test scaling

### Regression Tests
- Compare to literature values (e.g., Haldar & Mahadevan textbook examples)
- Consistency: FORM → SORM → MC should give similar β for same problem

## References

1. Haldar, A., & Mahadevan, S. (2000). *Probability, Reliability, and Statistical Methods in Engineering Design*. Wiley.
2. Melchers, R. E., & Beck, A. T. (2018). *Structural Reliability Analysis and Prediction* (3rd ed.). Wiley.
3. Phoon, K. K., & Kulhawy, F. H. (1999). Characterization of geotechnical variability. *Canadian Geotechnical Journal*, 36(4), 612-624.
4. Fenton, G. A., & Griffiths, D. V. (2008). *Risk Assessment in Geotechnical Engineering*. Wiley.

## Edge Cases and Gotchas

1. **Near-deterministic problems**: If all variables have very small stdv, β → ∞ and Pf → 0. pystra may fail to converge. Check COV before analysis.

2. **Correlated constants**: If correlation matrix includes rows/columns for constant variables, pystra may error. Remove constants from correlation or set their rows/cols to identity.

3. **Negative failure probabilities**: Can occur if SORM curvature correction is too aggressive. Fall back to FORM in such cases.

4. **Division by zero in LSF**: If limit state involves division (e.g., "R / S"), ensure S cannot be zero (use lognormal, not normal).

5. **Unstable design point**: If LSF has multiple local minima in U-space, FORM may converge to wrong design point. Try different starting points or use MC.

6. **Alpha interpretation**: α_i sign indicates resistance (+) vs load (-), but magnitude depends on all variables. A small |α_i| doesn't mean the variable is unimportant—it may be correlated with others.

7. **MC for rare events**: For Pf < 10^-6, crude MC becomes impractical. Use importance sampling or subset simulation (not in pystra).

8. **Unit consistency**: All variables must use consistent units. Mixing kPa and MPa will give wrong results. Follow project standard: SI with kPa.
