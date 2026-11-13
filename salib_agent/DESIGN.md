# SALib Agent — Design Notes

## Purpose

Wraps the SALib (Sensitivity Analysis Library) for identifying which input
parameters most influence geotechnical model outputs:

- **Sobol** — variance-based global sensitivity analysis (quantitative)
- **Morris** — elementary effects screening (qualitative, faster)

## Architecture

```
salib_agent/
  __init__.py          # exports sample + analyze functions + result classes
  salib_utils.py       # has_salib(), import functions
  sensitivity.py       # sobol_sample/analyze, morris_sample/analyze
  results.py           # SobolResult, MorrisResult
  tests/
    test_salib_agent.py
  DESIGN.md
salib_agent_foundry.py  # Foundry wrapper (project root)
```

## Key Design Decisions

1. **Two-step workflow** — Sampling and analysis are separate functions:
   - Step 1: Generate sample matrix (sobol_sample / morris_sample)
   - Step 2: User evaluates their model at all sample points
   - Step 3: Pass outputs to analyzer (sobol_analyze / morris_analyze)

   For the Foundry agent, built-in test functions (Ishigami, linear)
   are provided so the LLM can run complete analyses without external
   model evaluation.

2. **Problem definition** — SALib uses a `problem` dict with `num_vars`,
   `names`, and `bounds`. Our wrapper takes `var_names` and `bounds`
   separately for a cleaner API.

3. **Sobol not Saltelli** — `SALib.sample.saltelli` is deprecated since
   SALib 1.5; we use `SALib.sample.sobol` instead.

## Geotechnical Applications

1. **Bearing capacity** — Which parameters (phi, c, B, D, gamma) most
   affect bearing capacity? Sobol S1 ranks them quantitatively.

2. **Slope stability** — Screen 10+ parameters with Morris to find the
   3-4 that matter, then run detailed Sobol on those.

3. **Settlement** — Sensitivity of predicted settlement to soil modulus,
   layer thickness, OCR, Cc/Cr values.

4. **Pile design** — Which soil layers and parameters drive pile capacity?

## Interpreting Results

### Sobol Indices
- **S1** (first-order): fraction of variance explained by each variable alone
- **ST** (total-order): includes interaction effects (ST >= S1)
- **S1 close to ST**: variable acts mainly alone
- **ST >> S1**: variable has strong interactions with others

### Morris Indices
- **mu*** (mu-star): mean of |elementary effects| → importance
- **sigma**: std of elementary effects → nonlinearity/interactions
- **High mu*, low sigma**: important, linear/additive effect
- **High mu*, high sigma**: important, nonlinear or interacting

## Edge Cases

- **Correlated inputs**: SALib assumes independent inputs. Correlated
  geotechnical parameters (e.g., phi and c) need transformation first.
- **Sample size**: Sobol needs N*(2D+2) evaluations; with many variables,
  Morris screening first reduces dimensionality.

## Counts are checked against the design (GM-323)

A sample matrix fixes how many outputs an analysis must be given, so the
two-step methods check the data against the design instead of taking a
count on trust:

- `sobol_analyze`: a Saltelli/Sobol design for D variables has N*(2D+2)
  rows with second-order indices and N*(D+2) without. The length of Y must
  be a whole multiple; N is read from it (`n_base_samples` in the result;
  `n_samples` stays the number of outputs analysed). `n_samples`, when
  given, must equal that N or the call is refused. Until GM-323 it was
  never read.
- `morris_analyze`: a Morris design has T*(D+1) rows of D columns, one
  output per row. X and Y are checked, T is counted from X and returned as
  `n_trajectories`; an `n_trajectories` that disagrees is refused. Until
  GM-323 the given value was copied into the result unchecked.
- One ambiguity the check cannot see: a length that is a whole multiple of
  both 2D+2 and D+2 fits either design, so `calc_second_order` must still
  match the sampling. Giving `n_samples` removes the ambiguity.

## Expression limits (GM-323)

`salib_agent/expression.py` takes the reliability engines' limits (the same
numbers as `aec_agent_core.safe_expr` and `pystra_agent.pystra_utils`): at
most 10,000 characters, at most 200 levels of nesting, and every power
through `_bounded_pow`, which refuses an exact integer result above about
1e308 before computing it. The numbers are restated here because an
analysis module imports no other module's code;
`salib_agent/tests/test_gm323_counts_and_limits.py` pins the three copies
together.
