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
