# INF-68: the site-response tools on pystrata 0.6 as well as 0.5

Branch `claude/inf-68-pystrata-06`, from master 8daed85. No formula, curve
or default changed; pystrata is a library, so the "print" here is its own
source, read in both versions (0.5.4 `pystrata/site.py`, 0.6.0
`pystrata/site.py`, `pystrata/factories.py`, `pystrata/propagation.py`).

## The API differences found

| What | pystrata 0.5.4 (Python 3.12) | pystrata 0.6.0 (Python >= 3.13) |
|---|---|---|
| Darendeli soil | `site.DarendeliSoilType(unit_wt, plas_index, ocr, stress_mean, strains)` | `factories.darendeli_soil_type(unit_wt, stress_mean, plas_index=, ocr=, strains=)` returning a plain `SoilType` |
| Menq soil | `site.MenqSoilType(unit_wt, uniformity_coeff, diam_mean, stress_mean)` | `factories.menq_soil_type(unit_wt, stress_mean, coef_unif=, diam_mean=)` (argument renamed) |
| Curve class | `site.NonlinearProperty(name, strains, values, param)` | `site.ModulusReductionCurve` / `site.DampingCurve(name, strains, values, limits)` |
| Interpolation limits | none | G/Gmax clipped to (0.001, 1), damping to (0, 0.49) |
| `SoilType(name, unit_wt, mod_reduc, damping)` | unchanged | unchanged (now unit-aware through pint; plain floats are taken as kN/m3) |
| **EQL `tolerance`** | relative change in **percent**, signed (`100 * max((prev - val) / val)`) | relative change in **decimal**, absolute (`max(abs((prev - val) / val))`) |
| Layer, Profile.auto_discretize, TimeSeriesMotion, calculators, outputs, `_calc_waves`, `profile.location`, `IterativeValue.previous/value` | as used | unchanged for every call the agent makes |

The tolerance change was not in the backlog item. Passing the 0.5 value
(`tolerance * 100`) to 0.6 asks for a 100 % change, so the iteration
stopped after one pass and the tool reported "not converged" (two tests
caught it on the first 3.13 run).

0.6's `factories` say they are ported from 0.5's
`ModifiedHyperbolicSoilType`: the Darendeli (2001) and Menq (2003)
constants, the Masing damping correction (c1, c2, c3), the
`maximum.accumulate` on damping and the default strains
`logspace(-6, -1.5, 20)` are character for character the same.
(0.5.4's Menq docstring says its default strains are `logspace(-4, 0.5)`;
the code uses the base class's `logspace(-6, -1.5)`, as 0.6 does.)

## The compatibility layer

`pystrata_agent/compat.py`, the one module that knows which API is
installed. It checks for `pystrata.site.DarendeliSoilType` (present in 0.5,
removed in 0.6), not the version string:
`darendeli_soil_type`, `menq_soil_type`, `linear_soil_type`,
`custom_soil_type` and `eql_tolerance(pystrata, fraction)` (fraction x 100
on 0.5, the fraction on 0.6). `eql_site_response._build_soil_type` and the
calculator call use it; `opensees_agent`'s curve check
(`tests/test_site_response_curves.py::TestCurves`) builds its pystrata
soils through it (opensees_agent's own code never imports pystrata; its
tests already import pystrata_agent). One small change on 0.5: the custom
curves now pass `param="mod_reduc"` / `"damping"` to `NonlinearProperty`
(pystrata uses `param` only for its variation tools; the answers are the
same to the last bit, below).

## Pins

`pyproject.toml`:
`"pystrata>=0.5,<0.6; python_version < '3.13'"`,
`"pystrata>=0.6,<0.7; python_version >= '3.13'"`,
`"pyrvt<0.9; python_version < '3.13'"` (on 3.13, pystrata 0.6 declares its
own pyrvt; 0.9.0 resolved). The same three strings in the install lines of
`.github/workflows/tests.yml` and `nightly.yml` (double quotes around each,
single quotes inside; the YAML parses and pip / uv resolve them: 0.5.4 and
pyrvt 0.8.1 on 3.12, 0.6.0 and pyrvt 0.9.0 on 3.13).
`webapp/version_guard.py` covers only the agent stack and names neither
package, so `test_version_guard.py` needed no change.

## The check: pystrata 0.5.4 is the reference

Venvs: Python 3.12.3 + pystrata 0.5.4 + pyrvt 0.8.1 + numpy 2.5.4
(`pip install -e .`); Python 3.13.18 + pystrata 0.6.0 + pyrvt 0.9.0 +
numpy 2.5.4 + pint 0.26.2 (`uv venv --python 3.13`, `uv pip install -e .`).
Cases: the agent's own test profiles (Darendeli PI 30 over rock, Menq, a
Darendeli/Menq/linear stack, the custom curve, a Darendeli-over-Menq
column) with the built-in synthetic pulse, at its own PGA and scaled to
0.05, 0.2 and 0.6 g, equivalent-linear, plus two linear-elastic runs.
Script: `module_work/inf68/compare_versions.py`.

| case | amp 0.5.4 | amp 0.6 | diff % | compatible Vs, max diff % | iterations 0.5.4 / 0.6 | converged | max strain diff % | peak Sa diff % |
|---|---|---|---|---|---|---|---|---|
| darendeli_10m_pi30 | 1.4496 | 1.4496 | -0.000 | 0.000 (n=6/6) | 6 / 6 | True / True | +0.000 | -0.000 |
| darendeli_10m_pi30_0.05g | 1.1826 | 1.1826 | +0.000 | 0.000 (n=6/6) | 4 / 4 | True / True | +0.000 | -0.000 |
| menq_8m | 1.0795 | 1.0795 | +0.000 | 0.000 (n=4/4) | 6 / 6 | True / True | -0.000 | +0.000 |
| menq_8m_0.6g | 1.1842 | 1.1842 | +0.000 | 0.000 (n=4/4) | 8 / 8 | True / True | +0.000 | +0.000 |
| dar_menq_lin_15m | 1.2465 | 1.2465 | -0.000 | 0.000 (n=7/7) | 7 / 7 | True / True | -0.000 | +0.000 |
| custom_12m | 1.8588 | 1.8588 | -0.000 | 0.000 (n=10/10) | 4 / 4 | True / True | -0.000 | +0.000 |
| dar_over_menq_0.2g | 2.2159 | 2.2159 | +0.000 | 0.000 (n=9/9) | 11 / 11 | True / True | +0.000 | +0.000 |
| linear:darendeli_10m_pi30 | 1.1624 | 1.1624 | +0.000 | 0.000 (n=6/6) | 0 / 0 | True / True | +0.000 | +0.000 |
| linear:dar_menq_lin_15m | 1.1547 | 1.1547 | +0.000 | 0.000 (n=7/7) | 0 / 0 | True / True | +0.000 | +0.000 |

Largest relative difference over amplification, surface PGA, peak Sa,
peak strain and every compatible-Vs value: **1.5e-15** (round-off). The
branch on 0.5.4 gives results identical to master on 0.5.4 (every field of
every case equal), so **no answer changes on Python 3.12**.

Before the change, master's code on Python 3.13 / pystrata 0.6.0:
**20 failed**, 267 passed, 1 skipped (`pystrata_agent opensees_agent`;
`AttributeError` on the removed classes). With the classes fixed but the
tolerance not: 2 failed (converged False after one pass).

## Tests (pytest exit code 0 on each)

| Suite | Python 3.12 / pystrata 0.5.4 | Python 3.13 / pystrata 0.6.0 |
|---|---|---|
| `pystrata_agent` (incl. new `test_compat.py`, 13 tests) | 75 passed, 1 skipped | 75 passed, 1 skipped |
| `opensees_agent` | 225 passed | 225 passed |
| the above + funhouse_agent files naming them (`test_method_name_aliases`, `test_module_followup_adapters`, `test_nested_keys`, `test_new_adapters`, `test_optional_import_errors`, `test_reviewers`, `test_top_level_params`, `deep/tests/test_eval_scoring`, `deep/tests/test_wave1_geotech`) + `webapp/tests/test_page_fingerprints.py`, `test_public_text.py`, `test_version_guard.py` | 820 passed, 1 skipped | 820 passed, 1 skipped |
| `funhouse_agent/tests/test_defaults_used.py -k "pystrata or opensees"` | 9 passed | 9 passed |

The one skip is `test_import_pystrata_error_message` ("pystrata is installed;
can't test error path"). No tool description or prompt changed, so the
page fingerprints did not move and were not re-recorded.
`test_compat.py` pins the Darendeli and Menq curves to the formulas written
out in the test (1e-12), the tolerance unit per API, and four
equivalent-linear runs to the 0.5.4 amplification, iteration count and
lowest compatible Vs (1 %, INF-68's criterion), so CI's 3.12 and 3.13 jobs
both hold 0.6 to the 0.5.4 answers.

## Left

- The released 5.34.0 still says `pystrata>=0.5` with no pyrvt cap; the
  markers ship with the next release (the owner's call, as INF-68 says).
- Moving every host to 0.6 waits on the hosts reaching Python 3.13
  (and numpy >= 2.2; Foundry's transforms repos hold numpy 1.26).
- Noted, not fixed: 0.6 clips interpolated curve values to (0.001, 1) and
  (0, 0.49). A custom curve whose own points lie outside those ranges would
  read differently on 0.6; the agent does not check custom points against
  them. No saved example has such a curve.
- Noted, not fixed: `pystrata.__version__` on 0.5.4 reports pyrvt's version
  (0.8.1); nothing in the app reads it.

## Release note

The site-response tools (`pystrata_agent` equivalent-linear and linear
runs, and `opensees_agent`'s curve checks) now install and run on Python
3.13 with pystrata 0.6, as well as on Python 3.12 with pystrata 0.5.
No answer changes on Python 3.12; on 3.13 the Darendeli, Menq and custom
runs give the 0.5.4 answers to round-off.
