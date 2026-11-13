# GM-322 and GM-323: reliability, pystra, salib

Branch `claude/gm-322-323-reliability-stack`, from master 2feea3d. Modules: `reliability`, `pystra_agent`, `salib_agent` and their three adapters. Source: BACKLOG GM-322 and GM-323; the audit's section 4 (`module_work/module_audit_2026-11-12/AUDIT.md`), rows reliability #1 to #3, pystra #2 to #5 and #7, salib #1 to #3 and #5.

Every finding was confirmed on today's master before any change, by reading the cited lines and then by running the case. Two things were found while running that the audit (which ran nothing) could not see; they are marked "found here" in the table.

## What the print says

The held text is `geotech_references/dm7_2/text/chapter07.json`, which is UFC 3-220-20 (2025), chapter 7.

- **FOSM, Sec. 7-4.2.1.** "The first-order, second moment (FOSM) method uses only the mean and variance of the random variables to define beta." The mean of g is g at the means (Eq. 7-8). The variance comes from central differences one standard deviation above and below each mean (Eqs. 7-10, 7-11). The lognormal index for a safety-factor limit state is Eq. 7-12.
- **Point estimate method, Sec. 7-4.2.2.** g is "evaluated for 2^n cases with each RV either one standard deviation above or below the mean", for "n symmetric RV" (Eqs. 7-13, 7-14; Figures 7-12, 7-13).
- Neither procedure has a step that reads a truncation, and both are stated for the variable's own mean and standard deviation. The module's `mean` and `std` on a truncated variable are the moments of the UNTRUNCATED distribution. So running FOSM or PEM on a truncated variable answers for a different variable. That is the basis for the refusal.
- **Monte Carlo, Sec. 7-4.2.4.** The section gives the number of trials needed for a small probability and names Latin hypercube sampling. It prints no rule for a trial whose g cannot be evaluated, and no confidence interval formula. **No held reference prints a limit on dropped trials**, so the 5 % threshold is a call (below), not a cited value.
- **pystra and salib** cite no printed manual (the audit says the same). Nothing in those two modules was changed on engineering grounds: every change there is the wrapper reporting what the library returned, or plain counting (rows of a design).

## Findings taken

| Audit row | Finding | Confirmed? | What was done |
|---|---|---|---|
| reliability #1 (A) | FOSM and PEM ignore `lower`/`upper` on a normal or lognormal variable, which the tool said truncate | Yes | **Refused.** `reliability.variables.refuse_truncated`, called by `fosm()` and `pem()` (both PEM schemes). The message names the variable and says monte_carlo and form take a truncation. Uniform and triangular bounds are still accepted (they define the distribution, and its mean and std follow from them). Each engine's `variables` description now says what that engine does with `lower`/`upper`. |
| reliability #2 (A) | Monte Carlo drops non-finite g with no count | Yes | **Fixed.** The result carries `n_requested`, `n` (kept), `n_dropped`, `dropped_fraction`, `pf_bounds_dropped` and a line in `warnings`. Above 5 % dropped the run is **refused** with the count, the share and the first offending point. |
| reliability #3 (B, same function) | With zero failures the interval is exactly (0, 0) | Yes | **Fixed at the edge.** With no failure, or only failures, the interval is the exact binomial bound (0 to 1 - 0.025^(1/n)); `pf_ci95_method` names the formula used; a warning says pf = 0 is a count. Between the edges the interval is unchanged. |
| found here | `monte_carlo`'s `returns` listed `pf_normal`, which the result never carried | Yes | **Fixed** (text): removed from that method's returns. A test checks every promised key is in the result. |
| found here | An arithmetic failure of g at one realization (division by zero, `math domain error`, a complex value) ended the whole run with a bare `ValueError: math domain error` | Yes | **Fixed.** It counts as a dropped realization, as a NaN does. Any other exception from g is still raised. Without this, `n_dropped` could almost never be non-zero through the agent, since the expression compiler raises instead of returning NaN. |
| pystra #2 (A) | A failed SORM returns the FORM values under the SORM names | Yes. Real case: mean point in the failure domain (beta < 0), where Breitung's formula is undefined | **Fixed.** `beta_breitung` and `pf_breitung` are null, with `sorm_failed`, `sorm_converged`, `sorm_message` and a warning. pystra 1.x marks the same failure by setting both to exactly 0; the wrapper reads that marker. |
| pystra #4 (A) | `converged` is `beta > 0 and pf > 0` | Yes | **Fixed.** pystra's own flag on 2.x. On 1.x, which keeps none, "stopped before the iteration limit" (its own stopping rule); `converged_source` says which. |
| pystra #5 (A) | alpha, design point and curvatures fall back to `{}` / `[]` on any exception | Yes | **Fixed.** No blanket `except`. A quantity pystra did not return, or returned at the wrong length, is named in `warnings`. An unconverged FORM on 2.x has no beta, pf or design point (pystra gives none): they are null, not invented. |
| pystra #3 (A, closed as wording by DOC-24) | `n_function_calls` is an estimate | Yes | **Fixed further:** pystra's own count on 2.x; the estimate on 1.x, with `n_function_calls_counted` false and a warning. |
| pystra #7 (B) | The seed comes from numpy's global generator; `n_failures` and `cov_pf` | Yes | **Fixed.** `seed` is an input, local to the call; the global generator is neither read nor left changed (1.x samples from it, so its state is saved and restored). Declared when absent. With no failure, `beta` and `cov_pf` are null and `pf_upper95_no_failures` is given. |
| found here | pystra's Monte Carlo stops once the COV of pf reaches 0.05, so fewer samples were run than the `n_samples` reported, and `n_failures` was pf times the count asked for | Yes, on 2.0.0 and 1.6.0 | **Fixed.** Every sample asked for is run (the early stop is off); `n_samples` is the count run and `n_samples_requested` the count asked for. See "Calls made". |
| found here | pystra hands the limit state arrays of points, so any limit state with a math function or a conditional (`sqrt`, `max`, `a if c else b`) failed on every call with "only 0-dimensional arrays can be converted" | Yes, on 2.0.0 and 1.6.0 | **Fixed** (a few lines in `_compile_limit_state`): such an expression is evaluated one point at a time. Beyond the listed findings; taken because the SORM test needed a curved limit state and the failure was total. |
| salib #1 (B) | `sobol_analyze` never reads `n_samples` | Yes | **Checked against the design.** Y must have N*(2D+2) values, or N*(D+2) without second-order indices. `n_samples`, when given, must equal that N or the call is refused; left out, N is read from Y. The result adds `n_base_samples`. |
| salib #2 (B) | `morris_analyze` copies `n_trajectories` unchecked | Yes | **Checked against the design.** X must be T*(D+1) rows of D columns; the result's `n_trajectories` is counted from X; a value that disagrees is refused. |
| salib #3 (B) | `_validate_model_output` is never called | Yes | **Fixed.** Called for Morris (Y against the rows of X). For Sobol the design check does the same job. |
| salib #5 (B) | The expression compiler has no power or nesting limit | Yes | **Fixed.** 10,000 characters, 200 levels, every power through `_bounded_pow`. The numbers are restated in `salib_agent/expression.py` (an analysis module imports no other module's code) and a test pins them to `aec_agent_core.safe_expr` and `pystra_agent.pystra_utils`. |

Not taken (outside this section's scope): reliability #4 (the wrappers no tool reaches), pystra #9 and #10, salib #8 (calc packages, published examples). pystra #1 was closed by DOC-24.

## Calls made

For the owner to revisit.

1. **The dropped-realization limit is 5 %.** No held reference prints one. Reasoning: each dropped realization is one whose outcome is unknown, so the share dropped is the width of the band pf can lie in. Up to 5 % the band is reported with the result. Above it the samples kept describe only the part of the variables' range where g evaluates, and the band is wider than any 95 % interval the result quotes. A stricter limit (1 %, or any drop at all) is defensible; `MAX_DROPPED_FRACTION` in `reliability/monte_carlo.py` is the one number to change.
2. **An arithmetic failure of g counts as a dropped realization** (division by zero, overflow, math domain error, complex value). The other choice was to keep ending the run on the first one.
3. **The interval at zero failures is the Clopper-Pearson bound; elsewhere it stays the normal approximation.** The other choice was the Clopper-Pearson interval everywhere, which would move every reported interval slightly. Two forms are now in use, and `pf_ci95_method` says which.
4. **pystra Monte Carlo runs every sample asked for.** The other choice was to keep pystra's early stop and report the count run. The tool says "number of Monte Carlo samples", so the code now does that. A run can take up to about 1.6 times as long as before.
5. **pystra 1.x `converged`** is "stopped before the iteration limit", since 1.x keeps no flag. A run that met its tolerances on its very last allowed iteration reads as not converged.
6. **A Sobol output count that fits both designs** (a multiple of both 2D+2 and D+2) cannot be told apart by the check. `calc_second_order` must still match the sampling; giving `n_samples` removes the ambiguity. Stated in the tool text and `salib_agent/DESIGN.md`; since the review round the result's `warnings` and `choices` say so too.
7. **FOSM and PEM read bounds that cut off at most 1e-6 of a variable's probability as no truncation** (the lead's call in the review round, with review finding 7). The mass is both tails together, per variable (my reading of "in all"). Such a run goes ahead on the untruncated variable, and its `warnings` give the mass and say the bounds were not applied. Above 1e-6 the call is refused as before. `TRUNCATION_IGNORED_MASS` in `reliability/variables.py` is the one number to change.

## Before and after

All values made up.

**The truncated case through four engines** (`reliability/tests/test_truncation_engines.py`). F = R / S, failure at F < 1. R normal, mean 200, std 40, truncated to [130, 270]. S lognormal, mean 100, std 20.

| Engine | Untruncated R | Truncated R, before | Truncated R, after |
|---|---|---|---|
| Monte Carlo (n = 400,000, seed 7) | pf 0.0138 | 0.0036 | 0.0036 (unchanged) |
| FORM | pf 0.0126 | 0.0049 | 0.0049 (unchanged) |
| FOSM (lognormal index) | pf 0.0105 | 0.0105, the untruncated answer, unmarked | refused |
| PEM (lognormal index) | pf 0.0066 | 0.0066, the untruncated answer, unmarked | refused |

Monte Carlo and FORM both read the truncation and move the same way (pf falls by a factor of 2.6 to 3.8). They differ from each other by a factor of 1.4: FORM is first order and the truncated variable's transform is strongly curved near the bound. FOSM and PEM used to return an answer about three times and twice the truncated Monte Carlo pf.

**Monte Carlo, dropped realizations.** x normal (10, 1), g = x - 8.5 and NaN where x > 12, n = 4,000, seed 12.

- Before: `n` 3904, pf 0.0684, nothing else said.
- After: the same pf, plus `n_requested` 4000, `n_dropped` 96, `dropped_fraction` 0.024, `pf_bounds_dropped` [0.0668, 0.0908] and a warning.
- With NaN where x > 10 (49.7 % dropped): before, a pf on 2,013 samples; after, refused.

**Monte Carlo, no failures**, n = 10,000: the interval was (0, 0); it is (0, 0.000369).

**pystra SORM, mean point in the failure domain.** R normal (80, 10), S normal (100, 10), limit state R - S.

- Before: `beta_breitung` -1.4142 and `pf_breitung` 0.9214 (FORM's values), `converged` false, no flag.
- After: both null, `sorm_failed` true, `sorm_message` "The breitung formula is undefined for the fitted curvatures", `form_converged` true. `form_analysis` on the same case: `converged` was false, is true.

**pystra Monte Carlo, early stop.** R normal (200, 20), limit state R - 150, `n_samples` 100,000.

- Before: pystra ran 61,000 samples and found 400 failures (pf 0.006557); the result said `n_samples` 100,000 and `n_failures` 656.
- After (seed 1): 100,000 run, 673 failures, pf 0.00673, `cov_pf` 0.038.

**pystra SORM against sampling, a curved limit state.** R normal (200, 20), S lognormal (100, 30), R - 14 sqrt(S). Before: the call failed (math function). After: FORM pf 0.01861, SORM pf 0.01953, Monte Carlo (200,000, seed 1) 0.0195.

**salib.** A Morris sample of 6 trajectories analysed with `n_trajectories` 20: before, the result said `n_trajectories` 20; after, refused ("20 trajectories of 3 variables are 80 rows, and X has 24 (6 trajectories)"). `floor(a)**floor(a)**floor(a)` at a = 9: before, the interpreter held for minutes; after, refused at once.

## Tests run

Python 3.12 venv, pystra 2.0.0, SALib 1.6.0, numpy 2.5.4. Judged by pytest's exit code.

| What | Result |
|---|---|
| `reliability` | 192 passed |
| `pystra_agent` (pystra 2.0.0) | 66 passed |
| `pystra_agent/tests/test_gm322_results_say_what_they_are.py` on pystra 1.6.0 with numpy 1.26.4 (a second venv) | 21 passed |
| `salib_agent` | 59 passed |
| `funhouse_agent/tests/test_gm322_323_adapters.py`, `test_reliability_adapter.py` | 16 and 29 passed |
| The wide run: `funhouse_agent/tests`, `funhouse_agent/deep/tests`, `validation_examples`, `calc_package`, the three modules, `webapp/tests/test_page_fingerprints.py`, `test_public_text.py`, `test_docs_currency.py`, `test_backlog.py` | 6,160 passed, 2 failed, 39 skipped, 1 xfailed. Both failures are on untouched master too (below) |
| `module_work/doc10/check_manual.py` | OK |

`webapp/tests/test_page_fingerprints.py` passed without a re-record, and `python -m webapp.tests.page_builds record` wrote no change: these tool descriptions are not part of the page fingerprints. So there is no fingerprint commit.

Two tests failed in the cloud container, on this branch and on untouched master (2feea3d) alike. Both are figure tests in other modules: `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` and `funhouse_agent/tests/test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`. The reviewer ran both on the owner's machine, on base and on the branch: they pass on both. They are the container's failures (most likely its matplotlib or fonts), not the branch's or master's.

## Consumers checked

`grep` over the whole repo for the engines, the result keys and the changed parameters.

- `reliability.fosm` / `pem` are called by the reliability adapter, the reliability calc package (through the adapter's `run_*`) and `docs/gallery/build_gallery.py`. None passes a truncated variable. `slope_stability` has its own probabilistic code and does not import this module.
- No probe, worked example, validation example or manual example relied on FOSM or PEM with a truncation, on a dropped Monte Carlo realization, or on a mismatched salib count. The existing salib probes already matched their designs (48 = 8 x 6; 6 = 2 x 3).
- `funhouse_agent/tests/method_probes.json`: the pystra Monte Carlo probe now gives `seed` 1, so its result repeats.
- The reliability calc package's Monte Carlo step states the interval's formula and any dropped realizations.
- User manual: 15 rows hand-replaced (overview tables, parameter tables, method headings for the three modules). No rebuild.

## Release note

**reliability.** FOSM and PEM now refuse a normal or lognormal variable whose `lower`/`upper` cut off more than 1e-6 of its probability; they used to run on the untruncated moments without saying so. Use Monte Carlo or FORM for a truncated variable, or remove the bounds. Bounds that cut off nothing (infinite bounds, a lognormal with `lower=0`) run as before; bounds that cut off at most 1e-6 run on the untruncated variable, and `warnings` says how much they cut off. Monte Carlo reports how many realizations had a non-finite g and refuses to give a probability when more than 5 % did; a run that used to end on a division by zero or a domain error at one realization now counts it instead. With no failures (or only failures) the 95 % interval is the Clopper-Pearson bound, no longer 0 to 0. The reliability calc package's result gives the Monte Carlo counts (run, kept, dropped) and warnings its note prints. No pf, beta or moment changes for a run without truncation and without dropped realizations.

**pystra.** Monte Carlo now runs every sample asked for, so pf, `n_samples` and `n_failures` change for any run that used to stop early at a 5 % COV; it takes a `seed`, and without one the result differs from run to run. On pystra 2.x a seeded run saved before this release cannot be repeated with the same seed: the seed now goes to pystra directly, where it used to be drawn from numpy's global generator (on 1.x it repeats). With no failures, or with every sample failed, `beta` and `cov_pf` are null. A failed SORM returns null second-order values and `sorm_failed`, where it used to return FORM's values; an unconverged FORM runs no SORM on either pystra version. `converged` is pystra's own flag: a FORM run with beta < 0 that converged now says so. Limit states with math functions or conditionals now run.

**salib.** `sobol_analyze` and `morris_analyze` refuse a count that does not fit the data, and refuse data that is not a whole design. A Sobol Y that fits both designs, with `n_samples` left out, still runs and now says in `warnings` and `choices` which design was taken. `morris_analyze`'s `n_trajectories` in the result is counted from X. No index changes for data that fits its design.

## Noted, not fixed

- `pystra_agent/DESIGN.md` gives "typical" and "target" reliability indices with no citation. Not in this item.
- `reliability`'s docstrings cite UFC 3-220-20 "pp. 571-574" for Eqs. 7-8 to 7-14; the library's figure catalog puts Figure 7-12 on printed page 536 (PDF page 573). The cites look like PDF page numbers. This file cites sections and equations.
- The reliability worked example in `docs/user_manual/worked_examples.json` was captured before this change; its printed output lacks the new Monte Carlo keys. The values it shows are unchanged.
- pystra's `summary()` texts and the legacy `foundry/` wrappers were not re-read beyond what the tests cover.
- The two-step salib methods describe their parameters under `brief`, not `description`, so the manual shows no text for them. Left as found.

## Review round

An independent review (scratch `review_gm322.md`, 14 findings) and the lead's rulings. Branch `fix/gm322-review` from a694dc5.

**Merge of master.** `origin/master` merged with no conflict (`manual.html` and `method_probes.json` merged on their own; `check_manual.py` OK). The test loop below passed on the merge alone, before any fix.

| # | Finding | What it became | Before → after |
|---|---|---|---|
| 1 (A) | FOSM/PEM refused bounds that cut off nothing | Refuse only bounds that cut off probability (`_cut_mass`, upper tail from the survival function). Tests for infinite bounds, lognormal `lower=0`, the far tail and a real cut | infinite bounds and lognormal `lower=0`: refused → run, pf_lognormal 0.01051361 (FOSM) and 0.00655827 (PEM), the base's numbers |
| 7 (C) | far-tail bounds refused | With 1: at most 1e-6 cut off runs untruncated with a warning (call 7) | normal 200/40, `lower=0` (2.9e-7 cut off): refused → base's numbers plus a warning giving 2.9e-07; `lower=20` (3.4e-6): refused, the message gives the mass |
| 2 (B) | pystra 1.x fitted SORM after an unconverged FORM | 1.x branch takes 2.x's early return. Test on pystra 1.6.0 | `(R-190)**2+1`: `sorm_message` "TypeError: len() of unsized object" → "SORM was not run: … FORM did not converge"; no Breitung value either way |
| 3, 11 (B, C) | "exact" for the interval on pf and FORM's beta | "Clopper-Pearson bound" in the tool text, warnings, the package note, docstrings and both DESIGN files; `pf_ci95_method` values `exact_binomial_*` → `clopper_pearson_*` (the branch introduced them; consumers: `monte_carlo.py` and its test only). FORM brief: "beta is the first-order answer; it is the assumed model's own for a linear limit state of normal variables". Manual rows rebuilt | no number moves; the package page with no failures is one byte longer |
| 4 (B) | Sobol Y fitting both designs read silently | Still runs; a warning and a `choices` entry name the design taken, the other, and that `n_samples` settles it. Test D=3, N=64 | 320 values, `calc_second_order` true: N=40 both before and after, now with the warning |
| 5 (B) | package result left out the Monte Carlo drop | `n_requested`, `n_dropped`, `pf_bounds_dropped`, `pf_ci95`, `pf_ci95_method`, `warnings` added (and FOSM/PEM's warnings). Test | dropped case: pf on 3904 of 4000 now carries n_dropped 96 and bounds 0.06675 to 0.09075 |
| 6 (C) | drops matched on "expected a" | Not changed. The words are Python 3.14's own `math` messages ("expected a nonnegative input, got -1.0" for `sqrt(-1)`, "expected a positive input" for `log(0)`), raised as a plain `ValueError`; `aec_agent_core.safe_expr` raises no distinct type for them, so there is no narrower type to catch | — |
| 8 (C) | FORM "samples" the truncated distribution | "reads", in `_TRUNCATION_READ` and the refusal | — |
| 9 (C) | every sample failed: `cov_pf` 0.0 | `cov_pf` None alongside `beta`. Test | pf = 1: cov_pf 0.0 → None |
| 10 (C) | Breitung's limits | `pystra_agent/DESIGN.md`: undefined where 1 + beta·kappa ≤ 0; rejecting beta < 0 is pystra's own rule | — |
| 12 (C) | Duncan 2000 not in the library | Not this round (the lead's word) | — |
| 13 (C) | seeded pystra 2.x runs | Release note says a saved seeded 2.x run cannot be repeated with the same seed | — |
| 14 (C) | "failing on master too" | Tests run section corrected: the two failed in the cloud container only | — |

**Checks.** pystra 1.6.0, SALib, Python 3.14 on the owner's machine. The test loop, one line per file: the seven test files the branch touched, `test_probe_warnings.py`, `test_allowed_values.py`, `test_adapter_ergonomics.py`, `test_citations.py`, `test_calc_package_tool_parity.py`, `test_defaults_used.py -k "reliability or pystra or salib"`, `test_page_fingerprints.py`, `test_public_text.py`, plus `salib_agent/tests/test_salib_agent.py` after the fixes: all passed, 965 tests (the same loop without the salib file also passed on the merge alone). `reliability/tests/test_fosm.py` and `test_pem.py` passed too. `check_manual.py` OK (the old soe `* D_embed` line left as it was). No fingerprint re-record was needed. The reviewer's probes re-run on the fixed tree: of 80 cases, the six FOSM/PEM bound cases now run with the base's numbers, the Monte Carlo package cases gain the keys above, and no number moved for a call that was right (the other differences are timings).

**Calls made.** Call 7 above (the 1e-6 is the lead's; "per variable, both tails together" is my reading). The interval at no failure is called the upper end of the "two-sided 95 % Clopper-Pearson interval" where one number is given (pystra), and the "Clopper-Pearson bound" where the pair is.
