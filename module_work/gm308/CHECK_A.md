# GM-308 (a) (fem2d): a faster iteration for footing capacity on frictional soils

Branch `claude/gm-308-footing-iteration`, from master 9ae6560. All values
are made up. B = 2 m unless given, T6, 40x20 (the default), domain 10B x
5B, E = 1e5 kPa, nu = 0.3, psi = 0 unless given, defaults otherwise.

> **Lead's call, 2026-11-06: the option's code is not merged.** After the review fixes the option still reads far below the
> default on four of the eleven cases on Linux and differs by machine near collapse, so under the simple-baseline rule (add code
> only where it beats the plain method) only these notes and `results.jsonl` are kept on master. The code, its tests and the
> DESIGN/VALIDATION text stay on branch `claude/gm-308-footing-iteration` (head 0e8f602). Section 7's convergence-test idea is BACKLOG GM-308 (c).

**Verdict: no scheme met the goal on every case, so no default changed.**
The accelerated iteration (Anderson acceleration of the constant-stiffness
iteration) ships as an option, `analyze_footing_capacity(iteration=
'accelerated')`, a library input only; the tool, its description and its
parameters are unchanged, and every saved answer is unchanged.

## 1. The scheme and its source

No geotechnical manual in geotech-references prints a nonlinear solution
scheme (the 321 chapter JSON files searched for "BFGS", "quasi-Newton",
"Anderson acceleration", "initial stiffness method", "modified Newton",
"Newton-Raphson": no hit; "secant" hits only a secant modulus), so the
sources are the numerical-analysis papers, as the brief allows:

- **Anderson acceleration (shipped as the option).** D. G. Anderson (1965),
  "Iterative procedures for nonlinear integral equations", J. ACM 12(4),
  547-560; in the form of H. F. Walker & P. Ni (2011), "Anderson
  acceleration for fixed-point iterations", SIAM J. Numer. Anal. 49(4),
  1715-1735, Sec. 2 (Anderson(m), mixing 1). The constant-stiffness
  iteration is the fixed-point iteration u_k+1 = u_k + f_k with
  f_k = K0^-1 R(u_k) (K0 the one factorized elastic stiffness). With the
  differences dU_i = u_i+1 - u_i and dF_i = f_i+1 - f_i of the last m
  iterates of the load step: gamma = argmin ||f_k - dF gamma||,
  u_k+1 = u_k + f_k - (dU + dF) gamma. It keeps the one factorization
  (one back-substitution per iteration plus an m-column least squares) and
  needs no tangent. Code: `fem2d/solver.py::_anderson_step`,
  `run_nl(method='anderson')`; m = 5 (`_ANDERSON_MEMORY`, this module's
  choice).
- **BFGS (tried first, dropped).** H. Matthies & G. Strang (1979), "The
  solution of nonlinear finite element equations", Int. J. Numer. Methods
  Eng. 14, 1613-1626: BFGS updates of K0^-1 with a line search on
  d . R(u + s d). Prototyped (two-loop product form with H0 = K0^-1, line
  search tolerance 0.5, secant rule); not committed.

`fem2d/DESIGN.md` (footing section, references) cites both.

## 2. What was tried, with the numbers

The goal: on the default mesh, each frictional answer at the default
max_iter (1000) within 1 % of its own answer at 4x max_iter (4000), and no
lower than the default iteration's answer at 16000; Prandtl (c = 50,
phi = 0) within 0.5 % of (2 + pi) c = 257.1.

### 2.1 BFGS (Matthies & Strang 1979)

c = 20, phi = 25, gamma = 18: 499.0 kPa at max_iter 1000 (19 s), 542.4 at
4000 (65 s), 520.7 at 16000 (102 s). The 4000 run accepted step 25 as
settled with a residual ratio of 0.66 (the plain iteration's "still
falling" clause of the settled test); the residual swung from about 1e-2
to above 1 within a step. With psi < phi the flow is non-associated and
the residual is not the gradient of a potential, which the line search
assumes. Memory 5, 60 and 300 and one to four line-search trials did not
change the 1000-iteration answer. Prandtl: 265.1 on 40x20 (see 2.4).

### 2.2 Anderson, m = 10, then m = 5 (the option as shipped, m = 5)

c-phi with m = 10: 520.7 kPa at max_iter 1000 (13 s), 4000 (27 s) and
16000 (81 s; with m = 5: 520.7, 74 s), the failing step's displacements running away (growth ratio
2.1 at 1000, against a standstill for the plain iteration). m = 5 and 20
gave the same 520.7 at 1000. But with m = 10, c = 0, phi = 30, q = 10
stalled at a residual near 1e-3 and accepted two stalled steps (195.9,
below the default's 216.6); m = 3 and 5 read 278.6 with a real mechanism.
So m = 5.

GM-306's 15 default-mesh cases (one thread each; s = run time):

| Case | default, max_iter 1000 | accelerated (m = 5), 1000 | accelerated, 4000 | within 1 %? |
|---|---|---|---|---|
| c = 50, phi = 0 | 255.9 (10.3) | (not the default for phi = 0 in this run) | | |
| c = 50, phi = 0, gamma = 18 | 255.9 (13.1) | | | |
| c = 50, phi = 0, q = 36 | 295.8 (13.3) | | | |
| c = 20, phi = 25, gamma = 18 | 455.6 (19.7) | 520.7 (13.5) | 520.7 (26.4) | yes |
| same, q = 0.01 | 455.7 (20.5) | 520.8 (13.2) | 520.8 (25.2) | yes |
| c = 0, phi = 30, gamma = 18 | 43.0 (5.9) | 71.7 (6.1) | 86.0 (27.0) | no (GM-304: set by the mesh) |
| same, psi = 30 | 43.0 (6.6) | 86.0 (8.1) | 86.0 (19.7) | yes |
| c = 0, phi = 30, gamma = 18, q = 10 | 216.6 (15.1) | 278.6 (13.4) | 299.2 (40.0) | no (+7.4 %) |
| same, psi = 30 | 257.9 (17.2) | 340.5 (7.8) | 361.2 (28.6) | no (+6.1 %) |
| c = 0, phi = psi = 20, q = 20 | 130.9 (26.3) | 130.9 (8.9) | 130.9 (22.4) | yes |
| c = 5, phi = 35, gamma = 19, q = 19 | 897.0 (22.1) | 897.0 (20.2) | 897.0 (60.6) | yes (2 settled steps) |
| B = 1, c = 10, phi = 30, gamma = 18, E = 2e4 | 321.9 (22.5) | 321.9 (12.5) | 375.6 (44.1) | no (+17 %) |
| B = 3, c = 0, phi = 32, gamma = 19, q = 38 | 954.3 (28.3) | 1015.4 (24.8) | 465.6 (25.9) | no (lower at 4000) |
| c = 10, phi = psi = 20, gamma = 18, q = 18 | 316.0 (20.5) | 340.9 (8.9) | 353.3 (40.0) | no (+3.6 %) |
| rigid rough, c = 50, phi = 0 | 306.0 (32.8) | | | |

The phi = 0 rows were run with an interim default (accelerated only for
phi > 0) that kept the plain iteration on clay: 255.9, 255.9, 295.8 and
306.0 at 1000 and 4000, identical to the default. As shipped, every row's
default is the plain iteration, identical to master by construction (the
`method='elastic'` path of `run_nl` is untouched; pinned, section 4).

**Why the failing rows fail.** On each, a load step below collapse sits on
a residual plateau near 1e-3 (tol 1e-4) for thousands of iterations
(B = 3, step 8: 1.0e-3 to 1.8e-3 over 4000 iterations). GM-306's settled
test judges such a step on its displacements over the two halves of
max_iter, so whether it is accepted depends on max_iter: at 1000 the B = 3
step 8 was accepted (growth 0.017), at 4000 it was not (growth 0.16) and
the run stopped there (465.6). The plain iteration converged that same
step in 601 iterations.

On 20x10 the c-phi case read 542.4, 607.5 and 564.1 kPa at max_iter 400,
1600 and 6400 (accelerated; the plain iteration: 455.6 at 400, 520.7 at
1600).

### 2.3 Variants of Anderson (1000 / 4000)

| Case | m = 3 | stall fall-back, 100 its, gain 0.5 | fall-back 300, 0.5 | fall-back 200, 0.9 | settled test off (m = 5) |
|---|---|---|---|---|---|
| c-phi | 499.0 / 520.7 | 499.0 / 499.0 | 499.0 / 499.0 | 499.0 / 499.0 | 520.7 / 520.7 |
| c = 0, phi = 30 | 86.0 / 86.0 | 71.7 / 71.7 | 100.4 / 71.7 | 86.0 / 71.7 | – |
| same, psi = 30 | 86.0 / 86.0 | 71.7 / 71.7 | – | – | – |
| c = 0, phi = 30, q = 10 | 278.6 / 299.2 | 257.9 / 257.9 | 299.2 / 299.2 | 299.2 / 299.2 | 278.6 / 278.6 |
| same, psi = 30 | 340.5 / 381.8 | 319.9 / 319.9 | 361.2 / 340.5 | 340.5 / 361.2 | 340.5 / 361.2 |
| c = 0, phi = psi = 20, q = 20 | 130.9 / 130.9 | 130.9 / 130.9 | – | – | – |
| c = 5, phi = 35, q = 19 | 959.8 / 1085.2 | 897.0 / 897.0 | 897.0 / 897.0 | 897.0 / 897.0 | 395.3 / 395.3 |
| B = 1 | 339.8 / 375.6 | 321.9 / 321.9 | 321.9 / 321.9 | 321.9 / 321.9 | 196.7 / 196.7 |
| B = 3 | 1015.4 / 1015.4 | 954.3 / 954.3 | 1015.4 / 954.3 | 1015.4 / 1015.4 | 465.6 / 465.6 |
| GM-11 row | 353.3 / 340.9 | 340.9 / 340.9 | 365.7 / 353.3 | 353.3 / 353.3 | 340.9 / 340.9 |

"Stall fall-back": once the step's best residual has not fallen below
`gain` times its value `window` iterations earlier, the rest of the step
uses the plain iteration (this session's heuristic; no printed source).
The 100-iteration fall-back agrees at 1000 and 4000 on every row, but it
reads the c-phi case at 499.0, below the 520.7 that pure Anderson
converges to tol (so its agreement is not convergence), and on four rows
it reads exactly the default's 1000-iteration answer. With the settled
test off, q_ult cannot fall with max_iter (a step converges to tol or the
run stops), but three rows read far below the default (the plateau steps
never reach tol). No variant met the goal on every case.

### 2.4 Prandtl (phi = 0)

The plain iteration (the default, unchanged): 255.9 kPa on 40x20, 60x30,
80x40 (-0.5 %). The accelerated iteration (and BFGS) converge the next load
step on 40x20 to tol: 265.1 (+3.1 %), and 265.1 also with 180 load steps
(the 40x20 mesh's own limit lies between 265.1 and 267.4); 60x30 and 80x40
read 255.9, 20x10 283.4 (all re-run with m = 5: 283.4, 265.1, 255.9 on
20x10, 40x20, 60x30; 80x40 only with m = 10). So on the default
mesh the validated 255.9 is a stop of the plain iteration below the mesh's
own limit; the mesh converges to Prandtl from above. Not changed (the
default is unchanged); noted for the lead.

## 3. What changed

- `fem2d/solver.py`: `_anderson_step`, `run_nl(method='anderson',
  anderson_memory=None)`, `_ANDERSON_MEMORY = 5`. Under 'anderson' a step
  that reaches max_iter is accepted as settled only with a residual below
  `settled_residual_cap` (BFGS showed a "still falling" residual of 0.66
  being accepted). The `method='elastic'` and `'tangent'` paths are
  untouched except that two conditions gained a `method != 'anderson'`
  term that is True for them.
- `fem2d/analysis.py::analyze_footing_capacity`: new library input
  `iteration` ('constant_stiffness', the default, None meaning it; or
  'accelerated'), validated; passed to the footing stage of both footings
  (stage 1 always plain). A call that gives `iteration` gets a "nonlinear
  iteration" entry in `choices` (`how` 'given', basis, and the other scheme
  as the alternative with its numbers); left out, no entry is added (the
  tool cannot set it, and listing an input the tool refuses would mislead
  the agent), so tool output is unchanged: the three adapter tests that pin
  the `choices` list pass as before. The max_iter warning gains one
  sentence when the accelerated iteration was used.
- Not changed: the adapter (`funhouse_agent/adapters/fem2d_adapter.py`),
  the tool description and parameters, `defaults_used`, the page
  fingerprints, the user manual.
- `fem2d/DESIGN.md`, `fem2d/VALIDATION.md` §3: the scheme, its sources, why
  it is not the default, the table.
- `fem2d/tests/test_gm308_iteration.py` (new, 9 tests): a strength
  reduction and a staged excavation pinned to master's values (rel 1e-12;
  checked identical on master and the branch); the Anderson step on a
  linear system and its first step; `iteration` refused when unknown; the
  option and the default on 20x10 c-phi with their `choices` (the entry
  only when `iteration` is given); clay
  unchanged; the default-mesh c-phi 520.7 with the option (slow); the rigid
  footing takes the option.
- `module_work/gm308/run_width.py`: an optional 7th argument passes
  `iteration`.

## 4. Before / after

Default answers: unchanged (the default is the plain iteration; its code
path is untouched and the SRM / staged excavation pins match master to the
last digit printed by `repr`). Worked example with the option, c = 20 kPa,
phi = 25, gamma = 18, B = 2 m, 40x20: default 455.6 kPa (19.7 s);
`iteration='accelerated'` 520.7 kPa (13.5 s), the same at max_iter 4000.
(Review: on another machine the accelerated 1000 read 499.0; 4000 and
16000 read 520.7 on every machine checked; see "Review fixes".)

## 5. Part (b) re-run: the domain width at phi 40, q = 18

`run_width.py 40 <hw> <max_iter> 18 150 0.7 accelerated` (c = 5,
gamma = 18, element size 0.5 m at every width, q_max 2451.0, 150 steps):

| max_iter | 5 B | 10 B | 15 B |
|---|---|---|---|
| 2000 | 1137.2 (2 settled; 60 s) | 245.1 (stopped at step 15; 18 s) | 1526.5 (9 settled; 544 s) |
| 8000 | 1640.0 (6 settled; 395 s) | 1234.5 (11 settled; 894 s) | 1558.9 (15 settled; 1713 s) |

(Lines appended to `module_work/gm308/results.jsonl` with an
`"iteration": "accelerated"` key.)

Against part (b)'s plain-iteration rows (8000: 1850.9, 1915.8, 1948.2,
1526.5 at 5, 7.5, 10, 15 B; 16000: 1883.3 at 5 B, 1964.4 at 15 B).
**There is no converged iteration to re-run part (b) on, and the
accelerated one does not answer the width question.** On this case it
reads below the plain iteration at 5 B and 10 B at 8000 (5 B 1640.0
against 1850.9; 10 B 1234.5 against 1948.2; at 15 B 1558.9 against 1526.5,
a plain run that stopped well before collapse), it is not ordered by width
(10 B lowest), and it moves with max_iter by up to five times (10 B: 245.1
and 1234.5), with many steps accepted as settled. So part (b)'s finding stands
as it was (5 B 3.5-5 % below the wider domains, the safe side, on an
iteration-limited answer), and the re-run waits for a footing answer that
does not depend on max_iter (section 7).

## 6. Tests (venv /tmp/v, Python 3.12; each judged on pytest's own exit code)

| Command (`python -m pytest -q -p no:cacheprovider ...`) | Result | Exit |
|---|---|---|
| `-n 2 fem2d/` (whole module, incl. the 9 new tests in `test_gm308_iteration.py`) | 653 passed | 0 |
| `-n 3` the funhouse_agent tests that name fem2d (`test_adapter_ergonomics`, `test_calc_package_figure_layout`, `test_catalog_routing_hints`, `test_eval_suite_v54`, `test_fem2d_adapter_validation`, `test_fem2d_mesh_study_adapter`, `test_fem2d_stage_movements_adapter`, `test_footing_capacity_adapter`, `test_gm113_refusal_hints`, `test_local_fos_adapter`, `test_method_name_aliases`, `test_module_followup_adapters`, `test_nested_keys`, `test_phase34_adapters`, `test_reviewers`, `test_tool_usability_foundry_eval`), `validation_examples/test_published_v023_v024.py`, `calc_package/tests/test_interactive.py`, `calc_package/tests/test_calc_viz_integration.py` | 625 passed | 0 |
| `funhouse_agent/tests/test_defaults_used.py -k fem2d` | 46 passed | 0 |
| `webapp/tests/test_page_fingerprints.py webapp/tests/test_public_text.py` | 20 passed (no re-record) | 0 |

The first run of the funhouse batch had 3 failures in
`test_footing_capacity_adapter.py` (they pin the `choices` list, and the
first build listed the iteration in every result); the entry is now added
only when `iteration` is given, and the batch above is the re-run. fem2d
has no calc-package template of its own (`calc_package` names it only in
`interactive.py`); `method_probes.py` needed no entry (no tool input
changed).

## 7. What is left

- **A converged frictional answer still needs work on the plateau steps.**
  The obstacle is no longer the iteration's speed alone: steps below
  collapse stall at a residual near 1e-3 against tol 1e-4 (most likely
  Gauss points switching between Mohr-Coulomb's faces, edges and apex),
  and GM-306's settled test turns that into an answer that depends on
  max_iter. Candidates for the lead: a convergence test for the footing
  that does not depend on max_iter (for example a fixed iteration window
  for the displacement test), or a residual tolerance matched to that
  noise; each would move saved frictional answers and needs its own
  measurement. Would change saved values only if adopted.
- Prandtl on 40x20: the default's 255.9 is below the mesh's own limit
  (265.1 to 267.4 with a converged iteration); the agreement with
  (2 + pi) c on the default mesh is partly a stop of the plain iteration.
  Noted, not fixed (no value changes).
- Part (b)'s width re-run at phi 40, q = 18 (GM-308 CHECK section 7)
  still waits: the accelerated iteration does not give a converged answer
  there (section 5).
- The tool description's "On a frictional soil the flexible q_ult is a
  lower bound that a larger max_iter can raise" stays true of the default.

## 8. Release note

No calculated value changes. **fem2d footing capacity: an accelerated
iteration as an option (GM-308 (a)).** `analyze_footing_capacity` takes
`iteration='accelerated'` (Python only; the tool is unchanged): Anderson
acceleration of the constant-stiffness iteration (Anderson 1965; Walker &
Ni 2011). It reads the c-phi check (c = 20 kPa, phi = 25, B = 2 m, 40x20)
at 520.7 kPa at max_iter 4000 and 16000, where the default reads 477.3 and
499.0, but on cohesionless and high-phi checks its answer still moved with
max_iter, either way, and near collapse its answers vary by machine, so
the default stays the constant-stiffness iteration. A call that gives
`iteration` gets a "nonlinear iteration" entry in `choices`; tool output
is unchanged.

## 9. Review fixes (independent review, 2026-11-06)

The review found the default untouched: the master and branch default
outputs were identical on one machine (six footing cases and a strength
reduction: q_ult, warnings, choices, sum |u|, the rigid curve), before and
after these fixes. It also found that near collapse the accelerated answers
vary by machine. Three machines: A, the one this branch was built on
(Linux, Python 3.12; sections 2-6); B, Windows, Python 3.14.5, numpy
2.4.6, scipy 1.17.1; C, Linux (WSL), Python 3.12.15, numpy 2.5.4, scipy
1.18.1. The default reads the same on all three, on every row of section
2.2's table. The accelerated iteration, on the same rows:

| Case | accelerated 1000, A / B / C | accelerated 4000, A / B / C |
|---|---|---|
| c = 20, phi = 25, gamma = 18 | 520.7 / 520.7 / 499.0 | 520.7 / 520.7 / 520.7 |
| same, q = 0.01 | 520.8 / 520.8 / 542.5 | 520.8 / 520.8 / 520.8 |
| c = 0, phi = 30, gamma = 18 | 71.7 / 71.7 / 71.7 | 86.0 / 86.0 / 71.7 |
| same, psi = 30 | 86.0 / 86.0 / 86.0 | 86.0 / 86.0 / 86.0 |
| c = 0, phi = 30, q = 10 | 278.6 / 299.2 / 154.6 | 299.2 / 278.6 / 299.2 |
| same, psi = 30 | 340.5 / 340.5 / 340.5 | 361.2 / 361.2 / 361.2 |
| c = 0, phi = psi = 20, q = 20 | 130.9 / 130.9 / 130.9 | 130.9 / 130.9 / 130.9 |
| c = 5, phi = 35, q = 19 | 897.0 / 897.0 / 959.8 | 897.0 / 1085.2 / 395.3 |
| B = 1 | 321.9 / 321.9 / 196.7 | 375.6 / 375.6 / 375.6 |
| B = 3 | 1015.4 / 1137.6 / 465.6 | 465.6 / 1137.6 / 1015.4 |
| GM-11 row | 340.9 / 340.9 / 340.9 | 353.3 / 340.9 / 340.9 |

c-phi at 16000: 520.7 on A, B and C. 20x10 c-phi at 400 / 1600 / 6400:
A 542.4 / 607.5 / 564.1, B 564.1 / 564.1 / 564.1, C 564.1 / 564.1 /
585.8. Clay 40x20: 265.1 (the step converged to tol) on A, B and C; 60x30
255.9 on A and C. The least-squares matrix is well conditioned (cond at
most 8.5e3 on two runs on B), so the cause is rounding growing over the
hundreds of non-smooth iterations near collapse, not a singular solve. No
machine meets the goal on every row, so section 2's verdict stands, more
strongly. On C the option read below the default on four rows.

Fixes:

1. `test_accelerated_option_on_a_frictional_soil`: the 542.4 pin (A only)
   is dropped. It asserts the option reads above the default (455.6), and
   that the max_iter warning says "lower bound" by default and "not a
   bound" with the option.
2. `test_default_mesh_frictional_answer_accelerated` (slow): 520.7 is now
   pinned at max_iter 4000 (A, B, C) instead of 1000 (C read 499.0).
3. `test_staged_excavation_unchanged`: rel 1e-12 became 1e-8. Master itself
   reads sum |u| 5.419050308100497 on B and C, 2.5e-11 from the pin (the
   moments about 5e-10). The strength-reduction pin holds to 3e-16 on B
   and C and stays at 1e-12.
4. `fem2d/analysis.py` (docstring, the choices entry), `DESIGN.md`,
   `VALIDATION.md` §3 and section 8's release note state only numbers that
   hold on every machine checked ("520.7 at 4000 and 16000"). They add the
   sentence that the option's answers near collapse vary by machine.
   VALIDATION's table now gives A / B / C where they differ, and drops the
   run times.
5. The max_iter warning with the option no longer says "q_ult is a lower
   bound ... a larger max_iter can raise it". It says q_ult is not a bound
   and can move either way with max_iter. The default's warning is
   unchanged, byte for byte.
6. Two docstrings (`_side_boundary_warnings`, `analyze_slope_srm`) are back
   to "VALIDATION.md section 11", where the Duncan Ex 1 numbers are.
7. Section 5: "below the plain iteration at every width" became "at 5 B and
   10 B" (15 B: 1558.9 against a plain 1526.5 that stopped early). Part
   (b)'s conclusion is unchanged; the accelerated width rows are from A
   only and would vary by machine as well.

Sections 2-6 above are left as measured on A.
