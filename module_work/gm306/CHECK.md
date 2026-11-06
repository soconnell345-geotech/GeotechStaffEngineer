# GM-306 (fem2d): footing capacity on fine meshes

Branch `claude/gm-306-footing-fine-mesh`, based on GM-305's last commit
5da1a3c; GM-305 merged to master as 85ea79a. Master was merged in (merge
d5c64ab on `merge/gm306`) before the review fixes of section 7. All test
values are made up. B = 2 m, T6, domain 10B x 5B, E = 1e5 kPa, nu = 0.3,
psi = 0 throughout. Sections 1 to 5 are the branch as first built (every
run at `max_iter = 1000`); section 7 is what the review changed.

## 1. The cause of the 80x40 drop

The flexible footing (the default) took the first load step that did not
meet `tol = 1e-4` within `max_iter = 1000` as collapse. On c = 20 kPa,
phi = 25, gamma = 18, 80x40, the run stopped at step 9 (q_ult 173.6 kPa).
That step was not a mechanism:

- its residual was still falling: 2.2e-2 after the first iterations,
  1.8e-4 at the 1000th (tol 1e-4);
- its displacements stood still: the second 500 iterations moved the
  nodes by 0.095 of what the first 500 did. A mechanism keeps moving
  (ratio about 1).

The constant-stiffness iteration converges more slowly on finer meshes,
so the 1000-iteration ceiling arrives earlier in the load ramp. **It is a
numerical artefact, not mesh dependence.** With `max_iter = 4000` the
80x40 run reaches 455.6 kPa.

The same ceiling reaches the default mesh on frictional soils. Each
`max_iter = 1000` stop below had its residual still falling at the end:

| Case, flexible | Mesh | max_iter 1000 | max_iter 4000 | max_iter 16000 |
|---|---|---|---|---|
| c = 50, phi = 0, gamma = 0 | uniform 40x20 | 255.9 | 255.9 | – |
| c = 20, phi = 25, gamma = 18 | uniform 40x20 | 455.6 | 477.3 | 499.0 (review) |
| c = 0, phi = 30, gamma = 18 (q_max 200, 80 steps) | uniform 40x20 | 52.5 | 67.5 | – |
| c = 10, phi = psi = 20, gamma = 18, q = 18 | uniform 40x20 | 316.0 | 328.4 (review) | – |
| c = 20, phi = 25, gamma = 18 | uniform 80x40 (after the fix) | 217.0 | 455.6 | – |
| c = 20, phi = 25, gamma = 18 | graded 40x20 | 151.9 | 238.6 | – |
| same, rigid rough | graded 40x20 | >= 151.2 lb | >= 441.2 lb | – |

The validated Prandtl case (phi = 0) does not move: there the stop is a
real collapse. For phi > 0 the stopping step's displacement ratio (0.19 to
0.46) cannot be told from a real collapse's.

## 2. The fix, and the warning

- **Fix (`fem2d/solver.py`, `fem2d/analysis.py`).** A flexible step that
  reaches `max_iter` with its displacements standing still is accepted,
  the strength reduction's test (Griffiths & Lane 1999: failure is
  non-convergence with a dramatic increase in displacements;
  `settled_ratio = 0.1`, `_FLEX_SETTLED_RATIO`). The test runs only at the
  ceiling: new `run_nl(settled_early=False)`. So a step that meets tol on
  its own ends exactly as it did. `n_steps_settled` counts the accepted
  steps.
- **Default answers unchanged, to the last digit.** Nine default-mesh
  (40x20) cases were run before and after: q_ult and a hash of the whole
  displacement field are identical, and no step was accepted under the new
  test (the stopping steps' ratios were 0.19 to 0.90). Cases: c = 50;
  c = 50, gamma = 18; c = 50, q = 36; c = 20, phi = 25, gamma = 18, with
  q = 0 and 0.01; c = 0, phi = 30, gamma = 18 (psi 0 and 30; q 0 and 10);
  c = 0, phi = psi = 20, q = 20.
- **Partial cure.** It lifts 80x40 c-phi from 173.6 to 217.0. The rest of
  the gap is the same ceiling: the later steps' ratios sit just above 0.1
  (0.106 at the stop). Raising the threshold or `max_iter` would cure more,
  but would also move the default frictional answers (455.6 -> 477.3), which
  this brief forbids. That is the lead's and the owner's call. (The review
  fix O1, section 7, scales the default `max_iter` with the element count
  instead: 40x20 keeps 1000, 80x40 gets 4000 and reads 455.6.)
- **Warning.** A flexible result on a frictional soil (phi > 0) whose last
  step stopped at `max_iter` now says so, with the numbers above. It fires
  on default-mesh frictional runs too (no value changes). phi = 0 runs get
  no such warning. (Reworded by the review fix F2, section 7: a lower
  bound, not "rerun and compare".)

## 3. The graded mesh

`mesh='graded'` (option; default `'uniform'`). It keeps the nx x ny
elements, grades them towards x = +-B/2 and the surface, and keeps grid
lines at x = 0 and +-B/2. The elements at the edges next to the surface
are 1/4 the uniform size (`_GRADED_EDGE_FRACTION`, this module's choice; no
reference prints a grading); sizes grow by one ratio away from them
(`mesh.geometric_spacing`). nx must be even. Listed in `choices` ("footing
mesh") with its basis.

Study (`/tmp` scripts through the shipped function; flexible c = 0 at
q_max 200 and 80 steps, as GM-304; rigid rough at the default
settlement_max and 45 steps; uniform rigid rows from GM-305, unchanged by
this branch; rigid smooth on graded meshes not run). kPa; "lb" = the run
stopped before q levelled off.

| Case (closed form, GEC-6) | Footing, mesh | 20x10 | 40x20 | 60x30 | 80x40 |
|---|---|---|---|---|---|
| c = 50, phi = 0, gamma = 0 (Eq. 5-4, p. 52: 257.1) | flexible uniform | 274.2 | 255.9 | 255.9 | 255.9 |
| | flexible graded | 255.9 | 255.9 | 255.9 | 255.9 |
| | rigid rough uniform | 347.9 | 306.0 | 291.0 | 283.1 |
| | rigid rough graded | 287.8 | 273.1 | 267.9 | 265.3 |
| c = 20, phi = 25, gamma = 18 (Eqs. 5-7, 5-8, p. 53: 610.2; 512.3 to 806.0) | flexible uniform | 499.0 | 455.6 | 433.9 | 217.0 (was 173.6) |
| | flexible graded | 433.9 | 151.9 | 130.2 | 130.2 |
| | rigid rough uniform | 817.7 | >= 674.2 lb | >= 607.7 lb | >= 253.7 lb |
| | rigid rough graded | >= 380.6 lb | >= 151.2 lb | >= 150.8 lb | >= 149.7 lb |
| c = 0, phi = 30, gamma = 18 (p. 53: 403.2; 201.6 to 806.4) | flexible uniform | 102.5 | 52.5 | 35.0 | 27.5 |
| | flexible graded | 32.5 | 17.5 | 12.5 | 12.5 |
| | rigid rough uniform | >= 286.8 lb | >= 97.6 lb | >= 43.2 lb | >= 42.5 lb |
| | rigid rough graded | >= 42.5 lb | >= 41.3 lb | >= 40.2 lb | >= 39.8 lb |

- **Prandtl:** graded helps. The flexible footing is at 255.9 from 20x10
  (uniform needs 40x20). The rigid footing converges from above at +12,
  +6.2, +4.2, +3.2 % (uniform +35, +19, +13, +10 %). 80x40 rigid graded
  took about 58 minutes.
- **c-phi and c = 0:** graded does not converge where uniform does not.
  The flexible footing reads lower: its edge elements are smaller, and on
  c = 0 that is GM-304's edge failure. The iteration also hits its ceiling
  sooner. The rigid footing stops after one or two steps, again at the
  ceiling (40x20 c-phi: 151.2 at 1000 iterations, 441.2 at 4000, still
  rising).

**Should graded become the default? No.** It is better only on the case
the uniform flexible footing already gets right (Prandtl), and worse with
the flexible footing on frictional soils. I have not changed the default.
For the rigid footing on clay (phi = 0), graded is the better mesh; making
it rigid's own default would change no saved answer (rigid is new in
GM-305), but I have left that to the lead and owner too. (Review fix F3,
section 7: `mesh='graded'` is now refused when phi > 0.)

What a mesh-converged frictional answer needs is an iteration that
converges faster than the constant-stiffness one, or a far larger
iteration budget. Either would change saved frictional answers (default
c-phi 455.6 -> >= 477.3 kPa), so it is a candidate item for the lead.
GM-305 found a tangent iteration too slow for the rigid footing.

## 4. Whether any saved answer changes (as first built; after the review, section 7)

- **Default mesh (uniform 40x20, default max_iter): no value changes.**
  Nine cases compared to the last digit (section 2).
- **Finer meshes, or a lower max_iter:** q_ult rises where a step reached
  `max_iter` with its displacements standing still (80x40 c-phi 173.6 ->
  217.0; 20x10 c-phi at max_iter 100: 282.0 -> 368.8). 20x10 and 60x30 in
  the study are unchanged.
- **New in every footing result:** `choices` gains a "footing mesh" entry;
  results carry `mesh` (and `n_steps_settled`, in the tool output only
  when non-zero).
- **New warning:** flexible runs on a frictional soil whose last step
  stopped at `max_iter`, the default-mesh c-phi run included.
- **Rigid warning:** its text now quotes the graded Prandtl values beside
  the uniform ones.
- **Page fingerprints:** did not move.

## 5. Tests (venv Python 3.11; each judged on pytest's own exit code)

- `fem2d/tests/test_footing_capacity.py`, whole file (`-n 4`): 42 passed.
  New tests:
  - `TestGradedMeshInputs`: 9 fast tests (inputs, the grid lines,
    `geometric_spacing`, `generate_rect_mesh` coordinates, choices);
  - `TestFlexibleCollapseCriterion`: 2 fast tests (20x10 at max_iter 100,
    before and after; a step that converges is unchanged to the last
    digit; the max_iter warning on phi > 0 and not on phi = 0);
  - `TestRigidFooting::test_prandtl_rough_graded`: slow, 287.8.
- `funhouse_agent/tests/test_footing_capacity_adapter.py`: passed (3 new:
  graded through the adapter, bad mesh, METHOD_INFO).
- `funhouse_agent/tests/test_defaults_used.py -k fem2d`: 46 passed.
- `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  passed.
- `fem2d` whole module: measured after the review fixes, 644 passed
  (section 7).

## 6. Release note

**fem2d footing capacity on fine meshes (GM-306).** On meshes finer than the default the flexible footing read a slow load step as collapse. A step that runs out of iterations with its displacements standing still is now accepted, and the default `max_iter` scales with the element count, 1000 x max(1, nx ny / 800) (c = 20 kPa, phi = 25, B = 2 m: 80x40 173.6 -> 455.6 kPa, 60x30 433.9 -> 455.6, the 40x20 value).
Default-mesh (40x20) answers are unchanged to the last digit on every case checked (15).
On a frictional soil q_ult is a lower bound limited by the iteration count (40x20: 455.6, 477.3 and 499.0 at max_iter 1000, 4000 and 16000), and a flexible result whose run stopped at `max_iter` now says so.
New `mesh='graded'` (default stays uniform; undrained clay, phi = 0, only, refused when phi > 0) brings the rigid footing nearer Prandtl (c = 50: 265.3 kPa on 80x40 against 257.1); every result lists the mesh in `choices` and carries `mesh`.

## 7. Review fixes (REVIEW_GM-306, 2026-11-05)

Made on `merge/gm306` after merging master (d5c64ab), in 7f756cc (code,
tests, DESIGN.md, VALIDATION.md) and 1dd3c43 (the manual).

| Finding | Change |
|---|---|
| F1, conflicts with master in 4 files | Merged master; the rigid warning keeps master's sense (no mesh-converged value for c-phi and cohesionless soils, "on either mesh") with the graded Prandtl numbers; `test_prandtl_rough` keeps the branch's choices and master's phrase; the adapter keeps master's brief, `base`, `settlement_max`, `settlement_max_m` and `warnings` with the branch's additions; `manual.html` taken from master and regenerated. |
| O1, a max_iter that scales with the mesh | The footing's default `max_iter` is 1000 x max(1, nx ny / 800), rounded up (`_footing_max_iter`; both footings, stage 1 included). 40x20 and coarser keep 1000; 60x30 gets 2250, 80x40 4000. A given `max_iter` is used as given. The tool no longer declares a fixed default of 1000: `max_iter` left out is a case that states the rule (`defaults_used` lists it). |
| F2, the max_iter warning | "not at a mechanism" and "rerun and compare" are gone. It now says the step stopped at `max_iter` without converging, that such a stop cannot be told from collapse, and that q_ult is a lower bound of this mesh's answer limited by the iteration count (40x20 c-phi 455.6, 477.3, 499.0 at 1000, 4000, 16000, still rising; 80x40 217.0 and 455.6 at 1000 and 4000); a larger max_iter can raise it. The brief and `returns.warnings` follow. |
| F3, graded on frictional soils | `mesh='graded'` with phi > 0 is refused with a ValueError that says why (151.9 against 455.6 kPa uniform, flexible; at least 151.2 against at least 674.2, rigid; 40x20) and points to `mesh='uniform'`. The tool's `mesh` text and the choices basis say graded is for undrained clay (phi = 0) only. Tests: the refusal on both footings and through the tool. |
| F4, placeholders | The fem2d count is filled (section 5, measured below); the header names GM-305's merge. |
| F5, VALIDATION.md | The GM-305 table's 80x40 c-phi cell now says 217.0 after the settled-step test and 455.6 at the new default (60x30: 455.6 at 2250); the GM-11 row's gap is in part the iteration ceiling (316.0 at 1000, 328.4 at 4000); the max_iter table has a 16000 column and the GM-11 row; a paragraph on the scaled default. |
| F6, "unchanged to the last digit" | The release note says "on every case checked (15)". |
| O3, choices nit | The uniform mesh's basis says "over the domain", not "over 10B x 5B". |

**Default-mesh answers, before (d5c64ab) and after, 40x20, max_iter left
out.** 15 cases, all identical to the last digit in q_ult, u_max, a hash
of the whole displacement field and `n_steps_settled` (no step accepted as
settled): c = 50 (255.9); c = 50, gamma = 18 (255.9); c = 50, q = 36
(295.8); c = 20, phi = 25, gamma = 18, q = 0 and 0.01 (455.6, 455.7);
c = 0, phi = 30, gamma = 18, psi 0 and 30, q 0 and 10 (43.0, 43.0, 216.6,
257.9); c = 0, phi = psi = 20, q = 20 (130.9); c = 5, phi = 35,
gamma = 19, q = 19 (897.0); B = 1, c = 10, phi = 30, gamma = 18, E = 2e4
(321.9); B = 3, c = 0, phi = 32, gamma = 19, q = 38 (954.3); c = 10,
phi = psi = 20, gamma = 18, q = 18 (316.0); rigid rough c = 50 (306.0).
All 11 frictional ones carry the reworded max_iter warning.

**Finer meshes at the new default** (c = 20, phi = 25, gamma = 18): 60x30
455.6 kPa at 2250 (113 s; 433.9 at 1000) and 80x40 455.6 at 4000 (438 s;
217.0 at 1000 on the branch, 173.6 on master). The three meshes agree
because each one's budget grows with its elements, not because the answer
has converged (40x20 rises to 499.0 at 16000).

### Saved answers that move (against master)

- **Fine-mesh q_ult rises** wherever the mesh has more than 800 elements
  and a step reached 1000 iterations: c-phi flexible 60x30 433.9 -> 455.6,
  80x40 173.6 -> 455.6 kPa. Frictional rigid results on such meshes can
  rise too (not run). Such runs take longer (80x40 c-phi: 438 s).
- **Frictional flexible results carry the reworded max_iter warning**
  (every one of the 11 default-mesh frictional cases above).
- **Every footing result** gains the "footing mesh" entry in `choices` and
  a `mesh` output key (and `n_steps_settled` when non-zero); the rigid
  warning quotes the graded Prandtl numbers.
- **`defaults_used`** lists `max_iter` as its rule ("1000 x max(1, nx ny /
  800) iterations a load step: ...") instead of 1000.
- **No default-mesh q_ult or displacement moves** (15 cases above). fem2d
  SRM, excavation, staged and local FOS: unchanged (no change outside the
  footing; their tests pass); no slope_stability file is touched.

### Tests after the review fixes (venv Python, worktree root, one run at a time, judged by pytest's exit code)

`fem2d.__file__` and `funhouse_agent.__file__` point into the worktree.
pytest-xdist is not in the venv, so `fem2d/tests/` ran serially in chunks.

| Command (`python -m pytest -q -p no:cacheprovider ...`) | Result | Exit |
|---|---|---|
| `fem2d/tests/test_footing_capacity.py funhouse_agent/tests/test_footing_capacity_adapter.py` | 70 passed (56 + 14) | 0 |
| `fem2d/tests/` calc_*, consolidation_*, cross_validation, excavation_* | 171 passed | 0 |
| `fem2d/tests/` test_fem2d, groundwater, hs_*, layer_polylines, local_fos, mc3d, mesh_study, nr_hardening | 246 passed | 0 |
| `fem2d/tests/` seepage_flow, srm_defaults, srm_duncan, srm_robustness | 57 passed | 0 |
| `fem2d/tests/` stage_movements, staged, staged_thickness, t6, validation, wall_line | 114 passed | 0 |
| **fem2d/tests total** | **644 passed** | all 0 |
| `funhouse_agent/tests/test_top_level_params.py test_gm179_unused_inputs.py test_calc_package_tool_parity.py` | 256 passed | 0 |
| `funhouse_agent/tests/test_defaults_used.py` (in full) | 831 passed, 14 skipped | 0 |
| `webapp/tests/test_page_fingerprints.py webapp/tests/test_public_text.py` | 19 passed (no re-capture) | 0 |
| `funhouse_agent/tests/test_worked_examples.py` | 38 passed, 2 skipped (the refs' source-PDF folder is absent in a worktree; not footing tests) | 0 |

No probe was added: no input became conditional (graded's phi = 0 limit is
a refused value, and `max_iter`'s case follows nx and ny, not another
input's choice).

## 8. What is left (for the lead to file)

- **O2, frictional answers are iteration-limited even on the default
  mesh.** 40x20 c-phi reads 455.6, 477.3 and 499.0 kPa at max_iter 1000,
  4000 and 16000, one load step per four times the iterations, still
  rising; the GM-11 row reads 316.0 and 328.4 at 1000 and 4000. A converged
  frictional answer needs a faster nonlinear iteration than constant
  stiffness (GM-305 found a tangent iteration too slow for the rigid
  footing). It would move saved frictional answers, upwards (today's are
  lower bounds).
- **O4 (the review's flag, not verified by runs):** the default domain
  (half-width 5B) is narrower than the classical Prandtl-Reissner passive
  zone for phi of about 31 degrees or more (4.79B at phi = 30, 5.04B at
  phi = 31 from the centre); a psi = 0 FE mechanism is narrower than that.
- The branch's own open call: graded as the rigid footing's default on
  clay (phi = 0) would change no saved answer; left to the lead and owner.
