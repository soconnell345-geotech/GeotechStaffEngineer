# GM-306 (fem2d): footing capacity on fine meshes

Branch `claude/gm-306-footing-fine-mesh`, based on
`claude/gm-305-fem2d-rigid-footing` (GM-305 was not on master when this
started; merge it first). All test values are made up. B = 2 m, T6,
domain 10B x 5B, E = 1e5 kPa, nu = 0.3, psi = 0 throughout.

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

| Case, flexible | Mesh | max_iter 1000 | max_iter 4000 |
|---|---|---|---|
| c = 50, phi = 0, gamma = 0 | uniform 40x20 | 255.9 | 255.9 |
| c = 20, phi = 25, gamma = 18 | uniform 40x20 | 455.6 | 477.3 |
| c = 0, phi = 30, gamma = 18 (q_max 200, 80 steps) | uniform 40x20 | 52.5 | 67.5 |
| c = 20, phi = 25, gamma = 18 | uniform 80x40 (after the fix) | 217.0 | 455.6 |
| c = 20, phi = 25, gamma = 18 | graded 40x20 | 151.9 | 238.6 |
| same, rigid rough | graded 40x20 | >= 151.2 lb | >= 441.2 lb |

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
  this brief forbids. That is the lead's and the owner's call.
- **Warning.** A flexible result on a frictional soil (phi > 0) whose last
  step stopped at `max_iter` now says so, with the numbers above, and asks
  for a rerun with a larger `max_iter`. It fires on default-mesh frictional
  runs too (no value changes). phi = 0 runs get no such warning.

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
GM-305), but I have left that to the lead and owner too.

What a mesh-converged frictional answer needs is an iteration that
converges faster than the constant-stiffness one, or a far larger
iteration budget. Either would change saved frictional answers (default
c-phi 455.6 -> >= 477.3 kPa), so it is a candidate item for the lead.
GM-305 found a tangent iteration too slow for the rigid footing.

## 4. Whether any saved answer changes

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
- `fem2d` whole module (`-n 4`): 630 passed, 0 failed.

## 6. Release note

**fem2d footing capacity on fine meshes (GM-306).** A load step that runs out of iterations with its displacements standing still is no longer read as collapse (c = 20 kPa, phi = 25, B = 2 m, 80x40: 173.6 -> 217.0 kPa); default-mesh answers are unchanged to the last digit.
On a frictional soil q_ult can still rise with `max_iter` (40x20: 455.6 at 1000, 477.3 at 4000), and the result now warns when its run stopped at `max_iter`.
New `mesh='graded'` (default stays uniform) brings the rigid footing nearer Prandtl (c = 50: 265.3 kPa on 80x40 against 257.1), but reads lower on frictional soils; every result lists the mesh in `choices`.
