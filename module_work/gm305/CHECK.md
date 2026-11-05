# GM-305 (fem2d) — a rigid footing for footing capacity

Branch `claude/gm-305-fem2d-rigid-footing`. All test values are made up.

## What the print says

From the reference library's text of **GEC-6** (FHWA-SA-02-054, *Shallow
Foundations*, 2002), chapter 5 (`geotech_references/gec_6/text/chapter05.json`):

- p. 55 (Sec. 5.2.3): "Equation 5-7 assumes a rigid strip (continuous)
  footing that is loaded through its centroid bearing on a level surface of
  homogeneous soil."
- pp. 51-53 (Sec. 5.2.2): Eq. 5-1 q_ult = c Nc + q Nq; Eq. 5-2 Nq; Eq. 5-4
  Nc = 2 + pi = 5.14 for phi = 0 (p. 52); Eq. 5-7 q_ult = c Nc + q Nq +
  0.5 gamma B Ngamma; Eq. 5-8 Ngamma = 2 (Nq + 1) tan phi (Caquot and
  Kerisel 1948, used by FHWA and AASHTO). p. 53: published Ngamma values
  "range from about one-half to double those given here, depending on the
  friction angle (Chen & McCarron, 1991)".
- pp. 49-50 (Sec. 5.2.1): in general shear "a triangular wedge beneath the
  footing, Zone I, remains elastic and moves down with the footing".
- p. 99 (Sec. 5.5): "For concrete cast against cohesionless or granular
  material, the coefficient of friction, tan delta, will be equal to the
  tangent of the friction angle (tan phi) for the soil supporting the
  footing"; on clay "c_a is typically taken as equal to the undrained
  shear strength".

**Rough or smooth.** The wedge moving down with the footing (pp. 49-50)
and a concrete base that takes the soil's own friction or adhesion (p. 99)
are a footing whose base does not slip: `base='rough'` (the footing's
nodes held horizontally) is the default, with that basis in `choices`.
`'smooth'` (no base friction) is offered as the alternative; no held
reference prints it as the basis of Eq. 5-7, so it is not the default.

## What changed

- `fem2d/solver.py`: `run_nl(..., prescribed=(dofs, delta))` drives given
  DOFs from their `u_init` values by `lam * delta` (residual
  `penalty * (target - u)`; a step converges only on target; the free
  residual is normalised by the larger of the free load and the reaction,
  so a weightless soil can be driven), returning `reactions` and
  `reaction_lams` per converged step. `add_prescribed_dofs(ctx, dofs)`
  puts the DOFs on the penalty diagonal. Without `prescribed` nothing
  changes.
- `fem2d/analysis.py::analyze_footing_capacity`: new `footing`
  ('flexible' default, 'rigid'), `base` ('rough' default, 'smooth'; rigid
  only) and `settlement_max` (rigid only; default 10 q_est B (1 - nu^2) / E).
  - Rigid: stage 1 as GM-11/GM-304 (weight and surcharge over the whole
    surface, whenever gamma > 0 or a surcharge); stage 2 takes the strip's
    surcharge off the load (the reactions carry it, so q starts at the
    surcharge), keeps the weight and the surcharge beside the footing, and
    pushes every surface node under the footing (T6 mid-side nodes
    included) down in `n_load_steps` equal steps; rough also holds their
    horizontal displacement at the stage-1 value. q = -(sum of their
    vertical reactions) / B.
  - **q_ult is the largest q reached.** `collapse_bracketed` is True when
    q rose by less than 1 % over the last tenth of the settlement run (a
    plateau); otherwise q_ult is a lower bound of that mesh's answer and a
    warning says so. Steps whose displacements stand still are accepted
    (`settled_ratio = 0.1`, as the SRM, Griffiths & Lane 1999).
  - Every rigid result warns that q_ult falls as the mesh is refined.
  - Results now carry `footing`, `base`, `choices` (footing model; rigid:
    base, each with how it was set and its basis), and for rigid
    `pressure_settlement` and `settlement_max_m`; `q_max_kPa` is None for
    rigid and `collapse_load_fraction` is the fraction of settlement_max
    reached.
  - Refused: `q_max` with rigid; `base='smooth'` or `settlement_max` with
    flexible.
- `funhouse_agent/adapters/fem2d_adapter.py`: the new inputs (with
  allowed values), the new outputs, `choices` in the tool output; brief,
  parameter and return text say what the code does. `applies_when`:
  `q_max` for flexible, `base` and `settlement_max` for rigid;
  `settlement_max` has a `when_absent` case, so `defaults_used` stays true.
- `funhouse_agent/tests/method_probes.json` / `.py`: a `#rigid` probe
  (exempt on `max_iter` for the same reason as the flexible one).
- `fem2d/DESIGN.md`, `fem2d/VALIDATION.md` §3 (the table below),
  `docs/user_manual/manual.html` (overview row, heading and parameter table
  of `fem2d_footing_capacity`, re-rendered with `build_manual`'s
  `extract_catalog`, `inline` and `method_param_table`; no full rebuild).
- Page fingerprints did not move (the per-method text is not in them), so
  there is no re-record commit.

## Mesh study (B = 2 m, T6, 10B x 5B, E = 1e5, nu = 0.3, psi = 0)

Defaults otherwise (flexible c = 0: q_max 200, 80 steps, as GM-304). kPa;
"lb" = the run stopped before q levelled off (a lower bound of that
mesh's answer). Run with `/tmp/claude-0/s/{flex,rig}.py` through the
shipped `analyze_footing_capacity` (rigid at default settlement_max and
45 steps). Same table in `fem2d/VALIDATION.md` §3.

| Case (closed form, GEC-6) | Footing | 20x10 | 40x20 | 60x30 | 80x40 |
|---|---|---|---|---|---|
| c = 50, phi = 0, gamma = 0 (Eq. 5-4, p. 52: 257.1) | flexible | 274.2 | 255.9 | 255.9 | 255.9 |
| | rigid rough | 347.9 | 306.0 | 291.0 | 283.1 |
| | rigid smooth | 322.7 | 289.5 | 278.8 | 273.5 |
| c = 20, phi = 25, gamma = 18 (Eqs. 5-7, 5-8, p. 53: 610.2; 512.3 to 806.0 with Ngamma halved or doubled) | flexible | 499.0 | 455.6 | 433.9 | 173.6 |
| | rigid rough | 817.7 | >= 674.2 lb | >= 607.7 lb | >= 253.7 lb |
| | rigid smooth | 627.5 | 584.2 | >= 557.8 lb | >= 543.6 lb |
| c = 0, phi = 30, gamma = 18 (Eqs. 5-7, 5-8, p. 53: 403.2; 201.6 to 806.4) | flexible | 102.5 | 52.5 | 35.0 | 27.5 |
| | rigid rough | >= 286.8 lb | >= 97.6 lb | >= 43.2 lb | >= 42.5 lb |
| | rigid smooth | >= 146.1 lb | >= 44.1 lb | >= 42.8 lb | >= 42.1 lb |

GEC-6 p. 53: published Ngamma values range from about one half to double
Eq. 5-8's (Chen & McCarron 1991), so the closed forms with an Ngamma term
are a band, not a target; nothing here is tuned to them.

Rigid Prandtl against 257.1: rough +35, +19, +13, +10 %; smooth +26, +13,
+8, +6 % (20x10 to 80x40): it converges, slowly and from above; the
flexible footing is within one 9.1 kPa load step from 40x20 on (-0.5 %).
An 80x40 rigid run took 25-30 minutes.

## The default: flexible stays

The brief's rule: make rigid the default if it converges with the mesh and
the flexible footing does not. It does not:

- On the c = 0 case, where the flexible footing fails, the rigid footing's
  iteration stops before q levels off on every mesh, earlier the finer the
  mesh (12, 3, 1, 1 steps on 20x10 to 80x40). A tangent iteration with
  cutbacks was tried outside the module (rough, 20x10: 463.6 kPa at the
  full settlement, still rising, 147 s; 40x20: 304.1 kPa at 26 of 73 mm,
  still rising when it stopped, 939 s); it gives no plateau either and is
  too slow to ship.
- On the validated Prandtl case the rigid footing is worse than the
  flexible one on every mesh and falls slowly from above (rough +35, +19,
  +13, +10 %; smooth +26, +13, +8, +6 % on 20x10 to 80x40), where the flexible
  one is within one load step of 257.1 from 40x20 on.
- On c-phi both fall with the mesh; the rigid footing stops before q
  levels off from 40x20 on (rough) and on 60x30 (smooth).

So `footing='flexible'` stays the default and **no saved answer changes**.
The GM-304 warning stays on the flexible footing (c = 0, gamma > 0, no
surcharge); the rigid footing has its own two warnings.

## Tests (venv /tmp/v, Python 3.12)

Each judged on pytest's own exit code (0 for every line).

- `fem2d` whole module (`-n 2`): 617 passed, 0 failed (incl.
  `test_footing_capacity.py`, 12 new: 7 fast input/choices tests in
  `TestRigidFootingInputs`, 4 slow coarse-mesh 20x10 tests in
  `TestRigidFooting`, marked slow like the file's other collapse tests).
- `funhouse_agent/tests/test_defaults_used.py -k fem2d`: 46 passed (the new
  `#rigid` probe included).
- `test_footing_capacity.py`, `test_footing_capacity_adapter.py` (3 new),
  `test_defaults_used.py -k footing`, `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py`, `test_phase34_adapters.py`,
  `test_fem2d_adapter_validation.py`, `test_nested_keys.py`,
  `test_eval_suite_v54.py`: 224 passed, 0 failed.
- `validation_examples/test_published_v023_v024.py` (the only example that
  names fem2d; not footing capacity): 13 passed.
- No calc_package, report_ingest or webapp code calls the function; the
  eval suite's FTG-1 uses the default (flexible) path and passes.

## Release note

**fem2d footing capacity: a rigid footing as a choice (GM-305).**
`fem2d_footing_capacity` takes `footing='rigid'`: the footing is pushed
down as one body in settlement steps and q is read from the reactions
(GEC-6 p. 55: Eq. 5-7 assumes a rigid footing), with `base='rough'`
(default, no slip) or `'smooth'`, and returns the pressure-settlement curve.
The default stays `footing='flexible'`, so no saved answer changes: on the
module's uniform meshes the rigid footing reads high and falls as the mesh
is refined (c = 50 kPa, phi = 0, B = 2 m: 347.9, 306.0, 291.0 and 283.1 kPa
on 20x10 to 80x40 against (2 + pi) c = 257.1) and gives no plateau on
a cohesionless soil with no surcharge, where the flexible footing's q_ult is
still set by the mesh (GM-304). Every result now lists the footing model in
`choices`.

## Noted, not fixed

- A mesh graded towards the footing's edges is the likely way to a rigid
  (and a c = 0) answer that converges with the mesh; the module's mesh is
  uniform. A new item only if the owner wants a design-grade c = 0 answer
  from fem2d; it would change no saved output today.
- Flexible c = 20, phi = 25 on 80x40 drops to 173.6 kPa (433.9 on 60x30);
  not investigated (possibly the collapse-by-non-convergence criterion on a
  fine mesh). It is a value a saved 80x40 run would show, so it may deserve
  an item; the lead's call.
- The flexible Prandtl answer is 255.9 kPa on 40x20, 60x30 and 80x40: the
  same step of the 45-step ramp (9.1 kPa); its resolution, not convergence.
