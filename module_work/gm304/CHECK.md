# GM-304 (fem2d) — the soil's weight in a first stage whenever gamma > 0

Branch `claude/gm-304-fem2d-weight-first`. All test values are made up.

## What the print says

- **GEC-6** (FHWA-SA-02-054, *Shallow Foundations*, 2002), Sec. 5.2.2,
  p. 53:
  - Eq. 5-7: q_ult = c Nc + q Nq + 0.5 gamma B_f Ngamma, a centrically
    loaded strip footing; gamma the unit weight of the soil beneath it.
  - Eq. 5-8: Ngamma = 2 (Nq + 1) tan phi (Caquot and Kerisel 1948, used by
    FHWA and AASHTO). The same page notes published Ngamma values range
    from about one half to double these (Chen & McCarron 1991).
  - Eq. 5-7 assumes a RIGID strip footing (p. 55).
- The weight term is a property of soil already under its own weight: the
  footing is loaded onto the in-situ stress. That is what GM-11's surcharge
  path already did (stage 1: weight and q; stage 2: the footing).

## What changed

`fem2d/analysis.py::analyze_footing_capacity`: the staged path now runs
whenever `surcharge > 0 or gamma > 0` (was `surcharge > 0`). With
`surcharge = 0` stage 1 is the soil's weight alone; stage 2 ramps the
footing pressure from 0 to q_max. Only a weightless soil with no surcharge
keeps the single ramp from zero, so its answer is unchanged to the last
digit. No new inputs or result keys; `Nc_backfigured` is unchanged (it was
already None whenever gamma > 0).

Text that described the old staging now says what the code does: the
`gamma` docstring and the `surcharge` default note in `analysis.py`, the
adapter's `gamma` description (`funhouse_agent/adapters/fem2d_adapter.py`),
its row in `docs/user_manual/manual.html`, and `fem2d/VALIDATION.md` §3.
The page fingerprints did not move (per-method parameter descriptions are
not in them), so there is no re-record commit. `choices` / `defaults_used`
are untouched: the footing tool declares q_max, domain sizes, max_iter and
tol, none of which changed.

## Numbers (B = 2 m, T6, defaults 40 x 20 and 45 steps unless stated)

Before = master 855bf2f on the same inputs.

| Case | Closed form (GEC-6 Eq. 5-7, 5-8, p. 53) | Before | After |
|---|---|---|---|
| (a) c = 0, phi = 30, psi = 0, gamma = 18 | 0.5 x 18 x 2 x 22.40 = 403.2 | 0.0 | 43.0 |
| (a) same, psi = 30 | 403.2 | 0.0 | 43.0 |
| (a) psi = 0 / 30, q_max 150, 60 steps (finer ramp) | 403.2 | — | 52.5 / 50.0 |
| (b) c = 20, phi = 25, gamma = 18 | 20 x 20.72 + 0.5 x 18 x 2 x 10.88 = 610.2 | 433.9 | 455.6 (+5.0 %) |
| (b) same, surcharge 0.01 | 610.4 | 455.7 | 455.7 |
| (c) c = 0, phi = psi = 30, gamma = 18, 20 x 10: q = 0 | — | 0.0 | 86.03 |
| (c) same, q = 0.01 | — | 86.07 | 86.07 (same 6 steps) |
| (c) c = 20, phi = 25, 20 x 10: q = 0 / q = 0.01 | — | — | 499.0 / 499.1 (same 23 steps) |
| (d) c = 50, phi = 0, gamma = 0, no surcharge | 257.1 | 255.9370565342475 | 255.9370565342475 |
| c = 50, phi = 0, gamma = 18, nu = 0.3 | 257.1 (Ngamma = 0) | 255.94 | 255.94 |

- (c): q = 0 and q = 0.01 now converge the same number of steps; the
  difference is the 0.01 kPa start, far inside the step (14.3 and 21.7 kPa).
- (a): the FE answer is about 11-13 % of 0.5 gamma B Ngamma. This is the
  module's known limit, not tuned: the footing is a smooth, FLEXIBLE uniform
  pressure, not Eq. 5-7's rigid footing, and on c = 0 sand the unconfined
  surface at the footing edges yields at a small pressure (psi made no
  difference, which points at that local edge failure rather than the
  general mechanism). GM-11's case d already showed the Ngamma term carried
  at about half Eq. 5-8 with q > 0; with q = 0 the edge has no confinement
  at all. Reported for information; no test pins it against the closed form.
- (a) is set by the mesh, not by the footing: q_max 200 with 80 steps gave
  102.5, 52.5, 35.0 and 27.5 kPa on 20 x 10, 40 x 20, 60 x 30 and 80 x 40,
  about 105 kPa per metre of element width at the footing edge, falling
  towards zero as the mesh is refined. It is not a bearing capacity.
- (b): -25 % against Eq. 5-7, for the same reason in the Ngamma term. It
  also falls with the mesh: 499.0, 455.6 and 433.9 kPa on 20 x 10, 40 x 20
  and 60 x 30.

## Tests (venv /tmp/v, Python 3.12, `-n 8`)

- `fem2d` whole module: 605 passed, 0 failed (incl.
  `test_footing_capacity.py` 17 passed: 5 new in `TestWeightFirstStage`,
  marked slow like the file's other collapse tests, on the coarse 20 x 10
  mesh).
- `funhouse_agent/tests/test_defaults_used.py -k fem2d`: 43 passed.
- `test_eval_suite_v54.py`, `test_footing_capacity_adapter.py`,
  `test_phase34_adapters.py`, `test_fem2d_adapter_validation.py`,
  `test_nested_keys.py`, `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py`: 145 passed.
- `validation_examples/test_published_v023_v024.py` (the only one naming
  fem2d; it does not use footing capacity): 13 passed.
- No calc_package, report_ingest or webapp code calls the function.

## Release note

**fem2d footing capacity on a soil with weight (GM-304).** With no
surcharge, `analyze_footing_capacity` (`fem2d_footing_capacity`) used to
ramp the soil's weight together with the footing load, so a frictional soil
could not build its strength from its own weight: a c = 0 sand returned
q_ult = 0. The soil's weight is now applied first whenever gamma > 0, as
the surcharge path already did, and the footing is loaded onto it. Every
saved no-surcharge answer with gamma > 0 and phi > 0 changes: a c = 0
soil with no surcharge now returns a non-zero q_ult (43 kPa for phi = 30,
gamma = 18, B = 2 m on the default mesh), but that value is set by the mesh
(about halved when the elements are halved) and is not a bearing capacity;
c = 20, phi = 25 goes from 433.9 to 455.6 kPa (+5 %) on the default mesh,
a change smaller than its spread with the mesh (499.0, 455.6 and 433.9 kPa
on 20x10, 40x20 and 60x30). With phi = 0 the weight adds nothing and q_ult
does not move, but max_displacement_m changes on every gamma > 0 call
without a surcharge, as it now holds the whole self-weight settlement
(c = 50, phi = 0, gamma = 18: 31.6 to 38.4 mm); a weightless soil, and any
call with a surcharge, are unchanged.

## Noted, not fixed

- The sand q_ult stays far below 0.5 gamma B Ngamma and is set by the mesh
  (smooth flexible pressure, edge failure). A rigid, rough footing
  (prescribed uniform settlement) would be the way to approach Eq. 5-7;
  that is a new feature,
  not this item.
