# GM-11 and GM-156 (fem2d) — check

Branch `claude/gm-11-156-fem2d`. All test values are made up.

## GM-11: `analyze_footing_capacity` applies the surcharge beside the footing

### What the print says

- **GEC-6** (FHWA-IF-02-054, *Shallow Foundations*, 2002), Sec. 5.2.2,
  pp. 51-53:
  - Eq. 5-1: gross q_ult = c Nc + q Nq, with **q = gamma Df the surcharge at
    the base of the footing** (p. 51).
  - Eq. 5-2: Nq = e^(pi tan phi) tan^2(45 + phi/2) (p. 51).
  - Eq. 5-4: Nc = 2 + pi = 5.14 for phi = 0 (p. 52).
  - Eq. 5-7: q_ult = c Nc + q Nq + 0.5 gamma B Ngamma; Eq. 5-8:
    Ngamma = 2 (Nq + 1) tan phi (p. 53).
- **UFC 3-220-20** (DM 7.2), Sec. 5-3.1.2: the soil above the bearing
  elevation is replaced by an equivalent surcharge (Fig. 5-4(a)); Eq. 5-17,
  phi = 0 saturated: q_ult = su Nc + sigma_zD (the weight adds nothing).

The bearing-capacity equation's q is a pressure on the bearing level beside
the footing. The FE model's ground surface is the bearing level, so the
surcharge belongs there as a surface pressure. That is the approach these
sections describe, so no other way was needed.

### What changed

`fem2d/analysis.py::analyze_footing_capacity`:

- **`surcharge > 0`:** two stages.
  - Stage 1 brings the soil's weight and q over the WHOLE surface (the
    footing strip included) into equilibrium. This is the stress at the
    bearing level before the footing is loaded.
  - Stage 2 holds q beside the footing and ramps the pressure under it
    from q to q_max (`run_nl(..., F_start=stage-1 load, u_init/sigma_init
    from stage 1)`).
  - q_ult = q + (q_max - q) x the fraction of the ramp carried: the gross
    pressure of Eq. 5-1 / 5-7. `collapse_load_fraction` is that fraction.
  - A new result key, `surcharge_kPa`.
  - A negative surcharge is refused, and so is a `q_max` at or below the
    surcharge.
- **`surcharge = 0`:** the old single ramp, untouched. The test
  `test_no_surcharge_path_unchanged` pins 255.9 kPa for c = 50, B = 2.
- Doc changes: the comment that said the surcharge was "out of scope" is
  gone. The docstring, the adapter's `surcharge`, `gamma` and `q_ult_kPa`
  descriptions, and the two matching rows in `docs/user_manual/manual.html`
  now say what the code does. `fem2d/VALIDATION.md` §3 has the table below.

### Numbers (B = 2 m, T6 40 x 20, domain 10B x 5B, defaults otherwise)

| Case | Print | Closed form (kPa) | Before | Now | Now vs print |
|---|---|---|---|---|---|
| a. c = 50, phi = 0, gamma = 0, q = 36 (weightless) | GEC-6 Eq. 5-1, 5-4 | 5.14 x 50 + 36 = 293.1 | 260.5 | 295.8 | +0.9 % |
| b. c = 0, phi = psi = 20, gamma = 0, q = 20 (weightless) | GEC-6 Eq. 5-1, 5-2 (Nq = 6.40) | 128.0 | 0.0 | 130.9 | +2.3 % |
| c. c = 50, phi = 0, gamma = 18, nu = 0.49, q = 36 (with weight) | UFC 3-220-20 Eq. 5-17; GEC-6 Eq. 5-7 with Ngamma = 0 | 293.1 | 260.5 | 295.8 | +0.9 % |
| d. c = 10, phi = psi = 20, gamma = 18, q = 18 (with weight) | GEC-6 Eq. 5-7, 5-8 | 148.3 + 115.2 + 97.0 = 360.5 | 166.6 | 316.0 | -12 % |
| c = 50, no surcharge (control) | Eq. 5-4 | 257.1 | 255.9 | 255.9 | unchanged |

- Case b uses associated flow (psi = phi), the rigid-plastic basis of
  Prandtl-Reissner's Nq. Before the fix, a weightless sand footing collapsed
  at the first load step (q_ult = 0) because the surcharge never acted.
- Case d's gap is only in the Ngamma term. Taking c Nc and q Nq as printed,
  the FE carries 0.5 gamma B Ngamma with Ngamma ≈ 2.9, where Eq. 5-8 gives
  5.39. No held reference prints an exact Ngamma to judge either by, so this
  row is shown for information and is not tested.
- The "before" column comes from the master code (1d39240) run on the same
  inputs.

## GM-156: `fem2d_excavation` states its soil model and K0

The two trace reviews the item cites (`module_work/live_smoke/runs/w12-...`
and `w14-...`) are gitignored and not in this clone. BACKLOG's account of
them is what I worked from: F51 and F51j stated no soil model or K0.

`analyze_excavation`'s `choices` now has two entries per soil layer, after
the existing wall-contact entries (`choices[0]` is still the wall contact):

- **`soil model, <layer>`.** This is what `solver._material_arrays` does:
  - "Hardening Soil (model 'hs')" when the layer says so (how: given).
  - Otherwise "Mohr-Coulomb, elastic-perfectly plastic" where c or phi is
    above zero (how: default).
  - Otherwise "linear elastic" (how: default).
  - Basis: fem2d DESIGN.md (Mohr-Coulomb; Hardening Soil per Schanz,
    Vermeer & Bonnier 1999); UFS 3-220-10 (2026) Sec. 4-6.2 for linear
    elastic.
  - Alternatives: the other models the module takes.
- **`K0 for the in-situ stresses, <layer>`.** This is what `_layer_k0` does:
  - The layer's own 'K0' (given).
  - Otherwise Jaky's 1 - sin phi' (default): FHWA NHI-06-089 (Soils and
    Foundations Vol. II) Eq. 10-2, p. 10-10. Eq. 10-3, p. 10-11, is named
    as the overconsolidated alternative.
  - Otherwise, for phi = 0, nu/(1 - nu) (default): UFS 3-220-10 (2026)
    Sec. 4-6.4.1, which says a laterally constrained linear elastic mesh
    under gravity takes K0 = nu/(1 - nu). The text JSON gives that
    section's number but not its page.

Defaults are nested in `soil_layers`, so their `how` is set directly, as
`sheet_pile` does. Given values go through `how_given(..., within=
'soil_layers')`, so a calc package's "assumed" still reaches them.
`defaults_used` is unchanged: neither 'K0' nor 'model' is a top-level
declared default. No calculated value changes. The adapter's `choices`
return description says what the list now holds.

## Tests (venv /tmp/v, Python 3.12)

- `fem2d` (whole module, `-n 8`): 600 passed, 0 failed.
- `fem2d/tests/test_footing_capacity.py`: 12 passed (5 new: one
  validation, four slow surcharge tests).
- `fem2d/tests/test_excavation_soil_choices.py` (new, 3) +
  `test_excavation_staged.py` (+1) + `test_excavation_interface.py`:
  56 passed.
- `funhouse_agent/tests/test_defaults_used.py -k fem2d`,
  `test_footing_capacity_adapter.py`, `test_fem2d_adapter_validation.py`,
  `test_nested_keys.py`, `test_phase34_adapters.py`,
  `test_eval_suite_v54.py`, `webapp/tests/test_public_text.py`,
  `webapp/tests/test_page_fingerprints.py`: all passed (69 + 145).
  The fingerprints did not move: per-method descriptions are not in them.
- No validation_examples, calc_package or report_ingest code calls either
  function.

## Release note

**fem2d footing capacity with a surcharge (GM-11).** Before this release,
`analyze_footing_capacity` (`fem2d_footing_capacity`) ignored its
`surcharge` input except to size the load ramp, so an embedded footing's
overburden q = gamma Df gave no resistance. The surcharge is now a pressure
on the ground surface beside the footing:

- The soil's weight and q are applied first, then the footing pressure is
  raised from q.
- `q_ult_kPa` is the gross pressure (GEC-6 Eq. 5-1).
- On the checks above, q_ult rises by +14 % (c = 50 kPa clay, q = 36 kPa)
  and from 0 to 131 kPa (weightless sand, q = 20 kPa). It is within 1-2.3 %
  of c Nc + q Nq.
- Only calls that gave a `surcharge` change.

`fem2d_excavation`'s `choices` now state each layer's soil model and K0;
no value changes there.

## Noted, not fixed

- With `surcharge = 0` and gamma > 0 the soil's weight is still ramped
  with the footing load (kept so the surcharge = 0 answer does not change).
  For c = 0 that path cannot collapse: every load scales together and a
  frictional soil's yield condition does not change with scale. So
  `surcharge = 0` and a tiny surcharge give different answers for a heavy
  soil. **This changes values in saved outputs, so it is worth a BACKLOG
  item:** apply the weight in stage 1 whenever gamma > 0.
- The surcharge is uniform and acts on both sides. One-sided or unequal
  overburden is not modelled.
