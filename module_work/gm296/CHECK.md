# GM-296: water standing on a sheet pile's excavated side

Branch `claude/gm-296-sheet-pile-passive-water`, 2026-11-05. It also covers
the former GM-24, Das #3b's balanced-water case.

## What the print says

- **UFC 3-220-20 (2025), Fig. 4-41, p. 242.** The figure draws a water level
  on each side, with the note: "Water levels can be different on opposite
  sides due to pumping, tidal fluctuations and other reasons." Sec. 4-7.4
  (p. 243) adds: "The designer should be aware of cases where there is a
  difference in the water level on each side of the wall."
- **Fig. 4-42, p. 243**, the chart for sand. "A phreatic surface can be
  accommodated assuming that the level is the same on both sides of the
  sheet pile."
  - Water is at alpha H below the top on both sides.
  - The chart takes gamma = 124.8 pcf = 2 gamma' (62.4 pcf).
- **Fig. 4-43, p. 245**, the example. A 15 ft wall with water 5 ft below the
  top on both sides, so the water stands 10 ft above the dredge line.
  - Ka 0.25; phi'allow = atan(tan 35 / 1.5) = 25 deg; Kp 3.6.
  - Chart inputs: a = 0.33 and Kp/Ka = 14.4.
  - Read off the chart: D/H = 0.95, so D = 14.25 ft and D_design = 1.2 x 14.25 = 17 ft.
  - M/(gamma' Ka H^3) = 0.6, so Mmax = 31,590 ft-lb/ft.
- **The physics, as drawn in Fig. 4-41.** Below the dredge line the soil in
  front is submerged, so its effective stress grows by gamma' from the dredge
  line. Water standing in front pushes on the wall from its own level down.
  The net water pressure is the difference between the two sides, so equal
  levels give none.

## What changed and why

Before this change, a `gwt_depth_passive` above the dredge line was taken at
the dredge line, so the water standing in front of the wall was left out. Left
out, `gwt_depth_passive` copied the active level, and that was then clamped
to the dredge line in the same way.

**Datum.** Both depths are measured from the top of the wall (the retained
ground surface), as the code always did. The tool text did not say so.

All changes are in `sheet_pile/cantilever.py` and `anchored.py`:

- **`passive_water_level`.** A given level is used at any depth. Left out,
  the level is the excavation level, or `gwt_depth_active` where that is
  deeper. With `gwt_depth_active` also left out, there is no water.
  - This default is exactly the level the old code ended up using, so a call
    that leaves the input out gets the same answer as before.
- **`passive_water_pressure`.** This is the water pressure on the excavated
  face: gamma_w (z - level), also above the dredge line.
  - For a level at or below the dredge line it is the old expression, term
    for term.
  - The cantilever (embedment and moment) and the anchored wall (embedment,
    T and moment) now subtract it on both sides of the dredge line.
- **Passive effective stress is unchanged.** It is gamma' from the dredge
  line, which was already right.
- **`choices`.** When either side has water, the result gets one entry, "Water
  depth on the excavated side (gwt_depth_passive)". It gives the level taken,
  given or default, the print basis, and the alternatives (the retained side's
  level, i.e. balanced water, and the excavation level).
- **`defaults_used`.** A left-out level is now reported as the level the wall
  took (through `how_given(took=)`). Before, it said the active level. That
  was untrue whenever the active level was above the excavation: water behind
  at 3 m on a 4 m wall was reported as 3.0, while the calculation used 4.0.
  - With neither depth given, it still says "no water table".
- **Tool text.** The `gwt_depth_passive` description is new (`_GWT_PASSIVE`,
  shared by both walls and the calc package). `gwt_depth_active` now names its
  datum. The docstrings and the matching rows of `docs/user_manual/manual.html`
  were hand-edited to match.
- **Calc package drawing** (`calc_steps._plot_wall_section`). It draws the
  passive level where it is, and shades water standing in the excavation.
  Before, the line was drawn at the dredge line.
- **DESIGN.md:** new section "Water on the excavated side".

**Equivalence check.** A script ran 60 random walls on master and on the
branch, with the passive level left out or at or below the dredge line. It
mixed every pressure method, surcharges and cohesion. Results were
bit-for-bit identical: D, M, depth of M, the anchored D and T, and the
water-pressure profile.

## Before and after

**UFC Fig. 4-43, on the chart's own basis.** Inputs: gamma = 124.8 pcf,
gamma_w = gamma' = 62.4 pcf, Kp/Ka = 14.4 as printed (Rankine phi' 35.65 deg),
FOS_passive 1.0 because the print's Kp is already the allowable one.

| | D0/H | D_design | M/(gamma' Ka H^3) |
|---|---|---|---|
| Print, Fig. 4-42 chart (UFC full method) | 0.95 | 17 ft | 0.6 |
| Hand solve, Fig. 4-41 full method (P1 - P2 + P3 = 0, moments about F) | 0.934 | 16.8 ft | – |
| Module now (Simplified Method) | **0.868** | **15.6 ft** | **0.565** |
| Module before GM-296 (passive water taken at the dredge line) | 1.461 | 26.3 ft | 1.87 |

The hand solve reproduces the chart reading to within 2 %. The module
remains 8 % short on D. That gap is GM-290 P4: the module is the Caltrans
Simplified Method (moments about O), not UFC's full method. Both are printed
methods, and the gap was not changed here.

The print's own soil (gamma 122.4 / gamma' 60 pcf, Ka 0.25 from phi' 35 and
Kp 3.6 from phi'allow 25) cannot be entered directly. The module takes one
phi per layer and divides Kp by FOS_passive. So the pin is taken at the
printed Kp/Ka ratio. The chart prints no D for 122.4/60 pcf either.

**A made-up 5 m sand wall.** Phi' 32 deg, gamma 19 kN/m³, water 2 m below the
top behind the wall. Anchored walls have the anchor at 1 m.

| water in front | cantilever D (m) | M (kN-m/m) | anchored D (m) | T (kN/m) | M (kN-m/m) |
|---|---|---|---|---|---|
| left out, or at 5.0 m (before and now) | 14.48 | 1120.0 | 5.76 | 105.1 | 267.1 |
| 3.5 m (before: same as 5.0 m) | 11.89 | 713.4 | 4.47 | 77.2 | 164.7 |
| 2.0 m, balanced (before: same as 5.0 m) | 8.61 | 324.4 | 3.00 | 42.8 | 70.1 |

## Tests

- `sheet_pile/` — 89 passed, including the new `tests/test_gm296_passive_water.py`.
  - It covers: the level taken; the face pressure; left out = given at the
    resolved level; values pinned from master; balanced water giving zero net
    water; the net between two levels; Fig. 4-43; `choices` / `defaults_used`.
- A broader run, all at once. It covered `sheet_pile`, `calc_package`,
  `validation_examples`, `foundry_test_harness`, `test_public_text.py`,
  `test_page_fingerprints.py`, and these `funhouse_agent/tests` files:
  `test_defaults_used.py`, `test_reviewers.py`, `test_gm212_wall_sliding.py`,
  `test_calc_package_tool_parity.py`, `test_calc_package_derived_inputs.py`,
  `test_calc_package_adapter.py`, `test_earth_pressure_sources.py`,
  `test_nested_keys.py`, `test_calc_package_figure_layout.py` and
  `test_adapter_ergonomics.py`.
  - Result: 2322 passed, 26 skipped (pdflatex, pydiggs, python-ags4 not
    installed), 1 xfailed, **1 failed**.
  - The failure is `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
    (logx/logy inert). It **fails the same way on master** (checked in a
    worktree of 1d39240). It is unrelated, and I noted it rather than fixing it.
- After the drawing edit, a re-run of the sheet / wall / page subset of the
  above passed: 213 passed.
- `funhouse_agent/tests/test_defaults_used.py -k "sheet_pile or passive_water"`:
  19 passed.
  - `test_a_passive_water_table_left_out_follows_the_active_side` was updated.
    It now expects the level the wall took, max(3.0, H) = 4.0, instead of 3.0.
- `webapp/tests/test_page_fingerprints.py`: passed with no re-record. The
  fingerprints do not cover the parameter descriptions.

## Release note

**sheet_pile (cantilever and anchored walls): water standing in front of the
wall is now counted.** A `gwt_depth_passive` shallower than
`excavation_depth` means water standing in the excavation above the dredge
line. Before, it was taken at the dredge line, so its pressure on the wall was
left out. Now it acts on the excavated face from its level down, as UFC
3-220-20 Figs. 4-41 and 4-43 draw it. The same level on both sides (balanced
water) gives no net water pressure.

Only calls that gave a passive level above the dredge line move, and they get
a shorter embedment, a smaller moment and a smaller anchor force. Example: a
5 m wall in sand with water 2 m down behind it, and the same water in front.
The cantilever D goes from 14.5 m to 8.6 m and M from 1120 to 324 kN-m/m. The
anchored wall's T goes from 105 to 43 kN/m.

Left out, the level is the excavation level, or the retained side's level
where that is deeper. That is the answer the tool gave before. `defaults_used`
now reports that level; before, it reported the retained side's level.
Results with water now carry a `choices` entry for the excavated side's level.

## Noted, not fixed

- **Negative net water in the cantilever.** The embedment equilibrium leaves
  out a net water pressure toward the retained side (front water higher than
  behind), as it did below the dredge line before. The cantilever moment
  diagram and the anchored wall take it with its sign. The new region above
  the dredge line follows each loop's existing rule.
  - It matters only when the front water is higher than the back. It is
    conservative, and it is now stated in DESIGN.md.
- **`soe` has its own excavation-side water.** `soe.geometry.gwt_depth_excavation`
  is "never taken above the excavation level", clamped in
  `soe/free_earth.py:41-42` and printed as such in `soe/calc_steps.py:112`.
  It is stated, not silent, so I did not change `soe`, as the task asked.
- **Soldier-pile mode (OWN-21).** `sheet_pile` has no soldier-pile mode, so
  nothing changed there.
- **Calc package fallback.** `sheet_pile/calc_steps._compute_pressures` has a
  fallback path, used only when a result has no `profile`. That path takes
  passive stress from the top of the soil column, not from the dredge line.
  Every result built today has a profile.
- **Retired Foundry wrappers.** `foundry/sheet_pile_agent_foundry.py` and
  `foundry/calc_package_agent_foundry.py` describe `gwt_depth_passive` as "m
  from excavation". That is wrong: the datum is the top of the wall. They are
  retired, so I did not edit them.
- **The test failure on master**:
  `test_defaults_used.py::...[profile_figure|plot_data]` (logx/logy inert).
