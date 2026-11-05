# GM-299: the wave equation's cushion and helmet from the print

BACKLOG GM-299, 2026-11-05. Read from `geotech-references/docs/`: GEC-12
(FHWA-NHI-16-009, 2016) Vol 2, Ch. 12 (pp. 153-246), Ch. 15 (pp. 349-353)
and Appendix C (pp. 519-541); FHWA NHI-06-089 Vol II Appendix A.9; UFC
3-220-20 (2025) Sec. 6-9. Page numbers are the printed ones. Scope:
`wave_equation/` and `funhouse_agent/adapters/wave_equation.py` (the three
tools and `wave_equation_package`, which runs the bearing-graph tool's
code).

## What the print says

- **No table of cushion and helmet data by hammer is held.** Appendix C
  (pp. 519-541) lists each hammer's make, model, type, rated energy, ram
  weight and stroke, nothing else. GEC-12 Vol 2 p. 225: "GRLWEAP offers an
  extensive data file for those situations in which the contractor's
  available equipment is unknown"; p. 226: "Ideally, the contractor would
  provide the above drive system data for his actual hammer system", else
  the program's "Manufacturer's Recommended Driving System" or its
  "Cushion Material Properties" list (p. 227). p. 209: "for most commonly
  used cushions the Help section of the wave equation software contains
  extensive material property tables". Neither is printed in a held
  reference. NHI-06-089 App. A.9 (p. A.9-3) also sends the reader to the
  GRLWEAP help screen. UFC 3-220-20 prints no cushion values.
- **Complete driving systems printed for a named hammer, hammer cushion
  only, on a steel pile**: Example 6, Table 12-2 (p. 183):

  | | Vulcan 014 | ICE 42S |
  |---|---|---|
  | Helmet weight, kips | 1.67 | 2.05 |
  | H. cushion material | Nycast | Blue Nylon |
  | H. cushion E, ksi | 208 | 175 |
  | H. cushion area, in2 | 234 | 398 |
  | H. cushion thickness, in | 6.0 | 2.0 |
  | H. cushion COR | 0.91 | 0.92 |

  Stiffness E A / t (p. 155): 8,112 kips/in = 1,420,629 kN/m (the "about
  1.4 million kN/m" of the backlog item) and 34,825 kips/in = 6,098,792
  kN/m.
- **Partial or concrete-pile systems** (not offered by name): Example 1,
  Delmag D12-42, 1 in aluminum + 1 in Conbest, 227 in2, 530 ksi averaged,
  helmet 1.7 kips, no COR (p. 163); Example 3, Conmaco C 160, Nylon (no E
  or COR printed), helmet 4.07 kips, 10 in plywood pile cushion (p. 167);
  Example 4, Junttan HHK 3A on a concrete pile, 8 in, 360 ksi, COR 0.90,
  250 in2, helmet 1.0 kip, plus a plywood pile cushion (pp. 174-175);
  Example 5, Vulcan 08 on a concrete pile, 8.5 in Hammortex, 148 in2, no E
  or COR in the text, helmet 2.6 kips (p. 179).
- **COR**: 1.0 elastic, 0.0 plastic; "the default value for the COR of
  wood is 0.50", 0.25 for a badly worn softwood cushion (p. 156); hammer
  cushions of wood are prohibited (p. 352); a refined analysis adjusts COR
  between 0.10 and 1.0 (p. 216).
- **Helmet weight**: helmet, hammer cushion, striker plate, inserts and
  all components between hammer and pile; may be zero (p. 226).

## What changed and why

The module defaulted a left-out cushion to 500,000 kN/m, COR 0.80, and a
left-out helmet to 5 kN. None is printed. Since no held reference prints
cushion and helmet properties by hammer except Table 12-2, the fix takes
that table where it covers the hammer and asks for the inputs elsewhere.

- `wave_equation/driving_systems.py` (new): Table 12-2's two systems in the
  print's units (`DrivingSystem`, `PRINTED_DRIVING_SYSTEMS`,
  `get_driving_system`), the stiffness E A / t, and `PRINTED_OPTIONS`, the
  text every refusal carries (the submittal, the two systems, E A / t and
  series springs, the examples' COR 0.90-0.92, wood 0.50/0.25, the
  examples' helmets with pages, and that Appendix C and GRLWEAP's data
  file give nothing else).
- Library: `Cushion.cor` and `make_cushion_from_properties(cor=)` have no
  default; `helmet_weight` in `simulate_blow`, `generate_bearing_graph` and
  `drivability_study` defaults to None and is refused when left out
  (`require_helmet_weight`).
- Tools (`single_blow`, `bearing_graph`, `drivability`, and the package):
  - new input `driving_system` ('Vulcan 014' or 'ICE 42S');
  - no cushion given: the named hammer's printed system (only `hammer_name
    'Vulcan 014'` has one), or the one `driving_system` names; a given
    `cushion_cor` or `helmet_weight` replaces the system's;
  - a cushion given (stiffness or properties) needs `cushion_cor` and
    `helmet_weight`; a cushion and `driving_system` together are refused;
  - anything else is refused with `PRINTED_OPTIONS` (e.g. `hammer_name
    'Vulcan 010'` with no cushion, which before ran on the unprinted
    defaults).
- **Open methods.** Every run's `choices` carries one entry, "hammer
  cushion and helmet": the stiffness, COR and helmet used, how (default,
  given), a basis naming the submittal (p. 226) and both printed systems,
  and the alternatives. The same words whichever way the values came, so a
  run given its own `defaults_used` back reads the same (the GM-158
  guard). A named system run with another hammer says so. `defaults_used`
  lists `cushion_stiffness`, `cushion_cor` and `helmet_weight` with the
  values taken when the hammer's printed system supplied them; with no
  value to take, the declared case reads "the named hammer's printed
  driving system (hammer_name 'Vulcan 014'); any other hammer is refused".
- Tool text: `cushion_stiffness`, `cushion_cor` and `helmet_weight` lose
  their defaults and say the above with pages; `driving_system` describes
  both systems. The package's parameter list is built from the tool's, so
  it says the same, and its PDF prints the choice entry and the helmet row
  (`calc_steps` leaves the helmet row out only when an analysis dict
  carries none).
- `DESIGN.md` updated; `cushion.py` and `time_integration.py` docstrings
  say the value is required.
- Tests: `wave_equation/tests/test_cushion_cor_default.py` rewritten (12
  tests: the refusals and their printed options, Table 12-2 as printed, the
  Vulcan 014's default, a named system, given COR and helmet, the old
  values given). Library tests that relied on the old defaults now pass
  500,000 kN/m, COR 0.80 and 5 kN explicitly (made-up values), so their
  pinned numbers still check the solver. `method_probes.json`: the
  custom-hammer probes name `driving_system 'Vulcan 014'`, and a `#printed`
  variant of each probes `hammer_name 'Vulcan 014'` with no cushion, so the
  guard checks the three new defaults. The parity cases and the module
  comparator use the Vulcan 014's printed system.

## Before and after

Made-up steel pipe, 0.32 m x 10 mm, 18 m, all embedded, half the
resistance on the shaft, the other inputs at their defaults.

| Case | Before (500,000 kN/m, COR 0.80, 5 kN) | After |
|---|---|---|
| Vulcan 014, 900 kN | 45.1 blows/m, 145.4 MPa | 46.3 blows/m, 184.5 MPa (Table 12-2: 1,420,629 kN/m, 0.91, 7.43 kN) |
| Vulcan 014 bearing graph 300/600/900/1200/1500 kN, blows/m | 15.0 / 29.6 / 45.1 / 65.3 / 97.9 | 15.5 / 30.7 / 46.3 / 65.9 / 94.3 |
| same, max compression MPa | 130.1 / 133.6 / 145.4 / 183.8 / 219.8 | 177.8 / 181.2 / 184.5 / 187.9 / 213.2 |
| Vulcan 010, 900 kN | 62.5 blows/m, 137.2 MPa | refused: give a cushion, COR and helmet, or driving_system |
| Delmag D30-32, 900 kN | 29.8 blows/m, 258.3 MPa | refused, likewise |
| Vulcan 010 with the old three values given | 62.5 blows/m, 137.2 MPa | 62.5 blows/m, 137.2 MPa (unchanged) |

GEC-12 Example 6 as the module's tests run it (Table 12-2 given
explicitly) does not move.

## Tests run (Python 3.12, `pip install -e .`)

- `wave_equation`, `calc_package`, `validation_examples`,
  `funhouse_agent/tests/test_defaults_used.py`,
  `test_calc_package_figure_layout.py`, `test_calc_package_tool_parity.py`,
  `test_adapter_ergonomics.py`, `test_calc_package_adapter.py`,
  `test_top_level_params.py`, `test_gm179_unused_inputs.py`: **2,059
  passed, 26 skipped, 1 xfailed, 1 failed**. The failure is
  `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  (logx/logy change nothing), which fails the same way on master without
  this change (checked with the change stashed): not this item's.
- `foundry_test_harness`, `webapp/tests/test_public_text.py`,
  `webapp/tests/test_page_fingerprints.py`: **141 passed**. The
  fingerprints did not move (no page prompt or page tool changed), so
  nothing was re-recorded.
- The skips are pdflatex and optional backends not installed here.

## Release note

wave_equation: a run that leaves out the hammer cushion and helmet no
longer takes 500,000 kN/m, COR 0.80 and a 5 kN helmet, which no held
reference prints. With `hammer_name 'Vulcan 014'` it takes GEC-12 Vol 2
Table 12-2's driving system (Nycast, 1,420,629 kN/m, COR 0.91, helmet 1.67
kips = 7.43 kN; on the made-up 18 m pipe at 900 kN 46.3 blows/m and 184.5
MPa against 45.1 and 145.4). For every other hammer such a run is now
refused: give `cushion_stiffness` or `cushion_area` + `cushion_thickness` +
`cushion_E`, with `cushion_cor` and `helmet_weight` (ideally the
contractor's submittal, GEC-12 Vol 2 p. 226), or `driving_system` ('Vulcan
014' or 'ICE 42S', Table 12-2). A cushion given without its COR, or
without the helmet weight, is refused too (before: COR 0.80, 5 kN). In
the library, `Cushion` and `make_cushion_from_properties` need `cor`, and
`simulate_blow`, `generate_bearing_graph` and `drivability_study` need
`helmet_weight`. Results now list the cushion and helmet in their choices.

## Noted, not fixed

- The retired `foundry/wave_equation_agent_foundry.py` still defaults to
  500,000 kN/m, COR 0.80 and 5 kN, passing them to the library itself.
  The retired `foundry/calc_package_agent_foundry.py` calls
  `generate_bearing_graph` with no `helmet_weight`, so its wave equation
  package now raises "helmet_weight is required" where it used to take
  5 kN. Neither is on the app's path (CLAUDE.md: retired route), and no
  test calls the second; both are left as they are (their deletion is
  BACKLOG INF-06).
- `generate_all_pdfs.py` and `docs/gallery/build_gallery.py` now pass the
  old 5 kN helmet explicitly (their cushions were already given) so they
  run as before; their made-up cushions are not printed values either.
  Neither has been run here.
- The model has one cushion spring and no pile cushion (DESIGN.md), so
  Examples 3-5's concrete-pile systems cannot be offered by name; Example
  1's lacks a COR.
- `defaults_used` lists `hammer_custom_name: 'Custom'` on runs with a
  named hammer (already so on master).
- `simulate_blow`'s docstring calls its `cushion` the "pile cushion
  (between helmet and pile head)"; it is the hammer cushion. Wording only.

## Review fixes (2026-11-04)

An independent review checked the change against GEC-12 Vol 2: all 12
values of Table 12-2 (p. 183), the examples' systems (pp. 163, 167, 174,
179) and pp. 155-156, 209, 216, 225-226, 352 and 519 say what the code
says. GEC-12 Vol 3 and NHI-06-089 print no other system by hammer. Fixed
after it:

- This folder was `module_work/gm-299/`; it is now `module_work/gm299/`,
  the name `wave_equation/DESIGN.md` gives and the other GM folders use.
- The user manual's wave-equation worked example ran a custom ram with
  no cushion, which the tool now refuses. `gen_worked_examples.py` and
  the ask in `manual_content.py` now name `hammer_name 'Vulcan 014'`
  (its printed system), with a note on the other hammers. Without a full
  rebuild (which moves the cover version), only that entry of
  `worked_examples.json` was re-run through the live dispatch, and in
  `manual.html` only the worked example block and the five
  wave-equation parameter tables (single_blow, bearing_graph twice,
  drivability, wave_equation_package) were re-rendered with
  `build_manual`'s own helpers. Their rows differ only in
  `cushion_stiffness`, `cushion_cor`, `helmet_weight` and the new
  `driving_system`.
- The note above on the retired foundry wrappers: the calc package one
  now raises where it defaulted.

Noted by the review, not fixed: `generate_all_pdfs.py`'s analysis dict
carries no `helmet_weight`, so its sample PDF now leaves the helmet row
out (it printed the old 5 kN default); with only `cushion_cor` or
`helmet_weight` given for the Vulcan 014, the one choice entry says
"default" while `defaults_used` leaves the given one out; a calc package
that marks `driving_system` as assumed drops the "printed for another
hammer" sentence; a custom hammer's refusal reads "for the Custom".
