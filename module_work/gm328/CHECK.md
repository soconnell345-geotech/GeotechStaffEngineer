# GM-328 CHECK: sample calc packages rebuilt by script, the calc helper told of the worked examples, fem2d's package

Branch `claude/gm-328-sample-calc-packages`, from master 2feea3d, 2026-11-13.
Brief: `module_work/cloud_briefs/AUDIT_STAGE2.md`, section GM-328.

## What was found on today's master (confirm first)

| Audit finding (`results/calc_examples.md`, AUDIT §2.2 X5) | On master 2feea3d | Done |
|---|---|---|
| No `fem2d_package`; fem2d is in the renderer's `_IMPORT_MAP` (`calc_package/__init__.py`) with steps, figures and tables | Confirmed: `METHOD_REGISTRY` of `funhouse_agent/adapters/calc_package.py` had 16 `*_package` methods, none for fem2d; `render_figures` / `render_tables` need a named package, so fem2d's figures and tables had no agent route at all | Fixed: `fem2d_package` |
| The calc helper's prompt never names `worked_examples` | Confirmed: no "worked" in `_CALC_PREAMBLE`, `_CALC_FRAMING` or the other `_CALC_*` texts; the module is in the helper's scope (`ANALYSIS_MODULES`) | Fixed: one instruction, `_CALC_WORKED_EXAMPLES` |
| The committed samples are stale and three modules have none | Confirmed (`docs/examples/*.html` 2026-06-11, `sample_pdfs/` and `sample_calc_package.pdf` 2026-02-17) | The script and its table; the old files are left alone (OWN-144) |
| (4) the manual's "Plotly interactive viewers" have no agent route | Not in this task's section | Not done |
| (5) packages for settlement's separate methods, pystrata, signals, the structural engines | "Not this task" | Not done |

## 1. The script and its table

- `docs/examples/build_sample_packages.py [OUT_DIR] [--only NAME ...] [--no-pdf] [--list]`.
  For each row it makes the agent's own call, `call_agent("calc_package", <method>, params)`,
  then `call_agent("calc_package", "render_figures", {package: <method>, ...the same inputs})`
  for the figures, then the package call again with `format="pdf"`.
- `docs/examples/sample_packages.json` is the one table: `module`, `method`, `origin`,
  `example_id`, `source` (the printed citation), `params` (the whole call) and, for a
  worked-example row, `example_call` and `example_tool`. No package input is in the script.
- Every row's `params` names all its inputs in `given` (the example states them), so a
  sample shows the Given Inputs section filled, and carries `project_name`
  "Sample calc package" and the example id as `project_number`.
- Output (default `docs/examples/out/`, added to `.gitignore`; nothing committed):
  `<module>__<method>.html`, `.pdf`, `tables/<module>__<method>__<table>.csv`,
  `figures/<module>__<method>__<nn>_<title slug>.png`, `index.json`, `INDEX.md`.
  Names come from the table and from each figure's order and title, never the clock; a
  rebuild first removes that package's own files, and `--only` keeps the other packages'
  index entries. Each package is built in a temporary folder and copied, because the
  package and figure writers add `_2` to a name that already exists.
- The whole set (18 packages, 34 figures, PDFs by the PyMuPDF Story renderer; no
  pdflatex on this host) builds in 64 s; 20 s without PDFs.

### Where each sample's inputs come from

| Package method | Example | Origin | Agrees with the print? |
|---|---|---|---|
| bearing_capacity_package | WE-FOOTING-1, GEC-6 App. B Example 1 | worked example | as the worked example records |
| slope_stability_package | WE-SLOPE-1, Caltrans T&S Examples 10-3/10-4 | worked example | Fellenius 0.839 (printed 0.87; V-015 explains the arc length) |
| slope_report_package | WE-SF-4, NHI-06-088 Examples 6-1/6-1a | worked example | Bishop 1.956 |
| axial_pile_package | WE-PILE-1, GEC-12 Vol 3 App. D North Abutment | worked example | as the worked example records |
| drilled_shaft_package | WE-SHAFT-1, GEC-10 App. A | worked example | side 2,094.9 kN |
| retaining_wall_package | WE-SF-3, NHI-06-089 Example 10-3 | worked example | sliding 1.76, overturning 3.44 |
| ground_improvement_package | WE-GI-1, GEC-13 Vol 1 PVD example | worked example | U = 93.1 % at 0.82 yr |
| sheet_pile_package | WE-SOE-1, Caltrans T&S Example 7-1B | worked example | as the worked example records |
| soe_package | WE-SOE-2, GEC-4 App. A Design Example 1 | worked example | pe 43.6 kPa |
| pavement_design_package | WE-PAVE-1, AASHTO 1993 Fig. 3.1 example | worked example | SN 4.98 (printed 5.0) |
| fem2d_package | WE-FEM-1, Itasca 1-D consolidation | worked example | U = 0.994, final settlement 2.6 mm |
| seismic_package | V-062, GEC-11 Vol II Example E7 Step 6 (pp. E7-5, E7-6) | validation example | KAE 0.4782; PAE 286.8 kN/m = 19.65 kips/ft, as printed |
| pile_group_package | V-060, GEC-12 Vol 3 Tables D-11, D-12, D-14 (pp. 47-49), Configuration 2 | validation example | 1,176.3 kN = 264.4 kips and -163.3 kN = -36.7 kips a pile (printed 265 and -37) |
| settlement_package | NHI-06-089 Example 8-2 (pp. 8-48 to 8-52), pinned in `settlement/tests/test_printed_examples.py` | printed example | 3.31 mm = 0.130 in at 0.1 yr, as printed |
| downdrag_package | UFC 3-220-20 App. B-5.4 (pp. 633-643), pinned in `downdrag/tests/test_alpha_default.py` | printed example | drag force 413.4 kN (Table B-28's 413); neutral plane 13.56 m below original ground = 52.5 ft below the fill surface |
| reliability_package | Duncan (2000), JGGE 126(4) 307-316, retaining wall sliding, pinned in `reliability/tests/test_validation.py` | printed example | beta_LN 2.318, pf 1.0 % (printed 2.32, about 1 %) |
| lateral_pile_package | made-up (the package's probe) | made-up | n/a |
| wave_equation_package | made-up (the package's probe) | made-up | n/a |

Made-up, and why (also in the table's `note`):
- **lateral_pile_package.** V-017 (Micropile Manual Sample Problem 2) has its cap 0.305 m
  below the ground surface; `lateral_pile_analysis` takes a stick-up above ground, not a
  buried head, and the validation test uses a custom p-y class to shift the overburden.
- **wave_equation_package.** V-003 (GEC-12 Vol 3 drivability) uses a hammer the module's
  table does not hold, and its bearing graph is not reproduced (V-003 is N/A in RESULTS.md).

### The standing test, `funhouse_agent/tests/test_sample_packages.py`

- every `*_package` method in the registry has exactly one row (fails when a package is
  added without a sample);
- every row names its origin, id and source; a made-up row says why;
- a worked-example row's inputs equal that example's dispatch call in
  `worked_examples.json`, and its source text is the example's;
- every row's call runs and returns a package with its tables built and Given Inputs
  filled (all 18 in every run: about 20 s);
- the script builds two packages end to end (files named in the index exist; a rebuild
  keeps the names);
- the full build with every PDF is `@pytest.mark.slow` (about a minute).

## Findings from building the samples

| # | Package | What | Done |
|---|---|---|---|
| 1 | drilled_shaft_package (WE-SHAFT-1) | `tables_error`: the package had NO calculation tables. The capacity-against-length table re-runs the shaft at shorter lengths; one tips in the 3 m cased sand, which has no N60 (the design shaft passes through it), and that refusal (GM-142, right for the `capacity_vs_depth` tool) took every table with it | Fixed, a few lines in `drilled_shaft/calc_steps.py`: that table alone is left out and the last printed table's note says so and why. Before: 0 tables, `tables_error`. After: 4 tables + the input echo, the note printed. Test in `drilled_shaft/tests/test_calc_tables.py` |
| 2 | soe_package, `wall_type="fhwa_anchored"` | No Figures section: `soe/calc_steps.get_figures` returns none for this wall type by design (cantilever and braced draw pressure, shear and moment) | Noted, not fixed (a new figure, not a few lines) |
| 3 | reliability_package, one variable | "Which variable matters" is a single bar at 100 % with no title | Noted, not fixed. The sample is thin because the paper's example is given as one lognormal F; a sample over the wall's own inputs would need the paper's table rebuilt |
| 4 | slope_stability_package, slope_report_package | In the cross-section figure the FOS box sits over the circle-centre label, and in the first the legend covers the toe and its "Entry" label | Noted, not fixed (slope_stability change hold, GM-140) |
| 5 | pile_group_package (V-060) | `group_efficiency` is null and the group uplift note asks for more: correct for the inputs (Tables D-11 to D-14 give no diameter or soil), so the package says what it needs | No defect |
| 6 | every package | With no `given` in the call, "Given Inputs" reads "No input was named as given" and every input is listed as assumed. Correct behaviour (GM-155), but a poor sample: the table's rows name their inputs as given | Table, not code |

No package call failed on its example. Every figure was looked at (six contact sheets of
the 34 PNGs); the numbers on the figures match the packages' results (for example the
pile group's +1,176.3 / -163.3 kN, the wall's 139 kN/m, the M-O diagram's 286.8 kN/m).

## 2. fem2d_package

- `funhouse_agent/adapters/calc_package.py`: `_PACKAGE_TOOLS["fem2d_package"]`, selector
  `analysis_type`: `slope_srm` (default, `fem2d_slope_srm`), `excavation`,
  `consolidation`, `seepage`, `staged`. The parameters are the tool's own, one source
  (`_package_parameters`), so the package refuses what the tool refuses.
- `funhouse_agent/adapters/fem2d_adapter.py`: those five handlers are split as the other
  adapters' are. `run_fem2d_<x>(params, method=)` returns the result object and what the
  package prints of the model (the layers as given, the water table in words);
  `_run_analyze_<x>` turns it into the tool's dict as before. The tools' outputs are
  unchanged (fem2d's and the agent's fem2d tests pass unchanged, bar one test that read
  a hint out of the old function's source).
- The response carries the tool's own single-number results under its own keys, with its
  notes and warnings.
- `FOS_required` is the package's own input (the tool reports a FOS and checks nothing).
- One generic change: a package's own input can now apply to one selector value
  (`_run_tool` hands the selector to `_own_defaults_used`).
- Probes: `calc_package|fem2d_package` and `#consolidation`, `#excavation`, `#seepage`,
  `#staged` (the tool probes' inputs). Their `LABEL_ONLY` / `NOT_REACHED` entries are the
  tools', copied by a loop in `method_probes.py`, one source.
- Parity rows in `test_calc_package_tool_parity.py` (against the tool and against fem2d
  called directly), `funhouse_agent/tests/test_fem2d_package_gm328.py`, the manual's two
  rows (hand-edited from the generator's own functions; the cover's method count is left
  for the release rebuild), `calc_package/DESIGN.md`, `fem2d/DESIGN.md`.
- Not packaged: `fem2d_footing_capacity`, `fem2d_gravity`, `fem2d_foundation`,
  `fem2d_local_fos`, the mesh study and `fem2d_hs_parameters`.

Before and after, made-up slope (10 m high, 1:1, c 20 kPa, phi 20, 12 x 6 mesh):
before, `call_agent("calc_package", "fem2d_package", ...)` was an unknown method; after,
a 666 KB package, FOS 1.481 (the `fem2d_slope_srm` tool's own number),
`FOS_required` 1.5 in `defaults_used`, `meets_FOS_required` false.

## 3. The calc helper and the worked examples

Added to the helper's prompt after `_CALC_FRAMING`, before the budget line, only where
`worked_examples` is in the helper's scope:

> WORKED EXAMPLES: before a calc package or report that is not trivial, find the nearest
> worked example (call_agent('worked_examples', 'find_worked_examples', {topic}), then
> get_worked_example): a validated calculation from a real published design report.
> Follow its structure and its `report_notes` (what a professional calc report for that
> problem presents); do NOT copy its numbers: use the delegation's inputs.

- The main prompt's `view_worked_example_source` is left out: the helper does not hold
  that tool.
- No other prompt text changed. Pinned in
  `funhouse_agent/deep/tests/test_deep_calc_figures_offline.py`; the two tests that
  rebuild the helper's whole prompt gained the line.
- Fingerprints re-recorded in a commit of its own: one hash moved (the geotech page's
  calc helper prompt). The fem2d_package description moved no fingerprint.
- Live check: `module_work/LIVE_TEST_QUEUE.md`, check 14.
- `module_work/harness_theory/geotech_agent.md` §3.4's calc row says so.

## Calls made

1. **`FOS_required` defaults to 1.5.** `fem2d/calc_steps.py` already checked the SRM FOS
   against 1.5 when its caller gave nothing. No held reference prints 1.5 as a
   requirement for this package, so it is stated as this package's choice: in the
   parameter's description, in `defaults_used`, and echoed in the response. The other
   form: no adequacy check unless a value is given. For the owner to revisit.
2. **Which fem2d tools the package covers:** the five whose result types
   `fem2d/calc_steps.py` renders with sections of their own. Footing capacity returns a
   different result and is left out.
3. **The drilled-shaft length table is dropped, not shortened,** when a shorter shaft
   cannot be computed, with the reason in a printed note. The other form: start the table
   below the layer that cannot carry a base.
4. **`given` names every input of a sample.** A published example states its inputs; the
   made-up rows do the same.
5. **Three rows are "printed examples"**, pinned in a module's own tests rather than in
   `worked_examples.json` or `validation_examples/` (settlement, downdrag, reliability).

## Tests run (the venv of COMMON_MODULES; pytest's own exit code)

- `funhouse_agent/tests fem2d/tests/test_calc_tables.py fem2d/tests/test_calc_viz.py calc_package webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py` (before the last fixes): 4,578 passed, 35 skipped, 4 failed. Two were this branch's and are fixed (the method count 20 to 21; the `FOS_required` probe entry); two fail on master too (below).
- After the fixes, `test_calc_package_adapter.py test_defaults_used.py test_fem2d_package_gm328.py test_calc_package_tool_parity.py test_render_tables.py drilled_shaft fem2d/tests/test_calc_tables.py fem2d/tests/test_calc_viz.py test_public_text.py test_page_fingerprints.py test_docs_currency.py`: 1,398 passed, 23 skipped, 1 failed (`plot_data`, on master too).
- `fem2d/` and `funhouse_agent/tests -k fem2d` after the adapter split: 715 passed, 1 skipped, 1 failed (the test that read the old function's source; repointed, then 101 passed on the agent side).
- `funhouse_agent/tests/test_sample_packages.py`: 24 passed (81 s, the slow full build included).
- `funhouse_agent/deep/tests webapp/tests/test_public_text.py` after the prompt change: 958 passed, 4 skipped. `webapp/tests/test_page_fingerprints.py` after the re-record: 13 passed.
- Final run (`validation_examples drilled_shaft calc_package test_sample_packages.py test_worked_examples.py test_public_text.py test_page_fingerprints.py test_docs_currency.py test_backlog.py`): 904 passed, 5 skipped, 1 xfailed, 0 failed.
- `python module_work/doc10/check_manual.py`: OK.

Failing on master 2feea3d as well, not this branch's (checked with the changes stashed):
- `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (`logx`, `logy` change nothing in the result);
- `funhouse_agent/tests/test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`.

## Release note

A new tool, `calc_package.fem2d_package`: a calc package for a fem2d strength-reduction
slope analysis, staged excavation, consolidation, seepage or staged construction run
(the fem2d tools' own inputs and numbers; `FOS_required`, 1.5 unless given, is the
package's check). A drilled shaft calc package whose shaft passes through a sand layer
with no N60 now keeps its calculation tables (it had none), without the
capacity-against-length table, and says so. The calc helper is told to look up the
nearest published worked example before a calc package that is not trivial. No
calculated value changes.

## Noted, not fixed

- Findings 2, 3 and 4 above (no figure for the FHWA anchored wall; the one-bar
  reliability figure; overlapping labels on the slope cross-section).
- The two tests that fail on master (above).
- The manual's cover count of agent-callable methods and its prose are not rebuilt; the
  release's manual rebuild picks `fem2d_package` up.
- The script's PDFs are whatever the host renders (here the PyMuPDF Story renderer; with
  pdflatex they would differ).
- The old committed samples (`docs/examples/*.html`, `sample_pdfs/`,
  `sample_calc_package.pdf`, `generate_all_pdfs.py`) are untouched: OWN-144.
