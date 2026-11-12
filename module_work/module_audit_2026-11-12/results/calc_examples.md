# Calculation examples: state and what the calc agent is given

Read-only investigation, 2026-11-12, C:\dev\GeotechStaffEngineer (master). Dates are `git log -1 --date=short`.

## Q1 answer

1. The folder the owner means is almost certainly `docs/examples/`. It holds two calc package demos (slope LE, FEM strength-reduction) plus their interactive pages, all committed 2026-06-11. It has not been touched for a calc package since. The only later commits there (2026-11-09) are the `deployment_instructions/` folder, which is not a calc example.
2. The older top-level `sample_calc_package.pdf` (bearing capacity, 2026-02-17) and `sample_pdfs/` (13 modules, 2026-02-17) are also stale. They predate the Given/Assumed/Choices sections, calculation tables (GM-39) and the figure helper.
3. The examples that are kept current are not calc packages: `funhouse_agent/worked_examples.json` (28 published textbook/manual examples, last changed 2026-11-05) and `validation_examples/` (36 `test_published_*.py` files, INVENTORY/RESULTS, last changed 2026-11-05). They are inputs plus expected answers, not finished write-ups.
4. 16 modules have a canned `*_package` method; `fem2d` is in the renderer but has no `*_package` method. Of those 16, only slope (reinforced LE) has an example of today-ish output in `docs/examples/`; 13 have a Feb 2026 PDF; pavement_design, soe and reliability have no committed example at all (table in section C).
5. Neither committed set shows today's layout (given vs assumed inputs, choices, tables, drawn figures), so no committed example is in step with today's code.

## Q2 answer

1. A user's calc package request goes to the `calc` helper (`funhouse_agent/deep/agent.py:559` `build_calc_subagent`). The main agent is told to delegate to it (`agent.py:497` `_CALC_DELEGATION_NUDGE`). The helper calls a module's canned `*_package` method, or writes its own HTML and renders it with `calc_package.html_to_pdf`.
2. A canned `*_package` fills a fixed template from the module's `calc_steps.py` (code-written content: inputs, steps, tables, figures). Only the bespoke `html_to_pdf` route is model-written content.
3. No finished SAMPLE CALC PACKAGE is handed to, or referenced in the prompt of, any agent or helper. I found none.
4. The nearest thing is `worked_examples`: entries hold a problem, the dispatch calls, the published answer and prose `report_notes` ("what a professional calc report presents"). The calc helper can reach the module (it is in ANALYSIS_MODULES, `adapters/__init__.py:363`, not in REFERENCE_MODULES) but its own prompt never mentions it. Only the main agent's prompt (`prompt.py:177`) says to consult it.
5. The figure helper does get 9 worked figure examples (6 `plot_data` calls, 3 plot_code scripts) with the made-up-number rule. These are figures, not calc packages, and the calc helper does not get that tool.

## Q1 evidence

### A. Every place that holds examples

| Place | What it holds | Last changed | In step with today's code? |
|---|---|---|---|
| `docs/examples/slope_le_calc_package.html` (906 KB), `slope_le_interactive.html` | Slope LE package: reinforcement, method comparison, slice table, search, reliability, figures. Sections 1-11 (no Given/Assumed/Choices/Calculation Tables headings; grep of the file found none of those) | 2026-06-11 (e0d49de "calc-viz Phase 6: committed demo artifacts") | No |
| `docs/examples/fem_srm_calc_package.html` (1.4 MB), `fem_srm_interactive.html` (6 MB) | FEM SRM package, sections 1-6 | 2026-06-11 | No. fem2d has no `*_package` adapter method today |
| `docs/examples/model_setup_walkthrough.md` + 2 PNGs | Model-setup staged flow transcript | 2026-06-12 | Unchecked; not a calc package |
| `docs/examples/deployment_instructions/` (README, all, main, calc, references .md) | Made-up per-deployment instruction text (REF-64 B); `calc.md` is one sentence ("put the inputs table before the method section...") | 2026-11-09 | Current. Not a calc example |
| `sample_calc_package.pdf` (root) | One bearing-capacity package (`review_eval/tasks.py:801` cites its page 4) | 2026-02-17 (7143343) | No |
| `sample_pdfs/*.pdf`, built by `generate_all_pdfs.py` | 13 PDFs: axial_pile, bearing_capacity, downdrag, drilled_shaft, ground_improvement, lateral_pile, pile_group, retaining_walls, seismic_geotech, settlement, sheet_pile, slope_stability, wave_equation | 2026-02-17 (112da33) | No (script says "all 13 analysis modules"; 16 now) |
| `funhouse_agent/worked_examples.json` + `worked_examples.py`, adapter `adapters/worked_examples_adapter.py` | 28 entries WE-PILE-1, WE-SHAFT-1, WE-FOOTING-1, WE-MSE-1/2, WE-SOE-1/2, WE-GI-1/2, WE-SLOPE-1..5, WE-FEM-1, WE-PAVE-1..3, WE-SF-1..10. Each is a published example: problem, dispatch calls, published vs computed result, report_notes, source pages | json 2026-11-05 (97656cc); .py 2026-07-19; first added 2026-07-18 | Yes: tests run every entry's calls; GM-114 etc. keep it in step |
| `validation_examples/` | INVENTORY.md, RESULTS.md, 36 `test_published_v0xx_*.py`, `extracts/` | 2026-11-05 (97656cc) | Yes (run in CI) |
| `docs/user_manual/worked_examples.json` (18 captured examples, build input) and `manual.html` | Captured dispatch output for the manual; `gen_worked_examples.py` regenerates | manual dir 2026-11-10 | Regenerated per release (not opened in detail) |
| `docs/gallery/` (25 files, `build_gallery.py`, `figures/`) | Figure gallery | 2026-11-05 (98922c1) | Figures only; not opened in detail |
| `packages/aec-agent-core/.../charts/figure_examples.py` | 9 figure examples (see Q2) | 2026-11-09 | Yes |
| Module tests (e.g. `settlement/tests/test_printed_examples.py`, `drilled_shaft/tests/test_rock_socket_example.py`, `calc_package/tests/*`) | Code checks, not documents | various | n/a |
| `funhouse_agent/eval_samples/`, `report_ingest/*.EXAMPLE.*` | Input files for ingest/eval, not calc examples | not checked | n/a |

Stray, not tracked on GitHub: `downdrag_calc_20261015_214735.html` (+ CSVs), `retaining_walls_calc_20261018_172709.html`, and 24 `conditional_sim_*.csv` sit in the working-tree root. `git ls-files` lists none of them and `git check-ignore` does not ignore them. These are local run outputs; the downdrag and retaining-wall HTMLs are the only recent package outputs on disk, but they are not committed.

Most likely "the github folder": `docs/examples/`. Confirmed: its calc package content is exactly two packages, slope LE and FEM SRM, both 2026-06-11. `sample_pdfs/` (Feb) is the second candidate.

### B. Reasoning on "is it kept up"
`docs/CALC_VIZ_PLAN.md:70` records the demos as a one-off Phase 6 deliverable. `module_work/FUTURE_IDEAS.md:96` (2026-09-22 slimming note, item 1) already lists "Example-product inventory review: what docs/examples, docs/gallery, sample_pdfs, the calc_package templates and worked_examples.json are FOR (the idea was reference material for the calc-package and visualisation agents), which are outdated". It is PARKED behind Tiny Apps, via LT-03 (`BACKLOG.md:505`, Parked). No one has done that review.

### C. Per module with a calc package

"*_package method" = a key in `METHOD_REGISTRY` in `funhouse_agent/adapters/calc_package.py` (lines 3111-3127). Tables = a `get_tables` in `calc_steps.py` or `results.py` (BACKLOG_DONE GM-39 at line 692: "retaining walls and pavement hold only scalars"). Figures = `get_figures` in `calc_steps.py` (all 17 have one).

| Module | `*_package` method | Calc tables | Figures | Committed example output |
|---|---|---|---|---|
| bearing_capacity | bearing_capacity_package | yes (results.py) | yes | `sample_calc_package.pdf`, `sample_pdfs/bearing_capacity.pdf` (Feb) |
| lateral_pile | lateral_pile_package | yes | yes | `sample_pdfs/lateral_pile.pdf` (Feb) |
| slope_stability | slope_stability_package, slope_report_package | yes | yes | `docs/examples/slope_le_*` (Jun 11), `sample_pdfs/slope_stability.pdf` (Feb) |
| settlement | settlement_package | yes | yes | `sample_pdfs/settlement.pdf` (Feb) |
| axial_pile | axial_pile_package | yes | yes | `sample_pdfs/axial_pile.pdf` (Feb) |
| drilled_shaft | drilled_shaft_package | yes | yes | `sample_pdfs/drilled_shaft.pdf` (Feb) |
| downdrag | downdrag_package | yes | yes | `sample_pdfs/downdrag.pdf` (Feb) |
| seismic_geotech (incl. liquefaction runs) | seismic_package | yes (results.py) | yes | `sample_pdfs/seismic_geotech.pdf` (Feb) |
| retaining_walls | retaining_wall_package | no (scalars only) | yes | `sample_pdfs/retaining_walls.pdf` (Feb) |
| ground_improvement | ground_improvement_package | yes | yes | `sample_pdfs/ground_improvement.pdf` (Feb) |
| wave_equation | wave_equation_package | yes (bearing_graph.py) | yes | `sample_pdfs/wave_equation.pdf` (Feb) |
| pile_group | pile_group_package | yes (rigid_cap.py) | yes | `sample_pdfs/pile_group.pdf` (Feb) |
| sheet_pile | sheet_pile_package | yes | yes | `sample_pdfs/sheet_pile.pdf` (Feb) |
| soe | soe_package | yes | yes | none |
| reliability | reliability_package | yes | yes | none |
| pavement_design | pavement_design_package | no (scalars only, per GM-39) | yes | none |
| fem2d | none (in the renderer's `_IMPORT_MAP`, `calc_package/__init__.py:133`; reachable by `render_figures`/`render_tables`) | yes | yes | `docs/examples/fem_srm_calc_package.html` (Jun 11) |

Also: every module's `render_figures` / `render_tables` (GM-67) and the bespoke `html_to_pdf` route work without a canned package.

Analysis modules with no calc package at all (no `calc_steps.py`, not in `_IMPORT_MAP`): liquefaction (covered only through `seismic_package`), dxf_export, dxf_import, pdf_import, drawing_ir, profile_figure, worked_examples, pystra, salib, opensees, pystrata, liquepy, seismic_signals, gstools, section_props, concrete_props, pynite, subsurface (names from the `adapters/__init__.py` registry). I did not check whether any of these should have one.

## Q2 evidence

### A. How a package is made
- Primary agent: `funhouse_agent/deep/agent.py:497` nudge: "For any tool-heavy calculation — running an analysis method, a parameter sweep, or building a calc package — DELEGATE it to `calc` ... The calc agent has NO memory of this conversation".
- Helper: `build_calc_subagent` (`agent.py:559-672`): tools are core tools (`call_agent` over ANALYSIS_MODULES plus the reference modules' functions), `save_file`, `list_files`, `read_pdf_text`, `read_text_file`, `write_xlsx`, `analyze_image`, `analyze_pdf_page`, a calculate tool, `plot_code` when on. Prompt = `_CALC_PREAMBLE + _CALC_FRAMING` plus budget, look-fix and judgment-call additions. Model-call budget `DEFAULT_CALC_MAX_MODEL_CALLS`.
- Quoted from the helper's prompt (`agent.py:~398`): "If no canned *_package method fits the analysis, compose the report as self-contained HTML yourself and render it with calc_package method html_to_pdf (figures as the html_img_tag of a saved PNG ...)". And: "NEVER WRITE PLACEHOLDERS: every number, table row and label in a deliverable comes from a tool result or from the delegation text."
- Canned route: `adapters/calc_package.py` `_generate_*_package` functions run the module tool, then `_build_response` fills the fixed template from `calc_steps.py`: given/assumed inputs, choices, steps, tables, figures. Content is code-written, not model-written.
- Bespoke route: model-written HTML, rendered by `html_to_pdf`. The main prompt (`prompt.py:50-60`) says a bespoke report gets figures from `render_figures`, its calculation from `render_tables`, and plots from `plot_data`.
- Figures: a separate `figure` helper (`aec_agent_core/deep/figure_helper.py`) exists; the calc helper draws with `plot_data`/`plot_code` itself. The primary note says each figure brief "carries the nearest worked example" (`agent.py:1330`).
- Not opened: `review_agent.py` and `notebook.py` (grep for "example" found only comments at `review_agent.py:112,144`); neither is the calc writer.

### B. What examples the writer is shown or can fetch
1. **Calc helper's prompt**: no mention of `worked_examples`, `find_worked_examples`, any sample or example package (grep of `agent.py` for "worked_examples" finds nothing; the only "example" hit in the helper area is a comment at `agent.py:1332` about figures). It can still call the `worked_examples` module through `call_agent` because the module is an analysis module.
2. **Main agent's prompt** (`funhouse_agent/deep/prompt.py:177-188`), quoted: "**Consult a worked example before a nontrivial design or calc report.** The `worked_examples` module holds validated calculations from real published design reports (FHWA GEC pile/shaft/MSE/footing examples, Caltrans shoring, AASHTO/UFC pavements, dam-drawdown and probabilistic slope benchmarks). Call `find_worked_examples` with your topic; a hit gives you the proven method sequence + parameters, the published answer, and `report_notes` describing what a professional calc report for that problem presents. When the entry lists `source_pdf_pages`, `view_worked_example_source` renders the PRINTED page of the original design example (figures and all) so you can follow the real thing. Follow the exemplar's structure; do NOT copy its numbers — use the user's inputs." The helper does not receive this text and the delegation nudge does not pass it on (I did not find any code that does).
3. **What the 28 entries are**: published textbook/manual examples (GEC-4/6/10/11/12/13, Caltrans T&S, NHI-06-088/089, AASHTO 1993, UFC 3-250-01, Slide2, Itasca FLAC, Duncan 2000). Fields: id, title, domain, keywords, source, problem, dispatch_calls, published_result, computed_result, report_notes, provenance, source_doc, source_pdf_pages. Modules covered: axial pile, drilled shaft, bearing/settlement, MSE and retaining walls, SOE and sheet pile, ground improvement, slope, FEM, pavement. None carries a calc package (script check: 0 entries contain a `*_package` call or "calc_package"). `report_notes` is a short prose paragraph per entry, for example WE-MSE-1: "A good report tabulates the unfactored loads ... then shows each AASHTO load-factor combination ... Report CDRs ..."
4. **Figure helper's examples** (`aec_agent_core/charts/figure_examples.py`, 994 lines, 2026-11-09): `EXAMPLES` = plan_borings, spt_profile, cases, time_settlement, load_settlement, exceedance (plot_data calls); `CODE_EXAMPLES` = wall_pressures, settlement_stress_profile, pile_design_profile (matplotlib scripts for the locked plot_code box). Header: "Every number here is invented." Fetched through the `figure_examples` tool (`figure_helper.py:233`) on the figure helper only. The calc helper's tool list has no `figure_examples`.
5. **Deployment instructions** (REF-64 B): `GEOTECH_INSTRUCTIONS` folder, `calc.md` appended to the calc helper, 4,000-character cap per role. The repo's example `calc.md` is one sentence. This is a way to hand the calc helper layout rules, not a sample package; no real deployment folder was examined.
6. Plain answer: no finished calculation write-up showing the expected layout, level of detail, assumptions and checks is handed to or referenced by any agent or helper. The helper learns layout from the fixed template code (canned) or from the prose `report_notes` only if the primary looked them up and passed them in its brief (bespoke).

## History

- `module_work/FUTURE_IDEAS.md:7` (2026-07-20 update): the "sample-calc-as-defect-detector doctrine". Public worked examples were used to find four real defects (two unconservative up to 10x). Here the sample calcs are a test of the code, not material handed to the agent.
- `FUTURE_IDEAS.md:121-142`, item 1: worked-examples corpus v1 BUILT. "Phase 2 — the owner's own reports as exemplars. The owner asked for 'real ones in real reports': let the firm's actual calc packages join the corpus." Sketch: a local-only `GEOTECH_EXAMPLES_DIR` of PDFs, harvested with `pdf_import` into stubs, a human confirms each before it is trusted, firm reports never packaged to PyPI. So yes, the owner asked, in July 2026.
- `worked_examples.py` docstring (lines 1-26): the stated purpose is "when the calc agent works a nontrivial problem or builds a report, a matching exemplar shows ... what a professional calc report presents (`report_notes`)", and "a second, user-supplied corpus harvested from the owner's own project reports" is named as a future direction.
- `BACKLOG.md:107` OWN-41 [Parked] [owner]: "The owner's own reports as worked-example exemplars (Phase 2, local-only `GEOTECH_EXAMPLES_DIR`). Re-tiered 2026-11-04 (Later to Parked): No code yet; report ingest's calc reader and library now take in your reports, so revisit after the calc-module train." So the ask was recorded, parked, and nothing was built.
- `FUTURE_IDEAS.md:96`, via LT-03 (`BACKLOG.md:505`, Parked): the example-product inventory review (docs/examples, gallery, sample_pdfs, templates, worked_examples.json) was proposed 2026-09-22 and parked behind Tiny Apps.
- `docs/CALC_VIZ_PLAN.md:70`: the June demos were a one-time phase.
- GM-39 (BACKLOG_DONE:692): calculation tables built; "retaining walls and pavement hold only scalars" left. GM-41 (BACKLOG_DONE:572): SOE and reliability packages built 2026-10-19. GM-66 (BACKLOG:256, Parked): many CSV download cards. GM-67 (BACKLOG_DONE:374): `render_tables` for bespoke reports. GM-162/GM-111 (Parked) concern calc helper deliverable handling. None of these concerns example packages.
- `module_work/calc_qc/FINDINGS.md` (2026-06-06): a correctness QC of 13 modules and `calc_steps`; "tie out against a GEC-12 worked example" appears once as advice (line 174). No sample-package discussion.
- I found no BACKLOG item, owner answer or field-feedback file asking to hand the calc helper a sample calc package. I did not read the field-feedback folders' FINDINGS files line by line; I only grepped BACKLOG, BACKLOG_DONE, HANDOFF, FUTURE_IDEAS and calc_qc/FINDINGS.

## Not checked

- Contents of `docs/user_manual/manual.html` and `docs/gallery/` beyond file counts and dates.
- `module_work/field_feedback/*/FINDINGS.md` and REVIEW files, and `HANDOFF.md` beyond the grep for phrases ("sample calc", "example calc", "calc subagent", "calc helper", "calc QC", "office calcs"; the last four phrases returned only calc-helper bug items, no sample-package discussion). A narrower phrase I did not try may exist there.
- Whether today's `generate_all_pdfs.py` still runs, and whether any README or release page links `docs/examples/` to GitHub readers.
- Whether the GitHub remote differs from local master (I read the local checkout, not the remote).
- Whether any cloud or Foundry deployment folder (`GEOTECH_INSTRUCTIONS`) already holds sample-calc text; it would be outside the repo.
- Whether `review_agent.py` or `notebook.py` build packages (grep says no).
- The assertion that the helper receives no worked-example text rests on grep of `agent.py`, `prompt.py`, `tools.py`, `calc_figures.py`; a wrapper in another file (for example the webapp host) could add text through `subagent_extra_system_prompt`. I did not open the webapp host for this.
