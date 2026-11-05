# INF-31 check: the app split's steps 6, 7 and 8

Branch `claude/inf-31-split-steps-6-8`, from master 333f0e5. One commit per
step. All three are meant to change no behaviour. The plan is
`module_work/APP_SPLIT_PLAN.md`, sections 2.7, 2.8 and 4.

## The guard used for every step

`webapp/tests/test_page_fingerprints.py` pins what both pages' agents are
handed. It passed unchanged after each step, and
`webapp/tests/page_fingerprints.json` was never re-recorded (the file is
not in the diff).

One test-helper change was needed. At step 8 the review page compiles
through the shared assembly, so `webapp/tests/page_builds.py` now patches
`funhouse_agent.deep.assembly.create_deep_agent` as well as the copy in
`funhouse_agent.deep.agent`. Without that, the review page's build would
not be recorded. At step 7 page_builds also builds through the page's
profile, as `webapp/app.py` does. The recorded data is unchanged.

## Step 6: the chart stack moves to `funhouse_agent/charts/` (commit 3c3de09)

**What moved where**

- New package `funhouse_agent/charts/`:
  - `data_plot.py` (was `profile_figure/data_plot.py`);
  - `figure_examples.py`, the nine worked examples (was
    `profile_figure/figure_examples.py`; OWN-77 (b));
  - `layout_ink.py` and `plan_furniture.py` (were in `geotech_common/`);
  - `plotting.py`: `get_pyplot` (was in `geotech_common.plotting`) and
    `figure_to_png` (was in `profile_figure.plotting`);
  - `plot_tool.py`: the `plot_data` method itself. This is the half of the
    `profile_figure` adapter that both pages use: its parameters, its
    checks, how it draws, and `call_plot_data`.
- `funhouse_agent/method_call.py` (new) holds the generic half of
  `dispatch.call_agent`, as `run_method`:
  - parameter-name checks, attachments and working-folder inputs;
  - output paths moved into the working folder, and the output note;
  - the plain-words error for a signature mismatch;
  - `defaults_used`, and paths shown relative to the working folder.

  `dispatch.call_agent` now calls `run_method` too. `plot_data` therefore
  gives the same result on either path, without needing GSE's dispatch.
- `apply_aliases`, `require_params`, `reject_unknown_params`,
  `refused_reason`, `accepted_as_unused`, `declare_aliases` and
  `declare_when_absent` moved to `funhouse_agent/arg_checks.py`.
  `write_plotly_sidecar` moved to `funhouse_agent/_fileio.py`.
- `reference_chart` and the plotting box's chart lines stay GSE's. They
  reach the chart stack through a new registry,
  `funhouse_agent.charts.register_reference_charts`:
  - the `profile_figure` adapter registers the reference library's chart
    lines when it is imported;
  - `deep/tools.register_reference_charts()` is called by the geotech
    builds before any tool is described;
  - the box loads the registered `chart_lines` file, whose path is passed
    in the payload, and no longer looks for `geotech_references`;
  - the box loads `layout_ink` and `plan_furniture` from
    `funhouse_agent/charts/`.

**Old paths kept as aliases (the same objects)**

- Stub modules that hand back the moved module:
  - `geotech_common.layout_ink`
  - `geotech_common.plan_furniture`
  - `profile_figure.data_plot`
  - `profile_figure.figure_examples`
- Re-imports:
  - `geotech_common.plotting.get_pyplot`
  - `profile_figure.plotting.figure_to_png`
  - every moved name in `funhouse_agent.adapters` and `funhouse_agent.dispatch`
  - the `profile_figure` adapter's `_KEY_HINTS`, `_image_name` and similar
- The adapter keeps `plot_data` under `profile_figure` for `call_agent`. It
  draws through `plot_tool.run_plot_data`, with its own `reference_chart`
  overlay.

**Exit**

- Fingerprints identical.
- Coupling 4 is gone from the boundary list: `vision_kit -> dispatch`,
  `vision_kit -> profile_figure_adapter`, `vision_kit -> profile_figure`,
  `plot_box.child -> geotech_common` and `-> geotech_common.layout_ink`.
- The box's look for `geotech_references.chart_lines` is gone too.
- New static test `test_the_chart_stack_imports_nothing_of_gse`. It checks
  `funhouse_agent/charts`, `plot_box`, `method_call.py` and
  `arg_checks.py`.

## Step 7: the shell asks the page, not geotech (commit f1acca7)

**What moved where**

- `webapp/page_profile.py` (new) holds `AppProfile`, moved out of
  `profiles.py`, with the hooks the plan lists:
  - `disclaimer`, `placeholder`, `sidebar_extras`, `behavior_kwargs`;
  - `agent_builder`, plus `specialist_builder` for the Agent picker;
  - `on_build`, `turn_checks`, `diagnostics_checks`, `version_dists`;
  - `finish_build_kwargs`, where the review page builds its prompt.

  A hook is a callable, or `"module:name"` imported the first time it is
  used. So neither the shell nor the review page loads
  `webapp/geotech_page.py`.
- `webapp/geotech_page.py` (new) holds the geotech branches of the shell:
  - the disclaimer;
  - `build_page_agent`, with the report library (was
    `core._add_report_library`);
  - `build_reviewer_agent` and `_REVIEWER_BUILDERS`;
  - `register_reference_fetcher`;
  - `turn_checks`, the judgment-call check (was
    `core._turn_judgment_check`);
  - `agent_picker` (was in `app.py`);
  - the reference-PDF diagnostics: `reference_pdf_names`,
    `reference_docs_check` and the two messages (were in
    `diagnostics.py`).
- `webapp/review_page.py` (new) holds the review page's own parts:
  - `DOCUMENT_REVIEW`, the orientation texts and helpers;
  - its builder and its notice.
  - The current disclaimer text is kept (OWN-78 is not decided).
  - Until step 13 the page keeps three things it shares with the geotech
    page, each reached on use: the judgment-call check (it only records on
    this page), the reference-PDF diagnostics line (named by string), and
    the notice.
- `core.build_agent(..., profile=)` asks the page for its builder:
  - it adds what every page gets, runs `on_build`, and hands the rest to
    the page's `agent_builder`;
  - with no `profile`, it picks the review page for a `review_page` build,
    otherwise the current page;
  - it stores the page's turn check on the agent, and `stream_turn` reads
    it there (otherwise from the current page).
- `core.disclaimer_text(profile)` and
  `diagnostics.run_diagnostics(page_checks=)` ask the page.
- `app.py` takes from the profile: the build, the Agent picker
  (`sidebar_extras`), the placeholder, the version line and the
  diagnostics.

**Old names kept as aliases**

- `core.build_reviewer_agent`, `core._REVIEWER_BUILDERS`,
  `core._register_reference_fetcher`, `core._add_report_library` and
  `core._turn_judgment_check`. These resolve to `webapp.geotech_page`
  through a module `__getattr__`.
- `diagnostics.reference_pdf_names`, `diagnostics._reference_docs_check`,
  `REFERENCE_DOCS_ENV` and the two `_UNSET_MESSAGE*` names. These resolve
  the same way.
- `profiles.AppProfile`, `profiles.DOCUMENT_REVIEW` and every orientation
  name are re-exported from `profiles`.

**Exit**

- Fingerprints identical.
- New static test `test_the_shell_imports_nothing_of_geotech`.
  `app.py`, `core.py` and `diagnostics.py` import nothing from:
  - `funhouse_agent.dispatch`, `reviewers`, `_disclaimer` (or
    `funhouse_agent.disclaimer`);
  - `reference_docs`, `worked_examples`, `geotech_references`;
  - `webapp.reference_fetch`, `webapp.geotech_page`.
- `webapp.core -> webapp.reference_fetch` is gone from the boundary list.
- Coupling 1's importer moved from `webapp.core` to `webapp.review_page`.
  It was cut at step 8.
- `webapp.geotech_page` was added to `page_builds.GSE_MODULES`, and
  `webapp.review_page` to `REVIEW_MODULES`.

**Tests changed**

Five tests now patch where the code lives:

- `test_app_smoke`, `test_app_profiles`, `test_past_conversations_pages`
  and `test_wave2c_fixes` patch `webapp.geotech_page.build_reviewer_agent`
  (`app.py` calls the profile's hook, not `core`).
- `test_diagnostics` patches and calls `geotech_page.reference_docs_check`.

## Step 8: the review page builds itself (commit 1b79e4b)

**What moved where**

- `funhouse_agent/review_build.py` (new): `build_review_page_agent`.
  - It covers the `minimal` and `lean` builds behind their switches, and
    otherwise today's build.
  - The page's pieces moved with it from `deep/agent.py`: the review
    descriptions and rewrites, the reading helper, the budget,
    `REVIEW_PAGE_STEPS_PER_CALL`, `REVIEW_READER_DESCRIPTION`,
    `GEOTECH_ONLY_TOOLS` and `_BUDGET_OFF`.
- `funhouse_agent/deep/assembly.py` (new): what both builds share.
  - Memory backend and summarizer; `HELPER_PARTIAL_RULE`;
    `_TodoMiddleware`; `_GENERAL_PURPOSE_SPEC`; `_merge_tools`,
    `_save_tool`, `_join_prompts`.
  - `guard_helpers`: the guards round each helper, the general-purpose
    helper, each helper's own model, and the coverage recorders.
  - `create_kwargs_for`: memory and the main middleware chain. Each page
    passes its own middleware: HideTools on the geotech page;
    HideScratchFilesystem plus the budget on the review page.
  - `compile_agent`: the compile, plus the planning-middleware rebuild.
  - `record_build`: the attributes the host reads.
  - It imports nothing of GSE. A build with modules hands in
    `UnknownToolHint` and the calc helper's `FigureLooks` note type.
- `build_deep_agent` loses its `review_page` branches and its imports of
  `review_agent`, `minimal_agent` and `review_prompt`.
  - `review_page=True` hands off to the new build (`_review_page_build`)
    until step 15.
  - Any geotechnical part an old caller asked for is made by the GSE
    builder and placed where it was before: modules, reference or calc
    helpers, the setup helper, the domain prompt when there is no
    `system_prompt`.
- `webapp/review_page.build_page_agent` calls `build_review_page_agent`,
  and so does `review_eval.runner.build_page_agent` through the profile.
- `review_agent._summarizer` reads the assembly.

**Aliases**

- `funhouse_agent.deep.agent` re-exports every assembly name as the same
  object.
- The review build's names resolve lazily from `deep.agent`.

**The shared assembly's size**

`funhouse_agent/deep/assembly.py` is 583 lines:

| Kind | Lines |
|---|---|
| Not blank and not a comment | 414 |
| Docstrings | 133 |
| Code without docstrings or comments | 281 |

`review_build.py` is 392 lines. `deep/agent.py` went from 2,064 to 1,608
lines.

**Exit**

- Fingerprints identical.
- Boundary list for all three review builds, recorded on this branch.
  All three build and run, with GSE recorded or blocked. Only soft
  couplings are left:

```
funhouse_agent -> funhouse_agent._disclaimer                               (soft, step 13)
funhouse_agent.review_build -> funhouse_agent.adapters.profile_figure_adapter  (soft hook: the chart lines, step 13)
funhouse_agent.vision_tools -> funhouse_agent.reference_docs               (soft, step 11)
funhouse_agent.vision_tools -> geotech_references                          (soft, step 11)
funhouse_agent.vision_tools -> geotech_references._figures_db              (soft, step 11)
gse_into_review: []
```

- With every GSE module blocked (4 refused), all three review builds build
  and run. `test_review_page_works_with_gse_modules_blocked` lifted its
  strict-xfail mark by itself and now passes as a plain test.
- `KNOWN_REVERSE_COUPLINGS` keeps only `deep.prompt -> review_prompt`, the
  old-path re-export that goes at step 15.
- New static test `test_the_review_build_and_the_assembly_import_nothing_of_gse`.
  The only GSE import it allows is the listed chart-lines hook.

**Tests changed**

- `test_orientation_overview` and `test_report_library_wiring` patch
  `review_build.build_review_page_agent`, or the assembly's
  `create_deep_agent`, for the review case.
- `test_result_map` reads the guards out of the assembly's source.

## The boundary test's labels (relabelled as INF-31 asks)

- The disclaimer, labelled "step 10" before, now reads "step 13".
- The chart lines, labelled "step 12" before, now read "step 13". They
  are the review build's soft hook, and the box's own look for them is gone.
- The docstring now describes the exits after steps 6-8.

## Tests

All test runs used `/tmp/v`: Python 3.12, `pip install -e .` with
planlens and geotech-references from PyPI, and `-n 4`.

| Step | Suites | Result |
|---|---|---|
| 6 | 34 test files that import a moved name, plus `profile_figure`, `geotech_common` and `funhouse_agent/deep/tests` | 3,341 passed, 3 failed: 2 also fail on master, 1 was a test fixed in the step |
| 6 | `webapp/tests`, `review_eval/tests`, `pile_group`, `gstools_agent`, `subsurface_characterization`, `calc_package` | 2,031 passed, 0 failed |
| 6 | `funhouse_agent/tests`, `foundry_test_harness`, `validation_examples` | 4,640 passed, 2 failed (both also fail on master) |
| 7 | `webapp/tests` | 1,072 passed, 0 failed |
| 7 | `review_eval/tests`, `deep/tests`, and the funhouse tests that touch webapp | 1,677 passed, 0 failed |
| 8 | `webapp/tests`, `deep/tests`, `review_eval/tests`, `funhouse_agent/tests` | 6,536 passed, 37 skipped, 2 failed (both also fail on master) |
| All | The whole app suite (testpaths from pyproject) | see below |

The two failures that also fail on master 333f0e5 (checked in a worktree
of that commit with the same venv):

- `funhouse_agent/tests/test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
- `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`.
  On this test's input, `logx` and `logy` change nothing in the result.

`webapp/tests/test_public_text.py` was run before each commit and passed.

**Whole app suite** (`python -m pytest -q -n 4 -p no:cacheprovider` on step 8, testpaths from pyproject): **18,149 passed, 2 failed, 124 skipped, 1 xfailed** in 23 min 38 s. The 2 failures are the two above, which also fail on master. The skips are dependencies not installed in this venv (pydiggs, python-ags4, ipywidgets) and the suites' own skips.

## Noted, not fixed

- `geotech_common.plotting` now imports `funhouse_agent.charts`, which
  loads `funhouse_agent`'s package door.
  - So a library user who draws a module figure (for example
    `axial_pile`'s plot) can see the one-time professional-use notice on
    stderr. Today they would see it only on importing `funhouse_agent`.
  - The notice is once per user, is suppressed by `GEOTECH_NO_DISCLAIMER`,
    and never shows under pytest.
  - Importing `geotech_common` or `calc_package` still loads nothing of
    `funhouse_agent`: their chart imports run inside functions.
- The review page's build no longer registers the SharePoint reference-PDF
  fetcher. The fetcher is the geotech page's `on_build` hook, and only
  GSE's reference readers use it, so the review page sees no difference.
  On a host where only the review page is opened, the fetcher is
  registered later, at the first geotech-page build.
- An old caller that combined `build_deep_agent(review_page=True,
  enable_report_ingest=True)` used to get the ingest tool. The page's
  build drops it, as it does the other geotech-only arguments it has
  nothing to act on. No caller in the repo does this.
- These sidebar items stay in `app.py`, gated by `profile.specialists` as
  before: the three geotech-only Behavior pickers (references, analysis
  depth, calc helper) and the reference budget. They sit inside a shared
  expander, and moving them would reorder the widgets. They need splitting
  for the copy at step 13.
- `webapp/core.py` still uses `funhouse_agent.deep.notebook`, which is GSE,
  for its stream formatters. Plan 2.5 splits that file. It is not in step
  7's exit list.
- `funhouse_agent.review_eval.documents` still registers the reference
  fetcher through the `core._register_reference_fetcher` alias. Plan 2.7 B
  makes that a hook at step 11.
- The plan's docs (`APP_SPLIT_PLAN.md`, `funhouse_agent/DESIGN.md`, the
  module DESIGN notes) still name the old paths in places. They are not
  updated here, because BACKLOG, HANDOFF and CLAUDE.md are the lead's.
