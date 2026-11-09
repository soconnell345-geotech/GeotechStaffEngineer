# INF-32, app split step 11 (c) — check notes: guards and records, Word and Excel, the checks, the suite's framework, the document tools' binding and the shared assembly into the core

BACKLOG INF-32, plan `module_work/APP_SPLIT_PLAN.md` §2.3 ("everything
else"), §2.7 B, §5 (CI, "core-only") and "Step 11" (its Tests and Exit).
Branch `claude/inf-32-step11c` from `origin/claude/inf-32-step11b`
(`829158b`, step 11 (b), reviewed, not yet merged): `origin/master` did not
hold "Merge INF-32 step 11 (b)" when this branch was pushed, so it was not
merged in (§9 says what the merge will meet). Cloud session, 2026-11-09. No
version change, no tag, nothing published; BACKLOG, HANDOFF and CLAUDE.md
untouched.

**No behaviour change for a user**: both pages' agents are byte-identical
(`webapp/tests/page_fingerprints.json` unchanged, no re-record), every old
import path still imports and is the same module object, and GSE fills the
six new hooks with the code they replaced.

## 1. The file list: plan §2.3 against today's tree

Plan §2.3 was written 2026-10-21. Read against the tree at `829158b`:

| Plan §2.3 says | Today | (c) did |
|---|---|---|
| "What both agents also run": `keep_work`, `limits`, `tool_guards`, `scratch_guard`, `delivered_log`, `result_map`, `strict_args`, `warm_imports`, `calculate_tool` (all `deep/`), `safe_expr`, `runtime_check`, `review_flags` | all present | moved |
| `error_text`, `_fileio`, `arg_checks` | already in the core ((a): `error_text`, `_fileio`; (b): `arg_checks`) | nothing |
| `held_records` ("its report_ingest record shapes become a hook GSE fills") | present | moved, hook (§3) |
| "the assembly both builds share, extracted from `deep/agent.py` at step 8" | `funhouse_agent/deep/assembly.py` (556) | moved |
| Word and Excel: `docx_renderer`, `xlsx_writer`, `office_text` | present | moved |
| The run record and its checks: `webapp/activity_log.py`, `run_mirror`, `cited_pages`, `cited_documents`, `coverage`, `deep/coverage_tools.py` | `coverage.py` is 227 lines now (the page parsers; DR-40 removed the ledger, the gate and `document_coverage`); **`deep/coverage_tools.py` is gone** (DR-40); `coverage.py`'s `write_diggs` touch went with the ledger | moved the five that exist; `coverage` needs no hook any more |
| The suite's framework: `review_eval/runner.py`, `checks.py`, `documents.py`, `__init__.py`, and "the `Task` type and task selection from `tasks.py`" | present; `review_eval/tasks.py` 1,189 lines | `runner`, `checks`, `documents` moved; the `Task` type and selection split into a new core `review_eval/tasks.py` (§3); GSE's `review_eval/__init__.py`, its tasks and `report_fixture.py` stay; the core has its own `review_eval/__init__.py` |
| "The document tools' binding: what stays of `document_tools.py` after step 9" | 467 lines, binding `planlens.review` (PL-29 (b), 2026-11-09) | moved |
| `deep/busy_retry.py`, `deep/vision_kit.py` (held back from (a) to (c), CHECK_STEP11A §1) | present | moved |
| (not named in §2.3) `measure_tool.py`, `scale_labels.py`, `dxf_notes.py`: plan §2.6 *proposes* them for planlens | present; `vision_kit` imports all three, `measure_tool` imports `document_tools` and `scale_labels`; none imports GSE | **moved** (as (a) took `error_text`/`_fileio` and (b) `arg_checks`/`method_call`): without them the core's tool assembly would reach back into `funhouse_agent`. A later planlens release can still take them (they hand in nothing) |
| (not in the core) `markup_check.py`, `find_like.py`, `page_numbers.py` | 14-line stubs: old names for `planlens.review.*`, importing the binding first | stay in GSE (old names until step 16); they import the binding through its alias |
| (not in the core) `deep/document_gate.py` | the geotech page's gate (INF-67) | stays |
| Removed since the plan (DR-39, DR-40, 2026-11-09) | `deep/geometry_tools.py`, the findings ledger, `review_digest/`, `coverage_tools.py`, the sweep | nothing to move |

## 2. What moved

`git mv`, renames only (commit `d62b88d`), then the import rewrites and the
alias entries (`3212ba1`), then the hooks (`3f1a755`). Only the prefix changes: `funhouse_agent.X` and
`webapp.X` → `aec_agent_core.X`, `funhouse_agent.deep.X` →
`aec_agent_core.deep.X`, `funhouse_agent.review_eval.X` →
`aec_agent_core.review_eval.X`, in import statements only (docstrings that
name the old paths are text, left for step 16).

| From | To (`packages/aec-agent-core/aec_agent_core/`) | Lines (before → now) |
|---|---|---|
| `webapp/activity_log.py` | `activity_log.py` | 1,416 |
| `funhouse_agent/cited_documents.py` | `cited_documents.py` | 1,484 → 1,502 (hook, §3) |
| `funhouse_agent/cited_pages.py` | `cited_pages.py` | 1,382 |
| `funhouse_agent/coverage.py` | `coverage.py` | 227 |
| `funhouse_agent/deep/assembly.py` | `deep/assembly.py` | 556 |
| `funhouse_agent/deep/busy_retry.py` | `deep/busy_retry.py` | 295 |
| `funhouse_agent/deep/calculate_tool.py` | `deep/calculate_tool.py` | 322 |
| `funhouse_agent/deep/delivered_log.py` | `deep/delivered_log.py` | 103 |
| `funhouse_agent/deep/held_records.py` | `deep/held_records.py` | 295 → 316 (hook, §3) |
| `funhouse_agent/deep/keep_work.py` | `deep/keep_work.py` | 948 |
| `funhouse_agent/deep/limits.py` | `deep/limits.py` | 431 |
| `funhouse_agent/deep/result_map.py` | `deep/result_map.py` | 134 |
| `funhouse_agent/deep/scratch_guard.py` | `deep/scratch_guard.py` | 634 |
| `funhouse_agent/deep/strict_args.py` | `deep/strict_args.py` | 126 |
| `funhouse_agent/deep/tool_guards.py` | `deep/tool_guards.py` | 436 |
| `funhouse_agent/deep/vision_kit.py` | `deep/vision_kit.py` | 1,602 |
| `funhouse_agent/deep/warm_imports.py` | `deep/warm_imports.py` | 58 |
| `funhouse_agent/document_tools.py` | `document_tools.py` | 467 |
| `funhouse_agent/docx_renderer.py` | `docx_renderer.py` | 1,096 |
| `funhouse_agent/dxf_notes.py` | `dxf_notes.py` | 233 |
| `funhouse_agent/measure_tool.py` | `measure_tool.py` | 342 |
| `funhouse_agent/office_text.py` | `office_text.py` | 311 |
| `funhouse_agent/review_eval/checks.py` | `review_eval/checks.py` | 1,010 |
| `funhouse_agent/review_eval/documents.py` | `review_eval/documents.py` | 272 → 292 (hooks, §3) |
| `funhouse_agent/review_eval/runner.py` | `review_eval/runner.py` | 777 → 852 (hook, §3) |
| `funhouse_agent/review_flags.py` | `review_flags.py` | 351 |
| `funhouse_agent/run_mirror.py` | `run_mirror.py` | 447 |
| `funhouse_agent/runtime_check.py` | `runtime_check.py` | 203 |
| `funhouse_agent/safe_expr.py` | `safe_expr.py` | 191 |
| `funhouse_agent/scale_labels.py` | `scale_labels.py` | 287 |
| `funhouse_agent/xlsx_writer.py` | `xlsx_writer.py` | 346 |
| (new) | `review_eval/__init__.py` (a lazy door), `review_eval/tasks.py` (161, §3) | |

**16,782 lines moved** (31 files). New on GSE's side: `funhouse_agent/
core_hooks.py` (45), `funhouse_agent/review_eval/host.py` (50) and
`webapp/suite_page.py` (110), each the code it replaced (§3).

The shared assembly's compile stays one place
(`aec_agent_core/deep/assembly.py`, `compile_agent`), so REF-64 (B)'s
deployment-appended instructions can be added there later
(`module_work/REFERENCE_PACKS_PLAN.md` §3); nothing for it was built.

Eight tests of the moved code that pass with only the core and planlens
moved into `packages/aec-agent-core/tests/` (renames `61685ce`, then
their imports name the core `3f1187c`): `test_calculate_lists`,
`test_calculate_many`, `test_coverage_pages_offline`, `test_result_map`,
`test_runtime_check_offline`, `test_conversation_isolation`, and the two
(a) held back for (b)/(c), `test_list_files` and `test_page_reads_offline`.
The rest of the moved code's tests stay in the app for now: they build an
agent through `webapp.core`, import GSE's adapters or tools, or use the
app's fixtures (`test_hint_targets` scans GSE's adapters; tried and kept).

## 3. The hooks (each documented where it is asked for)

All through `aec_agent_core.hooks` (a callable, or a `"module:attribute"`
string imported the first time the hook is asked for), filled by GSE's
package doors as strings, as (a) did. Commit `3f1a755`.

| Hook | Asked for at | Unfilled (the core alone) | GSE fills it with (the code it replaced, moved as written) |
|---|---|---|---|
| `held_records.record_fields` | `aec_agent_core/deep/held_records.py:55` (`RECORD_FIELDS_HOOK`), read by `_fields` at :60 | no record is held (as when `report_ingest` would not import) | `funhouse_agent/core_hooks.py` `record_fields` (`report_ingest.model`'s `Investigation` / `LabTest` fields and kinds); filled at `funhouse_agent/__init__.py:120` |
| `cited_documents.reference_modules` | `aec_agent_core/cited_documents.py:972` (`REFERENCE_MODULES_HOOK`), `_reference_modules` at :975, used at :952 | `MODULE_DOCUMENTS`'s names (what the code did when the dispatch would not import) | `funhouse_agent/core_hooks.py` `reference_modules` (`funhouse_agent.dispatch.REFERENCE_MODULES`); filled at `funhouse_agent/__init__.py:122` |
| `review_eval.page_host` (plan 2.7 B: "each page hands in its own turn") | `aec_agent_core/review_eval/runner.py:89` (`PAGE_HOST_HOOK`; `PAGE_HOST_ATTRIBUTES` lists what a host has), `page_host()` at :123, used by `build_page_agent` (:143), `run_task` (:262) and the run record's `_app()` (:472) | `NoPageHost` (a run records it as a build error); `run.json` names `aec-agent-core` as the app | `webapp/suite_page.py` (`host()` returns the module): the runner's former calls into `webapp.core`, `webapp.profiles` and `webapp.output_capture`, looked up at call time so a patch of `webapp.core.stream_turn` still applies, and `APP_DIST`/`APP_PACKAGE`/`APP_COMMIT_ENV` = `geotech-staff-engineer`/`funhouse_agent`/`GEOTECH_APP_COMMIT` (what `run.json` named before); filled at `funhouse_agent/review_eval/__init__.py:22` |
| `review_eval.open_tasks` | `aec_agent_core/review_eval/tasks.py:29` (`OPEN_TASKS_HOOK`), `open_tasks()` at :97, `load_tasks(..., open=None)` at :105 | no shipped tasks (a run has the tasks it is given) | `funhouse_agent/review_eval/tasks.py` `open_tasks` (its `OPEN_TASKS`); GSE's `load_tasks` passes them in directly; filled at `funhouse_agent/review_eval/__init__.py:20` |
| `review_eval.document_dirs` | `aec_agent_core/review_eval/documents.py:135`, `_dev_dirs` at :145 | none | `funhouse_agent/review_eval/host.py` `document_dirs` (the checkout's folders, which the module computed from its own path before it moved) |
| `review_eval.fetch_document` | `documents.py:136`, used at :178 | none | `host.py` `fetch_document` (registers the SharePoint `primary_references/` fetcher, then `reference_docs.fetch`) |
| `review_eval.fixture` | `documents.py:137`, used at :211 | `KeyError` "unknown fixture" (as before for an unknown name) | `host.py` `fixture` (GSE's synthetic report, `report_fixture`) |

**The one split file.** `review_eval/tasks.py` is not a rename: plan §2.3
moves "the `Task` type and task selection" and leaves the tasks. The core's
`review_eval/tasks.py` (161 lines) holds `Task`, `GIVE_UP_PHRASES`,
`AUTO_CHECKS`, `load_tasks`, `_read_extra` and `select`, copied as written,
plus the hook; GSE's `funhouse_agent/review_eval/tasks.py` (1,189 → 1,098)
keeps its tasks, `CATEGORIES`, `DOC_TYPES`, `PAGES` and the term helpers,
re-exports `Task`, `AUTO_CHECKS`, `GIVE_UP_PHRASES` and `select`, and its own
`load_tasks` hands its `OPEN_TASKS` to the core's. `funhouse_agent.
review_eval.Task` and `...tasks.Task` are the core's class (tested).

**What the runner records is unchanged on GSE's page**: `run.json`'s
`versions` names `geotech-staff-engineer`, planlens, deepagents and
langchain; `commits` names `app` (from `GEOTECH_APP_COMMIT` or the checkout)
and `planlens`; the existing test `test_run_json_says_what_the_run_ran_on`
passes unchanged. `SUITE_AUTHOR` ("Review suite via GeotechStaffEngineer (AI
draft)") stays in the core as text; Document Review's copy can hand in its
own name at step 13 (noted, not changed).

## 4. Aliases, and the proof nothing loads twice

`funhouse_agent/__init__.py` `CORE_ALIASES` gains 30
(`funhouse_agent.{cited_documents, cited_pages, coverage, document_tools,
docx_renderer, dxf_notes, measure_tool, office_text, review_flags,
run_mirror, runtime_check, safe_expr, scale_labels, xlsx_writer}`,
`funhouse_agent.deep.{assembly, busy_retry, calculate_tool, delivered_log,
held_records, keep_work, limits, result_map, scratch_guard, strict_args,
tool_guards, vision_kit, warm_imports}` and
`funhouse_agent.review_eval.{checks, documents, runner}` one by one: the
package stays GSE's); `webapp/__init__.py` gains `webapp.activity_log`.
**54 aliases in all** (23 before). `calc_package.docx_renderer` (INF-30
step 5's own alias) now points at `aec_agent_core.docx_renderer` directly,
so importing it still loads nothing of `funhouse_agent`.

`webapp/tests/test_core_move_11c.py`:

- every one of the 31 is the same object under both names, and as the
  parent's attribute; a `monkeypatch.setattr("funhouse_agent.deep.limits.
  ...")` reaches the core; the old `from` imports give the core's objects;
- **no module loaded twice in any import order**: eight orders, each in a
  fresh interpreter (core name first, old name first, GSE's suite door
  before and after the core's runner, the run record beside GSE's builder,
  the binding beside a planlens stub, the review build beside the shell,
  `calc_package.docx_renderer` with both other names); after each, no core
  file is behind two module objects in `sys.modules`;
- the package doors fill the six hooks with the strings of §3, and each
  resolves to the code it replaced (record shapes equal `report_ingest.
  model`'s, the dispatch's reference modules, GSE's tasks, `webapp.
  suite_page` with every `PAGE_HOST_ATTRIBUTES` name, the report fixture);
  without the hooks, `held_records` holds nothing and `cited_documents`
  falls back as before.

`webapp/tests/test_core_move_11a.py` counts 54.

## 5. The no-GSE-import test

`packages/aec-agent-core/tests/test_core_imports.py` reads every import of
every core file at any depth. **`NOT_MOVED_YET` is now empty**: the core
imports nothing of `funhouse_agent` or `webapp` at all, inside functions
included (the eleven entries (a) and (b) left — `cited_documents`,
`document_tools`, `docx_renderer`, `office_text`, `dxf_notes`,
`review_flags`, `deep.keep_work`, `deep.limits`, `deep.strict_args`,
`deep.calculate_tool`, `deep.vision_kit` — are the core's own now). The
sanity walk names the 32 (c) files. The test covers every moved file without
a list.

`webapp/tests/test_review_boundary.py` reads the assembly at its core path;
its recorded couplings are unchanged (the review page's three builds, run
for a turn, reach GSE only through the listed soft ones), and the review
page still builds and runs with every GSE module refused.
`webapp/tests/page_builds.py`: middleware from `aec_agent_core` counts as
"ours" (the budgets and guards moved; without this the recorder would have
dropped their numbers and texts from the fingerprint, though nothing
changed), `funhouse_agent.core_hooks` and `funhouse_agent.review_eval.host`
are listed as GSE modules, and the capture works where GSE's builder is
absent (the core-only job).

## 6. The fingerprints

`webapp/tests/page_fingerprints.json` is byte-identical to `829158b`
(`git diff --quiet 829158b7 HEAD -- webapp/tests/page_fingerprints.json`),
and `test_page_fingerprints.py` passes for all three builds with no
re-record. Both pages' agents are unchanged.

## 7. The core-only CI job (new, `.github/workflows/tests.yml`, job `core-only`)

Python 3.11 and 3.12. Installs `packages/aec-agent-core`, `planlens>=0.15`
from PyPI, pytest and the hosting fleet's pins
(`tinyapps/wrapper_repo/requirements-thirdparty.txt`: streamlit 1.62.0,
PyJWT, msal, the Key Vault libraries); no submodule checkout; checks that
neither `geotech-staff-engineer` nor `geotech-references` nor an analysis
back end is installed. Then:

1. every core module imports (`pkgutil.walk_packages`, `python -I`, outside
   the checkout) with no app module loaded, and the core's tests pass
   (copied beside a link to the installed package);
2. `PYTHONPATH=<checkout> python -m webapp.tests.core_only`: today's two
   review builds (`review_default`, `review_minimal`) are built from this
   checkout's review files with every GSE module refused (the boundary
   test's blocker), each runs one turn on a fake model (`plot_data`,
   `calculate`, `write_docx`, a helper), and each build's fingerprint equals
   the recorded one.

**Run locally, both versions, from a clean worktree of the branch**
(`git worktree add`, fresh venvs, the job's steps as written):

| Step | Python 3.11.x | Python 3.12.3 |
|---|---|---|
| install; "geotech installed: [] []" | exit 0 | exit 0 |
| 66 core modules import, app modules loaded: [] | exit 0 | exit 0 |
| core tests | **131 passed** | **131 passed** |
| `webapp.tests.core_only` | "review_default, review_minimal build, run and match", exit 0 | same, exit 0 |

(Resolved there: aec-agent-core from source, planlens 0.15.0, PyMuPDF
1.28.2, deepagents 0.7.24, langchain 1.4.5, streamlit 1.62.0, pandas 3.1.1.)
The 131 were counted before the eight tests of §2 moved in; the fresh-venv
run of §8 is the later count.

## 8. Tests (judged by pytest's exit code)

Python 3.12.3, venv `/tmp/v` (`pip install -e packages/aec-agent-core -e
../geotech-references -e . pytest pytest-xdist build`; planlens 0.15.0 from
PyPI; the reference library from the sibling checkout, so the reference
tests that failed for (a) and (b) against PyPI's 1.5.0 pass here),
`pytest -q -n 6 -p no:cacheprovider`, each chunk's exit code recorded by
the runner script, on the branch head before the notes commit (`a5ee2e2`).

| Chunk | Exit | Result |
|---|---|---|
| `webapp/tests` (fingerprints, public text, boundary, the 11a/11b/11c alias tests, version guard, wheels' fast checks) | **0** | 1,286 passed, 5 skipped |
| `funhouse_agent/tests` + `funhouse_agent/deep/tests` | 1 | 5,163 passed, 34 skipped, **2 failed** (below) |
| `funhouse_agent/review_eval` + `funhouse_agent/chart_eval/tests` + `report_ingest/tests` + `packages/aec-agent-core/tests` | **0** | 5,218 passed, 23 skipped |
| `calc_package` (the one module suite that imports moved code: its `docx_renderer` alias; a grep for every moved name outside `funhouse_agent/`, `webapp/`, `report_ingest/` and the core finds only it and the user manual's prose) | **0** | 264 passed, 3 skipped |
| `webapp/tests/test_wheels.py` with `GEOTECH_REQUIRE_WHEEL_TEST=1` (builds both wheels, fresh venv, imports the (c) modules, checks the core wheel holds them) | **0** | 11 passed |
| `test_page_fingerprints.py` + `test_public_text.py` + `test_review_boundary.py` (in the first chunk) | **0** | no re-record |

**The 2 failures happen on unmodified base too**: the same two node ids
run on a worktree of `829158b` with `PYTHONPATH=<worktree>:<worktree>/
packages/aec-agent-core` (its own core imported, checked) fail the same
way, exit 1: `test_calc_package_figure_conventions.py::
test_large_pile_group_leaves_the_forces_to_the_table` and
`test_defaults_used.py::test_every_default_a_run_lists_changes_its_result
[profile_figure|plot_data]`, the matplotlib-version failures (a) and (b)
recorded.

An earlier full run of the same suites (before the tests commit, so without
the new tests) found one failure of this branch's:
`deep/tests/test_deep_tools_offline.py::
test_write_docx_without_a_renderer_returns_an_instruction` blocked the
renderer by its old name only; it now blocks the core name too (`9a9d568`).

**The fresh venv (core and planlens only)**, `/tmp/corevenv`, Python 3.12:
`pip install packages/aec-agent-core planlens==0.15.0 pytest` (exit 0;
no `geotech-staff-engineer`, `geotech_references` or `streamlit`), the core
reinstalled from the final tree:

- `python -I import_all.py` (`pkgutil.walk_packages` over
  `aec_agent_core`, each imported): **all 66 modules import**, no
  `funhouse_agent`, `webapp`, `geotech_*`, `report_ingest`,
  `calc_package` or `profile_figure` module loaded; exit 0;
- the core's tests, copied beside a link to the installed package,
  `python -I -m pytest -q tests`: **219 passed, exit 0** (123 from (a)
  and (b), 8 new in `test_core_suite_and_records.py`, 88 in the eight
  moved files);
- `PYTHONPATH=. python -m webapp.tests.core_only`: "review_default,
  review_minimal build, run and match", exit 0.

The core-only job's steps on 3.11 and 3.12 are §7.

## 9. What steps 12 and 13 need from here

**Merging master.** `origin/master` (`9a4c552`) holds (a)'s merge with its
review fixes but not (b). A trial `git merge-tree` of this branch into it
conflicts in two files only, both docs: `packages/aec-agent-core/README.md`
(the Status section: take this branch's, which names no development
version) and `aec_agent_core/__init__.py` (the docstring: take this
branch's; the version line `__version__ = "1.0.0.dev1"` is master's and
must win, with the pyproject's `1.0.0.dev1`, which merges cleanly). The
lead merges (b) first; (c) then merges onto it.

**Step 12 (the core's first real release, then GSE pinned to it):**

- The core is now complete as plan §2.3 defines it: nothing of
  `funhouse_agent` or `webapp` is imported anywhere in it, and the
  core-only job proves it installs and builds the review page with planlens
  alone. Its requirements did not change in (c) (everything the moved files
  import was listed at step 10/(a)/(b)); no package data (no moved file
  reads a file beside itself).
- The release itself: bump `1.0.0.dev1` → the release version in three
  places (`packages/aec-agent-core/pyproject.toml`, `aec_agent_core.
  __version__`, GSE's exact pin), `publish-core.yml` on a `core-v*` tag,
  then GSE's pin. `test_wheels.py` checks the three agree.
- The plan's one deliberate fingerprint change is at step 13 (the review
  build stops handing in the chart lines), not step 12.

**Step 13 (Document Review as `aec-document-review`):**

- Its page fills the core's hooks itself: the three `prompter.*` (its Key
  Vault lookup and per-tester keys), `review_eval.page_host` (its copy of
  `webapp/suite_page.py`, `APP_DIST="aec-document-review"`),
  `review_eval.open_tasks` (its 38 tasks, out of GSE's `tasks.py`),
  `review_eval.document_dirs` / `fetch_document` / `fixture` (its own
  documents, or left unfilled); `held_records.record_fields` and
  `cited_documents.reference_modules` stay unfilled there (no record
  shapes, `MODULE_DOCUMENTS`), and `vision_tools.*` reference hooks too.
- The documents table `review_eval/documents.py` `DOCUMENTS` still lists
  every task's document, GSE's report fixture included (resolved through
  the hook); when the review tasks leave GSE, the table can move with them
  (it is task data) and the core keep the resolver only.
- `SUITE_AUTHOR` names GeotechStaffEngineer (text); Document Review may want
  its own (hand it in through `build_agent`'s `markup_author`, which the
  host already receives).
- The boundary test's soft couplings are unchanged; step 13 removes the
  two that remain (`review_build -> profile_figure_adapter`,
  `aec_agent_core.hooks -> funhouse_agent.reference_vision`).

## 10. Release note

None for users: nothing a user sees changes. For library callers: the
modules in §2 now live in `aec-agent-core`; their old names
(`funhouse_agent.deep.assembly`, `funhouse_agent.review_eval.runner`,
`webapp.activity_log`, `calc_package.docx_renderer`, ...) keep working
until step 16. `funhouse_agent.review_eval.tasks.Task` is the core's class
(`aec_agent_core.review_eval.tasks.Task`); `funhouse_agent.review_eval.
runner._profile` is gone (the page's profile is chosen by
`webapp.suite_page`). A run of the review suite needs the page host, which
importing `funhouse_agent.review_eval` (or any of its old names) fills.

## 11. Noted, not fixed

- `runner.SUITE_AUTHOR` and the runner's docstrings still say
  "GeotechStaffEngineer" and name `webapp.*` (text; §9).
- `test_palantir_sdk_engine.py` (an (a) test) passes with only the core and
  planlens but was left in `webapp/tests`, outside (c)'s scope.
- `funhouse_agent/tests/test_hint_targets.py` reads GSE's adapters, so the
  hint tables' test stays in the app.
- Docstrings in the moved files still name `funhouse_agent.…` paths (text,
  step 16), as (a) and (b) left them.
