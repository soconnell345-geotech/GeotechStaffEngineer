# INF-32, app split step 11 (b) — check notes: the figure helper, plotting box, chart stack and worked examples into the core

BACKLOG INF-32, plan `module_work/APP_SPLIT_PLAN.md` §1.2, §2.3, §2.7,
§2.8 (OWN-77 decided (b)) and "Step 11". Branch `claude/inf-32-step11b`
from `origin/claude/inf-32-step11a` (`6362b21`, step 11 (a), under review),
2026-11-09. Cloud session. No version change, no tag, nothing published,
no planlens push; BACKLOG, HANDOFF and CLAUDE.md untouched.

**No behaviour change for a user**: the default pages' tools and prompts
are byte-identical (`webapp/tests/test_page_fingerprints.py` passes with
no re-record), every old import path still imports and is the same module
object, and GSE hands the chart stack its reference chart lines exactly as
before (the registration moved with the stack).

## 1. What moved

`git mv`, so git records renames (commit `b1a108c`, renames only); the
import rewrites are commit `15cb1a3`. Only the prefix changes (plan 2.3):
`funhouse_agent.X` → `aec_agent_core.X`, in import statements only
(docstrings that name the old paths are left; they are text, and the old
names stay valid through the aliases until step 16).

| From | To (`packages/aec-agent-core/aec_agent_core/`) | Lines (before → now) |
|---|---|---|
| `funhouse_agent/charts/__init__.py` | `charts/__init__.py` | 51 → 64 (docstring: it is the chart-lines hook, §3) |
| `funhouse_agent/charts/data_plot.py` | `charts/data_plot.py` | 1,845 |
| `funhouse_agent/charts/plot_tool.py` | `charts/plot_tool.py` | 530 |
| `funhouse_agent/charts/layout_ink.py` | `charts/layout_ink.py` | 1,285 |
| `funhouse_agent/charts/plan_furniture.py` | `charts/plan_furniture.py` | 140 |
| `funhouse_agent/charts/plotting.py` | `charts/plotting.py` | 47 |
| `funhouse_agent/charts/figure_examples.py` | `charts/figure_examples.py` | 994 (the nine geotech worked examples: six `plot_data` calls, three `plot_code` scripts) |
| `funhouse_agent/plot_box/__init__.py` | `plot_box/__init__.py` | 954 |
| `funhouse_agent/plot_box/child.py` | `plot_box/child.py` | 1,181 |
| `funhouse_agent/plot_box/figures.py` | `plot_box/figures.py` | 608 |
| `funhouse_agent/deep/figure_helper.py` | `deep/figure_helper.py` | 789 → 778 (the calc-helper name shim out, below) |
| `funhouse_agent/drawing_wait.py` | `drawing_wait.py` | 64 |
| `funhouse_agent/progress_notes.py` | `progress_notes.py` | 46 |
| `funhouse_agent/arg_checks.py` | `arg_checks.py` | 306 |
| `funhouse_agent/method_call.py` | `method_call.py` | 637 |

**9,477 lines moved** (15 files; 9,479 now).

**How the list was settled** (plan 2.3's "figure helper" paragraph and
2.8's column (a), read against each file's imports at any depth):

- The chart stack is step 6's `funhouse_agent/charts/` package, moved whole
  (its alias covers every module under it). That includes `plot_tool.py`,
  the `plot_data` method step 6 cut out of the geotech adapter; plan 2.3
  counted the stack before `plot_tool` existed, but `plot_data` is on both
  pages and the box and the helper call it.
- **`arg_checks.py` and `method_call.py` came along** although plan 2.3
  puts `arg_checks` in (c) and does not list `method_call` (step 6 made
  it): `plot_tool` imports `arg_checks` at module level and `method_call`
  in its call; both are leaves (`method_call` reaches only `_fileio`, moved
  in (a), and `arg_checks`). Without them the core would not import with
  only the core and planlens — the same reason (a) took `error_text` and
  `_fileio`. `funhouse_agent.dispatch` re-exports `method_call`'s names as
  before (the same objects, through the alias).
- `figure_to_png` and `get_pyplot` are `charts/plotting.py`.
- **The figure helper's `__getattr__` for `CALC_REDRAW_NOTE`,
  `FigureLooks` and `MODULE_FIGURE_LOOK` went.** Those three are the
  geotech calc helper's (`funhouse_agent.deep.calc_figures` since INF-30
  step 5); the shim imported that GSE module, which the core may not. Its
  only reader, `deep/tests/test_figure_turn.py`, now imports them from
  `calc_figures`; nothing else in the repository read them by the old name.
- Not moved: `funhouse_agent/deep/calc_figures.py` (GSE's calc helper; it
  imports the figure helper's private names by the old path, which the
  alias serves), the `profile_figure` adapter (keeps `plot_data` under
  `profile_figure` for `call_agent` and holds `reference_chart`), and
  step 6's stubs in GSE's packages (`geotech_common.layout_ink`,
  `geotech_common.plan_furniture`, `geotech_common.plotting`,
  `profile_figure.data_plot`, `profile_figure.figure_examples`): each
  imports `funhouse_agent.charts.X` and puts it in `sys.modules`, so it now
  lands on the core's module (tested, §2).

The plotting box loads `layout_ink.py` and `plan_furniture.py` **by file**
from `../charts/` beside `plot_box/` when it runs as `python -I child.py`;
both folders moved together, so the relative path holds (tested, §4).

## 2. The aliases added

`funhouse_agent/__init__.py`, `CORE_ALIASES` gains 7 (23 in all):

- `funhouse_agent.charts` → `aec_agent_core.charts` and
  `funhouse_agent.plot_box` → `aec_agent_core.plot_box`, as packages, so
  `funhouse_agent.charts.data_plot`, `funhouse_agent.plot_box.child` and
  the rest are covered;
- `funhouse_agent.{arg_checks, drawing_wait, method_call,
  progress_notes}` and `funhouse_agent.deep.figure_helper`.

`webapp/tests/test_core_move_11b.py` checks every entry and the eight
modules under the two packages (the old name and the new are the same
object, also as the parent's attribute), the old `from` imports
(`from funhouse_agent import plot_box`,
`funhouse_agent.charts.register_reference_charts`,
`funhouse_agent.dispatch.run_method`), a patch by the old name, and that
step 6's four GSE stubs reach the core's modules.
`test_core_move_11a.py` now counts 23 aliases. Callers keep the old names
(no churn; step 16 rewrites them).

## 3. The chart-lines hook, and how GSE fills it

The hook plan 2.8 asks for already existed: step 6 made
`funhouse_agent.charts.register_reference_charts(provider)` and
`reference_charts()`, and moving the package made it the core's
(`aec_agent_core.charts.register_reference_charts`). It stays a
registration of a provider OBJECT rather than an `aec_agent_core.hooks`
string, because the provider is four things at once — `chart_lines`, a
module the box loads from its file (`__file__`); `overlay(params)`, what
`plot_data`'s `reference_chart` draws; `parameter`, that argument's entry;
`returns`, its result note — and the geotech builds hand it in again on
every build. Commit `87d5a13` documents this in `charts/__init__.py` and
in `hooks.py`'s list of hooks.

**GSE fills it** with
`funhouse_agent.adapters.profile_figure_adapter.register_reference_charts()`
(the reference library's `geotech_references.chart_lines`, wrapped in
`_ReferenceCharts`): on that adapter's import, from each geotech build
(`deep/agent.py` → `deep/tools.register_reference_charts`), and from the
review build until step 13 (`review_build._register_reference_charts`,
inside a `try`; the boundary test's listed soft coupling, unchanged).

**Unfilled (the core alone)**: `plot_data` has no `reference_chart`
parameter and refuses it by name; the box's description says nothing of
`chart_lines` and loads no chart-lines file; everything else draws as
before. Tested in the core-only venv (§6).

## 4. The no-GSE-import test

`packages/aec-agent-core/tests/test_core_imports.py` (from (a)) reads every
import in every core file at any depth, so it covers the 15 moved files
without a list; its sanity walk now names them.

- `NOT_MOVED_YET` loses `funhouse_agent.plot_box.figures` (the core's own
  now; `vision_tools.list_files` imports `aec_agent_core.plot_box.figures`)
  and gains the (c) modules the figure helper and the box reach, each
  inside a function: `review_flags`, `deep.keep_work`, `deep.limits`,
  `deep.strict_args`, `deep.calculate_tool` and `deep.vision_kit`. None is
  a GSE file (the list's own test). `keep_work` is reached inside a `try`
  in the box (no turn scope → none), so the box runs in the core alone.
- **Checked by mutation**: a planted `from funhouse_agent import dispatch`
  and `import geotech_common.layout_ink` inside a function of
  `charts/plotting.py`, and a module-level `from funhouse_agent.deep import
  keep_work` in `plot_box/figures.py`, failed two tests with file and line
  (`plotting.py:51 -> funhouse_agent.dispatch`, `plotting.py:52 ->
  geotech_common.layout_ink`, `figures.py:610 -> funhouse_agent.deep.
  keep_work`); the files were restored.
- `webapp/tests/test_review_boundary.py`'s step-6 check (`CHART_STACK`)
  now points at the core's paths and also covers the figure helper,
  `drawing_wait` and `progress_notes`.
- New core-only tests, `packages/aec-agent-core/tests/test_core_charts.py`
  (7): `plot_data` draws with no chart lines registered; `reference_chart`
  refused by name; the box silent on `chart_lines`; a made-up provider read
  through the registration (offered, overlaid, its file found); the box's
  by-file chart folder beside it; the nine worked examples; `plot_code`
  runs in the box.

## 5. Requirements

None added (commit `0de90f7`, comment only): everything the moved code
imports — numpy, scipy, matplotlib, plotly, Pillow, langchain (the helper's
middleware and `StructuredTool`) — was already in the core's list from
step 10. pandas, which the box lets the model's code import, stays
unlisted, as in the app. `webapp/tests/test_version_guard.py` and
`test_wheels.py`' fast checks pass; `test_wheels.py`' fresh-venv import
list gains the 15 modules and the core wheel must hold
`charts/layout_ink.py`, `plot_box/child.py` and `deep/figure_helper.py`.

## 6. The fresh venv (core and planlens only)

```
python3.12 -m venv /tmp/corevenv
/tmp/corevenv/bin/pip install packages/aec-agent-core planlens==0.15.0 pytest   # exit 0
```

Installed `aec-agent-core 1.0.0.dev0` (built from the source), planlens
0.15.0, PyMuPDF 1.28.2, Pillow 13.0.0, matplotlib 3.11.2; no
`geotech-staff-engineer`, `geotech_references` or `streamlit`.

- `python -I import_all.py` (`pkgutil.walk_packages` over
  `aec_agent_core`, each imported): **all 34 modules import** from
  site-packages, no `funhouse_agent`, `webapp`, `geotech_*`,
  `report_ingest`, `calc_package` or `profile_figure` module loaded;
  exit 0.
- `python -I draw.py`: `reference_charts()` is None, `run_plot_data` with a
  made-up series writes a 48 KB PNG, status `success`, no app module
  loaded; exit 0.
- The core's tests, copied beside a link to the installed package:
  `python -I -m pytest -q tests` **123 passed, exit 0** (116 from (a) + 7).

## 7. Tests (judged by pytest's exit code)

Python 3.12.3, venv `/tmp/v` (`pip install -e packages/aec-agent-core -e .
pytest pytest-xdist build`; planlens 0.15.0 and geotech-references 1.5.0
from PyPI), `pytest -q -n 4 -p no:cacheprovider`, each chunk's exit code
recorded by the runner script. The base, `origin/claude/inf-32-step11a`
(`6362b21`), ran in a worktree with the same venv and
`PYTHONPATH=<worktree>:<worktree>/packages/aec-agent-core` (so its own
core, not this branch's editable one, was imported).

| Chunk | Branch: exit, result | Base `6362b21`: exit, result |
|---|---|---|
| `webapp/tests` (fingerprints, public text, boundary, wheels' fast checks, version guard, the 11a and 11b alias tests) | **0**: 1,205 passed, 5 skipped | 0: 1,174 passed, 5 skipped |
| `funhouse_agent/tests` + `funhouse_agent/deep/tests` | 1: 5,235 passed, 36 skipped, **16 failed** | 1: 5,235 passed, 36 skipped, **16 failed** |
| `funhouse_agent/review_eval` + `funhouse_agent/chart_eval/tests` + `report_ingest/tests` + `packages/aec-agent-core/tests` | **0**: 5,122 passed, 23 skipped | (not re-run; (a) recorded 0: 5,115 passed) |
| The 25 module suites that draw figures or import the chart stack, the box or the helper (grep for `layout_ink`, `geotech_common.plotting`, `figure_to_png`, `get_pyplot`, `plan_furniture`, `data_plot`, `figure_examples`, `plot_box`, `figure_helper`, `funhouse_agent.charts`): `axial_pile calc_package downdrag drilled_shaft fem2d geo_project geotech_common ground_improvement gstools_agent liquepy_agent opensees_agent pavement_design pile_group profile_figure pystra_agent pystrata_agent retaining_walls salib_agent seismic_geotech seismic_signals_agent settlement sheet_pile slope_stability subsurface_characterization wave_equation` | **0**: 5,578 passed, 52 skipped | (not re-run: the branch passed) |
| `packages/aec-agent-core/tests` alone | **0**: 123 passed | |
| `webapp/tests/test_page_fingerprints.py` + `test_public_text.py` (in the first targeted run) | **0**, **no re-record** | |
| `webapp/tests/test_version_guard.py` + `test_wheels.py` (fast checks) | **0**: 14 passed, 5 skipped | |

**The 16 failures are the same node ids on both trees** (`diff` of the
sorted `FAILED` lines is empty) and the same ones (a) recorded
(CHECK_STEP11A §7): 13 in `test_reference_adapters.py` (REF-61/62),
`test_reference_example_ids.py`, `test_citations.py` and two in
`test_fhwa_soils_foundations_reference.py` read the reference library,
here geotech-references 1.5.0 from PyPI, older than the pinned checkout
(not initialised in this container); `test_calc_package_figure_conventions.py::
test_large_pile_group_leaves_the_forces_to_the_table` and
`test_defaults_used.py::...[profile_figure|plot_data]` are matplotlib's
version. The webapp count moves by the new `test_core_move_11b.py` (31).

Not run: `webapp/tests/test_wheels.py` with
`GEOTECH_REQUIRE_WHEEL_TEST=1` (the wheel build); §6's fresh venv built the
core from source and imported every module instead.

## 8. What (c) must know

- **`NOT_MOVED_YET` grows by six for (c)**: `review_flags`,
  `deep.keep_work`, `deep.limits`, `deep.strict_args`,
  `deep.calculate_tool`, `deep.vision_kit`. When (c) moves them the
  imports in `plot_box/__init__.py`, `plot_box/figures.py` and
  `deep/figure_helper.py` become `aec_agent_core.…` and the entries go
  (the list-only-shrinks test fails until they do).
- **The core alone cannot yet build the figure helper's tools**:
  `make_figure_examples_tool`, `make_plot_code_tool` and
  `build_figure_subagent` reach `strict_args`, `limits`, `calculate_tool`
  and `vision_kit`. Drawing (`plot_data`, `run_plot_code`, the worked
  examples, the layout check) works now.
- **Tests stayed in the app**: the figure, box and chart tests
  (`funhouse_agent/tests/test_plot_box*.py`, `deep/tests/test_figure_*`,
  `profile_figure/tests/test_data_plot*`, `geotech_common/tests/
  test_layout_ink.py`, `test_plan_furniture.py` ...) import GSE's adapter,
  builder or stub names, so none passes in the core-only venv yet; they
  move (or are rewritten onto the core's names) when (c) and step 16 let
  them.
- `arg_checks` and `method_call` are already in the core (§1); (c) need not
  move them.
- The chart lines stay a registration, not a hooks-table entry (§3); step
  13 stops the review build filling it (the one deliberate fingerprint
  change of the plan).
- Docstrings in the moved files still name `funhouse_agent.charts…` and
  `funhouse_agent/charts/…`; left as text for step 16 (some sit next to
  prompt text the fingerprints read).

## 9. Release note

None for users: nothing a user sees changes. For library callers: the
chart stack (`aec_agent_core.charts`: `data_plot`, `plot_tool`,
`layout_ink`, `plan_furniture`, `plotting`, `figure_examples`), the
plotting box (`aec_agent_core.plot_box`), the figure helper
(`aec_agent_core.deep.figure_helper`), `drawing_wait`, `progress_notes`,
`arg_checks` and `method_call` now live in `aec-agent-core`; the old
`funhouse_agent.…` names (and step 6's `geotech_common.layout_ink`,
`profile_figure.data_plot` and the rest) keep working until step 16. One
name goes: `funhouse_agent.deep.figure_helper.FigureLooks`,
`CALC_REDRAW_NOTE` and `MODULE_FIGURE_LOOK` (old aliases since INF-30 step
5) — import them from `funhouse_agent.deep.calc_figures`.

## 10. Noted, not fixed

- The box offers `pandas` to the model's code but neither the core nor the
  app requires it (§5); unchanged by the move.
