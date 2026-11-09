# INF-32, app split step 11 (a) — check notes: model connections and vision calls into the core

BACKLOG INF-32, plan `module_work/APP_SPLIT_PLAN.md` §1.2, §2.3, §2.5,
§2.7 and "Step 11". Branch `claude/inf-32-step11a` from master `3a57539`
(2026-11-09). Cloud session. No version change, no tag, nothing published,
no planlens push; BACKLOG, HANDOFF and CLAUDE.md untouched (the lead's
notes at merge).

**No behaviour change for a user**: the default pages' tools and prompts
are byte-identical (`webapp/tests/test_page_fingerprints.py` passes with
no re-record), every old import path still imports and is the same module
object, and the GSE pages fill the two new hooks with the code they
replaced.

## 1. What moved

`git mv`, so git records renames (commit `ba24002`, renames only); the
import rewrites inside them are commit `0437b08`. Only the prefix changes
(plan 2.3): `funhouse_agent.X` and `webapp.X` → `aec_agent_core.X`,
`funhouse_agent.deep.X` → `aec_agent_core.deep.X`.

| From | To (`packages/aec-agent-core/aec_agent_core/`) | Lines (before → now) |
|---|---|---|
| `webapp/engine_config.py` | `engine_config.py` | 962 |
| `webapp/palantir_sdk_engine.py` | `palantir_sdk_engine.py` | 1,679 |
| `webapp/http_trace.py` | `http_trace.py` | 58 |
| `webapp/tinyapps_engine.py` | `tinyapps_engine.py` | 666 → 492 (the per-tester half to the page, §3) |
| `funhouse_agent/deep/databricks_bridge.py` | `deep/databricks_bridge.py` | 785 |
| `funhouse_agent/deep/pacing.py` | `deep/pacing.py` | 241 |
| `funhouse_agent/deep/vision_engine.py` | `deep/vision_engine.py` | 852 |
| `funhouse_agent/deep/inline_images.py` | `deep/inline_images.py` | 177 |
| `funhouse_agent/vision_tools.py` | `vision_tools.py` | 4,707 → 4,459 (GSE's readers out, §3) |
| `funhouse_agent/vision_view.py` | `vision_view.py` | 574 |
| `funhouse_agent/image_view.py` | `image_view.py` | 462 |
| `funhouse_agent/vision_probe.py` | `vision_probe.py` | 519 |
| `funhouse_agent/inline_store.py` | `inline_store.py` | 69 |
| `funhouse_agent/turn_question.py` | `turn_question.py` | 97 |
| `funhouse_agent/error_text.py` | `error_text.py` | 517 |
| `funhouse_agent/_fileio.py` | `_fileio.py` | 666 |
| (new) | `deep/__init__.py` (an empty door), `hooks.py` | 8, 106 |

**13,031 lines moved** (16 files). New on the app side:
`funhouse_agent/reference_vision.py` (312) and `webapp/tinyapps_keys.py`
(264), both cut from the moved files as written.

**How the list was settled** (plan 2.3's "model connections" and "vision
calls", read against each file's imports at any depth):

- `error_text.py` and `_fileio.py` came along although plan 2.3 puts them
  in (c) ("file handling"): both are leaves (only each other and the
  standard library), and the moved files import them everywhere, two at
  module level (`palantir_sdk_engine` → `error_text`). Without them the
  core would not import with only the core and planlens installed.
- **Held back to (c): `deep/busy_retry.py` (295) and `deep/vision_kit.py`
  (1,602).** `busy_retry` imports `tool_guards` at module level and
  `keep_work` and `limits` (the guard cluster, (c)). `vision_kit` imports
  `document_tools` (the binding, (c)), `limits` and `strict_args` at module
  level, and `charts`, `plot_box`, `drawing_wait` (b), `xlsx_writer`,
  `measure_tool` and `dxf_notes` inside functions: it is the tool
  assembly over (b) and (c), and moves after both. Moving either now would
  have made the core import GSE's package at module level.
- Plan 2.3's "vision calls" lines (9,335) counted `vision_view.py` at
  about 860 after step 9; the file is 574 now (step 9 took the rest).

## 2. The alias tables

Installed first thing in each old package door, before anything there
imports a moved name, with `aec_agent_core.install_aliases` (lazy: nothing
loads until an old name is looked up):

- `funhouse_agent/__init__.py`, `CORE_ALIASES` (12):
  `funhouse_agent.{_fileio, error_text, image_view, inline_store,
  turn_question, vision_probe, vision_tools, vision_view}` and
  `funhouse_agent.deep.{databricks_bridge, inline_images, pacing,
  vision_engine}`;
- `webapp/__init__.py`, `CORE_ALIASES` (4):
  `webapp.{engine_config, http_trace, palantir_sdk_engine, tinyapps_engine}`.

`webapp/tests/test_core_move_11a.py` checks every entry: the old name and
the new one are the same object (also as the parent's attribute), and a
`monkeypatch.setattr("funhouse_agent.vision_tools.X", ...)` reaches the
core's module. Callers outside the moved files keep the old names (no
churn now; step 16 rewrites them). `webapp/__init__.py` now imports the
core's alias helper and hook registry, both standard library only, so
`import webapp` still needs no Streamlit and no model.

## 3. The hooks, and how the pages fill them

`aec_agent_core/hooks.py`: `set_hook(name, target)`, `get_hook(name)`,
`clear_hook`, `hooks()`. A target is a callable or a `"module:attribute"`
string imported the first time the hook is asked for, so a package door
can fill a hook without importing anything. (b) and (c) can use it for
theirs (the chart lines, `held_records`, `coverage`, `cited_documents`,
the suite's turn runner).

**The Prompter connection** (plan Step 11 (a)). `aec_agent_core.
tinyapps_engine` no longer imports `webapp`:

| Hook | Asked for by | Unfilled (the core alone) | GSE's pages fill it with |
|---|---|---|---|
| `prompter.get_setting(name)` | `settings()`, `register()` | `os.environ` | `webapp.tinyapps_settings:get_setting` (env, else Key Vault, else `.env`) |
| `prompter.resolve_for(ident)` | `resolve_for()`, `build_for()` | the shared settings for everyone, or `EngineUnavailable(NOT_CONFIGURED_MESSAGE)` | `webapp.tinyapps_keys:resolve_for` |
| `prompter.tester_keys_possible()` | `register()` | `False` | `webapp.tinyapps_keys:tester_keys_possible` |

The Key Vault naming scheme and the per-tester keys by sign-in name
(`tester_suffix`, `tester_secret_name`, `key_mapping`, `PROMPTER-KEYS`,
`shared_fallback_allowed`, `no_key_message`, `key_source_report`, the
`TESTER_SEPARATOR` / `KEY_VAULT_NAME_MAX` / `ENV_KEYS` constants and the
"Per-tester keys" docstring) moved as written to the page's
`webapp/tinyapps_keys.py`, which reads settings through
`webapp.tinyapps_settings` and the caller through `webapp.identity` as
before. `NoPrompterKey`, `ResolvedPrompter`, the `SOURCE_*` names and
`SHARED_FALLBACK_ENV` (named in the key-source text) stay in the core:
they are the resolver's contract. `webapp/__init__.py` fills the three
hooks (strings), so every route that imports `webapp.tinyapps_engine`
(the two-page entry, the Tiny Apps entry, diagnostics, the tests) behaves
as before. `diagnostics._prompter_key_check` reads
`tinyapps_keys.key_source_report`. Document Review's page fills the hooks
from step 13.

**GSE's reference library in `vision_tools`** (plan 2.3: "GSE's 318
lines"; 2.7: "cut at step 11"). The core's `vision_tools` no longer
imports `reference_docs`, `worked_examples`, `chart_reading` or
`geotech_references`:

- `reference_read_roots()` reads `GEOTECH_REFERENCES_DOCS`, then asks
  `vision_tools.reference_read_roots` (inside a `try`, as before);
- `read_reference_figure` and `view_worked_example_source` are
  `APP_READER_TOOLS`, dispatched to the hooks `vision_tools.<name>`
  (`reader(arguments, engine)`); unfilled, "Unknown extended tool".

The readers, their two read-off notes and GSE's three read roots moved as
written to `funhouse_agent/reference_vision.py`; they reach the core
through the module (`_vt.ask_vision`, `_vt.display_path`, ...), so a
patch by either name still applies, and the checkout path
(`geotech-references/docs`) is computed from `funhouse_agent/`, the folder
`vision_tools.py` sat in. `funhouse_agent/__init__.py` fills the three
hooks. The boundary test now lists the filled hook,
`aec_agent_core.hooks -> funhouse_agent.reference_vision` (soft, step 13),
in place of the three `vision_tools` couplings it replaces;
`funhouse_agent.reference_vision` joined `page_builds.GSE_MODULES`.

`HAND_CHART_NOTE` stayed in the core: it is about `plot_data`, not
geotech. `VISION_TOOL_DESCRIPTIONS` (the v1 agent's prompt text) still
names the two readers; it is text, imports nothing, and is left for (c)
or step 16.

## 4. The no-GSE-import test

`packages/aec-agent-core/tests/test_core_imports.py` reads every import in
every core file with Python's parser, at any depth (plan 2.7 B):

1. nothing of `webapp` or of GSE's packages (the analysis packages, the
   reference library, `report_ingest`, `calc_package`, `profile_figure`,
   `geotech_common` ...);
2. of `funhouse_agent`, only `NOT_MOVED_YET` — plan 2.3 core files that
   (b)/(c) move — each **inside a function**; none of them is a GSE file
   (dispatch, adapters, the agent, prompts, readers, review files);
3. no first-party import at module level, so every core module imports
   with only the core and planlens;
4. `NOT_MOVED_YET` only shrinks (an entry nothing imports fails).

`NOT_MOVED_YET` today: `cited_documents`, `document_tools`,
`docx_renderer`, `office_text` (c); `plot_box.figures` (b); `dxf_notes`
((c) or planlens, plan 2.6's proposal). Checked by mutation: a planted
`from funhouse_agent import dispatch` and `import webapp.identity` inside a
function, and a module-level `from funhouse_agent import document_tools`,
each fail the test with file and line.

## 5. Requirements added (`packages/aec-agent-core/pyproject.toml`)

- `planlens>=0.15` (was 0.14): the app's pin since PL-29 (b).
- `PyMuPDF>=1.27.2,<2`: the vision tools open and render PDFs with
  `pymupdf` themselves; the app's range (INF-18), which also settles step
  10's open question about the cap for the core.
- `Pillow>=8`: the vision tools import `PIL` directly; matplotlib already
  requires it.

Nothing new installs anywhere: all three were already installed with the
app. Not added: `httpx` (a fallback import behind the SDKs' own client
classes), and the host SDKs (`palantir_models`, `databricks`, `anthropic`),
each imported inside a `try` as in the app. `test_version_guard`,
`test_tinyapps_entry` (pandas-3 rule) and `test_wheels`' fast checks pass.

## 6. Wheels and docs

- `docs/FOUNDRY.md` §1 items 3-4: **four** test wheels (app, core,
  planlens, geotech-references); a set from master without the core's
  wheel fails at import, and `--no-deps` hides it.
- `constraints/foundry_requirements.py`: a missing core is now an error
  when the installed app pins it (still skipped for 5.34.0 and older).
- `docs/DATABRICKS_INSTALL.md` §1: where the core comes from once a release
  from master is out (step 12), and from wheels or a checkout before that.
- `tinyapps/TINYAPPS.md`, `tinyapps/wrapper_repo/README.md`: route A needs
  `aec_agent_core/` in `app_src/`, route B the core's wheel.
- `webapp/tests/test_wheels.py`: the fresh venv imports the 19 core
  modules, and checks every old name is its core module; the core wheel
  must hold `hooks.py`, `vision_tools.py`, `engine_config.py` and
  `deep/`.
- `module_work/FOUNDRY_BRIEF_CHART_SUITE.md` ("three wheels") is a run
  already done; left as history.

## 7. Tests (judged by pytest's exit code)

Python 3.12.3, venv `/tmp/v` (`pip install -e packages/aec-agent-core -e .
pytest pytest-xdist build`; planlens 0.15.0 and geotech-references 1.5.0
from PyPI), `pytest -q -n 4`. Master `3a57539` run in a worktree, same
venv, same commands.

| Chunk | Branch: exit, result | Master `3a57539`: exit, result |
|---|---|---|
| `webapp/tests` (incl. fingerprints, public text, boundary, wheels' fast checks) | **0**: 1,174 passed, 5 skipped | 0: 1,153 passed, 5 skipped |
| `funhouse_agent/tests` + `funhouse_agent/deep/tests` | 1: 5,235 passed, 36 skipped, **16 failed** | 1: 5,306 passed, 36 skipped, **16 failed** |
| `funhouse_agent/review_eval` + `funhouse_agent/chart_eval/tests` + `report_ingest/tests` + `packages/aec-agent-core/tests` | **0**: 5,115 passed, 23 skipped | 0: 5,020 passed, 23 skipped |
| `packages/aec-agent-core/tests` alone | **0**: 116 passed | 0: 21 passed |
| `webapp/tests/test_page_fingerprints.py` + `test_public_text.py` alone | **0**: 18 passed, **no re-record** | |
| `webapp/tests/test_wheels.py` with `GEOTECH_REQUIRE_WHEEL_TEST=1` (builds both wheels, fresh venv) | **0**: 11 passed | |

**The 16 failures are the same node ids on both trees** (`diff` of the
sorted `FAILED` lines is empty): 13 in `test_reference_adapters.py`
(REF-61/62), `test_reference_example_ids.py`, `test_citations.py` and two in
`test_fhwa_soils_foundations_reference.py` read the reference library,
which here is geotech-references 1.5.0 from PyPI, older than the pinned
checkout (`geotech-references` at `8a09592`, not initialised in this
container); `test_calc_package_figure_conventions.py::
test_large_pile_group_leaves_the_forces_to_the_table` and
`test_defaults_used.py::...[profile_figure|plot_data]` are matplotlib's
version (as CHECK_STEP10 §8 found). None touches the moved code.

The counts move by the tests that moved: `funhouse_agent` 71 fewer (seven
files into the core), the core's chunk 95 more (those 71 plus 24 new);
`webapp/tests` 21 more (`test_core_move_11a.py`).

**Module suites that import the moved code:** a grep over the repository
outside `funhouse_agent/`, `webapp/`, `report_ingest/` and the core finds
only the root `conftest.py` (`funhouse_agent._fileio`), the script
`docs/foundry_app_launcher.py` and the run skill's
`.claude/skills/run-geotech-staff-engineer/driver.py`
(`webapp.engine_config`, `webapp.palantir_sdk_engine`), all by old names,
which the aliases serve; no analysis, calc_package, profile_figure or
geotech_common module imports them.

## 8. The fresh venv (core and planlens only)

```
python3.12 -m venv /tmp/corevenv
cd /tmp && /tmp/corevenv/bin/pip install packages/aec-agent-core planlens==0.15.0 pytest   # exit 0
```

Installed: `aec-agent-core 1.0.0.dev0` (built from the source), planlens
0.15.0, PyMuPDF 1.28.2, Pillow 13.0.0 and the core's other requirements;
no `geotech-staff-engineer`, `geotech_references` or `streamlit`.

- `python -I import_all.py` (walks `aec_agent_core` with `pkgutil` and
  imports each): **all 19 modules import** from site-packages; no
  `funhouse_agent`, `webapp`, `geotech_*`, `report_ingest`, `calc_package`
  or `profile_figure` module loaded; exit 0.
- The core's tests, copied beside a link to the installed package and run
  with `python -I -m pytest -q tests`: **116 passed, exit 0**.

Two tests first moved into the core went back to the app
(`test_list_files.py`, `test_page_reads_offline.py`, commit `cff42e6`):
in the fresh venv 11 of their cases failed, because `list_files` imports
`plot_box.figures` and `read_text_file` reads Office files through
`funhouse_agent` (both on `NOT_MOVED_YET`). They move with (b)/(c).

## 9. What (b) and (c) must know

- **The core alone cannot yet list files or read Office files**
  (`list_files` → `funhouse_agent.plot_box.figures.CODE_DIR`;
  `read_text_file` → `office_text`/`document_tools`; also `save_file`'s
  citation check → `cited_documents`, `write_docx` → `docx_renderer`, the
  DXF save check → `dxf_notes`). Each is on `NOT_MOVED_YET`; (b) moves
  `plot_box`, (c) the rest, then the entries go (the test fails until they
  do) and the import becomes the core's own.
- **`deep/busy_retry.py` and `deep/vision_kit.py` are (c)'s** (§1).
- **Hooks:** use `aec_agent_core.hooks` for (b)'s chart lines and (c)'s
  `held_records`, `coverage`, `cited_documents` and the suite's turn
  runner; fill them from the package doors as strings, as here.
- **Logger names follow the module:** `aec_agent_core.palantir_sdk_engine`
  and the rest (one test changed, `test_palantir_reasoning.py`).
- **Tests that move into the core** must pass with only the core and
  planlens (§8's recipe); a test whose code path reaches `NOT_MOVED_YET`
  stays in the app until that module moves.
- **The boundary test's hook coupling** (`aec_agent_core.hooks ->
  funhouse_agent.reference_vision`) goes at step 13, when Document Review's
  page stops loading GSE's package door.
- `VISION_TOOL_DESCRIPTIONS` in the core still names GSE's two readers
  (text only).

## 10. Release note

None for users: nothing a user sees changes. For library callers: the
modules listed in §1 now live in `aec-agent-core`
(`aec_agent_core.vision_tools`, `aec_agent_core.engine_config`, ...);
the old names (`funhouse_agent.vision_tools`, `webapp.engine_config`, ...)
keep working through aliases until step 16. Two names moved within GSE:
the per-tester key functions (`tester_suffix`, `tester_secret_name`,
`key_mapping`, `shared_fallback_allowed`, `no_key_message`,
`key_source_report`) are in `webapp.tinyapps_keys` (no longer on
`webapp.tinyapps_engine`), and the private reference readers
(`_dispatch_read_reference_figure`, `_dispatch_view_worked_example`) in
`funhouse_agent.reference_vision`; the tools themselves answer as before.
A GSE release from master still waits for the core on PyPI (step 12).

## 11. Noted, not fixed

- Pre-existing pyflakes notes in the moved `vision_tools.py` (two
  availability-probe imports, one f-string without placeholders):
  unchanged by a pure move.
- The master baseline's failures in this venv (§7) are environment ones
  (geotech-references 1.5.0 from PyPI is older than the pinned refs
  checkout; matplotlib's version), not this branch's.

## 12. Review fixes (independent review, 2026-11-09)

The review (verdict "merge after fixes") found one install problem, two
stale texts and one fail-open default, and fixed them in four commits:

1. **The core is 1.0.0.dev1** (pyproject, `__version__`, the app's exact
   pin). PyPI holds only 1.0.0.dev0, the EMPTY package that reserved the
   name (`pip download` gives `__init__.py` alone). Pinned at 1.0.0.dev0,
   an install of the app from master could take that package and fail at
   import (`No module named 'aec_agent_core.hooks'`). `publish.yml`'s
   guard ("the pinned core is on PyPI") also passed, so a `v*` tag would
   have published that app.
   - The guard's own script, run locally against PyPI, stops dev1 (exit 1,
     "not on PyPI ... Publish the core first") and passes master's dev0
     (exit 0).
   - `test_wheels.py` and `test_version_guard.py` pass, and so does the
     opt-in wheel build test.
   - Stale comment fixed: the app's pyproject said the core was not on
     PyPI at all.
2. **The core's README and `__init__` docstring** say what it holds since
   (a), not "an empty package".
3. **Doc pointers** follow the moved files:
   - `tinyapps/TINYAPPS.md`: the Prompter engine, the testers' resolution
     order (now `webapp/tinyapps_keys.resolve_for`), the file list;
   - `docs/FOUNDRY.md`: the SDK wrapper, twice;
   - two `CLAUDE.md` path references.
4. **The key resolver fails closed without a page.** Before, with nothing
   in `prompter.resolve_for`, `aec_agent_core.tinyapps_engine.resolve_for`
   gave every caller the shared key, testers identified by the header
   included. A page that forgot the hook would have dropped the per-tester
   access list without a word. Now, unfilled:
   - a multi-user caller is refused (`NoPrompterKey`, "No AI key is set up
     for <name> yet — ask the app owner."), unless
     `PROMPTER_SHARED_KEY_FALLBACK` is on (then source `shared-fallback`);
   - a single-user caller, or none given, keeps the shared key.

   `no_key_message` moved into the core so both refusals use the same
   words. `webapp.tinyapps_keys` imports it, so it is importable from
   both; this corrects §3 and §10, which list it as page-only. Tests:
   three in `packages/aec-agent-core/tests/test_core_hooks.py` (refused;
   let in by the fallback setting; single-user unchanged). The GSE pages
   fill the hook and are unchanged: the review's 13-scenario key-route
   probe gives output byte-identical to master's. The probe covers
   `resolve_for`, `key_source_report`, the secret names, the mapping,
   `register()` and `resolve_engine(identity=...)`.

**For later (step 16):** `funhouse_agent/chart_eval/runner.py:259` finds
the engine module by its old name (`sys.modules.get("webapp.engine_config")`)
to drop the CA clients a forked page run inherits. That works while every
model builder imports `webapp.engine_config`. Once step 16 moves callers to
the new names, it would silently drop nothing. Use
`sys.modules.get("aec_agent_core.engine_config")`, which is always present
once the module is loaded.
