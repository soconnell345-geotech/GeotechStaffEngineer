# INF-32, app split step 10 — check notes: the core package, empty

BACKLOG INF-32 (steps 10-11), plan `module_work/APP_SPLIT_PLAN.md` §1.1,
§1.2, §2.3, §4 "Step 10" and §5. Branch `claude/inf-32-step10-core`, from
master `05eab2d` (2026-11-06). Cloud session. No version change, no tag,
nothing published; BACKLOG, HANDOFF and CLAUDE.md untouched (the lead's
notes at merge). **No behaviour change: no code moved.**

## 1. What was added

| Path | What |
|---|---|
| `packages/aec-agent-core/pyproject.toml` | `aec-agent-core` 1.0.0.dev0 (OWN-72's names); MIT with the app's `LICENSE` copied beside it (a test keeps the two byte-equal); `requires-python >=3.10`; the dependency list below; `[anthropic]` extra; the wheel takes `aec_agent_core*` only, no tests |
| `packages/aec-agent-core/README.md` | The PyPI page: what the core is, that 1.0.0.dev0 is empty, how GSE installs it until step 12. No Department, host or deployment named |
| `packages/aec-agent-core/aec_agent_core/__init__.py` | Only the alias helper (§3) and `__version__` |
| `packages/aec-agent-core/tests/test_core_aliases.py` | 21 tests of the helper |
| `webapp/tests/test_wheels.py` | The wheel test (§4): 6 fast checks, 5 slow |
| `module_work/inf32/CHECK_STEP10.md` | This note |

Changed:

- `pyproject.toml` (the app): `"aec-agent-core==1.0.0.dev0"` in its core
  dependencies, exact, with a comment saying why and how to install it
  before step 12; `packages/aec-agent-core/tests` in `testpaths`. Nothing
  else: every dependency the app had stays (step 11 or later decides
  whether lines the core now carries come out of the app's list).
- `.github/workflows/tests.yml`: the main job installs
  `-e packages/aec-agent-core` before `-e .`; the numpy 1.26 job adds it to
  its `--no-deps` line; a new **"Wheels"** job (§4).
- `.github/workflows/nightly.yml`: installs the core before the app.
- `.github/workflows/publish.yml`: `python -m build --outdir dist/ .` (the
  root, named), then a step that fails if anything but
  `geotech_staff_engineer-*` is in `dist/`. Nothing builds or publishes the
  core; a `core-v*` tag matches no trigger (`v*` needs a leading `v`).
- `constraints/foundry_requirements.py`: the core is one of "ours" (its own
  name left out of the printed list, its requirements printed); an install
  that predates the core (5.34.0 wheels) is skipped for the core only.
- Tests that read `pyproject.toml` (plan's list, each checked):
  - `webapp/tests/test_public_text.py`: the shipped-files walk now covers
    every `packages/*/pyproject.toml` with its own `include`, plus each
    package's README and pyproject; a new test pins that the core is in the
    walk and its tests are not.
  - `webapp/tests/test_version_guard.py`: the guard's ceilings are checked
    against the caps in BOTH pyprojects.
  - `webapp/tests/test_tinyapps_entry.py`: a new test holds the core to the
    pandas-3 rule (no python-ags4, openpyxl kept), since the core is what
    Document Review installs beside the fleet pins.
  - `test_docs_currency`, `test_ws_upload`,
    `funhouse_agent/tests/test_disclaimer.py`, `test_drawing_ir_adapter.py`,
    `funhouse_agent/deep/tests/test_coverage_offline.py`: read, nothing the
    new file breaks, unchanged (they read the app's version, its
    package-data, its console script and its planlens floor, all as before;
    `test_coverage_offline` does not read the pyproject at all).

## 2. The dependency list and where each line comes from

Plan §1.1's list, at the floors and caps the app's pyproject has on
`05eab2d` (its comments carry the reasons; the core's file says so briefly):

| Requirement | Source |
|---|---|
| `numpy>=1.26` | app; INF-53 (Foundry's numpy 1.26.4) |
| `scipy>=1.13` | app; §1.1 (the plotting box lets code import it) |
| `deepagents>=0.6.8,<0.7.25`, `langchain>=1.3,<1.4.6`, `langgraph>=1.2,<1.3` | app; INF-03 caps (2026-10-27) |
| `typing_extensions>=4.13` | app; not in §1.1's list, added: it is the agent stack's floor (langchain-protocol's PEP 728 use) and every host of the core runs that stack |
| `openai<3.29.1` | app; INF-03 |
| `langchain-openai>=1.3` | app |
| `matplotlib>=3.8`, `plotly>=5` | app |
| `python-docx>=1.1`, `markdown-it-py>=3`, `openpyxl>=3.1` | app |
| `planlens>=0.14` | app (5.34.0 pin) |
| `[anthropic]`: `langchain-anthropic>=1.4` | app; §1.1 |

Left out on purpose, as in §1.1: streamlit, `websockets<16` and `jinja2`
(the chat page's, each app's own), the analysis back ends,
geotech-references. **Open for the lead:** the app's `PyMuPDF>=1.27.2,<2`
cap (INF-18) is not in the core. With the core and planlens alone (Document
Review's install after the split) PyMuPDF 2 would resolve, with its
onnxruntime import cost. Either the core carries the cap, or Document
Review's own pins do; decide before step 13.

Step 10 installs nothing new: every requirement of the core is already a
core requirement of the app.

## 3. The alias helper (for step 11)

`aec_agent_core.install_aliases({"funhouse_agent.vision_tools":
"aec_agent_core.vision_tools", ...})`, called from the OLD package's
`__init__.py`. A meta-path finder, first in `sys.meta_path`, answers an old
name with a spec whose loader hands back the already-imported new module,
so:

- the old name and the new are the same module object (`sys.modules` under
  both; the parent package gets the attribute too); `from old import x`,
  `import old.x` and `monkeypatch.setattr("old.x.NAME", ...)` all reach the
  real module (tested);
- lazy: nothing imports until the old name does (tested; matters for the
  boundary test's import counts). `importlib.util.find_spec(old)` counts
  as a look-up: it imports the target (review, §9);
- a package alias covers what is under it (`old.box.child` →
  `new.box.child`), the longest alias wins;
- the module keeps its own `__name__`, `__file__` and `__spec__` (the import
  system overwrites `__spec__` on any module a loader hands it; the loader
  puts the real one back), so `importlib.reload` and pickling by reference
  work (tested);
- a stale file left at the old path cannot shadow the move (tested);
- refused with `ValueError`: an alias into itself, a re-pointed alias, and
  an old name already imported as a different module (two copies).
  `remove_aliases()` and `aliases()` for tests and step 16.

Step 6's four stub files (`profile_figure/data_plot.py`,
`profile_figure/figure_examples.py`, `geotech_common/layout_ink.py`,
`geotech_common/plan_furniture.py`, each `sys.modules[__name__] = moved`)
can become alias-table entries at step 11 (b); an import of the old name
through a stub already yields the moved module, so `install_aliases`
accepts it.

## 4. The wheel test and how CI runs it

`webapp/tests/test_wheels.py`:

- **Fast (everywhere):** the app's exact pin equals the core's version and
  `aec_agent_core.__version__`; the two licences are equal; the app's
  `include` has no core and the core's has only itself; publish builds `.`
  only; tests and nightly install the core before the app.
- **Slow (`@pytest.mark.slow`, about 15 s here):** builds both wheels as a
  release does (`python -m build --no-isolation`: sdist, then the wheel from
  the sdist, so no `build/` folder in the tree and no network), then:
  - neither wheel has a `tests`, `module_work`, `raw` or `packages` path
    part, a `tinyapps/reference` path, a `.pyc` or a `.env`;
  - the core wheel is `aec_agent_core` alone, with its licence, version and
    `[anthropic]` extra in its METADATA;
  - the app wheel has `report_ingest`, `webapp`, `funhouse_agent`, the
    upload component page, the eval questions, the worked examples, the
    report checklist, no `aec_agent_core`, and `Requires-Dist:
    aec-agent-core==1.0.0.dev0`;
  - both are installed `--no-deps --no-index` into a FRESH venv, which sees
    the running interpreter's site-packages by a plain path line (a path
    line runs no `.pth`, so this repository's editable installs are NOT
    visible there; the geotech-references checkout is added by its own
    folder); 13 modules (`aec_agent_core`, `funhouse_agent` and its deep
    agent, `webapp.core`, `webapp.profiles`, the Databricks launcher,
    `report_ingest`, `geotech_common`, `calc_package`, `profile_figure`)
    are imported there with `python -I` and each one's file is checked to
    be inside the venv; the four page entry points
    (`webapp.app`, `pages_entry`, `tinyapps_entry`, `foundry_entry`) are
    found in the venv without being run; installed metadata agrees on the
    core's version and the app's pin;
  - the `geotech-disclaimer` console script runs from the venv.
- The slow tests are opt-in (since the review, §9): they run only with
  `GEOTECH_REQUIRE_WHEEL_TEST=1` and skip, saying so, without it; with it,
  a missing `build` package is a failure. So the main "Tests" job, nightly,
  and a local or xdist run skip them (building the root writes into the
  source tree, so two builds at once must not happen). On Windows, if
  Smart App Control refuses the fresh `geotech-disclaimer.exe` (WinError
  4551), the console-script check skips with that reason.
- **CI's "Wheels" job** (tests.yml, Python 3.12, minutes): installs the
  core, then the app and geotech-references, plus `pytest build
  setuptools wheel`, sets `GEOTECH_REQUIRE_WHEEL_TEST=1` and runs this file
  alone.

## 5. Install docs: what breaks, to change at step 12 (not rewritten now)

The released 5.34.0 is untouched by any of this. What changes is installing
**from master** (or from a wheel built from master) from now on:

1. **A GSE release from master needs the core on PyPI first.** The exact pin
   `aec-agent-core==1.0.0.dev0` cannot resolve from PyPI or Nexus until the
   core is published, so `pip install geotech-staff-engineer==<next>` would
   fail. Any GSE release after this merge therefore waits for step 12
   (core first, then GSE), or this pin is taken out for it. **The owner
   should know this before the next release word.** Since the review the
   publish workflow enforces it: a step before the upload fails the
   publish unless the pinned core version is on PyPI (§9, F1).
2. `docs/DATABRICKS_INSTALL.md` §1 (`%pip install
   "geotech-staff-engineer==X"`): unchanged in form after step 12; the core
   then arrives beside it. The log section ("what good looks like", the
   expected warnings) should gain the line `aec-agent-core-<v>` among the
   installed packages, and the history table a row. The "Nexus mirror lag"
   note now applies to two packages of ours plus planlens: the core must
   have cleared the Nexus scan before GSE is installed there.
3. `docs/FOUNDRY.md` §1 item 3 ("Test wheels"): **three wheels become
   four** (app, core, planlens, geotech-references): upload, rename check
   and the start-up `pip install --no-deps --no-index --target` all need the
   core's wheel too; item 4 (numpy 1.26) likewise, and
   `constraints/foundry_requirements.py` already counts four. A test-wheel
   set built from master after this merge without the core wheel will fail
   at import only after step 11 (until then nothing imports the core); with
   `--no-deps` pip will not complain either way, so the hand-off notes for
   the next Foundry build must list it.
4. `tinyapps/TINYAPPS.md` and `tinyapps/wrapper_repo/`: the Nexus route
   (`packages.txt`: `geotech-staff-engineer>=5.26`) gets the core through
   the resolver once it is on Nexus; the wheel route (`wheels/*.whl`) needs
   the core's wheel beside GSE's; the source route (`app_src/` on
   `PYTHONPATH`) needs the core's source folder on the path too and its
   requirements (`app_src/requirements.txt` or the third-party pins). The
   dosdev round (OWN-29/OWN-30) should be run on a set that has it.
5. Any developer or session setup that installs master with `pip install
   -e .` must install the core in the same command: `pip install -e
   packages/aec-agent-core -e . ...`. (Corrected after the review:
   `README.md` has no developer install, only `pip install
   geotech-staff-engineer`, an end-user install for step 12.) The two
   setups that do this, the cloud briefs
   `module_work/cloud_briefs/COMMON_MODULES.md` and
   `COMMON_REPORT_INGEST.md`, are changed on this branch (§9, F2); a
   search of `CLAUDE.md`, `README.md`, `.claude/`, the cloud briefs and
   the workflows found no other. The end-user guides
   (`docs/DATABRICKS_INSTALL.md`, `docs/FOUNDRY.md`, `tinyapps/`) wait for
   step 12 as items 2-4 say.

## 6. What step 11 and step 12 need from this

- **Step 11:** the files of §2.3 move under
  `packages/aec-agent-core/aec_agent_core/`; `funhouse_agent/__init__.py`
  and `webapp/__init__.py` call `install_aliases` with their tables (the
  core must never import either app). The core's tests move with their
  code into `packages/aec-agent-core/tests/` (already in `testpaths`), with
  unique file names (pytest's default import mode needs unique basenames).
  The wheel test's `IMPORTS["aec_agent_core"]` should grow to the moved
  modules, and its "core wheel is `aec_agent_core` alone" check stays. The
  "core-only" CI job (§5) is step 11's, not added here.
- **Step 11, the alias helper (the review's notes):**
  - `python -m <old name>` fails (runpy needs `get_code`; for a package
    the target's `__main__` runs once as a plain import before the
    error). No documented `-m` command targets a module on the §2.3
    list; use the new names.
  - `importlib.util.find_spec(<old name>)` imports the target; only
    `install_aliases` itself is lazy.
  - Logger names follow the new `__name__`, so a test that captures by
    logger name needs the new one: `webapp/tests/test_palantir_reasoning.py`
    (`caplog.at_level(..., logger="webapp.palantir_sdk_engine")`) stops
    capturing when `palantir_sdk_engine` moves (it fails loudly).
  - A package alias covers every name under it (that is how a stale file
    at the old path is kept out). A package that splits, such as
    `review_eval` (the framework to the core, GSE's tasks stay), must be
    aliased module by module, or GSE's own file at the old path can no
    longer be imported.
  - Install the table at the top of the old package's `__init__`, before
    any import of a moved name.
  - Tests that start a Python subprocess importing the app need
    `packages/aec-agent-core` on that subprocess's `PYTHONPATH` (the root
    conftest covers only the pytest process). Runs from source outside
    pytest (the run skill's driver, `streamlit run`) in the shared local
    venv need the core on the path too.
  - The Foundry test wheels need a fourth wheel, the core's, at step 11 (a):
    `docs/FOUNDRY.md` §1 items 3-4 (and the brief template's list of
    wanted wheels) still say three.
- **Step 12:** bump the core from 1.0.0.dev0 to 1.0.0 in its pyproject,
  `__version__` and the app's exact pin together (the fast wheel test
  fails until all three agree); add the `core-v*` tag trigger and a job that
  builds `packages/aec-agent-core` by name, with the same "only this
  package in dist/" check; publish the core before GSE; the doc changes of
  §5.

## 7. Left open

- The name `aec-agent-core` is not registered on PyPI (the review found
  `https://pypi.org/pypi/aec-agent-core/json` answering 404), while this
  PUBLIC repository pins `aec-agent-core==1.0.0.dev0`: anyone could take
  the name and upload that version. PyPI reserves a name only with an
  upload, so registering it means publishing, which is the owner's
  decision. Until then every documented install puts the local core
  first (CI, nightly, the cloud briefs), so none resolves it from an
  index, and the publish guard stops a GSE release that would.
- PyMuPDF `<2` in the core or in Document Review's pins (§2).
- The app's package-data line `calc_package = ["templates/*.html",
  "templates/*.tex"]` matches nothing (`calc_package/templates/` does not
  exist); noted, not fixed (no output changes).
- Whether the app keeps listing requirements the core now carries, or
  drops them for the core's (after step 11).
- The 14 environment failures of §8.

## 8. Exit: the gate on the exact tree

Python 3.11.18, a fresh venv (core editable first, then the app and
geotech-references editable), `pytest -n 4 --dist worksteal`, judged by
pytest's exit code, chunk by chunk (2026-11-06, the tree of commits `3308a0d` and `47ca9dc`; nothing
under test changed after them):

| Chunk | Exit | Result |
|---|---|---|
| `webapp/tests` + `packages/aec-agent-core/tests` | 0 | 1,165 passed |
| `funhouse_agent/tests` | 1 | 4,240 passed, 28 skipped, **2 failed** |
| `funhouse_agent/deep/tests` | 0 | 1,187 passed, 4 skipped |
| `funhouse_agent/review_eval` + `review_digest` | 0 | 203 passed |
| the module suites (every other `testpaths` entry) | 1 | 18,424 passed, 104 skipped, 3 xfailed, **12 failed** |
| `webapp/tests/test_wheels.py` alone, with `build` | 0 | 11 passed (5 slow), about 15 s |

**The 14 failures are not this branch's.** The branch changes nothing under
`opensees_agent`, `funhouse_agent` (code), `profile_figure`, `calc_package`
or `geotech_common` (`git diff --stat 05eab2d -- ...` is empty), and the
two `funhouse_agent/tests` failures fail the same way from a checkout of
master in the same venv. They are this cloud venv's versions, newer than
CI's:

- `test_calc_package_figure_conventions::test_large_pile_group_leaves_the_forces_to_the_table`
  and `test_defaults_used::...[profile_figure|plot_data]` (`logx`/`logy`
  changed nothing): matplotlib **3.11.2** here;
- 12 OpenSees site-response tests (`test_opensees_agent.py`
  `TestSiteResponseIntegration`, `TestSiteResponseBenchmarks`,
  `TestPorePressureRatio`; all five of `test_site_response_wiki_example.py`):
  the solver child process reports "See stderr output" with openseespy
  **3.7.1.2** on Python 3.11 / Linux here.

Neither is pursued here (no output of a saved report changes from this
branch); the lead may want a BACKLOG line if CI's nightly 3.11 job shows
the same.

The branch merges cleanly onto master `76d27ca` (checked with
`git merge-tree`); the gate was run on the branch as based on `05eab2d`.

## 9. Review and fixes (2026-11-05)

An independent review of this branch (F1-F8, most severe first) and what
changed for each. No behaviour change to the app; no code moved.

| Finding | What changed |
|---|---|
| F1 (HIGH) a GSE release before step 12 could not be installed | `publish.yml`: a step before the upload reads the exact `aec-agent-core` pin from `pyproject.toml` and asks `https://pypi.org/pypi/aec-agent-core/<version>/json`; a 404 fails the publish with a plain error naming step 12. No core requirement: nothing to check. A requirement that is not one exact pin, or PyPI unreachable: fails, saying why. The build, the `dist/` check, the upload and the publish job are unchanged. At step 12 the core goes up first and the step passes by itself. |
| F2 (HIGH) cloud sessions stop at setup | Both cloud briefs (`COMMON_MODULES.md`, `COMMON_REPORT_INGEST.md`) now set up with `pip install -e packages/aec-agent-core -e . pytest pytest-xdist`, one command. A search of `CLAUDE.md`, `README.md`, `.claude/`, the cloud briefs and the workflows found no other developer setup with `pip install -e .` (there is no CONTRIBUTING, devcontainer or setup hook); old CHECK notes that record how they ran are history, left. §5.5 corrected (README has no developer install). End-user guides wait for step 12. |
| F3 (HIGH, local) the shared-venv gate broke | (a) The root `conftest.py` puts this checkout's `packages/aec-agent-core` first on `sys.path`, so a plain `pytest` collects the core's tests with no install. (b) The slow wheel tests are opt-in (`GEOTECH_REQUIRE_WHEEL_TEST=1`, which CI's Wheels job already sets; tests.yml's comment updated), so a local or xdist run never builds the root, let alone twice at once; with the variable set a missing `build` is a failure. The console-script check skips with the reason when Windows refuses the fresh exe (WinError 4551, Smart App Control). Docstring's "a minute or two" now "under a minute". |
| F4 (MEDIUM, step 11) a worktree would test another checkout's core | Fixed by F3 (a): each checkout's conftest puts its own core first. Subprocess tests need `PYTHONPATH` at step 11 (§6). |
| F5 (MEDIUM, step 11) the core's tests removed the apps' aliases | `test_core_aliases.py`: the fixture saves the table and removes only the names its test added (never `remove_aliases()` with no name, which removes all); assertions check only the test's own entries; the finder is checked gone only when no other alias is left, and checked still there when one is. Checked with an alias installed before the suite: 21 passed, and that alias and the finder were still in place after. |
| F6 (MEDIUM) the name can be taken on PyPI | The owner's decision (registering means an upload): §7. Until then every documented install puts the local core first, and F1's guard stops a release that would resolve it from PyPI. |
| F7 (LOW) CI has never run on this branch | `tests.yml` already has `workflow_dispatch` (line 11), so the lead can run it on this branch before merging; nothing added. |
| F8 (LOW, step 11 (a)) the Foundry test wheels | Left for step 11 (a), as the review says; noted in §6. |
| Step-11 notes on the alias helper | Added to §6. §3's "lazy" now says `find_spec` on an old name imports the target, as does the helper's module docstring. |

Tests rerun from this worktree with the shared Windows `.venv` (Python
3.14, pytest 9.0.3; the core NOT installed there), no `PYTHONPATH`,
`-p no:cacheprovider`, judged by exit code:

| Run | Exit | Result |
|---|---|---|
| `packages/aec-agent-core/tests` (collects through the conftest) | 0 | 21 passed |
| same, with an app alias installed before the suite (a throwaway runner) | 0 | 21 passed; the alias and the finder still in place after |
| `webapp/tests/test_wheels.py`, plain | 0 | 6 passed, 5 skipped (opt-in) |
| `webapp/tests/test_wheels.py`, `GEOTECH_REQUIRE_WHEEL_TEST=1` | 0 | 11 passed in 24 s (Smart App Control let the exe run this time; the WinError 4551 skip was checked with a faked error: it skips, and any other `OSError` still fails) |
| `webapp/tests/test_public_text.py` | 0 | 5 passed |
| `webapp/tests/test_version_guard.py` | 0 | 8 passed |
| `webapp/tests/test_tinyapps_entry.py` | 0 | 9 passed |
| `webapp/tests/test_docs_currency.py` | 0 | 6 passed |
| `webapp/tests` whole, in one process (`-n 4` asked for, but the shared `.venv` has no pytest-xdist and nothing may be installed there) | 0 | 1,136 passed, 8 skipped (5 of them the opt-in wheel builds), 4 min 12 s |

The publish guard's script, taken from `publish.yml` and run locally
against PyPI in throwaway folders: the real `pyproject.toml` (core
1.0.0.dev0, not on PyPI) exits 1 with the step-12 error; the same logic
for `planlens==0.14.0` (on PyPI) exits 0 and for `planlens==9.9.9` exits 1;
no core requirement exits 0; `aec-agent-core>=1.0` exits 1 asking for an
exact pin. The step sits after the `dist/` check and before the upload.
The wheel run's `*.egg-info` folders were removed from the worktree.

## Release note (three lines)

- No behaviour changes: nothing moved, every answer and every tool is as before.
- New: `aec-agent-core`, the shared core the app split builds (empty for now, 1.0.0.dev0).
- From here the app installs with the core: install `aec-agent-core` first (from PyPI once it is released).
