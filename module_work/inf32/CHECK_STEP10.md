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
  boundary test's import counts);
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
- Without the `build` package the slow tests skip with the reason, unless
  `GEOTECH_REQUIRE_WHEEL_TEST=1`, which makes it a failure. The main
  "Tests" job has no `build`, so it skips them there.
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
   should know this before the next release word.**
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
5. `README.md`'s developer install (`pip install -e .`) and any session
   setup that installs master: `pip install -e packages/aec-agent-core`
   first (CI and this note say so; the README is a step-12 edit).

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
- **Step 12:** bump the core from 1.0.0.dev0 to 1.0.0 in its pyproject,
  `__version__` and the app's exact pin together (the fast wheel test
  fails until all three agree); add the `core-v*` tag trigger and a job that
  builds `packages/aec-agent-core` by name, with the same "only this
  package in dist/" check; publish the core before GSE; the doc changes of
  §5.

## 7. Left open

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

## Release note (three lines)

- No behaviour changes: nothing moved, every answer and every tool is as before.
- New: `aec-agent-core`, the shared core the app split builds (empty for now, 1.0.0.dev0).
- From here the app installs with the core: install `aec-agent-core` first (from PyPI once it is released).
