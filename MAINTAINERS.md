# Maintaining GeotechStaffEngineer

This is the short way in for an engineer who keeps the app up and has never
seen the repository. Read it first, then `RELEASING.md`, then section 0 of
`BACKLOG.md` (the open work) and section 0a-current of `HANDOFF.md` (the
current state and the release notes). `CLAUDE.md` holds the standing
instructions of the AI coding sessions; their long working notes up to
release 5.36.0 (about 2,700 lines) are in
`module_work/history/CLAUDE_before_trim_2026-11-17.md`: search them,
do not read them whole.

"The owner" below is the project's owner, who decides what the product does.

After the final release this public repository goes quiet: the work goes on
in the maintainer's own copy, and the owner means to bring it back here
later (section 9 says what to keep for that). A maintainer working from
their own copy builds wheels with a local version label for the platform
(`RELEASING.md` section 3) and never publishes to the public package index:
only the public GitHub repositories can publish (trusted publishing), and
releasing from them stays with the repository's owner.

## 1. What the repository holds

One repository builds the app, `geotech-staff-engineer`, and two smaller
packages under `packages/`. The reference library is a submodule.

| Folder or file | What it is |
|---|---|
| `bearing_capacity/`, `settlement/`, `axial_pile/`, `lateral_pile/`, `pile_group/`, `drilled_shaft/`, `wave_equation/`, `downdrag/`, `sheet_pile/`, `soe/`, `retaining_walls/`, `slope_stability/`, `seismic_geotech/`, `ground_improvement/`, `pavement_design/`, `reliability/`, `fem2d/` | The calculation modules. Each is a package with a `DESIGN.md`, a results dataclass and its own `tests/`. |
| `subsurface_characterization/`, `geo_project/`, `profile_figure/`, `dxf_import/`, `dxf_export/`, `calc_package/` | Site data (DIGGS, AGS4, CPT), the project set-up layer, profile figures, DXF in and out, calculation packages. |
| `*_agent/` (`opensees_agent`, `pystrata_agent`, `pynite_agent`, `section_props_agent`, `concrete_props_agent`, `seismic_signals_agent`, `liquepy_agent`, `gstools_agent`, `salib_agent`, `pystra_agent`) | Wrappers that give the agent an outside library through the same calling convention. |
| `geotech_common/` | What the modules share: `SoilProfile` and its adapters, units, `choices.py` (what a result chose), `numpy_compat.py`, plotting helpers. |
| `funhouse_agent/` | The agent: `dispatch.py` (how a tool call reaches a module), `adapters/` (one per module or reference), `deep/` (the agent's build, prompt and tools), the reference tools, `worked_examples.json`. `DESIGN.md` explains it. |
| `report_ingest/` | Reads a geotechnical report (PDF) into a record of explorations, logs and laboratory results. `README.md` explains it. |
| `webapp/` | The Streamlit app: `pages_entry.py` (both pages), `app.py` (the geotech page alone), `geotech_page.py`, the host entries (`foundry_entry.py`, `databricks_launcher.py`, `tinyapps_entry.py`), `diagnostics.py`. `README.md` explains running it. |
| `docs/` | Host guides (`FOUNDRY.md`, `DATABRICKS_INSTALL.md`, `SHAREPOINT_SETUP.md`), `REPORT_TABLES.md`, `examples/`, and `user_manual/` (the generated user manual and its build). |
| `validation_examples/` | The modules run against published worked examples (`INVENTORY.md`, `RESULTS.md`); part of the test suite. |
| `constraints/` | `foundry-numpy126.txt` and `foundry_requirements.py`: the install for an environment that holds numpy at 1.26.4. |
| `packages/aec-agent-core/` | The shared core, from upstream (section 2). |
| `packages/aec-document-review/` | Document Review on its own, from upstream; not a dependency of the app. |
| `geotech-references/` | The reference library, a submodule (section 3). |
| `module_work/` | Plans, per-item check files (`CHECK.md`), reviews and briefs: the history behind the code. Not shipped. One folder of it is in the test suite (`module_work/report_ingest_harness/tests`). |
| `tinyapps/` | The template that hosts Document Review alone on another platform; upstream's. |
| `foundry/`, `foundry_test_harness/` | Retired wrappers of an old deployment route; not shipped, kept because some wrapper tests still import them (BACKLOG INF-06). |
| `.github/workflows/` | `tests.yml` (the suite, eight jobs), `nightly.yml`, and the three publish workflows (`RELEASING.md`). |
| `BACKLOG.md`, `BACKLOG_DONE.md` | The one list of open work, and its archive (section 7). |
| `HANDOFF.md` | Current state (section 0a-current) and the history of every release. |
| `pyproject.toml`, `conftest.py` | The app's package and test configuration; the root conftest puts this tree's core first on the import path. |

Old notes and scratch at the root (`HANDOFF_2026-06-14.md`, `CONSOLIDATION_*.md`, `main.py`, `generate_all_pdfs.py`, `demo_groundhog_agent.ipynb`, `sample_pdfs/`, `reference_work/`) are history; nothing imports them.

The owner's private material (field sessions' raw files, saved runs) lives
only in gitignored folders and is not in the repository. Tests that need it
skip.

## 2. What a maintainer owns, and what arrives from upstream

**A maintainer owns:** the calculation modules and the wrappers, `geotech_common`,
the agent and its tools (`funhouse_agent`), the geotech page (`webapp`),
report ingest, the user manual, the validation examples, `BACKLOG.md`, and
the reference library (`geotech-references`, its own repository: its
functions, chapter text, catalogs and labelled builds; section 3). Public
releases of the app and the library stay with the owner (`RELEASING.md`,
the first of its three facts).

**Arriving from upstream as released wheels (the core with its source
folder, section 8), never edited in place:**

- `aec-agent-core` (`packages/aec-agent-core/`): model connections, vision
  and file tools, the figure helper and plotting box, the agent's guards and
  records, Word and Excel, conversation storage. The app pins it exactly
  (`aec-agent-core==` in `pyproject.toml`), and
  `webapp/tests/test_wheels.py` checks that the pin equals the version in
  `packages/aec-agent-core/pyproject.toml`.
- `aec-document-review` (`packages/aec-document-review/`). The app still
  carries its own copy of the Document Review page; those files are frozen
  by hash (`webapp/tests/test_review_freeze.py`).
- `planlens`, the document layer, from its own repository
  (`planlens>=` in `pyproject.toml`).

These three stay upstream after this repository goes quiet: upstream keeps
fixing and releasing them on the public package index, and the owner passes
each new wheel on (section 8).

Modules that moved into the core keep their old import names: `funhouse_agent.vision_tools`
is the same module object as `aec_agent_core.vision_tools` (the list is
`CORE_ALIASES` in `funhouse_agent/__init__.py` and `webapp/__init__.py`). So
a file you expect under `funhouse_agent/` or `webapp/` may live in the core.

**A need upstream** goes to the owner as a request with the failing case:
the call or the input, what came back, what was expected, and the installed
versions (the diagnostics report's versions line). A small test with
made-up values is the best form. Where the app can get what it needs through
the core's hooks (`aec_agent_core/hooks.py`; the app fills them in
`funhouse_agent/__init__.py` and `webapp/__init__.py`), do it there, in the
app's own code.

## 3. The reference library

`geotech-references` holds the printed manuals the modules cite (NAVFAC DM 7,
the FHWA GEC series, UFC, EM and others) as callable functions, chapter
text, and figure and table catalogs. The references helper and the
reference adapters (`funhouse_agent/adapters/dm7_adapter.py`,
`gec6_adapter.py` and the like) read it. It is
its own repository, checked out as a submodule at `geotech-references/`,
with its own README and CHANGELOG. The same maintainer keeps it, and its
own `MAINTAINERS.md` and `RELEASING.md` (at the root of that repository,
from its release 1.8.0; an older submodule commit lacks them)
cover its layout, its tests, how content is added and checked, and its
releases; this section covers only how the app uses and pins it. Its open
work is in this repository's `BACKLOG.md` (`REF-` ids). Its wheel carries no PDFs; the PDFs are
in the repository's `docs/` and a host provides them separately
(`docs/FOUNDRY.md` section 3d, `webapp/README.md` "Reference PDFs").

The app pins it twice, and the two move together:

- the **submodule commit**: what the tests and the manual build run against;
- **`geotech-references>=X.Y.Z`** in `pyproject.toml`: what an install takes.

When the library is released: move the submodule to its tag
(`git -C geotech-references fetch origin --tags`, then
`git -C geotech-references checkout vX.Y.Z` and `git add geotech-references`),
raise the pin, rebuild the manual (`RELEASING.md` step 7), run the suite.

From a git clone, fetch it with `git submodule update --init geotech-references`.
From source zips, extract the library's zip into the empty
`geotech-references/` folder (a `git archive` of the app leaves that folder
empty).

## 4. Setting up and running the tests

Python 3.11 or newer (the hosts run 3.12; CI runs 3.12 and 3.13). From the
repository root:

```bash
python -m venv .venv
. .venv/bin/activate                 # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e packages/aec-agent-core   # the core first: the app pins it exactly
pip install -e .
pip install -e geotech-references/
pip install pytest pytest-xdist
```

**The import path.** Run pytest from the repository root. pytest puts the
root on the path, and the root `conftest.py` puts `packages/aec-agent-core`
first, so the tests import this tree's app and core even where another copy
is installed. The reference library comes from the install. In an
environment that holds another copy of any of the three, set `PYTHONPATH`
to absolute paths in this order: `<root>/geotech-references`, `<root>`,
`<root>/packages/aec-agent-core` (joined by `:` on Linux, `;` on Windows),
and check once where each comes from:

```bash
python -c "import webapp, aec_agent_core, geotech_references as g; print(webapp.__file__, aec_agent_core.__file__, g.__file__)"
```

**The guards** (fast, run before every commit):

- `webapp/tests/test_backlog.py`: BACKLOG ids are unique, tiered and never
  reused; every id cited exists; every `[owner]` item has a card.
- `webapp/tests/test_public_text.py`: no guarded name (stored as hashes) and
  no real deployment host in any tracked file or shipped file. It reads
  `git ls-files`, so `git add` a new file before running it.
- `webapp/tests/test_docs_currency.py`: `CLAUDE.md`, `HANDOFF.md`
  section 0a-current and the install guide's history name the version in
  `pyproject.toml`, and `CLAUDE.md` stays under 40,000 bytes.

```bash
python -m pytest -q webapp/tests/test_backlog.py webapp/tests/test_public_text.py webapp/tests/test_docs_currency.py
```

These take about 20 seconds. Other tests a change often meets:
`webapp/tests/test_page_fingerprints.py` (what the agent is handed: prompts,
tool descriptions and schemas, pinned as hashes),
`webapp/tests/test_user_manual_current.py` (the manual is what its build
makes), `funhouse_agent/tests/test_defaults_used.py` (a result's
`defaults_used`).

**For a small change** run the module's own tests, every test file that
names what you changed, the fingerprints and the guards, for example:

```bash
python -m pytest -q bearing_capacity
python -m pytest -q funhouse_agent/tests/test_defaults_used.py -k bearing_capacity
python -m pytest -q webapp/tests/test_page_fingerprints.py
```

**The full suite** is `python -m pytest -n auto --dist worksteal -q` (test
paths are in `pyproject.toml`). On CI it takes 33 to 45 minutes a job on
four workers; in one process, 50 to 100 minutes. Release 5.35.0's gate was
25,877 passed, 84 skipped, 0 failed on Python 3.12. Judge a run by pytest's
exit code, never through a pipe that hides it.

CI (`.github/workflows/tests.yml`) runs eight jobs: the suite on Python 3.12
and 3.13; a subset of the suite (the numerical modules and wrappers the job
names, slow tests left out) in an environment holding numpy at 1.26.4
(installed with `constraints/foundry-numpy126.txt`); the wheel checks; and
the two upstream packages alone (`core-only` and `document-review`, each on
Python 3.11 and 3.12). It runs on pushes to `master` and to `ci/<name>`
branches, on pull requests to master, and by hand from the Actions tab. The
full suite runs there, not on a laptop: push a feature branch as
`ci/<name>` (`git push origin <branch>:ci/<name>`) and merge only on a green
run. A commit that changes only notes (`*.md`, `module_work/`, `docs/`
other than the user manual) starts no run. A copy kept anywhere else has no
such runs: there, run the full suite itself.

The code must also run on numpy 1.26: take a name numpy 2
added from `geotech_common.numpy_compat` (`trapezoid`, never
`np.trapezoid`); `geotech_common/tests/test_numpy_compat.py` scans for the
rest.

## 5. The owner's rules

Each is binding; where it is written follows the dash.

- **Print only.** Every formula, coefficient, curve and default traces to a
  printed page, cited by manual, edition and page, equation or table; no
  value from memory. A gap in the print is said, not filled.
  - `module_work/cloud_briefs/COMMON_MODULES.md` ("The owner's rules").
- **Open methods.** Where references differ, the module offers each as a
  cited option and states its default and why; never one silent basis.
  - The same file; `geotech_common/choices.py`.
- **Judgement stays with the agent, and is stated.** Modules stay general;
  what a run chose is in its `choices` and `defaults_used`, and the agent
  lists its judgement calls, each with its source or "no reference found".
  - `geotech_common/choices.py`; `funhouse_agent/judgment_check.py`.
- **Made-up values in tests.** Never a value from a real project's
  documents; no names of clients, projects, posts, departments or people,
  and no identifiers of a hosted platform (workspace, dataset or model ids,
  internal host names); the guard catches only some of these. The
  repository is public.
  - `COMMON_MODULES.md` ("Public repo"); `webapp/tests/test_public_text.py`.
- **One backlog file, stable ids.** Every to-do, follow-up and question goes
  into `BACKLOG.md` with the next free id; no list anywhere else; an id is
  never reused.
  - The top of `BACKLOG.md`; `webapp/tests/test_backlog.py`.
- **No edge-case chasing.** A follow-up becomes an item only if it changes
  a saved report's output; otherwise it is one "noted, not fixed" line.
  - `BACKLOG.md`, "How every session uses this file", item 2.
- **The generated manual is never edited by hand.** Change the words in
  `docs/user_manual/manual_content.py` or the tool's `METHOD_INFO`, then
  rebuild.
  - `docs/user_manual/README.md` lists `manual.html` and `toc.json` as build
    output; `webapp/tests/test_user_manual_current.py` fails on a hand edit.
- **Figures.** No ink that carries no data; plans to scale; elevation up a
  profile; kPa and kN; units and the coordinate system named.
  - The figure checklist in `packages/aec-agent-core/aec_agent_core/deep/figure_helper.py`
    (upstream; a change there is a request).

## 6. How a calculation module is laid out: `bearing_capacity`

| Part | Where | What it does |
|---|---|---|
| Theory and conventions | `bearing_capacity/DESIGN.md` | Read before changing the code. |
| Inputs | `footing.py` (`Footing`), `soil_profile.py` (`SoilLayer`, `BearingSoilProfile`) | SI units: m, kPa, kN, degrees. |
| Computation | `factors.py`, `capacity.py` (`BearingCapacityAnalysis.compute()`) | The printed equations, each cited. No import from another calculation module (`geotech_common` is the exception). |
| Result | `results.py` | A dataclass with `summary()` and `to_dict()`, and `choices`: one entry per choice (`what`, `chosen`, `how` = given, default, module or assumed, `basis`, `alternatives`). |
| Calc package steps | `calc_steps.py` | What a calculation package prints; `calc_package.generate_calc_package(module="bearing_capacity", ...)` loads these steps (`_IMPORT_MAP` in `calc_package/__init__.py`). |
| The adapter | `funhouse_agent/adapters/bearing_capacity.py` | Flat dict in, dict out. `_ANALYSIS_VALID_PARAMS` (an unknown name is refused), aliases, `METHOD_REGISTRY` (method name to function) and `METHOD_INFO`: what the agent reads, each parameter's type, default, allowed values and a description that says what the code does and cites the print. `declare_applies_when` says when a default applies; `declare_when_absent` what a left-out input stands for. |
| `defaults_used` | added by `funhouse_agent/dispatch.py` | The declared defaults a call left in force, on every result; it never contradicts `choices`. |
| Registration | `MODULE_REGISTRY` in `funhouse_agent/adapters/__init__.py` | The adapter's path and a one-line brief. |
| The manual row | `docs/user_manual/manual_content.py` | `MODULE_ORDER`, `MODULE_TITLES`, `HEADLINE_METHOD`, `WORKED` (the worked example) and `MODULE_NARRATIVE` (problems, methods with citations, limits). The parameter tables are generated from `METHOD_INFO`. |
| Tests against print | `bearing_capacity/tests/` | For example `test_meyerhof_linear_depth.py`: the docstring names UFC 3-220-20 Table 5-6, p. 294, and says the values are made up and worked by hand. |

**Checklist for adding or changing a module:**

1. Read its `DESIGN.md` and the printed source; note manual, edition and page.
2. Where sources differ, add an option (a parameter with allowed values),
   keep or state the default, and report the choice in `choices` with its
   basis and alternatives.
3. Adapter: the parameter in the valid list and in `METHOD_INFO` with its
   default, allowed values and a true description.
4. A new module: a new top folder goes into `pyproject.toml` twice
   (`[tool.setuptools.packages.find] include` and `testpaths`), into
   `MODULE_REGISTRY`, and into `manual_content.py`. A wheel without the
   `include` line builds and installs and simply lacks the package.
5. Tests with made-up values worked by hand from the printed equation, the
   docstring citing the page. Run the module's tests,
   `test_defaults_used.py -k <module>` and the fingerprints. If a tool
   description or prompt changed on purpose, re-record in a commit of its
   own: `python -m webapp.tests.page_builds record`.
6. For a result key, parameter or function you add, rename or remove,
   search the whole repository for consumers (`webapp/`, `funhouse_agent/`,
   `calc_package/`, `report_ingest/`, `docs/`, `validation_examples/`, JSON
   fixtures) and run their tests.
7. Rebuild the manual with the submodule at the pinned commit:
   `python module_work/doc10/check_manual.py --rebuild --write`; commit
   `manual.html` and `toc.json`.
8. If an answer moves, write the release note (what moves, for which
   inputs, with a before-and-after number) for `HANDOFF.md` section
   0a-current, and close the BACKLOG item with its commit.

The practice here is a short `module_work/<item>/CHECK.md` per item: what
the print says, what changed, numbers before and after, the tests run.

## 7. Where the open work and the history are

**`BACKLOG.md`** is the one list. An id is `PREFIX-N`; the prefix names the
area and its section:

| Prefix | Area | Prefix | Area |
|---|---|---|---|
| `OWN` | the owner's decisions and actions (section 1) | `DR` | Document Review (6) |
| `RI` | report ingest (2) | `VIS` | vision (7) |
| `INF` | hosts, infrastructure, releases (3, 10) | `PL` | planlens (8) |
| `GM` | the calculation modules (4) | `EV` | measurement and evals (3, 9) |
| `REF` | the reference library (5) | `TA` | the Tiny Apps host (10) |
| `LT` | long-term programmes (11) | `DOC` | stale docs (12) |

Tiers: **Now** (the current focus, named in section 0), **Next**,
**Later**, **Parked** (kept as a note, not planned; each says why).
`[owner]` marks an item that needs the owner. `*Src:*` names the document
with the detail. The next free id:

```bash
python module_work/backlog_dashboard/sweep_done.py --next GM
```

Finishing an item: delete its line and add it under "Done since the last
sweep" with the date and the commit. One command does both (`GM-0000` is a
made-up id: put the item's own id in its place):

```bash
python module_work/backlog_dashboard/close_item.py GM-0000 "2026-11-20 (abc1234): what was done"
```

`BACKLOG.md` keeps Windows line endings (CRLF); edit it by exact text and
keep them. A release moves the Done list to `BACKLOG_DONE.md`
(`RELEASING.md` step 6). An `[owner]` item needs a card in
`module_work/backlog_dashboard/context/` (the guard checks it). The link at
the top of `BACKLOG.md` is the owner's own page for answering; edit
`BACKLOG.md` itself.

**The history:** `HANDOFF.md` (section 0a-current is now; every release
before it is below, with its release notes by module), `BACKLOG_DONE.md`
(every closed item with its commit), `module_work/` (the check files, plans
and reviews; `module_work/harness_theory/` explains how the agents work),
and the commit messages, which say what changed and why.

## 8. Adopting a new upstream wheel

`RELEASING.md` section 4 has the steps. In short: the core arrives as its
wheel and its source folder; replace `packages/aec-agent-core/` whole with
the release's copy, set `aec-agent-core==X.Y.Z` in `pyproject.toml`, and
run `webapp/tests/test_wheels.py`. A new planlens: install it and run
`webapp/tests/test_page_fingerprints.py`, which records the document tools
by planlens version; record the new one in a commit of its own. Then the
full suite. Never edit the upstream files to make a test pass.

## 9. Bringing the work back

After the final release this public repository goes quiet; it is renewed
later from the maintainer's copy. Keep four things for that:

1. **A change log** (`LOCAL_CHANGES.md` at the root, newest first, the first
   entry naming the release the copy started from): one entry per change,
   with what, why, the printed source and page where a formula, curve or
   default moved, the tests added, and the upstream versions it ran against
   (planlens, aec-agent-core, aec-document-review: `pip show <name>`), which
   a renewal pins against.
2. **Version numbers above the final release**, with the local label
   (`RELEASING.md` section 3); publish nothing to the public index. The
   renewal chooses the next public number, and the owner releases it.
3. **The one backlog file, current**, with its ids (section 7).
4. **An export on request** of the source with its tests, as a zip
   (`git -c core.autocrlf=false archive --format=zip --prefix=gse/ -o gse-<date>.zip HEAD`)
   or a git bundle (`git bundle create gse-<date>.bundle --all`); neither
   carries the library, so export it from its own copy the same way. Keep
   the copy in git (the bundle and `test_public_text.py`, which reads
   `git ls-files`, need it), and make its first commit the release's source
   exactly as received, so that every later change is a diff against the
   release. The export comes back public, so the rules of section 5 and
   `test_public_text.py` hold in the copy too. `test_public_text.py` catches
   the words this repository already guards, not new ones, so read the
   change log and the diff for new identifiers as well before an export.

## 10. How the commands here were checked

Each command was run once on 2026-11-17 against master `6941d9d` (version
5.35.0) on a Windows laptop with Python 3.14, by exit code, unless it is
listed as not run. Files a command rewrote were restored afterwards.

- The import-path check (the three files of this tree), the guards (22
  passed in about 20 seconds), `pytest bearing_capacity` (395 passed in
  under a second), `test_defaults_used.py -k bearing_capacity` (23 passed),
  `test_page_fingerprints.py` (13 passed) and `page_builds record` (exit 0;
  the recorded file unchanged).
- `sweep_done.py --next GM` (printed `GM-355`), `close_item.py`, and
  `check_manual.py --rebuild --write` (exit 0).
- Section 9, on master `17314a3` committed into a new repository: the zip
  export (37 MB, Unix line endings) and the bundle (32 MB), checked with
  `git bundle verify` and a clone from it; `pip show planlens aec-agent-core
  aec-document-review` (exit 0; it warns for one not installed).

Not run here:

- The set-up block of section 4: the laptop's shared environment already
  holds these installs. CI's install step runs the same `pip install` lines
  on every push that changes code.
- The full suite: 50 to 100 minutes in one process; it runs on CI.
- `git submodule update --init geotech-references`: it would clone the
  library's history (about 1.4 GB) again; CI's checkout does the same.
- The submodule move of section 3: a release step (last done in `0187efb`
  on `release/5.36.0`, the library at 1.8.0).
