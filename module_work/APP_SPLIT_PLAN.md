# Splitting Document Review from GeotechStaffEngineer (plan, 2026-10-20; rewritten 2026-10-21)

BACKLOG: **INF-26** (this plan; its steps are INF-30 to INF-35 and the
planlens item of step 9), **INF-48** (this rewrite), the owner's open
decisions **OWN-72** (the core's name; CfA's yes for a source folder in the
GHE repo) and **OWN-78** (the testers' notice and SharePoint folder), and
the decided **OWN-74**, **OWN-75** (release the held train first),
**OWN-76** and **OWN-77** (the figure helper: option (b); its follow-up is
**DR-35**). **RI-34** (every geotech-report question goes to report_ingest)
is done. **INF-25** is GSE on Foundry. Steps 1-5 are built; nothing else in
this document is. It changes no code.

The owner's words:
- 2026-10-20 (Backlog Desk): "I want to divorce the two apps.
  Document-Review and GeotechStaffEngineer will now be separate apps ...
  because people will use it that aren't geotech engineers ... the
  production layout would look like: GeotechStaffEngineer running in
  Palantir Foundry as an app, and DE_DocReview being a wheel that I pull
  into TinyApps." The reason: unlimited Opus 5.5 and Sonnet 5.5 on Foundry
  for the owner, GSE's only user.
- 2026-10-21 (Backlog Desk): "for doc review, the whole code can live in
  the GHE repository. Nothing needs to be a package"; "I def don't want the
  same base code in two places"; "Leaning towards the small core package";
  and, on the layout below: "Ok, sounds good. You can proceed."
- 2026-10-21, later the same day (Backlog Desk): on the figure helper
  (OWN-77), option (b), "Follow what we talked about with the planlens
  fold"; on releasing the held train first (OWN-75), release it, and "If
  it's too late, no big deal, just go with hold. Don't expend big effort to
  revert back."

## In one page (for the owner)

**2026-10-21: rewritten for your decisions of that day (OWN-74, OWN-76),
then updated for your answers to OWN-75 and OWN-77 the same day.** The
earlier version planned a `de-docreview` wheel and a shared core that held
the chat page.

**The layout.**
- **Document Review:** private source in the GHE wrapper repo. No package,
  no wheel, never on PyPI.
- **planlens** (public, as today) takes the document tools that only wrap
  it: opening documents, markups and their check, `find_like`. It gets its
  own release.
- **A thin core package** (public, on PyPI like planlens; its name is yours
  to pick, OWN-72): the model connections, the vision calls, the figure
  helper and plotting code with the nine geotech worked examples for now
  (OWN-77 (b)), and the guards and records both agents run. Tiny Apps gets
  it from Nexus, the way it gets planlens.
- **GeotechStaffEngineer** (public, on PyPI as today) runs on Foundry.
- **Each app keeps its own chat page.** The page is copied once, at the
  split. After that, each changes with its own users' feedback. It is the
  only code that exists twice.

**What you get.**
- Testers install no geotechnical code: about a dozen packages fewer, and
  nothing tied to the geotech pins.
- Document Review's code is private from the copy on. What was published
  before stays published.
- One copy of everything else. A fix to a tool, the vision calls or a model
  connection lands once, for both apps.
- GSE on Foundry loses nothing.

**What it costs.**
- The core is thinner than before but not tiny: about 38,000 lines. Your
  three named parts are about 21,000 of them. The other 17,000 are what
  both agents run under the page: guards and limits, the file, Word and
  Excel tools, the run record and its checks, and the suite's framework.
  Leaving them out of the core would put them in two places.
- The chat page, about 12,900 lines, is in two places from the copy on, by
  design.
- Three releases come before testers move: the held train (your timing,
  before the first file leaves this repository), planlens with the
  document tools, and the core's first.
- Document Review's private copy needs its own tests and CI.

**What is still open.**
- **OWN-72:** the core's name (before step 10), and CfA's yes to running
  Document Review from a source folder in the GHE repo (before step 14).
  Until CfA answers, testers keep today's two-page package from Nexus.
- **OWN-78:** the testers' notice and SharePoint folder (before step 14).
- **New (D10, section 7):** where the private copy is worked on day to day.
  Either a private repository you carry to GHE, or GHE itself (before
  step 13).
- **Decided the same day:** the figure helper and plotting code go into the
  core, with the nine geotech worked examples for now; Document Review's
  own examples follow the split (OWN-77 (b), DR-35). The held train is
  released before the first file leaves this repository, or held if that
  is no longer practical (OWN-75).

**The steps.**
- Steps 1-5 are done.
- Steps 6-8 stand. Step 7 now prepares the copy.
- Then the held train is released (OWN-75), or held if that is no longer
  practical.
- Steps 9-16 are new:
  - planlens takes the document tools (9);
  - the core is built and filled (10, 11) and released (12);
  - Document Review gets its private home (13) and its Tiny Apps wrapper
    (14);
  - GSE lets go of the review page (15), and the old import paths go (16).

## 1. The target layout

### 1.1 What there will be

| What (install name / import name) | What it is | Who uses it, where | Depends on | How it is installed | Public |
|---|---|---|---|---|---|
| `planlens` / `planlens` | The document layer as today: reading PDFs and DXFs, page maps, markups, render budgets. From step 9 it also holds the document tools that only wrap it (2.6). It still calls no model: when a tool needs a look, the host hands it a look function. | Both apps | numpy, PyMuPDF, ezdxf, opencv-headless, rapidfuzz (unchanged) | PyPI (Nexus mirrors it) | yes |
| `geotech-references` / `geotech_references` | The cited reference library and chart catalogs | GSE only | numpy | PyPI | yes |
| **The core** (working name `docreview-core` / `docreview_core`, OWN-72; NEW) | What both agents run below the chat page: the model connections, the vision calls and the file tools beside them, the figure helper with its plotting box, chart stack and the nine geotech worked examples for now (OWN-77 (b)), the agent's guards, limits and kept work, Word and Excel, the run record and its checks, the switches, the suite's framework (2.3). It has no Streamlit page and nothing that encodes a Department deployment. | Both apps, every host | planlens, deepagents/langchain/langgraph (today's caps), openai <3, langchain-openai, matplotlib, plotly, python-docx, markdown-it-py, openpyxl, numpy, scipy (the plotting box lets code import it); an `[anthropic]` extra for langchain-anthropic | PyPI → Nexus; each app pins an exact version | yes |
| `geotech-staff-engineer` / `funhouse_agent`, `webapp`, the module packages | The geotech app: 33 modules, dispatch, the geotech agent and its helpers (calc, references, setup, reviewers), the chart lines it hands the figure helper, report_ingest, its own chat page, the Foundry and Funhouse entries | The owner on Foundry (Funhouse as backup) | the core with `[anthropic]` (exact), planlens, geotech-references, the analysis back ends as today | PyPI (Foundry's Libraries panel) or uploaded wheels | yes |
| **Document Review** (a source folder in the GHE wrapper repo, for example `docreview/`; NEW) | The Document Review app: its own chat page, its agent builds (today's agent, `lean` and `minimal` behind their switches), its prompt and orientation, the review-only tools (findings, geometry, sweep, digest), the Tiny Apps host files (settings, sign-in header, Graph SharePoint), its 38 suite tasks and its tests (2.2) | Testers on Tiny Apps; suite runs on Foundry | the core (exact), planlens (exact), streamlit and the fleet pins | Runs from the repo folder (needs CfA's yes, 1.4) | **no**: private from the copy (step 13) |

How they depend on each other:
- planlens depends on nothing of ours.
- The core depends on planlens.
- GSE depends on the core, planlens and geotech-references.
- Document Review depends on the core and planlens.

Nothing imports upward or sideways: the core imports neither app, and
neither app imports the other. Tests hold each line (2.7, section 5).

### 1.2 The rule for where code goes

1. **planlens:** code that reads or writes a document and calls no model.
   Where it needs a look, the host passes in a function, the way
   `ReviewToolkit` already takes `resolve_source` today.
2. **The core:** everything both agents run below the chat page. No
   Streamlit page, and nothing that names or encodes a Department deployment
   (a Key Vault naming scheme, the sign-in header, a host address). Those
   stay with the app that uses them.
3. **Each app's chat page:** the Streamlit page and the conversation
   machinery under it: turns, uploads, storage, identity, the panels and the
   entry. Copied at step 13; each app's own from then on.
4. **Document Review's source:** what only the review page uses.
5. **GSE:** everything else.

Judgment calls under these rules, named so the owner can overrule:
- **The Prompter connection goes in the core** (the owner's list). It is an
  OpenAI-compatible connection that takes its address, model, key and
  certificate from the caller. Picking a tester's key from Key Vault by the
  sign-in name stays in Document Review.
- **The Funhouse bridge** (`deep/databricks_bridge.py`) goes in the core,
  because the Foundry connection is built on it. It takes the Funhouse
  client from the caller and names no host.
- **SharePoint conversation storage and the SharePoint agent tools go with
  each chat page** (copied). They are part of how a page keeps its
  conversations, and each app's hosts differ: Graph on Tiny Apps, the
  Funhouse client on Funhouse, none on Foundry. At about 2,500 lines they
  are the largest copied part after the page itself. The other choice is
  the core, with each app handing in its storage back end.
- **The run record** (`activity_log.py`) goes in the core. One record
  format keeps the trace tools, the Harness Map and past TRACE_REVIEWs
  working for both apps.
- **The suite's framework goes in the core.** Each app hands it a way to
  run a turn on its page, so GSE keeps measuring its four report tasks. The
  other choice: the framework goes with Document Review and GSE's four tasks
  go to report_ingest's scoring. That takes about 2,200 lines out of the
  core.

### 1.3 What stays the same for every host

So that no setting, Key Vault secret, conversation or folder is lost:
- **Setting names** stay: `PROMPTER_*`, `PROMPTER-API-KEY--<TESTER>`,
  `GRAPH_*`, `SHAREPOINT_SITE_URL`, `KV_NAME`, `DEV_IDENTITY` and every
  `GEOTECH_*` setting and switch. Renaming them is a later, separate choice
  (old names still read).
- **Folders** stay: the data root (`GEOTECH_WEBAPP_DATA`, else
  `~/.geotech_webapp`), `users/<identity>/document_review` on Tiny Apps,
  `pages/document_review` single-user, and the SharePoint base folder
  `Shared Documents/GeotechStaffEngineer` unless the owner names another
  (OWN-78). Document Review's copy of the page keeps the same layout, so
  conversations already on the test machine show up in it.
- **The activity record** keeps its event names and fields
  (`activity.jsonl`, `run.json`), so the trace tools, the Harness Map and
  past TRACE_REVIEWs still read new runs from either app.

### 1.4 How Tiny Apps gets Document Review

Tiny Apps installs packages from the Nexus mirror only (`tinyapps/TINYAPPS.md`,
"Packages"). After step 14 the GHE wrapper repo holds:
- `app.py`, which calls Document Review's entry;
- the Document Review source folder;
- `packages.txt`: the core and planlens at exact versions from Nexus, the
  fleet pins with pandas 3.0.2 restored, the Key Vault libraries;
- `run.sh` and `.env.example`, as today.

The core and planlens come through Nexus like any public package (the route
5.26.0 took to the test machine). Document Review is not installed at all:
it runs from the folder.

| Route | The wrapper repo holds | Needs from CfA | Notes |
|---|---|---|---|
| **A. Source folder** (the owner's choice) | The folder and `packages.txt` | Yes: question 6a | The usual Tiny Apps shape. CfA's reviewers can read the code. |
| B. One wheel of ours in the repo | A wheel built from the same folder, installed by path from `packages.txt` | Yes: question 6b | The same code, zipped. Only if CfA would rather install a file than run a folder. Never uploaded anywhere. |
| C. A hosted repository on CfA's Nexus | Only `packages.txt` | Yes: question 6c | Only if CfA requires every package through Nexus. |

PyPI is not a route: the owner keeps Document Review private.

**What to ask CfA** (sent on the desk with OWN-72, 2026-10-21, replacing
question 6 in `tinyapps/TINYAPPS.md`):
- **6a.** May our app's source folder live in the GHE wrapper repo and run
  from there, with every installed package coming from Nexus? Our shared
  package is public on PyPI, like planlens.
- **6b.** If you would rather install a file than run a folder, may the repo
  carry one small pure-Python wheel of ours, installed by path from
  `packages.txt`?
- **6c.** Does your Nexus have a hosted repository where a team publishes
  its own package?
- **6d.** Are the fleet pins our tree must accept still streamlit 1.62.0,
  pandas 3.0.2, PyJWT 2.13.0, msal 1.37.0, azure-identity 1.21.0 and
  azure-keyvault-secrets 4.9.0?

The other open CfA questions (keys, SharePoint registration, the published
address, the identity header, the image-capable deployment, FIPS, the budget
response) are unchanged. **Until CfA answers, testers stay on GSE's two-page
package from Nexus** (5.33.0 or the next release). The geotech page leaves it
only at step 15, after Document Review's route works.

### 1.5 Where Document Review's private source is worked on (new, D10)

The GHE repo is the Department's. Sessions on the laptop and a CI run need
a copy they can edit and test. Two ways:
- **(a) A private repository on the owner's GitHub account** (like the PM
  bot's) is the working copy, with its own CI. At each tester release the
  owner copies the folder into the GHE wrapper repo, pushes, and asks for a
  sync, as with any wrapper change today.
- **(b) The GHE repo itself**, if the laptop can push to it and CI can run
  there.

Either way the GHE copy is the one testers run. While a change needs an
unreleased core, the working copy installs the core from this repository's
checkout, the way GSE uses planlens today; its CI installs the released core
at the pinned version. **Recommendation: (a), unless (b) is available.**
Needed before step 13.

### 1.6 How GSE gets to Foundry and Funhouse

- **Foundry (production for GSE):** `pip install geotech-staff-engineer` in
  the Libraries panel brings the core and planlens from PyPI. Nothing new for
  the owner to type. The INF-25 test-wheel route uploads four wheels in one
  command: GSE, the core, planlens and geotech-references (OWN-80: a Foundry
  Artifacts repository for them). The 2-line `app.py`
  (`webapp.foundry_entry`) never changes.
- **Funhouse/Databricks (backup for GSE):** `run_on_databricks` stays in
  GSE and stays importable as `webapp.databricks_launcher`, so the owner's
  notebook cell keeps working. Document Review is Tiny Apps only (owner,
  2026-10-20), so the launcher boots the geotech page alone from step 15.
- **Document Review's suite runs on Foundry:** its source folder goes into
  the Foundry workspace (as the reference PDFs do, OWN-80) beside the core
  and planlens wheels. The brief for suite runs says how. Nothing goes to
  PyPI.

## 2. Inventory by file

### 2.1 How this was found, and the totals

**2026-10-20 (kept as history).** An import graph of every first-party
import in `funhouse_agent/`, `webapp/`, `calc_package/` and
`report_ingest/`, imports inside functions included, plus two probes:
importing the review agent at ac94ba5 loaded 37 `funhouse_agent` modules,
twelve of them geotech; building the Document Review agent offline at
8ca1694 loaded 51, plus 20 modules from four geotech packages.

**Re-measured 2026-10-21 at 28f7946, after steps 1-5, for the new layout:**
- **Lines:** every non-test Python file in `funhouse_agent/` and `webapp/`
  (206 files, 101,493 lines of text), plus the chart stack and the worked
  examples in `profile_figure/` and `geotech_common/`, and the upload
  component's HTML.
- **Imports:** each file's imports read with Python's own parser, at any
  depth (inside functions too).
- **Split files:** counted by function boundaries (each `def` and `class`
  from its first line to its last); the line ranges are in 2.5 and 2.6.
- **Couplings:** the boundary test's list
  (`webapp/tests/test_review_boundary.py`), which builds and runs each of
  the review page's three builds and the geotech page with the geotech
  packages blocked and fails if the list is out of date. It passed on
  2026-10-21: 13 couplings left (7 hard, 6 soft), from 18 at step 1.
- The counting scripts were scratch work and are not kept; the method above
  is enough to redo it.

**Totals by destination.**

| Destination | Lines | What |
|---|---|---|
| **planlens** (step 9): the tools you named, and what they need | 4,319 | Opening documents, markups and their check, `find_like`, the page and image frame conversions, page numbers (2.6) |
| planlens, proposed: the same kind | 862 | `measure` and `log_grid`, the scale-label read, DXF notes (2.6) |
| **The core** (steps 10-11) | about 37,900 | The eight groups below, plus the assembly both agents share, which step 8 extracts and measures |
| - model connections | 3,869 | Your list |
| - vision calls and the file tools beside them | 9,335 | Your list |
| - the figure helper, plotting box, chart stack and worked examples | about 7,900 | Your list: the helper, box and examples (about 4,600; OWN-77 (b)) and the chart stack behind `plot_data` (about 3,300) |
| - guards, limits, kept work, `calculate`, file handling, switches | 5,561 | What both agents also run |
| - Word and Excel | 1,753 | |
| - the run record and its checks | 6,457 | |
| - the suite's framework | 2,191 | |
| - the document tools' binding | 848 | What stays of `document_tools.py` after step 9 |
| **Each app's chat page** (copied at step 13) | 12,867 | 12,722 lines of Python and 145 of HTML (2.9) |
| Host files, Document Review only | 664 | Tiny Apps settings, Graph SharePoint, the Tiny Apps entry |
| Host files, GSE only | 2,039 | The Foundry entry, the Databricks launcher, the reference fetcher |
| **Document Review only** (step 13) | about 7,700 | 6,301 in whole files, about 270 of review build (step 8), about 150 of profile and orientation, about 950 of its 38 tasks (2.2) |
| **GSE** | 39,865 here, plus every module package | `funhouse_agent/` and `webapp/` files that stay, the 72 adapter files (23,271 lines) among them (2.4) |

The old plan's core was about 46,000 lines with the chat page in it. Of the
new core, the owner's three named parts are about 21,100 lines and the rest
about 16,800.

### 2.2 To Document Review's source (step 13)

| File | Lines | Notes |
|---|---|---|
| `funhouse_agent/deep/review_agent.py` | 543 | The `lean` build; `REVIEW_TOOLS`, `REVIEW_DESCRIPTIONS`, the reading helper |
| `funhouse_agent/deep/minimal_agent.py` | 303 | The `minimal` build (kept for measurement) |
| `funhouse_agent/deep/sweep.py` | 259 | `GEOTECH_REVIEW_SWEEP` |
| `funhouse_agent/deep/findings_tools.py`, `review_findings.py` | 599 + 755 | `GEOTECH_REVIEW_FINDINGS`: the finding format and ledger |
| `funhouse_agent/deep/geometry_tools.py` | 1,144 | `GEOTECH_REVIEW_GEOMETRY` (`planlens.ir` constructs) |
| `funhouse_agent/deep/digest_tools.py`, `review_digest/` (four files) | 467 + 2,004 | `GEOTECH_REVIEW_DIGEST` and the digest layer |
| `funhouse_agent/review_prompt.py` | 227 | The review texts (step 3) |
| `funhouse_agent/review_build.py` | about 270 | Today's review build, extracted from `deep/agent.py` at step 8 |
| From `webapp/profiles.py` | about 150 | `DOCUMENT_REVIEW`, the orientation requests and their helpers |
| From `funhouse_agent/review_eval/tasks.py` | about 950 | The 38 review tasks of 42 (meck, set, ufc, calc, fixture, produce, scale-plan-distance) |
| `tinyapps/TINYAPPS.md`, `tinyapps/wrapper_repo/` | | Its deployment notes and wrapper template. The gitignored `tinyapps/reference/` stays local, never committed. |
| Its tests | | The six `deep/tests/test_review_*` files, the findings tests, `review_digest/tests`, the review half of `webapp/tests/test_app_profiles.py`, `test_tinyapps_entry.py`, `test_orientation_overview.py`, the review fingerprints |

With its page copy (2.9) and its host files (664 lines).

### 2.3 To the core (steps 10-11)

The file names stay. Only the prefix changes: `funhouse_agent.X` and
`webapp.X` both become `<core>.X` (no two names clash). That keeps every
rename mechanical and easy to map. Line counts in brackets.

**Model connections** (3,869): `webapp/engine_config.py` (901),
`tinyapps_engine.py` (666; its Key Vault lookup and the per-tester key by
sign-in name are handed in by Document Review's page),
`palantir_sdk_engine.py` (934), `http_trace.py` (58);
`funhouse_agent/deep/databricks_bridge.py` (774), `busy_retry.py` (295),
`pacing.py` (241). Budget handling (the spend line, the plain "budget used
up" message) is inside these; the sidebar panel belongs to the page.

**Vision calls and the file tools beside them** (9,335):
`deep/vision_engine.py` (826), `vision_view.py` (about 860 left after step
9), `image_view.py` (470), `vision_probe.py` (392), `deep/inline_images.py`
(176), `inline_store.py` (69), `deep/vision_kit.py` (1,654), and
`vision_tools.py` (5,204, less GSE's 318: the reference-figure and
worked-example readers at 4480-4740, their read roots at 468-502 and the
chart note at 5046-5067, which GSE registers through `added_tools` since
step 4).

**The figure helper** (about 7,900; OWN-77 (b), 2.8):
`deep/figure_helper.py` (789), `plot_box/__init__.py` (932), `child.py`
(1,189), `figures.py` (608), and beside them `funhouse_agent/drawing_wait.py`
(64) and `progress_notes.py` (46); the chart stack behind `plot_data`: `profile_figure/data_plot.py`
(1,845), `geotech_common/layout_ink.py` (1,283), `plan_furniture.py` (138),
`figure_to_png` and `get_pyplot` (about 35); the worked examples
`profile_figure/figure_examples.py` (994). The old paths keep aliases; GSE's
module figures go on importing `layout_ink` and `geotech_common.plotting`.

**What both agents also run** (5,561; `funhouse_agent/` and `deep/`):
`keep_work.py` (948), `limits.py` (431), `tool_guards.py` (436),
`scratch_guard.py` (634), `delivered_log.py` (103), `result_map.py` (134),
`strict_args.py` (126), `warm_imports.py` (56), `calculate_tool.py` (305),
`safe_expr.py` (160), `error_text.py` (405), `_fileio.py` (636),
`arg_checks.py` (172), `runtime_check.py` (203), `review_flags.py` (517;
every switch for both pages, a leaf module), `held_records.py` (295; its
report_ingest record shapes become a hook GSE fills). Also the assembly both
builds share, extracted from `deep/agent.py` at step 8 (the guard loop
round each helper, the general-purpose helper, the summarization trigger,
the busy-retry and keep-work wiring).

**Word and Excel** (1,753): `docx_renderer.py` (1,096), `xlsx_writer.py`
(346), `office_text.py` (311).

**The run record and its checks** (6,457): `webapp/activity_log.py`
(1,281), `run_mirror.py` (447), `cited_pages.py` (1,443),
`cited_documents.py` (1,484), `coverage.py` (1,249),
`deep/coverage_tools.py` (553). The geotech touches in `coverage.py` (the
`write_diggs` records) and `cited_documents.py` (the reference modules'
names) are inside `try` today and become hooks GSE fills.

**The suite's framework** (2,191; `funhouse_agent/review_eval/`):
`runner.py` (775), `checks.py` (1,010), `documents.py` (272), `__init__.py`
(24), and the `Task` type and task selection from `tasks.py` (about 110).
Each app brings its own tasks and hands the runner a way to run a turn on
its page (2.7 B).

**The document tools' binding** (about 850): what stays of
`document_tools.py` after step 9 (2.6).

### 2.4 Stays in GSE

**`funhouse_agent/`:** `__init__.py`, `_disclaimer.py`, `agent.py`,
`engine.py`, `native_tools.py`, `react_support.py`, `notebook.py`,
`run_local.py`, `run_scenarios.py`, `smoke_test_native.py`, `dispatch.py`,
`system_prompt.py`, `reviewer.py`, `reviewers.py`, `review_checklists.py`,
`worked_examples.py` with its `.json`, `reference_docs.py`,
`chart_reading.py`, `judgment_check.py`, `answer_sections.py`,
`report_checklist.py` (RI-34's alias), `geotech_test_suite.json`,
`test_scenarios.json`, `test_scenarios_notebook.ipynb`, `eval_samples/`,
`adapters/` (72 files, 23,271 lines), GSE's part of `vision_tools.py`
(318 lines, 2.3).

**`funhouse_agent/deep/`:** `__init__.py`, `agent.py` (the geotech build),
`prompt.py` (domain, judgment-call and ingest texts), `tools.py`
(`make_core_tools`, the report-ingest and library tools), `calc_figures.py`
(`FigureLooks`, the calc helper's, step 5), `fit_results.py`,
`setup_agent.py`, `setup_tools.py`, `unknown_tool.py`, `eval_harness.py`,
`selfcheck.py`, `rc_wheel_check.py`, `run_live.py`, `notebook.py`
(`DeepNotebookChat`).

**`webapp/`:** GSE's copy of the page (2.9), `webapp/geotech_page.py`
(step 7), `foundry_entry.py`, `databricks_launcher.py`,
`reference_fetch.py`, `requirements.txt`, `README.md`.

**Everything else:** every analysis package, `calc_package`,
`report_ingest`, `geo_project`, `profile_figure` and `geotech_common` (less
the chart stack and the worked examples, OWN-77 (b)),
`validation_examples`, and the repository's notes, desk and gates.

The four geotech suite tasks also stay: `report-extract-all`,
`report-extract-diggs`, `report-summary-vs-sheets` and `scale-log-depth`
(about 130 lines of `tasks.py`), with `review_eval/report_fixture.py`
(515). They run on the geotech page through the core's framework.

### 2.5 Files that split, and how

| File | To planlens | To the core | Copied / stays in GSE / to Document Review |
|---|---|---|---|
| `funhouse_agent/document_tools.py` | Toolkits per conversation, page conventions, DXF reads, markups (about 1,600) | The binding, the app's hints, Word and Excel reads (about 850) | |
| `funhouse_agent/vision_view.py` | Frame conversions, tiles, zoom, turned lettering, cut lines (about 730) | Image budgets per model, detail, text given to the model (about 860) | |
| `funhouse_agent/vision_tools.py` | | Everything else (4,886) | GSE: the reference-figure and worked-example readers, their read roots, the chart note (318) |
| `funhouse_agent/deep/agent.py` | | The shared assembly (step 8) | GSE: the geotech build. Document Review: the review build (step 8) |
| `funhouse_agent/deep/notebook.py` | | | The page: the stream formatters (about 380). GSE: `DeepNotebookChat` |
| `webapp/app.py`, `core.py`, `diagnostics.py`, `profiles.py` | | | Copied. Step 7 moves the geotech parts to `webapp/geotech_page.py` and the review parts to their own file, so each copy leaves one out |
| `webapp/tinyapps_engine.py` | | The Prompter connection | Document Review: the Key Vault lookup and the per-tester key |
| `funhouse_agent/review_eval/tasks.py` | | The `Task` type and selection | Document Review: 38 tasks. GSE: 4 tasks |
| `funhouse_agent/deep/held_records.py`, `coverage.py`, `cited_documents.py` | | All of it | GSE fills a hook (report_ingest record shapes, `write_diggs` records, reference module names) |
| `funhouse_agent/deep/prompt.py` | | | GSE (the review texts left at step 3) |
| `webapp/README.md`, `funhouse_agent/DESIGN.md` | | Its own README and DESIGN | Each app its own |

### 2.6 To planlens (step 9)

The owner named opening documents, markups and their check, and `find_like`
(OWN-74). Three of those are not pure wrappers today: the markup check and
`find_like` each ask the vision model (`vision_tools.ask_vision`), and the
document tools reach into the app's file confinement, attachments and Office
reader. planlens never calls a model and must import nothing of ours. So
each file moves with what it needs, and the host passes in the rest.

| File | Lines that move | What moves | What the host passes in |
|---|---|---|---|
| `funhouse_agent/document_tools.py` | about 1,600 of 2,446 | One toolkit per conversation (147-286); page-number conventions and viewer pages (645-931); reading DXF files (987-1286); writing, removing, placing and checking markups on a copy (1512-2382) | The source resolver (attachments, the conversation's folder, the confinement); the working folder; the signer's name. About 850 lines stay in the core (2.3): the tool names and the hints that name the app's own vision tools, the binding to the agent, and reading Word and Excel (python-docx and openpyxl are not planlens dependencies). |
| `funhouse_agent/markup_check.py` | 1,399 | All of it | The look function (an image and a question in, the model's text out; today `vision_tools.ask_vision`); where to keep the verdicts per conversation (today `vision_tools.reads_file`) |
| `funhouse_agent/find_like.py` | 410 | All of it | The look function; the time budget |
| `funhouse_agent/vision_view.py` | about 730 of 1,589 | Page and image frame conversions, tiles, zoom windows and location error (571-929); turned lettering and cut lines (1193-1560): what the moving tools convert boxes with | Nothing. The image budget per model, the detail level and the text given to the model stay in the core (planlens already owns `Document.render(budget=)`). |
| `funhouse_agent/page_numbers.py` | 185 | All of it | Nothing |
| *Proposed, the same kind:* `measure_tool.py` | 342 | The `measure` and `log_grid` glue | The look function, for scale labels |
| `scale_labels.py` | 287 | The scale-label read | The look function |
| `dxf_notes.py` | 233 | Notes on a copy of a DXF | Nothing (ezdxf is planlens' own) |

Not moving: `deep/geometry_tools.py` (1,144). It wraps only `planlens.ir`,
but it serves the review page alone and is switched off, so it goes with
Document Review; it is a later planlens candidate.

The tool specs move with the code and keep the app's descriptions and
schemas word for word, so neither page's agent changes (step 9's exit).
Their tests move to planlens too, rewritten against a fake look function.

### 2.7 The hard couplings

**Cut so far** (steps 1-5; merges 148cb80, 74f71fc, ed090a6): 18 known
couplings down to 13. Gone:
- the review tools' factory in GSE's tools module (old coupling 2; step 4);
- the review prompt's module importing GSE's system prompt (old coupling 5;
  step 3);
- `calculate`'s evaluator, the Word renderer and the figure helper's result
  cap (step 5);
- the package doors loading the old agent, the reviewers, the setup agent
  and geo_project (step 2).

**Left: the review page into GSE** (the boundary test's list, passing on
2026-10-21; the numbers are the ones the test uses, and 2 and 5 are cut):
1. **The review page is built by GSE's builder:** `webapp/core.py` →
   `deep/agent.py`, and `deep/review_agent.py` → `deep/agent.py`. Step 8.
3. **The package door's first-import disclaimer notice** (soft, inside
   `try`). The review files load `funhouse_agent`'s package door until they
   leave it at step 13. Document Review shows its own notice (OWN-78).
4. **The chart stack:** `deep/vision_kit.py` → `dispatch`, the
   profile_figure adapter and `profile_figure`; `plot_box/child.py` →
   `geotech_common` and `layout_ink` (loaded by file at 680-700). Step 6.

The other soft ones (inside `try`; the review page works without them):
- `plot_box` → `geotech_references.chart_lines`: step 6 makes it a source
  GSE registers, and step 13 drops it from Document Review (the one
  deliberate fingerprint change).
- `vision_tools` → `reference_docs`, `geotech_references` and
  `_figures_db`: what is left of GSE's two readers; cut at step 11.
- `webapp/core.py` → `reference_fetch`: cut at step 7.

The test labels each coupling with the step of the earlier plan that cuts
it. In this plan, "step 10" for the disclaimer and "step 12" for the chart
lines both read as step 13. Its "step 7", "step 8", "step 11" and
"step 15" still fit.

**New with this layout:**
- **A. planlens imports nothing of ours** (step 9). The files moving there
  call into the app today:
  - `vision_tools.ask_vision`: the markup check, `find_like`, and the
    scale-label read;
  - `vision_tools.reads_file`;
  - the confinement helpers (`find_readable_file`, `read_roots`,
    `display_path`, `PathRefused`, `files_here_text`, `SCRATCH_DIR`);
  - `_fileio.host_output_dir` and `into_working_folder`;
  - `error_text.tool_error_json`;
  - `image_view.shown_size`;
  - `office_text`;
  - `vision_view`'s conversions.

  Each either moves with them (the conversions, page numbers) or becomes
  something the host passes in: the look function, the source resolver,
  the working folder, the signer.
- **B. The core imports neither app** (steps 10-11):
  - `review_eval/runner.py` drives the page's turn path through
    `webapp.core`, `webapp.profiles` and `webapp.output_capture`;
  - `review_eval/documents.py` registers GSE's reference fetcher;
  - `tinyapps_engine.py` reads Key Vault (`tinyapps_settings`) and the
    sign-in name (`identity`);
  - `held_records.py` reads report_ingest's record shapes;
  - `coverage.py` and `cited_documents.py` look up GSE inside `try`.

  Each becomes a hook the app fills. A test fails if a core file imports
  `webapp`'s page files, `dispatch`, an adapter or a module package.
- **C. Neither app imports the other** (from step 13). Document Review's
  CI runs with nothing geotech installed. GSE's reverse test builds GSE
  with no review file present (step 15).

**GSE into the review code** (steps 8 and 15 cut these):
- `deep/agent.py` imports `review_agent`, `minimal_agent` and
  `review_prompt` (step 8).
- `deep/prompt.py` re-exports the review texts for old callers (step 15).
- `pages_entry.py` puts the review page in GSE's navigation, and
  `databricks_launcher.py` boots the two-page entry by default (step 15).

### 2.8 The figure helper and plotting box (OWN-77: DECIDED (b))

**Decided 2026-10-21 (OWN-77): option (b).** The owner: "Follow what we
talked about with the planlens fold."
- During the split the code moves as in column (a): the helper, the
  plotting box, the chart stack and the nine geotech worked examples all go
  into the core. GSE hands in the reference chart lines.
- The move into the core comes after the planlens fold: step 9 first, then
  step 11 (b).
- After the split, **DR-35** gives Document Review its own general examples
  (drawings, schedules, quantities) and checklist lines, registered with
  the helper and measured on the suite. Column (b)'s extra work is DR-35's,
  not a split step.
- The core's total in 2.1 (about 37,900) already counted all of this, so
  (b) leaves it unchanged. Re-counted on 2026-10-21 by counting the lines
  of each file in 2.3: the helper, box and examples 4,622, the chart stack
  about 3,300, about 7,900 in all.

The options as they were measured follow.

Since GM-227 (b4b9a08), `GEOTECH_FIGURE_HELPER` and `GEOTECH_PLOT_CODE`
are on by default on both pages. The owner on GM-92: "on everywhere".

The machinery is general: draw with `plot_data` or with code in the locked
box, look at the picture, redraw to a checklist. Its geotech content is
small:
- the standard chart lines from the reference library;
- nine worked examples (a boring plan, an SPT profile, wall pressures and
  the like);
- a few checklist lines (elevation axis, kPa and kN, "the FIGURE HELPER of
  a geotechnical agent").

`plot_data` and its chart stack are on both pages, so they go to the core
under all three options.

| | (a) To the core; GSE hands in the chart lines | (b) As (a), then Document Review's own examples | (c) Document Review drops both |
|---|---|---|---|
| Files to the core | `deep/figure_helper.py` (789; `FigureLooks` already left at step 5), `plot_box/` (932 + 1,189 + 608), `drawing_wait.py` and `progress_notes.py` (64 + 46), `profile_figure/figure_examples.py` (994): about 4,600 lines | The same; Document Review's own examples and checklist lines live in its private source and register with the helper | None; both stay in GSE |
| Imports still to cut | The worked examples (`figure_helper.py` imports `profile_figure.figure_examples` at 202-241) → the core's copy; `plot_box/child.py`'s loads of `layout_ink` and `plan_furniture` → the core's chart folder (step 6); the box's chart lines → a source GSE registers | The same, plus an examples-and-checklist registration each app fills | The review build leaves out the helper, its main-agent middleware and `plot_data_replaces` |
| New code | One hook (the chart lines) | The hook, Document Review's examples file and checklist lines, a suite task, a live wave | None |
| Document Review's fingerprint | Unchanged, except the box's chart-lines note goes at step 13 | Changes (helper prompt and examples): measured on the suite | Loses the figure helper and `plot_code` |
| Install | The core keeps SciPy: the box lets code import it | The same | SciPy can leave the core |
| The owner | Fits GM-92 | Fits GM-92; **chosen** | Reverses GM-92 |

The plan had recommended (a) during the split and (b) after it, as its own
item (one change at a time, as with D5). The owner's (b) is the same order:
the split moves the code, then DR-35 changes the examples.

### 2.9 Each app's chat page (copied at step 13)

**Copied, one copy to each app** (12,867 lines): `webapp/app.py` (1,613),
`core.py` (3,789), `turn_jobs.py` (457), `turn_record.py` (405),
`output_capture.py` (117), `ws_upload.py` (113) with
`ws_upload_component/index.html` (145), `budget_panel.py` (68),
`feedback.py` (227), `version_guard.py` (95), `diagnostics.py` (577),
`profiles.py` (288), `pages_entry.py` (85), `file_widgets.py` (323),
`page_errors.py` (184), `conversation_export.py` (353),
`conversation_restore.py` (436), `sharepoint_store.py` (970),
`sharepoint_tools.py` (1,578), `identity.py` (207), `helper_models.py`
(247), `email_tools.py` (198), `__init__.py` (12), and the stream
formatters the page uses from `deep/notebook.py` (about 380).

**Host files, one app each.** Document Review: `tinyapps_settings.py`
(176), `graph_sharepoint.py` (404), `tinyapps_entry.py` (84), 664 lines.
GSE: `foundry_entry.py` (48), `databricks_launcher.py` (1,920),
`reference_fetch.py` (71), 2,039 lines.

**What each copy leaves out at the copy.** Step 7 gathers each page's own
parts into its own file, so each copy can leave the other's file out:
- Document Review's copy leaves out `webapp/geotech_page.py` (the
  specialist and behaviour pickers, the reference fetcher, the
  judgment-call check, the reference-library diagnostics). It also leaves
  out the per-helper model choice (`helper_models.py`, a Foundry need), the
  websocket uploader (a fix for the Databricks proxy, which Tiny Apps does
  not have), and the e-mail tool (OWN-69).
- GSE's copy leaves out the review profile and the orientation.

Either copy may drop more later; from the copy on it is that app's own.

**Why a copy.** It is the owner's option (d). Each page changes with its
own users' feedback: a broad multidisciplinary group for Document Review,
the owner for GSE. The price: a fix to the machinery both pages had (turn
jobs, saving, uploads, storage) is made once per app after the copy. About
12,900 lines is the size of that.

## 3. Repositories

| Code | Repository | Public |
|---|---|---|
| GSE and the core | This repository. The core in `packages/<core>/` with its own `pyproject.toml` and `tests/`; GSE's `pyproject.toml` stays at the root. | yes |
| planlens | Its own repository, as today | yes |
| geotech-references | Its own repository, a submodule here, as today | yes |
| Document Review | A private repository (1.5 (a)) or the GHE repo (1.5 (b)); testers run the GHE copy | no |

**Why the core stays in this repository.**
- GSE and the core change together most often (figure helper, engines,
  vision).
- One change, one CI run, both tested together.
- Moves inside one repository are renames git can follow.

Document Review's CI tests against the released core, or against this
checkout while a change is in flight. Moving the core out later is a
mechanical copy of one folder.

**The public-repository rules.**
- No Department material in the core: no guides, no internal host names, no
  tester documents, no CfA mailboxes. `webapp/tests/test_public_text.py`
  walks the core and its docs from step 10.
- Department-specific code (the Key Vault naming scheme, the sign-in
  header, the Graph app-registration settings) lives in Document Review's
  private source from step 13.
- The GHE wrapper repo holds the deployment's own values (the published
  address, the Key Vault name).
- Privacy starts at the copy. The review code published in every GSE
  release since 5.26 stays published (OWN-76).

## 4. Steps in order

**Rules for every step.**
- One merge to master, small enough for one helper and one reviewer.
- Both apps work after it: the geotech page on Foundry and Funhouse, and
  Document Review. Until step 14, Document Review runs through GSE's
  two-page entry; after it, from its own copy.
- Its targeted tests and its exit check pass. Heavy suites never run two at
  a time.
- The notes checklist runs after each step (BACKLOG, HANDOFF, CLAUDE.md
  state clause, memory, desk).
- Each app pins the core at an exact version. The core carries its own
  tests.

**Four techniques used throughout.**
- **The fingerprint** (step 1) of each page's agent with default switches:
  - what the model is sent on its first call (the system prompt; each
    tool's name, description and argument schema);
  - each helper's name, prompt and tools;
  - the middleware classes in order.

  Steps marked *(no behaviour change)* leave every fingerprint identical. A
  deliberate change re-records it in its own commit, with the reason.
- **The boundary test** (step 1): build and run the Document Review agent
  in a child process with every geotech module blocked, recording each
  blocked import as a known coupling. The list may only shrink; from step 8
  it is empty apart from the soft ones. From step 11 a CI job runs with the
  geotech packages not installed at all.
- **Moves that git can follow** (inside this repository). A move commit
  holds `git mv` plus import-line rewrites only. Old import paths keep
  working through an alias table the old package installs: `sys.modules`
  maps the old name to the same module object, with no stub files at the
  old paths. So:
  - an in-flight branch that edited a moved file merges onto its new path;
  - `monkeypatch.setattr("funhouse_agent.vision_tools.x", ...)` still
    patches the real module, which a re-export stub would silently not do.

  The aliases come out in step 16. A move into planlens (step 9) is
  different: it is a copy into planlens and a delete here, and this
  repository keeps the history.
- **The copy** (step 13). Document Review's files are copied once, from a
  tagged commit, into its private repository. From that day every
  review-page change goes there only. The copies left here are frozen: a
  test of their contents fails on any change, until step 15 deletes them.

### Step 0. Decisions and preconditions (no code)
- **Decided:**
  - OWN-74 and OWN-76 (2026-10-21; this layout);
  - OWN-77 (2026-10-21; option (b): the figure helper and plotting code
    into the core at step 11 (b), Document Review's own examples after the
    split, DR-35);
  - OWN-75 (2026-10-21; the held train is released at the release point
    below, or held if that is no longer practical);
  - D4 (2026-10-20; the geotech page leaves Tiny Apps);
  - D5 (the lead's default; testers keep today's agent).
- **Still open, each with when it is needed:**
  - OWN-72's name: before step 10;
  - D10: before step 13;
  - OWN-72's CfA answer and OWN-78: before step 14.
- RI-34 is merged (done).
- In-flight branches that touch the moving files are merged or parked
  before steps 10 and 11.
- **Exit:** the decisions are folded into BACKLOG.

### Steps 1-5. Done (no behaviour change), unchanged by this rewrite
- **Step 1, the line measured:**
  - `webapp/tests/test_review_boundary.py` (a strict xfail until step 8);
  - `webapp/tests/test_page_fingerprints.py`, pinning four builds (the
    review default, `lean`, `minimal`, geotech);
  - merge 148cb80.
- **Step 2, lazy package doors:** `import funhouse_agent.deep` went from
  2.43 s to 0.21 s.
- **Step 3, the review prompt in its own module:**
  `funhouse_agent/review_prompt.py`.
- **Step 4, the vision kit out of the dispatch module:**
  - `deep/vision_kit.py`, which loads no dispatch;
  - GSE adds its two reference readers through `added_tools`.
- **Step 5, small helpers to neutral modules:**
  - `arg_checks.py`, `safe_expr.py`, `docx_renderer.py`, `run_mirror.py`;
  - the old paths are the same module objects;
  - merge 74f71fc, then the fix that import of `calc_package` or
    `report_ingest` no longer loads `funhouse_agent`'s package entry (merge
    ed090a6).

**Refreshing the fingerprints** (for everyone, from step 1 on):
- A split step never re-records them, except the one change step 13 names;
  it finds what moved instead.
- Any other deliberate change to a prompt, a tool's description or schema,
  a helper or a budget re-records in a commit of its own that says why. From
  the repository root: `python -m webapp.tests.page_builds record`.
- The planlens-served tools are recorded per planlens version (CI's PyPI
  release and planlens main side by side; `record` keeps the other
  entries). CI fails on a planlens with no entry.
- The rule is also the test's docstring
  (`webapp/tests/test_page_fingerprints.py`).

### Step 6. The chart stack to a neutral package (no behaviour change; unchanged)
- **Change:**
  - A new `funhouse_agent/charts/` package holds `data_plot.py`,
    `layout_ink.py`, `plan_furniture.py`, `figure_to_png` and `get_pyplot`,
    with aliases at `profile_figure.data_plot`, `geotech_common.layout_ink`,
    `geotech_common.plan_furniture` and `geotech_common.plotting`.
  - `reference_chart` stays in the GSE adapter, which wraps the chart
    function.
  - The worked examples (`profile_figure/figure_examples.py`) join the
    package (OWN-77 (b)).
  - The plotting box follows the move. It loads `layout_ink` and
    `plan_furniture` by file (`plot_box/child.py:680-700` and the plan
    furniture load).
  - The box's chart lines come from a source GSE registers instead of a
    look for `geotech_references`.
- **Tests:**
  - `profile_figure`, `geotech_common`, and every module whose figure tests
    import `layout_ink`, run in chunks;
  - `deep/tests/test_figure_*`;
  - `funhouse_agent/tests/test_plot_box.py`;
  - `webapp/tests/test_figure_code_cards.py`.
- **Exit:** fingerprints identical. Coupling 4 (2.7) is gone, apart from
  `reference_chart` and the box's chart lines, which are GSE's.

### Step 7. The shell asks the page, not geotech (no behaviour change; its purpose changes)
- **What changes from the earlier plan:** the hooks are no longer the way
  one shared shell serves two apps for good. Each app gets its own copy of
  the page at step 13. Until then both pages still run one shell, and this
  step cuts the seam the copy will follow.
- **Change:**
  - `AppProfile` gains hooks: `disclaimer`, `placeholder`,
    `sidebar_extras`, `behavior_kwargs`, `agent_builder`, `on_build` (the
    reference fetcher), `turn_checks` (the judgment-call check),
    `diagnostics_checks` and `version_dists`.
  - The geotech branches of `webapp/app.py`, `core.py` and
    `diagnostics.py` move into `webapp/geotech_page.py` and fill the
    `GEOTECH` profile's hooks.
  - The review profile's own parts move into one file of their own: the
    profile, the orientation, its builder (still
    `build_deep_agent(review_page=True)` here), and its notice. The current
    disclaimer text stays until the owner approves the wording (OWN-78).
- **Tests:** `webapp/tests` (all), including `test_app_smoke.py` for both
  pages.
- **Exit:**
  - Fingerprints identical.
  - `app.py`, `core.py` and `diagnostics.py` import nothing from
    `funhouse_agent.dispatch`, `reviewers`, `_disclaimer`,
    `reference_docs`, `worked_examples` or `geotech_references`.
  - Each page's own parts are in one file, so each copy at step 13 leaves
    the other's file out.

### Step 8. The review page builds itself (no behaviour change; unchanged)
- **Change:**
  - Today's review build is extracted from `build_deep_agent` into
    `funhouse_agent/review_build.py` (`build_review_page_agent`). It goes
    to Document Review at step 13.
  - The assembly it shares with the geotech build moves to
    `funhouse_agent/deep/assembly.py`. It goes to the core at step 11.
  - The review profile's builder and `review_eval.runner.build_page_agent`
    call the new build. It carries the figure helper and its main-agent
    middleware as today.
  - `build_deep_agent` loses its `review_page` branches and its imports of
    `review_agent`, `minimal_agent` and `review_prompt`.
    `review_page=True` delegates to the new build, for old callers, until
    step 15.
- **Tests:**
  - the fingerprints (this is the step they exist for);
  - `deep/tests`;
  - `review_eval/tests`;
  - `webapp/tests/test_app_profiles.py`.
- **Exit:**
  - Fingerprints identical.
  - **The boundary test's couplings list is empty for all three review
    builds**, apart from the soft ones, which now go through hooks.
  - The step measures the shared assembly's lines for 2.1.

### The release point: the held train goes out first (OWN-75, decided 2026-10-21)
- **The owner:** release the held train before the files move. "If it's
  too late, no big deal, just go with hold. Don't expend big effort to
  revert back."
- **Where it falls:** after step 8, before step 9 (b), the first step that
  takes files out of this repository (the document tools, to planlens).
  Steps 1-8 only rearrange files inside it, behind aliases, with the
  fingerprints unchanged, so they may already be in the held train.
- **What goes out:** the app with everything on master since 5.33.0
  (among it the pandas fix that lets the testers' install restore
  pandas==3.0.2), with planlens and geotech-references as the held train
  pairs them (OWN-55, REF-39). The release gate and the owner's word are
  as for any release.
- **If that is no longer practical** (the owner still holding the release
  when step 9 (b) is ready, or master not releasable without undoing
  merged work): nothing is reverted. The split goes on, and the held work
  goes out with the next release, at the latest the core's first at
  step 12.
- **Why first:** a clean version from before the split to fall back to, and
  the pandas fix on Nexus while CfA works.

### Step 9. planlens takes the document tools (new; a planlens item with its own release)
- **When:**
  - It can run beside steps 6-8: different files, and no new package.
  - Part (a) is planlens' own and can go ahead at any time. Part (b), in
    the app, comes after the release point above.
  - It needs the owner's word for a planlens release.
  - It is not waiting on OWN-80 or the Tiny Apps start.
- **(a) In planlens:**
  - `ReviewToolkit` gains a `look` argument next to `resolve_source`: a
    function taking an image, a question and a kind, and returning the
    model's text.
  - The code of 2.6 moves in as planlens modules, the named files first.
    The proposed three follow if the planlens review agrees.
  - Their tests move too, rewritten against a fake look function.
  - planlens still makes no model call and gains no dependency.
  - The tool specs keep the app's descriptions and schemas word for word.
  - planlens is released, published to PyPI before the app's pin is raised.
- **(b) In the app:**
  - The moved files are deleted.
  - `document_tools.py` becomes the binding (about 850 lines). It hands
    planlens the source resolver, the look function
    (`vision_tools.ask_vision` with the turn's engine), the working folder
    and the signer.
  - The feature detection (`has_tool`, `search_supports_fuzzy`) stays one
    release for an older planlens.
- **Tests:**
  - planlens' suite;
  - the app's document, markup, `find_like` and `measure` tests that stay
    (the binding);
  - the fingerprints.
- **Exit:**
  - The fingerprints are recorded for the new planlens version. That
    entry equals the previous version's, tool for tool (name, description,
    schema).
  - The boundary test is unchanged.
  - A marked copy, a `find_like` search and a markup check run end to end
    offline with a fake engine, as today.

### Step 10. The core package, empty, with its build and CI (was step 9)
- **Needs:** OWN-72's name; the release point passed (the held train
  released, or held as no longer practical; OWN-75).
- **Change:**
  - `packages/<core>/pyproject.toml`: version 1.0.0.dev0, the dependency
    list of 1.1, the `[anthropic]` extra.
  - `<core>/__init__.py`, holding only the alias helper.
  - GSE's `pyproject.toml` depends on the core at the same version,
    exactly; CI installs both editable.
  - `test_public_text.py` and the other tests that read `pyproject.toml`
    learn about the new file: `test_docs_currency`, `test_version_guard`,
    `test_tinyapps_entry`, `test_ws_upload`;
    `funhouse_agent/tests/test_disclaimer.py`,
    `test_drawing_ir_adapter.py`; `deep/tests/test_coverage_offline.py`.
  - The publish workflow builds named directories only (section 5).
- **Tests:** a new `webapp/tests/test_wheels.py`. It builds both wheels,
  installs them into a fresh venv, imports the entry points, and lists the
  files: no `tests/`, no `module_work`, no `raw/`, no
  `tinyapps/reference`.
- **Exit:** the full gate in chunks, plus the wheel test, on the exact
  tree.

### Step 11. Into the core (was steps 10-11, without the page)
- **Change:** the files of 2.3 move to `packages/<core>/<core>/`, pure
  renames and import rewrites, in three merges:
  - **(a)** model connections and vision calls. `tinyapps_engine.py`'s Key
    Vault lookup and per-tester key become a hook; GSE's two-page page
    fills it until step 13, then Document Review's page does.
  - **(b)** the figure helper, plotting box, chart stack and the nine
    geotech worked examples (OWN-77 (b)). GSE hands in the reference chart
    lines. Document Review's own examples come after the split (DR-35).
  - **(c)** everything else in 2.3: guards and records, Word and Excel, the
    checks, the suite's framework (with the page's turn handed in, 2.7 B),
    the document tools' binding, and step 8's shared assembly.
- **Aliases:** `funhouse_agent` and `webapp` keep alias tables, so
  `webapp.engine_config`, `funhouse_agent.vision_tools` and the rest still
  import.
- **Tests:**
  - per merge: `funhouse_agent/tests`, `deep/tests`, `review_eval/tests`,
    `webapp/tests`, and the module suites that import the charts;
  - each merge: the boundary test, the fingerprints, the wheel test.
- **Exit:**
  - The full gate.
  - In a fresh venv with only the core and planlens, every core module
    imports and the core's own tests pass.
  - A test shows the core imports nothing from GSE, from the page files or
    from the review files (2.7 B).
  - The new CI job **"core-only"** passes (section 5). Today's review page
    builds and runs there from this repository's review files, with
    nothing geotech installed. That proves the testers' install before
    the copy.

### Step 12. The core's first release, and GSE on it (new)
- **Needs:** the owner's release word.
- **Release order:** planlens (already at step 9), then the core, then GSE
  pinned to the core exactly. If the held train was held at the release
  point (OWN-75), it goes out here with GSE.
- **GSE still carries both pages.** Testers get it from Nexus as today,
  with the core arriving beside it. The core is a new public package for
  the Nexus scan, pure Python, like planlens.
- **Exit:**
  - The core and GSE install from PyPI in a fresh venv.
  - GSE runs on Foundry from the new wheels: diagnostics, a short question,
    a calc package, a figure (INF-25's list).
  - The testers' two-page install from Nexus still starts on the test
    machine.

### Step 13. Document Review's private home: the copy (new; replaces the de-docreview package and the suite move)
- **Needs:** D10 (where the copy is worked on), and step 12 (the core on
  PyPI).
- **Change:** a copy from a tagged commit into the private repository:
  - the files of 2.2;
  - Document Review's copy of the page (2.9, leaving out
    `webapp/geotech_page.py`), and its host files;
  - its tests and its 38 tasks;
  - `TINYAPPS.md` and the wrapper template.

  It gets a `pyproject.toml` for development and tests only (nothing built
  or published), pinning the core and planlens exactly. It gets its own CI:
  Python 3.11 and 3.12, the fleet pins, nothing geotech installed. That CI
  runs its tests, the boundary check as a plain test (there is no geotech
  left to block), and the review fingerprints.
- **The one deliberate fingerprint change.** It comes here, when Document
  Review stops installing geotech-references: `plot_data` loses its
  `reference_chart` argument, and `plot_code` loses `chart_lines()` and its
  note in the description (both are the standard geotech chart lines).
- **From this day:**
  - Review-page changes go to the private copy only.
  - This repository's review files are frozen (2.2, the review profile's
    file, the review tasks), guarded by a test of their contents, until
    step 15.
  - BACKLOG keeps Document Review's items (the one list), written with the
    same public-text care as today.
- **The suite:**
  - `score_review_suite` and the 38 tasks go with the copy; the framework
    is in the core.
  - GSE's four tasks stay and run through the framework against the
    geotech page.
  - Version reporting names the right distributions.
- **Exit:**
  - One baseline-arm suite run on Foundry from the private copy (its
    folder in the Foundry workspace, the core and planlens wheels beside
    it). It must match the last baseline task for task, with token use
    within noise.
  - Its full trace read is the owner's rule. The owner or the FDE runs it.

### Step 14. The Tiny Apps wrapper for Document Review (was step 14)
- **Needs:**
  - CfA's yes to 6a (else 6b or 6c; OWN-72);
  - OWN-78 (the notice and the SharePoint folder);
  - the core on Nexus.
- **Change:**
  - The GHE wrapper repo gets the source folder (route A, 1.4).
  - `app.py` calls Document Review's entry.
  - `packages.txt` holds the core and planlens at exact versions, the fleet
    pins with **pandas==3.0.2 restored**, and the Key Vault libraries.
  - `run.sh` is unchanged; `.env.example` and the README are revised.
  - In `TINYAPPS.md` (now in the private copy) the test-machine recipe
    loses "switch to the geotech page", and question 6 becomes 6a-6d.
- **Tests:**
  - The wrapper's `app.py` runs in a fresh venv built only from
    `packages.txt` (WSL, Python 3.11 and 3.12;
    `reference_pinned_version_testing`).
  - Streamlit starts headless with the "No model configured" banner.
- **Exit:**
  - The owner's round on the test machine passes (OWN-29, rewritten):
    install from `packages.txt`, start, sign-in line, diagnostics, upload,
    orientation, a marked-up copy and a Word memo, storage sync.
  - On the published app, the first-hour check (OWN-30) follows CfA.

### Step 15. GSE lets go of the review page (was step 15)
- **Only after** step 14's exit, or the owner's word.
- **Change:** deleted from this repository:
  - the frozen review files;
  - `DOCUMENT_REVIEW` and the review profile's file;
  - GSE's `tinyapps_entry.py` (GSE is not on Tiny Apps, D4);
  - the review page in `pages_entry` and in `run_on_databricks`'s
    two-page default;
  - the `review_page` delegate and the review aliases in
    `funhouse_agent`.

  GSE's README and disclaimer no longer describe a document-review page.
- **Tests:**
  - the reverse boundary test (GSE imports and builds with no review file
    present);
  - the geotech fingerprint;
  - the full job.
- **Exit:** a GSE run on Foundry from the new wheels (INF-25's list). Then
  a GSE release.

### Step 16. Aliases out, notes in (was step 16)
- **Change:**
  - One release after step 15, the alias tables in `webapp` and
    `funhouse_agent` are removed. The owner's notebook cell is given anew
    if its import path changed.
  - Docs updated: the CLAUDE.md state block and module table, HANDOFF
    §0a-current, `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md`, each
    package's README and DESIGN, `module_work/harness_theory/` paths, the
    Harness Map (`build.py --summaries-todo`), memory.
  - BACKLOG items gain their home: planlens, the core, GSE or Document
    Review (section 6).
  - The earlier plan's optional "move to a private repository" is gone: it
    happened at step 13.
- **Tests:** the three doc gates (`test_backlog`, `test_docs_currency`,
  `test_public_text`); the full job.
- **Exit:** no import of an old path remains (`grep`), and CI is green on
  every job.

## 5. Tests and CI

### Which suites go where

The rule: a test goes with the code it tests.
- A test that builds the geotech agent or calls a module stays in GSE.
- A test that builds the review page's agent goes to Document Review.
- A test of a moved planlens tool goes to planlens.
- The page's tests are copied with the page.

| Today (test files, counted 2026-10-21) | To planlens | To the core | To Document Review (private) | Stays in GSE |
|---|---|---|---|---|
| `webapp/tests` (66 files) | | The engine and activity-log tests | `test_tinyapps_entry`, `test_orientation_overview`, the review half of `test_app_profiles`, a copy of the page's tests (turn jobs, cards, SharePoint, identity, uploads, diagnostics base, `test_figure_code_cards`) | Repository gates (`test_backlog`, `test_docs_currency`, `test_public_text`, `test_wheels`, the boundary and fingerprint tests until step 13); `test_judgment_check_turn`; the geotech half of `test_app_profiles`; GSE's copy of the page's tests |
| `funhouse_agent/tests` (122 files) | Markup check, `find_like`, page numbers, `measure`, scale labels, DXF notes | Vision, cited pages and documents, coverage, office files, xlsx, the plotting box (`test_plot_box`, `test_plot_box_drafts`) | Findings ledger tests | Dispatch, adapters, reviewers, worked examples, system prompt, disclaimer, chart reading, v1 agent |
| `funhouse_agent/deep/tests` (70 files) | The document tools' planlens parts | Busy ladder, pacing, keep-work, limits, guards, vision engine, strict args, inline images, `calculate`, the figure helper (`test_figure_helper`, `test_figure_runs`, `test_figure_turn`) | `test_review_harness_offline`, `test_review_m1_offline`, `test_review_minimal_offline`, `test_review_geometry_offline`, `test_review_digest_tools_offline`, `test_review_prompt_rules` | The geotech build, the calc, references and setup helpers (with `test_deep_calc_figures_offline`), the eval harness, the coverage gate on the geotech page |
| `review_eval/tests`, `review_digest/tests` (7 files) | | Framework parts | Task parts; all of `review_digest/tests` | Geotech-task parts |

The repository gates stay at this repository's root, because they guard the
repository, not a package.

### CI

- **"Tests" (today's job, kept):** both Python versions, GSE and the core
  installed editable, everything run, planlens from PyPI and from main side
  by side as today. Any change to the core is tested against GSE here.
- **"core-only" (new at step 11; minutes):** Python 3.11 (the test
  machine's 3.11.9) and 3.12 (App Service). It installs planlens from PyPI,
  the core, and CfA's fleet pins. It does **not** check out the
  geotech-references submodule or install any analysis back end. It runs
  the core's tests and, until step 13, builds the review page from this
  repository's review files.
- **"Wheels" (new at step 10):** builds the GSE and core wheels and
  installs each into a clean venv. It imports the entry points and checks
  the file lists. This is the 5.19.0 lesson: a wheel can build and simply
  lack a package.
- **"Publish":** tags `v5.*` build the root (GSE); tags `core-v*` build
  `packages/<core>`. Nothing here builds Document Review, which never
  leaves its private repository except as the GHE copy.
- **Document Review's CI** runs in its private repository (step 13), with
  nothing geotech installed.
- The open CI items carry on:
  - INF-24 (Python 3.11) is covered for Document Review by its own CI and
    for the core by "core-only";
  - INF-28 (numpy 2.5) still applies to GSE.

### The local release gate

One chunk is added, "core", run in its own venv like the CI job. As now, no
two heavy chunks run at once (15.5 GB laptop).

### The suite (review_eval) after the split

- The framework is in the core, the 38 review tasks in Document Review, the
  four geotech tasks in GSE.
- A Document Review suite run installs planlens and the core and runs
  Document Review's source, so every run also proves the testers' install.
- The live smoke harness (`module_work/live_smoke/`) stays here and drives
  the geotech page. Its review-page waves move to the private copy at
  step 13.

## 6. What changes for the Tiny Apps tester deploy and for Foundry

### Tiny Apps (TA-*, the CfA owner items)

**Nothing changes for testers until step 14.** Today's route (GSE 5.33.0
or the next release from Nexus, two pages) keeps working. From step 14 the
wrapper runs Document Review from its source folder: one page, no geotech
packages.

| Item | After the split |
|---|---|
| OWN-29 install + test-machine round | Becomes "install the wrapper with the source folder" (step 14's exit). The pandas 3.0.2 pin returns: Document Review has no python-ags4. |
| OWN-30 first-hour check | Document Review only; the recipe's geotech-page step goes |
| OWN-43 to OWN-48, OWN-50, OWN-51 (keys, SharePoint, origin, identity, image deployment, FIPS, budget response, metering) | Unchanged: same settings. The connection code is in the core; the Key Vault and sign-in parts are in Document Review's page. |
| OWN-49 (Nexus vs vendoring; fleet pins; resource envelope) | The first two are CfA 6a-6d (OWN-72); the fem2d / Monte Carlo resource question went with the geotech page |
| TA-03, TA-04, TA-05, TA-06 (live GPT, unverified live, SDK-free spend line, budget-spent test) | Document Review, with TA-05 and TA-06's code in the core's connection |
| TA-07, TA-08, TA-10, TA-14, TA-16, TA-18, TA-23, TA-24, TA-28 (shell, uploads, cards, helpers, busy ladder, kept work, 401 retry) | Before step 13: fixed once in the shared page. After it: the busy ladder, kept work and 401 retry in the core; the shell, uploads and cards in each app's page |
| TA-27, TA-13 (Funhouse launcher live check, metering) | GSE's launcher |
| TA-09, TA-15 (email tool, old live checks) | Already proposed for dropping (OWN-69) |
| TA-11 (Plotly PNG twin for `subsurface.plot_*`) | GSE |
| Document Review items (DR-*) | Document Review, except: DR-11 (recompute from a report needs the modules) → GSE; DR-20, DR-21, DR-22 → report_ingest with RI-34; DR-23 splits (core coverage; its report_ingest part → GSE) |
| Open GM-* items on the figure helper and plotting box | The core; their chart-lines parts GSE |
| VIS-* | The core, or planlens where the code moved there (markups, `find_like`, frame conversions); VIS-06, VIS-10 and VIS-17 (reference chart reading and catalogs) → GSE |
| EV-* (suite measurement) | Document Review, except the geotech-task arms |

**The CfA approval path.**
- The SIA covers the code and its connections.
- The connections are unchanged (Prompter, Key Vault, SharePoint by app
  registration), and the dependencies shrink.
- Running a source folder from the wrapper repo is the usual Tiny Apps
  shape. Ask CfA whether it needs a new SIA or only a PCR.
- The App Services sync is unchanged: push, then ask for a sync.

### Foundry (INF-25 and after)

- **The GSE app on Foundry is the geotech page alone, as it already is**
  (`webapp.foundry_entry`). The split changes its imports, not its
  behaviour; the geotech fingerprint guards that at every step.
- **Install:** PyPI brings the core automatically. The INF-25 test-wheel
  route uploads one more file (four wheels, one command).
- **INF-27** (a Claude leg for `palantir_models`) is work in the core's
  `palantir_sdk_engine.py`, so Document Review's suite runs on Foundry gain
  it too.
- **INF-29** (host names in public docs): the extended public-text guard
  (step 10) covers the core's docs from the start.
- **What GSE loses:** the Document Review page and its switches. GSE keeps
  every document and vision tool it has: they are in the core and
  planlens.

## 7. Risks, and the owner's decisions

### Risks

| Risk | What guards it |
|---|---|
| Testers' page changes without anyone choosing it | Fingerprints at every step; one baseline suite run on Foundry at step 13; the only deliberate change is listed (step 13) |
| A Document Review install that quietly needs a geotech package | The boundary test (step 1 on), the "core-only" job (step 11 on), Document Review's own CI with nothing geotech installed (step 13 on) |
| The two copies of the page drift | Accepted by design (OWN-74 (d)): each page follows its own users. What both must share is in the core. |
| A review-page change lands in this repository after the copy | The freeze test (step 13) until step 15 deletes the files |
| A core change breaks the other app | Each app pins the core exactly; the core's own tests; "Tests" runs GSE on every core change; Document Review's CI runs against the pinned core and, before a bump, the new one |
| Department-specific code in the public core | Rule 2 (1.2); `test_public_text` over the core and its docs (step 10); the Key Vault, sign-in and Graph settings code in Document Review's private copy |
| The planlens release on the critical path | Step 9 runs early, beside steps 6-8; feature detection stays one release |
| Moves collide with in-flight branches | Drain the queue before steps 10-11; pure-rename move commits; alias tables instead of stub files, so git follows the rename |
| Tests that patch module attributes stop biting | Alias tables map old names to the same module object; no re-export stubs |
| A wheel missing a package or data file (5.19.0) | The wheel test in CI for GSE and the core |
| GSE and the core released out of step | Exact `==` pins; `version_guard` names the core; the release order is fixed (planlens, the core, GSE) |
| CfA refuses a source folder | Routes B and C (1.4); testers stay on the two-page GSE from Nexus meanwhile |
| Charts or the figure helper change for one app because of the other's needs | One copy, in the core; the chart lines handed in by GSE, which owns them (steps 6, 11); the nine geotech worked examples ride along until Document Review has its own (OWN-77 (b), DR-35) |
| Old paths in notes, notebooks, the Harness Map and helper prompts | Aliases for one release; the notebook cell given anew in step 16; docs swept in step 16 |
| Session limits during the large moves (steps 10-11) | One group per merge; helpers in worktrees; targeted tests per helper, the full gate in chunks by the lead |
| Existing conversations lost | Same data root, same folder layout, same SharePoint base folder (1.3) |

### The owner's decisions

The earlier plan's D1-D9 and where they stand. Each open one is a desk
card.

- **D1, the names → OWN-72 (open).** Only the core needs a name now;
  Document Review is a folder. Working name `docreview-core` /
  `docreview_core`. The core is public, like planlens, so the name carries
  no "DE". GSE uses it as much as Document Review does, so a neutral name
  may read better. Needed before step 10.
- **D2, where the shared code lives → OWN-74, DECIDED 2026-10-21:** option
  (d), made smaller. The document tools that only wrap planlens go into
  planlens. A thin public core holds the model connections, the vision
  calls and the figure helper. Each app keeps its own chat page. Each app
  pins a core version, and the core carries its own tests and nothing
  Department-specific.
- **D3, how Tiny Apps gets Document Review → OWN-72 (open, with CfA).**
  The owner's route is a source folder in the GHE repo. Ask CfA 6a-6d
  (1.4); fall back to 6b, then 6c. Needed before step 14.
- **D4, the geotech page leaves Tiny Apps: DECIDED 2026-10-20 (a).**
  - The owner on the desk, 08:33: "GeotechStaffEngineer running in
    Palantir Foundry as an app, and DE_DocReview being a wheel that I pull
    into TinyApps", and "now that I have foundry, I'll just go that route".
  - Testers get Document Review only. GSE runs on Foundry, with Funhouse as
    backup.
- **D5, which Document Review agent moves: DECIDED by the lead (a).**
  - It is the one testers get today, unchanged and proven by the
    fingerprint, with `lean` and `minimal` behind their switches as now.
  - DR-05 decides the default afterwards, on the suite.
  - The owner can overrule.
- **D6, release what is waiting before files move → OWN-75, DECIDED
  2026-10-21:** release first, or hold if that is no longer practical,
  with no effort spent undoing work. The release point falls after step 8,
  before step 9 (b) (section 4). The plan had recommended releasing before
  the core's first move, for a clean version to fall back to and the
  pandas fix on Nexus while CfA works; step 9 (b) is now the first file to
  leave.
- **D7, should the Document Review code be private → OWN-76, DECIDED
  2026-10-21:** private, as plain source in the GHE repo, from the copy
  (step 13). The code already published stays published.
- **D8, the figure helper and plotting code → OWN-77, DECIDED
  2026-10-21:** (b). The helper, plotting box, chart stack and the nine
  geotech worked examples go into the core (steps 6 and 11 (b)); GSE hands
  in the reference chart lines; Document Review's own general examples
  (drawings, schedules, quantities) follow the split, measured on the
  suite (DR-35; 2.8).
- **D9, what testers see named → OWN-78 (open).**
  - The proposed notice: "Document Review reads drawings, specifications,
    submittals, reports and calculation packages and hands back memos and
    marked-up PDFs. It is a review aid: it can miss or misread things, and
    the judgement stays with you. Check anything you rely on against the
    document."
  - Keep the testers' SharePoint folder `Shared Documents/GeotechStaffEngineer`
    until the published app is live, so nothing moves twice.
  - Needed before step 14.
- **D10, where Document Review's private copy is worked on (NEW, open).**
  - (a) a private repository on the owner's GitHub account, carried to the
    GHE wrapper repo at each tester release;
  - (b) the GHE repo itself, if the laptop can push to it and CI can run
    there.
  - Recommendation: (a), unless (b) is available (1.5). Needed before
    step 13.
