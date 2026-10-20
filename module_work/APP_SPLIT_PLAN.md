# Splitting Document Review from GeotechStaffEngineer (plan, 2026-10-20)

BACKLOG: **INF-26** (this plan and its steps), **OWN-72** (the owner's
decisions, section 7), **RI-34** (every geotech-report question goes to
report_ingest; treated here as done), **INF-25** (GSE on Foundry). Nothing
in this document is built. It changes no code.

The owner's words (2026-10-20, Backlog Desk): "I want to divorce the two
apps. Document-Review and GeotechStaffEngineer will now be separate apps ...
make Document Review its own package ... because people will use it that
aren't geotech engineers ... I wouldn't make it a PyPI package. Then, the
production layout would look like: GeotechStaffEngineer running in Palantir
Foundry as an app, and DE_DocReview being a wheel that I pull into
TinyApps." The reason: unlimited Opus 5.5 and Sonnet 5.5 on Foundry for
the owner, GSE's only user.

## In one page (for the owner)

**What you get.**
- Document Review installs on Tiny Apps without any geotechnical code. The
  testers' install drops about a dozen packages (OpenSees, pyStrata, PyNite
  and its numba chain, GSTools, SALib, pystra, liquepy, the signal tools,
  the reference library): fewer for the Nexus quarantine to catch, a
  faster start, nothing tied to the pandas pin. It keeps the figure helper
  and plotting code, which you ruled on everywhere (GM-92).
- GeotechStaffEngineer becomes one page, the geotech one, on Foundry with
  Claude. Nothing it does today is lost.
- Each app can change without the other's users feeling it. A test that
  builds the Document Review agent with every geotech package missing
  proves the line holds.

**What it costs.** 16 small steps of code work, each merged and tested on
its own, with both apps working after every one. Three packages of our own
instead of one, so a release has one more step. Every session's notes and
paths change once (step 16).

**What is blocked.**
- **CfA's answer to open question 6**: may Tiny Apps install a wheel kept in
  the wrapper repo instead of one from Nexus? Until CfA answers, testers keep
  today's package from Nexus, which carries both pages. Nothing is lost.
- **Seven decisions left for you** (section 7). Of the nine, you answered
  D4 on 2026-10-20 (the geotech page leaves Tiny Apps for Foundry), and the
  lead has taken D5 (testers keep today's agent; you can overrule). Five of
  the seven are best answered before files start to move: the names (D1),
  where the shared code lives (D2), the Tiny Apps route (D3), releasing the
  waiting train first (D6), and where the figure helper goes (D8).

**The recommendation, in short.** Three packages of our own, built from
this one repository (working names, OWN-72):
- **de-docreview**: the Document Review app. Not on PyPI; a wheel for Tiny
  Apps.
- **docreview-core**: what both apps run (the chat page, the document and
  vision tools, the model connections). Public, on PyPI, like planlens.
- **geotech-staff-engineer**: the geotech app. On PyPI as today, for
  Foundry.

Move Document Review to its own repository later only if it must be
private.

## 1. The target layout

### 1.1 The five packages

| Package (install name / import name) | What it is | Who uses it, where | Depends on | How it is installed | On PyPI |
|---|---|---|---|---|---|
| `planlens` / `planlens` | The document layer: reading PDFs and DXFs, page maps, markups, render budgets. No model calls. | Both apps | numpy, PyMuPDF, ezdxf, opencv-headless, rapidfuzz | PyPI (Nexus mirrors it) | yes, unchanged |
| `geotech-references` / `geotech_references` | The cited reference library and chart catalogs | GSE only (today Document Review installs it too) | numpy | PyPI | yes, unchanged |
| **`docreview-core` / `docreview_core`** (NEW, working name) | What both apps run: the Streamlit chat shell and its turn jobs, the activity record, model engines (Prompter key, Foundry proxy and SDK, Funhouse, Anthropic), identity, SharePoint storage, the document, vision and markup tools, Word/Excel/chart output, the figure helper and its plotting box (D8), the agent guards, the switches, the eval framework | Both apps, every host | planlens, deepagents/langchain/langgraph (same caps as today), openai <3, langchain-openai, streamlit, matplotlib, plotly, python-docx, markdown-it-py, openpyxl, numpy, scipy (the plotting box lets code import it); `[anthropic]` extra = langchain-anthropic | PyPI for GSE; a vendored wheel for Tiny Apps (1.3) | yes (decision D2) |
| **`de-docreview` / `de_docreview`** (NEW, name OWN-72) | The Document Review app: its page, prompt, agent builds (today's agent plus `lean` and `minimal` behind their switches), reading helper, review-only tools (findings, digest, geometry, sweep), review digest, its 38 suite tasks, the Tiny Apps entry | Testers on Tiny Apps; the owner on Funhouse; the suite on Foundry | docreview-core (exact version), planlens | A wheel: vendored in the Tiny Apps wrapper repo (pending CfA), uploaded on Foundry and Funhouse | **no** (classifier `Private :: Do Not Upload`, which PyPI refuses) |
| `geotech-staff-engineer` / `funhouse_agent`, `webapp`, the module packages | The geotech app: 33 modules, dispatch, the geotech agent and its helpers (calc, references, setup, reviewers), the chart lines it hands the figure helper, report_ingest, the Foundry entry | The owner on Foundry (Funhouse as backup) | docreview-core[anthropic] (exact version), geotech-references, planlens, the analysis back ends as today | PyPI (Foundry's Libraries panel) or uploaded wheels | yes, unchanged |

How they depend on each other:

```
planlens  <--  docreview-core  <--  de-docreview          (Tiny Apps)
                     ^
                     +------------  geotech-staff-engineer (Foundry)
                                        |
geotech-references  <-------------------+
```

GSE never imports de_docreview, and de_docreview never imports GSE. Two
tests hold both lines (step 1 and step 15).

### 1.2 What stays the same for every host

So that no setting, Key Vault secret, conversation or folder is lost:
- **Setting names** stay: `PROMPTER_*`, `PROMPTER-API-KEY--<TESTER>`,
  `GRAPH_*`, `SHAREPOINT_SITE_URL`, `KV_NAME`, `DEV_IDENTITY` and every
  `GEOTECH_*` setting and switch. Renaming them is a later, separate choice
  (old names still read).
- **Folders** stay: the data root (`GEOTECH_WEBAPP_DATA`, else
  `~/.geotech_webapp`), `users/<identity>/document_review` on Tiny Apps,
  `pages/document_review` single-user, and the SharePoint base folder
  `Shared Documents/GeotechStaffEngineer` unless the owner names another
  (D9). Conversations already on dosdev show up in the new app.
- **The activity record** keeps its event names and fields
  (`activity.jsonl`, `run.json`), so the trace tools, the Harness Map and
  past TRACE_REVIEWs still read new runs.

### 1.3 How Tiny Apps gets Document Review

Tiny Apps installs from the Nexus mirror only (`tinyapps/TINYAPPS.md`,
"Packages"). The Nexus firewall exists to scan what comes in. A wheel that
is not on PyPI cannot reach Nexus by itself. Four routes:

| Route | What the wrapper repo holds | Needs from CfA | Update speed | Notes |
|---|---|---|---|---|
| **A. Vendored wheels** (the owner's choice) | `wheels/de_docreview-X-py3-none-any.whl` and `wheels/docreview_core-X-py3-none-any.whl`; `packages.txt` names them by path and takes everything else from Nexus | Yes: question 6a | Instant: copy two files, bump, push, sync | Both are pure Python. pip installs a path from a requirements file and still resolves the rest from `--index-url`. The owner moves one or two files from the laptop to dosdev. |
| B. Source in the GHE repo | The `de_docreview/` source folder (the wheel unzipped) | Probably none: this is the normal Tiny Apps shape | Instant | Reviewers at CfA can read it. The core still comes as a wheel (A) or from Nexus (D). |
| C. A hosted repository on CfA's Nexus | Only `packages.txt` | Yes: question 6c, whether one exists for teams | Upload, then sync | Keeps "everything through Nexus" if CfA requires it. |
| D. PyPI → Nexus | Only `packages.txt` | None (proven: 5.26.0 installed on dosdev this way) | A day or two (the mirror's lag) | The owner does not want Document Review on PyPI. It remains the fallback. |

**Recommendation (D3):** plan on A and ask CfA questions 6a-6d (below) now.
If CfA wants source it can read, use B (the same thing, unzipped). If CfA
requires every package through Nexus, use C where it exists, else D (the
owner decides then). Until CfA answers, testers stay on GSE's two-page
package from Nexus (5.33.0 or the next release); the geotech page leaves it
only in step 15, after the de-docreview route works.

**What to ask CfA (replaces question 6 in `tinyapps/TINYAPPS.md`):**
- **6a.** May our wrapper repo carry our own two wheel files (small, pure
  Python, built by us), installed by path from `packages.txt`, with every
  other package from Nexus? Or must every package come through Nexus?
- **6b.** If you would rather read source than a wheel, may the app's source
  folder live in the GHE repo instead? (Same code, unzipped.)
- **6c.** Does your Nexus have a hosted repository where a team publishes its
  own package?
- **6d.** The fleet pins our tree must accept: still streamlit 1.62.0,
  pandas 3.0.2, PyJWT 2.13.0, msal 1.37.0, azure-identity 1.21.0,
  azure-keyvault-secrets 4.9.0? Document Review installs beside all of them
  (it has no python-ags4 at all).

The other open CfA questions (keys, SharePoint registration, the published
address, the identity header, the image-capable deployment, FIPS, the budget
response) are unchanged.

### 1.4 How GSE gets to Foundry and Funhouse

- **Foundry (production for GSE):** `pip install geotech-staff-engineer` in
  the Libraries panel brings docreview-core with it from PyPI. Nothing new
  for the owner to type. Test wheels (the INF-25 route): upload four wheels
  in one command: GSE, docreview-core, planlens, geotech-references. The
  2-line `app.py` (`webapp.foundry_entry`) never changes.
- **Funhouse/Databricks (backup for GSE; quick tests of Document Review):**
  `run_on_databricks` moves to docreview-core but stays importable as
  `webapp.databricks_launcher`, so the owner's notebook cell keeps working.
  It boots the geotech page by default. `run_on_databricks(app="docreview")`
  boots Document Review when de-docreview is installed (its wheel uploaded to
  the workspace and pip-installed by path).

## 2. Inventory by module and file

### 2.1 How this was found

An import graph of every first-party import in `funhouse_agent/`, `webapp/`,
`calc_package/` and `report_ingest/`, including imports inside functions,
with line numbers. On top of that, two probes:

- **Importing** `funhouse_agent.deep.review_agent` (ac94ba5) loads 37
  `funhouse_agent` modules. Twelve of them are geotech: `agent`,
  `dispatch`, `adapters`, `system_prompt`, `reviewers`, `reviewer`,
  `review_checklists`, `engine`, `deep.agent`, `deep.eval_harness`,
  `deep.setup_agent` and `deep.setup_tools`. Two `geo_project` modules come
  with them.
- **Building** the Document Review agent offline with a fake model (re-run
  on 8ca1694, after GM-227 turned the figure helper and plotting code on):
  today's build loads 51 `funhouse_agent` modules, among them
  `deep.figure_helper` and `plot_box` (three modules), and 20 modules from
  four geotech packages: `geo_project`, `geotech_common` (including
  `layout_ink`, through the plotting box), `geotech_references`
  (`chart_lines`) and `profile_figure`. The `lean` build (47 modules) has
  no figure helper and loads 19.

Of the non-test Python files in `funhouse_agent/` and `webapp/`, 62 (33,685
lines) go to the core whole, 12 (6,107) to de-docreview whole, 30 (about
12,000) plus the 71 adapters stay in GSE, and 17 (21,366) split, mostly to
the core. Result: a core of about 46,000 lines (with the chart stack and the
figure examples), de-docreview about 10,000.

### 2.2 Moves to de-docreview

| File | Lines | Notes |
|---|---|---|
| `funhouse_agent/deep/review_agent.py` | 493 | The `lean` build; `REVIEW_TOOLS`, `REVIEW_DESCRIPTIONS`, the reading helper |
| `funhouse_agent/deep/minimal_agent.py` | 302 | The `minimal` build (not a candidate, kept for measurement) |
| `funhouse_agent/deep/sweep.py` | 259 | `GEOTECH_REVIEW_SWEEP` |
| `funhouse_agent/deep/findings_tools.py` | 599 | `GEOTECH_REVIEW_FINDINGS` |
| `funhouse_agent/review_findings.py` | 755 | The finding format and ledger |
| `funhouse_agent/deep/geometry_tools.py` | 1,144 | `GEOTECH_REVIEW_GEOMETRY` (planlens.ir constructs) |
| `funhouse_agent/deep/digest_tools.py` | 467 | `GEOTECH_REVIEW_DIGEST` |
| `funhouse_agent/review_digest/__init__.py`, `build.py`, `digest.py`, `references.py` | 39 + 959 + 530 + 476 | The digest layer |
| `webapp/tinyapps_entry.py` | 84 | Becomes de-docreview's one-page Tiny Apps entry |
| From `funhouse_agent/deep/prompt.py` | ~190 of 542 | `_REVIEW_*` texts, `DOCUMENT_REVIEW_PROMPT(_LEAN)`, `DOCUMENT_REVIEW_READER_PROMPT`, `build_document_review_prompt` (lines 352-536) |
| From `webapp/profiles.py` | ~150 of 283 | `DOCUMENT_REVIEW`, the orientation requests, `orientation_request_for`, `orient_on_upload`, `pdf_page_count`, `IMAGE_SUFFIXES` |
| From `funhouse_agent/deep/agent.py` | ~270 of 1,939 | Today's review-page build: the `review_page` paths (lines 983-1030 and 1325-1345, the drop at 1380-1385, the reader at 1585-1588, the hide-filesystem and page-budget middleware at 1735-1758; the figure helper it now carries, 1508-1534 and 1729-1731), extracted in step 8 as `review_build.py` |
| From `funhouse_agent/review_eval/tasks.py` | 38 of 42 tasks | The review tasks (meck, set, ufc, calc, fixture, produce, scale-plan-distance) |
| `tinyapps/TINYAPPS.md`, `tinyapps/wrapper_repo/` | | Become the de-docreview wrapper template (step 14). The gitignored `tinyapps/reference/` stays local, never committed. |

### 2.3 Goes to docreview-core (both apps use it)

The file names stay. Only the prefix changes: `funhouse_agent.X` becomes
`docreview_core.X`, and `webapp.X` becomes `docreview_core.webapp.X`. That
keeps every rename mechanical and easy to map.

Line counts in brackets.

**The chat shell and its turn jobs** (each needs the step 7 profile hooks
first): `webapp/app.py` (1,402; the geotech sidebar goes to GSE),
`core.py` (3,649; the geotech behaviour goes to GSE, see 2.5),
`turn_jobs.py` (426), `turn_record.py` (405), `output_capture.py` (117),
`ws_upload.py` (113) with `ws_upload_component/index.html`,
`budget_panel.py` (68), `feedback.py` (227), `version_guard.py` (95),
`http_trace.py` (58), `activity_log.py` (777), `diagnostics.py` (487; its
reference-library checks go to GSE). From `profiles.py`: `AppProfile`,
`current`, `set_current`, `session_root`. From `pages_entry.py`:
`app_path`, `page_runner`, `build_navigation`, now taking the page list as
an argument.

**Engines and hosts:** `webapp/engine_config.py` (616),
`tinyapps_engine.py` (662), `tinyapps_settings.py` (176),
`palantir_sdk_engine.py` (934), `databricks_launcher.py` (1,920),
`email_tools.py` (197), `funhouse_agent/deep/databricks_bridge.py` (774).

**Identity and storage:** `webapp/identity.py` (207),
`graph_sharepoint.py` (404), `sharepoint_store.py` (957),
`sharepoint_tools.py` (1,577); the run mirror `report_ingest/mirror.py`
(447; standalone; report_ingest keeps an alias).

**Agent plumbing** (`funhouse_agent/deep/`): `vision_engine.py` (826),
`busy_retry.py` (295), `pacing.py` (241), `keep_work.py` (865),
`limits.py` (424), `tool_guards.py` (436), `scratch_guard.py` (461),
`delivered_log.py` (103), `result_map.py` (134), `strict_args.py` (126),
`warm_imports.py` (56), `inline_images.py` (176), `calculate_tool.py`
(307), `coverage_tools.py` (584, less the `report_checklist` tool, which
RI-34 moves to report_ingest). Also:
- from `deep/tools.py`, `make_vision_tools` and its helpers (lines 103-404
  and 899-2187), as `deep/vision_kit.py`;
- from `deep/notebook.py`, the stream formatters `webapp/core.py` uses
  (`_format_token_line`, `_update_messages`, `_final_answer_in_update`,
  `_format_update`, `_sum_callback_tokens`);
- from `deep/agent.py`, the assembly both builds use (the per-helper guard
  loop, the general-purpose helper with its guards, the summarization
  trigger, the busy-retry and keep-work wiring).

**Vision, documents and markups** (`funhouse_agent/`): `vision_tools.py`
(5,201, less its two geotech-only tools, see 2.5), `vision_view.py`
(1,589), `image_view.py` (470), `vision_probe.py` (362), `inline_store.py`
(69), `document_tools.py` (2,446), `office_text.py` (311),
`page_numbers.py` (185), `find_like.py` (410), `markup_check.py` (1,399),
`measure_tool.py` (342), `scale_labels.py` (287), `drawing_wait.py` (64),
`dxf_notes.py` (233).

**Output:** `funhouse_agent/xlsx_writer.py` (346);
`calc_package/docx_renderer.py` (1,096; standalone; only the shared tools
use it). **The chart stack behind `plot_data`:** `profile_figure/data_plot.py`
(1,845), `figure_to_png` from `profile_figure/plotting.py`,
`geotech_common/plan_furniture.py` (132), `geotech_common/layout_ink.py`
(1,283), `get_pyplot` from `geotech_common/plotting.py`. The old paths keep
aliases; about 30 GSE module files import `layout_ink` or
`geotech_common.plotting`. The `reference_chart` option (geotech chart
lines) stays in GSE.

**The figure helper and its plotting box** (on by default on both pages
since GM-227; not in the `lean` build; D8 (a), see 2.8):
`funhouse_agent/deep/figure_helper.py` (899; its `FigureLooks`, the calc
helper's, stays in GSE), `plot_box/__init__.py` (932), `child.py` (1,189),
`figures.py` (608), and the nine worked examples
`profile_figure/figure_examples.py` (994; matplotlib and numpy only).

**Checks, records and switches** (`funhouse_agent/`): `cited_pages.py`
(1,443), `cited_documents.py` (1,482), `coverage.py` (1,240),
`progress_notes.py` (46), `answer_sections.py` (114), `error_text.py`
(396), `_fileio.py` (636), `runtime_check.py` (203), `review_flags.py`
(465; every switch for both pages; a leaf module). Two helpers out of the
adapters: the safe formula compiler `_compile_g`
(`adapters/reliability_adapter.py:98-180`), which `calculate` uses, and
`check_nested_keys` (`adapters/__init__.py`), which `annotate_document`
uses.

**The eval framework** (from `funhouse_agent/review_eval/`): `runner.py`
(744), `checks.py` (1,010), `documents.py` (272), and the `Task` type and
task selection from `tasks.py`. Each app brings its own tasks, so GSE can
still measure its own document questions.

### 2.4 Stays in GSE

**`funhouse_agent/`:** `__init__.py`, `_disclaimer.py`, `agent.py`,
`engine.py`, `native_tools.py`, `react_support.py`, `notebook.py`,
`run_local.py`, `run_scenarios.py`, `smoke_test_native.py`, `dispatch.py`,
`system_prompt.py`, `reviewer.py`, `reviewers.py`, `review_checklists.py`,
`worked_examples.py` with its `.json`, `reference_docs.py`,
`chart_reading.py`, `judgment_check.py`, `report_checklist.py` with
`report_review_checklist.json` (to report_ingest by RI-34),
`geotech_test_suite.json`, `test_scenarios.json`,
`test_scenarios_notebook.ipynb`, `eval_samples/` (5 files), `adapters/`
(all 71 files, less the two helpers in 2.3).

**`funhouse_agent/deep/`:** `__init__.py`, `agent.py` (the geotech build),
`prompt.py` (domain, judgment-call and ingest texts), `tools.py`
(`make_core_tools`, the report-ingest and library tools), `fit_results.py`,
`setup_agent.py`, `setup_tools.py`, `unknown_tool.py`,
`eval_harness.py`, `selfcheck.py`, `rc_wheel_check.py`, `run_live.py`,
`notebook.py` (`DeepNotebookChat`).

**`webapp/`:** `foundry_entry.py`, `reference_fetch.py`,
`requirements.txt`, `README.md`, the `GEOTECH` profile, and a new
`webapp/geotech_page.py` holding the geotech sidebar and build (2.5).

**Everything else:** every analysis package, `calc_package` (less the Word
renderer), `report_ingest`, `geo_project`, `profile_figure` and
`geotech_common` (less the chart stack), `validation_examples`, and the
repository's notes, desk and gates.

The four geotech suite tasks also stay: `report-extract-all`,
`report-extract-diggs`, `report-summary-vs-sheets` and `scale-log-depth`,
with `review_eval/report_fixture.py` (515). Geotech reports belong to
report_ingest (RI-34), and two of the tasks already run on the geotech
page (`tasks.py:1014`, `1073`).

### 2.5 Files that split, and how

| File | To the core | Stays in GSE / goes to de-docreview |
|---|---|---|
| `webapp/app.py` | The page, chat, uploads, conversations, storage panel, cards | Geotech: the specialist picker (lines 681-700) and the three Behavior pickers (726-760+), placeholder text (1249), via profile hooks (step 7) |
| `webapp/core.py` | Attachments, artifacts, streaming, persistence, titles, errors, model lists | Geotech: `disclaimer_text` (53-80), `_register_reference_fetcher` (676-685), the call into `build_deep_agent` (739), `_REVIEWER_BUILDERS` / `build_reviewer_agent` (793-830), `ANALYSIS_DEPTHS` / `AGENT_TYPES` / `DEFAULT_BEHAVIOR` / depth prompts / `behavior_build_kwargs` (1999-2070, 2334-2380), the judgment-call turn check (1177-1199): to `webapp/geotech_page.py` |
| `webapp/profiles.py` | `AppProfile` and its helpers | `GEOTECH` to GSE; `DOCUMENT_REVIEW` and orientation to de-docreview |
| `webapp/pages_entry.py` | The page machinery | Each app declares its own pages |
| `webapp/diagnostics.py` | Versions, engine, Prompter key, stream/tool/vision checks | Reference-library and figure-PDF checks (294-436) to GSE |
| `funhouse_agent/vision_tools.py` | Everything else | `read_reference_figure`, `view_worked_example_source` (4480-4737), `reference_read_roots` (468-499), the chart note (5034-5063): to a GSE module that registers them with the core's tool dispatcher |
| `funhouse_agent/deep/tools.py` | `make_vision_tools` and helpers | `make_core_tools` (dispatch), report-ingest and library tools |
| `funhouse_agent/deep/agent.py` | Generic assembly (2.3) | The geotech build; the review-page build to de-docreview |
| `funhouse_agent/deep/prompt.py` | nothing | Domain prompt stays; review texts to de-docreview |
| `funhouse_agent/deep/notebook.py` | Stream formatters | `DeepNotebookChat` |
| `funhouse_agent/deep/coverage_tools.py` | The coverage kit | `report_checklist` tool to report_ingest (RI-34) |
| `funhouse_agent/review_eval/*` | Framework (2.3) | Tasks split (2.2, 2.4) |
| `webapp/README.md`, `funhouse_agent/DESIGN.md` | Each package gets its own README / DESIGN | |

### 2.6 Where each shared piece lives

| Shared piece | Home | Why not the alternatives |
|---|---|---|
| Vision engine (model side calls, busy ladder, pacing) | docreview-core | planlens makes no model calls and must stay framework-neutral; two copies would split every VIS-* fix. |
| View sizing (`vision_view.py`, `image_view.py`: page views, tiles, box conversions) | docreview-core now; **planlens later** | It is model-free page geometry, and planlens already owns render budgets (`Document.render(budget=)`). Moving it would add a planlens release to the critical path, so it waits until the split is done (a later planlens item). |
| Prompter / Tiny Apps engine (`tinyapps_engine.py`, `tinyapps_settings.py`) | docreview-core | Document Review's production engine. GSE uses it on Funhouse with a key (`run_on_databricks(prompter_key=...)`). Small, and loaded only when used. |
| Identity (`identity.py`) | docreview-core | Multi-user on Tiny Apps, single-user elsewhere; the shell reads it on every run. |
| SharePoint store and tools | docreview-core | Both apps mirror conversations; GSE on Foundry simply has none set up. |
| Streamlit chat shell and turn jobs | docreview-core, with profile hooks | The largest shared piece (about 12,000 lines). A copy would double every fix to persistence, turns and cards. |
| Activity logging (`activity_log.py`, `turn_record.py`, `http_trace.py`) | docreview-core | One record format for both apps keeps the trace-reading tools working. |
| Markup check (`markup_check.py`) | docreview-core | `annotate_document` is on both pages. |
| `write_docx` / `write_xlsx` | docreview-core (the Word renderer leaves `calc_package`) | Standalone; only the shared tools call it. |
| `plot_data` chart stack | docreview-core | It is on the review page by default (live wave 1, A12). The alternatives: a copy (geotech figure fixes would stop reaching it) or dropping it. GSE's figure conventions (GM-97) then also apply to Document Review's charts; the full CI job runs both apps' tests on every change. |
| Figure helper and `plot_code` box | docreview-core (D8 (a)) | On for both pages since GM-227 (owner on GM-92: "on everywhere"). General machinery; GSE hands in the chart lines; the examples stay with it for now. Options measured in 2.8. |
| Coverage, cited pages and cited documents | docreview-core | Both pages. Their geotech touches become hooks: `coverage.py:818` counts `write_diggs` records and `cited_documents.py:951` names reference modules, both through `try`. |
| Switches (`review_flags.py`) | docreview-core | A leaf module that both read. |

### 2.7 The hard couplings

**Review code into geotech code.** Each item says what breaks.

1. **`webapp/core.py:739`**: every Document Review agent is built by GSE's
   `funhouse_agent.deep.agent.build_deep_agent`. Today's review build lives
   inside it (`deep/agent.py:1325-1345`, `1377-1385`, `1585-1588`,
   `1735-1758`), and `lean` and `minimal` are reached from it.
2. **`funhouse_agent/deep/tools.py:40`**: the review tools' factory
   `make_vision_tools` sits in the module that imports geotech dispatch on
   load. `deep/review_agent.py:56` and `deep/minimal_agent.py:43` import it.
3. **`funhouse_agent/__init__.py:24-33`** and
   **`funhouse_agent/deep/__init__.py:26`**: importing any review module
   first loads the v1 geotech agent, dispatch, the reviewers, the geotech
   deep builder and the setup agent, and through `deep/setup_tools.py:40-41`
   `geo_project`.
4. **`funhouse_agent/deep/tools.py:1852`**: `plot_data` runs through
   geotech dispatch (`_call_agent("profile_figure", "plot_data")`) into
   `profile_figure.data_plot`. That module needs `geotech_common`
   (`data_plot.py:900`, `1409`, `1568`), and `tools.py:202`, `313` and
   `2096` read the adapter. With the reference library installed it also
   offers `reference_chart`.
5. **`funhouse_agent/deep/prompt.py:16`**: the review prompt's module imports
   the geotech system prompt, which imports the module registry
   (`system_prompt.py:6`).

Further couplings: `deep/calculate_tool.py:161` (`calculate` uses
`reliability_adapter._compile_g`); `vision_tools.py:4762` and
`office_text.py:113` (`write_docx` and reading `.docx` use
`calc_package.docx_renderer`); `document_tools.py:1711`
(`adapters.check_nested_keys`); `webapp/core.py:62` with `app.py:410` (the
review page shows GSE's geotechnical disclaimer); `webapp/profiles.py:178`
(the review prompt from `deep/prompt.py`); `review_eval/runner.py:83`
(`webapp.profiles.GEOTECH`) and `:659` (`report_ingest.mirror`);
`review_eval/documents.py:155`, `160` (the GSE reference fetcher);
`webapp/diagnostics.py:303`, `319`, `369`, `385` (reference library).

The figure helper and plotting box, on the review page since GM-227:
`deep/figure_helper.py:339` (`deep.agent.DEFAULT_MAX_RESULT_CHARS`),
`:342` (`deep.tools.make_vision_tools`), `:210`, `221`, `249`, `433`,
`464` (`profile_figure.figure_examples`), and `:813`
(`dispatch.parameters_object`, inside `FigureLooks`, which only the calc
helper uses, so it stays in GSE); `plot_box/child.py:686-693`, `959-966`
(`geotech_common.layout_ink` and `plan_furniture`, loaded by file).

Soft couplings (inside `try`; the review page works without them but
behaves differently): `vision_tools.py:482`, `4626` (`geotech_references`),
`coverage.py:818`, `cited_documents.py:951`, `plot_box/__init__.py:821` and
`plot_box/child.py:980-989` (the box's `chart_lines`, from
`geotech_references`).

**Geotech code into review code** (steps 8 and 15 cut these):
`deep/agent.py:987`, `1003-1004`, `1026`, `1336` and `1750` import
`deep/review_agent.py` and the reader prompt, and `:1328` imports
`minimal_agent`; `webapp/pages_entry.py:149-154` puts the review page in
GSE's navigation; `webapp/databricks_launcher.py:516-520` boots the
two-page entry by default; `review_eval/runner.py:83-96` builds GSE's pages
through `webapp.core`.

### 2.8 The figure helper and plotting box: three options measured (D8)

Since GM-227 (b4b9a08) `GEOTECH_FIGURE_HELPER` and `GEOTECH_PLOT_CODE` are
on by default on both pages (the owner on GM-92: "on everywhere"). The
machinery is general (draw with `plot_data` or code in the locked box, look
at the picture, redraw to a checklist). Its geotech content: the standard
chart lines from the reference library, nine worked examples (a boring
plan, an SPT profile, wall pressures and the like), and a few checklist
lines (elevation axis, kPa and kN, "the FIGURE HELPER of a geotechnical
agent").

| | (a) To the core; GSE hands in the chart lines | (b) As (a), then Document Review's own examples | (c) Document Review drops both |
|---|---|---|---|
| Files to the core | `deep/figure_helper.py` (899, less `FigureLooks` at 784-899, the calc helper's, which stays in GSE), `plot_box/__init__.py` (932), `child.py` (1,189), `figures.py` (608), `profile_figure/figure_examples.py` (994): about 4,500 lines | The same | None; all stay in GSE |
| Imports cut | `figure_helper.py:339` → `deep/limits.py`; `:342` → the vision kit (step 4); `:210-464` → the core's examples; `child.py:686-693`, `959-966` → the core's chart folder (step 6); `plot_box/__init__.py:817-864`, `child.py:975-990` → a chart-lines source GSE registers; `FigureLooks` and its `dispatch` import (`:813`) stay in GSE | The same, plus an examples-and-checklist registration each app fills | The review build leaves out the helper (`deep/agent.py:1508-1534`), its main-agent middleware `FigureTasks` (`:1729-1731`) and `plot_data_replaces` (`:925`) |
| New code | One hook (the chart lines) | The hook, a de-docreview examples file and checklist lines, a suite task, a live wave | None |
| Document Review's fingerprint | Unchanged, except the box's chart-lines note goes when the reference library does (step 12) | Changes (helper prompt and examples): measured on the suite | Loses the figure helper and `plot_code` |
| Install | The core keeps SciPy: the box lets code import it (`child.py:61-67`) | The same | SciPy can go |
| Owner | Fits GM-92 | Fits GM-92 | Reverses GM-92 |

**Recommendation: (a) during the split, (b) after it** as its own item
(the lead files the ID): one change at a time, as with D5. Before
GM-227 this plan assumed both switches were off: it left both helpers in
GSE (2.4, 2.6, D8) and dropped SciPy from the testers' install. Those are
corrected, and the switches' effect is now carried in 1.1 (the core's
contents, SciPy), 2.1 (the build probe), 2.3, 2.7, steps 5, 6, 8, 11 and
12, and the test table in section 5.

## 3. The repo question

| | A sub-package in this repo, built as its own wheel | A new repository |
|---|---|---|
| Shared code changes | One change, one CI run, both apps tested together | Two pull requests and a version bump in between |
| While the line is being drawn (steps 1-15) | Cheap: a move is a rename, and in-flight branches follow it | Each move is an extract-and-publish |
| History | Kept | Kept only with a history-filtering copy |
| Sessions and helpers | One checkout, the existing worktree routine | Two checkouts; a helper cannot test both sides |
| Privacy | Public (this repo is public) | Can be private |
| What a CfA or Department reviewer sees | A large repo (the "millions of tokens" remark, `project_repo_slimming`) | A small, readable repo |

**Recommendation (D7):** build the line inside this repository first, as
`packages/docreview_core/` and `packages/de_docreview/`, each with its own
`pyproject.toml` and `tests/`; GSE's `pyproject.toml` stays at the root.
After step 16, move `packages/de_docreview/` to its own repository only if
the Document Review code must be private or CfA's review wants a small
repo. Once the line holds that move is mechanical (a history-filtered copy
of one folder; the wrapper changes nothing). The core stays here either
way: GSE and the core change together most often.

**The public-repo rules carry over.**
- No Department material in any of the three packages: no guides, no
  internal host names, no tester documents, no CfA mailboxes.
  `webapp/tests/test_public_text.py` reads only the root `pyproject.toml`
  today; step 9 makes it walk all three and their docs (with INF-29).
- The GHE wrapper repo holds the deployment's own values (the published
  address, the Key Vault name).
- de-docreview carries the classifier `Private :: Do Not Upload`, so PyPI
  refuses it even by mistake. But "not on PyPI" does not make the code
  private while it lives in this public repo (it has been public in every
  GSE release since 5.26). If privacy is the point, see D7.

## 4. Steps in order

**Rules for every step.** One merge to master, small enough for one helper
and one reviewer. Both apps work after it: the geotech page on Foundry and
Funhouse, and Document Review through GSE's two-page entry until step 14
(through de-docreview after). Its targeted tests and its exit check pass.
Heavy suites never run two at a time. The notes checklist runs after each
step (BACKLOG, HANDOFF, CLAUDE.md state clause, memory, desk).

**Three techniques used throughout.**
- **The fingerprint** (step 1) of each page's agent with default switches:
  what the model is sent on its first call (the system prompt; each tool's
  name, description and argument schema), each helper's name, prompt and
  tools, and the middleware classes in order. Steps marked *(no behaviour
  change)* leave every fingerprint identical. A deliberate change
  re-records it in its own commit, with the reason.
- **The boundary test** (step 1): build and run the Document Review agent in
  a child process with every geotech module blocked, recording each blocked
  import as a known coupling. The list may only shrink; from step 8 it is
  empty. From step 12 a CI job runs with the geotech packages not installed
  at all.
- **Moves that git can follow.** A move commit holds `git mv` plus
  import-line rewrites only. Old import paths keep working through an alias
  table the old package installs (`sys.modules` maps the old name to the
  same module object), with no stub files at the old paths. So an in-flight
  branch that edited a moved file merges onto its new path, and
  `monkeypatch.setattr("funhouse_agent.vision_tools.x", ...)` still patches
  the real module, which a re-export stub would silently not do. The
  aliases come out in step 16, one release later.

### Step 0. Decisions and preconditions (no code)
- The owner answers D1-D3 (names, where shared code lives, the Tiny Apps
  route), D6 and D8; D7 and D9 can follow. D4 is answered (2026-10-20) and
  D5 is the lead's default. CfA gets questions 6a-6d.
- RI-34 is merged: the checklist and its tool are in report_ingest.
- The waiting train is released (OWN-55, D6) before step 9, so there is a
  clean fallback version and the pandas-3 fix is on Nexus for testers.
- In-flight branches that touch `webapp/` or the shared `funhouse_agent`
  files are merged or parked before the move steps (9-12).
- **Exit:** the decisions are folded into BACKLOG; RI-34 is in the "Done"
  list.

### Step 1. The line, measured (tests only, no behaviour change)
- **Change:** add `webapp/tests/test_review_boundary.py` (the boundary test
  and its known couplings, 2.7) and `webapp/tests/test_page_fingerprints.py`
  (the review page's three builds and the geotech page), plus a reverse
  check that GSE's own builders import nothing from the review files (it
  gains teeth as they move).
- **Tests:** the two new files; `webapp/tests/test_app_profiles.py`;
  `funhouse_agent/deep/tests/test_review_harness_offline.py`.
- **Exit:** both pass on master; the known-couplings list matches 2.7.

### Step 2. Lazy package doors (no behaviour change)
- **Change:** `funhouse_agent/__init__.py` and `deep/__init__.py` load their
  public names on first use (module `__getattr__`) instead of on import.
  Every name stays importable.
- **Tests:** `funhouse_agent/tests`, `funhouse_agent/deep/tests`, the boundary
  test.
- **Exit:** fingerprints identical. The couplings list loses the v1 agent,
  reviewers, setup agent and `geo_project`.

### Step 3. The review prompt in its own module (no behaviour change)
- **Change:** the review texts (`deep/prompt.py:352-536`) move to
  `funhouse_agent/review_prompt.py`, which imports nothing geotech;
  `deep/prompt.py` re-exports them. `webapp/profiles.py:178` and
  `deep/review_agent.py:52` import the new module.
- **Tests:** `deep/tests/test_review_prompt_rules.py`, the fingerprints.
- **Exit:** identical prompts; coupling 5 is gone.

### Step 4. The vision kit out of the dispatch module (no behaviour change)
- **Change:** `make_vision_tools` and its helpers (`deep/tools.py:103-404`,
  `899-2187`) move to `funhouse_agent/deep/vision_kit.py`, which must not
  import dispatch at load; `deep/tools.py` re-exports. `read_reference_figure`
  and `view_worked_example_source` (and their dispatchers in
  `vision_tools.py`) are added by a GSE registration instead of being built
  in, so the review page never gets them and the `GEOTECH_ONLY_TOOLS` filter
  goes. `plot_data` calls the chart function directly instead of through
  `_call_agent` (still in `profile_figure` until step 6).
- **Tests:** `deep/tests` (the vision, review and geotech-build files);
  `funhouse_agent/tests/test_vision_*`, `test_find_like_*`.
- **Exit:** all four fingerprints identical; coupling 2 is gone.

### Step 5. Small helpers to neutral modules (no behaviour change)
- **Change:** `check_nested_keys` → `funhouse_agent/arg_checks.py` (the
  adapters import it from there); `_compile_g` → `funhouse_agent/safe_expr.py`
  (the reliability adapter and `calculate` import it);
  `calc_package/docx_renderer.py` → `funhouse_agent/docx_renderer.py` and
  `report_ingest/mirror.py` → `funhouse_agent/run_mirror.py` (the old paths
  alias them); `DEFAULT_MAX_RESULT_CHARS` → `deep/limits.py` (the figure
  helper reads it, `figure_helper.py:339`); `FigureLooks` and
  `CALC_REDRAW_NOTE` (the calc helper's) out of `figure_helper.py` into a
  GSE module.
- **Tests:** `funhouse_agent/adapters` tests for reliability; `deep/tests`
  for `calculate`; `funhouse_agent/tests/test_read_office_files.py`; the
  `write_docx` tests; `report_ingest/tests` for the mirror;
  `review_eval/tests`.
- **Exit:** fingerprints identical; those couplings are gone.

### Step 6. The chart stack to a neutral package (no behaviour change)
- **Change:** a new `funhouse_agent/charts/` package holds `data_plot.py`,
  `layout_ink.py`, `plan_furniture.py`, `figure_to_png` and `get_pyplot`,
  with aliases at `profile_figure.data_plot`, `geotech_common.layout_ink`,
  `geotech_common.plan_furniture` and `geotech_common.plotting`.
  `reference_chart` stays in the GSE adapter, which wraps the chart
  function. The figure helper's worked examples
  (`profile_figure/figure_examples.py`) join the package (D8 (a)). The
  plotting box loads `layout_ink` and `plan_furniture` by file
  (`plot_box/child.py:680-700`, `955-970`) and follows the move; its chart
  lines (`child.py:975-990`, `plot_box/__init__.py:817-864`) come from a
  source GSE registers instead of a look for `geotech_references`.
- **Tests:** `profile_figure`, `geotech_common` and every module whose
  figure tests import `layout_ink`, run in chunks;
  `deep/tests/test_figure_*`; `funhouse_agent/tests/test_plot_box.py`;
  `webapp/tests/test_figure_code_cards.py`.
- **Exit:** fingerprints identical; coupling 4 is gone except
  `reference_chart` and the box's chart lines (GSE's).

### Step 7. The shell asks the page, not geotech (no behaviour change)
- **Change:** `AppProfile` gains hooks: `disclaimer`, `placeholder`,
  `sidebar_extras`, `behavior_kwargs`, `agent_builder`, `on_build` (the
  reference fetcher), `turn_checks` (the judgment-call check),
  `diagnostics_checks`, `version_dists`. The geotech branches of
  `webapp/app.py`, `core.py` and `diagnostics.py` (2.5) move into
  `webapp/geotech_page.py` and fill the `GEOTECH` profile's hooks. The
  review profile fills its own: no specialists, its builder (still
  `build_deep_agent(review_page=True)` here), and its own notice instead of
  the geotech disclaimer (D9; the current text stays until the owner
  approves the wording, so nothing changes yet).
- **Tests:** `webapp/tests` (all), including `test_app_smoke.py` for both
  pages.
- **Exit:** fingerprints identical. `app.py`, `core.py` and `diagnostics.py`
  import nothing from `funhouse_agent.dispatch`, `reviewers`,
  `_disclaimer`, `reference_docs`, `worked_examples` or
  `geotech_references`.

### Step 8. The review page builds itself (no behaviour change)
- **Change:** today's review build is extracted from `build_deep_agent`
  into `funhouse_agent/review_build.py` (`build_review_page_agent`); the
  assembly it shares with the geotech build moves to
  `funhouse_agent/deep/assembly.py`. The review profile's builder and
  `review_eval.runner.build_page_agent` call it. It carries the figure
  helper and its main-agent middleware as today (`deep/agent.py:1508-1534`,
  `1729-1731`). `build_deep_agent` loses its `review_page` branches and its imports of `review_agent` and
  `minimal_agent`; `review_page=True` delegates for one release, for old
  callers.
- **Tests:** the fingerprints (this is the step they exist for); `deep/tests`;
  `review_eval/tests`; `webapp/tests/test_app_profiles.py`.
- **Exit:** fingerprints identical. **The boundary test's couplings list is
  empty for all three review builds**, apart from the soft couplings named
  in 2.7, which now go through hooks.

### Step 9. The core package, empty, with its build and CI
- **Change:** `packages/docreview_core/pyproject.toml` (`docreview-core`,
  version 1.0.0.dev0, the dependency list of 1.1, the `[anthropic]` extra)
  and a `docreview_core/__init__.py` holding only the alias helper. GSE's
  `pyproject.toml` depends on `docreview-core[anthropic]==` the same version;
  CI installs both editable. `test_public_text.py` and the other tests that
  read `pyproject.toml` (`test_docs_currency`, `test_version_guard`,
  `test_tinyapps_entry`, `test_ws_upload`; `funhouse_agent/tests/test_disclaimer.py`,
  `test_drawing_ir_adapter.py`; `deep/tests/test_coverage_offline.py`) learn
  about the new file. The publish workflow builds named directories only
  (section 5).
- **Tests:** a new `webapp/tests/test_wheels.py`. It builds both wheels,
  installs them into a fresh venv, imports the entry points, and lists the
  files: no `tests/`, no `module_work`, no `raw/`, no `tinyapps/reference`.
- **Exit:** the full gate in chunks, plus the wheel test, on the exact tree.

### Step 10. The shell, engines, identity and storage move to the core
- **Change:** the `webapp/` files of 2.3 move to
  `packages/docreview_core/docreview_core/webapp/` (pure renames and
  import rewrites). The `webapp` package keeps an alias table, so
  `webapp.databricks_launcher.run_on_databricks`, `webapp.core` and the
  rest still import.
- **Tests:** `webapp/tests` (unchanged, through the aliases), the
  boundary and fingerprints, the wheel test.
- **Exit:** the full gate; Streamlit runs the geotech page and both pages
  locally; `run_on_databricks` imports from the old path.

### Step 11. Tools, vision and plumbing move to the core
- **Change:** the `funhouse_agent/` and `deep/` files of 2.3 move, with
  `vision_kit.py`, `arg_checks`, `safe_expr`, `docx_renderer`, `run_mirror`,
  `charts/`, the figure helper, `plot_box/` and the eval framework. Two
  merges: (a) `deep/` plumbing and engines; (b) tools, vision, checks,
  charts, the figure helper and plotting box, and the eval framework.
- **Tests:** `funhouse_agent/tests`, `deep/tests`, `review_eval/tests`,
  `review_digest/tests`, the module suites that import the charts.
- **Exit:** the full gate; a fresh venv with only `docreview-core` and
  `planlens` imports every core module, and the core's own tests pass
  there.

### Step 12. The de-docreview package
- **Change:** `packages/de_docreview/pyproject.toml` (`de-docreview`;
  depends on `docreview-core==` the same version and planlens; classifier
  `Private :: Do Not Upload`). The files of 2.2 move in: the review agents,
  the review build, the prompt, the page profile and orientation, the
  review-only tools, the digest, the 38 tasks. It gets its own one-page
  entry, `de_docreview.tinyapps_entry.main()`. GSE's two-page entry takes
  the review page from de-docreview when it is installed and shows only the
  geotech page when it is not.
- **Tests:** de-docreview's own suite (the six `deep/tests/test_review_*`
  files, the findings tests, `review_digest/tests`, the task tests of
  `review_eval/tests`, `test_tinyapps_entry.py`, the boundary test, the
  review fingerprints); the new CI job "docreview-only" (section 5).
- **Exit:** the docreview-only job passes with no GSE, no
  geotech-references and no analysis back end installed, on Python 3.11 and
  3.12 with CfA's fleet pins (pandas 3.0.2, streamlit 1.62.0); the full job
  passes; the wheel test covers three wheels.

**The one deliberate fingerprint change.** It comes here, when Document
Review stops installing geotech-references: `plot_data` loses its
`reference_chart` argument, and `plot_code` loses `chart_lines()` and its
note in the description (both are the standard geotech chart lines).

### Step 13. The suite moves with the page
- **Change:** `score_review_suite` and the task list live in de-docreview,
  the framework in the core, and the four geotech tasks in GSE (2.4), run
  through the same framework against the geotech page. Version reporting
  (`runner.py:380-440`) names the right distributions. The Foundry AI FDE
  brief for suite runs (`../foundry_handoff/`) changes to three wheels:
  planlens, docreview-core, de-docreview. Results folders and the blind
  half (DR-24, off the repo) are unchanged.
- **Tests:** `review_eval/tests` in both places; a dry run
  (`dry_run=True`) of the full suite listing all 42 tasks across the two
  apps.
- **Exit:** one baseline-arm suite run on Foundry with the new wheels,
  matching the last baseline task for task (token use within noise). Its
  full trace read is the owner's rule; the owner or the FDE runs it.

### Step 14. The Tiny Apps wrapper for de-docreview
- **Change:** `tinyapps/wrapper_repo/` becomes the de-docreview template:
  `app.py` calls `de_docreview.tinyapps_entry.main()`; `packages.txt` holds
  the vendored wheels (route A) or a Nexus pin (C/D), planlens, the fleet
  pins with **pandas==3.0.2 restored**, and the Key Vault libraries;
  `run.sh` is unchanged; `.env.example` and the README are revised. In
  `tinyapps/TINYAPPS.md` the dosdev recipe loses "switch to the geotech
  page" and question 6 becomes 6a-6d.
- **Tests:** `webapp/tests/test_tinyapps_entry.py` (moved) runs the
  wrapper's `app.py` in a fresh venv built only from `packages.txt` (WSL,
  Python 3.11 and 3.12; `reference_pinned_version_testing`); Streamlit
  headless start with the "No model configured" banner.
- **Exit:** the owner's dosdev round passes (OWN-29, rewritten): install
  from `packages.txt`, start, sign-in line, diagnostics, upload,
  orientation, a marked-up copy and a Word memo, storage sync. On the
  published app, the first-hour check (OWN-30) follows CfA.

### Step 15. GSE lets go of the review page
- **Only after** the testers' route works (step 14's exit), or the owner
  says so.
- **Change:** `DOCUMENT_REVIEW` leaves GSE, and so does GSE's
  `tinyapps_entry.py` (GSE is not on Tiny Apps, D4). `pages_entry` and
  `run_on_databricks` default to the geotech page, with `app="docreview"`
  booting de-docreview when it is installed. The `review_page` delegate and
  the review aliases in `funhouse_agent` go. GSE's README and disclaimer no
  longer describe a document-review page.
- **Tests:** the reverse boundary test (GSE imports nothing from
  `de_docreview` anywhere, and builds with it uninstalled); the geotech
  fingerprint; the full job.
- **Exit:** a GSE run on Foundry from the new wheels: diagnostics, a short
  question, a calc package, a figure (INF-25's list). Then a release: GSE,
  docreview-core and the de-docreview wheel, in that order after planlens.

### Step 16. Aliases out, notes in
- **Change:** one release after step 15, the alias tables in `webapp` and
  `funhouse_agent` are removed (the owner's notebook cell then imports
  `docreview_core.webapp.databricks_launcher`; give the new cell). Docs
  updated: the CLAUDE.md state block and module table, HANDOFF §0a-current,
  `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md`, each package's README and
  DESIGN, `module_work/harness_theory/` paths, the Harness Map (`build.py
  --summaries-todo`), memory. BACKLOG items gain the package they belong to
  (section 6).
- **Optional:** move `packages/de_docreview/` to its own repository (D7).
- **Tests:** the three doc gates (`test_backlog`, `test_docs_currency`,
  `test_public_text`); the full job.
- **Exit:** no import of an old path remains (`grep`), and CI is green on
  both jobs.

## 5. Tests and CI

### Which suites go where

The rule: a test goes with the module it tests. A test that builds the
geotech agent or calls a module stays in GSE. A test that builds the review
page's agent goes to de-docreview.

| Today | To the core | To de-docreview | Stays in GSE |
|---|---|---|---|
| `webapp/tests` (54 files) | Most: activity log, turn jobs, cards, sharepoint, identity, engines, launcher, upload, version guard, diagnostics base, `test_figure_code_cards` (both pages show figure code) | `test_tinyapps_entry`, `test_orientation_overview`, the review half of `test_app_profiles` | Repository gates (`test_backlog`, `test_docs_currency`, `test_public_text`, `test_wheels`, the boundary and fingerprint tests); `test_judgment_check_turn`; the geotech half of `test_app_profiles` |
| `funhouse_agent/tests` (104 files) | Vision, find_like, cited pages and documents, coverage, office files, markup, measure, scales, DXF notes, xlsx, the plotting box (`test_plot_box`, `test_plot_box_drafts`) | Findings ledger tests | Dispatch, adapters, reviewers, worked examples, system prompt, disclaimer, chart reading, v1 agent |
| `funhouse_agent/deep/tests` (65 files) | Busy ladder, pacing, keep-work, limits, guards, vision engine, strict args, inline images, calculate, the figure helper (`test_figure_helper`, `test_figure_runs`, `test_figure_turn`) | `test_review_harness_offline`, `test_review_m1_offline`, `test_review_minimal_offline`, `test_review_geometry_offline`, `test_review_digest_tools_offline`, `test_review_prompt_rules` | The geotech build, calc / references / setup helpers (with `test_deep_calc_figures_offline`), eval harness, coverage gate on the geotech page |
| `review_eval/tests` (4 files) | Framework parts | Task parts | Geotech-task parts |
| `review_digest/tests` (1 file) | | all | |

The repository gates stay at the root, because they guard the repository,
not a package.

### CI (`.github/workflows/`)

- **"Tests" (today's job, kept):** both Python versions, everything
  installed editable, everything run. Any change to the core is tested
  against both apps here.
- **"docreview-only" (new, fast, minutes):** Python 3.11 (dosdev's 3.11.9)
  and 3.12 (App Service). It installs planlens from PyPI, docreview-core
  and de-docreview, and CfA's fleet pins. It does **not** check out the
  geotech-references submodule or install any analysis back end. It runs
  the core and de-docreview suites, the boundary test and the review
  fingerprints. This is the job that proves the testers' install.
- **"Wheels" (new, or part of "docreview-only"):** builds the three wheels
  and installs each into a clean venv. It imports the entry points, checks
  file lists, and checks de-docreview's private classifier. This is the
  5.19.0 lesson: a wheel can build and simply lack a package.
- **"Publish":** tags `v5.*` build the root (GSE); tags `core-v*` build
  `packages/docreview_core`; nothing builds or uploads
  `packages/de_docreview`, and a test checks the workflow names no such
  path. The de-docreview wheel is built locally (or as a CI artifact while
  the code is public) for the owner to carry to dosdev.
- The open CI items carry on: INF-24 (Python 3.11) is covered for
  Document Review by the new job; INF-28 (numpy 2.5) still applies to GSE.
  The `ci/speed` work (worktree `ci`) should land before step 9 so the two
  don't collide.

### The local release gate

Two chunks are added, "core" and "de-docreview", each run in its own venv
like the CI job. As now, no two heavy chunks run at once (15.5 GB laptop).

### The suite (review_eval) after the split

The framework is in the core, the 38 review tasks in de-docreview, the
four geotech tasks in GSE. A Document Review suite run installs planlens,
docreview-core and de-docreview only, so every run also proves the
testers' install. The live smoke harness (`module_work/live_smoke/`) stays
here and drives either app's page (its waves move if de-docreview ever
gets its own repository).

## 6. What changes for the Tiny Apps tester deploy and for Foundry

### Tiny Apps (TA-*, the CfA owner items)

**Nothing changes for testers until step 14.** Today's route (GSE 5.33.0
or the next release from Nexus, two pages) keeps working. From step 14 the
wrapper installs de-docreview: one page, no geotech packages.

| Item | After the split |
|---|---|
| OWN-29 install + dosdev round | Becomes "install the de-docreview wrapper on dosdev" (step 14's exit). The pandas 3.0.2 pin returns: Document Review has no python-ags4. |
| OWN-30 first-hour check | Document Review only; drop step 4 of the recipe (the geotech page) |
| OWN-43-OWN-48, OWN-50, OWN-51 (keys, SharePoint, origin, identity, image deployment, FIPS, budget response, metering) | Unchanged: same settings, same engine code (now in the core) |
| OWN-49 (Nexus vs vendoring; fleet pins; resource envelope) | The first two become CfA 6a-6d (D3); the fem2d / Monte Carlo resource question drops with the geotech page |
| TA-03, TA-04, TA-05, TA-06 (live GPT, unverified live, SDK-free spend line, budget-spent test) | de-docreview (TA-06's code in the core's engine) |
| TA-07, TA-08, TA-10, TA-14, TA-16, TA-18, TA-23, TA-24, TA-28 (shell, uploads, cards, helpers, busy ladder, kept work, 401 retry) | The core: fixed once, for both apps |
| TA-27, TA-13 (Funhouse launcher live check, metering) | The core's launcher; whichever app is launched |
| TA-09, TA-15 (email tool, old live checks) | Already proposed for dropping (OWN-69) |
| TA-11 (Plotly PNG twin for `subsurface.plot_*`) | GSE |
| Document Review items (DR-*) | de-docreview, except: DR-11 (recompute from a report needs the modules) → GSE; DR-20, DR-21, DR-22 → report_ingest with RI-34; DR-23 splits (core coverage; its report_ingest part → GSE) |
| Open GM-* items on the figure helper and plotting box | The core; their chart-lines parts GSE |
| VIS-* | The core, except VIS-06, VIS-10 and VIS-17 (reference chart reading and catalogs) → GSE |
| EV-* (suite measurement) | de-docreview, except the geotech-task arms |

The CfA approval path: the SIA covers the code and its connections; the
connections are unchanged (Prompter, Key Vault, SharePoint by app
registration) and the dependencies shrink. Ask CfA whether "our wheel in
the repo" instead of "one package from Nexus" needs a new SIA or only a
PCR. The App Services sync is unchanged: push, then ask for a sync.

### Foundry (INF-25 and after)

- **The GSE app on Foundry is the geotech page alone, as it already is**
  (`webapp.foundry_entry`). The split changes its imports, not its
  behaviour; the geotech fingerprint guards that at every step.
- **Install:** PyPI brings docreview-core automatically. The INF-25
  test-wheel route uploads one more file (four wheels, one command); INF-25's
  own result says whether Foundry takes uploaded wheels at all.
- **INF-27** (a Claude leg for `palantir_models`) is work in the core's
  `palantir_sdk_engine.py`; Document Review's suite runs on Foundry gain it
  too.
- **INF-29** (host names in public docs): the extended public-text guard
  (step 9) covers the new packages' docs from the start.
- **What GSE loses:** the Document Review page and its switches. GSE keeps
  every document and vision tool it has (they are in the core). The owner
  reviews other documents in de-docreview, on Funhouse, or on Foundry with
  its wheel installed beside GSE (a one-line `de_docreview.foundry_entry`
  is cheap if wanted; not planned).

## 7. Risks, and the owner's decisions

### Risks

| Risk | What guards it |
|---|---|
| Testers' page changes without anyone choosing it | Fingerprints at every step; one baseline suite run on Foundry at step 13; the only deliberate change is listed (step 12) |
| A Document Review wheel that quietly needs a geotech package | The boundary test (step 1 on) and the docreview-only CI job with nothing geotech installed (step 12 on) |
| Moves collide with in-flight branches | Drain the queue before steps 9-12; pure-rename move commits; alias tables instead of stub files, so git follows the rename |
| Tests that patch module attributes stop biting | Alias tables map old names to the same module object; no re-export stubs |
| A wheel missing a package or data file (5.19.0) | The wheel test in CI for all three |
| de-docreview uploaded to PyPI by mistake | `Private :: Do Not Upload` (PyPI refuses it); the publish workflow names only two directories, checked by a test |
| GSE and the core released out of step | Exact `==` pins between the three during the monorepo phase; `version_guard` names all three; the release order is fixed (planlens, core, GSE, de-docreview) |
| Department material in a public package | `test_public_text` over all three packages and their docs; deployment values only in the GHE wrapper |
| CfA refuses vendored wheels | Routes B-D (1.3); testers stay on the two-page GSE from Nexus meanwhile |
| Charts or the figure helper change for one app because of the other's needs | One copy, both apps' tests on every change; geotech content (chart lines, later the examples, D8 (b)) handed in by the app that owns it (steps 6, 11) |
| Old paths in notes, notebooks, the Harness Map and helper prompts | Aliases for one release; the notebook cell given anew in step 16; docs swept in step 16 |
| Session limits during the large moves (steps 10-11) | One package family per merge; helpers in worktrees; targeted tests per helper, the full gate in chunks by the lead |
| Existing conversations lost | Same data root, same folder layout, same SharePoint base folder (1.2) |

### The owner's decisions

Each can become a desk card, and each says when it is needed. Seven are
left for the owner: D1, D2, D3, D6, D7, D8 and D9.

**D1. The names.** What are the two new packages called? Python's
convention is a lower-case install name with hyphens and an import name
with underscores.
- (a) `de-docreview` / `de_docreview` for the app, as you proposed, and
  `docreview-core` / `docreview_core` for the shared part.
- (b) Other names of your choosing.
- **Recommendation: (a).** The shared part is public, like planlens, so its
  name carries no "DE".
- Needed before step 9.

**D2. Where the code both apps use lives.** Both apps run the same chat page,
document and vision tools, figure helper and model connections, about
46,000 lines. It
has to be somewhere both can install.
- (a) Its own public package on PyPI, like planlens; GSE stays on PyPI, so
  Foundry's Libraries panel keeps working.
- (b) Inside Document Review, with GSE installing it: then GSE cannot be on
  PyPI unless Document Review is, and every Foundry install needs uploads.
- (c) Two copies: every fix to the page, tools or connections made twice.
- **Recommendation: (a).** The only option that keeps GSE on PyPI and
  Document Review off it, with one copy of the code.
- Needed before step 9.

**D3. How Tiny Apps gets Document Review.** (A) our wheels in the wrapper
repo; (B) the source folder in the wrapper repo; (C) a hosted package
repository on CfA's Nexus; (D) PyPI, through Nexus.
- **Recommendation:** ask CfA questions 6a-6d now and plan on A. Fall back to
  B, then C, then D (D only on your word). Testers are not affected while
  waiting.
- Needed before step 14.

**D4. Does the geotech page leave Tiny Apps? DECIDED: (a), yes.** The
owner on the desk, 2026-10-20 08:33: "GeotechStaffEngineer running in
Palantir Foundry as an app, and DE_DocReview being a wheel that I pull into
TinyApps", and "now that I have foundry, I'll just go that route". Testers
get Document Review only; GSE runs on Foundry, with Funhouse as backup. It
also drops the Tiny Apps resource question about large meshes and Monte
Carlo runs.

**D5. Which Document Review agent goes into the new package? DECIDED by
the lead: (a)**, the one testers get today, unchanged and proven by the
fingerprint, with `lean` and `minimal` behind their switches as now. One
change at a time; DR-05 (`lean` measured at full resolution on GPT) decides
the default afterwards, on the suite. The owner can overrule (the other
choice: switch testers to `lean` as part of the split).

**D6. Release what is waiting before files move?** Master carries a train
you are holding (OWN-55), including the fix that lets the testers' package
install beside CfA's pandas 3.0.2.
- (a) Release it before step 9.
- (b) Hold it and release once with the split.
- **Recommendation: (a).** It gives a clean version to fall back to, and it
  gets the pandas fix to Nexus while CfA works.
- Needed before step 9.

**D7. Should the Document Review code be private?** Today it is public: it
is in this public repo and in every GSE release since 5.26. Keeping it off
PyPI does not make it private.
- (a) Public: it stays in this repository as its own package.
- (b) Private: after the line is drawn (step 16) it moves to a private
  repository, and the wheel is built there.
- **Recommendation: (a)** unless the Department or CfA asks otherwise.
  Moving later is a short, mechanical step.
- Needed by step 16.

**D8. Where the figure helper and plotting code go.** Since 2026-10-20 both
are on for both pages, by your ruling on GM-92 ("on everywhere"). Most of
them is general (draw, look at the picture, redraw to a checklist). The
geotech part is small: standard chart lines from the reference library, nine
worked examples (a boring plan, an SPT profile, wall pressures and the
like), and a few checklist lines. Costs measured in 2.8.
- (a) Both go into the shared package, so both apps keep them. GSE hands in
  the chart lines (only it has the reference library). The examples go
  with them for now, so Document Review keeps exactly what it has.
- (b) As (a), then Document Review gets its own general examples
  (drawings, schedules, quantities) instead of the geotech ones: new work,
  measured on the suite, after the split.
- (c) Document Review drops both. The least code to move, but it reverses
  your GM-92 ruling and takes figures drawn in code away from testers.
- **Recommendation: (a) during the split, (b) after it.** The shared
  package grows by about 4,500 lines and keeps SciPy, which the plotting
  box lets code use.
- Needed before step 6.

**D9. What testers see named.** The Document Review page shows GSE's
geotechnical professional-use disclaimer today.
- Proposed notice, from the page's own caption: "Document Review reads
  drawings, specifications, submittals, reports and calculation packages
  and hands back memos and marked-up PDFs. It is a review aid: it can miss
  or misread things, and the judgement stays with you. Check anything you
  rely on against the document."
- Also: the SharePoint folder testers' conversations go to is
  `Shared Documents/GeotechStaffEngineer` today. Keep it, or name it
  `Shared Documents/DocumentReview`?
- **Recommendation:** approve the notice (or edit it), and keep the folder
  name until the published app is live, so nothing moves twice.
- Needed before step 14.
