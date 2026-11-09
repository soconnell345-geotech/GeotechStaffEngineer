# Local reference packs and deployment-appended instructions: design (BACKLOG REF-64)

Plan of record for BACKLOG REF-64 (lead's calls 2026-11-09: the `local_` prefix, the 4,000-character cap per role, and `report_ingest.md` in the first cut as one recorded block; the owner agreed the same day that built-in rules win, section 3.3). Design helper for the lead, 2026-11-09. Read-only: app master `3a57539`, refs `8a09592`, the step 11 (a) branch `origin/claude/inf-32-step11a` (`6362b21`). Nothing was edited or run against a model. Open work stays in `BACKLOG.md` (REF-64 already holds this request). This file keeps the evidence and the design.

## 0. In short

**Where things stand:**
1. The library already has a standard structured format: chapter JSON to `scripts/chapter_schema.json`, plus a figure catalog. Search and figure lookup each run over **one SQLite index, built by walking the installed package's folder only**. Extra roots would be a few lines.
2. The librarian reaches the library only through `reference_db` / `figure_db` and `read_reference_figure`. So a reference the index holds is searched, cited and read by vision **with no change to the librarian**.
3. Today a deployment can only supply the **PDFs** of references that are already catalogued (`GEOTECH_REFERENCES_DOCS`, the SharePoint fetcher, the cache). It cannot add a reference. No agent takes deployment instructions. The only hook is the host's `extra_system_prompt` / `subagent_extra_system_prompt` kwargs (feedback, file tools, presets).

**The design:**
- **(A) Reference packs.** A pack is a folder in the library's own layout (`pack.json`, `text/chapterNN.json`, an optional `figures_catalog.json`, optionally its PDF). `GEOTECH_REFERENCES_PACKS` names where packs live. The refs package indexes packs beside the built-in references. Two optional catalog fields give the FDE control over each figure's vision read: `bbox` (a crop) and `read_notes`. The FDE checks a pack with `python -m geotech_references.packs check <dir>`.
  - Size: about 400 lines of refs code, 250 of refs tests, 50 of app code and 120 of app tests.
- **(B) Appended instructions.** `GEOTECH_INSTRUCTIONS` names a folder holding `all.md` and one `<role>.md` per role. The text is appended in one place, the shared assembly's compile, to the main agent and every helper by name. Report ingest is reached through an engine wrapper. The build record carries a sha, a size and a capped text, so `activity.jsonl` shows what was appended. When the setting is unset nothing changes, and the fingerprints are untouched because the fingerprint builds clear every `GEOTECH_*`.
  - Size: about 180 lines of code and 150 of tests.

**Sequencing:**
- (A)'s refs half can start now.
- (A)'s app half waits for step 11 (a) to merge, because the figure reader moves to `funhouse_agent/reference_vision.py` there.
- (B) waits for step 11 (c) to merge, because it changes `deep/assembly.py`, which (c) moves into the core.
- None of this touches step 11 (b)'s files.

---

## 1. Findings

### 1.1 The reference library as built (`geotech-references`)

**The structured format.** Each reference has two kinds of content (in this section, `CLAUDE.md` means `geotech-references/CLAUDE.md`):
- Python lookups (`tables.py`, `figures.py`, `equations.py`), written by hand.
- Structured text, `<pkg>/text/chapterNN.json` (`geotech-references/CLAUDE.md:11-13`).

The text schema is `scripts/chapter_schema.json` (137 lines):
- **Top level:** `reference_id`, `reference_title`, `volume`, `chapter`, `chapter_title`, `sections`.
- **Each section:** `section_id`, `title`, `body`, `key_points` (at most 12, each at most 300 characters), `equations`, `figures`, `tables`, `applicability`; optional `summary` (at most 700) and nested `subsections`.
- **Equations:** objects with `id`, `description`, and optionally `formula` (typed from the printed page), `variables`, `notes`, `source` (the page it was typed from) and `implemented_in` (schema lines 63-104).
- **Figures and tables in a section:** caption strings only (lines 106-119). A table's values live in the body as printed.

The figure catalog is `<pkg>/figures_catalog.json`:
- **Header:** `reference_id`, `reference_title`, `package`, `volume`, `pdf_path`.
- **Each figure:** `figure_number`, `caption`, `chapter`, `pdf_page_index` (0-based), `printed_page`, `page_estimated`, `description`, and optionally its own `pdf_path` and `volume` (`_figures_db.py:126-161`).
- Page images are not stored. The PDF page is rendered when it is read.

The wheel ships only the data (`pyproject.toml`, package-data `*/text/*.json`, `*/figures_catalog.json`). `scripts/` is not installed (`CLAUDE.md:72`).

**Registration and discovery.** There is no registry: a reference is a folder of the package.
- `_retrieval._reference_dir`: `_PACKAGE_DIR / reference / "text"` (`_retrieval.py:39`, `:232-240`).
- `_retrieval_db._iter_text_jsons` walks `_PACKAGE_DIR.iterdir()` for `*/text/*.json` (`_retrieval_db.py:131-142`).
- `_figures_db._iter_catalogs` walks the same folder for `*/figures_catalog.json` (`_figures_db.py:57-66`).
- `agents/references_agent.py` builds its catalog from the installed package at import (`CLAUDE.md:437`).

**Search.**
- One FTS5 table (`porter unicode61`) holds one row per section at every level. The columns are `title`, `summary`, `body`, `key_points` and `applicability`, plus the structural JSON fields (`_retrieval_db.py:196-311`).
- The database is built lazily into the temp folder. Its file name is keyed by `_copy_key(package_dir)`, a hash of the folder and the installed version (REF-55; `_retrieval_db.py:102-124`). It is rebuilt when a source file is newer than it or its schema version differs (`:175-182`), and built atomically (`_atomic_db`).
- The API: `reference_search` (BM25, summary-only hits, `offset` / `max_chars`; `:421`), `reference_get` (any level, with `children`; `:679-736`), `reference_query` (read-only SELECT; `:764`) and `list_indexed_references` (`:881`).
- **Synonym expansion:** `_query_expansion.py` (435 lines) holds curated groups of geotechnical terms, applied as `auto` by default (`GEOTECH_RETRIEVAL_EXPANSION`; `CLAUDE.md:335-424`). It works on any indexed text, so it would apply to a pack too.
- **Figures:** `figure_search` runs BM25 over number, caption and description with the same expansion (`_figures_db.py:317`). `figure_get` returns the row (`:465`) and `resolve_pdf` returns the PDF and page (`:557-584`).

**How a figure is read** (app, master `funhouse_agent/vision_tools.py:4109-4241`):
1. `figure_get` and `resolve_pdf` find the page. `resolve_pdf` looks for the PDF by file name in `GEOTECH_REFERENCES_DOCS`, else repo-relative to the source checkout.
2. When the PDF is not found, the app falls back to `reference_docs.fetch(filename)`, the host's fetcher (`:4136-4144`).
3. The page is rendered with `vision_view.render_view(path, page=...)` at the chart budget (`chart_reading()`, `GEOTECH_CHART_BUDGET`; `vision_view.py:230`). The render has **no crop**, although `render_view` takes `bbox` (`vision_view.py:379`).
4. The model is asked with a fixed read-off prompt naming the reference and the caption, with a caveat when the page is estimated (`:4163-4193`).
5. The call is `ask_vision(..., "reference_figure")` (`:4203-4206`). The kind is in `turn_question.QUESTION_KINDS`, so the call also sees the user's question (`turn_question.py:36-37`).
6. An optional code voter measures the chart (`chart_reading`; `:4194-4239`).
7. The vision call runs on the main model's engine, not the helper's model (`webapp/helper_models.py:29-33`).
8. `view_worked_example_source` is the same path with kind `worked_example_page` (`:4086`).
9. **On the step 11 (a) branch** `vision_tools.py` moves to the core. GSE's two readers move to the new `funhouse_agent/reference_vision.py` (312 lines) and are handed to the core through the hooks `vision_tools.read_reference_figure`, `vision_tools.view_worked_example_source` and `vision_tools.reference_read_roots` (`packages/aec-agent-core/aec_agent_core/hooks.py`).

**Is there a build pipeline, and could an FDE run it?**
- `scripts/build_chapter_text.py`: PDF to chapter JSON through the API. It is **retired** (it cost $20 a reference on DM7; `CLAUDE.md:13`, `:74`). Chapter JSON has since been written by Claude Code sessions.
- `scripts/build_printed_text.py` (no API, needs PyMuPDF): reads chapter text and the figure catalog off the printed contents and the Lists of Figures and Tables, with `[p. 6-28]` markers and the print's wording (docstring `:1-56`). Its patterns are tuned to one manual (`_HEADER_END = December 2006`, `:80`), so it is not general.
- `scripts/build_figure_catalog.py` (no API, needs PyMuPDF): builds the catalog from the List of Figures and resolves each page by searching for its caption. It is fairly general, but its paths are hard-coded to the package (`:47`, `:849`).
- `scripts/audit_chapter_text.py`: the validator. `validate_schema` (`:218-288`) is self-contained: required fields, the section-id pattern (`:172`), key points, body floor, duplicate ids. The script's other checks import the reference's Python modules (`implemented_in`).
- `scripts/check_section_numbers.py`: checks section numbers against the PDF's outline.
- **Verdict:** none of these is in the wheel, and all of them write into the package folder. An FDE could run them only from a source checkout, and only after changes to the output paths. The no-API catalog builder and the validator are the two worth keeping. The text itself has in practice been model-written to the schema.

### 1.2 The librarian (the `references` helper)

**Built by** `build_references_subagent` (`funhouse_agent/deep/agent.py:180-282`).
- **Tools:** the core dispatch tools scoped to `REFERENCE_MODULES` (`dispatch.py:65-79`: `reference_db`, `figure_db` and the per-reference adapters dm7, gec4-14 and so on), plus `read_reference_figure` and `calculate` (`agent.py:236-248`). `HideTools(REFERENCE_HIDDEN_TOOLS)` hides the scratch filesystem tools, and a model-call budget applies (`:272-281`).
- **Prompt:** `reviewer.CONSULTANT_ROLE` (`reviewer.py:263-280`; it names DM7, the GECs, UFC, Eurocode 7 and "the others that list_agents names") + `_REFERENCES_CONCISION` (`agent.py:107-112`). With `GEOTECH_JUDGMENT_CALLS` on, `_REFERENCES_OPTIONS` and `_REFERENCES_JUDGMENT_CALLS` follow (`:117-143`, `:254-261`), then the stated budget (`:146-150`, `:262-263`), then `extra_system_prompt` (`:264-265`; the web app's `record_feedback` text).
- **Description shown to the main agent:** `_REFERENCES_DESCRIPTION` (`:157-166`). It lists the library's families.

**Who calls it:**
- The geotech page's main agent, through deepagents' `task`, attached when `reference_mode != "off"` (`agent.py:1238-1252`).
- Beside it, the `reviewer` helper has the same reference tools (`:665-710`, `:1253-1260`).
- The calc helper reaches the reference modules' functions but not `reference_db` / `figure_db` (`:577-586`).
- The **Document Review page builds no librarian**: `reference_mode="off"` (`review_build.py:166-167`, `:203-209`), except for old callers through `geotech_parts` (`agent.py:1460-1473`).
- The v1 notebook agent's `consult_references` (`native_tools.py:446-448`; `reviewer.py:286-316`) runs the same adapters.
- On Foundry the librarian runs on Sonnet 5.5 (`helper_models.py:55`).

**How it reaches the library:**
- `call_agent("reference_db", "reference_search" | "reference_get" | "reference_query" | "list_indexed_references")`. The adapter `funhouse_agent/adapters/reference_db_adapter.py:80-290` wraps `_retrieval_db`, resolving printed names to ids (`resolve_reference`, `:48-69`) from `list_indexed_references`, so any indexed id resolves.
- `call_agent("figure_db", "figure_search" | "figure_get")` through `figure_db_adapter.py`.
- `read_reference_figure(reference, figure_number, prompt)`.
- Every route ends in the refs package's two databases.

### 1.3 Extension points today

**`GEOTECH_REFERENCES_DOCS`:** a flat folder of PDFs, looked up by file name.
- Readers: `_figures_db.resolve_pdf` (`_figures_db.py:549-584`), `worked_examples.py:135`, the read roots (`vision_tools.py:468-502`), the diagnostics (`webapp/geotech_page.py:234-336`) and `review_eval/documents.py:134`.
- It only **supplies PDFs for figures already catalogued** in the installed package. A PDF put there that no catalog names is never searched or read.

**`GEOTECH_REFERENCES_SHAREPOINT_DIR`** (`webapp/reference_fetch.py:21-56`): the SharePoint folder (default `primary_references`) the web app's fetcher downloads a missing PDF from by file name, registered with `funhouse_agent.reference_docs.register_fetcher` when SharePoint is configured.

**`GEOTECH_REFERENCES_CACHE`** (`funhouse_agent/reference_docs.py:24`, `:40-69`): where fetched PDFs are kept for the session.

**So nothing lets a deployment add a reference, raw or structured.**
- On Foundry there is no fetcher. The PDFs sit in the workspace repo's `references/` folder (`docs/FOUNDRY.md:348-363`, BACKLOG INF-63), and whether that folder reaches the published app is INF-63's open check (U2).
- On Databricks the PDFs come from SharePoint or a docs folder (`webapp/README.md:374-415`).

**`extra_instructions`** has 3 hits, all in `funhouse_agent/deep/eval_harness.py:770`, `:784`, `:807`. That is the LLM judge's text for the eval harness, not an agent's.

**The live hook is `extra_system_prompt`:**
- `build_deep_agent(extra_system_prompt, subagent_extra_system_prompt, calc_extra_system_prompt)` (`agent.py:836`, `:848`, `:851`, applied at `:1165-1170` and `:1249`-`:1358`).
- Mirrored by `review_build.build_review_page_agent` (`review_build.py:150`, `:156`, `:265-266`, `:287`), `figure_helper.py:328`, `:358`, `setup_agent.py:111`, `:141` and `minimal_agent.py:247`, `:273`.
- `build_report_library_subagent(extra_system_prompt=)` (`report_ingest/library_agent.py:1059`, `:1082`) is never passed one.
- The host fills them in `webapp/core.build_agent` (`core.py:778-820`): `_add_feedback_tool` reaches the main agent and every helper (`:695-730`), `_add_file_tools` the main agent only (`:733-763`), and the conversation's preset the main agent (`:2329`).
- Granularity: main agent, all helpers, calc. **There is no per-role hook for the librarian or the reviewer, nothing for the compiled helpers, and nothing from the deployment.**

**Behaviour held as agent-editable data:** not in the app.
- The only instruction file an agent could edit is deepagents' `/memories/AGENTS.md`. It loads into the prompt when `enable_memory` is on (`deep/assembly.py:83-89`, `:405-411`), and the web app never turns it on (no `enable_memory` in `webapp/` beyond a docstring, `core.py:782`).
- Report ingest keeps conventions as data in code (`report_ingest/narrative_glossary.py`) and reads private files named by an environment variable: `GSE_REPORT_QUESTIONS`, `GSE_REPORT_POSTS` (`question_list.py:100`, `:2146`, `:2435`). That is the precedent for deployment-private data by path.
- GeoBot's `_agent/GUIDANCE.md` (capped, added to every prompt; logged edits) belongs to the separate PM bot (`project_manager_bot`).

**How each page's prompts are assembled:**
- **Geotech page:** the domain prompt (`build_domain_prompt`), then the judgment-calls block, `extra_system_prompt`, the calc nudge, `REPORT_INGEST_NUDGE`, `REPORT_LIBRARY_NUDGE` and `FIGURE_PRIMARY_NOTE` (`agent.py:1147-1179`, `:1202`, `:1223`, `:1301`). The helpers are built at `:1238-1360`.
- **Document Review:** `build_document_review_prompt`, then `extra_system_prompt`, the calc nudge and `FIGURE_PRIMARY_NOTE` (`review_build.py:259-268`, `:297`). The reading helper is named `general-purpose`, so it replaces the stock one (`review_build.py:88-115`, `:305-307`).
- **Both** then go through the shared assembly:
  - `assembly.guard_helpers` (`assembly.py:262-380`), which also adds deepagents' general-purpose helper at `:339-366`;
  - `assembly.create_kwargs_for`;
  - `assembly.compile_agent` (`:468-503`), which receives the final `system_prompt` and the complete helper list;
  - `assembly.record_build` (`:506-548`), which sets `agent.geotech_build`.
- The host copies `geotech_build` into `engine_record` (`core.py:2651-2670`). `ActivityLogger.turn_start` writes it to every `turn_start` record as `engine.build` (`activity_log.py:967-991`). `turn_end` carries the system prompt's sha and length (`activity_log.py:73-85`).
- **Report ingest is a `CompiledSubAgent`** (`report_ingest/subagent.py:72`, `:719`). It runs its own graph: triage, label vote and review, the log, lab, narrative, calc and sounding readers. Every model call goes through `Engine.complete(messages, system=...)` (`engine.py:1650-1666`; `ClaudeEngine:1793`, `PrompterEngine:2336`, `ChatModelEngine:2920`), on engines from `agent.py:799-813` (`engine_for`). The calls carry no stage label.
- `guard_helpers` skips compiled helpers (`assembly.py:292-300`).
- **Where a deployment block belongs:** once per agent, appended last to each agent's own prompt. The one place that sees every final prompt on both pages is `assembly.compile_agent`. Report ingest needs its engine wrapped. The report library needs its existing `extra_system_prompt`.
- **Not covered by the assembly:** the `minimal` review build (`minimal_agent.py:272-290`, a measurement build) and the v1 agent.

### 1.4 Constraints

**Fingerprints** (`webapp/tests/test_page_fingerprints.py:1-53`):
- Builds go through `core.build_agent` (`page_builds.py:15`, `:195`) inside `page_environment`, which **unsets every `GEOTECH_*`** (`page_builds.py:60-87`).
- New settings named `GEOTECH_REFERENCES_PACKS` and `GEOTECH_INSTRUCTIONS` are therefore unset in every fingerprint build. Unset, they append nothing and name nothing, so the fingerprints stay byte-identical.
- The fingerprints record the library only through its chart-line names (INF-58). The text index does not enter any prompt.
- Test `:164-178` shows how to prove a set value changes only the prompt hashes it should.

**The owner's rules:**
- Formulas come from the print only; every value cites a reference; methods stay open to assumptions (memory feedback files).
- The repos are public, and Department documents are never committed (app `CLAUDE.md:2486-2488`; `webapp/tests/test_public_text.py`).
- So a pack is private deployment data: never in either repo or the wheel, held to the same print-fidelity rules as the built-in text, and cited by its own reference id.

**Where each piece goes after the split** (`module_work/APP_SPLIT_PLAN.md` 1.1, 1.2, 2.3, 2.4):
- **Packs:** the refs package (GSE-only).
- **The figure reader:** GSE (`reference_vision.py` after 11 (a); 2.4 keeps `reference_docs.py` and the reviewer in GSE).
- **The appended-instructions mechanism:** the core. It names no Department deployment (rule 1.2.2), and the assembly goes to the core at 11 (c) (`APP_SPLIT_PLAN.md:398-401`, `:954-956`).
- **The two compiled helpers' wiring:** GSE (`agent.py` stays).

**Step 11 paths:**
- (a) moves `vision_tools.py` and adds `reference_vision.py` and `hooks.py`.
- (b) moves the figure helper, `plot_box`, the chart stack and `figure_examples` (`:382-389`).
- (c) moves `keep_work`, `limits`, `tool_guards`, `review_flags`, `activity_log`, `run_mirror`, `cited_*`, the suite framework, the document binding and **`deep/assembly.py`** (`:391-420`, `:954-956`).

**Hosts:**
- **Foundry (GSE production):** the workspace repo beside `app.py` (`FOUNDRY.md:93-100`, `:348-363`), or a dataset read with `foundry.transforms.Dataset` (INF-38's store; `/foundry/outputs`).
- **Databricks (backup):** Workspace folders or volumes, with environment variables set before `run_on_databricks` (`webapp/README.md:402-415`).
- **Tiny Apps:** Document Review only, which has no librarian; the wrapper repo plus settings (`tinyapps/TINYAPPS.md`). Per-tester SharePoint is not relevant to deployment-wide data.

---

## 2. Design A: local reference packs

### 2.1 What a pack is

A folder in exactly the layout of a built-in reference package, minus the Python:

```
<packs root>/                          # a folder named by GEOTECH_REFERENCES_PACKS
  local_agency_gdm/                    # the pack; folder name = pack id
    pack.json                          # who, what, which edition, which PDF
    text/chapter01.json ...            # chapter_schema.json, unchanged
    figures_catalog.json               # optional; the catalog format, two optional fields (2.5)
    Agency GDM 2024.pdf                # optional here; else found by name (2.5)
```

`pack.json`:
- `id`: must start with `local_`, so it never collides with a built-in or a future one, and every citation and record shows it is the deployment's own;
- `reference_id` and `reference_title`: as printed;
- `edition` or `date`;
- `pdf_files`;
- `prepared_by`, `prepared_on`, `method` (`"printed"` read off the print, `"model"` written by a model to the schema, `"hand"`);
- `distribution`: a free-text note, for example "agency internal; not for the public repos".

There is no executable code in a pack (section 5).

### 2.2 How the FDE makes one

1. `python -m geotech_references.packs new <dir> --id local_x --title "..." --pdf <file>` writes the skeleton `pack.json` and an empty `text/`.
2. **Text:** the FDE, or the FDE's own model, writes `text/chapterNN.json` to `chapter_schema.json`. The rules go in a new `docs/REFERENCE_PACKS.md` in the refs repo:
   - bodies in the print's own wording with `[p. N]` page markers;
   - `summary` and `key_points` only where they restate the print;
   - each equation typed from the printed page, with `source` naming the page and `implemented_in: null`;
   - tables as printed lines in the body;
   - section numbers as printed.

   This is the "standard structured manner" the built-in references follow (`build_printed_text.py:37-41` states the same rules).
3. **Catalog:** `scripts/build_figure_catalog.py --pack <dir>` builds it from a source checkout with no API, from the List of Figures, finding each page by caption search. On a host with only the wheel, the FDE writes the catalog from the List of Figures, and `check` (2.6) confirms each caption is printed on the page it names.
4. A table printed as a picture can go in the catalog as an entry numbered `Table 5-3`. `_norm_fig_number` only strips "Figure" (`_figures_db.py:186-198`), so it is read like a figure.

### 2.3 Where packs live, and how they are found

`GEOTECH_REFERENCES_PACKS` holds one or more folders separated by `os.pathsep`. Each folder is either a pack (it has `pack.json`) or a folder of packs.
- It is read at call time by the refs package, as `GEOTECH_REFERENCES_DOCS` already is.
- A pack whose id is a built-in name, lacks the `local_` prefix or repeats an earlier pack's id is skipped and named in `packs.problems()`.

| Host | Packs live | Set |
|---|---|---|
| Foundry | committed beside `app.py` (`references/packs/`, next to §3d's `references/` PDFs); if INF-63's U2 shows committed folders do not reach the published app, then in the dataset store, copied to a temp folder at boot by the same fetcher INF-63 builds for the PDFs ("where the PDFs go, the packs go") | `os.environ.setdefault("GEOTECH_REFERENCES_PACKS", <dir>/references/packs)` in `app.py` |
| Databricks | a Workspace folder or a volume (`/Workspace/Users/<you>/geotech_agents/packs`) | env before `run_on_databricks` |
| Tiny Apps | not needed: Document Review has no librarian. Until step 15, a geotech page there could use a folder in the wrapper repo and the same setting | `.env` / `run.sh` |

The index itself is still built into the local temp folder (2.4), so a read-only pack folder is fine.

### 2.4 Search and citation beside the built-in library

**One index over both:**
- `_iter_text_jsons` and `_iter_catalogs` walk `[_PACKAGE_DIR, *pack_dirs()]`.
- The database key covers the pack set, not only the package copy: `keyed_db_path(name, package_dir)` hashes the folder, the version and, when packs are set, each pack file's path, size and mtime. A pack edit, or a copy that keeps an old mtime, then gets a database of its own.
- **Unset, the key input is unchanged**, so existing temp databases stay valid.
- **Ranking:** the same BM25 and synonym expansion, with no boost or penalty for packs.

**Citation:**
- Hits carry the pack's `reference` (`local_...`) and its `reference_title` and section ids from its own chapter files, so the librarian cites "Agency GDM 2024, Sec. 4.3" as it cites "GEC-12, Sec. 7.2.1.3.1".
- `reference_get`, `children`, `include_subsections` and `reference_query` work unchanged because they read the database.
- `list_indexed_references` lists the packs.
- The adapter's `resolve_reference` resolves pack ids from that list (`reference_db_adapter.py:92-94`, `:146-149`).

**Telling the agents a pack exists:** when packs are present, one generated line goes into both the librarian's prompt and `_REFERENCES_DESCRIPTION`. The line reads "This deployment adds: <title> (`local_x`), ...", so the main agent routes questions about its own manual to the librarian and the librarian can scope a search with `reference=`. With no packs there is no line, and no fingerprint change.

### 2.5 Figures and the vision call (the "vision call dynamics")

**Resolving the PDF:**
- A pack catalog's `pdf_path` holds the file name only, so no host path reaches the model in `figure_get`.
- `resolve_pdf(reference, n)` looks in this order: the pack's own folder, then `GEOTECH_REFERENCES_DOCS` by name, then repo-relative, then `FileNotFoundError`.
- On `FileNotFoundError` the app's existing fallback asks the host's fetcher by file name: SharePoint `primary_references` on Funhouse, INF-63's dataset fetcher on Foundry. A pack PDF can therefore go wherever the built-in PDFs go.

**The read** is unchanged: the chart budget, the read-off prompt naming the reference and caption, the `page_estimated` check-the-page step, `turn_question`, and the code voter where on. It runs on the main model's vision engine.

**Two optional catalog fields, honoured for any figure** (built-in figures carry neither, so their render and prompt stay byte-identical):
- `bbox: [x0, y0, x1, y1]`, in PDF points with a top-left origin as `render_region` takes them. The reader passes it to `vision_view.render_view(..., bbox=)` (`vision_view.py:379`), so the chart, not the whole page, fills the image budget. This is the lever for dense curve families.
- `read_notes`: at most 500 characters, for example "x axis is logarithmic; curves labelled by OCR; read at the curve's centreline". The reader appends it to the read prompt under a fixed heading ("Notes from the reference's catalog on reading this figure").

**Where:** about 20 lines in `funhouse_agent/reference_vision.py` (after 11 (a)).

### 2.6 The check before use

`python -m geotech_references.packs check <pack dir>` exits 0 or 1 and prints what it found:

1. **`pack.json`:** required keys, the `local_` prefix, `id` equal to the folder name, no built-in id. The pack folder must sit outside the installed package and outside any git work tree of the public repos.
2. **Every `text/*.json`:** the library's `validate_schema`, moved from `scripts/audit_chapter_text.py:150-288` into `geotech_references/_schema.py` and imported back by the script, so there is one validator. `reference_id` and `reference_title` must match `pack.json`, and chapter numbers must be unique.
3. **`figures_catalog.json`:** required fields, unique numbers, an integer page index of 0 or more, `read_notes` length.
   - With PyMuPDF and the PDF found: the page index is within the page count; the figure's number is printed on that page, else the entry must say `page_estimated: true` (warning); `bbox` lies inside the page.
4. **Print-fidelity warnings (not errors):** an equation with `formula` but no `source`; a section body with no `[p. ` marker.
5. **Smoke test:** builds the index with built-ins plus this pack and checks that a search for each chapter's title returns a hit from the pack and that `figure_get` works for the first figure. It prints the counts.

Also:
- `python -m geotech_references.packs list` shows what `GEOTECH_REFERENCES_PACKS` resolves to, with counts and problems: what the app will see.
- `python -m geotech_references.packs index --rebuild` rebuilds the index.
- The app's Connection diagnostics row for reference PDFs (`geotech_page.py:237-270`) reads catalogs through `_figures_db._iter_catalogs()` where available, so it lists pack PDFs too.

### 2.7 In the record

- Tool results already carry the `local_...` ids, so `activity.jsonl` shows which answer came from a pack.
- GSE adds `reference_packs: [{id, title, files, sha}]` to `agent.geotech_build` after `record_build`. It reaches every `turn_start` (`engine.build`), so a reviewer knows which pack edition was in force. It is absent when there are no packs.

### 2.8 Files, packages, sequencing, tests, size

| Change | Package | Lines |
|---|---|---|
| `geotech_references/_packs.py`: roots, `pack_dirs()`, `pack_dir(id)`, `problems()`, the key part | refs | ~80 |
| `_retrieval_db.py` / `_figures_db.py`: iterators over the roots, `keyed_db_path` extra key, `resolve_pdf` pack folder first | refs | ~30 changed |
| `geotech_references/_schema.py` (moved `validate_schema` and helpers; the script imports it) | refs | ~110 moved |
| `geotech_references/packs.py` CLI: `new`, `check`, `list`, `index` | refs | ~220 |
| `scripts/build_figure_catalog.py --pack` | refs (source checkout) | ~25 |
| `docs/REFERENCE_PACKS.md` (format, rules, hosts) | refs | ~120 prose |
| `reference_vision.py`: `bbox`, `read_notes` | GSE | ~20 |
| `agent.py`: the packs line in the librarian's prompt and description; `reference_packs` in `geotech_build` | GSE | ~25 |
| `geotech_page.py`: diagnostics read pack catalogs | GSE | ~10 |
| `FOUNDRY.md` §3d, `webapp/README.md`, `DATABRICKS_INSTALL.md` | GSE docs | ~30 prose |

**Feature detection:** the app tests `hasattr(_retrieval_db, "pack_dirs")`, as the adapters already use `takes(...)`, so it runs on refs 1.5.0 with no packs.

**Release:** refs 1.6.0, then the pin. The owner gives the release word.

**Sequencing:**
- The refs half is independent of step 11 and can start now.
- The app half waits for **11 (a)** to merge. On master the reader is still in `vision_tools.py`, which (a) moves to the core and splits.
- It is not in (b)'s or (c)'s path. `agent.py` gets only the import rewrites in (c), and `geotech_page.py` stays in GSE.

**Tests** (made-up values only: a fixture pack in `tmp_path` with one chapter of three sections, two figures, and a one-page PDF drawn with PyMuPDF, skipped without it):
- **refs, `tests/test_packs.py`:**
  - with the setting, `list_indexed_references` holds the pack; a made-up term is found only in it; `reference_get` with children works; `figure_search` and `figure_get` work; `resolve_pdf` finds the pack's PDF;
  - unset, the iterators and `keyed_db_path` equal today's;
  - the key changes with a pack edit;
  - a built-in-id collision, a missing prefix and a duplicate id are skipped and named;
  - `check` passes the good pack and catches each error class;
  - `_schema` gives the same results as before on all built-in chapters (the existing audit test).
- **app:**
  - `read_reference_figure` on a pack figure with a fake vision engine: `bbox` reaches `render_view` and `read_notes` reaches the prompt;
  - a built-in figure's prompt is pinned unchanged;
  - the packs line appears only with packs present;
  - diagnostics list a pack PDF;
  - a guard test: no `pack.json` tracked in either repo (`git ls-files`), beside `test_public_text.py`.

**Total:** about 400 lines of refs code, 250 of refs tests, 50 of app code and 120 of app tests.

---

## 3. Design B: instructions a deployment appends to every agent and helper

### 3.1 The setting and the files

`GEOTECH_INSTRUCTIONS` names a folder. The files are UTF-8 text, read when an agent is built, so an edit reaches the next conversation without a restart.

| File | Appended to | Page |
|---|---|---|
| `all.md` | every agent below (first) | both |
| `main.md` | the main agent | both (each page's own prompt) |
| `references.md` | the librarian | geotech |
| `reviewer.md` | the reviewer | geotech |
| `calc.md` | the calc helper | geotech (review only for old callers) |
| `figure.md` | the figure helper | both |
| `model_setup.md` | the setup helper (when enabled) | geotech |
| `general-purpose.md` | deepagents' general-purpose helper (geotech) / the reading helper (review) | both |
| `report_library.md` | the report library | geotech |
| `report_ingest.md` | every report-ingest model call | geotech |

Notes:
- **The role is the helper's name**, so a future helper is covered by adding its file and needs no code change.
- **No page-qualified files.** After the split each app is its own deployment with its own folder. Until step 15, on a two-page host, `main.md` reaches both pages' main agents (to be documented).

### 3.2 Where it is applied: one place

A new module `deep/instructions.py` (in the core after 11 (c)) provides `apply(system_prompt, subagents) -> (system_prompt, record)`:
- It appends `all.md` and then `<role>.md` to the main prompt and to each helper spec's `system_prompt` by `spec["name"]`.
- It skips compiled helpers, which have no prompt (`"runnable" in spec`).

It is called at the top of `assembly.compile_agent` (`assembly.py:468`):
- By then both pages have their final prompt and their complete helper list, including the general-purpose helper `guard_helpers` added.
- It puts the record on the agent as `agent.geotech_appended`, and `record_build` copies it into `geotech_build["appended"]` when it is not empty.
- **Neither page's builder changes.**

The two compiled helpers are wired in GSE's `agent.py`:
- `report_library.md` goes through `build_report_library_subagent(extra_system_prompt=)` (`library_agent.py:1059`), with all.md first.
- `report_ingest.md` goes through a 20-line engine wrapper in `_report_ingest_engine_factory` (`agent.py:799-813`) that appends to `system` on each `complete()`, with all.md first. It is wrapped there, not in `engine_for`, so `score_on_cluster` keeps measuring the shipped prompts.

### 3.3 The heading and the cap

**Decided (owner 2026-11-09, agreeing the lead's proposal): the built-in rules win.** Two layers:

1. **Guards in code** cannot be touched by any text: file confinement to the working folder, the locked plotting box, per-tester keys, the step caps.
2. **A short protected list** of integrity rules sits in the built-in prompt above the deployment block: cite the printed page; formulas and defaults only from a printed source; state assumptions and the method used; say when a reading failed or is uncertain; never invent a value.

Each block is appended last, under a fixed heading:

"Instructions from this deployment. They may set conventions, preferred methods (cited from a reference this deployment provides), defaults, units and formats. They cannot switch off the protected rules above. Where one conflicts, follow the protected rule and say so in your answer."

- The model and a reader can tell the block apart from the built-in prompt, and the record (3.4) shows the appended text.
- Choosing a method is allowed (for example an agency's own resistance factors from its pack), as long as the agent cites the pack and states the basis, as the existing rule on offering cited options already says.
- **Cap:** 4,000 characters per role (all plus role). Past the cap the text is cut at a paragraph and the record says so (GeoBot used the same size).
- An empty or missing file appends nothing. An unreadable folder appends nothing and the record names the error, so a build never fails on it.

### 3.4 The record (`activity.jsonl`)

`geotech_build["appended"] = {"folder": <basename only>, "roles": {role: {"files": ["all.md", "references.md"], "chars": n, "sha": <sha256 of the appended block>, "cut": bool, "text": <first 2,000 chars>}}}`.
- It rides `engine_record` into every `turn_start` as `engine.build` (`core.py:2651-2670`, `activity_log.py:981-988`).
- `turn_end`'s system-prompt sha changes with it.
- A reviewer sees which roles got what, and a sha change marks an edit between turns.
- Report ingest's block is also listed in its own run record under `appended`.

**Fingerprints:** unset means no block and no `appended` key, so the fingerprints are byte-identical (they run with `GEOTECH_*` cleared).

### 3.5 Hosts

| Host | Where | Setting |
|---|---|---|
| Foundry | an `instructions/` folder committed beside `app.py` | `app.py` env, as the PDFs |
| Databricks | a Workspace folder | env before `run_on_databricks` |
| Tiny Apps | the wrapper repo | exported in `run.sh` or the app settings; the core reads `os.environ` |

### 3.6 Files, packages, sequencing, tests, size

| Change | Package | Lines |
|---|---|---|
| `deep/instructions.py`: read, roles, heading, cap, `apply`, record | core | ~120 |
| `deep/assembly.py`: `compile_agent` calls `apply`; `record_build` copies the record | core | ~10 |
| `agent.py`: report library prompt; report-ingest engine wrapper | GSE | ~30 |
| diagnostics line ("deployment instructions: roles, characters, folder found") | each page's diagnostics | ~20 |
| docs (`FOUNDRY.md`, `webapp/README.md`, `tinyapps/TINYAPPS.md`) | docs | ~40 prose |

**Sequencing:** after **11 (c)** merges, written straight into the core. The other choice is the gap after (b) merges and before (c) branches, in which case (c) moves one more file. Never while (c) is open, because it changes `deep/assembly.py`. It touches no file of (b).

**Tests:**
- Unset: the built agent is unchanged (the fingerprint suite as is, plus a direct check that no `appended` key appears).
- Set with `references.md` only: in a fingerprint-style build, the only difference is the librarian's prompt hash, like `test_fingerprint_sees_a_changed_prompt`.
- `all.md` reaches every helper, the review page's reading helper included.
- The cap cuts and says so.
- The record's sha, chars and files are right.
- The wrapper appends to `system` on a fake ingest engine.
- A compiled helper is skipped without error.
- An unreadable folder still builds.

**Total:** about 180 lines of code and 150 of tests.

---

## 4. Sequencing against step 11

1. **Now:** (A)'s refs half on a refs branch: packs, key, `resolve_pdf`, `_schema`, the CLI, docs, tests. It is outside step 11, and it is released with the owner's word (refs 1.6.0).
2. **After 11 (a) merges:** (A)'s app half (`reference_vision.py`, the `agent.py` line, diagnostics, docs). It does not touch (b)'s files, and (c) only rewrites imports in `agent.py`.
3. **After 11 (c) merges:** (B), in the core's `deep/instructions.py` and `deep/assembly.py`, plus GSE's `agent.py` wiring.
4. **Foundry delivery** of packs, and of PDFs, follows INF-63's U2 check.

## 5. What I would NOT do, and why

- **No Python in packs** (no `tables.py` or `figures.py` lookups). Running a deployment folder's code on locked hosts is a security and review problem. Numbers come from the printed text the librarian reads, and arithmetic goes through `calculate`, as CONSULTANT_ROLE already says. Computed lookups stay in the public library.
- **No indexing of raw PDFs on the fly.** Unchecked text cannot carry page-accurate citations or formulas from the print, and the retired API pipeline cost $20 a reference. A pack is built and checked first.
- **No embeddings or vector store.** Lexical search plus the synonyms stays the library's lever (`geotech-references/CLAUDE.md:340-346`).
- **No pack-specific synonym groups for now.** The expansion table is global and curated, so a pack's synonyms would move the ranking of built-in queries (REF-54's measured guard).
- **No per-stage report-ingest instructions, no vision-side-call role, no page-qualified roles for now.** Report ingest's prompts are measured against hand truth, so one recorded block is the most to add before a measurement says more is worth it. Its conventions belong in `narrative_glossary.py` and `GSE_REPORT_QUESTIONS`.
- **No agent-editable appended instructions, and no per-user "my instructions" box, for now.** On shared hosts both need per-user isolation and a log of every change (the GeoBot pattern). The deployment layer comes first, and the record's sha already shows any edit.
- **No change to `CONSULTANT_ROLE`'s wording.** Packs get a generated line only when present, so the role and the fingerprints stay constant.

## 6. Decisions to take (for the lead to put on the backlog under REF-64)

- **Owner:** do a deployment's instructions sit under the built-in rules (the heading in 3.3)? **Decided 2026-11-09: yes**, with the two layers in 3.3.
- **Lead (small calls):**
  - the `local_` id prefix;
  - the 4,000-character cap;
  - whether `report_ingest.md` ships in the first cut or waits for a measured need.
