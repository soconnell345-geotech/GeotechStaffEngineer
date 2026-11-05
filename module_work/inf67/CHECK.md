# INF-67 check: the document tools on a geotech turn with no document

Branch `claude/inf-67-doc-tools-hidden`. Source: BACKLOG INF-67;
`module_work/field_feedback/2026-11-03_foundry_build_i/TRACE_REVIEW.md` §2.2.

## 1. The measurement

**How:** `python -m module_work.inf67.measure_prefix` (add `--rows` for every tool).
- It builds the geotech page offline, the way `webapp/tests/page_builds.py` does: the page as shipped, every `GEOTECH_*` setting unset, a fake chat model. That build goes through `webapp.core.build_agent`, `webapp.geotech_page.build_page_agent`, `build_deep_agent` and `funhouse_agent/deep/assembly.py`.
- It runs one turn ("Rankine Ka for phi = 30 deg?") with the conversation's working folder bound, as the web app binds it.
- It records the main agent's first model call: the system message, plus every tool in the OpenAI function form (`convert_to_openai_tool`) that the GPT routes send.

**Tokenizer:** tiktoken `o200k_base` (the GPT-4o / GPT-5 encoding), a stand-in for the GPT routes. Anthropic's tokenizer is not public. The build i trace measured 2.9 chars a token on the Claude route, so the character counts are given as well.

**Two stacks were measured:**
- The build container: deepagents 0.7.24, langchain 1.4.5, planlens 0.14.0, Python 3.11. That is the highest the app's pins allow, so a fresh install gets it.
- The review-fix laptop venv: deepagents 0.6.8, langchain 1.3.4, planlens 0.14.0 (source checkout), Python 3.14. That is the pins' floor.

The 18 document and drawing tools measure the same on both: 7,078 tokens and 28,889 chars. deepagents' own prompt and tools are larger on 0.6.8, so its prefix is larger.

| main agent, first call | tools | tool tokens | system tokens | prefix tokens | tool chars |
|---|---:|---:|---:|---:|---:|
| **build container** | | | | | |
| before (master), any geotech turn | 40 | 15,378 | 6,657 | **22,035** | 64,492 |
| the 18 document and drawing tools within that | 18 | **7,078** (32.1 % of prefix) | | | 28,889 |
| after, no document (15 hidden) | 25 | 9,118 | 6,657 | **15,775** | 38,928 |
| after, a document attached | 40 | 15,378 | 6,657 | 22,035 | 64,492 |
| **laptop venv** | | | | | |
| before (master), any geotech turn | 40 | 16,404 | 8,687 | **25,091** | 68,940 |
| after, no document (15 hidden) | 25 | 10,144 | 8,687 | **18,831** | 43,376 |
| after, a document attached | 40 | 16,404 | 8,687 | 25,091 | 68,940 |

The container's "after, no document" row is the gate's first version, as measured there (24 tools, 8,865 tool tokens, 15,522 prefix, 37,867 chars), plus `analyze_image` (253 tokens, 1,061 chars), which the review put back on the list. The laptop rows are measured after the review fixes. The system message is the same with and without the gate.

The 18 tools are the same set the trace review counted (28,889 chars here, 29,064 on its build). Three stay on the list without a document: `read_pdf_text`, `analyze_pdf_page` and `analyze_image`. By o200k tokens:

| tool | tokens | | tool | tokens |
|---|---:|---|---|---:|
| `annotate_document` | 1,648 | | `open_document` | 263 |
| `log_grid` | 590 | | `annotate_dxf` | 264 |
| `render_region` | 585 | | `analyze_image` | 253 (kept) |
| `measure` | 545 | | `document_page_map` | 239 |
| `analyze_pdf_page` | 405 (kept) | | `render_page_thumbnails` | 186 |
| `find_like` | 350 | | `document_markups` | 170 |
| `search_document` | 344 | | `document_structure` | 168 |
| `read_document` | 330 | | `read_pdf_text` | 160 (kept) |
| `kept_readings` | 303 | | `find_quantities` | 275 |

**Saving on a turn with no document (15 tools hidden):**
- 6,260 o200k tokens on every main-agent model call. That is 28.4 % of the build container's prefix and 24.9 % of the laptop's.
- It is 25,564 chars, about 8.8k tokens at the Claude route's 2.9 chars a token.
- On the build i turn (four calls, 145,102 tokens in), that is about 35k tokens in, roughly a quarter of the turn.
- With the prompt cache read, the saving is the same share of the written prefix and of each read.
- When a document arrives, the tool list changes, so the cached prefix is written again, once.

## 2. How a document gets into a conversation

| path | what it does | how the gate sees it |
|---|---|---|
| The page's upload (the attach widget, the websocket uploader, the chat box's file drop, a pasted screenshot) | `core.stage_uploads` puts the bytes in the conversation's shared `attachments` dict and writes the file into the files folder (`webapp/app.py` line 229) | attachments not empty |
| A restored conversation (sidebar restore, SharePoint mirror, Foundry dataset) | `core.load_attachments` puts the uploads back into the same dict (`app.py` line 299) | attachments not empty |
| A notebook FileUpload (`DeepNotebookChat`) | adds to `agent.geotech_attachments`, the same dict | attachments not empty (a notebook usually binds no folder, and then nothing is hidden anyway) |
| `sharepoint_download_file`, the only tool that fetches a file into the conversation | writes into the working folder and records the file on the conversation's `downloads.json` ledger as the download finishes | its result in the messages (`FETCH_TOOLS`), and the web app's signal reading the ledger (that survives a summarized history and a restart) |
| A working folder the user chose (sidebar "working folder", `core.set_working_dir`) | the folder may already hold their documents | the web app's signal: the bound folder is not the conversation's own `files/` |
| A path typed in chat, library or notebook use | the document tools resolve real paths when no folder is bound | no folder bound: the gate hides nothing |
| A path typed in chat, the web app on Databricks (`/Volumes/...`, `/Workspace/...`) or with `GEOTECH_EXTRA_READ_ROOTS` set | the read tools may read those folders as well as the working folder, and the prompt sends the agent to `list_files` there for an uploaded report | the web app's signal: any extra read root opens the gate, so nothing is hidden on those deployments. Foundry and Tiny Apps set neither and keep the saving |
| The Agent picker's specialists (seismic, foundations, ...) | built by `build_reviewer_agent`, with the same attachments dict | the same signal as the page (`build_reviewer_agent` passes it), so an earlier turn's download and a chosen folder count there too |
| A URL | no tool on either page fetches a URL (the HTTP code in `webapp/` serves the model engines, SharePoint and storage) | nothing to see |
| Files the agent made (calc packages, plots, Word, PDFs) | written into the working folder | not a document of the user's. `read_pdf_text` and `analyze_pdf_page` stay on the list, because the prompt has the agent read back and look at a PDF it built. `analyze_image` stays too, because `plot_data` asks for a look at each figure the agent draws |

## 3. The same-turn case

- `HideTools` filters `request.tools` in `wrap_model_call`, which runs before every model call of every turn.
- `DocumentToolsGate` (`funhouse_agent/deep/document_gate.py`) subclasses it and decides again on each call: are the attachments non-empty, is there a fetch tool's `ToolMessage` in the request's messages, and does the host's signal say yes?
- A document fetched by a tool call is therefore on the list at the very next model call of the same turn: the tool's result is in the messages, and the ledger was written inside the tool.
- An upload between turns lands in the dict the agent closes over.

**A hidden tool still runs if called by name.** Checked directly: `create_agent` with a `HideTools` middleware executes a call to a hidden registered tool. So a model that calls `open_document` from the prompt's own wording, before any signal, is not refused.

## 4. The decision

**Built.**
- The saving is real: 28.4 % of every main-agent call's prefix on a geotech turn with no document (24.9 % on the older deepagents), about a quarter of a short turn's input.
- The same-turn case is handled cleanly by the existing per-call middleware pattern, with no state.

## 5. What changed

- **`funhouse_agent/deep/document_gate.py`** (new):
  - `DOCUMENT_TOOLS`: the 15 hidden tools. They are the trace's 18 less `read_pdf_text`, `analyze_pdf_page` and `analyze_image`.
  - `FETCH_TOOLS` = `{"sharepoint_download_file"}`.
  - `DocumentToolsGate(attachments, signal=)`: hides nothing when no working folder is bound. A signal that raises counts as "document held", so the tools stay offered as before.
  - Tool results and unknown-tool errors pass through unchanged.
- **`funhouse_agent/deep/agent.py`:**
  - `build_deep_agent(document_signal=None)` is a new keyword.
  - The geotech page's `page_middleware` is now `[HideTools(UNRUNNABLE_TOOLS), DocumentToolsGate(attachments, signal=document_signal)]`.
  - The review-page route (`review_page=True`) does not pass the new keyword on.
- **`webapp/geotech_page.py`:**
  - `document_signal(temp_dir)` returns true when the deployment opens folders beyond the conversation (`_extra_read_roots()`: Databricks, `GEOTECH_EXTRA_READ_ROOTS`), when the conversation's downloads ledger is non-empty, or when the bound working folder is not its `files/`.
  - `build_page_agent` and `build_reviewer_agent` (the specialists) pass it in.
- **`webapp/tests/page_fingerprints.json`:** re-recorded twice, each time in its own commit.
  - First: one entry moved. `geotech_default` `/calls[0]/middleware` gains `DocumentToolsGate`, with 16 hidden names.
  - After the review: the same entry loses `analyze_image`, leaving 15.
  - No prompt, tool text, helper or budget moved, and the three review builds did not move.
- **`module_work/inf67/measure_prefix.py`:** the measurement above.

**Not done (noted, not fixed):** the general-purpose helper inherits the main agent's tools, and its own calls still list the document tools. Helpers make fewer calls than the main agent, and the gate's job was the main agent's prefix.

## 6. Tests (offline, scripted models, made-up files)

**`funhouse_agent/deep/tests/test_document_gate.py`, 15 passed:**
- a geotech turn with no document has none of the 15 (they stay registered; `read_pdf_text`, `analyze_pdf_page` and `analyze_image` are offered);
- an attached document turns them on;
- an upload into the same dict after the build turns them on next turn;
- `sharepoint_download_file` called mid-turn turns them on for the next call of that turn, and that call's `open_document` runs;
- the host's signal flipping inside a tool turns them on for the next call;
- no working folder bound hides nothing;
- a signal that raises offers them;
- the Document Review page keeps them;
- the web app's signal reads the downloads ledger and counts a folder the user chose;
- the web app's signal opens on Databricks and with `GEOTECH_EXTRA_READ_ROOTS` set;
- a specialist with an earlier turn's download, or in a folder the user chose, has `open_document` on its first call; one with neither does not.

The five tests added after the review fail without the `webapp/geotech_page.py` fixes (4 failed; the "neither" control passed).

**Wider runs after the review fixes (laptop venv), judged by pytest's exit code:**
- `webapp/tests/test_page_fingerprints.py`: 15 passed, exit 0, after the second re-record.
- `webapp/tests/test_public_text.py`, `test_app_profiles.py`, `test_review_boundary.py`: 16 passed, exit 0.
- The specialist builds in `webapp/tests/test_core.py` and `test_feedback.py` (`-k "specialist or reviewer"`): 6 passed, exit 0.
- `funhouse_agent/deep/tests`: 1,189 passed, 2 skipped, exit 0.

**Before the review, in the build container:** `webapp/tests` gave 1,092 passed and 1 failed. The failure was `test_tinyapps_wiring.py::test_ssl_context_accepts_pem_text_or_path_and_rejects_junk` (`pyo3_runtime.PanicException` from the container's system `cryptography` package building an SSL context). It does not touch this change. The `liquepy` failure the container saw in the deep suite does not occur on the laptop.

## 7. Release note

- Geotech page and its specialists: the document and drawing tools stay off the model's list until the conversation holds a document (an upload, a SharePoint download, a folder you chose). On a question with no document, every model call is about a quarter smaller. The figure check (`analyze_image`) stays on. On Databricks, or where extra read folders are set, nothing is hidden.
- When a document arrives, even mid-turn, the tools are offered from the next model call. The Document Review page is unchanged. No answer changes.
