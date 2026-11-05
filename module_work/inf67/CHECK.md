# INF-67 check: the document tools on a geotech turn with no document

Branch `claude/inf-67-doc-tools-hidden`. Source: BACKLOG INF-67;
`module_work/field_feedback/2026-11-03_foundry_build_i/TRACE_REVIEW.md` §2.2.

## 1. The measurement

**How:** `python -m module_work.inf67.measure_prefix` (add `--rows` for every tool).
- It builds the geotech page offline, the way `webapp/tests/page_builds.py` does: the page as shipped, every `GEOTECH_*` setting unset, a fake chat model. That build goes through `webapp.core.build_agent`, `webapp.geotech_page.build_page_agent`, `build_deep_agent` and `funhouse_agent/deep/assembly.py`.
- It runs one turn ("Rankine Ka for phi = 30 deg?") with the conversation's working folder bound, as the web app binds it.
- It records the main agent's first model call: the system message, plus every tool in the OpenAI function form (`convert_to_openai_tool`) that the GPT routes send.

**Tokenizer:** tiktoken `o200k_base` (the GPT-4o / GPT-5 encoding), a stand-in for the GPT routes. Anthropic's tokenizer is not public. The build i trace measured 2.9 chars a token on the Claude route, so the character counts are given as well.

**Stack in this container:** deepagents 0.7.24, langchain 1.4.5, planlens 0.14.0, Python 3.11.

| main agent, first call | tools | tool tokens | system tokens | prefix tokens | tool chars |
|---|---:|---:|---:|---:|---:|
| before (master), any geotech turn | 40 | 15,378 | 6,657 | **22,035** | 64,492 |
| the 18 document and drawing tools within that | 18 | **7,078** (32.1 % of prefix) | | | 28,889 |
| after, no document | 24 | 8,865 | 6,657 | **15,522** | 37,867 |
| after, a document attached | 40 | 15,378 | 6,657 | 22,035 | 64,492 |

The 18 tools are the same set the trace review counted (28,889 chars here, 29,064 on its build). By o200k tokens:

| tool | tokens | | tool | tokens |
|---|---:|---|---|---:|
| `annotate_document` | 1,648 | | `open_document` | 263 |
| `log_grid` | 590 | | `annotate_dxf` | 264 |
| `render_region` | 585 | | `analyze_image` | 253 |
| `measure` | 545 | | `document_page_map` | 239 |
| `analyze_pdf_page` | 405 (kept) | | `render_page_thumbnails` | 186 |
| `find_like` | 350 | | `document_markups` | 170 |
| `search_document` | 344 | | `document_structure` | 168 |
| `read_document` | 330 | | `read_pdf_text` | 160 (kept) |
| `kept_readings` | 303 | | `find_quantities` | 275 |

**Saving on a turn with no document:**
- 6,513 o200k tokens on every main-agent model call, 29.6 % of the prefix. That is 26,625 chars, about 9.2k tokens at the Claude route's 2.9 chars a token.
- On the build i turn (four calls, 145,102 tokens in), that is about 37k tokens in, roughly a quarter of the turn.
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
| A path typed in chat (library or notebook use, `/Volumes/...`) | the document tools resolve real paths when no folder is bound | no folder bound: the gate hides nothing |
| A URL | no tool on either page fetches a URL (the HTTP code in `webapp/` serves the model engines, SharePoint and storage) | nothing to see |
| Files the agent made (calc packages, plots, Word, PDFs) | written into the working folder | not a document of the user's. `read_pdf_text` and `analyze_pdf_page` stay on the list, because the prompt has the agent read back and look at a PDF it built |

## 3. The same-turn case

- `HideTools` filters `request.tools` in `wrap_model_call`, which runs before every model call of every turn.
- `DocumentToolsGate` (`funhouse_agent/deep/document_gate.py`) subclasses it and decides again on each call: are the attachments non-empty, is there a fetch tool's `ToolMessage` in the request's messages, and does the host's signal say yes?
- A document fetched by a tool call is therefore on the list at the very next model call of the same turn: the tool's result is in the messages, and the ledger was written inside the tool.
- An upload between turns lands in the dict the agent closes over.

**A hidden tool still runs if called by name.** Checked directly: `create_agent` with a `HideTools` middleware executes a call to a hidden registered tool. So a model that calls `open_document` from the prompt's own wording, before any signal, is not refused.

## 4. The decision

**Built.**
- The saving is real: 29.6 % of every main-agent call's prefix on a geotech turn with no document, about a quarter of a short turn's input.
- The same-turn case is handled cleanly by the existing per-call middleware pattern, with no state.

## 5. What changed

- **`funhouse_agent/deep/document_gate.py`** (new):
  - `DOCUMENT_TOOLS`: the 16 hidden tools. They are the trace's 18 less `read_pdf_text` and `analyze_pdf_page`.
  - `FETCH_TOOLS` = `{"sharepoint_download_file"}`.
  - `DocumentToolsGate(attachments, signal=)`: hides nothing when no working folder is bound. A signal that raises counts as "document held", so the tools stay offered as before.
  - Tool results and unknown-tool errors pass through unchanged.
- **`funhouse_agent/deep/agent.py`:**
  - `build_deep_agent(document_signal=None)` is a new keyword.
  - The geotech page's `page_middleware` is now `[HideTools(UNRUNNABLE_TOOLS), DocumentToolsGate(attachments, signal=document_signal)]`.
  - The review-page route (`review_page=True`) does not pass the new keyword on.
- **`webapp/geotech_page.py`:** `document_signal(temp_dir)` returns true when the conversation's downloads ledger is non-empty, or when the bound working folder is not its `files/`. `build_page_agent` passes it in.
- **`webapp/tests/page_fingerprints.json`:** re-recorded in its own commit. One entry moved: `geotech_default` `/calls[0]/middleware` gains `DocumentToolsGate` with its 16 hidden names. No prompt, tool text, helper or budget moved, and the three review builds did not move.
- **`module_work/inf67/measure_prefix.py`:** the measurement above.

**Not done (noted, not fixed):** the general-purpose helper inherits the main agent's tools, and its own calls still list the document tools. Helpers make fewer calls than the main agent, and the gate's job was the main agent's prefix.

## 6. Tests (offline, scripted models, made-up files)

**`funhouse_agent/deep/tests/test_document_gate.py`, 10 passed:**
- a geotech turn with no document has none of the 16 (they stay registered; `read_pdf_text` and `analyze_pdf_page` are offered);
- an attached document turns them on;
- an upload into the same dict after the build turns them on next turn;
- `sharepoint_download_file` called mid-turn turns them on for the next call of that turn, and that call's `open_document` runs;
- the host's signal flipping inside a tool turns them on for the next call;
- no working folder bound hides nothing;
- a signal that raises offers them;
- the Document Review page keeps them;
- the web app's signal reads the downloads ledger and counts a folder the user chose.

**Wider runs, judged by pytest's exit code:**
- `webapp/tests/test_page_fingerprints.py`: 15 passed, exit 0, after the re-record.
- `funhouse_agent/deep/tests`: 1,181 passed, 4 skipped, 1 failed. The failure is `test_wave1_geotech.py::test_the_flags_match_what_the_code_demands` ("liquepy is not installed"; liquepy's dependency eqsig does not build in this container). It fails the same way on `origin/master` here.
- `webapp/tests`: 1,092 passed, 1 failed. The failure is `test_tinyapps_wiring.py::test_ssl_context_accepts_pem_text_or_path_and_rejects_junk` (`pyo3_runtime.PanicException` from the container's system `cryptography` package building an SSL context). It does not touch this change.
- `webapp/tests/test_public_text.py`: 4 passed, exit 0.

## 7. Release note

- Geotech page: the document and drawing tools stay off the model's list until the conversation holds a document (an upload, a SharePoint download, a folder you chose). On a question with no document, every model call is about 30 % smaller.
- When a document arrives, even mid-turn, the tools are offered from the next model call. The Document Review page is unchanged. No answer changes.
