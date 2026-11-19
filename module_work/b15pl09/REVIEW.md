**MERGE AS IS** — `claude/b15-pl09-host-options` at bc9c986 (source commit 2db8016), reviewed against 7eed79b (v0.17.0). No must-fix finding, no fix made; this branch adds one test file and this report.

# Review B15 / PL-09: the three reading options for a host that asks

Every answer below comes from a run on a made-up file built in the script (page 0: two columns drawn row by row; pages 1-2: one ruled table that runs on; a list of tables; a caption the list does not name), on two trees: 7eed79b in a second worktree (`git worktree add`, imported by `PYTHONPATH`, the tree printed at each run) and the branch.

## Answers

1. **Each option alone reaches its result and only that.** Through `ReviewHost`, 17 calls over 15 tools, asking host beside a host that does not ask:
   - `column_warning`: one `!` line under page 0's header in `read_document`; nothing else differs.
   - `continued_tables`: two `note:` lines in `read_document` with `include_tables` (`continues on page 2 as table p2.tbl0`; `continues table p1.tbl0 from page 1 ...`); nothing without `include_tables`; nothing else differs.
   - `captions_on_every_page`: no result of any host tool differs, and the handle is the plain one. On the conversation's shared toolkit `document_roles(outline=true)` gives the asking host the caption `(page 4, "8")` that no list names, and the other host `[]`, on one document.
   - The asking host's handle differs for the first two options (as a plotted DXF's does). No specification moves.
2. **A host that does not ask: byte for byte 7eed79b.** One JSON of 91 values per tree (the host and toolkit specifications, 77 tool results for three hosts on a file in a folder and on an upload, the marks, the toolkit's keys and handles, the list of opens): the two files are identical (`cmp`, same sha256). Opens are 1 for two hosts on one file, 3 after a marked copy and a forget, in both trees: nothing is opened twice. `test_host_specs.py` passes; `app_tool_specs.json` and the test are untouched since 7eed79b.
3. **Two hosts on one file: B is never handed option output, A always is.** 16 runs (each option alone, both `Document` options together; A first and B first; A forgets and B forgets), each host calling every reading tool with its own handle, with the OTHER's handle, and again after the forget and reopen: 0 results of B differ from the plain host's; A's `read_document` carries its lines in every one.
   - (a) Marks: `annotate_document` writes a copy on disk, not the open document. A's mark and B's mark land in the one `set_marked.pdf` (`in_file.markups` 2); `document_markups` shows both to both hosts. As at 7eed79b. A forget closes both hosts' documents of the file (2).
   - (b) `max_open`: see finding 1. B's document IS closed under A's documents, but its handle is never refused and no mark is lost (none is held in memory).
   - (c) Each host finds its own document again under its own handle after a forget (same handle, same results). Nothing is stored per document that a second document loses but the page caches. In a conversation where no host asks, nothing is computed twice (answer 2).
4. **A wrong name or value is handled as 0.17.0 handles its options.** `{"column_warnings": True}` on the first host: every call answers `TypeError ... unexpected keyword argument 'column_warnings'`, word for word what `{"dxf_plot": True}` gives in both trees; on a later host it is ignored, as before. A value is read by truth: `"yes"`, `"false"` and `1` turn the option on, `0` and `None` do not; `sheet_tools: "false"` offers the sheet tools at 7eed79b the same way.
5. **Suites and documents.** Counts below. CHANGELOG and both DESIGN.md say what the code does; the one thing they leave out is finding 1.

## Findings (all notes; none changes a result a host is handed)

| # | What | Made-up input | Before (7eed79b) | After | |
|---|---|---|---|---|---|
| 1 | One file is two open documents when one host asks, so `max_open` (8) is reached sooner and the other host's document is closed under it | B opens `f0.pdf`; A (both options) opens `f0`..`f7` | 8 opens; B's document stays open | 10 opens; B's document closed, then reopened without a word at B's next call: same handle, `read_document`, `document_page_map`, `document_markups`, `annotate_document` byte for byte | noted; worth one sentence in `review/DESIGN.md` |
| 2 | The added lines count against a result's size | A (both options) reads pages 0-6 with tables and locations at 12,000 characters | pages 0-5 returned | pages 0-4 returned, `next` at page 4 | noted, expected |
| 3 | `captions_on_every_page` changes no result through `ReviewHost` | answer 1 | | | noted; the builder says so; serving `document_roles` is the owner's question |
| 4 | A string value such as `"false"` turns an option on | answer 4 | the same for `sheet_tools`, `dxf_plots` | the same | noted; not new |

## What I ran

- Builder's claims on the base, confirmed: at 7eed79b a first host with `column_warning` fails on `TypeError`; a later one is ignored.
- `planlens/review planlens/tools` and the three `planlens/document/tests/test_pl09_*.py` files on the branch (venv of the common brief plus Pillow, `-p no:cacheprovider`, exit code read directly): **588 passed, 2 skipped, exit 0**.
- The builder's `test_b15_pl09_host_options.py`: 9 passed on the branch; copied to the base: **9 failed** (with `test_host_specs.py`: 26 passed there). It tests results, not the presence of text.
- New here, `planlens/review/tests/test_rv_b15_pl09_host_options.py`, own made-up file: **17 passed** on the branch; on the base 16 failed and 1 passed by design (an unknown name is refused the same way). With the builder's file and `test_host_specs.py`: 52 passed.
- Scope: source changes only in `planlens/tools/toolkit.py` and `planlens/review/conversation.py`; no existing test edited; nothing under `planlens/document/` but one DESIGN.md sentence; no word the public-repository rule forbids.
- Threads: the two pools that call the toolkit (`find_like`, `markup_check`) copy the context, so the calling host is the one asked there.

## Not checked

- The app (Document Review) against this branch, and its fingerprints: no default specification or result moved, so not run.
- Real pages, a model, the MCP server, OCR text sources.
- A DXF plot opened by a host that asks for a reading option, then its handle given to a host that asked for no plot.
- A handle whose source no longer resolves (the builder's "Noted, not fixed"): read, not run.
- Two hosts calling at the same moment on threads: `_entry` reopens outside the toolkit's guard and `open` takes it; read, not run under load.
- The whole planlens suite (`planlens/document`, `ir`, `pdf`, `dxf` beyond the files named): for the `ci/` run.
