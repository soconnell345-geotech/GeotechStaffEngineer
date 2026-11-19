# CHECK B15 / PL-09: the three reading options for a host that asks

Branch `claude/b15-pl09-host-options` from `main` 7eed79b (v0.17.0). One source commit, 2db8016; this file alone in the last.

## Confirmed on main before changing it
- `Document(..., column_warning=, continued_tables=)` (`planlens/document/document.py`) and `outline_from(..., captions_on_every_page=)` (`roles.py`) exist; `ReviewToolkit.open` built every `Document` without them and `_tool_document_roles` called `outline_from` without the option.
- `_new_toolkit` (`planlens/review/conversation.py`) passes the FIRST host's `toolkit_options` to `ReviewToolkit(**kw)`. On main a host with `toolkit_options={"column_warning": True}` therefore failed (`TypeError`, unexpected keyword) when the toolkit was built; the new tests fail on main that way.
- `document_roles` is a toolkit tool; the host layer does not serve it ("unknown tool").

## Changed (78 changed source lines: toolkit.py +58 -5, conversation.py +15)
- `planlens/tools/toolkit.py`: `READING_OPTIONS`; `ReviewToolkit(column_warning=, continued_tables=, captions_on_every_page=)`, each a bool or a callable asked at each call (as `dxf_plots`); `_reading`, `_document_options`. `open` passes the two `Document` options and adds `:column_warning` / `:continued_tables` to the content key (nothing with both off, so the default key and handle are what they were). `_entry` answers a handle whose document was opened under other options from the same source opened with the call's options. `_tool_document_roles` passes the captions option to `outline_from` at each call.
- `planlens/review/conversation.py`: `_asked_for_reading(host, option)`; `_new_toolkit` gives the shared toolkit one callable per option that answers for the CALLING host, overriding whatever the first host's `toolkit_options` held.
- Names kept: `column_warning`, `continued_tables`, `captions_on_every_page` (they are the names `Document` and the outline use, and read like `dxf_plots`).
- All three are off unless asked; no setting, no repair. What the options do is untouched (no change under `planlens/document/`, bar one DESIGN.md sentence).
- `CHANGELOG.md` (Unreleased), `planlens/document/DESIGN.md` (the sentence that said neither passes an option through), `planlens/review/DESIGN.md` (how hosts are kept apart).

## Each option: tool, made-up before and after, characters added to a specification
| Option | Result that changes for a host that asks | Spec |
|---|---|---|
| `column_warning` | `read_document`: one `!` line under the header of a page set in columns and drawn row by row; the document's handle | 0 |
| `continued_tables` | `read_document` with `include_tables`: a `note:` line under a table that runs on; the handle | 0 |
| `captions_on_every_page` | `document_roles(outline=true)` on a `ReviewToolkit`: captions and placed entries. Through `ReviewHost`: no result (the tool is not served) | 0 |

- column_warning, before: `=== page 0 [pdf_page 1] [text] ===\nA MADE-UP HEADING ...`. After: `=== page 0 [pdf_page 1] [text] ===\n! reading order not reconstructed: the text is set in 2 columns and the file draws its lines across them ...\nA MADE-UP HEADING ...` (+187 characters on the made-up page).
- continued_tables, before: `| TH-30 | 30.5 | 50 | none |` then page 1. After: the same row, then `note: continues on page 1 as table p1.tbl0 ...`; and under page 1's table `note: continues table p0.tbl0 from page 0: its first row repeats that table's column titles` (+137 characters).
- captions_on_every_page, before: `"n_entries_placed": 0, ... "captions": []`. After: `"n_entries_placed": 2`, `"page": 2` on Table 1, `"captions": [{"kind": "caption", "number": "1", "text": "Table 1: Summary of Laboratory Test Results", "page": 2}, ...]`.
- Handle: for the first two the asking host's handle differs from the plain one (`doc_e0829848f4` -> `doc_588ffaae7a`), as a plotted DXF's does.

## Tests
New, `planlens/review/tests/test_b15_pl09_host_options.py`, 9 tests; all 9 FAIL on main (source stashed: `9 failed`, exit 1) and pass on the branch:
- `test_two_hosts_on_one_file_each_get_their_own_reading` x4 (two options, plain first and asked first): the plain host's result equals what it gets alone in a conversation; the asked result is that plus the added lines and its own handle; each host given the OTHER's handle is answered with its own reading.
- `test_each_option_is_asked_for_by_itself` x2; `test_the_captions_are_the_calling_hosts_on_the_shared_toolkit`; `test_a_toolkit_takes_each_option_and_is_plain_without`; `test_asking_moves_no_tool_name_and_no_specification`.

Runs (`/tmp/pl/bin/python -m pytest -q -p no:cacheprovider ...`, exit code read directly, venv as the common brief's planlens setup):
- the new file: 9 passed.
- the new file + the 6 files naming an option + the 11 host-layer files naming `toolkit_options` + `test_host_specs.py`: 114 passed, 3 skipped (DXF plots need Pillow), exit 0. With Pillow installed (as CI) and `test_rv_b13_forget_closes_a_dxf_plot.py` added: 119 passed, 0 skipped.
- beyond what was asked, `planlens/tools planlens/review` whole: 473 passed, 7 skipped, exit 0 (before the docs edits; no source change after).
- The task's `git grep -l "toolkit_options" -- "planlens/**/tests"` matches nothing in this git; the files were listed with `git grep -l toolkit_options -- planlens | grep /tests/`.

## Fingerprints / the default surface
Not moved. `planlens/review/tests/test_host_specs.py` (descriptions and schemas word for word against `app_tool_specs.json`) passes unedited; nothing re-recorded. No existing test edited. The app's tests were not run: no default result or specification changes.

## Calls made
- A handle is answered with the CALLING host's options (`_entry` reopens from the entry's source). This goes one step past the DXF rule, which only keeps the key apart: without it a host that did not ask, given a handle from an earlier turn of the conversation, would have read the other host's page. The result names the handle the call gave.
- `captions_on_every_page` is not part of a document's key: nothing of it is stored on the document, so two hosts share one document safely and the asking host's handle does not move.
- A bool or a callable for each toolkit option, as `dxf_plots`.

## Release note
For a host that asks, `read_document` can carry a warning on a page whose columns are drawn row by row and a note on a table that continues on the next page. Nothing changes for a host that does not ask.

## For the lead
- `captions_on_every_page` reaches no tool a `ReviewHost` serves. To give it an effect there, `document_roles` (or its outline) must be served by the host layer for a host that asks: a new opt-in tool, so a specification question for the owner.
- Neither option has been run over real pages or past a model (PL-09's own open point); the column check still warns on some unruled tables.
- The MCP server (`planlens/tools/mcp_server.py`) has no flag for the three options; not asked for.
- README.md's 0.17.0 line "three `Document` options no tool passes yet" describes 0.17.0 and was left; reword it at the next release.

## Noted, not fixed
- A handle whose document was opened under other options is reopened from its source; if that source no longer resolves, the call is refused ("could not open") where the same host's own handle would still answer.
- With both `Document` options asked the key carries both suffixes: up to four documents of one file (max_open counts each).
