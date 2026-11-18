# REF-114: the reference library's table reader in the core (check notes)

## What the item asked

Two gaps under `packages/aec-agent-core`, found by the review of the
references helper's table tools (REF-92), neither seen by a user today:
(1) the vision tool lists name no table reader; (2) `cited_documents` does
not count a table read as a reference source. The table reader is the
geotech page's `read_reference_table` (it renders a reference table's
printed page and reads it with vision; built in `funhouse_agent/deep/tools.py`
and `funhouse_agent/reference_vision.py`, not in the core).

## What was built, and where

- (2) `aec_agent_core/cited_documents.py`: `REFERENCE_TOOLS` gains
  `read_reference_table`, and a new constant `REFERENCE_PAGE_READERS`
  (`read_reference_figure`, `read_reference_table`) names the readers whose
  `reference` argument names the library document they render.
  `_note_call` marks that document looked up for either reader (it did so
  for the figure reader only). A table read now counts exactly as a figure
  read: as a references result, by the reference the call names, whatever
  its text prints. The module docstring says so.
- (1) `aec_agent_core/vision_tools.py`: `APP_READER_TOOLS` names
  `read_reference_table` beside the figure reader and the worked examples'
  page reader. As for those two, the core holds no reader: a call goes to
  the hook `vision_tools.read_reference_table` where the app filled it and
  is an unknown tool where it did not (the same result, word for word, as
  before the core listed it). The hook is added to the list in
  `aec_agent_core/hooks.py` and to the hooks table in
  `packages/aec-agent-core/README.md`.
- (1) `aec-document-review/aec_document_review/review_build.py`:
  `GEOTECH_ONLY_TOOLS` (the geotech page's tools this page does not have)
  names the table reader too. Nothing reads it in the package; it is the
  documented list, and the page is built without the reader as before.

## Calls made

- **The table reader gets its own hook, with no fall-back to the figure
  reader's.** The geotech page fills only the figure reader's hook and
  sends its table reads under that name (its figure hook passes a call
  naming a `table_number` to the table reader). Falling back to the figure
  hook would rest on that one app's contract; another app's figure reader
  need not take a table call. So a call by the table reader's name reaches
  a reader only where an app fills `vision_tools.read_reference_table`.
  The geotech page's routing is unchanged.
- **`EXTENDED_TOOLS` is not changed.** It is the other core list without
  the table reader. The geotech page's text-mode agent
  (`funhouse_agent/agent.py`) checks a model's tool call against it and
  accepts `read_reference_table` only where its prompt offers the reader
  (`TABLE_READ_SECTION`: a host with the reference PDFs and the table
  catalogs). Adding the name to `EXTENDED_TOOLS` would make that agent
  accept a call to it on every host, with cited-documents mode off: a
  change to what the model is handed (a dispatched read in place of
  "Invalid tool_name"). Left for the lead's ruling.
- **`VISION_TOOL_DESCRIPTIONS` is not changed.** It is the text-mode tool
  text (it names `read_reference_figure` as tool 10, no table reader). The
  geotech page appends its own `TABLE_READ_SECTION` where the reader is
  offered; adding a section in the core would change the default prompt
  and print the reader twice there. Left for the lead's ruling.
- **A recorded check changes with the mode off.** The turn's
  `cited_documents` event in `activity.jsonl` (and a writer's) is written
  whatever the mode. After a main-agent table read it now lists the
  document under `looked_up` where it listed it under `not_looked_up`.
  This is the repair itself; neither the model nor the user is handed the
  event (the run report names it as a record line).

## Settings

None added. `GEOTECH_CITED_DOCUMENTS` (off by default; `note` or `retry`)
is where (2) shows: with it on, a table read no longer draws a false
"cited but never looked up" note on the answer, the retry note or a
writer's `citation_note`.

## Tests

New file `packages/aec-agent-core/tests/test_ref114_table_reads.py` (core
only; no app import, made-up values, no model):

| Test | Proves | On the release this starts from |
|---|---|---|
| `test_the_core_lists_the_table_reader_among_the_app_readers` | (1) the list names it; the two others stay | fails |
| `test_an_app_serves_the_table_reader_by_its_own_name` | (1) a filled hook gets the call, its arguments and engine | fails (unknown tool) |
| `test_without_its_hook_the_table_reader_is_unknown_as_before` | (1) with only the figure hook filled (the geotech page today) a table-name call is unknown and never reaches the figure reader; a figure-name call still does | passes (unchanged) |
| `test_a_table_read_is_a_reference_read` | (2) `source_kind` gives a table read the figure read's kind | fails |
| `test_without_a_read_the_citation_is_not_looked_up` | the control: nothing read, the citation is flagged | passes |
| `test_a_page_read_of_a_reference_looks_its_document_up[figure / table]` | (2) a read whose text names no document looks up the `reference` it names, the table read as the figure read | table case fails; figure case passes |
| `test_a_table_read_of_another_reference_does_not_look_this_one_up` | (2) only the named reference counts | passes |
| `test_mode_off_hands_the_model_and_the_user_the_same[no read / table read]` | mode off: the answer goes out as written, no model note, a Word writer's result unchanged and no fields added, with or without a table read; only the record differs | table-read case fails at the record's assert only |
| `test_note_mode_no_longer_flags_a_table_read` | `note`: the answer is noted without a read and not after a table read; a writer adds nothing after it | fails |
| `test_retry_mode_gives_no_note_after_a_table_read` | `retry`: no note to the model after a table read | fails |

"On the release" was run by putting back the two changed core files
(`cited_documents.py`, `vision_tools.py`) as they stand before the item's
commit and running the new file: 7 failed, 5 passed, exactly as the last
column says (the mode-off table-read case at its record assert, line 209).
On the branch all 12 pass.

Existing tests run, all exit 0 (316 passed with the new file; 25 passed in
Document Review's run): the core's
`test_core_hooks.py` (it iterates `APP_READER_TOOLS`: the new name is an
unknown tool without its hook), `test_core_imports.py`,
`test_core_suite_and_records.py`; the app's `test_cited_documents.py`,
`test_cited_documents_turn.py` (both pages), `test_core_move_11a.py`,
`test_core_move_11c.py`, `test_table_reach.py`,
`test_read_reference_figure.py`, `test_review_switches_offline.py` (both);
the fingerprints of both pages (`webapp/tests/test_page_fingerprints.py`,
Document Review's `tests/test_review_fingerprints.py`). No existing test
was edited.

## Noted, not fixed

- Document Review's `stream_format._KNOWN_TOOLS` and
  `helper_models.describe` name the figure reader and not the table reader;
  this page has no reference readers, so neither line can show.
- The core's comments on chart read-offs (`vision_view.DEFAULT_CHART_BUDGET`,
  `vision_tools.VISION_OUTPUT_CAPS`) name the figure reader only; the
  geotech page's table read runs under the same chart budget and kind.

## For the lead

- Nothing to measure on the hosted platform for (1): no page dispatches
  the table reader by its own name today. The geotech page can fill
  `vision_tools.read_reference_table` (its `_dispatch_read_reference_table`)
  and drop the routing under the figure reader's name in a later release of
  the app (the maintainer's code).
- For (2), with `GEOTECH_CITED_DOCUMENTS=note` on the geotech page: a turn
  where the main agent reads a page-only table (`read_reference_table`)
  and cites its document with no earlier `table_db` or references result
  should carry no citation note, where the release this starts from noted
  it.
