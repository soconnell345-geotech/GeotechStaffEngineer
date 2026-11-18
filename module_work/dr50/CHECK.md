# DR-50 check: a reading helper's open items carry a cut text read and drop a redone zoom

## What the item asked

DR-46 added the open items a reading helper's tools flagged to its `task` result (`keep_work.KeepWork(readers=...)`). It missed three cases:
1. a text read cut at its size limit was not carried, so the main agent was not told the text was cut;
2. a zoom that failed and was then redone with another box stayed listed as unread;
3. the minimal build's helper (`GEOTECH_REVIEW_AGENT=minimal`, a suite arm) gets no open items at all.

The lead's ruling: (1) and (2) are repairs, on by default, with no setting. (3) is built only if it is the same mechanism switched on for that helper, with no change to the arm's prompt or tool list.

## What was built

All of it is in `packages/aec-agent-core/aec_agent_core/deep/keep_work.py`, in `open_items` and four small helpers beside it.

**(1) A cut text read is an open item.** `_TEXT_READ_TOOLS`, `_text_cut`, `_text_cut_read`. Two tools are covered:
- `read_document` (planlens). This is the text read Document Review's reading helper actually has; its tools do not include `read_pdf_text` (`review_agent.READER_TOOLS`). A cut result carries a `next` cursor: `pages` (counted from 0) and `start_line` for a PDF, or `start_line` only for a Word, Excel or DXF file.
- `read_pdf_text` (the core's `vision_tools._dispatch_read_pdf_text`), the tool the item names. A cut result has `truncated: true`, and its `truncated_note` names the pages to ask for next (`pdf_pages='N-M'`).

The line goes under the file and the page to go on from (as a viewer counts pages), most serious first, right after a cut-off vision answer. It quotes the tool's own cursor or note, so the main agent sees where to go on from in the tool's words. Code reads only the page number from it, to group and settle the line. For example:

```
- doc_1 page 5: a text read (read_document) was cut at its size limit, so the rest of the text it asked for was not read (the tool's cursor to go on from: next={"pages": "4-7", "start_line": 31})
- made_up_set.pdf page 3: a text read (read_pdf_text) was cut at its size limit, so the rest of the text it asked for was not read (the tool's note: Output hit the character budget. Request specific later pages (e.g. pdf_pages='3-10') to continue reading)
```

The line is settled by a later read of the same file, by the same tool, that read on from the cut:
- `read_pdf_text`: the later read returned that page.
- `read_document` on a PDF: the later read returned that page, and if the page was the first one it returned, it started at or before the cut line.
- `read_document` on a file read by lines: the later read started at or before the cut line and returned text.

Nothing else settles it: not a failed read, other pages, another file, or a look at the page. If the later read was cut again further on, only the new cut is listed.

**(2) A failed zoom redone in another box is settled.** It is settled by a later zoom (`render_region`) that came back without an error, on the same file and page, asking the same question (the `prompt` as sent, on one line). The failure's own key and its question are kept in `failed_asks`. Only the error is dropped; the later zoom's own flags still stand, such as lettering past its box's edges.

**(3) Not built.** See "Noted, not fixed".

## Calls made

- **`read_document` is covered as well as `read_pdf_text`.** The item names `read_pdf_text`, but Document Review's reader cannot call it, so a fix to that tool alone would change no Document Review run. `read_document` is the same case: the text read of pages the helper asked for, cut at its size limit, with the tool saying where to go on. Covering it was the smaller error. The lead may narrow it back.
- **`read_text_file` is left out.** It returns as many characters as were asked for, from an offset, and a reader opens its kept files at the offsets their index gives. A part of a file is what it asked for, so listing every such read would bury the real holes. That is the same reason DR-46 left out tiles-only rows.
- **A look at the cut page does not settle a cut text read.** A look reads that page, not the pages after it in the range that was asked for. Conservative, as the lead asked: a false "unread" costs a re-read.
- **"The same ask" means the same `prompt`, not empty.** A failed zoom that sent no question of its own stays listed, even when a later zoom without one came back from another box: two zooms with the tool's default question could be on two different parts of the page. Another question, another page, or the whole-page look (`analyze_pdf_page`) does not settle it either. The same box still settles it, as in DR-46.
- **The page to go on from is read from the tool's note or cursor, never worked out.** A `read_pdf_text` note that names no page is still listed, under the file alone, and nothing settles it.

## Settings

None. Both changes are repairs, on by default. The geotech page names no reader (`KeepWork()` with no `readers`), so its helpers get exactly what they got before. A reader with no cut text read and no failed look gets exactly the result DR-46 gave it.

## Tests

`packages/aec-agent-core/tests/test_dr50_open_items.py` (new; no model, made-up values):
- `test_the_real_text_read_cut_at_its_budget_is_carried_where_it_says`: the core's real `read_pdf_text` on a made-up four-page PDF. The cut is listed at the page its note names, with the note's words; reading on from there settles it. This pins the coupling to the tool's note.
- `test_a_cut_text_read_stays_until_it_is_read_on_from_there`: other pages, a look at the page, another file and a failed read do not settle the cut; reading on does; a second cut further on replaces the first.
- `test_a_cut_note_that_names_no_page_is_carried_and_stays`.
- `test_a_cut_document_read_carries_the_tool_s_own_cursor`: `read_document` on a PDF; three ways of reading on settle it, while later pages only, a start past the cut line, or another handle do not.
- `test_a_cut_read_of_a_file_read_by_lines_is_carried`: `read_document` on a Word file.
- `test_a_part_of_a_file_read_by_offset_is_no_open_item`: `read_text_file`.
- `test_a_failed_zoom_redone_in_another_box_with_the_same_ask_is_settled`: a box given two ways, the question with other spacing, and a failure by the tool's status.
- `test_a_failed_zoom_stays_when_the_redo_is_not_clearly_it`: another question, no question, another page, the whole-page look, a redo that failed too, an earlier success; and the same box still settles it.
- `test_a_redo_settles_only_the_failure_not_its_own_flags`.
- `test_a_clean_reader_s_result_is_exactly_as_before`: a look, a zoom, a whole `read_pdf_text`, two whole `read_document` reads and an offset `read_text_file`. The whole `task` result is checked word for word: the answer, then DR-46's kept-file note, and no open-items block.
- `test_the_geotech_page_s_helpers_are_still_unchanged`: no reader named, so even a helper with a cut text read and a failed zoom comes back as it did, and nothing is kept.
- `test_a_reader_s_result_names_the_cut_and_the_kept_file_lists_it`.

`packages/aec-document-review/tests/test_dr50_reader_open_items_offline.py` (new) runs the page's real build for one turn, on a scripted model and a fake vision engine, as DR-46's boundary test does. Reader T reads six made-up pages with one `read_document` call; it stops at its size limit with a cursor. T then zooms with a box given two ways (the zoom fails) and zooms again in another box with the same question, then answers "Open items: none.". Reader L looks at one page. The test asserts that the main agent receives:
- T's cut, with the tool's own cursor and the right page;
- no failure for the redone zoom;
- for L, its answer and its kept-file note only.

On the base branch (`claude/dr-42-page-texts`, `keep_work.py` swapped back), these fail:
- 8 of the 12 core tests: all but the clean reader, the geotech helpers, the offset read and the "stays" test, which pass on both by design;
- the Document Review turn: no cut is listed, and the redone zoom is listed as failed.

Every one passes on this branch.

Also run, all green: `test_dr46_open_items.py`, `test_core_imports.py`, `test_saved_files_and_settings.py`, `funhouse_agent/deep/tests/test_keep_work_offline.py`, `test_records_as_files_offline.py`, `webapp/tests/test_kept_turn_work.py`, `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`, and Document Review's `test_dr46_reader_boundary_offline.py`, `test_kept_turn_work.py`, `test_review_fingerprints.py`, `test_dr42_notes_name_page_tools.py`.

## Fingerprints

Neither moves: no prompt, tool description, schema, helper text or middleware changed, only what a `task` result carries.

## Noted, not fixed

- **The minimal build's helper (part 3) needs more than switching DR-46's mechanism on.** Its `task` (`minimal_agent._helper_tool`) is a tool of its own that runs a helper agent; nothing records that helper. Putting `KeepWork(readers={"page_looker"})` on the main agent and `HelperRecorder` in the helper would also change three other things:
  - it keeps a file and names it in a note that says to open it with `read_text_file`, a tool the arm does not have;
  - it caps the arm's helpers at once (`GEOTECH_MAX_HELPERS`, 2 by default), a cap the arm has never had;
  - it marks the helper's calls as a helper's. That moves its budget-reached note from the turn to the helper (`note_budget_reached`; today a minimal helper that uses up its budget marks the turn's budget as reached), and it adds task fields to the activity log.

  Open items alone would take a recording path of the helper's own (a `HelperRecord` set around its `invoke`, the recorder in its middleware, `open_items_note` added to its answer) and a decision on that budget note.
- **A `read_pdf_text` result cut by the result cap is not parsed.** If a result runs past the page's result cap, the tool kit cuts the JSON as a string, and `open_items` cannot read it. The text budget (12,000 characters) sits under the cap (16,000), so this should not happen.

## For the lead: what a measurement on the hosted platform should compare

In a whole-set review's saved runs, count the reader results whose kept record holds a `read_document` result with `next` (or a `read_pdf_text` result with `truncated`) that the helper did not read on from. Before this branch the main agent was never told about these cuts; after it, each one is a line in the open items. Then check that the main agent then reads on, or hands the rest to another helper, instead of reporting "not found" on pages it never read. The failed-and-redone zoom was 1 in 1,802 helper results in DR-46's runs, so no measurable change is expected from it.
