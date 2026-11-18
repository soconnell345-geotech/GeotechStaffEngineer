# DR-50 check: a reading helper's open items carry a cut text read and drop a redone zoom

## What the item asked

DR-46 added the open items a reading helper's tools flagged to its `task` result (`keep_work.KeepWork(readers=...)`). It missed three cases:
1. a text read cut at its size limit was not carried, so the main agent was not told the text was cut;
2. a zoom that failed and was then redone with another box stayed listed as unread;
3. the minimal build's helper (`GEOTECH_REVIEW_AGENT=minimal`, a suite arm) gets no open items at all.

The lead's ruling: (1) and (2) are repairs, on by default, with no setting. (3) is built only if it is the same mechanism switched on for that helper, with no change to the arm's prompt or tool list.

## What was built

All of it is in `packages/aec-agent-core/aec_agent_core/deep/keep_work.py`, in `open_items` and the small helpers beside it.

The lead's standing ruling governs both parts: where it is not clear that a later call replaced an earlier one, the earlier one stays listed. A false "unread" costs a re-look; a false "read" loses evidence. The review found the first build too ready to settle in both parts; the rules below are as rebuilt after that review.

**(1) A cut text read is an open item.** `_TEXT_READ_TOOLS`, `_text_cut`, `_text_read_on`, `_text_cut_words`, `_pdf_text_asked`. Two tools are covered:
- `read_document` (planlens). This is the text read Document Review's reading helper actually has; its tools do not include `read_pdf_text` (`review_agent.READER_TOOLS`). A cut result carries a `next` cursor: `pages` (counted from 0) and `start_line` for a PDF, or `start_line` only for a Word, Excel or DXF file.
- `read_pdf_text` (the core's `vision_tools._dispatch_read_pdf_text`), the tool the item names. A cut result has `truncated: true`, and its `truncated_note` names the pages to ask for next (`pdf_pages='N-M'`).

A cut keeps what it left unread, in the tool's own page numbers (`_text_cut`):
- `read_document`: the pages of its `next` cursor, the first from the cursor's `start_line` (planlens 0.16.0 `_tool_read_document`: the cursor is `{"pages": <the pages asked for from the cut page on>, "start_line": <line>}`; `pages_returned` lists the pages it returned, the cut page included once some of its lines are out). A file read by lines owes its cursor's `start_line`.
- `read_pdf_text`: the pages the call asked for, from the page its note names on (the page cut short, or the one after the last read whole). The pages asked for are resolved by the tool's own reading of its arguments (`vision_tools._read_pdf_text_pages` and `_parse_pages`: `pdf_pages`, `pages`, or no page argument for its first `max_pages`), so nothing is worked out here that the tool does not do itself. It reads a page from its top.

Later reads of the same file by the same tool take off what they read (`_text_read_on`):
- a page they returned whole, from at or before the first line owed on it: `start_line` applies to the first page a read returned, so a read that starts on an owed page past its owed line reads none of it;
- the page a later read was itself cut on counts as read up to that cut (its owed line moves on), and the later read's own cut is added to what is owed;
- a file read by lines: a later read that started at or before the owed line and returned text; cut again, the owed line moves to its cut.

The cut is settled when nothing is left. All the cuts of one file by one tool are one line, under the first page still owed (as a viewer counts pages), most serious first, right after a cut-off vision answer. While nothing has been read since the last cut, the line quotes the tool's own cursor or note. Once part of it has been read, it names what is left as the tool's own arguments, so the main agent can call the tool with them (`_text_cut_words`). For example:

```
- doc_1 page 5: a text read (read_document) was cut at its size limit, so the rest of the text it asked for was not read (the tool's cursor to go on from: next={"pages": "4-7", "start_line": 31})
- doc_1 page 5: a text read (read_document) was cut at its size limit, so the rest of the text it asked for was not read (still to read, in the tool's own numbers: pages='4-9')
- doc_1 page 6: a text read (read_document) was cut at its size limit, so the rest of the text it asked for was not read (still to read, in the tool's own numbers: pages='5-9', start_line=12)
- made_up_set.pdf page 3: a text read (read_pdf_text) was cut at its size limit, so the rest of the text it asked for was not read (the tool's note: Output hit the character budget. Request specific later pages (e.g. pdf_pages='3-10') to continue reading)
- made_up_set.pdf page 4: a text read (read_pdf_text) was cut at its size limit, so the rest of the text it asked for was not read (still to read, in the tool's own numbers: pdf_pages='4-8')
```

Nothing else settles it: not a failed read, other pages, another file, or a look at the page. A helper that goes on page by page, or in smaller ranges, settles it only when its reads between them have returned every page the cut left; one that stops early leaves the pages it did not reach.

**(2) A failed zoom is settled by a real redo.** A failed zoom (`render_region`) keeps its question (the `prompt` as sent, on one line), the box it asked for, whether the failure was in the box's arguments, and where its result came in the record (`_failed_zoom_box`). A later zoom settles it only when all of these hold:
- it is on the same file and page, asks the same question, and did not fail;
- it was called in a model step after the failure's result came back. The record keeps each model step and then the results of its calls, so the step that sent the redo must come after the failed result (`HelperRecord.steps`). A zoom sent in the same step as the failed one went out before the failure was known, so it is not a redo. If the record cannot show where the redo was called, the failure stays.
- when the failed zoom had a box, the redo's box overlaps it (positive area; boxes that only touch do not). The failed zoom's box is its `bbox` of four numbers, or its `view` + `image_box` turned into page points by the function the tool itself uses (`vision_view.image_box_to_page`). The redo's box is its `bbox`, else its result's `bbox`. One redo settles only the failures whose boxes it overlaps.
- when the failure was in the box's arguments, any such redo settles it: this is the helper fixing a bad box. These are the failures `_dispatch_render_region` returns before it renders: a `bbox` given with a `view` or `image_box`, and a `view` + `image_box` the conversion refuses. A `bbox` that is not four numbers is counted with them: there is no box to compare a redo with.

A failed zoom that named no box (a whole page or an image) is settled only by the same zoom again (DR-46's rule). Only the error is dropped; the redo's own flags still stand, such as lettering past its box's edges.

**(3) Built afterwards, on its own branch.** See "Part 3: the minimal build's helper" at the end of this file.

## Calls made

- **`read_document` is covered as well as `read_pdf_text`.** The item names `read_pdf_text`, but Document Review's reader cannot call it, so a fix to that tool alone would change no Document Review run. `read_document` is the same case: the text read of pages the helper asked for, cut at its size limit, with the tool saying where to go on. Covering it was the smaller error. The lead may narrow it back.
- **`read_text_file` is left out.** It returns as many characters as were asked for, from an offset, and a reader opens its kept files at the offsets their index gives. A part of a file is what it asked for, so listing every such read would bury the real holes. That is the same reason DR-46 left out tiles-only rows.
- **A look at the cut page does not settle a cut text read.** A look reads that page, not the pages after it in the range that was asked for. Conservative, as the lead asked: a false "unread" costs a re-read.
- **"The same ask" means the same `prompt`, not empty.** A failed zoom that sent no question of its own stays listed, even when a later zoom without one came back from another box: two zooms with the tool's default question could be on two different parts of the page. Another question, another page, or the whole-page look (`analyze_pdf_page`) does not settle it either. The same box still settles it, as in DR-46.
- **Same page and same question in another box is not a redo.** Covering a sheet in several zooms with one general question is an ordinary way to read its notes, so a zoom with the same question is a redo only when it was sent after the failure came back and its box overlaps the failed one (review finding 2).
- **The pages left are the tool's own, never worked out.** For `read_document` they are the cursor's pages. For `read_pdf_text`, whose note names only the page to go on from (its range is an example), they are the pages the call asked for, as the tool's own functions resolve its arguments. A `read_pdf_text` note that names no page, or a call whose pages cannot be resolved, is still listed, under the file alone, and nothing settles it.
- **One line per file and tool.** Several cuts of one file are merged into what is still owed on it, so the line names one set of pages to read. Where pages are left from more than one place, each run of pages is its own argument (`pages='4-5', start_line=31; pages='9'`).
- **Each result is matched to the latest call with its id sent before it**, not the last one anywhere in the record. With unique ids, as the models give them, this is what DR-46 did; it is needed to place a call in its model step.

## Settings

None. Both changes are repairs, on by default. The geotech page names no reader (`KeepWork()` with no `readers`), so its helpers get exactly what they got before. A reader with no cut text read and no failed look gets exactly the result DR-46 gave it.

## Tests

`packages/aec-agent-core/tests/test_dr50_open_items.py` (new; no model, made-up values):
- `test_the_real_text_read_cut_at_its_budget_is_carried_where_it_says`: the core's real `read_pdf_text` on a made-up four-page PDF. The cut is listed at the page its note names, with the note's words; reading on to the end settles it; the cut page alone leaves the page after it, named as `pdf_pages`. This pins the coupling to the tool's note and its reading of `pdf_pages`.
- `test_a_cut_text_read_stays_until_it_is_read_on_from_there`: other pages, a look at the page, another file and a failed read do not settle the cut; reading on does; a second cut further on carries on as the new cut.
- `test_a_cut_text_read_stays_until_every_page_it_left_is_read` (review finding 1): `'1-8'` cut at 3, then page 3 alone, or a narrower read ending at page 3, leaves `pdf_pages='4-8'`; reads that between them return pages 3 to 8 settle it, and missing one page leaves that page.
- `test_a_cut_note_that_names_no_page_is_carried_and_stays`.
- `test_a_cut_document_read_carries_the_tool_s_own_cursor`: `read_document` on a PDF; reading every cursor page on from the cursor settles it; part of its pages, a start past the cut line, or later pages only leave what is left, named in the cursor's numbers; another handle reads none of it.
- `test_a_cut_document_read_stays_until_every_cursor_page_is_read` (review finding 1): a cursor over pages 3 to 9, then page 3 alone from the cut line, or `'1-3'`, leaves `pages='4-9'`; page by page to the end settles it; a read cut again on page 5 leaves `pages='5-9', start_line=12`; the same cursor cut again gives the tool's own words again.
- `test_a_cut_read_of_a_file_read_by_lines_is_carried`: `read_document` on a Word file.
- `test_a_part_of_a_file_read_by_offset_is_no_open_item`: `read_text_file`.
- `test_a_failed_zoom_redone_in_another_box_with_the_same_ask_is_settled`: failures in the box's arguments (a box given two ways, a `view` the tool refuses, a box of three numbers), redone in another box; the question with other spacing.
- `test_a_failed_zoom_with_a_box_is_settled_by_a_redo_that_overlaps_it`: a failure by the tool's status redone in an overlapping box; a `view` + `image_box` zoom whose failure is redone by a `bbox` or by a `view` whose result gives the box; a redo far from it leaves it.
- `test_a_failed_zoom_stays_when_the_redo_is_not_clearly_it`: another question, no question, another page, the whole-page look, a redo that failed too, an earlier success; and the same box still settles it.
- `test_a_failed_zoom_stays_when_the_same_ask_covers_other_parts` (review finding 2): two zooms with one general question sent in the same model step (either order), one after the other with boxes apart, and boxes that only touch, all leave the failure listed; overlapping and after, it is settled.
- `test_one_redo_settles_only_the_failures_it_overlaps` (review finding 2): two failed zooms with the same question and one later success settle only the one it overlaps; neither overlapped leaves both; one redo each leaves none.
- `test_a_redo_settles_only_the_failure_not_its_own_flags`.
- `test_a_clean_reader_s_result_is_exactly_as_before`: a look, a zoom, a whole `read_pdf_text`, two whole `read_document` reads and an offset `read_text_file`. The whole `task` result is checked word for word: the answer, then DR-46's kept-file note, and no open-items block.
- `test_the_geotech_page_s_helpers_are_still_unchanged`: no reader named, so even a helper with a cut text read and a failed zoom comes back as it did, and nothing is kept.
- `test_a_reader_s_result_names_the_cut_and_the_kept_file_lists_it`.

`packages/aec-document-review/tests/test_dr50_reader_open_items_offline.py` (new) runs the page's real build for one turn, on a scripted model and a fake vision engine, as DR-46's boundary test does. Reader T reads six made-up pages with one `read_document` call; it stops at its size limit with a cursor. T then zooms with a box given two ways (the zoom fails) and zooms again in another box with the same question, then answers "Open items: none.". Reader L looks at one page. The test asserts that the main agent receives:
- T's cut, with the tool's own cursor and the right page;
- no failure for the redone zoom;
- for L, its answer and its kept-file note only.

On the base branch (`claude/dr-42-page-texts`, `keep_work.py` swapped back), the first build's core tests failed 8 of 12 (all but the clean reader, the geotech helpers, the offset read and the "stays" test, which pass on both by design), and the Document Review turn listed no cut and listed the redone zoom as failed. The tests added after the review hold the reviewer's sequences, which the first build settled; on this branch they come back listed. The Document Review turn's zoom gives a box and a view together, an argument failure, so its redo still settles it.

Every one passes on this branch.

Also run, all green: `test_dr46_open_items.py`, `test_core_imports.py`, `test_saved_files_and_settings.py`, `funhouse_agent/deep/tests/test_keep_work_offline.py`, `test_records_as_files_offline.py`, `webapp/tests/test_kept_turn_work.py`, `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`, and Document Review's `test_dr46_reader_boundary_offline.py`, `test_kept_turn_work.py`, `test_review_fingerprints.py`, `test_dr42_notes_name_page_tools.py`.

## Fingerprints

Neither moves: no prompt, tool description, schema, helper text or middleware changed, only what a `task` result carries.

## Noted, not fixed

- **The minimal build's helper (part 3) needs more than switching DR-46's mechanism on.** Its `task` (`minimal_agent._helper_tool`) is a tool of its own that runs a helper agent; nothing records that helper. Putting `KeepWork(readers={"page_looker"})` on the main agent and `HelperRecorder` in the helper would also change three other things:
  - it keeps a file and names it in a note that says to open it with `read_text_file`, a tool the arm does not have;
  - it caps the arm's helpers at once (`GEOTECH_MAX_HELPERS`, 2 by default), a cap the arm has never had;
  - it marks the helper's calls as a helper's. That moves its budget-reached note from the turn to the helper (`note_budget_reached`; today a minimal helper that uses up its budget marks the turn's budget as reached), and it adds task fields to the activity log.

  Open items alone would take a recording path of the helper's own (a `HelperRecord` set around its `invoke`, the recorder in its middleware, `open_items_note` added to its answer) and a decision on that budget note. Part 3, below, reads the record from the helper's messages after it returns instead, so none of the three changes happens.
- **Two zooms with bad box arguments sent together, then one good redo, settle both.** The rule for a failure in the box's arguments has no box to match a redo to. Such a pair should be rare (a helper that sends two malformed zooms at once usually fixes both); it changes no saved report, so it is noted, not built.
- **A `read_pdf_text` result cut by the result cap is not parsed.** If a result runs past the page's result cap, the tool kit cuts the JSON as a string, and `open_items` cannot read it. The text budget (12,000 characters) sits under the cap (16,000), so this should not happen.

## For the lead: what a measurement on the hosted platform should compare

In a whole-set review's saved runs, count the reader results whose kept record holds a `read_document` result with `next` (or a `read_pdf_text` result with `truncated`) that the helper did not read on from. Before this branch the main agent was never told about these cuts; after it, each one is a line in the open items. Then check that the main agent then reads on, or hands the rest to another helper, instead of reporting "not found" on pages it never read. The failed-and-redone zoom was 1 in 1,802 helper results in DR-46's runs, so no measurable change is expected from it.

## Part 3: the minimal build's helper

### What was asked

The minimal build's helper (`GEOTECH_REVIEW_AGENT=minimal`, a suite arm; `minimal_agent.py`) kept the old compact-findings contract and got none of the open items DR-46 gives a reading helper. The lead's ruling: the arm measures; testers do not run it, so no switch. Its helper's hand-back gets the same open items as the default build's reader, by the same code, not a second copy of the rules. The default build and its fingerprints must not move. The arm's own fingerprint entry may move only if a text its model is handed changes.

### What was built

- `packages/aec-agent-core/aec_agent_core/deep/keep_work.py`, `record_of_messages(messages, subagent, brief)` (new): a `HelperRecord` made after the fact from a finished helper's own messages. Each AI message with its tool calls becomes a model step, each tool message a result (name, call id, text, status), in order; the brief and any other message are skipped. The helper's state holds each result as the helper received it (inline images are added to a request, never to the state), so `open_items` reads this record as it reads one `HelperRecorder` kept.
- `packages/aec-document-review/aec_document_review/minimal_agent.py`, `_with_open_items` (new), called by `_helper_tool`'s `task`. After the helper returns, its record is made from `result["messages"]`, `keep_work.open_items` lists what its tools flagged, and `keep_work.open_items_note(items, None)` goes after the answer. It is joined as `KeepWork` joins a note: the answer's end trimmed, a blank line, the note. With no open item, or if reading the record fails, the answer goes back exactly as before.

No rule is copied. What counts as open, what settles it, the order, the 300-word cap and the words all come from the functions the default build's reader uses.

### Calls made

- **The record is read from the helper's messages after it returns, not recorded as it runs.** Recording as it runs means setting the running helper (`keep_work._HELPER`) around the helper's `invoke`. More than the recorder reads that: the budget-reached note (`note_budget_reached` would mark the helper instead of the turn), the activity log's task fields, the scratch guard, the plot box and the figure helper. Reading the messages afterwards changes none of them. The helper has no summarizer, so its final state keeps its whole conversation. Both ways give the same results (in the order called, plus the budget's closing message), so `open_items` gives the same lines. The live recorder keeps results in the order they finished, and a helper that used up its budget has one more model step there (the closing message, added without a model call); neither changes which items are listed.
- **No kept file, and the note names none.** The arm has no `read_text_file`, so a note telling it to open a file would point at a tool it does not have. With no file name, `open_items_note` past its cap says only how many items are not shown.
- **No helper cap.** `KeepWork` is not put on the arm, so `GEOTECH_MAX_HELPERS` does not reach it.
- **The helper's prompt (`READER_PROMPT`, "ONLY a compact list") is unchanged.** The item's done line asks for the open items. The arm's short, generic prompt is part of what it measures, and changing it would move the `review_minimal` fingerprint entry. If the lead wants DR-46's three-list contract in the arm too, that is a prompt change of its own.
- **A helper that raised gets nothing added.** Its exception carries no state to read, so the arm's "The helper failed: ..." line goes back as before. The default build's reader that stops on an error gets no open items either; it gets its kept file.

### What the arm's helper can flag

Its tools are `analyze_pdf_page` and `render_region` (both shown to the model itself, no vision call), `analyze_image` (an upload: a vision call) and `page_count`. So in a real run its open items are a call that failed (a look, a zoom not redone, a page count) and, from `analyze_image`, a vision answer cut off at its output limit or a whole-image view or tiles not read. It has no text tool, so a cut text read cannot arise in the arm. The test adds the core's `read_pdf_text` to its helper only, to show the same code carries one.

Its zooms carry no question (its prompt asks for `view` + `image_box` only), so the redo rule, which needs the same question, never settles one of its failed zooms: only a later zoom of the same `view` + `image_box` does. A failed zoom redone in a merely overlapping place stays listed, so the note may name more than it must, which is the safe side; the rule is left as it is.

### A repair from review: a zoom's place in its key (all builds, no switch)

`open_items` keys each flag by tool, file, page and box, and the box was taken from the call's `bbox` only. A zoom asked for by `view` + `image_box` (the way the reading prompts ask) had no box in its key, so every such zoom on a page shared one key, and a later good zoom anywhere on that page dropped an earlier failed one: a region never read went unnamed. This has been so since 1.1.0, in the default build too. Now a zoom's key carries its place as it was given, its `bbox` or else its `view` + `image_box` (`keep_work._zoom_place`), so only a later zoom of the same place settles a flag by its key. A good zoom of another place settles a failure only by the redo rule above (same question, overlapping box, called after the failure came back), which is unchanged. A look's own flags (lettering past the edges, an answer cut off) are kept per place the same way. A `bbox` zoom's key is what it was; a whole-page zoom (no box) still has none.

### Settings

None (the lead's ruling). With `GEOTECH_REVIEW_AGENT` unset nothing changes: the default build never imports `minimal_agent`, and only `minimal_agent` calls `record_of_messages`.

### Tests

- `packages/aec-document-review/tests/test_dr50_minimal_helper_open_items_offline.py` (new). The page's minimal build (`review_flags.ARMS["minimal"]`) runs one turn on a scripted model and a fake vision engine; the main agent hands two jobs to `task`.
  - Helper T reads six made-up pages with one `read_pdf_text` call that stops at its size limit. It zooms on page 1 (the render fails), then zooms again with the same question in an overlapping box. It zooms on page 2 (the render fails) and never redoes it. It looks at an uploaded made-up picture whose vision answer is cut off, and answers that nothing is open.
  - Helper L looks at one page.
  - The test checks what T's tools returned, then that the main agent receives T's answer followed by exactly `open_items_note([...], None)` with three lines: the page-2 zoom failure, the cut-off picture read, and the cut text read with the tool's own note. There is no line for page 1 (the zoom redone in an overlapping box) and no kept file. L's result is its answer, byte for byte.
  - A failed render is made by a `vision_view.render_view` that raises `ValueError` for the two failing boxes, the path a real failed render takes (`_render_error`).
  - Without the change to `minimal_agent.py`, the tool results are the same and the last check fails: T's result is its answer alone.
  - A second turn, in the arm's own shape: helper V zooms by `view` + `image_box` with no question on pages 3 to 6 (the first render at each place fails). The failure on page 4, never redone, is listed; the one on page 5, redone at the same `view` + `image_box`, is not; the one on page 6, followed by a good zoom far away, stays; the one on page 3, followed by a good zoom in an overlapping place, stays (no question, no redo rule). Before the key repair the page-6 and page-3 lines were missing.
- `packages/aec-agent-core/tests/test_dr50_open_items.py`, two cases added for the key repair, in the default build's shape (a zoom with a question). `test_a_close_up_by_view_is_settled_only_at_its_place_or_by_a_redo`: a `view` + `image_box` zoom fails, then a good zoom far away on the page (same question or another, or sent in the same step) leaves it listed (this failed before the repair); the same `view` + `image_box` read settles it; the same question in an overlapping place settles it by the redo rule (given as `view` + `image_box` or as `bbox`), while no question there leaves it; a look's edge note stays when another place is read. `test_a_close_up_by_bbox_is_keyed_as_before`: the `bbox` cases as they were.
- `packages/aec-agent-core/tests/test_dr50_record_of_messages.py` (new). The record holds the steps in order with their ids, texts and statuses, and sets no running helper. Its open items are the default reader's: a zoom redone in an overlapping box after its failure came back is settled, and a failed look is listed. No messages make an empty record. It fails on the base, where the function does not exist.

Also run, all green: `test_dr50_open_items.py`, `test_dr46_open_items.py`, `test_core_imports.py`, `funhouse_agent/deep/tests/test_keep_work_offline.py`; Document Review's `test_review_fingerprints.py`, `test_review_minimal_offline.py`, `test_dr50_reader_open_items_offline.py`, `test_dr46_reader_boundary_offline.py`, `test_ta38_turn_ceilings.py`, `test_review_harness_offline.py`; `webapp/tests/test_public_text.py`.

### Fingerprints

Nothing the model is handed changed: no prompt, tool description, schema, helper or middleware. Only what a minimal helper's `task` result carries changed. `test_review_fingerprints.py` passes for the default and the minimal build with no re-record.

### Noted, not fixed

- A minimal helper that used up its model-call budget is not kept as a file the way the default build's reader that gave up is. The arm cannot read a kept file, and nothing in a saved report changes.

### For the lead: what a measurement should compare

In the minimal arm's saved runs, count the helper results that carry the open-items block. Check that the main agent then looks again or hands the item on, instead of reporting "not found" on what a helper could not read. Expect it rarely: the arm's looks make no vision call, so its flags are mostly failed calls and cut-off picture reads. On open items the two arms are now alike. What still differs in what a helper hands back is the helper's prompt contract.
