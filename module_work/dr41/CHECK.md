# DR-41 CHECK: what the file tools lose (branch `claude/dr-41-file-tools-say`)

Files changed: core `docx_renderer.py`, `office_text.py`; new test `packages/aec-agent-core/tests/test_dr41_office_files.py`. No setting; all repairs.

## Confirmed on master, then changed
1. **Word out** (`docx_renderer.markdown_to_docx`, `_add_inline`, `_picture`). Confirmed: `<sup>`/`<sub>` were `html_inline` tokens written as text; `~~x~~` was never parsed (strikethrough rule not enabled, so tildes stayed); picture scaled on width only. Now: `strikethrough` enabled in the parser; `<sup>`/`<sub>` track state across the inline stream and set `font.superscript`/`subscript` (also in table cells and code spans); struck text gets `font.strike`; `_picture` fits within text width AND text height (new `_Context.text_height`), never enlarges, keeps aspect.
2. **Excel in** (`office_text._xlsx`). Confirmed: read-only openpyxl sheets carry no merge info, only the corner holds the value. Now `_merged_ranges` reads `<mergeCell>` from the workbook XML; in the table every row and column of a range gets the corner value, across no more columns than the widest row's own cells; the header and the rows above it are found on the cells as stored, before that fill (review fix 1); each sheet's block says "Merged ranges in this sheet: A2:A4, ...; in the table every cell of a merged range shows the merged cell's value" (first 20, then "and N more"). A sheet with no merges is unchanged.
3. **Word in** (`_DocxReader`, `docx_to_markdown`; `office_text._docx`). Confirmed none of these were read, and a text box inside `mc:AlternateContent` was invisible while a bare text-box drawing came out as a phantom "[picture N — not extracted]". Now, after the body, sections: "Comments in the Word file" (author, date, what each is attached to, its words), "Footnotes ...", "Endnotes ...", "Headers and footers ..." (distinct, unlinked ones), "Text boxes ...", and "Not read in the Word file" (charts, SmartArt, embedded objects named by their program (review fix 3), unparsable comment/note parts: named, counted, marked not read). `docx_to_markdown` returns a new key `extras` (the headings added); `read_office`'s fields are unchanged in shape; its note says the extra sections are read-only and to leave them out when writing back.
- Left, as told: Excel-out formulas.

## Before / after (made-up)
- "12 m<sup>2</sup>" → Word text "12 m2" with the 2 raised (was the literal tag). "~~gone~~" → struck run.
- A 100x2000 px picture → fits the page height (was ~20x the text width tall).
- `A2:A4` merged "Level 1": rows 3, 4 read "Level 1" (were empty).
- Comment by "J. Doe" on "The footing is 2 m wide": listed with author, date, quote, text.

## Tests
New `test_dr41_office_files.py`, 9 tests, all 9 fail on master (stashed core), all pass here. Runs (exit codes read directly):
- venv b: `packages/aec-agent-core/tests/test_dr41_office_files.py` 9 passed; `calc_package/tests/test_docx_renderer.py`, `test_docx_roundtrip.py`, `funhouse_agent/tests/test_read_office_files.py`, `packages/aec-agent-core/tests/test_saved_files_and_settings.py`, `webapp/tests/test_public_text.py`, `test_review_freeze.py`: 64 passed, 0 failed (new file not included in that count).
- venv a: DR `tests/test_page_turn.py`, `test_boundary.py`, `test_review_fingerprints.py`: pass; `test_review_binding_offline.py`: 6 passed, 1 failed (`test_the_process_host_is_the_apps`: `own_mark_words` is ('AEC agent', '(AI draft)'), test wants "Document Review"). **Fails identically on the commit before mine**: not DR-41's; likely the installed planlens/core mismatch; for the lead.
- venv b: `webapp/tests/test_page_fingerprints.py` 13 passed.
- Existing tests edited: none.

## Fingerprints
Not moved (no tool name, description or schema touched). Not re-recorded.

## Calls made
- Struck text and sup/sub only on text and code runs; links, images not marked.
- Word-in extras are separate sections after the body, not in-place markers, so a read-edit-write does not corrupt the body (the note tells the model to drop them).
- Reader still does not write `~~` for struck runs or `<sup>` for raised runs when reading Word (not in the item; a Word file's strike/superscript still reads as plain text).
- A comment's "text it is on" is capped at 120 characters.
- Merged ranges filled no wider than the widest row's own cells (review fix 1; was at most 1000 columns).

## Release note
A Word file written by `write_docx` now shows superscripts, subscripts and struck-through text as such (in GSE too; the geotech calc packages write HTML, LaTeX and PDF, not Word, so their own output does not change), and a tall picture no longer runs off the page. Reading an Excel schedule with merged cells now gives every row its merged label, and reading a Word file now also returns its comments, footnotes, endnotes, headers and footers and text boxes after the body, and names charts or objects it cannot read.

## For the lead
- If `read_office`'s tool description should mention the extra sections or merged ranges, the text lives in `vision_tools.py` (not mine); the result's own note already says so.
- Live check: one real Word file with tracked comments from Word itself (hand-built parts here), and one real workbook with merged cells from Excel.
- Failing `test_the_process_host_is_the_apps` above.

## Noted, not fixed
- Word-in does not read comment replies' threading (replies are listed as comments in order); content controls inside headers read as text.
- Writing Markdown containing a literal `~~x~~` now strikes it (CommonMark+strikethrough).
- Still dropped from a Word file without a word: equations in Word's own format (`m:oMath`), Symbol-font characters (`w:sym`, e.g. a Greek phi) and tracked deletions (the text reads as if the changes were accepted, and nothing says the file holds tracked changes). The lead files the backlog item.

## Review fixes (2026-11-13)
From the local review of 957ff72; one commit on `fix/dr-41-review`.
1. **Excel in, merged title taken as the header** (`office_text._xlsx`): the fill now runs after the cells are read, the header and the rows above it are chosen on the cells as stored, and a range is filled across no more columns than the widest row's own cells (`MERGE_MAX_COLUMNS` removed). Made-up: "Made-up boring log" merged A1:XFD1 over a header in row 2 was the table's header, 1000 columns wide; now it is "Row 1, above the table", row 2 is the header, 4 columns.
2. **Word in, pictures and links in headers, footers, comments and notes** (`_DocxReader._image`, `_link_url`, new `_rels_part`; `read_extras` sets the part being read): their r:ids are looked up in their own part, not the body's. Made-up: a header logo was saved as an XML part of the file and listed in `pictures`; a footnote link lost its URL. Now the logo itself is saved, and the link keeps its URL.
3. **Word in, embedded objects** (`segments`: `w:object` joins `w:drawing` and `w:pict`; `_picture_or_box` names its program): an Equation 3.0, MathType or embedded workbook object was neither shown nor counted; now "1 x embedded object (Equation.3): present in the file, not read".
4. **Word in, what a comment is attached to** (`read_container` and `table` note comment ranges between paragraphs, tables, rows and cells, through new `_range_mark`/`_with_ranges`; notes, headers and footers are read before the comments): "(on no text)" is gone. A comment shows the text it is on; or "(attached to a place with no text, such as a picture)" when its range holds no text; or "(the text it is attached to was not found)".
- Tests: four added to `test_dr41_office_files.py` (13 there), one per fix; each fails on 957ff72 (reasoned from the code).
- Runs (local, the app venv, code resolved inside the checkout): the 13 above with the `calc_package` docx tests, `test_read_office_files`, core `test_saved_files_and_settings`, `test_core_suite_and_records`, `test_core_imports`, `test_app_details` (core and webapp), `test_core_move_11c`, deep `test_review_binding_offline` and `test_deep_tools_offline`, `test_page_fingerprints`, `test_wheels`, `test_public_text`, `test_review_freeze`: 205 passed, 5 skipped (wheel build opt-in), exit 0. DR `test_review_fingerprints`, `test_page_turn`, `test_review_binding_offline`: 16 passed, 1 failed (`test_the_process_host_is_the_apps`, as above, not DR-41's). Fingerprints not moved.
