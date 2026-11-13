# DR-41 CHECK: what the file tools lose (branch `claude/dr-41-file-tools-say`)

Files changed: core `docx_renderer.py`, `office_text.py`; new test `packages/aec-agent-core/tests/test_dr41_office_files.py`. No setting; all repairs.

## Confirmed on master, then changed
1. **Word out** (`docx_renderer.markdown_to_docx`, `_add_inline`, `_picture`). Confirmed: `<sup>`/`<sub>` were `html_inline` tokens written as text; `~~x~~` was never parsed (strikethrough rule not enabled, so tildes stayed); picture scaled on width only. Now: `strikethrough` enabled in the parser; `<sup>`/`<sub>` track state across the inline stream and set `font.superscript`/`subscript` (also in table cells and code spans); struck text gets `font.strike`; `_picture` fits within text width AND text height (new `_Context.text_height`), never enlarges, keeps aspect.
2. **Excel in** (`office_text._xlsx`). Confirmed: read-only openpyxl sheets carry no merge info, only the corner holds the value. Now `_merged_ranges` reads `<mergeCell>` from the workbook XML; every row and column of a range gets the corner value (columns capped at 1000); each sheet's block says "Merged ranges in this sheet: A2:A4, ... every cell of a merged range shows the merged cell's value" (first 20, then "and N more"). A sheet with no merges is unchanged.
3. **Word in** (`_DocxReader`, `docx_to_markdown`; `office_text._docx`). Confirmed none of these were read, and a text box inside `mc:AlternateContent` was invisible while a bare text-box drawing came out as a phantom "[picture N — not extracted]". Now, after the body, sections: "Comments in the Word file" (author, date, the text each is on, its words), "Footnotes ...", "Endnotes ...", "Headers and footers ..." (distinct, unlinked ones), "Text boxes ...", and "Not read in the Word file" (charts, SmartArt, embedded objects, unparsable comment/note parts: named, counted, marked not read). `docx_to_markdown` returns a new key `extras` (the headings added); `read_office`'s fields are unchanged in shape; its note says the extra sections are read-only and to leave them out when writing back.
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
- Merged columns filled to at most 1000.

## Release note
A Word file written by `write_docx` now shows superscripts, subscripts and struck-through text as such (also in the geotech calc packages' Word output wherever they hold `<sup>`/`<sub>`/`~~`), and a tall picture no longer runs off the page. Reading an Excel schedule with merged cells now gives every row its merged label, and reading a Word file now also returns its comments, footnotes, endnotes, headers and footers and text boxes after the body, and names charts or objects it cannot read.

## For the lead
- If `read_office`'s tool description should mention the extra sections or merged ranges, the text lives in `vision_tools.py` (not mine); the result's own note already says so.
- Live check: one real Word file with tracked comments from Word itself (hand-built parts here), and one real workbook with merged cells from Excel.
- Failing `test_the_process_host_is_the_apps` above.

## Noted, not fixed
- Word-in does not read comment replies' threading (replies are listed as comments in order); content controls inside headers read as text.
- Writing Markdown containing a literal `~~x~~` now strikes it (CommonMark+strikethrough).
