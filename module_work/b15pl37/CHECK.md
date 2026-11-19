# CHECK B15 / PL-37: the sheet's printed scale in the sheet index's rows

Branch `claude/b15-pl37-sheet-scale`, from `main` 7eed79b (v0.17.0). Work commit b818d8e; this file alone in the last commit.

## Confirmed on main before changing it
- `planlens/document/sheets.py` `SheetRow` held page, number, title, revision, revision_note, count, how, confidence, bbox, reason; no scale. `_NOT_A_TITLE` already kept a "SCALE..." line out of the title. Nothing already fixed, nothing found wrong.
- The index is made only by `Document.sheet_index()`, called from the sheet tools, `open_document` with the sheet tools on, and the `sheets`/`sheet` arguments (`planlens/tools/toolkit.py`). A host that did not ask never makes it.

## What changed (97 changed source lines; no repair, no setting: it rides the existing `sheet_tools` choice, off by default)
- `sheets.py`: `SheetRow.scale`, `SheetRow.scale_how` (also in `to_dict`); `SCALE_LABEL`, `SCALE_WORDS`, `HOW_SCALE_LINE/CELL/CORNER`; `_scale_value(text)`; `_scale(sheet, block, places)`; one call in `sheet_index()`, for every sheet with a text layer.
- `tools/toolkit.py` `_tool_sheet_index`: a row with a scale gains `scale` and `scale_from`. `tools/specs.py`: the `sheet_index` description reads "its title, scale and latest revision": **+7 characters** (758 to 765). `TOOL_SPECS` and the other two sheet tools' specs compare equal to main's.
- Rule: the value of a scale label inside the title block, on the label's line or alone in the cell beside or under it. The title block is the existing `_block_box`, cut to the band (right or bottom) holding most of the set's fixed wording. No label, an unreadable value, or two labels that disagree: no scale.

## What was reused (no second parser)
- `planlens.document.scale.parse_ratio`: the reader behind `scale_from_stated` and `scalefinder._plan_scales` (the measuring tools' printed and stored scales). It decides whether a value is a ratio or an equation; the row keeps the printed text.
- From `sheets.py` itself: `_block_box`, `_beside_label` (the cell beside or under a label, turned lettering included), `_inside`, `_at`, `_clean`, the band and corner constants.
- Not reused: `planlens.pdf.scale.parse_scale_annotations`. It searches anywhere in a line and returns a factor, not the printed scale. New wording here is only the label and the words (AS NOTED, AS SHOWN, AS INDICATED, VARIES, NONE, NO SCALE, N/A, NTS, NOT TO SCALE), which no reader in planlens had.

## Before and after (made-up set, `sheet_index` row)
- Before: `{"page": 1, "sheet": "S-102", "title": "FOUNDATION PLAN"}`
- After: `{"page": 1, "sheet": "S-102", "title": "FOUNDATION PLAN", "scale": "1/4\" = 1'-0\"", "scale_from": "in the cell by the title block's scale label"}`
- S-104, whose title block prints no scale while two view titles on it print one: the same row before and after.

## Tests
New file `planlens/document/tests/test_b15_pl37_sheet_scale.py`, 11 tests. On main's source 10 fail and 1 passes (`test_the_rest_of_the_row_is_read_as_it_was`, a guard that numbers, titles and `how` did not move). Forms covered: ratio on the label's line; imperial under its label and beside it (cell apart); AS NOTED; NTS; no scale; a bare label; a label over words; two labels that disagree; a block across the bottom with a key plan's "SCALE: NTS" up the right edge; one sheet alone; the four shared fixture styles (view scales only: every row has none).
- `/tmp/pl/bin/python -m pytest -q -p no:cacheprovider <new file>`: 11 passed.
- The new file, the 8 test files that name `sheet_index`, and `review/tests/test_host_specs.py`, `test_tools.py`, `test_dr44_drawing_tools.py`, `test_pl27_wiring.py`: 246 passed, 0 failed. The brief's `git grep -l "sheet_index" -- "planlens/**/tests"` lists nothing (the pathspec matches no file); I used `git grep -l sheet_index -- planlens` filtered to `/tests/`.
- Default surface: `review/tests/test_sheet_tools.py::test_a_host_that_did_not_ask_is_offered_what_it_was` and `test_host_specs.py` (against `app_tool_specs.json`) pass; nothing re-recorded.
- Whole suite (`pytest planlens -q -x`): ran to the end on this branch's source with exit code 0, so no failure (run with `-q -q`, which prints no count); the lead's `ci/<name>` push gives the counts.
- The app as a consumer: venv (a) built from the `claude/b13-integration` clone (it pins planlens 0.17.0), this branch installed over it (`--no-deps --force-reinstall -e`); from `packages/aec-document-review`, `tests/test_review_sheet_tools_offline.py tests/test_review_fingerprints.py`: 21 passed, 0 failed.

## Fingerprints
planlens has none. The app's `test_review_fingerprints.py` passes on this branch as recorded, so nothing moved there; a recording of the sheet-tools build made elsewhere would differ by the 7 characters of the `sheet_index` description.

## Calls made
- The scale is given as printed ("AS NOTED", not a class), so nothing is restated; a caller that wants a number runs it through `parse_ratio`.
- `scale_from` is shown on every tool row that has a scale (about 60 characters a row), not only on the lesser readings as `from` is for the number: the values explain themselves, so the description needed one word.
- A sheet read alone (one or two sheets, no title block shown) takes a label only in its lower right corner (`CORNER_X`, `CORNER_Y`) and says "title block not located".
- Two scale labels with different values in the block give no scale, not the first.
- `open_document`'s short index is left as it was (page, number, title).

## Markup ids in the rows (PL-27), from reading only; not built
1. Markups already carry a stable id: `Markup.uid` (`annotations.py`, `p<page>.n<NM>` or `p<page>.h<hash>`), read per page by `Document.markups(pages)`.
2. A row would need `markups`: the uids on its page, from `Document._page_annots(page)`, which the index does not read today; it costs one annotation read per sheet (cached per page).
3. The index is cached on the `Document` (`_sheet_index`) while marks are written by `annotate_document` to another file, so the ids belong in the tool row (`_tool_sheet_index`), not in the cached `SheetRow`.
4. `document_markups` prints the positional `[p3.m2]` (`formatting.render_markup`), not the uid, so a row's uid matches nothing the model sees until that line shows the uid too: that is a change to a default tool's result, so it needs the same opt-in.
5. Size: a marked-up sheet can hold dozens; the row needs a count plus the first few uids and a pointer to `document_markups(sheets=...)`, within `fit_items`' budget.

## Release note
With the sheet tools on, each sheet in the sheet index shows the scale its title block prints and how it was read. Nothing changes otherwise.

## For the lead
- Not measured on a model, and on made-up sheets only: real title blocks are the check (EV-45's `sheet_tools` arm).
- All forms the brief names are read; none was left out.

## Noted, not fixed
- A view the set repeats at one place inside the title block's own band (a key plan with "SCALE: NTS") is read as part of the block: its scale is given where the block prints none, and where the block prints another the two disagree and the row has none.
- "1:100 @ A1" and two scales in one cell ("H 1:100 V 1:10") are not read: no scale.
- A sheet with no text layer has no scale, as it has no number.
