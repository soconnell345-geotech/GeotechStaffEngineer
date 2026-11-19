# Brief 15: the run report on records that are not well formed (EV-17)

Branch `claude/b15-report-bad-records`, from 9367934 (1.3.0). A repair; no setting. One source file: `packages/aec-agent-core/aec_agent_core/run_report.py` (131 lines added, 121 without comments and docstrings; 36 changed).

## What each malformed record did, and does now

Each was made from the existing tests' made-up run with one record changed, and the report made by both versions. "Viewer dead" was confirmed in a browser: the page opens, a page square does nothing, no message.

| The malformed record | On the head (9367934) | Now |
|---|---|---|
| A view box holding nan (a zoom, or a tile) | Viewer dead (its data holds `NaN`, which a browser's JSON does not read). Part 4 gone whole: "could not be made: ValueError: cannot convert float NaN to integer". | Look listed without a box, counted; the page's other boxes drawn; viewer works and says it on that page. |
| A whole-page view holding nan or infinity | Viewer dead; part 4 draws a rect with `width="nan"`. | As above. |
| A view box that is null, text or three numbers (a zoom) | Report made; the box asked for (`bbox`) was drawn in its place, with no word. | Look listed without a box, counted; the asked box is not put in its place. |
| A view box of four numbers written as text | Drawn as if they were numbers. | Left out, counted. |
| A box outside its page, wholly or partly (PDF found) | Listed with its coordinates as if drawn; nothing shows (it is outside the drawing). | Not drawn, not moved, counted. Without the PDF it cannot be told and is drawn as before. |
| A look with no page at all | Put on the tool's first page. | Unchanged, and no fault: the tool's own default is its first page, and a look at an image names none. |
| A look whose page is given and is not a number; a `reads.json` row with no page | Dropped without a word. | Counted, placed on no page. |
| A page number beyond the document (a look on page 41 of 6; a text read naming page 5,001) | Page grid and viewer stretched to 41 and to 5,001 squares. | Left out, counted; the document keeps its 6 pages. |
| A negative page number | A figure headed "PDF page -2" in part 4. | Left out, counted. |
| The last line cut off mid-record | Skipped; said only in part 5's table. | Also counted near the top. |
| An empty record (`{}`) | Shown as a nameless note of the turn. | Skipped and counted with the bad lines. |
| A page image that cannot be made; a PDF no longer in the folder | Report made; outlines only; the reason in part 4 and in the viewer. | Unchanged: no record is left out. |
| An empty `activity.jsonl`; an older record (no turn, task id, usage, text, view, pdf_page) | Report made. | Unchanged; an older shape is not counted. |

**Example**, near the top of the report (made-up run, one zoom's view `[50.0, NaN, 200.0, 220.0]`): "Left out, as the record holds them (nothing is repaired or guessed): View boxes that are not four finite numbers with some area (nan, null, text or too few numbers), each look listed without its box: **1**". Before: no line, and a viewer that did not open.

## What changed (all in `run_report.py`)

- `_box`: four finite numbers with some area, numbers only. `_view` (new): a box, or why not, counting a value that is there and cannot be drawn. `_looks_of` returns the reason with each look and no longer falls back to the asked `bbox` when a view came back malformed (it still does when none came back: the older shape).
- `PageMap.left`: one kind per record left out; `PageMap._call` and `_vision_records` fill it; `anchor` skips a page that was left out.
- `RunRecord`: an empty record is dropped and counted with the unreadable lines.
- `Report.__init__`, `_drop`: pages below 0 or at or beyond the document's count go; a box more than `EDGE_PT` (1 pt) past its PDF page's rect loses its box. `_left_out`: the lines near the top, above the contents, one a kind (`LEFT_OUT`). `_viewer_data`: a page's own note in the viewer (through the data's existing `rf` notes; the viewer's script is unchanged). `_figure`: part 4 says why beside the look.
- `_script_json`: `allow_nan=False`, so any non-finite value that slips past ends in the report's existing "page viewer could not be made" line, never in a dead viewer.

**Byte for byte.** The existing tests' runs, each report made by the head's `run_report.py` and by this one from the same folder at the same moment (a pytest plugin wrapping `Report.html`, kept out of the repository): 37 reports the same; 1 differs only by the new lines near the top, and its run holds a line that is not JSON (`test_an_older_record_still_reads`); 1 not compared (its test replaces `_viewer_data`). Eight more reports of four made-up runs, with and without `--documents`: same hashes.

## Tests

New file `packages/aec-agent-core/tests/test_b15_run_report_bad_records.py`, 24 tests. On the head 17 fail: the seven bad boxes, nan in a page view and a tile, a box outside its page (2), a page that is not a number, a `reads.json` row with no page, a page the document has not (2), a text read far beyond, a cut line and empty records, every kind at once. 7 pass on the head too, because they pin what must not change: a box a point past the edge is drawn, a far box without the PDF is not counted, a look that names no page, an empty log, a PDF broken or gone, an older record, a well-formed run.

| Command | Result |
|---|---|
| venv (a), from `packages/aec-agent-core`: `pytest tests/test_b15_run_report_bad_records.py tests/test_ev17_run_report.py tests/test_ev17b_page_viewer.py tests/test_ev20_review_report.py` | 58 passed |
| the new file against the head's source | 17 failed, 7 passed |
| venv (a), from `packages/aec-document-review`: `pytest tests/test_review_fingerprints.py tests/test_boundary.py` | 9 passed |
| `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | 13 passed |

Not run: `webapp/tests/test_page_fingerprints.py` (venv (b) was not built). `run_report.py` is not part of what a page's model is handed; Fingerprints: none moved (Document Review's pass unchanged). No setting; no prompt or tool description touched. No existing test was edited.

## Calls made

- A box may pass its page's edge by 1 pt (`EDGE_PT`): a view is logged in whole points or tenths, and a page can be 595.28 pt wide.
- "Outside its page" is judged only against the PDF's own page rect; with no PDF nothing is judged.
- A partly outside box is left out whole, not cut to the page (cutting it would be moving it).
- Four numbers written as text are not a box (reading them as numbers would be a guess about the writer).
- A page beyond the document is judged against the count `open_document` returned, else the PDF's; a document whose count is only "the last page reached" has no end to judge by, so a page number of 10^9 there still stretches the grid (noted, not fixed).
- The count is of records left out of the drawing, one line a kind, five kinds; page images that cannot be made are not records and stay where they were said (part 4, the viewer).

**Release note.** A report made from a run whose record holds a value that cannot be drawn is still made whole; it says near its top what it left out, one line a kind. Reports of well-formed runs are unchanged.

## For the lead

- The real hosted records (EV-45): run the report on them and read the lines near the top first; any kind met there that this table lacks is the next item.
- A writer that can write nan: `vision_view.view_payload` (`"view": [round(float(v), 1) for v in info["clip"]]`) and `vision_tools.py`'s `[int(v) ...]` / `[round(v, 1) ...]` views do not check that a clip is finite, and `activity_log.py` writes records with `json.dumps` at its default (`allow_nan` on), so a nan clip reaches the record as `NaN`. Left as it is (the brief's limit).
- `coverage.pages_from_call` puts a look whose `page` is not an int and whose arguments name none on page 0; left (not this file).

## Noted, not fixed

- In the viewer's list a look left out still reads "(no view box in the record)": that wording is in the viewer's script, which had to stay byte for byte; the page's note above it gives the true reason.
