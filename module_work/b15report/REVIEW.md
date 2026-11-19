**MERGE AFTER FIXES** (the one fix is on this branch, `claude/rv-b15-report-bad-records`); reviewed head 167c7e2, base 9367934.

# Review of brief 15: the run report on malformed records (EV-17)

All runs below are my own made-up records, written line by line (not the builder's helper). Before = 9367934's `run_report.py`, after = the branch's, each loaded by file path on the same run folder with the clock held still.

## Findings

| # | What | The made-up record | Before (9367934) | After (167c7e2) | Fixed or noted |
|---|---|---|---|---|---|
| 1 | **A well-formed look was left out** and counted as "beyond its last page". Two documents of one file name are one document in the report, with the count of the one opened last. | `open made_up_a/made_up_set.pdf` (6 pages), a zoom on its page 4, `open made_up_b/made_up_set.pdf` (3 pages), a zoom on page 1 | Both zooms drawn | The page-4 zoom gone, "past_end: 1" | **Fixed here** (7 source lines: the rule judges by the largest count any `open_document` gave for the name). Its report is again byte for byte the base's. 2 tests, both fail without the fix. |
| 2 | A page given as 2.5 is placed on page 2, with no word (`coverage.pages_from_call`: `int()` of the argument). Same before and after. | zoom, `"page": 2.5` in arguments and result | Drawn on page 2 | The same | Noted (not this file; the builder notes its neighbour) |
| 3 | A `reads.json` row whose `region` AND whose kept result's `view` are both malformed could be counted twice (read, not run). | - | - | - | Noted |
| 4 | "Outside its page" is judged against the PDF found by NAME; a file of that name with another page size (a later revision in `--documents`) would lose real boxes. They are counted and said, never silent. | not run | - | - | Noted |

## The questions

1. **Made, and the viewer works: yes, every case.** Opened in headless Chromium (Playwright): no page error, every looked-at page walked by its address, the boxes in the page counted. Before, the viewer was dead (its data unreadable to `JSON.parse`) for nan or infinity in a zoom, a tile and a whole-page view; after, it opens and says so on the page. Cases: nan (zoom, tile, page view), infinity, a dict, null, text, three numbers, numbers as strings, a reversed box, a zero-area box, a box at negative coordinates (counted as outside its page), page 2.0 (page 2), page "2" (page 2), page "two" (counted, on no page), page 40 of 6 and page -2 (2 counted, the grid stays 6), lines `[1, 2]`, `"text"`, `3`, `null`, `true`, `{}` and a cut line (7 counted), 1,500 zooms on 40 pages (about 9,000 events: 2.7 MB, about 1 s, 1,500 boxes drawn, same bytes as before).
2. **Nothing guessed; one well-formed record lost (finding 1), fixed.** Ten well-formed zooms alternating with ten malformed of five shapes: 10 drawn, "box: 10". Views recorded with the core's own `vision_view.render_view`, `view_payload` and `tile_boxes` (planlens 0.17.0) on PDFs with /Rotate 90, 180 and 270, a CropBox inside the MediaBox, a MediaBox at origin 200,300, that with a CropBox, and a CropBox with /Rotate 90: a whole page, 13 tiles (2x2 and 3x3) and 6 zooms (the whole page, both corners, a 5 pt corner, the middle) on each, 20 of 20 drawn on all eight, no count, same bytes as before. The tools write views in the page's displayed frame from 0,0, which is PyMuPDF's `page.rect`, the frame the rule tests. On a 595.28 x 841.89 page: the view `[0, 0, 595.3, 841.9]`, a box on the edge, a corner tile and one 0.7 pt over are all drawn. Every count met equalled what was left out, by kind.
3. **Byte for byte: yes.** A pytest plugin (kept out of the repository) wrapped `Report.html` in the three existing test files and made the base's report of the same folder at the same instant: 37 the same, 1 differs only by the new lines (its run holds a line that is not JSON), 1 not compared (its test replaces `_viewer_data`). My well-formed runs (plain, the eight real-tool runs, the edge run, the large run, a failed look, a zoom with no view): same SHA-256 before and after, and after the fix.
4. **No new secret or path.** A key-shaped text where a box should be and a path where a page should be: each appears twice in the report before and twice after (in the call's own arguments and result, as before); the lines near the top and the page notes are fixed words and a count. `tests/test_record_masks_secrets.py` passes.
5. **No older shape is lost.** Every writer of a `view` in the history of `vision_view.py`, `vision_tools.py` and `activity_log.py` (and their old `funhouse_agent` paths) writes a list of numbers (`round(float(v), 1)`, `int(v)`, `list(boxes[k])`); none wrote text or a dict. A zoom whose result has no `view` still takes the asked `bbox` (same bytes); a result cut short is still read from its text (the existing tests' runs with `max_chars=400`, same bytes).
6. **Nothing else moved** (table below). A failed look (`{"error": ...}`) is not counted: `_call` returns before the new line.

## What I ran (venv (a) of COMMON_DOCREVIEW, Python 3.11, PyMuPDF 1.28.2)

| Command | Exit | Result |
|---|---|---|
| core: `pytest tests/test_rv_b15_run_report_bad_records.py tests/test_b15_run_report_bad_records.py tests/test_ev17_run_report.py tests/test_ev17b_page_viewer.py tests/test_ev20_review_report.py tests/test_record_masks_secrets.py` (with the fix) | 0 | 100 passed |
| the same without my file, on 167c7e2 | 0 | 98 passed |
| the builder's new file on 9367934 | 1 | 17 failed, 7 passed (as CHECK.md says) |
| my new file without the fix | 1 | 2 failed |
| Document Review: `pytest tests/test_review_fingerprints.py tests/test_boundary.py` (before and after the fix) | 0 | 9 passed; no fingerprint moved |
| `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | 0 | 13 passed |

CHECK.md: its table, counts and byte-for-byte figures held on my runs; nothing in it found untrue. Scope: one source file, one new test file, the check file.

## Not checked

- `webapp/tests/test_page_fingerprints.py` (venv (b) not built; `run_report.py` is not handed to a model).
- A real hosted record (EV-45), a browser other than Chromium, Python 3.12, a report made where PyMuPDF is missing.
- Findings 3 and 4 were read, not run.
