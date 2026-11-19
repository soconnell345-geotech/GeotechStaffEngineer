MERGE AFTER FIXES (one fix, on this branch: 7b2a3a0). Reviewed head a6e17f5, change 11efe26; branch `claude/rv-ev-18-label`.

# Review: the viewer-number-label repair in `cited_pages.check_answer` (EV-18)

The repair is right for what it names, and it lost more right firings than its message says. One loss is repaired here; two are noted for the lead. All input below is made up: a 40-page report, chapter 1 printed "1-1" to "1-10" on PDF pages 11 to 20, chapter 2 on 21 to 30; PDF pages 13 to 15 opened. "Before" is a6e17f5 with 11efe26's source hunk reversed (so the image-file repair is in both), "after" is a6e17f5, "fix" is this branch.

## Findings

| # | What | Made-up input | Before | After | Here |
|---|---|---|---|---|---|
| 1 | The repair does what it says. A right pairing on a page whose printed number the map did not read is no longer listed. | Page 13 labelled only "13". "PDF page 13 (printed page 1-3)", "page 1-3 (PDF p. 13)", "Source: PDF page 13, printed page 1-3.", two pairings in one sentence (13 with 1-3, 15 with 1-5) | each listed `never_opened`, "pages 1-3" as PDF pages 1-3 | quiet | holds; fix keeps it |
| 2 | **A wrong pairing on such a page went silent**, though the map knows where the cited label is printed. Not the loss the commit states: the page does not print its viewer number. | "PDF page 13 (printed page 1-7)" (1-7 is PDF page 17, unopened); "page 2-4 (PDF p. 15)" | listed, PDF page 17 / 24 | silent | **fixed, 7b2a3a0**: listed again. In `apart`, a page labelled only with its viewer number is still another place when another page is known to print the cited label by a label that is not its own viewer number (12 source lines) |
| 3 | **The commoner loss, not fixed.** A report whose body prints its viewer numbers and whose appendix prints "A-1"... is "chapter" to `numbering()` (one part label anywhere is enough). Every wrong pairing on a body page now passes. | Body pages 1 to 30 labelled "1" to "30", appendix "A-1" to "A-10" on 31 to 40. "PDF page 13 (printed page 20)", "page 20 (PDF p. 15)", "PDF page 14 (printed page 33)" | listed, PDF page 20 / 33 | silent | noted; silent with the fix too. A viewer-number label cannot be told from an unread one, so this needs the page map to say which labels it read (planlens), not a rule in `apart` |
| 4 | **Two documents: the repair reaches the wrong one, not fixed.** `numbering()` answers once for the conversation: "chapter" when any document has one part label, else "plain" when any document is half plain. A plainly numbered memo beside a chapter-numbered report loses its check. | The report above plus a 40-page memo printing its viewer numbers, pages 5 and 6 opened. "In made_up_memo.pdf the value is on PDF page 5 (printed page 22)." | listed, PDF page 22 | silent | noted; needs `apart` to know which document a pairing names (more than 15 lines). The other way round (report with unknown numbers beside a plain memo) `numbering()` says "plain", the repair does not apply and the right pairing "PDF page 13 (printed page 1-3)" is listed as before the repair |
| 5 | The loss the commit states reproduces, and is finding 3 seen from the page: a page that truly prints its viewer number, in a document not plainly numbered, beside another printed number. | finding 3's input | listed | silent | noted |
| 6 | Nothing else moved. 98 answers compared; the only differences are findings 1 to 4. | Roman front matter then pages from 1 (labels with and without the viewer number), printed equal to viewer on every page, no printed numbers known, labels read on every chapter page | same | same | holds |
| 7 | Right firings kept in each numbering: a printed page nobody opened beside an opened PDF page ("The load is on PDF page 14; the depth is on printed page 2-6" / "... printed page 20"), and a wrong pairing where the page's printed number was read. | chapter (read and unread labels), roman, agree | listed | listed | holds. With no printed numbers known a wrong pairing is silent before and after (nothing to tell them apart) |
| 8 | "p. 13 of the PDF, numbered 1-3" and "PDF pages 13-14 (printed pages 1-3 to 1-4)" are quiet before and after in every numbering: the first is found as one citation, so no pairing is tested. | as written | quiet | quiet | noted |
| 9 | With the image-file repair (cd6f0d5): a zoom on `made_up_site.png` and the chapter report in one conversation give exactly the report-alone results for all 11 answers, after and with the fix. | the report's calls plus one `render_region` on the picture; the picture's labels unknown | (the repair was needed here too) | as the report alone | holds |
| 10 | One existing assertion became untrue with the fix and was changed: `test_ev18_viewer_number_label.py` paired PDF page 1 with "2-3", which its own page map prints on PDF page 23, and asserted silence. That is finding 2 written as a test. Its answer now cites "3-3", printed on no known page; the two tests using it still fail before the repair. | the builder's `PAIR_CHAPTER` | | | edited, one line and a comment |

## Cost (question 5)

`check_answer` on a made-up 500-page report (chapters of 20 pages, every seventh label unread, 100 pages opened) and an answer of 200 lines, each a pairing (400 citations), median of 7 runs, labels already loaded: before 45.3 ms, after 46.8 ms, with the fix 50.9 ms. `numbering()` is asked once per pairing and walks every label row; the fix's scan runs only for a pairing on an unread label (29 of 200 here).

## Nothing a tester sees (question 6)

No prompt, tool description, setting or fingerprint moved, before or with the fix. The `cited_pages` event in every run's record changes as in findings 1 to 4; with `GEOTECH_CITED_PAGES` unset nothing else reads it.

## What I ran (exit codes)

- Venv (a) of COMMON_DOCREVIEW, Python 3.12; each tree chosen by `PYTHONPATH` and proved by printing `cited_pages.__file__`.
- Core, `--noconftest`: `test_rv_ev18_viewer_label.py`, `test_ev18_viewer_number_label.py`, `test_ev18_image_files.py`, `test_ev18_page_check_repairs.py`, `test_rv_ev18_review_fixes.py`, `test_ev23_document_check_repairs.py`, `test_coverage_pages_offline.py`: 153 passed, exit 0.
- The new file on a6e17f5: 2 failed, 3 passed, exit 1 (the two wrong pairings). On the tree before the repair: 3 failed (the right pairings), exit 1. The builder's edited file before the repair: 2 failed, exit 1.
- Document Review, from its folder: `tests/test_review_fingerprints.py tests/test_boundary.py`: 9 passed, exit 0.
- `funhouse_agent/tests/test_cited_pages.py webapp/tests/test_cited_pages_turn.py`: 53 passed, exit 0.
- Guards: `webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed, exit 0.

## Not checked

- `webapp/tests/test_page_fingerprints.py` (venv b was not built; no prompt or tool text changed).
- A real PDF through planlens' page map: every label here came from a made-up `label_source`. Whether planlens gives an unread page its viewer number as its only label, and how often, decides how much findings 3 and 4 cost; I did not measure it.
- The check with `GEOTECH_CITED_PAGES` set in a live turn; any model run.
- Two chapter-numbered documents printing the same label: the fix then splits a right pairing on an unread label, as before the repair. Not run.
- The full suite; the lead's `ci/` push runs it.
