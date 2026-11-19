**MERGE AFTER FIXES** (the one fix is on this branch, 2cfc3c7). Reviewed: cd6f0d5 as merged at befe56c. Review branch `claude/rv-ev-18-image-files`, cut from befe56c.

`claude/b13-integration` had moved on to a6e17f5 (11efe26, the page-map label repair) when this review started. This branch is cut from befe56c as the prompt names it; merged onto a6e17f5 in a scratch worktree it merges clean and the tests pass (below).

## Findings

Every row was run on made-up records, on cd6f0d5~1 ("before", a second worktree), on befe56c ("builder") and on this branch ("here"). Report = `drain_memo.pdf`, 40 pages, pages 3-4 read as text; the answer, unless given: "The pipe schedule is on PDF page 1 and the invert levels on page 17; see also p. 3."

| # | What | Made-up input | Before | Builder | Here |
|---|---|---|---|---|---|
| 1 | **The defect is gone.** | the report, and one `render_region` on a picture, in either order: `yard.png`, `Site.PNG`, `scan.tif`, `uploads/photos/yard.jpeg`, `C:\up\photos\Yard.JPG`, `.webp` `.GIF` `.bmp` `.tiff`; two reports and a picture; a picture opened with `open_document` and zoomed by handle | `never_opened` empty, `unverified` "page 17" | `never_opened` "PDF page 1", "page 17" (as the report alone) | the same as builder |
| 2 | Holds: a PDF with an image suffix inside its name is a document. | `plan.png.pdf`, page 1 looked at; "The plan is on PDF page 1; the legend on PDF page 4." | `never_opened` "PDF page 4" | the same | the same |
| 3 | Holds: a record key that is not a file name. | a read by handle `doc_zz` with no `open_document`; a document named `drain memo` (no extension) with a picture | first: unchanged in all three. Second: `never_opened` empty | second: "PDF page 1" listed (the repair) | the same as builder |
| 4 | **New false firing, fixed here.** A page cited of the picture itself was read as a page of the report. With `GEOTECH_CITED_PAGES=note` the tester is told the citation was never opened, of the picture the model looked at. | the report, a look at `site.png`; "site.png, page 1, shows the ditch." (also "page 1 of site.png", "PDF page 1 (Site.PNG)") | nothing listed | `never_opened` "page 1"; in `note` mode the note is appended | nothing listed |
| 5 | **The same with only a picture, fixed here** (the builder's first "noted" line). | only a look at `scan.tif`; "The stamp is on PDF page 1." / "page 1 of the scan." / "p. 1 (scan.tif)" | nothing listed | "PDF page 1" in `never_opened` (the note fires); the other two in `unverified` (no note) | nothing listed; "PDF page 2" of it still `never_opened` |
| 6 | Noted, not fixed: a picture looked at by a handle the record never named. | the report, then `render_region` with handle `doc_q` and no `open_document` line | `unverified` "page 17" | the same: the old defect in full | the same |
| 7 | Noted, not fixed: a PNG that a tool rendered from a PDF page grants no page of that PDF. | the report, a look at `drain_memo_p12.png` | "PDF page 1" counted as opened (by accident: any picture did) | `never_opened` "PDF page 1", "page 17" | the same |
| 8 | Noted: "see photo site.png" and "Figure 3 (uploaded image)" are no page citations. | those words beside "the schedule is on page 3" | nothing listed | the same | the same |
| 9 | Noted: on real files only half of the defect showed. | a real made-up 20-page PDF and a PNG (PyMuPDF), planlens' own labels | "PDF page 1" opened; "page 17" still `never_opened` | both listed | both listed |

Row 4 and 5, how common: a model asked to give file and page for each finding writes "site.png, page 1" readily (the look tools name the picture's page as page 0 / PDF page 1); "PDF page 1" of a lone picture is rarer. The fix (`_of_picture`, 13 lines of code in `cited_pages.py`, none near `_described`): a citation passes when its clause names an image file that was looked at, or when image files are all the conversation has, and that page of the picture was looked at. The picture still opens no page of a report: "The photo site.png shows the ditch. The pipe schedule is on PDF page 1." lists "PDF page 1". Not caught: "page 1 of the scan" beside a report where the file name is not written; it is read as the report's page 1, as any bare "page 1" is.

Row 6, can it happen on the Document Review page: not in a live turn. An upload is looked at with `analyze_image` (not a recorded page) or zoomed with `render_region` by its attachment key, so the key is the file name; a handle comes only from `open_document`, whose line is in the same `activity.jsonl`, and while the process lives `document_tools.handle_source` names it as well. It needs a saved record read offline whose `open_document` line is missing. A bare handle cannot be told from a PDF's, so there is no small fix.

Row 9: planlens opens a PNG as a one-page document with no printed label, so on a host that holds the file the labels did not go unknown; they do where the file cannot be opened (a saved run read elsewhere). The "PDF page 1" half is the same on both.

## Every other reader of the record (question 2)

- `TurnPageCheck.feedback` and `finish` (`note`, `retry`): both go through `check_answer`; run in `note` mode on rows 1 and 4 (test `test_note_mode_says_nothing_of_a_picture_that_was_looked_at`).
- The `cited_pages` event: written from the same result; its lists change as rows 1, 3, 7 say, in every run's record, setting or not.
- The photograph-page check (`photo_coverage`, `attached_documents`): reads `pages_opened` unfiltered; an uploaded image is still in the attachment list and under `left_out` (`test_dr48_photo_pages.py`, `test_dr48_photo_pages_turn.py`, `test_vis33_photo_pages_one_look.py` pass unchanged).
- `without_images` leaves `runs`, `references`, `reference_names` and `photos_told` as they are: none is keyed by an upload's page looks; nothing lost.
- `cited_documents.py`: imports only `record_dir`; goes by what a result says. The builder's test of it passes before and after.
- The suite's automatic checks and `run_report`: neither reads the lists (`run_report` only names the event); grep of `review_eval/` for `never_opened` and `cited_pages` is empty.
- `record_entries` (`write_diggs`, the geotech page only): still reads the unfiltered record, so a looked-at picture's page 0 still counts for a record citing page 0. Read, not run; the geotech page is quiet here. Noted, not fixed.

## An edit to the builder's tests

`test_only_a_picture_pdf_page_1_is_no_page_of_it` and `test_only_a_picture_page_1_is_as_with_no_document` stated row 5's listing as intended. They are renamed (`..._is_the_picture_looked_at`) and state what the check returns here. The lead may prefer the builder's reading of a lone picture: then drop `lone` from `_of_picture` and restore the two tests; row 4 does not depend on it.

## What I ran

- venv (a), `packages/aec-agent-core`: `tests/test_ev18_page_check_repairs.py test_rv_ev18_review_fixes.py test_ev18_image_files.py test_rv_ev18_image_files.py test_dr48_photo_pages.py test_vis33_photo_pages_one_look.py`: 166 passed, exit 0.
- The new file without the fix (source stashed): 7 failed, 23 passed, exit 1. The 23 state the repair and pass on befe56c by design.
- venv (a), `packages/aec-document-review`: `tests/test_cited_documents_turn.py test_cited_pages_turn.py test_review_fingerprints.py test_boundary.py test_dr48_photo_pages_turn.py`: 33 passed, exit 0. Fingerprints not moved.
- venv (b), root: `funhouse_agent/tests/test_cited_pages.py test_cited_documents.py webapp/tests/test_cited_pages_turn.py test_cited_documents_turn.py test_page_fingerprints.py`: 179 passed, exit 0. Fingerprints not moved.
- venv (a), root, `--noconftest`: `webapp/tests/test_public_text.py test_review_freeze.py`: 13 passed, exit 0.
- This branch merged onto a6e17f5 (scratch worktree, not pushed): the core's tests matching `ev18 or dr48 or vis33`: 172 passed, exit 0.
- Scripts in `/tmp` (not committed): 26 made-up conversations on the three trees (18 rows differ before to builder, 12 before to here; the 12 are rows 1, 3, 7); the real-file run of row 9.

## With every setting unset (question 5)

Nothing a tester sees changes: no prompt, tool description, setting or fingerprint moved, and the note is made only with `GEOTECH_CITED_PAGES` on `note` or `retry`. The recorded `cited_pages` event changes as above.

## Not checked

No model run, no hosted run, no turn through the Document Review page itself (the turn object was run directly). The full suites on GitHub. `record_entries` with a picture (read only). The `retry` mode was read (it shares `check_answer`), not run on its own.
