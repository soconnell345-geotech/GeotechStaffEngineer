# Review of EV-18, the cited-pages check repairs

## 1. Verdict

**MERGE AFTER FIXES**, reviewed head `d1396b1` (base `5ce9ce2`). The fixes are on this branch, `claude/rv-ev-18`, head `fdb70b6`: merge this branch, not `d1396b1`. Re-run the offline pass over the saved runs first (section 7): fixes 1 and 3 put firings back, and how many of the saved runs' false firings come back with them is not known here.

## 2. Findings

Recall on 70 made-up answers of my own, each citing a page no tool opened (pages 1-2 read of a 60-page report with a page map and an outline), under four made-up numberings (plain, offset by two, none known, by chapter):

| | fired | lost | lost to |
|---|---|---|---|
| base `5ce9ce2` | 66 | 4 | printed labels no page prints (`unverified`, as before the branch) |
| reviewed `d1396b1` | 45 (44 by chapter) | 25 (26) | A: 20 or 21; C: 1 where numbering is none or by chapter; the same 4 labels |
| this branch `fdb70b6` | 63 to 65 | 5 to 7 | the same 4 labels; A: 1 or 2; C: 1 where numbering is none or by chapter |

| Sev. | Where (`cited_pages.py`) | What is wrong, and the input that shows it | Fixed by |
|---|---|---|---|
| must-fix | `_described`, rule (c) | A statement passed when half its words were the page's title words, so a finding stopped firing: "The slope stability results (PDF page 20) fail.", "PDF page 15 has no drainage details.", "Foundation recommendations (PDF page 25): bearing capacity of one hundred fifty kilopascals." Fix 1 brought back 19 of my 70. | `c200586`: every word must be the page map's; "no" and "not" count; what the sentence says of another page being unread is left out |
| must-fix | `fire`, the statement | The statement ends at the line break, so a heading that cites a page and repeats its title was "what the page map said" though its findings stand under it: "### PDF page 25: Foundation recommendations" over a list. | `38d0326`: a heading line, a bold line or a line ending in a colon, with text under it, is not read that way |
| must-fix | `_unread_cue` | The cue was taken for a page it is not said of: "I did not open page 9, and page 34 shows a ratio of 0.6" (34 excused); "Pages 34 and 47 (47 not opened) give ..." (34 excused); "The contractor did not read the notes: page 34 gives a ratio of 0.6"; "Page 34 (ratio 0.6) was not checked by the contractor"; "I can also check the wall: page 15 shows a 300 mm stem". | `2142083`: a cue before the page counts only when no value follows the page (with another page between, only in a bare list); a cue after it only with no value between |
| must-fix | `_paired`, `fire`, `_unread_cue` | Every citation was compared with every other, and `finish` runs the check at every turn's end with the switch off. 15,000 distinct citations in 677,000 characters: 370 s, against 8 s on the base. (3,000 citations in 133,000 characters: 4.8 s against 0.5 s.) | `05c66b8`: compared only within a line, a lead or a statement; 9 s and 0.7 s |
| should-fix | `_paired` | Any PDF page within 40 characters was "the same place", and the page number left the check with no record row: "The ratio of 0.6 is on page 34; PDF page 1 is the cover." fired nothing. | `8b6725a`: not a pair where the page map knows that PDF page prints another number |
| should-fix | `numbering` | One page printing "A-1" made the document "by chapter": "p. 12-16", and "12-16" in a column headed "Page", went to `unverified` though pages print 12 and 16. | `b40e990`: the run of pages, where no page prints "12-16" and pages print both ends |
| should-fix | `refs_for` | A reference's name matched as a bare substring: figure "5-2" was named by "Table 15-2 is on PDF page 88 of the report", and the report's unopened page 88 counted as opened. | `fdb70b6`: not matched inside a longer number |

Left for a decision (not fixed):

- **C, numbering not known.** The half rule is computed from the whole document's page map (`planlens_labels` reads every row), not from what the turn read. A document with no printed numbers does not silence every citation: only a hyphenated number under a one-page lead ("p. 12-16", "page 12-16", a "Page" column cell "12-16") goes to `unverified`; "pages 12-16", "p. 12" and every PDF page still fire. That is 1 of my 70 answers in the two such documents of four. Whether "page 12-16" in a scanned report should fire instead is the builder's call C; it lost one true firing here.
- **A, a where-statement in a list of sources.** "Groundwater: p. 26, PDF page 28." is filed under `page_map` where the page is titled "Groundwater observations". It reads as where something is; in a traceability list it is a source.
- **A, an identifier split by a line break** ("boring\nMB-\n1"): the statement stops at the break, so the rest is not read. Fix 2 covers headings only.
- **A, numbers the check adds itself.** The page count ("of 60 pages") and printed labels count as given by the page map: "PDF page 25: bearing capacity 60." in a 60-page document is filed under `page_map`. Number words of three letters ("ten") are not counted as words.
- **B, a cue about something else with no value beside it** still excuses the page: "Page 34 says the test was not read at 28 days.", "The signature on page 47 was not rendered legibly."
- **D, which document.** A markup listing of page 12 of one document makes "PDF page 12" of another count as opened. The check gives no citation a document (as on the base, where any document's page counted); a listing of one page marks only that page.

## 3. Paths checked

- Switch unset: `finish` returns the answer object itself, `feedback` returns `None`; holds (builder's test, and read).
- `note` and `retry`: only `never_opened` makes either note; holds.
- `record_entries` (`write_diggs`) and `photo_coverage`: read `PagesOpened.status`, `documents`, `looked`, `photos_told`; none reads the new fields. `document_markups` stays a hit for them. Holds; `test_dr48_photo_pages*.py` pass unchanged.
- The record: `never_opened`, `partly_opened`, `hit_only`, `unverified`, `other_documents` keep their shape. `page_map` rows may now carry `matched`; `named_unread` is new. No code in the repository reads these keys but the tests and `activity_log.py`'s comment, which does not name `named_unread`.
- D: one listed markup page marks that page only (page 13 beside it still fires). G: a rendered page counts only that page; a failed render does not count; with a document open the line must name the reference (fix 7).
- H: "pp. 4-1 to 4-3" is one citation of three labels; holds.
- Nothing fitted to one run: the word lists (`_UNREAD`, `_KIND_WORDS`, `_DUPLICATE_WORDS`, `_PLAIN_WORDS`) are general English; no rule names a phrase of one answer.
- Regexes: no backtracking found. 200,000 spaces after a lead, 100,000 list items, 100,000 repeats of "not" or "were", 200,000 bars: each under 1 s.
- Shared host: the check runs in the turn's own thread; the activity cache is keyed by the log's path with a lock per log; the new fields are copied under that lock. No new lock, thread or module table.
- Merge with `origin/claude/b13-integration` (`fb27052`): automatic, no conflict, for `d1396b1` and for this branch. That branch has no commit on `cited_pages.py` since the base.

## 4. Tests

Venv (b) of COMMON_DOCREVIEW (Python 3.12, planlens 0.16.0 and geotech-references 1.8.0 from PyPI), Document Review added with `--no-deps`.

- Reviewed head: the builder's file and the five existing files CHECK.md names: 121 passed.
- Base: the builder's 38 tests: 23 failed, 15 passed. The 15 are of the "still fires" kind by CHECK.md's account; I did not go through them one by one.
- This branch, from the root: `test_ev18_page_check_repairs.py`, `test_rv_ev18_review_fixes.py`, `funhouse_agent/tests/test_cited_pages.py`, `test_dr48_photo_pages.py`, `test_coverage_pages_offline.py`, `webapp/tests/test_cited_pages_turn.py`, `test_cited_documents_turn.py`, `test_page_fingerprints.py`: 172 passed.
- This branch, from `packages/aec-document-review`: `test_cited_pages_turn.py`, `test_dr48_photo_pages_turn.py`, `test_cited_documents_turn.py`, `test_review_fingerprints.py`, `test_boundary.py`: 33 passed.
- Both fingerprint files are unchanged since the base.
- The trial merge: the builder's and the existing tests 109 passed; with the fixes 121 passed.
- New here: `packages/aec-agent-core/tests/test_rv_ev18_review_fixes.py`, 38 tests. Failing without their fix: 10 (fix 1), 3 (fix 2), 7 (fix 3), 4 (fix 4), 1 (fix 5), 1 (fix 6: 24 s against 1.8 s, bound 10 s), 1 (fix 7). The other 11 guard what the builder's repairs still let through and pass either way. No builder test was edited; all pass after every fix.
- Guards before each commit: `test_public_text.py`, `test_review_freeze.py`: 13 passed.
- My probe scripts stayed in `/tmp` and are not committed.

## 5. Scope

The builder's branch changes `cited_pages.py`, one new test file and `CHECK.md`: in scope. `REFERENCE_TOOLS` names three of the geotech page's tools inside the core; `test_boundary.py` passes, and `HIT_TOOLS` already named tools the same way. My commits change `cited_pages.py` and add one test file.

## 6. CHECK.md

- "Nothing is dropped silently": untrue for a page number paired with a PDF page, which gets no row in any list.
- "the identifier rule (2) carries the recall": not supported. Of the 66 answers that fired on the base, 45 still fired; findings said without a digit were lost (fix 1).
- "Any number or identifier the page map did not give keeps it firing": untrue where the number equals the page count or a printed label.
- B, "What it lets through" names one case; the review found five more forms (fix 3).
- The base's own cost is not mentioned: `find_citations` is already quadratic in the number of leads (8 s for 15,000 citations on the base).

## 7. Could not check

- **The saved runs.** They are private and not in this checkout. Run the builder's offline pass again on this branch and compare `never_opened` per turn with the base and with `d1396b1`: fix 1 (every word) and fix 3 (a value beside a cue) are strict, and each `page_map` or `named_unread` row lost to them is either a true firing regained or a false one back. If too many false ones return, the dial is `_described`'s rule (c).
- **A real page map.** My documents' printed numbers are made up; `planlens_labels` on a real file (labels read from header and footer bands) was not exercised, and with a document that cannot be opened on the host every bare page citation is `unverified`, as on the base.
- **The full suite** runs on GitHub when the lead pushes the branch as `ci/<name>`.
