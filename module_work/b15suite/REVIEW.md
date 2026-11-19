**MERGE AFTER FIXES, with the fix on this branch** (`claude/rv-b15-suite-unprompted`, fix commit 9b3e3ca on top of the reviewed head c41a42a). One must-fix, repaired here; nothing else must change.

# Review of brief 15: `coord-second-issue`

## Findings

| # | What | The made-up answer | What the check said | Fixed or noted |
|---|---|---|---|---|
| 1 | must-fix. Check 4 ("said not flagged") passed findings that found the change and never said it was unmarked: the term took any no, not, never or without, and such a finding nearly always holds one about something else. 9 of 10 passed wrongly. | "S-001 note 2 ... now reads 200 kPa; the earlier issue read 150 kPa. ... the footing schedule on S-501 has not changed ... ask the engineer"; a table row with "Affects our price?: No" | PASS ("together within 91 characters in one finding") | Fixed, 9b3e3ca: the task's own term `_NOT_MARKED` in `tasks.py` (the negative word with a word for the cloud, the delta, the revision block, table, row or entry, or the flag). Each wrong case is a test that failed before. |
| 2 | Check 4 failed two answers that did say it. | "It isn't clouded."; "neither clouded nor listed" | fail (missing the term) | Fixed by the same term; tests failed before. |
| 3 | Said three paragraphs away fails check 4; a human marker would pass it. | The S-001 change in paragraph 1; "The structural change above has no cloud and no revision entry" in paragraph 4 | fail ("never within 400 characters ... in one finding") | Noted. The builder's known limit; widening it rejoins a neighbour's "not clouded". Read the trace. |
| 4 | A change that is not there costs nothing. | The three changes, plus "S-501: footing F2 is now 2400 square, was 2100. Not clouded." | all six PASS | Noted. No false-alarm check on this task (CHECK.md says so); only the full read shows it. |
| 5 | The question does not ask whether a change is marked, so an answer with all three changes, each with its sheet and new value, is complete by the question's own words and scores 3 of 4. | "S-001: allowable bearing pressure is 200 kPa, was 150 kPa." | checks 1 to 3 PASS, check 4 fail | Noted, by design: read check 4 as "volunteered", never as a miss; check 3 is the finding. |
| 6 | Truth: "every other sheet is as it was" holds for the content; every sheet's title block also changes its issue line ("ISSUE 0 - 2026-01-12" to "ISSUE 1 - 2026-03-02"). | (page-by-page text diff of the two built issues) | n/a | Noted; no answer is scored on it. |
| 7 | CHECK.md is now out of date in three places: check 4's term ("`_UNFLAGGED` (the existing term)"), its two "Limit:" sentences, and "Noted, not fixed". The three assertions of the builder's test file that pinned those limits and the term's identity are turned over in 9b3e3ca. | | | Noted for the lead (CHECK.md left as the builder wrote it). |

The other answer kinds scored as a careful marker would: complete, 4 of 4; only the two flagged changes, 2 of 4; third change found and not said, 3 of 4 (after the fix); said in a table ("Shown as a revision?: Not clouded, no revision entry"), 4 of 4; old values and no new ones, 0 of 4 (right: the question asks what the sheet says now).

## The question, as it stands (not reworded)

"These are two issues of the same drawing set. We priced the job from the earlier one. What in the later issue do we have to look at again before we confirm our price? Give the sheet and what it says now."

It is a contractor's question, in the voice of the suite's others (its twin opens with the same sentence). It names no cloud, revision, block, table, flag or sheet, and hints at nothing unannounced. It does call for the unflagged change: an answer from the clouds and revision rows alone leaves out a bearing pressure that moved by a third, and is wrong by the question's own words. It does not call for saying the change is unmarked (finding 5).

## Check 4 on the twenty answers

Ten findings that found the change and did not say it was unmarked, ten that said it (no cloud, not clouded, unclouded, not in the revision table, no revision row, silent, unannounced, without a revision entry, not flagged in a table cell, no delta). All twenty pass check 3.
- Before (`_UNFLAGGED`): 11 of 20 right. Said: 10 of 10. Not said: 1 of 10 (only "Quantity change: None until footings are redesigned" failed, as it should).
- After (`_NOT_MARKED`): 20 of 20 right, and 7 of 7 further wordings (four of those failed before).
- `_UNFLAGGED` and `coord-revisions` byte for byte: the twin's `to_dict()` hash (the builder's value, taken at 9367934) and the term's regex hash are pinned in the new file.

## Truth against the fixture (both issues built and compared page by page)

Confirmed: A-401 (page 8) 7200 to 7500, clouded, a revision 1 row "REVISED CORE LAYOUT"; A-601 (page 10) door 107 HW-1 to HW-2, clouded, a revision 1 row "REVISED DOOR HARDWARE"; S-001 (page 11) note 2 150 kPa to 200 kPa with no new drawing object and no revision row; the cover's index and revision block changed ("SHEET INDEX UPDATED"); no other sheet's content changed. The truth passes its own four checks before and after the fix.

## Counts, scope, stand-ins

- No stale count in `packages/` (`git grep` for 54, 16 and sixteen: none left; 55 and 17 in the docstrings, the README and the three edited tests). Outside it only the released 1.3.0 note keeps 54 and 147, as CHECK.md says.
- Stand-in answers: PERFECT 4, CARELESS 2, FOUND_ONLY 3, NOTHING 0, and at least three passing and three failing replies per check; all still score as before the fix.
- `git diff 9367934 c41a42a -- webapp funhouse_agent`: empty. The fix touches `tasks.py` and the two test files only. Fingerprints not moved, no re-record.

## What I ran (venv (a), Python 3.12, planlens from PyPI)

- From `packages/aec-document-review`: `pytest -q -p no:cacheprovider tests/test_review_eval*.py` (four files) `tests/test_b15_revisions_unprompted_offline.py tests/test_rv_b15_revisions_unprompted_offline.py tests/test_dr25_spec_and_reissue_offline.py tests/test_package.py tests/test_review_fingerprints.py tests/test_boundary.py`: 276 passed, 2 skipped, exit 0.
- The new file against the builder's `tasks.py` (the fix stashed): 14 failed, 34 passed, exit 1. The 34 that pass there: check 3 on the twenty, the ten said answers, and three further wordings, by design.
- From the root: `pytest -q --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed, exit 0. `packages/aec-agent-core/tests/test_review_checks_together.py`: 23 passed, exit 0.

## Not checked

- No model run: whether a model finds the change unasked, and how often real answers trip the new term, needs a run read in full. The twenty answers are mine, not a model's.
- The builder's claim that 13 of its 15 tests fail on 9367934 was not re-run.
- CI's other jobs (the geotech app, Python 3.11) were not run; the core is unchanged.
- The new term can still pass wrongly where a finding says a negative within one clause of a marking word about another thing ("no change to the clouded dimension on A-401" inside the S-001 finding), and fail "the sheet still shows revision 0" said alone.
