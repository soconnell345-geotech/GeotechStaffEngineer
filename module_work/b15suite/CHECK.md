# Brief 15: the suite's unprompted question on two issues (`coord-second-issue`)

Branch `claude/b15-suite-unprompted`, from 9367934 (1.3.0). Work commit a9cb7c0; this file alone in the last commit. For BACKLOG DR-25 (suite gaps) and DR-06 (the method's comparison of what no cloud marks).

## Confirmed on today's code, before the change

- `review_eval/tasks.py`, `COORDINATION_TASKS`: `coord-revisions` is the only task on issues 0 and 1 (`fixture_coord_set_prev`, `fixture_coord_set`); its question asks, for each change, whether it is clouded and listed, and its third check wants S-001, 200 kPa and `_UNFLAGGED` within 400 characters. `coord-reissue` (issues 1 and 2) names clouds and the revision block too. So no task showed an unmarked change found unasked.
- `coordination_fixture.REVISIONS`: two flagged changes (A-401 grids B to C 7200 to 7500; A-601 door 107 HW-1 to HW-2) and one not (S-001 note 2, 150 to 200 kPa). No new fixture was needed.

## The task

- **Id:** `coord-second-issue`, placed straight after `coord-revisions`; same documents, category `compare`, doc type `coordination_set`.
- **Question, in full:** "These are two issues of the same drawing set. We priced the job from the earlier one. What in the later issue do we have to look at again before we confirm our price? Give the sheet and what it says now."
- **Checks** (all `stated_together`; plus the two every task gets):
  1. `core-dimension`: A-401 beside 7500 (the twin's check, the same object).
  2. `door-hardware`: 107 beside HW-2 (the twin's check).
  3. `bearing-pressure`: S-001 beside 200 kPa, within 300 characters. The change is found.
  4. `bearing-pressure, said not flagged`: S-001, 200 kPa and `_UNFLAGGED` (the existing term) within 400 characters in ONE finding (`scope="finding"`: a line, a list item with what is nested under it, a table row read with its column heads, or under a heading that holds the word).
- **Truth:** the fixture's two sentences, the flagged pair's sheets, the S-001 change with no cloud and no entry, the cover; in the others' form, and it passes its own four checks.
- `coord-revisions` is untouched: its `to_dict()` is pinned by sha256 in the new test file, taken from 9367934.

Before and after, made up. "S-001: bearing pressure is 200 kPa, was 150 kPa." Before: no task scored this unless asked about clouds (on `coord-revisions` it scores 0 of 1 on the third check). After: check 3 passes, check 4 fails; with "Not clouded." added to the same item, both pass.

## The two answers the brief asks about (run offline, in the test file)

- **Finds the change, calls it a drafting error** ("S-001 reads 200 kPa where the earlier issue read 150 kPa; likely a drafting error, ask the engineer"): check 3 PASSES, check 4 FAILS. Limit: if the same finding holds "no", "not", "never" or "without" about anything else ("...as the footings on S-501 were not resized"), check 4 passes.
- **All three changes in a table with a flagged yes/no column:** checks 1 to 3 pass; check 4 PASSES when the S-001 row says No, FAILS on Yes or an empty cell, whatever the other rows say (the row is read as "Flagged?: No"). Limit: "No" under any other head in that row ("Affects our price?: No") passes too.
- **False alarms:** no `no_false_alarm` check is on this task (`test_the_tasks_cover_every_planted_problem` still pins the six tasks that have one), and the fixture model those checks draw on is not read here. An answer naming every sound sheet, a good callout and a door with words like "not missing" passes all six checks. Only the two automatic checks (giving up on legibility; a "page 0" citation) can fail besides; "issue 0" does not trip the second.

## What a run would and would not show

- Would show: per run, whether the unmarked change is found when nothing says to look (check 3), and apart from that whether the answer says it is unmarked (check 4). Beside `coord-revisions` on the same documents, the difference the prompt makes; between the baseline and `method` arms, whether the method's comparison clause earns anything (DR-06).
- Would not show: that the answer is right about why (check 4 reads a negative word in the finding, not its object: read each trace); anything about false alarms; anything a reviewer finds on other documents. One task, three changes: a single run is an anecdote, so repeat it (three repeats as brief 8).
- Check 4 can fail a sound answer that states the change on one plain line and its silence on the next plain line with no heading over them (each plain line is its own finding; the known limit of `scope="finding"`, DR-25).

## Counts moved (found with `git grep`; each edit to an existing test is here)

- `tasks.py` docstring: sixteen to seventeen coordination tasks, 54 to 55 in all; and a paragraph on the pair in the coordination tasks' block comment.
- `review_eval/__init__.py` docstring and `packages/aec-document-review/README.md`: 54 to 55 tasks.
- `tests/test_package.py`: `test_the_suite_has_its_55_tasks_all_on_this_page` (name, two `== 55`), the dry run's "55 tasks".
- `tests/test_review_eval_coordination_offline.py`: `len(COORDINATION_TASKS) == 17`, the dry run's "17 tasks".
- `tests/test_review_eval_offline.py`: `len(coordination) == 17` and its comment's "seventeen".
- Left as it is: `module_work/aec_1_3_0/RELEASE_NOTE.md` line 68 ("54 tasks with 147 checks"), the note of a released version; the code now holds 55 and 151. `webapp/` and `funhouse_agent/` keep no copy of these counts or of any `coord-` id (grep: none).

## Tests

New file `tests/test_b15_revisions_unprompted_offline.py`, 15 tests. On 9367934 (the file run against that tree's package) 13 fail with `KeyError: 'coord-second-issue'`; two pass there by design: `test_the_twin_is_exactly_as_it_was` (the pin) and `test_every_check_has_its_replies` (the reply table's shape). Stand-in answers: `PERFECT` (4 of 4), `CARELESS` (reads only what is marked: 2), `FOUND_ONLY` (3), `NOTHING` (0), and per check at least three replies that pass and three that fail.

Venv (a), Python 3.12, planlens 0.17.0, from `packages/aec-document-review`:
- `pytest -q tests/test_review_eval_coordination_offline.py tests/test_review_eval_delivered_offline.py tests/test_review_eval_offline.py tests/test_review_eval_scales_offline.py tests/test_dr25_spec_and_reissue_offline.py tests/test_review_fingerprints.py tests/test_boundary.py tests/test_package.py tests/test_b15_revisions_unprompted_offline.py`: 228 passed, 2 skipped (public documents not in this checkout), exit 0.
- From the root, `pytest -q --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed. `packages/aec-agent-core/tests/test_review_checks_together.py`: 23 passed (the core is not changed).

**Fingerprints:** not moved; `test_review_fingerprints.py` passes with no re-record. No setting added.

## Calls made

- The id does not start with `coord-revisions`: `select` takes ids by prefix, so `ids=["coord-revisions"]` still runs the twin alone (tested).
- Placed after its twin, not at the end: `test_dr25_spec_and_reissue_offline.py` asserts the last two tasks are DR-25's, and that is no count.
- Check 4 reads one finding, where the twin's reads a 400-character window: unasked, a "not" in the neighbouring item is likely, and with the window it passed ("A-401 ... Not clouded" beside the S-001 item is a failing reply in the tests).
- The question says "earlier" and "later", not first and second: the documents are issue 0 and issue 1.

**Release note:** None. A suite task; no tester or page sees it.
## For the lead

- Not run on a model. A run needs the two issues only (30 pages), costs about what `coord-revisions` does, and is read in full. Its total is not comparable with a 54-task run's; the 1.3.0 release note's counts (above) are for a later note to restate.

## Noted, not fixed

- `_UNFLAGGED` takes a bare "no" or "not" (the two limits above). A narrower term would change what `coord-revisions` scores, which this brief leaves as it is.
