# Foundry brief 8, the standing tests: trace review

Brief 8 (BACKLOG EV-43) ran the standing tests on GPT-5.6 Sol (the Tiny Apps
testers' model) and Claude Opus 5.5: a floor of five single-question
coordination tasks once, then the revision comparison and the two whole-set
reviews three times, Sol also with `sheet_tools`. Raw hand-back: the
gitignored `module_work/field_feedback/2026-11-13_foundry_brief8/raw/`.

## What was read

- Every failed check of every run (`failed_checks.txt` in the lead's scratch,
  from the saved `results_*.json`), each set against what the run delivered.
- Every Sol miss the scorer marked "found only in the run's working notes",
  traced through the run's event record (`activity.jsonl`: tool results, model
  texts, reasoning summaries), `run.json`, the delivered files and the kept
  helper files, by an Opus reader; its write-up is the second half of this
  file.
- The three Opus floor failures, statement by statement (the false-alarm check).
- Not done: a full step-by-step read of each passing run. The usage left in
  the week went to the failures and their fixes (the orchestrating session's
  order of 2026-11-14).

## The scores, corrected

Two scorer defects were fixed and every saved run re-scored offline
(`score_review_suite(..., rescore=True)` logic, no model call):

- **EV-46, the false-alarm check charged correct statements.** Opus failed
  three floor tasks for a row on the planted door 204 that named the rooms
  either side ("the wall between 203 and 205"), a row on the planted callout
  2/A-302 that named the matching cut 2/A-301, and a row on the planted
  footing F4 that said the schedule ends at F3. A table row is now the
  record of the item its first naming cell names, and "between X and Y"
  names a place.
- **EV-47, "was not found on the plan" was not read as absent.** One Sol
  run delivered "Door 509 is scheduled but was not found on the fifth-floor
  plan" and failed the check.

Tasks passed (whole-set = the two whole-set reviews, three repeats each):

| | floor (5) | revisions (3) | whole-set (6) |
|---|---|---|---|
| Opus 5.5, as shipped | 5 (was 2) | 3 | 6 |
| Sol, as shipped | 5 | 3 | 0 |
| Sol, `sheet_tools` | not run | 2 | 1 |

Planted problems found on the long review by Sol as shipped: 19, 16 and 13 of
21 recorded; 20, 16 and 13 after EV-47. No verdict of brief 7 moves with
either fix (GPT-5.4's false alarms there are still caught).

## Why Sol misses what Opus finds

The reader's finding, in short: nothing Sol's main agent held was lost
later. Sol hands contiguous page ranges to reading helpers and gets back
only each helper's judged findings, so the floor plans and the door
schedules are read in different helpers, what one read that another sheet
must be compared with dies with it, and nothing compares the two sides.
Opus read the pages itself, or asked its helpers for inventories and
compared them itself. Any model that splits the work Sol's way would lose
the same items. The fixes: the reader's return contract and its open items
kept in code (BACKLOG DR-46), and whole-set mark reconciliation in code
(PL-42); smaller ones VIS-32 and EV-48; for the next brief, EV-45.

---

## The reader's write-up: Sol's "found only in the working notes" misses, brief 8: where they drop out

Scope: the 15 Sol misses on coord-full-review and coord-long-review that the scorer flags as "found only in the run's working notes". Sources: each run's `activity.jsonl` (tool results, model texts, reasoning summaries), `run.json`, the delivered files and the kept helper files. Scripts are in `b8-trace/scripts/`. `t` = seconds from turn start.

### Bottom line

- **No finding was lost after the main agent had it.** None of the 15 reached the main agent and was then dropped. No context summary replaced one. The log-writing step left none out, and no helper was ended mid-call.
- **The scorer's flag does not mean the model saw the problem.** It is a bare regex over `page_readings.json`/`reads.json`. In those files 204, 509, 109 and 407 appear as box coordinates (`[648, 189, 755, 204]`), as transcribed schedule rows (`204 OFFICE 915 2135 A HM - HW-2`) and as plan tags; A-302 appears as a transcribed callout. In repeat2/sheet_tools "204" never appears in the event record at all.
- **What actually happened:**
  - 1 miss was delivered and then failed by the scorer.
  - 8 were never stated as a problem anywhere.
  - 6 were noticed, or raised as a "verify this against another sheet" question, inside a helper. They died at that helper's return.
- **The common mechanism:** Sol's main agent hands "coordination review of pages X-Y" to reading helpers split by page range. It gets back only each helper's judged findings, and nothing does the cross-sheet comparison.

### Per-miss table

| # | Run | Finding | Where it dropped | Evidence (t = s) |
|---|---|---|---|---|
| 1 | r1 baseline long | door-not-on-plan (204) | Never noticed | One helper (pages 1-20) held A-102 and both schedules. "204" appears only as coordinates and as A-601 schedule rows. Its return (731.2) carries 509 and 407, not 204. |
| 2 | r1 baseline long | door-not-on-plan-5 (509) | **Delivered; scorer false negative** | Comment log row CR-013 (write_xlsx 1751.7): "Door 509 is scheduled but was not found on the fifth-floor plan". `_NOT_WORD` in `review_eval/tasks.py` deliberately excludes "not found", so the check reads it as absent. |
| 3 | r1 sheet_tools long | 204 | Never noticed; across the split | Plans went to helper 1 (pages 1-13), schedules to helper 2 (14-20). Helper 2 reached into the plans for 109 and 407 only. "204" appears only as coordinates. |
| 4 | r1 sheet_tools long | 509 | Evidence held by helper 2, never concluded | It transcribed A-602 with row 509 (101.8-128.4). It searched `109\|407\|509…` over pages 3-19 at 146.2 (text hits only on A-602) and zoomed A-104 for 407 (329.7). It never looked at A-105. Its return (381.5) has no 509. |
| 5 | r2 baseline full (15 sheets) | door-not-scheduled (109) | Lost between helpers; main agent was told | The helper with pages 1-5 listed door 109 on A-101 in its readings (kept file). Its 743-character return (246.5) says "Could not check: hardware assignments are on A-601 … outside the assigned page range". The helper with pages 6-10 (A-601) mentions 109 only as a swing question (205.5). The main agent's follow-ups (361.5) searched A-302, A-202, levels and 07 84 00, never 109. |
| 6 | r2 baseline long | 109 | Never noticed | The plans helper (pages 1-14) had A-101 (text "CORE 109") and read A-601's text layer (371.2). There is no statement anywhere that 109 is unscheduled. |
| 7 | r2 sheet_tools long | 204 | Never noticed; across the split | Helpers covered pages 1-10 and 11-20. "204" does not occur anywhere in the event record. |
| 8 | r2 sheet_tools long | 509 | Never noticed; across the split | The schedules helper transcribed row 509 (247.0/259.0) and noted that 407 was absent. A-105 belonged to the other helper. |
| 9 | r3 baseline long | callout-missing-sheet (A-302) | Raised in a helper's reading, dropped at its return | The pages 1-10 helper also held the G-001 index. Its A-102 reading (233.0/265.1) says "Verify the following references resolve correctly elsewhere in the set: Section reference 2/A-302". The helper never checked, and its return (340.2) omits it. |
| 10 | r3 baseline long | 109 | Never noticed; across the split | The plans helper saw "Door 109 area" on A-101 (205.6). The schedules helper saw tag 109 on A-401 (135.1, a remark on the wall line) and read A-601. Neither compared the plan with the schedule. |
| 11 | r3 baseline long | 204 | Never noticed | Only coordinates and schedule rows (312.7, 325.6). |
| 12 | r3 baseline long | door-not-scheduled-4 (407) | Raised in a helper's reading, dropped at its return | A-104 reading (256.3): "COPY ROOM appears to have three access doors [406, 407, 408] … Confirm … coordinated with the door schedule". The schedule was in the other helper's pages. Not checked, not returned. |
| 13 | r3 baseline long | 509 | Never noticed | Only coordinates. |
| 14 | r3 sheet_tools long | 204 | Seen in a helper, dropped at its return | The pages 1-13 helper zoomed A-102 (317.2) and got "Door 205 … door numbering skips 204 in this view". The schedule was in another helper's pages, so the gap could not be confirmed there. Its return (387.4) omits 204. |
| 15 | r3 sheet_tools long | 509 | Seen in a helper; conflicting readings; dropped | The schedules helper (pages 14-20) asked A-105 about 509 (≈755-760). Four tiles said "Door 509 is not visible in this tile". The whole-page answer, given a question that named 509, placed "the specifically requested 509 tag" on the CORE/COPY ROOM partition, where the tiles show door 508. Its reasoning at 806.5 says "Door 509 appearing in schedules but absent from plans. I need focused zoomed-in checks". Its 10 zooms (806.5) covered 407 and 109 but not 509. Its return (890.9) reports 109 and 407 (both passed) and omits 509. It used 6 of its 14 calls: no budget cut, no cancellation. |

Plans and door schedules went to different helpers in 6 of the 7 runs; r1 baseline long is the exception.

### Things ruled out

- **Context summary.** It fired once in these 7 runs: r1 baseline long, inside the global-audit helper. Its message count dropped from 76 to 1 at 1672.2, and the summary text came back at 1709.7, at about 234 thousand input tokens. That summary and that helper held none of the five missing items. The main agent never summarized in any Sol whole-set run; in the long runs its largest input was 40-75 thousand tokens. (Elsewhere it fired once more, in the r1 sheet_tools coord-revisions main agent, which is outside this question.)
- **The 20 unanswered model calls.** All 20 are vision calls inside `analyze_pdf_page` (17 tiles, 3 whole-page overviews). Each hit `vision_call_timeout_s` = 180 s, was retried once 180-217 s later, and the retry answered. In all 20 cases the tile is in the tool's result. None was at the end of a helper's run: helpers ran on for 1-6 minutes afterwards. The `analysis/standing_sol_unanswered_model_calls.csv` names the wrong calls; it pairs starts and ends by order. For example, its row #583 has a model_end at 862.25. The real cost is 20 × 180 s of waiting, not lost content.
- **Helper call budget.** No reading helper reached `READER_MAX_MODEL_CALLS` (14); the most used was 12.
- **The log-writing step.** Every Sol main agent wrote its log from the helper returns. The only missing item a helper ever returned (509 in r1 baseline long) was written into the log.

### The cause

The cause is general: any model that orchestrates the way Sol did would lose these. It is triggered by an orchestration choice Sol makes and Opus did not. On the 40-sheet set, Sol's main agent splits the review into helpers by contiguous page range and asks each to review its pages. The floor plans (PDF pages 4-8) and the door schedules (pages 19-20) land in different helpers, and the sheet index and callouts may too. The reading helper is told to return "a compact list of findings" (`DOCUMENT_REVIEW_READER_PROMPT`, `REVIEW_READER_DESCRIPTION`). So each helper's evidence stays in its own context and dies with it: the tag list per plan, the schedule rows, and the "verify this against another sheet" questions its own readings raised. The main agent never holds both sides, and nothing in code compares them. A helper catches a cross-sheet problem only when it wanders outside its pages: it did for 109 and 407 in some runs and not for 204 and 509.

Opus avoided this by choice, not by better reading:
- In 4 of its 6 whole-set runs (baseline arm only) its main agent read every page itself.
- In the other two long runs it used "reading job, analysis only" helpers asked for inventories ("every door tag number and which room/wall it sits at", "transcribe the full table… every row") and did the comparison in its own context. Its largest main-agent input was 284 thousand tokens.

Opus would lose the same items if it split the work Sol's way. That is inferred, since no Opus run did; the fake-model check below shows it without a paid run.

**Sol's own behaviour is the smaller part:**
- Item 15: it held conflicting readings and dropped the candidate.
- Items 9 and 12: its own readings asked it to verify, and it did not follow up.
- Items 1, 6 and 10: a helper had both sheets and did not compare them.

**Part of item 15 is a tool cause.** A whole-page answer, given a question naming door 509, claimed a tag that the tiles could not see. `analyze_pdf_page` reports items found only in the tiles (`found_note`), but not a direct contradiction between the page answer and the tiles.

### The fix

**Rule for every model: cross-sheet reconciliation runs in code over the whole set, never over one agent's slice. A reader's open cross-sheet questions survive the reader.**

1. **Whole-set mark reconciliation in code (the main fix).**
   - What: extend planlens' `check_cross_references` (`planlens/planlens/review/tools.py`, `specs.py`, `SHEET_TOOLS`), or add a sibling tool, to reconcile every mark tagged on a plan with the rows of its schedule, in both directions: a mark with no row, and a row with no mark. This covers doors here, and the same pass covers windows, equipment, panels and footings.
   - Source: the text layer first. Where a sheet has no text layer, use the kept page readings (`aec_agent_core/vision_tools.py`, `READINGS_FILE`, which helpers write too).
   - How it runs: in code when a multi-sheet set is opened for review, with the result put in the main agent's context, so it does not depend on the model choosing to call it.
   - Evidence it works: `check_cross_references` already does this for callouts against the sheet index. In the arm that had it, callout-missing-sheet passed 3 of 3 long runs, against 2 of 3 without it. Doors have no such code check, and they failed in both arms.
2. **The reader boundary.** In `aec_agent_core/deep/keep_work.py`, `KeepWork.wrap_tool_call` (the wrap of every `task` call):
   - Keep every reading helper's full record in `kept/`, not only when it "gave up" (`gave_up`), and name the file in the task result.
   - Carry the helper's "could not check" line, and any of its readings that asked for a check against another sheet, into the turn (`TurnScope`) as open items. The main agent sees them before it writes the deliverable.
   - Change the reader contract (`review_build.REVIEW_READER_DESCRIPTION`, `review_prompt.DOCUMENT_REVIEW_READER_PROMPT`) to "return the findings and what you read that another sheet must be compared with (marks, schedule rows, references, each with its sheet)". This is a return contract for every model, not a tune for Sol; it is what Opus already asks for.
3. **Smaller items:**
   - Have `analyze_pdf_page` report when the whole-page answer and the tiles disagree about the same thing (item 15).
   - Scorer: decide whether "was not found on the … plan" should pass (item 2).
   - Scorer: make the working-notes flag apply the same mark-plus-absent-word rule inside the notes, so coordinates and schedule rows stop counting as "seen".
   - Fix the unanswered-calls analysis to pair by `run_id`.

### Checks that need no paid run

1. **Reconciliation on the fixtures (deterministic).**
   - Build the 40-sheet set with `review_eval/coordination_fixture.py` and run the reconciliation.
   - Assert: marks with no schedule row = {109 on A-101, 407 on A-104}; schedule rows with no mark = {204, 509}; callout to a sheet not in the set includes 2/A-302.
   - Assert nothing is reported about doors 203 or 205: Opus's floor run raised that false alarm.
   - Repeat on the 15-sheet set (109, 204).
   - On the scan fixture, assert it either finds them from the readings or says plainly that it could not read the marks. It must never pass silently.
2. **The helper boundary with a fake model.** Use the scripted-model harness in `packages/aec-document-review/tests/page_builds.py` / `test_review_eval_delivered_offline.py`.
   - Script the main agent to send two reading-helper `task` calls split by pages (1-13 and 14-20).
   - Feed helper A the saved A-102 zoom text from r3 sheet_tools ("door numbering skips 204") and helper B the A-601/A-602 transcription, as canned tool results. Each helper's fake model returns findings without 204.
   - Assert the messages the main agent receives on its last model call before `write_xlsx` state that 204 is scheduled and not on A-102 (from the reconciliation or the carried open items).
   - Today's code fails this assertion, which reproduces the loss for any model.
3. **Scorer regressions.**
   - "Door 509 is scheduled but was not found on the fifth-floor plan" is scored as decided.
   - The working-notes flag does not fire on `[648, 189, 755, 204]` or on `204 OFFICE 915 2135`.

### What stays uncertain, and the measurements that would settle it

- **The Sol runs' working notes were not handed back.** `files/.scratch/page_readings.json` and `reads.json` are listed in each `run.json`, but the zip holds them for only one Opus revisions run; the dot-folder was dropped. My reading of the scorer flag rests on the event record, which carries the same tool results. In r2 sheet_tools "204" is in no logged text, so whatever the scorer matched there is not something any agent read back. **Next Foundry brief:** hand back `files/.scratch/` for every run.
- **Opus under Sol's split is inferred, not observed.** The fake-model check (2) settles the mechanism without a paid run. If a live confirmation is wanted, the brief can add one Opus long-review run told to delegate the review by page range.
- **How often Sol splits.** It split the plans from the schedules in 6 of 7 runs here, but it also split 15-sheet runs that passed. This does not change the fix, which removes the dependence on the split.
