# DR-46 check: what a reading helper saw reaches the main agent

Branch `fix/dr-reader-boundary`, from master 3d553a8. Commits:

- `bf319e6`: the change and its tests.
- `0aab333`: the Document Review fingerprint re-record, on its own.
- `bab876b`: this file.
- The review's fixes: one commit with the code, its tests and this file's update.
- The second fingerprint re-record, on its own, after it.

The cause comes from the trace review of the 40-sheet whole-set review. The main agent split the set into reading helpers by page range. The plans and the schedules they must agree with went to different helpers. Each helper returned only "a compact list of findings", so the main agent never held both sides. The same happens to any model that splits the work that way.

## What changed

### 1. The reader's return contract

Three texts changed:
- `packages/aec-document-review/aec_document_review/review_prompt.py`, `DOCUMENT_REVIEW_READER_PROMPT`
- `review_build.py`, `REVIEW_READER_DESCRIPTION`
- `review_agent.py`, `READER_NUDGE` (the helper's last step when its budget runs out)

A reading helper now returns three short lists. Each item carries its citation and says whether it was read from text or seen by looking:
- its findings;
- what it read **for this job** that must be checked against a page outside it: marks or tags with their sheet, schedule or table rows, and references to other sheets, sections or documents, given whole rather than only the ones that look wrong;
- its open items: what it could not check, read or be sure of, and what must be checked against another page.

"For this job" keeps a narrow job's answer short. The description shown to the main agent also tells it to compare those lists across helpers itself, and that each helper's record is kept as a file. The prompt keeps "You never invent what a page says". The new texts use no door, fixture or task words; a test guards this.

### 2. What survives the helper, in code

The change is in `packages/aec-agent-core/aec_agent_core/deep/keep_work.py`.
- `KeepWork(readers=...)` and `assembly.create_kwargs_for(keep_readers=())` are new. Document Review passes its reader (`general-purpose`).
- **The record is kept.** Each time a named reader returns after at least one tool call, its whole record goes to `kept/reader_<what>_<id>.md`. The file is named in the `task` result, so the main agent can open it with `read_text_file`. A reader that gave up is kept as before (`kept/helper_...`, same note, same records file), and its open items are added.
- **Open items are added.** The result carries the open items its tools flagged (`open_items`), whether or not its answer named them.

**What the code carries, plainly.** It carries how well pages were read: the looks' own result fields that say something was **not read**. These are the same JSON results that `page_readings.json` keeps under `result`. The fields:
- a call that failed (an error status or an `error` field);
- `overview_not_read` and `tiles_not_read`;
- `cut_off`;
- `scanned_pages` from a text read (pages with no text layer);
- `edge_note` (lettering past a zoom's edges).

No words in the helper's prose are matched.

**What the code does not carry:** the tag lists and schedule rows, and the "check this against another sheet" questions a reading raised. Nothing in the record holds those in a general form. Only the contract asks the helper for them, and the kept file holds everything the helper read.

**Left out after review.** Two kinds of flag are no longer open items:
- Things found only in the tiles (`found` rows marked `tiles_only`). These are the normal result of tiling, and the vision tool says to treat them as found. In the saved runs of the 40-sheet review they were on 87 % (Sol) and 97 % (Claude) of the readers' looks.
- The `legibility` caution: about 30 % of zooms.

Both would have buried the holes. In those runs the serious flags occurred on none of 1,195 looks, and 1 of 1,802 helper results was an error. So in practice the open-items block will rarely appear. It appears when a look really failed or came back incomplete.

**When an item is dropped.** The helper read the thing again in one of these ways:
- a later result of the same tool on the same file and page (for a zoom, the same box) without the flag;
- for a page with no text layer, a later look at it.

**The cap.** The list gives one line per file and page, most serious first: failures, unread parts, cut-offs, pages with no text layer, then lettering past a zoom's edge. At most `OPEN_ITEMS_MAX_WORDS` = 300 words go into the result. Past that, the result says how many items are not shown and that every one is in the kept file under "Open items its tools flagged". The kept file always lists them all.

### 3. A reader's record is no card, and it does not count as stopped

**No card under the reply.** `core.is_reader_record` (new) recognises `kept/reader_*`. `turn_jobs` leaves those files out of the turn's cards, beside the caches and scratch it already leaves out.

**Still mirrored to SharePoint.** That is right because:
- the task result in the conversation's history names the file, so a later turn, or a conversation reopened from the mirror, can still open it;
- it is the record of what was read, not a cache that can be rebuilt.

A stopped or gave-up helper's `kept/helper_*` file still gets its card, as before.

**Not counted as a helper that stopped.** A reader's record is not noted on the turn (`TurnScope.kept_files`). So a stopped turn's line "N helper(s) that stopped left what they had read" counts only helpers that stopped or gave up.

**Labelled for what it is.** A later turn's files note labels `kept/reader_*` as "what a reading helper read, kept whole". `kept/helper_*` keeps "what a helper that stopped had gathered".

### A made-up example: what the main agent now receives

The values come from the fake-model test: a plan helper whose second look came back cut off, with an answer that does not say so.

```
Findings: none on these pages.
To compare with other pages: marks on the plan, pdf page 1, by looking: QX-11, QX-12, QX-14.
Open items: none.

[This reading helper's whole record (2 tool results, 2,051 characters) is kept in the file 'kept/reader_reading-job-p-open-review-set_taskP.md' in the working folder. Read it with read_text_file -- its index gives the offset of each result -- to compare what it read with other pages, instead of reading the same pages again.]
[Open items its tools flagged, listed in code from its record whether or not its answer names them (it may have settled some in its own words):
- review_set.pdf page 2: the vision answer was cut off at its output limit (the whole-page view), so the end of that reading is missing]
```

The schedule helper's result lists `QX-11, QX-12, QX-13` and names its own kept file. Its look was complete, so no open-items block is added.

## What the geotech page's helpers now get

The geotech page's helpers get nothing new:
- The geotech page's build (`funhouse_agent/deep/agent.py` -> `create_kwargs_for`) passes no `keep_readers`, so its `KeepWork` names no reader.
- A geotech helper's result, kept file, threshold and turn count are exactly what they were. `test_the_geotech_page_s_helpers_are_unchanged` asserts the identical result and that no `kept/` folder is made; it passes on master and on this branch.
- The geotech page's prompts and card rules are untouched. `is_reader_record` is in Document Review's own `core.py`.

## Tests

PowerShell, with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH` set to this checkout's `packages\aec-agent-core;packages\aec-document-review`. The command was `C:\dev\GeotechStaffEngineer\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider <files>`. Both packages resolved inside this checkout.

**New tests**

`packages/aec-document-review/tests/test_dr46_reader_boundary_offline.py` (4 tests):
- One turn runs through the page's real build (`build_review_page_agent`) with a scripted model, answering each call from the conversation it is shown, and a fake vision engine. Two reading helpers split a made-up set:
  - the plan holds mark QX-14, which the schedule lacks;
  - the schedule holds row QX-13, which no plan shows;
  - the plan helper's second look is cut off at the output limit, and its answer says "Open items: none."
- One turn through the real turn worker (`turn_jobs`) checks the cards. A reader's record makes no card; a stopped helper's kept file and a memo do. The later files note labels both kinds of kept file.

`packages/aec-agent-core/tests/test_dr46_open_items.py` (10 tests) covers:
- the fields read, with tiles-only rows and the legibility caution left out;
- a routine tiled read of 13 pages adds nothing;
- the settling rules;
- plain-text results;
- kept and named as `kept/reader_*`, and not noted on the turn;
- the cap;
- a reader that gave up (`kept/helper_*`, noted on the turn);
- the geotech default unchanged;
- a reader that read nothing;
- another helper on a page with a reader.

**On today's code (master 3d553a8)** the new tests were run against `git archive 3d553a8` of both packages.

`test_both_sides_and_the_open_item_reach_the_main_agent`:
- These assertions fail on master:
  - the reader's system prompt asks for the lists;
  - the cut-off on page 2 is carried;
  - each result names a kept file.
- These pass on master too: QX-14 and QX-13 in the scripted answers. A fake model writes what it is scripted to write.

`test_a_reader_that_answers_findings_only_still_leaves_both_sides` fails on master: no kept file is named.

`test_a_reader_s_record_makes_no_card...` fails on master: the reader's file is carded.

`test_the_reader_contract_names_no_task_words` passes on master: it is a guard.

Core tests on master:
- 9 of 10 fail: `KeepWork` takes no `readers`, and there is no `open_items`.
- `test_the_geotech_page_s_helpers_are_unchanged` passes on master, as it must.

**Runs on this branch, after the review's fixes**

| Root | Files | Result |
|---|---|---|
| Core | `test_dr46_open_items`, `test_core_imports`, `test_core_suite_and_records`, `test_review_checks_together`, `test_saved_files_and_settings` | 54 passed |
| Document Review | `test_dr46_reader_boundary_offline`, `test_titles_cards_scratch`, `test_turn_jobs`, `test_interrupted_turn_files`, `test_working_files_note`, `test_core`, `test_kept_turn_work`, `test_review_page_vocabulary_offline`, `test_review_prompt_rules`, `test_ta37_stop`, `test_ta38_turn_ceilings`, `test_review_fingerprints` | 163 passed, 4 failed before the second re-record: the 3 `NoPageHost` failures below and `review_default`'s fingerprint, which the re-record commit settles (`test_review_fingerprints` 7 passed after it) |
| Root | `webapp/tests/` `test_page_fingerprints`, `test_public_text`, `test_kept_turn_work`, `test_core_move_11c`; `funhouse_agent/deep/tests/` `test_keep_work_offline`, `test_figure_runs`, `test_figure_turn`, `test_records_as_files_offline`, `test_busy_ladder_offline`; `funhouse_agent/tests/test_subsurface_write_diggs` | 232 passed, 1 failed (the `geotech_default` fingerprint below) |

**The pre-existing failures.** Both fail identically on master:
- `test_review_page_names_no_geotech_report_things[legacy|minimal|legacy+switches]` fails with `NoPageHost`. The prompt-level vocabulary test passes.
- The root `geotech_default` fingerprint: this machine's venv has `geotech-references` 1.2.0 against master's >= 1.7.0 pin.

## Fingerprints

Document Review's fingerprints were re-recorded deliberately, each time in a commit of its own, from `packages/aec-document-review` with `python -m tests.page_builds record`.
- **First (`0aab333`):** four hashes moved:
  - review_default's reader: `description`, `system_prompt`, and its budget's `final_turn_nudge`;
  - review_minimal's helper `final_turn_nudge`, which shares `READER_NUDGE`.
- **Second (after the review's fixes):** only the reader's `system_prompt` moved ("for this job").

The root's fingerprints were not touched.

## Release note (for a tester)

When Document Review splits a large set among reading helpers, each helper now hands back two more things:
- what it read that other pages must be checked against: tags, schedule rows, references to other sheets;
- what it could not check.

The agent is told to compare these across helpers itself. Each helper's full notes are kept in the conversation's folder, so the agent can open them again; they add no cards under the reply. If a helper's page reading failed or came back incomplete (cut off, or partly not read), the agent is told, even if the helper forgot to say. Whole-set reviews should catch more mismatches between drawings and the schedules or sheets they point to.

## Noted, not fixed (for the lead to file in BACKLOG)

- **The minimal build's helper** (`minimal_agent.py`, about line 80) keeps its own "compact list" prompt. It shares the new last-step nudge. Its helper is not a `task` call, so nothing is kept for it and no open items are added.
- **The GSE app's own copy of the review page** is `funhouse_agent/review_build.py` and `funhouse_agent/review_prompt.py`. It still has the old contract and names no reader.
- **A failed zoom that the helper fixes with a different box stays listed.** The key holds the exact box. In the saved runs, 1 in 1,802 helper results was an error. This was skipped as optional.
- **Cut-off text reads are not carried.** These are `read_pdf_text` results cut at the character budget (`truncated`). Telling whether the helper asked for the rest needs its page arguments parsed.
- **Activity log events.** A reader's kept record is logged with the `helper_kept` event, whose `why` says it is a reading helper. The event's comment in `activity_log.py` still calls it a stopped helper's.
- **The open items are not put on `TurnScope`.** They reach the main agent in the `task` result, which it reads before it writes.
- **The diagnosis's main fix is not in this branch.** That fix is the whole-set reconciliation of marks against schedule rows in code (planlens).
- **The submodule's empty folder is missing.** `geotech-references` shows as deleted in this worktree. It was not touched by this branch and is not staged.
