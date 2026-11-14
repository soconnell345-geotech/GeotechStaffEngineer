# DR-46 check: what a reading helper saw reaches the main agent

Branch `fix/dr-reader-boundary`, from master 3d553a8. Commits:

- `bf319e6`: the change and its tests.
- `0aab333`: the Document Review fingerprint re-record, on its own.
- this file, in a commit of its own.

The cause comes from the trace review of the 40-sheet whole-set review. The main agent split the set into reading helpers by page range. The plans and the schedules they must agree with went to different helpers. Each helper returned only "a compact list of findings", so the main agent never held both sides. The same happens to any model that splits the work that way.

## What changed

1. **The reader's return contract.** Three texts changed:
   - `packages/aec-document-review/aec_document_review/review_prompt.py`, `DOCUMENT_REVIEW_READER_PROMPT`
   - `review_build.py`, `REVIEW_READER_DESCRIPTION`
   - `review_agent.py`, `READER_NUDGE` (the helper's last step when its budget runs out)

   A reading helper now returns three short lists. Each item carries its citation and says whether it was read from text or seen by looking:
   - its findings;
   - what it read that must be checked against a page outside its job: marks or tags with their sheet, schedule or table rows, and references to other sheets, sections or documents, given whole rather than only the ones that look wrong;
   - its open items: what it could not check, read or be sure of, and what must be checked against another page.

   The description shown to the main agent also tells it to compare those lists across helpers itself, and that each helper's record is kept as a file. The prompt keeps "You never invent what a page says". The new texts use no door, fixture or task words; a test guards this.

2. **Open items survive the helper, in code.** The change is in `packages/aec-agent-core/aec_agent_core/deep/keep_work.py`.
   - `KeepWork(readers=...)` and `assembly.create_kwargs_for(keep_readers=())` are new. Document Review passes its reader (`general-purpose`).
   - Each time a named reader returns after at least one tool call, its whole record goes to `kept/helper_<what>_<id>.md`, whether or not it gave up. The file is named in the `task` result.
   - The result also carries the open items its tools flagged (`open_items`), whether or not its answer named them.
   - A reader that gave up is kept as before (same note, same records file), and its open items are added.

   **Where the open items come from.** The record holds enough general data. Each look's own result fields say what it did not see, could not read or left to confirm. These are the same JSON results that `page_readings.json` keeps under `result`, so the code reads them from the helper's record. The fields, from `vision_tools`:
   - a call that failed (an error status or an `error` field);
   - `overview_not_read` and `tiles_not_read`;
   - `cut_off`;
   - `scanned_pages` from a text read (pages with no text layer);
   - `edge_note` (lettering past a zoom's edges);
   - `found` rows marked `tiles_only` (things seen only in close-up tiles);
   - `legibility` (small lettering that may not read reliably).

   No words in the helper's prose are matched.

   **When an item is dropped.** The helper read the thing again in one of these ways:
   - a later result of the same tool on the same file and page (for a zoom, the same box) without the flag;
   - for `legibility` and `tiles_only`, a later zoom on that page;
   - for a page with no text layer, a later look at it.

   **The cap.** The list gives one line per file and page, most serious first: failures, then unread parts, cut-offs, pages with no text layer, then the cautions. At most `OPEN_ITEMS_MAX_WORDS` = 300 words go into the result. Past that, the result says how many items are not shown and that every one is in the kept file under "Open items its tools flagged". The kept file always lists them all.

### A made-up example: what the main agent now receives

The values come from the fake-model test: a plan helper whose second look came back cut off, with an answer that does not say so.

```
Findings: none on these pages.
To compare with other pages: marks on the plan, pdf page 1, by looking: QX-11, QX-12, QX-14.
Open items: none.

[This reading helper's whole record (2 tool results, 2,051 characters) is kept in the file 'kept/helper_reading-job-p-open-review-set_taskP.md' in the working folder. Read it with read_text_file -- its index gives the offset of each result -- to compare what it read with other pages, instead of reading the same pages again.]
[Open items its tools flagged, listed in code from its record whether or not its answer names them (it may have settled some in its own words):
- review_set.pdf page 2: the vision answer was cut off at its output limit (the whole-page view), so the end of that reading is missing; small lettering may not read reliably in this view (zoom before relying on it or concluding that something is absent)]
```

The schedule helper's result lists `QX-11, QX-12, QX-13` and names its own kept file.

## What the geotech page's helpers now get

The geotech page's helpers get nothing new:
- The geotech page's build (`funhouse_agent/deep/agent.py` -> `create_kwargs_for`) passes no `keep_readers`, so its `KeepWork` names no reader.
- A geotech helper's result, kept file and threshold are exactly what they were. `test_the_geotech_page_s_helpers_are_unchanged` asserts the identical result and that no `kept/` folder is made; it passes on master and on this branch.
- The geotech page's prompts are untouched.
- The geotech page could opt its `general-purpose` helper in with one keyword. That was not done.

## Tests

PowerShell, with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH` set to this checkout's `packages\aec-agent-core;packages\aec-document-review`. The command was `C:\dev\GeotechStaffEngineer\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider <files>`. Both packages resolved inside this checkout (`aec_document_review.__file__` and `aec_agent_core.__file__` were printed first).

**New tests**
- `packages/aec-document-review/tests/test_dr46_reader_boundary_offline.py` (3 tests). One turn runs through the page's real build (`build_review_page_agent`) with a scripted model, answering each call from the conversation it is shown, and a fake vision engine. Two reading helpers split a made-up set:
  - the plan holds mark QX-14, which the schedule lacks;
  - the schedule holds row QX-13, which no plan shows;
  - the plan helper's second look is cut off at the output limit, and its answer says "Open items: none."
- `packages/aec-agent-core/tests/test_dr46_open_items.py` (9 tests) covers:
  - the fields read;
  - the settling rules;
  - plain-text results;
  - kept and named;
  - the cap;
  - a reader that gave up;
  - the geotech default unchanged;
  - a reader that read nothing;
  - another helper on a page with a reader.

**On today's code (master 3d553a8)** the new tests were run against `git archive 3d553a8` of both packages, each assertion checked one by one with a probe.

`test_both_sides_and_the_open_item_reach_the_main_agent`:
- These assertions fail on master:
  - the reader's system prompt asks for "To compare with other pages", "Open items" and "schedule or table rows";
  - the cut-off on page 2 is carried into the plan result;
  - each result names a kept file that exists and holds what was read.
- These pass on master too: QX-14 in the plan result and QX-13 in the schedule result. The scripted answers carry them, and code cannot make a fake model obey a prompt. What the code guarantees is checked by the prompt assertion and by the kept files.

`test_a_reader_that_answers_findings_only_still_leaves_both_sides` fails on master. These helpers answer findings only, as the trace's did. No kept file is named, so neither side can be reached, and the open item is missing.

`test_the_reader_contract_names_no_task_words` passes on master: it is a guard.

Core tests on master:
- 8 of 9 fail: `KeepWork` takes no `readers`, and there is no `open_items`.
- `test_the_geotech_page_s_helpers_are_unchanged` passes on master, as it must.

**Runs on this branch**

| Root | Files | Result |
|---|---|---|
| Core | `test_dr46_open_items`, `test_core_imports`, `test_core_suite_and_records`, `test_review_checks_together`, `test_saved_files_and_settings` | 53 passed |
| Document Review | `test_dr46_reader_boundary_offline`, `test_core`, `test_kept_turn_work`, `test_review_page_vocabulary_offline`, `test_review_prompt_rules`, `test_ta37_stop`, `test_ta38_turn_ceilings`, `test_review_fingerprints` | 128 passed, 3 failed |
| Root | `webapp/tests/` `test_page_fingerprints`, `test_review_freeze`, `test_public_text`, `test_core_move_11c`, `test_kept_turn_work`, `test_record_ev06`, `test_record_ev14`; `funhouse_agent/deep/tests/` `test_busy_ladder_offline`, `test_figure_runs`, `test_figure_turn`, `test_keep_work_offline`, `test_records_as_files_offline`, `test_review_harness_offline`, `test_review_page_vocabulary_offline`, `test_review_prompt_rules`; `funhouse_agent/tests/` `test_consult_lookup_report`, `test_plot_box_drafts`, `test_subsurface_write_diggs` | 348 passed, 1 failed |

`webapp/tests/test_wheels.py` also names `keep_work`. It was not run, because it builds wheels.

**The Document Review failures (3).** These are `test_review_page_names_no_geotech_report_things[legacy|minimal|legacy+switches]`. They fail with `NoPageHost` (no app has handed in a page host) and fail the same way on master. The new description, nudge and note texts were checked by hand against the same vocabulary pattern, with no hits. The prompt-level vocabulary test passes.

**The root failure (1).** This is `test_app_fingerprint_unchanged[geotech_default]`, with 6 differences: the references helper's texts, and `read_reference_table` gone. It fails identically with master's packages. The reason: this machine's venv has `geotech-references` 1.2.0, while master pins >= 1.7.0. This branch does not move the geotech page's fingerprints.

## Fingerprints

Document Review's fingerprints were re-recorded deliberately, in `0aab333`, from `packages/aec-document-review` with `python -m tests.page_builds record` (planlens 0.15.1). Four hashes moved:
- review_default's reader: `description`, `system_prompt`, and its budget's `final_turn_nudge`;
- review_minimal's helper `final_turn_nudge`, which shares `READER_NUDGE`.

Nothing else moved. The root's fingerprints were not touched; its `review_default` and `review_minimal` still pass.

## Release note (for a tester)

When Document Review splits a large set among reading helpers, each helper now hands back two more things:
- what it read that other pages must be checked against: tags, schedule rows, references to other sheets;
- what it could not check.

The agent is told to compare these across helpers itself. Each helper's full notes are also kept as a file (`kept/helper_….md`, shown as a card under the reply). If a helper's page reading came back incomplete (cut off, partly unread or hard to read), the agent is told, even if the helper forgot to say. Whole-set reviews should catch more mismatches between drawings and the schedules or sheets they point to.

## Noted, not fixed (for the lead to file in BACKLOG)

- **The minimal build's helper** (`minimal_agent.py`, about line 80) keeps its own "compact list" prompt. It shares the new last-step nudge. Its helper is not a `task` call, so nothing is kept for it and no open items are added.
- **The GSE app's own copy of the review page** is `funhouse_agent/review_build.py` and `funhouse_agent/review_prompt.py` (root `review_default` fingerprints). It still has the old contract and names no reader. Its readers behave as before until it is retired.
- **One more card per reading helper.** Kept files get download cards, so a whole-set review now shows one more card for each reading helper. The owner may want these hidden or grouped.
- **Cut-off text reads are not carried.** These are `read_pdf_text` results cut at the character budget (`truncated`). Telling whether the helper asked for the rest needs its page arguments parsed.
- **The open items are not put on `TurnScope`.** The diagnosis suggested this. They reach the main agent in the `task` result, which it reads before it writes. A list kept on the turn would matter only if a host read it later.
- **The diagnosis's main fix is not in this branch.** That fix is the whole-set reconciliation of marks against schedule rows in code (planlens).
