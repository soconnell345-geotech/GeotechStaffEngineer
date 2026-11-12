# Document Review: brief 7 results, and the plan for after the usage reset

> Open work from this plan is in `BACKLOG.md`, not here: EV-42 (the full read), EV-43 (brief 8), INF-74 (Document Review on Foundry) and OWN-141 (the package question); EV-40 and EV-41 are done. The first table below is the score as recorded on Foundry; the corrected one is `RESCORED.md` beside this file. Written 2026-11-12 and approved by the owner the same day.

## Context

Foundry brief 7 came back (`brief7_handback.zip`): the 14 coordination tasks on
the made-up building, on GPT-5.6 Sol in five set-ups (baseline, `show_me`,
`method`, `sheet_tools`, all three) and on GPT-5.4 (baseline only). 84 runs,
no errors, no failed calls.

The owner asked for a review of those results now and a plan for after the
weekly limit resets, with three things folded in: my own view on the package
structure, a trial of Document Review on Foundry, and standing model tests
built by the FDE. The owner's steer: do not perfect the harness for this
model; newer models will do better.

What I did now was a first pass only: the tables, every failed check set
against what the model actually wrote, one comment log read in full, and a
quick re-score. The full read of every run (the owner's standing rule) is
step 2 below.

## What the results say

**1. The model matters far more than any of the three switches.**

| | tasks passed | checks passed | input tokens |
|---|---|---|---|
| Sol, baseline | 12 of 14 | 92 of 95 | 23.9 M |
| Sol, `show_me` | 12 of 14 | 93 of 95 | 27.0 M |
| Sol, `method` | 12 of 14 | 91 of 95 | 23.9 M |
| Sol, `sheet_tools` | 12 of 14 | 92 of 95 | 23.3 M |
| Sol, all three | 12 of 14 | 93 of 95 | 24.5 M |
| GPT-5.4, baseline | 10 of 14 | 80 of 95 | 11.6 M |

One run each, so a difference of one or two checks between set-ups is noise.
Sol to GPT-5.4 is twelve checks.

**2. The narrow tasks are at the ceiling on Sol.** Asked one thing (callouts,
schedules, levels, drawings against the manual or the geotech memo), Sol
finds every planted problem in every set-up. These twelve tasks can no longer
tell set-ups apart on this model. One exception: with `sheet_tools` on, the
revision comparison missed the bearing-pressure change that was made without
a cloud or revision entry (150 to 200 kPa). The other four set-ups found it.
One run; cannot say the switch caused it.

**3. The two whole-set reviews are where things are missed, and the recorded
score overstates them.** The scorer for these two tasks also read the model's
private page notes (`page_readings.json`, `reads.json`), which the user never
receives. Re-scored by me from the reply and the delivered comment log alone
(quick re-score, to be confirmed in step 1), planted problems found out of 32:

| Sol baseline | `show_me` | `method` | `sheet_tools` | all three | GPT-5.4 |
|---|---|---|---|---|---|
| 24 | 28 | 26 | 30 | 29 | 13 |

- In the 40-sheet baseline review, five problems were in the page notes and
  never reached the comment log. Findings are lost between reading and
  writing.
- The bay B to C dimension conflict (7200 against 7500) was missed in 9 of 10
  whole-set reviews, though the narrow levels-and-dimensions task finds it
  every time. That is breadth of attention, not ability.
- `sheet_tools` looks best here and used the fewest tokens, but it is one run.

**4. False alarms are not counted yet.** The one comment log I read in full
(Sol baseline, 15 sheets) has 30 comments for 13 planted problems. The extra
ones I saw are requests for information the made-up set does not carry (no
window schedule, no spot grades) and one comment on the set's filler notes;
none was a plainly wrong statement. A hand count over all twelve whole-set
reports is owed (step 2).

**5. Cost.** About 24 M input tokens per set-up against the brief's estimate
of 8 to 15 M; 122.7 M for Sol in all. A 15-sheet whole-set review is about
1.8 to 3 M tokens and 5 to 12 minutes; a 40-sheet one about 5 M and 20
minutes. GPT-5.4 used half the tokens because it read fewer sheets (13 of 40
on one task).

**6. One tool complaint from the model** (GPT-5.4): `analyze_pdf_page` takes
one page, and it tried a page range.

**Decision from this pass: no switch goes on.** None showed a gain beyond
noise on the recorded scores; `sheet_tools` is the one worth measuring again,
and that is folded into the standing tests (step 5), not a special run for
this model.

## My view on the packages (the owner's question)

Two layers are right; two separately released "cores" are not.

- **planlens and `aec-agent-core` do different jobs and do not overlap.**
  planlens reads documents and calls no model. The core runs a model
  (connections, vision calls, limits, Word and Excel). planlens never imports
  the core. That boundary is sound, and it sits where the lasting value is:
  as models improve, the core should shrink and planlens keeps its worth.
  Keep planlens as its own library.
- **The cost is in the release trains and the unfinished split, not the
  count.** One feature (the sheet tools) crossed three packages and needed
  three test wheels; one fix train on 2026-11-10 needed three releases in
  order. GSE still carries a frozen second copy of the Document Review page
  and alias modules.
- **What I would change, in this order:**
  1. Finish the split's last two steps (INF-35): remove GSE's frozen copy of
     the page and the alias modules.
  2. Release the core and Document Review as one unit: one version number,
     one tag, both wheels. They live in one repo and are pinned exactly to
     each other already. Code does not move.
  3. Bind planlens' tools once and generically, so a new planlens tool
     appears in the app with no hand-written wrapper in the core (part of
     INF-71).
  4. Do not merge planlens into the core, and do not split further.

Item 2 is an owner decision (a new OWN item); 1 and 3 are already on the
backlog.

## The plan, in order

All new items go in `GeotechStaffEngineer/BACKLOG.md` with IDs from
`module_work/backlog_dashboard/sweep_done.py --next`. Full suites run on
GitHub (`ci/<name>`), not on this machine. Every Agent call names its model.

### Step 0. File the hand-back and the notes (minutes)
- Unzip to `module_work/review_eval_results/2026-11-12_foundry_brief7/raw/`
  (gitignored; the logs may carry Foundry identifiers).
- BACKLOG: results note on EV-40, close OWN-140, new items for steps 1 to 6.
- HANDOFF, CLAUDE.md state, memory `project_core_train_state.md`, Backlog
  Desk republish.

### Step 1. Fix the scorer, then re-score brief 7 offline (no model)
- `packages/aec-agent-core/aec_agent_core/review_eval/checks.py`:
  `delivered_text()` reads every file handed to it. Find where the suite
  builds a run's `files` list (the runner in `aec_agent_core/review_eval/`)
  and leave out the conversation's working files (`READS_FILE` and
  `READINGS_FILE` from `aec_agent_core/vision_tools.py`, and the `kept/`
  helper files), using the same rule the page uses for download cards.
- Record both numbers in a run: found in the delivered report, and found
  only in the notes (the second is the "lost between reading and writing"
  measure, as an info line, not scored).
- Re-score the 84 saved runs with the suite's own `rescore=True`; corrected
  `RESULTS.md` beside the originals.
- Tests with made-up values in
  `packages/aec-document-review/tests/test_review_eval_coordination_offline.py`.

### Step 2. Read every run in full (the standing rule)
- Opus helpers, one coordinator, runs split by task group; the lead reads
  the write-up, not the traces. Write
  `module_work/review_eval_results/2026-11-12_foundry_brief7/TRACE_REVIEW.md`.
- Must answer: why the `sheet_tools` revision run missed the silent change;
  why findings in the notes did not reach the comment log; where the tokens
  go (about 3,800 model calls for 840 tool calls per set-up); whether the
  callout check's 200-character window is too strict for a table row (249
  and 209 characters in two runs).
- Hand count of false alarms in the twelve whole-set reports against the
  fixture's planted list, each extra comment classed as: wrong statement,
  true of the made-up set but not planted, or a request for information.
- Per the owner's steer and the no-edge-case rule: no prompt tuning for
  Sol's misses. A finding becomes a BACKLOG item only if it is a scorer or
  tool defect, or would hold on any model.

### Step 3. Document Review on Foundry: build the trial
The Foundry pieces exist for GSE and are reused:
- Move `webapp/foundry_dataset_store.py` into `aec_agent_core` (old name
  kept as an alias in `CORE_ALIASES`, like the other moved files).
- Add `aec_document_review/foundry_entry.py`, modelled on
  `webapp/foundry_entry.py`; pick the conversation store in
  `aec_document_review/core.py` the way GSE does (Foundry dataset when its
  alias is set, else SharePoint, else the local folder).
- Model list from `GEOTECH_FOUNDRY_MODELS` already works in the package
  (`core.py` `foundry_model_choices`).
- Doc: a Foundry section for Document Review beside `docs/FOUNDRY.md`.
- Offline tests copied from `webapp/tests/test_foundry_dataset_store.py`
  and `test_foundry_mode.py`; gate on GitHub.
- Test wheels built as for brief 7 (labels never committed); no PyPI
  release needed for the trial.

### Step 4. One brief to the FDE (brief 8), three parts
Written to `../foundry_handoff/AI_FDE_BRIEF_8.md`, local only; the FDE
does the installs.
- **A. Stand up the app.** Publish Document Review from the wheels;
  checklist: starts, diagnostics pass, upload a PDF, one review turn, a Word
  file back, conversation saved to the dataset and restored.
- **B. Probe the Google route before any code is written.** Today
  `engine_config.py` knows two model families on Foundry (`anthropic`,
  `openai`). Ask the FDE to find and report: how a Google model (Argon 4) is
  called from an app container, whether it takes images and in what form,
  the size and count limits per request, how tokens are reported, and the
  rate limit. A ten-line probe script goes in the brief. The third family
  is added to `engine_config.py` and the vision probe only after this comes
  back.
- **C. Standing model tests** (step 5).

### Step 5. Standing tests for each new model (the FDE builds, we specify)
The brief asks the FDE what their standing evaluation route is and gives
them this to wrap:
- **What runs:** the suite's own `score_review_suite` cell, baseline set-up
  plus `sheet_tools`, on a fixed sentinel set: the two whole-set reviews,
  the revision comparison, the two scanned-set tasks and three narrow tasks
  as a floor. Three repeats each, because one run cannot separate set-ups.
- **What is scored:** planted problems found in the delivered report
  (step 1's corrected scorer), found only in notes, tokens and minutes.
  False alarms stay a hand count from the saved comment logs.
- **When:** each time a new model is added to the workspace; results and
  run folders to a dataset, mirrored as written (never only in `/tmp`).
- **Budget line:** about 10 M input tokens per pass of the sentinel set on
  a Sol-class model, so about 60 M for two set-ups by three repeats; pause
  line stated in the brief.
- The full 52-task suite stays a by-hand run for releases.
- This also gives `sheet_tools` its repeat measurement on Sol and on the
  next model without a special run.

### Step 6. Package structure
- New OWN item with a desk card: "Release the core and Document Review as
  one unit?" with the view above.
- INF-35 (split steps 15 and 16) and INF-71's generic tool binding stay
  where they are on the backlog, ordered after the Foundry trial.

### Order and what runs together
0, then 1 (lead) while 2 runs (helpers), then 3, then 4 and 5 as one brief,
then 6. The Google route's code waits on the probe. Nothing is released
without the owner's word.

## Verification
- Step 1: the offline tests pass on GitHub (`ci/<name>`); the re-scored
  table reproduces the quick re-score above or explains each difference.
- Step 2: `TRACE_REVIEW.md` exists, names every run, and every follow-up in
  it has a BACKLOG ID.
- Step 3: `aec_document_review.foundry_entry` starts locally with the
  Foundry environment variables unset (falls back to the local folder);
  dataset-store tests pass; `test_public_text` passes; page fingerprints
  unchanged.
- Steps 4 and 5: the brief's setup cell and dry run pass on the built
  wheels before the zip goes to the owner (as `rc_check.py` did for
  brief 7).
- Notes checklist after each item: BACKLOG, HANDOFF, CLAUDE.md state,
  memory, Backlog Desk.
