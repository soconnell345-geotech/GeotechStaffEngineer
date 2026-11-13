# CHECK: task C, a turn the tester can stop, ceilings over the whole turn, a spend line

Branch `claude/ta-37-stop-and-turn-ceilings`, from master 1c69d95. Brief: `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`, task C. Items: TA-37, TA-38 (all but the image-call ceiling), TA-05, DR-43 part 1. Nothing here calls a model.

Commits:
- 3ec91f7: TA-37 and TA-38. Core `deep/keep_work.py`, `deep/limits.py` and `deep/tool_guards.py`; DR `core.py`, `turn_jobs.py`, `turn_record.py` and `app.py` (the Stop button); tests `test_ta37_stop.py` and `test_ta38_turn_ceilings.py`.
- 9bf343a: TA-05. DR `budget_panel.py` and `app.py` (the sidebar line); test `test_ta05_spend_line.py`.
- fa512c3: DR-43 part 1. DR `core.py`, GSE `webapp/core.py` (the one constant); test `test_dr43_attachment_note.py`.
- 3c1c63e: TA-05's AppTest check of the sidebar.

Files from another task: none edited. `activity_log.py`, `busy_retry.py`, `vision_engine.py`, `vision_tools.py` and `method_call.py` are untouched. The gm-316 branch's files are untouched.

## TA-37: a Stop button (new behaviour, behind `GEOTECH_STOP_BUTTON`, off by default)

**Confirmed on master.** Nothing could stop a turn once it was sent. `turn_jobs.TurnJob` had no flag. `_run_turn_job` ran `core.stream_turn` to its end. The running-turn view in `app.py` (`_follow_turn_job`) had no control.

**Changed.**
- **The setting.** `turn_jobs.STOP_BUTTON_ENV = "GEOTECH_STOP_BUTTON"`, read by `stop_button_on()` as 1/true/yes/on. A job is stoppable only if the setting was on when it started (`job.stoppable`).
- **The job.** `TurnJob` gains `stop_event`, `stop_requested` and `request_stop()`. `request_stop()` refuses when the job is not stoppable, is already done, or was already asked. When it takes a request it writes `turn_stop_requested` with `activity_log.append_event`.
- **The page.** `app.py` `_follow_turn_job` shows a "Stop" button on a stoppable, running job: the job of this session's own conversation. Its `on_click` calls `job.request_stop`. Once asked, the page says "Stopping…". The status label reads "Stopped" when the turn ends.
- **What the turn is handed.** The job's event goes to `core.stream_turn(stop=...)` only when the job is stoppable. `stream_turn` builds the turn's scope with it: `keep_work.TurnScope(stop_event=...)`.
- **The checks** (all in hooks the turn already had):
  - before each model call of every agent: `OutputLimitGuard.wrap_model_call`, which sits on the primary and on every helper; `keep_work.HelperRecorder.wrap_model_call`; `limits.ModelCallBudgetMiddleware.wrap_model_call`;
  - before each wait for a busy model: `TurnScope.spend_busy` returns False once stop is asked, so `busy_retry` raises instead of sleeping;
  - before each wait for a helper place: `KeepWork._begin` checks, and `_HelperPlaces.acquire(stoppable=True)` looks every 0.5 s;
  - before a continuation of a cut-off reply;
  - before each pass in `core.stream_turn`: the top of `_run_passes`' loop, which covers auto-continue and the page check's pass.
  
  A check raises `keep_work.TurnStopped`, an `Exception`. A helper that meets it has its work kept by `KeepWork` as before. The primary's next call raises it out of the run.
- **The stopped turn.** `turn_jobs` treats any exception raised after the flag was set as a stop: `job.stop_requested`, or a `TurnStopped`.
  - The turn's work is kept as an interrupted turn's (`record.kept`, TA-19).
  - The answer is `core.stopped_turn_text`. It holds `STOPPED_LABEL`, then the tool calls on the record by name, then `KeptWork.line()`, then any reply or useful text written by then, labelled.
  - There is no `error`. The transcript entry gets `"stopped": true`. The history note is `KeptWork.history_note(why="user")`.
  - `turn_stopped` is appended to the activity log after `turn_end`.
- **With the setting unset:** no event reaches the turn, `TurnScope.stop_event` is `None`, every check is a no-op, the helper-place wait is the old untimed `wait()`, no button is shown, and no middleware is added.

**Before and after (made up).** A tester asks "Check every sheet's concrete strength" on a 40-sheet set. Before: they wait out all 40 model calls. After, with the setting on: they press Stop while sheet 1 is read. The model is not called again. The answer reads "*Stopped by you before the turn finished.* Done before the stop: 1 call to read_sheet. What this turn had done is kept: its 1 finished tool result is now in the conversation's history …". "Continue" picks up from it.

**Not this task's (as the brief says):** a vision side call already running ends at its own limit (task B's files), and a wait already under way (a busy-ladder sleep) ends at its time.

## TA-38: ceilings over the whole turn (repairs)

**Confirmed on master.**
1. `core.stream_turn` auto-continued under `continuations < MAX_AUTO_CONTINUES and not _budget_ended_turn()`. The budget's count is langchain's `run_model_call_count`, an `UntrackedValue` reset on each `agent.stream`. So a pass that ended short of the budget started the next pass with a fresh one: the test turn made 8 calls on a budget of 5.
2. `tool_guards.OutputLimitGuard._continue_text` made up to 2 more model calls, uncounted by any budget. An exception in one left the guard, and `AgentBusyRetry` then re-ran the whole original call.
3. An empty forced last reply made `turn_jobs` answer "(no answer text)", with an "Ask again" error.

**Changed.**
1. **One budget for every pass.** `TurnScope(whole_turn_budget=True)` is set only by Document Review's `core._turn_scope`. On it:
   - `ModelCallBudgetMiddleware.after_model` (and `aafter_model`) counts each primary call on the scope (`keep_work.note_primary_model_call`).
   - `_calls_left` uses the larger of the per-run and whole-turn counts, minus counted continuations.
   - `before_model` jumps to the end (with its exhausted message) on a later pass once the forced last reply was made and the turn's count reached the limit.
   - `stream_turn` no longer runs the page check's pass once the budget ended. With `GEOTECH_CITED_PAGES=retry` the check then appends its note through `finish`, as after a retry.
   
   Helpers (`_HELPER` set) keep their per-run budgets. GSE's turns (no `whole_turn_budget`) keep the per-pass count, so GSE's frozen copy of the page behaves as before.
2. **Continuations counted.** `keep_work.take_extra_model_call()` decides whether one more continuation fits. It counts the continuation on the turn (primary) or on the helper's `HelperRecord.extra_model_calls`, and refuses one that would leave no call for the forced last reply. It refuses all on the forced call itself.
   - The budget middleware records each agent's limit (`note_budget`) and subtracts the extras (`extra_model_calls`).
   - Outside a whole-turn scope nothing is counted, as before.
   - A continuation that raises (anything but `GraphBubbleUp`) ends the continuing. The text paid for is delivered as it is.
3. **Empty forced reply.** When the turn ends with no text and `record.scope.budget_reached`:
   - the answer is `core.budget_ended_text`: `BUDGET_ENDED_LEAD`, what it did (tool calls by name from the record), `KeptWork.line()`, and "Ask the agent to continue, or ask a narrower question";
   - the turn's work is kept in the history with `history_note(why="budget")`;
   - the entry gets `"budget_ended": true` and no error.

**Before and after (made up).**
- Budget 5. Pass 1 reads two sheets and says "I'll open sheet 3 next." Before: the auto-continue had 5 more calls, 8 in all. After: it has 2 (one read, then the forced answer), 5 in all.
- Budget 4. Three reads, then an empty forced reply. Before: "(no answer text)". After: "This turn used its whole model-call budget, and the model's last reply, which had to answer, came back empty. What it did: 3 calls to read_sheet. What this turn had done is kept: …"

**Left: a count and a ceiling of image calls per turn. Design for a follow-up once tasks B and E are merged.**
- **Counter.** Put an `image_calls` counter and an `image_limit` on `keep_work.TurnScope`, the place the turn's agents already share across threads. Count in one place, `deep/vision_engine.invoke_side_call`, before each side call that carries an image. That covers tiles, whole pages, `find_like` reads and `show_me` looks, because every vision path goes through it.
- **Setting.** `GEOTECH_TURN_MAX_IMAGE_CALLS`; 0 or unset means no ceiling, so nothing moves by default.
- **Refusing a call.** Past the ceiling, `invoke_side_call` raises a typed `ImageBudgetSpent`. The vision tool in `vision_tools.py` turns it into a tool error the model reads: "this turn's image budget (N calls) is used up; answer from what was read, or ask the user before looking at more". The turn does not end.
- **Warning first.** Before a whole-set read, a page look reports its own cost (up to 17 calls). The `ImageBudgetSpent` warning goes out at 80 % of the limit, like `LOW_BUDGET_NOTE`.
- **The tester's word first.** `stream_turn` can yield a status line, `image calls: k of N`, through the existing `tool_call` items, so the running-turn view shows it.
- **Record.** `turn_end` should carry `image_calls`, which is task B's `activity_log.py`.

## TA-05: a spend line without the Funhouse SDK (new behaviour, behind `GEOTECH_SPEND_ESTIMATE`, off by default)

**Confirmed on master.** `budget_panel.get_budget_status` read only `funhouse.admin.budget`. Without the SDK the sidebar showed no spend at all.

**Changed.** New in `budget_panel.py`: `estimate_on`, `month_tokens`, `estimate_status`, `format_estimate_line`.
- **What is read.** `month_tokens(root)` reads `<root>/conversations/*/activity.jsonl`, where `root` is the caller's own session root (`app._ROOT`, from `profiles.session_root`). It sums `usage.input_tokens` and `usage.output_tokens` of each `turn_end` this calendar month (UTC). Files last changed before the month starts are skipped unopened.
- **Not read.** Another tester's folder, and the shared `<data root>/.trash`.
- **Prices.** `GEOTECH_SPEND_INPUT_USD_PER_MTOK` and `GEOTECH_SPEND_OUTPUT_USD_PER_MTOK`. The cap is `GEOTECH_SPEND_MONTHLY_BUDGET_USD`, optional.
- **The line.** With prices set: "AI spend this month (estimate): $2.80 of $50.00, from 1.0M input and 20k output tokens in your conversations". At 90 % of the cap it becomes a warning. Without prices: "AI use this month (estimate): … tokens …; no token prices set".
- **In `app.py`.** The estimate is shown only when the SDK is absent and the setting is on. It is read once per session, again after each turn (`ss.pop("spend_estimate")` in `_follow_turn_job`), and on a "Refresh estimate" button.
- **Unchanged.** The SDK line, wherever the SDK is present.

**Before and after (made up).** `corp\j.doe` on Tiny Apps, prices set to 2.50 and 15, budget 50. Before: no line. After: "AI spend this month (estimate): $2.80 of $50.00, from 1.0M input and 20k output tokens in your conversations". `corp\a.smith`'s turns are not counted.

## DR-43 part 1: dead `sandbox:` links (a repair)

**Confirmed on master.** DR `core.ATTACHMENT_NOTE`, which `make_save_fn` sets as `saved_note` through `attachment_note_for`, lacked `method_call`'s sentence. GSE's `webapp/core.ATTACHMENT_NOTE` lacked it too.

**Changed.** Appended "Tell the user the file name; a /tmp path or a sandbox: link would not reach them." to both. `method_call.py` is not edited. No other copy of the note exists (checked with grep over the repository).

**Before and after.**
- Before: `write_docx` → "This file is now ATTACHED … do NOT say you cannot attach files."
- After: the same, plus "Tell the user the file name; a /tmp path or a sandbox: link would not reach them."

## Tests

**New tests.** Each one fails on master: source changes stashed, tests kept, `8 failed, 12 errors`.
- `test_ta37_stop.py`, 7 tests. All error on master: the fixture reads `turn_jobs.STOP_BUTTON_ENV`, and the API is new.
  - the stopped turn end to end: no model call after the stop, the answer, the kept history, both events on the log;
  - unset, the turn cannot be stopped and runs as before;
  - a finished job takes no stop;
  - `check_stop` and `spend_busy`;
  - a pass checks first;
  - a helper's model call;
  - the helper-place wait ends within 1.5 s.
  
  On master, three of these guard behaviour that should not change and error only for the missing API: unset, finished, and the outside-a-turn part of `check_stop`.
- `test_ta38_turn_ceilings.py`, 6 tests. All fail on master.
  - auto-continue with what is left (5 calls, not 8);
  - continuations counted, and one left for the forced answer;
  - a helper's continuations go on its own record;
  - a failed continuation keeps its text (raises on master);
  - outside a whole-turn budget nothing is counted (a guard of unchanged behaviour; fails on master only because `note_budget` is new);
  - an empty forced reply.
- `test_ta05_spend_line.py`, 6 tests. All error on master (the fixture reads the new setting names).
  - this month and the caller's folder only;
  - no prices;
  - off unless set;
  - bad input never raises;
  - the SDK line is unchanged (a guard; errors on master only through the fixture);
  - the sidebar line with the setting on and off (AppTest).
- `test_dr43_attachment_note.py`, 2 tests. Both fail on master.

**Runs (pytest exit code, never through a pipe):**

| When | Venv | What | Result |
|---|---|---|---|
| Before writing the new tests | (a) | the nine listed DR files | 170 passed |
| Before writing the new tests | (a) | `test_app_smoke`, `test_review_fingerprints`, `test_boundary` | 17 passed |
| Before writing the new tests | (b) | the five listed GSE files plus `test_page_fingerprints` | 154 passed |
| Before writing the new tests | (b) | `packages/aec-agent-core/tests` | 282 passed |
| Before writing the new tests | (b) | the other tests naming `OutputLimitGuard`, `ModelCallBudgetMiddleware`, `TurnScope`, `turn_scope`, `ATTACHMENT_NOTE` or `budget_panel` (busy ladder, calc budget warning, deep tools, figure helper/runs/turn, judgment calls, plot box drafts, GSE budget panel, core move 11c, judgment check turn) | 225 passed |
| Final tree | (a) | the 4 new files plus every listed DR file plus `test_app_smoke`, `test_boundary`, `test_review_fingerprints` | 208 passed |
| Final tree | (b) | every listed GSE file plus `test_page_fingerprints` plus the 225-test set above plus the core's own tests | 661 passed |
| Final tree, before each commit | (a) | guards `test_public_text.py` and `test_review_freeze.py`, with `--noconftest` | 13 passed |

Existing tests edited: none.

**Fingerprints:** not moved. `tests/test_review_fingerprints.py` and `webapp/tests/test_page_fingerprints.py` pass unchanged, so there is no re-record commit. The stop checks and counters live inside existing middleware, no middleware is added, and the budget numbers are unchanged. `ATTACHMENT_NOTE` is a tool-result note, not a prompt.

## Calls made

- **Where the checks sit.** The stop is checked in `OutputLimitGuard` as well as `HelperRecorder` and `ModelCallBudgetMiddleware`. `OutputLimitGuard` is on every agent, the primary included, even with the page budget off (`GEOTECH_REVIEW_MAX_MODEL_CALLS=off`).
- **The stop is an `Exception`.** `TurnStopped` subclasses `Exception`, not `BaseException`, so `KeepWork` keeps a stopped helper's work. Any exception raised after the flag is set counts as the stop (for example a busy wait refused because of it).
- **Helper-place wait.** It is polled every 0.5 s, only on a stoppable turn.
- **Whole-turn budget only on Document Review.** It is opt-in on the scope (`whole_turn_budget`), set only by Document Review's `stream_turn`, so the geotech page and GSE's frozen copy of the review page keep the per-pass count.
- **Counting point.** Primary calls are counted in `after_model`, once per model node, not in `wrap_model_call`, which a busy retry outside it would call twice.
- **Continuations keep a call back.** A continuation is refused when it would leave no call for the forced answer, and on the forced call itself. A cut forced answer is then delivered as it is.
- **The page check's pass.** It is skipped once the budget ended, rather than counted as an over-budget call.
- **What "what the turn did" says.** For an empty forced reply and for a stop, it is the tool calls by name on the record plus `KeptWork.line()`, and the work is kept in the history for a continue.
- **Spend: missing split.** A `turn_end` with no input/output split (records before INF-66) counts its `turn_tokens` as input.
- **Spend: the month.** The calendar month is UTC.
- **Spend: deleted conversations.** They are not counted: `.trash` is shared across testers.
- **Spend: shown only without the SDK.** With the SDK present the estimate is not shown even if the setting is on, so no two numbers disagree.
- **Spend: price settings.** They are named `..._USD_PER_MTOK`, USD per million tokens.

## Release note

With the core and Document Review release:
- A turn on the Document Review page can be stopped with a Stop button when the host sets `GEOTECH_STOP_BUTTON=1`. What the turn had done is kept, and the agent can be asked to continue.
- With `GEOTECH_SPEND_ESTIMATE=1` and token prices set (`GEOTECH_SPEND_INPUT_USD_PER_MTOK`, `GEOTECH_SPEND_OUTPUT_USD_PER_MTOK`, optionally `GEOTECH_SPEND_MONTHLY_BUDGET_USD`), a host without the Funhouse SDK shows each tester an estimate of their own AI spend this month.
- Unset, both are as before.

Always on (repairs):
- The page's model-call budget now covers the whole turn. An automatic "continue" no longer starts a fresh budget, and the extra calls that finish a cut-off reply count against it. So a long turn can end a few calls sooner than before.
- A failed call while finishing a cut-off reply keeps the text already written.
- A turn whose budget ran out on an empty reply says so, and what it did, instead of "(no answer text)".
- The note after a saved file tells the agent to give the file name, not a `/tmp` path or a `sandbox:` link (Document Review and the geotech page).

## For the lead

- **Live check of the Stop button** (needs the testers' host or a local `streamlit run`, `GEOTECH_STOP_BUTTON=1`):
  1. Ask a question that reads many sheets, and press Stop during a page look.
  2. Expect the look to finish (its side calls run to their limit), then the turn to end with "Stopped by you…", "Stopped" in the status label, `turn_stop_requested` and `turn_stopped` in `activity.jsonl`, and a "continue" turn that picks up from the kept results.
  3. Also press Stop while a turn waits on a busy model, and while a second helper waits for a place.
- **Spend line on Tiny Apps.** Set the three spend settings and compare the estimate with the Prompter key's own meter after a few turns. The estimate leaves out vision side calls if their tokens are not in `turn_end`'s `usage` (that depends on the side calls' callbacks; worth a look in a real record).
- **Owner's call:** turning either setting on for a tester round.
- **Image-call ceiling:** the design above, once tasks B and E are merged.
- **GSE's copies of the turn** (`webapp/turn_jobs.py`, `webapp/core.stream_turn`) have none of this, by design (INF-35).
- **Merge with task H.** Task H edits DR `review_build.py`. This branch does not touch it.

## Noted, not fixed

- A stop requested while the primary writes its final answer, with no later check, lets the turn finish normally. The answer is then the full answer, not "stopped".
- The page check's `retry` mode, after a budget-ended turn, appends its note as after a retry, though no retry was made. Only with `GEOTECH_CITED_PAGES=retry`, off by default.
- A failed continuation of a cut-off reply leaves no event on the activity log. `activity_log.APP_EVENTS` (task B's file) would need a name for it.
