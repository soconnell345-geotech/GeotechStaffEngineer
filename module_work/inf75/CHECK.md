# CHECK: model calls, one ladder and every retry on the record (INF-75, VIS-31 part 2, TA-24)

Branch `claude/inf-75-model-call-record`, from master 1c69d95. Task B of
`module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`. Nothing here called a model.

| Commit | Piece |
|---|---|
| 32766d6 | INF-75 (1): side-call timeouts on the record and in the results table |
| 9c1f7d5 | INF-75 (2): engine retries on the record |
| ab2975a | INF-75 (3): what each offered tool said; `show_me` looks counted |
| 123b367 | VIS-31 (2): the side-call limit, 180 s to 90 s |
| 894dc17 | TA-24: one ladder, and a bound on the wait for a vision slot |
| (last) | tests folded so each fails on master; this file |

Files changed, all in the core (`packages/aec-agent-core/aec_agent_core/`):
`deep/vision_engine.py`, `deep/busy_retry.py`, `palantir_sdk_engine.py`,
`tinyapps_engine.py`, `activity_log.py`, `review_eval/runner.py`.
`engine_config.py` was on my list and is not changed.

## 1. INF-75 (1): a side call that times out

**Confirmed on master before changing.**
- `deep/vision_engine.invoke_side_call`: a `VisionCallTimeout` was counted and asked once more with `log.warning` only.
- `deep/vision_engine._invoke_once`: the abandoned call is left on its daemon thread inside `call_slot()`, so it keeps its slot until `model.invoke` returns. Nothing was written when it did.
- `review_eval/runner._activity` counted `model_error` events as "failed calls"; `summarize` counted runs with `error` as "errors". Neither sees a timeout or a refused tool call.
- How the record marks a refused tool call, read from the code: (a) `tool_error`, a tool that raised; (b) `tool_refused`, written by `ActivityLogger._delivered` when no tool's own output exists for the call, which is an unknown tool name or a guard's interception, and ALSO the error message the model is given after a tool raised; (c) a plain `tool_end` whose result is the JSON refusal of `deep/strict_args.unknown_arguments_error`. Brief 7's three refused `analyze_pdf_page(pdf_pages=...)` calls were kind (c): to the record they were ordinary tool results, which is why the table read 0.

**Changed.** A record change, so no setting.
- `vision_engine.invoke_side_call`: each ask gets a number and an id, noted on the look's label (`vision.ask`, `vision.ask_id` on that ask's `model_start` and `model_end`). An ask given up writes the app event `side_call_timeout`: `limit_s`, `kind`, `page`, `tile`, `ask`, `ask_id`, `attempt`, `asked_again`.
- `vision_engine._invoke_once`: whether the caller gave the call up is settled under a lock; an abandoned call that ends writes `side_call_late_end`: `ask_id`, `limit_s`, `held_s` (seconds it held its slot), `outcome` (`answered` or `error`, with the error's class).
- `vision_engine._event`: dispatched as `busy_retry._event` does (`dispatch_custom_event`), never raising.
- `activity_log.APP_EVENTS` and the module docstring: the two events. `LOOK_EVENTS`: they are written with the `tool` and `tool_run_id` of the tool call the look ran in; a late end, whose tool call is over, gets them by its `ask_id`.
- `activity_log`: `tool_error` carries its `tool_call_id`; the `tool_refused` that follows a raise carries `raised: true`, so the pair is one call.
- `review_eval/runner.record_counts(records)`, new: per run `side_call_timeouts`, `side_calls_not_read`, `side_call_late_ends`, `tool_calls_refused` with `tool_calls_refused_by` (`raised`, `no_tool_ran`, `unknown_arguments`), `model_calls_unended`. `run_task` puts them on `run.json` (so in `results.json`); `summarize` adds the columns "side-call timeouts" and "refused tool calls" after "step caps". "errors" and "failed calls" are unchanged.
- `runner.rescore_saved` fills the counts a saved record can give. A record with no timeout events leaves the timeout counts unset, and the table shows `·` for an arm where no run has a count: not recorded is not 0.

**Example (made up).** Sheet 4, tile r1c2, limit 90 s, first ask stalls.
Before: `model_start` (no end), then a second `model_start` and its `model_end`; RESULTS "errors 0, failed calls 0".
After, between the two starts:
`{"event": "side_call_timeout", "limit_s": 90.0, "kind": "tile", "page": 4, "tile": "r1c2", "ask": 1, "ask_id": "3f0c...", "attempt": 1, "asked_again": true, "tool": "analyze_pdf_page", "tool_run_id": "..."}`
and, if the stalled call returns 400 s later,
`{"event": "side_call_late_end", "ask_id": "3f0c...", "held_s": 400.2, "outcome": "answered", ...}`.
RESULTS row gains `| 1 | 0 |`.

## 2. INF-75 (2): an engine retry

**Confirmed.** `palantir_sdk_engine.call_with_retries` retried on `transient_kind` with `log.info` only. Both Foundry chat models (`PalantirSdkChatModel._complete`, `._call_responses`, `PalantirClaudeChatModel._complete`) go through it.

**Changed.** A record change, no setting; the retry policy is untouched.
- `call_with_retries` writes the app event `engine_retry` before each wait: `engine` (`foundry`), `kind`, `attempt`, `wait_s`, `waited_s`, `error` (the class name), `status` where the failure has one. Never the error's text.
- Both models' `_generate` now note their run manager in a context variable for the length of the call (the old body is `_generate_call`). `_retry_event` writes to that run's own handlers under the model call's run id, and falls back to `dispatch_custom_event`. So the event arrives from an agent's graph, from a tool's side call and from a bare `model.invoke(..., config={"callbacks": [...]})`, where a plain dispatch would have had no parent run.
- `activity_log`: `engine_retry` in `APP_EVENTS`; `MODEL_CALL_EVENTS` puts the model call's `run_id` on the record, the same as its `model_start` and `model_end`.

**Example.** Before: `model_start` ... 904 s ... `model_end`. After: `model_start`, `{"event": "engine_retry", "engine": "foundry", "kind": "timeout", "attempt": 1, "wait_s": 1.3, "waited_s": 0.0, "error": "ReadTimeout", "run_id": "<the call's>"}`, `model_end`.

## 3. INF-75 (3): what was offered and used

**Confirmed.** `ActivityLogger._turn_facts` hashes `sorted(names)` only. `runner` had no count of `show_me` looks. `run.json`'s `fingerprint.tools` (the page's `tools_fingerprint`) does hash names, descriptions and schemas, and `turn_start`'s `engine.tools_sha` carries the same value.

**Changed.** A record change, no setting.
- `activity_log._tool_spec`, `_primary_call`, `_turn_facts`: `turn_end` also carries `tools_offered_spec_sha`, a fingerprint of each offered tool's `[name, description, argument schema]`, and `tools_offered_spec_from`: `calls` (the specifications the primary's model calls were sent, where the client shows them) or `build` (`engine.tools_sha`, where it does not, as on the Foundry SDK engines). `tools_offered_sha` is unchanged.
- `runner.record_counts`: `show_me_looks`, tool calls whose arguments carry `show_me` true, read from `tool_start` in the record (the full arguments, not `run.json`'s 300-character copies). RESULTS column "show_me looks"; `rescore_saved` fills it for a saved run.

**Example.** Two turns offered `look_at_page(page)` and `look_at_page(page, show_me)`: before, the same `tools_offered_sha` and nothing else; after, the same `tools_offered_sha` and different `tools_offered_spec_sha`.

## 4. VIS-31 (2): the side-call limit

**Confirmed.** `DEFAULT_TIMEOUT_S = 180.0`, `TIMEOUT_RETRIES = 1`.

**Changed.** `DEFAULT_TIMEOUT_S = 90.0`, with the reasoning beside it. `GEOTECH_VISION_CALL_TIMEOUT_S` still overrides (0 = as long as the engine waits). The single retry and the primary's own calls are untouched. No setting: the brief asks for the default to move.

## 5. TA-24: one ladder. A repair (an unbounded wait)

**Confirmed.**
- `tinyapps_engine`: `DEFAULT_MAX_RETRIES = 4`, `REQUEST_TIMEOUT_S = 180`, both given to `ChatOpenAI`. The OpenAI client retries a timeout like any other failure, so one stuck request is five requests, 900 s, inside ONE try of `AgentBusyRetry`.
- One detail the item's wording misses: the agent ladder does not then run five such tries. `TurnScope.spend_busy` (default 900 s a turn) refuses the first wait after 900 s of failed try, so the turn ends at about 15 minutes. That bound is not on the suite runner or a library caller, where it would be five tries.
- Vision path: the 180 s side-call limit fired before the client's first timeout, and the abandoned thread kept its slot for the client's five requests, up to 900 s.
- `FairSlots.acquire` waited on its condition with no timeout; `_invoke_once` waited on `started` with none.

**Changed.**
- `tinyapps_engine.DEFAULT_MAX_RETRIES = 1`. `GEOTECH_PROMPTER_MAX_RETRIES` still overrides.
- `vision_engine.is_timeout(exc)`: a request that got no answer in time (a `TimeoutError`, the clients' timeout classes by name, HTTP 408 or 504, an `error_name` that names a timeout), read off type, status and name. `client_retries(model)` reads the model's `max_retries`. `may_ask_timeout_again(model, timeouts)`: the client's retries and the ladder's come to at most `TIMEOUT_RETRIES` (1) in all.
- `busy_retry.AgentBusyRetry` (sync and async): counts the timed-out tries of a call and stops when the rule says so, annotating the error (`geotech_busy.why`: "the model gave no answer in time and had been asked again once"). Its `model_busy` event carries `timed_out: true` for a timeout it does wait on. `_decide` takes two new arguments with defaults.
- `vision_engine.invoke_side_call`: a client timeout that reaches it as a busy error counts with the side-call timeouts, under the same rule.
- `FairSlots.acquire(conversation, timeout=None)` returns `True`, or `False` when no slot came in `timeout` seconds (and leaves the line). `call_slot` waits at most `slot_wait_s()`, then writes the app event `vision_slot_timeout` (`waited_s`, `limit_s`, `in_flight`, the look) and raises `VisionSlotTimeout`, a `VisionCallTimeout`, so `describe_error` gives its text as the one line: "the vision model is busy with other looks: no vision call could start within 600 s; this image was NOT read — try again in a minute". It is not asked again and is not a busy error.
- `busy_retry.retry_policy()`: `vision_slot_wait_s` and `timeout_asks_again` on the turn's engine record.

**Left as they were:** `LADDER_S` (15, 30, 60, 120 s), `MAX_RETRY_AFTER_S`, the vision ladder's three tries and 2 s / 4 s waits, `GEOTECH_TOKENS_PER_MINUTE`, the ladder's bound per turn, the Foundry engine's retry policy, every tool description.

**Example.** The endpoint never answers a primary call. Before: 5 requests of 180 s, then the error, about 15 minutes. After: 2 requests, about 6 minutes, then the same error text. With `GEOTECH_PROMPTER_MAX_RETRIES=0`: 1 request, a 15 s wait, 1 more request.

## Settings

| Setting | Default | What |
|---|---|---|
| `GEOTECH_VISION_SLOT_WAIT_S` (new) | 600 | Longest wait for a vision call slot; 0 = no bound. A repair, so on by default. |
| `GEOTECH_VISION_CALL_TIMEOUT_S` | 90 (was 180) | The side-call limit; 0 = the engine's own. |
| `GEOTECH_PROMPTER_MAX_RETRIES` | 1 (was 4) | The Prompter client's own retries. |

No new behaviour sits behind an off switch: pieces 1 to 3 are the record, piece 4 is a default the brief moves, piece 5 is a repair.

## Tests

New, 44 tests in 6 files. On master every one fails; on the branch every one passes.

| File | Tests | On master |
|---|---|---|
| core `tests/test_inf75_side_call_events.py` | 7 | 7 failed |
| core `tests/test_inf75_engine_retry_events.py` | 4 | 4 failed |
| core `tests/test_inf75_tools_offered.py` | 5 | 5 failed |
| core `tests/test_inf75_side_call_limit.py` | 3 | 3 failed |
| core `tests/test_inf75_one_ladder.py` | 14 | the file cannot be imported (`VisionSlotTimeout` is new), so all 14 fail |
| DR `tests/test_inf75_results_counts.py` | 11 | the file cannot be imported (`record_counts` is new), so all 11 fail |

Checked by running the new files against a worktree of master (its packages first on `PYTHONPATH`). Seven must-not-change checks that passed on master by themselves (a timeout outside a run still only raises; a call that answers writes no event; the retry policy and its wait budget; the setting still overrides; one retry; no tools, no hash) are kept as assertions inside tests that fail on master.

`test_inf75_one_ladder.py::test_a_stuck_prompter_call_is_two_requests_in_all` runs the real `build_chat_model` client against a transport that never answers, with and without the client's own retry, and counts requests: 2 either way.

Existing tests edited (an assertion my change made untrue), each one line:
- `packages/aec-document-review/tests/test_wave2b_turns.py` and `webapp/tests/test_wave2b_turns.py`, `test_tiny_apps_sends_the_cap_and_retries_busy_calls`: `DEFAULT_MAX_RETRIES == 4` is now `== 1`.
- `packages/aec-agent-core/tests/test_vision_side_call_timeout_offline.py`: the docstring's "default 180 s" only; no assertion.

Runs, final tree, each judged by pytest's exit code (0 every time):

| Where | Command | Result |
|---|---|---|
| venv (a), core | `cd packages/aec-agent-core && /tmp/dr/bin/python -m pytest -q -p no:cacheprovider <14 files>`: the 5 new files and every core test file that names a changed module (the three `test_vision_*` files and `test_core_suite_and_records.py` among them) | 139 passed |
| venv (a), Document Review | `cd packages/aec-document-review && ... <31 files> tests/test_boundary.py tests/test_review_fingerprints.py`: the new file, the five the brief names, and every DR test file that names a changed module | 552 passed, 2 skipped |
| venv (b) | `/tmp/gse/bin/python -m pytest -q -p no:cacheprovider -n 4 <65 files under webapp/tests, funhouse_agent, report_ingest> <the 14 core files> webapp/tests/test_page_fingerprints.py`: the seven the brief names and every other test file that names a changed module | 1675 passed, 10 skipped |
| guards | `/tmp/dr/bin/python -m pytest -q --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` (before every commit) | 13 passed |

"Names a changed module" is a grep of test files for `vision_engine`, `busy_retry`, `activity_log`, `palantir_sdk_engine`, `tinyapps_engine`, `engine_config`, the suite runner.

## Fingerprints

Not moved. `tests/test_review_fingerprints.py` (venv a) and `webapp/tests/test_page_fingerprints.py` (venv b) pass unchanged; nothing was re-recorded.

## Calls made

1. **The side-call limit is 90 s, not the 60 s the trace review suggested.** Brief 7: 5 s median, under 22 s for 99 % of 15,545 tile calls. Brief 5, another deployment: 74 s at the 99th percentile. 60 s is under brief 5's tail: there, more than one call in a hundred would be given up while still working, asked again on an endpoint already slow, and keep its slot until it ended. 90 s is above both and halves the cost of a stall. `side_call_late_end.held_s` now says how long an abandoned call really took, so the next run shows whether 60 s would have cost an answer.
2. **Client retries on the Prompter route are 1, not 0.** One quick retry absorbs a dropped connection or a short rate limit in under a second, where the agent ladder's first wait is 15 s, and keeps a rate-limited side call at six asks; the vision ladder's own three tries had to stay as they are. The cost is bounded by the timeout rule. At 0 the ladder asks a timeout again itself; both are tested.
3. **"At most once in all" is counted from the model's `max_retries`.** The ladder cannot see what the client did inside a try, so with a client that retries (1 or more) the ladder never asks a timeout again. With the override set to 4, the client still asks a timeout five times: the operator's choice.
4. **408 and 504 count as timeouts**, with the clients' timeout classes and a service's named timeout: a gateway that gave up waiting costs its whole limit too. Before, the agent ladder would ask a 504 five times. This applies on every route, Foundry included, because the ladder is shared.
5. **The slot wait bound is 600 s.** Eight slots at the measured 5 s median serve about 960 calls in 600 s, more than fifty whole-page looks of 17 calls queued ahead, so a queue that is only long does not reach it. A shorter bound would turn heavy but healthy use by several testers into unread images.
6. **A slot timeout is not asked again**, by the vision ladder or as a side-call timeout: the wait was already the bound.
7. **An abandoned call still keeps its slot** until it ends, as before (the reason in `_invoke_once`: stalled connections must count against the cap). The record now shows it.
8. **Event names:** `side_call_timeout`, `side_call_late_end`, `engine_retry`, `vision_slot_timeout`. Field names as listed above.
9. **"Refused tool calls" counts three kinds** (raised, no tool ran, unknown argument). A tool that returns an error of its own inside a normal result (a file not found, a page out of range) is not counted: the tool ran and answered.
10. **An older record** pairs a `tool_error` with its `tool_refused` by tool name and agent, since it has no `raised` field. Two different calls of one tool, one raising and one refused, would count as one there.
11. **`model_calls_unended`** is on `run.json` only, not a table column: with the events it repeats the timeouts, and it is the one count an old record can give (44 for brief 7).
12. **`tools_offered_spec_sha` from two sources.** The `calls` value hashes the specifications as the client sent them; the `build` value is the page's fingerprint. They hash different renderings of the same tools, so they are compared only where `tools_offered_spec_from` matches. Within one suite on one engine it does.
13. **The engine retry event goes through the model call's run manager.** That is what puts the call's `run_id` on it, and what makes it arrive for a model invoked directly with callbacks.

## Release note

For the core and Document Review release. A model call that gets no answer is given up sooner: a stuck call to the model ends in about 6 minutes instead of 15, and a stalled page reading is given up after 90 seconds instead of 180 and asked once more. When every reading slot is taken for 10 minutes, a look now comes back as "the vision model is busy with other looks ... this image was NOT read" instead of waiting without end. Deployments that set `GEOTECH_PROMPTER_MAX_RETRIES` or `GEOTECH_VISION_CALL_TIMEOUT_S` keep their values; a new setting, `GEOTECH_VISION_SLOT_WAIT_S` (default 600, 0 for no bound), sets the wait for a reading slot. The run record (`activity.jsonl`) gains four events (`side_call_timeout`, `side_call_late_end`, `engine_retry`, `vision_slot_timeout`) and a per-turn fingerprint of the offered tools' wording; the review suite's RESULTS gains three columns (side-call timeouts, refused tool calls, show_me looks). GSE's geotech page runs the same core, so the same limits and record apply there.

## For the lead

1. **The next suite run measures the 90 s limit** (brief 8's wheels predate this branch). Read `side_call_timeout` (how many, `asked_again`) and `side_call_late_end.held_s` (would 60 s have cost answers: late ends with `outcome: answered` between 60 and 90 s say yes). Needs wheels built from this branch.
2. **The live check of the rollout gateway**, unchanged from TA-24: does its 429 carry `retry-after` or `x-ratelimit-*` headers (the `model_busy` event and `model_error`'s `http` already record them on this route), and which wait clears its window. That decides the ladder's waits, the vision ladder's tries and `GEOTECH_TOKENS_PER_MINUTE`. Also to watch there with client retries at 1: whether rate-limited side calls now come back "not read" more often (`model_error` on side calls with status 429).
3. **Whether to change the Foundry engine's retries** once `engine_retry` events are seen. Today a timed-out Foundry call (900 s read timeout) is still retried inside the call by the engine; the ladder above it now asks a timeout again once, not four times.
4. **Docs.** `GEOTECH_VISION_SLOT_WAIT_S`, the new defaults and the new events are documented beside the code only. `tinyapps/TINYAPPS.md` and the packages' READMEs are task G's files; `docs/` was not mine.
5. **Old suite results.** `score_review_suite(..., rescore=True)` on a saved run folder fills refused tool calls, show_me looks and `model_calls_unended` from its records. Brief 7's folders can be rescored for those; their timeouts stay `·`.
6. **Task H** builds on `review_eval/runner.py` from this branch.
7. **Merge order with task C.** C edits `deep/keep_work.py`, `deep/limits.py` and `deep/tool_guards.py`, which `busy_retry.py` imports from; I changed none of them and no name they export.

## Noted, not fixed

- **The vision ladder's own busy retries write no event** (`invoke_side_call`, `log.warning` only). A side call refused as busy and asked again shows only as a `model_error` followed by a new `model_start` with the next `ask`. Not in INF-75's three parts.
- **The Foundry proxy and direct routes** (`engine_config.py`: `ChatOpenAI` / `ChatAnthropic` at `MODEL_REQUEST_TIMEOUT_S` 600 s with the clients' default retries) have the same retries inside the ladder. The item names the Prompter route; a model whose `max_retries` is unset reads as 0 to the ladder, so there the ladder asks a timeout again once on top of the client's own default retries.
- **deepagents' summarization call** is made inside a before-model hook, outside the ladder (`deep/assembly.py`, task C's file). As the brief says.
- **The vision probe's own retries** (`vision_probe.py`, task A's file). As the brief says.
- **The Foundry engine's retry policy** is as it was; Foundry is not the testers' host.
- **A late end is only written while the process lives.** An abandoned call still running when the process stops leaves a timeout with no late end; the docstring says to read that as "still holding its slot when the record stopped".
- **Waits before a run's first model call** (F-14's 37 to 123 s): cause outside the packages, as INF-75 says. Not touched.
- **`call_with_retries` with waits of exactly 0** never reaches its wait budget (seen only with the test's stubbed random draw; a real draw is above 0).
