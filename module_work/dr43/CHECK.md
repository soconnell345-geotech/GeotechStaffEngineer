# DR-43 part 2: to-do updates beside the next call (task H)

Branch `claude/dr-43-todos-with-calls` from master 6e2b94c. Code commit 33b152a. Nothing pushed.

## Confirmed on today's code

- The page's rule: `review_prompt.DOCUMENT_REVIEW_PROMPT`, the last bullet of the working rules: "Plan multi-step work with `write_todos` ... and keep it updated. Skip it for a single question." (once in the prompt).
- The reading helper's prompt (`DOCUMENT_REVIEW_READER_PROMPT`) says nothing about a to-do list. It does have the tool: its spec in `review_build._review_reader_subagent` lists no planning middleware, and deepagents adds `TodoListMiddleware` to every custom subagent's base stack (`deepagents/graph.py`, installed 0.6.8, lines 619 and 695). On deepagents 0.7.2 and later the same holds only if their graph still does this; not checked there (see For the lead). `deep/assembly.py` attaches `TodoListMiddleware` to the primary only, and only when the compiled agent lacks the tool.
- The framework's texts (langchain 1.3.4, `langchain/agents/middleware/todo.py`): the tool description says to mark a task in progress BEFORE beginning work, to mark it complete IMMEDIATELY, "don't batch completions", that "you can make several updates at once", and that the answer must be text after the final `write_todos` call. The system-prompt section says never to call it "multiple times in parallel" and to write the final answer "in the message AFTER your last `write_todos` call ... not in the same turn as that call".
- A step holding `write_todos` beside another tool call is accepted: `test_a_step_with_write_todos_beside_another_call_runs_and_is_recorded` builds the real page agent with a fake model whose first reply holds both calls. Both ran, the list was updated, the look's vision call ran, and `activity.jsonl` holds one `model_end` with `n_tool_calls: 2` and a `tool_start` and `tool_end` for each. No guard (`tool_guards`, `limits`, `keep_work`) refused it.

## What changed

Behind `GEOTECH_REVIEW_TODOS_WITH_CALLS`, **off unless set** (arm `todos_with_calls`):

1. `packages/aec-agent-core/aec_agent_core/review_flags.py`: `TODOS_WITH_CALLS_ENV`, its place in `ALL_ENVS`, reader `todos_with_calls()`, module docstring entry, `__all__`, arm.
2. `packages/aec-document-review/aec_document_review/review_prompt.py`: `TODOS_WITH_CALLS_RULE`. `build_document_review_prompt` appends it to the to-do bullet when the switch is on. New `build_document_review_reader_prompt()` returns `DOCUMENT_REVIEW_READER_PROMPT`, plus, with the switch on, "If you keep a to-do list with `write_todos`: <the rule>". `review_build.py` builds the helper's prompt through it. The constants are unchanged.
3. `review_eval/runner.py`: `todo_only_steps(records)`, part of `record_counts`. RESULTS.md has two columns for every arm, "to-do-only steps" and "their share of input tokens"; results.json has per run `todo_only_steps`, `todo_only_input_tokens`, `model_input_tokens`, `todo_only_input_share`. A run saved before the count (no usage in its record) shows `·`, not 0. `rescore_saved` fills the keys for a saved run from its record.

The rule text: "Update the list in the same step as your next tool call, not in a step of its own: every step sends the whole conversation again, so a step that only updates the list costs as much as one that does work."

Before and after (made-up): unset, the bullet ends "Skip it for a single question." With the switch on it ends "Skip it for a single question. Update the list in the same step as your next tool call, not in a step of its own: every step ...".

Count definition: a step is one non-vision `model_end` of an agent; its tool calls are that agent's `tool_start` records before its next `model_end`. It is a to-do-only step when it asked for exactly one call and that call is `write_todos`. The denominator is every step's `usage.input_tokens` in the record (helpers included).

## How the texts sit together (item 4), switch on

| Point | Page's rule | Framework | Together |
|---|---|---|---|
| When to plan | multi-step work; skip for one question | 3 or more steps; skip if trivial | agree |
| Updating | in the same step as the next call | "mark in progress BEFORE beginning work"; "you can make several updates at once" | agree: marking the next task in progress can ride with that task's first call |
| Completion | (silent) | mark complete IMMEDIATELY, don't batch | pull a little apart: completing task N and starting N+1 in one write beside N+1's first call is allowed by the tool text ("when you complete a task, you can mark the next ... in_progress"), but a model reading "immediately" literally still writes a step of its own. The page's sentence cannot override that. |
| Parallel use | one list write beside another tool | "never called multiple times in parallel" | agree: one `write_todos` beside other tools is not multiple `write_todos` |
| Final answer | (silent) | answer text after the last `write_todos`, not in its turn | agree: the last `write_todos` can ride with the last tool call; the answer comes in the next step. If the last write is a separate step, that step still costs one call. |

The page can change its own rule and the helper's prompt. It cannot change the tool description or the section the middleware adds; neither was touched.

## Tests

New `packages/aec-document-review/tests/test_dr43_todos_with_calls.py` (10 tests, all fail on master: the module under test does not exist there): switch declared like the others; unset prompts byte-identical (page and helper) and the master bullet present; set, the rule is in the page's prompt (with and without memory) and the helper's; the helper's spec uses the builder; the built page agent is sent the rule only when set; a `write_todos` step beside a look runs and is recorded; the runner's count on a made-up record (primary, helper, a vision side call, a step beside another call); no usage counts steps and no tokens; RESULTS.md and results.json for every arm, `·` for an old run.

Edited (each makes an assertion untrue otherwise): `packages/aec-document-review/tests/test_review_switches_offline.py` and `funhouse_agent/deep/tests/test_review_switches_offline.py` (arm table names `todos_with_calls`), `funhouse_agent/deep/tests/test_judgment_calls.py` (reader table).

Runs, PowerShell, Windows, packages resolved inside the worktree (printed `__file__` for `aec_document_review`, `aec_agent_core`, `funhouse_agent`):

- from `packages/aec-document-review`: `test_dr43_todos_with_calls`, `test_review_switches_offline`, `test_review_show_me_offline`, `test_review_eval_offline`, `test_review_harness_offline`, `test_review_fingerprints`, `test_inf75_results_counts`, `test_review_prompt_rules`, `test_dr46_reader_boundary_offline`, `test_review_page_vocabulary_offline`: 173 passed, 2 skipped, exit 0.
- from the root: `funhouse_agent/deep/tests/test_review_switches_offline.py`, `test_judgment_calls.py`, `funhouse_agent/review_eval/tests/test_review_eval_offline.py`, `webapp/tests/test_page_fingerprints.py`, `test_public_text.py`, `test_review_freeze.py`: 146 passed, 6 failed. The 6 failures (5 in `test_judgment_calls.py`, `test_app_fingerprint_unchanged[geotech_default]`) fail the same on master 6e2b94c in this worktree (checked by stashing my changes): `geotech-references` is not checked out there, so `read_reference_table` and the references helper's text are absent from the geotech build. Environmental, not this change. Nothing re-recorded. Document Review's fingerprint test passes unchanged.

## Fingerprints

Not moved by this change. DR's `test_review_fingerprints.py` passes. GSE's `test_page_fingerprints.py` fails only on `geotech_default` for the missing submodule, as on master; its Document Review builds pass. The lead should confirm on GitHub, where the submodule is present.

## Calls made

- Name `GEOTECH_REVIEW_TODOS_WITH_CALLS`, arm `todos_with_calls`.
- The helper gets the line, worded conditionally ("If you keep a to-do list"), because its prompt never asks for one and the tool is there through deepagents' base stack.
- Denominator is the record's own sum of step input tokens, not the run's `tokens_in`, so the share is internally consistent; helpers' steps are in both numerator and denominator.
- The count is per run, for every arm, regardless of the switch.

## Release note

None for testers: off unless set. If the owner turns it on, the page asks the model to update its to-do list in the same step as its next call.

## For the lead

- Run the `todos_with_calls` arm beside baseline in the next standing run (EV-43); the default is the owner's call after it. Read the new columns, and read the runs: a model may keep writing separate to-do steps because of the framework's "immediately".
- Proposed `CLAUDE.md` line for the switches list: "`GEOTECH_REVIEW_TODOS_WITH_CALLS` (off; DR-43 part 2, arm `todos_with_calls`): the Document Review page's to-do rule, and its reading helper's prompt, add one sentence telling the agent to update the list in the same step as its next tool call; unset, the prompts are unchanged. The suite counts to-do-only steps and their share of input tokens for every arm."
- If the arm does not move the count, an override of the framework's text would be needed (for example one added line in the tool description saying completions may ride with the next call). I did not make it: it lives in `deep/assembly.py` and would change the geotech page too.
- Check that deepagents on the live hosts (newer than 0.6.8) still gives custom helpers `write_todos`.
- GSE's frozen copy of the prompt (`funhouse_agent/review_prompt.py`) and the geotech page are unchanged.

## Noted, not fixed

- Parallel helpers with the same agent label share one open step in the count; rare, and it only moves the count slightly.
