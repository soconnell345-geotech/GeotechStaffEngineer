# DR-29: the agent library's generic prompt off the review page (check notes)

## What the item asked

The Document Review page's default build still carries deepagents' generic
prompt: its base prompt ("You are a deep agent ...", with coding-agent advice
such as "read relevant files, check existing patterns"), and the long to-do
and helper (`task`) sections. Recommended: a middleware like
`HideScratchFilesystem` that removes the base prompt word for word and swaps
the to-do and task sections for short texts (the lean build's, removed in
DR-39), behind a switch, measured on the suite.

The lead's ruling: one new switch, off by default, `GEOTECH_REVIEW_LEAN_PROMPT`,
in `LATER_ENVS` (not `ALL_ENVS` or `ARMS`); unset, everything the model is
handed is byte for byte today's; the geotech page is not touched;
`read_pdf_text` is out of scope. Second ruling (the testers' host resolves
deepagents 0.7): the texts to cut are the ones the INSTALLED library
defines; a text it does not define is not required; every text it does
define must be found word for word, or nothing is cut, and the turn's record
says which.

## What was built, and where

- `packages/aec-agent-core/aec_agent_core/deep/lean_prompt.py` (new):
  `LeanFrameworkPrompt`, a middleware for a page's main agent. At each model
  call it removes the library's base prompt (with the blank line before it),
  swaps the to-do section and the helper section of the system prompt for
  the page's short texts, and swaps the `write_todos` and `task` tool
  descriptions (the `task` one keeps deepagents' list of helpers, its
  `{available_agents}`). The five library texts are looked up in the
  installed library by name (`LIBRARY_TEXTS`, `library_texts()`):
  `deepagents.graph.BASE_AGENT_PROMPT`, langchain's
  `WRITE_TODOS_SYSTEM_PROMPT` and `WRITE_TODOS_TOOL_DESCRIPTION`
  (`langchain.agents.middleware.todo`), deepagents' `TASK_SYSTEM_PROMPT` and
  `TASK_TOOL_DESCRIPTION` (`deepagents.middleware.subagents`). A name the
  module does not hold itself is not defined: deepagents 0.7 serves
  `BASE_AGENT_PROMPT` only through the module's `__getattr__`, deprecated,
  and puts it in no prompt. Every text that is defined must be found word for
  word where the library puts it, or the call goes out exactly as it came and
  the outcome names the texts not found. It is marked `keep_last`.
- `aec_agent_core/deep/assembly.py`, `compile_agent`: where deepagents does
  not attach the to-do middleware (0.7.2 on), the core attaches langchain's
  itself and rebuilds; it now goes at the end as before, but ahead of
  trailing middleware marked `keep_last`. With the switch unset there is none,
  so the list is what it was.
- `aec_agent_core/deep/keep_work.py`: `TurnScope.lean_prompt`,
  `note_lean_prompt()` (the primary's calls only; a call not applied is kept
  over later ones) and `lean_prompt_facts()`.
- `aec_agent_core/activity_log.py`: `turn_end` carries `lean_prompt` when the
  middleware ran: `{"applied": true, "cut": [...], "system_chars_cut",
  "tool_chars_cut"}` (`cut`: the labels of the texts taken out or made
  short: `base prompt`, `to-do section`, `task section`,
  `write_todos description`, `task description`) or `{"applied": false,
  "not_found": [...]}`. Unset, nothing is written. A not-applied build also
  logs one warning.
- `aec_agent_core/review_flags.py`: `LEAN_PROMPT_ENV`, `lean_prompt()`, in
  `LATER_ENVS` and `__all__`, documented in the module docstring.
- `packages/aec-document-review/aec_document_review/review_agent.py`: the
  page's texts `LEAN_TODO_PROMPT`, `LEAN_TODO_TOOL_DESCRIPTION`,
  `LEAN_TASK_PROMPT`, `LEAN_TASK_TOOL_DESCRIPTION`.
- `aec_document_review/review_build.py`: `_lean_prompt()`; with the switch
  set the middleware is the main agent's LAST middleware (after
  `OutputLimitGuard`; it hands the reply back untouched), so it sees the
  prompt and tools as the model gets them, including the to-do section the
  core's to-do middleware writes on 0.7.

## What each library version hands the model (the review page's build)

deepagents 0.6.8 (the laptop), from the installed source:

| Text | Where it comes from | Chars |
|---|---|---|
| Base prompt, appended to the page's prompt | `deepagents.graph.BASE_AGENT_PROMPT` | 2,258 |
| To-do section | `langchain.agents.middleware.todo.WRITE_TODOS_SYSTEM_PROMPT` (deepagents attaches the to-do middleware) | 1,370 |
| Helper section, then deepagents' list of helpers | `deepagents.middleware.subagents.TASK_SYSTEM_PROMPT` | 2,144 |
| `write_todos` description | `langchain.agents.middleware.todo.WRITE_TODOS_TOOL_DESCRIPTION` | 3,873 |
| `task` description around the list of helpers | `deepagents.middleware.subagents.TASK_TOOL_DESCRIPTION` | 6,551 fixed |

deepagents 0.7.24 (the newest the core's pin `<0.7.25` allows; it needs
langchain `>=1.4.4`, so with the core's `<1.4.6` a fresh install gets 1.4.4
or 1.4.5), read from the wheel's source, not run:

| Text | Where it comes from | Chars |
|---|---|---|
| Base prompt | none. `create_deep_agent` builds it as `_apply_profile_prompt(profile, "")`: empty unless the model has a harness profile. The built-in profiles (`anthropic:claude-haiku-4-5`, `anthropic:claude-opus-4-7`, `anthropic:claude-sonnet-4-6`, `openai:gpt-5.1/5.2/5.3-codex`, one NVIDIA model) add a model-specific `system_prompt_suffix`, not a base prompt; the page's models match none. `BASE_AGENT_PROMPT` is a deprecated stand-in (`__getattr__`, removal in 0.9.0) that no prompt uses. | 0 |
| To-do section | `langchain.agents.middleware.todo.WRITE_TODOS_SYSTEM_PROMPT`: deepagents no longer attaches the to-do middleware; the core's `compile_agent` attaches langchain's | 1,370 |
| Helper section | none: `TASK_SYSTEM_PROMPT` is gone and `SubAgentMiddleware(system_prompt=None)` writes none by default | 0 |
| Filesystem section | none: only a path-routing line when an `execute` tool is active; the page has none | 0 |
| `write_todos` description | `langchain.agents.middleware.todo.WRITE_TODOS_TOOL_DESCRIPTION` | 3,873 |
| `task` description around the list of helpers | `deepagents.middleware.subagents.TASK_TOOL_DESCRIPTION` (reworded in 0.7; same `{available_agents}` slot) | 1,054 fixed (125 before, 929 after) |
| `task` argument descriptions | `TaskToolSchema` fields `description` and `subagent_type` (new in 0.7; left as they are) | not cut |

langchain's two to-do texts are the same, character for character, in 1.3.4
(installed), 1.4.4 and 1.4.5 (compared from the wheels).

So on 0.7 DR-29's base-prompt part and helper-section part are done
upstream: the switch cuts the to-do section and the two tool descriptions.

## The saving

deepagents 0.6.8, langchain 1.3.4, measured on this branch: the system
prompt of the default build is 17,582 characters off, 12,270 on: 5,312
saved (the blank line and base prompt 2 + 2,258; the to-do section 1,370 for
314; the helper section 2,144 for 148). The two tool descriptions:
`write_todos` 3,873 for 161; `task` 9,076 for 2,820 (its fixed text, 6,551
characters, for 295; the list of helpers, 2,525, stays): 9,968 saved.

deepagents 0.7.24, langchain 1.4.x, counted from the wheels' source, not
from a run: the system prompt loses the to-do section's 1,370 for 314:
1,056 saved; `write_todos` 3,873 for 161 and the `task` fixed text 1,054 for
295: 4,471 saved in the two tool descriptions.

The turn's record gives both cuts (`system_chars_cut`, `tool_chars_cut`)
and which texts were cut (`cut`).

## Calls made

1. **The helper section's text.** The lean build had no helper section in
   its system prompt (its only `task` text was the tool description), so
   `LEAN_TASK_PROMPT` is two new sentences: hand a self-contained job to a
   helper below, the tool's description says what it must carry, several
   can run at once. deepagents' "Available subagent types" list after it is
   kept: it is the page's own helper descriptions. On 0.7 there is no
   section to swap, so it is not used.
2. **The `task` description reworded.** The lean `TASK_DESCRIPTION` described
   one helper, `page_reader`, which this build does not have. Here the
   helpers are the reading helper (`general-purpose`) and the figure helper,
   each described in deepagents' list, and the reading helper's own
   description already says what the lean text said of it (its tools and
   rules, the example job, that it cannot write files). The text keeps what
   is true of every helper (it sees nothing of the conversation, put
   everything in the description, several can run at once) and names the
   list for `subagent_type`. The to-do texts are the lean build's, unchanged.
3. **All or nothing over every text the library defines**, not only the base
   prompt: a prompt cut but tool texts left (or the reverse) is the partial
   cut the ruling forbids. A tool the call does not offer has nothing to swap
   and is not counted missing.
4. **Texts read from the installed library by name**, not pinned copies, so
   "word for word" means as the installed library wrote them; "defined"
   means the module holds the name itself.
5. **Last of the main agent's middleware**, and kept last by `compile_agent`:
   on 0.7 the to-do section is written by a middleware the core adds at the
   end of the list, and a middleware ahead of it would never see the section
   (a core test shows both orders).
6. **Main agent only.** The helpers (reading, figure) keep their prompts; the
   `minimal` build and the geotech page never read the switch (a test scans
   both packages for readers).

## The setting

| Setting | Default | On |
|---|---|---|
| `GEOTECH_REVIEW_LEAN_PROMPT` | off | `1` (`true`, `yes`, `on`); the review page's default build only |

## Tests

New files (both fail on master: the module, the switch and the texts do not
exist there):

- `packages/aec-agent-core/tests/test_dr29_lean_prompt.py` (23; made-up
  library texts in both shapes, so they hold on any installed version): the
  switch is declared like the three newest (in `LATER_ENVS`, not
  `ALL_ENVS`, `ARMS` or on by default; `switches()` clears it, `describe()`
  names it); with all five texts the cut is exact on a prompt held as blocks
  and as one text, other tools untouched, names and arguments kept; with
  fewer texts (no base prompt, no helper section) those are cut and must
  still be found word for word; each one of the five undefined is not
  required and the other four are cut; a library defining none changes
  nothing; a base prompt reworded, a section or tool text not as the library
  wrote it, and a description with no list each leave the very same message
  and tools and are named in the outcome; a name a module serves only
  through `__getattr__` (or empty, or absent) is not defined; a tool not
  offered is not missing; the turn's record (`turn_end.lean_prompt`) says
  applied with `cut`, or not found, and is absent without the middleware;
  after langchain's real to-do middleware the lean prompt cuts the section it
  wrote, ahead of it it cannot; `compile_agent` puts the to-do middleware
  ahead of a `keep_last` one and at the end otherwise; a not-applied call is
  kept over a later one, and helpers and calls outside a turn note nothing.
- `packages/aec-document-review/tests/test_dr29_lean_prompt_page.py` (9, none
  skips on either version): unset, `0` and `off` send the same system prompt
  and tool texts, with the installed library's texts word for word, and no
  `lean_prompt` in the record; the middleware is the main agent's last only
  when set, never on a helper or the `minimal` build; only the page's build
  reads the switch; set, on whichever library is installed, every text it
  defines is gone from what the model is handed and the short text is there,
  the prompt is today's with exactly those edits, every other tool is word
  for word as before, and the record's `cut` and numbers match; the length
  off and on (5,312 saved on the laptop's libraries; printed); the 0.7 shape
  simulated on any version (base prompt and helper section undefined: the
  rest is cut, those stay); an upgrade that reworded the base prompt, the
  to-do section or the `write_todos` text leaves prompt and tools as they
  are, with `not_found` in the record.

The applied-path tests have run on deepagents 0.6.8 only (the laptop). On
deepagents 0.7.24 they have NOT run: that path was written from the wheel's
source and runs only on GitHub.

Unchanged and passing: `tests/test_review_fingerprints.py` (the fingerprints
did not move), `test_dr43_todos_with_calls.py`, `test_review_switches_offline.py`,
`test_review_harness_offline.py`, `test_page_turn.py`, `test_record_ev06.py`
(79 passed with the new page file); in the core and the app, the tests that
reach `compile_agent` (`test_core_move_11c.py`, `test_final_answer.py`,
`test_wave2b_turns.py`, `test_deep_phase3_offline.py`,
`test_deep_phase4_offline.py`, `test_review_harness_offline.py`,
`test_review_minimal_offline.py`, `test_wave2b_guards_offline.py`,
`test_page_fingerprints.py`; 217 passed with the new core file). No existing
test was edited.

## Noted, not fixed

- `read_pdf_text` is still on the page's tool list (named in the item; out of
  scope by the ruling).
- On 0.6.8 the helpers' prompts still get langchain's to-do section
  (deepagents adds the to-do middleware to every declared helper); 0.7 adds
  none to them.
- The page's own to-do rule and `LEAN_TODO_PROMPT` overlap (both name a
  full-set review, a specification-versus-submittal check and a
  comment-response round); kept as the lean build had it.
- A model with one of deepagents 0.7's harness profiles gets that profile's
  suffix in its system prompt; the switch leaves it (it is the library's
  tuning for that model, not a generic base prompt). None of the page's
  models has one.
- On 0.6.8 a model profile of the library that replaces the base prompt
  means the base text is not found: nothing is cut and the record says so.

## For the lead: what a measurement should compare

The suite arm is the pair
`("lean_prompt", {"GEOTECH_REVIEW_LEAN_PROMPT": "1"})`, against `baseline`,
on the review suite's tasks, the same model and install for both.

- First, on every run of the arm, `turn_end.lean_prompt.applied` must be
  `true`, and `cut` says what was cut: all five on deepagents 0.6, the to-do
  section and the two tool texts on 0.7 (note which deepagents the host
  resolved: the saving differs by about four times).
- Then: the scores per task; the primary's input tokens per run and per
  model call (on 0.7 the system prompt is about 1.1 thousand characters
  shorter and the two tool texts about 4.5 thousand; on 0.6, 5.3 and 10
  thousand); model calls per run; to-do use (the runner's to-do-only steps
  and their input share: the to-do texts changed on both versions); helper
  use (`task` calls, by helper, and whether jobs carry what a helper needs:
  the helper texts changed); and the answers' openings and length (on 0.6
  the base prompt's "be concise, no preamble" advice is gone).
- Read the full record of every run, both arms, before believing a
  difference.
