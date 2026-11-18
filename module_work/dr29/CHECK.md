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
in `LATER_ENVS` (not `ALL_ENVS` or `ARMS`); if the installed library's base
text is not found word for word, the prompt is left untouched and the turn's
record says so; unset, everything the model is handed is byte for byte
today's; the geotech page is not touched; `read_pdf_text` is out of scope.

## What was built, and where

- `packages/aec-agent-core/aec_agent_core/deep/lean_prompt.py` (new):
  `LeanFrameworkPrompt`, a middleware for a page's main agent. At each model
  call it removes the library's base prompt (with the blank line before it),
  swaps the to-do section and the helper section of the system prompt for
  the page's short texts, and swaps the `write_todos` and `task` tool
  descriptions (the `task` one keeps deepagents' list of helpers, its
  `{available_agents}`). The five library texts are read from the installed
  library by name (`LIBRARY_TEXTS`, `library_texts()`): `deepagents.graph.
  BASE_AGENT_PROMPT`, langchain's `WRITE_TODOS_SYSTEM_PROMPT` and
  `WRITE_TODOS_TOOL_DESCRIPTION`, deepagents' `TASK_SYSTEM_PROMPT` and
  `TASK_TOOL_DESCRIPTION`. All or nothing: if any is not defined, or not
  found word for word where the library puts it, the call goes out exactly
  as it came and the outcome names the texts not found.
- `aec_agent_core/deep/keep_work.py`: `TurnScope.lean_prompt`,
  `note_lean_prompt()` (the primary's calls only; a call not applied is kept
  over later ones) and `lean_prompt_facts()`.
- `aec_agent_core/activity_log.py`: `turn_end` carries `lean_prompt` when the
  middleware ran: `{"applied": true, "system_chars_cut", "tool_chars_cut"}`
  or `{"applied": false, "not_found": [...]}`. Unset, nothing is written.
  A not-applied build also logs one warning.
- `aec_agent_core/review_flags.py`: `LEAN_PROMPT_ENV`, `lean_prompt()`, in
  `LATER_ENVS` and `__all__`, documented in the module docstring.
- `packages/aec-document-review/aec_document_review/review_agent.py`: the
  page's texts `LEAN_TODO_PROMPT`, `LEAN_TODO_TOOL_DESCRIPTION`,
  `LEAN_TASK_PROMPT`, `LEAN_TASK_TOOL_DESCRIPTION`.
- `aec_document_review/review_build.py`: `_lean_prompt()`; with the switch
  set the middleware goes into the main agent's page middleware, right after
  `HideScratchFilesystem`.

## The saving (deepagents 0.6.8, langchain 1.3.4, this branch)

System prompt of the default build: 17,582 characters off, 12,270 on:
5,312 saved (the blank line and base prompt 2 + 2,258; the to-do section
1,370 for 314; the helper section 2,144 for 148). The two tool descriptions:
`write_todos` 3,873 for 161; `task` 9,076 for 2,820 (its fixed text, 6,551
characters, for 295; the list of helpers, 2,525, stays): 9,968 saved. The
turn's record gives both cuts (`system_chars_cut`, `tool_chars_cut`).

## Calls made

1. **The helper section's text.** The lean build had no helper section in
   its system prompt (its only `task` text was the tool description), so
   `LEAN_TASK_PROMPT` is two new sentences: hand a self-contained job to a
   helper below, the tool's description says what it must carry, several
   can run at once. deepagents' "Available subagent types" list after it is
   kept: it is the page's own helper descriptions.
2. **The `task` description reworded.** The lean `TASK_DESCRIPTION` described
   one helper, `page_reader`, which this build does not have. Here the
   helpers are the reading helper (`general-purpose`) and the figure helper,
   each described in deepagents' list, and the reading helper's own
   description already says what the lean text said of it (its tools and
   rules, the example job, that it cannot write files). The text keeps what
   is true of every helper (it sees nothing of the conversation, put
   everything in the description, several can run at once) and names the
   list for `subagent_type`. The to-do texts are the lean build's, unchanged.
3. **All or nothing over all five texts**, not only the base prompt: a
   prompt cut but tool texts left (or the reverse) is the partial cut the
   ruling forbids. A tool the call does not offer has nothing to swap and is
   not counted missing.
4. **Texts read from the installed library by name**, not pinned copies, so
   "word for word" means as the installed library wrote them.
5. **Main agent only.** The helpers (reading, figure) keep their prompts; the
   `minimal` build and the geotech page never read the switch (a test scans
   both packages for readers).

## The setting

| Setting | Default | On |
|---|---|---|
| `GEOTECH_REVIEW_LEAN_PROMPT` | off | `1` (`true`, `yes`, `on`); the review page's default build only |

## Tests

New files (both fail on master: the module, the switch and the texts do not
exist there):

- `packages/aec-agent-core/tests/test_dr29_lean_prompt.py` (17; made-up
  library texts, so they hold on any installed version): the switch is
  declared like the three newest (in `LATER_ENVS`, not `ALL_ENVS`, `ARMS` or
  on by default; `switches()` clears it, `describe()` names it); the cut is
  exact on a prompt held as blocks and as one text, other tools untouched,
  names and arguments kept; each of the five texts undefined, a base prompt
  reworded, a section or tool text not as the library wrote it, and a
  description with no list each leave the very same message and tools and
  are named in the outcome; a tool not offered is not missing; the turn's
  record (`turn_end.lean_prompt`) says applied or not found, and is absent
  without the middleware; a not-applied call is kept over a later one, and
  helpers and calls outside a turn note nothing.
- `packages/aec-document-review/tests/test_dr29_lean_prompt_page.py` (8):
  unset, `0` and `off` send the same system prompt and tool texts, with the
  installed library's texts word for word, and no `lean_prompt` in the
  record; the middleware is in the main agent's list only when set, after
  `HideScratchFilesystem`, never on a helper or the `minimal` build; only the
  page's build reads the switch; set, the prompt is today's with exactly the
  three cuts (the page's own prompt and the list of helpers whole), the two
  tool texts are the short ones with the list kept, every other tool is word
  for word as before, and the record's numbers match; the length off and on
  (5,312 saved on the laptop's libraries); an upgrade that removed or
  reworded the base prompt leaves prompt and tools as they are, with
  `not_found` in the record; on a library without the texts the page is left
  as it is.

Unchanged and passing: `tests/test_review_fingerprints.py` (the fingerprints
did not move), `test_dr43_todos_with_calls.py`, `test_review_switches_offline.py`,
`test_review_harness_offline.py`, `test_page_turn.py`, `test_record_ev06.py`;
in the core and the app, the tests that read `keep_work`, `activity_log` or
`review_flags` (198 passed). No existing test was edited.

## Noted, not fixed

- **deepagents 0.7 already drops most of this.** From 0.7 (checked against
  the 0.7.24 wheel on PyPI) the library appends no base prompt
  (`BASE_AGENT_PROMPT` is gone; `_LEGACY_BASE_AGENT_PROMPT` is unused) and
  its helper middleware writes no helper section by default. The core allows
  `deepagents>=0.6.8,<0.7.25`, so a fresh resolve (CI, a new host) gets 0.7;
  the laptop has 0.6.8. On 0.7 the switch finds two texts missing and cuts
  nothing, by the all-or-nothing rule; there the page still carries
  langchain's to-do section and the long `write_todos` and `task`
  descriptions. On CI the page tests that need the 0.6 texts skip; the
  core's tests cover the mechanism on any version.
- `read_pdf_text` is still on the page's tool list (named in the item; out of
  scope by the ruling).
- The helpers' prompts still get langchain's to-do section (deepagents adds
  the to-do middleware to every declared helper).
- The page's own to-do rule and `LEAN_TODO_PROMPT` overlap (both name a
  full-set review, a specification-versus-submittal check and a
  comment-response round); kept as the lean build had it.
- On 0.6.8 a model profile of the library that replaces the base prompt
  means the base text is not found: nothing is cut and the record says so.

## For the lead: what a measurement should compare

The suite arm is the pair
`("lean_prompt", {"GEOTECH_REVIEW_LEAN_PROMPT": "1"})`, against `baseline`,
on the review suite's tasks, the same model and install for both.

- First, on every run of the arm, `turn_end.lean_prompt.applied` must be
  `true`; on a host whose deepagents is 0.7 it will say `not_found` and the
  arm is `baseline` (note which deepagents the host resolved).
- Then: the scores per task; the primary's input tokens per run and per
  model call (the system prompt is about 5.3 thousand characters shorter and
  the two tool texts about 10 thousand); model calls per run; to-do use (the
  runner's to-do-only steps and their input share: the to-do texts changed);
  helper use (`task` calls, by helper, and whether jobs carry what a helper
  needs: the helper texts changed); and the answers' openings and length
  (the base prompt's "be concise, no preamble" advice is gone).
- Read the full record of every run, both arms, before believing a
  difference.
