# EV-20, EV-31, EV-32, EV-16: check notes (third wave, 2026-11-18)

## Triage

| Item | Call | Why |
|---|---|---|
| EV-20 (the record could not say why, the rest) | BUILT, in part | Three gaps closed in the record only (below). Two parts left: a reasoning summary on tool-only steps needs a change to what the model is asked (a live model's setting); the "legacy review page" switches label no longer exists (`review_flags.describe()` names only the switches that are set). The final card list is already on the record since EV-19 (`turn_files`). |
| EV-32 (answer-number check, arithmetic across paragraphs) | SKIP | Parked, and the reason still holds (one hand-worked case; no-edge-case rule). The check is not in `packages/`: it is the live-smoke detector `module_work/live_smoke/detectors.py` (`unsourced_numbers`, `_derived_nearby`), outside what this branch may edit. |
| EV-31 (a cited equation is the one the library prints) | SKIP | Research, and it cannot be proven without the reference library's text (not in this working copy). It is a new end-of-turn check, so a switch, and it would sit beside `cited_pages.py` / `cited_documents.py`, which other builders are repairing today. Needed: the library's equation numbering per document (manual, edition, equation number to page) as a lookup the core can read. |
| EV-16 (record leftovers) | SKIP | Parked, and the reason still holds: the per-try HTTP trace of the backup host's sign-in client and lock waits for vision tools reading by handle are log-level gaps on the backup host. The client's request timeout is already recorded (`busy_retry.engine_facts`, `client_timeout_s`). |

## What was built (EV-20) and where

All three are additions to the run record. Nothing a model or the user is
handed changes, so there is no switch; no fingerprint moves.

1. **Each helper's own tools on the build record.**
   `aec_agent_core/deep/assembly.py`: `_helper_tool_names()` and one block in
   `record_build()` add `geotech_build["helper_tools"]` = `{helper: [tool
   names]}`. The turn's `turn_start` engine record and `meta.json` carry it
   (both pages' `engine_record` copy `geotech_build` whole).
2. **The image a look read.** `aec_agent_core/vision_tools.py`:
   `_note_image_read()`, called once in `_dispatch_analyze_image` right after
   the image is resolved, writes an `image_read` app event: `image` (the name
   as asked), `source` (`path` or `attachment`), `bytes`, `sha` (the
   fingerprint of the bytes read: the same 16 hex digits `file_facts` gives,
   so the same `sha` as the card's file in the turn's `turn_files`), and
   `mtime` for a file. `activity_log.py`: `image_read` added to `APP_EVENTS`
   and `LOOK_EVENTS` (so it carries the tool call it ran in). `run_report.py`:
   the event's words, and counted in the run's notes rather than listed once
   per look.
3. **The tool calls a model asked for that no tool started.**
   `activity_log.py`: `_asked_calls()`; `on_llm_end` keeps each asked call by
   its id, `on_tool_start` drops it when its tool starts (by
   `tool_call_id`), and `turn_end` writes `calls_not_started` (agent, name,
   args; at most `MAX_CALLS_NOT_STARTED` = 20) and `n_calls_not_started`
   when any are left.

## Calls made

- The helper list is what the build hands each helper. deepagents adds the
  same planning and file tools to every helper through its middleware; they
  are not in the spec, so not listed (the docstring says so). A compiled
  helper with no tool node is left out rather than guessed.
- The look event is written for `analyze_image` only, the tool the item
  names. Not for `render_region` or `analyze_pdf_page`.
- The event is written on every `analyze_image` call that resolved an
  image, a repeated look included: it says what bytes the call got, which
  is what matters when a figure is redrawn between looks.
- `calls_not_started` also lists a call that was answered without its tool
  starting (a guard's refusal), not only one cut off. Both are true "not
  started"; the key's comment says so.
- Shared files: `activity_log.py` (the two event lists, the 18-line
  `_asked_calls`, one line in `__init__`, three in `on_tool_start`, three
  in `on_llm_end`, nine in `turn_end`); `run_report.py` (two lines:
  `COUNTED_EVENTS`, `EVENT_WORDS`). `error_text.py`, `cited_pages.py`,
  `cited_documents.py` untouched.

## Tests

`packages/aec-agent-core/tests/test_ev20_record_why.py` (12 tests):

| Test | Proves | On the base commit |
|---|---|---|
| `test_each_helper_names_its_own_tools` | own tools, inherited tools, a compiled helper's graph, an empty list; a helper that cannot say is left out | fails (no `_helper_tool_names`) |
| `test_the_build_record_carries_them_and_nothing_else_moves` | `helper_tools` on `geotech_build`; `tools`, `helpers`, `prompt_chars` as before | fails |
| `test_no_helpers_no_key` | no helpers, no key | passes (a guard) |
| `test_a_look_names_the_file_version_it_read` | two looks at one figure drawn twice: two events whose `sha` equal `file_facts` of each version, tied to the tool call, with the file time | fails |
| `test_an_upload_is_named_without_a_file_time` | an upload: `attachment`, the bytes' fingerprint, no `mtime` | fails |
| `test_a_missing_image_writes_no_event` | a look at a missing file writes none | fails only on the missing constant |
| `test_the_tool_result_is_the_same_with_and_without_a_record` | the result handed to the model is byte for byte the same | passes (a guard) |
| `test_the_run_report_counts_the_reads` | the run report shows the event as a count | fails |
| `test_a_call_asked_for_and_never_started_is_on_turn_end` | of two calls asked, the one that never started is named with its args | fails |
| `test_a_turn_whose_calls_all_ran_has_no_key` | every call started: no key | passes (a guard) |
| `test_the_call_a_step_cap_cut_off_is_named` | a real agent graph stopped by its step cap: exactly the cut-off call is named, and it is the one after the last that ran | fails |
| `test_a_turn_that_finishes_names_none` | a real agent graph that answers: no key | passes (a guard) |

Should name / should not name, on these made-up turns: the look event 3 of 3
looks that read an image, 0 of 1 that read nothing; `calls_not_started` 2 of
2 turns with a call left over, 0 of 2 turns whose calls all ran.

## Noted, not fixed

- A reasoning summary on tool-only steps needs the live route to ask the
  model for one (a request setting); the record already keeps `reasoning`
  when a model returns it.
- On a geotech turn the switches line still lists any Document Review
  switch set in the environment, though the label "legacy review page" is
  gone.
- `render_region` on an image file writes no `image_read` event.

## For the lead: what a hosted measurement should compare

Nothing to measure on a model: these are record fields. On the next suite or
live run, check that a figure redrawn between two looks shows two
`image_read` events with different `sha`, one of which matches the card's
`sha` in `turn_files`, and that a turn stopped at its step cap has
`calls_not_started` naming the call in the last `model_end`'s `tool_names`.
