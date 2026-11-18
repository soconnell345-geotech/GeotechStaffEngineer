# Review: EV-20 (in part) on `claude/ev-32-answer-checks`

## 1. Verdict

**MERGE AFTER FIXES**, reviewed head `a0f5d99` (base `519c425`). Both fixes are made on `claude/rv-ev-20` (`6ddeef5`, `339fd9f`): merge that branch, not the reviewed one. One conflict to resolve by hand against today's integration branch (section 3, path 12).

## 2. Findings

| # | Severity | File and line (reviewed head) | What is wrong | The input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | should-fix | `activity_log.py` 1152-1162 (`turn_end`) | `calls_not_started` lists a call the model WAS answered about (a guard's refusal, an unknown tool name: each already on the record as `tool_refused`) exactly as it lists a call a step cap cut off, and on turns that end normally. It keeps the FIRST 20, so a turn with 20 refusals drops the cut-off last call, the one the item is for. The builder's "Calls made" says refusals are listed; the record itself did not say which kind an entry was. | A finished turn: three calls asked, one runs, a made-up guard refuses one, one names no tool: `n_calls_not_started` 2, nothing marks them. 25 refusals, then a call cut off by the step cap: the 20 listed are refusals `no0`..`no19`; the cut-off call is absent. Same on deepagents 0.6.8 and 0.7.24. | **Fixed, `6ddeef5`.** An entry the model got an answer to carries `"answered": true` (seen from `tool_delivered` and from the graph's tools node); unanswered entries come first under the cap; `n_calls_not_answered` counts them. An id asked again loses the mark of its earlier refusal. |
| 2 | should-fix | `run_report.py` 1801 (the call's view) with `activity_log.LOOK_EVENTS` | Every app event tied to a tool call is drawn in the warning colour. `image_read` is written on every look, so every `analyze_image` call that went well showed a warning line in the report. Its words called an upload "the image file". | Three made-up looks (a file, an upload, the file again): three `<div class='warn'>` lines, nothing wrong on any. | **Fixed, `339fd9f`.** `run_report.PLAIN_EVENTS = ("image_read",)` is drawn as a note; the timeouts stay warnings; the words no longer say "file". |
| 3 | note | `run_report.py` | The report does not show `calls_not_started` at all: a reader of the HTML sees "reached its step cap" but not which call was cut off. Where to show it is a design choice: left. | A made-up capped turn: the key is on `turn_end`, no line in the report. | Needs the lead's call (a line in the run's notes when `n_calls_not_answered` > 0). |
| 4 | note | `activity_log.py` `turn_end` | `args` of each listed call is not cut. Worst case below (section 3, path 11). `tool_start` writes args whole too, so this is no new kind of size. | One cut-off `write_file` call with a 200,000-character body: 200,000 characters on `turn_end`. | None now; cap at `DEFAULT_MAX_CHARS` if a record ever shows it. |
| 5 | note | `activity_log.py` `_asked_calls` | A call with no id is never listed (the docstring says so; `n_calls_not_started` cannot count it). On langchain 1.4.5 a tool call with no id fails the turn in the tools node anyway (`TypeError`), with or without this branch. | `id: None` on a made-up call. | None. |
| 6 | note | `activity_log.py` `on_llm_end` | Calls are keyed by id alone, not by agent. A helper and the main agent using the same ids: all-started turns are right (tested); if one of two same-id calls is cut off while the other is still pending, one entry is lost (an undercount, never a wrong entry). | Main asks `task` `call_0` and lookup `call_1`; its helper asks `call_1`. | None: narrow. |
| 7 | note | `activity_log.py` | An answer returned as a tool call (langchain's `ToolStrategy` structured output) is listed as not started, unmarked. No page uses it (`grep response_format`, `with_structured_output`: only `report_ingest`'s own engine). | A made-up `Verdict` schema: listed. | None unless a page adopts it. |
| 8 | note | `deep/assembly.py` `_tool_label` | A tool given as a dict (a provider's built-in tool) is recorded as the dict's whole text. No helper spec has one today (both pages built and read). | `{"name": "x", "description": ...}` in a spec's tools. | None now. |
| 9 | note | `vision_tools.py` `_note_image_read` | The look's bytes are always hashed; `file_facts` gives a card over 32 MB its `mtime` and no `sha`. So for such a file the look's `sha` matches no card (its `mtime` does). CHECK's "the same sha as the card's" holds to 32 MB. | A 100 MB made-up file: `sha` on the look; `file_facts` returns `mtime` only. | None. |

No must-fix findings.

## 3. Paths checked

1. **Byte for byte.** Both fingerprint tests pass unchanged on the branch (Document Review's, venv a; the app's two pages, venv b: 13 passed, none skipped). The builder's with/without-record test passes. `_note_image_read` returns nothing and its whole body is under `except Exception`; `None`, text, an int, an object, a `bytearray`, a `memoryview`, a `None` key and a Windows path as input: no raise. Outside a run `dispatch_custom_event` raises inside and nothing is written. Holds.
2. **Parallel calls** (two in one step, both run; normal end): no key. Holds.
3. **A helper's calls against the main agent's** (a real deep agent on both deepagents, the helper's ids equal to the main agent's, `task` and a lookup in one step): no key; each `tool_start` under its own agent. Holds (finding 6 for the cut-off case).
4. **An id reused at every step** (`call_0` twice): no key. Holds. After fix 1, a reused id does not inherit an earlier refusal's mark (tested).
5. **No ids**: finding 5.
6. **A retried model step.** A retry inside one model call (`engine_retry`) ends once: one `model_end`, so one set of asked calls. A model step that errors has no `model_end`: nothing asked. A turn whose main model errors after its tools ran: no key (made-up turn). Holds.
7. **A guard's refusal, an unknown tool name**: finding 1, fixed.
8. **A turn that ends normally** with every call run: no key. Holds.
9. **The logger's life.** One `ActivityLogger` per turn in all three makers (`aec_document_review/turn_jobs.py:276`, `webapp/turn_jobs.py:226`, `review_eval/runner.py:490`), so nothing is carried from turn to turn or between testers; the two new tables are instance fields under the logger's own lock. No module-level table, no new thread, no new lock held across a call.
10. **`image_read`.** The resolver reads the file once; the note hashes those bytes in memory and makes one `stat` (spied: one open of the file). 100 MB: 0.09 s, no second copy (`bytes()` of `bytes` is the same object). An upload: `source: attachment`, no `mtime`. A page render or contact sheet saved as a file: `source: path`. A document handle: the handle's own name. A path outside the working folder is refused by the resolver before the note (`PathRefused` is a `FileNotFoundError`): no event. The name kept is `os.path.basename` of what the call asked for; `tool_start` already keeps the whole argument and `turn_files` the relative path, so nothing new is held. `render_region` and `analyze_pdf_page` write no event (the builder says so).
11. **Size per turn, worst case.** `helper_tools`: names only, once per `turn_start` and in `meta.json`: 430 characters on Document Review (2 helpers), 1,232 on the geotech page (5 helpers, 83 names). `image_read`: about 250 bytes per `analyze_image` call, so bounded by the turn's step cap. `calls_not_started`: at most 20 entries, each with its args uncut (finding 4): 20 times the largest argument one model reply can carry. In memory the logger holds the args of unstarted calls only, for the turn.
12. **Merge with `origin/claude/b13-integration`** (now `821ffc6`, 26 commits past the base). Tried in a scratch worktree, not pushed: ONE conflict, `activity_log.py` in `on_llm_end`: the integration branch added `if cache_reported: record["cache_reported"] = True` at the line where this branch adds the `with self._lock:` block that keeps the asked calls. Keep both, either order. `assembly.py`, `run_report.py` (rewritten there by some 800 lines) and `vision_tools.py` merge by themselves. With the conflict resolved that way, the three EV-20 test files and `test_ev17_run_report.py` pass on the merged tree (36 passed). The same single conflict for the reviewed head and for `claude/rv-ev-20`.
13. **Old installs.** The pins allow langchain-core down to 1.4.0 (through `langchain>=1.3`); 1.4.0 and 1.6.10 both hand `tool_call_id` to `on_tool_start` (read from the wheels). 1.0.0 did not: there every call would be listed. Not reachable under the pins.
14. **Helper inheritance.** "A helper with no `tools` inherits the primary's": true in deepagents 0.6.8 (`graph.py:670`) and 0.7.24 (`graph.py:787`). The record equals what the page hands deepagents for each helper (compared on Document Review's default build).

## 4. Tests

Venvs: (a) `/tmp/dr` Document Review and the core, deepagents 0.7.24, langchain 1.4.5, planlens 0.16.0; (a68) the same with `deepagents==0.6.8`; (b) `/tmp/gse` the core with the app.

| Command | Result |
|---|---|
| (a), (a68), (b): `pytest packages/aec-agent-core/tests/test_ev20_record_why.py` on `a0f5d99` | 12 passed, each |
| The same file on the base (`/tmp/base`, `PYTHONPATH` to its core) | 8 failed, 4 passed: the four CHECK calls guards. As CHECK says. |
| (a) and (a68): `tests/test_ev20_review_answered.py` (mine, 3 tests) | on `a0f5d99`: 3 failed; on `6ddeef5`: 3 passed |
| (a): `tests/test_ev20_review_report.py` (mine, 2 tests) | on `6ddeef5`: the first fails (three warning lines seen in a made-up report), the second fails on the missing name; on `339fd9f`: 2 passed |
| (b): `pytest -n 4 packages/aec-agent-core/tests` on `a0f5d99` | 623 passed |
| (a): Document Review, `pytest tests` on `339fd9f` | 1149 passed, 3 skipped |
| (b): every test file under `webapp/tests` and `funhouse_agent` naming `activity_log`, `record_build`, `geotech_build` or `vision_tools`, on `339fd9f` | 1621 passed, 13 skipped |
| (b): `test_page_fingerprints.py`, `test_public_text.py`, `test_review_freeze.py`, `test_backlog.py`, `test_wheels.py` | 44 passed, 5 skipped; fingerprints 13 passed alone |
| (a): `test_review_fingerprints.py`, `test_boundary.py`, `test_page_turn.py` | 12 passed |

Fingerprints: not moved, by the branch or by the fixes.

My own runs, not committed (`/tmp/rv`): `turns.py` (nine made-up turns on `create_agent` and on a real `create_deep_agent` with a helper, on both deepagents: the results in section 3); `report.py` (a made-up report, read: the count line "3 times" in the notes, and the three per-call lines); `img.py` (100 MB, the open count, odd inputs); `build.py` and `gbuild.py` (the real pages' `helper_tools`).

The builder's tests check behaviour (the record a real graph or a real tool call writes), except `test_the_run_report_counts_the_reads`, which only finds two pieces of text in the page.

## 5. Scope

Files changed by the branch: four core files, one new test file, CHECK.md. No existing test edited; no frozen file; no prompt, tool description or setting. Nothing fitted to a model. Record only, so no switch is owed: confirmed (section 3, path 1). The branch's name says EV-32; it holds only EV-20 work (EV-32, EV-31 and EV-16 are skipped with reasons in CHECK).

The review's fixes add two keys the builder did not have: `answered` on an entry and `n_calls_not_answered` on `turn_end`, and one name in `run_report.py` (`PLAIN_EVENTS`). No reader of `turn_end` in the repository breaks on them (`budget_panel.py`, `review_eval/runner.py` read other keys).

## 6. CHECK.md

- "the same `sha` as the card's file in the turn's `turn_files`": true up to 32 MB only (finding 9).
- "`run_report.py`: ... counted in the run's notes rather than listed once per look": it was also listed once per look, in each call's view, as a warning (finding 2).
- "`calls_not_started` 2 of 2 turns with a call left over, 0 of 2 turns whose calls all ran": true of those four turns; a turn that finished with one refusal carried the key too (finding 1). "For the lead ... a turn stopped at its step cap has `calls_not_started` naming the call in the last `model_end`'s `tool_names`" did not hold after 20 refusals; it does after `6ddeef5` (the first entry).
- No "Release note" section. It would be "None": nothing a tester or a GSE user sees changes.
- The new event and the new `turn_end` keys are not in `activity_log.py`'s module docstring, which lists the others. Not fixed (notes only).

## 7. Could not check

- A live host or model. On the next suite or live run: (1) a turn stopped at its step cap has `n_calls_not_answered` >= 1 and its first `calls_not_started` entry is in the last `model_end`'s `tool_names`; (2) a finished turn has `n_calls_not_answered` 0 or no key; (3) a figure redrawn between two looks shows two `image_read` events with different `sha`, one equal to the card's `sha` in `turn_files`.
- Whether each hosted model client gives every tool call an id (finding 5): read one `messages.json` per client.
- The full eight CI jobs: the lead pushes `ci/<name>`; I pushed none.
