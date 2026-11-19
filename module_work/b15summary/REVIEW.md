MERGE AFTER FIXES, with the one fix on this branch (`claude/rv-b15-summary-history`). Reviewed head bd9bb02; before = 9367934. Nothing found that loses or mixes a conversation's history; the fix closes a case no supported library reaches today.

# Review of B15: a failing summary leaves nothing in the history files

Every number below is from my own conversations with a fake model (made-up sheets X-101, X-102; trigger 4 messages, keep 2), run before and after on four pairs: deepagents/langchain 0.6.8/1.3.14, 0.6.8/1.3.18, 0.7.0/1.4.5, 0.7.24/1.4.5 (what installs today), sync and async.

## Findings

| # | What | Made-up input | Before (9367934) | After (bd9bb02) | Status |
|---|---|---|---|---|---|
| 1 | Growth gone | three failing turns of one conversation, `FilesystemBackend` | files, bytes after turns 1, 2, 3. 0.6.8: 1 file, 160, 387, 654. 0.7.0: 1 file, 235, 587, 1029. 0.7.24: 1, 2, 3 files, 235, 587, 1029 | 0 files, 0 bytes after each turn, all four pairs, sync and async; the model was handed the whole history each turn | holds |
| 2 | A stored summary, then two failing turns | summary succeeds, then the provider refuses | file grows (0.7.x: 235 to 654 bytes) | file byte for byte as it was after both turns; stored summary unchanged | holds |
| 3 | A failure, then a success | refused, then answered | file holds each older message twice (the failed turn's copy and the success's) | one file, each older message once: the success's section (0.7.x: 352 bytes = 587 less the 235-byte copy) | holds |
| 4 | Two conversations on ONE agent at one moment, one refused | `conv-ok` and `conv-bad`, summary call 0.2 s, two threads; again as two tasks of one event loop | 2 files (the failing one left its copy) | 1 file, `conv-ok`'s only; its stored summary is `SUM[conv-ok]`; `conv-bad` has no file, no stored summary and was handed its whole history; neither model call saw the other's text; one `summary_failed` | holds: the blocking question, no loss, no mix |
| 5 | A summary that succeeds | one turn | 1 summary model call; one file | the same file bytes (time stamp and, on 0.7.24, the random file id set aside), the same stored summary and keys (`cutoff_index`, `file_path`, `summary_message`), no event, `summaries_skipped` 0, exactly 1 summary model call; all pairs, sync and async | holds |
| 6 | Kinds of failure, in a turn of two model calls | 429, `TimeoutError`, empty reply, the async turn cancelled 0.15 s into a 0.5 s summary call | 429 and timeout leave a file; cancel leaves a file | 429 and timeout: no file, one `summary_failed` (with `error_type` on langchain 1.3.15 and newer), `summaries_skipped` 1, as the 1.3.0 note says. Cancel: no file, no task left running, no "exception was never retrieved", and the next turn summarizes and writes its file | holds |
| 7 | An empty reply is not a failure | the summary model answers "" | stored as the summary, file written, no event | the same | noted, not fixed: unchanged by B15 and outside it, but the older work is then replaced by an empty summary; the file keeps it |
| 8 | The remembered summary was matched by the list's identity alone | under the guard: ask for `[a, b, c]`, append `d` to the same list, ask again | (no remembering) | the second request is answered `summary of 3`, not of 4 | FIXED here (commit "A summary is remembered only..."): the entry also keeps what the list held; a changed list is summarized again. Not reachable on 0.6.8 to 0.7.24: both steps get one untouched list |
| 9 | An id reused by a second list | six lists made and dropped one after another in one pass | | six summaries, each its own: the entry holds the list, so its id cannot be reused | holds (test added) |
| 10 | No test named the private parts | a library that renames `_offload_to_backend` | the builder's turn tests would fail only as "a file was left" | new test fails by name on a rename, a changed argument order, or a wrap that hands the write and the summary different lists | test added |
| 11 | A library without the two writes | a summarizer class with only the five required methods, its summary refused | | guard installed, turn goes on with the history as it was, one `summary_failed` | holds (test added) |
| 12 | The core's own backend | `build_summarization_middleware` over `build_memory_backend` (`CompositeBackend` over `StateBackend`), two failing turns | 1 then 2 history files in the state | 0 and 0; succeeding turns identical before and after (1 file, 384 then 562 characters) | holds |
| 13 | One whole turn of the default Document Review page | build `review_default` from `tests/page_builds.py`, summarizers forced to 4 messages, two failing turns | 1 then 2 history files in the state | 0 and 0; `done` both turns, one `summary_failed` a turn; succeeding turns identical | holds |

## What I ran (exit codes)

- My probes (`/tmp/rv/probe.py`, not committed), before and after, four pairs: exit 0, eight runs. After the fix, on 0.7.24/1.4.5 and 0.6.8/1.3.14: identical to the reviewed head.
- `test_b15_summary_leaves_nothing.py test_ta24_summary_guard.py test_ta24_summary_install.py test_ta24_summary_turn.py` (every test file that names the guard) on the reviewed head: 37 passed, exit 0, on each of the four pairs. With the fix and the new file: 46 passed, exit 0, four times.
- The builder's new file on 9367934 (0.7.24/1.4.5): 4 failed, 2 passed, exit 1, as CHECK.md says.
- `test_rv_b15_summary_history.py` on the reviewed head: 2 failed (the changed list, sync and async), 7 passed, exit 1; with the fix: 9 passed, exit 0, four pairs.
- Document Review, `tests/test_review_fingerprints.py tests/test_boundary.py`: 9 passed, exit 0, before and after the fix. Fingerprints did not move.
- `webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` (`--noconftest`): 13 passed, exit 0.

## CHECK.md

Its claims held where I ran them (the table's file counts match mine in shape; my byte counts differ only because my messages do). One thing to know that it does not say (read from the code, not run): Document Review's minimal build makes its summarizer from langchain's own class (`review_agent._summarizer`), which writes no history file and is not the class the guard wraps, so B15 changes nothing there, before or after.

## Not checked

- A real provider, a real page session through `core.stream_turn` with a checkpointer (finding 13 called `agent.invoke` and carried the state by hand), Tiny Apps.
- Inline images on deepagents 0.7 (the builder's one case left as it is): not re-run.
- `webapp/tests/test_page_fingerprints.py` (needs the geotech venv): not run; no prompt, tool or setting changed.
- A turn cancelled during the file write itself, after the summary: same code path as before the change, not timed.
- The full suites: for the lead to run as `ci/...`.
