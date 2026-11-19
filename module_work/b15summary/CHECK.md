# B15: a summary that fails leaves nothing in the history files

Branch `claude/b15-summary-history`, from `claude/b13-integration` at 9367934 (1.3.0). Repair and tests: 96b2358. No BACKLOG id was given; the limit is the release note's line "A failing summary still leaves one copy of the older history in the conversation's history files, once per failing turn".

## Confirmed on the head, before any change

- The copy is written by the agent library, not by the guard: `deepagents/middleware/summarization.py`, `SummarizationMiddleware._offload_to_backend` (called from `wrap_model_call` on the line before `_create_summary`) and `_aoffload_to_backend` (in `awrap_model_call`, in one `asyncio.gather` with `_acreate_summary`). The guard saw the failure only afterwards.
- deepagents 0.7.24 names the file by an id it stores only with a summary (`_get_session_id`), so each failing turn leaves a NEW file. deepagents 0.6.8 and 0.7.0 name it by the thread (`_get_thread_id`), so each failing turn adds a section to one file.
- Once per failing turn and not per model call, as the note says: the guard's "not again in the same turn" already held that.

## What changed (a repair, no setting)

`packages/aec-agent-core/aec_agent_core/deep/summary_guard.py` only:
- `_guard_offload`, `_guard_aoffload` (new) wrap the library's two history writes. Under the guard they ask for the summary of the same messages first; a failure raises `SummaryFailed` before anything is written. Nothing is stored, so nothing is removed.
- `_once` (new) around the guarded `_create_summary` / `_acreate_summary`: within one pass of the library's wrap, a second request for the same messages list is handed the first one's summary (or, async, its failure). So the library's own request costs no second model call.
- `_on`, `_off` (new) set and reset the guard's mark together with that pass's dict (`_made`); the four `_guarding.set` / `.reset` pairs in `_guard_wrap` and `_guard_awrap` use them.
- The two writes are in `_OPTIONAL`: a library without them is guarded as on the head, and a failed summary then leaves its copy.
- About 60 lines of code; the rest of the diff is the module docstring and comments.

## Library pairs run (fake model, `FilesystemBackend`, three failing turns of one conversation; files, bytes)

| deepagents / langchain | A failure is | Head, after turns 1, 2, 3 | Branch |
|---|---|---|---|
| 0.6.8 / 1.3.14 | the error line | 1 file: 147, 356, 598 bytes | 0 files, 0 bytes |
| 0.6.8 / 1.3.18 | raised | 1 file: 147, 356, 598 | 0, 0 |
| 0.7.0 / 1.4.5 | raised | 1 file: 222, 556, 973 | 0, 0 |
| 0.7.24 / 1.4.5 (what installs today) | raised | 1, 2, 3 files: 222, 556, 973 | 0, 0 |

Sync and async give the same counts on the branch. After a stored summary, three more failing turns leave the one existing file byte for byte as it was (on the head a section, or a file, was added each turn).

**Left as it is, one case:** inline images in the older history, deepagents 0.7 only. `_offload_inline_media` / `_aoffload_inline_media` (same file) upload each image to `conversation_history/media/<sha256[:16]>.<ext>` before the write and before the summary, and the summary is made from their output, so the upload cannot be put after it without rewriting the library's wrap. Run on 0.7.24: one made-up image, three failing turns: one media file of 148 bytes, the same file after each turn (it does not grow per turn); no history file. 0.6.8 uploads nothing.

## A summary that succeeds

Same summary, file, state (`_summarization_event`: `cutoff_index`, `file_path`, `summary_message`) and record; one summary model call. What differs is order only: the summary model call now comes before the file write (sync), and the write starts when the summary is in instead of beside it (async). If the write fails, the library's own warning and `file_path=None` are as before.

## Tests

New, `packages/aec-agent-core/tests/test_b15_summary_leaves_nothing.py` (each case sync and async):
- `test_three_failing_turns_leave_no_history_file`: fails on the head on all four pairs (the table's numbers).
- `test_a_failing_turn_after_a_summary_adds_nothing`: fails on the head on all four pairs.
- `test_a_summary_that_succeeds_is_made_once_and_stored_as_before`: passes on the head and on the branch, on purpose: it is the proof that a succeeding turn did not move.

No existing test was edited. Runs (exit code 0 each):
- Each of the four pairs, `python -m pytest -q -p no:cacheprovider` on the new file and `test_ta24_summary_guard.py`, `test_ta24_summary_install.py`, `test_ta24_summary_turn.py` (every file `git grep -l -e summary_guard -e summary_failed -- packages` finds under tests): 37 passed, 0 failed, four times.
- Document Review (venv a), `tests/test_review_fingerprints.py tests/test_boundary.py`: 9 passed.
- `webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` (`--noconftest`): 13 passed.
- `webapp/tests/test_page_fingerprints.py` (venv b): 13 passed.

Fingerprints: not moved. No prompt, tool description or setting changed; `webapp/` and `funhouse_agent/` untouched.

## Calls made

- Prevent, not remove: deepagents 0.6.8 backends have no `delete`, so removal could not serve every pair; making the summary first does.
- The summary is remembered by the identity of the messages list, per pass. Both library generations hand the same list to the write and to the summary; a library that hands two different lists would make two summary calls on a succeeding turn (and still write nothing on a failing one).
- Async: the summary runs as one task that both of the library's requests await.

## Release note

A summary that fails no longer leaves a copy of the older history in the conversation's history files; the known limit of 1.3.0 is gone, except that on deepagents 0.7 an inline image of the older history is still stored once.

## For the lead

- Not run on a real provider or in a page turn (fake model only), and only on `FilesystemBackend`, not on the backend the core's assembly builds (`deep/assembly.py`: a `CompositeBackend` over `StateBackend`): the write is skipped before any backend is reached, so the backend should not matter, but it was not run.
- `module_work/aec_1_3_0/RELEASE_NOTE.md` line 113 and the turn test's docstring ("no new history file") still describe the old limit; not edited (the note belongs to the release).

## Noted, not fixed

- The compact tool (`SummarizationToolMiddleware._run_compact`) is outside the guard, as on the head; on 0.7.24 it makes the summary before its write by itself.
