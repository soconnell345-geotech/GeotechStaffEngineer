# Review: TA-24's summarizer repair (`claude/b14-tinyapps-repairs`)

**Verdict: MERGE**, reviewed head `f4d25aa` (commits `096b3cb` and `f4d25aa`, `git diff 4d9ace9..f4d25aa`).
No must-fix finding. One should-fix needs a design decision and is left; so this branch carries no fix
commit, only this file.

Versions run (Python 3.12, the real libraries, a fake model, made-up messages):

| Name | deepagents | langchain | langgraph | A failed summary, unguarded |
|---|---|---|---|---|
| A | 0.7.24 | 1.4.5 | 1.2.22 | raises after 3 tries |
| B | 0.7.13 | 1.3.18 | 1.2.22 | raises after 3 tries |
| C | 0.6.8 | 1.3.18 | 1.2.22 | raises after 3 tries |
| D | 0.6.8 | 1.3.4 | 1.2.22 | the error line becomes the summary |

## Findings

| Severity | Where | What is wrong | The input that shows it | Smallest fix |
|---|---|---|---|---|
| should-fix, **needs a design decision, not fixed** | `deep/summary_guard.py` `_guard_wrap` / `_guard_awrap` (the library writes the history file before it makes the summary) | While the summary keeps failing, EVERY model call writes the older history again into the conversation's files (the page's default backend keeps them in the conversation state, so each checkpoint carries them). Pair A: one new orphan file per call (`/conversation_history/session_<new id>.md`; the id is not kept because the state update is dropped). Pair D: one file holding one more copy per call. CHECK.md says "can hold a part twice"; it is once per model call. | Four calls of one conversation over the threshold, summary always refused: A ends with 4 files, D with one file holding the first exchange 4 times; with a summary that succeeds, 1 file and 1 copy on both. | A choice for the lead: (1) after a failed summary, do not try again for the rest of the turn (changes "tries again on its next call"); (2) hold the history write until the summary is made (two more private methods patched, and 0.7's async path runs the two together); (3) accept it: it needs a failure that lasts, and ends with the turn. |
| note | `summary_guard.py` module docstring; CHECK.md "How each library generation summarizes" | The boundary is not langchain 1.4. langchain 1.3.14 and older return the error line; 1.3.15 and newer raise (wheels 1.3.4 to 1.3.18 read). deepagents 0.6.8 installed today resolves langchain 1.3.18 and raises. The guard reads the behaviour, not the version, so nothing is wrong at run time; the prose is. | Pair C: unguarded, the call ends on the refusal, no error line anywhere. | Reword "before 1.4" to "1.3.14 and older" in the docstring and CHECK.md. |
| note | `summary_guard.py` `_checked`, `failed_summary` | The prefix test runs on every version, also where the library can no longer return the line. A real summary that begins with exactly `Error generating summary:` is treated as failed. The outcome is safe (history kept, tried again next call), and such a summary is not worth keeping. | A summary model that answers `Error generating summary: ...` on pair A: `summary_failed` with no `error_type`, nothing stored. | None needed. |
| note | `_guard_wrap`, by design | A failure that lasts costs, on every later model call: 3 summary requests (the library's own retry) and 3 to 5 s of its back-off (measured 3.3 to 4.9 s on A, B, C; none on D), before the main call. On a rate-limited deployment that is 3 extra requests per call. Not a loop: bounded per call. | Always-refused summary, two turns: 6 summary calls, both turns answered. | Goes with the decision above (option 1 removes it). |
| note | `summary_guard.py` `install` | No lock: two first builds at the same moment can wrap twice. The effect is small (the record's `error_type` reads `SummaryFailed`, and a second failure ends on `SummaryFailed`, not the provider's error). Read from the code, not reproduced. | Two testers' first turns in the same instant after a restart. | A module lock around the check and the loop. |
| note | `install`, `compile_agent` | When the class or one of the five methods is missing, `install()` returns False, `compile_agent` ignores it and nothing is written anywhere: the guard is silently absent. The app starts, which is right; the tests catch it (below). | A class without `_should_summarize`: False, class untouched. | One log line when deepagents imports but the shape is unknown, if the lead wants it. |

## The eight questions

1. **Pair A end to end, on the real library** (`create_deep_agent`, the library's own class with a low
   trigger so short made-up messages reach it; sync and async; the main agent, the general-purpose
   helper, a named helper). Fails once: the model is handed the 7 messages as they were, nothing
   stored, one `summary_failed`; the next call is summarized and stored. Always fails: every call is
   handed the whole history, nothing stored. Succeeds: summarized as the library does. Both helpers the
   same (calls of 7 and 9 messages unsummarized when it fails; 7 then 3 when it fails once), and the
   main agent's state is untouched by a helper's failure. Same results on B, C and D. The record was
   read through the real `ActivityLogger`, not a patch: `{"agent": "primary", "event":
   "summary_failed", "history": "kept", "what": ..., "error_type": "Busy"}` (no `error_type` on D).
2. **The private class moving.** Class or a needed method gone: `install()` is False, nothing is
   patched, the app starts, the library behaves as unguarded. The error text gone: only pair D's kind
   depends on it, and newer langchain raises instead. `wrap_model_call` no longer calling
   `self._create_summary`: the guard does nothing and summaries still work. No case found where the
   guard breaks a summary. A test fails in each case: `_agent()` asserts `install()` is true and the
   end-to-end tests assert the history is handed whole, on whichever pair CI installs. Nothing in
   production says the guard is absent (finding, note).
3. **A successful summary.** Guarded against unguarded (`__wrapped__` restored on all six methods),
   two turns, sync and async, on A, B, C, D: the messages handed to the model, the stored event
   (cutoff, summary message and its `additional_kwargs`, file path) and the history files are
   identical once the random session id is masked.
4. **The prefix.** See the finding: safe, and applied on all versions.
5. **Threads and nesting.** The mark is a `contextvars` variable holding the instance, set and reset
   inside one call: per call, not on the instance or the class. Two conversations on ONE agent at the
   same moment, one whose summary fails slowly and one whose summary succeeds (two threads; and two
   `ainvoke` under one `gather`): the second is summarized and stored, the first keeps its history.
   A helper runs in the tool node, outside the main agent's model call, so the two never nest; two
   summarizers nested in one model call (`enable_summarization`, off by default) are kept apart by
   `SummaryFailed.source` (read from the code).
6. **Then too long for the model.** Always-refused summary and a model that refuses the 7 messages as
   too long: 3 summary tries, ONE main call (refused), 3 more summary tries, then the turn ends. On A,
   B, C it ends on the provider's own error (`Busy`, raised bare, so the page's busy and budget
   wording applies as before the guard). On D it ends on `SummaryFailed`: "The earlier part of the
   conversation could not be summarized; it is kept as it was." (the library swallowed the provider's
   error). Nothing is stored; the history is kept; the tester can send again. It cannot loop: at most
   two passes of the library's wrap per model call. When the second summary succeeds: B, C, D answer
   from the summary; A raised the library's own `ContextOverflowError` "still does not fit after
   recovery" because my made-up summary was longer than the seven short messages (the library's
   check, the same without the guard).
7. **A repair with no switch?** On langchain 1.3.14 and older, yes, plainly: the earlier work was
   lost without a word and a wrong line was handed to the model for the rest of the conversation. On
   every newer pair, and that is what a fresh host resolves today (the wrapper pins no deepagents:
   0.7.24 with langchain 1.4.x), it is not a repair by COMMON_DOCREVIEW's four cases: before, the turn
   ended on the provider's error with the history intact; now the turn goes on. That is a change of
   behaviour testers can meet, with no switch. It is what the lead asked for, the page and both
   fingerprints do not move, and the new behaviour is the kinder one; the lead should know it is a
   decision made here, not a defect repaired, and it brings the cost in the first and fourth findings.
8. **Merging with `origin/claude/b13-integration`** (`ce970b5`), scratch worktree, nothing pushed: ONE
   conflict, `activity_log.py` `APP_EVENTS` (this branch adds `"summary_failed"`, that one
   `"image_read"`, on the same line). Keep both names and this branch's three comment lines.
   `assembly.py` merges by itself. With that resolution the 16 tests pass on pair A.

## Paths checked

| Path | Holds |
|---|---|
| Main agent, sync and async, fails once / always / succeeds (A, B, C, D) | yes |
| General-purpose helper and a named helper, the same three (A, B, C, D) | yes |
| Two conversations at once on one agent, threads and asyncio | yes |
| Success identical to the library's own, with files | yes |
| Too long after the guarded call | yes, ends, no loop |
| The record through the real logger | yes; the provider's text is not in `summary_failed` (it is in the summary call's own `model_error`, as before this branch) |
| A stop asked during the summary | unchanged: the stop is checked in the app's middleware, not inside the summary call |
| A persistent failure's writes | no: see the first finding |
| The compact-conversation tool | not in the stacks; unchanged, as CHECK.md says |
| Readers of the new names (`summary_guard`, `summary_failed`), whole repository | only `assembly.py` and `activity_log.py` |

## Tests

- `python -m pytest -q -p no:cacheprovider packages/aec-agent-core/tests/test_ta24_summary_guard.py`
  from the root: 16 passed, 0 failed on each of A, B, C, D.
- On the base (`4d9ace9`, a worktree, the new file copied in): 1 error at collection (the module does
  not exist), so all 16 fail there, by construction. The defect itself is shown inside the branch by
  `test_without_the_guard_the_library_loses_history_or_turn`.
- The tests check behaviour (what the model was handed, what was stored). Their record check patches
  `_event`; I read the real record separately (question 1).
- Document Review's `tests/test_review_fingerprints.py` and `tests/test_boundary.py` on pair A:
  9 passed. `webapp/tests/test_public_text.py` and `test_review_freeze.py`: 13 passed. Neither
  fingerprint file is in the diff.
- My own scripts, kept out of the repository: the end-to-end probe (questions 1, 3, 5, 6), the record
  read, the file counts (first finding), the merge trial.

## Scope

Five files: the new module, its new test file, CHECK.md, three lines in `compile_agent`, one name and
three comment lines in `activity_log.py`. No existing test edited, no frozen file, no prompt, no tool
wording, no setting. Made-up values only.

## CHECK.md

- "langchain before 1.4" / "deepagents 0.6.8 with langchain 1.3": true only up to langchain 1.3.14.
- "that file can hold a part twice": once per model call while the failure lasts; on 0.7 a new file
  each time.
- "The installed generation runs for real here (0.6.8) and on GitHub (0.7.24)": true; I add 0.7.13.
- "A repair, no switch": see question 7.
- The rest I could support by running.

## Could not check

- GSE's `webapp/tests/test_page_fingerprints.py`: it needs the geotech app and the reference
  submodule. Run it from a full checkout, or read it from the `ci/` run.
- A real provider: whether the testers' gateway refuses the summary call with a 429 that clears
  within the library's three tries, and what a tester sees on the page when the turn ends on it. A
  long Document Review conversation on a busy deployment, then read `activity.jsonl` for
  `summary_failed` beside the summary call's `model_error` entries, and count the files under
  `/conversation_history/` in the conversation's state.
