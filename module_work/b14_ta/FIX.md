# Fix: TA-24's summarizer repair, after the review (`claude/fx-ta-24`)

From `origin/claude/rv-ta-24` at `a465382`. Four commits of change, then
this file alone. The lead's decision on the review's first finding (its
option 1) and the review's four notes.

| Commit | What |
|---|---|
| `2b04f10` | A failed summary is not tried again in the same turn |
| `b87cbed` | `install()` under a module lock |
| `1cab279` | One warning when the guard is absent because the library's shape is unknown |
| `ecc8933` | The docstring and CHECK.md name langchain's real boundary; CHECK.md's "part twice" made true |

No fingerprint file is in the diff, no existing test is edited, no frozen
file, no prompt, no tool wording, no setting. Made-up values only.

## 1. Not again in the same turn (`2b04f10`)

**Before.** While a summary kept failing, EVERY model call of the turn
ran the library's whole summary path first: the older history written to a
file again, three summary requests with 3 to 5 s of back-off (langchain
1.3.15 and newer), then the call asked again with the history whole.

**Now.** `summary_guard._guard_wrap` / `_guard_awrap`: after a summary
fails, the later model calls of that turn go straight to the pass the
guard already had for the failing call (the library's own wrap with its
two triggers saying "no"). The model is handed the history as it is, with
no summary request, no back-off and no new history file. The next turn
tries again.

**The turn boundary: the core's own `TurnScope`** (`deep/keep_work.py`),
the scope a host opens around one turn (`turn_scope()`, in a context
variable that LangGraph copies into every thread and task of the run; it
is where a turn's image calls and busy waits are already counted). It
gained `summaries_failed`, `note_summary_failed`, `summary_failed`,
`summaries_skipped` and `note_summary_skipped`, under its lock. It serves
a helper's own calls too: a helper runs inside the main agent's turn and
sees the same scope, so nothing had to be scoped by the main agent's turn
instead.

**The scope of the mark.**
- Per conversation: each turn of each conversation has its own
  `TurnScope`, so two conversations on one agent at the same moment stay
  apart, sync and async. Nothing is kept on the summarizer or its class.
- Per agent: the key is the summarizer instance (the main agent's, each
  helper type's: `create_deep_agent` builds one each) together with the
  running helper's record (`keep_work.current_helper()`, set by `KeepWork`
  around each `task` call; `None` for the main agent). So two runs of ONE
  helper type in a turn are apart as well, where `KeepWork` is in the
  stack (both pages).

**The record.** `summary_failed` once per failure, as before. For a call
that skipped its summary I chose **one count on `turn_end`**,
`summaries_skipped`, written only when it is above zero
(`keep_work.summary_facts`, read in `ActivityLogger.turn_end` beside the
image-call facts). Why not nothing: a trace reader who sees one
`summary_failed` and then ten long unsummarized calls has to infer the
rule from the code; the count says it, and it is the number that tells the
lead how much of a turn ran long because of one failure. Why not an event
per call: it would be one line per model call for the rest of the turn,
saying nothing the count does not. Only calls that were over the threshold
are counted (the guard asks the library's own trigger and keeps the
answer), not every later call.

**Calls made.**
- Outside a turn scope (a library call, a test that opens none) there is
  no turn to remember a failure in: every call tries, as before the fix.
  Both pages open a scope around each turn (`core._turn_scope`).
  Document Review's falls back to no scope only if the core cannot be
  imported.
- The library's own exception stays: a call the MODEL refuses as too long
  is summarized regardless of the triggers, in a later call of the turn
  too. It is the only way that call can be answered, it needs a second
  failure of a different kind, and it is what the failing call's own
  second pass already did. If that summary fails as well the call ends on
  the error, as before, with no second `summary_failed`.
- A summary that succeeds in that way does not clear the turn's mark; the
  conversation is under the threshold after it in any case.

**A summary that succeeds** runs the library's code as before: the first
pass is unchanged, and the existing comparison
(`test_successful_summary_passes_through_untouched`) passes on all four
pairs.

**Made-up example** (pair A, four model calls of one turn, summary always
refused): before, by the review's measures, 12 summary requests, 4
history files and 13 to 20 s of back-off; now 3 summary requests (the
first call's), 1 file, one call's back-off, and `summaries_skipped: 3` on
`turn_end` (the requests, files and count are asserted by the new test;
the seconds are not timed here).

## 2. `install()` under a lock (`b87cbed`)

`_INSTALL_LOCK` covers the mark check and the patch loop. The test forces
the race the review read from the code (a metaclass pauses the first
method patch until a second `install` has run): without the lock every
method is wrapped twice.

## 3. An absent guard says so once (`1cab279`)

`_warn_absent`: one `logging` warning per process (logger
`aec_agent_core.deep.summary_guard`), naming the class that could not be
imported or the methods it lacks. Without deepagents nothing is said.
`install()` still returns False and `compile_agent` still builds.

## 4. The wording (`ecc8933`)

- Docstring and CHECK.md: "langchain 1.3.14 and older" (the error line)
  against "1.3.15 and newer" (raises), and that deepagents 0.6.8 runs on
  either side.
- CHECK.md's "can hold a part twice" now says what the code does after
  change 1, measured: once per TURN whose summary fails. Two failing turns
  (four and three model calls) and then a turn whose summary is made:
  3 files with one copy each on pair A, 1 file with 3 copies on pair D.
  The copy per failing turn remains; removing it is the review's option 2.

## Tests

New files: `packages/aec-agent-core/tests/test_ta24_summary_turn.py`
(9 tests) and `test_ta24_summary_install.py` (6 tests). The real library
with a fake model: deepagents' own summarizer in a langchain agent, and
`create_deep_agent` for the helper (the library's class, built with a low
trigger so short made-up messages reach it).

| Test | On the starting head |
|---|---|
| `test_failed_summary_is_not_tried_again_in_the_turn` (four calls of one turn: asked at the first only, at most one file with one copy, every call whole, one `summary_failed`; the next turn asks again; a third turn's summary is stored) | fails |
| the same, async | fails |
| `test_a_helpers_failed_summary_does_not_stop_the_main_agents` | fails (the helper asks again at its fourth call) |
| `test_the_main_agents_failed_summary_does_not_stop_a_helpers` | fails |
| `test_two_runs_of_one_helper_are_apart` | fails |
| `test_two_conversations_at_once_threads`, `..._async` (the second conversation starts after the first one's failure, while its turn still runs) | fail (the failing conversation asks again) |
| `test_turn_end_counts_the_calls_not_summarized` (the real `ActivityLogger`) | fails |
| `test_outside_a_turn_every_call_tries` | passes: it pins what did not change |
| `test_two_installs_at_one_moment_wrap_once` | fails before `b87cbed` |
| `test_a_missing_method_is_said_once`, `test_a_moved_class_is_said_and_the_app_still_builds`, `test_no_library_is_not_a_warning` | fail before `1cab279` |
| `test_a_known_class_says_nothing` | passes: no warning where none is due |
| `test_the_docstring_names_langchains_real_boundary` | fails before `ecc8933` |

"Fails" was run, not read: the new files in a worktree of the commit
before each change.

Runs (Python 3.12, from the repository root unless said; exit code 0 each):

| Pair | deepagents | langchain | `python -m pytest -q -p no:cacheprovider packages/aec-agent-core/tests` |
|---|---|---|---|
| A | 0.7.24 | 1.4.5 | 642 passed, 0 failed |
| B | 0.7.13 | 1.3.18 | 642 passed, 0 failed |
| C | 0.6.8 | 1.3.18 | 642 passed, 0 failed |
| D | 0.6.8 | 1.3.4 | 642 passed, 0 failed |

(langgraph 1.2.22, langchain-core 1.8.3, planlens 0.16.0 on all four.)

- The three TA-24 files alone, on A and D: 31 passed (16 + 9 + 6).
- Document Review, from `packages/aec-document-review`, pair A:
  `tests/test_review_fingerprints.py tests/test_boundary.py`: 9 passed.
- The guards, before each commit:
  `--noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`:
  13 passed.

## Could not check

- GSE's `webapp/tests/test_page_fingerprints.py` and the app's tests that
  reach `keep_work` and `activity_log` through the old module names
  (`funhouse_agent/**/tests`, `webapp/tests`): this session had no
  reference submodule and no geotech venv. The change adds no middleware,
  tool, prompt or budget, so no hash should move; read it from the `ci/`
  run.
- `KeepWork` in the stack with a real `task` call: the two-runs-of-one-
  helper case is tested on the shared shape with two helper records, not
  through a page build.
- A real provider and a real page turn: `summaries_skipped` on a tester's
  `turn_end`, and the file count under `/conversation_history/` in the
  conversation's state (the page's default backend keeps them there, not
  on disk as in these tests).
- Merging with `claude/b13-integration`: the review found one conflict in
  `activity_log.py` `APP_EVENTS`; this branch adds two comment lines under
  that name and a second `try` block in `turn_end` beside the image-call
  facts, which may meet the same branch.

## Noted, not fixed

- The existing test file's own docstring (`test_ta24_summary_guard.py`)
  still says "langchain before 1.4"; no existing test is edited here.
- The warning is once per process whatever was missing, so a second,
  different absence in the same process is not said.
