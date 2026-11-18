# Tiny Apps repairs, third wave (BACKLOG TA-18, TA-06, TA-24, TA-13, TA-23, TA-10)

Six Tiny Apps items were read against the code. One is built; five are
left, each with the reason and what a live check would need to capture.

## TA-10, built: a re-imported figure keeps its chart

**What the item asked.** When a figure's interactive sidecar
(`plot.plotly.json`) is copied into a conversation folder that already holds
a different file of that name, the copy was named `plot.plotly_1.json`. That
name no longer ends in `.plotly.json`, so it showed as a plain file card,
and it no longer paired with its picture (`plot_1.png`): the chat showed two
cards where one chart belongs. The item names the app's copy of the page
(`webapp/core.py`); Document Review has its own copy of the same helper.

**What was built.** `aec_document_review/core.py`, `_unique_dest`: a name
ending in `.plotly.json` (any case) keeps that two-part suffix and takes the
number before it (`plot_1.plotly.json`, then `plot_2.plotly.json`). Every
other name is split at its last dot as before. Its three callers in Document
Review gain the repair: `import_external_artifacts`,
`import_reported_outputs` (the same `core.py`) and the SharePoint download in
`sharepoint_tools.py`.

**Judgement calls.**
- Only the one two-part suffix the page reads by name (`PLOTLY_SIDECAR_SUFFIX`)
  is kept whole; no general list of compound suffixes, since no other one
  changes how a file is shown.
- The app's own copy (`webapp/core.py`) is left as it is: the app is quiet in
  this repository and its keeper is the maintainer.

**Settings.** None: the repair of a defect. A free name, and every name not
ending in `.plotly.json`, come out exactly as before.

**Tests** (`packages/aec-document-review/tests/test_ta10_plotly_sidecar_name.py`,
made-up names and contents):
- a taken sidecar name becomes `plot_1.plotly.json`, then `plot_2.plotly.json`,
  and classifies as a chart;
- the suffix is matched in any case;
- free names, pictures, plain `.json` files, names with no suffix and a file
  named `plotly.json` come out as before;
- a figure re-made under names the folder already holds is copied as
  `plot_1.plotly.json` and `plot_1.png`, makes ONE chart card, and the older
  files are not overwritten.

**Noted, not fixed.** When only one of a figure's two files collides (the
picture is taken but the sidecar's name is free), the two copies get
different stems and do not pair; it needs a figure-level rename, not a
file-level one.

## Left for a live check or a decision

- **TA-18** (rebuild the client when a 401 survives the retry). Parked, and
  the reason holds: no refusal has been seen to survive the one retry. The
  live check must capture a 401 that comes back on the retry, with the time
  between the two tries and whether a new client then succeeds.
- **TA-06** (pin the budget-spent response in a test). It waits on TA-42's
  check (2): the generic budget wordings are already tested, but no answer
  of the testers' gateway to a spent budget has been read. The live check
  must capture the HTTP status, the error body (code, type, message) and the
  headers of that answer; the test then pins that body to the budget line
  with no retry.
- **TA-24** (the busy ladder's leftovers). The waits, tries and the tokens-
  per-minute setting need the live check the item names. The summarizer part
  is not a small repair: the summarizer that runs is the agent library's
  built-in one, built inside its agent constructor with no hook, and (in the
  library version installed now) it runs inside the model call rather than
  before it. Its own model call sits outside the ladder, and the library
  catches every error of that call and uses the error text as the summary,
  so a busy refusal there does not end the turn; it replaces the older
  history with an error line (the history itself is written to a file the
  summary names). Putting the ladder around it means replacing the built-in
  summarizer in its place in the stack: a design step for the shared core,
  not an afternoon repair.
- **TA-13** (metering on the backup host). Parked, and the reason holds:
  log labels on the backup host; the budget decorator belongs to the backup
  host's own client library, and the user attribution lives in the app's
  launcher (`webapp/`), which is quiet here.
- **TA-23** (the primary waits for its slowest helper). Parked as a design
  question; no small repair is plain.

## For the lead: what a measurement on the hosted platform should compare

TA-10 needs none: the tests show the names and the one card offline. For
TA-06 and TA-24, the captures listed above.

## TA-24's summarizer part, built: a summary that fails keeps the history

**What the lead asked (after the triage above).** When the agent library's
own summarizer call fails (a busy refusal, a timeout), the earlier part of
the conversation must not be replaced by the library's error line: that
model call goes ahead with the history as it was, nothing sticks in the
state, the library tries again at its next call, and the record says a
summary failed. No retry ladder and no new waits (those stay with TA-24's
live check). (After the review the lead changed "at its next call" to "at
the next turn": `module_work/b14_ta/FIX.md`.)

**How each library generation summarizes** (read from the sources):

- Both: the summarizer is deepagents' `SummarizationMiddleware`
  (`_DeepAgentsSummarizationMiddleware`), built by
  `create_summarization_middleware` inside `create_deep_agent` for the main
  agent, the general-purpose helper and every helper spec. It works in
  `wrap_model_call` (not a before-model hook): it decides with
  `_should_summarize` (0.7 adds `_over_budget`), writes the older messages
  to a history file, makes the summary with `_create_summary` BEFORE the
  model call, hands the model `[summary message, *kept messages]`, and
  returns the state update `_summarization_event` (cutoff, summary message,
  file). Every later call rebuilds what the model sees from that stored
  event, so a stored summary is reused for the rest of the conversation.
- langchain 1.3.14 and older (this laptop: deepagents 0.6.8 with
  langchain 1.3.4): langchain's `_create_summary` catches every error and
  returns `"Error generating summary: <error>"` as the summary. That line
  goes to the model and into `_summarization_event`, and sticks.
- langchain 1.3.15 and newer (GitHub and a fresh host: deepagents 0.7.24
  with langchain 1.4.4 / 1.4.5): langchain asks the summary call again on
  its own (`with_retry`) and then raises; deepagents does not catch it, so
  the model call and the turn end on it. The error line never appears.
- The boundary is langchain's, not deepagents': deepagents 0.6.8 installed
  today resolves langchain 1.3.18 and raises (the review read the wheels
  1.3.4 to 1.3.18). The guard reads the behaviour, not a version.

**What was built.** `aec_agent_core/deep/summary_guard.py`, `install()`:
it wraps five methods of deepagents' summarizer class in place, once:
`wrap_model_call`, `awrap_model_call`, `_create_summary`,
`_acreate_summary`, `_should_summarize`, plus `_over_budget` where it
exists. While the class's own wrap runs, a summary that fails (the error
line, or an exception) raises a private `SummaryFailed` before the model
call and before any state update. The guarded wrap catches it, writes a
`summary_failed` event, and asks the library's own wrap again with its two
triggers saying "no" for that one call (since the review's fix: and for
every later model call of the same turn, FIX.md). The library then hands the model
what it hands it when nothing needs summarizing (the history as it was,
with an earlier good summary if there is one), and it returns no state
update. `compile_agent` calls `install()` before it builds (both pages and
the Document Review page build through it). `activity_log.APP_EVENTS`
gains `summary_failed`.

**Judgement calls.**
- Where it stands: on the library's class, not in the middleware list. The
  summarizer is built inside `create_deep_agent`, and our middleware runs
  inside its wrap, after the summary is made and before its state write,
  so no middleware of ours can stop the write or hand the model the history
  as it was. Replacing the library's middleware was the other way; it would
  move it in the stack and depend on each version's profile machinery.
  Wrapping its methods keeps the library's own code for every step.
- How a failure is read: langchain 1.3.14 and older has no constant or
  code for it, so the exact prefix of its `except` branch
  (`summary_guard.ERROR_PREFIX`, `"Error generating summary:"`); langchain
  1.3.15 and newer by its code path, the exception from `_create_summary`.
- Both generations get the same outcome. On langchain 1.3.15 and newer a
  failed summary used to end the turn; now the turn goes on with the
  history kept.
  The turn is not cut short by a summary.
- If, asked again, the call is too long for the model, the library
  summarizes regardless of the threshold; if that summary fails too, the
  call ends on the provider's error (langchain 1.3.15 and newer) or on
  `SummaryFailed` (1.3.14 and older). The history is still kept, and
  nothing is stored.
- The record: `{"history": "kept", "what": ..., "error_type": <class
  name>}`. The provider's error text is left out: `mask_error_secrets` is
  not on this branch's base.

**Settings.** None: the repair of a defect. With no failing summary the
library's code runs unchanged. The middleware list is not touched, and
both fingerprint files are unmoved.

**Where the code changed.** `compile_agent` (`deep/assembly.py`): three
lines (a comment, an import, `install()`) just after the
`instructions` import, before `_instructions.apply` and before `create`.
They are nowhere near the to-do middleware rule that another branch adds
after `create`. `activity_log.py`: one name added to `APP_EVENTS`, and three
comment lines under it. The middleware list itself is not changed.

**Tests** (`packages/aec-agent-core/tests/test_ta24_summary_guard.py`;
a fake model whose summary call is refused as busy; made-up messages):
- the installed library, end to end (a langchain agent with deepagents'
  summarizer, trigger 4 messages, keep 2): the model is handed the five
  messages as they were, nothing is stored, the record has one
  `summary_failed` with no provider text; the next call summarizes and
  stores the summary; the same holds asynchronously;
- the defect itself on whichever generation is installed (the library's
  own wrap, `__wrapped__`): the error line is handed and stored (langchain
  1.3.14 and older), or the call ends on the refusal (1.3.15 and newer);
- a successful summary: the guarded wrap hands the model exactly what the
  library's own wrap hands it, and stores the same event;
- every summarizer `create_deep_agent` builds (main, general-purpose and a
  helper spec) is of the guarded class; `compile_agent` installs the guard;
- the shape both generations share, with each generation's way of failing
  (raises / error line), sync and async: the history is kept, nothing is
  returned to store, and the record is written; a successful summary and a
  short history come out untouched; too long and failing again ends on the
  error; an unknown class is left alone; `install` wraps once.
  The installed generation runs for real here (0.6.8) and on GitHub
  (0.7.24); the shape tests cover the other one on each machine.

**Noted, not fixed.**
- When a summary fails, the older messages were already written to a
  history file (the library writes it before it makes the summary). Since
  the review's fix that happens once per TURN whose summary fails, not
  once per model call: the later calls of the turn write nothing. The next
  turn's summary writes the older part again. On deepagents 0.7 that is a
  new file each such turn (the first file's id is in the state update that
  is dropped, so the file is not named again); on deepagents 0.6 it is one
  more copy in the one file. Measured on the real library, two failing
  turns of four and three model calls and then a turn whose summary is
  made: 3 files with one copy each (0.7.24 with langchain 1.4.5), 1 file
  with 3 copies (0.6.8 with langchain 1.3.4). With no failure: 1 file,
  1 copy.
- The compact-conversation tool (`SummarizationToolMiddleware`, not in
  our stacks) calls `_create_summary` outside the guarded wrap and is
  unchanged.

**For the lead: what a measurement on the hosted platform should compare.**
A long Document Review conversation on a busy deployment: a
`summary_failed` record and the model's next answer still using the
earliest pages, against no "Error generating summary" text anywhere in
`activity.jsonl`.
