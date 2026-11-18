# DR-48 and TA-41: check notes (2026-11-18)

One notes file for the two items of this branch (`claude/dr-48-ta-41-turn`):
DR-48 in commit c45c932, TA-41 in commit cb138ff, and the review's fixes on
top of them (one commit each, after this file's first version).

## DR-48: photograph pages nothing looked at

**Asked.** At turn end, list from the call log the attached documents with
photograph pages neither looked at nor seen on a contact sheet; tell the
model once, before the answer (nothing when all were covered).

**The lead's ruling (after review; it replaces the first one, "the record
always on").** The record is not free: the measure opens each attached PDF
with planlens and runs its page roles over every page, before the answer is
shown (the reviewer measured seconds for a made-up document of a few hundred
pages, more with several, at the first turn end after each restart). So the
switch `GEOTECH_PHOTO_PAGES` has three states:

| State | At turn end | The record | The model |
|---|---|---|---|
| unset / `off` (default) | nothing of DR-48 runs: no measure, no planlens pass | the `cited_pages` event exactly when and as before (none for an answer that cites no page) | handed exactly what it was |
| `record` | the measure runs | the event carries `photo_pages` (and `photo_mode`) | handed exactly what it was |
| `retry` (or `1`, `true`, `yes`, `on`) | the measure runs | as `record`, plus a `photos_fed_back` event when the note is handed | told once, before the answer goes out |

What `record` and `retry` cost: no model call, but one planlens pass over
every page of each attached PDF at the first turn end after it is attached
(cached per file while the process lives, so again after each restart),
before the answer is shown: a few seconds for a document of a few hundred
pages, more with several. The switch's text in `review_flags.py` says so.

**Built** (all in `packages/aec-agent-core/aec_agent_core/cited_pages.py`
unless named):
- `photograph_pages(path)`: the 0-based pages planlens' own page roles call
  `photos` (`planlens.document.roles.page_roles`), the document opened on its
  own with `planlens.document.open_document`, measured once per file (path,
  size and time; 64 files kept). `(None, why)` when the page kinds cannot be
  told: not a PDF, the file missing, a planlens without page roles, a file
  planlens cannot read.
- `attached_documents(record_folder, opened)`: the page's upload list
  (`attachments.json` beside `activity.jsonl`, each key a file in `files/`);
  without one, the documents the call log opened that are files in the
  conversation's own folder.
- `photo_coverage(...)`: per attached document, its photograph pages minus
  the pages looked at (`analyze_pdf_page`, `render_region`, the vision
  records) and the pages on a `render_page_thumbnails` contact sheet that a
  later call of a looking tool (`SHEET_LOOK_TOOLS`: `analyze_image`,
  `analyze_pdf_page`, `render_region`) named. Returns `checked`, `attached`,
  `against`, `documents`, `photos`, `unseen` (`{}` when all were covered),
  and `told_before` / `left_out` when any; pages are 1-based PDF page
  numbers.
- The activity reader (`_feed_activity`) notes contact sheets and the looks
  at them (kept apart in `PagesOpened.sheet_seen`, so the cited-pages check
  reads exactly what it read before) and the pages earlier notes told
  (`PagesOpened.photos_told`, from the record's own `cited_pages` events).
- `TurnPageCheck` carries it, by `photo_mode`: `off`, `finish` and `feedback`
  never call the measure; `record`, `finish` records `photo_pages` in the
  turn's event; `retry`, that and `feedback` adds `photo_note_for_model`'s
  note, once a turn, joined with the page check's note when both speak.
  `retry_status` names what was told (the page check's alone is
  `RETRY_STATUS`, as the hosts used before).
- Told only when handed: the photo note's record is its own
  `photos_fed_back` event, carrying `told`. It is written when `feedback`
  returns the note, or, for a check built with `confirms_notes=True`, when
  the page calls `TurnPageCheck.note_handed()`. Document Review's page
  (`aec_document_review/core.py`, `_page_check` and the turn loop) builds its
  check that way and calls `note_handed()` only on the path that hands the
  note to the model; a note it drops because the turn's budget has ended
  (TA-38) is not recorded as told, so a later turn tells those pages.
- `review_flags.py`: `GEOTECH_PHOTO_PAGES` (its text, with the cost) and
  `photo_pages()` returning `off`, `record` or `retry`.
- `review_eval/runner.py`: a suite run writes `attachments.json` beside its
  record, as the pages do.
- `activity_log.py`: the `cited_pages` event's description names
  `photo_pages` (with the switch on).

**Counting the gap after the fact, with no switch.** `photo_coverage(folder)`
reads a saved folder by itself: a suite run's folder
(`runs/<arm>/<task>/`: `activity.jsonl`, `attachments.json`, `files/`) or a
page conversation's folder (the same three). From a Python prompt with the
core and planlens installed:
`from aec_agent_core.cited_pages import photo_coverage;
photo_coverage("<run folder>")["unseen"]`. It costs the same planlens pass,
once per document, but outside any turn. A run made before this branch has
no `attachments.json`; the check then falls back to the documents the call
log opened (a document never opened is then missed: say so when counting).

**Calls made.**
- "Once" is once per turn and, across turns, once per page: pages a note
  named are not named again later in the conversation (read back from the
  record). Re-telling on every turn of a long conversation would cost a full
  extra pass each turn for pages the user may not need.
- The switch is in a new tuple `review_flags.LATER_ENVS`, not `ALL_ENVS`, and
  has no named arm in `ARMS`: the geotech app's own tests
  (`funhouse_agent/deep/tests/test_judgment_calls.py`,
  `funhouse_agent/deep/tests/test_review_switches_offline.py`) pin both
  whole, and that code is not edited here. `switches()` clears `LATER_ENVS`
  and `describe()` names them, so it behaves as a switch in every way.
- `confirms_notes` is an argument, default off, rather than a call every page
  must make: the geotech page's turn loop (`webapp/core.py`, quiet here)
  hands every note `feedback` returns, so for it "returned" is "handed" and
  it needs no change; only Document Review drops a note (at the budget's
  end), and it says when it handed one.
- The page check's own `fed_back` event is written in `feedback` as before
  (also when the note is then dropped): that is EV-18's behaviour, not
  changed here.
- With the switch off the call-log reader still notes contact sheets as it
  reads `activity.jsonl` (no file is opened, nothing reaches the record or
  the model); everything that costs time (the measure, planlens' pass) runs
  only with `record` or `retry`.
- A contact sheet counts only once a looking tool names its PNG (the way the
  model is told to view one: `document_tools.IMAGE_VIEW_HINT`). A helper
  given the sheet's path counts through its own `analyze_image` call, which
  is on the record; a call that merely mentions the file (listing it,
  handing it on) does not count.
- Photograph pages are measured on the file opened on its own, not through
  the conversation's document toolkit: no handle, eviction or open-document
  limit of the conversation is touched.
- The note says: look at the pages if the answer rests on those documents or
  names them as its basis, otherwise say plainly their photographs were not
  looked at. Worded for every model, naming no model's habit.

**Tests.**
- `packages/aec-agent-core/tests/test_dr48_photo_pages.py`: planlens names
  the made-up photograph pages; on a saved folder, the measure names each
  document's unseen pages, and `{}` when all were looked at; a contact sheet
  counts only once looked at, and leaves the cited-pages record unchanged;
  only a looking tool naming the sheet counts (a file listing and a helper
  call that mention it do not); a document whose page kinds cannot be told is
  left out with its reason; the call-log fallback ignores files elsewhere; no
  attached document, no record. Switch off: no event for an answer citing no
  page, the event for one citing pages equals the old form with no photo
  field, and the measure and planlens' pass are called zero times. `record`:
  the event carries the gap, nothing is told. An error is recorded and never
  fails the turn; the switch's three states, `switches()` and `describe()`;
  switch off, the page check's note is exactly today's; `retry`, one note
  naming the document and pages, once, and not again in a later turn; none
  when all were seen; both checks in one note (the page check's event as
  before, then `photos_fed_back`); with `confirms_notes`, a note never handed
  is not told and the next turn tells it, and `note_handed()` records it
  once.
- `packages/aec-document-review/tests/test_dr48_photo_pages_turn.py`, on
  `core.stream_turn` over two made-up PDFs: `retry` gives one note naming
  `b.pdf` and PDF pages 2-3 and delivers the reply after it, none when the
  turn had looked at them; switch off, no `cited_pages` event for an answer
  citing no page, the old event for one citing pages, and zero calls of the
  measure and of planlens' pass; `record`, the model's inputs and the stream
  equal the off state's and the event carries the gap; a note dropped because
  the budget ended is not told and the next turn tells it; the page check's
  retry note unchanged with the switch off; a suite run on the `record` arm
  keeps the upload list and records a document it never opened.
- Every test fails on master (the names it calls do not exist there).
- Existing tests run and passing: the cited-pages, cited-documents and
  judgment-check turn tests (both pages), the suite-runner and record tests
  of the core, the app and Document Review, the switch tests (the two
  app-side files that pin `ALL_ENVS` and `ARMS` unchanged), both fingerprint
  tests (nothing the model is handed moved).

**Noted, not fixed.**
- A photograph page with no caption in its text layer (a scanned photo
  appendix without OCR) is not a `photos` page to planlens, so it is not
  listed.
- A page planlens wrongly calls `photos` can only add a line. With `retry`
  on, that can lead the model to say a drawing sheet's "photographs were not
  looked at": one thing the measurement must count.

## TA-41: diagnostics and the turn's records on a slow day

**Asked.** Connection diagnostics runs in the page with no time limit, about
ten model calls in a row at up to 180 s each; run it detached, or give its
calls a short limit and stop at the first model failure. Lead's ruling: the
repair of a defect, on by default; the simpler route the page supports; do
not change what is checked or what a passing run shows.

**Built** (`packages/aec-document-review/aec_document_review/diagnostics.py`):
`_limited` runs one live check in a worker thread carrying the caller's
context and waits at most its limit; `_live_checks` runs the live checks in
order, stops at the first `fail` and lists the rest as
`not run: '<check>' failed first`; `_helper_steps` replaces
`_helper_request_checks` with the same names and notes. `run_diagnostics`
hands its live checks to `_live_checks`.

**The rule (after review).**
- Each live check waits at most its own limit (`CALL_LIMIT_S = 30`;
  `VISION_LIMIT_S = 60` for the vision probe), and a check that is started
  always gets that whole limit: the time left in the battery never shortens
  it. The battery stops at the first model failure.
- `BUDGET_S = 180` bounds STARTING a check: none is started once 180 s have
  passed since the first. Such a check reads
  `not run: the time allowed for the checks was used up` (`TIME_USED_UP`)
  with the skipped mark, never `fail`, and never names the service as slow.
  An ordinary passing run (three small calls, the probe, two calls for each
  of two helper models) takes about a minute on a slow day, so it passes
  with room to spare; the page is held at most 180 s plus one check's limit,
  and only while every check passes.

**Calls made.**
- The limit route, not a detached run: the page calls the battery inside a
  button and a spinner and has no job store or polling for it; detaching
  would add that machinery to the page, while the limit keeps the repair in
  one module and bounds notebook and terminal callers too.
- `BUDGET_S` is a plain 180 s rather than a sum of the checks' limits: the
  sum grows with every helper model (270 s with two), and the bound is there
  for the slow-but-passing day, where the checks take a fraction of their
  limits. Module constants, no new setting.
- A check that only warns (`warn`) does not stop the battery; a `fail` does,
  including a streaming failure (the item says the first model failure).
- A request that times out is left to finish in its thread (a daemon), which
  ends when the client's own timeout does: at most one per press.
- The checks are timed by `diagnostics._clock` (`time.monotonic`), so the
  tests play a slow day on it instead of waiting.

**Tests.** `packages/aec-document-review/tests/test_ta41_diagnostics_limits.py`:
a request that never answers holds the page under 5 s, the rest listed as not
run and never asked (fails on master: it waits the full 10 s); the first
failure stops the rest with its text kept (fails on master: streaming and
tools were asked); a passing run equals the checks run directly (passes on
master too: a guard); the caller's context reaches the call (a guard); a
started call that outlasts its own limit fails by that limit while the
battery's time has run out, and a battery with no time lists every check as
not run; the reviewer's slow day (small calls of 6 s, a 23 s probe, two
helper models) passes every check; a battery that uses up its time lists
the rest as not run, none failed, no line calling the service slow; a check
started with 3 s of the battery left still gets its whole limit; a hanging
helper is named (all but the guards fail on master: the functions are new).

**Edited tests.** `packages/aec-document-review/tests/test_diagnostics.py`:
`test_errors_reported_verbatim` and
`test_the_streaming_switch_is_named_for_the_host` read the streaming failure
after a failed plain request; they now take a model whose plain request works
and whose streaming fails, and the first also checks that streaming is not
run after a failed plain request.

**Noted, not fixed.**
- The turn's crash checkpoint (`turn_record.TurnRecord.checkpoint`) rewrites
  the whole record after every message: the bytes written grow with the
  square of a long turn's length. A cheaper checkpoint changes what a crash
  can recover, so it is not a small fix.
- The activity log (`aec_agent_core.activity_log.ActivityLogger._write`)
  opens the file once per event, slow on a network share, and keeps error
  text as the client raised it; keeping a handle open across threads and
  restarts is not a small fix.

## For the lead: measuring `GEOTECH_PHOTO_PAGES` on the hosted platform

- Arms (pass the pairs in `score_review_suite`'s `arms`; neither is in
  `ARMS`): `baseline`, `("photo_pages_record", {"GEOTECH_PHOTO_PAGES":
  "record"})` and `("photo_pages_retry", {"GEOTECH_PHOTO_PAGES": "retry"})`,
  on tasks with several attached documents where at least one has
  photograph pages, and on tasks where the photographs do not bear on the
  question (the cost of a note nobody needed).
- The gap with no note: the `record` arm, the share of runs whose last
  `cited_pages` event has a non-empty `photo_pages.unseen`, per task. The
  same count can be made on `baseline` (or any saved run) after the fact
  with `photo_coverage(<run folder>)`, no switch.
- `record` against `retry`: how often the note fired (`photos_fed_back`
  events, `photo_pages.told`); after it, whether the run looked at the named
  pages (look calls on those pages after the note) or said plainly they were
  not looked at; whether the delivered answer or file still names a document
  as its basis whose photographs were never seen; how often the model says
  a drawing sheet's "photographs were not looked at" (a wrong `photos` role).
- Scores of the delivered work per arm, and no loss on the tasks where the
  photographs do not matter.
- Time, both against `baseline`: `record` and `retry` pay the first turn
  end's planlens pass over each attached document (turn seconds of the first
  turn); `retry` adds the extra model pass (input tokens of the whole
  context) and the looks it causes.
- The measure itself: compare `photo_pages.photos` per document with the
  pages a reader would call photographs (pages flagged that are not
  photographs; photograph pages missed, such as scans without a text layer).
