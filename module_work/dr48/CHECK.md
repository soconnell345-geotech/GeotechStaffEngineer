# DR-48 and TA-41: check notes (2026-11-18)

One notes file for the two items of this branch (`claude/dr-48-ta-41-turn`):
DR-48 in commit c45c932, TA-41 in commit cb138ff.

## DR-48: photograph pages nothing looked at

**Asked.** At turn end, list from the call log the attached documents with
photograph pages neither looked at nor seen on a contact sheet; tell the
model once, before the answer (nothing when all were covered). Lead's ruling:
the record always (no switch, no model call, never fails a turn); telling the
model behind a new switch, off by default, through the cited-pages retry
machinery; with the switch off the model is handed exactly what it is today.

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
  later call looked at (any call whose arguments name the sheet's PNG).
  Returns `checked`, `attached`, `against`, `documents`, `photos`, `unseen`
  (`{}` when all were covered), and `told_before` / `left_out` when any;
  pages are 1-based PDF page numbers.
- The activity reader (`_feed_activity`) now notes contact sheets and the
  looks at them (kept apart in `PagesOpened.sheet_seen`, so the cited-pages
  check reads exactly what it read before) and the pages earlier notes told
  (`PagesOpened.photos_told`, from the record's own `cited_pages` events).
- `TurnPageCheck` carries it: `finish` always records `photo_pages` (and
  `photo_mode`) in the turn's `cited_pages` event; `feedback`, with the switch
  on, adds `photo_note_for_model`'s note, once a turn, joined with the page
  check's note when both speak; `retry_status` names what was told (the page
  check's alone is `RETRY_STATUS`, as the hosts used before).
- `review_flags.py`: `GEOTECH_PHOTO_PAGES` and `photo_pages()`.
- `review_eval/runner.py`: a suite run writes `attachments.json` beside its
  record, as the pages do.
- `activity_log.py`: the `cited_pages` event's description names
  `photo_pages`.

**Setting.** `GEOTECH_PHOTO_PAGES`, default off. `retry` (or `1`, `true`,
`yes`, `on`) tells the model once. Both pages (the geotech page gets it when
its core pin moves: its `webapp/core.py` builds the same `TurnPageCheck`).

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
- Photograph pages are measured on the file opened on its own, not through
  the conversation's document toolkit: no handle, eviction or open-document
  limit of the conversation is touched. The cost is one pass over the
  document's text the first time (cached after).
- A contact sheet counts only once a later call names its PNG: rendering
  thumbnails shows the model nothing until it looks.
- The note says: look at the pages if the answer rests on those documents or
  names them as its basis, otherwise say plainly their photographs were not
  looked at. Worded for every model, naming no model's habit.
- Events: the note alone is logged with `action: "photos_fed_back"`; with the
  page check's note in the same pass it rides on that check's `fed_back`
  event. Existing readers count `fed_back` for the page and judgment checks.
- With documents attached, a turn whose answer cites no page now writes a
  `cited_pages` event too (it carries `photo_pages`; `n_cited` 0).

**Tests.**
- `packages/aec-agent-core/tests/test_dr48_photo_pages.py`: planlens names
  the made-up photograph pages; the record names each document's unseen
  pages, and `{}` when all were looked at; a contact sheet counts only once
  looked at, and leaves the cited-pages record unchanged; a document whose
  page kinds cannot be told is left out with its reason; the call-log
  fallback ignores files elsewhere; no attached document, no record; `finish`
  records it with the switch off and hands nothing; an error is recorded and
  never fails the turn; the switch's values, `switches()` and `describe()`;
  switch off, the page check's note is exactly today's; switch on, one note
  naming the document and pages, once, and not again in a later turn; none
  when all were seen; both checks in one note and one event.
- `packages/aec-document-review/tests/test_dr48_photo_pages_turn.py`: the
  item's Done line on `core.stream_turn` over two made-up PDFs (one note
  naming `b.pdf` and PDF pages 2-3, the reply after it delivered; none when
  the turn had looked at them); switch off, the model's inputs and the
  stream equal those of the same turn with the check removed; the page
  check's retry note unchanged with the switch off; a suite run keeps the
  upload list and records a document it never opened.
- Every test fails on master (the names it calls do not exist there).
- Existing tests run and passing: the cited-pages, cited-documents and
  judgment-check turn tests (both pages), the suite-runner and record tests
  of the core, the app and Document Review, both fingerprint tests (nothing
  the model is handed moved).

**Noted, not fixed.**
- A photograph page with no caption in its text layer (a scanned photo
  appendix without OCR) is not a `photos` page to planlens, so it is not
  listed.
- The first turn end after a large document is attached pays one planlens
  pass over it (cached for the process after).

## TA-41: diagnostics and the turn's records on a slow day

**Asked.** Connection diagnostics runs in the page with no time limit, about
ten model calls in a row at up to 180 s each; run it detached, or give its
calls a short limit and stop at the first model failure. Lead's ruling: the
repair of a defect, on by default; the simpler route the page supports; do
not change what is checked or what a passing run shows.

**Built** (`packages/aec-document-review/aec_document_review/diagnostics.py`):
`_limited` runs one live check in a worker thread carrying the caller's
context and waits at most its limit; `_live_checks` runs the live checks in
order within an overall budget, stops at the first `fail` and lists the rest
as `not run: '<check>' failed first` (or that the time was spent);
`_helper_steps` replaces `_helper_request_checks` with the same names and
notes. `run_diagnostics` hands its live checks to `_live_checks`.

**Calls made.**
- The limit route, not a detached run: the page calls the battery inside a
  button and a spinner and has no job store or polling for it; detaching
  would add that machinery to the page, while the limit keeps the repair in
  one module and bounds notebook and terminal callers too.
- `CALL_LIMIT_S = 30` per live check (a few words; a working service answers
  in seconds), `VISION_LIMIT_S = 60` for the probe's four small calls,
  `BUDGET_S = 60` for all live checks together. Module constants, no new
  setting.
- A check that only warns (`warn`) does not stop the battery; a `fail` does,
  including a streaming failure (the item says the first model failure).
- A request that times out is left to finish in its thread (a daemon), which
  ends when the client's own timeout does.

**Tests.** `packages/aec-document-review/tests/test_ta41_diagnostics_limits.py`:
a request that never answers holds the page under 5 s, the rest listed as not
run and never asked (fails on master: it waits the full 10 s); the first
failure stops the rest with its text kept (fails on master: streaming and
tools were asked); a passing run equals the checks run directly (passes on
master too: a guard); the caller's context reaches the call (a guard); the
overall budget and a hanging helper named (fail on master: the functions are
new).

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

- Arms: `baseline` against `("photo_pages_retry", {"GEOTECH_PHOTO_PAGES":
  "retry"})` (pass the pair in `score_review_suite`'s `arms`), on tasks with several attached
  documents where at least one has photograph pages, and on tasks where the
  photographs do not bear on the question (the cost of a note nobody needed).
- From the record (both arms, no extra call): the share of runs whose last
  `cited_pages` event has a non-empty `photo_pages.unseen`, per task. This is
  the size of the gap without any note.
- With the switch: how often the note fired (`action` `photos_fed_back` or
  `photo_pages.told`); after it, whether the run looked at the named pages
  (look calls on those pages after the note) or said plainly they were not
  looked at; whether the delivered answer or file still names a document as
  its basis whose photographs were never seen.
- Scores of the delivered work per arm, and no loss on the tasks where the
  photographs do not matter.
- Cost per run: the extra model pass (input tokens of the whole context),
  the looks it caused, turn seconds, and the first turn's time to measure a
  large document.
- The measure itself: compare `photo_pages.photos` per document with the
  pages a reader would call photographs (pages flagged that are not
  photographs; photograph pages missed, such as scans without a text layer).
