# Field session 2026-10-14: a report's borings to DIGGS (app 5.33.0, Funhouse, Prompter key)

**The session.** The owner asked the geotech page to build a DIGGS file of
the new and historic explorations in Volume 1 of a two-part geotechnical
report (a 242-page PDF fetched from SharePoint), with an x-y grid read off
the investigation plan where no coordinates are given. It ran on Funhouse
through the Prompter API key route (`webapp/tinyapps_entry.py` on the
Databricks launcher), model GPT-5.6 Sol, recursion limit 200.

**What came back.** The turn ran 3,065 s (51 min): 997 model calls, 349 tool
calls, helpers (`task` → general-purpose) extracting the logs, up to five at
once. It ended on a rate-limit refusal from the model endpoint
(`OpenAIRateLimitError: Too Many Requests`, 8 model errors in the log). One
file was saved: a DIGGS file for one test pit. No answer.

**Raw material** (gitignored `raw/`, never committed; it carries the
report and the project's names): `raw/conversation/` (the app's export:
`activity.jsonl` 2,696 records, `messages.json`, `transcript.jsonl`,
`trace.jsonl`, `page_readings.json`, `reads.json`, `meta.json`,
`downloads.json`), `raw/files/` (the report PDF and the saved DIGGS file),
and the two zips as received.

## First read (lead, 2026-10-14)

1. **The conversation kept none of the turn's work.** `messages.json` holds
   the question and an error stub; every helper's extracted data exists
   only in `activity.jsonl`. The app has no checkpointer (history is
   `messages.json`), so "continue" restarts the extraction from nothing,
   while the error message says "Nothing is lost ... ask the agent to
   continue". BACKLOG TA-19.
2. **Rate limit.** Five helpers started at once on one key at t = 823 s; the
   refusals followed. The engine's busy retries did not carry the turn
   through. To read in full: when each 429 came, what was in flight, how
   long the retries waited.
3. **No token accounting.** `usage` is null on every `model_end` and
   `turn_tokens` is 0 in `trace.jsonl`: the Prompter key route records no
   tokens, so the session's cost cannot be told from the record.
4. **The refusal rows.** A helper reported 17 SPT rows refused by
   `write_diggs` because a printed N of ">50" went into `n`; the hint now
   names the refusal field (RI-26, cc51fa7).

## The full read (2026-10-14)

`TRACE_REVIEW.md` (an Opus subagent read every record; private tables,
scripts and page images are in `raw/notes/`). In short: where the reading
finished it was good (401 of 406 checked values right), but the work was
lost: four helpers stopped holding extracted data and returned apologies;
the requested grid was mirrored and never reached a file; agent model calls
had no busy ladder, and the final refusals fit a tokens-per-minute limit
reached through very large tool results. This was a pipeline job done by a
chat loop.

Dispositions (BACKLOG; all fixes on master, unreleased):
- FIXED 2026-10-14:
  - TA-19, an errored turn keeps its work (6ffa347);
  - TA-21, a stopped helper returns its reading as a file (6ffa347);
  - TA-20, busy ladder on agent calls (8cd38f4);
  - TA-22, two helpers at once, a token pacer, a helper budget (6ffa347,
    8cd38f4);
  - EV-14, the record (4dee54a);
  - RI-27, DIGGS: not-encountered water, unknown positions as NaN / None,
    printed refusals written (d81c1f8);
  - RI-26, the error message for a printed ">50" (cc51fa7).
  - TA-25, the continue turn ran over 3 hours, more than 2.5 of them
    inside one `reviewer` delegation that never returned (the owner's
    snapshots at 26 and 180 minutes end on the same line): the reviewer
    had no model-call budget and was asked to check data it cannot open.
- PLANNED, Next: RI-28, RI-30, PL-24, PL-25, VIS-24, VIS-25, EV-15, TA-24.
- PLANNED, Later: RI-29 (report to DIGGS through `report_ingest`, after
  OWN-11), TA-23, EV-16.
- Live checks (only a real run shows them): whether the gateway honours
  token-use reporting and what its 429s carry; GPT replayed on its own
  earlier tool calls after an errored turn.
