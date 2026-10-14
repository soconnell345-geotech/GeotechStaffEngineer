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

The full step-by-step read (CLAUDE.md "REVIEW EVERY MODEL RUN IN FULL")
follows in `TRACE_REVIEW.md`; its fix list goes into `BACKLOG.md` by ID.
