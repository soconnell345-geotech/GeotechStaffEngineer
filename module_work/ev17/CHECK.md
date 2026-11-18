# EV-17: an offline report of a saved run (first version)

## What the item asked

BACKLOG EV-17, "Show what the agents took in and where they looked": per
conversation, what the main agent and each helper took into context (the
question and brief, the documents and pages read, tool results and their
size, tokens), as a panel, a page or an HTML report; and each page looked at
with a box wherever a vision look, tile or zoom was made. The lead's brief
for this first version: an OFFLINE report only, one self-contained HTML file
made from a saved run's record, that an engineer who is not a developer
opens in a browser to read a run in full (the owner's rule "Review every
model run in full"), now that far less AI capacity is there to read records.
No page of the app changes, no tool or prompt, nothing a tester sees.

## What was built, and where

`packages/aec-agent-core/aec_agent_core/run_report.py`, run as

```
python -m aec_agent_core.run_report <run folder> [-o report.html] [--documents <folder>]
```

- `RunRecord`: reads each file of the record, each optional: `activity.jsonl`
  (unreadable lines counted and skipped), `run.json`, `meta.json`,
  `transcript.jsonl`, `messages.json`, `reads.json` and `page_readings.json`
  (in the folder, its `.scratch`, or `files/.scratch` as the suite keeps
  them), and the whole copies in `activity_full/`.
- `build_turns`: sorts the records of each turn into the main agent's and
  each helper's steps (a step = one model call of an agent and the tool
  calls it asked for), with each call's start and end, what the model
  received where that differs (`tool_delivered`, `tool_refused`), its
  vision calls (by `tool_run_id`) and its look events. A helper is told
  apart by the `task_id` its records carry, tied to its `task` call by that
  id when the call's own record carries it, else by the brief's first 80
  characters against the call's `description`, else (records without
  `task_id`) to the open `task` call of its type. Calls are numbered through
  the run in the order the report shows them.
- `PageMap`: per document and page, how it was reached and every look with
  its box. Pages read come from `coverage.pages_from_call` (the same reading
  `cited_pages` uses); a look's boxes from the `view` rects the tool
  returned (the page's whole view, each tile's view, a zoom's view), read
  from the whole copy when the record cut the result, else found in the cut
  text; contact sheets from `render_page_thumbnails`' `sheets[].pages`;
  document handles named through `open_document`'s results, which also give
  the page count. `reads.json` / `page_readings.json` add, unnumbered and
  named by file, looks on pages the activity log shows no look on.
- `PageImages`: small JPEGs of the pages looked at, made with PyMuPDF
  (already a core dependency) when the PDF is in the run folder or under
  `--documents`; otherwise outlines, said once.
- `Report.html()`: the page. Top: the line "Holds this run's text and page
  images; treat it as you treat the documents.", a table of contents, then
  1. the run (asked, model, when, totals, files delivered, notes the record
  carries), 2. step by step, 3. what was taken in (the page grid and counts
  per document), 4. where they looked (an SVG outline per page, boxes
  numbered and linked to the calls), 5. what the record holds (each file
  found or missing and what it gives, and the older shapes met). Each part
  is made inside its own guard: a part that fails says so in one line,
  never a traceback.
- `write_report` / `main`: writes only to `-o` or, by default,
  `run_report.html` beside the record; prints the path and the line above.
  A missing folder exits 2 with a message.
- `packages/aec-agent-core/README.md`: a section "Reading a saved run".

Both folder kinds are handled: a page conversation's folder (`meta.json`,
`transcript.jsonl`, the PDFs in `files/` are found and drawn) and a review
suite task folder (`run.json`, vision records in `files/.scratch`; the
suite deletes its uploads, so `--documents`).

## Calls made

- **Numbering in reading order.** Calls are numbered depth first, a helper's
  calls right after the `task` call that started it, so the numbers in part 2
  run in order down the page; part 4 uses the same numbers.
- **A tiled read counts as "tiled", not also "whole"**: its whole-page call
  is asked for the layout only. The legend says "looked at whole and in
  tiles".
- **"On a contact sheet"** = a page on a sheet `render_page_thumbnails`
  made, whether or not the model then looked at the image. `find_like`'s
  sheets (crops of candidates, not pages) are not counted.
- **A task call's arguments are not repeated** when they are only the brief
  and the helper type: the brief is shown once, at the top of the helper's
  block.
- **Helper blocks open, long payloads folded**: a payload over 600
  characters or 12 lines is folded with its size; payloads are shown up to
  200,000 characters (the rest named). JSON payloads are laid out one field
  a line with short lists (view boxes) kept on one.
- **Page images**: 420 px on the long side, JPEG quality 60, at most 150
  pages drawn (about 10 KB each for a plain made-up page; a dense drawing
  takes more).
- **Vision records only fill gaps**: `reads.json` and `page_readings.json`
  are used for pages the activity log shows no look on (an older or missing
  log), so a look is never counted twice.
- **No setting.** The report is a command a reader runs on a saved record;
  nothing a page or a tester sees changes, so no switch was added.

## Tests (`packages/aec-agent-core/tests/test_ev17_run_report.py`)

The main record is written by the real writer (`ActivityLogger`'s
callbacks, a `side_call` label for the vision calls, `keep_work`'s helper
context for the helper's records), with made-up documents and words, and a
small `max_chars` so a result is cut and kept whole beside the log. Each
test fails on master (the module does not exist there) and passes here.

- `test_every_call_in_order_and_the_helper_under_its_brief` (both with and
  without the helper's id on the task call's own record): all eight calls
  present, numbered 1 to 8 in order; the helper's block holds its brief,
  then its two calls, before what it handed back; the model's own words
  shown.
- `test_page_grid_counts_and_totals`: the grid's counts (text 2, whole 1,
  tiled 1, zoomed 2, contact sheet 4, never opened 1); tool calls, helpers,
  model calls, vision calls, tokens in and out; the cut result read from its
  whole copy; the delivered file and the question shown.
- `test_boxes_numbered_as_the_steps`: four tile boxes, a whole-page box and
  a zoom box at its recorded view on page 3, linked to calls 3 and 8; the
  helper's zoom on page 5; part 2 links to the drawings; "outlines only"
  said once.
- `test_self_contained_and_says_what_it_holds`: no network reference of any
  kind; the content line before the contents; light and dark themes and a
  print style.
- `test_pages_drawn_behind_the_boxes_when_the_pdf_is_found`: a made-up PDF
  under `--documents` gives three page images, no "outlines only" note.
- `test_a_missing_activity_log_is_named_not_a_traceback`: a folder with only
  `run.json` names the missing log at the top and in part 5, and still shows
  the question and the files.
- `test_an_older_record_still_reads`: no `task_id`, no usage, no vision
  labels, a result cut with no whole copy, an unreadable line: the helper
  still sits under its task call, the older shapes are named, the zoom's
  box is found in the cut text, tokens say "not recorded".
- `test_suite_run_folder_with_its_vision_records`: the whole question from
  `run.json` (past the log's 500 characters); `files/.scratch/reads.json`
  adds a look on a page the log does not show, and is not counted twice
  where the log has one.
- `test_page_conversation_folder`: `meta.json`'s title, the transcript's
  file and the answer as the user received it.
- `test_command_line_writes_only_where_asked`: beside the record by
  default, only at `-o` when given; a missing folder exits 2 without a
  traceback.

## Noted, not fixed

- A look that handed back a kept reading (`reuse=true`) counts as a look,
  though no new image was made.
- Pages named only in a search hit are not shown (the grid shows pages
  opened).
- Looks on uploaded images (not PDFs) are listed and outlined, never drawn.
- On records before 2026-10-14, helpers of one type that ran together share
  one block (their records cannot be told apart); part 5 says so.
- `messages.json` is used only for the question when nothing else holds it;
  the history the model was given is not shown.

## For the lead: what a run on real records should compare

Run it on one saved suite task folder (with `--documents` at the suite's
documents) and one page conversation folder of the owner's, and compare
against the raw files: every `tool_start` of the log appears once, in
order, under the right agent; each helper's block holds the calls that
carry its `task_id`; the page grid agrees with `reads.json`; the boxes sit
where the zoom and tile views say on the drawn page; the size and the time
to make the report on a 40-sheet run. Also check whether a real `task`
call's own `tool_start` carries the helper's `task_id` (then helpers are
tied by id; otherwise by the brief's first 80 characters).

## Added in review

An independent review found five things; each is fixed with a test in
`test_ev17_run_report.py`.

1. **The transcript was read by the wrong key.** `RunRecord.questions` and
   `RunRecord.answers` read `content`; the pages' writer
   (`append_transcript`, entries `{"role": "user", "text": ...}` and
   `{"role": "assistant", "text": ..., "artifacts": [...]}`) writes `text`,
   as the core's own reader `cited_documents._transcript_user_texts` reads
   it. Now read by `text`: "The answer as the user received it" appears, a
   question over 500 characters is whole, and a folder with a transcript
   and no activity log names its questions. A turn whose answer the record
   does not hold says so. "When" prints `meta.json`'s `created` (seconds)
   as a date and time. The conversation test builds its transcript in the
   writer's shape (a copy of the shape: the core's tests do not import
   Document Review).
2. **The page grid said "never opened" with no activity log.** With no
   `activity.jsonl` no look is recorded at all, so the empty squares, their
   count and the part's introduction say "no look recorded". With the log
   present nothing changes.
3. **The whole-copy path check had no separator.** `RunRecord.sidecar` now
   reads a file only when its resolved path lies inside the run folder
   itself; a made-up sibling folder whose name starts with the run folder's
   name is not read.
4. **No traceback.** `main` ends a record it cannot build a report from
   (the turns and the page map are built before any part's own guard) with
   one plain line naming what failed and exit code 1.
5. **No absolute path in the page.** The header names the run folder by its
   own name only; an error line inside a part has the run folder's and the
   documents folder's paths cut to their names (`Report._plain`). The
   record's own paths are relative already and page images are inline.
   Judgement call: the documents folder is not named in the page at all
   (it never was); nothing was added to name it.

**Size**, measured in review on a made-up run of 300 large sheets and 600
tiled looks: the page comes out about 1.5 times the size of the activity
log (a 12 MB log gave a 17.7 MB page in about 12 s), with at most 150 page
images drawn (`MAX_PAGE_IMAGES`).
