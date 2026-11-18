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
images drawn (`MAX_PAGE_IMAGES`; part (b) below changed the caps).

# EV-17 part (b): the page viewer, and a follow-up to part (a)

## What the item asked

Part (b) of EV-17, inside the same report: from the page grid a reader
opens a page and sees its image with a box wherever a look, a tile or a
zoom was made; each box tells which call it was (step, tool, who made it:
the main agent or which helper) and, on choosing it, what was asked and the
start of what came back; previous / next page and previous / next page
looked at. One self-contained file, no network, works opened from disk;
bounded for a long document, saying what was left out; a page with no image
still lists its looks. Follow-up to (a): a folder with a transcript and no
`activity.jsonl` lists each turn's question and answer. The rules of (a)
stay: reads only, no absolute local path in the page, a broken part of the
record gives a plain line, nothing read outside the run folder but the
documents folder given.

## What was built, and where

All in `packages/aec-agent-core/aec_agent_core/run_report.py`.

- `Report._plan_images`: draws the pages once, within the caps, in this
  order: each page looked at, small (part 4) and large (the viewer) while
  the large ones stay under their cap; then the other pages of each
  document whose PDF is found, small, the pages taken in some other way
  (text read, contact sheet) first. `PageImages.draw` renders the largest
  size once and scales the smaller from it (`fitz.Pixmap(pix, w, h)`, else a
  second render); `PageImages.rect` gives a page's size.
  `Report._image_note` says what the caps left out (part 4, and the viewer
  on a page with no image).
- `Report._page_rect`: a page's outline, from the PDF, else the whole-page
  view a look returned, else round its boxes (the logic part 4 had,
  shared).
- `_places`: call number -> turn, step of its agent, and who made it ("Main
  agent", "Helper 1 (general-purpose)"), as part 2 shows them.
- `Report._view_calls` and `Report._viewer_data`: the JSON block
  `viewer-data`: per document its pages (size, how taken in with the call
  numbers, looks `[call, kind, box, tile, tool, source file]`, image or the
  reason for none); per call named on a page: tool, who, step, turn, what
  was asked and what came back (each laid out as part 2 lays out JSON, up
  to `VIEW_ASKED_CHARS` / `VIEW_BACK_CHARS` = 1,500 characters, with the
  whole size), its status (refused, error, no result) and whether the model
  received something else (part 2 shows it). `_script_json` writes `<`, `>`
  and `&` as JSON escapes, so a record's text cannot end the block.
- `_VIEWER_HTML`, `_VIEWER_SCRIPT`, CSS: a full-window panel over the
  report. The address names the page open (`#view-<document>-<PDF page>`):
  opening adds one history entry, moving between pages replaces it, so the
  browser's Back (or Esc, or Close) returns to where the reader was.
  Boxes are drawn on the page image as part 4 draws them, each with its
  call number (and tile name) and a hover title giving call, tool, agent
  and step; choosing a box, a numbered line or a call number shows the
  call, highlights its boxes and dims the others, and links to the call in
  part 2. The grid's squares (and "Open in the page viewer" under each
  drawing in part 4, shown by the script) open it; with scripts off the
  report is what it was.
- `Report._turns_without_log` and `RunRecord.answer_files`: with no
  activity log, part 2 lists each turn's question, its answer (open) and
  the files shown with it, from the transcript or the suite's `run.json`;
  a turn without one says so; neither gives one plain line.
- Part 3's count line now wraps at phone width (its spans had no break
  between them; seen while checking the viewer at 512 px).
- `packages/aec-agent-core/README.md`, "Reading a saved run": the viewer,
  the caps and the measured sizes.

## Calls made

- **Image sizes and caps, from a measurement.** Made-up 200-page PDFs of
  three kinds (letter-size report pages of dense text and tables; 36 by 24
  inch drawing sheets of dense line work and small text; letter-size scans,
  a noisy raster), 20 pages of each drawn, JPEG quality 60, KB per page and
  ms to draw:

  | long side | report | drawing | scan | ms (report / drawing / scan) |
  |---|---|---|---|---|
  | 420 px | 20.8 | 22.3 | 19.3 | 20 / 57 / 25 |
  | 700 px | 60.9 | 55.5 | 49.1 | 41 / 75 / 75 |
  | 900 px | 96.0 | 87.1 | 75.2 | 61 / 111 / 84 |
  | 1100 px | 134.7 | 115.0 | 104.4 | 89 / 186 / 109 |
  | 1400 px | 192.4 | 168.2 | 154.2 | 129 / 247 / 188 |

  Quality 50 saves under 10 % at 900 px; 75 costs about 20 % more. Chosen:
  `VIEW_IMAGE_PX = 900` for the pages looked at (a letter page at 1.1 pixels
  a point, so 10 pt text about 11 pixels high; 1100 costs 40 % more for the
  same pages), `PAGE_IMAGE_PX = 420` kept for part 4 and the other pages,
  `JPEG_QUALITY = 60` kept. Caps: `MAX_VIEW_IMAGE_BYTES` 8 MB (about 90 to
  110 large pages; past it the viewer shows part 4's small image),
  `MAX_OTHER_IMAGE_BYTES` 4 MB (about 200 small pages), `MAX_PAGE_IMAGES`
  raised from 150 to 300 (a 200-page document looked at whole is drawn
  whole), `MAX_IMAGES_DRAWN` 800 (bounds the time to draw). Upper bound of
  images: about 300 x 25 KB + 8 MB + 4 MB, under 20 MB (27 MB written as
  base64).

  The whole report, measured on made-up runs written by the real writer
  over those PDFs (each page looked at whole and in 2 x 2 tiles, every
  other one zoomed; or one page in five looked at, all pages read as text):

  | run | page | images | built in | large / small only / others drawn |
  |---|---|---|---|---|
  | drawing sheets, all 200 looked at | 17.5 MB | 11.8 MB | 16.9 s | 95 / 105 / 0 |
  | drawing sheets, 40 looked at | 10.4 MB | 7.5 MB | 11.8 s | 40 / 0 / 160 |
  | report pages, all 200 looked at | 17.7 MB | 12.0 MB | 11.0 s | 86 / 114 / 0 |
  | report pages, 40 looked at | 10.7 MB | 7.7 MB | 6.6 s | 40 / 0 / 160 |
  | scans, all 200 looked at | 17.5 MB | 11.8 MB | 13.8 s | 109 / 91 / 0 |

  In headless Chrome the 17.5 MB page was ready (DOMContentLoaded) in
  0.35 s and the 10.7 MB one in 0.2 s; the viewer then showed page 1 with
  its large image and pages 150 and 200 with part 4's small one.
- **Each image once.** Part 4 keeps its small images inline (it works with
  scripts off and in print); the viewer's large images and the other pages'
  images sit in the JSON block only, and for a page looked at with no large
  image the viewer takes part 4's.
- **The viewer covers every page of a document**, not only the pages looked
  at, so previous / next page click through the document as the owner's
  idea put it; a page not taken in says so. Previous / next page stay in
  one document; previous / next page looked at run through the whole run,
  document after document.
- **What a box answers to**: the whole box, not only its line; where boxes
  overlap the one on top is chosen (a zoom over a tile over the whole
  page), and the list beside the page reaches every look.
- **The viewer also lists how the page was taken in** (text read, contact
  sheet, with their call numbers), so a page read only as text can be
  checked against the call that read it.
- **No setting.** The report reads a saved record and writes one file;
  nothing a model or a tester is handed changes.

## Tests (`packages/aec-agent-core/tests/test_ev17b_page_viewer.py`)

The made-up run of `test_ev17_run_report.py` (the real writer), hand-built
records of the same shape, and PDFs made in the test. Each fails on the
base (no viewer, no `_VIEWER_SCRIPT`; part 2 said only "the steps cannot be
shown") and passes here.

- `test_every_box_names_its_call_step_tool_and_agent`: page 3's looks
  (call 3 whole page and four tiles, call 8 zoom at its box), the helper's
  zoom on page 5; how page 1 was taken in; calls 3 and 8 "Main agent" steps
  2 and 4, calls 6 and 7 "Helper 1 (general-purpose)" step 1; what was
  asked and the start of what came back, a cut result's start from its
  whole copy with its whole size; only the calls a page names.
- `test_the_viewer_is_in_the_same_file_and_reaches_nothing_else`: no
  network reference of any kind (the patterns of part (a)'s test), the
  panel and its five buttons, the grid's document index, part 4's link.
- `test_a_record_text_cannot_end_the_viewer_data`: a result holding
  `</script>` stays inside the JSON block and reads back whole.
- `test_page_images_for_the_viewer_when_the_pdf_is_found`: pages looked at
  900 px for the viewer, once in the file, 420 px in part 4; the other
  pages 420 px; the page size once per document.
- `test_caps_bound_the_images_and_say_what_was_left_out`: with each cap
  made small: the large images stop (the viewer falls back to part 4's),
  the other pages stop after those taken in, pages looked at past the count
  cap are not drawn but keep their looks and their outline, the cap on all
  images; the report says each.
- `test_without_a_pdf_the_viewer_draws_outlines_and_says_why`.
- `test_the_viewer_from_the_vision_records_alone`: no activity log, looks
  from `reads.json`, named by file.
- `test_a_viewer_that_cannot_be_made_is_one_line`: the rest of the report
  stands, the line names no absolute path.
- `test_the_viewer_script_parses`: `node --check` on the scripts (skipped
  where node is not installed).
- `test_a_transcript_without_its_activity_log_lists_each_answer`,
  `test_a_suite_summary_without_its_activity_log_lists_each_answer`,
  `test_no_question_and_no_answer_is_one_plain_line`: the follow-up.

Edited in `test_ev17_run_report.py`: `test_pages_drawn_behind_the_boxes_
when_the_pdf_is_found` counted the text `data:image/jpeg;base64,`, which
the viewer's script now names once more; it counts part 4's `<image
href="data:image/jpeg;base64,` instead (still 3).

Checked by hand (not in the suite): a scripted click-through in headless
Chrome on a made-up report (open from a grid square and from part 4, choose
a box, a listed look and a call number, arrow keys, Shift-arrows, the
disabled buttons at the ends, Esc, Close, Back, the jump to part 2) and
screenshots at 1400 and 512 pixels wide (the zoom box sits on the PDF's own
rectangle at the same place). Also built over saved runs kept outside the
repository: every report built with its viewer, every look with its box,
every call with its agent and step.

## Noted, not fixed

- At a 900-pixel image a 36 by 24 inch sheet shows where the boxes are,
  not its small print; the reader opens the PDF for that.
- A box's number can sit on another's where many looks start at one corner
  (the labels stack as in part 4; the list beside the page names each).
- The viewer has no zoom of its own; the browser's zoom works.

## For the lead: what a run on the hosted platform should compare

On one saved suite task folder with its documents: open three squares of
the grid (a page looked at, a page only read as text, a page never opened)
and check each box against the zoom and tile views in part 2; step through
the pages looked at with Shift and the right arrow and see the count in
the bar go up by one; and note the size and the time to build for a
40-sheet run, against the table above.

# EV-17: the turn's image calls in the report (asked in the review of 1.2.0)

## What was asked

The report neither read nor explained the turn's image-call record. From
1.2.0, `keep_work.image_call_facts` puts `image_calls` on every
`turn_end`, `image_limit` when a ceiling was set
(`GEOTECH_TURN_MAX_IMAGE_CALLS`) and `image_calls_refused` when the
ceiling refused any. The report's "Vision calls" counts the side calls that
answered (`model_end`), so the two numbers differ and nothing said why.
Wanted: the turn's image calls, its ceiling and the refused calls beside
the step cap the report already lists; one plain sentence where "Vision
calls" is shown saying what each number counts; a record without the keys
(a 1.1.0 run) shows what it showed, byte for byte.

## What was built, and where

`packages/aec-agent-core/aec_agent_core/run_report.py`:

- `_image_facts(turn_end)`: `(image_calls, image_limit,
  image_calls_refused)`, or `None` when the record keeps no count (or a
  count that is not a number); a ceiling or refused count without a count
  is ignored.
- `Report._image_total` and `_totals`: where any turn keeps the count, an
  "Image calls" row right under "Vision calls" (`data-total="image_calls"`):
  the sum, "images sent to the vision model for a separate reading, as the
  turns counted them", the ceiling or ceilings, the refused calls, and
  "counted in k of n turns' records" when some turns lack it; under the
  table, `IMAGE_COUNTS_NOTE`, the one sentence on what each count counts.
- `_notes`: per turn that keeps the count, after the step-cap line, "image
  calls: 12 of a ceiling of 40; 3 more refused at the ceiling", or "image
  calls: 5, no ceiling set".
- README, "Reading a saved run": one paragraph.

## Calls made

- **What the sentence says, from the code.** `TurnScope.take_image_call`
  counts each image once before it is sent (`vision_engine.analyze_image`;
  a retry of that image is not counted again) and counts a refused one
  apart. The report counts every side call with a `model_end`: the vision
  probe's calls (`vision_probe`, labelled `kind: probe`, not counted on the
  turn) are in it, a failed or given-up image is not, and an ask given up
  at its time limit that answered after all is a second answer for one
  image. The sentence names those three cases.
- **Always shown where the record has it**, also at 0 and with no ceiling:
  the count is kept "so a sensible number can be chosen from real turns"
  (the 1.2.0 release note), and a 0 on a turn that looked at pages is
  itself worth seeing.
- **Gated on the key, not on a version.** Nothing is added unless a turn's
  `turn_end` holds a number under `image_calls`.

## Tests (`packages/aec-agent-core/tests/test_ev17_run_report.py`)

Both fail on the base (no row, no line) and pass here.

- `test_image_calls_beside_the_step_cap_and_what_each_count_counts`: the
  made-up run written by the real writer inside a `TurnScope` with a
  ceiling of 4, 4 images counted and 2 refused: the record's `turn_end`
  holds the three keys; the totals show 3 vision calls and 4 image calls,
  the row right under "Vision calls" with its ceiling and refused count, the
  sentence once, under the table; the notes hold the step-cap line and then
  the image-call line.
- `test_a_record_without_image_calls_shows_what_it_showed`: the made-up run
  with no scope (no `image_calls`, as a 1.1.0 record) shows no row, no
  sentence, no line; the same record with `image_calls: 5` added differs
  from it only by the row, the sentence and the notes line (removing those
  three gives the first page byte for byte); a count that is not a number
  (`"many"`, `true`) or missing beside a ceiling and a refused count gives
  the first page byte for byte.

Checked by hand (not in the suite): the module before this change and
after it, on two made-up runs, one with its PDF: every page the same, byte
for byte, with the "Made" time held still. Also rendered with the old
module and the new one over saved runs kept outside the repository: every
page the same.

## Noted, not fixed

- None.

## For the lead: what a run on the hosted platform should compare

On one 1.2.0 turn with `GEOTECH_TURN_MAX_IMAGE_CALLS` set low enough to
refuse: the totals' image calls against the "image calls this turn: k of N"
status the page showed, and the refused count against the line under the
answer.
