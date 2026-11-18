# EV-48: the "found only in the working notes" line reads only what was said

## What the item asked

The review suite marks a failed `with_files` check "found only in the run's
working notes" when the check would have passed on the run's working files
(`.scratch/` and `kept/`). Those files were read whole as text, so a mark
number that stood only in a box's coordinates (`[648, 189, 755, 204]`), in
the prompt a reading was asked with, or in a copied schedule row passed, and
the line said a run had seen problems it never stated (brief 8's trace
review). Read only the readings' text. The line is information only.

## What was built

All in `packages/aec-agent-core/aec_agent_core/review_eval/checks.py`:

- One block of named fields, read from the writers' code:
  - `SAID_FIELDS = ("analysis",)` and `SAID_PART_LISTS = ("tiles",)`: what a
    vision reading said. A kept reading's result (`vision_tools._keep_reading`,
    the full JSON result of `analyze_pdf_page` or `render_region`) holds the
    answer in `analysis`, and a page read in tiles holds each tile's answer in
    `tiles[i].analysis`. Every other field (view, view_px, bbox, region,
    window padding, precision, the tool's notes, the `found` list and the
    `reading_note`, which are copies made in code from the answer) is not read.
  - `KEPT_READINGS_FILE = "page_readings.json"`: an entry's own fields
    (document, file id, page, view, region, when, the prompt) are not read;
    its `result` is read through `SAID_FIELDS`.
  - `READ_RECORD_FILE = "reads.json"` (`vision_tools._note_read`): document,
    page, view, tool, prompt, when. It holds no reading's words, so nothing in
    it is read.
  - A helper's kept record (`kept/*.md`, `keep_work.helper_file_text`): its
    answer (`## Its answer`) and each step's own words (`### Its own words`)
    are read, plus `SAID_FIELDS` of each tool result that is a JSON object.
    The brief it was given, the summary made in code, the open items its
    tools flagged, the index and the arguments of its calls are not.
  - `BOX_IN_TEXT`: a box as a vision answer writes one (four numbers in square
    brackets, or in brackets tagged `px`/`pixels`, as planlens'
    `frames._BOX`). It is taken out of every working text, because a vision
    answer carries its boxes inside its prose.
- `reading_said(reading)`, `working_file_said(path)` and a new
  `working_text(files)` with the same signature and return value. The file
  reading of `_files_text` moved into `_file_text`, shared, unchanged.

## Calls made

- **The read record says nothing.** `reads.json` holds only which page was
  looked at, how, and the first 100 characters of the prompt, which is a
  copied tool argument. So a mark that only an agent's question named no
  longer counts.
- **The prompt is never read**, in either file: it is what the agent asked,
  not what the reading said (the copied schedule row of the trace review sat
  in prompts like this).
- **Helpers' kept records are covered too.** The item names the two JSON
  files, but `working_text` also reads `kept/`, and a reader's record holds
  the same raw vision results and every call's arguments in full. Leaving it
  whole would leave the same false line. A result's text is taken by the
  character count in its heading (a result may itself hold text that looks
  like the record's headings); where the count does not land on the next
  heading (newlines changed on another system) it runs to the next heading.
- **Text a tool read is not "said".** In a kept record, a result that is not
  a JSON object (a document's text from `read_text_file`, say) is the
  document's text, not the helper's statement, and is not read. A vision
  answer that transcribes a schedule row is what the reading said, and does
  count.
- **Unknown means unread.** A field not named, a field added later, a JSON or
  XML working file of another writer (for example `kept/helper_*_records.json`)
  is not read. A missed line is cheaper than a false one.
- **Plain prose notes still count, minus boxes.** A working `.md`, `.txt`,
  `.csv`, `.tsv`, `.docx` or `.xlsx` that is not a kept record (a note an
  agent wrote for itself) is read whole, as before, with box notation taken
  out.
- **Older shapes.** A reading kept as plain text (a bare string, or a
  `result` that is not JSON) is its own text, minus boxes. A file that is not
  JSON, a `readings` that is not a list, entries that are not objects: read
  as far as they go, never an error (the check's existing "never raises" rule
  holds; `working_text` skips a file that raises).
- **Only files that said something are named** in the line (today
  `reads.json` was named even when it held nothing that matched).

No setting was added: this is the repair of a defect in an information line.

## Scores do not move

`score()` counts only each result's `passed` and `info`. In `run_check`,
`working_text` is called only after the check has failed (`not ok`), and its
result sets only `notes_only` and the detail text; `ok` is not touched. The
delivered files are read by `delivered_text` through the same `_file_text` as
before, with no box stripping. `test_the_delivered_files_are_read_as_before`
shows a delivered log whose mark stands in a box still passes.

## Tests

New file `packages/aec-agent-core/tests/test_ev48_working_text_said.py`
(made-up records in each writer's shape). On master 10 of its 15 cases fail;
the 5 that pass on master are the delivered-files guard and the four older
shapes that master also read (whole).

- `test_a_mark_only_in_boxes_prompts_or_copied_rows_no_longer_counts`: the
  mark in a box in the answer, in tile and zoom boxes, the view, the region,
  the `found` list, the prompt with a copied schedule row, and `reads.json`.
  No notes line; the score is 0 of 1 as before. Fails on master
  (`notes_only`).
- `test_the_same_mark_said_in_a_reading_still_counts`: the same mark in the
  answer's words gives the line, naming `page_readings.json` and not
  `reads.json`. Fails on master (master names `reads.json` too).
- `test_a_tiles_answer_counts_and_its_view_does_not`. Fails on master.
- `test_the_delivered_files_are_read_as_before`: a guard. Passes on both.
- `test_a_kept_record_counts_only_the_helpers_words`: a record built by the
  real `keep_work.helper_file_text`, with the mark in the brief, a call's
  arguments (a prompt and a `write_xlsx` row's cell), a vision result's box,
  a read document's text that imitates the record's headings, and the open
  items: no line. The mark in its own words, or in its answer, gives the line.
  Fails on master.
- `test_the_named_fields_match_their_writers`: the real
  `vision_tools._keep_reading` and `_note_read` write the two files under a
  bound folder; only the answer is read. It also keeps the file names and
  headings in step with their writers. Fails on master.
- `test_an_older_or_other_shape_never_raises` (8 cases): bare strings, a bare
  list, a plain-text or object `result`, junk entries, a non-list
  `readings`, an unknown field, a truncated file. 4 fail on master.
- `test_other_working_files`: `reads.json`, a records JSON and an XML file
  say nothing; a plain note counts minus boxes. Fails on master.

The existing `test_review_checks_together.py` is unchanged and passes (its
readings are bare strings, an older shape that is still read).

## Noted, not fixed

- A vision result cut by a result cap before it reached a helper is not
  valid JSON in the kept record, so its answer is not read there (the same
  reading is usually in `page_readings.json` whole).
- Four numbers in square brackets in an answer's prose (a list of four
  marks, `[406, 407, 408, 409]`) are taken as a box and dropped, as planlens
  takes them.

## For the lead

On the hosted platform, re-score a saved brief 8 or brief 9 run offline
(`rescore=True`) before and after, and compare the RESULTS.md section "In the
run's working notes, not in what it delivered": the coordinate-only flags of
the trace review (marks 204 and 509 that stood only in boxes and schedule
rows) should drop out, and a flag whose reading stated the problem in its own
words (the A-302 "verify" note) should stay where the check's words match it.
The scores must be identical.
