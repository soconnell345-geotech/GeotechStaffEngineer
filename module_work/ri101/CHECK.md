# RI-101 part (1) and RI-170: the question list in the ingest run, posts matched

Branch `claude/ri-101-170-question-list-run`. Parts (2) and (3) of RI-101
are not in scope (a live run, and the reviewer's two notes with no live case).

## RI-101 part (1): the list wired into the ingest run

**What changed.**

- `report_ingest/graph.py:226` `QUESTION_LIST_ENV = "GSE_REPORT_QUESTIONS"`:
  the existing `question_list.LIST_ENV`, spelt in the graph so a run with it
  unset never imports `question_list` (a test pins the two equal).
- `report_ingest/graph.py:1068`: after `_run` returns the finished record,
  while the document is still open, `_question_list` (`graph.py:1117`) runs
  only when the setting is set. It hands the bound reports' records (by
  bound_id, from the children just read), counts the list's calls in the
  run's cost (`_Spend.mark` / `settle`), and appends its QA entries to the
  record's.
- `report_ingest/question_list.py:2396` `ingest_sheet`: reads as the harness
  does (`pages_to_read`, `page_texts` off the open document, printed numbers
  from `doc.page_map()`, `run_questions` on the run's main engine), writes
  `report.questions.xlsx` (`SHEET_NAME`, `:2314`) beside
  `report.record.json`. Every failure (list unreadable, posts unreadable,
  an engine error, a sheet that will not write) is a `not_run` QA entry with
  `where="question_list"`, never an exception; each of the run's warnings
  is a QA note.
- `question_list.py:2321` `_ReplayEngine`: each structured reply is kept in
  `items/question_list.calls.json` under a digest of its request (messages,
  schema, `PROMPT_VERSION`); a resumed run sending the same request gets it
  back with no call. A reply with no answer is not kept, so it is asked
  again next run.
- The narrative's own answers are untouched (a test compares
  `record.narrative` with and without the setting).
- The harness (`module_work/report_ingest_harness/run_question_list.py`)
  takes `--posts` (default `GSE_REPORT_POSTS`) so it writes the same sheet.

**Unset, nothing changes.** `test_graph_model_first.TestOffIsToday` (the
snapshot of master's requests and files) passes unchanged, as does the
fresh-interpreter import test; a new test makes `_question_list` raise and
runs the graph with the setting unset (no extra call, no sheet, no
`items/question_list.calls.json`, no QA entry).

**Made-up before/after.** Before: `out/` holds `report.record.json`,
`report.summary.md`, ... and no sheet. After, with
`GSE_REPORT_QUESTIONS=/private/q.xlsx`: also `out/report.questions.xlsx`
(Answers tab: `is_study` yes, `office_name` Varnholow, `plan_pages` from
code), the record's `document.model_calls` up by 2.

## RI-170: the list of posts for the post-name row

**What changed.**

- The code map names the row: `"post_name": "<Key>"`
  (`question_list.py:445`), checked to be a String row the report text
  answers; `QuestionList.post_key` holds it.
- `GSE_REPORT_POSTS` (`POSTS_ENV`, `:2145`): a JSON file (a list of objects,
  or `{"posts": [...]}`) or a CSV with a heading row. Columns read: `name`
  (required) and `full_name`, headings matched without regard to case, a
  space or dash read as `_`; every other column ignored (`load_posts`,
  `:2182`).
- `match_post` (`:2237`): names folded (lower case, punctuation and runs of
  spaces to one space); the name exactly, then the full name exactly, then
  the best `difflib.SequenceMatcher.ratio` of the printed name against each
  post's name or full name, matched when at least `POST_SIMILARITY = 0.85`
  (`:2150`). Two posts found at one step, or two equally close (equal best
  ratio), match none.
- `apply_posts` (`:2278`): matched, `Answer.value` is the post's name,
  `Answer.printed` the printed name, `Answer.matched_by` "exact name", "full
  name" or "close spelling (similarity 0.95)", and the note says
  `as printed: 'Varnholow'; matched to the post Varnhollow (Varnhollow Survey
  Station) by close spelling (similarity 0.95)`. Not matched, the value stays
  as printed, the note says why, and the record gets an open QA note
  (`kind="note"`, `rank="open"`, `where="question_list.<Key>"`).
- With no posts file the row is answered as today. A posts file given but no
  `post_name` in the code map is a `not_run` QA entry.
- `answer_sheet(posts_name=...)` adds a "List of posts" line to the Run tab
  only when given (the harness's sheets without posts are unchanged).

**Made-up before/after.** Printed "Varnholow". Before: Answer "Varnholow".
After, with a posts file of Quellbrook / Varnhollow / Tessidor: Answer
"Varnhollow", note "as printed: 'Varnholow'; matched ... by close spelling
(similarity 0.95)". Printed "Ostravel": Answer "Ostravel", open QA note
"matches no one post ... kept as printed, unmatched".

## Tests

`report_ingest/tests/test_ri101_question_list_run.py`, 29 tests, all made-up
(a three-row list, a code map, three invented posts): loading posts (JSON,
wrapped JSON, CSV, refusals), every match step, ties and duplicate names,
`apply_posts` in place, the code map's `post_name` checks, the replay cache,
and seven graph runs (unset: nothing; set: the sheet beside the record, cost
+2 calls, narrative unchanged; unmatched post an open QA entry; no posts
file; an unreadable list; a bad posts file; an engine error; a resumed run
asks nothing). On master the file fails at collection (the names it imports
do not exist).

Runs (this session's venv, planlens 0.14.0):

- `pytest -q report_ingest/tests/test_ri101_question_list_run.py`: 29 passed.
- `pytest -q -n 4 test_graph_model_first.py test_question_list.py
  test_ri195.py test_record_fields.py webapp/tests/test_public_text.py`:
  141 passed (the snapshot comparison ran, not skipped).
- The brief's gate, `pytest -q -n 4 report_ingest
  webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  GATE_RESULT

**Prompt versions.** No model-facing text changed: `run_questions`' calls
are master's (`question_list.PROMPT_VERSION` stays `ql-1`), and no reader
prompt moved. The post matching is code after the call.

## Release note

Nothing changes unless a setting is set. With `GSE_REPORT_QUESTIONS` naming
the owner's question list, each ingested report also gets
`report.questions.xlsx` beside its record, the list's two calls are counted
in the record's cost, and any failure of the list is a `not_run` QA entry on
the record. With `GSE_REPORT_POSTS` naming a list of posts as well, the list's
post-name row (named by the code map's `post_name`) answers with the matched
post's name and keeps the printed name in its note; an unmatched name is an
open QA entry. The record's other values and the narrative's answers do not
change.

## To measure live

1. On Foundry, with the owner's list and code map in the project folder and
   `GSE_REPORT_QUESTIONS` set: ingest two reports; check
   `report.questions.xlsx` sits beside each `report.record.json`, its Run tab
   shows two calls with tokens, and the record's QA has no `question_list`
   `not_run` entry. (RI-101 part 2's schema-acceptance and size checks ride
   on this run.)
2. Rerun one report with resume on: the Run tab's calls show zero tokens and
   the engine's notes say "taken from the run's folder".
3. Add `"post_name": "<the post row's Key>"` to the code map and set
   `GSE_REPORT_POSTS` to an export of the post list (columns `name`,
   `full_name`; the Ontology object's other columns are ignored). On every
   report, count the post rows matched by each way and the open notes; read
   every "close spelling" match and every open note by eye, and say whether
   0.85 is right (no wrong close match; misses only where the report names
   no post).

## Noted, not fixed

- The list runs on the run's main engine, not the narrative stage's engine
  when `stage_engines` names one. Worth a setting only if the owner wants the
  list on another model.
- Bound reports do not get sheets of their own; the parent's sheet reads
  their records (as the harness does).
- Matching folds case and punctuation but not accents; a printed name with an
  accent the list lacks falls to the close-spelling step (still matched when
  close).
