# RI-207: the question list's row with one record per investigation

Branch `claude/ri-207-investigations-reading`, from master 50ef609. One file
of code changed (`report_ingest/question_list.py`) and one test file added.
Nothing a model is sent changes, and nothing changes on merge until the lead
maps the row in the private code map.

I could not run the real row: the list is private. Everything below was run
on a made-up row. **The lead should run the real row before merging** (the
call is under "To measure live").

## What changed, and why

Code can now answer the row "one record per investigation, current and
prior" with what the record holds, and says its list may be short.

| What | Where |
|---|---|
| New code-map reading `investigations` | `report_ingest/question_list.py:1378` (`_investigations`), registered at `:1792` |
| One investigation's parts from its logs and field work; a part code cannot settle is left out with the reason | `:1314` (`_investigation`) |
| Part matching by words, in any order, singular or plural | `:801` (`_part_field`), used by `_fill` at `:824`; the words at `:1233` (`_INVESTIGATION_WORDS`) |
| The current-or-prior part is found by its two options, not its name | `:1256` (`_status_part`), `:1266` (`_investigation_fields`) |
| The partial flag (RI-184's `partial` entry at `document.completeness`) | `:1283` (`_partial`) |
| The lower-bound mark on an answer | `Answer.lower_bound`, `:702` |
| The comparison for a lower-bound list only | `:2018` in `compare`, `:1970` (`_compare_lower_bound`), `:1942` (`_unmatched`), `:1920` (`_investigation_match`) |

**What code lists.**

- The report's own investigation, when the record holds a log of it or
  `field_work`: this file's logs outside any bound prior or data report,
  plus its bound volumes' records (the same placing `hole_count` uses).
- One record for each bound report of kind `appended_prior_report`, from
  that report's own record (`bound=` or `record_dir=`, as `prior_hole_count`
  reads it), plus any log of the parent placed on its pages, counted and
  read for the year and the firm with the prior record's own logs.
- A prior report whose record holds no log is not listed. One whose record
  was not given but whose pointer counts logs is listed as prior with empty
  parts and the reason.
- With nothing to list, the row is "not said".

**Parts.**

| Part | Given | Left empty, with the reason in the note |
|---|---|---|
| year | the year in the logs' printed dates, when every dated log prints the same four-digit year | no log prints a four-digit year. Or the logs print more than one year, or a four-digit year beside dates with none (a two-digit year such as '99): then the counts are empty too, and the note gives the years printed, the holes of each kind under each year, how many holes print a date with no four-digit year and how many print no date. A prior report's printed date is never used; the note quotes it and says why |
| consultant | the one firm the logs name (firm, company or consultant header field) | the logs name two or more (the note names them), or none |
| each count | the holes of that kind; 0 only where the record holds a log of another counted kind, none of this kind, and no partial flag | the copy is flagged partial and holds none of the kind; the record holds no log of any counted kind; a bound record with logs was not given; the logs print more than one year, or a four-digit year beside dates with none (see year) |
| current or prior | the row's own option for it | the row has no two-option part with a word for each |

The counts default to `boring`, `test_pit` and `cpt` logs. The code map can
name others so the row counts as rows 19 to 21 do:
`investigations:borings=boring,well`.

**Part names.** A part takes a field when it is named for it exactly, or
when one word of its name is one of that field's words: year (year, yr);
consultant (consultant, firm, company, engineer); borings (boring, borehole,
bh, drillhole); test pits (pit, testpit, tp); cones (cone, cpt, cptu,
sounding). A part whose words name two fields takes neither. The status part
is found by its options: the one closed-list part with an option for
"current" (current, present, this, new, subject, latest) and another for
"prior" (prior, previous, earlier, past, old, historic, historical, former).
The other readings' part matching is unchanged: the wider matching runs only
when a reading passes its words.

**The comparison** (a "Both" row, this reading only). Each investigation
code lists needs a record of its own in the model's list: the same
current-or-prior, the year equal where both give one, each count equal where
both give one. Investigations only the model lists are not a disagreement,
and the note counts them. The consultant is not compared (two spellings of
one firm are common; it is on the sheet for the owner). The From column
reads "both: agree" or "both: disagree" as on other rows; the note carries
"code (may be short): ...; code's note: ...", so the reason for each empty
part reaches the sheet. An empty part is not compared. A model answer of
"not said" beside a code list is still a disagreement, as on every other
row.

**Lower-bound mark.** Code's note starts "code's list may be short (a lower
bound): ...", and the answer carries `lower_bound=True`. No sheet column was
added.

`investigation_year` and `investigation_firm` are untouched.
`PROMPT_VERSION` stays `ql-1`.

## Made-up before/after

A made-up report with borings B-1 and B-2 and pit TP-1 (dated March 2031,
consultant field "Invented Geo") and one bound prior report whose record
holds boring BH-1 and cone C-1 (August 2018, "Older Geo"). Row parts: year;
consultant; borings; test_pits; cones; status [current, prior].

| | Answer |
|---|---|
| Before (master) | the code map refuses `investigations`: not one of code's readings |
| After | `[{2031, Invented Geo, 2, 1, 0, current}, {2018, Older Geo, 1, 0, 1, prior}]`, marked a lower bound |

Compared with a model's list:

| The model lists | From |
|---|---|
| the same two, and a third from 2009 the text describes | both: agree (note: 1 investigation only the report text lists) |
| the current one as 2030 | both: disagree |
| the current one with 3 borings | both: disagree |
| the current one with the year empty and "2 borings" | both: agree |

## Tests

New: `report_ingest/tests/test_question_list_investigations.py`, 14 tests.
All 14 fail on master (run with `question_list.py` stashed: 14 failed). One
of them, the guard that another list-of-records row still compares as
before, fails on master only because `Answer.lower_bound` does not exist
there; its assertion about the old rule would hold.

They cover: the own investigation alone; one bound prior report (given by
`bound=`, found by `record_dir=`, and not given); two firms on the logs; a
prior report with undated logs; a partial copy; nothing to list; part names
in other words and another order, and a part naming two counts; the code
map's kinds argument and the arguments it refuses; a row that is not a list
of records; the model listing one more prior investigation (agree); a
different year, count or status, and an empty list (disagree); two prior
reports each needing a record of their own; the sheet's note.

Runs (venv `/tmp/v`, Python 3.12), each judged by pytest's exit code:

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  **4888 passed, 23 skipped (pydiggs not installed), 0 failed**, exit 0.
- Consumers outside `report_ingest` (the only one is
  `module_work/report_ingest_harness/run_question_list.py`):
  `module_work/report_ingest_harness/tests` and
  `webapp/tests/test_backlog.py`: **193 passed, 34 skipped, 0 failed**,
  exit 0.

The graph snapshots did not move: the graph does not import this module
unless `GSE_REPORT_QUESTIONS` is set.

## Prompt versions

None moved. No model-facing text changed; the answer schema of a row is
built from the list as before.

## Release note

None on merge: the reading does nothing until the private code map names
it, so no saved answer changes.

For when the lead maps the row: "The question list's row with one record per
investigation is now also answered by code, from the report's own logs and
the logs of each prior report bound into the file. Code's list is a lower
bound, because a report can describe an earlier investigation it does not
bind in. Where one investigation's logs print more than one year, or a
four-digit year beside dates with none, code gives no year and no counts
for it, and its note gives the years printed and the holes of each kind
under each year: code cannot tell field campaigns of one investigation from
an earlier investigation's logs reproduced in the report. An empty part is
not compared. Where code lists an investigation, the row's From column
changes from report text to 'both: agree' or 'both: disagree': it disagrees
when an investigation code lists has no match in the model's list, or when
the model answers 'not said' beside code's list. Where the record holds no
log (R05 of the six open saved records), code says 'not said' and From stays
'report text'. The compared answer's note carries code's own note, with the
reason for each empty part. The model's answer itself is unchanged."

## What an export would need

No record field was added. `Answer.lower_bound` is on the in-memory answer
only; an export of the answer sheet's data would need one more boolean
beside the note.

## To measure live

1. Map the row in the private code map, under `"readings"`:
   `"all_investigations": "investigations"`. `load_questions` then checks
   it. If the row's Answered by is "Report text (model)", the map is
   refused: the row must be "Both, compared" (or "Code"). The longer form
   `"investigations:borings=boring,well"` also counts a well logged in a
   hole of its own as a boring, as the boring-count row does; it matters
   only on a record with such a well (a well logged in a boring is that
   boring already), and on the six open saved records both forms give the
   same answers.
2. Code only, on saved records, no model call:

   ```
   python -m module_work.report_ingest_harness.run_question_list \
       --list <the list> --engine none --no-text --only <IDs>
   ```

   Or one row directly:

   ```python
   from pathlib import Path
   from report_ingest import question_list as ql
   from report_ingest.model import ReportRecord

   qlist = ql.load_questions(LIST)            # with its code map beside it
   KEY = "<the row's Key>"
   row = qlist.by_key()[KEY]
   print(ql._investigation_fields(row))       # which part took which field
   folder = Path(RECORDS) / "<ID>"
   record = ReportRecord.model_validate_json(
       (folder / "report.record.json").read_bytes())
   code = ql.code_answers(record, qlist, record_dir=folder)[KEY]
   print(code.status, code.value, code.note, sep="\n")
   ```

3. The comparison against a model answer already paid for, no model call:
   the saved replies are in `<folder>/items/question_list.calls.json`.

   ```python
   import json
   kept = json.loads((folder / "items" / "question_list.calls.json")
                     .read_text("utf-8"))
   said = next(v["parsed"][KEY] for v in kept.values()
               if KEY in v["parsed"])
   model = ql.Answer(value=said["value"], status=said["status"])
   print(ql.compare(row, model, code).source)
   ```

Check first that `_investigation_fields(row)` shows all six fields (year,
consultant, borings, test_pits, cones, status). A missing one means the
real part's name or pick words are outside the lists above: add the word to
`_INVESTIGATION_WORDS`, `_CURRENT` or `_PRIOR` (words only, never the list's
own part names). Then, per report, check against the PDF: each count, each
year, and every "both: disagree".

Things worth counting over the 13 reports: how many list a current
investigation; how many consultants are empty (field_work reads only a firm,
company or consultant header field, so this may be most of them); how often
an investigation gets no year and no counts because its logs print
several years.

## Calls made

- **Year when the logs span two years**: the lead decided on 2026-11-13:
  several printed years leave the year and counts empty, with the years and
  holes per year in the note. The same holds where a log prints a date with
  no four-digit year beside logs that print one. Not split into one record
  per year. (The branch first gave the earliest year, as
  `investigation_year` does; `investigation_year` is unchanged.)
- **Consultant when no log names a firm**: empty. The other reading: the
  firm the bound report prints for itself (`BoundReport.firm`) or the
  narrative's firm. Both are model answers, and RI-102 kept them out of
  `field_work`.
- **When the current investigation is listed**: only when the record holds
  a log of it or field work. The other reading: always list a current one
  with empty parts. That would disagree on a desk study, where the model
  rightly lists none.
- **A bound prior report with no logs is not listed**, since a prior
  document is not always an investigation. The other reading: list every
  bound prior report.
- **0 for a kind** needs a log of another *counted* kind (borings, pits,
  cones as mapped). A record with only hand-auger or well logs gives no 0.
- **A partial copy still gives the counts it holds**, with a note that a
  count may be short. The brief empties only the kinds it holds none of.
- **The consultant is not compared.** The brief's match rule names status,
  year and counts only.
- **Numbers in the model's text**: the row's parts are text ("as printed"),
  so "2", "2 borings" and "none" are read as numbers; text with no number,
  or two different numbers, is not compared (not a disagreement).
- **No status part found**: the reading still answers with that part empty
  and a note, and the comparison then matches on year and counts alone. The
  other reading: refuse the row.
- **The partial flag** is any `partial` QA entry whose `where` starts
  `document.completeness`. If RI-185 (in flight) puts its new entries under
  another `where`, `_partial` needs that name too.
- **The wider part matching is not applied to `appendix_list`,
  `bound_documents` or `page_labels`**: it could fill a part on a real row
  that is empty today, and I cannot see those rows.

## Noted, not fixed

- A neighbouring site's investigation that is bound in under a kind other
  than `appended_prior_report` (a data report, "other") is not listed. The
  model lists it and the lower bound absorbs it.
- Two prior investigations bound as one report give one record.
- `module_work/report_ingest_harness/run_question_list.py` still collects
  bound records itself and does not pass `record_dir=`; it works as is.
- Two-digit years on logs are not read as years (RI-102's rule); beside a
  four-digit year they now leave the year and counts empty (review fix 1).

## Review fixes (2026-11-13)

The review's verdict was "fix first". Fixed on `fix/ri207-review`; lines
are `report_ingest/question_list.py` on the fix commit, before the merge
with master.

1. **Several printed years** (`_investigation`, `:1373`; the note,
   `_several_years`, `:1325`; a log's years, `_printed_years`, `:1297`).
   When the logs taken as one investigation print more than one four-digit
   year, or a log prints a date with digits but no four-digit year beside
   logs that print one, the year and all three counts are left empty. The
   note gives the years printed, the holes of each kind under each year,
   how many holes print a date with no four-digit year and how many print
   no date, and says code cannot tell field campaigns of one investigation
   from an earlier investigation's logs reproduced in the report. It holds
   for the current investigation and each prior report alike. Where every
   dated log prints one year, nothing changes. An empty part compares as
   not given.
2. **The reasons reach the sheet** (`_compare_lower_bound`, `:2044`): on
   the lower-bound path only, the compared answer's note carries "code (may
   be short): ...; code's note: ...".
3. **A prior report's logs in the main record** (`:1526`): its year and
   firm are read off its own record's logs and the parent's logs on its
   pages together (`field_work` over both), also when its record is not
   given.
4. **The repeated merge** (`_field_work`, `:1166`) now calls `_work_of`;
   the behaviour is the same.
5. **This file**: the release note corrected, and the year rule added to it
   and to "Calls made".

Tests: six more in `report_ingest/tests/test_question_list_investigations.py`
(from `:276`; 20 in all): logs of two years, current and prior; a two-digit
year beside a four-digit one, and two-digit years alone as before; all logs
of one year as before; empty parts compare as not given; the reasons on a
Both row; a prior report's logs in the main record with its own record
empty or not given.

Measured on the six open saved records with the row mapped, no model call:
R16, R28, R30 and R21 (its current investigation and its prior report) now
carry no year and no counts, with the years and the holes per year in the
note; R15 and R05 are unchanged. With the row unmapped, all 162
row-answers are the same as master's and as the branch's before the fixes.
