# RI-102: the record keeps what the question list's code rows need

Branch `claude/ri-102-record-fields`. Every value below is one the pipeline
already reads or builds. No new question to a model, no prompt text changed.

## Which rows answer now (after the review's fixes)

The first version of this file said rows 2, 25, 27 and 29 answer once the
lead maps them. That was written without the private list and was not true:
the review ran the real rows and only row 25 answered. After the fixes, run
locally on the real v2 and v3 rows with the test's made-up record (nothing
from the list is in the repo):

| v2 row (v3) | map it to | answers? |
|---|---|---|
| 2 (2), yes/no | `document_type:<types that mean yes>` | **yes**, when triage's type is one listed; else not said (code never answers no). With no types listed it is not said |
| 25 (26), a count | `caption_count:appendix_figures` or `caption_labels:appendix_figures` | **yes** (2 on the made-up record) |
| 27 (28), a list of records | `appendix_list` | **yes**, every part filled (two of its parts had names the reading did not know, so they stayed empty) |
| 29 (30), a list of records, one per investigation | none | **no**: left for a follow-up (below) |

Which document types mean yes on row 2 is the owner's call and goes in the
code map, not the code. The review suggests `geotechnical report|recommendation
letter|report addendum`; whether "report appendix or figure(s)" and "partial
report" also mean yes is not settled.

**Row 29 is left unanswered.** It asks for every investigation, current
and prior, one record each. Code can list only the prior reports that are
bound in, so on a report that describes an earlier investigation without
binding it, code's list would be short and would read as a disagreement.
Filling the parts also needs calls the record cannot settle: which firm when
the logs name more than one; whether a prior report's year is its printed
date or its logs' years; whether a kind with no log is 0 or not printed. `investigation_year` and
`investigation_firm` still return one value each, which the row refuses. The
record now holds what a reading would need (`field_work`, the bound reports'
own records through `record_dir`), so an `investigations` reading is a
follow-up, not part of this branch.

## What changed, and why

The RI-90 code-only run could not answer rows 2, 25, 27 and 29 because the
record did not keep five things. Rows 19 to 21 needed a bound prior report's
own record to hold its logs. Each one now:

| Value | Where it comes from | Kept on the record as | Code |
|---|---|---|---|
| triage's document type | triage's answer (fresh, or `triage.json` on resume) | `DocumentFacts.document_type` | `report_ingest/graph.py:1244`; model `report_ingest/model.py:952` |
| a bound report's document type | its identity call (already on `BoundReport.document_type`) | the child's `DocumentFacts.document_type` | `report_ingest/graph.py:2304` |
| each page's printed page number | planlens' page map (`summary.printed_page`), already read by the graph | `DocumentFacts.printed_pages` (`{0-based page: "as printed"}`; a child keeps its own pages) | `report_ingest/graph.py:1211`, `:2304`; `report_ingest/record_facts.py` `printed_pages` |
| the outline's appendix list | planlens' `document_outline` (the parent) / `outline_from` (a child), already built | `DocumentFacts.appendices` (`AppendixEntry`: number, title, printed page, PDF page) | same lines; `record_facts.appendices`; model `report_ingest/model.py:862` |
| the appendix figure caption labels | `count_captions`, which returned them and `NarrativeBuilder.build` dropped | `NarrativeFacts.caption_labels` (`tables`, `figures`, `appendix_figures`) | `report_ingest/narrative_reader.py:1783`; model `report_ingest/model.py:2769` |
| year and firm of the report's own investigation | the record's own logs: four-digit years in `date_started`/`date_finished`; header fields named firm, company or consultant, but not one named for the driller or a person (drill, excavat, foreman, operator, contractor), and not a value like "NA" | `ReportRecord.field_work` (`FieldWork`: years, dates, firms, holes, pages) | `report_ingest/graph.py:1396`, `:2419`; `record_facts.field_work`; model `report_ingest/model.py:2784`, `:3409` |

A log on the pages of a bound report that is not a volume of this one does
not count toward `field_work`. The question list places logs the same way.

All five fields are optional and are left out of the written JSON when empty.
They use the same `_without_empty` pattern as `Investigation.qualifier` and
`DocumentFacts.resumed`. So a record saved before this change loads, writes
byte for byte as it did, and exports the same: the summary, the library page
and DIGGS read none of these fields. `SCHEMA_VERSION` stays 4.0, and the
comment above it says why (`report_ingest/model.py:140`).

**Under model first, the caption labels are kept too.** With checks set to
flag, the narrative leaves `counted` empty, because no code-read value
becomes an *answer*. The labels are a reading of the pages kept beside the
answers, like `narrative.pages`. They are not an answer, so both arms keep
them. That matters because model first is the default.

**Wiring the code rows** (`report_ingest/question_list.py`), five new
readings for the private code map:

- `document_type`: triage's type. On a closed list it takes the nearest
  option, or is not given if no option is near. On a yes/no row,
  `document_type:<type|type>` names the triage types that mean yes: yes when
  triage's type is one of them, else not given. A type outside triage's
  vocabulary is refused when the map is read.
- `caption_labels:<tables|figures|appendix_figures>`: the labels. The row
  gets a list, a count or text, depending on its Data Type.
- `appendix_list`: the document's own list of appendices, as records
  (number or letter or label, title or name, pdf_page or first_pdf_page,
  printed_page) or as text.
- `investigation_year`, `investigation_firm`: from `field_work`, merged with
  the bound volumes' records. On a record saved before the field, the same
  reading is computed off its logs.
- `caption_count` now also reads `appendix_figures`. It falls back to the
  length of the kept labels when `counted` lacks the kind (a model-first
  record).
- `page_labels` printed page: `code_answers` / `run_questions` read the
  record's `document.printed_pages` when the caller passes none (or an empty
  map). A caller's own map still wins.
- Rows 19 to 21: `code_answers(..., record_dir=)` and `run_questions(...,
  record_dir=)` load the bound reports' own records from
  `BoundReport.record_path` (new `bound_records()`), so RI-101's wiring does
  not have to collect them by hand. The graph has handed a bound range's log
  pages to that report's own pass since RI-14 (2026-10-23). The 2026-10-04
  records RI-90 ran on predate RI-14, which is the likely reason R13's child
  holds no logs. A new test pins that a triage-drawn prior report's log is
  in its own record and not the parent's, and that `prior_hole_count`
  answers from it through `record_dir`.

The rows themselves are named only in the private list. The lead maps
rows 2, 25 and 27 in `<list>.code.json` as the table at the top says; row 29
stays unmapped.

## Made-up before/after

The same made-up record, before (as saved before RI-102) and after (with the
fields), run through nine made-up rows
(`test_rows_the_record_could_not_answer_now_answer`):

| row (made up) | reading | before | after |
|---|---|---|---|
| doc_kind | `document_type` | not said ("triage's document type is not on the record") | "geotechnical report" |
| is_soils_report (yes/no) | `document_type:geotechnical report\|report addendum` | not said | true |
| back_figures | `caption_labels:appendix_figures` | not said | ["figure A-1", "figure A-2"] |
| back_figure_total | `caption_count:appendix_figures` | not said | 2 |
| appendix_names | `appendix_list` | not said | [{A, Boring Logs, pdf 3}, {B, Laboratory Data, pdf null}] |
| appendix_index (label, title, first_pdf_page) | `appendix_list` | not said | [{A, Boring Logs, 3}, {B, Laboratory Data, null}] |
| field_year | `investigation_year` | not said | 2030 (note: the earliest of 2030, 2031) |
| field_firm | `investigation_firm` | not said | "Invented Geo" |
| page_kinds | `page_labels` | printed_page "" | printed_page "1" |

## Tests

New: `report_ingest/tests/test_record_fields.py`, 13 tests. All 13 fail on
master: the file does not import there (`AppendixEntry`, `FieldWork` and
`report_ingest.record_facts` do not exist). One part of
`test_a_bound_prior_report_s_record_holds_its_logs` would pass on master
(the child holding BH-1, which RI-14 did). The `record_dir` half would not.

- An old record loads and writes byte for byte, and the new fields
  round-trip.
- The summary, library page and DIGGS of an old record do not change when
  the fields are filled.
- `printed_pages`, `caption_labels`, `field_work`: the logs' own years and
  firms. A prior report's logs, the drilling contractor, a "Drilling
  Company" or "Contractor Foreman" field, an "Engineer" field, "NA" and
  two-digit years are left out.
- The graph (synthetic report, today's readers) keeps the document type,
  the printed pages (the child's within its own pages), the appendix list
  A to D, the caption labels, the child's identity type, and the parent's
  field work: its consultant header field (added to the form reader's
  header fields by the test), not the driller the scripted reading names.
  The child has none.
- Rows 19 to 21: the bound prior report's own record holds BH-1, and
  `prior_hole_count` answers 1 with `record_dir` and "not said" without it.
- The rows that now answer (table above). An old record answers the year
  off its logs. A document type off the row's list is not given. A yes/no
  row is yes only for the types the map names, and not given with none.
  The new readings refuse arguments they cannot use.

Changed: `report_ingest/tests/test_model.py`, the record's top-level
property set (adds `field_work`).

**Graph snapshot re-recorded** (`report_ingest/tests/graph_off_snapshot.json`,
`write_off_snapshot`), because today's path now writes the new fields. Exactly
six digests moved: `report.record.json`, `bound/bound1/report.record.json` and
`items/item_2.json` (the narrative item's saved facts, now carrying
`caption_labels`), in both the `standard` and `rules` runs. Every request
digest and every other file stayed the same: DIGGS, summary, library page,
`reports.db`, `triage.json`, labels and the other items. `taken_with` now
names this Linux / planlens 0.14.0 run.

Runs (venv `/tmp/v`, Python 3.12, planlens 0.14.0):

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  **4406 passed, 8 skipped (pydiggs not installed), 0 failed**, exit 0.
- Consumers: `webapp/tests/test_report_library_wiring.py`,
  `test_foundry_dataset_store.py`, `funhouse_agent/tests/test_subsurface_write_diggs.py`,
  `test_report_checklist_shim.py`, `funhouse_agent/deep/tests/test_report_ingest_chat_model.py`,
  `test_report_ingest_page.py`, `test_coverage_offline.py`,
  `module_work/report_ingest_harness/tests`: **373 passed, 34 skipped, 0
  failed**, exit 0.

After the review's fixes (Windows, the app's venv, `-p no:cacheprovider`):

- `test_record_fields.py`, `test_model.py`, `test_model_first_schema.py`,
  `test_graph_model_first.py`: **285 passed, 0 skipped**, exit 0. The graph
  snapshot did not move: the synthetic report's logs name no firm.
- `test_question_list.py`, `webapp/tests/test_public_text.py`: **39 passed**,
  exit 0.

## Prompt versions

None moved. No model-facing template text changed. `run_questions` now takes
the record's printed page numbers into the page list when the caller passes
none. That is the same data the RI-90 harness already passes from the PDF,
and the template is unchanged, so `PROMPT_VERSION` stays `ql-1`. The page
fingerprints test passes.

## Release note

None needed. No existing value of a saved record changes: the fields are new
and optional, and a record with none of them is written as before. New runs'
`report.record.json` gain the fields where they have a value.

## To measure live

- Re-run `module_work/report_ingest_harness/run_question_list.py` (engine
  `none`) on records written by a build with this branch, after the lead maps
  rows 2, 25 and 27 in the private code map as the table at the top says.
  Count how many of the 13 reports each row answers, and check each answer
  against the PDF.
- Rows 19 to 21 on R13 from a post-RI-14 run: check that the bound earlier
  report's record holds its logs (build f's review counts 13 holes there),
  and that `prior_hole_count` answers through `record_dir`.
- How often `field_work` is empty on real records, meaning the logs' dates
  or contractor were not read into the investigations. That measures the
  log readers' header fields, not this change.

## Noted, not fixed

- `printed_pages` is planlens' page-map reading. R09, R11 and R24 print none
  that it finds, so the row stays not said there. Improving that would be a
  planlens change.
- `field_work.firms` takes firm, company and consultant header fields only,
  never the driller (the review found drillers, foremen and "NA" saved as the
  firm on real records under the first rule). A report whose logs name its
  firm only in a drilling-contractor field now has no firm there, which is
  "not said", not a wrong firm. The report's own geotechnical firm is the
  narrative's `geotechnicalEngineerFirm`, a model answer already on the
  record, and is not mixed in. Two-digit years ("3/8/31") are not read, since
  the century is not printed.
- With `ingest_bound` off there are no bound documents, so every log in the
  record counts toward `field_work`.
- The harness `run_question_list.py` still collects bound records itself; it
  could pass `record_dir=` instead.
