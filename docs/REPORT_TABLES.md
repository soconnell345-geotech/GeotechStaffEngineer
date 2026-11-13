# The report tables

Every time the report ingest reads a geotechnical report, it now also writes
what it found as **flat tables**: one row for the report, one row per page,
one row per standard question, one row per file, and one row per ingest. A
data pipeline reads each table as one step, and an Ontology can be built on
them. The DIGGS file, the whole record and the summary page go out beside
the tables. (BACKLOG RI-209; the code is `report_ingest/tables.py`.)

Nothing about the ingest itself changes: the same calls, the same record,
the same files in the report's folder. The tables are made afterwards, from
the files the ingest saved, so the same code also makes them for reports
ingested before this existed, without calling a model.

## Where they are written

Under one folder, `report_tables/v1/`, inside a root folder:

| Where the ingest ran | Root (default) |
|---|---|
| A folder of reports (`run_folder`) | the folder that holds the run's `reports.db` (the run's output folder, unless `db_path` names another) |
| The corpus run on the cluster (`score_on_cluster`, the `ingest` stage) | `<out_dir>/ingest/`, beside its shared `reports.db` |
| The app, with a shared report library (`GEOTECH_REPORT_LIBRARY` set) | the library's own folder: ONE `report_tables/` for every report ingested into the library |
| One report on its own (the app without a shared library, a notebook call) | that report's own output folder |

On a Foundry deployment with a dataset named (below), each report's export
is also sent to that dataset, under the application's own top folder:
`GeotechStaffEngineer/report_tables/v1/...`, the same layout. (Importing
`report_ingest` names the application to the shared core, so the command
line, a notebook and the app all send under that one folder; a
`GEOTECH_SHAREPOINT_ROOT` setting, which the conversation store also reads,
names another.) In a Foundry build (a
job running `score_on_cluster`) there is no dataset hand-off: the job's
local `report_tables/v1/` folder is the interface, and the job carries it to
its output dataset as it carries everything else.

### Settings

| Setting | What it does | Unset |
|---|---|---|
| `GSE_REPORT_TABLES` | `0`, `off`, `false` or `no`: no export, and nothing at all is written (not even the run file below). A folder: the root is that folder for every report. `1`, `on`: the default root | on, at the default root |
| `GSE_REPORT_TABLES_DATASET` | a Foundry workspace dataset alias (the same kind of value as `GEOTECH_FOUNDRY_CONVERSATION_DATASET`). On a Foundry deployment, each report's export files go to it in one save | nothing is sent, and nothing of Foundry's is loaded |

In code, `ingest_report(..., report_tables=...)` wins over the setting:
`False` is off, `True` the default root, a folder that root. `run_folder`
and `score_on_cluster` pass it through under the same name.

The rule, in order: a root named by the setting or the argument; else the
folder holding the ingest's `reports.db` when the caller names one (a folder
run, the cluster stage); else, when `GEOTECH_REPORT_LIBRARY` names the app's
shared report library and the report's folder is inside it, the library's
own folder; else the report's own folder. So in the app with a shared
library, every report read into it (one folder per file) exports into the
library's one `report_tables/`; without one, a report's tables sit in its
own folder inside the conversation's. On Foundry with the dataset set, every
report's files land in the dataset's one
`GeotechStaffEngineer/report_tables/v1/` either way, whichever process sent
them.

## The folder

```
report_tables/v1/
  schema.json                              the tables, their columns, types and keys
  report/<report_key>.jsonl                one row: the report as it stands now
  report_page/<report_key>.jsonl           one row per page: what the page is
  report_answer/<report_key>.jsonl         one row per standard question
  report_file/<report_key>.jsonl           one row per file delivered below
  ingest_run/<report_key>.<run_id>.jsonl   one row per ingest, never overwritten
  diggs/<report_key>.diggs.xml             when the ingest wrote one
  record/<report_key>.record.json          the whole record
  summary/<report_key>.summary.md          the summary page
  questions/<report_key>.questions.xlsx    when the owner's question list ran
```

- One kind of file per folder, and every file in a folder has the same
  columns, so a pipeline reads `report_page/*.jsonl` as one table.
- JSON Lines: UTF-8, one row per line. Every row carries every column of
  its table, with nulls written out.
- Column types are only these: string, integer, double, boolean, timestamp
  (UTC, `2026-01-02T03:04:05Z`), date (`2026-01-02`), list of text, list of
  whole numbers. A structured value is a JSON string in a column whose name
  ends `_json`.
- **Pages are PDF pages counted from 1**, as a PDF viewer shows them
  (`pdf_page`, `pdf_pages`). The page number printed on the page is a
  separate column (`printed_page`). The record itself counts pages from 0;
  the tables do not.
- `schema.json` is written from the same definitions as the rows. So is the
  column list below (a test holds this document to them).
- Files are written whole and then put in place, so a reader never sees half
  a file.

## Keys and links

`report_key` is the key the report library already uses: the library
page's `id` and the `reports.db` row. It comes from the PDF itself (its size
and first megabyte), so the same file gets the same key whatever it is
called and whether it arrived as a path, as bytes or as an open document
(the cluster stage's way; it names the PDF with `source_path`). Only a
document whose file nobody names, and that was not opened from a file, keys
off what the record says it is (its report id, project number and name), as
the library always has; its `source_*` columns are null. Cluster libraries
built before RI-209 keyed every report that way, so their reports get a new
key once. A report bound inside another report (an
earlier firm's report reproduced as an appendix, say) is a report of its
own: its key comes from its parent's key and its handle inside the parent
(`bound_id`: `bound1`, `bound2` in page order).

How the tables join:

```
report (report_key)
  |-- report_page   (report_key, pdf_page)
  |-- report_answer (report_key, question_set, question_key)
  |-- report_file   (report_key, kind)
  |-- ingest_run    (report_key, run_id)      report.latest_run_id names the latest
  '-- report        parent_report_key -> report_key   (bound-in reports)
report_page.file_report_key  -> report    the file pdf_page counts in
report_page.bound_report_key -> report    the bound-in report that page belongs to
```

A bound-in report's pages are numbered in its parent's file: its page rows
carry `file_report_key` = the parent's key, and the parent's own rows for
those pages carry `bound_report_key` = the bound-in report's key.

## What a re-ingest does

- The report's rows in `report/`, `report_page/`, `report_answer/` and
  `report_file/`, and its files in the document folders, are replaced.
- A file the new ingest did not write is removed locally (a DIGGS file from
  an earlier ingest when this one wrote none; the answer sheet when the
  question list did not run this time).
- A bound-in report the new ingest no longer finds has its files removed
  locally (its `ingest_run` rows stay, as history).
- `ingest_run/` only gains files: one per ingest.

**What to trust in a dataset.** A dataset keeps every file it was ever sent,
so a file removed locally stays in the dataset. Build from the rows, not
from a folder listing:

- the files of a report are the ones its `report_file` rows name;
- a report row with a `parent_report_key` belongs only while its parent's
  `bound_report_keys` lists it. A bound-in report's key changes when the
  ingest finds the bound-in reports differently (they are renumbered), so
  keep a bound-in report's rows only when its parent lists its key, and
  drop the pages, answers and files of a report you drop.

## Building them for reports already ingested

```
python -m report_ingest.tables <folder of ingested reports> [--root <folder>] [--deliver]
```

or, in a notebook, `report_ingest.tables.export_folder(run_root, root=None,
deliver=False)`. It finds every report under the folder (each with the
reports bound inside it), writes the tables under `<root>/report_tables/v1/`
(the root defaults to that folder) and calls no model. `--deliver` also
sends each report to the dataset `GSE_REPORT_TABLES_DATASET` names, on
Foundry; it is also how to re-send what an ingest could not send (a busy
dataset).

A folder ingested before the tables existed has no run file, so:

- its key is read from its library page (as the library does);
- `source_name`, `source_bytes` and `source_sha256` are null;
- `ingested_at` is the record file's modified time, and `run_id` is that
  time and part of the record file's hash;
- the package versions are null, except planlens's, which the record keeps;
- where the question list ran then, its answers are read back from the
  answer sheet (`report.questions.xlsx`): the answer as the sheet shows it,
  its pages, note and From column, the status from the Answer cell (`N/A`
  is does not apply, empty is not said); the sheet holds no data type or
  typed value, so those columns are null.

## The run file beside each record

At the end of every ingest a small file, `report.run.json`, is written
beside each record (the report's and each bound-in report's): the run id
and time, the report's key and its parent's, the source file's name, size
and SHA-256, the package versions, whether the question list ran, and, in
the report's own file, what the export did (`export`: `written` or `error`,
how many rows of each table, and what the dataset got). The ingest never
fails because of the export: a failure is logged and kept there. In code,
`report_ingest.tables.export_outcome(out_dir)` reads it back; the cluster
stage copies it into each report's `run.json` (`report_tables`).

The question list's answers are now also saved as data beside the sheet,
`report.questions.json` (the same answers, typed, with the Run tab's
facts), which is what the tables are built from.

## The tables

<!-- columns: generated by `python -m report_ingest.tables --columns`; a test holds it to the code -->

### `report`

One row per report, as it stands after its latest ingest. A report bound inside another one is a report of its own, with parent_report_key set.

Files: `report/<report_key>.jsonl`. Primary key: `report_key`.

| Column | Type | Null? | What it holds |
|---|---|---|---|
| `export_version` | string | no | The export's version (v1). |
| `report_key` | string | no | The report's key: the library key the ingest computes, the same as the library page's id and the reports.db row. For a report, it comes from its source file's size and first megabyte, whether the file came as a path, as bytes or as an open document (only a document with no known file keys off its report id, project number and name); for a report bound inside another one, from its parent's key and its bound_id. |
| `parent_report_key` | string | yes | For a report bound inside another one, the parent's report_key; null for a report of its own. |
| `bound_id` | string | yes | For a bound-in report, its handle inside the parent (bound1, bound2, in page order); null otherwise. It is a count, not a name: when the ingest finds the bound-in reports differently, they are renumbered and their report_keys change. |
| `bound_report_keys` | array<string> | no | The report_keys of the reports bound inside this one, as of its latest ingest; empty when none. A report row whose parent_report_key does not list it any more is left over from an earlier ingest. |
| `report_id` | string | yes | The caller's handle for the report (a file stem or a corpus id); null when none was given. |
| `source_name` | string | yes | The source file's name, never a path; for a bound-in report, its parent's file. Null when the ingest was given the file's bytes, or a document whose file is not known, and for a folder ingested before this export. |
| `source_bytes` | integer | yes | The source file's size in bytes. Null when the ingest had neither the file nor its bytes (a document whose file is not known), and for a folder ingested before this export. |
| `source_sha256` | string | yes | The SHA-256 of the whole source file, in hex, so the PDF a dataset holds can be matched to its report; null as source_bytes. |
| `n_pages` | integer | no | Pages in the report: the whole file for a report, its own range for a bound-in report. |
| `title` | string | no | What the library page calls the report: its project name and document type. |
| `document_type` | string | yes | The owner's documentType answer (geotechnical report, recommendation letter, ...); null when the record does not state it. |
| `project_name` | string | yes | The project's name as printed (the narrative's answer, else the record's project block); null when not stated. |
| `project_number` | string | yes | The project or job number as printed; null when not stated. |
| `firm` | string | yes | The firm that wrote the report (geotechnicalEngineerFirm); null when not stated. |
| `post_name` | string | yes | The post the site belongs to, as the report names it (postName); null when not stated. |
| `report_date` | date | yes | The report's date (reportDateISO); null when the record has no full date. |
| `workflow` | string | yes | The workflow triage chose: standard, appendix_only, partial, multi_document, scanned or needs_person (nothing read: no page rows); null when no triage ran. |
| `n_investigations` | integer | no | Explorations (borings, test pits, soundings) read into the record. |
| `n_lab_tests` | integer | no | Laboratory tests read into the record. |
| `n_calculations` | integer | no | Calculation printouts read into the record. |
| `n_answers` | integer | no | Rows of report_answer for the report. |
| `n_answers_answered` | integer | no | Of those, the rows whose status is answered. |
| `n_qa_open` | integer | no | QA entries that need a person (rank open: a disagreement nothing settled). |
| `n_qa_changed` | integer | no | QA entries for a value the record changed or dropped by its own rule (rank changed). |
| `n_qa_note` | integer | no | QA entries that are notes (rank note). |
| `diggs_written` | boolean | no | A DIGGS 2.6 file was written by the ingest (a report_file row of kind diggs then exists). None is written for a record with no explorations and no laboratory tests, or one the DIGGS writer could not hold (the record's QA says why). |
| `diggs_schema_valid` | boolean | yes | true: the DIGGS file passed the DIGGS 2.6 schema; false: it did not; null: no file, or the check could not run where the ingest ran. |
| `diggs_roundtrip_ok` | boolean | yes | true: reading the DIGGS file back gives the record's own values; false: it does not; null: no file. |
| `latest_run_id` | string | no | The ingest that wrote this row: an ingest_run row's run_id. |
| `ingested_at` | timestamp | no | When the ingest wrote its outputs. For a folder ingested before this export, the record file's modified time. |

### `report_page`

One row per page of a report: what the page is, and who said so. A report triage sent to a person (workflow needs_person) has none. A bound-in report's pages are numbered in its parent's file, and the parent's own rows for those pages name it.

Files: `report_page/<report_key>.jsonl`. Primary key: `report_key`, `pdf_page`.

| Column | Type | Null? | What it holds |
|---|---|---|---|
| `export_version` | string | no | The export's version (v1). |
| `report_key` | string | no | The report's key: the library key the ingest computes, the same as the library page's id and the reports.db row. For a report, it comes from its source file's size and first megabyte, whether the file came as a path, as bytes or as an open document (only a document with no known file keys off its report id, project number and name); for a report bound inside another one, from its parent's key and its bound_id. |
| `pdf_page` | integer | no | The page as a PDF viewer numbers it, from 1 at the file's first page. For a bound-in report, the page of its parent's file (file_report_key). |
| `file_report_key` | string | no | The report_key of the file pdf_page counts in: the report's own for a report, its parent's for a bound-in report. |
| `printed_page` | string | yes | The page number printed on the page, as printed; null when none was read. |
| `label` | string | no | What the page is, one of: narrative, figure, plan, profile, boring_log, test_pit_log, cpt_log, dcp_log, lab_test, field_test, calculation, appended_report, photos, divider, cover, letter, toc, other. |
| `label_confidence` | double | yes | The highest confidence (0 to 1) among the voters that gave this label; null where none of them stated one (the section labeller, which settles every page on the default path, states none). |
| `agreed` | boolean | no | Every voter with a view gave the same label. |
| `settled_by` | string | no | vote; review where the label review moved the page; parent where a bound-in report keeps the label the whole file's pass gave the page. |
| `policy` | string | yes | The policy that combined the voters: structural, trust, confidence or rules. |
| `label_rules` | string | yes | The rules' label (planlens), when they gave one. |
| `label_vision` | string | yes | The vision voter's label, when it ran (model-first labels off). |
| `label_ranges` | string | yes | The section labeller's label, when it ran (model-first labels, the default). |
| `template_family` | string | yes | The printed log form the template voter recognised, when it recognised one. |
| `reason` | string | yes | Why the label is what it is, in a few words (a policy and a label, never the page's own words), from the run's labels.json; null when not saved. |
| `bound_report_key` | string | yes | On a report's own page: the report bound inside it whose pages include this one; null otherwise. |
| `title` | string | yes | The page's title or main heading as printed, from its saved description (model-first labels only); null otherwise. |
| `kind` | string | yes | What kind of page it is, in plain words, from its saved description; null otherwise. |
| `summary` | string | yes | One sentence on what the page shows, from its saved description; null otherwise. |
| `exploration` | string | yes | The boring, test pit, probe or well the page logs, as printed, from its saved description; null when it logs none or no description was saved. |

### `report_answer`

One row per standard question asked of a report: the owner's two standing schemas (general, natural_hazards), every field every time, and the owner's question list (question_list) when it ran. The caller's own free-text questions are not here.

Files: `report_answer/<report_key>.jsonl`. Primary key: `report_key`, `question_set`, `question_key`.

| Column | Type | Null? | What it holds |
|---|---|---|---|
| `export_version` | string | no | The export's version (v1). |
| `report_key` | string | no | The report's key: the library key the ingest computes, the same as the library page's id and the reports.db row. For a report, it comes from its source file's size and first megabyte, whether the file came as a path, as bytes or as an open document (only a document with no known file keys off its report id, project number and name); for a report bound inside another one, from its parent's key and its bound_id. |
| `question_set` | string | no | general or natural_hazards (the owner's two schemas), or question_list (the owner's question list). |
| `question_key` | string | no | The owner's field name (general, natural_hazards) or the list's Key (question_list). |
| `position` | integer | no | The question's place in its set, from 1, in the owner's order. |
| `question_label` | string | yes | The list's Output; null for the two schemas, whose key is the owner's own field name. |
| `category` | string | yes | The list's Category; null for the two schemas. |
| `data_type` | string | yes | The answer's type in the question list's words: string, boolean, integer, date, list of strings, list of records, dictionary, integer or list of integers. For the two schemas, from the field's type; null for a list answer read back from an older answer sheet. |
| `status` | string | yes | answered; not said (the report is silent); does not apply (the question has no bearing; question list only). Null where no answer came back at all (a question-list call that failed; the note says so). |
| `value_text` | string | yes | The answer as text: the schema's answer as printed (a list joined with '; '), or the question list's answer as its sheet shows it; 'none' for an empty list. |
| `value_number` | double | yes | The answer as a number, when it is one. |
| `value_boolean` | boolean | yes | The answer as true or false: a boolean answer, or for the schemas' yes/no questions yes as true and no as false (mixed and unclear stay null). |
| `value_date` | date | yes | The answer as a date, when it is a full date (the schema's reportDate gives its reportDateISO). |
| `value_list` | array<string> | yes | The answer as a list of text, when it is a list of plain values; null otherwise. |
| `value_json` | string | yes | The answer's typed value as JSON. Where the schema keeps a typed twin beside the printed answer (reportDateISO, siteClassNormalized, asceSevenVersionNormalized, bearingCapacityValues, strataList, the yes/no answers' verdicts), the twin's value; the twins' own page references are in pdf_pages and quotes, not here. Null when there is no answer. |
| `pdf_pages` | array<integer> | no | The pages the answer was read from, from 1 at the file's first page (a bound-in report's: its parent's file). |
| `quotes` | array<string> | no | The page's own words for the answer, where saved (the schemas' citations; the question list keeps none). |
| `note` | string | yes | What the answer comes with: the question list's note. |
| `source` | string | yes | Who answered, the question list's From: code, report text, both: agree, both: disagree, both: compare by eye, screening; null for the two schemas. |
| `code_value_text` | string | yes | On a question-list row answered by both code and the report text: code's reading, as text. |
| `qa_kinds` | array<string> | no | The kinds of the record's QA entries about this answer (QA entries whose where is '<question_set>.<question_key>'). |
| `question_list` | string | yes | For a question-list row: the list file's name and when it was last modified; null otherwise. |
| `prompt_version` | string | yes | The version of what asked it: the question list's (ql-...), or the model-first narrative reader's (mf-...), where saved. |

### `report_file`

One row per file delivered for a report in the export's document folders. Trust these rows, not a folder listing: a dataset keeps a file an earlier ingest sent even when the latest ingest wrote none.

Files: `report_file/<report_key>.jsonl`. Primary key: `report_key`, `kind`.

| Column | Type | Null? | What it holds |
|---|---|---|---|
| `export_version` | string | no | The export's version (v1). |
| `report_key` | string | no | The report's key: the library key the ingest computes, the same as the library page's id and the reports.db row. For a report, it comes from its source file's size and first megabyte, whether the file came as a path, as bytes or as an open document (only a document with no known file keys off its report id, project number and name); for a report bound inside another one, from its parent's key and its bound_id. |
| `kind` | string | no | diggs, record, summary or questions. |
| `path` | string | no | The file, relative to the export's v1 folder, with forward slashes. |
| `bytes` | integer | no | The file's size in bytes. |
| `sha256` | string | no | The file's SHA-256, in hex. |
| `media_type` | string | no | The file's media type. |

### `ingest_run`

One row per ingest of a report, never overwritten: the history, which is the ingest ledger BACKLOG RI-11 asked for. Each ingest adds one small file.

Files: `ingest_run/<report_key>.<run_id>.jsonl`. Primary key: `report_key`, `run_id`.

| Column | Type | Null? | What it holds |
|---|---|---|---|
| `export_version` | string | no | The export's version (v1). |
| `report_key` | string | no | The report's key: the library key the ingest computes, the same as the library page's id and the reports.db row. For a report, it comes from its source file's size and first megabyte, whether the file came as a path, as bytes or as an open document (only a document with no known file keys off its report id, project number and name); for a report bound inside another one, from its parent's key and its bound_id. |
| `run_id` | string | no | The ingest: its UTC time and a short random suffix. A bound-in report shares its parent's. For a folder ingested before this export, the record file's time and part of its hash. |
| `ingested_at` | timestamp | no | When the ingest wrote its outputs. For a folder ingested before this export, the record file's modified time. |
| `report_id` | string | yes | The caller's handle for the report (a file stem or a corpus id); null when none was given. |
| `source_name` | string | yes | The source file's name, never a path; for a bound-in report, its parent's file. Null when the ingest was given the file's bytes, or a document whose file is not known, and for a folder ingested before this export. |
| `source_bytes` | integer | yes | The source file's size in bytes. Null when the ingest had neither the file nor its bytes (a document whose file is not known), and for a folder ingested before this export. |
| `source_sha256` | string | yes | The SHA-256 of the whole source file, in hex, so the PDF a dataset holds can be matched to its report; null as source_bytes. |
| `n_pages` | integer | no | Pages in the report: the whole file for a report, its own range for a bound-in report. |
| `document_type` | string | yes | The owner's documentType answer (geotechnical report, recommendation letter, ...); null when the record does not state it. |
| `workflow` | string | yes | The workflow triage chose: standard, appendix_only, partial, multi_document, scanned or needs_person (nothing read: no page rows); null when no triage ran. |
| `n_investigations` | integer | no | Explorations (borings, test pits, soundings) read into the record. |
| `n_lab_tests` | integer | no | Laboratory tests read into the record. |
| `n_calculations` | integer | no | Calculation printouts read into the record. |
| `n_answers` | integer | no | Rows of report_answer for the report. |
| `n_answers_answered` | integer | no | Of those, the rows whose status is answered. |
| `n_qa_open` | integer | no | QA entries that need a person (rank open: a disagreement nothing settled). |
| `n_qa_changed` | integer | no | QA entries for a value the record changed or dropped by its own rule (rank changed). |
| `n_qa_note` | integer | no | QA entries that are notes (rank note). |
| `diggs_written` | boolean | no | A DIGGS 2.6 file was written by the ingest (a report_file row of kind diggs then exists). None is written for a record with no explorations and no laboratory tests, or one the DIGGS writer could not hold (the record's QA says why). |
| `diggs_schema_valid` | boolean | yes | true: the DIGGS file passed the DIGGS 2.6 schema; false: it did not; null: no file, or the check could not run where the ingest ran. |
| `diggs_roundtrip_ok` | boolean | yes | true: reading the DIGGS file back gives the record's own values; false: it does not; null: no file. |
| `record_schema_version` | string | no | The record's schema version. |
| `app_version` | string | yes | The geotech-staff-engineer version that ran the ingest; null when not installed as a package, and for a folder ingested before this export. |
| `core_version` | string | yes | The aec-agent-core version, likewise. |
| `planlens_version` | string | yes | The planlens version that read the file; null when not recorded. |
| `prompt_versions` | array<string> | no | The versions of what the models were asked, where saved: the model-first readers' (mf-...), the page describer's, the question list's (ql-...). |
| `models` | array<string> | no | The models the run's saved readings name (its item files, page descriptions and question-list calls). |
| `model_calls` | integer | no | The model calls the ingest made (for a report, its bound-in reports' included; for a bound-in report, its own). A step taken from the folder made none. |
| `input_tokens` | integer | no | Their input tokens, likewise. |
| `output_tokens` | integer | no | Their output tokens, likewise. |
| `dollars` | double | no | Their cost in dollars, likewise (0 where pricing is off). |
| `seconds` | double | no | The ingest's wall clock, likewise. |
| `resumed` | boolean | no | The ingest took some steps from results an earlier run of the same report saved. |

<!-- end columns -->

## Examples (made-up values)

Each file carries every column; a few are shown here.

`report/3f2a9c0d41b7e865.jsonl`

| report_key | parent_report_key | bound_report_keys | source_name | n_pages | document_type | project_name | report_date | workflow | n_answers | n_answers_answered | diggs_written | latest_run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3f2a9c0d41b7e865 | null | ["9b01c2d3e4f5a678"] | sample_report.pdf | 42 | geotechnical report | Example Warehouse Expansion | 2026-01-15 | standard | 38 | 21 | true | 20260202T101500Z-a1b2c3 |

`report_page/3f2a9c0d41b7e865.jsonl`

| pdf_page | file_report_key | printed_page | label | label_confidence | agreed | settled_by | label_rules | label_ranges | bound_report_key | title | exploration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3f2a9c0d41b7e865 | null | cover | null | true | vote | cover | cover | null | Geotechnical Report | null |
| 12 | 3f2a9c0d41b7e865 | 4 | boring_log | null | true | vote | boring_log | boring_log | null | Log of Boring B-2 | B-2 |
| 31 | 3f2a9c0d41b7e865 | null | appended_report | null | true | vote | appended_report | appended_report | 9b01c2d3e4f5a678 | Earlier Site Study | null |

`report_answer/3f2a9c0d41b7e865.jsonl`

| question_set | question_key | position | status | value_text | value_number | value_boolean | value_date | value_json | pdf_pages | quotes |
|---|---|---|---|---|---|---|---|---|---|---|
| general | boringCount | 13 | answered | 6 | 6.0 | null | null | 6 | [5] | ["Six borings were drilled"] |
| natural_hazards | siteClass | 9 | answered | Site Class C | null | null | null | "C" | [9] | ["Site Class C"] |
| natural_hazards | soilCorrosion | 6 | answered | The soils are mildly corrosive. | null | true | null | "yes" | [10] | ["mildly corrosive"] |
| general | primeContractor | 9 | not said | null | null | null | null | null | [] | [] |
| question_list | site_fill | 4 | answered | yes | null | true | null | true | [6] | [] |

`report_file/3f2a9c0d41b7e865.jsonl`

| kind | path | bytes | sha256 | media_type |
|---|---|---|---|---|
| diggs | diggs/3f2a9c0d41b7e865.diggs.xml | 81234 | 5e8f... | application/xml |
| record | record/3f2a9c0d41b7e865.record.json | 76512 | 0c1d... | application/json |

`ingest_run/3f2a9c0d41b7e865.20260202T101500Z-a1b2c3.jsonl`

| run_id | ingested_at | source_sha256 | app_version | planlens_version | prompt_versions | models | model_calls | dollars | resumed |
|---|---|---|---|---|---|---|---|---|---|
| 20260202T101500Z-a1b2c3 | 2026-02-02T10:15:00Z | 9d4e... | 5.36.0 | 0.15.1 | ["mf-10", "ql-1"] | ["Opus 5.5"] | 31 | 4.12 | false |

## A suggested Ontology

The FDE decides; this is the shape the tables were made for.

| Object type | From | Primary key | Title |
|---|---|---|---|
| Report | `report` | `report_key` | `title` |
| Report page | `report_page` | `report_key` + `pdf_page` (joined into one id, e.g. `3f2a...:12`) | `label` and `pdf_page` |
| Report answer | `report_answer` | `report_key` + `question_set` + `question_key` | `question_key` |
| Ingest run | `ingest_run` | `report_key` + `run_id` | `run_id` |

Links:

- Report has many Report pages, Report answers and Ingest runs
  (`report_key`); a Report's latest Ingest run is `report_key` +
  `latest_run_id`.
- Report to its bound-in Reports: `parent_report_key` -> `report_key`, kept
  only while the parent's `bound_report_keys` lists the child (above).
- Report page to the bound-in Report it belongs to: `bound_report_key`.
- The files (`report_file`): the DIGGS file, the record and the summary as
  properties or media of the Report, from the rows' `path`.
- **Post.** The existing Post object links to a Report by the post's name.
  The report's own words are `post_name` on the report row. Where the
  owner's question list ran with a list of posts (`GSE_REPORT_POSTS`), its
  post-name row in `report_answer` (`question_set` = `question_list`, the
  list's post-name Key) holds the matched post's name, and its `note` says
  what was printed and how it was matched; that is the better join where it
  exists. A name that matched no post stays as printed, with the reason in
  the note.

## Limits

- The caller's own free-text questions are not in `report_answer`.
- A report ingested from an upload before this change was keyed off its
  project name and number (the file's bytes were not used). A re-ingest now
  keys it off the file: its library page and a new `reports.db` row carry
  the new key, and the old row stays in `reports.db` beside it (nothing
  deletes rows there; the library's search follows the page's key).
  If such a folder was exported with `export_folder` before the re-ingest,
  its old key's rows stay beside the new ones; `source_sha256` is null on
  the old ones.
- `app_version` is the installed package's version: a checkout run without
  installing reports whatever version was last installed, or null.
