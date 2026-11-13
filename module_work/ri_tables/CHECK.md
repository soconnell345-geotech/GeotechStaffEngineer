# CHECK: the report tables (BACKLOG RI-209), branch feat/ri-report-tables

The owner's ask (2026-11-12): the report ingest "wired to data in foundry".
After every ingest the app now writes flat, keyed tables plus the DIGGS
file, the record and the summary, so the FDE can build an Ontology on them,
and sends them to a Foundry dataset when one is named. Reference for the FDE
and the owner: `docs/REPORT_TABLES.md`. The `ingest_run` table is the ingest
ledger BACKLOG RI-11 asked for. (The first commit, c78573d, cites RI-11 in
its message; RI-209 was filed for this work the same day.)

## What was built

- `report_ingest/tables.py` (new): the five tables defined once (`TABLES`:
  columns with a type and a line each, primary keys, links); `schema.json`,
  the row checks (`make_row`, `check_row`) and the doc's column list
  (`columns_markdown`) come from it. `export_report` builds one report's
  export (its bound-in reports with it) from the folder on disk;
  `export_folder` and `python -m report_ingest.tables` do it for reports
  already ingested, no model call; `after_ingest` is the graph's last step
  (saves `report.run.json` beside each record, exports, sends);
  `send_to_dataset` makes one save per report through
  `aec_agent_core.foundry_dataset_store` (its top folder, lock and outputs
  folder); `export_outcome(out_dir)` reads back what the export did.
- `report_ingest/graph.py`: `ingest_report(report_tables=None)`; the export
  after the writers; bytes now key the library row off the bytes;
  `_question_list` says whether a sheet was written.
- `report_ingest/writers.py`: bytes as the source write no source path (it
  would have written the bytes' repr into the page).
- `report_ingest/question_list.py`: `answer_json` writes
  `report.questions.json` beside the sheet (the one change there; nothing a
  model is sent changes, no prompt version).
- `report_ingest/run_folder.py`, `report_ingest/cluster_scoring.py`:
  `report_tables` passed through; the cluster stage's `run.json` gains
  `report_tables` (the outcome) when the export ran.
- `report_ingest/tests/test_tables.py` (new, 63 tests), `docs/REPORT_TABLES.md`
  (new), a README section, two rows in `docs/FOUNDRY.md` §3 (nothing else
  there touched).

## Decisions where the brief left room or the code said otherwise

1. `report_page.explorations` (a list) became `exploration` (text): a saved
   description names the one exploration the page logs; its `identifiers`
   mix boring numbers with sheet and figure numbers, so a list built from
   them would guess.
2. `report_answer` gained `position` (the question's place in its set, in
   the owner's order), so a pipeline can show the questions in order.
3. `quotes` is empty on question-list rows: the list keeps no quotes.
4. A question-list call that failed gives `status` null and the note "not
   run" (the sheet's own convention); the three status words otherwise.
5. `data_type` on the two schemas is the field's type in the list's words
   (string, integer, list of strings).
6. Typed twins: `value_json` carries the twin with its own page references
   taken out; the twin's citations join `pdf_pages` and `quotes`. A yes/no
   twin (the `*Answer` verdicts, `outsideProject`) gives `value_boolean`
   (mixed and unclear stay null) and `value_json` is the verdict.
7. The export's outcome is not on the record (the brief forbids changing
   it, and the outputs object is internal to the writer): it is in
   `report.run.json` (`tables.export_outcome(out_dir)`) and in the cluster
   stage's `run.json`. `ingest_report`'s return is unchanged.
8. Each delivery sends the report's files, `schema.json` and every
   `ingest_run` file of its keys, so a delivery missed earlier catches up.
9. A bound-in report shares its parent's `run_id`.
10. Off (`GSE_REPORT_TABLES=0`) writes nothing at all, not even the run file.
11. The default root, in order (the lead's rule for the app, 2026-11-12): a
    root the setting or argument names; the folder holding `reports.db`
    when the caller names one (a folder run, the cluster stage); the app's
    shared report library (`GEOTECH_REPORT_LIBRARY`) when the report's
    folder is inside it, so the library has ONE `report_tables/`; else the
    report's own folder (the app without a library: inside the
    conversation's folder, whose mirror then carries a small copy, left so).
12. An old folder with only the answer sheet: status from the Answer cell
    (`N/A` does not apply, empty not said, else answered); data type and
    typed value null.

## Keys of reports already in a library

Uploads handed in as bytes were keyed off the project's name and number;
they are now keyed off the file, like a path. A re-ingest of such a report
reuses its folder (the app's folder name already came from the file's
hash), rewrites the library page with the new key, and adds a `reports.db`
row under it; the old row stays (nothing deletes rows there). Reports
ingested from a path keep their keys. The app's shared-library folder name
(`funhouse_agent/deep/tools.py`) uses the same `source_digest`, so it stays
consistent and needed no change.

## What was measured

Each run alone, in the foreground, with at least 2 GB free:

| Test file | Result | Exit |
|---|---|---|
| `report_ingest/tests/test_tables.py` | 63 passed | 0 |
| `report_ingest/tests/test_question_list.py` | 35 passed | 0 |
| `report_ingest/tests/test_ri101_question_list_run.py` | 31 passed | 0 |
| `report_ingest/tests/test_writers.py` | 48 passed | 0 |
| `report_ingest/tests/test_run_folder.py` | 11 passed | 0 |
| `report_ingest/tests/test_graph.py` | 85 passed | 0 |
| `report_ingest/tests/test_graph_model_first.py` | 63 passed (with the export off in its off runs, below) | 0 |
| `webapp/tests/test_public_text.py` | 5 passed | 0 |

Not run here (left to the suites on GitHub): `test_cluster_scoring.py`,
the app's ingest tool tests, the store's own tests (the store was not
changed).

## For the lead

- Settled by the lead (2026-11-12): `TestOffIsToday` in
  `test_graph_model_first.py` compares every file a lone ingest leaves with
  `graph_off_snapshot.json`, which the export's files would break. Its
  `_off_run` now sets `GSE_REPORT_TABLES=0` (one comment line); the snapshot
  is untouched. The guard is kept by two tests in `test_tables.py`
  (`TestTheIngestsOwnFiles`): with the export off nothing of it is written
  (no `report.run.json`, no `report_tables/`, the bound report's folder
  included), and two runs with the export on leave the ingest's own files
  byte for byte the same as an export-off run (the run's folder, the clock
  and the DIGGS creation date taken out), adding only the export's folder
  and the run files.
- Settled by the lead (2026-11-12): with `GEOTECH_REPORT_LIBRARY` set, the
  root is the library's folder (decision 11, tested in `test_tables.py`
  `TestTheSwitch`); the conversation mirror is left alone.

## Follow-ups (one line each, for the lead to file)

- Have the FDE build the Ontology from `docs/REPORT_TABLES.md` and confirm the Post link (question-list post row vs `post_name`).
- Run `python -m report_ingest.tables <folder> --deliver` over the reports already ingested on Foundry once the dataset alias exists.
