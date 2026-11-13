# CHECK: the report tables (BACKLOG RI-11), branch feat/ri-report-tables

The owner's ask (2026-11-12): the report ingest "wired to data in foundry".
After every ingest the app now writes flat, keyed tables plus the DIGGS
file, the record and the summary, so the FDE can build an Ontology on them,
and sends them to a Foundry dataset when one is named. Reference for the FDE
and the owner: `docs/REPORT_TABLES.md`.

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
- `report_ingest/tests/test_tables.py` (new, 58 tests), `docs/REPORT_TABLES.md`
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
11. The default root follows the brief (the folder holding `reports.db`). In
    the app each report keeps its own `reports.db` in its own folder, so
    the default there is each report's folder (see "For the lead").
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
| `report_ingest/tests/test_tables.py` | 58 passed | 0 |
| `report_ingest/tests/test_question_list.py` | 35 passed | 0 |
| `report_ingest/tests/test_ri101_question_list_run.py` | 31 passed | 0 |
| `report_ingest/tests/test_writers.py` | 48 passed | 0 |
| `report_ingest/tests/test_run_folder.py` | 11 passed | 0 |
| `report_ingest/tests/test_graph.py` | 85 passed | 0 |
| `report_ingest/tests/test_graph_model_first.py::TestOffIsToday` | 2 failed (below) | 1 |
| `webapp/tests/test_public_text.py` | 5 passed | 0 |

Not run here (left to the suites on GitHub): `test_cluster_scoring.py`,
the app's ingest tool tests, the store's own tests (the store was not
changed).

## For the lead

- `TestOffIsToday` in `test_graph_model_first.py` compares every file a lone
  ingest leaves with `graph_off_snapshot.json`; the export adds
  `report.run.json` and `report_tables/v1/...`, and the second comparison
  fails because the run id differs between two runs. Proposed: its
  `_off_run` sets `GSE_REPORT_TABLES=0` with a comment, the snapshot stays.
  Not changed until you decide.
- In the app the default root is each report's own folder, which sits in the
  conversation's folder unless `GEOTECH_REPORT_LIBRARY` is set; on Foundry
  that folder is mirrored to the conversation dataset, so the export files
  (about 20 small files and copies of the record, DIGGS file and summary per
  report) go there too. Setting `GSE_REPORT_TABLES` to a folder outside the
  conversation's avoids it.

## Follow-ups (one line each, for the lead to file)

- Decide the app's default root for the report tables (each report's folder today) and whether the conversation mirror should skip `report_tables/`.
- Have the FDE build the Ontology from `docs/REPORT_TABLES.md` and confirm the Post link (question-list post row vs `post_name`).
- Run `python -m report_ingest.tables <folder> --deliver` over the reports already ingested on Foundry once the dataset alias exists.
