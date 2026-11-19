7 findings, 5 fixed here, 3 suspected and not reproduced

# Audit of the files a tester meets in Document Review (2026-11-19, branch `claude/audit-files`)

Everything below was run on made-up inputs in venv (a) of `module_work/cloud_briefs/COMMON_DOCREVIEW.md` (core and Document Review alone, planlens from PyPI), with no network calls and no key. The scratch scripts were not committed. Each fix is in its own commit, and its test is in a new file (`packages/aec-agent-core/tests/test_audit_files.py` or `packages/aec-document-review/tests/test_audit_files.py`). Each of those tests fails on master and passes on the branch.

## Findings, ranked by how often a tester would meet them

| # | File and function | Made-up input | What happens (master) | What should happen | Status | How often |
|---|---|---|---|---|---|---|
| 1 | `aec_document_review/app.py` (sidebar uploader and chat box), new `core.uploads_to_stage` | `report.pdf` uploaded again with different bytes (a revision); or `Prüfung.pdf` and then `Präfung.pdf`, which both become `Pr_fung.pdf` | Only names not yet held were staged, so the second file was dropped without a word. The agent kept reading the first file. | Stage it beside the first as `report_2.pdf` (the attach line and the agent's note name it). The same bytes sent again on a rerun are still skipped. | fixed here | often: re-uploading a revised report is ordinary |
| 2 | `xlsx_writer.cell_value` (`write_xlsx`) | a Markdown table cell `first line<br>second line` | The cell held the literal text `<br>`. | A line break inside the cell. | fixed here | often: models write `<br>` in Markdown table cells |
| 3 | `aec_document_review/core.make_save_fn` | the tester uploads `memo.docx`, then the agent runs `write_docx` with path `memo.docx` | The agent's file replaced the upload on disk, and the result said only `saved`. After a reload or a restore, `load_attachments` read the agent's file as the tester's upload. | Save it as `memo_2.docx` and name that file in the tool result. A file the agent itself delivered is still rewritten in place. | fixed here | sometimes: "write the revised memo" under the same name |
| 4 | `cited_pages.check_answer` | two uploads; PDF page 3 opened only in `boring_logs_2.pdf`; the answer cites "boring_logs.pdf, PDF page 3" | The page counts as opened. `check_answer` checks every citation against all documents (`opened.documents()`), so the record says a page was read when it was not. | Weigh the citation against the document it names when the answer names one | noted (not small; the check is off by default, so this reaches the record only) | sometimes, in conversations with several uploads |
| 5 | `conversation_restore._read_layout` | a conversation with files uploaded and no question asked yet, downloaded and then restored | The export makes the zip, but it holds no `activity.jsonl`, and the restore refused it: "not a conversation zip". | Restore it: its `transcript.jsonl` marks it as the export's own zip. A zip holding meta.json alone is still refused. | fixed here | now and then |
| 6 | `aec_document_review/app.py` Downloads list (`except OSError: pass`) | a restored zip from which a delivered file was missing (deleted before the export) | The transcript still lists the file, but the card is skipped without a word. | One line saying the file is no longer in the conversation | noted | rare |
| 7 | `xlsx_writer.build_workbook` | a cell of 40,000 characters | openpyxl cut it to 32,767 characters without a word. The file opens but holds less than the agent wrote. | Cut it, and name the cut in `warnings` | fixed here | rare |

## Suspected, not reproduced

1. `annotate_document`: a marked copy saved under the upload's own name would replace it the way finding 3 did. The marked copy is written through planlens' host. I did not run it.
2. `cited_pages.doc_key` case-folds file names. Two uploads that differ only in case are two files on Linux but one key in the page check. Found by reading the code, not run.
3. A conversation cut off in the middle of a turn: the export leaves out `partial.json`. If the zip is made before the page recovers the turn, the half-written reply is not in it. Not run; the page normally recovers the turn when the conversation is opened.

## Questions without a finding (what I ran showed nothing to fix)

- Q1, Word: python-docx opens every file, and it holds what was written. I tried a heading with `**` and `` ` `` marks, `## Sec #2 ##`, a table inside a numbered list, `&`, `<x>` (kept as text, with a warning), an emoji, a tab, a form feed (became a space), `007` (stays text), and 40,000 characters. Excel: a 500-row log, a line break, a tab, `&`, `<tag>`, an emoji, a form feed (dropped), `007` (text), `1,250` (the number 1250), a 40-character sheet name (cut to 31; the summary names it), and `Soil/Rock` (became `Soil-Rock`).
- Q2: export, then restore into a fresh root and onto a shared host (where the thread id changes). Equal after the round trip: the transcript (including the delivered files' paths, which now point into the new folder), `messages.json` with its tool calls, and the uploads' bytes. The names tried: non-ASCII, two dots, 200 characters, and a delivered file in a subfolder with a space in its name.
- Q3: `open_document` returns a plain error with a hint for an empty file, a zip named `.pdf` and a password-protected PDF. A Word file is read as Markdown, and an image opens as one page. Names with spaces, non-ASCII letters, two dots or 200 characters all stage under safe names (see finding 1 for names that collide).
- Q4: `Identity.key`: `CORP\J.Doe`, `corp\j.doe` and `j.doe@corp` share one key, as designed (account names are case-insensitive). `o_brien` / `a_b` against `a b` / `josé` get distinct keys (a hash is appended). `..` as a part stays inside one path component. Uploads, deliverables, feedback (`feedback.record(conv_dir)`) and the restored conversation all live under `users/<key>/<page>/conversations/<id>`, and prepared downloads are kept in session state. I found no way for one tester to reach another's files.
- Q5: a bare `page 1` where the printed 1 is PDF page 5, `p. ii` and `page iii` (front matter), a cover with no number, a page that does not exist ("no document read here has that page") and the same page cited twice (counted once) were all handled as the module docstring says.

## What I ran (exit codes)

- `q1.py`: the writers, then reading the files back with openpyxl and python-docx: exit 0. A raw XML check of the 40,000-character cell: exit 0.
- `q2.py`: export and restore. The first run exit 1: it found finding 5, and the script stopped on that restore. After adding an activity log: exit 0. `q3.py` (`open_document` on six uploads): exit 0. `q5.py` (`check_answer`): exit 0. `q6.py` (`write_docx` over an upload): exit 0. It found finding 3.
- `pytest packages/aec-agent-core/tests/test_audit_files.py`: exit 1 before the fixes, exit 0 after. Also `test_saved_files_and_settings.py` and `test_core_suite_and_records.py`: exit 0.
- `pytest packages/aec-document-review/tests/test_audit_files.py`: exit 1 before each fix. Also `test_conversation_restore.py` and `test_conversation_export.py`: exit 0, 39 passed.
- `pytest` on every aec-document-review test file that names `stage_upload`, `sanitize_key`, `save_fn`, `make_save_fn` or `AppTest`, plus `tests/test_review_fingerprints.py`: exit 0, after the upload fix (252 passed) and after the save fix (237 passed). No fingerprint moved.
- Guards before each commit: `webapp/tests/test_public_text.py`: exit 0 each time. `test_backlog.py` and `test_docs_currency.py`: exit 0.

## Files read

- Read whole or nearly whole: `xlsx_writer.py`, `ws_upload.py`, `conversation_restore.py`, `identity.py` (from line 50).
- Read in part: `docx_renderer.py` (writer only, lines 85-520 and 719-770), `_fileio.py` (140-272), `conversation_export.py` (137-232), `file_widgets.py` (60-295), `cited_pages.py` (docstring, `PagesOpened`, `_elsewhere`, the head of `check_answer`), `document_tools.py` (dispatch only), and in `core.py` the staging, attachment, transcript and `make_save_fn` code. Also `app.py`'s upload and download code.
- Not reached: `office_text.py`, `cited_documents.py` (docstring only), `deep/upload_view.py` (function list only), `feedback.py` (grep only), and the reader half of `docx_renderer.py`.
