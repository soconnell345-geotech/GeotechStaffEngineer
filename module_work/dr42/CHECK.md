# DR-42 and DR-34: text the Document Review page's model reads

Branch `claude/dr-42-page-texts`, 2026-11-18. Both items are repairs of a
defect (the model was told about tools the page does not have, and about
soil tests on a page that is not geotechnical): on by default, no switch.

## DR-42: the two notes name no tool the page lacks

What the item asked: the upload note and the core's note after a helper's
saves named tools only the geotech page has; reword for this page, the
core's note through a host hook.

Built:

- `packages/aec-document-review/aec_document_review/core.py`
  `attachment_note`: each file is now "pass source='x' or
  attachment_key='x' to the tools that open, read or look at it (a report,
  drawing set, spreadsheet or image); it is also the file 'x' in the
  working folder, so a tool that takes a file path takes that name". No
  tool is named. 327 characters for one file (was 417), 851 for three
  (was 1,118).
- `packages/aec-agent-core/aec_agent_core/deep/keep_work.py` `saved_note`:
  the sentence after the file names comes from the new hook
  `keep_work.saved_files_hint` (`hint(names) -> str`; constants
  `SAVED_FILES_HINT_HOOK`, `DEFAULT_SAVED_FILES_HINT`). Unfilled, raising,
  unimportable, empty or not text: the old sentence, so the geotech page's
  note is unchanged byte for byte and the app's code is not touched. The
  hook is listed in `aec_agent_core/hooks.py`'s docstring.
- `packages/aec-document-review/aec_document_review/__init__.py` fills the
  hook with `core.saved_files_hint`, which returns `core.SAVED_FILES_HINT`:
  "Give a file to the next tool by its name (as its source or file path)
  -- never retype what it holds." (100 characters; the old sentence 141).

Calls made:

- The upload note names no tool at all, rather than the tools of the
  default build. The page's builds differ: the minimal build
  (`GEOTECH_REVIEW_AGENT=minimal`) has neither `open_document` nor
  `read_pdf_text`, which the old note also named. `source` and
  `attachment_key` are arguments on both builds. The routing the old note
  gave ("to review a PDF, open_document") is in the default build's system
  prompt ("Open every PDF with `open_document` first"; "Any tool that takes
  a `source` accepts an attachment key or a real path").
- The hook words only the sentence after the names, not the whole note:
  the list of names (ten at most, "and N more") stays in one place.
- The hook is given the names, so an app could word it by file kind; this
  page does not.

## DR-34: the SharePoint tools' text names no soil tests

Built, in `packages/aec-document-review/aec_document_review/sharepoint_tools.py`:

- `_DOWNLOAD_DOC`: "(read_pdf_text, open_document, subsurface parsers,
  ...)" is now "(open_document, read_pdf_text, read_text_file, ...)":
  three tools the page has; the last reads a Word or Excel file.
- `_SEARCH_DOC`: the example queries "boring log" and a place name are now
  "site plan" and "Block C" (a file name and a folder name, as before; one
  character shorter).

The geotech page's own copy (`webapp/sharepoint_tools.py`) is not touched.

## Tests and what each proves

- `packages/aec-agent-core/tests/test_dr42_saved_note_hook.py` (new):
  unfilled, the note equals the old text byte for byte (passes on master
  too, by design: it pins the geotech page's text); a filled hook words the
  sentence and receives the names; a raising, empty, blank, non-text or
  unimportable hook keeps the default. The last three fail on master (no
  hook there).
- `packages/aec-document-review/tests/test_dr42_notes_name_page_tools.py`
  (new): both notes, with made-up uploads, checked against the tool names
  and argument names of each build of the page as built offline
  (`tests/page_builds.py`), so the check follows the tools when they
  change; plus the vocabulary guard's words. Both fail on master (the
  upload note names `pdf_import` and others; `core.saved_files_hint` does
  not exist).
- `packages/aec-document-review/tests/test_review_page_vocabulary_offline.py`
  (edited, DR-34): every build is made with SharePoint configured, so the
  SharePoint tools' descriptions are read, and the builds that add tools
  must carry them; the prompt test also reads `SHAREPOINT_PROMPT`. On
  master the guard fails on "subsurface" and "boring log".
- `packages/aec-document-review/tests/test_core.py` (edited): one assertion
  wanted `read_pdf_text` and `dxf_import` in the upload note; it now wants
  them out and `source='report.pdf'` in.

The "fails on master" statements are from reading the code, not from a
run on master.

Fingerprints: neither item moves one. Both notes are run-time text, not
part of what a build hands the agent framework, and the fingerprint builds
run with SharePoint unset. `tests/test_review_fingerprints.py` and
`webapp/tests/test_page_fingerprints.py` pass on the branch.

## Noted, not fixed

- The hook is process-wide: a process that imports both Document Review
  and the geotech app gives both pages Document Review's sentence (the
  `prompter.*` hooks already behave so). No host does that, and the root
  CI job does not install Document Review.
- The conversation-bound upload tool says "(a report, figure or calc
  package)"; the module-level upload tool, which the page does not use,
  says "save_file / calc-package tools". Neither is in the guard's words.
- The module docstring of `sharepoint_tools.py` uses "borings/" as an
  example path (developer text, not handed to the model).
- A real place name is still the made-up query of one test in
  `tests/test_sharepoint_links_and_failures.py`.

## For the lead: what a measurement on the hosted platform should compare

A suite run before and after, same model: per task, the score; calls to a
tool the page lacks (should be none); the first call on each upload (does
it pass the key as `source` or `attachment_key`, and to the right tool);
and, on tasks where a helper saves a file, whether the primary passes the
file by name rather than retyping it.
