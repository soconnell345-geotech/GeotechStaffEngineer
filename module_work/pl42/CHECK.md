# CHECK: PL-42, the schedules checked against the marks on the sheets

Two branches, neither pushed:

- planlens `claude/pl-42-schedule-marks` from main 68183cc: f8f4bbd (the
  check, the tool, the open_document block, the fixture and the tests) and
  7605152 (DESIGN section, CHANGELOG Unreleased, README, review DESIGN);
  after the review of 2026-11-14, f1f5541 (the review fixes and their
  tests) and 79e5645 (DESIGN and CHANGELOG for them). planlens keeps no
  CHECK file; this is it.
- app `claude/pl-42-app` from master dae4abb: b7fb060 (the pass-through
  behind `GEOTECH_REVIEW_SHEET_TOOLS` and its offline tests), 6bf06b2 (this
  file) and this file's revision after the review.

## Design, and why

A schedule lists marks and the plans draw them, on different sheets. A
reviewer whose helpers read the plans and the schedules in separate
contexts cannot hold both sides at once. So the comparison is made in code
over the whole set, and the agent gets both sides, with each item's place,
when it opens the document.

1. **Schedules from text geometry, no names known.** A table is anchored by
   a mark heading (MARK, TAG, NO., NUMBER, ID, #, with up to two words
   before it) or by the header row under a title with the word SCHEDULE.
   The title is looked for above the headings (past a row of group
   headings) and below the last row (past a view-title bubble). A header
   row is cut where a gap is much wider than its others (drawing text on
   its line).
   - The mark column is read down from the anchor while the lettering keeps
     its size (within a fifth) and the rows their spacing (no step over
     about twice the rows'). A sheet number of the set, a callout bubble or
     a mark of one or two characters after longer ones ends it, so a view
     title under the table is not a row.
   - A cell that is one mark is a row. A cell with several marks of the
     column's forms ("EF-3, EF-4", "W-1 THRU W-3") gives each. A cell of
     words (a sub-heading, "SECOND FLOOR") is passed over. A cell with
     figures and no mark is passed over and kept as not read
     (`cells_not_read`).
   - A table counts as a schedule if it is titled SCHEDULE with two or
     more headings, or untitled with a mark heading, three or more headings
     and two or more rows. No line above or below it may name a legend,
     keynote list, notes, revisions, symbols, an index or a drawing list.
   - Dropped: a header row at the same place on most sheets (the title
     block's revision block); an untitled mark heading at one place on
     three sheets or more (a consultant's revision table); a table whose
     marks are mostly the set's own sheet numbers (a drawing index).
   - A title repeated with CONTINUED, or an untitled table with the same
     headings, joins the schedule it continues.
   - The only word lists are the generic ones: "schedule", the mark
     headings, and the words that name what is not a schedule. No mark
     types and no schedule names.
2. **Each schedule's mark form comes from its own rows.** Letters before
   the digits stay literal, digit runs match by length, and letters after
   the digits match any letter: W-1 gives W-9, FCU-1-1 gives FCU-9-9, LP-4A
   gives LP-9A, 101 gives 999. Marks of one or two characters are not
   looked for, and the result says how many rows that was.
3. **A mark is drawn where a piece of text is that mark alone.** The text
   includes lettering a CAD program keeps only as hidden annotations
   ("AutoCAD SHX Text"), read as planlens reads it elsewhere. Text inside
   schedules, callout bubbles, the title block's number box and the set's
   sheet numbers does not count. A mark inside a note or a range ("DOORS
   202 TO 205") is named, not drawn. That keeps its row out of no_mark,
   but it is not taken as a tag.
4. **Four statuses, kept apart:**
   - `no_row`: a mark is drawn and no schedule of its form has a row for
     it.
   - `no_mark`: a row of a schedule titled SCHEDULE whose mark is drawn
     nowhere and named nowhere, where at least half of that schedule's rows
     are found in the sheets' text.
   - `may_be_on_unread_sheet`: a row not found where some sheets have no
     usable text or are mostly an image (image over half the sheet); and a
     mark with no row where an unread page sits next to one of its
     schedule's pages, since part of the schedule could be printed there.
     The unread sheets are listed apart with their pages. A mark with no
     row stays `no_row` when the unread pages are away from its schedule,
     and its why names them.
   - `cannot_tell`: the reason is in `why`. Among them: a row of a table
     with no SCHEDULE title (a list need not have every entry drawn); a
     row of a schedule with fewer than half its rows found anywhere (its
     marks may be drawn as lines, so absence is not established); a mark
     with no row where a cell of its schedule's mark column was not read.

   Each item carries its sheet, `page` (0-based), `pdf_page` (1-based), the
   schedule by title and sheets, a bbox and the why.
5. **Plain numbers are judged by their lettering. This guard keeps the
   false alarms down.** A plain number counts as a mark only in the
   lettering (font, size, bold) of the schedule's other marks on that
   sheet, and only on sheets showing two or more of them. A number with a
   word just above or below it is a room's name and number.
   - The lettering is unclear: cannot_tell.
   - The lettering is settled only by which one stands alone (doors
     numbered as their rooms): an unlisted number is cannot_tell, since a
     room numbered as its door reads the same.
   - Otherwise one unlisted number: no_row.
   - Two or three: cannot_tell for each.
   - More than three: a single cannot_tell for the sheet, with a sample.

   A family guard covers prefixed forms: three or more different marks of
   one form with no row, outnumbering half the rows found, give
   cannot_tell.
6. **Where it sits:**
   - `Document.schedule_marks()` is cached. The word above or below a
     number is found by bisection on the page's word boxes, worked out
     once per page.
   - The tool is `check_schedule_marks(handle, pages, status, offset)`, the
     third opt-in sheet tool (`SHEET_TOOL_SPECS`, `review.specs.SHEET_TOOLS`,
     `conversation._OPT_IN_TOOLS`).
   - With the sheet tools asked for, `open_document` adds `sheet_index`
     (page, pdf_page, number and title of each sheet) and `schedule_marks`
     (counts, the schedules' names, and up to 12 items without bboxes).
     Together they take at most three quarters of the room left after the
     map, and each says which tool gives the rest.
   - The host layer decides per call from the calling host
     (`toolkit.sheets_at_open`). A host that did not ask gets
     `open_document` byte for byte as before, as does a document with no
     drawing sheets.
7. **App.** Where the installed planlens publishes it,
   `document_tools.OPTIONAL_SHEET_TOOL_NAMES = ("check_schedule_marks",)`
   and `sheet_tool_names()` add it to a page built with the sheet tools. It
   reaches both the page and its reading helper, described in planlens'
   own words. On the pinned planlens 0.15.1 nothing changes, and
   `SHEET_TOOL_NAMES` and `sheet_tools_available()` are as they were.

## What defeats it

- **Marks drawn as lines.** The first version of this file said such a
  mark is "not seen". That was wrong in the worse direction: where a
  plan's tags were AutoCAD SHX lettering (strokes, with the text only in
  hidden annotations), every row of the schedule came back as a definite
  `no_mark`, "every sheet was read" (the reviewer's set 1c: 8 false items).
  Now:
  - hidden SHX annotations are read as text, so those tags are found;
  - tags drawn as lines with no hidden text: their schedule's rows are
    `cannot_tell` when fewer than half of its rows are found anywhere;
  - tags in a raster plan (an image over half the sheet): rows not found
    are `may_be_on_unread_sheet`;
  - still defeated: a schedule whose marks are mostly text with a few
    drawn as lines. Those few rows are `no_mark`.
- **Text only otherwise.** A schedule drawn as lines or on a scan is not
  seen. A turned schedule is not read. A table with no SCHEDULE title and
  no mark heading is not a schedule.
- **Untitled tables.** A real schedule with no SCHEDULE title (headed TAG
  or MARK only) gets no `no_mark`: its rows drawn nowhere are
  `cannot_tell`. Its marks with no row are still `no_row`.
- **Plain-number marks:**
  - They are looked for only on sheets showing two of that schedule's
    marks. A floor whose doors are all missing from the schedule is not
    seen.
  - Where rooms are numbered as the doors are, in the same lettering, a
    door's row can be found as a room's number, and an unlisted door
    number is `cannot_tell`, not `no_row`.
  - planlens' own `build_sheet_set` shows a related limit: 201A to 203A on
    A-102 are not reported, because no scheduled mark on that sheet fixes
    the lettering.
- **Shared forms.** Unrelated tags of one form below the family guard (two
  wall types shaped like window marks) give no_row. The item says to view
  it.
- **Unread sheets.** Every row not found becomes `may_be_on_unread_sheet`.
  A part of a schedule printed on an unread page that is not next to its
  other pages is not allowed for: a mark whose row is there is a false
  `no_row`, whose why names the unread pages.
- **Marks of one or two characters** are not looked for.
- **Legends and keynote lists** are told from schedules only by the words
  in the lines above and below them.
- **Not measured on real drawing sets.** Every figure below is on made-up
  sets.

## The review's probe sets, before and after the fixes (no model)

The reviewer built four sets with what real sets hold and the fixture did
not. "Definite false" counts `no_row` or `no_mark` items a reader could
pass to a design team as findings when they are not; "planted" counts the
items the set was built to show that came back as the ideal status.

| Set | What it holds | Definite false, before | after | Planted, before | after |
|---|---|---|---|---|---|
| 1 clinic, 10 sheets | two marks in one cell, a keynote list titled below, a view title under a schedule | 6 | 0 | 9/9 | 9/9 |
| 1b, P-101 scanned | set 1 with the plumbing plan scanned | 0 | 0 | 8/8, all as may_be (18 may_be in all) | 8/8; W5 and VAV-1-05 stay no_row |
| 1c, M-101 tags as SHX annotations | set 1 with the mechanical tags as hidden CAD text | 12 (8 of them its own) | 0 | 8/9 (VAV-1-05 missed) | 9/9 |
| 2 school | doors numbered as their rooms | 1 (206 no_row) | 0 | 2/2 | 2/2 |
| 3 | consultant revision tables on 4 of 10 sheets | 1 (C1 no_mark) | 0 | 1/1 | 1/1 |
| 4 | a view-title bubble under a schedule, two marks in one cell | 1 ("S-501" no_mark), 4 cannot_tell noise | 0, no noise | none planted | none planted |
| 40-sheet coordination set, page 16 scanned | the fixture with one detail sheet scanned | 0 | 0 | 7 found, all may_be (5 should be definite) | 7 found, 5 no_row definite, 2 may_be |

In all: 21 definite false items before (17 distinct, the reviewer's count;
1c repeats four of set 1's), none after, and no planted item lost. Left as
it was: set 2's `cannot_tell` items for 107, 900 and 207 (the reviewer's
finding 8, rooms in a finish schedule; see below), now with 206 among
them.

On the reviewer's dense probe (8 plan sheets of 4,000 text lines each) the
check took 16.2 s before and 0.6 s after; the drawing index took 3.8 s
both times. Before, it also read two other tables of the probe as
schedules (66 rows for 60); after, only the real one.

## Replay, no model (`build_drawing_set` of the coordination fixture)

Unchanged by the fixes, apart from the scanned-page row.

| Set | Sheets | Schedules found | Planted mark problems | Flagged | Missed | Other items | Time |
|---|---|---|---|---|---|---|---|
| 15 sheets (`build_drawing_set(2, 1)`) | 15 | DOOR (A-601, 14 rows), FOOTING (S-501, 3) | 3 | 109 no_row (A-101), F4 no_row (S-101), 204 no_mark | 0 | 0 | 0.05 s |
| 40 sheets (`build_drawing_set(5, 1)`) | 40 | DOOR (A-601, A-602; 38), FOOTING (3), MECHANICAL EQUIPMENT (M-601; 20; forms AHU-9, FCU-9-9), PANEL (E-601; 10; LP-9A) | 7 | 109, 407, F4, FCU-3-4, LP-4B no_row; 204, 509 no_mark | 0 | 0 | 0.1 s |
| 40 sheets, page 16 scanned | 40 | as above | 7 | 109, 407, F4, FCU-3-4, LP-4B no_row; 204, 509 may_be_on_unread_sheet | 0 | 0 | 0.1 s |
| 15 sheets scanned (`build_scanned_set()`) | 15 | none | 3 | none | 3 (only a look reads a scan) | 0 | 0.6 s |

On the scanned set all 15 sheets are listed unread with their reason, and
the note says no schedule was found on the 0 sheets that could be read and
15 could not be read. It does not guess. Through the toolkit, the 40-sheet
`open_document` with the block came to 7,395 characters at a 15,000 limit
(6,830 before the fixes; the why texts are longer), with all 7 items in
it.

## Tests, and where the packages came from

Every run printed `planlens.__file__`, `aec_agent_core.__file__` and
`aec_document_review.__file__` first, and all three resolved inside the
worktrees (`...\scratchpad\pl42p\planlens`,
`...\scratchpad\pl42a\packages\aec-agent-core\aec_agent_core` and
`...\scratchpad\pl42a\packages\aec-document-review\aec_document_review`).
Runs were judged by exit code. Figures are from the run after the review
fixes.

- **planlens, new tests:**
  - `document/tests/test_schedules.py` (26): the fixture's truth, the set
    with a sheet scanned and the whole set scanned, no schedule, a document
    with no drawing sheets, mark forms, ranges, the plain-number and family
    guards, revision blocks and untitled tables.
  - `document/tests/test_schedules_hard_cases.py` (16, from the review):
    hidden SHX tags read; tags drawn as lines and a raster plan leave the
    rows open; a view title under a schedule; the column stopping at a
    lettering change and a wide step; two marks in one cell; a sub-heading
    row; a cell not read; an untitled table; a keynote list titled below;
    a consultant's revision table on 3 of 8 sheets; a header row cut at a
    wide gap; doors numbered as rooms; a scanned sheet away from and next
    to the schedule; the word lookup giving the same answers as the old
    page-wide scan.
  - `tools/tests/test_schedule_marks_tool.py` (8): items, filters, paging
    under a small limit, unread sheets, the open block and its cap, and
    byte-identical output when not asked.
  - `review/tests/test_schedule_marks_host.py` (2): pdf_pages, and only a
    host that asked gets the block, on one shared toolkit.
- **planlens, touched files:** 481 passed, 1 skipped (exit 0; 465 and 1
  before the review, plus the 16 new). That run covered
  `review/tests/test_sheet_tools.py`, `test_host_specs.py`,
  `test_tools.py`, `test_conversation.py`, `test_page_numbers.py`,
  `test_host_per_conversation.py`, `test_pl38_viewer_pages_note.py`,
  `document/tests/test_sheets.py`, `test_crossrefs.py`,
  `test_structure.py`, `test_document.py`, `tools/tests/test_toolkit.py`,
  `test_mcp_server.py`, `test_structure_tools.py` and
  `tools/tests/test_sheet_tools.py`, plus the new files; then
  `tests/test_readme_claims.py` (11 passed). Two existing tests pin three
  sheet tools rather than two, and `open_document` as unchanged apart from
  the two added keys; two assertions of `test_schedules.py` follow the new
  why texts ("every sheet's text was read"; an unread page next to the
  schedule's pages).
- **Document Review:** `test_review_sheet_tools_offline.py` and
  `test_review_fingerprints.py` gave 19 passed, 2 skipped (exit 0);
  `test_review_switches_offline.py` gave 9 passed. The new offline tests:
  - the optional name;
  - with the switch off, neither block appears and the tool is absent;
  - on a planlens without the check, only the two other tools;
  - the page and its reading helper have it;
  - on the 15- and 40-sheet sets, `open_document` and the check name the 3
    and the 7 planted problems and nothing else, with `pdf_page`, sheet
    and schedule;
  - the scanned set is listed unread.
- **Core:** no core test names `sheet_tools`. I ran `test_core_imports.py`,
  `test_conversation_isolation.py` and `test_app_details.py` instead: 25
  passed.
- **App root:** `webapp/tests/test_public_text.py` passed.
  `webapp/tests/test_page_fingerprints.py` fails
  `test_app_fingerprint_unchanged[geotech_default]` (read_reference_table
  gone, the helper's prompt changed). The geotech-references submodule is
  not checked out in this worktree. The same failure shows on an export of
  master dae4abb with planlens main, so it is noted and not fixed.

## Fingerprints

- **Switch unset:** Document Review's app fingerprints equal the recorded
  ones (`test_app_fingerprint_unchanged` passed for both builds).
- **The planlens part** is skipped locally. planlens main's surface
  3e415249e4551735 is not recorded; only 0.15.1's 709cbea6e23e8aaa is. This
  branch's surface equals main's, because the new spec sits in
  `SHEET_TOOL_SPECS`, outside the surface (still so after the tool's
  description changed in the review fixes).
- **Full comparison.** `python -m tests.page_builds fingerprints` with
  planlens main and with this branch, both on the app branch, gave
  identical JSON (first round).
- **Switch on.** The sheet-tools arm is not pinned, as before.

## Release notes

planlens (next release, paste into the release text):

```
A drawing set's schedules checked against the marks on its sheets, for a
host that asks (app BACKLOG PL-42). Document.schedule_marks() reads every
schedule in the text layer (titled SCHEDULE, or headed MARK, TAG, NO.,
NUMBER or ID with three columns or more; legends, keynote lists, revision
blocks and sheet indexes are not schedules; one printed over two sheets is
one), learns each schedule's mark form from its rows, and checks every
sheet both ways, hidden CAD lettering included: no_row, no_mark,
may_be_on_unread_sheet (unread sheets listed apart) and cannot_tell, each
with its sheet, page, pdf_page, schedule, box and why. Where the text does
not settle an item (a schedule whose rows are mostly found nowhere, a
table with no SCHEDULE title, doors numbered as their rooms) it says
cannot_tell rather than a definite item. A third sheet tool,
check_schedule_marks, serves it, and with the sheet tools on,
open_document brings the sheet index and this check with the map. Off by
default: without the sheet tools nothing changes. Measured on made-up sets
only.
```

Testers (Document Review, once the app pins that planlens and the
deployment sets GEOTECH_REVIEW_SHEET_TOOLS=1):

```
When you open a drawing set, the review now starts with the set's sheet
list and a check of its schedules against the tags on the plans: a tag on a
plan with no row in its schedule (a door, a fan coil unit, a panel), and a
schedule row whose tag appears on no sheet. The reviewer views each one
before commenting. It reads the PDF's text, so it cannot check scanned
sheets or tags drawn only as lines: it lists scanned sheets instead, and
says when it cannot tell. Plain-number tags (door 105) are checked only
where they are lettered like the schedule's other tags on that sheet. Tell
us about any it flags wrongly or misses.
```

## Noted, not fixed (the lead files any in BACKLOG)

- `webapp/tests/test_page_fingerprints.py::test_app_fingerprint_unchanged[geotech_default]`
  fails without the references submodule. The same happens on master, so
  it is not this change.
- Document Review's planlens fingerprints have no entry for planlens
  main's surface. Record them when the app's planlens pin moves.
- The open block's cap (three quarters of the room after the map, 12
  items) is set by judgement and has not been measured with a model.
- The review's finding 8 (room numbers from a finish schedule giving
  `cannot_tell` items, set 2's 107, 900 and 207) is left. Counting a row
  as found in any lettering whenever two or more rows are found there
  would hide the fixture's door tag 203, drawn only as a room's number;
  the items are `cannot_tell`, which a reader views, not definite ones.
- A schedule with most marks in text and a few drawn as lines gives
  `no_mark` for those few; a schedule part on an unread page away from its
  other pages gives `no_row` for its marks (see "What defeats it").
- Before turning the arm on for testers, it needs a run on real drawing
  sets.
