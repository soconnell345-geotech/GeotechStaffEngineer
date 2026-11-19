<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/rv-pl-09` (planlens keeps no review file on its main); the branch was then deleted. -->

# Review of PL-09 (reading gaps), branch `claude/pl-09-reading-gaps`

## 1. Verdict

**MERGE AFTER FIXES**, reviewed head `0f15d9a` (base `6196f13`). Both fixes are
on this branch (`claude/rv-pl-09`): merge this branch instead of the reviewed
one, after the lead decides finding 3.

## 2. Findings

| # | Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | must-fix, **fixed in `5ef754f`** | `reading_order.py`, `order_runs_across_columns` | The always-on warning fired on pages whose row order IS the reading order. Any two columns of lines with four words drawn row by row warned. | Made-up pages, 40 of each kind: a table of two text columns with no rules 40/40, a three-column table with no rules 40/40, a form with two fields per row 40/40, a list set in two columns 40/40, a form with labels left and values right 29/40, an index whose dot leaders are a run of their own 24/40, a title page with two address blocks 14/40. | Every column must read as running text: 3/5 of its lines at least 3/4 as wide as its widest, and 1/3 of its lines that have a small letter opening with one. A word of prose must have a letter (dot leaders). |
| 2 | should-fix, **fixed in `0185fd3`** | `continued_tables.py`, `table_continuation` | Two one-page logs of different borings on one form were noted as one table running on when the boring's number is printed in the page foot or beside the grid. Only the top band and the lines above the later table were compared. | Two pages, the same ruled grid with the same column titles, "B-1 LOG" / "B-2 LOG" at y = 770 of 792, or to the right of the grid. | The footer is compared as well as the header; a line beside the later table counts as a line above it; a table's own lines inside a band are left out. |
| 3 | note, **needs a decision** | `reading_order.py` | The warning cannot be made certain. A table with no rules whose cells are all about as long as each other and open with small letters is the same page, to this check, as prose in columns drawn row by row. After the fix 13 of 840 made-up pages that should not warn still do (1.5 %). | Title page with two address blocks in small letters 10/40; three-column table with no rules 2/40; index in two columns 1/40. All 13 have cells opening with small letters. | Count the warning on the real corpus before release. If it fires on any page that is not columns of prose, put it behind a switch as the other two pieces are. I recommend the switch unless that count is clean. |
| 4 | note, needs a decision | `continued_tables.py` | Two separate tables with the same column titles on consecutive pages, nothing else on either page, are linked even when the first ends at 39 % of the page height. The rule does not ask that the earlier table reach the page foot. | Two pages, each a 10-row table with the same titles and no caption. | Either require the earlier table to end near the foot (a threshold to choose), or word the note "may continue". |
| 5 | note, needs a decision | `roles.py`, `_is_list_page` | With `captions_on_every_page=True` a page is skipped as a "list page" when its top text holds the word "contents", "figures" or "tables" anywhere. A page captioned "TABLE 2 - MOISTURE CONTENTS" is skipped, so its own caption is not read. This mirrors the existing rule of `_read_list_pages`; narrowing it lets a list page with no page numbers give false captions. | Page of prose, then the line "TABLE 2 - MOISTURE CONTENTS": no caption read. Same page with "TABLE 2 - STRENGTH" and "moisture contents" in a sentence above: also none. | Skip only the pages that actually gave list entries, or leave as is and say so in DESIGN. |
| 6 | note | `roles.py`, `_match_entries` | With the captions switch, a table whose later page prints "Table 1. Summary ... (continued)" has two captions that match its list entry, so the entry stays unplaced. A miss, not a wrong page. | List entry "Table 1 Summary of Test Results"; captions on pages 1 and 3. | Leave; or drop captions that say "continued" from the match. |
| 7 | note | `reading_order.py`, `column_order` | A line whose direction is unknown (`rotation=None`, hidden CAD text) is sorted into the columns as if horizontal; the docstring says only horizontal lines are. Only for a host that calls it. | — | Leave, or say so in the docstring. |

What finding 1's fix costs: text in columns drawn row by row whose lines do
not wrap at the column width (my first two "true" kinds, lines of random
length) no longer warns: 1 of 80, was 80 of 80. Text that wraps warns as
before: 120 of 120, in small letters, in capitals throughout, and with short
paragraphs. Text in a script without case is judged on line width alone.

## 3. Paths checked

- **Warning, false positives.** Made-up pages, 40 per kind, counted with the
  real `Document.page`. Reviewed head: 227 false warnings on 720 pages (18
  kinds). After `5ef754f`: 13 on 840 (21 kinds). No warning before or after:
  single-column prose, two columns drawn column by column (two layouts), a
  drawing sheet with notes in columns (two draw orders), a boring log, short
  centred lines, a title page, a page with a wide figure caption, prose with
  legend labels beside it, two parallel texts drawn paragraph by paragraph.
- **Switches unset, byte for byte.** For the five suite fixtures and four
  made-up documents, every page at `words` on and off and `tables` on and
  off (`to_dict` with blocks, plus `repr`), and every outline: 236 values,
  identical at `6196f13`, at `0f15d9a` and at this branch's fix head. A page
  read twice returns the same object. Pages read in reverse order on a
  second `Document` are identical.
- **Switch on, page cache.** Two `Document` objects of one file, one with
  `continued_tables=True`: lines and `tables=False` pages identical; tables
  identical but for `notes`; the `words=True` page carries the same notes and
  keeps its words; the kept neighbour pages (`_unlinked`) carry no notes.
- **Lines with no box.** A made-up text source giving one line, then every
  line, no box: `page()` with and without tables, with the switch on and off,
  both outlines, `column_order` and the column check. Nothing raises; no
  table is linked; no warning. I found no other place that assumes a box
  (`Table.bbox` is required by the model).
- **`continued_tables`.** Header repeated: linked. Header not repeated and
  no word: not linked. "(continued)" above: linked, and the note says the
  titles are on the earlier page. Three pages: the middle table gets both
  notes. Table ending at y = 740, 758 and 776 of 792: linked each time. Two
  tables with captions of their own: not linked. Two borings, number above
  the grid or in the top band: not linked. Number in the foot or beside the
  grid: linked at the reviewed head, not linked after `0185fd3`. A later
  table starting inside the top band: missed at the reviewed head, linked
  after `0185fd3`.
- **Cost.** The column check takes 0.08 to 0.21 ms a page on pages of 17 to
  111 lines, 3 ms at 500 prose lines and 18 ms at 3,000. A 300-page PDF of
  100 lines a page reads in 1.89 s and 1.96 s with the check and 1.94 s and
  1.79 s without: the difference is inside the noise.
- **Merge with `origin/claude/b13-integration`.** One conflict, in
  `CHANGELOG.md`: both sides add entries at the same place under the same
  heading; keep both. The code merges clean, and 127 tests pass on the
  merged tree (the five PL-09 files, `test_document.py`, `test_roles.py`).
  Nothing was pushed.
- **Consumers.** `PageContent.warnings` is read by `model.to_dict`, the
  page a tool renders, and in the app by `report_ingest`'s narrative, calc,
  lab and sounding readers, which put each warning in their model prompts.
  No tool name, description or argument changed.

## 4. Tests

- Reviewed head, `/tmp/pl/bin/python -m pytest -q -p no:cacheprovider` on
  the three `test_pl09_*.py` files: 28 passed.
- On the base `6196f13` with those files copied in: `test_pl09_reading_order.py`
  does not import (no module); the other two give 10 failed, 4 errors, 0
  passed. Every new test fails without the change.
- Whole suite, `pytest planlens -q -p no:cacheprovider`, started after
  `5ef754f`: 2429 passed, 6 skipped, exit 0. On the fix head `0185fd3`,
  `planlens/document/tests` and `planlens/tools`: 1379 passed, 3 skipped,
  exit 0.
- New in `5ef754f`, `test_rv_pl09_columns_not_tables.py`, 6 tests: the four
  "says nothing" tests fail at `0f15d9a`; the two "still says so" tests pass
  there by design (they hold the repair in place).
- New in `0185fd3`, `test_rv_pl09_two_logs_one_form.py`, 4 tests: three
  fail at `5ef754f`; "one boring over two pages is linked" passes there by
  design.
- The tests check behaviour (warnings, notes, line order), not wording alone.
- planlens has no fingerprint test of its own. The app's was not run (see 7).

## 5. Scope

- Files changed by the builder: the three new modules' worth of code in
  `planlens/document/`, three new test files, `DESIGN.md`, `CHANGELOG.md`.
  No existing test was edited by the builder.
- Edited by me: `reading_order.py`, `continued_tables.py`, the two documents
  for the changed rules, two new test files, and ONE existing test file:
  `test_pl09_reading_order.py`. Its fixture tagged every column line with a
  capital ("C0R0 the bearing ..."), which the tightened check reads as
  cells. The tag is now written small and compared in capitals; no
  assertion changed.
- `Document.page` was split into `page()` and `_read_page()` as claimed.
- Public text: the diff holds made-up values only.

## 6. CHECK.md

There is none on the branch. The claims in the routine's prompt:

- "A page drawn row by row ... gets a page warning; its lines are
  unchanged": true.
- "A table read row by row carries no warning" (CHANGELOG, DESIGN): untrue
  at the reviewed head for a table with no rules (finding 1); true for most
  after the fix, not certain (finding 3).
- "The logs of two borings, one page each, are two tables even when their
  forms match" (module docstring): untrue at the reviewed head when the
  number is in the foot or beside the grid (finding 2).
- "With the switches unset every page returns what it did at the base":
  true on everything I built.
- Each switch is a keyword argument documented beside the code with its
  default, not a `GEOTECH_...` variable; planlens has no host settings, so
  the host that binds it would add one.

## 7. Could not check

- **The real corpus.** The false-positive rate above is on made-up pages.
  Run the column check over the ground-truth corpus in the app repository
  and read every page that warns (finding 3).
- **The app as a consumer.** I did not build the Document Review venv
  against this branch. Install this branch over the pinned planlens
  (`pip install --no-deps --force-reinstall -e <checkout>`) and run its
  tests; no tool specification changed, so its fingerprints should hold.
- **Models.** Whether the warning's wording leads a model to read the boxes,
  and what a report reader does with a false one, needs live runs.
