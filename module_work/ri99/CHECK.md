# RI-99 parts (1) and (2): placement on vertical lab tables and repeated calc blocks

Branch `claude/ri-99-placement-tables-blocks`. Scope: BACKLOG RI-99 parts (1)
and (2) only. Parts (3) to (6) are untouched (judgment calls or model-facing).

## (1) Vertical lab tables: a test lands in its sample's column

**Before.** On a table printed one column per sample (sample labels across a
header row, each property's values along its row), a test's first anchor,
`sample label`, found the label once and boxed the label itself, in the
header. The reviewer clicked through to the header, not to the test's values.

**Change.** There is a new first lab anchor, `sample column`
(`report_ingest/locate.py:1446`), wired in `lab_targets`
(`report_ingest/locate.py:1574`) and added to `WORD_ANCHORS`
(`report_ingest/locate.py:173`), so a test still moves to another page of
its call by it.
- **Orientation, read from the printed layout** (`sample_columns`,
  `report_ingest/locate.py:1374`). Take the label's shape: its letters, then
  its number (`_LABEL_SHAPE`, `:1338`; `_shaped_runs`, `:1341`). If two or
  more other labels of that shape are on the label's row, and fewer are
  straight above or below it, the samples run across a header row. Anything
  else (one row per sample, or unclear) returns None, and today's
  `sample label` anchor places the test exactly as before.
- **The column** runs from midway to the neighbouring labels (as far again
  past the outer ones), down to the row above the next row of such labels
  (the next table's header) or the first row that prints nothing in the
  column (the table's foot) (`_column_cells`, `:1430`). The box spans
  the column's cells that print the test's own results, from its first
  property's row to its last.
- **The same label in several columns** (one `S-1` under each boring of the
  header): the anchor takes the column that prints the most of the test's
  results, depth and boring. On a tie it gives nothing, and today's
  placement follows.
- **A test with no results to match** (for example nonplastic), or whose
  column prints none of them, gets nothing from this anchor and falls back
  to today's placement.

**Made-up example.** Header `Sample | S-1 | S-2 | S-3`, then rows for depth,
LL, PL and PI. The test S-2 has LL 41, PL 19, PI 22.
- Before: box on `S-2` in the header (y about 87).
- After: box over 41/19/22 in the S-2 column (y about 110 to 147),
  anchor `sample column`.

## (2) Calc printouts with repeated blocks: a value lands in the block it names

**Before.** `_label_and_number` scores each printed copy of the number by
its label, which includes up to `CONTEXT_ROWS` (6) heading rows above it.
In a block printed once per boring, each block's label lines are themselves
"heading rows" (mostly words). So the six nearest headings stop short of
the value's own block heading, or reach the block above as readily as their
own. The value either came out `several`, or a stray digit decided it.
- In the test sheet, the decimal `23.2` in block B-1 folds to the tokens
  `23 2`. That token `2` gave B-1's copy a half-weight match for `B-2`, so
  master placed "Allowable bearing pressure B-2" in the **B-1** block, which
  is wrong.

**Change.** `heading_rows` itself is unchanged, because other anchors use
it. Instead there is a new first calc anchor, `number in its block`
(`_number_in_block`, `report_ingest/locate.py:1856`; order in
`value_targets`, `:1945`).
- **Block detection** (`block_headings`, `:1795`). Take the rows where the
  value prints with half its label or more. If they print the same words,
  numbers aside (one label printed once per block), they are read upwards
  in step, line by line, for as long as every block prints the same words.
  The first line where the blocks differ is each block's own heading
  (`Boring B-1` / `Boring B-2`). The search goes no further up than the
  block above begins. A line that prints a number in the value's column
  is not a heading, and lines alike but for a number elsewhere stop the
  search (see Review fixes); both give no block.
- **Choosing a block.** The anchor takes the block whose heading carries
  the most of the value's label words that its own line does not print
  (`_block_words`, `:1839`; `_heading_score`, `:1848`, with label numbers
  matched exactly).
- **No block named, or a tie:** the anchor returns every block's copy, so
  the value stays `several`, with the count of blocks.
- `_label_and_number` (`:1767`) steps aside when its candidates are one
  label's lines in repeated blocks, so its six-row context can no longer
  pick the wrong block. Everywhere else it is unchanged; its scoring is
  factored into `_labelled_cells` with the same arithmetic.

**Made-up example.** Two blocks, each headed `Boring B-1` / `Boring B-2`,
each printing `Footing width B 6.0 ft` … `Allowable bearing pressure 4.2 ksf`.

| value | before | after |
|---|---|---|
| "Footing width B (B-2)" 6.0 | placed (B-2) | placed (B-2), anchor `number in its block` |
| "Allowable bearing pressure B-2" 4.2 | **placed in B-1 (wrong)** | placed in B-2 |
| "Footing width B" 6.0 | several (2) | several (2) |
| "Embedment depth Df (Boring B-7)" | several (2) | several (2) |

## Tests (`report_ingest/tests/test_locate.py`)

All pages are drawn with PyMuPDF, using made-up values only.

| test | line | on master |
|---|---|---|
| `test_a_vertical_tables_test_lands_in_its_samples_column` (three samples) | 902 | **fails** |
| `test_a_horizontal_tables_test_stays_on_its_samples_row` (unchanged) | 917 | passes (pins today) |
| `test_a_table_of_unclear_orientation_keeps_todays_place` | 929 | passes (pins today) |
| `test_a_value_in_a_repeated_block_lands_in_the_block_it_names` (two blocks) | 969 | **fails** (wrong block, wrong anchor) |
| `test_a_value_naming_no_block_stays_several` | 983 | passes (pins today) |
| `test_headings_differing_only_by_a_number_give_no_block` (two sheets) | 1024 | passes (pins today) |
| `test_a_line_of_values_above_is_not_a_blocks_heading` | 1049 | passes (pins today) |
| `test_a_heading_echoing_the_values_own_words_does_not_pick_it` | 1061 | **fails** (`label and number`, not the block) |
| `test_a_samples_column_ends_at_its_tables_foot` | 1074 | **fails** (`sample label`, the header) |

The last four rows come from the review (see Review fixes). Both columns
for them were run here: master is the branch's base (25c2525), and the
first build is this branch before the fixes (1973d0e). Every one of them
fails on the first build.

**Runs.**
- `report_ingest/tests/test_locate.py`: 60 passed. On master's `locate.py`,
  2 failed and 58 passed: the two "fails" rows above.
- Gate, `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  **4540 passed, 8 skipped (pydiggs not installed), exit 0**. The graph
  snapshot tests and `test_model_first_schema.py` pass unchanged.
- Consumer `module_work/report_ingest_harness/tests/test_measure_placement.py`:
  11 passed. The harness counts anchors by name, so the two new names
  appear in its tables without any change.

**Prompt versions:** none moved. Nothing a model is sent changes. Page
fingerprints are unchanged.

## Release note

Placement under model-first ingest changes in two cases:
- A laboratory test on a table printed one column per sample now gets the
  box of its own cells in that column, with anchor `sample column`. Before,
  it got the box of its sample label in the header row. Its page and
  confidence do not change, with one exception. Suppose a test is not
  found on its cited page, and on another page of its call its sample
  label was printed in several places. Before, it stayed `not_found`. Now,
  if that page has its sample column, the test moves there (`moved`).
- A calculation value printed with its label in repeated blocks (per
  boring, per load case) now takes the block whose own heading names it,
  with anchor `number in its block`. Before, it was `several` (page only,
  no box), or, where a stray digit tipped the score, a box in the wrong
  block. A value that names no block stays `several`. So do values in
  blocks whose headings differ only by a number (`Load Case 1` /
  `Load Case 2`), as before.

Saved records keep their placements until they are re-placed: a cached item
reuses its saved `placement`. No value, page or confidence moves.

## To measure live

The lead needs the saved MF-3 run folders (the model-first `out_dir` of
each report) and the PDFs (with DI, as the pipeline opens them).

1. **Before.** Read `placement.json` beside `items/` in each report's
   folder. For the `calc` units, count `status == "several"`. For the `lab`
   units, count anchors (`sample label`, `result values`, `boring label`,
   `test title`).
2. **After.** For each saved `items/lab/*.json` and `items/calc/*.json`:
   - Rebuild the objects. For lab, `LabTest.model_validate(t)` for each
     `t` in `blob["tests"]`. For calc,
     `Calculation.model_validate(blob["calculation"])`.
   - Open the report with `Corpus.open_report(id, di="auto")` and take
     `locate.PageWords(doc)`.
   - Call `locate.place_lab_tests(words, tests, pages, write=False, numbers=PageNumbers(tuple(pages)))`
     or `locate.place_calculation(...)` the same way. `pages` are the
     unit's pages: the `page` values in its saved placement, or the group's
     pages from the item key.
   - Count the same things. No model is called.
3. **Expect:**
   - the calc `several` count to fall, from 22 held out in calc_opus;
   - lab tests to move from `sample label` to `sample column` only on
     vertical sheets;
   - no `placed` value to become `several` or `not_found`.
   - List every value whose anchor changed and open five on their pages,
     to confirm the column or block is the right one.
4. `measure_placement.py` re-run on the RI-86 sets gives the same view on
   in-sample and held-out answers. Its "Placed by anchor" line shows the
   two new anchors.

## Review fixes

An independent review of the first build (8a0dfa9, 1973d0e) found three
defects. All three are fixed in `report_ingest/locate.py`, each with a test
from the review's made-up probe pages.

1. **Headings that differ only by a number.** `Load Case 1` / `Load Case 2`
   read as one line once numbers are set aside, so the block search climbed
   past them into the block above. It then took a line of values for a
   heading, and could box a value in the wrong case, or box one that named
   no case. Two checks were added to `block_headings`. A line found as the
   heading that prints a number in the value's column is a line of values,
   so there is no block. Lines alike but for numbers printed outside the
   value's column stop the search, so there is no block either. With no
   block, today's placement follows: `several` for the load cases.
   - The first check also stops a page with no repeated blocks from losing
     a right placement. Example: `Factor of safety 2.0` under "Sliding
     check" and again under "Overturning check" is placed by `label and
     number`, as on master.
   - The second check goes beyond the review's patch. Without it, a block
     whose last line prints no number in the value's column, and repeats a
     word of the label, still put "…, Load Case 1" in Case 2. This is the
     test's second sheet.
2. **The block pick counted the value's own words.** A heading such as
   `Bearing pressure check, Boring B-1` repeats "bearing pressure" from the
   value's own line, so it won for any value. Now only the label's words
   that its own line does not print are scored (`_block_words`). "Allowable
   bearing pressure B-2" lands in B-2. "Allowable bearing pressure" names no
   block and stays `several`.
3. **The sample column ran to the foot of the page.** A table printed one
   row per sample below a vertical table has one label to a row, so the
   column did not end. A number below that equals one of the test's results
   stretched the box over both tables. The column now also ends at the
   first row that prints nothing in it.

The release note above still holds. The fixes narrow where `number in its
block` applies, and keep the sample column inside its table. The note now
also names the one way a lab test's page can change, and says that blocks
headed by a number alone stay as before. The live-measure expectation "no
`placed` value becomes `several`" holds for every probe page.

**Runs after the fixes.**
- `report_ingest/tests/test_locate.py` and `test_model_first_schema.py`:
  96 passed, exit 0.
- `report_ingest/tests/test_graph_model_first.py`: 63 passed, exit 0.
- `webapp/tests/test_public_text.py`: 4 passed, exit 0.
- The review's probe script, on the fixed branch: S1 `several` for all
  three values, S3 placed in both sections, S4 B-2 and `several`, S5 boxed
  at y 108 to 149. S0, S6 and S7 are unchanged from the first build.

## Noted, not fixed

- A decimal in a keyed heading row splits into two tokens in
  `cell_matches` (`23.2` gives `23` and `2`), so a label number can match a
  decimal's fraction digits. In repeated blocks the new anchor now decides
  before this can matter. Elsewhere it is the same defect class: worth an
  item only if a live count shows a misplacement it causes.
- Text values (`_label_and_text`) in repeated blocks still take the best
  line by their label. Blocks are not read for them.
- Blocks whose rows do not line up (one block printing an extra line) are
  read as not repeated, so they keep today's placement.
- Sample labels with no letters (`1`, `2`, `3` across a header) do not
  have a label shape, so their orientation is not read, and they keep
  today's placement.
