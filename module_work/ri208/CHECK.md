# RI-208: the calc reader keeps only the last row of a printed table

Branch `claude/ri-208-calc-table-rows`, cloud helper, 2026-11-06. Commit
38f9fa3 (code and tests) plus this note. Every value in the tests is made up.

## Where the rows were lost

- **The floor reader.** On master, `report_ingest/calc_reader.py:801`
  (`row = body[-1]` in `_values_from_table`, lines 788-814): a table with a
  header row and several filled body rows gave one value per column taken
  from the LAST filled row, named `<heading>, last row of the table`
  (`LAST_ROW_SUFFIX`, line 156). Every other row was dropped. Today's reader
  keeps a floor value the model did not return, so these labels reached the
  saved records: the 2,112 labels of `module_work/dr11/LABELS_SUMMARY.md`.
- **The model-first reader** never sees the floor (`read_calcs` builds with
  no floor), so nothing in its code dropped a row of an answer. Two things
  made rows unsafe there: its instruction never asked for every row of a
  table, and the fold of two answers (`_read_twice`) took two rows as one
  printed value when their labels passed the partial ratio of 80 ("Phi
  (deg), Layer 2" and "Phi (deg), Layer 4" do) on one page with the same
  number, because model-first answers carry no box.
- **The merge** (`_twin`) picked the reader's row for a floor value among
  rows passing the same partial ratio, and among those at the table's box
  took the first with the same value, so with rows present one layer's
  reading could settle another's.
- **The scorer** (`calc_scoring._find`) credited a truth value from any row
  whose label passed the partial ratio and whose value matched: the wrong
  row of a table could answer.

## The fix

- `calc_reader.py:769` `_row_names`: the qualifier each body row is named
  by: its first cell as printed when it has words ("Silty sand", "Case B"),
  a bare number with its column heading ("Layer 2", "Depth (ft) 10.0"),
  `row N` for an empty first cell, and `, row N` added where two rows would
  share one. A one-row table keeps no qualifier, as before.
- `calc_reader.py:829` `_values_from_table`: one value per row and column,
  `<heading>, <qualifier>`; the qualifier's own column is not a value. The
  printed place is the table's box, as today. Rows with one filled cell
  (ruled empty lines) are still left out. `LAST_ROW_SUFFIX` stays defined so
  saved labels still compare as before.
- `calc_scoring.py:281` `other_rows(a, b)`: two labels name different rows
  when both print row numbers and neither set is within the other (bracketed
  units and exponents ignored: "(kN/m^3)" names no row), or both carry a
  qualifier after their last comma and neither qualifier's words are all in
  the other label. A label with no qualifier, or the old ", last row of the
  table", is no row's in particular. `_find` (line 389) uses it, so a truth
  value is credited only from its own row. Program and method matching
  (`same_name` elsewhere) are unchanged.
- `calc_reader.py:1075` `_same_label` = `_same_name` and not `other_rows`;
  used in `_twin` (line 1186), in `merge_calculation`'s split of floor values
  read at their own place and its "added" count (line 1303 on), and in
  `_read_twice` (line 2556). `_twin` now prefers a row whose label names the
  same thing (`_names_same_thing`) before one that only shares the value.
  The merge's rules are as before: the reader's value at its own printed
  line, both where the place is unknown, a floor value the reader did not
  return kept.
- `model_first.py:578` `CALC_TABLE_ROWS`, added to the model-first calc
  instruction only (the plain baseline's text is unchanged): every row's
  values, labelled with the column heading, a comma and the row's own name.

### Before / after (made up)

A printed table `Layer | Thickness (ft) | Unit weight (pcf) | Phi (deg)` with
rows 1-4 (4.0/115/30, 6.5/120/32, 8.0/125/34, 10.0/130/32).

- Before (floor): `Thickness (ft), last row of the table` 10.0, `Unit weight
  (pcf), last row of the table` 130, `Phi (deg), last row of the table` 32:
  three values, rows 1-3 lost.
- After (floor): twelve values, `Thickness (ft), Layer 1` 4.0 ... `Phi (deg),
  Layer 4` 32. Merged with a model-first answer naming the same twelve
  labels and no boxes: twelve rows, twelve reconciled, no disagreement,
  although layers 2 and 4 print the same phi.

## Prompt version

**Moved: `READ_PROMPT_VERSION` mf-9 -> mf-10** (`model_first.py:316`), its
digest recorded in `report_ingest/tests/test_model_first_schema.py`
(`MODEL_FIRST_ASKS["mf-10"]`). The next live run re-reads every saved
model-first answer. Today's reader's prompts (`CALC_READER_SYSTEM`,
`CALC_TOOLS`) and the page fingerprints are unchanged (fingerprints test
passes without a re-record).

## Tests

`report_ingest/tests/test_ri208.py`, 10 tests. Run against master's code
(the test file with the two new helpers stubbed): **9 fail, 1 passes** (the
guard that one row read twice still folds to one, which is meant to pass on
both).

- floor: the 4-row table gives all 12 values, none "last row"; one-row
  tables carry no qualifier; qualifiers from the first cell, distinct;
- merge: a canned model-first answer (no boxes) against the floor keeps all
  12, reconciles 12, no disagreement; a row the model left out is kept from
  the floor;
- model-first fold: two answers whose rows of layers 2 and 4 print one phi
  keep all 4 rows; one row read twice still folds;
- scorer: `other_rows` cases; a truth of four layers credits 3 when layer 4
  is missing although layer 2 prints the same value (master credits 4);
- the model-first calc instruction carries `CALC_TABLE_ROWS`, the plain one
  does not.

Changed: `test_calc_reader.py::test_every_filled_row_of_a_table_is_read`
(was `test_the_last_filled_row_of_a_table_is_read`, which asserted the old
"last row" label); `test_model_first_schema.py::
test_the_model_first_instructions_add_to_the_plain_ones` (calc now adds one
sentence).

Runs (python 3.12 venv per the brief):

- `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py
  webapp/tests/test_page_fingerprints.py`: **4674 passed, 23 skipped
  (pydiggs not installed), 0 failed**.
- `pytest -q -n 4 report_ingest -k calc`: 224 passed before the test
  updates (1 failed: the old last-row test, as expected).
- `pytest -q -n 4 module_work/report_ingest_harness/tests`: **182 passed,
  34 skipped, 0 failed**.

Consumers grepped: `recompute.py`, `cluster_scoring.py`,
`model_first_scoring.py`, `measure_wp5_calc.py`, `dry_run_model_first.py`,
`measure_grouping.py`, `run_plain_baseline.py`; none reads the
"last row" label. `webapp/`, `funhouse_agent/`: only `funhouse_agent/deep/
agent.py` wires `recompute_report_calcs`, whose code is unchanged.

## For the lead

Get the real before/after offline, no model call:

1. Re-score the saved calc runs: `measure_wp5_calc.py` (and
   `score_on_cluster(stages=("calc",), rescore=True)` on the saved `calc/`
   run files) with this branch against master: floor inputs/results found,
   and the counts of floor values, reconciled and disagreements per printout.
   The floor half re-runs over the PDFs; the saved model answers fold and
   merge again under the new label rule.
2. Re-run the DR-11 label measurement locally over re-floored records: the
   count of labels ending "last row of the table" should go to 0 and the
   per-kind label counts should rise by the rows recovered (bearing 31,
   settlement 83, axial 40, slope 52, site response 13, pavement 16 readings
   had them).

## To measure live

- One model-first calc run under mf-10 on a few printouts with layer or case
  tables: does the model give every row, labelled `<heading>, <row>`, and
  does the output stay inside the cap on long tables (a 40-row p-y or
  settlement table is now 40 x columns values)?

## Release note

Calculation tables now keep every row. A printed table inside a calculation
printout (a soil profile, a layer-by-layer settlement, a load schedule)
reaches the record as one value per row and column, labelled by the column
heading and the row ("Unit weight (pcf), Layer 2"), instead of only its last
row labelled "..., last row of the table". Saved calc values gain the other
rows when a report is ingested again; the last row is still there, under its
own row's label. The model-first calc prompt is mf-10, so the next live run
re-reads saved model-first answers. `recompute_report_calcs` sees a
parameter printed once per layer as "printed more than once with different
values" and declines it rather than recomputing from the last row's value.

## Review fixes

An independent review (2026-11-06) asked for four changes. Each is its own
commit with made-up tests in `test_ri208.py`; the prompt text is unchanged,
so the version stays mf-10.

1. **Scorer row match** (`calc_scoring.other_rows`, 98be680 and bd19bc1).
   Labels split only at commas outside brackets and not between digits. A
   number names a row only after a row word (layer, case, scenario, depth,
   row ...), after a setting's "=", or at the end of a row's name. Small
   words, units and a place on the printout ("second sheet") are not words
   of a row's name. Two rows named by ids of different kinds are not told
   apart. The scorecard counts two names as one row's when they share more
   than half the words of the shorter one; the merge keeps the strict rule.
2. **Merge** (`_readings_for`, 14b4180). `FloorValue.row` carries the row a
   table value is one of. Such a value settles only onto a reading whose
   label names that row (`names_row`), never through the table's shared
   box. With no such reading, the row is kept as its own, with no
   disagreement. A detected table has no row boxes, so "the same printed
   line" cannot be told for a grid row.
3. **Row numbers compared as numbers** (98be680), in the scorer, the merge
   and the model-first fold: 5.0 and 5.5 are two rows, as are 0.5 and 5.0,
   while 10 and 10.0 are one.
4. **Numeric first column** (65672ef). A bare-number first cell (thickness,
   depth) is a value again, one per row, labelled "<heading>, row N". A
   column of row numbers ("Layer", "No.", "Case") or a numeric "heading"
   gives none.

Offline re-measurement on the saved records: no model call; the floor was
re-run over the text-layer pages, and the saved answers were re-merged and
re-scored.

| | master | branch before | branch after |
|---|---|---|---|
| Hand values found, saved records (2,622) | 2,238 | 2,153 | 2,237 |
| ... saved reader answer alone (2,459) | 2,103 | 2,019 | 2,102 |
| ... new floor + saved answer merged (2,459) | 2,119 | 2,037 | 2,124 |
| ... the same, under master's scorer | 2,119 | 2,119 | 2,125 |
| ... new floor alone (2,622) | 1,660 | 1,706 | 1,766 |
| Floor values, 792 printouts (table values) | 11,280 (1,828) | 13,309 (3,857) | 13,660 (4,208) |
| Grid tables / rows kept (unique tables) | 190 / 190 | 226 / 755 | 226 / 757 |
| Labels ending "last row of the table" | 805 | 0 | 0 |
| Master's last-row values still a value / only inside a label / gone | | 651 / 149 / 5 | 711 / 87 / 7 |
| Merge: reconciled / disagreements / kept / added | 2,439 / 384 / 2,715 / 1,278 | 2,700 / 512 / 3,995 / 947 | 2,531 / 433 / 4,313 / 865 |
| Table rows settled onto a reading naming something else | | 196 | 27 |
| Row pairs of one column the merge rule cannot tell apart | | 736 of 5,561 | 289 of 6,512 |

- **Scores.** The one credit the scorer still refuses was given from another
  scenario's number. Every other credit master gave is kept, and each now
  goes to the right row's label. The review counted 3 wrong-row credits;
  re-checked, 2 of those had the right row in the record as well, and it is
  credited now.
- **Rows kept.** Rows are counted as the most row names any one column
  carries. The review counted names across all columns (774), which would
  count fix 4's "row N" values twice.
- **Merge.** The 27 rows still settled onto another label name the same row
  in other words. Kept rows rise because a table row is no longer merged
  into a reading that names no row.
- **Master's last-row values.** The 87 left only inside a label are row
  numbers under "Layer" or "Scenario", or junk headings. The 2 more gone
  come from the one printout at the 300-value cap.
- **Row pairs.** The pairs still not told apart are mostly rows named by a
  single letter or code, or by OCR-broken headings.

Tests, run from the worktree, each judged by its exit code: the 12
report_ingest files matching calc, ri208, model_first or recompute, exit 0,
575 passed; `module_work/report_ingest_harness/tests`, exit 0, 201 passed
and 33 skipped; `webapp/tests/test_page_fingerprints.py`, exit 0, 15 passed
(no re-record); `webapp/tests/test_public_text.py`, exit 0, 5 passed.

## What is left / noted, not fixed

- Noted, not fixed: the floor still gives a whole-table box for every row's
  values; planlens cells carry no per-row box here. A per-row box would let
  the merge settle a row at its own line rather than the table's.
- Noted, not fixed: `MAX_FLOOR_KEEP` (300) and `MAX_FLOOR_VALUES` (140, the
  part shown to today's model) are unchanged; a long tabulated printout now
  reaches them sooner, and the cap is warned of as before.
- Noted, not fixed: today's (non-model-first) calc prompt does not ask for
  row-qualified labels; its floor now hands it every row, and rows it reads
  under another label stand beside the floor's.
- `recompute` (DR-11) declines a layered table instead of using one row;
  mapping rows to layers is the later slice the DR-11 summary names.
